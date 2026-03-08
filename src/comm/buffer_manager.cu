// Double-Buffered Staging for Communication Overlap in GPT-OSS-120B
//
// Pre-allocates worst-case send/recv buffers for EP-N grouped P2P expert
// output exchange. Double buffering allows layer N+1's dispatch to begin
// filling its send buffer while layer N's combine is still in flight on
// the comm stream. The two buffer sets are indexed by (layer_idx % 2).
//
// Memory budget:
//   Per buffer pair (send + recv):
//     max_batch_tokens * num_active_experts * hidden_size * sizeof(bf16)
//   = 2048 * 4 * 2880 * 2 = ~45 MB (for a typical max batch of 2048)
//   Two pairs (double buffer): ~90 MB
//   Total per GPU: ~90 MB  (negligible vs 192 GB HBM on MI300X)
//
// In the worst case, ALL tokens in the batch could be routed to experts
// on remote GPUs, each token visiting num_active_experts experts.
// The sizing covers the total remote token count regardless of ep_size.
// We allocate for worst-case to avoid dynamic allocation during inference.

#include "hip_compat.h"
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cassert>

#include "config.h"
#include "cuda_utils.h"

namespace gptoss {

class BufferManager {
public:
    BufferManager() = default;
    ~BufferManager() { destroy(); }

    // Non-copyable
    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;

    // -------------------------------------------------------------------
    // Initialize double-buffered staging areas.
    //
    // Allocates 2 buffer pairs (send + recv) on the specified GPU.
    // Each buffer is sized for the worst-case scenario where every token
    // in the batch is routed to all active experts on the remote GPU.
    //
    // @param max_batch_tokens  Maximum number of tokens per batch
    // @param hidden_size       Hidden dimension per token (2880)
    // @param device_id         HIP device to allocate on
    // -------------------------------------------------------------------
    void init(int max_batch_tokens, int hidden_size, int device_id) {
        assert(!initialized_ && "BufferManager already initialized");
        assert(max_batch_tokens > 0);
        assert(hidden_size > 0);

        device_id_        = device_id;
        max_batch_tokens_ = max_batch_tokens;
        hidden_size_      = hidden_size;

        CUDA_CHECK(hipSetDevice(device_id_));

        // Worst-case tokens per direction:
        // Each token can be routed to num_active_experts experts. In EP-N,
        // up to all of those experts may reside on remote GPUs.
        // So the maximum number of token-expert pairs destined for one
        // direction is: max_batch_tokens * num_active_experts.
        //
        // Each token-expert pair carries `hidden_size` BF16 elements.
        const size_t max_token_expert_pairs =
            static_cast<size_t>(max_batch_tokens) *
            ModelConfig::num_active_experts;

        buffer_elements_ = max_token_expert_pairs * hidden_size;
        buffer_bytes_    = buffer_elements_ * sizeof(__hip_bfloat16);

        // Allocate 4 buffers total: 2 pairs x (send + recv)
        for (int i = 0; i < NUM_BUFFERS; ++i) {
            CUDA_CHECK(hipMalloc(&buffers_[i].send, buffer_bytes_));
            CUDA_CHECK(hipMalloc(&buffers_[i].recv, buffer_bytes_));

            // Zero-initialize to avoid uninitialized memory issues in
            // debug/validation scenarios.
            CUDA_CHECK(hipMemset(buffers_[i].send, 0, buffer_bytes_));
            CUDA_CHECK(hipMemset(buffers_[i].recv, 0, buffer_bytes_));
        }

        initialized_ = true;

        const double total_mb = (4.0 * buffer_bytes_) / (1024.0 * 1024.0);
        fprintf(stderr,
                "[BufferManager] GPU %d: allocated %.1f MB "
                "(%d buffer pairs, %zu bytes each, "
                "max_batch=%d, hidden=%d, active_experts=%d)\n",
                device_id_, total_mb, NUM_BUFFERS, buffer_bytes_,
                max_batch_tokens_, hidden_size_,
                ModelConfig::num_active_experts);
    }

    // -------------------------------------------------------------------
    // Get the send buffer for a given MoE layer.
    //
    // Alternates between buffer[0] and buffer[1] based on layer index,
    // enabling pipelined execution where layer N's combine and layer
    // N+1's dispatch can overlap.
    //
    // @param layer_idx  MoE layer index (0-based)
    // @return           Device pointer to the send staging buffer (BF16)
    // -------------------------------------------------------------------
    void* get_send_buffer(int layer_idx) {
        assert(initialized_ && "BufferManager not initialized");
        return buffers_[layer_idx % NUM_BUFFERS].send;
    }

    const void* get_send_buffer(int layer_idx) const {
        assert(initialized_ && "BufferManager not initialized");
        return buffers_[layer_idx % NUM_BUFFERS].send;
    }

    // -------------------------------------------------------------------
    // Get the recv buffer for a given MoE layer.
    //
    // @param layer_idx  MoE layer index (0-based)
    // @return           Device pointer to the recv staging buffer (BF16)
    // -------------------------------------------------------------------
    void* get_recv_buffer(int layer_idx) {
        assert(initialized_ && "BufferManager not initialized");
        return buffers_[layer_idx % NUM_BUFFERS].recv;
    }

    const void* get_recv_buffer(int layer_idx) const {
        assert(initialized_ && "BufferManager not initialized");
        return buffers_[layer_idx % NUM_BUFFERS].recv;
    }

    // -------------------------------------------------------------------
    // Get both buffers for a layer as a pair (convenience).
    //
    // @param layer_idx  MoE layer index
    // @param send_out   Output: device pointer to send buffer
    // @param recv_out   Output: device pointer to recv buffer
    // -------------------------------------------------------------------
    void get_buffers(int layer_idx, void** send_out, void** recv_out) {
        assert(initialized_ && "BufferManager not initialized");
        int idx = layer_idx % NUM_BUFFERS;
        *send_out = buffers_[idx].send;
        *recv_out = buffers_[idx].recv;
    }

    // -------------------------------------------------------------------
    // Tear down all resources.
    // Safe to call multiple times or on an uninitialized manager.
    // -------------------------------------------------------------------
    void destroy() {
        if (!initialized_) return;

        CUDA_CHECK(hipSetDevice(device_id_));

        for (int i = 0; i < NUM_BUFFERS; ++i) {
            if (buffers_[i].send) {
                CUDA_CHECK(hipFree(buffers_[i].send));
                buffers_[i].send = nullptr;
            }
            if (buffers_[i].recv) {
                CUDA_CHECK(hipFree(buffers_[i].recv));
                buffers_[i].recv = nullptr;
            }
        }

        fprintf(stderr, "[BufferManager] GPU %d: freed staging buffers\n",
                device_id_);
        initialized_ = false;
    }

    // ---- Accessors ----
    size_t buffer_bytes()            const { return buffer_bytes_; }
    size_t buffer_elements()         const { return buffer_elements_; }
    int    max_batch_tokens()        const { return max_batch_tokens_; }
    int    hidden_size()             const { return hidden_size_; }
    int    device_id()               const { return device_id_; }
    bool   is_initialized()         const { return initialized_; }

    // Total memory consumed by all buffers (for diagnostics)
    size_t total_memory_bytes() const {
        return NUM_BUFFERS * 2 * buffer_bytes_;  // 2 directions per pair
    }

    // Maximum number of token-expert pairs that fit in one buffer
    size_t max_token_expert_pairs() const {
        return buffer_elements_ / hidden_size_;
    }

private:
    // A buffer pair for one direction of All-to-All communication
    struct BufferPair {
        void* send = nullptr;  // Tokens being sent to the peer GPU
        void* recv = nullptr;  // Tokens received from the peer GPU
    };

    static constexpr int NUM_BUFFERS = 2;  // Double buffering

    BufferPair buffers_[NUM_BUFFERS] = {};

    int    device_id_        = -1;
    int    max_batch_tokens_ = 0;
    int    hidden_size_      = 0;
    size_t buffer_bytes_     = 0;
    size_t buffer_elements_  = 0;
    bool   initialized_      = false;
};

// ---------------------------------------------------------------------------
// Extern wrapper functions
// ---------------------------------------------------------------------------

BufferManager* buffer_manager_create() {
    return new BufferManager();
}

void buffer_manager_destroy(BufferManager* mgr) {
    if (mgr) delete mgr;
}

void buffer_manager_init(BufferManager* mgr, int max_batch_tokens, int hidden_size, int device_id) {
    mgr->init(max_batch_tokens, hidden_size, device_id);
}

} // namespace gptoss
