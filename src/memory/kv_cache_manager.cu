// KV Cache Manager for GPT-OSS-120B inference engine
// Paged KV cache with slab allocation and free list management.
// Each GPU maintains its own complete KV cache (replicated attention).
//
// Block layout: [num_kv_heads][block_size][head_dim] in BF16
// Block size: 16 tokens * num_kv_heads * 64 head_dim * 2 bytes (per K or V)
// Total per block per layer: 32768 bytes (32KB) for K+V combined
//
// Physical blocks are allocated from a contiguous slab and tracked via a free list.
// Block table maps (seq_id, layer, logical_block) -> physical_block_id.

#include "hip_compat.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mutex>

#include "config.h"
#include "tensor.h"
#include "cuda_utils.h"

namespace gptoss {

// ============================================================================
// HIP kernel: scatter new K/V tokens into paged cache blocks
// ============================================================================

// Each thread handles one element (one head_dim component for one token/head).
// Grid: (num_tokens, num_kv_heads, cdiv(head_dim, blockDim.x))
__global__ void update_kv_cache_kernel(
    __hip_bfloat16* __restrict__ cache,          // [num_physical_blocks, num_kv_heads, block_size, head_dim]
    const __hip_bfloat16* __restrict__ new_data, // [num_tokens, num_kv_heads, head_dim]
    const int32_t* __restrict__ block_table,    // [max_blocks_per_seq] for this (seq, layer)
    int start_pos,
    int num_tokens,
    int num_kv_heads,
    int block_size,
    int head_dim)
{
    int token_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int d         = blockIdx.z * blockDim.x + threadIdx.x;

    if (token_idx >= num_tokens || d >= head_dim) return;

    int abs_pos = start_pos + token_idx;
    int logical_block = abs_pos / block_size;
    int offset_in_block = abs_pos % block_size;

    int physical_block = block_table[logical_block];

    // cache layout: [physical_block][num_kv_heads][block_size][head_dim]
    int64_t cache_idx = (int64_t)physical_block * num_kv_heads * block_size * head_dim
                      + (int64_t)head_idx * block_size * head_dim
                      + (int64_t)offset_in_block * head_dim
                      + d;

    // new_data layout: [num_tokens][num_kv_heads][head_dim]
    int64_t src_idx = (int64_t)token_idx * num_kv_heads * head_dim
                    + (int64_t)head_idx * head_dim
                    + d;

    cache[cache_idx] = new_data[src_idx];
}


// ============================================================================
// KVCacheManager
// ============================================================================

class KVCacheManager {
public:
    KVCacheManager() = default;
    ~KVCacheManager() { destroy(); }

    // Non-copyable, non-movable
    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;

    // -----------------------------------------------------------------------
    // init: allocate all GPU memory and host-side metadata
    // -----------------------------------------------------------------------
    void init(int max_seqs, int max_seq_len, int num_layers, int num_kv_heads,
              int head_dim, int block_size, int device_id,
              int64_t target_total_tokens = 680000)
    {
        std::lock_guard<std::mutex> lock(mu_);

        max_seqs_       = max_seqs;
        max_seq_len_    = max_seq_len;
        num_layers_     = num_layers;
        num_kv_heads_   = num_kv_heads;
        head_dim_       = head_dim;
        block_size_     = block_size;
        device_id_      = device_id;
        initialized_    = true;

        max_blocks_per_seq_ = cdiv(max_seq_len, block_size);

        // Elements per block for one of K or V:
        // [num_kv_heads][block_size][head_dim]
        elements_per_block_ = (int64_t)num_kv_heads * block_size * head_dim;
        bytes_per_block_    = elements_per_block_ * sizeof(__hip_bfloat16); // one of K or V

        // Calculate total physical blocks from the caller-provided token budget.
        // Each physical block covers one (layer, block_slot), and the K+V
        // slabs each contain num_physical_blocks entries.
        //
        // Memory: num_physical_blocks * bytes_per_block * 2 (K + V)
        //       = (target_tokens / block_size) * num_layers * bytes_per_block * 2
        int64_t blocks_per_layer = cdiv(static_cast<int>(target_total_tokens), block_size);
        num_physical_blocks_ = blocks_per_layer * num_layers;

        // Check: theoretically max_seqs * max_blocks_per_seq * num_layers
        // blocks needed if every seq uses full context. In practice that's too
        // large (e.g. 256 * 8192 * 36 = 75M blocks).  Just warn and proceed
        // with whatever fits in memory -- the runtime will reject requests that
        // exceed the actual allocation.
        int64_t min_required = (int64_t)max_seqs * max_blocks_per_seq_ * num_layers;
        if (num_physical_blocks_ < min_required) {
            fprintf(stderr, "[KVCacheManager] Note: target blocks (%lld) < theoretical max (%lld). "
                    "Long sequences may be rejected at runtime.\n",
                    (long long)num_physical_blocks_, (long long)min_required);
        }

        CUDA_CHECK(hipSetDevice(device_id_));

        // Allocate K and V cache slabs on GPU
        size_t slab_bytes = (size_t)num_physical_blocks_ * bytes_per_block_;
        CUDA_CHECK(hipMalloc(&k_cache_, slab_bytes));
        CUDA_CHECK(hipMalloc(&v_cache_, slab_bytes));
        CUDA_CHECK(hipMemset(k_cache_, 0, slab_bytes));
        CUDA_CHECK(hipMemset(v_cache_, 0, slab_bytes));

        // Block table on GPU: [max_seqs][num_layers][max_blocks_per_seq]
        // This is the lookup table that attention kernels read from.
        size_t bt_elements = (size_t)max_seqs * num_layers * max_blocks_per_seq_;
        size_t bt_bytes    = bt_elements * sizeof(int32_t);
        CUDA_CHECK(hipMalloc(&d_block_table_, bt_bytes));
        CUDA_CHECK(hipMemset(d_block_table_, 0xFF, bt_bytes)); // -1 = invalid

        // Host-side mirror of block table (for allocation logic)
        h_block_table_.resize(bt_elements, -1);

        // Per-sequence metadata (host side)
        seq_len_.resize(max_seqs, 0);
        seq_allocated_blocks_.resize(max_seqs);
        for (int s = 0; s < max_seqs; ++s) {
            seq_allocated_blocks_[s].resize(num_layers, 0);
        }
        seq_active_.resize(max_seqs, false);

        // Initialize free list: all physical blocks are free
        free_list_.reserve(num_physical_blocks_);
        for (int64_t i = num_physical_blocks_ - 1; i >= 0; --i) {
            free_list_.push_back(static_cast<int32_t>(i));
        }

        fprintf(stderr, "[KVCacheManager] GPU %d: allocated %lld physical blocks "
                "(%.2f GB K + %.2f GB V), max %d seqs, max %d tokens/seq\n",
                device_id_, (long long)num_physical_blocks_,
                slab_bytes / (1024.0 * 1024.0 * 1024.0),
                slab_bytes / (1024.0 * 1024.0 * 1024.0),
                max_seqs, max_seq_len);
    }

    // -----------------------------------------------------------------------
    // allocate_blocks: ensure enough physical blocks for num_new_tokens
    //   for all layers of a given sequence
    // Returns true on success, false if out of blocks.
    // -----------------------------------------------------------------------
    bool allocate_blocks(int seq_id, int num_new_tokens)
    {
        std::lock_guard<std::mutex> lock(mu_);
        assert(initialized_ && seq_id >= 0 && seq_id < max_seqs_);

        if (!seq_active_[seq_id]) {
            // First allocation for this sequence -- reset metadata
            seq_len_[seq_id] = 0;
            for (int l = 0; l < num_layers_; ++l) {
                seq_allocated_blocks_[seq_id][l] = 0;
            }
            seq_active_[seq_id] = true;
        }

        int new_total_tokens = seq_len_[seq_id] + num_new_tokens;
        int blocks_needed_total = cdiv(new_total_tokens, block_size_);

        // For each layer, allocate any additional blocks needed
        for (int layer = 0; layer < num_layers_; ++layer) {
            int already_allocated = seq_allocated_blocks_[seq_id][layer];
            int need = blocks_needed_total - already_allocated;
            if (need <= 0) continue;

            if ((int64_t)need > (int64_t)free_list_.size()) {
                fprintf(stderr, "[KVCacheManager] OOM: seq %d layer %d needs %d blocks, "
                        "only %zu free\n", seq_id, layer, need, free_list_.size());
                return false;
            }

            int bt_base = block_table_index(seq_id, layer, 0);
            for (int b = 0; b < need; ++b) {
                int32_t phys = free_list_.back();
                free_list_.pop_back();
                int logical_slot = already_allocated + b;
                h_block_table_[bt_base + logical_slot] = phys;
            }
            seq_allocated_blocks_[seq_id][layer] = blocks_needed_total;
        }

        // Update sequence length
        seq_len_[seq_id] = new_total_tokens;

        // Sync block table to GPU (only the region for this sequence)
        sync_block_table_for_seq(seq_id);

        return true;
    }

    // -----------------------------------------------------------------------
    // get_block_table: return GPU pointer to block table for a sequence
    // Layout on GPU: [max_seqs][num_layers][max_blocks_per_seq]
    // Returns pointer to [num_layers][max_blocks_per_seq] for given seq.
    // -----------------------------------------------------------------------
    const int32_t* get_block_table(int seq_id) const
    {
        assert(initialized_ && seq_id >= 0 && seq_id < max_seqs_);
        return d_block_table_ + (int64_t)seq_id * num_layers_ * max_blocks_per_seq_;
    }

    // -----------------------------------------------------------------------
    // get_block_table_for_layer: convenience to get table for (seq, layer)
    // Returns GPU pointer to max_blocks_per_seq int32s.
    // -----------------------------------------------------------------------
    const int32_t* get_block_table_for_layer(int seq_id, int layer) const
    {
        assert(initialized_ && seq_id >= 0 && seq_id < max_seqs_);
        assert(layer >= 0 && layer < num_layers_);
        int offset = block_table_index(seq_id, layer, 0);
        return d_block_table_ + offset;
    }

    // -----------------------------------------------------------------------
    // get_k_cache / get_v_cache: base pointers into cache slabs
    // -----------------------------------------------------------------------
    __hip_bfloat16* get_k_cache() const { return k_cache_; }
    __hip_bfloat16* get_v_cache() const { return v_cache_; }

    // -----------------------------------------------------------------------
    // update_cache: write new K/V data into the paged cache
    //
    //   new_k, new_v: [num_tokens, num_kv_heads, head_dim] in BF16, on GPU
    //   start_pos:    absolute token position in the sequence
    //   num_tokens:   number of new tokens to write
    // -----------------------------------------------------------------------
    void update_cache(int seq_id, int layer,
                      const __hip_bfloat16* new_k, const __hip_bfloat16* new_v,
                      int start_pos, int num_tokens, hipStream_t stream)
    {
        assert(initialized_ && seq_id >= 0 && seq_id < max_seqs_);
        assert(layer >= 0 && layer < num_layers_);
        assert(start_pos >= 0 && start_pos + num_tokens <= seq_len_[seq_id]);

        if (num_tokens == 0) return;

        // Pointer to block table for this (seq, layer) on GPU
        const int32_t* bt_ptr = get_block_table_for_layer(seq_id, layer);

        // Launch kernel dimensions
        int threads_per_block = 64; // head_dim is 64, one thread per element
        dim3 grid(num_tokens, num_kv_heads_, cdiv(head_dim_, threads_per_block));
        dim3 block(threads_per_block);

        // Update K cache
        update_kv_cache_kernel<<<grid, block, 0, stream>>>(
            k_cache_, new_k, bt_ptr,
            start_pos, num_tokens,
            num_kv_heads_, block_size_, head_dim_);

        // Update V cache
        update_kv_cache_kernel<<<grid, block, 0, stream>>>(
            v_cache_, new_v, bt_ptr,
            start_pos, num_tokens,
            num_kv_heads_, block_size_, head_dim_);
    }

    // -----------------------------------------------------------------------
    // free_sequence: return all physical blocks for a sequence to the free list
    // -----------------------------------------------------------------------
    void free_sequence(int seq_id)
    {
        std::lock_guard<std::mutex> lock(mu_);
        assert(initialized_ && seq_id >= 0 && seq_id < max_seqs_);

        if (!seq_active_[seq_id]) return;

        for (int layer = 0; layer < num_layers_; ++layer) {
            int num_blocks = seq_allocated_blocks_[seq_id][layer];
            int bt_base = block_table_index(seq_id, layer, 0);
            for (int b = 0; b < num_blocks; ++b) {
                int32_t phys = h_block_table_[bt_base + b];
                if (phys >= 0) {
                    free_list_.push_back(phys);
                    h_block_table_[bt_base + b] = -1;
                }
            }
            seq_allocated_blocks_[seq_id][layer] = 0;
        }

        seq_len_[seq_id] = 0;
        seq_active_[seq_id] = false;

        // Sync invalidated block table to GPU
        sync_block_table_for_seq(seq_id);
    }

    // -----------------------------------------------------------------------
    // get_seq_len
    // -----------------------------------------------------------------------
    int get_seq_len(int seq_id) const
    {
        assert(initialized_ && seq_id >= 0 && seq_id < max_seqs_);
        return seq_len_[seq_id];
    }

    // -----------------------------------------------------------------------
    // Accessors for cache geometry
    // -----------------------------------------------------------------------
    int get_num_physical_blocks() const { return static_cast<int>(num_physical_blocks_); }
    int get_num_free_blocks() const { return static_cast<int>(free_list_.size()); }
    int get_max_blocks_per_seq() const { return max_blocks_per_seq_; }
    int get_block_size() const { return block_size_; }
    int get_device_id() const { return device_id_; }

    // -----------------------------------------------------------------------
    // destroy: free all GPU memory
    // -----------------------------------------------------------------------
    void destroy()
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return;

        CUDA_CHECK(hipSetDevice(device_id_));

        if (k_cache_) { CUDA_CHECK(hipFree(k_cache_)); k_cache_ = nullptr; }
        if (v_cache_) { CUDA_CHECK(hipFree(v_cache_)); v_cache_ = nullptr; }
        if (d_block_table_) { CUDA_CHECK(hipFree(d_block_table_)); d_block_table_ = nullptr; }

        h_block_table_.clear();
        free_list_.clear();
        seq_len_.clear();
        seq_allocated_blocks_.clear();
        seq_active_.clear();

        initialized_ = false;
    }

private:
    // Compute flat index into block table array
    inline int block_table_index(int seq_id, int layer, int logical_block) const
    {
        return ((int64_t)seq_id * num_layers_ + layer) * max_blocks_per_seq_ + logical_block;
    }

    // Sync block table for one sequence from host to GPU
    void sync_block_table_for_seq(int seq_id)
    {
        CUDA_CHECK(hipSetDevice(device_id_));
        size_t offset = (size_t)seq_id * num_layers_ * max_blocks_per_seq_;
        size_t count  = (size_t)num_layers_ * max_blocks_per_seq_;
        CUDA_CHECK(hipMemcpy(
            d_block_table_ + offset,
            h_block_table_.data() + offset,
            count * sizeof(int32_t),
            hipMemcpyHostToDevice));
    }

    // Configuration
    int max_seqs_           = 0;
    int max_seq_len_        = 0;
    int num_layers_         = 0;
    int num_kv_heads_       = 0;
    int head_dim_           = 0;
    int block_size_         = 0;
    int max_blocks_per_seq_ = 0;
    int device_id_          = 0;
    bool initialized_       = false;

    // Physical block geometry
    int64_t num_physical_blocks_ = 0;
    int64_t elements_per_block_  = 0;
    size_t  bytes_per_block_     = 0;

    // GPU memory
    __hip_bfloat16* k_cache_     = nullptr; // [num_physical_blocks][num_kv_heads][block_size][head_dim]
    __hip_bfloat16* v_cache_     = nullptr; // same layout
    int32_t*       d_block_table_ = nullptr; // [max_seqs][num_layers][max_blocks_per_seq]

    // Host-side metadata
    std::vector<int32_t> h_block_table_;              // mirror of d_block_table_
    std::vector<int32_t> free_list_;                   // stack of free physical block IDs
    std::vector<int>     seq_len_;                     // current token count per sequence
    std::vector<std::vector<int>> seq_allocated_blocks_; // [seq][layer] -> num blocks allocated
    std::vector<bool>    seq_active_;                  // whether sequence slot is in use

    std::mutex mu_; // protects allocation/free operations
};

// ---------------------------------------------------------------------------
// Extern wrapper functions
// ---------------------------------------------------------------------------

KVCacheManager* kv_cache_manager_create() {
    return new KVCacheManager();
}

void kv_cache_manager_destroy(KVCacheManager* mgr) {
    if (mgr) delete mgr;
}

void kv_cache_manager_init(KVCacheManager* mgr, int max_seqs, int max_seq_len,
                            int num_layers, int num_kv_heads, int head_dim,
                            int block_size, int device_id,
                            int64_t target_total_tokens) {
    mgr->init(max_seqs, max_seq_len, num_layers, num_kv_heads, head_dim, block_size, device_id,
              target_total_tokens);
}

void kv_cache_manager_free_sequence(KVCacheManager* mgr, int seq_id) {
    mgr->free_sequence(seq_id);
}

__hip_bfloat16* kv_cache_manager_k_cache(KVCacheManager* mgr) {
    return mgr->get_k_cache();
}

__hip_bfloat16* kv_cache_manager_v_cache(KVCacheManager* mgr) {
    return mgr->get_v_cache();
}

int kv_cache_manager_num_blocks(KVCacheManager* mgr) {
    return mgr->get_num_physical_blocks();
}

} // namespace gptoss
