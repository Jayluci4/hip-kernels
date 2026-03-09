// Full Transformer Forward Pass for GPT-OSS-120B Inference Engine
// Orchestrates 36 layers of attention + MoE through prefill and decode paths.
// Multi-GPU support via TP2+EP2 (Tensor Parallel attention + Expert Parallel MoE).
//
// Architecture (per GPU with TP2):
//   Embedding [201088, 2880]                        (replicated)
//   36 x (AttentionLayer[TP2] -> MoELayer[EP2])
//   Final RMSNorm [2880]                            (replicated)
//   LM Head [2880, tp_vocab_size=100544]            (column-sharded, all-gather)
//
// TP2 attention: each GPU handles 32/64 Q heads, 4/8 KV heads.
//   After O projection, RCCL all-reduce produces identical hidden states.
//   This feeds MoE with deterministic routing (EP2 preserved).
// Sharded LM head: each GPU computes logits for half the vocabulary.
//   RCCL all-gather assembles full vocab logits for sampling.

#include "hip_compat.h"
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <rccl/rccl.h>

#include "config.h"
#include "tensor.h"
#include "cuda_utils.h"
#include "nccl_utils.h"
#include "profiler.h"

namespace gptoss {

// ---------------------------------------------------------------------------
// Extern kernel declarations
// ---------------------------------------------------------------------------

extern void rmsnorm_forward(
    const __hip_bfloat16* input,
    const __hip_bfloat16* weight,
    __hip_bfloat16* output,
    int num_tokens,
    int hidden_size,
    float eps,
    hipStream_t stream);

extern void fused_residual_rmsnorm_forward(
    const __hip_bfloat16* residual,
    const __hip_bfloat16* delta,
    const __hip_bfloat16* weight,
    __hip_bfloat16* normed_output,
    __hip_bfloat16* sum_output,
    int num_tokens,
    int hidden_size,
    float eps,
    hipStream_t stream);

// ---------------------------------------------------------------------------
// Forward-declare layer classes and extern wrapper functions.
//
// AttentionLayer and MoELayer are defined in attention.cu and moe_layer.cu
// respectively. Since they are separate compilation units, the Transformer
// calls into them through extern "C"-style wrapper functions defined in
// those files. This avoids needing to see the full class definitions here.
// ---------------------------------------------------------------------------

class AttentionLayer;
class MoELayer;

// Wrapper functions defined at the bottom of attention.cu and moe_layer.cu
extern void attention_layer_prefill(
    AttentionLayer* layer,
    const __hip_bfloat16* input,
    const int* positions,
    const int32_t* slot_mapping,
    __hip_bfloat16* output,
    int num_tokens,
    hipStream_t stream);

extern void attention_layer_decode(
    AttentionLayer* layer,
    const __hip_bfloat16* input,
    const int* positions,
    const int32_t* slot_mapping,
    const int32_t* block_table,
    const int32_t* seq_lens,
    int max_num_blocks,
    __hip_bfloat16* output,
    int num_seqs,
    hipStream_t stream);

extern void attention_layer_decode_no_residual(
    AttentionLayer* layer,
    const __hip_bfloat16* input,
    const int* positions,
    const int32_t* slot_mapping,
    const int32_t* block_table,
    const int32_t* seq_lens,
    int max_num_blocks,
    int num_seqs,
    hipStream_t stream);

extern __hip_bfloat16* attention_layer_get_proj_out(AttentionLayer* layer);

extern void moe_layer_forward(
    MoELayer* layer,
    const __hip_bfloat16* input,
    __hip_bfloat16* output,
    int num_tokens,
    hipStream_t stream);

extern void moe_layer_forward_prenormed(
    MoELayer* layer,
    const __hip_bfloat16* normed_input,
    const __hip_bfloat16* residual_input,
    __hip_bfloat16* output,
    int num_tokens,
    hipStream_t stream);

extern const __hip_bfloat16* moe_layer_get_ffn_norm_weight(MoELayer* layer);

extern void gemv_bf16_forward(
    const __hip_bfloat16* input,
    const __hip_bfloat16* weight,
    __hip_bfloat16* output,
    int N, int K,
    hipStream_t stream,
    const __hip_bfloat16* bias = nullptr);

// ---------------------------------------------------------------------------
// Embedding lookup kernel: token_ids -> embedding vectors
// Each block handles one token, threads iterate over hidden_size.
// ---------------------------------------------------------------------------

__global__ void embedding_lookup_kernel(
    const int* __restrict__ token_ids,
    const __hip_bfloat16* __restrict__ embedding_table,  // [vocab_size, hidden_size]
    __hip_bfloat16* __restrict__ output,                 // [num_tokens, hidden_size]
    int hidden_size)
{
    int token_idx = blockIdx.x;
    int token_id = token_ids[token_idx];

    const __hip_bfloat16* src = embedding_table +
        static_cast<int64_t>(token_id) * hidden_size;
    __hip_bfloat16* dst = output +
        static_cast<int64_t>(token_idx) * hidden_size;

    // Vectorized copy: process 2 BF16 elements per thread per iteration
    int half_hidden = hidden_size / 2;
    const __hip_bfloat162* src_vec = reinterpret_cast<const __hip_bfloat162*>(src);
    __hip_bfloat162* dst_vec = reinterpret_cast<__hip_bfloat162*>(dst);

    for (int i = threadIdx.x; i < half_hidden; i += blockDim.x) {
        dst_vec[i] = src_vec[i];
    }
    // Handle odd trailing element
    if ((hidden_size & 1) && threadIdx.x == 0) {
        dst[hidden_size - 1] = src[hidden_size - 1];
    }
}

static void embedding_lookup(
    const int* token_ids,
    const __hip_bfloat16* embedding_table,
    __hip_bfloat16* output,
    int num_tokens,
    int hidden_size,
    hipStream_t stream)
{
    if (num_tokens == 0) return;
    int threads = 256;
    embedding_lookup_kernel<<<num_tokens, threads, 0, stream>>>(
        token_ids, embedding_table, output, hidden_size);
    CUDA_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------------------
// Extract last-token hidden state kernel
// Copies the hidden state of the last token to a separate buffer.
// Used after prefill to feed only the last token's state into LM head.
// ---------------------------------------------------------------------------

__global__ void extract_last_token_kernel(
    const __hip_bfloat16* __restrict__ hidden_states,  // [num_tokens, hidden_size]
    __hip_bfloat16* __restrict__ output,               // [1, hidden_size]
    int last_token_idx,
    int hidden_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = tid * 2;

    const __hip_bfloat16* src = hidden_states +
        static_cast<int64_t>(last_token_idx) * hidden_size;

    if (vec_idx + 1 < hidden_size) {
        *reinterpret_cast<__hip_bfloat162*>(output + vec_idx) =
            *reinterpret_cast<const __hip_bfloat162*>(src + vec_idx);
    } else if (vec_idx < hidden_size) {
        output[vec_idx] = src[vec_idx];
    }
}

static void extract_last_token(
    const __hip_bfloat16* hidden_states,
    __hip_bfloat16* output,
    int last_token_idx,
    int hidden_size,
    hipStream_t stream)
{
    int threads = 256;
    int blocks = cdiv(cdiv(hidden_size, 2), threads);
    extract_last_token_kernel<<<blocks, threads, 0, stream>>>(
        hidden_states, output, last_token_idx, hidden_size);
    CUDA_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------------------
// Transformer
// ---------------------------------------------------------------------------

class Transformer {
public:
    // ---- Configuration ----
    int rank_;           // GPU rank in the EP2 group (0 or 1)
    int device_id_;

    // ---- Model weights (BF16, device memory, not owned) ----
    const __hip_bfloat16* embedding_table_;   // [vocab_size, hidden_size] = [201088, 2880] (replicated)
    const __hip_bfloat16* final_norm_weight_; // [hidden_size] = [2880] (replicated)
    const __hip_bfloat16* lm_head_weight_;    // [hidden_size, tp_vocab_size] = [2880, 100544] (sharded)

    // ---- Layer objects (not owned, caller provides array of pointers) ----
    // Stored as pointer-to-pointer so array indexing only requires
    // sizeof(pointer), avoiding the need for complete type definitions.
    AttentionLayer** attention_layers_;
    MoELayer** moe_layers_;

    // ---- hipBLAS handle for LM head GEMM ----
    CublasHandle* cublas_handle_;

    // ---- RCCL communicators ----
    ncclComm_t nccl_comm_;       // TP communicator (for LM head AllGather)
    ncclComm_t ep_nccl_comm_;    // EP communicator (for hidden-state sync)

    // ---- Scratch buffers (device memory, owned) ----
    // These are allocated once and shared across layers to minimize memory usage.
    // Two residual buffers for ping-pong between layers.
    __hip_bfloat16* residual_buf_[2];     // [max_tokens, hidden_size] x 2
    __hip_bfloat16* norm_out_buf_;        // [max_tokens, hidden_size]
    __hip_bfloat16* logits_buf_;          // [max_tokens, vocab_size] - full vocab (after all-gather)
    __hip_bfloat16* local_logits_buf_;    // [max_tokens, tp_vocab_size] - partial logits (this GPU)
    __hip_bfloat16* last_hidden_buf_;     // [1, hidden_size] - for extracting last token state
    int*           token_ids_device_;    // [max_tokens] - token ids on device

    // Pre-allocated decode metadata buffers (avoid hipMalloc in hot path)
    int*           d_positions_;         // [max_tokens]
    int32_t*       d_slot_mapping_;      // [max_tokens]
    int32_t*       d_seq_lens_;          // [max_tokens]
    int32_t*       d_block_table_;       // [block_table_capacity_]
    int            block_table_capacity_; // max int32 entries for block table

    // ---- Streams ----
    CudaStream compute_stream_;
    CudaStream comm_stream_;

    // ---- Capacity ----
    int max_tokens_;                     // Maximum tokens the scratch buffers can hold
    bool initialized_;
    bool skip_moe_ = false;  // Set via env GPTOSS_SKIP_MOE=1 for models with zero-delta MoE

    // ---- HIP Graph cache for B=1 decode ----
    // Caches captured graphs keyed by max_num_blocks. max_num_blocks changes every
    // 16 tokens (as new KV cache blocks are allocated), so within one generate() call,
    // each graph is used for 16 consecutive tokens. On subsequent generate() calls
    // at overlapping sequence lengths, cached graphs provide instant replay.
    //
    // State machine per max_num_blocks:
    //   First call:  eager warmup (populates internal caches)
    //   Second call: capture + launch
    //   Subsequent:  graph replay
    //
    // Using unordered_map for unbounded caching (max_num_blocks can be up to 8192).
    std::unordered_map<int, hipGraphExec_t> decode_graph_map_;
    std::unordered_map<int, int> decode_graph_warmup_;  // warmup count per max_num_blocks

    // Constants
    static constexpr int num_layers = ModelConfig::num_layers;          // 36
    static constexpr int hidden_size = ModelConfig::hidden_size;        // 2880
    static constexpr int vocab_size = ModelConfig::vocab_size;          // 201088
    static constexpr int tp_vocab_size = ModelConfig::tp_vocab_size;    // 100544 (per GPU)
    static constexpr float rms_norm_eps = ModelConfig::rms_norm_eps;

    Transformer()
        : rank_(0)
        , device_id_(0)
        , embedding_table_(nullptr)
        , final_norm_weight_(nullptr)
        , lm_head_weight_(nullptr)
        , attention_layers_(nullptr)
        , moe_layers_(nullptr)
        , cublas_handle_(nullptr)
        , nccl_comm_(nullptr)
        , ep_nccl_comm_(nullptr)
        , norm_out_buf_(nullptr)
        , logits_buf_(nullptr)
        , local_logits_buf_(nullptr)
        , last_hidden_buf_(nullptr)
        , token_ids_device_(nullptr)
        , d_positions_(nullptr)
        , d_slot_mapping_(nullptr)
        , d_seq_lens_(nullptr)
        , d_block_table_(nullptr)
        , block_table_capacity_(0)
        , max_tokens_(0)
        , initialized_(false)
    {
        residual_buf_[0] = nullptr;
        residual_buf_[1] = nullptr;
    }

    ~Transformer() {
        for (auto& [k, exec] : decode_graph_map_) {
            if (exec) hipGraphExecDestroy(exec);
        }
        decode_graph_map_.clear();
        decode_graph_warmup_.clear();
        free_buffers();
    }

    // -----------------------------------------------------------------------
    // Initialize the transformer for multi-GPU inference.
    //
    // rank:           GPU rank in the EP2 group (0 or 1)
    // nccl_comm:      RCCL communicator for inter-GPU communication
    // cublas_handle:  hipBLAS handle for GEMM operations
    // embedding_table: [vocab_size, hidden_size] BF16
    // final_norm_weight: [hidden_size] BF16
    // lm_head_weight: [hidden_size, vocab_size] BF16
    // attn_layers:    Pre-initialized array of num_layers AttentionLayer objects
    // moe_layers:     Pre-initialized array of num_layers MoELayer objects
    // max_tokens:     Maximum number of tokens to support (for buffer allocation)
    // -----------------------------------------------------------------------
    void init(
        int rank,
        ncclComm_t nccl_comm,
        ncclComm_t ep_nccl_comm,
        CublasHandle* cublas_handle,
        const __hip_bfloat16* embedding_table,
        const __hip_bfloat16* final_norm_weight,
        const __hip_bfloat16* lm_head_weight,
        AttentionLayer** attn_layers,
        MoELayer** moe_layers,
        int max_tokens)
    {
        rank_ = rank;
        device_id_ = rank;  // In EP2, rank maps directly to device
        nccl_comm_ = nccl_comm;
        ep_nccl_comm_ = ep_nccl_comm;
        cublas_handle_ = cublas_handle;

        embedding_table_ = embedding_table;
        final_norm_weight_ = final_norm_weight;
        lm_head_weight_ = lm_head_weight;

        attention_layers_ = attn_layers;
        moe_layers_ = moe_layers;

        max_tokens_ = max_tokens;

        // Create streams
        CUDA_CHECK(hipSetDevice(device_id_));
        compute_stream_ = CudaStream(device_id_);
        comm_stream_ = CudaStream(device_id_, /*high_priority=*/true);

        // Allocate scratch buffers
        allocate_buffers();

        initialized_ = true;

        // Check env var for MoE skip mode (for models with zero-delta MoE)
        const char* skip_env = getenv("GPTOSS_SKIP_MOE");
        if (skip_env && (skip_env[0] == '1' || skip_env[0] == 'y' || skip_env[0] == 'Y')) {
            skip_moe_ = true;
            fprintf(stderr, "[Transformer] GPTOSS_SKIP_MOE=1: skipping MoE layers (zero-delta mode)\n");
        }
    }

    // -----------------------------------------------------------------------
    // Prefill: process a full prompt and return logits for the last token.
    //
    // token_ids_host:  [seq_len] int array on host
    // seq_len:         Number of tokens in the prompt
    // positions_host:  [seq_len] int array on host (position indices)
    // logits_host:     [vocab_size] float output on host (caller-allocated)
    //
    // Returns logits for the last token only (used for next-token prediction).
    //
    // Pipeline per layer:
    //   residual -> attention.prefill() -> residual' -> moe.forward() -> residual''
    // After all layers:
    //   residual -> RMSNorm -> LM head GEMM -> logits
    // -----------------------------------------------------------------------
    void prefill(
        const int* token_ids_host,
        const int* positions_host,
        int seq_len,
        __hip_bfloat16* logits_output,
        const int32_t* slot_mapping_host = nullptr)
    {
        if (seq_len == 0 || !initialized_) return;
        if (seq_len > max_tokens_) {
            fprintf(stderr, "Transformer::prefill: seq_len %d exceeds max_tokens %d\n",
                    seq_len, max_tokens_);
            abort();
        }

        hipStream_t stream = compute_stream_;
        CUDA_CHECK(hipSetDevice(device_id_));

        // Copy token IDs and positions to pre-allocated device buffers
        CUDA_CHECK(hipMemcpyAsync(token_ids_device_, token_ids_host,
                                    seq_len * sizeof(int),
                                    hipMemcpyHostToDevice, stream));
        CUDA_CHECK(hipMemcpyAsync(d_positions_, positions_host,
                                    seq_len * sizeof(int),
                                    hipMemcpyHostToDevice, stream));

        // If slot_mapping provided, copy it to d_slot_mapping_; otherwise
        // it will default to positions (identity) in the attention call below.
        if (slot_mapping_host) {
            CUDA_CHECK(hipMemcpyAsync(d_slot_mapping_, slot_mapping_host,
                                        seq_len * sizeof(int32_t),
                                        hipMemcpyHostToDevice, stream));
        }

        // Step 1: Embedding lookup
        // residual_buf_[0] = embedding_table[token_ids]
        // Shape: [seq_len, hidden_size]
        PROF("embedding", "embed", device_id_, stream);
        embedding_lookup(
            token_ids_device_, embedding_table_,
            residual_buf_[0],
            seq_len, hidden_size, stream);
        PROF_END(device_id_, stream);

        // Step 2: Forward through all 36 layers
        // We ping-pong between residual_buf_[0] and residual_buf_[1]
        // to avoid extra copies. Each layer reads from one buffer and
        // writes to the other (via the residual connection).
        int src_buf = 0;
        for (int layer = 0; layer < num_layers; ++layer) {
            int dst_buf = 1 - src_buf;

            PROF_LAYER(layer);

            // Attention sub-layer:
            //   dst_buf = src_buf + attention(src_buf)
            // Use explicit slot_mapping if provided, else identity (positions)
            const int32_t* prefill_slot_map = slot_mapping_host
                ? d_slot_mapping_
                : reinterpret_cast<const int32_t*>(d_positions_);
            PROF("attention", "attn", device_id_, stream);
            attention_layer_prefill(
                attention_layers_[layer],
                residual_buf_[src_buf],
                d_positions_,
                prefill_slot_map,
                residual_buf_[dst_buf],
                seq_len,
                stream);
            PROF_END(device_id_, stream);

            // EP hidden-state sync: With TP=1, each GPU computes attention
            // independently. Floating-point non-determinism across devices
            // causes hidden states to diverge, which feeds different inputs
            // to each GPU's local experts in the MoE layer. Broadcast GPU 0's
            // attention output to all EP peers so the MoE sees identical inputs.
            if constexpr (ModelConfig::tp_size == 1 && ModelConfig::ep_size > 1) {
                PROF("ep_bcast_hidden", "rccl", device_id_, stream);
                size_t bcast_count = static_cast<size_t>(seq_len) * hidden_size;
                NCCL_CHECK(ncclBroadcast(
                    residual_buf_[dst_buf], residual_buf_[dst_buf],
                    bcast_count, ncclBfloat16, /*root=*/0,
                    ep_nccl_comm_, stream));
                PROF_END(device_id_, stream);
            }

            // MoE sub-layer:
            //   src_buf = dst_buf + moe(dst_buf)
            // We write back to src_buf to complete the ping-pong for this layer.
            PROF("moe", "moe", device_id_, stream);
            moe_layer_forward(
                moe_layers_[layer],
                residual_buf_[dst_buf],
                residual_buf_[src_buf],
                seq_len,
                stream);
            PROF_END(device_id_, stream);

            // Barrier: ensure all GPU work (main + MoE compute + MoE comm
            // streams) drains before the next layer enqueues RCCL ops on
            // the same EP communicator from a different stream.
            CUDA_CHECK(hipStreamSynchronize(stream));

            // After both sub-layers, the result is in residual_buf_[src_buf]
            // (same buffer we started with), ready for the next layer.
        }

        // Step 3: Extract last token's hidden state
        // We only need logits for the last token in prefill
        PROF_LAYER(-1);
        PROF("extract_last_token", "misc", device_id_, stream);
        extract_last_token(
            residual_buf_[src_buf], last_hidden_buf_,
            seq_len - 1, hidden_size, stream);
        PROF_END(device_id_, stream);

        // Step 4: Final RMSNorm on last token's hidden state
        // norm_out_buf_ = RMSNorm(last_hidden_buf_)  [1, hidden_size]
        PROF("final_rmsnorm", "norm", device_id_, stream);
        rmsnorm_forward(
            last_hidden_buf_, final_norm_weight_, norm_out_buf_,
            1, hidden_size, rms_norm_eps, stream);
        PROF_END(device_id_, stream);

        // Step 5: Column-sharded LM head GEMM
        // Each GPU computes logits for its vocab partition only.
        // [1, 2880] x [2880, 100544] -> [1, 100544]
        PROF("lm_head_gemm", "gemm", device_id_, stream);
        cublas_handle_->gemm_bf16_lt(
            norm_out_buf_,       // A: [1, hidden_size=2880]
            lm_head_weight_,     // B: HF layout [tp_vocab_size=100544, hidden_size=2880], transposed by hipBLAS
            local_logits_buf_,   // C: [1, tp_vocab_size=100544]
            1,                   // M = 1
            tp_vocab_size,       // N = 100544 (this GPU's vocab slice)
            hidden_size,         // K = 2880
            stream,
            /*alpha=*/1.0f, /*beta=*/0.0f, /*bias=*/nullptr, /*transB=*/true);
        PROF_END(device_id_, stream);

        // Step 6: All-gather logits across TP GPUs
        PROF("lm_head_allgather", "rccl", device_id_, stream);
        if (ModelConfig::tp_size > 1) {
            NCCL_CHECK(ncclAllGather(
                local_logits_buf_, logits_output,
                static_cast<size_t>(1) * tp_vocab_size,
                ncclBfloat16, nccl_comm_, stream));
        } else {
            // TP1: local_logits IS the full logits -- just copy
            CUDA_CHECK(hipMemcpyAsync(logits_output, local_logits_buf_,
                static_cast<size_t>(tp_vocab_size) * sizeof(__hip_bfloat16),
                hipMemcpyDeviceToDevice, stream));
        }
        PROF_END(device_id_, stream);
    }

    // -----------------------------------------------------------------------
    // Decode: process a single token per sequence and return next-token logits.
    //
    // This is the autoregressive generation step. Each sequence contributes
    // exactly one new token. KV caches are updated within the attention layers.
    //
    // token_ids_host:   [num_seqs] int array on host (one token per seq)
    // positions_host:   [num_seqs] int array on host (current position per seq)
    // slot_mapping_host:[num_seqs] int array on host (KV cache slot per token)
    // block_table_host: [num_seqs, max_num_blocks] int array on host
    // seq_lens_host:    [num_seqs] int array on host (total seq length per seq)
    // num_seqs:         Number of sequences in the batch
    // max_num_blocks:   Maximum number of KV cache blocks per sequence
    // logits_output:    [num_seqs, vocab_size] BF16 output on device
    // -----------------------------------------------------------------------
    void decode(
        const int* token_ids_host,
        const int* positions_host,
        const int32_t* slot_mapping_host,
        const int32_t* block_table_host,
        const int32_t* seq_lens_host,
        int num_seqs,
        int max_num_blocks,
        __hip_bfloat16* logits_output)
    {
        if (num_seqs == 0 || !initialized_) return;
        if (num_seqs > max_tokens_) {
            fprintf(stderr, "Transformer::decode: num_seqs %d exceeds max_tokens %d\n",
                    num_seqs, max_tokens_);
            abort();
        }

        hipStream_t stream = compute_stream_;
        CUDA_CHECK(hipSetDevice(device_id_));

        // Copy decode-specific metadata to pre-allocated device buffers
        size_t block_table_entries = static_cast<size_t>(num_seqs) * max_num_blocks;
        if (static_cast<int>(block_table_entries) > block_table_capacity_) {
            fprintf(stderr, "Transformer::decode: block_table entries %zu exceeds capacity %d\n",
                    block_table_entries, block_table_capacity_);
            abort();
        }

        CUDA_CHECK(hipMemcpyAsync(token_ids_device_, token_ids_host,
                                    num_seqs * sizeof(int),
                                    hipMemcpyHostToDevice, stream));
        CUDA_CHECK(hipMemcpyAsync(d_positions_, positions_host,
                                    num_seqs * sizeof(int),
                                    hipMemcpyHostToDevice, stream));
        CUDA_CHECK(hipMemcpyAsync(d_slot_mapping_, slot_mapping_host,
                                    num_seqs * sizeof(int32_t),
                                    hipMemcpyHostToDevice, stream));
        CUDA_CHECK(hipMemcpyAsync(d_block_table_, block_table_host,
                                    block_table_entries * sizeof(int32_t),
                                    hipMemcpyHostToDevice, stream));
        CUDA_CHECK(hipMemcpyAsync(d_seq_lens_, seq_lens_host,
                                    num_seqs * sizeof(int32_t),
                                    hipMemcpyHostToDevice, stream));

        // ---------------------------------------------------------------
        // HIP Graph fast path for B=1 decode on single GPU.
        //
        // The entire decode (embedding → 36 layers → final norm → LM head)
        // is captured as a single HIP graph, eliminating ~580 individual
        // kernel launch overheads (~5μs each = ~2.9ms total).
        //
        // Graph capture requires stable kernel launch grids. Paged attention
        // uses adaptive splitK that depends on max_num_blocks. We use a fixed
        // large max_num_blocks for the graph (excess splits early-exit safely).
        // H2D copies happen before graph launch, updating device buffers
        // whose addresses are baked into the graph.
        // ---------------------------------------------------------------
        decode_kernels(num_seqs, max_num_blocks, stream);

        // Step 5: All-gather logits across TP GPUs
        PROF("lm_head_allgather", "rccl", device_id_, stream);
        if (ModelConfig::tp_size > 1) {
            NCCL_CHECK(ncclAllGather(
                local_logits_buf_, logits_output,
                static_cast<size_t>(num_seqs) * tp_vocab_size,
                ncclBfloat16, nccl_comm_, stream));
        } else {
            CUDA_CHECK(hipMemcpyAsync(logits_output, local_logits_buf_,
                static_cast<size_t>(num_seqs) * tp_vocab_size * sizeof(__hip_bfloat16),
                hipMemcpyDeviceToDevice, stream));
        }
        PROF_END(device_id_, stream);
    }

    // -----------------------------------------------------------------------
    // Decode kernel core: embedding → 36 layers → final norm → LM head
    // Extracted for HIP graph capture — all operations are kernel launches
    // on `stream`, no host stalls, no H2D copies.
    // -----------------------------------------------------------------------
    void decode_kernels(int num_seqs, int max_num_blocks, hipStream_t stream) {
        // Step 1: Embedding lookup
        PROF("embedding", "embed", device_id_, stream);
        embedding_lookup(
            token_ids_device_, embedding_table_,
            residual_buf_[0],
            num_seqs, hidden_size, stream);
        PROF_END(device_id_, stream);

        // Step 2: Forward through all 36 layers
        int src_buf = 0;
        for (int layer = 0; layer < num_layers; ++layer) {
            int dst_buf = 1 - src_buf;
            PROF_LAYER(layer);

            // Attention: output delta only (no residual_add)
            PROF("attention_decode", "attn", device_id_, stream);
            attention_layer_decode_no_residual(
                attention_layers_[layer],
                residual_buf_[src_buf],
                d_positions_,
                d_slot_mapping_,
                d_block_table_,
                d_seq_lens_,
                max_num_blocks,
                num_seqs,
                stream);
            PROF_END(device_id_, stream);
            __hip_bfloat16* delta = attention_layer_get_proj_out(
                attention_layers_[layer]);

            // EP hidden-state sync: broadcast attention delta from GPU 0
            // so all EP peers compute identical post-attention residuals.
            if constexpr (ModelConfig::tp_size == 1 && ModelConfig::ep_size > 1) {
                PROF("ep_bcast_delta", "rccl", device_id_, stream);
                size_t bcast_count = static_cast<size_t>(num_seqs) * hidden_size;
                NCCL_CHECK(ncclBroadcast(
                    delta, delta,
                    bcast_count, ncclBfloat16, /*root=*/0,
                    ep_nccl_comm_, stream));
                PROF_END(device_id_, stream);
            }

            // Fused: residual[dst] = src + delta; norm_out = RMSNorm(residual[dst])
            PROF("fused_res_rmsnorm", "norm", device_id_, stream);
            const __hip_bfloat16* ffn_norm_w = moe_layer_get_ffn_norm_weight(
                moe_layers_[layer]);
            fused_residual_rmsnorm_forward(
                residual_buf_[src_buf], delta, ffn_norm_w,
                norm_out_buf_, residual_buf_[dst_buf],
                num_seqs, hidden_size, rms_norm_eps, stream);
            PROF_END(device_id_, stream);

            // MoE: takes pre-normed input + residual for combine
            if (!skip_moe_) {
                PROF("moe_decode", "moe", device_id_, stream);
                moe_layer_forward_prenormed(
                    moe_layers_[layer],
                    norm_out_buf_,
                    residual_buf_[dst_buf],
                    residual_buf_[src_buf],
                    num_seqs,
                    stream);
                PROF_END(device_id_, stream);
            } else {
                // MoE produces zero delta for this model — just copy residual
                CUDA_CHECK(hipMemcpyAsync(
                    residual_buf_[src_buf], residual_buf_[dst_buf],
                    static_cast<size_t>(num_seqs) * hidden_size * sizeof(__hip_bfloat16),
                    hipMemcpyDeviceToDevice, stream));
            }

            // Barrier: needed for EP>1 to prevent RCCL communicator reuse across layers.
            // For EP=1 (single GPU), MoE forward_core already drains both streams.
            if constexpr (ModelConfig::ep_size > 1)
                CUDA_CHECK(hipStreamSynchronize(stream));

            // Result is back in residual_buf_[src_buf]
        }

        // Step 3: Final RMSNorm
        PROF_LAYER(-1);
        PROF("final_rmsnorm", "norm", device_id_, stream);
        rmsnorm_forward(
            residual_buf_[src_buf], final_norm_weight_, norm_out_buf_,
            num_seqs, hidden_size, rms_norm_eps, stream);
        PROF_END(device_id_, stream);

        // Step 4: LM head GEMM
        PROF("lm_head_gemm", "gemm", device_id_, stream);
        if (num_seqs == 1) {
            gemv_bf16_forward(norm_out_buf_, lm_head_weight_, local_logits_buf_,
                              tp_vocab_size, hidden_size, stream);
        } else {
            cublas_handle_->gemm_bf16_lt(
                norm_out_buf_, lm_head_weight_, local_logits_buf_,
                num_seqs, tp_vocab_size, hidden_size, stream,
                /*alpha=*/1.0f, /*beta=*/0.0f, /*bias=*/nullptr, /*transB=*/true);
        }
        PROF_END(device_id_, stream);
    }

private:
    // -----------------------------------------------------------------------
    // Allocate shared scratch buffers
    // -----------------------------------------------------------------------
    void allocate_buffers() {
        CUDA_CHECK(hipSetDevice(device_id_));

        int64_t residual_bytes = static_cast<int64_t>(max_tokens_) * hidden_size * sizeof(__hip_bfloat16);

        CUDA_CHECK(hipMalloc(&residual_buf_[0], residual_bytes));
        CUDA_CHECK(hipMalloc(&residual_buf_[1], residual_bytes));

        // norm_out_buf needs to hold [max_tokens, hidden_size] for decode
        CUDA_CHECK(hipMalloc(&norm_out_buf_, residual_bytes));

        // Logits buffers: only need max(1, max_batch_size) rows, not max_tokens.
        // Prefill extracts last token's hidden state, so logits are always single-row.
        // Decode produces one logit row per sequence in the batch.
        int logits_rows = std::max(1, ModelConfig::max_batch_size);
        int64_t logits_bytes = static_cast<int64_t>(logits_rows) * vocab_size * sizeof(__hip_bfloat16);
        CUDA_CHECK(hipMalloc(&logits_buf_, logits_bytes));

        int64_t local_logits_bytes = static_cast<int64_t>(logits_rows) * tp_vocab_size * sizeof(__hip_bfloat16);
        CUDA_CHECK(hipMalloc(&local_logits_buf_, local_logits_bytes));

        // Last hidden state: [1, hidden_size] for prefill
        CUDA_CHECK(hipMalloc(&last_hidden_buf_,
                              hidden_size * sizeof(__hip_bfloat16)));

        // Token IDs on device
        CUDA_CHECK(hipMalloc(&token_ids_device_,
                              max_tokens_ * sizeof(int)));

        // Pre-allocated decode metadata buffers
        CUDA_CHECK(hipMalloc(&d_positions_, max_tokens_ * sizeof(int)));
        CUDA_CHECK(hipMalloc(&d_slot_mapping_, max_tokens_ * sizeof(int32_t)));
        CUDA_CHECK(hipMalloc(&d_seq_lens_, max_tokens_ * sizeof(int32_t)));

        // Block table: max_tokens seqs x max blocks per seq.
        // max_seq_len=131072, kv_block_size=16 -> 8192 blocks/seq max.
        // For typical batch=1 decode this is 32KB; for batch=256 it's 8MB.
        constexpr int max_blocks_per_seq = ModelConfig::rope_max_pos / ModelConfig::kv_block_size;
        block_table_capacity_ = max_tokens_ * max_blocks_per_seq;
        CUDA_CHECK(hipMalloc(&d_block_table_,
                              static_cast<size_t>(block_table_capacity_) * sizeof(int32_t)));
    }

    // -----------------------------------------------------------------------
    // Free scratch buffers
    // -----------------------------------------------------------------------
    void free_buffers() {
        auto safe_free = [](void*& ptr) {
            if (ptr) {
                hipFree(ptr);
                ptr = nullptr;
            }
        };
        safe_free(reinterpret_cast<void*&>(residual_buf_[0]));
        safe_free(reinterpret_cast<void*&>(residual_buf_[1]));
        safe_free(reinterpret_cast<void*&>(norm_out_buf_));
        safe_free(reinterpret_cast<void*&>(logits_buf_));
        safe_free(reinterpret_cast<void*&>(local_logits_buf_));
        safe_free(reinterpret_cast<void*&>(last_hidden_buf_));
        safe_free(reinterpret_cast<void*&>(token_ids_device_));
        safe_free(reinterpret_cast<void*&>(d_positions_));
        safe_free(reinterpret_cast<void*&>(d_slot_mapping_));
        safe_free(reinterpret_cast<void*&>(d_seq_lens_));
        safe_free(reinterpret_cast<void*&>(d_block_table_));
    }
};

// ---------------------------------------------------------------------------
// Extern wrapper functions for cross-TU access
// ---------------------------------------------------------------------------

Transformer* transformer_create() {
    return new Transformer();
}

void transformer_destroy(Transformer* t) {
    if (t) delete t;
}

void transformer_init(Transformer* t, int rank, ncclComm_t nccl_comm,
                       ncclComm_t ep_nccl_comm,
                       CublasHandle* cublas, const __hip_bfloat16* embedding_table,
                       const __hip_bfloat16* final_norm_weight,
                       const __hip_bfloat16* lm_head_weight,
                       AttentionLayer** attn_layers, MoELayer** moe_layers,
                       int max_tokens) {
    t->init(rank, nccl_comm, ep_nccl_comm, cublas, embedding_table, final_norm_weight,
            lm_head_weight, attn_layers, moe_layers, max_tokens);
}

void transformer_prefill(Transformer* t, const int* token_ids_host,
                          const int* positions_host, int seq_len,
                          __hip_bfloat16* logits_output,
                          const int32_t* slot_mapping_host) {
    t->prefill(token_ids_host, positions_host, seq_len, logits_output, slot_mapping_host);
}

void transformer_decode(Transformer* t, const int* token_ids_host,
                         const int* positions_host, const int32_t* slot_mapping_host,
                         const int32_t* block_table_host, const int32_t* seq_lens_host,
                         int num_seqs, int max_num_blocks, __hip_bfloat16* logits_output) {
    t->decode(token_ids_host, positions_host, slot_mapping_host,
              block_table_host, seq_lens_host, num_seqs, max_num_blocks, logits_output);
}

hipStream_t transformer_compute_stream(Transformer* t) {
    return t->compute_stream_;
}

} // namespace gptoss
