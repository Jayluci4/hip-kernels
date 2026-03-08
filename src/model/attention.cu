// Attention Layer for GPT-OSS-120B Inference Engine
// Handles prefill (flash attention) and decode (paged attention) paths.
//
// Tensor Parallel (TP2): each GPU handles half the attention heads.
//   GPU 0: Q heads 0-31, KV heads 0-3   (GQA ratio 8:1 preserved)
//   GPU 1: Q heads 32-63, KV heads 4-7
// After output projection, an RCCL all-reduce combines partial results
// so both GPUs hold identical hidden states for the subsequent MoE layer.
// Cost: ~0.013ms per all-reduce over Infinity Fabric.
// Savings: 50% of all attention compute + 50% KV cache memory per GPU.
//
// Sliding window (128 tokens) on even layers; full attention on odd layers.

#include "hip_compat.h"
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <rccl/rccl.h>

#include "config.h"
#include "tensor.h"
#include "cuda_utils.h"
#include "nccl_utils.h"
#include "profiler.h"

namespace gptoss {

// Custom BF16 GEMV for M=1 decode -- replaces hipBLASLt overhead
extern void gemv_bf16_forward(
    const __hip_bfloat16* input,
    const __hip_bfloat16* weight,
    __hip_bfloat16* output,
    int N, int K,
    hipStream_t stream,
    const __hip_bfloat16* bias = nullptr);

// ---------------------------------------------------------------------------
// Extern kernel declarations (defined in src/kernels/*.cu)
// ---------------------------------------------------------------------------

extern void rmsnorm_forward(
    const __hip_bfloat16* input,
    const __hip_bfloat16* weight,
    __hip_bfloat16* output,
    int num_tokens,
    int hidden_size,
    float eps,
    hipStream_t stream);

extern void rope_forward(
    const __hip_bfloat16* q,
    const __hip_bfloat16* k,
    __hip_bfloat16* q_out,
    __hip_bfloat16* k_out,
    const float* cos_table,
    const float* sin_table,
    const float2* cos_sin_table,
    const int* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    hipStream_t stream);

extern void flash_attention_forward(
    const __hip_bfloat16* Q,
    const __hip_bfloat16* K,
    const __hip_bfloat16* V,
    __hip_bfloat16* output,
    const float* sink_values,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int window_size,
    hipStream_t stream);

extern void paged_attention_forward(
    const __hip_bfloat16* query,
    const __hip_bfloat16* k_cache,
    const __hip_bfloat16* v_cache,
    __hip_bfloat16* output,
    const int32_t* block_table,
    const int32_t* seq_lens,
    const float* sink_values,
    const float2* cos_sin_table,
    int num_seqs,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_num_blocks,
    int window_size,
    float* partial_out,
    float* partial_max,
    float* partial_sum,
    int max_splitK,
    hipStream_t stream);

// ---------------------------------------------------------------------------
// Residual-add kernel: output[i] = residual[i] + x[i]
// Fused BF16 residual connection to avoid extra kernel launch overhead.
// ---------------------------------------------------------------------------

__global__ void residual_add_kernel(
    const __hip_bfloat16* __restrict__ residual,
    const __hip_bfloat16* __restrict__ x,
    __hip_bfloat16* __restrict__ output,
    int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Process 2 elements per thread via vectorized loads when possible
    int vec_idx = idx * 2;
    if (vec_idx + 1 < num_elements) {
        __hip_bfloat162 r = *reinterpret_cast<const __hip_bfloat162*>(residual + vec_idx);
        __hip_bfloat162 v = *reinterpret_cast<const __hip_bfloat162*>(x + vec_idx);
        float r0 = __bfloat162float(__low2bfloat16(r));
        float r1 = __bfloat162float(__high2bfloat16(r));
        float v0 = __bfloat162float(__low2bfloat16(v));
        float v1 = __bfloat162float(__high2bfloat16(v));
        __hip_bfloat162 out = __halves2bfloat162(
            __float2bfloat16(r0 + v0),
            __float2bfloat16(r1 + v1));
        *reinterpret_cast<__hip_bfloat162*>(output + vec_idx) = out;
    } else if (vec_idx < num_elements) {
        float r0 = __bfloat162float(residual[vec_idx]);
        float v0 = __bfloat162float(x[vec_idx]);
        output[vec_idx] = __float2bfloat16(r0 + v0);
    }
}

// ---------------------------------------------------------------------------
// Vec4 residual-add kernel (MI300X optimization): 128-bit int4 loads
//
// Processes 8 BF16 elements per thread using int4 (128-bit) loads/stores.
// This doubles memory throughput vs the vec2 kernel (__hip_bfloat162 = 32-bit).
// int4 contains 4x __hip_bfloat162, each holding 2x BF16 values.
//
// Pipeline per thread:
//   1. Load int4 from residual (128 bits = 8 BF16 elements)
//   2. Load int4 from x        (128 bits = 8 BF16 elements)
//   3. Reinterpret as 4x __hip_bfloat162 pairs
//   4. Convert each pair to FP32, add, convert back to BF16
//   5. Pack results into int4 and store
// ---------------------------------------------------------------------------
__global__ void residual_add_kernel_vec4(
    const __hip_bfloat16* __restrict__ residual,
    const __hip_bfloat16* __restrict__ x,
    __hip_bfloat16* __restrict__ output,
    int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_start = idx * 8;

    if (elem_start + 7 >= num_elements) return;  // Caller guarantees alignment

    // Load 128 bits (8 BF16 elements) from each input
    const int4* res_ptr = reinterpret_cast<const int4*>(residual + elem_start);
    const int4* x_ptr   = reinterpret_cast<const int4*>(x + elem_start);
    int4 r_vec = *res_ptr;
    int4 x_vec = *x_ptr;

    // Reinterpret int4 as 4x __hip_bfloat162
    const __hip_bfloat162* r_pairs = reinterpret_cast<const __hip_bfloat162*>(&r_vec);
    const __hip_bfloat162* x_pairs = reinterpret_cast<const __hip_bfloat162*>(&x_vec);

    int4 out_vec;
    __hip_bfloat162* out_pairs = reinterpret_cast<__hip_bfloat162*>(&out_vec);

    // Process 4 pairs of BF16 values (8 elements total)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float r0 = __bfloat162float(__low2bfloat16(r_pairs[i]));
        float r1 = __bfloat162float(__high2bfloat16(r_pairs[i]));
        float v0 = __bfloat162float(__low2bfloat16(x_pairs[i]));
        float v1 = __bfloat162float(__high2bfloat16(x_pairs[i]));
        out_pairs[i] = __halves2bfloat162(
            __float2bfloat16(r0 + v0),
            __float2bfloat16(r1 + v1));
    }

    // Store 128 bits (8 BF16 elements) to output
    *reinterpret_cast<int4*>(output + elem_start) = out_vec;
}

static void residual_add(
    const __hip_bfloat16* residual,
    const __hip_bfloat16* x,
    __hip_bfloat16* output,
    int num_elements,
    hipStream_t stream)
{
    if (num_elements == 0) return;

    int threads = 256;

    if (num_elements % 8 == 0) {
        // Vec4 path (MI300X optimized): each thread processes 8 BF16 elements
        // via 128-bit int4 loads, doubling memory throughput vs vec2.
        int blocks = cdiv(num_elements / 8, threads);
        residual_add_kernel_vec4<<<blocks, threads, 0, stream>>>(
            residual, x, output, num_elements);
    } else {
        // Vec2 fallback: each thread processes 2 elements
        int blocks = cdiv(cdiv(num_elements, 2), threads);
        residual_add_kernel<<<blocks, threads, 0, stream>>>(
            residual, x, output, num_elements);
    }
    CUDA_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------------------
// KV cache update kernel: copy new K/V into the paged cache at given slots
// ---------------------------------------------------------------------------

__global__ void kv_cache_update_kernel(
    __hip_bfloat16* __restrict__ k_cache,   // [max_num_blocks, num_kv_heads, block_size, head_dim]
    __hip_bfloat16* __restrict__ v_cache,   // [max_num_blocks, num_kv_heads, block_size, head_dim]
    const __hip_bfloat16* __restrict__ k_new,  // [num_tokens, num_kv_heads * head_dim]
    const __hip_bfloat16* __restrict__ v_new,  // [num_tokens, num_kv_heads * head_dim]
    const int32_t* __restrict__ slot_mapping,   // [num_tokens] -> flat slot index
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size)
{
    // Grid: one block per token, threads iterate over kv_heads * head_dim
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int kv_dim = num_kv_heads * head_dim;
    int slot = slot_mapping[token_idx];
    int block_idx = slot / block_size;
    int offset_in_block = slot % block_size;

    // Cache layout: [block_idx, head, offset_in_block, head_dim]
    // Stride: block_idx * (num_kv_heads * block_size * head_dim)
    //       + head * (block_size * head_dim)
    //       + offset_in_block * head_dim
    //       + d
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        int head = i / head_dim;
        int d = i % head_dim;
        int64_t cache_offset =
            static_cast<int64_t>(block_idx) * (num_kv_heads * block_size * head_dim) +
            static_cast<int64_t>(head) * (block_size * head_dim) +
            static_cast<int64_t>(offset_in_block) * head_dim +
            d;
        int64_t src_offset = static_cast<int64_t>(token_idx) * kv_dim + i;

        k_cache[cache_offset] = k_new[src_offset];
        v_cache[cache_offset] = v_new[src_offset];
    }
}

static void kv_cache_update(
    __hip_bfloat16* k_cache,
    __hip_bfloat16* v_cache,
    const __hip_bfloat16* k_new,
    const __hip_bfloat16* v_new,
    const int32_t* slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size,
    hipStream_t stream)
{
    if (num_tokens == 0) return;
    int kv_dim = num_kv_heads * head_dim;
    int threads = (kv_dim <= 256) ? kv_dim : 256;
    kv_cache_update_kernel<<<num_tokens, threads, 0, stream>>>(
        k_cache, v_cache, k_new, v_new, slot_mapping,
        num_tokens, num_kv_heads, head_dim, block_size);
    CUDA_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------------------
// Fused split_qkv + rope_K + kv_cache_update for decode (B=1).
// Eliminates 2 kernel launches per layer (split_qkv + rope_k merged with kv_update).
// Reads QKV buffer once, writes Q to q_out, applies RoPE to K, writes K+V to cache.
// ---------------------------------------------------------------------------
__global__ void fused_split_rope_kv_kernel(
    const __hip_bfloat16* __restrict__ qkv,      // [num_tokens, qkv_size]
    __hip_bfloat16* __restrict__ q_out,            // [num_tokens, q_dim]
    __hip_bfloat16* __restrict__ k_cache,          // [max_blocks, kv_heads, block_size, head_dim]
    __hip_bfloat16* __restrict__ v_cache,          // [max_blocks, kv_heads, block_size, head_dim]
    const float2* __restrict__ cos_sin_table,      // [max_pos, half_dim]
    const int* __restrict__ positions,             // [num_tokens]
    const int32_t* __restrict__ slot_mapping,      // [num_tokens]
    int num_tokens)
{
    constexpr int qkv_size = ModelConfig::tp_qkv_size;
    constexpr int q_dim    = ModelConfig::tp_q_dim;
    constexpr int kv_dim   = ModelConfig::tp_kv_dim;
    constexpr int num_kv_heads = ModelConfig::tp_kv_heads;
    constexpr int head_dim = ModelConfig::head_dim;
    constexpr int half_dim = head_dim / 2;
    constexpr int block_size = ModelConfig::kv_block_size;

    int token = blockIdx.x;
    if (token >= num_tokens) return;

    const __hip_bfloat16* src = qkv + static_cast<int64_t>(token) * qkv_size;
    __hip_bfloat16* q_dst = q_out + static_cast<int64_t>(token) * q_dim;

    // Step 1: Copy Q portion
    for (int i = threadIdx.x; i < q_dim; i += blockDim.x) {
        q_dst[i] = src[i];
    }

    // Step 2: Read K from QKV, apply RoPE, write to KV cache
    int pos = positions[token];
    int slot = slot_mapping[token];
    int block_idx = slot / block_size;
    int offset_in_block = slot % block_size;

    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        int head = i / head_dim;
        int d = i % head_dim;

        // Read K from QKV buffer
        float k_val = __bfloat162float(src[q_dim + i]);

        // Apply RoPE rotation (half-split convention)
        float rotated;
        if (d < half_dim) {
            float k_pair = __bfloat162float(src[q_dim + head * head_dim + half_dim + d]);
            float2 cs = cos_sin_table[pos * half_dim + d];
            rotated = k_val * cs.x - k_pair * cs.y;
        } else {
            int d_lo = d - half_dim;
            float k_pair = __bfloat162float(src[q_dim + head * head_dim + d_lo]);
            float2 cs = cos_sin_table[pos * half_dim + d_lo];
            rotated = k_pair * cs.y + k_val * cs.x;
        }

        // Write rotated K to cache
        int64_t cache_offset =
            static_cast<int64_t>(block_idx) * (num_kv_heads * block_size * head_dim) +
            static_cast<int64_t>(head) * (block_size * head_dim) +
            static_cast<int64_t>(offset_in_block) * head_dim + d;
        k_cache[cache_offset] = __float2bfloat16(rotated);

        // Read V from QKV buffer and write to cache
        v_cache[cache_offset] = src[q_dim + kv_dim + i];
    }
}

// ---------------------------------------------------------------------------
// Split fused QKV tensor into separate Q, K, V buffers (free function)
//
// TP2 QKV layout per token: [Q(tp_q_dim) | K(tp_kv_dim) | V(tp_kv_dim)]
// = [2048 | 256 | 256] = 2560 total (per GPU)
// ---------------------------------------------------------------------------
__global__ void split_qkv_kernel(
    const __hip_bfloat16* __restrict__ qkv,  // [num_tokens, qkv_size]
    __hip_bfloat16* __restrict__ q_out,       // [num_tokens, q_dim]
    __hip_bfloat16* __restrict__ k_out,       // [num_tokens, kv_dim]
    __hip_bfloat16* __restrict__ v_out,       // [num_tokens, kv_dim]
    int num_tokens)
{
    constexpr int qkv_size = ModelConfig::tp_qkv_size;
    constexpr int q_dim    = ModelConfig::tp_q_dim;
    constexpr int kv_dim   = ModelConfig::tp_kv_dim;

    int token = blockIdx.x;
    if (token >= num_tokens) return;

    const __hip_bfloat16* src = qkv + static_cast<int64_t>(token) * qkv_size;
    __hip_bfloat16* q_dst = q_out + static_cast<int64_t>(token) * q_dim;
    __hip_bfloat16* k_dst = k_out + static_cast<int64_t>(token) * kv_dim;
    __hip_bfloat16* v_dst = v_out + static_cast<int64_t>(token) * kv_dim;

    for (int i = threadIdx.x; i < q_dim; i += blockDim.x) {
        q_dst[i] = src[i];
    }
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        k_dst[i] = src[q_dim + i];
    }
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        v_dst[i] = src[q_dim + kv_dim + i];
    }
}

// ---------------------------------------------------------------------------
// AttentionLayer
// ---------------------------------------------------------------------------

class AttentionLayer {
public:
    // ---- Configuration ----
    int layer_id_;
    int device_id_;
    int rank_;                             // GPU rank in TP group (0 or 1)
    int window_size_;                      // 0 = full attention, 128 = sliding window

    // ---- RCCL communicator for TP all-reduce (not owned) ----
    ncclComm_t nccl_comm_;

    // ---- Per-layer weights (BF16, device memory, not owned) ----
    // Column-parallel QKV: each GPU stores its local head slice
    const __hip_bfloat16* qkv_weight_;      // [hidden_size, tp_qkv_size] = [2880, 2560]
    const __hip_bfloat16* qkv_bias_;        // [tp_qkv_size] fused Q/K/V bias (nullptr if none)
    // Row-parallel output projection: each GPU stores its local head slice
    const __hip_bfloat16* o_proj_weight_;   // [tp_attn_out_size, hidden_size] = [2048, 2880]
    const __hip_bfloat16* o_proj_bias_;     // [hidden_size] output projection bias (nullptr if none)
    const __hip_bfloat16* attn_norm_weight_; // [hidden_size] = [2880]

    // Sink token attention values for sliding window layers
    const float* sink_values_;             // Layer-specific sink values (may be nullptr)

    // ---- RoPE tables (shared across layers, device memory, not owned) ----
    const float* cos_table_;               // [max_pos, head_dim/2]
    const float* sin_table_;               // [max_pos, head_dim/2]
    const float2* cos_sin_table_;          // [max_pos, head_dim/2] interleaved {cos,sin}

    // ---- KV cache (device memory, not owned) ----
    // TP2: each GPU caches only its local KV heads (4 instead of 8)
    __hip_bfloat16* k_cache_;              // [max_num_blocks, tp_kv_heads, block_size, head_dim]
    __hip_bfloat16* v_cache_;              // [max_num_blocks, tp_kv_heads, block_size, head_dim]
    int max_num_blocks_;
    int cache_blocks_per_layer_;

    // ---- Scratch buffers (device memory, not owned) ----
    // All Q/K/V buffers sized for LOCAL (TP) head counts, not global.
    __hip_bfloat16* norm_buf_;              // [max_tokens, hidden_size]
    __hip_bfloat16* qkv_buf_;              // [max_tokens, tp_qkv_size]
    __hip_bfloat16* q_buf_;                // [max_tokens, tp_q_dim]
    __hip_bfloat16* k_buf_;                // [max_tokens, tp_kv_dim]
    __hip_bfloat16* v_buf_;                // [max_tokens, tp_kv_dim]
    __hip_bfloat16* q_rope_buf_;           // [max_tokens, tp_q_dim]
    __hip_bfloat16* k_rope_buf_;           // [max_tokens, tp_kv_dim]
    __hip_bfloat16* attn_out_buf_;         // [max_tokens, tp_attn_out_size]
    __hip_bfloat16* proj_out_buf_;         // [max_tokens, hidden_size] (full -- receives all-reduce)

    // ---- SplitK partial buffers (owned, allocated in init) ----
    float* partial_out_ = nullptr;      // [max_batch, kv_heads, max_splitK, gqa_ratio, head_dim]
    float* partial_max_ = nullptr;      // [max_batch, kv_heads, max_splitK, gqa_ratio]
    float* partial_sum_ = nullptr;      // [max_batch, kv_heads, max_splitK, gqa_ratio]
    static constexpr int pa_max_splitK_ = 64;

    // ---- hipBLAS handle (not owned) ----
    const CublasHandle* cublas_;

    // ---- HIP Graph cache for decode attention (steps 1-7) ----
    // At B>=64, paged attention always launches exactly 1 kernel (splitK=1),
    // making the kernel count fixed and graph-safe. Each layer needs its own
    // graph since weights differ. Keyed by batch size (num_seqs).
    static constexpr int MIN_GRAPH_BATCH = 64;
    std::unordered_map<int, hipGraphExec_t> decode_graph_cache_;

    // ---- Dedicated comm stream + events for overlapping all-reduce ----
    // The all-reduce after O-projection runs on comm_stream_ while the
    // compute stream can proceed with non-dependent work. Two events
    // synchronize the handoff: ar_compute_done_ (compute -> comm before
    // all-reduce) and ar_comm_done_ (comm -> compute after all-reduce).
    hipStream_t comm_stream_;
    hipEvent_t  ar_compute_done_;
    hipEvent_t  ar_comm_done_;
    bool         owns_comm_resources_;  // true if we created the stream/events

    // ---------------------------------------------------------------------------
    // Tensor-parallel dimensions (TP2: each GPU handles half the heads)
    //
    // Megatron-LM style parallelism:
    //   QKV projection: column-parallel (each GPU produces local heads)
    //   Attention:       independent (each GPU processes local heads)
    //   O projection:    row-parallel (each GPU produces partial hidden_size)
    //   All-reduce:      sum partial outputs across GPUs via Infinity Fabric
    //
    // GQA ratio preserved: 32 Q heads / 4 KV heads = 8 (same as full model)
    // ---------------------------------------------------------------------------
    static constexpr int hidden_size = ModelConfig::hidden_size;           // 2880
    static constexpr int head_dim = ModelConfig::head_dim;                 // 64
    static constexpr int kv_block_size = ModelConfig::kv_block_size;       // 16
    static constexpr float rms_norm_eps = ModelConfig::rms_norm_eps;

    // TP-local head counts (this GPU's share)
    static constexpr int num_q_heads = ModelConfig::tp_q_heads;            // 32
    static constexpr int num_kv_heads = ModelConfig::tp_kv_heads;          // 4
    static constexpr int q_dim = ModelConfig::tp_q_dim;                    // 2048
    static constexpr int kv_dim = ModelConfig::tp_kv_dim;                  // 256
    static constexpr int qkv_size = ModelConfig::tp_qkv_size;             // 2560
    static constexpr int attn_out_size = ModelConfig::tp_attn_out_size;    // 2048

    AttentionLayer()
        : comm_stream_(nullptr)
        , ar_compute_done_(nullptr)
        , ar_comm_done_(nullptr)
        , owns_comm_resources_(false)
    {}

    ~AttentionLayer() {
        for (auto& [batch_size, exec] : decode_graph_cache_)
            hipGraphExecDestroy(exec);
        decode_graph_cache_.clear();
        destroy_comm_resources();
    }

    // ---- Initialization ----
    void init(
        int layer_id,
        int device_id,
        int rank,
        ncclComm_t nccl_comm,
        const CublasHandle* cublas,
        const __hip_bfloat16* qkv_weight,     // [hidden_size, tp_qkv_size] per-GPU slice
        const __hip_bfloat16* qkv_bias,        // [tp_qkv_size] fused bias (nullptr if none)
        const __hip_bfloat16* o_proj_weight,   // [tp_attn_out_size, hidden_size] per-GPU slice
        const __hip_bfloat16* o_proj_bias,      // [hidden_size] bias (nullptr if none)
        const __hip_bfloat16* attn_norm_weight,
        const float* cos_table,
        const float* sin_table,
        const float2* cos_sin_table,
        const float* sink_values,
        __hip_bfloat16* k_cache,               // [max_blocks, tp_kv_heads, block_size, head_dim]
        __hip_bfloat16* v_cache,
        int max_num_blocks,
        __hip_bfloat16* norm_buf,
        __hip_bfloat16* qkv_buf,
        __hip_bfloat16* q_buf,
        __hip_bfloat16* k_buf,
        __hip_bfloat16* v_buf,
        __hip_bfloat16* q_rope_buf,
        __hip_bfloat16* k_rope_buf,
        __hip_bfloat16* attn_out_buf,
        __hip_bfloat16* proj_out_buf)
    {
        layer_id_ = layer_id;
        device_id_ = device_id;
        rank_ = rank;
        nccl_comm_ = nccl_comm;
        cublas_ = cublas;

        window_size_ = ModelConfig::get_window_size(layer_id);

        // Weights (already sliced for this GPU's TP partition)
        qkv_weight_ = qkv_weight;
        qkv_bias_ = qkv_bias;
        o_proj_weight_ = o_proj_weight;
        o_proj_bias_ = o_proj_bias;
        attn_norm_weight_ = attn_norm_weight;

        // RoPE (shared -- position-based, head-independent)
        cos_table_ = cos_table;
        sin_table_ = sin_table;
        cos_sin_table_ = cos_sin_table;
        sink_values_ = sink_values;

        if (max_num_blocks <= 0 || (max_num_blocks % ModelConfig::num_layers) != 0) {
            fprintf(stderr,
                    "AttentionLayer::init invalid KV geometry: total_blocks=%d layers=%d\n",
                    max_num_blocks, ModelConfig::num_layers);
            abort();
        }
        max_num_blocks_ = max_num_blocks;
        cache_blocks_per_layer_ = max_num_blocks_ / ModelConfig::num_layers;

        // KV cache (TP-local) must be layer-sliced. Without this offset, all
        // layers alias the same cache blocks and decode diverges quickly.
        const int64_t layer_block_offset =
            static_cast<int64_t>(layer_id_) * cache_blocks_per_layer_;
        const int64_t layer_elem_offset =
            layer_block_offset * num_kv_heads * kv_block_size * head_dim;
        k_cache_ = k_cache + layer_elem_offset;
        v_cache_ = v_cache + layer_elem_offset;

        // Create dedicated comm stream and sync events for TP all-reduce overlap.
        // With TP1, all-reduce is a no-op -- skip creating these resources entirely.
        CUDA_CHECK(hipSetDevice(device_id_));
        if (ModelConfig::tp_size > 1) {
            int low_prio, high_prio;
            CUDA_CHECK(hipDeviceGetStreamPriorityRange(&low_prio, &high_prio));
            CUDA_CHECK(hipStreamCreateWithPriority(
                &comm_stream_, hipStreamNonBlocking, high_prio));
            CUDA_CHECK(hipEventCreateWithFlags(
                &ar_compute_done_, hipEventDisableTiming));
            CUDA_CHECK(hipEventCreateWithFlags(
                &ar_comm_done_, hipEventDisableTiming));
            owns_comm_resources_ = true;
        }

        // Allocate splitK partial buffers for flash decoding
        // batch_size and splitK trade off: large batches use splitK=1, B=1 uses high splitK.
        // Max concurrent = max(max_batch_size, pa_max_splitK_) to avoid overallocation.
        {
            int max_concurrent = (ModelConfig::max_batch_size > pa_max_splitK_)
                                 ? ModelConfig::max_batch_size : pa_max_splitK_;
            int partial_entries = max_concurrent * num_kv_heads * ModelConfig::gqa_ratio;
            CUDA_CHECK(hipMalloc(&partial_out_,
                partial_entries * head_dim * sizeof(float)));
            CUDA_CHECK(hipMalloc(&partial_max_,
                partial_entries * sizeof(float)));
            CUDA_CHECK(hipMalloc(&partial_sum_,
                partial_entries * sizeof(float)));
        }

        // Scratch buffers (sized for TP-local dimensions)
        norm_buf_ = norm_buf;
        qkv_buf_ = qkv_buf;
        q_buf_ = q_buf;
        k_buf_ = k_buf;
        v_buf_ = v_buf;
        q_rope_buf_ = q_rope_buf;
        k_rope_buf_ = k_rope_buf;
        attn_out_buf_ = attn_out_buf;
        proj_out_buf_ = proj_out_buf;
    }

    void destroy_comm_resources() {
        if (partial_out_) { hipFree(partial_out_); partial_out_ = nullptr; }
        if (partial_max_) { hipFree(partial_max_); partial_max_ = nullptr; }
        if (partial_sum_) { hipFree(partial_sum_); partial_sum_ = nullptr; }
        if (!owns_comm_resources_) return;
        if (ar_compute_done_) { hipEventDestroy(ar_compute_done_); ar_compute_done_ = nullptr; }
        if (ar_comm_done_)    { hipEventDestroy(ar_comm_done_);    ar_comm_done_    = nullptr; }
        if (comm_stream_)     { hipStreamDestroy(comm_stream_);    comm_stream_     = nullptr; }
        owns_comm_resources_ = false;
    }

    // -----------------------------------------------------------------------
    // Prefill path: processes a full prompt sequence with flash attention
    //
    // Tensor Parallel (TP2): each GPU processes its LOCAL head partition.
    //   Steps 1-5: purely local (no cross-GPU communication)
    //   Step 6: O projection produces partial hidden_size on each GPU
    //   Step 7: RCCL all-reduce sums partials across GPUs via Infinity Fabric
    //   Step 8: Residual addition on the full (identical) hidden state
    //
    // input:      [num_tokens, hidden_size]  - residual stream (BF16, identical on both GPUs)
    // positions:  [num_tokens]               - token position indices
    // output:     [num_tokens, hidden_size]  - updated residual (BF16, identical after all-reduce)
    // stream:     HIP stream for all operations
    // -----------------------------------------------------------------------
    void prefill(
        const __hip_bfloat16* input,
        const int* positions,
        const int32_t* slot_mapping,
        __hip_bfloat16* output,
        int num_tokens,
        hipStream_t stream)
    {
        if (num_tokens == 0) return;

        // Step 1: RMSNorm (operates on full hidden_size, identical on both GPUs)
        PROF("attn_rmsnorm", "norm", device_id_, stream);
        rmsnorm_forward(
            input, attn_norm_weight_, norm_buf_,
            num_tokens, hidden_size, rms_norm_eps, stream);
        PROF_END(device_id_, stream);

        // Step 2: Column-parallel QKV projection
        PROF("qkv_gemm", "gemm", device_id_, stream);
        cublas_->gemm_bf16_lt(
            norm_buf_,      // A: [num_tokens, hidden_size=2880]
            qkv_weight_,    // B: HF layout [tp_qkv_size=2560, hidden_size=2880], transposed by hipBLAS
            qkv_buf_,       // C: [num_tokens, tp_qkv_size=2560]
            num_tokens,     // M
            qkv_size,       // N = 2560 (TP-local)
            hidden_size,    // K = 2880
            stream,
            /*alpha=*/1.0f, /*beta=*/0.0f, /*bias=*/qkv_bias_, /*transB=*/true);
        PROF_END(device_id_, stream);

        // Step 3: Split QKV into separate Q, K, V tensors
        PROF("split_qkv", "misc", device_id_, stream);
        split_qkv(qkv_buf_, q_buf_, k_buf_, v_buf_, num_tokens, stream);
        PROF_END(device_id_, stream);

        // Step 4: Apply RoPE to local Q and K heads
        PROF("rope", "misc", device_id_, stream);
        rope_forward(
            q_buf_, k_buf_,
            q_rope_buf_, k_rope_buf_,
            cos_table_, sin_table_, cos_sin_table_, positions,
            num_tokens, num_q_heads, num_kv_heads, head_dim, stream);
        PROF_END(device_id_, stream);

        // Step 5a: Store rotated K and V into the paged KV cache
        PROF("kv_cache_update", "misc", device_id_, stream);
        kv_cache_update(
            k_cache_, v_cache_,
            k_rope_buf_, v_buf_,
            slot_mapping,
            num_tokens, num_kv_heads, head_dim, kv_block_size, stream);
        PROF_END(device_id_, stream);

        // Step 5b: Flash attention on local heads
        PROF("flash_attention", "attn", device_id_, stream);
        flash_attention_forward(
            q_rope_buf_, k_rope_buf_, v_buf_,
            attn_out_buf_, sink_values_,
            num_tokens, num_q_heads, num_kv_heads, head_dim,
            window_size_, stream);
        PROF_END(device_id_, stream);

        // Step 6: Row-parallel output projection
        PROF("o_proj_gemm", "gemm", device_id_, stream);
        cublas_->gemm_bf16_lt(
            attn_out_buf_,   // A: [num_tokens, tp_attn_out_size=2048]
            o_proj_weight_,  // B: HF layout [hidden_size=2880, tp_attn_out_size=2048], transposed by hipBLAS
            proj_out_buf_,   // C: [num_tokens, hidden_size=2880] PARTIAL
            num_tokens,      // M
            hidden_size,     // N = 2880
            attn_out_size,   // K = 2048 (TP-local)
            stream,
            /*alpha=*/1.0f, /*beta=*/0.0f,
            /*bias=*/(ModelConfig::tp_size == 1) ? o_proj_bias_ : nullptr,
            /*transB=*/true);
        PROF_END(device_id_, stream);

        // Step 7: TP all-reduce on dedicated comm stream for overlap
        if (ModelConfig::tp_size > 1) {
            PROF("attn_allreduce", "nccl", device_id_, stream);
            CUDA_CHECK(hipEventRecord(ar_compute_done_, stream));
            CUDA_CHECK(hipStreamWaitEvent(comm_stream_, ar_compute_done_, 0));

            NCCL_CHECK(ncclAllReduce(
                proj_out_buf_, proj_out_buf_,
                static_cast<size_t>(num_tokens) * hidden_size,
                ncclBfloat16, ncclSum,
                nccl_comm_, comm_stream_));

            CUDA_CHECK(hipEventRecord(ar_comm_done_, comm_stream_));
            CUDA_CHECK(hipStreamWaitEvent(stream, ar_comm_done_, 0));
            PROF_END(device_id_, stream);
        }

        // Step 8: Residual connection (both GPUs now hold identical result)
        PROF("residual_add", "misc", device_id_, stream);
        int total_elements = num_tokens * hidden_size;
        residual_add(input, proj_out_buf_, output, total_elements, stream);
        PROF_END(device_id_, stream);

    }

    // -----------------------------------------------------------------------
    // Decode path: processes one token per sequence with paged KV cache
    //
    // TP2: same partitioning as prefill. Each GPU handles local heads,
    // stores local KV cache, and all-reduces after O projection.
    //
    // input:        [num_seqs, hidden_size]  - residual stream (identical on both GPUs)
    // positions:    [num_seqs]               - current position for each seq
    // slot_mapping: [num_seqs]               - flat KV cache slot for each new token
    // block_table:  [num_seqs, max_num_blocks] - block indices for each seq
    // seq_lens:     [num_seqs]               - total sequence length for each seq
    // output:       [num_seqs, hidden_size]  - updated residual (identical after all-reduce)
    // stream:       HIP stream
    // -----------------------------------------------------------------------
    void decode(
        const __hip_bfloat16* input,
        const int* positions,
        const int32_t* slot_mapping,
        const int32_t* block_table,
        const int32_t* seq_lens,
        int max_num_blocks,
        __hip_bfloat16* output,
        int num_seqs,
        hipStream_t stream)
    {
        if (num_seqs == 0) return;

        // Step 1: RMSNorm
        rmsnorm_forward(
            input, attn_norm_weight_, norm_buf_,
            num_seqs, hidden_size, rms_norm_eps, stream);

        // Step 2: Column-parallel QKV projection (TP-local)
        cublas_->gemm_bf16_lt(
            norm_buf_, qkv_weight_, qkv_buf_,
            num_seqs, qkv_size, hidden_size, stream,
            /*alpha=*/1.0f, /*beta=*/0.0f, /*bias=*/qkv_bias_, /*transB=*/true);

        // Step 3: Split QKV (TP-local sizes)
        split_qkv(qkv_buf_, q_buf_, k_buf_, v_buf_, num_seqs, stream);

        // Step 4: RoPE on K only (Q rotation fused into paged attention)
        rope_forward(
            nullptr, k_buf_,
            nullptr, k_rope_buf_,
            cos_table_, sin_table_, cos_sin_table_, positions,
            num_seqs, num_q_heads, num_kv_heads, head_dim, stream);

        // Step 5: Update TP-local KV cache
        kv_cache_update(
            k_cache_, v_cache_,
            k_rope_buf_, v_buf_,
            slot_mapping,
            num_seqs, num_kv_heads, head_dim, kv_block_size, stream);

        // Step 6: Paged attention with fused RoPE Q + flash decoding (splitK)
        paged_attention_forward(
            q_buf_, k_cache_, v_cache_,
            attn_out_buf_,
            block_table, seq_lens, sink_values_,
            cos_sin_table_,
            num_seqs, num_q_heads, num_kv_heads, head_dim,
            kv_block_size, max_num_blocks,
            window_size_,
            partial_out_, partial_max_, partial_sum_, pa_max_splitK_,
            stream);

        // Step 7: Row-parallel O projection (produces partial hidden_size)
        cublas_->gemm_bf16_lt(
            attn_out_buf_, o_proj_weight_, proj_out_buf_,
            num_seqs, hidden_size, attn_out_size, stream,
            /*alpha=*/1.0f, /*beta=*/0.0f,
            /*bias=*/(ModelConfig::tp_size == 1) ? o_proj_bias_ : nullptr,
            /*transB=*/true);

        // Step 8: TP all-reduce (skip with TP1 -- single GPU, no-op)
        if (ModelConfig::tp_size > 1) {
            CUDA_CHECK(hipEventRecord(ar_compute_done_, stream));
            CUDA_CHECK(hipStreamWaitEvent(comm_stream_, ar_compute_done_, 0));

            NCCL_CHECK(ncclAllReduce(
                proj_out_buf_, proj_out_buf_,
                static_cast<size_t>(num_seqs) * hidden_size,
                ncclBfloat16, ncclSum,
                nccl_comm_, comm_stream_));

            CUDA_CHECK(hipEventRecord(ar_comm_done_, comm_stream_));
            CUDA_CHECK(hipStreamWaitEvent(stream, ar_comm_done_, 0));
        }

        // Step 9: Residual connection (both GPUs now hold identical result)
        int total_elements = num_seqs * hidden_size;
        residual_add(input, proj_out_buf_, output, total_elements, stream);
    }

    // -----------------------------------------------------------------------
    // Decode path WITHOUT residual add: same as decode() but stops after
    // the AllReduce on proj_out_buf_. The caller retrieves the delta via
    // get_proj_out() and fuses the residual add + RMSNorm externally.
    //
    // HIP Graph optimization: At B>=64, steps 1-7 (rmsnorm through O-proj)
    // are captured into a per-layer HIP graph on first invocation at each
    // batch size. Subsequent calls replay the graph (~3us) instead of
    // launching 7 individual kernels (~35us). The AllReduce remains eager
    // to avoid RCCL graph capture complexity.
    // -----------------------------------------------------------------------
    void decode_no_residual(
        const __hip_bfloat16* input,
        const int* positions,
        const int32_t* slot_mapping,
        const int32_t* block_table,
        const int32_t* seq_lens,
        int max_num_blocks,
        int num_seqs,
        hipStream_t stream)
    {
        if (num_seqs == 0) return;

        const int graph_key = (num_seqs << 16) ^ (max_num_blocks & 0xFFFF);
        auto it = decode_graph_cache_.find(graph_key);
        if (it != decode_graph_cache_.end()) {
            // Cached graph replay (~3us vs ~35us for 7 individual launches)
            CUDA_CHECK(hipGraphLaunch(it->second, stream));
        } else if (num_seqs >= MIN_GRAPH_BATCH) {
            // First call at this batch size: eager warmup (populates hipBLAS caches)
            decode_compute_steps(input, positions, slot_mapping, block_table,
                                 seq_lens, max_num_blocks, num_seqs, stream);
            CUDA_CHECK(hipStreamSynchronize(stream));  // one-time stall

            // Capture graph for future calls
            hipGraph_t graph;
            CUDA_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
            decode_compute_steps(input, positions, slot_mapping, block_table,
                                 seq_lens, max_num_blocks, num_seqs, stream);
            CUDA_CHECK(hipStreamEndCapture(stream, &graph));

            hipGraphExec_t exec;
            CUDA_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            hipGraphDestroy(graph);
            decode_graph_cache_[graph_key] = exec;
            // Note: first call's output already produced by eager warmup above
        } else {
            // Small batch: eager (splitK may vary)
            decode_compute_steps(input, positions, slot_mapping, block_table,
                                 seq_lens, max_num_blocks, num_seqs, stream);
        }

        // AllReduce always runs eagerly after graph/eager compute
        do_allreduce(num_seqs, stream);
    }

    __hip_bfloat16* get_proj_out() const { return proj_out_buf_; }

private:
    // Steps 1-7 of decode: rmsnorm, QKV proj, split, RoPE K, KV update,
    // paged attention, O proj. Graph-capture-safe at B>=64 (fixed kernel count).
    void decode_compute_steps(
        const __hip_bfloat16* input,
        const int* positions,
        const int32_t* slot_mapping,
        const int32_t* block_table,
        const int32_t* seq_lens,
        int max_num_blocks,
        int num_seqs,
        hipStream_t stream)
    {
        rmsnorm_forward(input, attn_norm_weight_, norm_buf_,
                        num_seqs, hidden_size, rms_norm_eps, stream);
        if (num_seqs == 1) {
            // Decode B=1: custom GEMV — eliminates ~1.5ms hipBLASLt overhead per call
            gemv_bf16_forward(norm_buf_, qkv_weight_, qkv_buf_,
                              qkv_size, hidden_size, stream, qkv_bias_);
        } else {
            cublas_->gemm_bf16_lt(norm_buf_, qkv_weight_, qkv_buf_,
                                  num_seqs, qkv_size, hidden_size, stream,
                                  /*alpha=*/1.0f, /*beta=*/0.0f, /*bias=*/qkv_bias_,
                                  /*transB=*/true);
        }
        if (num_seqs == 1) {
            // Fused split_qkv + RoPE_K + KV cache update — 1 kernel instead of 3
            fused_split_rope_kv_kernel<<<1, 256, 0, stream>>>(
                qkv_buf_, q_buf_, k_cache_, v_cache_,
                cos_sin_table_, positions, slot_mapping, 1);
        } else {
            split_qkv(qkv_buf_, q_buf_, k_buf_, v_buf_, num_seqs, stream);
            rope_forward(nullptr, k_buf_, nullptr, k_rope_buf_,
                         cos_table_, sin_table_, cos_sin_table_, positions,
                         num_seqs, num_q_heads, num_kv_heads, head_dim, stream);
            kv_cache_update(k_cache_, v_cache_, k_rope_buf_, v_buf_,
                            slot_mapping, num_seqs, num_kv_heads, head_dim,
                            kv_block_size, stream);
        }
        paged_attention_forward(q_buf_, k_cache_, v_cache_, attn_out_buf_,
                                block_table, seq_lens, sink_values_,
                                cos_sin_table_, num_seqs, num_q_heads,
                                num_kv_heads, head_dim, kv_block_size,
                                max_num_blocks, window_size_,
                                partial_out_, partial_max_, partial_sum_,
                                pa_max_splitK_, stream);
        if (num_seqs == 1) {
            const __hip_bfloat16* o_bias = (ModelConfig::tp_size == 1) ? o_proj_bias_ : nullptr;
            gemv_bf16_forward(attn_out_buf_, o_proj_weight_, proj_out_buf_,
                              hidden_size, attn_out_size, stream, o_bias);
        } else {
            cublas_->gemm_bf16_lt(attn_out_buf_, o_proj_weight_, proj_out_buf_,
                                  num_seqs, hidden_size, attn_out_size, stream,
                                  /*alpha=*/1.0f, /*beta=*/0.0f,
                                  /*bias=*/(ModelConfig::tp_size == 1) ? o_proj_bias_ : nullptr,
                                  /*transB=*/true);
        }
    }

    // Steps 8-10: event sync + RCCL AllReduce on comm_stream_ + event sync back
    void do_allreduce(int num_seqs, hipStream_t stream) {
        if (ModelConfig::tp_size <= 1) return;  // TP1: no all-reduce needed
        CUDA_CHECK(hipEventRecord(ar_compute_done_, stream));
        CUDA_CHECK(hipStreamWaitEvent(comm_stream_, ar_compute_done_, 0));
        NCCL_CHECK(ncclAllReduce(proj_out_buf_, proj_out_buf_,
                                 static_cast<size_t>(num_seqs) * hidden_size,
                                 ncclBfloat16, ncclSum, nccl_comm_, comm_stream_));
        CUDA_CHECK(hipEventRecord(ar_comm_done_, comm_stream_));
        CUDA_CHECK(hipStreamWaitEvent(stream, ar_comm_done_, 0));
    }

    void split_qkv(
        const __hip_bfloat16* qkv,
        __hip_bfloat16* q_out,
        __hip_bfloat16* k_out,
        __hip_bfloat16* v_out,
        int num_tokens,
        hipStream_t stream)
    {
        if (num_tokens == 0) return;
        // Use 256 threads per block; q_dim=2048 means ~8 iterations per thread (TP2)
        int threads = 256;
        split_qkv_kernel<<<num_tokens, threads, 0, stream>>>(
            qkv, q_out, k_out, v_out, num_tokens);
        CUDA_CHECK(hipGetLastError());
    }
};

// ---------------------------------------------------------------------------
// Wrapper functions for cross-compilation-unit access from transformer.cu
// These allow the Transformer class to call AttentionLayer methods without
// needing the full class definition in its compilation unit.
// ---------------------------------------------------------------------------

void attention_layer_prefill(
    AttentionLayer* layer,
    const __hip_bfloat16* input,
    const int* positions,
    const int32_t* slot_mapping,
    __hip_bfloat16* output,
    int num_tokens,
    hipStream_t stream)
{
    layer->prefill(input, positions, slot_mapping, output, num_tokens, stream);
}

void attention_layer_decode(
    AttentionLayer* layer,
    const __hip_bfloat16* input,
    const int* positions,
    const int32_t* slot_mapping,
    const int32_t* block_table,
    const int32_t* seq_lens,
    int max_num_blocks,
    __hip_bfloat16* output,
    int num_seqs,
    hipStream_t stream)
{
    layer->decode(input, positions, slot_mapping, block_table, seq_lens,
                  max_num_blocks, output, num_seqs, stream);
}

void attention_layer_decode_no_residual(
    AttentionLayer* layer,
    const __hip_bfloat16* input,
    const int* positions,
    const int32_t* slot_mapping,
    const int32_t* block_table,
    const int32_t* seq_lens,
    int max_num_blocks,
    int num_seqs,
    hipStream_t stream)
{
    layer->decode_no_residual(input, positions, slot_mapping, block_table,
                               seq_lens, max_num_blocks, num_seqs, stream);
}

__hip_bfloat16* attention_layer_get_proj_out(AttentionLayer* layer) {
    return layer->get_proj_out();
}

AttentionLayer* attention_layer_create() { return new AttentionLayer(); }
void attention_layer_destroy(AttentionLayer* l) { if (l) delete l; }

void attention_layer_init(
    AttentionLayer* layer,
    int layer_id, int device_id, int rank, ncclComm_t nccl_comm,
    const CublasHandle* cublas,
    const __hip_bfloat16* qkv_weight, const __hip_bfloat16* qkv_bias,
    const __hip_bfloat16* o_proj_weight, const __hip_bfloat16* o_proj_bias,
    const __hip_bfloat16* attn_norm_weight,
    const float* cos_table, const float* sin_table, const float2* cos_sin_table,
    const float* sink_values,
    __hip_bfloat16* k_cache, __hip_bfloat16* v_cache, int max_num_blocks,
    __hip_bfloat16* norm_buf, __hip_bfloat16* qkv_buf,
    __hip_bfloat16* q_buf, __hip_bfloat16* k_buf, __hip_bfloat16* v_buf,
    __hip_bfloat16* q_rope_buf, __hip_bfloat16* k_rope_buf,
    __hip_bfloat16* attn_out_buf, __hip_bfloat16* proj_out_buf) {
    layer->init(layer_id, device_id, rank, nccl_comm, cublas,
                qkv_weight, qkv_bias, o_proj_weight, o_proj_bias,
                attn_norm_weight,
                cos_table, sin_table, cos_sin_table, sink_values,
                k_cache, v_cache, max_num_blocks,
                norm_buf, qkv_buf, q_buf, k_buf, v_buf,
                q_rope_buf, k_rope_buf, attn_out_buf, proj_out_buf);
}

} // namespace gptoss
