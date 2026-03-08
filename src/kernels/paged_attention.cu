// paged_attention.cu -- Decode-phase paged KV cache attention
// GPT-OSS-120B inference engine
//
// HIP port for AMD MI300X (CDNA3, wavefront64)
//
// MAJOR rewrite from CUDA version:
//   - Removed #include <cuda_pipeline.h>
//   - PA_WARP_SIZE = 32 -> 64
//   - PA_NUM_WARPS = 4 -> 2 (keep 128 threads = 2 wavefronts)
//   - PA_D_PER_THREAD = 64/64 = 1 (each thread handles 1 d-element)
//   - Replaced __pipeline_memcpy_async with synchronous int4 copies
//   - Replaced __pipeline_commit() with no-op
//   - Replaced __pipeline_wait_prior(1) with __syncwarp()
//   - Removed L2 prefetch (pa_prefetch_l2 -> no-op)
//   - All warp shuffles: removed sync mask (use hip_compat.h wrappers)
//   - d_idx = lane_id (covers all 64 dims with 64 lanes, no stride)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cfloat>
#include <cstdint>

#include "config.h"
#include "hip_compat.h"
#include "tensor.h"
#include "cuda_utils.h"

namespace gptoss {

// ---------------------------------------------------------------------------
// Compile-time constants -- adjusted for wavefront64
// ---------------------------------------------------------------------------
static constexpr int PA_WARP_SIZE    = 64;
static constexpr int PA_NUM_WARPS    = 2;
static constexpr int PA_THREADS      = PA_NUM_WARPS * PA_WARP_SIZE; // 128
static constexpr int PA_HEAD_DIM     = 64;
static constexpr int PA_KV_BLOCK     = 16;
static constexpr int PA_D_PER_THREAD = PA_HEAD_DIM / PA_WARP_SIZE; // 1
static constexpr int PA_GQA_RATIO    = ModelConfig::gqa_ratio; // 8
static constexpr int PA_HEAD_GROUP   = 2;
static constexpr int PA_NUM_BUFS     = 2;

// Per-warp smem tile sizes
static constexpr int PA_KV_TILE_ELEMS = PA_KV_BLOCK * PA_HEAD_DIM; // 1024 bf16
static constexpr int PA_KV_TILE_BYTES = PA_KV_TILE_ELEMS * sizeof(__hip_bfloat16); // 2048 bytes

// Max block table entries in smem
static constexpr int PA_MAX_BT_SMEM = 1024;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float pa_bf16_to_fp32(__hip_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __hip_bfloat16 pa_fp32_to_bf16(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float pa_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = PA_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// L2 prefetch: no-op on MI300X
__device__ __forceinline__ void pa_prefetch_l2(const void*) {}

// Synchronous copy of one KV tile into smem (replaces cp.async)
__device__ __forceinline__ void pa_sync_copy_tile(
    __hip_bfloat16* __restrict__ dst,
    const __hip_bfloat16* __restrict__ src,
    int lane_id)
{
    constexpr int I4_COUNT = PA_KV_TILE_BYTES / sizeof(int4); // 128
    static_assert(I4_COUNT % PA_WARP_SIZE == 0, "must be divisible");

    const int4* src4 = reinterpret_cast<const int4*>(src);
    int4* dst4 = reinterpret_cast<int4*>(dst);

    #pragma unroll
    for (int i = lane_id; i < I4_COUNT; i += PA_WARP_SIZE) {
        dst4[i] = src4[i];
    }
}

// ---------------------------------------------------------------------------
// Paged Attention kernel
//
// Template parameter:
//   IsPartial=false: standard full-attention (splitK=1 path)
//   IsPartial=true:  partial attention for flash decoding (splitK>1)
//
// Grid:
//   Full:    (num_kv_heads, num_seqs, 1)
//   Partial: (num_kv_heads, num_seqs, splitK)
// Block: (128, 1, 1) -- 2 wavefronts
//
// Shared memory layout (~24 KB with 2 warps):
//   s_V_all: [NUM_WARPS, NUM_BUFS, KV_BLOCK, HEAD_DIM] BF16 = 8 KB
//   s_K_all: [NUM_WARPS, NUM_BUFS, KV_BLOCK, HEAD_DIM] BF16 = 8 KB
//   Cross-warp reduction:
//     s_warp_max: [NUM_WARPS, GQA_RATIO]                = 64 bytes
//     s_warp_sum: [NUM_WARPS, GQA_RATIO]                = 64 bytes
//     s_warp_out: [NUM_WARPS, GQA_RATIO, HEAD_DIM]      = 4096 bytes
//   Block table: up to 1024 * 4 = 4 KB
//   Total: ~24 KB
// ---------------------------------------------------------------------------
template<bool IsPartial>
__global__ void __launch_bounds__(PA_THREADS, 4)
paged_attention_kernel_impl(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ k_cache,
    const __hip_bfloat16* __restrict__ v_cache,
    __hip_bfloat16* __restrict__ output,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    const float* __restrict__ sink_values,
    const float2* __restrict__ cos_sin_table,
    const int num_seqs,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_num_blocks,
    const int window_size,
    float* __restrict__ partial_out_buf,
    float* __restrict__ partial_max_buf,
    float* __restrict__ partial_sum_buf,
    const int splitK)
{
    const int kv_head = blockIdx.x;
    const int seq_id  = blockIdx.y;
    const int split_id = IsPartial ? static_cast<int>(blockIdx.z) : 0;

    if (seq_id >= num_seqs) return;

    const int seq_len = seq_lens[seq_id];
    if (seq_len <= 0) return;

    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int q_head_start = kv_head * gqa_ratio;

    const int tid     = threadIdx.x;
    const int warp_id = tid / PA_WARP_SIZE;
    const int lane_id = tid % PA_WARP_SIZE;

    const int query_pos = seq_len - 1;

    // -----------------------------------------------------------------------
    // Load query vectors with fused RoPE rotation (half-split convention)
    //
    // With PA_D_PER_THREAD=1 and PA_WARP_SIZE=64:
    //   d_idx = lane_id covers all 64 dims
    //   half_dim = 32, lane_id < 32 maps to first half, >= 32 to second half
    //   RoPE pair: x[lane_id] and x[lane_id XOR 32] (first/second half swap)
    // -----------------------------------------------------------------------
    float q_reg[PA_GQA_RATIO][PA_D_PER_THREAD];
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    const int half_dim = head_dim / 2;

    // For fused RoPE: each lane handles d_idx = lane_id
    // The RoPE pair is (d, d+half_dim) or (d-half_dim, d)
    // lane_id < half_dim: d_idx=lane_id, pair is at lane_id+half_dim
    // lane_id >= half_dim: d_idx=lane_id, pair is at lane_id-half_dim
    float2 cs = make_float2(1.0f, 0.0f);
    if (cos_sin_table != nullptr) {
        int rope_d = lane_id < half_dim ? lane_id : lane_id - half_dim;
        cs = cos_sin_table[query_pos * half_dim + rope_d];
    }

    #pragma unroll
    for (int h = 0; h < PA_GQA_RATIO; h++) {
        int q_head = q_head_start + h;
        int base = seq_id * num_q_heads * head_dim + q_head * head_dim;

        if (cos_sin_table != nullptr) {
            if (lane_id < half_dim) {
                // First half: x0 = q[d], x1 = q[d + half_dim]
                float x0 = pa_bf16_to_fp32(query[base + lane_id]);
                float x1 = pa_bf16_to_fp32(query[base + lane_id + half_dim]);
                q_reg[h][0] = (x0 * cs.x - x1 * cs.y) * scale;
            } else {
                // Second half: this is d_idx = lane_id >= half_dim
                // x0 = q[d - half_dim], x1 = q[d]
                float x0 = pa_bf16_to_fp32(query[base + lane_id - half_dim]);
                float x1 = pa_bf16_to_fp32(query[base + lane_id]);
                q_reg[h][0] = (x0 * cs.y + x1 * cs.x) * scale;
            }
        } else {
            q_reg[h][0] = pa_bf16_to_fp32(query[base + lane_id]) * scale;
        }
    }

    // Load per-head sink biases
    float sink_bias[PA_GQA_RATIO];
    if (!IsPartial) {
        #pragma unroll
        for (int h = 0; h < PA_GQA_RATIO; h++) {
            sink_bias[h] = (sink_values != nullptr) ? sink_values[q_head_start + h] : 0.0f;
        }
    }

    // -----------------------------------------------------------------------
    // Determine KV token range
    // -----------------------------------------------------------------------
    int kv_start_token = 0;
    if (window_size > 0) {
        kv_start_token = max(0, seq_len - window_size);
    }
    const int kv_end_token = seq_len;

    int start_block_idx = kv_start_token / block_size;
    const int end_block_idx = cdiv(kv_end_token, block_size);
    int num_blocks_to_process = end_block_idx - start_block_idx;

    if (IsPartial) {
        int blocks_per_split = cdiv(num_blocks_to_process, splitK);
        int split_start = split_id * blocks_per_split;
        int split_end = min(split_start + blocks_per_split, num_blocks_to_process);
        if (split_start >= num_blocks_to_process) {
            num_blocks_to_process = 0;
        } else {
            start_block_idx += split_start;
            num_blocks_to_process = split_end - split_start;
        }
    }

    const int kv_block_stride = num_kv_heads * block_size * head_dim;
    const int kv_head_stride  = block_size * head_dim;

    // -----------------------------------------------------------------------
    // Shared memory layout
    // -----------------------------------------------------------------------
    extern __shared__ char pa_smem_raw[];

    __hip_bfloat16* s_V_all = reinterpret_cast<__hip_bfloat16*>(pa_smem_raw);
    constexpr int v_total_bytes = PA_NUM_WARPS * PA_NUM_BUFS * PA_KV_TILE_BYTES;
    __hip_bfloat16* s_K_all = reinterpret_cast<__hip_bfloat16*>(pa_smem_raw + v_total_bytes);
    constexpr int vk_total_bytes = 2 * v_total_bytes;

    float* s_warp_max = reinterpret_cast<float*>(pa_smem_raw + vk_total_bytes);
    float* s_warp_sum = s_warp_max + PA_NUM_WARPS * PA_GQA_RATIO;
    float* s_warp_out = s_warp_sum + PA_NUM_WARPS * PA_GQA_RATIO;

    constexpr int reduce_total_bytes =
        PA_NUM_WARPS * PA_GQA_RATIO * sizeof(float) * 2 +
        PA_NUM_WARPS * PA_GQA_RATIO * PA_HEAD_DIM * sizeof(float);
    int32_t* s_block_table = reinterpret_cast<int32_t*>(
        pa_smem_raw + vk_total_bytes + reduce_total_bytes);

    // -----------------------------------------------------------------------
    // Preload block table
    // -----------------------------------------------------------------------
    const int32_t* my_block_table_gmem = block_table + seq_id * max_num_blocks;
    const int bt_entries = min(max_num_blocks, PA_MAX_BT_SMEM);
    for (int i = tid; i < bt_entries; i += PA_THREADS) {
        s_block_table[i] = my_block_table_gmem[i];
    }
    __syncthreads();

    // Per-warp buffer offset helpers
    auto get_V = [&](int buf) -> __hip_bfloat16* {
        return s_V_all + (warp_id * PA_NUM_BUFS + buf) * PA_KV_TILE_ELEMS;
    };
    auto get_K = [&](int buf) -> __hip_bfloat16* {
        return s_K_all + (warp_id * PA_NUM_BUFS + buf) * PA_KV_TILE_ELEMS;
    };

    // -----------------------------------------------------------------------
    // Per-warp online softmax state
    // -----------------------------------------------------------------------
    float warp_max_val[PA_GQA_RATIO];
    float warp_sum_val[PA_GQA_RATIO];
    float warp_acc[PA_GQA_RATIO][PA_D_PER_THREAD];

    #pragma unroll
    for (int h = 0; h < PA_GQA_RATIO; h++) {
        warp_max_val[h] = -FLT_MAX;
        warp_sum_val[h] = 0.0f;
        #pragma unroll
        for (int dd = 0; dd < PA_D_PER_THREAD; dd++) {
            warp_acc[h][dd] = 0.0f;
        }
    }

    // -----------------------------------------------------------------------
    // Count blocks for this warp
    // -----------------------------------------------------------------------
    int warp_block_count = 0;
    for (int bi = warp_id; bi < num_blocks_to_process; bi += PA_NUM_WARPS) {
        warp_block_count++;
    }

    if (warp_block_count == 0) goto reduction;

    // -----------------------------------------------------------------------
    // Main loop -- synchronous double-buffered KV loads
    // -----------------------------------------------------------------------
    {
        // Prologue: load first block into buffer 0
        int first_bi = warp_id;
        int first_logical = start_block_idx + first_bi;
        int first_physical = (first_logical < bt_entries)
            ? s_block_table[first_logical]
            : my_block_table_gmem[first_logical];

        const __hip_bfloat16* first_k_ptr =
            k_cache + first_physical * kv_block_stride + kv_head * kv_head_stride;
        const __hip_bfloat16* first_v_ptr =
            v_cache + first_physical * kv_block_stride + kv_head * kv_head_stride;

        pa_sync_copy_tile(get_K(0), first_k_ptr, lane_id);
        pa_sync_copy_tile(get_V(0), first_v_ptr, lane_id);
        __syncwarp();

        int cur_buf = 0;

        for (int iter = 0; iter < warp_block_count; iter++) {
            int bi = warp_id + iter * PA_NUM_WARPS;
            int block_token_start = (start_block_idx + bi) * block_size;

            // Prefetch next block into alternate buffer
            int next_iter = iter + 1;
            if (next_iter < warp_block_count) {
                int next_bi = warp_id + next_iter * PA_NUM_WARPS;
                int next_logical = start_block_idx + next_bi;
                int next_physical = (next_logical < bt_entries)
                    ? s_block_table[next_logical]
                    : my_block_table_gmem[next_logical];

                const __hip_bfloat16* next_k_ptr =
                    k_cache + next_physical * kv_block_stride + kv_head * kv_head_stride;
                const __hip_bfloat16* next_v_ptr =
                    v_cache + next_physical * kv_block_stride + kv_head * kv_head_stride;

                int next_buf = 1 - cur_buf;
                pa_sync_copy_tile(get_K(next_buf), next_k_ptr, lane_id);
                pa_sync_copy_tile(get_V(next_buf), next_v_ptr, lane_id);
            }

            __syncwarp();

            __hip_bfloat16* my_s_K = get_K(cur_buf);
            __hip_bfloat16* my_s_V = get_V(cur_buf);

            const bool block_all_valid =
                (block_token_start >= kv_start_token) &&
                (block_token_start + PA_KV_BLOCK - 1 <= query_pos);

            // Head-group-of-2 batch QK + fused softmax+PV
            #pragma unroll
            for (int hg = 0; hg < PA_GQA_RATIO; hg += PA_HEAD_GROUP) {
                float block_scores[PA_HEAD_GROUP][PA_KV_BLOCK];

                if (block_all_valid) {
                    #pragma unroll
                    for (int t = 0; t < PA_KV_BLOCK; t++) {
                        // PA_D_PER_THREAD=1: d_idx = lane_id
                        float k_local = pa_bf16_to_fp32(my_s_K[t * PA_HEAD_DIM + lane_id]);

                        #pragma unroll
                        for (int g = 0; g < PA_HEAD_GROUP; g++) {
                            float dot = q_reg[hg + g][0] * k_local;
                            block_scores[g][t] = pa_warp_reduce_sum(dot);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int t = 0; t < PA_KV_BLOCK; t++) {
                        int token_pos = block_token_start + t;
                        bool valid = (token_pos < kv_end_token)
                                  && (token_pos >= kv_start_token)
                                  && (token_pos <= query_pos);
                        if (!valid) {
                            #pragma unroll
                            for (int g = 0; g < PA_HEAD_GROUP; g++) {
                                block_scores[g][t] = -FLT_MAX;
                            }
                            continue;
                        }

                        float k_local = pa_bf16_to_fp32(my_s_K[t * PA_HEAD_DIM + lane_id]);

                        #pragma unroll
                        for (int g = 0; g < PA_HEAD_GROUP; g++) {
                            float dot = q_reg[hg + g][0] * k_local;
                            block_scores[g][t] = pa_warp_reduce_sum(dot);
                        }
                    }
                }

                // Phase 2: Block-max batched softmax + PV accumulation
                #pragma unroll
                for (int g = 0; g < PA_HEAD_GROUP; g++) {
                    int h = hg + g;

                    float block_max = -FLT_MAX;
                    #pragma unroll
                    for (int t = 0; t < PA_KV_BLOCK; t++) {
                        if (block_scores[g][t] > block_max)
                            block_max = block_scores[g][t];
                    }

                    if (block_max <= -FLT_MAX) continue;

                    float exp_val[PA_KV_BLOCK];
                    float block_sum = 0.0f;
                    #pragma unroll
                    for (int t = 0; t < PA_KV_BLOCK; t++) {
                        if (block_scores[g][t] > -FLT_MAX) {
                            exp_val[t] = expf(block_scores[g][t] - block_max);
                            block_sum += exp_val[t];
                        } else {
                            exp_val[t] = 0.0f;
                        }
                    }

                    // PV accumulation: d_idx = lane_id
                    float block_acc[PA_D_PER_THREAD];
                    #pragma unroll
                    for (int dd = 0; dd < PA_D_PER_THREAD; dd++)
                        block_acc[dd] = 0.0f;

                    #pragma unroll
                    for (int t = 0; t < PA_KV_BLOCK; t++) {
                        if (exp_val[t] > 0.0f) {
                            #pragma unroll
                            for (int dd = 0; dd < PA_D_PER_THREAD; dd++) {
                                int d_idx = lane_id + dd * PA_WARP_SIZE;
                                float v_val = pa_bf16_to_fp32(my_s_V[t * PA_HEAD_DIM + d_idx]);
                                block_acc[dd] += exp_val[t] * v_val;
                            }
                        }
                    }

                    // Merge block result with running warp state
                    float m_prev = warp_max_val[h];
                    if (block_max > m_prev) {
                        float correction = (m_prev > -FLT_MAX) ? expf(m_prev - block_max) : 0.0f;
                        warp_sum_val[h] = warp_sum_val[h] * correction + block_sum;
                        #pragma unroll
                        for (int dd = 0; dd < PA_D_PER_THREAD; dd++)
                            warp_acc[h][dd] = warp_acc[h][dd] * correction + block_acc[dd];
                        warp_max_val[h] = block_max;
                    } else {
                        float correction = expf(block_max - m_prev);
                        warp_sum_val[h] += block_sum * correction;
                        #pragma unroll
                        for (int dd = 0; dd < PA_D_PER_THREAD; dd++)
                            warp_acc[h][dd] += block_acc[dd] * correction;
                    }
                }
            }

            cur_buf = 1 - cur_buf;
        }
    }

reduction:
    // -----------------------------------------------------------------------
    // Add learned sink bias (full-mode only)
    // -----------------------------------------------------------------------
    if (!IsPartial) {
        #pragma unroll
        for (int h = 0; h < PA_GQA_RATIO; h++) {
            if (sink_bias[h] != 0.0f && warp_max_val[h] > -FLT_MAX) {
                warp_sum_val[h] += expf(sink_bias[h] - warp_max_val[h]);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Cross-warp reduction
    // -----------------------------------------------------------------------
    #pragma unroll
    for (int h = 0; h < PA_GQA_RATIO; h++) {
        #pragma unroll
        for (int dd = 0; dd < PA_D_PER_THREAD; dd++) {
            int d_idx = lane_id + dd * PA_WARP_SIZE;
            s_warp_out[(warp_id * PA_GQA_RATIO + h) * PA_HEAD_DIM + d_idx] = warp_acc[h][dd];
        }
        if (lane_id == 0) {
            s_warp_max[warp_id * PA_GQA_RATIO + h] = warp_max_val[h];
            s_warp_sum[warp_id * PA_GQA_RATIO + h] = warp_sum_val[h];
        }
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Warp 0 merges all partial results
    // -----------------------------------------------------------------------
    if (warp_id == 0) {
        #pragma unroll
        for (int h = 0; h < PA_GQA_RATIO; h++) {
            float global_max = -FLT_MAX;
            for (int w = 0; w < PA_NUM_WARPS; w++) {
                global_max = fmaxf(global_max, s_warp_max[w * PA_GQA_RATIO + h]);
            }

            float global_sum = 0.0f;
            float corrections[PA_NUM_WARPS];
            for (int w = 0; w < PA_NUM_WARPS; w++) {
                corrections[w] = (s_warp_max[w * PA_GQA_RATIO + h] > -FLT_MAX) ?
                    expf(s_warp_max[w * PA_GQA_RATIO + h] - global_max) : 0.0f;
                global_sum += s_warp_sum[w * PA_GQA_RATIO + h] * corrections[w];
            }

            if (IsPartial) {
                int partial_idx = ((seq_id * num_kv_heads + kv_head) * splitK + split_id)
                                  * PA_GQA_RATIO + h;
                if (lane_id == 0) {
                    partial_max_buf[partial_idx] = global_max;
                    partial_sum_buf[partial_idx] = global_sum;
                }

                #pragma unroll
                for (int dd = 0; dd < PA_D_PER_THREAD; dd++) {
                    int d_idx = lane_id + dd * PA_WARP_SIZE;
                    float merged = 0.0f;
                    for (int w = 0; w < PA_NUM_WARPS; w++) {
                        merged += s_warp_out[(w * PA_GQA_RATIO + h) * PA_HEAD_DIM + d_idx]
                                  * corrections[w];
                    }
                    partial_out_buf[partial_idx * PA_HEAD_DIM + d_idx] = merged;
                }
            } else {
                float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
                int q_head = q_head_start + h;

                #pragma unroll
                for (int dd = 0; dd < PA_D_PER_THREAD; dd++) {
                    int d_idx = lane_id + dd * PA_WARP_SIZE;
                    float merged = 0.0f;
                    for (int w = 0; w < PA_NUM_WARPS; w++) {
                        merged += s_warp_out[(w * PA_GQA_RATIO + h) * PA_HEAD_DIM + d_idx]
                                  * corrections[w];
                    }
                    merged *= inv_sum;

                    output[seq_id * num_q_heads * head_dim + q_head * head_dim + d_idx] =
                        pa_fp32_to_bf16(merged);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reduce kernel: merges partial results from splitK partial blocks
//
// Grid:  (num_q_heads, num_seqs)
// Block: (64, 1, 1) -- 1 wavefront
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(PA_WARP_SIZE)
paged_attention_reduce_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const float* __restrict__ sink_values,
    __hip_bfloat16* __restrict__ output,
    const int num_seqs,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int splitK)
{
    const int q_head = blockIdx.x;
    const int seq_id = blockIdx.y;

    if (seq_id >= num_seqs || q_head >= num_q_heads) return;

    const int lane_id = threadIdx.x;
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head = q_head / gqa_ratio;
    const int h = q_head % gqa_ratio;

    // Find global max across all splits
    float global_max = -FLT_MAX;
    for (int s = 0; s < splitK; s++) {
        int idx = ((seq_id * num_kv_heads + kv_head) * splitK + s) * gqa_ratio + h;
        global_max = fmaxf(global_max, partial_max[idx]);
    }

    // Merge with log-sum-exp correction
    // PA_D_PER_THREAD=1: only acc[0]
    float global_sum = 0.0f;
    float acc[PA_D_PER_THREAD] = {0.0f};

    for (int s = 0; s < splitK; s++) {
        int idx = ((seq_id * num_kv_heads + kv_head) * splitK + s) * gqa_ratio + h;
        float correction = (partial_max[idx] > -FLT_MAX) ?
            expf(partial_max[idx] - global_max) : 0.0f;
        global_sum += partial_sum[idx] * correction;

        #pragma unroll
        for (int dd = 0; dd < PA_D_PER_THREAD; dd++) {
            int d_idx = lane_id + dd * PA_WARP_SIZE;
            acc[dd] += partial_out[idx * head_dim + d_idx] * correction;
        }
    }

    // Apply sink bias (deferred from partial kernel)
    if (sink_values != nullptr && global_max > -FLT_MAX) {
        float bias = sink_values[q_head];
        if (bias != 0.0f) {
            global_sum += expf(bias - global_max);
        }
    }

    // Normalize and write BF16 output
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    #pragma unroll
    for (int dd = 0; dd < PA_D_PER_THREAD; dd++) {
        int d_idx = lane_id + dd * PA_WARP_SIZE;
        output[seq_id * num_q_heads * head_dim + q_head * head_dim + d_idx] =
            pa_fp32_to_bf16(acc[dd] * inv_sum);
    }
}

// ---------------------------------------------------------------------------
// Host-side launch function
// ---------------------------------------------------------------------------
void paged_attention_forward(
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
    hipStream_t stream)
{
    if (num_seqs == 0) return;

    const int gqa_ratio = num_q_heads / num_kv_heads;

    // Shared memory
    const size_t vk_smem = 2 * PA_NUM_WARPS * PA_NUM_BUFS * PA_KV_BLOCK * PA_HEAD_DIM * sizeof(__hip_bfloat16);
    const size_t reduce_smem =
        PA_NUM_WARPS * gqa_ratio * sizeof(float) +
        PA_NUM_WARPS * gqa_ratio * sizeof(float) +
        PA_NUM_WARPS * gqa_ratio * PA_HEAD_DIM * sizeof(float);
    const int bt_entries = (max_num_blocks < PA_MAX_BT_SMEM) ? max_num_blocks : PA_MAX_BT_SMEM;
    const size_t bt_smem = bt_entries * sizeof(int32_t);
    const size_t smem_bytes = vk_smem + reduce_smem + bt_smem;

    // Adaptive splitK selection — target ~304 blocks for MI300X (304 CUs)
    int kv_blocks;
    if (window_size > 0) {
        kv_blocks = cdiv(window_size, block_size);
    } else {
        kv_blocks = max_num_blocks;
    }

    int splitK = 1;
    if (partial_out != nullptr && max_splitK > 1 && kv_blocks > 1) {
        int total_base = num_kv_heads * num_seqs;
        // Target enough blocks to fill MI300X CUs (304)
        constexpr int TARGET_BLOCKS = 304;
        splitK = (TARGET_BLOCKS + total_base - 1) / total_base;
        splitK = min(splitK, kv_blocks);     // can't split more than available blocks
        splitK = min(splitK, max_splitK);    // cap at caller's limit
        if (splitK < 2) splitK = 1;
    }

    if (splitK <= 1) {
        dim3 grid(num_kv_heads, num_seqs);
        dim3 block(PA_THREADS);

        paged_attention_kernel_impl<false><<<grid, block, smem_bytes, stream>>>(
            query, k_cache, v_cache, output,
            block_table, seq_lens, sink_values, cos_sin_table,
            num_seqs, num_q_heads, num_kv_heads, head_dim,
            block_size, max_num_blocks, window_size,
            nullptr, nullptr, nullptr, 1);
    } else {
        dim3 partial_grid(num_kv_heads, num_seqs, splitK);
        dim3 partial_block(PA_THREADS);

        paged_attention_kernel_impl<true><<<partial_grid, partial_block, smem_bytes, stream>>>(
            query, k_cache, v_cache, output,
            block_table, seq_lens, nullptr, cos_sin_table,
            num_seqs, num_q_heads, num_kv_heads, head_dim,
            block_size, max_num_blocks, window_size,
            partial_out, partial_max, partial_sum, splitK);

        dim3 reduce_grid(num_q_heads, num_seqs);
        dim3 reduce_block(PA_WARP_SIZE);

        paged_attention_reduce_kernel<<<reduce_grid, reduce_block, 0, stream>>>(
            partial_out, partial_max, partial_sum,
            sink_values, output,
            num_seqs, num_q_heads, num_kv_heads, head_dim, splitK);
    }

    CUDA_CHECK(hipGetLastError());
}

} // namespace gptoss
