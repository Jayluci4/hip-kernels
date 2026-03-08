// flash_attention.cu -- Flash Attention v2 for prefill with sliding window + learned sinks
// GPT-OSS-120B inference engine
//
// HIP port for AMD MI300X (CDNA3, wavefront64)
//
// MAJOR rewrite: replaced all nvcuda::wmma Tensor Core code with scalar FP32 dot products.
// Key parameter changes for wavefront64:
//   - FA_WARP_SIZE = 32 -> 64
//   - FA_NUM_WARPS = 4 -> 2 (keep 128 threads = 2 wavefronts)
//   - FA_ROWS_PER_WARP = 64/2 = 32 rows per warp
//   - FA_D_PER_THREAD = 64/64 = 1 (each thread handles 1 d-element)
//
// QK^T computed via cooperative warp reduction:
//   Each thread computes partial dot = s_Q[r, lane_id] * s_K[c, lane_id]
//   Then warp_reduce_sum to get full dot product.
//
// PV accumulated per-thread:
//   Each thread handles 1 output dimension (d_idx = lane_id).
//   For each row r: acc[r] += sum_c(P[r,c] * V[c, lane_id])
//
// Shared memory layout (56 KB, fits in MI300X's 64 KB LDS):
//   s_Q:  [64, 64] BF16  =  8 KB
//   s_K:  [64, 64] BF16  =  8 KB
//   s_V:  [64, 64] BF16  =  8 KB
//   s_S:  [64, 64] FP32  = 16 KB
//   s_PV: [64, 64] FP32  = 16 KB
//   Total: 56 KB

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
// Compile-time tile parameters
// ---------------------------------------------------------------------------
static constexpr int FA_BLOCK_M = 64;
static constexpr int FA_BLOCK_N = 64;
static constexpr int FA_HEAD_DIM = 64;
static constexpr int FA_WARP_SIZE = 64;

static constexpr int FA_NUM_WARPS = 2;
static constexpr int FA_THREADS = FA_NUM_WARPS * FA_WARP_SIZE; // 128

// Each warp owns 32 rows of Q
static constexpr int FA_ROWS_PER_WARP = FA_BLOCK_M / FA_NUM_WARPS; // 32

// Each thread handles 1 d-element (64 dims / 64 lanes = 1)
static constexpr int FA_D_PER_THREAD = FA_HEAD_DIM / FA_WARP_SIZE; // 1

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float bf16_to_fp32(__hip_bfloat16 v) {
    return __bfloat162float(v);
}

static __device__ __forceinline__ __hip_bfloat16 fp32_to_bf16(float v) {
    return __float2bfloat16(v);
}

// Warp-level max reduction (wavefront64)
static __device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = FA_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Warp-level sum reduction (wavefront64)
static __device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = FA_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Flash Attention v2 kernel -- Scalar version for MI300X
//
// Grid:  (cdiv(num_tokens, BLOCK_M), num_q_heads, 1)
// Block: (128, 1, 1) -- 2 wavefronts
//
// Shared memory layout (56 KB):
//   s_Q:  [BLOCK_M, HEAD_DIM] BF16     =  8 KB
//   s_K:  [BLOCK_N, HEAD_DIM] BF16     =  8 KB
//   s_V:  [BLOCK_N, HEAD_DIM] BF16     =  8 KB
//   s_S:  [BLOCK_M, BLOCK_N]  FP32     = 16 KB
//   s_PV: [BLOCK_M, HEAD_DIM] FP32     = 16 KB
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(FA_THREADS)
flash_attention_kernel(
    const __hip_bfloat16* __restrict__ Q,
    const __hip_bfloat16* __restrict__ K,
    const __hip_bfloat16* __restrict__ V,
    __hip_bfloat16* __restrict__ output,
    const float* __restrict__ sink_values,
    const int num_tokens,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int window_size)
{
    const int tile_row = blockIdx.x;
    const int q_head   = blockIdx.y;
    const int kv_head  = q_head / (num_q_heads / num_kv_heads);

    const int q_row_start = tile_row * FA_BLOCK_M;

    if (q_row_start >= num_tokens) return;

    const int q_rows_this_tile = min(FA_BLOCK_M, num_tokens - q_row_start);

    const int tid      = threadIdx.x;
    const int warp_id  = tid / FA_WARP_SIZE;
    const int lane_id  = tid % FA_WARP_SIZE;

    const int warp_row_start = warp_id * FA_ROWS_PER_WARP;

    const int q_head_stride = head_dim;
    const int q_token_stride = num_q_heads * head_dim;
    const int k_head_stride = head_dim;
    const int k_token_stride = num_kv_heads * head_dim;

    // -----------------------------------------------------------------------
    // Shared memory layout
    // -----------------------------------------------------------------------
    extern __shared__ char smem_raw[];

    constexpr int OFF_Q = 0;
    constexpr int OFF_K = FA_BLOCK_M * FA_HEAD_DIM * sizeof(__hip_bfloat16);
    constexpr int OFF_V = OFF_K + FA_BLOCK_N * FA_HEAD_DIM * sizeof(__hip_bfloat16);
    constexpr int OFF_S = OFF_V + FA_BLOCK_N * FA_HEAD_DIM * sizeof(__hip_bfloat16);
    constexpr int OFF_PV = OFF_S + FA_BLOCK_M * FA_BLOCK_N * sizeof(float);

    __hip_bfloat16* s_Q  = reinterpret_cast<__hip_bfloat16*>(smem_raw + OFF_Q);
    __hip_bfloat16* s_K  = reinterpret_cast<__hip_bfloat16*>(smem_raw + OFF_K);
    __hip_bfloat16* s_V  = reinterpret_cast<__hip_bfloat16*>(smem_raw + OFF_V);
    float*         s_S  = reinterpret_cast<float*>(smem_raw + OFF_S);
    float*         s_PV = reinterpret_cast<float*>(smem_raw + OFF_PV);

    // -----------------------------------------------------------------------
    // Load Q tile into shared memory
    // -----------------------------------------------------------------------
    {
        const int total_elems = FA_BLOCK_M * FA_HEAD_DIM;
        for (int idx = tid; idx < total_elems; idx += FA_THREADS) {
            int row = idx / FA_HEAD_DIM;
            int col = idx % FA_HEAD_DIM;
            int global_row = q_row_start + row;
            if (global_row < num_tokens) {
                s_Q[row * FA_HEAD_DIM + col] =
                    Q[global_row * q_token_stride + q_head * q_head_stride + col];
            } else {
                s_Q[row * FA_HEAD_DIM + col] = fp32_to_bf16(0.0f);
            }
        }
    }
    __syncthreads();

    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    const float sink_bias = (sink_values != nullptr) ? sink_values[q_head] : 0.0f;

    // -----------------------------------------------------------------------
    // Per-row online softmax state
    // -----------------------------------------------------------------------
    float row_max[FA_ROWS_PER_WARP];
    float row_sum[FA_ROWS_PER_WARP];
    float acc[FA_ROWS_PER_WARP][FA_D_PER_THREAD];

    #pragma unroll
    for (int r = 0; r < FA_ROWS_PER_WARP; r++) {
        row_max[r] = -FLT_MAX;
        row_sum[r] = 0.0f;
        #pragma unroll
        for (int d = 0; d < FA_D_PER_THREAD; d++) {
            acc[r][d] = 0.0f;
        }
    }

    // -----------------------------------------------------------------------
    // Determine KV column range
    // -----------------------------------------------------------------------
    const int last_q_row = q_row_start + q_rows_this_tile - 1;
    const int kv_end = last_q_row + 1;

    int kv_start = 0;
    if (window_size > 0) {
        int earliest = q_row_start - window_size + 1;
        kv_start = (earliest > 0) ? earliest : 0;
    }
    kv_start = (kv_start / FA_BLOCK_N) * FA_BLOCK_N;

    const int num_kv_tiles = cdiv(kv_end - kv_start, FA_BLOCK_N);

    // -----------------------------------------------------------------------
    // Main loop over KV tiles
    // -----------------------------------------------------------------------
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_col_start = kv_start + kv_tile * FA_BLOCK_N;

        // Load K tile
        {
            const int total_elems = FA_BLOCK_N * FA_HEAD_DIM;
            for (int idx = tid; idx < total_elems; idx += FA_THREADS) {
                int row = idx / FA_HEAD_DIM;
                int col = idx % FA_HEAD_DIM;
                int global_kv_row = kv_col_start + row;
                if (global_kv_row < num_tokens) {
                    s_K[row * FA_HEAD_DIM + col] =
                        K[global_kv_row * k_token_stride + kv_head * k_head_stride + col];
                } else {
                    s_K[row * FA_HEAD_DIM + col] = fp32_to_bf16(0.0f);
                }
            }
        }

        // Load V tile
        {
            const int total_elems = FA_BLOCK_N * FA_HEAD_DIM;
            for (int idx = tid; idx < total_elems; idx += FA_THREADS) {
                int row = idx / FA_HEAD_DIM;
                int col = idx % FA_HEAD_DIM;
                int global_kv_row = kv_col_start + row;
                if (global_kv_row < num_tokens) {
                    s_V[row * FA_HEAD_DIM + col] =
                        V[global_kv_row * k_token_stride + kv_head * k_head_stride + col];
                } else {
                    s_V[row * FA_HEAD_DIM + col] = fp32_to_bf16(0.0f);
                }
            }
        }
        __syncthreads();

        // -------------------------------------------------------------------
        // Phase 1: Compute S = Q @ K^T using scalar FP32 dot products
        //
        // Each warp handles FA_ROWS_PER_WARP=32 rows.
        // For each (row, col) pair, each thread contributes one partial product
        // (lane_id maps to dimension d), then warp_reduce_sum gives the full dot.
        //
        // With FA_D_PER_THREAD=1, d_idx = lane_id covers all 64 dims.
        // -------------------------------------------------------------------
        for (int r = 0; r < FA_ROWS_PER_WARP; r++) {
            int q_local_row = warp_row_start + r;
            // Load Q[row, lane_id] once for all columns
            float q_val = bf16_to_fp32(s_Q[q_local_row * FA_HEAD_DIM + lane_id]);

            for (int c = 0; c < FA_BLOCK_N; c++) {
                float k_val = bf16_to_fp32(s_K[c * FA_HEAD_DIM + lane_id]);
                float partial = q_val * k_val;
                float dot = warp_reduce_sum(partial);

                // Only lane 0 has the correct result; write to s_S
                if (lane_id == 0) {
                    s_S[q_local_row * FA_BLOCK_N + c] = dot * scale;
                }
            }
        }
        __syncthreads();

        // -------------------------------------------------------------------
        // Apply causal + sliding window masking to s_S
        // -------------------------------------------------------------------
        {
            const int total_elems = FA_BLOCK_M * FA_BLOCK_N;
            for (int idx = tid; idx < total_elems; idx += FA_THREADS) {
                int r = idx / FA_BLOCK_N;
                int c = idx % FA_BLOCK_N;
                int q_global_row = q_row_start + r;
                int kv_global_col = kv_col_start + c;

                bool masked = false;
                if (kv_global_col > q_global_row) masked = true;
                if (kv_global_col >= num_tokens) masked = true;
                if (r >= q_rows_this_tile) masked = true;
                if (window_size > 0 && kv_global_col < q_global_row - window_size + 1)
                    masked = true;

                if (masked) s_S[r * FA_BLOCK_N + c] = -FLT_MAX;
            }
        }
        __syncthreads();

        // -------------------------------------------------------------------
        // Phase 2: Online softmax (per-warp, each warp handles its 32 rows)
        // -------------------------------------------------------------------
        for (int r = 0; r < FA_ROWS_PER_WARP; r++) {
            int q_local_row = warp_row_start + r;
            if (q_local_row >= q_rows_this_tile) {
                // Zero out S for out-of-bounds rows
                for (int c = lane_id; c < FA_BLOCK_N; c += FA_WARP_SIZE) {
                    s_S[q_local_row * FA_BLOCK_N + c] = 0.0f;
                }
                continue;
            }

            // Find tile-local max
            float local_max = -FLT_MAX;
            for (int c = lane_id; c < FA_BLOCK_N; c += FA_WARP_SIZE) {
                local_max = fmaxf(local_max, s_S[q_local_row * FA_BLOCK_N + c]);
            }
            local_max = warp_reduce_max(local_max);

            // Include learned sink bias
            if (kv_tile == 0 && sink_bias != 0.0f) {
                local_max = fmaxf(local_max, sink_bias);
            }

            float m_prev = row_max[r];
            float m_new = fmaxf(m_prev, local_max);

            float correction = (m_prev > -FLT_MAX) ? expf(m_prev - m_new) : 0.0f;

            // Compute exp and sum
            float tile_exp_sum = 0.0f;
            for (int c = lane_id; c < FA_BLOCK_N; c += FA_WARP_SIZE) {
                float s_val = s_S[q_local_row * FA_BLOCK_N + c];
                float e = (s_val > -FLT_MAX) ? expf(s_val - m_new) : 0.0f;
                s_S[q_local_row * FA_BLOCK_N + c] = e;
                tile_exp_sum += e;
            }
            tile_exp_sum = warp_reduce_sum(tile_exp_sum);

            float l_new = row_sum[r] * correction + tile_exp_sum;

            if (kv_tile == 0 && sink_bias != 0.0f) {
                l_new += expf(sink_bias - m_new);
            }

            // Rescale previous accumulator
            #pragma unroll
            for (int dd = 0; dd < FA_D_PER_THREAD; dd++) {
                acc[r][dd] *= correction;
            }

            row_max[r] = m_new;
            row_sum[r] = l_new;
        }
        __syncthreads();

        // -------------------------------------------------------------------
        // Phase 3: Compute PV accumulation (scalar)
        //
        // Each thread owns d_idx = lane_id (1 output dimension).
        // For each row r: acc[r][0] += sum_c(P[r,c] * V[c, lane_id])
        // P values are in s_S (already exp'd).
        // -------------------------------------------------------------------
        for (int r = 0; r < FA_ROWS_PER_WARP; r++) {
            int q_local_row = warp_row_start + r;
            if (q_local_row >= q_rows_this_tile) continue;

            float pv_sum = 0.0f;
            for (int c = 0; c < FA_BLOCK_N; c++) {
                float p_val = s_S[q_local_row * FA_BLOCK_N + c];
                if (p_val > 0.0f) {
                    float v_val = bf16_to_fp32(s_V[c * FA_HEAD_DIM + lane_id]);
                    pv_sum += p_val * v_val;
                }
            }
            acc[r][0] += pv_sum;
        }
        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // Finalize: divide accumulator by running sum, write output
    // -----------------------------------------------------------------------
    for (int r = 0; r < FA_ROWS_PER_WARP; r++) {
        int q_local_row = warp_row_start + r;
        int q_global_row = q_row_start + q_local_row;
        if (q_global_row >= num_tokens || q_local_row >= q_rows_this_tile) continue;

        float inv_sum = (row_sum[r] > 0.0f) ? (1.0f / row_sum[r]) : 0.0f;

        #pragma unroll
        for (int dd = 0; dd < FA_D_PER_THREAD; dd++) {
            int d_idx = lane_id * FA_D_PER_THREAD + dd;
            float val = acc[r][dd] * inv_sum;
            output[q_global_row * q_token_stride + q_head * q_head_stride + d_idx] =
                fp32_to_bf16(val);
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side launch function
// ---------------------------------------------------------------------------
void flash_attention_forward(
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
    hipStream_t stream)
{
    if (num_tokens == 0) return;

    const int num_m_tiles = cdiv(num_tokens, FA_BLOCK_M);

    dim3 grid(num_m_tiles, num_q_heads);
    dim3 block(FA_THREADS);

    // Shared memory:
    //  s_Q:  BLOCK_M * head_dim * sizeof(bf16)       =  8 KB
    //  s_K:  BLOCK_N * head_dim * sizeof(bf16)       =  8 KB
    //  s_V:  BLOCK_N * head_dim * sizeof(bf16)       =  8 KB
    //  s_S:  BLOCK_M * BLOCK_N  * sizeof(float)      = 16 KB
    //  s_PV: BLOCK_M * head_dim * sizeof(float)      = 16 KB
    //  Total                                          = 56 KB
    const size_t smem_bytes =
        FA_BLOCK_M * FA_HEAD_DIM * sizeof(__hip_bfloat16) +
        FA_BLOCK_N * FA_HEAD_DIM * sizeof(__hip_bfloat16) +
        FA_BLOCK_N * FA_HEAD_DIM * sizeof(__hip_bfloat16) +
        FA_BLOCK_M * FA_BLOCK_N * sizeof(float) +
        FA_BLOCK_M * FA_HEAD_DIM * sizeof(float);

    // MI300X supports up to 64 KB LDS per workgroup; 56 KB fits without opt-in.
    // Set attribute anyway for safety.
    CUDA_CHECK(hipFuncSetAttribute(
        reinterpret_cast<const void*>(flash_attention_kernel),
        hipFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));

    flash_attention_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, output, sink_values,
        num_tokens, num_q_heads, num_kv_heads, head_dim, window_size);

    CUDA_CHECK(hipGetLastError());
}

} // namespace gptoss
