// gemv_bf16.cu -- Custom BF16 GEMV kernel for MI300X decode (M=1)
//
// Replaces hipBLASLt for M=1 GEMMs which have ~1ms CPU overhead per call.
// This kernel has ~5µs launch overhead and achieves near-peak HBM bandwidth.
//
// Computes: output[1, N] = input[1, K] × weight[K, N]^T
//   i.e. output[n] = dot(input[0:K], weight[n, 0:K])
//
// Weight layout: row-major [N, K] (each row is one output neuron's weights)
// This matches the transB=true convention used by the engine.
//
// Optimizations (v2):
//   1. Vectorized 128-bit (int4) loads — 8 BF16 elements per load
//   2. Warp shuffle reduction — no __syncthreads for intra-warp
//   3. Minimal shared memory — only 4 floats per warp for inter-warp reduction
//   4. TILE_N=4: 4 output elements per block, amortizes input loads

#include "hip_compat.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

namespace gptoss {

constexpr int GEMV_BLOCK_SIZE = 256;   // threads per block
constexpr int GEMV_TILE_N = 4;         // output elements per block
constexpr int GEMV_WARPS = GEMV_BLOCK_SIZE / 64;  // 4 warps (wavefront64)

// Warp-level reduction using butterfly shuffle (wavefront64: 6 rounds)
__device__ __forceinline__
float gemv_warp_reduce_sum(float val) {
    val += __shfl_xor(val, 32);
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
    return val;
}

__global__ void __launch_bounds__(GEMV_BLOCK_SIZE)
gemv_bf16_kernel(
    const __hip_bfloat16* __restrict__ input,   // [1, K]
    const __hip_bfloat16* __restrict__ weight,  // [N, K] row-major
    __hip_bfloat16* __restrict__ output,        // [1, N]
    const __hip_bfloat16* __restrict__ bias,    // [N] or nullptr
    int N, int K)
{
    const int block_n = blockIdx.x * GEMV_TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 6;
    const int lane_id = tid & 63;

    float acc[GEMV_TILE_N];
    #pragma unroll
    for (int t = 0; t < GEMV_TILE_N; t++) acc[t] = 0.0f;

    // Vectorized path: process 8 BF16 elements per iteration via int4 loads
    const int K_vec = K >> 3;  // K / 8
    const int k_vec_start = tid;

    for (int kv = k_vec_start; kv < K_vec; kv += GEMV_BLOCK_SIZE) {
        const int k_base = kv << 3;  // kv * 8

        // Load 8 input elements (128-bit)
        const int4* in_ptr = reinterpret_cast<const int4*>(input + k_base);
        int4 in_vec = *in_ptr;
        const __hip_bfloat16* in_elems = reinterpret_cast<const __hip_bfloat16*>(&in_vec);

        #pragma unroll
        for (int t = 0; t < GEMV_TILE_N; t++) {
            const int n = block_n + t;
            if (n < N) {
                // Load 8 weight elements (128-bit) from row n
                const int4* w_ptr = reinterpret_cast<const int4*>(weight + static_cast<int64_t>(n) * K + k_base);
                int4 w_vec = *w_ptr;
                const __hip_bfloat16* w_elems = reinterpret_cast<const __hip_bfloat16*>(&w_vec);

                // Dot product of 8 elements
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    acc[t] += __bfloat162float(in_elems[i]) * __bfloat162float(w_elems[i]);
                }
            }
        }
    }

    // Handle remainder (K not divisible by 8)
    const int k_rem_start = K_vec << 3;
    for (int k = k_rem_start + tid; k < K; k += GEMV_BLOCK_SIZE) {
        float in_val = __bfloat162float(input[k]);
        #pragma unroll
        for (int t = 0; t < GEMV_TILE_N; t++) {
            int n = block_n + t;
            if (n < N) {
                acc[t] += in_val * __bfloat162float(weight[static_cast<int64_t>(n) * K + k]);
            }
        }
    }

    // Warp-level reduction (no __syncthreads needed)
    #pragma unroll
    for (int t = 0; t < GEMV_TILE_N; t++) {
        acc[t] = gemv_warp_reduce_sum(acc[t]);
    }

    // Inter-warp reduction via shared memory (only 1 __syncthreads)
    __shared__ float smem[GEMV_TILE_N][GEMV_WARPS];

    if (lane_id == 0) {
        #pragma unroll
        for (int t = 0; t < GEMV_TILE_N; t++) {
            smem[t][warp_id] = acc[t];
        }
    }
    __syncthreads();

    // First warp reduces across warps and writes output
    if (warp_id == 0 && lane_id < GEMV_TILE_N) {
        float val = 0.0f;
        #pragma unroll
        for (int w = 0; w < GEMV_WARPS; w++) {
            val += smem[lane_id][w];
        }
        int n = block_n + lane_id;
        if (n < N) {
            if (bias) val += __bfloat162float(bias[n]);
            output[n] = __float2bfloat16(val);
        }
    }
}

// Multi-GEMV: processes multiple independent GEMVs (different A/B/C pointers, same N/K)
// Used for batched expert processing in MoE layer.
__global__ void __launch_bounds__(GEMV_BLOCK_SIZE)
gemv_bf16_multi_kernel(
    const __hip_bfloat16* const* __restrict__ A_array,  // [count] input pointers, each [M, K]
    const __hip_bfloat16* const* __restrict__ B_array,  // [count] weight pointers, each [N, K]
    __hip_bfloat16* const* __restrict__ C_array,        // [count] output pointers, each [M, N]
    const int* __restrict__ M_array,                     // [count] M values (expected 1 for decode)
    int N, int K, int count)
{
    const int gemm_idx = blockIdx.y;
    if (gemm_idx >= count) return;

    const int M = M_array[gemm_idx];
    if (M == 0) return;

    const __hip_bfloat16* input = A_array[gemm_idx];
    const __hip_bfloat16* weight = B_array[gemm_idx];
    __hip_bfloat16* out = C_array[gemm_idx];

    const int block_n = blockIdx.x * GEMV_TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 6;
    const int lane_id = tid & 63;

    const int K_vec = K >> 3;

    for (int m = 0; m < M; m++) {
        const __hip_bfloat16* in_row = input + static_cast<int64_t>(m) * K;
        __hip_bfloat16* out_row = out + static_cast<int64_t>(m) * N;

        float acc[GEMV_TILE_N];
        #pragma unroll
        for (int t = 0; t < GEMV_TILE_N; t++) acc[t] = 0.0f;

        // Vectorized K loop
        for (int kv = tid; kv < K_vec; kv += GEMV_BLOCK_SIZE) {
            const int k_base = kv << 3;
            const int4* in_ptr = reinterpret_cast<const int4*>(in_row + k_base);
            int4 in_vec = *in_ptr;
            const __hip_bfloat16* in_elems = reinterpret_cast<const __hip_bfloat16*>(&in_vec);

            #pragma unroll
            for (int t = 0; t < GEMV_TILE_N; t++) {
                const int n = block_n + t;
                if (n < N) {
                    const int4* w_ptr = reinterpret_cast<const int4*>(weight + static_cast<int64_t>(n) * K + k_base);
                    int4 w_vec = *w_ptr;
                    const __hip_bfloat16* w_elems = reinterpret_cast<const __hip_bfloat16*>(&w_vec);
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        acc[t] += __bfloat162float(in_elems[i]) * __bfloat162float(w_elems[i]);
                    }
                }
            }
        }

        // Remainder
        const int k_rem_start = K_vec << 3;
        for (int k = k_rem_start + tid; k < K; k += GEMV_BLOCK_SIZE) {
            float in_val = __bfloat162float(in_row[k]);
            #pragma unroll
            for (int t = 0; t < GEMV_TILE_N; t++) {
                int n = block_n + t;
                if (n < N) {
                    acc[t] += in_val * __bfloat162float(weight[static_cast<int64_t>(n) * K + k]);
                }
            }
        }

        // Warp shuffle reduction
        #pragma unroll
        for (int t = 0; t < GEMV_TILE_N; t++) {
            acc[t] = gemv_warp_reduce_sum(acc[t]);
        }

        __shared__ float smem[GEMV_TILE_N][GEMV_WARPS];

        if (lane_id == 0) {
            #pragma unroll
            for (int t = 0; t < GEMV_TILE_N; t++) {
                smem[t][warp_id] = acc[t];
            }
        }
        __syncthreads();

        if (warp_id == 0 && lane_id < GEMV_TILE_N) {
            float val = 0.0f;
            #pragma unroll
            for (int w = 0; w < GEMV_WARPS; w++) {
                val += smem[lane_id][w];
            }
            int n = block_n + lane_id;
            if (n < N) {
                out_row[n] = __float2bfloat16(val);
            }
        }

        if (M > 1) __syncthreads();
    }
}

// ---- Host API ----

void gemv_bf16_forward(
    const __hip_bfloat16* input,
    const __hip_bfloat16* weight,
    __hip_bfloat16* output,
    int N, int K,
    hipStream_t stream,
    const __hip_bfloat16* bias)
{
    int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
    gemv_bf16_kernel<<<grid_n, GEMV_BLOCK_SIZE, 0, stream>>>(
        input, weight, output, bias, N, K);
}

void gemv_bf16_multi_forward(
    const __hip_bfloat16* const* A_array,
    const __hip_bfloat16* const* B_array,
    __hip_bfloat16* const* C_array,
    const int* M_array,
    int N, int K, int count,
    hipStream_t stream)
{
    if (count == 0) return;
    int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
    dim3 grid(grid_n, count);
    gemv_bf16_multi_kernel<<<grid, GEMV_BLOCK_SIZE, 0, stream>>>(
        A_array, B_array, C_array, M_array, N, K, count);
}

} // namespace gptoss
