// fused_mxfp4_gemv.cu -- Fused MXFP4 dequant + GEMV for MI300X decode (M=1)
//
// Phase 2 optimization: eliminates separate dequant kernel + staging buffer.
// Reads FP4 packed weights + E8M0 scales directly, dequants in VGPRs,
// computes dot product — single kernel, single HBM read of weights.
//
// Optimizations:
//   1. TILE_N=16: 4x fewer blocks vs TILE_N=4 (360 vs 1440 for W1)
//   2. Warp shuffle reduction: no __syncthreads for intra-warp (6 rounds)
//   3. Only 1 syncthreads for inter-warp reduction (4 warps)
//   4. Vectorized 4-byte packed weight loads (8 elements per uint32)
//
// Weight layout: row-major [N, K] stored as MXFP4
//   packed: [N, K/2] bytes (2 E2M1 nibbles per byte)
//   scales: [N, K/32] bytes (E8M0, one scale per 32 elements)

#include "hip_compat.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

namespace gptoss {

// ---------------------------------------------------------------------------
// E2M1 (MXFP4) inline decode — register LUT, no constant memory pressure
// ---------------------------------------------------------------------------
__device__ __forceinline__
float e2m1_decode(int nibble) {
    constexpr uint32_t LUT[16] = {
        0x00000000u, 0x3F000000u, 0x3F800000u, 0x3FC00000u,  // +0, +0.5, +1.0, +1.5
        0x40000000u, 0x40400000u, 0x40800000u, 0x40C00000u,  // +2.0, +3.0, +4.0, +6.0
        0x80000000u, 0xBF000000u, 0xBF800000u, 0xBFC00000u,  // -0, -0.5, -1.0, -1.5
        0xC0000000u, 0xC0400000u, 0xC0800000u, 0xC0C00000u,  // -2.0, -3.0, -4.0, -6.0
    };
    return __int_as_float(static_cast<int>(LUT[nibble & 0xF]));
}

// ---------------------------------------------------------------------------
// E8M0 scale decode — single shift + reinterpret
// ---------------------------------------------------------------------------
__device__ __forceinline__
float e8m0_decode(uint8_t scale_byte) {
    uint32_t fp32_bits = static_cast<uint32_t>(scale_byte) << 23;
    return __int_as_float(static_cast<int>(fp32_bits));
}

// ---------------------------------------------------------------------------
// Warp-level reduction using DPP (Data Parallel Primitives) shuffle
// MI300X wavefront64: 6 rounds to reduce 64 lanes
// ---------------------------------------------------------------------------
__device__ __forceinline__
float warp_reduce_sum(float val) {
    // 6 rounds of butterfly reduction for wavefront64
    val += __shfl_xor(val, 32);
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
    return val;
}

// ---------------------------------------------------------------------------
// Fused MXFP4 GEMV kernel — TILE_N=16, warp shuffle reduction
//
// Block: 256 threads = 4 warps (wavefront64)
// Each block computes 16 output elements.
// All 256 threads cooperate on K dimension, then reduce.
// Warp shuffle for intra-warp, shared memory for inter-warp.
// ---------------------------------------------------------------------------

constexpr int FGEMV_BLOCK_SIZE = 256;
constexpr int FGEMV_TILE_N = 4;
constexpr int FGEMV_WARPS = FGEMV_BLOCK_SIZE / 64;  // 4 warps

__global__ void __launch_bounds__(FGEMV_BLOCK_SIZE)
fused_mxfp4_gemv_kernel(
    const __hip_bfloat16* __restrict__ input,    // [1, K]
    const uint8_t* __restrict__ packed_weights,  // [N, K/2] row-major
    const uint8_t* __restrict__ scales,          // [N, K/32] row-major
    __hip_bfloat16* __restrict__ output,         // [1, N]
    const __hip_bfloat16* __restrict__ bias,     // [N] or nullptr
    int N, int K)
{
    const int block_n = blockIdx.x * FGEMV_TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 6;
    const int lane_id = tid & 63;

    const int K_half = K >> 1;
    const int K_scale = K >> 5;

    float acc[FGEMV_TILE_N];
    #pragma unroll
    for (int t = 0; t < FGEMV_TILE_N; t++) acc[t] = 0.0f;

    // Vectorized path: load 4 packed bytes (8 elements) per iteration via uint32
    const int K_half4 = K_half >> 2;  // number of uint32 loads per row

    for (int kb4 = tid; kb4 < K_half4; kb4 += FGEMV_BLOCK_SIZE) {
        const int kb = kb4 << 2;     // byte offset
        const int k0 = kb << 1;      // element offset

        // Load 8 input elements via int4 (128-bit) — shared across all tiles
        const int4* in_ptr = reinterpret_cast<const int4*>(input + k0);
        int4 in_vec = *in_ptr;
        const __hip_bfloat16* in_e = reinterpret_cast<const __hip_bfloat16*>(&in_vec);

        #pragma unroll
        for (int t = 0; t < FGEMV_TILE_N; t++) {
            const int n = block_n + t;
            if (n < N) {
                // Load 4 packed bytes at once (32-bit)
                uint32_t packed4 = *reinterpret_cast<const uint32_t*>(
                    packed_weights + static_cast<int64_t>(n) * K_half + kb);

                // Process 4 bytes (8 elements), each byte pair shares a scale
                float dot = 0.0f;
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    uint8_t packed = (packed4 >> (b * 8)) & 0xFF;
                    int scale_idx = (k0 + b * 2) >> 5;
                    float scale = e8m0_decode(scales[static_cast<int64_t>(n) * K_scale + scale_idx]);
                    float w0 = e2m1_decode(packed & 0x0F) * scale;
                    float w1 = e2m1_decode((packed >> 4) & 0x0F) * scale;
                    dot += __bfloat162float(in_e[b * 2]) * w0
                         + __bfloat162float(in_e[b * 2 + 1]) * w1;
                }
                acc[t] += dot;
            }
        }
    }

    // Handle remainder (K_half not divisible by 4)
    for (int kb = K_half4 * 4 + tid; kb < K_half; kb += FGEMV_BLOCK_SIZE) {
        const int k0 = kb << 1;
        float in0 = __bfloat162float(input[k0]);
        float in1 = __bfloat162float(input[k0 + 1]);
        const int scale_col = k0 >> 5;

        #pragma unroll
        for (int t = 0; t < FGEMV_TILE_N; t++) {
            const int n = block_n + t;
            if (n < N) {
                uint8_t packed = packed_weights[static_cast<int64_t>(n) * K_half + kb];
                float scale = e8m0_decode(scales[static_cast<int64_t>(n) * K_scale + scale_col]);
                float w0 = e2m1_decode(packed & 0x0F) * scale;
                float w1 = e2m1_decode((packed >> 4) & 0x0F) * scale;
                acc[t] += in0 * w0 + in1 * w1;
            }
        }
    }

    // Warp-level reduction (no syncthreads needed)
    #pragma unroll
    for (int t = 0; t < FGEMV_TILE_N; t++) {
        acc[t] = warp_reduce_sum(acc[t]);
    }

    // Inter-warp reduction: lane 0 of each warp writes to smem
    __shared__ float smem[FGEMV_TILE_N][FGEMV_WARPS];

    if (lane_id == 0) {
        #pragma unroll
        for (int t = 0; t < FGEMV_TILE_N; t++) {
            smem[t][warp_id] = acc[t];
        }
    }
    __syncthreads();

    // First warp reduces across warps and writes output
    if (warp_id == 0 && lane_id < FGEMV_TILE_N) {
        float val = 0.0f;
        #pragma unroll
        for (int w = 0; w < FGEMV_WARPS; w++) {
            val += smem[lane_id][w];
        }
        int n = block_n + lane_id;
        if (n < N) {
            if (bias) val += __bfloat162float(bias[n]);
            output[n] = __float2bfloat16(val);
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-expert fused MXFP4 GEMV: all experts in one launch via blockIdx.y
// ---------------------------------------------------------------------------

struct FusedMxfp4GemvDesc {
    const __hip_bfloat16* input;
    const uint8_t* packed;
    const uint8_t* scales;
    __hip_bfloat16* output;
    const __hip_bfloat16* bias;  // [N] or nullptr
    int M;
};

__global__ void __launch_bounds__(FGEMV_BLOCK_SIZE)
fused_mxfp4_gemv_multi_kernel(
    const FusedMxfp4GemvDesc* __restrict__ descs,
    int N, int K, int count)
{
    const int gemm_idx = blockIdx.y;
    if (gemm_idx >= count) return;

    FusedMxfp4GemvDesc desc = descs[gemm_idx];
    if (desc.M == 0) return;

    const int K_half = K >> 1;
    const int K_scale = K >> 5;

    const int block_n = blockIdx.x * FGEMV_TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 6;
    const int lane_id = tid & 63;

    for (int m = 0; m < desc.M; m++) {
        const __hip_bfloat16* in_row = desc.input + static_cast<int64_t>(m) * K;
        __hip_bfloat16* out_row = desc.output + static_cast<int64_t>(m) * N;

        float acc[FGEMV_TILE_N];
        #pragma unroll
        for (int t = 0; t < FGEMV_TILE_N; t++) acc[t] = 0.0f;

        // Vectorized path: uint32 packed loads (4 bytes = 8 elements) + int4 input loads
        const int K_half4 = K_half >> 2;

        for (int kb4 = tid; kb4 < K_half4; kb4 += FGEMV_BLOCK_SIZE) {
            const int kb = kb4 << 2;
            const int k0 = kb << 1;

            const int4* in_ptr = reinterpret_cast<const int4*>(in_row + k0);
            int4 in_vec = *in_ptr;
            const __hip_bfloat16* in_e = reinterpret_cast<const __hip_bfloat16*>(&in_vec);

            #pragma unroll
            for (int t = 0; t < FGEMV_TILE_N; t++) {
                const int n = block_n + t;
                if (n < N) {
                    uint32_t packed4 = *reinterpret_cast<const uint32_t*>(
                        desc.packed + static_cast<int64_t>(n) * K_half + kb);

                    float dot = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < 4; b++) {
                        uint8_t packed = (packed4 >> (b * 8)) & 0xFF;
                        int scale_idx = (k0 + b * 2) >> 5;
                        float scale = e8m0_decode(desc.scales[static_cast<int64_t>(n) * K_scale + scale_idx]);
                        float w0 = e2m1_decode(packed & 0x0F) * scale;
                        float w1 = e2m1_decode((packed >> 4) & 0x0F) * scale;
                        dot += __bfloat162float(in_e[b * 2]) * w0
                             + __bfloat162float(in_e[b * 2 + 1]) * w1;
                    }
                    acc[t] += dot;
                }
            }
        }

        // Remainder
        for (int kb = K_half4 * 4 + tid; kb < K_half; kb += FGEMV_BLOCK_SIZE) {
            const int k0 = kb << 1;
            float in0 = __bfloat162float(in_row[k0]);
            float in1 = __bfloat162float(in_row[k0 + 1]);
            const int scale_col = k0 >> 5;

            #pragma unroll
            for (int t = 0; t < FGEMV_TILE_N; t++) {
                const int n = block_n + t;
                if (n < N) {
                    uint8_t packed = desc.packed[static_cast<int64_t>(n) * K_half + kb];
                    float scale = e8m0_decode(desc.scales[static_cast<int64_t>(n) * K_scale + scale_col]);
                    float w0 = e2m1_decode(packed & 0x0F) * scale;
                    float w1 = e2m1_decode((packed >> 4) & 0x0F) * scale;
                    acc[t] += in0 * w0 + in1 * w1;
                }
            }
        }

        #pragma unroll
        for (int t = 0; t < FGEMV_TILE_N; t++) {
            acc[t] = warp_reduce_sum(acc[t]);
        }

        __shared__ float smem[FGEMV_TILE_N][FGEMV_WARPS];

        if (lane_id == 0) {
            #pragma unroll
            for (int t = 0; t < FGEMV_TILE_N; t++) {
                smem[t][warp_id] = acc[t];
            }
        }
        __syncthreads();

        if (warp_id == 0 && lane_id < FGEMV_TILE_N) {
            float val = 0.0f;
            #pragma unroll
            for (int w = 0; w < FGEMV_WARPS; w++) {
                val += smem[lane_id][w];
            }
            int n = block_n + lane_id;
            if (n < N) {
                if (desc.bias) val += __bfloat162float(desc.bias[n]);
                out_row[n] = __float2bfloat16(val);
            }
        }

        if (desc.M > 1) __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Multi-expert fused MXFP4 GEMV with pointer-array arguments (no device desc buffer)
// Avoids H2D memcpy of descriptors — pointers passed directly in kernel args.
// M=1 only (decode path).
// ---------------------------------------------------------------------------

constexpr int FGEMV_MAX_BATCH = 8;

__global__ void __launch_bounds__(FGEMV_BLOCK_SIZE)
fused_mxfp4_gemv_multi_ptrs_kernel(
    const __hip_bfloat16* const* __restrict__ inputs,    // [count] input pointers
    const uint8_t* const* __restrict__ packed_ptrs,       // [count] packed weight pointers
    const uint8_t* const* __restrict__ scale_ptrs,        // [count] scale pointers
    __hip_bfloat16* const* __restrict__ outputs,          // [count] output pointers
    const __hip_bfloat16* const* __restrict__ bias_ptrs,  // [count] bias pointers (may be nullptr)
    int N, int K, int count)
{
    const int gemm_idx = blockIdx.y;
    if (gemm_idx >= count) return;

    const __hip_bfloat16* in_row = inputs[gemm_idx];
    const uint8_t* packed = packed_ptrs[gemm_idx];
    const uint8_t* scales = scale_ptrs[gemm_idx];
    __hip_bfloat16* out_row = outputs[gemm_idx];
    const __hip_bfloat16* bias = bias_ptrs[gemm_idx];

    const int K_half = K >> 1;
    const int K_scale = K >> 5;

    const int block_n = blockIdx.x * FGEMV_TILE_N;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 6;
    const int lane_id = tid & 63;

    float acc[FGEMV_TILE_N];
    #pragma unroll
    for (int t = 0; t < FGEMV_TILE_N; t++) acc[t] = 0.0f;

    const int K_half4 = K_half >> 2;

    for (int kb4 = tid; kb4 < K_half4; kb4 += FGEMV_BLOCK_SIZE) {
        const int kb = kb4 << 2;
        const int k0 = kb << 1;

        const int4* in_ptr = reinterpret_cast<const int4*>(in_row + k0);
        int4 in_vec = *in_ptr;
        const __hip_bfloat16* in_e = reinterpret_cast<const __hip_bfloat16*>(&in_vec);

        #pragma unroll
        for (int t = 0; t < FGEMV_TILE_N; t++) {
            const int n = block_n + t;
            if (n < N) {
                uint32_t packed4 = *reinterpret_cast<const uint32_t*>(
                    packed + static_cast<int64_t>(n) * K_half + kb);

                float dot = 0.0f;
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    uint8_t p = (packed4 >> (b * 8)) & 0xFF;
                    int scale_idx = (k0 + b * 2) >> 5;
                    float scale = e8m0_decode(scales[static_cast<int64_t>(n) * K_scale + scale_idx]);
                    float w0 = e2m1_decode(p & 0x0F) * scale;
                    float w1 = e2m1_decode((p >> 4) & 0x0F) * scale;
                    dot += __bfloat162float(in_e[b * 2]) * w0
                         + __bfloat162float(in_e[b * 2 + 1]) * w1;
                }
                acc[t] += dot;
            }
        }
    }

    // Remainder
    for (int kb = K_half4 * 4 + tid; kb < K_half; kb += FGEMV_BLOCK_SIZE) {
        const int k0 = kb << 1;
        float in0 = __bfloat162float(in_row[k0]);
        float in1 = __bfloat162float(in_row[k0 + 1]);
        const int scale_col = k0 >> 5;

        #pragma unroll
        for (int t = 0; t < FGEMV_TILE_N; t++) {
            const int n = block_n + t;
            if (n < N) {
                uint8_t p = packed[static_cast<int64_t>(n) * K_half + kb];
                float scale = e8m0_decode(scales[static_cast<int64_t>(n) * K_scale + scale_col]);
                float w0 = e2m1_decode(p & 0x0F) * scale;
                float w1 = e2m1_decode((p >> 4) & 0x0F) * scale;
                acc[t] += in0 * w0 + in1 * w1;
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < FGEMV_TILE_N; t++) {
        acc[t] = warp_reduce_sum(acc[t]);
    }

    __shared__ float smem[FGEMV_TILE_N][FGEMV_WARPS];

    if (lane_id == 0) {
        #pragma unroll
        for (int t = 0; t < FGEMV_TILE_N; t++) {
            smem[t][warp_id] = acc[t];
        }
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < FGEMV_TILE_N) {
        float val = 0.0f;
        #pragma unroll
        for (int w = 0; w < FGEMV_WARPS; w++) {
            val += smem[lane_id][w];
        }
        int n = block_n + lane_id;
        if (n < N) {
            if (bias) val += __bfloat162float(bias[n]);
            out_row[n] = __float2bfloat16(val);
        }
    }
}

void fused_mxfp4_gemv_multi_ptrs_forward(
    const __hip_bfloat16* const* d_inputs,
    const uint8_t* const* d_packed,
    const uint8_t* const* d_scales,
    __hip_bfloat16* const* d_outputs,
    const __hip_bfloat16* const* d_biases,
    int N, int K, int count,
    hipStream_t stream)
{
    if (count == 0) return;
    int grid_n = (N + FGEMV_TILE_N - 1) / FGEMV_TILE_N;
    dim3 grid(grid_n, count);
    fused_mxfp4_gemv_multi_ptrs_kernel<<<grid, FGEMV_BLOCK_SIZE, 0, stream>>>(
        d_inputs, d_packed, d_scales, d_outputs, d_biases, N, K, count);
}

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------

void fused_mxfp4_gemv_forward(
    const __hip_bfloat16* input,
    const uint8_t* packed_weights,
    const uint8_t* scales,
    __hip_bfloat16* output,
    int N, int K,
    hipStream_t stream,
    const __hip_bfloat16* bias)
{
    int grid_n = (N + FGEMV_TILE_N - 1) / FGEMV_TILE_N;
    fused_mxfp4_gemv_kernel<<<grid_n, FGEMV_BLOCK_SIZE, 0, stream>>>(
        input, packed_weights, scales, output, bias, N, K);
}

void fused_mxfp4_gemv_multi_forward(
    const FusedMxfp4GemvDesc* d_descs,
    int N, int K, int count,
    hipStream_t stream)
{
    if (count == 0) return;
    int grid_n = (N + FGEMV_TILE_N - 1) / FGEMV_TILE_N;
    dim3 grid(grid_n, count);
    fused_mxfp4_gemv_multi_kernel<<<grid, FGEMV_BLOCK_SIZE, 0, stream>>>(
        d_descs, N, K, count);
}

} // namespace gptoss
