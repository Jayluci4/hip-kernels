// MXFP4 Dequantization Kernel for GPT-OSS-120B
// Unpacks E2M1 nibble pairs with E8M0 shared scales to BF16
//
// HIP port for AMD MI300X (CDNA3, wavefront64)
//
// Optimizations retained:
//   1. Inline bit manipulation for E2M1->FP32 (no constant memory LUT)
//   2. Fast E8M0->FP32 via bit shift + reinterpret (no ldexpf)
//   3. 128-bit vectorized BF16 stores (int4 = 8x BF16)
//   4. Shared memory scale caching

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <mutex>

#include "config.h"
#include "hip_compat.h"
#include "cuda_utils.h"

namespace gptoss {

// ---------------------------------------------------------------------------
// E2M1 (MXFP4) bit manipulation decode -- replaces constant memory LUT
//
// 4-bit layout: [sign(1)] [exp(2)] [mantissa(1)]
//   exp=00: subnormal -> value = (-1)^sign * 0.mantissa * 2^(1-bias) = +/-0 or +/-0.5
//   exp=01: normal     -> value = (-1)^sign * 1.mantissa * 2^(1-1) = +/-1 or +/-1.5
//   exp=10: normal     -> value = (-1)^sign * 1.mantissa * 2^(2-1) = +/-2 or +/-3
//   exp=11: normal     -> value = (-1)^sign * 1.mantissa * 2^(3-1) = +/-4 or +/-6
// ---------------------------------------------------------------------------
__device__ __forceinline__
float e2m1_to_fp32(int nibble) {
    constexpr uint32_t LUT[16] = {
        0x00000000u, 0x3F000000u, 0x3F800000u, 0x3FC00000u,  // +0, +0.5, +1.0, +1.5
        0x40000000u, 0x40400000u, 0x40800000u, 0x40C00000u,  // +2.0, +3.0, +4.0, +6.0
        0x80000000u, 0xBF000000u, 0xBF800000u, 0xBFC00000u,  // -0, -0.5, -1.0, -1.5
        0xC0000000u, 0xC0400000u, 0xC0800000u, 0xC0C00000u,  // -2.0, -3.0, -4.0, -6.0
    };
    return __int_as_float(static_cast<int>(LUT[nibble & 0xF]));
}

// ---------------------------------------------------------------------------
// E8M0 scale decode -- single shift + reinterpret
// ---------------------------------------------------------------------------
__device__ __forceinline__
float e8m0_to_fp32(uint8_t scale_byte) {
    uint32_t fp32_bits = static_cast<uint32_t>(scale_byte) << 23;
    return __int_as_float(static_cast<int>(fp32_bits));
}

// Legacy constant memory tables -- kept for scalar/vec4 fallback kernels
__device__ __constant__ float MXFP4_LUT[16] = {
    0.0f,   0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f,  -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __constant__ float E8M0_SCALE_TABLE[256] = {};

static std::once_flag e8m0_init_flag;
static float h_e8m0_table[256];

static void ensure_e8m0_table() {
    std::call_once(e8m0_init_flag, []() {
        for (int i = 0; i < 256; ++i) {
            h_e8m0_table[i] = ldexpf(1.0f, i - 127);
        }
        CUDA_CHECK(hipMemcpyToSymbol(E8M0_SCALE_TABLE, h_e8m0_table,
                                       256 * sizeof(float)));
    });
}

// Main dequantization kernel.
__global__ void mxfp4_dequant_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    __hip_bfloat16* __restrict__ output,
    int num_elements)
{
    constexpr int BLOCK_SIZE = ModelConfig::mxfp4_block_size; // 32

    int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_idx = byte_idx * 2;

    if (elem_idx >= num_elements) return;

    // Unpack the two nibbles
    uint8_t packed_byte = packed[byte_idx];
    int low_nibble  = packed_byte & 0x0F;        // even element
    int high_nibble = (packed_byte >> 4) & 0x0F;  // odd element

    // LUT lookup for the E2M1 mantissa+sign values
    float val_low  = MXFP4_LUT[low_nibble];
    float val_high = MXFP4_LUT[high_nibble];

    // Fetch the shared E8M0 scale for each element's block
    int scale_idx_low  = elem_idx / BLOCK_SIZE;
    int scale_idx_high = (elem_idx + 1) / BLOCK_SIZE;

    float scale_low  = E8M0_SCALE_TABLE[scales[scale_idx_low]];
    float scale_high = E8M0_SCALE_TABLE[scales[scale_idx_high]];

    // Apply scale and convert to BF16
    output[elem_idx] = __float2bfloat16(val_low * scale_low);

    if (elem_idx + 1 < num_elements) {
        output[elem_idx + 1] = __float2bfloat16(val_high * scale_high);
    }
}

// Vectorized kernel: processes 4 packed bytes (8 elements) per thread
__global__ void mxfp4_dequant_kernel_vec4(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    __hip_bfloat16* __restrict__ output,
    int num_elements)
{
    constexpr int BLOCK_SIZE = ModelConfig::mxfp4_block_size; // 32

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int byte_start = tid * 4;
    int elem_start = byte_start * 2;

    if (elem_start >= num_elements) return;

    // Load 4 bytes at once via uint32
    uint32_t packed4;
    if (byte_start + 3 < (num_elements + 1) / 2) {
        packed4 = *reinterpret_cast<const uint32_t*>(packed + byte_start);
    } else {
        // Near the boundary, load byte-by-byte
        packed4 = 0;
        for (int i = 0; i < 4 && (byte_start + i) < (num_elements + 1) / 2; ++i) {
            packed4 |= static_cast<uint32_t>(packed[byte_start + i]) << (i * 8);
        }
    }

    // Process 8 elements
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int elem_idx = elem_start + i * 2;
        if (elem_idx >= num_elements) break;

        uint8_t byte_val = (packed4 >> (i * 8)) & 0xFF;
        int low_nibble  = byte_val & 0x0F;
        int high_nibble = (byte_val >> 4) & 0x0F;

        float val_low  = MXFP4_LUT[low_nibble];
        float val_high = MXFP4_LUT[high_nibble];

        int scale_idx_low  = elem_idx / BLOCK_SIZE;
        int scale_idx_high = (elem_idx + 1) / BLOCK_SIZE;

        float scale_low  = E8M0_SCALE_TABLE[scales[scale_idx_low]];
        float scale_high = E8M0_SCALE_TABLE[scales[scale_idx_high]];

        output[elem_idx] = __float2bfloat16(val_low * scale_low);
        if (elem_idx + 1 < num_elements) {
            output[elem_idx + 1] = __float2bfloat16(val_high * scale_high);
        }
    }
}

// ---------------------------------------------------------------------------
// Fully optimized kernel: bit-manipulation + vectorized stores + smem scale cache
// ---------------------------------------------------------------------------
__global__ void mxfp4_dequant_kernel_smem(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    __hip_bfloat16* __restrict__ output,
    int num_elements)
{
    constexpr int BLOCK_SIZE_SHIFT = 5;                                // log2(32)
    constexpr int THREADS_PER_BLK = 256;
    constexpr int BYTES_PER_THREAD = 8;                                // 8 packed bytes = 16 elements
    constexpr int ELEMS_PER_BLK = THREADS_PER_BLK * BYTES_PER_THREAD * 2; // 4096
    constexpr int SCALES_PER_BLK = ELEMS_PER_BLK >> BLOCK_SIZE_SHIFT;     // 128

    __shared__ float smem_scales_fp32[SCALES_PER_BLK];

    int block_elem_start = blockIdx.x * ELEMS_PER_BLK;
    int block_scale_start = block_elem_start >> BLOCK_SIZE_SHIFT;

    // Cooperative scale loading
    int total_scales = (num_elements + 31) >> BLOCK_SIZE_SHIFT;
    for (int s = threadIdx.x; s < SCALES_PER_BLK; s += THREADS_PER_BLK) {
        int global_scale_idx = block_scale_start + s;
        smem_scales_fp32[s] = (global_scale_idx < total_scales) ?
            e8m0_to_fp32(scales[global_scale_idx]) : 0.0f;
    }
    __syncthreads();

    // Each thread processes 8 packed bytes = 16 elements
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int byte_start = tid * BYTES_PER_THREAD;
    int elem_start = byte_start * 2;

    if (elem_start >= num_elements) return;

    // Load 8 packed bytes as 2x uint32
    int total_packed_bytes = (num_elements + 1) >> 1;
    uint32_t packed_lo, packed_hi;

    if (byte_start + 7 < total_packed_bytes) {
        const uint32_t* p32 = reinterpret_cast<const uint32_t*>(packed + byte_start);
        packed_lo = p32[0];
        packed_hi = p32[1];
    } else {
        packed_lo = 0; packed_hi = 0;
        for (int i = 0; i < 4 && (byte_start + i) < total_packed_bytes; ++i)
            packed_lo |= static_cast<uint32_t>(packed[byte_start + i]) << (i * 8);
        for (int i = 0; i < 4 && (byte_start + 4 + i) < total_packed_bytes; ++i)
            packed_hi |= static_cast<uint32_t>(packed[byte_start + 4 + i]) << (i * 8);
    }

    __hip_bfloat16 out_regs[16];

    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        uint32_t pack4 = (half == 0) ? packed_lo : packed_hi;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int elem_idx = elem_start + half * 8 + i * 2;
            uint8_t byte_val = (pack4 >> (i * 8)) & 0xFF;

            float val_low  = e2m1_to_fp32(byte_val & 0x0F);
            float val_high = e2m1_to_fp32((byte_val >> 4) & 0x0F);

            // Bit-shift scale indexing + shared scale for element pair
            int local_scale = (elem_idx >> BLOCK_SIZE_SHIFT) - block_scale_start;
            float scale = smem_scales_fp32[local_scale];

            int reg_base = half * 8 + i * 2;
            out_regs[reg_base]     = __float2bfloat16(val_low * scale);
            out_regs[reg_base + 1] = __float2bfloat16(val_high * scale);
        }
    }

    if (elem_start + 15 < num_elements) {
        int4* out_vec = reinterpret_cast<int4*>(output + elem_start);
        out_vec[0] = *reinterpret_cast<int4*>(&out_regs[0]);
        out_vec[1] = *reinterpret_cast<int4*>(&out_regs[8]);
    } else {
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (elem_start + r < num_elements) {
                output[elem_start + r] = out_regs[r];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Batched dequant kernel: processes N experts in a single launch.
// ---------------------------------------------------------------------------
struct MxFp4BatchDesc {
    const uint8_t* packed;
    const uint8_t* scales;
    __hip_bfloat16* output;
    int num_elements;
};

__global__ void mxfp4_dequant_kernel_batched(
    const MxFp4BatchDesc* __restrict__ descs,
    int blocks_per_expert)
{
    constexpr int BLOCK_SIZE_SHIFT = 5;
    constexpr int THREADS_PER_BLK = 256;
    constexpr int BYTES_PER_THREAD = 8;
    constexpr int ELEMS_PER_BLK = THREADS_PER_BLK * BYTES_PER_THREAD * 2; // 4096
    constexpr int SCALES_PER_BLK = ELEMS_PER_BLK >> BLOCK_SIZE_SHIFT;     // 128

    __shared__ float smem_scales_fp32[SCALES_PER_BLK];

    int expert_idx = blockIdx.y;
    int block_in_expert = blockIdx.x;

    MxFp4BatchDesc desc = descs[expert_idx];
    const uint8_t* packed = desc.packed;
    const uint8_t* scales = desc.scales;
    __hip_bfloat16* output = desc.output;
    int num_elements = desc.num_elements;

    int block_elem_start = block_in_expert * ELEMS_PER_BLK;
    int block_scale_start = block_elem_start >> BLOCK_SIZE_SHIFT;

    int total_scales = (num_elements + 31) >> BLOCK_SIZE_SHIFT;
    for (int s = threadIdx.x; s < SCALES_PER_BLK; s += THREADS_PER_BLK) {
        int global_scale_idx = block_scale_start + s;
        smem_scales_fp32[s] = (global_scale_idx < total_scales) ?
            e8m0_to_fp32(scales[global_scale_idx]) : 0.0f;
    }
    __syncthreads();

    int tid = block_in_expert * blockDim.x + threadIdx.x;
    int byte_start = tid * BYTES_PER_THREAD;
    int elem_start = byte_start * 2;

    if (elem_start >= num_elements) return;

    int total_packed_bytes = (num_elements + 1) >> 1;
    uint32_t packed_lo, packed_hi;

    if (byte_start + 7 < total_packed_bytes) {
        const uint32_t* p32 = reinterpret_cast<const uint32_t*>(packed + byte_start);
        packed_lo = p32[0];
        packed_hi = p32[1];
    } else {
        packed_lo = 0; packed_hi = 0;
        for (int i = 0; i < 4 && (byte_start + i) < total_packed_bytes; ++i)
            packed_lo |= static_cast<uint32_t>(packed[byte_start + i]) << (i * 8);
        for (int i = 0; i < 4 && (byte_start + 4 + i) < total_packed_bytes; ++i)
            packed_hi |= static_cast<uint32_t>(packed[byte_start + 4 + i]) << (i * 8);
    }

    __hip_bfloat16 out_regs[16];

    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        uint32_t pack4 = (half == 0) ? packed_lo : packed_hi;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int elem_idx = elem_start + half * 8 + i * 2;
            uint8_t byte_val = (pack4 >> (i * 8)) & 0xFF;

            float val_low  = e2m1_to_fp32(byte_val & 0x0F);
            float val_high = e2m1_to_fp32((byte_val >> 4) & 0x0F);

            int local_scale = (elem_idx >> BLOCK_SIZE_SHIFT) - block_scale_start;
            float scale = smem_scales_fp32[local_scale];

            int reg_base = half * 8 + i * 2;
            out_regs[reg_base]     = __float2bfloat16(val_low * scale);
            out_regs[reg_base + 1] = __float2bfloat16(val_high * scale);
        }
    }

    if (elem_start + 15 < num_elements) {
        int4* out_vec = reinterpret_cast<int4*>(output + elem_start);
        out_vec[0] = *reinterpret_cast<int4*>(&out_regs[0]);
        out_vec[1] = *reinterpret_cast<int4*>(&out_regs[8]);
    } else {
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (elem_start + r < num_elements) {
                output[elem_start + r] = out_regs[r];
            }
        }
    }
}

// Batched launch: dequant N experts in a single kernel
void mxfp4_dequant_batched(
    const MxFp4BatchDesc* d_descs,
    int num_experts,
    int max_num_elements,
    hipStream_t stream)
{
    if (num_experts == 0) return;

    ensure_e8m0_table();

    constexpr int THREADS = 256;
    constexpr int BYTES_PER_THREAD = 8;

    int blocks_per_expert = cdiv(cdiv((max_num_elements + 1) / 2, BYTES_PER_THREAD), THREADS);

    dim3 grid(blocks_per_expert, num_experts);
    dim3 block(THREADS);

    mxfp4_dequant_kernel_batched<<<grid, block, 0, stream>>>(
        d_descs, blocks_per_expert);

    CUDA_CHECK(hipGetLastError());
}

// Launch function
void mxfp4_dequant(
    const uint8_t* packed,
    const uint8_t* scales,
    __hip_bfloat16* output,
    int num_elements,
    hipStream_t stream)
{
    if (num_elements == 0) return;

    // Ensure the E8M0 lookup table is uploaded to constant memory
    ensure_e8m0_table();

    // Use vectorized kernel when element count is large enough and aligned
    int num_bytes = (num_elements + 1) / 2;

    if (num_bytes >= 256 && (num_bytes % 8 == 0)) {
        // Fully optimized path: 16 elements per thread, bit-shift scale
        // indexing, int4 loads+stores, branchless E2M1 LUT, smem scale caching.
        int num_threads_needed = cdiv(num_bytes, 8);
        constexpr int THREADS = 256;
        int blocks = cdiv(num_threads_needed, THREADS);

        mxfp4_dequant_kernel_smem<<<blocks, THREADS, 0, stream>>>(
            packed, scales, output, num_elements);
    } else if (num_bytes >= 4 && (num_bytes % 4 == 0)) {
        // Vectorized fallback path without shared memory optimization
        int num_threads_needed = cdiv(num_bytes, 4);
        constexpr int THREADS = 256;
        int blocks = cdiv(num_threads_needed, THREADS);

        mxfp4_dequant_kernel_vec4<<<blocks, THREADS, 0, stream>>>(
            packed, scales, output, num_elements);
    } else {
        // Scalar path: each thread handles 1 packed byte = 2 elements
        int num_threads_needed = (num_elements + 1) / 2;
        constexpr int THREADS = 256;
        int blocks = cdiv(num_threads_needed, THREADS);

        mxfp4_dequant_kernel<<<blocks, THREADS, 0, stream>>>(
            packed, scales, output, num_elements);
    }

    CUDA_CHECK(hipGetLastError());
}

} // namespace gptoss
