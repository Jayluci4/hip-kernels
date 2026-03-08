/**
 * moe_combine.cu - Weighted combine + fused residual add (dispatch-free EP-N)
 *
 * HIP port for AMD MI300X (CDNA3, wavefront64)
 * Type translations: __nv_bfloat16 -> __hip_bfloat16, cudaStream_t -> hipStream_t
 * __ldg() calls compile via hip_compat.h template (dereferences pointer directly)
 *
 * Copyright 2026 GPT-OSS Project
 */

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#include "config.h"
#include "hip_compat.h"
#include "tensor.h"
#include "cuda_utils.h"

namespace gptoss {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int COMBINE_THREADS = 256;
static constexpr int COMBINE_TOKENS_PER_BLOCK = 2;

// ---------------------------------------------------------------------------
// Kernel: Vectorized combine + fused residual add (hidden_size % 8 == 0 path)
// ---------------------------------------------------------------------------
__global__ void combine_kernel_vectorized(
    __hip_bfloat16*       __restrict__ output,
    const __hip_bfloat16* __restrict__ local_expert_out,
    const __hip_bfloat16* __restrict__ remote_expert_out,
    const int32_t*       __restrict__ gather_map,
    const float*         __restrict__ expert_weights,
    const __hip_bfloat16* __restrict__ residual,
    int num_tokens,
    int hidden_size,
    int top_k)
{
    const int block_token_start = blockIdx.x * COMBINE_TOKENS_PER_BLOCK;
    const int tid = threadIdx.x;

    constexpr int VEC_ELEMS = 8;
    const int vec_hidden = hidden_size / VEC_ELEMS;

    for (int t = 0; t < COMBINE_TOKENS_PER_BLOCK; ++t) {
        const int token_id = block_token_start + t;
        if (token_id >= num_tokens) return;

        // Preload routing weights and gather map values for this token
        float weights[ModelConfig::num_active_experts];
        int   map_vals[ModelConfig::num_active_experts];
        #pragma unroll
        for (int k = 0; k < ModelConfig::num_active_experts; ++k) {
            const int slot = token_id * top_k + k;
            weights[k]  = expert_weights[slot];
            map_vals[k] = gather_map[slot];
        }

        const int64_t token_offset = static_cast<int64_t>(token_id) * hidden_size;
        int4* out_vec = reinterpret_cast<int4*>(output + token_offset);

        // Residual row pointer
        const int4* res_vec = residual ?
            reinterpret_cast<const int4*>(residual + token_offset) : nullptr;

        for (int vi = tid; vi < vec_hidden; vi += COMBINE_THREADS) {
            float accum[VEC_ELEMS];
            #pragma unroll
            for (int e = 0; e < VEC_ELEMS; ++e) accum[e] = 0.0f;

            #pragma unroll
            for (int k = 0; k < ModelConfig::num_active_experts; ++k) {
                const float w = weights[k];
                const int mv = map_vals[k];
                int4 src_vec;

                if (mv >= 0) {
                    const int4* src_row = reinterpret_cast<const int4*>(
                        local_expert_out + static_cast<int64_t>(mv) * hidden_size);
                    src_vec = src_row[vi];
                } else {
                    const int recv_idx = -(mv + 1);
                    const int4* src_row = reinterpret_cast<const int4*>(
                        remote_expert_out + static_cast<int64_t>(recv_idx) * hidden_size);
                    src_vec = src_row[vi];
                }

                const __hip_bfloat16* src_bf16 = reinterpret_cast<const __hip_bfloat16*>(&src_vec);
                #pragma unroll
                for (int e = 0; e < VEC_ELEMS; ++e) {
                    accum[e] += w * __bfloat162float(src_bf16[e]);
                }
            }

            // Fused residual add: output = residual + combine_result
            if (res_vec) {
                int4 res_packed = res_vec[vi];
                const __hip_bfloat16* res_bf16 = reinterpret_cast<const __hip_bfloat16*>(&res_packed);
                #pragma unroll
                for (int e = 0; e < VEC_ELEMS; ++e) {
                    accum[e] += __bfloat162float(res_bf16[e]);
                }
            }

            int4 out_packed;
            __hip_bfloat16* out_bf16 = reinterpret_cast<__hip_bfloat16*>(&out_packed);
            #pragma unroll
            for (int e = 0; e < VEC_ELEMS; ++e) {
                out_bf16[e] = __float2bfloat16(accum[e]);
            }
            out_vec[vi] = out_packed;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: Scalar fallback combine + fused residual
// ---------------------------------------------------------------------------
__global__ void combine_kernel_scalar(
    __hip_bfloat16*       __restrict__ output,
    const __hip_bfloat16* __restrict__ local_expert_out,
    const __hip_bfloat16* __restrict__ remote_expert_out,
    const int32_t*       __restrict__ gather_map,
    const float*         __restrict__ expert_weights,
    const __hip_bfloat16* __restrict__ residual,
    int num_tokens,
    int hidden_size,
    int top_k)
{
    const int token_id = blockIdx.x;
    if (token_id >= num_tokens) return;

    const int tid = threadIdx.x;

    const int64_t token_offset = static_cast<int64_t>(token_id) * hidden_size;
    __hip_bfloat16* out_row = output + token_offset;

    // Preload per-token routing info
    float weights[ModelConfig::num_active_experts];
    int   map_vals[ModelConfig::num_active_experts];
    for (int k = 0; k < top_k; ++k) {
        const int slot = token_id * top_k + k;
        weights[k]  = expert_weights[slot];
        map_vals[k] = gather_map[slot];
    }

    for (int h = tid; h < hidden_size; h += COMBINE_THREADS) {
        float accum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            const int mv = map_vals[k];
            float expert_val;
            if (mv >= 0) {
                expert_val = __bfloat162float(
                    local_expert_out[static_cast<int64_t>(mv) * hidden_size + h]);
            } else {
                const int recv_idx = -(mv + 1);
                expert_val = __bfloat162float(
                    remote_expert_out[static_cast<int64_t>(recv_idx) * hidden_size + h]);
            }
            accum += weights[k] * expert_val;
        }
        if (residual) {
            accum += __bfloat162float(residual[token_offset + h]);
        }
        out_row[h] = __float2bfloat16(accum);
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------
void moe_combine_forward(
    __hip_bfloat16*       output,
    const __hip_bfloat16* local_expert_out,
    const __hip_bfloat16* remote_expert_out,
    const int32_t*       gather_map,
    const float*         expert_weights,
    const __hip_bfloat16* residual,
    int num_tokens,
    int hidden_size,
    int top_k,
    hipStream_t stream)
{
    if (num_tokens == 0) return;

    dim3 grid(cdiv(num_tokens, COMBINE_TOKENS_PER_BLOCK));
    dim3 block(COMBINE_THREADS);

    if (hidden_size % 8 == 0) {
        combine_kernel_vectorized<<<grid, block, 0, stream>>>(
            output, local_expert_out, remote_expert_out,
            gather_map, expert_weights, residual,
            num_tokens, hidden_size, top_k);
    } else {
        combine_kernel_scalar<<<grid, block, 0, stream>>>(
            output, local_expert_out, remote_expert_out,
            gather_map, expert_weights, residual,
            num_tokens, hidden_size, top_k);
    }
    CUDA_CHECK(hipGetLastError());
}

} // namespace gptoss
