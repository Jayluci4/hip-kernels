/**
 * topk_softmax.cu - Expert routing kernel for GPT-OSS 120B MoE inference
 *
 * HIP port for AMD MI300X (CDNA3, wavefront64)
 *
 * Key warp-size changes from CUDA version:
 *   - WARP_SIZE = 32 -> 64
 *   - NUM_WARPS_PER_BLOCK = 4 -> 2 (keep 128 threads, fewer warps)
 *   - warp_reduce_max: loop starts at 32 instead of 16
 *   - MAX_PER_LANE = 2 -> 1 (64 lanes per warp, chunk=64, 1 expert/lane)
 *   - __shfl_xor_sync -> __shfl_xor (via hip_compat.h wrappers)
 *   - __shfl_sync -> __shfl (via hip_compat.h wrappers)
 *
 * Copyright 2026 GPT-OSS Project
 */

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

#include "config.h"
#include "hip_compat.h"
#include "tensor.h"
#include "cuda_utils.h"

namespace gptoss {

// ---------------------------------------------------------------------------
// Constants — adjusted for MI300X wavefront64
// ---------------------------------------------------------------------------
static constexpr int WARP_SIZE = 64;
static constexpr int MAX_TOP_K = 8;

// 128 experts, 64 lanes per warp = 2 warps to cover all experts
static constexpr int NUM_WARPS_PER_BLOCK = 2;
static constexpr int THREADS_PER_BLOCK = NUM_WARPS_PER_BLOCK * WARP_SIZE; // 128

// Process multiple tokens per block to amortize launch overhead
static constexpr int TOKENS_PER_BLOCK = 8;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Sanitize NaN/Inf to -FLT_MAX
__device__ __forceinline__
float sanitize_logit(float v) {
    unsigned bits = __float_as_uint(v);
    return (bits & 0x7F800000u) == 0x7F800000u ? -FLT_MAX : v;
}

struct ValIdx {
    float val;
    int   idx;
};

// Warp-level max reduction (wavefront64: starts at offset=32)
__device__ __forceinline__
ValIdx warp_reduce_max(float val, int idx) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor(val, offset);
        int   other_idx = __shfl_xor(idx, offset);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
    return {val, idx};
}

// ---------------------------------------------------------------------------
// Main kernel: one block per token
// ---------------------------------------------------------------------------
__global__ void topk_softmax_kernel(
    const __hip_bfloat16* __restrict__ router_logits,
    int32_t*             __restrict__ expert_indices,
    float*               __restrict__ expert_weights,
    int32_t*             __restrict__ tokens_per_expert,
    int num_tokens,
    int num_experts,
    int top_k,
    int gpu_id,
    int experts_per_gpu)
{
    const int token_id = blockIdx.x;
    if (token_id >= num_tokens) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    extern __shared__ char smem_raw[];
    float* smem_logits     = reinterpret_cast<float*>(smem_raw);
    float* smem_warp_vals  = smem_logits + num_experts;
    int*   smem_warp_idxs  = reinterpret_cast<int*>(smem_warp_vals + NUM_WARPS_PER_BLOCK * MAX_TOP_K);
    float* smem_final_vals = reinterpret_cast<float*>(smem_warp_idxs + NUM_WARPS_PER_BLOCK * MAX_TOP_K);
    int*   smem_final_idxs = reinterpret_cast<int*>(smem_final_vals + MAX_TOP_K);

    // Step 1: Load BF16 logits into shared memory as FP32
    const __hip_bfloat16* token_logits = router_logits + static_cast<int64_t>(token_id) * num_experts;
    for (int i = tid; i < num_experts; i += THREADS_PER_BLOCK) {
        smem_logits[i] = sanitize_logit(__bfloat162float(token_logits[i]));
    }
    __syncthreads();

    // Step 2: Each warp finds its local top-k from its slice of experts
    // With WARP_SIZE=64 and 2 warps: chunk=64, each lane gets 1 expert
    const int chunk = (num_experts + NUM_WARPS_PER_BLOCK - 1) / NUM_WARPS_PER_BLOCK;
    const int warp_start = warp_id * chunk;
    const int warp_end   = min(warp_start + chunk, num_experts);

    // MAX_PER_LANE=1: with 64 lanes per warp and chunk=64, each lane handles exactly 1 expert
    static constexpr int MAX_PER_LANE = 1;
    float local_vals[MAX_PER_LANE];
    int   local_idxs[MAX_PER_LANE];

    #pragma unroll
    for (int j = 0; j < MAX_PER_LANE; ++j) {
        int expert_id = warp_start + lane_id + j * WARP_SIZE;
        if (expert_id < warp_end) {
            local_vals[j] = smem_logits[expert_id];
            local_idxs[j] = expert_id;
        } else {
            local_vals[j] = -FLT_MAX;
            local_idxs[j] = -1;
        }
    }

    // Iterative selection: find top_k from this warp's slice
    for (int k_iter = 0; k_iter < top_k; ++k_iter) {
        float best_val = -FLT_MAX;
        int   best_idx = -1;
        #pragma unroll
        for (int j = 0; j < MAX_PER_LANE; ++j) {
            if (local_vals[j] > best_val) {
                best_val = local_vals[j];
                best_idx = local_idxs[j];
            }
        }

        ValIdx result = warp_reduce_max(best_val, best_idx);

        if (lane_id == 0) {
            smem_warp_vals[warp_id * MAX_TOP_K + k_iter] = result.val;
            smem_warp_idxs[warp_id * MAX_TOP_K + k_iter] = result.idx;
        }

        // Invalidate the selected expert
        int winner_idx = __shfl(result.idx, 0);
        #pragma unroll
        for (int j = 0; j < MAX_PER_LANE; ++j) {
            if (local_idxs[j] == winner_idx) {
                local_vals[j] = -FLT_MAX;
            }
        }
    }
    __syncthreads();

    // Step 3: Merge warp-local top-k into global top-k (single warp)
    // total_candidates = 2 * top_k = 8, fits easily in 64-wide warp
    if (warp_id == 0) {
        const int total_candidates = NUM_WARPS_PER_BLOCK * top_k;

        float cand_val = -FLT_MAX;
        int   cand_idx = -1;
        if (lane_id < total_candidates) {
            int warp_idx = lane_id / top_k;
            int k_idx    = lane_id % top_k;
            cand_val = smem_warp_vals[warp_idx * MAX_TOP_K + k_idx];
            cand_idx = smem_warp_idxs[warp_idx * MAX_TOP_K + k_idx];
        }

        for (int k_iter = 0; k_iter < top_k; ++k_iter) {
            ValIdx result = warp_reduce_max(cand_val, cand_idx);
            if (lane_id == 0) {
                smem_final_vals[k_iter] = result.val;
                smem_final_idxs[k_iter] = result.idx;
            }
            int winner_idx = __shfl(result.idx, 0);
            if (cand_idx == winner_idx) {
                cand_val = -FLT_MAX;
            }
        }
    }
    __syncthreads();

    // Step 4: Softmax over top-k selected values only
    if (warp_id == 0 && lane_id == 0) {
        float max_val = -FLT_MAX;
        for (int k = 0; k < top_k; ++k) {
            if (smem_final_vals[k] > max_val) {
                max_val = smem_final_vals[k];
            }
        }

        float sum_exp = 0.0f;
        float exp_vals[MAX_TOP_K];
        for (int k = 0; k < top_k; ++k) {
            exp_vals[k] = expf(smem_final_vals[k] - max_val);
            sum_exp += exp_vals[k];
        }

        float inv_sum = 1.0f / sum_exp;
        int32_t* out_indices = expert_indices + static_cast<int64_t>(token_id) * top_k;
        float*   out_weights = expert_weights + static_cast<int64_t>(token_id) * top_k;

        for (int k = 0; k < top_k; ++k) {
            out_indices[k] = smem_final_idxs[k];
            out_weights[k] = exp_vals[k] * inv_sum;
            atomicAdd(&tokens_per_expert[smem_final_idxs[k]], 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Optimized kernel: TOKENS_PER_BLOCK tokens per block
// ---------------------------------------------------------------------------
__global__ void topk_softmax_kernel_v2(
    const __hip_bfloat16* __restrict__ router_logits,
    int32_t*             __restrict__ expert_indices,
    float*               __restrict__ expert_weights,
    int32_t*             __restrict__ tokens_per_expert,
    int num_tokens,
    int num_experts,
    int top_k,
    int gpu_id,
    int experts_per_gpu)
{
    const int block_token_start = blockIdx.x * TOKENS_PER_BLOCK;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    extern __shared__ char smem_raw[];

    float* smem_logits     = reinterpret_cast<float*>(smem_raw);
    float* smem_warp_vals  = smem_logits + TOKENS_PER_BLOCK * num_experts;
    int*   smem_warp_idxs  = reinterpret_cast<int*>(smem_warp_vals + TOKENS_PER_BLOCK * NUM_WARPS_PER_BLOCK * MAX_TOP_K);
    float* smem_final_vals = reinterpret_cast<float*>(smem_warp_idxs + TOKENS_PER_BLOCK * NUM_WARPS_PER_BLOCK * MAX_TOP_K);
    int*   smem_final_idxs = reinterpret_cast<int*>(smem_final_vals + TOKENS_PER_BLOCK * MAX_TOP_K);

    for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
        const int token_id = block_token_start + t;
        if (token_id >= num_tokens) break;

        float* tok_logits     = smem_logits + t * num_experts;
        float* tok_warp_vals  = smem_warp_vals + t * NUM_WARPS_PER_BLOCK * MAX_TOP_K;
        int*   tok_warp_idxs  = smem_warp_idxs + t * NUM_WARPS_PER_BLOCK * MAX_TOP_K;
        float* tok_final_vals = smem_final_vals + t * MAX_TOP_K;
        int*   tok_final_idxs = smem_final_idxs + t * MAX_TOP_K;

        // Step 1: Load logits
        const __hip_bfloat16* token_logits = router_logits + static_cast<int64_t>(token_id) * num_experts;
        for (int i = tid; i < num_experts; i += THREADS_PER_BLOCK) {
            tok_logits[i] = sanitize_logit(__bfloat162float(token_logits[i]));
        }
        __syncthreads();

        // Step 2: Each warp finds local top-k
        const int chunk = (num_experts + NUM_WARPS_PER_BLOCK - 1) / NUM_WARPS_PER_BLOCK;
        const int warp_start = warp_id * chunk;
        const int warp_end   = min(warp_start + chunk, num_experts);

        static constexpr int MAX_PER_LANE = 1;
        float local_vals[MAX_PER_LANE];
        int   local_idxs[MAX_PER_LANE];

        #pragma unroll
        for (int j = 0; j < MAX_PER_LANE; ++j) {
            int expert_id = warp_start + lane_id + j * WARP_SIZE;
            if (expert_id < warp_end) {
                local_vals[j] = tok_logits[expert_id];
                local_idxs[j] = expert_id;
            } else {
                local_vals[j] = -FLT_MAX;
                local_idxs[j] = -1;
            }
        }

        for (int k_iter = 0; k_iter < top_k; ++k_iter) {
            float best_val = -FLT_MAX;
            int   best_idx = -1;
            #pragma unroll
            for (int j = 0; j < MAX_PER_LANE; ++j) {
                if (local_vals[j] > best_val) {
                    best_val = local_vals[j];
                    best_idx = local_idxs[j];
                }
            }

            ValIdx result = warp_reduce_max(best_val, best_idx);

            if (lane_id == 0) {
                tok_warp_vals[warp_id * MAX_TOP_K + k_iter] = result.val;
                tok_warp_idxs[warp_id * MAX_TOP_K + k_iter] = result.idx;
            }

            int winner_idx = __shfl(result.idx, 0);
            #pragma unroll
            for (int j = 0; j < MAX_PER_LANE; ++j) {
                if (local_idxs[j] == winner_idx) {
                    local_vals[j] = -FLT_MAX;
                }
            }
        }
        __syncthreads();

        // Step 3: Merge warp-local top-k (single warp)
        if (warp_id == 0) {
            const int total_candidates = NUM_WARPS_PER_BLOCK * top_k;
            float cand_val = -FLT_MAX;
            int   cand_idx = -1;
            if (lane_id < total_candidates) {
                int warp_idx = lane_id / top_k;
                int k_idx    = lane_id % top_k;
                cand_val = tok_warp_vals[warp_idx * MAX_TOP_K + k_idx];
                cand_idx = tok_warp_idxs[warp_idx * MAX_TOP_K + k_idx];
            }

            for (int k_iter = 0; k_iter < top_k; ++k_iter) {
                ValIdx result = warp_reduce_max(cand_val, cand_idx);
                if (lane_id == 0) {
                    tok_final_vals[k_iter] = result.val;
                    tok_final_idxs[k_iter] = result.idx;
                }
                int winner_idx = __shfl(result.idx, 0);
                if (cand_idx == winner_idx) {
                    cand_val = -FLT_MAX;
                }
            }
        }
        __syncthreads();

        // Step 4: Softmax over top-k
        if (warp_id == 0 && lane_id < top_k) {
            bool degenerate = (tok_final_idxs[0] == -1);

            if (degenerate) {
                int32_t* out_indices = expert_indices + static_cast<int64_t>(token_id) * top_k;
                float*   out_weights = expert_weights + static_cast<int64_t>(token_id) * top_k;
                int base = gpu_id * experts_per_gpu;
                out_indices[lane_id] = base + lane_id;
                out_weights[lane_id] = 1.0f / top_k;
            } else {
                float max_val = -FLT_MAX;
                for (int k = 0; k < top_k; ++k) {
                    if (tok_final_vals[k] > max_val) {
                        max_val = tok_final_vals[k];
                    }
                }

                float my_exp = expf(tok_final_vals[lane_id] - max_val);

                // Warp-level sum reduction across top_k lanes
                float sum_exp = my_exp;
                #pragma unroll
                for (int offset = top_k / 2; offset > 0; offset >>= 1) {
                    sum_exp += __shfl_xor(sum_exp, offset);
                }

                float inv_sum = 1.0f / sum_exp;

                int32_t* out_indices = expert_indices + static_cast<int64_t>(token_id) * top_k;
                float*   out_weights = expert_weights + static_cast<int64_t>(token_id) * top_k;

                out_indices[lane_id] = tok_final_idxs[lane_id];
                out_weights[lane_id] = my_exp * inv_sum;
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------
void topk_softmax_forward(
    const __hip_bfloat16* router_logits,
    int32_t*             expert_indices,
    float*               expert_weights,
    int32_t*             tokens_per_expert,
    int num_tokens,
    int num_experts,
    int top_k,
    int gpu_id,
    int experts_per_gpu,
    hipStream_t stream)
{
    if (num_tokens == 0) return;

    if (top_k > MAX_TOP_K) {
        fprintf(stderr, "topk_softmax: top_k=%d exceeds MAX_TOP_K=%d\n", top_k, MAX_TOP_K);
        abort();
    }

    if (num_tokens <= TOKENS_PER_BLOCK) {
        // V1: one block per token
        size_t smem_v1 = num_experts * sizeof(float)
                       + NUM_WARPS_PER_BLOCK * MAX_TOP_K * sizeof(float)
                       + NUM_WARPS_PER_BLOCK * MAX_TOP_K * sizeof(int)
                       + MAX_TOP_K * sizeof(float)
                       + MAX_TOP_K * sizeof(int);

        dim3 grid(num_tokens);
        dim3 block(THREADS_PER_BLOCK);

        topk_softmax_kernel<<<grid, block, smem_v1, stream>>>(
            router_logits, expert_indices, expert_weights,
            tokens_per_expert, num_tokens, num_experts, top_k,
            gpu_id, experts_per_gpu);
    } else {
        // V2: TOKENS_PER_BLOCK tokens per block (throughput path)
        size_t smem_v2 = TOKENS_PER_BLOCK * num_experts * sizeof(float)
                       + TOKENS_PER_BLOCK * NUM_WARPS_PER_BLOCK * MAX_TOP_K * sizeof(float)
                       + TOKENS_PER_BLOCK * NUM_WARPS_PER_BLOCK * MAX_TOP_K * sizeof(int)
                       + TOKENS_PER_BLOCK * MAX_TOP_K * sizeof(float)
                       + TOKENS_PER_BLOCK * MAX_TOP_K * sizeof(int);

        dim3 grid(cdiv(num_tokens, TOKENS_PER_BLOCK));
        dim3 block(THREADS_PER_BLOCK);

        topk_softmax_kernel_v2<<<grid, block, smem_v2, stream>>>(
            router_logits, expert_indices, expert_weights,
            tokens_per_expert, num_tokens, num_experts, top_k,
            gpu_id, experts_per_gpu);
    }

    CUDA_CHECK(hipGetLastError());
}

} // namespace gptoss
