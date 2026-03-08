// Fused RMSNorm Kernel for GPT-OSS-120B — HIP port for MI300X
// Warp size changed from 32 → 64 (wavefront64)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cfloat>

#include "config.h"
#include "hip_compat.h"
#include "cuda_utils.h"

namespace gptoss {

static constexpr int WARP_SIZE_RMS = 64; // MI300X wavefront

static __device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE_RMS / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

static __device__ float block_reduce_sum(float val) {
    __shared__ float shared[16]; // max 1024/64 = 16 warps

    int lane = threadIdx.x % WARP_SIZE_RMS;
    int warp_id = threadIdx.x / WARP_SIZE_RMS;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE_RMS - 1) / WARP_SIZE_RMS;
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }

    return val;
}

__global__ void rmsnorm_kernel(
    const __hip_bfloat16* __restrict__ input,
    const __hip_bfloat16* __restrict__ weight,
    __hip_bfloat16* __restrict__ output,
    int hidden_size,
    float eps)
{
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    const __hip_bfloat16* x = input  + static_cast<int64_t>(token_idx) * hidden_size;
    __hip_bfloat16*       o = output + static_cast<int64_t>(token_idx) * hidden_size;

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(x[i]);
        sum_sq += val * val;
    }

    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        float mean_sq = sum_sq / static_cast<float>(hidden_size);
        s_inv_rms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(x[i]);
        float w   = __bfloat162float(weight[i]);
        float normed = val * inv_rms * w;
        o[i] = __float2bfloat16(normed);
    }
}

__global__ void rmsnorm_kernel_vec2(
    const __hip_bfloat16* __restrict__ input,
    const __hip_bfloat16* __restrict__ weight,
    __hip_bfloat16* __restrict__ output,
    int hidden_size,
    float eps)
{
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    int half_hidden = hidden_size / 2;

    const __hip_bfloat162* x_vec = reinterpret_cast<const __hip_bfloat162*>(
        input + static_cast<int64_t>(token_idx) * hidden_size);
    const __hip_bfloat162* w_vec = reinterpret_cast<const __hip_bfloat162*>(weight);
    __hip_bfloat162* o_vec = reinterpret_cast<__hip_bfloat162*>(
        output + static_cast<int64_t>(token_idx) * hidden_size);

    float sum_sq = 0.0f;
    for (int i = tid; i < half_hidden; i += blockDim.x) {
        __hip_bfloat162 xv = x_vec[i];
        float x0 = __bfloat162float(__low2bfloat16(xv));
        float x1 = __bfloat162float(__high2bfloat16(xv));
        sum_sq += x0 * x0 + x1 * x1;
    }

    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(input[static_cast<int64_t>(token_idx) * hidden_size + hidden_size - 1]);
        sum_sq += val * val;
    }

    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        float mean_sq = sum_sq / static_cast<float>(hidden_size);
        s_inv_rms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;

    for (int i = tid; i < half_hidden; i += blockDim.x) {
        __hip_bfloat162 xv = x_vec[i];
        __hip_bfloat162 wv = w_vec[i];

        float x0 = __bfloat162float(__low2bfloat16(xv));
        float x1 = __bfloat162float(__high2bfloat16(xv));
        float w0 = __bfloat162float(__low2bfloat16(wv));
        float w1 = __bfloat162float(__high2bfloat16(wv));

        float o0 = x0 * inv_rms * w0;
        float o1 = x1 * inv_rms * w1;

        o_vec[i] = __halves2bfloat162(__float2bfloat16(o0), __float2bfloat16(o1));
    }

    if ((hidden_size & 1) && tid == 0) {
        int idx = static_cast<int64_t>(token_idx) * hidden_size + hidden_size - 1;
        float val = __bfloat162float(input[idx]);
        float w   = __bfloat162float(weight[hidden_size - 1]);
        output[idx] = __float2bfloat16(val * inv_rms * w);
    }
}

__global__ void rmsnorm_kernel_vec4(
    const __hip_bfloat16* __restrict__ input,
    const __hip_bfloat16* __restrict__ weight,
    __hip_bfloat16* __restrict__ output,
    int hidden_size,
    float eps)
{
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    int vec_count = hidden_size / 8;

    const int4* x_vec = reinterpret_cast<const int4*>(
        input + static_cast<int64_t>(token_idx) * hidden_size);
    const int4* w_vec = reinterpret_cast<const int4*>(weight);
    int4* o_vec = reinterpret_cast<int4*>(
        output + static_cast<int64_t>(token_idx) * hidden_size);

    static constexpr int MAX_VECS_PER_THREAD = 4;
    float cached_vals[MAX_VECS_PER_THREAD * 8];
    int num_vecs_this_thread = 0;

    float sum_sq = 0.0f;
    for (int i = tid; i < vec_count; i += blockDim.x) {
        int local_idx = num_vecs_this_thread;
        num_vecs_this_thread++;

        int4 xv = x_vec[i];
        const __hip_bfloat162* pairs = reinterpret_cast<const __hip_bfloat162*>(&xv);

        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float v0 = __bfloat162float(__low2bfloat16(pairs[p]));
            float v1 = __bfloat162float(__high2bfloat16(pairs[p]));
            cached_vals[local_idx * 8 + p * 2]     = v0;
            cached_vals[local_idx * 8 + p * 2 + 1] = v1;
            sum_sq += v0 * v0 + v1 * v1;
        }
    }

    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        float mean_sq = sum_sq / static_cast<float>(hidden_size);
        s_inv_rms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;

    int vec_idx = 0;
    for (int i = tid; i < vec_count; i += blockDim.x) {
        int4 wv = w_vec[i];
        const __hip_bfloat162* wpairs = reinterpret_cast<const __hip_bfloat162*>(&wv);

        __hip_bfloat162 out_pairs[4];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float w0 = __bfloat162float(__low2bfloat16(wpairs[p]));
            float w1 = __bfloat162float(__high2bfloat16(wpairs[p]));
            float o0 = cached_vals[vec_idx * 8 + p * 2]     * inv_rms * w0;
            float o1 = cached_vals[vec_idx * 8 + p * 2 + 1] * inv_rms * w1;
            out_pairs[p] = __halves2bfloat162(__float2bfloat16(o0), __float2bfloat16(o1));
        }

        int4 out_vec;
        memcpy(&out_vec, out_pairs, sizeof(int4));
        o_vec[i] = out_vec;

        vec_idx++;
    }
}

__global__ void fused_residual_rmsnorm_kernel_vec4(
    const __hip_bfloat16* __restrict__ residual,
    const __hip_bfloat16* __restrict__ delta,
    const __hip_bfloat16* __restrict__ weight,
    __hip_bfloat16* __restrict__ normed_output,
    __hip_bfloat16* __restrict__ sum_output,
    int hidden_size,
    float eps)
{
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    int vec_count = hidden_size / 8;

    const int4* r_vec = reinterpret_cast<const int4*>(
        residual + static_cast<int64_t>(token_idx) * hidden_size);
    const int4* d_vec = reinterpret_cast<const int4*>(
        delta + static_cast<int64_t>(token_idx) * hidden_size);
    const int4* w_vec = reinterpret_cast<const int4*>(weight);
    int4* n_vec = reinterpret_cast<int4*>(
        normed_output + static_cast<int64_t>(token_idx) * hidden_size);
    int4* s_vec = reinterpret_cast<int4*>(
        sum_output + static_cast<int64_t>(token_idx) * hidden_size);

    static constexpr int MAX_VECS_PER_THREAD = 4;
    float cached_sums[MAX_VECS_PER_THREAD * 8];
    int num_vecs_this_thread = 0;

    float sum_sq = 0.0f;
    for (int i = tid; i < vec_count; i += blockDim.x) {
        int local_idx = num_vecs_this_thread++;

        int4 rv = r_vec[i];
        int4 dv = d_vec[i];
        const __hip_bfloat162* r_pairs = reinterpret_cast<const __hip_bfloat162*>(&rv);
        const __hip_bfloat162* d_pairs = reinterpret_cast<const __hip_bfloat162*>(&dv);

        __hip_bfloat162 sum_pairs[4];

        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float r0 = __bfloat162float(__low2bfloat16(r_pairs[p]));
            float r1 = __bfloat162float(__high2bfloat16(r_pairs[p]));
            float d0 = __bfloat162float(__low2bfloat16(d_pairs[p]));
            float d1 = __bfloat162float(__high2bfloat16(d_pairs[p]));
            float s0 = r0 + d0;
            float s1 = r1 + d1;
            cached_sums[local_idx * 8 + p * 2]     = s0;
            cached_sums[local_idx * 8 + p * 2 + 1] = s1;
            sum_sq += s0 * s0 + s1 * s1;
            sum_pairs[p] = __halves2bfloat162(__float2bfloat16(s0), __float2bfloat16(s1));
        }

        int4 sum_vec;
        memcpy(&sum_vec, sum_pairs, sizeof(int4));
        s_vec[i] = sum_vec;
    }

    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        float mean_sq = sum_sq / static_cast<float>(hidden_size);
        s_inv_rms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;

    int vec_idx = 0;
    for (int i = tid; i < vec_count; i += blockDim.x) {
        int4 wv = w_vec[i];
        const __hip_bfloat162* wpairs = reinterpret_cast<const __hip_bfloat162*>(&wv);

        __hip_bfloat162 out_pairs[4];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float w0 = __bfloat162float(__low2bfloat16(wpairs[p]));
            float w1 = __bfloat162float(__high2bfloat16(wpairs[p]));
            float o0 = cached_sums[vec_idx * 8 + p * 2]     * inv_rms * w0;
            float o1 = cached_sums[vec_idx * 8 + p * 2 + 1] * inv_rms * w1;
            out_pairs[p] = __halves2bfloat162(__float2bfloat16(o0), __float2bfloat16(o1));
        }

        int4 out_vec;
        memcpy(&out_vec, out_pairs, sizeof(int4));
        n_vec[i] = out_vec;

        vec_idx++;
    }
}

void fused_residual_rmsnorm_forward(
    const __hip_bfloat16* residual,
    const __hip_bfloat16* delta,
    const __hip_bfloat16* weight,
    __hip_bfloat16* normed_output,
    __hip_bfloat16* sum_output,
    int num_tokens,
    int hidden_size,
    float eps,
    hipStream_t stream)
{
    if (num_tokens == 0) return;

    int threads = 256;
    if (hidden_size <= 128) threads = 64; // min one wavefront
    else if (hidden_size <= 512) threads = 128;

    fused_residual_rmsnorm_kernel_vec4<<<num_tokens, threads, 0, stream>>>(
        residual, delta, weight, normed_output, sum_output, hidden_size, eps);

    CUDA_CHECK(hipGetLastError());
}

void rmsnorm_forward(
    const __hip_bfloat16* input,
    const __hip_bfloat16* weight,
    __hip_bfloat16* output,
    int num_tokens,
    int hidden_size,
    float eps,
    hipStream_t stream)
{
    if (num_tokens == 0) return;

    int threads = 256;
    if (hidden_size <= 128) threads = 64;
    else if (hidden_size <= 512) threads = 128;

    int blocks = num_tokens;

    if (hidden_size % 8 == 0) {
        rmsnorm_kernel_vec4<<<blocks, threads, 0, stream>>>(
            input, weight, output, hidden_size, eps);
    } else if (hidden_size % 2 == 0) {
        rmsnorm_kernel_vec2<<<blocks, threads, 0, stream>>>(
            input, weight, output, hidden_size, eps);
    } else {
        rmsnorm_kernel<<<blocks, threads, 0, stream>>>(
            input, weight, output, hidden_size, eps);
    }

    CUDA_CHECK(hipGetLastError());
}

} // namespace gptoss
