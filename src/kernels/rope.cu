// RoPE (Rotary Position Embeddings) with YaRN Extension for GPT-OSS-120B
// NTK-by-parts interpolation: factor=32, original_max=4096, extended to 131072
// theta=150000, head_dim=64, partial_rotary_factor=1.0
//
// HIP port for AMD MI300X (CDNA3, wavefront64)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>

#include "config.h"
#include "hip_compat.h"
#include "cuda_utils.h"

namespace gptoss {

// ============================================================================
// YaRN NTK-by-parts: compute per-dimension frequency scaling factors
//
// Matches HuggingFace's implementation:
//   1. yarn_find_correction_dim(num_rotations, dim, base, max_pos):
//      returns (dim * log(max_pos / (num_rotations * 2*pi))) / (2 * log(base))
//
//   2. yarn_find_correction_range(beta_fast, beta_slow, dim, base, max_pos):
//      low  = floor(yarn_find_correction_dim(beta_fast, dim, base, max_pos))
//      high = ceil(yarn_find_correction_dim(beta_slow, dim, base, max_pos))
//
//   3. Linear ramp mask from dim_index=low to dim_index=high:
//      ramp(d) = clamp((d - low) / (high - low), 0, 1)
//      inv_freq_mask = 1 - ramp
//      inv_freq = freq_inter * (1 - mask) + freq_extra * mask
//
// For GPT-OSS-120B: beta_fast=32, beta_slow=1, factor=32,
//   theta=150000, head_dim=64, original_max_pos=4096
//   low=8, high=18 (dim indices into half_dim=32 pairs)
// ============================================================================

// Kernel to precompute cos/sin tables for all positions
// Output: cos_table[max_seq_len * half_dim], sin_table[max_seq_len * half_dim]
// where half_dim = head_dim / 2
__global__ void rope_precompute_freqs_kernel(
    float* __restrict__ cos_table,
    float* __restrict__ sin_table,
    int max_seq_len,
    int half_dim,
    int head_dim,
    float theta,
    float factor,
    int original_max_pos,
    int correction_dim_low,
    int correction_dim_high)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int d   = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos >= max_seq_len || d >= half_dim) return;

    // Compute base inverse frequency for dimension d
    float dim_ratio = static_cast<float>(2 * d) / static_cast<float>(head_dim);
    float inv_freq_extra = 1.0f / powf(theta, dim_ratio);  // original (no interp)
    float inv_freq_inter = inv_freq_extra / factor;          // fully interpolated

    // YaRN NTK-by-parts: linear ramp in dimension index space
    // ramp goes 0->1 from correction_dim_low to correction_dim_high
    float ramp;
    if (correction_dim_low == correction_dim_high) {
        ramp = (d < correction_dim_low) ? 0.0f : 1.0f;
    } else {
        ramp = (static_cast<float>(d) - static_cast<float>(correction_dim_low))
             / (static_cast<float>(correction_dim_high) - static_cast<float>(correction_dim_low));
        ramp = fminf(1.0f, fmaxf(0.0f, ramp));
    }

    // inv_freq_mask = 1 - ramp
    // inv_freq = freq_inter * (1 - mask) + freq_extra * mask
    //          = freq_inter * ramp + freq_extra * (1 - ramp)
    float mask = 1.0f - ramp;
    float corrected_inv_freq = inv_freq_inter * (1.0f - mask) + inv_freq_extra * mask;

    // angle = position * corrected_inv_freq
    float angle = static_cast<float>(pos) * corrected_inv_freq;

    // YaRN mscale: scale cos/sin by attention_factor (matches HF behavior)
    // This effectively scales Q_rot and K_rot, so QK logits scale by mscale^2
    float mscale = ModelConfig::rope_yarn_mscale;

    int idx = pos * half_dim + d;
    cos_table[idx] = cosf(angle) * mscale;
    sin_table[idx] = sinf(angle) * mscale;
}

// Interleaved precompute kernel: outputs float2 = {cos, sin} per (pos, d) pair.
// One 64-bit load replaces two separate 32-bit loads at apply time.
// Output: cos_sin_table[max_seq_len * half_dim] where each float2 = {cos, sin}
__global__ void rope_precompute_freqs_interleaved_kernel(
    float2* __restrict__ cos_sin_table,
    int max_seq_len,
    int half_dim,
    int head_dim,
    float theta,
    float factor,
    int original_max_pos,
    int correction_dim_low,
    int correction_dim_high)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int d   = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos >= max_seq_len || d >= half_dim) return;

    // Compute base inverse frequency for dimension d
    float dim_ratio = static_cast<float>(2 * d) / static_cast<float>(head_dim);
    float inv_freq_extra = 1.0f / powf(theta, dim_ratio);
    float inv_freq_inter = inv_freq_extra / factor;

    // YaRN NTK-by-parts: linear ramp in dimension index space
    float ramp;
    if (correction_dim_low == correction_dim_high) {
        ramp = (d < correction_dim_low) ? 0.0f : 1.0f;
    } else {
        ramp = (static_cast<float>(d) - static_cast<float>(correction_dim_low))
             / (static_cast<float>(correction_dim_high) - static_cast<float>(correction_dim_low));
        ramp = fminf(1.0f, fmaxf(0.0f, ramp));
    }

    float mask = 1.0f - ramp;
    float corrected_inv_freq = inv_freq_inter * (1.0f - mask) + inv_freq_extra * mask;

    float angle = static_cast<float>(pos) * corrected_inv_freq;

    // YaRN mscale baked into cos/sin
    float mscale = ModelConfig::rope_yarn_mscale;

    int idx = pos * half_dim + d;
    cos_sin_table[idx] = make_float2(cosf(angle) * mscale, sinf(angle) * mscale);
}

// HF-compatible yarn_find_correction_dim:
//   (dim * log(max_pos / (num_rotations * 2*pi))) / (2 * log(base))
static int yarn_find_correction_dim(float num_rotations, int dim, float base, int max_pos) {
    float val = (static_cast<float>(dim) * logf(static_cast<float>(max_pos) / (num_rotations * 2.0f * M_PI)))
              / (2.0f * logf(base));
    return static_cast<int>(val);
}

// Initialize cos/sin lookup tables on GPU
// Allocates and fills cos_table and sin_table of shape [max_seq_len, head_dim/2]
void rope_init(
    float** cos_table,
    float** sin_table,
    float2** cos_sin_table_interleaved,
    int max_seq_len,
    int head_dim,
    float theta,
    hipStream_t stream)
{
    int half_dim = head_dim / 2;
    size_t table_bytes = static_cast<size_t>(max_seq_len) * half_dim * sizeof(float);

    // Compute YaRN correction dim range (matches HF yarn_find_correction_range)
    float beta_fast = ModelConfig::rope_yarn_beta_fast;
    float beta_slow = ModelConfig::rope_yarn_beta_slow;
    int original_max_pos = ModelConfig::rope_original_max_pos;

    // low from beta_fast (high rotation count -> low dim index)
    int correction_dim_low = yarn_find_correction_dim(beta_fast, head_dim, theta, original_max_pos);
    // high from beta_slow (low rotation count -> high dim index)
    float high_f = (static_cast<float>(head_dim) * logf(static_cast<float>(original_max_pos) / (beta_slow * 2.0f * M_PI)))
                 / (2.0f * logf(theta));
    int correction_dim_high = static_cast<int>(ceilf(high_f));

    // Clamp to valid range [0, half_dim-1]
    correction_dim_low  = (correction_dim_low < 0) ? 0 : (correction_dim_low >= half_dim ? half_dim - 1 : correction_dim_low);
    correction_dim_high = (correction_dim_high < 0) ? 0 : (correction_dim_high >= half_dim ? half_dim - 1 : correction_dim_high);

    fprintf(stderr, "[RoPE] YaRN correction range: dim %d to %d (of %d pairs), "
            "beta_fast=%.1f, beta_slow=%.1f, factor=%d, theta=%.0f, orig_max_pos=%d\n",
            correction_dim_low, correction_dim_high, half_dim,
            beta_fast, beta_slow, ModelConfig::rope_yarn_factor, theta, original_max_pos);

    CUDA_CHECK(hipMalloc(cos_table, table_bytes));
    CUDA_CHECK(hipMalloc(sin_table, table_bytes));

    // Launch 2D grid: x=positions, y=dimensions
    dim3 threads(32, 8); // 256 threads per block
    dim3 blocks(
        cdiv(max_seq_len, threads.x),
        cdiv(half_dim, threads.y)
    );

    rope_precompute_freqs_kernel<<<blocks, threads, 0, stream>>>(
        *cos_table, *sin_table,
        max_seq_len, half_dim, head_dim, theta,
        static_cast<float>(ModelConfig::rope_yarn_factor),
        ModelConfig::rope_original_max_pos,
        correction_dim_low, correction_dim_high
    );

    CUDA_CHECK(hipGetLastError());

    // Allocate and fill interleaved cos_sin table: float2 = {cos, sin} per entry
    size_t interleaved_bytes = static_cast<size_t>(max_seq_len) * half_dim * sizeof(float2);
    CUDA_CHECK(hipMalloc(cos_sin_table_interleaved, interleaved_bytes));

    rope_precompute_freqs_interleaved_kernel<<<blocks, threads, 0, stream>>>(
        *cos_sin_table_interleaved,
        max_seq_len, half_dim, head_dim, theta,
        static_cast<float>(ModelConfig::rope_yarn_factor),
        ModelConfig::rope_original_max_pos,
        correction_dim_low, correction_dim_high
    );

    CUDA_CHECK(hipGetLastError());
}

// ============================================================================
// Runtime RoPE application kernel
//
// For each (token, head, dim_pair):
//   q_rot = q[..., 2d] * cos - q[..., 2d+1] * sin
//   q_out[..., 2d]   = q_rot_real
//   q_out[..., 2d+1] = q[..., 2d] * sin + q[..., 2d+1] * cos
//
// Input layouts (row-major):
//   q: [num_tokens, num_q_heads, head_dim]
//   k: [num_tokens, num_kv_heads, head_dim]
//   positions: [num_tokens] - position index for each token
//   cos_table: [max_seq_len, half_dim]
//   sin_table: [max_seq_len, half_dim]
// ============================================================================

__global__ void rope_apply_kernel(
    const __hip_bfloat16* __restrict__ q,
    const __hip_bfloat16* __restrict__ k,
    __hip_bfloat16* __restrict__ q_out,
    __hip_bfloat16* __restrict__ k_out,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int* __restrict__ positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim)
{
    int half_dim = head_dim / 2;

    int total_q_items = num_tokens * num_q_heads * half_dim;
    int total_k_items = num_tokens * num_kv_heads * half_dim;
    int total_items = total_q_items + total_k_items;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_items) return;

    bool is_q = (idx < total_q_items);

    int local_idx, num_heads;
    const __hip_bfloat16* src;
    __hip_bfloat16* dst;

    if (is_q) {
        local_idx = idx;
        num_heads = num_q_heads;
        src = q;
        dst = q_out;
    } else {
        local_idx = idx - total_q_items;
        num_heads = num_kv_heads;
        src = k;
        dst = k_out;
    }

    // Decompose linear index into (token, head, d)
    int d = local_idx % half_dim;
    int remainder = local_idx / half_dim;
    int head = remainder % num_heads;
    int token = remainder / num_heads;

    // Fetch position for this token
    int pos = positions[token];

    // Load cos/sin for (pos, d)
    int table_idx = pos * half_dim + d;
    float cos_val = cos_table[table_idx];
    float sin_val = sin_table[table_idx];

    // Load the pair of values to rotate (half-split convention:
    // first half paired with second half, matching reference impl)
    int base_offset = (token * num_heads + head) * head_dim;
    float x0 = __bfloat162float(src[base_offset + d]);
    float x1 = __bfloat162float(src[base_offset + half_dim + d]);

    // Apply rotation
    float out0 = x0 * cos_val - x1 * sin_val;
    float out1 = x0 * sin_val + x1 * cos_val;

    // Store
    dst[base_offset + d]            = __float2bfloat16(out0);
    dst[base_offset + half_dim + d] = __float2bfloat16(out1);
}

// Optimized version: separate Q and K kernels to avoid branch divergence

__global__ void rope_apply_q_kernel(
    const __hip_bfloat16* __restrict__ q,
    __hip_bfloat16* __restrict__ q_out,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int* __restrict__ positions,
    int num_tokens,
    int num_q_heads,
    int head_dim)
{
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * num_q_heads * half_dim;
    if (idx >= total) return;

    int d = idx % half_dim;
    int remainder = idx / half_dim;
    int head = remainder % num_q_heads;
    int token = remainder / num_q_heads;

    int pos = positions[token];
    int table_idx = pos * half_dim + d;
    float cos_val = cos_table[table_idx];
    float sin_val = sin_table[table_idx];

    int base = (token * num_q_heads + head) * head_dim;
    float x0 = __bfloat162float(q[base + d]);
    float x1 = __bfloat162float(q[base + half_dim + d]);

    q_out[base + d]            = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    q_out[base + half_dim + d] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}

__global__ void rope_apply_k_kernel(
    const __hip_bfloat16* __restrict__ k,
    __hip_bfloat16* __restrict__ k_out,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int* __restrict__ positions,
    int num_tokens,
    int num_kv_heads,
    int head_dim)
{
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * num_kv_heads * half_dim;
    if (idx >= total) return;

    int d = idx % half_dim;
    int remainder = idx / half_dim;
    int head = remainder % num_kv_heads;
    int token = remainder / num_kv_heads;

    int pos = positions[token];
    int table_idx = pos * half_dim + d;
    float cos_val = cos_table[table_idx];
    float sin_val = sin_table[table_idx];

    int base = (token * num_kv_heads + head) * head_dim;
    float x0 = __bfloat162float(k[base + d]);
    float x1 = __bfloat162float(k[base + half_dim + d]);

    k_out[base + d]            = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    k_out[base + half_dim + d] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}

// Half-split Q kernel: loads x[d] and x[d+half_dim] as the rotation pair,
// matching the reference JAX/Flax implementation's split-in-half convention.
// Uses the interleaved cos_sin_table (single 64-bit load gets both cos and sin).
__global__ void rope_apply_q_kernel_vec(
    const __hip_bfloat16* __restrict__ q,
    __hip_bfloat16* __restrict__ q_out,
    const float2* __restrict__ cos_sin_table,
    const int* __restrict__ positions,
    int num_tokens,
    int num_q_heads,
    int head_dim)
{
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * num_q_heads * half_dim;
    if (idx >= total) return;

    int d = idx % half_dim;
    int remainder = idx / half_dim;
    int head = remainder % num_q_heads;
    int token = remainder / num_q_heads;

    int pos = positions[token];

    // Single 64-bit load for both cos and sin
    float2 cs = cos_sin_table[pos * half_dim + d];
    float cos_val = cs.x;
    float sin_val = cs.y;

    // Half-split: load x[d] and x[d+half_dim] (not adjacent, so scalar loads)
    int base = (token * num_q_heads + head) * head_dim;
    float x0 = __bfloat162float(q[base + d]);
    float x1 = __bfloat162float(q[base + half_dim + d]);

    float out0 = x0 * cos_val - x1 * sin_val;
    float out1 = x0 * sin_val + x1 * cos_val;

    q_out[base + d]            = __float2bfloat16(out0);
    q_out[base + half_dim + d] = __float2bfloat16(out1);
}

// Vectorized K kernel: same optimization as Q kernel but for K heads
__global__ void rope_apply_k_kernel_vec(
    const __hip_bfloat16* __restrict__ k,
    __hip_bfloat16* __restrict__ k_out,
    const float2* __restrict__ cos_sin_table,
    const int* __restrict__ positions,
    int num_tokens,
    int num_kv_heads,
    int head_dim)
{
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * num_kv_heads * half_dim;
    if (idx >= total) return;

    int d = idx % half_dim;
    int remainder = idx / half_dim;
    int head = remainder % num_kv_heads;
    int token = remainder / num_kv_heads;

    int pos = positions[token];

    // Single 64-bit load for both cos and sin
    float2 cs = cos_sin_table[pos * half_dim + d];
    float cos_val = cs.x;
    float sin_val = cs.y;

    // Half-split: load x[d] and x[d+half_dim] (not adjacent, so scalar loads)
    int base = (token * num_kv_heads + head) * head_dim;
    float x0 = __bfloat162float(k[base + d]);
    float x1 = __bfloat162float(k[base + half_dim + d]);

    float out0 = x0 * cos_val - x1 * sin_val;
    float out1 = x0 * sin_val + x1 * cos_val;

    k_out[base + d]            = __float2bfloat16(out0);
    k_out[base + half_dim + d] = __float2bfloat16(out1);
}

// Launch function: applies RoPE to Q and K tensors
// Uses vectorized kernels with interleaved cos_sin table when available,
// falls back to scalar kernels with separate cos/sin tables otherwise.
void rope_forward(
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
    hipStream_t stream)
{
    if (num_tokens == 0) return;

    int half_dim = head_dim / 2;
    constexpr int THREADS = 256;

    if (cos_sin_table != nullptr) {
        // Vectorized path: interleaved cos/sin table
        if (q != nullptr && q_out != nullptr) {
            int total_q = num_tokens * num_q_heads * half_dim;
            int blocks_q = cdiv(total_q, THREADS);
            rope_apply_q_kernel_vec<<<blocks_q, THREADS, 0, stream>>>(
                q, q_out, cos_sin_table, positions,
                num_tokens, num_q_heads, head_dim);
        }

        {
            int total_k = num_tokens * num_kv_heads * half_dim;
            int blocks_k = cdiv(total_k, THREADS);
            rope_apply_k_kernel_vec<<<blocks_k, THREADS, 0, stream>>>(
                k, k_out, cos_sin_table, positions,
                num_tokens, num_kv_heads, head_dim);
        }
    } else {
        // Fallback: separate cos/sin tables with scalar loads
        if (q != nullptr && q_out != nullptr) {
            int total_q = num_tokens * num_q_heads * half_dim;
            int blocks_q = cdiv(total_q, THREADS);
            rope_apply_q_kernel<<<blocks_q, THREADS, 0, stream>>>(
                q, q_out, cos_table, sin_table, positions,
                num_tokens, num_q_heads, head_dim);
        }

        {
            int total_k = num_tokens * num_kv_heads * half_dim;
            int blocks_k = cdiv(total_k, THREADS);
            rope_apply_k_kernel<<<blocks_k, THREADS, 0, stream>>>(
                k, k_out, cos_table, sin_table, positions,
                num_tokens, num_kv_heads, head_dim);
        }
    }

    CUDA_CHECK(hipGetLastError());
}

} // namespace gptoss
