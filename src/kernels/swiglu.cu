// GPT-OSS-120B Gated Activation Kernel
// Matches HF GptOssExperts._apply_gate():
//   gate, up = gate_up[..., ::2], gate_up[..., 1::2]   (INTERLEAVED layout)
//   gate = gate.clamp(min=None, max=limit)              (upper-only clamp)
//   up = up.clamp(min=-limit, max=limit)                (symmetric clamp)
//   glu = gate * sigmoid(alpha * gate), alpha=1.702     (GPT-OSS variant)
//   output = (up + 1) * glu                             (residual +1 on up)
//
// Input:  [N, 5760] interleaved [g0, u0, g1, u1, ...]
// Output: [N, 2880]
//
// HIP port for AMD MI300X (CDNA3, wavefront64)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#include "config.h"
#include "hip_compat.h"
#include "cuda_utils.h"

namespace gptoss {

// GPT-OSS gate activation: gate * sigmoid(alpha * gate)
__device__ __forceinline__ float gptoss_gate_act(float gate) {
    constexpr float alpha = ModelConfig::activation_alpha;
    return gate / (1.0f + expf(-alpha * gate));
}

// Scalar kernel: reads interleaved gate/up pairs
__global__ void swiglu_kernel(
    const __hip_bfloat16* __restrict__ input,
    __hip_bfloat16* __restrict__ output,
    int num_tokens,
    int intermediate_size,
    float limit)
{
    int gate_up_size = intermediate_size * 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * intermediate_size;
    if (idx >= total) return;

    int token = idx / intermediate_size;
    int dim   = idx % intermediate_size;

    // Interleaved layout: gate at even indices, up at odd indices
    int gate_idx  = token * gate_up_size + dim * 2;
    int value_idx = token * gate_up_size + dim * 2 + 1;

    float gate = __bfloat162float(input[gate_idx]);
    float value = __bfloat162float(input[value_idx]);

    // Asymmetric clamping (matches HF):
    // gate: clamp upper only (min=None, max=limit)
    // up: clamp symmetric (-limit, limit)
    gate  = fminf(gate, limit);
    value = fminf(fmaxf(value, -limit), limit);

    // GPT-OSS activation: gate * sigmoid(alpha * gate), then (up + 1) * glu
    float glu = gptoss_gate_act(gate);
    float result = (value + 1.0f) * glu;

    output[idx] = __float2bfloat16(result);
}

// Vectorized kernel: processes 2 output elements per thread
// Reads 4 consecutive BF16 values = 2 interleaved (gate, up) pairs
__global__ void swiglu_kernel_vec2(
    const __hip_bfloat16* __restrict__ input,
    __hip_bfloat16* __restrict__ output,
    int num_tokens,
    int intermediate_size,
    float limit)
{
    int gate_up_size = intermediate_size * 2;
    int half_inter = intermediate_size / 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * half_inter;
    if (idx >= total) return;

    int token = idx / half_inter;
    int dim2  = idx % half_inter;

    // Read 2 interleaved pairs: [g0, u0, g1, u1] as a single 128-bit load
    const float2* in_base = reinterpret_cast<const float2*>(
        input + token * gate_up_size + dim2 * 4);
    __hip_bfloat162* out_base = reinterpret_cast<__hip_bfloat162*>(
        output + token * intermediate_size);

    float2 raw = in_base[0]; // 4 BF16: [g0, u0, g1, u1]

    __hip_bfloat162 pair0 = *reinterpret_cast<__hip_bfloat162*>(&raw.x); // g0, u0
    __hip_bfloat162 pair1 = *reinterpret_cast<__hip_bfloat162*>(&raw.y); // g1, u1

    float g0 = __bfloat162float(__low2bfloat16(pair0));
    float u0 = __bfloat162float(__high2bfloat16(pair0));
    float g1 = __bfloat162float(__low2bfloat16(pair1));
    float u1 = __bfloat162float(__high2bfloat16(pair1));

    // Asymmetric clamping
    g0 = fminf(g0, limit);
    g1 = fminf(g1, limit);
    u0 = fminf(fmaxf(u0, -limit), limit);
    u1 = fminf(fmaxf(u1, -limit), limit);

    // GPT-OSS activation
    float r0 = (u0 + 1.0f) * gptoss_gate_act(g0);
    float r1 = (u1 + 1.0f) * gptoss_gate_act(g1);

    out_base[dim2] = __halves2bfloat162(__float2bfloat16(r0), __float2bfloat16(r1));
}

// Vectorized kernel: processes 4 output elements per thread
// Reads 8 consecutive BF16 values = 4 interleaved (gate, up) pairs
__global__ void swiglu_kernel_vec4(
    const __hip_bfloat16* __restrict__ input,
    __hip_bfloat16* __restrict__ output,
    int num_tokens,
    int intermediate_size,
    float limit)
{
    int gate_up_size = intermediate_size * 2;
    int quarter_inter = intermediate_size / 4;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * quarter_inter;
    if (idx >= total) return;

    int token = idx / quarter_inter;
    int dim4  = idx % quarter_inter;

    // Read 4 interleaved pairs: [g0,u0,g1,u1,g2,u2,g3,u3] as two 128-bit loads
    const float2* in_base = reinterpret_cast<const float2*>(
        input + token * gate_up_size + dim4 * 8);

    float2 raw0 = in_base[0]; // g0,u0,g1,u1
    float2 raw1 = in_base[1]; // g2,u2,g3,u3

    __hip_bfloat162 p0 = *reinterpret_cast<__hip_bfloat162*>(&raw0.x);
    __hip_bfloat162 p1 = *reinterpret_cast<__hip_bfloat162*>(&raw0.y);
    __hip_bfloat162 p2 = *reinterpret_cast<__hip_bfloat162*>(&raw1.x);
    __hip_bfloat162 p3 = *reinterpret_cast<__hip_bfloat162*>(&raw1.y);

    float g[4], u[4];
    g[0] = __bfloat162float(__low2bfloat16(p0));
    u[0] = __bfloat162float(__high2bfloat16(p0));
    g[1] = __bfloat162float(__low2bfloat16(p1));
    u[1] = __bfloat162float(__high2bfloat16(p1));
    g[2] = __bfloat162float(__low2bfloat16(p2));
    u[2] = __bfloat162float(__high2bfloat16(p2));
    g[3] = __bfloat162float(__low2bfloat16(p3));
    u[3] = __bfloat162float(__high2bfloat16(p3));

    float r[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        g[i] = fminf(g[i], limit);              // gate: upper clamp only
        u[i] = fminf(fmaxf(u[i], -limit), limit); // up: symmetric clamp
        r[i] = (u[i] + 1.0f) * gptoss_gate_act(g[i]);
    }

    // Pack results
    float2* out_base = reinterpret_cast<float2*>(
        output + token * intermediate_size);

    __hip_bfloat162 r01 = __halves2bfloat162(__float2bfloat16(r[0]), __float2bfloat16(r[1]));
    __hip_bfloat162 r23 = __halves2bfloat162(__float2bfloat16(r[2]), __float2bfloat16(r[3]));

    float2 out_raw;
    out_raw.x = *reinterpret_cast<float*>(&r01);
    out_raw.y = *reinterpret_cast<float*>(&r23);
    out_base[dim4] = out_raw;
}

// Launch function
void swiglu_forward(
    const __hip_bfloat16* input,
    __hip_bfloat16* output,
    int num_tokens,
    int intermediate_size,
    float limit,
    hipStream_t stream)
{
    if (num_tokens == 0) return;

    constexpr int THREADS = 256;

    // Use the most vectorized kernel that the intermediate_size allows
    if (intermediate_size % 4 == 0) {
        int quarter_inter = intermediate_size / 4;
        int total = num_tokens * quarter_inter;
        int blocks = cdiv(total, THREADS);
        swiglu_kernel_vec4<<<blocks, THREADS, 0, stream>>>(
            input, output, num_tokens, intermediate_size, limit);
    } else if (intermediate_size % 2 == 0) {
        int half_inter = intermediate_size / 2;
        int total = num_tokens * half_inter;
        int blocks = cdiv(total, THREADS);
        swiglu_kernel_vec2<<<blocks, THREADS, 0, stream>>>(
            input, output, num_tokens, intermediate_size, limit);
    } else {
        int total = num_tokens * intermediate_size;
        int blocks = cdiv(total, THREADS);
        swiglu_kernel<<<blocks, THREADS, 0, stream>>>(
            input, output, num_tokens, intermediate_size, limit);
    }

    CUDA_CHECK(hipGetLastError());
}

} // namespace gptoss
