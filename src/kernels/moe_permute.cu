/**
 * moe_permute.cu - Dispatch-free token permutation for EP-N inference
 *
 * HIP port for AMD MI300X (CDNA3, wavefront64)
 * Minimal changes: type and API translations only (no warp-size dependencies)
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
// Kernel 1: classify_and_count_kernel
// ---------------------------------------------------------------------------
__global__ void classify_and_count_kernel(
    const int32_t* __restrict__ expert_indices,
    int32_t*       __restrict__ local_expert_counts,
    int32_t*       __restrict__ per_peer_counts,
    int total_slots,
    int gpu_id,
    int experts_per_gpu,
    int ep_size)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) return;

    const int expert_id  = expert_indices[slot];
    const int owning_ep_rank = expert_id / experts_per_gpu;

    if (owning_ep_rank == gpu_id) {
        const int local_id = expert_id - gpu_id * experts_per_gpu;
        atomicAdd(&local_expert_counts[local_id], 1);
    }
    atomicAdd(&per_peer_counts[owning_ep_rank], 1);
}

// ---------------------------------------------------------------------------
// Kernel 2: prefix_sums_kernel
// ---------------------------------------------------------------------------
__global__ void prefix_sums_kernel(
    const int32_t* __restrict__ local_expert_counts,
    const int32_t* __restrict__ per_peer_counts,
    int32_t*       __restrict__ expert_offsets,
    int32_t*       __restrict__ peer_recv_offsets,
    int gpu_id,
    int experts_per_gpu,
    int ep_size)
{
    if (threadIdx.x != 0) return;

    // Phase 1: Expert offsets (exclusive prefix sum of local_expert_counts)
    int running = 0;
    expert_offsets[0] = 0;
    for (int i = 0; i < experts_per_gpu; ++i) {
        running += local_expert_counts[i];
        expert_offsets[i + 1] = running;
    }

    // Phase 2: Peer recv offsets (exclusive prefix sum, skipping self)
    running = 0;
    for (int p = 0; p < ep_size; ++p) {
        peer_recv_offsets[p] = running;
        if (p != gpu_id) {
            running += per_peer_counts[p];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: scatter_tokens_kernel
// ---------------------------------------------------------------------------
static constexpr int SCATTER_THREADS = 128;
static constexpr int SLOTS_PER_BLOCK = 8;

__global__ void scatter_tokens_kernel(
    const __hip_bfloat16* __restrict__ hidden_states,
    const int32_t*       __restrict__ gather_map,
    __hip_bfloat16*       __restrict__ permuted_tokens,
    int num_tokens,
    int hidden_size,
    int top_k)
{
    const int block_slot_start = blockIdx.x * SLOTS_PER_BLOCK;
    const int total_slots = num_tokens * top_k;
    const int tid = threadIdx.x;

    // Vectorized copy parameters
    constexpr int VEC_SIZE = 8;
    const int vec_hidden = hidden_size / VEC_SIZE;
    const int remainder_start = vec_hidden * VEC_SIZE;

    for (int s = 0; s < SLOTS_PER_BLOCK; ++s) {
        const int slot = block_slot_start + s;
        if (slot >= total_slots) break;

        const int map_val = gather_map[slot];
        if (map_val < 0) continue;

        const int token_id = slot / top_k;

        const __hip_bfloat16* src = hidden_states + static_cast<int64_t>(token_id) * hidden_size;
        __hip_bfloat16*       dst = permuted_tokens + static_cast<int64_t>(map_val) * hidden_size;

        const int4* src4 = reinterpret_cast<const int4*>(src);
        int4*       dst4 = reinterpret_cast<int4*>(dst);

        for (int i = tid; i < vec_hidden; i += SCATTER_THREADS) {
            dst4[i] = src4[i];
        }

        for (int i = remainder_start + tid; i < hidden_size; i += SCATTER_THREADS) {
            dst[i] = src[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 1: GPU classify + count + prefix sums
// ---------------------------------------------------------------------------
void moe_permute_classify(
    const int32_t*       expert_indices,
    int32_t*             local_expert_counts,
    int32_t*             per_peer_counts,
    int32_t*             expert_offsets,
    int32_t*             peer_recv_offsets,
    int num_tokens,
    int top_k,
    int gpu_id,
    int experts_per_gpu,
    int ep_size,
    hipStream_t stream)
{
    if (num_tokens == 0) return;

    const int total_slots = num_tokens * top_k;

    // Zero counters
    CUDA_CHECK(hipMemsetAsync(local_expert_counts, 0,
                               experts_per_gpu * sizeof(int32_t), stream));
    CUDA_CHECK(hipMemsetAsync(per_peer_counts, 0,
                               ep_size * sizeof(int32_t), stream));

    // Kernel 1: Classify + count
    {
        constexpr int THREADS = 256;
        const int blocks = cdiv(total_slots, THREADS);
        classify_and_count_kernel<<<blocks, THREADS, 0, stream>>>(
            expert_indices,
            local_expert_counts,
            per_peer_counts,
            total_slots,
            gpu_id,
            experts_per_gpu,
            ep_size);
        CUDA_CHECK(hipGetLastError());
    }

    // Kernel 2: Prefix sums
    {
        prefix_sums_kernel<<<1, 1, 0, stream>>>(
            local_expert_counts,
            per_peer_counts,
            expert_offsets,
            peer_recv_offsets,
            gpu_id,
            experts_per_gpu,
            ep_size);
        CUDA_CHECK(hipGetLastError());
    }
}

// ---------------------------------------------------------------------------
// Phase 2: CPU gather_map computation
// ---------------------------------------------------------------------------
void moe_compute_gather_map_cpu(
    const int32_t* h_expert_indices,
    const int32_t* h_expert_offsets,
    const int32_t* h_peer_recv_offsets,
    int32_t*       h_gather_map,
    int total_slots,
    int gpu_id,
    int experts_per_gpu)
{
    constexpr int ep_size = ModelConfig::ep_size;

    int per_expert_pos[ModelConfig::experts_per_gpu] = {};

    int peer_expert_counts[ModelConfig::max_ep_size][ModelConfig::experts_per_gpu] = {};
    for (int slot = 0; slot < total_slots; ++slot) {
        const int expert_id = h_expert_indices[slot];
        const int rank = expert_id / experts_per_gpu;
        if (rank != gpu_id) {
            const int local_id = expert_id - rank * experts_per_gpu;
            peer_expert_counts[rank][local_id]++;
        }
    }

    int peer_expert_offsets[ModelConfig::max_ep_size][ModelConfig::experts_per_gpu + 1] = {};
    for (int p = 0; p < ep_size; ++p) {
        if (p == gpu_id) continue;
        int running = 0;
        for (int e = 0; e < experts_per_gpu; ++e) {
            peer_expert_offsets[p][e] = running;
            running += peer_expert_counts[p][e];
        }
        peer_expert_offsets[p][experts_per_gpu] = running;
    }

    int peer_per_expert_pos[ModelConfig::max_ep_size][ModelConfig::experts_per_gpu] = {};

    for (int slot = 0; slot < total_slots; ++slot) {
        const int expert_id = h_expert_indices[slot];
        const int owning_ep_rank = expert_id / experts_per_gpu;

        if (owning_ep_rank == gpu_id) {
            const int local_id = expert_id - gpu_id * experts_per_gpu;
            const int pos = h_expert_offsets[local_id] + per_expert_pos[local_id]++;
            h_gather_map[slot] = pos;
        } else {
            const int peer_local_id = expert_id - owning_ep_rank * experts_per_gpu;
            const int peer_pos = peer_expert_offsets[owning_ep_rank][peer_local_id]
                               + peer_per_expert_pos[owning_ep_rank][peer_local_id]++;
            const int recv_idx = h_peer_recv_offsets[owning_ep_rank] + peer_pos;
            h_gather_map[slot] = -(recv_idx + 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3: GPU scatter tokens
// ---------------------------------------------------------------------------
void moe_scatter_tokens(
    const __hip_bfloat16* hidden_states,
    const int32_t*       gather_map,
    __hip_bfloat16*       permuted_tokens,
    int num_tokens,
    int hidden_size,
    int top_k,
    hipStream_t stream)
{
    if (num_tokens == 0) return;

    const int total_slots = num_tokens * top_k;
    dim3 grid(cdiv(total_slots, SLOTS_PER_BLOCK));
    dim3 block(SCATTER_THREADS);
    scatter_tokens_kernel<<<grid, block, 0, stream>>>(
        hidden_states, gather_map, permuted_tokens,
        num_tokens, hidden_size, top_k);
    CUDA_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------------------
// Legacy all-in-one launcher (for benchmarks)
// ---------------------------------------------------------------------------
void moe_permute_forward(
    const __hip_bfloat16* hidden_states,
    const int32_t*       expert_indices,
    __hip_bfloat16*       permuted_tokens,
    int32_t*             expert_offsets,
    int32_t*             gather_map,
    int32_t*             per_peer_counts,
    int32_t*             peer_recv_offsets,
    int32_t*             local_expert_counts,
    int num_tokens,
    int hidden_size,
    int top_k,
    int gpu_id,
    int experts_per_gpu,
    int ep_size,
    hipStream_t stream)
{
    if (num_tokens == 0) return;

    const int total_slots = num_tokens * top_k;

    // Phase 1: GPU classify + prefix sums
    moe_permute_classify(
        expert_indices, local_expert_counts, per_peer_counts,
        expert_offsets, peer_recv_offsets,
        num_tokens, top_k, gpu_id, experts_per_gpu, ep_size, stream);

    // Phase 2: CPU gather_map (requires sync to read expert_offsets)
    static int32_t* h_expert_indices = nullptr;
    static int32_t* h_expert_offsets = nullptr;
    static int32_t* h_peer_recv_offsets = nullptr;
    static int32_t* h_gather_map = nullptr;
    static int cached_slots = 0;
    static int cached_epg = 0;
    static int cached_eps = 0;

    if (total_slots > cached_slots || experts_per_gpu > cached_epg || ep_size > cached_eps) {
        if (h_expert_indices) hipHostFree(h_expert_indices);
        if (h_expert_offsets) hipHostFree(h_expert_offsets);
        if (h_peer_recv_offsets) hipHostFree(h_peer_recv_offsets);
        if (h_gather_map) hipHostFree(h_gather_map);
        int alloc_slots = total_slots;
        int alloc_epg = experts_per_gpu;
        int alloc_eps = ep_size;
        CUDA_CHECK(hipHostMalloc(&h_expert_indices, alloc_slots * sizeof(int32_t)));
        CUDA_CHECK(hipHostMalloc(&h_expert_offsets, (alloc_epg + 1) * sizeof(int32_t)));
        CUDA_CHECK(hipHostMalloc(&h_peer_recv_offsets, alloc_eps * sizeof(int32_t)));
        CUDA_CHECK(hipHostMalloc(&h_gather_map, alloc_slots * sizeof(int32_t)));
        cached_slots = alloc_slots;
        cached_epg = alloc_epg;
        cached_eps = alloc_eps;
    }

    CUDA_CHECK(hipMemcpyAsync(h_expert_indices, expert_indices,
                               total_slots * sizeof(int32_t),
                               hipMemcpyDeviceToHost, stream));
    CUDA_CHECK(hipMemcpyAsync(h_expert_offsets, expert_offsets,
                               (experts_per_gpu + 1) * sizeof(int32_t),
                               hipMemcpyDeviceToHost, stream));
    CUDA_CHECK(hipMemcpyAsync(h_peer_recv_offsets, peer_recv_offsets,
                               ep_size * sizeof(int32_t),
                               hipMemcpyDeviceToHost, stream));
    CUDA_CHECK(hipStreamSynchronize(stream));

    moe_compute_gather_map_cpu(
        h_expert_indices, h_expert_offsets, h_peer_recv_offsets,
        h_gather_map, total_slots, gpu_id, experts_per_gpu);

    CUDA_CHECK(hipMemcpyAsync(gather_map, h_gather_map,
                               total_slots * sizeof(int32_t),
                               hipMemcpyHostToDevice, stream));

    // Phase 3: GPU scatter
    moe_scatter_tokens(hidden_states, gather_map, permuted_tokens,
                       num_tokens, hidden_size, top_k, stream);
}

} // namespace gptoss
