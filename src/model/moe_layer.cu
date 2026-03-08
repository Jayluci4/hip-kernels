// Mixture-of-Experts Layer for GPT-OSS-120B Inference Engine (EP-N)
//
// Expert Parallelism across ep_size GPUs: 128 experts split evenly.
// Top-4 routing with softmax normalization.
// MXFP4 quantized expert weights dequantized on-the-fly.
//
// KEY INSIGHT: Both GPUs hold identical hidden states after replicated
// attention, and routing is deterministic, so both GPUs compute identical
// routing decisions. This means:
//   - NO dispatch phase needed (each GPU already has all tokens)
//   - Each GPU computes expert outputs only for its LOCAL 64 experts
//   - Only expert OUTPUTS are exchanged via a single bidirectional RCCL transfer
//   - Both GPUs know exchange sizes from deterministic routing (no count exchange)
//
// Performance improvements over the previous implementation:
//   1. ONE host stall (pinned async D2H + event sync) instead of 3 hipStreamSynchronize
//   2. No dispatch phase: eliminates redundant token send/recv
//   3. Grouped GEMM: 8 experts batched with shared hipBLASLt descriptors + 8 dequant staging
//   4. Uses the EPComm class for clean, event-based RCCL communication
//   5. Correct remote token counts derived from deterministic routing
//   6. Batched SwiGLU: all batch tokens in 1 kernel launch (64->8 launches/layer)

#include "hip_compat.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include "config.h"
#include "tensor.h"
#include "profiler.h"
#include "cuda_utils.h"
#include "nccl_utils.h"

namespace gptoss {

// ---------------------------------------------------------------------------
// Forward-declare EPComm and extern wrappers (defined in src/comm/ep_comm.cu)
// ---------------------------------------------------------------------------
class EPComm;
extern void ep_comm_combine_v(EPComm* ep,
                              const void* send_buf, int send_count,
                              void* recv_buf,
                              const int* peer_recv_counts,
                              const int* peer_recv_offsets,
                              hipStream_t compute_stream,
                              hipStream_t comm_stream);
extern void ep_comm_wait_combine_complete(EPComm* ep, hipStream_t stream);

// ---------------------------------------------------------------------------
// Extern kernel declarations (defined in src/kernels/*.cu)
// ---------------------------------------------------------------------------

extern void rmsnorm_forward(
    const __hip_bfloat16* input,
    const __hip_bfloat16* weight,
    __hip_bfloat16* output,
    int num_tokens,
    int hidden_size,
    float eps,
    hipStream_t stream);

extern void swiglu_forward(
    const __hip_bfloat16* input,
    __hip_bfloat16* output,
    int num_tokens,
    int intermediate_size,
    float limit,
    hipStream_t stream);

extern void topk_softmax_forward(
    const __hip_bfloat16* router_logits,
    int32_t* expert_indices,
    float* expert_weights,
    int32_t* tokens_per_expert,
    int num_tokens,
    int num_experts,
    int top_k,
    int gpu_id,
    int experts_per_gpu,
    hipStream_t stream);

// Split permute API: classify -> CPU gather_map -> scatter
// Phase 1: GPU classify + count + prefix sums (~5us)
extern void moe_permute_classify(
    const int32_t* expert_indices,
    int32_t* local_expert_counts,
    int32_t* per_peer_counts,
    int32_t* expert_offsets,
    int32_t* peer_recv_offsets,
    int num_tokens, int top_k, int gpu_id,
    int experts_per_gpu, int ep_size, hipStream_t stream);

// Phase 2: CPU gather_map computation (~0.5us, runs on host)
extern void moe_compute_gather_map_cpu(
    const int32_t* h_expert_indices,
    const int32_t* h_expert_offsets,
    const int32_t* h_peer_recv_offsets,
    int32_t* h_gather_map,
    int total_slots, int gpu_id, int experts_per_gpu);

// Phase 3: GPU scatter tokens using pre-computed gather_map (~10us)
extern void moe_scatter_tokens(
    const __hip_bfloat16* hidden_states,
    const int32_t* gather_map,
    __hip_bfloat16* permuted_tokens,
    int num_tokens, int hidden_size, int top_k, hipStream_t stream);

// New moe_combine interface: uses pre-built gather_map with local/remote
// expert outputs. Optionally fuses residual add (if residual != nullptr).
extern void moe_combine_forward(
    __hip_bfloat16* output,
    const __hip_bfloat16* local_expert_out,
    const __hip_bfloat16* remote_expert_out,
    const int32_t* gather_map,
    const float* expert_weights,
    const __hip_bfloat16* residual,  // if non-null: output = residual + combine
    int num_tokens, int hidden_size, int top_k,
    hipStream_t stream);

// Custom BF16 GEMV for M=1 decode -- replaces hipBLASLt overhead
extern void gemv_bf16_forward(
    const __hip_bfloat16* input,
    const __hip_bfloat16* weight,
    __hip_bfloat16* output,
    int N, int K,
    hipStream_t stream,
    const __hip_bfloat16* bias = nullptr);

extern void gemv_bf16_multi_forward(
    const __hip_bfloat16* const* A_array,
    const __hip_bfloat16* const* B_array,
    __hip_bfloat16* const* C_array,
    const int* M_array,
    int N, int K, int count,
    hipStream_t stream);

extern void mxfp4_dequant(
    const uint8_t* packed,
    const uint8_t* scales,
    __hip_bfloat16* output,
    int num_elements,
    hipStream_t stream);

struct MxFp4BatchDesc {
    const uint8_t* packed;
    const uint8_t* scales;
    __hip_bfloat16* output;
    int num_elements;
};

extern void mxfp4_dequant_batched(
    const MxFp4BatchDesc* d_descs,
    int num_experts,
    int max_num_elements,
    hipStream_t stream);

// Fused MXFP4 dequant + GEMV: reads FP4 directly, no staging buffer
extern void fused_mxfp4_gemv_forward(
    const __hip_bfloat16* input,
    const uint8_t* packed_weights,
    const uint8_t* scales,
    __hip_bfloat16* output,
    int N, int K,
    hipStream_t stream,
    const __hip_bfloat16* bias = nullptr);

// ---------------------------------------------------------------------------
// Bias-add kernel: adds a 1D bias vector to each row of a 2D matrix.
// Used for expert MLP biases after batched GEMMs (hipBLAS multi doesn't have epilogue).
// ---------------------------------------------------------------------------
__global__ void bias_add_rows_kernel(
    __hip_bfloat16* __restrict__ data,       // [rows, cols]
    const __hip_bfloat16* __restrict__ bias,  // [cols]
    int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;
        float val = __bfloat162float(data[idx]) + __bfloat162float(bias[col]);
        data[idx] = __float2bfloat16(val);
    }
}

static void bias_add_rows(
    __hip_bfloat16* data, const __hip_bfloat16* bias,
    int rows, int cols, hipStream_t stream)
{
    if (!bias || rows == 0) return;
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bias_add_rows_kernel<<<blocks, threads, 0, stream>>>(data, bias, rows, cols);
}

// ---------------------------------------------------------------------------
// GPU-driven expert dispatch: reads expert_indices from device memory,
// populates fused MXFP4 GEMV descriptors without any D2H.
// ---------------------------------------------------------------------------
struct FusedMxfp4GemvDesc {
    const __hip_bfloat16* input;
    const uint8_t* packed;
    const uint8_t* scales;
    __hip_bfloat16* output;
    const __hip_bfloat16* bias;  // [N] or nullptr
    int M;
};

extern void fused_mxfp4_gemv_multi_forward(
    const FusedMxfp4GemvDesc* d_descs,
    int N, int K, int count,
    hipStream_t stream);

extern void fused_mxfp4_gemv_multi_ptrs_forward(
    const __hip_bfloat16* const* d_inputs,
    const uint8_t* const* d_packed,
    const uint8_t* const* d_scales,
    __hip_bfloat16* const* d_outputs,
    const __hip_bfloat16* const* d_biases,
    int N, int K, int count,
    hipStream_t stream);

__global__ void build_fused_gemv_descs_kernel(
    FusedMxfp4GemvDesc* __restrict__ descs,         // [top_k] output descriptors
    const int32_t* __restrict__ expert_indices,      // [top_k] global expert IDs
    const uint8_t* const* __restrict__ packed_table, // [experts_per_gpu] weight ptr table
    const uint8_t* const* __restrict__ scales_table, // [experts_per_gpu] scale ptr table
    const __hip_bfloat16* input,                     // shared input for all experts
    __hip_bfloat16* output_base,                     // base output buffer
    int output_stride,                               // elements per expert output
    int gpu_id, int experts_per_gpu, int top_k)
{
    int k = threadIdx.x;
    if (k >= top_k) return;

    int global_e = expert_indices[k];
    int local_e = global_e - gpu_id * experts_per_gpu;

    descs[k].input = input;
    descs[k].packed = packed_table[local_e];
    descs[k].scales = scales_table[local_e];
    descs[k].output = output_base + static_cast<int64_t>(k) * output_stride;
    descs[k].M = 1;
}

// W2 variant: each expert has a different input (SwiGLU output at different offset)
__global__ void build_fused_gemv_descs_w2_kernel(
    FusedMxfp4GemvDesc* __restrict__ descs,
    const int32_t* __restrict__ expert_indices,
    const uint8_t* const* __restrict__ packed_table,
    const uint8_t* const* __restrict__ scales_table,
    const __hip_bfloat16* input_base,       // SwiGLU output base
    int input_stride,                        // expert_intermediate_size
    __hip_bfloat16* output_base,            // expert_out_buf base
    int output_stride,                       // hidden_size
    int gpu_id, int experts_per_gpu, int top_k)
{
    int k = threadIdx.x;
    if (k >= top_k) return;

    int global_e = expert_indices[k];
    int local_e = global_e - gpu_id * experts_per_gpu;

    descs[k].input = input_base + static_cast<int64_t>(k) * input_stride;
    descs[k].packed = packed_table[local_e];
    descs[k].scales = scales_table[local_e];
    descs[k].output = output_base + static_cast<int64_t>(k) * output_stride;
    descs[k].M = 1;
}

static void build_fused_gemv_descs(
    FusedMxfp4GemvDesc* d_descs,
    const int32_t* expert_indices,
    const uint8_t* const* packed_table,
    const uint8_t* const* scales_table,
    const __hip_bfloat16* input,
    __hip_bfloat16* output_base,
    int output_stride,
    int gpu_id, int experts_per_gpu, int top_k,
    hipStream_t stream)
{
    build_fused_gemv_descs_kernel<<<1, top_k, 0, stream>>>(
        d_descs, expert_indices, packed_table, scales_table,
        input, output_base, output_stride, gpu_id, experts_per_gpu, top_k);
}

// ---------------------------------------------------------------------------
// GPU-driven expert bias add for fast B=1 decode.
// Reads expert_indices from device memory — no D2H needed.
// ---------------------------------------------------------------------------
__global__ void expert_bias_add_decode_kernel(
    __hip_bfloat16* __restrict__ data,                // [top_k, dim] expert outputs
    const __hip_bfloat16* __restrict__ bias_table,    // [num_experts, dim] all biases
    const int32_t* __restrict__ expert_indices,       // [top_k] global expert IDs
    int dim, int top_k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= top_k * dim) return;

    int k = idx / dim;
    int d = idx % dim;
    int global_e = expert_indices[k];

    float val = __bfloat162float(data[idx]);
    val += __bfloat162float(bias_table[static_cast<int64_t>(global_e) * dim + d]);
    data[idx] = __float2bfloat16(val);
}

static void expert_bias_add_decode(
    __hip_bfloat16* data,
    const __hip_bfloat16* bias_table,
    const int32_t* expert_indices,
    int dim, int top_k,
    hipStream_t stream)
{
    if (!bias_table) return;
    int total = top_k * dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    expert_bias_add_decode_kernel<<<blocks, threads, 0, stream>>>(
        data, bias_table, expert_indices, dim, top_k);
}

// ---------------------------------------------------------------------------
// Fast B=1 decode combine kernel: output = residual + sum_k(weight_k * expert_out_k)
// For top_k=4 experts, each with 1 token output of hidden_size dimensions.
// ---------------------------------------------------------------------------
__global__ void moe_fast_combine_kernel(
    __hip_bfloat16* __restrict__ output,           // [1, hidden_size]
    const __hip_bfloat16* __restrict__ residual,   // [1, hidden_size]
    const __hip_bfloat16* __restrict__ expert_outs, // [top_k, hidden_size] contiguous
    const float* __restrict__ expert_weights,       // [top_k] routing weights
    int hidden_size, int top_k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_size) return;

    float val = __bfloat162float(residual[idx]);
    for (int k = 0; k < top_k; k++) {
        val += expert_weights[k] * __bfloat162float(expert_outs[k * hidden_size + idx]);
    }
    output[idx] = __float2bfloat16(val);
}

static void moe_fast_combine(
    __hip_bfloat16* output,
    const __hip_bfloat16* residual,
    const __hip_bfloat16* expert_outs,
    const float* expert_weights,
    int hidden_size, int top_k,
    hipStream_t stream)
{
    int threads = 256;
    int blocks = (hidden_size + threads - 1) / threads;
    moe_fast_combine_kernel<<<blocks, threads, 0, stream>>>(
        output, residual, expert_outs, expert_weights, hidden_size, top_k);
}

// ---------------------------------------------------------------------------
// Grouped GEMM staging buffers (shared across all MoELayer instances per GPU)
//
// Enables batched expert processing: dequant ALL experts in a batch first,
// then run ALL GEMMs with shared hipBLASLt descriptors, then SwiGLU on all
// batch tokens at once. Reduces hipBLAS descriptor overhead by ~50% and
// SwiGLU launches from 64 to 8 per layer.
//
// Memory per GPU:
//   8 dequant buffers x 33.2 MB = 265.6 MB
//   gate_up staging:                94.4 MB
//   swiglu staging:                 47.2 MB
//   Total:                        ~407 MB
// ---------------------------------------------------------------------------
static constexpr int GROUPED_BATCH_SIZE = 8;

struct GroupedGemmStaging {
    // Double-buffered dequant staging: 2 banks of 8 buffers each.
    // While GEMMs read from bank[cur], dequant writes next batch to bank[nxt].
    __hip_bfloat16* dequant[2][GROUPED_BATCH_SIZE] = {};
    __hip_bfloat16* gate_up = nullptr;
    __hip_bfloat16* swiglu  = nullptr;
    int max_total_local = 0;
    bool initialized = false;
};

static GroupedGemmStaging s_staging[ModelConfig::world_size] = {};  // Indexed by device_id

static void ensure_grouped_staging(int device_id, int max_total_local) {
    GroupedGemmStaging& stg = s_staging[device_id];
    if (stg.initialized && stg.max_total_local >= max_total_local) return;

    CUDA_CHECK(hipSetDevice(device_id));

    // Free old if resizing
    if (stg.initialized) {
        for (int bank = 0; bank < 2; bank++)
            for (int i = 0; i < GROUPED_BATCH_SIZE; i++)
                if (stg.dequant[bank][i]) { CUDA_CHECK(hipFree(stg.dequant[bank][i])); stg.dequant[bank][i] = nullptr; }
        if (stg.gate_up) { CUDA_CHECK(hipFree(stg.gate_up)); stg.gate_up = nullptr; }
        if (stg.swiglu)  { CUDA_CHECK(hipFree(stg.swiglu));  stg.swiglu  = nullptr; }
    }

    // Dequant: 2 banks x 8 buffers, each max(mlp1, mlp2) elements
    constexpr int max_dequant_elems =
        ModelConfig::hidden_size * ModelConfig::expert_gate_up_size;  // 2880 x 5760
    for (int bank = 0; bank < 2; bank++)
        for (int i = 0; i < GROUPED_BATCH_SIZE; i++)
            CUDA_CHECK(hipMalloc(&stg.dequant[bank][i], max_dequant_elems * sizeof(__hip_bfloat16)));

    // gate_up: [max_total_local, expert_gate_up_size]
    CUDA_CHECK(hipMalloc(&stg.gate_up,
        static_cast<size_t>(max_total_local) * ModelConfig::expert_gate_up_size * sizeof(__hip_bfloat16)));

    // swiglu: [max_total_local, expert_intermediate_size]
    CUDA_CHECK(hipMalloc(&stg.swiglu,
        static_cast<size_t>(max_total_local) * ModelConfig::expert_intermediate_size * sizeof(__hip_bfloat16)));

    stg.max_total_local = max_total_local;
    stg.initialized = true;

    double total_mb =
        (2 * GROUPED_BATCH_SIZE * max_dequant_elems * 2.0 +
         (double)max_total_local * ModelConfig::expert_gate_up_size * 2.0 +
         (double)max_total_local * ModelConfig::expert_intermediate_size * 2.0) /
        (1024.0 * 1024.0);
    fprintf(stderr, "[MoELayer] GPU %d: double-buffered GEMM staging %.1f MB (max_local=%d)\n",
            device_id, total_mb, max_total_local);
}

// ---------------------------------------------------------------------------
// MXFP4 Expert Weight Descriptor
// Stores packed MXFP4 weights and their E8M0 block scales for one matrix.
// ---------------------------------------------------------------------------

struct MXFP4Weight {
    const uint8_t* packed;   // Packed MXFP4 nibble pairs
    const uint8_t* scales;   // E8M0 block scales (one per 32 elements)
    int num_elements;        // Total number of logical elements
};

// ---------------------------------------------------------------------------
// MoELayer
//
// New architecture: no dispatch, exchange only expert outputs.
//
// Forward pipeline:
//   1. RMSNorm
//   2. Router GEMM
//   3. TopK routing (deterministic, identical on both GPUs)
//   4. Split permute: GPU classify+count+prefix -> D2H -> CPU gather_map -> H2D -> GPU scatter
//      (~17us total vs 137us with old GPU serial scan)
//   5. Expert GEMMs with double-buffered dequant staging
//   6. Exchange expert outputs via EPComm::combine_v() (N-way grouped P2P)
//   7+8. Fused combine + residual add in one kernel
// ---------------------------------------------------------------------------

class MoELayer {
public:
    // ---- Configuration ----
    int layer_id_;
    int device_id_;
    int gpu_id_;    // EP rank in [0, ep_size)

    // ---- Weights ----
    // RMSNorm weight for the MoE sub-layer normalization
    const __hip_bfloat16* moe_norm_weight_;   // [hidden_size] = [2880]

    // Router weight (BF16): [hidden_size, num_experts] = [2880, 128]
    const __hip_bfloat16* router_weight_;
    const __hip_bfloat16* router_bias_;     // [num_experts] router bias (nullptr if none)

    // Expert MLP biases (BF16): packed [num_experts, dim] -- full tensor, index by global expert
    const __hip_bfloat16* gate_up_proj_bias_;  // [num_experts, expert_gate_up_size] (nullptr if none)
    const __hip_bfloat16* down_proj_bias_;     // [num_experts, hidden_size] (nullptr if none)

    // Expert weights in MXFP4 format, stored only for local experts (64 per GPU).
    // Each expert has:
    //   mlp1: [hidden_size, expert_gate_up_size] = [2880, 5760] in MXFP4
    //   mlp2: [expert_intermediate_size, hidden_size] = [2880, 2880] in MXFP4
    MXFP4Weight* expert_mlp1_;  // Array of experts_per_gpu MXFP4Weight descriptors
    MXFP4Weight* expert_mlp2_;  // Array of experts_per_gpu MXFP4Weight descriptors

    // Pre-dequantized BF16 weights for all local experts.
    // Eliminates runtime dequant kernels + hipStreamSynchronize overhead.
    // Memory: 128 experts × ~47.5 MB each ≈ 6.1 GB
    __hip_bfloat16* expert_mlp1_bf16_[ModelConfig::experts_per_gpu] = {};
    __hip_bfloat16* expert_mlp2_bf16_[ModelConfig::experts_per_gpu] = {};

    // ---- hipBLAS handles ----
    const CublasHandle* cublas_compute_;
    const CublasHandle* cublas_comm_;

    // ---- Streams for communication/compute overlap ----
    hipStream_t compute_stream_;
    hipStream_t comm_stream_;

    // ---- Synchronization events ----
    CudaEvent local_done_;      // Signaled after local expert compute completes
    CudaEvent combine_done_;    // Signaled after expert output exchange completes

    // ---- Double-buffered dequant pipeline ----
    hipStream_t dequant_stream_ = nullptr;
    // Persistent events (avoid create/destroy overhead: ~2us x 576 events/fwd = ~1.2ms saved)
    hipEvent_t w1_dq_done_[2] = {nullptr, nullptr};  // W1 dequant done for bank 0/1
    hipEvent_t gemm1_done_ = nullptr;                  // MLP1 GEMM done (bank safe to overwrite)
    hipEvent_t w2_dq_done_ = nullptr;                  // W2 dequant done

    // ---- EPComm (borrowed, not owned) ----
    EPComm* ep_comm_;

    // ---- RCCL communicator for routing broadcast (borrowed, not owned) ----
    ncclComm_t nccl_comm_ = nullptr;

    // ---- Scratch buffers (device memory, not owned) ----
    __hip_bfloat16* norm_buf_;            // [max_tokens, hidden_size]
    __hip_bfloat16* router_logits_buf_;   // [max_tokens, num_experts]
    int32_t*       expert_indices_buf_;   // [max_tokens, top_k]
    float*         expert_weights_buf_;   // [max_tokens, top_k]
    int32_t*       tokens_per_expert_;    // [num_experts]
    __hip_bfloat16* permuted_tokens_;     // [max_tokens * top_k, hidden_size]
    int32_t*       expert_offsets_;       // [experts_per_gpu + 1]
    int32_t*       gather_map_;          // [max_tokens * top_k]
    int32_t*       per_peer_counts_;     // [ep_size] device memory
    int32_t*       peer_recv_offsets_;    // [ep_size] device memory
    int32_t*       local_expert_counts_; // [experts_per_gpu] scratch for permute
    __hip_bfloat16* recv_buffer_;         // [max_tokens * top_k, hidden_size]

    // Pinned host memory for split permute API (CPU gather_map path)
    int32_t* h_expert_offsets_pinned_;      // [experts_per_gpu + 1]
    int32_t* h_per_peer_counts_pinned_;     // [ep_size]
    int32_t* h_peer_recv_offsets_pinned_;   // [ep_size]
    int32_t* h_expert_indices_pinned_;      // [pinned_slots_capacity_]
    int32_t* h_gather_map_pinned_;          // [pinned_slots_capacity_]
    int pinned_slots_capacity_;             // current capacity of pinned slot buffers

    // Double-buffered dequantized expert weight buffers (BF16)
    __hip_bfloat16* dequant_mlp1_buf_[2]; // Each: [hidden_size, expert_gate_up_size] = [2880, 5760]
    __hip_bfloat16* dequant_mlp2_buf_[2]; // Each: [expert_intermediate_size, hidden_size] = [2880, 2880]

    // Expert computation intermediate buffers
    __hip_bfloat16* gate_up_buf_;         // [max_expert_tokens, expert_gate_up_size]
    __hip_bfloat16* swiglu_buf_;          // [max_expert_tokens, expert_intermediate_size]
    __hip_bfloat16* expert_out_buf_;      // [max_tokens * top_k, hidden_size]
    __hip_bfloat16* moe_output_buf_;      // [max_tokens, hidden_size]

    // Batched dequant descriptors (persistent, avoids per-call alloc).
    // All dequant runs on compute_stream_ (no separate dequant_stream_ pipeline)
    // to avoid race conditions with pinned host memory DMA.
    MxFp4BatchDesc* d_dequant_descs_ = nullptr;   // device [GROUPED_BATCH_SIZE]
    MxFp4BatchDesc* h_dequant_descs_ = nullptr;    // pinned host [GROUPED_BATCH_SIZE]

    // Device-side expert weight pointer tables for fast B=1 decode.
    // Eliminates D2H of expert IDs — GPU kernel reads expert_indices from device
    // memory and looks up weight pointers directly.
    const uint8_t** d_mlp1_packed_ = nullptr;   // [experts_per_gpu] packed ptrs
    const uint8_t** d_mlp1_scales_ = nullptr;   // [experts_per_gpu] scale ptrs
    const uint8_t** d_mlp2_packed_ = nullptr;   // [experts_per_gpu]
    const uint8_t** d_mlp2_scales_ = nullptr;   // [experts_per_gpu]
    FusedMxfp4GemvDesc* d_fused_descs_ = nullptr; // [top_k] device descriptors
    FusedMxfp4GemvDesc* h_fused_descs_ = nullptr; // [top_k] pinned host descriptors

    // Pinned host pointer arrays for batched multi-expert GEMV.
    // GPU reads these directly from host memory (zero-copy) — no memcpy needed.
    // Only 5 × 8 × 8 = 320 bytes, so host-memory latency is negligible.
    const __hip_bfloat16** h_gemv_inputs_ = nullptr;    // [GROUPED_BATCH_SIZE] pinned
    const uint8_t** h_gemv_packed_ = nullptr;            // [GROUPED_BATCH_SIZE] pinned
    const uint8_t** h_gemv_scales_ = nullptr;            // [GROUPED_BATCH_SIZE] pinned
    __hip_bfloat16** h_gemv_outputs_ = nullptr;          // [GROUPED_BATCH_SIZE] pinned
    const __hip_bfloat16** h_gemv_biases_ = nullptr;     // [GROUPED_BATCH_SIZE] pinned

    // Constants
    static constexpr int hidden_size = ModelConfig::hidden_size;
    static constexpr int num_experts = ModelConfig::num_experts;
    static constexpr int top_k = ModelConfig::num_active_experts;
    static constexpr int experts_per_gpu = ModelConfig::experts_per_gpu;
    static constexpr int world_size = ModelConfig::world_size;
    static constexpr int expert_intermediate_size = ModelConfig::expert_intermediate_size;
    static constexpr int expert_gate_up_size = ModelConfig::expert_gate_up_size;
    static constexpr float rms_norm_eps = ModelConfig::rms_norm_eps;
    static constexpr float swiglu_limit = ModelConfig::swiglu_limit;
    static constexpr int mlp1_elements = hidden_size * expert_gate_up_size;    // 2880 * 5760
    static constexpr int mlp2_elements = expert_intermediate_size * hidden_size; // 2880 * 2880

    MoELayer() = default;

    // ---- Initialization ----
    void init(
        int layer_id,
        int device_id,
        int gpu_id,
        EPComm* ep_comm,
        const CublasHandle* cublas_compute,
        const CublasHandle* cublas_comm,
        hipStream_t compute_stream,
        hipStream_t comm_stream,
        const __hip_bfloat16* moe_norm_weight,
        const __hip_bfloat16* router_weight,
        const __hip_bfloat16* router_bias,
        const __hip_bfloat16* gate_up_proj_bias,
        const __hip_bfloat16* down_proj_bias,
        MXFP4Weight* expert_mlp1,
        MXFP4Weight* expert_mlp2,
        __hip_bfloat16* norm_buf,
        __hip_bfloat16* router_logits_buf,
        int32_t* expert_indices_buf,
        float* expert_weights_buf,
        int32_t* tokens_per_expert,
        __hip_bfloat16* permuted_tokens,
        int32_t* expert_offsets,
        int32_t* gather_map,
        int32_t* per_peer_counts,
        int32_t* peer_recv_offsets,
        __hip_bfloat16* recv_buffer,
        __hip_bfloat16* dequant_mlp1_buf_0,
        __hip_bfloat16* dequant_mlp1_buf_1,
        __hip_bfloat16* dequant_mlp2_buf_0,
        __hip_bfloat16* dequant_mlp2_buf_1,
        __hip_bfloat16* gate_up_buf,
        __hip_bfloat16* swiglu_buf,
        __hip_bfloat16* expert_out_buf,
        __hip_bfloat16* moe_output_buf,
        ncclComm_t nccl_comm = nullptr)
    {
        layer_id_ = layer_id;
        device_id_ = device_id;
        gpu_id_ = gpu_id;
        ep_comm_ = ep_comm;
        cublas_compute_ = cublas_compute;
        cublas_comm_ = cublas_comm;
        compute_stream_ = compute_stream;
        comm_stream_ = comm_stream;

        moe_norm_weight_ = moe_norm_weight;
        router_weight_ = router_weight;
        router_bias_ = router_bias;
        gate_up_proj_bias_ = gate_up_proj_bias;
        down_proj_bias_ = down_proj_bias;
        expert_mlp1_ = expert_mlp1;
        expert_mlp2_ = expert_mlp2;

        norm_buf_ = norm_buf;
        router_logits_buf_ = router_logits_buf;
        expert_indices_buf_ = expert_indices_buf;
        expert_weights_buf_ = expert_weights_buf;
        tokens_per_expert_ = tokens_per_expert;
        permuted_tokens_ = permuted_tokens;
        expert_offsets_ = expert_offsets;
        gather_map_ = gather_map;
        per_peer_counts_ = per_peer_counts;
        peer_recv_offsets_ = peer_recv_offsets;
        recv_buffer_ = recv_buffer;

        dequant_mlp1_buf_[0] = dequant_mlp1_buf_0;
        dequant_mlp1_buf_[1] = dequant_mlp1_buf_1;
        dequant_mlp2_buf_[0] = dequant_mlp2_buf_0;
        dequant_mlp2_buf_[1] = dequant_mlp2_buf_1;
        gate_up_buf_ = gate_up_buf;
        swiglu_buf_ = swiglu_buf;
        expert_out_buf_ = expert_out_buf;
        moe_output_buf_ = moe_output_buf;
        nccl_comm_ = nccl_comm;

        // Allocate pinned host memory for async D2H of expert_offsets + per-peer counts.
        CUDA_CHECK(hipHostMalloc(
            &h_expert_offsets_pinned_,
            (experts_per_gpu + 1) * sizeof(int32_t)));
        CUDA_CHECK(hipHostMalloc(
            &h_per_peer_counts_pinned_,
            ModelConfig::max_ep_size * sizeof(int32_t)));
        CUDA_CHECK(hipHostMalloc(
            &h_peer_recv_offsets_pinned_,
            ModelConfig::max_ep_size * sizeof(int32_t)));

        // Pre-allocate scratch for moe_permute (avoids hipMallocAsync per call).
        CUDA_CHECK(hipMalloc(&local_expert_counts_,
                              experts_per_gpu * sizeof(int32_t)));

        // Pinned slot buffers allocated lazily in forward() on first use.
        h_expert_indices_pinned_ = nullptr;
        h_gather_map_pinned_ = nullptr;
        pinned_slots_capacity_ = 0;

        // Create dequant stream for pipelined weight decompression.
        // Non-blocking so it doesn't serialize with compute_stream_.
        CUDA_CHECK(hipSetDevice(device_id));
        CUDA_CHECK(hipStreamCreateWithFlags(&dequant_stream_, hipStreamNonBlocking));

        // Batched dequant descriptor buffers (persistent)
        CUDA_CHECK(hipMalloc(&d_dequant_descs_, GROUPED_BATCH_SIZE * sizeof(MxFp4BatchDesc)));
        CUDA_CHECK(hipHostMalloc(&h_dequant_descs_, GROUPED_BATCH_SIZE * sizeof(MxFp4BatchDesc)));

        // Persistent sync events for double-buffered dequant pipeline.
        // hipEventDisableTiming minimizes event overhead.
        for (int i = 0; i < 2; i++)
            CUDA_CHECK(hipEventCreateWithFlags(&w1_dq_done_[i], hipEventDisableTiming));
        CUDA_CHECK(hipEventCreateWithFlags(&gemm1_done_, hipEventDisableTiming));
        CUDA_CHECK(hipEventCreateWithFlags(&w2_dq_done_, hipEventDisableTiming));

        // expert_mlp1_bf16_ / expert_mlp2_bf16_ are NOT pre-allocated.
        // Instead, we use shared staging buffers and restructured batching
        // to minimize hipStreamSynchronize calls at runtime.

        // Build device-side expert weight pointer tables for fast B=1 decode.
        // Allows GPU kernels to look up expert weights without D2H of expert IDs.
        {
            const uint8_t* h_packed1[experts_per_gpu];
            const uint8_t* h_scales1[experts_per_gpu];
            const uint8_t* h_packed2[experts_per_gpu];
            const uint8_t* h_scales2[experts_per_gpu];
            for (int e = 0; e < static_cast<int>(experts_per_gpu); e++) {
                h_packed1[e] = expert_mlp1[e].packed;
                h_scales1[e] = expert_mlp1[e].scales;
                h_packed2[e] = expert_mlp2[e].packed;
                h_scales2[e] = expert_mlp2[e].scales;
            }
            CUDA_CHECK(hipMalloc(&d_mlp1_packed_, experts_per_gpu * sizeof(const uint8_t*)));
            CUDA_CHECK(hipMalloc(&d_mlp1_scales_, experts_per_gpu * sizeof(const uint8_t*)));
            CUDA_CHECK(hipMalloc(&d_mlp2_packed_, experts_per_gpu * sizeof(const uint8_t*)));
            CUDA_CHECK(hipMalloc(&d_mlp2_scales_, experts_per_gpu * sizeof(const uint8_t*)));
            CUDA_CHECK(hipMemcpy(d_mlp1_packed_, h_packed1, experts_per_gpu * sizeof(const uint8_t*), hipMemcpyHostToDevice));
            CUDA_CHECK(hipMemcpy(d_mlp1_scales_, h_scales1, experts_per_gpu * sizeof(const uint8_t*), hipMemcpyHostToDevice));
            CUDA_CHECK(hipMemcpy(d_mlp2_packed_, h_packed2, experts_per_gpu * sizeof(const uint8_t*), hipMemcpyHostToDevice));
            CUDA_CHECK(hipMemcpy(d_mlp2_scales_, h_scales2, experts_per_gpu * sizeof(const uint8_t*), hipMemcpyHostToDevice));
            CUDA_CHECK(hipMalloc(&d_fused_descs_, GROUPED_BATCH_SIZE * sizeof(FusedMxfp4GemvDesc)));
            CUDA_CHECK(hipHostMalloc(&h_fused_descs_, GROUPED_BATCH_SIZE * sizeof(FusedMxfp4GemvDesc)));
            // Pinned host pointer arrays for batched GEMV (zero-copy from GPU)
            CUDA_CHECK(hipHostMalloc(&h_gemv_inputs_, GROUPED_BATCH_SIZE * sizeof(const __hip_bfloat16*)));
            CUDA_CHECK(hipHostMalloc(&h_gemv_packed_, GROUPED_BATCH_SIZE * sizeof(const uint8_t*)));
            CUDA_CHECK(hipHostMalloc(&h_gemv_scales_, GROUPED_BATCH_SIZE * sizeof(const uint8_t*)));
            CUDA_CHECK(hipHostMalloc(&h_gemv_outputs_, GROUPED_BATCH_SIZE * sizeof(__hip_bfloat16*)));
            CUDA_CHECK(hipHostMalloc(&h_gemv_biases_, GROUPED_BATCH_SIZE * sizeof(const __hip_bfloat16*)));
        }
    }

    // ---- Cleanup ----
    void destroy() {
        if (h_expert_offsets_pinned_) {
            CUDA_CHECK(hipHostFree(h_expert_offsets_pinned_));
            h_expert_offsets_pinned_ = nullptr;
        }
        if (h_per_peer_counts_pinned_) {
            CUDA_CHECK(hipHostFree(h_per_peer_counts_pinned_));
            h_per_peer_counts_pinned_ = nullptr;
        }
        if (h_peer_recv_offsets_pinned_) {
            CUDA_CHECK(hipHostFree(h_peer_recv_offsets_pinned_));
            h_peer_recv_offsets_pinned_ = nullptr;
        }
        if (local_expert_counts_) { CUDA_CHECK(hipFree(local_expert_counts_)); local_expert_counts_ = nullptr; }
        if (h_expert_indices_pinned_) { CUDA_CHECK(hipHostFree(h_expert_indices_pinned_)); h_expert_indices_pinned_ = nullptr; }
        if (h_gather_map_pinned_) { CUDA_CHECK(hipHostFree(h_gather_map_pinned_)); h_gather_map_pinned_ = nullptr; }
        pinned_slots_capacity_ = 0;
        if (d_dequant_descs_) { CUDA_CHECK(hipFree(d_dequant_descs_)); d_dequant_descs_ = nullptr; }
        if (h_dequant_descs_) { CUDA_CHECK(hipHostFree(h_dequant_descs_)); h_dequant_descs_ = nullptr; }
        if (d_fused_descs_) { CUDA_CHECK(hipFree(d_fused_descs_)); d_fused_descs_ = nullptr; }
        if (h_fused_descs_) { CUDA_CHECK(hipHostFree(h_fused_descs_)); h_fused_descs_ = nullptr; }
        if (h_gemv_inputs_) { CUDA_CHECK(hipHostFree(h_gemv_inputs_)); h_gemv_inputs_ = nullptr; }
        if (h_gemv_packed_) { CUDA_CHECK(hipHostFree(h_gemv_packed_)); h_gemv_packed_ = nullptr; }
        if (h_gemv_scales_) { CUDA_CHECK(hipHostFree(h_gemv_scales_)); h_gemv_scales_ = nullptr; }
        if (h_gemv_outputs_) { CUDA_CHECK(hipHostFree(h_gemv_outputs_)); h_gemv_outputs_ = nullptr; }
        if (h_gemv_biases_) { CUDA_CHECK(hipHostFree(h_gemv_biases_)); h_gemv_biases_ = nullptr; }
        if (dequant_stream_) {
            hipStreamDestroy(dequant_stream_);
            dequant_stream_ = nullptr;
        }
        for (int i = 0; i < 2; i++) {
            if (w1_dq_done_[i]) { hipEventDestroy(w1_dq_done_[i]); w1_dq_done_[i] = nullptr; }
        }
        if (gemm1_done_) { hipEventDestroy(gemm1_done_); gemm1_done_ = nullptr; }
        if (w2_dq_done_) { hipEventDestroy(w2_dq_done_); w2_dq_done_ = nullptr; }
        for (int e = 0; e < experts_per_gpu; ++e) {
            if (expert_mlp1_bf16_[e]) { CUDA_CHECK(hipFree(expert_mlp1_bf16_[e])); expert_mlp1_bf16_[e] = nullptr; }
            if (expert_mlp2_bf16_[e]) { CUDA_CHECK(hipFree(expert_mlp2_bf16_[e])); expert_mlp2_bf16_[e] = nullptr; }
        }
    }

    // -----------------------------------------------------------------------
    // Forward pass: full MoE computation for one layer
    //
    // input:   [num_tokens, hidden_size]  - residual stream (BF16)
    // output:  [num_tokens, hidden_size]  - updated residual stream (BF16)
    //
    // New pipeline (no dispatch, exchange expert outputs only):
    //
    //   COMPUTE STREAM:                    COMM STREAM:
    //   1. RMSNorm                         |
    //   2. Router GEMM                     |
    //   3. TopK routing (deterministic)    |
    //   4. Split permute:                  |
    //      GPU classify+count+prefix (~5us)|
    //      D2H -> CPU gather_map (~0.5us)  |
    //      H2D -> GPU scatter (~10us)      |
    //   5. Expert GEMMs (grouped batches   |
    //      of 8 with shared hipBLASLt desc)|
    //   ----event: local_done----------->  |
    //                                      | 6. EPComm::combine_v()
    //                                      |    (N-way grouped P2P exchange
    //                                      |     of expert outputs ONLY)
    //   <---event: combine_done----------- |
    //   7+8. Fused combine + residual add  |
    //      (gather + weight sum + resid)   |
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // forward_prenormed: accepts pre-normed input and separate residual.
    // Skips internal RMSNorm (step 1). Used by fused residual+RMSNorm path.
    // -----------------------------------------------------------------------
    void forward_prenormed(
        const __hip_bfloat16* normed_input,
        const __hip_bfloat16* residual_input,
        __hip_bfloat16* output,
        int num_tokens,
        hipStream_t main_stream)
    {
        if (num_tokens == 0) return;
        CudaEvent input_ready;
        input_ready.record(main_stream);
        input_ready.wait(compute_stream_);
        forward_core(normed_input, residual_input, output, num_tokens, main_stream);
    }

    void forward(
        const __hip_bfloat16* input,
        __hip_bfloat16* output,
        int num_tokens,
        hipStream_t main_stream)
    {
        if (num_tokens == 0) return;

        // -----------------------------------------------------------------
        // Stream synchronization: main_stream -> compute_stream_
        // -----------------------------------------------------------------
        CudaEvent input_ready;
        input_ready.record(main_stream);
        input_ready.wait(compute_stream_);

        // -----------------------------------------------------------------
        // Step 1: RMSNorm
        // -----------------------------------------------------------------
        rmsnorm_forward(
            input, moe_norm_weight_, norm_buf_,
            num_tokens, hidden_size, rms_norm_eps, compute_stream_);

        forward_core(norm_buf_, input, output, num_tokens, main_stream);
    }

    const __hip_bfloat16* get_ffn_norm_weight() const { return moe_norm_weight_; }

private:
    // Core forward logic after RMSNorm: router -> permute -> expert GEMMs -> combine
    void forward_core(
        const __hip_bfloat16* normed,    // normed data for router GEMM + scatter
        const __hip_bfloat16* residual,  // for combine+residual
        __hip_bfloat16* output,
        int num_tokens,
        hipStream_t main_stream)
    {
        // -----------------------------------------------------------------
        // Steps 2-3: Router GEMM + TopK Routing
        //
        // Routing is deterministic -- both GPUs would produce identical
        // expert_indices and expert_weights from identical inputs.
        //
        // For prefill (many tokens), GPU 0 computes routing and broadcasts
        // results via RCCL, saving ~595us of redundant router GEMM + TopK
        // per layer. For decode (few tokens), both GPUs compute locally
        // since RCCL launch overhead (~10us) exceeds the compute savings.
        // -----------------------------------------------------------------
        // Always broadcast routing from GPU 0 when running EP.
        // Independent routing on each GPU causes floating-point divergence
        // -> different expert selections -> RCCL send/recv size mismatch -> deadlock.
        if (nccl_comm_ != nullptr) {
            // --- Prefill path: compute on GPU 0, broadcast to all ---
            if (gpu_id_ == 0) {
                PROF("router_gemm", "moe_sub", device_id_, compute_stream_);
                if (num_tokens == 1) {
                    gemv_bf16_forward(normed, router_weight_, router_logits_buf_,
                                      num_experts, hidden_size, compute_stream_, router_bias_);
                } else {
                    cublas_compute_->gemm_bf16_lt(
                        normed, router_weight_, router_logits_buf_,
                        num_tokens, num_experts, hidden_size, compute_stream_,
                        /*alpha=*/1.0f, /*beta=*/0.0f, /*bias=*/router_bias_, /*transB=*/true);
                }
                PROF_END(device_id_, compute_stream_);

                PROF("topk_softmax", "moe_sub", device_id_, compute_stream_);
                topk_softmax_forward(
                    router_logits_buf_, expert_indices_buf_, expert_weights_buf_,
                    tokens_per_expert_, num_tokens, num_experts, top_k,
                    gpu_id_, experts_per_gpu, compute_stream_);
                PROF_END(device_id_, compute_stream_);

                // Sync compute -> comm before broadcast reads the results
                CudaEvent route_done;
                route_done.record(compute_stream_);
                route_done.wait(comm_stream_);
            }

            // Ensure comm_stream_ waits for main_stream before using the RCCL
            // communicator. The hidden-state broadcast (transformer.cu) runs on
            // main_stream with the SAME communicator. RCCL requires the user to
            // serialize operations across different streams on the same communicator.
            // On GPU 0 this is already satisfied transitively via route_done ->
            // compute_stream_ -> input_ready -> main_stream, but GPU 1 skips the
            // routing compute, leaving comm_stream_ with no dependency on main_stream.
            {
                CudaEvent main_done;
                main_done.record(main_stream);
                main_done.wait(comm_stream_);
            }

            // All ranks participate in broadcast (RCCL collective requirement).
            // GPU 0 sends, GPU 1 receives. ~64KB over 800 GB/s Infinity Fabric: <1us.
            size_t route_count = static_cast<size_t>(num_tokens) * top_k;
            PROF("route_bcast", "moe_sub", device_id_, comm_stream_);
            NCCL_CHECK(ncclGroupStart());
            NCCL_CHECK(ncclBroadcast(
                expert_indices_buf_, expert_indices_buf_,
                route_count, ncclInt32, /*root=*/0, nccl_comm_, comm_stream_));
            NCCL_CHECK(ncclBroadcast(
                expert_weights_buf_, expert_weights_buf_,
                route_count, ncclFloat, /*root=*/0, nccl_comm_, comm_stream_));
            NCCL_CHECK(ncclGroupEnd());
            PROF_END(device_id_, comm_stream_);

            // Sync comm -> compute so permute sees the broadcast results
            CudaEvent bcast_done;
            bcast_done.record(comm_stream_);
            bcast_done.wait(compute_stream_);

        } else {
            // --- Single-GPU path: no RCCL, compute locally ---
            PROF("router_gemm", "moe_sub", device_id_, compute_stream_);
            if (num_tokens == 1) {
                // Decode: custom GEMV avoids ~1ms hipBLASLt CPU overhead
                gemv_bf16_forward(normed, router_weight_, router_logits_buf_,
                                  num_experts, hidden_size, compute_stream_, router_bias_);
            } else {
                cublas_compute_->gemm_bf16_lt(
                    normed, router_weight_, router_logits_buf_,
                    num_tokens, num_experts, hidden_size, compute_stream_,
                    /*alpha=*/1.0f, /*beta=*/0.0f, /*bias=*/router_bias_, /*transB=*/true);
            }
            PROF_END(device_id_, compute_stream_);

            PROF("topk_softmax", "moe_sub", device_id_, compute_stream_);
            topk_softmax_forward(
                router_logits_buf_, expert_indices_buf_, expert_weights_buf_,
                tokens_per_expert_, num_tokens, num_experts, top_k,
                gpu_id_, experts_per_gpu, compute_stream_);
            PROF_END(device_id_, compute_stream_);
        }

        // -----------------------------------------------------------------
        // Steps 4-5: Permute pipeline
        // -----------------------------------------------------------------
        const int total_slots = num_tokens * top_k;
        int total_local_tokens;

        {
            if (total_slots > pinned_slots_capacity_) {
                if (h_expert_indices_pinned_) CUDA_CHECK(hipHostFree(h_expert_indices_pinned_));
                if (h_gather_map_pinned_) CUDA_CHECK(hipHostFree(h_gather_map_pinned_));
                CUDA_CHECK(hipHostMalloc(&h_expert_indices_pinned_, total_slots * sizeof(int32_t)));
                CUDA_CHECK(hipHostMalloc(&h_gather_map_pinned_, total_slots * sizeof(int32_t)));
                pinned_slots_capacity_ = total_slots;
            }

            PROF("permute", "moe_sub", device_id_, compute_stream_);
            moe_permute_classify(
                expert_indices_buf_, local_expert_counts_, per_peer_counts_,
                expert_offsets_, peer_recv_offsets_,
                num_tokens, top_k, gpu_id_, experts_per_gpu,
                ModelConfig::ep_size, compute_stream_);

            CUDA_CHECK(hipMemcpyAsync(h_expert_indices_pinned_, expert_indices_buf_,
                total_slots * sizeof(int32_t), hipMemcpyDeviceToHost, compute_stream_));
            CUDA_CHECK(hipMemcpyAsync(h_expert_offsets_pinned_, expert_offsets_,
                (experts_per_gpu + 1) * sizeof(int32_t), hipMemcpyDeviceToHost, compute_stream_));
            CUDA_CHECK(hipMemcpyAsync(h_per_peer_counts_pinned_, per_peer_counts_,
                ModelConfig::ep_size * sizeof(int32_t), hipMemcpyDeviceToHost, compute_stream_));
            CUDA_CHECK(hipMemcpyAsync(h_peer_recv_offsets_pinned_, peer_recv_offsets_,
                ModelConfig::ep_size * sizeof(int32_t), hipMemcpyDeviceToHost, compute_stream_));

            CUDA_CHECK(hipStreamSynchronize(compute_stream_));

            moe_compute_gather_map_cpu(
                h_expert_indices_pinned_, h_expert_offsets_pinned_,
                h_peer_recv_offsets_pinned_, h_gather_map_pinned_,
                total_slots, gpu_id_, experts_per_gpu);

            CUDA_CHECK(hipMemcpyAsync(gather_map_, h_gather_map_pinned_,
                total_slots * sizeof(int32_t), hipMemcpyHostToDevice, compute_stream_));

            moe_scatter_tokens(normed, gather_map_, permuted_tokens_,
                num_tokens, hidden_size, top_k, compute_stream_);
            PROF_END(device_id_, compute_stream_);

            total_local_tokens = h_expert_offsets_pinned_[experts_per_gpu];
        }

        // -----------------------------------------------------------------
        // Step 5: Expert GEMMs -- Double-buffered dequant pipeline
        //
        // Process experts in batches of 8 with pipelined weight decompression:
        //
        //   compute_stream:  [---GEMM1_i---][SwiGLU][----GEMM2_i----|---GEMM1_{i+1}---]
        //   dequant_stream:  .......[W2_dq_i][W1_dq_{i+1}]...........[W2_dq_{i+1}]....
        //                            ^ bank A  ^ bank B               ^ bank B
        //
        // Key overlap: MLP2 GEMMs read bank[cur] while W1 dequant writes bank[nxt].
        // MI300X's 8-stack HBM3 (5.3 TB/s) can service reads+writes to different
        // memory regions concurrently, providing 15-25% overlap benefit.
        //
        // Memory: 2 banks x 8 buffers x 33.2MB = 531.2 MB (was 265.6 MB)
        // -----------------------------------------------------------------
        ensure_grouped_staging(device_id_, num_tokens * top_k);
        GroupedGemmStaging& stg = s_staging[device_id_];

        // Collect ALL active experts into a flat list, then pack into
        // dense batches of GROUPED_BATCH_SIZE.  During decode (top_k=4),
        // this gives 1 batch of 4 experts instead of up to 4 batches of 1,
        // reducing dequant+sync+GEMM cycles by ~4x.
        struct ActiveExpert {
            int expert_id;
            int start;       // offset into permuted_tokens
            int count;       // number of tokens assigned
        };

        ActiveExpert active_experts[ModelConfig::num_experts];  // max possible
        int num_active = 0;

        for (int e = 0; e < static_cast<int>(experts_per_gpu); ++e) {
            int expert_start = h_expert_offsets_pinned_[e];
            int expert_tokens = h_expert_offsets_pinned_[e + 1] - expert_start;
            if (expert_tokens == 0) continue;
            active_experts[num_active++] = {e, expert_start, expert_tokens};
        }

        // Pack active experts into dense batches
        struct BatchInfo {
            int active_count;
            int expert_ids[GROUPED_BATCH_SIZE];
            int expert_starts[GROUPED_BATCH_SIZE];
            int expert_counts[GROUPED_BATCH_SIZE];
            int gate_up_offsets[GROUPED_BATCH_SIZE];
            int total_tokens;
        };

        constexpr int MAX_EXPERT_BATCHES = 16;  // 128 experts / 8 = max 16
        BatchInfo batch_infos[MAX_EXPERT_BATCHES];
        int num_batches = 0;

        for (int i = 0; i < num_active; i += GROUPED_BATCH_SIZE) {
            BatchInfo& bi = batch_infos[num_batches];
            bi.active_count = 0;
            bi.total_tokens = 0;

            int batch_end = min(i + GROUPED_BATCH_SIZE, num_active);
            for (int j = i; j < batch_end; ++j) {
                int s = bi.active_count;
                bi.expert_ids[s] = active_experts[j].expert_id;
                bi.expert_starts[s] = active_experts[j].start;
                bi.expert_counts[s] = active_experts[j].count;
                bi.gate_up_offsets[s] = bi.total_tokens;
                bi.total_tokens += active_experts[j].count;
                bi.active_count++;
            }

            if (bi.active_count > 0) num_batches++;
        }

        // Main loop: fused MXFP4 GEMV (decode) or dequant → sync → GEMM (prefill).
        // Dense batching ensures minimal batches (decode: 1 batch of 4 experts).
        for (int b = 0; b < num_batches; ++b) {
            BatchInfo& bi = batch_infos[b];

            // Check if all experts in batch are M=1 (decode mode).
            // If so, use fused MXFP4 GEMV — reads FP4 directly, no staging buffer.
            bool all_decode = true;
            for (int s = 0; s < bi.active_count; ++s)
                if (bi.expert_counts[s] != 1) { all_decode = false; break; }

            if (all_decode) {
                // ============================================================
                // DECODE PATH: Fused MXFP4 GEMV (Phase 2)
                // No dequant kernel, no staging buffer, no hipStreamSync.
                // Reads FP4 packed weights + E8M0 scales → dequant in VGPRs → dot product.
                // ============================================================

                // -- W1: fused MXFP4 GEMV (with bias fused) --
                PROF("w1_fused_gemv", "moe_sub", device_id_, compute_stream_);
                for (int s = 0; s < bi.active_count; ++s) {
                    int e = bi.expert_ids[s];
                    const __hip_bfloat16* a = permuted_tokens_ + static_cast<int64_t>(bi.expert_starts[s]) * hidden_size;
                    __hip_bfloat16* c = stg.gate_up + static_cast<int64_t>(bi.gate_up_offsets[s]) * expert_gate_up_size;
                    const __hip_bfloat16* w1_bias = nullptr;
                    if (gate_up_proj_bias_) {
                        int global_e = e + gpu_id_ * experts_per_gpu;
                        w1_bias = gate_up_proj_bias_ + static_cast<int64_t>(global_e) * expert_gate_up_size;
                    }
                    fused_mxfp4_gemv_forward(a, expert_mlp1_[e].packed, expert_mlp1_[e].scales,
                                             c, expert_gate_up_size, hidden_size, compute_stream_,
                                             w1_bias);
                }
                PROF_END(device_id_, compute_stream_);

                // -- SwiGLU (bias already fused into GEMV above) --
                PROF("bias_swiglu", "moe_sub", device_id_, compute_stream_);
                swiglu_forward(
                    stg.gate_up, stg.swiglu,
                    bi.total_tokens, expert_intermediate_size,
                    swiglu_limit, compute_stream_);
                PROF_END(device_id_, compute_stream_);

                // -- W2: fused MXFP4 GEMV (with bias fused) --
                PROF("w2_fused_gemv", "moe_sub", device_id_, compute_stream_);
                for (int s = 0; s < bi.active_count; ++s) {
                    int e = bi.expert_ids[s];
                    const __hip_bfloat16* a = stg.swiglu + static_cast<int64_t>(bi.gate_up_offsets[s]) * expert_intermediate_size;
                    __hip_bfloat16* c = expert_out_buf_ + static_cast<int64_t>(bi.expert_starts[s]) * hidden_size;
                    const __hip_bfloat16* w2_bias = nullptr;
                    if (down_proj_bias_) {
                        int global_e = e + gpu_id_ * experts_per_gpu;
                        w2_bias = down_proj_bias_ + static_cast<int64_t>(global_e) * hidden_size;
                    }
                    fused_mxfp4_gemv_forward(a, expert_mlp2_[e].packed, expert_mlp2_[e].scales,
                                             c, hidden_size, expert_intermediate_size, compute_stream_,
                                             w2_bias);
                }
                PROF_END(device_id_, compute_stream_);

            } else {
                // ============================================================
                // PREFILL PATH: dequant → sync → hipBLASLt GEMM (or GEMV for M=1)
                // ============================================================

                // -- W1 dequant --
                PROF("w1_dequant", "moe_sub", device_id_, compute_stream_);
                constexpr int mlp1_elements = hidden_size * expert_gate_up_size;
                for (int s = 0; s < bi.active_count; ++s) {
                    int e = bi.expert_ids[s];
                    h_dequant_descs_[s].packed = expert_mlp1_[e].packed;
                    h_dequant_descs_[s].scales = expert_mlp1_[e].scales;
                    h_dequant_descs_[s].output = stg.dequant[0][s];
                    h_dequant_descs_[s].num_elements = mlp1_elements;
                }
                CUDA_CHECK(hipMemcpy(d_dequant_descs_, h_dequant_descs_,
                    bi.active_count * sizeof(MxFp4BatchDesc), hipMemcpyHostToDevice));
                mxfp4_dequant_batched(d_dequant_descs_, bi.active_count,
                                       mlp1_elements, compute_stream_);
                PROF_END(device_id_, compute_stream_);

                // hipBLASLt requires sync after custom dequant kernels (ROCm bug)
                CUDA_CHECK(hipStreamSynchronize(compute_stream_));

                // -- MLP1 GEMMs --
                PROF("w1_gemm", "moe_sub", device_id_, compute_stream_);
                for (int s = 0; s < bi.active_count; ++s) {
                    const __hip_bfloat16* a = permuted_tokens_ + static_cast<int64_t>(bi.expert_starts[s]) * hidden_size;
                    const __hip_bfloat16* b = stg.dequant[0][s];
                    __hip_bfloat16* c = stg.gate_up + static_cast<int64_t>(bi.gate_up_offsets[s]) * expert_gate_up_size;
                    if (bi.expert_counts[s] == 1) {
                        gemv_bf16_forward(a, b, c, expert_gate_up_size, hidden_size, compute_stream_);
                    } else {
                        cublas_compute_->gemm_bf16_lt(
                            a, b, c, bi.expert_counts[s],
                            expert_gate_up_size, hidden_size,
                            compute_stream_, 1.0f, 0.0f, /*bias=*/nullptr, /*transB=*/true);
                    }
                }
                PROF_END(device_id_, compute_stream_);

                // -- Add expert MLP1 biases + SwiGLU --
                PROF("bias_swiglu", "moe_sub", device_id_, compute_stream_);
                if (gate_up_proj_bias_) {
                    for (int s = 0; s < bi.active_count; ++s) {
                        int global_e = bi.expert_ids[s] + gpu_id_ * experts_per_gpu;
                        const __hip_bfloat16* expert_bias = gate_up_proj_bias_
                            + static_cast<int64_t>(global_e) * expert_gate_up_size;
                        __hip_bfloat16* c_ptr = stg.gate_up + static_cast<int64_t>(bi.gate_up_offsets[s]) * expert_gate_up_size;
                        bias_add_rows(c_ptr, expert_bias, bi.expert_counts[s],
                                      expert_gate_up_size, compute_stream_);
                    }
                }

                swiglu_forward(
                    stg.gate_up, stg.swiglu,
                    bi.total_tokens, expert_intermediate_size,
                    swiglu_limit, compute_stream_);
                PROF_END(device_id_, compute_stream_);

                // -- W2 dequant --
                PROF("w2_dequant", "moe_sub", device_id_, compute_stream_);
                constexpr int mlp2_elements = expert_intermediate_size * hidden_size;
                for (int s = 0; s < bi.active_count; ++s) {
                    int e = bi.expert_ids[s];
                    h_dequant_descs_[s].packed = expert_mlp2_[e].packed;
                    h_dequant_descs_[s].scales = expert_mlp2_[e].scales;
                    h_dequant_descs_[s].output = stg.dequant[0][s];
                    h_dequant_descs_[s].num_elements = mlp2_elements;
                }
                CUDA_CHECK(hipMemcpy(d_dequant_descs_, h_dequant_descs_,
                    bi.active_count * sizeof(MxFp4BatchDesc), hipMemcpyHostToDevice));
                mxfp4_dequant_batched(d_dequant_descs_, bi.active_count,
                                       mlp2_elements, compute_stream_);
                PROF_END(device_id_, compute_stream_);

                // hipBLASLt requires sync after custom dequant kernels
                CUDA_CHECK(hipStreamSynchronize(compute_stream_));

                // -- MLP2 GEMMs --
                PROF("w2_gemm", "moe_sub", device_id_, compute_stream_);
                for (int s = 0; s < bi.active_count; ++s) {
                    const __hip_bfloat16* a = stg.swiglu + static_cast<int64_t>(bi.gate_up_offsets[s]) * expert_intermediate_size;
                    const __hip_bfloat16* b = stg.dequant[0][s];
                    __hip_bfloat16* c = expert_out_buf_ + static_cast<int64_t>(bi.expert_starts[s]) * hidden_size;
                    if (bi.expert_counts[s] == 1) {
                        gemv_bf16_forward(a, b, c, hidden_size, expert_intermediate_size, compute_stream_);
                    } else {
                        cublas_compute_->gemm_bf16_lt(
                            a, b, c, bi.expert_counts[s],
                            hidden_size, expert_intermediate_size,
                            compute_stream_, 1.0f, 0.0f, /*bias=*/nullptr, /*transB=*/true);
                    }
                }
                PROF_END(device_id_, compute_stream_);
            }

            // -- Add expert MLP2 biases (prefill path only; decode path fuses bias into GEMV) --
            if (!all_decode) {
                PROF("w2_bias", "moe_sub", device_id_, compute_stream_);
                if (down_proj_bias_) {
                    for (int s = 0; s < bi.active_count; ++s) {
                        int global_e = bi.expert_ids[s] + gpu_id_ * experts_per_gpu;
                        const __hip_bfloat16* expert_bias = down_proj_bias_
                            + static_cast<int64_t>(global_e) * hidden_size;
                        __hip_bfloat16* c_ptr = expert_out_buf_ + static_cast<int64_t>(bi.expert_starts[s]) * hidden_size;
                        bias_add_rows(c_ptr, expert_bias, bi.expert_counts[s],
                                      hidden_size, compute_stream_);
                    }
                }
                PROF_END(device_id_, compute_stream_);
            }

        }

        // -----------------------------------------------------------------
        // Step 6: Exchange expert outputs via N-way grouped P2P
        // -----------------------------------------------------------------
        PROF("ep_exchange", "moe_sub", device_id_, comm_stream_);
        ep_comm_combine_v(
            ep_comm_,
            expert_out_buf_,                // send: our local expert outputs
            total_local_tokens,             // send_count
            recv_buffer_,                   // recv: packed [peer_0|...|peer_{N-1}]
            h_per_peer_counts_pinned_,      // [ep_size] tokens per peer
            h_peer_recv_offsets_pinned_,    // [ep_size] offsets into recv_buffer
            compute_stream_,
            comm_stream_);
        PROF_END(device_id_, comm_stream_);

        // Wait for RCCL exchange to complete before compute_stream_ reads recv_buffer_.
        // A host-side hipStreamSynchronize(comm_stream_) is NOT sufficient:
        // it only blocks the CPU, not the GPU's compute pipeline. The combine
        // kernel on compute_stream_ can execute before RCCL writes land in
        // device memory. We need a GPU-side dependency via hipEvent.
        {
            hipEvent_t nccl_done;
            CUDA_CHECK(hipEventCreateWithFlags(&nccl_done, hipEventDisableTiming));
            CUDA_CHECK(hipEventRecord(nccl_done, comm_stream_));
            CUDA_CHECK(hipStreamWaitEvent(compute_stream_, nccl_done, 0));
            CUDA_CHECK(hipEventDestroy(nccl_done));
        }

        // -----------------------------------------------------------------
        // Steps 7+8 fused: Combine + residual add in one kernel
        //
        // Using the pre-built gather_map from step 4:
        //   gather_map[slot] >= 0: read from expert_out_buf_[index]
        //   gather_map[slot] <  0: read from recv_buffer_[-(index+1)]
        //
        // For each token:
        //   output[token] = input[token] + sum_k weight[k] * expert_out[slot(k)]
        //
        // Fusing the residual add saves 1 kernel launch + 2 memory passes
        // of intermediate moe_output_buf_ (~3us per layer x 36 = ~108us).
        // -----------------------------------------------------------------
        PROF("combine", "moe_sub", device_id_, compute_stream_);
        moe_combine_forward(
            output,
            expert_out_buf_,
            recv_buffer_,
            gather_map_,
            expert_weights_buf_,
            residual,  // fused residual: output = residual + combine_result
            num_tokens, hidden_size, top_k,
            compute_stream_);
        PROF_END(device_id_, compute_stream_);

        // Drain both MoE streams before returning to main_stream.
        // The combine kernel on compute_stream_ reads recv_buffer_ (written
        // by RCCL on comm_stream_) and expert_out_buf_ / gather_map_. These
        // shared device buffers are reused by the next layer's MoE forward,
        // so both streams must be fully idle before main_stream proceeds.
        CUDA_CHECK(hipStreamSynchronize(comm_stream_));
        CUDA_CHECK(hipStreamSynchronize(compute_stream_));
    }
};

// ---------------------------------------------------------------------------
// Wrapper function for cross-compilation-unit access from transformer.cu
// ---------------------------------------------------------------------------

void moe_layer_forward(
    MoELayer* layer,
    const __hip_bfloat16* input,
    __hip_bfloat16* output,
    int num_tokens,
    hipStream_t stream)
{
    layer->forward(input, output, num_tokens, stream);
}

void moe_layer_forward_prenormed(
    MoELayer* layer,
    const __hip_bfloat16* normed_input,
    const __hip_bfloat16* residual_input,
    __hip_bfloat16* output,
    int num_tokens,
    hipStream_t stream)
{
    layer->forward_prenormed(normed_input, residual_input, output, num_tokens, stream);
}

const __hip_bfloat16* moe_layer_get_ffn_norm_weight(MoELayer* layer) {
    return layer->get_ffn_norm_weight();
}

MoELayer* moe_layer_create() { return new MoELayer(); }
void moe_layer_destroy(MoELayer* l) { if (l) { l->destroy(); delete l; } }

void moe_layer_init(
    MoELayer* layer,
    int layer_id, int device_id, int gpu_id,
    EPComm* ep_comm, const CublasHandle* cublas_compute, const CublasHandle* cublas_comm,
    hipStream_t compute_stream, hipStream_t comm_stream,
    const __hip_bfloat16* moe_norm_weight, const __hip_bfloat16* router_weight,
    const __hip_bfloat16* router_bias,
    const __hip_bfloat16* gate_up_proj_bias, const __hip_bfloat16* down_proj_bias,
    MXFP4Weight* expert_mlp1, MXFP4Weight* expert_mlp2,
    __hip_bfloat16* norm_buf, __hip_bfloat16* router_logits_buf,
    int32_t* expert_indices_buf, float* expert_weights_buf,
    int32_t* tokens_per_expert, __hip_bfloat16* permuted_tokens,
    int32_t* expert_offsets, int32_t* gather_map,
    int32_t* per_peer_counts, int32_t* peer_recv_offsets,
    __hip_bfloat16* recv_buffer,
    __hip_bfloat16* dequant_mlp1_buf_0, __hip_bfloat16* dequant_mlp1_buf_1,
    __hip_bfloat16* dequant_mlp2_buf_0, __hip_bfloat16* dequant_mlp2_buf_1,
    __hip_bfloat16* gate_up_buf, __hip_bfloat16* swiglu_buf,
    __hip_bfloat16* expert_out_buf, __hip_bfloat16* moe_output_buf,
    ncclComm_t nccl_comm) {
    layer->init(layer_id, device_id, gpu_id, ep_comm, cublas_compute, cublas_comm,
                compute_stream, comm_stream, moe_norm_weight, router_weight,
                router_bias, gate_up_proj_bias, down_proj_bias,
                expert_mlp1, expert_mlp2, norm_buf, router_logits_buf,
                expert_indices_buf, expert_weights_buf, tokens_per_expert,
                permuted_tokens, expert_offsets, gather_map,
                per_peer_counts, peer_recv_offsets, recv_buffer,
                dequant_mlp1_buf_0, dequant_mlp1_buf_1,
                dequant_mlp2_buf_0, dequant_mlp2_buf_1,
                gate_up_buf, swiglu_buf, expert_out_buf, moe_output_buf,
                nccl_comm);
}

} // namespace gptoss
