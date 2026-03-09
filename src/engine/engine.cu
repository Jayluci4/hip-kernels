// GPT-OSS-120B Inference Engine
// Main orchestrator for multi-GPU inference with decoupled TP x EP parallelism.
//
// Architecture:
//   - Single process controlling tp_size * ep_size GPUs via hipSetDevice()
//   - Rank layout: global_rank = tp_rank + ep_rank * tp_size
//   - TP groups: attention AllReduce (QKV column-parallel, O-proj row-parallel)
//   - EP groups: expert output exchange via N-way grouped P2P
//   - RCCL sub-communicators created via ncclCommSplit for TP and EP groups
//   - GPU 0's logits are used for sampling
//
// Lifecycle:
//   1. init(weights_dir)  -- detect GPUs, load weights, warm up
//   2. generate(...)      -- autoregressive generation loop
//   3. shutdown()         -- release all resources
//
// IMPORTANT: This file uses only forward-declared types and extern wrapper
// functions to interact with subsystem classes. The complete types are
// defined in their respective .cu files.

#include "hip_compat.h"
#include <rccl/rccl.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "config.h"
#include "tensor.h"
#include "cuda_utils.h"
#include "nccl_utils.h"

namespace gptoss {

// ---------------------------------------------------------------------------
// Forward declarations of classes defined in other translation units
// ---------------------------------------------------------------------------

class Transformer;
class NCCLManager;
class KVCacheManager;
class WeightLoader;
class BufferManager;
class AttentionLayer;
class MoELayer;
class EPComm;

// ---------------------------------------------------------------------------
// Extern wrapper function declarations
// ---------------------------------------------------------------------------

// NCCLManager wrappers (nccl_manager.cu)
extern NCCLManager* nccl_manager_create();
extern void nccl_manager_destroy(NCCLManager* mgr);
extern void nccl_manager_init_with_id(NCCLManager* mgr, int rank, int world_size, const ncclUniqueId& nccl_id);
extern ncclComm_t nccl_manager_comm(NCCLManager* mgr);
extern hipStream_t nccl_manager_comm_stream(NCCLManager* mgr);
extern void nccl_manager_create_sub_communicators(NCCLManager* mgr, int tp_size, int ep_size);
extern void nccl_manager_set_rank(NCCLManager* mgr, int rank, int world_size);
extern void nccl_manager_set_sub_comm_ranks(NCCLManager* mgr, int tp_size, int ep_size);
extern void nccl_manager_create_tp_comm(NCCLManager* mgr);
extern void nccl_manager_create_ep_comm(NCCLManager* mgr);
extern ncclComm_t nccl_manager_tp_comm(NCCLManager* mgr);
extern ncclComm_t nccl_manager_ep_comm(NCCLManager* mgr);
extern int nccl_manager_tp_rank(NCCLManager* mgr);
extern int nccl_manager_ep_rank(NCCLManager* mgr);

// BufferManager wrappers (buffer_manager.cu)
extern BufferManager* buffer_manager_create();
extern void buffer_manager_destroy(BufferManager* mgr);
extern void buffer_manager_init(BufferManager* mgr, int max_batch_tokens, int hidden_size, int device_id);

// EPComm wrappers (ep_comm.cu)
extern EPComm* ep_comm_create();
extern void ep_comm_destroy(EPComm* ep);
extern void ep_comm_init(EPComm* ep, int device_id, int ep_rank, int ep_size,
                          int tp_rank, int tp_size, ncclComm_t ep_comm);

// KVCacheManager wrappers (kv_cache_manager.cu)
extern KVCacheManager* kv_cache_manager_create();
extern void kv_cache_manager_destroy(KVCacheManager* mgr);
extern void kv_cache_manager_init(KVCacheManager* mgr, int max_seqs, int max_seq_len,
                                   int num_layers, int num_kv_heads, int head_dim,
                                   int block_size, int device_id,
                                   int64_t target_total_tokens);
extern void kv_cache_manager_free_sequence(KVCacheManager* mgr, int seq_id);
extern __hip_bfloat16* kv_cache_manager_k_cache(KVCacheManager* mgr);
extern __hip_bfloat16* kv_cache_manager_v_cache(KVCacheManager* mgr);
extern int kv_cache_manager_num_blocks(KVCacheManager* mgr);

// WeightLoader wrappers (weight_loader.cu)
extern WeightLoader* weight_loader_create();
extern void weight_loader_destroy(WeightLoader* wl);
extern void weight_loader_init(WeightLoader* wl, const std::string& weights_dir,
                                int tp_rank, int ep_rank, int tp_size, int ep_size,
                                int device_id);
extern void weight_loader_load_all(WeightLoader* wl);
extern const __hip_bfloat16* weight_loader_embedding(WeightLoader* wl);
extern const __hip_bfloat16* weight_loader_final_norm(WeightLoader* wl);
extern const __hip_bfloat16* weight_loader_lm_head(WeightLoader* wl);
extern const __hip_bfloat16* weight_loader_attn_qkv(WeightLoader* wl, int layer);
extern const __hip_bfloat16* weight_loader_attn_o_proj(WeightLoader* wl, int layer);
extern const __hip_bfloat16* weight_loader_attn_norm(WeightLoader* wl, int layer);
extern const __hip_bfloat16* weight_loader_attn_qkv_bias(WeightLoader* wl, int layer);
extern const __hip_bfloat16* weight_loader_attn_o_proj_bias(WeightLoader* wl, int layer);
extern const float* weight_loader_attn_sinks(WeightLoader* wl, int layer);
extern const __hip_bfloat16* weight_loader_moe_router(WeightLoader* wl, int layer);
extern const __hip_bfloat16* weight_loader_moe_norm(WeightLoader* wl, int layer);
extern const __hip_bfloat16* weight_loader_moe_router_bias(WeightLoader* wl, int layer);
extern const __hip_bfloat16* weight_loader_expert_gate_up_bias(WeightLoader* wl, int layer);
extern const __hip_bfloat16* weight_loader_expert_down_bias(WeightLoader* wl, int layer);
extern const uint8_t* weight_loader_expert_mlp1_packed(WeightLoader* wl, int layer, int local_expert);
extern const uint8_t* weight_loader_expert_mlp1_scales(WeightLoader* wl, int layer, int local_expert);
extern int weight_loader_expert_mlp1_numel(WeightLoader* wl, int layer, int local_expert);
extern const uint8_t* weight_loader_expert_mlp2_packed(WeightLoader* wl, int layer, int local_expert);
extern const uint8_t* weight_loader_expert_mlp2_scales(WeightLoader* wl, int layer, int local_expert);
extern int weight_loader_expert_mlp2_numel(WeightLoader* wl, int layer, int local_expert);

// Transformer wrappers (transformer.cu)
extern Transformer* transformer_create();
extern void transformer_destroy(Transformer* t);
extern void transformer_init(Transformer* t, int rank, ncclComm_t nccl_comm,
                              ncclComm_t ep_nccl_comm,
                              CublasHandle* cublas, const __hip_bfloat16* embedding_table,
                              const __hip_bfloat16* final_norm_weight,
                              const __hip_bfloat16* lm_head_weight,
                              AttentionLayer** attn_layers, MoELayer** moe_layers,
                              int max_tokens);
extern void transformer_prefill(Transformer* t, const int* token_ids_host,
                                 const int* positions_host, int seq_len,
                                 __hip_bfloat16* logits_output,
                                 const int32_t* slot_mapping_host = nullptr);
extern void transformer_decode(Transformer* t, const int* token_ids_host,
                                const int* positions_host, const int32_t* slot_mapping_host,
                                const int32_t* block_table_host, const int32_t* seq_lens_host,
                                int num_seqs, int max_num_blocks, __hip_bfloat16* logits_output);
extern hipStream_t transformer_compute_stream(Transformer* t);

// AttentionLayer / MoELayer factory wrappers (attention.cu / moe_layer.cu)
extern AttentionLayer* attention_layer_create();
extern void attention_layer_destroy(AttentionLayer* l);
extern void attention_layer_init(
    AttentionLayer* layer,
    int layer_id, int device_id, int rank, ncclComm_t nccl_comm,
    const CublasHandle* cublas,
    const __hip_bfloat16* qkv_weight, const __hip_bfloat16* qkv_bias,
    const __hip_bfloat16* o_proj_weight, const __hip_bfloat16* o_proj_bias,
    const __hip_bfloat16* attn_norm_weight,
    const float* cos_table, const float* sin_table, const float2* cos_sin_table,
    const float* sink_values,
    __hip_bfloat16* k_cache, __hip_bfloat16* v_cache, int max_num_blocks,
    __hip_bfloat16* norm_buf, __hip_bfloat16* qkv_buf,
    __hip_bfloat16* q_buf, __hip_bfloat16* k_buf, __hip_bfloat16* v_buf,
    __hip_bfloat16* q_rope_buf, __hip_bfloat16* k_rope_buf,
    __hip_bfloat16* attn_out_buf, __hip_bfloat16* proj_out_buf);

// MXFP4Weight struct (must match moe_layer.cu definition)
struct MXFP4Weight {
    const uint8_t* packed;
    const uint8_t* scales;
    int num_elements;
};

extern MoELayer* moe_layer_create();
extern void moe_layer_destroy(MoELayer* l);
extern void moe_layer_init(
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
    ncclComm_t nccl_comm);

// RoPE init (rope.cu)
extern void rope_init(float** cos_table, float** sin_table, float2** cos_sin_table,
                       int max_seq_len, int head_dim, float theta, hipStream_t stream);

// ---------------------------------------------------------------------------
// BF16 -> FP32 conversion kernel
// ---------------------------------------------------------------------------

__global__ void bf16_to_fp32_kernel(const __hip_bfloat16* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __bfloat162float(in[idx]);
}

static void bf16_to_fp32(const __hip_bfloat16* in, float* out, int n, hipStream_t stream) {
    if (n == 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bf16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
    CUDA_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------------------
// Per-GPU resource bundle
// ---------------------------------------------------------------------------
struct GPUResources {
    int device_id = -1;

    // Owned HIP resources
    CudaStream compute_stream;
    CudaStream comm_stream;
    CublasHandle cublas;

    // Sub-systems (one instance per GPU, managed via wrapper functions)
    NCCLManager*    nccl     = nullptr;
    Transformer*    model    = nullptr;
    KVCacheManager* kv_cache = nullptr;
    WeightLoader*   weights  = nullptr;
    BufferManager*  buffers  = nullptr;
    EPComm*         ep_comm  = nullptr;

    // Per-layer objects (owned, created during init_subsystems)
    AttentionLayer* attn_layers[ModelConfig::num_layers] = {};
    MoELayer*       moe_layers[ModelConfig::num_layers]  = {};

    // MXFP4Weight descriptors per layer (experts_per_gpu per layer)
    MXFP4Weight* mxfp4_mlp1[ModelConfig::num_layers] = {};  // arrays of experts_per_gpu
    MXFP4Weight* mxfp4_mlp2[ModelConfig::num_layers] = {};

    // RoPE tables (shared across layers, per-GPU)
    float*  rope_cos  = nullptr;
    float*  rope_sin  = nullptr;
    float2* rope_cs   = nullptr;

    // Shared attention scratch buffers (reused across layers)
    __hip_bfloat16* attn_norm_buf   = nullptr;
    __hip_bfloat16* attn_qkv_buf   = nullptr;
    __hip_bfloat16* attn_q_buf     = nullptr;
    __hip_bfloat16* attn_k_buf     = nullptr;
    __hip_bfloat16* attn_v_buf     = nullptr;
    __hip_bfloat16* attn_q_rope    = nullptr;
    __hip_bfloat16* attn_k_rope    = nullptr;
    __hip_bfloat16* attn_out_buf   = nullptr;
    __hip_bfloat16* attn_proj_out  = nullptr;

    // Shared MoE scratch buffers (reused across layers)
    __hip_bfloat16* moe_norm_buf        = nullptr;
    __hip_bfloat16* moe_router_logits   = nullptr;
    int32_t*       moe_expert_indices   = nullptr;
    float*         moe_expert_weights   = nullptr;
    int32_t*       moe_tokens_per_expert = nullptr;
    __hip_bfloat16* moe_permuted_tokens  = nullptr;
    int32_t*       moe_expert_offsets    = nullptr;
    int32_t*       moe_gather_map       = nullptr;
    int32_t*       moe_per_peer_counts  = nullptr;
    int32_t*       moe_peer_recv_offs   = nullptr;
    __hip_bfloat16* moe_recv_buffer      = nullptr;
    __hip_bfloat16* moe_dq_mlp1[2]      = {};
    __hip_bfloat16* moe_dq_mlp2[2]      = {};
    __hip_bfloat16* moe_gate_up_buf      = nullptr;
    __hip_bfloat16* moe_swiglu_buf       = nullptr;
    __hip_bfloat16* moe_expert_out       = nullptr;
    __hip_bfloat16* moe_output_buf       = nullptr;

    // Staging buffer on host for copying logits back from GPU 0
    float* host_logits = nullptr;

    // Device-side logits buffers
    __hip_bfloat16* device_logits_bf16 = nullptr;  // BF16 output from Transformer
    float*         device_logits_fp32 = nullptr;   // FP32 conversion for sampling
};

// ---------------------------------------------------------------------------
// InferenceEngine
// ---------------------------------------------------------------------------

class InferenceEngine {
public:
    // Special token IDs
    static constexpr int32_t EOS_TOKEN_ID  = 200002;
    static constexpr int32_t PAD_TOKEN_ID  = 199999;
    static constexpr int      NUM_GPUS     = ModelConfig::world_size; // 2

    InferenceEngine() = default;
    ~InferenceEngine() { shutdown(); }

    // Non-copyable, non-movable
    InferenceEngine(const InferenceEngine&)            = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&)                 = delete;
    InferenceEngine& operator=(InferenceEngine&&)      = delete;

    // NOTE: Full HIP graph capture for decode is blocked because:
    //   1. RCCL ops (all-reduce, send/recv, all-gather) require RCCL 2.19+
    //      graph-safe communicators (ncclCommRegister)
    //   2. MoE forward has host-blocking hipStreamSynchronize for expert
    //      offsets D2H (illegal during graph capture)
    //   3. Variable-shape params (seq_lens, block_table, position) change
    //      every step, preventing static sub-graph capture for attention
    //
    // Instead we optimize decode overhead via:
    //   - Pre-allocated host buffers (no per-step heap allocation)
    //   - Reusable block_table with incremental growth
    //   - Monotonic position tracking to avoid redundant setup

    // ------------------------------------------------------------------
    // Initialization
    // ------------------------------------------------------------------
    void init(const std::string& weights_dir) {
        if (initialized_) {
            throw std::runtime_error("[InferenceEngine] Already initialized");
        }

        fprintf(stderr, "[InferenceEngine] Initializing GPT-OSS-120B engine...\n");

        // ---- Step 1: Detect and verify GPUs ----
        detect_and_verify_gpus();

        // ---- Step 2: Initialize RCCL communicators ----
        init_nccl();

        // ---- Step 3: Create per-GPU HIP resources ----
        for (int r = 0; r < NUM_GPUS; ++r) {
            init_gpu_resources(r);
        }

        // ---- Step 4: Load weights ----
        load_weights(weights_dir);

        // ---- Step 5: Initialize sub-systems ----
        init_subsystems();

        // ---- Step 6: Set L2 cache persistence for KV cache (MI300X) ----
        set_l2_persistence();

        // ---- Step 7: Pre-allocate decode state ----
        {
            int total_blocks = kv_cache_manager_num_blocks(gpus_[0].kv_cache);
            int blocks_per_layer = total_blocks / ModelConfig::num_layers;
            decode_state_.init(blocks_per_layer);
        }

        // ---- Step 8: Warm up ----
        warmup();

        initialized_ = true;
        fprintf(stderr, "[InferenceEngine] Initialization complete.\n");
    }

    // ------------------------------------------------------------------
    // Autoregressive generation
    // ------------------------------------------------------------------
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens,
                                  int max_tokens     = 512,
                                  float temperature  = 0.8f,
                                  float top_p        = 0.95f) {
        if (!initialized_) {
            throw std::runtime_error("[InferenceEngine] Not initialized");
        }
        if (prompt_tokens.empty()) {
            throw std::runtime_error("[InferenceEngine] Empty prompt");
        }
        if (max_tokens <= 0) {
            throw std::runtime_error("[InferenceEngine] max_tokens must be > 0");
        }

        const int seq_len = static_cast<int>(prompt_tokens.size());
        fprintf(stderr, "[InferenceEngine::generate] prompt_len=%d, first_token=%d, last_token=%d\n",
                seq_len, prompt_tokens[0], prompt_tokens[seq_len-1]);

        // Output accumulator: starts with the full prompt
        std::vector<int32_t> output_tokens(prompt_tokens.begin(), prompt_tokens.end());
        output_tokens.reserve(seq_len + max_tokens);

        // ---- Prefill phase ----
        // Process the entire prompt in one forward pass.
        run_prefill(prompt_tokens);

        // Diagnostic: dump top-10 logits after prefill
        {
            const float* logits = gpus_[0].host_logits;
            const int V = ModelConfig::vocab_size;
            // Find top 10
            int top_ids[10] = {};
            float top_vals[10];
            for (int i = 0; i < 10; i++) top_vals[i] = -1e30f;
            for (int i = 0; i < V; i++) {
                int pos = -1;
                for (int j = 0; j < 10; j++) {
                    if (logits[i] > top_vals[j]) {
                        pos = j;
                        break;
                    }
                }
                if (pos >= 0) {
                    for (int j = 9; j > pos; j--) {
                        top_ids[j] = top_ids[j-1];
                        top_vals[j] = top_vals[j-1];
                    }
                    top_ids[pos] = i;
                    top_vals[pos] = logits[i];
                }
            }
            fprintf(stderr, "[DIAG] Prefill top-10 logits:\n");
            for (int i = 0; i < 10; i++) {
                fprintf(stderr, "  [%d] token=%d logit=%.4f\n", i, top_ids[i], top_vals[i]);
            }
            // Also show logit stats
            float lmin = logits[0], lmax = logits[0];
            double lsum = 0;
            int nan_count = 0, inf_count = 0;
            for (int i = 0; i < V; i++) {
                if (std::isnan(logits[i])) { nan_count++; continue; }
                if (std::isinf(logits[i])) { inf_count++; continue; }
                if (logits[i] < lmin) lmin = logits[i];
                if (logits[i] > lmax) lmax = logits[i];
                lsum += logits[i];
            }
            fprintf(stderr, "[DIAG] Logit stats: min=%.4f max=%.4f mean=%.4f nans=%d infs=%d\n",
                    lmin, lmax, (float)(lsum / V), nan_count, inf_count);
        }

        // Sample the first new token from the prefill logits
        int32_t next_token = sample(gpus_[0].host_logits, temperature, top_p);
        output_tokens.push_back(next_token);

        if (next_token == EOS_TOKEN_ID) {
            return output_tokens;
        }

        // ---- Decode phase ----
        // Generate tokens one at a time.
        for (int step = 0; step < max_tokens - 1; ++step) {
            int position = seq_len + step;
            run_decode_step(next_token, position);

            next_token = sample(gpus_[0].host_logits, temperature, top_p);
            output_tokens.push_back(next_token);

            if (next_token == EOS_TOKEN_ID) {
                break;
            }
        }

        return output_tokens;
    }

    // ------------------------------------------------------------------
    // Batched autoregressive generation
    // ------------------------------------------------------------------
    std::vector<std::vector<int32_t>> generate_batch(
            const std::vector<std::vector<int32_t>>& prompts,
            int max_tokens     = 512,
            float temperature  = 0.0f,
            float top_p        = 1.0f) {
        if (!initialized_) {
            throw std::runtime_error("[InferenceEngine] Not initialized");
        }
        const int B = static_cast<int>(prompts.size());
        if (B == 0) {
            throw std::runtime_error("[InferenceEngine] Empty batch");
        }
        if (B > ModelConfig::max_batch_size) {
            throw std::runtime_error("[InferenceEngine] Batch size exceeds max_batch_size");
        }
        if (max_tokens <= 0) {
            throw std::runtime_error("[InferenceEngine] max_tokens must be > 0");
        }

        // All prompts must have the same length for batched decode
        // (different lengths would need padding or continuous batching)
        const int prompt_len = static_cast<int>(prompts[0].size());
        for (int i = 1; i < B; ++i) {
            if (static_cast<int>(prompts[i].size()) != prompt_len) {
                throw std::runtime_error(
                    "[InferenceEngine] All prompts must have the same length for batched generation");
            }
        }

        fprintf(stderr, "[InferenceEngine::generate_batch] B=%d, prompt_len=%d, max_tokens=%d\n",
                B, prompt_len, max_tokens);

        // Reconfigure block table for this batch size
        decode_state_.setup_block_table(B);

        // Verify the KV cache can hold the full generation
        int total_tokens_per_seq = prompt_len + max_tokens;
        int blocks_needed = (total_tokens_per_seq + ModelConfig::kv_block_size - 1)
                          / ModelConfig::kv_block_size;
        if (blocks_needed > decode_state_.blocks_per_seq) {
            fprintf(stderr, "[InferenceEngine] ERROR: B=%d needs %d blocks/seq (%d tokens) "
                    "but only %d blocks/seq available\n",
                    B, blocks_needed, total_tokens_per_seq, decode_state_.blocks_per_seq);
            throw std::runtime_error("[InferenceEngine] KV cache too small for batch_size * seq_len");
        }

        // Output accumulators: one per sequence
        std::vector<std::vector<int32_t>> outputs(B);
        for (int i = 0; i < B; ++i) {
            outputs[i] = prompts[i];
            outputs[i].reserve(prompt_len + max_tokens);
        }

        // Track which sequences are still active (not EOS)
        std::vector<bool> active(B, true);

        // ---- Prefill phase ----
        // Process each prompt sequentially and sample immediately.
        // Prefill overwrites the shared device_logits_bf16 buffer,
        // so we must convert + sample before the next prompt overwrites it.
        std::vector<int32_t> next_tokens(B);
        for (int i = 0; i < B; ++i) {
            run_prefill_seq(prompts[i], /*seq_id=*/i);
            convert_and_copy_logits(1);
            next_tokens[i] = sample(gpus_[0].host_logits, temperature, top_p);
            outputs[i].push_back(next_tokens[i]);
            if (next_tokens[i] == EOS_TOKEN_ID) {
                active[i] = false;
            }
        }

        // ---- Decode phase ----
        // Now run batched decode steps. All B sequences advance together.
        const int V = ModelConfig::vocab_size;

        for (int step = 0; step < max_tokens - 1; ++step) {
            // Check if all sequences are done
            bool any_active = false;
            for (int i = 0; i < B; ++i) {
                if (active[i]) { any_active = true; break; }
            }
            if (!any_active) break;

            int position = prompt_len + step;

            // Fill decode state for all B sequences
            int max_num_blocks = (position + 1 + ModelConfig::kv_block_size - 1)
                               / ModelConfig::kv_block_size;

            for (int i = 0; i < B; ++i) {
                decode_state_.token_ids[i]    = next_tokens[i];
                decode_state_.positions[i]    = position;
                decode_state_.slot_mapping[i] = i * decode_state_.blocks_per_seq * ModelConfig::kv_block_size + position;
                decode_state_.seq_lens[i]     = position + 1;
            }

            // Pack block table: [B][max_num_blocks] from full [MAX_BATCH][blocks_per_seq]
            decode_state_.packed_block_table.resize(static_cast<size_t>(B) * max_num_blocks);
            for (int i = 0; i < B; ++i) {
                const int32_t* src = decode_state_.block_table.data()
                                   + i * decode_state_.blocks_per_seq;
                int32_t* dst = decode_state_.packed_block_table.data()
                             + i * max_num_blocks;
                std::memcpy(dst, src, max_num_blocks * sizeof(int32_t));
            }

            // Launch batched decode on ALL GPUs concurrently
            std::vector<std::thread> threads;
            for (int r = 0; r < NUM_GPUS; ++r) {
                threads.emplace_back(&InferenceEngine::run_gpu_decode, this,
                                     r, decode_state_.token_ids, decode_state_.positions,
                                     decode_state_.slot_mapping,
                                     decode_state_.packed_block_table.data(),
                                     decode_state_.seq_lens, B, max_num_blocks);
            }
            for (auto& t : threads) t.join();

            // Convert and copy B logits vectors
            convert_and_copy_logits(B);

            // Sample next token for each sequence
            for (int i = 0; i < B; ++i) {
                if (!active[i]) {
                    next_tokens[i] = PAD_TOKEN_ID;
                    continue;
                }
                const float* seq_logits = gpus_[0].host_logits + i * V;
                next_tokens[i] = sample(seq_logits, temperature, top_p);
                outputs[i].push_back(next_tokens[i]);
                if (next_tokens[i] == EOS_TOKEN_ID) {
                    active[i] = false;
                }
            }
        }

        return outputs;
    }

    // ------------------------------------------------------------------
    // Shutdown: release all GPU resources
    // ------------------------------------------------------------------
    void shutdown() {
        if (!initialized_) return;

        fprintf(stderr, "[InferenceEngine] Shutting down...\n");

        // Synchronize all GPUs before cleanup
        for (int r = 0; r < NUM_GPUS; ++r) {
            CUDA_CHECK(hipSetDevice(r));
            CUDA_CHECK(hipDeviceSynchronize());
        }

        // Reset L2 persistence windows on both GPUs
        for (int r = 0; r < NUM_GPUS; ++r) {
            CUDA_CHECK(hipSetDevice(r));
            hipStream_t stream = transformer_compute_stream(gpus_[r].model);
            // Reset the access policy window to default (no persistence)
            hipStreamAttrValue reset_attr = {};
            reset_attr.accessPolicyWindow.num_bytes    = 0;
            reset_attr.accessPolicyWindow.hitRatio     = 0.0f;
            reset_attr.accessPolicyWindow.hitProp      = hipAccessPropertyNormal;
            reset_attr.accessPolicyWindow.missProp     = hipAccessPropertyNormal;
            hipStreamSetAttribute(stream, hipStreamAttributeAccessPolicyWindow, &reset_attr);
        }

        // Destroy sub-systems in reverse order of construction
        for (int r = 0; r < NUM_GPUS; ++r) {
            auto& gpu = gpus_[r];
            CUDA_CHECK(hipSetDevice(r));

            if (gpu.buffers) {
                buffer_manager_destroy(gpu.buffers);
                gpu.buffers = nullptr;
            }
            if (gpu.kv_cache) {
                kv_cache_manager_destroy(gpu.kv_cache);
                gpu.kv_cache = nullptr;
            }
            if (gpu.model) {
                transformer_destroy(gpu.model);
                gpu.model = nullptr;
            }
            if (gpu.weights) {
                weight_loader_destroy(gpu.weights);
                gpu.weights = nullptr;
            }
            if (gpu.nccl) {
                nccl_manager_destroy(gpu.nccl);
                gpu.nccl = nullptr;
            }

            // Free logits buffers
            if (gpu.device_logits_bf16) {
                CUDA_CHECK(hipFree(gpu.device_logits_bf16));
                gpu.device_logits_bf16 = nullptr;
            }
            if (gpu.device_logits_fp32) {
                CUDA_CHECK(hipFree(gpu.device_logits_fp32));
                gpu.device_logits_fp32 = nullptr;
            }
            if (gpu.host_logits) {
                CUDA_CHECK(hipHostFree(gpu.host_logits));
                gpu.host_logits = nullptr;
            }
        }

        initialized_ = false;
        fprintf(stderr, "[InferenceEngine] Shutdown complete.\n");
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------
    bool is_initialized() const { return initialized_; }

private:
    // ==================================================================
    // Initialization helpers
    // ==================================================================

    void detect_and_verify_gpus() {
        // In container environments (Modal/Docker), the HIP driver may not be
        // immediately available. Retry with backoff to handle startup delays.
        int device_count = 0;
        hipError_t hip_err = hipGetDeviceCount(&device_count);
        for (int attempt = 0; hip_err != hipSuccess && attempt < 10; ++attempt) {
            fprintf(stderr, "[InferenceEngine] HIP not ready (attempt %d/10): %s. Retrying in 2s...\n",
                    attempt + 1, hipGetErrorString(hip_err));
            hipGetLastError();  // clear error
            sleep(2);
            hip_err = hipGetDeviceCount(&device_count);
        }
        if (hip_err != hipSuccess) {
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,
                    hipGetErrorString(hip_err));
            abort();
        }

        if (device_count < NUM_GPUS) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "[InferenceEngine] Need %d GPUs but found %d",
                     NUM_GPUS, device_count);
            throw std::runtime_error(msg);
        }

        fprintf(stderr, "[InferenceEngine] Found %d GPU(s)\n", device_count);

        for (int i = 0; i < NUM_GPUS; ++i) {
            hipDeviceProp_t props;
            CUDA_CHECK(hipGetDeviceProperties(&props, i));

            fprintf(stderr, "  GPU %d: %s  (GCN arch %s, %.1f GB, %d CUs)\n",
                    i, props.name, props.gcnArchName,
                    static_cast<double>(props.totalGlobalMem) / (1ULL << 30),
                    props.multiProcessorCount);

            gpus_[i].device_id = i;
        }

        // Enable peer access for Infinity Fabric direct transfers
        for (int i = 0; i < NUM_GPUS; ++i) {
            for (int j = 0; j < NUM_GPUS; ++j) {
                if (i == j) continue;
                int can_access = 0;
                CUDA_CHECK(hipDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    CUDA_CHECK(hipSetDevice(i));
                    hipError_t peer_err = hipDeviceEnablePeerAccess(j, 0);
                    if (peer_err == hipErrorPeerAccessAlreadyEnabled) {
                        hipGetLastError(); // Clear the error
                    } else {
                        CUDA_CHECK(peer_err);
                    }
                    fprintf(stderr, "  Enabled peer access: GPU %d -> GPU %d\n", i, j);
                } else {
                    fprintf(stderr, "  WARNING: No peer access: GPU %d -> GPU %d "
                            "(will use staging buffers)\n", i, j);
                }
            }
        }
    }

    void init_nccl() {
        if (NUM_GPUS == 1) {
            // Single GPU: skip ALL RCCL init — no collectives needed.
            // NCCLManager is still created but comms stay null; all RCCL
            // collectives (AllReduce, Send/Recv) are no-ops with world_size=1.
            gpus_[0].nccl = nccl_manager_create();
            nccl_manager_set_rank(gpus_[0].nccl, 0, 1);
            nccl_manager_set_sub_comm_ranks(
                gpus_[0].nccl, ModelConfig::tp_size, ModelConfig::ep_size);
            fprintf(stderr, "[InferenceEngine] Single GPU — RCCL skipped.\n");
            return;
        }

        fprintf(stderr, "[InferenceEngine] Initializing RCCL communicators "
                "(TP%d x EP%d = %d GPUs)...\n",
                ModelConfig::tp_size, ModelConfig::ep_size, NUM_GPUS);

        ncclUniqueId nccl_id;
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));

        NCCL_CHECK(ncclGroupStart());
        for (int r = 0; r < NUM_GPUS; ++r) {
            CUDA_CHECK(hipSetDevice(r));
            gpus_[r].nccl = nccl_manager_create();
            nccl_manager_init_with_id(gpus_[r].nccl, r, NUM_GPUS, nccl_id);
        }
        NCCL_CHECK(ncclGroupEnd());

        for (int r = 0; r < NUM_GPUS; ++r) {
            nccl_manager_set_sub_comm_ranks(
                gpus_[r].nccl, ModelConfig::tp_size, ModelConfig::ep_size);
        }

        NCCL_CHECK(ncclGroupStart());
        for (int r = 0; r < NUM_GPUS; ++r) {
            CUDA_CHECK(hipSetDevice(r));
            nccl_manager_create_tp_comm(gpus_[r].nccl);
        }
        NCCL_CHECK(ncclGroupEnd());

        NCCL_CHECK(ncclGroupStart());
        for (int r = 0; r < NUM_GPUS; ++r) {
            CUDA_CHECK(hipSetDevice(r));
            nccl_manager_create_ep_comm(gpus_[r].nccl);
        }
        NCCL_CHECK(ncclGroupEnd());

        for (int r = 0; r < NUM_GPUS; ++r) {
            int tp_r = nccl_manager_tp_rank(gpus_[r].nccl);
            int ep_r = nccl_manager_ep_rank(gpus_[r].nccl);
            fprintf(stderr, "[InferenceEngine] Rank %d: (tp_rank=%d, ep_rank=%d)\n",
                    r, tp_r, ep_r);
        }

        fprintf(stderr, "[InferenceEngine] RCCL initialized.\n");
    }

    void init_gpu_resources(int rank) {
        auto& gpu = gpus_[rank];
        CUDA_CHECK(hipSetDevice(rank));

        // Compute stream (high priority for latency-sensitive inference)
        gpu.compute_stream = CudaStream(rank, /*high_priority=*/true);

        // Communication stream
        gpu.comm_stream = CudaStream(rank, /*high_priority=*/true);

        // hipBLAS handle bound to the compute stream
        gpu.cublas.init(rank, gpu.compute_stream);

        // Device-side logits buffers: BF16 (written by Transformer) and FP32 (for conversion)
        // Sized for max_batch_size sequences to support batched decode.
        constexpr size_t max_logits = static_cast<size_t>(ModelConfig::max_batch_size)
                                    * ModelConfig::vocab_size;
        CUDA_CHECK(hipMalloc(&gpu.device_logits_bf16,
                              max_logits * sizeof(__hip_bfloat16)));
        CUDA_CHECK(hipMalloc(&gpu.device_logits_fp32,
                              max_logits * sizeof(float)));

        // Pinned host buffer for logits transfer (only need GPU 0's)
        if (rank == 0) {
            CUDA_CHECK(hipHostMalloc(&gpu.host_logits,
                                      max_logits * sizeof(float)));
        }

        fprintf(stderr, "[InferenceEngine] GPU %d resources initialized.\n", rank);
    }

    void load_weights(const std::string& weights_dir) {
        fprintf(stderr, "[InferenceEngine] Loading weights from %s ...\n",
                weights_dir.c_str());

        for (int r = 0; r < NUM_GPUS; ++r) {
            CUDA_CHECK(hipSetDevice(r));
            int tp_rank = ModelConfig::tp_rank_of(r);
            int ep_rank = ModelConfig::ep_rank_of(r);
            gpus_[r].weights = weight_loader_create();
            weight_loader_init(gpus_[r].weights, weights_dir,
                               tp_rank, ep_rank,
                               ModelConfig::tp_size, ModelConfig::ep_size, r);
            weight_loader_load_all(gpus_[r].weights);
            fprintf(stderr, "[InferenceEngine] GPU %d (tp=%d, ep=%d) weights loaded.\n",
                    r, tp_rank, ep_rank);
        }

        fprintf(stderr, "[InferenceEngine] All weights loaded.\n");
    }

    void init_subsystems() {
        constexpr int NL = ModelConfig::num_layers;
        constexpr int MAX_TOKENS = 8192;
        constexpr int hidden = ModelConfig::hidden_size;
        constexpr int tp_qkv = ModelConfig::tp_qkv_size;
        constexpr int tp_q   = ModelConfig::tp_q_dim;
        constexpr int tp_kv  = ModelConfig::tp_kv_dim;
        constexpr int tp_attn_out = ModelConfig::tp_attn_out_size;
        constexpr int num_experts = ModelConfig::num_experts;
        constexpr int top_k = ModelConfig::num_active_experts;
        constexpr int epg = ModelConfig::experts_per_gpu;
        constexpr int expert_gate_up = ModelConfig::expert_gate_up_size;
        constexpr int expert_inter = ModelConfig::expert_intermediate_size;

        for (int r = 0; r < NUM_GPUS; ++r) {
            auto& gpu = gpus_[r];
            CUDA_CHECK(hipSetDevice(r));

            // ---- KV cache ----
            size_t free_mem = 0, total_mem = 0;
            CUDA_CHECK(hipMemGetInfo(&free_mem, &total_mem));

            // Reserve headroom for layer scratch, transformer buffers, hipBLAS workspace.
            // Attn scratch ~60MB + MoE scratch ~600MB + transformer ~1.5GB + hipBLAS ~256MB
            constexpr size_t KV_HEADROOM = 10ULL * 1024 * 1024 * 1024; // 10 GB

            constexpr int kv_max_blocks_per_seq =
                (ModelConfig::rope_max_pos + ModelConfig::kv_block_size - 1) / ModelConfig::kv_block_size;
            constexpr size_t bt_mem =
                256ULL * NL * kv_max_blocks_per_seq * sizeof(int32_t);

            size_t kv_available = (free_mem > KV_HEADROOM + bt_mem)
                                ? (free_mem - KV_HEADROOM - bt_mem) : 0;

            constexpr size_t kv_bytes_per_token =
                static_cast<size_t>(NL) * ModelConfig::tp_kv_heads *
                ModelConfig::head_dim * sizeof(__hip_bfloat16) * 2;

            int64_t kv_target_tokens = (kv_bytes_per_token > 0)
                ? static_cast<int64_t>(kv_available / kv_bytes_per_token)
                : 680000;
            kv_target_tokens = std::max(kv_target_tokens, static_cast<int64_t>(65536));

            fprintf(stderr, "[InferenceEngine] GPU %d: %.1f GB free, KV target %lld tokens "
                    "(%.1f GB)\n", r,
                    static_cast<double>(free_mem) / (1024.0 * 1024.0 * 1024.0),
                    (long long)kv_target_tokens,
                    static_cast<double>(kv_target_tokens * kv_bytes_per_token) /
                        (1024.0 * 1024.0 * 1024.0));

            gpu.kv_cache = kv_cache_manager_create();
            kv_cache_manager_init(gpu.kv_cache, 256, 131072, NL,
                                  ModelConfig::tp_kv_heads, ModelConfig::head_dim,
                                  ModelConfig::kv_block_size, r, kv_target_tokens);

            // Buffer manager for RCCL send/recv staging
            gpu.buffers = buffer_manager_create();
            buffer_manager_init(gpu.buffers, MAX_TOKENS, hidden, r);

            // ---- EPComm ----
            int tp_rank = nccl_manager_tp_rank(gpu.nccl);
            int ep_rank = nccl_manager_ep_rank(gpu.nccl);
            gpu.ep_comm = ep_comm_create();
            ncclComm_t ep_sub_comm = (NUM_GPUS > 1) ? nccl_manager_ep_comm(gpu.nccl) : nullptr;
            ep_comm_init(gpu.ep_comm, r, ep_rank, ModelConfig::ep_size,
                         tp_rank, ModelConfig::tp_size, ep_sub_comm);

            // ---- RoPE tables ----
            hipStream_t cs = (hipStream_t)gpu.compute_stream;
            rope_init(&gpu.rope_cos, &gpu.rope_sin, &gpu.rope_cs,
                      ModelConfig::rope_max_pos, ModelConfig::head_dim,
                      ModelConfig::rope_theta, cs);
            CUDA_CHECK(hipStreamSynchronize(cs));

            // ---- Attention scratch buffers (shared across layers) ----
            auto alloc_bf16 = [&](size_t count) -> __hip_bfloat16* {
                __hip_bfloat16* p = nullptr;
                CUDA_CHECK(hipMalloc(&p, count * sizeof(__hip_bfloat16)));
                return p;
            };

            gpu.attn_norm_buf  = alloc_bf16((size_t)MAX_TOKENS * hidden);
            gpu.attn_qkv_buf   = alloc_bf16((size_t)MAX_TOKENS * tp_qkv);
            gpu.attn_q_buf     = alloc_bf16((size_t)MAX_TOKENS * tp_q);
            gpu.attn_k_buf     = alloc_bf16((size_t)MAX_TOKENS * tp_kv);
            gpu.attn_v_buf     = alloc_bf16((size_t)MAX_TOKENS * tp_kv);
            gpu.attn_q_rope    = alloc_bf16((size_t)MAX_TOKENS * tp_q);
            gpu.attn_k_rope    = alloc_bf16((size_t)MAX_TOKENS * tp_kv);
            gpu.attn_out_buf   = alloc_bf16((size_t)MAX_TOKENS * tp_attn_out);
            gpu.attn_proj_out  = alloc_bf16((size_t)MAX_TOKENS * hidden);

            // ---- Attention layers ----
            __hip_bfloat16* k_cache = kv_cache_manager_k_cache(gpu.kv_cache);
            __hip_bfloat16* v_cache = kv_cache_manager_v_cache(gpu.kv_cache);
            int max_blocks = kv_cache_manager_num_blocks(gpu.kv_cache);

            for (int l = 0; l < NL; ++l) {
                gpu.attn_layers[l] = attention_layer_create();
                attention_layer_init(
                    gpu.attn_layers[l], l, r, tp_rank,
                    nccl_manager_tp_comm(gpu.nccl), &gpu.cublas,
                    weight_loader_attn_qkv(gpu.weights, l),
                    weight_loader_attn_qkv_bias(gpu.weights, l),
                    weight_loader_attn_o_proj(gpu.weights, l),
                    weight_loader_attn_o_proj_bias(gpu.weights, l),
                    weight_loader_attn_norm(gpu.weights, l),
                    gpu.rope_cos, gpu.rope_sin, gpu.rope_cs,
                    weight_loader_attn_sinks(gpu.weights, l),
                    k_cache, v_cache, max_blocks,
                    gpu.attn_norm_buf, gpu.attn_qkv_buf,
                    gpu.attn_q_buf, gpu.attn_k_buf, gpu.attn_v_buf,
                    gpu.attn_q_rope, gpu.attn_k_rope,
                    gpu.attn_out_buf, gpu.attn_proj_out);
            }

            fprintf(stderr, "[InferenceEngine] GPU %d: %d attention layers created.\n", r, NL);

            // ---- MoE scratch buffers ----
            size_t max_routed = (size_t)MAX_TOKENS * top_k;

            gpu.moe_norm_buf       = alloc_bf16((size_t)MAX_TOKENS * hidden);
            gpu.moe_router_logits  = alloc_bf16((size_t)MAX_TOKENS * num_experts);
            CUDA_CHECK(hipMalloc(&gpu.moe_expert_indices,  MAX_TOKENS * top_k * sizeof(int32_t)));
            CUDA_CHECK(hipMalloc(&gpu.moe_expert_weights,  MAX_TOKENS * top_k * sizeof(float)));
            CUDA_CHECK(hipMalloc(&gpu.moe_tokens_per_expert, num_experts * sizeof(int32_t)));
            gpu.moe_permuted_tokens = alloc_bf16(max_routed * hidden);
            CUDA_CHECK(hipMalloc(&gpu.moe_expert_offsets,  (epg + 1) * sizeof(int32_t)));
            CUDA_CHECK(hipMalloc(&gpu.moe_gather_map,      max_routed * sizeof(int32_t)));
            CUDA_CHECK(hipMalloc(&gpu.moe_per_peer_counts, ModelConfig::ep_size * sizeof(int32_t)));
            CUDA_CHECK(hipMalloc(&gpu.moe_peer_recv_offs,  ModelConfig::ep_size * sizeof(int32_t)));
            gpu.moe_recv_buffer    = alloc_bf16(max_routed * hidden);

            // Double-buffered dequant staging
            gpu.moe_dq_mlp1[0] = alloc_bf16((size_t)hidden * expert_gate_up);
            gpu.moe_dq_mlp1[1] = alloc_bf16((size_t)hidden * expert_gate_up);
            gpu.moe_dq_mlp2[0] = alloc_bf16((size_t)expert_inter * hidden);
            gpu.moe_dq_mlp2[1] = alloc_bf16((size_t)expert_inter * hidden);

            // Expert computation intermediates
            gpu.moe_gate_up_buf  = alloc_bf16(max_routed * expert_gate_up);
            gpu.moe_swiglu_buf   = alloc_bf16(max_routed * expert_inter);
            gpu.moe_expert_out   = alloc_bf16(max_routed * hidden);
            gpu.moe_output_buf   = alloc_bf16((size_t)MAX_TOKENS * hidden);

            // ---- MoE layers ----
            for (int l = 0; l < NL; ++l) {
                // Build MXFP4Weight descriptor arrays for this layer
                gpu.mxfp4_mlp1[l] = new MXFP4Weight[epg];
                gpu.mxfp4_mlp2[l] = new MXFP4Weight[epg];
                for (int e = 0; e < epg; ++e) {
                    gpu.mxfp4_mlp1[l][e] = {
                        weight_loader_expert_mlp1_packed(gpu.weights, l, e),
                        weight_loader_expert_mlp1_scales(gpu.weights, l, e),
                        weight_loader_expert_mlp1_numel(gpu.weights, l, e)
                    };
                    gpu.mxfp4_mlp2[l][e] = {
                        weight_loader_expert_mlp2_packed(gpu.weights, l, e),
                        weight_loader_expert_mlp2_scales(gpu.weights, l, e),
                        weight_loader_expert_mlp2_numel(gpu.weights, l, e)
                    };
                }

                gpu.moe_layers[l] = moe_layer_create();
                moe_layer_init(
                    gpu.moe_layers[l], l, r, ep_rank,
                    gpu.ep_comm, &gpu.cublas, &gpu.cublas,
                    (hipStream_t)gpu.compute_stream, (hipStream_t)gpu.comm_stream,
                    weight_loader_moe_norm(gpu.weights, l),
                    weight_loader_moe_router(gpu.weights, l),
                    weight_loader_moe_router_bias(gpu.weights, l),
                    weight_loader_expert_gate_up_bias(gpu.weights, l),
                    weight_loader_expert_down_bias(gpu.weights, l),
                    gpu.mxfp4_mlp1[l], gpu.mxfp4_mlp2[l],
                    gpu.moe_norm_buf, gpu.moe_router_logits,
                    gpu.moe_expert_indices, gpu.moe_expert_weights,
                    gpu.moe_tokens_per_expert, gpu.moe_permuted_tokens,
                    gpu.moe_expert_offsets, gpu.moe_gather_map,
                    gpu.moe_per_peer_counts, gpu.moe_peer_recv_offs,
                    gpu.moe_recv_buffer,
                    gpu.moe_dq_mlp1[0], gpu.moe_dq_mlp1[1],
                    gpu.moe_dq_mlp2[0], gpu.moe_dq_mlp2[1],
                    gpu.moe_gate_up_buf, gpu.moe_swiglu_buf,
                    gpu.moe_expert_out, gpu.moe_output_buf,
                    nccl_manager_ep_comm(gpu.nccl));
            }

            fprintf(stderr, "[InferenceEngine] GPU %d: %d MoE layers created.\n", r, NL);

            // ---- Transformer (wired with actual layers and weights) ----
            gpu.model = transformer_create();
            transformer_init(gpu.model,
                             r,  // device_id (not tp_rank -- with TP1xEP2, tp_rank=0 for both)
                             nccl_manager_tp_comm(gpu.nccl),
                             nccl_manager_ep_comm(gpu.nccl),
                             &gpu.cublas,
                             weight_loader_embedding(gpu.weights),
                             weight_loader_final_norm(gpu.weights),
                             weight_loader_lm_head(gpu.weights),
                             gpu.attn_layers,
                             gpu.moe_layers,
                             MAX_TOKENS);

            fprintf(stderr, "[InferenceEngine] GPU %d sub-systems initialized.\n", r);
        }
    }

    // -----------------------------------------------------------------------
    // L2 Cache Persistence for KV Cache (MI300X optimization)
    //
    // MI300X has L2 cache per die. We pin the most frequently accessed
    // sliding-window KV data in L2 to reduce L2 miss rate for attention
    // kernels.
    //
    // Target data:
    //   128 tokens (sliding window) * 36 layers * 8 KV heads * 64 head_dim
    //   * 2 (K+V) * 2 bytes (BF16) = ~9.4 MB
    //
    // We set hitRatio=1.0 and hitProp=Persisting so the hardware L2
    // eviction policy strongly favors keeping this data resident.
    // -----------------------------------------------------------------------
    void set_l2_persistence() {
        // L2 cache persistence (access policy window) is not supported on AMD GPUs.
        // This is a performance optimization only — not needed for correctness.
        fprintf(stderr, "[InferenceEngine] L2 persistence skipped (not available on AMD).\n");
    }

    // Helper: run transformer_prefill on a specific GPU in its own thread.
    // MoE layers contain host-side sync (CPU gather_map) + RCCL collectives,
    // so all GPUs must run concurrently -- a sequential for-loop deadlocks.
    void run_gpu_prefill(int r, const int32_t* tokens, const int* positions, int seq_len) {
        CUDA_CHECK(hipSetDevice(r));
        auto& gpu = gpus_[r];
        transformer_prefill(gpu.model, tokens, positions, seq_len, gpu.device_logits_bf16);
        hipStream_t stream = transformer_compute_stream(gpu.model);
        CUDA_CHECK(hipStreamSynchronize(stream));
    }

    // Prefill variant with explicit slot_mapping for batched KV cache
    void run_gpu_prefill_seq(int r, const int32_t* tokens, const int* positions,
                             const int32_t* slot_mapping, int seq_len) {
        CUDA_CHECK(hipSetDevice(r));
        auto& gpu = gpus_[r];
        transformer_prefill(gpu.model, tokens, positions, seq_len,
                            gpu.device_logits_bf16, slot_mapping);
        hipStream_t stream = transformer_compute_stream(gpu.model);
        CUDA_CHECK(hipStreamSynchronize(stream));
    }

    // Helper: run transformer_decode on a specific GPU in its own thread.
    void run_gpu_decode(int r, const int32_t* token_ids, const int* positions,
                        const int32_t* slot_mapping, const int32_t* block_table,
                        const int32_t* seq_lens, int num_seqs, int max_num_blocks) {
        CUDA_CHECK(hipSetDevice(r));
        auto& gpu = gpus_[r];
        transformer_decode(gpu.model, token_ids, positions, slot_mapping,
                           block_table, seq_lens, num_seqs, max_num_blocks,
                           gpu.device_logits_bf16);
        hipStream_t stream = transformer_compute_stream(gpu.model);
        CUDA_CHECK(hipStreamSynchronize(stream));
    }

    void warmup() {
        fprintf(stderr, "[InferenceEngine] Warming up...\n");

        // Short dummy prompt
        std::vector<int32_t> dummy_tokens = {1, 2, 3, 4};
        const int dummy_len = static_cast<int>(dummy_tokens.size());

        // Build positions array
        std::vector<int> positions(dummy_len);
        for (int i = 0; i < dummy_len; ++i) positions[i] = i;

        // Run prefill on ALL GPUs concurrently (required: MoE layers have
        // RCCL collectives + host-side sync, so sequential launch deadlocks).
        fprintf(stderr, "[InferenceEngine] Spawning %d warmup threads...\n", NUM_GPUS);
        fflush(stderr);
        std::vector<std::thread> threads;
        for (int r = 0; r < NUM_GPUS; ++r) {
            threads.emplace_back(&InferenceEngine::run_gpu_prefill, this,
                                 r, dummy_tokens.data(), positions.data(), dummy_len);
        }
        fprintf(stderr, "[InferenceEngine] Waiting for warmup threads to complete...\n");
        fflush(stderr);
        for (auto& t : threads) t.join();
        fprintf(stderr, "[InferenceEngine] Warmup threads joined.\n");
        fflush(stderr);

        // Free KV cache entries from warmup
        for (int r = 0; r < NUM_GPUS; ++r) {
            kv_cache_manager_free_sequence(gpus_[r].kv_cache, /*seq_id=*/0);
        }

        fprintf(stderr, "[InferenceEngine] Warm-up complete.\n");
    }

    // ==================================================================
    // Inference methods
    // ==================================================================

    // Run prefill for a specific sequence in the batch.
    // seq_id determines where in the KV cache this sequence's data goes.
    void run_prefill_seq(const std::vector<int32_t>& prompt_tokens, int seq_id) {
        const int seq_len = static_cast<int>(prompt_tokens.size());

        // Build positions array: [0, 1, 2, ..., seq_len-1]
        std::vector<int> positions(seq_len);
        for (int i = 0; i < seq_len; ++i) positions[i] = i;

        // Build slot_mapping: each token maps to a unique KV cache slot.
        // Sequence seq_id uses physical slots starting at
        // seq_id * blocks_per_seq * kv_block_size.
        const int slot_base = seq_id * decode_state_.blocks_per_seq
                            * ModelConfig::kv_block_size;
        std::vector<int32_t> slot_mapping(seq_len);
        for (int i = 0; i < seq_len; ++i) {
            slot_mapping[i] = slot_base + i;
        }

        // Launch on ALL GPUs concurrently
        std::vector<std::thread> threads;
        for (int r = 0; r < NUM_GPUS; ++r) {
            threads.emplace_back(&InferenceEngine::run_gpu_prefill_seq, this,
                                 r, prompt_tokens.data(), positions.data(),
                                 slot_mapping.data(), seq_len);
        }
        for (auto& t : threads) t.join();

        // Convert BF16 logits to FP32 on GPU 0 (single sequence logits)
        // Caller is responsible for calling convert_and_copy_logits().
    }

    // Run prefill on both GPUs with the full prompt (legacy B=1 path).
    // Writes FP32 logits to gpus_[0].host_logits for the last token position.
    void run_prefill(const std::vector<int32_t>& prompt_tokens) {
        const int seq_len = static_cast<int>(prompt_tokens.size());

        // Build positions array on host: [0, 1, 2, ..., seq_len-1]
        std::vector<int> positions(seq_len);
        for (int i = 0; i < seq_len; ++i) positions[i] = i;

        // Launch prefill on ALL GPUs concurrently (MoE RCCL collectives
        // require all GPUs to participate; sequential loop deadlocks).
        std::vector<std::thread> threads;
        for (int r = 0; r < NUM_GPUS; ++r) {
            threads.emplace_back(&InferenceEngine::run_gpu_prefill, this,
                                 r, prompt_tokens.data(), positions.data(), seq_len);
        }
        for (auto& t : threads) t.join();

        // Convert BF16 logits to FP32 on GPU 0 and copy to host
        convert_and_copy_logits();
    }

    // Run a single decode step on both GPUs via direct kernel launch.
    // Uses pre-allocated decode_state_ buffers to avoid per-step heap
    // allocations. The block_table is pre-filled at init; we just
    // compute the prefix length needed for the current position.
    void run_decode_step(int32_t token_id, int position) {
        // Write into pre-allocated arrays (no heap allocation)
        decode_state_.token_ids[0]    = token_id;
        decode_state_.positions[0]    = position;
        decode_state_.slot_mapping[0] = position;  // simplified: slot = position
        decode_state_.seq_lens[0]     = position + 1;

        // Block table is pre-filled with identity mapping; just compute length
        int max_num_blocks = (position + 1 + ModelConfig::kv_block_size - 1)
                           / ModelConfig::kv_block_size;

        // Launch decode on ALL GPUs concurrently (MoE RCCL collectives
        // require all GPUs to participate; sequential loop deadlocks).
        std::vector<std::thread> threads;
        for (int r = 0; r < NUM_GPUS; ++r) {
            threads.emplace_back(&InferenceEngine::run_gpu_decode, this,
                                 r, decode_state_.token_ids, decode_state_.positions,
                                 decode_state_.slot_mapping, decode_state_.block_table.data(),
                                 decode_state_.seq_lens, 1, max_num_blocks);
        }
        for (auto& t : threads) t.join();

        // Convert BF16 logits to FP32 on GPU 0 and copy to host
        convert_and_copy_logits();
    }

    // Convert BF16 device logits to FP32, then D2H copy to host_logits on GPU 0.
    // num_seqs: number of sequences (1 for legacy single-sequence, B for batch)
    void convert_and_copy_logits(int num_seqs = 1) {
        auto& gpu = gpus_[0];
        CUDA_CHECK(hipSetDevice(0));

        hipStream_t stream = transformer_compute_stream(gpu.model);

        const int total_elements = num_seqs * ModelConfig::vocab_size;

        // BF16 -> FP32 conversion on device
        bf16_to_fp32(gpu.device_logits_bf16, gpu.device_logits_fp32,
                     total_elements, stream);

        // Copy FP32 logits to pinned host memory
        CUDA_CHECK(hipMemcpyAsync(gpu.host_logits,
                                   gpu.device_logits_fp32,
                                   total_elements * sizeof(float),
                                   hipMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(hipStreamSynchronize(stream));
    }

    // ==================================================================
    // Sampling
    // ==================================================================

    // Top-p (nucleus) sampling with temperature scaling.
    int32_t sample(const float* logits_ptr, float temperature, float top_p) {
        const int V = ModelConfig::vocab_size;

        // Greedy decoding
        if (temperature <= 1e-6f) {
            int32_t best = 0;
            float best_val = logits_ptr[0];
            for (int i = 1; i < V; ++i) {
                if (logits_ptr[i] > best_val) {
                    best_val = logits_ptr[i];
                    best = static_cast<int32_t>(i);
                }
            }
            return best;
        }

        // ---- Temperature-scaled softmax ----
        float max_logit = logits_ptr[0];
        for (int i = 1; i < V; ++i) {
            max_logit = std::max(max_logit, logits_ptr[i]);
        }

        std::vector<float> probs(V);
        double sum = 0.0;
        const float inv_temp = 1.0f / temperature;
        for (int i = 0; i < V; ++i) {
            float val = std::exp((logits_ptr[i] - max_logit) * inv_temp);
            probs[i] = val;
            sum += static_cast<double>(val);
        }

        const float inv_sum = static_cast<float>(1.0 / sum);
        for (int i = 0; i < V; ++i) {
            probs[i] *= inv_sum;
        }

        // ---- Top-p filtering ----
        std::vector<int32_t> indices(V);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(),
                          indices.begin() + std::min(V, 1024),
                          indices.end(),
                          [&probs](int32_t a, int32_t b) {
                              return probs[a] > probs[b];
                          });

        float cumulative = 0.0f;
        int nucleus_size = 0;
        for (int i = 0; i < V; ++i) {
            cumulative += probs[indices[i]];
            nucleus_size = i + 1;
            if (cumulative >= top_p) break;
        }

        float nucleus_sum = 0.0f;
        for (int i = 0; i < nucleus_size; ++i) {
            nucleus_sum += probs[indices[i]];
        }

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float u = dist(rng_);
        float acc = 0.0f;
        for (int i = 0; i < nucleus_size; ++i) {
            acc += probs[indices[i]] / nucleus_sum;
            if (u <= acc) {
                return indices[i];
            }
        }

        // Fallback: return the highest-probability token
        return indices[0];
    }

    // ==================================================================
    // State
    // ==================================================================

    GPUResources gpus_[NUM_GPUS];
    bool initialized_ = false;

    // Random number generator for sampling (seeded for reproducibility)
    std::mt19937 rng_{42};

    // ------------------------------------------------------------------
    // Pre-allocated decode step buffers
    //
    // Eliminates per-step heap allocations (std::vector) that cause
    // ~1-5us overhead per decode step from malloc/free + cache thrashing.
    // The block_table grows monotonically during generation and is
    // pre-allocated to the maximum possible size.
    // ------------------------------------------------------------------
    struct DecodeState {
        static constexpr int MAX_BATCH = ModelConfig::max_batch_size; // 256

        // blocks_per_seq: set dynamically based on actual KV cache capacity.
        // The KV cache is [num_physical_blocks, ...] where physical blocks
        // are split across layers. Each layer sees blocks_per_layer blocks.
        // We divide that evenly among MAX_BATCH sequences.
        int blocks_per_seq = 0;  // set by init(blocks_per_layer)

        // B=1 legacy arrays (kept for backward compat)
        int32_t token_ids[MAX_BATCH]     = {};
        int32_t positions[MAX_BATCH]     = {};
        int32_t slot_mapping[MAX_BATCH]  = {};
        int32_t seq_lens[MAX_BATCH]      = {};

        // Block table: [MAX_BATCH][blocks_per_seq]
        // For batched decode, each sequence gets its own row of block IDs.
        // Laid out contiguously: seq 0 blocks, seq 1 blocks, ...
        std::vector<int32_t> block_table;

        // Packed block table for decode: [num_seqs][max_num_blocks]
        // (subset of block_table, packed for the transformer)
        std::vector<int32_t> packed_block_table;

        int     last_num_blocks  = 0;     // track growth to avoid redundant init

        int blocks_per_layer_ = 0;  // total blocks available per layer

        void init(int blocks_per_layer) {
            blocks_per_layer_ = blocks_per_layer;
            // Default: B=1, full capacity for single sequence
            blocks_per_seq = blocks_per_layer;
            setup_block_table(1);
        }

        // Reconfigure block table for a specific batch size.
        // Divides the per-layer KV cache evenly among B sequences.
        void setup_block_table(int B) {
            blocks_per_seq = blocks_per_layer_ / std::max(B, 1);
            if (blocks_per_seq < 1) blocks_per_seq = 1;

            int max_seq_tokens = blocks_per_seq * ModelConfig::kv_block_size;
            fprintf(stderr, "[DecodeState] B=%d, blocks_per_layer=%d, blocks_per_seq=%d, "
                    "max_seq_tokens=%d\n", B, blocks_per_layer_, blocks_per_seq, max_seq_tokens);

            // Pre-fill block table with identity mapping per sequence.
            // Sequence i uses blocks [i*blocks_per_seq, (i+1)*blocks_per_seq).
            block_table.resize(static_cast<size_t>(B) * blocks_per_seq);
            for (int s = 0; s < B; ++s) {
                int base = s * blocks_per_seq;
                for (int b = 0; b < blocks_per_seq; ++b) {
                    block_table[base + b] = s * blocks_per_seq + b;
                }
            }
            last_num_blocks = 0;
        }
    } decode_state_;
};

// ---------------------------------------------------------------------------
// Extern wrapper functions for bridge.cu
// ---------------------------------------------------------------------------

InferenceEngine* inference_engine_create() {
    return new InferenceEngine();
}

void inference_engine_destroy(InferenceEngine* eng) {
    if (eng) delete eng;
}

void inference_engine_init(InferenceEngine* eng, const std::string& weights_dir) {
    eng->init(weights_dir);
}

void inference_engine_shutdown(InferenceEngine* eng) {
    eng->shutdown();
}

bool inference_engine_is_initialized(InferenceEngine* eng) {
    return eng->is_initialized();
}

std::vector<int32_t> inference_engine_generate(
    InferenceEngine* eng, const std::vector<int32_t>& prompt,
    int max_tokens, float temperature, float top_p) {
    return eng->generate(prompt, max_tokens, temperature, top_p);
}

std::vector<std::vector<int32_t>> inference_engine_generate_batch(
    InferenceEngine* eng, const std::vector<std::vector<int32_t>>& prompts,
    int max_tokens, float temperature, float top_p) {
    return eng->generate_batch(prompts, max_tokens, temperature, top_p);
}

} // namespace gptoss
