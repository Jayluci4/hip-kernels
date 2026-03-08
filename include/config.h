#pragma once
// v4.9: multi-threaded warmup/prefill/decode for EP2 NCCL collectives
// Identical to CUDA version — no GPU-specific code

#include <cstdint>

namespace gptoss {

struct ModelConfig {
    // Architecture
    static constexpr int num_layers = 36;
    static constexpr int hidden_size = 2880;
    static constexpr int num_q_heads = 64;
    static constexpr int num_kv_heads = 8;
    static constexpr int head_dim = 64;
    static constexpr int qkv_size = (num_q_heads + 2 * num_kv_heads) * head_dim; // 5120
    static constexpr int attn_out_size = num_q_heads * head_dim; // 4096

    // MoE
    static constexpr int num_experts = 128;
    static constexpr int num_active_experts = 4;
    static constexpr int expert_intermediate_size = 2880;
    static constexpr int expert_gate_up_size = expert_intermediate_size * 2; // 5760 (SwiGLU)

    // Vocabulary
    static constexpr int vocab_size = 201088;

    // RoPE
    static constexpr float rope_theta = 150000.0f;
    static constexpr int rope_yarn_factor = 32;
    static constexpr int rope_original_max_pos = 4096;
    static constexpr int rope_max_pos = 131072;
    static constexpr float partial_rotary_factor = 1.0f;
    static constexpr float rope_yarn_beta_fast = 32.0f;
    static constexpr float rope_yarn_beta_slow = 1.0f;
    static constexpr float rope_yarn_mscale = 1.3466f;

    // Normalization
    static constexpr float rms_norm_eps = 1e-5f;

    // GPT-OSS activation
    static constexpr float swiglu_limit = 7.0f;
    static constexpr float activation_alpha = 1.702f;

    // Attention
    static constexpr int sliding_window_size = 128;
    static constexpr int full_attention_interval = 2;

    // GQA ratio
    static constexpr int gqa_ratio = num_q_heads / num_kv_heads; // 8

    // MXFP4
    static constexpr int mxfp4_block_size = 32;

    // Paged KV cache
    static constexpr int kv_block_size = 16;
    static constexpr int max_batch_size = 256;

    // Parallelism
    static constexpr int tp_size = 1;
    static constexpr int ep_size = 1;  // single GPU for correctness debugging
    static constexpr int world_size = tp_size * ep_size;
    static constexpr int experts_per_gpu = num_experts / ep_size;
    static constexpr int max_ep_size = 8;

    static constexpr int tp_rank_of(int global_rank) { return global_rank % tp_size; }
    static constexpr int ep_rank_of(int global_rank) { return global_rank / tp_size; }
    static constexpr int global_rank_of(int tp_r, int ep_r) { return tp_r + ep_r * tp_size; }

    static constexpr int tp_q_heads = num_q_heads / tp_size;
    static constexpr int tp_kv_heads = num_kv_heads / tp_size;
    static constexpr int tp_q_dim = tp_q_heads * head_dim;
    static constexpr int tp_kv_dim = tp_kv_heads * head_dim;
    static constexpr int tp_qkv_size = tp_q_dim + 2 * tp_kv_dim;
    static constexpr int tp_attn_out_size = tp_q_dim;

    static constexpr int tp_vocab_size = vocab_size / tp_size;

    static constexpr bool is_full_attention(int layer) {
        return (layer % full_attention_interval) == (full_attention_interval - 1);
    }

    static constexpr int get_window_size(int layer) {
        return is_full_attention(layer) ? 0 : sliding_window_size;
    }
};

} // namespace gptoss
