// Weight Loader for GPT-OSS-120B inference engine
// Loads safetensors model shards, parses headers, and partitions weights
// across GPUs for decoupled TP x EP parallelism.
//
// Weight partitioning strategy:
//   - Shared weights (replicated): embeddings, norms, router weights
//   - Attention weights (TP-sharded by tp_rank): QKV, O-proj, LM head
//   - Expert weights (EP-sharded by ep_rank): each EP rank loads its experts
//   - Expert weights remain in packed MXFP4 format (0.5 bytes/param)
//   - Shared/attention weights loaded as BF16
//
// Safetensors format:
//   - First 8 bytes: uint64 LE header_size
//   - Next header_size bytes: JSON metadata mapping tensor names to
//     { dtype, shape, data_offsets: [begin, end] }
//   - Remaining bytes: raw tensor data (offsets relative to end of header)

#include "hip_compat.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "config.h"
#include "tensor.h"
#include "cuda_utils.h"

namespace gptoss {

// ============================================================================
// Safetensors header parsing utilities
// ============================================================================

// Metadata for a single tensor within a safetensors file
struct SafetensorEntry {
    std::string name;
    std::string dtype;           // "BF16", "F32", "U8", etc.
    std::vector<int64_t> shape;
    int64_t data_offset_begin = 0; // relative to data start (after header)
    int64_t data_offset_end   = 0;

    int64_t nbytes() const { return data_offset_end - data_offset_begin; }
};

// Represents one safetensors file (shard)
struct SafetensorShard {
    std::string filepath;
    int fd = -1;
    void* mmap_ptr = nullptr;
    size_t file_size = 0;
    uint64_t header_size = 0;
    const uint8_t* data_start = nullptr; // points past the header
    std::vector<SafetensorEntry> entries;
};

// ============================================================================
// Minimal JSON parser for safetensors headers
//
// Safetensors headers are JSON objects mapping tensor names to metadata.
// We parse just enough to extract name, dtype, shape, data_offsets.
// The format also includes a top-level "__metadata__" key which we skip.
// ============================================================================

static void skip_whitespace(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
}

static std::string parse_json_string(const char*& p, const char* end) {
    assert(*p == '"');
    ++p;
    std::string result;
    result.reserve(128);
    while (p < end && *p != '"') {
        if (*p == '\\') {
            ++p;
            if (p >= end) break;
            switch (*p) {
                case '"':  result += '"'; break;
                case '\\': result += '\\'; break;
                case '/':  result += '/'; break;
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case 'r':  result += '\r'; break;
                case 'u':  // Skip unicode escapes -- not expected in tensor names
                    result += "\\u";
                    break;
                default:   result += *p; break;
            }
        } else {
            result += *p;
        }
        ++p;
    }
    if (p < end) ++p; // skip closing quote
    return result;
}

static int64_t parse_json_int(const char*& p, const char* end) {
    bool neg = false;
    if (*p == '-') { neg = true; ++p; }
    int64_t val = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        ++p;
    }
    return neg ? -val : val;
}

// Skip an arbitrary JSON value (string, number, object, array, bool, null)
static void skip_json_value(const char*& p, const char* end) {
    skip_whitespace(p, end);
    if (p >= end) return;
    if (*p == '"') {
        parse_json_string(p, end);
    } else if (*p == '{') {
        ++p;
        int depth = 1;
        while (p < end && depth > 0) {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            else if (*p == '"') { parse_json_string(p, end); continue; }
            ++p;
        }
    } else if (*p == '[') {
        ++p;
        int depth = 1;
        while (p < end && depth > 0) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            else if (*p == '"') { parse_json_string(p, end); continue; }
            ++p;
        }
    } else {
        // number, bool, null
        while (p < end && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t') ++p;
    }
}

// Parse the metadata object for one tensor: { "dtype": "...", "shape": [...], "data_offsets": [...] }
static SafetensorEntry parse_tensor_metadata(const std::string& name,
                                              const char*& p, const char* end)
{
    SafetensorEntry entry;
    entry.name = name;

    assert(*p == '{');
    ++p;

    while (p < end && *p != '}') {
        skip_whitespace(p, end);
        if (*p == '}') break;

        std::string key = parse_json_string(p, end);
        skip_whitespace(p, end);
        assert(*p == ':'); ++p;
        skip_whitespace(p, end);

        if (key == "dtype") {
            entry.dtype = parse_json_string(p, end);
        } else if (key == "shape") {
            assert(*p == '['); ++p;
            while (p < end && *p != ']') {
                skip_whitespace(p, end);
                if (*p == ']') break;
                entry.shape.push_back(parse_json_int(p, end));
                skip_whitespace(p, end);
                if (*p == ',') ++p;
            }
            if (p < end) ++p; // skip ']'
        } else if (key == "data_offsets") {
            assert(*p == '['); ++p;
            skip_whitespace(p, end);
            entry.data_offset_begin = parse_json_int(p, end);
            skip_whitespace(p, end);
            if (*p == ',') ++p;
            skip_whitespace(p, end);
            entry.data_offset_end = parse_json_int(p, end);
            skip_whitespace(p, end);
            if (p < end && *p == ']') ++p;
        } else {
            skip_json_value(p, end);
        }

        skip_whitespace(p, end);
        if (*p == ',') ++p;
    }
    if (p < end) ++p; // skip '}'
    return entry;
}

// Parse the full safetensors JSON header
static std::vector<SafetensorEntry> parse_safetensors_header(const char* json, size_t json_len)
{
    std::vector<SafetensorEntry> entries;
    const char* p = json;
    const char* end = json + json_len;

    skip_whitespace(p, end);
    if (p >= end || *p != '{') {
        fprintf(stderr, "[WeightLoader] Error: invalid safetensors header\n");
        return entries;
    }
    ++p; // skip '{'

    while (p < end && *p != '}') {
        skip_whitespace(p, end);
        if (*p == '}') break;

        std::string key = parse_json_string(p, end);
        skip_whitespace(p, end);
        assert(*p == ':'); ++p;
        skip_whitespace(p, end);

        if (key == "__metadata__") {
            // Skip the metadata object entirely
            skip_json_value(p, end);
        } else {
            // This is a tensor entry
            entries.push_back(parse_tensor_metadata(key, p, end));
        }

        skip_whitespace(p, end);
        if (*p == ',') ++p;
    }

    return entries;
}

// ============================================================================
// Structured weight containers returned by accessors
// ============================================================================

struct AttentionWeights {
    Tensor qkv_proj;     // [qkv_size, hidden_size] BF16 -- fused Q/K/V projection
    Tensor o_proj;       // [hidden_size, attn_out_size] BF16
    Tensor norm;         // [hidden_size] BF16 -- pre-attention RMSNorm
    Tensor qkv_bias;     // [qkv_size] BF16 -- fused Q/K/V bias (may be empty)
    Tensor o_proj_bias;  // [hidden_size] BF16 -- output projection bias (may be empty)
    Tensor sinks;        // [num_q_heads] BF16 -- learned attention sink biases (may be empty)
};

struct MoEWeights {
    Tensor router;                                              // [hidden_size, num_experts] BF16
    Tensor norm;                                                // [hidden_size] BF16 -- pre-MoE RMSNorm
    Tensor router_bias;                                         // [num_experts] BF16 -- router bias (may be empty)
    Tensor expert_mlp1[ModelConfig::experts_per_gpu];           // [expert_gate_up_size, hidden_size/2] UINT8 (MXFP4)
    Tensor expert_mlp1_scales[ModelConfig::experts_per_gpu];    // scale tensors for MXFP4 blocks
    Tensor expert_mlp2[ModelConfig::experts_per_gpu];           // [hidden_size, expert_intermediate_size/2] UINT8 (MXFP4)
    Tensor expert_mlp2_scales[ModelConfig::experts_per_gpu];    // scale tensors for MXFP4 blocks
    Tensor gate_up_proj_bias;                                   // [num_experts, expert_gate_up_size] BF16 (EP-sliced)
    Tensor down_proj_bias;                                      // [num_experts, hidden_size] BF16 (EP-sliced)
};


// ============================================================================
// DType conversion utilities
// ============================================================================

static DType safetensors_dtype_to_dtype(const std::string& st_dtype) {
    if (st_dtype == "BF16" || st_dtype == "bf16")  return DType::BF16;
    if (st_dtype == "F32"  || st_dtype == "f32")   return DType::FP32;
    if (st_dtype == "F16"  || st_dtype == "f16")   return DType::FP16;
    if (st_dtype == "I8"   || st_dtype == "i8")    return DType::INT8;
    if (st_dtype == "U8"   || st_dtype == "u8")    return DType::UINT8;
    if (st_dtype == "I32"  || st_dtype == "i32")   return DType::INT32;
    fprintf(stderr, "[WeightLoader] Warning: unknown safetensors dtype '%s', defaulting to UINT8\n",
            st_dtype.c_str());
    return DType::UINT8;
}


// ============================================================================
// WeightLoader
// ============================================================================

class WeightLoader {
public:
    WeightLoader() = default;
    ~WeightLoader() { destroy(); }

    // Non-copyable
    WeightLoader(const WeightLoader&) = delete;
    WeightLoader& operator=(const WeightLoader&) = delete;

    // -----------------------------------------------------------------------
    // init: scan the weights directory for safetensors shards
    //
    // Decoupled TP x EP: tp_rank selects attention/LM-head shard,
    // ep_rank selects expert partition.
    // -----------------------------------------------------------------------
    void init(const std::string& weights_dir,
              int tp_rank, int ep_rank, int tp_size, int ep_size,
              int device_id)
    {
        weights_dir_ = weights_dir;
        tp_rank_     = tp_rank;
        ep_rank_     = ep_rank;
        tp_size_     = tp_size;
        ep_size_     = ep_size;
        rank_        = tp_rank;   // TP rank for attention weight sharding
        world_size_  = tp_size;   // TP size for attention weight sharding
        device_id_   = device_id;

        // Expert range determined by EP rank
        expert_start_ = ep_rank * ModelConfig::experts_per_gpu;
        expert_end_   = expert_start_ + ModelConfig::experts_per_gpu;

        CUDA_CHECK(hipSetDevice(device_id_));

        // Enumerate .safetensors files in directory
        scan_shard_files();
        if (shards_.empty()) {
            fprintf(stderr, "[WeightLoader] Error: no .safetensors files found in %s\n",
                    weights_dir.c_str());
            abort();
        }

        // Open and parse headers for all shards
        for (auto& shard : shards_) {
            open_and_parse_shard(shard);
        }

        // Build name -> (shard_index, entry_index) mapping
        build_tensor_index();

        initialized_ = true;
        fprintf(stderr, "[WeightLoader] GPU %d (tp=%d, ep=%d): found %zu shards, %zu tensors, "
                "experts [%d, %d)\n",
                device_id_, tp_rank_, ep_rank_, shards_.size(), tensor_index_.size(),
                expert_start_, expert_end_);
    }

    // -----------------------------------------------------------------------
    // load_all_weights: load every tensor we need onto the GPU
    // -----------------------------------------------------------------------
    void load_all_weights()
    {
        assert(initialized_);
        CUDA_CHECK(hipSetDevice(device_id_));

        fprintf(stderr, "[WeightLoader] GPU %d: loading all weights...\n", device_id_);

        // 1. Embedding
        embedding_ = load_tensor_to_gpu("model.embed_tokens.weight", DType::BF16);

        // 2. Final norm
        final_norm_ = load_tensor_to_gpu("model.norm.weight", DType::BF16);

        // 3. LM head -- TP-sharded: each GPU loads its vocab partition only.
        // lm_head.weight shape: [vocab_size, hidden_size] = [201088, 2880]
        // GPU r loads rows [r*tp_vocab_size : (r+1)*tp_vocab_size]
        {
            int64_t tp_vocab = ModelConfig::tp_vocab_size;  // 100544
            int64_t row_start = static_cast<int64_t>(rank_) * tp_vocab;
            lm_head_ = load_tensor_row_shard_to_gpu("lm_head.weight", DType::BF16,
                                                     row_start, tp_vocab);
        }

        // 4. Per-layer weights
        attn_weights_.resize(ModelConfig::num_layers);
        moe_weights_.resize(ModelConfig::num_layers);

        for (int layer = 0; layer < ModelConfig::num_layers; ++layer) {
            load_layer_weights(layer);
        }

        fprintf(stderr, "[WeightLoader] GPU %d: all weights loaded.\n", device_id_);
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    Tensor get_embedding_weight() const { return embedding_; }
    Tensor get_final_norm() const { return final_norm_; }
    Tensor get_lm_head() const { return lm_head_; }

    AttentionWeights get_attention_weights(int layer) const {
        assert(layer >= 0 && layer < ModelConfig::num_layers);
        return attn_weights_[layer];
    }

    MoEWeights get_moe_weights(int layer) const {
        assert(layer >= 0 && layer < ModelConfig::num_layers);
        return moe_weights_[layer];
    }

    // -----------------------------------------------------------------------
    // destroy: free all GPU tensors and unmap shard files
    // -----------------------------------------------------------------------
    void destroy()
    {
        if (!initialized_) return;

        CUDA_CHECK(hipSetDevice(device_id_));

        free_tensor(embedding_);
        free_tensor(final_norm_);
        free_tensor(lm_head_);

        for (int layer = 0; layer < (int)attn_weights_.size(); ++layer) {
            free_tensor(attn_weights_[layer].qkv_proj);
            free_tensor(attn_weights_[layer].o_proj);
            free_tensor(attn_weights_[layer].norm);
            free_tensor(attn_weights_[layer].qkv_bias);
            free_tensor(attn_weights_[layer].o_proj_bias);
            free_tensor(attn_weights_[layer].sinks);
        }

        for (int layer = 0; layer < (int)moe_weights_.size(); ++layer) {
            free_tensor(moe_weights_[layer].router);
            free_tensor(moe_weights_[layer].norm);
            free_tensor(moe_weights_[layer].router_bias);
            free_tensor(moe_weights_[layer].gate_up_proj_bias);
            free_tensor(moe_weights_[layer].down_proj_bias);
            for (int e = 0; e < ModelConfig::experts_per_gpu; ++e) {
                free_tensor(moe_weights_[layer].expert_mlp1[e]);
                free_tensor(moe_weights_[layer].expert_mlp1_scales[e]);
                free_tensor(moe_weights_[layer].expert_mlp2[e]);
                free_tensor(moe_weights_[layer].expert_mlp2_scales[e]);
            }
        }

        attn_weights_.clear();
        moe_weights_.clear();

        // Unmap all shards
        for (auto& shard : shards_) {
            if (shard.mmap_ptr && shard.mmap_ptr != MAP_FAILED) {
                munmap(shard.mmap_ptr, shard.file_size);
                shard.mmap_ptr = nullptr;
            }
            if (shard.fd >= 0) {
                close(shard.fd);
                shard.fd = -1;
            }
        }
        shards_.clear();
        tensor_index_.clear();

        initialized_ = false;
    }

private:
    // -----------------------------------------------------------------------
    // Shard file discovery
    // -----------------------------------------------------------------------
    void scan_shard_files()
    {
        DIR* dir = opendir(weights_dir_.c_str());
        if (!dir) {
            fprintf(stderr, "[WeightLoader] Error: cannot open directory %s\n",
                    weights_dir_.c_str());
            abort();
        }

        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string name(entry->d_name);
            if (name.size() > 12 && name.substr(name.size() - 12) == ".safetensors") {
                SafetensorShard shard;
                shard.filepath = weights_dir_ + "/" + name;
                shards_.push_back(std::move(shard));
            }
        }
        closedir(dir);

        // Sort for deterministic ordering
        std::sort(shards_.begin(), shards_.end(),
                  [](const SafetensorShard& a, const SafetensorShard& b) {
                      return a.filepath < b.filepath;
                  });
    }

    // -----------------------------------------------------------------------
    // Open a shard file, mmap it, and parse its header
    // -----------------------------------------------------------------------
    void open_and_parse_shard(SafetensorShard& shard)
    {
        shard.fd = open(shard.filepath.c_str(), O_RDONLY);
        if (shard.fd < 0) {
            fprintf(stderr, "[WeightLoader] Error: cannot open %s\n", shard.filepath.c_str());
            abort();
        }

        // Get file size
        struct stat st;
        if (fstat(shard.fd, &st) < 0) {
            fprintf(stderr, "[WeightLoader] Error: cannot stat %s\n", shard.filepath.c_str());
            abort();
        }
        shard.file_size = static_cast<size_t>(st.st_size);

        // mmap the entire file
        shard.mmap_ptr = mmap(nullptr, shard.file_size, PROT_READ, MAP_PRIVATE, shard.fd, 0);
        if (shard.mmap_ptr == MAP_FAILED) {
            fprintf(stderr, "[WeightLoader] Error: mmap failed for %s\n", shard.filepath.c_str());
            abort();
        }

        // Advise kernel for sequential access
        madvise(shard.mmap_ptr, shard.file_size, MADV_SEQUENTIAL);

        const uint8_t* base = static_cast<const uint8_t*>(shard.mmap_ptr);

        // Read header size (first 8 bytes, little-endian uint64)
        if (shard.file_size < 8) {
            fprintf(stderr, "[WeightLoader] Error: file too small: %s\n", shard.filepath.c_str());
            abort();
        }
        memcpy(&shard.header_size, base, sizeof(uint64_t));

        if (8 + shard.header_size > shard.file_size) {
            fprintf(stderr, "[WeightLoader] Error: header size (%llu) exceeds file size in %s\n",
                    (unsigned long long)shard.header_size, shard.filepath.c_str());
            abort();
        }

        // Parse JSON header
        const char* json_start = reinterpret_cast<const char*>(base + 8);
        shard.entries = parse_safetensors_header(json_start, shard.header_size);
        shard.data_start = base + 8 + shard.header_size;

        fprintf(stderr, "[WeightLoader] Parsed %s: %zu tensors, header=%llu bytes\n",
                shard.filepath.c_str(), shard.entries.size(),
                (unsigned long long)shard.header_size);
    }

    // -----------------------------------------------------------------------
    // Build name->(shard,entry) index
    // -----------------------------------------------------------------------
    struct TensorLocation {
        int shard_idx;
        int entry_idx;
    };

    void build_tensor_index()
    {
        for (int si = 0; si < (int)shards_.size(); ++si) {
            for (int ei = 0; ei < (int)shards_[si].entries.size(); ++ei) {
                const std::string& name = shards_[si].entries[ei].name;
                tensor_index_[name] = {si, ei};
            }
        }
    }

    // -----------------------------------------------------------------------
    // Look up a tensor by name and return a pointer to its raw data + entry
    // -----------------------------------------------------------------------
    const uint8_t* get_tensor_data(const std::string& name, SafetensorEntry& out_entry) const
    {
        auto it = tensor_index_.find(name);
        if (it == tensor_index_.end()) {
            fprintf(stderr, "[WeightLoader] Error: tensor '%s' not found\n", name.c_str());
            abort();
        }
        const auto& loc = it->second;
        const auto& shard = shards_[loc.shard_idx];
        out_entry = shard.entries[loc.entry_idx];
        return shard.data_start + out_entry.data_offset_begin;
    }

    // Check if a tensor name exists
    bool has_tensor(const std::string& name) const {
        return tensor_index_.count(name) > 0;
    }

    // -----------------------------------------------------------------------
    // Load a tensor from safetensors to GPU memory
    // -----------------------------------------------------------------------
    Tensor load_tensor_to_gpu(const std::string& name, DType expected_dtype)
    {
        SafetensorEntry entry;
        const uint8_t* src_data = get_tensor_data(name, entry);

        DType file_dtype = safetensors_dtype_to_dtype(entry.dtype);
        DType use_dtype = (expected_dtype != DType::FP32) ? expected_dtype : file_dtype;

        std::vector<int64_t> shape = entry.shape;
        if (shape.empty()) {
            shape.push_back(1);
        }

        int64_t numel = 1;
        for (auto d : shape) numel *= d;

        size_t file_element_size = dtype_size(file_dtype);
        size_t file_bytes = numel * file_element_size;

        size_t gpu_element_size = dtype_size(use_dtype);
        size_t gpu_bytes = numel * gpu_element_size;

        void* gpu_ptr = nullptr;
        {
            size_t free_mem = 0, total_mem = 0;
            hipMemGetInfo(&free_mem, &total_mem);
            if (gpu_bytes > free_mem || free_mem < (size_t)1024*1024*100) {
                fprintf(stderr, "[WeightLoader] OOM risk: tensor '%s' shape=[", name.c_str());
                for (size_t i = 0; i < shape.size(); i++) fprintf(stderr, "%s%ld", i?",":"", shape[i]);
                fprintf(stderr, "] needs %zu MB, free=%zu MB / total=%zu MB\n",
                        gpu_bytes/(1024*1024), free_mem/(1024*1024), total_mem/(1024*1024));
            }
        }
        CUDA_CHECK(hipMalloc(&gpu_ptr, gpu_bytes));

        if (file_dtype == use_dtype) {
            CUDA_CHECK(hipMemcpy(gpu_ptr, src_data, file_bytes, hipMemcpyHostToDevice));
        } else if (file_dtype == DType::FP32 && use_dtype == DType::BF16) {
            std::vector<__hip_bfloat16> converted(numel);
            const float* fp32_data = reinterpret_cast<const float*>(src_data);
            for (int64_t i = 0; i < numel; ++i) {
                converted[i] = __float2bfloat16(fp32_data[i]);
            }
            CUDA_CHECK(hipMemcpy(gpu_ptr, converted.data(), gpu_bytes, hipMemcpyHostToDevice));
        } else {
            CUDA_CHECK(hipMemcpy(gpu_ptr, src_data, file_bytes, hipMemcpyHostToDevice));
            use_dtype = file_dtype;
            gpu_bytes = file_bytes;
        }

        Tensor t;
        t.data = gpu_ptr;
        t.dtype = use_dtype;
        t.device_id = device_id_;
        t.ndim = static_cast<int>(shape.size());
        for (int i = 0; i < t.ndim && i < MAX_DIMS; ++i) {
            t.shape[i] = shape[i];
        }
        t.compute_strides();

        return t;
    }

    // -----------------------------------------------------------------------
    // Load a row-shard of a tensor to GPU.
    // -----------------------------------------------------------------------
    Tensor load_tensor_row_shard_to_gpu(const std::string& name, DType expected_dtype,
                                         int64_t row_start, int64_t num_rows)
    {
        SafetensorEntry entry;
        const uint8_t* src_data = get_tensor_data(name, entry);

        DType file_dtype = safetensors_dtype_to_dtype(entry.dtype);
        DType use_dtype = (expected_dtype != DType::FP32) ? expected_dtype : file_dtype;

        if (entry.shape.size() != 2) {
            fprintf(stderr, "[WeightLoader] Error: row shard requires 2D tensor, '%s' has %zu dims\n",
                    name.c_str(), entry.shape.size());
            abort();
        }

        int64_t total_rows = entry.shape[0];
        int64_t cols = entry.shape[1];

        if (row_start + num_rows > total_rows) {
            fprintf(stderr, "[WeightLoader] Error: row shard [%lld, %lld) exceeds tensor '%s' rows %lld\n",
                    (long long)row_start, (long long)(row_start + num_rows),
                    name.c_str(), (long long)total_rows);
            abort();
        }

        int64_t numel = num_rows * cols;
        size_t file_element_size = dtype_size(file_dtype);
        size_t gpu_element_size = dtype_size(use_dtype);
        size_t gpu_bytes = numel * gpu_element_size;

        size_t row_offset_bytes = static_cast<size_t>(row_start) * cols * file_element_size;
        size_t shard_file_bytes = static_cast<size_t>(num_rows) * cols * file_element_size;
        const uint8_t* shard_src = src_data + row_offset_bytes;

        void* gpu_ptr = nullptr;
        CUDA_CHECK(hipMalloc(&gpu_ptr, gpu_bytes));

        if (file_dtype == use_dtype) {
            CUDA_CHECK(hipMemcpy(gpu_ptr, shard_src, shard_file_bytes, hipMemcpyHostToDevice));
        } else if (file_dtype == DType::FP32 && use_dtype == DType::BF16) {
            std::vector<__hip_bfloat16> converted(numel);
            const float* fp32_data = reinterpret_cast<const float*>(shard_src);
            for (int64_t i = 0; i < numel; ++i) {
                converted[i] = __float2bfloat16(fp32_data[i]);
            }
            CUDA_CHECK(hipMemcpy(gpu_ptr, converted.data(), gpu_bytes, hipMemcpyHostToDevice));
        } else {
            CUDA_CHECK(hipMemcpy(gpu_ptr, shard_src, shard_file_bytes, hipMemcpyHostToDevice));
            use_dtype = file_dtype;
            gpu_bytes = shard_file_bytes;
        }

        Tensor t;
        t.data = gpu_ptr;
        t.dtype = use_dtype;
        t.device_id = device_id_;
        t.ndim = 2;
        t.shape[0] = num_rows;
        t.shape[1] = cols;
        t.compute_strides();

        return t;
    }

    // -----------------------------------------------------------------------
    // Load a raw packed tensor (MXFP4) to GPU without dtype conversion
    // -----------------------------------------------------------------------
    Tensor load_packed_tensor_to_gpu(const std::string& name)
    {
        SafetensorEntry entry;
        const uint8_t* src_data = get_tensor_data(name, entry);

        int64_t raw_bytes = entry.nbytes();

        void* gpu_ptr = nullptr;
        CUDA_CHECK(hipMalloc(&gpu_ptr, raw_bytes));
        CUDA_CHECK(hipMemcpy(gpu_ptr, src_data, raw_bytes, hipMemcpyHostToDevice));

        Tensor t;
        t.data = gpu_ptr;
        t.dtype = DType::UINT8;
        t.device_id = device_id_;

        t.ndim = static_cast<int>(entry.shape.size());
        if (t.ndim == 0) {
            t.ndim = 1;
            t.shape[0] = raw_bytes;
        } else {
            for (int i = 0; i < t.ndim && i < MAX_DIMS; ++i) {
                t.shape[i] = entry.shape[i];
            }
        }
        t.compute_strides();

        return t;
    }

    // -----------------------------------------------------------------------
    // Load a scale tensor for MXFP4 blocks
    // -----------------------------------------------------------------------
    Tensor load_scale_tensor_to_gpu(const std::string& name)
    {
        if (!has_tensor(name)) {
            Tensor t;
            return t;
        }

        SafetensorEntry entry;
        const uint8_t* src_data = get_tensor_data(name, entry);
        int64_t raw_bytes = entry.nbytes();

        void* gpu_ptr = nullptr;
        CUDA_CHECK(hipMalloc(&gpu_ptr, raw_bytes));
        CUDA_CHECK(hipMemcpy(gpu_ptr, src_data, raw_bytes, hipMemcpyHostToDevice));

        Tensor t;
        t.data = gpu_ptr;
        t.dtype = DType::UINT8;
        t.device_id = device_id_;
        t.ndim = static_cast<int>(entry.shape.size());
        if (t.ndim == 0) {
            t.ndim = 1;
            t.shape[0] = raw_bytes;
        } else {
            for (int i = 0; i < t.ndim && i < MAX_DIMS; ++i) {
                t.shape[i] = entry.shape[i];
            }
        }
        t.compute_strides();
        return t;
    }

    // -----------------------------------------------------------------------
    // Load a slice of a packed tensor to GPU.
    // -----------------------------------------------------------------------
    Tensor load_expert_slice_packed(const std::string& name, int global_expert_idx,
                                    int64_t num_experts)
    {
        SafetensorEntry entry;
        const uint8_t* src_data = get_tensor_data(name, entry);

        int64_t total_bytes = entry.nbytes();
        int64_t bytes_per_expert = total_bytes / num_experts;

        const uint8_t* expert_data = src_data + (int64_t)global_expert_idx * bytes_per_expert;

        void* gpu_ptr = nullptr;
        CUDA_CHECK(hipMalloc(&gpu_ptr, bytes_per_expert));
        CUDA_CHECK(hipMemcpy(gpu_ptr, expert_data, bytes_per_expert, hipMemcpyHostToDevice));

        Tensor t;
        t.data = gpu_ptr;
        t.dtype = DType::UINT8;
        t.device_id = device_id_;

        if (entry.shape.size() >= 2) {
            t.ndim = static_cast<int>(entry.shape.size()) - 1;
            for (int i = 0; i < t.ndim && i < MAX_DIMS; ++i) {
                t.shape[i] = entry.shape[i + 1];
            }
        } else {
            t.ndim = 1;
            t.shape[0] = bytes_per_expert;
        }
        t.compute_strides();
        return t;
    }

    // -----------------------------------------------------------------------
    // Load experts from packed format
    // -----------------------------------------------------------------------
    void load_packed_experts(int layer, const char* prefix, MoEWeights& moe)
    {
        char name_buf[256];
        constexpr int64_t num_experts = ModelConfig::num_experts;

        for (int local_e = 0; local_e < ModelConfig::experts_per_gpu; ++local_e) {
            int global_e = expert_start_ + local_e;

            snprintf(name_buf, sizeof(name_buf),
                     "%s%d.mlp.experts.gate_up_proj_blocks", prefix, layer);
            moe.expert_mlp1[local_e] = load_expert_slice_packed(name_buf, global_e, num_experts);

            snprintf(name_buf, sizeof(name_buf),
                     "%s%d.mlp.experts.gate_up_proj_scales", prefix, layer);
            moe.expert_mlp1_scales[local_e] = load_expert_slice_packed(name_buf, global_e, num_experts);

            snprintf(name_buf, sizeof(name_buf),
                     "%s%d.mlp.experts.down_proj_blocks", prefix, layer);
            moe.expert_mlp2[local_e] = load_expert_slice_packed(name_buf, global_e, num_experts);

            snprintf(name_buf, sizeof(name_buf),
                     "%s%d.mlp.experts.down_proj_scales", prefix, layer);
            moe.expert_mlp2_scales[local_e] = load_expert_slice_packed(name_buf, global_e, num_experts);
        }
    }

    // -----------------------------------------------------------------------
    // Load experts from per-expert format
    // -----------------------------------------------------------------------
    void load_per_expert_weights(int layer, const char* prefix, MoEWeights& moe)
    {
        char name_buf[256];

        for (int local_e = 0; local_e < ModelConfig::experts_per_gpu; ++local_e) {
            int global_e = expert_start_ + local_e;

            snprintf(name_buf, sizeof(name_buf),
                     "%s%d.mlp.experts.%d.mlp1.weight", prefix, layer, global_e);
            if (!has_tensor(name_buf)) {
                snprintf(name_buf, sizeof(name_buf),
                         "%s%d.mlp.experts.%d.gate_up_proj.weight", prefix, layer, global_e);
            }
            if (!has_tensor(name_buf)) {
                snprintf(name_buf, sizeof(name_buf),
                         "%s%d.mlp.experts.%d.w1.weight", prefix, layer, global_e);
            }
            moe.expert_mlp1[local_e] = load_packed_tensor_to_gpu(name_buf);

            snprintf(name_buf, sizeof(name_buf),
                     "%s%d.mlp.experts.%d.mlp1.weight_scale", prefix, layer, global_e);
            if (!has_tensor(name_buf)) {
                snprintf(name_buf, sizeof(name_buf),
                         "%s%d.mlp.experts.%d.gate_up_proj.weight_scale", prefix, layer, global_e);
            }
            moe.expert_mlp1_scales[local_e] = load_scale_tensor_to_gpu(name_buf);

            snprintf(name_buf, sizeof(name_buf),
                     "%s%d.mlp.experts.%d.mlp2.weight", prefix, layer, global_e);
            if (!has_tensor(name_buf)) {
                snprintf(name_buf, sizeof(name_buf),
                         "%s%d.mlp.experts.%d.down_proj.weight", prefix, layer, global_e);
            }
            if (!has_tensor(name_buf)) {
                snprintf(name_buf, sizeof(name_buf),
                         "%s%d.mlp.experts.%d.w2.weight", prefix, layer, global_e);
            }
            moe.expert_mlp2[local_e] = load_packed_tensor_to_gpu(name_buf);

            snprintf(name_buf, sizeof(name_buf),
                     "%s%d.mlp.experts.%d.mlp2.weight_scale", prefix, layer, global_e);
            if (!has_tensor(name_buf)) {
                snprintf(name_buf, sizeof(name_buf),
                         "%s%d.mlp.experts.%d.down_proj.weight_scale", prefix, layer, global_e);
            }
            moe.expert_mlp2_scales[local_e] = load_scale_tensor_to_gpu(name_buf);
        }
    }

    // -----------------------------------------------------------------------
    // Load all weights for a single layer
    // -----------------------------------------------------------------------
    void load_layer_weights(int layer)
    {
        char name_buf[256];
        const char* prefix = "model.layers.";

        AttentionWeights& attn = attn_weights_[layer];

        constexpr int64_t tp_q_dim  = ModelConfig::tp_q_dim;
        constexpr int64_t tp_kv_dim = ModelConfig::tp_kv_dim;
        constexpr int64_t tp_qkv    = ModelConfig::tp_qkv_size;
        constexpr int64_t tp_attn_out = ModelConfig::tp_attn_out_size;
        constexpr int64_t full_q_dim = ModelConfig::num_q_heads * ModelConfig::head_dim;
        constexpr int64_t full_kv_dim = ModelConfig::num_kv_heads * ModelConfig::head_dim;

        // Try fused QKV first
        snprintf(name_buf, sizeof(name_buf), "%s%d.self_attn.qkv_proj.weight", prefix, layer);
        if (has_tensor(name_buf)) {
            SafetensorEntry entry;
            const uint8_t* src_data = get_tensor_data(name_buf, entry);
            int64_t cols = ModelConfig::hidden_size;
            size_t elem_sz = sizeof(__hip_bfloat16);

            size_t gpu_bytes = tp_qkv * cols * elem_sz;
            void* gpu_ptr = nullptr;
            CUDA_CHECK(hipMalloc(&gpu_ptr, gpu_bytes));

            __hip_bfloat16* dst = static_cast<__hip_bfloat16*>(gpu_ptr);

            int64_t q_row_start = rank_ * tp_q_dim;
            size_t q_offset = q_row_start * cols * elem_sz;
            size_t q_bytes = tp_q_dim * cols * elem_sz;
            CUDA_CHECK(hipMemcpy(dst, src_data + q_offset, q_bytes, hipMemcpyHostToDevice));
            dst += tp_q_dim * cols;

            int64_t k_row_start = full_q_dim + rank_ * tp_kv_dim;
            size_t k_offset = k_row_start * cols * elem_sz;
            size_t k_bytes = tp_kv_dim * cols * elem_sz;
            CUDA_CHECK(hipMemcpy(dst, src_data + k_offset, k_bytes, hipMemcpyHostToDevice));
            dst += tp_kv_dim * cols;

            int64_t v_row_start = full_q_dim + full_kv_dim + rank_ * tp_kv_dim;
            size_t v_offset = v_row_start * cols * elem_sz;
            size_t v_bytes = tp_kv_dim * cols * elem_sz;
            CUDA_CHECK(hipMemcpy(dst, src_data + v_offset, v_bytes, hipMemcpyHostToDevice));

            attn.qkv_proj.data = gpu_ptr;
            attn.qkv_proj.dtype = DType::BF16;
            attn.qkv_proj.device_id = device_id_;
            attn.qkv_proj.ndim = 2;
            attn.qkv_proj.shape[0] = tp_qkv;
            attn.qkv_proj.shape[1] = cols;
            attn.qkv_proj.compute_strides();
        } else {
            snprintf(name_buf, sizeof(name_buf), "%s%d.self_attn.q_proj.weight", prefix, layer);
            Tensor q = load_tensor_row_shard_to_gpu(name_buf, DType::BF16,
                rank_ * tp_q_dim, tp_q_dim);

            snprintf(name_buf, sizeof(name_buf), "%s%d.self_attn.k_proj.weight", prefix, layer);
            Tensor k = load_tensor_row_shard_to_gpu(name_buf, DType::BF16,
                rank_ * tp_kv_dim, tp_kv_dim);

            snprintf(name_buf, sizeof(name_buf), "%s%d.self_attn.v_proj.weight", prefix, layer);
            Tensor v = load_tensor_row_shard_to_gpu(name_buf, DType::BF16,
                rank_ * tp_kv_dim, tp_kv_dim);

            int64_t cols = ModelConfig::hidden_size;
            attn.qkv_proj = allocate_tensor(DType::BF16,
                {tp_qkv, cols}, device_id_);

            __hip_bfloat16* dst = attn.qkv_proj.bf16_ptr();
            size_t q_bytes = tp_q_dim * cols * sizeof(__hip_bfloat16);
            size_t k_bytes = tp_kv_dim * cols * sizeof(__hip_bfloat16);
            size_t v_bytes = tp_kv_dim * cols * sizeof(__hip_bfloat16);

            CUDA_CHECK(hipMemcpy(dst, q.bf16_ptr(), q_bytes, hipMemcpyDeviceToDevice));
            dst += tp_q_dim * cols;
            CUDA_CHECK(hipMemcpy(dst, k.bf16_ptr(), k_bytes, hipMemcpyDeviceToDevice));
            dst += tp_kv_dim * cols;
            CUDA_CHECK(hipMemcpy(dst, v.bf16_ptr(), v_bytes, hipMemcpyDeviceToDevice));

            free_tensor(q);
            free_tensor(k);
            free_tensor(v);
        }

        // QKV bias
        snprintf(name_buf, sizeof(name_buf), "%s%d.self_attn.q_proj.bias", prefix, layer);
        if (has_tensor(name_buf)) {
            size_t elem_sz = sizeof(__hip_bfloat16);

            SafetensorEntry q_entry, k_entry, v_entry;
            snprintf(name_buf, sizeof(name_buf), "%s%d.self_attn.q_proj.bias", prefix, layer);
            const uint8_t* q_src = get_tensor_data(name_buf, q_entry);
            snprintf(name_buf, sizeof(name_buf), "%s%d.self_attn.k_proj.bias", prefix, layer);
            const uint8_t* k_src = get_tensor_data(name_buf, k_entry);
            snprintf(name_buf, sizeof(name_buf), "%s%d.self_attn.v_proj.bias", prefix, layer);
            const uint8_t* v_src = get_tensor_data(name_buf, v_entry);

            DType q_dtype = safetensors_dtype_to_dtype(q_entry.dtype);

            size_t bias_bytes = tp_qkv * elem_sz;
            void* bias_ptr = nullptr;
            CUDA_CHECK(hipMalloc(&bias_ptr, bias_bytes));
            __hip_bfloat16* dst = static_cast<__hip_bfloat16*>(bias_ptr);

            auto copy_bias_shard = [&](const uint8_t* src, DType src_dtype,
                                        int64_t shard_start, int64_t shard_len) {
                if (src_dtype == DType::BF16) {
                    CUDA_CHECK(hipMemcpy(dst, src + shard_start * elem_sz,
                                          shard_len * elem_sz, hipMemcpyHostToDevice));
                } else {
                    std::vector<__hip_bfloat16> tmp(shard_len);
                    const float* fp32 = reinterpret_cast<const float*>(src) + shard_start;
                    for (int64_t i = 0; i < shard_len; ++i) tmp[i] = __float2bfloat16(fp32[i]);
                    CUDA_CHECK(hipMemcpy(dst, tmp.data(), shard_len * elem_sz, hipMemcpyHostToDevice));
                }
                dst += shard_len;
            };

            copy_bias_shard(q_src, q_dtype, rank_ * tp_q_dim, tp_q_dim);
            copy_bias_shard(k_src, safetensors_dtype_to_dtype(k_entry.dtype),
                            rank_ * tp_kv_dim, tp_kv_dim);
            copy_bias_shard(v_src, safetensors_dtype_to_dtype(v_entry.dtype),
                            rank_ * tp_kv_dim, tp_kv_dim);

            attn.qkv_bias.data = bias_ptr;
            attn.qkv_bias.dtype = DType::BF16;
            attn.qkv_bias.device_id = device_id_;
            attn.qkv_bias.ndim = 1;
            attn.qkv_bias.shape[0] = tp_qkv;
            attn.qkv_bias.compute_strides();

            if (layer == 0)
                fprintf(stderr, "[WeightLoader] Loaded QKV bias [%lld] (src dtype: %s)\n",
                        (long long)tp_qkv, q_entry.dtype.c_str());
        }

        // O-proj
        snprintf(name_buf, sizeof(name_buf), "%s%d.self_attn.o_proj.weight", prefix, layer);
        if constexpr (ModelConfig::tp_size == 1) {
            attn.o_proj = load_tensor_to_gpu(name_buf, DType::BF16);
        } else {
            attn.o_proj = load_tensor_row_shard_to_gpu(name_buf, DType::BF16,
                rank_ * tp_attn_out, tp_attn_out);
        }

        // O-proj bias
        snprintf(name_buf, sizeof(name_buf), "%s%d.self_attn.o_proj.bias", prefix, layer);
        if (has_tensor(name_buf)) {
            attn.o_proj_bias = load_tensor_to_gpu(name_buf, DType::BF16);
            if (layer == 0)
                fprintf(stderr, "[WeightLoader] Loaded O-proj bias [%lld]\n",
                        (long long)attn.o_proj_bias.shape[0]);
        }

        // Learned attention sinks
        snprintf(name_buf, sizeof(name_buf), "%s%d.self_attn.sinks", prefix, layer);
        if (has_tensor(name_buf)) {
            Tensor sinks_bf16 = load_tensor_to_gpu(name_buf, DType::BF16);
            int n = static_cast<int>(sinks_bf16.shape[0]);

            float* sinks_fp32 = nullptr;
            CUDA_CHECK(hipMalloc(&sinks_fp32, n * sizeof(float)));

            std::vector<__hip_bfloat16> h_bf16(n);
            std::vector<float> h_fp32(n);
            CUDA_CHECK(hipMemcpy(h_bf16.data(), sinks_bf16.bf16_ptr(),
                                  n * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i)
                h_fp32[i] = __bfloat162float(h_bf16[i]);
            CUDA_CHECK(hipMemcpy(sinks_fp32, h_fp32.data(),
                                  n * sizeof(float), hipMemcpyHostToDevice));
            free_tensor(sinks_bf16);

            attn.sinks.data = sinks_fp32;
            attn.sinks.dtype = DType::FP32;
            attn.sinks.device_id = device_id_;
            attn.sinks.ndim = 1;
            attn.sinks.shape[0] = n;
            attn.sinks.compute_strides();

            if (layer == 0)
                fprintf(stderr, "[WeightLoader] Loaded attention sinks [%d] (converted BF16->FP32)\n", n);
        }

        // Attention input layernorm
        snprintf(name_buf, sizeof(name_buf), "%s%d.input_layernorm.weight", prefix, layer);
        attn.norm = load_tensor_to_gpu(name_buf, DType::BF16);

        // --- MoE weights ---
        MoEWeights& moe = moe_weights_[layer];

        // Router
        snprintf(name_buf, sizeof(name_buf), "%s%d.mlp.gate.weight", prefix, layer);
        if (!has_tensor(name_buf)) {
            snprintf(name_buf, sizeof(name_buf), "%s%d.mlp.router.weight", prefix, layer);
        }
        moe.router = load_tensor_to_gpu(name_buf, DType::BF16);
        if (layer == 0)
            fprintf(stderr, "[WeightLoader] Loaded router weight '%s' shape=[%lld, %lld] ndim=%d\n",
                    name_buf,
                    (long long)moe.router.shape[0], (long long)moe.router.shape[1],
                    moe.router.ndim);

        // Router bias
        snprintf(name_buf, sizeof(name_buf), "%s%d.mlp.gate.bias", prefix, layer);
        if (!has_tensor(name_buf)) {
            snprintf(name_buf, sizeof(name_buf), "%s%d.mlp.router.bias", prefix, layer);
        }
        if (has_tensor(name_buf)) {
            moe.router_bias = load_tensor_to_gpu(name_buf, DType::BF16);
            if (layer == 0)
                fprintf(stderr, "[WeightLoader] Loaded router bias [%lld]\n",
                        (long long)moe.router_bias.shape[0]);
        }

        // Post-attention / pre-MoE layernorm
        snprintf(name_buf, sizeof(name_buf), "%s%d.post_attention_layernorm.weight", prefix, layer);
        moe.norm = load_tensor_to_gpu(name_buf, DType::BF16);

        // Expert weights
        snprintf(name_buf, sizeof(name_buf),
                 "%s%d.mlp.experts.gate_up_proj_blocks", prefix, layer);
        bool packed_experts = has_tensor(name_buf);

        if (layer == 0) {
            fprintf(stderr, "[WeightLoader] Expert format: %s\n",
                    packed_experts ? "packed (gate_up_proj_blocks)" : "per-expert");
            if (!packed_experts) {
                fprintf(stderr, "[WeightLoader] Dumping tensor names containing 'expert' or 'mlp':\n");
                int count = 0;
                for (const auto& kv : tensor_index_) {
                    if (kv.first.find("expert") != std::string::npos ||
                        (kv.first.find("layers.0.mlp") != std::string::npos)) {
                        fprintf(stderr, "  [%d] %s\n", count, kv.first.c_str());
                        if (++count >= 30) break;
                    }
                }
            }
        }

        if (packed_experts) {
            load_packed_experts(layer, prefix, moe);
        } else {
            load_per_expert_weights(layer, prefix, moe);
        }

        // Expert MLP biases
        snprintf(name_buf, sizeof(name_buf), "%s%d.mlp.experts.gate_up_proj_bias", prefix, layer);
        if (has_tensor(name_buf)) {
            moe.gate_up_proj_bias = load_tensor_to_gpu(name_buf, DType::BF16);
            if (layer == 0)
                fprintf(stderr, "[WeightLoader] Loaded gate_up_proj_bias [%lld, %lld]\n",
                        (long long)moe.gate_up_proj_bias.shape[0],
                        (long long)moe.gate_up_proj_bias.shape[1]);
        }
        snprintf(name_buf, sizeof(name_buf), "%s%d.mlp.experts.down_proj_bias", prefix, layer);
        if (has_tensor(name_buf)) {
            moe.down_proj_bias = load_tensor_to_gpu(name_buf, DType::BF16);
            if (layer == 0)
                fprintf(stderr, "[WeightLoader] Loaded down_proj_bias [%lld, %lld]\n",
                        (long long)moe.down_proj_bias.shape[0],
                        (long long)moe.down_proj_bias.shape[1]);
        }

        fprintf(stderr, "[WeightLoader] GPU %d: loaded layer %d/%d\n",
                device_id_, layer + 1, ModelConfig::num_layers);
    }

    // -----------------------------------------------------------------------
    // Member data
    // -----------------------------------------------------------------------
    std::string weights_dir_;
    int rank_        = 0;
    int world_size_  = 1;
    int tp_rank_     = 0;
    int ep_rank_     = 0;
    int tp_size_     = 1;
    int ep_size_     = 1;
    int device_id_   = 0;
    int expert_start_ = 0;
    int expert_end_   = 0;
    bool initialized_ = false;

    std::vector<SafetensorShard> shards_;
    std::unordered_map<std::string, TensorLocation> tensor_index_;

    Tensor embedding_;
    Tensor final_norm_;
    Tensor lm_head_;
    std::vector<AttentionWeights> attn_weights_;
    std::vector<MoEWeights> moe_weights_;
};

// ---------------------------------------------------------------------------
// Extern wrapper functions
// ---------------------------------------------------------------------------

WeightLoader* weight_loader_create() {
    return new WeightLoader();
}

void weight_loader_destroy(WeightLoader* wl) {
    if (wl) delete wl;
}

void weight_loader_init(WeightLoader* wl, const std::string& weights_dir,
                         int tp_rank, int ep_rank, int tp_size, int ep_size,
                         int device_id) {
    wl->init(weights_dir, tp_rank, ep_rank, tp_size, ep_size, device_id);
}

void weight_loader_load_all(WeightLoader* wl) {
    wl->load_all_weights();
}

// Accessors
const __hip_bfloat16* weight_loader_embedding(WeightLoader* wl) {
    return static_cast<const __hip_bfloat16*>(wl->get_embedding_weight().data);
}

const __hip_bfloat16* weight_loader_final_norm(WeightLoader* wl) {
    return static_cast<const __hip_bfloat16*>(wl->get_final_norm().data);
}

const __hip_bfloat16* weight_loader_lm_head(WeightLoader* wl) {
    return static_cast<const __hip_bfloat16*>(wl->get_lm_head().data);
}

const __hip_bfloat16* weight_loader_attn_qkv(WeightLoader* wl, int layer) {
    return static_cast<const __hip_bfloat16*>(wl->get_attention_weights(layer).qkv_proj.data);
}

const __hip_bfloat16* weight_loader_attn_o_proj(WeightLoader* wl, int layer) {
    return static_cast<const __hip_bfloat16*>(wl->get_attention_weights(layer).o_proj.data);
}

const __hip_bfloat16* weight_loader_attn_norm(WeightLoader* wl, int layer) {
    return static_cast<const __hip_bfloat16*>(wl->get_attention_weights(layer).norm.data);
}

const __hip_bfloat16* weight_loader_attn_qkv_bias(WeightLoader* wl, int layer) {
    return static_cast<const __hip_bfloat16*>(wl->get_attention_weights(layer).qkv_bias.data);
}

const __hip_bfloat16* weight_loader_attn_o_proj_bias(WeightLoader* wl, int layer) {
    return static_cast<const __hip_bfloat16*>(wl->get_attention_weights(layer).o_proj_bias.data);
}

const float* weight_loader_attn_sinks(WeightLoader* wl, int layer) {
    return static_cast<const float*>(wl->get_attention_weights(layer).sinks.data);
}

const __hip_bfloat16* weight_loader_moe_router(WeightLoader* wl, int layer) {
    return static_cast<const __hip_bfloat16*>(wl->get_moe_weights(layer).router.data);
}

const __hip_bfloat16* weight_loader_moe_norm(WeightLoader* wl, int layer) {
    return static_cast<const __hip_bfloat16*>(wl->get_moe_weights(layer).norm.data);
}

const __hip_bfloat16* weight_loader_moe_router_bias(WeightLoader* wl, int layer) {
    return static_cast<const __hip_bfloat16*>(wl->get_moe_weights(layer).router_bias.data);
}

const __hip_bfloat16* weight_loader_expert_gate_up_bias(WeightLoader* wl, int layer) {
    return static_cast<const __hip_bfloat16*>(wl->get_moe_weights(layer).gate_up_proj_bias.data);
}

const __hip_bfloat16* weight_loader_expert_down_bias(WeightLoader* wl, int layer) {
    return static_cast<const __hip_bfloat16*>(wl->get_moe_weights(layer).down_proj_bias.data);
}

const uint8_t* weight_loader_expert_mlp1_packed(WeightLoader* wl, int layer, int local_expert) {
    return static_cast<const uint8_t*>(wl->get_moe_weights(layer).expert_mlp1[local_expert].data);
}

const uint8_t* weight_loader_expert_mlp1_scales(WeightLoader* wl, int layer, int local_expert) {
    return static_cast<const uint8_t*>(wl->get_moe_weights(layer).expert_mlp1_scales[local_expert].data);
}

int weight_loader_expert_mlp1_numel(WeightLoader* wl, int layer, int local_expert) {
    auto t = wl->get_moe_weights(layer).expert_mlp1[local_expert];
    int64_t numel = 1;
    for (int i = 0; i < t.ndim; i++) numel *= t.shape[i];
    return static_cast<int>(numel);
}

const uint8_t* weight_loader_expert_mlp2_packed(WeightLoader* wl, int layer, int local_expert) {
    return static_cast<const uint8_t*>(wl->get_moe_weights(layer).expert_mlp2[local_expert].data);
}

const uint8_t* weight_loader_expert_mlp2_scales(WeightLoader* wl, int layer, int local_expert) {
    return static_cast<const uint8_t*>(wl->get_moe_weights(layer).expert_mlp2_scales[local_expert].data);
}

int weight_loader_expert_mlp2_numel(WeightLoader* wl, int layer, int local_expert) {
    auto t = wl->get_moe_weights(layer).expert_mlp2[local_expert];
    int64_t numel = 1;
    for (int i = 0; i < t.ndim; i++) numel *= t.shape[i];
    return static_cast<int>(numel);
}

} // namespace gptoss
