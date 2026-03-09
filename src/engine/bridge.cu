// Bridge between pybind11 (compiled by g++) and InferenceEngine (compiled by hipcc)

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>

// Forward declare
namespace gptoss {
class InferenceEngine;
}

// The InferenceEngine wrapper functions
namespace gptoss {

extern InferenceEngine* inference_engine_create();
extern void inference_engine_destroy(InferenceEngine* eng);
extern void inference_engine_init(InferenceEngine* eng, const std::string& weights_dir);
extern void inference_engine_shutdown(InferenceEngine* eng);
extern bool inference_engine_is_initialized(InferenceEngine* eng);
extern std::vector<int32_t> inference_engine_generate(
    InferenceEngine* eng, const std::vector<int32_t>& prompt,
    int max_tokens, float temperature, float top_p);
extern std::vector<std::vector<int32_t>> inference_engine_generate_batch(
    InferenceEngine* eng, const std::vector<std::vector<int32_t>>& prompts,
    int max_tokens, float temperature, float top_p);

} // namespace gptoss

extern "C" {

void* gptoss_engine_create() {
    return gptoss::inference_engine_create();
}

void gptoss_engine_destroy(void* engine) {
    gptoss::inference_engine_destroy(
        static_cast<gptoss::InferenceEngine*>(engine));
}

void gptoss_engine_init(void* engine, const char* weights_dir) {
    gptoss::inference_engine_init(
        static_cast<gptoss::InferenceEngine*>(engine),
        std::string(weights_dir));
}

void gptoss_engine_shutdown(void* engine) {
    gptoss::inference_engine_shutdown(
        static_cast<gptoss::InferenceEngine*>(engine));
}

int gptoss_engine_generate(void* engine,
                            const int32_t* prompt_tokens,
                            int prompt_len,
                            int max_tokens,
                            float temperature,
                            float top_p,
                            int32_t* output_tokens,
                            int output_capacity) {
    auto* eng = static_cast<gptoss::InferenceEngine*>(engine);
    std::vector<int32_t> prompt(prompt_tokens, prompt_tokens + prompt_len);
    auto result = gptoss::inference_engine_generate(
        eng, prompt, max_tokens, temperature, top_p);
    int n = std::min(static_cast<int>(result.size()), output_capacity);
    std::memcpy(output_tokens, result.data(), n * sizeof(int32_t));
    return n;
}

int gptoss_engine_is_initialized(void* engine) {
    return gptoss::inference_engine_is_initialized(
        static_cast<gptoss::InferenceEngine*>(engine)) ? 1 : 0;
}

// Batched generation: prompts is [batch_size][prompt_len] flattened.
// Returns outputs flattened as [seq0_len, seq0_tokens..., seq1_len, seq1_tokens..., ...]
int gptoss_engine_generate_batch(void* engine,
                                  const int32_t* prompts_flat,
                                  int batch_size,
                                  int prompt_len,
                                  int max_tokens,
                                  float temperature,
                                  float top_p,
                                  int32_t* output_flat,
                                  int output_capacity) {
    auto* eng = static_cast<gptoss::InferenceEngine*>(engine);
    std::vector<std::vector<int32_t>> prompts(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        prompts[i].assign(prompts_flat + i * prompt_len,
                          prompts_flat + (i + 1) * prompt_len);
    }
    auto results = gptoss::inference_engine_generate_batch(
        eng, prompts, max_tokens, temperature, top_p);

    // Pack results: [len0, tok0_0, tok0_1, ..., len1, tok1_0, ...]
    int offset = 0;
    for (int i = 0; i < batch_size; ++i) {
        int n = static_cast<int>(results[i].size());
        if (offset + 1 + n > output_capacity) break;
        output_flat[offset++] = n;
        std::memcpy(output_flat + offset, results[i].data(), n * sizeof(int32_t));
        offset += n;
    }
    return offset;
}

} // extern "C"
