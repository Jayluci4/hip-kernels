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

} // extern "C"
