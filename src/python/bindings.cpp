// Python bindings for GPT-OSS-120B inference engine
// Module name: gptoss
//
// Exposes:
//   - InferenceEngine: main class for multi-GPU inference
//   - ModelConfig: read-only architecture constants
//
// Usage from Python:
//   import gptoss
//   engine = gptoss.InferenceEngine()
//   engine.init("/weights/gpt-oss-120b")
//   tokens = engine.generate([1, 2, 3], max_tokens=128)
//   engine.shutdown()

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

// The InferenceEngine class is defined in engine.cu. In the real build system
// this would be linked together; here we include the forward declaration and
// rely on the linker.
namespace gptoss {

class InferenceEngine;

// We need the ModelConfig for exposing constants.
} // namespace gptoss

#include "config.h"
#include "profiler.h"

// Forward declaration -- the actual class is compiled from engine.cu and
// linked into the shared library.
namespace gptoss {

// Minimal wrapper that forwards to the HIP-compiled InferenceEngine.
// This avoids exposing HIP headers to the pybind11 compilation unit.
// In the build system, bindings.cpp is compiled with a standard C++ compiler
// while engine.cu is compiled with hipcc; they are linked together.

// Thin C-linkage interface to the HIP-compiled engine
extern "C" {
    void* gptoss_engine_create();
    void  gptoss_engine_destroy(void* engine);
    void  gptoss_engine_init(void* engine, const char* weights_dir);
    void  gptoss_engine_shutdown(void* engine);
    int   gptoss_engine_generate(void* engine,
                                 const int32_t* prompt_tokens,
                                 int prompt_len,
                                 int max_tokens,
                                 float temperature,
                                 float top_p,
                                 int32_t* output_tokens,
                                 int output_capacity);
    int   gptoss_engine_is_initialized(void* engine);
}

// Python-facing wrapper that translates between pybind11 types and the
// C-linkage engine interface.
class PyInferenceEngine {
public:
    PyInferenceEngine() {
        engine_ = gptoss_engine_create();
        if (!engine_) {
            throw std::runtime_error("Failed to create InferenceEngine");
        }
    }

    ~PyInferenceEngine() {
        if (engine_) {
            gptoss_engine_destroy(engine_);
            engine_ = nullptr;
        }
    }

    // Non-copyable
    PyInferenceEngine(const PyInferenceEngine&) = delete;
    PyInferenceEngine& operator=(const PyInferenceEngine&) = delete;

    void init(const std::string& weights_dir) {
        try {
            gptoss_engine_init(engine_, weights_dir.c_str());
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("Engine init failed: ") + e.what());
        }
    }

    std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens,
                                  int max_tokens    = 512,
                                  float temperature = 0.8f,
                                  float top_p       = 0.95f) {
        if (prompt_tokens.empty()) {
            throw std::invalid_argument("prompt_tokens must not be empty");
        }
        if (max_tokens <= 0) {
            throw std::invalid_argument("max_tokens must be positive");
        }
        if (temperature < 0.0f) {
            throw std::invalid_argument("temperature must be non-negative");
        }
        if (top_p <= 0.0f || top_p > 1.0f) {
            throw std::invalid_argument("top_p must be in (0, 1]");
        }

        // Allocate output buffer large enough for prompt + generated tokens
        const int output_capacity =
            static_cast<int>(prompt_tokens.size()) + max_tokens;
        std::vector<int32_t> output(output_capacity);

        int num_tokens = 0;
        try {
            num_tokens = gptoss_engine_generate(
                engine_,
                prompt_tokens.data(),
                static_cast<int>(prompt_tokens.size()),
                max_tokens,
                temperature,
                top_p,
                output.data(),
                output_capacity);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("Generation failed: ") + e.what());
        }

        if (num_tokens < 0) {
            throw std::runtime_error("Generation returned error code");
        }
        output.resize(num_tokens);
        return output;
    }

    void shutdown() {
        try {
            gptoss_engine_shutdown(engine_);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("Shutdown failed: ") + e.what());
        }
    }

    bool is_initialized() const {
        return gptoss_engine_is_initialized(engine_) != 0;
    }

private:
    void* engine_ = nullptr;
};

} // namespace gptoss

namespace py = pybind11;

// -----------------------------------------------------------------------
// Module definition
// -----------------------------------------------------------------------

PYBIND11_MODULE(gptoss, m) {
    m.doc() = "GPT-OSS-120B inference engine -- multi-GPU HIP/RCCL backend";

    // ------------------------------------------------------------------
    // InferenceEngine
    // ------------------------------------------------------------------
    py::class_<gptoss::PyInferenceEngine>(m, "InferenceEngine",
        "Main inference engine for GPT-OSS-120B.\n\n"
        "Controls 2x MI300X GPUs with Expert Parallelism (EP2).\n"
        "Typical usage:\n"
        "  engine = gptoss.InferenceEngine()\n"
        "  engine.init('/weights/gpt-oss-120b')\n"
        "  tokens = engine.generate([1, 2, 3], max_tokens=128)\n"
        "  engine.shutdown()")

        .def(py::init<>(),
             "Create an InferenceEngine instance (does not initialize GPUs).")

        .def("init",
             [](gptoss::PyInferenceEngine& self, const std::string& weights_dir) {
                 // Release the GIL during init (heavy I/O + HIP work)
                 py::gil_scoped_release release;
                 self.init(weights_dir);
             },
             py::arg("weights_dir"),
             "Initialize the engine: detect GPUs, load weights, warm up.\n\n"
             "Args:\n"
             "    weights_dir: Path to the model weights directory.\n\n"
             "Raises:\n"
             "    RuntimeError: If GPU detection fails or weights cannot be loaded.")

        .def("generate",
             [](gptoss::PyInferenceEngine& self,
                const std::vector<int32_t>& prompt_tokens,
                int max_tokens,
                float temperature,
                float top_p) -> std::vector<int32_t> {
                 // Release the GIL during generation (GPU-bound)
                 py::gil_scoped_release release;
                 return self.generate(prompt_tokens, max_tokens,
                                     temperature, top_p);
             },
             py::arg("prompt_tokens"),
             py::arg("max_tokens")   = 512,
             py::arg("temperature")  = 0.8f,
             py::arg("top_p")        = 0.95f,
             "Generate tokens autoregressively.\n\n"
             "Args:\n"
             "    prompt_tokens: List of input token IDs.\n"
             "    max_tokens: Maximum number of new tokens to generate (default 512).\n"
             "    temperature: Sampling temperature; 0 = greedy (default 0.8).\n"
             "    top_p: Nucleus sampling probability threshold (default 0.95).\n\n"
             "Returns:\n"
             "    List of token IDs (prompt + generated tokens).\n\n"
             "Raises:\n"
             "    ValueError: If arguments are invalid.\n"
             "    RuntimeError: If the engine is not initialized or generation fails.")

        .def("shutdown",
             [](gptoss::PyInferenceEngine& self) {
                 py::gil_scoped_release release;
                 self.shutdown();
             },
             "Shut down the engine and release all GPU resources.")

        .def_property_readonly("is_initialized",
                               &gptoss::PyInferenceEngine::is_initialized,
                               "True if the engine has been initialized.")

        .def("__repr__",
             [](const gptoss::PyInferenceEngine& self) {
                 return std::string("<gptoss.InferenceEngine initialized=") +
                        (self.is_initialized() ? "True" : "False") + ">";
             });

    // ------------------------------------------------------------------
    // ModelConfig (read-only properties)
    // ------------------------------------------------------------------
    py::class_<gptoss::ModelConfig>(m, "ModelConfig",
        "Read-only model architecture constants for GPT-OSS-120B.")

        // We expose a default-constructible instance so users can do:
        //   cfg = gptoss.ModelConfig()
        //   print(cfg.num_layers)
        .def(py::init<>())

        // Architecture
        .def_property_readonly_static("num_layers",
            [](py::object) { return gptoss::ModelConfig::num_layers; },
            "Number of transformer layers (36).")
        .def_property_readonly_static("hidden_size",
            [](py::object) { return gptoss::ModelConfig::hidden_size; },
            "Hidden dimension (2880).")
        .def_property_readonly_static("num_q_heads",
            [](py::object) { return gptoss::ModelConfig::num_q_heads; },
            "Number of query attention heads (64).")
        .def_property_readonly_static("num_kv_heads",
            [](py::object) { return gptoss::ModelConfig::num_kv_heads; },
            "Number of key/value attention heads (8).")
        .def_property_readonly_static("head_dim",
            [](py::object) { return gptoss::ModelConfig::head_dim; },
            "Dimension per attention head (64).")
        .def_property_readonly_static("qkv_size",
            [](py::object) { return gptoss::ModelConfig::qkv_size; },
            "QKV projection size (5120).")
        .def_property_readonly_static("attn_out_size",
            [](py::object) { return gptoss::ModelConfig::attn_out_size; },
            "Attention output size (4096).")

        // MoE
        .def_property_readonly_static("num_experts",
            [](py::object) { return gptoss::ModelConfig::num_experts; },
            "Total number of MoE experts (128).")
        .def_property_readonly_static("num_active_experts",
            [](py::object) { return gptoss::ModelConfig::num_active_experts; },
            "Number of active experts per token (4).")
        .def_property_readonly_static("expert_intermediate_size",
            [](py::object) { return gptoss::ModelConfig::expert_intermediate_size; },
            "Expert FFN intermediate dimension (2880).")
        .def_property_readonly_static("experts_per_gpu",
            [](py::object) { return gptoss::ModelConfig::experts_per_gpu; },
            "Experts per GPU under EP2 (64).")

        // Vocabulary
        .def_property_readonly_static("vocab_size",
            [](py::object) { return gptoss::ModelConfig::vocab_size; },
            "Vocabulary size (201088).")

        // RoPE
        .def_property_readonly_static("rope_theta",
            [](py::object) { return gptoss::ModelConfig::rope_theta; },
            "RoPE base frequency (150000).")
        .def_property_readonly_static("rope_max_pos",
            [](py::object) { return gptoss::ModelConfig::rope_max_pos; },
            "Maximum sequence length (131072).")

        // Attention
        .def_property_readonly_static("sliding_window_size",
            [](py::object) { return gptoss::ModelConfig::sliding_window_size; },
            "Sliding window attention size (128).")
        .def_property_readonly_static("full_attention_interval",
            [](py::object) { return gptoss::ModelConfig::full_attention_interval; },
            "Full attention every N layers (4).")
        .def_property_readonly_static("gqa_ratio",
            [](py::object) { return gptoss::ModelConfig::gqa_ratio; },
            "GQA ratio: query heads / KV heads (8).")

        // KV cache
        .def_property_readonly_static("kv_block_size",
            [](py::object) { return gptoss::ModelConfig::kv_block_size; },
            "Tokens per KV cache block (16).")

        // Parallelism
        .def_property_readonly_static("world_size",
            [](py::object) { return gptoss::ModelConfig::world_size; },
            "Number of GPUs (2).")

        // Normalization
        .def_property_readonly_static("rms_norm_eps",
            [](py::object) { return gptoss::ModelConfig::rms_norm_eps; },
            "RMSNorm epsilon (1e-5).")

        // Methods
        .def_static("is_full_attention",
            &gptoss::ModelConfig::is_full_attention,
            py::arg("layer"),
            "Returns True if the given layer uses full attention (not sliding window).")
        .def_static("get_window_size",
            &gptoss::ModelConfig::get_window_size,
            py::arg("layer"),
            "Returns the sliding window size for a layer (0 = full attention).")

        .def("__repr__",
            [](const gptoss::ModelConfig&) {
                return "<gptoss.ModelConfig "
                       "layers=36 hidden=2880 q_heads=64 kv_heads=8 "
                       "experts=128 active=4 vocab=201088>";
            });

    // ------------------------------------------------------------------
    // Profiler (available when built with -DGPTOSS_ENABLE_PROFILE=ON)
    // ------------------------------------------------------------------
#ifdef GPTOSS_PROFILE
    m.def("profiler_save_trace", [](const std::string& path) {
        gptoss::profiler_save_chrome_trace(path.c_str());
    }, py::arg("path"), "Save kernel trace to Chrome Trace Format JSON.");

    m.def("profiler_save_summary", [](const std::string& path) {
        gptoss::profiler_save_summary_json(path.c_str());
    }, py::arg("path"), "Save aggregated kernel timing summary to JSON.");

    m.def("profiler_reset", []() {
        gptoss::profiler_reset();
    }, "Clear all profiler records.");

    m.def("profiler_enabled", []() { return true; },
          "Returns True if profiling is compiled in.");
#else
    m.def("profiler_enabled", []() { return false; },
          "Returns True if profiling is compiled in.");
#endif

    // ------------------------------------------------------------------
    // Module-level attributes
    // ------------------------------------------------------------------
    m.attr("__version__") = "0.1.0";
    m.attr("MODEL_NAME")  = "gpt-oss-120b";
}

// -----------------------------------------------------------------------
// C-linkage bridge implementation
//
// These functions are compiled by hipcc (from engine.cu or a separate bridge
// file) and linked with this pybind11 module. They translate between the
// opaque void* handle and the actual InferenceEngine class.
//
// NOTE: In the real build, these would be in a separate .cu file compiled
// by hipcc. They are placed here for reference; the build system must
// ensure they are compiled with hipcc and linked into the final .so.
// -----------------------------------------------------------------------

// The actual implementations are provided by the HIP compilation unit.
// See engine.cu for the InferenceEngine class.
// The build system compiles engine.cu with hipcc, bindings.cpp with g++/clang,
// and links them together with pybind11.
//
// The bridge functions (gptoss_engine_*) must be defined in a .cu file:
//
//   #include "engine.cu"
//
//   extern "C" {
//
//   void* gptoss_engine_create() {
//       return new gptoss::InferenceEngine();
//   }
//
//   void gptoss_engine_destroy(void* engine) {
//       delete static_cast<gptoss::InferenceEngine*>(engine);
//   }
//
//   void gptoss_engine_init(void* engine, const char* weights_dir) {
//       static_cast<gptoss::InferenceEngine*>(engine)->init(weights_dir);
//   }
//
//   void gptoss_engine_shutdown(void* engine) {
//       static_cast<gptoss::InferenceEngine*>(engine)->shutdown();
//   }
//
//   int gptoss_engine_generate(void* engine,
//                              const int32_t* prompt_tokens,
//                              int prompt_len,
//                              int max_tokens,
//                              float temperature,
//                              float top_p,
//                              int32_t* output_tokens,
//                              int output_capacity) {
//       auto* eng = static_cast<gptoss::InferenceEngine*>(engine);
//       std::vector<int32_t> prompt(prompt_tokens, prompt_tokens + prompt_len);
//       auto result = eng->generate(prompt, max_tokens, temperature, top_p);
//       int n = std::min(static_cast<int>(result.size()), output_capacity);
//       std::memcpy(output_tokens, result.data(), n * sizeof(int32_t));
//       return n;
//   }
//
//   int gptoss_engine_is_initialized(void* engine) {
//       return static_cast<gptoss::InferenceEngine*>(engine)->is_initialized()
//              ? 1 : 0;
//   }
//
//   } // extern "C"
