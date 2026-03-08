#pragma once

#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <mutex>

#include "hip_compat.h"

#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <vector>

namespace gptoss {

#define CUDA_CHECK(expr)                                                       \
    do {                                                                        \
        hipError_t err = (expr);                                                \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    hipGetErrorString(err));                                     \
            abort();                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(expr)                                                     \
    do {                                                                        \
        hipblasStatus_t status = (expr);                                        \
        if (status != HIPBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "hipBLAS error at %s:%d: %d\n", __FILE__, __LINE__,\
                    static_cast<int>(status));                                   \
            abort();                                                            \
        }                                                                       \
    } while (0)

// Ceiling division
inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// Round up to next power of two
inline int m_bucket(int m) {
    if (m <= 1) return 1;
    int b = 1;
    while (b < m) b <<= 1;
    return b;
}

// Stream wrapper for RAII
struct CudaStream {
    hipStream_t stream = nullptr;
    int device_id = 0;

    CudaStream() = default;

    explicit CudaStream(int device, bool high_priority = false) : device_id(device) {
        hipSetDevice(device);
        if (high_priority) {
            int low, high;
            hipDeviceGetStreamPriorityRange(&low, &high);
            CUDA_CHECK(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, high));
        } else {
            CUDA_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        }
    }

    ~CudaStream() {
        if (stream) hipStreamDestroy(stream);
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& o) noexcept : stream(o.stream), device_id(o.device_id) { o.stream = nullptr; }
    CudaStream& operator=(CudaStream&& o) noexcept {
        if (this != &o) { if (stream) hipStreamDestroy(stream); stream = o.stream; device_id = o.device_id; o.stream = nullptr; }
        return *this;
    }

    void sync() const { CUDA_CHECK(hipStreamSynchronize(stream)); }
    operator hipStream_t() const { return stream; }
};

// Event wrapper
struct CudaEvent {
    hipEvent_t event = nullptr;

    CudaEvent() { CUDA_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming)); }
    ~CudaEvent() { if (event) hipEventDestroy(event); }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& o) noexcept : event(o.event) { o.event = nullptr; }

    void record(hipStream_t stream) { CUDA_CHECK(hipEventRecord(event, stream)); }
    void wait(hipStream_t stream) { CUDA_CHECK(hipStreamWaitEvent(stream, event, 0)); }
    operator hipEvent_t() const { return event; }
};

// ---------------------------------------------------------------------------
// hipBLASLt algorithm cache key
// ---------------------------------------------------------------------------
struct LtAlgoKey {
    int mb, n, k;
    bool trans;
    bool operator==(const LtAlgoKey& o) const {
        return mb == o.mb && n == o.n && k == o.k && trans == o.trans;
    }
};

struct LtAlgoKeyHash {
    size_t operator()(const LtAlgoKey& key) const {
        size_t h = static_cast<size_t>(key.mb);
        h ^= static_cast<size_t>(key.n) << 16;
        h ^= static_cast<size_t>(key.k) << 32;
        if (key.trans) h ^= 0x1ULL << 48;
        return h;
    }
};

using LtAlgoCache = std::unordered_map<LtAlgoKey, hipblasLtMatmulAlgo_t, LtAlgoKeyHash>;

// ---------------------------------------------------------------------------
// hipBLASLt layout cache key
// ---------------------------------------------------------------------------
struct LayoutKey {
    int64_t rows, cols, ld;
    bool operator==(const LayoutKey& o) const {
        return rows == o.rows && cols == o.cols && ld == o.ld;
    }
};

struct LayoutKeyHash {
    size_t operator()(const LayoutKey& k) const {
        return static_cast<size_t>(k.rows) ^ (static_cast<size_t>(k.cols) << 20)
               ^ (static_cast<size_t>(k.ld) << 40);
    }
};

using LayoutCache = std::unordered_map<LayoutKey, hipblasLtMatrixLayout_t, LayoutKeyHash>;

// hipBLAS handle wrapper
struct CublasHandle {
    hipblasHandle_t handle = nullptr;
    hipblasLtHandle_t lt_handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_size = 32 * 1024 * 1024; // 32MB
    hipStream_t bound_stream = nullptr;

    LtAlgoCache* algo_cache = nullptr;
    LayoutCache* layout_cache = nullptr;

    hipblasLtMatmulDesc_t cached_matmul_desc_ = nullptr;
    hipblasLtMatmulDesc_t cached_matmul_desc_transA_ = nullptr;

    CublasHandle() = default;

    void init(int device, hipStream_t stream) {
        hipSetDevice(device);
        CUBLAS_CHECK(hipblasCreate(&handle));
        CUBLAS_CHECK(hipblasSetStream(handle, stream));
        CUBLAS_CHECK(hipblasLtCreate(&lt_handle));
        CUDA_CHECK(hipMalloc(&workspace, workspace_size));
        bound_stream = stream;
        algo_cache = new LtAlgoCache();
        layout_cache = new LayoutCache();

        CUBLAS_CHECK(hipblasLtMatmulDescCreate(
            &cached_matmul_desc_, HIPBLAS_COMPUTE_32F, HIP_R_32F));

        CUBLAS_CHECK(hipblasLtMatmulDescCreate(
            &cached_matmul_desc_transA_, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        hipblasOperation_t transA_op = HIPBLAS_OP_T;
        CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
            cached_matmul_desc_transA_, HIPBLASLT_MATMUL_DESC_TRANSA,
            &transA_op, sizeof(transA_op)));
    }

    hipblasLtMatrixLayout_t get_or_create_layout(int64_t rows, int64_t cols, int64_t ld) const {
        LayoutKey key{rows, cols, ld};
        auto it = layout_cache->find(key);
        if (it != layout_cache->end()) return it->second;
        hipblasLtMatrixLayout_t desc;
        CUBLAS_CHECK(hipblasLtMatrixLayoutCreate(&desc, HIP_R_16BF, rows, cols, ld));
        layout_cache->emplace(key, desc);
        return desc;
    }

    ~CublasHandle() {
        if (cached_matmul_desc_transA_) hipblasLtMatmulDescDestroy(cached_matmul_desc_transA_);
        if (cached_matmul_desc_) hipblasLtMatmulDescDestroy(cached_matmul_desc_);
        if (layout_cache) {
            for (auto& [key, desc] : *layout_cache)
                hipblasLtMatrixLayoutDestroy(desc);
            delete layout_cache;
            layout_cache = nullptr;
        }
        if (handle) hipblasDestroy(handle);
        if (lt_handle) hipblasLtDestroy(lt_handle);
        if (workspace) hipFree(workspace);
        delete algo_cache;
        algo_cache = nullptr;
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    // Legacy BF16 GEMM
    void gemm_bf16(
        const __hip_bfloat16* A, const __hip_bfloat16* B, __hip_bfloat16* C,
        int M, int N, int K,
        float alpha = 1.0f, float beta = 0.0f) const
    {
        CUBLAS_CHECK(hipblasGemmEx(
            handle,
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            N, M, K,
            &alpha,
            B, HIP_R_16BF, N,
            A, HIP_R_16BF, K,
            &beta,
            C, HIP_R_16BF, N,
            HIPBLAS_COMPUTE_32F,
            HIPBLAS_GEMM_DEFAULT));
    }

    // hipBLASLt GEMM
    void gemm_bf16_lt(
        const __hip_bfloat16* A, const __hip_bfloat16* B, __hip_bfloat16* C,
        int M, int N, int K,
        hipStream_t stream,
        float alpha = 1.0f, float beta = 0.0f,
        const __hip_bfloat16* bias = nullptr,
        bool transB = false) const
    {
        if (M == 0) return;

        hipblasLtMatmulDesc_t matmul_desc = transB ? cached_matmul_desc_transA_
                                                    : cached_matmul_desc_;
        bool own_desc = false;

        if (bias) {
            CUBLAS_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
            own_desc = true;
            if (transB) {
                hipblasOperation_t op = HIPBLAS_OP_T;
                CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
                    matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)));
            }
            hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
            CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
                matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
            CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
                matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
            hipDataType bias_type = HIP_R_16BF;
            CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
                matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));
            beta = 0.0f;
        }

        hipblasLtMatrixLayout_t a_lt_desc = transB
            ? get_or_create_layout(K, N, K)
            : get_or_create_layout(N, K, N);
        hipblasLtMatrixLayout_t b_lt_desc = get_or_create_layout(K, M, K);
        hipblasLtMatrixLayout_t c_lt_desc = get_or_create_layout(N, M, N);

        LtAlgoKey cache_key{m_bucket(M), N, K, transB};
        const hipblasLtMatmulAlgo_t* algo_ptr = nullptr;
        hipblasLtMatmulAlgo_t found_algo;

        auto it = algo_cache->find(cache_key);
        if (it != algo_cache->end()) {
            algo_ptr = &it->second;
        } else {
            hipblasLtMatmulPreference_t pref;
            CUBLAS_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
            CUBLAS_CHECK(hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_size, sizeof(workspace_size)));

            hipblasLtMatmulHeuristicResult_t heuristic;
            int returned_count = 0;
            CUBLAS_CHECK(hipblasLtMatmulAlgoGetHeuristic(
                lt_handle, matmul_desc,
                a_lt_desc, b_lt_desc, c_lt_desc, c_lt_desc,
                pref, 1, &heuristic, &returned_count));

            if (returned_count > 0) {
                found_algo = heuristic.algo;
                algo_cache->emplace(cache_key, found_algo);
                algo_ptr = &(*algo_cache)[cache_key];
            }

            hipblasLtMatmulPreferenceDestroy(pref);
        }

        CUBLAS_CHECK(hipblasLtMatmul(
            lt_handle, matmul_desc,
            &alpha,
            B, a_lt_desc,
            A, b_lt_desc,
            &beta,
            C, c_lt_desc,
            C, c_lt_desc,
            algo_ptr,
            workspace, workspace_size,
            stream));

        if (own_desc) hipblasLtMatmulDescDestroy(matmul_desc);
    }

    // Multi-GEMM with shared descriptors
    void gemm_bf16_lt_multi(
        const __hip_bfloat16* const* A_array,
        const __hip_bfloat16* const* B_array,
        __hip_bfloat16* const* C_array,
        const int* M_array,
        int N, int K,
        int count,
        hipStream_t stream,
        float alpha = 1.0f, float beta = 0.0f,
        bool transB = false) const
    {
        if (count == 0) return;

        hipblasLtMatrixLayout_t b_lt_desc = transB
            ? get_or_create_layout(K, N, K)
            : get_or_create_layout(N, K, N);

        hipblasLtMatmulDesc_t desc = transB ? cached_matmul_desc_transA_
                                            : cached_matmul_desc_;

        int max_m = 0;
        for (int i = 0; i < count; i++) {
            if (M_array[i] > max_m) max_m = M_array[i];
        }
        if (max_m == 0) return;

        LtAlgoKey cache_key{m_bucket(max_m), N, K, transB};
        const hipblasLtMatmulAlgo_t* algo_ptr = nullptr;
        hipblasLtMatmulAlgo_t found_algo;

        auto it = algo_cache->find(cache_key);
        if (it != algo_cache->end()) {
            algo_ptr = &it->second;
        } else {
            hipblasLtMatrixLayout_t tmp_a = get_or_create_layout(K, max_m, K);
            hipblasLtMatrixLayout_t tmp_c = get_or_create_layout(N, max_m, N);

            hipblasLtMatmulPreference_t pref;
            CUBLAS_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
            CUBLAS_CHECK(hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_size, sizeof(workspace_size)));

            hipblasLtMatmulHeuristicResult_t heuristic;
            int returned = 0;
            CUBLAS_CHECK(hipblasLtMatmulAlgoGetHeuristic(
                lt_handle, desc,
                b_lt_desc, tmp_a, tmp_c, tmp_c,
                pref, 1, &heuristic, &returned));

            if (returned > 0) {
                found_algo = heuristic.algo;
                algo_cache->emplace(cache_key, found_algo);
                algo_ptr = &(*algo_cache)[cache_key];
            }

            hipblasLtMatmulPreferenceDestroy(pref);
        }

        for (int i = 0; i < count; i++) {
            if (M_array[i] == 0) continue;

            hipblasLtMatrixLayout_t a_lt = get_or_create_layout(K, M_array[i], K);
            hipblasLtMatrixLayout_t c_lt = get_or_create_layout(N, M_array[i], N);

            CUBLAS_CHECK(hipblasLtMatmul(
                lt_handle, desc,
                &alpha,
                B_array[i], b_lt_desc,
                A_array[i], a_lt,
                &beta,
                C_array[i], c_lt,
                C_array[i], c_lt,
                algo_ptr,
                workspace, workspace_size,
                stream));
        }
    }

    // Grouped GEMM: single kernel launch for all experts.
    // C[i] = A[i] * B[i]^T  (each with different M, shared N and K)
    void gemm_bf16_grouped(
        const __hip_bfloat16* const* A_array,
        const __hip_bfloat16* const* B_array,
        __hip_bfloat16* const* C_array,
        const int* M_array,
        int N, int K,
        int count,
        hipStream_t stream,
        float alpha_val = 1.0f, float beta_val = 0.0f,
        bool transB = false) const
    {
        if (count == 0) return;

        // Filter out zero-M entries
        int actual = 0;
        for (int i = 0; i < count; i++)
            if (M_array[i] > 0) actual++;
        if (actual == 0) return;

        // For single GEMM, fall through to individual call (less overhead)
        if (actual == 1) {
            gemm_bf16_lt_multi(A_array, B_array, C_array, M_array,
                               N, K, count, stream, alpha_val, beta_val, transB);
            return;
        }

        hipblasOperation_t opA = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        hipblasOperation_t opB = HIPBLAS_OP_N;

        hipblaslt_ext::GroupedGemm groupedGemm(
            lt_handle,
            opA, opB,
            HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF,
            HIPBLAS_COMPUTE_32F);

        std::vector<int64_t> m_vec, n_vec, k_vec, batch_vec;
        std::vector<int64_t> lda_vec, ldb_vec, ldc_vec, ldd_vec;
        std::vector<int64_t> strideA_vec, strideB_vec, strideC_vec, strideD_vec;
        std::vector<hipblaslt_ext::GemmEpilogue> epilogue_vec;
        std::vector<hipblaslt_ext::GemmInputs> inputs_vec;

        for (int i = 0; i < count; i++) {
            if (M_array[i] == 0) continue;

            // hipBLASLt uses column-major: C = B * A where our row-major C = A * B^T
            // In col-major terms: m=N, n=M, k=K
            // A (our B, the weight): if transB, layout [K, N] with ld=K; else [N, K] with ld=N
            // B (our A, activations): [K, M] with ld=K
            // C: [N, M] with ld=N
            int64_t cm = N;
            int64_t cn = M_array[i];
            int64_t ck = K;

            m_vec.push_back(cm);
            n_vec.push_back(cn);
            k_vec.push_back(ck);
            batch_vec.push_back(1);
            lda_vec.push_back(transB ? ck : cm);  // A in col-major = our B
            ldb_vec.push_back(ck);                 // B in col-major = our A
            ldc_vec.push_back(cm);
            ldd_vec.push_back(cm);
            strideA_vec.push_back(0);
            strideB_vec.push_back(0);
            strideC_vec.push_back(0);
            strideD_vec.push_back(0);

            hipblaslt_ext::GemmEpilogue ep;
            epilogue_vec.push_back(std::move(ep));

            hipblaslt_ext::GemmInputs inp;
            inp.setA(B_array[i]);  // col-major A = our B (weights)
            inp.setB(A_array[i]);  // col-major B = our A (activations)
            inp.setC(C_array[i]);
            inp.setD(C_array[i]);
            inp.setAlpha(&alpha_val);
            inp.setBeta(&beta_val);
            inputs_vec.push_back(std::move(inp));
        }

        hipblaslt_ext::GemmProblemType problemType(
            opA, opB,
            HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF,
            HIPBLAS_COMPUTE_32F);

        CUBLAS_CHECK(groupedGemm.setProblem(
            m_vec, n_vec, k_vec, batch_vec,
            lda_vec, ldb_vec, ldc_vec, ldd_vec,
            strideA_vec, strideB_vec, strideC_vec, strideD_vec,
            epilogue_vec, inputs_vec, problemType));

        // Find algorithm
        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(workspace_size);

        std::vector<hipblasLtMatmulHeuristicResult_t> heuristics;
        CUBLAS_CHECK(groupedGemm.algoGetHeuristic(1, pref, heuristics));

        if (heuristics.empty()) {
            fprintf(stderr, "hipBLASLt GroupedGemm: no algorithm found!\n");
            // Fallback to individual GEMMs
            gemm_bf16_lt_multi(A_array, B_array, C_array, M_array,
                               N, K, count, stream, alpha_val, beta_val, transB);
            return;
        }

        CUBLAS_CHECK(groupedGemm.initialize(heuristics[0].algo, workspace, false, stream));
        CUBLAS_CHECK(groupedGemm.run(stream));
    }
};

} // namespace gptoss
