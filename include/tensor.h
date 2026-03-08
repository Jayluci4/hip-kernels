#pragma once

#include <cstdint>
#include <cstddef>
#include <cassert>

#include "hip_compat.h"

namespace gptoss {

enum class DType : uint8_t {
    FP32,
    BF16,
    FP16,
    INT8,
    UINT8,
    INT32,
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FP32:  return 4;
        case DType::BF16:  return 2;
        case DType::FP16:  return 2;
        case DType::INT8:  return 1;
        case DType::UINT8: return 1;
        case DType::INT32: return 4;
        default: return 0;
    }
}

static constexpr int MAX_DIMS = 4;

struct Tensor {
    void* data = nullptr;
    int64_t shape[MAX_DIMS] = {0};
    int64_t stride[MAX_DIMS] = {0};
    int ndim = 0;
    DType dtype = DType::FP32;
    int device_id = -1;

    Tensor() = default;

    Tensor(void* data, DType dtype, std::initializer_list<int64_t> dims, int device = 0)
        : data(data), dtype(dtype), device_id(device) {
        ndim = static_cast<int>(dims.size());
        assert(ndim <= MAX_DIMS);
        int i = 0;
        for (auto d : dims) shape[i++] = d;
        compute_strides();
    }

    void compute_strides() {
        if (ndim == 0) return;
        stride[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
    }

    int64_t numel() const {
        int64_t n = 1;
        for (int i = 0; i < ndim; ++i) n *= shape[i];
        return n;
    }

    size_t nbytes() const {
        return static_cast<size_t>(numel()) * dtype_size(dtype);
    }

    template <typename T>
    T* ptr() const { return static_cast<T*>(data); }

    __hip_bfloat16* bf16_ptr() const { return static_cast<__hip_bfloat16*>(data); }
    float* fp32_ptr() const { return static_cast<float*>(data); }
    uint8_t* u8_ptr() const { return static_cast<uint8_t*>(data); }
};

inline Tensor allocate_tensor(DType dtype, std::initializer_list<int64_t> dims, int device = 0) {
    Tensor t;
    t.dtype = dtype;
    t.ndim = static_cast<int>(dims.size());
    int i = 0;
    for (auto d : dims) t.shape[i++] = d;
    t.compute_strides();
    t.device_id = device;
    hipSetDevice(device);
    hipMalloc(&t.data, t.nbytes());
    return t;
}

inline void free_tensor(Tensor& t) {
    if (t.data) {
        hipSetDevice(t.device_id);
        hipFree(t.data);
        t.data = nullptr;
    }
}

} // namespace gptoss
