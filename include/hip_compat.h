#pragma once
// HIP compatibility layer for CUDA→HIP port of GPT-OSS-120B
// Provides type aliases, warp size constants, and missing intrinsic replacements

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// ============================================================================
// BF16 type aliases — map NVIDIA names to HIP names (ROCm 7.1)
// ============================================================================
using __nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;

// ============================================================================
// Warp (wavefront) size — MI300X CDNA3 uses wavefront64
// ============================================================================
static constexpr int HIP_WARP_SIZE = 64;

// ============================================================================
// BF16 conversion intrinsics — ROCm 7.1 already provides:
//   __bfloat162float, __float2bfloat16, __low2bfloat16,
//   __high2bfloat16, __halves2bfloat162
// Do NOT redefine them here.
// ============================================================================

// ============================================================================
// Warp shuffle replacements (HIP has no sync mask parameter)
// ============================================================================
static __device__ __forceinline__ float __shfl_xor_sync(unsigned /*mask*/, float val, int lane_mask) {
    return __shfl_xor(val, lane_mask);
}

static __device__ __forceinline__ int __shfl_xor_sync(unsigned /*mask*/, int val, int lane_mask) {
    return __shfl_xor(val, lane_mask);
}

static __device__ __forceinline__ float __shfl_sync(unsigned /*mask*/, float val, int src_lane) {
    return __shfl(val, src_lane);
}

static __device__ __forceinline__ int __shfl_sync(unsigned /*mask*/, int val, int src_lane) {
    return __shfl(val, src_lane);
}

// ============================================================================
// __ldg replacement — HIP doesn't have separate texture cache; use regular load
// ============================================================================
template <typename T>
static __device__ __forceinline__ T __ldg(const T* ptr) {
    return *ptr;
}

// ============================================================================
// Bit reinterpret helpers — ROCm 7.1 already provides:
//   __int_as_float, __float_as_uint
// Do NOT redefine them here.
// ============================================================================

// ============================================================================
// Pipeline async replacement — HIP doesn't have __pipeline_*
// Use synchronous copies instead. This is correct but slower.
// For correctness debugging, this is fine.
// ============================================================================
static __device__ __forceinline__ void __pipeline_commit() {}
static __device__ __forceinline__ void __pipeline_wait_prior(int) { __syncwarp(); }

static __device__ __forceinline__ void pipeline_memcpy_sync(void* dst, const void* src, size_t sz) {
    const char* s = static_cast<const char*>(src);
    char* d = static_cast<char*>(dst);
    size_t i = 0;
    for (; i + 16 <= sz; i += 16) {
        *reinterpret_cast<int4*>(d + i) = *reinterpret_cast<const int4*>(s + i);
    }
    for (; i < sz; i++) {
        d[i] = s[i];
    }
}

// ============================================================================
// CUDA API → HIP API name mappings (host-side)
// ============================================================================
#ifndef cudaSuccess
#define cudaSuccess hipSuccess
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaError_t hipError_t
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// CUDA Graph API → HIP Graph API
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphDestroy hipGraphDestroy
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#endif
