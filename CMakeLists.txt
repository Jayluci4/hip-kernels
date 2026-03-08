cmake_minimum_required(VERSION 3.24)
project(gpt_oss_engine_hip LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# MI300X = gfx942
set(CMAKE_HIP_ARCHITECTURES gfx942)
set(GPU_TARGETS gfx942)

# Profiling (pass -DGPTOSS_ENABLE_PROFILE=ON to enable)
option(GPTOSS_ENABLE_PROFILE "Enable kernel profiling hooks" OFF)

# ROCm
list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(hip REQUIRED)
find_package(hipblas REQUIRED)
find_package(hipblaslt REQUIRED)

# RCCL
find_library(RCCL_LIB rccl HINTS /opt/rocm/lib)
find_path(RCCL_INCLUDE rccl/rccl.h HINTS /opt/rocm/include)

# pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# Kernel library
add_library(gptoss_kernels STATIC
    src/kernels/gemv_bf16.cu
    src/kernels/fused_mxfp4_gemv.cu
    src/kernels/mxfp4_dequant.cu
    src/kernels/rmsnorm.cu
    src/kernels/rope.cu
    src/kernels/swiglu.cu
    src/kernels/flash_attention.cu
    src/kernels/paged_attention.cu
    src/kernels/topk_softmax.cu
    src/kernels/moe_permute.cu
    src/kernels/moe_combine.cu
)
set_source_files_properties(
    src/kernels/gemv_bf16.cu
    src/kernels/fused_mxfp4_gemv.cu
    src/kernels/mxfp4_dequant.cu
    src/kernels/rmsnorm.cu
    src/kernels/rope.cu
    src/kernels/swiglu.cu
    src/kernels/flash_attention.cu
    src/kernels/paged_attention.cu
    src/kernels/topk_softmax.cu
    src/kernels/moe_permute.cu
    src/kernels/moe_combine.cu
    PROPERTIES LANGUAGE HIP
)
target_include_directories(gptoss_kernels PUBLIC include ${RCCL_INCLUDE})
target_link_libraries(gptoss_kernels PUBLIC
    hip::host hip::device roc::hipblas roc::hipblaslt ${RCCL_LIB}
)

# Communication library
add_library(gptoss_comm STATIC
    src/comm/nccl_manager.cu
    src/comm/ep_comm.cu
    src/comm/buffer_manager.cu
)
set_source_files_properties(
    src/comm/nccl_manager.cu
    src/comm/ep_comm.cu
    src/comm/buffer_manager.cu
    PROPERTIES LANGUAGE HIP
)
target_include_directories(gptoss_comm PUBLIC include ${RCCL_INCLUDE})
target_link_libraries(gptoss_comm PUBLIC gptoss_kernels)

# Model library
add_library(gptoss_model STATIC
    src/model/transformer.cu
    src/model/moe_layer.cu
    src/model/attention.cu
)
set_source_files_properties(
    src/model/transformer.cu
    src/model/moe_layer.cu
    src/model/attention.cu
    PROPERTIES LANGUAGE HIP
)
target_include_directories(gptoss_model PUBLIC include ${RCCL_INCLUDE})
target_link_libraries(gptoss_model PUBLIC gptoss_kernels gptoss_comm)

# Engine library
add_library(gptoss_engine STATIC
    src/engine/engine.cu
    src/engine/bridge.cu
    src/engine/profiler.cu
    src/memory/kv_cache_manager.cu
    src/memory/weight_loader.cu
)
set_source_files_properties(
    src/engine/engine.cu
    src/engine/bridge.cu
    src/engine/profiler.cu
    src/memory/kv_cache_manager.cu
    src/memory/weight_loader.cu
    PROPERTIES LANGUAGE HIP
)
target_include_directories(gptoss_engine PUBLIC include ${RCCL_INCLUDE})
target_link_libraries(gptoss_engine PUBLIC gptoss_model)

# Python module
pybind11_add_module(gptoss_python src/python/bindings.cpp)
set_source_files_properties(src/python/bindings.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(gptoss_python PRIVATE gptoss_engine)
target_include_directories(gptoss_python PRIVATE include ${RCCL_INCLUDE})
set_target_properties(gptoss_python PROPERTIES OUTPUT_NAME "gptoss")

# Propagate GPTOSS_PROFILE to all targets when enabled
if(GPTOSS_ENABLE_PROFILE)
    target_compile_definitions(gptoss_kernels PUBLIC GPTOSS_PROFILE)
    target_compile_definitions(gptoss_comm PUBLIC GPTOSS_PROFILE)
    target_compile_definitions(gptoss_model PUBLIC GPTOSS_PROFILE)
    target_compile_definitions(gptoss_engine PUBLIC GPTOSS_PROFILE)
    target_compile_definitions(gptoss_python PRIVATE GPTOSS_PROFILE)
endif()
