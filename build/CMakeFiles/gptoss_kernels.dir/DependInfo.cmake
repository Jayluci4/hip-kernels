
# Consider dependencies only in project.
set(CMAKE_DEPENDS_IN_PROJECT_ONLY OFF)

# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "HIP"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_HIP
  "/root/diagnose-kernels-gpt/hip-kernels/src/kernels/flash_attention.cu" "/root/diagnose-kernels-gpt/hip-kernels/build/CMakeFiles/gptoss_kernels.dir/src/kernels/flash_attention.cu.o"
  "/root/diagnose-kernels-gpt/hip-kernels/src/kernels/moe_combine.cu" "/root/diagnose-kernels-gpt/hip-kernels/build/CMakeFiles/gptoss_kernels.dir/src/kernels/moe_combine.cu.o"
  "/root/diagnose-kernels-gpt/hip-kernels/src/kernels/moe_permute.cu" "/root/diagnose-kernels-gpt/hip-kernels/build/CMakeFiles/gptoss_kernels.dir/src/kernels/moe_permute.cu.o"
  "/root/diagnose-kernels-gpt/hip-kernels/src/kernels/mxfp4_dequant.cu" "/root/diagnose-kernels-gpt/hip-kernels/build/CMakeFiles/gptoss_kernels.dir/src/kernels/mxfp4_dequant.cu.o"
  "/root/diagnose-kernels-gpt/hip-kernels/src/kernels/paged_attention.cu" "/root/diagnose-kernels-gpt/hip-kernels/build/CMakeFiles/gptoss_kernels.dir/src/kernels/paged_attention.cu.o"
  "/root/diagnose-kernels-gpt/hip-kernels/src/kernels/rmsnorm.cu" "/root/diagnose-kernels-gpt/hip-kernels/build/CMakeFiles/gptoss_kernels.dir/src/kernels/rmsnorm.cu.o"
  "/root/diagnose-kernels-gpt/hip-kernels/src/kernels/rope.cu" "/root/diagnose-kernels-gpt/hip-kernels/build/CMakeFiles/gptoss_kernels.dir/src/kernels/rope.cu.o"
  "/root/diagnose-kernels-gpt/hip-kernels/src/kernels/swiglu.cu" "/root/diagnose-kernels-gpt/hip-kernels/build/CMakeFiles/gptoss_kernels.dir/src/kernels/swiglu.cu.o"
  "/root/diagnose-kernels-gpt/hip-kernels/src/kernels/topk_softmax.cu" "/root/diagnose-kernels-gpt/hip-kernels/build/CMakeFiles/gptoss_kernels.dir/src/kernels/topk_softmax.cu.o"
  )
set(CMAKE_HIP_COMPILER_ID "Clang")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_HIP
  "HIPBLASLT_USE_ROCROLLER"
  "USE_PROF_API=1"
  "__HIP_PLATFORM_AMD__=1"
  "__HIP_ROCclr__=1"
  )

# The include file search paths:
set(CMAKE_HIP_TARGET_INCLUDE_PATH
  "/root/diagnose-kernels-gpt/hip-kernels/include"
  )

# The set of dependency files which are needed:
set(CMAKE_DEPENDS_DEPENDENCY_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_LINKED_INFO_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_FORWARD_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
