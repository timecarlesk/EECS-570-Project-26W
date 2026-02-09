cmake_minimum_required(VERSION 3.18)
project(cp_async_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
endif()

find_package(CUDAToolkit REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/cuda/include)

set(COMMON_CUDA_FLAGS
  --expt-relaxed-constexpr
  --use_fast_math
)

add_executable(benchmark_gemm cuda/src/benchmark_gemm.cu)
target_compile_options(benchmark_gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
)
target_link_libraries(benchmark_gemm PRIVATE CUDA::cublas)

add_executable(benchmark_stencil cuda/src/benchmark_stencil.cu)
target_compile_options(benchmark_stencil PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
)

add_executable(microbench_pointer_chase cuda/src/microbench_pointer_chase.cu)
target_compile_options(microbench_pointer_chase PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
)
