#pragma once

#include <cuda_runtime.h>

// Minimal cp.async wrappers with fallback paths for pre-Ampere architectures.

__device__ __forceinline__ void cp_async_cg_4(void* shared_dst, const void* global_src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const unsigned shared_addr = static_cast<unsigned>(__cvta_generic_to_shared(shared_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 4;\n" : : "r"(shared_addr), "l"(global_src));
#else
  *reinterpret_cast<float*>(shared_dst) = *reinterpret_cast<const float*>(global_src);
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n" : :);
#endif
}

template <int kPendingGroups>
__device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group %0;\n" : : "n"(kPendingGroups));
#endif
}
