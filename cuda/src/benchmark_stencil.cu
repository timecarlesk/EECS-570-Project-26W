#include "benchmark_common.cuh"
#include "cp_async_utils.cuh"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

constexpr int kRadius = 2;
constexpr int kBlockThreads = 256;

template <int BLOCK_THREADS>
__device__ __forceinline__ void load_segment_sync(
    float* stage,
    const float* in,
    int n,
    int segment_start,
    int tid) {
  const int center_idx = segment_start + tid;
  stage[kRadius + tid] = (center_idx < n) ? in[center_idx] : 0.0f;

  if (tid < kRadius) {
    const int left_idx = segment_start + tid - kRadius;
    const int right_idx = segment_start + BLOCK_THREADS + tid;
    stage[tid] = (left_idx >= 0 && left_idx < n) ? in[left_idx] : 0.0f;
    stage[kRadius + BLOCK_THREADS + tid] = (right_idx < n) ? in[right_idx] : 0.0f;
  }
}

template <int BLOCK_THREADS>
__device__ __forceinline__ void load_segment_async(
    float* stage,
    const float* in,
    int n,
    int segment_start,
    int tid) {
  const int center_idx = segment_start + tid;
  float* center_dst = &stage[kRadius + tid];
  if (center_idx < n) {
    cp_async_cg_4(center_dst, &in[center_idx]);
  } else {
    *center_dst = 0.0f;
  }

  if (tid < kRadius) {
    const int left_idx = segment_start + tid - kRadius;
    const int right_idx = segment_start + BLOCK_THREADS + tid;

    float* left_dst = &stage[tid];
    float* right_dst = &stage[kRadius + BLOCK_THREADS + tid];

    if (left_idx >= 0 && left_idx < n) {
      cp_async_cg_4(left_dst, &in[left_idx]);
    } else {
      *left_dst = 0.0f;
    }

    if (right_idx < n) {
      cp_async_cg_4(right_dst, &in[right_idx]);
    } else {
      *right_dst = 0.0f;
    }
  }
}

__device__ __forceinline__ float stencil_compute(float left2, float left1, float center, float right1, float right2, int compute_iters) {
  float v = center;
  for (int i = 0; i < compute_iters; ++i) {
    v = 0.5f * v + 0.125f * (left1 + right1 + left2 + right2);
  }
  return v;
}

template <int BLOCK_THREADS>
__global__ void stencil_v1_kernel(const float* in, float* out, int n, int compute_iters) {
  extern __shared__ float shared[];
  const int tid = threadIdx.x;

  const int total_segments = ceil_div_int(n, BLOCK_THREADS);
  for (int seg = blockIdx.x; seg < total_segments; seg += gridDim.x) {
    const int segment_start = seg * BLOCK_THREADS;

    load_segment_sync<BLOCK_THREADS>(shared, in, n, segment_start, tid);
    __syncthreads();

    const int idx = segment_start + tid;
    if (idx < n) {
      if (idx < kRadius || idx >= n - kRadius) {
        out[idx] = in[idx];
      } else {
        const float l2 = shared[kRadius + tid - 2];
        const float l1 = shared[kRadius + tid - 1];
        const float c = shared[kRadius + tid];
        const float r1 = shared[kRadius + tid + 1];
        const float r2 = shared[kRadius + tid + 2];
        out[idx] = stencil_compute(l2, l1, c, r1, r2, compute_iters);
      }
    }

    __syncthreads();
  }
}

template <int BLOCK_THREADS, int STAGES>
__global__ void stencil_v3_kernel(const float* in, float* out, int n, int compute_iters) {
  static_assert(STAGES >= 2 && STAGES <= 4, "Supported STAGES: 2..4");

  extern __shared__ float shared[];
  constexpr int stage_span = BLOCK_THREADS + 2 * kRadius;

  const int tid = threadIdx.x;
  const int total_segments = ceil_div_int(n, BLOCK_THREADS);
  const int stride_segments = gridDim.x;
  const int first_seg = blockIdx.x;

  int block_iters = 0;
  for (int seg = first_seg; seg < total_segments; seg += stride_segments) {
    ++block_iters;
  }

  if (block_iters <= 0) {
    return;
  }

  const int preload = (STAGES - 1 < block_iters) ? (STAGES - 1) : block_iters;
  for (int i = 0; i < preload; ++i) {
    const int seg = first_seg + i * stride_segments;
    const int stage = i % STAGES;
    float* stage_ptr = shared + stage * stage_span;
    load_segment_async<BLOCK_THREADS>(stage_ptr, in, n, seg * BLOCK_THREADS, tid);
    cp_async_commit_group();
  }

  for (int i = 0; i < block_iters; ++i) {
    const int produce_i = i + STAGES - 1;
    if (produce_i < block_iters) {
      const int seg = first_seg + produce_i * stride_segments;
      const int stage = produce_i % STAGES;
      float* stage_ptr = shared + stage * stage_span;
      load_segment_async<BLOCK_THREADS>(stage_ptr, in, n, seg * BLOCK_THREADS, tid);
      cp_async_commit_group();
    }

    cp_async_wait_group<STAGES - 1>();
    __syncthreads();

    const int consume_seg = first_seg + i * stride_segments;
    const int consume_stage = i % STAGES;
    float* stage_ptr = shared + consume_stage * stage_span;

    const int idx = consume_seg * BLOCK_THREADS + tid;
    if (idx < n) {
      if (idx < kRadius || idx >= n - kRadius) {
        out[idx] = in[idx];
      } else {
        const float l2 = stage_ptr[kRadius + tid - 2];
        const float l1 = stage_ptr[kRadius + tid - 1];
        const float c = stage_ptr[kRadius + tid];
        const float r1 = stage_ptr[kRadius + tid + 1];
        const float r2 = stage_ptr[kRadius + tid + 2];
        out[idx] = stencil_compute(l2, l1, c, r1, r2, compute_iters);
      }
    }

    __syncthreads();
  }

  cp_async_wait_group<0>();
}

void fill_vector(std::vector<float>* vec) {
  for (size_t i = 0; i < vec->size(); ++i) {
    (*vec)[i] = static_cast<float>((static_cast<int>(i * 17) % 31) - 15) / 15.0f;
  }
}

float max_abs_diff(const std::vector<float>& ref, const std::vector<float>& out) {
  float max_diff = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float d = std::fabs(ref[i] - out[i]);
    if (d > max_diff) {
      max_diff = d;
    }
  }
  return max_diff;
}

void launch_stencil_v3(const float* d_in, float* d_out, int n, int compute_iters, int grid_blocks, int stage) {
  const dim3 block(kBlockThreads);
  const dim3 grid(grid_blocks);

  if (stage == 2) {
    const size_t smem = static_cast<size_t>(2 * (kBlockThreads + 2 * kRadius)) * sizeof(float);
    stencil_v3_kernel<kBlockThreads, 2><<<grid, block, smem>>>(d_in, d_out, n, compute_iters);
  } else if (stage == 3) {
    const size_t smem = static_cast<size_t>(3 * (kBlockThreads + 2 * kRadius)) * sizeof(float);
    stencil_v3_kernel<kBlockThreads, 3><<<grid, block, smem>>>(d_in, d_out, n, compute_iters);
  } else if (stage == 4) {
    const size_t smem = static_cast<size_t>(4 * (kBlockThreads + 2 * kRadius)) * sizeof(float);
    stencil_v3_kernel<kBlockThreads, 4><<<grid, block, smem>>>(d_in, d_out, n, compute_iters);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<int> default_lengths = {1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26};
  const std::vector<int> default_stage_values = {2, 3, 4};

  std::vector<int> lengths = parse_csv_ints(get_arg(argc, argv, "--lengths", ""));
  if (lengths.empty()) {
    lengths = default_lengths;
  }

  std::vector<int> stage_values = parse_csv_ints(get_arg(argc, argv, "--stages", ""));
  if (stage_values.empty()) {
    stage_values = default_stage_values;
  }

  const int warmup = get_arg_int(argc, argv, "--warmup", 3);
  const int iters = get_arg_int(argc, argv, "--iters", 10);
  const int check_max_n = get_arg_int(argc, argv, "--check-max-n", 1 << 22);
  const int compute_iters = get_arg_int(argc, argv, "--compute-iters", 4);
  const int grid_blocks = get_arg_int(argc, argv, "--grid-blocks", 120);
  const bool enable_check = !has_flag(argc, argv, "--no-check");
  const float atol = get_arg_float(argc, argv, "--atol", 1e-4f);
  const float rtol = get_arg_float(argc, argv, "--rtol", 1e-4f);

  const std::string output_path = get_arg(argc, argv, "--output", "outputs/stencil_raw.csv");
  const std::string gpu_name_arg = get_arg(argc, argv, "--gpu-name", "");
  const std::string gpu_name = gpu_name_arg.empty() ? detect_gpu_name() : gpu_name_arg;

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const bool supports_cp_async = (prop.major * 10 + prop.minor) >= 80;

  std::cout << "[Stencil] GPU=" << gpu_name << " compute_capability=" << prop.major << "." << prop.minor
            << " cp.async=" << (supports_cp_async ? "yes" : "no") << std::endl;

  write_csv_header_if_needed(
      output_path,
      "workload,gpu,problem_size,variant,stage,tile_size,time_ms,gflops,correct,max_abs_error");

  for (int n : lengths) {
    const size_t numel = static_cast<size_t>(n);
    const size_t bytes = numel * sizeof(float);

    std::vector<float> h_in(numel);
    std::vector<float> h_ref;
    std::vector<float> h_out;
    fill_vector(&h_in);

    float* d_in = nullptr;
    float* d_out = nullptr;
    float* d_ref = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_ref, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(kBlockThreads);
    const dim3 grid(grid_blocks);
    const size_t smem_v1 = static_cast<size_t>(kBlockThreads + 2 * kRadius) * sizeof(float);

    auto launch_v1 = [&]() {
      stencil_v1_kernel<kBlockThreads><<<grid, block, smem_v1>>>(d_in, d_out, n, compute_iters);
    };

    const double v1_ms = benchmark_ms(launch_v1, warmup, iters);
    CUDA_CHECK(cudaGetLastError());

    stencil_v1_kernel<kBlockThreads><<<grid, block, smem_v1>>>(d_in, d_ref, n, compute_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bool check_ok = true;
    float max_err = 0.0f;
    if (enable_check && n <= check_max_n) {
      h_ref.resize(numel);
      h_out.resize(numel);
      CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, bytes, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
      max_err = max_abs_diff(h_ref, h_out);
      check_ok = check_close_host(h_ref, h_out, atol, rtol, nullptr, nullptr, nullptr);
    }

    const double flops_per_elem = 7.0 * static_cast<double>(compute_iters);
    const double v1_gflops = (flops_per_elem * static_cast<double>(n)) / (v1_ms * 1.0e6);

    {
      std::ostringstream oss;
      oss << "stencil," << gpu_name << "," << n << ",V1,1," << kBlockThreads << ","
          << std::fixed << std::setprecision(6) << v1_ms << ","
          << std::fixed << std::setprecision(6) << v1_gflops << ","
          << (check_ok ? 1 : 0) << "," << std::fixed << std::setprecision(6) << max_err;
      append_csv_line(output_path, oss.str());
    }

    for (int stage : stage_values) {
      if (stage < 2 || stage > 4) {
        continue;
      }
      if (!supports_cp_async) {
        continue;
      }

      auto launch_v3 = [&]() {
        launch_stencil_v3(d_in, d_out, n, compute_iters, grid_blocks, stage);
      };
      const double v3_ms = benchmark_ms(launch_v3, warmup, iters);
      CUDA_CHECK(cudaGetLastError());

      bool v3_check = true;
      float v3_max_err = 0.0f;
      if (enable_check && n <= check_max_n) {
        h_out.resize(numel);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        v3_max_err = max_abs_diff(h_ref, h_out);
        v3_check = check_close_host(h_ref, h_out, atol, rtol, nullptr, nullptr, nullptr);
      }

      const double v3_gflops = (flops_per_elem * static_cast<double>(n)) / (v3_ms * 1.0e6);
      std::ostringstream oss;
      oss << "stencil," << gpu_name << "," << n << ",V3," << stage << "," << kBlockThreads << ","
          << std::fixed << std::setprecision(6) << v3_ms << ","
          << std::fixed << std::setprecision(6) << v3_gflops << ","
          << (v3_check ? 1 : 0) << "," << std::fixed << std::setprecision(6) << v3_max_err;
      append_csv_line(output_path, oss.str());
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_ref));

    std::cout << "[Stencil] finished L=" << n << std::endl;
  }

  std::cout << "[Stencil] wrote CSV: " << output_path << std::endl;
  return 0;
}
