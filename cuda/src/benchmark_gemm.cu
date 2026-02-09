#include "benchmark_common.cuh"
#include "cp_async_utils.cuh"

#include <cublas_v2.h>

#include <cmath>
#include <cstring>
#include <vector>

#define CUBLAS_CHECK(call)                                                        \
  do {                                                                            \
    cublasStatus_t _st = (call);                                                  \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                           \
      std::cerr << "cuBLAS error code=" << static_cast<int>(_st)                 \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

namespace {

constexpr int kThreadsX = 16;
constexpr int kThreadsY = 16;
constexpr int kThreads = kThreadsX * kThreadsY;

__global__ void gemm_v0_kernel(const float* a, const float* b, float* c, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int k = 0; k < n; ++k) {
    acc += a[row * n + k] * b[k * n + col];
  }
  c[row * n + col] = acc;
}

template <int TILE>
__device__ __forceinline__ void load_stage_sync(
    float* a_stage,
    float* b_stage,
    const float* a,
    const float* b,
    int n,
    int block_row,
    int block_col,
    int k_base,
    int tx,
    int ty) {
  const int tid = ty * kThreadsX + tx;
  const int elems = TILE * TILE;

  for (int linear = tid; linear < elems; linear += kThreads) {
    const int r = linear / TILE;
    const int c = linear - r * TILE;

    const int a_row = block_row + r;
    const int a_col = k_base + c;
    a_stage[linear] = (a_row < n && a_col < n) ? a[a_row * n + a_col] : 0.0f;

    const int b_row = k_base + r;
    const int b_col = block_col + c;
    b_stage[linear] = (b_row < n && b_col < n) ? b[b_row * n + b_col] : 0.0f;
  }
}

template <int TILE>
__device__ __forceinline__ void load_stage_async(
    float* a_stage,
    float* b_stage,
    const float* a,
    const float* b,
    int n,
    int block_row,
    int block_col,
    int k_base,
    int tx,
    int ty) {
  const int tid = ty * kThreadsX + tx;
  const int elems = TILE * TILE;

  for (int linear = tid; linear < elems; linear += kThreads) {
    const int r = linear / TILE;
    const int c = linear - r * TILE;

    float* a_dst = &a_stage[linear];
    float* b_dst = &b_stage[linear];

    const int a_row = block_row + r;
    const int a_col = k_base + c;
    if (a_row < n && a_col < n) {
      cp_async_cg_4(a_dst, &a[a_row * n + a_col]);
    } else {
      *a_dst = 0.0f;
    }

    const int b_row = k_base + r;
    const int b_col = block_col + c;
    if (b_row < n && b_col < n) {
      cp_async_cg_4(b_dst, &b[b_row * n + b_col]);
    } else {
      *b_dst = 0.0f;
    }
  }
}

template <int TILE>
__global__ void gemm_v1_kernel(const float* a, const float* b, float* c, int n) {
  static_assert(TILE % kThreadsX == 0, "TILE must be divisible by thread dimensions.");
  static_assert(TILE % kThreadsY == 0, "TILE must be divisible by thread dimensions.");

  constexpr int ROWS_PER_THREAD = TILE / kThreadsY;
  constexpr int COLS_PER_THREAD = TILE / kThreadsX;

  extern __shared__ float shared[];
  float* a_tile = shared;
  float* b_tile = shared + TILE * TILE;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int block_row = blockIdx.y * TILE;
  const int block_col = blockIdx.x * TILE;

  float acc[ROWS_PER_THREAD][COLS_PER_THREAD];
#pragma unroll
  for (int r = 0; r < ROWS_PER_THREAD; ++r) {
#pragma unroll
    for (int c_idx = 0; c_idx < COLS_PER_THREAD; ++c_idx) {
      acc[r][c_idx] = 0.0f;
    }
  }

  const int num_k_tiles = ceil_div_int(n, TILE);

  for (int tile = 0; tile < num_k_tiles; ++tile) {
    const int k_base = tile * TILE;
    load_stage_sync<TILE>(a_tile, b_tile, a, b, n, block_row, block_col, k_base, tx, ty);
    __syncthreads();

#pragma unroll 1
    for (int k = 0; k < TILE; ++k) {
      float a_frag[ROWS_PER_THREAD];
#pragma unroll
      for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row_in_tile = ty * ROWS_PER_THREAD + r;
        a_frag[r] = a_tile[row_in_tile * TILE + k];
      }

#pragma unroll
      for (int c_idx = 0; c_idx < COLS_PER_THREAD; ++c_idx) {
        const int col_in_tile = tx * COLS_PER_THREAD + c_idx;
        const float b_val = b_tile[k * TILE + col_in_tile];
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
          acc[r][c_idx] += a_frag[r] * b_val;
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < ROWS_PER_THREAD; ++r) {
    const int out_row = block_row + ty * ROWS_PER_THREAD + r;
    if (out_row >= n) {
      continue;
    }
#pragma unroll
    for (int c_idx = 0; c_idx < COLS_PER_THREAD; ++c_idx) {
      const int out_col = block_col + tx * COLS_PER_THREAD + c_idx;
      if (out_col < n) {
        c[out_row * n + out_col] = acc[r][c_idx];
      }
    }
  }
}

template <int TILE, int STAGES>
__global__ void gemm_v3_kernel(const float* a, const float* b, float* c, int n) {
  static_assert(STAGES >= 2 && STAGES <= 4, "Supported STAGES: 2..4");
  static_assert(TILE % kThreadsX == 0, "TILE must be divisible by thread dimensions.");
  static_assert(TILE % kThreadsY == 0, "TILE must be divisible by thread dimensions.");

  constexpr int ROWS_PER_THREAD = TILE / kThreadsY;
  constexpr int COLS_PER_THREAD = TILE / kThreadsX;

  extern __shared__ float shared[];
  float* a_ring = shared;
  float* b_ring = shared + STAGES * TILE * TILE;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int block_row = blockIdx.y * TILE;
  const int block_col = blockIdx.x * TILE;

  float acc[ROWS_PER_THREAD][COLS_PER_THREAD];
#pragma unroll
  for (int r = 0; r < ROWS_PER_THREAD; ++r) {
#pragma unroll
    for (int c_idx = 0; c_idx < COLS_PER_THREAD; ++c_idx) {
      acc[r][c_idx] = 0.0f;
    }
  }

  const int num_k_tiles = ceil_div_int(n, TILE);

  const int preload = (STAGES - 1 < num_k_tiles) ? (STAGES - 1) : num_k_tiles;
  for (int t = 0; t < preload; ++t) {
    const int stage = t % STAGES;
    float* a_stage = a_ring + stage * TILE * TILE;
    float* b_stage = b_ring + stage * TILE * TILE;
    load_stage_async<TILE>(a_stage, b_stage, a, b, n, block_row, block_col, t * TILE, tx, ty);
    cp_async_commit_group();
  }

  for (int tile = 0; tile < num_k_tiles; ++tile) {
    const int produce = tile + STAGES - 1;
    if (produce < num_k_tiles) {
      const int stage = produce % STAGES;
      float* a_stage = a_ring + stage * TILE * TILE;
      float* b_stage = b_ring + stage * TILE * TILE;
      load_stage_async<TILE>(a_stage, b_stage, a, b, n, block_row, block_col, produce * TILE, tx, ty);
      cp_async_commit_group();
    }

    cp_async_wait_group<STAGES - 1>();
    __syncthreads();

    const int consume = tile % STAGES;
    float* a_stage = a_ring + consume * TILE * TILE;
    float* b_stage = b_ring + consume * TILE * TILE;

#pragma unroll 1
    for (int k = 0; k < TILE; ++k) {
      float a_frag[ROWS_PER_THREAD];
#pragma unroll
      for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row_in_tile = ty * ROWS_PER_THREAD + r;
        a_frag[r] = a_stage[row_in_tile * TILE + k];
      }

#pragma unroll
      for (int c_idx = 0; c_idx < COLS_PER_THREAD; ++c_idx) {
        const int col_in_tile = tx * COLS_PER_THREAD + c_idx;
        const float b_val = b_stage[k * TILE + col_in_tile];
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
          acc[r][c_idx] += a_frag[r] * b_val;
        }
      }
    }

    __syncthreads();
  }

  cp_async_wait_group<0>();

#pragma unroll
  for (int r = 0; r < ROWS_PER_THREAD; ++r) {
    const int out_row = block_row + ty * ROWS_PER_THREAD + r;
    if (out_row >= n) {
      continue;
    }
#pragma unroll
    for (int c_idx = 0; c_idx < COLS_PER_THREAD; ++c_idx) {
      const int out_col = block_col + tx * COLS_PER_THREAD + c_idx;
      if (out_col < n) {
        c[out_row * n + out_col] = acc[r][c_idx];
      }
    }
  }
}

float max_abs_diff(const std::vector<float>& ref, const std::vector<float>& out) {
  const size_t n = ref.size();
  float max_diff = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    const float d = std::fabs(ref[i] - out[i]);
    if (d > max_diff) {
      max_diff = d;
    }
  }
  return max_diff;
}

void fill_matrix(std::vector<float>* buf) {
  for (size_t i = 0; i < buf->size(); ++i) {
    const float v = static_cast<float>((static_cast<int>(i * 13) % 23) - 11) / 11.0f;
    (*buf)[i] = v;
  }
}

void append_row(
    const std::string& output_path,
    const std::string& gpu_name,
    int n,
    const std::string& variant,
    int stage,
    int tile,
    double time_ms,
    double gflops,
    int correct,
    float max_err) {
  std::ostringstream oss;
  oss << "gemm," << gpu_name << "," << n << "," << variant << "," << stage << "," << tile << ","
      << std::fixed << std::setprecision(6) << time_ms << ","
      << std::fixed << std::setprecision(6) << gflops << "," << correct << ","
      << std::fixed << std::setprecision(6) << max_err;
  append_csv_line(output_path, oss.str());
}

template <int TILE>
bool set_kernel_smem_attr(size_t smem, size_t optin_limit) {
  if (smem <= 48 * 1024) {
    return true;
  }
  if (smem > optin_limit) {
    return false;
  }
  // Only set V1 here; V3 kernels set their own smem in launch_gemm_v3_impl.
  CUDA_CHECK(cudaFuncSetAttribute(gemm_v1_kernel<TILE>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  return true;
}

template <int TILE>
bool launch_gemm_v1_impl(const float* d_a, const float* d_b, float* d_c, int n, size_t optin_limit) {
  const dim3 block(kThreadsX, kThreadsY);
  const dim3 grid(ceil_div_int(n, TILE), ceil_div_int(n, TILE));
  const size_t smem = static_cast<size_t>(2 * TILE * TILE) * sizeof(float);

  if (!set_kernel_smem_attr<TILE>(smem, optin_limit)) {
    return false;
  }

  gemm_v1_kernel<TILE><<<grid, block, smem>>>(d_a, d_b, d_c, n);
  return true;
}

template <int TILE, int STAGES>
bool launch_gemm_v3_impl(const float* d_a, const float* d_b, float* d_c, int n, size_t optin_limit) {
  const dim3 block(kThreadsX, kThreadsY);
  const dim3 grid(ceil_div_int(n, TILE), ceil_div_int(n, TILE));
  const size_t smem = static_cast<size_t>(2 * STAGES * TILE * TILE) * sizeof(float);

  if (smem > optin_limit) {
    return false;
  }

  CUDA_CHECK(cudaFuncSetAttribute(gemm_v3_kernel<TILE, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  gemm_v3_kernel<TILE, STAGES><<<grid, block, smem>>>(d_a, d_b, d_c, n);
  return true;
}

bool launch_gemm_v1(const float* d_a, const float* d_b, float* d_c, int n, int tile, size_t optin_limit) {
  if (tile == 16) {
    return launch_gemm_v1_impl<16>(d_a, d_b, d_c, n, optin_limit);
  }
  if (tile == 32) {
    return launch_gemm_v1_impl<32>(d_a, d_b, d_c, n, optin_limit);
  }
  if (tile == 64) {
    return launch_gemm_v1_impl<64>(d_a, d_b, d_c, n, optin_limit);
  }
  if (tile == 128) {
    return launch_gemm_v1_impl<128>(d_a, d_b, d_c, n, optin_limit);
  }
  return false;
}

bool launch_gemm_v3(
    const float* d_a,
    const float* d_b,
    float* d_c,
    int n,
    int tile,
    int stage,
    size_t optin_limit) {
  if (tile == 16) {
    if (stage == 2) {
      return launch_gemm_v3_impl<16, 2>(d_a, d_b, d_c, n, optin_limit);
    }
    if (stage == 3) {
      return launch_gemm_v3_impl<16, 3>(d_a, d_b, d_c, n, optin_limit);
    }
    if (stage == 4) {
      return launch_gemm_v3_impl<16, 4>(d_a, d_b, d_c, n, optin_limit);
    }
  }
  if (tile == 32) {
    if (stage == 2) {
      return launch_gemm_v3_impl<32, 2>(d_a, d_b, d_c, n, optin_limit);
    }
    if (stage == 3) {
      return launch_gemm_v3_impl<32, 3>(d_a, d_b, d_c, n, optin_limit);
    }
    if (stage == 4) {
      return launch_gemm_v3_impl<32, 4>(d_a, d_b, d_c, n, optin_limit);
    }
  }
  if (tile == 64) {
    if (stage == 2) {
      return launch_gemm_v3_impl<64, 2>(d_a, d_b, d_c, n, optin_limit);
    }
    if (stage == 3) {
      return launch_gemm_v3_impl<64, 3>(d_a, d_b, d_c, n, optin_limit);
    }
    if (stage == 4) {
      return launch_gemm_v3_impl<64, 4>(d_a, d_b, d_c, n, optin_limit);
    }
  }
  if (tile == 128) {
    if (stage == 2) {
      return launch_gemm_v3_impl<128, 2>(d_a, d_b, d_c, n, optin_limit);
    }
    if (stage == 3) {
      return launch_gemm_v3_impl<128, 3>(d_a, d_b, d_c, n, optin_limit);
    }
    if (stage == 4) {
      return launch_gemm_v3_impl<128, 4>(d_a, d_b, d_c, n, optin_limit);
    }
  }
  return false;
}

double run_cublas_ref(cublasHandle_t handle, const float* d_a, const float* d_b, float* d_c, int n, int warmup, int iters) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  auto launch_ref = [&]() {
    // Row-major C = A * B  <=> column-major C^T = B^T * A^T.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_b, n, d_a, n, &beta, d_c, n));
  };

  const double ref_ms = benchmark_ms(launch_ref, warmup, iters);
  CUDA_CHECK(cudaGetLastError());
  return ref_ms;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<int> default_n_values = {256, 512, 1024, 2048, 3072, 4096, 6144, 8192};
  const std::vector<int> default_stage_values = {2, 3, 4};
  const std::vector<int> default_tile_values = {16, 32, 64, 128};

  std::vector<int> n_values = parse_csv_ints(get_arg(argc, argv, "--n-values", ""));
  if (n_values.empty()) {
    n_values = default_n_values;
  }

  std::vector<int> stage_values = parse_csv_ints(get_arg(argc, argv, "--stages", ""));
  if (stage_values.empty()) {
    stage_values = default_stage_values;
  }

  std::vector<int> tile_values = parse_csv_ints(get_arg(argc, argv, "--tile-sizes", ""));
  if (tile_values.empty()) {
    tile_values = default_tile_values;
  }

  const int warmup = get_arg_int(argc, argv, "--warmup", 3);
  const int iters = get_arg_int(argc, argv, "--iters", 10);
  const int check_max_n = get_arg_int(argc, argv, "--check-max-n", 2048);
  const bool run_v0 = has_flag(argc, argv, "--run-v0");
  const bool run_ref = !has_flag(argc, argv, "--no-ref");
  const bool enable_check = !has_flag(argc, argv, "--no-check");
  const float atol = get_arg_float(argc, argv, "--atol", 1e-2f);
  const float rtol = get_arg_float(argc, argv, "--rtol", 1e-2f);

  const std::string output_path = get_arg(argc, argv, "--output", "outputs/gemm_raw.csv");
  const std::string gpu_name_arg = get_arg(argc, argv, "--gpu-name", "");
  const std::string gpu_name = gpu_name_arg.empty() ? detect_gpu_name() : gpu_name_arg;

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const bool supports_cp_async = (prop.major * 10 + prop.minor) >= 80;
  const size_t optin_limit = static_cast<size_t>(prop.sharedMemPerBlockOptin);

  std::cout << "[GEMM] GPU=" << gpu_name << " compute_capability=" << prop.major << "." << prop.minor
            << " cp.async=" << (supports_cp_async ? "yes" : "no") << " optin_smem=" << optin_limit << std::endl;

  write_csv_header_if_needed(
      output_path,
      "workload,gpu,problem_size,variant,stage,tile_size,time_ms,gflops,correct,max_abs_error");

  cublasHandle_t cublas = nullptr;
  if (run_ref) {
    CUBLAS_CHECK(cublasCreate(&cublas));
  }

  for (int n : n_values) {
    const size_t numel = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = numel * sizeof(float);

    std::vector<float> h_a(numel);
    std::vector<float> h_b(numel);
    std::vector<float> h_ref;
    std::vector<float> h_out;

    fill_matrix(&h_a);
    fill_matrix(&h_b);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    float* d_ref = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMalloc(&d_ref, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    double ref_ms = 0.0;
    double ref_gflops = 0.0;

    if (run_ref) {
      ref_ms = run_cublas_ref(cublas, d_a, d_b, d_ref, n, warmup, iters);
      ref_gflops = (2.0 * static_cast<double>(n) * n * n) / (ref_ms * 1.0e6);

      if (enable_check && n <= check_max_n) {
        h_ref.resize(numel);
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, bytes, cudaMemcpyDeviceToHost));
      }
    }

    for (size_t ti = 0; ti < tile_values.size(); ++ti) {
      const int tile = tile_values[ti];

      if (!(tile == 16 || tile == 32 || tile == 64 || tile == 128)) {
        std::cout << "[GEMM] skip unsupported tile=" << tile << std::endl;
        continue;
      }

      if (run_ref) {
        append_row(output_path, gpu_name, n, "Ref", 0, tile, ref_ms, ref_gflops, 1, 0.0f);
      }

      bool launched_v1 = false;
      auto launch_v1 = [&]() {
        launched_v1 = launch_gemm_v1(d_a, d_b, d_c, n, tile, optin_limit);
      };

      const double v1_ms = benchmark_ms(launch_v1, warmup, iters);
      CUDA_CHECK(cudaGetLastError());

      if (!launched_v1) {
        std::cout << "[GEMM] skip V1 tile=" << tile << " due shared-memory limit." << std::endl;
        continue;
      }

      bool v1_check = true;
      float v1_max_err = 0.0f;
      if (enable_check && n <= check_max_n) {
        if (h_ref.empty()) {
          // Fallback if ref is disabled.
          h_ref.resize(numel);
          CUDA_CHECK(cudaMemcpy(h_ref.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        }
        h_out.resize(numel);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        v1_max_err = max_abs_diff(h_ref, h_out);
        v1_check = check_close_host(h_ref, h_out, atol, rtol, nullptr, nullptr, nullptr);
      }

      const double v1_gflops = (2.0 * static_cast<double>(n) * n * n) / (v1_ms * 1.0e6);
      append_row(output_path, gpu_name, n, "V1", 1, tile, v1_ms, v1_gflops, v1_check ? 1 : 0, v1_max_err);

      if (run_v0) {
        const dim3 block(kThreadsX, kThreadsY);
        const dim3 grid(ceil_div_int(n, kThreadsX), ceil_div_int(n, kThreadsY));

        auto launch_v0 = [&]() {
          gemm_v0_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
        };

        const double v0_ms = benchmark_ms(launch_v0, warmup, iters);
        CUDA_CHECK(cudaGetLastError());

        bool v0_check = true;
        float v0_max_err = 0.0f;
        if (enable_check && n <= check_max_n) {
          h_out.resize(numel);
          CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, bytes, cudaMemcpyDeviceToHost));
          v0_max_err = max_abs_diff(h_ref, h_out);
          v0_check = check_close_host(h_ref, h_out, atol, rtol, nullptr, nullptr, nullptr);
        }

        const double v0_gflops = (2.0 * static_cast<double>(n) * n * n) / (v0_ms * 1.0e6);
        append_row(output_path, gpu_name, n, "V0", 0, tile, v0_ms, v0_gflops, v0_check ? 1 : 0, v0_max_err);
      }

      for (int stage : stage_values) {
        if (stage < 2 || stage > 4) {
          continue;
        }
        if (!supports_cp_async) {
          continue;
        }

        bool launched_v3 = false;
        auto launch_v3 = [&]() {
          launched_v3 = launch_gemm_v3(d_a, d_b, d_c, n, tile, stage, optin_limit);
        };

        const double v3_ms = benchmark_ms(launch_v3, warmup, iters);
        CUDA_CHECK(cudaGetLastError());

        if (!launched_v3) {
          std::cout << "[GEMM] skip V3 tile=" << tile << " stage=" << stage
                    << " due shared-memory limit." << std::endl;
          continue;
        }

        bool v3_check = true;
        float v3_max_err = 0.0f;
        if (enable_check && n <= check_max_n) {
          h_out.resize(numel);
          CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, bytes, cudaMemcpyDeviceToHost));
          v3_max_err = max_abs_diff(h_ref, h_out);
          v3_check = check_close_host(h_ref, h_out, atol, rtol, nullptr, nullptr, nullptr);
        }

        const double v3_gflops = (2.0 * static_cast<double>(n) * n * n) / (v3_ms * 1.0e6);
        append_row(output_path, gpu_name, n, "V3", stage, tile, v3_ms, v3_gflops, v3_check ? 1 : 0, v3_max_err);
      }
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_ref));

    std::cout << "[GEMM] finished N=" << n << std::endl;
  }

  if (run_ref) {
    CUBLAS_CHECK(cublasDestroy(cublas));
  }

  std::cout << "[GEMM] wrote CSV: " << output_path << std::endl;
  return 0;
}
