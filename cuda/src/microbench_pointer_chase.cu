#include "benchmark_common.cuh"

#include <numeric>
#include <vector>

namespace {

__global__ void pointer_chase_kernel(
    const int* next_idx,
    int start_idx,
    int iterations,
    unsigned long long* cycles_out,
    int* sink_out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int idx = start_idx;
    const unsigned long long begin = clock64();
    for (int i = 0; i < iterations; ++i) {
#if __CUDA_ARCH__ >= 350
      idx = __ldg(&next_idx[idx]);
#else
      idx = next_idx[idx];
#endif
    }
    const unsigned long long end = clock64();
    cycles_out[0] = end - begin;
    sink_out[0] = idx;
  }
}

__global__ void streaming_read_kernel(const float* data, int n, int passes, float* sink) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  float acc = 0.0f;
  for (int p = 0; p < passes; ++p) {
    for (int i = tid; i < n; i += stride) {
#if __CUDA_ARCH__ >= 350
      acc += __ldg(&data[i]);
#else
      acc += data[i];
#endif
    }
  }

  if (acc != 0.0f) {
    atomicAdd(sink, acc);
  }
}

std::vector<size_t> default_sizes_bytes() {
  const size_t mb = 1024ULL * 1024ULL;
  return {
      1 * mb,  2 * mb,  3 * mb,  4 * mb,  6 * mb,  8 * mb,  10 * mb,
      12 * mb, 16 * mb, 20 * mb, 24 * mb, 28 * mb, 32 * mb, 36 * mb,
      40 * mb,
  };
}

void build_stride_ring(std::vector<int>* host_next, int stride) {
  const int n = static_cast<int>(host_next->size());
  for (int i = 0; i < n; ++i) {
    (*host_next)[i] = (i + stride) % n;
  }
}

void fill_stream_data(std::vector<float>* data) {
  for (size_t i = 0; i < data->size(); ++i) {
    (*data)[i] = static_cast<float>((static_cast<int>(i * 7) % 19) - 9) / 9.0f;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int iterations = get_arg_int(argc, argv, "--iterations", 1 << 22);
  const int repeats = get_arg_int(argc, argv, "--repeats", 7);
  const int warmup = get_arg_int(argc, argv, "--warmup", 3);
  const int stride = get_arg_int(argc, argv, "--stride", 32);
  const int stream_passes = get_arg_int(argc, argv, "--stream-passes", 16);
  const bool measure_bandwidth = !has_flag(argc, argv, "--no-bandwidth");

  std::vector<size_t> sizes = parse_csv_sizes(get_arg(argc, argv, "--sizes-bytes", ""));
  if (sizes.empty()) {
    sizes = default_sizes_bytes();
  }

  const std::string output_path = get_arg(argc, argv, "--output", "outputs/pointer_chase_raw.csv");
  const std::string gpu_name_arg = get_arg(argc, argv, "--gpu-name", "");
  const std::string gpu_name = gpu_name_arg.empty() ? detect_gpu_name() : gpu_name_arg;

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

  std::cout << "[PointerChase] GPU=" << gpu_name << " core_clock_khz=" << prop.clockRate << std::endl;

  write_csv_header_if_needed(
      output_path,
      "gpu,size_bytes,elements,stride,iterations,cycles_per_load,ns_per_load,stream_bandwidth_gbps");

  unsigned long long* d_cycles = nullptr;
  int* d_sink = nullptr;
  float* d_stream_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_stream_sink, sizeof(float)));

  for (size_t size_bytes : sizes) {
    const size_t numel = std::max<size_t>(size_bytes / sizeof(int), static_cast<size_t>(stride + 1));

    std::vector<int> h_next(numel);
    build_stride_ring(&h_next, stride);

    int* d_next = nullptr;
    CUDA_CHECK(cudaMalloc(&d_next, numel * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_next, h_next.data(), numel * sizeof(int), cudaMemcpyHostToDevice));

    // Warmup runs for latency.
    for (int i = 0; i < warmup; ++i) {
      pointer_chase_kernel<<<1, 1>>>(d_next, 0, iterations, d_cycles, d_sink);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> sample_cycles;
    sample_cycles.reserve(static_cast<size_t>(repeats));

    for (int i = 0; i < repeats; ++i) {
      pointer_chase_kernel<<<1, 1>>>(d_next, 0, iterations, d_cycles, d_sink);
      CUDA_CHECK(cudaDeviceSynchronize());

      unsigned long long h_cycles = 0;
      CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      const double cycles_per_load = static_cast<double>(h_cycles) / static_cast<double>(iterations);
      sample_cycles.push_back(cycles_per_load);
    }

    std::sort(sample_cycles.begin(), sample_cycles.end());
    const double median_cycles =
        (sample_cycles.size() % 2 == 1)
            ? sample_cycles[sample_cycles.size() / 2]
            : 0.5 * (sample_cycles[sample_cycles.size() / 2 - 1] + sample_cycles[sample_cycles.size() / 2]);

    const double ns_per_cycle = 1.0e6 / static_cast<double>(prop.clockRate);
    const double ns_per_load = median_cycles * ns_per_cycle;

    double stream_gbps = 0.0;
    if (measure_bandwidth) {
      std::vector<float> h_stream(numel);
      fill_stream_data(&h_stream);

      float* d_stream = nullptr;
      CUDA_CHECK(cudaMalloc(&d_stream, numel * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(d_stream, h_stream.data(), numel * sizeof(float), cudaMemcpyHostToDevice));

      const int block = 256;
      const int grid = std::max(1, std::min(4096, ceil_div_int(static_cast<int>(numel), block)));

      auto launch_stream = [&]() {
        CUDA_CHECK(cudaMemset(d_stream_sink, 0, sizeof(float)));
        streaming_read_kernel<<<grid, block>>>(d_stream, static_cast<int>(numel), stream_passes, d_stream_sink);
      };

      const double stream_ms = benchmark_ms(launch_stream, warmup, repeats);
      CUDA_CHECK(cudaGetLastError());

      const double bytes_read = static_cast<double>(numel) * sizeof(float) * static_cast<double>(stream_passes);
      stream_gbps = bytes_read / (stream_ms * 1.0e6);

      CUDA_CHECK(cudaFree(d_stream));
    }

    std::ostringstream oss;
    oss << gpu_name << "," << (numel * sizeof(int)) << "," << numel << "," << stride << "," << iterations << ","
        << std::fixed << std::setprecision(6) << median_cycles << ","
        << std::fixed << std::setprecision(6) << ns_per_load << ","
        << std::fixed << std::setprecision(6) << stream_gbps;
    append_csv_line(output_path, oss.str());

    CUDA_CHECK(cudaFree(d_next));
    std::cout << "[PointerChase] size_bytes=" << (numel * sizeof(int))
              << " cycles/load=" << median_cycles << " ns/load=" << ns_per_load
              << " stream_gbps=" << stream_gbps << std::endl;
  }

  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_stream_sink));

  std::cout << "[PointerChase] wrote CSV: " << output_path << std::endl;
  return 0;
}
