#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    cudaError_t _err = (call);                                                    \
    if (_err != cudaSuccess) {                                                    \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                    \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

inline int ceil_div_int(int x, int y) {
  return (x + y - 1) / y;
}

inline std::string trim_copy(const std::string& s) {
  const auto begin = s.find_first_not_of(" \t\n\r");
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = s.find_last_not_of(" \t\n\r");
  return s.substr(begin, end - begin + 1);
}

inline std::string get_arg(int argc, char** argv, const std::string& key, const std::string& default_value) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) {
      return std::string(argv[i + 1]);
    }
  }
  return default_value;
}

inline bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key) {
      return true;
    }
  }
  return false;
}

inline int get_arg_int(int argc, char** argv, const std::string& key, int default_value) {
  const std::string raw = get_arg(argc, argv, key, "");
  if (raw.empty()) {
    return default_value;
  }
  return std::stoi(raw);
}

inline float get_arg_float(int argc, char** argv, const std::string& key, float default_value) {
  const std::string raw = get_arg(argc, argv, key, "");
  if (raw.empty()) {
    return default_value;
  }
  return std::stof(raw);
}

inline std::vector<int> parse_csv_ints(const std::string& raw) {
  std::vector<int> values;
  std::stringstream ss(raw);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = trim_copy(token);
    if (!token.empty()) {
      values.push_back(std::stoi(token));
    }
  }
  return values;
}

inline std::vector<size_t> parse_csv_sizes(const std::string& raw) {
  std::vector<size_t> values;
  std::stringstream ss(raw);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = trim_copy(token);
    if (!token.empty()) {
      values.push_back(static_cast<size_t>(std::stoull(token)));
    }
  }
  return values;
}

inline double median_ms(std::vector<float> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  if (n % 2 == 1) {
    return static_cast<double>(values[n / 2]);
  }
  return 0.5 * (static_cast<double>(values[n / 2 - 1]) + static_cast<double>(values[n / 2]));
}

inline std::string sanitize_gpu_name(const std::string& name) {
  std::string out = name;
  for (char& c : out) {
    if (std::isspace(static_cast<unsigned char>(c)) || c == '/') {
      c = '_';
    }
  }
  return out;
}

inline std::string detect_gpu_name() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  return sanitize_gpu_name(prop.name);
}

inline void write_csv_header_if_needed(const std::string& path, const std::string& header_line) {
  std::ifstream in(path.c_str());
  if (in.good() && in.peek() != std::ifstream::traits_type::eof()) {
    return;
  }
  std::ofstream out(path.c_str(), std::ios::out | std::ios::trunc);
  if (!out.good()) {
    throw std::runtime_error("Failed to open output file: " + path);
  }
  out << header_line << "\n";
}

inline void append_csv_line(const std::string& path, const std::string& line) {
  std::ofstream out(path.c_str(), std::ios::out | std::ios::app);
  if (!out.good()) {
    throw std::runtime_error("Failed to append output file: " + path);
  }
  out << line << "\n";
}

template <typename LaunchFn>
double benchmark_ms(const LaunchFn& launch, int warmup, int iters) {
  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  std::vector<float> samples;
  samples.reserve(static_cast<size_t>(iters));

  for (int i = 0; i < warmup + iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start_event));
    launch();
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    if (i >= warmup) {
      samples.push_back(elapsed_ms);
    }
  }

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
  return median_ms(samples);
}

inline bool almost_equal(float a, float b, float atol, float rtol) {
  const float diff = std::fabs(a - b);
  const float bound = atol + rtol * std::fabs(b);
  return diff <= bound;
}

inline bool check_close_host(
    const std::vector<float>& ref,
    const std::vector<float>& out,
    float atol,
    float rtol,
    int* mismatch_index,
    float* ref_value,
    float* out_value) {
  if (ref.size() != out.size()) {
    return false;
  }
  for (size_t i = 0; i < ref.size(); ++i) {
    if (!almost_equal(out[i], ref[i], atol, rtol)) {
      if (mismatch_index) {
        *mismatch_index = static_cast<int>(i);
      }
      if (ref_value) {
        *ref_value = ref[i];
      }
      if (out_value) {
        *out_value = out[i];
      }
      return false;
    }
  }
  return true;
}
