// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "experimental/openpi_image_encoder_hip_backend.h"
#include "experimental/openpi_image_encoder_hip_bundle_bridge.h"
#include "experimental/openpi_image_encoder_raw_bundle.h"

namespace {

constexpr size_t kTokens = 256;
constexpr size_t kWidth = 1152;
constexpr size_t kHeads = 16;
constexpr size_t kHeadDim = 72;
constexpr size_t kQDim = kHeads * kHeadDim;
constexpr size_t kMlpDim = 4304;

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    const hipError_t status = (call);                                          \
    if (status != hipSuccess) {                                                \
      std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,       \
                   hipGetErrorString(status));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define ROCBLAS_CHECK(call)                                                    \
  do {                                                                         \
    const rocblas_status status = (call);                                      \
    if (status != rocblas_status_success) {                                    \
      std::fprintf(stderr, "rocBLAS error %s:%d: %d\n", __FILE__, __LINE__,   \
                   static_cast<int>(status));                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Args {
  std::string bundle;
  std::string attention = "serial";
  bool strict = true;
  double tol = 3.0;
  int warmup = 3;
  int runs = 20;
};

struct Metrics {
  double max_abs = 0.0;
  double rms = 0.0;
  double mean_abs = 0.0;
};

struct ProjectionResult {
  Metrics metrics;
  float gemm_ms = 0.0f;
  std::vector<float> cpu;
};

__global__ void AddBiasToBf16Kernel(const float* x, const float* bias,
                                    rocblas_bfloat16* out, int rows,
                                    int cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * cols;
  if (idx >= count) return;
  out[idx] = rocblas_bfloat16(x[idx] + bias[idx % cols]);
}

__global__ void AddResidualBf16ToF32Kernel(const float* residual,
                                           const rocblas_bfloat16* branch,
                                           float* out, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  out[idx] =
      static_cast<float>(rocblas_bfloat16(residual[idx] +
                                          static_cast<float>(branch[idx])));
}

__global__ void GeluBf16Kernel(rocblas_bfloat16* values, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  const float x = static_cast<float>(values[idx]);
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  const float y =
      0.5f * x *
      (1.0f + tanhf(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
  values[idx] = rocblas_bfloat16(y);
}

__global__ void LayerNormToBf16Kernel(const float* x, const float* scale,
                                      const float* bias,
                                      rocblas_bfloat16* out, int rows,
                                      int cols) {
  extern __shared__ float shared[];
  float* sums = shared;
  float* sums_sq = shared + blockDim.x;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  float sum = 0.0f;
  float sum_sq = 0.0f;
  const int base = row * cols;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float value = x[base + col];
    sum += value;
    sum_sq += value * value;
  }
  sums[tid] = sum;
  sums_sq[tid] = sum_sq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
      sums_sq[tid] += sums_sq[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float mean = sums[0] / static_cast<float>(cols);
    float variance = sums_sq[0] / static_cast<float>(cols) - mean * mean;
    variance = fmaxf(variance, 0.0f);
    sums[0] = mean;
    sums_sq[0] = rsqrtf(variance + 1e-6f);
  }
  __syncthreads();

  const float mean = sums[0];
  const float inv = sums_sq[0];
  for (int col = tid; col < cols; col += blockDim.x) {
    const float normalized = (x[base + col] - mean) * inv * scale[col] +
                             bias[col];
    out[base + col] = rocblas_bfloat16(normalized);
  }
}

__global__ void AttentionBf16Kernel(const rocblas_bfloat16* q,
                                    const rocblas_bfloat16* k,
                                    const rocblas_bfloat16* v,
                                    rocblas_bfloat16* out) {
  __shared__ float logits[kTokens];
  __shared__ float probs[kTokens];

  const int q_token = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  if (tid != 0) return;

  const float scale =
      static_cast<float>(rocblas_bfloat16(1.0f / sqrtf(float(kHeadDim))));

  float max_logit = -INFINITY;
  for (int k_token = 0; k_token < int(kTokens); ++k_token) {
    float acc = 0.0f;
    for (int d = 0; d < int(kHeadDim); ++d) {
      const int q_index = (q_token * int(kHeads) + head) * int(kHeadDim) + d;
      const int k_index = (k_token * int(kHeads) + head) * int(kHeadDim) + d;
      const float q_scaled =
          static_cast<float>(rocblas_bfloat16(static_cast<float>(q[q_index]) *
                                              scale));
      acc += q_scaled * static_cast<float>(k[k_index]);
    }
    logits[k_token] = acc;
    max_logit = fmaxf(max_logit, acc);
  }

  float denom = 0.0f;
  for (int k_token = 0; k_token < int(kTokens); ++k_token) {
    const float prob = expf(logits[k_token] - max_logit);
    probs[k_token] = prob;
    denom += prob;
  }
  for (int k_token = 0; k_token < int(kTokens); ++k_token) {
    probs[k_token] = static_cast<float>(rocblas_bfloat16(probs[k_token] / denom));
  }

  for (int d = 0; d < int(kHeadDim); ++d) {
    float acc = 0.0f;
    for (int k_token = 0; k_token < int(kTokens); ++k_token) {
      const int v_index =
          (k_token * int(kHeads) + head) * int(kHeadDim) + d;
      acc += probs[k_token] * static_cast<float>(v[v_index]);
    }
    const int out_index = (q_token * int(kHeads) + head) * int(kHeadDim) + d;
    out[out_index] = rocblas_bfloat16(acc);
  }
}

__global__ void AttentionBf16HybridKernel(const rocblas_bfloat16* q,
                                          const rocblas_bfloat16* k,
                                          const rocblas_bfloat16* v,
                                          rocblas_bfloat16* out) {
  __shared__ float logits[kTokens];
  __shared__ float probs[kTokens];

  const int q_token = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const float scale =
      static_cast<float>(rocblas_bfloat16(1.0f / sqrtf(float(kHeadDim))));

  if (tid < int(kTokens)) {
    float acc = 0.0f;
    for (int d = 0; d < int(kHeadDim); ++d) {
      const int q_index = (q_token * int(kHeads) + head) * int(kHeadDim) + d;
      const int k_index = (tid * int(kHeads) + head) * int(kHeadDim) + d;
      const float q_scaled =
          static_cast<float>(rocblas_bfloat16(static_cast<float>(q[q_index]) *
                                              scale));
      acc += q_scaled * static_cast<float>(k[k_index]);
    }
    logits[tid] = acc;
  }
  __syncthreads();

  if (tid == 0) {
    float max_logit = -INFINITY;
    for (int k_token = 0; k_token < int(kTokens); ++k_token) {
      max_logit = fmaxf(max_logit, logits[k_token]);
    }

    float denom = 0.0f;
    for (int k_token = 0; k_token < int(kTokens); ++k_token) {
      const float prob = expf(logits[k_token] - max_logit);
      probs[k_token] = prob;
      denom += prob;
    }
    for (int k_token = 0; k_token < int(kTokens); ++k_token) {
      probs[k_token] =
          static_cast<float>(rocblas_bfloat16(probs[k_token] / denom));
    }
  }
  __syncthreads();

  if (tid < int(kHeadDim)) {
    float acc = 0.0f;
    for (int k_token = 0; k_token < int(kTokens); ++k_token) {
      const int v_index =
          (k_token * int(kHeads) + head) * int(kHeadDim) + tid;
      acc += probs[k_token] * static_cast<float>(v[v_index]);
    }
    const int out_index = (q_token * int(kHeads) + head) * int(kHeadDim) + tid;
    out[out_index] = rocblas_bfloat16(acc);
  }
}

__global__ void AttentionBf16ParallelKernel(const rocblas_bfloat16* q,
                                            const rocblas_bfloat16* k,
                                            const rocblas_bfloat16* v,
                                            rocblas_bfloat16* out) {
  __shared__ float logits[kTokens];
  __shared__ float probs[kTokens];
  __shared__ float reduction[kTokens];

  const int q_token = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const float scale =
      static_cast<float>(rocblas_bfloat16(1.0f / sqrtf(float(kHeadDim))));

  if (tid < int(kTokens)) {
    float acc = 0.0f;
    for (int d = 0; d < int(kHeadDim); ++d) {
      const int q_index = (q_token * int(kHeads) + head) * int(kHeadDim) + d;
      const int k_index = (tid * int(kHeads) + head) * int(kHeadDim) + d;
      const float q_scaled =
          static_cast<float>(rocblas_bfloat16(static_cast<float>(q[q_index]) *
                                              scale));
      acc += q_scaled * static_cast<float>(k[k_index]);
    }
    logits[tid] = acc;
    reduction[tid] = acc;
  }
  __syncthreads();

  for (int stride = int(kTokens) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  const float max_logit = reduction[0];

  if (tid < int(kTokens)) {
    const float prob = expf(logits[tid] - max_logit);
    probs[tid] = prob;
    reduction[tid] = prob;
  }
  __syncthreads();

  for (int stride = int(kTokens) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  const float inv_denom = 1.0f / reduction[0];
  if (tid < int(kTokens)) {
    probs[tid] = static_cast<float>(rocblas_bfloat16(probs[tid] * inv_denom));
  }
  __syncthreads();

  if (tid < int(kHeadDim)) {
    float acc = 0.0f;
    for (int k_token = 0; k_token < int(kTokens); ++k_token) {
      const int v_index =
          (k_token * int(kHeads) + head) * int(kHeadDim) + tid;
      acc += probs[k_token] * static_cast<float>(v[v_index]);
    }
    const int out_index = (q_token * int(kHeads) + head) * int(kHeadDim) + tid;
    out[out_index] = rocblas_bfloat16(acc);
  }
}

bool ParseBool(const char* value, bool& out) {
  if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0) {
    out = true;
    return true;
  }
  if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0) {
    out = false;
    return true;
  }
  return false;
}

bool ParseDouble(const char* value, double& out) {
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || *end != '\0') return false;
  out = parsed;
  return true;
}

bool ParseInt(const char* value, int& out) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  out = static_cast<int>(parsed);
  return true;
}

bool ParseArgs(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--bundle" && i + 1 < argc) {
      args.bundle = argv[++i];
      continue;
    }
    if (arg == "--attention" && i + 1 < argc) {
      args.attention = argv[++i];
      if (args.attention != "serial" && args.attention != "hybrid" &&
          args.attention != "parallel") {
        std::fprintf(stderr, "unknown attention mode: %s\n",
                     args.attention.c_str());
        return false;
      }
      continue;
    }
    if (arg == "--strict" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.strict)) return false;
      continue;
    }
    if (arg == "--tol" && i + 1 < argc) {
      if (!ParseDouble(argv[++i], args.tol)) return false;
      continue;
    }
    if (arg == "--warmup" && i + 1 < argc) {
      if (!ParseInt(argv[++i], args.warmup)) return false;
      continue;
    }
    if (arg == "--runs" && i + 1 < argc) {
      if (!ParseInt(argv[++i], args.runs)) return false;
      continue;
    }
    std::fprintf(stderr, "unknown or incomplete argument: %s\n", arg.c_str());
    return false;
  }
  return !args.bundle.empty() && args.runs > 0;
}

float RoundBfloat16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;
  bits &= 0xFFFF0000u;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::vector<float> SliceLayer(const std::vector<float>& stacked, size_t layer,
                              size_t layer_size) {
  const size_t offset = layer * layer_size;
  return std::vector<float>(stacked.begin() + offset,
                            stacked.begin() + offset + layer_size);
}

std::vector<float> LayerNormBfloat16(const std::vector<float>& x,
                                     const std::vector<float>& scale,
                                     const std::vector<float>& bias) {
  std::vector<float> y(x.size());
  for (size_t token = 0; token < kTokens; ++token) {
    const size_t base = token * kWidth;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (size_t d = 0; d < kWidth; ++d) {
      const float value = x[base + d];
      sum += value;
      sum_sq += value * value;
    }
    const float mean = sum / static_cast<float>(kWidth);
    float variance = sum_sq / static_cast<float>(kWidth) - mean * mean;
    variance = std::max(variance, 0.0f);
    const float inv = 1.0f / std::sqrt(variance + 1e-6f);
    for (size_t d = 0; d < kWidth; ++d) {
      y[base + d] = RoundBfloat16((x[base + d] - mean) * inv * scale[d] +
                                  bias[d]);
    }
  }
  return y;
}

std::vector<float> DenseBfloat16(const std::vector<float>& x, size_t rows,
                                 size_t in_dim, const std::vector<float>& kernel,
                                 const std::vector<float>& bias,
                                 size_t out_dim) {
  std::vector<float> y(rows * out_dim);
  for (size_t row = 0; row < rows; ++row) {
    float* out = y.data() + row * out_dim;
    std::copy(bias.begin(), bias.begin() + out_dim, out);
    for (size_t in = 0; in < in_dim; ++in) {
      const float value = x[row * in_dim + in];
      const float* weights = kernel.data() + in * out_dim;
      for (size_t out_col = 0; out_col < out_dim; ++out_col) {
        out[out_col] += value * weights[out_col];
      }
    }
    for (size_t out_col = 0; out_col < out_dim; ++out_col) {
      out[out_col] = RoundBfloat16(out[out_col]);
    }
  }
  return y;
}

void RoundVectorBfloat16(std::vector<float>& values) {
  for (float& value : values) {
    value = RoundBfloat16(value);
  }
}

std::vector<float> AddVectorsBfloat16(const std::vector<float>& a,
                                      const std::vector<float>& b) {
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] + b[i];
  }
  RoundVectorBfloat16(out);
  return out;
}

float GeluApprox(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  return 0.5f * x *
         (1.0f + std::tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}

void ApplyGeluBfloat16(std::vector<float>& values) {
  for (float& value : values) {
    value = RoundBfloat16(GeluApprox(value));
  }
}

std::vector<float> AttentionBfloat16(const std::vector<float>& query,
                                     const std::vector<float>& key,
                                     const std::vector<float>& value) {
  std::vector<float> context(kTokens * kQDim);
  std::vector<float> logits(kTokens);
  std::vector<float> probs(kTokens);
  const float scale =
      RoundBfloat16(1.0f / std::sqrt(static_cast<float>(kHeadDim)));
  for (size_t head = 0; head < kHeads; ++head) {
    for (size_t q_token = 0; q_token < kTokens; ++q_token) {
      float max_logit = -std::numeric_limits<float>::infinity();
      for (size_t k_token = 0; k_token < kTokens; ++k_token) {
        float acc = 0.0f;
        for (size_t d = 0; d < kHeadDim; ++d) {
          const size_t q_index = (q_token * kHeads + head) * kHeadDim + d;
          const size_t k_index = (k_token * kHeads + head) * kHeadDim + d;
          acc += RoundBfloat16(query[q_index] * scale) * key[k_index];
        }
        logits[k_token] = acc;
        max_logit = std::max(max_logit, acc);
      }
      float denom = 0.0f;
      for (size_t k_token = 0; k_token < kTokens; ++k_token) {
        const float prob = std::exp(logits[k_token] - max_logit);
        probs[k_token] = prob;
        denom += prob;
      }
      for (size_t k_token = 0; k_token < kTokens; ++k_token) {
        probs[k_token] = RoundBfloat16(probs[k_token] / denom);
      }
      for (size_t d = 0; d < kHeadDim; ++d) {
        float acc = 0.0f;
        for (size_t k_token = 0; k_token < kTokens; ++k_token) {
          const size_t v_index = (k_token * kHeads + head) * kHeadDim + d;
          acc += probs[k_token] * value[v_index];
        }
        const size_t out_index = (q_token * kHeads + head) * kHeadDim + d;
        context[out_index] = RoundBfloat16(acc);
      }
    }
  }
  return context;
}

std::vector<rocblas_bfloat16> F32ToBf16(const std::vector<float>& input) {
  std::vector<rocblas_bfloat16> out(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    out[i] = rocblas_bfloat16(input[i]);
  }
  return out;
}

std::vector<float> Bf16ToF32(const std::vector<rocblas_bfloat16>& input) {
  std::vector<float> out(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    out[i] = static_cast<float>(input[i]);
  }
  return out;
}

Metrics Compare(const std::vector<float>& actual,
                const std::vector<float>& expected) {
  Metrics metrics;
  double sum_sq = 0.0;
  double sum_abs = 0.0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const double diff = static_cast<double>(actual[i]) - expected[i];
    const double abs_diff = std::abs(diff);
    metrics.max_abs = std::max(metrics.max_abs, abs_diff);
    sum_sq += diff * diff;
    sum_abs += abs_diff;
  }
  metrics.rms = std::sqrt(sum_sq / actual.size());
  metrics.mean_abs = sum_abs / actual.size();
  return metrics;
}

void PrintMetrics(const char* name, const Metrics& metrics) {
  std::printf("%s max_abs=%.9g rms=%.9g mean_abs=%.9g\n", name,
              metrics.max_abs, metrics.rms, metrics.mean_abs);
}

void PrintMetrics(const std::string& name, const Metrics& metrics) {
  PrintMetrics(name.c_str(), metrics);
}

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  size_t count = 0;

  explicit DeviceBuffer(size_t count_in) : count(count_in) {
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
  }

  ~DeviceBuffer() {
    if (ptr != nullptr) {
      static_cast<void>(hipFree(ptr));
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename T>
void CopyToDevice(DeviceBuffer<T>& dst, const std::vector<T>& src) {
  if (dst.count != src.size()) {
    std::fprintf(stderr, "device copy size mismatch\n");
    std::exit(1);
  }
  HIP_CHECK(hipMemcpy(dst.ptr, src.data(), src.size() * sizeof(T),
                      hipMemcpyHostToDevice));
}

template <typename T>
void CopyToHost(std::vector<T>& dst, const DeviceBuffer<T>& src) {
  dst.resize(src.count);
  HIP_CHECK(hipMemcpy(dst.data(), src.ptr, src.count * sizeof(T),
                      hipMemcpyDeviceToHost));
}

void LaunchAddBiasToBf16(const DeviceBuffer<float>& x,
                         const DeviceBuffer<float>& bias,
                         DeviceBuffer<rocblas_bfloat16>& out, int rows,
                         int cols) {
  constexpr int kThreads = 256;
  const int count = rows * cols;
  const int blocks = (count + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(AddBiasToBf16Kernel, dim3(blocks), dim3(kThreads), 0,
                     nullptr, x.ptr, bias.ptr, out.ptr, rows, cols);
  HIP_CHECK(hipGetLastError());
}

void LaunchAddResidualBf16ToF32(const DeviceBuffer<float>& residual,
                                const DeviceBuffer<rocblas_bfloat16>& branch,
                                DeviceBuffer<float>& out, int count) {
  constexpr int kThreads = 256;
  const int blocks = (count + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(AddResidualBf16ToF32Kernel, dim3(blocks), dim3(kThreads),
                     0, nullptr, residual.ptr, branch.ptr, out.ptr, count);
  HIP_CHECK(hipGetLastError());
}

float RunGeluOnceMs(DeviceBuffer<rocblas_bfloat16>& values, int count) {
  constexpr int kThreads = 256;
  const int blocks = (count + kThreads - 1) / kThreads;
  auto run_once = [&]() {
    hipLaunchKernelGGL(GeluBf16Kernel, dim3(blocks), dim3(kThreads), 0,
                       nullptr, values.ptr, count);
    HIP_CHECK(hipGetLastError());
  };

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, nullptr));
  run_once();
  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return elapsed_ms;
}

float RunBf16RowMajorGemmRawMs(rocblas_handle handle,
                               const rocblas_bfloat16* a_mk,
                               const rocblas_bfloat16* b_kn, float* c_mn,
                               size_t rows, size_t in_dim, size_t out_dim,
                               int warmup, int runs) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  auto run_once = [&]() {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvariadic-macro-arguments-omitted"
#endif
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none,
        static_cast<rocblas_int>(out_dim), static_cast<rocblas_int>(rows),
        static_cast<rocblas_int>(in_dim), &alpha, b_kn,
        rocblas_datatype_bf16_r, static_cast<rocblas_int>(out_dim), a_mk,
        rocblas_datatype_bf16_r, static_cast<rocblas_int>(in_dim), &beta, c_mn,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(out_dim), c_mn,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(out_dim),
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, rocblas_gemm_flags_none));
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
  };

  for (int i = 0; i < warmup; ++i) run_once();
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, nullptr));
  for (int i = 0; i < runs; ++i) run_once();
  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return elapsed_ms / static_cast<float>(runs);
}

float RunBf16RowMajorGemmMs(rocblas_handle handle,
                            const DeviceBuffer<rocblas_bfloat16>& a_mk,
                            const DeviceBuffer<rocblas_bfloat16>& b_kn,
                            DeviceBuffer<float>& c_mn, size_t rows,
                            size_t in_dim, size_t out_dim, int warmup,
                            int runs) {
  return RunBf16RowMajorGemmRawMs(handle, a_mk.ptr, b_kn.ptr, c_mn.ptr, rows,
                                  in_dim, out_dim, warmup, runs);
}

float RunLayerNormRawMs(const float* x, const float* scale, const float* bias,
                        rocblas_bfloat16* out, int warmup, int runs) {
  constexpr int kThreads = 256;
  const size_t shared_bytes = kThreads * 2 * sizeof(float);
  auto run_once = [&]() {
    hipLaunchKernelGGL(LayerNormToBf16Kernel, dim3(kTokens), dim3(kThreads),
                       shared_bytes, nullptr, x, scale, bias, out,
                       static_cast<int>(kTokens), static_cast<int>(kWidth));
    HIP_CHECK(hipGetLastError());
  };

  for (int i = 0; i < warmup; ++i) run_once();
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, nullptr));
  for (int i = 0; i < runs; ++i) run_once();
  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return elapsed_ms / static_cast<float>(runs);
}

float RunLayerNormMs(const DeviceBuffer<float>& x,
                     const DeviceBuffer<float>& scale,
                     const DeviceBuffer<float>& bias,
                     DeviceBuffer<rocblas_bfloat16>& out, int warmup,
                     int runs) {
  return RunLayerNormRawMs(x.ptr, scale.ptr, bias.ptr, out.ptr, warmup, runs);
}

float RunAttentionMs(const DeviceBuffer<rocblas_bfloat16>& q,
                     const DeviceBuffer<rocblas_bfloat16>& k,
                     const DeviceBuffer<rocblas_bfloat16>& v,
                     DeviceBuffer<rocblas_bfloat16>& out,
                     const std::string& attention, int warmup, int runs) {
  auto run_once = [&]() {
    if (attention == "parallel") {
      hipLaunchKernelGGL(AttentionBf16ParallelKernel, dim3(kTokens, kHeads),
                         dim3(kTokens), 0, nullptr, q.ptr, k.ptr, v.ptr,
                         out.ptr);
    } else if (attention == "hybrid") {
      hipLaunchKernelGGL(AttentionBf16HybridKernel, dim3(kTokens, kHeads),
                         dim3(kTokens), 0, nullptr, q.ptr, k.ptr, v.ptr,
                         out.ptr);
    } else {
      hipLaunchKernelGGL(AttentionBf16Kernel, dim3(kTokens, kHeads), dim3(1),
                         0, nullptr, q.ptr, k.ptr, v.ptr, out.ptr);
    }
    HIP_CHECK(hipGetLastError());
  };

  for (int i = 0; i < warmup; ++i) run_once();
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, nullptr));
  for (int i = 0; i < runs; ++i) run_once();
  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return elapsed_ms / static_cast<float>(runs);
}

ProjectionResult RunProjectionCheck(
    const char* label, rocblas_handle handle,
    const DeviceBuffer<rocblas_bfloat16>& d_input,
    const std::vector<float>& input, size_t rows, size_t in_dim,
    const std::vector<float>& kernel, const std::vector<float>& bias,
    size_t out_dim, const Args& args,
    DeviceBuffer<rocblas_bfloat16>& d_biased_out) {
  const std::vector<float> cpu =
      DenseBfloat16(input, rows, in_dim, kernel, bias, out_dim);
  const std::vector<rocblas_bfloat16> kernel_bf16 = F32ToBf16(kernel);

  DeviceBuffer<rocblas_bfloat16> d_kernel(kernel_bf16.size());
  DeviceBuffer<float> d_bias(bias.size());
  DeviceBuffer<float> d_out(rows * out_dim);
  CopyToDevice(d_kernel, kernel_bf16);
  CopyToDevice(d_bias, bias);

  ProjectionResult result;
  result.cpu = cpu;
  result.gemm_ms = RunBf16RowMajorGemmMs(handle, d_input, d_kernel, d_out,
                                         rows, in_dim, out_dim, args.warmup,
                                         args.runs);
  LaunchAddBiasToBf16(d_out, d_bias, d_biased_out, static_cast<int>(rows),
                      static_cast<int>(out_dim));

  std::vector<rocblas_bfloat16> hip_bf16;
  CopyToHost(hip_bf16, d_biased_out);

  result.metrics = Compare(Bf16ToF32(hip_bf16), cpu);
  const std::string metric_name =
      std::string("hip_block00_") + label + "_vs_cpu_reference";
  PrintMetrics(metric_name, result.metrics);
  std::printf("hip_block00_%s_gemm_ms=%.6g rows=%zu in=%zu "
              "out=%zu runs=%d warmup=%d\n",
              label, result.gemm_ms, rows, in_dim, out_dim, args.runs,
              args.warmup);
  return result;
}

void PrintUsage() {
  std::fprintf(stderr,
               "usage: openpi_image_encoder_hip_probe --bundle DIR "
               "[--attention serial|hybrid|parallel] [--strict 0|1] [--tol F] "
               "[--warmup N] [--runs N]\n");
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, args)) {
    PrintUsage();
    return 2;
  }
  std::printf("attention_mode %s\n", args.attention.c_str());

  gcpp::experimental::OpenPiImageEncoderRawBundle bundle;
  if (!gcpp::experimental::LoadOpenPiImageEncoderRawBundle(args.bundle,
                                                           bundle)) {
    return 1;
  }
  const std::vector<float>& image = bundle.image;
  const std::vector<float>& golden_stem = bundle.golden_stem;
  const std::vector<float>& with_posemb = bundle.golden_with_posemb;
  const std::vector<float>& block_ln0_scale_all =
      bundle.block_ln0_scale_all;
  const std::vector<float>& block_ln0_bias_all = bundle.block_ln0_bias_all;
  const std::vector<float>& block_q_kernel_all = bundle.block_q_kernel_all;
  const std::vector<float>& block_q_bias_all = bundle.block_q_bias_all;
  const std::vector<float>& block_k_kernel_all = bundle.block_k_kernel_all;
  const std::vector<float>& block_k_bias_all = bundle.block_k_bias_all;
  const std::vector<float>& block_v_kernel_all = bundle.block_v_kernel_all;
  const std::vector<float>& block_v_bias_all = bundle.block_v_bias_all;
  const std::vector<float>& block_attn_out_kernel_all =
      bundle.block_attn_out_kernel_all;
  const std::vector<float>& block_attn_out_bias_all =
      bundle.block_attn_out_bias_all;
  const std::vector<float>& block_ln1_scale_all =
      bundle.block_ln1_scale_all;
  const std::vector<float>& block_ln1_bias_all = bundle.block_ln1_bias_all;
  const std::vector<float>& block_mlp_up_kernel_all =
      bundle.block_mlp_up_kernel_all;
  const std::vector<float>& block_mlp_up_bias_all =
      bundle.block_mlp_up_bias_all;
  const std::vector<float>& block_mlp_down_kernel_all =
      bundle.block_mlp_down_kernel_all;
  const std::vector<float>& block_mlp_down_bias_all =
      bundle.block_mlp_down_bias_all;
  const std::vector<float>& golden_encoded = bundle.golden_encoded;
  const std::vector<float>& golden_tokens = bundle.golden_tokens;
  const std::vector<float>& golden_block00_sa = bundle.golden_block00_sa;
  const std::vector<float>& golden_block00_plus_sa =
      bundle.golden_block00_plus_sa;
  const std::vector<float>& golden_block00_mlp = bundle.golden_block00_mlp;
  const std::vector<float>& golden_block00_plus_mlp =
      bundle.golden_block00_plus_mlp;

  const std::vector<float> ln0_scale =
      SliceLayer(block_ln0_scale_all, 0, kWidth);
  const std::vector<float> ln0_bias =
      SliceLayer(block_ln0_bias_all, 0, kWidth);
  const std::vector<float> q_kernel =
      SliceLayer(block_q_kernel_all, 0, kWidth * kQDim);
  const std::vector<float> q_bias = SliceLayer(block_q_bias_all, 0, kQDim);
  const std::vector<float> k_kernel =
      SliceLayer(block_k_kernel_all, 0, kWidth * kQDim);
  const std::vector<float> k_bias = SliceLayer(block_k_bias_all, 0, kQDim);
  const std::vector<float> v_kernel =
      SliceLayer(block_v_kernel_all, 0, kWidth * kQDim);
  const std::vector<float> v_bias = SliceLayer(block_v_bias_all, 0, kQDim);
  const std::vector<float> attn_out_kernel =
      SliceLayer(block_attn_out_kernel_all, 0, kQDim * kWidth);
  const std::vector<float> attn_out_bias =
      SliceLayer(block_attn_out_bias_all, 0, kWidth);
  const std::vector<float> ln1_scale =
      SliceLayer(block_ln1_scale_all, 0, kWidth);
  const std::vector<float> ln1_bias =
      SliceLayer(block_ln1_bias_all, 0, kWidth);
  const std::vector<float> mlp_up_kernel =
      SliceLayer(block_mlp_up_kernel_all, 0, kWidth * kMlpDim);
  const std::vector<float> mlp_up_bias =
      SliceLayer(block_mlp_up_bias_all, 0, kMlpDim);
  const std::vector<float> mlp_down_kernel =
      SliceLayer(block_mlp_down_kernel_all, 0, kMlpDim * kWidth);
  const std::vector<float> mlp_down_bias =
      SliceLayer(block_mlp_down_bias_all, 0, kWidth);

  const std::vector<float> ln0 =
      LayerNormBfloat16(with_posemb, ln0_scale, ln0_bias);

  DeviceBuffer<float> d_with_posemb(with_posemb.size());
  DeviceBuffer<float> d_ln0_scale(ln0_scale.size());
  DeviceBuffer<float> d_ln0_bias(ln0_bias.size());
  DeviceBuffer<rocblas_bfloat16> d_ln0(ln0.size());
  CopyToDevice(d_with_posemb, with_posemb);
  CopyToDevice(d_ln0_scale, ln0_scale);
  CopyToDevice(d_ln0_bias, ln0_bias);
  const float ln0_ms = RunLayerNormMs(d_with_posemb, d_ln0_scale, d_ln0_bias,
                                      d_ln0, args.warmup, args.runs);

  std::vector<rocblas_bfloat16> ln0_hip_bf16;
  CopyToHost(ln0_hip_bf16, d_ln0);
  const Metrics ln0_metrics = Compare(Bf16ToF32(ln0_hip_bf16), ln0);
  PrintMetrics("hip_block00_ln0_vs_cpu_reference", ln0_metrics);
  std::printf("hip_block00_ln0_ms=%.6g rows=%zu width=%zu runs=%d warmup=%d\n",
              ln0_ms, kTokens, kWidth, args.runs, args.warmup);

  rocblas_handle handle = nullptr;
  ROCBLAS_CHECK(rocblas_create_handle(&handle));
  DeviceBuffer<rocblas_bfloat16> d_q(kTokens * kQDim);
  DeviceBuffer<rocblas_bfloat16> d_k(kTokens * kQDim);
  DeviceBuffer<rocblas_bfloat16> d_v(kTokens * kQDim);
  const ProjectionResult q = RunProjectionCheck(
      "q_projection", handle, d_ln0, ln0, kTokens, kWidth, q_kernel, q_bias,
      kQDim, args, d_q);
  const ProjectionResult k = RunProjectionCheck(
      "k_projection", handle, d_ln0, ln0, kTokens, kWidth, k_kernel, k_bias,
      kQDim, args, d_k);
  const ProjectionResult v = RunProjectionCheck(
      "v_projection", handle, d_ln0, ln0, kTokens, kWidth, v_kernel, v_bias,
      kQDim, args, d_v);

  const float total_projection_ms = q.gemm_ms + k.gemm_ms + v.gemm_ms;
  std::printf("hip_block00_qkv_projection_total_gemm_ms=%.6g\n",
              total_projection_ms);

  const std::vector<float>& q_cpu = q.cpu;
  const std::vector<float>& k_cpu = k.cpu;
  const std::vector<float>& v_cpu = v.cpu;
  const std::vector<float> attn_cpu = AttentionBfloat16(q_cpu, k_cpu, v_cpu);

  const std::vector<rocblas_bfloat16> q_cpu_bf16 = F32ToBf16(q_cpu);
  const std::vector<rocblas_bfloat16> k_cpu_bf16 = F32ToBf16(k_cpu);
  const std::vector<rocblas_bfloat16> v_cpu_bf16 = F32ToBf16(v_cpu);
  DeviceBuffer<rocblas_bfloat16> d_q_cpu(q_cpu_bf16.size());
  DeviceBuffer<rocblas_bfloat16> d_k_cpu(k_cpu_bf16.size());
  DeviceBuffer<rocblas_bfloat16> d_v_cpu(v_cpu_bf16.size());
  DeviceBuffer<rocblas_bfloat16> d_attn_cpu_qkv(kTokens * kQDim);
  CopyToDevice(d_q_cpu, q_cpu_bf16);
  CopyToDevice(d_k_cpu, k_cpu_bf16);
  CopyToDevice(d_v_cpu, v_cpu_bf16);
  const float attn_cpu_qkv_ms =
      RunAttentionMs(d_q_cpu, d_k_cpu, d_v_cpu, d_attn_cpu_qkv,
                     args.attention, args.warmup, args.runs);
  std::vector<rocblas_bfloat16> attn_cpu_qkv_hip_bf16;
  CopyToHost(attn_cpu_qkv_hip_bf16, d_attn_cpu_qkv);
  const Metrics attn_cpu_qkv_metrics =
      Compare(Bf16ToF32(attn_cpu_qkv_hip_bf16), attn_cpu);
  PrintMetrics("hip_block00_attention_cpu_qkv_vs_cpu_reference",
               attn_cpu_qkv_metrics);
  std::printf("hip_block00_attention_cpu_qkv_ms=%.6g rows=%zu heads=%zu "
              "head_dim=%zu runs=%d warmup=%d\n",
              attn_cpu_qkv_ms, kTokens, kHeads, kHeadDim, args.runs,
              args.warmup);

  DeviceBuffer<rocblas_bfloat16> d_attn(kTokens * kQDim);
  const float attn_ms =
      RunAttentionMs(d_q, d_k, d_v, d_attn, args.attention, args.warmup,
                     args.runs);
  std::vector<rocblas_bfloat16> attn_hip_bf16;
  CopyToHost(attn_hip_bf16, d_attn);
  const Metrics attn_metrics = Compare(Bf16ToF32(attn_hip_bf16), attn_cpu);
  PrintMetrics("hip_block00_attention_vs_cpu_reference", attn_metrics);
  std::printf("hip_block00_attention_ms=%.6g rows=%zu heads=%zu head_dim=%zu "
              "runs=%d warmup=%d\n",
              attn_ms, kTokens, kHeads, kHeadDim, args.runs, args.warmup);

  const std::vector<float> sa_cpu = DenseBfloat16(
      attn_cpu, kTokens, kQDim, attn_out_kernel, attn_out_bias, kWidth);
  const std::vector<rocblas_bfloat16> attn_out_kernel_bf16 =
      F32ToBf16(attn_out_kernel);
  DeviceBuffer<rocblas_bfloat16> d_attn_out_kernel(
      attn_out_kernel_bf16.size());
  DeviceBuffer<float> d_attn_out_bias(attn_out_bias.size());
  DeviceBuffer<float> d_sa_f32(kTokens * kWidth);
  DeviceBuffer<rocblas_bfloat16> d_sa(kTokens * kWidth);
  CopyToDevice(d_attn_out_kernel, attn_out_kernel_bf16);
  CopyToDevice(d_attn_out_bias, attn_out_bias);
  const float sa_gemm_ms = RunBf16RowMajorGemmMs(
      handle, d_attn, d_attn_out_kernel, d_sa_f32, kTokens, kQDim, kWidth,
      args.warmup, args.runs);
  LaunchAddBiasToBf16(d_sa_f32, d_attn_out_bias, d_sa,
                      static_cast<int>(kTokens), static_cast<int>(kWidth));

  std::vector<rocblas_bfloat16> sa_hip_bf16;
  CopyToHost(sa_hip_bf16, d_sa);
  const std::vector<float> sa_hip = Bf16ToF32(sa_hip_bf16);
  const Metrics sa_cpu_metrics = Compare(sa_hip, sa_cpu);
  const Metrics sa_golden_metrics = Compare(sa_hip, golden_block00_sa);
  PrintMetrics("hip_block00_sa_vs_cpu_reference", sa_cpu_metrics);
  PrintMetrics("hip_block00_sa_vs_golden", sa_golden_metrics);
  std::printf("hip_block00_sa_projection_gemm_ms=%.6g rows=%zu in=%zu "
              "out=%zu runs=%d warmup=%d\n",
              sa_gemm_ms, kTokens, kQDim, kWidth, args.runs, args.warmup);

  const std::vector<float> plus_sa_cpu = AddVectorsBfloat16(with_posemb, sa_cpu);
  DeviceBuffer<float> d_plus_sa_f32(kTokens * kWidth);
  LaunchAddResidualBf16ToF32(d_with_posemb, d_sa, d_plus_sa_f32,
                             static_cast<int>(kTokens * kWidth));
  std::vector<float> plus_sa_hip;
  CopyToHost(plus_sa_hip, d_plus_sa_f32);
  const Metrics plus_sa_cpu_metrics = Compare(plus_sa_hip, plus_sa_cpu);
  const Metrics plus_sa_golden_metrics =
      Compare(plus_sa_hip, golden_block00_plus_sa);
  PrintMetrics("hip_block00_plus_sa_vs_cpu_reference", plus_sa_cpu_metrics);
  PrintMetrics("hip_block00_plus_sa_vs_golden", plus_sa_golden_metrics);

  const std::vector<float> ln1 =
      LayerNormBfloat16(plus_sa_hip, ln1_scale, ln1_bias);
  DeviceBuffer<float> d_ln1_scale(ln1_scale.size());
  DeviceBuffer<float> d_ln1_bias(ln1_bias.size());
  DeviceBuffer<rocblas_bfloat16> d_ln1(kTokens * kWidth);
  CopyToDevice(d_ln1_scale, ln1_scale);
  CopyToDevice(d_ln1_bias, ln1_bias);
  const float ln1_ms = RunLayerNormMs(d_plus_sa_f32, d_ln1_scale, d_ln1_bias,
                                      d_ln1, args.warmup, args.runs);
  std::vector<rocblas_bfloat16> ln1_hip_bf16;
  CopyToHost(ln1_hip_bf16, d_ln1);
  const Metrics ln1_metrics = Compare(Bf16ToF32(ln1_hip_bf16), ln1);
  PrintMetrics("hip_block00_ln1_vs_cpu_reference", ln1_metrics);
  std::printf("hip_block00_ln1_ms=%.6g rows=%zu width=%zu runs=%d warmup=%d\n",
              ln1_ms, kTokens, kWidth, args.runs, args.warmup);

  DeviceBuffer<rocblas_bfloat16> d_mlp_up(kTokens * kMlpDim);
  ProjectionResult mlp_up = RunProjectionCheck(
      "mlp_up", handle, d_ln1, ln1, kTokens, kWidth, mlp_up_kernel,
      mlp_up_bias, kMlpDim, args, d_mlp_up);
  std::vector<float> mlp_up_gelu_cpu = mlp_up.cpu;
  ApplyGeluBfloat16(mlp_up_gelu_cpu);
  const float gelu_ms =
      RunGeluOnceMs(d_mlp_up, static_cast<int>(kTokens * kMlpDim));
  std::vector<rocblas_bfloat16> mlp_up_gelu_hip_bf16;
  CopyToHost(mlp_up_gelu_hip_bf16, d_mlp_up);
  const Metrics gelu_metrics =
      Compare(Bf16ToF32(mlp_up_gelu_hip_bf16), mlp_up_gelu_cpu);
  PrintMetrics("hip_block00_mlp_up_gelu_vs_cpu_reference", gelu_metrics);
  std::printf("hip_block00_gelu_ms=%.6g elements=%zu\n", gelu_ms,
              kTokens * kMlpDim);

  DeviceBuffer<rocblas_bfloat16> d_mlp(kTokens * kWidth);
  ProjectionResult mlp_down = RunProjectionCheck(
      "mlp_down", handle, d_mlp_up, mlp_up_gelu_cpu, kTokens, kMlpDim,
      mlp_down_kernel, mlp_down_bias, kWidth, args, d_mlp);
  ROCBLAS_CHECK(rocblas_destroy_handle(handle));

  std::vector<rocblas_bfloat16> mlp_hip_bf16;
  CopyToHost(mlp_hip_bf16, d_mlp);
  const std::vector<float> mlp_hip = Bf16ToF32(mlp_hip_bf16);
  const Metrics mlp_golden_metrics = Compare(mlp_hip, golden_block00_mlp);
  PrintMetrics("hip_block00_mlp_vs_golden", mlp_golden_metrics);

  const std::vector<float> plus_mlp_cpu =
      AddVectorsBfloat16(plus_sa_hip, mlp_down.cpu);
  DeviceBuffer<float> d_plus_mlp_f32(kTokens * kWidth);
  LaunchAddResidualBf16ToF32(d_plus_sa_f32, d_mlp, d_plus_mlp_f32,
                             static_cast<int>(kTokens * kWidth));
  std::vector<float> plus_mlp_hip;
  CopyToHost(plus_mlp_hip, d_plus_mlp_f32);
  const Metrics plus_mlp_cpu_metrics =
      Compare(plus_mlp_hip, plus_mlp_cpu);
  const Metrics plus_mlp_golden_metrics =
      Compare(plus_mlp_hip, golden_block00_plus_mlp);
  PrintMetrics("hip_block00_plus_mlp_vs_cpu_reference",
               plus_mlp_cpu_metrics);
  PrintMetrics("hip_block00_plus_mlp_vs_golden", plus_mlp_golden_metrics);

  const std::unique_ptr<gcpp::experimental::OpenPiImageEncoderHipBackend>
      backend =
          gcpp::experimental::CreateOpenPiImageEncoderHipBackendFromRawBundle(
              bundle);
  gcpp::experimental::OpenPiImageEncoderHipOptions backend_options;
  backend_options.attention = args.attention;
  backend_options.warmup = args.warmup;
  backend_options.runs = args.runs;
  const gcpp::experimental::OpenPiImageEncoderHipImageRunResult full_result =
      backend->RunFromImageWithGolden(image, golden_stem, with_posemb,
                                      golden_encoded, golden_tokens,
                                      backend_options);

  const std::vector<double> max_abs_values = {
      ln0_metrics.max_abs,
      q.metrics.max_abs,
      k.metrics.max_abs,
      v.metrics.max_abs,
      attn_cpu_qkv_metrics.max_abs,
      attn_metrics.max_abs,
      sa_golden_metrics.max_abs,
      plus_sa_golden_metrics.max_abs,
      ln1_metrics.max_abs,
      mlp_up.metrics.max_abs,
      gelu_metrics.max_abs,
      mlp_down.metrics.max_abs,
      mlp_golden_metrics.max_abs,
      plus_mlp_golden_metrics.max_abs,
      full_result.stem_metrics.max_abs,
      full_result.posemb_metrics.max_abs,
      full_result.encoded_metrics.max_abs,
      full_result.tokens_metrics.max_abs,
  };
  const double max_abs =
      *std::max_element(max_abs_values.begin(), max_abs_values.end());
  if (args.strict && max_abs > args.tol) {
    std::fprintf(stderr,
                 "hip_block00_full_block max_abs %.9g exceeds tolerance "
                 "%.9g\n",
                 max_abs, args.tol);
    return 1;
  }
  return 0;
}
