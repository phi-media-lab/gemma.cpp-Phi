// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include "experimental/openpi_image_encoder_hip_backend.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace gcpp {
namespace experimental {
namespace {

constexpr size_t kLayers = 27;
constexpr size_t kImageW = 224;
constexpr size_t kChannels = 3;
constexpr size_t kPatch = 14;
constexpr size_t kGrid = 16;
constexpr size_t kTokens = 256;
constexpr size_t kWidth = 1152;
constexpr size_t kHeads = 16;
constexpr size_t kHeadDim = 72;
constexpr size_t kQDim = kHeads * kHeadDim;
constexpr size_t kMlpDim = 4304;
constexpr size_t kDecoderWidth = 2048;

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

__global__ void PatchEmbedPosembKernel(const float* image,
                                       const float* kernel,
                                       const float* bias,
                                       const float* pos_embedding,
                                       float* stem,
                                       float* with_posemb) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = int(kTokens * kWidth);
  if (idx >= count) return;

  const int token = idx / int(kWidth);
  const int out = idx % int(kWidth);
  const int gy = token / int(kGrid);
  const int gx = token % int(kGrid);

  float acc = bias[out];
  for (int py = 0; py < int(kPatch); ++py) {
    for (int px = 0; px < int(kPatch); ++px) {
      for (int c = 0; c < int(kChannels); ++c) {
        const int image_index =
            (((gy * int(kPatch) + py) * int(kImageW) +
              (gx * int(kPatch) + px)) *
             int(kChannels)) +
            c;
        const int kernel_index =
            (((py * int(kPatch) + px) * int(kChannels) + c) *
             int(kWidth)) +
            out;
        acc += image[image_index] * kernel[kernel_index];
      }
    }
  }
  stem[idx] = acc;
  with_posemb[idx] = acc + pos_embedding[idx];
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
    probs[k_token] =
        static_cast<float>(rocblas_bfloat16(probs[k_token] / denom));
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

template <typename T>
const T* DeviceOffset(const DeviceBuffer<T>& buffer, size_t offset) {
  if (offset > buffer.count) {
    std::fprintf(stderr, "device offset out of range\n");
    std::exit(1);
  }
  return buffer.ptr + offset;
}

void CopyF32AsBf16ToDevice(const std::vector<float>& src,
                           DeviceBuffer<rocblas_bfloat16>& dst) {
  const std::vector<rocblas_bfloat16> bf16 = F32ToBf16(src);
  CopyToDevice(dst, bf16);
}

OpenPiImageEncoderHipMetrics Compare(const std::vector<float>& actual,
                                     const std::vector<float>& expected) {
  if (actual.size() != expected.size()) {
    std::fprintf(stderr, "compare size mismatch: got %zu expected %zu\n",
                 actual.size(), expected.size());
    std::exit(1);
  }
  OpenPiImageEncoderHipMetrics metrics;
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

void PrintMetrics(const char* name, const OpenPiImageEncoderHipMetrics& metrics) {
  std::printf("%s max_abs=%.9g rms=%.9g mean_abs=%.9g\n", name,
              metrics.max_abs, metrics.rms, metrics.mean_abs);
}

struct OpenPiImageEncoderResidentWeights {
  DeviceBuffer<float> patch_embed_kernel;
  DeviceBuffer<float> patch_embed_bias;
  DeviceBuffer<float> pos_embedding;
  DeviceBuffer<float> block_ln0_scale;
  DeviceBuffer<float> block_ln0_bias;
  DeviceBuffer<float> block_ln1_scale;
  DeviceBuffer<float> block_ln1_bias;
  DeviceBuffer<float> block_q_bias;
  DeviceBuffer<float> block_k_bias;
  DeviceBuffer<float> block_v_bias;
  DeviceBuffer<float> block_attn_out_bias;
  DeviceBuffer<float> block_mlp_up_bias;
  DeviceBuffer<float> block_mlp_down_bias;
  DeviceBuffer<float> encoder_norm_scale;
  DeviceBuffer<float> encoder_norm_bias;
  DeviceBuffer<float> head_bias;

  DeviceBuffer<rocblas_bfloat16> block_q_kernel;
  DeviceBuffer<rocblas_bfloat16> block_k_kernel;
  DeviceBuffer<rocblas_bfloat16> block_v_kernel;
  DeviceBuffer<rocblas_bfloat16> block_attn_out_kernel;
  DeviceBuffer<rocblas_bfloat16> block_mlp_up_kernel;
  DeviceBuffer<rocblas_bfloat16> block_mlp_down_kernel;
  DeviceBuffer<rocblas_bfloat16> head_kernel;

  OpenPiImageEncoderResidentWeights(
      const std::vector<float>& block_ln0_scale_all,
      const std::vector<float>& block_ln0_bias_all,
      const std::vector<float>& block_ln1_scale_all,
      const std::vector<float>& block_ln1_bias_all,
      const std::vector<float>& block_q_bias_all,
      const std::vector<float>& block_k_bias_all,
      const std::vector<float>& block_v_bias_all,
      const std::vector<float>& block_attn_out_bias_all,
      const std::vector<float>& block_mlp_up_bias_all,
      const std::vector<float>& block_mlp_down_bias_all,
      const std::vector<float>& encoder_norm_scale_host,
      const std::vector<float>& encoder_norm_bias_host,
      const std::vector<float>& head_bias_host,
      const std::vector<float>& block_q_kernel_all,
      const std::vector<float>& block_k_kernel_all,
      const std::vector<float>& block_v_kernel_all,
      const std::vector<float>& block_attn_out_kernel_all,
      const std::vector<float>& block_mlp_up_kernel_all,
      const std::vector<float>& block_mlp_down_kernel_all,
      const std::vector<float>& head_kernel_host,
      const std::vector<float>& patch_embed_kernel_host,
      const std::vector<float>& patch_embed_bias_host,
      const std::vector<float>& pos_embedding_host)
      : patch_embed_kernel(patch_embed_kernel_host.size()),
        patch_embed_bias(patch_embed_bias_host.size()),
        pos_embedding(pos_embedding_host.size()),
        block_ln0_scale(block_ln0_scale_all.size()),
        block_ln0_bias(block_ln0_bias_all.size()),
        block_ln1_scale(block_ln1_scale_all.size()),
        block_ln1_bias(block_ln1_bias_all.size()),
        block_q_bias(block_q_bias_all.size()),
        block_k_bias(block_k_bias_all.size()),
        block_v_bias(block_v_bias_all.size()),
        block_attn_out_bias(block_attn_out_bias_all.size()),
        block_mlp_up_bias(block_mlp_up_bias_all.size()),
        block_mlp_down_bias(block_mlp_down_bias_all.size()),
        encoder_norm_scale(encoder_norm_scale_host.size()),
        encoder_norm_bias(encoder_norm_bias_host.size()),
        head_bias(head_bias_host.size()),
        block_q_kernel(block_q_kernel_all.size()),
        block_k_kernel(block_k_kernel_all.size()),
        block_v_kernel(block_v_kernel_all.size()),
        block_attn_out_kernel(block_attn_out_kernel_all.size()),
        block_mlp_up_kernel(block_mlp_up_kernel_all.size()),
        block_mlp_down_kernel(block_mlp_down_kernel_all.size()),
        head_kernel(head_kernel_host.size()) {
    CopyToDevice(patch_embed_kernel, patch_embed_kernel_host);
    CopyToDevice(patch_embed_bias, patch_embed_bias_host);
    CopyToDevice(pos_embedding, pos_embedding_host);
    CopyToDevice(block_ln0_scale, block_ln0_scale_all);
    CopyToDevice(block_ln0_bias, block_ln0_bias_all);
    CopyToDevice(block_ln1_scale, block_ln1_scale_all);
    CopyToDevice(block_ln1_bias, block_ln1_bias_all);
    CopyToDevice(block_q_bias, block_q_bias_all);
    CopyToDevice(block_k_bias, block_k_bias_all);
    CopyToDevice(block_v_bias, block_v_bias_all);
    CopyToDevice(block_attn_out_bias, block_attn_out_bias_all);
    CopyToDevice(block_mlp_up_bias, block_mlp_up_bias_all);
    CopyToDevice(block_mlp_down_bias, block_mlp_down_bias_all);
    CopyToDevice(encoder_norm_scale, encoder_norm_scale_host);
    CopyToDevice(encoder_norm_bias, encoder_norm_bias_host);
    CopyToDevice(head_bias, head_bias_host);

    CopyF32AsBf16ToDevice(block_q_kernel_all, block_q_kernel);
    CopyF32AsBf16ToDevice(block_k_kernel_all, block_k_kernel);
    CopyF32AsBf16ToDevice(block_v_kernel_all, block_v_kernel);
    CopyF32AsBf16ToDevice(block_attn_out_kernel_all, block_attn_out_kernel);
    CopyF32AsBf16ToDevice(block_mlp_up_kernel_all, block_mlp_up_kernel);
    CopyF32AsBf16ToDevice(block_mlp_down_kernel_all, block_mlp_down_kernel);
    CopyF32AsBf16ToDevice(head_kernel_host, head_kernel);
  }
};

void LaunchAddBiasToBf16Raw(const float* x, const float* bias,
                            rocblas_bfloat16* out, int rows, int cols) {
  constexpr int kThreads = 256;
  const int count = rows * cols;
  const int blocks = (count + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(AddBiasToBf16Kernel, dim3(blocks), dim3(kThreads), 0,
                     nullptr, x, bias, out, rows, cols);
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

struct PatchEmbedPosembOutput {
  std::vector<float> stem;
  std::vector<float> with_posemb;
  float patch_embed_ms = 0.0f;
};

PatchEmbedPosembOutput RunPatchEmbedPosemb(
    const OpenPiImageEncoderResidentWeights& weights,
    const std::vector<float>& image,
    DeviceBuffer<float>* d_with_posemb_out,
    bool copy_host_outputs) {
  DeviceBuffer<float> d_image(image.size());
  DeviceBuffer<float> d_stem(kTokens * kWidth);
  DeviceBuffer<float> d_with_posemb(kTokens * kWidth);
  CopyToDevice(d_image, image);

  constexpr int kThreads = 256;
  const int count = static_cast<int>(kTokens * kWidth);
  const int blocks = (count + kThreads - 1) / kThreads;

  auto run_once = [&]() {
    hipLaunchKernelGGL(PatchEmbedPosembKernel, dim3(blocks), dim3(kThreads),
                       0, nullptr, d_image.ptr, weights.patch_embed_kernel.ptr,
                       weights.patch_embed_bias.ptr, weights.pos_embedding.ptr,
                       d_stem.ptr, d_with_posemb.ptr);
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

  PatchEmbedPosembOutput output;
  HIP_CHECK(hipEventElapsedTime(&output.patch_embed_ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  if (d_with_posemb_out != nullptr) {
    if (d_with_posemb_out->count != d_with_posemb.count) {
      std::fprintf(stderr, "with_posemb device output size mismatch\n");
      std::exit(1);
    }
    HIP_CHECK(hipMemcpy(d_with_posemb_out->ptr, d_with_posemb.ptr,
                        d_with_posemb.count * sizeof(float),
                        hipMemcpyDeviceToDevice));
  }

  if (copy_host_outputs) {
    CopyToHost(output.stem, d_stem);
    CopyToHost(output.with_posemb, d_with_posemb);
  }
  return output;
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

float RunProjectionResident(rocblas_handle handle,
                            const DeviceBuffer<rocblas_bfloat16>& d_input,
                            const rocblas_bfloat16* d_kernel,
                            const float* d_bias, size_t rows, size_t in_dim,
                            size_t out_dim,
                            DeviceBuffer<rocblas_bfloat16>& d_biased_out) {
  DeviceBuffer<float> d_out(rows * out_dim);
  const float gemm_ms = RunBf16RowMajorGemmRawMs(
      handle, d_input.ptr, d_kernel, d_out.ptr, rows, in_dim, out_dim,
      /*warmup=*/0, /*runs=*/1);
  LaunchAddBiasToBf16Raw(d_out.ptr, d_bias, d_biased_out.ptr,
                         static_cast<int>(rows), static_cast<int>(out_dim));
  return gemm_ms;
}

void PrintTiming(const OpenPiImageEncoderHipOutput& result) {
  std::printf("hip_all_layers_timing_ms ln=%.6g qkv=%.6g attention=%.6g "
              "sa=%.6g mlp_up=%.6g gelu=%.6g mlp_down=%.6g "
              "final_norm=%.6g head=%.6g\n",
              result.ln_ms, result.qkv_ms, result.attention_ms, result.sa_ms,
              result.mlp_up_ms, result.gelu_ms, result.mlp_down_ms,
              result.final_norm_ms, result.head_ms);
}

OpenPiImageEncoderHipOutput RunFullImageEncoderResidentDeviceInput(
    const OpenPiImageEncoderResidentWeights& weights,
    const DeviceBuffer<float>& d_with_posemb,
    const OpenPiImageEncoderHipOptions& options) {
  OpenPiImageEncoderHipOutput result;
  rocblas_handle full_handle = nullptr;
  ROCBLAS_CHECK(rocblas_create_handle(&full_handle));

  DeviceBuffer<float> d_full_current(kTokens * kWidth);
  DeviceBuffer<float> d_full_next(kTokens * kWidth);
  DeviceBuffer<float> d_full_plus_sa(kTokens * kWidth);
  DeviceBuffer<rocblas_bfloat16> d_full_ln0(kTokens * kWidth);
  DeviceBuffer<rocblas_bfloat16> d_full_ln1(kTokens * kWidth);
  DeviceBuffer<rocblas_bfloat16> d_full_q(kTokens * kQDim);
  DeviceBuffer<rocblas_bfloat16> d_full_k(kTokens * kQDim);
  DeviceBuffer<rocblas_bfloat16> d_full_v(kTokens * kQDim);
  DeviceBuffer<rocblas_bfloat16> d_full_attn(kTokens * kQDim);
  DeviceBuffer<rocblas_bfloat16> d_full_sa(kTokens * kWidth);
  DeviceBuffer<rocblas_bfloat16> d_full_mlp_up(kTokens * kMlpDim);
  DeviceBuffer<rocblas_bfloat16> d_full_mlp(kTokens * kWidth);
  HIP_CHECK(hipMemcpy(d_full_current.ptr, d_with_posemb.ptr,
                      d_full_current.count * sizeof(float),
                      hipMemcpyDeviceToDevice));

  DeviceBuffer<float>* d_layer_in = &d_full_current;
  DeviceBuffer<float>* d_layer_out = &d_full_next;
  for (size_t layer = 0; layer < kLayers; ++layer) {
    result.ln_ms += RunLayerNormRawMs(
        d_layer_in->ptr, DeviceOffset(weights.block_ln0_scale, layer * kWidth),
        DeviceOffset(weights.block_ln0_bias, layer * kWidth), d_full_ln0.ptr,
        /*warmup=*/0, /*runs=*/1);

    result.qkv_ms += RunProjectionResident(
        full_handle, d_full_ln0,
        DeviceOffset(weights.block_q_kernel, layer * kWidth * kQDim),
        DeviceOffset(weights.block_q_bias, layer * kQDim), kTokens, kWidth,
        kQDim, d_full_q);
    result.qkv_ms += RunProjectionResident(
        full_handle, d_full_ln0,
        DeviceOffset(weights.block_k_kernel, layer * kWidth * kQDim),
        DeviceOffset(weights.block_k_bias, layer * kQDim), kTokens, kWidth,
        kQDim, d_full_k);
    result.qkv_ms += RunProjectionResident(
        full_handle, d_full_ln0,
        DeviceOffset(weights.block_v_kernel, layer * kWidth * kQDim),
        DeviceOffset(weights.block_v_bias, layer * kQDim), kTokens, kWidth,
        kQDim, d_full_v);

    result.attention_ms += RunAttentionMs(d_full_q, d_full_k, d_full_v,
                                          d_full_attn, options.attention,
                                          /*warmup=*/0, /*runs=*/1);

    result.sa_ms += RunProjectionResident(
        full_handle, d_full_attn,
        DeviceOffset(weights.block_attn_out_kernel, layer * kQDim * kWidth),
        DeviceOffset(weights.block_attn_out_bias, layer * kWidth), kTokens,
        kQDim, kWidth, d_full_sa);
    LaunchAddResidualBf16ToF32(*d_layer_in, d_full_sa, d_full_plus_sa,
                               static_cast<int>(kTokens * kWidth));

    result.ln_ms += RunLayerNormRawMs(
        d_full_plus_sa.ptr,
        DeviceOffset(weights.block_ln1_scale, layer * kWidth),
        DeviceOffset(weights.block_ln1_bias, layer * kWidth), d_full_ln1.ptr,
        /*warmup=*/0, /*runs=*/1);

    result.mlp_up_ms += RunProjectionResident(
        full_handle, d_full_ln1,
        DeviceOffset(weights.block_mlp_up_kernel, layer * kWidth * kMlpDim),
        DeviceOffset(weights.block_mlp_up_bias, layer * kMlpDim), kTokens,
        kWidth, kMlpDim, d_full_mlp_up);
    result.gelu_ms +=
        RunGeluOnceMs(d_full_mlp_up, static_cast<int>(kTokens * kMlpDim));
    result.mlp_down_ms += RunProjectionResident(
        full_handle, d_full_mlp_up,
        DeviceOffset(weights.block_mlp_down_kernel, layer * kMlpDim * kWidth),
        DeviceOffset(weights.block_mlp_down_bias, layer * kWidth), kTokens,
        kMlpDim, kWidth, d_full_mlp);
    LaunchAddResidualBf16ToF32(d_full_plus_sa, d_full_mlp, *d_layer_out,
                               static_cast<int>(kTokens * kWidth));
    std::swap(d_layer_in, d_layer_out);
  }

  DeviceBuffer<rocblas_bfloat16> d_encoded(kTokens * kWidth);
  result.final_norm_ms = RunLayerNormRawMs(
      d_layer_in->ptr, weights.encoder_norm_scale.ptr,
      weights.encoder_norm_bias.ptr, d_encoded.ptr, /*warmup=*/0, /*runs=*/1);
  std::vector<rocblas_bfloat16> encoded_hip_bf16;
  CopyToHost(encoded_hip_bf16, d_encoded);
  result.encoded = Bf16ToF32(encoded_hip_bf16);

  DeviceBuffer<rocblas_bfloat16> d_tokens(kTokens * kDecoderWidth);
  result.head_ms =
      RunProjectionResident(full_handle, d_encoded, weights.head_kernel.ptr,
                            weights.head_bias.ptr, kTokens, kWidth,
                            kDecoderWidth, d_tokens);
  ROCBLAS_CHECK(rocblas_destroy_handle(full_handle));
  std::vector<rocblas_bfloat16> tokens_hip_bf16;
  CopyToHost(tokens_hip_bf16, d_tokens);
  result.tokens = Bf16ToF32(tokens_hip_bf16);
  return result;
}

OpenPiImageEncoderHipOutput RunFullImageEncoderResident(
    const OpenPiImageEncoderResidentWeights& weights,
    const std::vector<float>& with_posemb,
    const OpenPiImageEncoderHipOptions& options) {
  DeviceBuffer<float> d_with_posemb(kTokens * kWidth);
  CopyToDevice(d_with_posemb, with_posemb);
  return RunFullImageEncoderResidentDeviceInput(weights, d_with_posemb,
                                                options);
}

}  // namespace

struct OpenPiImageEncoderHipBackend::Impl {
  OpenPiImageEncoderResidentWeights weights;

  Impl(const std::vector<float>& block_ln0_scale_all,
       const std::vector<float>& block_ln0_bias_all,
       const std::vector<float>& block_ln1_scale_all,
       const std::vector<float>& block_ln1_bias_all,
       const std::vector<float>& block_q_bias_all,
       const std::vector<float>& block_k_bias_all,
       const std::vector<float>& block_v_bias_all,
       const std::vector<float>& block_attn_out_bias_all,
       const std::vector<float>& block_mlp_up_bias_all,
       const std::vector<float>& block_mlp_down_bias_all,
       const std::vector<float>& encoder_norm_scale,
       const std::vector<float>& encoder_norm_bias,
       const std::vector<float>& head_bias,
       const std::vector<float>& block_q_kernel_all,
       const std::vector<float>& block_k_kernel_all,
       const std::vector<float>& block_v_kernel_all,
       const std::vector<float>& block_attn_out_kernel_all,
       const std::vector<float>& block_mlp_up_kernel_all,
       const std::vector<float>& block_mlp_down_kernel_all,
       const std::vector<float>& head_kernel,
       const std::vector<float>& patch_embed_kernel,
       const std::vector<float>& patch_embed_bias,
       const std::vector<float>& pos_embedding)
      : weights(block_ln0_scale_all, block_ln0_bias_all, block_ln1_scale_all,
                block_ln1_bias_all, block_q_bias_all, block_k_bias_all,
                block_v_bias_all, block_attn_out_bias_all,
                block_mlp_up_bias_all, block_mlp_down_bias_all,
                encoder_norm_scale, encoder_norm_bias, head_bias,
                block_q_kernel_all, block_k_kernel_all, block_v_kernel_all,
                block_attn_out_kernel_all, block_mlp_up_kernel_all,
                block_mlp_down_kernel_all, head_kernel, patch_embed_kernel,
                patch_embed_bias, pos_embedding) {}
};

OpenPiImageEncoderHipBackend::OpenPiImageEncoderHipBackend(
    const std::vector<float>& block_ln0_scale_all,
    const std::vector<float>& block_ln0_bias_all,
    const std::vector<float>& block_ln1_scale_all,
    const std::vector<float>& block_ln1_bias_all,
    const std::vector<float>& block_q_bias_all,
    const std::vector<float>& block_k_bias_all,
    const std::vector<float>& block_v_bias_all,
    const std::vector<float>& block_attn_out_bias_all,
    const std::vector<float>& block_mlp_up_bias_all,
    const std::vector<float>& block_mlp_down_bias_all,
    const std::vector<float>& encoder_norm_scale,
    const std::vector<float>& encoder_norm_bias,
    const std::vector<float>& head_bias,
    const std::vector<float>& block_q_kernel_all,
    const std::vector<float>& block_k_kernel_all,
    const std::vector<float>& block_v_kernel_all,
    const std::vector<float>& block_attn_out_kernel_all,
    const std::vector<float>& block_mlp_up_kernel_all,
    const std::vector<float>& block_mlp_down_kernel_all,
    const std::vector<float>& head_kernel,
    const std::vector<float>& patch_embed_kernel,
    const std::vector<float>& patch_embed_bias,
    const std::vector<float>& pos_embedding)
    : impl_(std::make_unique<Impl>(
          block_ln0_scale_all, block_ln0_bias_all, block_ln1_scale_all,
          block_ln1_bias_all, block_q_bias_all, block_k_bias_all,
          block_v_bias_all, block_attn_out_bias_all, block_mlp_up_bias_all,
          block_mlp_down_bias_all, encoder_norm_scale, encoder_norm_bias,
          head_bias, block_q_kernel_all, block_k_kernel_all,
          block_v_kernel_all, block_attn_out_kernel_all,
          block_mlp_up_kernel_all, block_mlp_down_kernel_all, head_kernel,
          patch_embed_kernel, patch_embed_bias, pos_embedding)) {}

OpenPiImageEncoderHipBackend::~OpenPiImageEncoderHipBackend() = default;

OpenPiImageEncoderHipOutput OpenPiImageEncoderHipBackend::Run(
    const std::vector<float>& with_posemb,
    const OpenPiImageEncoderHipOptions& options) const {
  return RunFullImageEncoderResident(impl_->weights, with_posemb, options);
}

OpenPiImageEncoderHipOutput OpenPiImageEncoderHipBackend::RunFromImage(
    const std::vector<float>& image,
    const OpenPiImageEncoderHipOptions& options) const {
  DeviceBuffer<float> d_with_posemb(kTokens * kWidth);
  const PatchEmbedPosembOutput patch =
      RunPatchEmbedPosemb(impl_->weights, image, &d_with_posemb,
                          /*copy_host_outputs=*/false);
  OpenPiImageEncoderHipOutput result =
      RunFullImageEncoderResidentDeviceInput(impl_->weights, d_with_posemb,
                                             options);
  result.patch_embed_ms = patch.patch_embed_ms;
  return result;
}

OpenPiImageEncoderHipRunResult OpenPiImageEncoderHipBackend::RunWithGolden(
    const std::vector<float>& with_posemb,
    const std::vector<float>& golden_encoded,
    const std::vector<float>& golden_tokens,
    const OpenPiImageEncoderHipOptions& options) const {
  OpenPiImageEncoderHipRunResult result;
  static_cast<OpenPiImageEncoderHipOutput&>(result) = Run(with_posemb, options);
  result.encoded_metrics = Compare(result.encoded, golden_encoded);
  PrintMetrics("hip_all_layers_encoded_vs_golden", result.encoded_metrics);
  result.tokens_metrics = Compare(result.tokens, golden_tokens);
  PrintMetrics("hip_all_layers_tokens_vs_golden", result.tokens_metrics);
  PrintTiming(result);
  return result;
}

OpenPiImageEncoderHipImageRunResult
OpenPiImageEncoderHipBackend::RunFromImageWithGolden(
    const std::vector<float>& image,
    const std::vector<float>& golden_stem,
    const std::vector<float>& golden_with_posemb,
    const std::vector<float>& golden_encoded,
    const std::vector<float>& golden_tokens,
    const OpenPiImageEncoderHipOptions& options) const {
  DeviceBuffer<float> d_with_posemb(kTokens * kWidth);
  const PatchEmbedPosembOutput patch =
      RunPatchEmbedPosemb(impl_->weights, image, &d_with_posemb,
                          /*copy_host_outputs=*/true);

  OpenPiImageEncoderHipImageRunResult result;
  result.patch_embed_ms = patch.patch_embed_ms;
  result.stem_metrics = Compare(patch.stem, golden_stem);
  PrintMetrics("hip_patch_embed_stem_vs_golden", result.stem_metrics);
  result.posemb_metrics = Compare(patch.with_posemb, golden_with_posemb);
  PrintMetrics("hip_patch_embed_with_posemb_vs_golden",
               result.posemb_metrics);

  static_cast<OpenPiImageEncoderHipOutput&>(result) =
      RunFullImageEncoderResidentDeviceInput(impl_->weights, d_with_posemb,
                                             options);
  result.patch_embed_ms = patch.patch_embed_ms;
  result.encoded_metrics = Compare(result.encoded, golden_encoded);
  PrintMetrics("hip_all_layers_encoded_vs_golden", result.encoded_metrics);
  result.tokens_metrics = Compare(result.tokens, golden_tokens);
  PrintMetrics("hip_all_layers_tokens_vs_golden", result.tokens_metrics);
  std::printf("hip_patch_embed_ms=%.6g\n", result.patch_embed_ms);
  PrintTiming(result);
  return result;
}

}  // namespace experimental
}  // namespace gcpp
