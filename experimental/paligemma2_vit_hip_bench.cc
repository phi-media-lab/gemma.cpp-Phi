// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

#define HIP_CHECK(expr)                                                     \
  do {                                                                      \
    hipError_t status = (expr);                                             \
    if (status != hipSuccess) {                                             \
      std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,     \
                   hipGetErrorString(status));                              \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

#define ROCBLAS_CHECK(expr)                                                 \
  do {                                                                      \
    rocblas_status status = (expr);                                         \
    if (status != rocblas_status_success) {                                 \
      std::fprintf(stderr, "rocBLAS error %s:%d: status=%d\n", __FILE__,    \
                   __LINE__, static_cast<int>(status));                     \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

// Single-kernel prototype for one ViT self-attention layer. The input is laid
// out as qkv[token][head][q/k/v][dim], matching the logical shape produced by a
// QKV projection. One HIP block computes one output token for one head:
//
//   1. compute all Q.K scores for that token/head;
//   2. reduce the max for stable softmax;
//   3. exponentiate and reduce the softmax denominator;
//   4. compute the weighted sum over V.
//
// This is intentionally simple and is used as a negative/feasibility
// measurement. It does not tile Q/K/V, does not use wave-level reductions, and
// rereads V for each output token, so it should not be treated as an optimized
// attention implementation.
__global__ void VitAttentionFusedKernel(const float* __restrict__ qkv,
                                        float* __restrict__ out, int seq,
                                        int heads, int dim, float scale) {
  extern __shared__ float shared[];
  float* scores = shared;
  float* reduce = shared + seq;

  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int block_threads = blockDim.x;

  const float* q =
      qkv + ((token * heads + head) * 3 * dim);

  // First pass: each thread computes a strided subset of Q.K scores and keeps a
  // local max. Scores are kept in shared memory so the later softmax and AV
  // phase can reuse them without writing an intermediate score matrix to global
  // memory.
  float local_max = -3.4028234663852886e38f;
  for (int key = tid; key < seq; key += block_threads) {
    const float* k = qkv + ((key * heads + head) * 3 * dim + dim);
    float dot = 0.0f;
    for (int d = 0; d < dim; ++d) {
      dot += q[d] * k[d];
    }
    dot *= scale;
    scores[key] = dot;
    local_max = fmaxf(local_max, dot);
  }

  // Block reduction over the local maxima. The benchmark uses power-of-two
  // block sizes, so this simple tree reduction is enough for measurement.
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce[0];

  // Second pass: convert scores to exp(score - max_score), store them back in
  // shared memory, and reduce the denominator for softmax normalization.
  float local_sum = 0.0f;
  for (int key = tid; key < seq; key += block_threads) {
    const float weight = expf(scores[key] - max_score);
    scores[key] = weight;
    local_sum += weight;
  }

  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / reduce[0];

  // Normalize the shared score buffer in place. This keeps the final AV pass
  // simple, at the cost of a full shared-memory score vector per block.
  for (int key = tid; key < seq; key += block_threads) {
    scores[key] *= inv_sum;
  }
  __syncthreads();

  // Final pass: each thread owns one or more output dimensions and scans all
  // keys. This is the main reason the prototype is slow for seq=1024: V is read
  // repeatedly for every query token instead of being tiled/reused.
  for (int d = tid; d < dim; d += block_threads) {
    float sum = 0.0f;
    for (int key = 0; key < seq; ++key) {
      const float* v = qkv + ((key * heads + head) * 3 * dim + 2 * dim);
      sum += scores[key] * v[d];
    }
    out[(token * heads + head) * dim + d] = sum;
  }
}

__global__ void SplitQKVForAttentionKernel(const float* qkv, float* q,
                                           float* k, float* v, int rows,
                                           int heads, int qkv_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * heads * qkv_dim;
  if (idx >= count) return;

  const int dim = idx % qkv_dim;
  const int row = (idx / qkv_dim) % rows;
  const int head = idx / (qkv_dim * rows);
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_base = row * qkv_cols + head * 3 * qkv_dim + dim;
  q[idx] = qkv[qkv_base];
  k[idx] = qkv[qkv_base + qkv_dim];
  v[idx] = qkv[qkv_base + 2 * qkv_dim];
}

__global__ void SplitQKVForAttentionFloat4Kernel(const float* qkv, float* q,
                                                 float* k, float* v, int rows,
                                                 int heads, int qkv_dim) {
  const int vec_dim = qkv_dim / 4;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * heads * vec_dim;
  if (idx >= count) return;

  const int dim4 = idx % vec_dim;
  const int row = (idx / vec_dim) % rows;
  const int head = idx / (vec_dim * rows);
  const int dim = dim4 * 4;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_base = row * qkv_cols + head * 3 * qkv_dim + dim;
  const int out_base = (head * rows + row) * qkv_dim + dim;

  const float4 q_vec =
      *reinterpret_cast<const float4*>(qkv + qkv_base);
  const float4 k_vec =
      *reinterpret_cast<const float4*>(qkv + qkv_base + qkv_dim);
  const float4 v_vec =
      *reinterpret_cast<const float4*>(qkv + qkv_base + 2 * qkv_dim);
  *reinterpret_cast<float4*>(q + out_base) = q_vec;
  *reinterpret_cast<float4*>(k + out_base) = k_vec;
  *reinterpret_cast<float4*>(v + out_base) = v_vec;
}

__global__ void SplitQKVToBF16ForAttentionKernel(
    const float* qkv, rocblas_bfloat16* q, rocblas_bfloat16* k,
    rocblas_bfloat16* v, int rows, int heads, int qkv_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * heads * qkv_dim;
  if (idx >= count) return;

  const int dim = idx % qkv_dim;
  const int row = (idx / qkv_dim) % rows;
  const int head = idx / (qkv_dim * rows);
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_base = row * qkv_cols + head * 3 * qkv_dim + dim;
  q[idx] = rocblas_bfloat16(qkv[qkv_base]);
  k[idx] = rocblas_bfloat16(qkv[qkv_base + qkv_dim]);
  v[idx] = rocblas_bfloat16(qkv[qkv_base + 2 * qkv_dim]);
}

__global__ void SplitQKToBF16VFloatForAttentionKernel(
    const float* qkv, rocblas_bfloat16* q, rocblas_bfloat16* k, float* v,
    int rows, int heads, int qkv_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * heads * qkv_dim;
  if (idx >= count) return;

  const int dim = idx % qkv_dim;
  const int row = (idx / qkv_dim) % rows;
  const int head = idx / (qkv_dim * rows);
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_base = row * qkv_cols + head * 3 * qkv_dim + dim;

  // This split supports a narrower attention experiment than the full BF16
  // path above: only Q and K are converted so rocBLAS can use BF16 math for
  // the Q.K score GEMM. V remains F32, which lets the following AV GEMM consume
  // the F32 softmax scores directly and avoids a separate score-quantization
  // kernel on the large heads x rows x rows score matrix.
  q[idx] = rocblas_bfloat16(qkv[qkv_base]);
  k[idx] = rocblas_bfloat16(qkv[qkv_base + qkv_dim]);
  v[idx] = qkv[qkv_base + 2 * qkv_dim];
}

__global__ void F32ToBF16Kernel(const float* src, rocblas_bfloat16* dst,
                                int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  dst[idx] = rocblas_bfloat16(src[idx]);
}

__global__ void SoftmaxRowsKernel(float* scores, int rows) {
  extern __shared__ float reduce[];
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int block_threads = blockDim.x;
  float* row_scores =
      scores + (static_cast<size_t>(head) * rows + row) * rows;

  float local_max = -3.4028234663852886e38f;
  for (int col = tid; col < rows; col += block_threads) {
    local_max = fmaxf(local_max, row_scores[col]);
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce[0];

  float local_sum = 0.0f;
  for (int col = tid; col < rows; col += block_threads) {
    const float value = expf(row_scores[col] - max_score);
    row_scores[col] = value;
    local_sum += value;
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / reduce[0];

  for (int col = tid; col < rows; col += block_threads) {
    row_scores[col] *= inv_sum;
  }
}

template <int kRows, int kThreads>
__global__ void SoftmaxRowsFixedKernel(float* scores) {
  __shared__ float reduce[kThreads];
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  float* row_scores =
      scores + (static_cast<size_t>(head) * kRows + row) * kRows;

  float local_max = -3.4028234663852886e38f;
  for (int col = tid; col < kRows; col += kThreads) {
    local_max = fmaxf(local_max, row_scores[col]);
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce[0];

  float local_sum = 0.0f;
  for (int col = tid; col < kRows; col += kThreads) {
    const float value = expf(row_scores[col] - max_score);
    row_scores[col] = value;
    local_sum += value;
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / reduce[0];

  for (int col = tid; col < kRows; col += kThreads) {
    row_scores[col] *= inv_sum;
  }
}

template <int kRows, int kThreads>
__global__ void SoftmaxRowsFixedFloat4Kernel(float* scores) {
  static_assert((kRows % 4) == 0, "kRows must be divisible by 4");
  __shared__ float reduce[kThreads];
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  float* row_scores =
      scores + (static_cast<size_t>(head) * kRows + row) * kRows;
  constexpr int kVecRows = kRows / 4;

  float local_max = -3.4028234663852886e38f;
  for (int vec = tid; vec < kVecRows; vec += kThreads) {
    const float4 values =
        *reinterpret_cast<const float4*>(row_scores + vec * 4);
    local_max = fmaxf(local_max, values.x);
    local_max = fmaxf(local_max, values.y);
    local_max = fmaxf(local_max, values.z);
    local_max = fmaxf(local_max, values.w);
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce[0];

  float local_sum = 0.0f;
  for (int vec = tid; vec < kVecRows; vec += kThreads) {
    const float4 input =
        *reinterpret_cast<const float4*>(row_scores + vec * 4);
    float4 output;
    output.x = expf(input.x - max_score);
    output.y = expf(input.y - max_score);
    output.z = expf(input.z - max_score);
    output.w = expf(input.w - max_score);
    *reinterpret_cast<float4*>(row_scores + vec * 4) = output;
    local_sum += output.x + output.y + output.z + output.w;
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / reduce[0];

  for (int vec = tid; vec < kVecRows; vec += kThreads) {
    float4 values = *reinterpret_cast<const float4*>(row_scores + vec * 4);
    values.x *= inv_sum;
    values.y *= inv_sum;
    values.z *= inv_sum;
    values.w *= inv_sum;
    *reinterpret_cast<float4*>(row_scores + vec * 4) = values;
  }
}

template <int kRows, int kThreads>
__global__ void SoftmaxRowsFixedFloat4RecomputeKernel(float* scores) {
  static_assert((kRows % 4) == 0, "kRows must be divisible by 4");
  __shared__ float reduce[kThreads];
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  float* row_scores =
      scores + (static_cast<size_t>(head) * kRows + row) * kRows;
  constexpr int kVecRows = kRows / 4;

  float local_max = -3.4028234663852886e38f;
  for (int vec = tid; vec < kVecRows; vec += kThreads) {
    const float4 values =
        *reinterpret_cast<const float4*>(row_scores + vec * 4);
    local_max = fmaxf(local_max, values.x);
    local_max = fmaxf(local_max, values.y);
    local_max = fmaxf(local_max, values.z);
    local_max = fmaxf(local_max, values.w);
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce[0];

  float local_sum = 0.0f;
  for (int vec = tid; vec < kVecRows; vec += kThreads) {
    const float4 input =
        *reinterpret_cast<const float4*>(row_scores + vec * 4);
    local_sum += expf(input.x - max_score);
    local_sum += expf(input.y - max_score);
    local_sum += expf(input.z - max_score);
    local_sum += expf(input.w - max_score);
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / reduce[0];

  for (int vec = tid; vec < kVecRows; vec += kThreads) {
    const float4 input =
        *reinterpret_cast<const float4*>(row_scores + vec * 4);
    float4 output;
    output.x = expf(input.x - max_score) * inv_sum;
    output.y = expf(input.y - max_score) * inv_sum;
    output.z = expf(input.z - max_score) * inv_sum;
    output.w = expf(input.w - max_score) * inv_sum;
    *reinterpret_cast<float4*>(row_scores + vec * 4) = output;
  }
}

__global__ void AttentionQkSoftmaxRowsKernel(const float* __restrict__ q,
                                             const float* __restrict__ k,
                                             float* __restrict__ scores,
                                             int rows, int qkv_dim,
                                             float scale) {
  extern __shared__ float shared[];
  float* row_scores_shared = shared;
  float* reduce = shared + rows;

  const int query = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int block_threads = blockDim.x;
  const float* head_q = q + static_cast<size_t>(head) * rows * qkv_dim;
  const float* head_k = k + static_cast<size_t>(head) * rows * qkv_dim;
  float* row_scores =
      scores + (static_cast<size_t>(head) * rows + query) * rows;
  const float* query_q = head_q + static_cast<size_t>(query) * qkv_dim;

  // One workgroup owns one output row of one attention head. It computes all
  // Q.K scores for that query row, performs the stable softmax reduction in
  // shared memory, and writes the normalized score row once. This removes the
  // intermediate global raw-score matrix traffic between rocBLAS QK and the
  // softmax kernel, but the dot products are scalar F32 and intentionally
  // simple so this can serve as a conservative fused-kernel baseline.
  float local_max = -3.4028234663852886e38f;
  for (int key = tid; key < rows; key += block_threads) {
    const float* key_k = head_k + static_cast<size_t>(key) * qkv_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qkv_dim; ++dim) {
      dot += query_q[dim] * key_k[dim];
    }
    dot *= scale;
    row_scores_shared[key] = dot;
    local_max = fmaxf(local_max, dot);
  }

  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce[0];

  float local_sum = 0.0f;
  for (int key = tid; key < rows; key += block_threads) {
    const float weight = expf(row_scores_shared[key] - max_score);
    row_scores_shared[key] = weight;
    local_sum += weight;
  }

  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / reduce[0];

  for (int key = tid; key < rows; key += block_threads) {
    row_scores[key] = row_scores_shared[key] * inv_sum;
  }
}

__global__ void SoftmaxRowsStoreExpAndInvSumKernel(float* scores,
                                                   float* inv_sums,
                                                   int rows) {
  extern __shared__ float reduce[];
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int block_threads = blockDim.x;
  float* row_scores =
      scores + (static_cast<size_t>(head) * rows + row) * rows;

  float local_max = -3.4028234663852886e38f;
  for (int col = tid; col < rows; col += block_threads) {
    local_max = fmaxf(local_max, row_scores[col]);
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce[0];

  float local_sum = 0.0f;
  for (int col = tid; col < rows; col += block_threads) {
    const float value = expf(row_scores[col] - max_score);
    row_scores[col] = value;
    local_sum += value;
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    inv_sums[static_cast<size_t>(head) * rows + row] = 1.0f / reduce[0];
  }
}

__global__ void SoftmaxRowsToBF16Kernel(const float* scores,
                                        rocblas_bfloat16* scores_bf16,
                                        int rows) {
  extern __shared__ float reduce[];
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int block_threads = blockDim.x;
  const float* row_scores =
      scores + (static_cast<size_t>(head) * rows + row) * rows;
  rocblas_bfloat16* row_scores_bf16 =
      scores_bf16 + (static_cast<size_t>(head) * rows + row) * rows;

  float local_max = -3.4028234663852886e38f;
  for (int col = tid; col < rows; col += block_threads) {
    local_max = fmaxf(local_max, row_scores[col]);
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce[0];

  float local_sum = 0.0f;
  for (int col = tid; col < rows; col += block_threads) {
    local_sum += expf(row_scores[col] - max_score);
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / reduce[0];

  // This keeps the softmax reduction in F32 but stores the normalized scores as
  // BF16 directly. It is a benchmark-only precision experiment for the WMMA AV
  // path, replacing "F32 softmax write + F32->BF16 score quantization" with one
  // kernel boundary.
  for (int col = tid; col < rows; col += block_threads) {
    row_scores_bf16[col] =
        rocblas_bfloat16(expf(row_scores[col] - max_score) * inv_sum);
  }
}

__global__ void SoftmaxAvRowsKernel(const float* scores, const float* v,
                                    float* head_major, int rows,
                                    int qkv_dim) {
  extern __shared__ float reduce[];
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int block_threads = blockDim.x;
  const float* row_scores =
      scores + (static_cast<size_t>(head) * rows + row) * rows;
  const float* head_v =
      v + static_cast<size_t>(head) * rows * qkv_dim;
  float* out =
      head_major + (static_cast<size_t>(head) * rows + row) * qkv_dim;

  // Keep the numerically sensitive softmax reduction in F32, but do not write
  // normalized scores back to global memory. This prototype tests whether
  // fusing softmax with AV can beat the current "softmax kernel + rocBLAS AV"
  // boundary before we invest in a more complex tiled attention kernel.
  float local_max = -3.4028234663852886e38f;
  for (int col = tid; col < rows; col += block_threads) {
    local_max = fmaxf(local_max, row_scores[col]);
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce[0];

  float local_sum = 0.0f;
  for (int col = tid; col < rows; col += block_threads) {
    local_sum += expf(row_scores[col] - max_score);
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / reduce[0];

  for (int dim = tid; dim < qkv_dim; dim += block_threads) {
    float sum = 0.0f;
    for (int col = 0; col < rows; ++col) {
      const float weight = expf(row_scores[col] - max_score) * inv_sum;
      sum += weight * head_v[static_cast<size_t>(col) * qkv_dim + dim];
    }
    out[dim] = sum;
  }
}

template <int kRows, int kDim, int kGroups>
__global__ __launch_bounds__(kGroups * 32) void SoftmaxAvRowsGroupedKernel(
    const float* __restrict__ scores, const float* __restrict__ v,
    float* __restrict__ head_major) {
  static_assert(kGroups > 0, "kGroups must be positive");
  __shared__ float weights[kRows];
  __shared__ float reduce[kGroups * 32];
  __shared__ float partial[kGroups][kDim];

  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int group = tid >> 5;
  const int lane = tid & 31;
  constexpr int block_threads = kGroups * 32;
  const float* row_scores =
      scores + (static_cast<size_t>(head) * kRows + row) * kRows;
  const float* head_v =
      v + static_cast<size_t>(head) * kRows * kDim;
  float* out =
      head_major + (static_cast<size_t>(head) * kRows + row) * kDim;

  // Fused softmax+AV with a more realistic scalar schedule than
  // SoftmaxAvRowsKernel. One workgroup still owns one [head, query] row, but
  // it stores the exp(score - max) weights once in LDS and splits the AV
  // accumulation for each output dimension across kGroups lane groups. This
  // tests whether eliminating repeated exp() work and adding key-parallel
  // reduction is enough to compete with "fixed softmax + rocBLAS AV".
  float local_max = -3.4028234663852886e38f;
  for (int key = tid; key < kRows; key += block_threads) {
    local_max = fmaxf(local_max, row_scores[key]);
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce[0];

  float local_sum = 0.0f;
  for (int key = tid; key < kRows; key += block_threads) {
    const float weight = expf(row_scores[key] - max_score);
    weights[key] = weight;
    local_sum += weight;
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / reduce[0];

  for (int dim = lane; dim < kDim; dim += 32) {
    float sum = 0.0f;
    for (int key = group; key < kRows; key += kGroups) {
      sum += (weights[key] * inv_sum) *
             head_v[static_cast<size_t>(key) * kDim + dim];
    }
    partial[group][dim] = sum;
  }
  __syncthreads();

  for (int dim = tid; dim < kDim; dim += block_threads) {
    float sum = 0.0f;
    for (int g = 0; g < kGroups; ++g) {
      sum += partial[g][dim];
    }
    out[dim] = sum;
  }
}

__global__ void AttentionOnlineHeadMajorKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ head_major, int rows,
    int qkv_dim, float scale) {
  extern __shared__ float shared[];
  float* scores = shared;
  float* reduce = shared + rows;

  const int query = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int block_threads = blockDim.x;
  const float* head_q = q + static_cast<size_t>(head) * rows * qkv_dim;
  const float* head_k = k + static_cast<size_t>(head) * rows * qkv_dim;
  const float* head_v = v + static_cast<size_t>(head) * rows * qkv_dim;
  float* out =
      head_major + (static_cast<size_t>(head) * rows + query) * qkv_dim;
  const float* query_q = head_q + static_cast<size_t>(query) * qkv_dim;

  // Bench-only structural prototype. Unlike the earliest fused attention
  // kernel, this consumes the already-split head-major Q/K/V layout, so K and V
  // rows for one head are contiguous. It still uses a simple scalar schedule:
  // one workgroup owns one [head, query] row, stores that row's scores in LDS,
  // then produces all 72 AV output dimensions. If this cannot approach
  // rocBLAS QK + fixed softmax + rocBLAS AV, a serious no-score-matrix path
  // needs a multi-query tiled design rather than another per-row scalar kernel.
  float local_max = -3.4028234663852886e38f;
  for (int key = tid; key < rows; key += block_threads) {
    const float* key_k = head_k + static_cast<size_t>(key) * qkv_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qkv_dim; ++dim) {
      dot += query_q[dim] * key_k[dim];
    }
    dot *= scale;
    scores[key] = dot;
    local_max = fmaxf(local_max, dot);
  }

  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce[0];

  float local_sum = 0.0f;
  for (int key = tid; key < rows; key += block_threads) {
    const float weight = expf(scores[key] - max_score);
    scores[key] = weight;
    local_sum += weight;
  }

  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / reduce[0];

  for (int key = tid; key < rows; key += block_threads) {
    scores[key] *= inv_sum;
  }
  __syncthreads();

  for (int dim = tid; dim < qkv_dim; dim += block_threads) {
    float sum = 0.0f;
    for (int key = 0; key < rows; ++key) {
      sum += scores[key] * head_v[static_cast<size_t>(key) * qkv_dim + dim];
    }
    out[dim] = sum;
  }
}

template <int kQueryRows>
__global__ __launch_bounds__(256) void AttentionOnlineMultiQueryKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ head_major, int rows,
    int qkv_dim, float scale) {
  static_assert(kQueryRows > 0, "kQueryRows must be positive");
  extern __shared__ float shared[];
  float* scores = shared;
  float* reduce = scores + static_cast<size_t>(kQueryRows) * rows;

  const int query_base = blockIdx.x * kQueryRows;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  constexpr int block_threads = 256;
  const float* head_q = q + static_cast<size_t>(head) * rows * qkv_dim;
  const float* head_k = k + static_cast<size_t>(head) * rows * qkv_dim;
  const float* head_v = v + static_cast<size_t>(head) * rows * qkv_dim;

  float local_max[kQueryRows];
#pragma unroll
  for (int qr = 0; qr < kQueryRows; ++qr) {
    local_max[qr] = -3.4028234663852886e38f;
  }

  // Compute QK for several neighboring query rows in one workgroup. Each
  // thread owns a strided subset of keys and reuses the same K row across all
  // query rows before moving to the next key. Scores stay in LDS so the large
  // [heads, rows, rows] matrix is never written to global memory.
  for (int key = tid; key < rows; key += block_threads) {
    const float* key_k = head_k + static_cast<size_t>(key) * qkv_dim;
#pragma unroll
    for (int qr = 0; qr < kQueryRows; ++qr) {
      const int query = query_base + qr;
      float dot = 0.0f;
      if (query < rows) {
        const float* query_q =
            head_q + static_cast<size_t>(query) * qkv_dim;
        for (int dim = 0; dim < qkv_dim; ++dim) {
          dot += query_q[dim] * key_k[dim];
        }
        dot *= scale;
        local_max[qr] = fmaxf(local_max[qr], dot);
      }
      scores[static_cast<size_t>(qr) * rows + key] = dot;
    }
  }

#pragma unroll
  for (int qr = 0; qr < kQueryRows; ++qr) {
    reduce[static_cast<size_t>(qr) * block_threads + tid] = local_max[qr];
  }
  __syncthreads();

  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
#pragma unroll
      for (int qr = 0; qr < kQueryRows; ++qr) {
        float* qr_reduce = reduce + static_cast<size_t>(qr) * block_threads;
        qr_reduce[tid] = fmaxf(qr_reduce[tid], qr_reduce[tid + stride]);
      }
    }
    __syncthreads();
  }

  float local_sum[kQueryRows];
#pragma unroll
  for (int qr = 0; qr < kQueryRows; ++qr) {
    local_sum[qr] = 0.0f;
  }

  for (int key = tid; key < rows; key += block_threads) {
#pragma unroll
    for (int qr = 0; qr < kQueryRows; ++qr) {
      const int query = query_base + qr;
      float weight = 0.0f;
      if (query < rows) {
        const float max_score = reduce[static_cast<size_t>(qr) * block_threads];
        weight = expf(scores[static_cast<size_t>(qr) * rows + key] -
                      max_score);
        local_sum[qr] += weight;
      }
      scores[static_cast<size_t>(qr) * rows + key] = weight;
    }
  }

#pragma unroll
  for (int qr = 0; qr < kQueryRows; ++qr) {
    reduce[static_cast<size_t>(qr) * block_threads + tid] = local_sum[qr];
  }
  __syncthreads();

  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
#pragma unroll
      for (int qr = 0; qr < kQueryRows; ++qr) {
        float* qr_reduce = reduce + static_cast<size_t>(qr) * block_threads;
        qr_reduce[tid] += qr_reduce[tid + stride];
      }
    }
    __syncthreads();
  }

  float inv_sum[kQueryRows];
#pragma unroll
  for (int qr = 0; qr < kQueryRows; ++qr) {
    inv_sum[qr] = 1.0f / reduce[static_cast<size_t>(qr) * block_threads];
  }

  // AV phase: for each output dimension, load V[key, dim] once and reuse it
  // across all query rows in this tile. This is the main improvement over the
  // single-query online prototype, which reread the same V row for every query.
  for (int dim = tid; dim < qkv_dim; dim += block_threads) {
    float accum[kQueryRows];
#pragma unroll
    for (int qr = 0; qr < kQueryRows; ++qr) {
      accum[qr] = 0.0f;
    }
    for (int key = 0; key < rows; ++key) {
      const float v_value = head_v[static_cast<size_t>(key) * qkv_dim + dim];
#pragma unroll
      for (int qr = 0; qr < kQueryRows; ++qr) {
        const int query = query_base + qr;
        if (query < rows) {
          const float prob =
              scores[static_cast<size_t>(qr) * rows + key] * inv_sum[qr];
          accum[qr] += prob * v_value;
        }
      }
    }
#pragma unroll
    for (int qr = 0; qr < kQueryRows; ++qr) {
      const int query = query_base + qr;
      if (query < rows) {
        head_major[(static_cast<size_t>(head) * rows + query) * qkv_dim +
                   dim] = accum[qr];
      }
    }
  }
}

template <int BM, int BN, int BK>
__global__ void AttentionQkTiledKernel(const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       float* __restrict__ scores, int rows,
                                       int qkv_dim, float scale) {
  __shared__ float q_tile[BM][BK];
  __shared__ float k_tile[BN][BK];

  const int head = blockIdx.z;
  const int query = blockIdx.y * BM + threadIdx.y;
  const int key = blockIdx.x * BN + threadIdx.x;
  const int local_query = threadIdx.y;
  const int local_key = threadIdx.x;
  const int thread_index = threadIdx.y * BN + threadIdx.x;
  constexpr int block_threads = BM * BN;

  const float* head_q = q + static_cast<size_t>(head) * rows * qkv_dim;
  const float* head_k = k + static_cast<size_t>(head) * rows * qkv_dim;
  float sum = 0.0f;

  // Matrix multiply view of QK for one head:
  //   score[query, key] = scale * sum_dim Q[query, dim] * K[key, dim].
  // The output score matrix is written in the row-major layout consumed by the
  // softmax kernels. This scalar prototype stages a BM x BK Q tile and a
  // BN x BK K tile in LDS so neighboring query/key outputs share operand loads.
  // It is deliberately bench-only: rocBLAS is already strong on this small-K
  // batched GEMM, so the purpose is to quantify the gap before attempting an
  // RDNA WMMA/MFMA QK kernel.
  for (int dim_base = 0; dim_base < qkv_dim; dim_base += BK) {
    for (int i = thread_index; i < BM * BK; i += block_threads) {
      const int tile_query = i / BK;
      const int tile_dim = i % BK;
      const int global_query = blockIdx.y * BM + tile_query;
      const int global_dim = dim_base + tile_dim;
      q_tile[tile_query][tile_dim] =
          global_query < rows && global_dim < qkv_dim
              ? head_q[static_cast<size_t>(global_query) * qkv_dim +
                       global_dim]
              : 0.0f;
    }
    for (int i = thread_index; i < BN * BK; i += block_threads) {
      const int tile_key = i / BK;
      const int tile_dim = i % BK;
      const int global_key = blockIdx.x * BN + tile_key;
      const int global_dim = dim_base + tile_dim;
      k_tile[tile_key][tile_dim] =
          global_key < rows && global_dim < qkv_dim
              ? head_k[static_cast<size_t>(global_key) * qkv_dim + global_dim]
              : 0.0f;
    }
    __syncthreads();

    if (query < rows && key < rows) {
      for (int dim = 0; dim < BK; ++dim) {
        sum += q_tile[local_query][dim] * k_tile[local_key][dim];
      }
    }
    __syncthreads();
  }

  if (query < rows && key < rows) {
    scores[(static_cast<size_t>(head) * rows + query) * rows + key] =
        sum * scale;
  }
}

template <int BM, int BN, int BK>
__global__ void AttentionAvTiledKernel(const float* __restrict__ scores,
                                       const float* __restrict__ v,
                                       float* __restrict__ head_major,
                                       int rows, int qkv_dim) {
  __shared__ float score_tile[BM][BK];
  __shared__ float v_tile[BK][BN];

  const int head = blockIdx.z;
  const int row = blockIdx.y * BM + threadIdx.y;
  const int dim = blockIdx.x * BN + threadIdx.x;
  const int local_row = threadIdx.y;
  const int local_dim = threadIdx.x;
  const int thread_index = threadIdx.y * BN + threadIdx.x;
  constexpr int block_threads = BM * BN;

  const float* head_scores =
      scores + static_cast<size_t>(head) * rows * rows;
  const float* head_v = v + static_cast<size_t>(head) * rows * qkv_dim;
  float sum = 0.0f;

  // Matrix multiply view of AV for one head:
  //   out[query, dim] = sum_key softmax_score[query, key] * V[key, dim].
  // This first tiled prototype keeps all math in scalar F32. It stages a small
  // BM x BK score tile and a BK x BN V tile in LDS so several query rows reuse
  // the same V load. That directly tests whether a simple tiled HIP kernel can
  // beat the cached rocBLAS AV baseline before investing in MFMA-specific code.
  for (int key_base = 0; key_base < rows; key_base += BK) {
    for (int i = thread_index; i < BM * BK; i += block_threads) {
      const int tile_row = i / BK;
      const int tile_key = i % BK;
      const int global_row = blockIdx.y * BM + tile_row;
      const int global_key = key_base + tile_key;
      score_tile[tile_row][tile_key] =
          global_row < rows && global_key < rows
              ? head_scores[static_cast<size_t>(global_row) * rows +
                            global_key]
              : 0.0f;
    }
    for (int i = thread_index; i < BK * BN; i += block_threads) {
      const int tile_key = i / BN;
      const int tile_dim = i % BN;
      const int global_key = key_base + tile_key;
      const int global_dim = blockIdx.x * BN + tile_dim;
      v_tile[tile_key][tile_dim] =
          global_key < rows && global_dim < qkv_dim
              ? head_v[static_cast<size_t>(global_key) * qkv_dim + global_dim]
              : 0.0f;
    }
    __syncthreads();

    if (row < rows && dim < qkv_dim) {
      for (int k = 0; k < BK; ++k) {
        sum += score_tile[local_row][k] * v_tile[k][local_dim];
      }
    }
    __syncthreads();
  }

  if (row < rows && dim < qkv_dim) {
    head_major[(static_cast<size_t>(head) * rows + row) * qkv_dim + dim] =
        sum;
  }
}

template <int BM, int BK, int kDim4>
__global__ void AttentionAvTiledDim4Kernel(const float* __restrict__ scores,
                                           const float* __restrict__ v,
                                           float* __restrict__ head_major,
                                           int rows) {
  __shared__ float score_tile[BM][BK];
  __shared__ float4 v_tile[BK][kDim4];

  const int row = blockIdx.x * BM + threadIdx.y;
  const int head = blockIdx.y;
  const int local_dim4 = threadIdx.x;
  const int local_row = threadIdx.y;
  const int thread_index = threadIdx.y * kDim4 + threadIdx.x;
  constexpr int block_threads = BM * kDim4;
  constexpr int kDim = kDim4 * 4;

  const float* head_scores =
      scores + static_cast<size_t>(head) * rows * rows;
  const float* head_v = v + static_cast<size_t>(head) * rows * kDim;
  float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};

  // Matrix view: C[row, dim] = sum_key P[row, key] * V[key, dim].
  // Compared with AttentionAvTiledKernel, this kernel owns the full 72-wide
  // head dimension in one workgroup row tile. A thread accumulates four adjacent
  // dimensions, so each staged score value is reused across a float4 of V and
  // the launch avoids five separate dim tiles for the 72-wide PaliGemma2 head.
  for (int key_base = 0; key_base < rows; key_base += BK) {
    for (int i = thread_index; i < BM * BK; i += block_threads) {
      const int tile_row = i / BK;
      const int tile_key = i % BK;
      const int global_row = blockIdx.x * BM + tile_row;
      const int global_key = key_base + tile_key;
      score_tile[tile_row][tile_key] =
          global_row < rows && global_key < rows
              ? head_scores[static_cast<size_t>(global_row) * rows +
                            global_key]
              : 0.0f;
    }
    for (int i = thread_index; i < BK * kDim4; i += block_threads) {
      const int tile_key = i / kDim4;
      const int tile_dim4 = i % kDim4;
      const int global_key = key_base + tile_key;
      const int global_dim = tile_dim4 * 4;
      v_tile[tile_key][tile_dim4] =
          global_key < rows
              ? *reinterpret_cast<const float4*>(
                    head_v + static_cast<size_t>(global_key) * kDim +
                    global_dim)
              : float4{0.0f, 0.0f, 0.0f, 0.0f};
    }
    __syncthreads();

    if (row < rows) {
#pragma unroll
      for (int k = 0; k < BK; ++k) {
        const float weight = score_tile[local_row][k];
        const float4 values = v_tile[k][local_dim4];
        sum.x += weight * values.x;
        sum.y += weight * values.y;
        sum.z += weight * values.z;
        sum.w += weight * values.w;
      }
    }
    __syncthreads();
  }

  if (row < rows) {
    *reinterpret_cast<float4*>(
        head_major + (static_cast<size_t>(head) * rows + row) * kDim +
        local_dim4 * 4) = sum;
  }
}

__global__ void PackAttentionHeadsKernel(const float* head_major,
                                         float* row_major, int rows,
                                         int heads, int qkv_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * heads * qkv_dim;
  if (idx >= count) return;

  const int dim = idx % qkv_dim;
  const int row = (idx / qkv_dim) % rows;
  const int head = idx / (qkv_dim * rows);
  const int out_cols = heads * qkv_dim;
  row_major[row * out_cols + head * qkv_dim + dim] = head_major[idx];
}

__global__ void PackAttentionHeadsToBF16Kernel(
    const float* head_major, rocblas_bfloat16* row_major_bf16, int rows,
    int heads, int qkv_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * heads * qkv_dim;
  if (idx >= count) return;

  const int dim = idx % qkv_dim;
  const int row = (idx / qkv_dim) % rows;
  const int head = idx / (qkv_dim * rows);
  const int out_cols = heads * qkv_dim;
  row_major_bf16[row * out_cols + head * qkv_dim + dim] =
      rocblas_bfloat16(head_major[idx]);
}

__global__ void PackAttentionHeadsToBF16Float4Kernel(
    const float* head_major, rocblas_bfloat16* row_major_bf16, int rows,
    int heads, int qkv_dim) {
  const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int vec_dim = qkv_dim / 4;
  const int vec_count = rows * heads * vec_dim;
  if (vec_idx >= vec_count) return;

  const int dim4 = vec_idx % vec_dim;
  const int row = (vec_idx / vec_dim) % rows;
  const int head = vec_idx / (vec_dim * rows);
  const int dim = dim4 * 4;
  const int out_cols = heads * qkv_dim;

  // Head-major rows are 72 floats wide in PaliGemma2. hipMalloc gives a
  // sufficiently aligned base pointer and 72 floats is a multiple of 16 bytes,
  // so every four-float load below is naturally aligned for these shapes.
  const float4 values = *reinterpret_cast<const float4*>(
      head_major + (static_cast<size_t>(head) * rows + row) * qkv_dim + dim);
  rocblas_bfloat16* dst =
      row_major_bf16 + static_cast<size_t>(row) * out_cols + head * qkv_dim +
      dim;
  dst[0] = rocblas_bfloat16(values.x);
  dst[1] = rocblas_bfloat16(values.y);
  dst[2] = rocblas_bfloat16(values.z);
  dst[3] = rocblas_bfloat16(values.w);
}

__global__ void PackScaledAttentionHeadsToBF16Kernel(
    const float* head_major, const float* inv_sums,
    rocblas_bfloat16* row_major_bf16, int rows, int heads, int qkv_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * heads * qkv_dim;
  if (idx >= count) return;

  const int dim = idx % qkv_dim;
  const int row = (idx / qkv_dim) % rows;
  const int head = idx / (qkv_dim * rows);
  const int out_cols = heads * qkv_dim;
  const float scale = inv_sums[static_cast<size_t>(head) * rows + row];
  row_major_bf16[row * out_cols + head * qkv_dim + dim] =
      rocblas_bfloat16(head_major[idx] * scale);
}

void LaunchRecommendedSplitQKV(const float* qkv, float* q, float* k, float* v,
                               int rows, int heads, int qkv_dim) {
  constexpr int kThreads = 256;
  const int qkv_count = rows * heads * qkv_dim;

  // Keep this helper in sync with the runtime backend. The 224px path uses the
  // exact vector4 split that was promoted there; 448px stays scalar because
  // longer split-only samples showed vector4 was not stable for that shape.
  if (rows == 256 && (qkv_dim % 4) == 0) {
    const int vector_count = rows * heads * (qkv_dim / 4);
    const int vector_blocks = (vector_count + kThreads - 1) / kThreads;
    hipLaunchKernelGGL(SplitQKVForAttentionFloat4Kernel, dim3(vector_blocks),
                       dim3(kThreads), 0, 0, qkv, q, k, v, rows, heads,
                       qkv_dim);
    return;
  }

  const int qkv_blocks = (qkv_count + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(SplitQKVForAttentionKernel, dim3(qkv_blocks),
                     dim3(kThreads), 0, 0, qkv, q, k, v, rows, heads,
                     qkv_dim);
}

void LaunchAttentionPackBF16(const float* head_major,
                             rocblas_bfloat16* row_major_bf16, int rows,
                             int heads, int qkv_dim) {
  constexpr int kThreads = 256;

  if ((qkv_dim % 4) == 0) {
    const int vector_count = rows * heads * (qkv_dim / 4);
    const int vector_blocks = (vector_count + kThreads - 1) / kThreads;
    hipLaunchKernelGGL(PackAttentionHeadsToBF16Float4Kernel,
                       dim3(vector_blocks), dim3(kThreads), 0, 0, head_major,
                       row_major_bf16, rows, heads, qkv_dim);
    return;
  }

  const int count = rows * heads * qkv_dim;
  const int blocks = (count + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(PackAttentionHeadsToBF16Kernel, dim3(blocks),
                     dim3(kThreads), 0, 0, head_major, row_major_bf16, rows,
                     heads, qkv_dim);
}

void LaunchRecommendedSoftmax(float* scores, int rows, int heads) {
  // Mirrors the backend's current fixed-shape choices. Both promoted shapes use
  // the float4 row kernel because the score matrix is contiguous and 16-byte
  // aligned, and qkv_dim/head-major layouts keep the row length divisible by 4.
  if (rows == 256) {
    hipLaunchKernelGGL((SoftmaxRowsFixedFloat4Kernel<256, 64>),
                       dim3(rows, heads), dim3(64), 0, 0, scores);
    return;
  }
  if (rows == 1024) {
    hipLaunchKernelGGL((SoftmaxRowsFixedFloat4Kernel<1024, 128>),
                       dim3(rows, heads), dim3(128), 0, 0, scores);
    return;
  }

  const int softmax_threads = rows <= 256 ? 64 : 256;
  const size_t softmax_shared =
      static_cast<size_t>(softmax_threads) * sizeof(float);
  hipLaunchKernelGGL(SoftmaxRowsKernel, dim3(rows, heads),
                     dim3(softmax_threads), softmax_shared, 0, scores, rows);
}

int RecommendedQkSolution(int rows) {
  // Current generated cache:
  //   rows=256  -> -111 (near-tied with -110);
  //   rows=1024 -> -110.
  return rows <= 256 ? -111 : -110;
}

int RecommendedAvSolution(int rows) {
  return rows <= 256 ? -1161823143 : -451;
}

// Scalar tiled BF16 GEMM prototype for the ViT MLP projections:
//
//   C[M,N] = A[M,K] * B[K,N]
//
// The kernel deliberately mirrors the row-major layout used by the host-side
// gemma.cpp tensors instead of going through rocBLAS' transposed column-major
// view. Each block owns one BM x BN tile of C and streams the K dimension in BK
// chunks through shared memory. Each thread computes one output element.
//
// This is a feasibility baseline, not the intended final optimization. It uses
// scalar BF16-to-F32 conversion and F32 FMA in normal vector ALUs, so it does not
// exercise AMD matrix instructions. If this loses badly to rocBLAS, the next
// serious custom path must be an MFMA/rocWMMA-style kernel rather than a more
// polished version of this scalar kernel.
template <int BM, int BN, int BK>
__global__ void Bf16TiledGemmF32Kernel(
    const rocblas_bfloat16* __restrict__ a,
    const rocblas_bfloat16* __restrict__ b, float* __restrict__ c, int m,
    int k, int n) {
  __shared__ rocblas_bfloat16 tile_a[BM][BK];
  __shared__ rocblas_bfloat16 tile_b[BK][BN];

  const int local_col = threadIdx.x;
  const int local_row = threadIdx.y;
  const int row = blockIdx.y * BM + local_row;
  const int col = blockIdx.x * BN + local_col;
  const int thread_index = local_row * BN + local_col;
  constexpr int block_threads = BM * BN;

  float acc = 0.0f;
  for (int k0 = 0; k0 < k; k0 += BK) {
    // Cooperative load for A's BM x BK panel. Threads whose linear id is larger
    // than the panel keep stepping by block size until the whole tile is filled.
    // Out-of-range K or M entries are explicitly zeroed so the inner loop can
    // stay branch-free.
    for (int idx = thread_index; idx < BM * BK; idx += block_threads) {
      const int tile_row = idx / BK;
      const int tile_k = idx - tile_row * BK;
      const int global_row = blockIdx.y * BM + tile_row;
      const int global_k = k0 + tile_k;
      tile_a[tile_row][tile_k] =
          (global_row < m && global_k < k)
              ? a[static_cast<size_t>(global_row) * k + global_k]
              : rocblas_bfloat16(0.0f);
    }

    // Cooperative load for B's BK x BN panel. B is row-major with leading
    // dimension N, so consecutive local columns map to consecutive memory.
    for (int idx = thread_index; idx < BK * BN; idx += block_threads) {
      const int tile_k = idx / BN;
      const int tile_col = idx - tile_k * BN;
      const int global_k = k0 + tile_k;
      const int global_col = blockIdx.x * BN + tile_col;
      tile_b[tile_k][tile_col] =
          (global_k < k && global_col < n)
              ? b[static_cast<size_t>(global_k) * n + global_col]
              : rocblas_bfloat16(0.0f);
    }
    __syncthreads();

    // One thread owns one C element. The accumulation is intentionally scalar;
    // this tells us how far a straightforward HIP kernel is from rocBLAS before
    // we spend engineering time on MFMA tile scheduling.
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      acc += static_cast<float>(tile_a[local_row][kk]) *
             static_cast<float>(tile_b[kk][local_col]);
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[static_cast<size_t>(row) * n + col] = acc;
  }
}

using WmmaBf16Frag = short __attribute__((ext_vector_type(16)));
using WmmaF32Frag = float __attribute__((ext_vector_type(8)));

__device__ short Bf16Bits(rocblas_bfloat16 value) {
  return static_cast<short>(value.data);
}

// RDNA wave32 WMMA prototype for the ViT MLP projections. One HIP block is one
// wave and computes one 16 x 16 output tile. The builtin consumes a 16 x 16 A
// tile and a 16 x 16 B tile per K step, accumulating into F32 output.
//
// The fragment layout follows AMD's RDNA WMMA convention:
//   * lanes 0-15 and 16-31 must carry replicated A/B fragments;
//   * for the BF16/F32 wave32 builtin used here, lane%16 selects the M row for
//     A and each fragment element selects the K column;
//   * B and C/D are row-major, so lane%16 selects the N column.
//
// The source tensors remain gemma.cpp row-major. Only the per-lane loads map
// that row-major memory into the WMMA fragment layout.
__global__ __launch_bounds__(32) void Bf16WmmaGemmF32Kernel(
    const rocblas_bfloat16* __restrict__ a,
    const rocblas_bfloat16* __restrict__ b, float* __restrict__ c, int m,
    int k, int n) {
  const int lane = threadIdx.x;
  const int lane16 = lane & 15;
  const int row_parity = lane >> 4;
  const int tile_m = blockIdx.y * 16;
  const int tile_n = blockIdx.x * 16;

  WmmaF32Frag acc = {};
  for (int k0 = 0; k0 < k; k0 += 16) {
    WmmaBf16Frag a_frag;
    WmmaBf16Frag b_frag;

    // A fragment: lane16 is the M row inside the output tile, and each vector
    // element is one K column. This duplicates the required GFX11 half-wave
    // fragment copies with direct loads; a shuffle-based replication variant was
    // measured slower on this iGPU because the added lane dependencies outweighed
    // the saved memory transactions.
#pragma unroll
    for (int ele = 0; ele < 16; ++ele) {
      const int row = tile_m + lane16;
      const int col_k = k0 + ele;
      a_frag[ele] = (row < m && col_k < k)
                        ? Bf16Bits(a[static_cast<size_t>(row) * k + col_k])
                        : 0;
    }

    // B fragment: lane16 is the N-column inside the output tile, and each
    // vector element is one K row. This is the row-major B[K,N] layout expected
    // by the RDNA WMMA builtin.
#pragma unroll
    for (int ele = 0; ele < 16; ++ele) {
      const int row_k = k0 + ele;
      const int col = tile_n + lane16;
      b_frag[ele] = (row_k < k && col < n)
                        ? Bf16Bits(b[static_cast<size_t>(row_k) * n + col])
                        : 0;
    }

    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc);
  }

  // Each lane owns eight output elements. Lanes 0-15 write even rows and lanes
  // 16-31 write odd rows for the same 16 output columns.
#pragma unroll
  for (int ele = 0; ele < 8; ++ele) {
    const int row = tile_m + ele * 2 + row_parity;
    const int col = tile_n + lane16;
    if (row < m && col < n) {
      c[static_cast<size_t>(row) * n + col] = acc[ele];
    }
  }
}

// Multi-wave RDNA WMMA prototype. The one-wave kernel above reloads the same
// A[M,K] 16 x 16 tile once for every neighboring N tile. That is particularly
// wasteful for the MLP-up shape, where N=4304 and many output-column tiles share
// the same input-token rows. This kernel groups WAVES_N waves in one workgroup:
//
//   * every wave computes a different 16-column output tile for the same 16
//     rows of C;
//   * all waves share one staged A tile in LDS for each K chunk;
//   * each wave still loads its own B tile directly because those columns differ
//     across waves.
//
// This is still much simpler than rocBLAS: it only reuses A across N tiles and
// does not implement deeper K blocking, double buffering, vectorized global
// loads, or multiple M tiles per workgroup. The point is to measure whether the
// most obvious inter-wave reuse moves the 448px MLP case in the right direction.
template <int WAVES_N>
__global__ __launch_bounds__(32 * WAVES_N) void Bf16WmmaGemmF32GroupedNKernel(
    const rocblas_bfloat16* __restrict__ a,
    const rocblas_bfloat16* __restrict__ b, float* __restrict__ c, int m,
    int k, int n) {
  static_assert(WAVES_N > 0, "WAVES_N must be positive");
  __shared__ short shared_a[16][16];

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int lane16 = lane & 15;
  const int row_parity = lane >> 4;
  const int tile_m = blockIdx.y * 16;
  const int tile_n = blockIdx.x * (16 * WAVES_N) + wave * 16;
  constexpr int block_threads = 32 * WAVES_N;

  WmmaF32Frag acc = {};
  for (int k0 = 0; k0 < k; k0 += 16) {
    // Cooperatively stage A's 16 x 16 tile once for the whole workgroup. This
    // is the only reuse this prototype adds over the one-wave kernel: all
    // WAVES_N waves consume the same rows and K chunk while writing different
    // output columns.
    for (int idx = tid; idx < 16 * 16; idx += block_threads) {
      const int row_in_tile = idx >> 4;
      const int k_in_tile = idx & 15;
      const int row = tile_m + row_in_tile;
      const int col_k = k0 + k_in_tile;
      shared_a[row_in_tile][k_in_tile] =
          (row < m && col_k < k)
              ? Bf16Bits(a[static_cast<size_t>(row) * k + col_k])
              : 0;
    }
    __syncthreads();

    WmmaBf16Frag a_frag;
    WmmaBf16Frag b_frag;

    // Build the A fragment from LDS. The per-lane fragment layout is identical
    // to Bf16WmmaGemmF32Kernel; the only difference is that repeated global
    // loads have been replaced by shared-memory reads.
#pragma unroll
    for (int ele = 0; ele < 16; ++ele) {
      a_frag[ele] = shared_a[lane16][ele];
    }

    // B is unique for each wave because each wave owns a different N tile. It
    // remains a direct row-major global load in this prototype.
#pragma unroll
    for (int ele = 0; ele < 16; ++ele) {
      const int row_k = k0 + ele;
      const int col = tile_n + lane16;
      b_frag[ele] = (row_k < k && col < n)
                        ? Bf16Bits(b[static_cast<size_t>(row_k) * n + col])
                        : 0;
    }

    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc);
    __syncthreads();
  }

#pragma unroll
  for (int ele = 0; ele < 8; ++ele) {
    const int row = tile_m + ele * 2 + row_parity;
    const int col = tile_n + lane16;
    if (row < m && col < n) {
      c[static_cast<size_t>(row) * n + col] = acc[ele];
    }
  }
}

// Alternative multi-wave prototype for wide-output MLP-up shapes. Grouped-N
// reuses A across neighboring output columns; this grouped-M version reuses B
// across neighboring input-token rows:
//
//   * every wave computes a different 16-row output tile for the same 16
//     columns of C;
//   * all waves share one staged B[K,N] tile in LDS for each K chunk;
//   * each wave still loads its own A tile directly because those rows differ.
//
// This tests the hypothesis that `mlp_up_448` is limited by repeated linear_0
// weight reads more than by repeated activation reads.
template <int WAVES_M>
__global__ __launch_bounds__(32 * WAVES_M) void Bf16WmmaGemmF32GroupedMKernel(
    const rocblas_bfloat16* __restrict__ a,
    const rocblas_bfloat16* __restrict__ b, float* __restrict__ c, int m,
    int k, int n) {
  static_assert(WAVES_M > 0, "WAVES_M must be positive");
  __shared__ short shared_b[16][16];

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int lane16 = lane & 15;
  const int row_parity = lane >> 4;
  const int tile_m = blockIdx.y * (16 * WAVES_M) + wave * 16;
  const int tile_n = blockIdx.x * 16;
  constexpr int block_threads = 32 * WAVES_M;

  WmmaF32Frag acc = {};
  for (int k0 = 0; k0 < k; k0 += 16) {
    // Stage the shared B/weight tile once per workgroup. All WAVES_M waves
    // consume the same columns and K chunk while producing different rows.
    for (int idx = tid; idx < 16 * 16; idx += block_threads) {
      const int k_in_tile = idx >> 4;
      const int col_in_tile = idx & 15;
      const int row_k = k0 + k_in_tile;
      const int col = tile_n + col_in_tile;
      shared_b[k_in_tile][col_in_tile] =
          (row_k < k && col < n)
              ? Bf16Bits(b[static_cast<size_t>(row_k) * n + col])
              : 0;
    }
    __syncthreads();

    WmmaBf16Frag a_frag;
    WmmaBf16Frag b_frag;

    // A is unique per wave because each wave owns a different M tile.
#pragma unroll
    for (int ele = 0; ele < 16; ++ele) {
      const int row = tile_m + lane16;
      const int col_k = k0 + ele;
      a_frag[ele] = (row < m && col_k < k)
                        ? Bf16Bits(a[static_cast<size_t>(row) * k + col_k])
                        : 0;
    }

    // B comes from LDS and is shared by all waves in the workgroup.
#pragma unroll
    for (int ele = 0; ele < 16; ++ele) {
      b_frag[ele] = shared_b[ele][lane16];
    }

    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc);
    __syncthreads();
  }

#pragma unroll
  for (int ele = 0; ele < 8; ++ele) {
    const int row = tile_m + ele * 2 + row_parity;
    const int col = tile_n + lane16;
    if (row < m && col < n) {
      c[static_cast<size_t>(row) * n + col] = acc[ele];
    }
  }
}

// Two-dimensional multi-wave RDNA WMMA prototype. Grouped-N proved that A-side
// reuse is valuable, while grouped-M proved that staging B alone makes the
// remaining A reads too strided and expensive. This kernel tests the next
// schedule: one workgroup owns a small M x N macro-tile, stages both operands in
// LDS, and assigns one wave to each 16 x 16 output tile inside that macro-tile.
//
// For example, WAVES_M=2 and WAVES_N=4 creates an 8-wave workgroup that
// computes a 32 x 64 output tile. For every 16-wide K chunk it stages:
//
//   * A[(2 * 16) rows, 16 K columns]
//   * B[16 K rows, (4 * 16) columns]
//
// All global operand loads are cooperative and row-major contiguous inside the
// staged panels. This directly addresses the grouped-M weakness where each wave
// loaded its own A fragment with a large row stride.
template <int WAVES_M, int WAVES_N>
__global__ __launch_bounds__(32 * WAVES_M * WAVES_N)
    void Bf16WmmaGemmF32Grouped2DKernel(
        const rocblas_bfloat16* __restrict__ a,
        const rocblas_bfloat16* __restrict__ b, float* __restrict__ c, int m,
        int k, int n) {
  static_assert(WAVES_M > 0, "WAVES_M must be positive");
  static_assert(WAVES_N > 0, "WAVES_N must be positive");
  __shared__ short shared_a[16 * WAVES_M][16];
  __shared__ short shared_b[16][16 * WAVES_N];

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int wave_m = wave / WAVES_N;
  const int wave_n = wave - wave_m * WAVES_N;
  const int lane = tid & 31;
  const int lane16 = lane & 15;
  const int row_parity = lane >> 4;
  const int macro_m = blockIdx.y * (16 * WAVES_M);
  const int macro_n = blockIdx.x * (16 * WAVES_N);
  const int tile_m = macro_m + wave_m * 16;
  const int tile_n = macro_n + wave_n * 16;
  constexpr int block_threads = 32 * WAVES_M * WAVES_N;

  WmmaF32Frag acc = {};
  for (int k0 = 0; k0 < k; k0 += 16) {
    // Stage the full A side of the macro-tile. Flattening the panel gives
    // coalesced row-major loads and avoids the strided per-wave A traffic that
    // made the grouped-M-only prototype unattractive.
    for (int idx = tid; idx < 16 * WAVES_M * 16; idx += block_threads) {
      const int row_in_macro = idx >> 4;
      const int k_in_tile = idx & 15;
      const int row = macro_m + row_in_macro;
      const int col_k = k0 + k_in_tile;
      shared_a[row_in_macro][k_in_tile] =
          (row < m && col_k < k)
              ? Bf16Bits(a[static_cast<size_t>(row) * k + col_k])
              : 0;
    }

    // Stage the full B side of the macro-tile. Adjacent threads walk adjacent
    // N columns for each K row, matching the row-major B[K,N] layout.
    for (int idx = tid; idx < 16 * 16 * WAVES_N; idx += block_threads) {
      const int k_in_tile = idx / (16 * WAVES_N);
      const int col_in_macro = idx - k_in_tile * (16 * WAVES_N);
      const int row_k = k0 + k_in_tile;
      const int col = macro_n + col_in_macro;
      shared_b[k_in_tile][col_in_macro] =
          (row_k < k && col < n)
              ? Bf16Bits(b[static_cast<size_t>(row_k) * n + col])
              : 0;
    }
    __syncthreads();

    WmmaBf16Frag a_frag;
    WmmaBf16Frag b_frag;

    // Each wave selects its own 16 x 16 tile from the staged macro-panels. The
    // lane-to-fragment mapping is the same as the validated one-wave and
    // grouped-N kernels; only the source of the fragment changed from global
    // memory to LDS.
#pragma unroll
    for (int ele = 0; ele < 16; ++ele) {
      a_frag[ele] = shared_a[wave_m * 16 + lane16][ele];
      b_frag[ele] = shared_b[ele][wave_n * 16 + lane16];
    }

    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc);
    __syncthreads();
  }

#pragma unroll
  for (int ele = 0; ele < 8; ++ele) {
    const int row = tile_m + ele * 2 + row_parity;
    const int col = tile_n + lane16;
    if (row < m && col < n) {
      c[static_cast<size_t>(row) * n + col] = acc[ele];
    }
  }
}

template <int WAVES_M, int WAVES_N>
__global__ __launch_bounds__(32 * WAVES_M * WAVES_N)
    void Bf16WmmaGemmF32Grouped2DBatchedKernel(
        const rocblas_bfloat16* __restrict__ a,
        const rocblas_bfloat16* __restrict__ b, float* __restrict__ c, int m,
        int k, int n, size_t stride_a, size_t stride_b, size_t stride_c) {
  static_assert(WAVES_M > 0, "WAVES_M must be positive");
  static_assert(WAVES_N > 0, "WAVES_N must be positive");
  __shared__ short shared_a[16 * WAVES_M][16];
  __shared__ short shared_b[16][16 * WAVES_N];

  const int batch = blockIdx.z;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int wave_m = wave / WAVES_N;
  const int wave_n = wave - wave_m * WAVES_N;
  const int lane = tid & 31;
  const int lane16 = lane & 15;
  const int row_parity = lane >> 4;
  const int macro_m = blockIdx.y * (16 * WAVES_M);
  const int macro_n = blockIdx.x * (16 * WAVES_N);
  const int tile_m = macro_m + wave_m * 16;
  const int tile_n = macro_n + wave_n * 16;
  constexpr int block_threads = 32 * WAVES_M * WAVES_N;
  const rocblas_bfloat16* batch_a = a + static_cast<size_t>(batch) * stride_a;
  const rocblas_bfloat16* batch_b = b + static_cast<size_t>(batch) * stride_b;
  float* batch_c = c + static_cast<size_t>(batch) * stride_c;

  WmmaF32Frag acc = {};
  for (int k0 = 0; k0 < k; k0 += 16) {
    for (int idx = tid; idx < 16 * WAVES_M * 16; idx += block_threads) {
      const int row_in_macro = idx >> 4;
      const int k_in_tile = idx & 15;
      const int row = macro_m + row_in_macro;
      const int col_k = k0 + k_in_tile;
      shared_a[row_in_macro][k_in_tile] =
          (row < m && col_k < k)
              ? Bf16Bits(batch_a[static_cast<size_t>(row) * k + col_k])
              : 0;
    }
    for (int idx = tid; idx < 16 * 16 * WAVES_N; idx += block_threads) {
      const int k_in_tile = idx / (16 * WAVES_N);
      const int col_in_macro = idx - k_in_tile * (16 * WAVES_N);
      const int row_k = k0 + k_in_tile;
      const int col = macro_n + col_in_macro;
      shared_b[k_in_tile][col_in_macro] =
          (row_k < k && col < n)
              ? Bf16Bits(batch_b[static_cast<size_t>(row_k) * n + col])
              : 0;
    }
    __syncthreads();

    WmmaBf16Frag a_frag;
    WmmaBf16Frag b_frag;
#pragma unroll
    for (int ele = 0; ele < 16; ++ele) {
      a_frag[ele] = shared_a[wave_m * 16 + lane16][ele];
      b_frag[ele] = shared_b[ele][wave_n * 16 + lane16];
    }

    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc);
    __syncthreads();
  }

#pragma unroll
  for (int ele = 0; ele < 8; ++ele) {
    const int row = tile_m + ele * 2 + row_parity;
    const int col = tile_n + lane16;
    if (row < m && col < n) {
      batch_c[static_cast<size_t>(row) * n + col] = acc[ele];
    }
  }
}

template <int WAVES_M, int WAVES_N>
__global__ __launch_bounds__(32 * WAVES_M * WAVES_N)
    void Bf16WmmaGemmF32Grouped2DBatchedTransBKernel(
        const rocblas_bfloat16* __restrict__ a,
        const rocblas_bfloat16* __restrict__ b, float* __restrict__ c, int m,
        int k, int n, size_t stride_a, size_t stride_b, size_t stride_c,
        float alpha) {
  static_assert(WAVES_M > 0, "WAVES_M must be positive");
  static_assert(WAVES_N > 0, "WAVES_N must be positive");
  __shared__ short shared_a[16 * WAVES_M][16];
  __shared__ short shared_b[16][16 * WAVES_N];

  const int batch = blockIdx.z;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int wave_m = wave / WAVES_N;
  const int wave_n = wave - wave_m * WAVES_N;
  const int lane = tid & 31;
  const int lane16 = lane & 15;
  const int row_parity = lane >> 4;
  const int macro_m = blockIdx.y * (16 * WAVES_M);
  const int macro_n = blockIdx.x * (16 * WAVES_N);
  const int tile_m = macro_m + wave_m * 16;
  const int tile_n = macro_n + wave_n * 16;
  constexpr int block_threads = 32 * WAVES_M * WAVES_N;
  const rocblas_bfloat16* batch_a = a + static_cast<size_t>(batch) * stride_a;
  const rocblas_bfloat16* batch_b = b + static_cast<size_t>(batch) * stride_b;
  float* batch_c = c + static_cast<size_t>(batch) * stride_c;

  WmmaF32Frag acc = {};
  for (int k0 = 0; k0 < k; k0 += 16) {
    for (int idx = tid; idx < 16 * WAVES_M * 16; idx += block_threads) {
      const int row_in_macro = idx >> 4;
      const int k_in_tile = idx & 15;
      const int row = macro_m + row_in_macro;
      const int col_k = k0 + k_in_tile;
      shared_a[row_in_macro][k_in_tile] =
          (row < m && col_k < k)
              ? Bf16Bits(batch_a[static_cast<size_t>(row) * k + col_k])
              : 0;
    }
    for (int idx = tid; idx < 16 * 16 * WAVES_N; idx += block_threads) {
      const int k_in_tile = idx / (16 * WAVES_N);
      const int col_in_macro = idx - k_in_tile * (16 * WAVES_N);
      const int row_k = k0 + k_in_tile;
      const int col = macro_n + col_in_macro;
      // QK uses A=Q[M,K] and B=K[N,K] in their natural head-major row-major
      // layout, but the GEMM is C[M,N] = Q * K^T. The regular grouped-2D
      // kernel expects B[K,N]; this variant stages the same WMMA fragment by
      // reading transposed B elements from K[col, row_k].
      shared_b[k_in_tile][col_in_macro] =
          (row_k < k && col < n)
              ? Bf16Bits(batch_b[static_cast<size_t>(col) * k + row_k])
              : 0;
    }
    __syncthreads();

    WmmaBf16Frag a_frag;
    WmmaBf16Frag b_frag;
#pragma unroll
    for (int ele = 0; ele < 16; ++ele) {
      a_frag[ele] = shared_a[wave_m * 16 + lane16][ele];
      b_frag[ele] = shared_b[ele][wave_n * 16 + lane16];
    }

    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc);
    __syncthreads();
  }

#pragma unroll
  for (int ele = 0; ele < 8; ++ele) {
    const int row = tile_m + ele * 2 + row_parity;
    const int col = tile_n + lane16;
    if (row < m && col < n) {
      batch_c[static_cast<size_t>(row) * n + col] = acc[ele] * alpha;
    }
  }
}

struct Shape {
  const char* name;
  int m;
  int k;
  int n;
  bool bf16;
};

struct BatchedShape {
  const char* name;
  int m;
  int k;
  int n;
  int batch;
};

struct AttentionShape {
  const char* name;
  int seq;
  int heads;
  int dim;
};

struct Args {
  int samples = 10;
  int warmup = 3;
  int iters = 1;
  std::string filter;
  bool pipeline = false;
  bool mlp_kernel = false;
  bool mlp_wmma = false;
  bool mlp_wmma_group = false;
  bool mlp_wmma_group_sweep = false;
  bool mlp_wmma_m_group_sweep = false;
  bool mlp_wmma_2d_group_sweep = false;
  bool projection_wmma_2d_sweep = false;
  bool attention_phase_profile = false;
  bool attention_recommended_phase_profile = false;
  bool attention_deferred_softmax_scale_profile = false;
  bool attention_bf16_phase_profile = false;
  bool attention_bf16_qk_phase_profile = false;
  bool attention_qk_bf16_wmma_profile = false;
  bool attention_qk_softmax_fused_profile = false;
  bool attention_softmax_av_fused_profile = false;
  bool attention_softmax_av_grouped_profile = false;
  bool attention_softmax_thread_sweep = false;
  bool attention_softmax_fixed_profile = false;
  bool attention_split_vectorized_profile = false;
  bool attention_pack_vectorized_profile = false;
  int attention_rocblas_solution_sweep = 0;
  bool attention_qk_tiled_profile = false;
  bool attention_av_tiled_profile = false;
  bool attention_online_head_major_profile = false;
  bool attention_online_multiquery_profile = false;
  bool attention_av_bf16_wmma_profile = false;
  bool kernel_occupancy_report = false;
  bool shapes = true;
};

void Usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [--samples N] [--warmup N] [--iters N] "
               "[--filter SUBSTR] [--pipeline] [--pipeline_only] "
               "[--mlp_kernel] [--mlp_kernel_only] "
               "[--mlp_wmma] [--mlp_wmma_only] "
               "[--mlp_wmma_group] [--mlp_wmma_group_only] "
               "[--mlp_wmma_group_sweep] [--mlp_wmma_group_sweep_only] "
               "[--mlp_wmma_m_group_sweep] "
               "[--mlp_wmma_m_group_sweep_only] "
               "[--mlp_wmma_2d_group_sweep] "
               "[--mlp_wmma_2d_group_sweep_only] "
               "[--projection_wmma_2d_sweep] "
               "[--projection_wmma_2d_sweep_only] "
               "[--attention_phase_profile] "
               "[--attention_phase_profile_only] "
               "[--attention_recommended_phase_profile] "
               "[--attention_recommended_phase_profile_only] "
               "[--attention_deferred_softmax_scale_profile] "
               "[--attention_deferred_softmax_scale_profile_only] "
               "[--attention_bf16_phase_profile] "
               "[--attention_bf16_phase_profile_only] "
               "[--attention_bf16_qk_phase_profile] "
               "[--attention_bf16_qk_phase_profile_only] "
               "[--attention_qk_bf16_wmma_profile] "
               "[--attention_qk_bf16_wmma_profile_only] "
               "[--attention_qk_softmax_fused_profile] "
               "[--attention_qk_softmax_fused_profile_only] "
               "[--attention_softmax_av_fused_profile] "
               "[--attention_softmax_av_fused_profile_only] "
               "[--attention_softmax_av_grouped_profile] "
               "[--attention_softmax_av_grouped_profile_only] "
               "[--attention_softmax_thread_sweep] "
               "[--attention_softmax_thread_sweep_only] "
               "[--attention_softmax_fixed_profile] "
               "[--attention_softmax_fixed_profile_only] "
               "[--attention_split_vectorized_profile] "
               "[--attention_split_vectorized_profile_only] "
               "[--attention_pack_vectorized_profile] "
               "[--attention_pack_vectorized_profile_only] "
               "[--attention_rocblas_solution_sweep N] "
               "[--attention_rocblas_solution_sweep_only N] "
               "[--attention_qk_tiled_profile] "
               "[--attention_qk_tiled_profile_only] "
               "[--attention_av_tiled_profile] "
               "[--attention_av_tiled_profile_only] "
               "[--attention_online_head_major_profile] "
               "[--attention_online_head_major_profile_only] "
               "[--attention_online_multiquery_profile] "
               "[--attention_online_multiquery_profile_only] "
               "[--attention_av_bf16_wmma_profile] "
               "[--attention_av_bf16_wmma_profile_only] "
               "[--kernel_occupancy_report] "
               "[--kernel_occupancy_report_only]\n"
               "\n"
               "Benchmarks rocBLAS GEMM for PaliGemma2 ViT shapes on HIP.\n"
               "Matrices are interpreted as row-major C[M,N] = A[M,K] * "
               "B[K,N] by calling rocBLAS on the transposed column-major "
               "view.\n"
               "\n"
               "--mlp_kernel compares rocBLAS against a scalar tiled HIP "
               "BF16->F32 GEMM for the ViT MLP shapes.\n"
               "--mlp_wmma compares rocBLAS against a wave32 RDNA WMMA "
               "BF16->F32 GEMM for the ViT MLP shapes.\n"
               "--mlp_wmma_group compares rocBLAS against a grouped-N "
               "multi-wave WMMA kernel that stages A tiles in LDS.\n"
               "--mlp_wmma_group_sweep runs grouped-N WMMA with 2, 4, 6, 8, "
               "12, and 16 waves per workgroup.\n"
               "--mlp_wmma_m_group_sweep runs grouped-M WMMA with 2, 4, and "
               "8 waves per workgroup.\n"
               "--mlp_wmma_2d_group_sweep runs grouped 2D WMMA macro-tiles "
               "that stage both A and B in LDS for MLP shapes.\n"
               "--projection_wmma_2d_sweep runs the same grouped 2D WMMA "
               "macro-tiles for non-MLP ViT projection shapes.\n"
               "--attention_phase_profile times split, QK, softmax, AV, and "
               "pack phases of the current device attention path.\n"
               "--attention_recommended_phase_profile times the recommended "
               "F32 attention path with cached QK/AV solution indices and "
               "pack-to-BF16.\n"
               "--attention_deferred_softmax_scale_profile compares that "
               "recommended path against an F32 variant that stores exp() "
               "scores, runs AV before row normalization, and applies the "
               "softmax scale while packing to BF16.\n"
               "--attention_bf16_phase_profile times an experimental BF16 "
               "QK/AV attention path with F32 softmax.\n"
               "--attention_bf16_qk_phase_profile times an experimental "
               "BF16 QK + F32 AV attention path.\n"
               "--attention_qk_bf16_wmma_profile compares cached-solution "
               "F32 rocBLAS QK against BF16 rocBLAS QK and a local grouped-2D "
               "BF16 WMMA QK kernel.\n"
               "--attention_qk_softmax_fused_profile compares cached-solution "
               "F32 rocBLAS QK + fixed softmax against an experimental fused "
               "F32 QK+softmax kernel.\n"
               "--attention_softmax_av_fused_profile times an experimental "
               "F32 QK + fused F32 softmax/AV attention path.\n"
               "--attention_softmax_av_grouped_profile compares cached QK + "
               "fixed softmax + cached AV against an experimental grouped "
               "F32 softmax/AV kernel.\n"
               "--attention_softmax_thread_sweep times the F32 softmax kernel "
               "with several block sizes after a rocBLAS QK setup.\n"
               "--attention_softmax_fixed_profile compares the current "
               "dynamic-row F32 softmax kernel with fixed-shape variants for "
               "the PaliGemma2 224/448 attention row lengths.\n"
               "--attention_split_vectorized_profile compares the scalar "
               "F32 Q/K/V split kernel with a float4 vectorized split for "
               "PaliGemma2 attention layouts.\n"
               "--attention_pack_vectorized_profile compares the scalar "
               "head-major pack-to-BF16 kernel with a float4 load variant "
               "for PaliGemma2 attention layouts.\n"
               "--attention_rocblas_solution_sweep tests rocBLAS solution "
               "indices for the current F32 QK and AV attention GEMMs.\n"
               "--attention_qk_tiled_profile compares cached-solution "
               "rocBLAS QK against a scalar tiled F32 HIP QK kernel.\n"
               "--attention_av_tiled_profile compares cached-solution "
               "rocBLAS AV against a scalar tiled F32 HIP AV kernel.\n"
               "--attention_online_head_major_profile compares the current "
               "F32 QK+softmax+AV path against a bench-only online F32 "
               "head-major attention kernel that does not write scores to "
               "global memory.\n"
               "--attention_online_multiquery_profile compares that same "
               "baseline against a bench-only online F32 attention kernel "
               "that processes multiple query rows per workgroup to reuse "
               "K/V tiles better than the single-query prototype.\n"
               "--attention_av_bf16_wmma_profile compares cached-solution "
               "F32 rocBLAS AV against BF16 rocBLAS and local BF16 WMMA AV.\n"
               "--kernel_occupancy_report prints HIP occupancy API estimates "
               "for the custom kernels used by the current recommended path.\n",
               argv0);
}

bool ParseInt(const char* value, int& out) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0) return false;
  out = static_cast<int>(parsed);
  return true;
}

bool ParseArgs(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      Usage(argv[0]);
      return false;
    }
    if (arg == "--samples" && i + 1 < argc) {
      if (!ParseInt(argv[++i], args.samples)) return false;
      continue;
    }
    if (arg == "--warmup" && i + 1 < argc) {
      if (!ParseInt(argv[++i], args.warmup)) return false;
      continue;
    }
    if (arg == "--iters" && i + 1 < argc) {
      if (!ParseInt(argv[++i], args.iters)) return false;
      continue;
    }
    if (arg == "--filter" && i + 1 < argc) {
      args.filter = argv[++i];
      continue;
    }
    if (arg == "--pipeline") {
      args.pipeline = true;
      continue;
    }
    if (arg == "--pipeline_only") {
      args.pipeline = true;
      args.shapes = false;
      continue;
    }
    if (arg == "--mlp_kernel") {
      args.mlp_kernel = true;
      continue;
    }
    if (arg == "--mlp_kernel_only") {
      args.mlp_kernel = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--mlp_wmma") {
      args.mlp_wmma = true;
      continue;
    }
    if (arg == "--mlp_wmma_only") {
      args.mlp_wmma = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--mlp_wmma_group") {
      args.mlp_wmma_group = true;
      continue;
    }
    if (arg == "--mlp_wmma_group_only") {
      args.mlp_wmma_group = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--mlp_wmma_group_sweep") {
      args.mlp_wmma_group_sweep = true;
      continue;
    }
    if (arg == "--mlp_wmma_group_sweep_only") {
      args.mlp_wmma_group_sweep = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--mlp_wmma_m_group_sweep") {
      args.mlp_wmma_m_group_sweep = true;
      continue;
    }
    if (arg == "--mlp_wmma_m_group_sweep_only") {
      args.mlp_wmma_m_group_sweep = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--mlp_wmma_2d_group_sweep") {
      args.mlp_wmma_2d_group_sweep = true;
      continue;
    }
    if (arg == "--mlp_wmma_2d_group_sweep_only") {
      args.mlp_wmma_2d_group_sweep = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--projection_wmma_2d_sweep") {
      args.projection_wmma_2d_sweep = true;
      continue;
    }
    if (arg == "--projection_wmma_2d_sweep_only") {
      args.projection_wmma_2d_sweep = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_phase_profile") {
      args.attention_phase_profile = true;
      continue;
    }
    if (arg == "--attention_phase_profile_only") {
      args.attention_phase_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_recommended_phase_profile") {
      args.attention_recommended_phase_profile = true;
      continue;
    }
    if (arg == "--attention_recommended_phase_profile_only") {
      args.attention_recommended_phase_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_deferred_softmax_scale_profile") {
      args.attention_deferred_softmax_scale_profile = true;
      continue;
    }
    if (arg == "--attention_deferred_softmax_scale_profile_only") {
      args.attention_deferred_softmax_scale_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_bf16_phase_profile") {
      args.attention_bf16_phase_profile = true;
      continue;
    }
    if (arg == "--attention_bf16_phase_profile_only") {
      args.attention_bf16_phase_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_bf16_qk_phase_profile") {
      args.attention_bf16_qk_phase_profile = true;
      continue;
    }
    if (arg == "--attention_bf16_qk_phase_profile_only") {
      args.attention_bf16_qk_phase_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_qk_bf16_wmma_profile") {
      args.attention_qk_bf16_wmma_profile = true;
      continue;
    }
    if (arg == "--attention_qk_bf16_wmma_profile_only") {
      args.attention_qk_bf16_wmma_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_qk_softmax_fused_profile") {
      args.attention_qk_softmax_fused_profile = true;
      continue;
    }
    if (arg == "--attention_qk_softmax_fused_profile_only") {
      args.attention_qk_softmax_fused_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_softmax_av_fused_profile") {
      args.attention_softmax_av_fused_profile = true;
      continue;
    }
    if (arg == "--attention_softmax_av_fused_profile_only") {
      args.attention_softmax_av_fused_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_softmax_av_grouped_profile") {
      args.attention_softmax_av_grouped_profile = true;
      continue;
    }
    if (arg == "--attention_softmax_av_grouped_profile_only") {
      args.attention_softmax_av_grouped_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_softmax_thread_sweep") {
      args.attention_softmax_thread_sweep = true;
      continue;
    }
    if (arg == "--attention_softmax_thread_sweep_only") {
      args.attention_softmax_thread_sweep = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_softmax_fixed_profile") {
      args.attention_softmax_fixed_profile = true;
      continue;
    }
    if (arg == "--attention_softmax_fixed_profile_only") {
      args.attention_softmax_fixed_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_split_vectorized_profile") {
      args.attention_split_vectorized_profile = true;
      continue;
    }
    if (arg == "--attention_split_vectorized_profile_only") {
      args.attention_split_vectorized_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_pack_vectorized_profile") {
      args.attention_pack_vectorized_profile = true;
      continue;
    }
    if (arg == "--attention_pack_vectorized_profile_only") {
      args.attention_pack_vectorized_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_rocblas_solution_sweep" && i + 1 < argc) {
      if (!ParseInt(argv[++i], args.attention_rocblas_solution_sweep)) {
        return false;
      }
      continue;
    }
    if (arg == "--attention_rocblas_solution_sweep_only" && i + 1 < argc) {
      if (!ParseInt(argv[++i], args.attention_rocblas_solution_sweep)) {
        return false;
      }
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_qk_tiled_profile") {
      args.attention_qk_tiled_profile = true;
      continue;
    }
    if (arg == "--attention_qk_tiled_profile_only") {
      args.attention_qk_tiled_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_av_tiled_profile") {
      args.attention_av_tiled_profile = true;
      continue;
    }
    if (arg == "--attention_av_tiled_profile_only") {
      args.attention_av_tiled_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_online_head_major_profile") {
      args.attention_online_head_major_profile = true;
      continue;
    }
    if (arg == "--attention_online_head_major_profile_only") {
      args.attention_online_head_major_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_online_multiquery_profile") {
      args.attention_online_multiquery_profile = true;
      continue;
    }
    if (arg == "--attention_online_multiquery_profile_only") {
      args.attention_online_multiquery_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--attention_av_bf16_wmma_profile") {
      args.attention_av_bf16_wmma_profile = true;
      continue;
    }
    if (arg == "--attention_av_bf16_wmma_profile_only") {
      args.attention_av_bf16_wmma_profile = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    if (arg == "--kernel_occupancy_report") {
      args.kernel_occupancy_report = true;
      continue;
    }
    if (arg == "--kernel_occupancy_report_only") {
      args.kernel_occupancy_report = true;
      args.pipeline = false;
      args.shapes = false;
      continue;
    }
    std::fprintf(stderr, "Unknown or malformed argument: %s\n", argv[i]);
    Usage(argv[0]);
    return false;
  }
  return true;
}

template <typename Kernel>
void PrintKernelOccupancy(const hipDeviceProp_t& props, const char* name,
                          Kernel kernel, int block_threads,
                          size_t dynamic_shared_bytes) {
  int active_blocks = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel, block_threads, dynamic_shared_bytes));

  const int wave_size = props.warpSize > 0 ? props.warpSize : 32;
  const int max_threads = props.maxThreadsPerMultiProcessor;
  const int active_threads = active_blocks * block_threads;
  const double active_waves =
      static_cast<double>(active_threads) / static_cast<double>(wave_size);
  const double thread_occupancy =
      max_threads > 0
          ? static_cast<double>(active_threads) /
                static_cast<double>(max_threads)
          : 0.0;

  std::printf("%-34s block=%4d active_blocks_per_CU=%2d "
              "active_waves_per_CU=%5.1f thread_occ=%5.1f%% "
              "dynamic_smem=%5zu B\n",
              name, block_threads, active_blocks, active_waves,
              thread_occupancy * 100.0, dynamic_shared_bytes);
}

void BenchKernelOccupancyReport(const hipDeviceProp_t& props) {
  std::printf("\nCustom kernel occupancy report:\n");
  std::printf("  CUs=%d max_threads_per_CU=%d warp_size=%d\n",
              props.multiProcessorCount, props.maxThreadsPerMultiProcessor,
              props.warpSize);
  std::printf("  Note: rocBLAS internal GEMM kernels are not visible through "
              "this benchmark's HIP occupancy calls.\n");

  PrintKernelOccupancy(props, "wmma2d_8x4_current",
                       Bf16WmmaGemmF32Grouped2DKernel<8, 4>, 32 * 8 * 4, 0);
  PrintKernelOccupancy(props, "wmma2d_4x8_compare",
                       Bf16WmmaGemmF32Grouped2DKernel<4, 8>, 32 * 4 * 8, 0);
  PrintKernelOccupancy(props, "wmma2d_4x4_compare",
                       Bf16WmmaGemmF32Grouped2DKernel<4, 4>, 32 * 4 * 4, 0);
  PrintKernelOccupancy(props, "mlp_down_wmma8_current",
                       Bf16WmmaGemmF32GroupedNKernel<8>, 32 * 8, 0);
  PrintKernelOccupancy(props, "mlp_down_wmma12_candidate",
                       Bf16WmmaGemmF32GroupedNKernel<12>, 32 * 12, 0);
  PrintKernelOccupancy(props, "mlp_down_wmma16_compare",
                       Bf16WmmaGemmF32GroupedNKernel<16>, 32 * 16, 0);
  PrintKernelOccupancy(props, "softmax_224_float4_64",
                       SoftmaxRowsFixedFloat4Kernel<256, 64>, 64, 0);
  PrintKernelOccupancy(props, "softmax_448_float4_128",
                       SoftmaxRowsFixedFloat4Kernel<1024, 128>, 128, 0);
  PrintKernelOccupancy(props, "softmax_448_recompute_128",
                       SoftmaxRowsFixedFloat4RecomputeKernel<1024, 128>, 128,
                       0);
  PrintKernelOccupancy(props, "split_qkv_scalar_256t",
                       SplitQKVForAttentionKernel, 256, 0);
  PrintKernelOccupancy(props, "pack_bf16_float4_256t",
                       PackAttentionHeadsToBF16Float4Kernel, 256, 0);
}

// Fill matrices with deterministic nonzero values. The benchmark only measures
// timing, but deterministic input prevents the compiler/runtime from seeing
// all-zero buffers and keeps a cheap output read meaningful.
template <class T>
void Fill(std::vector<T>& values) {
  for (size_t i = 0; i < values.size(); ++i) {
    const float x = static_cast<float>((static_cast<int>(i * 17 + 11) % 251) -
                                       125) *
                    (1.0f / 128.0f);
    values[i] = T(x);
  }
}

template <>
void Fill(std::vector<float>& values) {
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>((static_cast<int>(i * 17 + 11) % 251) -
                                   125) *
                (1.0f / 128.0f);
  }
}

void FillNonBf16Exact(std::vector<float>& values) {
  for (size_t i = 0; i < values.size(); ++i) {
    const float base =
        static_cast<float>((static_cast<int>(i * 17 + 11) % 251) - 125) *
        (1.0f / 127.0f);
    const float perturb =
        static_cast<float>(static_cast<int>(i % 13) - 6) * 0.000137f;
    values[i] = base + perturb;
  }
}

template <class T>
rocblas_datatype RocblasType();

template <>
rocblas_datatype RocblasType<float>() {
  return rocblas_datatype_f32_r;
}

template <>
rocblas_datatype RocblasType<rocblas_bfloat16>() {
  return rocblas_datatype_bf16_r;
}

template <class T>
void BenchOne(rocblas_handle handle, const Shape& shape, const Args& args) {
  const size_t a_count = static_cast<size_t>(shape.m) * shape.k;
  const size_t b_count = static_cast<size_t>(shape.k) * shape.n;
  const size_t c_count = static_cast<size_t>(shape.m) * shape.n;

  std::vector<T> h_a(a_count);
  std::vector<T> h_b(b_count);
  Fill(h_a);
  Fill(h_b);

  T* d_a = nullptr;
  T* d_b = nullptr;
  T* d_c = nullptr;
  HIP_CHECK(hipMalloc(&d_a, a_count * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_b, b_count * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_c, c_count * sizeof(T)));
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), a_count * sizeof(T),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), b_count * sizeof(T),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_c, 0, c_count * sizeof(T)));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const rocblas_datatype type = RocblasType<T>();

  // gemma.cpp stores matrices as row-major. rocBLAS expects column-major, so the
  // call swaps A/B and M/N to compute the transpose view:
  //   row-major C[M,N] = A[M,K] * B[K,N]
  //   column-major C^T[N,M] = B^T[N,K] * A^T[K,M]
  // The underlying buffers stay contiguous and no explicit transpose is timed.
  auto run_gemm = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, shape.n,
        shape.m, shape.k, &alpha, d_b, type, shape.n, d_a, type, shape.k,
        &beta, d_c, type, shape.n, d_c, type, shape.n,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };

  // Warmup absorbs rocBLAS lazy setup, kernel selection, and any first-use HIP
  // overhead. Only the timed loop below is reported.
  for (int i = 0; i < args.warmup; ++i) {
    for (int j = 0; j < args.iters; ++j) run_gemm();
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  std::vector<float> ms;
  ms.reserve(args.samples);
  // HIP events measure device elapsed time for the submitted work. `iters`
  // allows tiny kernels to be repeated inside one event window.
  for (int sample = 0; sample < args.samples; ++sample) {
    HIP_CHECK(hipEventRecord(start));
    for (int iter = 0; iter < args.iters; ++iter) run_gemm();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    ms.push_back(elapsed_ms / static_cast<float>(args.iters));
  }

  std::vector<T> h_c(1);
  HIP_CHECK(hipMemcpy(h_c.data(), d_c, sizeof(T), hipMemcpyDeviceToHost));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));

  std::sort(ms.begin(), ms.end());
  const float best_ms = ms.front();
  const float median_ms = ms[ms.size() / 2];
  const double flops = 2.0 * static_cast<double>(shape.m) *
                       static_cast<double>(shape.k) *
                       static_cast<double>(shape.n);
  std::printf("%-24s %-4s M=%4d K=%4d N=%4d best=%8.3f ms median=%8.3f ms "
              "%8.1f GFLOPS\n",
              shape.name, shape.bf16 ? "bf16" : "f32", shape.m, shape.k,
              shape.n, best_ms, median_ms, flops / (median_ms * 1.0E6));
}

void BenchBatchedF32(rocblas_handle handle, const BatchedShape& shape,
                     const Args& args) {
  const size_t a_count =
      static_cast<size_t>(shape.batch) * shape.m * shape.k;
  const size_t b_count =
      static_cast<size_t>(shape.batch) * shape.k * shape.n;
  const size_t c_count =
      static_cast<size_t>(shape.batch) * shape.m * shape.n;

  std::vector<float> h_a(a_count);
  std::vector<float> h_b(b_count);
  Fill(h_a);
  Fill(h_b);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  HIP_CHECK(hipMalloc(&d_a, a_count * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_b, b_count * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_c, c_count * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), a_count * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), b_count * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_c, 0, c_count * sizeof(float)));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const rocblas_stride stride_a =
      static_cast<rocblas_stride>(shape.m) * shape.k;
  const rocblas_stride stride_b =
      static_cast<rocblas_stride>(shape.k) * shape.n;
  const rocblas_stride stride_c =
      static_cast<rocblas_stride>(shape.m) * shape.n;

  // Batched QK/AV models one full 16-head ViT attention GEMM call. This removes
  // per-head launch overhead from the comparison against CPU per-head timings.
  auto run_gemm = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_none, rocblas_operation_none, shape.n,
        shape.m, shape.k, &alpha, d_b, rocblas_datatype_f32_r, shape.n,
        stride_b, d_a, rocblas_datatype_f32_r, shape.k, stride_a, &beta, d_c,
        rocblas_datatype_f32_r, shape.n, stride_c, d_c,
        rocblas_datatype_f32_r, shape.n, stride_c, shape.batch,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };

  for (int i = 0; i < args.warmup; ++i) {
    for (int j = 0; j < args.iters; ++j) run_gemm();
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  std::vector<float> ms;
  ms.reserve(args.samples);
  for (int sample = 0; sample < args.samples; ++sample) {
    HIP_CHECK(hipEventRecord(start));
    for (int iter = 0; iter < args.iters; ++iter) run_gemm();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    ms.push_back(elapsed_ms / static_cast<float>(args.iters));
  }

  float h_c = 0.0f;
  HIP_CHECK(hipMemcpy(&h_c, d_c, sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));

  std::sort(ms.begin(), ms.end());
  const float best_ms = ms.front();
  const float median_ms = ms[ms.size() / 2];
  const double flops = 2.0 * static_cast<double>(shape.batch) *
                       static_cast<double>(shape.m) *
                       static_cast<double>(shape.k) *
                       static_cast<double>(shape.n);
  std::printf("%-24s f32x%-2d M=%4d K=%4d N=%4d best=%8.3f ms "
              "median=%8.3f ms %8.1f GFLOPS\n",
              shape.name, shape.batch, shape.m, shape.k, shape.n, best_ms,
              median_ms, flops / (median_ms * 1.0E6));
}

void BenchFusedAttention(const AttentionShape& shape, const Args& args) {
  // Allocate one QKV tensor and one output tensor for a single ViT layer. The
  // prototype owns Q, K, and V in one contiguous tensor because that is the most
  // likely layout after a fused QKV projection.
  const size_t qkv_count =
      static_cast<size_t>(shape.seq) * shape.heads * 3 * shape.dim;
  const size_t out_count =
      static_cast<size_t>(shape.seq) * shape.heads * shape.dim;
  std::vector<float> h_qkv(qkv_count);
  Fill(h_qkv);

  float* d_qkv = nullptr;
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_qkv, qkv_count * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, out_count * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_qkv, h_qkv.data(), qkv_count * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_out, 0, out_count * sizeof(float)));

  const int block_threads = 256;
  const dim3 grid(shape.seq, shape.heads);
  // Shared memory holds one score per key plus one reduction slot per thread.
  // This is acceptable for 224/448 measurements but not a scalable design.
  const size_t shared_bytes =
      (static_cast<size_t>(shape.seq) + block_threads) * sizeof(float);
  const float scale = 1.0f / sqrtf(static_cast<float>(shape.dim));

  auto run_kernel = [&]() {
    hipLaunchKernelGGL(VitAttentionFusedKernel, grid, block_threads,
                       shared_bytes, 0, d_qkv, d_out, shape.seq, shape.heads,
                       shape.dim, scale);
  };

  // The timed region only includes the fused kernel launch and execution. It
  // intentionally excludes host/device copies so this result can be compared
  // against the rocBLAS QK/AV device-only microbenchmarks.
  for (int i = 0; i < args.warmup; ++i) {
    for (int j = 0; j < args.iters; ++j) run_kernel();
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  std::vector<float> ms;
  ms.reserve(args.samples);
  for (int sample = 0; sample < args.samples; ++sample) {
    HIP_CHECK(hipEventRecord(start));
    for (int iter = 0; iter < args.iters; ++iter) run_kernel();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    ms.push_back(elapsed_ms / static_cast<float>(args.iters));
  }
  HIP_CHECK(hipGetLastError());

  float h_out = 0.0f;
  HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d_qkv));
  HIP_CHECK(hipFree(d_out));

  std::sort(ms.begin(), ms.end());
  const float best_ms = ms.front();
  const float median_ms = ms[ms.size() / 2];
  const double flops = 4.0 * static_cast<double>(shape.heads) * shape.seq *
                       shape.seq * shape.dim;
  std::printf("%-24s fused MHA seq=%4d heads=%2d dim=%2d best=%8.3f ms "
              "median=%8.3f ms %8.1f GFLOPS\n",
              shape.name, shape.seq, shape.heads, shape.dim, best_ms,
              median_ms, flops / (median_ms * 1.0E6));
}

template <class T>
struct DeviceBuffer {
  T* ptr = nullptr;
  size_t count = 0;

  // RAII device buffer used by the pipeline benchmark. It allocates once and is
  // reused across many shape calls, approximating the intended resident-weight
  // GPU backend where inputs/weights are not copied for every GEMM.
  explicit DeviceBuffer(size_t count_in) : count(count_in) {
    HIP_CHECK(hipMalloc(&ptr, count * sizeof(T)));
    std::vector<T> host(count);
    Fill(host);
    HIP_CHECK(hipMemcpy(ptr, host.data(), count * sizeof(T),
                        hipMemcpyHostToDevice));
  }

  ~DeviceBuffer() {
    if (ptr != nullptr) {
      (void)hipFree(ptr);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

void PrintPipelineResult(const char* name, const std::vector<float>& ms,
                         double flops) {
  std::vector<float> sorted = ms;
  std::sort(sorted.begin(), sorted.end());
  const float best_ms = sorted.front();
  const float median_ms = sorted[sorted.size() / 2];
  std::printf("%-32s best=%8.3f ms median=%8.3f ms %8.1f GFLOPS\n", name,
              best_ms, median_ms, flops / (median_ms * 1.0E6));
}

float MedianMs(std::vector<float> ms) {
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

struct StridedBatchedGemmSpec {
  const char* label = "";
  rocblas_operation trans_a = rocblas_operation_none;
  rocblas_operation trans_b = rocblas_operation_none;
  int m = 0;
  int n = 0;
  int k = 0;
  const void* a = nullptr;
  int lda = 0;
  rocblas_stride stride_a = 0;
  const void* b = nullptr;
  int ldb = 0;
  rocblas_stride stride_b = 0;
  void* d = nullptr;
  int ldd = 0;
  rocblas_stride stride_d = 0;
  int batch_count = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
};

struct RocblasSolutionMeasurement {
  int solution = 0;
  bool failed = false;
  std::vector<float> samples_ms;
};

std::vector<rocblas_int> QueryStridedBatchedF32GemmSolutions(
    rocblas_handle handle, const StridedBatchedGemmSpec& spec) {
  rocblas_int list_size = 0;
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
  rocblas_status status = rocblas_gemm_strided_batched_ex_get_solutions(
      handle, spec.trans_a, spec.trans_b, spec.m, spec.n, spec.k, &spec.alpha,
      spec.a, rocblas_datatype_f32_r, spec.lda, spec.stride_a, spec.b,
      rocblas_datatype_f32_r, spec.ldb, spec.stride_b, &spec.beta, spec.d,
      rocblas_datatype_f32_r, spec.ldd, spec.stride_d, spec.d,
      rocblas_datatype_f32_r, spec.ldd, spec.stride_d, spec.batch_count,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
      rocblas_gemm_flags_none, /*list_array=*/nullptr, &list_size);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
  if (status != rocblas_status_success || list_size <= 0) return {};

  std::vector<rocblas_int> solutions(static_cast<size_t>(list_size));
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
  status = rocblas_gemm_strided_batched_ex_get_solutions(
      handle, spec.trans_a, spec.trans_b, spec.m, spec.n, spec.k, &spec.alpha,
      spec.a, rocblas_datatype_f32_r, spec.lda, spec.stride_a, spec.b,
      rocblas_datatype_f32_r, spec.ldb, spec.stride_b, &spec.beta, spec.d,
      rocblas_datatype_f32_r, spec.ldd, spec.stride_d, spec.d,
      rocblas_datatype_f32_r, spec.ldd, spec.stride_d, spec.batch_count,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
      rocblas_gemm_flags_none, solutions.data(), &list_size);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
  if (status != rocblas_status_success || list_size <= 0) return {};
  solutions.resize(static_cast<size_t>(list_size));
  return solutions;
}

rocblas_status RunStridedBatchedF32GemmSolution(
    rocblas_handle handle, const StridedBatchedGemmSpec& spec,
    int solution_index) {
  const rocblas_gemm_algo algo =
      solution_index == 0 ? rocblas_gemm_algo_standard
                          : rocblas_gemm_algo_solution_index;
  return rocblas_gemm_strided_batched_ex(
      handle, spec.trans_a, spec.trans_b, spec.m, spec.n, spec.k, &spec.alpha,
      spec.a, rocblas_datatype_f32_r, spec.lda, spec.stride_a, spec.b,
      rocblas_datatype_f32_r, spec.ldb, spec.stride_b, &spec.beta, spec.d,
      rocblas_datatype_f32_r, spec.ldd, spec.stride_d, spec.d,
      rocblas_datatype_f32_r, spec.ldd, spec.stride_d, spec.batch_count,
      rocblas_datatype_f32_r, algo, solution_index, rocblas_gemm_flags_none);
}

template <class Run>
bool TimeRocblasStatusMedian(const Args& args, const Run& run,
                             float& median_ms) {
  for (int i = 0; i < args.warmup; ++i) {
    for (int j = 0; j < args.iters; ++j) {
      if (run() != rocblas_status_success) return false;
    }
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  std::vector<float> ms;
  ms.reserve(args.samples);
  for (int sample = 0; sample < args.samples; ++sample) {
    HIP_CHECK(hipEventRecord(start));
    for (int iter = 0; iter < args.iters; ++iter) {
      if (run() != rocblas_status_success) {
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        return false;
      }
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    ms.push_back(elapsed_ms / static_cast<float>(args.iters));
  }
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  median_ms = MedianMs(ms);
  return true;
}

template <class Run>
float TimeHipMedian(const Args& args, const Run& run) {
  for (int i = 0; i < args.warmup; ++i) {
    for (int j = 0; j < args.iters; ++j) run();
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  std::vector<float> ms;
  ms.reserve(args.samples);
  for (int sample = 0; sample < args.samples; ++sample) {
    HIP_CHECK(hipEventRecord(start));
    for (int iter = 0; iter < args.iters; ++iter) run();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    ms.push_back(elapsed_ms / static_cast<float>(args.iters));
  }
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return MedianMs(ms);
}

template <int kWavesM, int kWavesN>
void LaunchAttentionQkBf16WmmaTransB(const rocblas_bfloat16* q,
                                     const rocblas_bfloat16* k, float* scores,
                                     int rows, int heads, int qkv_dim,
                                     float query_scale) {
  const dim3 block(32 * kWavesM * kWavesN);
  const dim3 grid((rows + 16 * kWavesN - 1) / (16 * kWavesN),
                  (rows + 16 * kWavesM - 1) / (16 * kWavesM), heads);
  hipLaunchKernelGGL(
      (Bf16WmmaGemmF32Grouped2DBatchedTransBKernel<kWavesM, kWavesN>), grid,
      block, 0, 0, q, k, scores, rows, qkv_dim, rows,
      static_cast<size_t>(rows) * qkv_dim,
      static_cast<size_t>(rows) * qkv_dim, static_cast<size_t>(rows) * rows,
      query_scale);
}

template <int kWavesM, int kWavesN>
void LaunchAttentionAvBf16Wmma(const rocblas_bfloat16* scores,
                               const rocblas_bfloat16* v, float* head_major,
                               int rows, int heads, int qkv_dim) {
  const dim3 block(32 * kWavesM * kWavesN);
  const dim3 grid((qkv_dim + 16 * kWavesN - 1) / (16 * kWavesN),
                  (rows + 16 * kWavesM - 1) / (16 * kWavesM), heads);
  hipLaunchKernelGGL(
      (Bf16WmmaGemmF32Grouped2DBatchedKernel<kWavesM, kWavesN>), grid, block,
      0, 0, scores, v, head_major, rows, rows, qkv_dim,
      static_cast<size_t>(rows) * rows,
      static_cast<size_t>(rows) * qkv_dim,
      static_cast<size_t>(rows) * qkv_dim);
}

std::pair<float, float> ErrorStats(const std::vector<float>& got,
                                   const std::vector<float>& expected) {
  float max_abs = 0.0f;
  double sum_sq = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float diff = got[i] - expected[i];
    max_abs = std::max(max_abs, std::fabs(diff));
    sum_sq += static_cast<double>(diff) * diff;
  }
  return {max_abs, static_cast<float>(std::sqrt(sum_sq / got.size()))};
}

template <int kWavesM, int kWavesN>
void ProfileAttentionQkBf16WmmaCandidate(
    const Args& args, const AttentionShape& shape, const char* label,
    const rocblas_bfloat16* d_q_bf16, const rocblas_bfloat16* d_k_bf16,
    float* d_scores_wmma, const std::vector<float>& f32_out,
    const std::vector<float>& bf16_rocblas_out, float f32_ms,
    float query_scale) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int scores_count = heads * rows * rows;
  const double flops = 2.0 * static_cast<double>(heads) * rows * rows * qkv_dim;

  auto run = [&]() {
    LaunchAttentionQkBf16WmmaTransB<kWavesM, kWavesN>(
        d_q_bf16, d_k_bf16, d_scores_wmma, rows, heads, qkv_dim, query_scale);
  };
  const float wmma_ms = TimeHipMedian(args, run);
  run();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> wmma_out(scores_count);
  HIP_CHECK(hipMemcpy(wmma_out.data(), d_scores_wmma,
                      static_cast<size_t>(scores_count) * sizeof(float),
                      hipMemcpyDeviceToHost));
  const auto [vs_f32_max, vs_f32_rms] = ErrorStats(wmma_out, f32_out);
  const auto [vs_bf16_max, vs_bf16_rms] =
      ErrorStats(wmma_out, bf16_rocblas_out);

  std::printf("  qk_wmma%-3s waves_m=%d waves_n=%d wmma=%8.3f "
              "rel_f32=%6.3f wmma_gflops=%8.1f "
              "vs_f32_max=%9.6f vs_f32_rms=%9.6f "
              "vs_bf16_max=%9.6f vs_bf16_rms=%9.6f\n",
              label, kWavesM, kWavesN, wmma_ms, wmma_ms / f32_ms,
              flops / (wmma_ms * 1.0E6), vs_f32_max, vs_f32_rms,
              vs_bf16_max, vs_bf16_rms);
}

template <int kWavesM, int kWavesN>
float ProfileAttentionAvBf16WmmaCandidate(
    const Args& args, const AttentionShape& shape, const char* label,
    const rocblas_bfloat16* d_scores_bf16, const rocblas_bfloat16* d_v_bf16,
    float* d_att_bf16_wmma, const std::vector<float>& f32_out,
    const std::vector<float>& bf16_rocblas_out, float f32_av_ms,
    float f32_total_ms, float softmax_bf16_ms, float quant_v_ms) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const double flops = 2.0 * static_cast<double>(heads) * rows * rows * qkv_dim;

  auto run = [&]() {
    LaunchAttentionAvBf16Wmma<kWavesM, kWavesN>(
        d_scores_bf16, d_v_bf16, d_att_bf16_wmma, rows, heads, qkv_dim);
  };
  const float wmma_ms = TimeHipMedian(args, run);
  run();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> wmma_out(qkv_count);
  HIP_CHECK(hipMemcpy(wmma_out.data(), d_att_bf16_wmma,
                      static_cast<size_t>(qkv_count) * sizeof(float),
                      hipMemcpyDeviceToHost));
  const auto [vs_f32_max, vs_f32_rms] = ErrorStats(wmma_out, f32_out);
  const auto [vs_bf16_max, vs_bf16_rms] =
      ErrorStats(wmma_out, bf16_rocblas_out);
  const float direct_total_ms = softmax_bf16_ms + quant_v_ms + wmma_ms;

  std::printf("  av_wmma%-3s waves_m=%d waves_n=%d wmma_av=%8.3f "
              "direct_total=%8.3f rel_f32_av=%6.3f rel_f32_total=%6.3f "
              "wmma_gflops=%8.1f vs_f32_max=%9.6f vs_f32_rms=%9.6f "
              "vs_bf16_max=%9.6f vs_bf16_rms=%9.6f\n",
              label, kWavesM, kWavesN, wmma_ms, direct_total_ms,
              wmma_ms / f32_av_ms, direct_total_ms / f32_total_ms,
              flops / (wmma_ms * 1.0E6), vs_f32_max, vs_f32_rms,
              vs_bf16_max, vs_bf16_rms);
  return wmma_ms;
}

template <int kRows, int kDim, int kGroups, class RunQk>
void ProfileAttentionSoftmaxAvGroupedCandidate(
    const Args& args, const AttentionShape& shape, const char* label,
    const RunQk& run_qk, float* d_scores, const float* d_v, float* d_att,
    const std::vector<float>& baseline_out, float baseline_ms, float qk_ms,
    float av_ms) {
  auto run_softmax_av = [&]() {
    hipLaunchKernelGGL((SoftmaxAvRowsGroupedKernel<kRows, kDim, kGroups>),
                       dim3(kRows, shape.heads), dim3(kGroups * 32), 0, 0,
                       d_scores, d_v, d_att);
  };
  auto run_total = [&]() {
    ROCBLAS_CHECK(run_qk());
    run_softmax_av();
  };

  ROCBLAS_CHECK(run_qk());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  const float softmax_av_ms = TimeHipMedian(args, run_softmax_av);
  const float total_ms = TimeHipMedian(args, run_total);

  run_total();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> grouped_out(baseline_out.size());
  HIP_CHECK(hipMemcpy(grouped_out.data(), d_att,
                      grouped_out.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  const auto [max_abs_error, rms_error] =
      ErrorStats(grouped_out, baseline_out);

  std::printf("  grouped_softmax_av%-3s groups=%2d qk=%8.3f av=%8.3f "
              "softmax_av=%8.3f baseline_total=%8.3f grouped_total=%8.3f "
              "rel_total=%6.3f max_abs=%9.6f rms=%9.6f\n",
              label, kGroups, qk_ms, av_ms, softmax_av_ms, baseline_ms,
              total_ms, total_ms / baseline_ms, max_abs_error, rms_error);
}

template <int BM, int BK, int kDim4>
void ProfileAttentionAvTiledDim4Candidate(
    const Args& args, const AttentionShape& shape, const char* label,
    const float* d_scores, const float* d_v, float* d_att,
    const std::vector<float>& rocblas_out, float rocblas_ms) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const double flops = 2.0 * static_cast<double>(heads) * rows * rows * qkv_dim;

  if (qkv_dim != kDim4 * 4) {
    std::printf("  av_dim4%-10s skipped: dim=%d expected=%d\n", label,
                qkv_dim, kDim4 * 4);
    return;
  }

  auto run = [&]() {
    const dim3 block(kDim4, BM);
    const dim3 grid((rows + BM - 1) / BM, heads);
    hipLaunchKernelGGL((AttentionAvTiledDim4Kernel<BM, BK, kDim4>), grid,
                       block, 0, 0, d_scores, d_v, d_att, rows);
  };

  const float dim4_ms = TimeHipMedian(args, run);
  run();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> dim4_out(qkv_count);
  HIP_CHECK(hipMemcpy(dim4_out.data(), d_att,
                      static_cast<size_t>(qkv_count) * sizeof(float),
                      hipMemcpyDeviceToHost));
  const auto [max_abs_error, rms_error] = ErrorStats(dim4_out, rocblas_out);

  std::printf("  av_dim4%-10s bm=%2d bk=%3d rocblas=%8.3f ms "
              "dim4=%8.3f ms rel=%6.3f rocblas_gflops=%8.1f "
              "dim4_gflops=%8.1f max_abs=%9.6f rms=%9.6f\n",
              label, BM, BK, rocblas_ms, dim4_ms, dim4_ms / rocblas_ms,
              flops / (rocblas_ms * 1.0E6), flops / (dim4_ms * 1.0E6),
              max_abs_error, rms_error);
}

int BenchOneAttentionRocblasSolutionSweep(
    rocblas_handle handle, const Args& args, int max_solutions,
    const StridedBatchedGemmSpec& spec) {
  std::vector<rocblas_int> solutions =
      QueryStridedBatchedF32GemmSolutions(handle, spec);
  std::vector<RocblasSolutionMeasurement> candidates;
  candidates.push_back({/*solution=*/0, /*failed=*/false, {}});

  int tested = 0;
  for (rocblas_int solution : solutions) {
    if (solution == 0) continue;
    if (tested >= max_solutions) break;
    candidates.push_back({static_cast<int>(solution), /*failed=*/false, {}});
    ++tested;
  }

  const double flops = 2.0 * static_cast<double>(spec.batch_count) * spec.m *
                       static_cast<double>(spec.n) * spec.k;
  std::printf("  %-10s m=%4d n=%4d k=%3d batch=%2d solutions=%zu "
              "tested=%d samples=%d warmup=%d iters=%d\n",
              spec.label, spec.m, spec.n, spec.k, spec.batch_count,
              solutions.size(), tested, args.samples, args.warmup,
              args.iters);

  for (RocblasSolutionMeasurement& candidate : candidates) {
    auto run_candidate = [&]() {
      return RunStridedBatchedF32GemmSolution(handle, spec,
                                             candidate.solution);
    };
    float median_ms = 0.0f;
    if (TimeRocblasStatusMedian(args, run_candidate, median_ms)) {
      candidate.samples_ms.push_back(median_ms);
    } else {
      candidate.failed = true;
    }
  }

  int best_solution = 0;
  float best_ms = INFINITY;
  float default_ms = INFINITY;
  for (const RocblasSolutionMeasurement& candidate : candidates) {
    if (candidate.failed || candidate.samples_ms.empty()) {
      std::printf("    %-8s solution=%7d failed\n", spec.label,
                  candidate.solution);
      continue;
    }
    const float median_ms = candidate.samples_ms[0];
    if (candidate.solution == 0) default_ms = median_ms;
    const char* solution_label =
        candidate.solution == 0 ? "default" : "solution";
    std::printf("    %-8s %-8s=%7d median=%8.3f ms %8.1f GFLOPS\n",
                spec.label, solution_label, candidate.solution, median_ms,
                flops / (median_ms * 1.0E6));
    if (median_ms < best_ms) {
      best_ms = median_ms;
      best_solution = candidate.solution;
    }
  }

  if (std::isfinite(best_ms)) {
    const float rel = std::isfinite(default_ms) && default_ms > 0.0f
                          ? best_ms / default_ms
                          : INFINITY;
    std::printf("    %-8s selected=%7d median=%8.3f ms rel_to_default=%6.3f\n",
                spec.label, best_solution, best_ms, rel);
  }
  return best_solution;
}

void BenchAttentionPhaseProfile(rocblas_handle handle,
                                const AttentionShape& shape,
                                const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q(qkv_count);
  DeviceBuffer<float> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_scores(scores_count);
  DeviceBuffer<float> d_att_head(qkv_count);
  DeviceBuffer<float> d_att_row(qkv_count);

  const int threads = 256;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const size_t softmax_shared = static_cast<size_t>(threads) * sizeof(float);
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;

  auto run_split = [&]() {
    hipLaunchKernelGGL(SplitQKVForAttentionKernel, dim3(qkv_blocks),
                       dim3(threads), 0, 0, d_qkv.ptr, d_q.ptr, d_k.ptr,
                       d_v.ptr, rows, heads, qkv_dim);
  };
  auto run_qk = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
        qkv_dim, &alpha_qk, d_k.ptr, rocblas_datatype_f32_r, qkv_dim,
        stride_qkv, d_q.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        &beta, d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, heads,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };
  auto run_softmax = [&]() {
    hipLaunchKernelGGL(SoftmaxRowsKernel, dim3(rows, heads), dim3(threads),
                       softmax_shared, 0, d_scores.ptr, rows);
  };
  auto run_av = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
        rows, &alpha_av, d_v.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, &beta,
        d_att_head.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        d_att_head.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv, heads,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };
  auto run_pack = [&]() {
    hipLaunchKernelGGL(PackAttentionHeadsKernel, dim3(qkv_blocks),
                       dim3(threads), 0, 0, d_att_head.ptr, d_att_row.ptr,
                       rows, heads, qkv_dim);
  };
  auto run_total = [&]() {
    run_split();
    run_qk();
    run_softmax();
    run_av();
    run_pack();
  };

  run_total();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return MedianMs(ms);
  };

  const float split_ms = time_device(run_split);
  const float qk_ms = time_device(run_qk);
  const float softmax_ms = time_device(run_softmax);
  const float av_ms = time_device(run_av);
  const float pack_ms = time_device(run_pack);
  const float total_ms = time_device(run_total);
  const float phase_sum = split_ms + qk_ms + softmax_ms + av_ms + pack_ms;
  std::printf("%-24s attention phases rows=%4d heads=%2d dim=%2d "
              "split=%8.3f qk=%8.3f softmax=%8.3f av=%8.3f "
              "pack=%8.3f sum=%8.3f total=%8.3f ms\n",
              shape.name, rows, heads, qkv_dim, split_ms, qk_ms, softmax_ms,
              av_ms, pack_ms, phase_sum, total_ms);
}

void BenchAttentionRecommendedPhaseProfile(rocblas_handle handle,
                                           const AttentionShape& shape,
                                           const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q(qkv_count);
  DeviceBuffer<float> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_scores(scores_count);
  DeviceBuffer<float> d_scores_direct(scores_count);
  DeviceBuffer<float> d_att_head(qkv_count);
  DeviceBuffer<float> d_att_head_direct(qkv_count);
  DeviceBuffer<rocblas_bfloat16> d_att_row_bf16(qkv_count);
  DeviceBuffer<rocblas_bfloat16> d_att_row_direct_bf16(qkv_count);

  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_qkv_direct =
      static_cast<rocblas_stride>(3) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;
  // These are the current gfx1150 cache entries generated by
  // scripts/build_paligemma2_vit_attention_solution_cache.sh.
  const int qk_solution = RecommendedQkSolution(rows);
  const int av_solution = RecommendedAvSolution(rows);
  const rocblas_gemm_algo qk_algo =
      qk_solution == 0 ? rocblas_gemm_algo_standard
                       : rocblas_gemm_algo_solution_index;
  const rocblas_gemm_algo av_algo =
      av_solution == 0 ? rocblas_gemm_algo_standard
                       : rocblas_gemm_algo_solution_index;

  auto run_split = [&]() {
    LaunchRecommendedSplitQKV(d_qkv.ptr, d_q.ptr, d_k.ptr, d_v.ptr, rows,
                              heads, qkv_dim);
  };
  auto run_qk = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
        qkv_dim, &alpha_qk, d_k.ptr, rocblas_datatype_f32_r, qkv_dim,
        stride_qkv, d_q.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        &beta, d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, heads,
        rocblas_datatype_f32_r, qk_algo, qk_solution, /*flags=*/0));
  };
  auto run_qk_direct = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
        qkv_dim, &alpha_qk, d_qkv.ptr + qkv_dim, rocblas_datatype_f32_r,
        qkv_cols, stride_qkv_direct, d_qkv.ptr, rocblas_datatype_f32_r,
        qkv_cols, stride_qkv_direct, &beta, d_scores_direct.ptr,
        rocblas_datatype_f32_r, rows, stride_scores, d_scores_direct.ptr,
        rocblas_datatype_f32_r, rows, stride_scores, heads,
        rocblas_datatype_f32_r, qk_algo, qk_solution, /*flags=*/0));
  };
  auto run_softmax = [&]() {
    LaunchRecommendedSoftmax(d_scores.ptr, rows, heads);
  };
  auto run_softmax_direct = [&]() {
    LaunchRecommendedSoftmax(d_scores_direct.ptr, rows, heads);
  };
  auto run_av = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
        rows, &alpha_av, d_v.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, &beta,
        d_att_head.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        d_att_head.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv, heads,
        rocblas_datatype_f32_r, av_algo, av_solution, /*flags=*/0));
  };
  auto run_av_direct = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
        rows, &alpha_av, d_qkv.ptr + 2 * qkv_dim, rocblas_datatype_f32_r,
        qkv_cols, stride_qkv_direct, d_scores_direct.ptr,
        rocblas_datatype_f32_r, rows, stride_scores, &beta,
        d_att_head_direct.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        d_att_head_direct.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        heads, rocblas_datatype_f32_r, av_algo, av_solution, /*flags=*/0));
  };
  auto run_pack_bf16 = [&]() {
    LaunchAttentionPackBF16(d_att_head.ptr, d_att_row_bf16.ptr, rows, heads,
                            qkv_dim);
  };
  auto run_pack_direct_bf16 = [&]() {
    LaunchAttentionPackBF16(d_att_head_direct.ptr, d_att_row_direct_bf16.ptr,
                            rows, heads, qkv_dim);
  };
  auto run_total = [&]() {
    run_split();
    run_qk();
    run_softmax();
    run_av();
    run_pack_bf16();
  };
  auto run_direct_total = [&]() {
    run_qk_direct();
    run_softmax_direct();
    run_av_direct();
    run_pack_direct_bf16();
  };

  run_total();
  run_direct_total();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return MedianMs(ms);
  };

  const float split_ms = time_device(run_split);
  const float qk_ms = time_device(run_qk);
  const float qk_direct_ms = time_device(run_qk_direct);
  const float softmax_ms = time_device(run_softmax);
  const float av_ms = time_device(run_av);
  const float av_direct_ms = time_device(run_av_direct);
  const float pack_bf16_ms = time_device(run_pack_bf16);
  const float total_ms = time_device(run_total);
  const float direct_total_ms = time_device(run_direct_total);
  const float phase_sum =
      split_ms + qk_ms + softmax_ms + av_ms + pack_bf16_ms;
  const float direct_phase_sum =
      qk_direct_ms + softmax_ms + av_direct_ms + pack_bf16_ms;

  run_total();
  run_direct_total();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> att(qkv_count);
  std::vector<float> att_direct(qkv_count);
  HIP_CHECK(hipMemcpy(att.data(), d_att_head.ptr,
                      static_cast<size_t>(qkv_count) * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(att_direct.data(), d_att_head_direct.ptr,
                      static_cast<size_t>(qkv_count) * sizeof(float),
                      hipMemcpyDeviceToHost));
  float max_abs_error = 0.0f;
  double sum_sq = 0.0;
  for (size_t i = 0; i < att.size(); ++i) {
    const float diff = att_direct[i] - att[i];
    max_abs_error = std::max(max_abs_error, std::fabs(diff));
    sum_sq += static_cast<double>(diff) * diff;
  }
  const float rms_error =
      static_cast<float>(std::sqrt(sum_sq / att.size()));

  std::printf("%-24s attention recommended phases rows=%4d heads=%2d dim=%2d "
              "qk_solution=%7d av_solution=%7d split=%8.3f qk=%8.3f "
              "softmax=%8.3f av=%8.3f pack_bf16=%8.3f sum=%8.3f "
              "total=%8.3f direct_qk=%8.3f direct_av=%8.3f "
              "direct_sum=%8.3f direct_total=%8.3f "
              "direct_max_abs=%9.6f direct_rms=%9.6f ms\n",
              shape.name, rows, heads, qkv_dim, qk_solution, av_solution,
              split_ms, qk_ms, softmax_ms, av_ms, pack_bf16_ms, phase_sum,
              total_ms, qk_direct_ms, av_direct_ms, direct_phase_sum,
              direct_total_ms, max_abs_error, rms_error);
}

void BenchAttentionDeferredSoftmaxScaleProfile(rocblas_handle handle,
                                               const AttentionShape& shape,
                                               const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;
  const int inv_sum_count = heads * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q(qkv_count);
  DeviceBuffer<float> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_scores(scores_count);
  DeviceBuffer<float> d_inv_sums(inv_sum_count);
  DeviceBuffer<float> d_att_head(qkv_count);
  DeviceBuffer<rocblas_bfloat16> d_att_row_bf16(qkv_count);
  DeviceBuffer<rocblas_bfloat16> d_att_row_deferred_bf16(qkv_count);

  const int threads = 256;
  const int softmax_threads = rows <= 256 ? 64 : 256;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const size_t softmax_shared =
      static_cast<size_t>(softmax_threads) * sizeof(float);
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;
  const int qk_solution = -110;
  const int av_solution = rows <= 256 ? -1161823143 : -451;
  const rocblas_gemm_algo qk_algo =
      qk_solution == 0 ? rocblas_gemm_algo_standard
                       : rocblas_gemm_algo_solution_index;
  const rocblas_gemm_algo av_algo =
      av_solution == 0 ? rocblas_gemm_algo_standard
                       : rocblas_gemm_algo_solution_index;

  auto run_split = [&]() {
    hipLaunchKernelGGL(SplitQKVForAttentionKernel, dim3(qkv_blocks),
                       dim3(threads), 0, 0, d_qkv.ptr, d_q.ptr, d_k.ptr,
                       d_v.ptr, rows, heads, qkv_dim);
  };
  auto run_qk = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
        qkv_dim, &alpha_qk, d_k.ptr, rocblas_datatype_f32_r, qkv_dim,
        stride_qkv, d_q.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        &beta, d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, heads,
        rocblas_datatype_f32_r, qk_algo, qk_solution, /*flags=*/0));
  };
  auto run_softmax = [&]() {
    hipLaunchKernelGGL(SoftmaxRowsKernel, dim3(rows, heads),
                       dim3(softmax_threads), softmax_shared, 0, d_scores.ptr,
                       rows);
  };
  auto run_softmax_deferred = [&]() {
    hipLaunchKernelGGL(SoftmaxRowsStoreExpAndInvSumKernel, dim3(rows, heads),
                       dim3(softmax_threads), softmax_shared, 0, d_scores.ptr,
                       d_inv_sums.ptr, rows);
  };
  auto run_av = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
        rows, &alpha_av, d_v.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, &beta,
        d_att_head.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        d_att_head.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv, heads,
        rocblas_datatype_f32_r, av_algo, av_solution, /*flags=*/0));
  };
  auto run_pack_bf16 = [&]() {
    LaunchAttentionPackBF16(d_att_head.ptr, d_att_row_bf16.ptr, rows, heads,
                            qkv_dim);
  };
  auto run_pack_scaled_bf16 = [&]() {
    hipLaunchKernelGGL(PackScaledAttentionHeadsToBF16Kernel, dim3(qkv_blocks),
                       dim3(threads), 0, 0, d_att_head.ptr, d_inv_sums.ptr,
                       d_att_row_deferred_bf16.ptr, rows, heads, qkv_dim);
  };
  auto run_current_total = [&]() {
    run_split();
    run_qk();
    run_softmax();
    run_av();
    run_pack_bf16();
  };
  auto run_deferred_total = [&]() {
    run_split();
    run_qk();
    run_softmax_deferred();
    run_av();
    run_pack_scaled_bf16();
  };

  run_current_total();
  run_deferred_total();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return MedianMs(ms);
  };

  const float softmax_ms = time_device(run_softmax);
  const float softmax_deferred_ms = time_device(run_softmax_deferred);
  const float pack_bf16_ms = time_device(run_pack_bf16);
  const float pack_scaled_bf16_ms = time_device(run_pack_scaled_bf16);
  const float current_total_ms = time_device(run_current_total);
  const float deferred_total_ms = time_device(run_deferred_total);

  run_current_total();
  run_deferred_total();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<rocblas_bfloat16> current(qkv_count);
  std::vector<rocblas_bfloat16> deferred(qkv_count);
  HIP_CHECK(hipMemcpy(current.data(), d_att_row_bf16.ptr,
                      qkv_count * sizeof(rocblas_bfloat16),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(deferred.data(), d_att_row_deferred_bf16.ptr,
                      qkv_count * sizeof(rocblas_bfloat16),
                      hipMemcpyDeviceToHost));

  float max_abs_error = 0.0f;
  double sum_sq = 0.0;
  for (size_t i = 0; i < current.size(); ++i) {
    const float diff = static_cast<float>(deferred[i]) -
                       static_cast<float>(current[i]);
    max_abs_error = std::max(max_abs_error, std::fabs(diff));
    sum_sq += static_cast<double>(diff) * diff;
  }
  const float rms_error =
      static_cast<float>(std::sqrt(sum_sq / current.size()));
  const float current_tail_ms = softmax_ms + pack_bf16_ms;
  const float deferred_tail_ms = softmax_deferred_ms + pack_scaled_bf16_ms;
  std::printf("%-24s attention deferred softmax-scale rows=%4d heads=%2d "
              "dim=%2d softmax_norm=%8.3f softmax_exp=%8.3f "
              "pack_bf16=%8.3f pack_scaled_bf16=%8.3f tail=%8.3f/%8.3f "
              "total=%8.3f/%8.3f rel_total=%6.3f max_abs=%9.6f "
              "rms=%9.6f\n",
              shape.name, rows, heads, qkv_dim, softmax_ms,
              softmax_deferred_ms, pack_bf16_ms, pack_scaled_bf16_ms,
              current_tail_ms, deferred_tail_ms, current_total_ms,
              deferred_total_ms, deferred_total_ms / current_total_ms,
              max_abs_error, rms_error);
}

void BenchAttentionBf16PhaseProfile(rocblas_handle handle,
                                    const AttentionShape& shape,
                                    const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<rocblas_bfloat16> d_q(qkv_count);
  DeviceBuffer<rocblas_bfloat16> d_k(qkv_count);
  DeviceBuffer<rocblas_bfloat16> d_v(qkv_count);
  DeviceBuffer<float> d_scores(scores_count);
  DeviceBuffer<rocblas_bfloat16> d_scores_bf16(scores_count);
  DeviceBuffer<float> d_att_head(qkv_count);
  DeviceBuffer<float> d_att_row(qkv_count);

  const int threads = 256;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const int scores_blocks = (scores_count + threads - 1) / threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const size_t softmax_shared = static_cast<size_t>(threads) * sizeof(float);
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;

  auto run_split = [&]() {
    hipLaunchKernelGGL(SplitQKVToBF16ForAttentionKernel, dim3(qkv_blocks),
                       dim3(threads), 0, 0, d_qkv.ptr, d_q.ptr, d_k.ptr,
                       d_v.ptr, rows, heads, qkv_dim);
  };
  auto run_qk = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
        qkv_dim, &alpha_qk, d_k.ptr, rocblas_datatype_bf16_r, qkv_dim,
        stride_qkv, d_q.ptr, rocblas_datatype_bf16_r, qkv_dim, stride_qkv,
        &beta, d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, heads,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };
  auto run_softmax = [&]() {
    hipLaunchKernelGGL(SoftmaxRowsKernel, dim3(rows, heads), dim3(threads),
                       softmax_shared, 0, d_scores.ptr, rows);
  };
  auto run_quant_scores = [&]() {
    hipLaunchKernelGGL(F32ToBF16Kernel, dim3(scores_blocks), dim3(threads), 0,
                       0, d_scores.ptr, d_scores_bf16.ptr, scores_count);
  };
  auto run_av = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
        rows, &alpha_av, d_v.ptr, rocblas_datatype_bf16_r, qkv_dim, stride_qkv,
        d_scores_bf16.ptr, rocblas_datatype_bf16_r, rows, stride_scores, &beta,
        d_att_head.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        d_att_head.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv, heads,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };
  auto run_pack = [&]() {
    hipLaunchKernelGGL(PackAttentionHeadsKernel, dim3(qkv_blocks),
                       dim3(threads), 0, 0, d_att_head.ptr, d_att_row.ptr,
                       rows, heads, qkv_dim);
  };
  auto run_total = [&]() {
    run_split();
    run_qk();
    run_softmax();
    run_quant_scores();
    run_av();
    run_pack();
  };

  run_total();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return MedianMs(ms);
  };

  const float split_ms = time_device(run_split);
  const float qk_ms = time_device(run_qk);
  const float softmax_ms = time_device(run_softmax);
  const float quant_ms = time_device(run_quant_scores);
  const float av_ms = time_device(run_av);
  const float pack_ms = time_device(run_pack);
  const float total_ms = time_device(run_total);
  const float phase_sum =
      split_ms + qk_ms + softmax_ms + quant_ms + av_ms + pack_ms;
  std::printf("%-24s attention bf16 phases rows=%4d heads=%2d dim=%2d "
              "split_bf16=%8.3f qk_bf16=%8.3f softmax=%8.3f "
              "quant_scores=%8.3f av_bf16=%8.3f pack=%8.3f "
              "sum=%8.3f total=%8.3f ms\n",
              shape.name, rows, heads, qkv_dim, split_ms, qk_ms, softmax_ms,
              quant_ms, av_ms, pack_ms, phase_sum, total_ms);
}

void BenchAttentionBf16QkPhaseProfile(rocblas_handle handle,
                                      const AttentionShape& shape,
                                      const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<rocblas_bfloat16> d_q(qkv_count);
  DeviceBuffer<rocblas_bfloat16> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_scores(scores_count);
  DeviceBuffer<float> d_att_head(qkv_count);
  DeviceBuffer<float> d_att_row(qkv_count);

  const int threads = 256;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const size_t softmax_shared = static_cast<size_t>(threads) * sizeof(float);
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;

  auto run_split = [&]() {
    hipLaunchKernelGGL(SplitQKToBF16VFloatForAttentionKernel,
                       dim3(qkv_blocks), dim3(threads), 0, 0, d_qkv.ptr,
                       d_q.ptr, d_k.ptr, d_v.ptr, rows, heads, qkv_dim);
  };
  auto run_qk = [&]() {
    // Only the score GEMM changes datatype in this experiment. The output
    // remains F32 so the existing softmax kernel can run unchanged and can feed
    // the F32 AV GEMM without quantizing the rows x rows score matrix.
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
        qkv_dim, &alpha_qk, d_k.ptr, rocblas_datatype_bf16_r, qkv_dim,
        stride_qkv, d_q.ptr, rocblas_datatype_bf16_r, qkv_dim, stride_qkv,
        &beta, d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, heads,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };
  auto run_softmax = [&]() {
    hipLaunchKernelGGL(SoftmaxRowsKernel, dim3(rows, heads), dim3(threads),
                       softmax_shared, 0, d_scores.ptr, rows);
  };
  auto run_av = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
        rows, &alpha_av, d_v.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, &beta,
        d_att_head.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        d_att_head.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv, heads,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };
  auto run_pack = [&]() {
    hipLaunchKernelGGL(PackAttentionHeadsKernel, dim3(qkv_blocks),
                       dim3(threads), 0, 0, d_att_head.ptr, d_att_row.ptr,
                       rows, heads, qkv_dim);
  };
  auto run_total = [&]() {
    run_split();
    run_qk();
    run_softmax();
    run_av();
    run_pack();
  };

  run_total();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return MedianMs(ms);
  };

  const float split_ms = time_device(run_split);
  const float qk_ms = time_device(run_qk);
  const float softmax_ms = time_device(run_softmax);
  const float av_ms = time_device(run_av);
  const float pack_ms = time_device(run_pack);
  const float total_ms = time_device(run_total);
  const float phase_sum = split_ms + qk_ms + softmax_ms + av_ms + pack_ms;
  std::printf("%-24s attention bf16-qk phases rows=%4d heads=%2d dim=%2d "
              "split_qk_bf16_v_f32=%8.3f qk_bf16=%8.3f softmax=%8.3f "
              "av_f32=%8.3f pack=%8.3f sum=%8.3f total=%8.3f ms\n",
              shape.name, rows, heads, qkv_dim, split_ms, qk_ms, softmax_ms,
              av_ms, pack_ms, phase_sum, total_ms);
}

void BenchAttentionQkBf16WmmaProfile(rocblas_handle handle,
                                     const AttentionShape& shape,
                                     const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q_f32(qkv_count);
  DeviceBuffer<float> d_k_f32(qkv_count);
  DeviceBuffer<float> d_v_f32(qkv_count);
  DeviceBuffer<rocblas_bfloat16> d_q_bf16(qkv_count);
  DeviceBuffer<rocblas_bfloat16> d_k_bf16(qkv_count);
  DeviceBuffer<float> d_v_unused(qkv_count);
  DeviceBuffer<float> d_scores_f32(scores_count);
  DeviceBuffer<float> d_scores_bf16_rocblas(scores_count);
  DeviceBuffer<float> d_scores_wmma(scores_count);

  const int threads = 256;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float beta = 0.0f;
  const int qk_solution = RecommendedQkSolution(rows);

  std::vector<float> h_qkv(qkv_projection_count);
  FillNonBf16Exact(h_qkv);
  HIP_CHECK(hipMemcpy(d_qkv.ptr, h_qkv.data(),
                      static_cast<size_t>(qkv_projection_count) *
                          sizeof(float),
                      hipMemcpyHostToDevice));

  hipLaunchKernelGGL(SplitQKVForAttentionKernel, dim3(qkv_blocks),
                     dim3(threads), 0, 0, d_qkv.ptr, d_q_f32.ptr,
                     d_k_f32.ptr, d_v_f32.ptr, rows, heads, qkv_dim);
  hipLaunchKernelGGL(SplitQKToBF16VFloatForAttentionKernel, dim3(qkv_blocks),
                     dim3(threads), 0, 0, d_qkv.ptr, d_q_bf16.ptr,
                     d_k_bf16.ptr, d_v_unused.ptr, rows, heads, qkv_dim);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  StridedBatchedGemmSpec qk_f32;
  qk_f32.label = "qk_f32";
  qk_f32.trans_a = rocblas_operation_transpose;
  qk_f32.trans_b = rocblas_operation_none;
  qk_f32.m = rows;
  qk_f32.n = rows;
  qk_f32.k = qkv_dim;
  qk_f32.a = d_k_f32.ptr;
  qk_f32.lda = qkv_dim;
  qk_f32.stride_a = stride_qkv;
  qk_f32.b = d_q_f32.ptr;
  qk_f32.ldb = qkv_dim;
  qk_f32.stride_b = stride_qkv;
  qk_f32.d = d_scores_f32.ptr;
  qk_f32.ldd = rows;
  qk_f32.stride_d = stride_scores;
  qk_f32.batch_count = heads;
  qk_f32.alpha = query_scale;
  qk_f32.beta = 0.0f;

  auto run_f32_rocblas = [&]() {
    return RunStridedBatchedF32GemmSolution(handle, qk_f32, qk_solution);
  };
  auto run_bf16_rocblas = [&]() {
    return rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
        qkv_dim, &query_scale, d_k_bf16.ptr, rocblas_datatype_bf16_r, qkv_dim,
        stride_qkv, d_q_bf16.ptr, rocblas_datatype_bf16_r, qkv_dim,
        stride_qkv, &beta, d_scores_bf16_rocblas.ptr, rocblas_datatype_f32_r,
        rows, stride_scores, d_scores_bf16_rocblas.ptr,
        rocblas_datatype_f32_r, rows, stride_scores, heads,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0);
  };

  float f32_ms = 0.0f;
  float bf16_rocblas_ms = 0.0f;
  if (!TimeRocblasStatusMedian(args, run_f32_rocblas, f32_ms)) {
    std::printf("  qk F32 rocBLAS solution=%d failed\n", qk_solution);
    return;
  }
  if (!TimeRocblasStatusMedian(args, run_bf16_rocblas, bf16_rocblas_ms)) {
    std::printf("  qk BF16 rocBLAS failed\n");
    return;
  }

  ROCBLAS_CHECK(run_f32_rocblas());
  ROCBLAS_CHECK(run_bf16_rocblas());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> f32_out(scores_count);
  std::vector<float> bf16_rocblas_out(scores_count);
  HIP_CHECK(hipMemcpy(f32_out.data(), d_scores_f32.ptr,
                      static_cast<size_t>(scores_count) * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(bf16_rocblas_out.data(), d_scores_bf16_rocblas.ptr,
                      static_cast<size_t>(scores_count) * sizeof(float),
                      hipMemcpyDeviceToHost));

  const auto [bf16_vs_f32_max, bf16_vs_f32_rms] =
      ErrorStats(bf16_rocblas_out, f32_out);
  const double flops = 2.0 * static_cast<double>(heads) * rows * rows * qkv_dim;
  std::printf("%-24s attention QK BF16 WMMA tile sweep rows=%4d heads=%2d "
              "dim=%2d f32_solution=%7d f32=%8.3f bf16_rocblas=%8.3f "
              "f32_gflops=%8.1f bf16_rocblas_gflops=%8.1f "
              "bf16_vs_f32_max=%9.6f bf16_vs_f32_rms=%9.6f\n",
              shape.name, rows, heads, qkv_dim, qk_solution, f32_ms,
              bf16_rocblas_ms, flops / (f32_ms * 1.0E6),
              flops / (bf16_rocblas_ms * 1.0E6), bf16_vs_f32_max,
              bf16_vs_f32_rms);

  ProfileAttentionQkBf16WmmaCandidate<1, 4>(
      args, shape, "1x4", d_q_bf16.ptr, d_k_bf16.ptr, d_scores_wmma.ptr,
      f32_out, bf16_rocblas_out, f32_ms, query_scale);
  ProfileAttentionQkBf16WmmaCandidate<1, 8>(
      args, shape, "1x8", d_q_bf16.ptr, d_k_bf16.ptr, d_scores_wmma.ptr,
      f32_out, bf16_rocblas_out, f32_ms, query_scale);
  ProfileAttentionQkBf16WmmaCandidate<2, 2>(
      args, shape, "2x2", d_q_bf16.ptr, d_k_bf16.ptr, d_scores_wmma.ptr,
      f32_out, bf16_rocblas_out, f32_ms, query_scale);
  ProfileAttentionQkBf16WmmaCandidate<2, 4>(
      args, shape, "2x4", d_q_bf16.ptr, d_k_bf16.ptr, d_scores_wmma.ptr,
      f32_out, bf16_rocblas_out, f32_ms, query_scale);
  ProfileAttentionQkBf16WmmaCandidate<2, 8>(
      args, shape, "2x8", d_q_bf16.ptr, d_k_bf16.ptr, d_scores_wmma.ptr,
      f32_out, bf16_rocblas_out, f32_ms, query_scale);
  ProfileAttentionQkBf16WmmaCandidate<4, 2>(
      args, shape, "4x2", d_q_bf16.ptr, d_k_bf16.ptr, d_scores_wmma.ptr,
      f32_out, bf16_rocblas_out, f32_ms, query_scale);
  ProfileAttentionQkBf16WmmaCandidate<4, 4>(
      args, shape, "4x4", d_q_bf16.ptr, d_k_bf16.ptr, d_scores_wmma.ptr,
      f32_out, bf16_rocblas_out, f32_ms, query_scale);
  ProfileAttentionQkBf16WmmaCandidate<4, 8>(
      args, shape, "4x8", d_q_bf16.ptr, d_k_bf16.ptr, d_scores_wmma.ptr,
      f32_out, bf16_rocblas_out, f32_ms, query_scale);
  ProfileAttentionQkBf16WmmaCandidate<8, 1>(
      args, shape, "8x1", d_q_bf16.ptr, d_k_bf16.ptr, d_scores_wmma.ptr,
      f32_out, bf16_rocblas_out, f32_ms, query_scale);
  ProfileAttentionQkBf16WmmaCandidate<8, 2>(
      args, shape, "8x2", d_q_bf16.ptr, d_k_bf16.ptr, d_scores_wmma.ptr,
      f32_out, bf16_rocblas_out, f32_ms, query_scale);
  ProfileAttentionQkBf16WmmaCandidate<8, 4>(
      args, shape, "8x4", d_q_bf16.ptr, d_k_bf16.ptr, d_scores_wmma.ptr,
      f32_out, bf16_rocblas_out, f32_ms, query_scale);
}

void BenchAttentionQkSoftmaxFusedProfile(rocblas_handle handle,
                                         const AttentionShape& shape,
                                         const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q(qkv_count);
  DeviceBuffer<float> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_scores_baseline(scores_count);
  DeviceBuffer<float> d_scores_fused(scores_count);

  const int threads = 256;
  const int fused_threads = rows <= 256 ? 64 : 256;
  const int softmax_threads = rows <= 256 ? 64 : 256;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const size_t fused_shared =
      static_cast<size_t>(rows + fused_threads) * sizeof(float);
  const size_t softmax_shared =
      static_cast<size_t>(softmax_threads) * sizeof(float);
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const int qk_solution = -110;

  hipLaunchKernelGGL(SplitQKVForAttentionKernel, dim3(qkv_blocks),
                     dim3(threads), 0, 0, d_qkv.ptr, d_q.ptr, d_k.ptr,
                     d_v.ptr, rows, heads, qkv_dim);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  StridedBatchedGemmSpec qk;
  qk.label = "qk";
  qk.trans_a = rocblas_operation_transpose;
  qk.trans_b = rocblas_operation_none;
  qk.m = rows;
  qk.n = rows;
  qk.k = qkv_dim;
  qk.a = d_k.ptr;
  qk.lda = qkv_dim;
  qk.stride_a = stride_qkv;
  qk.b = d_q.ptr;
  qk.ldb = qkv_dim;
  qk.stride_b = stride_qkv;
  qk.d = d_scores_baseline.ptr;
  qk.ldd = rows;
  qk.stride_d = stride_scores;
  qk.batch_count = heads;
  qk.alpha = query_scale;
  qk.beta = 0.0f;

  auto run_qk = [&]() {
    return RunStridedBatchedF32GemmSolution(handle, qk, qk_solution);
  };
  auto run_softmax = [&]() {
    if (rows == 256) {
      hipLaunchKernelGGL((SoftmaxRowsFixedKernel<256, 64>), dim3(rows, heads),
                         dim3(64), 0, 0, d_scores_baseline.ptr);
    } else if (rows == 1024) {
      hipLaunchKernelGGL((SoftmaxRowsFixedKernel<1024, 256>),
                         dim3(rows, heads), dim3(256), 0, 0,
                         d_scores_baseline.ptr);
    } else {
      hipLaunchKernelGGL(SoftmaxRowsKernel, dim3(rows, heads),
                         dim3(softmax_threads), softmax_shared, 0,
                         d_scores_baseline.ptr, rows);
    }
  };
  auto run_baseline = [&]() {
    ROCBLAS_CHECK(run_qk());
    run_softmax();
  };
  auto run_fused = [&]() {
    hipLaunchKernelGGL(AttentionQkSoftmaxRowsKernel, dim3(rows, heads),
                       dim3(fused_threads), fused_shared, 0, d_q.ptr, d_k.ptr,
                       d_scores_fused.ptr, rows, qkv_dim, query_scale);
  };

  run_baseline();
  run_fused();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return MedianMs(ms);
  };

  float qk_ms = 0.0f;
  if (!TimeRocblasStatusMedian(args, run_qk, qk_ms)) {
    std::printf("  qk cached rocBLAS solution=%d failed\n", qk_solution);
    return;
  }
  const float baseline_ms = time_device(run_baseline);
  const float fused_ms = time_device(run_fused);

  run_baseline();
  run_fused();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> baseline_out(scores_count);
  std::vector<float> fused_out(scores_count);
  HIP_CHECK(hipMemcpy(baseline_out.data(), d_scores_baseline.ptr,
                      static_cast<size_t>(scores_count) * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(fused_out.data(), d_scores_fused.ptr,
                      static_cast<size_t>(scores_count) * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_abs_error = 0.0f;
  double sum_sq = 0.0;
  for (size_t i = 0; i < baseline_out.size(); ++i) {
    const float diff = fused_out[i] - baseline_out[i];
    max_abs_error = std::max(max_abs_error, std::fabs(diff));
    sum_sq += static_cast<double>(diff) * diff;
  }
  const float rms_error =
      static_cast<float>(std::sqrt(sum_sq / baseline_out.size()));

  std::printf("%-24s attention QK+softmax fused rows=%4d heads=%2d dim=%2d "
              "qk_solution=%7d qk=%8.3f qk_softmax=%8.3f fused=%8.3f "
              "rel=%6.3f max_abs=%9.6f rms=%9.6f\n",
              shape.name, rows, heads, qkv_dim, qk_solution, qk_ms,
              baseline_ms, fused_ms, fused_ms / baseline_ms, max_abs_error,
              rms_error);
}

void BenchAttentionSoftmaxAvFusedProfile(rocblas_handle handle,
                                         const AttentionShape& shape,
                                         const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q(qkv_count);
  DeviceBuffer<float> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_scores(scores_count);
  DeviceBuffer<float> d_att_head(qkv_count);
  DeviceBuffer<float> d_att_row(qkv_count);

  const int threads = 256;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const size_t softmax_shared = static_cast<size_t>(threads) * sizeof(float);
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float beta = 0.0f;

  auto run_split = [&]() {
    hipLaunchKernelGGL(SplitQKVForAttentionKernel, dim3(qkv_blocks),
                       dim3(threads), 0, 0, d_qkv.ptr, d_q.ptr, d_k.ptr,
                       d_v.ptr, rows, heads, qkv_dim);
  };
  auto run_qk = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
        qkv_dim, &alpha_qk, d_k.ptr, rocblas_datatype_f32_r, qkv_dim,
        stride_qkv, d_q.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        &beta, d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, heads,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };
  auto run_softmax_av = [&]() {
    hipLaunchKernelGGL(SoftmaxAvRowsKernel, dim3(rows, heads), dim3(threads),
                       softmax_shared, 0, d_scores.ptr, d_v.ptr,
                       d_att_head.ptr, rows, qkv_dim);
  };
  auto run_pack = [&]() {
    hipLaunchKernelGGL(PackAttentionHeadsKernel, dim3(qkv_blocks),
                       dim3(threads), 0, 0, d_att_head.ptr, d_att_row.ptr,
                       rows, heads, qkv_dim);
  };
  auto run_total = [&]() {
    run_split();
    run_qk();
    run_softmax_av();
    run_pack();
  };

  run_total();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return MedianMs(ms);
  };

  const float split_ms = time_device(run_split);
  const float qk_ms = time_device(run_qk);
  const float softmax_av_ms = time_device(run_softmax_av);
  const float pack_ms = time_device(run_pack);
  const float total_ms = time_device(run_total);
  const float phase_sum = split_ms + qk_ms + softmax_av_ms + pack_ms;
  std::printf("%-24s attention softmax-av fused rows=%4d heads=%2d dim=%2d "
              "split=%8.3f qk=%8.3f softmax_av=%8.3f pack=%8.3f "
              "sum=%8.3f total=%8.3f ms\n",
              shape.name, rows, heads, qkv_dim, split_ms, qk_ms,
              softmax_av_ms, pack_ms, phase_sum, total_ms);
}

void BenchAttentionSoftmaxAvGroupedProfile(rocblas_handle handle,
                                           const AttentionShape& shape,
                                           const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  if (qkv_dim != 72 || (rows != 256 && rows != 1024)) {
    std::printf("%-24s attention grouped softmax/AV skipped rows=%4d "
                "heads=%2d dim=%2d\n",
                shape.name, rows, heads, qkv_dim);
    return;
  }

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q(qkv_count);
  DeviceBuffer<float> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_scores_baseline(scores_count);
  DeviceBuffer<float> d_scores_grouped(scores_count);
  DeviceBuffer<float> d_att_baseline(qkv_count);
  DeviceBuffer<float> d_att_grouped(qkv_count);

  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const int qk_solution = RecommendedQkSolution(rows);
  const int av_solution = RecommendedAvSolution(rows);

  LaunchRecommendedSplitQKV(d_qkv.ptr, d_q.ptr, d_k.ptr, d_v.ptr, rows, heads,
                            qkv_dim);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  StridedBatchedGemmSpec qk_baseline;
  qk_baseline.label = "qk";
  qk_baseline.trans_a = rocblas_operation_transpose;
  qk_baseline.trans_b = rocblas_operation_none;
  qk_baseline.m = rows;
  qk_baseline.n = rows;
  qk_baseline.k = qkv_dim;
  qk_baseline.a = d_k.ptr;
  qk_baseline.lda = qkv_dim;
  qk_baseline.stride_a = stride_qkv;
  qk_baseline.b = d_q.ptr;
  qk_baseline.ldb = qkv_dim;
  qk_baseline.stride_b = stride_qkv;
  qk_baseline.d = d_scores_baseline.ptr;
  qk_baseline.ldd = rows;
  qk_baseline.stride_d = stride_scores;
  qk_baseline.batch_count = heads;
  qk_baseline.alpha = query_scale;
  qk_baseline.beta = 0.0f;

  StridedBatchedGemmSpec qk_grouped = qk_baseline;
  qk_grouped.d = d_scores_grouped.ptr;

  StridedBatchedGemmSpec av;
  av.label = "av";
  av.trans_a = rocblas_operation_none;
  av.trans_b = rocblas_operation_none;
  av.m = qkv_dim;
  av.n = rows;
  av.k = rows;
  av.a = d_v.ptr;
  av.lda = qkv_dim;
  av.stride_a = stride_qkv;
  av.b = d_scores_baseline.ptr;
  av.ldb = rows;
  av.stride_b = stride_scores;
  av.d = d_att_baseline.ptr;
  av.ldd = qkv_dim;
  av.stride_d = stride_qkv;
  av.batch_count = heads;
  av.alpha = 1.0f;
  av.beta = 0.0f;

  auto run_qk_baseline = [&]() {
    return RunStridedBatchedF32GemmSolution(handle, qk_baseline, qk_solution);
  };
  auto run_qk_grouped = [&]() {
    return RunStridedBatchedF32GemmSolution(handle, qk_grouped, qk_solution);
  };
  auto run_softmax = [&]() {
    LaunchRecommendedSoftmax(d_scores_baseline.ptr, rows, heads);
  };
  auto run_av = [&]() {
    return RunStridedBatchedF32GemmSolution(handle, av, av_solution);
  };
  auto run_baseline = [&]() {
    ROCBLAS_CHECK(run_qk_baseline());
    run_softmax();
    ROCBLAS_CHECK(run_av());
  };

  run_baseline();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  float qk_ms = 0.0f;
  float av_ms = 0.0f;
  if (!TimeRocblasStatusMedian(args, run_qk_baseline, qk_ms)) {
    std::printf("  qk cached rocBLAS solution=%d failed\n", qk_solution);
    return;
  }
  run_qk_baseline();
  run_softmax();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  if (!TimeRocblasStatusMedian(args, run_av, av_ms)) {
    std::printf("  av cached rocBLAS solution=%d failed\n", av_solution);
    return;
  }

  const float baseline_ms = TimeHipMedian(args, run_baseline);

  run_baseline();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> baseline_out(qkv_count);
  HIP_CHECK(hipMemcpy(baseline_out.data(), d_att_baseline.ptr,
                      static_cast<size_t>(qkv_count) * sizeof(float),
                      hipMemcpyDeviceToHost));

  std::printf("%-24s attention grouped softmax/AV sweep rows=%4d heads=%2d "
              "dim=%2d qk_solution=%7d av_solution=%7d qk=%8.3f "
              "av=%8.3f baseline_qk_softmax_av=%8.3f\n",
              shape.name, rows, heads, qkv_dim, qk_solution, av_solution,
              qk_ms, av_ms, baseline_ms);

  if (rows == 256) {
    ProfileAttentionSoftmaxAvGroupedCandidate<256, 72, 1>(
        args, shape, "g1", run_qk_grouped, d_scores_grouped.ptr, d_v.ptr,
        d_att_grouped.ptr, baseline_out, baseline_ms, qk_ms, av_ms);
    ProfileAttentionSoftmaxAvGroupedCandidate<256, 72, 2>(
        args, shape, "g2", run_qk_grouped, d_scores_grouped.ptr, d_v.ptr,
        d_att_grouped.ptr, baseline_out, baseline_ms, qk_ms, av_ms);
    ProfileAttentionSoftmaxAvGroupedCandidate<256, 72, 4>(
        args, shape, "g4", run_qk_grouped, d_scores_grouped.ptr, d_v.ptr,
        d_att_grouped.ptr, baseline_out, baseline_ms, qk_ms, av_ms);
    ProfileAttentionSoftmaxAvGroupedCandidate<256, 72, 8>(
        args, shape, "g8", run_qk_grouped, d_scores_grouped.ptr, d_v.ptr,
        d_att_grouped.ptr, baseline_out, baseline_ms, qk_ms, av_ms);
    ProfileAttentionSoftmaxAvGroupedCandidate<256, 72, 16>(
        args, shape, "g16", run_qk_grouped, d_scores_grouped.ptr, d_v.ptr,
        d_att_grouped.ptr, baseline_out, baseline_ms, qk_ms, av_ms);
  } else {
    ProfileAttentionSoftmaxAvGroupedCandidate<1024, 72, 1>(
        args, shape, "g1", run_qk_grouped, d_scores_grouped.ptr, d_v.ptr,
        d_att_grouped.ptr, baseline_out, baseline_ms, qk_ms, av_ms);
    ProfileAttentionSoftmaxAvGroupedCandidate<1024, 72, 2>(
        args, shape, "g2", run_qk_grouped, d_scores_grouped.ptr, d_v.ptr,
        d_att_grouped.ptr, baseline_out, baseline_ms, qk_ms, av_ms);
    ProfileAttentionSoftmaxAvGroupedCandidate<1024, 72, 4>(
        args, shape, "g4", run_qk_grouped, d_scores_grouped.ptr, d_v.ptr,
        d_att_grouped.ptr, baseline_out, baseline_ms, qk_ms, av_ms);
    ProfileAttentionSoftmaxAvGroupedCandidate<1024, 72, 8>(
        args, shape, "g8", run_qk_grouped, d_scores_grouped.ptr, d_v.ptr,
        d_att_grouped.ptr, baseline_out, baseline_ms, qk_ms, av_ms);
    ProfileAttentionSoftmaxAvGroupedCandidate<1024, 72, 16>(
        args, shape, "g16", run_qk_grouped, d_scores_grouped.ptr, d_v.ptr,
        d_att_grouped.ptr, baseline_out, baseline_ms, qk_ms, av_ms);
  }
}

void BenchAttentionOnlineHeadMajorProfile(rocblas_handle handle,
                                          const AttentionShape& shape,
                                          const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q(qkv_count);
  DeviceBuffer<float> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_scores(scores_count);
  DeviceBuffer<float> d_att_baseline(qkv_count);
  DeviceBuffer<float> d_att_online(qkv_count);

  const int threads = 256;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const int online_threads = rows <= 256 ? 64 : 256;
  const size_t online_shared =
      static_cast<size_t>(rows + online_threads) * sizeof(float);
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;
  const int qk_solution = RecommendedQkSolution(rows);
  const int av_solution = RecommendedAvSolution(rows);
  const rocblas_gemm_algo qk_algo =
      qk_solution == 0 ? rocblas_gemm_algo_standard
                       : rocblas_gemm_algo_solution_index;
  const rocblas_gemm_algo av_algo =
      av_solution == 0 ? rocblas_gemm_algo_standard
                       : rocblas_gemm_algo_solution_index;

  auto run_split = [&]() {
    LaunchRecommendedSplitQKV(d_qkv.ptr, d_q.ptr, d_k.ptr, d_v.ptr, rows,
                              heads, qkv_dim);
  };
  auto run_qk = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
        qkv_dim, &alpha_qk, d_k.ptr, rocblas_datatype_f32_r, qkv_dim,
        stride_qkv, d_q.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        &beta, d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, heads,
        rocblas_datatype_f32_r, qk_algo, qk_solution, /*flags=*/0));
  };
  auto run_softmax = [&]() { LaunchRecommendedSoftmax(d_scores.ptr, rows, heads); };
  auto run_av = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
        rows, &alpha_av, d_v.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, &beta,
        d_att_baseline.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        d_att_baseline.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv, heads,
        rocblas_datatype_f32_r, av_algo, av_solution, /*flags=*/0));
  };
  auto run_baseline_core = [&]() {
    run_qk();
    run_softmax();
    run_av();
  };
  auto run_online_core = [&]() {
    hipLaunchKernelGGL(AttentionOnlineHeadMajorKernel, dim3(rows, heads),
                       dim3(online_threads), online_shared, 0, d_q.ptr,
                       d_k.ptr, d_v.ptr, d_att_online.ptr, rows, qkv_dim,
                       query_scale);
  };
  auto run_baseline_total = [&]() {
    run_split();
    run_baseline_core();
  };
  auto run_online_total = [&]() {
    run_split();
    run_online_core();
  };

  run_split();
  run_baseline_core();
  run_online_core();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return MedianMs(ms);
  };

  const float split_ms = time_device(run_split);
  const float baseline_core_ms = time_device(run_baseline_core);
  const float online_core_ms = time_device(run_online_core);
  const float baseline_total_ms = time_device(run_baseline_total);
  const float online_total_ms = time_device(run_online_total);

  run_split();
  run_baseline_core();
  run_online_core();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> baseline_out(qkv_count);
  std::vector<float> online_out(qkv_count);
  HIP_CHECK(hipMemcpy(baseline_out.data(), d_att_baseline.ptr,
                      static_cast<size_t>(qkv_count) * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(online_out.data(), d_att_online.ptr,
                      static_cast<size_t>(qkv_count) * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_abs_error = 0.0f;
  double sum_sq = 0.0;
  for (size_t i = 0; i < baseline_out.size(); ++i) {
    const float diff = online_out[i] - baseline_out[i];
    max_abs_error = std::max(max_abs_error, std::fabs(diff));
    sum_sq += static_cast<double>(diff) * diff;
  }
  const float rms_error =
      static_cast<float>(std::sqrt(sum_sq / baseline_out.size()));

  std::printf("%-24s attention online head-major rows=%4d heads=%2d dim=%2d "
              "qk_solution=%7d av_solution=%7d split=%8.3f "
              "baseline_core=%8.3f online_core=%8.3f "
              "baseline_total=%8.3f online_total=%8.3f rel_core=%6.3f "
              "rel_total=%6.3f max_abs=%9.6f rms=%9.6f\n",
              shape.name, rows, heads, qkv_dim, qk_solution, av_solution,
              split_ms, baseline_core_ms, online_core_ms, baseline_total_ms,
              online_total_ms, online_core_ms / baseline_core_ms,
              online_total_ms / baseline_total_ms, max_abs_error, rms_error);
}

void BenchAttentionOnlineMultiQueryProfile(rocblas_handle handle,
                                           const AttentionShape& shape,
                                           const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q(qkv_count);
  DeviceBuffer<float> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_scores(scores_count);
  DeviceBuffer<float> d_att_baseline(qkv_count);
  DeviceBuffer<float> d_att_mq2(qkv_count);
  DeviceBuffer<float> d_att_mq4(qkv_count);
  DeviceBuffer<float> d_att_mq8(qkv_count);

  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;
  const int qk_solution = RecommendedQkSolution(rows);
  const int av_solution = RecommendedAvSolution(rows);
  const rocblas_gemm_algo qk_algo =
      qk_solution == 0 ? rocblas_gemm_algo_standard
                       : rocblas_gemm_algo_solution_index;
  const rocblas_gemm_algo av_algo =
      av_solution == 0 ? rocblas_gemm_algo_standard
                       : rocblas_gemm_algo_solution_index;

  auto run_split = [&]() {
    LaunchRecommendedSplitQKV(d_qkv.ptr, d_q.ptr, d_k.ptr, d_v.ptr, rows,
                              heads, qkv_dim);
  };
  auto run_qk = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
        qkv_dim, &alpha_qk, d_k.ptr, rocblas_datatype_f32_r, qkv_dim,
        stride_qkv, d_q.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        &beta, d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, heads,
        rocblas_datatype_f32_r, qk_algo, qk_solution, /*flags=*/0));
  };
  auto run_softmax = [&]() { LaunchRecommendedSoftmax(d_scores.ptr, rows, heads); };
  auto run_av = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
        rows, &alpha_av, d_v.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, &beta,
        d_att_baseline.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        d_att_baseline.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv, heads,
        rocblas_datatype_f32_r, av_algo, av_solution, /*flags=*/0));
  };
  auto run_baseline_core = [&]() {
    run_qk();
    run_softmax();
    run_av();
  };
  auto run_baseline_total = [&]() {
    run_split();
    run_baseline_core();
  };

  auto run_mq2_core = [&]() {
    constexpr int kQueryRows = 2;
    const dim3 grid((rows + kQueryRows - 1) / kQueryRows, heads);
    const size_t shared =
        static_cast<size_t>(kQueryRows) * (rows + 256) * sizeof(float);
    hipLaunchKernelGGL((AttentionOnlineMultiQueryKernel<kQueryRows>), grid,
                       dim3(256), shared, 0, d_q.ptr, d_k.ptr, d_v.ptr,
                       d_att_mq2.ptr, rows, qkv_dim, query_scale);
  };
  auto run_mq4_core = [&]() {
    constexpr int kQueryRows = 4;
    const dim3 grid((rows + kQueryRows - 1) / kQueryRows, heads);
    const size_t shared =
        static_cast<size_t>(kQueryRows) * (rows + 256) * sizeof(float);
    hipLaunchKernelGGL((AttentionOnlineMultiQueryKernel<kQueryRows>), grid,
                       dim3(256), shared, 0, d_q.ptr, d_k.ptr, d_v.ptr,
                       d_att_mq4.ptr, rows, qkv_dim, query_scale);
  };
  auto run_mq8_core = [&]() {
    constexpr int kQueryRows = 8;
    const dim3 grid((rows + kQueryRows - 1) / kQueryRows, heads);
    const size_t shared =
        static_cast<size_t>(kQueryRows) * (rows + 256) * sizeof(float);
    hipLaunchKernelGGL((AttentionOnlineMultiQueryKernel<kQueryRows>), grid,
                       dim3(256), shared, 0, d_q.ptr, d_k.ptr, d_v.ptr,
                       d_att_mq8.ptr, rows, qkv_dim, query_scale);
  };
  auto run_mq2_total = [&]() {
    run_split();
    run_mq2_core();
  };
  auto run_mq4_total = [&]() {
    run_split();
    run_mq4_core();
  };
  auto run_mq8_total = [&]() {
    run_split();
    run_mq8_core();
  };

  run_split();
  run_baseline_core();
  run_mq2_core();
  run_mq4_core();
  run_mq8_core();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return MedianMs(ms);
  };

  const float split_ms = time_device(run_split);
  const float baseline_core_ms = time_device(run_baseline_core);
  const float mq2_core_ms = time_device(run_mq2_core);
  const float mq4_core_ms = time_device(run_mq4_core);
  const float mq8_core_ms = time_device(run_mq8_core);
  const float baseline_total_ms = time_device(run_baseline_total);
  const float mq2_total_ms = time_device(run_mq2_total);
  const float mq4_total_ms = time_device(run_mq4_total);
  const float mq8_total_ms = time_device(run_mq8_total);

  run_split();
  run_baseline_core();
  run_mq2_core();
  run_mq4_core();
  run_mq8_core();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> baseline_out(qkv_count);
  std::vector<float> mq2_out(qkv_count);
  std::vector<float> mq4_out(qkv_count);
  std::vector<float> mq8_out(qkv_count);
  HIP_CHECK(hipMemcpy(baseline_out.data(), d_att_baseline.ptr,
                      static_cast<size_t>(qkv_count) * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(mq2_out.data(), d_att_mq2.ptr,
                      static_cast<size_t>(qkv_count) * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(mq4_out.data(), d_att_mq4.ptr,
                      static_cast<size_t>(qkv_count) * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(mq8_out.data(), d_att_mq8.ptr,
                      static_cast<size_t>(qkv_count) * sizeof(float),
                      hipMemcpyDeviceToHost));

  auto error_stats = [&](const std::vector<float>& candidate) {
    float max_abs_error = 0.0f;
    double sum_sq = 0.0;
    for (size_t i = 0; i < baseline_out.size(); ++i) {
      const float diff = candidate[i] - baseline_out[i];
      max_abs_error = std::max(max_abs_error, std::fabs(diff));
      sum_sq += static_cast<double>(diff) * diff;
    }
    return std::pair<float, float>(
        max_abs_error,
        static_cast<float>(std::sqrt(sum_sq / baseline_out.size())));
  };

  auto print_candidate = [&](const char* label, float core_ms, float total_ms,
                             const std::vector<float>& output) {
    const auto [max_abs_error, rms_error] = error_stats(output);
    std::printf("%-24s attention online multiquery %-3s rows=%4d heads=%2d "
                "dim=%2d qk_solution=%7d av_solution=%7d split=%8.3f "
                "baseline_core=%8.3f online_core=%8.3f "
                "baseline_total=%8.3f online_total=%8.3f rel_core=%6.3f "
                "rel_total=%6.3f max_abs=%9.6f rms=%9.6f\n",
                shape.name, label, rows, heads, qkv_dim, qk_solution,
                av_solution, split_ms, baseline_core_ms, core_ms,
                baseline_total_ms, total_ms, core_ms / baseline_core_ms,
                total_ms / baseline_total_ms, max_abs_error, rms_error);
  };

  print_candidate("mq2", mq2_core_ms, mq2_total_ms, mq2_out);
  print_candidate("mq4", mq4_core_ms, mq4_total_ms, mq4_out);
  print_candidate("mq8", mq8_core_ms, mq8_total_ms, mq8_out);
}

void BenchAttentionSoftmaxThreadSweep(rocblas_handle handle,
                                      const AttentionShape& shape,
                                      const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q(qkv_count);
  DeviceBuffer<float> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_scores(scores_count);

  const int qkv_threads = 256;
  const int qkv_blocks = (qkv_count + qkv_threads - 1) / qkv_threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float beta = 0.0f;

  hipLaunchKernelGGL(SplitQKVForAttentionKernel, dim3(qkv_blocks),
                     dim3(qkv_threads), 0, 0, d_qkv.ptr, d_q.ptr, d_k.ptr,
                     d_v.ptr, rows, heads, qkv_dim);
  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
      qkv_dim, &alpha_qk, d_k.ptr, rocblas_datatype_f32_r, qkv_dim,
      stride_qkv, d_q.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv, &beta,
      d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, d_scores.ptr,
      rocblas_datatype_f32_r, rows, stride_scores, heads,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
      /*solution_index=*/0, /*flags=*/0));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  auto time_softmax = [&](int threads) {
    const size_t softmax_shared =
        static_cast<size_t>(threads) * sizeof(float);
    auto run = [&]() {
      hipLaunchKernelGGL(SoftmaxRowsKernel, dim3(rows, heads), dim3(threads),
                         softmax_shared, 0, d_scores.ptr, rows);
    };
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) run();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) run();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return MedianMs(ms);
  };

  const int thread_options[] = {64, 128, 256, 512, 1024};
  std::printf("%-24s attention softmax thread sweep rows=%4d heads=%2d "
              "dim=%2d",
              shape.name, rows, heads, qkv_dim);
  for (int threads : thread_options) {
    const float ms = time_softmax(threads);
    std::printf(" t%d=%8.3f", threads, ms);
  }
  std::printf(" ms\n");
}

void BenchAttentionSoftmaxFixedProfile(rocblas_handle handle,
                                       const AttentionShape& shape,
                                       const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q(qkv_count);
  DeviceBuffer<float> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_seed(scores_count);
  DeviceBuffer<float> d_dynamic(scores_count);
  DeviceBuffer<float> d_fixed(scores_count);
  DeviceBuffer<float> d_float4(scores_count);

  const int qkv_threads = 256;
  const int qkv_blocks = (qkv_count + qkv_threads - 1) / qkv_threads;
  const int dynamic_threads = rows <= 256 ? 64 : 256;
  const size_t dynamic_shared =
      static_cast<size_t>(dynamic_threads) * sizeof(float);
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float beta = 0.0f;

  hipLaunchKernelGGL(SplitQKVForAttentionKernel, dim3(qkv_blocks),
                     dim3(qkv_threads), 0, 0, d_qkv.ptr, d_q.ptr, d_k.ptr,
                     d_v.ptr, rows, heads, qkv_dim);
  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
      qkv_dim, &alpha_qk, d_k.ptr, rocblas_datatype_f32_r, qkv_dim,
      stride_qkv, d_q.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv, &beta,
      d_seed.ptr, rocblas_datatype_f32_r, rows, stride_scores, d_seed.ptr,
      rocblas_datatype_f32_r, rows, stride_scores, heads,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
      /*solution_index=*/0, /*flags=*/0));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  auto reset_scores = [&](float* dst) {
    HIP_CHECK(hipMemcpy(dst, d_seed.ptr,
                        static_cast<size_t>(scores_count) * sizeof(float),
                        hipMemcpyDeviceToDevice));
  };
  auto run_dynamic = [&]() {
    hipLaunchKernelGGL(SoftmaxRowsKernel, dim3(rows, heads),
                       dim3(dynamic_threads), dynamic_shared, 0,
                       d_dynamic.ptr, rows);
  };
  auto run_fixed_256_64 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedKernel<256, 64>), dim3(rows, heads),
                       dim3(64), 0, 0, d_fixed.ptr);
  };
  auto run_float4_256_64 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedFloat4Kernel<256, 64>),
                       dim3(rows, heads), dim3(64), 0, 0, d_float4.ptr);
  };
  auto run_recompute_256_64 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedFloat4RecomputeKernel<256, 64>),
                       dim3(rows, heads), dim3(64), 0, 0, d_float4.ptr);
  };
  auto run_recompute_256_128 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedFloat4RecomputeKernel<256, 128>),
                       dim3(rows, heads), dim3(128), 0, 0, d_float4.ptr);
  };
  auto run_recompute_256_256 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedFloat4RecomputeKernel<256, 256>),
                       dim3(rows, heads), dim3(256), 0, 0, d_float4.ptr);
  };
  auto run_fixed_256_128 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedKernel<256, 128>), dim3(rows, heads),
                       dim3(128), 0, 0, d_fixed.ptr);
  };
  auto run_fixed_256_256 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedKernel<256, 256>), dim3(rows, heads),
                       dim3(256), 0, 0, d_fixed.ptr);
  };
  auto run_fixed_1024_64 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedKernel<1024, 64>), dim3(rows, heads),
                       dim3(64), 0, 0, d_fixed.ptr);
  };
  auto run_fixed_1024_128 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedKernel<1024, 128>), dim3(rows, heads),
                       dim3(128), 0, 0, d_fixed.ptr);
  };
  auto run_float4_1024_128 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedFloat4Kernel<1024, 128>),
                       dim3(rows, heads), dim3(128), 0, 0, d_float4.ptr);
  };
  auto run_recompute_1024_128 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedFloat4RecomputeKernel<1024, 128>),
                       dim3(rows, heads), dim3(128), 0, 0, d_float4.ptr);
  };
  auto run_recompute_1024_64 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedFloat4RecomputeKernel<1024, 64>),
                       dim3(rows, heads), dim3(64), 0, 0, d_float4.ptr);
  };
  auto run_recompute_1024_256 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedFloat4RecomputeKernel<1024, 256>),
                       dim3(rows, heads), dim3(256), 0, 0, d_float4.ptr);
  };
  auto run_recompute_1024_512 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedFloat4RecomputeKernel<1024, 512>),
                       dim3(rows, heads), dim3(512), 0, 0, d_float4.ptr);
  };
  auto run_fixed_1024_256 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedKernel<1024, 256>), dim3(rows, heads),
                       dim3(256), 0, 0, d_fixed.ptr);
  };
  auto run_fixed_1024_512 = [&]() {
    hipLaunchKernelGGL((SoftmaxRowsFixedKernel<1024, 512>), dim3(rows, heads),
                       dim3(512), 0, 0, d_fixed.ptr);
  };

  auto time_device = [&](const auto& prepare, const auto& fn) {
    prepare();
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    prepare();
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return MedianMs(ms);
  };

  auto compare_candidate = [&](float* candidate_ptr, const auto& run_candidate) {
    reset_scores(d_dynamic.ptr);
    reset_scores(candidate_ptr);
    run_dynamic();
    run_candidate();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> dynamic(scores_count);
    std::vector<float> candidate(scores_count);
    HIP_CHECK(hipMemcpy(dynamic.data(), d_dynamic.ptr,
                        static_cast<size_t>(scores_count) * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(candidate.data(), candidate_ptr,
                        static_cast<size_t>(scores_count) * sizeof(float),
                        hipMemcpyDeviceToHost));
    float max_abs_error = 0.0f;
    double sum_sq = 0.0;
    for (size_t i = 0; i < dynamic.size(); ++i) {
      const float diff = candidate[i] - dynamic[i];
      max_abs_error = std::max(max_abs_error, std::fabs(diff));
      sum_sq += static_cast<double>(diff) * diff;
    }
    const float rms_error =
        static_cast<float>(std::sqrt(sum_sq / dynamic.size()));
    std::printf(" max_abs=%9.6f rms=%9.6f", max_abs_error, rms_error);
  };

  const float dynamic_ms = time_device([&]() { reset_scores(d_dynamic.ptr); },
                                       run_dynamic);
  std::printf("%-24s attention softmax fixed profile rows=%4d heads=%2d "
              "dim=%2d dynamic_t%d=%8.3f",
              shape.name, rows, heads, qkv_dim, dynamic_threads, dynamic_ms);
  if (rows == 256) {
    const float fixed64_ms =
        time_device([&]() { reset_scores(d_fixed.ptr); }, run_fixed_256_64);
    const float fixed128_ms =
        time_device([&]() { reset_scores(d_fixed.ptr); }, run_fixed_256_128);
    const float fixed256_ms =
        time_device([&]() { reset_scores(d_fixed.ptr); }, run_fixed_256_256);
    const float float4_ms =
        time_device([&]() { reset_scores(d_float4.ptr); }, run_float4_256_64);
    const float recompute_ms = time_device(
        [&]() { reset_scores(d_float4.ptr); }, run_recompute_256_64);
    const float recompute128_ms = time_device(
        [&]() { reset_scores(d_float4.ptr); }, run_recompute_256_128);
    const float recompute256_ms = time_device(
        [&]() { reset_scores(d_float4.ptr); }, run_recompute_256_256);
    std::printf(" fixed64=%8.3f fixed128=%8.3f fixed256=%8.3f "
                "float4_64=%8.3f recompute_64=%8.3f "
                "recompute_128=%8.3f recompute_256=%8.3f",
                fixed64_ms, fixed128_ms, fixed256_ms, float4_ms,
                recompute_ms, recompute128_ms, recompute256_ms);
    compare_candidate(d_float4.ptr, run_float4_256_64);
    compare_candidate(d_float4.ptr, run_recompute_256_64);
  } else if (rows == 1024) {
    const float fixed64_ms =
        time_device([&]() { reset_scores(d_fixed.ptr); }, run_fixed_1024_64);
    const float fixed128_ms =
        time_device([&]() { reset_scores(d_fixed.ptr); }, run_fixed_1024_128);
    const float fixed256_ms =
        time_device([&]() { reset_scores(d_fixed.ptr); }, run_fixed_1024_256);
    const float fixed512_ms =
        time_device([&]() { reset_scores(d_fixed.ptr); }, run_fixed_1024_512);
    const float float4_ms =
        time_device([&]() { reset_scores(d_float4.ptr); },
                    run_float4_1024_128);
    const float recompute_ms = time_device(
        [&]() { reset_scores(d_float4.ptr); }, run_recompute_1024_128);
    const float recompute64_ms = time_device(
        [&]() { reset_scores(d_float4.ptr); }, run_recompute_1024_64);
    const float recompute256_ms = time_device(
        [&]() { reset_scores(d_float4.ptr); }, run_recompute_1024_256);
    const float recompute512_ms = time_device(
        [&]() { reset_scores(d_float4.ptr); }, run_recompute_1024_512);
    std::printf(" fixed64=%8.3f fixed128=%8.3f fixed256=%8.3f "
                "fixed512=%8.3f float4_128=%8.3f recompute_128=%8.3f",
                fixed64_ms, fixed128_ms, fixed256_ms, fixed512_ms,
                float4_ms, recompute_ms);
    std::printf(" recompute_64=%8.3f recompute_256=%8.3f "
                "recompute_512=%8.3f",
                recompute64_ms, recompute256_ms, recompute512_ms);
    compare_candidate(d_float4.ptr, run_float4_1024_128);
    compare_candidate(d_float4.ptr, run_recompute_1024_128);
  }
  std::printf(" ms\n");
}

void BenchAttentionSplitVectorizedProfile(const AttentionShape& shape,
                                          const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scalar_threads = 256;
  const int vector_threads = 256;
  const int scalar_blocks = (qkv_count + scalar_threads - 1) / scalar_threads;
  const int vector_count = rows * heads * (qkv_dim / 4);
  const int vector_blocks = (vector_count + vector_threads - 1) / vector_threads;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q_scalar(qkv_count);
  DeviceBuffer<float> d_k_scalar(qkv_count);
  DeviceBuffer<float> d_v_scalar(qkv_count);
  DeviceBuffer<float> d_q_vector(qkv_count);
  DeviceBuffer<float> d_k_vector(qkv_count);
  DeviceBuffer<float> d_v_vector(qkv_count);
  HIP_CHECK(hipMemset(d_qkv.ptr, 1,
                      static_cast<size_t>(qkv_projection_count) *
                          sizeof(float)));

  auto run_scalar = [&]() {
    hipLaunchKernelGGL(SplitQKVForAttentionKernel, dim3(scalar_blocks),
                       dim3(scalar_threads), 0, 0, d_qkv.ptr, d_q_scalar.ptr,
                       d_k_scalar.ptr, d_v_scalar.ptr, rows, heads, qkv_dim);
  };
  auto run_vector = [&]() {
    hipLaunchKernelGGL(SplitQKVForAttentionFloat4Kernel, dim3(vector_blocks),
                       dim3(vector_threads), 0, 0, d_qkv.ptr, d_q_vector.ptr,
                       d_k_vector.ptr, d_v_vector.ptr, rows, heads, qkv_dim);
  };

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return MedianMs(ms);
  };

  const float scalar_ms = time_device(run_scalar);
  const float vector_ms = time_device(run_vector);

  run_scalar();
  run_vector();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> q_scalar(qkv_count);
  std::vector<float> k_scalar(qkv_count);
  std::vector<float> v_scalar(qkv_count);
  std::vector<float> q_vector(qkv_count);
  std::vector<float> k_vector(qkv_count);
  std::vector<float> v_vector(qkv_count);
  HIP_CHECK(hipMemcpy(q_scalar.data(), d_q_scalar.ptr,
                      qkv_count * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(k_scalar.data(), d_k_scalar.ptr,
                      qkv_count * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(v_scalar.data(), d_v_scalar.ptr,
                      qkv_count * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(q_vector.data(), d_q_vector.ptr,
                      qkv_count * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(k_vector.data(), d_k_vector.ptr,
                      qkv_count * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(v_vector.data(), d_v_vector.ptr,
                      qkv_count * sizeof(float), hipMemcpyDeviceToHost));

  float max_abs_error = 0.0f;
  double sum_sq = 0.0;
  auto accumulate_error = [&](const std::vector<float>& vector,
                              const std::vector<float>& scalar) {
    for (size_t i = 0; i < scalar.size(); ++i) {
      const float diff = vector[i] - scalar[i];
      max_abs_error = std::max(max_abs_error, std::fabs(diff));
      sum_sq += static_cast<double>(diff) * diff;
    }
  };
  accumulate_error(q_vector, q_scalar);
  accumulate_error(k_vector, k_scalar);
  accumulate_error(v_vector, v_scalar);
  const float rms_error =
      static_cast<float>(std::sqrt(sum_sq / (3.0 * qkv_count)));

  std::printf("%-24s attention split vectorized rows=%4d heads=%2d dim=%2d "
              "scalar=%8.3f vector4=%8.3f rel=%6.3f max_abs=%9.6f "
              "rms=%9.6f\n",
              shape.name, rows, heads, qkv_dim, scalar_ms, vector_ms,
              vector_ms / scalar_ms, max_abs_error, rms_error);
}

void BenchAttentionPackVectorizedProfile(const AttentionShape& shape,
                                         const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int scalar_threads = 256;
  const int vector_threads = 256;
  const int scalar_blocks = (qkv_count + scalar_threads - 1) / scalar_threads;
  const int vector_count = rows * heads * (qkv_dim / 4);
  const int vector_blocks =
      (vector_count + vector_threads - 1) / vector_threads;

  DeviceBuffer<float> d_head_major(qkv_count);
  DeviceBuffer<rocblas_bfloat16> d_scalar(qkv_count);
  DeviceBuffer<rocblas_bfloat16> d_vector(qkv_count);
  HIP_CHECK(hipMemset(d_head_major.ptr, 1,
                      static_cast<size_t>(qkv_count) * sizeof(float)));

  auto run_scalar = [&]() {
    hipLaunchKernelGGL(PackAttentionHeadsToBF16Kernel, dim3(scalar_blocks),
                       dim3(scalar_threads), 0, 0, d_head_major.ptr,
                       d_scalar.ptr, rows, heads, qkv_dim);
  };
  auto run_vector = [&]() {
    hipLaunchKernelGGL(PackAttentionHeadsToBF16Float4Kernel,
                       dim3(vector_blocks), dim3(vector_threads), 0, 0,
                       d_head_major.ptr, d_vector.ptr, rows, heads, qkv_dim);
  };

  if ((qkv_dim % 4) != 0) {
    std::printf("%-24s attention pack BF16 vectorized rows=%4d heads=%2d "
                "dim=%2d skipped: dim is not divisible by 4\n",
                shape.name, rows, heads, qkv_dim);
    return;
  }

  const float scalar_ms = TimeHipMedian(args, run_scalar);
  const float vector_ms = TimeHipMedian(args, run_vector);

  run_scalar();
  run_vector();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<rocblas_bfloat16> scalar(qkv_count);
  std::vector<rocblas_bfloat16> vector(qkv_count);
  HIP_CHECK(hipMemcpy(scalar.data(), d_scalar.ptr,
                      qkv_count * sizeof(rocblas_bfloat16),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(vector.data(), d_vector.ptr,
                      qkv_count * sizeof(rocblas_bfloat16),
                      hipMemcpyDeviceToHost));

  float max_abs_error = 0.0f;
  double sum_sq = 0.0;
  for (size_t i = 0; i < scalar.size(); ++i) {
    const float diff =
        static_cast<float>(vector[i]) - static_cast<float>(scalar[i]);
    max_abs_error = std::max(max_abs_error, std::fabs(diff));
    sum_sq += static_cast<double>(diff) * diff;
  }
  const float rms_error =
      static_cast<float>(std::sqrt(sum_sq / scalar.size()));

  std::printf("%-24s attention pack BF16 vectorized rows=%4d heads=%2d "
              "dim=%2d scalar=%8.3f vector4=%8.3f rel=%6.3f "
              "max_abs=%9.6f rms=%9.6f\n",
              shape.name, rows, heads, qkv_dim, scalar_ms, vector_ms,
              vector_ms / scalar_ms, max_abs_error, rms_error);
}

void BenchAttentionRocblasSolutionSweep(rocblas_handle handle,
                                        const AttentionShape& shape,
                                        const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q(qkv_count);
  DeviceBuffer<float> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_scores(scores_count);
  DeviceBuffer<float> d_att_head(qkv_count);

  const int qkv_threads = 256;
  const int softmax_threads = rows <= 256 ? 64 : 256;
  const int qkv_blocks = (qkv_count + qkv_threads - 1) / qkv_threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;

  hipLaunchKernelGGL(SplitQKVForAttentionKernel, dim3(qkv_blocks),
                     dim3(qkv_threads), 0, 0, d_qkv.ptr, d_q.ptr, d_k.ptr,
                     d_v.ptr, rows, heads, qkv_dim);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::printf("%-24s attention rocBLAS solution sweep rows=%4d heads=%2d "
              "dim=%2d max_candidates=%d\n",
              shape.name, rows, heads, qkv_dim,
              args.attention_rocblas_solution_sweep);

  StridedBatchedGemmSpec qk;
  qk.label = "qk";
  qk.trans_a = rocblas_operation_transpose;
  qk.trans_b = rocblas_operation_none;
  qk.m = rows;
  qk.n = rows;
  qk.k = qkv_dim;
  qk.a = d_k.ptr;
  qk.lda = qkv_dim;
  qk.stride_a = stride_qkv;
  qk.b = d_q.ptr;
  qk.ldb = qkv_dim;
  qk.stride_b = stride_qkv;
  qk.d = d_scores.ptr;
  qk.ldd = rows;
  qk.stride_d = stride_scores;
  qk.batch_count = heads;
  qk.alpha = query_scale;
  qk.beta = 0.0f;

  (void)BenchOneAttentionRocblasSolutionSweep(
      handle, args, args.attention_rocblas_solution_sweep, qk);

  // Recreate the normalized F32 score matrix before sweeping AV. The QK sweep
  // above writes raw scores with multiple solution candidates, while AV should
  // be measured against the same post-softmax input used by the backend path.
  ROCBLAS_CHECK(RunStridedBatchedF32GemmSolution(handle, qk, 0));
  const size_t softmax_shared =
      static_cast<size_t>(softmax_threads) * sizeof(float);
  hipLaunchKernelGGL(SoftmaxRowsKernel, dim3(rows, heads),
                     dim3(softmax_threads), softmax_shared, 0, d_scores.ptr,
                     rows);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  StridedBatchedGemmSpec av;
  av.label = "av";
  av.trans_a = rocblas_operation_none;
  av.trans_b = rocblas_operation_none;
  av.m = qkv_dim;
  av.n = rows;
  av.k = rows;
  av.a = d_v.ptr;
  av.lda = qkv_dim;
  av.stride_a = stride_qkv;
  av.b = d_scores.ptr;
  av.ldb = rows;
  av.stride_b = stride_scores;
  av.d = d_att_head.ptr;
  av.ldd = qkv_dim;
  av.stride_d = stride_qkv;
  av.batch_count = heads;
  av.alpha = 1.0f;
  av.beta = 0.0f;

  (void)BenchOneAttentionRocblasSolutionSweep(
      handle, args, args.attention_rocblas_solution_sweep, av);
}

void BenchAttentionQkTiledProfile(rocblas_handle handle,
                                  const AttentionShape& shape,
                                  const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q(qkv_count);
  DeviceBuffer<float> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_scores_rocblas(scores_count);
  DeviceBuffer<float> d_scores_tiled(scores_count);

  const int qkv_threads = 256;
  const int qkv_blocks = (qkv_count + qkv_threads - 1) / qkv_threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;

  hipLaunchKernelGGL(SplitQKVForAttentionKernel, dim3(qkv_blocks),
                     dim3(qkv_threads), 0, 0, d_qkv.ptr, d_q.ptr, d_k.ptr,
                     d_v.ptr, rows, heads, qkv_dim);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  StridedBatchedGemmSpec qk;
  qk.label = "qk";
  qk.trans_a = rocblas_operation_transpose;
  qk.trans_b = rocblas_operation_none;
  qk.m = rows;
  qk.n = rows;
  qk.k = qkv_dim;
  qk.a = d_k.ptr;
  qk.lda = qkv_dim;
  qk.stride_a = stride_qkv;
  qk.b = d_q.ptr;
  qk.ldb = qkv_dim;
  qk.stride_b = stride_qkv;
  qk.d = d_scores_rocblas.ptr;
  qk.ldd = rows;
  qk.stride_d = stride_scores;
  qk.batch_count = heads;
  qk.alpha = query_scale;
  qk.beta = 0.0f;

  int rocblas_solution = RecommendedQkSolution(rows);
  std::printf("%-24s attention QK tiled profile rows=%4d heads=%2d dim=%2d\n",
              shape.name, rows, heads, qkv_dim);
  if (args.attention_rocblas_solution_sweep > 0) {
    rocblas_solution = BenchOneAttentionRocblasSolutionSweep(
        handle, args, args.attention_rocblas_solution_sweep, qk);
  } else {
    std::printf("  qk cached_solution=%7d samples=%d warmup=%d iters=%d\n",
                rocblas_solution, args.samples, args.warmup, args.iters);
  }

  auto run_rocblas_selected = [&]() {
    return RunStridedBatchedF32GemmSolution(handle, qk, rocblas_solution);
  };
  auto run_tiled = [&]() {
    constexpr int kBM = 8;
    constexpr int kBN = 16;
    constexpr int kBK = 32;
    const dim3 block(kBN, kBM);
    const dim3 grid((rows + kBN - 1) / kBN, (rows + kBM - 1) / kBM,
                    heads);
    hipLaunchKernelGGL((AttentionQkTiledKernel<kBM, kBN, kBK>), grid, block,
                       0, 0, d_q.ptr, d_k.ptr, d_scores_tiled.ptr, rows,
                       qkv_dim, query_scale);
  };

  float rocblas_ms = 0.0f;
  if (!TimeRocblasStatusMedian(args, run_rocblas_selected, rocblas_ms)) {
    std::printf("  qk selected rocBLAS solution=%d failed\n",
                rocblas_solution);
    return;
  }

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return MedianMs(ms);
  };

  const float tiled_ms = time_device(run_tiled);

  ROCBLAS_CHECK(run_rocblas_selected());
  run_tiled();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> rocblas_out(scores_count);
  std::vector<float> tiled_out(scores_count);
  HIP_CHECK(hipMemcpy(rocblas_out.data(), d_scores_rocblas.ptr,
                      static_cast<size_t>(scores_count) * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tiled_out.data(), d_scores_tiled.ptr,
                      static_cast<size_t>(scores_count) * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_abs_error = 0.0f;
  double sum_sq = 0.0;
  for (size_t i = 0; i < rocblas_out.size(); ++i) {
    const float diff = tiled_out[i] - rocblas_out[i];
    max_abs_error = std::max(max_abs_error, std::fabs(diff));
    sum_sq += static_cast<double>(diff) * diff;
  }
  const float rms_error =
      static_cast<float>(std::sqrt(sum_sq / rocblas_out.size()));
  const double flops = 2.0 * static_cast<double>(heads) * rows * rows * qkv_dim;
  std::printf("  qk_tiled8x16x32 rocblas_solution=%7d rocblas=%8.3f ms "
              "tiled=%8.3f ms rel=%6.3f rocblas_gflops=%8.1f "
              "tiled_gflops=%8.1f max_abs=%9.6f rms=%9.6f\n",
              rocblas_solution, rocblas_ms, tiled_ms, tiled_ms / rocblas_ms,
              flops / (rocblas_ms * 1.0E6), flops / (tiled_ms * 1.0E6),
              max_abs_error, rms_error);
}

void BenchAttentionAvTiledProfile(rocblas_handle handle,
                                  const AttentionShape& shape,
                                  const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q(qkv_count);
  DeviceBuffer<float> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_scores(scores_count);
  DeviceBuffer<float> d_att_rocblas(qkv_count);
  DeviceBuffer<float> d_att_tiled(qkv_count);

  const int qkv_threads = 256;
  const int softmax_threads = rows <= 256 ? 64 : 256;
  const int qkv_blocks = (qkv_count + qkv_threads - 1) / qkv_threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float beta = 0.0f;

  auto run_prepare_raw_scores = [&]() {
    hipLaunchKernelGGL(SplitQKVForAttentionKernel, dim3(qkv_blocks),
                       dim3(qkv_threads), 0, 0, d_qkv.ptr, d_q.ptr, d_k.ptr,
                       d_v.ptr, rows, heads, qkv_dim);
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
        qkv_dim, &alpha_qk, d_k.ptr, rocblas_datatype_f32_r, qkv_dim,
        stride_qkv, d_q.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        &beta, d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, heads,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
    const size_t softmax_shared =
        static_cast<size_t>(softmax_threads) * sizeof(float);
    hipLaunchKernelGGL(SoftmaxRowsKernel, dim3(rows, heads),
                       dim3(softmax_threads), softmax_shared, 0, d_scores.ptr,
                       rows);
  };

  run_prepare_raw_scores();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  StridedBatchedGemmSpec av;
  av.label = "av";
  av.trans_a = rocblas_operation_none;
  av.trans_b = rocblas_operation_none;
  av.m = qkv_dim;
  av.n = rows;
  av.k = rows;
  av.a = d_v.ptr;
  av.lda = qkv_dim;
  av.stride_a = stride_qkv;
  av.b = d_scores.ptr;
  av.ldb = rows;
  av.stride_b = stride_scores;
  av.d = d_att_rocblas.ptr;
  av.ldd = qkv_dim;
  av.stride_d = stride_qkv;
  av.batch_count = heads;
  av.alpha = 1.0f;
  av.beta = 0.0f;

  const int max_solutions =
      args.attention_rocblas_solution_sweep > 0
          ? args.attention_rocblas_solution_sweep
          : 8;
  std::printf("%-24s attention AV tiled profile rows=%4d heads=%2d dim=%2d\n",
              shape.name, rows, heads, qkv_dim);
  const int rocblas_solution =
      BenchOneAttentionRocblasSolutionSweep(handle, args, max_solutions, av);

  auto run_rocblas_selected = [&]() {
    return RunStridedBatchedF32GemmSolution(handle, av, rocblas_solution);
  };
  auto run_tiled = [&]() {
    constexpr int kBM = 8;
    constexpr int kBN = 16;
    constexpr int kBK = 32;
    const dim3 block(kBN, kBM);
    const dim3 grid((qkv_dim + kBN - 1) / kBN, (rows + kBM - 1) / kBM,
                    heads);
    hipLaunchKernelGGL((AttentionAvTiledKernel<kBM, kBN, kBK>), grid, block,
                       0, 0, d_scores.ptr, d_v.ptr, d_att_tiled.ptr, rows,
                       qkv_dim);
  };

  float rocblas_ms = 0.0f;
  if (!TimeRocblasStatusMedian(args, run_rocblas_selected, rocblas_ms)) {
    std::printf("  av selected rocBLAS solution=%d failed\n",
                rocblas_solution);
    return;
  }

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return MedianMs(ms);
  };

  const float tiled_ms = time_device(run_tiled);

  ROCBLAS_CHECK(run_rocblas_selected());
  run_tiled();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> rocblas_out(qkv_count);
  std::vector<float> tiled_out(qkv_count);
  HIP_CHECK(hipMemcpy(rocblas_out.data(), d_att_rocblas.ptr,
                      qkv_count * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tiled_out.data(), d_att_tiled.ptr,
                      qkv_count * sizeof(float), hipMemcpyDeviceToHost));

  float max_abs_error = 0.0f;
  double sum_sq = 0.0;
  for (size_t i = 0; i < rocblas_out.size(); ++i) {
    const float diff = tiled_out[i] - rocblas_out[i];
    max_abs_error = std::max(max_abs_error, std::fabs(diff));
    sum_sq += static_cast<double>(diff) * diff;
  }
  const float rms_error =
      static_cast<float>(std::sqrt(sum_sq / rocblas_out.size()));
  const double flops = 2.0 * static_cast<double>(heads) * rows * rows * qkv_dim;
  std::printf("  av_tiled8x16x32 rocblas_solution=%7d rocblas=%8.3f ms "
              "tiled=%8.3f ms rel=%6.3f rocblas_gflops=%8.1f "
              "tiled_gflops=%8.1f max_abs=%9.6f rms=%9.6f\n",
              rocblas_solution, rocblas_ms, tiled_ms, tiled_ms / rocblas_ms,
              flops / (rocblas_ms * 1.0E6), flops / (tiled_ms * 1.0E6),
              max_abs_error, rms_error);

  if (qkv_dim == 72) {
    ProfileAttentionAvTiledDim4Candidate<4, 32, 18>(
        args, shape, "bm4bk32", d_scores.ptr, d_v.ptr, d_att_tiled.ptr,
        rocblas_out, rocblas_ms);
    ProfileAttentionAvTiledDim4Candidate<8, 32, 18>(
        args, shape, "bm8bk32", d_scores.ptr, d_v.ptr, d_att_tiled.ptr,
        rocblas_out, rocblas_ms);
    ProfileAttentionAvTiledDim4Candidate<16, 32, 18>(
        args, shape, "bm16bk32", d_scores.ptr, d_v.ptr, d_att_tiled.ptr,
        rocblas_out, rocblas_ms);
    ProfileAttentionAvTiledDim4Candidate<24, 32, 18>(
        args, shape, "bm24bk32", d_scores.ptr, d_v.ptr, d_att_tiled.ptr,
        rocblas_out, rocblas_ms);
    ProfileAttentionAvTiledDim4Candidate<32, 32, 18>(
        args, shape, "bm32bk32", d_scores.ptr, d_v.ptr, d_att_tiled.ptr,
        rocblas_out, rocblas_ms);
    ProfileAttentionAvTiledDim4Candidate<32, 16, 18>(
        args, shape, "bm32bk16", d_scores.ptr, d_v.ptr, d_att_tiled.ptr,
        rocblas_out, rocblas_ms);
    ProfileAttentionAvTiledDim4Candidate<8, 64, 18>(
        args, shape, "bm8bk64", d_scores.ptr, d_v.ptr, d_att_tiled.ptr,
        rocblas_out, rocblas_ms);
    ProfileAttentionAvTiledDim4Candidate<16, 64, 18>(
        args, shape, "bm16bk64", d_scores.ptr, d_v.ptr, d_att_tiled.ptr,
        rocblas_out, rocblas_ms);
    ProfileAttentionAvTiledDim4Candidate<8, 128, 18>(
        args, shape, "bm8bk128", d_scores.ptr, d_v.ptr, d_att_tiled.ptr,
        rocblas_out, rocblas_ms);
  }
}

void BenchAttentionAvBf16WmmaProfile(rocblas_handle handle,
                                     const AttentionShape& shape,
                                     const Args& args) {
  const int rows = shape.seq;
  const int heads = shape.heads;
  const int qkv_dim = shape.dim;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_cols = heads * 3 * qkv_dim;
  const int qkv_projection_count = rows * qkv_cols;
  const int scores_count = heads * rows * rows;

  DeviceBuffer<float> d_qkv(qkv_projection_count);
  DeviceBuffer<float> d_q(qkv_count);
  DeviceBuffer<float> d_k(qkv_count);
  DeviceBuffer<float> d_v(qkv_count);
  DeviceBuffer<float> d_scores(scores_count);
  DeviceBuffer<rocblas_bfloat16> d_scores_bf16(scores_count);
  DeviceBuffer<rocblas_bfloat16> d_v_bf16(qkv_count);
  DeviceBuffer<float> d_att_f32(qkv_count);
  DeviceBuffer<float> d_att_bf16_rocblas(qkv_count);
  DeviceBuffer<float> d_att_bf16_wmma(qkv_count);

  const int qkv_threads = 256;
  const int softmax_threads = rows <= 256 ? 64 : 256;
  const int qkv_blocks = (qkv_count + qkv_threads - 1) / qkv_threads;
  const int scores_blocks = (scores_count + qkv_threads - 1) / qkv_threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;

  auto run_prepare_raw_scores = [&]() {
    hipLaunchKernelGGL(SplitQKVForAttentionKernel, dim3(qkv_blocks),
                       dim3(qkv_threads), 0, 0, d_qkv.ptr, d_q.ptr, d_k.ptr,
                       d_v.ptr, rows, heads, qkv_dim);
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
        qkv_dim, &alpha_qk, d_k.ptr, rocblas_datatype_f32_r, qkv_dim,
        stride_qkv, d_q.ptr, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
        &beta, d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores,
        d_scores.ptr, rocblas_datatype_f32_r, rows, stride_scores, heads,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };
  auto run_softmax_f32 = [&]() {
    const size_t softmax_shared =
        static_cast<size_t>(softmax_threads) * sizeof(float);
    hipLaunchKernelGGL(SoftmaxRowsKernel, dim3(rows, heads),
                       dim3(softmax_threads), softmax_shared, 0, d_scores.ptr,
                       rows);
  };
  auto run_softmax_bf16 = [&]() {
    const size_t softmax_shared =
        static_cast<size_t>(softmax_threads) * sizeof(float);
    hipLaunchKernelGGL(SoftmaxRowsToBF16Kernel, dim3(rows, heads),
                       dim3(softmax_threads), softmax_shared, 0, d_scores.ptr,
                       d_scores_bf16.ptr, rows);
  };
  auto run_quant_scores = [&]() {
    hipLaunchKernelGGL(F32ToBF16Kernel, dim3(scores_blocks), dim3(qkv_threads),
                       0, 0, d_scores.ptr, d_scores_bf16.ptr, scores_count);
  };
  auto run_quant_v = [&]() {
    hipLaunchKernelGGL(F32ToBF16Kernel, dim3(qkv_blocks), dim3(qkv_threads), 0,
                       0, d_v.ptr, d_v_bf16.ptr, qkv_count);
  };

  // Initialize all buffers with the conservative path: F32 QK, F32 softmax, then
  // explicit BF16 quantization. Later timings reset the score buffer as needed
  // so each reported phase is measured against the intended input format.
  run_prepare_raw_scores();
  run_softmax_f32();
  run_quant_scores();
  run_quant_v();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  StridedBatchedGemmSpec av_f32;
  av_f32.label = "av_f32";
  av_f32.trans_a = rocblas_operation_none;
  av_f32.trans_b = rocblas_operation_none;
  av_f32.m = qkv_dim;
  av_f32.n = rows;
  av_f32.k = rows;
  av_f32.a = d_v.ptr;
  av_f32.lda = qkv_dim;
  av_f32.stride_a = stride_qkv;
  av_f32.b = d_scores.ptr;
  av_f32.ldb = rows;
  av_f32.stride_b = stride_scores;
  av_f32.d = d_att_f32.ptr;
  av_f32.ldd = qkv_dim;
  av_f32.stride_d = stride_qkv;
  av_f32.batch_count = heads;
  av_f32.alpha = 1.0f;
  av_f32.beta = 0.0f;

  const int max_solutions =
      args.attention_rocblas_solution_sweep > 0
          ? args.attention_rocblas_solution_sweep
          : 8;
  std::printf("%-24s attention AV BF16 WMMA profile rows=%4d heads=%2d "
              "dim=%2d\n",
              shape.name, rows, heads, qkv_dim);
  const int f32_rocblas_solution =
      BenchOneAttentionRocblasSolutionSweep(handle, args, max_solutions,
                                            av_f32);

  auto run_f32_rocblas = [&]() {
    return RunStridedBatchedF32GemmSolution(handle, av_f32,
                                           f32_rocblas_solution);
  };
  auto run_bf16_rocblas = [&]() {
    return rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
        rows, &alpha_av, d_v_bf16.ptr, rocblas_datatype_bf16_r, qkv_dim,
        stride_qkv, d_scores_bf16.ptr, rocblas_datatype_bf16_r, rows,
        stride_scores, &beta, d_att_bf16_rocblas.ptr, rocblas_datatype_f32_r,
        qkv_dim, stride_qkv, d_att_bf16_rocblas.ptr, rocblas_datatype_f32_r,
        qkv_dim, stride_qkv, heads, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0);
  };

  float f32_rocblas_ms = 0.0f;
  float bf16_rocblas_ms = 0.0f;
  if (!TimeRocblasStatusMedian(args, run_f32_rocblas, f32_rocblas_ms)) {
    std::printf("  av F32 rocBLAS solution=%d failed\n",
                f32_rocblas_solution);
    return;
  }
  if (!TimeRocblasStatusMedian(args, run_bf16_rocblas, bf16_rocblas_ms)) {
    std::printf("  av BF16 rocBLAS failed\n");
    return;
  }

  run_prepare_raw_scores();
  const float f32_softmax_ms = TimeHipMedian(args, run_softmax_f32);
  run_prepare_raw_scores();
  const float softmax_bf16_ms = TimeHipMedian(args, run_softmax_bf16);
  run_prepare_raw_scores();
  run_softmax_f32();
  const float quant_scores_ms = TimeHipMedian(args, run_quant_scores);
  const float quant_v_ms = TimeHipMedian(args, run_quant_v);

  // Produce a F32 reference from raw scores, then overwrite only the BF16 score
  // buffer through the direct softmax->BF16 path. This makes the error numbers
  // reflect the precision boundary that a real BF16 WMMA AV backend would add,
  // rather than the older "F32 softmax then quantize scores" staging path.
  run_prepare_raw_scores();
  run_softmax_f32();
  ROCBLAS_CHECK(run_f32_rocblas());
  run_prepare_raw_scores();
  run_softmax_bf16();
  run_quant_v();
  ROCBLAS_CHECK(run_bf16_rocblas());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> f32_out(qkv_count);
  std::vector<float> bf16_rocblas_out(qkv_count);
  HIP_CHECK(hipMemcpy(f32_out.data(), d_att_f32.ptr, qkv_count * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(bf16_rocblas_out.data(), d_att_bf16_rocblas.ptr,
                      qkv_count * sizeof(float), hipMemcpyDeviceToHost));

  const auto [bf16_vs_f32_max, bf16_vs_f32_rms] =
      ErrorStats(bf16_rocblas_out, f32_out);

  const double flops = 2.0 * static_cast<double>(heads) * rows * rows * qkv_dim;
  const float f32_total_ms = f32_softmax_ms + f32_rocblas_ms;
  const float direct_bf16_rocblas_ms =
      softmax_bf16_ms + quant_v_ms + bf16_rocblas_ms;
  std::printf("  av_bf16_wmma tile sweep f32_solution=%7d "
              "f32_softmax=%8.3f ms f32_av=%8.3f ms f32_total=%8.3f ms "
              "softmax_bf16=%8.3f ms quant_scores=%8.3f ms quant_v=%8.3f ms "
              "bf16_rocblas_av=%8.3f ms direct_rocblas_total=%8.3f ms "
              "direct_rocblas/f32_total=%6.3f f32_av_gflops=%8.1f "
              "bf16_rocblas_gflops=%8.1f bf16_vs_f32_max=%9.6f "
              "bf16_vs_f32_rms=%9.6f\n",
              f32_rocblas_solution, f32_softmax_ms, f32_rocblas_ms,
              f32_total_ms, softmax_bf16_ms, quant_scores_ms, quant_v_ms,
              bf16_rocblas_ms, direct_bf16_rocblas_ms,
              direct_bf16_rocblas_ms / f32_total_ms,
              flops / (f32_rocblas_ms * 1.0E6),
              flops / (bf16_rocblas_ms * 1.0E6), bf16_vs_f32_max,
              bf16_vs_f32_rms);

  ProfileAttentionAvBf16WmmaCandidate<1, 4>(
      args, shape, "1x4", d_scores_bf16.ptr, d_v_bf16.ptr,
      d_att_bf16_wmma.ptr, f32_out, bf16_rocblas_out, f32_rocblas_ms,
      f32_total_ms, softmax_bf16_ms, quant_v_ms);
  ProfileAttentionAvBf16WmmaCandidate<1, 8>(
      args, shape, "1x8", d_scores_bf16.ptr, d_v_bf16.ptr,
      d_att_bf16_wmma.ptr, f32_out, bf16_rocblas_out, f32_rocblas_ms,
      f32_total_ms, softmax_bf16_ms, quant_v_ms);
  ProfileAttentionAvBf16WmmaCandidate<2, 2>(
      args, shape, "2x2", d_scores_bf16.ptr, d_v_bf16.ptr,
      d_att_bf16_wmma.ptr, f32_out, bf16_rocblas_out, f32_rocblas_ms,
      f32_total_ms, softmax_bf16_ms, quant_v_ms);
  ProfileAttentionAvBf16WmmaCandidate<2, 4>(
      args, shape, "2x4", d_scores_bf16.ptr, d_v_bf16.ptr,
      d_att_bf16_wmma.ptr, f32_out, bf16_rocblas_out, f32_rocblas_ms,
      f32_total_ms, softmax_bf16_ms, quant_v_ms);
  ProfileAttentionAvBf16WmmaCandidate<2, 8>(
      args, shape, "2x8", d_scores_bf16.ptr, d_v_bf16.ptr,
      d_att_bf16_wmma.ptr, f32_out, bf16_rocblas_out, f32_rocblas_ms,
      f32_total_ms, softmax_bf16_ms, quant_v_ms);
  ProfileAttentionAvBf16WmmaCandidate<4, 2>(
      args, shape, "4x2", d_scores_bf16.ptr, d_v_bf16.ptr,
      d_att_bf16_wmma.ptr, f32_out, bf16_rocblas_out, f32_rocblas_ms,
      f32_total_ms, softmax_bf16_ms, quant_v_ms);
  ProfileAttentionAvBf16WmmaCandidate<4, 4>(
      args, shape, "4x4", d_scores_bf16.ptr, d_v_bf16.ptr,
      d_att_bf16_wmma.ptr, f32_out, bf16_rocblas_out, f32_rocblas_ms,
      f32_total_ms, softmax_bf16_ms, quant_v_ms);
  ProfileAttentionAvBf16WmmaCandidate<4, 8>(
      args, shape, "4x8", d_scores_bf16.ptr, d_v_bf16.ptr,
      d_att_bf16_wmma.ptr, f32_out, bf16_rocblas_out, f32_rocblas_ms,
      f32_total_ms, softmax_bf16_ms, quant_v_ms);
  ProfileAttentionAvBf16WmmaCandidate<8, 1>(
      args, shape, "8x1", d_scores_bf16.ptr, d_v_bf16.ptr,
      d_att_bf16_wmma.ptr, f32_out, bf16_rocblas_out, f32_rocblas_ms,
      f32_total_ms, softmax_bf16_ms, quant_v_ms);
  ProfileAttentionAvBf16WmmaCandidate<8, 2>(
      args, shape, "8x2", d_scores_bf16.ptr, d_v_bf16.ptr,
      d_att_bf16_wmma.ptr, f32_out, bf16_rocblas_out, f32_rocblas_ms,
      f32_total_ms, softmax_bf16_ms, quant_v_ms);
  ProfileAttentionAvBf16WmmaCandidate<8, 4>(
      args, shape, "8x4", d_scores_bf16.ptr, d_v_bf16.ptr,
      d_att_bf16_wmma.ptr, f32_out, bf16_rocblas_out, f32_rocblas_ms,
      f32_total_ms, softmax_bf16_ms, quant_v_ms);
}

void BenchMlpCustomGemm(rocblas_handle handle, const Shape& shape,
                        const Args& args) {
  const size_t a_count = static_cast<size_t>(shape.m) * shape.k;
  const size_t b_count = static_cast<size_t>(shape.k) * shape.n;
  const size_t c_count = static_cast<size_t>(shape.m) * shape.n;

  DeviceBuffer<rocblas_bfloat16> d_a(a_count);
  DeviceBuffer<rocblas_bfloat16> d_b(b_count);
  DeviceBuffer<float> d_rocblas(c_count);
  DeviceBuffer<float> d_custom(c_count);
  HIP_CHECK(hipMemset(d_rocblas.ptr, 0, c_count * sizeof(float)));
  HIP_CHECK(hipMemset(d_custom.ptr, 0, c_count * sizeof(float)));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto run_rocblas = [&]() {
    // Same row-major trick as BenchOne, but with F32 output so it matches the
    // MLP backend's accumulation buffer. This is the reference both for timing
    // and for a loose numerical check of the custom kernel.
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, shape.n,
        shape.m, shape.k, &alpha, d_b.ptr, rocblas_datatype_bf16_r, shape.n,
        d_a.ptr, rocblas_datatype_bf16_r, shape.k, &beta, d_rocblas.ptr,
        rocblas_datatype_f32_r, shape.n, d_rocblas.ptr, rocblas_datatype_f32_r,
        shape.n, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };

  constexpr int BM = 16;
  constexpr int BN = 16;
  constexpr int BK = 32;
  const dim3 block(BN, BM);
  const dim3 grid((shape.n + BN - 1) / BN, (shape.m + BM - 1) / BM);
  auto run_custom = [&]() {
    hipLaunchKernelGGL((Bf16TiledGemmF32Kernel<BM, BN, BK>), grid, block, 0, 0,
                       d_a.ptr, d_b.ptr, d_custom.ptr, shape.m, shape.k,
                       shape.n);
  };

  // Check one full output against rocBLAS before timing. The comparison is not
  // bit-exact because rocBLAS uses a different reduction order, but a small RMS
  // error confirms that the row-major indexing and BF16 conversion are right.
  run_rocblas();
  run_custom();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> h_rocblas(c_count);
  std::vector<float> h_custom(c_count);
  HIP_CHECK(hipMemcpy(h_rocblas.data(), d_rocblas.ptr,
                      c_count * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_custom.data(), d_custom.ptr, c_count * sizeof(float),
                      hipMemcpyDeviceToHost));
  double squared_error = 0.0;
  float max_abs_error = 0.0f;
  for (size_t i = 0; i < c_count; ++i) {
    const float diff = h_custom[i] - h_rocblas[i];
    max_abs_error = std::max(max_abs_error, std::fabs(diff));
    squared_error += static_cast<double>(diff) * diff;
  }
  const float rms_error =
      static_cast<float>(std::sqrt(squared_error / c_count));

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms;
  };

  const std::vector<float> rocblas_ms = time_device(run_rocblas);
  const std::vector<float> custom_ms = time_device(run_custom);
  HIP_CHECK(hipGetLastError());

  const float rocblas_median = MedianMs(rocblas_ms);
  const float custom_median = MedianMs(custom_ms);
  const double flops = 2.0 * static_cast<double>(shape.m) *
                       static_cast<double>(shape.k) *
                       static_cast<double>(shape.n);
  std::printf("%-24s mlp_custom bf16->f32 rocBLAS=%8.3f ms custom=%8.3f ms "
              "custom/rocBLAS=%6.2fx custom_GFLOPS=%8.1f "
              "max_abs=%9.5f rms=%9.5f\n",
              shape.name, rocblas_median, custom_median,
              custom_median / rocblas_median, flops / (custom_median * 1.0E6),
              max_abs_error, rms_error);
}

void BenchMlpWmmaGemm(rocblas_handle handle, const Shape& shape,
                      const Args& args) {
  const size_t a_count = static_cast<size_t>(shape.m) * shape.k;
  const size_t b_count = static_cast<size_t>(shape.k) * shape.n;
  const size_t c_count = static_cast<size_t>(shape.m) * shape.n;

  DeviceBuffer<rocblas_bfloat16> d_a(a_count);
  DeviceBuffer<rocblas_bfloat16> d_b(b_count);
  DeviceBuffer<float> d_rocblas(c_count);
  DeviceBuffer<float> d_wmma(c_count);
  HIP_CHECK(hipMemset(d_rocblas.ptr, 0, c_count * sizeof(float)));
  HIP_CHECK(hipMemset(d_wmma.ptr, 0, c_count * sizeof(float)));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto run_rocblas = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, shape.n,
        shape.m, shape.k, &alpha, d_b.ptr, rocblas_datatype_bf16_r, shape.n,
        d_a.ptr, rocblas_datatype_bf16_r, shape.k, &beta, d_rocblas.ptr,
        rocblas_datatype_f32_r, shape.n, d_rocblas.ptr, rocblas_datatype_f32_r,
        shape.n, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };

  const dim3 block(32);
  const dim3 grid((shape.n + 15) / 16, (shape.m + 15) / 16);
  auto run_wmma = [&]() {
    hipLaunchKernelGGL(Bf16WmmaGemmF32Kernel, grid, block, 0, 0, d_a.ptr,
                       d_b.ptr, d_wmma.ptr, shape.m, shape.k, shape.n);
  };

  // Run both paths once before timing. The full-output comparison catches the
  // common WMMA failure modes: A/B fragment transposition mistakes, missing
  // half-wave replication, and incorrect even/odd row stores.
  run_rocblas();
  run_wmma();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> h_rocblas(c_count);
  std::vector<float> h_wmma(c_count);
  HIP_CHECK(hipMemcpy(h_rocblas.data(), d_rocblas.ptr,
                      c_count * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_wmma.data(), d_wmma.ptr, c_count * sizeof(float),
                      hipMemcpyDeviceToHost));
  double squared_error = 0.0;
  float max_abs_error = 0.0f;
  for (size_t i = 0; i < c_count; ++i) {
    const float diff = h_wmma[i] - h_rocblas[i];
    max_abs_error = std::max(max_abs_error, std::fabs(diff));
    squared_error += static_cast<double>(diff) * diff;
  }
  const float rms_error =
      static_cast<float>(std::sqrt(squared_error / c_count));

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms;
  };

  const std::vector<float> rocblas_ms = time_device(run_rocblas);
  const std::vector<float> wmma_ms = time_device(run_wmma);
  HIP_CHECK(hipGetLastError());

  const float rocblas_median = MedianMs(rocblas_ms);
  const float wmma_median = MedianMs(wmma_ms);
  const double flops = 2.0 * static_cast<double>(shape.m) *
                       static_cast<double>(shape.k) *
                       static_cast<double>(shape.n);
  std::printf("%-24s mlp_wmma   bf16->f32 rocBLAS=%8.3f ms wmma=%8.3f ms "
              "wmma/rocBLAS=%6.2fx wmma_GFLOPS=%8.1f "
              "max_abs=%9.5f rms=%9.5f\n",
              shape.name, rocblas_median, wmma_median,
              wmma_median / rocblas_median, flops / (wmma_median * 1.0E6),
              max_abs_error, rms_error);
}

template <int kWavesN>
void BenchMlpWmmaGroupedGemm(rocblas_handle handle, const Shape& shape,
                             const Args& args, const char* label) {
  const size_t a_count = static_cast<size_t>(shape.m) * shape.k;
  const size_t b_count = static_cast<size_t>(shape.k) * shape.n;
  const size_t c_count = static_cast<size_t>(shape.m) * shape.n;

  DeviceBuffer<rocblas_bfloat16> d_a(a_count);
  DeviceBuffer<rocblas_bfloat16> d_b(b_count);
  DeviceBuffer<float> d_rocblas(c_count);
  DeviceBuffer<float> d_grouped(c_count);
  HIP_CHECK(hipMemset(d_rocblas.ptr, 0, c_count * sizeof(float)));
  HIP_CHECK(hipMemset(d_grouped.ptr, 0, c_count * sizeof(float)));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto run_rocblas = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, shape.n,
        shape.m, shape.k, &alpha, d_b.ptr, rocblas_datatype_bf16_r, shape.n,
        d_a.ptr, rocblas_datatype_bf16_r, shape.k, &beta, d_rocblas.ptr,
        rocblas_datatype_f32_r, shape.n, d_rocblas.ptr, rocblas_datatype_f32_r,
        shape.n, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };

  const dim3 block(32 * kWavesN);
  const dim3 grid((shape.n + 16 * kWavesN - 1) / (16 * kWavesN),
                  (shape.m + 15) / 16);
  auto run_grouped = [&]() {
    hipLaunchKernelGGL((Bf16WmmaGemmF32GroupedNKernel<kWavesN>), grid, block, 0,
                       0, d_a.ptr, d_b.ptr, d_grouped.ptr, shape.m, shape.k,
                       shape.n);
  };

  // The grouped kernel has two extra correctness risks beyond the single-wave
  // version: cooperative LDS staging must cover every A element exactly once,
  // and each wave's N-tile offset must land in a disjoint output region.
  run_rocblas();
  run_grouped();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> h_rocblas(c_count);
  std::vector<float> h_grouped(c_count);
  HIP_CHECK(hipMemcpy(h_rocblas.data(), d_rocblas.ptr,
                      c_count * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_grouped.data(), d_grouped.ptr,
                      c_count * sizeof(float), hipMemcpyDeviceToHost));
  double squared_error = 0.0;
  float max_abs_error = 0.0f;
  for (size_t i = 0; i < c_count; ++i) {
    const float diff = h_grouped[i] - h_rocblas[i];
    max_abs_error = std::max(max_abs_error, std::fabs(diff));
    squared_error += static_cast<double>(diff) * diff;
  }
  const float rms_error =
      static_cast<float>(std::sqrt(squared_error / c_count));

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms;
  };

  const std::vector<float> rocblas_ms = time_device(run_rocblas);
  const std::vector<float> grouped_ms = time_device(run_grouped);
  HIP_CHECK(hipGetLastError());

  const float rocblas_median = MedianMs(rocblas_ms);
  const float grouped_median = MedianMs(grouped_ms);
  const double flops = 2.0 * static_cast<double>(shape.m) *
                       static_cast<double>(shape.k) *
                       static_cast<double>(shape.n);
  std::printf("%-24s %-10s bf16->f32 rocBLAS=%8.3f ms %s=%8.3f ms "
              "%s/rocBLAS=%6.2fx %s_GFLOPS=%8.1f "
              "max_abs=%9.5f rms=%9.5f\n",
              shape.name, label, rocblas_median, label, grouped_median, label,
              grouped_median / rocblas_median, label,
              flops / (grouped_median * 1.0E6), max_abs_error, rms_error);
}

template <int kWavesM>
void BenchMlpWmmaGroupedMGemm(rocblas_handle handle, const Shape& shape,
                              const Args& args, const char* label) {
  const size_t a_count = static_cast<size_t>(shape.m) * shape.k;
  const size_t b_count = static_cast<size_t>(shape.k) * shape.n;
  const size_t c_count = static_cast<size_t>(shape.m) * shape.n;

  DeviceBuffer<rocblas_bfloat16> d_a(a_count);
  DeviceBuffer<rocblas_bfloat16> d_b(b_count);
  DeviceBuffer<float> d_rocblas(c_count);
  DeviceBuffer<float> d_grouped(c_count);
  HIP_CHECK(hipMemset(d_rocblas.ptr, 0, c_count * sizeof(float)));
  HIP_CHECK(hipMemset(d_grouped.ptr, 0, c_count * sizeof(float)));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto run_rocblas = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, shape.n,
        shape.m, shape.k, &alpha, d_b.ptr, rocblas_datatype_bf16_r, shape.n,
        d_a.ptr, rocblas_datatype_bf16_r, shape.k, &beta, d_rocblas.ptr,
        rocblas_datatype_f32_r, shape.n, d_rocblas.ptr, rocblas_datatype_f32_r,
        shape.n, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };

  const dim3 block(32 * kWavesM);
  const dim3 grid((shape.n + 15) / 16,
                  (shape.m + 16 * kWavesM - 1) / (16 * kWavesM));
  auto run_grouped = [&]() {
    hipLaunchKernelGGL((Bf16WmmaGemmF32GroupedMKernel<kWavesM>), grid, block, 0,
                       0, d_a.ptr, d_b.ptr, d_grouped.ptr, shape.m, shape.k,
                       shape.n);
  };

  run_rocblas();
  run_grouped();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> h_rocblas(c_count);
  std::vector<float> h_grouped(c_count);
  HIP_CHECK(hipMemcpy(h_rocblas.data(), d_rocblas.ptr,
                      c_count * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_grouped.data(), d_grouped.ptr,
                      c_count * sizeof(float), hipMemcpyDeviceToHost));
  double squared_error = 0.0;
  float max_abs_error = 0.0f;
  for (size_t i = 0; i < c_count; ++i) {
    const float diff = h_grouped[i] - h_rocblas[i];
    max_abs_error = std::max(max_abs_error, std::fabs(diff));
    squared_error += static_cast<double>(diff) * diff;
  }
  const float rms_error =
      static_cast<float>(std::sqrt(squared_error / c_count));

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms;
  };

  const std::vector<float> rocblas_ms = time_device(run_rocblas);
  const std::vector<float> grouped_ms = time_device(run_grouped);
  HIP_CHECK(hipGetLastError());

  const float rocblas_median = MedianMs(rocblas_ms);
  const float grouped_median = MedianMs(grouped_ms);
  const double flops = 2.0 * static_cast<double>(shape.m) *
                       static_cast<double>(shape.k) *
                       static_cast<double>(shape.n);
  std::printf("%-24s %-11s bf16->f32 rocBLAS=%8.3f ms %s=%8.3f ms "
              "%s/rocBLAS=%6.2fx %s_GFLOPS=%8.1f "
              "max_abs=%9.5f rms=%9.5f\n",
              shape.name, label, rocblas_median, label, grouped_median, label,
              grouped_median / rocblas_median, label,
              flops / (grouped_median * 1.0E6), max_abs_error, rms_error);
}

template <int kWavesM, int kWavesN>
void BenchMlpWmmaGrouped2DGemm(rocblas_handle handle, const Shape& shape,
                               const Args& args, const char* label) {
  const size_t a_count = static_cast<size_t>(shape.m) * shape.k;
  const size_t b_count = static_cast<size_t>(shape.k) * shape.n;
  const size_t c_count = static_cast<size_t>(shape.m) * shape.n;

  DeviceBuffer<rocblas_bfloat16> d_a(a_count);
  DeviceBuffer<rocblas_bfloat16> d_b(b_count);
  DeviceBuffer<float> d_rocblas(c_count);
  DeviceBuffer<float> d_grouped(c_count);
  HIP_CHECK(hipMemset(d_rocblas.ptr, 0, c_count * sizeof(float)));
  HIP_CHECK(hipMemset(d_grouped.ptr, 0, c_count * sizeof(float)));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto run_rocblas = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, shape.n,
        shape.m, shape.k, &alpha, d_b.ptr, rocblas_datatype_bf16_r, shape.n,
        d_a.ptr, rocblas_datatype_bf16_r, shape.k, &beta, d_rocblas.ptr,
        rocblas_datatype_f32_r, shape.n, d_rocblas.ptr, rocblas_datatype_f32_r,
        shape.n, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };

  const dim3 block(32 * kWavesM * kWavesN);
  const dim3 grid((shape.n + 16 * kWavesN - 1) / (16 * kWavesN),
                  (shape.m + 16 * kWavesM - 1) / (16 * kWavesM));
  auto run_grouped = [&]() {
    hipLaunchKernelGGL((Bf16WmmaGemmF32Grouped2DKernel<kWavesM, kWavesN>),
                       grid, block, 0, 0, d_a.ptr, d_b.ptr, d_grouped.ptr,
                       shape.m, shape.k, shape.n);
  };

  // Validate the entire output once before timing. The 2D schedule has more
  // offset arithmetic than the 1D grouped kernels, so this catches swapped
  // wave-M/wave-N mappings and boundary mistakes before a fast-looking result
  // can mislead the tuning loop.
  run_rocblas();
  run_grouped();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> h_rocblas(c_count);
  std::vector<float> h_grouped(c_count);
  HIP_CHECK(hipMemcpy(h_rocblas.data(), d_rocblas.ptr,
                      c_count * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_grouped.data(), d_grouped.ptr,
                      c_count * sizeof(float), hipMemcpyDeviceToHost));
  double squared_error = 0.0;
  float max_abs_error = 0.0f;
  for (size_t i = 0; i < c_count; ++i) {
    const float diff = h_grouped[i] - h_rocblas[i];
    max_abs_error = std::max(max_abs_error, std::fabs(diff));
    squared_error += static_cast<double>(diff) * diff;
  }
  const float rms_error =
      static_cast<float>(std::sqrt(squared_error / c_count));

  auto time_device = [&](const auto& fn) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) fn();
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) fn();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms;
  };

  const std::vector<float> rocblas_ms = time_device(run_rocblas);
  const std::vector<float> grouped_ms = time_device(run_grouped);
  HIP_CHECK(hipGetLastError());

  const float rocblas_median = MedianMs(rocblas_ms);
  const float grouped_median = MedianMs(grouped_ms);
  const double flops = 2.0 * static_cast<double>(shape.m) *
                       static_cast<double>(shape.k) *
                       static_cast<double>(shape.n);
  std::printf("%-24s %-12s bf16->f32 rocBLAS=%8.3f ms %s=%8.3f ms "
              "%s/rocBLAS=%6.2fx %s_GFLOPS=%8.1f "
              "max_abs=%9.5f rms=%9.5f\n",
              shape.name, label, rocblas_median, label, grouped_median, label,
              grouped_median / rocblas_median, label,
              flops / (grouped_median * 1.0E6), max_abs_error, rms_error);
}

void BenchVit448Pipeline(rocblas_handle handle, const Args& args) {
  // Fixed PaliGemma2 448 ViT dimensions. The goal is not to reproduce the whole
  // image encoder, but to time the dominant GEMM schedule with the same sequence
  // length, hidden size, MLP size, and 16-head attention dimensions.
  const Shape patch = {"patch_embed_448", 1024, 588, 1152, true};
  const Shape qkv = {"qkv_448", 1024, 1152, 3456, true};
  const Shape attn_out = {"attn_out_448", 1024, 1152, 1152, true};
  const Shape mlp_up = {"mlp_up_448", 1024, 1152, 4304, true};
  const Shape mlp_down = {"mlp_down_448", 1024, 4304, 1152, true};
  const Shape head = {"head_proj_448_3b", 1024, 1152, 2304, true};
  const BatchedShape qk = {"vit_qk_448_16h_f32", 1024, 72, 1024, 16};
  const BatchedShape av = {"vit_av_448_16h_f32", 1024, 1024, 72, 16};

  const size_t max_bf16_a =
      static_cast<size_t>(mlp_down.m) * mlp_down.k;
  const size_t max_bf16_b =
      static_cast<size_t>(mlp_down.k) * mlp_down.n;
  const size_t max_bf16_c = static_cast<size_t>(mlp_up.m) * mlp_up.n;
  DeviceBuffer<rocblas_bfloat16> bf16_a(max_bf16_a);
  DeviceBuffer<rocblas_bfloat16> bf16_b(max_bf16_b);
  DeviceBuffer<rocblas_bfloat16> bf16_c(max_bf16_c);

  const size_t qk_count = static_cast<size_t>(qk.batch) * qk.m * qk.n;
  const size_t f32_a_count = static_cast<size_t>(av.batch) * av.m * av.k;
  const size_t f32_b_count = static_cast<size_t>(av.batch) * av.k * av.n;
  DeviceBuffer<float> f32_a(f32_a_count);
  DeviceBuffer<float> f32_b(f32_b_count);
  DeviceBuffer<float> f32_c(qk_count);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Reuse maximum-sized BF16 buffers for every projection shape. This avoids
  // allocation noise and approximates the cost of issuing the GEMMs when tensors
  // are already resident on the GPU.
  auto run_bf16 = [&](const Shape& shape) {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, shape.n,
        shape.m, shape.k, &alpha, bf16_b.ptr, rocblas_datatype_bf16_r,
        shape.n, bf16_a.ptr, rocblas_datatype_bf16_r, shape.k, &beta,
        bf16_c.ptr, rocblas_datatype_bf16_r, shape.n, bf16_c.ptr,
        rocblas_datatype_bf16_r, shape.n, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
  };

  // Reuse maximum-sized F32 buffers for QK/AV. These numbers intentionally omit
  // softmax, so they are an optimistic rocBLAS-only attention schedule.
  auto run_f32_batched = [&](const BatchedShape& shape) {
    const rocblas_stride stride_a =
        static_cast<rocblas_stride>(shape.m) * shape.k;
    const rocblas_stride stride_b =
        static_cast<rocblas_stride>(shape.k) * shape.n;
    const rocblas_stride stride_c =
        static_cast<rocblas_stride>(shape.m) * shape.n;
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle, rocblas_operation_none, rocblas_operation_none, shape.n,
        shape.m, shape.k, &alpha, f32_b.ptr, rocblas_datatype_f32_r, shape.n,
        stride_b, f32_a.ptr, rocblas_datatype_f32_r, shape.k, stride_a, &beta,
        f32_c.ptr, rocblas_datatype_f32_r, shape.n, stride_c, f32_c.ptr,
        rocblas_datatype_f32_r, shape.n, stride_c, shape.batch,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };

  // Projection-only schedule: patch embedding, then four projection GEMMs per
  // ViT layer, then the image-token head projection used by PaliGemma2 3B.
  auto run_projection_pipeline = [&]() {
    run_bf16(patch);
    for (int layer = 0; layer < 27; ++layer) {
      run_bf16(qkv);
      run_bf16(attn_out);
      run_bf16(mlp_up);
      run_bf16(mlp_down);
    }
    run_bf16(head);
  };

  // Projection plus rocBLAS QK/AV schedule. This estimates the upper bound of a
  // naive rocBLAS attention path before adding softmax and layout work.
  auto run_projection_plus_attention_gemm = [&]() {
    run_bf16(patch);
    for (int layer = 0; layer < 27; ++layer) {
      run_bf16(qkv);
      run_f32_batched(qk);
      run_f32_batched(av);
      run_bf16(attn_out);
      run_bf16(mlp_up);
      run_bf16(mlp_down);
    }
    run_bf16(head);
  };

  auto shape_flops = [](const Shape& shape) {
    return 2.0 * static_cast<double>(shape.m) * shape.k * shape.n;
  };
  auto batched_flops = [](const BatchedShape& shape) {
    return 2.0 * static_cast<double>(shape.batch) * shape.m * shape.k *
           shape.n;
  };
  const double projection_flops =
      shape_flops(patch) +
      27.0 * (shape_flops(qkv) + shape_flops(attn_out) +
              shape_flops(mlp_up) + shape_flops(mlp_down)) +
      shape_flops(head);
  const double projection_attention_flops =
      projection_flops + 27.0 * (batched_flops(qk) + batched_flops(av));

  // Common pipeline timing wrapper. Warmup and samples measure an entire
  // repeated schedule instead of individual kernels, which exposes launch
  // sequencing overhead more realistically than isolated shape timings.
  auto bench_pipeline = [&](const char* name, const auto& func, double flops) {
    for (int i = 0; i < args.warmup; ++i) {
      for (int j = 0; j < args.iters; ++j) func();
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> ms;
    ms.reserve(args.samples);
    for (int sample = 0; sample < args.samples; ++sample) {
      HIP_CHECK(hipEventRecord(start));
      for (int iter = 0; iter < args.iters; ++iter) func();
      HIP_CHECK(hipEventRecord(stop));
      HIP_CHECK(hipEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      ms.push_back(elapsed_ms / static_cast<float>(args.iters));
    }
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    PrintPipelineResult(name, ms, flops);
  };

  std::printf("\nViT 448 rocBLAS schedule estimates, excluding layernorm, GELU, "
              "softmax, residual adds, and CPU/GPU transfer:\n");
  bench_pipeline("vit448_projection_gemms", run_projection_pipeline,
                 projection_flops);
  bench_pipeline("vit448_projection_plus_qk_av", run_projection_plus_attention_gemm,
                 projection_attention_flops);
}

int Main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, args)) return 1;

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::fprintf(stderr, "No HIP device found.\n");
    return 1;
  }
  int device = 0;
  HIP_CHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::printf("HIP device %d: %s arch=%s CUs=%d global_mem=%.1f GiB\n", device,
              props.name, props.gcnArchName, props.multiProcessorCount,
              static_cast<double>(props.totalGlobalMem) / (1024.0 * 1024.0 *
                                                           1024.0));

  rocblas_handle handle = nullptr;
  ROCBLAS_CHECK(rocblas_create_handle(&handle));
  ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

  const Shape shapes[] = {
      {"patch_embed_224", 256, 588, 1152, true},
      {"qkv_224", 256, 1152, 3456, true},
      {"attn_out_224", 256, 1152, 1152, true},
      {"mlp_up_224", 256, 1152, 4304, true},
      {"mlp_down_224", 256, 4304, 1152, true},
      {"head_proj_224_3b", 256, 1152, 2304, true},
      {"patch_embed_448", 1024, 588, 1152, true},
      {"qkv_448", 1024, 1152, 3456, true},
      {"attn_out_448", 1024, 1152, 1152, true},
      {"mlp_up_448", 1024, 1152, 4304, true},
      {"mlp_down_448", 1024, 4304, 1152, true},
      {"head_proj_448_3b", 1024, 1152, 2304, true},
      {"vit_qk_head_224_f32", 256, 72, 256, false},
      {"vit_av_head_224_f32", 256, 256, 72, false},
      {"vit_qk_head_448_f32", 1024, 72, 1024, false},
      {"vit_av_head_448_f32", 1024, 1024, 72, false},
  };

  const BatchedShape batched_shapes[] = {
      {"vit_qk_224_16h_f32", 256, 72, 256, 16},
      {"vit_av_224_16h_f32", 256, 256, 72, 16},
      {"vit_qk_448_16h_f32", 1024, 72, 1024, 16},
      {"vit_av_448_16h_f32", 1024, 1024, 72, 16},
  };

  // Direct fused attention shapes are reported beside rocBLAS GEMMs so we can
  // make an explicit keep/drop decision for custom HIP attention work.
  const AttentionShape attention_shapes[] = {
      {"vit_attn_fused_224_f32", 256, 16, 72},
      {"vit_attn_fused_448_f32", 1024, 16, 72},
  };

  const Shape mlp_kernel_shapes[] = {
      {"mlp_up_224", 256, 1152, 4304, true},
      {"mlp_down_224", 256, 4304, 1152, true},
      {"mlp_up_448", 1024, 1152, 4304, true},
      {"mlp_down_448", 1024, 4304, 1152, true},
  };

  const Shape projection_kernel_shapes[] = {
      {"patch_embed_224", 256, 588, 1152, true},
      {"qkv_224", 256, 1152, 3456, true},
      {"attn_out_224", 256, 1152, 1152, true},
      {"head_proj_224_3b", 256, 1152, 2304, true},
      {"patch_embed_448", 1024, 588, 1152, true},
      {"qkv_448", 1024, 1152, 3456, true},
      {"attn_out_448", 1024, 1152, 1152, true},
      {"head_proj_448_3b", 1024, 1152, 2304, true},
  };

  if (args.kernel_occupancy_report) {
    BenchKernelOccupancyReport(props);
  }

  if (args.shapes) {
    for (const Shape& shape : shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      if (shape.bf16) {
        BenchOne<rocblas_bfloat16>(handle, shape, args);
      } else {
        BenchOne<float>(handle, shape, args);
      }
    }
    for (const BatchedShape& shape : batched_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchBatchedF32(handle, shape, args);
    }
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchFusedAttention(shape, args);
    }
  }
  if (args.pipeline) {
    BenchVit448Pipeline(handle, args);
  }
  if (args.mlp_kernel) {
    std::printf("\nViT MLP scalar HIP kernel prototype, BF16 inputs and F32 "
                "output:\n");
    for (const Shape& shape : mlp_kernel_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchMlpCustomGemm(handle, shape, args);
    }
  }
  if (args.mlp_wmma) {
    std::printf("\nViT MLP RDNA wave32 WMMA prototype, BF16 inputs and F32 "
                "output:\n");
    for (const Shape& shape : mlp_kernel_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchMlpWmmaGemm(handle, shape, args);
    }
  }
  if (args.mlp_wmma_group) {
    std::printf("\nViT MLP grouped-N RDNA WMMA prototype, BF16 inputs and F32 "
                "output:\n");
    for (const Shape& shape : mlp_kernel_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchMlpWmmaGroupedGemm<8>(handle, shape, args, "mlp_wmma8");
    }
  }
  if (args.mlp_wmma_group_sweep) {
    std::printf("\nViT MLP grouped-N RDNA WMMA wave-count sweep, BF16 inputs "
                "and F32 output:\n");
    for (const Shape& shape : mlp_kernel_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchMlpWmmaGroupedGemm<2>(handle, shape, args, "mlp_wmma2");
      BenchMlpWmmaGroupedGemm<4>(handle, shape, args, "mlp_wmma4");
      BenchMlpWmmaGroupedGemm<6>(handle, shape, args, "mlp_wmma6");
      BenchMlpWmmaGroupedGemm<8>(handle, shape, args, "mlp_wmma8");
      BenchMlpWmmaGroupedGemm<12>(handle, shape, args, "mlp_wmma12");
      BenchMlpWmmaGroupedGemm<16>(handle, shape, args, "mlp_wmma16");
    }
  }
  if (args.mlp_wmma_m_group_sweep) {
    std::printf("\nViT MLP grouped-M RDNA WMMA wave-count sweep, BF16 inputs "
                "and F32 output:\n");
    for (const Shape& shape : mlp_kernel_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchMlpWmmaGroupedMGemm<2>(handle, shape, args, "mlp_wmma_m2");
      BenchMlpWmmaGroupedMGemm<4>(handle, shape, args, "mlp_wmma_m4");
      BenchMlpWmmaGroupedMGemm<8>(handle, shape, args, "mlp_wmma_m8");
    }
  }
  if (args.mlp_wmma_2d_group_sweep) {
    std::printf("\nViT MLP grouped-2D RDNA WMMA macro-tile sweep, BF16 inputs "
                "and F32 output:\n");
    for (const Shape& shape : mlp_kernel_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchMlpWmmaGrouped2DGemm<2, 2>(handle, shape, args, "mlp_wmma_2x2");
      BenchMlpWmmaGrouped2DGemm<2, 4>(handle, shape, args, "mlp_wmma_2x4");
      BenchMlpWmmaGrouped2DGemm<2, 8>(handle, shape, args, "mlp_wmma_2x8");
      BenchMlpWmmaGrouped2DGemm<4, 2>(handle, shape, args, "mlp_wmma_4x2");
      BenchMlpWmmaGrouped2DGemm<4, 4>(handle, shape, args, "mlp_wmma_4x4");
      BenchMlpWmmaGrouped2DGemm<4, 8>(handle, shape, args, "mlp_wmma_4x8");
      BenchMlpWmmaGrouped2DGemm<8, 2>(handle, shape, args, "mlp_wmma_8x2");
      BenchMlpWmmaGrouped2DGemm<8, 4>(handle, shape, args, "mlp_wmma_8x4");
    }
  }
  if (args.projection_wmma_2d_sweep) {
    std::printf("\nViT projection grouped-2D RDNA WMMA macro-tile sweep, BF16 "
                "inputs and F32 output:\n");
    for (const Shape& shape : projection_kernel_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchMlpWmmaGrouped2DGemm<2, 2>(handle, shape, args, "wmma2d_2x2");
      BenchMlpWmmaGrouped2DGemm<2, 4>(handle, shape, args, "wmma2d_2x4");
      BenchMlpWmmaGrouped2DGemm<2, 8>(handle, shape, args, "wmma2d_2x8");
      BenchMlpWmmaGrouped2DGemm<4, 2>(handle, shape, args, "wmma2d_4x2");
      BenchMlpWmmaGrouped2DGemm<4, 4>(handle, shape, args, "wmma2d_4x4");
      BenchMlpWmmaGrouped2DGemm<4, 8>(handle, shape, args, "wmma2d_4x8");
      BenchMlpWmmaGrouped2DGemm<8, 2>(handle, shape, args, "wmma2d_8x2");
      BenchMlpWmmaGrouped2DGemm<8, 4>(handle, shape, args, "wmma2d_8x4");
    }
  }
  if (args.attention_phase_profile) {
    std::printf("\nViT attention current device-path phase profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionPhaseProfile(handle, shape, args);
    }
  }
  if (args.attention_recommended_phase_profile) {
    std::printf("\nViT attention recommended F32 cached-solution phase "
                "profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionRecommendedPhaseProfile(handle, shape, args);
    }
  }
  if (args.attention_deferred_softmax_scale_profile) {
    std::printf("\nViT attention experimental deferred softmax-scale "
                "profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionDeferredSoftmaxScaleProfile(handle, shape, args);
    }
  }
  if (args.attention_bf16_phase_profile) {
    std::printf("\nViT attention experimental BF16 QK/AV phase profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionBf16PhaseProfile(handle, shape, args);
    }
  }
  if (args.attention_bf16_qk_phase_profile) {
    std::printf("\nViT attention experimental BF16 QK + F32 AV phase "
                "profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionBf16QkPhaseProfile(handle, shape, args);
    }
  }
  if (args.attention_qk_bf16_wmma_profile) {
    std::printf("\nViT attention experimental BF16 WMMA QK profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionQkBf16WmmaProfile(handle, shape, args);
    }
  }
  if (args.attention_qk_softmax_fused_profile) {
    std::printf("\nViT attention experimental F32 fused QK+softmax "
                "profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionQkSoftmaxFusedProfile(handle, shape, args);
    }
  }
  if (args.attention_softmax_av_fused_profile) {
    std::printf("\nViT attention experimental F32 QK + fused F32 softmax/AV "
                "phase profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionSoftmaxAvFusedProfile(handle, shape, args);
    }
  }
  if (args.attention_softmax_av_grouped_profile) {
    std::printf("\nViT attention experimental F32 grouped softmax/AV "
                "profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionSoftmaxAvGroupedProfile(handle, shape, args);
    }
  }
  if (args.attention_softmax_thread_sweep) {
    std::printf("\nViT attention F32 softmax block-size sweep:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionSoftmaxThreadSweep(handle, shape, args);
    }
  }
  if (args.attention_softmax_fixed_profile) {
    std::printf("\nViT attention F32 fixed-shape softmax profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionSoftmaxFixedProfile(handle, shape, args);
    }
  }
  if (args.attention_split_vectorized_profile) {
    std::printf("\nViT attention F32 Q/K/V split vectorization profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionSplitVectorizedProfile(shape, args);
    }
  }
  if (args.attention_pack_vectorized_profile) {
    std::printf("\nViT attention BF16 pack vectorization profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionPackVectorizedProfile(shape, args);
    }
  }
  if (args.attention_rocblas_solution_sweep > 0) {
    std::printf("\nViT attention F32 rocBLAS QK/AV solution sweep:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionRocblasSolutionSweep(handle, shape, args);
    }
  }
  if (args.attention_qk_tiled_profile) {
    std::printf("\nViT attention experimental F32 QK tiled profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionQkTiledProfile(handle, shape, args);
    }
  }
  if (args.attention_av_tiled_profile) {
    std::printf("\nViT attention experimental F32 AV tiled profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionAvTiledProfile(handle, shape, args);
    }
  }
  if (args.attention_online_head_major_profile) {
    std::printf("\nViT attention experimental F32 online head-major "
                "profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionOnlineHeadMajorProfile(handle, shape, args);
    }
  }
  if (args.attention_online_multiquery_profile) {
    std::printf("\nViT attention experimental F32 online multi-query "
                "profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionOnlineMultiQueryProfile(handle, shape, args);
    }
  }
  if (args.attention_av_bf16_wmma_profile) {
    std::printf("\nViT attention experimental BF16 WMMA AV profile:\n");
    for (const AttentionShape& shape : attention_shapes) {
      if (!args.filter.empty() &&
          std::string(shape.name).find(args.filter) == std::string::npos) {
        continue;
      }
      BenchAttentionAvBf16WmmaProfile(handle, shape, args);
    }
  }

  ROCBLAS_CHECK(rocblas_destroy_handle(handle));
  return 0;
}

}  // namespace

int main(int argc, char** argv) { return Main(argc, argv); }
