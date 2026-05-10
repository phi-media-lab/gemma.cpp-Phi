// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include "experimental/paligemma2_vit_hip_backend.h"

#include <hip/hip_runtime.h>
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gemma/gemma.h"
#include "gemma/weights.h"
#include "hwy/base.h"
#include "hwy/timer.h"

namespace gcpp {
namespace experimental {
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

struct Shape {
  const char* name;
  int m;
  int k;
  int n;
};

// Minimal RAII wrapper for HIP allocations. The backend creates many temporary
// device buffers while probing shapes; tying hipFree to object lifetime avoids
// leaking memory when a benchmark path exits early on an error.
template <class T>
struct DeviceBuffer {
  T* ptr = nullptr;
  size_t count = 0;

  explicit DeviceBuffer(size_t count_in) : count(count_in) {
    HIP_CHECK(hipMalloc(&ptr, count * sizeof(T)));
    HIP_CHECK(hipMemset(ptr, 0, count * sizeof(T)));
  }

  ~DeviceBuffer() {
    if (ptr != nullptr) {
      (void)hipFree(ptr);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct ResidentBF16Mat {
  // Resident projection weight in rocBLAS-friendly B[K,N] layout. `k` is the
  // input dimension and `n` is the output dimension; the original gemma.cpp
  // MatPtr is stored as W[N,K].
  std::string label;
  int k = 0;
  int n = 0;
  std::unique_ptr<DeviceBuffer<rocblas_bfloat16>> device;
};

struct ResidentF32Mat {
  // F32 shadow weight used only for the patch embedding correctness bridge.
  // rocBLAS does not support F32 activations x BF16 weights in one GEMM, so this
  // lets us match current CPU semantics before testing the faster BF16 path.
  std::string label;
  int k = 0;
  int n = 0;
  std::unique_ptr<DeviceBuffer<float>> device;
};

struct ResidentF32Vec {
  // Small F32 vectors used by non-GEMM ViT operations. Keeping these resident
  // removes repeated H2D uploads of LayerNorm scales/biases and projection
  // biases from the production image-token path.
  std::string label;
  int count = 0;
  std::unique_ptr<DeviceBuffer<float>> device;
};

struct ResidentProjectionLayer {
  ResidentBF16Mat qkv;
  ResidentBF16Mat attn_out;
  ResidentBF16Mat linear_0;
  ResidentBF16Mat linear_1;
  ResidentF32Vec ln0_scale;
  ResidentF32Vec ln0_bias;
  ResidentF32Vec ln1_scale;
  ResidentF32Vec ln1_bias;
  ResidentF32Vec qkv_bias;
  ResidentF32Vec attn_out_bias;
  ResidentF32Vec linear_0_bias;
  ResidentF32Vec linear_1_bias;
};

struct ResidentProjectionWeights {
  // patch_embed/head plus four per-layer projection weights are the first
  // resident subset because they account for most ViT dense GEMM work and fit
  // comfortably in the local iGPU shared/global pool.
  ResidentBF16Mat patch_embed;
  std::unique_ptr<ResidentF32Mat> patch_embed_f32;
  ResidentBF16Mat head;
  ResidentF32Vec patch_bias;
  ResidentF32Vec pos_embedding;
  ResidentF32Vec enc_norm_scale;
  ResidentF32Vec enc_norm_bias;
  ResidentF32Vec head_bias;
  std::vector<ResidentProjectionLayer> layers;

  size_t TotalBytes() const {
    size_t bf16_count = patch_embed.device->count + head.device->count;
    size_t f32_count = patch_bias.device->count + pos_embedding.device->count +
                       enc_norm_scale.device->count +
                       enc_norm_bias.device->count + head_bias.device->count;
    for (const ResidentProjectionLayer& layer : layers) {
      bf16_count += layer.qkv.device->count;
      bf16_count += layer.attn_out.device->count;
      bf16_count += layer.linear_0.device->count;
      bf16_count += layer.linear_1.device->count;
      f32_count += layer.ln0_scale.device->count;
      f32_count += layer.ln0_bias.device->count;
      f32_count += layer.ln1_scale.device->count;
      f32_count += layer.ln1_bias.device->count;
      f32_count += layer.qkv_bias.device->count;
      f32_count += layer.attn_out_bias.device->count;
      f32_count += layer.linear_0_bias.device->count;
      f32_count += layer.linear_1_bias.device->count;
    }
    size_t bytes = bf16_count * sizeof(rocblas_bfloat16) +
                   f32_count * sizeof(float);
    if (patch_embed_f32 != nullptr) {
      bytes += patch_embed_f32->device->count * sizeof(float);
    }
    return bytes;
  }
};

struct ErrorStats {
  float max_abs = 0.0f;
  float rms = 0.0f;
};

// Adds the two non-GEMM terms of patch embedding:
//   output[row, col] += img_emb_bias[col] + img_pos_emb[row, col]
// Keeping this as a separate tiny kernel makes the GEMM timing include the
// same tensor boundary that the CPU EmbedImagePatches path produces.
__global__ void AddBiasAndPosKernel(float* c, const float* bias,
                                    const float* pos, int rows, int cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * cols;
  if (idx >= count) return;
  c[idx] += bias[idx % cols] + pos[idx];
}

__global__ void AddBiasKernel(float* c, const float* bias, int rows, int cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * cols;
  if (idx >= count) return;
  c[idx] += bias[idx % cols];
}

__global__ void AddResidualKernel(float* c, const float* residual, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  c[idx] += residual[idx];
}

// Device GELU mirrors ops/ops-inl.h's tanh approximation. Keeping GELU on
// device is important for the MLP probe because otherwise the up-projection
// timing would hide a large GPU->CPU->GPU transfer that the final backend must
// not perform.
__device__ float GeluApprox(float x) {
  return x * (0.5f + 0.5f * tanhf(x * (0.797884560804236f +
                                        0.03567740813636141f * x * x)));
}

__global__ void GeluKernel(float* c, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  c[idx] = GeluApprox(c[idx]);
}

// Quantizes the first MLP projection after adding its bias. The CPU ViT
// activations store C1 as BF16, so the BF16 route should round after
// linear_0+bias before GELU instead of carrying an unrealistically precise F32
// hidden tensor into the second GEMM.
__global__ void AddBiasToBF16Kernel(const float* c, const float* bias,
                                    rocblas_bfloat16* out, int rows,
                                    int cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * cols;
  if (idx >= count) return;
  out[idx] = rocblas_bfloat16(c[idx] + bias[idx % cols]);
}

__global__ void GeluBF16Kernel(rocblas_bfloat16* c, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  c[idx] = rocblas_bfloat16(GeluApprox(static_cast<float>(c[idx])));
}

__global__ void AddBiasGeluToBF16Kernel(const float* c, const float* bias,
                                        rocblas_bfloat16* out, int rows,
                                        int cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * cols;
  if (idx >= count) return;
  // Preserve the current BF16 route's rounding boundary:
  //   linear_0_accum + bias -> BF16 -> GELU -> BF16.
  // This fuses the two elementwise kernel launches without changing the tensor
  // precision seen by the second MLP GEMM.
  const rocblas_bfloat16 biased = rocblas_bfloat16(c[idx] + bias[idx % cols]);
  out[idx] = rocblas_bfloat16(GeluApprox(static_cast<float>(biased)));
}

__global__ void F32ToBF16Kernel(const float* src, rocblas_bfloat16* dst,
                                int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  dst[idx] = rocblas_bfloat16(src[idx]);
}

using WmmaBf16Frag = short __attribute__((ext_vector_type(16)));
using WmmaF32Frag = float __attribute__((ext_vector_type(8)));

__device__ short Bf16Bits(rocblas_bfloat16 value) {
  return static_cast<short>(value.data);
}

// Probe-only grouped-N RDNA WMMA kernel for the PaliGemma2 ViT MLP down GEMM:
//
//   out[rows, model_dim] = up_bf16[rows, mlp] * linear_1[mlp, model_dim]
//
// The resident linear_1 weight is already uploaded in row-major B[K,N] layout
// for the rocBLAS row-major transpose trick, so it is also the natural input for
// this direct row-major WMMA kernel. Eight waves in one workgroup compute eight
// adjacent 16-column output tiles for the same 16 rows. They cooperatively stage
// one A/up tile in LDS per 16-wide K chunk, then each wave loads its own B tile.
//
// The wave count is deliberately a template parameter because this shape has a
// narrow output dimension (`model_dim=1152`) and small changes in N macro-tile
// width can move occupancy and tail-work balance. It is not used unless
// PaliGemma2VitHipOptions::use_mlp_down_wmma8 is enabled.
template <int WAVES_N, bool FUSE_BIAS_RESIDUAL>
__global__ __launch_bounds__(32 * WAVES_N) void MlpDownWmmaGroupedNKernel(
    const rocblas_bfloat16* __restrict__ up_bf16,
    const rocblas_bfloat16* __restrict__ linear_1_b_kn,
    const float* __restrict__ linear_1_bias,
    const float* __restrict__ residual, float* __restrict__ out, int rows,
    int mlp, int model_dim) {
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
  for (int k0 = 0; k0 < mlp; k0 += 16) {
    // Stage the shared A/up tile once per workgroup. All eight waves consume
    // the same rows and K chunk while producing different output columns.
    for (int idx = tid; idx < 16 * 16; idx += block_threads) {
      const int row_in_tile = idx >> 4;
      const int k_in_tile = idx & 15;
      const int row = tile_m + row_in_tile;
      const int col_k = k0 + k_in_tile;
      shared_a[row_in_tile][k_in_tile] =
          (row < rows && col_k < mlp)
              ? Bf16Bits(up_bf16[static_cast<size_t>(row) * mlp + col_k])
              : 0;
    }
    __syncthreads();

    WmmaBf16Frag a_frag;
    WmmaBf16Frag b_frag;
#pragma unroll
    for (int ele = 0; ele < 16; ++ele) {
      a_frag[ele] = shared_a[lane16][ele];
    }

    // B is unique per wave because each wave owns a different group of output
    // columns. linear_1_b_kn is resident row-major [mlp, model_dim].
#pragma unroll
    for (int ele = 0; ele < 16; ++ele) {
      const int row_k = k0 + ele;
      const int col = tile_n + lane16;
      b_frag[ele] =
          (row_k < mlp && col < model_dim)
              ? Bf16Bits(linear_1_b_kn[static_cast<size_t>(row_k) * model_dim +
                                       col])
              : 0;
    }

    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc);
    __syncthreads();
  }

#pragma unroll
  for (int ele = 0; ele < 8; ++ele) {
    const int row = tile_m + ele * 2 + row_parity;
    const int col = tile_n + lane16;
    if (row < rows && col < model_dim) {
      const size_t idx = static_cast<size_t>(row) * model_dim + col;
      float value = acc[ele];
      if constexpr (FUSE_BIAS_RESIDUAL) {
        // Preserve the old two-kernel epilogue order exactly:
        // first AddBiasKernel, then AddResidualKernel. The only change is that
        // the value is written once instead of being round-tripped through
        // global memory between three kernels.
        value = value + linear_1_bias[col];
        value = value + residual[idx];
      }
      out[idx] = value;
    }
  }
}

bool IsSupportedMlpDownWmmaWaves(int waves) {
  return waves == 2 || waves == 4 || waves == 6 || waves == 8 ||
         waves == 12 || waves == 16;
}

const char* MlpDownWmmaTileName(int waves) {
  switch (waves) {
    case 2:
      return "wmma2";
    case 4:
      return "wmma4";
    case 6:
      return "wmma6";
    case 8:
      return "wmma8";
    case 12:
      return "wmma12";
    case 16:
      return "wmma16";
    default:
      return "wmma?";
  }
}

const char* MlpDownWmmaTileName(int waves, bool fused_residual) {
  if (!fused_residual) return MlpDownWmmaTileName(waves);
  switch (waves) {
    case 2:
      return "wmma2+residual";
    case 4:
      return "wmma4+residual";
    case 6:
      return "wmma6+residual";
    case 8:
      return "wmma8+residual";
    case 12:
      return "wmma12+residual";
    case 16:
      return "wmma16+residual";
    default:
      return "wmma?+residual";
  }
}

template <int WAVES_N>
void LaunchMlpDownWmmaGroupedNTile(
    const rocblas_bfloat16* d_up_bf16,
    const rocblas_bfloat16* d_linear_1, float* d_out, int rows, int mlp,
    int model_dim) {
  const dim3 block(32 * WAVES_N);
  const dim3 grid((model_dim + 16 * WAVES_N - 1) / (16 * WAVES_N),
                  (rows + 15) / 16);
  hipLaunchKernelGGL((MlpDownWmmaGroupedNKernel<WAVES_N, false>), grid, block,
                     0, 0, d_up_bf16, d_linear_1,
                     static_cast<const float*>(nullptr),
                     static_cast<const float*>(nullptr), d_out, rows, mlp,
                     model_dim);
  HIP_CHECK(hipGetLastError());
}

template <int WAVES_N>
void LaunchMlpDownWmmaGroupedNBiasResidualTile(
    const rocblas_bfloat16* d_up_bf16,
    const rocblas_bfloat16* d_linear_1, const float* d_linear_1_bias,
    const float* d_residual, float* d_out, int rows, int mlp, int model_dim) {
  const dim3 block(32 * WAVES_N);
  const dim3 grid((model_dim + 16 * WAVES_N - 1) / (16 * WAVES_N),
                  (rows + 15) / 16);
  hipLaunchKernelGGL((MlpDownWmmaGroupedNKernel<WAVES_N, true>), grid, block,
                     0, 0, d_up_bf16, d_linear_1, d_linear_1_bias,
                     d_residual, d_out, rows, mlp, model_dim);
  HIP_CHECK(hipGetLastError());
}

void LaunchMlpDownWmmaGroupedN(const rocblas_bfloat16* d_up_bf16,
                               const rocblas_bfloat16* d_linear_1,
                               float* d_out, int rows, int mlp,
                               int model_dim, int waves) {
  switch (waves) {
    case 2:
      LaunchMlpDownWmmaGroupedNTile<2>(d_up_bf16, d_linear_1, d_out, rows,
                                       mlp, model_dim);
      return;
    case 4:
      LaunchMlpDownWmmaGroupedNTile<4>(d_up_bf16, d_linear_1, d_out, rows,
                                       mlp, model_dim);
      return;
    case 6:
      LaunchMlpDownWmmaGroupedNTile<6>(d_up_bf16, d_linear_1, d_out, rows,
                                       mlp, model_dim);
      return;
    case 8:
      LaunchMlpDownWmmaGroupedNTile<8>(d_up_bf16, d_linear_1, d_out, rows,
                                       mlp, model_dim);
      return;
    case 12:
      LaunchMlpDownWmmaGroupedNTile<12>(d_up_bf16, d_linear_1, d_out, rows,
                                        mlp, model_dim);
      return;
    case 16:
      LaunchMlpDownWmmaGroupedNTile<16>(d_up_bf16, d_linear_1, d_out, rows,
                                        mlp, model_dim);
      return;
    default:
      std::fprintf(stderr, "Unsupported MLP-down WMMA wave count: %d\n",
                   waves);
      std::exit(1);
  }
}

void LaunchMlpDownWmmaGroupedNBiasResidual(
    const rocblas_bfloat16* d_up_bf16,
    const rocblas_bfloat16* d_linear_1, const float* d_linear_1_bias,
    const float* d_residual, float* d_out, int rows, int mlp, int model_dim,
    int waves) {
  switch (waves) {
    case 2:
      LaunchMlpDownWmmaGroupedNBiasResidualTile<2>(
          d_up_bf16, d_linear_1, d_linear_1_bias, d_residual, d_out, rows, mlp,
          model_dim);
      return;
    case 4:
      LaunchMlpDownWmmaGroupedNBiasResidualTile<4>(
          d_up_bf16, d_linear_1, d_linear_1_bias, d_residual, d_out, rows, mlp,
          model_dim);
      return;
    case 6:
      LaunchMlpDownWmmaGroupedNBiasResidualTile<6>(
          d_up_bf16, d_linear_1, d_linear_1_bias, d_residual, d_out, rows, mlp,
          model_dim);
      return;
    case 8:
      LaunchMlpDownWmmaGroupedNBiasResidualTile<8>(
          d_up_bf16, d_linear_1, d_linear_1_bias, d_residual, d_out, rows, mlp,
          model_dim);
      return;
    case 12:
      LaunchMlpDownWmmaGroupedNBiasResidualTile<12>(
          d_up_bf16, d_linear_1, d_linear_1_bias, d_residual, d_out, rows, mlp,
          model_dim);
      return;
    case 16:
      LaunchMlpDownWmmaGroupedNBiasResidualTile<16>(
          d_up_bf16, d_linear_1, d_linear_1_bias, d_residual, d_out, rows, mlp,
          model_dim);
      return;
    default:
      std::fprintf(stderr, "Unsupported MLP-down WMMA wave count: %d\n",
                   waves);
      std::exit(1);
  }
}

// Probe-only grouped-2D RDNA WMMA kernel for PaliGemma2 ViT BF16 projection
// GEMMs:
//
//   C[rows, out_cols] = A[rows, in_cols] * B[in_cols, out_cols]
//
// One wave still owns one 16 x 16 output tile, but a workgroup covers a
// WAVES_M x WAVES_N macro-tile. For each 16-wide K chunk the workgroup stages
// both the activation panel and the linear_0 weight panel in LDS. The measured
// best standalone schedule is 4 x 4 waves: a 512-thread workgroup computing a
// 64 x 64 output macro-tile.
//
// This is intentionally separate from MlpDownWmmaGroupedNKernel. MLP down has a
// different winning schedule on this iGPU: the simpler grouped-N A-reuse kernel
// remains faster there, while MLP up and QKV benefit from reusing both A and B.
template <int WAVES_M, int WAVES_N>
__global__ __launch_bounds__(32 * WAVES_M * WAVES_N)
    void Bf16GemmWmma2DKernel(const rocblas_bfloat16* __restrict__ a,
                              const rocblas_bfloat16* __restrict__ b_kn,
                              float* __restrict__ c, int rows, int in_cols,
                              int out_cols) {
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
  for (int k0 = 0; k0 < in_cols; k0 += 16) {
    // Stage the activation side of the macro-tile with contiguous row-major
    // loads. This avoids the high-stride per-wave A traffic that made the
    // grouped-M-only MLP-up prototype slower than expected.
    for (int idx = tid; idx < 16 * WAVES_M * 16; idx += block_threads) {
      const int row_in_macro = idx >> 4;
      const int k_in_tile = idx & 15;
      const int row = macro_m + row_in_macro;
      const int col_k = k0 + k_in_tile;
      shared_a[row_in_macro][k_in_tile] =
          (row < rows && col_k < in_cols)
              ? Bf16Bits(a[static_cast<size_t>(row) * in_cols + col_k])
              : 0;
    }

    // Stage the weight side of the macro-tile. b_kn is the resident
    // row-major B[K,N] upload used by the rocBLAS row-major transpose trick.
    for (int idx = tid; idx < 16 * 16 * WAVES_N; idx += block_threads) {
      const int k_in_tile = idx / (16 * WAVES_N);
      const int col_in_macro = idx - k_in_tile * (16 * WAVES_N);
      const int row_k = k0 + k_in_tile;
      const int col = macro_n + col_in_macro;
      shared_b[k_in_tile][col_in_macro] =
          (row_k < in_cols && col < out_cols)
              ? Bf16Bits(b_kn[static_cast<size_t>(row_k) * out_cols + col])
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
    if (row < rows && col < out_cols) {
      c[static_cast<size_t>(row) * out_cols + col] = acc[ele];
    }
  }
}

template <int WAVES_M, int WAVES_N>
__global__ __launch_bounds__(32 * WAVES_M * WAVES_N)
    void Bf16GemmWmma2DBiasKernel(const rocblas_bfloat16* __restrict__ a,
                                  const rocblas_bfloat16* __restrict__ b_kn,
                                  const float* __restrict__ bias,
                                  float* __restrict__ c, int rows,
                                  int in_cols, int out_cols) {
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
  for (int k0 = 0; k0 < in_cols; k0 += 16) {
    for (int idx = tid; idx < 16 * WAVES_M * 16; idx += block_threads) {
      const int row_in_macro = idx >> 4;
      const int k_in_tile = idx & 15;
      const int row = macro_m + row_in_macro;
      const int col_k = k0 + k_in_tile;
      shared_a[row_in_macro][k_in_tile] =
          (row < rows && col_k < in_cols)
              ? Bf16Bits(a[static_cast<size_t>(row) * in_cols + col_k])
              : 0;
    }
    for (int idx = tid; idx < 16 * 16 * WAVES_N; idx += block_threads) {
      const int k_in_tile = idx / (16 * WAVES_N);
      const int col_in_macro = idx - k_in_tile * (16 * WAVES_N);
      const int row_k = k0 + k_in_tile;
      const int col = macro_n + col_in_macro;
      shared_b[k_in_tile][col_in_macro] =
          (row_k < in_cols && col < out_cols)
              ? Bf16Bits(b_kn[static_cast<size_t>(row_k) * out_cols + col])
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
    if (row < rows && col < out_cols) {
      c[static_cast<size_t>(row) * out_cols + col] = acc[ele] + bias[col];
    }
  }
}

template <int WAVES_M, int WAVES_N>
__global__ __launch_bounds__(32 * WAVES_M * WAVES_N)
    void Bf16GemmWmma2DBiasGeluBF16Kernel(
        const rocblas_bfloat16* __restrict__ a,
        const rocblas_bfloat16* __restrict__ b_kn,
        const float* __restrict__ bias, rocblas_bfloat16* __restrict__ c,
        int rows, int in_cols, int out_cols) {
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
  for (int k0 = 0; k0 < in_cols; k0 += 16) {
    for (int idx = tid; idx < 16 * WAVES_M * 16; idx += block_threads) {
      const int row_in_macro = idx >> 4;
      const int k_in_tile = idx & 15;
      const int row = macro_m + row_in_macro;
      const int col_k = k0 + k_in_tile;
      shared_a[row_in_macro][k_in_tile] =
          (row < rows && col_k < in_cols)
              ? Bf16Bits(a[static_cast<size_t>(row) * in_cols + col_k])
              : 0;
    }
    for (int idx = tid; idx < 16 * 16 * WAVES_N; idx += block_threads) {
      const int k_in_tile = idx / (16 * WAVES_N);
      const int col_in_macro = idx - k_in_tile * (16 * WAVES_N);
      const int row_k = k0 + k_in_tile;
      const int col = macro_n + col_in_macro;
      shared_b[k_in_tile][col_in_macro] =
          (row_k < in_cols && col < out_cols)
              ? Bf16Bits(b_kn[static_cast<size_t>(row_k) * out_cols + col])
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
    if (row < rows && col < out_cols) {
      const rocblas_bfloat16 biased = rocblas_bfloat16(acc[ele] + bias[col]);
      c[static_cast<size_t>(row) * out_cols + col] =
          rocblas_bfloat16(GeluApprox(static_cast<float>(biased)));
    }
  }
}

const char* MlpUpWmma2DTileName(int rows) {
  return rows == 256 ? "wmma2d4x8" : "wmma2d8x4";
}

template <int WAVES_M, int WAVES_N>
void LaunchMlpUpWmma2DBiasGeluBF16Tile(
    const rocblas_bfloat16* d_pre_ffw_bf16,
    const rocblas_bfloat16* d_linear_0, const float* d_linear_0_bias,
    rocblas_bfloat16* d_up_bf16, int rows, int model_dim, int mlp) {
  const dim3 block(32 * WAVES_M * WAVES_N);
  const dim3 grid((mlp + 16 * WAVES_N - 1) / (16 * WAVES_N),
                  (rows + 16 * WAVES_M - 1) / (16 * WAVES_M));
  hipLaunchKernelGGL((Bf16GemmWmma2DBiasGeluBF16Kernel<WAVES_M, WAVES_N>),
                     grid, block, 0, 0, d_pre_ffw_bf16, d_linear_0,
                     d_linear_0_bias, d_up_bf16, rows, model_dim, mlp);
  HIP_CHECK(hipGetLastError());
}

void LaunchMlpUpWmma2DBiasGeluBF16(
    const rocblas_bfloat16* d_pre_ffw_bf16,
    const rocblas_bfloat16* d_linear_0, const float* d_linear_0_bias,
    rocblas_bfloat16* d_up_bf16, int rows, int model_dim, int mlp) {
  // Current gfx1150 microbench data is shape-sensitive: the 224px MLP-up shape
  // prefers a wider N macro-tile, while the 448px shape still prefers the
  // existing taller 8x4 tile. Both variants preserve the same F32 accumulate,
  // BF16 bias boundary, GELU, and final BF16 store contract.
  if (rows == 256) {
    LaunchMlpUpWmma2DBiasGeluBF16Tile<4, 8>(
        d_pre_ffw_bf16, d_linear_0, d_linear_0_bias, d_up_bf16, rows,
        model_dim, mlp);
    return;
  }
  LaunchMlpUpWmma2DBiasGeluBF16Tile<8, 4>(
      d_pre_ffw_bf16, d_linear_0, d_linear_0_bias, d_up_bf16, rows, model_dim,
      mlp);
}

template <int WAVES_M, int WAVES_N>
void LaunchMlpUpWmma2DPlainTile(const rocblas_bfloat16* d_pre_ffw_bf16,
                                const rocblas_bfloat16* d_linear_0,
                                float* d_up_f32, int rows, int model_dim,
                                int mlp) {
  const dim3 block(32 * WAVES_M * WAVES_N);
  const dim3 grid((mlp + 16 * WAVES_N - 1) / (16 * WAVES_N),
                  (rows + 16 * WAVES_M - 1) / (16 * WAVES_M));
  hipLaunchKernelGGL((Bf16GemmWmma2DKernel<WAVES_M, WAVES_N>), grid, block, 0,
                     0, d_pre_ffw_bf16, d_linear_0, d_up_f32, rows, model_dim,
                     mlp);
  HIP_CHECK(hipGetLastError());
}

void LaunchMlpUpWmma2DPlain(const rocblas_bfloat16* d_pre_ffw_bf16,
                            const rocblas_bfloat16* d_linear_0,
                            float* d_up_f32, int rows, int model_dim,
                            int mlp) {
  if (rows == 256) {
    LaunchMlpUpWmma2DPlainTile<4, 8>(
        d_pre_ffw_bf16, d_linear_0, d_up_f32, rows, model_dim, mlp);
    return;
  }
  LaunchMlpUpWmma2DPlainTile<8, 4>(
      d_pre_ffw_bf16, d_linear_0, d_up_f32, rows, model_dim, mlp);
}

template <int WAVES_M, int WAVES_N>
__global__ __launch_bounds__(32 * WAVES_M * WAVES_N)
    void Bf16GemmWmma2DBatchedKernel(
        const rocblas_bfloat16* __restrict__ a,
        const rocblas_bfloat16* __restrict__ b, float* __restrict__ c, int rows,
        int in_cols, int out_cols, size_t stride_a, size_t stride_b,
        size_t stride_c) {
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
  for (int k0 = 0; k0 < in_cols; k0 += 16) {
    for (int idx = tid; idx < 16 * WAVES_M * 16; idx += block_threads) {
      const int row_in_macro = idx >> 4;
      const int k_in_tile = idx & 15;
      const int row = macro_m + row_in_macro;
      const int col_k = k0 + k_in_tile;
      shared_a[row_in_macro][k_in_tile] =
          (row < rows && col_k < in_cols)
              ? Bf16Bits(batch_a[static_cast<size_t>(row) * in_cols + col_k])
              : 0;
    }
    for (int idx = tid; idx < 16 * 16 * WAVES_N; idx += block_threads) {
      const int k_in_tile = idx / (16 * WAVES_N);
      const int col_in_macro = idx - k_in_tile * (16 * WAVES_N);
      const int row_k = k0 + k_in_tile;
      const int col = macro_n + col_in_macro;
      shared_b[k_in_tile][col_in_macro] =
          (row_k < in_cols && col < out_cols)
              ? Bf16Bits(batch_b[static_cast<size_t>(row_k) * out_cols + col])
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
    if (row < rows && col < out_cols) {
      batch_c[static_cast<size_t>(row) * out_cols + col] = acc[ele];
    }
  }
}

template <int WAVES_M, int WAVES_N>
__global__ __launch_bounds__(32 * WAVES_M * WAVES_N)
    void Bf16GemmWmma2DTransBScaleBatchedKernel(
        const rocblas_bfloat16* __restrict__ a,
        const rocblas_bfloat16* __restrict__ b, float* __restrict__ c, int rows,
        int in_cols, int out_cols, size_t stride_a, size_t stride_b,
        size_t stride_c, float alpha) {
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
  for (int k0 = 0; k0 < in_cols; k0 += 16) {
    for (int idx = tid; idx < 16 * WAVES_M * 16; idx += block_threads) {
      const int row_in_macro = idx >> 4;
      const int k_in_tile = idx & 15;
      const int row = macro_m + row_in_macro;
      const int col_k = k0 + k_in_tile;
      shared_a[row_in_macro][k_in_tile] =
          (row < rows && col_k < in_cols)
              ? Bf16Bits(batch_a[static_cast<size_t>(row) * in_cols + col_k])
              : 0;
    }
    for (int idx = tid; idx < 16 * 16 * WAVES_N; idx += block_threads) {
      const int k_in_tile = idx / (16 * WAVES_N);
      const int col_in_macro = idx - k_in_tile * (16 * WAVES_N);
      const int row_k = k0 + k_in_tile;
      const int col = macro_n + col_in_macro;
      // Attention QK is C[query,key] = Q[query,dim] * K[key,dim]^T. Q and K
      // stay in their natural head-major row-major layout, so this variant
      // stages B by reading K[col, row_k] instead of B[row_k, col].
      shared_b[k_in_tile][col_in_macro] =
          (row_k < in_cols && col < out_cols)
              ? Bf16Bits(batch_b[static_cast<size_t>(col) * in_cols + row_k])
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
    if (row < rows && col < out_cols) {
      batch_c[static_cast<size_t>(row) * out_cols + col] = acc[ele] * alpha;
    }
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

  // PaliGemma2 uses qkv_dim=72, and every Q/K/V segment begins on a 4-float
  // boundary. Loading and storing four adjacent dimensions per thread preserves
  // the exact F32 attention tensor layout while reducing per-element index math
  // and memory instructions in the split phase.
  const float4 q_vec = *reinterpret_cast<const float4*>(qkv + qkv_base);
  const float4 k_vec =
      *reinterpret_cast<const float4*>(qkv + qkv_base + qkv_dim);
  const float4 v_vec =
      *reinterpret_cast<const float4*>(qkv + qkv_base + 2 * qkv_dim);
  *reinterpret_cast<float4*>(q + out_base) = q_vec;
  *reinterpret_cast<float4*>(k + out_base) = k_vec;
  *reinterpret_cast<float4*>(v + out_base) = v_vec;
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

  // QK-only BF16 keeps the high-volume attention score matrix and V path in
  // F32. That avoids the extra score-quantization kernel required by a full
  // BF16 QK/AV path, while still letting rocBLAS use BF16 input math for the
  // score GEMM that dominates the 448px attention phase profile.
  q[idx] = rocblas_bfloat16(qkv[qkv_base]);
  k[idx] = rocblas_bfloat16(qkv[qkv_base + qkv_dim]);
  v[idx] = qkv[qkv_base + 2 * qkv_dim];
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

template <int WAVES_M, int WAVES_N>
__global__ __launch_bounds__(32 * WAVES_M * WAVES_N)
    void Bf16GemmWmma2DBiasResidualKernel(
        const rocblas_bfloat16* __restrict__ a,
        const rocblas_bfloat16* __restrict__ b_kn,
        const float* __restrict__ bias, const float* __restrict__ residual,
        float* __restrict__ c, int rows, int in_cols, int out_cols) {
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
  for (int k0 = 0; k0 < in_cols; k0 += 16) {
    for (int idx = tid; idx < 16 * WAVES_M * 16; idx += block_threads) {
      const int row_in_macro = idx >> 4;
      const int k_in_tile = idx & 15;
      const int row = macro_m + row_in_macro;
      const int col_k = k0 + k_in_tile;
      shared_a[row_in_macro][k_in_tile] =
          (row < rows && col_k < in_cols)
              ? Bf16Bits(a[static_cast<size_t>(row) * in_cols + col_k])
              : 0;
    }
    for (int idx = tid; idx < 16 * 16 * WAVES_N; idx += block_threads) {
      const int k_in_tile = idx / (16 * WAVES_N);
      const int col_in_macro = idx - k_in_tile * (16 * WAVES_N);
      const int row_k = k0 + k_in_tile;
      const int col = macro_n + col_in_macro;
      shared_b[k_in_tile][col_in_macro] =
          (row_k < in_cols && col < out_cols)
              ? Bf16Bits(b_kn[static_cast<size_t>(row_k) * out_cols + col])
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
    if (row < rows && col < out_cols) {
      const size_t idx = static_cast<size_t>(row) * out_cols + col;
      c[idx] = acc[ele] + bias[col] + residual[idx];
    }
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

  // F32-preserving AV microkernel for the 224px PaliGemma2 head shape:
  //   head_major[row, dim] = sum_key softmax[row, key] * V[key, dim].
  // One workgroup owns BM query rows and all 72 output dimensions for one head.
  // Each thread accumulates four adjacent dimensions, which reuses the same
  // staged F32 score across a float4 V load without changing the softmax/AV
  // precision boundary.
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

  // Instead of normalizing the large [head, row, col] score matrix in place,
  // this variant stores the per-row reciprocal denominator. The following AV
  // GEMM consumes unnormalized exp() scores, and the much smaller attention
  // output is scaled while it is packed for attn_out.
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

  // Benchmark data showed that a separate F32-score-to-BF16 quantization pass
  // erases the AV WMMA benefit at 448px. This direct path keeps the reduction
  // in F32, then intentionally stores the normalized score boundary as BF16.
  for (int col = tid; col < rows; col += block_threads) {
    row_scores_bf16[col] =
        rocblas_bfloat16(expf(row_scores[col] - max_score) * inv_sum);
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

  // The BF16-only ViT layer immediately converts the packed attention output to
  // BF16 before attn_out. Fusing pack+conversion preserves the F32 QK/softmax/AV
  // attention core and only removes a row-major F32 temporary plus one kernel.
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

  // PaliGemma2 uses qkv_dim=72, so every head-major row is 288 bytes. Combined
  // with hipMalloc's aligned base pointer, the float4 load is naturally aligned
  // while preserving the exact scalar BF16 conversion boundary.
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

  // Paired with SoftmaxRowsStoreExpAndInvSumKernel: AV produces an unnormalized
  // weighted sum, so the saved reciprocal denominator is applied at the same
  // boundary where the normal pack-to-BF16 path already converts for attn_out.
  const float scale = inv_sums[static_cast<size_t>(head) * rows + row];
  row_major_bf16[row * out_cols + head * qkv_dim + dim] =
      rocblas_bfloat16(head_major[idx] * scale);
}

// Row-wise ViT layernorm for one [rows, model_dim] activation matrix. Each HIP
// block owns one token row; threads reduce sum and sum of squares across the
// model dimension, then write the normalized row back as F32.
//
// The CPU reference accumulates in double. This kernel intentionally accumulates
// in F32 because it is the intended device-resident inference path; validation
// below compares it against the host double-accumulating reference so any
// unacceptable drift is visible before this becomes a real inference backend.
__global__ void LayerNormKernel(const float* x, const float* scale,
                                const float* bias, float* out, int rows,
                                int cols) {
  extern __shared__ float shared[];
  float* sum_shared = shared;
  float* sum2_shared = shared + blockDim.x;
  const int row = blockIdx.x;
  if (row >= rows) return;

  const int tid = threadIdx.x;
  const int block_threads = blockDim.x;
  const float* src = x + static_cast<size_t>(row) * cols;
  // Stride through the row so the same kernel handles 1152-wide PaliGemma2
  // rows without assuming cols <= blockDim.x.
  float local_sum = 0.0f;
  float local_sum2 = 0.0f;
  for (int col = tid; col < cols; col += block_threads) {
    const float value = src[col];
    local_sum += value;
    local_sum2 += value * value;
  }
  sum_shared[tid] = local_sum;
  sum2_shared[tid] = local_sum2;
  __syncthreads();

  // Shared-memory tree reduction keeps one row's normalization statistics
  // inside the block. This is simple and sufficient for the fixed ViT width;
  // later tuning can replace it with wave reductions if LN becomes visible.
  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum_shared[tid] += sum_shared[tid + stride];
      sum2_shared[tid] += sum2_shared[tid + stride];
    }
    __syncthreads();
  }

  const float inv_cols = 1.0f / static_cast<float>(cols);
  const float mean = sum_shared[0] * inv_cols;
  const float mean2 = sum2_shared[0] * inv_cols;
  const float variance = fmaxf(mean2 - mean * mean, 0.0f);
  const float inv_std = rsqrtf(variance + 1E-6f);
  float* dst = out + static_cast<size_t>(row) * cols;
  // Scale and bias are uploaded as F32 vectors because the gemma.cpp MatPtr
  // storage may vary by conversion mode, but the ViT layernorm math consumes
  // them as ordinary float parameters.
  for (int col = tid; col < cols; col += block_threads) {
    dst[col] = (src[col] - mean) * scale[col] * inv_std + bias[col];
  }
}

// BF16 inference variant of LayerNormKernel. It intentionally copies the same
// row statistics and F32 normalization math, then rounds only the final
// per-element result to BF16 at the store site. This preserves the current
// activation boundary seen by the following resident BF16 GEMMs while removing
// a full F32 output write/read and a separate F32ToBF16 kernel launch.
__global__ void LayerNormToBF16Kernel(const float* x, const float* scale,
                                      const float* bias,
                                      rocblas_bfloat16* out, int rows,
                                      int cols) {
  extern __shared__ float shared[];
  float* sum_shared = shared;
  float* sum2_shared = shared + blockDim.x;
  const int row = blockIdx.x;
  if (row >= rows) return;

  const int tid = threadIdx.x;
  const int block_threads = blockDim.x;
  const float* src = x + static_cast<size_t>(row) * cols;
  float local_sum = 0.0f;
  float local_sum2 = 0.0f;
  for (int col = tid; col < cols; col += block_threads) {
    const float value = src[col];
    local_sum += value;
    local_sum2 += value * value;
  }
  sum_shared[tid] = local_sum;
  sum2_shared[tid] = local_sum2;
  __syncthreads();

  for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum_shared[tid] += sum_shared[tid + stride];
      sum2_shared[tid] += sum2_shared[tid + stride];
    }
    __syncthreads();
  }

  const float inv_cols = 1.0f / static_cast<float>(cols);
  const float mean = sum_shared[0] * inv_cols;
  const float mean2 = sum2_shared[0] * inv_cols;
  const float variance = fmaxf(mean2 - mean * mean, 0.0f);
  const float inv_std = rsqrtf(variance + 1E-6f);
  rocblas_bfloat16* dst = out + static_cast<size_t>(row) * cols;
  for (int col = tid; col < cols; col += block_threads) {
    const float normalized = (src[col] - mean) * scale[col] * inv_std +
                             bias[col];
    dst[col] = rocblas_bfloat16(normalized);
  }
}

double GiB(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

double MiB(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

float BF16ToF32(const BF16 value) { return hwy::F32FromBF16(value); }

float GeluHost(float x) {
  return x * (0.5f + 0.5f * std::tanh(x * (0.797884560804236f +
                                           0.03567740813636141f * x * x)));
}

size_t DenseBF16Bytes(const MatPtr& mat) {
  return mat.Rows() * mat.Cols() * sizeof(rocblas_bfloat16);
}

std::vector<rocblas_bfloat16> F32ToRocblasBF16(
    const std::vector<float>& input) {
  std::vector<rocblas_bfloat16> out(input.size());
  static_assert(sizeof(BF16) == sizeof(rocblas_bfloat16),
                "BF16 storage size must match rocBLAS BF16");
  for (size_t i = 0; i < input.size(); ++i) {
    const BF16 value = hwy::BF16FromF32(input[i]);
    std::memcpy(out.data() + i, &value, sizeof(value));
  }
  return out;
}

std::vector<float> CopyActivationMatToF32(const MatPtr& mat) {
  std::vector<float> out(mat.Rows() * mat.Cols());
  CallUpcastedActivation(&mat, [&](const auto* mat_t) {
    using T = typename std::remove_cv<
        typename std::remove_reference<decltype(*mat_t->Row(0))>::type>::type;
    for (size_t row = 0; row < mat_t->Rows(); ++row) {
      const T* src = mat_t->Row(row);
      float* dst = out.data() + row * mat_t->Cols();
      for (size_t col = 0; col < mat_t->Cols(); ++col) {
        if constexpr (hwy::IsSame<T, BF16>()) {
          dst[col] = BF16ToF32(src[col]);
        } else {
          dst[col] = src[col];
        }
      }
    }
  });
  return out;
}

std::vector<float> BuildImagePatches(const Image& image, int rows, int patch) {
  std::vector<float> patches(static_cast<size_t>(rows) * patch);
  for (int row = 0; row < rows; ++row) {
    image.GetPatch(static_cast<size_t>(row),
                   patches.data() + static_cast<size_t>(row) * patch);
  }
  return patches;
}

std::vector<float> BuildPatchEmbeddingReference(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const std::vector<float>& patches) {
  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  const int cols = static_cast<int>(model_config.vit_config.model_dim);
  const float* bias = weights.vit_img_embedding_bias.PackedScale1();
  const std::vector<float> pos =
      CopyActivationMatToF32(weights.vit_img_pos_embedding);
  const MatPtrT<BF16> weight(weights.vit_img_embedding_kernel);

  std::vector<float> reference(static_cast<size_t>(rows) * cols);
  for (int row = 0; row < rows; ++row) {
    const float* patch_row = patches.data() + static_cast<size_t>(row) * patch;
    for (int col = 0; col < cols; ++col) {
      const BF16* weight_row = weight.Row(static_cast<size_t>(col));
      float value = bias[col];
      for (int k = 0; k < patch; ++k) {
        value += patch_row[k] * BF16ToF32(weight_row[k]);
      }
      value += pos[static_cast<size_t>(row) * cols + col];
      reference[static_cast<size_t>(row) * cols + col] = value;
    }
  }
  return reference;
}

std::vector<float> LayerNormReference(const std::vector<float>& x, int rows,
                                      int cols, const MatPtr& scale,
                                      const MatPtr& bias) {
  const std::vector<float> scale_f32 = CopyActivationMatToF32(scale);
  const std::vector<float> bias_f32 = CopyActivationMatToF32(bias);
  HWY_ASSERT(scale_f32.size() == static_cast<size_t>(cols));
  HWY_ASSERT(bias_f32.size() == static_cast<size_t>(cols));

  std::vector<float> out(static_cast<size_t>(rows) * cols);
  for (int row = 0; row < rows; ++row) {
    const float* src = x.data() + static_cast<size_t>(row) * cols;
    double sum = 0.0;
    double sum2 = 0.0;
    for (int col = 0; col < cols; ++col) {
      sum += src[col];
      sum2 += static_cast<double>(src[col]) * src[col];
    }
    const double mu = sum / cols;
    const double mu2 = sum2 / cols;
    const double var = HWY_MAX(mu2 - mu * mu, 0.0);
    const float inv_std = static_cast<float>(1.0 / std::sqrt(var + 1E-6));

    float* dst = out.data() + static_cast<size_t>(row) * cols;
    for (int col = 0; col < cols; ++col) {
      dst[col] = (src[col] - static_cast<float>(mu)) * scale_f32[col] *
                     inv_std +
                 bias_f32[col];
    }
  }
  return out;
}

ErrorStats ComputeError(const std::vector<float>& actual,
                        const std::vector<float>& reference) {
  HWY_ASSERT(actual.size() == reference.size());
  ErrorStats stats;
  double squared_error = 0.0;
  for (size_t i = 0; i < reference.size(); ++i) {
    const float error = std::abs(actual[i] - reference[i]);
    stats.max_abs = HWY_MAX(stats.max_abs, error);
    squared_error += static_cast<double>(error) * error;
  }
  stats.rms =
      static_cast<float>(std::sqrt(squared_error / reference.size()));
  return stats;
}

std::vector<float> CopyImageTokensToVector(const ImageTokens& image_tokens) {
  std::vector<float> out(image_tokens.Rows() * image_tokens.Cols());
  for (size_t row = 0; row < image_tokens.Rows(); ++row) {
    const float* src = image_tokens.Row(row);
    float* dst = out.data() + row * image_tokens.Cols();
    std::memcpy(dst, src, image_tokens.Cols() * sizeof(float));
  }
  return out;
}

void CopyVectorToImageTokens(const std::vector<float>& values,
                             ImageTokens& image_tokens) {
  HWY_ASSERT(values.size() == image_tokens.Rows() * image_tokens.Cols());
  for (size_t row = 0; row < image_tokens.Rows(); ++row) {
    const float* src = values.data() + row * image_tokens.Cols();
    float* dst = image_tokens.Row(row);
    std::memcpy(dst, src, image_tokens.Cols() * sizeof(float));
  }
}

std::vector<float> ComputeAttentionWeightedSumReference(
    const std::vector<float>& qkv, int rows, int heads, int qkv_dim) {
  // Scalar mirror of gemma/vit.cc::DotSoftmaxWeightedSum. Q/K/V are laid out as
  // [row, head, q|k|v, qkv_dim] inside the contiguous QKV projection output.
  // The result is [row, head, qkv_dim], which is the input to attn_out_w.
  const int qkv_cols = heads * 3 * qkv_dim;
  const int out_cols = heads * qkv_dim;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  std::vector<float> out(static_cast<size_t>(rows) * out_cols, 0.0f);
  std::vector<float> scores(static_cast<size_t>(rows));

  for (int token = 0; token < rows; ++token) {
    const float* token_qkv = qkv.data() + static_cast<size_t>(token) * qkv_cols;
    for (int head = 0; head < heads; ++head) {
      const float* q = token_qkv + head * 3 * qkv_dim;
      float max_score = -INFINITY;
      for (int i = 0; i < rows; ++i) {
        const float* k = qkv.data() + static_cast<size_t>(i) * qkv_cols +
                         head * 3 * qkv_dim + qkv_dim;
        float score = 0.0f;
        for (int dim = 0; dim < qkv_dim; ++dim) {
          score += q[dim] * k[dim];
        }
        score *= query_scale;
        scores[static_cast<size_t>(i)] = score;
        max_score = HWY_MAX(max_score, score);
      }

      double sum = 0.0;
      for (int i = 0; i < rows; ++i) {
        const float shifted = scores[static_cast<size_t>(i)] - max_score;
        const float prob = std::exp(shifted);
        scores[static_cast<size_t>(i)] = prob;
        sum += prob;
      }
      const float inv_sum = static_cast<float>(1.0 / sum);

      float* dst = out.data() + static_cast<size_t>(token) * out_cols +
                   head * qkv_dim;
      for (int i = 0; i < rows; ++i) {
        const float prob = scores[static_cast<size_t>(i)] * inv_sum;
        const float* v = qkv.data() + static_cast<size_t>(i) * qkv_cols +
                         head * 3 * qkv_dim + 2 * qkv_dim;
        for (int dim = 0; dim < qkv_dim; ++dim) {
          dst[dim] += prob * v[dim];
        }
      }
    }
  }

  return out;
}

double ShapeFlops(const Shape& shape) {
  return 2.0 * static_cast<double>(shape.m) * shape.k * shape.n;
}

PaliGemma2VitHipTiming PrintTiming(const char* name, std::vector<float> ms,
                                   double flops) {
  std::sort(ms.begin(), ms.end());
  const float best_ms = ms.front();
  const float median_ms = ms[ms.size() / 2];
  const double gflops = flops / (median_ms * 1.0E6);
  std::printf("%-32s best=%8.3f ms median=%8.3f ms %8.1f GFLOPS\n", name,
              best_ms, median_ms, gflops);
  return PaliGemma2VitHipTiming{
      .best_ms = best_ms,
      .median_ms = median_ms,
      .gflops = gflops,
  };
}

void PrintMat(const char* label, const MatPtr& mat) {
  std::printf("  %-24s type=%-5s rows=%5zu cols=%5zu packed=%d "
              "dense_bf16=%8.2f MiB\n",
              label, TypeName(mat.GetType()), mat.Rows(), mat.Cols(),
              static_cast<int>(mat.IsPacked()), MiB(DenseBF16Bytes(mat)));
}

ResidentBF16Mat UploadBF16MatForRocblasB(const char* label,
                                         const MatPtr& mat, bool verbose) {
  if (mat.GetType() != Type::kBF16) {
    std::fprintf(stderr, "Cannot upload %s: expected bf16, got %s\n", label,
                 TypeName(mat.GetType()));
    std::exit(1);
  }

  const MatPtrT<BF16> bf16(mat);
  const size_t rows_n = bf16.Rows();
  const size_t cols_k = bf16.Cols();
  const size_t count = rows_n * cols_k;
  std::vector<rocblas_bfloat16> packed(count);
  static_assert(sizeof(BF16) == sizeof(rocblas_bfloat16),
                "BF16 storage size must match rocBLAS BF16");

  // gemma.cpp stores projection weights as W[N,K], where N is the output
  // dimension and K is the input dimension. The rocBLAS row-major mapping below
  // calls column-major GEMM with swapped A/B operands, so it expects the weight
  // operand as B[K,N]. Transpose once during upload so every timed GEMM can use
  // the resident layout directly.
  for (size_t n = 0; n < rows_n; ++n) {
    const BF16* row = bf16.Row(n);
    for (size_t k = 0; k < cols_k; ++k) {
      std::memcpy(packed.data() + k * rows_n + n, row + k,
                  sizeof(rocblas_bfloat16));
    }
  }

  auto device = std::make_unique<DeviceBuffer<rocblas_bfloat16>>(count);
  HIP_CHECK(hipMemcpy(device->ptr, packed.data(),
                      count * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));
  if (verbose) {
    std::printf("  uploaded %-20s %8.2f MiB as B[K,N]\n", label,
                MiB(count * sizeof(rocblas_bfloat16)));
  }

  return ResidentBF16Mat{
      .label = label,
      .k = static_cast<int>(cols_k),
      .n = static_cast<int>(rows_n),
      .device = std::move(device),
  };
}

ResidentF32Mat UploadBF16MatAsF32RocblasB(const char* label,
                                          const MatPtr& mat, bool verbose) {
  if (mat.GetType() != Type::kBF16) {
    std::fprintf(stderr, "Cannot upload %s: expected bf16, got %s\n", label,
                 TypeName(mat.GetType()));
    std::exit(1);
  }

  const MatPtrT<BF16> bf16(mat);
  const size_t rows_n = bf16.Rows();
  const size_t cols_k = bf16.Cols();
  const size_t count = rows_n * cols_k;
  std::vector<float> packed(count);

  // This is a F32 shadow of the BF16 weight matrix. rocBLAS does not support
  // F32 activations multiplied by BF16 weights in one GEMM, whereas gemma.cpp's
  // CPU MatMul does use F32 image patches with BF16 weights. The F32 shadow
  // keeps the first correctness boundary close to CPU semantics.
  for (size_t n = 0; n < rows_n; ++n) {
    const BF16* row = bf16.Row(n);
    for (size_t k = 0; k < cols_k; ++k) {
      packed[k * rows_n + n] = hwy::F32FromBF16(row[k]);
    }
  }

  auto device = std::make_unique<DeviceBuffer<float>>(count);
  HIP_CHECK(hipMemcpy(device->ptr, packed.data(), count * sizeof(float),
                      hipMemcpyHostToDevice));
  if (verbose) {
    std::printf("  uploaded %-20s %8.2f MiB as F32 B[K,N]\n", label,
                MiB(count * sizeof(float)));
  }

  return ResidentF32Mat{
      .label = label,
      .k = static_cast<int>(cols_k),
      .n = static_cast<int>(rows_n),
      .device = std::move(device),
  };
}

ResidentF32Vec UploadF32Vec(const char* label, const float* values,
                            size_t count, bool verbose) {
  auto device = std::make_unique<DeviceBuffer<float>>(count);
  HIP_CHECK(hipMemcpy(device->ptr, values, count * sizeof(float),
                      hipMemcpyHostToDevice));
  if (verbose) {
    std::printf("  uploaded %-20s %8.3f KiB as F32 vec\n", label,
                static_cast<double>(count * sizeof(float)) / 1024.0);
  }
  return ResidentF32Vec{
      .label = label,
      .count = static_cast<int>(count),
      .device = std::move(device),
  };
}

ResidentF32Vec UploadF32Vec(const char* label,
                            const std::vector<float>& values, bool verbose) {
  return UploadF32Vec(label, values.data(), values.size(), verbose);
}

std::vector<float> CopyPosEmbeddingToF32(const MatPtr& pos_embedding) {
  // Position embeddings may be F32 or BF16 depending on conversion mode. The
  // validation path compares GPU outputs against a F32 scalar reference, so
  // normalize the small positional tensor to F32 on the host once.
  std::vector<float> out(pos_embedding.Rows() * pos_embedding.Cols());
  CallUpcastedActivation(&pos_embedding, [&](const auto* pos_t) {
    using T = typename std::remove_cv<
        typename std::remove_reference<decltype(*pos_t->Row(0))>::type>::type;
    for (size_t row = 0; row < pos_t->Rows(); ++row) {
      const T* src = pos_t->Row(row);
      float* dst = out.data() + row * pos_t->Cols();
      for (size_t col = 0; col < pos_t->Cols(); ++col) {
        if constexpr (hwy::IsSame<T, BF16>()) {
          dst[col] = hwy::F32FromBF16(src[col]);
        } else {
          dst[col] = src[col];
        }
      }
    }
  });
  return out;
}

float TimeSamples(const PaliGemma2VitHipOptions& options,
                  const std::function<void()>& run) {
  // All timings use the same structure: warm the rocBLAS solution/cache first,
  // then measure with HIP events so host-side scheduling overhead is excluded.
  for (int i = 0; i < options.warmup; ++i) {
    for (int j = 0; j < options.iters; ++j) run();
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  std::vector<float> ms;
  ms.reserve(options.samples);
  for (int sample = 0; sample < options.samples; ++sample) {
    HIP_CHECK(hipEventRecord(start));
    for (int iter = 0; iter < options.iters; ++iter) {
      run();
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    ms.push_back(elapsed_ms / static_cast<float>(options.iters));
  }
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

float TimeHostSamples(const PaliGemma2VitHipOptions& options,
                      const std::function<void()>& run) {
  for (int i = 0; i < options.warmup; ++i) {
    for (int j = 0; j < options.iters; ++j) run();
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> ms;
  ms.reserve(options.samples);
  for (int sample = 0; sample < options.samples; ++sample) {
    const double start = hwy::platform::Now();
    for (int iter = 0; iter < options.iters; ++iter) {
      run();
    }
    HIP_CHECK(hipDeviceSynchronize());
    const double elapsed = hwy::platform::Now() - start;
    ms.push_back(static_cast<float>(elapsed * 1000.0 / options.iters));
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

bool TimeRocblasStatusSamples(const PaliGemma2VitHipOptions& options,
                              const std::function<rocblas_status()>& run,
                              float& median_ms) {
  for (int i = 0; i < options.warmup; ++i) {
    for (int j = 0; j < options.iters; ++j) {
      if (run() != rocblas_status_success) return false;
    }
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  std::vector<float> ms;
  ms.reserve(options.samples);
  for (int sample = 0; sample < options.samples; ++sample) {
    HIP_CHECK(hipEventRecord(start));
    for (int iter = 0; iter < options.iters; ++iter) {
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
    ms.push_back(elapsed_ms / static_cast<float>(options.iters));
  }
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  std::sort(ms.begin(), ms.end());
  median_ms = ms[ms.size() / 2];
  return true;
}

std::vector<rocblas_int> QueryGemmSolutions(
    rocblas_handle handle, rocblas_operation trans_a, rocblas_operation trans_b,
    int m, int n, int k, const float* alpha, const void* a,
    rocblas_datatype a_type, int lda, const void* b, rocblas_datatype b_type,
    int ldb, const float* beta, const void* c, rocblas_datatype c_type,
    int ldc, void* d, rocblas_datatype d_type, int ldd,
    rocblas_datatype compute_type) {
  rocblas_int list_size = 0;
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
  rocblas_status status = rocblas_gemm_ex_get_solutions(
      handle, trans_a, trans_b, m, n, k, alpha, a, a_type, lda, b, b_type, ldb,
      beta, c, c_type, ldc, d, d_type, ldd, compute_type,
      rocblas_gemm_algo_standard, rocblas_gemm_flags_none,
      /*list_array=*/nullptr, &list_size);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
  if (status != rocblas_status_success || list_size <= 0) return {};

  std::vector<rocblas_int> solutions(static_cast<size_t>(list_size));
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
  status = rocblas_gemm_ex_get_solutions(
      handle, trans_a, trans_b, m, n, k, alpha, a, a_type, lda, b, b_type, ldb,
      beta, c, c_type, ldc, d, d_type, ldd, compute_type,
      rocblas_gemm_algo_standard, rocblas_gemm_flags_none, solutions.data(),
      &list_size);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
  if (status != rocblas_status_success || list_size <= 0) return {};
  solutions.resize(static_cast<size_t>(list_size));
  return solutions;
}

rocblas_status RunBf16F32GemmSolution(rocblas_handle handle, int m, int n,
                                      int k, const void* a, int lda,
                                      const void* b, int ldb, void* d,
                                      int ldd, int solution_index) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const rocblas_gemm_algo algo =
      solution_index == 0 ? rocblas_gemm_algo_standard
                          : rocblas_gemm_algo_solution_index;
  return rocblas_gemm_ex(
      handle, rocblas_operation_none, rocblas_operation_none, m, n, k, &alpha,
      a, rocblas_datatype_bf16_r, lda, b, rocblas_datatype_bf16_r, ldb, &beta,
      d, rocblas_datatype_f32_r, ldd, d, rocblas_datatype_f32_r, ldd,
      rocblas_datatype_f32_r, algo, solution_index, rocblas_gemm_flags_none);
}

rocblas_gemm_algo RocblasAlgoForSolutionIndex(int solution_index) {
  return solution_index == 0 ? rocblas_gemm_algo_standard
                             : rocblas_gemm_algo_solution_index;
}

struct GemmSolutionMeasurement {
  int solution = 0;
  bool failed = false;
  std::vector<float> samples_ms;
};

struct AttentionGemmSpec {
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

struct MlpSolutionCacheKey {
  int hip_runtime = 0;
  int rocblas_major = ROCBLAS_VERSION_MAJOR;
  int rocblas_minor = ROCBLAS_VERSION_MINOR;
  int rocblas_patch = ROCBLAS_VERSION_PATCH;
  std::string device_arch;
  int compute_units = 0;
  int model = 0;
  int rows = 0;
  int model_dim = 0;
  int mlp = 0;
};

struct AttentionSolutionCacheKey {
  int hip_runtime = 0;
  int rocblas_major = ROCBLAS_VERSION_MAJOR;
  int rocblas_minor = ROCBLAS_VERSION_MINOR;
  int rocblas_patch = ROCBLAS_VERSION_PATCH;
  std::string device_arch;
  int compute_units = 0;
  int model = 0;
  int rows = 0;
  int model_dim = 0;
  int heads = 0;
  int qkv_dim = 0;
};

float MedianOf(std::vector<float> values) {
  if (values.empty()) return INFINITY;
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int HipRuntimeVersion() {
  int version = 0;
  if (hipRuntimeGetVersion(&version) != hipSuccess) return 0;
  return version;
}

MlpSolutionCacheKey MakeMlpSolutionCacheKey(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const PaliGemma2VitHipStats& stats) {
  MlpSolutionCacheKey key;
  key.hip_runtime = HipRuntimeVersion();
  key.device_arch = stats.device_arch;
  key.compute_units = stats.compute_units;
  key.model = static_cast<int>(model_config.model);
  key.rows = static_cast<int>(model_config.vit_config.seq_len);
  key.model_dim = static_cast<int>(model_config.vit_config.model_dim);
  key.mlp = weights.vit_layers.empty()
                ? 0
                : static_cast<int>(weights.VitLayer(0)->layer_config
                                       .ff_hidden_dim);
  return key;
}

AttentionSolutionCacheKey MakeAttentionSolutionCacheKey(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const PaliGemma2VitHipStats& stats) {
  AttentionSolutionCacheKey key;
  key.hip_runtime = HipRuntimeVersion();
  key.device_arch = stats.device_arch;
  key.compute_units = stats.compute_units;
  key.model = static_cast<int>(model_config.model);
  key.rows = static_cast<int>(model_config.vit_config.seq_len);
  key.model_dim = static_cast<int>(model_config.vit_config.model_dim);
  if (!weights.vit_layers.empty()) {
    const LayerConfig& layer_config = weights.VitLayer(0)->layer_config;
    key.heads = static_cast<int>(layer_config.heads);
    key.qkv_dim = static_cast<int>(layer_config.qkv_dim);
  }
  return key;
}

bool ExtractCacheField(const std::string& line, const char* name,
                       std::string& value) {
  const std::string prefix = std::string(name) + "=";
  std::istringstream in(line);
  std::string token;
  while (in >> token) {
    if (token.compare(0, prefix.size(), prefix) == 0) {
      value = token.substr(prefix.size());
      return true;
    }
  }
  return false;
}

bool ExtractCacheInt(const std::string& line, const char* name, int& value) {
  std::string text;
  if (!ExtractCacheField(line, name, text)) return false;
  char* end = nullptr;
  const long parsed = std::strtol(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0') return false;
  value = static_cast<int>(parsed);
  return true;
}

bool CacheLineMatchesKey(const std::string& line,
                         const MlpSolutionCacheKey& key) {
  int value = 0;
  std::string text;
  if (!ExtractCacheInt(line, "hip_runtime", value) ||
      value != key.hip_runtime) {
    return false;
  }
  if (!ExtractCacheInt(line, "rocblas_major", value) ||
      value != key.rocblas_major) {
    return false;
  }
  if (!ExtractCacheInt(line, "rocblas_minor", value) ||
      value != key.rocblas_minor) {
    return false;
  }
  if (!ExtractCacheInt(line, "rocblas_patch", value) ||
      value != key.rocblas_patch) {
    return false;
  }
  if (!ExtractCacheField(line, "device_arch", text) ||
      text != key.device_arch) {
    return false;
  }
  if (!ExtractCacheInt(line, "compute_units", value) ||
      value != key.compute_units) {
    return false;
  }
  if (!ExtractCacheInt(line, "model", value) || value != key.model) {
    return false;
  }
  if (!ExtractCacheInt(line, "rows", value) || value != key.rows) {
    return false;
  }
  if (!ExtractCacheInt(line, "model_dim", value) ||
      value != key.model_dim) {
    return false;
  }
  if (!ExtractCacheInt(line, "mlp", value) || value != key.mlp) {
    return false;
  }
  return true;
}

bool CacheLineMatchesKey(const std::string& line,
                         const AttentionSolutionCacheKey& key) {
  int value = 0;
  std::string text;
  if (!ExtractCacheInt(line, "hip_runtime", value) ||
      value != key.hip_runtime) {
    return false;
  }
  if (!ExtractCacheInt(line, "rocblas_major", value) ||
      value != key.rocblas_major) {
    return false;
  }
  if (!ExtractCacheInt(line, "rocblas_minor", value) ||
      value != key.rocblas_minor) {
    return false;
  }
  if (!ExtractCacheInt(line, "rocblas_patch", value) ||
      value != key.rocblas_patch) {
    return false;
  }
  if (!ExtractCacheField(line, "device_arch", text) ||
      text != key.device_arch) {
    return false;
  }
  if (!ExtractCacheInt(line, "compute_units", value) ||
      value != key.compute_units) {
    return false;
  }
  if (!ExtractCacheInt(line, "model", value) || value != key.model) {
    return false;
  }
  if (!ExtractCacheInt(line, "rows", value) || value != key.rows) {
    return false;
  }
  if (!ExtractCacheInt(line, "model_dim", value) ||
      value != key.model_dim) {
    return false;
  }
  if (!ExtractCacheInt(line, "heads", value) || value != key.heads) {
    return false;
  }
  if (!ExtractCacheInt(line, "qkv_dim", value) || value != key.qkv_dim) {
    return false;
  }
  return true;
}

int BenchmarkOneGemmSolutions(
    const char* label, rocblas_handle handle,
    const PaliGemma2VitHipOptions& options, int max_solutions, int m, int n,
    int k, const void* a, int lda, const void* b, int ldb, void* d, int ldd) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const double flops =
      2.0 * static_cast<double>(m) * static_cast<double>(n) * k;
  std::vector<rocblas_int> solutions = QueryGemmSolutions(
      handle, rocblas_operation_none, rocblas_operation_none, m, n, k, &alpha,
      a, rocblas_datatype_bf16_r, lda, b, rocblas_datatype_bf16_r, ldb, &beta,
      d, rocblas_datatype_f32_r, ldd, d, rocblas_datatype_f32_r, ldd,
      rocblas_datatype_f32_r);

  std::vector<GemmSolutionMeasurement> candidates;
  candidates.push_back({/*solution=*/0, /*failed=*/false, {}});
  int tested = 0;
  for (rocblas_int solution : solutions) {
    if (solution == 0) continue;
    if (tested >= max_solutions) break;
    candidates.push_back({static_cast<int>(solution), /*failed=*/false, {}});
    ++tested;
  }

  const int rounds = std::max(3, options.samples);
  PaliGemma2VitHipOptions once = options;
  once.samples = 1;
  std::printf("mlp_solution_bench %s m=%d n=%d k=%d solutions=%zu tested=%d "
              "rounds=%d warmup=%d iters=%d\n",
              label, m, n, k, solutions.size(), tested, rounds,
              options.warmup, options.iters);

  for (int round = 0; round < rounds; ++round) {
    once.warmup = round == 0 ? options.warmup : 0;
    for (size_t offset = 0; offset < candidates.size(); ++offset) {
      GemmSolutionMeasurement& candidate =
          candidates[(offset + static_cast<size_t>(round)) %
                     candidates.size()];
      if (candidate.failed) continue;
      auto run_candidate = [&]() {
        return RunBf16F32GemmSolution(handle, m, n, k, a, lda, b, ldb, d, ldd,
                                      candidate.solution);
      };
      float sample_ms = 0.0f;
      if (TimeRocblasStatusSamples(once, run_candidate, sample_ms)) {
        candidate.samples_ms.push_back(sample_ms);
      } else {
        candidate.failed = true;
      }
    }
  }

  int best_solution = 0;
  float best_ms = INFINITY;
  float default_ms = INFINITY;
  for (GemmSolutionMeasurement& candidate : candidates) {
    if (candidate.failed || candidate.samples_ms.empty()) {
      std::printf("  %-12s solution=%7d failed\n", label,
                  candidate.solution);
      continue;
    }
    const float median_ms = MedianOf(candidate.samples_ms);
    const auto [min_it, max_it] =
        std::minmax_element(candidate.samples_ms.begin(),
                            candidate.samples_ms.end());
    if (candidate.solution == 0) default_ms = median_ms;
    const char* solution_label =
        candidate.solution == 0 ? "default" : "solution";
    std::printf("  %-12s %-8s=%7d median=%8.3f ms min=%8.3f max=%8.3f "
                "spread=%8.3f %8.1f GFLOPS\n",
                label, solution_label, candidate.solution, median_ms, *min_it,
                *max_it, *max_it - *min_it, flops / (median_ms * 1.0E6));
    if (median_ms < best_ms) {
      best_ms = median_ms;
      best_solution = candidate.solution;
    }
  }

  if (std::isfinite(best_ms)) {
    const float rel = std::isfinite(default_ms) && default_ms > 0.0f
                          ? best_ms / default_ms
                          : INFINITY;
    std::printf("  %-12s selected=%7d median=%8.3f ms rel_to_default=%6.3f\n",
                label, best_solution, best_ms, rel);
  }
  return best_solution;
}

std::vector<rocblas_int> QueryAttentionGemmSolutions(
    rocblas_handle handle, const AttentionGemmSpec& spec) {
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

rocblas_status RunAttentionGemmSolution(rocblas_handle handle,
                                        const AttentionGemmSpec& spec,
                                        int solution_index) {
  return rocblas_gemm_strided_batched_ex(
      handle, spec.trans_a, spec.trans_b, spec.m, spec.n, spec.k, &spec.alpha,
      spec.a, rocblas_datatype_f32_r, spec.lda, spec.stride_a, spec.b,
      rocblas_datatype_f32_r, spec.ldb, spec.stride_b, &spec.beta, spec.d,
      rocblas_datatype_f32_r, spec.ldd, spec.stride_d, spec.d,
      rocblas_datatype_f32_r, spec.ldd, spec.stride_d, spec.batch_count,
      rocblas_datatype_f32_r, RocblasAlgoForSolutionIndex(solution_index),
      solution_index, rocblas_gemm_flags_none);
}

int BenchmarkOneAttentionGemmSolutions(
    rocblas_handle handle, const PaliGemma2VitHipOptions& options,
    int max_solutions, const AttentionGemmSpec& spec) {
  const double flops =
      2.0 * static_cast<double>(spec.batch_count) * spec.m *
      static_cast<double>(spec.n) * spec.k;
  std::vector<rocblas_int> solutions =
      QueryAttentionGemmSolutions(handle, spec);

  std::vector<GemmSolutionMeasurement> candidates;
  candidates.push_back({/*solution=*/0, /*failed=*/false, {}});
  int tested = 0;
  for (rocblas_int solution : solutions) {
    if (solution == 0) continue;
    if (tested >= max_solutions) break;
    candidates.push_back({static_cast<int>(solution), /*failed=*/false, {}});
    ++tested;
  }

  const int rounds = std::max(3, options.samples);
  PaliGemma2VitHipOptions once = options;
  once.samples = 1;
  std::printf("attention_solution_bench %s m=%d n=%d k=%d batch=%d "
              "solutions=%zu tested=%d rounds=%d warmup=%d iters=%d\n",
              spec.label, spec.m, spec.n, spec.k, spec.batch_count,
              solutions.size(), tested, rounds, options.warmup,
              options.iters);

  for (int round = 0; round < rounds; ++round) {
    once.warmup = round == 0 ? options.warmup : 0;
    for (size_t offset = 0; offset < candidates.size(); ++offset) {
      GemmSolutionMeasurement& candidate =
          candidates[(offset + static_cast<size_t>(round)) %
                     candidates.size()];
      if (candidate.failed) continue;
      auto run_candidate = [&]() {
        return RunAttentionGemmSolution(handle, spec, candidate.solution);
      };
      float sample_ms = 0.0f;
      if (TimeRocblasStatusSamples(once, run_candidate, sample_ms)) {
        candidate.samples_ms.push_back(sample_ms);
      } else {
        candidate.failed = true;
      }
    }
  }

  int best_solution = 0;
  float best_ms = INFINITY;
  float default_ms = INFINITY;
  for (GemmSolutionMeasurement& candidate : candidates) {
    if (candidate.failed || candidate.samples_ms.empty()) {
      std::printf("  %-12s solution=%7d failed\n", spec.label,
                  candidate.solution);
      continue;
    }
    const float median_ms = MedianOf(candidate.samples_ms);
    const auto [min_it, max_it] =
        std::minmax_element(candidate.samples_ms.begin(),
                            candidate.samples_ms.end());
    if (candidate.solution == 0) default_ms = median_ms;
    const char* solution_label =
        candidate.solution == 0 ? "default" : "solution";
    std::printf("  %-12s %-8s=%7d median=%8.3f ms min=%8.3f max=%8.3f "
                "spread=%8.3f %8.1f GFLOPS\n",
                spec.label, solution_label, candidate.solution, median_ms,
                *min_it, *max_it, *max_it - *min_it,
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
    std::printf("  %-12s selected=%7d median=%8.3f ms rel_to_default=%6.3f\n",
                spec.label, best_solution, best_ms, rel);
  }
  return best_solution;
}

void LaunchSplitQKVForAttention(const float* d_qkv, float* d_q, float* d_k,
                                float* d_v, int rows, int heads,
                                int qkv_dim) {
  constexpr int kThreads = 256;
  const int qkv_count = rows * heads * qkv_dim;

  // The vectorized split is a reliable small win for the 224px shape in the
  // local microbench, but the 448px measurements were noise-level and sometimes
  // slower than the scalar kernel. Keep it shape-gated instead of changing the
  // 448px recommended path.
  if (rows == 256 && (qkv_dim % 4) == 0) {
    const int vector_count = rows * heads * (qkv_dim / 4);
    const int vector_blocks = (vector_count + kThreads - 1) / kThreads;
    hipLaunchKernelGGL(SplitQKVForAttentionFloat4Kernel, dim3(vector_blocks),
                       dim3(kThreads), 0, 0, d_qkv, d_q, d_k, d_v, rows,
                       heads, qkv_dim);
    HIP_CHECK(hipGetLastError());
    return;
  }

  const int qkv_blocks = (qkv_count + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(SplitQKVForAttentionKernel, dim3(qkv_blocks),
                     dim3(kThreads), 0, 0, d_qkv, d_q, d_k, d_v, rows, heads,
                     qkv_dim);
  HIP_CHECK(hipGetLastError());
}

void LaunchSoftmaxRows(float* d_scores, int rows, int heads) {
  // PaliGemma2 currently uses rows=256 for 224px and rows=1024 for 448px. These
  // fixed-shape kernels expose row length and block size as compile-time
  // constants so the compiler can simplify address and reduction code. The
  // 448px path uses 128 threads because repeated current-machine sweeps found it
  // slightly faster than the original 256-thread fixed kernel while preserving
  // the same normalized F32 score boundary before AV.
  if (rows == 256) {
    // PaliGemma2 score rows are 16-byte aligned, so a float4 row pass cuts
    // memory instructions while keeping the same in-place F32 softmax boundary.
    hipLaunchKernelGGL((SoftmaxRowsFixedFloat4Kernel<256, 64>),
                       dim3(rows, heads), dim3(64), 0, 0, d_scores);
    HIP_CHECK(hipGetLastError());
    return;
  }
  if (rows == 1024) {
    hipLaunchKernelGGL((SoftmaxRowsFixedFloat4Kernel<1024, 128>),
                       dim3(rows, heads), dim3(128), 0, 0, d_scores);
    HIP_CHECK(hipGetLastError());
    return;
  }

  const int softmax_threads = rows <= 256 ? 64 : 256;
  const size_t softmax_shared =
      static_cast<size_t>(softmax_threads) * sizeof(float);
  hipLaunchKernelGGL(SoftmaxRowsKernel, dim3(rows, heads),
                     dim3(softmax_threads), softmax_shared, 0, d_scores, rows);
  HIP_CHECK(hipGetLastError());
}

void LaunchPackAttentionHeadsToBF16(const float* d_att_head_major,
                                    rocblas_bfloat16* d_att_row_major,
                                    int rows, int heads, int qkv_dim) {
  constexpr int kThreads = 256;

  // The recommended PaliGemma2 path packs head-major F32 attention output
  // directly into the BF16 row-major layout consumed by attn_out. For the fixed
  // head dimension 72, a float4 load cuts launch work by 4x without changing
  // where the F32->BF16 rounding happens.
  if ((qkv_dim % 4) == 0) {
    const int vector_count = rows * heads * (qkv_dim / 4);
    const int vector_blocks = (vector_count + kThreads - 1) / kThreads;
    hipLaunchKernelGGL(PackAttentionHeadsToBF16Float4Kernel,
                       dim3(vector_blocks), dim3(kThreads), 0, 0,
                       d_att_head_major, d_att_row_major, rows, heads,
                       qkv_dim);
    HIP_CHECK(hipGetLastError());
    return;
  }

  const int count = rows * heads * qkv_dim;
  const int blocks = (count + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(PackAttentionHeadsToBF16Kernel, dim3(blocks),
                     dim3(kThreads), 0, 0, d_att_head_major, d_att_row_major,
                     rows, heads, qkv_dim);
  HIP_CHECK(hipGetLastError());
}

bool CanUseAttentionAvF32Dim4(int rows, int qkv_dim) {
  return rows == 256 && qkv_dim == 72;
}

void LaunchAttentionAvF32Dim4(const float* d_scores, const float* d_v,
                              float* d_att_head_major, int rows, int heads,
                              int qkv_dim) {
  if (!CanUseAttentionAvF32Dim4(rows, qkv_dim)) {
    std::fprintf(stderr,
                 "AttentionAvF32Dim4 only supports rows=256 qkv_dim=72\n");
    std::exit(1);
  }

  // Best local bench candidate on gfx1150 for the 224px AV shape. BM=32 keeps
  // enough query rows in a workgroup to reuse V across rows, while BK=16 avoids
  // the occupancy loss observed with larger K tiles.
  constexpr int kBM = 32;
  constexpr int kBK = 16;
  constexpr int kDim4 = 18;
  const dim3 block(kDim4, kBM);
  const dim3 grid((rows + kBM - 1) / kBM, heads);
  hipLaunchKernelGGL((AttentionAvTiledDim4Kernel<kBM, kBK, kDim4>), grid,
                     block, 0, 0, d_scores, d_v, d_att_head_major, rows);
  HIP_CHECK(hipGetLastError());
}

void RunDeviceAttention(rocblas_handle handle, const float* d_qkv, float* d_q,
                        float* d_k, float* d_v, float* d_scores,
                        float* d_att_head_major, float* d_att_row_major,
                        int rows, int heads, int qkv_dim,
                        int qk_solution_index = 0,
                        int av_solution_index = 0) {
  // Device attention core:
  //   1. split QKV to head-major Q/K/V;
  //   2. batched QK via rocBLAS;
  //   3. row-wise softmax in a custom HIP kernel;
  //   4. batched AV via rocBLAS;
  //   5. pack head-major output back to [row, head, dim].
  // QK applies 1/sqrt(qkv_dim) as the GEMM alpha after dot accumulation, which
  // matches gemma/vit.cc more closely than scaling Q before the dot.
  //
  // This is intentionally an explicit multi-kernel prototype. It is the next
  // step after the negative single fused kernel: correctness first, then decide
  // which kernel boundaries deserve fusion.
  const int threads = 256;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;
  const rocblas_gemm_algo qk_algo =
      RocblasAlgoForSolutionIndex(qk_solution_index);
  const rocblas_gemm_algo av_algo =
      RocblasAlgoForSolutionIndex(av_solution_index);

  LaunchSplitQKVForAttention(d_qkv, d_q, d_k, d_v, rows, heads, qkv_dim);

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
      qkv_dim, &alpha_qk, d_k, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_q, rocblas_datatype_f32_r, qkv_dim, stride_qkv, &beta, d_scores,
      rocblas_datatype_f32_r, rows, stride_scores, d_scores,
      rocblas_datatype_f32_r, rows, stride_scores, heads,
      rocblas_datatype_f32_r, qk_algo, qk_solution_index, /*flags=*/0));

  LaunchSoftmaxRows(d_scores, rows, heads);

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
      rows, &alpha_av, d_v, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_scores, rocblas_datatype_f32_r, rows, stride_scores, &beta,
      d_att_head_major, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_att_head_major, rocblas_datatype_f32_r, qkv_dim, stride_qkv, heads,
      rocblas_datatype_f32_r, av_algo, av_solution_index, /*flags=*/0));

  hipLaunchKernelGGL(PackAttentionHeadsKernel, dim3(qkv_blocks), dim3(threads),
                     0, 0, d_att_head_major, d_att_row_major, rows, heads,
                     qkv_dim);
  HIP_CHECK(hipGetLastError());
}

void RunDeviceAttentionNoPack(rocblas_handle handle, const float* d_qkv,
                              float* d_q, float* d_k, float* d_v,
                              float* d_scores, float* d_att_head_major,
                              int rows, int heads, int qkv_dim,
                              int qk_solution_index = 0,
                              int av_solution_index = 0) {
  // F32-semantics attention core without the final head-major -> row-major pack.
  // The BF16-only ViT layer runner uses this when it wants to fuse the pack with
  // the existing BF16 conversion immediately before attn_out. Keeping the fused
  // pack in the attn_out stage also preserves the original producer/consumer
  // ordering around the following rocBLAS GEMM.
  const int threads = 256;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;
  const rocblas_gemm_algo qk_algo =
      RocblasAlgoForSolutionIndex(qk_solution_index);
  const rocblas_gemm_algo av_algo =
      RocblasAlgoForSolutionIndex(av_solution_index);

  LaunchSplitQKVForAttention(d_qkv, d_q, d_k, d_v, rows, heads, qkv_dim);

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
      qkv_dim, &alpha_qk, d_k, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_q, rocblas_datatype_f32_r, qkv_dim, stride_qkv, &beta, d_scores,
      rocblas_datatype_f32_r, rows, stride_scores, d_scores,
      rocblas_datatype_f32_r, rows, stride_scores, heads,
      rocblas_datatype_f32_r, qk_algo, qk_solution_index, /*flags=*/0));

  LaunchSoftmaxRows(d_scores, rows, heads);

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
      rows, &alpha_av, d_v, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_scores, rocblas_datatype_f32_r, rows, stride_scores, &beta,
      d_att_head_major, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_att_head_major, rocblas_datatype_f32_r, qkv_dim, stride_qkv, heads,
      rocblas_datatype_f32_r, av_algo, av_solution_index, /*flags=*/0));
}

void RunDeviceAttentionDirectQKVNoPack(rocblas_handle handle,
                                       const float* d_qkv, float* d_scores,
                                       float* d_att_head_major, int rows,
                                       int heads, int qkv_dim,
                                       int qk_solution_index = 0,
                                       int av_solution_index = 0) {
  // F32-semantics attention core that skips the Q/K/V split. The QKV projection
  // is row-major [rows, heads, q/k/v, dim], so each head's Q/K/V matrix has a
  // large row stride (`heads * 3 * qkv_dim`) but still presents a valid
  // column-major view to rocBLAS through lda=qkv_cols. This trades slightly less
  // contiguous GEMM reads for removing the split kernel and the Q/K/V scratch
  // writes while keeping QK, softmax, and AV in F32.
  const int qkv_cols = heads * 3 * qkv_dim;
  const float* d_q_direct = d_qkv;
  const float* d_k_direct = d_qkv + qkv_dim;
  const float* d_v_direct = d_qkv + 2 * qkv_dim;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv_direct =
      static_cast<rocblas_stride>(3) * qkv_dim;
  const rocblas_stride stride_qkv_head_major =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;
  const rocblas_gemm_algo qk_algo =
      RocblasAlgoForSolutionIndex(qk_solution_index);
  const rocblas_gemm_algo av_algo =
      RocblasAlgoForSolutionIndex(av_solution_index);

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
      qkv_dim, &alpha_qk, d_k_direct, rocblas_datatype_f32_r, qkv_cols,
      stride_qkv_direct, d_q_direct, rocblas_datatype_f32_r, qkv_cols,
      stride_qkv_direct, &beta, d_scores, rocblas_datatype_f32_r, rows,
      stride_scores, d_scores, rocblas_datatype_f32_r, rows, stride_scores,
      heads, rocblas_datatype_f32_r, qk_algo, qk_solution_index, /*flags=*/0));

  LaunchSoftmaxRows(d_scores, rows, heads);

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
      rows, &alpha_av, d_v_direct, rocblas_datatype_f32_r, qkv_cols,
      stride_qkv_direct, d_scores, rocblas_datatype_f32_r, rows,
      stride_scores, &beta, d_att_head_major, rocblas_datatype_f32_r, qkv_dim,
      stride_qkv_head_major, d_att_head_major, rocblas_datatype_f32_r, qkv_dim,
      stride_qkv_head_major, heads, rocblas_datatype_f32_r, av_algo,
      av_solution_index, /*flags=*/0));
}

void RunDeviceAttentionDirectQKV(rocblas_handle handle, const float* d_qkv,
                                 float* d_scores, float* d_att_head_major,
                                 float* d_att_row_major, int rows, int heads,
                                 int qkv_dim, int qk_solution_index = 0,
                                 int av_solution_index = 0) {
  RunDeviceAttentionDirectQKVNoPack(handle, d_qkv, d_scores, d_att_head_major,
                                    rows, heads, qkv_dim, qk_solution_index,
                                    av_solution_index);

  constexpr int kThreads = 256;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_blocks = (qkv_count + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(PackAttentionHeadsKernel, dim3(qkv_blocks),
                     dim3(kThreads), 0, 0, d_att_head_major, d_att_row_major,
                     rows, heads, qkv_dim);
  HIP_CHECK(hipGetLastError());
}

void RunDeviceAttentionAvF32Dim4NoPack(
    rocblas_handle handle, const float* d_qkv, float* d_q, float* d_k,
    float* d_v, float* d_scores, float* d_att_head_major, int rows, int heads,
    int qkv_dim, int qk_solution_index = 0) {
  // F32-preserving AV replacement:
  //   1. keep the recommended F32 QK rocBLAS score GEMM;
  //   2. keep the existing F32 softmax normalization;
  //   3. replace only AV with a 224px-specialized float4 tiled HIP kernel.
  //
  // This is intentionally narrower than the BF16 AV WMMA experiment. It does
  // not quantize scores or V, so generation differences should only come from
  // normal F32 accumulation-order changes inside AV.
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float beta = 0.0f;
  const rocblas_gemm_algo qk_algo =
      RocblasAlgoForSolutionIndex(qk_solution_index);

  LaunchSplitQKVForAttention(d_qkv, d_q, d_k, d_v, rows, heads, qkv_dim);

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
      qkv_dim, &alpha_qk, d_k, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_q, rocblas_datatype_f32_r, qkv_dim, stride_qkv, &beta, d_scores,
      rocblas_datatype_f32_r, rows, stride_scores, d_scores,
      rocblas_datatype_f32_r, rows, stride_scores, heads,
      rocblas_datatype_f32_r, qk_algo, qk_solution_index, /*flags=*/0));

  LaunchSoftmaxRows(d_scores, rows, heads);
  LaunchAttentionAvF32Dim4(d_scores, d_v, d_att_head_major, rows, heads,
                           qkv_dim);
}

void RunDeviceAttentionAvF32Dim4(
    rocblas_handle handle, const float* d_qkv, float* d_q, float* d_k,
    float* d_v, float* d_scores, float* d_att_head_major,
    float* d_att_row_major, int rows, int heads, int qkv_dim,
    int qk_solution_index = 0) {
  RunDeviceAttentionAvF32Dim4NoPack(handle, d_qkv, d_q, d_k, d_v, d_scores,
                                    d_att_head_major, rows, heads, qkv_dim,
                                    qk_solution_index);

  constexpr int kThreads = 256;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_blocks = (qkv_count + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(PackAttentionHeadsKernel, dim3(qkv_blocks),
                     dim3(kThreads), 0, 0, d_att_head_major, d_att_row_major,
                     rows, heads, qkv_dim);
  HIP_CHECK(hipGetLastError());
}

void RunDeviceAttentionNoPackDeferredScale(
    rocblas_handle handle, const float* d_qkv, float* d_q, float* d_k,
    float* d_v, float* d_scores, float* d_inv_sums,
    float* d_att_head_major, int rows, int heads, int qkv_dim,
    int qk_solution_index = 0, int av_solution_index = 0) {
  // F32 attention variant for the pack-to-BF16 route. QK and AV remain rocBLAS
  // F32. The only scheduling change is softmax normalization:
  //   - softmax stores exp(score - max) plus one reciprocal denominator per row;
  //   - AV accumulates against those unnormalized exp scores;
  //   - pack-to-BF16 multiplies the final head output by the saved denominator.
  // This removes the in-place normalization pass over the large score matrix
  // and replaces it with one multiply over the much smaller attention output.
  const int softmax_threads = rows <= 256 ? 64 : 256;
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
  const rocblas_gemm_algo qk_algo =
      RocblasAlgoForSolutionIndex(qk_solution_index);
  const rocblas_gemm_algo av_algo =
      RocblasAlgoForSolutionIndex(av_solution_index);

  LaunchSplitQKVForAttention(d_qkv, d_q, d_k, d_v, rows, heads, qkv_dim);

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
      qkv_dim, &alpha_qk, d_k, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_q, rocblas_datatype_f32_r, qkv_dim, stride_qkv, &beta, d_scores,
      rocblas_datatype_f32_r, rows, stride_scores, d_scores,
      rocblas_datatype_f32_r, rows, stride_scores, heads,
      rocblas_datatype_f32_r, qk_algo, qk_solution_index, /*flags=*/0));

  hipLaunchKernelGGL(SoftmaxRowsStoreExpAndInvSumKernel, dim3(rows, heads),
                     dim3(softmax_threads), softmax_shared, 0, d_scores,
                     d_inv_sums, rows);
  HIP_CHECK(hipGetLastError());

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
      rows, &alpha_av, d_v, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_scores, rocblas_datatype_f32_r, rows, stride_scores, &beta,
      d_att_head_major, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_att_head_major, rocblas_datatype_f32_r, qkv_dim, stride_qkv, heads,
      rocblas_datatype_f32_r, av_algo, av_solution_index, /*flags=*/0));
}

void RunDeviceAttentionQkBf16(rocblas_handle handle, const float* d_qkv,
                              rocblas_bfloat16* d_q, rocblas_bfloat16* d_k,
                              float* d_v, float* d_scores,
                              float* d_att_head_major,
                              float* d_att_row_major, int rows, int heads,
                              int qkv_dim, int av_solution_index = 0) {
  // Narrow mixed-precision attention experiment:
  //   1. split Q/K to BF16 and V to F32;
  //   2. run QK as BF16 inputs with F32 accumulation/output;
  //   3. keep the existing F32 softmax and F32 AV path.
  //
  // This targets the phase-profile result where BF16 QK was faster but full
  // BF16 AV paid an expensive score-quantization cost. The explicit separate
  // function keeps the default F32 attention path unchanged.
  const int threads = 256;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;
  const rocblas_gemm_algo av_algo =
      RocblasAlgoForSolutionIndex(av_solution_index);

  hipLaunchKernelGGL(SplitQKToBF16VFloatForAttentionKernel, dim3(qkv_blocks),
                     dim3(threads), 0, 0, d_qkv, d_q, d_k, d_v, rows, heads,
                     qkv_dim);
  HIP_CHECK(hipGetLastError());

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
      qkv_dim, &alpha_qk, d_k, rocblas_datatype_bf16_r, qkv_dim, stride_qkv,
      d_q, rocblas_datatype_bf16_r, qkv_dim, stride_qkv, &beta, d_scores,
      rocblas_datatype_f32_r, rows, stride_scores, d_scores,
      rocblas_datatype_f32_r, rows, stride_scores, heads,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
      /*solution_index=*/0, /*flags=*/0));

  LaunchSoftmaxRows(d_scores, rows, heads);

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
      rows, &alpha_av, d_v, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_scores, rocblas_datatype_f32_r, rows, stride_scores, &beta,
      d_att_head_major, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_att_head_major, rocblas_datatype_f32_r, qkv_dim, stride_qkv, heads,
      rocblas_datatype_f32_r, av_algo, av_solution_index, /*flags=*/0));

  hipLaunchKernelGGL(PackAttentionHeadsKernel, dim3(qkv_blocks), dim3(threads),
                     0, 0, d_att_head_major, d_att_row_major, rows, heads,
                     qkv_dim);
  HIP_CHECK(hipGetLastError());
}

void RunDeviceAttentionQkBf16NoPack(rocblas_handle handle, const float* d_qkv,
                                    rocblas_bfloat16* d_q,
                                    rocblas_bfloat16* d_k, float* d_v,
                                    float* d_scores,
                                    float* d_att_head_major, int rows,
                                    int heads, int qkv_dim,
                                    int av_solution_index = 0) {
  // No-pack variant for composing BF16 QK with pack-to-BF16. It produces the
  // same head-major F32 attention output as RunDeviceAttentionQkBf16, but leaves
  // the row-major pack and BF16 conversion to the following attn_out boundary.
  const int threads = 256;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;
  const rocblas_gemm_algo av_algo =
      RocblasAlgoForSolutionIndex(av_solution_index);

  hipLaunchKernelGGL(SplitQKToBF16VFloatForAttentionKernel, dim3(qkv_blocks),
                     dim3(threads), 0, 0, d_qkv, d_q, d_k, d_v, rows, heads,
                     qkv_dim);
  HIP_CHECK(hipGetLastError());

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
      qkv_dim, &alpha_qk, d_k, rocblas_datatype_bf16_r, qkv_dim, stride_qkv,
      d_q, rocblas_datatype_bf16_r, qkv_dim, stride_qkv, &beta, d_scores,
      rocblas_datatype_f32_r, rows, stride_scores, d_scores,
      rocblas_datatype_f32_r, rows, stride_scores, heads,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
      /*solution_index=*/0, /*flags=*/0));

  LaunchSoftmaxRows(d_scores, rows, heads);

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
      rows, &alpha_av, d_v, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_scores, rocblas_datatype_f32_r, rows, stride_scores, &beta,
      d_att_head_major, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_att_head_major, rocblas_datatype_f32_r, qkv_dim, stride_qkv, heads,
      rocblas_datatype_f32_r, av_algo, av_solution_index, /*flags=*/0));
}

void LaunchAttentionQkBf16Wmma(const rocblas_bfloat16* d_q,
                               const rocblas_bfloat16* d_k, float* d_scores,
                               int rows, int heads, int qkv_dim,
                               float query_scale, rocblas_stride stride_qkv,
                               rocblas_stride stride_scores) {
  // Current gfx1150 retunes favor the same grouped-2D macro-tile for both ViT
  // image resolutions: 2 M waves by 8 N waves. For 224px it is only a small
  // win over 4x4, but for 448px it is consistently the fastest candidate.
  // The whole path remains opt-in because BF16 QK changes attention scores
  // relative to the default F32 QK path.
  constexpr int kWavesM = 2;
  constexpr int kWavesN = 8;
  const dim3 block(32 * kWavesM * kWavesN);
  const dim3 grid((rows + 16 * kWavesN - 1) / (16 * kWavesN),
                  (rows + 16 * kWavesM - 1) / (16 * kWavesM), heads);
  hipLaunchKernelGGL(
      (Bf16GemmWmma2DTransBScaleBatchedKernel<kWavesM, kWavesN>), grid, block,
      0, 0, d_q, d_k, d_scores, rows, qkv_dim, rows,
      static_cast<size_t>(stride_qkv), static_cast<size_t>(stride_qkv),
      static_cast<size_t>(stride_scores), query_scale);
}

const char* AttentionQkBf16WmmaTileName(int rows) {
  (void)rows;
  return "wmma2d2x8";
}

void LaunchAttentionAvBf16Wmma(const rocblas_bfloat16* d_scores_bf16,
                               const rocblas_bfloat16* d_v_bf16,
                               float* d_att_head_major, int rows, int heads,
                               int qkv_dim) {
  // AV has a very skinny output dimension (qkv_dim=72). The tile sweep showed
  // that widening N beyond 32 columns wastes work on the tail, while grouping
  // eight M waves improves score-row reuse and occupancy for both 224px and
  // 448px shapes.
  constexpr int kWavesM = 8;
  constexpr int kWavesN = 2;
  const dim3 block(32 * kWavesM * kWavesN);
  const dim3 grid((qkv_dim + 16 * kWavesN - 1) / (16 * kWavesN),
                  (rows + 16 * kWavesM - 1) / (16 * kWavesM), heads);
  hipLaunchKernelGGL((Bf16GemmWmma2DBatchedKernel<kWavesM, kWavesN>), grid,
                     block, 0, 0, d_scores_bf16, d_v_bf16, d_att_head_major,
                     rows, rows, qkv_dim, static_cast<size_t>(rows) * rows,
                     static_cast<size_t>(rows) * qkv_dim,
                     static_cast<size_t>(rows) * qkv_dim);
}

void RunDeviceAttentionQkBf16Wmma(rocblas_handle handle, const float* d_qkv,
                                  rocblas_bfloat16* d_q,
                                  rocblas_bfloat16* d_k, float* d_v,
                                  float* d_scores,
                                  float* d_att_head_major,
                                  float* d_att_row_major, int rows, int heads,
                                  int qkv_dim, int av_solution_index = 0) {
  // Local-kernel variant of RunDeviceAttentionQkBf16. Q and K use the same BF16
  // split and the rest of attention stays F32, but the QK score GEMM is a
  // grouped-2D RDNA WMMA kernel that reads K as a transposed row-major operand.
  // This keeps the precision boundary identical to the rocBLAS BF16 QK
  // experiment while testing the custom matrix-instruction path in the real
  // device-resident layer runner.
  const int threads = 256;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;
  const rocblas_gemm_algo av_algo =
      RocblasAlgoForSolutionIndex(av_solution_index);

  hipLaunchKernelGGL(SplitQKToBF16VFloatForAttentionKernel, dim3(qkv_blocks),
                     dim3(threads), 0, 0, d_qkv, d_q, d_k, d_v, rows, heads,
                     qkv_dim);
  HIP_CHECK(hipGetLastError());

  LaunchAttentionQkBf16Wmma(d_q, d_k, d_scores, rows, heads, qkv_dim,
                            query_scale, stride_qkv, stride_scores);
  HIP_CHECK(hipGetLastError());

  LaunchSoftmaxRows(d_scores, rows, heads);

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
      rows, &alpha_av, d_v, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_scores, rocblas_datatype_f32_r, rows, stride_scores, &beta,
      d_att_head_major, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_att_head_major, rocblas_datatype_f32_r, qkv_dim, stride_qkv, heads,
      rocblas_datatype_f32_r, av_algo, av_solution_index, /*flags=*/0));

  hipLaunchKernelGGL(PackAttentionHeadsKernel, dim3(qkv_blocks), dim3(threads),
                     0, 0, d_att_head_major, d_att_row_major, rows, heads,
                     qkv_dim);
  HIP_CHECK(hipGetLastError());
}

void RunDeviceAttentionQkBf16WmmaNoPack(
    rocblas_handle handle, const float* d_qkv, rocblas_bfloat16* d_q,
    rocblas_bfloat16* d_k, float* d_v, float* d_scores,
    float* d_att_head_major, int rows, int heads, int qkv_dim,
    int av_solution_index = 0) {
  // No-pack variant for the local WMMA QK path. This is the fair full-path
  // comparison against F32 attention + pack_bf16: Q/K are still BF16, QK uses
  // the local transposed-B WMMA kernel, softmax and AV stay F32, and the final
  // head-major output is left for PackAttentionHeadsToBF16Kernel.
  const int threads = 256;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_av = 1.0f;
  const float beta = 0.0f;
  const rocblas_gemm_algo av_algo =
      RocblasAlgoForSolutionIndex(av_solution_index);

  hipLaunchKernelGGL(SplitQKToBF16VFloatForAttentionKernel, dim3(qkv_blocks),
                     dim3(threads), 0, 0, d_qkv, d_q, d_k, d_v, rows, heads,
                     qkv_dim);
  HIP_CHECK(hipGetLastError());

  LaunchAttentionQkBf16Wmma(d_q, d_k, d_scores, rows, heads, qkv_dim,
                            query_scale, stride_qkv, stride_scores);
  HIP_CHECK(hipGetLastError());

  LaunchSoftmaxRows(d_scores, rows, heads);

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_none, rocblas_operation_none, qkv_dim, rows,
      rows, &alpha_av, d_v, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_scores, rocblas_datatype_f32_r, rows, stride_scores, &beta,
      d_att_head_major, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_att_head_major, rocblas_datatype_f32_r, qkv_dim, stride_qkv, heads,
      rocblas_datatype_f32_r, av_algo, av_solution_index, /*flags=*/0));
}

void RunDeviceAttentionAvBf16Wmma(
    rocblas_handle handle, const float* d_qkv, float* d_q, float* d_k,
    float* d_v, float* d_scores, rocblas_bfloat16* d_scores_bf16,
    rocblas_bfloat16* d_v_bf16, float* d_att_head_major,
    float* d_att_row_major, int rows, int heads, int qkv_dim,
    int qk_solution_index = 0) {
  // Narrow AV-side precision experiment:
  //   1. keep Q/K/V split and QK GEMM in F32;
  //   2. keep the softmax reductions in F32, but store normalized scores as BF16;
  //   3. convert V to BF16;
  //   4. run AV with the local grouped-2D WMMA BF16->F32 kernel.
  //
  // The benchmark showed this removes the expensive separate score-quantization
  // pass. It still changes normalized attention-score storage precision, so this
  // function is only reachable through an explicit experimental option.
  const int threads = 256;
  const int softmax_threads = rows <= 256 ? 64 : 256;
  const int qkv_count = rows * heads * qkv_dim;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const int scores_count = heads * rows * rows;
  const int scores_blocks = (scores_count + threads - 1) / threads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));
  const size_t softmax_shared =
      static_cast<size_t>(softmax_threads) * sizeof(float);
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float alpha_qk = query_scale;
  const float beta = 0.0f;
  const rocblas_gemm_algo qk_algo =
      RocblasAlgoForSolutionIndex(qk_solution_index);

  LaunchSplitQKVForAttention(d_qkv, d_q, d_k, d_v, rows, heads, qkv_dim);

  ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
      handle, rocblas_operation_transpose, rocblas_operation_none, rows, rows,
      qkv_dim, &alpha_qk, d_k, rocblas_datatype_f32_r, qkv_dim, stride_qkv,
      d_q, rocblas_datatype_f32_r, qkv_dim, stride_qkv, &beta, d_scores,
      rocblas_datatype_f32_r, rows, stride_scores, d_scores,
      rocblas_datatype_f32_r, rows, stride_scores, heads,
      rocblas_datatype_f32_r, qk_algo, qk_solution_index, /*flags=*/0));

  hipLaunchKernelGGL(SoftmaxRowsToBF16Kernel, dim3(rows, heads),
                     dim3(softmax_threads), softmax_shared, 0, d_scores,
                     d_scores_bf16, rows);
  HIP_CHECK(hipGetLastError());

  hipLaunchKernelGGL(F32ToBF16Kernel, dim3(qkv_blocks), dim3(threads), 0, 0,
                     d_v, d_v_bf16, qkv_count);
  HIP_CHECK(hipGetLastError());

  LaunchAttentionAvBf16Wmma(d_scores_bf16, d_v_bf16, d_att_head_major, rows,
                            heads, qkv_dim);
  HIP_CHECK(hipGetLastError());

  hipLaunchKernelGGL(PackAttentionHeadsKernel, dim3(qkv_blocks), dim3(threads),
                     0, 0, d_att_head_major, d_att_row_major, rows, heads,
                     qkv_dim);
  HIP_CHECK(hipGetLastError());
}

float TimeDeviceAttention(rocblas_handle handle,
                          const PaliGemma2VitHipOptions& options,
                          const float* d_qkv, float* d_q, float* d_k,
                          float* d_v, float* d_scores, float* d_att_head_major,
                          float* d_att_row_major, int rows, int heads,
                          int qkv_dim) {
  auto run_attention = [&]() {
    RunDeviceAttention(handle, d_qkv, d_q, d_k, d_v, d_scores,
                       d_att_head_major, d_att_row_major, rows, heads, qkv_dim,
                       options.attention_qk_solution_index,
                       options.attention_av_solution_index);
  };
  return TimeSamples(options, run_attention);
}

struct DeviceVitLayerTimings {
  float ln0_f32_ms = 0.0f;
  float ln0_bf16_ms = 0.0f;
  float qkv_f32_ms = 0.0f;
  float qkv_bf16_ms = 0.0f;
  float attn_f32_ms = 0.0f;
  float attn_bf16_ms = 0.0f;
  float attn_out_f32_ms = 0.0f;
  float attn_out_bf16_ms = 0.0f;
  float ln1_f32_ms = 0.0f;
  float ln1_bf16_ms = 0.0f;
  float mlp_f32_ms = 0.0f;
  float mlp_bf16_ms = 0.0f;
  float mlp_up_bf16_ms = 0.0f;
  float mlp_act_bf16_ms = 0.0f;
  float mlp_down_bf16_ms = 0.0f;
  float mlp_residual_bf16_ms = 0.0f;
};

template <class Run>
void RunOrTimeBf16Stage(const PaliGemma2VitHipOptions* timing_options,
                        DeviceVitLayerTimings* timings,
                        float DeviceVitLayerTimings::*field,
                        const Run& run) {
  if (timing_options != nullptr && timings != nullptr) {
    timings->*field = TimeSamples(*timing_options, std::function<void()>(run));
    return;
  }
  run();
}

struct VitLayerBF16Workspace {
  // Per-image scratch for the no-timing BF16 return path. All ViT layers in
  // PaliGemma2 use the same dimensions today, but this is sized by maxima so
  // it stays correct if a future config varies heads/qkv/mlp across layers.
  VitLayerBF16Workspace(int rows, int model_dim, int max_qkv, int max_heads,
                        int max_qkv_dim, int max_att_cols, int max_mlp)
      : rows(rows),
        model_dim(model_dim),
        max_qkv(max_qkv),
        max_heads(max_heads),
        max_qkv_dim(max_qkv_dim),
        max_att_cols(max_att_cols),
        max_mlp(max_mlp),
        pre_att_bf16(static_cast<size_t>(rows) * model_dim),
        qkv_bf16(static_cast<size_t>(rows) * max_qkv),
        q_bf16(static_cast<size_t>(max_heads) * rows * max_qkv_dim),
        k_bf16(static_cast<size_t>(max_heads) * rows * max_qkv_dim),
        v_bf16(static_cast<size_t>(max_heads) * rows * max_qkv_dim),
        q_att_bf16(static_cast<size_t>(max_heads) * rows * max_qkv_dim),
        k_att_bf16(static_cast<size_t>(max_heads) * rows * max_qkv_dim),
        scores_bf16(static_cast<size_t>(max_heads) * rows * rows),
        attention_inv_sums(static_cast<size_t>(max_heads) * rows),
        scores_av_bf16(static_cast<size_t>(max_heads) * rows * rows),
        v_av_bf16(static_cast<size_t>(max_heads) * rows * max_qkv_dim),
        att_head_bf16(static_cast<size_t>(max_heads) * rows * max_qkv_dim),
        att_row_bf16_f32(static_cast<size_t>(rows) * max_att_cols),
        att_row_bf16(static_cast<size_t>(rows) * max_att_cols),
        x_att_bf16(static_cast<size_t>(rows) * model_dim),
        pre_ffw_bf16(static_cast<size_t>(rows) * model_dim),
        up_bf16_accum(static_cast<size_t>(rows) * max_mlp),
        up_bf16(static_cast<size_t>(rows) * max_mlp) {}

  bool Matches(int rows_in, int model_dim_in, int max_qkv_in,
               int max_heads_in, int max_qkv_dim_in, int max_att_cols_in,
               int max_mlp_in) const {
    return rows == rows_in && model_dim == model_dim_in &&
           max_qkv == max_qkv_in && max_heads == max_heads_in &&
           max_qkv_dim == max_qkv_dim_in && max_att_cols == max_att_cols_in &&
           max_mlp == max_mlp_in;
  }

  size_t TotalBytes() const {
    return pre_att_bf16.count * sizeof(rocblas_bfloat16) +
           qkv_bf16.count * sizeof(float) + q_bf16.count * sizeof(float) +
           k_bf16.count * sizeof(float) + v_bf16.count * sizeof(float) +
           q_att_bf16.count * sizeof(rocblas_bfloat16) +
           k_att_bf16.count * sizeof(rocblas_bfloat16) +
           scores_bf16.count * sizeof(float) +
           attention_inv_sums.count * sizeof(float) +
           scores_av_bf16.count * sizeof(rocblas_bfloat16) +
           v_av_bf16.count * sizeof(rocblas_bfloat16) +
           att_head_bf16.count * sizeof(float) +
           att_row_bf16_f32.count * sizeof(float) +
           att_row_bf16.count * sizeof(rocblas_bfloat16) +
           x_att_bf16.count * sizeof(float) +
           pre_ffw_bf16.count * sizeof(rocblas_bfloat16) +
           up_bf16_accum.count * sizeof(float) +
           up_bf16.count * sizeof(rocblas_bfloat16);
  }

  int rows = 0;
  int model_dim = 0;
  int max_qkv = 0;
  int max_heads = 0;
  int max_qkv_dim = 0;
  int max_att_cols = 0;
  int max_mlp = 0;
  DeviceBuffer<rocblas_bfloat16> pre_att_bf16;
  DeviceBuffer<float> qkv_bf16;
  DeviceBuffer<float> q_bf16;
  DeviceBuffer<float> k_bf16;
  DeviceBuffer<float> v_bf16;
  DeviceBuffer<rocblas_bfloat16> q_att_bf16;
  DeviceBuffer<rocblas_bfloat16> k_att_bf16;
  DeviceBuffer<float> scores_bf16;
  DeviceBuffer<float> attention_inv_sums;
  DeviceBuffer<rocblas_bfloat16> scores_av_bf16;
  DeviceBuffer<rocblas_bfloat16> v_av_bf16;
  DeviceBuffer<float> att_head_bf16;
  DeviceBuffer<float> att_row_bf16_f32;
  DeviceBuffer<rocblas_bfloat16> att_row_bf16;
  DeviceBuffer<float> x_att_bf16;
  DeviceBuffer<rocblas_bfloat16> pre_ffw_bf16;
  DeviceBuffer<float> up_bf16_accum;
  DeviceBuffer<rocblas_bfloat16> up_bf16;
};

bool RunDeviceVitLayer(rocblas_handle handle,
                       const PaliGemma2VitHipOptions& options,
                       const ResidentProjectionLayer& resident_layer,
                       const LayerWeightsPtrs& layer, int layer_idx, int rows,
                       int model_dim, const float* d_x_in_f32,
                       const float* d_x_in_bf16_f32, float* d_x_out_f32,
                       float* d_x_out_bf16_f32,
                       DeviceVitLayerTimings& timings) {
  // Reusable device-resident implementation of one PaliGemma2 ViT block:
  //   x -> LN0 -> QKV -> device attention -> attn_out + residual
  //     -> LN1 -> MLP + residual -> x_next.
  //
  // This helper intentionally keeps the F32-shadow and BF16 routes side by side.
  // The F32 route is the same device graph with widened weights; the BF16 route
  // is the candidate inference path with quantization at GEMM boundaries.
  const LayerConfig& layer_config = layer.layer_config;
  if (layer_config.activation != ActivationType::Gelu) {
    std::fprintf(stderr, "Device ViT layer runner only implements GELU.\n");
    return false;
  }

  const int heads = static_cast<int>(layer_config.heads);
  const int qkv_dim = static_cast<int>(layer_config.qkv_dim);
  const int qkv = heads * 3 * qkv_dim;
  const int att_cols = heads * qkv_dim;
  const int mlp = static_cast<int>(layer_config.ff_hidden_dim);
  HWY_ASSERT(layer_config.heads == layer_config.kv_heads);
  HWY_ASSERT(att_cols == model_dim);
  HWY_ASSERT(layer.vit.qkv_einsum_w.Rows() == static_cast<size_t>(qkv));
  HWY_ASSERT(layer.vit.qkv_einsum_w.Cols() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer.vit.attn_out_w.Rows() == static_cast<size_t>(model_dim));
  HWY_ASSERT(layer.vit.attn_out_w.Cols() == static_cast<size_t>(att_cols));
  HWY_ASSERT(layer.vit.linear_0_w.Rows() == static_cast<size_t>(mlp));
  HWY_ASSERT(layer.vit.linear_0_w.Cols() == static_cast<size_t>(model_dim));
  HWY_ASSERT(layer.vit.linear_1_w.Rows() == static_cast<size_t>(model_dim));
  HWY_ASSERT(layer.vit.linear_1_w.Cols() == static_cast<size_t>(mlp));

  char name[64];
  std::snprintf(name, sizeof(name), "prefix.layer%d.qkv", layer_idx);
  ResidentF32Mat qkv_f32 =
      UploadBF16MatAsF32RocblasB(name, layer.vit.qkv_einsum_w,
                                 options.verbose);
  std::snprintf(name, sizeof(name), "prefix.layer%d.attn_out", layer_idx);
  ResidentF32Mat attn_out_f32 =
      UploadBF16MatAsF32RocblasB(name, layer.vit.attn_out_w,
                                 options.verbose);
  std::snprintf(name, sizeof(name), "prefix.layer%d.linear_0", layer_idx);
  ResidentF32Mat linear0_f32 =
      UploadBF16MatAsF32RocblasB(name, layer.vit.linear_0_w, options.verbose);
  std::snprintf(name, sizeof(name), "prefix.layer%d.linear_1", layer_idx);
  ResidentF32Mat linear1_f32 =
      UploadBF16MatAsF32RocblasB(name, layer.vit.linear_1_w, options.verbose);

  std::vector<float> mlp_bias0(static_cast<size_t>(mlp), 0.0f);
  std::vector<float> mlp_bias1(static_cast<size_t>(model_dim), 0.0f);
  if (layer_config.ff_biases) {
    std::memcpy(mlp_bias0.data(), layer.vit.linear_0_b.PackedScale1(),
                mlp_bias0.size() * sizeof(float));
    std::memcpy(mlp_bias1.data(), layer.vit.linear_1_b.PackedScale1(),
                mlp_bias1.size() * sizeof(float));
  }

  const int threads = 256;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int model_count = rows * model_dim;
  const int model_blocks = (model_count + threads - 1) / threads;
  const int qkv_count = rows * qkv;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const int att_count = rows * att_cols;
  const int att_blocks = (att_count + threads - 1) / threads;
  const int up_count = rows * mlp;
  const int up_blocks = (up_count + threads - 1) / threads;
  const size_t layer_norm_shared =
      static_cast<size_t>(threads) * 2 * sizeof(float);

  const std::vector<float> ln0_scale =
      CopyActivationMatToF32(layer.vit.layer_norm_0_scale);
  const std::vector<float> ln0_bias =
      CopyActivationMatToF32(layer.vit.layer_norm_0_bias);
  const std::vector<float> ln1_scale =
      CopyActivationMatToF32(layer.vit.layer_norm_1_scale);
  const std::vector<float> ln1_bias =
      CopyActivationMatToF32(layer.vit.layer_norm_1_bias);
  DeviceBuffer<float> d_ln0_scale(ln0_scale.size());
  DeviceBuffer<float> d_ln0_bias(ln0_bias.size());
  DeviceBuffer<float> d_ln1_scale(ln1_scale.size());
  DeviceBuffer<float> d_ln1_bias(ln1_bias.size());
  DeviceBuffer<float> d_qkv_bias(static_cast<size_t>(qkv));
  DeviceBuffer<float> d_att_bias(static_cast<size_t>(model_dim));
  DeviceBuffer<float> d_mlp_bias0(mlp_bias0.size());
  DeviceBuffer<float> d_mlp_bias1(mlp_bias1.size());
  HIP_CHECK(hipMemcpy(d_ln0_scale.ptr, ln0_scale.data(),
                      ln0_scale.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ln0_bias.ptr, ln0_bias.data(),
                      ln0_bias.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ln1_scale.ptr, ln1_scale.data(),
                      ln1_scale.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ln1_bias.ptr, ln1_bias.data(),
                      ln1_bias.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qkv_bias.ptr, layer.vit.qkv_einsum_b.PackedScale1(),
                      static_cast<size_t>(qkv) * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_att_bias.ptr, layer.vit.attn_out_b.PackedScale1(),
                      static_cast<size_t>(model_dim) * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_mlp_bias0.ptr, mlp_bias0.data(),
                      mlp_bias0.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_mlp_bias1.ptr, mlp_bias1.data(),
                      mlp_bias1.size() * sizeof(float), hipMemcpyHostToDevice));

  DeviceBuffer<float> d_pre_att_f32(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<rocblas_bfloat16> d_pre_att_bf16(
      static_cast<size_t>(rows) * model_dim);
  auto run_ln0_f32 = [&]() {
    hipLaunchKernelGGL(LayerNormKernel, dim3(rows), dim3(threads),
                       layer_norm_shared, 0, d_x_in_f32, d_ln0_scale.ptr,
                       d_ln0_bias.ptr, d_pre_att_f32.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
  };
  auto run_ln0_bf16 = [&]() {
    // The BF16 projection path only needs the quantized normalized activation.
    // Write it directly from LayerNorm instead of staging an F32 matrix and
    // launching a second conversion kernel.
    hipLaunchKernelGGL(LayerNormToBF16Kernel, dim3(rows), dim3(threads),
                       layer_norm_shared, 0, d_x_in_bf16_f32,
                       d_ln0_scale.ptr, d_ln0_bias.ptr, d_pre_att_bf16.ptr,
                       rows, model_dim);
    HIP_CHECK(hipGetLastError());
  };
  timings.ln0_f32_ms = TimeSamples(options, run_ln0_f32);
  timings.ln0_bf16_ms = TimeSamples(options, run_ln0_bf16);

  DeviceBuffer<float> d_qkv_f32(static_cast<size_t>(rows) * qkv);
  DeviceBuffer<float> d_qkv_bf16(static_cast<size_t>(rows) * qkv);
  auto add_qkv_bias = [&](float* out) {
    hipLaunchKernelGGL(AddBiasKernel, dim3(qkv_blocks), dim3(threads), 0, 0,
                       out, d_qkv_bias.ptr, rows, qkv);
    HIP_CHECK(hipGetLastError());
  };
  auto run_qkv_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, qkv, rows,
        model_dim, &alpha, qkv_f32.device->ptr, rocblas_datatype_f32_r, qkv,
        d_pre_att_f32.ptr, rocblas_datatype_f32_r, model_dim, &beta,
        d_qkv_f32.ptr, rocblas_datatype_f32_r, qkv, d_qkv_f32.ptr,
        rocblas_datatype_f32_r, qkv, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_qkv_bias(d_qkv_f32.ptr);
  };
  auto run_qkv_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, qkv, rows,
        model_dim, &alpha, resident_layer.qkv.device->ptr,
        rocblas_datatype_bf16_r, qkv, d_pre_att_bf16.ptr,
        rocblas_datatype_bf16_r, model_dim, &beta, d_qkv_bf16.ptr,
        rocblas_datatype_f32_r, qkv, d_qkv_bf16.ptr, rocblas_datatype_f32_r,
        qkv, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
    add_qkv_bias(d_qkv_bf16.ptr);
  };
  timings.qkv_f32_ms = TimeSamples(options, run_qkv_f32);
  timings.qkv_bf16_ms = TimeSamples(options, run_qkv_bf16);

  const size_t head_count = static_cast<size_t>(heads) * rows * qkv_dim;
  const size_t scores_count = static_cast<size_t>(heads) * rows * rows;
  DeviceBuffer<float> d_q_f32(head_count);
  DeviceBuffer<float> d_k_f32(head_count);
  DeviceBuffer<float> d_v_f32(head_count);
  DeviceBuffer<float> d_scores_f32(scores_count);
  DeviceBuffer<float> d_att_head_f32(head_count);
  DeviceBuffer<float> d_att_row_f32(static_cast<size_t>(rows) * att_cols);
  DeviceBuffer<float> d_q_bf16(head_count);
  DeviceBuffer<float> d_k_bf16(head_count);
  DeviceBuffer<float> d_v_bf16(head_count);
  DeviceBuffer<float> d_scores_bf16(scores_count);
  DeviceBuffer<float> d_att_head_bf16(head_count);
  DeviceBuffer<float> d_att_row_bf16_f32(static_cast<size_t>(rows) * att_cols);
  DeviceBuffer<rocblas_bfloat16> d_att_row_bf16(
      static_cast<size_t>(rows) * att_cols);
  timings.attn_f32_ms = TimeDeviceAttention(
      handle, options, d_qkv_f32.ptr, d_q_f32.ptr, d_k_f32.ptr, d_v_f32.ptr,
      d_scores_f32.ptr, d_att_head_f32.ptr, d_att_row_f32.ptr, rows, heads,
      qkv_dim);
  timings.attn_bf16_ms = TimeDeviceAttention(
      handle, options, d_qkv_bf16.ptr, d_q_bf16.ptr, d_k_bf16.ptr,
      d_v_bf16.ptr, d_scores_bf16.ptr, d_att_head_bf16.ptr,
      d_att_row_bf16_f32.ptr, rows, heads, qkv_dim);

  DeviceBuffer<float> d_x_att_f32(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_x_att_bf16(static_cast<size_t>(rows) * model_dim);
  auto add_att_bias_and_residual_f32 = [&]() {
    hipLaunchKernelGGL(AddBiasKernel, dim3(model_blocks), dim3(threads), 0, 0,
                       d_x_att_f32.ptr, d_att_bias.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(AddResidualKernel, dim3(model_blocks), dim3(threads), 0,
                       0, d_x_att_f32.ptr, d_x_in_f32, model_count);
    HIP_CHECK(hipGetLastError());
  };
  auto add_att_bias_and_residual_bf16 = [&]() {
    hipLaunchKernelGGL(AddBiasKernel, dim3(model_blocks), dim3(threads), 0, 0,
                       d_x_att_bf16.ptr, d_att_bias.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(AddResidualKernel, dim3(model_blocks), dim3(threads), 0,
                       0, d_x_att_bf16.ptr, d_x_in_bf16_f32, model_count);
    HIP_CHECK(hipGetLastError());
  };
  auto run_attn_out_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, model_dim, rows,
        att_cols, &alpha, attn_out_f32.device->ptr, rocblas_datatype_f32_r,
        model_dim, d_att_row_f32.ptr, rocblas_datatype_f32_r, att_cols, &beta,
        d_x_att_f32.ptr, rocblas_datatype_f32_r, model_dim, d_x_att_f32.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_att_bias_and_residual_f32();
  };
  auto run_attn_out_bf16 = [&]() {
    hipLaunchKernelGGL(F32ToBF16Kernel, dim3(att_blocks), dim3(threads), 0, 0,
                       d_att_row_bf16_f32.ptr, d_att_row_bf16.ptr, att_count);
    HIP_CHECK(hipGetLastError());
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, model_dim, rows,
        att_cols, &alpha, resident_layer.attn_out.device->ptr,
        rocblas_datatype_bf16_r, model_dim, d_att_row_bf16.ptr,
        rocblas_datatype_bf16_r, att_cols, &beta, d_x_att_bf16.ptr,
        rocblas_datatype_f32_r, model_dim, d_x_att_bf16.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_att_bias_and_residual_bf16();
  };
  timings.attn_out_f32_ms = TimeSamples(options, run_attn_out_f32);
  timings.attn_out_bf16_ms = TimeSamples(options, run_attn_out_bf16);

  DeviceBuffer<float> d_pre_ffw_f32(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<rocblas_bfloat16> d_pre_ffw_bf16(
      static_cast<size_t>(rows) * model_dim);
  auto run_ln1_f32 = [&]() {
    hipLaunchKernelGGL(LayerNormKernel, dim3(rows), dim3(threads),
                       layer_norm_shared, 0, d_x_att_f32.ptr,
                       d_ln1_scale.ptr, d_ln1_bias.ptr, d_pre_ffw_f32.ptr,
                       rows, model_dim);
    HIP_CHECK(hipGetLastError());
  };
  auto run_ln1_bf16 = [&]() {
    // Same boundary as before for the MLP input: F32 LayerNorm math followed by
    // BF16 storage. The difference is only that rounding happens in the
    // producer kernel.
    hipLaunchKernelGGL(LayerNormToBF16Kernel, dim3(rows), dim3(threads),
                       layer_norm_shared, 0, d_x_att_bf16.ptr,
                       d_ln1_scale.ptr, d_ln1_bias.ptr, d_pre_ffw_bf16.ptr,
                       rows, model_dim);
    HIP_CHECK(hipGetLastError());
  };
  timings.ln1_f32_ms = TimeSamples(options, run_ln1_f32);
  timings.ln1_bf16_ms = TimeSamples(options, run_ln1_bf16);

  DeviceBuffer<float> d_up_f32(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<float> d_up_bf16_accum(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<rocblas_bfloat16> d_up_bf16(static_cast<size_t>(rows) * mlp);
  auto add_mlp_bias_and_residual_f32 = [&]() {
    hipLaunchKernelGGL(AddBiasKernel, dim3(model_blocks), dim3(threads), 0, 0,
                       d_x_out_f32, d_mlp_bias1.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(AddResidualKernel, dim3(model_blocks), dim3(threads), 0,
                       0, d_x_out_f32, d_x_att_f32.ptr, model_count);
    HIP_CHECK(hipGetLastError());
  };
  auto add_mlp_bias_and_residual_bf16 = [&]() {
    hipLaunchKernelGGL(AddBiasKernel, dim3(model_blocks), dim3(threads), 0, 0,
                       d_x_out_bf16_f32, d_mlp_bias1.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(AddResidualKernel, dim3(model_blocks), dim3(threads), 0,
                       0, d_x_out_bf16_f32, d_x_att_bf16.ptr, model_count);
    HIP_CHECK(hipGetLastError());
  };
  auto run_mlp_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, mlp, rows,
        model_dim, &alpha, linear0_f32.device->ptr, rocblas_datatype_f32_r, mlp,
        d_pre_ffw_f32.ptr, rocblas_datatype_f32_r, model_dim, &beta,
        d_up_f32.ptr, rocblas_datatype_f32_r, mlp, d_up_f32.ptr,
        rocblas_datatype_f32_r, mlp, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    hipLaunchKernelGGL(AddBiasKernel, dim3(up_blocks), dim3(threads), 0, 0,
                       d_up_f32.ptr, d_mlp_bias0.ptr, rows, mlp);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(GeluKernel, dim3(up_blocks), dim3(threads), 0, 0,
                       d_up_f32.ptr, up_count);
    HIP_CHECK(hipGetLastError());
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, model_dim, rows,
        mlp, &alpha, linear1_f32.device->ptr, rocblas_datatype_f32_r,
        model_dim, d_up_f32.ptr, rocblas_datatype_f32_r, mlp, &beta,
        d_x_out_f32, rocblas_datatype_f32_r, model_dim, d_x_out_f32,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_mlp_bias_and_residual_f32();
  };
  auto run_mlp_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, mlp, rows,
        model_dim, &alpha, resident_layer.linear_0.device->ptr,
        rocblas_datatype_bf16_r, mlp, d_pre_ffw_bf16.ptr,
        rocblas_datatype_bf16_r, model_dim, &beta, d_up_bf16_accum.ptr,
        rocblas_datatype_f32_r, mlp, d_up_bf16_accum.ptr,
        rocblas_datatype_f32_r, mlp, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    hipLaunchKernelGGL(AddBiasToBF16Kernel, dim3(up_blocks), dim3(threads), 0,
                       0, d_up_bf16_accum.ptr, d_mlp_bias0.ptr,
                       d_up_bf16.ptr, rows, mlp);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(GeluBF16Kernel, dim3(up_blocks), dim3(threads), 0, 0,
                       d_up_bf16.ptr, up_count);
    HIP_CHECK(hipGetLastError());
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none, model_dim, rows,
        mlp, &alpha, resident_layer.linear_1.device->ptr,
        rocblas_datatype_bf16_r, model_dim, d_up_bf16.ptr,
        rocblas_datatype_bf16_r, mlp, &beta, d_x_out_bf16_f32,
        rocblas_datatype_f32_r, model_dim, d_x_out_bf16_f32,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_mlp_bias_and_residual_bf16();
  };
  timings.mlp_f32_ms = TimeSamples(options, run_mlp_f32);
  timings.mlp_bf16_ms = TimeSamples(options, run_mlp_bf16);
  return true;
}

bool RunDeviceVitLayerBF16Only(rocblas_handle handle,
                               const ResidentProjectionLayer& resident_layer,
                               const LayerWeightsPtrs& layer, int rows,
                               int model_dim,
                               const float* d_x_in_bf16_f32,
                               float* d_x_out_bf16_f32,
                               VitLayerBF16Workspace& workspace,
                               const PaliGemma2VitHipOptions* timing_options =
                                   nullptr,
                               DeviceVitLayerTimings* timings = nullptr,
                               int mlp_up_solution_index = 0,
                               int mlp_down_solution_index = 0,
                               int attention_qk_solution_index = 0,
                               int attention_av_solution_index = 0,
                               bool use_attention_qk_bf16 = false,
                               bool use_attention_qk_bf16_wmma = false,
                               bool use_attention_av_bf16_wmma = false,
                               bool use_attention_av_f32_dim4 = false,
                               bool use_attention_pack_bf16 = false,
                               bool use_attention_direct_qkv = false,
                               bool use_attention_defer_softmax_scale = false,
                               bool use_qkv_wmma2d = false,
                               bool use_attn_out_wmma2d = false,
                               bool use_mlp_up_wmma2d = false,
                               bool use_mlp_down_wmma8 = false,
                               bool use_mlp_down_fused_residual = false,
                               int mlp_down_wmma_waves = 8) {
  // Candidate inference version of one ViT block. This is intentionally narrower
  // than RunDeviceVitLayer: it does not upload or execute F32-shadow projection
  // weights, and it does not insert timing events between sub-steps.
  const LayerConfig& layer_config = layer.layer_config;
  if (layer_config.activation != ActivationType::Gelu) {
    std::fprintf(stderr, "Device ViT layer runner only implements GELU.\n");
    return false;
  }

  const int heads = static_cast<int>(layer_config.heads);
  const int qkv_dim = static_cast<int>(layer_config.qkv_dim);
  const int qkv = heads * 3 * qkv_dim;
  const int att_cols = heads * qkv_dim;
  const int mlp = static_cast<int>(layer_config.ff_hidden_dim);
  HWY_ASSERT(layer_config.heads == layer_config.kv_heads);
  HWY_ASSERT(att_cols == model_dim);
  HWY_ASSERT(layer.vit.qkv_einsum_w.Rows() == static_cast<size_t>(qkv));
  HWY_ASSERT(layer.vit.qkv_einsum_w.Cols() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer.vit.attn_out_w.Rows() == static_cast<size_t>(model_dim));
  HWY_ASSERT(layer.vit.attn_out_w.Cols() == static_cast<size_t>(att_cols));
  HWY_ASSERT(layer.vit.linear_0_w.Rows() == static_cast<size_t>(mlp));
  HWY_ASSERT(layer.vit.linear_0_w.Cols() == static_cast<size_t>(model_dim));
  HWY_ASSERT(layer.vit.linear_1_w.Rows() == static_cast<size_t>(model_dim));
  HWY_ASSERT(layer.vit.linear_1_w.Cols() == static_cast<size_t>(mlp));
  HWY_ASSERT(resident_layer.ln0_scale.count == model_dim);
  HWY_ASSERT(resident_layer.ln0_bias.count == model_dim);
  HWY_ASSERT(resident_layer.ln1_scale.count == model_dim);
  HWY_ASSERT(resident_layer.ln1_bias.count == model_dim);
  HWY_ASSERT(resident_layer.qkv_bias.count == qkv);
  HWY_ASSERT(resident_layer.attn_out_bias.count == model_dim);
  HWY_ASSERT(resident_layer.linear_0_bias.count == mlp);
  HWY_ASSERT(resident_layer.linear_1_bias.count == model_dim);

  const int threads = 256;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int model_count = rows * model_dim;
  const int model_blocks = (model_count + threads - 1) / threads;
  const int qkv_count = rows * qkv;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const int att_count = rows * att_cols;
  const int att_blocks = (att_count + threads - 1) / threads;
  const int up_count = rows * mlp;
  const int up_blocks = (up_count + threads - 1) / threads;
  const size_t layer_norm_shared =
      static_cast<size_t>(threads) * 2 * sizeof(float);
  HWY_ASSERT(workspace.pre_att_bf16.count >=
             static_cast<size_t>(rows) * model_dim);
  HWY_ASSERT(workspace.qkv_bf16.count >= static_cast<size_t>(rows) * qkv);
  HWY_ASSERT(workspace.q_bf16.count >=
             static_cast<size_t>(heads) * rows * qkv_dim);
  HWY_ASSERT(workspace.q_att_bf16.count >=
             static_cast<size_t>(heads) * rows * qkv_dim);
  HWY_ASSERT(workspace.scores_bf16.count >=
             static_cast<size_t>(heads) * rows * rows);
  HWY_ASSERT(workspace.attention_inv_sums.count >=
             static_cast<size_t>(heads) * rows);
  HWY_ASSERT(workspace.scores_av_bf16.count >=
             static_cast<size_t>(heads) * rows * rows);
  HWY_ASSERT(workspace.v_av_bf16.count >=
             static_cast<size_t>(heads) * rows * qkv_dim);
  HWY_ASSERT(workspace.att_row_bf16_f32.count >=
             static_cast<size_t>(rows) * att_cols);
  if (use_attention_qk_bf16 && use_attention_qk_bf16_wmma) {
    std::fprintf(stderr,
                 "QK BF16 rocBLAS and QK BF16 WMMA are alternate "
                 "implementations of the same precision experiment; enable "
                 "only one at a time.\n");
    return false;
  }
  const bool use_attention_qk_bf16_any =
      use_attention_qk_bf16 || use_attention_qk_bf16_wmma;
  if (use_attention_qk_bf16_any && use_attention_av_bf16_wmma) {
    std::fprintf(stderr,
                 "QK BF16 and AV BF16 WMMA attention experiments are "
                 "independent opt-ins; enable only one at a time.\n");
    return false;
  }
  if (use_attention_qk_bf16_any && use_attention_av_f32_dim4) {
    std::fprintf(stderr,
                 "AV F32 dim4 is scoped to the F32 QK attention path; disable "
                 "QK BF16 attention flags.\n");
    return false;
  }
  if (use_attention_av_bf16_wmma && use_attention_av_f32_dim4) {
    std::fprintf(stderr,
                 "AV BF16 WMMA and AV F32 dim4 replace the same attention AV "
                 "stage; enable only one at a time.\n");
    return false;
  }
  if (use_attention_pack_bf16 && use_attention_av_bf16_wmma) {
    std::fprintf(stderr,
                 "Attention pack-to-BF16 is currently scoped to the F32 "
                 "attention and QK-BF16 paths; disable AV BF16 WMMA.\n");
    return false;
  }
  if (use_attention_defer_softmax_scale && !use_attention_pack_bf16) {
    std::fprintf(stderr,
                 "Deferred softmax scale requires attention pack-to-BF16 so "
                 "the saved scale can be applied at the pack boundary.\n");
    return false;
  }
  if (use_attention_direct_qkv && use_attention_qk_bf16_any) {
    std::fprintf(stderr,
                 "Direct-QKV F32 attention is scoped to the F32 QK path; "
                 "disable QK BF16 attention flags.\n");
    return false;
  }
  if (use_attention_direct_qkv && use_attention_av_bf16_wmma) {
    std::fprintf(stderr,
                 "Direct-QKV F32 attention is scoped to the F32 AV path; "
                 "disable AV BF16 WMMA.\n");
    return false;
  }
  if (use_attention_direct_qkv && use_attention_av_f32_dim4) {
    std::fprintf(stderr,
                 "Direct-QKV F32 attention and AV F32 dim4 replace the same "
                 "AV input layout; enable only one at a time.\n");
    return false;
  }
  if (use_attention_direct_qkv && use_attention_defer_softmax_scale) {
    std::fprintf(stderr,
                 "Direct-QKV F32 attention is not wired to deferred softmax "
                 "scale; enable only one scheduling experiment at a time.\n");
    return false;
  }
  if (use_attention_defer_softmax_scale && use_attention_qk_bf16_any) {
    std::fprintf(stderr,
                 "Deferred softmax scale is currently scoped to the F32 "
                 "attention path; disable QK BF16 attention flags.\n");
    return false;
  }
  if (use_attention_defer_softmax_scale && use_attention_av_f32_dim4) {
    std::fprintf(stderr,
                 "Deferred softmax scale changes the AV input normalization; "
                 "disable AV F32 dim4.\n");
    return false;
  }
  if (use_mlp_down_wmma8 &&
      !IsSupportedMlpDownWmmaWaves(mlp_down_wmma_waves)) {
    std::fprintf(stderr,
                 "Unsupported MLP-down WMMA wave count %d; use 2, 4, 6, 8, "
                 "12, or 16.\n",
                 mlp_down_wmma_waves);
    return false;
  }
  HWY_ASSERT(workspace.up_bf16_accum.count >=
             static_cast<size_t>(rows) * mlp);

  auto& d_pre_att_bf16 = workspace.pre_att_bf16;
  auto run_ln0_bf16 = [&]() {
    // F32 LayerNorm math still runs inside the kernel, but the only activation
    // consumed by the resident QKV path is BF16. Store that directly to avoid
    // materializing a full F32 pre-attention matrix per layer.
    hipLaunchKernelGGL(LayerNormToBF16Kernel, dim3(rows), dim3(threads),
                       layer_norm_shared, 0, d_x_in_bf16_f32,
                       resident_layer.ln0_scale.device->ptr,
                       resident_layer.ln0_bias.device->ptr,
                       d_pre_att_bf16.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
  };
  RunOrTimeBf16Stage(timing_options, timings,
                     &DeviceVitLayerTimings::ln0_bf16_ms, run_ln0_bf16);

  auto& d_qkv_bf16 = workspace.qkv_bf16;
  auto add_qkv_bias = [&](float* out) {
    hipLaunchKernelGGL(AddBiasKernel, dim3(qkv_blocks), dim3(threads), 0, 0,
                       out, resident_layer.qkv_bias.device->ptr, rows, qkv);
    HIP_CHECK(hipGetLastError());
  };
  auto run_qkv_bf16 = [&]() {
    if (use_qkv_wmma2d) {
      constexpr int kWavesM = 8;
      constexpr int kWavesN = 4;
      const dim3 block(32 * kWavesM * kWavesN);
      const dim3 grid((qkv + 16 * kWavesN - 1) / (16 * kWavesN),
                      (rows + 16 * kWavesM - 1) / (16 * kWavesM));
      hipLaunchKernelGGL((Bf16GemmWmma2DBiasKernel<kWavesM, kWavesN>), grid,
                         block, 0, 0, d_pre_att_bf16.ptr,
                         resident_layer.qkv.device->ptr,
                         resident_layer.qkv_bias.device->ptr,
                         d_qkv_bf16.ptr, rows, model_dim, qkv);
      HIP_CHECK(hipGetLastError());
    } else {
      ROCBLAS_CHECK(rocblas_gemm_ex(
          handle, rocblas_operation_none, rocblas_operation_none, qkv, rows,
          model_dim, &alpha, resident_layer.qkv.device->ptr,
          rocblas_datatype_bf16_r, qkv, d_pre_att_bf16.ptr,
          rocblas_datatype_bf16_r, model_dim, &beta, d_qkv_bf16.ptr,
          rocblas_datatype_f32_r, qkv, d_qkv_bf16.ptr, rocblas_datatype_f32_r,
          qkv, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
          /*solution_index=*/0, /*flags=*/0));
      add_qkv_bias(d_qkv_bf16.ptr);
    }
  };
  RunOrTimeBf16Stage(timing_options, timings,
                     &DeviceVitLayerTimings::qkv_bf16_ms, run_qkv_bf16);

  auto& d_q_bf16 = workspace.q_bf16;
  auto& d_k_bf16 = workspace.k_bf16;
  auto& d_v_bf16 = workspace.v_bf16;
  auto& d_q_att_bf16 = workspace.q_att_bf16;
  auto& d_k_att_bf16 = workspace.k_att_bf16;
  auto& d_scores_bf16 = workspace.scores_bf16;
  auto& d_attention_inv_sums = workspace.attention_inv_sums;
  auto& d_scores_av_bf16 = workspace.scores_av_bf16;
  auto& d_v_av_bf16 = workspace.v_av_bf16;
  auto& d_att_head_bf16 = workspace.att_head_bf16;
  auto& d_att_row_bf16_f32 = workspace.att_row_bf16_f32;
  auto& d_att_row_bf16 = workspace.att_row_bf16;
  auto run_attention_bf16 = [&]() {
    if (use_attention_defer_softmax_scale) {
      RunDeviceAttentionNoPackDeferredScale(
          handle, d_qkv_bf16.ptr, d_q_bf16.ptr, d_k_bf16.ptr, d_v_bf16.ptr,
          d_scores_bf16.ptr, d_attention_inv_sums.ptr,
          d_att_head_bf16.ptr, rows, heads, qkv_dim,
          attention_qk_solution_index, attention_av_solution_index);
    } else if (use_attention_pack_bf16) {
      if (use_attention_qk_bf16_wmma) {
        RunDeviceAttentionQkBf16WmmaNoPack(
            handle, d_qkv_bf16.ptr, d_q_att_bf16.ptr, d_k_att_bf16.ptr,
            d_v_bf16.ptr, d_scores_bf16.ptr, d_att_head_bf16.ptr, rows, heads,
            qkv_dim, attention_av_solution_index);
      } else if (use_attention_qk_bf16) {
        RunDeviceAttentionQkBf16NoPack(
            handle, d_qkv_bf16.ptr, d_q_att_bf16.ptr, d_k_att_bf16.ptr,
            d_v_bf16.ptr, d_scores_bf16.ptr, d_att_head_bf16.ptr, rows, heads,
            qkv_dim, attention_av_solution_index);
      } else if (use_attention_direct_qkv) {
        RunDeviceAttentionDirectQKVNoPack(
            handle, d_qkv_bf16.ptr, d_scores_bf16.ptr, d_att_head_bf16.ptr,
            rows, heads, qkv_dim, attention_qk_solution_index,
            attention_av_solution_index);
      } else if (use_attention_av_f32_dim4 &&
                 CanUseAttentionAvF32Dim4(rows, qkv_dim)) {
        RunDeviceAttentionAvF32Dim4NoPack(
            handle, d_qkv_bf16.ptr, d_q_bf16.ptr, d_k_bf16.ptr, d_v_bf16.ptr,
            d_scores_bf16.ptr, d_att_head_bf16.ptr, rows, heads, qkv_dim,
            attention_qk_solution_index);
      } else {
        RunDeviceAttentionNoPack(
            handle, d_qkv_bf16.ptr, d_q_bf16.ptr, d_k_bf16.ptr, d_v_bf16.ptr,
            d_scores_bf16.ptr, d_att_head_bf16.ptr, rows, heads, qkv_dim,
            attention_qk_solution_index, attention_av_solution_index);
      }
    } else if (use_attention_av_bf16_wmma) {
      RunDeviceAttentionAvBf16Wmma(
          handle, d_qkv_bf16.ptr, d_q_bf16.ptr, d_k_bf16.ptr, d_v_bf16.ptr,
          d_scores_bf16.ptr, d_scores_av_bf16.ptr, d_v_av_bf16.ptr,
          d_att_head_bf16.ptr, d_att_row_bf16_f32.ptr, rows, heads, qkv_dim,
          attention_qk_solution_index);
    } else if (use_attention_av_f32_dim4 &&
               CanUseAttentionAvF32Dim4(rows, qkv_dim)) {
      RunDeviceAttentionAvF32Dim4(
          handle, d_qkv_bf16.ptr, d_q_bf16.ptr, d_k_bf16.ptr, d_v_bf16.ptr,
          d_scores_bf16.ptr, d_att_head_bf16.ptr, d_att_row_bf16_f32.ptr, rows,
          heads, qkv_dim, attention_qk_solution_index);
    } else if (use_attention_qk_bf16_wmma) {
      RunDeviceAttentionQkBf16Wmma(handle, d_qkv_bf16.ptr, d_q_att_bf16.ptr,
                                   d_k_att_bf16.ptr, d_v_bf16.ptr,
                                   d_scores_bf16.ptr, d_att_head_bf16.ptr,
                                   d_att_row_bf16_f32.ptr, rows, heads,
                                   qkv_dim, attention_av_solution_index);
    } else if (use_attention_qk_bf16) {
      RunDeviceAttentionQkBf16(handle, d_qkv_bf16.ptr, d_q_att_bf16.ptr,
                               d_k_att_bf16.ptr, d_v_bf16.ptr,
                               d_scores_bf16.ptr, d_att_head_bf16.ptr,
                               d_att_row_bf16_f32.ptr, rows, heads, qkv_dim,
                               attention_av_solution_index);
    } else if (use_attention_direct_qkv) {
      RunDeviceAttentionDirectQKV(
          handle, d_qkv_bf16.ptr, d_scores_bf16.ptr, d_att_head_bf16.ptr,
          d_att_row_bf16_f32.ptr, rows, heads, qkv_dim,
          attention_qk_solution_index, attention_av_solution_index);
    } else {
      RunDeviceAttention(handle, d_qkv_bf16.ptr, d_q_bf16.ptr, d_k_bf16.ptr,
                         d_v_bf16.ptr, d_scores_bf16.ptr,
                         d_att_head_bf16.ptr, d_att_row_bf16_f32.ptr, rows,
                         heads, qkv_dim, attention_qk_solution_index,
                         attention_av_solution_index);
    }
  };
  RunOrTimeBf16Stage(timing_options, timings,
                     &DeviceVitLayerTimings::attn_bf16_ms,
                     run_attention_bf16);

  auto& d_x_att_bf16 = workspace.x_att_bf16;
  auto add_att_bias_and_residual_bf16 = [&]() {
    hipLaunchKernelGGL(AddBiasKernel, dim3(model_blocks), dim3(threads), 0, 0,
                       d_x_att_bf16.ptr,
                       resident_layer.attn_out_bias.device->ptr, rows,
                       model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(AddResidualKernel, dim3(model_blocks), dim3(threads), 0,
                       0, d_x_att_bf16.ptr, d_x_in_bf16_f32, model_count);
    HIP_CHECK(hipGetLastError());
  };
  auto run_attn_out_bf16 = [&]() {
    if (use_attention_defer_softmax_scale) {
      hipLaunchKernelGGL(PackScaledAttentionHeadsToBF16Kernel,
                         dim3(att_blocks), dim3(threads), 0, 0,
                         d_att_head_bf16.ptr, d_attention_inv_sums.ptr,
                         d_att_row_bf16.ptr, rows, heads, qkv_dim);
      HIP_CHECK(hipGetLastError());
    } else if (use_attention_pack_bf16) {
      LaunchPackAttentionHeadsToBF16(d_att_head_bf16.ptr, d_att_row_bf16.ptr,
                                     rows, heads, qkv_dim);
    } else {
      hipLaunchKernelGGL(F32ToBF16Kernel, dim3(att_blocks), dim3(threads), 0, 0,
                         d_att_row_bf16_f32.ptr, d_att_row_bf16.ptr,
                         att_count);
      HIP_CHECK(hipGetLastError());
    }
    if (use_attn_out_wmma2d) {
      constexpr int kWavesM = 8;
      constexpr int kWavesN = 4;
      const dim3 block(32 * kWavesM * kWavesN);
      const dim3 grid((model_dim + 16 * kWavesN - 1) / (16 * kWavesN),
                      (rows + 16 * kWavesM - 1) / (16 * kWavesM));
      hipLaunchKernelGGL(
          (Bf16GemmWmma2DBiasResidualKernel<kWavesM, kWavesN>), grid, block, 0,
          0, d_att_row_bf16.ptr, resident_layer.attn_out.device->ptr,
          resident_layer.attn_out_bias.device->ptr, d_x_in_bf16_f32,
          d_x_att_bf16.ptr, rows, att_cols, model_dim);
      HIP_CHECK(hipGetLastError());
    } else {
      ROCBLAS_CHECK(rocblas_gemm_ex(
          handle, rocblas_operation_none, rocblas_operation_none, model_dim,
          rows, att_cols, &alpha, resident_layer.attn_out.device->ptr,
          rocblas_datatype_bf16_r, model_dim, d_att_row_bf16.ptr,
          rocblas_datatype_bf16_r, att_cols, &beta, d_x_att_bf16.ptr,
          rocblas_datatype_f32_r, model_dim, d_x_att_bf16.ptr,
          rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
          rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
      add_att_bias_and_residual_bf16();
    }
  };
  RunOrTimeBf16Stage(timing_options, timings,
                     &DeviceVitLayerTimings::attn_out_bf16_ms,
                     run_attn_out_bf16);

  auto& d_pre_ffw_bf16 = workspace.pre_ffw_bf16;
  auto run_ln1_bf16 = [&]() {
    // Match the previous BF16 boundary for the MLP input while fusing away the
    // standalone F32ToBF16 pass.
    hipLaunchKernelGGL(LayerNormToBF16Kernel, dim3(rows), dim3(threads),
                       layer_norm_shared, 0, d_x_att_bf16.ptr,
                       resident_layer.ln1_scale.device->ptr,
                       resident_layer.ln1_bias.device->ptr,
                       d_pre_ffw_bf16.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
  };
  RunOrTimeBf16Stage(timing_options, timings,
                     &DeviceVitLayerTimings::ln1_bf16_ms, run_ln1_bf16);

  auto& d_up_bf16_accum = workspace.up_bf16_accum;
  auto& d_up_bf16 = workspace.up_bf16;
  const bool use_mlp_up_wmma2d_fused_act = use_mlp_up_wmma2d;
  const bool use_mlp_down_wmma8_fused_residual =
      use_mlp_down_wmma8 && use_mlp_down_fused_residual;
  auto run_mlp_residual_bf16 = [&]() {
    if (use_mlp_down_wmma8_fused_residual) return;
    hipLaunchKernelGGL(AddBiasKernel, dim3(model_blocks), dim3(threads), 0, 0,
                       d_x_out_bf16_f32,
                       resident_layer.linear_1_bias.device->ptr, rows,
                       model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(AddResidualKernel, dim3(model_blocks), dim3(threads), 0,
                       0, d_x_out_bf16_f32, d_x_att_bf16.ptr, model_count);
    HIP_CHECK(hipGetLastError());
  };
  auto run_mlp_up_bf16 = [&]() {
    if (use_mlp_up_wmma2d) {
      LaunchMlpUpWmma2DBiasGeluBF16(
          d_pre_ffw_bf16.ptr, resident_layer.linear_0.device->ptr,
          resident_layer.linear_0_bias.device->ptr, d_up_bf16.ptr, rows,
          model_dim, mlp);
    } else {
      const rocblas_gemm_algo algo =
          mlp_up_solution_index == 0 ? rocblas_gemm_algo_standard
                                     : rocblas_gemm_algo_solution_index;
      ROCBLAS_CHECK(rocblas_gemm_ex(
          handle, rocblas_operation_none, rocblas_operation_none, mlp, rows,
          model_dim, &alpha, resident_layer.linear_0.device->ptr,
          rocblas_datatype_bf16_r, mlp, d_pre_ffw_bf16.ptr,
          rocblas_datatype_bf16_r, model_dim, &beta, d_up_bf16_accum.ptr,
          rocblas_datatype_f32_r, mlp, d_up_bf16_accum.ptr,
          rocblas_datatype_f32_r, mlp, rocblas_datatype_f32_r, algo,
          mlp_up_solution_index, rocblas_gemm_flags_none));
    }
  };
  auto run_mlp_act_bf16 = [&]() {
    if (use_mlp_up_wmma2d_fused_act) return;
    hipLaunchKernelGGL(AddBiasGeluToBF16Kernel, dim3(up_blocks),
                       dim3(threads), 0, 0, d_up_bf16_accum.ptr,
                       resident_layer.linear_0_bias.device->ptr,
                       d_up_bf16.ptr, rows, mlp);
    HIP_CHECK(hipGetLastError());
  };
  auto run_mlp_down_bf16 = [&]() {
    if (use_mlp_down_wmma8) {
      if (use_mlp_down_wmma8_fused_residual) {
        LaunchMlpDownWmmaGroupedNBiasResidual(
            d_up_bf16.ptr, resident_layer.linear_1.device->ptr,
            resident_layer.linear_1_bias.device->ptr, d_x_att_bf16.ptr,
            d_x_out_bf16_f32, rows, mlp, model_dim, mlp_down_wmma_waves);
      } else {
        LaunchMlpDownWmmaGroupedN(
            d_up_bf16.ptr, resident_layer.linear_1.device->ptr,
            d_x_out_bf16_f32, rows, mlp, model_dim, mlp_down_wmma_waves);
      }
    } else {
      const rocblas_gemm_algo algo =
          mlp_down_solution_index == 0 ? rocblas_gemm_algo_standard
                                       : rocblas_gemm_algo_solution_index;
      ROCBLAS_CHECK(rocblas_gemm_ex(
          handle, rocblas_operation_none, rocblas_operation_none, model_dim,
          rows, mlp, &alpha, resident_layer.linear_1.device->ptr,
          rocblas_datatype_bf16_r, model_dim, d_up_bf16.ptr,
          rocblas_datatype_bf16_r, mlp, &beta, d_x_out_bf16_f32,
          rocblas_datatype_f32_r, model_dim, d_x_out_bf16_f32,
          rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r, algo,
          mlp_down_solution_index, rocblas_gemm_flags_none));
    }
  };
  if (timing_options != nullptr && timings != nullptr) {
    RunOrTimeBf16Stage(timing_options, timings,
                       &DeviceVitLayerTimings::mlp_up_bf16_ms,
                       run_mlp_up_bf16);
    if (use_mlp_up_wmma2d_fused_act) {
      timings->mlp_act_bf16_ms = 0.0f;
    } else {
      RunOrTimeBf16Stage(timing_options, timings,
                         &DeviceVitLayerTimings::mlp_act_bf16_ms,
                         run_mlp_act_bf16);
    }
    RunOrTimeBf16Stage(timing_options, timings,
                       &DeviceVitLayerTimings::mlp_down_bf16_ms,
                       run_mlp_down_bf16);
    if (use_mlp_down_wmma8_fused_residual) {
      timings->mlp_residual_bf16_ms = 0.0f;
    } else {
      RunOrTimeBf16Stage(timing_options, timings,
                         &DeviceVitLayerTimings::mlp_residual_bf16_ms,
                         run_mlp_residual_bf16);
    }
    timings->mlp_bf16_ms = timings->mlp_up_bf16_ms +
                           timings->mlp_act_bf16_ms +
                           timings->mlp_down_bf16_ms +
                           timings->mlp_residual_bf16_ms;
  } else {
    run_mlp_up_bf16();
    if (!use_mlp_up_wmma2d_fused_act) {
      run_mlp_act_bf16();
    }
    run_mlp_down_bf16();
    if (!use_mlp_down_wmma8_fused_residual) {
      run_mlp_residual_bf16();
    }
  }
  return true;
}

ResidentProjectionWeights UploadProjectionWeightsImpl(const WeightsPtrs& weights,
                                                      bool verbose,
                                                      bool upload_patch_f32,
                                                      double& elapsed) {
  const double start = hwy::platform::Now();

  // Upload the model-level patch/head projections first. The optional F32
  // shadow is deliberately limited to patch embedding because later ViT
  // projections already receive activation tensors whose precision we control.
  ResidentProjectionWeights resident{
      .patch_embed = UploadBF16MatForRocblasB(
          "img_emb_kernel", weights.vit_img_embedding_kernel, verbose),
      .patch_embed_f32 = nullptr,
      .head = UploadBF16MatForRocblasB("img_head_kernel",
                                       weights.vit_img_head_kernel, verbose),
      .patch_bias = UploadF32Vec(
          "img_emb_bias", weights.vit_img_embedding_bias.PackedScale1(),
          weights.vit_img_embedding_kernel.Rows(), verbose),
      .pos_embedding = UploadF32Vec(
          "img_pos_embedding",
          CopyActivationMatToF32(weights.vit_img_pos_embedding), verbose),
      .enc_norm_scale = UploadF32Vec(
          "enc_norm_scale",
          CopyActivationMatToF32(weights.vit_encoder_norm_scale), verbose),
      .enc_norm_bias = UploadF32Vec(
          "enc_norm_bias",
          CopyActivationMatToF32(weights.vit_encoder_norm_bias), verbose),
      .head_bias = UploadF32Vec(
          "img_head_bias", weights.vit_img_head_bias.PackedScale1(),
          weights.vit_img_head_kernel.Rows(), verbose),
      .layers = {},
  };
  if (upload_patch_f32) {
    resident.patch_embed_f32 = std::make_unique<ResidentF32Mat>(
        UploadBF16MatAsF32RocblasB("img_emb_kernel",
                                   weights.vit_img_embedding_kernel, verbose));
  }

  // Each ViT layer contributes four dense projection weights plus the small
  // LayerNorm and bias vectors required by the production return path. GELU,
  // attention softmax, residuals, and layout movement remain explicit kernels.
  resident.layers.reserve(weights.vit_layers.size());
  for (size_t i = 0; i < weights.vit_layers.size(); ++i) {
    const LayerWeightsPtrs& layer = *weights.VitLayer(i);
    const LayerConfig& layer_config = layer.layer_config;
    const size_t mlp = static_cast<size_t>(layer_config.ff_hidden_dim);
    const size_t model_dim = layer.vit.attn_out_w.Rows();
    std::vector<float> mlp_bias0(mlp, 0.0f);
    std::vector<float> mlp_bias1(model_dim, 0.0f);
    if (layer_config.ff_biases) {
      std::memcpy(mlp_bias0.data(), layer.vit.linear_0_b.PackedScale1(),
                  mlp_bias0.size() * sizeof(float));
      std::memcpy(mlp_bias1.data(), layer.vit.linear_1_b.PackedScale1(),
                  mlp_bias1.size() * sizeof(float));
    }

    char name[64];
    std::snprintf(name, sizeof(name), "layer%zu.qkv", i);
    ResidentBF16Mat qkv =
        UploadBF16MatForRocblasB(name, layer.vit.qkv_einsum_w, verbose);
    std::snprintf(name, sizeof(name), "layer%zu.attn_out", i);
    ResidentBF16Mat attn_out =
        UploadBF16MatForRocblasB(name, layer.vit.attn_out_w, verbose);
    std::snprintf(name, sizeof(name), "layer%zu.linear_0", i);
    ResidentBF16Mat linear_0 =
        UploadBF16MatForRocblasB(name, layer.vit.linear_0_w, verbose);
    std::snprintf(name, sizeof(name), "layer%zu.linear_1", i);
    ResidentBF16Mat linear_1 =
        UploadBF16MatForRocblasB(name, layer.vit.linear_1_w, verbose);
    std::snprintf(name, sizeof(name), "layer%zu.ln0_scale", i);
    ResidentF32Vec ln0_scale = UploadF32Vec(
        name, CopyActivationMatToF32(layer.vit.layer_norm_0_scale), verbose);
    std::snprintf(name, sizeof(name), "layer%zu.ln0_bias", i);
    ResidentF32Vec ln0_bias = UploadF32Vec(
        name, CopyActivationMatToF32(layer.vit.layer_norm_0_bias), verbose);
    std::snprintf(name, sizeof(name), "layer%zu.ln1_scale", i);
    ResidentF32Vec ln1_scale = UploadF32Vec(
        name, CopyActivationMatToF32(layer.vit.layer_norm_1_scale), verbose);
    std::snprintf(name, sizeof(name), "layer%zu.ln1_bias", i);
    ResidentF32Vec ln1_bias = UploadF32Vec(
        name, CopyActivationMatToF32(layer.vit.layer_norm_1_bias), verbose);
    std::snprintf(name, sizeof(name), "layer%zu.qkv_bias", i);
    ResidentF32Vec qkv_bias =
        UploadF32Vec(name, layer.vit.qkv_einsum_b.PackedScale1(),
                     layer.vit.qkv_einsum_w.Rows(), verbose);
    std::snprintf(name, sizeof(name), "layer%zu.attn_bias", i);
    ResidentF32Vec attn_out_bias =
        UploadF32Vec(name, layer.vit.attn_out_b.PackedScale1(),
                     layer.vit.attn_out_w.Rows(), verbose);
    std::snprintf(name, sizeof(name), "layer%zu.mlp_bias0", i);
    ResidentF32Vec linear_0_bias = UploadF32Vec(name, mlp_bias0, verbose);
    std::snprintf(name, sizeof(name), "layer%zu.mlp_bias1", i);
    ResidentF32Vec linear_1_bias = UploadF32Vec(name, mlp_bias1, verbose);
    resident.layers.push_back(ResidentProjectionLayer{
        .qkv = std::move(qkv),
        .attn_out = std::move(attn_out),
        .linear_0 = std::move(linear_0),
        .linear_1 = std::move(linear_1),
        .ln0_scale = std::move(ln0_scale),
        .ln0_bias = std::move(ln0_bias),
        .ln1_scale = std::move(ln1_scale),
        .ln1_bias = std::move(ln1_bias),
        .qkv_bias = std::move(qkv_bias),
        .attn_out_bias = std::move(attn_out_bias),
        .linear_0_bias = std::move(linear_0_bias),
        .linear_1_bias = std::move(linear_1_bias),
    });
  }
  HIP_CHECK(hipDeviceSynchronize());
  elapsed = hwy::platform::Now() - start;
  return resident;
}

}  // namespace

struct PaliGemma2VitHipBackend::Impl {
  explicit Impl(PaliGemma2VitHipOptions options_in) : options(options_in) {}

  PaliGemma2VitHipOptions options;
  PaliGemma2VitHipStats stats;
  rocblas_handle handle = nullptr;
  std::unique_ptr<ResidentProjectionWeights> resident;
  std::unique_ptr<VitLayerBF16Workspace> return_workspace;
  int mlp_up_solution_index = 0;
  int mlp_down_solution_index = 0;
  bool initialized = false;
};

size_t PaliGemma2VitProjectionBytes(const WeightsPtrs& weights) {
  size_t bytes = DenseBF16Bytes(weights.vit_img_embedding_kernel) +
                 DenseBF16Bytes(weights.vit_img_head_kernel);
  for (size_t i = 0; i < weights.vit_layers.size(); ++i) {
    const LayerWeightsPtrs& layer = *weights.VitLayer(i);
    bytes += DenseBF16Bytes(layer.vit.qkv_einsum_w);
    bytes += DenseBF16Bytes(layer.vit.attn_out_w);
    bytes += DenseBF16Bytes(layer.vit.linear_0_w);
    bytes += DenseBF16Bytes(layer.vit.linear_1_w);
  }
  return bytes;
}

void PrintPaliGemma2VitHipModelSummary(const ModelConfig& config,
                                       const WeightsPtrs& weights) {
  if (config.vit_config.layer_configs.empty()) {
    std::fprintf(stderr, "Model has no ViT config.\n");
    std::exit(1);
  }

  const VitConfig& vit = config.vit_config;
  const LayerConfig& layer = vit.layer_configs.front();
  std::printf("PaliGemma ViT HIP backend\n");
  std::printf("  model=%d image_size=%zu patch=%zu seq=%zu vit_dim=%zu "
              "llm_dim=%zu\n",
              static_cast<int>(config.model), static_cast<size_t>(vit.image_size),
              static_cast<size_t>(vit.patch_width),
              static_cast<size_t>(vit.seq_len),
              static_cast<size_t>(vit.model_dim),
              static_cast<size_t>(config.model_dim));
  std::printf("  layers=%zu heads=%zu qkv_dim=%zu ff_hidden=%zu\n",
              vit.layer_configs.size(), static_cast<size_t>(layer.heads),
              static_cast<size_t>(layer.qkv_dim),
              static_cast<size_t>(layer.ff_hidden_dim));

  std::printf("\nViT tensors selected for first resident-GPU projection plan:\n");
  PrintMat("img_emb_kernel", weights.vit_img_embedding_kernel);
  PrintMat("img_head_kernel", weights.vit_img_head_kernel);
  PrintMat("enc_norm_scale", weights.vit_encoder_norm_scale);
  PrintMat("enc_norm_bias", weights.vit_encoder_norm_bias);
  const LayerWeightsPtrs& first = *weights.VitLayer(0);
  PrintMat("layer0.qkv", first.vit.qkv_einsum_w);
  PrintMat("layer0.attn_out", first.vit.attn_out_w);
  PrintMat("layer0.linear_0", first.vit.linear_0_w);
  PrintMat("layer0.linear_1", first.vit.linear_1_w);

  std::printf("\nProjection weights as dense BF16: %.2f MiB\n",
              MiB(PaliGemma2VitProjectionBytes(weights)));
}

PaliGemma2VitHipBackend::PaliGemma2VitHipBackend(
    PaliGemma2VitHipOptions options)
    : impl_(new Impl(options)) {}

PaliGemma2VitHipBackend::~PaliGemma2VitHipBackend() {
  if (impl_->handle != nullptr) {
    (void)rocblas_destroy_handle(impl_->handle);
  }
}

bool PaliGemma2VitHipBackend::Initialize() {
  if (impl_->initialized) return true;

  // The local target exposes one ROCm-visible iGPU, so device 0 is the backend
  // default. Keeping device setup here avoids scattering HIP state through the
  // probe and makes a future explicit device flag straightforward.
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::fprintf(stderr, "No HIP device found.\n");
    return false;
  }
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  HIP_CHECK(hipMemGetInfo(&impl_->stats.free_bytes,
                          &impl_->stats.total_bytes));

  impl_->stats.device_name = props.name;
  impl_->stats.device_arch = props.gcnArchName;
  impl_->stats.compute_units = props.multiProcessorCount;

  ROCBLAS_CHECK(rocblas_create_handle(&impl_->handle));
  ROCBLAS_CHECK(rocblas_set_pointer_mode(impl_->handle,
                                         rocblas_pointer_mode_host));
  impl_->initialized = true;

  if (impl_->options.verbose) {
    std::printf("\nHIP device: %s arch=%s CUs=%d global=%.2f GiB free=%.2f "
                "GiB\n",
                impl_->stats.device_name.c_str(),
                impl_->stats.device_arch.c_str(),
                impl_->stats.compute_units, GiB(impl_->stats.total_bytes),
                GiB(impl_->stats.free_bytes));
  }
  return true;
}

bool PaliGemma2VitHipBackend::UploadProjectionWeights(
    const WeightsPtrs& weights) {
  if (!Initialize()) return false;
  if (impl_->resident != nullptr) return true;

  // Resident upload is a backend setup operation, not something we want on the
  // steady-state per-image path. The current CLI/probe pays it every process
  // run because it constructs a fresh backend; the future integrated backend
  // should reuse this object across images or turns.
  impl_->stats.resident_projection_bytes =
      PaliGemma2VitProjectionBytes(weights);
  if (impl_->options.verbose) {
    std::printf("Projection BF16 residency budget: %.3f GiB\n",
                GiB(impl_->stats.resident_projection_bytes));
    std::printf("\nUploading real projection weights as rocBLAS B[K,N]:\n");
  }

  double elapsed = 0.0;
  const bool upload_patch_f32 =
      impl_->options.validate_patch_embedding ||
      impl_->options.validate_layer0_block_device_attention ||
      impl_->options.validate_layer0_block_device_norm ||
      impl_->options.validate_layer_prefix2_device_norm ||
      impl_->options.validate_layer_stack_device_norm ||
      impl_->options.validate_image_tokens_device_norm;
  impl_->resident = std::make_unique<ResidentProjectionWeights>(
      UploadProjectionWeightsImpl(weights, impl_->options.verbose,
                                  upload_patch_f32, elapsed));
  const size_t total_bytes = impl_->resident->TotalBytes();
  impl_->stats.upload_seconds = elapsed;
  impl_->stats.upload_gib_per_second = GiB(total_bytes) / elapsed;
  if (impl_->options.verbose) {
    std::printf("Resident upload total: %.2f MiB in %.3f s (%.2f GiB/s)\n",
                MiB(total_bytes), elapsed, impl_->stats.upload_gib_per_second);
  }
  return true;
}

bool PaliGemma2VitHipBackend::RunModelProjectionSchedule(
    const ModelConfig& config) {
  if (!Initialize()) return false;

  // Dummy-buffer schedule. It uses the loaded model config for dimensions, but
  // allocates synthetic A/B/C matrices. Use this to separate raw rocBLAS shape
  // throughput from weight upload, host packing, and real memory locality.
  const VitConfig& vit = config.vit_config;
  const LayerConfig& layer = vit.layer_configs.front();

  const int seq = static_cast<int>(vit.seq_len);
  const int model_dim = static_cast<int>(vit.model_dim);
  const int patch = static_cast<int>(vit.patch_width * vit.patch_width * 3);
  const int qkv = static_cast<int>(layer.heads * 3 * layer.qkv_dim);
  const int mlp = static_cast<int>(layer.ff_hidden_dim);
  const int layers = static_cast<int>(vit.layer_configs.size());
  const int llm_dim = static_cast<int>(config.model_dim);

  const Shape patch_embed = {"patch_embed", seq, patch, model_dim};
  const Shape qkv_proj = {"qkv", seq, model_dim, qkv};
  const Shape attn_out = {"attn_out", seq, model_dim, model_dim};
  const Shape mlp_up = {"mlp_up", seq, model_dim, mlp};
  const Shape mlp_down = {"mlp_down", seq, mlp, model_dim};
  const Shape head = {"img_head", seq, model_dim, llm_dim};

  const size_t max_a = static_cast<size_t>(seq) * HWY_MAX(model_dim, mlp);
  const size_t max_b =
      static_cast<size_t>(HWY_MAX(model_dim, mlp)) * HWY_MAX(qkv, llm_dim);
  const size_t max_c =
      static_cast<size_t>(seq) * HWY_MAX(qkv, HWY_MAX(mlp, llm_dim));
  DeviceBuffer<rocblas_bfloat16> a(max_a);
  DeviceBuffer<rocblas_bfloat16> b(max_b);
  DeviceBuffer<rocblas_bfloat16> c(max_c);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto run_bf16 = [&](const Shape& shape) {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, shape.n,
        shape.m, shape.k, &alpha, b.ptr, rocblas_datatype_bf16_r, shape.n,
        a.ptr, rocblas_datatype_bf16_r, shape.k, &beta, c.ptr,
        rocblas_datatype_bf16_r, shape.n, c.ptr, rocblas_datatype_bf16_r,
        shape.n, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
  };

  auto run_projection = [&]() {
    // Projection-only ViT skeleton. This intentionally excludes layernorm,
    // GELU, attention softmax/weighted sum, residual adds, and CPU/GPU transfer,
    // so the result is a GEMM lower-bound probe rather than image-token latency.
    run_bf16(patch_embed);
    for (int i = 0; i < layers; ++i) {
      run_bf16(qkv_proj);
      run_bf16(attn_out);
      run_bf16(mlp_up);
      run_bf16(mlp_down);
    }
    run_bf16(head);
  };

  for (int i = 0; i < impl_->options.warmup; ++i) {
    for (int j = 0; j < impl_->options.iters; ++j) run_projection();
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  std::vector<float> ms;
  ms.reserve(impl_->options.samples);
  for (int sample = 0; sample < impl_->options.samples; ++sample) {
    HIP_CHECK(hipEventRecord(start));
    for (int iter = 0; iter < impl_->options.iters; ++iter) run_projection();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    ms.push_back(elapsed_ms / static_cast<float>(impl_->options.iters));
  }
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  const double flops =
      ShapeFlops(patch_embed) +
      layers * (ShapeFlops(qkv_proj) + ShapeFlops(attn_out) +
                ShapeFlops(mlp_up) + ShapeFlops(mlp_down)) +
      ShapeFlops(head);
  impl_->stats.model_projection_schedule =
      PrintTiming("model_projection_schedule", ms, flops);
  return true;
}

bool PaliGemma2VitHipBackend::RunResidentProjectionSchedule(
    const ModelConfig& config) {
  if (!Initialize()) return false;
  if (impl_->resident == nullptr) {
    std::fprintf(stderr, "Resident projection weights have not been uploaded.\n");
    return false;
  }

  const int seq = static_cast<int>(config.vit_config.seq_len);
  const LayerConfig& layer = config.vit_config.layer_configs.front();
  const int model_dim = static_cast<int>(config.vit_config.model_dim);
  const int mlp = static_cast<int>(layer.ff_hidden_dim);
  const int qkv = static_cast<int>(layer.heads * 3 * layer.qkv_dim);
  const int llm_dim = static_cast<int>(config.model_dim);

  const size_t max_a = static_cast<size_t>(seq) * HWY_MAX(model_dim, mlp);
  const size_t max_c =
      static_cast<size_t>(seq) * HWY_MAX(qkv, HWY_MAX(mlp, llm_dim));
  DeviceBuffer<rocblas_bfloat16> a(max_a);
  DeviceBuffer<rocblas_bfloat16> c(max_c);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto run_bf16 = [&](const ResidentBF16Mat& b) {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, b.n, seq,
        b.k, &alpha, b.device->ptr, rocblas_datatype_bf16_r, b.n, a.ptr,
        rocblas_datatype_bf16_r, b.k, &beta, c.ptr, rocblas_datatype_bf16_r,
        b.n, c.ptr, rocblas_datatype_bf16_r, b.n, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
  };

  auto run_projection = [&]() {
    // Same projection skeleton as RunModelProjectionSchedule, but the B operand
    // now points at real resident weights. A and C are still scratch buffers
    // because this path is measuring the resident weight schedule, not yet
    // executing the full ViT graph.
    run_bf16(impl_->resident->patch_embed);
    for (const ResidentProjectionLayer& layer : impl_->resident->layers) {
      run_bf16(layer.qkv);
      run_bf16(layer.attn_out);
      run_bf16(layer.linear_0);
      run_bf16(layer.linear_1);
    }
    run_bf16(impl_->resident->head);
  };

  for (int i = 0; i < impl_->options.warmup; ++i) {
    for (int j = 0; j < impl_->options.iters; ++j) run_projection();
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  std::vector<float> ms;
  ms.reserve(impl_->options.samples);
  for (int sample = 0; sample < impl_->options.samples; ++sample) {
    HIP_CHECK(hipEventRecord(start));
    for (int iter = 0; iter < impl_->options.iters; ++iter) run_projection();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    ms.push_back(elapsed_ms / static_cast<float>(impl_->options.iters));
  }
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  auto mat_flops = [seq](const ResidentBF16Mat& mat) {
    return 2.0 * static_cast<double>(seq) * mat.k * mat.n;
  };
  double flops = mat_flops(impl_->resident->patch_embed) +
                 mat_flops(impl_->resident->head);
  for (const ResidentProjectionLayer& layer : impl_->resident->layers) {
    flops += mat_flops(layer.qkv);
    flops += mat_flops(layer.attn_out);
    flops += mat_flops(layer.linear_0);
    flops += mat_flops(layer.linear_1);
  }
  impl_->stats.resident_projection_schedule =
      PrintTiming("resident_projection_schedule", ms, flops);
  return true;
}

bool PaliGemma2VitHipBackend::ValidatePatchEmbedding(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image) {
  // This is the first real-tensor validation boundary for the HIP backend. It
  // mirrors gemma/vit.cc::EmbedImagePatches up to the output activations.x:
  //   1. extract image patches on CPU, matching the current preprocessing path;
  //   2. compute patch GEMM on GPU;
  //   3. add embedding bias and position embedding on GPU;
  //   4. copy back and compare against a scalar CPU reference.
  //
  // Two precision routes are measured:
  // - F32 shadow: exact semantic bridge for current CPU F32 patches.
  // - BF16 A/B: faster rocBLAS path, but with activation rounding that must be
  //   validated at later model-level boundaries.
  if (!UploadProjectionWeights(weights)) return false;
  if (impl_->resident->patch_embed_f32 == nullptr) {
    std::fprintf(stderr, "Patch embedding F32 shadow weight is not resident.\n");
    return false;
  }

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  const int cols = static_cast<int>(model_config.vit_config.model_dim);

  std::vector<float> patches(static_cast<size_t>(rows) * patch);
  for (int row = 0; row < rows; ++row) {
    image.GetPatch(static_cast<size_t>(row),
                   patches.data() + static_cast<size_t>(row) * patch);
  }

  // BF16 activation copy for the fast path. We intentionally keep the original
  // F32 patches for the reference and F32-shadow path so the precision delta is
  // isolated to this conversion.
  std::vector<rocblas_bfloat16> patches_bf16(patches.size());
  static_assert(sizeof(BF16) == sizeof(rocblas_bfloat16),
                "BF16 storage size must match rocBLAS BF16");
  for (size_t i = 0; i < patches.size(); ++i) {
    const BF16 value = hwy::BF16FromF32(patches[i]);
    std::memcpy(patches_bf16.data() + i, &value, sizeof(value));
  }
  const std::vector<float> pos =
      CopyPosEmbeddingToF32(weights.vit_img_pos_embedding);
  HWY_ASSERT(pos.size() == static_cast<size_t>(rows) * cols);

  DeviceBuffer<float> d_patches(patches.size());
  DeviceBuffer<rocblas_bfloat16> d_patches_bf16(patches_bf16.size());
  DeviceBuffer<float> d_out(static_cast<size_t>(rows) * cols);
  DeviceBuffer<float> d_out_bf16(static_cast<size_t>(rows) * cols);
  DeviceBuffer<float> d_bias(static_cast<size_t>(cols));
  DeviceBuffer<float> d_pos(pos.size());
  HIP_CHECK(hipMemcpy(d_patches.ptr, patches.data(),
                      patches.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_patches_bf16.ptr, patches_bf16.data(),
                      patches_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));
  const float* bias = weights.vit_img_embedding_bias.PackedScale1();
  HIP_CHECK(hipMemcpy(d_bias.ptr, bias,
                      static_cast<size_t>(cols) * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pos.ptr, pos.data(), pos.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  const int threads = 256;
  const int count = rows * cols;
  const int blocks = (count + threads - 1) / threads;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Shared epilogue for both precision paths. Bias and position embeddings are
  // kept in F32 because this is what the CPU path effectively accumulates into
  // after the matmul result.
  auto add_bias_and_pos = [&](float* out) {
    hipLaunchKernelGGL(AddBiasAndPosKernel, dim3(blocks), dim3(threads), 0, 0,
                       out, d_bias.ptr, d_pos.ptr, rows, cols);
    HIP_CHECK(hipGetLastError());
  };

  // Exact bridge: F32 patches multiplied by the F32 shadow weight. This path is
  // not the final memory-efficient design, but it verifies the layout and
  // epilogue without introducing activation precision changes.
  auto run_patch_embedding_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, cols,
        rows, patch, &alpha, impl_->resident->patch_embed_f32->device->ptr,
        rocblas_datatype_f32_r, cols, d_patches.ptr, rocblas_datatype_f32_r,
        patch, &beta, d_out.ptr, rocblas_datatype_f32_r, cols, d_out.ptr,
        rocblas_datatype_f32_r, cols, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_bias_and_pos(d_out.ptr);
  };

  // Fast path candidate: BF16 patches multiplied by the resident BF16 weight,
  // with F32 accumulation/output. This is what we want if model-level tolerance
  // accepts the activation rounding.
  auto run_patch_embedding_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, cols,
        rows, patch, &alpha, impl_->resident->patch_embed.device->ptr,
        rocblas_datatype_bf16_r, cols, d_patches_bf16.ptr,
        rocblas_datatype_bf16_r, patch, &beta, d_out_bf16.ptr,
        rocblas_datatype_f32_r, cols, d_out_bf16.ptr, rocblas_datatype_f32_r,
        cols, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
    add_bias_and_pos(d_out_bf16.ptr);
  };

  impl_->stats.patch_embedding_ms =
      TimeSamples(impl_->options, run_patch_embedding_f32);
  impl_->stats.patch_embedding_bf16_ms =
      TimeSamples(impl_->options, run_patch_embedding_bf16);

  std::vector<float> gpu(static_cast<size_t>(rows) * cols);
  std::vector<float> gpu_bf16(static_cast<size_t>(rows) * cols);
  HIP_CHECK(hipMemcpy(gpu.data(), d_out.ptr, gpu.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gpu_bf16.data(), d_out_bf16.ptr,
                      gpu_bf16.size() * sizeof(float), hipMemcpyDeviceToHost));

  const MatPtrT<BF16> weight(weights.vit_img_embedding_kernel);
  std::vector<float> reference(static_cast<size_t>(rows) * cols);
  // Scalar reference intentionally avoids CallMatMul. It is slower, but it
  // makes the validation independent of CPU matmul autotuning and confirms the
  // exact row/column mapping used by the HIP GEMM.
  for (int row = 0; row < rows; ++row) {
    const float* patch_row = patches.data() + static_cast<size_t>(row) * patch;
    for (int col = 0; col < cols; ++col) {
      const BF16* weight_row = weight.Row(static_cast<size_t>(col));
      float value = bias[col];
      for (int k = 0; k < patch; ++k) {
        value += patch_row[k] * hwy::F32FromBF16(weight_row[k]);
      }
      value += pos[static_cast<size_t>(row) * cols + col];
      reference[static_cast<size_t>(row) * cols + col] = value;
    }
  }

  auto compute_error = [&](const std::vector<float>& actual) {
    ErrorStats stats;
    double squared_error = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
      const float error = std::abs(actual[i] - reference[i]);
      stats.max_abs = HWY_MAX(stats.max_abs, error);
      squared_error += static_cast<double>(error) * error;
    }
    stats.rms =
        static_cast<float>(std::sqrt(squared_error / reference.size()));
    return stats;
  };
  const ErrorStats f32_error = compute_error(gpu);
  const ErrorStats bf16_error = compute_error(gpu_bf16);

  impl_->stats.patch_embedding_validated = true;
  impl_->stats.patch_embedding_max_abs_error = f32_error.max_abs;
  impl_->stats.patch_embedding_rms_error = f32_error.rms;
  impl_->stats.patch_embedding_bf16_max_abs_error = bf16_error.max_abs;
  impl_->stats.patch_embedding_bf16_rms_error = bf16_error.rms;
  if (impl_->options.verbose) {
    std::printf("patch_embedding_f32          time=%8.3f ms max_abs=%9.6f "
                "rms=%9.6f\n",
                impl_->stats.patch_embedding_ms,
                impl_->stats.patch_embedding_max_abs_error,
                impl_->stats.patch_embedding_rms_error);
    std::printf("patch_embedding_bf16         time=%8.3f ms max_abs=%9.6f "
                "rms=%9.6f\n",
                impl_->stats.patch_embedding_bf16_ms,
                impl_->stats.patch_embedding_bf16_max_abs_error,
                impl_->stats.patch_embedding_bf16_rms_error);
  }
  return true;
}

bool PaliGemma2VitHipBackend::ValidateLayer0QKV(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image) {
  // Second real-tensor boundary. QKV is the first large ViT projection after
  // patch embedding and LayerNorm:
  //   patch embedding -> layer0 LayerNorm -> layer0 QKV projection.
  //
  // LayerNorm is computed on the host with the same formula as ops::LayerNorm
  // so this validation stays focused on the projection precision/layout choice.
  // The F32-shadow QKV output is treated as the reference for the BF16 fast path;
  // a cheap row-0 scalar spot check guards the F32 route's layout.
  if (!UploadProjectionWeights(weights)) return false;

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int model_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  const LayerWeightsPtrs& layer0 = *weights.VitLayer(0);
  const LayerConfig& layer_config = layer0.layer_config;
  const int qkv =
      static_cast<int>(layer_config.heads * 3 * layer_config.qkv_dim);
  HWY_ASSERT(layer0.vit.qkv_einsum_w.Rows() == static_cast<size_t>(qkv));
  HWY_ASSERT(layer0.vit.qkv_einsum_w.Cols() ==
             static_cast<size_t>(model_dim));

  const std::vector<float> patches = BuildImagePatches(image, rows, patch);
  const std::vector<float> patch_embedding =
      BuildPatchEmbeddingReference(model_config, weights, patches);
  const std::vector<float> pre_att = LayerNormReference(
      patch_embedding, rows, model_dim, layer0.vit.layer_norm_0_scale,
      layer0.vit.layer_norm_0_bias);
  const std::vector<rocblas_bfloat16> pre_att_bf16 =
      F32ToRocblasBF16(pre_att);

  ResidentF32Mat qkv_f32 =
      UploadBF16MatAsF32RocblasB("layer0.qkv", layer0.vit.qkv_einsum_w,
                                 impl_->options.verbose);

  DeviceBuffer<float> d_pre_att(pre_att.size());
  DeviceBuffer<rocblas_bfloat16> d_pre_att_bf16(pre_att_bf16.size());
  DeviceBuffer<float> d_out_f32(static_cast<size_t>(rows) * qkv);
  DeviceBuffer<float> d_out_bf16(static_cast<size_t>(rows) * qkv);
  DeviceBuffer<float> d_bias(static_cast<size_t>(qkv));
  HIP_CHECK(hipMemcpy(d_pre_att.ptr, pre_att.data(),
                      pre_att.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pre_att_bf16.ptr, pre_att_bf16.data(),
                      pre_att_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));
  const float* bias = layer0.vit.qkv_einsum_b.PackedScale1();
  HIP_CHECK(hipMemcpy(d_bias.ptr, bias, static_cast<size_t>(qkv) * sizeof(float),
                      hipMemcpyHostToDevice));

  const int threads = 256;
  const int count = rows * qkv;
  const int blocks = (count + threads - 1) / threads;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto add_qkv_bias = [&](float* out) {
    hipLaunchKernelGGL(AddBiasKernel, dim3(blocks), dim3(threads), 0, 0, out,
                       d_bias.ptr, rows, qkv);
    HIP_CHECK(hipGetLastError());
  };
  auto run_qkv_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, qkv,
        rows, model_dim, &alpha, qkv_f32.device->ptr, rocblas_datatype_f32_r,
        qkv, d_pre_att.ptr, rocblas_datatype_f32_r, model_dim, &beta,
        d_out_f32.ptr, rocblas_datatype_f32_r, qkv, d_out_f32.ptr,
        rocblas_datatype_f32_r, qkv, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_qkv_bias(d_out_f32.ptr);
  };
  auto run_qkv_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, qkv,
        rows, model_dim, &alpha, impl_->resident->layers[0].qkv.device->ptr,
        rocblas_datatype_bf16_r, qkv, d_pre_att_bf16.ptr,
        rocblas_datatype_bf16_r, model_dim, &beta, d_out_bf16.ptr,
        rocblas_datatype_f32_r, qkv, d_out_bf16.ptr, rocblas_datatype_f32_r,
        qkv, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
    add_qkv_bias(d_out_bf16.ptr);
  };

  impl_->stats.layer0_qkv_f32_ms =
      TimeSamples(impl_->options, run_qkv_f32);
  impl_->stats.layer0_qkv_bf16_ms =
      TimeSamples(impl_->options, run_qkv_bf16);

  std::vector<float> out_f32(static_cast<size_t>(rows) * qkv);
  std::vector<float> out_bf16(static_cast<size_t>(rows) * qkv);
  HIP_CHECK(hipMemcpy(out_f32.data(), d_out_f32.ptr,
                      out_f32.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(out_bf16.data(), d_out_bf16.ptr,
                      out_bf16.size() * sizeof(float), hipMemcpyDeviceToHost));

  const MatPtrT<BF16> qkv_weight(layer0.vit.qkv_einsum_w);
  float f32_spot_max_abs = 0.0f;
  for (int col = 0; col < qkv; ++col) {
    const BF16* weight_row = qkv_weight.Row(static_cast<size_t>(col));
    float reference = bias[col];
    for (int k = 0; k < model_dim; ++k) {
      reference += pre_att[k] * BF16ToF32(weight_row[k]);
    }
    f32_spot_max_abs = HWY_MAX(f32_spot_max_abs,
                               std::abs(out_f32[col] - reference));
  }

  const ErrorStats bf16_error = ComputeError(out_bf16, out_f32);
  impl_->stats.layer0_qkv_validated = true;
  impl_->stats.layer0_qkv_f32_spot_max_abs_error = f32_spot_max_abs;
  impl_->stats.layer0_qkv_bf16_max_abs_error = bf16_error.max_abs;
  impl_->stats.layer0_qkv_bf16_rms_error = bf16_error.rms;
  if (impl_->options.verbose) {
    std::printf("layer0_qkv_f32              time=%8.3f ms "
                "spot_max_abs=%9.6f\n",
                impl_->stats.layer0_qkv_f32_ms,
                impl_->stats.layer0_qkv_f32_spot_max_abs_error);
    std::printf("layer0_qkv_bf16             time=%8.3f ms "
                "max_abs_vs_f32=%9.6f rms_vs_f32=%9.6f\n",
                impl_->stats.layer0_qkv_bf16_ms,
                impl_->stats.layer0_qkv_bf16_max_abs_error,
                impl_->stats.layer0_qkv_bf16_rms_error);
  }
  return true;
}

bool PaliGemma2VitHipBackend::ValidateLayer0MLP(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image) {
  // Third real-tensor boundary. This validates the ViT MLP projection pair:
  //   layernormed image stream -> linear_0 -> GELU -> linear_1.
  //
  // This is deliberately not a full layer0 output check yet. The true CPU graph
  // computes attention, adds the attention residual into x, then applies
  // layer_norm_1 before the MLP. Until attention/residual movement is resident,
  // we feed layer_norm_1 with the patch-embedding stream so this probe stays
  // focused on the expensive MLP up/down GEMMs and the BF16 quantization points.
  if (!UploadProjectionWeights(weights)) return false;

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int model_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  const LayerWeightsPtrs& layer0 = *weights.VitLayer(0);
  const LayerConfig& layer_config = layer0.layer_config;
  if (layer_config.activation != ActivationType::Gelu) {
    std::fprintf(stderr, "Layer0 MLP validation only implements GELU.\n");
    return false;
  }

  // Shape contract for the two ViT MLP projections:
  //   linear_0: [mlp, model_dim] consumes normalized image tokens.
  //   linear_1: [model_dim, mlp] consumes GELU(linear_0(...)).
  // These assertions protect the row-major-to-rocBLAS B[K,N] mapping below.
  const int mlp = static_cast<int>(layer_config.ff_hidden_dim);
  HWY_ASSERT(layer0.vit.linear_0_w.Rows() == static_cast<size_t>(mlp));
  HWY_ASSERT(layer0.vit.linear_0_w.Cols() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.linear_1_w.Rows() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.linear_1_w.Cols() == static_cast<size_t>(mlp));

  // Deterministic host input for the projection boundary. We reuse the scalar
  // patch embedding reference and layer_norm_1 so the device work starts from a
  // real PaliGemma2 activation tensor, while avoiding a slow host-side full
  // attention reference inside every MLP timing run.
  const std::vector<float> patches = BuildImagePatches(image, rows, patch);
  const std::vector<float> patch_embedding =
      BuildPatchEmbeddingReference(model_config, weights, patches);
  const std::vector<float> pre_ffw = LayerNormReference(
      patch_embedding, rows, model_dim, layer0.vit.layer_norm_1_scale,
      layer0.vit.layer_norm_1_bias);
  const std::vector<rocblas_bfloat16> pre_ffw_bf16 =
      F32ToRocblasBF16(pre_ffw);

  // ViT has FFN biases for PaliGemma2, but keep the zero-bias path explicit so
  // this validation remains shape-correct if another ViT config disables them.
  std::vector<float> bias0(static_cast<size_t>(mlp), 0.0f);
  std::vector<float> bias1(static_cast<size_t>(model_dim), 0.0f);
  if (layer_config.ff_biases) {
    std::memcpy(bias0.data(), layer0.vit.linear_0_b.PackedScale1(),
                bias0.size() * sizeof(float));
    std::memcpy(bias1.data(), layer0.vit.linear_1_b.PackedScale1(),
                bias1.size() * sizeof(float));
  }

  // The resident copies are BF16 only. For correctness isolation, upload
  // transient F32 shadows of the two MLP weights so the reference route differs
  // from the BF16 route mainly by activation/weight precision and quantization
  // points, not by layout or epilogue.
  ResidentF32Mat linear0_f32 =
      UploadBF16MatAsF32RocblasB("layer0.linear_0",
                                 layer0.vit.linear_0_w,
                                 impl_->options.verbose);
  ResidentF32Mat linear1_f32 =
      UploadBF16MatAsF32RocblasB("layer0.linear_1",
                                 layer0.vit.linear_1_w,
                                 impl_->options.verbose);

  // Device buffers mirror the two precision routes:
  // - d_pre_ffw/d_up_f32/d_out_f32 are the F32-shadow reference route.
  // - d_pre_ffw_bf16/d_up_bf16_accum/d_up_bf16/d_out_bf16 are the resident
  //   BF16 route, with a temporary F32 accumulator before quantizing C1.
  DeviceBuffer<float> d_pre_ffw(pre_ffw.size());
  DeviceBuffer<rocblas_bfloat16> d_pre_ffw_bf16(pre_ffw_bf16.size());
  DeviceBuffer<float> d_up_f32(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<float> d_up_bf16_accum(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<rocblas_bfloat16> d_up_bf16(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<float> d_out_f32(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_out_bf16(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_bias0(bias0.size());
  DeviceBuffer<float> d_bias1(bias1.size());
  HIP_CHECK(hipMemcpy(d_pre_ffw.ptr, pre_ffw.data(),
                      pre_ffw.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pre_ffw_bf16.ptr, pre_ffw_bf16.data(),
                      pre_ffw_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_bias0.ptr, bias0.data(), bias0.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_bias1.ptr, bias1.data(), bias1.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  const int threads = 256;
  const int up_count = rows * mlp;
  const int up_blocks = (up_count + threads - 1) / threads;
  const int out_count = rows * model_dim;
  const int out_blocks = (out_count + threads - 1) / threads;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Both MLP routes end in a F32 ffw_out tensor in gemma.cpp, so the output
  // bias is added in F32 after linear_1.
  auto add_mlp_bias = [&](float* out) {
    hipLaunchKernelGGL(AddBiasKernel, dim3(out_blocks), dim3(threads), 0, 0,
                       out, d_bias1.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
  };

  // F32-shadow route: F32 activations x F32-shadow weights, F32 GELU, then F32
  // output. This is not the intended fast path; it is the layout/reference path
  // used to quantify the BF16 route's error.
  auto run_mlp_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, mlp,
        rows, model_dim, &alpha, linear0_f32.device->ptr,
        rocblas_datatype_f32_r, mlp, d_pre_ffw.ptr, rocblas_datatype_f32_r,
        model_dim, &beta, d_up_f32.ptr, rocblas_datatype_f32_r, mlp,
        d_up_f32.ptr, rocblas_datatype_f32_r, mlp, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    hipLaunchKernelGGL(AddBiasKernel, dim3(up_blocks), dim3(threads), 0, 0,
                       d_up_f32.ptr, d_bias0.ptr, rows, mlp);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(GeluKernel, dim3(up_blocks), dim3(threads), 0, 0,
                       d_up_f32.ptr, up_count);
    HIP_CHECK(hipGetLastError());
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, mlp, &alpha, linear1_f32.device->ptr,
        rocblas_datatype_f32_r, model_dim, d_up_f32.ptr,
        rocblas_datatype_f32_r, mlp, &beta, d_out_f32.ptr,
        rocblas_datatype_f32_r, model_dim, d_out_f32.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_mlp_bias(d_out_f32.ptr);
  };

  // BF16 route: BF16 activations x resident BF16 weights with F32 accumulation,
  // then explicit BF16 C1 quantization and BF16 GELU input before linear_1.
  // This is closer to the CPU ViT tensor contract and to the future resident
  // GPU backend than keeping the large hidden tensor in F32.
  auto run_mlp_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, mlp,
        rows, model_dim, &alpha, impl_->resident->layers[0].linear_0.device->ptr,
        rocblas_datatype_bf16_r, mlp, d_pre_ffw_bf16.ptr,
        rocblas_datatype_bf16_r, model_dim, &beta, d_up_bf16_accum.ptr,
        rocblas_datatype_f32_r, mlp, d_up_bf16_accum.ptr,
        rocblas_datatype_f32_r, mlp, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    hipLaunchKernelGGL(AddBiasToBF16Kernel, dim3(up_blocks), dim3(threads), 0,
                       0, d_up_bf16_accum.ptr, d_bias0.ptr, d_up_bf16.ptr,
                       rows, mlp);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(GeluBF16Kernel, dim3(up_blocks), dim3(threads), 0, 0,
                       d_up_bf16.ptr, up_count);
    HIP_CHECK(hipGetLastError());
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, mlp, &alpha,
        impl_->resident->layers[0].linear_1.device->ptr,
        rocblas_datatype_bf16_r, model_dim, d_up_bf16.ptr,
        rocblas_datatype_bf16_r, mlp, &beta, d_out_bf16.ptr,
        rocblas_datatype_f32_r, model_dim, d_out_bf16.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_mlp_bias(d_out_bf16.ptr);
  };

  // Time the two routes independently with the shared HIP-event timing helper.
  // Each timed iteration overwrites its GEMM outputs, so no explicit clearing is
  // required between samples.
  impl_->stats.layer0_mlp_f32_ms =
      TimeSamples(impl_->options, run_mlp_f32);
  impl_->stats.layer0_mlp_bf16_ms =
      TimeSamples(impl_->options, run_mlp_bf16);

  // Copy only final MLP outputs back. Intermediate hidden tensors stay on the
  // device to preserve the same transfer boundary the final backend should use.
  std::vector<float> out_f32(static_cast<size_t>(rows) * model_dim);
  std::vector<float> out_bf16(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(out_f32.data(), d_out_f32.ptr,
                      out_f32.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(out_bf16.data(), d_out_bf16.ptr,
                      out_bf16.size() * sizeof(float), hipMemcpyDeviceToHost));

  // Cheap scalar spot check for the F32-shadow route. Computing the whole MLP
  // on CPU would be slow and redundant; row 0 is enough to catch swapped
  // dimensions, wrong leading dimensions, missed biases, or GELU placement.
  const MatPtrT<BF16> linear0_weight(layer0.vit.linear_0_w);
  const MatPtrT<BF16> linear1_weight(layer0.vit.linear_1_w);
  std::vector<float> up_row0(static_cast<size_t>(mlp));
  for (int hidden = 0; hidden < mlp; ++hidden) {
    const BF16* weight_row = linear0_weight.Row(static_cast<size_t>(hidden));
    float value = bias0[hidden];
    for (int k = 0; k < model_dim; ++k) {
      value += pre_ffw[k] * BF16ToF32(weight_row[k]);
    }
    up_row0[hidden] = GeluHost(value);
  }

  float f32_spot_max_abs = 0.0f;
  for (int col = 0; col < model_dim; ++col) {
    const BF16* weight_row = linear1_weight.Row(static_cast<size_t>(col));
    float reference = bias1[col];
    for (int hidden = 0; hidden < mlp; ++hidden) {
      reference += up_row0[hidden] * BF16ToF32(weight_row[hidden]);
    }
    f32_spot_max_abs = HWY_MAX(f32_spot_max_abs,
                               std::abs(out_f32[col] - reference));
  }

  // Full-output error compares the candidate BF16 route to the F32-shadow
  // route. This is a precision/locality signal for the ViT backend, not yet an
  // end-to-end image-token equivalence proof.
  const ErrorStats bf16_error = ComputeError(out_bf16, out_f32);
  impl_->stats.layer0_mlp_validated = true;
  impl_->stats.layer0_mlp_f32_spot_max_abs_error = f32_spot_max_abs;
  impl_->stats.layer0_mlp_bf16_max_abs_error = bf16_error.max_abs;
  impl_->stats.layer0_mlp_bf16_rms_error = bf16_error.rms;
  if (impl_->options.verbose) {
    std::printf("layer0_mlp_f32              time=%8.3f ms "
                "spot_max_abs=%9.6f\n",
                impl_->stats.layer0_mlp_f32_ms,
                impl_->stats.layer0_mlp_f32_spot_max_abs_error);
    std::printf("layer0_mlp_bf16             time=%8.3f ms "
                "max_abs_vs_f32=%9.6f rms_vs_f32=%9.6f\n",
                impl_->stats.layer0_mlp_bf16_ms,
                impl_->stats.layer0_mlp_bf16_max_abs_error,
                impl_->stats.layer0_mlp_bf16_rms_error);
  }
  return true;
}

bool PaliGemma2VitHipBackend::ValidateMlpDownWmma8(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image) {
  // Replacement-kernel check for the exact MLP down boundary used by the
  // BF16-only ViT layer runner. This does not compare against CPU or F32
  // shadows; it answers one narrower question: given the same resident BF16
  // hidden tensor and the same resident BF16 linear_1 weight, does WMMA8 produce
  // the same F32 output as rocBLAS?
  if (!UploadProjectionWeights(weights)) return false;

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int model_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  const LayerWeightsPtrs& layer0 = *weights.VitLayer(0);
  const LayerConfig& layer_config = layer0.layer_config;
  if (layer_config.activation != ActivationType::Gelu) {
    std::fprintf(stderr, "MLP down WMMA8 validation only implements GELU.\n");
    return false;
  }

  const int mlp = static_cast<int>(layer_config.ff_hidden_dim);
  HWY_ASSERT(layer0.vit.linear_0_w.Rows() == static_cast<size_t>(mlp));
  HWY_ASSERT(layer0.vit.linear_0_w.Cols() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.linear_1_w.Rows() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.linear_1_w.Cols() == static_cast<size_t>(mlp));
  HWY_ASSERT(impl_->resident->layers[0].linear_1.k == mlp);
  HWY_ASSERT(impl_->resident->layers[0].linear_1.n == model_dim);

  const std::vector<float> patches = BuildImagePatches(image, rows, patch);
  const std::vector<float> patch_embedding =
      BuildPatchEmbeddingReference(model_config, weights, patches);
  const std::vector<float> pre_ffw = LayerNormReference(
      patch_embedding, rows, model_dim, layer0.vit.layer_norm_1_scale,
      layer0.vit.layer_norm_1_bias);
  const std::vector<rocblas_bfloat16> pre_ffw_bf16 =
      F32ToRocblasBF16(pre_ffw);

  std::vector<float> bias0(static_cast<size_t>(mlp), 0.0f);
  if (layer_config.ff_biases) {
    std::memcpy(bias0.data(), layer0.vit.linear_0_b.PackedScale1(),
                bias0.size() * sizeof(float));
  }

  DeviceBuffer<rocblas_bfloat16> d_pre_ffw_bf16(pre_ffw_bf16.size());
  DeviceBuffer<float> d_up_bf16_accum(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<rocblas_bfloat16> d_up_bf16(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<float> d_bias0(bias0.size());
  DeviceBuffer<float> d_down_rocblas(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_down_wmma(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(d_pre_ffw_bf16.ptr, pre_ffw_bf16.data(),
                      pre_ffw_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_bias0.ptr, bias0.data(), bias0.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  const int threads = 256;
  const int up_count = rows * mlp;
  const int up_blocks = (up_count + threads - 1) / threads;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  auto run_up_and_activation = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, mlp,
        rows, model_dim, &alpha, impl_->resident->layers[0].linear_0.device->ptr,
        rocblas_datatype_bf16_r, mlp, d_pre_ffw_bf16.ptr,
        rocblas_datatype_bf16_r, model_dim, &beta, d_up_bf16_accum.ptr,
        rocblas_datatype_f32_r, mlp, d_up_bf16_accum.ptr,
        rocblas_datatype_f32_r, mlp, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0,
        rocblas_gemm_flags_none));
    hipLaunchKernelGGL(AddBiasGeluToBF16Kernel, dim3(up_blocks),
                       dim3(threads), 0, 0, d_up_bf16_accum.ptr, d_bias0.ptr,
                       d_up_bf16.ptr, rows, mlp);
    HIP_CHECK(hipGetLastError());
  };

  auto run_down_rocblas = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, mlp, &alpha,
        impl_->resident->layers[0].linear_1.device->ptr,
        rocblas_datatype_bf16_r, model_dim, d_up_bf16.ptr,
        rocblas_datatype_bf16_r, mlp, &beta, d_down_rocblas.ptr,
        rocblas_datatype_f32_r, model_dim, d_down_rocblas.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0,
        rocblas_gemm_flags_none));
  };

  auto run_down_wmma8 = [&]() {
    LaunchMlpDownWmmaGroupedN(
        d_up_bf16.ptr, impl_->resident->layers[0].linear_1.device->ptr,
        d_down_wmma.ptr, rows, mlp, model_dim,
        impl_->options.mlp_down_wmma_waves);
  };

  run_up_and_activation();
  run_down_rocblas();
  run_down_wmma8();
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> down_rocblas(static_cast<size_t>(rows) * model_dim);
  std::vector<float> down_wmma(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(down_rocblas.data(), d_down_rocblas.ptr,
                      down_rocblas.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(down_wmma.data(), d_down_wmma.ptr,
                      down_wmma.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  const ErrorStats error = ComputeError(down_wmma, down_rocblas);

  const float rocblas_ms = TimeSamples(impl_->options, run_down_rocblas);
  const float wmma_ms = TimeSamples(impl_->options, run_down_wmma8);
  if (impl_->options.verbose) {
    std::printf("mlp_down_wmma8_check rows=%d model_dim=%d mlp=%d "
                "rocblas=%8.3f ms %s=%8.3f ms max_abs=%9.6f "
                "rms=%9.6f\n",
                rows, model_dim, mlp, rocblas_ms,
                MlpDownWmmaTileName(impl_->options.mlp_down_wmma_waves),
                wmma_ms, error.max_abs, error.rms);
  }
  return true;
}

bool PaliGemma2VitHipBackend::ValidateMlpUpWmma2D(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image) {
  // Replacement-kernel check for the exact MLP up boundary used by the BF16-only
  // ViT layer runner. It compares the local grouped-2D WMMA 4x4 kernel against
  // rocBLAS for:
  //   pre_ffw_bf16[rows, model_dim] * resident linear_0[model_dim, mlp].
  // Bias/GELU are deliberately excluded so the check isolates the GEMM that the
  // new kernel replaces.
  if (!UploadProjectionWeights(weights)) return false;

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int model_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  const LayerWeightsPtrs& layer0 = *weights.VitLayer(0);
  const LayerConfig& layer_config = layer0.layer_config;
  if (layer_config.activation != ActivationType::Gelu) {
    std::fprintf(stderr, "MLP up WMMA2D validation only implements GELU.\n");
    return false;
  }

  const int mlp = static_cast<int>(layer_config.ff_hidden_dim);
  HWY_ASSERT(layer0.vit.linear_0_w.Rows() == static_cast<size_t>(mlp));
  HWY_ASSERT(layer0.vit.linear_0_w.Cols() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(impl_->resident->layers[0].linear_0.k == model_dim);
  HWY_ASSERT(impl_->resident->layers[0].linear_0.n == mlp);

  const std::vector<float> patches = BuildImagePatches(image, rows, patch);
  const std::vector<float> patch_embedding =
      BuildPatchEmbeddingReference(model_config, weights, patches);
  const std::vector<float> pre_ffw = LayerNormReference(
      patch_embedding, rows, model_dim, layer0.vit.layer_norm_1_scale,
      layer0.vit.layer_norm_1_bias);
  const std::vector<rocblas_bfloat16> pre_ffw_bf16 =
      F32ToRocblasBF16(pre_ffw);

  DeviceBuffer<rocblas_bfloat16> d_pre_ffw_bf16(pre_ffw_bf16.size());
  DeviceBuffer<float> d_up_rocblas(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<float> d_up_wmma(static_cast<size_t>(rows) * mlp);
  HIP_CHECK(hipMemcpy(d_pre_ffw_bf16.ptr, pre_ffw_bf16.data(),
                      pre_ffw_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto run_up_rocblas = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, mlp,
        rows, model_dim, &alpha,
        impl_->resident->layers[0].linear_0.device->ptr,
        rocblas_datatype_bf16_r, mlp, d_pre_ffw_bf16.ptr,
        rocblas_datatype_bf16_r, model_dim, &beta, d_up_rocblas.ptr,
        rocblas_datatype_f32_r, mlp, d_up_rocblas.ptr,
        rocblas_datatype_f32_r, mlp, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0,
        rocblas_gemm_flags_none));
  };

  auto run_up_wmma2d = [&]() {
    LaunchMlpUpWmma2DPlain(d_pre_ffw_bf16.ptr,
                           impl_->resident->layers[0].linear_0.device->ptr,
                           d_up_wmma.ptr, rows, model_dim, mlp);
  };

  run_up_rocblas();
  run_up_wmma2d();
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> up_rocblas(static_cast<size_t>(rows) * mlp);
  std::vector<float> up_wmma(static_cast<size_t>(rows) * mlp);
  HIP_CHECK(hipMemcpy(up_rocblas.data(), d_up_rocblas.ptr,
                      up_rocblas.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(up_wmma.data(), d_up_wmma.ptr,
                      up_wmma.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  const ErrorStats error = ComputeError(up_wmma, up_rocblas);

  const float rocblas_ms = TimeSamples(impl_->options, run_up_rocblas);
  const float wmma_ms = TimeSamples(impl_->options, run_up_wmma2d);
  if (impl_->options.verbose) {
    std::printf("mlp_up_wmma2d_check rows=%d model_dim=%d mlp=%d "
                "rocblas=%8.3f ms %s=%8.3f ms max_abs=%9.6f "
                "rms=%9.6f\n",
                rows, model_dim, mlp, rocblas_ms,
                MlpUpWmma2DTileName(rows), wmma_ms, error.max_abs, error.rms);
  }
  return true;
}

bool PaliGemma2VitHipBackend::ValidateQkvWmma2D(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image) {
  // Replacement-kernel check for the exact QKV projection boundary used by the
  // BF16-only ViT layer runner. It compares the local grouped-2D WMMA 8x4
  // kernel against rocBLAS for:
  //   pre_att_bf16[rows, model_dim] * resident qkv[model_dim, qkv].
  // Bias and later Q/K/V splitting are excluded so this isolates the GEMM that
  // the new kernel replaces.
  if (!UploadProjectionWeights(weights)) return false;

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int model_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  const LayerWeightsPtrs& layer0 = *weights.VitLayer(0);
  const LayerConfig& layer_config = layer0.layer_config;
  const int heads = static_cast<int>(layer_config.heads);
  const int qkv_dim = static_cast<int>(layer_config.qkv_dim);
  const int qkv = heads * 3 * qkv_dim;
  HWY_ASSERT(layer_config.heads == layer_config.kv_heads);
  HWY_ASSERT(layer0.vit.qkv_einsum_w.Rows() == static_cast<size_t>(qkv));
  HWY_ASSERT(layer0.vit.qkv_einsum_w.Cols() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(impl_->resident->layers[0].qkv.k == model_dim);
  HWY_ASSERT(impl_->resident->layers[0].qkv.n == qkv);

  const std::vector<float> patches = BuildImagePatches(image, rows, patch);
  const std::vector<float> patch_embedding =
      BuildPatchEmbeddingReference(model_config, weights, patches);
  const std::vector<float> pre_att = LayerNormReference(
      patch_embedding, rows, model_dim, layer0.vit.layer_norm_0_scale,
      layer0.vit.layer_norm_0_bias);
  const std::vector<rocblas_bfloat16> pre_att_bf16 =
      F32ToRocblasBF16(pre_att);

  DeviceBuffer<rocblas_bfloat16> d_pre_att_bf16(pre_att_bf16.size());
  DeviceBuffer<float> d_qkv_rocblas(static_cast<size_t>(rows) * qkv);
  DeviceBuffer<float> d_qkv_wmma(static_cast<size_t>(rows) * qkv);
  HIP_CHECK(hipMemcpy(d_pre_att_bf16.ptr, pre_att_bf16.data(),
                      pre_att_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto run_qkv_rocblas = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, qkv,
        rows, model_dim, &alpha, impl_->resident->layers[0].qkv.device->ptr,
        rocblas_datatype_bf16_r, qkv, d_pre_att_bf16.ptr,
        rocblas_datatype_bf16_r, model_dim, &beta, d_qkv_rocblas.ptr,
        rocblas_datatype_f32_r, qkv, d_qkv_rocblas.ptr,
        rocblas_datatype_f32_r, qkv, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0,
        rocblas_gemm_flags_none));
  };

  auto run_qkv_wmma2d = [&]() {
    constexpr int kWavesM = 8;
    constexpr int kWavesN = 4;
    const dim3 block(32 * kWavesM * kWavesN);
    const dim3 grid((qkv + 16 * kWavesN - 1) / (16 * kWavesN),
                    (rows + 16 * kWavesM - 1) / (16 * kWavesM));
    hipLaunchKernelGGL((Bf16GemmWmma2DKernel<kWavesM, kWavesN>), grid, block,
                       0, 0, d_pre_att_bf16.ptr,
                       impl_->resident->layers[0].qkv.device->ptr,
                       d_qkv_wmma.ptr, rows, model_dim, qkv);
    HIP_CHECK(hipGetLastError());
  };

  run_qkv_rocblas();
  run_qkv_wmma2d();
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> qkv_rocblas(static_cast<size_t>(rows) * qkv);
  std::vector<float> qkv_wmma(static_cast<size_t>(rows) * qkv);
  HIP_CHECK(hipMemcpy(qkv_rocblas.data(), d_qkv_rocblas.ptr,
                      qkv_rocblas.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(qkv_wmma.data(), d_qkv_wmma.ptr,
                      qkv_wmma.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  const ErrorStats error = ComputeError(qkv_wmma, qkv_rocblas);

  const float rocblas_ms = TimeSamples(impl_->options, run_qkv_rocblas);
  const float wmma_ms = TimeSamples(impl_->options, run_qkv_wmma2d);
  if (impl_->options.verbose) {
    std::printf("qkv_wmma2d_check rows=%d model_dim=%d qkv=%d "
                "rocblas=%8.3f ms wmma2d8x4=%8.3f ms max_abs=%9.6f "
                "rms=%9.6f\n",
                rows, model_dim, qkv, rocblas_ms, wmma_ms, error.max_abs,
                error.rms);
  }
  return true;
}

bool PaliGemma2VitHipBackend::ValidateAttnOutWmma2D(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image) {
  // Replacement-kernel check for the exact attention-output projection GEMM
  // used by the BF16-only ViT layer runner. QKV and attention are prepared once
  // to produce a deterministic BF16 [rows, att_cols] input. The measured and
  // compared boundary is only:
  //   att_row_bf16[rows, att_cols] * resident attn_out[att_cols, model_dim].
  if (!UploadProjectionWeights(weights)) return false;

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int model_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  const LayerWeightsPtrs& layer0 = *weights.VitLayer(0);
  const LayerConfig& layer_config = layer0.layer_config;
  const int heads = static_cast<int>(layer_config.heads);
  const int qkv_dim = static_cast<int>(layer_config.qkv_dim);
  const int qkv = heads * 3 * qkv_dim;
  const int att_cols = heads * qkv_dim;
  HWY_ASSERT(layer_config.heads == layer_config.kv_heads);
  HWY_ASSERT(att_cols == model_dim);
  HWY_ASSERT(layer0.vit.qkv_einsum_w.Rows() == static_cast<size_t>(qkv));
  HWY_ASSERT(layer0.vit.qkv_einsum_w.Cols() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.attn_out_w.Rows() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.attn_out_w.Cols() == static_cast<size_t>(att_cols));
  HWY_ASSERT(impl_->resident->layers[0].attn_out.k == att_cols);
  HWY_ASSERT(impl_->resident->layers[0].attn_out.n == model_dim);

  const std::vector<float> patches = BuildImagePatches(image, rows, patch);
  const std::vector<float> patch_embedding =
      BuildPatchEmbeddingReference(model_config, weights, patches);
  const std::vector<float> pre_att = LayerNormReference(
      patch_embedding, rows, model_dim, layer0.vit.layer_norm_0_scale,
      layer0.vit.layer_norm_0_bias);

  ResidentF32Mat qkv_f32 =
      UploadBF16MatAsF32RocblasB("layer0.qkv", layer0.vit.qkv_einsum_w,
                                 impl_->options.verbose);
  DeviceBuffer<float> d_pre_att(pre_att.size());
  DeviceBuffer<float> d_qkv(static_cast<size_t>(rows) * qkv);
  DeviceBuffer<float> d_qkv_bias(static_cast<size_t>(qkv));
  HIP_CHECK(hipMemcpy(d_pre_att.ptr, pre_att.data(),
                      pre_att.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qkv_bias.ptr, layer0.vit.qkv_einsum_b.PackedScale1(),
                      static_cast<size_t>(qkv) * sizeof(float),
                      hipMemcpyHostToDevice));

  const int threads = 256;
  const int qkv_count = rows * qkv;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  ROCBLAS_CHECK(rocblas_gemm_ex(
      impl_->handle, rocblas_operation_none, rocblas_operation_none, qkv, rows,
      model_dim, &alpha, qkv_f32.device->ptr, rocblas_datatype_f32_r, qkv,
      d_pre_att.ptr, rocblas_datatype_f32_r, model_dim, &beta, d_qkv.ptr,
      rocblas_datatype_f32_r, qkv, d_qkv.ptr, rocblas_datatype_f32_r, qkv,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
      /*solution_index=*/0, /*flags=*/0));
  hipLaunchKernelGGL(AddBiasKernel, dim3(qkv_blocks), dim3(threads), 0, 0,
                     d_qkv.ptr, d_qkv_bias.ptr, rows, qkv);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> qkv_host(static_cast<size_t>(rows) * qkv);
  HIP_CHECK(hipMemcpy(qkv_host.data(), d_qkv.ptr,
                      qkv_host.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  const std::vector<float> att_out =
      ComputeAttentionWeightedSumReference(qkv_host, rows, heads, qkv_dim);
  const std::vector<rocblas_bfloat16> att_out_bf16 =
      F32ToRocblasBF16(att_out);

  DeviceBuffer<rocblas_bfloat16> d_att_out_bf16(att_out_bf16.size());
  DeviceBuffer<float> d_out_rocblas(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_out_wmma(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(d_att_out_bf16.ptr, att_out_bf16.data(),
                      att_out_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));

  auto run_attn_out_rocblas = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, att_cols, &alpha,
        impl_->resident->layers[0].attn_out.device->ptr,
        rocblas_datatype_bf16_r, model_dim, d_att_out_bf16.ptr,
        rocblas_datatype_bf16_r, att_cols, &beta, d_out_rocblas.ptr,
        rocblas_datatype_f32_r, model_dim, d_out_rocblas.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0,
        rocblas_gemm_flags_none));
  };

  auto run_attn_out_wmma2d = [&]() {
    constexpr int kWavesM = 8;
    constexpr int kWavesN = 4;
    const dim3 block(32 * kWavesM * kWavesN);
    const dim3 grid((model_dim + 16 * kWavesN - 1) / (16 * kWavesN),
                    (rows + 16 * kWavesM - 1) / (16 * kWavesM));
    hipLaunchKernelGGL((Bf16GemmWmma2DKernel<kWavesM, kWavesN>), grid, block,
                       0, 0, d_att_out_bf16.ptr,
                       impl_->resident->layers[0].attn_out.device->ptr,
                       d_out_wmma.ptr, rows, att_cols, model_dim);
    HIP_CHECK(hipGetLastError());
  };

  run_attn_out_rocblas();
  run_attn_out_wmma2d();
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> out_rocblas(static_cast<size_t>(rows) * model_dim);
  std::vector<float> out_wmma(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(out_rocblas.data(), d_out_rocblas.ptr,
                      out_rocblas.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(out_wmma.data(), d_out_wmma.ptr,
                      out_wmma.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  const ErrorStats error = ComputeError(out_wmma, out_rocblas);

  const float rocblas_ms = TimeSamples(impl_->options, run_attn_out_rocblas);
  const float wmma_ms = TimeSamples(impl_->options, run_attn_out_wmma2d);
  if (impl_->options.verbose) {
    std::printf("attn_out_wmma2d_check rows=%d att_cols=%d model_dim=%d "
                "rocblas=%8.3f ms wmma2d8x4=%8.3f ms max_abs=%9.6f "
                "rms=%9.6f\n",
                rows, att_cols, model_dim, rocblas_ms, wmma_ms,
                error.max_abs, error.rms);
  }
  return true;
}

bool PaliGemma2VitHipBackend::ValidateLayer0AttentionOut(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image) {
  // Fourth real-tensor boundary. This validates the tail of layer0 attention:
  //   QKV -> softmax weighted sum -> attn_out_w -> attention residual.
  //
  // QKV and softmax/weighted-sum are prepared deterministically before timing
  // so the measured device work is the projection that maps concatenated head
  // outputs back into model_dim, plus the residual add into the patch stream.
  if (!UploadProjectionWeights(weights)) return false;

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int model_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  const LayerWeightsPtrs& layer0 = *weights.VitLayer(0);
  const LayerConfig& layer_config = layer0.layer_config;
  const int heads = static_cast<int>(layer_config.heads);
  const int qkv_dim = static_cast<int>(layer_config.qkv_dim);
  const int qkv = heads * 3 * qkv_dim;
  const int att_cols = heads * qkv_dim;
  HWY_ASSERT(layer_config.heads == layer_config.kv_heads);
  HWY_ASSERT(att_cols == model_dim);
  HWY_ASSERT(layer0.vit.qkv_einsum_w.Rows() == static_cast<size_t>(qkv));
  HWY_ASSERT(layer0.vit.qkv_einsum_w.Cols() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.attn_out_w.Rows() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.attn_out_w.Cols() == static_cast<size_t>(att_cols));

  // Start from the same layer0 input as the CPU ViT: patch embedding with
  // position added, then layer_norm_0 before QKV.
  const std::vector<float> patches = BuildImagePatches(image, rows, patch);
  const std::vector<float> patch_embedding =
      BuildPatchEmbeddingReference(model_config, weights, patches);
  const std::vector<float> pre_att = LayerNormReference(
      patch_embedding, rows, model_dim, layer0.vit.layer_norm_0_scale,
      layer0.vit.layer_norm_0_bias);

  // Reuse the already-validated F32-shadow QKV layout to feed the scalar host
  // attention reference. This keeps the next boundary focused on attn_out_w and
  // residual while avoiding a very large CPU-side QKV matmul for 448px.
  ResidentF32Mat qkv_f32 =
      UploadBF16MatAsF32RocblasB("layer0.qkv", layer0.vit.qkv_einsum_w,
                                 impl_->options.verbose);
  DeviceBuffer<float> d_pre_att(pre_att.size());
  DeviceBuffer<float> d_qkv(static_cast<size_t>(rows) * qkv);
  DeviceBuffer<float> d_qkv_bias(static_cast<size_t>(qkv));
  HIP_CHECK(hipMemcpy(d_pre_att.ptr, pre_att.data(),
                      pre_att.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qkv_bias.ptr, layer0.vit.qkv_einsum_b.PackedScale1(),
                      static_cast<size_t>(qkv) * sizeof(float),
                      hipMemcpyHostToDevice));

  const int threads = 256;
  const int qkv_count = rows * qkv;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  ROCBLAS_CHECK(rocblas_gemm_ex(
      impl_->handle, rocblas_operation_none, rocblas_operation_none, qkv, rows,
      model_dim, &alpha, qkv_f32.device->ptr, rocblas_datatype_f32_r, qkv,
      d_pre_att.ptr, rocblas_datatype_f32_r, model_dim, &beta, d_qkv.ptr,
      rocblas_datatype_f32_r, qkv, d_qkv.ptr, rocblas_datatype_f32_r, qkv,
      rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
      /*solution_index=*/0, /*flags=*/0));
  hipLaunchKernelGGL(AddBiasKernel, dim3(qkv_blocks), dim3(threads), 0, 0,
                     d_qkv.ptr, d_qkv_bias.ptr, rows, qkv);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> qkv_host(static_cast<size_t>(rows) * qkv);
  HIP_CHECK(hipMemcpy(qkv_host.data(), d_qkv.ptr,
                      qkv_host.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  const double host_attention_start = hwy::platform::Now();
  const std::vector<float> att_out =
      ComputeAttentionWeightedSumReference(qkv_host, rows, heads, qkv_dim);
  const double host_attention_seconds =
      hwy::platform::Now() - host_attention_start;
  if (impl_->options.verbose) {
    std::printf("layer0_attention_reference  host_softmax=%8.3f s\n",
                host_attention_seconds);
  }
  const std::vector<rocblas_bfloat16> att_out_bf16 =
      F32ToRocblasBF16(att_out);

  // F32-shadow projection checks the row/column mapping and residual placement.
  // BF16 projection uses the resident attn_out_w and quantized attention output
  // to test the future fast route.
  ResidentF32Mat attn_out_f32 =
      UploadBF16MatAsF32RocblasB("layer0.attn_out",
                                 layer0.vit.attn_out_w,
                                 impl_->options.verbose);
  DeviceBuffer<float> d_att_out(att_out.size());
  DeviceBuffer<rocblas_bfloat16> d_att_out_bf16(att_out_bf16.size());
  DeviceBuffer<float> d_residual(patch_embedding.size());
  DeviceBuffer<float> d_out_f32(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_out_bf16(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_bias(static_cast<size_t>(model_dim));
  HIP_CHECK(hipMemcpy(d_att_out.ptr, att_out.data(),
                      att_out.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_att_out_bf16.ptr, att_out_bf16.data(),
                      att_out_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_residual.ptr, patch_embedding.data(),
                      patch_embedding.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_bias.ptr, layer0.vit.attn_out_b.PackedScale1(),
                      static_cast<size_t>(model_dim) * sizeof(float),
                      hipMemcpyHostToDevice));

  const int out_count = rows * model_dim;
  const int out_blocks = (out_count + threads - 1) / threads;
  auto add_bias_and_residual = [&](float* out) {
    hipLaunchKernelGGL(AddBiasKernel, dim3(out_blocks), dim3(threads), 0, 0,
                       out, d_bias.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(AddResidualKernel, dim3(out_blocks), dim3(threads), 0, 0,
                       out, d_residual.ptr, out_count);
    HIP_CHECK(hipGetLastError());
  };

  auto run_attn_out_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, att_cols, &alpha, attn_out_f32.device->ptr,
        rocblas_datatype_f32_r, model_dim, d_att_out.ptr,
        rocblas_datatype_f32_r, att_cols, &beta, d_out_f32.ptr,
        rocblas_datatype_f32_r, model_dim, d_out_f32.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_bias_and_residual(d_out_f32.ptr);
  };

  auto run_attn_out_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, att_cols, &alpha,
        impl_->resident->layers[0].attn_out.device->ptr,
        rocblas_datatype_bf16_r, model_dim, d_att_out_bf16.ptr,
        rocblas_datatype_bf16_r, att_cols, &beta, d_out_bf16.ptr,
        rocblas_datatype_f32_r, model_dim, d_out_bf16.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_bias_and_residual(d_out_bf16.ptr);
  };

  impl_->stats.layer0_attn_out_f32_ms =
      TimeSamples(impl_->options, run_attn_out_f32);
  impl_->stats.layer0_attn_out_bf16_ms =
      TimeSamples(impl_->options, run_attn_out_bf16);

  std::vector<float> out_f32(static_cast<size_t>(rows) * model_dim);
  std::vector<float> out_bf16(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(out_f32.data(), d_out_f32.ptr,
                      out_f32.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(out_bf16.data(), d_out_bf16.ptr,
                      out_bf16.size() * sizeof(float), hipMemcpyDeviceToHost));

  // Row-0 scalar check confirms the F32-shadow projection and residual add.
  // The full attention weighted sum was already computed on host, so this spot
  // check only needs one output row to catch attn_out_w layout mistakes.
  const MatPtrT<BF16> attn_out_weight(layer0.vit.attn_out_w);
  const float* bias = layer0.vit.attn_out_b.PackedScale1();
  float f32_spot_max_abs = 0.0f;
  for (int col = 0; col < model_dim; ++col) {
    const BF16* weight_row = attn_out_weight.Row(static_cast<size_t>(col));
    float reference = patch_embedding[static_cast<size_t>(col)] + bias[col];
    for (int k = 0; k < att_cols; ++k) {
      reference += att_out[static_cast<size_t>(k)] * BF16ToF32(weight_row[k]);
    }
    f32_spot_max_abs = HWY_MAX(f32_spot_max_abs,
                               std::abs(out_f32[col] - reference));
  }

  const ErrorStats bf16_error = ComputeError(out_bf16, out_f32);
  impl_->stats.layer0_attn_out_validated = true;
  impl_->stats.layer0_attn_out_f32_spot_max_abs_error = f32_spot_max_abs;
  impl_->stats.layer0_attn_out_bf16_max_abs_error = bf16_error.max_abs;
  impl_->stats.layer0_attn_out_bf16_rms_error = bf16_error.rms;
  if (impl_->options.verbose) {
    std::printf("layer0_attn_out_f32         time=%8.3f ms "
                "spot_max_abs=%9.6f\n",
                impl_->stats.layer0_attn_out_f32_ms,
                impl_->stats.layer0_attn_out_f32_spot_max_abs_error);
    std::printf("layer0_attn_out_bf16        time=%8.3f ms "
                "max_abs_vs_f32=%9.6f rms_vs_f32=%9.6f\n",
                impl_->stats.layer0_attn_out_bf16_ms,
                impl_->stats.layer0_attn_out_bf16_max_abs_error,
                impl_->stats.layer0_attn_out_bf16_rms_error);
  }
  return true;
}

bool PaliGemma2VitHipBackend::ValidateLayer0Block(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image) {
  // Whole-layer correctness boundary for ViT layer0. This chains:
  //   x0 -> layer_norm_0 -> QKV -> attention -> attn_out + residual
  //      -> layer_norm_1 -> MLP -> final residual.
  //
  // The attention softmax/weighted-sum is still a host reference here. That
  // keeps this stage focused on whether the BF16 resident projection path stays
  // numerically close after one complete layer, without pretending the current
  // implementation is already an end-to-end accelerated ViT block.
  if (!UploadProjectionWeights(weights)) return false;

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int model_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  const LayerWeightsPtrs& layer0 = *weights.VitLayer(0);
  const LayerConfig& layer_config = layer0.layer_config;
  if (layer_config.activation != ActivationType::Gelu) {
    std::fprintf(stderr, "Layer0 block validation only implements GELU.\n");
    return false;
  }

  const int heads = static_cast<int>(layer_config.heads);
  const int qkv_dim = static_cast<int>(layer_config.qkv_dim);
  const int qkv = heads * 3 * qkv_dim;
  const int att_cols = heads * qkv_dim;
  const int mlp = static_cast<int>(layer_config.ff_hidden_dim);
  HWY_ASSERT(layer_config.heads == layer_config.kv_heads);
  HWY_ASSERT(att_cols == model_dim);
  HWY_ASSERT(layer0.vit.qkv_einsum_w.Rows() == static_cast<size_t>(qkv));
  HWY_ASSERT(layer0.vit.qkv_einsum_w.Cols() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.attn_out_w.Rows() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.attn_out_w.Cols() == static_cast<size_t>(att_cols));
  HWY_ASSERT(layer0.vit.linear_0_w.Rows() == static_cast<size_t>(mlp));
  HWY_ASSERT(layer0.vit.linear_0_w.Cols() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.linear_1_w.Rows() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.linear_1_w.Cols() == static_cast<size_t>(mlp));

  const std::vector<float> patches = BuildImagePatches(image, rows, patch);
  const std::vector<float> x0 =
      BuildPatchEmbeddingReference(model_config, weights, patches);
  const std::vector<float> pre_att = LayerNormReference(
      x0, rows, model_dim, layer0.vit.layer_norm_0_scale,
      layer0.vit.layer_norm_0_bias);
  const std::vector<rocblas_bfloat16> pre_att_bf16 =
      F32ToRocblasBF16(pre_att);

  std::vector<float> mlp_bias0(static_cast<size_t>(mlp), 0.0f);
  std::vector<float> mlp_bias1(static_cast<size_t>(model_dim), 0.0f);
  if (layer_config.ff_biases) {
    std::memcpy(mlp_bias0.data(), layer0.vit.linear_0_b.PackedScale1(),
                mlp_bias0.size() * sizeof(float));
    std::memcpy(mlp_bias1.data(), layer0.vit.linear_1_b.PackedScale1(),
                mlp_bias1.size() * sizeof(float));
  }

  ResidentF32Mat qkv_f32 =
      UploadBF16MatAsF32RocblasB("layer0.qkv", layer0.vit.qkv_einsum_w,
                                 impl_->options.verbose);
  ResidentF32Mat attn_out_f32 =
      UploadBF16MatAsF32RocblasB("layer0.attn_out",
                                 layer0.vit.attn_out_w,
                                 impl_->options.verbose);
  ResidentF32Mat linear0_f32 =
      UploadBF16MatAsF32RocblasB("layer0.linear_0",
                                 layer0.vit.linear_0_w,
                                 impl_->options.verbose);
  ResidentF32Mat linear1_f32 =
      UploadBF16MatAsF32RocblasB("layer0.linear_1",
                                 layer0.vit.linear_1_w,
                                 impl_->options.verbose);

  const int threads = 256;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int qkv_count = rows * qkv;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const int model_count = rows * model_dim;
  const int model_blocks = (model_count + threads - 1) / threads;

  DeviceBuffer<float> d_pre_att(pre_att.size());
  DeviceBuffer<rocblas_bfloat16> d_pre_att_bf16(pre_att_bf16.size());
  DeviceBuffer<float> d_qkv_f32(static_cast<size_t>(rows) * qkv);
  DeviceBuffer<float> d_qkv_bf16(static_cast<size_t>(rows) * qkv);
  DeviceBuffer<float> d_qkv_bias(static_cast<size_t>(qkv));
  HIP_CHECK(hipMemcpy(d_pre_att.ptr, pre_att.data(),
                      pre_att.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pre_att_bf16.ptr, pre_att_bf16.data(),
                      pre_att_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qkv_bias.ptr, layer0.vit.qkv_einsum_b.PackedScale1(),
                      static_cast<size_t>(qkv) * sizeof(float),
                      hipMemcpyHostToDevice));

  auto add_qkv_bias = [&](float* out) {
    hipLaunchKernelGGL(AddBiasKernel, dim3(qkv_blocks), dim3(threads), 0, 0,
                       out, d_qkv_bias.ptr, rows, qkv);
    HIP_CHECK(hipGetLastError());
  };
  auto run_qkv_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, qkv,
        rows, model_dim, &alpha, qkv_f32.device->ptr, rocblas_datatype_f32_r,
        qkv, d_pre_att.ptr, rocblas_datatype_f32_r, model_dim, &beta,
        d_qkv_f32.ptr, rocblas_datatype_f32_r, qkv, d_qkv_f32.ptr,
        rocblas_datatype_f32_r, qkv, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_qkv_bias(d_qkv_f32.ptr);
  };
  auto run_qkv_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, qkv,
        rows, model_dim, &alpha, impl_->resident->layers[0].qkv.device->ptr,
        rocblas_datatype_bf16_r, qkv, d_pre_att_bf16.ptr,
        rocblas_datatype_bf16_r, model_dim, &beta, d_qkv_bf16.ptr,
        rocblas_datatype_f32_r, qkv, d_qkv_bf16.ptr, rocblas_datatype_f32_r,
        qkv, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
    add_qkv_bias(d_qkv_bf16.ptr);
  };

  impl_->stats.layer0_block_qkv_f32_ms =
      TimeSamples(impl_->options, run_qkv_f32);
  impl_->stats.layer0_block_qkv_bf16_ms =
      TimeSamples(impl_->options, run_qkv_bf16);

  std::vector<float> qkv_host_f32(static_cast<size_t>(rows) * qkv);
  std::vector<float> qkv_host_bf16(static_cast<size_t>(rows) * qkv);
  HIP_CHECK(hipMemcpy(qkv_host_f32.data(), d_qkv_f32.ptr,
                      qkv_host_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(qkv_host_bf16.data(), d_qkv_bf16.ptr,
                      qkv_host_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  const double att_f32_start = hwy::platform::Now();
  const std::vector<float> att_weighted_f32 =
      ComputeAttentionWeightedSumReference(qkv_host_f32, rows, heads, qkv_dim);
  impl_->stats.layer0_block_attention_f32_seconds =
      hwy::platform::Now() - att_f32_start;
  const double att_bf16_start = hwy::platform::Now();
  const std::vector<float> att_weighted_bf16 =
      ComputeAttentionWeightedSumReference(qkv_host_bf16, rows, heads, qkv_dim);
  impl_->stats.layer0_block_attention_bf16_seconds =
      hwy::platform::Now() - att_bf16_start;

  const std::vector<rocblas_bfloat16> att_weighted_bf16_input =
      F32ToRocblasBF16(att_weighted_bf16);
  DeviceBuffer<float> d_att_weighted_f32(att_weighted_f32.size());
  DeviceBuffer<rocblas_bfloat16> d_att_weighted_bf16(
      att_weighted_bf16_input.size());
  DeviceBuffer<float> d_x0(x0.size());
  DeviceBuffer<float> d_att_bias(static_cast<size_t>(model_dim));
  DeviceBuffer<float> d_x_att_f32(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_x_att_bf16(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(d_att_weighted_f32.ptr, att_weighted_f32.data(),
                      att_weighted_f32.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_att_weighted_bf16.ptr, att_weighted_bf16_input.data(),
                      att_weighted_bf16_input.size() *
                          sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x0.ptr, x0.data(), x0.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_att_bias.ptr, layer0.vit.attn_out_b.PackedScale1(),
                      static_cast<size_t>(model_dim) * sizeof(float),
                      hipMemcpyHostToDevice));

  auto add_att_bias_and_residual = [&](float* out) {
    hipLaunchKernelGGL(AddBiasKernel, dim3(model_blocks), dim3(threads), 0, 0,
                       out, d_att_bias.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(AddResidualKernel, dim3(model_blocks), dim3(threads), 0,
                       0, out, d_x0.ptr, model_count);
    HIP_CHECK(hipGetLastError());
  };
  auto run_attn_out_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, att_cols, &alpha, attn_out_f32.device->ptr,
        rocblas_datatype_f32_r, model_dim, d_att_weighted_f32.ptr,
        rocblas_datatype_f32_r, att_cols, &beta, d_x_att_f32.ptr,
        rocblas_datatype_f32_r, model_dim, d_x_att_f32.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_att_bias_and_residual(d_x_att_f32.ptr);
  };
  auto run_attn_out_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, att_cols, &alpha,
        impl_->resident->layers[0].attn_out.device->ptr,
        rocblas_datatype_bf16_r, model_dim, d_att_weighted_bf16.ptr,
        rocblas_datatype_bf16_r, att_cols, &beta, d_x_att_bf16.ptr,
        rocblas_datatype_f32_r, model_dim, d_x_att_bf16.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_att_bias_and_residual(d_x_att_bf16.ptr);
  };

  impl_->stats.layer0_block_attn_out_f32_ms =
      TimeSamples(impl_->options, run_attn_out_f32);
  impl_->stats.layer0_block_attn_out_bf16_ms =
      TimeSamples(impl_->options, run_attn_out_bf16);

  std::vector<float> x_att_f32(static_cast<size_t>(rows) * model_dim);
  std::vector<float> x_att_bf16(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(x_att_f32.data(), d_x_att_f32.ptr,
                      x_att_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(x_att_bf16.data(), d_x_att_bf16.ptr,
                      x_att_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  const std::vector<float> pre_ffw_f32 = LayerNormReference(
      x_att_f32, rows, model_dim, layer0.vit.layer_norm_1_scale,
      layer0.vit.layer_norm_1_bias);
  const std::vector<float> pre_ffw_bf16 = LayerNormReference(
      x_att_bf16, rows, model_dim, layer0.vit.layer_norm_1_scale,
      layer0.vit.layer_norm_1_bias);
  const std::vector<rocblas_bfloat16> pre_ffw_bf16_input =
      F32ToRocblasBF16(pre_ffw_bf16);

  DeviceBuffer<float> d_pre_ffw_f32(pre_ffw_f32.size());
  DeviceBuffer<rocblas_bfloat16> d_pre_ffw_bf16(pre_ffw_bf16_input.size());
  DeviceBuffer<float> d_up_f32(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<float> d_up_bf16_accum(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<rocblas_bfloat16> d_up_bf16(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<float> d_final_f32(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_final_bf16(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_mlp_bias0(mlp_bias0.size());
  DeviceBuffer<float> d_mlp_bias1(mlp_bias1.size());
  HIP_CHECK(hipMemcpy(d_pre_ffw_f32.ptr, pre_ffw_f32.data(),
                      pre_ffw_f32.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pre_ffw_bf16.ptr, pre_ffw_bf16_input.data(),
                      pre_ffw_bf16_input.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_mlp_bias0.ptr, mlp_bias0.data(),
                      mlp_bias0.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_mlp_bias1.ptr, mlp_bias1.data(),
                      mlp_bias1.size() * sizeof(float), hipMemcpyHostToDevice));

  const int up_count = rows * mlp;
  const int up_blocks = (up_count + threads - 1) / threads;
  auto add_mlp_bias_and_residual_f32 = [&]() {
    hipLaunchKernelGGL(AddBiasKernel, dim3(model_blocks), dim3(threads), 0, 0,
                       d_final_f32.ptr, d_mlp_bias1.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(AddResidualKernel, dim3(model_blocks), dim3(threads), 0,
                       0, d_final_f32.ptr, d_x_att_f32.ptr, model_count);
    HIP_CHECK(hipGetLastError());
  };
  auto add_mlp_bias_and_residual_bf16 = [&]() {
    hipLaunchKernelGGL(AddBiasKernel, dim3(model_blocks), dim3(threads), 0, 0,
                       d_final_bf16.ptr, d_mlp_bias1.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(AddResidualKernel, dim3(model_blocks), dim3(threads), 0,
                       0, d_final_bf16.ptr, d_x_att_bf16.ptr, model_count);
    HIP_CHECK(hipGetLastError());
  };

  auto run_mlp_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, mlp,
        rows, model_dim, &alpha, linear0_f32.device->ptr,
        rocblas_datatype_f32_r, mlp, d_pre_ffw_f32.ptr,
        rocblas_datatype_f32_r, model_dim, &beta, d_up_f32.ptr,
        rocblas_datatype_f32_r, mlp, d_up_f32.ptr, rocblas_datatype_f32_r, mlp,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
    hipLaunchKernelGGL(AddBiasKernel, dim3(up_blocks), dim3(threads), 0, 0,
                       d_up_f32.ptr, d_mlp_bias0.ptr, rows, mlp);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(GeluKernel, dim3(up_blocks), dim3(threads), 0, 0,
                       d_up_f32.ptr, up_count);
    HIP_CHECK(hipGetLastError());
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, mlp, &alpha, linear1_f32.device->ptr,
        rocblas_datatype_f32_r, model_dim, d_up_f32.ptr,
        rocblas_datatype_f32_r, mlp, &beta, d_final_f32.ptr,
        rocblas_datatype_f32_r, model_dim, d_final_f32.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_mlp_bias_and_residual_f32();
  };

  auto run_mlp_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, mlp,
        rows, model_dim, &alpha, impl_->resident->layers[0].linear_0.device->ptr,
        rocblas_datatype_bf16_r, mlp, d_pre_ffw_bf16.ptr,
        rocblas_datatype_bf16_r, model_dim, &beta, d_up_bf16_accum.ptr,
        rocblas_datatype_f32_r, mlp, d_up_bf16_accum.ptr,
        rocblas_datatype_f32_r, mlp, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    hipLaunchKernelGGL(AddBiasToBF16Kernel, dim3(up_blocks), dim3(threads), 0,
                       0, d_up_bf16_accum.ptr, d_mlp_bias0.ptr,
                       d_up_bf16.ptr, rows, mlp);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(GeluBF16Kernel, dim3(up_blocks), dim3(threads), 0, 0,
                       d_up_bf16.ptr, up_count);
    HIP_CHECK(hipGetLastError());
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, mlp, &alpha,
        impl_->resident->layers[0].linear_1.device->ptr,
        rocblas_datatype_bf16_r, model_dim, d_up_bf16.ptr,
        rocblas_datatype_bf16_r, mlp, &beta, d_final_bf16.ptr,
        rocblas_datatype_f32_r, model_dim, d_final_bf16.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_mlp_bias_and_residual_bf16();
  };

  impl_->stats.layer0_block_mlp_f32_ms =
      TimeSamples(impl_->options, run_mlp_f32);
  impl_->stats.layer0_block_mlp_bf16_ms =
      TimeSamples(impl_->options, run_mlp_bf16);

  std::vector<float> final_f32(static_cast<size_t>(rows) * model_dim);
  std::vector<float> final_bf16(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(final_f32.data(), d_final_f32.ptr,
                      final_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(final_bf16.data(), d_final_bf16.ptr,
                      final_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Final comparison is BF16-route output against the F32-shadow output from
  // the same device graph. It includes patch BF16 rounding, LN0 quantization,
  // QKV/attention/attn_out precision effects, LN1 quantization, and MLP BF16
  // routing. The per-boundary stats above identify where the error first
  // enters; this final stat answers whether it compounds after a full layer.
  const ErrorStats final_error = ComputeError(final_bf16, final_f32);
  impl_->stats.layer0_block_validated = true;
  impl_->stats.layer0_block_bf16_max_abs_error = final_error.max_abs;
  impl_->stats.layer0_block_bf16_rms_error = final_error.rms;
  if (impl_->options.verbose) {
    const float f32_device_ms = impl_->stats.layer0_block_qkv_f32_ms +
                                impl_->stats.layer0_block_attn_out_f32_ms +
                                impl_->stats.layer0_block_mlp_f32_ms;
    const float bf16_device_ms = impl_->stats.layer0_block_qkv_bf16_ms +
                                 impl_->stats.layer0_block_attn_out_bf16_ms +
                                 impl_->stats.layer0_block_mlp_bf16_ms;
    std::printf("layer0_block_attention_ref  host_f32=%8.3f s "
                "host_bf16=%8.3f s\n",
                impl_->stats.layer0_block_attention_f32_seconds,
                impl_->stats.layer0_block_attention_bf16_seconds);
    std::printf("layer0_block_f32            qkv=%8.3f ms "
                "attn_out=%8.3f ms mlp=%8.3f ms device_sum=%8.3f ms\n",
                impl_->stats.layer0_block_qkv_f32_ms,
                impl_->stats.layer0_block_attn_out_f32_ms,
                impl_->stats.layer0_block_mlp_f32_ms, f32_device_ms);
    std::printf("layer0_block_bf16           qkv=%8.3f ms "
                "attn_out=%8.3f ms mlp=%8.3f ms device_sum=%8.3f ms\n",
                impl_->stats.layer0_block_qkv_bf16_ms,
                impl_->stats.layer0_block_attn_out_bf16_ms,
                impl_->stats.layer0_block_mlp_bf16_ms, bf16_device_ms);
    std::printf("layer0_block_bf16_final     max_abs_vs_f32=%9.6f "
                "rms_vs_f32=%9.6f\n",
                impl_->stats.layer0_block_bf16_max_abs_error,
                impl_->stats.layer0_block_bf16_rms_error);
  }
  return true;
}

bool PaliGemma2VitHipBackend::ValidateLayer0DeviceAttention(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image) {
  // Device attention-core boundary. This validates replacing the current host
  // attention reference with:
  //   SplitQKV -> rocBLAS QK -> HIP softmax -> rocBLAS AV -> pack heads.
  //
  // QKV is still produced by the already-validated HIP projection routes. The
  // comparison is against the scalar host attention reference for the same QKV
  // tensor so errors are attributable to the device attention core.
  if (!UploadProjectionWeights(weights)) return false;

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int model_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  const LayerWeightsPtrs& layer0 = *weights.VitLayer(0);
  const LayerConfig& layer_config = layer0.layer_config;
  const int heads = static_cast<int>(layer_config.heads);
  const int qkv_dim = static_cast<int>(layer_config.qkv_dim);
  const int qkv = heads * 3 * qkv_dim;
  const int att_cols = heads * qkv_dim;
  HWY_ASSERT(layer_config.heads == layer_config.kv_heads);
  HWY_ASSERT(att_cols == model_dim);
  HWY_ASSERT(layer0.vit.qkv_einsum_w.Rows() == static_cast<size_t>(qkv));
  HWY_ASSERT(layer0.vit.qkv_einsum_w.Cols() ==
             static_cast<size_t>(model_dim));

  const std::vector<float> patches = BuildImagePatches(image, rows, patch);
  const std::vector<float> x0 =
      BuildPatchEmbeddingReference(model_config, weights, patches);
  const std::vector<float> pre_att = LayerNormReference(
      x0, rows, model_dim, layer0.vit.layer_norm_0_scale,
      layer0.vit.layer_norm_0_bias);
  const std::vector<rocblas_bfloat16> pre_att_bf16 =
      F32ToRocblasBF16(pre_att);

  ResidentF32Mat qkv_f32 =
      UploadBF16MatAsF32RocblasB("layer0.qkv", layer0.vit.qkv_einsum_w,
                                 impl_->options.verbose);

  const int threads = 256;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int qkv_count = rows * qkv;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;

  DeviceBuffer<float> d_pre_att(pre_att.size());
  DeviceBuffer<rocblas_bfloat16> d_pre_att_bf16(pre_att_bf16.size());
  DeviceBuffer<float> d_qkv_f32(static_cast<size_t>(rows) * qkv);
  DeviceBuffer<float> d_qkv_bf16(static_cast<size_t>(rows) * qkv);
  DeviceBuffer<float> d_qkv_bias(static_cast<size_t>(qkv));
  HIP_CHECK(hipMemcpy(d_pre_att.ptr, pre_att.data(),
                      pre_att.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pre_att_bf16.ptr, pre_att_bf16.data(),
                      pre_att_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qkv_bias.ptr, layer0.vit.qkv_einsum_b.PackedScale1(),
                      static_cast<size_t>(qkv) * sizeof(float),
                      hipMemcpyHostToDevice));

  auto add_qkv_bias = [&](float* out) {
    hipLaunchKernelGGL(AddBiasKernel, dim3(qkv_blocks), dim3(threads), 0, 0,
                       out, d_qkv_bias.ptr, rows, qkv);
    HIP_CHECK(hipGetLastError());
  };
  auto run_qkv_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, qkv,
        rows, model_dim, &alpha, qkv_f32.device->ptr, rocblas_datatype_f32_r,
        qkv, d_pre_att.ptr, rocblas_datatype_f32_r, model_dim, &beta,
        d_qkv_f32.ptr, rocblas_datatype_f32_r, qkv, d_qkv_f32.ptr,
        rocblas_datatype_f32_r, qkv, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_qkv_bias(d_qkv_f32.ptr);
  };
  auto run_qkv_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, qkv,
        rows, model_dim, &alpha, impl_->resident->layers[0].qkv.device->ptr,
        rocblas_datatype_bf16_r, qkv, d_pre_att_bf16.ptr,
        rocblas_datatype_bf16_r, model_dim, &beta, d_qkv_bf16.ptr,
        rocblas_datatype_f32_r, qkv, d_qkv_bf16.ptr, rocblas_datatype_f32_r,
        qkv, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
    add_qkv_bias(d_qkv_bf16.ptr);
  };

  // Run QKV once before computing host references and device-attention timings.
  // Timing of the attention core below intentionally excludes QKV projection,
  // which is already covered by the projection-boundary validators.
  run_qkv_f32();
  run_qkv_bf16();
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> qkv_host_f32(static_cast<size_t>(rows) * qkv);
  std::vector<float> qkv_host_bf16(static_cast<size_t>(rows) * qkv);
  HIP_CHECK(hipMemcpy(qkv_host_f32.data(), d_qkv_f32.ptr,
                      qkv_host_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(qkv_host_bf16.data(), d_qkv_bf16.ptr,
                      qkv_host_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  const double host_f32_start = hwy::platform::Now();
  const std::vector<float> host_att_f32 =
      ComputeAttentionWeightedSumReference(qkv_host_f32, rows, heads, qkv_dim);
  impl_->stats.layer0_device_attention_host_f32_seconds =
      hwy::platform::Now() - host_f32_start;
  const double host_bf16_start = hwy::platform::Now();
  const std::vector<float> host_att_bf16 =
      ComputeAttentionWeightedSumReference(qkv_host_bf16, rows, heads, qkv_dim);
  impl_->stats.layer0_device_attention_host_bf16_seconds =
      hwy::platform::Now() - host_bf16_start;

  const size_t head_count = static_cast<size_t>(heads) * rows * qkv_dim;
  const size_t scores_count = static_cast<size_t>(heads) * rows * rows;
  DeviceBuffer<float> d_q_f32(head_count);
  DeviceBuffer<float> d_k_f32(head_count);
  DeviceBuffer<float> d_v_f32(head_count);
  DeviceBuffer<float> d_scores_f32(scores_count);
  DeviceBuffer<float> d_att_head_f32(head_count);
  DeviceBuffer<float> d_att_row_f32(static_cast<size_t>(rows) * att_cols);
  DeviceBuffer<float> d_q_bf16(head_count);
  DeviceBuffer<float> d_k_bf16(head_count);
  DeviceBuffer<float> d_v_bf16(head_count);
  DeviceBuffer<float> d_scores_bf16(scores_count);
  DeviceBuffer<float> d_att_head_bf16(head_count);
  DeviceBuffer<float> d_att_row_bf16(static_cast<size_t>(rows) * att_cols);

  impl_->stats.layer0_device_attention_f32_ms = TimeDeviceAttention(
      impl_->handle, impl_->options, d_qkv_f32.ptr, d_q_f32.ptr, d_k_f32.ptr,
      d_v_f32.ptr, d_scores_f32.ptr, d_att_head_f32.ptr, d_att_row_f32.ptr,
      rows, heads, qkv_dim);
  impl_->stats.layer0_device_attention_bf16_ms = TimeDeviceAttention(
      impl_->handle, impl_->options, d_qkv_bf16.ptr, d_q_bf16.ptr,
      d_k_bf16.ptr, d_v_bf16.ptr, d_scores_bf16.ptr, d_att_head_bf16.ptr,
      d_att_row_bf16.ptr, rows, heads, qkv_dim);

  std::vector<float> device_att_f32(static_cast<size_t>(rows) * att_cols);
  std::vector<float> device_att_bf16(static_cast<size_t>(rows) * att_cols);
  HIP_CHECK(hipMemcpy(device_att_f32.data(), d_att_row_f32.ptr,
                      device_att_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(device_att_bf16.data(), d_att_row_bf16.ptr,
                      device_att_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  const ErrorStats f32_error = ComputeError(device_att_f32, host_att_f32);
  const ErrorStats bf16_error = ComputeError(device_att_bf16, host_att_bf16);
  impl_->stats.layer0_device_attention_validated = true;
  impl_->stats.layer0_device_attention_f32_max_abs_error = f32_error.max_abs;
  impl_->stats.layer0_device_attention_f32_rms_error = f32_error.rms;
  impl_->stats.layer0_device_attention_bf16_max_abs_error = bf16_error.max_abs;
  impl_->stats.layer0_device_attention_bf16_rms_error = bf16_error.rms;
  if (impl_->options.verbose) {
    std::printf("layer0_device_attention_ref host_f32=%8.3f s "
                "host_bf16=%8.3f s\n",
                impl_->stats.layer0_device_attention_host_f32_seconds,
                impl_->stats.layer0_device_attention_host_bf16_seconds);
    std::printf("layer0_device_attention_f32 time=%8.3f ms "
                "max_abs=%9.6f rms=%9.6f\n",
                impl_->stats.layer0_device_attention_f32_ms,
                impl_->stats.layer0_device_attention_f32_max_abs_error,
                impl_->stats.layer0_device_attention_f32_rms_error);
    std::printf("layer0_device_attention_bf16 time=%8.3f ms "
                "max_abs=%9.6f rms=%9.6f\n",
                impl_->stats.layer0_device_attention_bf16_ms,
                impl_->stats.layer0_device_attention_bf16_max_abs_error,
                impl_->stats.layer0_device_attention_bf16_rms_error);
  }
  return true;
}

bool PaliGemma2VitHipBackend::ValidateLayer0BlockDeviceNorm(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image) {
  // Whole-layer boundary with device input preparation wired into the graph:
  //   patches -> patch embedding -> layer_norm_0 -> QKV -> device QK/softmax/AV
  //      -> attn_out_w + residual -> layer_norm_1 -> MLP -> final residual.
  //
  // Compared with ValidateLayer0Block, this removes the host attention
  // weighted-sum round-trip before attn_out_w and keeps patch embedding plus
  // both layernorms resident on HIP. Host tensors are used only as validation
  // oracles, not as inputs to QKV or MLP.
  if (!UploadProjectionWeights(weights)) return false;

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int model_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  const LayerWeightsPtrs& layer0 = *weights.VitLayer(0);
  const LayerConfig& layer_config = layer0.layer_config;
  if (layer_config.activation != ActivationType::Gelu) {
    std::fprintf(stderr,
                 "Layer0 device-attention block validation only implements "
                 "GELU.\n");
    return false;
  }

  const int heads = static_cast<int>(layer_config.heads);
  const int qkv_dim = static_cast<int>(layer_config.qkv_dim);
  const int qkv = heads * 3 * qkv_dim;
  const int att_cols = heads * qkv_dim;
  const int mlp = static_cast<int>(layer_config.ff_hidden_dim);
  HWY_ASSERT(layer_config.heads == layer_config.kv_heads);
  HWY_ASSERT(att_cols == model_dim);
  HWY_ASSERT(layer0.vit.qkv_einsum_w.Rows() == static_cast<size_t>(qkv));
  HWY_ASSERT(layer0.vit.qkv_einsum_w.Cols() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.attn_out_w.Rows() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.attn_out_w.Cols() == static_cast<size_t>(att_cols));
  HWY_ASSERT(layer0.vit.linear_0_w.Rows() == static_cast<size_t>(mlp));
  HWY_ASSERT(layer0.vit.linear_0_w.Cols() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.linear_1_w.Rows() ==
             static_cast<size_t>(model_dim));
  HWY_ASSERT(layer0.vit.linear_1_w.Cols() == static_cast<size_t>(mlp));
  if (impl_->resident->patch_embed_f32 == nullptr) {
    std::fprintf(stderr,
                 "Device-input layer0 validation requires resident F32 patch "
                 "embedding weights.\n");
    return false;
  }

  // CPU-side image preprocessing is deliberately still outside this HIP
  // experiment. The current gemma.cpp image path already extracts patches on
  // the host, so this boundary starts with the same [rows, patch] F32 patch
  // matrix that the CPU EmbedImagePatches function would feed to CallMatMul.
  const std::vector<float> patches = BuildImagePatches(image, rows, patch);
  // BF16 patches are the fast-path activation input for rocBLAS. Keeping both
  // versions lets this validation report how much error is introduced exactly
  // at the first GPU precision boundary instead of attributing it to QKV/MLP.
  const std::vector<rocblas_bfloat16> patches_bf16 =
      F32ToRocblasBF16(patches);
  // x0_reference is never fed back into the device graph. It is a scalar host
  // oracle for the patch GEMM + bias + position embedding result, used only for
  // reporting patch-level error after the HIP kernels have run.
  const std::vector<float> x0_reference =
      BuildPatchEmbeddingReference(model_config, weights, patches);
  const std::vector<float> pos =
      CopyPosEmbeddingToF32(weights.vit_img_pos_embedding);

  std::vector<float> mlp_bias0(static_cast<size_t>(mlp), 0.0f);
  std::vector<float> mlp_bias1(static_cast<size_t>(model_dim), 0.0f);
  if (layer_config.ff_biases) {
    std::memcpy(mlp_bias0.data(), layer0.vit.linear_0_b.PackedScale1(),
                mlp_bias0.size() * sizeof(float));
    std::memcpy(mlp_bias1.data(), layer0.vit.linear_1_b.PackedScale1(),
                mlp_bias1.size() * sizeof(float));
  }

  // F32 shadow weights are allocated only for validation. The future steady
  // inference route should use resident BF16 weights, but the F32 shadow path
  // gives a same-graph reference that isolates BF16 activation/weight drift.
  ResidentF32Mat qkv_f32 =
      UploadBF16MatAsF32RocblasB("layer0.qkv", layer0.vit.qkv_einsum_w,
                                 impl_->options.verbose);
  ResidentF32Mat attn_out_f32 =
      UploadBF16MatAsF32RocblasB("layer0.attn_out",
                                 layer0.vit.attn_out_w,
                                 impl_->options.verbose);
  ResidentF32Mat linear0_f32 =
      UploadBF16MatAsF32RocblasB("layer0.linear_0",
                                 layer0.vit.linear_0_w,
                                 impl_->options.verbose);
  ResidentF32Mat linear1_f32 =
      UploadBF16MatAsF32RocblasB("layer0.linear_1",
                                 layer0.vit.linear_1_w,
                                 impl_->options.verbose);

  const int threads = 256;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int qkv_count = rows * qkv;
  const int qkv_blocks = (qkv_count + threads - 1) / threads;
  const int att_count = rows * att_cols;
  const int att_blocks = (att_count + threads - 1) / threads;
  const int model_count = rows * model_dim;
  const int model_blocks = (model_count + threads - 1) / threads;

  // Device input preparation for layer0:
  //   image patches -> patch embedding + bias + pos -> layer_norm_0.
  //
  // From this point forward, the real dataflow for the validation run remains
  // on HIP. Host copies later in this block are readback checks only; QKV
  // consumes d_pre_att and d_pre_att_bf16 produced by the kernels below.
  HWY_ASSERT(pos.size() == static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_patches(patches.size());
  DeviceBuffer<rocblas_bfloat16> d_patches_bf16(patches_bf16.size());
  DeviceBuffer<float> d_x0_f32(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_x0_bf16_f32(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_patch_bias(static_cast<size_t>(model_dim));
  DeviceBuffer<float> d_pos(pos.size());
  HIP_CHECK(hipMemcpy(d_patches.ptr, patches.data(),
                      patches.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_patches_bf16.ptr, patches_bf16.data(),
                      patches_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_patch_bias.ptr,
                      weights.vit_img_embedding_bias.PackedScale1(),
                      static_cast<size_t>(model_dim) * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pos.ptr, pos.data(), pos.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  // The patch GEMM writes only the projection result. This epilogue adds the
  // image embedding bias and the per-token position embedding, matching
  // gemma/vit.cc::EmbedImagePatches before the transformer layers start.
  auto add_patch_bias_and_pos = [&](float* out) {
    hipLaunchKernelGGL(AddBiasAndPosKernel, dim3(model_blocks), dim3(threads),
                       0, 0, out, d_patch_bias.ptr, d_pos.ptr, rows,
                       model_dim);
    HIP_CHECK(hipGetLastError());
  };
  // F32 route: F32 patches x F32-shadow patch weights -> F32 x0. This is the
  // tight reference for the device graph because it preserves current CPU patch
  // precision while still exercising the HIP layout and epilogue.
  auto run_patch_embedding_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, patch, &alpha,
        impl_->resident->patch_embed_f32->device->ptr, rocblas_datatype_f32_r,
        model_dim, d_patches.ptr, rocblas_datatype_f32_r, patch, &beta,
        d_x0_f32.ptr, rocblas_datatype_f32_r, model_dim, d_x0_f32.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_patch_bias_and_pos(d_x0_f32.ptr);
  };
  // BF16 route: BF16 patches x resident BF16 patch weights -> F32 x0. This is
  // the candidate inference route. Its output remains F32 because layernorm and
  // residual math are most naturally accumulated in F32 before the next BF16
  // GEMM boundary.
  auto run_patch_embedding_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, patch, &alpha, impl_->resident->patch_embed.device->ptr,
        rocblas_datatype_bf16_r, model_dim, d_patches_bf16.ptr,
        rocblas_datatype_bf16_r, patch, &beta, d_x0_bf16_f32.ptr,
        rocblas_datatype_f32_r, model_dim, d_x0_bf16_f32.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_patch_bias_and_pos(d_x0_bf16_f32.ptr);
  };
  impl_->stats.layer0_block_device_patch_f32_ms =
      TimeSamples(impl_->options, run_patch_embedding_f32);
  impl_->stats.layer0_block_device_patch_bf16_ms =
      TimeSamples(impl_->options, run_patch_embedding_bf16);

  // layer_norm_0 is the first transformer-layer operation after patch
  // embedding. Keeping it on device removes the old host-produced pre_att
  // tensor from the path feeding QKV.
  const std::vector<float> ln0_scale =
      CopyActivationMatToF32(layer0.vit.layer_norm_0_scale);
  const std::vector<float> ln0_bias =
      CopyActivationMatToF32(layer0.vit.layer_norm_0_bias);
  DeviceBuffer<float> d_ln0_scale(ln0_scale.size());
  DeviceBuffer<float> d_ln0_bias(ln0_bias.size());
  DeviceBuffer<float> d_pre_att(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_pre_att_bf16_f32(static_cast<size_t>(rows) *
                                         model_dim);
  DeviceBuffer<rocblas_bfloat16> d_pre_att_bf16(
      static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(d_ln0_scale.ptr, ln0_scale.data(),
                      ln0_scale.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ln0_bias.ptr, ln0_bias.data(),
                      ln0_bias.size() * sizeof(float), hipMemcpyHostToDevice));

  const size_t ln0_shared =
      static_cast<size_t>(threads) * 2 * sizeof(float);
  // F32 route: normalize d_x0_f32 into F32 d_pre_att for the F32-shadow QKV.
  auto run_ln0_f32 = [&]() {
    hipLaunchKernelGGL(LayerNormKernel, dim3(rows), dim3(threads),
                       ln0_shared, 0, d_x0_f32.ptr, d_ln0_scale.ptr,
                       d_ln0_bias.ptr, d_pre_att.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
  };
  // BF16 route: normalize the BF16 patch-embedding result in F32, then quantize
  // the normalized activation for the resident BF16 QKV GEMM. The conversion is
  // counted in LN0 timing because it is part of preparing QKV input.
  auto run_ln0_bf16 = [&]() {
    hipLaunchKernelGGL(LayerNormKernel, dim3(rows), dim3(threads),
                       ln0_shared, 0, d_x0_bf16_f32.ptr,
                       d_ln0_scale.ptr, d_ln0_bias.ptr, d_pre_att_bf16_f32.ptr,
                       rows, model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(F32ToBF16Kernel, dim3(model_blocks), dim3(threads), 0,
                       0, d_pre_att_bf16_f32.ptr, d_pre_att_bf16.ptr,
                       model_count);
    HIP_CHECK(hipGetLastError());
  };
  impl_->stats.layer0_block_device_ln0_f32_ms =
      TimeSamples(impl_->options, run_ln0_f32);
  impl_->stats.layer0_block_device_ln0_bf16_ms =
      TimeSamples(impl_->options, run_ln0_bf16);

  // Validation-only readback. None of these host vectors is copied back to the
  // GPU. Patch errors are measured against x0_reference; LN0 errors are
  // measured against host LayerNormReference using the device-produced x0 so
  // the LN comparison does not double-count patch precision differences.
  std::vector<float> x0_device_f32(static_cast<size_t>(rows) * model_dim);
  std::vector<float> x0_device_bf16(static_cast<size_t>(rows) * model_dim);
  std::vector<float> pre_att_device_f32(static_cast<size_t>(rows) *
                                        model_dim);
  std::vector<float> pre_att_device_bf16(static_cast<size_t>(rows) *
                                         model_dim);
  HIP_CHECK(hipMemcpy(x0_device_f32.data(), d_x0_f32.ptr,
                      x0_device_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(x0_device_bf16.data(), d_x0_bf16_f32.ptr,
                      x0_device_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(pre_att_device_f32.data(), d_pre_att.ptr,
                      pre_att_device_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(pre_att_device_bf16.data(), d_pre_att_bf16_f32.ptr,
                      pre_att_device_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  const ErrorStats patch_error_f32 =
      ComputeError(x0_device_f32, x0_reference);
  const ErrorStats patch_error_bf16 =
      ComputeError(x0_device_bf16, x0_reference);
  const std::vector<float> pre_att_reference_f32 = LayerNormReference(
      x0_device_f32, rows, model_dim, layer0.vit.layer_norm_0_scale,
      layer0.vit.layer_norm_0_bias);
  const std::vector<float> pre_att_reference_bf16 = LayerNormReference(
      x0_device_bf16, rows, model_dim, layer0.vit.layer_norm_0_scale,
      layer0.vit.layer_norm_0_bias);
  const ErrorStats ln0_error_f32 =
      ComputeError(pre_att_device_f32, pre_att_reference_f32);
  const ErrorStats ln0_error_bf16 =
      ComputeError(pre_att_device_bf16, pre_att_reference_bf16);
  impl_->stats.layer0_block_device_patch_f32_max_abs_error =
      patch_error_f32.max_abs;
  impl_->stats.layer0_block_device_patch_f32_rms_error =
      patch_error_f32.rms;
  impl_->stats.layer0_block_device_patch_bf16_max_abs_error =
      patch_error_bf16.max_abs;
  impl_->stats.layer0_block_device_patch_bf16_rms_error =
      patch_error_bf16.rms;
  impl_->stats.layer0_block_device_ln0_f32_max_abs_error =
      ln0_error_f32.max_abs;
  impl_->stats.layer0_block_device_ln0_f32_rms_error = ln0_error_f32.rms;
  impl_->stats.layer0_block_device_ln0_bf16_max_abs_error =
      ln0_error_bf16.max_abs;
  impl_->stats.layer0_block_device_ln0_bf16_rms_error = ln0_error_bf16.rms;

  // QKV consumes the device-produced LN0 outputs directly. The F32 route uses
  // d_pre_att, while the BF16 route uses d_pre_att_bf16 produced by run_ln0_bf16.
  DeviceBuffer<float> d_qkv_f32(static_cast<size_t>(rows) * qkv);
  DeviceBuffer<float> d_qkv_bf16(static_cast<size_t>(rows) * qkv);
  DeviceBuffer<float> d_qkv_bias(static_cast<size_t>(qkv));
  HIP_CHECK(hipMemcpy(d_qkv_bias.ptr, layer0.vit.qkv_einsum_b.PackedScale1(),
                      static_cast<size_t>(qkv) * sizeof(float),
                      hipMemcpyHostToDevice));

  auto add_qkv_bias = [&](float* out) {
    hipLaunchKernelGGL(AddBiasKernel, dim3(qkv_blocks), dim3(threads), 0, 0,
                       out, d_qkv_bias.ptr, rows, qkv);
    HIP_CHECK(hipGetLastError());
  };
  auto run_qkv_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, qkv,
        rows, model_dim, &alpha, qkv_f32.device->ptr, rocblas_datatype_f32_r,
        qkv, d_pre_att.ptr, rocblas_datatype_f32_r, model_dim, &beta,
        d_qkv_f32.ptr, rocblas_datatype_f32_r, qkv, d_qkv_f32.ptr,
        rocblas_datatype_f32_r, qkv, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_qkv_bias(d_qkv_f32.ptr);
  };
  auto run_qkv_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, qkv,
        rows, model_dim, &alpha, impl_->resident->layers[0].qkv.device->ptr,
        rocblas_datatype_bf16_r, qkv, d_pre_att_bf16.ptr,
        rocblas_datatype_bf16_r, model_dim, &beta, d_qkv_bf16.ptr,
        rocblas_datatype_f32_r, qkv, d_qkv_bf16.ptr, rocblas_datatype_f32_r,
        qkv, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
    add_qkv_bias(d_qkv_bf16.ptr);
  };

  impl_->stats.layer0_block_device_attention_qkv_f32_ms =
      TimeSamples(impl_->options, run_qkv_f32);
  impl_->stats.layer0_block_device_attention_qkv_bf16_ms =
      TimeSamples(impl_->options, run_qkv_bf16);

  const size_t head_count = static_cast<size_t>(heads) * rows * qkv_dim;
  const size_t scores_count = static_cast<size_t>(heads) * rows * rows;
  DeviceBuffer<float> d_q_f32(head_count);
  DeviceBuffer<float> d_k_f32(head_count);
  DeviceBuffer<float> d_v_f32(head_count);
  DeviceBuffer<float> d_scores_f32(scores_count);
  DeviceBuffer<float> d_att_head_f32(head_count);
  DeviceBuffer<float> d_att_row_f32(static_cast<size_t>(rows) * att_cols);
  DeviceBuffer<float> d_q_bf16(head_count);
  DeviceBuffer<float> d_k_bf16(head_count);
  DeviceBuffer<float> d_v_bf16(head_count);
  DeviceBuffer<float> d_scores_bf16(scores_count);
  DeviceBuffer<float> d_att_head_bf16(head_count);
  DeviceBuffer<float> d_att_row_bf16_f32(static_cast<size_t>(rows) * att_cols);
  DeviceBuffer<rocblas_bfloat16> d_att_row_bf16(
      static_cast<size_t>(rows) * att_cols);

  impl_->stats.layer0_block_device_attention_attn_f32_ms =
      TimeDeviceAttention(impl_->handle, impl_->options, d_qkv_f32.ptr,
                          d_q_f32.ptr, d_k_f32.ptr, d_v_f32.ptr,
                          d_scores_f32.ptr, d_att_head_f32.ptr,
                          d_att_row_f32.ptr, rows, heads, qkv_dim);
  impl_->stats.layer0_block_device_attention_attn_bf16_ms =
      TimeDeviceAttention(impl_->handle, impl_->options, d_qkv_bf16.ptr,
                          d_q_bf16.ptr, d_k_bf16.ptr, d_v_bf16.ptr,
                          d_scores_bf16.ptr, d_att_head_bf16.ptr,
                          d_att_row_bf16_f32.ptr, rows, heads, qkv_dim);

  DeviceBuffer<float> d_att_bias(static_cast<size_t>(model_dim));
  DeviceBuffer<float> d_x_att_f32(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_x_att_bf16(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(d_att_bias.ptr, layer0.vit.attn_out_b.PackedScale1(),
                      static_cast<size_t>(model_dim) * sizeof(float),
                      hipMemcpyHostToDevice));

  // Attention-output residual boundary. In the original CPU graph this adds the
  // attention output back into x0. For this validation route, use the matching
  // device-produced x0 for each precision path: F32 route adds d_x0_f32, BF16
  // route adds d_x0_bf16_f32. That preserves the precision history instead of
  // silently mixing BF16 attention with a host/F32 residual.
  auto add_att_bias_and_residual_f32 = [&]() {
    hipLaunchKernelGGL(AddBiasKernel, dim3(model_blocks), dim3(threads), 0, 0,
                       d_x_att_f32.ptr, d_att_bias.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(AddResidualKernel, dim3(model_blocks), dim3(threads), 0,
                       0, d_x_att_f32.ptr, d_x0_f32.ptr, model_count);
    HIP_CHECK(hipGetLastError());
  };
  auto add_att_bias_and_residual_bf16 = [&]() {
    hipLaunchKernelGGL(AddBiasKernel, dim3(model_blocks), dim3(threads), 0, 0,
                       d_x_att_bf16.ptr, d_att_bias.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(AddResidualKernel, dim3(model_blocks), dim3(threads), 0,
                       0, d_x_att_bf16.ptr, d_x0_bf16_f32.ptr, model_count);
    HIP_CHECK(hipGetLastError());
  };

  // F32 route: device attention output feeds F32-shadow attn_out_w, then the
  // F32 patch-embedding residual is added on device.
  auto run_attn_out_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, att_cols, &alpha, attn_out_f32.device->ptr,
        rocblas_datatype_f32_r, model_dim, d_att_row_f32.ptr,
        rocblas_datatype_f32_r, att_cols, &beta, d_x_att_f32.ptr,
        rocblas_datatype_f32_r, model_dim, d_x_att_f32.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_att_bias_and_residual_f32();
  };
  // BF16 route: quantize the F32 row-major attention result only at the GEMM
  // input boundary, then use resident BF16 attn_out_w and the BF16-path x0
  // residual. The quantization is timed with attn_out because it prepares that
  // projection's input.
  auto run_attn_out_bf16 = [&]() {
    hipLaunchKernelGGL(F32ToBF16Kernel, dim3(att_blocks), dim3(threads), 0, 0,
                       d_att_row_bf16_f32.ptr, d_att_row_bf16.ptr, att_count);
    HIP_CHECK(hipGetLastError());
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, att_cols, &alpha,
        impl_->resident->layers[0].attn_out.device->ptr,
        rocblas_datatype_bf16_r, model_dim, d_att_row_bf16.ptr,
        rocblas_datatype_bf16_r, att_cols, &beta, d_x_att_bf16.ptr,
        rocblas_datatype_f32_r, model_dim, d_x_att_bf16.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_att_bias_and_residual_bf16();
  };

  impl_->stats.layer0_block_device_attention_attn_out_f32_ms =
      TimeSamples(impl_->options, run_attn_out_f32);
  impl_->stats.layer0_block_device_attention_attn_out_bf16_ms =
      TimeSamples(impl_->options, run_attn_out_bf16);

  // Move layer_norm_1 onto HIP for the actual tensor handoff into the MLP.
  // This is the boundary that previously forced x_att back to the host. The
  // scale/bias tensors are small, so upload them as plain F32 vectors once for
  // this validation run.
  const std::vector<float> ln1_scale =
      CopyActivationMatToF32(layer0.vit.layer_norm_1_scale);
  const std::vector<float> ln1_bias =
      CopyActivationMatToF32(layer0.vit.layer_norm_1_bias);
  DeviceBuffer<float> d_ln1_scale(ln1_scale.size());
  DeviceBuffer<float> d_ln1_bias(ln1_bias.size());
  DeviceBuffer<float> d_pre_ffw_f32(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_pre_ffw_bf16_f32(static_cast<size_t>(rows) *
                                         model_dim);
  DeviceBuffer<rocblas_bfloat16> d_pre_ffw_bf16(
      static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(d_ln1_scale.ptr, ln1_scale.data(),
                      ln1_scale.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ln1_bias.ptr, ln1_bias.data(),
                      ln1_bias.size() * sizeof(float), hipMemcpyHostToDevice));

  const size_t layer_norm_shared =
      static_cast<size_t>(threads) * 2 * sizeof(float);
  // F32 shadow route: keep the normalized activation as F32 so it can feed the
  // F32-shadow MLP weights. This path is the local semantic reference for the
  // faster BF16 route below.
  auto run_ln1_f32 = [&]() {
    hipLaunchKernelGGL(LayerNormKernel, dim3(rows), dim3(threads),
                       layer_norm_shared, 0, d_x_att_f32.ptr,
                       d_ln1_scale.ptr, d_ln1_bias.ptr, d_pre_ffw_f32.ptr,
                       rows, model_dim);
    HIP_CHECK(hipGetLastError());
  };
  // BF16 route: run LN in F32 for numerical stability, then quantize the LN
  // output immediately because the following MLP GEMM consumes BF16
  // activations. The conversion is included in LN timing so MLP timing is not
  // credited with hidden input preparation.
  auto run_ln1_bf16 = [&]() {
    hipLaunchKernelGGL(LayerNormKernel, dim3(rows), dim3(threads),
                       layer_norm_shared, 0, d_x_att_bf16.ptr,
                       d_ln1_scale.ptr, d_ln1_bias.ptr, d_pre_ffw_bf16_f32.ptr,
                       rows, model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(F32ToBF16Kernel, dim3(model_blocks), dim3(threads), 0,
                       0, d_pre_ffw_bf16_f32.ptr, d_pre_ffw_bf16.ptr,
                       model_count);
    HIP_CHECK(hipGetLastError());
  };

  impl_->stats.layer0_block_device_attention_ln1_f32_ms =
      TimeSamples(impl_->options, run_ln1_f32);
  impl_->stats.layer0_block_device_attention_ln1_bf16_ms =
      TimeSamples(impl_->options, run_ln1_bf16);

  // These host copies are validation-only. They compare device LN against the
  // existing scalar host reference, but they are not used to feed the MLP. The
  // MLP reads d_pre_ffw_f32/d_pre_ffw_bf16 directly from the previous kernels.
  std::vector<float> x_att_f32(static_cast<size_t>(rows) * model_dim);
  std::vector<float> x_att_bf16(static_cast<size_t>(rows) * model_dim);
  std::vector<float> pre_ffw_device_f32(static_cast<size_t>(rows) *
                                        model_dim);
  std::vector<float> pre_ffw_device_bf16(static_cast<size_t>(rows) *
                                         model_dim);
  HIP_CHECK(hipMemcpy(x_att_f32.data(), d_x_att_f32.ptr,
                      x_att_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(x_att_bf16.data(), d_x_att_bf16.ptr,
                      x_att_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(pre_ffw_device_f32.data(), d_pre_ffw_f32.ptr,
                      pre_ffw_device_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(pre_ffw_device_bf16.data(), d_pre_ffw_bf16_f32.ptr,
                      pre_ffw_device_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  const std::vector<float> pre_ffw_reference_f32 = LayerNormReference(
      x_att_f32, rows, model_dim, layer0.vit.layer_norm_1_scale,
      layer0.vit.layer_norm_1_bias);
  const std::vector<float> pre_ffw_reference_bf16 = LayerNormReference(
      x_att_bf16, rows, model_dim, layer0.vit.layer_norm_1_scale,
      layer0.vit.layer_norm_1_bias);
  const ErrorStats ln1_error_f32 =
      ComputeError(pre_ffw_device_f32, pre_ffw_reference_f32);
  const ErrorStats ln1_error_bf16 =
      ComputeError(pre_ffw_device_bf16, pre_ffw_reference_bf16);
  impl_->stats.layer0_block_device_attention_ln1_f32_max_abs_error =
      ln1_error_f32.max_abs;
  impl_->stats.layer0_block_device_attention_ln1_f32_rms_error =
      ln1_error_f32.rms;
  impl_->stats.layer0_block_device_attention_ln1_bf16_max_abs_error =
      ln1_error_bf16.max_abs;
  impl_->stats.layer0_block_device_attention_ln1_bf16_rms_error =
      ln1_error_bf16.rms;

  DeviceBuffer<float> d_up_f32(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<float> d_up_bf16_accum(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<rocblas_bfloat16> d_up_bf16(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<float> d_final_f32(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_final_bf16(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_mlp_bias0(mlp_bias0.size());
  DeviceBuffer<float> d_mlp_bias1(mlp_bias1.size());
  HIP_CHECK(hipMemcpy(d_mlp_bias0.ptr, mlp_bias0.data(),
                      mlp_bias0.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_mlp_bias1.ptr, mlp_bias1.data(),
                      mlp_bias1.size() * sizeof(float), hipMemcpyHostToDevice));

  const int up_count = rows * mlp;
  const int up_blocks = (up_count + threads - 1) / threads;
  auto add_mlp_bias_and_residual_f32 = [&]() {
    hipLaunchKernelGGL(AddBiasKernel, dim3(model_blocks), dim3(threads), 0, 0,
                       d_final_f32.ptr, d_mlp_bias1.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(AddResidualKernel, dim3(model_blocks), dim3(threads), 0,
                       0, d_final_f32.ptr, d_x_att_f32.ptr, model_count);
    HIP_CHECK(hipGetLastError());
  };
  auto add_mlp_bias_and_residual_bf16 = [&]() {
    hipLaunchKernelGGL(AddBiasKernel, dim3(model_blocks), dim3(threads), 0, 0,
                       d_final_bf16.ptr, d_mlp_bias1.ptr, rows, model_dim);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(AddResidualKernel, dim3(model_blocks), dim3(threads), 0,
                       0, d_final_bf16.ptr, d_x_att_bf16.ptr, model_count);
    HIP_CHECK(hipGetLastError());
  };

  auto run_mlp_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, mlp,
        rows, model_dim, &alpha, linear0_f32.device->ptr,
        rocblas_datatype_f32_r, mlp, d_pre_ffw_f32.ptr,
        rocblas_datatype_f32_r, model_dim, &beta, d_up_f32.ptr,
        rocblas_datatype_f32_r, mlp, d_up_f32.ptr, rocblas_datatype_f32_r, mlp,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
        /*solution_index=*/0, /*flags=*/0));
    hipLaunchKernelGGL(AddBiasKernel, dim3(up_blocks), dim3(threads), 0, 0,
                       d_up_f32.ptr, d_mlp_bias0.ptr, rows, mlp);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(GeluKernel, dim3(up_blocks), dim3(threads), 0, 0,
                       d_up_f32.ptr, up_count);
    HIP_CHECK(hipGetLastError());
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, mlp, &alpha, linear1_f32.device->ptr,
        rocblas_datatype_f32_r, model_dim, d_up_f32.ptr,
        rocblas_datatype_f32_r, mlp, &beta, d_final_f32.ptr,
        rocblas_datatype_f32_r, model_dim, d_final_f32.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_mlp_bias_and_residual_f32();
  };

  auto run_mlp_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, mlp,
        rows, model_dim, &alpha, impl_->resident->layers[0].linear_0.device->ptr,
        rocblas_datatype_bf16_r, mlp, d_pre_ffw_bf16.ptr,
        rocblas_datatype_bf16_r, model_dim, &beta, d_up_bf16_accum.ptr,
        rocblas_datatype_f32_r, mlp, d_up_bf16_accum.ptr,
        rocblas_datatype_f32_r, mlp, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    hipLaunchKernelGGL(AddBiasToBF16Kernel, dim3(up_blocks), dim3(threads), 0,
                       0, d_up_bf16_accum.ptr, d_mlp_bias0.ptr,
                       d_up_bf16.ptr, rows, mlp);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(GeluBF16Kernel, dim3(up_blocks), dim3(threads), 0, 0,
                       d_up_bf16.ptr, up_count);
    HIP_CHECK(hipGetLastError());
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, mlp, &alpha,
        impl_->resident->layers[0].linear_1.device->ptr,
        rocblas_datatype_bf16_r, model_dim, d_up_bf16.ptr,
        rocblas_datatype_bf16_r, mlp, &beta, d_final_bf16.ptr,
        rocblas_datatype_f32_r, model_dim, d_final_bf16.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_mlp_bias_and_residual_bf16();
  };

  impl_->stats.layer0_block_device_attention_mlp_f32_ms =
      TimeSamples(impl_->options, run_mlp_f32);
  impl_->stats.layer0_block_device_attention_mlp_bf16_ms =
      TimeSamples(impl_->options, run_mlp_bf16);

  std::vector<float> final_f32(static_cast<size_t>(rows) * model_dim);
  std::vector<float> final_bf16(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(final_f32.data(), d_final_f32.ptr,
                      final_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(final_bf16.data(), d_final_bf16.ptr,
                      final_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  const ErrorStats final_error = ComputeError(final_bf16, final_f32);
  impl_->stats.layer0_block_device_attention_validated = true;
  impl_->stats.layer0_block_device_norm_validated = true;
  impl_->stats.layer0_block_device_attention_bf16_max_abs_error =
      final_error.max_abs;
  impl_->stats.layer0_block_device_attention_bf16_rms_error = final_error.rms;
  if (impl_->options.verbose) {
    const float f32_device_ms =
        impl_->stats.layer0_block_device_patch_f32_ms +
        impl_->stats.layer0_block_device_ln0_f32_ms +
        impl_->stats.layer0_block_device_attention_qkv_f32_ms +
        impl_->stats.layer0_block_device_attention_attn_f32_ms +
        impl_->stats.layer0_block_device_attention_attn_out_f32_ms +
        impl_->stats.layer0_block_device_attention_ln1_f32_ms +
        impl_->stats.layer0_block_device_attention_mlp_f32_ms;
    const float bf16_device_ms =
        impl_->stats.layer0_block_device_patch_bf16_ms +
        impl_->stats.layer0_block_device_ln0_bf16_ms +
        impl_->stats.layer0_block_device_attention_qkv_bf16_ms +
        impl_->stats.layer0_block_device_attention_attn_bf16_ms +
        impl_->stats.layer0_block_device_attention_attn_out_bf16_ms +
        impl_->stats.layer0_block_device_attention_ln1_bf16_ms +
        impl_->stats.layer0_block_device_attention_mlp_bf16_ms;
    std::printf("layer0_block_device_input_f32 patch=%8.3f ms "
                "ln0=%8.3f ms qkv=%8.3f ms attn=%8.3f ms "
                "attn_out=%8.3f ms ln1=%8.3f ms mlp=%8.3f ms "
                "device_sum=%8.3f ms\n",
                impl_->stats.layer0_block_device_patch_f32_ms,
                impl_->stats.layer0_block_device_ln0_f32_ms,
                impl_->stats.layer0_block_device_attention_qkv_f32_ms,
                impl_->stats.layer0_block_device_attention_attn_f32_ms,
                impl_->stats.layer0_block_device_attention_attn_out_f32_ms,
                impl_->stats.layer0_block_device_attention_ln1_f32_ms,
                impl_->stats.layer0_block_device_attention_mlp_f32_ms,
                f32_device_ms);
    std::printf("layer0_block_device_input_bf16 patch=%8.3f ms "
                "ln0=%8.3f ms qkv=%8.3f ms attn=%8.3f ms "
                "attn_out=%8.3f ms ln1=%8.3f ms mlp=%8.3f ms "
                "device_sum=%8.3f ms\n",
                impl_->stats.layer0_block_device_patch_bf16_ms,
                impl_->stats.layer0_block_device_ln0_bf16_ms,
                impl_->stats.layer0_block_device_attention_qkv_bf16_ms,
                impl_->stats.layer0_block_device_attention_attn_bf16_ms,
                impl_->stats.layer0_block_device_attention_attn_out_bf16_ms,
                impl_->stats.layer0_block_device_attention_ln1_bf16_ms,
                impl_->stats.layer0_block_device_attention_mlp_bf16_ms,
                bf16_device_ms);
    std::printf("layer0_block_device_input_patch f32_max_abs=%9.6f "
                "f32_rms=%9.6f bf16_max_abs=%9.6f bf16_rms=%9.6f\n",
                impl_->stats.layer0_block_device_patch_f32_max_abs_error,
                impl_->stats.layer0_block_device_patch_f32_rms_error,
                impl_->stats.layer0_block_device_patch_bf16_max_abs_error,
                impl_->stats.layer0_block_device_patch_bf16_rms_error);
    std::printf("layer0_block_device_input_ln0   f32_max_abs=%9.6f "
                "f32_rms=%9.6f bf16_max_abs=%9.6f bf16_rms=%9.6f\n",
                impl_->stats.layer0_block_device_ln0_f32_max_abs_error,
                impl_->stats.layer0_block_device_ln0_f32_rms_error,
                impl_->stats.layer0_block_device_ln0_bf16_max_abs_error,
                impl_->stats.layer0_block_device_ln0_bf16_rms_error);
    std::printf("layer0_block_device_norm_ln1  f32_max_abs=%9.6f "
                "f32_rms=%9.6f bf16_max_abs=%9.6f bf16_rms=%9.6f\n",
                impl_->stats
                    .layer0_block_device_attention_ln1_f32_max_abs_error,
                impl_->stats.layer0_block_device_attention_ln1_f32_rms_error,
                impl_->stats
                    .layer0_block_device_attention_ln1_bf16_max_abs_error,
                impl_->stats.layer0_block_device_attention_ln1_bf16_rms_error);
    std::printf("layer0_block_device_norm_final max_abs_vs_f32=%9.6f "
                "rms_vs_f32=%9.6f\n",
                impl_->stats
                    .layer0_block_device_attention_bf16_max_abs_error,
                impl_->stats.layer0_block_device_attention_bf16_rms_error);
  }
  return true;
}

bool PaliGemma2VitHipBackend::ValidateLayer0BlockDeviceAttention(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image) {
  return ValidateLayer0BlockDeviceNorm(model_config, weights, image);
}

bool PaliGemma2VitHipBackend::ValidateLayerPrefix2DeviceNorm(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image) {
  // First multi-layer validation boundary. This reuses RunDeviceVitLayer for
  // layer0 and layer1 so we can prove the device-resident layer body composes
  // before expanding to the full ViT stack.
  if (!UploadProjectionWeights(weights)) return false;
  if (weights.vit_layers.size() < 2 || impl_->resident->layers.size() < 2) {
    std::fprintf(stderr, "Two-layer prefix validation requires >=2 ViT layers.\n");
    return false;
  }
  if (impl_->resident->patch_embed_f32 == nullptr) {
    std::fprintf(stderr,
                 "Two-layer prefix validation requires resident F32 patch "
                 "embedding weights.\n");
    return false;
  }

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int model_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  const int threads = 256;
  const int model_count = rows * model_dim;
  const int model_blocks = (model_count + threads - 1) / threads;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const std::vector<float> patches = BuildImagePatches(image, rows, patch);
  const std::vector<rocblas_bfloat16> patches_bf16 =
      F32ToRocblasBF16(patches);
  const std::vector<float> x0_reference =
      BuildPatchEmbeddingReference(model_config, weights, patches);
  const std::vector<float> pos =
      CopyPosEmbeddingToF32(weights.vit_img_pos_embedding);
  HWY_ASSERT(pos.size() == static_cast<size_t>(rows) * model_dim);

  DeviceBuffer<float> d_patches(patches.size());
  DeviceBuffer<rocblas_bfloat16> d_patches_bf16(patches_bf16.size());
  DeviceBuffer<float> d_x_f32_a(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_x_f32_b(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_x_bf16_a(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_x_bf16_b(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_patch_bias(static_cast<size_t>(model_dim));
  DeviceBuffer<float> d_pos(pos.size());
  HIP_CHECK(hipMemcpy(d_patches.ptr, patches.data(),
                      patches.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_patches_bf16.ptr, patches_bf16.data(),
                      patches_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_patch_bias.ptr,
                      weights.vit_img_embedding_bias.PackedScale1(),
                      static_cast<size_t>(model_dim) * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pos.ptr, pos.data(), pos.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  auto add_patch_bias_and_pos = [&](float* out) {
    hipLaunchKernelGGL(AddBiasAndPosKernel, dim3(model_blocks), dim3(threads),
                       0, 0, out, d_patch_bias.ptr, d_pos.ptr, rows,
                       model_dim);
    HIP_CHECK(hipGetLastError());
  };
  auto run_patch_embedding_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, patch, &alpha,
        impl_->resident->patch_embed_f32->device->ptr, rocblas_datatype_f32_r,
        model_dim, d_patches.ptr, rocblas_datatype_f32_r, patch, &beta,
        d_x_f32_a.ptr, rocblas_datatype_f32_r, model_dim, d_x_f32_a.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_patch_bias_and_pos(d_x_f32_a.ptr);
  };
  auto run_patch_embedding_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, patch, &alpha, impl_->resident->patch_embed.device->ptr,
        rocblas_datatype_bf16_r, model_dim, d_patches_bf16.ptr,
        rocblas_datatype_bf16_r, patch, &beta, d_x_bf16_a.ptr,
        rocblas_datatype_f32_r, model_dim, d_x_bf16_a.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_patch_bias_and_pos(d_x_bf16_a.ptr);
  };
  impl_->stats.layer_prefix2_patch_f32_ms =
      TimeSamples(impl_->options, run_patch_embedding_f32);
  impl_->stats.layer_prefix2_patch_bf16_ms =
      TimeSamples(impl_->options, run_patch_embedding_bf16);

  std::vector<float> x0_device_f32(static_cast<size_t>(rows) * model_dim);
  std::vector<float> x0_device_bf16(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(x0_device_f32.data(), d_x_f32_a.ptr,
                      x0_device_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(x0_device_bf16.data(), d_x_bf16_a.ptr,
                      x0_device_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  const ErrorStats patch_error_f32 =
      ComputeError(x0_device_f32, x0_reference);
  const ErrorStats patch_error_bf16 =
      ComputeError(x0_device_bf16, x0_reference);
  impl_->stats.layer_prefix2_patch_f32_max_abs_error =
      patch_error_f32.max_abs;
  impl_->stats.layer_prefix2_patch_f32_rms_error = patch_error_f32.rms;
  impl_->stats.layer_prefix2_patch_bf16_max_abs_error =
      patch_error_bf16.max_abs;
  impl_->stats.layer_prefix2_patch_bf16_rms_error = patch_error_bf16.rms;

  DeviceVitLayerTimings layer0;
  DeviceVitLayerTimings layer1;
  if (!RunDeviceVitLayer(impl_->handle, impl_->options,
                         impl_->resident->layers[0], *weights.VitLayer(0),
                         /*layer_idx=*/0, rows, model_dim, d_x_f32_a.ptr,
                         d_x_bf16_a.ptr, d_x_f32_b.ptr, d_x_bf16_b.ptr,
                         layer0)) {
    return false;
  }
  if (!RunDeviceVitLayer(impl_->handle, impl_->options,
                         impl_->resident->layers[1], *weights.VitLayer(1),
                         /*layer_idx=*/1, rows, model_dim, d_x_f32_b.ptr,
                         d_x_bf16_b.ptr, d_x_f32_a.ptr, d_x_bf16_a.ptr,
                         layer1)) {
    return false;
  }

  auto sum_f32 = [](const DeviceVitLayerTimings& timing) {
    return timing.ln0_f32_ms + timing.qkv_f32_ms + timing.attn_f32_ms +
           timing.attn_out_f32_ms + timing.ln1_f32_ms + timing.mlp_f32_ms;
  };
  auto sum_bf16 = [](const DeviceVitLayerTimings& timing) {
    return timing.ln0_bf16_ms + timing.qkv_bf16_ms + timing.attn_bf16_ms +
           timing.attn_out_bf16_ms + timing.ln1_bf16_ms + timing.mlp_bf16_ms;
  };

  impl_->stats.layer_prefix2_ln0_f32_ms =
      layer0.ln0_f32_ms + layer1.ln0_f32_ms;
  impl_->stats.layer_prefix2_ln0_bf16_ms =
      layer0.ln0_bf16_ms + layer1.ln0_bf16_ms;
  impl_->stats.layer_prefix2_qkv_f32_ms =
      layer0.qkv_f32_ms + layer1.qkv_f32_ms;
  impl_->stats.layer_prefix2_qkv_bf16_ms =
      layer0.qkv_bf16_ms + layer1.qkv_bf16_ms;
  impl_->stats.layer_prefix2_attn_f32_ms =
      layer0.attn_f32_ms + layer1.attn_f32_ms;
  impl_->stats.layer_prefix2_attn_bf16_ms =
      layer0.attn_bf16_ms + layer1.attn_bf16_ms;
  impl_->stats.layer_prefix2_attn_out_f32_ms =
      layer0.attn_out_f32_ms + layer1.attn_out_f32_ms;
  impl_->stats.layer_prefix2_attn_out_bf16_ms =
      layer0.attn_out_bf16_ms + layer1.attn_out_bf16_ms;
  impl_->stats.layer_prefix2_ln1_f32_ms =
      layer0.ln1_f32_ms + layer1.ln1_f32_ms;
  impl_->stats.layer_prefix2_ln1_bf16_ms =
      layer0.ln1_bf16_ms + layer1.ln1_bf16_ms;
  impl_->stats.layer_prefix2_mlp_f32_ms =
      layer0.mlp_f32_ms + layer1.mlp_f32_ms;
  impl_->stats.layer_prefix2_mlp_bf16_ms =
      layer0.mlp_bf16_ms + layer1.mlp_bf16_ms;

  std::vector<float> final_f32(static_cast<size_t>(rows) * model_dim);
  std::vector<float> final_bf16(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(final_f32.data(), d_x_f32_a.ptr,
                      final_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(final_bf16.data(), d_x_bf16_a.ptr,
                      final_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  const ErrorStats final_error = ComputeError(final_bf16, final_f32);
  impl_->stats.layer_prefix2_bf16_max_abs_error = final_error.max_abs;
  impl_->stats.layer_prefix2_bf16_rms_error = final_error.rms;
  impl_->stats.layer_prefix2_device_norm_validated = true;

  if (impl_->options.verbose) {
    const float f32_total = impl_->stats.layer_prefix2_patch_f32_ms +
                            sum_f32(layer0) + sum_f32(layer1);
    const float bf16_total = impl_->stats.layer_prefix2_patch_bf16_ms +
                             sum_bf16(layer0) + sum_bf16(layer1);
    std::printf("layer_prefix2_device_layer0_f32 ln0=%8.3f ms qkv=%8.3f ms "
                "attn=%8.3f ms attn_out=%8.3f ms ln1=%8.3f ms "
                "mlp=%8.3f ms layer_sum=%8.3f ms\n",
                layer0.ln0_f32_ms, layer0.qkv_f32_ms, layer0.attn_f32_ms,
                layer0.attn_out_f32_ms, layer0.ln1_f32_ms, layer0.mlp_f32_ms,
                sum_f32(layer0));
    std::printf("layer_prefix2_device_layer1_f32 ln0=%8.3f ms qkv=%8.3f ms "
                "attn=%8.3f ms attn_out=%8.3f ms ln1=%8.3f ms "
                "mlp=%8.3f ms layer_sum=%8.3f ms\n",
                layer1.ln0_f32_ms, layer1.qkv_f32_ms, layer1.attn_f32_ms,
                layer1.attn_out_f32_ms, layer1.ln1_f32_ms, layer1.mlp_f32_ms,
                sum_f32(layer1));
    std::printf("layer_prefix2_device_layer0_bf16 ln0=%8.3f ms qkv=%8.3f ms "
                "attn=%8.3f ms attn_out=%8.3f ms ln1=%8.3f ms "
                "mlp=%8.3f ms layer_sum=%8.3f ms\n",
                layer0.ln0_bf16_ms, layer0.qkv_bf16_ms, layer0.attn_bf16_ms,
                layer0.attn_out_bf16_ms, layer0.ln1_bf16_ms,
                layer0.mlp_bf16_ms, sum_bf16(layer0));
    std::printf("layer_prefix2_device_layer1_bf16 ln0=%8.3f ms qkv=%8.3f ms "
                "attn=%8.3f ms attn_out=%8.3f ms ln1=%8.3f ms "
                "mlp=%8.3f ms layer_sum=%8.3f ms\n",
                layer1.ln0_bf16_ms, layer1.qkv_bf16_ms, layer1.attn_bf16_ms,
                layer1.attn_out_bf16_ms, layer1.ln1_bf16_ms,
                layer1.mlp_bf16_ms, sum_bf16(layer1));
    std::printf("layer_prefix2_device_total_f32  patch=%8.3f ms "
                "device_sum=%8.3f ms\n",
                impl_->stats.layer_prefix2_patch_f32_ms, f32_total);
    std::printf("layer_prefix2_device_total_bf16 patch=%8.3f ms "
                "device_sum=%8.3f ms\n",
                impl_->stats.layer_prefix2_patch_bf16_ms, bf16_total);
    std::printf("layer_prefix2_device_patch      f32_max_abs=%9.6f "
                "f32_rms=%9.6f bf16_max_abs=%9.6f bf16_rms=%9.6f\n",
                impl_->stats.layer_prefix2_patch_f32_max_abs_error,
                impl_->stats.layer_prefix2_patch_f32_rms_error,
                impl_->stats.layer_prefix2_patch_bf16_max_abs_error,
                impl_->stats.layer_prefix2_patch_bf16_rms_error);
    std::printf("layer_prefix2_device_final      max_abs_vs_f32=%9.6f "
                "rms_vs_f32=%9.6f\n",
                impl_->stats.layer_prefix2_bf16_max_abs_error,
                impl_->stats.layer_prefix2_bf16_rms_error);
  }
  return true;
}

bool PaliGemma2VitHipBackend::ValidateLayerStackDeviceNorm(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image) {
  // Full transformer-stack validation boundary. This deliberately stops at the
  // output of the last ViT transformer layer so the next step can isolate final
  // encoder norm, pooling, and image projection head as a separate contract.
  if (!UploadProjectionWeights(weights)) return false;
  const size_t num_layers = weights.vit_layers.size();
  if (num_layers == 0 || impl_->resident->layers.size() != num_layers) {
    std::fprintf(stderr,
                 "Layer-stack validation requires matching nonzero ViT layer "
                 "counts: weights=%zu resident=%zu.\n",
                 weights.vit_layers.size(), impl_->resident->layers.size());
    return false;
  }
  if (impl_->resident->patch_embed_f32 == nullptr) {
    std::fprintf(stderr,
                 "Layer-stack validation requires resident F32 patch "
                 "embedding weights.\n");
    return false;
  }

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int model_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  const int threads = 256;
  const int model_count = rows * model_dim;
  const int model_blocks = (model_count + threads - 1) / threads;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Build the exact same patch matrix and CPU scalar patch reference used by
  // the earlier boundary tests. The final stack comparison is BF16-vs-F32 on
  // device, but keeping the patch reference here catches regressions at the
  // input boundary before layer error accumulation makes them harder to read.
  const std::vector<float> patches = BuildImagePatches(image, rows, patch);
  const std::vector<rocblas_bfloat16> patches_bf16 =
      F32ToRocblasBF16(patches);
  const std::vector<float> x0_reference =
      BuildPatchEmbeddingReference(model_config, weights, patches);
  const std::vector<float> pos =
      CopyPosEmbeddingToF32(weights.vit_img_pos_embedding);
  HWY_ASSERT(pos.size() == static_cast<size_t>(rows) * model_dim);

  // Two ping-pong buffers are enough for the whole stack because each layer only
  // needs its input tensor and a distinct output tensor. Keeping both the F32
  // shadow route and the BF16 candidate route resident avoids host round trips
  // between transformer layers.
  DeviceBuffer<float> d_patches(patches.size());
  DeviceBuffer<rocblas_bfloat16> d_patches_bf16(patches_bf16.size());
  DeviceBuffer<float> d_x_f32_a(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_x_f32_b(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_x_bf16_a(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_x_bf16_b(static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_patch_bias(static_cast<size_t>(model_dim));
  DeviceBuffer<float> d_pos(pos.size());
  HIP_CHECK(hipMemcpy(d_patches.ptr, patches.data(),
                      patches.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_patches_bf16.ptr, patches_bf16.data(),
                      patches_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_patch_bias.ptr,
                      weights.vit_img_embedding_bias.PackedScale1(),
                      static_cast<size_t>(model_dim) * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pos.ptr, pos.data(), pos.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  auto add_patch_bias_and_pos = [&](float* out) {
    hipLaunchKernelGGL(AddBiasAndPosKernel, dim3(model_blocks), dim3(threads),
                       0, 0, out, d_patch_bias.ptr, d_pos.ptr, rows,
                       model_dim);
    HIP_CHECK(hipGetLastError());
  };
  auto run_patch_embedding_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, patch, &alpha,
        impl_->resident->patch_embed_f32->device->ptr, rocblas_datatype_f32_r,
        model_dim, d_patches.ptr, rocblas_datatype_f32_r, patch, &beta,
        d_x_f32_a.ptr, rocblas_datatype_f32_r, model_dim, d_x_f32_a.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_patch_bias_and_pos(d_x_f32_a.ptr);
  };
  auto run_patch_embedding_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none,
        model_dim, rows, patch, &alpha, impl_->resident->patch_embed.device->ptr,
        rocblas_datatype_bf16_r, model_dim, d_patches_bf16.ptr,
        rocblas_datatype_bf16_r, patch, &beta, d_x_bf16_a.ptr,
        rocblas_datatype_f32_r, model_dim, d_x_bf16_a.ptr,
        rocblas_datatype_f32_r, model_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_patch_bias_and_pos(d_x_bf16_a.ptr);
  };
  impl_->stats.layer_stack_patch_f32_ms =
      TimeSamples(impl_->options, run_patch_embedding_f32);
  impl_->stats.layer_stack_patch_bf16_ms =
      TimeSamples(impl_->options, run_patch_embedding_bf16);

  std::vector<float> x0_device_f32(static_cast<size_t>(rows) * model_dim);
  std::vector<float> x0_device_bf16(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(x0_device_f32.data(), d_x_f32_a.ptr,
                      x0_device_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(x0_device_bf16.data(), d_x_bf16_a.ptr,
                      x0_device_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  const ErrorStats patch_error_f32 =
      ComputeError(x0_device_f32, x0_reference);
  const ErrorStats patch_error_bf16 =
      ComputeError(x0_device_bf16, x0_reference);
  impl_->stats.layer_stack_patch_f32_max_abs_error = patch_error_f32.max_abs;
  impl_->stats.layer_stack_patch_f32_rms_error = patch_error_f32.rms;
  impl_->stats.layer_stack_patch_bf16_max_abs_error = patch_error_bf16.max_abs;
  impl_->stats.layer_stack_patch_bf16_rms_error = patch_error_bf16.rms;

  impl_->stats.layer_stack_ln0_f32_ms = 0.0f;
  impl_->stats.layer_stack_ln0_bf16_ms = 0.0f;
  impl_->stats.layer_stack_qkv_f32_ms = 0.0f;
  impl_->stats.layer_stack_qkv_bf16_ms = 0.0f;
  impl_->stats.layer_stack_attn_f32_ms = 0.0f;
  impl_->stats.layer_stack_attn_bf16_ms = 0.0f;
  impl_->stats.layer_stack_attn_out_f32_ms = 0.0f;
  impl_->stats.layer_stack_attn_out_bf16_ms = 0.0f;
  impl_->stats.layer_stack_ln1_f32_ms = 0.0f;
  impl_->stats.layer_stack_ln1_bf16_ms = 0.0f;
  impl_->stats.layer_stack_mlp_f32_ms = 0.0f;
  impl_->stats.layer_stack_mlp_bf16_ms = 0.0f;

  std::vector<DeviceVitLayerTimings> layer_timings(num_layers);
  for (size_t i = 0; i < num_layers; ++i) {
    // Alternate A/B buffers per layer. Even-numbered layers consume A and write
    // B; odd-numbered layers consume B and write A. This mirrors how the future
    // integrated path can keep the whole ViT stack resident without allocating a
    // separate activation tensor for every layer.
    const bool even = (i % 2) == 0;
    const float* d_x_in_f32 = even ? d_x_f32_a.ptr : d_x_f32_b.ptr;
    const float* d_x_in_bf16 = even ? d_x_bf16_a.ptr : d_x_bf16_b.ptr;
    float* d_x_out_f32 = even ? d_x_f32_b.ptr : d_x_f32_a.ptr;
    float* d_x_out_bf16 = even ? d_x_bf16_b.ptr : d_x_bf16_a.ptr;
    if (!RunDeviceVitLayer(impl_->handle, impl_->options,
                           impl_->resident->layers[i], *weights.VitLayer(i),
                           static_cast<int>(i), rows, model_dim, d_x_in_f32,
                           d_x_in_bf16, d_x_out_f32, d_x_out_bf16,
                           layer_timings[i])) {
      return false;
    }

    impl_->stats.layer_stack_ln0_f32_ms += layer_timings[i].ln0_f32_ms;
    impl_->stats.layer_stack_ln0_bf16_ms += layer_timings[i].ln0_bf16_ms;
    impl_->stats.layer_stack_qkv_f32_ms += layer_timings[i].qkv_f32_ms;
    impl_->stats.layer_stack_qkv_bf16_ms += layer_timings[i].qkv_bf16_ms;
    impl_->stats.layer_stack_attn_f32_ms += layer_timings[i].attn_f32_ms;
    impl_->stats.layer_stack_attn_bf16_ms += layer_timings[i].attn_bf16_ms;
    impl_->stats.layer_stack_attn_out_f32_ms +=
        layer_timings[i].attn_out_f32_ms;
    impl_->stats.layer_stack_attn_out_bf16_ms +=
        layer_timings[i].attn_out_bf16_ms;
    impl_->stats.layer_stack_ln1_f32_ms += layer_timings[i].ln1_f32_ms;
    impl_->stats.layer_stack_ln1_bf16_ms += layer_timings[i].ln1_bf16_ms;
    impl_->stats.layer_stack_mlp_f32_ms += layer_timings[i].mlp_f32_ms;
    impl_->stats.layer_stack_mlp_bf16_ms += layer_timings[i].mlp_bf16_ms;
  }

  // After N layers, an even N returns to buffer A and an odd N ends in buffer B.
  // Copy back only the final activation tensors; all intermediate validation has
  // already happened on device or at the patch input boundary.
  const bool even_layers = (num_layers % 2) == 0;
  const float* final_f32_ptr = even_layers ? d_x_f32_a.ptr : d_x_f32_b.ptr;
  const float* final_bf16_ptr =
      even_layers ? d_x_bf16_a.ptr : d_x_bf16_b.ptr;
  std::vector<float> final_f32(static_cast<size_t>(rows) * model_dim);
  std::vector<float> final_bf16(static_cast<size_t>(rows) * model_dim);
  HIP_CHECK(hipMemcpy(final_f32.data(), final_f32_ptr,
                      final_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(final_bf16.data(), final_bf16_ptr,
                      final_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  const ErrorStats final_error = ComputeError(final_bf16, final_f32);
  impl_->stats.layer_stack_bf16_max_abs_error = final_error.max_abs;
  impl_->stats.layer_stack_bf16_rms_error = final_error.rms;
  impl_->stats.layer_stack_device_norm_validated = true;

  auto sum_f32 = [](const DeviceVitLayerTimings& timing) {
    return timing.ln0_f32_ms + timing.qkv_f32_ms + timing.attn_f32_ms +
           timing.attn_out_f32_ms + timing.ln1_f32_ms + timing.mlp_f32_ms;
  };
  auto sum_bf16 = [](const DeviceVitLayerTimings& timing) {
    return timing.ln0_bf16_ms + timing.qkv_bf16_ms + timing.attn_bf16_ms +
           timing.attn_out_bf16_ms + timing.ln1_bf16_ms + timing.mlp_bf16_ms;
  };

  if (impl_->options.verbose) {
    const float f32_total =
        impl_->stats.layer_stack_patch_f32_ms +
        impl_->stats.layer_stack_ln0_f32_ms +
        impl_->stats.layer_stack_qkv_f32_ms +
        impl_->stats.layer_stack_attn_f32_ms +
        impl_->stats.layer_stack_attn_out_f32_ms +
        impl_->stats.layer_stack_ln1_f32_ms +
        impl_->stats.layer_stack_mlp_f32_ms;
    const float bf16_total =
        impl_->stats.layer_stack_patch_bf16_ms +
        impl_->stats.layer_stack_ln0_bf16_ms +
        impl_->stats.layer_stack_qkv_bf16_ms +
        impl_->stats.layer_stack_attn_bf16_ms +
        impl_->stats.layer_stack_attn_out_bf16_ms +
        impl_->stats.layer_stack_ln1_bf16_ms +
        impl_->stats.layer_stack_mlp_bf16_ms;
    std::printf("layer_stack_device_total_f32  layers=%zu patch=%8.3f ms "
                "ln0=%8.3f ms qkv=%8.3f ms attn=%8.3f ms "
                "attn_out=%8.3f ms ln1=%8.3f ms mlp=%8.3f ms "
                "device_sum=%8.3f ms\n",
                num_layers, impl_->stats.layer_stack_patch_f32_ms,
                impl_->stats.layer_stack_ln0_f32_ms,
                impl_->stats.layer_stack_qkv_f32_ms,
                impl_->stats.layer_stack_attn_f32_ms,
                impl_->stats.layer_stack_attn_out_f32_ms,
                impl_->stats.layer_stack_ln1_f32_ms,
                impl_->stats.layer_stack_mlp_f32_ms, f32_total);
    std::printf("layer_stack_device_total_bf16 layers=%zu patch=%8.3f ms "
                "ln0=%8.3f ms qkv=%8.3f ms attn=%8.3f ms "
                "attn_out=%8.3f ms ln1=%8.3f ms mlp=%8.3f ms "
                "device_sum=%8.3f ms\n",
                num_layers, impl_->stats.layer_stack_patch_bf16_ms,
                impl_->stats.layer_stack_ln0_bf16_ms,
                impl_->stats.layer_stack_qkv_bf16_ms,
                impl_->stats.layer_stack_attn_bf16_ms,
                impl_->stats.layer_stack_attn_out_bf16_ms,
                impl_->stats.layer_stack_ln1_bf16_ms,
                impl_->stats.layer_stack_mlp_bf16_ms, bf16_total);
    std::printf("layer_stack_device_first_last layer0_f32=%8.3f ms "
                "layer0_bf16=%8.3f ms layer%zu_f32=%8.3f ms "
                "layer%zu_bf16=%8.3f ms\n",
                sum_f32(layer_timings.front()), sum_bf16(layer_timings.front()),
                num_layers - 1, sum_f32(layer_timings.back()), num_layers - 1,
                sum_bf16(layer_timings.back()));
    std::printf("layer_stack_device_patch      f32_max_abs=%9.6f "
                "f32_rms=%9.6f bf16_max_abs=%9.6f bf16_rms=%9.6f\n",
                impl_->stats.layer_stack_patch_f32_max_abs_error,
                impl_->stats.layer_stack_patch_f32_rms_error,
                impl_->stats.layer_stack_patch_bf16_max_abs_error,
                impl_->stats.layer_stack_patch_bf16_rms_error);
    std::printf("layer_stack_device_final      max_abs_vs_f32=%9.6f "
                "rms_vs_f32=%9.6f\n",
                impl_->stats.layer_stack_bf16_max_abs_error,
                impl_->stats.layer_stack_bf16_rms_error);
  }
  return true;
}

bool PaliGemma2VitHipBackend::ValidateImageTokensDeviceNorm(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image, ImageTokens* image_tokens,
    const ImageTokens* cpu_reference) {
  // Decoder-visible boundary for the PaliGemma2 image encoder. This extends the
  // full transformer-stack path with:
  //   last-layer output -> final encoder LayerNorm -> img_head GEMM+bias.
  //
  // PaliGemma2 224/448 uses pool_dim=1, so image_tokens has one output row per
  // ViT patch token. GEMMA_VLM's 4x4 average pooling and soft-embedding RMSNorm
  // are intentionally left unsupported here so this validation stays scoped to
  // the model family we are accelerating.
  if (!UploadProjectionWeights(weights)) return false;
  if (model_config.vit_config.pool_dim != 1 ||
      model_config.wrapping == PromptWrapping::GEMMA_VLM) {
    std::fprintf(stderr,
                 "Image-token HIP validation currently supports only the "
                 "unpooled PaliGemma2 ViT path.\n");
    return false;
  }

  const size_t num_layers = weights.vit_layers.size();
  if (num_layers == 0 || impl_->resident->layers.size() != num_layers) {
    std::fprintf(stderr,
                 "Image-token validation requires matching nonzero ViT layer "
                 "counts: weights=%zu resident=%zu.\n",
                 weights.vit_layers.size(), impl_->resident->layers.size());
    return false;
  }
  if (impl_->resident->patch_embed_f32 == nullptr) {
    std::fprintf(stderr,
                 "Image-token validation requires resident F32 patch "
                 "embedding weights.\n");
    return false;
  }

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int vit_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int llm_dim = static_cast<int>(model_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  HWY_ASSERT(weights.vit_img_head_kernel.Rows() ==
             static_cast<size_t>(llm_dim));
  HWY_ASSERT(weights.vit_img_head_kernel.Cols() ==
             static_cast<size_t>(vit_dim));
  if (image_tokens != nullptr) {
    HWY_ASSERT(image_tokens->Rows() == static_cast<size_t>(rows));
    HWY_ASSERT(image_tokens->Cols() == static_cast<size_t>(llm_dim));
  }
  if (cpu_reference != nullptr) {
    HWY_ASSERT(cpu_reference->Rows() == static_cast<size_t>(rows));
    HWY_ASSERT(cpu_reference->Cols() == static_cast<size_t>(llm_dim));
  }

  const int threads = 256;
  const int vit_count = rows * vit_dim;
  const int vit_blocks = (vit_count + threads - 1) / threads;
  const int token_count = rows * llm_dim;
  const int token_blocks = (token_count + threads - 1) / threads;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const std::vector<float> patches = BuildImagePatches(image, rows, patch);
  const std::vector<rocblas_bfloat16> patches_bf16 =
      F32ToRocblasBF16(patches);
  const std::vector<float> pos =
      CopyPosEmbeddingToF32(weights.vit_img_pos_embedding);
  HWY_ASSERT(pos.size() == static_cast<size_t>(rows) * vit_dim);

  DeviceBuffer<float> d_patches(patches.size());
  DeviceBuffer<rocblas_bfloat16> d_patches_bf16(patches_bf16.size());
  DeviceBuffer<float> d_x_f32_a(static_cast<size_t>(rows) * vit_dim);
  DeviceBuffer<float> d_x_f32_b(static_cast<size_t>(rows) * vit_dim);
  DeviceBuffer<float> d_x_bf16_a(static_cast<size_t>(rows) * vit_dim);
  DeviceBuffer<float> d_x_bf16_b(static_cast<size_t>(rows) * vit_dim);
  DeviceBuffer<float> d_patch_bias(static_cast<size_t>(vit_dim));
  DeviceBuffer<float> d_pos(pos.size());
  HIP_CHECK(hipMemcpy(d_patches.ptr, patches.data(),
                      patches.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_patches_bf16.ptr, patches_bf16.data(),
                      patches_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_patch_bias.ptr,
                      weights.vit_img_embedding_bias.PackedScale1(),
                      static_cast<size_t>(vit_dim) * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pos.ptr, pos.data(), pos.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  auto add_patch_bias_and_pos = [&](float* out) {
    hipLaunchKernelGGL(AddBiasAndPosKernel, dim3(vit_blocks), dim3(threads), 0,
                       0, out, d_patch_bias.ptr, d_pos.ptr, rows, vit_dim);
    HIP_CHECK(hipGetLastError());
  };
  auto run_patch_embedding_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, vit_dim,
        rows, patch, &alpha, impl_->resident->patch_embed_f32->device->ptr,
        rocblas_datatype_f32_r, vit_dim, d_patches.ptr, rocblas_datatype_f32_r,
        patch, &beta, d_x_f32_a.ptr, rocblas_datatype_f32_r, vit_dim,
        d_x_f32_a.ptr, rocblas_datatype_f32_r, vit_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_patch_bias_and_pos(d_x_f32_a.ptr);
  };
  auto run_patch_embedding_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, vit_dim,
        rows, patch, &alpha, impl_->resident->patch_embed.device->ptr,
        rocblas_datatype_bf16_r, vit_dim, d_patches_bf16.ptr,
        rocblas_datatype_bf16_r, patch, &beta, d_x_bf16_a.ptr,
        rocblas_datatype_f32_r, vit_dim, d_x_bf16_a.ptr,
        rocblas_datatype_f32_r, vit_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_patch_bias_and_pos(d_x_bf16_a.ptr);
  };
  const float patch_f32_ms = TimeSamples(impl_->options, run_patch_embedding_f32);
  const float patch_bf16_ms =
      TimeSamples(impl_->options, run_patch_embedding_bf16);

  float stack_f32_ms = patch_f32_ms;
  float stack_bf16_ms = patch_bf16_ms;
  for (size_t i = 0; i < num_layers; ++i) {
    const bool even = (i % 2) == 0;
    const float* d_x_in_f32 = even ? d_x_f32_a.ptr : d_x_f32_b.ptr;
    const float* d_x_in_bf16 = even ? d_x_bf16_a.ptr : d_x_bf16_b.ptr;
    float* d_x_out_f32 = even ? d_x_f32_b.ptr : d_x_f32_a.ptr;
    float* d_x_out_bf16 = even ? d_x_bf16_b.ptr : d_x_bf16_a.ptr;
    DeviceVitLayerTimings timing;
    if (!RunDeviceVitLayer(impl_->handle, impl_->options,
                           impl_->resident->layers[i], *weights.VitLayer(i),
                           static_cast<int>(i), rows, vit_dim, d_x_in_f32,
                           d_x_in_bf16, d_x_out_f32, d_x_out_bf16, timing)) {
      return false;
    }
    stack_f32_ms += timing.ln0_f32_ms + timing.qkv_f32_ms +
                    timing.attn_f32_ms + timing.attn_out_f32_ms +
                    timing.ln1_f32_ms + timing.mlp_f32_ms;
    stack_bf16_ms += timing.ln0_bf16_ms + timing.qkv_bf16_ms +
                     timing.attn_bf16_ms + timing.attn_out_bf16_ms +
                     timing.ln1_bf16_ms + timing.mlp_bf16_ms;
  }

  const bool even_layers = (num_layers % 2) == 0;
  const float* final_stack_f32 =
      even_layers ? d_x_f32_a.ptr : d_x_f32_b.ptr;
  const float* final_stack_bf16 =
      even_layers ? d_x_bf16_a.ptr : d_x_bf16_b.ptr;

  const std::vector<float> enc_norm_scale =
      CopyActivationMatToF32(weights.vit_encoder_norm_scale);
  const std::vector<float> enc_norm_bias =
      CopyActivationMatToF32(weights.vit_encoder_norm_bias);
  HWY_ASSERT(enc_norm_scale.size() == static_cast<size_t>(vit_dim));
  HWY_ASSERT(enc_norm_bias.size() == static_cast<size_t>(vit_dim));
  DeviceBuffer<float> d_enc_norm_scale(enc_norm_scale.size());
  DeviceBuffer<float> d_enc_norm_bias(enc_norm_bias.size());
  DeviceBuffer<float> d_norm_f32(static_cast<size_t>(rows) * vit_dim);
  DeviceBuffer<rocblas_bfloat16> d_norm_bf16(static_cast<size_t>(rows) *
                                             vit_dim);
  DeviceBuffer<float> d_head_bias(static_cast<size_t>(llm_dim));
  DeviceBuffer<float> d_tokens_f32(static_cast<size_t>(rows) * llm_dim);
  DeviceBuffer<float> d_tokens_bf16(static_cast<size_t>(rows) * llm_dim);
  HIP_CHECK(hipMemcpy(d_enc_norm_scale.ptr, enc_norm_scale.data(),
                      enc_norm_scale.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_enc_norm_bias.ptr, enc_norm_bias.data(),
                      enc_norm_bias.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_head_bias.ptr, weights.vit_img_head_bias.PackedScale1(),
                      static_cast<size_t>(llm_dim) * sizeof(float),
                      hipMemcpyHostToDevice));

  const size_t layer_norm_shared =
      static_cast<size_t>(threads) * 2 * sizeof(float);
  auto run_final_norm_f32 = [&]() {
    hipLaunchKernelGGL(LayerNormKernel, dim3(rows), dim3(threads),
                       layer_norm_shared, 0, final_stack_f32,
                       d_enc_norm_scale.ptr, d_enc_norm_bias.ptr,
                       d_norm_f32.ptr, rows, vit_dim);
    HIP_CHECK(hipGetLastError());
  };
  auto run_final_norm_bf16 = [&]() {
    hipLaunchKernelGGL(LayerNormToBF16Kernel, dim3(rows), dim3(threads),
                       layer_norm_shared, 0, final_stack_bf16,
                       d_enc_norm_scale.ptr, d_enc_norm_bias.ptr,
                       d_norm_bf16.ptr, rows, vit_dim);
    HIP_CHECK(hipGetLastError());
  };
  impl_->stats.image_tokens_final_norm_f32_ms =
      TimeSamples(impl_->options, run_final_norm_f32);
  impl_->stats.image_tokens_final_norm_bf16_ms =
      TimeSamples(impl_->options, run_final_norm_bf16);

  // Validation-only F32-shadow head. The steady-state candidate route uses the
  // already-resident BF16 img_head weights below.
  ResidentF32Mat head_f32 =
      UploadBF16MatAsF32RocblasB("image_tokens.img_head",
                                 weights.vit_img_head_kernel,
                                 impl_->options.verbose);
  auto add_head_bias = [&](float* out) {
    hipLaunchKernelGGL(AddBiasKernel, dim3(token_blocks), dim3(threads), 0, 0,
                       out, d_head_bias.ptr, rows, llm_dim);
    HIP_CHECK(hipGetLastError());
  };
  auto run_head_f32 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, llm_dim,
        rows, vit_dim, &alpha, head_f32.device->ptr, rocblas_datatype_f32_r,
        llm_dim, d_norm_f32.ptr, rocblas_datatype_f32_r, vit_dim, &beta,
        d_tokens_f32.ptr, rocblas_datatype_f32_r, llm_dim, d_tokens_f32.ptr,
        rocblas_datatype_f32_r, llm_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_head_bias(d_tokens_f32.ptr);
  };
  auto run_head_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, llm_dim,
        rows, vit_dim, &alpha, impl_->resident->head.device->ptr,
        rocblas_datatype_bf16_r, llm_dim, d_norm_bf16.ptr,
        rocblas_datatype_bf16_r, vit_dim, &beta, d_tokens_bf16.ptr,
        rocblas_datatype_f32_r, llm_dim, d_tokens_bf16.ptr,
        rocblas_datatype_f32_r, llm_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_head_bias(d_tokens_bf16.ptr);
  };
  impl_->stats.image_tokens_head_f32_ms =
      TimeSamples(impl_->options, run_head_f32);
  impl_->stats.image_tokens_head_bf16_ms =
      TimeSamples(impl_->options, run_head_bf16);
  impl_->stats.image_tokens_stack_f32_ms = stack_f32_ms;
  impl_->stats.image_tokens_stack_bf16_ms = stack_bf16_ms;

  std::vector<float> tokens_f32(static_cast<size_t>(rows) * llm_dim);
  std::vector<float> tokens_bf16(static_cast<size_t>(rows) * llm_dim);
  HIP_CHECK(hipMemcpy(tokens_f32.data(), d_tokens_f32.ptr,
                      tokens_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tokens_bf16.data(), d_tokens_bf16.ptr,
                      tokens_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  if (image_tokens != nullptr) {
    CopyVectorToImageTokens(tokens_bf16, *image_tokens);
  }

  const ErrorStats bf16_vs_f32 = ComputeError(tokens_bf16, tokens_f32);
  impl_->stats.image_tokens_bf16_vs_f32_max_abs_error =
      bf16_vs_f32.max_abs;
  impl_->stats.image_tokens_bf16_vs_f32_rms_error = bf16_vs_f32.rms;
  impl_->stats.image_tokens_f32_vs_cpu_max_abs_error = 0.0f;
  impl_->stats.image_tokens_f32_vs_cpu_rms_error = 0.0f;
  impl_->stats.image_tokens_bf16_vs_cpu_max_abs_error = 0.0f;
  impl_->stats.image_tokens_bf16_vs_cpu_rms_error = 0.0f;
  if (cpu_reference != nullptr) {
    const std::vector<float> cpu_tokens =
        CopyImageTokensToVector(*cpu_reference);
    const ErrorStats f32_vs_cpu = ComputeError(tokens_f32, cpu_tokens);
    const ErrorStats bf16_vs_cpu = ComputeError(tokens_bf16, cpu_tokens);
    impl_->stats.image_tokens_f32_vs_cpu_max_abs_error =
        f32_vs_cpu.max_abs;
    impl_->stats.image_tokens_f32_vs_cpu_rms_error = f32_vs_cpu.rms;
    impl_->stats.image_tokens_bf16_vs_cpu_max_abs_error =
        bf16_vs_cpu.max_abs;
    impl_->stats.image_tokens_bf16_vs_cpu_rms_error = bf16_vs_cpu.rms;
  }
  impl_->stats.image_tokens_device_norm_validated = true;

  if (impl_->options.verbose) {
    const float total_f32 = stack_f32_ms +
                            impl_->stats.image_tokens_final_norm_f32_ms +
                            impl_->stats.image_tokens_head_f32_ms;
    const float total_bf16 = stack_bf16_ms +
                             impl_->stats.image_tokens_final_norm_bf16_ms +
                             impl_->stats.image_tokens_head_bf16_ms;
    std::printf("image_tokens_device_total_f32  layers=%zu stack=%8.3f ms "
                "final_norm=%8.3f ms head=%8.3f ms device_sum=%8.3f ms\n",
                num_layers, stack_f32_ms,
                impl_->stats.image_tokens_final_norm_f32_ms,
                impl_->stats.image_tokens_head_f32_ms, total_f32);
    std::printf("image_tokens_device_total_bf16 layers=%zu stack=%8.3f ms "
                "final_norm=%8.3f ms head=%8.3f ms device_sum=%8.3f ms\n",
                num_layers, stack_bf16_ms,
                impl_->stats.image_tokens_final_norm_bf16_ms,
                impl_->stats.image_tokens_head_bf16_ms, total_bf16);
    std::printf("image_tokens_device_final      bf16_vs_f32_max_abs=%9.6f "
                "bf16_vs_f32_rms=%9.6f\n",
                impl_->stats.image_tokens_bf16_vs_f32_max_abs_error,
                impl_->stats.image_tokens_bf16_vs_f32_rms_error);
    if (cpu_reference != nullptr) {
      std::printf("image_tokens_device_vs_cpu    f32_max_abs=%9.6f "
                  "f32_rms=%9.6f bf16_max_abs=%9.6f bf16_rms=%9.6f\n",
                  impl_->stats.image_tokens_f32_vs_cpu_max_abs_error,
                  impl_->stats.image_tokens_f32_vs_cpu_rms_error,
                  impl_->stats.image_tokens_bf16_vs_cpu_max_abs_error,
                  impl_->stats.image_tokens_bf16_vs_cpu_rms_error);
    }
  }
  return true;
}

bool PaliGemma2VitHipBackend::GenerateImageTokensDeviceNorm(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image, ImageTokens& image_tokens) {
  // Candidate return path for PaliGemma2 image tokens. This is the same BF16
  // route validated by ValidateImageTokensDeviceNorm, but without the F32-shadow
  // patch/layer/head work or BF16-vs-F32 readback.
  if (!UploadProjectionWeights(weights)) return false;
  if (model_config.vit_config.pool_dim != 1 ||
      model_config.wrapping == PromptWrapping::GEMMA_VLM) {
    std::fprintf(stderr,
                 "HIP image-token return currently supports only the unpooled "
                 "PaliGemma2 ViT path.\n");
    return false;
  }

  const size_t num_layers = weights.vit_layers.size();
  if (num_layers == 0 || impl_->resident->layers.size() != num_layers) {
    std::fprintf(stderr,
                 "HIP image-token return requires matching nonzero ViT layer "
                 "counts: weights=%zu resident=%zu.\n",
                 weights.vit_layers.size(), impl_->resident->layers.size());
    return false;
  }

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int vit_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int llm_dim = static_cast<int>(model_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  HWY_ASSERT(weights.vit_img_head_kernel.Rows() ==
             static_cast<size_t>(llm_dim));
  HWY_ASSERT(weights.vit_img_head_kernel.Cols() ==
             static_cast<size_t>(vit_dim));
  HWY_ASSERT(image_tokens.Rows() == static_cast<size_t>(rows));
  HWY_ASSERT(image_tokens.Cols() == static_cast<size_t>(llm_dim));

  const int threads = 256;
  const int vit_count = rows * vit_dim;
  const int vit_blocks = (vit_count + threads - 1) / threads;
  const int token_count = rows * llm_dim;
  const int token_blocks = (token_count + threads - 1) / threads;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const std::vector<float> patches = BuildImagePatches(image, rows, patch);
  const std::vector<rocblas_bfloat16> patches_bf16 =
      F32ToRocblasBF16(patches);
  HWY_ASSERT(impl_->resident->patch_bias.count == vit_dim);
  HWY_ASSERT(impl_->resident->pos_embedding.count == rows * vit_dim);

  DeviceBuffer<rocblas_bfloat16> d_patches_bf16(patches_bf16.size());
  DeviceBuffer<float> d_x_bf16_a(static_cast<size_t>(rows) * vit_dim);
  DeviceBuffer<float> d_x_bf16_b(static_cast<size_t>(rows) * vit_dim);
  HIP_CHECK(hipMemcpy(d_patches_bf16.ptr, patches_bf16.data(),
                      patches_bf16.size() * sizeof(rocblas_bfloat16),
                      hipMemcpyHostToDevice));

  auto add_patch_bias_and_pos = [&](float* out) {
    hipLaunchKernelGGL(AddBiasAndPosKernel, dim3(vit_blocks), dim3(threads), 0,
                       0, out, impl_->resident->patch_bias.device->ptr,
                       impl_->resident->pos_embedding.device->ptr, rows,
                       vit_dim);
    HIP_CHECK(hipGetLastError());
  };
  auto run_patch_embedding_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, vit_dim,
        rows, patch, &alpha, impl_->resident->patch_embed.device->ptr,
        rocblas_datatype_bf16_r, vit_dim, d_patches_bf16.ptr,
        rocblas_datatype_bf16_r, patch, &beta, d_x_bf16_a.ptr,
        rocblas_datatype_f32_r, vit_dim, d_x_bf16_a.ptr,
        rocblas_datatype_f32_r, vit_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_patch_bias_and_pos(d_x_bf16_a.ptr);
  };
  run_patch_embedding_bf16();

  int max_qkv = 0;
  int max_heads = 0;
  int max_qkv_dim = 0;
  int max_att_cols = 0;
  int max_mlp = 0;
  for (size_t i = 0; i < num_layers; ++i) {
    const LayerConfig& layer_config = weights.VitLayer(i)->layer_config;
    const int heads = static_cast<int>(layer_config.heads);
    const int qkv_dim = static_cast<int>(layer_config.qkv_dim);
    max_heads = std::max(max_heads, heads);
    max_qkv_dim = std::max(max_qkv_dim, qkv_dim);
    max_qkv = std::max(max_qkv, heads * 3 * qkv_dim);
    max_att_cols = std::max(max_att_cols, heads * qkv_dim);
    max_mlp = std::max(max_mlp,
                       static_cast<int>(layer_config.ff_hidden_dim));
  }
  if (impl_->return_workspace == nullptr ||
      !impl_->return_workspace->Matches(rows, vit_dim, max_qkv, max_heads,
                                        max_qkv_dim, max_att_cols, max_mlp)) {
    impl_->return_workspace = std::make_unique<VitLayerBF16Workspace>(
        rows, vit_dim, max_qkv, max_heads, max_qkv_dim, max_att_cols, max_mlp);
    if (impl_->options.verbose) {
      std::printf("image_tokens_return_workspace rows=%d vit_dim=%d "
                  "scratch=%.2f MiB lifetime=backend\n",
                  rows, vit_dim, MiB(impl_->return_workspace->TotalBytes()));
    }
  }
  VitLayerBF16Workspace& layer_workspace = *impl_->return_workspace;

  for (size_t i = 0; i < num_layers; ++i) {
    const bool even = (i % 2) == 0;
    const float* d_x_in = even ? d_x_bf16_a.ptr : d_x_bf16_b.ptr;
    float* d_x_out = even ? d_x_bf16_b.ptr : d_x_bf16_a.ptr;
    if (!RunDeviceVitLayerBF16Only(impl_->handle, impl_->resident->layers[i],
                                   *weights.VitLayer(i), rows, vit_dim,
                                   d_x_in, d_x_out, layer_workspace,
                                   /*timing_options=*/nullptr,
                                   /*timings=*/nullptr,
                                   impl_->mlp_up_solution_index,
                                   impl_->mlp_down_solution_index,
                                   impl_->options.attention_qk_solution_index,
                                   impl_->options.attention_av_solution_index,
                                   impl_->options.use_attention_qk_bf16,
                                   impl_->options.use_attention_qk_bf16_wmma,
                                   impl_->options.use_attention_av_bf16_wmma,
                                   impl_->options.use_attention_av_f32_dim4,
                                   impl_->options.use_attention_pack_bf16,
                                   impl_->options.use_attention_direct_qkv,
                                   impl_->options
                                       .use_attention_defer_softmax_scale,
                                   impl_->options.use_qkv_wmma2d,
                                   impl_->options.use_attn_out_wmma2d,
                                   impl_->options.use_mlp_up_wmma2d,
                                   impl_->options.use_mlp_down_wmma8,
                                   impl_->options
                                       .use_mlp_down_fused_residual,
                                   impl_->options.mlp_down_wmma_waves)) {
      return false;
    }
  }

  const bool even_layers = (num_layers % 2) == 0;
  const float* final_stack =
      even_layers ? d_x_bf16_a.ptr : d_x_bf16_b.ptr;

  HWY_ASSERT(impl_->resident->enc_norm_scale.count == vit_dim);
  HWY_ASSERT(impl_->resident->enc_norm_bias.count == vit_dim);
  HWY_ASSERT(impl_->resident->head_bias.count == llm_dim);
  DeviceBuffer<rocblas_bfloat16> d_norm_bf16(static_cast<size_t>(rows) *
                                             vit_dim);
  DeviceBuffer<float> d_tokens_bf16(static_cast<size_t>(rows) * llm_dim);

  const size_t layer_norm_shared =
      static_cast<size_t>(threads) * 2 * sizeof(float);
  auto run_final_norm_bf16 = [&]() {
    // Final encoder norm feeds only the resident BF16 image-head GEMM in the
    // return path, so it can use the same direct BF16 LayerNorm producer as the
    // transformer-layer LN0/LN1 stages.
    hipLaunchKernelGGL(LayerNormToBF16Kernel, dim3(rows), dim3(threads),
                       layer_norm_shared, 0, final_stack,
                       impl_->resident->enc_norm_scale.device->ptr,
                       impl_->resident->enc_norm_bias.device->ptr,
                       d_norm_bf16.ptr, rows, vit_dim);
    HIP_CHECK(hipGetLastError());
  };
  run_final_norm_bf16();

  auto add_head_bias = [&](float* out) {
    hipLaunchKernelGGL(AddBiasKernel, dim3(token_blocks), dim3(threads), 0, 0,
                       out, impl_->resident->head_bias.device->ptr, rows,
                       llm_dim);
    HIP_CHECK(hipGetLastError());
  };
  auto run_head_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, llm_dim,
        rows, vit_dim, &alpha, impl_->resident->head.device->ptr,
        rocblas_datatype_bf16_r, llm_dim, d_norm_bf16.ptr,
        rocblas_datatype_bf16_r, vit_dim, &beta, d_tokens_bf16.ptr,
        rocblas_datatype_f32_r, llm_dim, d_tokens_bf16.ptr,
        rocblas_datatype_f32_r, llm_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_head_bias(d_tokens_bf16.ptr);
  };
  run_head_bf16();

  std::vector<float> tokens_bf16(static_cast<size_t>(rows) * llm_dim);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(tokens_bf16.data(), d_tokens_bf16.ptr,
                      tokens_bf16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  CopyVectorToImageTokens(tokens_bf16, image_tokens);

  impl_->stats.image_tokens_device_norm_returned = true;
  impl_->stats.image_tokens_return_stack_bf16_ms = 0.0f;
  impl_->stats.image_tokens_return_final_norm_bf16_ms = 0.0f;
  impl_->stats.image_tokens_return_head_bf16_ms = 0.0f;

  if (impl_->options.verbose) {
    std::printf("image_tokens_return_bf16 layers=%zu timing=disabled\n",
                num_layers);
  }
  return true;
}

bool PaliGemma2VitHipBackend::ValidateAttentionQkBf16ImageTokens(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image, ImageTokens& f32_attention_tokens,
    ImageTokens& qk_bf16_tokens) {
  // Final-boundary A/B for the attention precision experiment. Both runs use
  // the same BF16-return image-token path and the same projection/MLP options;
  // only the attention core changes from F32 QK to BF16 QK. Comparing the
  // decoder-visible image tokens catches accumulated drift across all 27 ViT
  // layers without involving decoder sampling.
  const bool original_attention_qk_bf16 =
      impl_->options.use_attention_qk_bf16;
  const bool original_attention_qk_bf16_wmma =
      impl_->options.use_attention_qk_bf16_wmma;
  const bool validate_wmma = original_attention_qk_bf16_wmma;

  impl_->options.use_attention_qk_bf16 = false;
  impl_->options.use_attention_qk_bf16_wmma = false;
  const bool f32_ok = GenerateImageTokensDeviceNorm(
      model_config, weights, image, f32_attention_tokens);
  if (!f32_ok) {
    impl_->options.use_attention_qk_bf16 = original_attention_qk_bf16;
    impl_->options.use_attention_qk_bf16_wmma =
        original_attention_qk_bf16_wmma;
    return false;
  }

  impl_->options.use_attention_qk_bf16 = !validate_wmma;
  impl_->options.use_attention_qk_bf16_wmma = validate_wmma;
  const bool qk_bf16_ok = GenerateImageTokensDeviceNorm(
      model_config, weights, image, qk_bf16_tokens);
  impl_->options.use_attention_qk_bf16 = original_attention_qk_bf16;
  impl_->options.use_attention_qk_bf16_wmma =
      original_attention_qk_bf16_wmma;
  if (!qk_bf16_ok) return false;

  const std::vector<float> f32_tokens =
      CopyImageTokensToVector(f32_attention_tokens);
  const std::vector<float> qk_bf16 =
      CopyImageTokensToVector(qk_bf16_tokens);
  const ErrorStats error = ComputeError(qk_bf16, f32_tokens);
  impl_->stats.image_tokens_attention_qk_bf16_validated = true;
  impl_->stats.image_tokens_attention_qk_bf16_max_abs_error =
      error.max_abs;
  impl_->stats.image_tokens_attention_qk_bf16_rms_error = error.rms;

  std::printf("attention_qk_bf16_image_tokens rows=%zu cols=%zu "
              "max_abs_vs_f32_attn=%9.6f rms_vs_f32_attn=%9.6f "
              "qk_impl=%s qkv=%s attn_out=%s mlp_up=%s mlp_down=%s\n",
              qk_bf16_tokens.Rows(), qk_bf16_tokens.Cols(), error.max_abs,
              error.rms,
              validate_wmma
                  ? AttentionQkBf16WmmaTileName(
                        static_cast<int>(model_config.vit_config.seq_len))
                  : "rocblas",
              impl_->options.use_qkv_wmma2d ? "wmma2d8x4" : "rocblas",
              impl_->options.use_attn_out_wmma2d ? "wmma2d8x4" : "rocblas",
              impl_->options.use_mlp_up_wmma2d
                  ? MlpUpWmma2DTileName(
                        static_cast<int>(model_config.vit_config.seq_len))
                  : "rocblas",
              impl_->options.use_mlp_down_wmma8
                  ? MlpDownWmmaTileName(
                        impl_->options.mlp_down_wmma_waves,
                        impl_->options.use_mlp_down_fused_residual)
                  : "rocblas");
  return true;
}

bool PaliGemma2VitHipBackend::ProfileImageTokensReturnDeviceNorm(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const Image& image, ImageTokens& image_tokens) {
  if (!UploadProjectionWeights(weights)) return false;
  if (model_config.vit_config.pool_dim != 1 ||
      model_config.wrapping == PromptWrapping::GEMMA_VLM) {
    std::fprintf(stderr,
                 "HIP image-token return profiling currently supports only "
                 "the unpooled PaliGemma2 ViT path.\n");
    return false;
  }

  const size_t num_layers = weights.vit_layers.size();
  if (num_layers == 0 || impl_->resident->layers.size() != num_layers) {
    std::fprintf(stderr,
                 "HIP image-token return profiling requires matching nonzero "
                 "ViT layer counts: weights=%zu resident=%zu.\n",
                 weights.vit_layers.size(), impl_->resident->layers.size());
    return false;
  }

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int vit_dim = static_cast<int>(model_config.vit_config.model_dim);
  const int llm_dim = static_cast<int>(model_config.model_dim);
  const int patch =
      static_cast<int>(model_config.vit_config.patch_width *
                       model_config.vit_config.patch_width * 3);
  HWY_ASSERT(weights.vit_img_head_kernel.Rows() ==
             static_cast<size_t>(llm_dim));
  HWY_ASSERT(weights.vit_img_head_kernel.Cols() ==
             static_cast<size_t>(vit_dim));
  HWY_ASSERT(image_tokens.Rows() == static_cast<size_t>(rows));
  HWY_ASSERT(image_tokens.Cols() == static_cast<size_t>(llm_dim));
  HWY_ASSERT(impl_->resident->patch_bias.count == vit_dim);
  HWY_ASSERT(impl_->resident->pos_embedding.count == rows * vit_dim);
  HWY_ASSERT(impl_->resident->enc_norm_scale.count == vit_dim);
  HWY_ASSERT(impl_->resident->enc_norm_bias.count == vit_dim);
  HWY_ASSERT(impl_->resident->head_bias.count == llm_dim);

  const int threads = 256;
  const int vit_count = rows * vit_dim;
  const int vit_blocks = (vit_count + threads - 1) / threads;
  const int token_count = rows * llm_dim;
  const int token_blocks = (token_count + threads - 1) / threads;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  std::vector<float> patches;
  std::vector<rocblas_bfloat16> patches_bf16;
  const float patch_prepare_ms =
      TimeHostSamples(impl_->options, [&]() {
        patches = BuildImagePatches(image, rows, patch);
        patches_bf16 = F32ToRocblasBF16(patches);
      });

  const double alloc_start = hwy::platform::Now();
  DeviceBuffer<rocblas_bfloat16> d_patches_bf16(patches_bf16.size());
  DeviceBuffer<float> d_x_bf16_a(static_cast<size_t>(rows) * vit_dim);
  DeviceBuffer<float> d_x_bf16_b(static_cast<size_t>(rows) * vit_dim);
  DeviceBuffer<rocblas_bfloat16> d_norm_bf16(static_cast<size_t>(rows) *
                                             vit_dim);
  DeviceBuffer<float> d_tokens_bf16(static_cast<size_t>(rows) * llm_dim);
  HIP_CHECK(hipDeviceSynchronize());
  const float alloc_once_ms =
      static_cast<float>((hwy::platform::Now() - alloc_start) * 1000.0);

  const float patch_h2d_ms = TimeHostSamples(impl_->options, [&]() {
    HIP_CHECK(hipMemcpy(d_patches_bf16.ptr, patches_bf16.data(),
                        patches_bf16.size() * sizeof(rocblas_bfloat16),
                        hipMemcpyHostToDevice));
  });

  auto add_patch_bias_and_pos = [&](float* out) {
    hipLaunchKernelGGL(AddBiasAndPosKernel, dim3(vit_blocks), dim3(threads), 0,
                       0, out, impl_->resident->patch_bias.device->ptr,
                       impl_->resident->pos_embedding.device->ptr, rows,
                       vit_dim);
    HIP_CHECK(hipGetLastError());
  };
  auto run_patch_embedding_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, vit_dim,
        rows, patch, &alpha, impl_->resident->patch_embed.device->ptr,
        rocblas_datatype_bf16_r, vit_dim, d_patches_bf16.ptr,
        rocblas_datatype_bf16_r, patch, &beta, d_x_bf16_a.ptr,
        rocblas_datatype_f32_r, vit_dim, d_x_bf16_a.ptr,
        rocblas_datatype_f32_r, vit_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_patch_bias_and_pos(d_x_bf16_a.ptr);
  };
  const float patch_embed_ms =
      TimeSamples(impl_->options, run_patch_embedding_bf16);

  int max_qkv = 0;
  int max_heads = 0;
  int max_qkv_dim = 0;
  int max_att_cols = 0;
  int max_mlp = 0;
  for (size_t i = 0; i < num_layers; ++i) {
    const LayerConfig& layer_config = weights.VitLayer(i)->layer_config;
    const int heads = static_cast<int>(layer_config.heads);
    const int qkv_dim = static_cast<int>(layer_config.qkv_dim);
    max_heads = std::max(max_heads, heads);
    max_qkv_dim = std::max(max_qkv_dim, qkv_dim);
    max_qkv = std::max(max_qkv, heads * 3 * qkv_dim);
    max_att_cols = std::max(max_att_cols, heads * qkv_dim);
    max_mlp = std::max(max_mlp,
                       static_cast<int>(layer_config.ff_hidden_dim));
  }
  if (impl_->return_workspace == nullptr ||
      !impl_->return_workspace->Matches(rows, vit_dim, max_qkv, max_heads,
                                        max_qkv_dim, max_att_cols, max_mlp)) {
    impl_->return_workspace = std::make_unique<VitLayerBF16Workspace>(
        rows, vit_dim, max_qkv, max_heads, max_qkv_dim, max_att_cols, max_mlp);
    if (impl_->options.verbose) {
      std::printf("image_tokens_return_workspace rows=%d vit_dim=%d "
                  "scratch=%.2f MiB lifetime=backend\n",
                  rows, vit_dim, MiB(impl_->return_workspace->TotalBytes()));
    }
  }
  VitLayerBF16Workspace& layer_workspace = *impl_->return_workspace;

  DeviceVitLayerTimings stack;
  for (size_t i = 0; i < num_layers; ++i) {
    const bool even = (i % 2) == 0;
    const float* d_x_in = even ? d_x_bf16_a.ptr : d_x_bf16_b.ptr;
    float* d_x_out = even ? d_x_bf16_b.ptr : d_x_bf16_a.ptr;
    DeviceVitLayerTimings timing;
    if (!RunDeviceVitLayerBF16Only(
            impl_->handle, impl_->resident->layers[i], *weights.VitLayer(i),
            rows, vit_dim, d_x_in, d_x_out, layer_workspace, &impl_->options,
            &timing, impl_->mlp_up_solution_index,
            impl_->mlp_down_solution_index,
            impl_->options.attention_qk_solution_index,
            impl_->options.attention_av_solution_index,
            impl_->options.use_attention_qk_bf16,
            impl_->options.use_attention_qk_bf16_wmma,
            impl_->options.use_attention_av_bf16_wmma,
            impl_->options.use_attention_av_f32_dim4,
            impl_->options.use_attention_pack_bf16,
            impl_->options.use_attention_direct_qkv,
            impl_->options.use_attention_defer_softmax_scale,
            impl_->options.use_qkv_wmma2d,
            impl_->options.use_attn_out_wmma2d,
            impl_->options.use_mlp_up_wmma2d,
            impl_->options.use_mlp_down_wmma8,
            impl_->options.use_mlp_down_fused_residual,
            impl_->options.mlp_down_wmma_waves)) {
      return false;
    }
    stack.ln0_bf16_ms += timing.ln0_bf16_ms;
    stack.qkv_bf16_ms += timing.qkv_bf16_ms;
    stack.attn_bf16_ms += timing.attn_bf16_ms;
    stack.attn_out_bf16_ms += timing.attn_out_bf16_ms;
    stack.ln1_bf16_ms += timing.ln1_bf16_ms;
    stack.mlp_bf16_ms += timing.mlp_bf16_ms;
    stack.mlp_up_bf16_ms += timing.mlp_up_bf16_ms;
    stack.mlp_act_bf16_ms += timing.mlp_act_bf16_ms;
    stack.mlp_down_bf16_ms += timing.mlp_down_bf16_ms;
    stack.mlp_residual_bf16_ms += timing.mlp_residual_bf16_ms;
  }

  const bool even_layers = (num_layers % 2) == 0;
  const float* final_stack =
      even_layers ? d_x_bf16_a.ptr : d_x_bf16_b.ptr;

  const size_t layer_norm_shared =
      static_cast<size_t>(threads) * 2 * sizeof(float);
  auto run_final_norm_bf16 = [&]() {
    hipLaunchKernelGGL(LayerNormToBF16Kernel, dim3(rows), dim3(threads),
                       layer_norm_shared, 0, final_stack,
                       impl_->resident->enc_norm_scale.device->ptr,
                       impl_->resident->enc_norm_bias.device->ptr,
                       d_norm_bf16.ptr, rows, vit_dim);
    HIP_CHECK(hipGetLastError());
  };
  const float final_norm_ms =
      TimeSamples(impl_->options, run_final_norm_bf16);

  auto add_head_bias = [&](float* out) {
    hipLaunchKernelGGL(AddBiasKernel, dim3(token_blocks), dim3(threads), 0, 0,
                       out, impl_->resident->head_bias.device->ptr, rows,
                       llm_dim);
    HIP_CHECK(hipGetLastError());
  };
  auto run_head_bf16 = [&]() {
    ROCBLAS_CHECK(rocblas_gemm_ex(
        impl_->handle, rocblas_operation_none, rocblas_operation_none, llm_dim,
        rows, vit_dim, &alpha, impl_->resident->head.device->ptr,
        rocblas_datatype_bf16_r, llm_dim, d_norm_bf16.ptr,
        rocblas_datatype_bf16_r, vit_dim, &beta, d_tokens_bf16.ptr,
        rocblas_datatype_f32_r, llm_dim, d_tokens_bf16.ptr,
        rocblas_datatype_f32_r, llm_dim, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, /*solution_index=*/0, /*flags=*/0));
    add_head_bias(d_tokens_bf16.ptr);
  };
  const float head_ms = TimeSamples(impl_->options, run_head_bf16);

  std::vector<float> tokens_bf16(static_cast<size_t>(rows) * llm_dim);
  HIP_CHECK(hipDeviceSynchronize());
  const float d2h_ms = TimeHostSamples(impl_->options, [&]() {
    HIP_CHECK(hipMemcpy(tokens_bf16.data(), d_tokens_bf16.ptr,
                        tokens_bf16.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
  });
  const double copy_start = hwy::platform::Now();
  CopyVectorToImageTokens(tokens_bf16, image_tokens);
  const float host_copy_ms =
      static_cast<float>((hwy::platform::Now() - copy_start) * 1000.0);

  const float stack_ms = stack.ln0_bf16_ms + stack.qkv_bf16_ms +
                         stack.attn_bf16_ms + stack.attn_out_bf16_ms +
                         stack.ln1_bf16_ms + stack.mlp_bf16_ms;
  const float device_sum_ms =
      patch_embed_ms + stack_ms + final_norm_ms + head_ms;
  const float host_sum_ms = patch_prepare_ms + alloc_once_ms + patch_h2d_ms +
                            d2h_ms + host_copy_ms;
  const char* attention_mode =
      impl_->options.use_attention_av_bf16_wmma
          ? "av_bf16_wmma2d8x2"
          : (impl_->options.use_attention_av_f32_dim4 &&
                     CanUseAttentionAvF32Dim4(rows, 72)
                 ? (impl_->options.use_attention_pack_bf16
                        ? "av_f32_dim4_pack_bf16"
                        : "av_f32_dim4")
                 : (impl_->options.use_attention_qk_bf16_wmma
                        ? "qk_bf16_wmma2d2x8"
                        : (impl_->options.use_attention_qk_bf16
                               ? "qk_bf16"
                               : (impl_->options.use_attention_direct_qkv
                                      ? (impl_->options.use_attention_pack_bf16
                                             ? "f32_direct_qkv_pack_bf16"
                                             : "f32_direct_qkv")
                                      : (impl_->options.use_attention_pack_bf16
                                             ? "f32_pack_bf16"
                                             : "f32")))));

  std::printf("image_tokens_return_profile rows=%d layers=%zu samples=%d "
              "warmup=%d iters=%d attn=%s qkv=%s attn_out=%s "
              "mlp_up=%s mlp_down=%s qk_solution=%d av_solution=%d\n",
              rows, num_layers, impl_->options.samples, impl_->options.warmup,
              impl_->options.iters, attention_mode,
              impl_->options.use_qkv_wmma2d ? "wmma2d8x4" : "rocblas",
              impl_->options.use_attn_out_wmma2d ? "wmma2d8x4" : "rocblas",
              impl_->options.use_mlp_up_wmma2d ? MlpUpWmma2DTileName(rows)
                                                : "rocblas",
              impl_->options.use_mlp_down_wmma8
                  ? MlpDownWmmaTileName(
                        impl_->options.mlp_down_wmma_waves,
                        impl_->options.use_mlp_down_fused_residual)
                  : "rocblas",
              impl_->options.attention_qk_solution_index,
              impl_->options.attention_av_solution_index);
  std::printf("  host patch_prepare=%8.3f ms alloc_once=%8.3f ms "
              "patch_h2d=%8.3f ms d2h=%8.3f ms host_copy=%8.3f ms "
              "host_sum=%8.3f ms\n",
              patch_prepare_ms, alloc_once_ms, patch_h2d_ms, d2h_ms,
              host_copy_ms, host_sum_ms);
  std::printf("  device patch_embed=%8.3f ms ln0=%8.3f ms qkv=%8.3f ms "
              "attn=%8.3f ms attn_out=%8.3f ms ln1=%8.3f ms mlp=%8.3f ms "
              "final_norm=%8.3f ms head=%8.3f ms stack=%8.3f ms "
              "device_sum=%8.3f ms\n",
              patch_embed_ms, stack.ln0_bf16_ms, stack.qkv_bf16_ms,
              stack.attn_bf16_ms, stack.attn_out_bf16_ms,
              stack.ln1_bf16_ms, stack.mlp_bf16_ms, final_norm_ms, head_ms,
              stack_ms, device_sum_ms);
  std::printf("  mlp    up_gemm=%8.3f ms act=%8.3f ms down_gemm=%8.3f ms "
              "residual=%8.3f ms total=%8.3f ms\n",
              stack.mlp_up_bf16_ms, stack.mlp_act_bf16_ms,
              stack.mlp_down_bf16_ms, stack.mlp_residual_bf16_ms,
              stack.mlp_bf16_ms);
  return true;
}

bool PaliGemma2VitHipBackend::LoadMlpGemmSolutionCache(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const std::string& path) {
  if (path.empty()) return true;
  const MlpSolutionCacheKey key =
      MakeMlpSolutionCacheKey(model_config, weights, impl_->stats);
  std::ifstream in(path);
  if (!in) {
    if (impl_->options.verbose) {
      std::printf("mlp_solution_cache_load path=%s status=missing\n",
                  path.c_str());
    }
    return true;
  }

  bool found = false;
  int up = 0;
  int down = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (!CacheLineMatchesKey(line, key)) continue;
    int line_up = 0;
    int line_down = 0;
    if (!ExtractCacheInt(line, "up", line_up) ||
        !ExtractCacheInt(line, "down", line_down)) {
      continue;
    }
    up = line_up;
    down = line_down;
    found = true;
  }

  if (!found) {
    if (impl_->options.verbose) {
      std::printf("mlp_solution_cache_load path=%s status=miss rows=%d "
                  "model_dim=%d mlp=%d arch=%s hip_runtime=%d "
                  "rocblas=%d.%d.%d\n",
                  path.c_str(), key.rows, key.model_dim, key.mlp,
                  key.device_arch.c_str(), key.hip_runtime, key.rocblas_major,
                  key.rocblas_minor, key.rocblas_patch);
    }
    return true;
  }

  impl_->mlp_up_solution_index = up;
  impl_->mlp_down_solution_index = down;
  if (impl_->options.verbose) {
    std::printf("mlp_solution_cache_load path=%s status=hit up=%d down=%d "
                "rows=%d model_dim=%d mlp=%d arch=%s hip_runtime=%d "
                "rocblas=%d.%d.%d\n",
                path.c_str(), up, down, key.rows, key.model_dim, key.mlp,
                key.device_arch.c_str(), key.hip_runtime, key.rocblas_major,
                key.rocblas_minor, key.rocblas_patch);
  }
  return true;
}

bool PaliGemma2VitHipBackend::BenchmarkMlpGemmSolutions(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    int max_solutions) {
  if (max_solutions <= 0) return true;
  if (!UploadProjectionWeights(weights)) return false;
  if (weights.vit_layers.empty() || impl_->resident->layers.empty()) {
    std::fprintf(stderr, "MLP solution bench requires at least one ViT layer.\n");
    return false;
  }

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int model_dim = static_cast<int>(model_config.vit_config.model_dim);
  const LayerWeightsPtrs& layer0 = *weights.VitLayer(0);
  const int mlp = static_cast<int>(layer0.layer_config.ff_hidden_dim);
  const ResidentProjectionLayer& resident_layer = impl_->resident->layers[0];
  HWY_ASSERT(resident_layer.linear_0.k == model_dim);
  HWY_ASSERT(resident_layer.linear_0.n == mlp);
  HWY_ASSERT(resident_layer.linear_1.k == mlp);
  HWY_ASSERT(resident_layer.linear_1.n == model_dim);

  DeviceBuffer<rocblas_bfloat16> d_pre_ffw(
      static_cast<size_t>(rows) * model_dim);
  DeviceBuffer<float> d_up_accum(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<rocblas_bfloat16> d_up(static_cast<size_t>(rows) * mlp);
  DeviceBuffer<float> d_out(static_cast<size_t>(rows) * model_dim);

  std::printf("mlp_solution_bench_config rows=%d model_dim=%d mlp=%d "
              "max_solutions=%d samples=%d warmup=%d iters=%d\n",
              rows, model_dim, mlp, max_solutions, impl_->options.samples,
              impl_->options.warmup, impl_->options.iters);
  impl_->mlp_up_solution_index = BenchmarkOneGemmSolutions(
      "mlp_up", impl_->handle, impl_->options, max_solutions, mlp, rows,
      model_dim, resident_layer.linear_0.device->ptr, mlp, d_pre_ffw.ptr,
      model_dim, d_up_accum.ptr, mlp);
  impl_->mlp_down_solution_index = BenchmarkOneGemmSolutions(
      "mlp_down", impl_->handle, impl_->options, max_solutions, model_dim,
      rows, mlp, resident_layer.linear_1.device->ptr, model_dim, d_up.ptr, mlp,
      d_out.ptr, model_dim);
  std::printf("mlp_solution_selected up=%d down=%d\n",
              impl_->mlp_up_solution_index, impl_->mlp_down_solution_index);
  return true;
}

bool PaliGemma2VitHipBackend::SaveMlpGemmSolutionCache(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const std::string& path) const {
  if (path.empty()) return true;
  const MlpSolutionCacheKey key =
      MakeMlpSolutionCacheKey(model_config, weights, impl_->stats);
  std::ofstream out(path, std::ios::app);
  if (!out) {
    std::fprintf(stderr, "Failed to open MLP solution cache for append: %s\n",
                 path.c_str());
    return false;
  }
  out << "# gemma.cpp PaliGemma2 ViT HIP MLP solution cache v1\n";
  out << "entry"
      << " hip_runtime=" << key.hip_runtime
      << " rocblas_major=" << key.rocblas_major
      << " rocblas_minor=" << key.rocblas_minor
      << " rocblas_patch=" << key.rocblas_patch
      << " device_arch=" << key.device_arch
      << " compute_units=" << key.compute_units
      << " model=" << key.model
      << " rows=" << key.rows
      << " model_dim=" << key.model_dim
      << " mlp=" << key.mlp
      << " up=" << impl_->mlp_up_solution_index
      << " down=" << impl_->mlp_down_solution_index << "\n";
  if (!out) {
    std::fprintf(stderr, "Failed to write MLP solution cache: %s\n",
                 path.c_str());
    return false;
  }
  std::printf("mlp_solution_cache_save path=%s up=%d down=%d rows=%d "
              "model_dim=%d mlp=%d arch=%s hip_runtime=%d rocblas=%d.%d.%d\n",
              path.c_str(), impl_->mlp_up_solution_index,
              impl_->mlp_down_solution_index, key.rows, key.model_dim, key.mlp,
              key.device_arch.c_str(), key.hip_runtime, key.rocblas_major,
              key.rocblas_minor, key.rocblas_patch);
  return true;
}

bool PaliGemma2VitHipBackend::LoadAttentionGemmSolutionCache(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const std::string& path) {
  if (path.empty()) return true;
  const AttentionSolutionCacheKey key =
      MakeAttentionSolutionCacheKey(model_config, weights, impl_->stats);
  std::ifstream in(path);
  if (!in) {
    if (impl_->options.verbose) {
      std::printf("attention_solution_cache_load path=%s status=missing\n",
                  path.c_str());
    }
    return true;
  }

  bool found = false;
  int qk = 0;
  int av = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (!CacheLineMatchesKey(line, key)) continue;
    int line_qk = 0;
    int line_av = 0;
    if (!ExtractCacheInt(line, "qk", line_qk) ||
        !ExtractCacheInt(line, "av", line_av)) {
      continue;
    }
    qk = line_qk;
    av = line_av;
    found = true;
  }

  if (!found) {
    if (impl_->options.verbose) {
      std::printf("attention_solution_cache_load path=%s status=miss rows=%d "
                  "model_dim=%d heads=%d qkv_dim=%d arch=%s "
                  "hip_runtime=%d rocblas=%d.%d.%d\n",
                  path.c_str(), key.rows, key.model_dim, key.heads,
                  key.qkv_dim, key.device_arch.c_str(), key.hip_runtime,
                  key.rocblas_major, key.rocblas_minor, key.rocblas_patch);
    }
    return true;
  }

  impl_->options.attention_qk_solution_index = qk;
  impl_->options.attention_av_solution_index = av;
  if (impl_->options.verbose) {
    std::printf("attention_solution_cache_load path=%s status=hit qk=%d av=%d "
                "rows=%d model_dim=%d heads=%d qkv_dim=%d arch=%s "
                "hip_runtime=%d rocblas=%d.%d.%d\n",
                path.c_str(), qk, av, key.rows, key.model_dim, key.heads,
                key.qkv_dim, key.device_arch.c_str(), key.hip_runtime,
                key.rocblas_major, key.rocblas_minor, key.rocblas_patch);
  }
  return true;
}

bool PaliGemma2VitHipBackend::BenchmarkAttentionGemmSolutions(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    int max_solutions) {
  if (max_solutions <= 0) return true;
  if (weights.vit_layers.empty()) {
    std::fprintf(stderr,
                 "Attention solution bench requires at least one ViT layer.\n");
    return false;
  }

  const int rows = static_cast<int>(model_config.vit_config.seq_len);
  const int model_dim = static_cast<int>(model_config.vit_config.model_dim);
  const LayerConfig& layer_config = weights.VitLayer(0)->layer_config;
  const int heads = static_cast<int>(layer_config.heads);
  const int qkv_dim = static_cast<int>(layer_config.qkv_dim);
  const int head_count = rows * heads * qkv_dim;
  const int scores_count = heads * rows * rows;
  const rocblas_stride stride_qkv =
      static_cast<rocblas_stride>(rows) * qkv_dim;
  const rocblas_stride stride_scores =
      static_cast<rocblas_stride>(rows) * rows;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(qkv_dim));

  DeviceBuffer<float> d_q(head_count);
  DeviceBuffer<float> d_k(head_count);
  DeviceBuffer<float> d_v(head_count);
  DeviceBuffer<float> d_scores(scores_count);
  DeviceBuffer<float> d_att_head(head_count);

  AttentionGemmSpec qk;
  qk.label = "attention_qk";
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

  AttentionGemmSpec av;
  av.label = "attention_av";
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

  std::printf("attention_solution_bench_config rows=%d model_dim=%d heads=%d "
              "qkv_dim=%d max_solutions=%d samples=%d warmup=%d iters=%d\n",
              rows, model_dim, heads, qkv_dim, max_solutions,
              impl_->options.samples, impl_->options.warmup,
              impl_->options.iters);
  impl_->options.attention_qk_solution_index =
      BenchmarkOneAttentionGemmSolutions(impl_->handle, impl_->options,
                                         max_solutions, qk);
  impl_->options.attention_av_solution_index =
      BenchmarkOneAttentionGemmSolutions(impl_->handle, impl_->options,
                                         max_solutions, av);
  std::printf("attention_solution_selected qk=%d av=%d\n",
              impl_->options.attention_qk_solution_index,
              impl_->options.attention_av_solution_index);
  return true;
}

bool PaliGemma2VitHipBackend::SaveAttentionGemmSolutionCache(
    const ModelConfig& model_config, const WeightsPtrs& weights,
    const std::string& path) const {
  if (path.empty()) return true;
  const AttentionSolutionCacheKey key =
      MakeAttentionSolutionCacheKey(model_config, weights, impl_->stats);
  std::ofstream out(path, std::ios::app);
  if (!out) {
    std::fprintf(stderr,
                 "Failed to open attention solution cache for append: %s\n",
                 path.c_str());
    return false;
  }
  out << "# gemma.cpp PaliGemma2 ViT HIP attention solution cache v1\n";
  out << "entry"
      << " hip_runtime=" << key.hip_runtime
      << " rocblas_major=" << key.rocblas_major
      << " rocblas_minor=" << key.rocblas_minor
      << " rocblas_patch=" << key.rocblas_patch
      << " device_arch=" << key.device_arch
      << " compute_units=" << key.compute_units
      << " model=" << key.model
      << " rows=" << key.rows
      << " model_dim=" << key.model_dim
      << " heads=" << key.heads
      << " qkv_dim=" << key.qkv_dim
      << " qk=" << impl_->options.attention_qk_solution_index
      << " av=" << impl_->options.attention_av_solution_index << "\n";
  if (!out) {
    std::fprintf(stderr, "Failed to write attention solution cache: %s\n",
                 path.c_str());
    return false;
  }
  std::printf("attention_solution_cache_save path=%s qk=%d av=%d rows=%d "
              "model_dim=%d heads=%d qkv_dim=%d arch=%s hip_runtime=%d "
              "rocblas=%d.%d.%d\n",
              path.c_str(), impl_->options.attention_qk_solution_index,
              impl_->options.attention_av_solution_index, key.rows,
              key.model_dim, key.heads, key.qkv_dim, key.device_arch.c_str(),
              key.hip_runtime, key.rocblas_major, key.rocblas_minor,
              key.rocblas_patch);
  return true;
}

bool PaliGemma2VitHipBackend::TryGenerateImageTokens(
    const ModelConfig& model_config, const WeightsPtrs& weights, size_t seq_len,
    const Image& image, ImageTokens& image_tokens, MatMulEnv& env) {
  (void)seq_len;
  (void)image;
  (void)image_tokens;
  (void)env;

  // Real GenerateImageTokens integration point. For now this function performs
  // setup/validation/profiling and then returns false so the CPU ViT below the
  // hook remains the source of image_tokens. When the HIP backend owns the whole
  // image encoder, this is where it will fill `image_tokens` and return true.
  if (!UploadProjectionWeights(weights)) return false;
  if (impl_->options.validate_patch_embedding) {
    (void)ValidatePatchEmbedding(model_config, weights, image);
  }
  if (impl_->options.validate_layer0_qkv) {
    (void)ValidateLayer0QKV(model_config, weights, image);
  }
  if (impl_->options.validate_layer0_mlp) {
    (void)ValidateLayer0MLP(model_config, weights, image);
  }
  if (impl_->options.validate_layer0_attn_out) {
    (void)ValidateLayer0AttentionOut(model_config, weights, image);
  }
  if (impl_->options.validate_layer0_block) {
    (void)ValidateLayer0Block(model_config, weights, image);
  }
  if (impl_->options.validate_layer0_device_attention) {
    (void)ValidateLayer0DeviceAttention(model_config, weights, image);
  }
  if (impl_->options.validate_layer0_block_device_attention) {
    (void)ValidateLayer0BlockDeviceAttention(model_config, weights, image);
  }
  if (impl_->options.validate_layer0_block_device_norm) {
    (void)ValidateLayer0BlockDeviceNorm(model_config, weights, image);
  }
  if (impl_->options.validate_layer_prefix2_device_norm) {
    (void)ValidateLayerPrefix2DeviceNorm(model_config, weights, image);
  }
  if (impl_->options.validate_layer_stack_device_norm) {
    (void)ValidateLayerStackDeviceNorm(model_config, weights, image);
  }
  if (impl_->options.validate_image_tokens_device_norm &&
      !impl_->options.return_image_tokens_device_norm) {
    (void)ValidateImageTokensDeviceNorm(model_config, weights, image,
                                        &image_tokens,
                                        /*cpu_reference=*/nullptr);
  }
  if (impl_->options.return_image_tokens_device_norm) {
    return GenerateImageTokensDeviceNorm(model_config, weights, image,
                                         image_tokens);
  }
  if (impl_->options.benchmark_projection_schedule) {
    std::printf("\nrocBLAS projection schedule using resident real weights:\n");
    (void)RunResidentProjectionSchedule(model_config);
  }

  // This backend is wired into the real image-token control flow, but it is
  // still a projection-residency probe. Keep CPU ViT as the correctness path.
  return false;
}

const PaliGemma2VitHipStats& PaliGemma2VitHipBackend::stats() const {
  return impl_->stats;
}

bool PaliGemma2VitHipImageTokensBackend(void* backend,
                                        const ModelConfig& model_config,
                                        const WeightsPtrs& weights,
                                        size_t seq_len, const Image& image,
                                        ImageTokens& image_tokens,
                                        MatMulEnv& env) {
  return static_cast<PaliGemma2VitHipBackend*>(backend)->TryGenerateImageTokens(
      model_config, weights, seq_len, image, image_tokens, env);
}

}  // namespace experimental
}  // namespace gcpp
