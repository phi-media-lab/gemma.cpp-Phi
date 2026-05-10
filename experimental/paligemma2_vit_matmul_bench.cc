// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "ops/matmul.h"
#include "util/basics.h"
#include "util/threading_context.h"
#include "hwy/timer.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "experimental/paligemma2_vit_matmul_bench.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

#include "compression/compress-inl.h"
#include "compression/test_util-inl.h"
#include "ops/matmul-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

struct Shape {
  const char* name;
  size_t m;
  size_t k;
  size_t n;
};

// Times one gemma.cpp CPU MatMul shape with the same storage conventions used
// by the runtime. `B` is generated in transposed form because gemma.cpp's matmul
// kernels expect weight matrices in that layout. The benchmark therefore tests
// the real inference path instead of a generic row-major GEMM.
template <typename TA, typename TB = TA, typename TC = TA>
void BenchMatMul(const Shape& shape, size_t samples, bool print_best,
                 MatMulEnv& env) {
  fprintf(stderr, "\n%s: M=%zu K=%zu N=%zu TA=%s TB=%s TC=%s\n", shape.name,
          shape.m, shape.k, shape.n, TypeName<TA>(), TypeName<TB>(),
          TypeName<TC>());

  const Extents2D a_extents(shape.m, shape.k);
  const Extents2D b_extents(shape.n, shape.k);  // transposed B
  const Extents2D c_extents(shape.m, shape.n);

  MatStorageT<TA> A = GenerateMat<TA>(a_extents, MatPadding::kOdd, env.ctx);
  MatStorageT<TB> B =
      GenerateTransposedMat<TB>(b_extents, MatPadding::kOdd, env.ctx);
  MatStorageT<TC> C("C", c_extents, env.ctx.allocator, MatPadding::kOdd);

  BindB(env.ctx, B, sizeof(TC));
  BindC(env.ctx, C);
  C.AllocateAndAttachRowPtrs(env.row_ptrs);

  // Start worker spinning the same way inference may do for low-latency matmul
  // barriers. This keeps the benchmark close to the runtime execution model.
  Tristate use_spinning = Tristate::kDefault;
  env.ctx.pools.MaybeStartSpinning(use_spinning);

  std::vector<double> times;
  times.reserve(samples);
  double keep = 0.0;
  MMPerKey* per_key = nullptr;
  const bool old_print_best = env.print_best;
  env.print_best = print_best;

  // MatMul performs lazy autotuning. Only samples taken after a best config has
  // been selected are recorded, so the reported median is not polluted by search
  // iterations.
  while (times.size() < samples) {
    const double t0 = hwy::platform::Now();
    per_key = MatMul(A, B, nullptr, env, C);
    const double t1 = hwy::platform::Now();
    keep += static_cast<double>(C.Row(0)[hwy::Unpredictable1()]);
    if (per_key->autotune.Best()) {
      times.push_back(t1 - t0);
    }
  }

  hwy::PreventElision(keep);
  env.print_best = old_print_best;
  env.ctx.pools.MaybeStopSpinning(use_spinning);

  std::sort(times.begin(), times.end());
  const double median = times[times.size() / 2];
  const double best = times[0];
  const double flops = 2.0 * static_cast<double>(shape.m) *
                       static_cast<double>(shape.k) *
                       static_cast<double>(shape.n);
  fprintf(stderr, "%s result: best=%.3f ms median=%.3f ms %.1f GFLOPS\n",
          shape.name, best * 1E3, median * 1E3, flops / median * 1E-9);
}

void RunPaliGemma2VitMatMulBench(size_t samples, bool print_best) {
  // Use the default ThreadingContext so taskset and local topology discovery are
  // reflected exactly as they would be for the main `gemma` binary.
  ThreadingArgs threading_args;
  ThreadingContext ctx(threading_args);
  fprintf(stderr, "Target: %s\n", hwy::TargetName(HWY_TARGET));
  fprintf(stderr, "Topology: %s %s\n", ctx.topology.TopologyString(),
          ctx.pools.PinString());
  MatMulEnv env(ctx);

  // BF16 projection-heavy shapes from the SigLIP-style PaliGemma2 ViT encoder.
  // M is the number of image tokens: 256 for 224px, 1024 for 448px. The head
  // projection shapes differ between 3B and 10B decoder widths.
  const Shape bf16_shapes[] = {
      {"patch_embed_224", 256, 588, 1152},
      {"qkv_224", 256, 1152, 3456},
      {"attn_out_224", 256, 1152, 1152},
      {"mlp_up_224", 256, 1152, 4304},
      {"mlp_down_224", 256, 4304, 1152},
      {"head_proj_224_3b", 256, 1152, 2304},
      {"head_proj_224_10b", 256, 1152, 3584},
      {"patch_embed_448", 1024, 588, 1152},
      {"qkv_448", 1024, 1152, 3456},
      {"attn_out_448", 1024, 1152, 1152},
      {"mlp_up_448", 1024, 1152, 4304},
      {"mlp_down_448", 1024, 4304, 1152},
      {"head_proj_448_3b", 1024, 1152, 2304},
      {"head_proj_448_10b", 1024, 1152, 3584},
  };

  // Attention score and weighted-sum GEMM shapes for one head. The current CPU
  // ViT path does not run these exact GEMMs by default, but they provide a
  // useful baseline when comparing against rocBLAS QK/AV prototypes.
  const Shape f32_attention_shapes[] = {
      {"vit_qk_head_224_f32", 256, 72, 256},
      {"vit_av_head_224_f32", 256, 256, 72},
      {"vit_qk_head_448_f32", 1024, 72, 1024},
      {"vit_av_head_448_f32", 1024, 1024, 72},
  };

  for (const Shape& shape : bf16_shapes) {
    BenchMatMul<BF16, BF16, BF16>(shape, samples, print_best, env);
  }
  for (const Shape& shape : f32_attention_shapes) {
    BenchMatMul<float, float, float>(shape, samples, print_best, env);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace gcpp {

HWY_EXPORT(RunPaliGemma2VitMatMulBench);

namespace {

struct MainArgs {
  size_t samples = 8;
  bool print_best = true;
};

void PrintUsage(const char* argv0) {
  fprintf(stderr,
          "Usage: %s [--samples N] [--print_best 0|1]\n"
          "\n"
          "Benchmarks PaliGemma2 ViT matmul shapes with current gemma.cpp "
          "CPU kernels.\n",
          argv0);
}

bool ParseSize(const char* value, size_t& out) {
  char* end = nullptr;
  const unsigned long parsed = strtoul(value, &end, 10);
  if (end == value || *end != '\0') return false;
  out = static_cast<size_t>(parsed);
  return true;
}

bool ParseBool(const char* value, bool& out) {
  if (strcmp(value, "0") == 0) {
    out = false;
    return true;
  }
  if (strcmp(value, "1") == 0) {
    out = true;
    return true;
  }
  return false;
}

bool ParseArgs(int argc, char** argv, MainArgs& args) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    }
    if (arg == "--samples" && i + 1 < argc) {
      if (!ParseSize(argv[++i], args.samples) || args.samples == 0) return false;
      continue;
    }
    if (arg == "--print_best" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.print_best)) return false;
      continue;
    }
    fprintf(stderr, "Unknown or malformed argument: %s\n", argv[i]);
    PrintUsage(argv[0]);
    return false;
  }
  return true;
}

}  // namespace

int Main(int argc, char** argv) {
  MainArgs args;
  if (!ParseArgs(argc, argv, args)) return 1;

  // Dispatch through Highway so the benchmark runs under the same compiled CPU
  // target selection mechanism as the production matmul kernels.
  HWY_DYNAMIC_DISPATCH(RunPaliGemma2VitMatMulBench)(args.samples,
                                                    args.print_best);
  return 0;
}

}  // namespace gcpp

int main(int argc, char** argv) { return gcpp::Main(argc, argv); }

#endif  // HWY_ONCE
