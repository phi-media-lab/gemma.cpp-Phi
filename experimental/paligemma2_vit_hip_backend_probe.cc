// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

#include "experimental/paligemma2_vit_hip_backend.h"
#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "hwy/timer.h"
#include "util/args.h"

namespace {

struct ProbeArgs {
  // Extra HIP-only flags are parsed here instead of through gemma ArgsBase so
  // the probe can coexist with normal --weights/threading/inference flags. The
  // production CLI exposes only the subset that is useful from the real
  // GenerateImageTokens boundary.
  int samples = 5;
  int warmup = 3;
  int iters = 1;
  bool schedule = true;
  bool upload = false;
  bool resident_schedule = false;
  bool verbose = true;
  bool patch_validate = false;
  bool qkv_validate = false;
  bool mlp_validate = false;
  bool attn_out_validate = false;
  bool layer0_block_validate = false;
  bool device_attention_validate = false;
  bool layer0_block_device_attention_validate = false;
  bool layer0_block_device_norm_validate = false;
  bool layer_prefix2_device_norm_validate = false;
  bool layer_stack_device_norm_validate = false;
  bool image_tokens_device_norm_validate = false;
  int image_tokens_return_bench = 0;
  bool image_tokens_return_profile = false;
  int mlp_solution_bench = 0;
  std::string mlp_solution_cache;
  bool mlp_solution_cache_write = false;
  int attention_solution_bench = 0;
  std::string attention_solution_cache;
  bool attention_solution_cache_write = false;
  bool attention_qk_bf16 = false;
  bool attention_qk_bf16_wmma = false;
  bool attention_av_bf16_wmma = false;
  bool attention_av_f32_dim4 = false;
  bool attention_pack_bf16 = false;
  bool attention_direct_qkv = false;
  bool attention_defer_softmax_scale = false;
  bool attention_qk_bf16_check = false;
  int attention_qk_solution = 0;
  int attention_av_solution = 0;
  bool qkv_wmma2d = false;
  bool qkv_wmma2d_check = false;
  bool attn_out_wmma2d = false;
  bool attn_out_wmma2d_check = false;
  bool mlp_up_wmma2d = false;
  bool mlp_up_wmma2d_check = false;
  bool mlp_down_wmma8 = false;
  bool mlp_down_fused_residual = false;
  int mlp_down_wmma_waves = 8;
  bool mlp_down_wmma8_check = false;
};

bool ParseInt(const char* value, int& out) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0) return false;
  out = static_cast<int>(parsed);
  return true;
}

bool ParseNonNegativeInt(const char* value, int& out) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0) return false;
  out = static_cast<int>(parsed);
  return true;
}

bool ParseSignedInt(const char* value, int& out) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') return false;
  out = static_cast<int>(parsed);
  return true;
}

bool ParseBool(const char* value, bool& out) {
  if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0) {
    out = false;
    return true;
  }
  if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0) {
    out = true;
    return true;
  }
  return false;
}

bool ParseProbeArgs(int argc, char** argv, ProbeArgs& args) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--hip_samples" && i + 1 < argc) {
      if (!ParseInt(argv[++i], args.samples)) return false;
      continue;
    }
    if (arg == "--hip_warmup" && i + 1 < argc) {
      if (!ParseInt(argv[++i], args.warmup)) return false;
      continue;
    }
    if (arg == "--hip_iters" && i + 1 < argc) {
      if (!ParseInt(argv[++i], args.iters)) return false;
      continue;
    }
    if (arg == "--hip_schedule" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.schedule)) return false;
      continue;
    }
    if (arg == "--hip_upload" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.upload)) return false;
      continue;
    }
    if (arg == "--hip_resident_schedule" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.resident_schedule)) return false;
      continue;
    }
    if (arg == "--hip_verbose" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.verbose)) return false;
      continue;
    }
    if (arg == "--hip_patch_validate" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.patch_validate)) return false;
      continue;
    }
    if (arg == "--hip_qkv_validate" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.qkv_validate)) return false;
      continue;
    }
    if (arg == "--hip_mlp_validate" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.mlp_validate)) return false;
      continue;
    }
    if (arg == "--hip_attn_out_validate" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.attn_out_validate)) return false;
      continue;
    }
    if (arg == "--hip_layer0_block_validate" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.layer0_block_validate)) return false;
      continue;
    }
    if (arg == "--hip_device_attention_validate" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.device_attention_validate)) return false;
      continue;
    }
    if (arg == "--hip_layer0_block_device_attention_validate" &&
        i + 1 < argc) {
      if (!ParseBool(argv[++i],
                     args.layer0_block_device_attention_validate)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_layer0_block_device_norm_validate" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.layer0_block_device_norm_validate)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_layer_prefix2_device_norm_validate" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.layer_prefix2_device_norm_validate)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_layer_stack_device_norm_validate" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.layer_stack_device_norm_validate)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_image_tokens_device_norm_validate" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.image_tokens_device_norm_validate)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_image_tokens_return_bench" && i + 1 < argc) {
      if (!ParseNonNegativeInt(argv[++i],
                               args.image_tokens_return_bench)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_image_tokens_return_profile" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.image_tokens_return_profile)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_mlp_solution_bench" && i + 1 < argc) {
      if (!ParseNonNegativeInt(argv[++i], args.mlp_solution_bench)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_mlp_solution_cache" && i + 1 < argc) {
      args.mlp_solution_cache = argv[++i];
      continue;
    }
    if (arg == "--hip_mlp_solution_cache_write" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.mlp_solution_cache_write)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_attention_solution_bench" && i + 1 < argc) {
      if (!ParseNonNegativeInt(argv[++i], args.attention_solution_bench)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_attention_solution_cache" && i + 1 < argc) {
      args.attention_solution_cache = argv[++i];
      continue;
    }
    if (arg == "--hip_attention_solution_cache_write" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.attention_solution_cache_write)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_attention_qk_bf16" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.attention_qk_bf16)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_attention_qk_bf16_wmma" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.attention_qk_bf16_wmma)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_attention_av_bf16_wmma" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.attention_av_bf16_wmma)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_attention_av_f32_dim4" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.attention_av_f32_dim4)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_attention_pack_bf16" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.attention_pack_bf16)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_attention_direct_qkv" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.attention_direct_qkv)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_attention_defer_softmax_scale" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.attention_defer_softmax_scale)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_attention_qk_bf16_check" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.attention_qk_bf16_check)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_attention_qk_solution" && i + 1 < argc) {
      if (!ParseSignedInt(argv[++i], args.attention_qk_solution)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_attention_av_solution" && i + 1 < argc) {
      if (!ParseSignedInt(argv[++i], args.attention_av_solution)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_qkv_wmma2d" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.qkv_wmma2d)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_qkv_wmma2d_check" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.qkv_wmma2d_check)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_attn_out_wmma2d" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.attn_out_wmma2d)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_attn_out_wmma2d_check" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.attn_out_wmma2d_check)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_mlp_up_wmma2d" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.mlp_up_wmma2d)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_mlp_up_wmma2d_check" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.mlp_up_wmma2d_check)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_mlp_down_wmma8" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.mlp_down_wmma8)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_mlp_down_fused_residual" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.mlp_down_fused_residual)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_mlp_down_wmma_waves" && i + 1 < argc) {
      if (!ParseInt(argv[++i], args.mlp_down_wmma_waves)) {
        return false;
      }
      continue;
    }
    if (arg == "--hip_mlp_down_wmma8_check" && i + 1 < argc) {
      if (!ParseBool(argv[++i], args.mlp_down_wmma8_check)) {
        return false;
      }
      continue;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  gcpp::InternalInit();

  ProbeArgs probe_args;
  if (!ParseProbeArgs(argc, argv, probe_args)) {
    std::fprintf(stderr,
                 "Usage: %s --weights <model.sbs> [gemma args] "
                 "[--hip_samples N] [--hip_warmup N] [--hip_iters N] "
                 "[--hip_schedule 0|1] [--hip_upload 0|1] "
                 "[--hip_resident_schedule 0|1] [--hip_verbose 0|1] "
                 "[--hip_patch_validate 0|1] [--hip_qkv_validate 0|1] "
                 "[--hip_mlp_validate 0|1] "
                 "[--hip_attn_out_validate 0|1] "
                 "[--hip_layer0_block_validate 0|1] "
                 "[--hip_device_attention_validate 0|1] "
                 "[--hip_layer0_block_device_attention_validate 0|1] "
                 "[--hip_layer0_block_device_norm_validate 0|1] "
                 "[--hip_layer_prefix2_device_norm_validate 0|1] "
                 "[--hip_layer_stack_device_norm_validate 0|1] "
                 "[--hip_image_tokens_device_norm_validate 0|1] "
                 "[--hip_image_tokens_return_bench N] "
                 "[--hip_image_tokens_return_profile 0|1] "
                 "[--hip_mlp_solution_bench N] "
                 "[--hip_mlp_solution_cache PATH] "
                 "[--hip_mlp_solution_cache_write 0|1] "
                 "[--hip_attention_solution_bench N] "
                 "[--hip_attention_solution_cache PATH] "
                 "[--hip_attention_solution_cache_write 0|1] "
                 "[--hip_attention_qk_bf16 0|1] "
                 "[--hip_attention_qk_bf16_wmma 0|1] "
                 "[--hip_attention_av_bf16_wmma 0|1] "
                 "[--hip_attention_av_f32_dim4 0|1] "
                 "[--hip_attention_pack_bf16 0|1] "
                 "[--hip_attention_direct_qkv 0|1] "
                 "[--hip_attention_defer_softmax_scale 0|1] "
                 "[--hip_attention_qk_bf16_check 0|1] "
                 "[--hip_attention_qk_solution INDEX] "
                 "[--hip_attention_av_solution INDEX] "
                 "[--hip_qkv_wmma2d 0|1] "
                 "[--hip_qkv_wmma2d_check 0|1] "
                 "[--hip_attn_out_wmma2d 0|1] "
                 "[--hip_attn_out_wmma2d_check 0|1] "
                 "[--hip_mlp_up_wmma2d 0|1] "
                 "[--hip_mlp_up_wmma2d_check 0|1] "
                 "[--hip_mlp_down_wmma8 0|1] "
                 "[--hip_mlp_down_fused_residual 0|1] "
                 "[--hip_mlp_down_wmma_waves N] "
                 "[--hip_mlp_down_wmma8_check 0|1]\n",
                 argv[0]);
    return 1;
  }
  if (probe_args.mlp_solution_cache_write &&
      probe_args.mlp_solution_cache.empty()) {
    std::fprintf(stderr,
                 "--hip_mlp_solution_cache_write requires "
                 "--hip_mlp_solution_cache PATH\n");
    return 1;
  }
  if (probe_args.mlp_solution_cache_write &&
      probe_args.mlp_solution_bench <= 0) {
    std::fprintf(stderr,
                 "--hip_mlp_solution_cache_write requires "
                 "--hip_mlp_solution_bench N\n");
    return 1;
  }
  if (probe_args.attention_solution_cache_write &&
      probe_args.attention_solution_cache.empty()) {
    std::fprintf(stderr,
                 "--hip_attention_solution_cache_write requires "
                 "--hip_attention_solution_cache PATH\n");
    return 1;
  }
  if (probe_args.attention_solution_cache_write &&
      probe_args.attention_solution_bench <= 0) {
    std::fprintf(stderr,
                 "--hip_attention_solution_cache_write requires "
                 "--hip_attention_solution_bench N\n");
    return 1;
  }
  if (probe_args.resident_schedule) probe_args.upload = true;
  if (probe_args.patch_validate) probe_args.upload = true;
  if (probe_args.qkv_validate) probe_args.upload = true;
  if (probe_args.mlp_validate) probe_args.upload = true;
  if (probe_args.attn_out_validate) probe_args.upload = true;
  if (probe_args.layer0_block_validate) probe_args.upload = true;
  if (probe_args.device_attention_validate) probe_args.upload = true;
  if (probe_args.layer0_block_device_attention_validate) {
    probe_args.upload = true;
  }
  if (probe_args.layer0_block_device_norm_validate) probe_args.upload = true;
  if (probe_args.layer_prefix2_device_norm_validate) probe_args.upload = true;
  if (probe_args.layer_stack_device_norm_validate) probe_args.upload = true;
  if (probe_args.image_tokens_device_norm_validate) probe_args.upload = true;
  if (probe_args.image_tokens_return_bench > 0) probe_args.upload = true;
  if (probe_args.image_tokens_return_profile) probe_args.upload = true;
  if (probe_args.mlp_solution_bench > 0) probe_args.upload = true;
  if (probe_args.attention_qk_bf16_check) probe_args.upload = true;
  if (probe_args.qkv_wmma2d_check) probe_args.upload = true;
  if (probe_args.attn_out_wmma2d_check) probe_args.upload = true;
  if (probe_args.mlp_up_wmma2d_check) probe_args.upload = true;
  if (probe_args.mlp_down_wmma8_check) probe_args.upload = true;

  gcpp::LoaderArgs loader(argc, argv);
  gcpp::ThreadingArgs threading(argc, argv);
  gcpp::InferenceArgs inference(argc, argv);
  inference.verbosity = 0;

  gcpp::ThreadingContext ctx(threading);
  const gcpp::Gemma gemma(loader, inference, ctx);
  const gcpp::ModelConfig& config = gemma.Config();
  const gcpp::WeightsPtrs& weights = gemma.Weights();
  const auto load_probe_image = [&](gcpp::Image& image) {
    const std::string image_path =
        inference.image_file.path.empty()
            ? std::string("paligemma/testdata/image.ppm")
            : inference.image_file.path;
    if (!image.ReadPPM(image_path)) {
      std::fprintf(stderr, "Failed to read image_file '%s'\n",
                   image_path.c_str());
      return false;
    }
    const size_t image_size = config.vit_config.image_size;
    image.Resize(image_size, image_size);
    return true;
  };

  gcpp::experimental::PrintPaliGemma2VitHipModelSummary(config, weights);
  std::printf("  weight_mode=%s\n",
              gcpp::WeightsPtrs::ToString(gemma.WeightReadMode()));

  gcpp::experimental::PaliGemma2VitHipOptions options{
      .samples = probe_args.samples,
      .warmup = probe_args.warmup,
      .iters = probe_args.iters,
      .verbose = probe_args.verbose,
      .benchmark_projection_schedule = probe_args.resident_schedule,
      .validate_patch_embedding = probe_args.patch_validate,
      .validate_layer0_qkv = probe_args.qkv_validate,
      .validate_layer0_mlp = probe_args.mlp_validate,
      .validate_layer0_attn_out = probe_args.attn_out_validate,
      .validate_layer0_block = probe_args.layer0_block_validate,
      .validate_layer0_device_attention =
          probe_args.device_attention_validate,
      .validate_layer0_block_device_attention =
          probe_args.layer0_block_device_attention_validate,
      .validate_layer0_block_device_norm =
          probe_args.layer0_block_device_norm_validate,
      .validate_layer_prefix2_device_norm =
          probe_args.layer_prefix2_device_norm_validate,
      .validate_layer_stack_device_norm =
          probe_args.layer_stack_device_norm_validate,
      .validate_image_tokens_device_norm =
          probe_args.image_tokens_device_norm_validate,
      .use_attention_qk_bf16 = probe_args.attention_qk_bf16,
      .use_attention_qk_bf16_wmma = probe_args.attention_qk_bf16_wmma,
      .use_attention_av_bf16_wmma = probe_args.attention_av_bf16_wmma,
      .use_attention_av_f32_dim4 = probe_args.attention_av_f32_dim4,
      .use_attention_pack_bf16 = probe_args.attention_pack_bf16,
      .use_attention_direct_qkv = probe_args.attention_direct_qkv,
      .use_attention_defer_softmax_scale =
          probe_args.attention_defer_softmax_scale,
      .attention_qk_solution_index = probe_args.attention_qk_solution,
      .attention_av_solution_index = probe_args.attention_av_solution,
      .use_qkv_wmma2d = probe_args.qkv_wmma2d,
      .use_attn_out_wmma2d = probe_args.attn_out_wmma2d,
      .use_mlp_up_wmma2d = probe_args.mlp_up_wmma2d,
      .use_mlp_down_wmma8 = probe_args.mlp_down_wmma8,
      .use_mlp_down_fused_residual = probe_args.mlp_down_fused_residual,
      .mlp_down_wmma_waves = probe_args.mlp_down_wmma_waves,
  };
  gcpp::experimental::PaliGemma2VitHipBackend backend(options);
  if (!backend.Initialize()) return 1;
  if (!probe_args.mlp_solution_cache.empty()) {
    if (!backend.LoadMlpGemmSolutionCache(config, weights,
                                          probe_args.mlp_solution_cache)) {
      return 1;
    }
  }
  if (!probe_args.attention_solution_cache.empty()) {
    if (!backend.LoadAttentionGemmSolutionCache(
            config, weights, probe_args.attention_solution_cache)) {
      return 1;
    }
  }

  if (probe_args.upload) {
    if (!backend.UploadProjectionWeights(weights)) return 1;
  }
  if (probe_args.patch_validate) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;
    if (!backend.ValidatePatchEmbedding(config, weights, image)) return 1;
  }
  if (probe_args.qkv_validate) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;
    if (!backend.ValidateLayer0QKV(config, weights, image)) return 1;
  }
  if (probe_args.qkv_wmma2d_check) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;
    if (!backend.ValidateQkvWmma2D(config, weights, image)) return 1;
  }
  if (probe_args.mlp_validate) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;
    if (!backend.ValidateLayer0MLP(config, weights, image)) return 1;
  }
  if (probe_args.mlp_down_wmma8_check) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;
    if (!backend.ValidateMlpDownWmma8(config, weights, image)) return 1;
  }
  if (probe_args.mlp_up_wmma2d_check) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;
    if (!backend.ValidateMlpUpWmma2D(config, weights, image)) return 1;
  }
  if (probe_args.attn_out_validate) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;
    if (!backend.ValidateLayer0AttentionOut(config, weights, image)) return 1;
  }
  if (probe_args.attn_out_wmma2d_check) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;
    if (!backend.ValidateAttnOutWmma2D(config, weights, image)) return 1;
  }
  if (probe_args.layer0_block_validate) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;
    if (!backend.ValidateLayer0Block(config, weights, image)) return 1;
  }
  if (probe_args.device_attention_validate) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;
    if (!backend.ValidateLayer0DeviceAttention(config, weights, image)) {
      return 1;
    }
  }
  if (probe_args.layer0_block_device_attention_validate) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;
    if (!backend.ValidateLayer0BlockDeviceAttention(config, weights, image)) {
      return 1;
    }
  }
  if (probe_args.layer0_block_device_norm_validate) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;
    if (!backend.ValidateLayer0BlockDeviceNorm(config, weights, image)) {
      return 1;
    }
  }
  if (probe_args.layer_prefix2_device_norm_validate) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;
    if (!backend.ValidateLayerPrefix2DeviceNorm(config, weights, image)) {
      return 1;
    }
  }
  if (probe_args.layer_stack_device_norm_validate) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;
    if (!backend.ValidateLayerStackDeviceNorm(config, weights, image)) {
      return 1;
    }
  }
  if (probe_args.image_tokens_device_norm_validate) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;

    gcpp::MatMulEnv env(ctx);
    const size_t pool_dim = config.vit_config.pool_dim;
    gcpp::ImageTokens cpu_tokens(
        "cpu_image_tokens",
        gcpp::Extents2D(config.vit_config.seq_len / (pool_dim * pool_dim),
                        config.model_dim),
        env.ctx.allocator, gcpp::MatPadding::kOdd);
    cpu_tokens.AllocateAndAttachRowPtrs(env.row_ptrs);
    gcpp::RuntimeConfig runtime_config;
    runtime_config.verbosity = 0;
    gemma.GenerateImageTokens(runtime_config, /*seq_len=*/0, image, cpu_tokens,
                              env);

    gcpp::ImageTokens hip_tokens(
        "hip_image_tokens",
        gcpp::Extents2D(config.vit_config.seq_len / (pool_dim * pool_dim),
                        config.model_dim),
        env.ctx.allocator, gcpp::MatPadding::kOdd);
    hip_tokens.AllocateAndAttachRowPtrs(env.row_ptrs);
    if (!backend.ValidateImageTokensDeviceNorm(config, weights, image,
                                               &hip_tokens, &cpu_tokens)) {
      return 1;
    }
  }
  if (probe_args.mlp_solution_bench > 0) {
    if (!backend.BenchmarkMlpGemmSolutions(config, weights,
                                           probe_args.mlp_solution_bench)) {
      return 1;
    }
    if (probe_args.mlp_solution_cache_write) {
      if (!backend.SaveMlpGemmSolutionCache(config, weights,
                                            probe_args.mlp_solution_cache)) {
        return 1;
      }
    }
  }
  if (probe_args.attention_solution_bench > 0) {
    if (!backend.BenchmarkAttentionGemmSolutions(
            config, weights, probe_args.attention_solution_bench)) {
      return 1;
    }
    if (probe_args.attention_solution_cache_write) {
      if (!backend.SaveAttentionGemmSolutionCache(
              config, weights, probe_args.attention_solution_cache)) {
        return 1;
      }
    }
  }
  if (probe_args.image_tokens_return_bench > 0) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;

    gcpp::MatMulEnv env(ctx);
    const size_t pool_dim = config.vit_config.pool_dim;
    gcpp::ImageTokens hip_tokens(
        "hip_ret_tokens",
        gcpp::Extents2D(config.vit_config.seq_len / (pool_dim * pool_dim),
                        config.model_dim),
        env.ctx.allocator, gcpp::MatPadding::kOdd);
    hip_tokens.AllocateAndAttachRowPtrs(env.row_ptrs);

    for (int i = 0; i < probe_args.warmup; ++i) {
      if (!backend.GenerateImageTokensDeviceNorm(config, weights, image,
                                                 hip_tokens)) {
        return 1;
      }
    }
    std::vector<float> ms;
    ms.reserve(static_cast<size_t>(probe_args.image_tokens_return_bench));
    for (int i = 0; i < probe_args.image_tokens_return_bench; ++i) {
      const double start = hwy::platform::Now();
      if (!backend.GenerateImageTokensDeviceNorm(config, weights, image,
                                                 hip_tokens)) {
        return 1;
      }
      ms.push_back(static_cast<float>((hwy::platform::Now() - start) *
                                      1000.0));
    }
    std::sort(ms.begin(), ms.end());
    const float best_ms = ms.front();
    const float median_ms = ms[ms.size() / 2];
    std::printf("image_tokens_return_warm_bench warmup=%d iterations=%d "
                "best=%8.3f ms median=%8.3f ms\n",
                probe_args.warmup, probe_args.image_tokens_return_bench,
                best_ms, median_ms);
  }
  if (probe_args.image_tokens_return_profile) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;

    gcpp::MatMulEnv env(ctx);
    const size_t pool_dim = config.vit_config.pool_dim;
    gcpp::ImageTokens hip_tokens(
        "hip_prof_tokens",
        gcpp::Extents2D(config.vit_config.seq_len / (pool_dim * pool_dim),
                        config.model_dim),
        env.ctx.allocator, gcpp::MatPadding::kOdd);
    hip_tokens.AllocateAndAttachRowPtrs(env.row_ptrs);
    if (!backend.ProfileImageTokensReturnDeviceNorm(config, weights, image,
                                                    hip_tokens)) {
      return 1;
    }
  }
  if (probe_args.attention_qk_bf16_check) {
    gcpp::Image image;
    if (!load_probe_image(image)) return 1;

    gcpp::MatMulEnv env(ctx);
    const size_t pool_dim = config.vit_config.pool_dim;
    gcpp::ImageTokens f32_attention_tokens(
        "hip_f32_attn",
        gcpp::Extents2D(config.vit_config.seq_len / (pool_dim * pool_dim),
                        config.model_dim),
        env.ctx.allocator, gcpp::MatPadding::kOdd);
    f32_attention_tokens.AllocateAndAttachRowPtrs(env.row_ptrs);
    gcpp::ImageTokens qk_bf16_tokens(
        "hip_qk_bf16",
        gcpp::Extents2D(config.vit_config.seq_len / (pool_dim * pool_dim),
                        config.model_dim),
        env.ctx.allocator, gcpp::MatPadding::kOdd);
    qk_bf16_tokens.AllocateAndAttachRowPtrs(env.row_ptrs);

    if (!backend.ValidateAttentionQkBf16ImageTokens(
            config, weights, image, f32_attention_tokens, qk_bf16_tokens)) {
      return 1;
    }
  }
  if (probe_args.schedule) {
    std::printf("\nrocBLAS projection schedule from loaded model config:\n");
    if (!backend.RunModelProjectionSchedule(config)) return 1;
  }
  if (probe_args.resident_schedule) {
    std::printf("\nrocBLAS projection schedule using resident real weights:\n");
    if (!backend.RunResidentProjectionSchedule(config)) return 1;
  }

  return 0;
}
