// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "experimental/openpi_image_encoder_image_adapter.h"
#include "experimental/openpi_image_encoder_hip_backend.h"
#include "experimental/openpi_image_encoder_hip_bundle_bridge.h"
#include "experimental/openpi_image_encoder_raw_bundle.h"

namespace {

struct Args {
  std::string bundle;
  std::string image_path;
  std::string image_ppm_path;
  std::string tokens_out_path;
  std::string encoded_out_path;
  std::string attention = "hybrid";
  int warmup = 1;
  int runs = 5;
};

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
    if (arg == "--image" && i + 1 < argc) {
      args.image_path = argv[++i];
      continue;
    }
    if (arg == "--image_ppm" && i + 1 < argc) {
      args.image_ppm_path = argv[++i];
      continue;
    }
    if (arg == "--tokens_out" && i + 1 < argc) {
      args.tokens_out_path = argv[++i];
      continue;
    }
    if (arg == "--encoded_out" && i + 1 < argc) {
      args.encoded_out_path = argv[++i];
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
  if (args.bundle.empty()) return false;
  if (!args.image_path.empty() && !args.image_ppm_path.empty()) {
    std::fprintf(stderr, "--image and --image_ppm are mutually exclusive\n");
    return false;
  }
  return true;
}

bool WriteF32File(const std::string& path, const std::vector<float>& values) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::fprintf(stderr, "failed to open output file: %s\n", path.c_str());
    return false;
  }
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!file.good()) {
    std::fprintf(stderr, "failed to write output file: %s\n", path.c_str());
    return false;
  }
  return true;
}

struct Summary {
  double sum = 0.0;
  double mean = 0.0;
  double mean_abs = 0.0;
  float min = 0.0f;
  float max = 0.0f;
};

Summary Summarize(const std::vector<float>& values) {
  Summary summary;
  if (values.empty()) return summary;
  summary.min = values[0];
  summary.max = values[0];
  double sum_abs = 0.0;
  for (const float value : values) {
    summary.sum += value;
    sum_abs += std::abs(static_cast<double>(value));
    summary.min = std::min(summary.min, value);
    summary.max = std::max(summary.max, value);
  }
  summary.mean = summary.sum / static_cast<double>(values.size());
  summary.mean_abs = sum_abs / static_cast<double>(values.size());
  return summary;
}

void PrintUsage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --bundle DIR [--image IMAGE_F32] "
               "[--image_ppm IMAGE_PPM] "
               "[--attention serial|hybrid|parallel] [--warmup N] "
               "[--runs N] [--tokens_out TOKENS_F32] "
               "[--encoded_out ENCODED_F32]\n",
               argv0);
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, args)) {
    PrintUsage(argv[0]);
    return 2;
  }

  gcpp::experimental::OpenPiImageEncoderRawBundle bundle;
  if (!gcpp::experimental::LoadOpenPiImageEncoderRawBundle(args.bundle,
                                                           bundle)) {
    return 1;
  }
  std::string image_label = "<bundle>";
  if (!args.image_path.empty()) {
    if (!gcpp::experimental::LoadOpenPiImageEncoderRawImage(args.image_path,
                                                            bundle.image)) {
      return 1;
    }
    image_label = args.image_path;
  } else if (!args.image_ppm_path.empty()) {
    gcpp::Image image;
    if (!image.ReadPPM(args.image_ppm_path)) {
      return 1;
    }
    std::string error;
    if (!gcpp::experimental::CopyImageToOpenPiImageEncoderTensor(
            image, bundle.image, &error)) {
      std::fprintf(stderr, "failed to adapt PPM image: %s\n", error.c_str());
      return 1;
    }
    image_label = args.image_ppm_path;
  }

  const std::unique_ptr<gcpp::experimental::OpenPiImageEncoderHipBackend>
      backend =
          gcpp::experimental::CreateOpenPiImageEncoderHipBackendFromRawBundle(
              bundle);

  gcpp::experimental::OpenPiImageEncoderHipOptions options;
  options.attention = args.attention;
  options.warmup = args.warmup;
  options.runs = args.runs;
  const gcpp::experimental::OpenPiImageEncoderHipOutput output =
      backend->RunFromImage(bundle.image, options);
  if (!args.tokens_out_path.empty() &&
      !WriteF32File(args.tokens_out_path, output.tokens)) {
    return 1;
  }
  if (!args.encoded_out_path.empty() &&
      !WriteF32File(args.encoded_out_path, output.encoded)) {
    return 1;
  }
  const Summary tokens = Summarize(output.tokens);
  const Summary encoded = Summarize(output.encoded);

  std::printf("openpi_image_encoder_hip_run attention=%s image=%s "
              "tokens=%zu encoded=%zu\n",
              options.attention.c_str(), image_label.c_str(),
              output.tokens.size(), output.encoded.size());
  std::printf("encoded_summary sum=%.9g mean=%.9g mean_abs=%.9g min=%.9g "
              "max=%.9g\n",
              encoded.sum, encoded.mean, encoded.mean_abs, encoded.min,
              encoded.max);
  std::printf("tokens_summary sum=%.9g mean=%.9g mean_abs=%.9g min=%.9g "
              "max=%.9g\n",
              tokens.sum, tokens.mean, tokens.mean_abs, tokens.min,
              tokens.max);
  std::printf("hip_image_encoder_timing_ms patch=%.6g ln=%.6g qkv=%.6g "
              "attention=%.6g sa=%.6g mlp_up=%.6g gelu=%.6g "
              "mlp_down=%.6g final_norm=%.6g head=%.6g\n",
              output.patch_embed_ms, output.ln_ms, output.qkv_ms,
              output.attention_ms, output.sa_ms, output.mlp_up_ms,
              output.gelu_ms, output.mlp_down_ms, output.final_norm_ms,
              output.head_ms);
  return 0;
}
