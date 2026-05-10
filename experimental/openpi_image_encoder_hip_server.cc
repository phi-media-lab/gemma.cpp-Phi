// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "experimental/openpi_image_encoder_image_adapter.h"
#include "experimental/openpi_image_encoder_hip_backend.h"
#include "experimental/openpi_image_encoder_hip_bundle_bridge.h"
#include "experimental/openpi_image_encoder_raw_bundle.h"

namespace {

struct Args {
  std::string bundle;
  std::string attention = "hybrid";
  int warmup = 0;
  int runs = 1;
};

bool ParseInt(const std::string& value, int& out) {
  char* end = nullptr;
  const long parsed = std::strtol(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  out = static_cast<int>(parsed);
  return true;
}

bool IsAttentionMode(const std::string& value) {
  return value == "serial" || value == "hybrid" || value == "parallel";
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
      if (!IsAttentionMode(args.attention)) {
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
  return !args.bundle.empty();
}

void PrintUsage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --bundle DIR [--attention serial|hybrid|parallel] "
               "[--warmup N] [--runs N]\n",
               argv0);
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

std::unordered_map<std::string, std::string> ParseKeyValues(
    std::istringstream& stream) {
  std::unordered_map<std::string, std::string> values;
  std::string part;
  while (stream >> part) {
    const size_t equal = part.find('=');
    if (equal == std::string::npos || equal == 0) continue;
    values[part.substr(0, equal)] = part.substr(equal + 1);
  }
  return values;
}

bool LoadRequestImage(
    const std::unordered_map<std::string, std::string>& kv,
    const std::vector<float>& bundle_image,
    std::vector<float>& image,
    std::string& image_label,
    std::string& error) {
  const auto image_f32_it = kv.find("image_f32");
  const auto image_it = kv.find("image");
  const auto image_ppm_it = kv.find("image_ppm");
  const bool has_f32 = image_f32_it != kv.end() || image_it != kv.end();
  const bool has_ppm = image_ppm_it != kv.end();
  if (has_f32 && has_ppm) {
    error = "image_f32/image and image_ppm are mutually exclusive";
    return false;
  }

  if (has_ppm) {
    gcpp::Image gcpp_image;
    if (!gcpp_image.ReadPPM(image_ppm_it->second)) {
      error = "failed to read PPM image";
      return false;
    }
    if (!gcpp::experimental::CopyImageToOpenPiImageEncoderTensor(
            gcpp_image, image, &error)) {
      return false;
    }
    image_label = image_ppm_it->second;
    return true;
  }

  if (has_f32) {
    const std::string& path =
        image_f32_it != kv.end() ? image_f32_it->second : image_it->second;
    if (!gcpp::experimental::LoadOpenPiImageEncoderRawImage(path, image)) {
      error = "failed to read F32 image";
      return false;
    }
    image_label = path;
    return true;
  }

  image = bundle_image;
  image_label = "<bundle>";
  return true;
}

gcpp::experimental::OpenPiImageEncoderHipOptions RequestOptions(
    const Args& args,
    const std::unordered_map<std::string, std::string>& kv,
    std::string& error) {
  gcpp::experimental::OpenPiImageEncoderHipOptions options;
  options.attention = args.attention;
  options.warmup = args.warmup;
  options.runs = args.runs;

  if (const auto it = kv.find("attention"); it != kv.end()) {
    if (!IsAttentionMode(it->second)) {
      error = "unknown attention mode";
      return options;
    }
    options.attention = it->second;
  }
  if (const auto it = kv.find("warmup"); it != kv.end()) {
    if (!ParseInt(it->second, options.warmup)) {
      error = "invalid warmup";
      return options;
    }
  }
  if (const auto it = kv.find("runs"); it != kv.end()) {
    if (!ParseInt(it->second, options.runs)) {
      error = "invalid runs";
      return options;
    }
  }
  return options;
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
  const std::unique_ptr<gcpp::experimental::OpenPiImageEncoderHipBackend>
      backend =
          gcpp::experimental::CreateOpenPiImageEncoderHipBackendFromRawBundle(
              bundle);

  std::cout << "READY attention=" << args.attention
            << " warmup=" << args.warmup << " runs=" << args.runs << "\n"
            << std::flush;

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream stream(line);
    std::string command;
    stream >> command;
    if (command.empty()) continue;
    if (command == "QUIT") {
      std::cout << "BYE\n" << std::flush;
      return 0;
    }
    if (command == "PING") {
      std::cout << "OK pong=1\n" << std::flush;
      continue;
    }
    if (command != "RUN") {
      std::cout << "ERR unknown_command\n" << std::flush;
      continue;
    }

    const auto kv = ParseKeyValues(stream);
    std::string error;
    const gcpp::experimental::OpenPiImageEncoderHipOptions options =
        RequestOptions(args, kv, error);
    if (!error.empty()) {
      std::cout << "ERR " << error << "\n" << std::flush;
      continue;
    }

    std::vector<float> image;
    std::string image_label;
    if (!LoadRequestImage(kv, bundle.image, image, image_label, error)) {
      std::cout << "ERR " << error << "\n" << std::flush;
      continue;
    }

    const auto start = std::chrono::steady_clock::now();
    const gcpp::experimental::OpenPiImageEncoderHipOutput output =
        backend->RunFromImage(image, options);
    const auto end = std::chrono::steady_clock::now();

    if (const auto it = kv.find("tokens_out"); it != kv.end()) {
      if (!WriteF32File(it->second, output.tokens)) {
        std::cout << "ERR failed_to_write_tokens\n" << std::flush;
        continue;
      }
    }
    if (const auto it = kv.find("encoded_out"); it != kv.end()) {
      if (!WriteF32File(it->second, output.encoded)) {
        std::cout << "ERR failed_to_write_encoded\n" << std::flush;
        continue;
      }
    }

    const double wall_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "OK image=" << image_label << " attention="
              << options.attention << " tokens=" << output.tokens.size()
              << " encoded=" << output.encoded.size() << " wall_ms="
              << wall_ms << " patch_ms=" << output.patch_embed_ms
              << " ln_ms=" << output.ln_ms << " qkv_ms=" << output.qkv_ms
              << " attention_ms=" << output.attention_ms
              << " sa_ms=" << output.sa_ms << " mlp_up_ms="
              << output.mlp_up_ms << " gelu_ms=" << output.gelu_ms
              << " mlp_down_ms=" << output.mlp_down_ms
              << " final_norm_ms=" << output.final_norm_ms
              << " head_ms=" << output.head_ms << "\n"
              << std::flush;
  }
  return 0;
}
