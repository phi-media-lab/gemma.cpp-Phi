// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include "experimental/openpi_image_encoder_raw_bundle.h"

#include <cstdio>
#include <fstream>
#include <string>

namespace gcpp {
namespace experimental {
namespace {

std::string JoinPath(const std::string& dir, const std::string& file) {
  if (dir.empty() || dir.back() == '/') return dir + file;
  return dir + "/" + file;
}

bool LoadF32(const std::string& path, size_t count, std::vector<float>& out) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    std::fprintf(stderr, "failed to open %s\n", path.c_str());
    return false;
  }
  const std::streamsize bytes = stream.tellg();
  const std::streamsize expected =
      static_cast<std::streamsize>(count * sizeof(float));
  if (bytes != expected) {
    std::fprintf(stderr, "unexpected size for %s: got %lld expected %lld\n",
                 path.c_str(), static_cast<long long>(bytes),
                 static_cast<long long>(expected));
    return false;
  }
  stream.seekg(0, std::ios::beg);
  out.resize(count);
  if (!stream.read(reinterpret_cast<char*>(out.data()), bytes)) {
    std::fprintf(stderr, "failed to read %s\n", path.c_str());
    return false;
  }
  return true;
}

}  // namespace

bool LoadOpenPiImageEncoderRawBundle(
    const std::string& bundle_dir,
    OpenPiImageEncoderRawBundle& bundle) {
  constexpr size_t kLayers = kOpenPiImageEncoderLayers;
  constexpr size_t kImageH = kOpenPiImageEncoderImageH;
  constexpr size_t kImageW = kOpenPiImageEncoderImageW;
  constexpr size_t kChannels = kOpenPiImageEncoderChannels;
  constexpr size_t kPatch = kOpenPiImageEncoderPatch;
  constexpr size_t kTokens = kOpenPiImageEncoderTokens;
  constexpr size_t kWidth = kOpenPiImageEncoderWidth;
  constexpr size_t kQDim = kOpenPiImageEncoderQDim;
  constexpr size_t kMlpDim = kOpenPiImageEncoderMlpDim;
  constexpr size_t kDecoderWidth = kOpenPiImageEncoderDecoderWidth;

  return LoadF32(JoinPath(bundle_dir, "golden_image.f32"),
                 kImageH * kImageW * kChannels, bundle.image) &&
         LoadF32(JoinPath(bundle_dir, "weight_patch_embed_kernel.f32"),
                 kPatch * kPatch * kChannels * kWidth,
                 bundle.patch_embed_kernel) &&
         LoadF32(JoinPath(bundle_dir, "weight_patch_embed_bias.f32"), kWidth,
                 bundle.patch_embed_bias) &&
         LoadF32(JoinPath(bundle_dir, "weight_pos_embedding.f32"),
                 kTokens * kWidth, bundle.pos_embedding) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_ln0_scale.f32"),
                 kLayers * kWidth, bundle.block_ln0_scale_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_ln0_bias.f32"),
                 kLayers * kWidth, bundle.block_ln0_bias_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_q_kernel.f32"),
                 kLayers * kWidth * kQDim, bundle.block_q_kernel_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_q_bias.f32"),
                 kLayers * kQDim, bundle.block_q_bias_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_k_kernel.f32"),
                 kLayers * kWidth * kQDim, bundle.block_k_kernel_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_k_bias.f32"),
                 kLayers * kQDim, bundle.block_k_bias_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_v_kernel.f32"),
                 kLayers * kWidth * kQDim, bundle.block_v_kernel_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_v_bias.f32"),
                 kLayers * kQDim, bundle.block_v_bias_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_attn_out_kernel.f32"),
                 kLayers * kQDim * kWidth,
                 bundle.block_attn_out_kernel_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_attn_out_bias.f32"),
                 kLayers * kWidth, bundle.block_attn_out_bias_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_ln1_scale.f32"),
                 kLayers * kWidth, bundle.block_ln1_scale_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_ln1_bias.f32"),
                 kLayers * kWidth, bundle.block_ln1_bias_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_mlp_up_kernel.f32"),
                 kLayers * kWidth * kMlpDim,
                 bundle.block_mlp_up_kernel_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_mlp_up_bias.f32"),
                 kLayers * kMlpDim, bundle.block_mlp_up_bias_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_mlp_down_kernel.f32"),
                 kLayers * kMlpDim * kWidth,
                 bundle.block_mlp_down_kernel_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_block_mlp_down_bias.f32"),
                 kLayers * kWidth, bundle.block_mlp_down_bias_all) &&
         LoadF32(JoinPath(bundle_dir, "weight_encoder_norm_scale.f32"),
                 kWidth, bundle.encoder_norm_scale) &&
         LoadF32(JoinPath(bundle_dir, "weight_encoder_norm_bias.f32"),
                 kWidth, bundle.encoder_norm_bias) &&
         LoadF32(JoinPath(bundle_dir, "weight_head_kernel.f32"),
                 kWidth * kDecoderWidth, bundle.head_kernel) &&
         LoadF32(JoinPath(bundle_dir, "weight_head_bias.f32"),
                 kDecoderWidth, bundle.head_bias) &&
         LoadF32(JoinPath(bundle_dir, "golden_intermediate_stem.f32"),
                 kTokens * kWidth, bundle.golden_stem) &&
         LoadF32(JoinPath(bundle_dir, "golden_intermediate_with_posemb.f32"),
                 kTokens * kWidth, bundle.golden_with_posemb) &&
         LoadF32(JoinPath(bundle_dir, "golden_intermediate_encoded.f32"),
                 kTokens * kWidth, bundle.golden_encoded) &&
         LoadF32(JoinPath(bundle_dir, "golden_tokens.f32"),
                 kTokens * kDecoderWidth, bundle.golden_tokens) &&
         LoadF32(JoinPath(bundle_dir, "golden_intermediate_encoder_block00_sa.f32"),
                 kTokens * kWidth, bundle.golden_block00_sa) &&
         LoadF32(JoinPath(bundle_dir,
                          "golden_intermediate_encoder_block00_plus_sa.f32"),
                 kTokens * kWidth, bundle.golden_block00_plus_sa) &&
         LoadF32(JoinPath(bundle_dir,
                          "golden_intermediate_encoder_block00_mlp.f32"),
                 kTokens * kWidth, bundle.golden_block00_mlp) &&
         LoadF32(JoinPath(bundle_dir,
                          "golden_intermediate_encoder_block00_plus_mlp.f32"),
                 kTokens * kWidth, bundle.golden_block00_plus_mlp);
}

bool LoadOpenPiImageEncoderRawImage(
    const std::string& path,
    std::vector<float>& image) {
  return LoadF32(path,
                 kOpenPiImageEncoderImageH * kOpenPiImageEncoderImageW *
                     kOpenPiImageEncoderChannels,
                 image);
}

}  // namespace experimental
}  // namespace gcpp
