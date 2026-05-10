// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_OPENPI_IMAGE_ENCODER_RAW_BUNDLE_H_
#define THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_OPENPI_IMAGE_ENCODER_RAW_BUNDLE_H_

#include <stddef.h>

#include <string>
#include <vector>

namespace gcpp {
namespace experimental {

constexpr size_t kOpenPiImageEncoderLayers = 27;
constexpr size_t kOpenPiImageEncoderImageH = 224;
constexpr size_t kOpenPiImageEncoderImageW = 224;
constexpr size_t kOpenPiImageEncoderChannels = 3;
constexpr size_t kOpenPiImageEncoderPatch = 14;
constexpr size_t kOpenPiImageEncoderTokens = 256;
constexpr size_t kOpenPiImageEncoderWidth = 1152;
constexpr size_t kOpenPiImageEncoderHeads = 16;
constexpr size_t kOpenPiImageEncoderHeadDim = 72;
constexpr size_t kOpenPiImageEncoderQDim =
    kOpenPiImageEncoderHeads * kOpenPiImageEncoderHeadDim;
constexpr size_t kOpenPiImageEncoderMlpDim = 4304;
constexpr size_t kOpenPiImageEncoderDecoderWidth = 2048;

struct OpenPiImageEncoderRawBundle {
  std::vector<float> image;
  std::vector<float> patch_embed_kernel;
  std::vector<float> patch_embed_bias;
  std::vector<float> pos_embedding;

  std::vector<float> block_ln0_scale_all;
  std::vector<float> block_ln0_bias_all;
  std::vector<float> block_q_kernel_all;
  std::vector<float> block_q_bias_all;
  std::vector<float> block_k_kernel_all;
  std::vector<float> block_k_bias_all;
  std::vector<float> block_v_kernel_all;
  std::vector<float> block_v_bias_all;
  std::vector<float> block_attn_out_kernel_all;
  std::vector<float> block_attn_out_bias_all;
  std::vector<float> block_ln1_scale_all;
  std::vector<float> block_ln1_bias_all;
  std::vector<float> block_mlp_up_kernel_all;
  std::vector<float> block_mlp_up_bias_all;
  std::vector<float> block_mlp_down_kernel_all;
  std::vector<float> block_mlp_down_bias_all;
  std::vector<float> encoder_norm_scale;
  std::vector<float> encoder_norm_bias;
  std::vector<float> head_kernel;
  std::vector<float> head_bias;

  std::vector<float> golden_stem;
  std::vector<float> golden_with_posemb;
  std::vector<float> golden_encoded;
  std::vector<float> golden_tokens;
  std::vector<float> golden_block00_sa;
  std::vector<float> golden_block00_plus_sa;
  std::vector<float> golden_block00_mlp;
  std::vector<float> golden_block00_plus_mlp;
};

bool LoadOpenPiImageEncoderRawBundle(
    const std::string& bundle_dir,
    OpenPiImageEncoderRawBundle& bundle);

bool LoadOpenPiImageEncoderRawImage(
    const std::string& path,
    std::vector<float>& image);

}  // namespace experimental
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_OPENPI_IMAGE_ENCODER_RAW_BUNDLE_H_
