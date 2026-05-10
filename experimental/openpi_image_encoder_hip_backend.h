// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_OPENPI_IMAGE_ENCODER_HIP_BACKEND_H_
#define THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_OPENPI_IMAGE_ENCODER_HIP_BACKEND_H_

#include <memory>
#include <string>
#include <vector>

namespace gcpp {
namespace experimental {

struct OpenPiImageEncoderHipOptions {
  std::string attention = "serial";
  int warmup = 3;
  int runs = 20;
};

struct OpenPiImageEncoderHipMetrics {
  double max_abs = 0.0;
  double rms = 0.0;
  double mean_abs = 0.0;
};

struct OpenPiImageEncoderHipOutput {
  std::vector<float> encoded;
  std::vector<float> tokens;
  float patch_embed_ms = 0.0f;
  float ln_ms = 0.0f;
  float qkv_ms = 0.0f;
  float attention_ms = 0.0f;
  float sa_ms = 0.0f;
  float mlp_up_ms = 0.0f;
  float gelu_ms = 0.0f;
  float mlp_down_ms = 0.0f;
  float final_norm_ms = 0.0f;
  float head_ms = 0.0f;
};

struct OpenPiImageEncoderHipRunResult : public OpenPiImageEncoderHipOutput {
  OpenPiImageEncoderHipMetrics encoded_metrics;
  OpenPiImageEncoderHipMetrics tokens_metrics;
};

struct OpenPiImageEncoderHipImageRunResult
    : public OpenPiImageEncoderHipRunResult {
  OpenPiImageEncoderHipMetrics stem_metrics;
  OpenPiImageEncoderHipMetrics posemb_metrics;
};

// Probe-stage OpenPI image encoder backend for the fixed pi0.5 image-token
// shape: 27 ViT layers, 256 tokens, width 1152, and 2048 decoder-visible image
// token width. The caller owns raw-bundle loading; this object owns resident
// device weights and executes the full image encoder from the with-posemb tensor
// to decoder image tokens.
class OpenPiImageEncoderHipBackend {
 public:
  OpenPiImageEncoderHipBackend(
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
      const std::vector<float>& pos_embedding);
  ~OpenPiImageEncoderHipBackend();

  OpenPiImageEncoderHipBackend(const OpenPiImageEncoderHipBackend&) = delete;
  OpenPiImageEncoderHipBackend& operator=(
      const OpenPiImageEncoderHipBackend&) = delete;

  OpenPiImageEncoderHipOutput Run(
      const std::vector<float>& with_posemb,
      const OpenPiImageEncoderHipOptions& options) const;

  OpenPiImageEncoderHipOutput RunFromImage(
      const std::vector<float>& image,
      const OpenPiImageEncoderHipOptions& options) const;

  OpenPiImageEncoderHipRunResult RunWithGolden(
      const std::vector<float>& with_posemb,
      const std::vector<float>& golden_encoded,
      const std::vector<float>& golden_tokens,
      const OpenPiImageEncoderHipOptions& options) const;

  OpenPiImageEncoderHipImageRunResult RunFromImageWithGolden(
      const std::vector<float>& image,
      const std::vector<float>& golden_stem,
      const std::vector<float>& golden_with_posemb,
      const std::vector<float>& golden_encoded,
      const std::vector<float>& golden_tokens,
      const OpenPiImageEncoderHipOptions& options) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace experimental
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_OPENPI_IMAGE_ENCODER_HIP_BACKEND_H_
