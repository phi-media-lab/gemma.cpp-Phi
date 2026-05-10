// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include "experimental/openpi_image_encoder_hip_bundle_bridge.h"

namespace gcpp {
namespace experimental {

std::unique_ptr<OpenPiImageEncoderHipBackend>
CreateOpenPiImageEncoderHipBackendFromRawBundle(
    const OpenPiImageEncoderRawBundle& bundle) {
  return std::make_unique<OpenPiImageEncoderHipBackend>(
      bundle.block_ln0_scale_all, bundle.block_ln0_bias_all,
      bundle.block_ln1_scale_all, bundle.block_ln1_bias_all,
      bundle.block_q_bias_all, bundle.block_k_bias_all,
      bundle.block_v_bias_all, bundle.block_attn_out_bias_all,
      bundle.block_mlp_up_bias_all, bundle.block_mlp_down_bias_all,
      bundle.encoder_norm_scale, bundle.encoder_norm_bias, bundle.head_bias,
      bundle.block_q_kernel_all, bundle.block_k_kernel_all,
      bundle.block_v_kernel_all, bundle.block_attn_out_kernel_all,
      bundle.block_mlp_up_kernel_all, bundle.block_mlp_down_kernel_all,
      bundle.head_kernel, bundle.patch_embed_kernel, bundle.patch_embed_bias,
      bundle.pos_embedding);
}

}  // namespace experimental
}  // namespace gcpp
