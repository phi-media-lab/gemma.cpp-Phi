// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_OPENPI_IMAGE_ENCODER_HIP_BUNDLE_BRIDGE_H_
#define THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_OPENPI_IMAGE_ENCODER_HIP_BUNDLE_BRIDGE_H_

#include <memory>

#include "experimental/openpi_image_encoder_hip_backend.h"
#include "experimental/openpi_image_encoder_raw_bundle.h"

namespace gcpp {
namespace experimental {

std::unique_ptr<OpenPiImageEncoderHipBackend>
CreateOpenPiImageEncoderHipBackendFromRawBundle(
    const OpenPiImageEncoderRawBundle& bundle);

}  // namespace experimental
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_OPENPI_IMAGE_ENCODER_HIP_BUNDLE_BRIDGE_H_
