// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#ifndef THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_OPENPI_IMAGE_ENCODER_IMAGE_ADAPTER_H_
#define THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_OPENPI_IMAGE_ENCODER_IMAGE_ADAPTER_H_

#include <string>
#include <vector>

#include "paligemma/image.h"

namespace gcpp {
namespace experimental {

// Copies a gemma.cpp Image into the OpenPI image encoder input tensor layout.
//
// The caller is responsible for providing an already-normalized 224x224 RGB
// image. This function intentionally does not resize or normalize, because
// Image::ReadPPM/Image::Set already define those semantics in gemma.cpp.
bool CopyImageToOpenPiImageEncoderTensor(const Image& image,
                                         std::vector<float>& out,
                                         std::string* error = nullptr);

}  // namespace experimental
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_EXPERIMENTAL_OPENPI_IMAGE_ENCODER_IMAGE_ADAPTER_H_
