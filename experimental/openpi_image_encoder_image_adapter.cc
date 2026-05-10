// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include "experimental/openpi_image_encoder_image_adapter.h"

#include <stddef.h>

#include <string>

#include "experimental/openpi_image_encoder_raw_bundle.h"

namespace gcpp {
namespace experimental {
namespace {

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

}  // namespace

bool CopyImageToOpenPiImageEncoderTensor(const Image& image,
                                         std::vector<float>& out,
                                         std::string* error) {
  constexpr size_t kExpectedSize =
      kOpenPiImageEncoderImageH * kOpenPiImageEncoderImageW *
      kOpenPiImageEncoderChannels;

  if (!image) {
    SetError(error, "image is empty");
    return false;
  }
  if (image.width() != static_cast<int>(kOpenPiImageEncoderImageW) ||
      image.height() != static_cast<int>(kOpenPiImageEncoderImageH)) {
    SetError(error, "image must be 224x224 RGB for the OpenPI image encoder");
    return false;
  }
  if (image.size() != kExpectedSize) {
    SetError(error,
             "image tensor size does not match 224x224x3 OpenPI input");
    return false;
  }

  const float* const data = image.data();
  if (data == nullptr) {
    SetError(error, "image data pointer is null");
    return false;
  }
  out.assign(data, data + kExpectedSize);
  return true;
}

}  // namespace experimental
}  // namespace gcpp
