#!/usr/bin/env bash
# Builds a CPU smoke probe for the OpenPI image-encoder raw bundle.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX="${CXX:-g++}"
OUT="${OUT:-${ROOT_DIR}/build/openpi_image_encoder_bundle_probe}"

mkdir -p "$(dirname "${OUT}")"

"${CXX}" \
  -O3 \
  -std=c++17 \
  -Wall \
  -Wextra \
  -pedantic \
  "${ROOT_DIR}/experimental/openpi_image_encoder_bundle_probe.cc" \
  -o "${OUT}"

echo "Built ${OUT}"
