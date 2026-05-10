#!/usr/bin/env bash
# Builds the standalone HIP/rocBLAS OpenPI image-encoder raw-bundle probe.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
GPU_ARCH="${GPU_ARCH:-gfx1150}"
OUT="${OUT:-${ROOT_DIR}/build/openpi_image_encoder_hip_probe}"
BUNDLE_OBJ="${OUT}.bundle.o"
ADAPTER_OBJ="${OUT}.image_adapter.o"
BRIDGE_OBJ="${OUT}.bridge.o"
BACKEND_OBJ="${OUT}.backend.o"
PROBE_OBJ="${OUT}.probe.o"

if [[ ! -x "${HIPCC}" ]]; then
  echo "error: hipcc not found at ${HIPCC}" >&2
  echo "hint: set ROCM_PATH or HIPCC" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT}")"

"${HIPCC}" \
  --offload-arch="${GPU_ARCH}" \
  -O3 \
  -std=c++17 \
  -c \
  -Wall \
  -Wextra \
  -pedantic \
  -I"${ROOT_DIR}" \
  "${ROOT_DIR}/experimental/openpi_image_encoder_raw_bundle.cc" \
  -o "${BUNDLE_OBJ}"

"${HIPCC}" \
  --offload-arch="${GPU_ARCH}" \
  -O3 \
  -std=c++17 \
  -c \
  -Wall \
  -Wextra \
  -pedantic \
  -I"${ROOT_DIR}" \
  -I"${ROOT_DIR}/build/_deps/highway-src" \
  "${ROOT_DIR}/experimental/openpi_image_encoder_image_adapter.cc" \
  -o "${ADAPTER_OBJ}"

"${HIPCC}" \
  --offload-arch="${GPU_ARCH}" \
  -O3 \
  -std=c++17 \
  -c \
  -Wall \
  -Wextra \
  -pedantic \
  -I"${ROOT_DIR}" \
  "${ROOT_DIR}/experimental/openpi_image_encoder_hip_bundle_bridge.cc" \
  -o "${BRIDGE_OBJ}"

"${HIPCC}" \
  --offload-arch="${GPU_ARCH}" \
  -O3 \
  -std=c++17 \
  -x hip \
  -c \
  -Wall \
  -Wextra \
  -pedantic \
  -I"${ROOT_DIR}" \
  -I"${ROCM_PATH}/include" \
  "${ROOT_DIR}/experimental/openpi_image_encoder_hip_backend.cc" \
  -o "${BACKEND_OBJ}"

"${HIPCC}" \
  --offload-arch="${GPU_ARCH}" \
  -O3 \
  -std=c++17 \
  -x hip \
  -c \
  -Wall \
  -Wextra \
  -pedantic \
  -I"${ROOT_DIR}" \
  -I"${ROCM_PATH}/include" \
  "${ROOT_DIR}/experimental/openpi_image_encoder_hip_probe.cc" \
  -o "${PROBE_OBJ}"

"${HIPCC}" \
  --offload-arch="${GPU_ARCH}" \
  "${PROBE_OBJ}" \
  "${BACKEND_OBJ}" \
  "${BRIDGE_OBJ}" \
  "${ADAPTER_OBJ}" \
  "${BUNDLE_OBJ}" \
  -L"${ROCM_PATH}/lib" \
  -Wl,-rpath,"${ROCM_PATH}/lib" \
  -lrocblas \
  -o "${OUT}"

echo "Built ${OUT}"
