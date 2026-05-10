#!/usr/bin/env bash
# Builds the resident HIP/rocBLAS OpenPI image-encoder server binary.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
GPU_ARCH="${GPU_ARCH:-gfx1150}"
OUT="${OUT:-${ROOT_DIR}/build/openpi_image_encoder_hip_server}"
BUNDLE_OBJ="${OUT}.bundle.o"
ADAPTER_OBJ="${OUT}.image_adapter.o"
BRIDGE_OBJ="${OUT}.bridge.o"
BACKEND_OBJ="${OUT}.backend.o"
IO_OBJ="${OUT}.io.o"
IMAGE_OBJ="${OUT}.image.o"
SERVER_OBJ="${OUT}.server.o"

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
  -c \
  -Wall \
  -Wextra \
  -pedantic \
  -Wno-gnu-zero-variadic-macro-arguments \
  -Wno-c++20-designator \
  -Wno-unused-parameter \
  -DHWY_STATIC_DEFINE \
  -DTOOLCHAIN_MISS_ASM_HWCAP_H \
  -I"${ROOT_DIR}" \
  -I"${ROOT_DIR}/build/_deps/highway-src" \
  "${ROOT_DIR}/io/io.cc" \
  -o "${IO_OBJ}"

"${HIPCC}" \
  --offload-arch="${GPU_ARCH}" \
  -O3 \
  -std=c++17 \
  -c \
  -Wall \
  -Wextra \
  -pedantic \
  -Wno-gnu-zero-variadic-macro-arguments \
  -DHWY_STATIC_DEFINE \
  -DTOOLCHAIN_MISS_ASM_HWCAP_H \
  -I"${ROOT_DIR}" \
  -I"${ROOT_DIR}/build/_deps/highway-src" \
  "${ROOT_DIR}/paligemma/image.cc" \
  -o "${IMAGE_OBJ}"

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
  -I"${ROOT_DIR}/build/_deps/highway-src" \
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
  -I"${ROOT_DIR}/build/_deps/highway-src" \
  -I"${ROCM_PATH}/include" \
  "${ROOT_DIR}/experimental/openpi_image_encoder_hip_server.cc" \
  -o "${SERVER_OBJ}"

"${HIPCC}" \
  --offload-arch="${GPU_ARCH}" \
  "${SERVER_OBJ}" \
  "${BACKEND_OBJ}" \
  "${BRIDGE_OBJ}" \
  "${ADAPTER_OBJ}" \
  "${BUNDLE_OBJ}" \
  "${IMAGE_OBJ}" \
  "${IO_OBJ}" \
  -L"${ROCM_PATH}/lib" \
  -L"${ROOT_DIR}/build/_deps/highway-build" \
  -Wl,-rpath,"${ROCM_PATH}/lib" \
  -lrocblas \
  -lhwy \
  -o "${OUT}"

echo "Built ${OUT}"
