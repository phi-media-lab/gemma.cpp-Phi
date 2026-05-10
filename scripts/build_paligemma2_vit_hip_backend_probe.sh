#!/usr/bin/env bash
# Builds the model-driven HIP probe for a future PaliGemma2 ViT backend.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
GPU_ARCH="${GPU_ARCH:-gfx1150}"
OUT="${OUT:-${ROOT_DIR}/build/paligemma2_vit_hip_backend_probe}"
BACKEND_OBJ="${OUT}.backend.o"
PROBE_OBJ="${OUT}.probe.o"

if [[ ! -x "${HIPCC}" ]]; then
  echo "error: hipcc not found at ${HIPCC}" >&2
  echo "hint: set ROCM_PATH or HIPCC" >&2
  exit 1
fi

cmake --build "${ROOT_DIR}/build" --target libgemma -j "$(nproc)"
mkdir -p "$(dirname "${OUT}")"

"${HIPCC}" \
  --offload-arch="${GPU_ARCH}" \
  -O3 \
  -std=c++17 \
  -x hip \
  -c \
  -I"${ROOT_DIR}" \
  -I"${ROOT_DIR}/build/_deps/highway-src" \
  -I"${ROOT_DIR}/build/_deps/sentencepiece-src/src" \
  -I"${ROOT_DIR}/build/_deps/sentencepiece-build/src" \
  -I"${ROCM_PATH}/include" \
  "${ROOT_DIR}/experimental/paligemma2_vit_hip_backend.cc" \
  -o "${BACKEND_OBJ}"

"${HIPCC}" \
  --offload-arch="${GPU_ARCH}" \
  -O3 \
  -std=c++17 \
  -c \
  -I"${ROOT_DIR}" \
  -I"${ROOT_DIR}/build/_deps/highway-src" \
  -I"${ROOT_DIR}/build/_deps/sentencepiece-src/src" \
  -I"${ROOT_DIR}/build/_deps/sentencepiece-build/src" \
  -I"${ROCM_PATH}/include" \
  "${ROOT_DIR}/experimental/paligemma2_vit_hip_backend_probe.cc" \
  -o "${PROBE_OBJ}"

"${HIPCC}" \
  --offload-arch="${GPU_ARCH}" \
  "${PROBE_OBJ}" \
  "${BACKEND_OBJ}" \
  "${ROOT_DIR}/build/libgemma.a" \
  "${ROOT_DIR}/build/_deps/highway-build/libhwy_contrib.a" \
  "${ROOT_DIR}/build/_deps/highway-build/libhwy.a" \
  "${ROOT_DIR}/build/_deps/sentencepiece-build/src/libsentencepiece.a" \
  -L"${ROCM_PATH}/lib" \
  -Wl,-rpath,"${ROCM_PATH}/lib" \
  -lrocblas \
  -pthread \
  -ldl \
  -o "${OUT}"

echo "Built ${OUT}"
