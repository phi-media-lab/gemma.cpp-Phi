#!/usr/bin/env bash
# Builds the standalone HIP/rocBLAS PaliGemma2 ViT GEMM benchmark.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ROCm is not guaranteed to be on PATH on this machine. Keep every important
# path overridable so the script works with /opt/rocm, /opt/rocm-*, or a custom
# hipcc without editing the repo.
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
GPU_ARCH="${GPU_ARCH:-gfx1150}"
OUT="${OUT:-${ROOT_DIR}/build/paligemma2_vit_hip_bench}"

if [[ ! -x "${HIPCC}" ]]; then
  echo "error: hipcc not found at ${HIPCC}" >&2
  echo "hint: set ROCM_PATH or HIPCC" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT}")"

# Build outside CMake because this is a local ROCm feasibility benchmark, not a
# portable gemma.cpp target. The rpath keeps librocblas discoverable at runtime
# even when ROCm's lib directory is not in the shell environment.
"${HIPCC}" \
  --offload-arch="${GPU_ARCH}" \
  -O3 \
  -std=c++17 \
  -I"${ROCM_PATH}/include" \
  "${ROOT_DIR}/experimental/paligemma2_vit_hip_bench.cc" \
  -L"${ROCM_PATH}/lib" \
  -Wl,-rpath,"${ROCM_PATH}/lib" \
  -lrocblas \
  -o "${OUT}"

echo "Built ${OUT}"
