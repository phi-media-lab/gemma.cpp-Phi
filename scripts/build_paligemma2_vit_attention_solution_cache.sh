#!/usr/bin/env bash
# Benchmarks rocBLAS attention QK/AV solution indices for PaliGemma2 ViT and
# writes a shape/device/runtime keyed cache for 224px and 448px models.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE="${PROBE:-${ROOT_DIR}/build/paligemma2_vit_hip_backend_probe}"
MODEL_224="${MODEL_224:-${ROOT_DIR}/build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs}"
MODEL_448="${MODEL_448:-${ROOT_DIR}/build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs}"
CACHE="${CACHE:-${ROOT_DIR}/build/paligemma2_vit_attention_solution_cache.txt}"
MAX_SOLUTIONS="${MAX_SOLUTIONS:-8}"
SAMPLES="${SAMPLES:-20}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-1}"
REBUILD="${REBUILD:-1}"

if [[ "${REBUILD}" == "1" ]]; then
  "${ROOT_DIR}/scripts/build_paligemma2_vit_hip_backend_probe.sh"
fi
if [[ ! -x "${PROBE}" ]]; then
  echo "error: missing executable ${PROBE}" >&2
  echo "hint: run scripts/build_paligemma2_vit_hip_backend_probe.sh first" >&2
  exit 1
fi

bench_one() {
  local label="$1"
  local weights="$2"
  if [[ ! -f "${weights}" ]]; then
    echo "skip ${label}: missing ${weights}" >&2
    return 0
  fi

  "${PROBE}" \
    --weights "${weights}" \
    --hip_schedule 0 \
    --hip_verbose 0 \
    --hip_attention_solution_bench "${MAX_SOLUTIONS}" \
    --hip_attention_solution_cache "${CACHE}" \
    --hip_attention_solution_cache_write 1 \
    --hip_samples "${SAMPLES}" \
    --hip_warmup "${WARMUP}" \
    --hip_iters "${ITERS}"
}

mkdir -p "$(dirname "${CACHE}")"
rm -f "${CACHE}"

bench_one "224" "${MODEL_224}"
bench_one "448" "${MODEL_448}"

echo "Wrote ${CACHE}"
