#!/usr/bin/env bash
# Runs deterministic generation smoke tests for the current recommended
# PaliGemma2 ViT HIP configuration.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-${ROOT_DIR}/build/gemma_paligemma2_vit_hip}"
IMAGE="${IMAGE:-${ROOT_DIR}/paligemma/testdata/image.ppm}"
MODEL_224="${MODEL_224:-${ROOT_DIR}/build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs}"
MODEL_448="${MODEL_448:-${ROOT_DIR}/build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs}"
ATTENTION_CACHE="${ATTENTION_CACHE:-${ROOT_DIR}/build/paligemma2_vit_attention_solution_cache.txt}"
MLP_CACHE_448="${MLP_CACHE_448:-${ROOT_DIR}/build/paligemma2_vit_mlp_solution_cache_448.txt}"
PROMPT="${PROMPT:-Describe the image.}"
MAX_GENERATED_TOKENS="${MAX_GENERATED_TOKENS:-16}"
REBUILD="${REBUILD:-1}"

if [[ "${REBUILD}" == "1" ]]; then
  "${ROOT_DIR}/scripts/build_gemma_paligemma2_vit_hip.sh"
fi
if [[ ! -x "${BIN}" ]]; then
  echo "error: missing executable ${BIN}" >&2
  echo "hint: run scripts/build_gemma_paligemma2_vit_hip.sh first" >&2
  exit 1
fi
if [[ ! -f "${ATTENTION_CACHE}" ]]; then
  echo "error: missing attention cache ${ATTENTION_CACHE}" >&2
  echo "hint: run scripts/build_paligemma2_vit_attention_solution_cache.sh first" >&2
  exit 1
fi
if [[ ! -f "${IMAGE}" ]]; then
  echo "error: missing image ${IMAGE}" >&2
  exit 1
fi

run_one() {
  local label="$1"
  local weights="$2"
  shift 2
  local extra_args=("$@")
  if [[ ! -f "${weights}" ]]; then
    echo "skip ${label}: missing ${weights}" >&2
    return 0
  fi

  echo "== ${label} =="
  "${BIN}" \
    --weights "${weights}" \
    --image_file "${IMAGE}" \
    --prompt "${PROMPT}" \
    --verbosity 0 \
    --max_generated_tokens "${MAX_GENERATED_TOKENS}" \
    --temperature 0 \
    --top_k 1 \
    --paligemma_vit_backend hip_probe \
    --paligemma_vit_hip_return_image_tokens 1 \
    --paligemma_vit_hip_attention_pack_bf16 1 \
    --paligemma_vit_hip_attention_direct_qkv 1 \
    --paligemma_vit_hip_qkv_wmma2d 1 \
    --paligemma_vit_hip_attn_out_wmma2d 1 \
    --paligemma_vit_hip_mlp_up_wmma2d 1 \
    --paligemma_vit_hip_mlp_down_wmma8 1 \
    --paligemma_vit_hip_mlp_down_fused_residual 1 \
    --paligemma_vit_hip_attention_solution_cache "${ATTENTION_CACHE}" \
    "${extra_args[@]}"
}

run_one "224" "${MODEL_224}"

cache_args_448=()
if [[ -f "${MLP_CACHE_448}" ]]; then
  cache_args_448=(--paligemma_vit_hip_mlp_solution_cache "${MLP_CACHE_448}")
fi
cache_args_448+=(--paligemma_vit_hip_mlp_down_wmma_waves 12)
run_one "448" "${MODEL_448}" "${cache_args_448[@]}"
