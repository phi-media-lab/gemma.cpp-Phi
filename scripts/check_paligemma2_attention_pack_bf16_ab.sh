#!/usr/bin/env bash
# Runs deterministic generation A/B checks for the HIP PaliGemma2 F32 attention
# pack-to-BF16 optimization. The candidate keeps QK, softmax, and AV in F32 and
# only fuses the post-attention head pack with the existing BF16 conversion used
# by the following attn_out GEMM.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-${ROOT_DIR}/build/gemma_paligemma2_vit_hip}"
DEFAULT_IMAGE="${IMAGE:-${ROOT_DIR}/paligemma/testdata/image.ppm}"
MODEL_224="${MODEL_224:-${ROOT_DIR}/build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs}"
MODEL_448="${MODEL_448:-${ROOT_DIR}/build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs}"
ATTENTION_CACHE="${ATTENTION_CACHE:-${ROOT_DIR}/build/paligemma2_vit_attention_solution_cache.txt}"
MLP_CACHE_448="${MLP_CACHE_448:-${ROOT_DIR}/build/paligemma2_vit_mlp_solution_cache_448.txt}"
MAX_GENERATED_TOKENS="${MAX_GENERATED_TOKENS:-16}"
KEEP_GOING="${KEEP_GOING:-0}"

IMAGES=()
PROMPTS=()

load_images() {
  if [[ -n "${IMAGES_FILE:-}" ]]; then
    if [[ ! -f "${IMAGES_FILE}" ]]; then
      echo "error: missing IMAGES_FILE=${IMAGES_FILE}" >&2
      exit 1
    fi
    local line
    while IFS= read -r line || [[ -n "${line}" ]]; do
      [[ -z "${line//[[:space:]]/}" ]] && continue
      [[ "${line}" =~ ^[[:space:]]*# ]] && continue
      IMAGES+=("${line}")
    done <"${IMAGES_FILE}"
  else
    IMAGES=("${DEFAULT_IMAGE}")
  fi
  if [[ "${#IMAGES[@]}" -eq 0 ]]; then
    echo "error: no images to check" >&2
    exit 1
  fi
  for image in "${IMAGES[@]}"; do
    if [[ ! -f "${image}" ]]; then
      echo "error: missing image ${image}" >&2
      exit 1
    fi
  done
}

load_prompts() {
  if [[ -n "${PROMPTS_FILE:-}" ]]; then
    if [[ ! -f "${PROMPTS_FILE}" ]]; then
      echo "error: missing PROMPTS_FILE=${PROMPTS_FILE}" >&2
      exit 1
    fi
    local line
    while IFS= read -r line || [[ -n "${line}" ]]; do
      [[ -z "${line//[[:space:]]/}" ]] && continue
      [[ "${line}" =~ ^[[:space:]]*# ]] && continue
      PROMPTS+=("${line}")
    done <"${PROMPTS_FILE}"
  else
    PROMPTS=(
      "Describe the image."
      "What is in the image?"
      "Describe the building."
    )
  fi
  if [[ "${#PROMPTS[@]}" -eq 0 ]]; then
    echo "error: no prompts to check" >&2
    exit 1
  fi
}

if [[ ! -x "${BIN}" ]]; then
  echo "error: missing executable ${BIN}" >&2
  echo "hint: run scripts/build_gemma_paligemma2_vit_hip.sh first" >&2
  exit 1
fi
load_images
load_prompts

run_generation() {
  local image="$1"
  local weights="$2"
  local prompt="$3"
  local attention_mode="$4"
  shift 4
  local cache_args=("$@")
  local pack_args=()
  if [[ "${attention_mode}" == "pack_bf16" ]]; then
    pack_args=(--paligemma_vit_hip_attention_pack_bf16 1)
  fi

  "${BIN}" \
    --weights "${weights}" \
    --image_file "${image}" \
    --prompt "${prompt}" \
    --verbosity 0 \
    --max_generated_tokens "${MAX_GENERATED_TOKENS}" \
    --temperature 0 \
    --top_k 1 \
    --paligemma_vit_backend hip_probe \
    --paligemma_vit_hip_return_image_tokens 1 \
    --paligemma_vit_hip_attention_direct_qkv 1 \
    "${pack_args[@]}" \
    --paligemma_vit_hip_qkv_wmma2d 1 \
    --paligemma_vit_hip_attn_out_wmma2d 1 \
    --paligemma_vit_hip_mlp_up_wmma2d 1 \
    --paligemma_vit_hip_mlp_down_wmma8 1 \
    --paligemma_vit_hip_mlp_down_fused_residual 1 \
    "${cache_args[@]}"
}

check_model() {
  local label="$1"
  local weights="$2"
  shift 2
  local cache_args=("$@")
  local failures=0
  if [[ ! -f "${weights}" ]]; then
    echo "skip ${label}: missing ${weights}" >&2
    return 0
  fi

  for image in "${IMAGES[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
      local f32_output
      local candidate_output
      f32_output="$(run_generation "${image}" "${weights}" "${prompt}" f32 \
        "${cache_args[@]}")"
      candidate_output="$(run_generation "${image}" "${weights}" "${prompt}" \
        pack_bf16 "${cache_args[@]}")"
      if [[ "${f32_output}" != "${candidate_output}" ]]; then
        echo "FAIL ${label}: image=${image} prompt=${prompt}" >&2
        echo "  f32_attn   : ${f32_output}" >&2
        echo "  pack_bf16  : ${candidate_output}" >&2
        failures=$((failures + 1))
        if [[ "${KEEP_GOING}" != "1" ]]; then
          return 1
        fi
        continue
      fi
      echo "PASS ${label}: image=${image} prompt=${prompt}"
      echo "  ${f32_output}"
    done
  done
  if [[ "${failures}" -ne 0 ]]; then
    echo "FAIL ${label}: ${failures} mismatched generation case(s)" >&2
    return 1
  fi
}

common_cache_args=()
if [[ -f "${ATTENTION_CACHE}" ]]; then
  common_cache_args=(--paligemma_vit_hip_attention_solution_cache \
    "${ATTENTION_CACHE}")
fi

failures=0
if ! check_model "224" "${MODEL_224}" "${common_cache_args[@]}"; then
  failures=$((failures + 1))
  if [[ "${KEEP_GOING}" != "1" ]]; then
    exit 1
  fi
fi

cache_args_448=("${common_cache_args[@]}")
if [[ -f "${MLP_CACHE_448}" ]]; then
  cache_args_448+=(--paligemma_vit_hip_mlp_solution_cache "${MLP_CACHE_448}")
fi
if ! check_model "448" "${MODEL_448}" "${cache_args_448[@]}"; then
  failures=$((failures + 1))
fi
if [[ "${failures}" -ne 0 ]]; then
  echo "FAIL: ${failures} model group(s) had mismatches" >&2
  exit 1
fi
