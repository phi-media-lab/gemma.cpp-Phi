#!/usr/bin/env bash
# Runs the current recommended PaliGemma2 ViT HIP profile configuration:
# custom QKV/attention-output/MLP kernels, direct-QKV F32 attention, and cached
# rocBLAS attention QK/AV solution indices. Profiles are run sequentially
# because the iGPU is shared.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE="${PROBE:-${ROOT_DIR}/build/paligemma2_vit_hip_backend_probe}"
IMAGE="${IMAGE:-${ROOT_DIR}/paligemma/testdata/image.ppm}"
MODEL_224="${MODEL_224:-${ROOT_DIR}/build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs}"
MODEL_448="${MODEL_448:-${ROOT_DIR}/build/models/paligemma2-3b-mix-448-hf/paligemma2-3b-mix-448-sfp-from-hf.sbs}"
ATTENTION_CACHE="${ATTENTION_CACHE:-${ROOT_DIR}/build/paligemma2_vit_attention_solution_cache.txt}"
MLP_CACHE_448="${MLP_CACHE_448:-${ROOT_DIR}/build/paligemma2_vit_mlp_solution_cache_448.txt}"
SAMPLES="${SAMPLES:-5}"
WARMUP="${WARMUP:-2}"
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
if [[ ! -f "${ATTENTION_CACHE}" ]]; then
  echo "error: missing attention cache ${ATTENTION_CACHE}" >&2
  echo "hint: run scripts/build_paligemma2_vit_attention_solution_cache.sh first" >&2
  exit 1
fi
if [[ ! -f "${IMAGE}" ]]; then
  echo "error: missing image ${IMAGE}" >&2
  exit 1
fi

profile_one() {
  local label="$1"
  local weights="$2"
  shift 2
  local extra_args=("$@")
  if [[ ! -f "${weights}" ]]; then
    echo "skip ${label}: missing ${weights}" >&2
    return 0
  fi

  echo "== ${label} =="
  "${PROBE}" \
    --weights "${weights}" \
    --image_file "${IMAGE}" \
    --hip_schedule 0 \
    --hip_verbose 0 \
    --hip_image_tokens_return_profile 1 \
    --hip_samples "${SAMPLES}" \
    --hip_warmup "${WARMUP}" \
    --hip_iters "${ITERS}" \
    --hip_attention_pack_bf16 1 \
    --hip_attention_direct_qkv 1 \
    --hip_qkv_wmma2d 1 \
    --hip_attn_out_wmma2d 1 \
    --hip_mlp_up_wmma2d 1 \
    --hip_mlp_down_wmma8 1 \
    --hip_mlp_down_fused_residual 1 \
    --hip_attention_solution_cache "${ATTENTION_CACHE}" \
    "${extra_args[@]}"
}

profile_one "224" "${MODEL_224}"

cache_args_448=()
if [[ -f "${MLP_CACHE_448}" ]]; then
  cache_args_448=(--hip_mlp_solution_cache "${MLP_CACHE_448}")
fi
cache_args_448+=(--hip_mlp_down_wmma_waves 12)
profile_one "448" "${MODEL_448}" "${cache_args_448[@]}"
