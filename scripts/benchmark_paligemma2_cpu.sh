#!/usr/bin/env bash
# Runs repeatable CPU baselines for the local PaliGemma2 3B 224 SBS model.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Positional arguments keep the common path short while still allowing the same
# script to benchmark a different SBS/image/prompt. Environment variables control
# repeated-run behavior so command lines stay comparable in benchmark logs.
WEIGHTS="${1:-${ROOT_DIR}/build/models/paligemma2-3b-mix-224-hf/paligemma2-3b-mix-224-sfp-from-hf.sbs}"
IMAGE="${2:-${ROOT_DIR}/paligemma/testdata/image.ppm}"
PROMPT="${3:-Describe the image.}"
TOKENS="${TOKENS:-32}"
SAMPLES="${SAMPLES:-3}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/build/paligemma2_e2e_logs}"
EXTRA_GEMMA_ARGS="${EXTRA_GEMMA_ARGS:-}"

# Split optional gemma flags once so callers can pass experiments such as
# EXTRA_GEMMA_ARGS="--force_flash_attention 1 --image_skip_lps 4 ...".
read -r -a EXTRA_ARGS <<<"${EXTRA_GEMMA_ARGS}"

mkdir -p "${LOG_DIR}"

run_case() {
  local name="$1"
  shift
  echo "== ${name} =="
  for i in $(seq 1 "${SAMPLES}"); do
    local log="${LOG_DIR}/${name}_${TOKENS}_${i}.log"

    # The prefix command is either `env` for the default scheduler or a taskset
    # command for a fixed CPU set. `/usr/bin/time` writes wall time into the same
    # log as gemma's internal timing so each sample is self-contained.
    /usr/bin/time -f 'WALL_SECONDS %e' "$@" "${ROOT_DIR}/build/gemma" \
      --weights "${WEIGHTS}" \
      --image_file "${IMAGE}" \
      --prompt "${PROMPT}" \
      --max_generated_tokens "${TOKENS}" \
      --top_k 1 \
      --deterministic 1 \
      --verbosity 1 \
      "${EXTRA_ARGS[@]}" \
      >"${log}" 2>&1

    # Print only the benchmark lines needed for quick median calculations while
    # preserving the full raw output in LOG_DIR.
    rg 'Image token|Prefill:|Generate:|WALL_SECONDS' "${log}"
  done
}

# Compare the scheduler policies that mattered on this heterogeneous Ryzen AI
# machine: default all-core behavior, physical cores only, the 8 lower-frequency
# cores, and the 4 high-frequency cores.
run_case default env
run_case physical_0_11 taskset -c 0-11
run_case slow_4_11 taskset -c 4-11
run_case fast_0_3 taskset -c 0-3
