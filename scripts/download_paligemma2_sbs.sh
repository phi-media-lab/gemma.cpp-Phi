#!/usr/bin/env bash
# Downloads pre-converted PaliGemma2 gemma.cpp SBS weights from Kaggle.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Kaggle is the shortest path when credentials and model license acceptance are
# already configured. The HF conversion script is the fallback when Kaggle direct
# SBS downloads are unavailable.
VENV_DIR="${ROOT_DIR}/build/weights_tools_venv"
OUT_DIR="${1:-${ROOT_DIR}/build/models/paligemma2-3b-mix-224}"
MODEL_INSTANCE_VERSION="${2:-google/paligemma-2/gemmaCpp/paligemma2-3b-mix-224/1}"

if [[ ! -x "${VENV_DIR}/bin/kaggle" ]]; then
  python3 -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip kaggle
fi

mkdir -p "${OUT_DIR}"

# `--untar` mirrors the layout expected by gemma.cpp examples: weights and
# tokenizer-like files land directly in OUT_DIR.
echo "Downloading ${MODEL_INSTANCE_VERSION}"
echo "Output directory: ${OUT_DIR}"
"${VENV_DIR}/bin/kaggle" models instances versions download \
  "${MODEL_INSTANCE_VERSION}" \
  --path "${OUT_DIR}" \
  --untar \
  --force

echo
echo "Downloaded files:"
find "${OUT_DIR}" -maxdepth 1 -type f \( -name '*.sbs' -o -name '*.model' -o -name '*.spm' \) -printf '%f\n' | sort
