#!/usr/bin/env bash
# Downloads PaliGemma2 safetensors from Hugging Face and converts them to SBS.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Defaults target the smallest useful PaliGemma2 model. Callers can pass a 448 or
# 10B HF repo, and the case block below maps the repo name to gemma.cpp's model
# specifier expected by convert_from_safetensors.py.
HF_REPO="${1:-google/paligemma2-3b-mix-224}"
REPO_SLUG="${HF_REPO##*/}"
OUT_DIR="${2:-${ROOT_DIR}/build/models/${REPO_SLUG}-hf}"
MODEL_SPECIFIER="${3:-}"
TOKENIZER_REPO="${4:-google/paligemma-3b-mix-224}"
TOKENIZER_SLUG="${TOKENIZER_REPO##*/}"

# gemma.cpp conversion wants an explicit model specifier, but HF repo names are
# regular enough that we can infer the supported PaliGemma2 variants. If this
# inference fails, force the user to pass the specifier explicitly rather than
# silently producing weights for the wrong architecture.
if [[ -z "${MODEL_SPECIFIER}" ]]; then
  case "${REPO_SLUG}" in
    *paligemma2-3b*448*) MODEL_SPECIFIER="paligemma2-3b-448-sfp" ;;
    *paligemma2-3b*224*) MODEL_SPECIFIER="paligemma2-3b-224-sfp" ;;
    *paligemma2-10b*448*) MODEL_SPECIFIER="paligemma2-10b-448-sfp" ;;
    *paligemma2-10b*224*) MODEL_SPECIFIER="paligemma2-10b-224-sfp" ;;
    *)
      echo "error: could not infer model specifier from ${HF_REPO}" >&2
      echo "usage: $0 <hf_repo> <out_dir> <model_specifier> [tokenizer_repo]" >&2
      exit 1
      ;;
  esac
fi

SBS_FILE="${SBS_FILE:-${OUT_DIR}/${REPO_SLUG}-sfp-from-hf.sbs}"

# Use two virtual environments because the Hugging Face CLI can run on the
# system Python, while the Bazel-built pybind extensions are tied to the
# rules_python Python 3.11 interpreter discovered below.
HF_VENV="${ROOT_DIR}/build/weights_tools_venv"
PY311_VENV="${ROOT_DIR}/build/weights_tools_py311_venv"
HF_DIR="${ROOT_DIR}/build/models/hf-${REPO_SLUG}"
TOKENIZER_DIR="${ROOT_DIR}/build/models/hf-tokenizers/${TOKENIZER_SLUG}"

if [[ ! -x "${HF_VENV}/bin/hf" ]]; then
  python3 -m venv "${HF_VENV}"
  "${HF_VENV}/bin/python" -m pip install --upgrade pip huggingface_hub
fi

mkdir -p "${HF_DIR}" "${TOKENIZER_DIR}" "${OUT_DIR}"

echo "Downloading safetensors from ${HF_REPO}"
"${HF_VENV}/bin/hf" download "${HF_REPO}" \
  --include 'model*.safetensors*' \
  --local-dir "${HF_DIR}"

echo "Downloading SentencePiece tokenizer from ${TOKENIZER_REPO}"
"${HF_VENV}/bin/hf" download "${TOKENIZER_REPO}" tokenizer.model \
  --local-dir "${TOKENIZER_DIR}"

# Prefer bazelisk when available so the repository's Bazel version is honored.
# The conversion needs local pybind extensions from //compression/python and
# //python:configs.
BAZEL="${BAZEL:-}"
if [[ -z "${BAZEL}" ]]; then
  if command -v bazelisk >/dev/null 2>&1; then
    BAZEL=bazelisk
  elif command -v bazel >/dev/null 2>&1; then
    BAZEL=bazel
  else
    echo "error: install bazelisk or bazel before running conversion" >&2
    exit 1
  fi
fi

echo "Building Python conversion extensions"
(cd "${ROOT_DIR}" && "${BAZEL}" build //compression/python:compression //python:configs)

# The pybind extensions are built for Bazel's Python 3.11 toolchain. Running the
# converter with a different interpreter can fail to import those extensions, so
# locate that interpreter and build a matching conversion venv.
OUTPUT_BASE="$(cd "${ROOT_DIR}" && "${BAZEL}" info output_base)"
PY311="$(
  find "${OUTPUT_BASE}/external" -path '*/bin/python3.11' -type f -perm -111 \
    | head -1
)"
if [[ -z "${PY311}" ]]; then
  echo "error: could not find rules_python Python 3.11 interpreter" >&2
  exit 1
fi

if [[ ! -x "${PY311_VENV}/bin/python" ]]; then
  "${PY311}" -m venv "${PY311_VENV}"
fi
"${PY311_VENV}/bin/python" -m pip install --upgrade \
  pip numpy safetensors absl-py sentencepiece
"${PY311_VENV}/bin/python" -m pip install --upgrade \
  torch --index-url https://download.pytorch.org/whl/cpu

# The converter reads the HF sharded safetensors index plus a SentencePiece
# tokenizer and writes both the SBS weight file and metadata CSV expected by
# gemma.cpp.
echo "Converting ${HF_REPO} to SBS"
PYTHONPATH="${ROOT_DIR}/bazel-bin:${ROOT_DIR}" \
  "${PY311_VENV}/bin/python" "${ROOT_DIR}/python/convert_from_safetensors.py" \
  --model_specifier "${MODEL_SPECIFIER}" \
  --load_path "${HF_DIR}/model.safetensors.index.json" \
  --tokenizer_file "${TOKENIZER_DIR}/tokenizer.model" \
  --metadata_file "${OUT_DIR}/metadata.csv" \
  --sbs_file "${SBS_FILE}"

echo
echo "Created:"
ls -lh "${SBS_FILE}" "${OUT_DIR}/metadata.csv"
