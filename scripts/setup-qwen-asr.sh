#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/vendor/qwen-asr"
MODEL_DIR="${ROOT_DIR}/models/qwen3-asr-0.6b"
QWEN_ASR_REF="b00b789b17051aea61e9717458171100662318a4"

mkdir -p "${ROOT_DIR}/vendor" "${ROOT_DIR}/models"

if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
  git clone https://github.com/antirez/qwen-asr.git "${VENDOR_DIR}"
fi

git -C "${VENDOR_DIR}" fetch --tags origin
git -C "${VENDOR_DIR}" checkout "${QWEN_ASR_REF}"
make -C "${VENDOR_DIR}" blas
bash "${VENDOR_DIR}/download_model.sh" --model small --dir "${MODEL_DIR}"

echo "qwen-asr ready at ${VENDOR_DIR}"
echo "model ready at ${MODEL_DIR}"
