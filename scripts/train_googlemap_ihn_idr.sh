#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/experiments/ihn_idr_googlemap_320_to_256.yaml}"
DEVICE="${DEVICE:-cuda:0}"
RESUME="${RESUME:-}"
RESUME_MODEL_ONLY="${RESUME_MODEL_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_ROOT}"

cmd=(
  "${PYTHON_BIN}"
  training/train.py
  --config "${CONFIG_PATH}"
  --device "${DEVICE}"
)

if [[ -n "${RESUME}" ]]; then
  cmd+=(--resume "${RESUME}")
fi

if [[ "${RESUME_MODEL_ONLY}" == "1" ]]; then
  cmd+=(--resume_model_only)
fi

"${cmd[@]}" "$@"
