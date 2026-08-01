#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/experiments/ihn_test_googlemap_320_to_256_eval128.yaml}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT="${CHECKPOINT:-}"
FIXED_LABELS="${FIXED_LABELS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_ROOT}"

cmd=(
  "${PYTHON_BIN}"
  training/test.py
  --config "${CONFIG_PATH}"
  --device "${DEVICE}"
)

if [[ -n "${CHECKPOINT}" ]]; then
  cmd+=(--checkpoint "${CHECKPOINT}")
fi

if [[ -n "${FIXED_LABELS}" ]]; then
  cmd+=(--fixed-labels "${FIXED_LABELS}")
fi

"${cmd[@]}" "$@"
