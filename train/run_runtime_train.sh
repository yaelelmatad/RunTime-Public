#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for training RunTime locally or on a remote box (e.g., Lambda).
#
# Usage:
#   bash train/run_runtime_train.sh
#
# Override defaults:
#   CONFIG=train/runtime_trainer_config.yaml WANDB_API_KEY=... bash train/run_runtime_train.sh
#
# Notes:
# - This wrapper does not edit your YAML. For full-data runs, point `data.splits_dir` at
#   `../pipeline/training_splits` (RunTime-Full) or your own shard directory.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/train/runtime_trainer_config.yaml}"

PY_BIN="${PY_BIN:-python}"
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PY_BIN="${ROOT_DIR}/.venv/bin/python"
elif command -v "${PY_BIN}" >/dev/null 2>&1; then
  :
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
else
  echo "ERROR: could not find python (set PY_BIN=... if needed)"
  exit 1
fi

echo "[run_runtime_train] ROOT_DIR=${ROOT_DIR}"
echo "[run_runtime_train] CONFIG=${CONFIG}"
echo "[run_runtime_train] PY_BIN=${PY_BIN}"

exec "${PY_BIN}" "${ROOT_DIR}/train/Runtime_Trainer.py" --config "${CONFIG}"


