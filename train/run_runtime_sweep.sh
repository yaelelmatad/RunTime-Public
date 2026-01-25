#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for randomized hyperparameter sweeps.
#
# This uses `train/sweep_runtime_trainer.py` which:
# - generates N configs from a sweep spec
# - runs either one trial or all trials
#
# Usage (build configs only):
#   bash train/run_runtime_sweep.sh build
#
# Usage (run a single trial, index=2):
#   bash train/run_runtime_sweep.sh run_one 2
#
# Usage (run all trials):
#   bash train/run_runtime_sweep.sh run_all
#
# Override defaults:
#   OUT_DIR=/tmp/runtime_sweep BASE_CONFIG=train/runtime_trainer_config.yaml SWEEP_SPEC=train/runtime_trainer_random_sweep.yaml bash train/run_runtime_sweep.sh run_one 0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-run_one}"
INDEX="${2:-0}"

BASE_CONFIG="${BASE_CONFIG:-${ROOT_DIR}/train/runtime_trainer_config.yaml}"
SWEEP_SPEC="${SWEEP_SPEC:-${ROOT_DIR}/train/runtime_trainer_random_sweep.yaml}"
TRAINER_PY="${TRAINER_PY:-${ROOT_DIR}/train/Runtime_Trainer.py}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/train/sweeps}"

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

cd "${ROOT_DIR}/train"

CMD=( "${PY_BIN}" "${ROOT_DIR}/train/sweep_runtime_trainer.py"
  --base_config "${BASE_CONFIG}"
  --sweep_spec "${SWEEP_SPEC}"
  --trainer_py "${TRAINER_PY}"
  --out_dir "${OUT_DIR}"
  --mode "${MODE}"
)

if [[ "${MODE}" == "run_one" ]]; then
  CMD+=( --index "${INDEX}" )
fi

echo "[run_runtime_sweep] ${CMD[*]}"
exec "${CMD[@]}"


