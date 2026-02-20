#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper to run the tuned XGBoost baseline on the first 50 shards
# with sensible prepopulated defaults.
#
# From repo root:
#   bash Runtime/train/run-scripts/run_xgboost_tuning_50_local.sh
#
# You can still override any variable inline, e.g.:
#   N_TRIALS=50 XGB_NUM_BOOST_ROUND=3000 bash Runtime/train/run-scripts/run_xgboost_tuning_50_local.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Activate venv if it exists (recommended).
if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

# Defaults (override by setting env vars before calling this script)
export TUNE="${TUNE:-1}"
export N_TRIALS="${N_TRIALS:-25}"
export MAX_FILES="${MAX_FILES:-50}"
export XGB_NUM_BOOST_ROUND="${XGB_NUM_BOOST_ROUND:-2000}"
export XGB_EARLY_STOPPING_ROUNDS="${XGB_EARLY_STOPPING_ROUNDS:-80}"
export SEED="${SEED:-42}"

# Keep outputs separate from non-tuned runs by default
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/Runtime/train/xgb_tune_first_50_continuous}"

echo "[run_xgboost_tuning_50_local] ROOT_DIR=${ROOT_DIR}"
echo "[run_xgboost_tuning_50_local] TUNE=${TUNE} N_TRIALS=${N_TRIALS} MAX_FILES=${MAX_FILES} XGB_NUM_BOOST_ROUND=${XGB_NUM_BOOST_ROUND} XGB_EARLY_STOPPING_ROUNDS=${XGB_EARLY_STOPPING_ROUNDS} SEED=${SEED}"
echo "[run_xgboost_tuning_50_local] OUTPUT_DIR=${OUTPUT_DIR}"

exec bash "${ROOT_DIR}/Runtime/train/run-scripts/run_xgboost_tuning.sh"


