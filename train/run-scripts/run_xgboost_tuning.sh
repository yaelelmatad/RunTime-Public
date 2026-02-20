#!/usr/bin/env bash
set -euo pipefail

# Run XGBoost baseline training / tuning for RunTime (continuous-feature XGBoost baseline).
#
# Usage (pilot, fast):
#   bash Runtime/train/run-scripts/run_xgboost_tuning.sh
#
# Usage (override defaults):
#   MAX_FILES=50 XGB_NUM_BOOST_ROUND=2000 XGB_EARLY_STOPPING_ROUNDS=80 SEED=42 bash Runtime/train/run-scripts/run_xgboost_tuning.sh
#
# Enable random hyperparameter search (keeps features/artifacts the same; only tunes the model):
#   TUNE=1 N_TRIALS=25 MAX_FILES=10 bash Runtime/train/run-scripts/run_xgboost_tuning.sh
#
# Full run (slow; loads all shards):
#   MAX_FILES=999999 bash Runtime/train/run-scripts/run_xgboost_tuning.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PY="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "ERROR: expected venv python at: ${VENV_PY}"
  echo "Create/activate your venv first (e.g., python -m venv .venv && . .venv/bin/activate)."
  exit 1
fi

MAX_FILES="${MAX_FILES:-50}"
XGB_NUM_BOOST_ROUND="${XGB_NUM_BOOST_ROUND:-2000}"
XGB_EARLY_STOPPING_ROUNDS="${XGB_EARLY_STOPPING_ROUNDS:-80}"
SEED="${SEED:-42}"
TUNE="${TUNE:-0}"
N_TRIALS="${N_TRIALS:-25}"

TUNE_NORM="$(printf '%s' "${TUNE}" | tr '[:upper:]' '[:lower:]')"
TUNE_ENABLED=0
if [[ "${TUNE_NORM}" == "1" || "${TUNE_NORM}" == "true" || "${TUNE_NORM}" == "yes" ]]; then
  TUNE_ENABLED=1
fi

# Pick a sensible default output dir, but allow callers to override OUTPUT_DIR explicitly.
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  if [[ "${TUNE_ENABLED}" == "1" ]]; then
    OUTPUT_DIR="${ROOT_DIR}/Runtime/train/xgb_tune_first_${MAX_FILES}_continuous"
  else
    OUTPUT_DIR="${ROOT_DIR}/Runtime/train/xgb_first_${MAX_FILES}_continuous"
  fi
fi

SPLITS_GLOB="${ROOT_DIR}/Runtime/pipeline/training_splits/runners_split_*.pkl.gz"
SCRIPT="${ROOT_DIR}/Runtime/train/benchmark_baselines.py"

echo "[run_xgboost_tuning] ROOT_DIR=${ROOT_DIR}"
echo "[run_xgboost_tuning] MAX_FILES=${MAX_FILES} XGB_NUM_BOOST_ROUND=${XGB_NUM_BOOST_ROUND} XGB_EARLY_STOPPING_ROUNDS=${XGB_EARLY_STOPPING_ROUNDS} SEED=${SEED} TUNE=${TUNE} (enabled=${TUNE_ENABLED}) N_TRIALS=${N_TRIALS}"
echo "[run_xgboost_tuning] splits_glob=${SPLITS_GLOB}"
echo "[run_xgboost_tuning] outputs will be written to: ${OUTPUT_DIR}"

cd "${ROOT_DIR}/Runtime/train"

mkdir -p "${OUTPUT_DIR}"

CMD=( "${VENV_PY}" "${SCRIPT}"
  --output_dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --xgb_num_boost_round "${XGB_NUM_BOOST_ROUND}"
  --xgb_early_stopping_rounds "${XGB_EARLY_STOPPING_ROUNDS}"
  --splits_glob "${SPLITS_GLOB}"
  --max_files "${MAX_FILES}"
)

if [[ "${TUNE_ENABLED}" == "1" ]]; then
  CMD+=( --tune --n_trials "${N_TRIALS}" )
fi

exec "${CMD[@]}"


