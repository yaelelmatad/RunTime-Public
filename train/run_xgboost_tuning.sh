#!/usr/bin/env bash
set -euo pipefail

# Run XGBoost baseline training / tuning for RunTime (continuous-feature XGBoost baseline).
#
# Usage (pilot, fast):
#   bash train/run_xgboost_tuning.sh
#
# Usage (override defaults):
#   MAX_FILES=50 XGB_NUM_BOOST_ROUND=2000 XGB_EARLY_STOPPING_ROUNDS=80 SEED=42 bash train/run_xgboost_tuning.sh
#
# Enable random hyperparameter search (keeps features/artifacts the same; only tunes the model):
#   TUNE=1 N_TRIALS=25 MAX_FILES=10 bash train/run_xgboost_tuning.sh
#
# Full run (slow; loads all shards):
#   MAX_FILES=999999 bash train/run_xgboost_tuning.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MAX_FILES="${MAX_FILES:-50}"
XGB_NUM_BOOST_ROUND="${XGB_NUM_BOOST_ROUND:-2000}"
XGB_EARLY_STOPPING_ROUNDS="${XGB_EARLY_STOPPING_ROUNDS:-80}"
SEED="${SEED:-42}"
TUNE="${TUNE:-0}"
N_TRIALS="${N_TRIALS:-25}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/train/xgb_first${MAX_FILES}_continuous}"
if [[ "${TUNE}" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/train/xgb_tune_first${MAX_FILES}_continuous}"
fi

# Prefer full splits if present (RunTime-Full), otherwise fall back to the public sample shards.
SPLITS_GLOB_DEFAULT="${ROOT_DIR}/pipeline/training_splits/runners_split_*.pkl.gz"
if [[ ! -e "${ROOT_DIR}/pipeline/training_splits" ]]; then
  SPLITS_GLOB_DEFAULT="${ROOT_DIR}/data/samples/*.pkl.gz"
fi
SPLITS_GLOB="${SPLITS_GLOB:-${SPLITS_GLOB_DEFAULT}}"

SCRIPT="${ROOT_DIR}/train/Benchmark_Baselines.py"

echo "[run_xgboost_tuning] ROOT_DIR=${ROOT_DIR}"
echo "[run_xgboost_tuning] MAX_FILES=${MAX_FILES} XGB_NUM_BOOST_ROUND=${XGB_NUM_BOOST_ROUND} XGB_EARLY_STOPPING_ROUNDS=${XGB_EARLY_STOPPING_ROUNDS} SEED=${SEED} TUNE=${TUNE} N_TRIALS=${N_TRIALS}"
echo "[run_xgboost_tuning] splits_glob=${SPLITS_GLOB}"
echo "[run_xgboost_tuning] outputs will be written to: ${OUTPUT_DIR}"

cd "${ROOT_DIR}/train"

mkdir -p "${OUTPUT_DIR}"

# Prefer an activated venv if present; otherwise fall back to system python.
PY_BIN="${PY_BIN:-python}"
if command -v "${PY_BIN}" >/dev/null 2>&1; then
  :
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
else
  echo "ERROR: could not find python (set PY_BIN=... if needed)"
  exit 1
fi

CMD=( "${PY_BIN}" "${SCRIPT}"
  --output_dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --xgb_num_boost_round "${XGB_NUM_BOOST_ROUND}"
  --xgb_early_stopping_rounds "${XGB_EARLY_STOPPING_ROUNDS}"
  --splits_glob "${SPLITS_GLOB}"
  --max_files "${MAX_FILES}"
)

if [[ "${TUNE}" == "1" ]]; then
  CMD+=( --tune --n_trials "${N_TRIALS}" )
fi

exec "${CMD[@]}"


