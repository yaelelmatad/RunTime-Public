[![Work in progress](https://img.shields.io/badge/status-work_in_progress-orange)](#)
# Runtime (RunTime): Distributional Transformers for Irregular Event Sequences
[![DOI](https://zenodo.org/badge/1139424380.svg)](https://doi.org/10.5281/zenodo.18370743)

**Cite this work:** Yael S. Elmatad, *RunTime: Distributional Transformers for Irregular Event Sequences*, Zenodo (2026). https://doi.org/10.5281/zenodo.18370743

RunTime is a causal Transformer for **calibrated distributional forecasting on irregular event sequences** (TPP-aligned). Unlike standard approaches that rely on continuous embeddings or point estimates, RunTime combines:

- **Selective discretization as structural regularization** (environmental states are binned while time deltas stay continuous)
- **Gaussian-integrated soft targets** (precise label smoothing via error-function integration across bin boundaries)
- **Calibrated probability distributions** (Q-Q analysis yields a KS statistic D=0.025, i.e., ≤2.5 percentage-point deviation from perfect uniform percentiles)

This enables uncertainty-aware predictions while preserving interpretability through attention inspection. Every event block now emits the pace token before the time-delta tokens, so the model cannot peek at future cadence signals before predicting pace.

## Key Innovations

1. **Hybrid quantized-discrete grammar**: Environmental tokens (temperature, humidity, pace) are discretized to capture regime-specific behavior like trees, while inter-event gaps remain unquantized so attention stays elastic across irregular cadences; swapping the pace/time order prevents leakage.
2. **Gaussian-smoothed soft targets**: Instead of Chronos-style hard one-hot labels or uniform label smoothing, RunTime integrates a Gaussian kernel across each bin using the error function, preserving ordinality and enabling sub-bin interpolation.
3. **Calibrated distributional predictions**: The model predicts full PDFs, not just points. Quantile-quantile diagnostics show the predicted percentiles stay within 2.5 percentage points of the uniform CDF (KS D=0.025).
4. **Mechanistic interpretability**: Attention snapshots show time-delta tokens attracting dominant mass when uncertainty is high, providing interpretable insight into the learned rhythm.

## Why Discretization Over Continuous Embeddings?

Recent work (Gorishniy et al. 2021; Shwartz-Ziv & Armon 2022; Grinsztajn et al. 2022) shows that tabular Transformers consuming continuous embeddings still fall behind gradient-boosted trees because trees inherently perform implicit binning via splits, creating sharp regime boundaries that smooth networks average out. RunTime adopts explicit discretization but pairs it with:

Context windows are capped at 327 tokens so strides remain aligned without leaking future cadence.

- **Balanced quantization** (bins hold roughly uniform probability mass, not uniform width)
- **Gaussian-integrated soft targets** (smooth gradients despite the discrete vocabulary)
- **Calibration-first training** (distributional fidelity takes priority over single-number accuracy)

This lets RunTime model regime-specific behavior like trees while keeping the Transformer differentiable and interpretable.

**GitHub repo:** [yaelelmatad/RunTime-Public](https://github.com/yaelelmatad/RunTime-Public)

If you want the full writeup (with figures): see `paper/RunTime_Tabular_Main.tex` and the rendered PDF at `paper/RunTime_Tabular_Main.pdf`.

## Status: Work in Progress

Code in this repo is being actively developed.  It may not run out of the box but it is being shown here for illustrative purposes.  This will be addressed when the work is in a more final state, but the patterns remain largely unchanged.

## Current results (final; hyperparameter tuning completed)

✅ **Note**: These reported values come from the final RunTime and baseline configurations; we are no longer actively tuning the reported models.

### Baseline comparison (n=200,000 race predictions)

| Method | Median MAE (seconds/mile) | Status |
|--------|---------------------------:|--------|
| Naive Mean | 52.72 | ✅ Final |
| Last Pace | 52.72* | ✅ Final |
| Riegel Formula | 49.74 | ✅ Final |
| XGBoost (tuned) | 40.31 | ✅ Final |
| RunTime (median, σ=3) | **35.94** | ✅ Final |

_\*Last Pace uses the previous pace from the final shuffle; we report the same MAE as the Naive Mean baseline for consistency with the ablation sweep._

## What’s in this repo

### Core training + evaluation (`train/`)

- `train/runtime_trainer.py`: Train the RunTime Transformer from a YAML config (supports CUDA / Apple MPS / CPU).
- `train/runtime_trainer_config.yaml`: Default training config for this standalone repo (points at the included sample shards).
- `train/benchmark_baselines.py`: Baselines on the same serialized dataset shards (naive mean, last-pace, and XGBoost).
- `train/Inspect_Model_Outputs.ipynb`: Notebook used to compute aggregate metrics / visualizations from saved predictions.
- `train/Inspect_Model_Activations.ipynb`: Attention/activation inspection + figure export helpers.
- `train/setup_cloud.sh`: Convenience setup script intended for fresh GPU machines.

### Data artifacts (`data/`)

- `data/samples/runners_split_000.pkl.gz`, `data/samples/runners_split_001.pkl.gz`: **Small sample shards** of the final serialized training format (enough to run the trainer and baselines end-to-end).
- `data/pace_lookup.pickle`: Pace-bin definitions/statistics used by the trainer for discretization + decoding.

### Data engineering workflow (`pipeline/`)

The `pipeline/` directory is a step-by-step notebook workflow that transforms raw race results into the serialized “RunTime grammar” shards consumed by training. See `pipeline/Workflow_Overview.md`.

Practical note: to prevent abuse (e.g., automated scraping / bulk pulling of the underlying raw results), **not all data-acquisition and raw-data retrieval pipeline components are included**. Some parts of the original acquisition/enrichment also depend on non-public sources and/or third-party APIs. This repo is set up to be runnable and inspectable using the included **sample shards** in `data/samples/`.

If you’re interested in reproducing the full dataset or accessing raw data, please reach out to the authors/maintainers and we can share additional details as appropriate.

Included notebooks:

- `pipeline/01_Data_Acquisition.ipynb`
- `pipeline/02_Weather_Extraction.ipynb`
- `pipeline/03_Runner_Career_Grouping.ipynb`
- `pipeline/04_Weather_Grammar_Creation.ipynb`
- `pipeline/05_Distance_Grammar_Creation.ipynb`
- `pipeline/06_Pace_Grammar_Creation.ipynb`
- `pipeline/07_Unified_Grammar_Integration.ipynb`
- `pipeline/08_Final_Dataset_Generation.ipynb`
- `pipeline/09_Hydration_and_Tokenization.ipynb`

Note: the *conceptual* order is “hydration/tokenization → final dataset sharding”; see `pipeline/Workflow_Overview.md` for the intended flow.

### Figures + paper artifacts

- `figures/`: Exported plots referenced in the paper / notebooks.
- `paper/`: LaTeX source for `RunTime_Tabular_Main.pdf` plus bibliography and figure assets.

## Quickstart (runs on the included sample data)

### 1) Install deps

```bash
python -m pip install -r requirements.txt
```

### 2) Run baselines (naive / last-pace / XGBoost)

`benchmark_baselines.py` takes one or more `*.pkl.gz` shard paths and writes artifacts to an explicit output directory. A convenience wrapper is included as `train/run_xgboost_tuning.sh`.

```bash
bash train/run_xgboost_tuning.sh
```

Artifacts produced (under `train/xgb_*` by default): `baseline_results.json`, `xgboost_model.json`, `xgboost_feature_columns.pickle`, plus feature-importance CSVs.

To enable randomized hyperparameter search:

```bash
TUNE=1 N_TRIALS=25 MAX_FILES=10 bash train/run_xgboost_tuning.sh
```

### 3) Train RunTime (multiple configs)

RunTime has three supported configs:

1. **Adaptive sigma default** (`runtime_trainer_adaptive_sigma.yaml`) – the main reported model.
2. **Time-token ablation** (`runtime_trainer_time_token_ablation.yaml`) – drops the time token and keeps only the final age marker.
3. **Shuffled ablation** (`runtime_trainer_shuffled_ablation.yaml`) – drops the time token (like the time-token ablation) but feeds the remaining stride blocks in randomized order to test order sensitivity.

Each variant has its own trainer entry point.  

```bash
# adaptive sigma (main experiment)
bash train/run_runtime_train.sh

# time-token ablation (runs the specialized ablation trainer)
python train/runtime_trainer_ablation.py --config train/runtime_trainer_time_token_ablation.yaml

# shuffled ablation (uses its own trainer)
python train/runtime_trainer_ablation_shuffled.py --config train/runtime_trainer_shuffled_ablation.yaml
```

Checkpoints are saved under `train/<save_dir>/<run_name>/` as configured in each YAML (defaults to `checkpoints_clean_prod/Production_Scale_v2_HighCap/` for the adaptive run). If any config enables `use_wandb: true`, set `WANDB_API_KEY` before running so the logs reach WandB (otherwise the key can stay local to the YAML).

### 4) Evaluate predictions

Use `train/evaluate_models.py` (or the parallel-aware `train/evaluate_models_parallel.py`) to load saved checkpoints, replay inference, and compute MAE / calibration metrics. Both scripts (and the `evaluate/` notebooks) rely on the shared inference library `train/runtime_inference.py`, which exposes the `RuntimeModelInference` helper, split loaders, and calibration utilities.

```bash
# use a config file that lists checkpoints / splits
python train/evaluate_models.py --config evaluate/eval_config.yaml

# or run with on-the-fly arguments
python train/evaluate_models.py --input-glob "./data/samples/*.pkl.gz" \
    --models adaptive:train/checkpoints_clean_prod/Production_Scale_v2_HighCap_Corrected/checkpoint.pt \
    --num-examples 5000 \
    --output evaluate/results_adaptive.pickle
```

When you open the notebooks in `evaluate/`, launch Jupyter from the repo root so that `train/` is already on `sys.path` (they append `Path(__file__).resolve().parents[1]` to `sys.path` as a fallback). This makes `from runtime_inference import ...` work consistently across scripts, notebooks, and CLI tools.

```bash
# use a config file that lists checkpoints / splits
python train/evaluate_models.py --config evaluate/eval_config.yaml

# or run with on-the-fly arguments
python train/evaluate_models.py --input-glob "./data/samples/*.pkl.gz" \
    --models adaptive:train/checkpoints_clean_prod/Production_Scale_v2_HighCap_Corrected/checkpoint.pt \
    --num-examples 5000 \
    --output evaluate/results_adaptive.pickle
```

The `evaluate/` folder now holds notebooks revisiting the calibration sweep, activation inspection, and runner distribution plots. E.g., open `evaluate/Plot_Model_Results.ipynb` after running the evaluation script to generate the dashboards used in the writeup.

### Running on Lambda (GPU quickstart)

On a fresh Ubuntu GPU machine:

```bash
git clone git@github.com:yaelelmatad/RunTime-Public.git
cd RunTime-Public

# Create the venv + install deps
bash train/setup_cloud.sh

# The setup script installs cuda-enabled PyTorch and other packages into ~/.local; add it to PATH:
export PATH="$HOME/.local/bin:$PATH"

# Optional: set WANDB before training
export WANDB_API_KEY="..."

# Verify the machine via the Lambda helper (checks CUDA + data shards):
bash train/run-scripts/setup_lambda.sh

# Run baselines / trainer / evaluation as above:
bash train/run_xgboost_tuning.sh
CONFIG=train/runtime_trainer_adaptive_sigma.yaml bash train/run_runtime_train.sh
python train/evaluate_models.py --config evaluate/eval_config.yaml
```

`train/run-scripts/setup_lambda.sh` already installs system packages (python3-dev, pip) and user-level dependencies like torch, wandb, scipy, and optuna, so rerunning it after reboot ensures the Lambda env stays healthy.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total individuals | 600K |
| Total training examples | 5M |
| Average races per runner | ≈8 |
| Training set | 270K individuals (2.25M examples) |
| Validation set | 30K individuals (250K examples) |
| Test set | 60K individuals (500K predictions) |

## Performance summary

Filtered run-time metrics appear in `paper/RunTime_Tabular_Main.pdf`; consult that document for the full MAE table.

## License

- **Code**: Apache License 2.0 (see `LICENSE` and `NOTICE`)
- **Documentation / writeup** (including `paper/RunTime_Tabular_Main.tex` and `paper/RunTime_Tabular_Main.pdf`): Creative Commons Attribution 4.0 International (see `LICENSE-CC-BY-4.0`)

