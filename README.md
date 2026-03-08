[![Stable](https://img.shields.io/badge/status-stable-green)](#)
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

## Status

Code in this repo runs end-to-end on the included sample shards (tested on CUDA, Apple MPS, and CPU). The architecture, training loop, and evaluation pipeline are stable; incremental improvements may still land.

## Current results (final; hyperparameter tuning completed)

✅ **Note**: These reported values come from the final RunTime and baseline configurations; we are no longer actively tuning the reported models.

### Benchmark comparison (converged sweep)

| Model | Mean MAE | Median MAE | Mode MAE | Median RMSE |
| --- | ---: | ---: | ---: | ---: |
| RunTime (σ=3) | **36.54** | **35.94** | **38.50** | **71.83** |
| XGBoost (tuned) | 40.31 | 40.31 | 40.31 | 73.15 |
| Naive mean | 52.72 | 52.72 | 52.72 | 88.16 |
| Riegel formula | 49.74 | 49.74 | 49.74 | 94.71 |

These numbers mirror Table 1 in `paper/RunTime_Tabular_Main.pdf`, reporting the converged σ=3 sweep’s mean/median/mode MAE (plus median RMSE) alongside the classical baselines for a consistent comparison.
## What’s in this repo

### Core training + evaluation (`train/`)

- `train/runtime_trainer.py`: Main RunTime Transformer trainer with adaptive sigma (YAML-configured; supports CUDA / Apple MPS / CPU).
- `train/runtime_trainer_ablation.py`: Time-token ablation trainer — drops week-delta tokens and keeps only the final age marker.
- `train/runtime_trainer_ablation_shuffled.py`: Shuffled ablation trainer — same as the time-token ablation but feeds stride blocks in randomized order.
- `train/benchmark_baselines.py`: Baselines on the same serialized dataset shards (naive mean, last-pace, and XGBoost with optional hyperparameter tuning).
- `train/evaluate_models.py`: Load saved checkpoints, replay inference, and compute MAE / calibration metrics.
- `train/evaluate_models_parallel.py`: Parallel-GPU version of `evaluate_models.py`.
- `train/runtime_inference.py`: Shared inference library used by evaluation scripts and notebooks (`RuntimeModelInference`, split loaders, calibration utilities).
- `train/load_raw_predictions.py`: Helper to load and inspect saved raw prediction pickle files.
- `train/run_runtime_train.sh`: Convenience shell wrapper for launching training.
- `train/run-scripts/`: Additional helper scripts for cloud setup, XGBoost tuning, and checkpoint management.

### Configs (`train/*.yaml`)

- `train/runtime_trainer_adaptive_sigma.yaml`: Main model config — adaptive sigma smoothing (paper architecture).
- `train/runtime_trainer_config.yaml`: Alternative config using fixed sigma smoothing.
- `train/runtime_trainer_time_token_ablation.yaml`: Time-token ablation config.
- `train/runtime_trainer_shuffled_ablation.yaml`: Shuffled ablation config.
- `train/evaluation_config_local.yaml`: Example evaluation config for local runs.
- `train/evaluation_config_cluster.yaml`: Example evaluation config for GPU cluster runs.

### Data artifacts (`data/`)

- `data/samples/runners_split_000.pkl.gz`, `data/samples/runners_split_001.pkl.gz`: **Small sample shards** of the final serialized training format (enough to run the trainer and baselines end-to-end).
- `data/pace_lookup.pickle`: Pace-bin definitions/statistics used by the trainer for discretization + decoding.

### Data engineering workflow (`pipeline/`)

The `pipeline/` directory is a step-by-step notebook workflow that transforms raw race results into the serialized “RunTime grammar” shards consumed by training. See `pipeline/Workflow_Overview.md`.

Practical note: to prevent abuse (e.g., automated scraping / bulk pulling of the underlying raw results), **not all data-acquisition and raw-data retrieval pipeline components are included**. Some parts of the original acquisition/enrichment also depend on non-public sources and/or third-party APIs. This repo is set up to be runnable and inspectable using the included **sample shards** in `data/samples/`.

If you’re interested in reproducing the full dataset or accessing raw data, please reach out to the authors/maintainers and we can share additional details as appropriate.

Pipeline notebooks (Stage 01 is intentionally excluded from the public repo; see note above):

- `pipeline/02_Weather_Extraction.ipynb`
- `pipeline/03_Runner_Career_Grouping.ipynb`
- `pipeline/04_Weather_Grammar_Creation.ipynb`
- `pipeline/05_Distance_Grammar_Creation.ipynb`
- `pipeline/06_Pace_Grammar_Creation.ipynb`
- `pipeline/07_Unified_Grammar_Integration.ipynb`
- `pipeline/08_Final_Dataset_Generation.ipynb`
- `pipeline/09_Hydration_and_Tokenization.ipynb`

Note: the *conceptual* order is “hydration/tokenization → final dataset sharding”; see `pipeline/Workflow_Overview.md` for the intended flow.

### Evaluation notebooks (`evaluate/`)

- `evaluate/Examine_Distribution_Quantile_Predictions.ipynb`: Load a trained checkpoint, replay inference on held-out splits, and inspect predicted probability distributions — quantile calibration (Q-Q), per-bin mass, and decile-level diagnostics.
- `evaluate/Example_Runtime_Inference.ipynb`: Minimal end-to-end example of loading a checkpoint and running inference on a few examples.
- `evaluate/Inspect_Model_Outputs.ipynb`: Deep inspection of model outputs including XGBoost comparison and per-example prediction breakdowns.
- `evaluate/Inspect_Model_Activations.ipynb`: Visualize attention weights and intermediate activations to understand what the model focuses on.
- `evaluate/Plot_Model_Results.ipynb`: Generate the paper's main result plots (MAE curves, calibration figures).
- `evaluate/Example_Runner_Trajectories.ipynb`: Plot individual runner career trajectories alongside model predictions.

Launch Jupyter from the repo root so `train/` is on `sys.path` (notebooks append the parent directory as a fallback).

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

Checkpoints are saved under `<save_dir>/<run_name>/` as configured in each YAML (defaults to `checkpoints_clean_prod/runtime-adaptive-sigma/`). If any config enables `use_wandb: true`, set `WANDB_API_KEY` before running so the logs reach WandB.

### 4) Evaluate predictions

Use `train/evaluate_models.py` (or the parallel-aware `train/evaluate_models_parallel.py`) to load saved checkpoints, replay inference, and compute MAE / calibration metrics. Both scripts (and the `evaluate/` notebooks) rely on the shared inference library `train/runtime_inference.py`, which exposes the `RuntimeModelInference` helper, split loaders, and calibration utilities.

```bash
# use a config file that lists checkpoints and data splits
python train/evaluate_models.py --config train/evaluation_config_local.yaml
```

See `train/evaluation_config_local.yaml` and `train/evaluation_config_cluster.yaml` for example configs that specify model checkpoints, data glob patterns, and evaluation parameters.

To inspect saved raw predictions after evaluation:

```bash
python train/load_raw_predictions.py path/to/model_name_raw_predictions.pickle
```

When you open the notebooks in `evaluate/`, launch Jupyter from the repo root so that `train/` is already on `sys.path` (they append the parent directory as a fallback). This makes `from runtime_inference import ...` work consistently across scripts, notebooks, and CLI tools.

### Running on a cloud GPU (Lambda, etc.)

On a fresh Ubuntu GPU machine:

```bash
git clone git@github.com:yaelelmatad/RunTime-Public.git
cd RunTime-Public

# Create the venv + install deps (detects CUDA automatically)
bash train/setup_cloud.sh
source .venv/bin/activate

# Optional: set WANDB before training
export WANDB_API_KEY="..."

# Run baselines / trainer / evaluation as above:
bash train/run_xgboost_tuning.sh
CONFIG=train/runtime_trainer_adaptive_sigma.yaml bash train/run_runtime_train.sh
python train/evaluate_models.py --config train/evaluation_config_cluster.yaml
```

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

