# Runtime (RunTime): Distributional Transformers for Irregular Event Sequences
[![DOI](https://zenodo.org/badge/1139424380.svg)](https://doi.org/10.5281/zenodo.18370743)

**Cite this work:** Yael S. Elmatad, *RunTime: Distributional Transformers for Irregular Event Sequences*, Zenodo (2026). https://doi.org/10.5281/zenodo.18370743

This repo contains **RunTime**: a causal Transformer approach for **distributional regression on irregular event sequences** (TPP-aligned). It represents each event as a fixed-stride “grammar” block, treats **time deltas as tokens**, and trains with **Gaussian-integrated soft targets** so the model predicts a full probability distribution (PDF) over outcomes rather than only a point estimate.

**GitHub repo:** [yaelelmatad/RunTime-Public](https://github.com/yaelelmatad/RunTime-Public)

If you want the full writeup (with figures): see [Technical_Details.md](./Technical_Details.md) (or the rendered PDF: [Technical_Details.pdf](./Technical_Details.pdf)).

## Status: Work in Progress

Code in this repo is being actively developed.  It may not run out of the box but it is being shown here for illustrative purposes.  This will be addressed when the work is in a more final state, but the patterns remain largely unchanged.

## Current results (preliminary — hyperparameter tuning in progress)

⚠️ **Note**: These results use current hyperparameter configurations. We are currently conducting systematic hyperparameter optimization for both RunTime method. Updated results will be posted soon.

### Baseline comparison (n=200,000 race predictions)

| Method | MAE (seconds/mile) | Status |
|--------|-------------------:|--------|
| Naive Mean | 54.19 | ✅ Final |
| Last Pace | 61.31 | ✅ Final |
| Riegel Formula | 50.76 | ✅ Final |
| XGBoost | 40.94 |   ✅ Final  |
| RunTime (median) | **37.10** | ⚠️ Tuning in progress (single untuned run; better sweeps forthcoming - improvements expected) |

## What’s in this repo

### Core training + evaluation (`train/`)

- `train/Runtime_Trainer.py`: Train the RunTime Transformer from a YAML config (supports CUDA / Apple MPS / CPU).
- `train/runtime_trainer_config.yaml`: Default training config for this standalone repo (points at the included sample shards).
- `train/Benchmark_Baselines.py`: Baselines on the same serialized dataset shards (naive mean, last-pace, and XGBoost).
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

- `figures/`: Exported plots referenced in the technical doc / notebooks.
- [`Technical_Details.md`](./Technical_Details.md): Technical writeup (Markdown).
- [`Technical_Details.pdf`](./Technical_Details.pdf): Rendered technical writeup (PDF).
- `generate_white_paper_pdf.sh`, `White_Paper_Template.tex`: Build `Technical_Details.pdf` from `Technical_Details.md` via Pandoc.

## Quickstart (runs on the included sample data)

### 1) Install deps

```bash
python -m pip install -r requirements.txt
```

### 2) Run baselines (naive / last-pace / XGBoost)

`Benchmark_Baselines.py` takes one or more `*.pkl.gz` shard paths and writes artifacts to an explicit output directory. A convenience wrapper is included as `train/run_xgboost_tuning.sh`.

```bash
bash train/run_xgboost_tuning.sh
```

Artifacts produced (under `train/xgb_*` by default): `baseline_results.json`, `xgboost_model.json`, `xgboost_feature_columns.pickle`, plus feature-importance CSVs.

To enable randomized hyperparameter search:

```bash
TUNE=1 N_TRIALS=25 MAX_FILES=10 bash train/run_xgboost_tuning.sh
```

### 3) Train RunTime (from YAML config)

The trainer is config-driven via `--config`:

```bash
bash train/run_runtime_train.sh
```

Checkpoints are saved under `train/<save_dir>/<run_name>/` as configured in `train/runtime_trainer_config.yaml` (defaults to `checkpoints_clean_prod/Production_Scale_v2_HighCap/`).

If `use_wandb: true` in the config, set `WANDB_API_KEY` in your environment (recommended) or populate `wandb_api_key` in the YAML (not recommended to commit).

### 4) (Optional) Run a sweep

This repo includes a lightweight random sweep runner (no W\&B sweeps required) that writes generated configs under `train/sweeps/` by default.

```bash
# build configs only
bash train/run_runtime_sweep.sh build

# run a single trial (index 0)
bash train/run_runtime_sweep.sh run_one 0
```

### Running on Lambda (GPU quickstart)

On a fresh Ubuntu GPU machine:

```bash
git clone git@github.com:yaelelmatad/RunTime-Public.git
cd RunTime-Public

# one-time setup (creates .venv and installs deps)
bash train/setup_cloud.sh

# optional: enable W&B
export WANDB_API_KEY="..."

# run baselines
bash train/run_xgboost_tuning.sh

# run training (checkpoints under train/checkpoints_clean_prod/<run_name>/)
bash train/run_runtime_train.sh
```

## Performance summary

Filtered run-time metrics appear in `Technical_Details.md`; please refer to that document for the full MAE table.
## Build the PDF whitepaper

If you have `pandoc` installed:

```bash
bash generate_white_paper_pdf.sh
```

This generates/overwrites `Technical_Details.pdf` from `Technical_Details.md`.

## License

- **Code**: Apache License 2.0 (see `LICENSE` and `NOTICE`)
- **Documentation / writeup** (including `Technical_Details.md`): Creative Commons Attribution 4.0 International (see `LICENSE-CC-BY-4.0`)

