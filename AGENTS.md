This repo uses agent-friendly automation with strong expectations about where notebooks, configs,
and artifacts live. When interacting with `RunTime-Public` (or the sibling `RunTime-Full`), follow
these conventions so automated agents behave predictably.

## Core Principles
- The **paper** now lives in `paper/`: only that directory should contain `RunTime_Tabular_Main.*`.
- **Notebooks** belong under `evaluate/`. Do not leave untidy files in `train/`; it is reserved for
  scripts/configs only.
- When discussing figures, reference the files in `figures/`. Please do not reintroduce deleted PDFs,
  slides, or the old `Technical_Details.*` artifacts.
- All automation in `train/` now assumes the snake_case module names (`runtime_trainer.py`, `benchmark_baselines.py`, etc.).
  Avoid reversing these renames.
- Avoid regenerating artifacts nowhere near the source content (e.g., do not rebuild `RunTime_Tabular_Main.*`
  from the root; always `cd paper/` first).

## Helpful Sail-Points
- To reproduce the paper: `cd paper && latexmk -pdf -bibtex RunTime_Tabular_Main.tex`.
- To launch training: `bash train/run_runtime_train.sh` (defaults to `runtime_trainer_adaptive_sigma.yaml`), or call
  `python train/runtime_trainer_ablation.py --config train/runtime_trainer_time_token_ablation.yaml`, etc.
- Baseline/XGBoost work lives in `train/benchmark_baselines.py` and `train/run_xgboost_tuning.sh`.
- The evaluation scripts (`train/evaluate_models.py` and `_parallel.py`) depend on `train/runtime_inference.py`.
  Run notebooks from the repo root so those import paths resolve cleanly.
- For cloud setup: prefer `train/run-scripts/setup_cloud.sh` / `setup_lambda.sh`; `generate_white_paper_pdf.sh`,
  `Paper_Template.tex`, and the slides were removed intentionally—do not restore them.

## Git Hygiene
- You’ve already cleaned the repo: `Technical_Details.*`, `tabtransformer.pdf`, the old slides, `run_runtime_sweep.sh`, and the white‑paper generator are gone and shouldn’t be re-added.
- Keep `evaluate/` tracked (includes new notebooks plus renamed activation/output inspectors).
- Document any additional directories/change you make in this file to help future agents understand the layout.

## Deployment Notes
- When pushing, update both `RunTime-Public` and `RunTime-Full` with the same git history, except when `RunTime-Full`
  intentionally serves as a reference clone. Always run `git status` in both repostories before pushing to ensure no stray files
  (such as temporary figure exports) leak into commits.

## Need a refresher?
If you’re unsure what to do next, check `README.md` (covers the quickstart for training/baselines/evaluation) or open `paper/RunTime_Tabular_Main.tex`
to see how figure captions/sections are supposed to appear. Align every change with the structures described there.
