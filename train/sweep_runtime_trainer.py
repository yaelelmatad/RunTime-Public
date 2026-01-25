import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


def deep_set(d: Dict[str, Any], dot_key: str, value: Any) -> None:
    keys = dot_key.split(".")
    cur: Any = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def deep_get(d: Dict[str, Any], dot_key: str) -> Any:
    cur: Any = d
    for k in dot_key.split("."):
        cur = cur[k]
    return cur


def load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return yaml.safe_load(f)


def write_yaml(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def sample_param(rng: np.random.RandomState, spec: Dict[str, Any]) -> Any:
    t = spec["type"]
    if t == "categorical":
        vals = spec["values"]
        return vals[int(rng.randint(0, len(vals)))]
    if t == "int":
        lo, hi = int(spec["min"]), int(spec["max"])
        return int(rng.randint(lo, hi + 1))
    if t in ("float", "uniform"):
        lo, hi = float(spec["min"]), float(spec["max"])
        return float(rng.uniform(lo, hi))
    if t == "log_uniform":
        lo, hi = float(spec["min"]), float(spec["max"])
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    raise ValueError(f"Unknown type: {t}")


def short_run_name(index: int, overrides: Dict[str, Any]) -> str:
    h = hashlib.sha1(json.dumps(overrides, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]
    return f"rt_sweep_{index:03d}_{h}"


def build_configs(base_config: Path, sweep_spec: Path, out_dir: Path) -> List[Path]:
    base = load_yaml(base_config)
    spec = load_yaml(sweep_spec)

    method = str(spec.get("search_method", "random")).lower()
    n_trials = int(spec.get("n_trials", 1))
    seed = int(spec.get("seed", 42))
    params: Dict[str, Dict[str, Any]] = spec.get("parameters", {})
    fixed: Dict[str, Any] = spec.get("fixed", {})

    rng = np.random.RandomState(seed)

    configs_dir = out_dir / "configs"
    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_paths: List[Path] = []
    with open(manifest_path, "w") as mf:
        for i in range(n_trials):
            overrides: Dict[str, Any] = {}
            for k, pspec in params.items():
                overrides[k] = sample_param(rng, pspec)

            # hard constraint: d_model % nhead == 0
            dm = int(overrides.get("model.d_model", 0))
            nh = int(overrides.get("model.nhead", 1))
            if dm and nh and dm % nh != 0:
                for _ in range(200):
                    overrides = {k: sample_param(rng, pspec) for k, pspec in params.items()}
                    dm = int(overrides.get("model.d_model", 0))
                    nh = int(overrides.get("model.nhead", 1))
                    if dm and nh and dm % nh == 0:
                        break

            cfg = json.loads(json.dumps(base))
            for k, v in fixed.items():
                deep_set(cfg, k, v)
            for k, v in overrides.items():
                deep_set(cfg, k, v)

            deep_set(cfg, "logging.run_name", short_run_name(i, overrides))

            # Make paths absolute relative to the base config directory.
            anchor = base_config.resolve().parent
            for k in ("data.splits_dir", "data.pace_lookup"):
                try:
                    p = Path(str(deep_get(cfg, k)))
                    if not p.is_absolute():
                        deep_set(cfg, k, str((anchor / p).resolve()))
                except Exception:
                    pass

            cfg_path = configs_dir / f"run_{i:04d}.yaml"
            write_yaml(cfg_path, cfg)
            cfg_paths.append(cfg_path)

            mf.write(json.dumps({
                "index": i,
                "config_path": str(cfg_path),
                "run_name": deep_get(cfg, "logging.run_name"),
                "overrides": overrides,
                "fixed": fixed,
            }) + "\n")

    if method != "random":
        raise ValueError("Only random search is implemented in this runner.")
    return cfg_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", default="runtime_trainer_config.yaml")
    ap.add_argument("--sweep_spec", default="runtime_trainer_random_sweep.yaml")
    ap.add_argument("--trainer_py", default="Runtime_Trainer.py")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--mode", choices=["build", "run_all", "run_one"], default="run_one")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    base_config = Path(args.base_config).resolve()
    sweep_spec = Path(args.sweep_spec).resolve()
    trainer_py = Path(args.trainer_py).resolve()

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (Path("sweeps") / f"runtime_sweep_{ts}").resolve()

    cfgs = build_configs(base_config, sweep_spec, out_dir)
    print(f"[sweep] built {len(cfgs)} configs in: {out_dir}")

    if args.mode == "build":
        return

    if args.mode == "run_one":
        if args.index < 0 or args.index >= len(cfgs):
            raise SystemExit(f"index out of range: {args.index}")
        cfg = cfgs[args.index]
        cmd = [sys.executable, str(trainer_py), "--config", str(cfg)]
        print("[sweep] running:", " ".join(cmd))
        raise SystemExit(subprocess.call(cmd))

    failures = 0
    for i, cfg in enumerate(cfgs):
        cmd = [sys.executable, str(trainer_py), "--config", str(cfg)]
        print(f"\n[sweep] ===== run {i+1}/{len(cfgs)}: {cfg.name} =====")
        rc = subprocess.call(cmd)
        if rc != 0:
            failures += 1
            print(f"[sweep] WARN: failed rc={rc}: {cfg}")
    if failures:
        raise SystemExit(failures)


if __name__ == "__main__":
    main()


