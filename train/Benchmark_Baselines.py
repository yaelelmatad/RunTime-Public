import random
try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
except ModuleNotFoundError:
    # Allow `--help` / argparse usage even when ML deps aren't installed.
    np = None
    pd = None
    mean_absolute_error = None
from dataclasses import dataclass
from typing import List, Tuple
import sys
import argparse
import pickle
import gzip
import os 
import glob

# --- Distance token -> numeric miles (mirrors Runtime/train/Inspect_Model_Outputs.ipynb) ---
DISTANCE_MAP_MILES = {
    'distance_name_token_1_mile': 1.0,
    'distance_name_token_1POINT5_miles': 1.5,
    'distance_name_token_3_miles': 3.0,
    'distance_name_token_3_kilometers': 1.86411,
    'distance_name_token_5_kilometers': 3.10686,
    'distance_name_token_4_miles': 4.0,
    'distance_name_token_5_miles': 5.0,
    'distance_name_token_8_kilometers': 4.97097,
    'distance_name_token_10_kilometers': 6.21371,
    'distance_name_token_12_kilometers': 7.45645,
    'distance_name_token_15_kilometers': 9.32057,
    'distance_name_token_10_miles': 10.0,
    'distance_name_token_18_miles': 18.0,
    'distance_name_token_20_kilometers': 12.4274,
    'distance_name_token_25_kilometers': 15.5343,
    'distance_name_token_30_kilometers': 18.6411,
    'distance_name_token_half_marathon': 13.1094,
    'distance_name_token_marathon': 26.2188,
}

def _sample_xgb_params(rng: "np.random.RandomState"):
    """
    Random search over a tight-ish space targeted at this dataset.
    Keeps feature engineering unchanged; only tunes the model.
    """
    # Depth: user expectation 7-9, but keep a bit of spread around it.
    max_depth = int(rng.choice([5, 6, 7, 8, 9, 10]))

    # Learning rate: focus on 0.01-0.03, with a couple slightly higher options.
    eta = float(rng.choice([0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]))

    # Regularization / split constraints
    min_child_weight = float(rng.choice([1, 2, 3, 5, 7, 10]))
    gamma = float(rng.choice([0.0, 0.1, 0.25, 0.5, 1.0, 2.0]))
    reg_alpha = float(rng.choice([0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.1]))
    reg_lambda = float(rng.choice([0.5, 1.0, 2.0, 5.0, 10.0]))

    # Subsampling
    subsample = float(rng.choice([0.7, 0.8, 0.9, 1.0]))
    colsample_bytree = float(rng.choice([0.7, 0.8, 0.9, 1.0]))

    return {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'eta': eta,
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'gamma': gamma,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        # Slightly more robust on noisy regression
        'tree_method': 'hist',
    }


# --- 0. DATA CLASSES (Same as Transformer) ---

@dataclass
class TrainingExample:
    unpadded_example_sequence: List[str] 
    actual_pace_seconds: int 
    raw_pace_data: List[Tuple[str, str, int]] 

@dataclass
class RunnerForTraining:
    name_gender_dedup_int: Tuple[str, str, str, int]
    training_examples: List[TrainingExample]
    split_assignment: int

# --- 1. DATA LOADING UTILITY ---

def load_runners_data_from_files(file_paths: List[str]) -> List[RunnerForTraining]:
    all_runners = []
    for file_path in file_paths:
        print(f"Loading: {file_path}")
        try:
            with gzip.open(file_path, 'rb') as f:
                while True:
                    try:
                        batch = pickle.load(f)
                        if isinstance(batch, list):
                            all_runners.extend(batch)
                        elif isinstance(batch, RunnerForTraining):
                            all_runners.append(batch)
                    except EOFError:
                        break
        except Exception as e:
            print(f"ERROR loading {file_path}: {e}")
    return all_runners

# --- 2. TABULAR FEATURE ENGINEERING FOR XGBOOST ---

def extract_tabular_features(runners_data: List[RunnerForTraining]):
    """
    Converts sequence data into a fancy flat table for XGBoost.
    Includes temporal rhythm, performance volatility, and change deltas.
    """
    rows = []
    
    for runner in runners_data:
        gender = 1 if runner.name_gender_dedup_int[2] == 'M' else 0
        
        for example in runner.training_examples:
            if not example.raw_pace_data or len(example.raw_pace_data) < 2:
                continue
            
            # History = everything before the target race
            history = example.raw_pace_data[:-1]
            target_race = example.raw_pace_data[-1]
            
            prev_paces = [h[2] for h in history]
            
            # --- 1. Historical Stats ---
            avg_pace = np.mean(prev_paces)
            last_pace = prev_paces[-1]
            std_pace = np.std(prev_paces) if len(prev_paces) > 1 else 0
            min_pace = np.min(prev_paces)
            max_pace = np.max(prev_paces)
            # Volatility (Coefficient of Variation)
            volatility = (std_pace / avg_pace) if avg_pace > 0 else 0
            
            # EMA Pace (Recency weighting - alpha=0.3)
            ema_pace = prev_paces[0]
            for p in prev_paces[1:]:
                ema_pace = 0.3 * p + 0.7 * ema_pace
            
            # --- 2. Temporal Context & Weather (from tokens) ---
            # Sequence block [0:age, 1:gen, 2:cond, 3:hum, 4:temp, 5:feels, 6:wind, 7:dist, 8:w_next, 9:w_final, 10:pace]
            final_block = example.unpadded_example_sequence[-11:]
            prev_block = example.unpadded_example_sequence[-22:-11]
            
            try:
                age = float(final_block[0].split('_')[1])
                cond_token = final_block[2] # Categorical
                hum_token_val = float(final_block[3].split('_')[-1])
                temp_token_val = float(final_block[4].split('_')[-1])
                feels_token_val = float(final_block[5].split('_')[-1])
                wind_token_val = float(final_block[6].split('_')[-1])
                
                weeks_since_last = float(prev_block[8].split('_')[-1])
                total_span = float(final_block[9].split('_')[-1])
                
                # --- 3. Change Deltas ---
                prev_temp = float(prev_block[4].split('_')[-1])
                temp_shock = temp_token_val - prev_temp
                
                last_dist = target_race[0]
                if len(history) >= 1:
                    last_dist = history[-1][0]
                is_same_dist = 1 if target_race[0] == last_dist else 0
                
                same_dist_paces = [h[2] for h in history if h[0] == target_race[0]]
                avg_same_dist_pace = np.mean(same_dist_paces) if same_dist_paces else avg_pace
                
            except:
                continue

            # Distance token is a categorical token like distance_name_token_5_kilometers
            # For the "continuous token" XGBoost experiment, map this to numeric miles.
            dist_token = target_race[0]
            dist_miles = DISTANCE_MAP_MILES.get(dist_token)
            if dist_miles is None:
                # Skip rare/unknown distance tokens so we don't silently introduce NaNs into training.
                continue

            rows.append({
                'target_pace': example.actual_pace_seconds,
                'naive_mean_prediction': avg_pace,
                'last_pace': last_pace,
                'avg_historical_pace': avg_pace,
                'std_historical_pace': std_pace,
                'ema_historical_pace': ema_pace,
                'min_historical_pace': min_pace,
                'max_historical_pace': max_pace,
                'pace_volatility': volatility,
                'num_prev_races': len(prev_paces),
                'pace_trend': last_pace - prev_paces[0],
                'weeks_since_last': weeks_since_last,
                'total_career_span': total_span,
                'age': age,
                'gender': gender,
                'conditions': cond_token,
                'temp_binned': temp_token_val,
                'hum_binned': hum_token_val,
                'feels_like_binned': feels_token_val,
                'wind_binned': wind_token_val,
                'temp_feels_diff': temp_token_val - feels_token_val,
                'temp_shock': temp_shock,
                'is_same_distance': is_same_dist,
                'avg_same_dist_pace': avg_same_dist_pace,
                'distance_token': dist_token,
                'distance_miles': dist_miles,
            })
            
    return pd.DataFrame(rows)

# --- 3. MAIN BENCHMARK ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', metavar='FILE', type=str, nargs='*',
                        help="One or more streamed-pickle gzip files (runners_split_*.pkl.gz).")
    parser.add_argument('--splits_glob', type=str,
                        default=None,
                        help=("Glob for split shards to load when no FILE paths are provided. "
                              "If omitted, will auto-detect pipeline/training_splits (RunTime-Full) "
                              "or fall back to data/samples (RunTime-Public)."))
    parser.add_argument('--max_files', type=int, default=50,
                        help="When using --splits_glob, load at most this many shards (sorted).")
    parser.add_argument('--output_dir', type=str, default='.',
                        help="Where to write xgboost_model.json, xgboost_feature_columns.pickle, baseline_results.json")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--xgb_num_boost_round', type=int, default=500)
    parser.add_argument('--xgb_early_stopping_rounds', type=int, default=30)
    parser.add_argument('--tune', action='store_true',
                        help="Run randomized hyperparameter search (keeps feature engineering unchanged).")
    parser.add_argument('--n_trials', type=int, default=25,
                        help="Number of random hyperparameter trials when --tune is set.")
    args = parser.parse_args()

    # CRITICAL: Use same seed as Transformer for fair comparison
    random.seed(args.seed)
    np.random.seed(args.seed)

    # If no explicit shard paths are provided, default to first N shards from the glob.
    file_paths = list(args.file_paths) if args.file_paths else []
    if len(file_paths) == 0:
        # Auto-detect repo layout if caller didn't supply --splits_glob:
        # - RunTime-Full has: pipeline/training_splits/runners_split_*.pkl.gz
        # - RunTime-Public has: data/samples/*.pkl.gz
        splits_glob = args.splits_glob
        if not splits_glob:
            splits_glob = os.path.join("pipeline", "training_splits", "runners_split_*.pkl.gz")
            if not glob.glob(splits_glob):
                splits_glob = os.path.join("data", "samples", "*.pkl.gz")

        candidates = sorted(glob.glob(splits_glob))
        if not candidates:
            print(f"ERROR: no files found for --splits_glob: {splits_glob}")
            sys.exit(1)
        if args.max_files is not None and args.max_files > 0:
            candidates = candidates[: int(args.max_files)]
        file_paths = candidates

    print(f"Using {len(file_paths)} split shards.")
    if len(file_paths) <= 5:
        for p in file_paths:
            print(f"  - {p}")
    else:
        print(f"  - {file_paths[0]}")
        print(f"  - {file_paths[1]}")
        print(f"  - {file_paths[2]}")
        print("  - ...")
        print(f"  - {file_paths[-1]}")
    
    RUNNERS_DATA = load_runners_data_from_files(file_paths)
    if not RUNNERS_DATA:
        print("No data loaded.")
        sys.exit(1)

    random.shuffle(RUNNERS_DATA)
    split_idx = int(0.8 * len(RUNNERS_DATA))
    train_runners, val_runners = RUNNERS_DATA[:split_idx], RUNNERS_DATA[split_idx:]

    print(f"Creating Tabular Features...")
    df_train = extract_tabular_features(train_runners)
    df_val = extract_tabular_features(val_runners)

    print(f"Total training samples: {len(df_train)}")
    print(f"Total validation samples: {len(df_val)}")

    # --- Naive Mean Baseline ---
    naive_mae = mean_absolute_error(df_val['target_pace'], df_val['naive_mean_prediction'])
    print(f"\n[BASELINE] Naive Mean MAE: {naive_mae:.2f}s")

    # --- Last Pace Baseline ---
    last_pace_mae = mean_absolute_error(df_val['target_pace'], df_val['last_pace'])
    print(f"[BASELINE] Last Race Pace MAE: {last_pace_mae:.2f}s")

    # --- XGBoost Experiment ---
    try:
        import xgboost as xgb

        # Continuous-token variant:
        # - Keep numeric features numeric (age, weeks, weather, engineered pace stats, distance_miles)
        # - One-hot encode only categorical tokens (conditions, distance_token if you ever want it)
        base_features = [
            'avg_historical_pace', 'last_pace', 'ema_historical_pace',
            'min_historical_pace', 'max_historical_pace', 'std_historical_pace',
            'pace_volatility', 'num_prev_races', 'pace_trend', 'weeks_since_last',
            'total_career_span', 'age', 'gender', 'temp_binned', 'hum_binned',
            'feels_like_binned', 'wind_binned', 'temp_feels_diff', 'temp_shock',
            'is_same_distance', 'avg_same_dist_pace', 'distance_miles',
        ]

        # Build model matrices with one-hot encoding for categorical conditions
        train_X = df_train[base_features + ['conditions']].copy()
        val_X = df_val[base_features + ['conditions']].copy()
        train_X = pd.get_dummies(train_X, columns=['conditions'], drop_first=False)
        val_X = pd.get_dummies(val_X, columns=['conditions'], drop_first=False)

        # Align columns (critical for inference + notebook parity)
        train_cols = list(train_X.columns)
        val_X = val_X.reindex(columns=train_cols, fill_value=0)

        X_train = train_X
        y_train = df_train['target_pace'].astype(float)
        X_val = val_X
        y_val = df_val['target_pace'].astype(float)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Base params (used directly unless tuning is enabled)
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'eta': 0.05,
            'max_depth': 6,
            'min_child_weight': 1.0,
            'gamma': 0.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'tree_method': 'hist',
        }
        params['seed'] = int(args.seed)

        if args.tune:
            print(f"\nTuning XGBoost (random search; {args.n_trials} trials) ...")
            rng = np.random.RandomState(int(args.seed))
            best = {'mae': float('inf'), 'params': None, 'booster': None}

            # When using low eta, we generally need more rounds; use caller-provided rounds.
            num_boost_round = int(args.xgb_num_boost_round)
            early_stopping_rounds = int(args.xgb_early_stopping_rounds)

            for t in range(int(args.n_trials)):
                trial_params = _sample_xgb_params(rng)
                trial_params['seed'] = int(args.seed)

                booster_t = xgb.train(
                    params=trial_params,
                    dtrain=dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False,
                )

                # With eval_metric=mae + early stopping, best_score is val-mae at best iteration.
                try:
                    trial_mae = float(getattr(booster_t, "best_score", np.nan))
                except Exception:
                    trial_mae = np.nan

                if not np.isnan(trial_mae) and trial_mae < best['mae']:
                    best = {'mae': trial_mae, 'params': trial_params, 'booster': booster_t}
                    short_params = {
                        'eta': trial_params['eta'],
                        'max_depth': trial_params['max_depth'],
                        'min_child_weight': trial_params['min_child_weight'],
                        'gamma': trial_params['gamma'],
                        'reg_alpha': trial_params['reg_alpha'],
                        'reg_lambda': trial_params['reg_lambda'],
                        'subsample': trial_params['subsample'],
                        'colsample_bytree': trial_params['colsample_bytree'],
                    }
                    print(f"  trial {t+1}/{args.n_trials}: best so far mae={best['mae']:.4f} params={short_params}")

            if best['booster'] is None:
                raise RuntimeError("Tuning failed to produce a valid booster.")
            booster = best['booster']
            params = best['params']
            print(f"Tuning complete. Best val MAE (early-stopped): {best['mae']:.4f}")
        else:
            print(f"\nTraining XGBoost (continuous tokens; distance_miles numeric, conditions one-hot)...")
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=int(args.xgb_num_boost_round),
                evals=[(dval, 'val')],
                early_stopping_rounds=int(args.xgb_early_stopping_rounds),
                verbose_eval=False,
            )

        preds = booster.predict(dval)
        xgb_mae = mean_absolute_error(y_val, preds)
        print(f"[EXPERIMENT] XGBoost MAE: {xgb_mae:.2f}s")
        
        # --- SAVE ARTIFACTS ---
        os.makedirs(args.output_dir, exist_ok=True)
        xgb_model_path = os.path.join(args.output_dir, "xgboost_model.json")
        xgb_cols_path = os.path.join(args.output_dir, "xgboost_feature_columns.pickle")
        baseline_json_path = os.path.join(args.output_dir, "baseline_results.json")
        fi_gain_csv_path = os.path.join(args.output_dir, "xgboost_feature_importance_gain.csv")
        fi_weight_csv_path = os.path.join(args.output_dir, "xgboost_feature_importance_weight.csv")
        fi_cover_csv_path = os.path.join(args.output_dir, "xgboost_feature_importance_cover.csv")

        booster.save_model(xgb_model_path)
        with open(xgb_cols_path, "wb") as f:
            pickle.dump({'columns': train_cols}, f)
        
        baseline_results = {
            'naive_mean_mae': float(naive_mae),
            'last_pace_mae': float(last_pace_mae),
            'xgboost_mae': float(xgb_mae)
        }
        with open(baseline_json_path, "w") as f:
            import json
            json.dump(baseline_results, f, indent=4)

        # --- FEATURE IMPORTANCE ---
        # Note: With xgboost.train, feature names in get_score() are f0, f1, ...
        # Map these back to the pandas column names we trained on.
        def _importance_df(importance_type: str) -> pd.DataFrame:
            score = booster.get_score(importance_type=importance_type)
            rows = []
            for k, v in score.items():
                if isinstance(k, str) and k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    name = train_cols[idx] if 0 <= idx < len(train_cols) else k
                else:
                    name = k
                rows.append({"feature": name, importance_type: float(v)})
            df_imp = pd.DataFrame(rows)
            if df_imp.empty:
                return df_imp
            return df_imp.sort_values(by=importance_type, ascending=False).reset_index(drop=True)

        fi_gain = _importance_df("gain")
        fi_weight = _importance_df("weight")
        fi_cover = _importance_df("cover")

        if not fi_gain.empty:
            fi_gain.to_csv(fi_gain_csv_path, index=False)
            print("\nTop 20 features by gain:")
            print(fi_gain.head(20).to_string(index=False))
        else:
            print("\n[WARN] No feature importances returned for 'gain'.")

        if not fi_weight.empty:
            fi_weight.to_csv(fi_weight_csv_path, index=False)
        if not fi_cover.empty:
            fi_cover.to_csv(fi_cover_csv_path, index=False)
        
        print("\nArtifacts saved:")
        print(f"  - {xgb_model_path}")
        print(f"  - {xgb_cols_path}")
        print(f"  - {baseline_json_path}")
        if os.path.exists(fi_gain_csv_path):
            print(f"  - {fi_gain_csv_path}")
        if os.path.exists(fi_weight_csv_path):
            print(f"  - {fi_weight_csv_path}")
        if os.path.exists(fi_cover_csv_path):
            print(f"  - {fi_cover_csv_path}")

    except ImportError:
        print("\n[INFO] xgboost not installed. Run 'pip install xgboost' to see XGBoost results.")

