#!/usr/bin/env python3
"""
Standalone script to evaluate multiple Runtime Transformer models.

This script:
1. Loads training examples from a glob pattern
2. Runs inference on all specified models
3. Computes MAE, RMSE, percentiles, and calibration data
4. Groups errors by history length and week delta
5. Saves all results to a pickle file for plotting

Usage:
    python evaluate_models.py --input-glob "path/to/splits/*.pickle" \\
                               --models model1:checkpoint1.pt:config1.yaml:display_name1 \\
                               --num-examples 10000 \\
                               --output results.pickle

Or use a config file:
    python evaluate_models.py --config config.yaml
"""

import argparse
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

LOGGING_INTERVAL = 1000
MAX_CALIBRATION_EXAMPLES = 10000  # Maximum number of examples to use for calibration curve

from runtime_inference import (
    RuntimeModelInference,
    load_runners_from_splits,
    TrainingExample,
    RunnerForTraining
)

# XGBoost imports (optional)
try:
    import xgboost as xgb
    import pandas as pd
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. XGBoost baseline will be skipped.")

# Distance mapping for XGBoost
DISTANCE_MAP = {
    'distance_name_token_1_mile': 1.0, 'distance_name_token_1POINT5_miles': 1.5,
    'distance_name_token_1POINT7_miles': 1.7,  # Found in evaluation logs
    'distance_name_token_3_miles': 3.0, 'distance_name_token_3_kilometers': 1.86411,
    'distance_name_token_5_kilometers': 3.10686, 'distance_name_token_4_miles': 4.0,
    'distance_name_token_5_miles': 5.0, 'distance_name_token_8_kilometers': 4.97097,
    'distance_name_token_10_kilometers': 6.21371, 'distance_name_token_12_kilometers': 7.45645,
    'distance_name_token_12_miles': 12.0,  # Found in evaluation logs
    'distance_name_token_15_kilometers': 9.32057, 'distance_name_token_10_miles': 10.0,
    'distance_name_token_18_miles': 18.0, 'distance_name_token_20_kilometers': 12.4274,
    'distance_name_token_25_kilometers': 15.5343, 'distance_name_token_30_kilometers': 18.6411,
    'distance_name_token_half_marathon': 13.1094, 'distance_name_token_marathon': 26.2188
}


def compute_quantile(
    actual_pace: float,
    probs: np.ndarray,
    *,
    bin_starts: np.ndarray,
    bin_ends: np.ndarray,
) -> float:
    """Compute the percentile (0-100) of actual_pace in the predicted distribution.
    
    Args:
        actual_pace: Actual race pace in seconds
        probs: Probability array over bins (does not need to be normalized)
        bin_starts: numpy array of bin start boundaries (seconds)
        bin_ends: numpy array of bin end boundaries (seconds)
    
    Returns:
        Percentile (0-100) where the actual pace falls in the predicted distribution
    """
    probs_sum = float(np.sum(probs))
    probs_normalized = (probs / probs_sum) if probs_sum > 0 else probs
    
    # Compute CDF at bin ends
    cdf_at_bin_ends = np.cumsum(probs_normalized)
    
    # Find which bin contains actual_pace
    bin_idx = np.searchsorted(bin_ends, actual_pace)
    
    if bin_idx == 0:
        # actual_pace is before first bin
        percentile = 0.0
    elif bin_idx >= len(bin_ends):
        # actual_pace is after last bin
        percentile = 100.0
    else:
        # actual_pace is in bin[bin_idx]
        # CDF at start of bin (end of previous bin, or 0 if first bin)
        cdf_at_start = cdf_at_bin_ends[bin_idx - 1] if bin_idx > 0 else 0.0
        # CDF at end of bin
        cdf_at_end = cdf_at_bin_ends[bin_idx]
        
        bin_start = float(bin_starts[bin_idx])
        bin_end = float(bin_ends[bin_idx])
        
        if bin_end > bin_start:
            # Linear interpolation within bin
            fraction = (actual_pace - bin_start) / (bin_end - bin_start)
            cdf_at_actual = cdf_at_start + fraction * (cdf_at_end - cdf_at_start)
        else:
            # Degenerate bin (start == end), use end value
            cdf_at_actual = cdf_at_end
        
        percentile = cdf_at_actual * 100.0
    
    return percentile


def predict_naive_mean(example, runner):
    """Predict global mean pace from historical races."""
    if not example.raw_pace_data or len(example.raw_pace_data) < 2:
        return None
    history = example.raw_pace_data[:-1]
    prev_paces = [h[2] for h in history]
    if len(prev_paces) == 0:
        return None
    return np.mean(prev_paces)


def predict_last_race_pace(example):
    """Predict last observed pace in history."""
    if not example.raw_pace_data or len(example.raw_pace_data) < 2:
        return None
    history = example.raw_pace_data[:-1]
    if len(history) == 0:
        return None
    return history[-1][2]  # Last race pace


def predict_riegel_formula(example):
    """Predict using Riegel formula: pace_new = pace_old * (distance_new/distance_old)^0.06"""
    if not example.raw_pace_data or len(example.raw_pace_data) < 2:
        return None
    
    # Get previous race (second to last) and target race (last)
    p1_d = example.raw_pace_data[-2]  # Previous race: (distance_token, week_delta, pace)
    p2_d = example.raw_pace_data[-1]  # Target race: (distance_token, week_delta, pace)
    
    d1 = DISTANCE_MAP.get(p1_d[0])  # Previous race distance in miles
    d2 = DISTANCE_MAP.get(p2_d[0])  # Target race distance in miles
    
    if d1 and d2 and d1 > 0:
        p_ri = float(p1_d[2]) * (d2 / d1) ** 0.06
        return p_ri
    return None


def extract_week_delta_from_sequence(seq: List[str], block_stride: int = 11) -> Optional[int]:
    """Extract the week_delta_to_final token from the last historical race block."""
    num_races = len(seq) // block_stride
    if num_races < 2:
        return None
    last_hist_block_start = (num_races - 2) * block_stride
    week_delta_token = seq[last_hist_block_start + 8]  # week_delta_to_final is at position 8
    if week_delta_token.startswith('week_delta_'):
        try:
            return int(week_delta_token.split('_')[-1])
        except (ValueError, IndexError):
            return None
    return None


def bucket_history_length(h: int, max_history: int = 44) -> Optional[Union[int, str]]:
    """Bucket history lengths: 1-4 individual, then 5-9, 10-14, 15-19, etc. Capped at max_history."""
    if h > max_history:
        return None  # Exclude history lengths beyond max
    if h <= 4:
        return h  # Individual buckets for 1, 2, 3, 4
    else:
        # Group into ranges: 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44
        bucket_start = ((h - 5) // 5) * 5 + 5
        bucket_end = min(bucket_start + 4, max_history)
        if bucket_start == bucket_end:
            return bucket_start  # Single value if at max
        return f"{bucket_start}-{bucket_end}"


def bucket_week_delta(week_delta: int) -> str:
    """Bucket week deltas into ranges for plotting."""
    if week_delta < 4:
        return "0-3"
    elif week_delta < 8:
        return "4-7"
    elif week_delta < 13:
        return "8-12"
    elif week_delta < 26:
        return "13-25"
    elif week_delta < 52:
        return "26-51"
    elif week_delta < 104:
        return "52-103"
    else:
        return "104+"


def load_xgboost_model(xgb_model_path: str, xgb_feature_columns_path: str):
    """Load XGBoost model and feature columns.

    The feature columns pickle produced by benchmark_baselines.py is a dict:
        {'columns': List[str]}

    The original notebooks load it with:
        payload = pickle.load(f)
        xgb_feature_columns = payload.get('columns') if isinstance(payload, dict) else payload

    We mirror that logic here so that the DMatrix receives the exact
    training-time feature names instead of a single 'columns' field.
    """
    if not XGBOOST_AVAILABLE:
        return None, None
    
    try:
        booster = xgb.Booster()
        booster.load_model(xgb_model_path)
        
        with open(xgb_feature_columns_path, 'rb') as f:
            payload = pickle.load(f)
        # Handle both dict payloads and direct list/Index payloads
        if isinstance(payload, dict):
            feature_columns = payload.get('columns', None)
        else:
            feature_columns = payload

        if feature_columns is None:
            print("⚠️  Loaded XGBoost feature columns payload but could not find 'columns' key.")
            return booster, None
        
        return booster, feature_columns
    except Exception as e:
        print(f"⚠️  Failed to load XGBoost model or feature columns: {e}")
        return None, None


def extract_runner_features_full(runner, example):
    """Extract features for XGBoost prediction from a runner and example."""
    # Check if raw_pace_data exists and has enough entries
    if not hasattr(example, 'raw_pace_data') or not example.raw_pace_data or len(example.raw_pace_data) < 2:
        return None

    # Keep notebook evaluation aligned with benchmark_baselines.py:
    # That script implicitly requires at least one full previous race block (22 tokens total)
    # and will skip examples that don't have it.
    if len(example.unpadded_example_sequence) < 22:
        return None

    history = example.raw_pace_data[:-1]
    target_race = example.raw_pace_data[-1]

    prev_paces = [h[2] for h in history]
    if len(prev_paces) == 0:
        return None
    
    avg_pace = np.mean(prev_paces)
    last_pace = prev_paces[-1]
    std_pace = np.std(prev_paces) if len(prev_paces) > 1 else 0
    volatility = (std_pace / avg_pace) if avg_pace > 0 else 0

    ema_pace = prev_paces[0]
    for p in prev_paces[1:]:
        ema_pace = 0.3 * p + 0.7 * ema_pace

    final_block = example.unpadded_example_sequence[-11:]
    prev_block = example.unpadded_example_sequence[-22:-11]
    
    if len(final_block) < 11 or len(prev_block) < 11:
        return None

    try:
        age = float(final_block[0].split('_')[1])
        gender = 1 if runner.name_gender_dedup_int[2] == 'M' else 0
        cond_token = final_block[2]
        hum = float(final_block[3].split('_')[-1])
        temp = float(final_block[4].split('_')[-1])
        feels = float(final_block[5].split('_')[-1])
        wind = float(final_block[6].split('_')[-1])
        total_span = float(final_block[9].split('_')[-1])

        w_since = float(prev_block[8].split('_')[-1])
        prev_temp = float(prev_block[4].split('_')[-1])

        last_dist = history[-1][0]
        is_same_dist = 1 if target_race[0] == last_dist else 0

        same_dist_paces = [h[2] for h in history if h[0] == target_race[0]]
        avg_same_dist_pace = np.mean(same_dist_paces) if same_dist_paces else avg_pace

        return {
            'avg_historical_pace': avg_pace,
            'last_pace': last_pace,
            'ema_historical_pace': ema_pace,
            'min_historical_pace': np.min(prev_paces),
            'max_historical_pace': np.max(prev_paces),
            'std_historical_pace': std_pace,
            'pace_volatility': volatility,
            'num_prev_races': len(prev_paces),
            'pace_trend': last_pace - prev_paces[0],
            'weeks_since_last': w_since,
            'total_career_span': total_span,
            'age': age,
            'gender': gender,
            'temp_binned': temp,
            'hum_binned': hum,
            'feels_like_binned': feels,
            'wind_binned': wind,
            'temp_feels_diff': temp - feels,
            'temp_shock': temp - prev_temp,
            'is_same_distance': is_same_dist,
            'avg_same_dist_pace': avg_same_dist_pace,
            'distance': target_race[0],
            'conditions': cond_token,
        }
    except Exception as e:
        # Log first few errors for debugging
        if not hasattr(extract_runner_features_full, '_error_count'):
            extract_runner_features_full._error_count = 0
        if extract_runner_features_full._error_count < 3:
            print(f"  DEBUG: Feature extraction exception: {e}")
            print(f"  DEBUG: final_block length: {len(final_block)}, prev_block length: {len(prev_block)}")
            print(f"  DEBUG: raw_pace_data length: {len(example.raw_pace_data) if example.raw_pace_data else 0}")
            extract_runner_features_full._error_count += 1
        return None


def xgb_predict_from_feats(feats: dict, xgb_booster, xgb_feature_columns):
    """Predict pace using XGBoost from extracted features."""
    if feats is None or not XGBOOST_AVAILABLE or xgb_booster is None or not xgb_feature_columns:
        return None
    try:
        dist_token = feats.get('distance', '')
        dist_miles = DISTANCE_MAP.get(dist_token)
        if dist_miles is None:
            return None

        # Build a single-row frame with the same schema used in benchmark_baselines.py
        row = {
            'avg_historical_pace': feats.get('avg_historical_pace', 0),
            'last_pace': feats.get('last_pace', 0),
            'ema_historical_pace': feats.get('ema_historical_pace', 0),
            'min_historical_pace': feats.get('min_historical_pace', 0),
            'max_historical_pace': feats.get('max_historical_pace', 0),
            'std_historical_pace': feats.get('std_historical_pace', 0),
            'pace_volatility': feats.get('pace_volatility', 0),
            'num_prev_races': feats.get('num_prev_races', 0),
            'pace_trend': feats.get('pace_trend', 0),
            'weeks_since_last': feats.get('weeks_since_last', 0),
            'total_career_span': feats.get('total_career_span', 0),
            'age': feats.get('age', 0),
            'gender': feats.get('gender', 0),
            'temp_binned': feats.get('temp_binned', 0),
            'hum_binned': feats.get('hum_binned', 0),
            'feels_like_binned': feats.get('feels_like_binned', 0),
            'wind_binned': feats.get('wind_binned', 0),
            'temp_feels_diff': feats.get('temp_feels_diff', 0),
            'temp_shock': feats.get('temp_shock', 0),
            'is_same_distance': feats.get('is_same_distance', 0),
            'avg_same_dist_pace': feats.get('avg_same_dist_pace', 0),
            'distance_miles': float(dist_miles),
            'conditions': feats.get('conditions', ''),
        }

        df = pd.DataFrame([row])
        df = pd.get_dummies(df, columns=['conditions'], drop_first=False)
        
        # Reindex to match expected columns exactly, in the right order
        # This ensures all expected columns exist (fills missing with 0)
        df = df.reindex(columns=xgb_feature_columns, fill_value=0)
        
        # Use DMatrix directly from DataFrame (same as notebooks)
        # The DataFrame columns should match xgb_feature_columns after reindex
        dmat = xgb.DMatrix(df)
        pred = float(xgb_booster.predict(dmat)[0])
        return pred
    except Exception as e:
        # Log first few errors for debugging
        import traceback
        if not hasattr(xgb_predict_from_feats, '_error_logged'):
            print(f"  DEBUG: XGBoost prediction error: {e}")
            print(f"  DEBUG: Traceback: {traceback.format_exc()}")
            xgb_predict_from_feats._error_logged = True
        return None


def evaluate_models(
    input_glob: str,
    models: List[Tuple[str, str, str, str]],  # (model_name, checkpoint_path, config_path, display_name)
    num_examples: int,
    output_path: str,
    pace_lookup_path: str,
    xgb_model_path: Optional[str] = None,
    xgb_feature_columns_path: Optional[str] = None,
    device: str = "cpu",
    random_seed: Optional[int] = 42
):
    """
    Evaluate multiple models on training examples.
    
    Args:
        input_glob: Glob pattern for input pickle files
        models: List of (model_name, checkpoint_path, config_path, display_name) tuples
        num_examples: Number of examples to evaluate
        output_path: Path to save results pickle file
        pace_lookup_path: Path to pace_lookup.pickle
        xgb_model_path: Optional path to XGBoost model
        xgb_feature_columns_path: Optional path to XGBoost feature columns
        device: Device to use for inference
        random_seed: Random seed for reproducibility (default: 42). Sets seed for calibration curve sampling.
    """
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        import random
        random.seed(random_seed)
        print(f"Random seed set to: {random_seed}")
    
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    print(f"Input glob: {input_glob}")
    print(f"Number of examples: {num_examples}")
    print(f"Number of models: {len(models)}")
    print(f"Output path: {output_path}")
    print()
    
    # Load runners and examples
    print("Loading runners and collecting examples...")
    mae_runners = load_runners_from_splits(
        num_files=None,
        glob_pattern=input_glob,
        progress_interval=10
    )
    
    mae_examples = []
    mae_runners_list = []
    
    # Check if raw_pace_data is populated in first few examples
    raw_pace_data_check_count = 0
    raw_pace_data_missing = 0
    
    for runner in mae_runners:
        for ex in runner.training_examples:
            if len(ex.unpadded_example_sequence) >= 22:  # At least 1 prior race + 1 final race
                # Check if raw_pace_data is populated (sample first 10)
                if raw_pace_data_check_count < 10:
                    if not hasattr(ex, 'raw_pace_data') or not ex.raw_pace_data:
                        raw_pace_data_missing += 1
                    raw_pace_data_check_count += 1
                
                mae_examples.append(ex)
                mae_runners_list.append(runner)
                if len(mae_examples) >= num_examples:
                    break
        if len(mae_examples) >= num_examples:
            break
    
    unique_runner_ids = set(id(r) for r in mae_runners_list)
    print(f"✓ Loaded {len(mae_examples)} examples from {len(unique_runner_ids)} runners")
    if raw_pace_data_check_count > 0:
        if raw_pace_data_missing == raw_pace_data_check_count:
            print(f"⚠️  WARNING: raw_pace_data is missing in all checked examples!")
            print(f"   This will cause XGBoost feature extraction to fail.")
            print(f"   raw_pace_data should be populated when examples are loaded from pickle files.")
        elif raw_pace_data_missing > 0:
            print(f"⚠️  WARNING: raw_pace_data is missing in {raw_pace_data_missing}/{raw_pace_data_check_count} checked examples")
    print()
    
    # Load XGBoost if available
    xgb_booster = None
    xgb_feature_columns = None
    xgb_successful_indices = []
    xgb_predictions_cache = {}
    
    if xgb_model_path and xgb_feature_columns_path:
        print("Loading XGBoost model...")
        xgb_booster, xgb_feature_columns = load_xgboost_model(xgb_model_path, xgb_feature_columns_path)
        if xgb_booster:
            print("✓ XGBoost model loaded")
        else:
            print("⚠️  XGBoost model not loaded")
        print()
    
    # Load all transformer models
    print("Loading transformer models...")
    inference_engines = {}
    for model_name, checkpoint_path, config_path, display_name in models:
        print(f"  Loading {display_name} ({model_name})...")
        try:
            inference = RuntimeModelInference(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                pace_lookup_path=pace_lookup_path,
                device=device,
                enable_mps_fallback=True
            )
            inference_engines[model_name] = {
                'inference': inference,
                'display_name': display_name,
                'checkpoint_path': checkpoint_path,
                'config_path': config_path
            }
            print(f"    ✓ Loaded")
        except Exception as e:
            print(f"    ❌ Failed to load: {e}")
            continue
    print()
    
    # Determine which examples XGBoost can successfully predict on
    # Also compute XGBoost errors and cache metadata (history length, week delta) in a single pass
    xgb_errors = []  # Store XGBoost errors for final aggregation
    example_metadata_cache = {}  # Cache history length and week delta: {example_idx: {'history': int, 'week_delta': int or None}}
    
    if xgb_booster and xgb_feature_columns:
        print(f"Running XGBoost predictions to identify valid examples...")
        feats_failed = 0
        pred_failed = 0
        for i, (ex, runner) in enumerate(zip(mae_examples, mae_runners_list)):
            if (i + 1) % LOGGING_INTERVAL == 0:
                print(f"  Processed {i + 1}/{len(mae_examples)} examples...")
            
            # Extract features and predict
            feats = extract_runner_features_full(runner, ex)
            if feats:
                pred_xgb = xgb_predict_from_feats(feats, xgb_booster, xgb_feature_columns)
                if pred_xgb is not None:
                    xgb_successful_indices.append(i)
                    xgb_predictions_cache[i] = pred_xgb
                    
                    # Compute error immediately (don't recompute later)
                    actual_pace = ex.actual_pace_seconds
                    error_xgb = abs(pred_xgb - actual_pace)
                    xgb_errors.append((i, error_xgb))
                    
                    # Cache metadata (history length and week delta) to avoid recomputing
                    seq_len = len(ex.unpadded_example_sequence)
                    block_stride = 11
                    num_races = seq_len // block_stride
                    num_prev_races = num_races - 1
                    week_delta = extract_week_delta_from_sequence(ex.unpadded_example_sequence, block_stride)
                    example_metadata_cache[i] = {
                        'history': num_prev_races,
                        'week_delta': week_delta
                    }
                else:
                    pred_failed += 1
                    # Debug first few failures
                    if pred_failed <= 3:
                        dist_token = feats.get('distance', '')
                        dist_miles = DISTANCE_MAP.get(dist_token)
                        print(f"  DEBUG: XGBoost prediction failed for example {i}: dist_token={dist_token}, dist_miles={dist_miles}")
            else:
                feats_failed += 1
                # Debug first few failures
                if feats_failed <= 3:
                    print(f"  DEBUG: Feature extraction failed for example {i}: seq_len={len(ex.unpadded_example_sequence) if ex else 'N/A'}, raw_pace_data_len={len(ex.raw_pace_data) if ex and ex.raw_pace_data else 0}")
        
        print(f"✓ XGBoost: {len(xgb_successful_indices)} successful predictions out of {len(mae_examples)} examples")
        if len(xgb_successful_indices) == 0:
            print(f"  Feature extraction failed: {feats_failed} examples")
            print(f"  Prediction failed: {pred_failed} examples")
        if len(xgb_successful_indices) == 0:
            print("⚠️  No XGBoost predictions generated. Will evaluate all models on all examples.")
            xgb_successful_indices = list(range(len(mae_examples)))
        else:
            print(f"✓ Will evaluate all models on {len(xgb_successful_indices)} examples (where XGBoost succeeded)")
        print()
    else:
        print("⚠️  XGBoost not available. Will evaluate transformer models on all examples.")
        xgb_successful_indices = list(range(len(mae_examples)))
    
    print(f"Evaluating models on {len(xgb_successful_indices)} examples...")
    print()
    
    # Initialize results storage
    results = {}
    percentiles_by_model = {model_name: [] for model_name in inference_engines.keys()}
    # Store predicted paces for all examples (for Spearman correlation)
    predicted_paces_by_model = {
        model_name: {
            'weighted_mean': [],
            'weighted_median': [],
            'mode_pace': [],
            'actual_paces': []
        } for model_name in inference_engines.keys()
    }
    
    # Track errors by history length and week delta
    # Include XGBoost and baseline methods
    all_model_names_for_tracking = list(inference_engines.keys())
    if xgb_booster and xgb_feature_columns:
        all_model_names_for_tracking.append('XGBoost')
    # Add baseline methods
    all_model_names_for_tracking.extend(['NaiveMean', 'LastRacePace', 'RiegelFormula'])
    
    results_by_history_raw = defaultdict(lambda: {model_name: [] for model_name in all_model_names_for_tracking})
    results_by_week_delta = defaultdict(lambda: {model_name: [] for model_name in all_model_names_for_tracking})
    
    # Track baseline predictions and errors
    baseline_errors = {
        'NaiveMean': [],
        'LastRacePace': [],
        'RiegelFormula': []
    }
    baseline_predictions = {
        'NaiveMean': [],
        'LastRacePace': [],
        'RiegelFormula': []
    }
    baseline_actual_paces = {
        'NaiveMean': [],
        'LastRacePace': [],
        'RiegelFormula': []
    }
    
    # Initialize calibration data collection
    num_calibration_examples = min(MAX_CALIBRATION_EXAMPLES, len(xgb_successful_indices))
    calibration_data = {}
    
    # Cache predictions for example visualization (to avoid re-running inference)
    # Structure: {example_idx: {model_name: result_dict}}
    example_prediction_cache = {}
    seen_runners_for_examples = set()
    max_example_predictions = 5
    
    # Evaluate each model (single loop that does both MAE/percentiles and calibration curve data)
    for model_name, model_info in inference_engines.items():
        inference = model_info['inference']
        display_name = model_info['display_name']
        checkpoint_path = model_info['checkpoint_path']
        config_path = model_info['config_path']
        # Precompute bin boundaries once per model (huge speedup vs rebuilding per example)
        bin_starts = np.array([b['start'] for b in inference.pace_bins], dtype=np.float32)
        bin_ends = np.array([b['end'] for b in inference.pace_bins], dtype=np.float32)
        
        print(f"{'='*60}")
        print(f"Evaluating {display_name} ({model_name})...")
        print(f"{'='*60}")
        
        errors_mean = []
        errors_median = []
        errors_mode = []
        percentiles = []
        predicted_probs = []  # For calibration curve
        actual_outcomes = []   # For calibration curve
        
        # Store raw bin predictions for each example
        raw_predictions = []  # List of dicts: {example_idx, actual_pace, bin_probabilities, ...}
        
        for idx, example_idx in enumerate(xgb_successful_indices):
            if (idx + 1) % LOGGING_INTERVAL == 0:
                print(f"  Processed {idx + 1}/{len(xgb_successful_indices)} examples...")
            
            ex = mae_examples[example_idx]
            
            # Run inference once per example
            result = inference.predict_from_raw_example(ex)
            
            # Cache prediction for example visualization (if we need it and haven't seen this runner)
            # Count complete examples (with all models)
            complete_examples = sum(1 for preds in example_prediction_cache.values() 
                                   if len(preds) == len(inference_engines))
            
            if complete_examples < max_example_predictions:
                runner_id = id(mae_runners_list[example_idx])
                if runner_id not in seen_runners_for_examples:
                    if example_idx not in example_prediction_cache:
                        example_prediction_cache[example_idx] = {}
                    example_prediction_cache[example_idx][model_name] = {
                        'probabilities': result['probabilities'].copy(),
                        'weighted_mean': result['weighted_mean'],
                        'weighted_median': result['weighted_median'],
                        'mode_pace': result['mode_pace'],
                        'pace_bins': inference.pace_bins  # Reference is fine, pace_bins don't change
                    }
                    # Check if this example is now complete (has all models)
                    if len(example_prediction_cache[example_idx]) == len(inference_engines):
                        seen_runners_for_examples.add(runner_id)
                        # If we've collected enough complete examples, mark all remaining runners as seen
                        complete_examples = sum(1 for preds in example_prediction_cache.values() 
                                               if len(preds) == len(inference_engines))
                        if complete_examples >= max_example_predictions:
                            # Mark all runners we've seen so far to stop checking
                            for cached_idx in example_prediction_cache.keys():
                                seen_runners_for_examples.add(id(mae_runners_list[cached_idx]))
            
            # Compute errors
            actual_pace = ex.actual_pace_seconds
            error_mean = abs(result['weighted_mean'] - actual_pace)
            error_median = abs(result['weighted_median'] - actual_pace)
            error_mode = abs(result['mode_pace'] - actual_pace)
            
            # Store predicted paces for Spearman correlation
            predicted_paces_by_model[model_name]['weighted_mean'].append(result['weighted_mean'])
            predicted_paces_by_model[model_name]['weighted_median'].append(result['weighted_median'])
            predicted_paces_by_model[model_name]['mode_pace'].append(result['mode_pace'])
            predicted_paces_by_model[model_name]['actual_paces'].append(actual_pace)
            
            errors_mean.append(error_mean)
            errors_median.append(error_median)
            errors_mode.append(error_mode)
            
            # Compute percentile for calibration (Q-Q plots)
            percentile = compute_quantile(
                actual_pace,
                result['probabilities'],
                bin_starts=bin_starts,
                bin_ends=bin_ends,
            )
            percentiles.append(percentile)
            
            # Store raw bin predictions for this example
            # Convert probabilities array to dict: bin_number -> probability
            probs = result['probabilities']
            bin_probs_dict = {int(bin_idx): float(prob) for bin_idx, prob in enumerate(probs) if prob > 1e-10}  # Only store non-zero probabilities
            
            raw_predictions.append({
                'example_idx': example_idx,
                'actual_pace': actual_pace,
                'bin_probabilities': bin_probs_dict,  # Dict: bin_number -> probability
                'weighted_mean': result['weighted_mean'],
                'weighted_median': result['weighted_median'],
                'mode_pace': result['mode_pace'],
                'mode_bin_idx': result.get('mode_bin_idx', np.argmax(result['probabilities'])),
                'percentile': percentile
            })
            
            # Collect calibration curve data (if within limit)
            if idx < MAX_CALIBRATION_EXAMPLES:
                probs = result['probabilities']
                probs_normalized = probs / probs.sum() if probs.sum() > 0 else probs
                
                # Pre-compute CDF once per example (not per threshold sample)
                cdf = np.cumsum(probs_normalized)
                
                # Sample multiple random thresholds for each example
                # Note: This uses np.random which is seeded at the start of evaluate_models()
                n_samples_per_example = 5
                for _ in range(n_samples_per_example):
                    # Pick a random threshold bin (deterministic if seed is set)
                    threshold_bin = np.random.randint(0, len(inference.pace_bins))
                    
                    # Predicted probability of pace being <= threshold (CDF)
                    pred_prob = cdf[threshold_bin]
                    
                    # Actual outcome: did actual pace <= threshold?
                    threshold_value = float(bin_ends[threshold_bin])
                    actual = 1 if actual_pace <= threshold_value else 0
                    
                    predicted_probs.append(pred_prob)
                    actual_outcomes.append(actual)
            
            # Extract history length and week delta (use cached if available, otherwise compute and cache)
            if example_idx in example_metadata_cache:
                metadata = example_metadata_cache[example_idx]
                num_prev_races = metadata['history']
                week_delta = metadata['week_delta']
            else:
                seq_len = len(ex.unpadded_example_sequence)
                block_stride = 11
                num_races = seq_len // block_stride
                num_prev_races = num_races - 1
                week_delta = extract_week_delta_from_sequence(ex.unpadded_example_sequence, block_stride)
                example_metadata_cache[example_idx] = {
                    'history': num_prev_races,
                    'week_delta': week_delta
                }
            
            if num_prev_races >= 1:
                results_by_history_raw[num_prev_races][model_name].append(error_median)
            
            if week_delta is not None:
                results_by_week_delta[week_delta][model_name].append(error_median)
        
        # Compute MAE and RMSE
        mae_mean = np.mean(errors_mean)
        mae_median = np.mean(errors_median)
        mae_mode = np.mean(errors_mode)
        
        rmse_mean = np.sqrt(np.mean([e**2 for e in errors_mean]))
        rmse_median = np.sqrt(np.mean([e**2 for e in errors_median]))
        rmse_mode = np.sqrt(np.mean([e**2 for e in errors_mode]))
        
        results[model_name] = {
            'mae_mean': mae_mean,
            'mae_median': mae_median,
            'mae_mode': mae_mode,
            'rmse_mean': rmse_mean,
            'rmse_median': rmse_median,
            'rmse_mode': rmse_mode,
            'errors_mean': errors_mean,
            'errors_median': errors_median,
            'errors_mode': errors_mode,
            'predictions': {
                'weighted_mean': predicted_paces_by_model[model_name]['weighted_mean'],
                'weighted_median': predicted_paces_by_model[model_name]['weighted_median'],
                'mode_pace': predicted_paces_by_model[model_name]['mode_pace']
            },
            'actual_paces': predicted_paces_by_model[model_name]['actual_paces'],
            'display_name': display_name
        }
        
        percentiles_by_model[model_name] = percentiles
        
        # Process calibration curve data
        predicted_probs = np.clip(np.array(predicted_probs), 0.0, 1.0)
        actual_outcomes = np.array(actual_outcomes)
        
        calibration_data[model_name] = {
            'predicted_probs': predicted_probs,
            'actual_outcomes': actual_outcomes
        }
        
        print(f"✓ {display_name}: MAE={mae_mean:.2f}s, RMSE={rmse_mean:.2f}s")
        print(f"  Calibration data: {len(predicted_probs)} data points from {min(MAX_CALIBRATION_EXAMPLES, len(xgb_successful_indices))} examples")
        
        # Save raw predictions to separate file for this model
        raw_output_path = output_path.replace('.pickle', f'_{model_name}_raw_predictions.pickle')
        print(f"  Saving raw predictions to {raw_output_path}...")
        
        raw_predictions_data = {
            'model_name': model_name,
            'display_name': display_name,
            'num_examples': len(raw_predictions),
            'predictions': raw_predictions,  # List of dicts with bin probabilities for each example
            'pace_bins': inference.pace_bins,  # Store bin definitions for reference
            'bin_starts': bin_starts.tolist(),
            'bin_ends': bin_ends.tolist(),
            'metadata': {
                'checkpoint_path': checkpoint_path,
                'config_path': config_path,
                'num_bins': len(inference.pace_bins)
            }
        }
        
        with open(raw_output_path, 'wb') as f:
            pickle.dump(raw_predictions_data, f)
        
        print(f"  ✓ Saved {len(raw_predictions)} raw predictions to {raw_output_path}")
    
    # Compute baseline predictions (once per example, not per model)
    print(f"\n{'='*60}")
    print(f"Computing baseline predictions...")
    print(f"{'='*60}")
    
    for idx, example_idx in enumerate(xgb_successful_indices):
        if (idx + 1) % LOGGING_INTERVAL == 0:
            print(f"  Processed {idx + 1}/{len(xgb_successful_indices)} examples...")
        
        ex = mae_examples[example_idx]
        runner = mae_runners_list[example_idx]
        actual_pace = ex.actual_pace_seconds
        
        # Extract metadata (use cached if available)
        if example_idx in example_metadata_cache:
            metadata = example_metadata_cache[example_idx]
            num_prev_races = metadata['history']
            week_delta = metadata['week_delta']
        else:
            seq_len = len(ex.unpadded_example_sequence)
            block_stride = 11
            num_races = seq_len // block_stride
            num_prev_races = num_races - 1
            week_delta = extract_week_delta_from_sequence(ex.unpadded_example_sequence, block_stride)
            example_metadata_cache[example_idx] = {
                'history': num_prev_races,
                'week_delta': week_delta
            }
        
        # Naive mean baseline
        pred_naive = predict_naive_mean(ex, runner)
        if pred_naive is not None:
            error_naive = abs(pred_naive - actual_pace)
            baseline_errors['NaiveMean'].append(error_naive)
            baseline_predictions['NaiveMean'].append(pred_naive)
            baseline_actual_paces['NaiveMean'].append(actual_pace)
            # Add to history/week delta tracking
            if num_prev_races >= 1:
                results_by_history_raw[num_prev_races]['NaiveMean'].append(error_naive)
            if week_delta is not None:
                results_by_week_delta[week_delta]['NaiveMean'].append(error_naive)
        
        # Last race pace baseline
        pred_last = predict_last_race_pace(ex)
        if pred_last is not None:
            error_last = abs(pred_last - actual_pace)
            baseline_errors['LastRacePace'].append(error_last)
            baseline_predictions['LastRacePace'].append(pred_last)
            baseline_actual_paces['LastRacePace'].append(actual_pace)
            # Add to history/week delta tracking
            if num_prev_races >= 1:
                results_by_history_raw[num_prev_races]['LastRacePace'].append(error_last)
            if week_delta is not None:
                results_by_week_delta[week_delta]['LastRacePace'].append(error_last)
        
        # Riegel formula baseline
        pred_riegel = predict_riegel_formula(ex)
        if pred_riegel is not None:
            error_riegel = abs(pred_riegel - actual_pace)
            baseline_errors['RiegelFormula'].append(error_riegel)
            baseline_predictions['RiegelFormula'].append(pred_riegel)
            baseline_actual_paces['RiegelFormula'].append(actual_pace)
            # Add to history/week delta tracking
            if num_prev_races >= 1:
                results_by_history_raw[num_prev_races]['RiegelFormula'].append(error_riegel)
            if week_delta is not None:
                results_by_week_delta[week_delta]['RiegelFormula'].append(error_riegel)
    
    print()
    
    # Evaluate XGBoost if available (errors already computed in the initial loop)
    if xgb_booster and xgb_feature_columns and xgb_errors:
        print(f"\n{'='*60}")
        print(f"Evaluating XGBoost baseline...")
        print(f"{'='*60}")
        
        # Extract just the error values (already computed)
        errors_xgb = [error for _, error in xgb_errors]
        
        # Extract predictions and actual paces for Spearman correlation
        xgb_predictions_list = []
        xgb_actual_paces_list = []
        for example_idx in xgb_successful_indices:
            if example_idx in xgb_predictions_cache:
                xgb_predictions_list.append(xgb_predictions_cache[example_idx])
                xgb_actual_paces_list.append(mae_examples[example_idx].actual_pace_seconds)
        
        mae_xgb = np.mean(errors_xgb)
        rmse_xgb = np.sqrt(np.mean([e**2 for e in errors_xgb]))
        
        results['XGBoost'] = {
            'mae_mean': mae_xgb,
            'mae_median': mae_xgb,  # XGBoost gives point predictions, so median = mean
            'mae_mode': mae_xgb,
            'rmse_mean': rmse_xgb,
            'rmse_median': rmse_xgb,
            'rmse_mode': rmse_xgb,
            'errors_mean': errors_xgb,
            'errors_median': errors_xgb,
            'errors_mode': errors_xgb,
            'predictions': xgb_predictions_list,  # Store predictions for Spearman correlation
            'actual_paces': xgb_actual_paces_list,  # Store actual paces for Spearman correlation
            'display_name': 'XGBoost'
        }
        
        # Add XGBoost to history and week delta results (using cached metadata)
        for example_idx, error_xgb in xgb_errors:
            if example_idx in example_metadata_cache:
                metadata = example_metadata_cache[example_idx]
                num_prev_races = metadata['history']
                week_delta = metadata['week_delta']
                
                if num_prev_races >= 1:
                    results_by_history_raw[num_prev_races]['XGBoost'].append(error_xgb)
                
                if week_delta is not None:
                    results_by_week_delta[week_delta]['XGBoost'].append(error_xgb)
        
        print(f"✓ XGBoost: MAE={mae_xgb:.2f}s, RMSE={rmse_xgb:.2f}s")
    
    # Evaluate baseline methods
    print(f"\n{'='*60}")
    print(f"Evaluating baseline methods...")
    print(f"{'='*60}")
    
    for baseline_name in ['NaiveMean', 'LastRacePace', 'RiegelFormula']:
        errors = baseline_errors[baseline_name]
        predictions = baseline_predictions[baseline_name]
        actual_paces = baseline_actual_paces[baseline_name]
        if errors:
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean([e**2 for e in errors]))
            results[baseline_name] = {
                'mae_mean': mae,
                'mae_median': mae,  # Baselines give point predictions
                'mae_mode': mae,
                'rmse_mean': rmse,
                'rmse_median': rmse,
                'rmse_mode': rmse,
                'errors_mean': errors,
                'errors_median': errors,
                'errors_mode': errors,
                'predictions': predictions,  # Store predictions for Spearman correlation
                'actual_paces': actual_paces,  # Store actual paces for Spearman correlation
                'display_name': {
                    'NaiveMean': 'Naive Mean',
                    'LastRacePace': 'Last Race Pace',
                    'RiegelFormula': 'Riegel Formula'
                }[baseline_name]
            }
            print(f"✓ {results[baseline_name]['display_name']}: MAE={mae:.2f}s, RMSE={rmse:.2f}s")
        else:
            print(f"⚠️  {baseline_name}: No predictions generated")
    print()
    
    # Bucket history lengths and week deltas
    # Include XGBoost and baselines if available
    all_model_names = list(inference_engines.keys())
    if 'XGBoost' in results:
        all_model_names.append('XGBoost')
    for baseline_name in ['NaiveMean', 'LastRacePace', 'RiegelFormula']:
        if baseline_name in results:
            all_model_names.append(baseline_name)
    
    results_by_history = defaultdict(lambda: {model_name: [] for model_name in all_model_names})
    for hist_len, model_errors in results_by_history_raw.items():
        bucketed = bucket_history_length(hist_len)
        if bucketed is not None:
            for model_name, errors in model_errors.items():
                if model_name in all_model_names:
                    results_by_history[bucketed][model_name].extend(errors)
    
    results_by_week_delta_bucketed = defaultdict(lambda: {model_name: [] for model_name in all_model_names})
    for week_delta, model_errors in results_by_week_delta.items():
        bucketed = bucket_week_delta(week_delta)
        for model_name, errors in model_errors.items():
            if model_name in all_model_names:
                results_by_week_delta_bucketed[bucketed][model_name].extend(errors)
    
    # Build example_predictions from cache (collected during main loop)
    print("\nSaving example predictions for visualization...")
    example_predictions = []
    
    for example_idx, predictions_by_model in example_prediction_cache.items():
        # Verify we have predictions from all models for this example
        if len(predictions_by_model) == len(inference_engines):
            example_predictions.append({
                'example_idx': example_idx,
                'example': mae_examples[example_idx],
                'predictions_by_model': predictions_by_model
            })
    
    print(f"✓ Saved {len(example_predictions)} example predictions (collected during main evaluation, no extra inference)")
    
    # Prepare output data
    # Build model_config from the original models list to preserve display_name
    # (don't rely on inference_engines which might have been modified)
    model_config_output = {}
    for i, (model_name, checkpoint_path, config_path, display_name) in enumerate(models):
        if model_name in inference_engines:
            model_config_output[model_name] = {
                'display_name': display_name,  # Use display_name from input, not from inference_engines
                'checkpoint_path': checkpoint_path,
                'config_path': config_path
            }
    
    # Add XGBoost to model_config if available
    if 'XGBoost' in results:
        model_config_output['XGBoost'] = {
            'display_name': 'XGBoost',
            'checkpoint_path': xgb_model_path if xgb_model_path else None,
            'config_path': None
        }
    
    # Add baseline methods to model_config
    for baseline_name in ['NaiveMean', 'LastRacePace', 'RiegelFormula']:
        if baseline_name in results:
            model_config_output[baseline_name] = {
                'display_name': results[baseline_name]['display_name'],
                'checkpoint_path': None,
                'config_path': None
            }
    
    output_data = {
        'mae_examples': mae_examples,
        'mae_runners_list': mae_runners_list,
        'xgb_successful_indices': xgb_successful_indices,
        'results': results,
        'percentiles_by_model': percentiles_by_model,
        'results_by_history': dict(results_by_history),
        'results_by_week_delta': dict(results_by_week_delta_bucketed),
        'example_predictions': example_predictions,  # Add example predictions
        'calibration_data': calibration_data,  # Add calibration curve data
        'model_config': model_config_output,
        'metadata': {
            'num_examples': len(mae_examples),
            'num_evaluated': len(xgb_successful_indices),
            'input_glob': input_glob,
            'num_models': len(inference_engines)
        }
    }
    
    # Save to disk
    print()
    print("="*80)
    print("SAVING RESULTS")
    print("="*80)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    file_size_mb = output_path_obj.stat().st_size / (1024 * 1024)
    print(f"✓ Saved results to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Models evaluated: {len(results)}")
    print(f"  Examples: {len(mae_examples)}")
    print()
    print("="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


def parse_models_arg(models_str: str) -> List[Tuple[str, str, str, str]]:
    """Parse models argument: model_name:checkpoint:config:display_name"""
    models = []
    for model_str in models_str.split(','):
        parts = model_str.split(':')
        if len(parts) != 4:
            raise ValueError(f"Invalid model format: {model_str}. Expected: name:checkpoint:config:display_name")
        models.append(tuple(parts))
    return models


def main():
    parser = argparse.ArgumentParser(description='Evaluate Runtime Transformer models')
    parser.add_argument('--input-glob', type=str, help='Glob pattern for input pickle files')
    parser.add_argument('--models', type=str, help='Comma-separated list of models: name:checkpoint:config:display_name')
    parser.add_argument('--num-examples', type=int, default=10000, help='Number of examples to evaluate')
    parser.add_argument('--output', type=str, help='Output pickle file path (required unless --config is provided)')
    parser.add_argument('--pace-lookup', type=str, default='../data/pace_lookup.pickle', help='Path to pace_lookup.pickle')
    parser.add_argument('--xgb-model', type=str, help='Path to XGBoost model (optional)')
    parser.add_argument('--xgb-features', type=str, help='Path to XGBoost feature columns (optional)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda/mps)')
    parser.add_argument('--config', type=str, help='Path to YAML config file (alternative to command-line args)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        input_glob = config['input_glob']
        models = [tuple(m.values()) for m in config['models']]  # Convert dict list to tuple list
        num_examples = config.get('num_examples', 10000)
        output_path = config['output']
        pace_lookup_path = config.get('pace_lookup', '../data/pace_lookup.pickle')
        xgb_model_path = config.get('xgb_model')
        xgb_feature_columns_path = config.get('xgb_features')
        device = config.get('device', 'cpu')
        random_seed = config.get('random_seed', 42)  # Default to 42 if not specified
    else:
        if not args.input_glob or not args.models:
            parser.error("--input-glob and --models are required unless --config is provided")
        if not args.output:
            parser.error("--output is required unless --config is provided")
        
        input_glob = args.input_glob
        models = parse_models_arg(args.models)
        num_examples = args.num_examples
        output_path = args.output
        pace_lookup_path = args.pace_lookup
        xgb_model_path = args.xgb_model
        xgb_feature_columns_path = args.xgb_features
        device = args.device
        random_seed = args.random_seed
    
    evaluate_models(
        input_glob=input_glob,
        models=models,
        num_examples=num_examples,
        output_path=output_path,
        pace_lookup_path=pace_lookup_path,
        xgb_model_path=xgb_model_path,
        xgb_feature_columns_path=xgb_feature_columns_path,
        device=device,
        random_seed=random_seed
    )


if __name__ == '__main__':
    main()

