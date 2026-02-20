#!/usr/bin/env python3
"""
Parallelized version of evaluate_models.py for multi-GPU evaluation.

This script:
1. Splits models across available GPUs
2. Runs each model evaluation in parallel
3. Merges results at the end

Usage:
    # Single GPU (sequential, like original)
    python evaluate_models_parallel.py --config config.yaml --gpu-id 0
    
    # Multi-GPU (parallel, auto-assign)
    python evaluate_models_parallel.py --config config.yaml --parallel --num-gpus 4
    
    # Multi-GPU (parallel, specify exact GPUs)
    python evaluate_models_parallel.py --config config.yaml --parallel --gpu-list 0,2,4,6
    
    # Lambda cluster (one model per GPU via job array)
    python evaluate_models_parallel.py --config config.yaml --gpu-id $SLURM_ARRAY_TASK_ID --model-index $SLURM_ARRAY_TASK_ID
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
import multiprocessing as mp
from functools import partial
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

LOGGING_INTERVAL = 1000

from runtime_inference import (
    RuntimeModelInference,
    load_runners_from_splits,
    TrainingExample,
    RunnerForTraining
)

# Import from original evaluate_models
# We need to import the module and access functions directly
import evaluate_models as eval_models

DISTANCE_MAP = eval_models.DISTANCE_MAP
compute_quantile = eval_models.compute_quantile
extract_week_delta_from_sequence = eval_models.extract_week_delta_from_sequence
bucket_history_length = eval_models.bucket_history_length
bucket_week_delta = eval_models.bucket_week_delta
predict_naive_mean = eval_models.predict_naive_mean
predict_last_race_pace = eval_models.predict_last_race_pace
predict_riegel_formula = eval_models.predict_riegel_formula
xgb_predict_from_feats = eval_models.xgb_predict_from_feats
extract_runner_features_full = eval_models.extract_runner_features_full
MAX_CALIBRATION_EXAMPLES = eval_models.MAX_CALIBRATION_EXAMPLES


def extract_features_worker(args):
    """Worker function for parallel feature extraction."""
    idx, ex, runner = args
    feats = extract_runner_features_full(runner, ex)
    return idx, feats


def xgb_predict_batch(feats_list: List[dict], xgb_booster, xgb_feature_columns, valid_indices: List[int], use_gpu: bool = False):
    """
    Predict pace using XGBoost for a batch of features.
    Much faster than calling xgb_predict_from_feats one at a time.
    
    Args:
        use_gpu: If True, try to use GPU predictor (requires XGBoost built with GPU support)
    
    Returns:
        List of (index, prediction) tuples for successful predictions
    """
    if not feats_list or not XGBOOST_AVAILABLE or xgb_booster is None or not xgb_feature_columns:
        return []
    
    rows = []
    valid_batch_indices = []
    
    for feats, idx in zip(feats_list, valid_indices):
        if feats is None:
            continue
        
        dist_token = feats.get('distance', '')
        dist_miles = DISTANCE_MAP.get(dist_token)
        if dist_miles is None:
            continue
        
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
        rows.append(row)
        valid_batch_indices.append(idx)
    
    if not rows:
        return []
    
    try:
        # Create DataFrame for entire batch
        df = pd.DataFrame(rows)
        df = pd.get_dummies(df, columns=['conditions'], drop_first=False)
        df = df.reindex(columns=xgb_feature_columns, fill_value=0)
        
        # Predict on entire batch at once
        dmat = xgb.DMatrix(df)
        
        # Try to use GPU predictor if requested and available
        if use_gpu:
            try:
                # Set GPU predictor (requires XGBoost built with GPU support)
                predictions = xgb_booster.predict(dmat, iteration_range=(0, xgb_booster.num_boosted_rounds()))
                # Note: GPU predictor is set at model level, not prediction level
                # If GPU support is available, it should be used automatically
            except Exception as e:
                # Fall back to CPU if GPU fails
                predictions = xgb_booster.predict(dmat)
        else:
            predictions = xgb_booster.predict(dmat)
        
        # Return list of (index, prediction) tuples
        return [(valid_batch_indices[i], float(pred)) for i, pred in enumerate(predictions)]
    except Exception as e:
        # Log first error for debugging
        if not hasattr(xgb_predict_batch, '_error_logged'):
            print(f"  DEBUG: XGBoost batch prediction error: {e}")
            import traceback
            print(f"  DEBUG: Traceback: {traceback.format_exc()}")
            xgb_predict_batch._error_logged = True
        return []

# XGBoost imports (optional)
try:
    import xgboost as xgb
    import pandas as pd
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def evaluate_single_model(
    model_info: Tuple[str, str, str, str],  # (model_name, checkpoint_path, config_path, display_name)
    mae_examples: List[TrainingExample],
    mae_runners_list: List,
    xgb_successful_indices: List[int],
    pace_lookup_path: str,
    device: str,
    example_metadata_cache: Dict,
    output_path: str,  # For raw predictions file path
    gpu_id: Optional[int] = None,
    save_full_distributions: bool = False  # Set to True to save full bin_probabilities (makes files much larger)
) -> Dict:
    """
    Evaluate a single model on all examples.
    
    Returns a dictionary with all results for this model.
    """
    # Set CUDA_VISIBLE_DEVICES BEFORE importing torch (critical for multiprocessing)
    if device == "cuda" and gpu_id is not None:
        import os
        # Set CUDA_VISIBLE_DEVICES before any torch imports
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Now import torch - it will only see the specified GPU
        import torch
        # Force CUDA initialization
        if torch.cuda.is_available():
            torch.cuda.init()
            # Use device 0 since we've restricted visibility to just this GPU
            actual_device = "cuda:0"
        else:
            print(f"[{model_info[0]}] ⚠️  CUDA not available in subprocess, falling back to CPU")
            actual_device = "cpu"
    elif gpu_id is not None and device == "cuda":
        import torch
        actual_device = f"cuda:{gpu_id}"
    else:
        actual_device = device
    
    model_name, checkpoint_path, config_path, display_name = model_info
    
    print(f"[{model_name}] Loading model on {actual_device}...")
    try:
        inference = RuntimeModelInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            pace_lookup_path=pace_lookup_path,
            device=actual_device,
            enable_mps_fallback=True
        )
        print(f"[{model_name}] ✓ Model loaded")
    except Exception as e:
        print(f"[{model_name}] ❌ Failed to load: {e}")
        return None
    
    # Precompute bin boundaries once per model
    bin_starts = np.array([b['start'] for b in inference.pace_bins], dtype=np.float32)
    bin_ends = np.array([b['end'] for b in inference.pace_bins], dtype=np.float32)
    
    print(f"[{model_name}] Evaluating {display_name} on {len(xgb_successful_indices)} examples...")
    start_time = time.time()
    
    errors_mean = []
    errors_median = []
    errors_mode = []
    percentiles = []
    predicted_probs = []
    actual_outcomes = []
    
    # Store predictions for Spearman correlation
    predicted_paces = {
        'weighted_mean': [],
        'weighted_median': [],
        'mode_pace': [],
        'actual_paces': []
    }
    
    # Store raw bin predictions for each example
    raw_predictions = []
    
    # Track errors by history length and week delta
    results_by_history_raw = defaultdict(list)
    results_by_week_delta = defaultdict(list)
    
    # Use batched inference for better GPU utilization
    batch_size = 64  # Process 64 examples at a time
    num_batches = (len(xgb_successful_indices) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(xgb_successful_indices))
        batch_indices = xgb_successful_indices[batch_start:batch_end]
        batch_examples = [mae_examples[idx] for idx in batch_indices]
        
        if (batch_idx + 1) % max(1, num_batches // 20) == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - start_time
            rate = batch_end / elapsed if elapsed > 0 else 0
            remaining = (len(xgb_successful_indices) - batch_end) / rate if rate > 0 else 0
            print(f"[{model_name}] Processed {batch_end}/{len(xgb_successful_indices)} examples "
                  f"({rate:.1f} ex/s, ~{remaining:.0f}s remaining)")
        
        # Run batched inference
        batch_results = inference.predict_batch(batch_examples)
        
        # Process each result in the batch
        for local_idx, (example_idx, result) in enumerate(zip(batch_indices, batch_results)):
            idx = batch_start + local_idx
            ex = mae_examples[example_idx]
            
            # Compute errors
            actual_pace = ex.actual_pace_seconds
            error_mean = abs(result['weighted_mean'] - actual_pace)
            error_median = abs(result['weighted_median'] - actual_pace)
            error_mode = abs(result['mode_pace'] - actual_pace)
            
            # Store predictions
            predicted_paces['weighted_mean'].append(result['weighted_mean'])
            predicted_paces['weighted_median'].append(result['weighted_median'])
            predicted_paces['mode_pace'].append(result['mode_pace'])
            predicted_paces['actual_paces'].append(actual_pace)
            
            errors_mean.append(error_mean)
            errors_median.append(error_median)
            errors_mode.append(error_mode)
            
            # Compute percentile
            percentile = compute_quantile(
                actual_pace,
                result['probabilities'],
                bin_starts=bin_starts,
                bin_ends=bin_ends,
            )
            percentiles.append(percentile)
            
            # Store raw bin predictions for this example
            # Only store full distributions if requested (they make files very large)
            raw_pred = {
                'example_idx': example_idx,
                'actual_pace': actual_pace,
                'weighted_mean': result['weighted_mean'],
                'weighted_median': result['weighted_median'],
                'mode_pace': result['mode_pace'],
                'mode_bin_idx': result.get('mode_bin_idx', np.argmax(result['probabilities'])),
                'percentile': percentile
            }
            
            # Optionally store full bin probabilities (disabled by default to save space)
            # Only store if save_full_distributions=True (makes files much larger)
            if save_full_distributions:
                probs = result['probabilities']
                bin_probs_dict = {int(bin_idx): float(prob) for bin_idx, prob in enumerate(probs) if prob > 1e-10}  # Only store non-zero probabilities
                raw_pred['bin_probabilities'] = bin_probs_dict
            
            raw_predictions.append(raw_pred)
            
            # Collect calibration curve data (limit to MAX_CALIBRATION_EXAMPLES)
            if idx < min(MAX_CALIBRATION_EXAMPLES, len(xgb_successful_indices)):
                probs = result['probabilities']
                probs_normalized = probs / probs.sum() if probs.sum() > 0 else probs
                cdf = np.cumsum(probs_normalized)
                
                n_samples_per_example = 5
                for _ in range(n_samples_per_example):
                    threshold_bin = np.random.randint(0, len(inference.pace_bins))
                    pred_prob = cdf[threshold_bin]
                    threshold_value = float(bin_ends[threshold_bin])
                    actual = 1 if actual_pace <= threshold_value else 0
                    
                    predicted_probs.append(pred_prob)
                    actual_outcomes.append(actual)
            
            # Extract metadata
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
                results_by_history_raw[num_prev_races].append(error_median)
            
            if week_delta is not None:
                results_by_week_delta[week_delta].append(error_median)
    
    # Compute statistics
    mae_mean = np.mean(errors_mean)
    mae_median = np.mean(errors_median)
    mae_mode = np.mean(errors_mode)
    
    rmse_mean = np.sqrt(np.mean([e**2 for e in errors_mean]))
    rmse_median = np.sqrt(np.mean([e**2 for e in errors_median]))
    rmse_mode = np.sqrt(np.mean([e**2 for e in errors_mode]))
    
    elapsed = time.time() - start_time
    print(f"[{model_name}] ✓ Completed in {elapsed:.1f}s ({len(xgb_successful_indices)/elapsed:.1f} ex/s)")
    
    # Save raw predictions to separate file
    model_name, checkpoint_path, config_path, display_name = model_info
    raw_output_path = output_path.replace('.pickle', f'_{model_name}_raw_predictions.pickle')
    print(f"[{model_name}] Saving raw predictions to {raw_output_path}...")
    
    raw_predictions_data = {
        'model_name': model_name,
        'display_name': display_name,
        'num_examples': len(raw_predictions),
        'predictions': raw_predictions,
        'pace_bins': inference.pace_bins,
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
    
    print(f"[{model_name}] ✓ Saved {len(raw_predictions)} raw predictions")
    
    return {
        'model_name': model_name,
        'display_name': display_name,
        'mae_mean': mae_mean,
        'mae_median': mae_median,
        'mae_mode': mae_mode,
        'rmse_mean': rmse_mean,
        'rmse_median': rmse_median,
        'rmse_mode': rmse_mode,
        'errors_mean': errors_mean,
        'errors_median': errors_median,
        'errors_mode': errors_mode,
        'percentiles': percentiles,
        'predictions': {
            'weighted_mean': predicted_paces['weighted_mean'],
            'weighted_median': predicted_paces['weighted_median'],
            'mode_pace': predicted_paces['mode_pace'],
            'pace_bins': inference.pace_bins
        },
        'actual_paces': predicted_paces['actual_paces'],
        'calibration_data': {
            'predicted_probs': np.array(predicted_probs),
            'actual_outcomes': np.array(actual_outcomes)
        },
        'results_by_history_raw': dict(results_by_history_raw),
        'results_by_week_delta': dict(results_by_week_delta)
    }


def evaluate_models_parallel(
    input_glob: str,
    models: List[Tuple[str, str, str, str]],
    num_examples: int,
    output_path: str,
    pace_lookup_path: str,
    xgb_model_path: Optional[str] = None,
    xgb_feature_columns_path: Optional[str] = None,
    device: str = "cpu",
    parallel: bool = False,
    num_gpus: Optional[int] = None,
    gpu_id: Optional[int] = None,
    gpu_list: Optional[List[int]] = None,
    model_index: Optional[int] = None,
    random_seed: Optional[int] = 42,
    save_full_distributions: bool = False  # Set to True to save full bin_probabilities in raw predictions (makes files much larger)
):
    """
    Evaluate models in parallel across GPUs.
    
    Args:
        parallel: If True, use multiprocessing to run models in parallel
        num_gpus: Number of GPUs to use (if None, auto-detect)
        gpu_id: Specific GPU ID to use (for single model evaluation)
        gpu_list: List of specific GPU IDs to use (e.g., [0, 2, 4, 6])
        model_index: Specific model index to evaluate (for job arrays)
    """
    print("="*80)
    print("PARALLEL MODEL EVALUATION")
    print("="*80)
    print(f"Input glob: {input_glob}")
    print(f"Number of examples: {num_examples}")
    print(f"Number of models: {len(models)}")
    print(f"Output path: {output_path}")
    print(f"Device: {device}")
    print(f"Parallel: {parallel}")
    if gpu_id is not None:
        print(f"GPU ID: {gpu_id}")
    if gpu_list is not None:
        print(f"GPU List: {gpu_list}")
    if model_index is not None:
        print(f"Model index: {model_index}")
    print(f"Random seed: {random_seed}")
    print()
    
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        import random
        random.seed(random_seed)
        print(f"Random seed set to: {random_seed}")
    
    # Load runners and examples (shared across all models)
    print("Loading runners and collecting examples...")
    print(f"Glob pattern: {input_glob}")
    print("Searching for files...")
    import sys
    sys.stdout.flush()
    
    mae_runners = load_runners_from_splits(
        num_files=None,
        glob_pattern=input_glob,
        progress_interval=1  # Print progress for every file
    )
    print(f"✓ Loaded {len(mae_runners)} runners")
    sys.stdout.flush()
    
    mae_examples = []
    mae_runners_list = []
    
    for runner in mae_runners:
        for ex in runner.training_examples:
            if len(mae_examples) >= num_examples:
                break
            mae_examples.append(ex)
            mae_runners_list.append(runner)
        if len(mae_examples) >= num_examples:
            break
    
    print(f"✓ Loaded {len(mae_examples)} examples")
    
    # Pre-compute XGBoost and baseline predictions (shared)
    xgb_successful_indices = []
    xgb_predictions_cache = {}  # Store XGBoost predictions: {example_idx: prediction}
    example_metadata_cache = {}
    
    if xgb_model_path and xgb_feature_columns_path and XGBOOST_AVAILABLE:
        print("\nComputing XGBoost predictions...")
        xgb_booster = xgb.Booster()
        xgb_booster.load_model(xgb_model_path)
        with open(xgb_feature_columns_path, 'rb') as f:
            payload = pickle.load(f)
        # Handle both dict payloads and direct list/Index payloads
        # The feature columns pickle produced by benchmark_baselines.py is a dict: {'columns': List[str]}
        if isinstance(payload, dict):
            xgb_feature_columns = payload.get('columns', None)
            if xgb_feature_columns is None:
                print("⚠️  Warning: Feature columns dict has no 'columns' key")
                xgb_feature_columns = list(payload.keys()) if payload else None
        else:
            xgb_feature_columns = payload
        
        xgb_successful_indices = []
        import sys
        
        # Check if we can use GPU for XGBoost (requires XGBoost built with GPU support)
        use_gpu_xgb = False
        if device == "cuda":
            try:
                # Try to set GPU predictor (this will fail if XGBoost wasn't built with GPU support)
                # We'll detect this during prediction
                use_gpu_xgb = True
                print("  Attempting to use GPU for XGBoost predictions (if available)...")
            except:
                use_gpu_xgb = False
        
        # Parallel feature extraction using multiprocessing
        print("  Extracting features in parallel...")
        num_workers = min(mp.cpu_count(), 16)  # Use up to 16 workers
        print(f"  Using {num_workers} parallel workers for feature extraction...")
        
        # Prepare arguments for workers
        worker_args = [(idx, ex, mae_runners_list[idx]) for idx, ex in enumerate(mae_examples)]
        
        # Extract features in parallel
        all_feats = [None] * len(mae_examples)
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(extract_features_worker, worker_args)
            for idx, feats in results:
                all_feats[idx] = feats
        
        # Filter to valid features
        valid_indices = [idx for idx, feats in enumerate(all_feats) if feats is not None]
        valid_feats = [all_feats[idx] for idx in valid_indices]
        
        print(f"  ✓ Extracted features for {len(valid_feats)}/{len(mae_examples)} examples")
        print("  Running XGBoost predictions in batches (vectorized)...")
        
        # Batch predict (vectorized - much faster!)
        batch_size = 1000
        num_batches = (len(valid_feats) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(valid_feats))
            batch_feats = valid_feats[batch_start:batch_end]
            batch_valid_indices = valid_indices[batch_start:batch_end]
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                print(f"  Processed {batch_end}/{len(valid_feats)} predictions...")
                sys.stdout.flush()
            
            # Process entire batch at once (vectorized)
            batch_results = xgb_predict_batch(batch_feats, xgb_booster, xgb_feature_columns, batch_valid_indices, use_gpu=use_gpu_xgb)
            # Store both indices and predictions for statistics computation
            for idx, pred in batch_results:
                xgb_successful_indices.append(idx)
                xgb_predictions_cache[idx] = pred
        
        print(f"✓ XGBoost: {len(xgb_successful_indices)} successful predictions out of {len(mae_examples)} examples")
        
        # Compute XGBoost statistics
        if len(xgb_predictions_cache) > 0:
            xgb_errors = []
            xgb_predictions_list = []
            xgb_actual_paces_list = []
            
            for idx in xgb_successful_indices:
                if idx in xgb_predictions_cache:
                    pred_xgb = xgb_predictions_cache[idx]
                    actual_pace = mae_examples[idx].actual_pace_seconds
                    error_xgb = abs(pred_xgb - actual_pace)
                    xgb_errors.append(error_xgb)
                    xgb_predictions_list.append(pred_xgb)
                    xgb_actual_paces_list.append(actual_pace)
            
            if xgb_errors:
                mae_xgb = np.mean(xgb_errors)
                rmse_xgb = np.sqrt(np.mean([e**2 for e in xgb_errors]))
                print(f"✓ XGBoost: MAE={mae_xgb:.2f}s, RMSE={rmse_xgb:.2f}s")
                
                # Save XGBoost raw predictions (similar format to transformer models)
                print("Saving XGBoost raw predictions...")
                xgb_raw_predictions = []
                
                for idx in xgb_successful_indices:
                    if idx in xgb_predictions_cache:
                        pred_xgb = xgb_predictions_cache[idx]
                        actual_pace = mae_examples[idx].actual_pace_seconds
                        error_xgb = abs(pred_xgb - actual_pace)
                        
                        xgb_raw_predictions.append({
                            'example_idx': idx,
                            'actual_pace': actual_pace,
                            'predicted_pace': pred_xgb,  # Point prediction (not distribution)
                            'error': error_xgb,
                            # For consistency with transformer format
                            'weighted_mean': pred_xgb,
                            'weighted_median': pred_xgb,
                            'mode_pace': pred_xgb,
                        })
                
                # Save XGBoost raw predictions file
                xgb_raw_output_path = output_path.replace('.pickle', '_XGBoost_raw_predictions.pickle')
                xgb_raw_predictions_data = {
                    'model_name': 'XGBoost',
                    'display_name': 'XGBoost',
                    'num_examples': len(xgb_raw_predictions),
                    'predictions': xgb_raw_predictions,
                    'metadata': {
                        'xgb_model_path': xgb_model_path,
                        'xgb_features_path': xgb_feature_columns_path,
                        'is_point_prediction': True,  # XGBoost gives point predictions, not distributions
                    }
                }
                
                with open(xgb_raw_output_path, 'wb') as f:
                    pickle.dump(xgb_raw_predictions_data, f)
                
                print(f"✓ Saved {len(xgb_raw_predictions)} XGBoost raw predictions to {xgb_raw_output_path}")
            else:
                xgb_errors = []
                xgb_predictions_list = []
                xgb_actual_paces_list = []
        else:
            xgb_errors = []
            xgb_predictions_list = []
            xgb_actual_paces_list = []
    else:
        xgb_successful_indices = list(range(len(mae_examples)))
        xgb_predictions_cache = {}
        xgb_errors = []
        xgb_predictions_list = []
        xgb_actual_paces_list = []
        # Initialize baseline caches even if XGBoost is not available
        baseline_predictions_cache = {
            'NaiveMean': {},
            'RiegelFormula': {}
        }
        baseline_errors_cache = {
            'NaiveMean': [],
            'RiegelFormula': []
        }
        
        # Compute baselines for all examples when XGBoost is not available
        print("Computing baseline predictions...")
        for idx in xgb_successful_indices:
            ex = mae_examples[idx]
            runner = mae_runners_list[idx]
            actual_pace = ex.actual_pace_seconds
            
            # Compute Naive Mean baseline
            pred_naive = predict_naive_mean(ex, runner)
            if pred_naive is not None:
                error_naive = abs(pred_naive - actual_pace)
                baseline_predictions_cache['NaiveMean'][idx] = pred_naive
                baseline_errors_cache['NaiveMean'].append(error_naive)
            
            # Compute Riegel Formula baseline
            pred_riegel = predict_riegel_formula(ex)
            if pred_riegel is not None:
                error_riegel = abs(pred_riegel - actual_pace)
                baseline_predictions_cache['RiegelFormula'][idx] = pred_riegel
                baseline_errors_cache['RiegelFormula'].append(error_riegel)
        
        print(f"✓ Naive Mean: {len(baseline_predictions_cache['NaiveMean'])} predictions")
        print(f"✓ Riegel Formula: {len(baseline_predictions_cache['RiegelFormula'])} predictions")
    
    # Pre-compute metadata cache and baseline predictions
    print("Pre-computing example metadata and baseline predictions...")
    baseline_predictions_cache = {
        'NaiveMean': {},
        'RiegelFormula': {}
    }
    baseline_errors_cache = {
        'NaiveMean': [],
        'RiegelFormula': []
    }
    
    for idx in xgb_successful_indices:
        ex = mae_examples[idx]
        runner = mae_runners_list[idx]
        actual_pace = ex.actual_pace_seconds
        
        seq_len = len(ex.unpadded_example_sequence)
        block_stride = 11
        num_races = seq_len // block_stride
        num_prev_races = num_races - 1
        week_delta = extract_week_delta_from_sequence(ex.unpadded_example_sequence, block_stride)
        example_metadata_cache[idx] = {
            'history': num_prev_races,
            'week_delta': week_delta
        }
        
        # Compute Naive Mean baseline
        pred_naive = predict_naive_mean(ex, runner)
        if pred_naive is not None:
            error_naive = abs(pred_naive - actual_pace)
            baseline_predictions_cache['NaiveMean'][idx] = pred_naive
            baseline_errors_cache['NaiveMean'].append(error_naive)
        
        # Compute Riegel Formula baseline (uses penultimate race, which is already correct in the function)
        pred_riegel = predict_riegel_formula(ex)
        if pred_riegel is not None:
            error_riegel = abs(pred_riegel - actual_pace)
            baseline_predictions_cache['RiegelFormula'][idx] = pred_riegel
            baseline_errors_cache['RiegelFormula'].append(error_riegel)
    
    print(f"✓ Pre-computed metadata for {len(example_metadata_cache)} examples")
    print(f"✓ Naive Mean: {len(baseline_predictions_cache['NaiveMean'])} predictions")
    print(f"✓ Riegel Formula: {len(baseline_predictions_cache['RiegelFormula'])} predictions")
    
    # Handle single model evaluation (for job arrays)
    if model_index is not None:
        if model_index >= len(models):
            print(f"Error: model_index {model_index} >= {len(models)}")
            return
        models = [models[model_index]]
        print(f"Evaluating single model: {models[0][0]}")
    
    # Evaluate models
    if parallel and device == "cuda":
        # Parallel evaluation across GPUs
        # Set multiprocessing start method to 'spawn' to ensure clean CUDA initialization
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, ignore
            pass
        
        if gpu_list is not None:
            # Use specified GPU list
            available_gpus = gpu_list
            num_gpus = len(available_gpus)
            print(f"\nRunning {len(models)} models in parallel on GPUs: {available_gpus}")
        else:
            # Auto-detect or use num_gpus
            if num_gpus is None:
                import torch
                num_gpus = torch.cuda.device_count()
            available_gpus = list(range(num_gpus))
            print(f"\nRunning {len(models)} models in parallel on {num_gpus} GPUs...")
        
        # Create process pool
        with mp.Pool(processes=min(len(models), num_gpus)) as pool:
            # Assign GPU IDs round-robin from available_gpus list
            gpu_assignments = [available_gpus[i % len(available_gpus)] for i in range(len(models))]
            
            # Create evaluation tasks
            tasks = []
            for i, model_info in enumerate(models):
                tasks.append((
                    model_info,
                    mae_examples,
                    mae_runners_list,
                    xgb_successful_indices,
                    pace_lookup_path,
                    device,
                    example_metadata_cache,
                    output_path,  # Pass output_path for raw predictions
                    gpu_assignments[i],
                    save_full_distributions  # Pass flag for saving full distributions
                ))
            
            # Run in parallel
            results_list = pool.starmap(evaluate_single_model, tasks)
    else:
        # Sequential evaluation
        print(f"\nRunning {len(models)} models sequentially...")
        results_list = []
        for i, model_info in enumerate(models):
            result = evaluate_single_model(
                model_info,
                mae_examples,
                mae_runners_list,
                xgb_successful_indices,
                pace_lookup_path,
                device,
                example_metadata_cache,
                output_path,  # Pass output_path for raw predictions
                gpu_id,
                save_full_distributions  # Pass flag for saving full distributions
            )
            if result:
                results_list.append(result)
    
    # Merge results
    print("\nMerging results...")
    results = {}
    percentiles_by_model = {}
    calibration_data = {}
    results_by_history_raw = defaultdict(lambda: defaultdict(list))
    results_by_week_delta = defaultdict(lambda: defaultdict(list))
    
    for result in results_list:
        if result is None:
            continue
        
        model_name = result['model_name']
        results[model_name] = {
            'mae_mean': result['mae_mean'],
            'mae_median': result['mae_median'],
            'mae_mode': result['mae_mode'],
            'rmse_mean': result['rmse_mean'],
            'rmse_median': result['rmse_median'],
            'rmse_mode': result['rmse_mode'],
            'errors_mean': result['errors_mean'],
            'errors_median': result['errors_median'],
            'errors_mode': result['errors_mode'],
            'display_name': result['display_name'],
            'predictions': result['predictions'],
            'actual_paces': result['actual_paces']
        }
        
        percentiles_by_model[model_name] = result['percentiles']
        calibration_data[model_name] = result['calibration_data']
        
        # Merge history and week delta results
        for hist_len, errors in result['results_by_history_raw'].items():
            results_by_history_raw[hist_len][model_name].extend(errors)
        for week_delta, errors in result['results_by_week_delta'].items():
            results_by_week_delta[week_delta][model_name].extend(errors)
    
    # Bucket history lengths and week deltas
    # Note: Initialize with all model names, including XGBoost if it will be added
    all_model_names = list(results.keys())
    # Add XGBoost to the list if it will be added (to ensure it's in the defaultdict)
    if xgb_model_path and xgb_feature_columns_path and XGBOOST_AVAILABLE and len(xgb_errors) > 0:
        if 'XGBoost' not in all_model_names:
            all_model_names.append('XGBoost')
    
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
    
    # Add XGBoost results if available
    if xgb_model_path and xgb_feature_columns_path and XGBOOST_AVAILABLE and len(xgb_errors) > 0:
        mae_xgb = np.mean(xgb_errors)
        rmse_xgb = np.sqrt(np.mean([e**2 for e in xgb_errors]))
        results['XGBoost'] = {
            'mae_mean': mae_xgb,
            'mae_median': mae_xgb,  # XGBoost gives point predictions, so median = mean
            'mae_mode': mae_xgb,
            'rmse_mean': rmse_xgb,
            'rmse_median': rmse_xgb,
            'rmse_mode': rmse_xgb,
            'errors_mean': xgb_errors,
            'errors_median': xgb_errors,
            'errors_mode': xgb_errors,
            'predictions': xgb_predictions_list,
            'actual_paces': xgb_actual_paces_list,
            'display_name': 'XGBoost'
        }
        
        # Add XGBoost to history and week delta results
        for idx in xgb_successful_indices:
            if idx in example_metadata_cache:
                metadata = example_metadata_cache[idx]
                num_prev_races = metadata['history']
                week_delta = metadata['week_delta']
                
                if idx in xgb_predictions_cache:
                    actual_pace = mae_examples[idx].actual_pace_seconds
                    error_xgb = abs(xgb_predictions_cache[idx] - actual_pace)
                    
                    if num_prev_races >= 1:
                        bucketed_hist = bucket_history_length(num_prev_races)
                        if bucketed_hist is not None:
                            results_by_history[bucketed_hist]['XGBoost'].append(error_xgb)
                    
                    if week_delta is not None:
                        bucketed_week = bucket_week_delta(week_delta)
                        if bucketed_week is not None:
                            results_by_week_delta_bucketed[bucketed_week]['XGBoost'].append(error_xgb)
        
        print(f"✓ Added XGBoost results: MAE={mae_xgb:.2f}s, RMSE={rmse_xgb:.2f}s")
    
    # Save baseline raw predictions (NaiveMean, RiegelFormula)
    print("\n" + "="*80)
    print("Saving baseline raw predictions...")
    print(f"baseline_predictions_cache keys: {list(baseline_predictions_cache.keys()) if 'baseline_predictions_cache' in locals() else 'NOT IN SCOPE'}")
    
    # Ensure baseline_predictions_cache exists
    if 'baseline_predictions_cache' not in locals():
        print("  ⚠️  ERROR: baseline_predictions_cache not in scope!")
        print("  Attempting to recompute baseline predictions...")
        # Recompute if missing
        baseline_predictions_cache = {
            'NaiveMean': {},
            'RiegelFormula': {}
        }
        baseline_errors_cache = {
            'NaiveMean': [],
            'RiegelFormula': []
        }
        
        print("  Recomputing baseline predictions for all examples...")
        for idx in xgb_successful_indices:
            ex = mae_examples[idx]
            runner = mae_runners_list[idx]
            actual_pace = ex.actual_pace_seconds
            
            # Compute Naive Mean baseline
            pred_naive = predict_naive_mean(ex, runner)
            if pred_naive is not None:
                error_naive = abs(pred_naive - actual_pace)
                baseline_predictions_cache['NaiveMean'][idx] = pred_naive
                baseline_errors_cache['NaiveMean'].append(error_naive)
            
            # Compute Riegel Formula baseline
            pred_riegel = predict_riegel_formula(ex)
            if pred_riegel is not None:
                error_riegel = abs(pred_riegel - actual_pace)
                baseline_predictions_cache['RiegelFormula'][idx] = pred_riegel
                baseline_errors_cache['RiegelFormula'].append(error_riegel)
        
        print(f"  ✓ Recomputed: Naive Mean: {len(baseline_predictions_cache['NaiveMean'])} predictions")
        print(f"  ✓ Recomputed: Riegel Formula: {len(baseline_predictions_cache['RiegelFormula'])} predictions")
    
    for baseline_name in ['NaiveMean', 'RiegelFormula']:
        # Debug: Check if baseline_predictions_cache exists and has data
        if baseline_name not in baseline_predictions_cache:
            print(f"  ⚠️  {baseline_name} not in baseline_predictions_cache")
            continue
        
        if len(baseline_predictions_cache[baseline_name]) == 0:
            print(f"  ⚠️  {baseline_name} has 0 predictions in cache")
            continue
        
        print(f"  Processing {baseline_name}: {len(baseline_predictions_cache[baseline_name])} predictions in cache")
        baseline_raw_predictions = []
        
        for idx in xgb_successful_indices:
            if idx in baseline_predictions_cache[baseline_name]:
                pred = baseline_predictions_cache[baseline_name][idx]
                actual_pace = mae_examples[idx].actual_pace_seconds
                error = abs(pred - actual_pace)
                
                baseline_raw_predictions.append({
                    'example_idx': idx,
                    'actual_pace': actual_pace,
                    'predicted_pace': pred,  # Point prediction
                    'error': error,
                    # For consistency with transformer format
                    'weighted_mean': pred,
                    'weighted_median': pred,
                    'mode_pace': pred,
                })
        
        if len(baseline_raw_predictions) == 0:
            print(f"  ⚠️  {baseline_name}: No predictions after filtering by xgb_successful_indices")
            continue
        
        # Save baseline raw predictions file
        baseline_raw_output_path = output_path.replace('.pickle', f'_{baseline_name}_raw_predictions.pickle')
        print(f"  Saving {baseline_name} raw predictions to {baseline_raw_output_path}...")
        
        baseline_raw_predictions_data = {
            'model_name': baseline_name,
            'display_name': 'Naive Mean' if baseline_name == 'NaiveMean' else 'Riegel Formula',
            'num_examples': len(baseline_raw_predictions),
            'predictions': baseline_raw_predictions,
            'metadata': {
                'is_point_prediction': True,  # Baselines give point predictions, not distributions
                'riegel_uses_penultimate_race': True if baseline_name == 'RiegelFormula' else None
            }
        }
        
        try:
            with open(baseline_raw_output_path, 'wb') as f:
                pickle.dump(baseline_raw_predictions_data, f)
            print(f"  ✓ Saved {len(baseline_raw_predictions)} {baseline_name} raw predictions to {baseline_raw_output_path}")
        except Exception as e:
            print(f"  ✗ Error saving {baseline_name} raw predictions: {e}")
            continue
        
        # Compute and add statistics to results
        if baseline_name in baseline_errors_cache and len(baseline_errors_cache[baseline_name]) > 0:
            errors = baseline_errors_cache[baseline_name]
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean([e**2 for e in errors]))
            predictions_list = [baseline_predictions_cache[baseline_name][idx] for idx in xgb_successful_indices 
                              if idx in baseline_predictions_cache[baseline_name]]
            actual_paces_list = [mae_examples[idx].actual_pace_seconds for idx in xgb_successful_indices 
                                if idx in baseline_predictions_cache[baseline_name]]
            
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
                'predictions': predictions_list,
                'actual_paces': actual_paces_list,
                'display_name': 'Naive Mean' if baseline_name == 'NaiveMean' else 'Riegel Formula'
            }
            
            print(f"  ✓ Added {baseline_name} results: MAE={mae:.2f}s, RMSE={rmse:.2f}s")
        else:
            print(f"  ⚠️  {baseline_name}: No errors in baseline_errors_cache")
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    output_data = {
        'mae_examples': mae_examples,
        'mae_runners_list': mae_runners_list,
        'xgb_successful_indices': xgb_successful_indices,
        'results': results,
        'percentiles_by_model': percentiles_by_model,
        'results_by_history': dict(results_by_history),
        'results_by_week_delta': dict(results_by_week_delta_bucketed),
        'calibration_data': calibration_data,
        'metadata': {
            'num_examples': len(mae_examples),
            'num_models': len(results),
            'input_glob': input_glob
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print("✓ Evaluation complete!")
    print(f"\nResults summary:")
    for model_name, model_results in results.items():
        print(f"  {model_results['display_name']}: MAE={model_results['mae_median']:.2f}s, "
              f"RMSE={model_results['rmse_median']:.2f}s")




def main():
    parser = argparse.ArgumentParser(description='Evaluate Runtime Transformer models in parallel')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--input-glob', type=str, help='Glob pattern for input files')
    parser.add_argument('--models', type=str, help='Comma-separated model specs')
    parser.add_argument('--num-examples', type=int, default=10000, help='Number of examples')
    parser.add_argument('--output', type=str, help='Output pickle file path')
    parser.add_argument('--pace-lookup', type=str, help='Path to pace_lookup.pickle')
    parser.add_argument('--xgb-model', type=str, help='Path to XGBoost model')
    parser.add_argument('--xgb-features', type=str, help='Path to XGBoost feature columns')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--parallel', action='store_true', help='Run models in parallel')
    parser.add_argument('--num-gpus', type=int, help='Number of GPUs to use')
    parser.add_argument('--gpu-id', type=int, help='Specific GPU ID to use')
    parser.add_argument('--gpu-list', type=str, help='Comma-separated list of GPU IDs to use (e.g., "0,2,4,6")')
    parser.add_argument('--model-index', type=int, help='Specific model index (for job arrays)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--save-full-distributions', action='store_true', 
                      help='Save full bin_probabilities in raw predictions (makes files much larger, disabled by default)')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        input_glob = config['input_glob']
        models = [tuple(m.values()) for m in config['models']]
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
        
        # Parse models argument
        def parse_models_arg(models_str):
            models = []
            for model_str in models_str.split(','):
                parts = model_str.split(':')
                if len(parts) != 4:
                    raise ValueError(f"Invalid model spec: {model_str}")
                models.append(tuple(parts))
            return models
        
        input_glob = args.input_glob
        models = parse_models_arg(args.models)
        num_examples = args.num_examples
        output_path = args.output
        pace_lookup_path = args.pace_lookup
        xgb_model_path = args.xgb_model
        xgb_feature_columns_path = args.xgb_features
        device = args.device
        random_seed = args.random_seed
    
    # Parse GPU list if provided
    gpu_list = None
    if args.gpu_list:
        gpu_list = [int(x.strip()) for x in args.gpu_list.split(',')]
    
    evaluate_models_parallel(
        input_glob=input_glob,
        models=models,
        num_examples=num_examples,
        output_path=output_path,
        pace_lookup_path=pace_lookup_path,
        xgb_model_path=xgb_model_path,
        xgb_feature_columns_path=xgb_feature_columns_path,
        device=device,
        parallel=args.parallel,
        num_gpus=args.num_gpus,
        gpu_id=args.gpu_id,
        gpu_list=gpu_list,
        model_index=args.model_index,
        random_seed=random_seed,
        save_full_distributions=args.save_full_distributions
    )


if __name__ == '__main__':
    main()

