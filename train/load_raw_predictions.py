#!/usr/bin/env python3
"""
Helper script to load and inspect raw prediction files.

Usage:
    python load_raw_predictions.py path/to/model_name_raw_predictions.pickle
"""

import pickle
import numpy as np
import argparse
from pathlib import Path


def load_raw_predictions(file_path):
    """Load raw predictions file and display summary."""
    print(f"Loading raw predictions from: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print("\n" + "="*80)
    print("RAW PREDICTIONS SUMMARY")
    print("="*80)
    print(f"Model name: {data['model_name']}")
    print(f"Display name: {data['display_name']}")
    print(f"Number of examples: {data['num_examples']}")
    print(f"Number of bins: {data['metadata']['num_bins']}")
    print(f"Checkpoint: {data['metadata']['checkpoint_path']}")
    print(f"Config: {data['metadata']['config_path']}")
    
    print("\n" + "-"*80)
    print("PACE BINS")
    print("-"*80)
    print(f"Total bins: {len(data['pace_bins'])}")
    print(f"First bin: {data['pace_bins'][0]}")
    print(f"Last bin: {data['pace_bins'][-1]}")
    
    print("\n" + "-"*80)
    print("EXAMPLE PREDICTIONS")
    print("-"*80)
    predictions = data['predictions']
    
    if len(predictions) > 0:
        print(f"\nFirst example:")
        ex0 = predictions[0]
        print(f"  Example index: {ex0['example_idx']}")
        print(f"  Actual pace: {ex0['actual_pace']:.2f}s")
        print(f"  Weighted mean: {ex0['weighted_mean']:.2f}s")
        print(f"  Weighted median: {ex0['weighted_median']:.2f}s")
        print(f"  Mode pace: {ex0['mode_pace']:.2f}s")
        print(f"  Mode bin index: {ex0['mode_bin_idx']}")
        print(f"  Percentile: {ex0['percentile']:.2f}")
        print(f"  Bin probabilities: {len(ex0['bin_probabilities'])} non-zero bins")
        print(f"  Top 5 bins by probability:")
        sorted_bins = sorted(ex0['bin_probabilities'].items(), key=lambda x: x[1], reverse=True)[:5]
        for bin_idx, prob in sorted_bins:
            bin_info = data['pace_bins'][bin_idx]
            print(f"    Bin {bin_idx}: prob={prob:.4f}, range=[{bin_info['start']:.1f}, {bin_info['end']:.1f})s, median={bin_info['median']:.1f}s")
        
        if len(predictions) > 1:
            print(f"\nLast example:")
            ex_last = predictions[-1]
            print(f"  Example index: {ex_last['example_idx']}")
            print(f"  Actual pace: {ex_last['actual_pace']:.2f}s")
            print(f"  Weighted mean: {ex_last['weighted_mean']:.2f}s")
            print(f"  Weighted median: {ex_last['weighted_median']:.2f}s")
            print(f"  Mode pace: {ex_last['mode_pace']:.2f}s")
            print(f"  Percentile: {ex_last['percentile']:.2f}")
    
    # Statistics
    print("\n" + "-"*80)
    print("STATISTICS")
    print("-"*80)
    actual_paces = np.array([p['actual_pace'] for p in predictions])
    weighted_means = np.array([p['weighted_mean'] for p in predictions])
    weighted_medians = np.array([p['weighted_median'] for p in predictions])
    percentiles = np.array([p['percentile'] for p in predictions])
    
    print(f"Actual paces: mean={np.mean(actual_paces):.2f}s, std={np.std(actual_paces):.2f}s")
    print(f"Weighted means: mean={np.mean(weighted_means):.2f}s, std={np.std(weighted_means):.2f}s")
    print(f"Weighted medians: mean={np.mean(weighted_medians):.2f}s, std={np.std(weighted_medians):.2f}s")
    print(f"Percentiles: mean={np.mean(percentiles):.2f}, std={np.std(percentiles):.2f}")
    
    # Check probability distributions
    all_probs_sums = [sum(p['bin_probabilities'].values()) for p in predictions]
    print(f"\nProbability distributions:")
    print(f"  Mean sum per example: {np.mean(all_probs_sums):.6f}")
    print(f"  Min sum: {np.min(all_probs_sums):.6f}")
    print(f"  Max sum: {np.max(all_probs_sums):.6f}")
    print(f"  Mean non-zero bins per example: {np.mean([len(p['bin_probabilities']) for p in predictions]):.1f}")
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Load and inspect raw prediction files')
    parser.add_argument('file_path', type=str, help='Path to raw predictions pickle file')
    parser.add_argument('--example-idx', type=int, help='Show details for specific example index')
    
    args = parser.parse_args()
    
    data = load_raw_predictions(args.file_path)
    
    if args.example_idx is not None:
        predictions = data['predictions']
        if args.example_idx < len(predictions):
            ex = predictions[args.example_idx]
            print("\n" + "="*80)
            print(f"DETAILED VIEW: Example {args.example_idx}")
            print("="*80)
            print(f"Example index: {ex['example_idx']}")
            print(f"Actual pace: {ex['actual_pace']:.2f}s")
            print(f"Weighted mean: {ex['weighted_mean']:.2f}s")
            print(f"Weighted median: {ex['weighted_median']:.2f}s")
            print(f"Mode pace: {ex['mode_pace']:.2f}s")
            print(f"Mode bin index: {ex['mode_bin_idx']}")
            print(f"Percentile: {ex['percentile']:.2f}")
            print(f"\nFull probability distribution ({len(ex['bin_probabilities'])} non-zero bins):")
            sorted_bins = sorted(ex['bin_probabilities'].items(), key=lambda x: x[1], reverse=True)
            for bin_idx, prob in sorted_bins:
                if prob > 0.001:  # Only show bins with >0.1% probability
                    bin_info = data['pace_bins'][bin_idx]
                    print(f"  Bin {bin_idx:3d}: prob={prob:7.4f}, range=[{bin_info['start']:6.1f}, {bin_info['end']:6.1f})s, median={bin_info['median']:6.1f}s")
        else:
            print(f"Error: Example index {args.example_idx} >= {len(predictions)}")


if __name__ == '__main__':
    main()

