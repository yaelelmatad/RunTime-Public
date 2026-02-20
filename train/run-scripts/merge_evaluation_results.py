#!/usr/bin/env python3
"""
Merge individual model evaluation results into a single pickle file.
Run this after all parallel evaluation jobs complete.
"""

import pickle
import argparse
from pathlib import Path
from collections import defaultdict

def merge_results(result_files, output_path):
    """Merge multiple evaluation result files into one."""
    print(f"Merging {len(result_files)} result files...")
    
    # Load all results
    all_results = {}
    all_percentiles = {}
    all_calibration = {}
    all_history = defaultdict(lambda: defaultdict(list))
    all_week_delta = defaultdict(lambda: defaultdict(list))
    
    # Shared data (use from first file)
    mae_examples = None
    mae_runners_list = None
    xgb_successful_indices = None
    
    for i, result_file in enumerate(result_files):
        print(f"  Loading {result_file}...")
        with open(result_file, 'rb') as f:
            data = pickle.load(f)
        
        # Use shared data from first file
        if i == 0:
            mae_examples = data.get('mae_examples')
            mae_runners_list = data.get('mae_runners_list')
            xgb_successful_indices = data.get('xgb_successful_indices')
        
        # Merge model-specific results
        if 'results' in data:
            all_results.update(data['results'])
        
        if 'percentiles_by_model' in data:
            all_percentiles.update(data['percentiles_by_model'])
        
        if 'calibration_data' in data:
            all_calibration.update(data['calibration_data'])
        
        if 'results_by_history' in data:
            for hist_len, model_errors in data['results_by_history'].items():
                for model_name, errors in model_errors.items():
                    all_history[hist_len][model_name].extend(errors)
        
        if 'results_by_week_delta' in data:
            for week_delta, model_errors in data['results_by_week_delta'].items():
                for model_name, errors in model_errors.items():
                    all_week_delta[week_delta][model_name].extend(errors)
    
    # Save merged results
    print(f"Saving merged results to {output_path}...")
    merged_data = {
        'mae_examples': mae_examples,
        'mae_runners_list': mae_runners_list,
        'xgb_successful_indices': xgb_successful_indices,
        'results': all_results,
        'percentiles_by_model': all_percentiles,
        'results_by_history': dict(all_history),
        'results_by_week_delta': dict(all_week_delta),
        'calibration_data': all_calibration,
        'metadata': {
            'num_examples': len(mae_examples) if mae_examples else 0,
            'num_models': len(all_results),
            'merged_from': [str(f) for f in result_files]
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(merged_data, f)
    
    print(f"✓ Merged {len(all_results)} models into {output_path}")
    print(f"\nResults summary:")
    for model_name, model_results in all_results.items():
        print(f"  {model_results.get('display_name', model_name)}: "
              f"MAE={model_results.get('mae_median', 0):.2f}s, "
              f"RMSE={model_results.get('rmse_median', 0):.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Merge parallel evaluation results')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing individual model result files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output merged pickle file path')
    parser.add_argument('--pattern', type=str, default='model_*.pickle',
                       help='Glob pattern for result files (default: model_*.pickle)')
    
    args = parser.parse_args()
    
    # Find all result files
    input_dir = Path(args.input_dir)
    result_files = sorted(input_dir.glob(args.pattern))
    
    if not result_files:
        print(f"Error: No files found matching {args.pattern} in {args.input_dir}")
        return
    
    merge_results(result_files, args.output)


if __name__ == '__main__':
    main()


