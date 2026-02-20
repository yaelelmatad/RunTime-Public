#!/bin/bash
# Script to run parallel model evaluation on Lambda cluster
# 
# Usage:
#   # Run all models in parallel (one per GPU, using GPUs 0,1,2,3...)
#   sbatch --array=0-$(($NUM_MODELS-1)) run_evaluate_parallel_lambda.sh
#
#   # Run with specific GPU list (e.g., use GPUs 0,2,4,6)
#   export GPU_LIST="0,2,4,6"
#   sbatch --array=0-$(($NUM_MODELS-1)) run_evaluate_parallel_lambda.sh
#
#   # Or specify GPUs directly in the command
#   GPU_LIST="1,3,5,7" sbatch --array=0-3 run_evaluate_parallel_lambda.sh

#SBATCH --job-name=eval_models
#SBATCH --output=eval_models_%A_%a.out
#SBATCH --error=eval_models_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load environment
source ~/.bashrc
conda activate runtime  # Adjust to your conda env name

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Config file
CONFIG_FILE="evaluation_config_cluster.yaml"  # Update this path

# Output directory for individual model results
OUTPUT_DIR="evaluation_results_parallel"
mkdir -p "$OUTPUT_DIR"

# Determine GPU ID to use
# If GPU_LIST is set, map array task ID to GPU from the list
# Otherwise, use array task ID directly as GPU ID
if [ -n "$GPU_LIST" ]; then
    # Parse GPU list into array
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_LIST"
    NUM_GPUS=${#GPU_ARRAY[@]}
    
    # Map array task ID to GPU (round-robin)
    GPU_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_GPUS))
    GPU_ID=${GPU_ARRAY[$GPU_INDEX]}
    
    echo "Using GPU list: $GPU_LIST"
    echo "Array task ID: $SLURM_ARRAY_TASK_ID"
    echo "Mapped to GPU: $GPU_ID"
    
    # Set CUDA_VISIBLE_DEVICES to make only the desired GPU visible
    # This ensures the script uses the correct GPU even if SLURM assigned a different one
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    # When CUDA_VISIBLE_DEVICES is set, the script should use GPU 0 (which is now the mapped GPU)
    ACTUAL_GPU_ID=0
else
    # Use array task ID directly as GPU ID
    GPU_ID=$SLURM_ARRAY_TASK_ID
    ACTUAL_GPU_ID=$GPU_ID
    echo "Using GPU: $GPU_ID (from array task ID)"
fi

# Run evaluation for this model index
python evaluate_models_parallel.py \
    --config "$CONFIG_FILE" \
    --device cuda \
    --gpu-id $ACTUAL_GPU_ID \
    --model-index $SLURM_ARRAY_TASK_ID \
    --output "$OUTPUT_DIR/model_${SLURM_ARRAY_TASK_ID}.pickle"

if [ -n "$GPU_LIST" ]; then
    echo "Model $SLURM_ARRAY_TASK_ID evaluation complete (used physical GPU $GPU_ID, visible as GPU $ACTUAL_GPU_ID)"
else
    echo "Model $SLURM_ARRAY_TASK_ID evaluation complete (used GPU $GPU_ID)"
fi

