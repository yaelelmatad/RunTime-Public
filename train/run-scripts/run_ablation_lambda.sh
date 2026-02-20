#!/bin/bash

# Run script for Lambda Cloud node
# This runs the ablation training with the production config

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."  # Go up to Runtime/train directory

CONFIG_FILE="runtime_trainer_time_token_ablation.yaml"
TRAINER_SCRIPT="runtime_trainer_ablation.py"

echo "=========================================="
echo "RunTime Ablation Training"
echo "=========================================="
echo ""
echo "Config: $CONFIG_FILE"
echo "Trainer: $TRAINER_SCRIPT"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if trainer script exists
if [ ! -f "$TRAINER_SCRIPT" ]; then
    echo "ERROR: Trainer script not found: $TRAINER_SCRIPT"
    exit 1
fi

# Check if data files exist
if [ ! -f "../data/pace_lookup.pickle" ]; then
    echo "ERROR: pace_lookup.pickle not found at ../data/pace_lookup.pickle"
    exit 1
fi

if [ ! -d "../pipeline/training_splits" ]; then
    echo "ERROR: training_splits directory not found at ../pipeline/training_splits"
    exit 1
fi

# Verify GPU availability
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'" || {
    echo "ERROR: CUDA is not available. Please check your GPU setup."
    exit 1
}

echo "✓ All checks passed"
echo ""
echo "Starting training..."
echo ""

# Run the training
# Add ~/.local/bin to PATH for user-installed packages
export PATH="$HOME/.local/bin:$PATH"

# Run with nohup to allow disconnection, or run directly
# For direct run (recommended when using screen/tmux):
python3 "$TRAINER_SCRIPT" --config "$CONFIG_FILE"

# Alternative: Run in background with logging (uncomment if needed):
# nohup python3 "$TRAINER_SCRIPT" --config "$CONFIG_FILE" > training.log 2>&1 &
# echo "Training started in background. Logs: training.log"
# echo "To monitor: tail -f training.log"

