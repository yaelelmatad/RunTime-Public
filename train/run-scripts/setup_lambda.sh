#!/bin/bash

# Setup script for Lambda Cloud node
# This installs all necessary dependencies for running the ablation training

set -e  # Exit on error

echo "=========================================="
echo "Setting up Lambda Cloud node for RunTime Ablation Training"
echo "=========================================="
echo ""

# --- 1. System Updates ---
echo "[1/5] Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-dev git

# --- 2. Install Python Dependencies ---
echo "[2/5] Installing Python dependencies..."
pip3 install --user --upgrade pip

# Install PyTorch with CUDA support
echo "  Installing PyTorch with CUDA 12.1..."
pip3 install --user --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "  Installing other Python packages..."
pip3 install --user --upgrade "numpy>=1.24,<3.0" wandb pyyaml scipy pandas xgboost scikit-learn matplotlib ipykernel optuna

# --- 3. Verify GPU and Environment ---
echo "[3/5] Verifying environment..."
python3 -c "
import torch
import numpy
print(f'✓ Torch Version: {torch.__version__}')
print(f'✓ NumPy Version: {numpy.__version__}')
print(f'✓ GPU Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'✓ CUDA Version: {torch.version.cuda}')
"

# --- 4. Check Data Files ---
echo "[4/5] Checking data files..."
# Check relative to current directory (Runtime/train/run-scripts)
if [ -f "../../data/pace_lookup.pickle" ]; then
    echo "  ✓ pace_lookup.pickle found at ../../data/pace_lookup.pickle"
elif [ -f "/lambda/nfs/runtime-sweeper-datastore-arizona/nyrr-transform/Runtime/data/pace_lookup.pickle" ]; then
    echo "  ✓ pace_lookup.pickle found at /lambda/nfs/runtime-sweeper-datastore-arizona/nyrr-transform/Runtime/data/pace_lookup.pickle"
else
    echo "  ✗ WARNING: pace_lookup.pickle not found"
fi

if [ -d "../../pipeline/training_splits" ] && [ "$(ls -A ../../pipeline/training_splits/*.pkl.gz 2>/dev/null | wc -l)" -gt 0 ]; then
    SPLIT_COUNT=$(ls -1 ../../pipeline/training_splits/*.pkl.gz 2>/dev/null | wc -l)
    echo "  ✓ Found $SPLIT_COUNT training split files in ../../pipeline/training_splits/"
elif [ -d "/lambda/nfs/runtime-sweeper-datastore-arizona/nyrr-transform/Runtime/pipeline/training_splits" ] && [ "$(ls -A /lambda/nfs/runtime-sweeper-datastore-arizona/nyrr-transform/Runtime/pipeline/training_splits/*.pkl.gz 2>/dev/null | wc -l)" -gt 0 ]; then
    SPLIT_COUNT=$(ls -1 /lambda/nfs/runtime-sweeper-datastore-arizona/nyrr-transform/Runtime/pipeline/training_splits/*.pkl.gz 2>/dev/null | wc -l)
    echo "  ✓ Found $SPLIT_COUNT training split files in /lambda/nfs/runtime-sweeper-datastore-arizona/nyrr-transform/Runtime/pipeline/training_splits/"
else
    echo "  ✗ WARNING: No training split files found"
fi

# --- 5. WandB Setup (optional) ---
echo "[5/5] WandB setup..."
if [ -z "$WANDB_API_KEY" ]; then
    echo "  ⚠ WANDB_API_KEY not set in environment"
    echo "  You can set it with: export WANDB_API_KEY=your_key_here"
    echo "  Or run: wandb login"
else
    echo "  ✓ WANDB_API_KEY found in environment"
fi

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "To run the ablation training:"
echo "  bash run-scripts/run_ablation_lambda.sh"
echo ""
echo "Or manually:"
echo "  cd /lambda/nfs/runtime-sweeper-datastore-arizona/nyrr-transform/Runtime/train"
echo "  python3 runtime_trainer_ablation.py --config runtime_trainer_time_token_ablation.yaml"
echo ""

