#!/bin/bash

# --- 1. System Updates ---
echo "Updating system packages..."
# We skip the heavy upgrade to avoid the "Pink Screen" service prompts
sudo apt-get update

# --- 2. Install Python Dependencies ---
echo "Installing Python dependencies..."
# We use --user to avoid permission issues and ensure we override system packages
pip install --user --upgrade pip
pip install --user --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --user "numpy>=2.0.0" wandb scipy pandas xgboost scikit-learn matplotlib ipykernel

# --- 3. Login to WandB ---
echo "Logging into Weights & Biases..."
wandb login

# --- 4. Verify GPU and Environment ---
echo "Verifying environment..."
python3 -c "import torch; import numpy; print(f'Torch Version: {torch.__version__}'); print(f'NumPy Version: {numpy.__version__}'); print(f'GPU Available: {torch.cuda.is_available()}')"

echo "------------------------------------------------"
echo "SETUP COMPLETE."
echo "If GPU is True and versions look correct, run:"
echo "bash run_script_cross_entropy.sh"
echo "------------------------------------------------"
