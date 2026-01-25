#!/usr/bin/env bash
set -euo pipefail

# Setup script intended for fresh GPU machines (e.g., Lambda).
#
# This script:
# - Creates a venv at repo root: .venv/
# - Installs repo requirements
# - Verifies CUDA availability
#
# Usage:
#   bash train/setup_cloud.sh
#
# Notes:
# - If you want W&B logging, set WANDB_API_KEY in the environment before training.
# - This script does NOT run training automatically.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[setup_cloud] ROOT_DIR=${ROOT_DIR}"

echo "[setup_cloud] Updating system packages (apt-get update only)..."
sudo apt-get update -y

echo "[setup_cloud] Creating venv (.venv)..."
python3 -m venv .venv

echo "[setup_cloud] Installing python deps..."
. .venv/bin/activate
python -m pip install --upgrade pip wheel

# Prefer official PyTorch CUDA wheels when on GPU boxes. If this fails, fall back to
# requirements.txt (which may install CPU wheels).
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[setup_cloud] GPU detected; installing PyTorch CUDA wheels..."
  python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

python -m pip install -r requirements.txt

echo "[setup_cloud] Verifying environment..."
python - <<'PY'
import torch
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

echo "[setup_cloud] Done."
