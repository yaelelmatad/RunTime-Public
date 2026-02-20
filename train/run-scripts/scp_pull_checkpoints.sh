#!/bin/bash

# Script to pull down checkpoint folders from Lambda instance
# Remote location: ubuntu@129-151-27-10:~/runtime-sweeper-datastore-arizona/nyrr-transform/Runtime/train/checkpoints_clean_prod/
# Local destination: Runtime/train/ablation-studies/

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the train directory (one level up from run-scripts)
TRAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE_HOST="ubuntu@129.151.27.10"
REMOTE_BASE_PATH="~/runtime-sweeper-datastore-arizona/nyrr-transform/Runtime/train/checkpoints_clean_prod"
# Destination: Runtime/train/ablation-studies (relative to train directory)
LOCAL_DEST="$TRAIN_DIR/ablation-studies"

# List of folders to pull down (Corrected versions)
FOLDERS=(
    "Production_Scale_v2_HighCap_Corrected"
    "Production_Scale_v2_HighCap_SwappedOrder_Corrected"
    "Production_Scale_v2_HighCap_Ablation_NoTime_AgeLastFront_Corrected"
    "Production_Scale_v2_HighCap_Ablation_NoTime_AgeLastFront_Shuffled_Corrected"
    "Production_Scale_v2_HighCap_SwappedOrder_Corrected_Lambda_1"
    "Production_Scale_v2_HighCap_SwappedOrder_Corrected_Lambda_2"
    "Production_Scale_v2_HighCap_SwappedOrder_Corrected_Lambda_4"
    "Production_Scale_v2_HighCap_SwappedOrder_Corrected_Lambda_5"
)

# Create local destination directory if it doesn't exist
mkdir -p "$LOCAL_DEST"

echo "Pulling down checkpoint folders from $REMOTE_HOST..."
echo "Destination: $LOCAL_DEST"
echo ""

# Pull down each folder
for folder in "${FOLDERS[@]}"; do
    echo "Pulling $folder..."
    scp -r "${REMOTE_HOST}:${REMOTE_BASE_PATH}/${folder}" "$LOCAL_DEST/"
    if [ $? -eq 0 ]; then
        echo "✓ Successfully pulled $folder"
    else
        echo "✗ Failed to pull $folder"
    fi
    echo ""
done

echo "Done! All folders pulled to $LOCAL_DEST"

