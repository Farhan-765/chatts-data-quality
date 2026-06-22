#!/bin/bash
#SBATCH --job-name=train_mlp_combined
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/leuven/375/vsc37531/ChatTS/logs/train_mlp_%j.out
#SBATCH --error=/scratch/leuven/375/vsc37531/ChatTS/logs/train_mlp_%j.err

# ── Environment setup ─────────────────────────────────────────────
SCRATCH=/scratch/leuven/375/vsc37531
PROJECT_DIR=$SCRATCH/ChatTS/chatts-anomaly-detection

source $SCRATCH/miniforge3/etc/profile.d/conda.sh
conda activate chatts

cd $PROJECT_DIR

export VSC_SCRATCH=$SCRATCH
export CUDA_VISIBLE_DEVICES=0

echo "Training MLP segmentation model..."
echo "Data: combined industrial + synthetic"
echo "Epochs: 150"

python scripts/train_mlp.py \
    --data combined \
    --epochs 150 \
    --batch 64 \
    --lr 0.001 \
    --out $SCRATCH/ChatTS/timeseer_data/segmentation_mlp_final.pt

echo "Training complete."
