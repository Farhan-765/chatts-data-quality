#!/bin/bash
#SBATCH --job-name=chatts14b_approach3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/leuven/375/vsc37531/ChatTS/logs/chatts14b_%j.out
#SBATCH --error=/scratch/leuven/375/vsc37531/ChatTS/logs/chatts14b_%j.err

# ── Environment setup ─────────────────────────────────────────────
SCRATCH=/scratch/leuven/375/vsc37531
PROJECT_DIR=$SCRATCH/ChatTS/chatts-anomaly-detection

source $SCRATCH/miniforge3/etc/profile.d/conda.sh
conda activate chatts

cd $PROJECT_DIR

# ── Env vars required by ChatTS ───────────────────────────────────
export TRANSFORMERS_NO_FLASH_ATTENTION=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$SCRATCH/cache/huggingface
export HF_MODULES_CACHE=$SCRATCH/cache/hf_modules
export TRANSFORMERS_CACHE=$SCRATCH/cache/transformers
export TF_ENABLE_ONEDNN_OPTS=0
export VSC_SCRATCH=$SCRATCH

# ── Run Approach 3 on all 7 plant areas ───────────────────────────
AREAS=(
    "Reactor 1"
    "Reactor 2"
    "Separator"
    "Distillation"
    "Wastewater Treatment"
    "Utilities"
    "Packaging"
)

for AREA in "${AREAS[@]}"; do
    echo "======================================"
    echo "Running Approach 3 on: $AREA"
    echo "======================================"
    python scripts/run_approach3.py \
        --area "$AREA" \
        --model ChatTS-14B \
        --out "data/chatts14b_approach3_${AREA// /_}.txt"
done

echo "Done."
