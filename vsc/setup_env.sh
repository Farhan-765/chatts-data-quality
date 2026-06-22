#!/bin/bash
# VSC environment setup script.
# Run once after cloning the repository on the VSC cluster.
# This creates both conda environments needed for the project.

SCRATCH=/scratch/leuven/375/vsc37531
PROJECT_DIR=$SCRATCH/ChatTS/chatts-anomaly-detection

echo "=============================================="
echo "Setting up chatts-anomaly-detection on VSC"
echo "=============================================="

# ── 1. Create main chatts environment ────────────────────────────
echo ""
echo "[1/3] Creating chatts conda environment..."
conda env create -f $PROJECT_DIR/environment.yml
# Note: do NOT install chronos-forecasting here — accelerate conflict

# ── 2. Create chronos subprocess environment ──────────────────────
echo ""
echo "[2/3] Creating chatts_chronos conda environment..."
conda env create -f $PROJECT_DIR/environment_chronos.yml

echo ""
echo "[3/3] Installing chronos-forecasting in chatts_chronos..."
conda run -n chatts_chronos pip install chronos-forecasting

# ── 3. Create required directories ───────────────────────────────
echo ""
echo "Creating directories..."
mkdir -p $SCRATCH/ChatTS/ckpt
mkdir -p $SCRATCH/ChatTS/timeseer_data
mkdir -p $SCRATCH/ChatTS/figures
mkdir -p $SCRATCH/ChatTS/logs
mkdir -p $SCRATCH/cache/huggingface
mkdir -p $SCRATCH/cache/hf_modules
mkdir -p $SCRATCH/cache/transformers

# ── 4. Copy .env ──────────────────────────────────────────────────
if [ ! -f $PROJECT_DIR/.env ]; then
    cp $PROJECT_DIR/.env.example $PROJECT_DIR/.env
    echo ""
    echo "Created .env from .env.example"
    echo "IMPORTANT: Edit .env and add your TIMESEER_API_TOKEN and OPENAI_API_KEY"
fi

echo ""
echo "=============================================="
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. Download ChatTS checkpoints to: $SCRATCH/ChatTS/ckpt/"
echo "     - ChatTS-8B/"
echo "     - ChatTS-14B/"
echo "     - chronos-t5-small/"
echo "  2. Edit .env with your API credentials"
echo "  3. Submit SLURM job: sbatch vsc/slurm_chatts14b.sh"
echo "=============================================="
