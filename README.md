# ChatTS Industrial Anomaly Detection

MSc thesis research: evaluating **ChatTS-14B** (a time-series large language model)
for industrial sensor anomaly detection, compared against **GPT-4o** and a statistical
baseline.

Research site: **Timeseer / UHasselt** — 55 industrial sensor signals across 7 plant
areas, 90 days of data at 15-minute sampling intervals.

---

## Research Overview

### Three Approaches

| Approach | Method | Description |
|----------|--------|-------------|
| 1 | Statistical Only | Pre-screener classifies without any LLM |
| 2 | Stats Embedded | ChatTS-14B + signal statistics in prompt, middle 512pts |
| 3 | **Hybrid (best)** | Pre-screen → chunk selection → embedded stats → MCQ template |

### Anomaly Taxonomy

| Code | Name           | Description |
|------|----------------|-------------|
| A    | Drift          | Slow baseline migration over weeks |
| B    | Spikes         | Sudden transient outliers |
| C    | Frozen/Stale   | Signal flatlines at constant value |
| D    | Phase Change   | Abrupt permanent level shift |
| E    | None/Clean     | No anomaly |
| G    | Var Collapse   | Intermittent amplitude collapse |
| L    | Intermittent   | Physically impossible values |
| B+L  | Composite      | Spikes + intermittent |

### Segmentation Pipeline

Chronos-T5-small (512-dim) + ChatTS-14B `embed_tokens` (5120-dim) → MLP (5632→128→1).

- **Final model** (150 epochs, threshold=0.65, combined industrial+synthetic data):
  F1=0.699, Accuracy=88.0%, Precision=0.752, Recall=0.653
- **Multi-scale**: window sizes [32, 64, 128, 256], max-pooling per timestep

---

## Repository Structure

```
chatts-anomaly-detection/
├── configs/
│   ├── signals.yaml          # All 55 signal tags with ground truth labels
│   ├── prescreener.yaml      # Statistical pre-screener thresholds
│   ├── segmentation.yaml     # MLP architecture + final metrics
│   └── vsc.yaml              # VSC cluster paths
├── src/
│   ├── data/
│   │   ├── timeseer_client.py  # Timeseer REST API (fetch_series_api, list_series_api)
│   │   ├── csv_loader.py       # Local CSV loader for offline use
│   │   └── ground_truth.py     # GROUND_TRUTH dict + AREA_TAGS (55 signals)
│   ├── preprocessing/
│   │   ├── chunking.py         # get_chunk, get_time_chunk, downsample
│   │   └── windows.py          # extract_windows, build_training_sample
│   ├── prescreener/
│   │   └── analyze.py          # analyze_signal (Approach 1), hybrid_analyze (Approach 3)
│   ├── prompts/
│   │   ├── templates.py        # MCQ_CATEGORIES, MCQ_MULTI, MCQ_STALE_FOCUSED
│   │   └── builder.py          # build_smart_question (Approach 2)
│   ├── models/
│   │   ├── chatts_loader.py    # load_model (8B or 14B)
│   │   └── mlp.py              # TSAISegmentationMLP (ts_dim=512, text_dim=5120)
│   ├── inference/
│   │   ├── chatts_infer.py     # ask_chatts_chunk
│   │   ├── openai_infer.py     # ask_openai_chunk, ask_openai_naive, ask_openai_mcq_only
│   │   ├── chronos_server.py   # Persistent Chronos subprocess + cache
│   │   └── embeddings.py       # extract_chatts_embedding, encode_text_query
│   ├── parsing/
│   │   └── extract.py          # extract_category (A-E, G, L, B+L)
│   ├── evaluation/
│   │   ├── metrics.py          # compute_metrics, threshold_sweep
│   │   └── report.py           # print_summary_table, save_results
│   ├── segmentation/
│   │   ├── pipeline.py         # segment_signal, multiscale_segmentation
│   │   ├── training.py         # train_mlp (consolidated from 6 notebook cells)
│   │   └── cache.py            # load_cache, save_cache
│   └── visualization/
│       ├── segmentation_plots.py  # plot_segmentation, plot_segmentation_heatmap
│       └── thesis_figures.py      # plot_approach_comparison, plot_confusion_matrix
├── scripts/
│   ├── run_approach3.py        # Batch Approach 3 on any plant area
│   ├── run_segmentation.py     # Multi-scale segmentation on one signal
│   ├── train_mlp.py            # Train MLP from cached embeddings
│   └── evaluate_all.py         # Evaluate all approaches across all 55 signals
├── vsc/
│   ├── slurm_chatts14b.sh     # SLURM job for Approach 3 on all areas
│   ├── slurm_train_mlp.sh     # SLURM job for MLP training
│   └── setup_env.sh           # One-time environment setup on VSC
├── notebooks/
│   ├── exploration/           # Archive of original 153-cell research notebook
│   ├── final_experiments/     # Clean reproducible notebooks
│   │   ├── 01_approach1_statistical_only.ipynb
│   │   ├── 03_approach3_hybrid.ipynb
│   │   └── 05_segmentation_pipeline.ipynb
│   └── figures/
├── data/                      # Results + README (raw data not included)
├── models/                    # README (checkpoints not included)
├── figures/                   # Generated plots
├── configs/
├── .env.example               # Template for API credentials
├── environment.yml            # Main chatts conda environment (Python 3.11)
├── environment_chronos.yml    # Chronos subprocess environment (no accelerate)
└── requirements.txt           # Pip dependencies
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd chatts-anomaly-detection

# On VSC cluster:
bash vsc/setup_env.sh

# Locally (CPU only, for development):
conda env create -f environment.yml
conda activate chatts
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env — add your TIMESEER_API_TOKEN and OPENAI_API_KEY
```

### 3. Download model checkpoints

Place in `$VSC_SCRATCH/ChatTS/ckpt/`:
- `ChatTS-14B/` — from https://huggingface.co/bytedance-research/ChatTS-14B
- `chronos-t5-small/` — from https://huggingface.co/amazon/chronos-t5-small

### 4. Run Approach 3 (best approach)

```bash
# Interactive (in a notebook):
# Open notebooks/final_experiments/03_approach3_hybrid.ipynb

# CLI batch run:
python scripts/run_approach3.py --area "Reactor 1" --model ChatTS-14B

# SLURM job (all 7 areas):
sbatch vsc/slurm_chatts14b.sh
```

### 5. Run segmentation pipeline

```bash
# Start Chronos server first (automatic in script):
python scripts/run_segmentation.py \
    --tag R1-AT-102-COND \
    --area "Reactor 1" \
    --query "Identify temperature spikes"
```

---

## Key Technical Details

### Chronos isolation

Chronos-T5-small and ChatTS both require `accelerate` but at incompatible
versions. Solution: Chronos runs in a **separate conda environment** (`chatts_chronos`)
via a persistent subprocess server that communicates over stdin/stdout.

```python
from src.inference.chronos_server import start_server, get_chronos_embedding_cached
start_server()  # launches python3.11 /tmp/chronos_server.py
emb = get_chronos_embedding_cached(signal_array)  # np.ndarray (512,)
```

### Spike detection — self-contamination fix

Using `.rolling(10).std().shift(5)` instead of contemporaneous rolling std.
This prevents the spike itself from inflating its own detection threshold.

### Seasonality filter for drift detection

`roll_mean_std > 0.3 * global_std` prevents daily sinusoidal cycles (R2-AT-206-DENS,
R2-AT-203-PH) from being classified as drift.

### SKIP_SPIKE_TAGS

Signals with large daily cycles (ORP, PRESS, TEMP, DENS, VISC, etc.) are excluded
from spike detection because their peaks exceed the 4σ threshold but are not anomalies.

---

## Environment Variables

| Variable             | Required | Description |
|----------------------|----------|-------------|
| `TIMESEER_API_TOKEN` | Yes      | Base64-encoded API token |
| `TIMESEER_BASE_URL`  | Yes      | Your Timeseer instance URL |
| `TIMESEER_TENANT`    | Yes      | Your Timeseer tenant name |
| `OPENAI_API_KEY`     | For GPT-4o comparisons | OpenAI API key |
| `VSC_SCRATCH`        | On VSC   | Scratch directory path |

---

## Citation

If you use this codebase or the thesis results, please cite:

```
[MSc Thesis] Multimodal modeling for industrial IoT sensor data and language
Author: Muhammad Farhan
Institution: UHasselt / KU Leuven
Year: 2026
```

---

## License

Research code — not licensed for production use without permission.
