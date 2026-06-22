# Models Directory

## ChatTS Checkpoints

ChatTS model weights are not included in this repository (too large for git).
Download from HuggingFace and place in your VSC scratch directory:

```
$VSC_SCRATCH/ChatTS/ckpt/ChatTS-8B/
$VSC_SCRATCH/ChatTS/ckpt/ChatTS-14B/
$VSC_SCRATCH/ChatTS/ckpt/chronos-t5-small/
```

- **ChatTS-8B / ChatTS-14B**: https://huggingface.co/bytedance-research/ChatTS-14B
- **Chronos-T5-small**: https://huggingface.co/amazon/chronos-t5-small

## MLP Segmentation Model

The trained MLP checkpoint (`segmentation_mlp_final.pt`) is stored on VSC scratch:

```
$VSC_SCRATCH/ChatTS/timeseer_data/segmentation_mlp_final.pt
```

### Final model performance (threshold=0.65)

| Metric    | Value |
|-----------|-------|
| val_loss  | 0.5372 |
| Accuracy  | 88.0% |
| Precision | 0.752 |
| Recall    | 0.653 |
| F1        | 0.699 |

Trained on: combined industrial (14 labelled signals) + synthetic data, 150 epochs.

### Loading the MLP

```python
from src.models.mlp import load_mlp
mlp = load_mlp('/path/to/segmentation_mlp_final.pt')
```

### Training from scratch

```bash
sbatch vsc/slurm_train_mlp.sh
# or locally:
python scripts/train_mlp.py --data combined --epochs 150
```
