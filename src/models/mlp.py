"""
MLP classifier for the segmentation pipeline.

Architecture: Chronos-T5-small embedding (512-dim) concatenated with
ChatTS-14B embed_tokens output (5120-dim) → Linear(5632,128) → ReLU →
Dropout(0.3) → Linear(128,1).

Training uses BCEWithLogitsLoss (no sigmoid in forward).
Inference applies torch.sigmoid() to logits.

Final model metrics (threshold=0.65, combined industrial+synthetic, 150 epochs):
  val_loss=0.5372  accuracy=0.880  precision=0.752  recall=0.653  F1=0.699
"""

import os
import torch
import torch.nn as nn


class TSAISegmentationMLP(nn.Module):
    """
    Binary segmentation MLP.

    Parameters
    ----------
    ts_dim   : Chronos-T5-small hidden dim (512)
    text_dim : ChatTS-14B embed_tokens dim  (5120)
    """

    def __init__(self, ts_dim: int = 512, text_dim: int = 5120) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ts_dim + text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            # No sigmoid - BCEWithLogitsLoss during training.
            # Apply torch.sigmoid() at inference.
        )

    def forward(self, ts_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ts_emb  : [batch, 512]  - Chronos window embeddings
        txt_emb : [1, 5120] or [batch, 5120]

        Returns
        -------
        logits : [batch, 1]
        """
        combined = torch.cat([ts_emb, txt_emb], dim=1)
        return self.net(combined)


def load_mlp(checkpoint_path: str, ts_dim: int = 512, text_dim: int = 5120) -> TSAISegmentationMLP:
    """Load a saved MLP from a .pt checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state', ckpt)
    mlp = TSAISegmentationMLP(ts_dim=ts_dim, text_dim=text_dim)
    mlp.load_state_dict(state)
    mlp.eval()
    print(f'Loaded MLP from {checkpoint_path}')
    print(f'  Parameters : {sum(p.numel() for p in mlp.parameters()):,}')
    return mlp


def save_mlp(mlp: TSAISegmentationMLP, path: str, **metadata) -> None:
    """Save MLP state dict with optional metadata."""
    payload = {'model_state': mlp.state_dict(), **metadata}
    torch.save(payload, path)
    print(f'MLP saved → {path}')
