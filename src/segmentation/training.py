"""
MLP training loop for the segmentation pipeline.
Consolidates the 6 duplicated training cells from the original notebook
into a single clean function.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.models.mlp import TSAISegmentationMLP, save_mlp


def train_mlp(
    X_ts: torch.Tensor,
    X_txt: torch.Tensor,
    y: torch.Tensor,
    save_path: str,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    train_split: float = 0.8,
    pos_weight: float | None = None,
    ts_dim: int = 512,
    text_dim: int = 5120,
) -> tuple[TSAISegmentationMLP, dict]:
    """
    Train an MLP segmentation model from scratch.

    Parameters
    ----------
    X_ts         : Tensor [N, 512]   - Chronos window embeddings
    X_txt        : Tensor [N, 5120]  - ChatTS text embeddings
    y            : Tensor [N, 1]     - binary labels (float)
    save_path    : path to save the best model checkpoint
    epochs       : total training epochs
    batch_size   : mini-batch size
    lr           : Adam learning rate
    weight_decay : L2 regularisation
    train_split  : fraction of data used for training
    pos_weight   : BCEWithLogitsLoss positive class weight
                   (default: computed from class balance)
    ts_dim, text_dim : MLP input dimensions

    Returns
    -------
    mlp    : trained TSAISegmentationMLP
    history: dict with 'train_loss', 'val_loss', 'val_f1', 'best_val_loss'
    """
    if pos_weight is None:
        frac = float(y.mean())
        pos_weight = (1 - frac) / (frac + 1e-8)
        print(f'Auto pos_weight: {pos_weight:.2f} (class balance: {frac:.4f})')

    mlp       = TSAISegmentationMLP(ts_dim=ts_dim, text_dim=text_dim)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    dataset  = TensorDataset(X_ts, X_txt, y)
    n_train  = int(train_split * len(dataset))
    n_val    = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f'Training: {n_train} samples | Validation: {n_val} samples | Epochs: {epochs}')
    print('=' * 55)

    history       = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(epochs):
        mlp.train()
        train_loss = 0.0
        for ts_b, txt_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = mlp(ts_b, txt_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        mlp.eval()
        val_loss   = 0.0
        val_preds  = []
        val_true   = []
        with torch.no_grad():
            for ts_b, txt_b, y_b in val_loader:
                logits    = mlp(ts_b, txt_b)
                val_loss += criterion(logits, y_b).item()
                preds     = torch.sigmoid(logits)
                val_preds.extend(preds.squeeze().tolist())
                val_true.extend(y_b.squeeze().tolist())
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        val_preds_bin = [1 if p > 0.5 else 0 for p in val_preds]
        tp   = sum(p == 1 and t == 1 for p, t in zip(val_preds_bin, val_true))
        fp   = sum(p == 1 and t == 0 for p, t in zip(val_preds_bin, val_true))
        fn   = sum(p == 0 and t == 1 for p, t in zip(val_preds_bin, val_true))
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in mlp.state_dict().items()}
            marker = ' <- best'
        else:
            marker = ''

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f'  Epoch {epoch+1:3d} | '
                f'train={train_loss:.4f} val={val_loss:.4f} '
                f'f1={f1:.3f}{marker}'
            )

    mlp.load_state_dict(best_state)
    history['best_val_loss'] = best_val_loss
    print(f'\nBest val loss: {best_val_loss:.4f}')

    save_mlp(
        mlp, save_path,
        ts_dim=ts_dim, text_dim=text_dim,
        epochs=epochs, val_loss=best_val_loss,
    )
    return mlp, history
