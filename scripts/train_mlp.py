"""
CLI entry point - train the MLP segmentation model from cached embeddings.

Usage (on VSC):
    python scripts/train_mlp.py --data combined  --epochs 150
    python scripts/train_mlp.py --data industrial --epochs 50
"""

import argparse
import os
import torch

from dotenv import load_dotenv
load_dotenv()

from src.segmentation.training import train_mlp

VSC_SCRATCH = os.environ.get('VSC_SCRATCH', '/scratch/leuven/375/vsc37531')
DATA_DIR    = os.path.join(VSC_SCRATCH, 'ChatTS', 'timeseer_data')


def main():
    parser = argparse.ArgumentParser(description='Train MLP segmentation model.')
    parser.add_argument('--data',    choices=['industrial', 'combined'], default='combined')
    parser.add_argument('--epochs',  type=int, default=150)
    parser.add_argument('--batch',   type=int, default=64)
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--out',     default=None,
                        help='Save path for model checkpoint')
    args = parser.parse_args()

    if args.data == 'combined':
        X_ts  = torch.load(os.path.join(DATA_DIR, 'X_ts_combined.pt'),  weights_only=False)
        X_txt = torch.load(os.path.join(DATA_DIR, 'X_txt_combined.pt'), weights_only=False)
        y     = torch.load(os.path.join(DATA_DIR, 'y_combined.pt'),     weights_only=False)
        pos_weight = (1 - 0.2288) / 0.2288  # from segmentation.yaml
    else:
        X_ts  = torch.load(os.path.join(DATA_DIR, 'X_ts_industrial.pt'),  weights_only=False)
        X_txt = torch.load(os.path.join(DATA_DIR, 'X_txt_industrial.pt'), weights_only=False)
        y     = torch.load(os.path.join(DATA_DIR, 'y_industrial.pt'),     weights_only=False)
        pos_weight = (1 - 0.0829) / 0.0829

    save_path = args.out or os.path.join(DATA_DIR, 'segmentation_mlp_final.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f'Data     : {args.data}  ({X_ts.shape[0]} samples)')
    print(f'Epochs   : {args.epochs}')
    print(f'Save to  : {save_path}')
    print()

    mlp, history = train_mlp(
        X_ts, X_txt, y,
        save_path=save_path,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        pos_weight=pos_weight,
    )
    print(f'\nTraining complete. Best val_loss={history["best_val_loss"]:.4f}')


if __name__ == '__main__':
    main()
