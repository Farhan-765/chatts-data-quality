"""
CLI entry point - run multi-scale segmentation on specified signals.

Usage (on VSC):
    python scripts/run_segmentation.py --tag R1-AT-102-COND --area "Reactor 1"
    python scripts/run_segmentation.py --tag R1-AT-103-DO   --area "Reactor 1" --query "Detect stale data"
"""

import argparse
import os
import numpy as np
import torch

from dotenv import load_dotenv
load_dotenv()

from src.data.timeseer_client import fetch_series_api
from src.data.ground_truth import GROUND_TRUTH
from src.models.chatts_loader import load_model
from src.models.mlp import load_mlp
from src.inference.embeddings import encode_text_query
from src.inference.chronos_server import start_server, get_chronos_embedding_cached, shutdown_server
from src.preprocessing.chunking import downsample
from src.segmentation.pipeline import multiscale_segmentation
from src.visualization.segmentation_plots import plot_segmentation

VSC_SCRATCH = os.environ.get('VSC_SCRATCH', '/scratch/leuven/375/vsc37531')


def main():
    parser = argparse.ArgumentParser(description='Run multi-scale segmentation on one signal.')
    parser.add_argument('--tag',   required=True, help='Signal tag name')
    parser.add_argument('--area',  required=True, help='Plant area name')
    parser.add_argument('--query', default='Find periods with abnormal dynamics')
    parser.add_argument('--model', default='ChatTS-14B')
    parser.add_argument('--mlp',   default=None,
                        help='MLP checkpoint path (default: mlp_final_path from vsc.yaml)')
    parser.add_argument('--threshold', type=float, default=0.65)
    args = parser.parse_args()

    mlp_path = args.mlp or os.path.join(
        VSC_SCRATCH, 'ChatTS', 'timeseer_data', 'segmentation_mlp_final.pt'
    )

    print(f'Tag   : {args.tag}')
    print(f'Area  : {args.area}')
    print(f'Query : {args.query}')
    print('=' * 60)

    # Load ChatTS for text embeddings
    model, tokenizer, processor = load_model(args.model)

    def _encode_text(q):
        return encode_text_query(q, model=model, tokenizer=tokenizer)

    # Start Chronos server
    start_server()

    # Load MLP
    mlp = load_mlp(mlp_path)

    # Fetch signal
    vals, idx = fetch_series_api(args.tag, args.area)
    if vals is None:
        print('Failed to fetch data.')
        return

    vals_ds = downsample(vals, target=1024)
    print(f'Signal downsampled: {len(vals)} → {len(vals_ds)} pts')

    # Ground truth indices (scaled to 1024)
    gt_label = GROUND_TRUTH.get(args.tag, 'E')
    print(f'Ground truth label: {gt_label}')

    max_prob, binary, metrics = multiscale_segmentation(
        vals_ds, args.tag, args.query,
        mlp=mlp,
        encode_text_fn=_encode_text,
        get_chronos_fn=get_chronos_embedding_cached,
        threshold=args.threshold,
    )

    if metrics:
        print(f'\nEvaluation: {metrics}')

    save_path = os.path.join(
        'figures',
        f'segmentation_{args.tag.replace("-", "_")}.png',
    )
    plot_segmentation(
        vals_ds, max_prob, binary,
        query_text=args.query,
        tag=args.tag,
        save_path=save_path,
    )

    shutdown_server()


if __name__ == '__main__':
    main()
