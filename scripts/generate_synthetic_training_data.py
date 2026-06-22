"""
Generate synthetic training data using ChatTS's own ts_generator.
Produces X_ts_synthetic.pt, X_txt_synthetic.pt, y_synthetic.pt.

Synthetic samples augment the 14 industrial signals to improve MLP generalization.
The generator produces labelled anomalies (spikes, phase changes, etc.) with known
ground-truth local event annotations.

Run after generate_industrial_training_data.py.

Usage:
    python scripts/generate_synthetic_training_data.py --n-samples 500
"""

import argparse
import os
import sys
import numpy as np
import torch

from dotenv import load_dotenv
load_dotenv()

from src.models.chatts_loader import load_model
from src.inference.embeddings import encode_text_query
from src.inference.chronos_server import start_server, get_chronos_embedding_cached, shutdown_server
from src.preprocessing.windows import extract_windows

VSC_SCRATCH    = os.environ.get('VSC_SCRATCH', '/scratch/leuven/375/vsc37531')
DATA_DIR       = os.path.join(VSC_SCRATCH, 'ChatTS', 'timeseer_data')
CHATTS_ROOT    = os.path.join(VSC_SCRATCH, 'ChatTS')
WINDOW_SIZE    = 128

SPIKE_TYPES    = {
    'upward spike', 'downward spike', 'continuous upward spike',
    'continuous downward spike', 'wide upward spike', 'wide downward spike',
    'upward convex', 'downward convex', 'shake',
    'rapid rise followed by slow decline', 'slow rise followed by rapid decline',
    'rapid decline followed by slow rise', 'slow decline followed by rapid rise',
}
PHASE_TYPES    = {
    'sudden increase', 'sudden decrease',
    'decrease after upward spike', 'increase after downward spike',
}


def events_to_label_mask(local_events: list, n: int) -> np.ndarray:
    """Convert ChatTS local events to a binary label mask."""
    mask = np.zeros(n)
    for event in local_events:
        start = event.get('start', 0)
        end   = event.get('end', n)
        if event.get('type') in SPIKE_TYPES | PHASE_TYPES:
            mask[start:end] = 1.0
    return mask


def event_to_query(local_events: list) -> str:
    """Select the most relevant query for the generated events."""
    if not local_events:
        return 'Find periods with abnormal dynamics'
    types = {e['type'] for e in local_events}
    if types & SPIKE_TYPES:
        return 'Find periods with sudden extreme spikes returning to baseline'
    if types & PHASE_TYPES:
        return 'Find sudden increases or permanent level shifts'
    return 'Find periods with abnormal dynamics'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--seq-len',   type=int, default=256)
    args = parser.parse_args()

    sys.path.insert(0, CHATTS_ROOT)
    try:
        from chatts.ts_generator.generate import (
            generate_random_attributes,
            generate_time_series,
        )
    except ImportError:
        print(f'ERROR: ChatTS ts_generator not found in {CHATTS_ROOT}')
        print('Make sure ChatTS is cloned into $VSC_SCRATCH/ChatTS/')
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    print('Loading ChatTS-14B for text embeddings...')
    model, tokenizer, processor = load_model('ChatTS-14B')

    def _encode(q):
        return encode_text_query(q, model=model, tokenizer=tokenizer)

    print('Starting Chronos server...')
    start_server()

    all_ts_embs  = []
    all_txt_embs = []
    all_labels   = []

    print(f'\nGenerating {args.n_samples} synthetic samples...')
    print('=' * 55)

    for i in range(args.n_samples):
        attr      = generate_random_attributes(seq_len=args.seq_len)
        ts, attr  = generate_time_series(attr, seq_len=args.seq_len)
        ts_array  = np.array(ts, dtype=np.float32)
        events    = attr.get('local', [])
        n         = len(ts_array)

        label_mask = events_to_label_mask(events, n)
        query      = event_to_query(events)

        windows, positions = extract_windows(ts_array, WINDOW_SIZE)
        txt_emb = _encode(query)
        ts_embs = []
        window_labels = []

        for (start, end), window in zip(positions, windows):
            emb = get_chronos_embedding_cached(np.array(window, dtype=np.float32))
            if emb is None:
                continue
            ts_embs.append(emb)
            window_labels.append(float(label_mask[start:end].max()))

        if not ts_embs:
            continue

        all_ts_embs.append(np.array(ts_embs))
        all_txt_embs.append(np.tile(txt_emb, (len(ts_embs), 1)))
        all_labels.append(np.array(window_labels))

        if (i + 1) % 50 == 0:
            pos = sum(l for labels in all_labels for l in labels)
            tot = sum(len(l) for l in all_labels)
            print(f'  [{i+1}/{args.n_samples}] windows so far: {tot}  '
                  f'positive: {pos:.0f} ({100*pos/max(tot,1):.1f}%)')

    shutdown_server()

    X_ts  = torch.tensor(np.vstack(all_ts_embs),  dtype=torch.float32)
    X_txt = torch.tensor(np.vstack(all_txt_embs), dtype=torch.float32)
    y     = torch.tensor(np.concatenate(all_labels), dtype=torch.float32).unsqueeze(1)

    print(f'\nSynthetic dataset: {X_ts.shape[0]} windows  |  {float(y.mean())*100:.1f}% positive')

    torch.save(X_ts,  os.path.join(DATA_DIR, 'X_ts_synthetic.pt'))
    torch.save(X_txt, os.path.join(DATA_DIR, 'X_txt_synthetic.pt'))
    torch.save(y,     os.path.join(DATA_DIR, 'y_synthetic.pt'))

    # Merge with industrial data if available
    ind_ts_path = os.path.join(DATA_DIR, 'X_ts_industrial.pt')
    if os.path.exists(ind_ts_path):
        Xi  = torch.load(ind_ts_path,  weights_only=False)
        Xti = torch.load(os.path.join(DATA_DIR, 'X_txt_industrial.pt'), weights_only=False)
        yi  = torch.load(os.path.join(DATA_DIR, 'y_industrial.pt'),     weights_only=False)
        X_ts_comb  = torch.cat([Xi,  X_ts],  dim=0)
        X_txt_comb = torch.cat([Xti, X_txt], dim=0)
        y_comb     = torch.cat([yi,  y],     dim=0)
        torch.save(X_ts_comb,  os.path.join(DATA_DIR, 'X_ts_combined.pt'))
        torch.save(X_txt_comb, os.path.join(DATA_DIR, 'X_txt_combined.pt'))
        torch.save(y_comb,     os.path.join(DATA_DIR, 'y_combined.pt'))
        print(f'Combined dataset: {X_ts_comb.shape[0]} windows  |  '
              f'{float(y_comb.mean())*100:.1f}% positive')

    print(f'Saved to {DATA_DIR}')


if __name__ == '__main__':
    main()
