"""
Generate training tensors from the 14 labelled industrial signals.
Produces X_ts_industrial.pt, X_txt_industrial.pt, y_industrial.pt
in $VSC_SCRATCH/ChatTS/timeseer_data/.

Run this ONCE on VSC before training the MLP.
Requires: ChatTS-14B loaded, Chronos server running.

Usage:
    python scripts/generate_industrial_training_data.py
"""

import os
import numpy as np
import torch

from dotenv import load_dotenv
load_dotenv()

from src.data.timeseer_client import fetch_series_api
from src.models.chatts_loader import load_model
from src.inference.embeddings import encode_text_query
from src.inference.chronos_server import start_server, get_chronos_embedding_cached, shutdown_server
from src.preprocessing.windows import build_training_sample

VSC_SCRATCH = os.environ.get('VSC_SCRATCH', '/scratch/leuven/375/vsc37531')
DATA_DIR    = os.path.join(VSC_SCRATCH, 'ChatTS', 'timeseer_data')

# 14 labelled industrial signals with ground-truth anomaly index ranges.
# From configs/segmentation.yaml (industrial_signals list).
INDUSTRIAL_SIGNALS = [
    ('R1-AT-102-COND',  'Reactor 1',            'spikes',       1460, 1475),
    ('R1-AT-101-PH',    'Reactor 1',            'drift',        0,    8640),
    ('R1-AT-103-DO',    'Reactor 1',            'stale',        2131, 2259),
    ('R2-AT-202-TOC',   'Reactor 2',            'spikes',       2860, 2880),
    ('R2-AT-201-NH3',   'Reactor 2',            'spikes',       1465, 1480),
    ('SEP-AT-301-PH',   'Separator',            'drift',        0,    8640),
    ('SEP-AT-302-TURB', 'Separator',            'spikes',       8290, 8310),
    ('DIST-AT-407-REFR','Distillation',         'var_collapse', 5345, 5473),
    ('WT-AT-501-PH',    'Wastewater Treatment', 'stale',        2909, 3037),
    ('WT-AT-502-COD',   'Wastewater Treatment', 'drift',        0,    8640),
    ('WT-AT-505-TURB',  'Wastewater Treatment', 'spikes',       3878, 3890),
    ('UTL-AT-601-CL2',  'Utilities',            'stale',        7511, 7639),
    ('UTL-AT-602-CO2',  'Utilities',            'spikes',       1250, 1265),
    ('UTL-AT-605-COND', 'Utilities',            'spikes',       6283, 6295),
]

QUERY_MAP = {
    'spikes':       'Find periods with sudden extreme spikes returning to baseline',
    'drift':        'Find gradual baseline drift over the full signal',
    'stale':        'Detect stale data - frozen segments with zero variation',
    'var_collapse': 'Find periods where signal amplitude collapses to near-zero',
}

WINDOW_SIZE = 128


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print('Loading ChatTS-14B for text embeddings...')
    model, tokenizer, processor = load_model('ChatTS-14B')

    def _encode(q):
        return encode_text_query(q, model=model, tokenizer=tokenizer)

    print('\nStarting Chronos server...')
    start_server()

    all_ts_embs  = []
    all_txt_embs = []
    all_labels   = []

    for tag, area, anom_type, anom_start, anom_end in INDUSTRIAL_SIGNALS:
        print(f'\n{tag} ({anom_type})')
        vals, idx = fetch_series_api(tag, area)
        if vals is None:
            print('  FAILED to fetch - skipping')
            continue

        label_mask = np.zeros(len(vals))
        label_mask[anom_start:anom_end] = 1.0

        query = QUERY_MAP.get(anom_type, 'Find periods with abnormal dynamics')
        print(f'  Query: {query}')
        print(f'  Signal length: {len(vals)}  Anomaly: [{anom_start}:{anom_end}]')

        ts_embs, txt_emb, window_labels, positions = build_training_sample(
            vals, label_mask, query,
            window_size=WINDOW_SIZE,
            encode_text_fn=_encode,
            get_chronos_fn=get_chronos_embedding_cached,
        )

        if len(ts_embs) == 0:
            print('  No windows extracted - skipping')
            continue

        all_ts_embs.append(ts_embs)
        all_txt_embs.append(np.tile(txt_emb, (len(ts_embs), 1)))
        all_labels.append(window_labels)

        pos_count = int(window_labels.sum())
        print(f'  Windows: {len(ts_embs)}  Positive (anomaly): {pos_count}  '
              f'({100*pos_count/len(ts_embs):.1f}%)')

    shutdown_server()

    X_ts  = torch.tensor(np.vstack(all_ts_embs),  dtype=torch.float32)
    X_txt = torch.tensor(np.vstack(all_txt_embs), dtype=torch.float32)
    y     = torch.tensor(np.concatenate(all_labels), dtype=torch.float32).unsqueeze(1)

    pos_frac = float(y.mean())
    print(f'\nDataset: {X_ts.shape[0]} windows  |  {pos_frac*100:.1f}% positive')
    print(f'X_ts  shape : {X_ts.shape}')
    print(f'X_txt shape : {X_txt.shape}')
    print(f'y     shape : {y.shape}')

    torch.save(X_ts,  os.path.join(DATA_DIR, 'X_ts_industrial.pt'))
    torch.save(X_txt, os.path.join(DATA_DIR, 'X_txt_industrial.pt'))
    torch.save(y,     os.path.join(DATA_DIR, 'y_industrial.pt'))
    print(f'\nSaved to {DATA_DIR}')


if __name__ == '__main__':
    main()
