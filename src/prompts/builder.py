"""
Question builders for Approach 2 (stats-embedded) and Approach 3 (hybrid).
These functions construct the full prompt sent to ChatTS, embedding numerical
signal statistics so the model can reason from numbers rather than raw patterns.
"""

import numpy as np
import pandas as pd

from .templates import MCQ_CATEGORIES, MCQ_MULTI, MCQ_STALE_FOCUSED


def build_smart_question(
    vals: np.ndarray,
    idx,
    tag: str,
) -> tuple[np.ndarray, str]:
    """
    Approach 2: embed signal statistics into the question.
    No pre-screening - let ChatTS decide everything.
    Uses middle 512 points of the signal as the representative chunk.

    Returns
    -------
    chunk    : np.ndarray float32, length ≤ 512
    question : str - full prompt to pass to ask_chatts_chunk
    """
    n            = len(vals)
    global_mean  = float(vals.mean())
    global_std   = float(vals.std())
    val_min      = float(vals.min())
    val_max      = float(vals.max())
    signal_range = val_max - val_min

    series       = pd.Series(vals)
    roll_std_min = float(series.rolling(10).std().fillna(1.0).min())

    roll_mean        = series.rolling(96).mean().dropna()
    roll_mean_range  = float(roll_mean.max() - roll_mean.min()) if len(roll_mean) > 0 else 0

    mid   = n // 2
    start = max(0, mid - 256)
    chunk = vals[start:start + 512]

    spike_hi = global_mean + 3 * global_std
    spike_lo = global_mean - 3 * global_std

    question = (
        f'This is industrial sensor tag [{tag}] sampled every 15 minutes. '
        f'Full dataset statistics over {n} points ({n*15//60} hours):\n'
        f'  mean={global_mean:.3f}, std={global_std:.3f}, '
        f'min={val_min:.3f}, max={val_max:.3f}\n'
        f'  24hr rolling mean range={roll_mean_range:.3f} '
        f'(how much the daily average shifts over the full period)\n'
        f'  minimum 10pt rolling std={roll_std_min:.4f} '
        f'(near 0 means signal was frozen at some point)\n'
        f'  3-sigma thresholds: above {spike_hi:.3f} or below {spike_lo:.3f} = spike\n'
        f'\n'
        f'The chunk sent to you is 512 points from the middle of the dataset.\n'
        f'\n'
        f'Using the statistics above AND the chunk pattern, classify this signal:\n'
        f'\n'
        f'A) Drift - 24hr rolling mean migrates significantly over weeks. '
        f'Indicator: rolling mean range > 10% of signal range ({0.1*signal_range:.3f})\n'
        f'B) Spikes - sudden extreme values outside 3-sigma, '
        f'returning to baseline within 1-3 points\n'
        f'C) Frozen segment - minimum rolling std near zero (<0.01), '
        f'signal stuck at exact same value for many consecutive readings\n'
        f'D) Phase Change - sudden permanent shift in baseline level\n'
        f'E) None - clean data, no data quality issues\n'
        f'\n'
        f'Start your answer with the letter (A/B/C/D/E) then explain using '
        f'the statistics provided. Multiple categories are allowed if applicable.'
    )

    return chunk.astype(np.float32), question
