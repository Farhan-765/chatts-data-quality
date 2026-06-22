"""
Statistical pre-screener for industrial sensor anomaly detection.

Two public functions:
  analyze_signal(vals, idx)        - Approach 1: simple statistical only
  hybrid_analyze(vals, idx, tag)   - Approach 3: pre-screen + embedded stats

Thresholds are documented in configs/prescreener.yaml.
"""

import re
import numpy as np
import pandas as pd

from src.prompts.templates import MCQ_CATEGORIES, MCQ_MULTI, MCQ_STALE_FOCUSED

# Tags that have large daily cycles and must NOT trigger spike detection.
# Full suffix list tuned on the 55-signal industrial dataset.
SKIP_SPIKE_TAGS = [
    'ORP', 'PRESS', 'TEMP', 'TEMP-TOP', 'TEMP-BTM',
    'DENS', 'DENS-OH', 'DENS-BTM', 'DENS-2',
    'VISC', 'VISC-2', 'GC-C1', 'REFR', 'COLOR',
    'MOIST', 'MOIST-2', 'LEVEL', 'COND-CW',
    'TURB-CW', 'DEW', 'DEW-IA', 'PH-CW', 'PH-2',
    'UTL-AT-603-O2', 'NH3-IA', 'PH', 'PKG-AT-706-COND',
    'SEP-AT-303-COND', 'WT-AT-506-COND',
    'WT-AT-504-TSS',
    'WT-AT-503-DO',
    'WT-AT-502-COD',
]


def analyze_signal(
    vals: np.ndarray,
    idx,
) -> tuple:
    """
    Approach 1: simple statistical pre-screening with global thresholds.

    Returns
    -------
    chunk      : np.ndarray or list of np.ndarray
    template   : str - MCQ template string
    tname      : str - template name for logging
    context    : str - stats context to prepend to the template
    detected   : list[str] - pre-screener findings
    chunk_desc : str - human-readable description of chunk selection
    """
    n            = len(vals)
    series       = pd.Series(vals)
    global_mean  = vals.mean()
    global_std   = vals.std()

    # 1. Frozen segments - rolling std
    roll_std        = series.rolling(10).std().fillna(1.0)
    min_roll_std    = roll_std.min()
    frozen_candidate = min_roll_std < 0.01

    # 2. Drift - rolling mean range
    roll_mean = series.rolling(96).mean().dropna()
    if len(roll_mean) > 0:
        mean_range   = roll_mean.max() - roll_mean.min()
        signal_range = vals.max() - vals.min()
        drift_ratio  = mean_range / signal_range if signal_range > 0 else 0
        drift_candidate = drift_ratio > 0.15
    else:
        drift_candidate = False

    # 3. Spikes - global 3-sigma outliers
    outlier_mask    = np.abs(vals - global_mean) > 3 * global_std
    n_outliers      = outlier_mask.sum()
    spike_candidate = n_outliers >= 2

    # 4. Detected list
    detected = []
    if frozen_candidate: detected.append('stale')
    if drift_candidate:  detected.append('drift')
    if spike_candidate:  detected.append('spikes')
    if not detected:     detected.append('clean')

    # 5. Template selection
    if 'stale' in detected:
        template, tname = MCQ_STALE_FOCUSED, 'MCQ_STALE_FOCUSED'
    elif len(detected) > 1 and 'clean' not in detected:
        template, tname = MCQ_MULTI, 'MCQ_MULTI'
    else:
        template, tname = MCQ_CATEGORIES, 'MCQ_CATEGORIES'

    # 6. Chunk selection
    if 'stale' in detected:
        frozen_centre = int(roll_std.idxmin())
        start = max(0, frozen_centre - 64)
        end   = min(n, start + 128)
        start = max(0, end - 128)
        chunk      = vals[start:end]
        chunk_desc = f'128pt centred on frozen region (idx {start}-{end})'

    elif 'drift' in detected and 'spikes' not in detected:
        tenth = min(256, max(50, n // 10))
        early = vals[:tenth]
        late  = vals[n - tenth:]
        chunk      = [early, late]
        chunk_desc = f'Two chunks: first {tenth}pts vs last {tenth}pts'

    elif 'spikes' in detected:
        outlier_idx = int(np.argmax(np.abs(vals - global_mean)))
        start = max(0, outlier_idx - 64)
        end   = min(n, start + 128)
        start = max(0, end - 128)
        chunk      = vals[start:end]
        chunk_desc = f'128pt centred on spike at idx {outlier_idx}'

    else:
        mid   = n // 2
        start = max(0, mid - 256)
        chunk      = vals[start:start + 512]
        chunk_desc = f'Middle 512pts (idx {start}-{start+512})'

    # 7. Context - embed signal stats + spike thresholds
    spike_hi = global_mean + 3 * global_std
    spike_lo = global_mean - 3 * global_std
    context  = (
        f'This is an industrial sensor measurement sampled every 15 minutes. '
        f'Full signal stats: mean={global_mean:.3f}, std={global_std:.3f}, '
        f'min={vals.min():.3f}, max={vals.max():.3f}, n={n} points. '
    )
    if 'spikes' in detected:
        context += (
            f'IMPORTANT: Any point above {spike_hi:.2f} or below {spike_lo:.2f} '
            f'is a spike - these are pre-calculated 3-sigma thresholds. '
        )

    return chunk, template, tname, context, detected, chunk_desc


def hybrid_analyze(
    vals: np.ndarray,
    idx,
    tag: str,
) -> tuple:
    """
    Approach 3: statistical pre-screening + stats embedded in question.
    Pre-screening decides chunk and template.
    Stats are embedded so ChatTS reasons from numbers, not raw patterns.

    Returns
    -------
    chunk      : list of np.ndarray  (1 or 2 chunks for drift)
    question   : str - full prompt (stats context + MCQ template)
    tname      : str - template name for logging
    detected   : list[str] - pre-screener findings
    chunk_desc : str - human-readable description of chunk selection
    """
    n            = len(vals)
    series       = pd.Series(vals)
    global_mean  = float(vals.mean())
    global_std   = float(vals.std())
    val_min      = float(vals.min())
    val_max      = float(vals.max())
    signal_range = val_max - val_min

    seg = n // 3
    m1  = float(vals[:seg].mean())
    m2  = float(vals[seg:2*seg].mean())
    m3  = float(vals[2*seg:].mean())

    skip_spike = any(p in tag.upper() for p in SKIP_SPIKE_TAGS)

    # Pre-screen 1: frozen / variance collapse
    roll_std       = series.rolling(10).std().fillna(1.0)
    min_roll_std   = float(roll_std.min())
    near_zero_wins = int((roll_std < 0.01).sum())
    zero_diffs     = int((np.diff(vals) == 0).sum())
    frozen_candidate = min_roll_std < 0.01

    is_true_frozen  = frozen_candidate and zero_diffs > 10
    is_var_collapse = frozen_candidate and zero_diffs <= 10 and near_zero_wins > 50

    # Pre-screen 2: drift
    roll_mean       = series.rolling(96).mean().dropna()
    roll_mean_range = float(roll_mean.max() - roll_mean.min()) if len(roll_mean) > 0 else 0
    drift_ratio     = roll_mean_range / signal_range if signal_range > 0 else 0

    roll_mean_std   = float(roll_mean.std()) if len(roll_mean) > 0 else 0
    drift_candidate = drift_ratio > 0.15 and roll_mean_std > 0.3 * global_std

    # Pre-screen 3: spikes (with periodicity filter)
    if skip_spike:
        spike_candidate = False
        n_outliers      = 0
    else:
        roll_mean_pre = series.rolling(10).mean().shift(5).fillna(global_mean)
        roll_std_pre  = series.rolling(10).std().shift(5).fillna(global_std)
        local_dev     = np.abs(vals - roll_mean_pre.values)
        spike_mask    = local_dev > 4 * roll_std_pre.values

        spike_runs = []
        in_run, run_len, run_start = False, 0, 0
        for i, s in enumerate(spike_mask):
            if s:
                if not in_run:
                    in_run, run_start, run_len = True, i, 1
                else:
                    run_len += 1
            else:
                if in_run and 1 <= run_len <= 5:
                    spike_runs.append(run_start)
                in_run, run_len = False, 0

        n_runs          = len(spike_runs)
        spike_candidate = False
        if n_runs >= 1:
            outlier_amp = float(local_dev[spike_mask].max())
            if n_runs >= 10:
                gaps     = np.diff(spike_runs)
                gap_cv   = float(np.std(gaps)) / (float(np.mean(gaps)) + 1e-6)
                gap_mean = float(np.mean(gaps))
                is_periodic = gap_cv < 0.5 and 48 < gap_mean < 200
            else:
                is_periodic = False
            if not is_periodic:
                spike_candidate = outlier_amp > 0.20 * signal_range

    # Pre-screen 4: intermittent failure
    neg_count  = int((vals < 0).sum())
    neg_ratio  = neg_count / n
    intermittent_candidate = neg_count >= 2 and neg_ratio < 0.05

    # Detected list
    detected = []
    if is_true_frozen:          detected.append('stale')
    elif is_var_collapse:       detected.append('var_collapse')
    if drift_candidate:         detected.append('drift')
    if spike_candidate:         detected.append('spikes')
    if intermittent_candidate:  detected.append('intermittent')
    if not detected:            detected.append('clean')

    # Template selection
    if 'stale' in detected:
        template, tname = MCQ_STALE_FOCUSED, 'MCQ_STALE_FOCUSED'
    elif 'var_collapse' in detected:
        template, tname = MCQ_CATEGORIES, 'MCQ_CATEGORIES (var_collapse)'
    elif 'intermittent' in detected:
        template, tname = MCQ_CATEGORIES, 'MCQ_CATEGORIES (intermittent)'
    elif 'drift' in detected and 'spikes' in detected:
        template, tname = MCQ_CATEGORIES, 'MCQ_CATEGORIES (drift priority)'
    elif len([d for d in detected if d not in ('clean', 'drift')]) > 0 and 'drift' in detected:
        template, tname = MCQ_MULTI, 'MCQ_MULTI'
    elif len([d for d in detected if d != 'clean']) > 1:
        template, tname = MCQ_MULTI, 'MCQ_MULTI'
    else:
        template, tname = MCQ_CATEGORIES, 'MCQ_CATEGORIES'

    # Chunk selection
    if 'stale' in detected or 'var_collapse' in detected:
        frozen_centre = int(roll_std.idxmin())
        start = max(0, frozen_centre - 64)
        end   = min(n, start + 128)
        start = max(0, end - 128)
        chunk      = [vals[start:end]]
        chunk_desc = f'128pt centred on low-variance region (idx {start}-{end})'

    elif 'drift' in detected:
        tenth = max(96, n // 10)
        early = vals[:tenth]
        late  = vals[n - tenth:]
        chunk      = [early, late]
        chunk_desc = f'Two chunks: first {tenth}pts vs last {tenth}pts'

    elif 'spikes' in detected or 'intermittent' in detected:
        if neg_count > 0:
            neg_indices  = np.where(vals < 0)[0]
            spike_idx    = int(np.argmax(np.abs(vals - global_mean)))
            most_neg_idx = int(neg_indices[np.argmin(vals[neg_indices])])
            outlier_idx  = (
                most_neg_idx
                if abs(vals[most_neg_idx] - global_mean) > abs(vals[spike_idx] - global_mean)
                else spike_idx
            )
        else:
            outlier_idx = int(np.argmax(np.abs(vals - global_mean)))
        start = max(0, outlier_idx - 64)
        end   = min(n, start + 128)
        start = max(0, end - 128)
        chunk      = [vals[start:end]]
        chunk_desc = f'128pt centred on anomaly at idx {outlier_idx}'

    else:
        mid   = n // 2
        start = max(0, mid - 1008)
        end   = min(n, start + 2016)
        raw   = vals[start:end]
        if len(raw) > 512:
            idx_ds = np.linspace(0, len(raw) - 1, 512, dtype=int)
            chunk = [raw[idx_ds]]
        else:
            chunk = [raw]
        chunk_desc = f'3-week window downsampled to 512pts (idx {start}-{end})'

    # Build context string
    spike_hi = global_mean + 3 * global_std
    spike_lo = global_mean - 3 * global_std
    days     = n * 15 // 60 // 24

    if 'clean' in detected:
        stats_context = (
            f'Signal: [{tag}] | {n} pts | {days} days | 15-min sampling.\n'
            f'Stats: mean={global_mean:.3f}, std={global_std:.3f}, '
            f'min={val_min:.3f}, max={val_max:.3f}.\n'
            f'Monthly means: M1={m1:.3f}, M2={m2:.3f}, M3={m3:.3f} '
            f'(M1→M3 change = {m3-m1:.3f}).\n'
            f'Frozen indicator: min 10pt rolling std={min_roll_std:.4f}.\n'
            f'Daily sinusoidal cycles (~96pt period) are EXPECTED process behavior, '
            f'NOT drift. Only classify as A) Drift if monthly means differ '
            f'significantly (M1→M3 change > 1.0 units).\n\n'
            f'IMPORTANT: Start your answer with the letter only e.g. "E) None".\n\n'
        )

    elif 'stale' in detected:
        stats_context = (
            f'Signal: [{tag}] | {n} pts | {days} days | 15-min sampling.\n'
            f'Stats: mean={global_mean:.3f}, std={global_std:.3f}.\n'
            f'Frozen indicator: min 10pt rolling std={min_roll_std:.4f} '
            f'(near zero = frozen candidate).\n'
            f'Consecutive identical readings: {zero_diffs}.\n\n'
            f'IMPORTANT: Start your answer with the letter only e.g. "C) Frozen".\n\n'
        )

    elif 'var_collapse' in detected:
        stats_context = (
            f'Signal: [{tag}] | {n} pts | {days} days | 15-min sampling.\n'
            f'Stats: mean={global_mean:.3f}, std={global_std:.3f}.\n'
            f'Variance collapse evidence:\n'
            f'  min_roll_std={min_roll_std:.4f} (near zero)\n'
            f'  near_zero_windows={near_zero_wins} windows with std < 0.01\n'
            f'  zero_diffs={zero_diffs} consecutive identical readings\n'
            f'NOTE: {zero_diffs} consecutive identical readings means values '
            f'{"DO" if zero_diffs > 10 else "DO NOT"} completely stop changing.\n'
            f'If zero_diffs < 10 and near_zero_windows > 50 → G) Variance collapse.\n'
            f'If zero_diffs > 10 → C) Frozen segment.\n\n'
            f'IMPORTANT: Start your answer with the letter only e.g. "G) Variance collapse".\n\n'
        )

    elif 'intermittent' in detected:
        stats_context = (
            f'Signal: [{tag}] | {n} pts | {days} days | 15-min sampling.\n'
            f'Stats: mean={global_mean:.3f}, std={global_std:.3f}, '
            f'min={val_min:.3f}, max={val_max:.3f}.\n'
            f'WARNING: {neg_count} NEGATIVE VALUES detected at indices '
            f'{list(np.where(vals < 0)[0])} with values '
            f'{list(vals[vals < 0].round(3))}.\n'
            f'CRITICAL: Turbidity, TOC, dissolved oxygen, and concentration '
            f'measurements CANNOT be negative - they have a physical minimum of zero. '
            f'ANY negative value regardless of magnitude is physically impossible '
            f'and indicates sensor malfunction (L) Intermittent failure).\n'
            f'Largest spike at idx {int(np.argmax(np.abs(vals - global_mean)))}: '
            f'value={vals[int(np.argmax(np.abs(vals - global_mean)))]:.3f}.\n'
            f'Spike thresholds: above {spike_hi:.3f} or below {spike_lo:.3f}.\n\n'
            f'IMPORTANT: Start your answer with "B, L) Spikes and Intermittent failure" '
            f'if both spikes AND negative values are present.\n\n'
        )

    else:
        stats_context = (
            f'Signal: [{tag}] | {n} pts | {days} days | 15-min sampling.\n'
            f'Stats: mean={global_mean:.3f}, std={global_std:.3f}, '
            f'min={val_min:.3f}, max={val_max:.3f}, range={signal_range:.3f}.\n'
            f'Natural signal range: {val_min:.3f} to {val_max:.3f} - '
            f'values within this range are physically valid, NOT impossible values.\n'
            f'Monthly means: M1={m1:.3f}, M2={m2:.3f}, M3={m3:.3f} '
            f'(M1→M3 change = {m3-m1:.3f}).\n'
            f'Drift indicator: 24hr rolling mean range={roll_mean_range:.3f} '
            f'({drift_ratio*100:.1f}% of signal range).\n'
            f'Frozen indicator: min 10pt rolling std={min_roll_std:.4f}.\n'
            f'Spike thresholds: above {spike_hi:.3f} or below {spike_lo:.3f}.\n'
            f'Pre-screen detected: {", ".join(detected)}.\n'
            f'This chunk is centred on the largest detected anomaly. '
            f'Focus on that anomaly only - do not classify surrounding normal variation.\n\n'
            f'IMPORTANT: Start your answer with the letter only '
            f'e.g. "A) Drift" or "B) Spikes". '
            f'Only report L) Intermittent failure if values are physically impossible '
            f'(below absolute zero, negative concentration etc.) - '
            f'NOT just because values are at the natural min/max of this signal.\n\n'
        )

    if 'drift' in detected and len(chunk) == 2:
        stats_context += (
            f'TS1 = early period, TS2 = same signal ~{days//10} days later. '
            f'If mean(TS2) > mean(TS1) by more than 0.3 units → A) Drift.\n\n'
        )

    full_question = stats_context + template

    return chunk, full_question, tname, detected, chunk_desc
