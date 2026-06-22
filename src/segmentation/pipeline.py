"""
Segmentation pipeline - runs the trained MLP on a full industrial signal.

Two public functions:
  segment_signal()         - single window size, returns prob_map + binary mask
  multiscale_segmentation() - runs at [32, 64, 128, 256] and max-pools
"""

import numpy as np
import torch

from src.preprocessing.windows import extract_windows


def segment_signal(
    signal: np.ndarray,
    query_text: str,
    mlp,
    encode_text_fn,
    get_chronos_fn,
    window_size: int = 128,
    stride_ratio: float = 0.5,
    threshold: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run full segmentation pipeline on a signal.

    Parameters
    ----------
    signal         : 1-D float array (ideally downsampled to 1024 pts)
    query_text     : natural-language query e.g. 'Detect stale data'
    mlp            : trained TSAISegmentationMLP
    encode_text_fn : callable(str) → np.ndarray[5120]
    get_chronos_fn : callable(np.ndarray) → np.ndarray[512]
    window_size    : sliding window length
    stride_ratio   : overlap fraction
    threshold      : binary decision threshold

    Returns
    -------
    prob_map : np.ndarray float, shape (len(signal),) - per-timestep probability
    mask     : np.ndarray int,   shape (len(signal),) - binary anomaly mask
    """
    windows, positions = extract_windows(signal, window_size, stride_ratio)

    txt_emb    = torch.tensor(encode_text_fn(query_text), dtype=torch.float32).unsqueeze(0)
    ts_embs    = []
    valid_pos  = []

    for window, pos in zip(windows, positions):
        emb = get_chronos_fn(np.array(window, dtype=np.float32))
        if emb is not None:
            ts_embs.append(emb)
            valid_pos.append(pos)

    if not ts_embs:
        return np.zeros(len(signal)), np.zeros(len(signal), dtype=int)

    ts_emb_tensor    = torch.tensor(np.array(ts_embs), dtype=torch.float32)
    txt_emb_expanded = txt_emb.expand(ts_emb_tensor.shape[0], -1)

    mlp.eval()
    with torch.no_grad():
        logits = mlp(ts_emb_tensor, txt_emb_expanded)
        probs  = torch.sigmoid(logits).squeeze().numpy()

    prob_map  = np.zeros(len(signal))
    count_map = np.zeros(len(signal))
    for i, (start, end) in enumerate(valid_pos):
        p = float(probs[i]) if probs.ndim > 0 else float(probs)
        prob_map[start:end]  += p
        count_map[start:end] += 1

    count_map = np.maximum(count_map, 1)
    prob_map  = prob_map / count_map
    mask      = (prob_map > threshold).astype(int)

    return prob_map, mask


def multiscale_segmentation(
    vals: np.ndarray,
    tag: str,
    query: str,
    mlp,
    encode_text_fn,
    get_chronos_fn,
    true_anom_start: int | None = None,
    true_anom_end:   int | None = None,
    window_sizes: list[int] | None = None,
    threshold: float = 0.65,
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """
    Run segmentation at multiple window sizes; take max probability per timestep.

    Parameters
    ----------
    vals             : downsampled signal array
    tag              : signal tag name for logging
    query            : natural-language query string
    window_sizes     : list of window sizes (default: [32, 64, 128, 256])
    threshold        : binary decision threshold
    true_anom_start/end : optional ground-truth indices for evaluation

    Returns
    -------
    max_prob_map : np.ndarray float, shape (N,)
    binary_mask  : np.ndarray int,   shape (N,)
    eval_metrics : dict with TP/FP/FN/precision/recall/F1, or None
    """
    if window_sizes is None:
        window_sizes = [32, 64, 128, 256]

    n             = len(vals)
    all_prob_maps = []

    print(f'Multi-scale segmentation: {tag}')
    print(f'Query: "{query}"')

    for ws in window_sizes:
        prob_map, _ = segment_signal(
            vals, query, mlp, encode_text_fn, get_chronos_fn,
            window_size=ws, stride_ratio=0.5, threshold=threshold,
        )
        all_prob_maps.append(prob_map)
        print(f'  ws={ws:3d}: max_prob={prob_map.max():.3f}  '
              f'detections={int((prob_map > threshold).sum())}')

    prob_matrix  = np.vstack(all_prob_maps)
    max_prob_map = prob_matrix.max(axis=0)
    binary_mask  = (max_prob_map > threshold).astype(int)

    print(f'  Multi-scale max: max_prob={max_prob_map.max():.3f}  '
          f'detections={int(binary_mask.sum())}')

    eval_metrics = None
    if true_anom_start is not None and true_anom_end is not None:
        true_mask = np.zeros(n)
        true_mask[true_anom_start:true_anom_end] = 1.0
        tp   = int(((binary_mask == 1) & (true_mask == 1)).sum())
        fp   = int(((binary_mask == 1) & (true_mask == 0)).sum())
        fn   = int(((binary_mask == 0) & (true_mask == 1)).sum())
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        eval_metrics = {'TP': tp, 'FP': fp, 'FN': fn,
                        'precision': prec, 'recall': rec, 'F1': f1}
        print(f'\n  TP={tp}  FP={fp}  FN={fn}')
        print(f'  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}')

    return max_prob_map, binary_mask, eval_metrics
