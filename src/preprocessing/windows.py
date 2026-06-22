"""
Sliding-window extractor for the segmentation pipeline.
Windows feed into get_chronos_embedding_cached → MLP.
"""

import numpy as np


def extract_windows(
    signal: np.ndarray,
    window_size: int,
    stride_ratio: float = 0.5,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """
    Extract sliding windows from signal with overlap.

    Parameters
    ----------
    signal       : 1-D float array
    window_size  : number of points per window
    stride_ratio : fraction of window_size used as stride (0.5 = 50% overlap)

    Returns
    -------
    windows   : list of np.ndarray, each shape (window_size,)
    positions : list of (start, end) index tuples
    """
    stride = max(1, int(window_size * stride_ratio))
    windows   = []
    positions = []
    for start in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[start:start + window_size])
        positions.append((start, start + window_size))
    return windows, positions


def build_training_sample(
    signal: np.ndarray,
    label_mask: np.ndarray,
    query_text: str,
    window_size: int = 256,
    encode_text_fn=None,
    get_chronos_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Build one training sample from a labelled signal.

    Parameters
    ----------
    signal         : 1-D float array (full signal)
    label_mask     : binary array same length as signal, 1 = anomaly
    query_text     : natural-language query string
    window_size    : sliding window size
    encode_text_fn : callable(str) → np.ndarray[5120]  (encode_text_query)
    get_chronos_fn : callable(np.ndarray) → np.ndarray[512] (get_chronos_embedding_cached)

    Returns
    -------
    ts_embeddings  : np.ndarray [N_windows, 512]
    text_embedding : np.ndarray [5120]
    window_labels  : np.ndarray [N_windows], float  (1 if window overlaps anomaly)
    positions      : list of (start, end) tuples
    """
    windows, positions = extract_windows(signal, window_size)

    text_emb = encode_text_fn(query_text)

    ts_embs       = []
    window_labels = []

    for (start, end), window in zip(positions, windows):
        emb = get_chronos_fn(np.array(window, dtype=np.float32))
        if emb is None:
            continue
        ts_embs.append(emb)
        window_labels.append(float(label_mask[start:end].max()))

    return (
        np.array(ts_embs,       dtype=np.float32),
        text_emb,
        np.array(window_labels, dtype=np.float32),
        positions,
    )
