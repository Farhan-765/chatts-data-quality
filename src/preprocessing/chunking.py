"""
Chunking helpers - extract fixed-length windows from time series arrays.
All functions return np.float32 arrays matching the ChatTS 512-point limit.
"""

import numpy as np
import pandas as pd


def get_chunk(
    series_values: np.ndarray,
    center_idx: int,
    chunk_size: int = 512,
) -> tuple[np.ndarray, int, int]:
    """
    Extract a fixed-length chunk centred on center_idx.

    Returns
    -------
    chunk  : np.ndarray float32, length chunk_size (or less at boundaries)
    start  : actual start index in the original array
    end    : actual end index in the original array
    """
    half  = chunk_size // 2
    start = max(0, center_idx - half)
    end   = min(len(series_values), start + chunk_size)
    start = max(0, end - chunk_size)
    return series_values[start:end].astype(np.float32), start, end


def get_time_chunk(
    df_series: pd.Series,
    start_time: str,
    end_time: str,
    pad_hours: int = 6,
) -> np.ndarray:
    """
    Extract a chunk between two timestamps with optional padding.

    Parameters
    ----------
    df_series  : pd.Series with DatetimeIndex
    start_time : ISO 8601 string e.g. '2025-12-01'
    end_time   : ISO 8601 string e.g. '2025-12-22'
    pad_hours  : hours of context to include before/after the window

    Returns
    -------
    np.ndarray float32 of the selected points
    """
    pad  = pd.Timedelta(hours=pad_hours)
    mask = (
        (df_series.index >= pd.Timestamp(start_time) - pad) &
        (df_series.index <= pd.Timestamp(end_time)   + pad)
    )
    chunk = df_series[mask].values.astype(np.float32)
    print(f'Chunk: {len(chunk)} points  |  '
          f'{df_series[mask].index[0]} to {df_series[mask].index[-1]}')
    return chunk


def downsample(values: np.ndarray, target: int = 1024) -> np.ndarray:
    """
    Uniformly downsample values to target length using index selection.
    Used to fit long signals (8640 pts) into the segmentation pipeline.
    """
    if len(values) <= target:
        return values.astype(np.float32)
    idx = np.linspace(0, len(values) - 1, target, dtype=int)
    return values[idx].astype(np.float32)
