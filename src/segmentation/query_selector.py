"""
LLM→Segmentation feedback loop: map a classification result to the most
relevant segmentation query, then run the MLP segmentation with that query.

The feedback loop:
  1. Classify signal with Approach 3 (ChatTS-14B or GPT-4o)
  2. Map predicted category → targeted segmentation query
  3. Run segment_signal() with that query
  4. Return both classification and localisation results

This closes the loop between coarse classification and fine localisation.
"""

import numpy as np

# Category → natural-language segmentation query
CATEGORY_QUERIES: dict[str, str] = {
    'A': 'Find gradual baseline drift',
    'B': 'Identify sudden extreme spikes returning to baseline',
    'C': 'Detect stale data or frozen sensor readings',
    'D': 'Find sudden permanent level shifts or phase changes',
    'E': None,  # Clean - no segmentation needed
    'G': 'Detect variance collapse or loss of signal variability',
    'L': 'Find intermittent negative or physically impossible values',
}

# Fallback when category is unknown / ambiguous
DEFAULT_QUERY = 'Find periods with abnormal dynamics'


def category_to_query(category: str) -> str | None:
    """
    Return the segmentation query for a predicted category code.

    Returns None if the signal is classified as Clean (E) - no
    segmentation is needed.
    """
    return CATEGORY_QUERIES.get(category.strip().upper(), DEFAULT_QUERY)


def feedback_segment(
    vals_ds: np.ndarray,
    predicted_category: str,
    mlp,
    encode_fn,
    chronos_fn,
    window_size: int = 32,
    threshold: float = 0.65,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """
    Run segmentation using a query derived from the classification result.

    Parameters
    ----------
    vals_ds              : downsampled signal (1024 pts typical)
    predicted_category   : single-letter category from classify step
    mlp                  : loaded TSAISegmentationMLP
    encode_fn            : text embedding function
    chronos_fn           : Chronos embedding function
    window_size          : segmentation window size
    threshold            : anomaly probability threshold

    Returns
    -------
    (prob_map, binary_mask, query_used)
    prob_map and binary_mask are None if category is 'E' (clean).
    """
    from src.segmentation.pipeline import segment_signal

    query = category_to_query(predicted_category)
    if query is None:
        return None, None, 'N/A (clean signal)'

    prob_map, mask = segment_signal(
        vals_ds, query, mlp, encode_fn, chronos_fn,
        window_size=window_size, threshold=threshold,
    )
    return prob_map, mask, query


def multiscale_feedback_segment(
    vals_ds: np.ndarray,
    predicted_category: str,
    mlp,
    encode_fn,
    chronos_fn,
    window_sizes: list[int] | None = None,
    threshold: float = 0.65,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """
    Multi-scale version: max-pool across window sizes using the
    category-derived query.
    """
    from src.segmentation.pipeline import multiscale_segmentation

    if window_sizes is None:
        window_sizes = [32, 64, 128, 256]

    query = category_to_query(predicted_category)
    if query is None:
        return None, None, 'N/A (clean signal)'

    prob_map, mask = multiscale_segmentation(
        vals_ds, query, mlp, encode_fn, chronos_fn,
        window_sizes=window_sizes, threshold=threshold,
    )
    return prob_map, mask, query
