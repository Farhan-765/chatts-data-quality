"""
Disk cache for Chronos embeddings.
Embeddings are expensive (~8s each), so we persist them between sessions.
"""

import os
import pickle

CACHE_PATH = os.environ.get(
    'CHRONOS_CACHE_PATH',
    '/scratch/leuven/375/vsc37531/ChatTS/timeseer_data/chronos_cache.pkl',
)


def load_cache(path: str = CACHE_PATH) -> dict:
    """Load the embedding cache from disk. Returns empty dict if file absent."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            cache = pickle.load(f)
        print(f'Loaded {len(cache)} cached embeddings from {path}')
        return cache
    return {}


def save_cache(cache: dict, path: str = CACHE_PATH) -> None:
    """Persist the embedding cache to disk."""
    with open(path, 'wb') as f:
        pickle.dump(cache, f)
