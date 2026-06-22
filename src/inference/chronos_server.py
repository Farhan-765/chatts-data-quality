"""
Persistent Chronos subprocess server + embedding cache.

Chronos (chronos-t5-small) and ChatTS have an `accelerate` version conflict,
so Chronos runs in a SEPARATE conda environment (chatts_chronos) via a
persistent subprocess server that reads JSON from stdin and writes embeddings
to stdout. The ready signal is sent to stderr.

Usage
-----
    from src.inference.chronos_server import start_server, get_chronos_embedding_cached, shutdown_server

    start_server()
    emb = get_chronos_embedding_cached(signal_array)   # np.ndarray shape (512,)
    shutdown_server()
"""

import os
import json
import subprocess
import time
import pickle
import numpy as np

VSC_SCRATCH     = os.environ.get('VSC_SCRATCH', '/scratch/leuven/375/vsc37531')
CHRONOS_CKPT    = os.path.join(VSC_SCRATCH, 'ChatTS', 'ckpt', 'chronos-t5-small')
CHRONOS_PYTHON  = os.path.join(VSC_SCRATCH, 'chatTS_new_env', 'bin', 'python3.11')
CHRONOS_PKGS    = os.path.join(VSC_SCRATCH, 'chatTS_new_env', 'lib', 'python3.11', 'site-packages')
CACHE_PATH      = os.path.join(VSC_SCRATCH, 'ChatTS', 'timeseer_data', 'chronos_cache.pkl')
SERVER_SCRIPT   = '/tmp/chronos_server.py'

_chronos_proc  = None
_chronos_cache: dict = {}

_SERVER_SCRIPT_CONTENT = f"""
import sys, json, torch
sys.path.insert(0, '{CHRONOS_PKGS}')
from chronos import ChronosPipeline
pipeline = ChronosPipeline.from_pretrained(
    '{CHRONOS_CKPT}',
    torch_dtype=torch.float32,
)
sys.stderr.write('CHRONOS_READY\\n')
sys.stderr.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        signal = torch.tensor(json.loads(line)).unsqueeze(0)
        with torch.no_grad():
            embedding, _ = pipeline.embed(signal)
        emb = embedding.squeeze(0).mean(dim=0).tolist()
        print(json.dumps(emb))
        sys.stdout.flush()
    except Exception as e:
        sys.stderr.write(f'ERROR: {{e}}\\n')
        sys.stderr.flush()
        print(json.dumps([0.0] * 512))
        sys.stdout.flush()
"""


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)


def start_server(timeout: int = 60) -> None:
    """Write server script and launch subprocess. Blocks until CHRONOS_READY."""
    global _chronos_proc, _chronos_cache

    _chronos_cache = load_cache()
    print(f'Loaded {len(_chronos_cache)} cached embeddings from disk.')

    # Kill existing server if running
    shutdown_server()

    with open(SERVER_SCRIPT, 'w') as f:
        f.write(_SERVER_SCRIPT_CONTENT)

    _chronos_proc = subprocess.Popen(
        [CHRONOS_PYTHON, SERVER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    print('Waiting for Chronos to load...')
    start = time.time()
    ready = False
    while time.time() - start < timeout:
        line = _chronos_proc.stderr.readline().strip()
        if 'CHRONOS_READY' in line:
            ready = True
            print(f'Chronos ready in {time.time()-start:.1f}s')
            break
        if line:
            print(f'  Chronos: {line}')

    if not ready:
        print('WARNING: Chronos did not send ready signal - proceeding anyway.')


def shutdown_server() -> None:
    """Terminate the Chronos subprocess and persist the cache."""
    global _chronos_proc
    if _chronos_proc is not None:
        try:
            _chronos_proc.terminate()
            time.sleep(1)
        except Exception:
            pass
        _chronos_proc = None
    if _chronos_cache:
        save_cache(_chronos_cache)


def get_chronos_embedding_cached(signal_array: np.ndarray) -> np.ndarray | None:
    """
    Get Chronos-T5-small embedding for a signal window.
    Returns cached result if available; otherwise queries the persistent server.

    Parameters
    ----------
    signal_array : np.ndarray float32, shape (window_size,)

    Returns
    -------
    np.ndarray float32, shape (512,)  or None on error
    """
    key = hash(signal_array.tobytes())
    if key in _chronos_cache:
        return _chronos_cache[key]

    if _chronos_proc is None:
        print('ERROR: Chronos server not started. Call start_server() first.')
        return None

    try:
        _chronos_proc.stdin.write(json.dumps(signal_array.tolist()) + '\n')
        _chronos_proc.stdin.flush()
        line = _chronos_proc.stdout.readline().strip()
        if not line:
            print('WARNING: empty response from Chronos server')
            return None
        emb = np.array(json.loads(line), dtype=np.float32)
        _chronos_cache[key] = emb
        if len(_chronos_cache) % 20 == 0:
            save_cache(_chronos_cache)
        return emb
    except Exception as e:
        print(f'Chronos embedding error: {e}')
        return None
