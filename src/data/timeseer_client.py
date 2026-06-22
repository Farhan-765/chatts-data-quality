"""
Timeseer API client.
Credentials are read from environment variables - never hardcoded.
Set TIMESEER_API_TOKEN, TIMESEER_BASE_URL, and TIMESEER_TENANT in .env.
"""

import os
import subprocess
import json
import numpy as np
import pandas as pd

BASE_URL  = os.environ.get('TIMESEER_BASE_URL', 'https://app.timeseer.ai')
TENANT    = os.environ.get('TIMESEER_TENANT', 'UHasselt')
API_TOKEN = os.environ.get('TIMESEER_API_TOKEN', '')
DS_NAME   = 'Industrial analyzers'

FROM_DT = '2025-11-14T00:00:00Z'
TO_DT   = '2026-02-12T00:00:00Z'

AREAS = {
    'Reactor 1':            'Analyzers - area Reactor 1',
    'Reactor 2':            'Analyzers - area Reactor 2',
    'Distillation':         'Analyzers - area Distillation',
    'Packaging':            'Analyzers - area Packaging',
    'Separator':            'Analyzers - area Separator',
    'Utilities':            'Analyzers - area Utilities',
    'Wastewater Treatment': 'Analyzers - area Wastewater Treatment',
}


def api_get(endpoint: str) -> dict:
    """GET request to the Timeseer REST API."""
    result = subprocess.run([
        'curl', '-s', '-4', '-X', 'GET',
        f'{BASE_URL}{endpoint}',
        '-H', f'x-tenant: {TENANT}',
        '-H', 'accept: application/json',
        '-H', f'Authorization: Basic {API_TOKEN}',
    ], capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def api_post(endpoint: str, body: dict) -> dict:
    """POST request to the Timeseer REST API."""
    result = subprocess.run([
        'curl', '-s', '-4', '-X', 'POST',
        f'{BASE_URL}{endpoint}',
        '-H', f'x-tenant: {TENANT}',
        '-H', 'accept: application/json',
        '-H', 'Content-Type: application/json',
        '-H', f'Authorization: Basic {API_TOKEN}',
        '-d', json.dumps(body),
    ], capture_output=True, text=True, timeout=60)
    return json.loads(result.stdout)


def list_series_api(area_name: str) -> list[str]:
    """Return list of tag names available in a plant area."""
    view = AREAS[area_name]
    resp = api_get(
        f'/public/api/v1/data-services/view/series'
        f'?dataServiceName={DS_NAME.replace(" ", "%20")}'
        f'&dataServiceViewName={view.replace(" ", "%20")}'
    )
    return [s['tags']['series name'] for s in resp['body']]


def fetch_series_api(
    tag: str,
    area_name: str,
    from_dt: str = FROM_DT,
    to_dt: str = TO_DT,
) -> tuple[np.ndarray | None, pd.DatetimeIndex | None]:
    """
    Fetch a single time-series tag from the Timeseer API.

    Returns
    -------
    vals : np.ndarray of float32, shape (N,)
    idx  : pd.DatetimeIndex, length N
    Both are None on API error.
    """
    view = AREAS[area_name]
    body = {'body': {
        'dataServiceName':     DS_NAME,
        'dataServiceViewName': view,
        'selector': {
            'source': 'Industrial Analyzers',
            'tags':   {'series name': tag},
        },
        'from': from_dt,
        'to':   to_dt,
    }}
    resp = api_post('/public/api/v1/data-services/view/get-data', body)

    if 'message' in resp:
        print(f'API error for {tag}: {resp["message"]}')
        return None, None

    df = pd.DataFrame(resp['body'])
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df = df.sort_values('ts').set_index('ts')
    vals = df['value'].values.astype(np.float32)
    print(f'Fetched {tag} from {area_name}: {len(vals)} points '
          f'({df.index[0].date()} → {df.index[-1].date()})')
    return vals, df.index
