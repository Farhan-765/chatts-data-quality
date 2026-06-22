"""
CSV loader for Reactor 1 local data (offline fallback when API is unavailable).
CSV files were exported from Timeseer and stored in the reactor1_csv/ directory.
Each file has columns: ts (UTC timestamp), value (float).
"""

import os
import pandas as pd


def load_reactor1_csv(csv_dir: str) -> pd.DataFrame:
    """
    Load all Reactor 1 CSV files from csv_dir and merge into a single DataFrame.

    Parameters
    ----------
    csv_dir : path to directory containing per-tag CSV files

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC) and one column per tag
    """
    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
    if not csv_files:
        raise FileNotFoundError(f'No CSV files found in {csv_dir}')

    dfs = []
    for fname in csv_files:
        path = os.path.join(csv_dir, fname)
        df = pd.read_csv(path, parse_dates=['ts'], index_col='ts')
        dfs.append(df)
        print(f'Loaded {fname:55s}  shape={df.shape}')

    merged = pd.concat(dfs, axis=1)
    merged.sort_index(inplace=True)
    print(f'\nMerged : {merged.shape}')
    print(f'Range  : {merged.index[0]}  to  {merged.index[-1]}')
    print(f'Columns: {list(merged.columns)}')
    return merged
