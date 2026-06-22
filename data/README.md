# Data Directory

## Industrial Dataset

The industrial sensor data used in this thesis is proprietary and cannot be
distributed publicly. Access requires a Timeseer account with credentials for
the UHasselt research site.

### How to obtain the data

Set the following environment variables in your `.env` file:

```
TIMESEER_API_TOKEN=<your token>
TIMESEER_BASE_URL=https://app.timeseer.ai
TIMESEER_TENANT=UHasselt
```

Then fetch any signal with:

```python
from src.data.timeseer_client import fetch_series_api
vals, idx = fetch_series_api('R1-AT-102-COND', 'Reactor 1')
```

### Dataset specification

| Property       | Value                                      |
|----------------|--------------------------------------------|
| Signals        | 55 industrial sensor tags                  |
| Period         | Nov 14 2025 – Feb 12 2026 (~90 days)       |
| Sampling       | 15-minute intervals                        |
| Points/signal  | 8,640                                      |
| Plant areas    | 7 (Reactor 1 & 2, Separator, Distillation, |
|                | Wastewater Treatment, Utilities, Packaging)|
| Ground truth   | Manually annotated (see `configs/signals.yaml`) |

### Reactor 1 CSV files (offline mode)

Reactor 1 signals were also exported as CSV files for local development.
Place them in `data/reactor1_csv/` with columns `ts,value`.

Load with:
```python
from src.data.csv_loader import load_reactor1_csv
reactor1 = load_reactor1_csv('data/reactor1_csv/')
```

## Training tensors (segmentation pipeline)

The segmentation pipeline requires pre-computed embedding tensors.
Generate them by running the training data scripts on VSC:

```bash
python scripts/generate_industrial_training_data.py
python scripts/generate_synthetic_training_data.py
```

This produces `X_ts_combined.pt`, `X_txt_combined.pt`, `y_combined.pt`
in `$VSC_SCRATCH/ChatTS/timeseer_data/`.
