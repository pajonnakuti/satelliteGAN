# Colab Guide: Argo Spatial Validation (LSTM / N-BEATS / Moirai)

This guide runs spatial validation against Argo points using LSTM, N-BEATS, and Moirai models in Google Colab.

## 1) Required inputs

Prepare these files in Colab at `/content/`:

- `validation_data/argo_validation_tsfm.csv`
- `validation_data/master_appended_tsfm.csv`
- `validation_data/reanalysis_tsfm.csv`
- `master_region_data_new.npy`
- `master_region_anomalies_new.npy`
- LSTM checkpoint: `outputs/56_lstm_rolling_7day/model_best.pt`
- N-BEATS checkpoint: `outputs/57_nbeats_rolling_7day/model_best.pt` (or `57_nbeats_rolling_7day_6_temp/model_best.pt`)

Moirai uses pretrained weights from HuggingFace (auto-downloaded, no local checkpoint needed).

Upload via Colab file uploader or clone your repo:

```python
from google.colab import files
uploaded = files.upload()
```

If filenames differ, set env vars `ARGO_CSV`, `MASTER_CSV`, `REAN_CSV` to the full paths.

## 2) Colab runtime setup

Enable GPU: **Runtime > Change runtime type > T4 GPU** (recommended).

Install model packages:

```bash
pip install -q torch numpy pandas scipy matplotlib
pip install -q hydra-core jaxtyping lightning einops gluonts datasets accelerate python-dotenv
pip install -q --no-deps uni2ts
```

Set working directory and environment:

```python
import os
os.chdir('/content')

# Optional: override default data paths
os.environ['ARGO_CSV'] = '/content/validation_data/argo_validation_tsfm.csv'
os.environ['MASTER_CSV'] = '/content/validation_data/master_appended_tsfm.csv'
os.environ['REAN_CSV'] = '/content/validation_data/reanalysis_tsfm.csv'
```

## 3) Folder layout

Organize your Colab workspace like this:

```
/content/
├── validation_data/
│   ├── argo_validation_tsfm.csv
│   ├── master_appended_tsfm.csv
│   └── reanalysis_tsfm.csv
├── master_region_data_new.npy
├── master_region_anomalies_new.npy
├── outputs/
│   ├── 56_lstm_rolling_7day/
│   │   └── model_best.pt
│   └── 57_nbeats_rolling_7day/
│       └── model_best.pt
├── validate_argo_spatial_models.py
└── validation_outputs/               # Generated outputs
```

## 4) Run Argo spatial validation

Place `validate_argo_spatial_models.py` in `/content/` and run:

### All models (default)

```bash
python validate_argo_spatial_models.py \
  --models lstm,nbeats,moirai \
  --lstm_ckpt /content/outputs/56_lstm_rolling_7day/model_best.pt \
  --nbeats_ckpt /content/outputs/57_nbeats_rolling_7day/model_best.pt \
  --output_dir /content/validation_outputs
```

### Single model

**LSTM only:**

```bash
python validate_argo_spatial_models.py \
  --models lstm \
  --lstm_ckpt /content/outputs/56_lstm_rolling_7day/model_best.pt \
  --output_dir /content/validation_outputs
```

**N-BEATS only:**

```bash
python validate_argo_spatial_models.py \
  --models nbeats \
  --nbeats_ckpt /content/outputs/57_nbeats_rolling_7day/model_best.pt \
  --output_dir /content/validation_outputs
```

**Moirai only:**

```bash
python validate_argo_spatial_models.py \
  --models moirai \
  --output_dir /content/validation_outputs
```

## 5) Outputs

Files saved to `/content/validation_outputs/`:

- `argo_spatial_validation_predictions.csv` (per-point predictions with columns: `argo_temp`, `lstm_pred`, `nbeats_pred`, `moirai_pred`)
- `argo_spatial_validation_metrics.csv` (RMSE/MAE/R/R2/slope/intercept per model)
- `plot_overlay_timeseries.png` (daily mean + Argo points)
- `plot_correlation_scatter.png` (model vs Argo scatter, one panel per model)

**Note:** `/content` is ephemeral. Download outputs before session ends:

```python
import zipfile
import os

os.chdir('/content')
with zipfile.ZipFile('validation_outputs.zip', 'w') as z:
    for root, dirs, files in os.walk('validation_outputs'):
        for f in files:
            z.write(os.path.join(root, f))

from google.colab import files
files.download('validation_outputs.zip')
```

## 6) Model-specific notes

### LSTM

- Uses `LevelConditionedLSTM` architecture (4 input channels, hidden=64, 2 layers).
- Checkpoint format: `model_best.pt` from `56_lstm_rolling_7day` training run.
- SEQ_LEN=60, HORIZON=7, rolling 7-day forecast with adaptive drift correction.
- Batch size tuned for Colab T4.

### N-BEATS

- Single forward pass for all 7 horizon days (no autoregressive inference).
- Uses 4 stacks (Trend/Seasonal/Generic x2), 4 blocks each, hidden_dim=256.
- SEQ_LEN=90, HORIZON=7, residual de-trending with rolling mean.
- Checkpoint format: `model_best.pt` from `57_nbeats_rolling_7day` training run.

### Moirai

- Foundation model from Salesforce `uni2ts` ecosystem.
- Uses pretrained `Salesforce/moirai-1.0-R-small` (auto-downloaded on first run).
- Context length: 365 days, prediction length: 7 days.
- Requires `uni2ts` package (install via pip commands above).

## 7) General notes

- SST is computed from Argo as min-pressure per profile/time.
- Reanalysis SST was converted from Kelvin to Celsius in `reanalysis_tsfm.csv`.
- Colab sessions timeout after inactivity; save checkpoints and outputs regularly.
- Use `num_workers=0` in any DataLoader to avoid fork failures in Colab.
- All models use adaptive offset correction and post-gain calibration.
