# Dataset Map — INCOIS SST Forecasting (M. Medha)

Maps every input dataset to the production scripts that use them.

## Master Data Files

| File | Shape | Format | Description | Used By |
|------|-------|--------|-------------|---------|
| `master-harry-appended/master_region_data_new.npy` | (16300, 60, 50) | float32 | SST absolute temperature, 60×50 grid | 56, 57, 58f, 59 |
| `master-harry-appended/master_region_anomalies_new.npy` | (16300, 60, 50) | float32 | SST anomaly (data - long-term daily mean) | 56, 57, 58f |
**Source:** Derived from OISST v2.1 satellite SST — Arabian Sea region (5°N–20°N, 60°E–72°E) at 0.25° resolution.

---

## Model Checkpoints (in `outputs/`)

| File | Size | Used By |
|------|------|---------|
| `outputs/lstm-outputs/model_best.pt` | 460 KB | 56 (LSTM trained weights) |
| `outputs/nbeats-outputs/model_best.pt` | 9.6 MB | 57 (N-BEATS trained weights) |

Moirai (58f) uses pre-trained HuggingFace weights (moirai-1.0-R-small, 55M params), auto-downloaded.

---

## Rolling Predictions (in `model_comparison/`)

| File | Size | Used By |
|------|------|---------|
| `model_comparison/rolling_predictions_56.csv` | 14 KB | 59 (LSTM baseline predictions) |
| `model_comparison/rolling_predictions_57.csv` | 14 KB | 59 (N-BEATS predictions) |
| `model_comparison/rolling_predictions_58f.csv` | 12 KB | 59 (Moirai champion predictions) |
| `model_comparison/skill_scores.csv` | 2 KB | 59 (aggregate skill scores) |

---

## Validation Data (in `validation_data/`)

| File | Size | Used By |
|------|------|---------|
| `validation_data/Argo_validsation_TSFM.xlsx` | 15 KB | `validate_argo_spatial_models.py` |
| `validation_data/argo_validation_tsfm.csv` | 4 KB | `validate_argo_spatial_models.py` |
| `validation_data/master_appended_tsfm.csv` | 4 KB | `validate_argo_spatial_models.py` |
| `validation_data/reanalysis_tsfm.csv` | 4 KB | `validate_argo_spatial_models.py` |
| `validation_data/Argo_validsation_TSFM_filtered_to_master.csv` | 11 KB | `validate_argo_spatial_models.py` |
| `validation_data/Argo_validsation_TSFM_reanalysis.nc` | 2.3 MB | `validate_argo_spatial_models.py` |

---

## Path Resolution

| Dataset | Lookup Order |
|---------|-------------|
| Master .npy files | `input_datasets/master-harry-appended/` |
| Model checkpoints | `outputs/<model>/model_best.pt` |
| Rolling predictions | `model_comparison/rolling_predictions_<script>.csv` |
| Validation data | `validation_data/` |
| Moirai weights | Auto-downloaded from HuggingFace |

---

*June 15, 2026*
