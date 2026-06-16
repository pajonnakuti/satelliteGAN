# Final Results - Verified Metrics

## LSTM Baseline (Script 56)

**Source**: `outputs/lstm-outputs/`

| Metric | Value |
|--------|-------|
| 7-Day RMSE | 0.138°C |
| 14-Day RMSE | 0.151°C |
| 30-Day RMSE | 0.165°C |
| 7-Day MAE | 0.102°C |
| 14-Day MAE | 0.115°C |
| 30-Day MAE | 0.128°C |
| 7-Day Pearson R | 0.882 |
| 14-Day Pearson R | 0.814 |
| 30-Day Pearson R | 0.765 |
| February RMSE | 0.215°C |
| March RMSE | 0.102°C |

---

## N-BEATS Optimized (Script 57)

**Source**: `outputs/nbeats-outputs/`

| Metric | Value |
|--------|-------|
| 7-Day RMSE | 0.124°C |
| 14-Day RMSE | 0.141°C |
| 30-Day RMSE | 0.158°C |
| 7-Day MAE | 0.091°C |
| 14-Day MAE | 0.108°C |
| 30-Day MAE | 0.120°C |
| 7-Day Pearson R | 0.912 |
| 14-Day Pearson R | 0.849 |
| 30-Day Pearson R | 0.803 |
| February RMSE | 0.213°C |
| March RMSE | 0.102°C |

---

## Moirai Zero-Shot (Script 58f Zero-Shot)

**Source**: `outputs/moirai-outputs/`

| Metric | Value |
|--------|-------|
| 7-Day RMSE | 0.129°C |
| 14-Day RMSE | 0.148°C |
| 30-Day RMSE | 0.161°C |
| 7-Day MAE | 0.095°C |
| 14-Day MAE | 0.112°C |
| 30-Day MAE | 0.124°C |
| 7-Day Pearson R | 0.898 |
| 14-Day Pearson R | 0.831 |
| 30-Day Pearson R | 0.794 |

---

## Moirai Fine-Tuned (Script 58f Fine-Tuned) — CHAMPION

**Source**: `outputs/moirai-outputs/`

| Metric | Value |
|--------|-------|
| 7-Day RMSE | **0.108°C** |
| 14-Day RMSE | **0.122°C** |
| 30-Day RMSE | **0.134°C** |
| 7-Day MAE | **0.080°C** |
| 14-Day MAE | **0.091°C** |
| 30-Day MAE | **0.103°C** |
| 7-Day Pearson R | **0.938** |
| 14-Day Pearson R | **0.875** |
| 30-Day Pearson R | **0.842** |

---

## Three-Model Comparison (Best Configuration Each)

| Model | 7d RMSE | 14d RMSE | 30d RMSE | 7d R | Feb RMSE | Mar RMSE |
|-------|---------|----------|----------|------|----------|----------|
| **Moirai FT** | **0.108°C** | **0.122°C** | **0.134°C** | **0.938** | **0.193°C** | 0.106°C |
| N-BEATS | 0.124°C | 0.141°C | 0.158°C | 0.912 | 0.213°C | **0.102°C** |
| LSTM Baseline | 0.138°C | 0.151°C | 0.165°C | 0.882 | 0.215°C | **0.102°C** |

---

## Argo Float Spatial Validation

**Source**: `validation_data/argo-validation-outputs/` and `validation_data/`

| Model | RMSE | R |
|-------|------|---|
| **Moirai (Ridge-Corrected)** | **0.298°C** | **0.93** |
| N-BEATS | 0.311°C | 0.91 |
| LSTM | 0.320°C | 0.89 |

**Pipeline**: `validation_data/validate_argo_spatial_models.py` → reads `validation_data/` CSVs → outputs to `validation_data/argo-validation-outputs/`

---

## 4-Stage Post-Processing Pipeline Impact

| Stage | Description | Impact |
|-------|-------------|--------|
| 1 | Additive Quartile Bias Correction | Thermal-state-dependent bias removal |
| 2 | Per-Pixel Spatial Correction | Spatial RMSE 0.93 → 0.18°C |
| 3 | Gated Multiplicative Scale (R² > 0.60) | Amplitude restoration |
| 4 | Trend-Aware Nudge | Extended-horizon stabilization |
| **Total** | **Cumulative** | **0.013°C RMSE reduction at 30d** |

---

## Verification Sources

| File/Dir | Content |
|----------|---------|
| `Book_Chapter_Report_LSTM_SST_FINAL.html` | Primary source for all metrics (Table 1) |
| `model_comparison/comparison-outputs/skill_scores.csv` | Monthly RMSE breakdown |
| `outputs/moirai-outputs/` | Moirai fine-tuned/zero-shot metrics |
| `validation_data/argo-validation-outputs/` | Argo validation metrics |
| `validation_data/argo_validation_tsfm.csv` | 37 Argo validation points |

---

*Verified: June 15, 2026*
