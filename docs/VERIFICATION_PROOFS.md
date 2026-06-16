# Verification Proofs - Source References

## How to Verify Each Result

### LSTM Baseline (Script 56)

**Output Location**: `outputs/lstm-outputs/`

Verify by checking:
- `run_summary.csv` — All metrics (7d RMSE 0.138, 14d 0.151, 30d 0.165)
- `plot_timeseries.png` — 90-day time series overlay
- `plot_correlation.png` — Scatter correlation

### N-BEATS Optimized (Script 57)

**Output Location**: `outputs/nbeats-outputs/`

Verify by checking:
- `run_summary.csv` — All metrics (7d RMSE 0.124, 14d 0.141, 30d 0.158)
- `plot_timeseries.png` — 90-day time series overlay
- `plot_correlation.png` — Scatter correlation

### Moirai (Script 58f)

**Output Location**: `outputs/moirai-outputs/`

Verify by checking:
- `run_summary.csv` — All metrics (fine-tuned: 7d RMSE 0.108, zero-shot: 7d 0.129)
- `plot_timeseries.png` — 90-day time series overlay
- `plot_correlation.png` — Scatter correlation

### Comparison (Script 59)

**Output Location**: `model_comparison/comparison-outputs/`

Verify by checking:
- `skill_scores.csv` — Per-model, per-horizon RMSE/MAE/R
- `plotA_taylor_diagram.png` — Taylor diagram
- `plotB_error_density.png` — Error density curves
- `plotC_monthly_rmse_bar.png` — Monthly RMSE comparison
- `plotD_error_violin.png` — Error violin plots
- `plotE_timeseries_overlay.png` — Multi-model timeseries
- `plotF_skill_table.png` — Skill score table

### Argo Validation (validate_argo_spatial_models.py)

**Data Location**: `validation_data/` (6 CSV/NetCDF files)
**Output Location**: `validation_data/argo-validation-outputs/`

Verify by checking:
- `argo_spatial_validation_metrics.csv` — Aggregate metrics
- `argo_spatial_validation_predictions.csv` — Per-point predictions (37 rows)
- Expected: Moirai RMSE 0.298, N-BEATS 0.311, LSTM 0.320

---

## Verification Checklist

- [ ] LSTM 56 outputs show RMSE 0.138/0.151/0.165
- [ ] N-BEATS 57 outputs show RMSE 0.124/0.141/0.158
- [ ] Moirai 58f fine-tuned shows RMSE 0.108/0.122/0.134
- [ ] Comparison 59 has all 6 plot files + skill_scores.csv
- [ ] Argo validation has metrics for all 3 models
- [ ] Argo has 37 validation points
- [ ] Validation data directory has 6 files
- [ ] February RMSE values match: LSTM 0.215, N-BEATS 0.213, Moirai 0.193
- [ ] March RMSE values match: LSTM 0.102, N-BEATS 0.102, Moirai 0.106

---

## Primary Metric Source

**Table 1** in `Book_Chapter_Report_LSTM_SST_FINAL.html` contains the authoritative results for all models and horizons. Monthly breakdowns verified against `model_comparison/comparison-outputs/skill_scores.csv`.

---

*June 15, 2026*
