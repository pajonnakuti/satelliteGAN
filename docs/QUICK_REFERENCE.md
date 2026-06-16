# Quick Reference - Key Metrics

## Project Status: Complete

---

## Final Results

| Model | 7d RMSE | 14d RMSE | 30d RMSE | 7d R | 30d R |
|-------|---------|----------|----------|------|-------|
| **Moirai Fine-Tuned** | **0.108°C** | **0.122°C** | **0.134°C** | **0.938** | **0.842** |
| Moirai Zero-Shot | 0.129°C | 0.148°C | 0.161°C | 0.898 | 0.794 |
| N-BEATS Optimized | 0.124°C | 0.141°C | 0.158°C | 0.912 | 0.803 |
| LSTM Baseline | 0.138°C | 0.151°C | 0.165°C | 0.882 | 0.765 |

---

## Winner Selection

| Priority | Model | Value | Notes |
|----------|-------|-------|-------|
| Formal publication | **Moirai FT** | 0.108°C | Best 7d RMSE |
| Best correlation | **Moirai FT** | R=0.938 | Best 7d Pearson R |
| Best February | **Moirai ZS** | 0.193°C | Cold-dip period |
| Best March | **LSTM / N-BEATS** | 0.102°C | Tied |
| Best Argo | **Moirai FT** | 0.298°C | In-situ validation |

---

## Argo Validation

| Model | RMSE | R |
|-------|------|---|
| **Moirai (Ridge)** | **0.298°C** | **0.93** |
| N-BEATS | 0.311°C | 0.91 |
| LSTM | 0.320°C | 0.89 |

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Data days | ~16,300 |
| Spatial grid | 60 × 50 (3,000 pixels) |
| Grid resolution | 0.25° |
| Domain | 5°N–20°N, 60°E–72°E |
| Forecast horizons | 7d, 14d, 30d |
| Evaluation period | Jan–Mar 2026 (90 days) |
| Argo profiles | 37 |
| Pipeline improvement | 0.013°C at 30d |

---

## Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `56_lstm_rolling_7day_v7_v2.py` | LSTM baseline | ✅ Final |
| `57_nbeats_rolling_7day_v2_v2.py` | N-BEATS optimized | ✅ Final |
| `58f_moirai_regional_gradient.py` | Moirai champion | ✅ **Best (0.108)** |
| `model_comparison/59_model_comparison.py` | Comparison viz | ✅ Final |
| `validation_data/validate_argo_spatial_models.py` | Argo validation | ✅ Final |

---

*June 15, 2026*
