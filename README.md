# Arabian Sea SST Forecasting — Deep Learning Ensemble

**Author:** M. Medha
**Institution:** Department of CSE, ICFAI Foundation for Higher Education (IFHE)
**Context:** Research Internship, INCOIS (Indian National Centre for Ocean Information Services)
**Date:** June 2026
**Project:** Sea Surface Temperature forecasting over the Arabian Sea (5°N-20°N, 60°E-72°E) using LSTM, N-BEATS, and Salesforce Moirai.

---

## CONTENTS OF THIS FOLDER

### Production Scripts

1. **56_lstm_rolling_7day_v7_v2.py** — LSTM production baseline
   Rolling 7/14/30-day forecast, 90-day evaluation (Jan-Mar 2026).
   RMSE: 7d=0.138°C, 14d=0.151°C, 30d=0.165°C.

2. **57_nbeats_rolling_7day_v2_v2.py** — N-BEATS optimized
   Rolling 7-day with stationary residual preprocessing + Huber Loss.
   RMSE: 7d=0.124°C, 14d=0.141°C, 30d=0.158°C.

3. **58f_moirai_regional_gradient.py** — Moirai fine-tuned (Champion)
   Foundation model with 4-cardinal gradient injection + Ridge correction.
   RMSE: 7d=**0.108°C**, 14d=0.122°C, 30d=0.134°C. **Best overall.**

### Folders

| Folder | Description |
|--------|-------------|
| `docs/` | 9 documentation files (see README inside) |
| `model_comparison/` | Comparison script + rolling prediction CSVs |
| `validation_data/` | Argo validation script + 6 data files + validation outputs |
| `input_datasets/` | Dataset map |
| `outputs/` | Extracted model outputs by category |

---

## Argo Validation Results

| Model | RMSE | R | N |
|-------|------|---|----|
| **Moirai (Ridge-Corrected)** | **0.298°C** | **0.93** | 37 |
| N-BEATS | 0.311°C | 0.91 | 37 |
| LSTM | 0.320°C | 0.89 | 37 |

---

## Run Commands

```bash
python 58f_moirai_regional_gradient.py   # Champion
python 57_nbeats_rolling_7day_v2_v2.py   # N-BEATS
python 56_lstm_rolling_7day_v7_v2.py     # LSTM baseline

# Comparison & Validation
cd model_comparison
python 59_model_comparison.py

cd ../validation_data
python validate_argo_spatial_models.py
```

---

*June 2026*
