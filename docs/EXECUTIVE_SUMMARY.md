# INCOIS Internship - Executive Summary

## Project: SST Forecasting System
**Author:** M. Medha
**Institution:** Department of CSE, ICFAI Foundation for Higher Education (IFHE)
**Context:** Research Internship, INCOIS
**Date:** June 2026

---

## Project Overview

This project developed a comprehensive Sea Surface Temperature forecasting system using three model families:
1. **Level-Conditioned LSTM** — Custom deep learning model with delta-prediction formulation
2. **N-BEATS** — Neural basis expansion with interpretable trend/seasonal decomposition
3. **Salesforce Moirai** — Patch-based time-series foundation transformer (fine-tuned)

All models were evaluated through a **4-Stage Causal Post-Processing Pipeline** — the central contribution of this work — which corrects systematic forecast biases through sequential stages.

## Data Scope

- **Spatial Grid**: 60 × 50 pixels (5°N–20°N, 60°E–72°E)
- **Time Period**: September 1, 1981 – March 31, 2026 (~44.7 years)
- **Forecast Horizons**: 7-day, 14-day, 30-day
- **Evaluation Period**: Jan–Mar 2026 (90 consecutive forecast days)
- **Independent Validation**: 37 in-situ Argo float profiles

---

## Final Results

### Moirai Fine-Tuned (Script 58f) — CHAMPION

| Metric | 7-Day | 14-Day | 30-Day |
|--------|-------|--------|--------|
| RMSE | **0.108°C** | **0.122°C** | **0.134°C** |
| MAE | 0.080°C | 0.091°C | 0.103°C |
| Pearson R | 0.938 | 0.875 | 0.842 |

### N-BEATS Optimized (Script 57)

| Metric | 7-Day | 14-Day | 30-Day |
|--------|-------|--------|--------|
| RMSE | 0.124°C | 0.141°C | 0.158°C |
| MAE | 0.091°C | 0.108°C | 0.120°C |
| Pearson R | 0.912 | 0.849 | 0.803 |

### LSTM Baseline (Script 56)

| Metric | 7-Day | 14-Day | 30-Day |
|--------|-------|--------|--------|
| RMSE | 0.138°C | 0.151°C | 0.165°C |
| MAE | 0.102°C | 0.115°C | 0.128°C |
| Pearson R | 0.882 | 0.814 | 0.765 |

---

## 4-Stage Post-Processing Pipeline Contribution

The pipeline contributed **0.013°C RMSE reduction** at the 30-day horizon:

1. **Stage 1** — Additive Quartile Bias Correction (thermal-state-dependent bias)
2. **Stage 2** — Per-Pixel Spatial Correction (coastal upwelling zones)
3. **Stage 3** — Gated Multiplicative Scale Correction (R² > 0.60 gate)
4. **Stage 4** — Trend-Aware Nudge (extended-horizon drift stabilization)

---

## Argo Float Validation

Independent validation against 37 Argo float profiles (Jan–Mar 2026):

| Model | RMSE | R |
|-------|------|---|
| **Moirai (Ridge-Corrected)** | **0.298°C** | **0.93** |
| N-BEATS | 0.311°C | 0.91 |
| LSTM | 0.320°C | 0.89 |

---

## Key Findings

1. **Moirai fine-tuned is the champion**: RMSE 0.108°C at 7-day horizon
2. **4-Stage Pipeline delivers**: 0.013°C RMSE reduction at 30-day horizon
3. **N-BEATS second**: RMSE 0.124°C at 7-day, interpretable basis functions
4. **LSTM baseline**: Solid baseline at 0.138°C at 7-day
5. **February cold-dip** is hardest period: LSTM 0.215°C, N-BEATS 0.213°C, Moirai 0.193°C
6. **Argo confirms hierarchy**: Moirai generalizes best to in-situ data

---

## Documentation Structure

```
upload folder/
├── README.md                     ← This guide
├── 56/57/58f*.py                 ← Production scripts
├── docs/
│   ├── EXECUTIVE_SUMMARY.md      ← This file
│   ├── FINAL_RESULTS_TABLE.md    ← All verified results
│   ├── MODEL_COMPARISON.md       ← Detailed model comparison
│   ├── QUICK_REFERENCE.md        ← Key metrics at a glance
│   ├── VERIFICATION_PROOFS.md    ← Source file references
│   ├── SCRIPT_INDEX.md           ← All scripts
│   ├── COLAB_ARGO_*.md           ← Argo validation Colab guide
│   └── manuscript-medha.md       ← IEEE-style research paper
├── model_comparison/             ← Comparison script + CSVs
├── validation_data/              ← Argo validation script + data + outputs
├── input_datasets/               ← Master data files + map
└── outputs/                      ← Extracted model outputs
```

---

*Last Updated: June 15, 2026*
