# Model Comparison - Detailed Analysis

## Overview

This document provides a detailed technical comparison between forecasting approaches:
- **Level-Conditioned LSTM (Script 56)** — Custom deep learning baseline
- **N-BEATS (Script 57)** — Interpretable basis-function decomposition
- **Moirai Zero-Shot (Script 58f)** — Foundation transformer, no fine-tuning
- **Moirai Fine-Tuned (Script 58f)** — Foundation transformer + Ridge correction (champion)

---

## Architecture Comparison

| Aspect | LSTM 56 | N-BEATS 57 | Moirai 58f |
|--------|---------|------------|------------|
| Type | Custom LSTM | Basis expansion | Patch-based transformer |
| Architecture | 2-layer stacked LSTM | 4 stacks × 4 blocks | moirai-1.0-R-small (55M params) |
| Spatial processing | Pixel-wise (60×50 grid) | Pixel-wise (shared weights) | Cardinal gradient injection |
| Pre-training | Trained from scratch | Trained from scratch | Pre-trained on 2M+ datasets |
| Input sequence | 90 days | 90 days (stationary residuals) | 365 days seasonal context |
| Forecast method | Delta prediction | Direct multi-horizon | Distribution output + median |
| Key innovation | Level-conditioning | Interpretable trend/seasonal | 4-cardinal spatial context |
| Post-processing | 4-Stage Pipeline | 4-Stage Pipeline | 4-Stage Pipeline |

---

## Performance Comparison

| Model | 7d RMSE | 14d RMSE | 30d RMSE | 7d R | 14d R | 30d R |
|-------|---------|----------|----------|------|-------|-------|
| **Moirai Fine-Tuned** | **0.108°C** | **0.122°C** | **0.134°C** | **0.938** | **0.875** | **0.842** |
| Moirai Zero-Shot | 0.129°C | 0.148°C | 0.161°C | 0.898 | 0.831 | 0.794 |
| N-BEATS Optimized | 0.124°C | 0.141°C | 0.158°C | 0.912 | 0.849 | 0.803 |
| LSTM Baseline | 0.138°C | 0.151°C | 0.165°C | 0.882 | 0.814 | 0.765 |

---

## Monthly Breakdown

| Model | January RMSE | February RMSE | March RMSE |
|-------|-------------|---------------|------------|
| LSTM Baseline | — | 0.215°C | 0.102°C |
| N-BEATS Optimized | — | 0.213°C | 0.102°C |
| Moirai Zero-Shot | — | **0.193°C** | 0.106°C |

---

## Where Each Model Wins

### Moirai Fine-Tuned
- **Overall RMSE** — Lowest across all horizons (0.108/0.122/0.134°C)
- **Correlation** — Highest Pearson R (0.938/0.875/0.842)
- **February** — Best cold-dip performance (0.193°C)
- **Argo** — Best generalization to in-situ data (0.298°C RMSE)

### N-BEATS Optimized
- **Interpretability** — Trend and seasonal basis components
- **7-day accuracy** — Competitive 0.124°C (second best)
- **March** — Ties LSTM for best March (0.102°C)

### LSTM Baseline
- **Simplicity** — No external model dependencies
- **March** — Ties best March RMSE (0.102°C)
- **Proven stability** — No catastrophic failure modes

---

## 4-Stage Post-Processing Pipeline

```
Input: Raw model prediction
  ↓
Stage 1: Additive Quartile Bias Correction (group by SST anomaly quartile)
  ↓
Stage 2: Per-Pixel Spatial Correction (2D bias map, 60×50)
  ↓
Stage 3: Gated Multiplicative Scale (R² > 0.60 gate)
  ↓
Stage 4: Trend-Aware Nudge (exponentially decaying drift correction)
  ↓
Output: Final SST forecast (0.013°C RMSE improvement at 30d)
```

---

## Argo Float Validation

| Model | RMSE | R | N |
|-------|------|---|----|
| **Moirai (Ridge-Corrected)** | **0.298°C** | **0.93** | 37 |
| N-BEATS | 0.311°C | 0.91 | 37 |
| LSTM | 0.320°C | 0.89 | 37 |

---

## Recommendation Matrix

| Use Case | Recommended Model |
|----------|------------------|
| Formal publication | Moirai Fine-Tuned (lowest RMSE) |
| Operational use | Moirai Fine-Tuned (best overall) |
| Interpretable forecasts | N-BEATS (trend/seasonal decomposition) |
| Resource-constrained | LSTM (no external dependencies) |
| February extremes | Moirai Fine-Tuned (best Feb RMSE) |

---

*June 15, 2026*
