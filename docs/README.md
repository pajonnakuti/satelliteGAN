# Documentation Folder - README

This folder contains all the formal documentation files for the INCOIS internship SST forecasting project.

## Quick Start

**For a quick overview:** Start with `QUICK_REFERENCE.md`

**For detailed analysis:** Read `MODEL_COMPARISON.md`

**For verification:** Check `VERIFICATION_PROOFS.md`

---

## File List

| File | Purpose |
|------|---------|
| `EXECUTIVE_SUMMARY.md` | High-level overview (2 pages) |
| `FINAL_RESULTS_TABLE.md` | All runs leaderboard with verified metrics |
| `MODEL_COMPARISON.md` | Detailed multi-model comparison |
| `QUICK_REFERENCE.md` | Key metrics at a glance |
| `VERIFICATION_PROOFS.md` | How to verify each result |
| `SCRIPT_INDEX.md` | All scripts listed |
| `COLAB_ARGO_SPATIAL_VALIDATION_LSTM_NBEATS_MOIRAI.md` | Argo validation Colab guide |
| `manuscript-medha.md` | IEEE-style research paper |
| `README.md` | This file |

---

## Output Directories

| Model | Location |
|-------|----------|
| LSTM 56 | `outputs/lstm-outputs/` |
| N-BEATS 57 | `outputs/nbeats-outputs/` |
| Moirai 58f | `outputs/moirai-outputs/` |
| Comparison 59 | `model_comparison/comparison-outputs/` |
| Argo Validation | `validation_data/argo-validation-outputs/` |

---

## Model Summary

| Model | Pipeline | 7d RMSE | 14d RMSE | 30d RMSE | Status |
|-------|----------|---------|----------|----------|--------|
| **Moirai Fine-Tuned** | 4-Stage Pipeline | **0.108°C** | **0.122°C** | **0.134°C** | **Champion** |
| N-BEATS Optimized | 4-Stage Pipeline | 0.124°C | 0.141°C | 0.158°C | Second |
| LSTM Baseline | 4-Stage Pipeline | 0.138°C | 0.151°C | 0.165°C | Baseline |

---

## 4-Stage Post-Processing Pipeline

The central contribution: (1) Additive Quartile Bias Correction, (2) Per-Pixel Spatial Correction, (3) Gated Multiplicative Scale Correction (R² > 0.60), (4) Trend-Aware Nudge. Cumulative impact: 0.013°C RMSE reduction at 30-day horizon.

---

*Last Updated: June 15, 2026*
