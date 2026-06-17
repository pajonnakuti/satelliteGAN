[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/Moirai-55M-FFD700?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/Salesforce/moirai-1.0-R-small)
[![Git LFS](https://img.shields.io/badge/Git_LFS-tracked-purple?style=flat&logo=gitlfs&logoColor=white)](https://git-lfs.com)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)]()

# 🌊 Arabian Sea SST Forecasting — Deep Learning Ensemble

### 7/14/30-Day Rolling Forecasts · LSTM · N-BEATS · Salesforce Moirai

*Arabian Sea · 5°N–20°N, 60°E–72°E · 0.25° Resolution · 60×50 Grid*

🏛️ **INCOIS** — Indian National Centre for Ocean Information Services  
👤 **M. Medha** · ICFAI Foundation for Higher Education (IFHE)

---

> 🏆 **Milestone:** Moirai Fine-Tuned achieved **0.108°C RMSE** at 7-day horizon with our novel **4-Stage Post-Processing Pipeline** — the **best foundation model result** for Arabian Sea SST forecasting to date. Argo validation confirms **0.298°C RMSE** against real in-situ ocean measurements.

---

## 📋 Table of Contents

- [🚀 What This Project Does](#-what-this-project-does)
- [🏗️ Architecture](#%EF%B8%8F-architecture)
- [🥇 Leaderboard](#-leaderboard)
- [🔧 4-Stage Post-Processing Pipeline](#-4-stage-post-processing-pipeline)
- [🌊 Argo Float Validation](#-argo-float-validation)
- [📁 Project Structure](#-project-structure)
- [⚡ Quick Start](#-quick-start)
- [📦 Dependencies](#-dependencies)
- [📚 Documentation](#-documentation)
- [📖 References](#-references)

---

## 🚀 What This Project Does

We compare **three fundamentally different approaches** for predicting Sea Surface Temperature over a **60×50 spatial grid** in the Arabian Sea at **7, 14, and 30-day horizons**:

| Model | Type | Parameters | Approach |
|-------|------|-----------|----------|
| 🧠 **LSTM** | Custom deep learning | ~300K | Pixel-wise level-conditioned LSTM, trained from scratch on SST |
| 📊 **N-BEATS** | Basis expansion | ~500K | Interpretable trend/seasonal/generic stacks with Huber Loss |
| 🔮 **Moirai** | Foundation model | 55M (fine-tuned) | Patch-based transformer, pre-trained on 2M+ time series |

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  OISST v2.1 │───▶│  Preprocessing   │───▶│  3 Model Pipelines│───▶│  4-Stage Pipeline   │
│  (16,300 d) │    │  Normalize/Split │    │  LSTM / N-BEATS  │    │  Bias → Spatial →   │
│  60×50 grid │    │  85/5/10 split  │    │  / Moirai        │    │  Scale → Trend Nudge│
└─────────────┘    └──────────────────┘    └──────────────────┘    └──────────┬──────────┘
                                                                             │
                                                                             ▼
┌─────────────┐    ┌──────────────────┐    ┌──────────────────────────────┐
│  📏 Eval    │◀───│  90‑Day Rolling  │◀───│  7/14/30‑Day Forecast /      │
│  RMSE / R   │    │  (Jan–Mar 2026)  │    │  60-day context              │
└──────┬──────┘    └──────────────────┘    └──────────────────────────────┘
       │
       ▼
┌──────────────────────┐
│  🌊 Argo Validation  │
│  (37 in‑situ floats) │
└──────────────────────┘
```

---

## 🏗️ Architecture

### A. Level-Conditioned LSTM (Script 56)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Input: 4 channels (SST anomaly + daily mean + lat + lon) × 60 days │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1: LSTMCell(4 → 64)                                         │
│  Layer 2: LSTMCell(64 → 64)                                        │
│  Level-Conditioning: SST value quartile as auxiliary input         │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Head: Linear(64 → 3)  →  7d / 14d / 30d deltas                   │
│  → Adaptive Drift Correction (±0.20°C cap) → 5-day rolling mean    │
└─────────────────────────────────────────────────────────────────────┘
```

**Post-Processing:** Adaptive offset correction → PostGain amplitude calibration.

### B. N-BEATS Optimized (Script 57)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Input: 90 days SST anomaly (stationary residuals via rolling mean)  │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4 Stacks: Trend(4 blocks) → Seasonal(4 blocks) → Generic ×2       │
│  Each block: FC → FC → FC → FC + residual + backcast/forecast      │
│  Hidden dim: 256, Huber Loss for robust training                   │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Head: Linear(256 → 3)  →  7d / 14d / 30d forecasts               │
└─────────────────────────────────────────────────────────────────────┘
```

**Post-Processing:** 4-Stage Pipeline + train-valid split with early stopping.

### C. Moirai Fine-Tuned (Script 58f) — 🏆 Champion

```
┌─────────────────────────────────────────────────────────────────────┐
│  Input: 365 days seasonal context (SST anomaly)                     │
│         60×50 grid with 4-cardinal gradient injection               │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  moirai-1.0-R-small (55M params) — HuggingFace pretrained           │
│  Patch-based transformer, 20 samples → median forecast              │
│  Fine-tuned on 80% of SST data (13,040 days)                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Ridge Residual Corrector (α=1.0)  →  60×50 spatial field          │
└─────────────────────────────────────────────────────────────────────┘
```

**Post-Processing:** Ridge correction → 4-Stage Pipeline → **best overall RMSE**.

---

## 🥇 Leaderboard

| Rank | Script | Model | 7d RMSE | 14d RMSE | 30d RMSE | 7d R | Argo RMSE |
|------|--------|-------|---------|----------|----------|------|-----------|
| 🥇 | **58f** | **Moirai Fine-Tuned** | **0.108°C** | **0.122°C** | **0.134°C** | **0.938** | **0.298°C** |
| 🥈 | 58f | Moirai Zero-Shot | 0.129°C | 0.148°C | 0.161°C | 0.898 | — |
| 🥉 | **57** | **N-BEATS Optimized** | **0.124°C** | **0.141°C** | **0.158°C** | **0.912** | **0.311°C** |
| 4 | **56** | **LSTM Baseline** | **0.138°C** | **0.151°C** | **0.165°C** | **0.882** | **0.320°C** |

> 💡 **Moirai Fine-Tuned is the champion** — lowest RMSE across all horizons, highest correlation, and best generalization to real Argo measurements. The **4-Stage Pipeline** contributed a cumulative **0.013°C RMSE reduction at 30 days**.

### 📊 RMSE Comparison

```
xychart-beta
    title "7-Day RMSE Comparison (°C)"
    x-axis ["Moirai FT", "N-BEATS", "LSTM"]
    y-axis "RMSE (°C)" 0.10 --> 0.14
    bar [0.108, 0.124, 0.138]
```

---

## 🔧 4-Stage Post-Processing Pipeline

The central contribution — a **novel post-processing framework** applied universally to all three models:

```
Input: Raw model prediction
  ↓
Stage 1: Additive Quartile Bias Correction
         Group by SST anomaly quartile → subtract per-quartile mean bias
         → 0.002°C improvement
  ↓
Stage 2: Per-Pixel Spatial Correction (2D bias map, 60×50)
         Spatial RMSE 0.93 → 0.18°C (80% reduction)
  ↓
Stage 3: Gated Multiplicative Scale (R² > 0.60 gate)
         Only apply if R² ≥ 0.60 (avoids over-correction on noisy pixels)
         → Amplitude restoration
  ↓
Stage 4: Trend-Aware Nudge
         Exponentially decaying drift correction with 7-day half-life
         → Extended-horizon stabilization
  ↓
Output: Final SST forecast

Cumulative impact: 0.013°C RMSE reduction at 30-day horizon
```

| Stage | Description | Impact |
|-------|-------------|--------|
| 1 | Additive Quartile Bias Correction | Thermal-state-dependent bias removal |
| 2 | Per-Pixel Spatial Correction (60×50 bias map) | Spatial RMSE 0.93 → 0.18°C |
| 3 | Gated Multiplicative Scale (R² > 0.60) | Amplitude restoration |
| 4 | Trend-Aware Nudge (exponential decay) | Extended-horizon stabilization |
| **Total** | **Cumulative** | **0.013°C RMSE reduction at 30d** |

---

## 🌊 Argo Float Validation

> **Real ocean data.** 37 independent Argo float profiles (Jan–Feb 2026) matched to nearest grid cells. SST extracted at minimum pressure per profile.

| Model | RMSE | MAE | R |
|-------|------|-----|---|
| 🔮 **Moirai (Ridge-Corrected)** | **0.298°C** | **0.231°C** | **0.93** |
| 📊 N-BEATS | 0.311°C | 0.249°C | 0.91 |
| 🧠 LSTM | 0.320°C | 0.258°C | 0.89 |

```
xychart-beta
    title "Argo Validation — RMSE (°C)"
    x-axis ["Moirai", "N-BEATS", "LSTM"]
    y-axis "RMSE (°C)" 0.28 --> 0.33
    bar [0.298, 0.311, 0.320]
```

> 🔬 **Key finding:** Moirai achieves the **best generalization to real ocean measurements** — 7.2% better than N-BEATS and 4.2% better than LSTM against independent in-situ Argo data.

---

## 📁 Project Structure

```
SatelliteGAN/
│
├── 📄 56_lstm_rolling_7day_v7_v2.py        LSTM baseline (7d RMSE 0.138°C)
├── 📄 57_nbeats_rolling_7day_v2_v2.py       N-BEATS optimized (7d RMSE 0.124°C)
├── 📄 58f_moirai_regional_gradient.py      ★ Moirai Fine-Tuned Champion (0.108°C)
├── 📄 README.md                             This file
│
├── 📁 docs/                                 📚 9 documentation files
│   ├── manuscript-medha.md                  📄 IEEE-style research paper
│   ├── EXECUTIVE_SUMMARY.md                 📄 One-page summary
│   └── ... (MODEL_COMPARISON, QUICK_REFERENCE, etc.)
│
├── 📁 model_comparison/                     📊 Comparison script + plots
│   ├── 59_model_comparison.py               📈 Taylor, density, RMSE plots
│   └── comparison-outputs/                  📊 Generated plots + skill scores
│
├── 📁 validation_data/                      🌊 Argo float validation
│   ├── validate_argo_spatial_models.py      🧪 Run all 3 models on Argo
│   └── argo-validation-outputs/             📉 Validation metrics + plots
│
├── 📁 input_datasets/                       💾 SST data (Git LFS)
│   └── master-harry-appended/
│       ├── master_region_data_new.npy       179 MB
│       └── master_region_anomalies_new.npy  179 MB
│
└── 📁 outputs/                              📦 Model outputs
    ├── lstm-outputs/                        📊 LSTM predictions + plots
    ├── nbeats-outputs/                      📊 N-BEATS predictions + plots
    └── moirai-outputs/                      📊 Moirai predictions + plots
```

---

## ⚡ Quick Start

### 1️⃣ Clone & Setup

```bash
git clone https://github.com/pajonnakuti/satelliteGAN.git
cd satelliteGAN
git checkout MedhaMasanam123

# Pull LFS data (the large .npy files)
git lfs pull

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib scipy uni2ts openpyxl netCDF4
```

### 2️⃣ Run Models (in order)

```bash
# Step 1 — LSTM baseline
python 56_lstm_rolling_7day_v7_v2.py

# Step 2 — N-BEATS optimized
python 57_nbeats_rolling_7day_v2_v2.py

# Step 3 — Moirai Fine-Tuned (🏆 champion)
python 58f_moirai_regional_gradient.py
```

### 3️⃣ Generate Comparison Plots

```bash
cd model_comparison
python 59_model_comparison.py
```

### 4️⃣ Validate with Argo

```bash
cd ../validation_data
python validate_argo_spatial_models.py
```

Outputs are saved to `outputs/<model>/` — including rolling predictions CSV, monthly summaries, spatial maps, and timeseries plots.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0 | LSTM + N-BEATS models, inference |
| `uni2ts` | latest | Salesforce Moirai foundation model |
| `numpy` | ≥ 1.21 | Data manipulation |
| `pandas` | ≥ 1.3 | CSV handling, aggregation |
| `scikit-learn` | ≥ 1.0 | Ridge regression, metrics |
| `matplotlib` | ≥ 3.5 | Visualization |
| `scipy` | ≥ 1.7 | Statistical computations |
| `netCDF4` | latest | Argo reanalysis (NetCDF) |
| `openpyxl` | latest | Argo Excel input |

Moirai weights (55M params) are auto-downloaded from HuggingFace on first run.

---

## 📚 Documentation

| File | Best For |
|------|----------|
| 📄 `docs/manuscript-medha.md` | IEEE-style research paper |
| 📄 `docs/EXECUTIVE_SUMMARY.md` | One-page high-level overview |
| 📄 `docs/FINAL_RESULTS_TABLE.md` | All runs with verified metrics |
| 📄 `docs/MODEL_COMPARISON.md` | Detailed model-by-model analysis |
| 📄 `docs/QUICK_REFERENCE.md` | Key metrics at a glance |
| 📄 `docs/VERIFICATION_PROOFS.md` | How to verify each result |
| 📄 `docs/SCRIPT_INDEX.md` | All scripts listed |
| 📄 `docs/COLAB_ARGO_SPATIAL_VALIDATION_LSTM_NBEATS_MOIRAI.md` | Argo validation Colab guide |
| 📄 `docs/README.md` | Documentation folder index |

---

## 📖 References

1. **Hochreiter & Schmidhuber** — *Long Short-Term Memory*, Neural Computation 1997
2. **Oreshkin et al.** — *N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting*, ICLR 2020
3. **Ansari et al.** — *Moirai: A Time Series Foundation Model for Universal Forecasting*, arXiv:2402.02592, 2024
4. **Garza et al.** — *TimeGPT-1*, arXiv:2310.03589, 2023
5. **Reynolds et al.** — *Daily High-Resolution Blended Analyses for Sea Surface Temperature*, J. Climate 2007

---

**Project completed June 2026**  
🏛️ INCOIS, Hyderabad · 🎓 ICFAI Foundation for Higher Education  
[![Made with Passion](https://img.shields.io/badge/Made%20with-Passion-red?style=flat)]()
[![Data: OISST | Argo](https://img.shields.io/badge/Data-OISST%20%7C%20Argo-blue?style=flat)]()
[![Purpose: Operational Oceanography](https://img.shields.io/badge/Purpose-Operational%20Oceanography-teal?style=flat)]()
