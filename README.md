# Sea Surface Temperature Forecasting — Indian Ocean

**7‑day rolling SST forecasts · ConvLSTM · Amazon Chronos · IBM Granite TSFM**  
*Arabian Sea & Laccadive Sea · 5°N–20°N, 60°E–72°E · 0.25° resolution*

> Research internship project at **INCOIS** (Indian National Centre for Ocean Information Services)  
> Author: **G. Dhanush** — ICFAI Foundation for Higher Education (IFHE)  
> Branch: `dhanushofc`

---

## What This Project Does

We predict **Sea Surface Temperature** 7 days into the future over a 60×48 grid covering the Laccadive Sea and Arabian Sea. Three model families are compared:

| Model | Type | Approach |
|-------|------|----------|
| **ConvLSTM** | Custom deep learning | CNN + LSTM hybrid, trained from scratch on SST data |
| **Amazon Chronos** | Foundation model | Transformer (200M params), zero-shot, frozen weights |
| **IBM Granite TSFM** | Foundation model | MLP-Mixer (71K params), zero-shot, frozen weights |

All models are evaluated on a **90‑day rolling forecast** period (Jan–Mar 2026) against a **Five‑Gate framework**. Predictions are also validated independently against **37 in‑situ Argo float profiles**.

---

## Quick Results

| Rank | Model | RMSE | Slope | Gates | Key Innovation |
|------|-------|------|-------|-------|---------------|
| 1 | **Granite PostGain** ★ | **0.1196°C** | **0.9436** | **5/5** | First foundation model to pass all gates |
| 2 | Chronos PostGain det | 0.1200°C | 0.9488 | 5/5 | Deterministic variant |
| 3 | Chronos PostGain | 0.1205°C | 0.9412 | 5/5 | |
| 4 | ConvLSTM baseline | 0.1417°C | 0.9408 | 5/5 | Robust custom architecture |

### The PostGain Breakthrough

Foundation models systematically **under‑predict temperature swings** (amplitude compression). Our lightweight **PostGain slope correction** fixes this with a single gain multiplier — no retraining needed:

```
y_corrected = g × y_pred      # g = 1.020 (Granite), 1.040 (Chronos)
```

This is the core contribution: **zero‑shot foundation models can match a custom‑trained ConvLSTM** with just one line of post‑processing.

---

## Project Structure

```
📁 dhanush-work/              ← All code, data, and docs
├── 📄 69_convlstm_rolling_7day_fixed.py        ConvLSTM baseline
├── 📄 86_spatial_chronos_only.py               Chronos + PostGain
├── 📄 87_spatial_granite_only.py              ★ Champion model
├── 📄 88_spatial_chronos_only_deterministic.py  Deterministic variant
├── 📄 model_stage2_best.pt                     Trained ConvLSTM weights
│
├── 📁 docs/                  13 documentation files
│   ├── manuscript-dhanush.md          IEEE‑style research paper
│   ├── FINAL_SUBMISSION_REPORT.md     Full project report (17 phases)
│   ├── BOOK_CHAPTER.md                Academic book chapter
│   ├── EXECUTIVE_SUMMARY.md           One‑page overview
│   └── ... (QUICK_REFERENCE, MODEL_COMPARISON, SCRIPT_INDEX, etc.)
│
├── 📁 validation_data/       Argo float spatial validation
│   ├── build_argo_validation_sets.py
│   ├── argo_filter_to_master.py
│   └── validate_argo_spatial_models.py
│
├── 📁 model_comparison/      Multi‑model comparison plots
│   └── model_comparison_kaggle.py
│
└── 📁 input_datasets/        SST data (Git LFS)
    └── master_region_data_new.npy + master_region_anomalies_new.npy
```

---

## Five‑Gate Evaluation Framework

| Gate | Metric | Threshold | Why It Matters |
|------|--------|-----------|---------------|
| 1 | Overall RMSE | < 0.1466°C | Global accuracy |
| 2 | February RMSE | < 0.2093°C | Monsoon transition performance |
| 3 | March RMSE | ≤ 0.1003°C | Pre‑monsoon accuracy |
| 4 | Big error days (≥0.20°C) | ≤ 12 days | Extreme event reliability |
| 5 | Slope (pred vs observed) | 0.94–1.00 | Amplitude fidelity |

---

## Argo Float Validation — Real Ocean Data

| Model | RMSE | Pearson R | Slope |
|-------|------|-----------|-------|
| **ConvLSTM** | **0.324°C** | **0.971** | 0.899 |
| Granite TSFM | 0.394°C | 0.959 | 0.892 |
| Chronos t5‑base | 0.418°C | 0.955 | 0.914 |

ConvLSTM wins on real‑world accuracy — **17.7% better RMSE** than Granite against independent in‑situ measurements (37 Argo profiles, Jan–Feb 2026).

---

## How to Run

```bash
# Clone this branch
git clone --branch dhanushofc https://github.com/pajonnakuti/satelliteGAN.git
cd satelliteGAN/dhanush-work

# Pull LFS data
git lfs pull

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib scipy chronos-forecasting tsfm_public netCDF4 openpyxl

# Run models (in order)
python 69_convlstm_rolling_7day_fixed.py
python 86_spatial_chronos_only.py
python 87_spatial_granite_only.py
python 88_spatial_chronos_only_deterministic.py

# Argo validation
cd validation_data
python build_argo_validation_sets.py
python argo_filter_to_master.py
python validate_argo_spatial_models.py

# Comparison plots
cd ../model_comparison
python model_comparison_kaggle.py
```

Each script saves outputs to `outputs/<script_name>/` — rolling predictions CSV, monthly summary, and plots.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | ConvLSTM model & inference |
| `chronos-forecasting` | Amazon Chronos foundation model |
| `tsfm_public` | IBM Granite foundation model |
| `numpy`, `pandas` | Data processing |
| `scikit‑learn` | Ridge regression, metrics |
| `matplotlib`, `scipy` | Visualization |
| `netCDF4` | Argo reanalysis (NetCDF) |
| `openpyxl` | Argo Excel input |

---

## Key Documentation

| File | Best For |
|------|----------|
| `dhanush-work/docs/manuscript-dhanush.md` | Ready‑to‑submit IEEE research paper |
| `dhanush-work/docs/FINAL_SUBMISSION_REPORT.md` | Full 17‑phase project report (130 KB) |
| `dhanush-work/docs/EXECUTIVE_SUMMARY.md` | One‑page high‑level summary |
| `dhanush-work/docs/FINAL_RESULTS_TABLE.md` | All 25+ runs with verified metrics |
| `dhanush-work/docs/QUICK_REFERENCE.md` | Key numbers at a glance |

---

## References

- Shi et al., *Convolutional LSTM Network for Precipitation Nowcasting*, NeurIPS 2015
- Ansari et al., *Chronos: Learning the Language of Time Series*, arXiv:2403.07815, 2024
- IBM Research, *Granite Time‑Series Foundation Model*
- Hu et al., *LoRA: Low‑Rank Adaptation of Large Language Models*, ICLR 2022
- Reynolds et al., *Daily High‑Resolution Blended Analyses for SST*, J. Climate 2007

---

*Project completed June 2026 · INCOIS, Hyderabad · ICFAI Foundation for Higher Education*
