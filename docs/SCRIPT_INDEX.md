# Script Index - All Python Scripts

## Production Scripts (Final/Champion)

| Script | Model | Description | Status |
|--------|-------|-------------|--------|
| `56_lstm_rolling_7day_v7_v2.py` | Pixel-Wise Level-Conditioned LSTM | Rolling 7/14/30-day forecast + 4-Stage Pipeline | ✅ Baseline (0.138°C) |
| `57_nbeats_rolling_7day_v2_v2.py` | N-BEATS | Rolling 7-day with stationary residuals + Huber Loss | ✅ Optimized (0.124°C) |
| `58f_moirai_regional_gradient.py` | Salesforce Moirai | Foundation model + cardinal gradients + Ridge correction | ✅ **Champion (0.108°C)** |
| `model_comparison/59_model_comparison.py` | Comparison & Visualization | Taylor diagram, RMSE bars, error density, skill scores | ✅ Final |
| `validation_data/validate_argo_spatial_models.py` | Argo Validation | Validates all 3 models vs 37 in-situ Argo profiles | ✅ Final |

---

## LSTM Scripts (38-109)

### Phase 1: Foundation (38-46)

| Script | Description | Notes |
|--------|-------------|-------|
| `38_lstm_baseline_point.py` | Single-point baseline LSTM | Point forecast |
| `38_lstm_baseline_colab_gen.py` | Colab generator for baseline | Colab version |
| `39_lstm_research_optimized.py` | Research-optimized LSTM | Optimized config |
| `39_lstm_research_colab_gen.py` | Colab generator for research | Colab version |
| `40_lstm_superclean_colab_gen.py` | Super-clean data generator | Clean data |
| `41_lstm_superclean_fullgrid_colab_gen.py` | Full-grid super-clean Colab | Full grid |
| `46_lstm_flat_global_patches_tuned.py` | Flat global patches tuned | Tuned patches |

### Phase 2: Multi-Horizon (50-54)

| Script | Description | Notes |
|--------|-------------|-------|
| `50_lstm_mimo.py` | MIMO multi-output strategy | Multi-horizon |
| `51_lstm_fullgrid_7day_colab_optimized_FIXED.py` | Full-grid 7-day optimized | Colab optimized |
| `53_lstm_mimo_regional_FIXED.py` | Regional MIMO | Regional focus |
| `54_lstm_FINAL_v7.py` | Final Level-Conditioned LSTM v7 | **Definitive LSTM** |
| `54_lstm_FINAL.py` | Final LSTM original | Final variant |
| `54_lstm_final_fixed.py` | Fixed final LSTM | Bug fix variant |
| `54_lstm_single_horizon_FIXED.py` | Single horizon fixed | Single horizon |
| `54_lstm_7daysperfect.py` | 7-day perfect | 7-day focused |
| `54_lstm_COLAB_OPTIMIZED.py` | Colab optimized | Colab tuned |
| `54b_lstm_14day_FIXED.py` | 14-day fixed | 14-day horizon |
| `54b_lstm_14day_FIXED_fix.py` | 14-day fix iteration | Bug fix |
| `54b_lstm_14day_IMPROVED.py` | 14-day improved | Improved accuracy |
| `54c_lstm_30day_FIXED.py` | 30-day fixed | 30-day horizon |
| `54c_lstm_30day_FIXED_8.py` | 30-day fixed v8 | Optimized 30-day |

### Phase 3: Production (55-66)

| Script | Description | Notes |
|--------|-------------|-------|
| `55_lstm_github_adapted_FIXED.py` | GitHub-adapted LSTM | Adapted version |
| `56_lstm_rolling_7day.py` | Rolling 7-day initial | Initial rolling |
| `56_lstm_rolling_7day_fixed.py` | Rolling 7-day fixed | Bug fix |
| `56_lstm_rolling_7day_v6.py` | Rolling 7-day v6 | Iteration |
| `56_lstm_rolling_7day_v7.py` | Rolling 7-day v7 | Pre-v7_v2 |
| `56_lstm_rolling_7day_v7_v2.py` | Rolling 7-day v7_v2 | **Production** |
| `65_lstm_7day_64gb.py` | 7-day 64GB memory | Memory optimized |
| `66_multi_horizon_forecast_colab_gen.py` | Multi-horizon Colab gen | Generator |

### Phase 4: 7-Day Research (100-109)

| Script | Description | Notes |
|--------|-------------|-------|
| `102_lstm_7day_v1_cpu.py` | 7-day v1 CPU | CPU version |
| `103_lstm_7day_v2_cpu.py` | 7-day v2 CPU | CPU optimized |
| `106_lstm_7day_colab_15e_decade_v1.py` | 7-day Colab 15e decade v1 | Colab |
| `107_lstm_7day_colab_15e_decade_v1.py` | 7-day Colab 15e decade v1 dup | Colab |
| `108_lstm_7day_research_optimized.py` | Research-optimized 7-day | Research |
| `109_lstm_7day_papers_enhanced.py` | Papers-enhanced 7-day | Enhanced |

---

## N-BEATS Scripts (57)

| Script | Description | Status |
|--------|-------------|--------|
| `57_nbeats_rolling_7day_v2.py` | Rolling 7-day N-BEATS v2 | Intermediate |
| `57_nbeats_rolling_7day_v2_v2.py` | Rolling 7-day N-BEATS v2_v2 | **Production** |

---

## Moirai Scripts (58 Series)

| Script | Model | Description | Status |
|--------|-------|-------------|--------|
| `58_moirai_rolling_7day.py` | Moirai | Initial rolling 7-day | Baseline |
| `58_moirai_true_zero_shot.py` | Moirai | True zero-shot inference | Zero-Shot |
| `58b_moirai_zero_shot_optimized.py` | Moirai | Optimized zero-shot (63× speedup) | Zero-Shot |
| `58c_moirai_finetuned.py` | Moirai | Fine-tuning implementation | Fine-Tuned |
| `58d_moirai_few_shot.py` | Moirai | Few-shot with Ridge correction | Few-Shot |
| `58d_moirai_spatial_grid.py` | Moirai | Spatial grid propagation | Spatial |
| `58e_moirai_ultimate_forecast.py` | Moirai | Ultimate fine-tuned forecast | Fine-Tuned |
| **`58f_moirai_regional_gradient.py`** | Moirai | **4-cardinal gradient + Ridge** | **Champion** |
| `58g_moirai_detrended_residuals.py` | Moirai | De-trended residuals + anchoring | Experimental |

---

## Comparison & Visualization Scripts

| Script | Description | Outputs |
|--------|-------------|---------|
| `1_dataset_explorer_full.py` | Dataset exploration | Statistics |
| `2_visualization_full.py` | Full visualization | Plots |
| `3_pattern_comparison_full.py` | Pattern comparison | Comparison plots |
| `4_chronos_sktime_full.py` | Chronos sktime integration | Baseline |
| `9_model_comparison_table.py` | Metric table generation | Tables |
| `59_model_comparison.py` | Multi-model Taylor/violin/RMSE | **Final comparison** |

---

## Utility & Notebook Generation Scripts

| Script | Purpose |
|--------|---------|
| `23_generate_master_colab.py` | Generate master Colab notebook |
| `24_generate_grid_master_colab.py` | Generate grid master Colab |
| `33_merge_daily_npy.py` | Merge daily NPY files |
| `34_analyze_data_content.py` | Analyze data content |
| `35_calculate_anomalies.py` | Calculate SST anomalies |
| `36_standardize_anomalies.py` | Standardize anomalies |
| `generate_colab_notebooks.py` | Generate Colab notebooks |
| `generate_lstm_notebook.py` | Generate LSTM notebook |
| `generate_taylor_diagram.py` | Generate Taylor diagram |
| `compile_html.py` | Compile HTML report |

---

## Argo Validation

| Script | Description | Status |
|--------|-------------|--------|
| `validate_argo_spatial_models.py` | Validates LSTM/N-BEATS/Moirai vs 37 Argo profiles | ✅ Final |

---

## Notebooks (.ipynb)

| Notebook | Model | Content |
|----------|-------|---------|
| `37_lstm_standardized_forecast_colab.ipynb` | LSTM | Standardized forecast |
| `39_lstm_research_colab.ipynb` | LSTM | Research optimized |
| `41_lstm_superclean_fullgrid_colab.ipynb` | LSTM | Super-clean full grid |
| `67_7day_flat_lstm_forecast_colab.ipynb` | LSTM | 7-day flat forecast |
| `68_lstm_multi_horizon_colab.ipynb` | LSTM | Multi-horizon |
| `16_nbeats_grid_forecast_colab.ipynb` | N-BEATS | Grid forecast |
| `17_nbeats_finetuned_eval_colab.ipynb` | N-BEATS | Fine-tuned eval |
| `25_lstm_optimized_colab.ipynb` | LSTM | Optimized |
| `100-107_ref_66_lstm_*.ipynb` | LSTM | Reference runs |

---

## Run Commands

```bash
# LSTM baseline
python 56_lstm_rolling_7day_v7_v2.py

# N-BEATS
python 57_nbeats_rolling_7day_v2_v2.py

# Moirai champion
python 58f_moirai_regional_gradient.py

# Comparison viz
cd model_comparison
python 59_model_comparison.py

# Argo validation
cd ../validation_data
python validate_argo_spatial_models.py
```

---

*June 15, 2026*
