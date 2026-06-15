INCOIS SST Forecasting - Final Result Codes
=============================================

Author:    Ginkala Dhanush
Institution: FST [IFHE, ICFAI], CSE Batch 2026
Date:      19 May 2026
Advisor:   INCOIS (Indian National Centre for Ocean Information Services)
Project:   Sea Surface Temperature (SST) forecasting over the Arabian Sea
           (5.125-19.875 deg N, 60.125-71.875 deg E) using ConvLSTM,
           Amazon Chronos, and IBM Granite TSFM.


CONTENTS OF THIS FOLDER
-----------------------

ROOT FILES (production scripts + ConvLSTM checkpoint):

1. 69_convlstm_rolling_7day_fixed.py
   ConvLSTM production baseline.
   Rolling 7-day forecast, 90-day evaluation (Jan-Mar 2026).
   RMSE = 0.1417 deg C, Gates = 5/5.
   Status: Production Baseline.

2. 86_spatial_chronos_only.py
   Chronos-only zero-shot + Post-Hoc correction (PostGain).
   Single-model spatial pipeline.
   RMSE = 0.1205 deg C, Slope = 0.9412, Gates = 5/5.
   Status: Highest slope.

3. 87_spatial_granite_only.py                  [NEW SINGLE-MODEL CHAMPION]
   Granite-only zero-shot + Post-Hoc correction (PostGain).
   Single-model spatial pipeline.
   RMSE = 0.1196 deg C, Slope = 0.9436, Gates = 5/5.
   Status: New single-model champion. PostGain = 1.020.

4. 88_spatial_chronos_only_deterministic.py
   Chronos-only Post-Hoc correction (PostGain) - DETERMINISTIC variant.
   Same architecture as 86 but with NUM_SAMPLES=1, TEMPERATURE=0.0, TOP_P=1.0.
   RMSE = 0.1200 deg C, Slope = 0.9488, Gates = 5/5.
   Status: Deterministic reproducibility. PostGain = 1.040.

5. model_stage2_best.pt
   ConvLSTM Stage 2 fine-tuned checkpoint (PyTorch state dict).
   Required by Script 69 (and validate_argo). Size: 1.89 MB.
   Original location: 63_convlstm_v2finetune/66_convlstm_7day_stage2_final/


SUBFOLDER validation_data/ (Argo float validation pipeline):

  Scripts:
  - build_argo_validation_sets.py
    Builds aligned Argo/master/reanalysis CSVs from raw inputs.
    Reads Argo_validsation_TSFM.xlsx (XLSX) + Argo_validsation_TSFM_reanalysis.nc (NetCDF).
    Outputs the 3 aligned CSVs below.

  - argo_filter_to_master.py
    Maps Argo validation points to the master 60x48 grid using 0.25 deg resolution
    and START_DATE = 1981-09-01. Reads the XLSX, writes the filtered CSV.

  - validate_argo_spatial_models.py
    Runs all three models (ConvLSTM, Chronos, Granite) against 37 Argo float
    profiles. ConvLSTM loads the model_stage2_best.pt checkpoint. Foundation
    models (Chronos, Granite) use beta-map spatial propagation. Outputs metrics
    and per-point predictions.

  Inputs (in validation_data/):
  - argo_validation_tsfm.csv             Argo validation points (37)
  - master_appended_tsfm.csv             Master grid SST at Argo locations
  - reanalysis_tsfm.csv                  Reanalysis SST at Argo locations
  - Argo_validsation_TSFM_filtered_to_master.csv   Filtered Argo + master mapping
  - Argo_validsation_TSFM.xlsx           Raw Argo float profiles (with QC flags)
  - Argo_validsation_TSFM_reanalysis.nc  Reanalysis SST field (NetCDF, 2.23 MB)


SUBFOLDER model_comparison/ (multi-model comparison pipeline):

  Script:
  - model_comparison_kaggle.py
    Produces Taylor diagram, error density curves, timeseries overlay, RMSE
    comparison, monthly RMSE bars, error violin, and skill table from the
    3 rolling_predictions CSVs below.

  Inputs (in model_comparison/):
  - rolling_predictions_code_69.csv   ConvLSTM 69 rolling forecasts
  - rolling_predictions_code_86.csv   Chronos 86 rolling forecasts
  - rolling_predictions_code_87.csv   Granite 87 rolling forecasts

  Reads ConvLSTM/Chronos/Granite rolling predictions and produces:
    taylor_diagram.png, error_density_curves.png, timeseries_overlay.png,
    overall_rmse_comparison.png, monthly_rmse_bar.png, error_violin.png,
    skill_table.png, skill_scores.csv


SUBFOLDER docs/ (important documentation):

  Main reports (the formal write-ups):
  - manuscript-dhanush.md                  18 KB    IEEE-style research manuscript

  Supporting docs (summaries and references):
  - EXECUTIVE_SUMMARY.md                    7 KB    One-page high-level summary
  - FINAL_RESULTS_TABLE.md                 11 KB    All verified results with proofs
  - README.md                               4 KB    Docs folder entry point
  - SCRIPT_INDEX.md                         8 KB    What each of the 54 scripts does
  - MODEL_COMPARISON.md                     8 KB    Detailed model comparison
  - QUICK_REFERENCE.md                      4 KB    Key metrics at a glance
  - VERIFICATION_PROOFS.md                  9 KB    Source file references for results
  - KAGGLE_ARGO_SPATIAL_VALIDATION.md       2 KB    Argo validation Kaggle guide


HOW TO RUN
----------

The scripts expect to be run in a Kaggle notebook environment with the
following data mounted at /kaggle/input/datasets/rayofc/:

  /kaggle/input/datasets/rayofc/
    master-harry-appended/
      master_region_data_new.npy           (shape: 16290, 60, 48)
      master_region_anomalies_new.npy      (shape: 16290, 60, 48)
    checkpoints-66/
      model_stage2_best.pt                 (ConvLSTM checkpoint - already in this folder)

For local runs, the scripts fall back to a relative path:
  master-harry-appended/master_region_data_new.npy
  master-harry-appended/master_region_anomalies_new.npy

For Script 69 (ConvLSTM) ONLY:
  Place model_stage2_best.pt at:
    63_convlstm_v2finetune/66_convlstm_7day_stage2_final/model_stage2_best.pt
  OR edit the CKPT_FILE path in the script to point to this folder.

For Scripts 86, 87, 88 (Foundation Models):
  Chronos/Granite weights are auto-downloaded from HuggingFace on first run.
  No additional local files required for the foundation model weights.


EVALUATION FRAMEWORK (FIVE GATE)
--------------------------------

  Gate 1: Overall RMSE      < 0.1466 deg C
  Gate 2: February RMSE     < 0.2093 deg C
  Gate 3: March RMSE        <= 0.1003 deg C
  Gate 4: Big Error Days    <= 12  (days with |error| >= 0.20 deg C)
  Gate 5: Slope             [0.94, 1.00]  (amplitude response fidelity)

All four production scripts pass all five gates (5/5).


ARGO FLOAT VALIDATION (independent ground truth)
-------------------------------------------------

  Independent validation of ConvLSTM, Chronos, and Granite against 37 in-situ
  Argo float profiles (Jan-Feb 2026) using the validation_data/ pipeline.

    Model      RMSE      MAE     R       R^2     Slope
    --------   --------  ------  ------  ------  ------
    ConvLSTM   0.324 C   0.262   0.971   0.943   0.899
    Granite    0.394 C   0.301   0.959   0.920   0.892
    Chronos    0.418 C   0.322   0.955   0.911   0.914

  Key finding: ConvLSTM achieves the lowest RMSE (0.324 C), outperforming
  Granite by 17.7% and Chronos by 22.5% against independent Argo measurements.
  This confirms ConvLSTM's superior generalizability to in-situ data even
  though the PostGain-corrected foundation models win on the gridded RMSE.


RESULT SUMMARY
--------------

  Rank  Script  Model                 RMSE       Slope   Gates
  ----  ------  --------------------  --------   ------  -----
  1     87      Granite PostGain      0.1196     0.9436  5/5  [CHAMPION]
  2     88      Chronos PostGain det  0.1200     0.9488  5/5
  3     86      Chronos PostGain      0.1205     0.9412  5/5
  4     69      ConvLSTM              0.1417     0.9408  5/5  [Baseline]


POSTGAIN SLOPE CORRECTION
-------------------------

Scripts 86, 87, 88 implement a novel post-hoc technique to resolve the
systematic amplitude compression observed in foundation model forecasts.
The technique fits a multiplicative gain g on the validation set such that
the linear regression slope of (g * y_pred) vs y_true is >= 0.94.
Fitted gains: 1.020 (Granite 87) and 1.040 (Chronos 86, 88).


DEPENDENCIES
------------

  Python 3.10+
  torch >= 2.0
  numpy
  pandas
  scikit-learn
  matplotlib
  scipy
  chronos (pip install chronos-forecasting)        # for Chronos
  tsfm_public (pip install tsfm_public)            # for Granite TSFM
  netCDF4 (pip install netCDF4)                    # for Argo reanalysis NC
  openpyxl (pip install openpyxl)                  # for Argo XLSX


FURTHER DOCUMENTATION
---------------------

  The 8 docs in this folder's docs/ subfolder cover the key project
  documentation. The full gitpush-final/ project (with all 23 docs, 90
  result files, 54 scripts) is available locally in the project archive.


NOTES
-----

  - Ensemble pipelines (Scripts 84, 85) are intentionally EXCLUDED.
    They are documented as a SECONDARY investigation in the main report.
  - Few-shot / LoRA / Zero-shot baselines (Scripts 78-83) are
    intentionally EXCLUDED. They are precursors to the PostGain scripts
    and their metrics are listed as "Historical Results" in the
    FINAL_RESULTS_TABLE.
  - The Argo float validation pipeline is INCLUDED in validation_data/
    as the secondary validation that confirms ConvLSTM's superior
    generalizability to in-situ data (RMSE 0.324 C, R = 0.971).
