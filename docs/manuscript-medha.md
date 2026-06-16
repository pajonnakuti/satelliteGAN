# IEEE-Style Research Manuscript

## Document Identification
* **Title**: Sea Surface Temperature Forecasting in the Arabian Sea Using Deep Learning Ensembles with a 4-Stage Causal Post-Processing Pipeline
* **Authors**: M. Medha
* **Affiliations**: Department of Computer Science and Engineering, ICFAI Foundation for Higher Education (IFHE)
* **Context**: Research Internship, Indian National Centre for Ocean Information Services (INCOIS) [2]

---

### Abstract

Accurate sea surface temperature forecasts are critical for operational oceanography, monsoon prediction, and marine ecosystem management in the Arabian Sea region. This study presents a comprehensive comparative evaluation of three distinct forecasting paradigms — a Pixel-Wise Level-Conditioned Long Short-Term Memory network, a Neural Basis Expansion Analysis for Interpretable Time Series (N-BEATS) model, and the Moirai patch-based time-series foundation transformer — applied to multi-horizon SST anomaly prediction over a 60 by 50 spatial grid (3,000 independent pixel locations) spanning approximately 44.7 years of daily OISST v2.1 observations. A 4-stage causal post-processing pipeline was developed to address systematic forecast biases: quartile-based additive bias correction, per-pixel spatial correction maps, gated multiplicative scale correction activated only in high-predictability regimes (R-squared greater than 0.60), and a trend-aware nudge mechanism for extended-horizon stabilization. The Moirai fine-tuned model achieved the best overall performance with RMSE values of 0.108 degrees Celsius at 7 days, 0.122 degrees Celsius at 14 days, and 0.134 degrees Celsius at 30 days. The N-BEATS model achieved competitive results (RMSE 0.124 to 0.158 degrees Celsius) through interpretable basis-function decomposition, while the level-conditioned LSTM established a solid baseline (RMSE 0.138 to 0.165 degrees Celsius). The 4-stage post-processing pipeline contributed 0.013 degrees Celsius RMSE reduction at the 30-day horizon, confirming the critical importance of validation-calibrated correction in operational forecasting systems. Independent validation against 37 in-situ Argo float profiles confirmed the ensemble hierarchy: Moirai achieved the strongest generalization (RMSE 0.298 degrees Celsius), followed by N-BEATS (RMSE 0.311 degrees Celsius) and LSTM (RMSE 0.320 degrees Celsius).

**Keywords**—Sea Surface Temperature, Arabian Sea, LSTM, N-BEATS, Moirai, Foundation Models, Post-Processing Pipeline, Deep Learning, Time Series Forecasting, Ensemble Methods.

---

## I. Introduction

Sea surface temperature serves as a fundamental parameter in oceanographic science, directly modulating air-sea heat exchange, monsoon dynamics, marine ecosystem productivity, and tropical cyclone intensity. The Arabian Sea, bounded by latitudes 5 degrees North to 20 degrees North and longitudes 60 degrees East to 72 degrees East, is one of the most dynamically complex ocean basins on Earth. Its oceanographic character is dominated by the biannual reversal of the monsoon wind system, coastal upwelling along the Somali and Omani coasts, and a rapid warming trend exceeding 0.15 degrees Celsius per decade [1]. SST variability in this region drives the Southwest Monsoon that sustains the agricultural economy of over one billion people. Operational forecasting centers such as the Indian National Centre for Ocean Information Services [2] require reliable SST predictions for fisheries management, navigation safety, and climate monitoring.

Traditional approaches to SST forecasting have followed two primary paths. Numerical Weather Prediction models such as the Regional Ocean Modeling System and the Hybrid Coordinate Ocean Model solve coupled partial differential equations governing ocean dynamics, achieving physically consistent forecasts at substantial computational cost [3], [4]. Statistical methods including autoregressive models and empirical orthogonal function analysis offer computational efficiency but assume linear dynamics that do not hold for tropical SST variability. The emergence of deep learning has transformed time-series forecasting. Long Short-Term Memory networks demonstrated superior temporal modeling capability over statistical methods [5]. N-BEATS introduced a deep feedforward architecture with interpretable basis-function decomposition for time-series forecasting [6]. More recently, transformer architectures [7] and foundation models pre-trained on diverse multi-domain corpora — such as Salesforce Moirai [8] — have demonstrated remarkable zero-shot and fine-tuned forecasting capabilities across domains.

Despite their promise, deep learning models applied to SST forecasting exhibit a critical limitation: systematic forecast drift beyond 12 to 14 days in the absence of atmospheric forcing fields [9]. All architectures consistently exhibit day-1 initialization bias, amplitude smoothing of extreme anomalies, and horizon-dependent error growth that compounds with forecast horizon. Previous approaches have addressed these issues through architectural modifications and ad-hoc corrections, but no unified post-processing framework has been established for the Arabian Sea region.

This work contributes four novel advances. First, the Level-Conditioned LSTM architecture introduces a delta-prediction formulation anchored to the last observed anomaly state, with the initial anomaly level concatenated directly to the LSTM hidden state, effectively eliminating day-1 structural bias. Second, the application of N-BEATS to SST data with stationary residual preprocessing demonstrates that interpretable basis-function decomposition achieves competitive accuracy for oceanographic forecasting. Third, the Moirai foundation transformer is adapted for regional SST forecasting through cardinal gradient injection (North, South, East, and West spatial context), enabling a 1D transformer to preserve geographic spatial structure. Fourth, and most critically, a 4-stage causal post-processing pipeline is developed that corrects thermal-state-dependent bias, spatially heterogeneous bias, amplitude smoothing, and extended-horizon drift through a sequential validation-calibrated correction framework.

---

## II. Data and Methodology

The methodology is structured around a single unifying narrative: starting from raw observational data, three modeling paradigms were developed and refined through a structured evaluation pipeline, and the best-performing configurations were independently validated against in-situ Argo float measurements.

### A. Dataset Specification

The Optimum Interpolation Sea Surface Temperature (OISST) version 2.1 dataset, produced by NOAA's National Centers for Environmental Information [10], forms the observational foundation of this study. The spatial domain spans latitudes 5 degrees North to 20 degrees North and longitudes 60 degrees East to 72 degrees East at 0.25 degree resolution, yielding a 60 by 50 pixel grid (3,000 independent spatial locations). The temporal domain covers approximately 16,300 daily observations from September 1, 1981 to April 30, 2026, encompassing approximately 44.7 years of data.

The data was partitioned into a training set comprising September 1, 1981 to December 31, 2023 (approximately 95.6 percent of data), a validation set comprising January 1, 2024 to December 31, 2025 (approximately 730 days), and a test set comprising the 90-day evaluation period from January 1, 2026 to March 31, 2026. All normalization statistics were computed exclusively on the training partition to prevent data leakage.

Long-term daily means were computed from the full training period and subtracted from absolute SST to produce anomaly fields. For N-BEATS, a 30-day rolling mean was further subtracted to produce stationary residuals, which significantly improved training stability. For Moirai, the anomaly sequence was provided with a 365-day context window and cardinal gradient covariates computed as North, South, East, and West differences at 2-degree offsets.

To handle the high computational load of the 60 by 50 grid, a pixel-wise tensor reshaping operation was performed: the input tensor of dimensions (Batch, Sequence, Channels, Height, Width) was reshaped to (Batch multiplied by Height multiplied by Width, Sequence, Channels), expanding the effective dataset by a factor of 3,000. This approach treats each grid location as an independent time-series, preserving localized features and enabling per-pixel post-processing corrections.

### B. Architectural Framework and Custom Configurations

Three distinct modeling architectures were evaluated. The first architecture is a purpose-built Pixel-Wise Level-Conditioned LSTM comprising a 2-layer stacked LSTM with hidden size selected through ablation studies, followed by a linear prediction head. The model processes 90-day input sequences over the full 60 by 50 spatial grid. The key innovation is level-conditioning: rather than predicting absolute SST, the model predicts delta changes (Delta y at time t plus h) conditioned on the last observed anomaly value. The initial anomaly level is concatenated directly to the LSTM hidden state before the final fully-connected layer, ensuring the model is conditioned on the starting thermal state. A One-Sided Variance Penalty is added to the training objective — defined as ReLU(Target Variance minus Predicted Variance) — which penalizes the model only when predicted variance falls below target variance, preserving amplitude in extreme anomalies without forcing artificial noise. Training was conducted using the AdamW optimizer [11] with weight decay of 1e-2 and OneCycleLR learning rate scheduling [12] on an NVIDIA T4 GPU for 15 to 25 epochs with a batch size of 8.

The second architecture is N-BEATS, specifically a 4-stack configuration with 4 blocks per stack. The stack configuration comprises a Trend stack (polynomial basis of degree 3), a Seasonal stack (Fourier basis with 6 harmonics), and two Generic stacks (identity basis). Each N-BEATS block implements a doubly-residual topology: a backcast branch reconstructs the input, and a forecast branch predicts the future. The residual connections between blocks enable the model to learn progressively refined representations of the time-series components. N-BEATS operates on stationary residuals (anomaly minus 30-day rolling mean) with shared weights across all 3,000 pixels. Huber Loss [13] was used for training stability, and a horizon-weighted loss (power of 2.0) ensured accurate short-term predictions.

The third architecture is Salesforce Moirai, specifically the moirai-1.0-R-small variant with 55 million parameters. Moirai is a patch-based time-series foundation transformer pre-trained on over 2 million diverse time-series datasets. This study deploys Moirai in both zero-shot and fine-tuned configurations with a 365-day seasonal context window. To preserve spatial structure within the 1D transformer framework, cardinal gradient injection was developed: for each target pixel, four gradient values were computed as the difference between the target and its North, South, East, and West neighbors at 2-degree offset, and these were fed as additional past-feature-dynamic-real covariates. A Ridge regression residual correction was fitted on the 150-day validation window using 15 spatial and temporal features, mapping systematic residual errors per forecast horizon. The zero-shot configuration faced a critical speed challenge — initial execution times of approximately 506 seconds per window were reduced to approximately 8 seconds (a 63-fold improvement) by reducing context length from 1,095 to 365 days, reducing sample paths from 200 to 50, and increasing batch size from 8 to 64.

### C. The 4-Stage Causal Post-Processing Pipeline

A rigorous post-processing pipeline was established as the central contribution of this work. The pipeline operates causally (using only information available at forecast time) and is calibrated entirely on the held-out validation partition. It addresses four distinct error modes through four sequential stages.

Stage 1 is Additive Quartile Bias Correction. Forecast errors are strongly correlated with the initial thermal state (SST anomaly quartile). A global bias correction would over-correct in some thermal regimes and under-correct in others. This stage groups forecasts by the quartile of the initial SST anomaly, computes the median bias within each quartile on the validation set, and applies the corresponding correction per forecast.

Stage 2 is Per-Pixel Spatial Correction. Certain geographic regions — particularly coastal upwelling zones along the Somali and Omani coasts — exhibit persistent local biases that are not captured by global corrections. This stage computes a 2D bias map over the validation set (60 by 50 pixels) and subtracts it from the corresponding pixel predictions, reducing spatial RMSE from 0.93 degrees Celsius to 0.18 degrees Celsius.

Stage 3 is Gated Multiplicative Scale Correction. Deep learning models systematically underestimate the amplitude of extreme anomalies. This stage applies a multiplicative scale factor to restore variance, but only when the model exhibits high predictability (R-squared greater than 0.60). This gating mechanism ensures that confident forecasts are preserved while only low-confidence predictions are corrected.

Stage 4 is Trend-Aware Nudge. At forecast horizons beyond 14 days, predictions exhibit artificial upward or downward drift. This stage fits a linear slope to a trailing window of anchor days and applies an exponentially decaying trend correction to stabilize late-horizon forecasts toward the climatological mean.

The pipeline is applied sequentially: additive corrections precede multiplicative corrections, and local corrections precede global corrections. The cumulative impact of the 4-stage pipeline was a 0.013 degrees Celsius RMSE reduction at the 30-day horizon, confirming the critical importance of validation-calibrated correction in operational forecasting systems.

---

## III. Results and Discussion

### A. Post-Game Evaluation Analysis

The evaluation was conducted over a 90-day rolling forecast period from January 1, 2026 to March 31, 2026, comprising 90 consecutive forecast days. For each window, the model ingests historical SST anomaly data and produces 7-day, 14-day, and 30-day ahead forecasts. The post-game evaluation computes per-horizon and aggregate metrics comparing predicted values against the observed SST. All experiments were conducted on Google Colab T4 GPU (16 GB VRAM) with Python 3.10, PyTorch 2.x, NumPy 1.26, and SciPy 1.11. Random seeds were fixed (seed equals 42) across all experiments to ensure reproducibility.

The 4-stage post-processing pipeline was identified as the critical intervention that enabled all three models to achieve operationally viable performance at extended horizons. The quartile-based correction was most impactful for extreme thermal states (cold-dip events in February and warm anomalies in March). The per-pixel spatial correction was particularly significant in coastal upwelling zones, where raw model errors exceeded 0.5 degrees Celsius.

Independent Argo float validation [14] was conducted to assess generalizability beyond the OISST-derived evaluation product. Thirty-seven Argo float profiles spanning January to February 2026 were matched to the nearest grid cell in the 60 by 50 domain. SST was extracted at minimum pressure per profile with rigorous quality control filtering.

### B. Metric Performance Comparison

The following table presents the core metric performance for all evaluated configurations:

| Model | Horizon | RMSE (degrees C) | MAE (degrees C) | Pearson R |
| :--- | :--- | :--- | :--- | :--- |
| LSTM Baseline | 7-Day | 0.138 | 0.102 | 0.882 |
| LSTM Baseline | 14-Day | 0.151 | 0.115 | 0.814 |
| LSTM Baseline | 30-Day | 0.165 | 0.128 | 0.765 |
| N-BEATS Optimized | 7-Day | 0.124 | 0.091 | 0.912 |
| N-BEATS Optimized | 14-Day | 0.141 | 0.108 | 0.849 |
| N-BEATS Optimized | 30-Day | 0.158 | 0.120 | 0.803 |
| Moirai Zero-Shot | 7-Day | 0.129 | 0.095 | 0.898 |
| Moirai Zero-Shot | 14-Day | 0.148 | 0.112 | 0.831 |
| Moirai Zero-Shot | 30-Day | 0.161 | 0.124 | 0.794 |
| **Moirai Fine-Tuned** | **7-Day** | **0.108** | **0.080** | **0.938** |
| **Moirai Fine-Tuned** | **14-Day** | **0.122** | **0.091** | **0.875** |
| **Moirai Fine-Tuned** | **30-Day** | **0.134** | **0.103** | **0.842** |

The Moirai fine-tuned configuration occupies the top rank across all horizons and all metrics. It achieves a 21.7 percent RMSE improvement over the LSTM baseline at the 7-day horizon and an 18.8 percent improvement at the 30-day horizon. N-BEATS occupies the second rank, with RMSE values 8.7 to 10.1 percent lower than the LSTM baseline. The LSTM baseline, while competitive at the 7-day horizon, exhibits larger performance degradation at extended horizons due to cumulative error accumulation.

The independent Argo spatial validation results are presented below:

| Model | RMSE (degrees C) | R (Correlation) |
| :--- | :--- | :--- |
| **Moirai (Ridge-Corrected)** | **0.298** | **0.93** |
| N-BEATS | 0.311 | 0.91 |
| LSTM | 0.320 | 0.89 |

Moirai achieves the lowest RMSE (0.298 degrees Celsius) among all three models when validated against in-situ Argo measurements, outperforming N-BEATS by 4.2 percent and LSTM by 6.9 percent. This confirms that the ensemble hierarchy observed on the OISST evaluation product generalizes to independent observational data. All three models exhibit a predictable decline in accuracy when moving from satellite-derived to in-situ observations, consistent with the expected increase in observational uncertainty and the point-vs-pixel representativeness mismatch.

The monthly breakdown reveals important temporal patterns. February 2026 presented the greatest forecasting challenge across all models due to a pronounced cold-dip event associated with winter monsoon strengthening, with RMSE values of 0.215 degrees Celsius (LSTM), 0.213 degrees Celsius (N-BEATS), and 0.193 degrees Celsius (Moirai zero-shot) — reflecting the irreducible atmospheric forcing component common to all ocean-only forecasting approaches. March showed the lowest errors across all models (LSTM: 0.102, N-BEATS: 0.102, Moirai zero-shot: 0.106 degrees Celsius), consistent with the seasonal reduction in synoptic-scale atmospheric variability during the spring inter-monsoon period.

### C. Figure Asset and Image Formatting Guidelines

All figures in this manuscript should be rendered at a minimum resolution of 300 DPI and stored in a dedicated figures directory. The following caption structure is recommended:

**Fig. 1.** Level-conditioned LSTM architecture diagram showing the pixel-wise reshape operation, LSTM cell structure with level-conditioning, and the delta-prediction output head.

**Fig. 2.** RMSE horizon comparison bar chart for LSTM, N-BEATS, and Moirai (fine-tuned) across 7-day, 14-day, and 30-day forecast horizons.

**Fig. 3.** Ninety-day time series overlay comparing predicted SST against ground truth for Moirai fine-tuned at the target location.

**Fig. 4.** Ninety-day time series overlay for N-BEATS optimized configuration.

**Fig. 5.** Ninety-day time series overlay for LSTM baseline configuration.

**Fig. 6.** Multi-Model Skill Scores bar chart comparing the three architectures.

**Fig. 7.** Ensemble Taylor Diagram with publication-quality formatting, showing RMSE target zone and model positions.

**Fig. 8.** Monthly RMSE bar comparison for January, February, and March 2026 across all three models.

**Fig. 9.** Argo Validation Taylor Diagram showing Moirai, N-BEATS, and LSTM performance against in-situ observations.

**Fig. 10.** Argo observed versus predicted SST overlay across 37 validation profiles.

---

## IV. Conclusion and Future Work

This study demonstrates that a 4-stage causal post-processing pipeline, when applied to deep learning SST forecasts, enables operational-grade prediction accuracy across multiple horizons in the Arabian Sea region. The Moirai fine-tuned configuration achieved the best overall performance with an RMSE of 0.108 degrees Celsius at the 7-day horizon, outperforming N-BEATS (0.124 degrees Celsius) and the level-conditioned LSTM (0.138 degrees Celsius). The 4-stage pipeline — incorporating quartile-based bias correction, per-pixel spatial correction, gated multiplicative scale correction, and trend-aware nudge — contributed a 0.013 degrees Celsius RMSE reduction at the 30-day horizon, establishing post-processing as an essential component of operational deep learning forecasting systems.

The independent Argo validation confirms the ensemble hierarchy: Moirai (Ridge-corrected) achieves the strongest generalization to in-situ measurements with an RMSE of 0.298 degrees Celsius, followed by N-BEATS (0.311 degrees Celsius) and LSTM (0.320 degrees Celsius). This finding establishes Moirai as the most reliable model for operational use when both satellite and in-situ accuracy are considered.

Several directions for future research emerge from this work. First, the post-processing pipeline could be extended to a dynamic formulation where correction parameters adapt to seasonal and interannual variability. Second, atmospheric forcing fields such as ERA5 wind stress and heat flux anomalies could be incorporated as additional covariates to extend the predictability horizon beyond the current 12 to 14-day limit. Third, the evaluation framework could be expanded to include uncertainty quantification through Monte Carlo dropout or Bayesian approaches. Fourth, real-time streaming inference pipelines could be developed for operational deployment at INCOIS. Finally, the 4-stage pipeline methodology may generalize to other geophysical forecasting tasks where deep learning models exhibit systematic drift and bias.

---

## References

[1] M. K. Roxy, K. Ritika, P. Terray, and S. Masson, "The curious case of Indian Ocean warming," *Journal of Climate*, vol. 27, no. 22, pp. 8501-8529, 2014.

[2] Indian National Centre for Ocean Information Services, "Operational oceanographic services," INCOIS, Ministry of Earth Sciences, Government of India, 2025.

[3] A. F. Shchepetkin and J. C. McWilliams, "The regional ocean modeling system (ROMS): A split-explicit, free-surface, topography-following-coordinate ocean model," *Ocean Modelling*, vol. 9, no. 4, pp. 347-404, 2005.

[4] E. P. Chassignet, H. E. Hurlburt, O. M. Smedstad, G. R. Halliwell, P. J. Hogan, A. J. Wallcraft, and R. Bleck, "The HYCOM (Hybrid Coordinate Ocean Model) data assimilative system," *Journal of Marine Systems*, vol. 65, no. 1-4, pp. 445-467, 2003.

[5] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, 1997.

[6] B. N. Oreshkin, D. Carpov, N. Chapados, and Y. Bengio, "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting," in *Proc. International Conference on Learning Representations (ICLR)*, 2019.

[7] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 5998-6008, 2017.

[8] G. Woo, C. Liu, D. Sahoo, A. Kumar, and S. Hoi, "Unified training of universal time series forecasting transformers," *arXiv preprint arXiv:2402.02592*, 2024.

[9] A. G. Barnston, M. Chelliah, and C. F. Ropelewski, "Verification of operational long-lead climate forecasts at the Climate Prediction Center," *Weather and Forecasting*, vol. 14, no. 4, pp. 491-508, 1999.

[10] R. W. Reynolds, T. M. Smith, C. Liu, D. B. Chelton, K. S. Casey, and S. D. Woodruff, "Daily high-resolution-blended analyses for sea surface temperature," *Journal of Climate*, vol. 20, no. 22, pp. 5473-5496, 2007.

[11] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. International Conference on Learning Representations (ICLR)*, 2019.

[12] L. N. Smith, "A disciplined approach to neural network hyper-parameters: Part 1 — Learning rate, batch size, momentum, and weight decay," *arXiv preprint arXiv:1803.09820*, 2018.

[13] P. J. Huber, "Robust estimation of a location parameter," *Annals of Mathematical Statistics*, vol. 35, no. 1, pp. 73-101, 1964.

[14] D. Roemmich, S. Riser, R. Davis, and L. Talley, "The Argo program: Observing the global ocean with profiling floats," *Oceanography*, vol. 22, no. 2, pp. 34-43, 2009.
