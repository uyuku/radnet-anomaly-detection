# Phase 6 Report: Weather-Aware Anomaly Detection in EPA RadNet — Final Report

> **Historical report:** These results predate feature schema 2 (2026-10-18). They have not been revalidated after correcting missing-weather features. The status and numerical claims below describe the earlier run. See [the maintenance report](MAINTENANCE_2026-10-18.md).

**Date**: 2026-10-17
**Project**: Weather-Aware Anomaly Detection in RadNet (EPA RadNet + NOAA NCEI ISD)
**Status**: Final verified report. Every number below traces directly to the regenerated CSV tables in `data/processed/` and is verified by the automated test suite in [`tests/test_radnet_pipeline.py`](../tests/test_radnet_pipeline.py).

---

## Abstract

EPA RadNet stationary gamma air monitors collect ambient particulates on glass-fiber filters and count them with NaI(Tl) detectors in eight gamma energy channels. Rain washes natural radon progeny ($^{214}\text{Pb}$, $^{214}\text{Bi}$) out of the atmosphere onto the filter, inflating the gross count rate and triggering threshold alarms that operators cannot distinguish from anthropogenic releases on gross counts alone. On 2017–2025 data from five U.S. stations (334,567 valid RadNet hours; 324,550 synchronous radiation-weather hours), a conventional rolling 7-day + $3\sigma_{\text{dry}}$ threshold on gross count rate fires 6.59–62.91 alarm episodes per station-year depending on climate, and in humid continental stations 69.5%–88.5% of those episodes coincide with rainfall within 3 hours. We built a physically parameterized synthetic plume catalog (450 filter-synchronized injections of $^{137}\text{Cs}$, $^{131}\text{I}$, $^{60}\text{Co}$, $^{134}\text{Cs}$ mixtures, 250 train / 200 test, calendar-disjoint) and trained gradient-boosted classifiers that fuse gross count rate, multi-channel spectral ratios, and synchronous NOAA surface weather. Evaluated across 20 random seeds under a strictly causal validation-frozen protocol (2017–2020 train, 2021–2022 validation threshold selection, 2023–2025 out-of-sample test across 106,628 observed hours = 12.16 station-years), the weather-fused spectral model (Tier 3) achieves **92.65 ± 1.37%** plume detection with **4.32 ± 1.29** clean false alarms per station-year (**0.0% rain-coincident**), compared to **303.03** false alarms per station-year for a detection-matched rolling $Z$-score rule ($k = 0.75$) and **7.21 ± 3.93** for the spectral model without weather features (Tier 2). Multi-channel spectral ratios carry the primary discrimination; weather fusion adds a statistically significant further reduction (−40.1% mean, −30.4% median from 6.08 to 4.24 FA/yr; paired $t = 3.111, p = 0.00575$; Wilcoxon $W = 26.0, p = 0.00318$), primarily by suppressing storm-time false-alarm variance ($\sigma$: 3.93 → 1.29 FA/yr). Detection performance is measured on synthetic events; there is no labeled real fission-product ground truth in the 2017–2025 archive. The weakest operational regimes are short, low-amplitude plumes in dry weather (79.7%–81.4% detection) and single-line $^{131}\text{I}$ releases in leave-one-station-out transfer (66.7% detection).

---

## 1. Introduction

RadNet is the U.S. Environmental Protection Agency's national continuous air-monitoring network for gamma-emitting radionuclides. Its stationary monitors pull air through glass-fiber filters, count gamma rays in eight energy channels (R02: 101–200 keV through R09: 1801–2200 keV), and report hourly gross count rate and ambient exposure rate. Operationally, the network must answer one question quickly: does a count-rate excursion indicate a genuine radiological release, or natural radon progeny washout?

The physics of the confounder is well established. Rain scavenges short-lived radon daughters from the air column and deposits them on the filter and ground surface; their decay ($^{214}\text{Pb}\ T_{1/2} = 26.8\text{ min}$, $^{214}\text{Bi}\ T_{1/2} = 19.9\text{ min}$) produces surges of +40% to +180% above baseline within an hour of rainfall onset. A fixed threshold cannot distinguish this from a plume on gross count rate alone.

This project builds and evaluates a weather-aware anomaly detector:

1. Quantify the false-alarm burden of current fixed-threshold practice on real multi-station data.
2. Construct a physically grounded synthetic fission/activation plume benchmark with strict train/test separation and hard cases embedded in active rain.
3. Test, under a causal (no test-tuning) protocol across 20 random seeds, whether multi-channel spectrometry and NOAA weather fusion measurably reduce false alarms at fixed detection probability.

## 2. Data

- **Radiation**: EPA RadNet hourly CSV archives (2017–2025) for 5 pilot stations chosen for climate diversity and data completeness: Birmingham AL (KBHM), Washington DC (KDCA), San Diego CA (KSAN), Dallas TX (KDFW), Tampa FL (KTPA). Complete 8-channel records only; detector outages and incomplete-channel hours are labeled `unobserved` (59,873 hours across the 394,440-hour 5-station calendar grid) and never interpolated.
- **Weather**: NOAA NCEI Global-Hourly (ISD) from the colocated airport ASOS (hourly AA1 liquid precipitation, temperature, sea-level pressure, dew point, relative humidity). Precipitation is filtered strictly to 1-hour accumulation periods (`01`) with verified quality codes `1` and `5` and deduplicated per UTC hour (routine FM-15 prioritized over FM-16 SPECI); suspect codes and missing depths are recorded as `NaN` (never coerced to verified dry `0.0 mm`).
- **Merge**: Both streams are rounded to the nearest UTC hour and reindexed onto continuous 78,888-hour calendar grids per station. Result: **334,567 valid RadNet hours** (227,939 train in 2017–2022; 106,628 test in 2023–2025 = 12.16 station-years) and **324,550 synchronous radiation-weather observation hours** network-wide, including **15,606 rain hours** with concurrent RadNet observations (`merge_audit_summary.csv`).

Key empirical calibration results (Phases 1–3):

- **Rain→radiation cross-correlation** peaks at lag 0 to +1 h ($r = 0.31$–$0.39$ in continental stations) — the washout response is immediate, consistent with hourly accumulation and ~20–30 min progeny decay (`lag_cross_correlation.csv`).
- **Rain-hour surge distribution**: median +6.9%, 90th percentile +41.4%, maximum +197% of local dry baseline (`rain_washout_surge_distribution.csv`).
- **Empirical dose coupling** (trailing 168h baseline window): $\Delta\text{dose} = k_{\text{dose}} \cdot \Delta\text{CPM}$, with $k_{\text{dose}} \in [0.01142, 0.02359]\text{ nSv/h per CPM}$ by station (pooled $0.01344\text{ nSv/h per CPM}$, $R^2 = 0.767$ over 5,324 verified rain hours; `dose_rate_cpm_regression_summary.csv`).
- **Washout spectral share** (ratio-of-sums over 5,494 substantial rain hours): R07+R08 high-energy share = **5.60%** (95% CI: [5.55%, 5.67%]; `rain_washout_spectral_shares.csv`) — directly overlapping the modeled fresh-reactor $^{137}\text{Cs}/^{134}\text{Cs}/^{131}\text{I}$ mixture (4.54%), confirming that high-energy share alone cannot separate reactor plumes from washout.

## 3. Methods

### 3.1 Baseline (current practice)

Three rule families evaluated on real data (`baseline_threshold_evaluation.csv`): global dry-baseline $\sigma$, rolling 7-day mean + global dry $\sigma$, and global exposure-rate $\sigma$, at $k \in \{3, 4, 5\}$ (plus local rolling 168h $Z$-score on the Phase 5 benchmark). Alarm episodes are clustered on the continuous calendar grid; rain coincidence is scored at both hour and episode levels in 1 h, 3 h, and 6 h windows.

### 3.2 Synthetic injection benchmark

Injections accumulate particulate activity on the filter (passage phase with shape families `step`, `linear_ramp` for training; `sigmoidal`, `exponential` for test), then persist and decay ($^{131}\text{I}$-dominated $\lambda = 0.00360\ \mathrm{h}^{-1}$) until the **real** filter replacement step drop ends the event — 450/450 injections are synchronized to empirically detected filter changes, with a strict 48 h non-overlap buffer (`injection_catalog_audit_verification.csv`). Five release scenarios ($^{137}\text{Cs}$, $^{131}\text{I}$, $^{60}\text{Co}$, fresh-reactor $^{137}\text{Cs}/^{134}\text{Cs}/^{131}\text{I}$, mixed) are balanced to exactly 20% in each of six environments (dry/rain × train/standard/stress). Magnitude bands: A 250–600 CPM (subtle), B 700–1200 CPM (moderate, test-interpolation), C 1400–2500 CPM (large). The test split (2023–2025) is calendar-disjoint from training (2017–2022) and uses unseen shapes and magnitude bands. 100 test injections are rain-coincident hard cases. Unobserved detector-outage hours inside long injection windows remain labeled `unobserved` (`labeled_dataset_reconciliation_summary.csv`).

### 3.3 Models and features

Four LightGBM multi-class tiers (`normal` / `radon_washout` / `fission_product`):
- **Tier 1**: Gross-rate & dose dynamics (12 features).
- **Tier 1b**: Gross-rate & dose + NOAA weather (29 features).
- **Tier 2**: Gross-rate & dose + NaI(Tl) spectral ratios (30 features).
- **Tier 3**: Full spectral + NOAA weather fusion (48 features).
- **Tier 3 MLP**: 2-layer neural network (raw and isotonic-calibrated) on the 48 Tier 3 features.

All 48 features are strictly causal (backward-looking windows only, verified via perturbation invariance in `tests/test_radnet_pipeline.py`), and test-split featurization prepends 250 hours of clean pre-2023 history so 168-hour rolling statistics at `2023-01-01` have full warm-up history.

### 3.4 Evaluation protocol

- Operating thresholds $\tau^*$ are selected strictly on the 2021–2022 validation fold (74,199 observed hours = 8.46 station-years; 81 validation injections) after fitting on the 2017–2020 training fold (153,740 observed hours; 169 injections), and frozen before evaluating on the 2023–2025 test split (106,628 observed hours = 12.16 station-years; 200 test injections).
- Dual-series design: detection is measured on the injected series under the **net-alarm criterion** (alarm on injected series and no alarm on clean series, unified across both `train_and_evaluate_rigorous.py` and `evaluate_phase5_protocol.py`); false alarms are measured on unmodified clean background in discrete episodes per station-year ($N_{\text{obs}}/8766$).
- 20-seed replication (Seeds 42–61), paired $t$ and Wilcoxon signed-rank tests, station-month block bootstrap ($B = 1000$), matched-detection comparisons, leave-one-station-out (LOSO), ±10% gain-drift and ±20% spectral-template sensitivity sweeps.

## 4. Results

### 4.1 The baseline problem (real data, Phase 2)

At $3\sigma$ (`rolling_cpm_3s`, rolling 7-day mean + $3\sigma_{\text{dry}}$), alarm episodes per station-year (`baseline_threshold_evaluation.csv`): Birmingham 53.57, Washington 52.98, Dallas 62.91, San Diego 20.91, Tampa 6.59. Rain coincidence within 3 h among those alarm hours (and episodes): Washington 86.50% (88.52% of episodes), Dallas 68.47% (70.50% of episodes), Birmingham 61.43% (69.54% of episodes), Tampa 21.11% (27.66% of episodes), San Diego 0.0%. At $5\sigma$ the total drops (13.19–27.35 episodes/yr in humid stations) but the remaining alarms are even more rain-dominated (85.37%–97.20% of alarm hours). Fixed thresholds trade alarm volume against blindness; they do not solve the discrimination problem.

### 4.2 Detection at validation-frozen operating points (synthetic test events)

20-seed mean ± SD on the 2023–2025 test split, thresholds frozen on 2021–2022 (`rigorous_benchmark_summary.csv`, 95% validation target row):

| Model / Rule | Frozen $\tau$ | Realized Test Detection | Clean FA / stn-yr | 95% CI (FA) | Rain-Coincident FA | Median Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Rolling 7d $3\sigma$ Baseline | $k=3.0$ | 52.50% [45.0, 59.0] | 82.62 | [74.40, 91.85] | 32.5% | 10.0 h |
| Global Dry $3\sigma$ Baseline | $k=3.0$ | 57.00% [50.0, 64.0] | 52.86 | [44.72, 61.34] | 24.9% | 9.0 h |
| Rolling 7d $5\sigma$ Baseline | $k=5.0$ | 6.00% [3.0, 9.5] | 23.51 | [19.48, 28.04] | 57.6% | 8.5 h |
| Tier 1: Gross GBDT | 0.647 | 92.00 ± 0.69% [88.2, 95.4] | 166.78 ± 19.69 | [150.94, 185.14] | 17.3% | 6.7 h |
| Tier 1b: Gross + Weather | 0.878 | 85.58 ± 1.59% [80.8, 90.1] | 29.27 ± 3.93 | [23.54, 35.75] | 0.0% | 9.5 h |
| Tier 2: Spectral GBDT | 0.990 | 92.98 ± 0.92% [89.3, 96.0] | 7.21 ± 3.93 | [4.41, 10.57] | 4.7% | 4.0 h |
| **Tier 3: Weather Fusion GBDT** | **0.988** | **92.65 ± 1.37% [89.2, 95.7]** | **4.32 ± 1.29** | **[2.66, 6.31]** | **0.0%** | **4.0 h** |
| Tier 3: Weather-Fused MLP | 0.993 | 90.18 ± 2.81% [87.0, 93.2] | 58.30 ± 23.63 | [42.17, 77.38] | 3.7% | 3.6 h |

At matched ~92.5% detection on the continuous test sweep (`rigorous_matched_detection_comparison.csv`), a rolling $Z$-score rule requires $k = 0.75$ and fires **303.03 FA/stn-yr** — roughly 70× the Tier 3 rate.

### 4.3 What each modality contributes (`rigorous_seed_statistical_tests.csv`)

- **Spectrometry is the primary discriminator**: Tier 2 removes **95.7%** of Tier 1's false alarms at ~92–93% detection (`166.78 → 7.21 FA/yr`). Physically, natural radon washout is concentrated in R02/R03 (`42.27%` / `33.75%` ratio-of-sums) and decays within hours, whereas filter-trapped $^{137}\text{Cs}$ and $^{60}\text{Co}$ persistently shift mid- and high-energy channel shares across the 3–5 day filter cycle.
- **Weather on gross-only detectors** (Tier 1b at 98% target vs Tier 1 at 95% target, matched `91.95%` vs `92.00%` realized detection): **−58.1% FA** (`166.78 → 69.81 FA/yr`, mean reduction `96.96 FA/yr`, 95% CI: `[85.05, 111.26]`, paired $t = 20.399, p < 10^{-14}$, Wilcoxon $W = 0.0, p < 10^{-5}$, 20/20 seeds). For monitoring networks without spectrometry, weather telemetry alone eliminates the majority of washout false alarms.
- **Weather on top of spectrometry** (Tier 3 vs Tier 2 at 95% target, matched `92.65%` vs `92.98%` realized detection): **−40.1% mean FA reduction** (`7.21 → 4.32 FA/yr`, mean difference `2.89 FA/yr`, paired bootstrap 95% CI: `[0.31, 5.74]`, paired $t = 3.111, p = 0.00575$) and **−30.4% median FA reduction** (`6.08 → 4.24 FA/yr`, Wilcoxon $W = 26.0, p = 0.00318$, 15 of 20 seeds lower). Crucially, weather fusion reduces seed-to-seed false-alarm standard deviation by 3× ($\sigma$: `3.93 → 1.29 FA/yr`) and eliminates 100% of rain-coincident false alarms (`4.7% → 0.0%`).
- **Neural network (MLP) comparison**: Under frozen validation thresholds across 20 seeds, the 2-layer MLP achieves `90.18 ± 2.81%` detection with `58.30 ± 23.63 FA/yr` (~13.5× higher than Tier 3 LightGBM). On Seed 42 (`eval_mlp_calibration_comparison.csv`), post-hoc isotonic regression fitted on validation probabilities maps scores to empirical probabilities (`89.5%` test detection, `45.38 FA/yr` vs `91.5%` and `43.98 FA/yr` uncalibrated), confirming that monotonic post-hoc calibration preserves rank order and cannot close the structural tree-vs-MLP gap on tabular physical ratios.

### 4.4 Timing and environmental stratification (`rigorous_stratified_deadlines.csv`)

20-seed mean net-alarm detection deadlines at the frozen 95% validation threshold:
- **All Test Injections ($N=200$)**: Tier 3 detects **70.55%** within $\le 6\text{h}$, **88.53%** within $\le 12\text{h}$, **91.92%** within $\le 24\text{h}$, and **92.65%** total (median delay **4.0 h**); Tier 2 detects **73.08%** $\le 6\text{h}$, **90.10%** $\le 12\text{h}$, **92.75%** $\le 24\text{h}$, and **92.97%** total (median delay **4.0 h**).
- **Rain-Coincident Environments ($N=100$)**: Tier 3 achieves **95.55%** total detection (**90.70%** $\le 12\text{h}$, median delay **5.0 h**).
- **Standard Plume Regime ($N=100$, Band B)**: Tier 3 achieves **98.90%** total detection (**99.80%** in Standard Dry, **98.00%** in Standard Rain).
- **Stress Plume Regime ($N=100$, Band A)**: Tier 3 achieves **86.40%** total detection (**79.70%** in Stress Strictly Dry, **93.10%** in Stress Rain).

### 4.5 Where the model is weak (operational boundary conditions)

- **Stress Strictly Dry regime** (8–20 h plumes, 255–599 CPM, zero rain): **79.7%** detection for Tier 3 (**81.4%** for Tier 2) — roughly one miss in five. Short, subtle plumes in quiet weather are the primary sensitivity boundary.
- **Leave-One-Station-Out (LOSO) transfer** (`rigorous_loso_summary.csv`, `rigorous_loso_stratified_detection.csv`):
  - Holding out **San Diego** (`ca_san_diego`): **0.00 FA/yr** (rule-of-three 95% upper bound $\le 1.05\text{ FA/yr}$) and **67.5%** detection (27/40: Band B **90.0%**, Band A **45.0%**).
  - Holding out **Birmingham** (`al_birmingham`): **0.00 FA/yr** (rule-of-three upper bound $\le 1.08\text{ FA/yr}$) and **87.5%** detection (35/40: Band B **100.0%**, Band A **75.0%**).
  - Holding out **Dallas** (`tx_dallas`): **15.29 FA/yr** (27 episodes across 1.77 station-years) and **90.0%** detection (36/40: Band B **100.0%**, Band A **80.0%**).
  - Pooled across all 120 held-out events: Band B standard plumes achieve **96.7%** detection (58/60), whereas Band A subtle plumes drop to **66.7%** (40/60). By scenario: $^{137}\text{Cs}$ **100.0%** (24/24), fresh-reactor mix **91.7%** (22/24), mixed fission-activation **79.2%** (19/24), $^{60}\text{Co}$ **70.8%** (17/24), and pure $^{131}\text{I}$ **66.7%** (16/24).
- **Instrumental gain drift** (`rigorous_gain_drift_evaluation.csv`, 5 seeds): Symmetrically perturbing channel boundaries by $-10\%, -5\%, 0\%, +5\%, +10\%$ yields clean false alarms of `7.51 ± 1.85`, `5.69 ± 1.55`, `4.67 ± 1.21`, `5.44 ± 1.43`, and `21.93 ± 6.18 FA/yr`. Upward gain drift of $+10\%$ shifts low-energy Compton counts into photopeak channels and raises false alarms nearly 5×; automated field stabilization (e.g. locking onto the natural $^{40}\text{K}$ 1461 keV photopeak) is required for deployment.
- **Spectral template sensitivity** (`rigorous_template_sensitivity.csv`, 5 seeds): Scaling photopeak shares by $-20\%, -10\%, 0\%, +10\%, +20\%$ shifts test detection across `87.3 ± 1.40%`, `89.4 ± 0.66%`, `92.6 ± 1.59%`, `94.1 ± 0.73%`, and `94.8 ± 1.03%` (~2–3 percentage points per 10% shift).

---

## 5. Reproducibility, CLI Diagnostics & Automated Verification

- **Master Pipeline Runner**: `bash run_pipeline.sh` executes Phases 0 through 5 sequentially.
- **Automated Verification Suite**: `.venv/bin/pytest -v` runs [`tests/test_radnet_pipeline.py`](../tests/test_radnet_pipeline.py), verifying causal feature engineering (zero lookahead), spectral channel normalization, train-only dry baseline non-leakage, NOAA hourly precipitation deduplication, unobserved detector-outage labeling, and cross-table CSV consistency.
- **Operational CLI Diagnostic Utility**: [`src/detect_anomalies_cli.py`](../src/detect_anomalies_cli.py) provides command-line inference, episode clustering, and physical attribution on any station and time window.

---

## Appendix A. Provenance of Headline Numbers

| Claim | Source Table |
| :--- | :--- |
| 334,567 valid RadNet hours; 324,550 synchronous hours; 15,606 rain hours | `merge_audit_summary.csv`, `labeled_dataset_reconciliation_summary.csv` |
| Deduplicated annual rainfall (Birmingham 1,691.6 mm/yr, etc.) | `noaa_precipitation_audit.csv`, `noaa_station_summary.csv`, `merge_audit_summary.csv` |
| Baseline episodes/yr & hour/episode rain coincidence | `baseline_threshold_evaluation.csv` |
| Filter cycle intervals across all 5 stations (median 2.92–4.54 days) | `multi_station_filter_cycle_summary.csv` |
| Rain surge distribution | `rain_washout_surge_distribution.csv` |
| $k_{\text{dose}}$ regression (5,324 rain hours, pooled 0.01344 nSv/h per CPM) | `dose_rate_cpm_regression_summary.csv` |
| Washout spectral shares (5,494 rain hours, R07+R08 = 5.60%) | `rain_washout_spectral_shares.csv` |
| 450 injections, balance, 48h buffer, empirical filter-drop verification | `synthetic_injection_catalog.csv`, `injection_catalog_audit_verification.csv` |
| 20-seed benchmark (Tier 3: 92.65% det, 4.32 FA/yr, 0.0% rain FA) | `rigorous_benchmark_summary.csv`, `rigorous_seed_level_results.csv` |
| Paired hypothesis tests (Tier 3 vs Tier 2: $t=3.111, p=0.00575, W=26.0, p=0.00318$) | `rigorous_seed_statistical_tests.csv` |
| Matched-detection baselines (303.03 FA/yr at $k=0.75$) | `rigorous_matched_detection_comparison.csv` |
| 20-seed stratified detection deadlines | `rigorous_stratified_deadlines.csv` |
| Multi-station LOSO evaluation | `rigorous_loso_summary.csv`, `rigorous_loso_stratified_detection.csv` |
| Gain drift & spectral template sensitivity (5-seed sweeps) | `rigorous_gain_drift_evaluation.csv`, `rigorous_template_sensitivity.csv` |
| Seed 42 bootstrap uncertainty, empirical delays & MLP comparison | `eval_benchmark_uncertainty_summary.csv`, `eval_mlp_comparison_summary.csv`, `eval_mlp_calibration_comparison.csv`, `eval_empirical_delays.csv` |
