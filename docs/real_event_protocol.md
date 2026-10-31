# Real-Event Validation Pre-Registration Protocol

**Protocol Version**: 1.0.0  
**Registration Date**: October 31, 2026, 19:35 UTC  
**Target Event**: Fukushima Daiichi Atmospheric Trans-Pacific Transport (March–April 2011)  
**Status**: PRE-REGISTERED (Committed prior to execution of any real-event model evaluation)

---

## 1. Objectives and Scope

The goal of this evaluation is to test the frozen machine learning detectors and non-ML baselines—trained, tuned, and calibrated strictly on 2017–2022 data—on the only documented trans-Pacific nuclear release captured by modern RadNet stations: the March–April 2011 Fukushima Daiichi accident.

This protocol specifies all models, thresholds, stations, temporal windows, feature transformations, evaluation metrics, and reporting rules **prior to executing any evaluation code or inspecting any model outputs on the 2011 data**.

---

## 2. Frozen Models and Operating Thresholds

All models and decision thresholds evaluated in this protocol are frozen from earlier development folds (trained on 2017–2020, calibrated on 2021–2022 validation fold). **Zero parameters, weights, or thresholds will be modified after seeing 2011 results.**

> [!IMPORTANT]
> **Threshold Documentation Discrepancy Note:** An earlier draft of this protocol document listed tentative threshold values ($\tau = 0.5802$ for Tier 4, $\tau = 0.5214$ for Tier 3, $\tau = 0.4950$ for Tier 2, and $\tau = 0.4120$ for Tier 1). However, the actual evaluation script (`src/evaluate_real_event.py`) loaded the pre-specified validation-selected thresholds calibrated at the 5 FA/yr operating budget:
> - **Tier 4 (Physics-Motivated Nuclear)**: $\tau = 0.8719$ (59 features)
> - **Tier 3 (Full Weather Fusion)**: $\tau = 0.8726$ (48 features)
> - **Tier 2 (Spectral Only)**: $\tau = 0.9475$ (30 features)
> - **Tier 1 (Gross Count Only)**: $\tau = 0.9238$ (12 features)
> Per Ground Rule 2 (no post-hoc tuning on evaluation events), the evaluation code was executed with these frozen thresholds, and no thresholds have been retroactively modified.

1. **LightGBM Tier 4 (Physics-Motivated Nuclear)**:
   - Features: `TIER_4_NUCLEAR_FEATURES` (59 features including Bateman decay ratio, radon projection residuals, stripped excess $z$-scores, local 168h rolling statistics, and weather fusion).
   - Validation-calibrated operating point: 5 FA/yr validation budget operating threshold ($\tau = 0.8719$, frozen from validation calibration).
   - Model weights: Loaded from frozen LightGBM ensemble trained on 2017–2020 pilot stations.
2. **LightGBM Tier 3 (Full Weather Fusion)**:
   - Features: `TIER_3_FEATURES` (48 features including gross rates, spectral shares, rainfall accumulations, pressure, and humidity interactions).
   - Validation-calibrated operating threshold: $\tau = 0.8726$.
3. **LightGBM Tier 2 (Spectral Only)**:
   - Features: `TIER_2_FEATURES` (30 features: gross + 8 energy channel shares and ratios).
   - Validation-calibrated operating threshold: $\tau = 0.9475$.
4. **LightGBM Tier 1 (Gross Count Only)**:
   - Features: `TIER_1_FEATURES` (12 features: gross count $z$-scores and rolling ratios).
   - Validation-calibrated operating threshold: $\tau = 0.9238$.
5. **Non-ML Baseline B1 (Gross Global Dry 3$\sigma$)**:
   - Threshold: Static pre-event baseline $+ 3.0 \times \sigma_{dry}$.
6. **Non-ML Baseline B2 (Rolling 168h 3$\sigma$)**:
   - Threshold: Causal 168h trailing mean $+ 3.0 \times \sigma_{168h}$.
7. **Non-ML Baseline B3 (Rolling 168h 4$\sigma$)**:
   - Threshold: Causal 168h trailing mean $+ 4.0 \times \sigma_{168h}$.
8. **Non-ML Baseline B4c (Radon Projection Residual 3$\sigma$)**:
   - Threshold: Causal 168h radon background subspace projection error $+ 3.0 \times \sigma_{res}$.

---

## 3. Stations and Geographical Rationale

We evaluate three Pacific gateway stations closest to the trans-Pacific transport corridor, plus one continental interior control station:
1. **Honolulu, HI (`hi_honolulu`)**: Primary Pacific oceanic station. First documented US RadNet air filter detection of $^{131}\text{I}$ occurred here on March 18, 2011 (EPA Japanese Nuclear Emergency Report; Thakur et al., 2012).
2. **San Diego, CA (`ca_san_diego`)**: Primary contiguous US West Coast station. Documented $^{131}\text{I}$ and $^{134/137}\text{Cs}$ arrival March 18–20, 2011.
3. **Anchorage, AK (`ak_anchorage`)**: Northern Pacific subarctic gateway. Plume traversed the Aleutian arc and Gulf of Alaska during late March 2011.
4. **Dallas, TX (`tx_dallas`)**: Continental interior control station with continuous 2011 monitoring.

---

## 4. Evaluation Windows

To prevent temporal boundary leakage and to evaluate false alarms before and after the event under identical operating conditions, each station series is partitioned into three distinct, non-overlapping temporal windows:

1. **Pre-Event Clean Baseline Window**:
   - `2011-01-01 00:00:00` to `2011-03-11 05:46:00` UTC (moment of Great East Japan Earthquake).
   - Duration: 1,661 hours (~69.2 days).
   - Purpose: Establish causal local baselines and verify clean background false alarm rates prior to any fission product release.
2. **Active Event Transport Window**:
   - `2011-03-11 05:47:00` to `2011-04-30 23:59:00` UTC.
   - Duration: 1,218 hours (~50.8 days).
   - Covers: Initial reactor meltdowns, primary trans-Pacific air mass transit, first documented detections (March 18, 2011), and peak atmospheric concentrations across the Western United States.
3. **Post-Event Clean Window**:
   - `2011-05-01 00:00:00` to `2011-06-30 23:59:00` UTC.
   - Duration: 1,464 hours (~61.0 days).
   - Covers: Atmosphere returning to baseline following seasonal scavenging.

---

## 5. Domain Shift Adaptation Protocol

Because the 2011 dataset precedes the training fold (2017–2020) by 6 years, two domain shift factors must be addressed causally without test-set tuning:

1. **Absent Dose Rate Feature (`dose_rate = NaN`)**:
   - As documented in `docs/real_event_feasibility.md`, `DOSE EQUIVALENT RATE (nSv/h)` was not populated by EPA RadNet in 2011.
   - Adaptation: The pipeline feature computer sets `dose_present = 0.0`. All dose-derived features (`dose_z_score_168h`, `dose_to_gross_ratio`, `dose_spectral_residual_z`) are processed by the frozen pipeline's `SimpleImputer(strategy='median')`. No model retraining or architecture alteration is permitted.
2. **Local Baseline and Gain Calibration**:
   - Detectors relying on static baselines ($z_{global}$, dry baseline shares) will use statistics computed **strictly from the Pre-Event Clean Baseline Window (`2011-01-01` to `2011-03-10`)**. No data from the event or post-event window is used to fit baselines.
   - Rolling features (168h and 24h rolling mean/std) operate causally online with a trailing window.

---

## 6. Pre-Registered Hypotheses and Primary/Secondary Metrics

### 6.1 Pre-Registered Hypotheses
- **Hypothesis 1 (Physical Activity Constraint)**: Based on atmospheric dispersion physics and published concentrations ($<0.1\text{ Bq/m}^3$ $^{131}\text{I}$, producing $<0.001\text{ CPM}$ on a 2"$\times$2" $\text{NaI(Tl)}$ detector), no frozen detector will detect an ambient radiological excess above Poisson counting noise in hourly data.
- **Hypothesis 2 (False Alarm Suppression Under Washout)**: Precipitation events during the active window will produce natural $^{214}\text{Pb}/^{214}\text{Bi}$ radon washout spikes. Models with weather fusion and physics-motivated radon stripping (Tier 3, Tier 4) will suppress these natural rain spikes, whereas gross-count baselines (B1, B2, Tier 1) will trigger elevated false alarms during rain events.

### 6.2 Metrics
1. **Primary Metric**:
   - **Binary Event Detection ($Y_{event} \in \{0, 1\}$)**: Did the detector raise an alarm at any station during the active event window (`2011-03-11` to `2011-04-30`)?
   - **Lead/Lag Relative to Earliest Documented Detection**: If an alarm was raised, what was the timestamp and difference relative to the earliest documented RadNet filter detection (Honolulu: `2011-03-18 00:00:00` UTC)?
2. **Secondary Metrics**:
   - **Alarm Hours and Rate (FA/yr equivalent)**:
     - Pre-event clean rate ($FA_{pre}/\text{yr}$)
     - Active event window alarm rate ($FA_{event}/\text{yr}$)
     - Post-event clean rate ($FA_{post}/\text{yr}$)
   - **Rain-Coincidence Fraction**: Percentage of alarms occurring during active precipitation ($precip > 0$) or within 6 hours of rain.
   - **Spectral Channel Signature Analysis**: For any alarms raised during the event, compare the mean channel shares ($R_{02}$–$R_{09}$) against the Monte Carlo synthetic template families (Family A Cs-137 / I-131 vs. natural radon washout) to adjudicate whether the alarm was driven by fission spectral energy or radon daughter peaks ($R_{03}$, $R_{05}$).

---

## 7. Execution and Reporting Rules

1. This protocol is committed to git before running any script that generates 2011 model predictions.
2. The evaluation script will be executed once (`src/evaluate_real_event.py`).
3. All outputs will be written to `data/processed/real_event_fukushima_evaluation.csv` and `data/processed/real_event_fukushima_station_summary.csv`.
4. Timeline figures will be plotted in `reports/figures/real_event_fukushima_timeline.png`.
5. Under no circumstances will any threshold or hyperparameter be retuned to "improve" detection on the 2011 data. If zero alarms are raised on the plume, it will be reported as confirmation that diluted trans-oceanic releases fall below single-station hourly $\text{NaI(Tl)}$ detection limits.
