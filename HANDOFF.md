# Project Architecture & Pipeline Guide: Weather-Aware Anomaly Detection in RadNet

> **Maintenance update (2026-10-18):** Feature schema 2 corrects missing-weather handling and the research CLI calibrates each fitted model. The archived benchmark figures and numbers below describe the earlier schema. Use the portable `python run_pipeline.py` runner and current [README](README.md); see [maintenance verification](reports/MAINTENANCE_2026-10-18.md). The full 20-seed evaluation has not been rerun for schema 2.

**Date**: 2026-10-17  
**Project**: Weather-Aware Anomaly Detection in RadNet (EPA RadNet + NOAA NCEI ISD)  

---

## 1. Executive Summary & Repository Status

This repository contains the data processing pipeline, physical gamma spectrometry modeling, multi-seed evaluation suite, automated verification tests, and operational CLI diagnostic utility for weather-aware radiological anomaly detection on EPA RadNet near-real-time gamma monitors.

- **Status**: Phases 0 through 6 are implemented, verified end-to-end, and covered by automated `pytest` tests (`tests/test_radnet_pipeline.py`).
- **Documentation**: All architectural decisions, citations, and formulas are documented in [`DECISIONS.md`](DECISIONS.md), [`PHASE_REPORT.md`](PHASE_REPORT.md), [`reports/PHASE6_REPORT.md`](reports/PHASE6_REPORT.md), and [`reports/BUG_AUDIT.md`](reports/BUG_AUDIT.md). Open research questions are logged in [`QUESTIONS.md`](QUESTIONS.md).

---

## 2. What Each Phase Produced

### Phase 0: Station Selection & Exploratory Data Analysis
- **Objective**: Select 5 geographically and meteorologically diverse EPA RadNet stations with high completeness across the continuous 2017–2025 observation window.
- **Key Scripts**:
  - `src/parse_stations.py`: Scrapes EPA RadNet downloads index.
  - `src/download_pilot_data.py`: Downloads pilot station archives.
  - `src/analyze_pilot_stations.py`: Audits channel completeness and missingness.
- **Key Artifacts**:
  - `data/processed/radnet_station_inventory.csv`: Complete EPA inventory (127+ stations).
  - `data/processed/candidate_pilot_stations_comparison.csv`: Candidate evaluation metrics.
  - `data/processed/pilot_station_metrics.csv` & `pilot_station_channel_missingness.csv`: Completeness audits.
- **Selected Stations**:
  1. `AL: Birmingham` (`KBHM`, Inland Southeast, karst/limestone geology, high radon potential).
  2. `CA: San Diego` (`KSAN`, Coastal Pacific, marine boundary layer, coastal humidity inversion).
  3. `DC: Washington` (`KDCA`, Mid-Atlantic Fall Line, humid subtropical).
  4. `FL: Tampa` (`KTPA`, Coastal Gulf, convective thunderstorms, phosphate geology).
  5. `TX: Dallas` (`KDFW`, Southern Plains, continental fronts, rapid barometric drops).

### Phase 1: Weather Data Ingestion, Quality Filtering & Synchronous Merging
- **Objective**: Download synchronous NOAA NCEI Global-Hourly (ISD) precipitation and weather telemetry; merge with RadNet on a continuous calendar grid.
- **Key Scripts**:
  - `src/download_noaa_data.py`: Fetches NOAA ISD observations (2017–2025).
  - `src/audit_noaa_precipitation.py`: Audits hourly precipitation report frequencies, deduplication, and quality codes.
  - `src/merge_radnet_weather.py`: Merges RadNet gamma and NOAA weather on a continuous UTC hourly grid.
  - `src/plot_rain_events.py`: Visualizes multi-station rain washout episodes.
- **Key Artifacts & Figures**:
  - `data/processed/merged_{station}_2017_2025.csv.gz`: 5 station merged datasets (**334,567 valid RadNet hours** and **324,550 synchronous radiation-weather observation hours** across 394,440 grid hours).
  - `data/processed/noaa_precipitation_audit.csv`, `noaa_station_summary.csv`, & `merge_audit_summary.csv`.
  - `reports/figures/rain_event_{station}.png` (5 figures) & `rain_washout_multi_station_comparison.png`.
- **Key Rules Applied**: AA1 precipitation condition codes restricted strictly to 1-hour period (`01`) with quality codes `'1'` and `'5'` (passed automated & manual checks) and deduplicated per UTC hour (routine FM-15 prioritized over FM-16 SPECI); suspect codes (`'C'`, `'S'`) and missing depths are recorded as `NaN` (never coerced to verified dry `0.0 mm`). Peak physical lag confirmed at 0 to +1 hour.

### Phase 2: Physical Baseline Characterization & Static Alarm Evaluation
- **Objective**: Quantify false alarm burden of current-practice fixed-threshold monitors under natural radon washout and environmental cycling.
- **Key Scripts**:
  - `src/characterize_baseline_cycles.py`: Analyzes diurnal radon cycle and filter advance step-drops.
  - `src/analyze_filter_cycles_multi_station.py`: Multi-station filter advance analysis across all 5 pilot stations.
  - `src/analyze_timestamp_lags.py`: Calendar-reindexed cross-correlation lag profile.
  - `src/analyze_rain_washout_distribution.py`: Rain surge amplitude distribution.
  - `src/evaluate_baseline.py` & `src/plot_baseline_timeline.py`: Evaluates static $3\sigma$, $4\sigma$, and $5\sigma$ thresholds.
- **Key Artifacts & Figures**:
  - `data/processed/baseline_threshold_evaluation.csv`, `lag_cross_correlation.csv`, `rain_washout_surge_distribution.csv`, `multi_station_filter_cycle_summary.csv`.
  - Figures: `baseline_alarm_rate_by_threshold.png`, `baseline_alarm_rain_coincidence.png`, `baseline_alarm_example_timeline.png`, `diurnal_radon_cycle.png`, `filter_cycle_sawtooth.png`.
- **Key Findings**: Standard rolling 7-day + $3\sigma_{\text{dry}}$ alarms produce 53.0–62.9 alarm episodes per station-year in humid continental climates (Birmingham 53.6, Washington 53.0, Dallas 62.9, San Diego 20.9, Tampa 6.6); 58%–87% of alarm hours (70%–89% of alarm episodes) in humid cities coincide directly with rain washout within 3 hours. Raising the threshold to $5\sigma$ leaves 13.2–27.4 episodes/yr in humid cities, of which 85%–97% of alarm hours remain rain-coincident false alarms.

### Phase 3: Synthetic Injection Design & Labeled Benchmark Catalog
- **Objective**: Generate physically grounded synthetic anthropogenic anomaly injections across realistic meteorological conditions, enforcing strict train/test disjointness and non-overlap.
- **Key Scripts**:
  - `src/compute_train_only_baselines.py`: Computes frozen pre-2023 dry-weather baselines per station.
  - `src/calibrate_dose_coupling.py`: Calibrates gross CPM to dose-rate coupling using trailing 168h medians.
  - `src/analyze_washout_spectrum.py`: Derives empirical multi-channel washout shares.
  - `src/synthetic_injection.py`: Applies the 450 frozen synthetic plumes to continuous background while preserving `"unobserved"` labels on detector-outage hours.
  - `src/audit_injection_catalog.py`: Empirically verifies non-overlap buffer, filter-drop anchoring, channel completeness, strictly dry constraints, and spectral normalization against the merged station panels.
  - `src/plot_synthetic_injections.py`: Plots shapes, nuclide spectra, and hard-case rain regimes.
- **Key Artifacts & Figures**:
  - `data/processed/synthetic_injection_catalog.csv`: 450 injections (250 train [2017–2022], 200 test [2023–2025]).
  - `data/processed/labeled_{station}_{split}.csv.gz`: 10 pre-injected continuous hourly datasets (`334,567` observed hours: `227,939` train, `106,628` test = `12.16` station-years; `59,873` unobserved hours).
  - `data/processed/station_dry_baselines_train_only.json`: Frozen pre-2023 dry-baseline reference.
  - Figures: `synthetic_injection_shapes_and_nuclides.png`, `synthetic_injection_hard_case_rain.png`, `synthetic_injection_train_test_disjointness.png`.

### Phase 4: Feature Engineering & Machine Learning Detection Pipeline
- **Objective**: Engineer strictly causal physical features (spectral channel ratios, temporal diffs, rolling stats, weather interactions) and train LightGBM classifiers.
- **Key Scripts**:
  - `src/feature_engineering.py`: Computes 48 causal physics-informed spatio-temporal features across Tiers 1, 1b, 2, and 3.
  - `src/train_and_evaluate_models.py`: Trains LightGBM tiers with validation-frozen thresholds and pre-2023 warm-up history, plus Leave-One-Station-Out (LOSO) evaluation.
  - `src/plot_model_evaluation.py`: Generates ROC and feature importance figures dynamically from CSV outputs.
- **Key Artifacts & Figures**:
  - `data/processed/model_feature_importance.csv`, `model_roc_curve_data.csv`, `model_operating_points_summary.csv`, `model_loso_evaluation_summary.csv`.
  - Figures: `model_detection_vs_false_alarms_roc.png`, `model_feature_importance.png`, `model_hard_case_timeline.png`, `model_stratified_performance.png`.

### Phase 5 & 6: Rigorous Evaluation Protocol, Multi-Seed Replication, CLI & Test Suite
- **Objective**: 20-seed replication, continuous ROC evaluation on the 2023–2025 test split, matched-detection comparison at fixed false-alarm rates, causal validation threshold freezing, temperature/gain drift robustness, MLP neural network calibration comparison, synthesis visualization, operational CLI diagnostics, and automated test verification.
- **Key Scripts**:
  - `src/train_and_evaluate_rigorous.py`: Core runner for 20-seed replication, continuous ROC, matched FAR detection, LOSO, gain drift, and template sensitivity.
  - `src/evaluate_phase5_protocol.py`: Seed 42 uncertainty, confusion matrices, empirical delay CDFs, and MLP comparison under the unified net-alarm + validation-frozen protocol.
  - `src/plot_rigorous_evaluation.py`, `src/plot_phase5_evaluation.py`, & `src/plot_environmental_engineering_synthesis.py`: Phase 5 and synthesis visualizations built strictly from empirical telemetry and benchmark CSVs.
  - `src/detect_anomalies_cli.py`: Operational CLI diagnostic utility for inspecting any station, time window, model tier, and clean/injected series.
  - `tests/test_radnet_pipeline.py`: Automated `pytest` verification suite testing causal feature engineering, spectral conservation, train-only baseline non-leakage, NOAA precipitation deduplication, unobserved outage labeling, and cross-table CSV consistency.

---

## 3. How to Run the Pipeline & Tests

Execute commands from the repository root using the project virtual environment:

```bash
# Activate environment
source .venv/bin/activate

# Run the automated test suite (9 comprehensive verification checks)
pytest -v

# Run operational CLI diagnostics on a specific station/window
python src/detect_anomalies_cli.py --station al_birmingham --start "2023-01-24 18:00:00" --end "2023-01-29 06:00:00" --use-injected

# Run the full pipeline end-to-end (or a specific phase)
bash run_pipeline.sh
bash run_pipeline.sh --phase 5
```
