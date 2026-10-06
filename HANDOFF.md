# Project Handoff: Weather-Aware Anomaly Detection in RadNet

**Date**: 2026-10-06  
**Project**: Weather-Aware Anomaly Detection in RadNet (EPA RadNet + NOAA NCEI ISD)  
**Roles & Governance**:
- **MiMo**: Lead Developer & Project Boss / Executive Decision Maker (granted full ownership, administrative clearance, pipeline execution authority, and code leadership).
- **Claude**: Reviewer / Technical Advisor / Co-Boss (evaluates methodology, audits statistical integrity).
- **Ömer**: Human Project Lead / Final Authority.
- **Gemini**: Retired / Previous Assistant (handed off all materials and stepped down).

---

## 1. Executive Summary & Repository Status

This repository contains the complete implementation, data processing pipeline, physical spectrometry modeling, and 20-seed rigorous evaluation of a machine-learning-based, weather-aware radiological anomaly detection system for EPA RadNet near-real-time gamma monitors.

- **Git Commit**: `3ea6567` (`docs(phase5): address Fourth Pass review items on baseline reporting, MLP framing, and neutral tone`)
- **Status**: Phases 0 through 5 are fully implemented.
- **Rule of Engagement**: **Nothing in previous phases is to be edited, fixed, or regenerated without explicit instruction.** All decisions, citations, and formulas are documented in [`DECISIONS.md`](file:///Users/o/Projects/radnet-anomaly-detection/DECISIONS.md) and [`PHASE_REPORT.md`](file:///Users/o/Projects/radnet-anomaly-detection/PHASE_REPORT.md). Open questions remain logged in [`QUESTIONS.md`](file:///Users/o/Projects/radnet-anomaly-detection/QUESTIONS.md).

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
  - `src/audit_noaa_precipitation.py`: Audits precipitation report frequencies and quality codes.
  - `src/merge_radnet_weather.py`: Merges RadNet gamma and NOAA weather on UTC hourly grid.
  - `src/plot_rain_events.py`: Visualizes multi-station rain washout episodes.
- **Key Artifacts & Figures**:
  - `data/processed/merged_{station}_2017_2025.csv.gz`: 5 station merged datasets (**324,550 synchronous observation hours**).
  - `data/processed/noaa_precipitation_audit.csv` & `merge_audit_summary.csv`.
  - `reports/figures/rain_event_{station}.png` (5 figures) & `rain_washout_multi_station_comparison.png`.
- **Key Rules Applied**: AA1 precipitation condition codes restricted strictly to `'1'` and `'5'` (passed automated & manual checks); `'C'` and `'S'` (suspect) completely excluded. Peak physical lag confirmed at 0 to +1 hour.

### Phase 2: Physical Baseline Characterization & Static Alarm Evaluation
- **Objective**: Quantify false alarm burden of current-practice fixed-threshold monitors under natural radon washout and environmental cycling.
- **Key Scripts**:
  - `src/characterize_baseline_cycles.py`: Analyzes diurnal radon cycle and filter advance step-drops.
  - `src/analyze_filter_cycles_multi_station.py`: Multi-station filter advance analysis.
  - `src/analyze_timestamp_lags.py`: Calendar-reindexed cross-correlation lag profile.
  - `src/analyze_rain_washout_distribution.py`: Rain surge amplitude distribution.
  - `src/evaluate_baseline.py` & `src/plot_baseline_timeline.py`: Evaluates static $3\sigma$ and $5\sigma$ thresholds.
- **Key Artifacts & Figures**:
  - `data/processed/baseline_threshold_evaluation.csv`, `lag_cross_correlation.csv`, `rain_washout_surge_distribution.csv`, `multi_station_filter_cycle_summary.csv`.
  - Figures: `baseline_alarm_rate_by_threshold.png`, `baseline_alarm_rain_coincidence.png`, `baseline_alarm_example_timeline.png`, `diurnal_radon_cycle.png`, `filter_cycle_sawtooth.png`.
- **Key Findings**: Standard $3\sigma$ alarm produces 52–78 alarms per station-year in humid climates; 62%–97% of all alarms coincide directly with rain washout. Raising threshold to $5\sigma$ leaves 12–32 alarms/yr, of which 73%–97% remain false alarms from rain.

### Phase 3: Synthetic Injection Design & Labeled Benchmark Catalog
- **Objective**: Generate physically grounded synthetic anthropogenic anomaly injections across realistic meteorological conditions, enforcing strict train/test disjointness and non-overlap.
- **Key Scripts**:
  - `src/calibrate_dose_coupling.py`: Calibrates gross CPM to dose-rate coupling.
  - `src/analyze_washout_spectrum.py`: Derives empirical multi-channel washout shares.
  - `src/synthetic_injection.py`: Injects 2,000 synthetic plumes into continuous background.
  - `src/audit_injection_catalog.py`: Verifies non-overlap buffer, calendar consistency, and split integrity.
  - `src/plot_synthetic_injections.py`: Plots shapes, nuclide spectra, and hard-case rain regimes.
- **Key Artifacts & Figures**:
  - `data/processed/synthetic_injection_catalog.csv`: 2,000 injections (1,000 train [2017–2022], 1,000 test [2023–2025]).
  - `data/processed/labeled_{station}_{split}.csv.gz`: 10 pre-injected continuous hourly datasets.
  - `data/processed/station_dry_baselines_train_only.json`: Frozen dry-baseline reference.
  - Figures: `synthetic_injection_shapes_and_nuclides.png`, `synthetic_injection_hard_case_rain.png`, `synthetic_injection_train_test_disjointness.png`.
- **Key Design Constraints**: 48-hour non-overlap buffer enforced; real NaI(Tl) channel shares for Cs-137, Co-60, I-131, Ir-192, Am-241; step, ramp, and Gaussian profiles; hard-case rain co-occurrence (150 train, 150 test).

### Phase 4: Feature Engineering & Machine Learning Detection Pipeline
- **Objective**: Engineer physical features (spectral channel ratios, temporal diffs, rolling stats, weather interactions) and train LightGBM classifier.
- **Key Scripts**:
  - `src/feature_engineering.py`: Computes 40+ physics-informed spatio-temporal features.
  - `src/train_and_evaluate_models.py`: Trains LightGBM and runs initial Leave-One-Station-Out (LOSO).
  - `src/plot_model_evaluation.py`: Generates ROC and feature importance figures.
- **Key Artifacts & Figures**:
  - `data/processed/model_feature_importance.csv`, `model_roc_curve_data.csv`, `model_operating_points_summary.csv`, `model_loso_evaluation_summary.csv`.
  - Figures: `model_detection_vs_false_alarms_roc.png`, `model_feature_importance.png`, `model_hard_case_timeline.png`, `model_stratified_performance.png`.

### Phase 5: Rigorous Evaluation Protocol, Multi-Seed Replication & Causal Benchmarking
- **Objective**: Multi-seed replication, continuous ROC evaluation on test split, matched-detection comparison at fixed false-alarm rates, causal validation threshold freezing, temperature/gain drift robustness, MLP neural network comparison.
- **Key Scripts**:
  - `src/train_and_evaluate_rigorous.py`: Core runner for 20-seed replication, continuous ROC, matched FAR detection, LOSO, gain drift, and template sensitivity.
  - `src/evaluate_phase5_protocol.py`: Statistical testing, block bootstrap uncertainty, and delay metrics.
  - `src/plot_rigorous_evaluation.py` & `src/plot_phase5_evaluation.py`: All Phase 5 visualizations.
- **Key Artifacts & Figures**:
  - `data/processed/rigorous_benchmark_summary.csv`: Comprehensive 4-tier benchmark summary.
  - `data/processed/rigorous_seed_level_results.csv` & `rigorous_seed_statistical_tests.csv`: 20-seed metrics ($p < 10^{-15}$).
  - `data/processed/rigorous_matched_detection_comparison.csv`: Matched detection at identical FAR.
  - `data/processed/rigorous_continuous_roc_all_tiers.csv`: Full ROC points across 131,520 test hours.
  - `data/processed/rigorous_loso_summary.csv` & `rigorous_loso_stratified_detection.csv`: LOSO generalization.
  - `data/processed/rigorous_gain_drift_evaluation.csv`: $\pm 10\%$ detector gain drift audit.
  - `data/processed/eval_mlp_calibration_comparison.csv`: LightGBM vs. Calibrated MLP comparison.
  - `data/processed/eval_benchmark_uncertainty_summary.csv`: Block bootstrap confidence bounds.
  - Figures: `rigorous_matched_roc_curves.png`, `rigorous_ablation_seeds.png`, `rigorous_loso_multistation.png`, `rigorous_detection_deadlines.png`, `rigorous_template_sensitivity.png`, `eval_bootstrap_uncertainty.png`, `eval_confusion_matrices.png`, `eval_detection_delay_distributions.png`.

---

## 3. How to Run the Pipeline in Order

All commands should be executed from the repository root using the project virtual environment:

```bash
# Activate environment
source .venv/bin/activate
# Or install dependencies if setting up clean:
# pip install -r requirements.txt
```

### Step 1: Phase 0 (Station Inventory & Pilot Selection)
```bash
python src/parse_stations.py
python src/download_pilot_data.py
python src/analyze_pilot_stations.py
```

### Step 2: Phase 1 (NOAA Weather Ingestion & Synchronous Merge)
```bash
python src/download_noaa_data.py
python src/audit_noaa_precipitation.py
python src/merge_radnet_weather.py
python src/plot_rain_events.py
```

### Step 3: Phase 2 (Baseline Characterization & Static Alarm Evaluation)
```bash
python src/characterize_baseline_cycles.py
python src/analyze_filter_cycles_multi_station.py
python src/analyze_timestamp_lags.py
python src/analyze_rain_washout_distribution.py
python src/evaluate_baseline.py
python src/plot_baseline_timeline.py
```

### Step 4: Phase 3 (Synthetic Injections & Dataset Generation)
```bash
python src/calibrate_dose_coupling.py
python src/analyze_washout_spectrum.py
python src/synthetic_injection.py
python src/audit_injection_catalog.py
python src/plot_synthetic_injections.py
```

### Step 5: Phase 4 (Feature Engineering & Baseline Model Training)
```bash
python src/train_and_evaluate_models.py
python src/plot_model_evaluation.py
```

### Step 6: Phase 5 (Rigorous Multi-Seed Evaluation & Visualization)
```bash
python src/train_and_evaluate_rigorous.py
python src/plot_rigorous_evaluation.py
python src/evaluate_phase5_protocol.py
python src/plot_phase5_evaluation.py
```

---

## 4. What Was Never Run End-to-End

To ensure complete transparency during the handoff, the following aspects were **never executed as a single automated end-to-end run**:

1. **No Monolithic Single-Command Master Runner**:
   - There is no single `run_all.sh` or automated DAG runner script that executes from `download_pilot_data.py` to `plot_rigorous_evaluation.py` in one invocation.
   - Each phase and script was executed incrementally by the agent at the terminal, validating outputs against gate requirements before proceeding.
2. **Network Downloads Decoupled from Downstream Pipeline**:
   - Raw data retrieval (`download_pilot_data.py` and `download_noaa_data.py`) requires live HTTP access to NOAA NCEI ISD and EPA RadNet REST endpoints.
   - All subsequent analysis and modeling scripts assume local existence of files in `data/raw/` and `data/processed/`.
3. **Phase 5 Execution Relies on Frozen Labeled Datasets**:
   - The 20-seed replication and rigorous benchmarking in `train_and_evaluate_rigorous.py` were run against the pre-generated `labeled_{station}_{split}.csv.gz` files produced in Phase 3. The synthetic injection generation step (`synthetic_injection.py`) was not re-executed inside the Phase 5 training loop.
4. **`review_notes/` Directory**:
   - A dedicated folder named `review_notes/` was never created on disk; review bundles and review text were managed directly in chat and as zip archives (`phase0_review.zip` through `phase5_review.zip`).

---

## 5. Guidance & Full Authority for MiMo (Lead Developer & Co-Boss)

1. **Full Ownership & Administrative Clearance**: MiMo has full administrative authority and developer leadership over the repository, code, pipelines, and execution. Gemini has officially stepped down and transferred all responsibilities.
2. **Review & Gate Alignment**: Claude functions as reviewer and technical co-advisor. Collaborate closely with Claude's rigorous feedback, especially on methodology and reporting.
3. **Data Freezing Policy**: `data/raw/` is strictly read-only. Injections in `synthetic_injection_catalog.csv` and labeled datasets are the frozen ground truth for benchmarks—any changes to benchmark splits or generation require explicit justification.
4. **Reproducibility**: All random seeds remain controlled via `configs/config.yaml`. Adhere to causal validation (threshold freezing on validation folds; zero test-set leakage).
5. **Tone & Objectivity**: Maintain neutral, rigorous environmental science phrasing throughout all documentation and reports.
