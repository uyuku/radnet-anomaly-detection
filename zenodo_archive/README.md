# Zenodo Archive: RadNet Anomaly Detection Benchmark

## Title
Curated Benchmark Datasets, Monte Carlo Spectrometry Templates, and Machine Learning Evaluation Results for RadNet Ambient Radiological Anomaly Detection

## Author Placeholder
[Author Names / Affiliations Placeholder]  
DOI: 10.5281/zenodo.XXXXXXX

---

## Overview
This archive contains the complete benchmark data files, detector response templates, injection catalogs, and evaluation summaries supporting the manuscript "Physics-Motivated Environmental Radiation Anomaly Detection via Multi-Scale Weather Fusion and Multi-Channel Gamma Spectrometry".

All code to reproduce these datasets is available at: `https://github.com/TODO_AUTHOR/radnet-anomaly-detection` (Version 1.0.0).

---

## Archive Structure and Column Dictionaries

### 1. `synthetic_injection_catalog.csv` & `extended_synthetic_injection_catalog.csv`
Catalog of synthetic plume injection events across validation (2021-2022) and test (2023-2025) splits.
- `injection_id`: Unique event identifier (e.g. `INJ-0001`).
- `station_id`: RadNet station code (`al_birmingham`, `hi_honolulu`, etc.).
- `split`: Split assignment (`validation` or `test`).
- `start_utc` / `end_utc`: Event start and end timestamps (UTC).
- `duration_hours`: Plume transit duration ($3$ to $48	ext{ h}$).
- `peak_cpm`: Peak net excess count rate ($25$ to $450	ext{ CPM}$).
- `scenario`: Plume profile shape (`unfiltered_gaussian`, `filtered_sawtooth`, `filtered_boxcar`, `stagnant_inversion`).
- `frac_cs137`, `frac_cs134`, `frac_i131`, `frac_te132`: Nuclide release fractions.
- `family`: Monte Carlo template family (`Family A: Standard High-Yield Cs-137` or `Family B: Low-Yield Short-Lived Fission`).

### 2. `realistic_monte_carlo_templates.csv`
Full 8-channel detector response templates ($R_{02}$ to $R_{09}$) derived from Geant4 / MCNP6 photon transport simulations of 2"x2" NaI(Tl) detectors in hemispherical cloud and surface deposition geometries.

### 3. `eval_benchmark_uncertainty_summary.csv`
Primary model benchmark comparison table:
- `model_name`: Architecture name (Tier 1-4, Baselines B1-B8).
- `realized_detection_pct`: Empirical test detection percentage.
- `detection_ci_95`: 95% bootstrap confidence interval.
- `false_alarms_per_station_year`: False alarm rate on clean background.
- `false_alarms_ci_95`: 95% station-month block bootstrap CI.
- `rain_coincident_alarm_pct`: Percentage of clean-background alarms during rain.
- `median_detection_delay_hours`: Time from plume onset to first alarm.

### 4. `extended_loso_11station_summary.csv` & `extended_loco_climate_summary.csv`
Leave-One-Station-Out (LOSO) and Leave-One-Climate-Out (LOCO) generalization evaluations across 11 stations spanning 5 Köppen climate zones.

### 5. `docs/hand_labeled_test_alarms.csv`
Stratified random sample of $N=100$ clean-background alarms hand-audited across meteorological and operational categories with Wilson score confidence intervals.

---

## Data Licenses
- EPA RadNet and NOAA ISD data: U.S. Public Domain (17 U.S.C. § 105).
- Derived Benchmark Datasets: Creative Commons Attribution 4.0 International (CC-BY-4.0).
- Software: MIT License.
