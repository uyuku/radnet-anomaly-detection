"""
Prepares a clean, Zenodo-ready archive folder containing:
- DATA_MANIFEST.csv
- Core benchmark tables, catalogs, MC templates, and per-plume outputs
- Comprehensive README with dataset schemas and column dictionaries
- docs/zenodo_upload_instructions.md
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ZENODO_DIR = REPO_ROOT / "zenodo_archive"


KEY_FILES = [
    # Catalogs and Templates
    "data/processed/synthetic_injection_catalog.csv",
    "data/processed/extended_synthetic_injection_catalog.csv",
    "data/processed/response_templates.csv",
    "data/processed/benchmark_holdout_templates.csv",
    "data/processed/station_dry_baselines_train_only.json",
    "data/processed/extended_station_dry_baselines_2017_2020.json",
    # Benchmark Evaluation Tables
    "data/processed/eval_benchmark_uncertainty_summary.csv",
    "data/processed/eval_bootstrap_distributions.csv",
    "data/processed/eval_confusion_matrices.csv",
    "data/processed/eval_detection_delay_summary.csv",
    "data/processed/benchmark_fa_budget_operating_points.csv",
    "data/processed/threshold_calibration_fa_budget_comparison.csv",
    "data/processed/baseline_per_plume.csv",
    "data/processed/baseline_per_month.csv",
    "data/processed/tier_per_plume.csv",
    "data/processed/tier_per_month.csv",
    "data/processed/generalization_loso_per_plume.csv",
    "data/processed/generalization_loso_per_month.csv",
    # Extended Stations and Generalization
    "data/processed/extended_loso_11station_summary.csv",
    "data/processed/extended_loco_climate_summary.csv",
    "data/processed/loso_normalization_variants.csv",
    "data/processed/loso_local_calibration_results.csv",
    "data/processed/temporal_generalization_by_year.csv",
    "data/processed/all_stations_yearly_baseline_drift.csv",
    # Real Event and Alarm Adjudication
    "data/processed/real_event_fukushima_station_summary.csv",
    "data/processed/hand_labeled_test_alarms_summary.csv",
    "docs/hand_labeled_test_alarms.csv",
    "data/DATA_MANIFEST.csv",
]


README_CONTENT = """# Zenodo Archive: RadNet Anomaly Detection Benchmark

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
- `duration_hours`: Plume transit duration ($3$ to $48\text{ h}$).
- `peak_cpm`: Peak net excess count rate ($25$ to $450\text{ CPM}$).
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
"""


def main():
    print(f"Preparing Zenodo archive directory at {ZENODO_DIR}...")
    ZENODO_DIR.mkdir(parents=True, exist_ok=True)
    (ZENODO_DIR / "data").mkdir(parents=True, exist_ok=True)
    (ZENODO_DIR / "docs").mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for rel_path_str in KEY_FILES:
        src = REPO_ROOT / rel_path_str
        if src.exists():
            dst = ZENODO_DIR / rel_path_str
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_count += 1
        else:
            print(f"  Warning: {rel_path_str} not found on disk")

    # Write Zenodo README
    with open(ZENODO_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(README_CONTENT.strip() + "\n")

    print(f"Zenodo archive ready with {copied_count} files in {ZENODO_DIR}.")


if __name__ == "__main__":
    main()
