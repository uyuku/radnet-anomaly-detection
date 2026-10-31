# Phase 1 Summary: Claim Audit (`docs/claims_audit.csv` & `recompute_claims.py`)

## Overview
Audited every quantitative, methodological, physical, citation, and scope claim in `README.md` and `report/interim_report.tex` against `src/*.py` and `data/processed/*.csv`. Built `recompute_claims.py` to recompute every dataset, washout, filter-cycle, baseline, and benchmark metric from the committed data and verify zero discrepancies (exit code `0`).

## Key Findings & Corrections
1. **Stale Benchmark Table Numbers Fixed (`README.md` & `report/interim_report.tex` Table 2)**:
   - Several rows in `README.md` and `report/interim_report.tex` had stale values that contradicted `data/processed/rigorous_benchmark_summary.csv` and `data/processed/eval_benchmark_uncertainty_summary.csv`. All values now match the committed CSV outputs exactly.
2. **Baseline False-Alarm Disentanglement**:
   - Separated the 2023–2025 held-out test split rolling 7-day $3\sigma$ rates (`82.62` pooled; `66.37` San Diego, `82.69` Dallas, `86.22` DC, `86.54` Birmingham, `96.17` Tampa) and global dry $3\sigma$ rates (`52.86` pooled; `9.44` to `78.64` per station) from the 2017–2025 `baseline_threshold_evaluation.csv` `rolling_cpm_3s` rates (`6.59` Tampa, `20.91` San Diego, `52.98` DC, `53.57` Birmingham, `62.91` Dallas).
3. **Methodological & Physical Claims Aligned to Code**:
   - Replaced `"physics-informed"` with `"physics-motivated"` across prose where the model is a GBDT on domain features rather than a physics-constrained loss.
   - Corrected LightGBM `min_child_samples` in `report/interim_report.tex` from `20` to `50` (`src/train_and_evaluate_rigorous.py:136`).
   - Replaced the contradicted `bi-weekly` filter replacement claim and $Q = 1.0\text{ m}^3/\text{min}$ flow-rate claim in `report/interim_report.tex` with the empirical 2021–2024 dry-spell filter intervals (`2.92–4.54` days median, `3.97–5.43` days mean, `342.6–716.9` CPM step-drops) and normalized discrete accumulation formula (`src/nuclear_physics.py:252`).
   - Removed the unsupported claim that `NAI_INTRINSIC_EFFICIENCY` was calibrated via Monte Carlo transfer factors from Knoll (2010) and documented that it is an assumed 8-element vector (`[0.92, 0.68, 0.48, 0.38, 0.32, 0.26, 0.21, 0.18]`).
   - Removed the unsupported claim that synthetic plumes are calibrated to atmospheric dispersion physics with Gaussian puff profiles; documented the actual profiles (`linear_ramp`, `step`, `sigmoidal`, `exponential`), raw CPM bands (`250–600`, `700–1200`, `1400–2500` CPM), hand-assumed uniform/Dirichlet channel templates (`sample_operational_spectrum`), and heuristic ODE parameters ($\beta = 0.65$, $\tau_{\text{eff}} = 0.65\text{ h}$).
   - Replaced nonexistent feature names (`Ratio_radon`, `Ratio_fission`, `12h` precip, `wind speed`, `saturation vapor pressure deficit`, `stripped_energy_weighted_dose_anomaly`) with the exact feature names in `src/feature_engineering.py`.
   - Added explicit scope statements in the first paragraph of `README.md` and the Abstract of `report/interim_report.tex`, and tagged deferred items with  through .

## Before / After Table of Corrected Numbers

| Metric / Claim | Before (Stale / Contradicted) | After (Verified from CSVs / Code) | Source File |
| :--- | :---: | :---: | :--- |
| Rain Surge Summary | `+6.9% (+7.2% active)` | `+6.9%` median (`+12.05%` mean), `+41.4%` p90, `+97.9%` p99, `+196.7%` max | `rain_washout_surge_distribution.csv` |
| Global Dry $3\sigma$ Det / Delay | `56.50%` / `9.0 hrs` | `57.00%` / `13.5 hrs` | `eval_benchmark_uncertainty_summary.csv` |
| Conservative $5\sigma$ Delay | `8.5 hrs` | `8.0 hrs` | `eval_benchmark_uncertainty_summary.csv` |
| Tier 1 (Gross, 95%) FA/yr | `169.36 ± 23.53` | `102.83 ± 9.30` | `rigorous_benchmark_summary.csv` |
| Tier 1 (Gross, 95%) Rain FA / Det / Delay | `17.3%` / `92.22 ± 0.72%` / `6.7 hrs` | `25.8%` / `89.98 ± 0.94%` / `6.1 hrs` | `rigorous_benchmark_summary.csv` |
| Tier 1b (Gross+Weather, 95%) FA/yr | `38.64 ± 14.37` | `10.39 ± 2.62` | `rigorous_benchmark_summary.csv` |
| Tier 1b (Gross+Weather, 95%) Det / Delay | `87.92 ± 2.74%` / `9.1 hrs` | `79.62 ± 1.47%` / `8.1 hrs` | `rigorous_benchmark_summary.csv` |
| Tier 2 / Tier 3 / Tier 4 (95%) Rain FA | `1.9%` / `0.2%` / `0.0%` | `5.5%` / `0.3%` / `1.2%` | `rigorous_benchmark_summary.csv` |
| Tier 4 (90% Target) Rain FA / Delay | `0.0%` / `4.5 hrs` | `1.1%` / `4.4 hrs` | `rigorous_benchmark_summary.csv` |
| Tier 3 MLP FA/yr / Rain FA / Det / Delay | `60.75 ± 22.86` / `3.8%` / `90.30 ± 2.75%` / `3.4 hrs` | `57.10 ± 23.67` / `3.5%` / `89.62 ± 2.93%` / `3.6 hrs` | `rigorous_benchmark_summary.csv` |
| Filter Replacement Cadence (`interim_report.tex`) | `bi-weekly cycle` | `2.92–4.54 days` median (`3.97–5.43 days` mean) | `multi_station_filter_cycle_summary.csv` |
| LightGBM `min_child_samples` (`interim_report.tex`) | `20` | `50` | `src/train_and_evaluate_rigorous.py:136` |
