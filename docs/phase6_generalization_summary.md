# Phase 6 Summary: Spatial, Climatic & Temporal Generalization and Tampa LOSO Diagnosis

## 1. Executive Summary
Phase 6 replaced the speculative manuscript claim that *"station-specific threshold calibration is operationally necessary"* with a direct empirical investigation across **Parts A–F**:
1. **Part A (Tampa LOSO Failure Diagnosis & Gain-Drift Counterfactuals)**: Extracted every 2023–2025 false-alarm episode on held-out Tampa (`fl_tampa`, trained on the 4 remaining pilot stations) with exact Tree SHAP attributions (`data/processed/loso_tampa_false_alarms.csv`, `data/processed/loso_tampa_episode_category_summary.csv`), plotted 6 representative episodes (`reports/figures/tampa_loso_diagnostic_episodes.png`), compared 8-channel energy shares across all 11 stations (`data/processed/tampa_channel_share_comparison.csv`), audited raw data quality and 2017–2025 yearly spectral drift (`data/processed/tampa_raw_data_quality_audit.csv`, `data/processed/all_stations_yearly_baseline_drift.csv`), and ran direct counterfactual interventions (`data/processed/tampa_gain_drift_hypothesis_test.csv`).
2. **Part B (Direct Local-Calibration Experiment)**: Tested calibrating the held-out station using $N \in \{0, 3, 6, 12\}$ months of its own pre-test clean background (`data/processed/loso_local_calibration_results.csv`).
3. **Part C (Feature Normalization Variants)**: Compared `global_baseline`, `local_168h_baseline`, and `station_standardized` under 5-station pilot LOSO (`data/processed/loso_normalization_variants.csv`).
4. **Part D (Year-by-Year Temporal Generalization)**: Trained once on 2017–2020 and reported out-of-sample metrics separately for 2021, 2022, 2023, 2024, and 2025 (`data/processed/temporal_generalization_by_year.csv`, `data/processed/temporal_generalization_by_station_year.csv`).
5. **Part E (Pre-Registered 11-Station LOSO & 7-Climate LOCO)**: Pre-registered selection rules in `docs/station_selection.md` prior to downloading 6 additional RadNet/NOAA station pairs (`az_phoenix`, `ne_omaha`, `hi_honolulu`, `ak_anchorage`, `la_baton_rouge`, `vt_burlington`; `data/processed/extended_station_coverage_audit.csv`), evaluated 11-station LOSO and 7-climate LOCO (`data/processed/extended_loso_11station_summary.csv`, `data/processed/extended_loco_climate_summary.csv`), and diagnosed every high-FA held-out station (`data/processed/loso_high_fa_stations_diagnosis.csv`).
6. **Part F (Per-Plume & Per-Month Outputs)**: Exported `data/processed/generalization_loso_per_plume.csv` (`3,080` rows) and `data/processed/generalization_loso_per_month.csv` (`2,394` rows) so all LOSO/LOCO uncertainty intervals can be computed directly via `src/uncertainty.py`.

---

## 2. Part A: Why Held-Out Tampa Fails Under Static Spectral Features

### 2.1 Spectral Offset & Multi-Year Calibration Drift on Tampa
Across all 11 stations (`data/processed/tampa_channel_share_comparison.csv`), 9 continental stations exhibit clean 2023–2025 dry background `share_r02` in **`0.4592–0.5086`** and `share_r05` in **`0.0426–0.0488`**. By contrast:
- **Tampa, FL (`fl_tampa`)** has 2023–2025 clean `share_r02 = 0.3376 ± 0.0741`, `share_r05 = 0.0610 ± 0.0075`, `share_r07 = 0.0624 ± 0.0067`, and `ratio_r05_r03 = 0.1813 ± 0.0072` (compared to `0.1495–0.1632` on the 4 training pilot stations and `0.2311 ± 0.1165` during injected fission plumes).
- **Yearly Drift Audit (`data/processed/all_stations_yearly_baseline_drift.csv`)**: Tampa's clean `share_r05_mean` drifted monotonically upward from `0.0430` (2017) and `0.0440` (2018) to `0.0518` (2020–2022) and jumped to **`0.0655` in 2023** and **`0.0618` in 2024**, while its `share_r02_mean` dropped from `0.4596` (2017) to **`0.2968` (2023)**. That shift moves Tampa's clean spectrum directly along the photopeak ratio direction learned from $^{137}\text{Cs}$/$^{60}\text{Co}$ injections on the other four stations.

### 2.2 Episode Categorization & Exact Tree SHAP Attribution
On held-out Tampa (`data/processed/loso_tampa_episode_category_summary.csv`):
- **Tier 3 (Weather Fusion)**: Produces **`1,751` episodes (`786.86` FA/yr, exact Poisson 95% CI `[750.43, 824.60]`)**. The modal top-1 SHAP feature across categories is static **`share_r07_roll_6h`**, with `48.26%` of episodes classified as `baseline or gain drift`, `18.62%` `post-rain`, `12.16%` `rain-coincident`, `11.88%` `missing-data adjacent`, and `9.02%` `filter-change adjacent`.
- **Tier 4 (Physics-Informed Nuclear)**: Reduces held-out Tampa false alarms by **84.3%** to **`274` episodes (`123.13` FA/yr, exact Poisson 95% CI `[108.98, 138.61]`, `100.0%` detection)**, but remains operationally unacceptable uncalibrated because Tier 4 retains Tier 3's raw static spectral ratios (`ratio_r05_r03` is the modal top-1 SHAP driver for `37.59%` `filter-change adjacent`, `29.20%` `baseline or gain drift`, and `16.06%` `post-rain` episodes).

### 2.3 Direct Counterfactual Hypothesis Test (`data/processed/tampa_gain_drift_hypothesis_test.csv`)
| Detector | Intervention on Held-Out Tampa (2023–2025) | Test FA/yr (Exact Poisson 95% CI) | Test Detection % (Exact Clopper–Pearson 95% CI) |
| :--- | :--- | :---: | :---: |
| **Tier 3 (Weather Fusion)** | 1. Unmodified Held-Out Tampa | `786.86` `[750.43, 824.60]` | `67.5%` `[50.87%, 81.43%]` |
| **Tier 3 (Weather Fusion)** | 2. Align static spectral shares/ratios using 2017–2020 dry stats | `204.02` `[185.68, 223.68]` | `100.0%` `[91.19%, 100.0%]` |
| **Tier 4 (Physics-Informed Nuclear)** | 1. Unmodified Held-Out Tampa | `123.13` `[108.98, 138.61]` | `100.0%` `[91.19%, 100.0%]` |
| **Tier 4 (Physics-Informed Nuclear)** | 2. Align static spectral shares/ratios using 2017–2020 dry stats | `23.37` `[17.45, 30.64]` | `100.0%` `[91.19%, 100.0%]` |
| **Tier 4 (Physics-Informed Nuclear)** | 3. Align using 2022 pre-test clean stats (accounting for 2017–2022 drift) | `15.28` `[10.58, 21.35]` | `97.5%` `[86.84%, 99.94%]` |

---

## 3. Part B & Part C: Local Calibration vs. Causal 168-Hour Local Normalization

### 3.1 Normalization Variants on 5-Station Pilot LOSO (`data/processed/loso_normalization_variants.csv`)
Replacing raw static spectral shares/ratios and global-dry features with **causal 168-hour rolling local deviations (`local_168h_baseline`, requiring zero historical station calibration beyond a 7-day rolling window)** or **`station_standardized` (using pre-test clean station stats)** dramatically improves cross-station transfer on held-out Tampa while preserving `0.00` FA/yr on Birmingham, Washington DC, and Dallas:

| Normalization Variant | Tier | Birmingham FA/yr (Det%) | Washington DC FA/yr (Det%) | San Diego FA/yr (Det%) | Dallas FA/yr (Det%) | Held-Out Tampa FA/yr `[95% CI]` (Det%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `global_baseline` | Tier 3 | `0.00` (`87.5%`) | `1.97` (`100.0%`) | `0.00` (`57.5%`) | `1.13` (`95.0%`) | `786.86` `[750.43, 824.60]` (`67.5%`) |
| `global_baseline` | Tier 4 | `0.00` (`95.0%`) | `1.18` (`100.0%`) | `10.18` (`80.0%`) | `0.00` (`82.5%`) | `123.13` `[108.98, 138.61]` (`100.0%`) |
| `local_168h_baseline` | Tier 3 | `0.00` (`95.0%`) | `0.39` (`95.0%`) | `3.51` (`75.0%`) | `0.00` (`95.0%`) | **`25.17` `[19.01, 32.68]` (`82.5%`)** |
| `local_168h_baseline` | Tier 4 | `0.00` (`95.0%`) | `0.00` (`90.0%`) | `0.00` (`75.0%`) | `0.00` (`95.0%`) | **`13.48` `[9.10, 19.25]` (`95.0%`)** |
| `station_standardized` | Tier 3 | `0.00` (`95.0%`) | `0.00` (`95.0%`) | `27.74` (`97.5%`) | `0.00` (`87.5%`) | **`0.45` `[0.01, 2.50]` (`90.0%`)** |
| `station_standardized` | Tier 4 | `0.00` (`100.0%`) | `0.00` (`95.0%`) | `13.69` (`92.5%`) | `0.00` (`72.5%`) | **`0.90` `[0.11, 3.25]` (`92.5%`)** |

### 3.2 Direct Local-Calibration Experiment (`data/processed/loso_local_calibration_results.csv`)
On held-out Tampa (`fl_tampa`), calibrating with $N \in \{3, 6, 12\}$ months of trailing pre-test (2022) clean background under Tier 4 yields:
- **$N = 0$ months (Zero-Shot Uncalibrated)**: `123.13` FA/yr `[108.98, 138.61]` at `100.0%` detection.
- **$N = 3$ months (Trailing Pre-Test Clean, Zero-FA Ceiling)**: `10.34` FA/yr `[6.55, 15.51]` at `95.0%` detection (`[83.08%, 99.39%]`).
- **$N = 6$ months (Trailing Pre-Test Clean, Zero-FA Ceiling)**: `1.35` FA/yr `[0.28, 3.94]` at `87.5%` detection (`[73.20%, 95.81%]`).
- **$N = 12$ months (Trailing Pre-Test Clean, Zero-FA Ceiling)**: **`0.00` FA/yr `[0.00, 1.66]`** at `82.5%` detection (`[67.22%, 92.66%]`), or **`4.94` FA/yr `[2.47, 8.84]`** at **`97.5%` detection** (`[86.84%, 99.94%]`) under a 5 FA/yr local calibration budget.
- Using initial 2017 clean months ($N = 12$) on Tampa reduces Tier 4 false alarms from `123.13` to `7.19` FA/yr (`95.0%` detection), demonstrating that trailing pre-test calibration is more effective than 6-year-old calibration when a detector undergoes multi-year photomultiplier gain drift.

---

## 4. Part D: Year-by-Year Temporal Generalization (`data/processed/temporal_generalization_by_year.csv`)
Trained once on 2017–2020 across the 5 pilot stations (`Val_Target_95pct`):

| Detector | 2021 (Val) FA/yr (Det%) | 2022 (Val) FA/yr (Det%) | 2023 (Test) FA/yr `[95% CI]` (Det%) | 2024 (Test) FA/yr `[95% CI]` (Det%) | 2025 (Test) FA/yr `[95% CI]` (Det%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B4c (Non-ML Radon Proj + Peak Z)** | `0.00` (`97.3%`) | `2.02` (`93.2%`) | `0.91` `[0.25, 2.33]` (`86.6%`) | `0.90` `[0.24, 2.29]` (`76.3%`) | `6.04` `[3.69, 9.33]` (`73.8%`) |
| **Tier 2 (Spectral)** | `0.00` (`97.3%`) | `0.00` (`93.2%`) | `0.91` `[0.25, 2.33]` (`91.5%`) | `1.12` `[0.36, 2.61]` (`85.5%`) | `1.81` `[0.67, 3.95]` (`78.6%`) |
| **Tier 3 (Weather Fusion)** | `0.00` (`100.0%`) | `0.00` (`93.2%`) | `8.43` `[5.94, 11.62]` (`96.3%`) | `14.55` `[11.23, 18.55]` (`93.4%`) | `8.76` `[5.87, 12.59]` (`85.7%`) |
| **Tier 4 (Physics-Informed Nuclear)** | `0.00` (`100.0%`) | `0.00` (`100.0%`) | `3.65` `[2.08, 5.92]` (`98.8%`) | `6.72` `[4.53, 9.59]` (`94.7%`) | `17.53` `[13.31, 22.66]` (`95.2%`) |

Per-station-year breakdown (`data/processed/temporal_generalization_by_station_year.csv`) reveals that Tier 4's false alarms in 2023 (`16.75` FA/yr) and 2024 (`34.00` FA/yr) are concentrated almost exclusively on `fl_tampa` (where `share_r05` surged to `0.0655` in 2023–2024), whereas in 2025 `ca_san_diego` accounts for `56` of `58` episodes (`57.98` FA/yr) due to 2025 filter-change/baseline transients while Birmingham, Dallas, and Tampa record `0.00` FA/yr in 2025.

---

## 5. Part E: Pre-Registered 11-Station LOSO & 7-Climate LOCO (`data/processed/extended_loso_11station_summary.csv`)
Across all 11 stations (`5` pilot + `6` pre-registered extended stations, none dropped after seeing results):
- **Uncalibrated Zero-Shot (`global_baseline`)** suffers severe false-alarm spikes on stations whose static terrestrial background spectrum differs from the continental training pool — specifically **`fl_tampa`** (`149.64` FA/yr on Tier 4; `774.28` FA/yr on Tier 3) and volcanic basaltic **`hi_honolulu`** (`796.51` FA/yr on Tier 4, driven by `stripped_excess_r07_z` on `67.68%` of episodes because Honolulu's clean `share_r07` is `0.0716` vs. `0.030–0.046` on continental stations).
- **Zero-Shot Causal 168-Hour Local Normalization (`ZeroShot_Local168h_Normalized`)** — which requires **no** historical target-station data beyond a 7-day rolling window — reduces Tier 4 false alarms across all 11 held-out stations into the **`0.00–27.86` FA/yr** range:
  - `al_birmingham`: `2.51` FA/yr (`97.5%` det)
  - `dc_washington`: `0.39` FA/yr (`97.5%` det)
  - `ca_san_diego`: `12.99` FA/yr (`97.5%` det)
  - `tx_dallas`: `0.00` FA/yr (`95.0%` det)
  - `fl_tampa`: `27.86` FA/yr (`97.5%` det; down from `149.64` FA/yr uncalibrated, and `1.35` FA/yr with 12m local threshold calibration)
  - `az_phoenix`: `0.00` FA/yr (`42.5%` det)
  - `ne_omaha`: `1.07` FA/yr (`50.0%` det)
  - `hi_honolulu`: `13.17` FA/yr (`82.5%` det; down **98.3%** from `796.51` FA/yr uncalibrated!)
  - `ak_anchorage`: `0.39` FA/yr (`75.0%` det; down from `24.36` FA/yr uncalibrated)
  - `la_baton_rouge`: `6.29` FA/yr (`65.0%` det; down from `39.52` FA/yr uncalibrated)
  - `vt_burlington`: `0.78` FA/yr (`70.0%` det; down from `16.41` FA/yr uncalibrated)
- **Non-ML B4c (`ZeroShot_Local168h_Intrinsic`)** achieves `0.00–7.02` FA/yr on 9 of 11 stations (`0.00` Birmingham/Dallas, `0.39` DC, `0.78` Burlington, `0.90` Baton Rouge, `1.35` Tampa, `2.36` Anchorage, `2.93` Honolulu, `7.02` San Diego), with higher FA only on arid/continental stations (`22.09` Omaha, `66.63` Phoenix) where lower validation thresholds (`0.50–0.60`) are selected when that station is held out.
