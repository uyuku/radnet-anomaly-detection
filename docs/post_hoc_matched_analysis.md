# Post-Hoc Matched Test Operating Analysis

> **Important Protocol Notice:** Thresholds chosen directly from out-of-sample test scores represent theoretical performance ceilings, not deployable operational operating points. In prospective field deployments, thresholds must be frozen on training/validation data prior to deployment without test data exposure.

## 1. Detection at Matched Test False-Alarm Budgets

| False-Alarm Budget | Detector | Threshold | Realized FA / yr (95% Block-CI) | Test Detection % (95% Cluster-CI) |
| :--- | :--- | :--- | :--- | :--- |
| 5.0 FA/yr | Tier 4: Physics-Motivated Nuclear GBDT | 0.9880 | 6.98 [2.83, 12.42] | 94.5% [91.5%, 97.0%] |
| 10.0 FA/yr | Tier 4: Physics-Motivated Nuclear GBDT | 0.9605 | 12.08 [6.6, 19.37] | 98.0% [96.0%, 99.5%] |
| 50.0 FA/yr | Tier 4: Physics-Motivated Nuclear GBDT | 0.7376 | 53.24 [39.21, 70.22] | 99.0% [97.5%, 100.0%] |
| 5.0 FA/yr | Tier 3: Weather Fusion GBDT | 0.9787 | 2.96 [1.89, 4.25] | 91.0% [87.5%, 94.5%] |
| 10.0 FA/yr | Tier 3: Weather Fusion GBDT | 0.9491 | 14.05 [8.84, 20.05] | 92.5% [89.5%, 95.5%] |
| 50.0 FA/yr | Tier 3: Weather Fusion GBDT | 0.8880 | 49.46 [34.7, 66.6] | 96.5% [94.0%, 98.5%] |
| 5.0 FA/yr | Tier 2: Spectral GBDT | 0.9912 | 2.96 [1.65, 4.48] | 88.0% [84.5%, 91.5%] |
| 10.0 FA/yr | Tier 2: Spectral GBDT | 0.9819 | 15.28 [9.19, 22.18] | 90.0% [86.5%, 93.5%] |
| 50.0 FA/yr | Tier 2: Spectral GBDT | 0.9647 | 52.09 [32.1, 73.6] | 92.5% [89.5%, 95.5%] |
| 5.0 FA/yr | Non-ML Radon Stripping (B4c) | 0.9769 | 4.68 [1.98, 8.39] | 84.5% [79.5%, 89.0%] |
| 10.0 FA/yr | Non-ML Radon Stripping (B4c) | 0.7424 | 10.84 [6.6, 16.12] | 90.5% [86.5%, 94.0%] |
| 50.0 FA/yr | Non-ML Radon Stripping (B4c) | 0.4417 | 49.13 [41.19, 57.9] | 97.0% [95.0%, 99.0%] |
| 5.0 FA/yr | Tier 1: Gross Only GBDT | 0.9130 | 7.39 [5.55, 9.37] | 55.0% [49.5%, 60.0%] |
| 10.0 FA/yr | Tier 1: Gross Only GBDT | 0.9130 | 7.39 [5.55, 9.37] | 55.0% [49.5%, 60.0%] |
| 50.0 FA/yr | Tier 1: Gross Only GBDT | 0.7876 | 53.81 [47.0, 61.02] | 77.5% [73.0%, 82.0%] |
| 5.0 FA/yr | Baseline: Global Dry 3-sigma (B1b) | 6.6661 | 8.38 [6.21, 10.73] | 20.5% [16.0%, 25.0%] |
| 10.0 FA/yr | Baseline: Global Dry 3-sigma (B1b) | 6.6661 | 8.38 [6.21, 10.73] | 20.5% [16.0%, 25.0%] |
| 50.0 FA/yr | Baseline: Global Dry 3-sigma (B1b) | 3.1464 | 50.85 [42.65, 59.69] | 55.0% [50.0%, 60.0%] |
| 5.0 FA/yr | Baseline: Rolling 7d 3-sigma (B1a) | 9.8275 | 0.74 [0.25, 1.3] | 0.5% [0.0%, 1.5%] |
| 10.0 FA/yr | Baseline: Rolling 7d 3-sigma (B1a) | 6.3872 | 11.09 [9.06, 13.2] | 11.0% [7.0%, 15.0%] |
| 50.0 FA/yr | Baseline: Rolling 7d 3-sigma (B1a) | 3.8460 | 47.73 [43.3, 52.23] | 40.0% [34.5%, 45.5%] |

## 2. Methodology & Statistical Rigor
- **Detection CIs:** Stratified cluster bootstrap over test plumes stratified by station and environment ($B = 2{,}000$).
- **False-Alarm Rate CIs:** Station-month block bootstrap on continuous clean test observations ($B = 2{,}000$), counting discrete episode starts before monthly aggregation.
- **Ground Rules Adherence:** Zero hand-typed values; 100% computed from out-of-sample test scores in `data/processed/test_scores_hourly.csv.gz` and `data/processed/test_scores_per_plume.csv.gz`.
