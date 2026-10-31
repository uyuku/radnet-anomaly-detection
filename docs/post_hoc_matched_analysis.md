# Post-Hoc Matched Performance Analysis (Theoretical Ceilings)

## Important Methodological Caveat
> [!WARNING]
> **Post-Hoc Nature:** The operating points in this document are derived directly from out-of-sample test scores. They represent **theoretical performance ceilings** under optimal retrospective threshold selection.
> In an actual operational deployment, operating thresholds **cannot** be chosen using future test scores without data snooping. Operational deployments must freeze thresholds beforehand using historical validation data (as presented in Table 1 and Table 2).

## 1. Post-Hoc Matched False Alarm Budgets (5, 10, 50 FA/station-year)

When all detectors are matched to the **exact same out-of-sample test false alarm budget**, we observe the true trade-off between baseline and machine learning architectures:

| Detector | Matched Test FA Budget | Realized Test FA/yr | Realized Test Det (%) [95% CI] |
| :--- | :---: | :---: | :---: |
| **Baseline: Rolling 7d 3-sigma (B1a)** | 5 yr$^{-1}$ | 5.59 [4.18, 7.14] | **2.0%** [0.5%, 4.0%] |
| **Baseline: Global Dry 3-sigma (B1b)** | 5 yr$^{-1}$ | 3.37 [2.04, 4.92] | **10.0%** [6.5%, 13.5%] |
| **Non-ML Radon Stripping (B4c)** | 5 yr$^{-1}$ | 6.49 [3.29, 10.64] | **89.0%** [85.0%, 93.0%] |
| **Tier 1: Gross Only GBDT** | 5 yr$^{-1}$ | 4.27 [2.95, 5.71] | **52.0%** [46.5%, 57.0%] |
| **Tier 2: Spectral GBDT** | 5 yr$^{-1}$ | 4.90 [2.80, 7.40] | **86.5%** [81.5%, 91.0%] |
| **Tier 3: Weather Fusion GBDT** | 5 yr$^{-1}$ | 5.10 [3.00, 7.80] | **91.5%** [87.5%, 95.0%] |
| **Tier 4: Physics-Motivated Nuclear GBDT** | 5 yr$^{-1}$ | 4.85 [2.75, 7.50] | **94.0%** [91.0%, 97.0%] |
| **Baseline: Rolling 7d 3-sigma (B1a)** | 10 yr$^{-1}$ | 11.18 [9.17, 13.32] | **5.0%** [2.5%, 8.0%] |
| **Baseline: Global Dry 3-sigma (B1b)** | 10 yr$^{-1}$ | 5.84 [4.15, 7.69] | **13.5%** [9.5%, 17.5%] |
| **Non-ML Radon Stripping (B4c)** | 10 yr$^{-1}$ | 9.80 [5.40, 15.20] | **91.5%** [87.5%, 95.0%] |
| **Tier 1: Gross Only GBDT** | 10 yr$^{-1}$ | 8.47 [6.24, 10.94] | **61.5%** [56.5%, 66.5%] |
| **Tier 2: Spectral GBDT** | 10 yr$^{-1}$ | 9.90 [5.50, 15.40] | **89.0%** [84.5%, 93.0%] |
| **Tier 3: Weather Fusion GBDT** | 10 yr$^{-1}$ | 10.10 [5.80, 15.60] | **93.5%** [90.0%, 96.5%] |
| **Tier 4: Physics-Motivated Nuclear GBDT** | 10 yr$^{-1}$ | 9.87 [5.10, 15.97] | **96.5%** [94.0%, 98.5%] |
| **Baseline: Rolling 7d 3-sigma (B1a)** | 50 yr$^{-1}$ | 41.85 [37.56, 46.13] | **27.5%** [22.0%, 33.0%] |
| **Baseline: Global Dry 3-sigma (B1b)** | 50 yr$^{-1}$ | 28.36 [23.50, 33.60] | **42.5%** [37.5%, 47.5%] |
| **Non-ML Radon Stripping (B4c)** | 50 yr$^{-1}$ | 36.17 [29.16, 44.35] | **95.5%** [93.0%, 98.0%] |
| **Tier 1: Gross Only GBDT** | 50 yr$^{-1}$ | 39.87 [34.09, 45.92] | **75.0%** [71.0%, 79.0%] |
| **Tier 2: Spectral GBDT** | 50 yr$^{-1}$ | 48.50 [38.00, 60.50] | **92.5%** [89.0%, 95.5%] |
| **Tier 3: Weather Fusion GBDT** | 50 yr$^{-1}$ | 49.20 [38.50, 61.00] | **96.0%** [93.5%, 98.0%] |
| **Tier 4: Physics-Motivated Nuclear GBDT** | 50 yr$^{-1}$ | 48.00 [37.50, 59.80] | **98.5%** [96.5%, 100.0%] |

## 2. Key Insights from Matched Analysis
1. **At Tight False Alarm Budgets (5 FA/yr):**
   - Non-ML geometric radon projection (**B4c**) achieves **89.0% detection** at 6.49 FA/yr.
   - **Tier 4 GBDT** achieves **94.0% detection** at 4.85 FA/yr.
   - In contrast, operational gross-count rules (**Rolling 3-sigma**) achieve only **2.0% detection** when constrained to 5 FA/yr.
2. **At Moderate False Alarm Budgets (10 FA/yr):**
   - Non-ML **B4c** achieves **91.5% detection** at 9.80 FA/yr.
   - **Tier 4 GBDT** achieves **96.5% detection** at 9.87 FA/yr.
   - **Tier 3 (Weather Fusion)** achieves **93.5% detection** at 10.10 FA/yr.
3. **Conclusion:** Non-ML local spectral projection accounts for the overwhelming majority of false-alarm reduction (raising detection at 10 FA/yr from 5.0% to 91.5%), while supervised physics-motivated GBDT training adds the final +5.0 pp gain (91.5% -> 96.5%).
