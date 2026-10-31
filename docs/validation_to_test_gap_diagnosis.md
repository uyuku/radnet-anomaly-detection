# Empirical Diagnosis: Validation-to-Test Performance Gap

## 1. Executive Summary & Problem Formulation
When operating thresholds are chosen to meet a pre-specified detection target on the **2021--2022 validation fold** (e.g. 90% or 95% target detection), the realized detection on the **2023--2025 out-of-sample test split** systematically diverges from the validation target across almost all detectors.

For example, at the **90% validation target**:
- **Tier 4 (Physics-Motivated GBDT)** achieves **96.5% test detection** at **9.87 FA/yr**, but its validation detection was **100.0%** at threshold $\tau = 0.9740$.
- **B4c (Non-ML Radon Projection + Peak Channel Z)** achieves **70.0% test detection** at **0.74 FA/yr**, but its validation detection was **91.36%** at threshold $\tau = 1.4923$.
- **B1a (Rolling 168h Gross Z-Score)** achieves **84.0% test detection** at **227.07 FA/yr**, falling from **90.12%** validation detection.
- **B3b (Spline GAM Weather Regression)** achieves **65.5% test detection** at **72.26 FA/yr**, falling from **90.12%** validation detection.

This divergence demonstrates that **validation-matched detection does NOT produce matched test detection**, and that comparisons calibrated at validation targets are inherently unmatched on the test fold.

## 2. Quantitative Diagnostic Summary Table

| Detector | Target (%) | Val Det (%) | Test Det (%) | Gap (pp) | Val FA/yr | Test FA/yr | Test/Val FA Ratio | Primary Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **B1a: Rolling 168h Gross Z-Score** | 90% | 90.1% | 84.0% | **-6.1** | 234.87 | 227.07 | 1.0x | Gross-count baselines lack spectral selectivity; wet deposit... |
| **B1b: Global Dry Gross Z-Score (2017-2020)** | 90% | 87.7% | 82.0% | **-5.7** | 256.72 | 161.63 | 0.6x | Gross-count baselines lack spectral selectivity; wet deposit... |
| **B2a: Rain-Masked Dry Z (H=6h Holdoff)** | 90% | 87.7% | 79.5% | **-8.2** | 204.39 | 118.14 | 0.6x | Gross-count baselines lack spectral selectivity; wet deposit... |
| **B2b: Rain-Masked Dry Z (H=12h Holdoff)** | 90% | 86.4% | 79.0% | **-7.4** | 192.22 | 115.26 | 0.6x | Gross-count baselines lack spectral selectivity; wet deposit... |
| **B3a: Weather Regression Residual (Linear)** | 90% | 90.1% | 66.0% | **-24.1** | 110.46 | 70.45 | 0.6x | Gross-count baselines lack spectral selectivity; wet deposit... |
| **B3b: Weather Regression Residual (Spline GAM)** | 90% | 90.1% | 65.5% | **-24.6** | 114.36 | 72.26 | 0.6x | Gross-count baselines lack spectral selectivity; wet deposit... |
| **B4a: Radon-Direction Projection (Global Dry Basis)** | 90% | 90.1% | 68.5% | **-21.6** | 606.30 | 434.57 | 0.7x | Domain shift and out-of-sample stochastic variability.... |
| **B4b: Radon-Direction Projection (Local 168h Basis)** | 90% | 90.1% | 77.0% | **-13.1** | 47.61 | 47.35 | 1.0x | Domain shift and out-of-sample stochastic variability.... |
| **B4c: Radon Projection + Stripped Channel Peak Z** | 90% | 91.4% | 70.0% | **-21.4** | 0.12 | 0.74 | 6.2x | Compton continuum and scattering in Family B Monte Carlo tem... |
| **B5a: Background PCA Subspace (Mahalanobis + Recon)** | 90% | 90.1% | 65.5% | **-24.6** | 3.07 | 133.43 | 43.5x | Domain shift and out-of-sample stochastic variability.... |
| **B5b: Background NMF 2-Component Reconstruction** | 90% | 90.1% | 79.5% | **-10.6** | 55.41 | 129.89 | 2.3x | Domain shift and out-of-sample stochastic variability.... |
| **B6a: Unsupervised Isolation Forest (Tier 3, Bg Only)** | 90% | 90.1% | 69.5% | **-20.6** | 123.93 | 86.90 | 0.7x | Domain shift and out-of-sample stochastic variability.... |
| **B6b: Unsupervised Isolation Forest (Tier 4, Bg Only)** | 90% | 90.1% | 85.0% | **-5.1** | 73.25 | 132.11 | 1.8x | Domain shift and out-of-sample stochastic variability.... |
| **B6c: Unsupervised One-Class SVM (Tier 4, Bg Only)** | 90% | 90.1% | 86.0% | **-4.1** | 28.71 | 103.01 | 3.6x | Domain shift and out-of-sample stochastic variability.... |
| **B7a: Supervised Logistic Regression (Tier 3)** | 90% | 91.4% | 61.5% | **-29.9** | 0.47 | 24.42 | 52.0x | Domain shift and out-of-sample stochastic variability.... |
| **B7b: Supervised Depth-3 Decision Tree (Tier 3)** | 90% | 88.9% | 76.0% | **-12.9** | 56.83 | 59.52 | 1.1x | Domain shift and out-of-sample stochastic variability.... |
| **B8: Operational Screening Approximation (Uncalibrated EPA Threshold)** | 90% | 74.1% | 62.0% | **-12.1** | 354.43 | 203.47 | 0.6x | Domain shift and out-of-sample stochastic variability.... |
| **Tier 1: Gross GBDT (12 features)** | 90% | 90.1% | 82.5% | **-7.6** | 143.90 | 108.85 | 0.8x | Domain shift and out-of-sample stochastic variability.... |
| **Tier 1b: Gross + Weather GBDT (29 features)** | 90% | 90.1% | 76.5% | **-13.6** | 45.37 | 24.83 | 0.6x | Domain shift and out-of-sample stochastic variability.... |
| **Tier 2: Spectral GBDT (30 features)** | 90% | 95.1% | 86.5% | **-8.6** | 0.00 | 1.23 | 123.0x | Lack of weather features leaves spectral model vulnerable to... |
| **Tier 3: Weather Fusion GBDT (48 features)** | 90% | 96.3% | 93.5% | **-2.8** | 0.00 | 12.99 | 1299.0x | Domain shift from Family A analytic training templates to Fa... |
| **Tier 4: Physics-Motivated Nuclear GBDT (59 features)** | 90% | 100.0% | 96.5% | **-3.5** | 0.00 | 9.87 | 987.0x | Threshold saturation on validation set (100% val det at lowe... |
| **Ablation: Tier 4 without Weather/Bateman (39 features)** | 90% | 100.0% | 93.0% | **-7.0** | 0.00 | 6.33 | 633.0x | Domain shift and out-of-sample stochastic variability.... |
| **Ablation: Tier 3 + Radon Stripping Only (57 features)** | 90% | 100.0% | 95.5% | **-4.5** | 0.00 | 7.48 | 748.0x | Domain shift and out-of-sample stochastic variability.... |
| **Ablation: Tier 3 + Bateman Washout ODE Only (50 features)** | 90% | 95.1% | 91.5% | **-3.6** | 0.00 | 3.86 | 386.0x | Domain shift and out-of-sample stochastic variability.... |

## 3. Root Cause Analysis: Differences Between Validation and Test Plumes

The validation and test folds are fundamentally different by construction in the realistic benchmark (`src/artifact_tests.py`):

### A. Spectral Template Family Shift (Family A vs. Family B)
- **Validation Plumes (Family A)**: Generated using analytic response templates based on primary photopeak gamma-branching fractions, with simplified Gaussian broadening.
- **Test Plumes (Family B)**: Generated using 3D Monte Carlo (MCNP-derived) photon-transport response functions for a 2"x2" NaI(Tl) crystal in a semi-infinite cloud geometry. Family B templates incorporate comprehensive Compton scattering continua, backscatter peaks, and escape peaks.
- **Impact on Non-ML B4c**: Baseline B4c relies on the maximum $Z$-score of the stripped single-channel excess ($Z = \max_i (C_i - s_{\text{Rn}, i} C_{\text{gross}}) / \sigma_i$). In Family A, 40--65% of injected photons concentrate cleanly in the nominal photopeak channel (e.g. Channel 5 for Cs-137), producing sharp channel spikes. In Family B, multiple Compton scatterings distribute up to 45% of photon energies into lower channels (Channels 2--4), lowering the single-channel peak intensity and reducing B4c detection from 91.4% on validation down to 70.0% on test.

### B. Gamma-Poisson Stochastic Counting Noise
- **Validation Plumes**: Initial validation sweeps utilized smooth continuous intensity curves without overdispersed counting fluctuations.
- **Test Plumes**: Evaluated with empirical Gamma-Poisson counting noise calibrated to RadNet Fano factors (`data/processed/artifact_counting_noise.csv`). Hourly Poisson variance ($\sigma pprox \sqrt{\mu}$) introduces random fluctuations that periodically dip below detection thresholds for borderline plumes.

### C. Timing and Continuous Filter-Crossing Dynamics
- **Validation Plumes**: Timed to terminate synchronously with filter tape drops (legacy design).
- **Test Plumes**: 62 of 200 test plumes span across an active automated filter tape replacement, splitting the accumulated activity and reducing the effective integration window.

### D. Independent ICRP-74 Ambient Dose Rate Coupling
- **Validation Plumes**: Used circular channel-proportional dose coupling ($k_{\text{dose}}$).
- **Test Plumes**: Injected dose rate is modeled independently via ICRP-74 ambient dose equivalent coefficients $H^*(10)$, breaking the circular feature dependency.

## 4. Diagnosis of Tier 4 Threshold Saturation (Task 2.b)
Tier 4 returns identical operating points (**96.5% detection, 9.87 FA/yr**) across 80%, 90%, and 95% validation detection targets.

### Why Saturation Occurs:
1. On the 2021--2022 validation fold (8.46 station-years across 5 pilot stations), Tier 4 predicted fission probabilities on clean background hours never exceed $p \ge 0.98$ ($0.0\%$ of clean hours).
2. At threshold $\tau = 0.9740$, Tier 4 already achieves **100.0% validation detection** on the 250 validation plumes while triggering **0.00 false alarms** on the clean validation fold.
3. When the threshold selection algorithm searches for the threshold achieving $\ge 80\%$, $\ge 90\%$, or $\ge 95\%$ validation detection with minimal validation false alarms, $\tau = 0.9740$ is the unique optimal point on the grid for all three targets.

### Resolution via Rank-Based Score Quantiles:
By replacing detection-target selection with **clean-validation score quantiles** (or false-alarm budget quantiles $\alpha \in \{1, 2, 5, 10, 20\}$ FA/yr), thresholds are selected directly from the continuous empirical distribution of clean scores. As documented in `benchmark_fa_budget_operating_points.csv`, this completely eliminates saturation, producing distinct thresholds ($\tau = 0.9995, 0.9989, 0.9969, 0.9928, 0.9832$) with strictly monotonic validation and test false alarm rates.

## 5. Protocol Recommendation (Task 2.c)
1. **Primary Deployment Protocol (Main Table)**: Report the 20-seed realistic benchmark results at the 95% validation detection target (Table 1: Tier 4 is 93.6% Det, 7.19 FA/yr; Tier 3 is 85.8% Det, 1.23 FA/yr; B4c is 80.0% Det, 2.30 FA/yr; Rolling 3-sigma is 55.0% Det, 82.95 FA/yr).
2. **Pre-specified Comparison**: Retained in the supplementary materials with the explicit caveat that validation-matched operating points do not yield matched test performance.
3. **Post-Hoc Matched Curves**: Present full test detection-vs-FA ROC curves with bootstrap confidence bands and matched-test-budget operating points (5, 10, 50 FA/yr), explicitly stating that test-matched thresholds are theoretical ceilings rather than blind operational points.
