# Pre-Specified Statistical Analysis Plan (Phase 5)

**Timestamp & Pre-Specification Note:** This statistical analysis plan was written and committed prior to running the Phase 5 resampling, paired comparison, and multi-catalog variance decomposition scripts (`src/uncertainty.py`, `src/evaluate_uncertainty.py`).

---

## 1. Primary Pre-Specified Hypothesis & Endpoint

- **Competing Detectors:**
  1. `ML_Tier4_PhysicsNuclear` (**Tier 4: Physics-Motivated Nuclear GBDT**, 59 features, trained on 2017–2020 with synthetic plumes).
  2. `B4c_RadonProjection_Combined` (**B4c: Non-ML Local Spectral Radon Projection + Stripped Channel Peak Z**, fitted on 2017–2020 clean background only, identified in Phase 3 as the strongest competing non-ML detector).
- **Primary Endpoint:**
  - The paired difference in out-of-sample test false-alarm episodes per station-year,
    $$\Delta \text{FA}_{90\%} = \text{FA}_{\text{Tier 4}}(\tau^*_{\text{Tier 4}, 90\%}) - \text{FA}_{\text{B4c}}(\tau^*_{\text{B4c}, 90\%}),$$
    at the **90% validation detection target** on the 2023–2025 test split (`106,628` observed hours = `12.1638` station-years across 5 stations).
- **Primary Estimation & Inference Procedure:**
  - Joint matched-operating-point bootstrap ($B = 10,000$ replicates): in each replicate $b = 1,\dots,B$, we jointly resample the 2021–2022 validation plumes (stratified by station and environment) and 2021–2022 validation station-months, re-derive $\tau^{(b)}_{\text{Tier 4}, 90\%}$ and $\tau^{(b)}_{\text{B4c}, 90\%}$ on the resampled validation fold (incorporating threshold-selection noise), and evaluate both thresholds on the jointly resampled 2023–2025 test station-months and test plumes.
  - We also report the fixed-threshold paired station-month block bootstrap ($B = 10,000$) holding $\tau^*$ fixed at the point estimate from the full 2021–2022 validation fold.
- **Decision Rule:**
  - A statistically supported difference on the primary endpoint is claimed if and only if the two-sided 95% joint bootstrap confidence interval excludes zero ($\alpha = 0.05$).

### Important Methodological Caveat: Non-Matched Nature of Validation Targets
> [!NOTE]
> **Validation Calibration vs. Test Matching:** While this primary comparison was pre-specified prior to test evaluation, empirical testing reveals that validation detection targets do NOT produce matched test detection due to out-of-sample domain and noise shift (Family A analytic validation plumes vs. Family B Monte Carlo test plumes with Gamma-Poisson counting noise).
> Specifically, at the pre-specified 90% validation target:
> - `Tier 4` achieves **96.5% test detection** at **9.87 false alarms/station-year**.
> - `B4c` achieves **70.0% test detection** at **0.74 false alarms/station-year**.
> Because the test detection rates diverge by 26.5 percentage points, this pre-specified comparison is inherently unmatched on the test split.
> A complementary **post-hoc matched analysis** evaluating full test ROC curves, matched test FA budgets (5, 10, 50 FA/yr), and matched test detection rates is provided in `docs/post_hoc_matched_analysis.md` and `data/processed/post_hoc_matched_test_table.csv`. Note that thresholds chosen from test scores represent post-hoc theoretical ceilings and are not deployable operational operating points.

---

## 2. Exploratory Comparisons & Multiplicity Control (Holm-Bonferroni)

All comparisons other than the single primary comparison above are pre-specified as **Exploratory**. We organize exploratory paired tests into two families and apply the **Holm-Bonferroni step-down procedure** (`holm_adjust_pvalues`) within each family to control the family-wise error rate at $\alpha = 0.05$:

### Exploratory Family A: Paired Test False-Alarm Differences ($\Delta \text{FA/yr}$) and Detection Differences ($\Delta \text{Det \%}$) at Validation Detection Targets
1. `Tier 4` vs. `Tier 3 (Weather Fusion)` at the 90% validation detection target
2. `Tier 4` vs. `Tier 3 (Weather Fusion)` at the 95% validation detection target
3. `Tier 4` vs. `B4c (Non-ML Radon Projection)` at the 95% validation detection target
4. `Tier 4` vs. `Tier 2 (Spectral)` at the 95% validation detection target

### Exploratory Family B: Paired Test Detection Differences ($\Delta \text{Det \%}$) and False-Alarm Differences ($\Delta \text{FA/yr}$) at Validation FA Budgets
1. `Tier 4` vs. `B4c (Non-ML Radon Projection)` at the 5 validation FA/yr budget
2. `Tier 4` vs. `B4c (Non-ML Radon Projection)` at the 10 validation FA/yr budget
3. `Tier 4` vs. `Tier 3 (Weather Fusion)` at the 5 validation FA/yr budget
4. `Tier 4` vs. `Tier 3 (Weather Fusion)` at the 10 validation FA/yr budget

**Interpretation Rule for Tier 4 vs. Tier 3:**
Per Ground Rule 3 and Phase 5 Step 4, paired t-tests and Wilcoxon signed-rank p-values across 20 LightGBM random seeds on a single fixed dataset (`rigorous_seed_statistical_tests.csv`) measure only solver initialization/subsampling randomness and are removed from headline claims. A difference between Tier 4 and Tier 3 is claimed only if the paired plume/block bootstrap 95% confidence interval excludes zero.

---

## 3. Single-Detector Interval Methods (Step 1a–b)

1. **Event Detection Rate (`cluster_bootstrap_detection` & `clopper_pearson_ci`):**
   - **Stratified Plume Cluster Bootstrap ($B = 10,000$):** Plumes are the primary sampling unit. We resample test plumes with replacement within `(station_id, environment)` strata (`dry_standard`, `rain_stress`, etc.) and compute the percentile 95% confidence interval `[2.5%, 97.5%]`.
   - **Exact Clopper-Pearson Binomial CI:** Reported alongside the bootstrap CI for every single proportion $k / n$ (`scipy.stats.beta` exact quantiles, handling $k = 0$ and $k = n$ cleanly).
2. **False-Alarm Rate (`block_bootstrap_fa_rate` & `exact_poisson_rate_ci`):**
   - **Station-Month Block Bootstrap ($B = 10,000$):** Discrete false-alarm episode starts are identified on the continuous hourly station series *before* binning into station-months (`180` station-months on the 2023–2025 test split) so episodes crossing month boundaries are never split or double-counted. In each bootstrap replicate, `M` station-months are resampled with replacement and the rate $\sum \text{episodes}_m / \sum \text{exposure\_years}_m$ is computed.
   - **Station-Week Block Bootstrap Sensitivity Check ($B = 10,000$):** Same procedure using 7-day station-week blocks (`~785` station-weeks on the 2023–2025 test split) to verify insensitivity to block length.
   - **Exact Poisson (Garwood $\chi^2$) Rate CI:** For total episode count $K$ over observed exposure $T$ station-years, exact 95% rate bounds are $[\frac{1}{2T}\chi^2_{2K, 0.025}, \frac{1}{2T}\chi^2_{2(K+1), 0.975}]$ (`[0, -\ln(0.05)/T] = [0, 2.9957/T]` when $K=0$ for one-sided 95% or `[0, 3.6889/T]` for two-sided 97.5%).

---

## 4. Benchmark-Design Variance Decomposition (Step 2)

To separate model-seed randomness from synthetic plume catalog sampling variance:
- We generate **$C = 10$ independent synthetic plume catalogs** (`catalog_seed = 100..109`, each drawing 250 train/validation plumes and 200 test plumes under the Phase 2 Family A/B realistic generator constraints).
- On each catalog $c \in \{1,\dots,10\}$, we train each GBDT tier across **$S = 3$ LightGBM seeds** (`42, 43, 44`), yielding $30$ full train-validate-test runs per tier.
- We decompose total variance across the $C \times S = 30$ runs into:
  - **Between-Catalog Variance ($\sigma^2_{\text{catalog}}$):** variance attributable to which synthetic plume scenarios, start times, and weather states are sampled.
  - **Within-Catalog Model-Seed Variance ($\sigma^2_{\text{seed}}$):** variance attributable to LightGBM random seed holding the catalog fixed.

---

## 5. Coverage Simulation Verification (Step 5)

Before trusting the interval estimators, `verify_interval_coverage_simulation` runs $M = 500$ Monte Carlo trials with known ground truth:
1. **Detection Coverage:** $N = 200$ plumes stratified across 5 stations $\times$ 2 environments with known true detection probability $p_{\text{true}} \in \{0.80, 0.90\}$; checks that both Stratified Bootstrap ($B = 1,000$ per trial) and Clopper-Pearson 95% CIs cover $p_{\text{true}}$ in $[93.0\%, 97.0\%]$ of trials (or conservative $\ge 94.0\%$ for exact Clopper-Pearson).
2. **False-Alarm Rate Coverage:** $M_{\text{blocks}} = 180$ station-month blocks with known true rate $\lambda_{\text{true}} \in \{5.0, 10.0\}$ episodes/station-year; checks that Station-Month Block Bootstrap and Exact Poisson 95% CIs cover $\lambda_{\text{true}}$ within nominal bounds.
