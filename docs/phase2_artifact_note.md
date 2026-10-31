# Phase 2: Synthetic Plume Realism & Generator Artifact Audit Note

## 1. Executive Summary
To test whether the v1 synthetic benchmark results (`94.0%` Tier 4 detection at `2.71` false alarms/station-year) were inflated by generator artifacts or shared train/test spectral templates, we rebuilt the synthetic injection framework (`src/detector_response.py`, `src/nuclear_physics.py`, `src/synthetic_injection.py`) and executed four systematic generator-artifact audits (`src/artifact_tests.py`).

Three generator artifacts moved test detection by **more than 2 percentage points (pp)** on one or more tiers:
1. **Shared Hand-Written Templates $\to$ Independent Derived Detector-Response Families (Family A Analytic Train vs. Family B 3D Monte Carlo Test)**: Moved detection by **`-4.93 pp`** on Tier 2 (`93.53% -> 88.60%`), **`-2.58 pp`** on Tier 1b (`79.28% -> 76.70%`), **`-2.30 pp`** on Tier 3 (`90.20% -> 87.90%`), and **`-1.40 pp`** on Tier 4 (`94.10% -> 92.70%`).
2. **Filter-Terminated Plume Timing & Filter-Age Leakage (`artifact_filter_age_leakage.csv`, `plume_timing_comparison.csv`)**: In the legacy `filter_terminated` catalog (where 100% of plumes ended at a filter replacement drop and never crossed a drop), a LightGBM classifier using **only** `hours_since_last_filter_drop` and `hour_of_day` achieved validation ROC-AUC **`0.6311`** (test ROC-AUC **`0.6249`**), confirming **filter-age leakage** because multi-day plumes were packed into the late half of filter cycles. Shifting to `continuous_arbitrary_end` timing (where 62 of 200 test plumes cross a real filter change with filter activity resetting and re-accumulating from the ongoing airborne tail, and only 1.1% of plumes end at a filter drop) reduced the filter-age-only validation ROC-AUC to **`0.5195`** (test ROC-AUC **`0.5061`**, passing the $\le 0.55$ non-leakage threshold) and moved detection by **`-4.98 pp`** on Tier 2 (`87.90% -> 82.92%`), **`-3.28 pp`** on Tier 1b (`86.70% -> 83.42%`), **`-2.65 pp`** on Tier 1 (`90.60% -> 87.95%`), **`-0.67 pp`** on Tier 3 (`86.50% -> 85.83%`), and **`+4.70 pp`** on Tier 4 (`88.90% -> 93.60%`, accompanied by a validation threshold shift that raised Tier 4 false alarms from `2.15` to `7.19` FA/yr).
3. **Zero Counting Noise on Smooth Curves (`artifact_counting_noise.csv`)**: Legacy v1 injections added deterministic smooth activity curves without Poisson or channel-overdispersion counting noise (high-frequency variance ratio `0.0002` to `0.6565` of empirical target). Adding Gamma-Poisson counting noise calibrated to empirical RadNet channel Fano factors (`new_to_target_variance_ratio = 0.933 to 1.033`, passing across all 8 channels) reduced overall Tier 1b detection by **`-3.20 pp`** (`76.70% -> 73.50%`) and reduced **sub-$1\sigma$ plume detection** by **`-41.67 pp`** on Tier 2 (`74.17% -> 32.50%`), **`-28.61 pp`** on Tier 3 (`65.28% -> 36.67%`), and **`-7.50 pp`** on Tier 4 (`68.33% -> 60.83%`).

Per the Ground Rules, because multiple generator artifacts moved detection by $>2\text{ pp}$, **both the legacy (v1) and realistic (v2) benchmark numbers are reported side by side** in `README.md`, `report/interim_report.tex`, and `data/processed/benchmark_realism_comparison.csv`.

---

## 2. Side-by-Side Comparison: Old (Legacy v1) vs. New (Realistic v2) Benchmark
*Evaluated across 20 random seeds (`42..61`) at the 95% validation detection target (`data/processed/benchmark_realism_comparison.csv`).*

| Model Tier | Old Det (%) | New Det (%) | $\Delta$ Det (pp) | Old FA/yr | New FA/yr | $\Delta$ FA/yr | Old Delay (h) | New Delay (h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 1 (Gross)** | `90.08 ± 1.33` | `87.95 ± 2.38` | `-2.13` | `104.77 ± 14.42` | `206.47 ± 52.91` | `+101.70` | `6.15` | `8.20` |
| **Tier 1b (Gross+Weather)** | `79.28 ± 0.84` | `83.42 ± 1.60` | `+4.14` | `9.70 ± 1.69` | `55.43 ± 6.91` | `+45.73` | `8.25` | `10.15` |
| **Tier 2 (Spectral)** | `93.53 ± 1.09` | `82.92 ± 3.60` | `-10.61` | `35.68 ± 41.01` | `0.33 ± 0.37` | `-35.35` | `4.00` | `6.12` |
| **Tier 3 (Weather Fusion)** | `90.20 ± 1.72` | `85.83 ± 2.48` | `-4.37` | `3.44 ± 0.73` | `1.23 ± 1.08` | `-2.21` | `4.00` | `5.90` |
| **Tier 4 (Physics-Motivated Nuclear)** | `94.10 ± 1.12` | `93.60 ± 1.03` | `-0.50` | `2.78 ± 1.55` | `7.19 ± 1.17` | `+4.41` | `4.50` | `5.00` |

### Step-by-Step Attribution of Detection Changes (`benchmark_realism_comparison.csv`)
| Model Tier | Old Legacy (v1) | Step 1: Derived Templates (`A -> B`) | Step 2: `+` Counting Noise | Step 3: `+` Independent Dose | Step 4 (New Realistic): `+` Arbitrary-End Timing |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tier 1 (Gross)** | `90.08%` | `91.10%` (`+1.02`) | `91.10%` (`+0.00`) | `90.60%` (`-0.50`) | `87.95%` (`-2.65`) |
| **Tier 1b (Gross+Weather)** | `79.28%` | `76.70%` (`-2.58`) | `73.50%` (`-3.20`) | `86.70%` (`+13.20`) | `83.42%` (`-3.28`) |
| **Tier 2 (Spectral)** | `93.53%` | `88.60%` (`-4.93`) | `89.50%` (`+0.90`) | `87.90%` (`-1.60`) | `82.92%` (`-4.98`) |
| **Tier 3 (Weather Fusion)** | `90.20%` | `87.90%` (`-2.30`) | `87.90%` (`+0.00`) | `86.50%` (`-1.40`) | `85.83%` (`-0.67`) |
| **Tier 4 (Physics-Motivated Nuclear)** | `94.10%` | `92.70%` (`-1.40`) | `92.90%` (`+0.20`) | `88.90%` (`-4.00`) | `93.60%` (`+4.70`) |

---

## 3. Summary of the Four Artifact Tests & Hold-Out Template Failures

1. **Artifact Test 1 — Filter-Age Leakage (`data/processed/artifact_filter_age_leakage.csv`)**:
   - *Legacy `filter_terminated`*: `100.0%` of plumes ended at a filter drop; a model using only `[hours_since_last_filter_drop, hour_of_day]` achieved validation AUC `0.6311` and test AUC `0.6249` (**ARTIFACT_DETECTED**).
   - *Realistic `continuous_arbitrary_end`*: `1.1%` of plumes end at a filter drop (`62/200` test plumes cross an active filter change); validation AUC dropped to `0.5195` and test AUC to `0.5061` (**PASS**).
2. **Artifact Test 2 — Counting Noise (`data/processed/artifact_counting_noise.csv`)**:
   - *Legacy noiseless curves*: First-difference residual variance was `0.0002x` to `0.6565x` of the target Poisson + overdispersed background variance (**ARTIFACT_DETECTED**).
   - *Realistic noisy injections*: Gamma-Poisson sampling matched the empirical channel Fano factors (`0.933x` to `1.033x` across `R02..R09`, **PASS**).
3. **Artifact Test 3 — Dose Circularity (`data/processed/artifact_dose_circularity.csv`)**:
   - In the legacy pipeline, both the synthetic generator (`compute_injected_dose_rate`) and the Tier 4 feature (`dose_spectral_residual_z`) used the same radon-washout regression vector `k_dose_channel`.
   - Replacing the generator with an independent ICRP-74 + 3D detector efficiency model (`compute_independent_injected_dose_rate`) yielded `93.60 ± 1.03%` detection (`7.19 ± 1.17` FA/yr), while removing `dose_spectral_residual_z` completely yielded `93.75 ± 0.46%` detection (`7.70 ± 1.29` FA/yr), confirming that `dose_spectral_residual_z` contributes `< 1 pp` once circularity is removed (**PASS**).
4. **Artifact Test 4 — Sub-$1\sigma$ Detection & SNR Stratification (`data/processed/artifact_snr_stratification.csv`)**:
   - In the legacy benchmark, the 18 test plumes with gross peak SNR $< 1.0\sigma$ (mean `0.79 sigma` relative to 168h gross SD) were detected **`68.33%`** of the time by Tier 4 and **`65.28%`** by Tier 3 because concentrating 38–68% of counts into a single narrow photopeak channel (`R03`/`R05`/`R07`, whose dry SD is ~18% of gross SD) produced a mean onset stripped channel $z$-score of `1.15` (and peak channel $z$-scores $>2\sigma$) on smooth, noiseless templates.
   - Under the realistic benchmark (Family A $\to$ B + counting noise + arbitrary end), sub-$1\sigma$ detection drops to **`32.50%`** for Tier 2 (`-41.67 pp`), **`36.67%`** for Tier 3 (`-28.61 pp`), and **`60.83%`** for Tier 4 (`-7.50 pp`).
5. **Part B — Hold-Out Template & Unseen Radionuclide Failure Modes (`data/processed/benchmark_holdout_templates.csv`)**:
   - **Leave-one-scenario-out**: Holding out `fission_pure_i131` (photopeak in `R03`) reduces detection on `I-131` to **`0.0%–1.67%`** across Tiers 2–4 because no remaining training scenario populates `R03` without `R05`. Holding out `activation_orphan_co60` (photopeaks in `R07`) reduces detection to **`0.0%`** (Tier 2), **`11.67%`** (Tier 3), and **`33.33%`** (Tier 4). Conversely, scenarios sharing `R05` (`Cs-137`, `Fukushima`, `Mixed`) generalize across each other at `88.33%–100.0%`.
   - **Unseen test-only radionuclides**: `Zr-95/Nb-95` (724/757/766 keV in `R05`, overlapping `Cs-137`) is detected at **`96.00%–98.83%`**, whereas `Ru-103` (497 keV in `R04`) drops to **`0.50%–2.00%`** on Tiers 2–3 (`48.17%` on Tier 4) and `Ba-140/La-140` (537 keV in `R04` + 1596 keV in `R08`, overlapping the Bi-214 radon band) fails at **`0.00%–5.00%`**.
   - **Gain & resolution shifts**: Gain shifts of `-5%` to `+10%` and resolution degradation up to `15%` FWHM maintain `83.83%–94.33%` detection on Tiers 3–4, while a `-10%` gain shift drops Tier 3 to `74.83%` (`86.50%` on Tier 4).
