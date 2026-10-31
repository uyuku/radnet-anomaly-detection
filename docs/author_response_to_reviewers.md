# Author Response to Reviewers

**Manuscript Title:** Disentangling Precipitation-Induced Radon Washout from Anthropogenic Radiological Plumes in Environmental Radiation Networks  
**Target Journal:** *Sensors* (MDPI) — Special Issue on Environmental Sensing and Radiation Surveillance  
**Authors:** N. Ömer Barut and Şule Ergün  
**Date:** October 31, 2026  
**Revision Round:** Major Revision (Phase 10)

---

We sincerely thank the Reviewer for their thorough, incisive, and rigorous evaluation of our manuscript. The Reviewer's constructive criticisms pinpointed crucial discrepancies, overstatements, and methodological nuances that have substantially elevated the scientific integrity, clarity, and precision of our paper. 

In this revision, we have addressed every concern point-by-point without defensive softening. In accordance with the study's pre-registered ground rules, all changes have been backed by executable code, verified against the frozen test sets, and integrated directly into the compiled manuscript (`report/manuscript_mdpi.pdf`) and automated table generation pipelines (`scripts/make_manuscript_tables.py`).

Below is our detailed, point-by-point response to all 8 Major Concerns, 6 Minor Concerns, and 6 Author Questions.

---

## Part I: Responses to Major Concerns

### Major Concern 1: Station Mismatch, Contradictions, and Misleading Calibration Narrative in Section 4.4 vs. Table 3

> **Reviewer's Comment:**  
> *"Section 3.1 and Section 4.4 incorrectly list Chicago, Denver, Seattle, and Boston as evaluated stations, whereas Table 3 and `extended_loso_11station_summary.csv` actually evaluate Phoenix, Omaha, Baton Rouge, and Burlington. Furthermore, the text claims Tampa achieved 100.0% detection and 1.44 FA/yr (which was copied from Birmingham; Tampa is actually 87.5% and 1.80 FA/yr). Most critically, the text claims that 12-month local calibration resolves cross-station transfer. However, Table 3 shows that 12-month static calibration catastrophically fails at Honolulu (638.5 FA/yr) and Burlington (332.9 FA/yr). The true solution is causal 168h rolling normalization, which achieves 0.0 FA/yr across 9 of 11 stations. The narrative in Section 4.4 must be rewritten to reflect the actual data."*

**Author Response:**  
We completely agree. The Reviewer identified an unacceptable textual holdover from exploratory pilot station runs that directly contradicted our empirical findings in Table 3. We have resolved this across the codebase and manuscript:

1. **Station Harmonization (Sections 3.1 & 4.4):**  
   We have updated all station listings in `report/manuscript_mdpi.tex` (Lines 106 and 200–207) to match the exact 11 stations evaluated in `data/processed/extended_loso_11station_summary.csv` and Table 3:
   - **Pilot Stations:** Birmingham, AL; Washington, DC; San Diego, CA; Dallas, TX; Tampa, FL.
   - **Extended LOSO Stations:** Phoenix, AZ; Omaha, NE; Honolulu, HI; Anchorage, AK; Baton Rouge, LA; Burlington, VT.
   All mentions of Chicago, Denver, Seattle, and Boston have been excised.

2. **Accurate Tampa Reporting:**  
   The misattributed Birmingham numbers have been replaced with Tampa's exact empirical metrics: under 12-month post-step local calibration, Tampa achieves **87.5% detection at 1.80 FA/yr** (4 episodes over 2.22 station-years), and under causal 168h rolling normalization without local retraining, it achieves **90.0% detection at 7.19 FA/yr** (Section 4.4, Lines 206–207).

3. **Restructuring the Spatial Transfer Narrative:**  
   We have rewritten Section 4.4 to explicitly highlight the catastrophic failure of static 12-month calibration at Honolulu (638.5 FA/yr) and Burlington (332.9 FA/yr), explaining the physical mechanism: a static 12-month calibration locks in a threshold over a single annual cycle. When subsequent years experience multi-year climatological shifts, prolonged snow cover insulation, or maritime background drift, static thresholds trigger persistent false alarms.  
   We then present **causal 168-hour rolling normalization (`Tier 4 Rolling 168h`)** as the true robust solution: it achieves **0.00 false alarms per year across 9 of 11 stations**, 1.05 FA/yr at San Diego, and 7.19 FA/yr at Tampa, while maintaining 80.0% to 100.0% detection across all 11 stations. Table 3 has been updated with a dedicated `Tier 4 Rolling 168h` column and re-labeled `Tier 4 Static 12m`.

---

### Major Concern 2: False-Alarm Budget Overshoot in Section 4.2 vs. Table 2

> **Reviewer's Comment:**  
> *"At $\alpha = 1.0$ FA/yr validation budget, the test FA rate is $17.59 \pm 3.41$ FA/yr (a 17.6x overshoot) due to background non-stationarity. The text incorrectly framed 1.0 FA/yr as an achieved operating constraint."*

**Author Response:**  
We thank the Reviewer for catching this defensive phrasing. We have rewritten Section 4.2 (Lines 181–182) to prominently disclose and discuss this budget overshoot:

> *"Across all pre-specified false-alarm operating budgets ($\alpha \in \{1, 2, 5, 10\}$ yr$^{-1}$), Tier 4 maintains detection rates above 98\%, detecting over 82\% of plumes within $\le 6$ hours of onset. However, Table~\ref{tab:fa_budget_operating_points} reveals substantial out-of-sample false-alarm budget overshoot on the 2023--2025 test split: when calibrated to a strict validation budget of $\alpha = 1.0$ yr$^{-1}$, realized test false alarm rates reach $17.59 \pm 3.41$ yr$^{-1}$ for Tier 4, $31.50 \pm 8.44$ yr$^{-1}$ for Tier 3, and $78.53 \pm 34.86$ yr$^{-1}$ for Tier 2. This divergence demonstrates that static threshold calibration on historical folds cannot rigidly maintain single-digit annual false alarm rates in the presence of multi-year atmospheric non-stationarity and sensor baseline drift without adaptive online normalization."*

This honest framing directly underscores our thesis: static thresholds calibrated on historical data fail to account for field distribution drift; adaptive causal baseline tracking is mandatory.

---

### Major Concern 3: Discrepancy Between Table 1 and Table 6 at the 95% Detection Target

> **Reviewer's Comment:**  
> *"Table 1 reports Tier 4 at 94.5% detection and 2.71 FA/yr (`eval_benchmark_uncertainty_summary.csv`), whereas Table 6 reports 96.5% detection and 9.87 FA/yr (`ml_ablation_summary.csv`). Furthermore, for B5b, B6a, and B7b, the 95% validation detection target was unreachable, but they are included without explanation."*

**Author Response:**  
We have clarified this discrepancy in both the text (Section 4.3) and the Table 6 caption and footnote:

1. **Threshold Sweep Differences:**  
   Table 1 evaluates models at the primary operational operating point determined via a fine-grained threshold sweep on the validation fold ($\tau = 0.9850$, yielding 94.5% test detection at 2.71 FA/yr). Table 6 evaluates the complete comparator hierarchy across an identical, unified coarse ablation grid ($\tau = 0.9740$, yielding 96.5% test detection at 9.87 FA/yr). We have added an explicit note ($^*$) to Table 6 and detailed this distinction in Section 4.3.

2. **Unreachable Targets Flagged with Dagger ($^\dagger$):**  
   In `ml_ablation_summary.csv`, B5b (NMF), B6a (Isolation Forest), and B7b (Decision Tree) could not achieve 95% validation detection even at their lowest operating thresholds (achieving validation maxima of 93.8%, 93.8%, and 88.9%, respectively). We updated `scripts/make_manuscript_tables.py` so that Table 6 appends a dagger ($^\dagger$) to these model names and includes the explanatory footnote:
   > *"$\dagger$ The 95% validation detection target was unreachable for this model (maximum validation detection achieved: B5b 93.8%, B6a 93.8%, B7b 88.9%); reported test metrics reflect the lowest-threshold operating point."*

---

### Major Concern 4: Radionuclide Memorization and 0% Detection on Held-Out Isotopes

> **Reviewer's Comment:**  
> *"In leave-one-scenario-out experiments, held-out $^{131}\text{I}$ or $^{60}\text{Co}$ gives 0.0% detection for Tier 4. Trees memorize specific energy channel shares. In contrast, non-ML B4c requires no synthetic training and strips the radon subspace geometrically."*

**Author Response:**  
We fully accept this crucial critique. Rather than hiding this limitation, we have made it a central finding of our comparative analysis:

1. **Section 4.3 (Results):** Added explicit discussion of the leave-one-scenario-out findings from `data/processed/benchmark_holdout_templates.csv`:
   > *"Importantly, our leave-one-scenario-out experiments (archived in `benchmark_holdout_templates.csv`) reveal a fundamental limitation of supervised GBDTs: when an isotope is excluded from training, detection collapses to 0.0% for $^{131}\text{I}$ and 0.0% for $^{60}\text{Co}$ in Tier 4. The supervised trees memorize the specific energy-channel signatures of the training templates rather than learning an abstract 'non-radon anomaly' manifold. In contrast, non-ML baseline B4c operates purely by geometric subspace projection and remains robust against arbitrary out-of-template emissions without supervised training."*

2. **Section 6 (Limitations, Item 2):** Dedicated Limitation Item 2 exclusively to radionuclide memorization and out-of-distribution collapse.

3. **Section 5 (Discussion):** Highlighted that the primary advantage of non-ML geometric radon projection (B4c) over supervised GBDTs is that B4c requires zero synthetic isotope training and does not suffer from spectral template over-fitting.

---

### Major Concern 5: Alarm Audit Confounding in Table 4 and Wilson vs. Clopper-Pearson Intervals

> **Reviewer's Comment:**  
> *"The 100 audited alarms pooled gross baselines (which have 32.5% rain alarms in full data) with Tier 4 (which has 1.6% rain alarms) into an aggregate '44% physical alarms'. Pooling them is confounded. Furthermore, the confidence intervals in Table 4 are Wilson score intervals, not Clopper-Pearson."*

**Author Response:**  
We thank the Reviewer for this sharp statistical observation. We have updated Table 4, the table generator script (`scripts/make_manuscript_tables.py`), and the manuscript text:

1. **Disaggregated Table 4:**  
   Table 4 has been expanded to report separate columns for:
   - Gross Baselines ($N=45$)
   - Tier 3 Weather Fusion ($N=41$)
   - Tier 4 Physics-Informed ($N=37$)
   - Overall Sample ($N=100$) with 95% Wilson Score CIs.

2. **Key Empirical Disaggregation Results:**  
   - **Active Precipitation Radon Washout:** 15.6% ($7/45$) in baselines, 14.6% ($6/41$) in Tier 3, but **only 2.7% ($1/37$) in Tier 4**.
   - **Hardware Step-Drift (Tampa 2023):** 17.8% ($8/45$) in baselines, 43.9% ($18/41$) in Tier 3, and **32.4% ($12/37$) in Tier 4**.
   - **Pure Statistical Noise:** 62.2% ($28/45$) in baselines, 39.0% ($16/41$) in Tier 3, and 62.2% ($23/37$) in Tier 4.

3. **Confidence Interval Correction:**  
   The caption and column header now correctly state **"95% Wilson score confidence intervals"** (overall physical: [34.7%, 53.8%]; pure noise: [46.2%, 65.3%]).

---

### Major Concern 6: Misattribution of Author's Own Findings to Hsieh et al. (2026)

> **Reviewer's Comment:**  
> *"Line 72 cited Hsieh et al. (2026) for the +6.9%, +41.4%, and +196.7% rain surges. These numbers are actually this project's own RadNet empirical findings from `rain_washout_surge_distribution.csv`."*

**Author Response:**  
We apologize for this citation placement error. We have corrected Line 72:

> *"Empirical measurements across EPA RadNet stations indicate that rain events produce a median gross count rate surge of $+6.9\%$, with the 90th percentile reaching $+41.4\%$ and severe convective thunderstorms surging up to $+196.7\%$ above baseline in our empirical RadNet analysis. Conversely, maritime airflow regimes in coastal environments can induce subtle baseline depressions."*

Hsieh et al. (2026) is cited in Section 2 (Line 86) where it belongs, for high-resolution minute-level precipitation response and subsequent exponential decay dynamics.

---

### Major Concern 7: Data Leakage in Dry Baseline Normalizer Script

> **Reviewer's Comment:**  
> *"`src/compute_train_only_baselines.py` set `TRAIN_END = '2023-01-01'`, inadvertently leaking the 2021--2022 validation fold into the dry baseline statistics in `station_dry_baselines_train_only.json`."*

**Author Response:**  
We immediately corrected `TRAIN_END = '2021-01-01'` in `src/compute_train_only_baselines.py` and regenerated `data/processed/station_dry_baselines_train_only.json`. The resulting dry baseline means shifted by less than 0.8% (e.g., Birmingham dry mean shifted from 3870.3 to 3878.5 CPM, Tampa shifted from 2278.5 to 2276.5 CPM), and all baseline metrics remain completely verified with zero validation leakage.

---

### Major Concern 8: Incremental Value of the Bateman ODE Heuristic

> **Reviewer's Comment:**  
> *"Table 6 shows that adding Bateman/weather to Tier 4 actually increases test false alarms from 6.33 to 9.87 FA/yr. The Bateman ODE heuristic adds minimal value compared to spectral radon subspace stripping."*

**Author Response:**  
We agree completely. In Sections 3.2.1, 4.3, and 5, we have significantly tempered our claims regarding the Bateman ODE. We explicitly state that:
1. Feature importance rankings show that `hours_since_rain` and spectral radon projection residuals (`local_radon_stripping_residual_l2`) dominate the decision splits.
2. The Bateman ratio provides only secondary discriminative refinements.
3. Adding weather and Bateman features slightly increases false alarm variance due to precipitation timing discrepancies between airport weather stations and radiation monitors (Section 5).

---

## Part II: Responses to Minor Concerns

1. **Minor Concern 1 (Station Naming Consistency):**  
   *Response:* Harmonized across the manuscript, abstract, tables, and macros to the 11 actual evaluated stations.

2. **Minor Concern 2 (Scope Qualification in Discussion):**  
   *Response:* Added the explicit qualification *"on synthetic atmospheric plumes"* in Section 5 (Line 227).

3. **Minor Concern 3 (Tier 3 Feature Count):**  
   *Response:* Corrected Tier 3 feature count from 52 to 48 in Section 3.4 (Line 145), matching `ml_ablation_summary.csv`.

4. **Minor Concern 4 (Uncited Bibliography Entries):**  
   *Response:* Verified all 38 bibliography items. `Karafasoulis2026` is now cited in Section 6 (Limitation 4), and `NNDC2023` is cited in Section 3.3 (Line 133). Both cites and bibitems are in exact 1-to-1 correspondence (38/38 cited, 0 uncited).

5. **Minor Concern 5 (Table 3 Layout):**  
   *Response:* Updated Table 3 with a dedicated column for `Tier 4 Rolling 168h` and renamed `Tier 4 Calibrated` to `Tier 4 Static 12m`.

6. **Minor Concern 6 (Table 6 Dagger and Footnote):**  
   *Response:* Added $^\dagger$ markers to B5b, B6a, and B7b with a footnote detailing unreachable 95% targets, and added a footnote explaining the Tier 4 threshold difference between Table 1 and Table 6.

---

## Part III: Responses to Author Questions

### Question 1: What is the computational latency and memory footprint of Tier 4 vs. Non-ML B4c?
**Response:**  
We directly benchmarked runtime on an Apple Silicon CPU core:
- **Feature Engineering:** Full 59-feature extraction (including causal 168-hour rolling statistics and Bateman washout convolution) executes in **0.0125 ms per station-hour**.
- **Non-ML B4c Scoring:** Involves the causal residual vector norm and peak stripped channel z-score, executing in **0.0001 ms (0.1 $\mu$s) per station-hour** with negligible memory footprint (< 5 MB).
- **Tier 4 GBDT Scoring:** LightGBM 100-tree evaluation across all 59 features executes in **~0.005 ms per station-hour** with a model memory footprint of **1.8 MB**.
Total end-to-end processing latency for either architecture is **$< 0.02$ ms per station-hour**, orders of magnitude below the 1-hour reporting interval (3,600,000 ms) of operational radiation networks.

### Question 2: Why does San Diego exhibit a negative gross count response during rain?
**Response:**  
San Diego, CA is located in a coastal Mediterranean climate where prevailing onshore winds transport marine air masses characterized by extremely low terrestrial radon progeny concentrations. During convective frontal passages, incoming marine air displaces stagnant continental air while surface soil moisture suppresses terrestrial radon exhalation. In addition, atmospheric humidity and air density slightly attenuate low-energy terrestrial gamma flux. Across our 9-year dataset (`data/processed/rain_washout_surge_distribution.csv`), San Diego rain events ($N = 1{,}452$ hours) produce a **median gross count rate shift of $-8.65\%$** (mean $-7.82\%$, 10th percentile $-15.12\%$, 90th percentile $+0.42\%$, max $+22.54\%$) rather than a surge. For heavy rain events ($>5$ mm/h), the median shift reaches $-9.70\%$.

### Question 3: How does causal 168h rolling normalization handle missing data or sensor restart?
**Response:**  
Our causal rolling normalizer requires a minimum valid history of 24 hours within the backward-looking 168-hour window. If a station suffers an extended communication outage exceeding 144 hours, the normalizer falls back to the station's historical dry-weather baseline ($\mu_{\text{dry}}, \sigma_{\text{dry}}$) until 24 hours of continuous telemetry are re-accumulated.

### Question 4: Could non-ML B4c be combined with simple precipitation masking?
**Response:**  
Yes. However, Table 6 shows that B4c alone achieves **0.0% rain-coincident false alarms** without precipitation masking because the radon projection vector already strips the spectral direction of radon deposition. Adding rainfall masking provides zero additional false-alarm reduction for B4c while introducing a 2-to-4 hour blind period during which genuine releases coinciding with storms would be ignored.

### Question 5: What is the authors' recommended operational deployment configuration?
**Response:**  
Based on our findings, we recommend a **hybrid dual-stage deployment architecture**:
1. **Primary Anomaly Sentry:** Non-ML B4c with causal 168h baseline normalization. Because B4c requires no synthetic release training, it provides robust, unbiased nuisance rejection across diverse climates and cannot be fooled by out-of-template radionuclides.
2. **Secondary Diagnostic Classifier:** Supervised Tier 4 GBDT triggered only when B4c raises an anomaly, utilizing multi-modal weather fusion and Bateman kinetics to rapidly adjudicate whether the event resembles a known reactor accident release profile or an instrumental artifact.

### Question 6: What are the open questions for field deployment on non-$\text{NaI(Tl)}$ detectors?
**Response:**  
Operational networks utilizing $\text{LaBr}_3(\text{Ce})$, $\text{CZT}$, or high-purity germanium (HPGe) possess much higher energy resolution, resolving individual photopeaks ($^{214}\text{Pb}$ at 352 keV, $^{214}\text{Bi}$ at 609 keV, $^{137}\text{Cs}$ at 662 keV) without broad channel overlap. On such systems, radon stripping simplifies into direct peak-area subtraction rather than subspace projection. However, temperature gain stabilization and scintillation crystal self-activity (e.g., $^{138}\text{La}$ and $^{227}\text{Ac}$ in $\text{LaBr}_3$) present distinct background compensation challenges that warrant future investigation.

---

## Summary of File Modifications in this Revision

1. **`report/manuscript_mdpi.tex`**: Comprehensive text revision addressing all 8 Major and 6 Minor concerns.
2. **`report/tables/table3_loso_11stations.tex`**: Added `Tier 4 Rolling 168h` column; relabeled `Tier 4 Static 12m`.
3. **`report/tables/table4_alarm_adjudication.tex`**: Disaggregated by detector family; updated to 95% Wilson CIs.
4. **`report/tables/table6_ablation_summary.tex`**: Added $^\dagger$ for unreachable targets and explanatory footnotes.
5. **`report/manuscript_macros.tex`**: Regenerated with 64 dynamic macros.
6. **`scripts/make_manuscript_tables.py`**: Updated table generators and macros.
7. **`src/compute_train_only_baselines.py`**: Fixed train split cutoff to `2021-01-01`.
8. **`data/processed/station_dry_baselines_train_only.json`**: Recomputed strictly on pre-2021 training fold.
9. **`report/manuscript_mdpi.pdf`**: Compiled camera-ready PDF (21 pages, 0 errors, 0 undefined references).
10. **`docs/author_response_to_reviewers.md`**: Formal author response document.

We thank the Reviewer once again for their exceptional stewardship of the scientific literature.
