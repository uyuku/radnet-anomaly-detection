# Phase 6 Report: Weather-Aware Anomaly Detection in EPA RadNet — Final Report

**Date**: 2026-10-06
**Project**: Weather-Aware Anomaly Detection in RadNet (EPA RadNet + NOAA NCEI ISD)
**Status**: Phase 6 draft. All numbers below are taken from frozen result tables in
`data/processed/`; where two evaluation scripts disagree, both values are shown.

---

## Abstract

EPA RadNet stationary gamma air monitors collect ambient particulates on filters and
count them with NaI(Tl) detectors in eight gamma energy channels. Rain washes natural
radon progeny ($^{214}$Pb, $^{214}$Bi) out of the atmosphere onto the filter, inflating
the gross count rate and triggering threshold alarms that operators cannot distinguish
from anthropogenic releases. On 2017–2025 data from five U.S. stations (324,550
synchronous radiation-weather hours), a conventional rolling 3σ threshold on gross
count rate fires 6.6–62.7 alarm episodes per station-year depending on climate, and in
humid stations 62–97% of those episodes coincide with rainfall within 3 hours. We
built a physically parameterized synthetic plume catalog (450 filter-synchronized
injections of $^{137}$Cs, $^{131}$I, $^{60}$Co, $^{134}$Cs mixtures, 250 train /
200 test, calendar-disjoint) and trained gradient-boosted classifiers that fuse gross
count rate, multi-channel spectral ratios, and synchronous NOAA surface weather. At
matched ~92% detection of injected plumes, a weather-fused spectral model (Tier 3)
produces 4.22 ± 1.29 clean false alarms per station-year on unmodified background data,
versus 303.11 for a detection-matched rolling σ rule and 8.52 ± 6.23 for the same model
without weather features. Spectral ratios carry the primary discrimination; weather
fusion adds a statistically significant but smaller reduction (paired t-test
$p = 0.005$; median reduction 14%), mainly by suppressing variance during storms.
Detection performance is measured entirely on synthetic events; there is no labeled
real fission-product ground truth. The realistic weakest regime is short, low-amplitude
plumes in dry weather (78–82% detection) and single-line $^{131}$I releases
(58% detection in leave-one-station-out transfer).

---

## 1. Introduction

RadNet is the U.S. Environmental Protection Agency's national continuous air-monitoring
network for gamma-emitting radionuclides. Its stationary monitors pull air through
glass-fiber filters, count gamma rays in energy channels (R02: 101–200 keV through
R09: 1801–2200 keV), and report hourly gross count rate and, increasingly, ambient
exposure rate. Operationally, the network must answer one question quickly: does a
count-rate excursion indicate a genuine radiological release, or natural radon
progeny washout?

The physics of the confounder is well established. Rain scavenges short-lived radon
daughters from the air column and deposits them on the filter and ground surface;
their decay ($^{214}$Pb $T_{1/2} = 26.8$ min, $^{214}$Bi $T_{1/2} = 19.9$ min) produces
surges of +40% to +180% above baseline within the hour of rainfall onset. A fixed
threshold cannot distinguish this from a plume on gross count rate alone.

This project builds and evaluates a weather-aware anomaly detector:

1. Quantify the false-alarm burden of current fixed-threshold practice on real data.
2. Construct a physically grounded synthetic fission/activation plume benchmark with
   strict train/test separation and hard cases embedded in active rain.
3. Test, under a causal (no test-tuning) protocol across 20 random seeds, whether
   multi-channel spectrometry and NOAA weather fusion measurably reduce false alarms
   at fixed detection probability.

## 2. Data

- **Radiation**: EPA RadNet hourly CSV archives (2017–2025) for 5 pilot stations chosen
  for climate diversity and data completeness: Birmingham AL (KBHM), Washington DC
  (KDCA), San Diego CA (KSAN), Dallas TX (KDFW), Tampa FL (KTPA). Channels R02–R09
  plus exposure rate where reported. Complete-channel records only; gaps are labeled
  `unobserved`, never interpolated.
- **Weather**: NOAA NCEI Global-Hourly (ISD) from the adjacent airport ASOS
  (hourly AA1 liquid precipitation, temperature, pressure, dew point). Precipitation
  quality codes restricted to `1` and `5`; suspect codes excluded.
- **Merge**: both streams rounded to the nearest UTC hour (RadNet collection times and
  METARs both land at :50–:55), reindexed onto continuous 78,888-hour calendar grids.
  Result: 324,550 synchronous observation hours network-wide; 15,606 rain hours with
  concurrent RadNet observations.

Key empirical calibration results (Phase 1–3):

- Rain→radiation cross-correlation peaks at lag 0 to +1 h ($r = 0.31$–$0.39$) — the
  washout response is immediate, consistent with hourly accumulation and ~20–30 min
  progeny decay.
- Rain-hour surge distribution (15,602 fully observed rain hours): median +6.9%,
  90th percentile +41.4%, maximum +197% of local dry baseline.
- Empirical dose coupling: $\Delta\text{dose} = k_{\text{dose}} \cdot \Delta\text{CPM}$,
  $k_{\text{dose}} \in [0.0119, 0.0239]$ nSv/h per CPM by station (pooled 0.01357,
  $R^2 = 0.79$ over 5,623 rain hours).
- Washout spectral share (ratio-of-sums over 5,485 substantial rain hours):
  R07+R08 high-energy = 5.58% [5.52, 5.65] — overlapping the modeled fresh-reactor
  mixture (4.54%), so high-energy share alone cannot separate reactor plumes from
  washout.

## 3. Methods

### 3.1 Baseline (current practice)

Three rule families on real data: global dry-baseline σ, rolling 7-day σ, and global
exposure-rate σ, at $k \in \{3, 4, 5\}$. Alarm episodes are clustered on the calendar
grid; rain coincidence is scored in 3 h and 6 h windows.

### 3.2 Synthetic injection benchmark

Injections accumulate particulate activity on the filter (passage phase with shape
families `step`, `linear_ramp` for training; `sigmoidal`, `exponential` for test), then
persist and decay ($^{131}$I-dominated $\lambda = 0.00360\ \mathrm{h}^{-1}$) until the
**real** filter replacement step drop ends the event — 450/450 injections are
synchronized to empirically detected filter changes, with a strict 48 h non-overlap
buffer. Five release scenarios ($^{137}$Cs, $^{131}$I, $^{60}$Co, fresh-reactor
$^{137}$Cs/$^{134}$Cs/$^{131}$I, mixed) are balanced to exactly 20% in each of six
environments (dry/rain × train/standard/stress). Magnitude bands: A 250–600 CPM
(subtle), B 700–1200 CPM (moderate, test-interpolation), C 1400–2500 CPM (large).
The test split (2023–2025) is calendar-disjoint from training (2017–2022) and uses
unseen shapes and magnitude bands. 100 test injections are rain-coincident hard cases.
Channel shares per injection are stored in the catalog and drawn from broad Dirichlet
dispersions around semi-empirical NaI(Tl) working templates.

### 3.3 Models and features

Four LightGBM multi-class tiers (`normal` / `radon_washout` / `fission_product`):
Tier 1 gross-rate dynamics (12 features); Tier 1b gross + weather (29); Tier 2 gross +
spectral ratios (30); Tier 3 full fusion (48). A 2-layer MLP (calibrated and raw) is the
neural comparison. Features are causal (past-only windows): channel shares, photopeak
ratios (R05/R03, R03/R07, R05/R07), high-energy share, multi-scale diffs and rolling
stats, precipitation accumulations, pressure tendencies, and physics-motivated
interaction terms ($Z_{\text{gross}}/(\sqrt{P_{3h}}+0.1)$).

### 3.4 Evaluation protocol

- Operating thresholds $\tau^*$ are selected on a 2021–2022 validation fold only and
  frozen before touching the 2023–2025 test split.
- Dual-series design: detection measured on injected series; false alarms measured on
  unmodified clean background, in episodes per station-year with
  $N_{\text{obs}}/8766$ denominators.
- Event detection uses the net-alarm criterion (alarm on injected series and not on the
  clean series) in `train_and_evaluate_rigorous.py`; `evaluate_phase5_protocol.py` uses
  a plain any-alarm criterion **and selects its operating thresholds on the test split
  itself** (confirmed bug, see Appendix B.3) — its `eval_*` tables are therefore
  optimistic and are quoted here only where explicitly labeled. The `rigorous_*` tables
  (validation-frozen thresholds, net-alarm) are the authoritative numbers.
- 20-seed replication (42–61), paired t and Wilcoxon tests, station-month block
  bootstrap ($B = 1000$), matched-detection comparisons, leave-one-station-out (LOSO),
  ±10% gain-drift and ±20% spectral-template sensitivity sweeps.

## 4. Results

### 4.1 The baseline problem (real data, Phase 2)

At 3σ (rolling 7-day), alarms per station-year: Birmingham 52.0, Washington 52.9,
Dallas 62.7, San Diego 19.9, Tampa 6.6. Rain coincidence within 3 h among those alarms:
Washington 86.5%, Dallas 69.1%, Birmingham 62.1% (San Diego 0% — its 3σ alarms are dry
episodes instead). At 5σ the total drops (12.6–26.8/yr in humid stations) but the
remaining alarms are even more rain-dominated (86–97%). Fixed thresholds trade alarm
volume against blindness; they do not solve the discrimination problem.

### 4.2 Detection at matched false-alarm budget (synthetic test events)

20-seed mean ± SD on the 2023–2025 split, thresholds frozen on 2021–2022
(`rigorous_benchmark_summary.csv`, 95% detection target row):

| Model | Realized detection | Clean FA / stn-yr | 95% CI (FA) |
| --- | --- | --- | --- |
| Tier 1 gross GBDT | 92.10 ± 0.55% | 168.62 ± 20.57 | [153.1, 186.0] |
| Tier 1b gross + weather | 92.82 ± 1.10% | 68.21 ± 9.75 | [57.9, 78.8] |
| Tier 2 gross + spectral | 92.48 ± 1.38% | 8.52 ± 6.23 | [5.4, 12.4] |
| Tier 3 full fusion | 92.08 ± 1.77% | 4.22 ± 1.29 | [2.6, 6.2] |
| Tier 3 MLP | 90.22 ± 2.84% | 58.52 ± 16.18 | [43.4, 76.9] |

At matched ~92% detection, a rolling σ rule needs $k = 0.75$ and then fires
**303.11 FA/stn-yr** — roughly 70× the Tier 3 rate. The conventional $k = 3$ setting
detects only 52–63% of events (criterion-dependent) at 82.95 FA/stn-yr.

### 4.3 What each modality contributes

- **Spectrometry is the workhorse**: Tier 2 removes ~95% of Tier 1's false alarms at
  the same detection (168.62 → 8.52). Physically consistent: washout is R02/R03-heavy
  (42.3%/33.7% ratio-of-sums) and decays in hours, while filter-trapped $^{137}$Cs
  elevates R05 persistently (see `INJ_0067` walkthrough, `PHASE_REPORT.md` §6).
- **Weather on gross-only detectors** (Tier 1b vs Tier 1): −59.5% FA (paired
  $t = 18.4$, $p < 10^{-15}$, 20/20 seeds). Weather data alone can gate most washout.
- **Weather on top of spectrometry** (Tier 3 vs Tier 2): mean −50.5% but median only
  −14.0% ($4.69 \to 4.03$); paired $t = 3.17$, $p = 0.005$; Wilcoxon $p = 0.019$;
  bootstrap 95% CI on the difference [1.30, 7.65]. The honest reading: weather fusion
  mainly suppresses variance and storm-time outliers ($\sigma$: 6.23 → 1.29) rather
  than shifting the typical false-alarm rate. At the 98% target both tiers saturate
  (~187–189 FA/yr, difference not significant) — the tail regime is threshold-limited.
- **MLP** under frozen thresholds plateaus at ~90% detection with 14–21× more false
  alarms than GBDT; isotonic calibration preserves ranking and does not close the gap.

### 4.4 Timing

Net-alarm detection deadlines (Tier 3): 69.0% within 6 h, 88.1% within 12 h, 91.5%
within 24 h, 92.0% by event end; median delay 4.0 h. Rain-coincident plumes: 96%
total detection, 93% within 12 h. Weather fusion does not accelerate detection; it
keeps sensitivity during storms without tripping on washout.

### 4.5 Where the model is weak (the realistic picture)

- **Stress dry regime** (8–20 h plumes, 250–600 CPM, no rain): 78–82% detection —
  roughly one miss in five. Short subtle plumes in quiet weather are the honest
  hard case.
- **LOSO transfer** (train on 4 stations, test on the 5th, thresholds frozen):
  0 clean false alarms (rule-of-three upper bounds 1.05–1.70 FA/yr) but Band A
  detection drops to 60% overall (45% San Diego). Scenario breakdown across 120
  held-out events: $^{137}$Cs 100%, reactor mix 91.7%, mixed 75%, $^{60}$Co 62.5%,
  $^{131}$I 58.3%. Single-line low-energy releases transfer worst.
- **Validation→test gap**: validation-fold thresholds are optimistic (e.g. Tier 3 at
  the 90% target realizes 86.7% on test) because validation draws on training shape
  families. Reported test numbers use frozen validation thresholds throughout; no test
  retuning was performed.
- **Gain drift**: +10% upward channel shift raises clean FA from 4.27 to 19.06/yr
  (5-seed sweep); field gain stabilization (e.g. tracking the $^{40}$K 1461 keV line)
  is operationally necessary.
- **Spectral template sensitivity**: ±20% photopeak scaling moves detection 86.5–94.7%
  — models do rely on spectral structure, and template misspecification costs ~2–4
  points per 10%.

### 4.6 Uncertainty

Block-bootstrap 95% CIs for Tier 3 (from `rigorous_benchmark_summary.csv`,
validation-frozen protocol): detection 92.1% [88.5, 95.2], FA 4.22
[2.56, 6.22], median delay 4.0 h [4.0, 5.0]. Distributions from the alternate
protocol are in `eval_benchmark_uncertainty_summary.csv` /
`eval_bootstrap_distributions.csv` (any-alarm, test-tuned thresholds — see
Appendix B.3; not directly comparable).

## 5. Discussion

**The operational claim is supported, with scope.** Weather-fused multi-channel
monitoring reduces rain-driven false alarms by two orders of magnitude relative to
detection-matched fixed thresholds on this five-station benchmark, and the reduction
holds across 20 seeds, station-month block bootstrap, and LOSO transfer with frozen
thresholds. The mechanism is physical, not a statistical artifact: washout dominates
low channels and decays within hours; filter-retained fission products persist across
the 4-day filter cycle and shift mid-energy shares.

**Spectrometry first, weather second.** The ablation ordering is unambiguous: most of
the discrimination comes from the eight-channel ratios already present in RadNet data.
Weather fusion is a robustness layer — statistically significant on average, but its
typical-case benefit is modest and its real value is storm-time variance suppression.
For networks without spectrometry (GM tubes, total-count scintillators), weather
features are much more valuable (−59.5% FA at matched detection).

**Synthetic ground truth is the central limitation.** All detection rates are for
physically parameterized synthetic plumes on real backgrounds. The parameter ranges
were calibrated from data and literature (magnitude bands from empirical rain surges;
spectra from semi-empirical NaI templates; dose coupling from station regressions), and
train/test parameter families are disjoint, but a real release will not be distributed
like our catalog. In particular: (i) real plumes arrive with transport-driven
correlations between nuclide mix, magnitude, and weather that our balanced design
deliberately removes; (ii) gaseous radioiodine passes the particulate filter and is out
of scope; (iii) the stress-rain set still has median 49 h durations — sub-12 h plumes
during downpours are unmeasured.

**Remaining confounders in the real data.** The `radon_washout` label is a rule-based
heuristic (rain + 2σ excess) and is partly circular; it is used only as an auxiliary
diagnostic. Station-to-station background differences (Tampa's low counts and shifted
R02 share; San Diego's negative rain–radiation correlation) are documented empirically
but not causally explained — see open questions in `QUESTIONS.md`.

**What would make this deployable:** (1) validation against real episodic data —
regional tracer releases, reactor emission records, or expert-adjudicated alarm logs;
(2) per-site gain-stabilization monitoring; (3) conformal or otherwise calibrated
false-alarm guarantees (stretch goal in `PROJECT_SPEC.md`); (4) wind-informed
multi-station confirmation to catch low-energy single-nuclide releases that transfer
worst.

## 6. Reproducibility

- Seeds: `configs/config.yaml` (42; replication seeds 42–61).
- Frozen artifacts: `data/processed/` summary tables and `synthetic_injection_catalog.csv`
  are the reference outputs; `station_dry_baselines_train_only.json` holds train-only
  baselines (no test leakage).
- Raw and intermediate panel files are not archived (`data/raw/` policy); regeneration
  path is `run_pipeline.sh`, which documents and enforces the required inputs per phase.
- Known convention difference: net-alarm vs any-alarm event criteria between the two
  evaluation scripts (§3.4); every quoted detection number names its source table.

## 7. Keywords

radon washout; gamma spectrometry; NaI(Tl); radiological air monitoring; anomaly
detection; false alarm reduction; gradient boosting; weather data fusion; synthetic
event injection; environmental radiation

## 8. References

1. U.S. EPA. *RadNet Air Data*. https://www.epa.gov/radnet/radnet-air-data
2. NOAA NCEI. *Global Hourly (ISD / Global-Hourly)*. https://www.ncei.noaa.gov/data/global-hourly/
3. NOAA Federal Climate Complex (2018). *Integrated Surface Database (ISD) Format Document*, Data Version 8.
4. Knoll, G. F. (2010). *Radiation Detection and Measurement* (4th ed.). Wiley.
5. Vieira, C. L. Z., et al. (2019). Short-term effects of particle gamma radiation activities on pulmonary function in COPD patients. *Environmental Research*, 175, 221–227.
6. Huang, S., et al. (2020). Short-term exposure to ambient particle gamma radiation and mortality in the US Medicare population. *Environmental Research*, 182, 108995.
7. Masson, O., et al. (2011). Tracking of airborne radionuclides from the damaged Fukushima Dai-ichi nuclear reactors by European networks. *Environmental Science & Technology*, 45(18), 7670–7677.
8. Steinhauser, G., Brandl, A., & Johnson, T. E. (2014). Comparison of the Chernobyl and Fukushima nuclear accidents: a review of the environmental impacts. *Science of the Total Environment*, 470–471, 800–817.
9. U.S. EPA (1993). *EPA Map of Radon Zones* (EPA-402-R-93-071).
10. NCRP Report No. 50 (1976). *Environmental Radiation Measurement*.

---

## Appendix A. Provenance of headline numbers

| Claim | Source table |
| --- | --- |
| 324,550 synchronous hours | `merge_audit_summary.csv` (sum of per-station counts) |
| Baseline episodes/yr & rain coincidence | `baseline_threshold_evaluation.csv` |
| Surge distribution | `rain_washout_surge_distribution.csv` |
| $k_{\text{dose}}$ regression | `dose_rate_cpm_regression_summary.csv` |
| Washout spectral shares (R07+R08 = 5.58%) | `rain_washout_spectral_shares.csv` |
| 450 injections, balance, buffer, filter sync | `synthetic_injection_catalog.csv`, `injection_catalog_audit_summary.csv` |
| 20-seed benchmark | `rigorous_benchmark_summary.csv`, `rigorous_seed_level_results.csv` |
| Paired tests | `rigorous_seed_statistical_tests.csv` |
| Matched-detection baselines (303.11 FA/yr at k=0.75) | `rigorous_matched_detection_comparison.csv` |
| Detection deadlines | `rigorous_stratified_deadlines.csv` |
| LOSO | `rigorous_loso_summary.csv`, `rigorous_loso_stratified_detection.csv` |
| Gain drift / template sensitivity | `rigorous_gain_drift_evaluation.csv`, `rigorous_template_sensitivity.csv` |
| MLP comparison | `eval_mlp_calibration_comparison.csv` |
| Bootstrap CIs (authoritative) | `rigorous_benchmark_summary.csv` |
| Bootstrap CIs (alt. protocol, test-tuned) | `eval_benchmark_uncertainty_summary.csv` |

## Appendix B. Corrections applied at Phase 6 handoff (2026-10-06)

1. `HANDOFF.md` and the incoming handoff prompt stated **2,000 injections
   (1,000 train / 1,000 test; 150+150 hard-case rain)**. The frozen catalog contains
   **450 injections (250 train / 200 test; 75 train + 100 test rain-coincident)**.
   `PHASE_REPORT.md`, `DECISIONS.md`, `QUESTIONS.md`, and all CSVs consistently
   describe 450; the handoff summary was the outlier and has been corrected.
2. Phase 4 headline text ("96% reduction", "flawless detection") is superseded by the
   Phase 5 protocol numbers quoted here; the Phase 4 section in `PHASE_REPORT.md`
   carries an explicit supersession notice and is retained as history.
3. Detection criteria and threshold selection differ between the two Phase 5 scripts:
   `evaluate_phase5_protocol.py` uses an any-alarm criterion **and tunes its operating
   thresholds on the test split** (`src/evaluate_phase5_protocol.py:245–266`), which
   inflates its `eval_*` numbers; `train_and_evaluate_rigorous.py` freezes thresholds on
   the 2021–2022 validation fold and uses the stricter net-alarm criterion. The
   `rigorous_*` tables are the trustworthy ones. Before publication submission:
   (a) re-select `evaluate_phase5_protocol.py` thresholds on the validation fold,
   (b) unify the detection criterion. A code audit report with all reproducible bugs
   found at handoff is maintained alongside this report (see `reports/BUG_AUDIT.md`).
