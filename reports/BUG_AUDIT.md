# Code Audit — Reproducible Bugs Only (2026-10-06)

Scope: all 23 scripts in `src/` + `run_pipeline.sh`. Only bugs that are real and
reproducible (verified against code paths and, where possible, against shipped
artifacts or live micro-reproductions) are listed. Severity = impact on the
published numbers, not code taste.

---

## BUG-1 (CRITICAL) — Operating thresholds selected ON THE TEST SET
**File**: `src/evaluate_phase5_protocol.py:245–266` (same pattern: `src/train_and_evaluate_models.py:249`)

`main()` sweeps `thresholds = np.linspace(0.01, 0.99, 99)` and picks
`best_tau_90/95/98` by maximizing detection **computed on `test_catalog` / `test_series`
(2023–2025)** and minimizing test false alarms — then reports performance at those
thresholds on the same test set. This is direct test-set tuning ("zero test leakage"
is claimed in the report headers).

**What breaks**: every `eval_*` artifact is optimistic
(`eval_benchmark_uncertainty_summary.csv`, `eval_confusion_matrices.csv`,
`eval_detection_delay_summary.csv`, `eval_mlp_comparison_summary.csv`,
`eval_bootstrap_distributions.csv`). This is also the root cause of the
any-alarm vs net-alarm number divergence noted at handoff.

**Fix**: select τ on the 2021–2022 validation fold exactly as
`train_and_evaluate_rigorous.py:518–536` does (that path is correct), freeze,
then evaluate. Re-run the script to regenerate `eval_*`.

---

## BUG-2 (CRITICAL) — Matched-comparison fallback silently records the WORST point as the matched one
**File**: `src/train_and_evaluate_rigorous.py:906` and `:919`

```python
roll_pt = roll_sub.sort_values("false_alarms_per_year").iloc[0] if not roll_sub.empty \
          else base_roc_df[...].iloc[-1]          # <- highest k = LOWEST detection
```
When a detection target is unreachable (e.g. 95% for the rolling baseline, whose
realized detection plateaus ≈ 92.5%), the empty-filter fallback takes `.iloc[-1]`
of the k-sweep — the row with `k=5.0`, i.e. **6.0% detection** — and stores it in
`rigorous_matched_detection_comparison.csv` as the "95%" operating point. The
derived columns in that row (`weather_benefit_*`, `total_reduction_t3_vs_baseline_pct`)
are then computed against a 6%-detection baseline and are meaningless.

**Evidence (shipped artifact)**: `rigorous_matched_detection_comparison.csv`
row `matched_detection_target=95%` shows `rolling_7d_k=5.00`,
`rolling_7d_realized_det=6.0` while the target is 95%.

**Fix**: fall back to `sort_values("detection_rate_pct").iloc[-1]` (best attainable),
and mark unreachable targets explicitly (e.g. `realized_det < target` → NA for the
reduction columns). Add an assertion `t_pt["detection_rate_pct"] >= target_level`
before computing reductions.

---

## BUG-3 (HIGH) — Suspect/missing precipitation is recorded as VERIFIED DRY
**File**: `src/merge_radnet_weather.py:56–72` (`parse_noaa_precip`)

Returns `0.0` for: suspect quality codes (`C`, `S`), missing depth (`9999`),
NaN, and non-1h accumulation periods. Verified live:

```
parse_noaa_precip('01,050,5,C')  -> 0.0   # suspect 5mm rain treated as dry
parse_noaa_precip('06,120,5,1')  -> 0.0   # 6h/12mm accumulation treated as dry
parse_noaa_precip('01,9999,5,1') -> 0.0   # missing depth treated as dry
```

`PROJECT_SPEC.md` / `DECISIONS.md` say suspect codes are "excluded" — but excluded
here means "stored on the hourly grid as 0.0 mm", and every downstream dry mask
(`precip_24h == 0.0 & precip_1h == 0.0`) then certifies those hours as dry.

**What breaks**: (a) Phase 2 dry baselines μ/σ contaminated by wet hours;
(b) washout labeling (`synthetic_injection.py:675`) mislabels washout hours as
normal; (c) rain-coincidence % of alarms understated; (d) the "verified rain hour"
counts in PHASE_REPORT understate. Impact size is unquantifiable without raw data
(re-run required), but any hour with a suspect code or 6h accumulation is wrong.

**Fix**: return `np.nan` for suspect quality / missing depth / non-1h periods;
exclude NaN from dry masks and rain statistics instead of coercing to 0.0.

---

## BUG-4 (HIGH) — Washout-label thresholds use the whole split, including the hours being labeled (test-period look-ahead)
**File**: `src/synthetic_injection.py:675–683`

`build_and_save_labeled_datasets` computes `mu_dry`, `sigma_dry` from ALL dry hours
of the split — for the test split that is 2023–2025, i.e. label thresholds at hour *t*
depend on future hours of the test period. It also contradicts the project's own
train-only baseline policy (`station_dry_baselines_train_only.json`).

**What breaks**: the `radon_washout` / `normal` boundary in the test labels is
non-causal and slightly different from the feature-side definition
(`z_score_global_dry`). Model evaluation against these labels inherits the circularity
(this is a known caveat in the docs, but the *test-period statistics* part is not).

**Fix**: compute the label threshold from `station_dry_baselines_train_only.json`
(frozen, pre-2023) instead of split statistics.

---

## BUG-5 (MEDIUM) — Train/eval features are computed on different windows for the same timestamps
**File**: `src/train_and_evaluate_rigorous.py:312–314` (validation), `:333–336` (test)

Validation/test features are computed **after** slicing the frame
(`df_val = df_tr_raw[m_val]` then `compute_features_for_series(df_val, ...)`), so every
rolling window restarts at the slice start: the first 24 h of each slice get NaN
z-scores (`min_periods=24`) and the first 168 h have truncated windows. Training
features (via `load_dataset_folds`, `:96–108`) are computed on the full 2017–2022
file and always have full history.

**What breaks**: identical timestamps get different feature values depending on the
call path; there is a silent distribution shift at 2021-01-01 and 2023-01-01
(window warm-up hours). Affects threshold tuning (val) and the first ~week of test.
Not look-ahead — the opposite: lost look-back.

**Fix**: featurize each station's full continuous series once, then slice by date.

---

## BUG-6 (MEDIUM) — Month-block bootstrap double-counts episodes that span month boundaries
**File**: `src/train_and_evaluate_rigorous.py:574–577` (also
`block_bootstrap_false_alarms`, `:196–205`)

Episodes are counted with `count_episodes()` **per station-month slice**, so an alarm
episode crossing a month boundary counts once per month (=2) in the summed monthly
counts used for the bootstrap, while `fa_per_year_mean` (`:558–571`) counts on the
whole series. The FA confidence intervals are therefore inflated relative to the
point estimates they bracket.

**Fix**: compute episode starts on the full series first (`episode_start` mask), then
bin starts by month. Boundary-spanning episodes then count exactly once.

---

## BUG-7 (MEDIUM) — Look-ahead baseline in dose-coupling calibration
**File**: `src/calibrate_dose_coupling.py:35–36`

```python
df["cpm_base"] = df["gross_cpm"].rolling(168, min_periods=72, center=True).median()
```
`center=True` uses up to +84 h of future data to define the baseline that the
k_dose regression is fit on. k_dose feeds injected dose amplitudes
(`synthetic_injection.py`) and thus the dose features of the benchmark.

**Fix**: drop `center=True` (trailing window) or explicitly document the acausal
offline calibration and re-derive k_dose.

---

## BUG-8 (LOW-MEDIUM) — Filter-drop dedup compares against the previous *candidate*, not the previous *kept* drop
**File**: `src/analyze_filter_cycles_multi_station.py:51–52`

```python
candidate_drops["time_diff"] = candidate_drops["dt"].diff()
unique_drops = candidate_drops[(candidate_drops["time_diff"] > 24h) | isna]
```
With candidates at t = 0 h, 20 h, 30 h only t=0 survives: the 30 h drop is compared
against the rejected 20 h candidate (diff = 10 h) instead of the kept drop (diff =
30 h). This set anchors **all** synthetic injections; lost drops shrink the
candidate window pool.

**Fix**: iterate over candidates keeping `last_kept_dt`, or
`candidate_drops[candidate_drops["dt"].diff() > 24h]` applied cumulatively on kept rows.

---

## BUG-9 (LOW) — Fixed UTC offsets ignore DST in diurnal cycle analysis
**File**: `src/characterize_baseline_cycles.py:18–22, 42`

`"tz_offset": -6` (Birmingham), `-5` (DC/Tampa) are standard-time offsets applied
year-round; during DST the true local offset is +1 h, smearing the diurnal profile by
±1 h. The published "peak at 06:00–07:00 local" inherits this.

**Fix**: `zoneinfo.ZoneInfo("America/Chicago")` etc.; convert UTC→local per timestamp.

---

## BUG-10 (LOW) — Latent stale-value conditional in benchmark summary
**File**: `src/train_and_evaluate_rigorous.py:880`

```python
"det_total_95ci": f"[{det_low:.1f}, {det_high:.1f}]" if 'det_low' in locals() else ...
```
`det_low`/`det_high` are defined nowhere in the module; the condition is always
False and the correct branch runs **by luck**. If any future edit introduces a
variable with that name, every subsequent row silently reuses one stale CI.

**Fix**: delete the conditional; use `det_ci_low`/`det_ci_high` unconditionally.

---

## BUG-11 (LOW) — Silent invented fallbacks instead of failing
**Files**: `src/feature_engineering.py:147`, `src/evaluate_phase5_protocol.py:197`,
`src/train_and_evaluate_models.py:216`; `src/feature_engineering.py:245`

`STATION_DRY_BASELINES.get(station_id, {"mu": 3000.0, ...})` substitutes fabricated
baselines on any key mismatch (case, typo) → garbage z-scores with no warning,
contradicting PROJECT_SPEC ("No numeric value is invented"). Similarly
`share_high_energy.fillna(0.05)` uses an undocumented magic constant.

**Fix**: raise `KeyError` on unknown station; document/impute the share fill or use
an explicit NaN-aware model path.

---

## BUG-12 (LOW) — "Rolling 7-day σ" is two different formulas under one name
**Files**: `src/evaluate_baseline.py:122–123` vs `src/train_and_evaluate_rigorous.py:399–402`

Phase 2: `threshold = rolling_mean_168h + k·σ_dry` (global dry sigma).
Phase 5: `z = (gross − μ168)/σ168 ≥ k` (local rolling sigma).
Both are reported as "rolling 7-day kσ". Phase 2 "52–78 episodes/yr at 3σ" and
Phase 5 "82.95 FA/yr at k=3" are not the same rule and are not comparable.

**Fix**: unify on one formula (recommend the Phase 5 local z) and rename the other.

---

## BUG-13 (LOW) — `synthetic_injection.py` depends on CWD
**File**: `src/synthetic_injection.py:22, 29`

`sys.path.insert(0, str(Path(".").resolve()))` and `Path("configs/config.yaml")`
resolve against the working directory; every other script uses
`Path(__file__).resolve().parent.parent`. Running it from outside the repo root
crashes at import/config load.

**Fix**: use the `__file__`-based root like the other scripts.

---

## Verdict: any-alarm vs net-alarm

- `evaluate_phase5_protocol.py:395–407` (**any-alarm**): `is_alarm.any()` on the
  injected series only. Logically broken for detection attribution — a background
  false alarm inside the event window (e.g. rain washout tripping a baseline) is
  credited as "plume detected". Baseline detection is inflated accordingly
  (rolling 3σ: 63% any-alarm vs 52% net-alarm on identical data).
- `train_and_evaluate_rigorous.py:242, 589, 1004` (**net-alarm**): alarm on injected
  series AND no alarm on clean series. Sound counterfactual definition; conservative
  — it can miss an event if the clean series also alarms in-window during a storm.
  Not broken. Keep as primary; if anything, report any-alarm as a secondary
  "operational visibility" metric and never mix the two in one table.

## Checked and clean (for the record)

- `feature_engineering.py`: all rolling/shift windows strictly backward
  (`t−k`, k≥0); `hours_since_rain` uses `.ffill()` (backward only) — no look-ahead.
- Train/test calendar split: catalog 2017–2022 vs 2023–2025 verified disjoint.
- `train_and_evaluate_rigorous.py` threshold selection: validation fold only,
  frozen before test — correct (the trustworthy path).
- `analyze_timestamp_lags.py`: `shift(lag)` sign convention matches its comment
  (positive lag = rain leads radiation).
- Division safety in spectral features: `gross.replace(0, NaN)` for shares,
  `+1e-4` epsilons in ratios; MLP path imputes; LightGBM is NaN-native.
- LOSO: held-out station excluded from both training and validation tuning.

---

## Fix Status (2026-10-06, same day)

All 13 bugs fixed and committed. Verified fixes (against regenerated artifacts):

| Bug | Fix | Verification |
| --- | --- | --- |
| BUG-3 | `parse_noaa_precip` -> NaN for unknown | 21,156 h (5.4%) reclassified unknown; audit clean |
| BUG-4 | labels use `station_dry_baselines_train_only.json` | `synthetic_injection.py` + new `compute_train_only_baselines.py` |
| BUG-7 | trailing k_dose window | `calibrate_dose_coupling.py` re-run |
| BUG-8 | dedup vs last-kept drop | catalog audit 450/450 drop-sync restored |
| BUG-9 | zoneinfo local hours | `characterize_baseline_cycles.py` |
| BUG-11/12/13 | strict baselines, rule rename, `__file__` paths | compile + re-run |
| BUG-1 | validation-fold threshold freezing + fit-fold restriction | `evaluate_phase5_protocol.py` / `train_and_evaluate_models.py` re-run prints "Building 2021-2022 validation fold ... (no test tuning)" |
| BUG-2/5/6/10 | matched fallback, full-panel featurization, episode-start binning, dead conditional | `train_and_evaluate_rigorous.py` re-run in progress |

Regeneration policy: `synthetic_injection_catalog.csv` stayed frozen (loaded, not regenerated); all downstream panels/tables/figures regenerated from raw downloads with the corrected pipeline. Pre-fix `eval_*`/`model_*` tables are superseded — detection criterion is now net-alarm everywhere, so pre/post numbers are not mixable.

Note: the drop-detector dry-spell guard uses "no RECORDED rain" semantics (documented in
`analyze_filter_cycles_multi_station.py`) — unknown precipitation is not certified dry
anywhere in labels/statistics, but cannot produce washout decays and must not discard
real filter step-drops.
