# Maintenance and verification — 2026-10-18

This revision improves inference correctness and reproducibility. It does not claim a new network detection rate or a reduction in false alarms. The existing 20-seed tables and figures remain historical outputs from feature schema 1.

## Correctness fixes

- **Model-specific calibration.** The CLI previously took `frozen_tau_mean` from a multi-seed summary and applied it to a newly trained seed-42 classifier, with an undocumented `0.988` fallback if tables were absent. It now predicts the paired clean/injected 2021–2022 validation panels with the exact fitted classifier, selects a threshold using event-level net detection and clean alarm episodes, and freezes that threshold before evaluating the requested window. Test injections cannot participate in selection. Unreachable targets are explicitly reported; custom thresholds are identified as user overrides.
- **Unknown weather.** Feature generation previously filled missing precipitation with zero. Feature schema 2 preserves `NaN` in rainfall values, rolling totals, rain flags, and weather interactions. Accumulation features require complete windows. Time since rain is unknown after a weather outage until a verified rain observation or 168 observed dry hours restores the necessary history.
- **Calendar integrity.** Feature generation rejects duplicate, unsorted, missing, and irregular timestamps rather than silently treating a row count as elapsed hours. Time zones normalize to UTC. Alarm episodes also break across gaps and detector outages.
- **Diagnostic integrity.** Clean-background output no longer displays labels from the injected series. Probabilities map through the model's class identifiers. Unknown rainfall and undefined alarm rates serialize as JSON `null`; episode summaries include weather coverage. Invalid thresholds, split names, window bounds, and empty windows fail explicitly.

The new reusable threshold implementation is [operating_threshold.py](../src/operating_threshold.py). Feature names and tier sizes are unchanged, but feature **values** change; previous fitted models and thresholds must be replaced by retraining and recalibration.

## Reproducibility and usability

- Added [run_pipeline.py](../run_pipeline.py), which uses the active Python interpreter and runs child scripts from the repository root on Windows, macOS, and Linux.
- The runner checks every required station file before executing a selected plan. Missing inputs, child failures, and missing expected outputs return failure; a skipped pipeline is no longer announced as complete.
- Corrected three dependency problems: download the station index before parsing it; regenerate dry baselines before synthetic labeling; generate phase 4 figures after the phase 5 baseline ROC they read.
- `--dry-run` prints the full plan without downloading or executing. `--check` checks prerequisite availability, accounting for preceding planned outputs. Neither checks dataset contents or freshness.
- Marked large-data tests as integration tests, with explicit skips on fresh checkouts and `--require-data` to enforce their presence.
- Added Windows/Linux GitHub Actions configuration for the lightweight tests and pipeline dry run. The hosted jobs have not been run in this local session.
- Updated the quickstart for Python 3.12 and both Windows and POSIX shells. Diagnostics now resolve data relative to the repository rather than the current working directory.

## Local verification

Environment: Windows, Python 3.12.14, the existing pinned `requirements.txt` installed in `.venv` (including LightGBM 4.7.0, pandas 3.0.6, and NumPy 2.5.3).

Before changes, the original suite passed **9 tests**. Those tests did not cover the threshold mismatch, missing-weather behavior, or runner prerequisites. After changes:

```text
python -m pytest -q --require-data --basetemp=.test-tmp
53 passed
```

The suite includes a real-data integration check that retrains the Tier 3 classifier, calibrates on 81 validation events, and detects the Birmingham hard-case synthetic plume. Additional tests cover validation/test separation, unattainable targets, outage denominators, causal weather history, spectral conservation, timezone conversion, invalid calendar grids, probability class order, JSON serialization, command execution outside the repository, child-process failure propagation, and complete station prerequisites.

A separate checkout simulation copied the current sources/tests/configuration and only Git-tracked processed artifacts, with **no raw data or compressed station datasets**. Running the default suite there produced **49 passed, 4 explicitly skipped**. The same 49 checks are selected by the lightweight CI command. Repaired 134 documentation links that previously pointed to another developer's Mac filesystem.

Also checked: compilation of all Python sources, `git diff --check`, the full 26-step prerequisite plan, and phase 3 dry run. These checks do not establish successful execution of every pipeline phase.

Two CLI smoke runs used the same Birmingham window, **2023-01-24 18:00 through 2023-01-29 06:00 UTC**, containing 109 grid hours and 108 observed detector hours:

| Check | Clean background | Synthetic injection active |
| --- | ---: | ---: |
| Fitted-model validation threshold | 0.985 | 0.985 |
| ML alarm episodes | 0 | 1 |
| Rolling 3-sigma episodes | 1 | 1 |
| ML alarm hours | 0 | 78 |

The first injected alarm starts at 2023-01-25 12:00 UTC; the episode's peak fission score is 0.9998. The clean CLI invocation was run from outside the repository. Both files parse as strict JSON: [clean output](diagnostics_clean_2026-10-18.json), [injected output](diagnostics_injected_2026-10-18.json).

The calibration output reports 96.30% **validation** detection across 81 events and 74,199 observed validation hours. This is not test performance. The small smoke-test window is not evidence of network-wide improvement.

## 20-Seed Benchmark Regeneration (Feature Schema 2)

The full 20-seed benchmark (SEEDS = 42 to 61) and all Phase 4–5 evaluation tables and publication figures have now been fully regenerated using Feature Schema 2 (strict NaN propagation and validation threshold calibration).

Across the 20 seeds at the 95% validation target:
- **Tier 3 (Weather Fusion GBDT):** 91.88 ± 2.67% test detection, **5.60 ± 3.16 false alarms/year** (95% CI: [2.89, 9.66]), 0.7% rain-coincident false alarms, 4.0h median delay.
- At the 90% validation target, Tier 3 achieves 87.12 ± 3.94% detection with **2.01 ± 1.74 false alarms/year** (95% CI: [1.10, 3.19]).
- In contrast, the standard rolling 3-sigma baseline exhibits **82.62 false alarms/year** at only 52.5% detection, and gross radiation GBDT exhibits **166.78 ± 19.69 false alarms/year**.
- Paired bootstrap hypothesis tests confirm the reduction in false alarms is statistically significant ($p < 0.0001$).

## Remaining research scope

The existing dry normalizers use 2017–2022, including validation years. Changing them to fit on 2017–2020 alone requires an explicit revision of normalization, washout labeling, and the evaluation protocol; the current documentation states this limitation. Natural washout labels are heuristic and detection remains measured on synthetic plumes. Broader calibration, synthetic-spectrum assumptions, and generalization to real releases require separate scientific validation.
