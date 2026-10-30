# Repository Cleanup & Control Experiment Report

## 1. Baseline vs Final Test Results
- Python environment: Python 3.14.7 (macOS darwin).
- Task 0 Baseline lightweight tests: 49 passed, 4 deselected in 4.30s (0 failed).
- Task 8 Final lightweight tests: 49 passed, 4 deselected in 2.80s (0 failed).
- Task 6 Integration tests (with local data): 53 passed in 9.10s (0 failed).

## 2. What Changed
- **Moved to `docs/` and `docs/dev-notes/`**: `DECISIONS.md` -> `docs/`; `HANDOFF.md`, `QUESTIONS.md`, `PHASE_REPORT.md`, `DATA_LOG.md`, `PROJECT_SPEC.md`, `reports/BUG_AUDIT.md`, `reports/MAINTENANCE_2026-10-18.md`, `reports/PHASE6_REPORT.md` -> `docs/dev-notes/`.
- **Created docs/README.md**: Documentation index pointing to decisions and notes.
- **Header additions**: Added historical working note banners to `HANDOFF.md` and `PHASE_REPORT.md`.
- **Logger path updates**: `src/download_noaa_data.py` and `src/download_pilot_data.py` updated to write to `docs/dev-notes/DATA_LOG.md`.
- **Reviewer comments neutralized**: 3 occurrences of reviewer references replaced with "the internal review" in `src/train_and_evaluate_rigorous.py`, `src/plot_synthetic_injections.py`, and `src/synthetic_injection.py`.
- **README additions**: Added "What is real and what is simulated" and "Repository Layout" sections.
- **Link integrity**: 153 relative markdown links checked, 0 broken.

## 3. Number Check (Task 5)
- **Numeric tokens checked**: 41 tokens collected from the Benchmark Results table.
- **FOUND (41/41)**: `82.62`, `32.5`, `52.50`, `10.0`, `52.86`, `24.9`, `57.00`, `9.0`, `23.51`, `57.6`, `6.00`, `8.5`, `166.78`, `19.69`, `17.3`, `92.00`, `0.69`, `6.7`, `35.74`, `18.21`, `0.0`, `87.42`, `3.66`, `9.2`, `7.21`, `3.93`, `4.7`, `92.98`, `0.92`, `4.0`, `5.60`, `3.16`, `0.7`, `91.88`, `2.67`, `63.84`, `25.09`, `3.3`, `90.42`, `3.40`, `3.5`.
- **NOT FOUND**: None (0).
- **Headline claims vs `rigorous_benchmark_summary.csv`**:
  - 5.60 ± 3.16 false alarms/stn-yr (95% target): **CONFIRMED**
  - 91.88 ± 2.67 % detection (95% target): **CONFIRMED**
  - 2.01 ± 1.74 false alarms/stn-yr (90% target): **CONFIRMED**
  - 87.12 ± 3.94 % detection (90% target): **CONFIRMED**

## 4. Integration Tests (Task 6)
- Result: 53 passed in 9.10s (`data/processed/labeled_al_birmingham_test.csv.gz` and merged dataset present).

## 5. Rain-Mask Control Experiment (Task 7)
- Script: `src/evaluate_rain_masked_baseline.py` created and run; reproduction check: max |dFA|=0.000, max |dDet|=0.000.
- Summary table:
  - Mask `none`: 215.48 FA/yr (at 87.5% det); 303.03 FA/yr (at 92.5% det).
  - Mask `0h` (rain): 197.47 FA/yr (at 87.5% det); 285.77 FA/yr (at 92.5% det).
  - Mask `3h`: 179.80 FA/yr (at 87.5% det); 251.65 FA/yr (at 92.5% det).
  - Mask `6h`: 181.77 FA/yr (at 88.0% det); 245.98 FA/yr (at 92.5% det).
  - Mask `12h`: 177.49 FA/yr (at 87.5% det); 238.66 FA/yr (at 92.0% det).
  - Tier 3 ML: **2.01 FA/yr** (at 87.12% det); **5.60 FA/yr** (at 91.88% det).
- Conclusion: Muting alarms during/after rain reduces rain-coincident false alarms, but blinds the system to plumes during rain and cannot prevent diurnal/baseline false alarms without lowering the threshold to noise levels. At matched ~92% detection, the ML model yields 5.60 FA/yr vs 238.66 for the 12h mask (97.6% lower).

## 6. Problems and Omissions
- None. All tasks completed and checks passed with 0 errors.
