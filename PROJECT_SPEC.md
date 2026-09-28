# Project Spec: Weather-Aware Anomaly Detection in RadNet

Single source of truth for all AI agents (Gemini agentic on the laptop, Claude in chat) and the human (Ömer). If something here conflicts with an agent's own idea, this file wins until the human changes it.

## 1. Goal

Build and evaluate a model that separates two signal types in RadNet gamma data:

1. Natural radon-progeny washout (Pb-214, Bi-214) caused by rain. Harmless.
2. Anthropogenic fission-product signal (Cs-137, I-131 like). Real event.

Baseline for comparison: a fixed-threshold alarm, representing current practice. The claim to test: a weather-fused model gives fewer false alarms at the same detection rate.

Classes: `normal`, `radon_washout`, `fission_product`.

## 2. Operating rules for agents

- Raw data is never modified. Raw files live in `data/raw/` and are read-only after download.
- Every download is logged in `DATA_LOG.md` with the URL, date retrieved, station, and file name.
- No numeric value is invented. Parameters that are not known from data or a cited source are written to `QUESTIONS.md` as open questions, and work continues on other tasks meanwhile.
- Every literature-derived number carries a citation (paper, agency page, or handbook) in `DECISIONS.md`.
- Every design decision is recorded in `DECISIONS.md` with a one-line reason.
- Random seeds are fixed and stored in a config file.
- Work happens in a git repo. Each phase ends with a commit and a short `PHASE_REPORT.md` for the human.
- Phases end at a gate (Section 4). The agent stops at each gate and waits for the human to approve.
- Findings that look surprisingly good are treated as suspected leakage or bugs until checked.

## 3. Facts about the data (from EPA pages, May to June 2026)

- All RadNet stationary monitors report gamma gross count rate from particulates collected on the filter.
- Some, not all, monitors also report exposure rate. EPA is adding it to all stations over time.
- Hourly near-real-time data is downloadable as zipped CSV per station, organized by year.
- The CSVs split count rate by gamma channel range.
- Timestamps are in UTC.
- Gaps occur from telecom outages, server issues, and instrument repairs.

Unknown and to be discovered in Phase 0: channel range boundaries in keV, which stations have exposure rate and since when, and how much data is missing per station.

Weather source: NOAA hourly observations (precipitation, pressure, humidity) from the nearest station, with matching by coordinates and UTC time.

## 4. Phases and gates

**Phase 0. Data discovery.** Download the station list. Identify stations with both exposure rate and long histories. Record channel boundaries. Pick 3 to 5 pilot stations with a nearby NOAA station. Gate: human approves station choice.

**Phase 1. Merge and exploration.** Clean and merge radiation and weather on UTC hour. Document missingness. Plot rain events against count rate and exposure rate. Gate: human reviews plots and confirms rain events visibly raise the signal.

**Phase 2. Baseline.** Fixed-threshold alarm, with threshold set by an explicit rule recorded in `DECISIONS.md`. Report alarms per station-year and how many coincide with rain. Gate: human approves.

**Phase 3. Synthetic injection.** See Section 5. Gate: human and Claude review injection design before any model training.

**Phase 4. Models.** Gradient boosting first. Small neural net only if the boosting result leaves a clear gap. Gate: human approves results tables.

**Phase 5. Evaluation.** See Section 6.

**Phase 6. Report.** Introduction, methods, results, discussion, keywords, references. Drafted with Claude, numbers taken only from logged results.

**Stretch (only after Phase 5):** wind-informed graph across stations, HYSPLIT or FLEXPART dispersion for injections, conformal prediction for calibrated false alarm bounds.

## 5. Synthetic injection design

Physical reasoning to encode:

- Radon progeny washout: rises with rain onset and fades after rain stops, on a timescale set by short progeny half-lives (Pb-214 about 27 min, Bi-214 about 20 min).
- Fission products on a filter: rise and then persist, because Cs-137 (about 30 years) and I-131 (about 8 days) do not decay away on an hourly timescale.
- Spectral weight differs between the two, but NaI-type resolution blurs the pairs Pb-214 vs I-131 (352 vs 364 keV) and Bi-214 vs Cs-137 (609 vs 662 keV). Channel-level features are expected to help only partly.

Rules:

- Injections are parametrized families: onset time, rise shape, duration, magnitude relative to local background, and nuclide mix.
- Training and test injections come from disjoint parameter ranges and different shape families, so the model cannot memorize one template.
- A dedicated hard-case set injects fission-product signals during rain, since real plumes are also rained out.
- Parameter ranges are chosen from data (background variability per station) and cited sources, and logged. Anything not derivable goes to `QUESTIONS.md`.
- Reports state clearly that detection performance is measured on synthetic events.

Open question for the radon washout class: real data has no ground-truth labels. Any rule-based labeling (rain plus rise plus decay) is partly circular. The chosen approach and its limitation go in `DECISIONS.md` and the report.

## 6. Evaluation protocol

- Splits are by station and by time period, never random rows.
- Primary metric: false alarms per station-year at a fixed detection probability for injected events.
- Secondary: detection delay, per-class confusion matrix, performance during rain vs dry periods, performance on the rain-coincident hard-case set.
- Baseline and model are compared on identical test data.
- Uncertainty is reported (for example, bootstrap over stations or months).

## 7. Cross-review between AIs

- After each phase, the human pastes `PHASE_REPORT.md`, key code, and result tables into a Claude chat for independent review.
- Claude's review covers physics, leakage, and evaluation validity. The Gemini agent's role is execution and code.
- Where the two disagree, both positions and the evidence are written to `DECISIONS.md`, and the human decides.
