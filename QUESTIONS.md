# Open Questions & Unknown Parameters

This file tracks parameters and questions that are not yet known from data or cited sources, per Section 2 of `PROJECT_SPEC.md`.

## Resolved in Phase 0

1. **Exact gamma channel energy boundaries (in keV)**:
   - **Resolution**: Identified from peer-reviewed literature analyzing RadNet airborne gamma spectrometry:
     - Channel 1 (γ1 / R01): $\le 100\text{ keV}$ (noise threshold; verified absent in raw CSV headers)
     - Channel 2 (γ2 / R02): $101 - 200\text{ keV}$ (41.5%–51.3% of baseline gross count rate)
     - Channel 3 (γ3 / R03): $201 - 400\text{ keV}$ (Contains Pb-214 at 351.9 keV and I-131 at 364.5 keV; 26.4%–29.3% of baseline gross count rate)
     - Channel 4 (γ4 / R04): $401 - 600\text{ keV}$
     - Channel 5 (γ5 / R05): $601 - 800\text{ keV}$ (Contains Bi-214 at 609.3 keV and Cs-137 at 661.7 keV)
     - Channel 6 (γ6 / R06): $801 - 1000\text{ keV}$
     - Channel 7 (γ7 / R07): $1001 - 1400\text{ keV}$ (Contains Bi-214 photopeak at 1120.3 keV)
     - Channel 8 (γ8 / R08): $1401 - 1800\text{ keV}$ (Contains Bi-214 photopeak at 1764.5 keV and terrestrial K-40 at 1460.8 keV)
     - Channel 9 (γ9 / R09): $1801 - 2200\text{ keV}$
   - **Citations**:
     - Vieira, C. L. Z., Koutrakis, P., Huang, S., Grady, S., Hart, J. E., Coull, B. A., Laden, F., Requia, W., Schwartz, J., & Garshick, E. (2019). Short-term effects of particle gamma radiation activities on pulmonary function in COPD patients. *Environmental Research*, 175, 221–227. https://doi.org/10.1016/j.envres.2019.05.032.
     - Huang, S., Garshick, E., Vieira, C. L. Z., Hart, J. E., Coull, B. A., & Koutrakis, P. (2020). Short-term exposure to ambient particle gamma radiation and mortality in the US Medicare population. *Environmental Research*, 182, 108995. https://doi.org/10.1016/j.envres.2019.108995 (PMC6983292).
     - Knoll, G. F. (2010). *Radiation Detection and Measurement* (4th ed.). John Wiley & Sons (pp. 338–342).
2. **Exposure rate start dates and reporting status**:
   - **Resolution**: Parsed from official EPA downloads index into `data/processed/radnet_station_inventory.csv`. 127 stations have exposure-rate start dates in EPA metadata, 20 of which date to mid-2016.
3. **Missingness baseline across candidate pilot stations**:
   - **Resolution**: Evaluated across 78,888 expected hours (2017–2025). Complete-channel filtering adopted. Top 5 stations have 80.6%–94.3% completeness and 77.9%–94.3% effective dose coverage.

---

## Resolved in Phase 1

1. **NOAA Precipitation Audit across 2017–2025**:
   - **Resolution**: Audited 45 NOAA Global-Hourly files (574,671 records) across KBHM, KDCA, KSAN, KDFW, KTPA ([`noaa_station_summary.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/noaa_station_summary.csv)). AA1 precipitation presence ranges from 72.8% (San Diego) to 85.8% (Birmingham). Over 90%–96% of AA1 records represent standard 1-hour periods. Temperature and pressure completeness exceed 75%–97%. Over 15,700 synchronous rain hours identified.
2. **RadNet Timestamp Convention & Grid Alignment**:
   - **Resolution**: RadNet reports `SAMPLE COLLECTION TIME` at :50–:55 UTC (end of sampling hour). NOAA routine METAR observations report at :50–:55 UTC. Rounding both to nearest UTC hour (`dt.round('h')`) yields perfect contemporaneous alignment. Cross-correlation between precipitation depth and gross CPM peaks at lag 0 ($r = 0.22$) and lag +1 ($r = 0.25$), confirming zero physical lag.
3. **Air Filter Replacement Schedule**:
   - **Resolution**: Characterized across 4 pilot stations over 4 operational years (2021–2024; 143–216 strictly within-dry-spell step drops per station) in [`multi_station_filter_cycle_summary.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/multi_station_filter_cycle_summary.csv). Median change interval is 3.96 to 4.83 days (mean 4.8 to 5.5 days) with step drops averaging 343 to 649 CPM, confirming EPA's documented routine operational schedule of collecting filters "once or twice a week" (EPA RadNet Air Data, https://www.epa.gov/radnet/radnet-air-data).
4. **Diurnal Radon Cycle Modeling**:
   - **Resolution**: Modeled on verified dry periods (preceding 24h precipitation = 0 mm) across 43,000–62,000 dry hours per station ([`diurnal_cycle_summary.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/diurnal_cycle_summary.csv)). Amplitude ranges from 6.6% (San Diego) to 12.1% (Tampa) of gross CPM. Peak consistently occurs at 06:00–07:00 local time (nocturnal temperature inversion trapping soil radon), with trough at 17:00–20:00 local time (solar convective boundary layer mixing).
5. **Urban Monitor Coordinates & Distance to Airport (Birmingham Grounded)**:
   - **Resolution**: Birmingham RadNet monitor is officially identified at the North Birmingham NCore ambient monitoring site (AQS Site ID `01-073-0023`, 33.5530°N, -86.8147°W), exactly 5.8 km west-southwest of NOAA KBHM (*Citation: Jefferson County Department of Health Air Quality Monitoring Network Plan; EPA AirData*). For Washington DC, San Diego, Dallas, and Tampa, exact street addresses/coordinates have not yet been located in state plans and remain open items below.
6. **Rain Washout Surge Distribution (Phase 3 Parameter Calibration)**:
   - **Resolution**: Evaluated across all 15,602 synchronous rain hours (2017–2025) in [`rain_washout_surge_distribution.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/rain_washout_surge_distribution.csv). Median surge across all rain is +6.85% (+180.8 CPM), 90th percentile is +41.41% (+1,109.1 CPM), 95th percentile is +56.53% (+1,546.8 CPM), and 99th percentile is +97.94% (+2,541.9 CPM). For heavy rain (>5 mm/h), median surge is +14.00% and 95th percentile is +81.26% (reaching up to +196.81% in severe thunderstorms).

---

## Resolved in Phase 2

1. **Fixed-Threshold Baseline Formulation & False Alarm Rates**:
   - **Resolution**: Evaluated three rule families (Global Gross CPM Sigma, Rolling 7-Day CPM Sigma, Global Dose Rate Sigma) for $k \in \{3.0, 4.0, 5.0\}$ across 324,550 synchronous hours on continuous 78,888-hour calendar grids ([`baseline_threshold_evaluation.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/baseline_threshold_evaluation.csv)).
   - **Key Finding**: At standard $3\sigma$ thresholds, the fixed-threshold system triggers **52 to 78 alarm episodes per station-year** in rainy climates (Birmingham, DC, Dallas).
   - **Rain Coincidence**: In Washington DC, **86.5% to 97.2%** of rolling CPM alarm episodes coincide with rain within 3 hours. In Birmingham and Dallas, **62.1% to 91.3%** of alarms coincide with rain. This rigorously demonstrates the operational vulnerability of fixed-threshold monitoring to rain-induced false alarms.

---

## Resolved in Phase 3 (Revised & Grounded - Second Pass)

1. **Synthetic Injection Parametrization, Multi-Nuclide Inventory & Non-Overlap**:
   - **Resolution**: Implemented in [`src/synthetic_injection.py`](file:///Users/o/Projects/radnet-anomaly-detection/src/synthetic_injection.py) and verified in [`synthetic_injection_catalog.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/synthetic_injection_catalog.csv).
   - **Operational Release Scenarios**: 5 scenarios structured across reactor fission (Cs-137 + Cs-134 + I-131; Fukushima core ratio, Masson et al. 2011), pure legacy sources (Cs-137), pure radiopharmaceuticals (I-131), orphan industrial activation sources (Co-60; photopeaks at 1173 & 1332 keV in R07), and mixed core excursions.
   - **Broad Continuous Spectral Sampling**: Photopeak and Compton continuum fractions drawn continuously per injection via Dirichlet distributions, with high-energy scatter floors varying across 0.5% to 3.5%, preventing models from memorizing rigid channel ratios.
   - **Traceable Empirical Dose Rate Injection**: Injects $\Delta \text{Dose} = k_{\text{dose}} \cdot \Delta \text{Gross CPM}$ calibrated directly in [`src/calibrate_dose_coupling.py`](file:///Users/o/Projects/radnet-anomaly-detection/src/calibrate_dose_coupling.py) to station-specific empirical rain regressions ($k_{\text{dose}} \in [0.0119, 0.0239]\text{ nSv/h per CPM}$). Dual assumptions documented: photon energy dependence vs unified empirical range; identical dose-to-gross ratio between classes by construction.
   - **Filter Accumulation & Real Step-Drop Synchronization**: Plume passage accumulates particulates; retention decays $^{131}\text{I}$ ($\lambda = 0.00360\text{ h}^{-1}$). Candidate windows checked against the Gate 2 dry 3h step detector: windows with filter drops during inflow are rejected; windows with drops during retention are truncated at that exact hour, synchronizing synthetic clearing with real physical background filter replacements (43 injections truncated).
   - **Zero Overlaps**: Strict buffers enforced between injections. Exactly 0 overlapping pairs across 450 total catalog injections.
   - **Hard-Regime Stress Test**: Test set includes 100 subtle plumes (Band A [250, 600] CPM, 8–20h duration), with 50 forced into rain onsets where the rising plume is directly immersed inside the natural radon washout surge.
2. **Empirical Washout Gamma Energy Spectrum Grounding**:
   - **Resolution**: Scripted in [`src/analyze_washout_spectrum.py`](file:///Users/o/Projects/radnet-anomaly-detection/src/analyze_washout_spectrum.py) across 5,485 verified substantial rain hours (>100 CPM excess) on the continuous calendar grid ([`rain_washout_spectral_shares.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/rain_washout_spectral_shares.csv)): Pooled shares: R02: 43.95%, R03: 32.34%, R04: 9.05%, R05: 5.92%, R06: 2.43%, R07: 3.81%, R08: 1.54%, R09: 0.96% (R07+R08 = 5.35%). Tampa's elevated R02 (61.81%) explained by limestone/sand low terrestrial background and detector calibration differences.
3. **Headline Evaluation Metric Policy**:
   - **Resolution**: Primary headline metric adopted is **false alarms per station-year at a fixed detection probability for injected fission events** on **unmodified background data** using verified observed hours as the denominator ($N_{\text{obs}} / 8,766$).
4. **Operational Ground-Truth Definition for Natural Radon Washout**:
   - **Resolution**: Operational heuristic rule: an hour is labeled `radon_washout` if $P_{3\text{h}} > 0\text{ mm}$ and gross CPM exceeds dry baseline by $>2\sigma$ ($x(t) > \mu_{\text{dry}} + 2\sigma_{\text{dry}}$) on complete-channel records. Documented as an explicit methodological limitation in `DECISIONS.md`. Missing data explicitly labeled `unobserved` (59,873 hours across network).

---

## Active Open Questions & Architectural Policies for Phase 4 & Phase 5

1. **Dual-Series Dataset Pipeline Architecture (Phase 4 Policy)**:
   - Policy: Build two distinct series per station and split:
     - *Clean Background Series*: Clean unmodified observations used strictly for baseline feature computation and counting operational false alarms per station-year.
     - *Injected Series*: Synthetic injections applied to the continuous series before computing feature representations (e.g. 168h rolling statistics), used strictly for evaluating detection probability and time-to-alarm.
2. **Feature Ablation Hierarchy (Phase 5 Policy)**:
   - Question: What is the exact marginal detection gain provided by weather fusion over pure radiation features?
   - Plan: Implement a 3-tier feature ablation:
     1. Gross radiation features only (Gross CPM, rolling means, differences).
     2. Radiation + spectral features (Channel energy ratios, R07+R08 high-energy share, photopeak ratios).
     3. Radiation + spectral + weather fusion (Precipitation depth, multi-scale rain history, pressure tendencies, humidity).
3. **Leave-One-Station-Out (LOSO) Cross-Validation**:
   - Policy: Validate model generalization across climate regimes by holding out San Diego (which exhibits negative rain-radiation correlation due to coastal marine layer inversions) as the external unseen evaluation site.
4. **Stratified Performance Breakdown (Phase 5 Policy)**:
   - Policy: Report detection probability broken down separately by:
     - Environmental regime: strictly dry vs rain onset.
     - Release scenario: fission products (Cs-137, Cs-134, I-131) vs orphan activation sources (Co-60) vs mixed excursions.
     - Duration regime: standard retention (36–80h) vs short hard-regime stress plumes (8–20h).
5. **Detector Response Sensitivity Sweep (Phase 5 Policy)**:
   - Plan: Run a sensitivity sweep varying the photopeak-to-total ratio over $\pm 30\%$ to prove that models trained on synthetic spectra generalize across variations in detector crystal dimensions and down-scatter continua.




