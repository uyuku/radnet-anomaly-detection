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

## Resolved in Phase 3 (Final Revision - Training Freeze)

1. **Synthetic Injection Parametrization, Multi-Nuclide Inventory & Non-Overlap**:
   - **Resolution**: Implemented in [`src/synthetic_injection.py`](file:///Users/o/Projects/radnet-anomaly-detection/src/synthetic_injection.py) and verified in [`synthetic_injection_catalog.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/synthetic_injection_catalog.csv).
   - **100% Filter Drop Synchronization**: All 450 of 450 injections (100.0%) end at an empirically detected real physical filter replacement step drop (`truncated_by_filter_change = True`), completely eliminating synthetic-only drop cliffs. Candidate windows with intermediate drops are rejected. Injection onset cadence (`hours_since_last_detected_drop`, median 84.0h, IQR [40.2h, 176.8h]) matches natural normal baseline operational cycles (median 97.0h, IQR [41.0h, 226.0h]).
   - **Exact 20.0% Scenario Balance Across All Environments**: All 6 environments (`train_strictly_dry`, `train_rain_coincident`, `test_standard_strictly_dry`, `test_standard_rain`, `test_stress_strictly_dry`, `test_stress_rain`) have exactly 20.0% of each of the 5 operational scenarios (90 total per scenario across the network), completely eliminating confounding between scenario mix and weather state.
   - **Cs-134 Lines in Channel R07**: Fresh reactor fission (`fission_reactor_fukushima`) incorporates $^{134}\text{Cs}$ high-energy lines (1168 & 1365 keV in R07 via $p_{\text{R07, Cs134}} \sim \mathcal{U}[0.025, 0.055]$), yielding an R07+R08 high-energy share (Mean = 4.54%, IQR = [3.65%, 5.42%]) that directly overlaps the empirical radon washout distribution (ratio-of-sums 5.58% [5.52%, 5.65%], IQR [4.04%, 7.21%]).
   - **Broad Continuous Spectral Sampling**: Photopeak and Compton continuum fractions drawn continuously per injection via Dirichlet distributions, with high-energy scatter floors varying across 0.5% to 3.5%, preventing models from memorizing rigid channel ratios. All 8 sampled shares (`share_r02` through `share_r09`) are preserved directly in the catalog.
   - **Traceable Empirical Dose Rate Injection**: Injects $\Delta \text{Dose} = k_{\text{dose}} \cdot \Delta \text{Gross CPM}$ calibrated directly in [`src/calibrate_dose_coupling.py`](file:///Users/o/Projects/radnet-anomaly-detection/src/calibrate_dose_coupling.py) to station-specific empirical rain regressions ($k_{\text{dose}} \in [0.0119, 0.0239]\text{ nSv/h per CPM}$). Dual assumptions documented: photon energy dependence vs unified empirical range; identical dose-to-gross ratio between classes by construction.
   - **Filter Accumulation & Clearing Physics**: Plume passage accumulates particulates; retention decays $^{131}\text{I}$ ($\lambda = 0.00360\text{ h}^{-1}$). Particulate excess drops to zero synchronously with the real background filter step drop on the station.
   - **Zero Overlaps & Strict 48h Buffer**: Enforced across all branches; exactly 0 buffer violations (<48h) across all 450 injections.
   - **Exact Realized Duration Ranges**: Documented as 10–35h (`train_strictly_dry`), 25–94h (`train_rain_coincident`), 36–75h (`test_standard_strictly_dry`), 38–164h (`test_standard_rain`), 8–20h (`test_stress_strictly_dry`), 28–140h (`test_stress_rain`).
   - **Hard-Regime Stress Test**: Test set includes 100 subtle plumes (Band A [250, 600] CPM), with 50 forced into rain onsets where the rising plume is directly immersed inside the natural radon washout surge.
2. **Empirical Washout Gamma Energy Spectrum Grounding**:
   - **Resolution**: Scripted in [`src/analyze_washout_spectrum.py`](file:///Users/o/Projects/radnet-anomaly-detection/src/analyze_washout_spectrum.py) across 5,485 verified substantial rain hours (>100 CPM excess, $P_{\text{1h}} \ge 1.0\text{ mm/h}$) on continuous calendar grids ([`rain_washout_spectral_shares.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/rain_washout_spectral_shares.csv)).
   - **Ratio-of-Sums Formulation**: Pooled network ratio of sums: R02: 42.31%, R03: 33.73%, R04: 9.16%, R05: 6.04%, R06: 2.48%, R07: 3.87%, R08: 1.72%, R09: 0.70%.
   - **R07+R08 High-Energy Ratio of Sums**: **5.58%** with 1,000-draw bootstrap 95% CI of **[5.52%, 5.65%]**. Pooled hour-level percentiles: 5th: -3.34%, 25th: 4.04%, 50th: 5.64%, 75th: 7.21%, 95th: 13.45%.
   - **Tampa Baseline Framing**: Tampa exhibits lower baseline channel counts across all lower-energy channels (Tampa dry mean R02 is 882.7 CPM against 1,980.4 CPM in Birmingham; dry gross mean is 2,081.8 CPM in Tampa vs 3,887.3 CPM in Birmingham, traceable to `candidate_pilot_stations_comparison.csv` and empirical dry hours). Causal geological/detector hypotheses moved to Open Questions below.
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
   - Policy: Validate model generalization across climate regimes by holding out San Diego (which exhibits an empirical negative rain-radiation correlation, $r = -0.06$ to $-0.13$) as the external unseen evaluation site.
4. **Stratified Performance Breakdown (Phase 5 Policy)**:
   - Policy: Report detection probability broken down separately by:
     - Environmental regime: strictly dry vs rain onset.
     - Release scenario: fission products (Cs-137, Cs-134, I-131) vs orphan activation sources (Co-60) vs mixed excursions.
     - Duration regime: standard retention (36–80h) vs short hard-regime stress plumes (8–20h).
5. **Detector Response Sensitivity Sweep (Phase 5 Policy)**:
   - Plan: Run a sensitivity sweep varying the photopeak-to-total ratio over $\pm 30\%$ to prove that models trained on synthetic spectra generalize across variations in detector crystal dimensions and down-scatter continua.
6. **Physical Mechanisms Governing Low Baseline Counts and Shifted Spectral Ratios in Tampa (Open Research Question)**:
   - *Observation*: In `candidate_pilot_stations_comparison.csv` and empirical dry baseline records, Tampa displays substantially lower gross background counts (2,081.8 CPM vs 3,887.3 CPM in Birmingham; dry R02 882.7 CPM vs 1,980.4 CPM) and an elevated R02 washout share (ratio of sums 53.49% vs 40.96% in Birmingham).
   - *Open Hypotheses for Further Investigation*:
     (a) Terrestrial lithology: Florida's quartz sand, phosphorite, and carbonate platform may contain lower natural concentrations of primordial thorium, uranium, and potassium than the Appalachian Paleozoic formations surrounding Birmingham.
     (b) Hardware / instrumentation: Differing PMT bias voltage, gain stabilization, crystal housing, or lower-level discriminator (LLD) thresholds between stationary monitor deployment batches.
     (c) Washout droplet microphysics: Subtropical maritime convective rain scavenging mechanisms differing from continental convective systems.
7. **Physical Mechanisms Governing Negative Rain-Radiation Correlation in San Diego (Open Research Question)**:
   - *Observation*: San Diego exhibits an empirical negative cross-correlation ($r = -0.06$ to $-0.13$) between hourly precipitation and gross gamma CPM, with only 5 hours exceeding $+2\sigma$ during rain over 2017–2025.
   - *Open Hypotheses for Further Investigation*:
     (a) Pacific marine layer advection: Coastal Southern California precipitation is frequently associated with Pacific marine air masses depleted of terrestrial radon progeny relative to continental air.
     (b) Aerosol scavenging dynamics: Precipitation clearing existing ambient aerosols from the surface layer without replenishing short-lived radon progeny from local soil exhalation.
     (c) Local topography and coastal microclimate at KSAN (San Diego International Airport / Lindbergh Field).




