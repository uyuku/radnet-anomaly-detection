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
   - **Resolution**: Characterized across 4 pilot stations over 4 operational years (2021–2024; 211–247 step drops per station) in [`multi_station_filter_cycle_summary.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/multi_station_filter_cycle_summary.csv). Median change interval is 3.96 to 4.29 days (mean 4.6 to 5.1 days) with step drops averaging 359 to 669 CPM, confirming EPA's documented routine operational schedule of collecting filters "once or twice a week" (EPA RadNet Air Data, https://www.epa.gov/radnet/radnet-air-data).
4. **Diurnal Radon Cycle Modeling**:
   - **Resolution**: Modeled on verified dry periods (preceding 24h precipitation = 0 mm) across 43,000–62,000 dry hours per station ([`diurnal_cycle_summary.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/diurnal_cycle_summary.csv)). Amplitude ranges from 6.6% (San Diego) to 12.1% (Tampa) of gross CPM. Peak consistently occurs at 06:00–07:00 local time (nocturnal temperature inversion trapping soil radon), with trough at 17:00–20:00 local time (solar convective boundary layer mixing).
5. **Urban Monitor Coordinates & Distance to Airport**:
   - **Resolution**: Birmingham RadNet monitor is officially identified at the North Birmingham NCore ambient monitoring site (AQS Site ID `01-073-0023`, 33.5530°N, -86.8147°W), exactly 5.8 km west-southwest of NOAA KBHM. For the remaining stations, EPA NAREL does not publish public GPS coordinates as a matter of program security/administrative policy; monitors are located in urban municipal network sites (within 5–20 km of airport ASOS).
6. **Rain Washout Surge Distribution (Phase 3 Parameter Calibration)**:
   - **Resolution**: Evaluated across all 15,602 synchronous rain hours (2017–2025) in [`rain_washout_surge_distribution.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/rain_washout_surge_distribution.csv). Median surge across all rain is +6.8% (+179 CPM), 90th percentile is +41.4% (+1,109 CPM), 95th percentile is +56.5% (+1,546 CPM), and 99th percentile is +97.9% (+2,542 CPM). For heavy rain (>5 mm/h), median surge is +14.0% and 95th percentile is +81.3% (reaching up to +196.8% in severe thunderstorms).

---

## Resolved in Phase 2

1. **Fixed-Threshold Baseline Formulation & False Alarm Rates**:
   - **Resolution**: Evaluated three rule families (Global Gross CPM Sigma, Rolling 7-Day CPM Sigma, Global Dose Rate Sigma) for $k \in \{3.0, 4.0, 5.0\}$ across 324,550 synchronous hours ([`baseline_threshold_evaluation.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/baseline_threshold_evaluation.csv)).
   - **Key Finding**: At standard $3\sigma$ thresholds, the fixed-threshold system triggers **50 to 74 alarm episodes per station-year** in rainy climates (Birmingham, DC, Dallas).
   - **Rain Coincidence**: In Washington DC, **86.7% to 97.2%** of rolling CPM alarm episodes coincide with rain within 3 hours. In Birmingham and Dallas, **62.8% to 91.5%** of alarms coincide with rain. This rigorously demonstrates the operational vulnerability of fixed-threshold monitoring to rain-induced false alarms.

---

## Active Open Questions for Phase 3 & Later

1. **Synthetic Fission-Product Injection Parametrization (Phase 3 Gate)**:
   - Question: What exact mathematical function families and parameter bounds should govern synthetic fission-product plumes (Cs-137, I-131)?
   - Parameters to ground before Gate 3:
     - Onset shape: Step function, linear ramp, or Gaussian plume arrival.
     - Duration / Persistence: Bounded by filter replacement interval (median 4.0 days).
     - Peak amplitude relative to background: Calibrated against the empirical rain surge distribution (e.g., 25th to 95th percentile: +200 to +1,500 CPM above baseline).
     - Spectral energy distribution: Channel R03 (I-131, 364.5 keV photopeak + Compton) and Channel R05 (Cs-137, 661.7 keV photopeak + Compton) relative weightings.
2. **Hard-Case Set: Rain-Coincident Fission Plumes**:
   - Question: What fraction of synthetic injections should be placed during verified rain hours to evaluate discrimination when washout and fission products co-occur?
   - Plan for Phase 3: Allocate a dedicated 30% test subset of injections during verified precipitation events.
3. **Operational Ground-Truth Definition for Radon Washout**:
   - Question: How to define non-circular ground truth for natural radon washout given that real data lacks external labels?
   - Plan for Phase 3: Use an explicit physical filter combining verified NOAA rain ($P_{1\text{h}} > 0$), synchronous R03/R05 rise, and subsequent exponential decay consistent with $T_{1/2} \le 30\text{ min}$.


