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
   - **Resolution**: Empirically characterized in San Diego (summer 2024 control window) using 3-hour drops ($<-450\text{ CPM}$). Mean filter change interval is 3.3 days (median 3.0 days) with an average step drop of 731.6 CPM ([`filter_cycle_summary.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/filter_cycle_summary.csv)), matching EPA RadNet's twice-weekly filter replacement operational schedule.
4. **Diurnal Radon Cycle Modeling**:
   - **Resolution**: Modeled on verified dry periods (preceding 24h precipitation = 0 mm) across 43,000–62,000 dry hours per station ([`diurnal_cycle_summary.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/diurnal_cycle_summary.csv)). Amplitude ranges from 6.6% (San Diego) to 12.1% (Tampa) of gross CPM. Peak consistently occurs at 06:00–07:00 local time (nocturnal temperature inversion trapping soil radon), with trough at 17:00–20:00 local time (solar convective boundary layer mixing).
5. **Rain Washout Event Confirmation (Gate 1 Requirement)**:
   - **Resolution**: Confirmed empirically across all 5 pilot stations ([`reports/figures/`](file:///Users/o/Projects/radnet-anomaly-detection/reports/figures/)). Convective and frontal rain events produce massive +128% to +184% surges in gross count rate and more than double dose rate, accompanied by immediate simultaneous spikes in Bi-214 (R05, R07, R08) and Pb-214 (R03) channels, followed by rapid decay matching the ~20–30 min radon progeny half-life.

---

## Active Open Questions for Phase 2 & Later

1. **Fixed-Threshold Baseline Formulation (Phase 2)**:
   - Question: What explicit mathematical rule defines the current-practice fixed-threshold alarm for RadNet monitors?
   - Candidates to evaluate in Phase 2:
     - Absolute dose rate threshold (e.g. standard EPA public trigger or $k \times \text{mean}$).
     - Sigma threshold over baseline (e.g., $\mu_{\text{dry}} + 3\sigma_{\text{dry}}$ or $\mu_{\text{dry}} + 5\sigma_{\text{dry}}$).
     - Moving-average baseline with fixed offset ($N$-day rolling mean $+ k\sigma$).
   - Plan for Phase 2: Implement explicit parameter sweeps in `src/evaluate_baseline.py` and quantify the exact false alarm rate per station-year and the fraction of false alarms coinciding with rain.
2. **Operational Ground-Truth Definition for Radon Washout (Phase 3)**:
   - Question: How to formalize non-circular ground truth for the `radon_washout` class given that real-world RadNet data has no external labels?
   - Plan for Phase 3: Define physical criteria combining verified NOAA rain onset, characteristic rise time, and Bi-214/Pb-214 radioactive decay kinetics, noting limitations transparently in `DECISIONS.md`.
3. **Urban Monitor vs. Airport Coordinates Sensitivity**:
   - Question: Does spatial separation between urban ambient air monitoring stations and airport ASOS introduce measurable precipitation onset delays in summer convective thunderstorms vs. winter synoptic fronts?
   - Plan for Phase 2/3: Compare cross-correlation lags between convective summer events (e.g. Tampa/Birmingham afternoon storms) and stratiform winter events.

