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

## Active Open Questions for Phase 1 & Later

1. **NOAA Precipitation Audit across 2017–2025**:
   - Question: What is the exact completeness and accumulation period distribution of the `AA1` liquid precipitation field across all 5 airport ASOS stations?
   - Plan for Phase 1 Task 1: Download full 2017–2025 NOAA ISD records for KBHM, KDCA, KSAN, KDFW, and KTPA, parsing period lengths and quality codes.
2. **RadNet Timestamp Convention**:
   - Question: Does RadNet's `SAMPLE COLLECTION TIME` mark the beginning or end of the hourly measurement interval?
   - Plan for Phase 1 Task 3: Compare lag correlations between NOAA precipitation onset (recorded at :50–:55 UTC) and RadNet count rate surges to confirm alignment.
3. **Air Filter Replacement Schedule**:
   - Question: What is the empirical operational cycle for filter replacement across stations?
   - Context: Cs-137 ($T_{1/2} \approx 30.17\text{ yr}$) and I-131 ($T_{1/2} \approx 8.02\text{ d}$) do not decay away on hourly timescales, but the particulate accumulation on the filter resets when the filter is replaced.
   - Plan for Phase 1 Task 4: Scan count rate time series for characteristic sharp drops (sawtooth pattern) indicating filter changes.
4. **Diurnal Radon Cycle Modeling**:
   - Question: What is the amplitude and phase of the nocturnal boundary-layer inversion cycle across pilot stations?
   - Plan for Phase 1 Task 5: Quantify hourly diurnal profiles on verified dry days to establish normal baseline variation before evaluating weather-fused anomaly thresholds.
5. **Urban Monitor vs. Airport Coordinates**:
   - Question: Can exact or approximate coordinates of RadNet stations be located from municipal air monitoring network plans to determine distance to NOAA airport stations?
   - Plan for Phase 1 Task 7: Query annual air monitoring network plans for Jefferson County AL, District of Columbia, San Diego County CA, Dallas County TX, and Hillsborough County FL.
6. **Operational Ground-Truth Definition for Radon Washout**:
   - Question: How to formalize non-circular ground truth for the `radon_washout` class given that real-world RadNet data has no external labels?
   - Plan for Phase 3: Develop physical criteria combining verified NOAA rain onset, characteristic rise time, and Bi-214/Pb-214 radioactive decay kinetics.
