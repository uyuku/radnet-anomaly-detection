# Decisions Log

Format:
- **Date**: YYYY-MM-DD
- **Decision**: Summary of decision
- **Reason**: One-line rationale
- **Citation / Reference**: Source or literature citation if applicable

---

### Phase 0: Data Discovery Decisions

- **2026-09-28 | Git repository initialized in `~/Projects/radnet-anomaly-detection`**
  - **Reason**: Standard local workspace for projects, avoiding cluttering root home directory.
- **2026-09-28 | Raw data storage policy (`data/raw/` read-only, gitignored)**
  - **Reason**: Compliance with Section 2 operating rule ("Raw files live in `data/raw/` and are read-only after download") and avoiding committing bulky binary/CSV archives into git. Full SHA256 hashes tracked in `data/raw/MANIFEST.sha256`.
- **2026-09-28 | Fixed random seed set to 42 in `configs/config.yaml`**
  - **Reason**: Ensures full reproducibility across all data splits, modeling, and synthetic injections.
- **2026-09-28 | Adopted EPA RadNet gamma channel energy boundaries (keV)**
  - **Reason**: Formal mapping of channel designations R02–R09 to physical photon energy ranges.
  - **Values**:
    - Channel 1 (γ1 / R01): $\le 100\text{ keV}$ (noise threshold, omitted in public CSVs)
    - Channel 2 (γ2 / R02): $101 - 200\text{ keV}$ (Low-energy Compton backscatter; 41.5%–51.3% of baseline counts)
    - Channel 3 (γ3 / R03): $201 - 400\text{ keV}$ (Contains Pb-214 at 351.9 keV and I-131 at 364.5 keV; 26.4%–29.3% of baseline counts)
    - Channel 4 (γ4 / R04): $401 - 600\text{ keV}$ (Intermediate Compton continuum)
    - Channel 5 (γ5 / R05): $601 - 800\text{ keV}$ (Contains Bi-214 at 609.3 keV and Cs-137 at 661.7 keV)
    - Channel 6 (γ6 / R06): $801 - 1000\text{ keV}$ (Upper Compton continuum)
    - Channel 7 (γ7 / R07): $1001 - 1400\text{ keV}$ (Contains Bi-214 photopeak at 1120.3 keV)
    - Channel 8 (γ8 / R08): $1401 - 1800\text{ keV}$ (Contains Bi-214 photopeak at 1764.5 keV and natural terrestrial K-40 at 1460.8 keV)
    - Channel 9 (γ9 / R09): $1801 - 2200\text{ keV}$ (High-energy cosmic/prompt background)
  - **Citations**:
    - Vieira, C. L. Z., Koutrakis, P., Huang, S., Grady, S., Hart, J. E., Coull, B. A., Laden, F., Requia, W., Schwartz, J., & Garshick, E. (2019). Short-term effects of particle gamma radiation activities on pulmonary function in COPD patients. *Environmental Research*, 175, 221–227. https://doi.org/10.1016/j.envres.2019.05.032.
    - Huang, S., Garshick, E., Vieira, C. L. Z., Hart, J. E., Coull, B. A., & Koutrakis, P. (2020). Short-term exposure to ambient particle gamma radiation and mortality in the US Medicare population. *Environmental Research*, 182, 108995. https://doi.org/10.1016/j.envres.2019.108995 (PMC6983292).
    - Knoll, G. F. (2010). *Radiation Detection and Measurement* (4th ed.). John Wiley & Sons (pp. 338–342 for NaI(Tl) energy resolution of 7–9% at 662 keV).
- **2026-09-28 | Complete-channel gross CPM calculation policy**
  - **Reason**: Addressed review finding 2.3: summing rows with missing channel values skips NaN by default, producing partial sums and artificially inflating variance. Gross count statistics are computed strictly on complete 8-channel records.
- **2026-09-28 | Selected 5 pilot stations for Phase 1**
  - **Reason**: Maximum historical exposure rate coverage (>=2016), >80–94% completeness over 2017–2025, effective dose-rate coverage of 77.9% to 94.3%, distinct climatic zones, and colocation with primary NOAA airport ASOS stations.
  - **Pilot Set**:
    1. Birmingham, AL (`AL_BIRMINGHAM`): Southeast Humid Subtropical; paired with NOAA KBHM.
    2. Washington, DC (`DC_WASHINGTON`): Mid-Atlantic Temperate 4-season; paired with NOAA KDCA.
    3. San Diego, CA (`CA_SAN_DIEGO`): West Coast Mediterranean / arid baseline control; paired with NOAA KSAN.
    4. Dallas, TX (`TX_DALLAS`): Southern Plains Subtropical Continental; paired with NOAA KDFW.
    5. Tampa, FL (`FL_TAMPA`): Gulf Coast Subtropical Peninsula; paired with NOAA KTPA.
  - **Geologic Context Citation**: U.S. Environmental Protection Agency (1993). *EPA Map of Radon Zones* (EPA-402-R-93-071).
- **2026-09-28 | Weather data source: NOAA NCEI Global-Hourly (ISD/LCD ASOS)**
  - **Reason**: Official, verified hourly weather observations with UTC timestamps, hourly liquid precipitation depth (AA1), sea level pressure (SLP), temperature (TMP), and dew point (DEW).
  - **Citation**: NOAA National Centers for Environmental Information (NCEI) Integrated Surface Database (ISD) / Global Hourly Data. https://www.ncei.noaa.gov/data/global-hourly/

---

### Phase 1: Merge and Exploration Decisions

- **2026-09-29 | NOAA Global-Hourly `AA1` liquid precipitation parsing policy**
  - **Reason**: NOAA ISD encodes precipitation in the repeatable `AA1` section formatted as `AA1_1,AA1_2,AA1_3,AA1_4` representing `period_quantity,depth_dimension,condition_code,quality_code`. To ensure clean pairing with RadNet 1-hour counts, records are filtered to standard 1-hour intervals (`period == 1`) with accepted quality codes (`1`, `5`, `C`, `S`). Depths (tenths of mm) are converted to mm. Over 90%–96% of airport AA1 records represent standard 1-hour periods.
  - **Citation**: NOAA Federal Climate Complex (2018). *Integrated Surface Database (ISD) Format Document*, Data Version 8, NOAA NCEI.
- **2026-09-29 | Hourly temporal grid alignment policy (`dt.round('h')`)**
  - **Reason**: RadNet observations report collection times typically between :50 and :55 UTC. NOAA routine airport METAR surface observations are likewise transmitted between :50 and :55 UTC. Rounding both datasets to the nearest UTC hour (`dt.round('h')`) aligns contemporaneous observations into identical hour bins with zero artificial lag distortion. Cross-correlation between precipitation and count rate peaks at lag 0 ($r = 0.22$) and lag +1 ($r = 0.25$).
- **2026-09-29 | Verified dry day definition for baseline modeling**
  - **Reason**: To isolate true background diurnal cycles and avoid contamination from residual radon progeny washout, dry periods are strictly defined as hours where both the current hour's precipitation is 0.0 mm and the rolling preceding 24-hour precipitation sum is 0.0 mm.
- **2026-09-29 | Filter replacement detection criteria**
  - **Reason**: Operational particulate air filters accumulate dust and long-lived nuclides over several days before being physically replaced by technicians. This produces a gradual multi-day upward drift terminated by an abrupt step drop. Step drops are identified during dry weather by a 3-hour gross CPM drop exceeding 450 CPM (`gross_cpm.diff(3) < -450`), clustered with a 24-hour refractory window. Evaluated in San Diego, this yields an empirical mean replacement interval of 3.3 days (median 3.0 days) with an average drop of 731.6 CPM, matching EPA RadNet's twice-weekly filter change schedule.

