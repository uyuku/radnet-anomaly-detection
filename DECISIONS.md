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
  - **Reason**: Compliance with Section 2 operating rule ("Raw files live in `data/raw/` and are read-only after download") and avoiding committing bulky binary/CSV archives into git.
- **2026-09-28 | Fixed random seed set to 42 in `configs/config.yaml`**
  - **Reason**: Ensures full reproducibility across all data splits, modeling, and synthetic injections.
- **2026-09-28 | Adopted EPA RadNet gamma channel energy boundaries (keV)**
  - **Reason**: Formal mapping of channel designations R02–R09 to physical photon energy ranges.
  - **Values**:
    - Channel 1 (γ1 / R01): $\le 100\text{ keV}$ (noise threshold, omitted in public near-real-time CSVs)
    - Channel 2 (γ2 / R02): $101 - 200\text{ keV}$
    - Channel 3 (γ3 / R03): $201 - 400\text{ keV}$ (Contains Pb-214 at 352 keV and I-131 at 364 keV)
    - Channel 4 (γ4 / R04): $401 - 600\text{ keV}$
    - Channel 5 (γ5 / R05): $601 - 800\text{ keV}$ (Contains Bi-214 at 609 keV and Cs-137 at 662 keV)
    - Channel 6 (γ6 / R06): $801 - 1000\text{ keV}$
    - Channel 7 (γ7 / R07): $1001 - 1400\text{ keV}$ (Contains Bi-214 at 1120 keV)
    - Channel 8 (γ8 / R08): $1401 - 1800\text{ keV}$ (Contains Bi-214 at 1764 keV)
    - Channel 9 (γ9 / R09): $1801 - 2200\text{ keV}$
  - **Citation**: Vieira, C. L. Z., Koutrakis, P., Huang, S., Grady, S., Hart, J. E., Coull, B. A., Laden, F., Requia, W., Schwartz, J., & Garshick, E. (2019). Short-term effects of particle gamma radiation activities on pulmonary function in COPD patients. *Environmental Research*, 175, 221–227. https://doi.org/10.1016/j.envres.2019.05.021.
- **2026-09-28 | Selected 5 pilot stations for Phase 1**
  - **Reason**: Maximum historical exposure rate coverage (>=2016), >80-94% completeness over 2017–2025, distinct climatic/radon regimes, and direct colocation with first-order NOAA airport ASOS stations.
  - **Pilot Set**:
    1. Birmingham, AL (Southeast Humid Subtropical / frequent rain washout; NOAA KBHM)
    2. Washington, DC (Mid-Atlantic Temperate / distinct 4 seasons; NOAA KDCA)
    3. San Diego, CA (West Coast Mediterranean / arid baseline control; NOAA KSAN)
    4. Dallas, TX (Southern Plains / severe thunderstorms & convective fronts; NOAA KDFW)
    5. Tampa, FL (Gulf Coast Subtropical / summer convective thunderstorms; NOAA KTPA)
- **2026-09-28 | Weather data source: NOAA NCEI Global-Hourly (ISD/LCD ASOS)**
  - **Reason**: Official, verified hourly weather observations with UTC timestamps, hourly liquid precipitation depth (AA1), sea level pressure (SLP), temperature (TMP), and dew point (DEW).
  - **Citation**: NOAA National Centers for Environmental Information (NCEI) Integrated Surface Database (ISD) / Global Hourly Data. https://www.ncei.noaa.gov/data/global-hourly/
