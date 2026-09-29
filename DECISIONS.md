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
  - **Reason**: NOAA ISD encodes precipitation in the repeatable `AA1` section formatted as `AA1_1,AA1_2,AA1_3,AA1_4` representing `period_quantity,depth_dimension,condition_code,quality_code`. To ensure clean pairing with RadNet 1-hour counts, records are filtered strictly to standard 1-hour intervals (`period == 1`) with verified quality codes `1` (passed standard checks) and `5` (passed all checks). Suspect codes (e.g. `C`, `S`) are excluded. Depths (tenths of mm) are converted to mm.
  - **Citation**: NOAA Federal Climate Complex (2018). *Integrated Surface Database (ISD) Format Document*, Data Version 8, NOAA NCEI.
- **2026-09-29 | Continuous calendar grid temporal operations policy**
  - **Reason**: All time-series operations (rolling 168h means, 3h/6h/24h precipitation windows, lag shifts, 3h step diffs, and alarm episode clustering) are executed on the continuous 78,888-hour regular hourly grid (`pd.DatetimeIndex`) before applying observation masks. This guarantees that window lengths and lag times represent exact physical calendar hours rather than row indices across data gaps.
- **2026-09-29 | Hourly temporal grid alignment policy (`dt.round('h')`)**
  - **Reason**: RadNet observations report collection times typically between :50 and :55 UTC. NOAA routine airport METAR surface observations are likewise transmitted between :50 and :55 UTC. Rounding both datasets to the nearest UTC hour (`dt.round('h')`) aligns contemporaneous observations into identical hour bins with minimal lag distortion. Full lag profile analysis indicates cross-correlation peaks at lag 0 to +1 ($r \approx 0.22\text{--}0.39$), consistent with physical deposition and subsequent 20–30 min decay of progeny.
- **2026-09-29 | Verified dry day definition for baseline modeling**
  - **Reason**: To isolate true background diurnal cycles and avoid contamination from residual radon progeny washout, dry periods are strictly defined as hours where both the current hour's precipitation is 0.0 mm and the rolling preceding 24-hour precipitation sum is 0.0 mm.
- **2026-09-29 | Particulate filter replacement interval and bounds**
  - **Reason**: Empirically characterized across 4 pilot stations over 4 operational years (2021–2024, 143–216 strictly within-dry-spell step drops per station). The network median interval between filter changes is 3.96 to 4.83 days (mean 4.8 to 5.5 days) with step drops averaging 343 to 649 CPM, directly matching EPA's documented operational schedule of filter collection "once or twice a week". Bounds the persistence of synthetic particulate injections in Phase 3.
  - **Citation**: U.S. Environmental Protection Agency (EPA). *RadNet Air Data*. https://www.epa.gov/radnet/radnet-air-data.

---

### Phase 2: Fixed-Threshold Baseline Decisions

- **2026-09-29 | Fixed-threshold baseline alarm formulation**
  - **Reason**: Required by Section 4 of `PROJECT_SPEC.md` to establish current-practice operational alarm benchmarks. Evaluated three explicit rule families across all 5 pilot stations (2017–2025; 324,550 synchronous hours):
    1. **Global Gross CPM Sigma**: Threshold $T = \mu_{\text{dry}} + k \cdot \sigma_{\text{dry}}$ ($k = 3.0, 4.0, 5.0$). Represents standard static threshold monitoring.
    2. **Rolling 7-Day CPM Sigma**: Threshold $T(t) = \mu_{168\text{h}}(t) + k \cdot \sigma_{\text{dry}}$ ($k = 3.0, 4.0, 5.0$). Accommodates slow seasonal and detector baseline drifts while flagging acute surges.
    3. **Global Dose Rate Sigma**: Threshold $T = \mu_{\text{dose, dry}} + k \cdot \sigma_{\text{dose, dry}}$ ($k = 3.0, 4.0, 5.0$). Evaluates exposure rate channels calibrated in nSv/h.
- **2026-09-29 | Discrete alarm episode clustering on calendar grid**
  - **Reason**: Individual alarm hours occurring consecutively reflect a single prolonged incident rather than independent alarms. On the continuous calendar grid, an alarm episode is defined as starting when $x(t) \ge T$ and $x(t-1) < T$ (or $t-1$ was not in alarm/gap). This prevents artificial merging of separate alarm episodes across data gaps.
- **2026-09-29 | Rain coincidence window definition**
  - **Reason**: Radon progeny washout occurs during rain, and deposited progeny emit gamma radiation with half-lives of ~20 to 27 min (decaying over 2 to 3 hours). To capture true washout-induced false alarms, rain coincidence is evaluated over a 3-hour window ($\sum_{i=0}^{2} P(t-i) > 0$), as well as strict 1-hour ($P(t) > 0$) and extended 6-hour storm windows.

---

### Phase 3: Synthetic Injection Design Decisions

- **2026-09-29 | Synthetic injection catalog generation & deterministic seeding**
  - **Reason**: Required by Section 5 of `PROJECT_SPEC.md`. 500 total injections (300 Train 2017–2022, 200 Test 2023–2025) generated deterministically using seed `42` across all 5 pilot stations (60 train / 40 test per station), saved to `data/processed/synthetic_injection_catalog.csv`.
- **2026-09-29 | Strict train/test parameter disjointness policy**
  - **Reason**: To prevent machine learning models from memorizing artificial injection templates or specific pulse durations/magnitudes, train and test sets are constructed with zero overlap across four distinct parameter axes:
    1. **Shape Family**: Train uses `linear_ramp` ($S(t) = \min(t/t_{\text{rise}}, 1.0)$) and `step` ($S(t) = 1.0$). Test uses `sigmoidal` (logistic growth $S(t) = 1/(1+e^{-(t-t_{\text{mid}})/\tau})$) and `exponential` ($S(t) = 1 - e^{-t/\tau}$).
    2. **Duration**: Train uses 6 to 24 hours (short/acute events). Test uses 28 to 72 hours (medium/prolonged events). A 4-hour gap [24h, 28h] separates the two sets. Upper bound (72h) is physically bounded by the empirical ~4-day filter replacement cycle.
    3. **Magnitude Bands**: Train uses Band A [250, 600] CPM (subtle) and Band C [1400, 2500] CPM (severe). Test uses Band B [700, 1200] CPM (strictly interpolating between Band A and Band C, separated by 100–200 CPM exclusion gaps).
    4. **Nuclide Mix**: Train uses balanced mixtures ($f_{\text{Cs}} \in [0.30, 0.70]$). Test evaluates pure/skewed plumes: pure Cs-137 ($f_{\text{Cs}} \in [0.85, 1.00]$, 98 cases) and pure I-131 ($f_{\text{Cs}} \in [0.00, 0.15]$, 102 cases), separated by wide exclusion gaps [0.15, 0.30] and [0.70, 0.85].
- **2026-09-29 | Physical NaI(Tl) spectrometry channel allocation policy**
  - **Reason**: Grounded in standard gamma spectrometry principles for 2"x2" to 3"x3" NaI(Tl) scintillation detectors (7–9% FWHM energy resolution at 662 keV).
  - **Values**:
    - **Cs-137 (661.7 keV)**: Photopeak in R05 (601–800 keV; 45% of counts), Compton continuum in R02 (20%), R03 (15%), and R04 (20%). R06–R09 = 0.0% (no emission above 662 keV).
    - **I-131 (364.5 keV)**: Photopeak in R03 (201–400 keV; 65% of counts), Compton continuum in R02 (25%), forward scatter/weak lines in R04 (5%) and R05 (5%). R06–R09 = 0.0% (no emission above 637 keV).
    - **Physical Discrimination against Radon Washout**: Natural radon progeny (Bi-214) emit prominent high-energy gamma lines at 1120.3 keV (R07) and 1764.5 keV (R08), accounting for ~5.4% of total excess counts during rain. In sharp contrast, pure Cs-137 and I-131 plumes produce strictly zero counts in R07 and R08, providing an indelible spectral discriminator.
  - **Citations**:
    - Knoll, G. F. (2010). *Radiation Detection and Measurement* (4th ed.). John Wiley & Sons (pp. 338–342).
    - Vieira et al. (2019). *Environmental Research*, 175, 221–227. https://doi.org/10.1016/j.envres.2019.05.032.
- **2026-09-29 | Dedicated hard-case rain-coincident injection set (30% of test set)**
  - **Reason**: Real atmospheric fission plumes are subject to wet scavenging and precipitation washout. To evaluate whether models can discriminate real plumes from harmless natural washout when both occur simultaneously, exactly 30% of test injections (12 per station, 60 total) are forced to start during verified active rain ($P_{1\text{h}} > 0$).
- **2026-09-29 | Non-circular ground-truth labeling rule for natural radon washout**
  - **Reason**: Historical RadNet data lacks external ground-truth labels. An operational labeling rule is established: an hour is labeled `radon_washout` if $P_{3\text{h}} > 0\text{ mm}$ and gross CPM exceeds the station's dry baseline by $> 2\sigma_{\text{dry}}$ ($x(t) > \mu_{\text{dry}} + 2\sigma_{\text{dry}}$) on complete 8-channel records.
  - **Methodological Limitation**: Acknowledged per Section 5 of `PROJECT_SPEC.md` that rule-based labeling is partially circular. Minor rain showers without detectable count rate surges remain labeled `normal`, and potential sensor drift during rain could be misclassified.




