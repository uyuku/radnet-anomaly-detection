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

### Phase 3: Synthetic Injection Design Decisions (Revised & Grounded - Second Pass)

- **2026-09-29 | Real filter step-drop synchronization & clearing physics (Addressing Review Item 2.1)**
  - **Reason**: To eliminate artificial artifacts where synthetic injected plumes ride through real physical filter changes or end abruptly without one:
    1. **Step Detector Integration**: Candidate injection windows are evaluated against the Gate 2 dry 3-hour negative step detector (`src/analyze_filter_cycles_multi_station.py`).
    2. **Intake Disruption Rejection**: If a detected real filter replacement occurs during the plume intake / passage phase ($t < t_{\text{start}} + T_{\text{passage}}$), the candidate window is rejected and resampled, preventing unnatural disruption of the plume build-up.
    3. **Retention Truncation & Synchronization**: If a detected real filter replacement occurs during the retention phase ($t_{\text{start}} + T_{\text{passage}} \le t < t_{\text{start}} + D$), the injection retention is truncated to terminate at that exact hour ($D = t_{\text{drop}} - t_{\text{start}} + 1$). The synthetic particulate activity drops to zero synchronously with the real physical filter replacement step drop on the station (43 injections in the catalog truncated cleanly).
    4. **Unsynchronized Windows**: If no filter replacement is detected within the window, the sampled duration $D$ is retained (representing ambient cloud departure, noble gas component, or filter replacements occurring under wet/subtle conditions where dry 3-hour differential thresholding cannot trigger).
    5. **Documented Limitation**: The step detector operates exclusively during dry spells ($P = 0.0\text{ mm/h}$) where 3-hour drops exceed $-250\text{ to }-450\text{ CPM}$. Filter replacements performed during active precipitation or with subtle count drops cannot be identified by differential thresholding.
- **2026-09-29 | Traceable empirical dose coupling & core physical assumptions (Addressing Review Item 2.2)**
  - **Reason**: Ambient dose rate coupling $\Delta \text{Dose}(t) = k_{\text{dose}} \cdot \Delta \text{Gross CPM}(t)$ is calibrated directly via dedicated empirical regressions in [`src/calibrate_dose_coupling.py`](file:///Users/o/Projects/radnet-anomaly-detection/src/calibrate_dose_coupling.py), producing [`data/processed/dose_rate_cpm_regression_summary.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/dose_rate_cpm_regression_summary.csv) across 5,623 verified rain hours:
    - Birmingham: $n=1,863$, slope $= 0.01399\text{ nSv/h per CPM}$, $R^2 = 0.831$, $95\%\text{ CI } [0.01371, 0.01428]$.
    - Washington, DC: $n=1,748$, slope $= 0.01687\text{ nSv/h per CPM}$, $R^2 = 0.745$, $95\%\text{ CI } [0.01641, 0.01733]$.
    - San Diego: $n=110$, slope $= 0.02388\text{ nSv/h per CPM}$, $R^2 = 0.532$, $95\%\text{ CI } [0.01965, 0.02811]$.
    - Dallas: $n=1,237$, slope $= 0.01246\text{ nSv/h per CPM}$, $R^2 = 0.916$, $95\%\text{ CI } [0.01225, 0.01267]$.
    - Tampa: $n=665$, slope $= 0.01191\text{ nSv/h per CPM}$, $R^2 = 0.815$, $95\%\text{ CI } [0.01147, 0.01234]$.
    - Pooled Network: $n=5,623$, slope $= 0.01357\text{ nSv/h per CPM}$, $R^2 = 0.790$, $95\%\text{ CI } [0.01339, 0.01375]$.
  - **Two Mandatory Stated Assumptions**:
    1. *Photon Energy Dependence*: In nature, dose delivered per photon varies substantially with gamma energy ($^{60}\text{Co}$ at 1.17/1.33 MeV delivers significantly higher dose per photon than $^{131}\text{I}$ at 364 keV). Here, a unified empirical $k_{\text{dose}}$ distribution calibrated from RadNet field data is used across nuclide mixes.
    2. *Non-Shortcut / Identical Ratio*: Because $k_{\text{dose}}$ is calibrated to empirical radon washout events, the dose-to-gross ratio for synthetic fission injections is identical by construction to that of natural radon washout. Therefore, ambient dose rate carries almost no class-separating information between washout and fission plumes. This is a deliberate, conservative design choice preventing models from exploiting dose rate as an artificial shortcut.
  - **Citation Context**: NCRP Report No. 50 (1976), pp. 45–48 discusses environmental radiation exposure rates from radon progeny and fallout, but the numeric CPM-to-dose conversion factor is empirically derived from RadNet observations, not an NCRP table.
- **2026-09-29 | Fission spectra relabeled as operational working templates & broadened sampling (Addressing Review Item 2.3)**
  - **Reason**: The nominal spectra are explicitly designated as **operational working approximations / semi-empirical templates**, NOT fundamental derived quantities. While the principal gamma line energies (662 keV for Cs-137; 364 keV for I-131; 1173 & 1332 keV for Co-60; 605, 796, 802 keV for Cs-134) are fundamental nuclear data (ENSDF/IAEA), the channel fractions recorded by a 2.5" NaI(Tl) detector depend on crystal response, geometry, and Compton scatter.
  - **Broadened Dispersion**: To prevent classifiers from memorizing rigid channel ratios, channel shares are sampled across broad continuous Dirichlet and uniform distributions per injection:
    - Photopeak fractions vary widely: Cs-137 in R05 drawn from $[0.38, 0.55]$; I-131 in R03 drawn from $[0.52, 0.70]$; Co-60 in R07 drawn from $[0.22, 0.35]$.
    - High-energy scatter tails (R06–R08) vary independently from $0.5\%\text{ to }3.5\%$.
    - Relative channel perturbation ($\pm 15\%$ Gaussian) applied to all channels.
  - **Phase 5 Plan**: A formal sensitivity sweep over the photopeak-to-total ratio will be executed in Phase 5 to assess model robustness against detector response variations.
- **2026-09-29 | Operational release scenarios, Cs-134 inclusion, & Phase 5 reporting (Addressing Review Item 2.4)**
  - **Reason**: Test injections are structured under 5 concrete operational radiological release scenarios:
    1. `fission_reactor_fukushima`: Fresh reactor fission inventory containing volatile fission products $^{137}\text{Cs}$, $^{134}\text{Cs}$, and $^{131}\text{I}$ in realistic proportions (Masson et al. 2011; Steinhauser et al. 2014). High-energy share (R07+R08) is ~2% to 4%, directly overlapping or below radon progeny.
    2. `fission_pure_cs137`: Legacy sealed source / industrial gauge breach or dispersal ($^{137}\text{Cs} \ge 85\%$).
    3. `fission_pure_i131`: Radiopharmaceutical / medical isotope release ($^{131}\text{I} \ge 85\%$).
    4. `activation_orphan_co60`: Orphan industrial radiography / radiotherapy source breach or scrap metal smelting incident (e.g. Ciudad Juárez 1983, Algeciras 1998, Goiânia). Emits prominently in R07 (22%–35%), demonstrating detection of non-fission activation threats.
    5. `mixed_fission_activation`: Severe core excursion with structural activation debris (Cs-137 + Cs-134 + I-131 + Co-60). High-energy share is ~5% to 10%, directly spanning the radon 5.3% threshold.
  - **Phase 5 Commitments**:
    - Detection probability will be evaluated and reported separately by nuclide scenario (Cs/I fission vs Co activation vs mixed) and rain state (dry vs rain).
    - A 3-tier feature ablation will be conducted: (1) Gross radiation only, (2) Radiation + spectral ratios, (3) Radiation + spectral ratios + weather fusion, isolating the true marginal gain of weather data.
- **2026-09-29 | Citation corrections & empirical washout spectrum accounting (Addressing Review Item 2.5)**
  - **Citations**:
    - Masson et al. (2011) DOI corrected to `10.1021/es2017158`. Title: "Tracking of Airborne Radionuclides from the Damaged Fukushima Dai-ichi Nuclear Reactors by European Networks", *Environ. Sci. Technol.* 2011, 45(18), 7670–7677.
    - Radioiodine collection: RadNet stationary monitors draw air through glass-fiber particulate filters which collect aerosol-bound radioiodine. Gaseous radioiodine species ($I_2, CH_3I$) penetrate particulate filters and require charcoal cartridges analyzed off-site. The synthetic injection amplitude models the *effective particulate activity deposited on the filter*, rather than total atmospheric gaseous release.
  - **Washout Spectrum Script**: [`src/analyze_washout_spectrum.py`](file:///Users/o/Projects/radnet-anomaly-detection/src/analyze_washout_spectrum.py) updated to compute rolling 24h precipitation strictly on the continuous calendar grid prior to filtering for observed hours. Across 5,485 verified rain hours (>100 CPM excess), pooled shares are: R02: 43.95%, R03: 32.34%, R04: 9.05%, R05: 5.92%, R06: 2.43%, R07: 3.81%, R08: 1.54%, R09: 0.96% (R07+R08 = 5.35%).
  - **Tampa Geology & Spectrum**: Tampa exhibits higher R02 (61.81%) and lower R03 (18.41%) due to lower terrestrial background from Florida limestone/sand geology (dry baseline R02 is 881 CPM in Tampa vs 1,973 CPM in Birmingham) and PMT gain/threshold variations across monitor models.
- **2026-09-29 | Rain-onset anchoring & washout overlap logging (Addressing Review Item 2.6)**
  - **Reason**: Forced-rain injections are anchored directly to precipitation onsets ($P_{\text{1h}} \ge 1.0\text{ mm/h}$ following dry hours or at the start of a storm surge). This immerses the plume's rising intake phase directly inside the rising radon washout surge. Each injection logs `washout_overlap_hours` (total concurrent rain hours) and `rise_washout_overlap_hours` (concurrent rain hours during plume rise) in the catalog.
- **2026-09-29 | Exact duration alignment & hard-regime framing (Addressing Review Item 2.7)**
  - **Reason**: Duration parameters are strictly aligned across scripts, catalog, and documentation:
    - Standard test: 36 to 80 hours (clamped exact).
    - Stress test: 8 to 20 hours (clamped exact).
    - Training: 10 to 36 hours.
    - Stress test set is framed as a "hard-regime test set" testing low amplitude (Band A) and short duration during active rain, rather than asserting total disjointness from training.
- **2026-09-29 | Phase 4 & Phase 5 Architectural Policies (Addressing Review Section 3 Guidance)**
  - **Dual Series Generation**: Two series per station and split will be constructed:
    1. *Clean Background Series*: Contains zero synthetic injections, used exclusively for computing baseline features and counting false alarms.
    2. *Injected Series*: Contains all injections applied before feature engineering (e.g. 168h rolling statistics), used exclusively for evaluating detection probability and delay.
  - **Observed Hours Denominator**: False alarm rates per station-year will use verified observed hours ($N_{\text{obs}} / 8,766$) as the denominator, preventing distortion from station outages (e.g. Dallas 2023–2025 outage of 10,826 unobserved hours).
  - **Leave-One-Station-Out (LOSO) Validation**: Models will be validated with LOSO cross-validation, holding out San Diego (which exhibits negative rain-radiation correlation due to marine atmospheric inversions).
  - **Washout Class as Diagnostic**: The `radon_washout` label is preserved strictly as an auxiliary diagnostic tool; the core headline metric is false alarms per station-year at fixed detection on unmodified background data.





