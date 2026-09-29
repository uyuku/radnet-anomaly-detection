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

### Phase 3: Synthetic Injection Design Decisions (Final Revision - Training Freeze)

- **2026-09-29 | Real filter step-drop synchronization & clearing physics (Addressing Review Item 2.1)**
  - **Reason**: To completely eliminate artificial artifacts where synthetic injected plumes ride through real physical filter changes or end abruptly in unphysical cliffs without a real filter drop:
    1. **100% Filter Drop Anchoring**: Every single candidate injection window is anchored backwards directly from an empirically detected real physical filter replacement step drop (`end_utc = drop_utc`), ensuring $D = t_{\text{drop}} - t_{\text{start}} + 1$. Exactly 450 of 450 injections (100.0%) terminate at a real detected physical filter drop (`truncated_by_filter_change = True`).
    2. **Zero Synthetic-Only Cliffs**: Because the synthetic particulate accumulation resets to zero at the exact physical hour when the real station filter is replaced, zero injections produce unphysical step-drop cliffs against a flat background.
    3. **Intermediate Drop Rejection**: Any candidate window spanning an intermediate detected filter replacement between $t_{\text{start}}$ and $t_{\text{drop}}$ is strictly rejected and resampled, preventing plume intake disruption or multiple filter changes within a single event window.
    4. **Filter Cadence Verification**: To verify that backward anchoring does not introduce an artificial cadence feature distinguishing injected onsets from normal operational cycles, the elapsed time since the previous detected drop (`hours_since_last_detected_drop`) is logged at injection onset. Injected onsets exhibit a median of 84.0 h (IQR [40.2 h, 176.8 h], 5–95th [9.0 h, 525.9 h]), closely matching the natural baseline distribution across normal observed hours (median 97.0 h, IQR [41.0 h, 226.0 h], 5–95th [8.0 h, 779.2 h]).
- **2026-09-29 | Balanced scenario proportions across all environments (Addressing Review Item 2.2)**
  - **Reason**: To eliminate confounding between meteorological rain state and radiological scenario mix:
    - In the previous pass, training rain injections had 0 pure Cs-137 and 0 pure I-131, while dry injections had 0 Co-60, allowing models to exploit an artificial nuclide-weather association.
    - In this final revision, every training and test environment contains exactly 20.0% of each of the 5 operational scenarios:
      - `train_strictly_dry` ($N=175$): exactly 35 per scenario (20.0%)
      - `train_rain_coincident` ($N=75$): exactly 15 per scenario (20.0%)
      - `test_standard_strictly_dry` ($N=50$): exactly 10 per scenario (20.0%)
      - `test_standard_rain` ($N=50$): exactly 10 per scenario (20.0%)
      - `test_stress_strictly_dry` ($N=50$): exactly 10 per scenario (20.0%)
      - `test_stress_rain` ($N=50$): exactly 10 per scenario (20.0%)
    - Across the network, each scenario has exactly 90 injections (20.0% of 450). This guarantees that pure Cs-137 and pure I-131 plumes rising inside active rain storms are fully represented in training (15 each) and test (20 each), preventing models from relying on spurious weather correlations.
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
- **2026-09-29 | Operational release scenarios, Cs-134 inclusion, & empirical high-energy overlap (Addressing Review Item 2.3 & 2.4)**
  - **Reason**: Test injections are structured under 5 concrete operational radiological release scenarios:
    1. `fission_reactor_fukushima`: Fresh reactor fission inventory containing volatile fission products $^{137}\text{Cs}$, $^{134}\text{Cs}$, and $^{131}\text{I}$ in realistic proportions (Masson et al. 2011; Steinhauser et al. 2014). High-energy $^{134}\text{Cs}$ gamma emissions at 1168 keV (1.8% yield) and 1365 keV (3.0% yield) fall directly into Channel R07. Modeled via $p_{\text{R07, Cs134}} \sim \mathcal{U}[0.025, 0.055]$, yielding an R07+R08 share:
       - Catalog ($N=90$): Mean $= 4.54\%$, 5–95th $= [2.82\%, 6.47\%]$, IQR $= [3.65\%, 5.42\%]$
       - 4,000 draws: Mean $= 4.51\%$, 5–95th $= [2.84\%, 6.48\%]$, IQR $= [3.65\%, 5.28\%]$
       - Directly spans and overlaps the empirical radon washout ratio-of-sums of **5.58%** (95% CI: [5.52%, 5.65%]) and the empirical hour-level IQR band [4.04%, 7.21%].
    2. `fission_pure_cs137`: Legacy sealed source / industrial gauge breach or dispersal ($^{137}\text{Cs} \ge 85\%$). R07+R08 share: Catalog Mean $= 1.30\%$ [0.67%, 2.02%]; 4,000 draws Mean $= 1.32\%$ [0.78%, 1.91%].
    3. `fission_pure_i131`: Radiopharmaceutical / medical isotope release ($^{131}\text{I} \ge 85\%$). R07+R08 share: Catalog Mean $= 1.35\%$ [0.90%, 1.94%]; 4,000 draws Mean $= 1.31\%$ [0.78%, 1.94%].
    4. `activation_orphan_co60`: Orphan industrial radiography / radiotherapy source breach or scrap metal smelting incident (e.g. Ciudad Juárez 1983, Algeciras 1998, Goiânia). Emits prominently in R07 (1173 & 1332 keV photopeaks). R07+R08 share: Catalog Mean $= 30.72\%$ [22.33%, 39.89%]; 4,000 draws Mean $= 30.99\%$ [22.88%, 39.75%].
    5. `mixed_fission_activation`: Severe core excursion with structural activation debris (Cs-137 + Cs-134 + I-131 + Co-60). R07+R08 share: Catalog Mean $= 12.17\%$ [8.05%, 17.70%]; 4,000 draws Mean $= 11.51\%$ [7.63%, 16.08%].
  - **Full Literature Citations**:
    - Masson, O., Baeza, A., Bieringer, J., Brudecki, K., Bucci, S., Cappai, M., ... & Wershofen, H. (2011). Tracking of Airborne Radionuclides from the Damaged Fukushima Dai-ichi Nuclear Reactors by European Networks. *Environmental Science & Technology*, 45(18), 7670–7677. https://doi.org/10.1021/es2017158.
    - Steinhauser, G., Brandl, A., & Johnson, T. E. (2014). Comparison of the Chernobyl and Fukushima nuclear accidents: A review of the environmental impacts. *Science of the Total Environment*, 470–471, 800–817. https://doi.org/10.1016/j.scitotenv.2013.10.029.
  - **Phase 5 Commitments**:
    - Detection probability will be evaluated and reported separately by nuclide scenario (Cs/I fission vs Co activation vs mixed) and rain state (dry vs rain).
    - A 3-tier feature ablation will be conducted: (1) Gross radiation only, (2) Radiation + spectral ratios, (3) Radiation + spectral ratios + weather fusion, isolating the true marginal gain of weather data.
- **2026-09-29 | Ratio-of-sums washout spectrum accounting & empirical baseline framing (Addressing Review Item 2.3 & 2.5)**
  - **Washout Spectrum Script**: [`src/analyze_washout_spectrum.py`](file:///Users/o/Projects/radnet-anomaly-detection/src/analyze_washout_spectrum.py) updated to compute ratio-of-sums $\sum \Delta C_k / \sum \Delta \text{Gross}$ across 5,485 verified rain hours (>100 CPM excess gross, $P_{\text{1h}} \ge 1.0\text{ mm/h}$) on the continuous calendar grid:
    - Pooled shares: R02: 42.31%, R03: 33.73%, R04: 9.16%, R05: 6.04%, R06: 2.48%, R07: 3.87%, R08: 1.72%, R09: 0.70%.
    - Combined R07+R08 ratio of sums: **5.58%** (1,000-draw bootstrap 95% CI: **[5.52%, 5.65%]**).
    - Per-station R07+R08 ratio of sums: Birmingham 5.63% [5.57, 5.69], Washington DC 5.73% [5.63, 5.84], San Diego 3.52% [2.15, 4.84], Dallas 5.69% [5.58, 5.80], Tampa 4.67% [4.27, 5.07].
    - Pooled hour-level percentiles of R07+R08 share: 5th: -3.34%, 25th: 4.04%, 50th (median): 5.64%, 75th: 7.21%, 95th: 13.45% (hour mean 5.35%).
  - **Tampa Spectrum & Baseline Count Rates (Sanitized of Causal Claims)**:
    - Tampa exhibits an empirically higher R02 ratio of sums (53.49%) and hour mean (61.81%) and lower R03 (ratio of sums 24.73%, hour mean 18.41%).
    - Traceable baseline counts: As reported in [`candidate_pilot_stations_comparison.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/candidate_pilot_stations_comparison.csv) and empirical dry background records ($\text{precip} = 0, \text{precip}_{24\text{h}} = 0$), Tampa's dry baseline channel counts are lower than Birmingham's across all lower channels (Tampa dry mean R02 is 882.7 CPM against 1,980.4 CPM in Birmingham; dry gross mean is 2,081.8 CPM in Tampa vs 3,887.3 CPM in Birmingham).
    - Causal geological hypotheses (limestone/sand substrate) and instrument differences are removed from operational decisions and documented as open research questions in `QUESTIONS.md`.
  - **Particulate Radioiodine Framing**:
    - RadNet stationary monitors draw air through glass-fiber particulate filters which collect aerosol-bound radioiodine. Gaseous radioiodine species ($I_2, CH_3I$) penetrate particulate filters and require charcoal cartridges analyzed off-site. The synthetic injection amplitude models the *effective particulate activity deposited on the filter*, rather than total atmospheric gaseous release.
- **2026-09-29 | Rain-onset anchoring, overlap logging, & pure Cs-137 walkthrough (Addressing Review Item 2.4 & 2.6)**
  - **Reason**: Forced-rain injections are anchored directly to precipitation onsets ($P_{\text{1h}} \ge 1.0\text{ mm/h}$ after dry hours or at the start of a storm surge), immersing the plume's rising intake phase directly inside the rising radon washout surge.
  - `washout_overlap_hours` and `rise_washout_overlap_hours` are computed strictly on the realized injection window (from start to the filter drop).
  - Walkthrough figure ([`reports/figures/synthetic_injection_hard_case_rain.png`](file:///Users/o/Projects/radnet-anomaly-detection/reports/figures/synthetic_injection_hard_case_rain.png)) features `INJ_0067` in Birmingham, AL (Dec 9–10, 2024): a subtle Band A (479.1 CPM peak) pure Cs-137 plume rising inside a 13.5 mm/h rainstorm and ending at a verified physical filter change drop (3,885 $\to$ 3,578 CPM background drop), illustrating the difficult regime where the plume has zero R07 signature during the storm.
- **2026-09-29 | Exact duration ranges, strict 48h buffer, & catalog spectrum storage (Addressing Review Item 2.4 & 2.7)**
  - **Reason**: Duration parameters and constraints are aligned across scripts, catalog, and documentation:
    - **Strict 48-Hour Buffer**: Enforced across all injection branches. Exactly 0 buffer violations (<48h) exist across all 450 injections in the catalog.
    - **Exact Realized Duration Ranges**:
      - `train_strictly_dry`: 10 to 35 hours (Mean 19.1 h, $N=175$)
      - `train_rain_coincident`: 25 to 94 hours (Mean 52.3 h, $N=75$)
      - `test_standard_strictly_dry`: 36 to 75 hours (Mean 47.7 h, $N=50$)
      - `test_standard_rain`: 38 to 164 hours (Mean 78.4 h, $N=50$)
      - `test_stress_strictly_dry`: 8 to 20 hours (Mean 12.8 h, $N=50$)
      - `test_stress_rain`: 28 to 140 hours (Mean 61.8 h, $N=50$)
    - **Catalog Spectrum Storage**: The eight sampled channel shares (`share_r02` through `share_r09`) are logged directly in `data/processed/synthetic_injection_catalog.csv` and applied identically when building the labeled datasets, allowing Phase 5 evaluations to stratify by spectral share directly from catalog metadata.
    - Stress test set is framed as a "hard-regime test set" testing low amplitude (Band A) and short duration during active rain.
- **2026-09-29 | Phase 4 & Phase 5 Architectural Policies (Addressing Review Section 3 Guidance)**
  - **Dual Series Generation**: Two series per station and split will be constructed:
    1. *Clean Background Series*: Contains zero synthetic injections, used exclusively for computing baseline features and counting false alarms.
    2. *Injected Series*: Contains all injections applied before feature engineering (e.g. 168h rolling statistics), used exclusively for evaluating detection probability and delay.
  - **Observed Hours Denominator**: False alarm rates per station-year will use verified observed hours ($N_{\text{obs}} / 8,766$) as the denominator, preventing distortion from station outages (e.g. Dallas 2023–2025 outage of 10,826 unobserved hours).
  - **Leave-One-Station-Out (LOSO) Validation**: Models will be validated with LOSO cross-validation, holding out San Diego (which exhibits an empirical negative rain-radiation correlation, $r = -0.06$ to $-0.13$).
  - **Washout Class as Diagnostic**: The `radon_washout` label is preserved strictly as an auxiliary diagnostic tool; the core headline metric is false alarms per station-year at fixed detection on unmodified background data.

---

### Phase 4: Model Training, Ablation Hierarchy & Evaluation Decisions

- **2026-09-29 | Three-Tier Feature Ablation Hierarchy**
  - **Reason**: Formally evaluates the incremental contribution of physical feature families:
    - *Tier 1 (Gross Radiation Only, 12 features)*: Rolling means (3h, 6h, 24h, 168h), short-term differences ($\Delta_{1\text{h}}, \Delta_{3\text{h}}, \Delta_{6\text{h}}$), rolling Z-scores, global dry baseline Z-score ($(x-\mu_{\text{dry}})/\sigma_{\text{dry}}$), and exposure rate features.
    - *Tier 2 (Radiation + NaI(Tl) Spectral Ratios, 30 features)*: All Tier 1 features plus 8 channel energy fractions ($R02/\text{Gross} \dots R09/\text{Gross}$), high-energy share ($(R07+R08)/\text{Gross}$), photopeak-to-washout ratios ($R05/R07, R03/R07, R05/R03, R02/R03$), and rolling 6h spectral shares.
    - *Tier 3 (Full Weather Fusion, 48 features)*: All Tier 2 features plus NOAA ASOS hourly precipitation ($P_{1\text{h}}, P_{3\text{h}}, P_{6\text{h}}, P_{24\text{h}}$), hours since last rain, barometric pressure tendencies ($\Delta P_{3\text{h}}, \Delta P_{24\text{h}}$), temperature, dewpoint, relative humidity, and cross-domain interaction terms (`washout_expected_ratio`, `rain_high_energy_interaction`, `dry_excess_interaction`).
  - **Result**: Tier 2 spectral ratios achieve the largest jump in false alarm reduction (from 124.3 to 3.12 FA/yr at 90% detection), while Tier 3 weather fusion provides critical physical context during active precipitation events and hard-case plumes.
- **2026-09-29 | Dual-Series Execution Protocol (Clean vs Injected)**
  - **Reason**: Strict compliance with the Phase 3 gate policy:
    - *Injected Series*: Injections applied before computing rolling statistics; models predict $P(\text{fission})$ to measure event detection rate and delay.
    - *Clean Background Series*: Clean unmodified observations; models predict $P(\text{fission})$ to measure operational false alarm rate on genuine operational background.
    - An event is counted as detected if $\max_{t \in \text{event}} P(\text{fission}_t) \ge \tau$ during observed event hours.
    - Clean alarms are clustered into contiguous discrete episodes ($N_{\text{episodes}}$) and normalized by $N_{\text{obs}} / 8,766$ station-years.
- **2026-09-29 | Multi-Class LightGBM with Balanced Weighting**
  - **Reason**: Multi-class formulation (`normal` [0], `radon_washout` [1], `fission_product` [2]) allows the gradient booster to partition tree leaves into distinct physical subspaces rather than conflating background and washout into a bimodal negative class.
  - **Hyperparameters**: `n_estimators=150`, `learning_rate=0.05`, `max_depth=6`, `num_leaves=31`, `subsample=0.8`, `colsample_bytree=0.8`, `class_weight='balanced'`, fixed seed 42.
- **2026-09-29 | Leave-One-Station-Out (LOSO) Generalization on San Diego**
  - **Reason**: To test external generalization across climate regimes, San Diego (West Coast Mediterranean / coastal climate with empirical negative rain-radiation correlation, $r = -0.06$ to $-0.13$) was held out entirely from training.
  - **Result**: The Tier 3 model trained on the other 4 stations achieves **87.5% event detection** (35 of 40 events) on held-out San Diego test data with only **3.86 clean false alarms per station-year**, proving scale-invariant feature generalization without site-specific re-tuning.





