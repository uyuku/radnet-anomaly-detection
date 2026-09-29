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

### Phase 3: Synthetic Injection Design Decisions (Revised & Grounded)

- **2026-09-29 | Headline evaluation metric policy (Addressing Review Item 2.7)**
  - **Reason**: To eliminate vulnerability to label circularity in natural radon washout, the project adopts Claude's recommendation and Section 6 of `PROJECT_SPEC.md`: the primary headline metric is **false alarms per station-year at a fixed detection probability for injected fission events** evaluated on **unmodified real background data**. The three-class breakdown (`normal`, `radon_washout`, `fission_product`) is retained strictly as an auxiliary diagnostic, guaranteeing that the core scientific claim does not rely on a circular label definition.
- **2026-09-29 | Multi-nuclide inventory & elimination of spectral shortcut (Addressing Review Item 2.1)**
  - **Reason**: The initial draft assigned strictly 0.0% counts to R06–R09 for fission injections, handing classifiers an artificial shortcut against radon washout (where Bi-214 emits in R07 and R08). In real radiological emergencies, anthropogenic plumes contain isotopes with lines above 800 keV:
    - **$^{60}\text{Co}$** ($T_{1/2} = 5.271\text{ y}$): Prominent cascading gamma lines at 1173.2 keV and 1332.5 keV fall directly inside **Channel R07 (1001–1400 keV)**, emitting 28% of counts into R07 and Compton scatter across R02–R06.
    - **$^{134}\text{Cs}$** ($T_{1/2} = 2.065\text{ y}$): Major gamma lines at 604.7 keV (R05), 795.9 keV (R05/R06), 802.0 keV (R06), and 1365.2 keV (R07).
    - **High-energy scatter/continuum floor**: A realistic continuum/pileup tail (1.5% in R06, 0.5% in R07) is included for all nuclides.
    - **Result**: Anthropogenic plumes can and do elevate Channel R07. A classifier cannot use `R07 > 0` as a trivial shortcut for radon washout.
  - **Citations**:
    - International Atomic Energy Agency (IAEA) Nuclear Data Section / Evaluated Nuclear Structure Data File (ENSDF).
    - Heath, R. L. (1964). *Scintillation Spectrometry Gamma-Ray Spectrum Catalogue* (2nd ed., IDO-16880). Phillips Petroleum Company / U.S. Atomic Energy Commission.
    - Knoll, G. F. (2010). *Radiation Detection and Measurement* (4th ed.). John Wiley & Sons.
- **2026-09-29 | Ambient dose rate injection coupling (Addressing Review Item 2.1)**
  - **Reason**: Particulate gamma plumes collected on the filter and passing overhead elevate ambient exposure rate probes. Without dose-rate injection, the Phase 2 baseline "Global Dose Rate" rule would be blind by construction. A physical dose-rate coupling $\Delta \text{Dose}(t) = k_{\text{dose}} \cdot \Delta \text{Gross CPM}(t)$ is injected into `dose_rate_nsvh`, with $k_{\text{dose}} \sim \mathcal{N}(0.016, 0.002)\text{ nSv/h per CPM}$ (clamped to $[0.012, 0.020]$), matching empirical rain regressions across the pilot stations ($0.014\text{ to }0.021\text{ nSv/h per CPM}$) and health physics standards (NCRP Report No. 50).
  - **Citation**: National Council on Radiation Protection and Measurements (NCRP). (1976). *Environmental Radiation Measurements* (NCRP Report No. 50).
- **2026-09-29 | Filter accumulation, retention decay, and replacement physics (Addressing Review Item 2.2)**
  - **Reason**: Resolves the unrealistic single-hour "cliff" drop and incorporates genuine filter operation:
    1. **Plume Passage Phase** ($t < T_{\text{passage}}$): Plume passes overhead and air is continuously sampled through the filter at ~60 m³/h. Activity accumulates cumulatively: $A(t) = \sum_{u=0}^t C(u) e^{-\lambda (t-u)}$.
    2. **Retention Phase** ($T_{\text{passage}} \le t < D$): Plume has passed ($C=0$). Particulates remain trapped on the filter, decaying purely according to radiological half-life ($e^{-\lambda t}$).
    3. **$^{131}\text{I}$ Radiological Decay**: Applied during accumulation and retention using $\lambda = \ln(2) / (8.025 \times 24\text{ h}) = 0.00360\text{ h}^{-1}$ (~8.3% loss per 24 hours, ~12% loss over 36h retention).
    4. **Filter Replacement Drop**: Activity terminates ($S(t) \to 0$) strictly when the filter is replaced (total duration $D = T_{\text{passage}} + T_{\text{retention}}$, bounded by empirical filter replacement intervals of ~4 days).
    5. **Iodine Chemistry Assumption**: Explicitly models the **particulate-bound fraction** of radioiodine collected on the glass fiber filter (typically 10%–30% in environmental releases; gaseous fraction collected on downstream charcoal cartridges is analyzed off-site).
  - **Citations**:
    - Masson, O., et al. (2011). Tracking of airborne radionuclides from the Fukushima Dai-ichi nuclear accident. *Environmental Science & Technology*, 45(18), 7670–7677. https://doi.org/10.1021/es201605m.
    - U.S. EPA (2005). *RadNet Air Sampling Procedures* (EPA 402-R-05-001).
- **2026-09-29 | Scripted empirical washout spectrum & random perturbation (Addressing Review Item 2.3)**
  - **Reason**: Replaced hardcoded constants with a dedicated analysis script ([`src/analyze_washout_spectrum.py`](file:///Users/o/Projects/radnet-anomaly-detection/src/analyze_washout_spectrum.py)) computing excess channel shares across 5,479 verified substantial rain hours across all 5 stations ([`rain_washout_spectral_shares.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/rain_washout_spectral_shares.csv)):
    - Pooled shares: R02: 43.96%, R03: 32.37%, R04: 9.04%, R05: 5.91%, R06: 2.43%, R07: 3.81%, R08: 1.54%, R09: 0.94% (R07+R08 = 5.34%).
    - Random spectral perturbation ($\pm 10\%$ relative Gaussian per channel) applied to every synthetic injection so the model cannot memorize fixed channel ratios.
    - Channel R01 ($\le 100\text{ keV}$) is omitted by EPA as a noise threshold; all shares are defined over reported channels R02–R09.
- **2026-09-29 | Strict non-overlap constraint & true dry partitioning (Addressing Review Items 2.4 & 2.5)**
  - **Reason**: Solved catalog overlaps by tracking calendar index occupancy with a 48-hour buffer before and after each injection. Detected overlapping injection pairs reduced from 37 to **exactly 0**. Standard dry injections are required to have $P_{1\text{h}} = 0.0\text{ mm}$ across the entire injection window and preceding 24h. Forced rain-onset injections have $P_{1\text{h}} \ge 1.0\text{ mm}$ at onset, included in both Train (30%) and Test (50%).
- **2026-09-29 | Short-duration hard-regime stress test set (Addressing Review Item 2.6)**
  - **Reason**: Test set includes 100 short plumes (8–20h) with subtle magnitudes (Band A [250, 600] CPM) forced during active rain, directly testing the genuine hard regime where plume duration is comparable to storm duration.
- **2026-09-29 | Reconciled continuous calendar accounting & missing data handling**
  - **Reason**: Timestamps parsed with `< "2023-01-01"` and `< "2026-01-01"`, capturing exactly 52,584 train hours and 26,304 test hours (78,888 hours per station, 394,440 hours across network). Missing RadNet data explicitly labeled `unobserved` (59,873 hours across network), not `normal`. All numbers verified from [`data/processed/labeled_dataset_reconciliation_summary.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/labeled_dataset_reconciliation_summary.csv).





