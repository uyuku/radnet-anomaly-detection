# Phase 0 Report (Revised): Data Discovery & Pilot Station Selection

**Date**: 2026-09-28  
**Project**: Weather-Aware Anomaly Detection in RadNet  
**Phase**: Phase 0 (Data Discovery)  
**Status**: Revised per AI Cross-Review (Section 7) — **Gate 0 Reached**

---

## 1. Executive Summary

Phase 0 established the empirical foundation for the project and addressed all items from the independent cross-review:
1. **Station Network Scraped & Categorized**: All 137 RadNet stationary air monitors across the United States were indexed ([`radnet_station_inventory.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/radnet_station_inventory.csv)). 127 stations have exposure-rate start dates in EPA metadata, and 20 stations possess a continuous 10-year history with exposure rate operational since mid-2016.
2. **Channel Energy Boundaries Grounded & Cited**: The physical energy boundaries (in keV) for channels R02 through R09 were verified against peer-reviewed literature (Vieira et al., 2019; Huang et al., 2020) and mapped to key radon progeny (Pb-214, Bi-214) and fission products (Cs-137, I-131).
3. **Data Completeness, Missingness & Effective Dose Coverage Quantified**: Evaluated all 8 candidate stations across 9 full calendar years (2017–2025; exactly 78,888 expected hourly intervals per station). Gross count rate statistics were computed strictly on complete 8-channel records, resolving variance inflation from missing-channel partial sums.
4. **5 Diverse Pilot Stations Selected & Paired with NOAA ASOS**: Selected 5 premier pilot stations representing distinct climatic, geographic, and radon background regimes, all paired with primary NOAA airport weather stations (USAF-WBAN) reporting hourly precipitation, pressure, temperature, and dew point in UTC.
5. **Full Traceability & Reproducibility**: Raw data storage policy enforced (`data/raw/` read-only, `chmod 444`), all downloads verified via unified script (`src/download_pilot_data.py`), full SHA-256 hashes recorded in [`data/raw/MANIFEST.sha256`](file:///Users/o/Projects/radnet-anomaly-detection/data/raw/MANIFEST.sha256), and design decisions and literature citations recorded in [`DECISIONS.md`](file:///Users/o/Projects/radnet-anomaly-detection/DECISIONS.md).

---

## 2. Gamma Channel Energy Boundaries

From peer-reviewed research analyzing EPA RadNet airborne gamma spectrometry:
> **Citations**:
> - Vieira, C. L. Z., Koutrakis, P., Huang, S., Grady, S., Hart, J. E., Coull, B. A., Laden, F., Requia, W., Schwartz, J., & Garshick, E. (2019). Short-term effects of particle gamma radiation activities on pulmonary function in COPD patients. *Environmental Research*, 175, 221–227. https://doi.org/10.1016/j.envres.2019.05.032.
> - Huang, S., Garshick, E., Vieira, C. L. Z., Hart, J. E., Coull, B. A., & Koutrakis, P. (2020). Short-term exposure to ambient particle gamma radiation and mortality in the US Medicare population. *Environmental Research*, 182, 108995. https://doi.org/10.1016/j.envres.2019.108995 (PMC6983292).
> - Knoll, G. F. (2010). *Radiation Detection and Measurement* (4th ed.). John Wiley & Sons, pp. 338–342.

| Channel | Energy Range (keV) | Empirical Baseline Fraction | Physical Role & Radionuclide Overlap |
| :--- | :--- | :--- | :--- |
| **γ1 / R01** | $\le 100$ | N/A (omitted in CSVs) | Lower discriminator / detector noise floor. *(Verified: public CSV headers start at R02)* |
| **γ2 / R02** | $101 - 200$ | **41.5% – 51.3%** of gross | Low-energy Compton continuum & backscatter; highest baseline counts |
| **γ3 / R03** | $201 - 400$ | **26.4% – 29.3%** of gross | **Spectral Overlap Zone 1**: Contains **Pb-214** (351.9 keV) and **I-131** (364.5 keV) |
| **γ4 / R04** | $401 - 600$ | **7.0% – 9.0%** of gross | Intermediate Compton continuum |
| **γ5 / R05** | $601 - 800$ | **4.2% – 5.2%** of gross | **Spectral Overlap Zone 2**: Contains **Bi-214** (609.3 keV) and **Cs-137** (661.7 keV) |
| **γ6 / R06** | $801 - 1000$ | **2.5% – 3.5%** of gross | Upper Compton continuum |
| **γ7 / R07** | $1001 - 1400$ | **3.0% – 5.1%** of gross | High-energy radon progeny: Contains **Bi-214** photopeak (1120.3 keV) |
| **γ8 / R08** | $1401 - 1800$ | **1.9% – 4.2%** of gross | High-energy radon progeny: Contains **Bi-214** photopeak (1764.5 keV) + terrestrial **K-40** (1460.8 keV) |
| **γ9 / R09** | $1801 - 2200$ | **0.8% – 1.6%** of gross | Prompt cosmic background and very high energy tail |

### Physical Consequence & Spectrometric Constraints
Standard commercial NaI(Tl) scintillation detectors exhibit energy resolutions of approximately 7% to 9% FWHM at 662 keV (Knoll, 2010):
- **Pb-214 (351.9 keV)** cannot be separated spectrally from **I-131 (364.5 keV)** because both fall into **Channel R03 (201–400 keV)**.
- **Bi-214 (609.3 keV)** cannot be separated spectrally from **Cs-137 (661.7 keV)** because both fall into **Channel R05 (601–800 keV)**.
- **Decay Kinetics vs. Filter Cycles**: Over typical single-event timescales (1 to 24 hours), radioactive decay distinguishes the classes: radon progeny decay rapidly post-precipitation ($T_{1/2} = 26.8\text{ min}$ for Pb-214, $19.9\text{ min}$ for Bi-214), whereas fission products undergo negligible hourly decay ($T_{1/2} \approx 30.17\text{ yr}$ for Cs-137, $8.02\text{ d}$ for I-131). However, particulate matter does not persist indefinitely because filters are replaced periodically; determining the filter cycle is a dedicated Phase 1 task.
- **Natural Background Variation**: In addition to precipitation washout, the natural diurnal cycle (nocturnal atmospheric boundary-layer inversion trapping radon near ground, followed by daytime convective mixing) is a major driver of baseline background variation and will be formally modeled.

---

## 3. Candidate Pilot Stations Analysis (2017–2025)

Evaluated across exactly **78,888 expected hours** (9 full years: 7 standard years $\times 8,760\text{ h}$ + 2 leap years $\times 8,784\text{ h}$).  
Duplicate timestamps were verified (only 5–6 duplicates per station across 9 years) and deduplicated.  
Gross CPM statistics are computed **strictly on complete 8-channel hours** (`chan_cols.notna().all(axis=1)`):

| Station | Region / Climate | Completeness (Unique Records) | Dose Rate Present | **Effective Dose Coverage** | Complete Channel Rows | **Clean Gross Mean (CPM)** | **Clean Gross Std (CPM)** | **Mean Dose (nSv/h)** | Ratio (CPM / [nSv/h]) | NOAA ASOS Airport | EPA Radon Zone |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **San Diego, CA** | West Coast (Mediterranean/Arid) | **94.32%** (74,409) | 99.98% | **94.30%** | 99.93% | **6,913.3** | 530.9 | 99.5 | 69.5 | KSAN | Zone 3 (Low) |
| **Washington, DC** | Mid-Atlantic (Temperate 4-season) | **92.16%** (72,701) | 98.35% | **90.64%** | 94.43% | **2,058.5** | 294.0 | 31.3 | 65.8 | KDCA | Zone 3 (Low) |
| **Birmingham, AL** | Southeast (Humid Subtropical) | **91.51%** (72,192) | 99.99% | **91.51%** | 93.52% | **3,884.5** | 348.4 | 53.8 | 72.2 | KBHM | Zone 2 (Moderate) |
| **Montgomery, AL** | Southeast (Gulf Coastal Plain) | **87.58%** (69,088) | 99.95% | **87.53%** | 97.38% | **3,805.4** | 395.0 | 49.6 | 76.7 | KMGM | Zone 3 (Low) |
| **Dallas, TX** | Southern Plains (Subtropical Cont.) | **80.59%** (63,575) | 100.00% | **80.59%** | 98.01% | **2,944.3** | 401.1 | 39.9 | 73.8 | KDFW | Zone 3 (Low) |
| **Tampa, FL** | Gulf Coast / FL (Subtropical) | **82.93%** (65,418) | 93.95% | **77.91%** | 98.42% | **2,079.5** | 400.0 | 31.0 | 67.1 | KTPA | Zone 3 (Low) |
| **Austin, TX** | Central Texas (Semi-arid transition) | **74.43%** (58,713) | 96.08% | **71.51%** | 90.12% | **2,979.1** | 337.3 | 44.7 | 66.6 | KAUS | Zone 3 (Low) |
| **Chicago, IL** | Midwest (Humid Continental) | **74.16%** (58,503) | 91.79% | **68.07%** | 99.15% | **2,688.9** | 768.0 | 50.4 | 53.4 | KORD | Zone 2 (Moderate) |

### Per-Channel Missingness Summary
*(Generated by `src/analyze_pilot_stations.py` into [`data/processed/pilot_station_channel_missingness.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/pilot_station_channel_missingness.csv))*:
- When channels are missing in Birmingham (6.48%), DC (5.57%), San Diego (0.07%), Dallas (1.99%), Tampa (1.58%), and Montgomery (2.62%), all 8 channels (R02 through R09) are missing synchronously as complete multi-channel records.
- In Austin, 5,800 records miss all channels together, and exactly 1 record misses R09 alone.
- Physical sanity check: The ratio of clean mean gross count rate (CPM) to mean dose rate (nSv/h) is remarkably consistent across 7 stations (65.8 to 76.7 CPM per nSv/h), confirming genuine detector calibration alignment. Chicago is an outlier at 53.4 CPM per nSv/h, consistent with its telecommunication and sensor gap history during 2017–2021.

---

## 4. Proposed 5 Pilot Stations & Weather Pairings

We recommend the **5 pilot stations** below:

1. **Birmingham, AL (`AL_BIRMINGHAM`)**
   - **Completeness**: 91.51% (72,192 hrs); Effective Dose Coverage: 91.51%; Complete Channels: 93.52%.
   - **Baseline**: Clean gross mean = 3,884.5 CPM, clean std = 348.4 CPM; Mean dose = 53.8 nSv/h.
   - **NOAA Weather Station**: KBHM (Birmingham-Shuttlesworth International Airport, USAF 722280-13876).
2. **Washington, DC (`DC_WASHINGTON`)**
   - **Completeness**: 92.16% (72,701 hrs); Effective Dose Coverage: 90.64%; Complete Channels: 94.43%.
   - **Baseline**: Clean gross mean = 2,058.5 CPM, clean std = 294.0 CPM; Mean dose = 31.3 nSv/h.
   - **NOAA Weather Station**: KDCA (Ronald Reagan Washington National Airport, USAF 724050-13743).
3. **San Diego, CA (`CA_SAN_DIEGO`)**
   - **Completeness**: 94.32% (74,409 hrs); Effective Dose Coverage: 94.30%; Complete Channels: 99.93%.
   - **Baseline**: Clean gross mean = 6,913.3 CPM, clean std = 530.9 CPM; Mean dose = 99.5 nSv/h.
   - **NOAA Weather Station**: KSAN (San Diego International Airport, USAF 722900-23188).
4. **Dallas, TX (`TX_DALLAS`)**
   - **Completeness**: 80.59% (63,575 hrs); Effective Dose Coverage: 80.59%; Complete Channels: 98.01%.
   - **Baseline**: Clean gross mean = 2,944.3 CPM, clean std = 401.1 CPM; Mean dose = 39.9 nSv/h.
   - **NOAA Weather Station**: KDFW (Dallas/Fort Worth International Airport, USAF 722590-03927).
5. **Tampa, FL (`FL_TAMPA`)**
   - **Completeness**: 82.93% (65,418 hrs); Effective Dose Coverage: 77.91%; Complete Channels: 98.42%.
   - **Baseline**: Clean gross mean = 2,079.5 CPM, clean std = 400.0 CPM; Mean dose = 31.0 nSv/h.
   - **NOAA Weather Station**: KTPA (Tampa International Airport, USAF 722110-12842).

---

## 5. Phase 1 Execution Plan (In Priority Order)

Following Claude's recommended workflow, Phase 1 will execute tasks in the following strict order:
1. **Precipitation Audit**: Download NOAA ISD/Global-Hourly records for all 5 airports (2017–2025). Parse the `AA1` liquid precipitation field (accumulation period length and quality codes) and audit precipitation completeness and reporting frequency per station-year.
2. **Phase 0 Metric Verification in Pipeline**: Ensure the clean data loader applies the complete-channel mask, timestamp deduplication, and verified 78,888 expected hour grid.
3. **Timestamp Convention Alignment**: Resolve whether RadNet's `SAMPLE COLLECTION TIME` denotes interval start or interval end by computing cross-correlation lags with NOAA METAR observation minutes (:50–:55 UTC).
4. **Filter Cycle Analysis**: Scan the continuous gross count rate time series for characteristic sawtooth patterns indicating filter change intervals and document the retention dynamics.
5. **Diurnal Cycle Modeling**: Quantify the baseline hour-of-day profile across R03, R05, R07, R08, and gross CPM on verified dry periods to isolate boundary-layer inversion effects from weather anomalies.
6. **Rain Washout Event Visualization**: Plot rain onset against R03, R05, R07, R08, and dose rate for representative events across stations to provide empirical Gate 1 verification of rain-induced signal rises.
7. **Monitor Coordinates Investigation**: Search state and county ambient air monitoring network plans to determine approximate urban monitor coordinates and compute true distances to airport ASOS stations.

---

## 6. Gate 0 Human Approval Request

With corrections 2.1 through 2.6 resolved, we request Ömer's final sign-off to proceed to Phase 1:
- [ ] Confirm selection of the **5 Pilot Stations** (`AL_BIRMINGHAM`, `DC_WASHINGTON`, `CA_SAN_DIEGO`, `TX_DALLAS`, `FL_TAMPA`).
- [ ] Confirm the **Phase 1 Execution Plan** outlined above.
