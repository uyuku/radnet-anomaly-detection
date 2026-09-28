# Phase 0 Report: Data Discovery & Pilot Station Selection

**Date**: 2026-09-28  
**Project**: Weather-Aware Anomaly Detection in RadNet  
**Phase**: Phase 0 (Data Discovery)  
**Status**: Completed — **Gate 0 reached (Awaiting Human Approval)**

---

## 1. Executive Summary

Phase 0 established the empirical foundation for the project:
1. **Station Network Scraped & Categorized**: All 137 RadNet stationary air monitors across the United States were indexed. 127 stations currently report exposure rate (`DOSE EQUIVALENT RATE (nSv/h)`). Exactly 20 stations possess a 10-year history with exposure rate operational since mid-2016.
2. **Channel Energy Boundaries Discovered & Cited**: The physical energy boundaries (in keV) for channels R02 through R09 were identified from peer-reviewed literature and mapped to key radon progeny (Pb-214, Bi-214) and fission products (Cs-137, I-131).
3. **Data Completeness & Missingness Quantified**: Evaluated 8 candidate stations across 9 full calendar years (2017–2025; 78,888 expected hourly records per station).
4. **5 Diverse Pilot Stations Selected & Paired with NOAA ASOS**: Identified 5 premier pilot stations representing distinct climatic, geographic, and radon background regimes, all paired with first-order NOAA airport weather stations (USAF-WBAN) reporting hourly precipitation, pressure, temperature, and dew point in UTC.
5. **Reproducibility & Traceability Enforced**: Raw data storage policy established (`data/raw/` read-only, `chmod 444`), all downloads hashed and logged in `DATA_LOG.md`, design decisions and literature citations recorded in `DECISIONS.md`, unknown questions cataloged in `QUESTIONS.md`, and fixed random seed (42) stored in `configs/config.yaml`.

---

## 2. Gamma Channel Energy Boundaries

From peer-reviewed research analyzing EPA RadNet airborne gamma spectrometry:
> **Citation**: Vieira, C. L. Z., Koutrakis, P., Huang, S., Grady, S., Hart, J. E., Coull, B. A., Laden, F., Requia, W., Schwartz, J., & Garshick, E. (2019). Short-term effects of particle gamma radiation activities on pulmonary function in COPD patients. *Environmental Research*, 175, 221–227. https://doi.org/10.1016/j.envres.2019.05.021.

| Channel | Energy Range (keV) | Physical Role & Radionuclide Overlap |
| :--- | :--- | :--- |
| **γ1 / R01** | $\le 100$ | Lower discriminator / detector noise floor (omitted in public CSVs) |
| **γ2 / R02** | $101 - 200$ | Low-energy Compton scatter & backscatter; highest baseline counts (~50%) |
| **γ3 / R03** | $201 - 400$ | **Critical overlap zone**: Contains **Pb-214** (351.9 keV) and **I-131** (364.5 keV) |
| **γ4 / R04** | $401 - 600$ | Intermediate Compton continuum |
| **γ5 / R05** | $601 - 800$ | **Critical overlap zone**: Contains **Bi-214** (609.3 keV) and **Cs-137** (661.7 keV) |
| **γ6 / R06** | $801 - 1000$ | Upper Compton continuum |
| **γ7 / R07** | $1001 - 1400$ | High-energy radon progeny: Contains **Bi-214** photopeak (1120.3 keV) |
| **γ8 / R08** | $1401 - 1800$ | High-energy radon progeny: Contains **Bi-214** photopeak (1764.5 keV) |
| **γ9 / R09** | $1801 - 2200$ | Highest energy window (cosmic background / very high energy prompt lines) |

### Physical Consequence for Project Spec
As noted in Section 5 of `PROJECT_SPEC.md`, NaI(Tl) scintillation detectors have typical energy resolutions of 7–9% FWHM at 662 keV. Consequently:
- **Pb-214 (352 keV)** cannot be separated spectrally from **I-131 (364 keV)** because both fall into **Channel R03 (201–400 keV)**.
- **Bi-214 (609 keV)** cannot be separated spectrally from **Cs-137 (662 keV)** because both fall into **Channel R05 (601–800 keV)**.
- *Crucial differentiator*: **Temporal dynamics and weather fusion**. Radon progeny decay rapidly ($T_{1/2} = 26.8\text{ min}$ for Pb-214, $19.9\text{ min}$ for Bi-214) following precipitation cessation, whereas fission products persist indefinitely on the air filter. Channels R07 and R08 provide pure Bi-214 verification (high-energy lines without Cs-137/I-131 counterparts).

---

## 3. Candidate Pilot Stations Analysis (2017–2025)

We analyzed 8 candidate stations with exposure rate histories dating back to 2016 over 9 full calendar years (2017–2025; 78,888 possible hourly intervals per station):

| Station | Region / Climate | Completeness (2017–2025) | Dose Rate Present | Mean Dose (nSv/h) | Mean Gross (CPM) | Std Gross (CPM) | NOAA Airport ASOS | Geologic / Radon Context |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **San Diego, CA** | West Coast (Mediterranean/Arid) | **94.3%** (74,415 hrs) | **100.0%** | 99.5 | 6,908.3 | 561.8 | KSAN | Coastal marine layer; arid control baseline |
| **Washington, DC** | Mid-Atlantic (Temperate) | **92.2%** (72,706 hrs) | **98.4%** | 31.3 | 1,943.8 | 551.9 | KDCA | Fall line; temperate 4-season precipitation |
| **Birmingham, AL** | Southeast (Humid Subtropical) | **91.5%** (72,197 hrs) | **100.0%** | 53.8 | 3,633.0 | 1,013.6 | KBHM | Appalachian foothills; high rain washout surge |
| **Montgomery, AL** | Southeast (Gulf Coastal Plain) | **87.6%** (69,093 hrs) | **99.9%** | 49.6 | 3,705.7 | 721.9 | KMGM | Coastal plain; frequent heavy rainfall |
| **Tampa, FL** | Gulf Coast / FL (Subtropical) | **82.9%** (65,424 hrs) | **93.9%** | 31.0 | 2,046.5 | 474.2 | KTPA | Low altitude coastal plain; intense summer thunderstorms |
| **Dallas, TX** | Southern Plains (Subtropical) | **80.6%** (63,581 hrs) | **100.0%** | 39.9 | 2,885.8 | 571.5 | KDFW | Blackland prairie; severe frontal thunderstorms |
| **Austin, TX** | Central Texas (Subtropical/Semi-arid) | **74.4%** (58,718 hrs) | **96.1%** | 44.7 | 2,684.8 | 944.8 | KAUS | Balcones fault; convective storm episodes |
| **Chicago, IL** | Midwest (Humid Continental) | **74.2%** (58,508 hrs) | **91.8%** | 50.4 | 2,666.0 | 803.6 | KORD | Lake Michigan lake-effect; winter freeze/snow |

---

## 4. Proposed 5 Pilot Stations & Weather Pairings

We recommend the top **5 pilot stations** below:

1. **Birmingham, AL (`AL_BIRMINGHAM`)**
   - **Why**: Premier high-variability rain washout station. 91.5% completeness across 9 years with 100% dose rate coverage. High background gross count rate variability ($\sigma = 1,013.6\text{ CPM}$) driven by intense convective rain in the Appalachian foothills.
   - **NOAA Weather Station**: KBHM (Birmingham-Shuttlesworth International Airport, USAF 722280-13876).
2. **Washington, DC (`DC_WASHINGTON`)**
   - **Why**: Mid-Atlantic temperate baseline. 92.2% completeness, 98.4% dose rate coverage. Excellent representation of 4-season temperate weather patterns (frontal rain, autumn showers, winter precipitation).
   - **NOAA Weather Station**: KDCA (Ronald Reagan Washington National Airport, USAF 724050-13743).
3. **San Diego, CA (`CA_SAN_DIEGO`)**
   - **Why**: Arid / low-rainfall control station. 94.3% completeness (highest in network) with 100% dose rate presence. Stable high background ($\mu = 6,908\text{ CPM}, \sigma = 561.8\text{ CPM}, \text{CV} = 0.08$). Crucial for verifying baseline false alarm rate during prolonged dry conditions.
   - **NOAA Weather Station**: KSAN (San Diego International Airport, USAF 722900-23188).
4. **Dallas, TX (`TX_DALLAS`)**
   - **Why**: Southern Great Plains storm dynamics. 80.6% completeness, 100% dose rate presence. Subject to severe mesoscale convective systems and strong precipitation spikes.
   - **NOAA Weather Station**: KDFW (Dallas/Fort Worth International Airport, USAF 722590-03927).
5. **Tampa, FL (`FL_TAMPA`)**
   - **Why**: Subtropical maritime / coastal plain. 82.9% completeness, 93.9% dose rate presence. Low terrestrial background ($\mu = 2,046.5\text{ CPM}$), high humidity, and frequent rapid afternoon convective thunderstorms.
   - **NOAA Weather Station**: KTPA (Tampa International Airport, USAF 722110-12842).

*Alternative options available if preferred*:
- **Montgomery, AL** (87.6% completeness, paired with KMGM) can replace Birmingham or Tampa.
- **Chicago, IL** (74.2% completeness, paired with KORD) can be included if winter freeze/lake-effect dynamics are desired despite lower historical completeness in 2017–2020.

---

## 5. Artifacts and Commits

- **Code**:
  - `src/parse_stations.py`: Full RadNet web scraper and inventory generator.
  - `src/download_pilot_data.py`: Secure download utility enforcing `chmod 444` and logging.
  - `src/analyze_pilot_stations.py`: Completeness, missingness, and radiation statistics calculator.
- **Data**:
  - `data/processed/radnet_station_inventory.csv`: Complete metadata for all 137 RadNet stations.
  - `data/processed/candidate_pilot_stations_comparison.csv`: Comparative metrics for candidate stations.
- **Documentation**:
  - `DATA_LOG.md`: Fully updated download log with URLs, dates, and SHA256 hashes.
  - `DECISIONS.md`: Logged design decisions, channel boundaries, and literature citations.
  - `QUESTIONS.md`: Documented resolved parameters and Phase 1 research questions.
  - `configs/config.yaml`: Frozen config with random seed (42), pilot station configurations, and channel energy specs.

---

## 6. Gate 0 Checklist for Human (Ömer)

To pass **Gate 0** and proceed to **Phase 1 (Merge and Exploration)**, human approval is requested on:

- [ ] **Approval of the 5 pilot stations**:
  1. Birmingham, AL (`KBHM`)
  2. Washington, DC (`KDCA`)
  3. San Diego, CA (`KSAN`)
  4. Dallas, TX (`KDFW`)
  5. Tampa, FL (`KTPA`)
  *(Or any substitution from the candidate table above, such as Montgomery or Chicago)*.
- [ ] **Approval of the channel keV boundary mapping** (Vieira et al., 2019).
- [ ] **Approval of NOAA NCEI Global-Hourly ASOS as the weather data source**.
