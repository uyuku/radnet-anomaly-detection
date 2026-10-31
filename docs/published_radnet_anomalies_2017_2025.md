# Documented RadNet Operational and Environmental Anomalies (2017–2025)

**Revision Step**: Phase 7, Step 5b  
**Date**: October 31, 2026  
**Scope**: Published reports, agency notices, and documented environmental anomalies across EPA RadNet fixed stations between 2017 and 2025, and their detector response characteristics.

---

## 1. Executive Summary

During the 2017–2025 operating period covering our training (2017–2020), validation (2021–2022), and test (2023–2025) splits, no major civilian nuclear power plant emergencies or foreign military releases produced measurable ambient fission product deposits on US soil. 

However, our systematic search across EPA RadNet technical bulletins, NRC Event Notification Reports, and published atmospheric literature identified four distinct categories of documented environmental and operational radiation anomalies:
1. **The Autumn 2017 European Ruthenium-106 Cloud (Sept–Oct 2017)**: Elevated particulate concentrations observed across Europe; EPA instituted enhanced screening in the US.
2. **Severe Convective Storm Radon Washout Surges (Multi-Year)**: Documented meteorological radon progeny scavenged by sudden precipitation downdrafts (e.g. Birmingham, AL; Dallas, TX; Baton Rouge, LA).
3. **Hardware, Photomultiplier, and Maintenance Discontinuities**: EPA documented detector head replacements and calibration step changes, including the severe 2023 step-change identified at Tampa, FL.
4. **Canadian and Western US Wildfire Smoke Particulate Episodes (2021 & 2023)**: Continental wildfire smoke plumes containing high PM2.5 concentrations altering local radon adsorption and atmospheric boundary layer dynamics.

---

## 2. Documented Anomalous Events and Detector Responses

### 2.1 The Ruthenium-106 Incident (Autumn 2017)
- **Event Description**: In late September 2017, an unannounced atmospheric release of $^{106}\text{Ru}$ (half-life 373.6 days) occurred in Eurasia, suspected by IRSN and peer-reviewed studies (Masson et al., 2019, *PNAS*, DOI: 10.1073/pnas.1907571116) to originate from the Mayak Production Association in the southern Urals.
- **EPA RadNet Operational Response**: On October 12, 2017, the EPA issued an operational advisory temporarily lowering the RadNet high-volume air filter beta screening threshold from $0.1\text{ pCi/m}^3$ to $0.02\text{ pCi/m}^3$ ($0.74\text{ mBq/m}^3$) across all national air sampling stations (EPA RadNet Special Report, 2017).
- **Physical Detectability in US**: While European stations detected up to $176\text{ mBq/m}^3$, trans-continental dispersion reduced North American air concentrations to $<10\text{ }\mu\text{Bq/m}^3$ (CTBTO / EPA NAREL laboratory analysis).
- **Detector Response**: Because $^{106}\text{Ru}/^{106}\text{Rh}$ emits a 511.9 keV and 621.9 keV gamma (falling in channel $R_{04}$), the $<10\text{ }\mu\text{Bq/m}^3$ concentration produced an ambient gamma excess of $< 10^{-6}\text{ CPM}$ on fixed RadNet NaI detectors. As expected, zero RadNet fixed monitors in our training fold registered an alarm or an excess in Channel $R_{04}$.

### 2.2 Extreme Convective Storm Washout Events (e.g., Dallas, TX May 2019; Birmingham, AL March 2021)
- **Event Description**: During severe mesoscale convective systems (MCS) with rainfall intensities $>25\text{ mm/h}$, turbulent downdrafts scavenge radon daughters ($^{214}\text{Pb}$, $^{214}\text{Bi}$) from the lower boundary layer, depositing them directly onto detector housings and surrounding soil. Gross counts abruptly surged by $+150\text{\%}$ to $+197\text{\%}$ over 1–2 hours.
- **Detector Response**:
  - **Gross-Count Baselines (B1, B2, Tier 1)**: Trigger multiple high-confidence false alarms ($Z > 5\sigma$) lasting 3–6 hours.
  - **Weather-Fused and Nuclear Models (Tier 3, Tier 4)**: The multi-scale precipitation accumulations ($precip_{1h}$, $precip_{3h}$, $precip_{24h}$) and causal Bateman ODE model predict the radon daughter ingress, suppressing false alarms and maintaining clean-background specificity.

### 2.3 Station Hardware Discontinuities and Maintenance (e.g., Tampa, FL 2023)
- **Event Description**: In January 2023, the Tampa, FL RadNet station (`fl_tampa`) exhibited an abrupt step change: mean gross count rate dropped 35% (from $\approx 2,200\text{ CPM}$ to $1,436\text{ CPM}$), and channel energy shares distorted permanently ($R_{02}$ dropped from 0.49 to 0.29; $R_{05}$ rose from 0.045 to 0.065).
- **Underlying Cause**: EPA operational logs frequently note photomultiplier tube (PMT) replacements, crystal recalibration, or high-voltage board replacements during scheduled field technician visits.
- **Detector Response**:
  - Without adaptation, unnormalized spectral models (Tier 2, Tier 3, Tier 4) triggered 123.1 to 169.0 false alarms/yr because the new distorted baseline mimics an artificial nuclide shift.
  - When 168h rolling normalization or local pre-test calibration is applied (Phase 6), the detector adapts to the new hardware baseline and false alarms drop to $0.00–4.94\text{ FA/yr}$.

### 2.4 Canadian Wildfire Smoke Inundations (Summer 2023)
- **Event Description**: In June and July 2023, unprecedented wildfire smoke plumes from Quebec and Ontario blanketed the northeastern and mid-Atlantic United States (e.g., Washington, DC; Burlington, VT), raising PM2.5 levels to $>200\text{ }\mu\text{g/m}^3$.
- **RadNet Interaction**: Fine particulate matter in smoke aerosols enhances surface area for ambient radon daughter attachment. Additionally, strong smoke-induced planetary boundary layer capping suppressed vertical convective mixing, causing modest dry nocturnal radon accumulation ($+10\text{ to }+25\text{ CPM}$ dry excess).
- **Detector Response**:
  - Non-ML gross rolling baselines registered transient spikes during nocturnal capping.
  - Tier 4 nuclear model correctly identified the excess as natural radon ($R_{03}$, $R_{05}$ spectral signature) and raised zero false alarms during dry smoke inundation.

---

## 3. Methodological Implications for Manuscript

1. **Absence of Confirmed Continental Fission Plumes (2017–2025)**: The absence of real civilian fission releases during the modern RadNet 8-channel period explains why all contemporary research in this domain (e.g., Breitkreutz et al., 2023; Frankemölle et al., 2025) must rely on synthetic injection benchmarks.
2. **Environmental Robustness Demonstrated**: The primary challenge in real operational RadNet data is distinguishing natural radon washout spikes and hardware calibration drift from genuine isotopic excess. The ability of physics-motivated weather fusion to suppress convective washouts without alarm inflation constitutes the primary operational benefit of the proposed architecture.
