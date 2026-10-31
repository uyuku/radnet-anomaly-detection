# Pre-Registered Extended RadNet Station Selection Protocol (Phase 6 Part E)

**Pre-Registration Date:** 2026-10-31 (written prior to downloading or inspecting any extended-station RadNet or NOAA archives)

---

## 1. Objective

The original 5-station benchmark evaluated `al_birmingham` (Humid Subtropical), `dc_washington` (Mid-Atlantic Temperate), `ca_san_diego` (Coastal Mediterranean), `tx_dallas` (Southern Plains Convective), and `fl_tampa` (Gulf Subtropical). To test spatial and climatic generalization across the broader U.S. EPA RadNet network without post-hoc station cherry-picking, we pre-register an objective station-selection rule applied solely to the metadata columns of `data/processed/radnet_station_inventory.csv` (`gamma_start`, `exposure_start`, `has_exposure_rate`, `state`, `city`) before downloading or inspecting any new station data.

---

## 2. Pre-Registered Eligibility and Inclusion Rule

1. **Metadata Eligibility Filter (from `data/processed/radnet_station_inventory.csv` only)**:
   - `has_exposure_rate == True`
   - `gamma_start` prior to `2017-01-01` (ensuring 8-channel `R02..R09` gamma spectrometer operation across the entire 2017–2025 study period)
   - `exposure_start` on or before `2017-06-30` (ensuring continuous `dose_rate_nsvh` availability across the 2017–2020 training fold, 2021–2022 validation fold, and 2023–2025 test fold)
   - Valid `historical_zip_url` ending in `2025-*.zip`

2. **Target Climate Regimes (5 New Regimes Not Covered by Pilot Set)**:
   From the metadata-eligible stations, we select one primary candidate (and one pre-specified backup in case of download failure or `< 65%` valid complete-channel hourly coverage over 2017–2025) for each of the five required climate regimes:

| Target Climate Regime | Köppen Class | Selected Station (`station_id`) | Inventory `gamma_start` | Inventory `exposure_start` | Colocated NOAA ISD Airport (`call` / USAF-WBAN) | Pre-Specified Backup (if download/coverage fails) |
| :--- | :---: | :--- | :---: | :---: | :--- | :--- |
| **1. Arid Interior** | `BWh` (Hot Desert) | **Phoenix, AZ** (`az_phoenix`) | 2007-03-23 | 2016-08-01 | `KPHX` (`72278023183`, Phoenix Sky Harbor Int'l) | **Richland, WA** (`wa_richland`, `BSk`, `KRLD` / `KMWH`) |
| **2. Cold Continental** | `Dfa` / `Dfb` (Cold Continental) | **Omaha, NE** (`ne_omaha`) | 2008-05-20 | 2016-07-01 | `KOMA` (`72550014942`, Omaha Eppley Airfield) | **Burlington, VT** (`vt_burlington`, `Dfb`, `KBTV` `72617014742`) |
| **3. Tropical** | `As` / `Af` (Tropical Trade-Wind) | **Honolulu, HI** (`hi_honolulu`) | 2008-04-22 | 2017-06-06 | `PHNL` (`91182022521`, Honolulu Daniel K. Inouye Int'l) | **San Juan, PR** (`pr_san_juan`, `Af`, `TJSJ` `78526011641`) |
| **4. High-Latitude** | `Dfc` (Subarctic High-Latitude) | **Anchorage, AK** (`ak_anchorage`) | 2008-06-05 | 2016-09-16 | `PANC` (`70273026451`, Ted Stevens Anchorage Int'l) | **Fairbanks, AK** (`ak_fairbanks`, `Dfc`, `PAFA` `70261026411`) |
| **5. High-Rainfall Coastal** | `Cfa` (Gulf Coastal Floodplain, ~1,600 mm/yr) | **Baton Rouge, LA** (`la_baton_rouge`) | 2008-05-03 | 2016-07-01 | `KBTR` (`72232013970`, Baton Rouge Metro Airport) | **Houston, TX** (`tx_houston`, `Cfa`, `KIAH` `72243012960`) |
| **6. Northern Cold Continental (Bonus 6th Station)** | `Dfb` (Cold Humid Continental) | **Burlington, VT** (`vt_burlington`) | 2009-12-16 | 2016-08-01 | `KBTV` (`72617014742`, Burlington Int'l Airport) | **Chicago, IL** (`il_chicago`, `Dfa`, `KORD` `72530094846`) |

---

## 3. Data Quality & Non-Substitution Rules

1. **Minimum Data Completeness Rule**:
   - Every downloaded station is audited over the 2017–2025 hourly grid (`78,888` hours).
   - A station is included in the extended LOSO and Leave-One-Climate-Out (LOCO) evaluation if it has:
     - At least `50,000` valid complete-channel (`R02..R09`) hours (`>= 63.4%` completeness over 2017–2025), AND
     - Non-empty valid hours in each of the three splits (`2017–2020` fit, `2021–2022` validation, `2023–2025` test).
   - If a primary station fails to download from the EPA/NOAA servers or fails the completeness gate above, that failure and its exact missingness statistics are recorded in `data/processed/extended_station_coverage_audit.csv`, and its pre-specified backup is evaluated.
2. **Strict No-Post-Hoc-Deletion Rule**:
   - **No station may ever be dropped, replaced, or excluded after its model detection or false-alarm results are computed.**
   - Every station that passes the pre-model data-completeness check is included in all extended-station LOSO and LOCO tables with its exact exposure (`clean_station_years`), Clopper-Pearson binomial detection interval, and Garwood exact Poisson false-alarm interval.
