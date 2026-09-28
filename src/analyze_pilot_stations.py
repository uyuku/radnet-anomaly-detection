"""
Analyzes data completeness, missingness, and baseline statistics
for candidate pilot stations (2016-2025) and summarizes nearby NOAA stations.
"""

import zipfile
import io
import pandas as pd
import numpy as np
from pathlib import Path


PILOT_CONFIGS = {
    "Birmingham, AL": {
        "zip": "data/raw/al_birmingham_2025-2007.zip",
        "state_code": "AL",
        "city_prefix": "AL_BIRMINGHAM",
        "exposure_start_year": 2016,
        "exposure_start_date": "July 5, 2016",
        "noaa_station": "KBHM",
        "noaa_name": "Birmingham-Shuttlesworth International Airport",
        "noaa_usaf_wban": "722280-13876",
        "noaa_lat": 33.565,
        "noaa_lon": -86.745,
        "climate_zone": "Humid Subtropical (Cfa)",
        "radon_context": "Moderate/High background (Valley & Ridge / Appalachian foothills)",
    },
    "Chicago, IL": {
        "zip": "data/raw/il_chicago_2025-2006.zip",
        "state_code": "IL",
        "city_prefix": "IL_CHICAGO",
        "exposure_start_year": 2016,
        "exposure_start_date": "November 15, 2016",
        "noaa_station": "KORD",
        "noaa_name": "Chicago O'Hare International Airport",
        "noaa_usaf_wban": "725300-94846",
        "noaa_lat": 41.974,
        "noaa_lon": -87.907,
        "climate_zone": "Humid Continental (Dfa)",
        "radon_context": "Midwest glacial till, seasonal snow and convective frontal systems",
    },
    "Austin, TX": {
        "zip": "data/raw/tx_austin_2025-2007.zip",
        "state_code": "TX",
        "city_prefix": "TX_AUSTIN",
        "exposure_start_year": 2016,
        "exposure_start_date": "July 13, 2016",
        "noaa_station": "KAUS",
        "noaa_name": "Austin-Bergstrom International Airport",
        "noaa_usaf_wban": "722540-13904",
        "noaa_lat": 30.194,
        "noaa_lon": -97.670,
        "climate_zone": "Humid Subtropical / Semi-arid transition (Cfa)",
        "radon_context": "Central TX limestone/Balcones Fault, intense convective thunderstorms",
    },
    "Tampa, FL": {
        "zip": "data/raw/fl_tampa_2025-2008.zip",
        "state_code": "FL",
        "city_prefix": "FL_TAMPA",
        "exposure_start_year": 2016,
        "exposure_start_date": "August 15, 2016",
        "noaa_station": "KTPA",
        "noaa_name": "Tampa International Airport",
        "noaa_usaf_wban": "722110-12842",
        "noaa_lat": 27.975,
        "noaa_lon": -82.533,
        "climate_zone": "Humid Subtropical (Cfa)",
        "radon_context": "Coastal plain / phosphate geology, frequent afternoon convective rain",
    },
    "San Diego, CA": {
        "zip": "data/raw/ca_san_diego_2025-2006.zip",
        "state_code": "CA",
        "city_prefix": "CA_SAN_DIEGO",
        "exposure_start_year": 2016,
        "exposure_start_date": "September 22, 2016",
        "noaa_station": "KSAN",
        "noaa_name": "San Diego International Airport",
        "noaa_usaf_wban": "722900-23188",
        "noaa_lat": 32.733,
        "noaa_lon": -117.183,
        "climate_zone": "Semi-arid / Mediterranean (CSa/BSk)",
        "radon_context": "Low rainfall / arid control; baseline false-alarm verification",
    },
}


def load_station_history(cfg: dict, start_year: int = 2017, end_year: int = 2025) -> pd.DataFrame:
    dfs = []
    zip_path = Path(cfg["zip"])
    with zipfile.ZipFile(zip_path, "r") as z:
        namelist = z.namelist()
        for year in range(start_year, end_year + 1):
            expected_suffix = f"{cfg['city_prefix']}_{year}.csv".lower()
            matched = [n for n in namelist if n.lower().endswith(expected_suffix)]
            if matched:
                with z.open(matched[0]) as f:
                    df_year = pd.read_csv(f)
                    dfs.append(df_year)
            else:
                print(f"Warning: {expected_suffix} not found in {zip_path.name}")

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    full_df["timestamp"] = pd.to_datetime(full_df["SAMPLE COLLECTION TIME"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    return full_df


def analyze_station(station_name: str, cfg: dict):
    # Analyze complete 2017-2025 period where exposure rate was active for the full calendar years
    df = load_station_history(cfg, start_year=2017, end_year=2025)
    
    total_hours_expected = (pd.Timestamp("2025-12-31 23:59:59") - pd.Timestamp("2017-01-01 00:00:00")).total_seconds() / 3600
    total_records = len(df)
    
    # Missing timestamps
    # Channels
    channel_cols = [c for c in df.columns if "GAMMA COUNT RATE" in c]
    dose_col = "DOSE EQUIVALENT RATE (nSv/h)"
    
    dose_missing = df[dose_col].isna().sum() if dose_col in df.columns else total_records
    dose_missing_pct = (dose_missing / total_records) * 100 if total_records > 0 else 100.0
    
    r03_col = [c for c in channel_cols if "R03" in c][0]
    r05_col = [c for c in channel_cols if "R05" in c][0]
    
    gross_gamma_series = df[channel_cols].sum(axis=1)
    
    return {
        "station": station_name,
        "years": "2017-2025 (9 full years)",
        "total_expected_hours": int(total_hours_expected),
        "total_records": total_records,
        "completeness_pct": round((total_records / total_hours_expected) * 100, 2),
        "dose_rate_present_pct": round(100.0 - dose_missing_pct, 2),
        "mean_dose_rate_nSv_h": round(df[dose_col].dropna().mean(), 1) if dose_col in df.columns else np.nan,
        "median_dose_rate_nSv_h": round(df[dose_col].dropna().median(), 1) if dose_col in df.columns else np.nan,
        "mean_gross_cpm": round(gross_gamma_series.mean(), 1),
        "median_gross_cpm": round(gross_gamma_series.median(), 1),
        "std_gross_cpm": round(gross_gamma_series.std(), 1),
        "r03_mean_cpm": round(df[r03_col].dropna().mean(), 1),
        "r05_mean_cpm": round(df[r05_col].dropna().mean(), 1),
        "noaa_station": cfg["noaa_station"],
        "noaa_name": cfg["noaa_name"],
        "climate_zone": cfg["climate_zone"],
        "radon_context": cfg["radon_context"],
    }


if __name__ == "__main__":
    results = []
    for name, cfg in PILOT_CONFIGS.items():
        res = analyze_station(name, cfg)
        results.append(res)

    res_df = pd.DataFrame(results)
    out_path = Path("data/processed/pilot_station_metrics.csv")
    res_df.to_csv(out_path, index=False)
    print("=== Candidate Pilot Station Metrics (2017-2025) ===")
    print(res_df[["station", "completeness_pct", "dose_rate_present_pct", "mean_dose_rate_nSv_h", "mean_gross_cpm", "std_gross_cpm", "noaa_station"]].to_string(index=False))
