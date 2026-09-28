"""
Analyzes data completeness, missingness, and baseline statistics
for all 8 candidate pilot stations (2017-2025).
Enforces complete-channel filtering, checks for duplicate timestamps,
computes effective dose coverage, and outputs per-channel missingness.
"""

import zipfile
import pandas as pd
import numpy as np
from pathlib import Path


CANDIDATE_STATIONS = {
    "Birmingham, AL": {
        "zip": "data/raw/al_birmingham_2025-2007.zip",
        "city_prefix": "AL_BIRMINGHAM",
        "noaa_station": "KBHM",
        "noaa_name": "Birmingham-Shuttlesworth International Airport",
        "noaa_usaf_wban": "722280-13876",
        "region": "Southeast",
        "climate": "Humid Subtropical (Cfa)",
        "epa_radon_zone": "Zone 2 (Moderate, 2-4 pCi/L; Jefferson County, AL)",
    },
    "Washington, DC": {
        "zip": "data/raw/dc_washington_2025-2006.zip",
        "city_prefix": "DC_WASHINGTON",
        "noaa_station": "KDCA",
        "noaa_name": "Ronald Reagan Washington National Airport",
        "noaa_usaf_wban": "724050-13743",
        "region": "Mid-Atlantic",
        "climate": "Temperate Maritime/Continental (Cfa)",
        "epa_radon_zone": "Zone 3 (Low, <2 pCi/L; District of Columbia)",
    },
    "San Diego, CA": {
        "zip": "data/raw/ca_san_diego_2025-2006.zip",
        "city_prefix": "CA_SAN_DIEGO",
        "noaa_station": "KSAN",
        "noaa_name": "San Diego International Airport",
        "noaa_usaf_wban": "722900-23188",
        "region": "West Coast",
        "climate": "Mediterranean / Semi-arid (CSa/BSk)",
        "epa_radon_zone": "Zone 3 (Low, <2 pCi/L; San Diego County, CA)",
    },
    "Dallas, TX": {
        "zip": "data/raw/tx_dallas_2025-2007.zip",
        "city_prefix": "TX_DALLAS",
        "noaa_station": "KDFW",
        "noaa_name": "Dallas/Fort Worth International Airport",
        "noaa_usaf_wban": "722590-03927",
        "region": "Southern Plains",
        "climate": "Subtropical Continental (Cfa)",
        "epa_radon_zone": "Zone 3 (Low, <2 pCi/L; Dallas County, TX)",
    },
    "Tampa, FL": {
        "zip": "data/raw/fl_tampa_2025-2008.zip",
        "city_prefix": "FL_TAMPA",
        "noaa_station": "KTPA",
        "noaa_name": "Tampa International Airport",
        "noaa_usaf_wban": "722110-12842",
        "region": "Gulf Coast / FL",
        "climate": "Humid Subtropical (Cfa)",
        "epa_radon_zone": "Zone 3 (Low, <2 pCi/L; Hillsborough County, FL)",
    },
    "Montgomery, AL": {
        "zip": "data/raw/al_montgomery_2025-2006.zip",
        "city_prefix": "AL_MONTGOMERY",
        "noaa_station": "KMGM",
        "noaa_name": "Montgomery Regional Airport",
        "noaa_usaf_wban": "722260-13895",
        "region": "Southeast",
        "climate": "Humid Subtropical (Cfa)",
        "epa_radon_zone": "Zone 3 (Low, <2 pCi/L; Montgomery County, AL)",
    },
    "Chicago, IL": {
        "zip": "data/raw/il_chicago_2025-2006.zip",
        "city_prefix": "IL_CHICAGO",
        "noaa_station": "KORD",
        "noaa_name": "Chicago O'Hare International Airport",
        "noaa_usaf_wban": "725300-94846",
        "region": "Midwest",
        "climate": "Humid Continental (Dfa)",
        "epa_radon_zone": "Zone 2 (Moderate, 2-4 pCi/L; Cook County, IL)",
    },
    "Austin, TX": {
        "zip": "data/raw/tx_austin_2025-2007.zip",
        "city_prefix": "TX_AUSTIN",
        "noaa_station": "KAUS",
        "noaa_name": "Austin-Bergstrom International Airport",
        "noaa_usaf_wban": "722540-13904",
        "region": "Central Texas",
        "climate": "Subtropical / Semi-arid transition (Cfa)",
        "epa_radon_zone": "Zone 3 (Low, <2 pCi/L; Travis County, TX)",
    },
}

CHANNEL_KEYS = ["R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09"]
TOTAL_EXPECTED_HOURS = 78888  # 7 non-leap years * 8760 + 2 leap years * 8784 = 78,888


def load_raw_station_data(cfg: dict, start_year: int = 2017, end_year: int = 2025) -> pd.DataFrame:
    dfs = []
    zip_path = Path(cfg["zip"])
    with zipfile.ZipFile(zip_path, "r") as z:
        namelist = z.namelist()
        for year in range(start_year, end_year + 1):
            expected_suffix = f"{cfg['city_prefix']}_{year}.csv".lower()
            matched = [n for n in namelist if n.lower().endswith(expected_suffix)]
            if matched:
                with z.open(matched[0]) as f:
                    dfs.append(pd.read_csv(f))
            else:
                print(f"Warning: {expected_suffix} not found in {zip_path.name}")

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


def analyze_station(station_name: str, cfg: dict):
    df = load_raw_station_data(cfg, start_year=2017, end_year=2025)
    total_raw_rows = len(df)

    # Duplicate check on timestamp
    time_col = "SAMPLE COLLECTION TIME"
    num_duplicates = df[time_col].duplicated().sum()
    # Deduplicate keeping first occurrence
    df_dedup = df.drop_duplicates(subset=[time_col]).copy()
    unique_records = len(df_dedup)

    completeness_pct = round((unique_records / TOTAL_EXPECTED_HOURS) * 100, 2)

    # Dose rate analysis
    dose_col = "DOSE EQUIVALENT RATE (nSv/h)"
    has_dose = dose_col in df_dedup.columns
    dose_present_count = df_dedup[dose_col].notna().sum() if has_dose else 0
    dose_present_pct = round((dose_present_count / unique_records) * 100, 2) if unique_records > 0 else 0.0
    effective_dose_coverage_pct = round((dose_present_count / TOTAL_EXPECTED_HOURS) * 100, 2)

    # Channel columns
    chan_cols_map = {k: [c for c in df_dedup.columns if k in c][0] for k in CHANNEL_KEYS}
    chan_cols = [chan_cols_map[k] for k in CHANNEL_KEYS]

    # Per-channel missingness on deduplicated records
    missing_by_channel = {}
    for k in CHANNEL_KEYS:
        col = chan_cols_map[k]
        missing_count = df_dedup[col].isna().sum()
        missing_by_channel[k] = missing_count

    # Complete-channel filtering: ALL 8 channels must be non-null
    complete_channel_mask = df_dedup[chan_cols].notna().all(axis=1)
    complete_channel_records = complete_channel_mask.sum()
    complete_channel_pct = round((complete_channel_records / unique_records) * 100, 2) if unique_records > 0 else 0.0

    # Calculate statistics ONLY on complete-channel rows
    df_complete = df_dedup.loc[complete_channel_mask]
    clean_gross_series = df_complete[chan_cols].sum(axis=1)

    mean_gross = round(clean_gross_series.mean(), 1)
    median_gross = round(clean_gross_series.median(), 1)
    std_gross = round(clean_gross_series.std(), 1)

    mean_dose = round(df_dedup[dose_col].dropna().mean(), 1) if has_dose else np.nan
    median_dose = round(df_dedup[dose_col].dropna().median(), 1) if has_dose else np.nan

    # Channel proportions on complete data
    r02_pct = round((df_complete[chan_cols_map["R02"]] / clean_gross_series).mean() * 100, 1)
    r03_pct = round((df_complete[chan_cols_map["R03"]] / clean_gross_series).mean() * 100, 1)
    r05_pct = round((df_complete[chan_cols_map["R05"]] / clean_gross_series).mean() * 100, 1)

    metrics_row = {
        "Station": station_name,
        "Region / Climate": f"{cfg['region']} ({cfg['climate']})",
        "Total Expected Hours": TOTAL_EXPECTED_HOURS,
        "Unique Records": unique_records,
        "Duplicate Timestamps": num_duplicates,
        "Completeness (%)": completeness_pct,
        "Dose Rate Present (%)": dose_present_pct,
        "Effective Dose Coverage (%)": effective_dose_coverage_pct,
        "Complete 8-Channel Records": complete_channel_records,
        "Complete Channel (%)": complete_channel_pct,
        "Clean Gross Mean (CPM)": mean_gross,
        "Clean Gross Median (CPM)": median_gross,
        "Clean Gross Std (CPM)": std_gross,
        "Mean Dose (nSv/h)": mean_dose,
        "Median Dose (nSv/h)": median_dose,
        "R02 Fraction (%)": r02_pct,
        "R03 Fraction (%)": r03_pct,
        "R05 Fraction (%)": r05_pct,
        "NOAA ASOS Station": cfg["noaa_station"],
        "EPA Radon Zone": cfg["epa_radon_zone"],
    }

    missingness_row = {
        "Station": station_name,
        "Unique Records": unique_records,
        "Dose Rate Missing": unique_records - dose_present_count,
        "Dose Rate Missing (%)": round((100.0 - dose_present_pct), 2),
    }
    for k in CHANNEL_KEYS:
        m = missing_by_channel[k]
        missingness_row[f"{k} Missing"] = m
        missingness_row[f"{k} Missing (%)"] = round((m / unique_records) * 100, 2) if unique_records else 0.0

    return metrics_row, missingness_row


def run_full_analysis():
    metrics_list = []
    missing_list = []

    for name, cfg in CANDIDATE_STATIONS.items():
        m_row, miss_row = analyze_station(name, cfg)
        metrics_list.append(m_row)
        missing_list.append(miss_row)

    df_metrics = pd.DataFrame(metrics_list)
    df_missing = pd.DataFrame(missing_list)

    # Save to data/processed
    out_comparison = Path("data/processed/candidate_pilot_stations_comparison.csv")
    out_missing = Path("data/processed/pilot_station_channel_missingness.csv")
    out_pilot_metrics = Path("data/processed/pilot_station_metrics.csv")

    df_metrics.to_csv(out_comparison, index=False)
    df_metrics.to_csv(out_pilot_metrics, index=False)
    df_missing.to_csv(out_missing, index=False)

    print("=== Candidate Pilot Stations Comparison (2017-2025, 78,888 Expected Hours) ===")
    cols_display = [
        "Station",
        "Completeness (%)",
        "Dose Rate Present (%)",
        "Effective Dose Coverage (%)",
        "Complete Channel (%)",
        "Clean Gross Mean (CPM)",
        "Clean Gross Std (CPM)",
        "Mean Dose (nSv/h)",
        "NOAA ASOS Station",
    ]
    print(df_metrics[cols_display].to_string(index=False))

    print("\n=== Per-Channel Missingness (Count & %) ===")
    cols_miss_display = [
        "Station",
        "Unique Records",
        "Dose Rate Missing (%)",
        "R02 Missing (%)",
        "R03 Missing (%)",
        "R05 Missing (%)",
        "R07 Missing (%)",
        "R08 Missing (%)",
    ]
    print(df_missing[cols_miss_display].to_string(index=False))


if __name__ == "__main__":
    run_full_analysis()
