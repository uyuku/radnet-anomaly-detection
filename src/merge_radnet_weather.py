"""
Cleans and merges RadNet radiation data with NOAA NCEI weather observations
on a continuous UTC hourly grid (2017-2025, 78,888 hours) for the 5 pilot stations.
Enforces complete-channel validation, timestamp deduplication, and computes
relative humidity via the Magnus-Tetens formula.
"""

import zipfile
from pathlib import Path
import pandas as pd
import numpy as np


PILOT_STATIONS = [
    {
        "id": "AL_BIRMINGHAM",
        "name": "Birmingham, AL",
        "radnet_zip": "data/raw/al_birmingham_2025-2007.zip",
        "city_prefix": "AL_BIRMINGHAM",
        "noaa_call": "KBHM",
    },
    {
        "id": "DC_WASHINGTON",
        "name": "Washington, DC",
        "radnet_zip": "data/raw/dc_washington_2025-2006.zip",
        "city_prefix": "DC_WASHINGTON",
        "noaa_call": "KDCA",
    },
    {
        "id": "CA_SAN_DIEGO",
        "name": "San Diego, CA",
        "radnet_zip": "data/raw/ca_san_diego_2025-2006.zip",
        "city_prefix": "CA_SAN_DIEGO",
        "noaa_call": "KSAN",
    },
    {
        "id": "TX_DALLAS",
        "name": "Dallas, TX",
        "radnet_zip": "data/raw/tx_dallas_2025-2007.zip",
        "city_prefix": "TX_DALLAS",
        "noaa_call": "KDFW",
    },
    {
        "id": "FL_TAMPA",
        "name": "Tampa, FL",
        "radnet_zip": "data/raw/fl_tampa_2025-2008.zip",
        "city_prefix": "FL_TAMPA",
        "noaa_call": "KTPA",
    },
]

YEARS = list(range(2017, 2026))
CHANNEL_KEYS = ["R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09"]


def parse_noaa_precip(val):
    if pd.isna(val) or not isinstance(val, str):
        return 0.0
    parts = val.split(",")
    if len(parts) >= 2 and parts[0] == "01":
        try:
            d = int(parts[1])
            if d == 9999:
                return 0.0
            # Quality code verification per DECISIONS.md:
            # 1 = passed standard check, 5 = passed all checks, C = suspect, S = suspect
            quality = parts[3].strip() if len(parts) >= 4 else "1"
            if quality not in ["1", "5", "C", "S"]:
                return 0.0
            return d / 10.0
        except (ValueError, IndexError):
            return 0.0
    return 0.0


def parse_noaa_value(val, missing_val="99999", divisor=10.0):
    if pd.isna(val) or not isinstance(val, str):
        return np.nan
    parts = val.split(",")
    if len(parts) >= 1:
        s = parts[0].strip()
        if s != missing_val and s not in ["+9999", "9999", "99999"]:
            try:
                return float(s) / divisor
            except ValueError:
                return np.nan
    return np.nan


def calculate_relative_humidity(temp_c, dewpoint_c):
    """
    Magnus-Tetens formula (Alduchov & Eskridge, 1996)
    """
    if pd.isna(temp_c) or pd.isna(dewpoint_c):
        return np.nan
    a = 17.625
    b = 243.04
    alpha = (a * dewpoint_c) / (b + dewpoint_c)
    beta = (a * temp_c) / (b + temp_c)
    rh = 100.0 * np.exp(alpha - beta)
    return float(np.clip(rh, 0.0, 100.0))


def load_and_clean_radnet(station: dict) -> pd.DataFrame:
    dfs = []
    zip_path = Path(station["radnet_zip"])
    with zipfile.ZipFile(zip_path, "r") as z:
        for yr in YEARS:
            suffix = f"{station['city_prefix']}_{yr}.csv".lower()
            matched = [n for n in z.namelist() if n.lower().endswith(suffix)]
            if matched:
                with z.open(matched[0]) as f:
                    dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)

    # Time handling
    df["dt_raw"] = pd.to_datetime(df["SAMPLE COLLECTION TIME"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["dt_raw"])
    # Round to nearest UTC hour
    df["utc_hour"] = df["dt_raw"].dt.round("h")
    # Deduplicate keeping first
    df = df.drop_duplicates(subset=["utc_hour"]).copy()

    # Column mapping
    chan_cols_map = {k: [c for c in df.columns if k in c][0] for k in CHANNEL_KEYS}
    chan_cols = [chan_cols_map[k] for k in CHANNEL_KEYS]

    # Complete channel mask
    df["rad_complete_channels"] = df[chan_cols].notna().all(axis=1)

    # Clean gross CPM
    df["gross_cpm"] = np.nan
    df.loc[df["rad_complete_channels"], "gross_cpm"] = df.loc[df["rad_complete_channels"], chan_cols].sum(axis=1)

    dose_col = "DOSE EQUIVALENT RATE (nSv/h)"
    has_dose = dose_col in df.columns
    df["dose_rate_nsvh"] = df[dose_col] if has_dose else np.nan

    # Rename channels for clarity
    rename_dict = {chan_cols_map[k]: f"cpm_{k.lower()}" for k in CHANNEL_KEYS}
    df = df.rename(columns=rename_dict)

    keep_cols = [
        "utc_hour",
        "rad_complete_channels",
        "gross_cpm",
        "dose_rate_nsvh",
        "cpm_r02",
        "cpm_r03",
        "cpm_r04",
        "cpm_r05",
        "cpm_r06",
        "cpm_r07",
        "cpm_r08",
        "cpm_r09",
    ]
    return df[keep_cols]


def load_and_clean_noaa(station: dict) -> pd.DataFrame:
    dfs = []
    call_sign = station["noaa_call"].lower()
    for yr in YEARS:
        file_path = Path(f"data/raw/noaa_{call_sign}_{yr}.csv")
        if file_path.exists():
            cols = ["DATE", "TMP", "DEW", "SLP", "AA1"]
            df_yr = pd.read_csv(file_path, usecols=lambda c: c in cols, low_memory=False)
            dfs.append(df_yr)

    df = pd.concat(dfs, ignore_index=True)
    df["dt_raw"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["dt_raw"])
    df["utc_hour"] = df["dt_raw"].dt.round("h")

    # Parse weather variables
    df["precip_1h_mm"] = df["AA1"].apply(parse_noaa_precip)
    df["pressure_hpa"] = df["SLP"].apply(lambda v: parse_noaa_value(v, missing_val="99999", divisor=10.0))
    df["temp_c"] = df["TMP"].apply(lambda v: parse_noaa_value(v, missing_val="+9999", divisor=10.0))
    df["dewpoint_c"] = df["DEW"].apply(lambda v: parse_noaa_value(v, missing_val="+9999", divisor=10.0))

    # Calculate relative humidity row-wise
    df["rel_humidity_pct"] = [
        calculate_relative_humidity(t, d) for t, d in zip(df["temp_c"], df["dewpoint_c"])
    ]

    # Aggregate to unique UTC hour
    hourly = df.groupby("utc_hour").agg(
        {
            "precip_1h_mm": "max",
            "pressure_hpa": "mean",
            "temp_c": "mean",
            "dewpoint_c": "mean",
            "rel_humidity_pct": "mean",
        }
    ).reset_index()

    # Round floating aggregates
    hourly["pressure_hpa"] = hourly["pressure_hpa"].round(1)
    hourly["temp_c"] = hourly["temp_c"].round(1)
    hourly["dewpoint_c"] = hourly["dewpoint_c"].round(1)
    hourly["rel_humidity_pct"] = hourly["rel_humidity_pct"].round(1)

    return hourly


def merge_station(station: dict) -> tuple[pd.DataFrame, dict]:
    print(f"\nProcessing {station['name']} ({station['id']})...")
    df_rad = load_and_clean_radnet(station)
    df_weather = load_and_clean_noaa(station)

    # Complete 78,888 hour target grid (2017-01-01 00:00 to 2025-12-31 23:00)
    grid = pd.date_range("2017-01-01 00:00:00", "2025-12-31 23:00:00", freq="h", name="utc_hour")
    df_grid = pd.DataFrame(index=grid).reset_index()

    # Merge on complete grid
    merged = pd.merge(df_grid, df_rad, on="utc_hour", how="left")
    merged = pd.merge(merged, df_weather, on="utc_hour", how="left")

    # Add station identifier and calendar features
    merged["station_id"] = station["id"]
    merged["station_name"] = station["name"]
    merged["year"] = merged["utc_hour"].dt.year
    merged["month"] = merged["utc_hour"].dt.month
    merged["day"] = merged["utc_hour"].dt.day
    merged["hour_utc"] = merged["utc_hour"].dt.hour
    merged["dayofweek"] = merged["utc_hour"].dt.dayofweek

    # Flags
    merged["has_radnet_obs"] = merged["gross_cpm"].notna()
    merged["has_weather_obs"] = merged["temp_c"].notna()
    merged["is_raining"] = merged["precip_1h_mm"] > 0.0

    # Save compressed CSV
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"merged_{station['id'].lower()}_2017_2025.csv.gz"
    merged.to_csv(out_file, compression="gzip", index=False)
    print(f"Saved merged dataset ({len(merged):,} rows) to {out_file}")

    # Audit stats
    total_hours = len(merged)
    rad_hours = merged["has_radnet_obs"].sum()
    weather_hours = merged["has_weather_obs"].sum()
    both_hours = (merged["has_radnet_obs"] & merged["has_weather_obs"]).sum()
    rain_hours = (merged["is_raining"] & merged["has_radnet_obs"]).sum()
    total_precip = round(merged["precip_1h_mm"].sum(), 1)

    stats = {
        "Station": station["name"],
        "Station ID": station["id"],
        "Total Grid Hours": total_hours,
        "RadNet Valid Hours": rad_hours,
        "RadNet Coverage (%)": round(rad_hours / total_hours * 100, 2),
        "Weather Valid Hours": weather_hours,
        "Weather Coverage (%)": round(weather_hours / total_hours * 100, 2),
        "Synchronous Both Hours": both_hours,
        "Synchronous Coverage (%)": round(both_hours / total_hours * 100, 2),
        "Rain Hours (with RadNet)": rain_hours,
        "Total Precip (mm)": total_precip,
    }
    return merged, stats


def run_all_merges():
    audit_rows = []
    for st in PILOT_STATIONS:
        _, stats = merge_station(st)
        audit_rows.append(stats)

    df_audit = pd.DataFrame(audit_rows)
    out_path = Path("data/processed/merge_audit_summary.csv")
    df_audit.to_csv(out_path, index=False)
    print("\n=== Phase 1 Merge Audit Summary (2017-2025) ===")
    print(df_audit.to_string(index=False))


if __name__ == "__main__":
    run_all_merges()
