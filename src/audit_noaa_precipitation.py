"""
Performs a comprehensive audit of NOAA NCEI Global-Hourly weather observations
for the 5 pilot airport stations across 2017-2025.
Audits the AA1 liquid precipitation field (accumulation period, depth, quality),
sea level pressure (SLP), temperature (TMP), and dew point (DEW).
"""

import pandas as pd
import numpy as np
from pathlib import Path


NOAA_STATIONS = {
    "KBHM": {"name": "Birmingham, AL", "station_id": "72228013876"},
    "KDCA": {"name": "Washington, DC", "station_id": "72405013743"},
    "KSAN": {"name": "San Diego, CA", "station_id": "72290023188"},
    "KDFW": {"name": "Dallas, TX", "station_id": "72259003927"},
    "KTPA": {"name": "Tampa, FL", "station_id": "72211012842"},
}

YEARS = list(range(2017, 2026))


def parse_aa1(val):
    """
    Parses AA1 string: period_hours,depth_tenths_mm,condition_code,quality_code
    Returns (period_hours, depth_mm, quality_code) or (None, None, None)
    """
    if pd.isna(val) or not isinstance(val, str):
        return None, None, None
    parts = val.split(",")
    if len(parts) >= 2:
        try:
            period = int(parts[0])
            depth_tenths = int(parts[1])
            # 9999 indicates missing depth
            depth_mm = depth_tenths / 10.0 if depth_tenths != 9999 else None
            quality = parts[3] if len(parts) >= 4 else None
            return period, depth_mm, quality
        except (ValueError, IndexError):
            return None, None, None
    return None, None, None


def audit_station_year(call_sign: str, station_name: str, year: int) -> dict:
    file_path = Path(f"data/raw/noaa_{call_sign.lower()}_{year}.csv")
    if not file_path.exists():
        return {"station": call_sign, "city": station_name, "year": year, "status": "MISSING"}

    # Read relevant columns
    cols_to_read = ["DATE", "REPORT_TYPE", "TMP", "DEW", "SLP", "AA1"]
    df = pd.read_csv(file_path, usecols=lambda c: c in cols_to_read, low_memory=False)

    total_records = len(df)
    # Parse dates
    df["dt"] = pd.to_datetime(df["DATE"], errors="coerce")
    
    # Report types
    report_counts = df["REPORT_TYPE"].str.strip().value_counts().to_dict()
    # FM-15 = routine hourly METAR; FM-16 = special SPECI observation
    fm15_count = report_counts.get("FM-15", 0)
    fm16_count = report_counts.get("FM-16", 0)

    # Precipitation AA1
    aa1_present_mask = df["AA1"].notna()
    aa1_present_count = aa1_present_mask.sum()
    aa1_present_pct = round((aa1_present_count / total_records) * 100, 2) if total_records else 0

    parsed_aa1 = [parse_aa1(v) for v in df.loc[aa1_present_mask, "AA1"]]
    periods = [p[0] for p in parsed_aa1 if p[0] is not None]
    depths = [p[1] for p in parsed_aa1 if p[1] is not None]
    qualities = [p[2] for p in parsed_aa1 if p[2] is not None]

    # Period breakdown
    period_1hr_count = sum(1 for p in periods if p == 1)
    period_1hr_pct = round((period_1hr_count / len(periods)) * 100, 2) if periods else 0

    # Quality codes: 1 = passed, 5 = passed all checks
    valid_qual_count = sum(1 for q in qualities if q in ["1", "5"])
    valid_qual_pct = round((valid_qual_count / len(qualities)) * 100, 2) if qualities else 0

    # Rain occurrences (depth > 0)
    rain_events = [d for d in depths if d > 0]
    rain_hours_count = len(rain_events)
    total_precip_mm = round(sum(rain_events), 1) if rain_events else 0.0
    max_1hr_precip_mm = round(max(rain_events), 1) if rain_events else 0.0

    # Sea Level Pressure (SLP)
    # Format: depth_hPa_tenths,quality
    slp_valid = 0
    if "SLP" in df.columns:
        for val in df["SLP"].dropna():
            if isinstance(val, str) and "," in val:
                slp_str = val.split(",")[0]
                if slp_str != "99999":
                    slp_valid += 1
    slp_pct = round((slp_valid / total_records) * 100, 2) if total_records else 0

    # Temperature and Dew Point
    tmp_valid = 0
    if "TMP" in df.columns:
        for val in df["TMP"].dropna():
            if isinstance(val, str) and "," in val:
                if val.split(",")[0] not in ["+9999", "9999"]:
                    tmp_valid += 1
    tmp_pct = round((tmp_valid / total_records) * 100, 2) if total_records else 0

    return {
        "call_sign": call_sign,
        "station_name": station_name,
        "year": year,
        "total_records": total_records,
        "fm15_routine_count": fm15_count,
        "fm16_speci_count": fm16_count,
        "aa1_present_count": aa1_present_count,
        "aa1_present_pct": aa1_present_pct,
        "period_1hr_count": period_1hr_count,
        "period_1hr_pct": period_1hr_pct,
        "rain_hours_count": rain_hours_count,
        "total_precip_mm": total_precip_mm,
        "max_1hr_precip_mm": max_1hr_precip_mm,
        "slp_valid_pct": slp_pct,
        "tmp_valid_pct": tmp_pct,
        "quality_pass_pct": valid_qual_pct,
    }


def run_noaa_audit():
    audit_rows = []
    for call_sign, info in NOAA_STATIONS.items():
        print(f"Auditing {call_sign} ({info['name']})...")
        for yr in YEARS:
            res = audit_station_year(call_sign, info["name"], yr)
            audit_rows.append(res)

    audit_df = pd.DataFrame(audit_rows)
    out_path = Path("data/processed/noaa_precipitation_audit.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_path, index=False)
    print(f"Saved NOAA precipitation audit to {out_path}")

    # Station summary across 2017-2025
    summary_rows = []
    for call_sign, grp in audit_df.groupby("call_sign"):
        name = grp["station_name"].iloc[0]
        total_obs = grp["total_records"].sum()
        total_rain_hrs = grp["rain_hours_count"].sum()
        mean_annual_precip = round(grp["total_precip_mm"].mean(), 1)
        mean_aa1_pct = round(grp["aa1_present_pct"].mean(), 2)
        mean_1hr_pct = round(grp["period_1hr_pct"].mean(), 2)
        mean_slp_pct = round(grp["slp_valid_pct"].mean(), 2)
        mean_tmp_pct = round(grp["tmp_valid_pct"].mean(), 2)

        summary_rows.append({
            "Station": call_sign,
            "City": name,
            "Total Obs (2017-2025)": total_obs,
            "AA1 Present (%)": mean_aa1_pct,
            "1-Hour Period Ratio (%)": mean_1hr_pct,
            "Total Rain Hours": total_rain_hrs,
            "Mean Annual Precip (mm)": mean_annual_precip,
            "SLP Completeness (%)": mean_slp_pct,
            "TMP Completeness (%)": mean_tmp_pct,
        })

    summary_df = pd.DataFrame(summary_rows)
    sum_path = Path("data/processed/noaa_station_summary.csv")
    summary_df.to_csv(sum_path, index=False)
    print("\n=== NOAA Weather Audit Summary (2017-2025) ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_noaa_audit()
