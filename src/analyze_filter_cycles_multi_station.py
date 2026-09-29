"""
Analyzes particulate filter replacement cycles across multiple stations and multi-year dry windows.
Identifies step drops (diff(3) < -threshold) during verified dry periods (precip_24h == 0),
measures the time intervals between successive replacements, and compares to EPA's documented
"once or twice a week" operational schedule.
Saves data/processed/multi_station_filter_cycle_summary.csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np


STATIONS = [
    {"id": "ca_san_diego", "name": "San Diego, CA", "threshold": -450.0},
    {"id": "tx_dallas", "name": "Dallas, TX", "threshold": -300.0},
    {"id": "al_birmingham", "name": "Birmingham, AL", "threshold": -300.0},
    {"id": "dc_washington", "name": "Washington, DC", "threshold": -250.0},
]


def detect_filter_drops(df: pd.DataFrame, drop_threshold: float) -> pd.DataFrame:
    df["dt"] = pd.to_datetime(df["utc_hour"])
    # 24-hour dry mask
    df["precip_24h"] = df["precip_1h_mm"].rolling(24, min_periods=12).sum()
    dry_mask = (df["precip_24h"] == 0.0) & (df["precip_1h_mm"] == 0.0) & df["has_radnet_obs"]
    
    sub = df[dry_mask].copy().sort_values("dt")
    sub["cpm_diff_3h"] = sub["gross_cpm"].diff(3)
    
    candidate_drops = sub[sub["cpm_diff_3h"] < drop_threshold].copy()
    if candidate_drops.empty:
        return pd.DataFrame()

    # Cluster consecutive hours of the same drop (minimum 24 hours separation)
    candidate_drops["time_diff"] = candidate_drops["dt"].diff()
    unique_drops = candidate_drops[(candidate_drops["time_diff"] > pd.Timedelta(hours=24)) | (candidate_drops["time_diff"].isna())].copy()
    return unique_drops


def analyze_all():
    summaries = []
    
    for st in STATIONS:
        csv_file = Path(f"data/processed/merged_{st['id']}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        
        # Analyze 2021-2024 (4 full years of modern operation)
        df["dt"] = pd.to_datetime(df["utc_hour"])
        df_sub = df[(df["dt"] >= "2021-01-01") & (df["dt"] <= "2024-12-31")].copy()
        
        drops = detect_filter_drops(df_sub, st["threshold"])
        if len(drops) < 3:
            continue
            
        intervals = drops["dt"].diff().dt.total_seconds() / 86400.0
        # Filter intervals to plausible operational filter change windows (1.5 days to 14 days)
        # to exclude long multi-month gaps between dry seasons
        valid_intervals = intervals[(intervals >= 1.5) & (intervals <= 14.0)]
        
        mean_int = round(valid_intervals.mean(), 2) if len(valid_intervals) > 0 else np.nan
        median_int = round(valid_intervals.median(), 2) if len(valid_intervals) > 0 else np.nan
        mean_drop = round(abs(drops["cpm_diff_3h"].mean()), 1)
        
        summaries.append({
            "Station": st["name"],
            "Analysis_Window": "2021-2024",
            "Drop_Threshold_CPM": st["threshold"],
            "Detected_Step_Drops": len(drops),
            "Plausible_Operational_Intervals": len(valid_intervals),
            "Mean_Interval_Days": mean_int,
            "Median_Interval_Days": median_int,
            "Mean_Step_Drop_CPM": mean_drop,
        })
        print(f"{st['name']}: {len(drops)} drops detected. Median interval: {median_int} days (Mean: {mean_int} days). Mean step drop: {mean_drop} CPM.")

    out_df = pd.DataFrame(summaries)
    out_csv = Path("data/processed/multi_station_filter_cycle_summary.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved multi-station filter cycle summary to {out_csv}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    analyze_all()
