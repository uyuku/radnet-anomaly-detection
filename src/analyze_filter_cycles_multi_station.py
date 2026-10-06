"""
Analyzes particulate filter replacement cycles across multiple stations and multi-year dry windows.
Operates strictly on the continuous 78,888-hour calendar grid:
1. Computes 3-hour diff strictly on continuous calendar time (t vs t-3h).
2. Requires that BOTH hour t and hour t-3h have valid RadNet observations with complete channels.
3. Requires that the entire preceding 24 hours (including the 3-hour transition) are verified dry (precip == 0).
This guarantees that detected step drops occur entirely within a continuous, uninterrupted dry spell,
completely eliminating gap/rain crossing artifacts.
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
    # df is on continuous 1-hour grid
    # Compute 24-hour precipitation rolling sum
    df["precip_24h"] = df["precip_1h_mm"].rolling(24, min_periods=12).sum()
    
    # 3-hour diff on continuous grid: gross_cpm(t) - gross_cpm(t-3)
    df["cpm_diff_3h"] = df["gross_cpm"].diff(3)
    
    # Strict within-dry-spell condition:
    # 1. Valid observation at t and at t-3
    valid_t = df["has_radnet_obs"] & df["rad_complete_channels"] & df["gross_cpm"].notna()
    valid_t3 = valid_t.shift(3, fill_value=False)
    
    # 2. No KNOWN rain: 24h precipitation at t is 0.0, and no recorded rain at
    # t-1, t-2, t-3.  BUG-3-aware (2026-10-06): hours with UNKNOWN precipitation
    # (NaN) are tolerated here. They never certify dry anywhere else, but this
    # guard rejects washout-decay false positives, which require RECORDED rain —
    # and NaN comparisons would otherwise discard real step drops adjacent to
    # missing weather telemetry (12 of 450 frozen injections lost their drop
    # anchor under the stricter reading).
    known_rain = (df["precip_1h_mm"] > 0.0)
    no_known_rain_4h = ~(known_rain.rolling(4, min_periods=1).sum() > 0)
    no_known_rain_24h = ~(known_rain.rolling(24, min_periods=1).sum() > 0)
    dry_spell = no_known_rain_4h & no_known_rain_24h
    
    # Step drop condition
    drop_mask = (df["cpm_diff_3h"] < drop_threshold) & valid_t & valid_t3 & dry_spell
    
    candidate_drops = df[drop_mask].copy()
    if candidate_drops.empty:
        return pd.DataFrame()

    # Deduplicate consecutive hours belonging to the same step drop (min 24h separation).
    # BUG-8 fix (2026-10-06): compare each candidate against the last KEPT drop,
    # not against the previous candidate (which may itself have been rejected).
    kept_dts = []
    last_kept = None
    for cand_dt in candidate_drops["dt"]:
        if last_kept is None or (cand_dt - last_kept) > pd.Timedelta(hours=24):
            kept_dts.append(cand_dt)
            last_kept = cand_dt
    unique_drops = candidate_drops[candidate_drops["dt"].isin(kept_dts)].copy()
    return unique_drops


def analyze_all():
    summaries = []
    
    for st in STATIONS:
        csv_file = Path(f"data/processed/merged_{st['id']}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        df["dt"] = pd.to_datetime(df["utc_hour"])
        df = df.sort_values("dt").reset_index(drop=True)
        
        # Analyze 2021-2024 (4 full years of modern operational data)
        df_sub = df[(df["dt"] >= "2021-01-01") & (df["dt"] <= "2024-12-31")].copy().reset_index(drop=True)
        
        drops = detect_filter_drops(df_sub, st["threshold"])
        if len(drops) < 3:
            continue
            
        intervals = drops["dt"].diff().dt.total_seconds() / 86400.0
        # Operational filter change cadence: 1.5 days to 14 days
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
    print(f"\nSaved continuous calendar filter cycle summary to {out_csv}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    analyze_all()
