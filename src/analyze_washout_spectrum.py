"""
Phase 3 Script: Empirical Analysis of Radon Progeny Washout Gamma Energy Spectrum.

Computes the empirical excess channel shares across all verified rain hours across
the 5 pilot stations (2017-2025) on continuous regular grids.

For every rain hour (precip >= 1.0 mm/h and gross excess > 100 CPM):
    Delta C_k(t) = C_k(t) - mu_dry,k
    Share_k(t) = Delta C_k(t) / sum_j Delta C_j(t)

Saves:
- data/processed/rain_washout_spectral_shares.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np


STATIONS = [
    {"id": "al_birmingham", "name": "Birmingham, AL"},
    {"id": "dc_washington", "name": "Washington, DC"},
    {"id": "ca_san_diego", "name": "San Diego, CA"},
    {"id": "tx_dallas", "name": "Dallas, TX"},
    {"id": "fl_tampa", "name": "Tampa, FL"},
]

CHANNELS = [f"cpm_r0{i}" for i in range(2, 10)]
CHANNEL_NAMES = {
    "cpm_r02": "R02 (101-200 keV)",
    "cpm_r03": "R03 (201-400 keV)",
    "cpm_r04": "R04 (401-600 keV)",
    "cpm_r05": "R05 (601-800 keV)",
    "cpm_r06": "R06 (801-1000 keV)",
    "cpm_r07": "R07 (1001-1400 keV)",
    "cpm_r08": "R08 (1401-1800 keV)",
    "cpm_r09": "R09 (1801-2200 keV)",
}


def analyze_washout_spectrum():
    all_shares = []
    station_rows = []

    for st in STATIONS:
        st_id = st["id"]
        csv_file = Path(f"data/processed/merged_{st_id}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        df["dt"] = pd.to_datetime(df["utc_hour"])
        df = df.sort_values("dt").reset_index(drop=True)
        
        # Continuous calendar grid operation: compute precip_24h before any row filtering
        df["precip_24h"] = df["precip_1h_mm"].rolling(24, min_periods=12).sum()
        
        valid = df["has_radnet_obs"] & df["rad_complete_channels"]
        df_valid = df[valid].copy()
        
        # Dry baseline: current rain == 0 and preceding 24h rain == 0
        dry = (df_valid["precip_1h_mm"] == 0.0) & (df_valid["precip_24h"] == 0.0)
        dry_means = df_valid.loc[dry, CHANNELS].mean()

        # Rain hours: precip >= 1.0 mm/h
        rain = (df_valid["precip_1h_mm"] >= 1.0)
        rain_df = df_valid[rain].copy()

        excess = rain_df[CHANNELS] - dry_means
        excess_gross = excess.sum(axis=1)

        # Select hours with significant positive surge (> 100 CPM)
        pos_mask = excess_gross > 100.0
        shares = excess[pos_mask].div(excess_gross[pos_mask], axis=0)
        shares["station_id"] = st_id
        shares["station_name"] = st["name"]
        shares["excess_gross_cpm"] = excess_gross[pos_mask]
        all_shares.append(shares)

        # Summary for station
        st_summary = {
            "Scope": st["name"],
            "Station_ID": st_id,
            "Rain_Hours_Analyzed": len(shares),
        }
        for ch in CHANNELS:
            st_summary[f"{ch}_mean_pct"] = round(shares[ch].mean() * 100, 2)
            st_summary[f"{ch}_std_pct"] = round(shares[ch].std() * 100, 2)
            st_summary[f"{ch}_median_pct"] = round(shares[ch].median() * 100, 2)
        st_summary["r07_plus_r08_mean_pct"] = round((shares["cpm_r07"].mean() + shares["cpm_r08"].mean()) * 100, 2)
        station_rows.append(st_summary)

    # Pooled network summary
    df_pooled = pd.concat(all_shares, ignore_index=True)
    pooled_summary = {
        "Scope": "Pooled Network (All 5 Stations)",
        "Station_ID": "pooled_all",
        "Rain_Hours_Analyzed": len(df_pooled),
    }
    for ch in CHANNELS:
        pooled_summary[f"{ch}_mean_pct"] = round(df_pooled[ch].mean() * 100, 2)
        pooled_summary[f"{ch}_std_pct"] = round(df_pooled[ch].std() * 100, 2)
        pooled_summary[f"{ch}_median_pct"] = round(df_pooled[ch].median() * 100, 2)
    pooled_summary["r07_plus_r08_mean_pct"] = round((df_pooled["cpm_r07"].mean() + df_pooled["cpm_r08"].mean()) * 100, 2)
    station_rows.append(pooled_summary)

    out_df = pd.DataFrame(station_rows)
    out_csv = Path("data/processed/rain_washout_spectral_shares.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved empirical washout spectral shares to {out_csv}")
    print("\n=== Empirical Washout Channel Excess Shares (% of Excess Gross) ===")
    display_cols = ["Scope", "Rain_Hours_Analyzed", "cpm_r02_mean_pct", "cpm_r03_mean_pct", "cpm_r04_mean_pct", "cpm_r05_mean_pct", "cpm_r07_mean_pct", "cpm_r08_mean_pct", "r07_plus_r08_mean_pct"]
    print(out_df[display_cols].to_string(index=False))

    return out_df, df_pooled


if __name__ == "__main__":
    analyze_washout_spectrum()
