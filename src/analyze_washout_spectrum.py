"""
Phase 3 Script: Empirical Analysis of Radon Progeny Washout Gamma Energy Spectrum.

Computes the empirical excess channel shares across all verified rain hours across
the 5 pilot stations (2017-2025) on continuous regular grids.

For every rain hour (precip >= 1.0 mm/h and gross excess > 100 CPM):
    Delta C_k(t) = C_k(t) - mu_dry,k
    Delta Gross(t) = sum_j Delta C_j(t)

Calculates:
1. Ratio-of-Sums: sum_t(Delta C_k) / sum_t(Delta Gross)
2. 1,000-draw bootstrap 95% confidence intervals on R07+R08 ratio of sums.
3. Hour-level percentiles (5th, 25th, 50th, 75th, 95th) of R07+R08 share.
4. Conventional hour-level mean, std, and median shares.

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


def compute_ratio_and_ci(df_exc: pd.DataFrame, rng: np.random.RandomState, n_boot: int = 1000):
    """
    Computes ratio of sums: sum(Delta C_k) / sum(Delta Gross),
    and 95% bootstrap confidence intervals for the R07+R08 ratio of sums.
    """
    sum_ch = df_exc[CHANNELS].sum()
    sum_gross = df_exc["excess_gross_cpm"].sum()
    ratio_sums = sum_ch / sum_gross
    r07_08_ratio = (sum_ch["cpm_r07"] + sum_ch["cpm_r08"]) / sum_gross

    n = len(df_exc)
    boot_ratios = []
    for _ in range(n_boot):
        sample = df_exc.iloc[rng.randint(0, n, size=n)]
        s_ch = sample[CHANNELS].sum()
        s_g = sample["excess_gross_cpm"].sum()
        boot_ratios.append((s_ch["cpm_r07"] + s_ch["cpm_r08"]) / s_g)

    ci_low, ci_high = np.percentile(boot_ratios, [2.5, 97.5])
    return ratio_sums, r07_08_ratio, ci_low, ci_high


def analyze_washout_spectrum():
    rng = np.random.RandomState(42)
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

        valid = df["has_radnet_obs"].fillna(False).astype(bool) & df["rad_complete_channels"].fillna(False).astype(bool)
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
        excess_pos = excess[pos_mask].copy()
        excess_pos["excess_gross_cpm"] = excess_gross[pos_mask]
        excess_pos["station_id"] = st_id
        excess_pos["station_name"] = st["name"]

        # Hour-level shares (ratio per hour)
        shares = excess_pos[CHANNELS].div(excess_pos["excess_gross_cpm"], axis=0)
        shares["station_id"] = st_id
        shares["station_name"] = st["name"]
        shares["excess_gross_cpm"] = excess_pos["excess_gross_cpm"]
        shares["r07_plus_r08"] = shares["cpm_r07"] + shares["cpm_r08"]
        all_shares.append(shares)

        # Ratio of sums and bootstrap CI
        ratio_sums, r07_08_ratio, ci_low, ci_high = compute_ratio_and_ci(excess_pos, rng)

        # Hour-level percentiles of R07+R08
        hour_pcts = np.percentile(shares["r07_plus_r08"] * 100, [5, 25, 50, 75, 95])

        st_summary = {
            "Scope": st["name"],
            "Station_ID": st_id,
            "Rain_Hours_Analyzed": len(shares),
            "r07_plus_r08_ratio_of_sums_pct": round(r07_08_ratio * 100, 2),
            "r07_plus_r08_ci95_low_pct": round(ci_low * 100, 2),
            "r07_plus_r08_ci95_high_pct": round(ci_high * 100, 2),
            "r07_plus_r08_hour_p05_pct": round(hour_pcts[0], 2),
            "r07_plus_r08_hour_p25_pct": round(hour_pcts[1], 2),
            "r07_plus_r08_hour_median_pct": round(hour_pcts[2], 2),
            "r07_plus_r08_hour_p75_pct": round(hour_pcts[3], 2),
            "r07_plus_r08_hour_p95_pct": round(hour_pcts[4], 2),
            "r07_plus_r08_hour_mean_pct": round(shares["r07_plus_r08"].mean() * 100, 2),
        }
        for ch in CHANNELS:
            st_summary[f"{ch}_ratio_of_sums_pct"] = round(ratio_sums[ch] * 100, 2)
            st_summary[f"{ch}_hour_mean_pct"] = round(shares[ch].mean() * 100, 2)
            st_summary[f"{ch}_hour_std_pct"] = round(shares[ch].std() * 100, 2)
            st_summary[f"{ch}_hour_median_pct"] = round(shares[ch].median() * 100, 2)

        station_rows.append(st_summary)

    # Pooled network summary
    df_pooled = pd.concat(all_shares, ignore_index=True)
    pooled_excess = pd.DataFrame()
    for ch in CHANNELS:
        pooled_excess[ch] = df_pooled[ch] * df_pooled["excess_gross_cpm"]
    pooled_excess["excess_gross_cpm"] = df_pooled["excess_gross_cpm"]

    ratio_sums_pool, r07_08_pool, ci_low_pool, ci_high_pool = compute_ratio_and_ci(pooled_excess, rng)
    hour_pcts_pool = np.percentile(df_pooled["r07_plus_r08"] * 100, [5, 25, 50, 75, 95])

    pooled_summary = {
        "Scope": "Pooled Network (All 5 Stations)",
        "Station_ID": "pooled_all",
        "Rain_Hours_Analyzed": len(df_pooled),
        "r07_plus_r08_ratio_of_sums_pct": round(r07_08_pool * 100, 2),
        "r07_plus_r08_ci95_low_pct": round(ci_low_pool * 100, 2),
        "r07_plus_r08_ci95_high_pct": round(ci_high_pool * 100, 2),
        "r07_plus_r08_hour_p05_pct": round(hour_pcts_pool[0], 2),
        "r07_plus_r08_hour_p25_pct": round(hour_pcts_pool[1], 2),
        "r07_plus_r08_hour_median_pct": round(hour_pcts_pool[2], 2),
        "r07_plus_r08_hour_p75_pct": round(hour_pcts_pool[3], 2),
        "r07_plus_r08_hour_p95_pct": round(hour_pcts_pool[4], 2),
        "r07_plus_r08_hour_mean_pct": round(df_pooled["r07_plus_r08"].mean() * 100, 2),
    }
    for ch in CHANNELS:
        pooled_summary[f"{ch}_ratio_of_sums_pct"] = round(ratio_sums_pool[ch] * 100, 2)
        pooled_summary[f"{ch}_hour_mean_pct"] = round(df_pooled[ch].mean() * 100, 2)
        pooled_summary[f"{ch}_hour_std_pct"] = round(df_pooled[ch].std() * 100, 2)
        pooled_summary[f"{ch}_hour_median_pct"] = round(df_pooled[ch].median() * 100, 2)

    station_rows.append(pooled_summary)

    out_df = pd.DataFrame(station_rows)
    out_csv = Path("data/processed/rain_washout_spectral_shares.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved empirical washout spectral shares to {out_csv}")
    print("\n=== Empirical Washout Spectral Shares Summary ===")
    display_cols = [
        "Scope", "Rain_Hours_Analyzed",
        "r07_plus_r08_ratio_of_sums_pct", "r07_plus_r08_ci95_low_pct", "r07_plus_r08_ci95_high_pct",
        "r07_plus_r08_hour_p25_pct", "r07_plus_r08_hour_median_pct", "r07_plus_r08_hour_p75_pct"
    ]
    print(out_df[display_cols].to_string(index=False))

    return out_df, df_pooled


if __name__ == "__main__":
    analyze_washout_spectrum()
