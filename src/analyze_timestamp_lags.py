"""
Computes and plots full cross-correlation lag profiles between hourly precipitation
and RadNet radiation metrics (Gross CPM, Dose Rate) across lags from -12 to +12 hours
for all 5 pilot stations.
Operates strictly on the continuous 78,888-hour calendar grid so that lag shifts represent
exact calendar hours regardless of telemetry gaps.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATIONS = [
    {"id": "al_birmingham", "name": "Birmingham, AL", "color": "#1f77b4"},
    {"id": "dc_washington", "name": "Washington, DC", "color": "#2ca02c"},
    {"id": "ca_san_diego", "name": "San Diego, CA", "color": "#ff7f0e"},
    {"id": "tx_dallas", "name": "Dallas, TX", "color": "#d62728"},
    {"id": "fl_tampa", "name": "Tampa, FL", "color": "#9467bd"},
]

LAGS = list(range(-12, 13))


def compute_lag_correlations():
    results = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for st in STATIONS:
        csv_file = Path(f"data/processed/merged_{st['id']}_2017_2025.csv.gz")
        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)
        df["dt"] = pd.to_datetime(df["utc_hour"])
        df = df.sort_values("dt").reset_index(drop=True)

        cpm_corrs = []
        dose_corrs = []

        for lag in LAGS:
            # Shift on continuous calendar grid:
            # lag > 0: rain leads radiation (radiation at t, rain at t - lag)
            shifted_precip = df["precip_1h_mm"].shift(lag)

            # Filter pairs where both radiation and shifted_precip are valid
            mask_cpm = df["has_radnet_obs"] & df["rad_complete_channels"] & df["gross_cpm"].notna() & shifted_precip.notna()
            mask_dose = df["has_radnet_obs"] & df["dose_rate_nsvh"].notna() & shifted_precip.notna()

            r_cpm = df.loc[mask_cpm, "gross_cpm"].corr(shifted_precip.loc[mask_cpm])
            r_dose = df.loc[mask_dose, "dose_rate_nsvh"].corr(shifted_precip.loc[mask_dose])

            cpm_corrs.append(r_cpm)
            dose_corrs.append(r_dose)

            results.append({
                "Station": st["name"],
                "Station_ID": st["id"],
                "Lag_Hours": lag,
                "Corr_Precip_GrossCPM": round(r_cpm, 4) if pd.notna(r_cpm) else np.nan,
                "Corr_Precip_DoseRate": round(r_dose, 4) if pd.notna(r_dose) else np.nan,
            })

        axes[0].plot(LAGS, cpm_corrs, marker="o", linewidth=2.0, color=st["color"], label=st["name"])
        axes[1].plot(LAGS, dose_corrs, marker="s", linewidth=2.0, color=st["color"], label=st["name"])

    # Panel 0: Gross CPM
    axes[0].axvline(0, color="gray", linestyle="--", alpha=0.7, label="Zero Lag (Synchronous)")
    axes[0].set_ylabel("Pearson Correlation (r)", fontsize=11, fontweight="bold")
    axes[0].set_title("Cross-Correlation Lag Profile: Hourly Precipitation vs. Gross Count Rate (CPM)\n(Calendar-Grid Shifts: Positive Lag = Rain Leads Radiation)", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel 1: Dose Rate
    axes[1].axvline(0, color="gray", linestyle="--", alpha=0.7, label="Zero Lag (Synchronous)")
    axes[1].set_ylabel("Pearson Correlation (r)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Lag (Hours): Rain Shift Relative to Radiation", fontsize=11, fontweight="bold")
    axes[1].set_title("Cross-Correlation Lag Profile: Hourly Precipitation vs. Dose Rate (nSv/h)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(LAGS)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "precipitation_radnet_lag_cross_correlation.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved lag cross-correlation figure to {fig_path}")

    df_results = pd.DataFrame(results)
    out_csv = Path("data/processed/lag_cross_correlation.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Saved lag cross-correlation table to {out_csv}")


if __name__ == "__main__":
    compute_lag_correlations()
