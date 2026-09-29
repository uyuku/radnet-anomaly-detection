"""
Plots an illustrative 30-day timeline of RadNet Gross CPM, rolling baseline, 3-sigma and 5-sigma thresholds,
and hourly precipitation, explicitly highlighting the false alarms triggered by rain events.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_alarm_timeline():
    csv_file = Path("data/processed/merged_al_birmingham_2017_2025.csv.gz")
    df = pd.read_csv(csv_file)
    df["dt"] = pd.to_datetime(df["utc_hour"])

    # Focus on April 15 to May 15, 2023 (Spring convective season in Birmingham)
    mask = (df["dt"] >= "2023-04-15 00:00:00") & (df["dt"] <= "2023-05-15 23:00:00")
    sub = df[mask].copy().sort_values("dt")

    # Compute dry parameters for Birmingham
    dry_df = df[(df["precip_1h_mm"] == 0) & (df["precip_1h_mm"].rolling(24, min_periods=12).sum() == 0)]
    mu_dry = dry_df["gross_cpm"].mean()
    sigma_dry = dry_df["gross_cpm"].std()

    thresh_3s = mu_dry + 3.0 * sigma_dry
    thresh_5s = mu_dry + 5.0 * sigma_dry

    sub["is_alarm_3s"] = sub["gross_cpm"] > thresh_3s
    sub["is_alarm_5s"] = sub["gross_cpm"] > thresh_5s

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    # Top Panel: Radiation & Thresholds
    ax0 = axes[0]
    ax0.plot(sub["dt"], sub["gross_cpm"], color="#1f4e79", linewidth=1.6, label="Gross Count Rate (CPM)")
    ax0.axhline(mu_dry, color="black", linestyle="--", linewidth=1.2, label=f"Dry Mean ({mu_dry:.0f} CPM)")
    ax0.axhline(thresh_3s, color="#ff7f0e", linestyle="--", linewidth=1.5, label=f"3-Sigma Threshold ({thresh_3s:.0f} CPM)")
    ax0.axhline(thresh_5s, color="#d62728", linestyle="--", linewidth=1.8, label=f"5-Sigma Threshold ({thresh_5s:.0f} CPM)")

    # Highlight alarm periods
    ax0.fill_between(sub["dt"], thresh_3s, sub["gross_cpm"], where=(sub["gross_cpm"] > thresh_3s),
                     color="#ff7f0e", alpha=0.3, label="False Alarm Region (> 3-Sigma)")

    ax0.set_ylabel("Gross Count Rate (CPM)", fontsize=11, fontweight="bold")
    ax0.set_title("Operational Failure of Fixed-Threshold Alarms During Rain Events\nBirmingham, AL (April 15 - May 15, 2023): Repeated False Alarms Triggered by Natural Washout",
                  fontsize=12, fontweight="bold")
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(loc="upper right", fontsize=9)

    # Bottom Panel: Precipitation
    ax1 = axes[1]
    ax1.bar(sub["dt"], sub["precip_1h_mm"], width=0.035, color="#2ca02c", edgecolor="#1b611b", label="NOAA Hourly Rainfall (mm)")
    ax1.set_ylabel("Precipitation (mm/h)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Date (UTC)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "baseline_alarm_example_timeline.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved timeline figure to {fig_path}")


if __name__ == "__main__":
    plot_alarm_timeline()
