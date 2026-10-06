"""
Characterizes non-weather baseline variations:
1. Diurnal atmospheric radon cycle driven by nocturnal boundary-layer inversions (on verified dry periods).
2. Particulate filter replacement cycle (sawtooth loading and reset step drops).
Generates publication figures and summary metric CSVs.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATIONS = [
    {"id": "al_birmingham", "name": "Birmingham, AL", "tz": "America/Chicago", "color": "#1f77b4"},
    {"id": "dc_washington", "name": "Washington, DC", "tz": "America/New_York", "color": "#2ca02c"},
    {"id": "ca_san_diego", "name": "San Diego, CA", "tz": "America/Los_Angeles", "color": "#ff7f0e"},
    {"id": "tx_dallas", "name": "Dallas, TX", "tz": "America/Chicago", "color": "#d62728"},
    {"id": "fl_tampa", "name": "Tampa, FL", "tz": "America/New_York", "color": "#9467bd"},
]


def analyze_diurnal_cycles():
    diurnal_summaries = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for st in STATIONS:
        csv_file = Path(f"data/processed/merged_{st['id']}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        df["dt"] = pd.to_datetime(df["utc_hour"])

        # Identify dry periods: 24h rolling precipitation == 0 on continuous grid
        df["precip_24h"] = df["precip_1h_mm"].rolling(24, min_periods=12).sum()
        dry_mask = (df["precip_24h"] == 0.0) & (df["precip_1h_mm"] == 0.0) & df["has_radnet_obs"] & df["rad_complete_channels"]
        dry_df = df.loc[dry_mask].copy()


        # Local solar hour
        dry_df["local_hour"] = dry_df["dt"].dt.tz_localize("UTC").dt.tz_convert(st["tz"]).dt.hour

        grouped = dry_df.groupby("local_hour").agg(
            gross_mean=("gross_cpm", "mean"),
            gross_std=("gross_cpm", "std"),
            gross_count=("gross_cpm", "count"),
            r03_mean=("cpm_r03", "mean"),
            r05_mean=("cpm_r05", "mean"),
            r07_mean=("cpm_r07", "mean"),
            r08_mean=("cpm_r08", "mean"),
            dose_mean=("dose_rate_nsvh", "mean"),
        ).reset_index()

        grouped["gross_sem"] = grouped["gross_std"] / np.sqrt(grouped["gross_count"])

        # Metrics
        mean_cpm = grouped["gross_mean"].mean()
        amp_cpm = grouped["gross_mean"].max() - grouped["gross_mean"].min()
        amp_pct = round((amp_cpm / mean_cpm) * 100, 2)
        peak_hr = int(grouped.loc[grouped["gross_mean"].idxmax(), "local_hour"])
        trough_hr = int(grouped.loc[grouped["gross_mean"].idxmin(), "local_hour"])

        mean_dose = grouped["dose_mean"].mean()
        amp_dose = grouped["dose_mean"].max() - grouped["dose_mean"].min()

        diurnal_summaries.append({
            "Station": st["name"],
            "Dry Hours Analyzed": len(dry_df),
            "Mean Dry Gross (CPM)": round(mean_cpm, 1),
            "Diurnal Amplitude (CPM)": round(amp_cpm, 1),
            "Diurnal Amplitude (%)": amp_pct,
            "Peak Local Hour": f"{peak_hr:02d}:00",
            "Trough Local Hour": f"{trough_hr:02d}:00",
            "Mean Dry Dose (nSv/h)": round(mean_dose, 1),
            "Dose Amplitude (nSv/h)": round(amp_dose, 1),
        })

        # Plot Panel 1: Gross Count Rate relative to station mean
        norm_gross = (grouped["gross_mean"] / mean_cpm - 1.0) * 100.0
        axes[0].plot(grouped["local_hour"], norm_gross, marker="o", linewidth=2.0, color=st["color"], label=f"{st['name']} (Amp: {amp_pct:.1f}%)")

        # Plot Panel 2: Dose rate relative to station mean
        norm_dose = (grouped["dose_mean"] / mean_dose - 1.0) * 100.0
        axes[1].plot(grouped["local_hour"], norm_dose, marker="s", linewidth=1.8, linestyle="--", color=st["color"], label=f"{st['name']} (Dose)")

    axes[0].set_ylabel("Gross Count Deviation (%)", fontsize=11, fontweight="bold")
    axes[0].set_title("Atmospheric Radon Diurnal Cycle on Verified Dry Days (2017-2025)\nNormalized Gross Count Rate (Precipitation = 0 mm for preceding 24h)", fontsize=13, fontweight="bold")
    axes[0].axhline(0, color="gray", linestyle=":", alpha=0.7)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].set_ylabel("Dose Rate Deviation (%)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Local Standard Time (Hour of Day)", fontsize=11, fontweight="bold")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    axes[1].axhline(0, color="gray", linestyle=":", alpha=0.7)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    diurnal_fig = OUTPUT_DIR / "diurnal_radon_cycle.png"
    plt.savefig(diurnal_fig, dpi=200)
    plt.close()
    print(f"Generated diurnal cycle plot: {diurnal_fig}")

    df_diurnal = pd.DataFrame(diurnal_summaries)
    out_csv = Path("data/processed/diurnal_cycle_summary.csv")
    df_diurnal.to_csv(out_csv, index=False)
    print(f"Saved diurnal summary to {out_csv}")
    print(df_diurnal.to_string(index=False))


def analyze_filter_replacement_cycles():
    """
    Examines particulate filter change signatures.
    Filters collect airborne particulates over several days, creating steady accumulation
    punctuated by a sharp step drop when the loaded filter is replaced by an operator.
    """
    # San Diego provides an ideal dry testbed with minimal rain interference
    csv_file = Path("data/processed/merged_ca_san_diego_2017_2025.csv.gz")
    df = pd.read_csv(csv_file)
    df["dt"] = pd.to_datetime(df["utc_hour"])

    # Look at a prolonged dry window in summer 2024
    start_dt = "2024-06-01 00:00:00"
    end_dt = "2024-07-15 00:00:00"
    mask = (df["dt"] >= start_dt) & (df["dt"] <= end_dt) & df["has_radnet_obs"]
    sub = df.loc[mask].copy().sort_values("dt")

    # Detect step drops: 3-hour difference < -400 CPM during dry weather
    sub["cpm_diff_3h"] = sub["gross_cpm"].diff(3)
    drop_threshold = -450.0
    filter_changes = sub[sub["cpm_diff_3h"] < drop_threshold].copy()

    # Cluster consecutive hours of the same drop (keep against last KEPT drop;
    # BUG-8 fix: previous version also silently dropped the first candidate)
    kept_dts = []
    last_kept = None
    for cand_dt in filter_changes["dt"]:
        if last_kept is None or (cand_dt - last_kept) > pd.Timedelta(hours=24):
            kept_dts.append(cand_dt)
            last_kept = cand_dt
    unique_drops = filter_changes[filter_changes["dt"].isin(kept_dts)].copy()

    # Compute intervals between filter changes
    intervals_days = unique_drops["dt"].diff().dt.total_seconds() / 86400.0
    mean_interval = round(intervals_days.dropna().mean(), 1)
    median_interval = round(intervals_days.dropna().median(), 1)

    print(f"\n=== Filter Replacement Cycle in San Diego (Summer 2024) ===")
    print(f"Detected filter drops: {len(unique_drops)}")
    print(f"Mean change interval: {mean_interval} days (Median: {median_interval} days)")

    # Plot filter change time series
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(sub["dt"], sub["gross_cpm"], color="#1f4e79", linewidth=1.5, label="Gross Count Rate (CPM)")

    for _, row in unique_drops.iterrows():
        ax.axvline(row["dt"], color="#d62728", linestyle="--", alpha=0.8, linewidth=1.5)
        ax.text(row["dt"], row["gross_cpm"] + 150, "Filter\nChange", color="#d62728", fontsize=8, fontweight="bold", ha="center")

    ax.set_ylabel("Gross Count Rate (CPM)", fontsize=11, fontweight="bold")
    ax.set_title("Particulate Filter Replacement Cycle in Dry Conditions (San Diego, June-July 2024)\nCharacteristic Step Drops and Multi-Day Accumulation Sawtooth", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    filter_fig = OUTPUT_DIR / "filter_cycle_sawtooth.png"
    plt.savefig(filter_fig, dpi=200)
    plt.close()
    print(f"Generated filter cycle plot: {filter_fig}")

    summary = {
        "Station": "San Diego, CA (Dry Control Window)",
        "Period": "2024-06-01 to 2024-07-15",
        "Detected Filter Changes": len(unique_drops),
        "Mean Change Interval (Days)": mean_interval,
        "Median Change Interval (Days)": median_interval,
        "Typical Step Drop (CPM)": round(abs(unique_drops["cpm_diff_3h"].mean()), 1),
    }
    df_filter = pd.DataFrame([summary])
    df_filter.to_csv("data/processed/filter_cycle_summary.csv", index=False)


if __name__ == "__main__":
    analyze_diurnal_cycles()
    analyze_filter_replacement_cycles()
