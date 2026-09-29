"""
Generates publication-quality multi-panel time-series figures illustrating
rain-induced radon progeny washout surges across all 5 pilot stations.
Shows Gross CPM, Dose Rate, Channel Spectra (R03, R05, R07, R08),
and NOAA Precipitation / Weather variables.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Curated high-impact rain events with clear before/during/after baselines
EVENTS = [
    {
        "station_id": "al_birmingham",
        "name": "Birmingham, AL (KBHM)",
        "start": "2023-04-28 12:00:00",
        "end": "2023-05-02 12:00:00",
        "filename": "rain_event_birmingham.png",
        "event_desc": "Convective Thunderstorm Washout (April 30, 2023)",
    },
    {
        "station_id": "dc_washington",
        "name": "Washington, DC (KDCA)",
        "start": "2022-07-07 12:00:00",
        "end": "2022-07-11 12:00:00",
        "filename": "rain_event_washington.png",
        "event_desc": "Severe Frontal Summer Rainstorm (July 9, 2022)",
    },
    {
        "station_id": "tx_dallas",
        "name": "Dallas, TX (KDFW)",
        "start": "2021-11-09 12:00:00",
        "end": "2021-11-13 12:00:00",
        "filename": "rain_event_dallas.png",
        "event_desc": "Cold Frontal Rainstorm Surge (Nov 11, 2021)",
    },
    {
        "station_id": "fl_tampa",
        "name": "Tampa, FL (KTPA)",
        "start": "2024-06-27 12:00:00",
        "end": "2024-07-01 12:00:00",
        "filename": "rain_event_tampa.png",
        "event_desc": "Subtropical Convective Rain Washout (June 29, 2024)",
    },
    {
        "station_id": "ca_san_diego",
        "name": "San Diego, CA (KSAN)",
        "start": "2020-03-10 12:00:00",
        "end": "2020-03-15 12:00:00",
        "filename": "rain_event_sandiego.png",
        "event_desc": "Pacific Frontal Washout Surge (March 12, 2020)",
    },
]


def plot_single_event(event_cfg: dict):
    st_id = event_cfg["station_id"]
    csv_file = Path(f"data/processed/merged_{st_id}_2017_2025.csv.gz")
    df = pd.read_csv(csv_file)
    df["dt"] = pd.to_datetime(df["utc_hour"])

    mask = (df["dt"] >= event_cfg["start"]) & (df["dt"] <= event_cfg["end"])
    sub = df.loc[mask].copy().sort_values("dt")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1.5]})

    # Panel 1: Gross Count Rate & Dose Equivalent Rate
    color_gross = "#1f4e79"
    color_dose = "#b22222"

    ax1.plot(sub["dt"], sub["gross_cpm"], color=color_gross, linewidth=2.0, label="Gross Count Rate (CPM)")
    ax1.set_ylabel("Gross Count Rate (CPM)", color=color_gross, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_gross)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(sub["dt"], sub["dose_rate_nsvh"], color=color_dose, linewidth=1.8, linestyle="--", label="Dose Equivalent Rate (nSv/h)")
    ax1_twin.set_ylabel("Dose Rate (nSv/h)", color=color_dose, fontsize=11, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor=color_dose)

    # Annotate peak
    peak_idx = sub["gross_cpm"].idxmax()
    if pd.notna(peak_idx):
        peak_row = sub.loc[peak_idx]
        ax1.annotate(
            f"Peak: {peak_row['gross_cpm']:.0f} CPM\n({peak_row['dose_rate_nsvh']:.0f} nSv/h)",
            xy=(peak_row["dt"], peak_row["gross_cpm"]),
            xytext=(15, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=color_gross, lw=1.5),
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_gross, alpha=0.8),
        )

    ax1.set_title(f"{event_cfg['name']} — {event_cfg['event_desc']}\nPrecipitation Washout Signal Response", fontsize=13, fontweight="bold")

    # Panel 2: Energy Channels (R03, R05, R07, R08)
    ax2.plot(sub["dt"], sub["cpm_r03"], color="#2ca02c", linewidth=1.8, label="R03: 201-400 keV (Pb-214: 352 keV, I-131: 364 keV)")
    ax2.plot(sub["dt"], sub["cpm_r05"], color="#ff7f0e", linewidth=1.8, label="R05: 601-800 keV (Bi-214: 609 keV, Cs-137: 662 keV)")
    ax2.plot(sub["dt"], sub["cpm_r07"], color="#9467bd", linewidth=1.5, linestyle="-.", label="R07: 1001-1400 keV (Bi-214: 1120 keV)")
    ax2.plot(sub["dt"], sub["cpm_r08"], color="#17becf", linewidth=1.5, linestyle=":", label="R08: 1401-1800 keV (Bi-214: 1764 keV + K-40)")
    ax2.set_ylabel("Spectral Channel (CPM)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: NOAA Precipitation (Bar chart) & Pressure (twin)
    color_rain = "#3498db"
    color_pres = "#7f8c8d"

    ax3.bar(sub["dt"], sub["precip_1h_mm"], width=0.035, color=color_rain, label="1-Hour Precip (mm)", alpha=0.8, edgecolor="#2980b9")
    ax3.set_ylabel("Precipitation (mm/h)", color="#2980b9", fontsize=11, fontweight="bold")
    ax3.tick_params(axis="y", labelcolor="#2980b9")
    ax3.set_ylim(bottom=0)
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(sub["dt"], sub["pressure_hpa"], color=color_pres, linewidth=1.5, linestyle="--", label="Sea Level Pressure (hPa)")
    ax3_twin.set_ylabel("Pressure (hPa)", color=color_pres, fontsize=10)
    ax3_twin.tick_params(axis="y", labelcolor=color_pres)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax3.set_xlabel("UTC Date & Time", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_file = OUTPUT_DIR / event_cfg["filename"]
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Generated event plot: {out_file}")


def plot_multi_station_comparison():
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=False)

    for i, event in enumerate(EVENTS):
        ax = axes[i]
        st_id = event["station_id"]
        csv_file = Path(f"data/processed/merged_{st_id}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        df["dt"] = pd.to_datetime(df["utc_hour"])

        mask = (df["dt"] >= event["start"]) & (df["dt"] <= event["end"])
        sub = df.loc[mask].copy().sort_values("dt")

        # Normalize time to hours relative to rain onset
        # Find rain onset
        rainy_hours = sub[sub["precip_1h_mm"] >= 1.0]
        if not rainy_hours.empty:
            onset_dt = rainy_hours.iloc[0]["dt"]
        else:
            onset_dt = sub.iloc[len(sub)//2]["dt"]

        sub["hours_from_onset"] = (sub["dt"] - onset_dt).dt.total_seconds() / 3600.0

        # Plot gross count rate
        color_cpm = "#1f4e79"
        ax.plot(sub["hours_from_onset"], sub["gross_cpm"], color=color_cpm, linewidth=2.0, label="Gross Count Rate (CPM)")
        ax.set_ylabel(f"{event['name'].split()[0]}\nCPM", color=color_cpm, fontweight="bold", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Twin for rain
        ax_twin = ax.twinx()
        ax_twin.bar(sub["hours_from_onset"], sub["precip_1h_mm"], width=0.8, color="#3498db", alpha=0.45, label="Rain (mm/h)")
        ax_twin.set_ylabel("Rain (mm)", color="#2980b9", fontsize=9)
        ax_twin.set_ylim(bottom=0)

        # Baseline and peak labels
        base = sub[sub["hours_from_onset"] < 0]["gross_cpm"].median()
        peak = sub["gross_cpm"].max()
        surge_pct = (peak - base) / base * 100
        ax.set_title(f"{event['name']} — Rain Washout Surge: Baseline {base:.0f} CPM -> Peak {peak:.0f} CPM (+{surge_pct:.1f}%)", fontsize=11, fontweight="bold")

        if i == 4:
            ax.set_xlabel("Hours Relative to Rain Onset (t = 0)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    comp_file = OUTPUT_DIR / "rain_washout_multi_station_comparison.png"
    plt.savefig(comp_file, dpi=200)
    plt.close()
    print(f"Generated multi-station comparison: {comp_file}")


if __name__ == "__main__":
    for ev in EVENTS:
        plot_single_event(ev)
    plot_multi_station_comparison()
    print("All Phase 1 rain event plots successfully generated.")
