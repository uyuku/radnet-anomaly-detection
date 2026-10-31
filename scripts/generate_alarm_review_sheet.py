#!/usr/bin/env python3
"""
scripts/generate_alarm_review_sheet.py

Generates an independent, blind review sheet and comprehensive diagnostic plots
for the author to audit 100 clean-background test alarms.

For each alarm, produces:
1. Multi-panel diagnostic plot (Gross CPM, multi-channel spectrometry, weather, filter steps).
2. Blank review sheet in docs/alarm_review_sheet.csv and docs/alarm_review_sheet.md.
All assigned label columns are strictly EMPTY for the author to complete.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DOCS_DIR = REPO_ROOT / "docs"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "alarm_review"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    alarms_csv = DOCS_DIR / "hand_labeled_test_alarms.csv"
    assert alarms_csv.exists(), f"Missing {alarms_csv}"

    df_alarms = pd.read_csv(alarms_csv)
    print(f"Loaded {len(df_alarms)} alarms from {alarms_csv}")

    # Pre-load station test data files
    station_dfs = {}
    unique_stations = df_alarms["station_id"].unique()
    for st_id in unique_stations:
        st_file = DATA_PROCESSED / f"labeled_{st_id}_test.csv.gz"
        print(f"Loading {st_file}...")
        sdf = pd.read_csv(st_file)
        sdf["dt"] = pd.to_datetime(sdf["dt"])
        sdf = sdf.sort_values("dt").reset_index(drop=True)
        station_dfs[st_id] = sdf

    review_rows = []

    print(f"Generating diagnostic plots for {len(df_alarms)} alarms...")
    for idx, row in df_alarms.iterrows():
        alm_id = row["alarm_id"]
        st_id = row["station_id"]
        st_name = row["station_name"]
        utc_dt = pd.to_datetime(row["utc_hour"])
        trig = str(row["triggering_models"])

        sdf = station_dfs[st_id]
        # Extract 48h context around alarm (+- 24 hours)
        t_start = utc_dt - pd.Timedelta(hours=24)
        t_end = utc_dt + pd.Timedelta(hours=24)
        sub = sdf[(sdf["dt"] >= t_start) & (sdf["dt"] <= t_end)].copy()

        # Generate 3-panel diagnostic plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.5, 1.3]})

        # Panel 1: Gross & Key Spectrometry Channels
        ax1.plot(sub["dt"], sub["gross_cpm"], color="#1f77b4", lw=2.0, label="Gross CPM")
        if "cpm_r02" in sub.columns:
            ax1.plot(sub["dt"], sub["cpm_r02"], color="#2ca02c", lw=1.2, alpha=0.8, label="Ch 2 (250-400 keV)")
        if "cpm_r03" in sub.columns:
            ax1.plot(sub["dt"], sub["cpm_r03"], color="#ff7f0e", lw=1.2, alpha=0.8, label="Ch 3 (400-550 keV)")
        if "cpm_r05" in sub.columns:
            ax1.plot(sub["dt"], sub["cpm_r05"], color="#9467bd", lw=1.2, alpha=0.8, label="Ch 5 (700-900 keV)")
        if "cpm_r07" in sub.columns:
            ax1.plot(sub["dt"], sub["cpm_r07"], color="#8c564b", lw=1.2, alpha=0.8, label="Ch 7 (1200-1600 keV)")

        ax1.axvline(utc_dt, color="#d62728", ls="--", lw=2.2, label=f"Alarm: {utc_dt.strftime('%Y-%m-%d %H:%M')}")
        ax1.set_ylabel("Count Rate (CPM)", fontsize=9, fontweight="bold")
        ax1.set_title(f"[{alm_id}] {st_name} — Alarm: {utc_dt.strftime('%Y-%m-%d %H:%M UTC')} | Triggered: {trig}", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=8, ncol=3, frameon=True)
        ax1.grid(True, ls=":", alpha=0.6)

        # Panel 2: Weather Telemetry (Precipitation & Pressure / RH / Temp)
        ax2_bar = ax2.twinx()
        if "precip_1h_mm" in sub.columns:
            ax2_bar.bar(sub["dt"], sub["precip_1h_mm"], width=0.03, color="#0072b2", alpha=0.45, label="Rain (mm/h)")
            ax2_bar.set_ylabel("Precip 1h (mm)", color="#0072b2", fontsize=9)
            ax2_bar.tick_params(axis="y", labelcolor="#0072b2")
            ax2_bar.set_ylim(bottom=0.0, top=max(5.0, sub["precip_1h_mm"].max() * 1.3))

        if "pressure_hpa" in sub.columns:
            ax2.plot(sub["dt"], sub["pressure_hpa"], color="#333333", lw=1.2, ls="-", label="Pressure (hPa)")
        if "rel_humidity_pct" in sub.columns:
            ax2.plot(sub["dt"], sub["rel_humidity_pct"], color="#009e73", lw=1.0, ls="--", label="RH (%)")
        if "temp_c" in sub.columns:
            ax2.plot(sub["dt"], sub["temp_c"], color="#d55e00", lw=1.0, ls=":", label="Temp (°C)")

        ax2.axvline(utc_dt, color="#d62728", ls="--", lw=2.2)
        ax2.set_ylabel("Weather Covariates", fontsize=9, fontweight="bold")
        ax2.legend(loc="upper left", fontsize=8, ncol=3, frameon=True)
        ax2.grid(True, ls=":", alpha=0.6)

        # Panel 3: Spectral Ratios & Step Continuity
        # Compute local relative change and spectral share ratio
        gross_series = sub["gross_cpm"].values
        diff_6h = np.zeros(len(gross_series))
        for k in range(len(gross_series)):
            if k >= 6 and gross_series[k-6] > 0:
                diff_6h[k] = (gross_series[k] - gross_series[k-6]) / gross_series[k-6]
        
        ax3.plot(sub["dt"], diff_6h, color="#e7298a", lw=1.2, label="6h Relative Δ CPM")
        if "cpm_r05" in sub.columns and "cpm_r03" in sub.columns:
            ratio_r05_r03 = sub["cpm_r05"] / (sub["cpm_r03"] + 1e-3)
            ax3.plot(sub["dt"], ratio_r05_r03, color="#7570b3", lw=1.2, label="R05 / R03 Ratio")
        
        ax3.axvline(utc_dt, color="#d62728", ls="--", lw=2.2)
        ax3.set_ylabel("Step / Spectral", fontsize=9, fontweight="bold")
        ax3.set_xlabel("Time (UTC)", fontsize=9, fontweight="bold")
        ax3.legend(loc="upper left", fontsize=8, ncol=2, frameon=True)
        ax3.grid(True, ls=":", alpha=0.6)

        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        plt.tight_layout()

        plot_filename = f"alarm_{alm_id}.png"
        plot_path = FIGURES_DIR / plot_filename
        plt.savefig(plot_path, dpi=180)
        plt.close()

        # Add to blank review sheet (label columns strictly empty)
        review_rows.append({
            "alarm_id": alm_id,
            "station_id": st_id,
            "station_name": st_name,
            "utc_hour": str(row["utc_hour"]),
            "gross_cpm": row["gross_cpm"],
            "precip_1h_mm": row["precip_1h_mm"] if pd.notna(row["precip_1h_mm"]) else 0.0,
            "hours_since_rain": row["hours_since_rain"] if pd.notna(row["hours_since_rain"]) else 168.0,
            "triggering_models": trig,
            "plot_relative_path": f"reports/figures/alarm_review/{plot_filename}",
            "assigned_category": "",        # BLANK FOR AUTHOR TO COMPLETE
            "confidence_score_1_to_5": "",  # BLANK FOR AUTHOR TO COMPLETE
            "rationale_notes": "",          # BLANK FOR AUTHOR TO COMPLETE
        })

    # Save CSV review sheet
    df_review = pd.DataFrame(review_rows)
    review_csv_path = DOCS_DIR / "alarm_review_sheet.csv"
    df_review.to_csv(review_csv_path, index=False)
    print(f"\nWrote {review_csv_path} with {len(df_review)} rows (label columns empty).")

    # Save Markdown review sheet
    md_lines = [
        "# Clean-Background Test Alarm Audit: Author Review Sheet",
        "",
        "This review sheet contains a stratified random sample of $N = 100$ clean-background alarms raised during the 2023–2025 held-out test split across the five pilot stations.",
        "Each alarm includes synchronous multi-channel scintillation spectrometry, surface meteorology, and step continuity context.",
        "",
        "## Categorization Guidelines for the Author",
        "Please inspect each alarm's diagnostic plot (`reports/figures/alarm_review/alarm_ALM-XXX.png`) and assign one of the following candidate categories in the `assigned_category` column, along with a confidence rating (1 = Low to 5 = High) and explanatory rationale:",
        "",
        "1. `rain_coincident`: Active rainfall at or immediately preceding the alarm ($P_{\\text{1h}} \\ge 0.1\\text{ mm/h}$), accompanied by characteristic radon progeny gamma surges ($^{214}\\text{Pb}$ and $^{214}\\text{Bi}$).",
        "2. `post_rain_washout`: Dry at the alarm timestamp ($P_{\\text{1h}} = 0\\text{ mm}$), but occurring within 6 hours of rain cessation with residual elevated $^{214}\\text{Bi}$ ($R_{05}$) and $^{214}\\text{Pb}$ ($R_{03}$).",
        "3. `filter_change_artifact`: Discontinuity coincident with particulate filter tape advancement or maintenance cycle (sharp baseline step $\\ge 25\\%$).",
        "4. `instrument_step_drift`: Sensor hardware modification, phototube gain adjustment, or permanent baseline step (e.g., Tampa post-March 2023 PMT replacement).",
        "5. `other_natural`: Non-precipitation atmospheric phenomena, such as strong nocturnal temperature inversions trapping terrestrial radon, or barometric pressure front passages.",
        "6. `unexplained`: Pure statistical counting fluctuation or Poisson noise without identifiable environmental or technical cause.",
        "",
        "## Review Table",
        "",
        "| Alarm ID | Station | Timestamp (UTC) | Gross CPM | Rain (mm/h) | Triggering Models | Diagnostic Plot | Assigned Category | Confidence (1-5) | Author Rationale |",
        "| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- | :---: | :--- |",
    ]

    for r in review_rows:
        alm_id = r["alarm_id"]
        st = r["station_name"]
        t = r["utc_hour"]
        gross = f"{r['gross_cpm']:.0f}"
        rain = f"{r['precip_1h_mm']:.1f}"
        trig = r["triggering_models"][:35] + ("..." if len(r["triggering_models"]) > 35 else "")
        plot_link = f"[{alm_id} Plot](file:///{REPO_ROOT}/{r['plot_relative_path']})"
        md_lines.append(f"| {alm_id} | {st} | {t} | {gross} | {rain} | {trig} | {plot_link} | | | |")

    review_md_path = DOCS_DIR / "alarm_review_sheet.md"
    review_md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {review_md_path}")


if __name__ == "__main__":
    main()
