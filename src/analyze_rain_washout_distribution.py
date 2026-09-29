"""
Analyzes the comprehensive distribution of radiation surges across all synchronous rain hours
(15,606 rain hours across 2017-2025) for all 5 pilot stations.
Computes quantiles (p10, p25, median, p75, p90, p95, p99, max) of absolute and percentage
CPM and dose rate surges above verified dry baseline.
Stratifies by rainfall intensity: Light (<=1mm), Moderate (1-5mm), Heavy (>5mm).
Saves data/processed/rain_washout_surge_distribution.csv and publication figures.
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


def analyze_washout_distributions():
    all_rain_records = []
    station_baselines = {}
    distribution_rows = []

    # 1. Load data and compute dry baselines
    for st in STATIONS:
        csv_file = Path(f"data/processed/merged_{st['id']}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        
        # Valid synchronous records
        valid = df[df["has_radnet_obs"] & df["has_weather_obs"] & df["rad_complete_channels"]].copy()
        
        # 24h rolling precipitation for dry baseline
        valid["precip_24h"] = valid["precip_1h_mm"].rolling(24, min_periods=12).sum()
        dry_mask = (valid["precip_24h"] == 0.0) & (valid["precip_1h_mm"] == 0.0)
        dry_df = valid[dry_mask]

        mu_cpm = dry_df["gross_cpm"].mean()
        sigma_cpm = dry_df["gross_cpm"].std()
        mu_dose = dry_df["dose_rate_nsvh"].mean()
        sigma_dose = dry_df["dose_rate_nsvh"].std()

        station_baselines[st["id"]] = {
            "mu_cpm": mu_cpm,
            "sigma_cpm": sigma_cpm,
            "mu_dose": mu_dose,
            "sigma_dose": sigma_dose,
        }

        # Filter to rain hours
        rain_mask = valid["precip_1h_mm"] > 0.0
        rain_df = valid[rain_mask].copy()

        # Compute surge metrics
        rain_df["delta_cpm"] = rain_df["gross_cpm"] - mu_cpm
        rain_df["surge_pct_cpm"] = (rain_df["gross_cpm"] / mu_cpm - 1.0) * 100.0
        rain_df["z_cpm"] = (rain_df["gross_cpm"] - mu_cpm) / sigma_cpm

        rain_df["delta_dose"] = rain_df["dose_rate_nsvh"] - mu_dose
        rain_df["surge_pct_dose"] = (rain_df["dose_rate_nsvh"] / mu_dose - 1.0) * 100.0

        rain_df["station_name"] = st["name"]
        rain_df["station_id"] = st["id"]

        # Rain category
        rain_df["rain_category"] = pd.cut(
            rain_df["precip_1h_mm"],
            bins=[-np.inf, 1.0, 5.0, np.inf],
            labels=["Light (<=1mm)", "Moderate (1-5mm)", "Heavy (>5mm)"]
        )

        all_rain_records.append(rain_df)

    pooled_rain = pd.concat(all_rain_records, ignore_index=True)
    print(f"Total analyzed rain hours across 5 stations: {len(pooled_rain)}")

    # 2. Compute Quantiles Function
    def summarize_series(series, group_label, cat_label, n_obs):
        qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        q_vals = series.quantile(qs).to_dict()
        return {
            "Group": group_label,
            "Category": cat_label,
            "Rain_Hours": n_obs,
            "Mean": round(series.mean(), 2),
            "Std": round(series.std(), 2),
            "p10": round(q_vals[0.10], 2),
            "p25": round(q_vals[0.25], 2),
            "p50_Median": round(q_vals[0.50], 2),
            "p75": round(q_vals[0.75], 2),
            "p90": round(q_vals[0.90], 2),
            "p95": round(q_vals[0.95], 2),
            "p99": round(q_vals[0.99], 2),
            "Max": round(series.max(), 2),
        }

    # A. Overall Pooled
    distribution_rows.append(summarize_series(pooled_rain["surge_pct_cpm"], "All Stations Pooled", "All Rain", len(pooled_rain)))
    distribution_rows.append(summarize_series(pooled_rain["delta_cpm"], "All Stations Pooled (Delta CPM)", "All Rain", len(pooled_rain)))
    distribution_rows.append(summarize_series(pooled_rain["z_cpm"], "All Stations Pooled (Z-Score)", "All Rain", len(pooled_rain)))

    # B. Stratified by rain intensity (Pooled)
    for cat in ["Light (<=1mm)", "Moderate (1-5mm)", "Heavy (>5mm)"]:
        sub = pooled_rain[pooled_rain["rain_category"] == cat]
        distribution_rows.append(summarize_series(sub["surge_pct_cpm"], f"Pooled ({cat})", "Rain Intensity", len(sub)))

    # C. Per-Station Distributions
    for st in STATIONS:
        sub = pooled_rain[pooled_rain["station_id"] == st["id"]]
        distribution_rows.append(summarize_series(sub["surge_pct_cpm"], st["name"], "All Rain", len(sub)))
        # Also heavy rain per station
        sub_heavy = sub[sub["rain_category"] == "Heavy (>5mm)"]
        if len(sub_heavy) > 0:
            distribution_rows.append(summarize_series(sub_heavy["surge_pct_cpm"], f"{st['name']} (Heavy >5mm)", "Heavy Rain", len(sub_heavy)))

    df_dist = pd.DataFrame(distribution_rows)
    out_csv = Path("data/processed/rain_washout_surge_distribution.csv")
    df_dist.to_csv(out_csv, index=False)
    print(f"\nSaved distribution summary to {out_csv}")
    print(df_dist.to_string(index=False))

    # 3. Publication Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Histogram of Pooled Surge %
    ax = axes[0, 0]
    p_99 = pooled_rain["surge_pct_cpm"].quantile(0.99)
    filtered = pooled_rain[pooled_rain["surge_pct_cpm"] <= p_99]["surge_pct_cpm"]
    ax.hist(filtered, bins=60, color="#1f4e79", edgecolor="black", alpha=0.75, density=True)
    ax.axvline(0, color="gray", linestyle="--")
    ax.axvline(pooled_rain["surge_pct_cpm"].median(), color="#d62728", linestyle="-", linewidth=2, label=f"Median: {pooled_rain['surge_pct_cpm'].median():.1f}%")
    ax.axvline(pooled_rain["surge_pct_cpm"].quantile(0.90), color="#ff7f0e", linestyle="--", linewidth=1.8, label=f"90th %ile: {pooled_rain['surge_pct_cpm'].quantile(0.90):.1f}%")
    ax.axvline(pooled_rain["surge_pct_cpm"].quantile(0.99), color="#9467bd", linestyle=":", linewidth=2, label=f"99th %ile: {pooled_rain['surge_pct_cpm'].quantile(0.99):.1f}%")
    ax.set_title("A. Distribution of Gross Count Rate Surges\n(All 15,606 Rain Hours, 2017-2025)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Gross CPM Surge Above Station Dry Baseline (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    # Panel B: Surge % by Rain Intensity (Boxplot)
    ax = axes[0, 1]
    box_data = [
        pooled_rain[pooled_rain["rain_category"] == "Light (<=1mm)"]["surge_pct_cpm"].dropna(),
        pooled_rain[pooled_rain["rain_category"] == "Moderate (1-5mm)"]["surge_pct_cpm"].dropna(),
        pooled_rain[pooled_rain["rain_category"] == "Heavy (>5mm)"]["surge_pct_cpm"].dropna(),
    ]
    bp = ax.boxplot(box_data, tick_labels=["Light\n(<=1 mm)", "Moderate\n(1-5 mm)", "Heavy\n(>5 mm)"], patch_artist=True, showfliers=False)
    colors = ["#a6cee3", "#1f78b4", "#b2df8a"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_title("B. Surge Magnitude by Rain Intensity\n(Boxplot Showing Median, IQR, Whiskers)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gross CPM Surge (%)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Panel C: Cumulative Distribution Function (CDF) per station
    ax = axes[1, 0]
    for st in STATIONS:
        sub = pooled_rain[pooled_rain["station_id"] == st["id"]]["surge_pct_cpm"].sort_values()
        y = np.linspace(0, 1, len(sub))
        ax.plot(sub, y, label=st["name"], color=st["color"], linewidth=2.0)
    ax.set_xlim(-15, 120)
    ax.set_title("C. Empirical CDF of Surge % Across Stations\n(X-axis clipped to 120% for resolution)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Gross CPM Surge (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Cumulative Probability", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)

    # Panel D: Scatter Plot Rain Depth vs. Gross CPM Surge
    ax = axes[1, 1]
    sample = pooled_rain.sample(n=min(3000, len(pooled_rain)), random_state=42)
    ax.scatter(sample["precip_1h_mm"], sample["surge_pct_cpm"], alpha=0.25, s=15, color="#1f77b4")
    ax.set_title("D. Rainfall Intensity vs. Count Rate Surge\n(Random Sample of 3,000 Rain Hours)", fontsize=11, fontweight="bold")
    ax.set_xlabel("1-Hour Precipitation Depth (mm)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Gross CPM Surge (%)", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 35)
    ax.set_ylim(-20, 200)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "rain_washout_surge_distribution.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved washout distribution figure to {fig_path}")


if __name__ == "__main__":
    analyze_washout_distributions()
