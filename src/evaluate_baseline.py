"""
Phase 2: Evaluates fixed-threshold baseline alarm systems on RadNet data (2017-2025).
Operates strictly on the continuous 78,888-hour calendar grid so that rolling windows
and lag/diff operations represent exact calendar time, not row indices.

Implements explicit threshold rules recorded in DECISIONS.md:
1. Global Gross CPM Sigma: Threshold = mu_dry + k * sigma_dry (k = 3.0, 4.0, 5.0)
2. Rolling 7-Day Mean + Global Sigma: Threshold(t) = rolling_mean_168h(t) + k * sigma_dry (k = 3.0, 4.0, 5.0)
   (NOTE: distinct from the Phase 5 "Rolling 7d Local Z-Score" rule
    Z = (gross - mu_168)/sigma_168 >= k used in train_and_evaluate_rigorous.py.
    BUG-12 fix (2026-10-06): the two formulas were previously both called
    "Rolling 7-day k-sigma" despite being different rules.)
3. Global Dose Rate Sigma: Threshold = mu_dose_dry + k * sigma_dose_dry (k = 3.0, 4.0, 5.0)

Quantifies:
- Alarm hours per station-year
- Discrete alarm episodes per station-year (clustered on continuous calendar grid)
- Rain coincidence rate (P_1h > 0, P_3h > 0, P_6h > 0)
- Dry alarm rate (P_24h == 0)
- Mean episode duration (hours)

Saves data/processed/baseline_threshold_evaluation.csv and generates publication figures.
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

HOURS_PER_YEAR = 8760.0


def evaluate_thresholds():
    all_results = []
    station_dfs = {}

    # 1. Load data, compute baselines and rolling features on continuous calendar grid
    for st in STATIONS:
        csv_file = Path(f"data/processed/merged_{st['id']}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        df["dt"] = pd.to_datetime(df["utc_hour"])
        # Ensure strict hourly chronological order
        df = df.sort_values("dt").reset_index(drop=True)

        # Compute calendar-time rolling sums on full 78,888-hour grid BEFORE any filtering
        df["precip_3h"] = df["precip_1h_mm"].rolling(3, min_periods=1).sum()
        df["precip_6h"] = df["precip_1h_mm"].rolling(6, min_periods=1).sum()
        df["precip_24h"] = df["precip_1h_mm"].rolling(24, min_periods=12).sum()

        # Valid observation mask
        valid_obs_mask = df["has_radnet_obs"] & df["has_weather_obs"] & df["rad_complete_channels"]

        # Dry mask on continuous grid (24h rain == 0 and current rain == 0)
        dry_mask = (df["precip_24h"] == 0.0) & (df["precip_1h_mm"] == 0.0) & valid_obs_mask
        dry_df = df[dry_mask]

        mu_cpm = dry_df["gross_cpm"].mean()
        sigma_cpm = dry_df["gross_cpm"].std()

        mu_dose = dry_df["dose_rate_nsvh"].mean()
        sigma_dose = dry_df["dose_rate_nsvh"].std()

        # Rolling 168h (7-day calendar window) on continuous grid
        df["rolling_cpm_168h"] = df["gross_cpm"].rolling(168, min_periods=48).mean()
        # Fill remaining gaps with station dry mean
        df["rolling_cpm_168h"] = df["rolling_cpm_168h"].fillna(mu_cpm)

        valid_count = int(valid_obs_mask.sum())
        years_covered = valid_count / HOURS_PER_YEAR

        station_dfs[st["id"]] = {
            "df": df,
            "valid_obs_mask": valid_obs_mask,
            "name": st["name"],
            "color": st["color"],
            "mu_cpm": mu_cpm,
            "sigma_cpm": sigma_cpm,
            "mu_dose": mu_dose,
            "sigma_dose": sigma_dose,
            "valid_count": valid_count,
            "years_covered": years_covered,
        }

    # 2. Define Rules to evaluate
    rules = [
        {"rule_family": "Global Gross CPM Sigma", "param": 3.0, "code": "global_cpm_3s"},
        {"rule_family": "Global Gross CPM Sigma", "param": 4.0, "code": "global_cpm_4s"},
        {"rule_family": "Global Gross CPM Sigma", "param": 5.0, "code": "global_cpm_5s"},
        {"rule_family": "Rolling 7-Day Mean + Global Sigma", "param": 3.0, "code": "rolling_cpm_3s"},
        {"rule_family": "Rolling 7-Day Mean + Global Sigma", "param": 4.0, "code": "rolling_cpm_4s"},
        {"rule_family": "Rolling 7-Day Mean + Global Sigma", "param": 5.0, "code": "rolling_cpm_5s"},
        {"rule_family": "Global Dose Rate Sigma", "param": 3.0, "code": "dose_rate_3s"},
        {"rule_family": "Global Dose Rate Sigma", "param": 4.0, "code": "dose_rate_4s"},
        {"rule_family": "Global Dose Rate Sigma", "param": 5.0, "code": "dose_rate_5s"},
    ]

    for rule in rules:
        family = rule["rule_family"]
        k = rule["param"]
        code = rule["code"]

        for st in STATIONS:
            s_info = station_dfs[st["id"]]
            df = s_info["df"]
            valid_mask = s_info["valid_obs_mask"]
            n_years = s_info["years_covered"]

            # Compute threshold and alarm on continuous grid
            if family == "Global Gross CPM Sigma":
                threshold = s_info["mu_cpm"] + k * s_info["sigma_cpm"]
                is_alarm = valid_mask & (df["gross_cpm"] > threshold)
                thresh_desc = f"{threshold:.1f} CPM ({k}s over {s_info['mu_cpm']:.1f})"
            elif family == "Rolling 7-Day Mean + Global Sigma":
                threshold = df["rolling_cpm_168h"] + k * s_info["sigma_cpm"]
                is_alarm = valid_mask & (df["gross_cpm"] > threshold)
                thresh_desc = f"Rolling 168h + {k}s ({k*s_info['sigma_cpm']:.1f} CPM)"
            elif family == "Global Dose Rate Sigma":
                threshold = s_info["mu_dose"] + k * s_info["sigma_dose"]
                is_alarm = valid_mask & (df["dose_rate_nsvh"] > threshold)
                thresh_desc = f"{threshold:.1f} nSv/h ({k}s over {s_info['mu_dose']:.1f})"

            # Discrete episode clustering on continuous calendar grid:
            # An episode starts when is_alarm is True at t, and was False (or gap) at t-1 hour
            prev_alarm = is_alarm.shift(1, fill_value=False)
            episode_starts = is_alarm & (~prev_alarm)
            alarm_episodes = int(episode_starts.sum())
            alarm_hours = int(is_alarm.sum())

            alarm_hours_per_year = round(alarm_hours / n_years, 2)
            episodes_per_year = round(alarm_episodes / n_years, 2)

            if alarm_hours > 0:
                alarm_df = df[is_alarm]
                rain_1h_frac = round((alarm_df["precip_1h_mm"] > 0.0).mean() * 100.0, 2)
                rain_3h_frac = round((alarm_df["precip_3h"] > 0.0).mean() * 100.0, 2)
                rain_6h_frac = round((alarm_df["precip_6h"] > 0.0).mean() * 100.0, 2)
                dry_24h_frac = round((alarm_df["precip_24h"] == 0.0).mean() * 100.0, 2)
                mean_duration = round(alarm_hours / alarm_episodes, 2) if alarm_episodes > 0 else 0.0
            else:
                rain_1h_frac = 0.0
                rain_3h_frac = 0.0
                rain_6h_frac = 0.0
                dry_24h_frac = 0.0
                mean_duration = 0.0

            all_results.append({
                "Rule_Code": code,
                "Rule_Family": family,
                "Multiplier_k": k,
                "Station": st["name"],
                "Station_ID": st["id"],
                "Years_Evaluated": round(n_years, 2),
                "Threshold_Description": thresh_desc,
                "Total_Alarm_Hours": alarm_hours,
                "Alarm_Hours_Per_Year": alarm_hours_per_year,
                "Total_Alarm_Episodes": alarm_episodes,
                "Episodes_Per_Year": episodes_per_year,
                "Coincident_Rain_1h_Pct": rain_1h_frac,
                "Coincident_Rain_3h_Pct": rain_3h_frac,
                "Coincident_Rain_6h_Pct": rain_6h_frac,
                "Dry_Weather_Alarms_Pct": dry_24h_frac,
                "Mean_Episode_Hours": mean_duration,
            })

    res_df = pd.DataFrame(all_results)
    out_csv = Path("data/processed/baseline_threshold_evaluation.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"Saved continuous calendar baseline evaluation table to {out_csv}")

    # 3. Print high-level summary table across stations
    print("\n=== Baseline Fixed-Threshold Performance (Continuous Calendar Grid) ===")
    summary_view = res_df[["Rule_Family", "Multiplier_k", "Station", "Episodes_Per_Year", "Coincident_Rain_3h_Pct", "Coincident_Rain_6h_Pct", "Dry_Weather_Alarms_Pct"]]
    print(summary_view.to_string(index=False))

    # 4. Publication Figures
    # Figure 1: Episodes per Year by Rule and Station (Bar Chart)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sub_3s = res_df[res_df["Multiplier_k"] == 3.0]
    piv_3s = sub_3s.pivot(index="Station", columns="Rule_Family", values="Episodes_Per_Year")
    piv_3s.plot(kind="bar", ax=axes[0], colormap="viridis", width=0.8, edgecolor="black")
    axes[0].set_title("A. Annual Alarm Episodes at 3-Sigma Threshold\n(Continuous Calendar Grid)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Discrete Alarm Episodes / Year", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].tick_params(axis="x", rotation=30)

    piv_rain = sub_3s.pivot(index="Station", columns="Rule_Family", values="Coincident_Rain_3h_Pct")
    piv_rain.plot(kind="bar", ax=axes[1], colormap="plasma", width=0.8, edgecolor="black")
    axes[1].set_title("B. Fraction of Alarms Coinciding with Rain (3h Window)\n(Empirical Demonstration of False Alarms Caused by Washout)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Rain-Coincident Alarms (%)", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig1_path = OUTPUT_DIR / "baseline_alarm_rate_by_threshold.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved figure 1 to {fig1_path}")

    # Figure 2: Detailed Trade-off: Episodes/Year vs Rain Coincidence as k increases
    fig, ax = plt.subplots(figsize=(10, 6))
    for st in STATIONS:
        st_data = res_df[(res_df["Station_ID"] == st["id"]) & (res_df["Rule_Family"] == "Global Gross CPM Sigma")].sort_values("Multiplier_k")
        ax.plot(st_data["Multiplier_k"], st_data["Episodes_Per_Year"], marker="o", linewidth=2.2, color=st["color"], label=f"{st['name']}")
        for _, row in st_data.iterrows():
            ax.annotate(f"{row['Coincident_Rain_3h_Pct']:.0f}% rain", (row["Multiplier_k"], row["Episodes_Per_Year"] + 1.5), fontsize=8, ha="center", color=st["color"], fontweight="bold")

    ax.set_title("Annual False Alarm Rate vs. Sigma Multiplier (Global Gross CPM Rule)\nLabels indicate % of alarms coinciding with rain (3-hour storm window)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Threshold Sigma Multiplier (k in mu + k*sigma)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Discrete Alarm Episodes / Station-Year", fontsize=10, fontweight="bold")
    ax.set_xticks([3.0, 4.0, 5.0])
    ax.set_xticklabels(["3-Sigma", "4-Sigma", "5-Sigma"])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    fig2_path = OUTPUT_DIR / "baseline_alarm_rain_coincidence.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved figure 2 to {fig2_path}")


if __name__ == "__main__":
    evaluate_thresholds()
