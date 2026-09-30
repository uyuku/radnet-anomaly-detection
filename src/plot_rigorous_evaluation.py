"""
Publication-Quality Plotting Script for Rigorous Evaluation (Second Revision).

Generates 5 comprehensive figures:
1. reports/figures/rigorous_matched_roc_curves.png:
   - Continuous test ROC curves for Baselines and all ML Tiers across tau in [0.01, 0.999] and k in [0.5, 5.0].
   - Direct visualization of matched detection at ~92%: Tier 1 (162.99 FA/yr) vs Tier 1b (67.08 FA/yr, 58.8% reduction),
     Tier 2 (9.62 FA/yr), and Tier 3 (3.73 FA/yr, 98.8% reduction over baseline).
2. reports/figures/rigorous_ablation_seeds.png:
   - Four-tier ablation bar chart + strip plot across 20 random seeds with mean +/- std error bars.
3. reports/figures/rigorous_detection_deadlines.png:
   - Grouped bar chart showing early detection (<=6h, <=12h, <=24h, total) and stratification across Dry vs Rain and Standard vs Stress.
4. reports/figures/rigorous_template_sensitivity.png:
   - Two-panel figure:
     (a) Injected photopeak scaling (+/-10%, +/-20%) showing moderate-to-strong sensitivity.
     (b) Instrumental gain drift (+/-5%, +/-10%) showing both detection and clean false alarm impact.
5. reports/figures/rigorous_loso_multistation.png:
   - Multi-station leave-one-station-out (LOSO) comparison across San Diego, Birmingham, and Dallas.
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def plot_matched_roc_curves():
    """Plots full continuous ROC curves comparing Baselines and all ML Tiers on test set."""
    base_file = Path("data/processed/rigorous_baseline_continuous_roc.csv")
    ml_file = Path("data/processed/rigorous_continuous_roc_all_tiers.csv")
    matched_file = Path("data/processed/rigorous_matched_detection_comparison.csv")

    if not base_file.exists() or not ml_file.exists():
        print("Missing ROC files, skipping matched ROC plot.")
        return

    base_df = pd.read_csv(base_file)
    ml_df = pd.read_csv(ml_file)

    fig, ax = plt.subplots(figsize=(10.5, 7))

    # Baselines
    roll_df = base_df[base_df["baseline_type"] == "rolling_7d_local_z"].sort_values("detection_rate_pct")
    glob_df = base_df[base_df["baseline_type"] == "global_dry_sigma"].sort_values("detection_rate_pct")

    ax.plot(roll_df["detection_rate_pct"], roll_df["false_alarms_per_year"],
            label="Baseline: Rolling 7d Local Z-Score (k swept 0.5 to 5.0)",
            color="#d9534f", linestyle="--", linewidth=2.2)
    ax.plot(glob_df["detection_rate_pct"], glob_df["false_alarms_per_year"],
            label="Baseline: Global Dry Sigma (k swept 0.5 to 5.0)",
            color="#f0ad4e", linestyle=":", linewidth=2.0)

    # ML Tiers continuous curves
    tier_styles = {
        "Tier 1 (Gross)": {"col": "#7f8c8d", "ls": "-", "lw": 1.8, "name": "Tier 1: Gross Only"},
        "Tier 1b (Gross+Weather)": {"col": "#e67e22", "ls": "-", "lw": 2.0, "name": "Tier 1b: Gross + Weather"},
        "Tier 2 (Spectral)": {"col": "#2980b9", "ls": "-", "lw": 2.2, "name": "Tier 2: Spectral Only"},
        "Tier 3 (Weather Fusion)": {"col": "#27ae60", "ls": "-", "lw": 2.6, "name": "Tier 3: Weather Fusion (GBDT)"},
        "Tier 3 MLP": {"col": "#8e44ad", "ls": "--", "lw": 1.8, "name": "Tier 3: MLP Neural Net"},
    }

    for t_name, sty in tier_styles.items():
        sub = ml_df[ml_df["tier_name"] == t_name].sort_values("detection_rate_pct")
        if not sub.empty:
            ax.plot(sub["detection_rate_pct"], sub["false_alarms_per_year"],
                    label=sty["name"], color=sty["col"], linestyle=sty["ls"], linewidth=sty["lw"])

    # Highlight operating points at ~92% detection
    if matched_file.exists():
        m_df = pd.read_csv(matched_file)
        row_92 = m_df[m_df["matched_detection_target"] == "92%"].iloc[0]

        # Annotate matched operating points
        pts = [
            ("Baseline (k=0.75)", row_92["rolling_7d_realized_det"], row_92["rolling_7d_fa"], "#d9534f", "s"),
            ("Tier 1 (Gross)", row_92["t1_gross_det"], row_92["t1_gross_fa"], "#7f8c8d", "s"),
            ("Tier 1b (Gross+Wx)", row_92["t1b_gross_weather_det"], row_92["t1b_gross_weather_fa"], "#e67e22", "D"),
            ("Tier 2 (Spectral)", row_92["t2_spectral_det"], row_92["t2_spectral_fa"], "#2980b9", "^"),
            ("Tier 3 (Wx Fusion)", row_92["t3_fusion_det"], row_92["t3_fusion_fa"], "#27ae60", "o"),
        ]

        for lbl, x_val, y_val, col, marker in pts:
            ax.scatter([x_val], [y_val], color=col, marker=marker, s=120, zorder=6, edgecolor="black", linewidth=1.0)
            ax.annotate(
                f"{lbl}\n{y_val:.1f} FA/yr",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(10, -5 if y_val < 50 else 5),
                fontsize=8.5,
                color=col,
                fontweight="bold"
            )

    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlim(50, 102)
    ax.set_ylim(-0.2, 600)
    ax.set_xlabel("Test Event Detection Probability (%) on Synthetic Plumes (2023–2025)")
    ax.set_ylabel("Clean Operational False Alarms per Station-Year (Symlog Scale)")
    ax.set_title("Continuous Operational Detection vs. False Alarm Curves (Test Split 2023–2025)", fontweight="bold", pad=12)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="upper left", framealpha=0.92, fontsize=9.5)

    out_file = Path("reports/figures/rigorous_matched_roc_curves.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_ablation_seeds():
    """Bar chart + strip plot showing 4-tier ablation across 20 random seeds."""
    seed_file = Path("data/processed/rigorous_seed_level_results.csv")
    agg_file = Path("data/processed/rigorous_benchmark_summary.csv")
    if not seed_file.exists() or not agg_file.exists():
        return

    seed_df = pd.read_csv(seed_file)
    agg_df = pd.read_csv(agg_file)

    sub_agg = agg_df[agg_df["target_detection_pct"] == 95.0].copy()
    tier_order = ["Tier 1 (Gross)", "Tier 1b (Gross+Weather)", "Tier 2 (Spectral)", "Tier 3 (Weather Fusion)", "Tier 3 MLP"]
    sub_agg["tier_name"] = pd.Categorical(sub_agg["tier_name"], categories=tier_order, ordered=True)
    sub_agg = sub_agg.sort_values("tier_name").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, 6))
    x = np.arange(len(sub_agg))

    colors = ["#7f8c8d", "#e67e22", "#2980b9", "#27ae60", "#8e44ad"]
    bars = ax.bar(x, sub_agg["fa_per_year_mean"], yerr=sub_agg["fa_per_year_std"],
                  capsize=6, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8, zorder=2)

    # Add individual seed points
    for i, t_name in enumerate(tier_order):
        seed_pts = seed_df[(seed_df["tier_name"] == t_name) & (seed_df["target_detection"] == 95.0)]["clean_false_alarms_per_year"].values
        jitter = np.random.uniform(-0.12, 0.12, size=len(seed_pts))
        ax.scatter(x[i] + jitter, seed_pts, color="black", alpha=0.45, s=25, zorder=3)

    labels = [
        "Tier 1\n(Gross Only)",
        "Tier 1b\n(Gross+Weather)",
        "Tier 2\n(Spectral Ratios)",
        "Tier 3\n(Weather Fusion)",
        "Tier 3\n(MLP Neural Net)"
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Clean Operational False Alarms per Station-Year (Symlog Scale)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-0.2, 500)
    ax.set_title("Ablation Hierarchy at Matched Target Detection Across 20 Random Seeds", fontweight="bold", pad=12)
    ax.grid(True, which="both", axis="y", alpha=0.35)

    for bar, r in zip(bars, sub_agg.itertuples()):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h * 1.35 if h > 5 else h + 1.2,
                f"{r.fa_per_year_mean:.1f} ± {r.fa_per_year_std:.1f}\n(Realized Det: {r.det_total_mean:.1f}%)",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    out_file = Path("reports/figures/rigorous_ablation_seeds.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_detection_deadlines():
    """Grouped bar chart showing early detection deadlines and environmental stratification."""
    strat_file = Path("data/processed/rigorous_stratified_deadlines.csv")
    agg_file = Path("data/processed/rigorous_benchmark_summary.csv")
    if not strat_file.exists():
        return

    strat_df = pd.read_csv(strat_file)
    t3_strat = strat_df[strat_df["tier_name"] == "Tier 3 (Weather Fusion)"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Deadlines for Tier 2 vs Tier 3 on All Injections
    tiers = ["Tier 1b (Gross+Weather)", "Tier 2 (Spectral)", "Tier 3 (Weather Fusion)", "Tier 3 MLP"]
    labels = ["Tier 1b\n(Gross+Wx)", "Tier 2\n(Spectral)", "Tier 3\n(Wx Fusion)", "Tier 3\n(MLP)"]

    agg_df = pd.read_csv(agg_file)
    sub = agg_df[agg_df["target_detection_pct"] == 95.0].set_index("tier_name")

    x = np.arange(len(tiers))
    width = 0.20

    d6 = [sub.loc[t, "det_6h_mean"] for t in tiers]
    d12 = [sub.loc[t, "det_12h_mean"] for t in tiers]
    d24 = [sub.loc[t, "det_24h_mean"] for t in tiers]
    dtot = [sub.loc[t, "det_total_mean"] for t in tiers]

    ax1.bar(x - 1.5*width, d6, width, label="≤ 6h Early Detection", color="#1976d2", alpha=0.9)
    ax1.bar(x - 0.5*width, d12, width, label="≤ 12h Detection", color="#388e3c", alpha=0.9)
    ax1.bar(x + 0.5*width, d24, width, label="≤ 24h Detection", color="#f57c00", alpha=0.9)
    ax1.bar(x + 1.5*width, dtot, width, label="Total Window", color="#7b1fa2", alpha=0.9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Detection Probability (%) Under Net Alarm Criterion")
    ax1.set_title("(a) Detection Deadlines Across Model Tiers", fontweight="bold", pad=10)
    ax1.set_ylim(0, 115)
    ax1.grid(True, axis="y", alpha=0.35)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # Panel 2: Stratified Deadlines for Tier 3 across Dry vs Rain and Standard vs Stress
    strata_keys = ["Strictly Dry Environments", "Rain-Coincident Environments", "Standard Plume Regime", "Stress Plume Regime"]
    strata_labels = ["Strictly\nDry", "Rain-\nCoincident", "Standard\nPlume", "Stress\nPlume"]
    sub_t3 = t3_strat[t3_strat["stratum"].isin(strata_keys)].set_index("stratum").loc[strata_keys]

    xs = np.arange(len(strata_keys))
    ax2.bar(xs - 1.5*width, sub_t3["detection_le_6h_pct"], width, label="≤ 6h", color="#1976d2", alpha=0.9)
    ax2.bar(xs - 0.5*width, sub_t3["detection_le_12h_pct"], width, label="≤ 12h", color="#388e3c", alpha=0.9)
    ax2.bar(xs + 0.5*width, sub_t3["detection_le_24h_pct"], width, label="≤ 24h", color="#f57c00", alpha=0.9)
    ax2.bar(xs + 1.5*width, sub_t3["detection_total_pct"], width, label="Total", color="#7b1fa2", alpha=0.9)

    ax2.set_xticks(xs)
    ax2.set_xticklabels(strata_labels, fontsize=10)
    ax2.set_ylabel("Detection Probability (%) Under Net Alarm Criterion")
    ax2.set_title("(b) Tier 3 Deadlines Stratified by Regime", fontweight="bold", pad=10)
    ax2.set_ylim(0, 115)
    ax2.grid(True, axis="y", alpha=0.35)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)

    out_file = Path("reports/figures/rigorous_detection_deadlines.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_template_sensitivity():
    """Two-panel plot showing injected template sensitivity and instrumental gain drift."""
    sens_file = Path("data/processed/rigorous_template_sensitivity.csv")
    gain_file = Path("data/processed/rigorous_gain_drift_evaluation.csv")
    if not sens_file.exists() or not gain_file.exists():
        return

    sens_df = pd.read_csv(sens_file)
    gain_df = pd.read_csv(gain_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Injected Photopeak Scaling
    x_pct = (sens_df["perturbation_factor"] - 1.0) * 100.0
    y_det = sens_df["test_detection_rate_pct_mean"] if "test_detection_rate_pct_mean" in sens_df.columns else sens_df["test_detection_rate_pct"]
    y_err = sens_df["test_detection_rate_pct_std"] if "test_detection_rate_pct_std" in sens_df.columns else None

    if y_err is not None:
        ax1.errorbar(x_pct, y_det, yerr=y_err, fmt="-o", color="#27ae60", linewidth=2.5, markersize=8, capsize=5)
    else:
        ax1.plot(x_pct, y_det, marker="o", color="#27ae60", linewidth=2.5, markersize=8)

    for px, py in zip(x_pct, y_det):
        ax1.annotate(f"{py:.1f}%", (px, py), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontweight="bold", fontsize=9.5)

    ax1.set_xlabel("Injected Photopeak Excess Scaling (%)")
    ax1.set_ylabel("Test Event Detection Rate (%) (5-Seed Mean ± Std)")
    ax1.set_title("(a) Sensitivity to Injected Photopeak Branching", fontweight="bold", pad=10)
    ax1.set_ylim(65, 105)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.grid(True, alpha=0.35)

    # Panel B: Instrumental Gain Drift (Dual Axis: Detection & Clean FA)
    gx = gain_df["gain_drift_shift"] * 100.0
    g_det = gain_df["test_detection_rate_pct_mean"] if "test_detection_rate_pct_mean" in gain_df.columns else gain_df["test_detection_rate_pct"]
    g_det_err = gain_df["test_detection_rate_pct_std"] if "test_detection_rate_pct_std" in gain_df.columns else None

    g_fa = gain_df["clean_false_alarms_per_year_mean"] if "clean_false_alarms_per_year_mean" in gain_df.columns else gain_df["clean_false_alarms_per_year"]
    g_fa_err = gain_df["clean_false_alarms_per_year_std"] if "clean_false_alarms_per_year_std" in gain_df.columns else None

    color_det = "#2980b9"
    color_fa = "#d9534f"

    if g_det_err is not None:
        ax2.errorbar(gx, g_det, yerr=g_det_err, fmt="-s", color=color_det, linewidth=2.2, markersize=7, capsize=4, label="Detection Rate (%)")
    else:
        ax2.plot(gx, g_det, marker="s", color=color_det, linewidth=2.2, markersize=7, label="Detection Rate (%)")

    ax2.set_xlabel("Instrumental Energy Calibration Drift (%)")
    ax2.set_ylabel("Test Detection Rate (%) (5-Seed Mean ± Std)", color=color_det)
    ax2.tick_params(axis="y", labelcolor=color_det)
    ax2.set_ylim(65, 105)
    ax2.grid(True, alpha=0.35)

    ax2_fa = ax2.twinx()
    if g_fa_err is not None:
        ax2_fa.errorbar(gx, g_fa, yerr=g_fa_err, fmt="--^", color=color_fa, linewidth=2.2, markersize=7, capsize=4, label="Clean False Alarms")
    else:
        ax2_fa.plot(gx, g_fa, marker="^", color=color_fa, linewidth=2.2, linestyle="--", markersize=7, label="Clean False Alarms")

    ax2_fa.set_ylabel("Clean False Alarms / Station-Year (5-Seed Mean ± Std)", color=color_fa)
    ax2_fa.tick_params(axis="y", labelcolor=color_fa)
    ax2_fa.set_ylim(-0.5, max(g_fa.max() * 1.3, 10.0))

    ax2.set_title("(b) Instrumental Gain Drift Across All Data", fontweight="bold", pad=10)

    out_file = Path("reports/figures/rigorous_template_sensitivity.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_loso_multistation():
    """Plots multi-station LOSO performance comparing San Diego, Birmingham, and Dallas."""
    loso_file = Path("data/processed/rigorous_loso_summary.csv")
    if not loso_file.exists():
        return
    loso_df = pd.read_csv(loso_file)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(loso_df))
    width = 0.35

    color_det = "#2980b9"
    color_fa = "#d9534f"

    bars1 = ax1.bar(x - width/2, loso_df["realized_detection_rate_pct"], width, label="Realized Detection Rate (%)", color=color_det, alpha=0.88, edgecolor="black")
    ax1.set_ylabel("Realized Detection Rate (%)", color=color_det)
    ax1.tick_params(axis="y", labelcolor=color_det)
    ax1.set_ylim(0, 115)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, loso_df["clean_false_alarms_per_year"], width, label="Clean False Alarms / Year", color=color_fa, alpha=0.88, edgecolor="black")
    ax2.set_ylabel("Clean False Alarms / Station-Year", color=color_fa)
    ax2.tick_params(axis="y", labelcolor=color_fa)
    ax2.set_ylim(-0.2, max(loso_df["clean_false_alarms_per_year"].max() * 1.4, 6.0))

    labels = [
        "San Diego, CA\n(Marine, 0 Washout Hrs)",
        "Birmingham, AL\n(High Rain, Active Washouts)",
        "Dallas, TX\n(Convective, Active Washouts)",
    ]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_title("Multi-Station Leave-One-Station-Out (LOSO) Validation", fontweight="bold", pad=12)
    ax1.grid(True, axis="y", alpha=0.35)

    # Annotate bars
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2., h + 2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=color_det)
    for b, r in zip(bars2, loso_df.itertuples()):
        h = b.get_height()
        annot = f"{h:.1f}" if h > 0 else f"0.0 (≤{r.rule_of_three_95ci_upper_fa})"
        ax2.text(b.get_x() + b.get_width()/2., h + 0.2, annot, ha="center", va="bottom", fontsize=9, fontweight="bold", color=color_fa)

    out_file = Path("reports/figures/rigorous_loso_multistation.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main():
    print("Generating Rigorous Publication Figures...")
    plot_matched_roc_curves()
    plot_ablation_seeds()
    plot_detection_deadlines()
    plot_template_sensitivity()
    plot_loso_multistation()
    print("All Rigorous Publication Figures generated successfully.")


if __name__ == "__main__":
    main()
