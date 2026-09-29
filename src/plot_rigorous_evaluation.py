"""
Rigorous Evaluation Plotting Script.

Generates 4 publication figures:
1. reports/figures/rigorous_matched_roc_curves.png:
   - Full continuous ROC trade-off curves for Baselines and ML Tiers.
2. reports/figures/rigorous_ablation_seeds.png:
   - 4-tier ablation comparison across 5 random seeds with error bars (mean +/- std).
3. reports/figures/rigorous_detection_deadlines.png:
   - Detection probability within operational deadlines (<=6h, <=12h, <=24h).
4. reports/figures/rigorous_template_sensitivity.png:
   - Spectral template perturbation sweep (+/-10%, +/-20%).
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
    """Plots full continuous ROC curves comparing Baselines and ML Tiers."""
    base_file = Path("data/processed/rigorous_baseline_continuous_roc.csv")
    agg_file = Path("data/processed/rigorous_benchmark_summary.csv")
    if not base_file.exists() or not agg_file.exists():
        return

    base_df = pd.read_csv(base_file)
    agg_df = pd.read_csv(agg_file)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Baselines
    roll_df = base_df[base_df["baseline_type"] == "rolling_7d"].sort_values("detection_rate_pct")
    glob_df = base_df[base_df["baseline_type"] == "global_dry"].sort_values("detection_rate_pct")

    ax.plot(roll_df["detection_rate_pct"], roll_df["false_alarms_per_year"],
            label="Baseline: Rolling 7d Sigma (k swept 0.5 to 5.0)",
            color="#d9534f", linestyle="-", linewidth=2.2)
    ax.plot(glob_df["detection_rate_pct"], glob_df["false_alarms_per_year"],
            label="Baseline: Global Dry Sigma (k swept 0.5 to 5.0)",
            color="#f0ad4e", linestyle="--", linewidth=2.2)

    # ML Tiers at operating points
    tier_markers = {
        "Tier 1 (Gross)": {"col": "#808080", "sym": "s", "name": "Tier 1: Gross Only"},
        "Tier 1b (Gross+Weather)": {"col": "#e67e22", "sym": "D", "name": "Tier 1b: Gross + Weather"},
        "Tier 2 (Spectral)": {"col": "#337ab7", "sym": "^", "name": "Tier 2: Spectral Ratios"},
        "Tier 3 (Weather Fusion)": {"col": "#2e7d32", "sym": "o", "name": "Tier 3: Weather Fusion"},
        "Tier 3 MLP": {"col": "#7b1fa2", "sym": "*", "name": "Tier 3: MLP Neural Net"},
    }

    for t_id, cfg in tier_markers.items():
        sub = agg_df[agg_df["tier_name"] == t_id]
        ax.scatter(sub["det_total_mean"], sub["fa_per_year_mean"],
                   color=cfg["col"], marker=cfg["sym"], s=100, zorder=5, label=cfg["name"])

        # Error bars for std across seeds
        ax.errorbar(sub["det_total_mean"], sub["fa_per_year_mean"],
                    yerr=sub["fa_per_year_std"],
                    fmt="none", ecolor=cfg["col"], elinewidth=1.5, capsize=3, zorder=4)

        for _, r in sub.iterrows():
            ax.annotate(
                f"{r['fa_per_year_mean']:.1f}",
                (r["det_total_mean"], r["fa_per_year_mean"]),
                textcoords="offset points", xytext=(8, -3),
                fontsize=8.5, color=cfg["col"], fontweight="bold"
            )

    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlim(25, 102)
    ax.set_ylim(-0.2, 800)
    ax.set_xlabel("Event Detection Probability (%) on Synthetic Test Plumes")
    ax.set_ylabel("Clean Operational False Alarms per Station-Year (Symlog Scale)")
    ax.set_title("Operational Trade-off Curves at Matched Detection (Test Split 2023–2025)", fontweight="bold", pad=12)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="upper left", framealpha=0.9)

    out_file = Path("reports/figures/rigorous_matched_roc_curves.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_ablation_seeds():
    """Bar chart showing 4-tier ablation mean +/- std across 5 random seeds."""
    agg_file = Path("data/processed/rigorous_benchmark_summary.csv")
    if not agg_file.exists():
        return
    agg_df = pd.read_csv(agg_file)

    sub_95 = agg_df[agg_df["target_detection_pct"] == 95.0].copy()
    tier_order = ["Tier 1 (Gross)", "Tier 1b (Gross+Weather)", "Tier 2 (Spectral)", "Tier 3 (Weather Fusion)", "Tier 3 MLP"]
    sub_95["tier_name"] = pd.Categorical(sub_95["tier_name"], categories=tier_order, ordered=True)
    sub_95 = sub_95.sort_values("tier_name").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(sub_95))

    colors = ["#808080", "#e67e22", "#337ab7", "#2e7d32", "#7b1fa2"]
    bars = ax.bar(x, sub_95["fa_per_year_mean"], yerr=sub_95["fa_per_year_std"],
                  capsize=5, color=colors, alpha=0.88, edgecolor="black", linewidth=0.8)

    labels = ["Tier 1\n(Gross Only)", "Tier 1b\n(Gross+Weather)", "Tier 2\n(Spectral Ratios)", "Tier 3\n(Weather Fusion)", "Tier 3\n(MLP Neural Net)"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Clean False Alarms per Station-Year (Symlog Scale)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-0.2, 500)
    ax.set_title("Four-Tier Feature Ablation at ~95% Target Detection (Mean ± Std over 5 Seeds)", fontweight="bold", pad=12)
    ax.grid(True, which="both", axis="y", alpha=0.35)

    for bar, r in zip(bars, sub_95.itertuples()):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h * 1.25 if h > 5 else h + 1.2,
                f"{r.fa_per_year_mean:.1f} ± {r.fa_per_year_std:.1f}\n(Det: {r.det_total_mean:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    out_file = Path("reports/figures/rigorous_ablation_seeds.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_detection_deadlines():
    """Grouped bar chart showing detection probability within operational deadlines."""
    agg_file = Path("data/processed/rigorous_benchmark_summary.csv")
    if not agg_file.exists():
        return
    agg_df = pd.read_csv(agg_file)

    sub = agg_df[agg_df["target_detection_pct"] == 95.0].set_index("tier_name")
    tiers = ["Tier 1b (Gross+Weather)", "Tier 2 (Spectral)", "Tier 3 (Weather Fusion)", "Tier 3 MLP"]
    labels = ["Tier 1b (Gross+Wx)", "Tier 2 (Spectral)", "Tier 3 (Wx Fusion)", "Tier 3 (MLP)"]

    x = np.arange(len(tiers))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5.5))

    d6 = [sub.loc[t, "det_6h_mean"] for t in tiers]
    d12 = [sub.loc[t, "det_12h_mean"] for t in tiers]
    d24 = [sub.loc[t, "det_24h_mean"] for t in tiers]
    dtot = [sub.loc[t, "det_total_mean"] for t in tiers]

    b1 = ax.bar(x - 1.5*width, d6, width, label="≤ 6 Hours (During Active Passage)", color="#1976d2", alpha=0.9)
    b2 = ax.bar(x - 0.5*width, d12, width, label="≤ 12 Hours", color="#388e3c", alpha=0.9)
    b3 = ax.bar(x + 0.5*width, d24, width, label="≤ 24 Hours", color="#f57c00", alpha=0.9)
    b4 = ax.bar(x + 1.5*width, dtot, width, label="Total Window (Until Filter Swap)", color="#7b1fa2", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Detection Probability (%) Under Net Alarm Criterion")
    ax.set_title("Operational Detection Deadlines on Synthetic Test Injections", fontweight="bold", pad=12)
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="upper left", framealpha=0.9)

    out_file = Path("reports/figures/rigorous_detection_deadlines.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_template_sensitivity():
    """Plots detection rate under spectral template perturbation."""
    sens_file = Path("data/processed/rigorous_template_sensitivity.csv")
    if not sens_file.exists():
        return
    sens_df = pd.read_csv(sens_file)

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pct = (sens_df["perturbation_factor"] - 1.0) * 100.0
    y_det = sens_df["test_detection_rate_pct"]

    ax.plot(x_pct, y_det, marker="o", color="#2e7d32", linewidth=2.5, markersize=8)
    for px, py in zip(x_pct, y_det):
        ax.annotate(f"{py:.1f}%", (px, py), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontweight="bold", fontsize=9.5)

    ax.set_xlabel("Photopeak Channel Share Perturbation (%)")
    ax.set_ylabel("Test Event Detection Rate (%)")
    ax.set_title("Spectral Template Sensitivity Analysis (Tier 3 GBDT)", fontweight="bold", pad=12)
    ax.set_ylim(70, 102)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True, alpha=0.35)

    out_file = Path("reports/figures/rigorous_template_sensitivity.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main():
    print("Generating Rigorous Evaluation Figures...")
    plot_matched_roc_curves()
    plot_ablation_seeds()
    plot_detection_deadlines()
    plot_template_sensitivity()
    print("All Rigorous Evaluation Figures generated successfully.")


if __name__ == "__main__":
    main()
