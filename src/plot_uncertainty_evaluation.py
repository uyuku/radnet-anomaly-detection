"""
Bootstrap Uncertainty, Detection Delay, and Confusion Matrix Plotting Script.

Generates 3 publication-quality figures:
1. reports/figures/eval_bootstrap_uncertainty.png:
   - Forest plot showing Operational False Alarms/yr (with 95% CIs) and Realized Detection Rate (with 95% CIs).
2. reports/figures/eval_detection_delay_distributions.png:
   - Cumulative detection probability over time elapsed (CDF) and median delay across release scenarios.
3. reports/figures/eval_confusion_matrices.png:
   - 4-panel heatmap showing multi-class hourly confusion matrices (Baseline vs Tier 2 vs Tier 3 vs Tier 4).
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure publication-quality matplotlib style
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


def plot_bootstrap_uncertainty():
    """Forest plot of False Alarm Rate and Detection Rate with 95% Bootstrap CIs."""
    csv_path = Path("data/processed/eval_benchmark_uncertainty_summary.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Clean display names
    display_names = {
        "Baseline: Global Dry 3-sigma": "Baseline: Global Dry 3σ",
        "Baseline: Rolling 7d 3-sigma": "Baseline: Rolling 7d 3σ",
        "Baseline: Rolling 7d 4-sigma": "Baseline: Rolling 7d 4σ",
        "Baseline: Rolling 7d 5-sigma": "Baseline: Rolling 7d 5σ",
        "Tier 1: Gross Radiation GBDT (95% target)": "Tier 1: Gross GBDT (95% target)",
        "Tier 2: Radiation + Spectrometry GBDT (95% target)": "Tier 2: Spectral GBDT (95% target)",
        "Tier 3: Full Weather Fusion GBDT (95% target)": "Tier 3: Weather Fusion GBDT (95% target)",
        "Tier 3: Full Weather Fusion GBDT (90% target)": "Tier 3: Weather Fusion GBDT (90% target)",
        "Tier 4: Physics-Informed Nuclear GBDT (95% target)": "Tier 4: Nuclear Physics GBDT (95% target)",
        "Tier 4: Physics-Informed Nuclear GBDT (90% target)": "Tier 4: Nuclear Physics GBDT (90% target)",
        "Tier 3: Weather-Fused Neural Net (MLP) (95% target)": "Tier 3: Neural Net MLP (95% target)",
    }
    df["display_name"] = df["model_name"].map(lambda x: display_names.get(x, x))

    # Parse 95% CIs
    def parse_ci(val_str):
        clean = val_str.replace("[", "").replace("]", "").replace("%", "").replace("h", "")
        parts = clean.split(",")
        return float(parts[0].strip()), float(parts[1].strip())

    df["fa_ci_low"] = df["false_alarms_ci_95"].apply(lambda x: parse_ci(x)[0])
    df["fa_ci_high"] = df["false_alarms_ci_95"].apply(lambda x: parse_ci(x)[1])

    df["det_ci_low"] = df["detection_ci_95"].apply(lambda x: parse_ci(x)[0])
    df["det_ci_high"] = df["detection_ci_95"].apply(lambda x: parse_ci(x)[1])

    # Reverse order for plotting from top to bottom
    df_plot = df.iloc[::-1].reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 7.2), sharey=True)

    y_pos = np.arange(len(df_plot))

    # Color palette
    colors = []
    for m in df_plot["model_name"]:
        if "Rolling 7d 3-sigma" in m or "Global Dry 3-sigma" in m:
            colors.append("#d9534f")  # Red for primary baselines
        elif "Baseline" in m:
            colors.append("#f0ad4e")  # Orange for secondary baselines
        elif "Tier 1" in m:
            colors.append("#808080")  # Gray for Tier 1
        elif "Tier 2" in m:
            colors.append("#337ab7")  # Blue for Tier 2
        elif "Tier 3: Full Weather Fusion" in m:
            colors.append("#2e7d32")  # Deep Green for Tier 3 GBDT
        elif "Tier 4: Physics-Informed Nuclear" in m:
            colors.append("#0d47a1")  # Deep Navy for Tier 4 Nuclear GBDT
        else:
            colors.append("#7b1fa2")  # Purple for MLP

    # Panel 1: False Alarms per Station-Year
    x_fa = df_plot["false_alarms_per_station_year"]
    xerr_fa_low = np.maximum(0, x_fa - df_plot["fa_ci_low"])
    xerr_fa_high = np.maximum(0, df_plot["fa_ci_high"] - x_fa)

    for i in range(len(df_plot)):
        ax1.errorbar(
            x_fa.iloc[i], y_pos[i],
            xerr=[[xerr_fa_low.iloc[i]], [xerr_fa_high.iloc[i]]],
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            elinewidth=2.5,
            capsize=5,
            capthick=2,
            markersize=7,
            zorder=4,
        )
        x_val = x_fa.iloc[i]
        y_p = y_pos[i]
        ci_h = df_plot["fa_ci_high"].iloc[i]
        text_x = ci_h * 1.25 if ci_h > 5 else ci_h + 1.8
        is_bold = ("Weather Fusion" in df_plot["model_name"].iloc[i]) or ("Physics-Informed Nuclear" in df_plot["model_name"].iloc[i])
        ax1.text(
            text_x,
            y_p,
            f"{x_val:.1f} [{df_plot['fa_ci_low'].iloc[i]:.1f}, {df_plot['fa_ci_high'].iloc[i]:.1f}]",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#333333",
            fontweight="bold" if is_bold else "normal",
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_plot["display_name"])
    ax1.set_xlabel("Operational False Alarms per Station-Year (95% CI)")
    ax1.set_title("(a) Clean False Alarm Rates (Log Scale)", fontweight="bold", pad=10)
    ax1.set_xscale("symlog", linthresh=1.0)
    ax1.set_xlim(-0.2, 550)
    ax1.grid(True, which="both", axis="x", alpha=0.3)
    ax1.axvline(0, color="gray", linestyle="-", linewidth=0.8)

    # Highlight Tier 4 row (or Tier 3 if Tier 4 not present)
    t4_match = df_plot[df_plot["model_name"].str.contains(r"Tier 4: Physics-Informed Nuclear GBDT \(95%", regex=True)]
    if not t4_match.empty:
        t4_idx = t4_match.index[0]
        ax1.axhspan(t4_idx - 0.45, t4_idx + 0.45, color="#e3f2fd", alpha=0.6, zorder=1)

    # Panel 2: Realized Detection Rate
    x_det = df_plot["realized_detection_pct"]
    xerr_det_low = np.maximum(0, x_det - df_plot["det_ci_low"])
    xerr_det_high = np.maximum(0, df_plot["det_ci_high"] - x_det)

    for i in range(len(df_plot)):
        ax2.errorbar(
            x_det.iloc[i], y_pos[i],
            xerr=[[xerr_det_low.iloc[i]], [xerr_det_high.iloc[i]]],
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            elinewidth=2.5,
            capsize=5,
            capthick=2,
            markersize=7,
            zorder=4,
        )
        x_val = x_det.iloc[i]
        y_p = y_pos[i]
        ci_l = df_plot["det_ci_low"].iloc[i]
        ci_h = df_plot["det_ci_high"].iloc[i]

        if x_val >= 80:
            label_x = ci_l - 2.0
            ha_align = "right"
        else:
            label_x = ci_h + 2.0
            ha_align = "left"

        is_bold = ("Weather Fusion" in df_plot["model_name"].iloc[i]) or ("Physics-Informed Nuclear" in df_plot["model_name"].iloc[i])
        ax2.text(
            label_x,
            y_p,
            f"{x_val:.1f}% [{ci_l:.1f}%, {ci_h:.1f}%]",
            va="center",
            ha=ha_align,
            fontsize=8.5,
            color="#333333",
            fontweight="bold" if is_bold else "normal",
        )

    if not t4_match.empty:
        ax2.axhspan(t4_idx - 0.45, t4_idx + 0.45, color="#e3f2fd", alpha=0.6, zorder=1)
    ax2.set_xlabel("Event Detection Probability (%) (95% CI)")
    ax2.set_title("(b) Radiological Plume Detection Rates", fontweight="bold", pad=10)
    ax2.set_xlim(-2, 106)
    ax2.grid(True, which="both", axis="x", alpha=0.3)
    ax2.axvline(90, color="gray", linestyle=":", linewidth=1.2, label="90% Target")
    ax2.axvline(95, color="black", linestyle="--", linewidth=1.2, label="95% Target")
    ax2.legend(loc="lower left", framealpha=0.9)

    plt.suptitle("Multi-Model Statistical Benchmark with 95% Bootstrap Confidence Intervals", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = Path("reports/figures/eval_bootstrap_uncertainty.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_detection_delay_distributions():
    """Plots empirical CDF of detection delay and breakdown across release scenarios."""
    csv_path = Path("data/processed/eval_detection_delay_summary.csv")
    emp_path = Path("data/processed/eval_empirical_delays.csv")
    if not csv_path.exists() or not emp_path.exists():
        print(f"Error: {csv_path} or {emp_path} not found.")
        return

    df = pd.read_csv(csv_path)
    emp_df = pd.read_csv(emp_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    time_hours = np.arange(0, 25, 1, dtype=float)

    model_specs = [
        (
            "Tier 4: Physics-Informed Nuclear GBDT (95% target)",
            "Tier 4: Nuclear Physics GBDT",
            "#0d47a1", "-", 3.0,
        ),
        (
            "Tier 3: Full Weather Fusion GBDT (95% target)",
            "Tier 3: Weather Fusion GBDT",
            "#2e7d32", "-", 2.5,
        ),
        (
            "Tier 2: Radiation + Spectrometry GBDT (95% target)",
            "Tier 2: Spectral GBDT",
            "#337ab7", "--", 2.2,
        ),
        (
            "Tier 3: Weather-Fused Neural Net (MLP) (95% target)",
            "Tier 3: Neural Net (MLP)",
            "#7b1fa2", "-.", 2.0,
        ),
        (
            "Baseline: Rolling 7d 3-sigma",
            "Baseline: Rolling 7d 3-sigma",
            "#d9534f", ":", 2.3,
        ),
    ]

    t4_6h_pct = 0.0
    t4_12h_pct = 0.0
    t4_med_h = 4.0
    t4_at_med = 50.0

    for full_name, m_label, col, style, lw in model_specs:
        sub_ev = emp_df[emp_df["model_name"] == full_name]
        if sub_ev.empty:
            continue
        n_tot = len(sub_ev)
        tot_det_pct = (sub_ev["detected"] == 1).mean() * 100.0
        cdf = np.array([
            ((sub_ev["detected"] == 1) & (sub_ev["delay_h"] <= h)).sum() / n_tot * 100.0
            for h in time_hours
        ])
        ax1.step(
            time_hours, cdf, where="post",
            label=f"{m_label} (Total {tot_det_pct:.1f}%)",
            color=col, linestyle=style, linewidth=lw,
        )
        if "Physics-Informed Nuclear GBDT" in full_name:
            t4_6h_pct = float(cdf[time_hours == 6.0][0])
            t4_12h_pct = float(cdf[time_hours == 12.0][0])
            det_only = sub_ev.loc[sub_ev["detected"] == 1, "delay_h"]
            t4_med_h = float(det_only.median()) if len(det_only) > 0 else np.nan
            t4_at_med = float(cdf[time_hours == round(t4_med_h)][0]) if not np.isnan(t4_med_h) else 50.0

    ax1.set_xlabel("Time Elapsed Since Plume Arrival (Hours)", fontweight="medium")
    ax1.set_ylabel("Empirical Cumulative Detection Probability (%)", fontweight="medium")
    ax1.set_title("(a) Empirical Time-to-Alarm Survival CDF (N = 200 Test Events)", fontweight="bold", pad=10)
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, 105)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(3))
    ax1.grid(True, alpha=0.35)
    ax1.axhline(90, color="gray", linestyle=":", alpha=0.7)
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=8.8)

    ax1.annotate(
        f"Tier 4 Nuclear GBDT:\nMedian delay = {t4_med_h:.1f} h\n{t4_6h_pct:.1f}% detected ≤ 6 h | {t4_12h_pct:.1f}% ≤ 12 h",
        xy=(t4_med_h, t4_at_med),
        xytext=(7.5, 34.0),
        arrowprops=dict(facecolor="#0d47a1", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9.2,
        fontweight="bold",
        color="#0d47a1",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e3f2fd", ec="#0d47a1", alpha=0.9),
    )

    # Right Panel: Scenario Delay Breakdown
    sub_sc = df[(df["category"] == "Nuclide Scenario")].copy()

    scenario_labels = {
        "fission_reactor_fukushima": "Fukushima\n(Multi-Nuclide)",
        "fission_pure_cs137": "Cs-137\n(Sealed Source)",
        "activation_orphan_co60": "Co-60\n(Orphan Source)",
        "mixed_fission_activation": "Mixed Core\n(Excursion)",
        "fission_pure_i131": "I-131\n(Radiopharma)",
    }
    sub_sc["display_scenario"] = sub_sc["group"].map(lambda x: scenario_labels.get(x, x))

    models_to_compare = [
        ("Baseline: Rolling 7d 3-sigma", "Rolling 3σ", "#d9534f"),
        ("Tier 2: Radiation + Spectrometry GBDT (95% target)", "Tier 2 (Spectral)", "#337ab7"),
        ("Tier 3: Full Weather Fusion GBDT (95% target)", "Tier 3 (Weather Fusion)", "#2e7d32"),
        ("Tier 4: Physics-Informed Nuclear GBDT (95% target)", "Tier 4 (Nuclear Physics)", "#0d47a1"),
    ]

    scenarios = ["fission_reactor_fukushima", "fission_pure_cs137", "activation_orphan_co60", "mixed_fission_activation", "fission_pure_i131"]
    x = np.arange(len(scenarios))
    bar_width = 0.20

    for i, (m_id, m_short, col) in enumerate(models_to_compare):
        m_data = sub_sc[sub_sc["model_name"] == m_id].set_index("group")
        medians = [m_data.loc[sc, "delay_median_h"] if sc in m_data.index else 0 for sc in scenarios]
        p25s = [m_data.loc[sc, "delay_p25_h"] if sc in m_data.index else 0 for sc in scenarios]
        p75s = [m_data.loc[sc, "delay_p75_h"] if sc in m_data.index else 0 for sc in scenarios]

        yerr_low = np.maximum(0, np.array(medians) - np.array(p25s))
        yerr_high = np.maximum(0, np.array(p75s) - np.array(medians))

        ax2.bar(
            x + (i - 1.5) * bar_width,
            medians,
            width=bar_width,
            yerr=[yerr_low, yerr_high],
            capsize=3,
            label=m_short,
            color=col,
            alpha=0.88,
            edgecolor="black",
            linewidth=0.6,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([scenario_labels[sc] for sc in scenarios], fontsize=9.5)
    ax2.set_ylabel("Median Detection Delay (Hours) [IQR]", fontweight="medium")
    ax2.set_title("(b) Detection Delay by Radiological Scenario", fontweight="bold", pad=10)
    ax2.set_xlim(-0.6, 4.6)
    ax2.set_ylim(0, 23)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(4))
    ax2.grid(True, axis="y", alpha=0.35)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=9.0)

    plt.suptitle("Temporal Response and Detection Delay Characterization", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = Path("reports/figures/eval_detection_delay_distributions.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_confusion_matrices():
    """Generates 4-panel heatmap of multi-class confusion matrices."""
    csv_path = Path("data/processed/eval_confusion_matrices.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    panels = [
        ("Baseline: Rolling 7d 3-sigma", "(a) Baseline: Rolling 7d 3σ"),
        ("Tier 2: Radiation + Spectrometry GBDT", "(b) Tier 2: Radiation + Spectrometry"),
        ("Tier 3: Full Weather Fusion GBDT", "(c) Tier 3: Full Weather Fusion GBDT"),
        ("Tier 4: Physics-Informed Nuclear GBDT", "(d) Tier 4: Physics-Informed Nuclear GBDT"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    classes = ["normal", "radon_washout", "fission_product"]
    display_classes = ["Normal\n(Clean Bkg)", "Radon\nWashout", "Fission\nProduct"]

    for ax, (m_id, title) in zip(axes, panels):
        sub = df[df["model_name"] == m_id]
        mat_recall = np.zeros((3, 3))
        mat_counts = np.zeros((3, 3), dtype=int)

        for _, row in sub.iterrows():
            i = classes.index(row["true_class"])
            j = classes.index(row["predicted_class"])
            mat_recall[i, j] = row["recall_fraction"] * 100.0
            mat_counts[i, j] = int(row["raw_count"])

        im = ax.imshow(mat_recall, cmap="Blues", vmin=0, vmax=100)

        for i in range(3):
            for j in range(3):
                val_pct = mat_recall[i, j]
                cnt = mat_counts[i, j]
                text_color = "white" if val_pct > 55 else "black"
                ax.text(
                    j, i,
                    f"{val_pct:.1f}%\n({cnt:,})",
                    ha="center", va="center",
                    color=text_color,
                    fontsize=9.5,
                    fontweight="bold" if i == j else "normal",
                )

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(display_classes, fontsize=9.5)
        ax.set_yticklabels(display_classes, fontsize=9.5)
        ax.set_xlabel("Predicted Class", fontweight="medium")
        ax.set_ylabel("True Class", fontweight="medium")
        ax.set_title(title, fontweight="bold", pad=8)

        if "Baseline" in m_id:
            rect = plt.Rectangle((1.5, 0.5), 1, 1, fill=False, edgecolor="#d9534f", linewidth=2.5, linestyle="--")
            ax.add_patch(rect)
        elif "Tier 3" in m_id or "Tier 4" in m_id:
            rect = plt.Rectangle((1.5, 0.5), 1, 1, fill=False, edgecolor="#2e7d32", linewidth=2.5, linestyle="-")
            ax.add_patch(rect)

    plt.suptitle("Multi-Class Hourly Confusion Matrices (Recall % Normalized by True Class)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = Path("reports/figures/eval_confusion_matrices.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main():
    print("Generating uncertainty and delay evaluation figures...")
    plot_bootstrap_uncertainty()
    plot_detection_delay_distributions()
    plot_confusion_matrices()
    print("All uncertainty and delay evaluation figures generated successfully.")


if __name__ == "__main__":
    main()
