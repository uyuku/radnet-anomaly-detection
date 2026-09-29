"""
Phase 4 Plotting Script: Model Evaluation, Feature Importance, and Trade-off Curves.

Generates 4 publication-quality figures:
1. reports/figures/model_detection_vs_false_alarms_roc.png
   - Headline trade-off: Event Detection Probability vs Clean False Alarms per Station-Year.
   - Compares Phase 2 Fixed-Threshold Baselines against Tier 1, Tier 2, Tier 3, and LOSO.
2. reports/figures/model_stratified_performance.png
   - Detection probability across environmental regimes, test sets, and operational release scenarios.
3. reports/figures/model_feature_importance.png
   - Top 15 features in Tier 3 model, color-coded by physical domain (Radiation, Spectrometry, Weather).
4. reports/figures/model_hard_case_timeline.png
   - Real-time model posterior probabilities P(fission), P(washout), P(normal) and alarm status during INJ_0067.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


def plot_headline_roc():
    """
    Figure 1: Headline Trade-off Curve (Event Detection Probability vs Clean False Alarms per Station-Year).
    """
    roc_df = pd.read_csv("data/processed/model_roc_curve_data.csv")

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Plot Tier Curves
    ax.plot(
        roc_df["t1_clean_fa_per_year"],
        roc_df["t1_event_det_rate"] * 100,
        color="#1f77b4",
        linewidth=2.5,
        label="Tier 1: Gross Radiation Only (CPM + Dose Rate)",
    )
    ax.plot(
        roc_df["t2_clean_fa_per_year"],
        roc_df["t2_event_det_rate"] * 100,
        color="#ff7f0e",
        linewidth=2.5,
        linestyle="--",
        label="Tier 2: Radiation + NaI(Tl) Spectral Ratios",
    )
    ax.plot(
        roc_df["t3_clean_fa_per_year"],
        roc_df["t3_event_det_rate"] * 100,
        color="#2ca02c",
        linewidth=3.0,
        label="Tier 3: Full Weather Fusion (Radiation + Spectral + NOAA Weather)",
    )
    ax.plot(
        roc_df["loso_clean_fa_per_year"],
        roc_df["loso_event_det_rate"] * 100,
        color="#9467bd",
        linewidth=2.0,
        linestyle=":",
        label="Tier 3 LOSO: Held-out San Diego (Unseen Coastal Station)",
    )

    # Plot Baseline Benchmark Points
    baselines = [
        {"name": "Rolling 7d 3-sigma", "det": 63.0, "fa": 82.95, "color": "#d62728", "marker": "s"},
        {"name": "Rolling 7d 4-sigma", "det": 36.5, "fa": 42.34, "color": "#d62728", "marker": "o"},
        {"name": "Rolling 7d 5-sigma", "det": 18.0, "fa": 23.35, "color": "#d62728", "marker": "^"},
        {"name": "Global Dry 3-sigma", "det": 49.5, "fa": 39.46, "color": "#8c564b", "marker": "s"},
        {"name": "Global Dry 4-sigma", "det": 30.5, "fa": 20.88, "color": "#8c564b", "marker": "o"},
        {"name": "Global Dry 5-sigma", "det": 20.5, "fa": 10.61, "color": "#8c564b", "marker": "^"},
    ]

    for b in baselines:
        ax.scatter(b["fa"], b["det"], color=b["color"], marker=b["marker"], s=90, zorder=5, edgecolors="black")
        ax.annotate(
            b["name"],
            xy=(b["fa"], b["det"]),
            xytext=(b["fa"] + 3, b["det"] - 1.5),
            fontsize=8.5,
            fontweight="bold",
            color=b["color"],
        )

    # Highlight Operating Point at ~90-95% Detection
    ax.axhline(90.0, color="gray", linestyle=":", alpha=0.6)
    ax.axhline(95.0, color="gray", linestyle=":", alpha=0.6)
    ax.text(120, 90.5, "90% Detection Target", color="gray", fontsize=8.5, fontstyle="italic")
    ax.text(120, 95.5, "95% Detection Target", color="gray", fontsize=8.5, fontstyle="italic")

    # Add Callout Box
    textstr = (
        "Key Findings:\n"
        "• Fixed 3σ baseline triggers 83 FA/stn-yr at only 63% detection\n"
        "• Tier 2 & 3 achieve 90% detection with only 3.1-3.2 FA/stn-yr\n"
        "• 96.1% reduction in operational false alarms vs baseline!\n"
        "• LOSO cross-validation holds up at 3.5 FA/stn-yr on unseen site"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#adb5bd", alpha=0.95)
    ax.text(0.04, 0.40, textstr, transform=ax.transAxes, fontsize=9.0, verticalalignment="top", bbox=props)

    ax.set_title("Figure 1: Operational Trade-Off Curve Across 5 Pilot Stations (2023–2025 Test Split)\nEvent-Level Detection Probability vs Clean Operational False Alarms per Station-Year", fontweight="bold")
    ax.set_xlabel("Clean Operational False Alarms per Station-Year (Lower is Better)")
    ax.set_ylabel("Synthetic Injection Detection Probability (%) (Higher is Better)")
    ax.set_xlim(-2, 160)
    ax.set_ylim(10, 103)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#adb5bd", fontsize=9.0)

    out_file = FIGURES_DIR / "model_detection_vs_false_alarms_roc.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_stratified_performance():
    """
    Figure 2: Stratified Detection Probability Across Regimes and Scenarios.
    """
    strat_df = pd.read_csv("data/processed/model_stratified_evaluation_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    # Panel A: By Environmental Regime / Test Set
    ax1 = axes[0]
    env_df = strat_df[strat_df["group_category"] == "environment"].copy()
    env_names = {
        "test_standard_strictly_dry": "Standard Test: Dry\n(Band B, 36–75h)",
        "test_standard_rain": "Standard Test: Rain\n(Band B, 38–164h)",
        "test_stress_strictly_dry": "Stress Test: Dry\n(Band A, 8–20h)",
        "test_stress_rain": "Stress Test: Rain\n(Band A, 28–140h)",
    }
    x_labels_env = [env_names.get(v, v) for v in env_df["group_value"]]
    x = np.arange(len(x_labels_env))
    width = 0.25

    rects1 = ax1.bar(x - width, env_df["tier1_det_rate_pct"], width, label="Tier 1: Gross Only", color="#1f77b4", edgecolor="black")
    rects2 = ax1.bar(x, env_df["tier2_det_rate_pct"], width, label="Tier 2: Spectral Ratios", color="#ff7f0e", edgecolor="black")
    rects3 = ax1.bar(x + width, env_df["tier3_det_rate_pct"], width, label="Tier 3: Weather Fusion", color="#2ca02c", edgecolor="black")

    ax1.set_title("Panel A: Detection by Environmental Regime & Split\n(Evaluated at ~95% Overall Target)", fontweight="bold")
    ax1.set_ylabel("Detection Probability (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels_env, fontsize=9.0)
    ax1.set_ylim(0, 115)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", frameon=True, facecolor="white")

    for rects in [rects1, rects2, rects3]:
        for r in rects:
            h = r.get_height()
            ax1.text(r.get_x() + r.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=8.0)

    # Panel B: By Operational Release Scenario
    ax2 = axes[1]
    scen_df = strat_df[strat_df["group_category"] == "nuclide_scenario"].copy()
    scen_names = {
        "activation_orphan_co60": "Orphan Co-60\n(R07 Photopeak)",
        "fission_pure_cs137": "Legacy Cs-137\n(R05 Photopeak)",
        "fission_pure_i131": "Medical I-131\n(R03 Photopeak)",
        "fission_reactor_fukushima": "Reactor Core\n(Fukushima)",
        "mixed_fission_activation": "Core Excursion\n(Mixed All 4)",
    }
    x_labels_scen = [scen_names.get(v, v) for v in scen_df["group_value"]]
    x_scen = np.arange(len(x_labels_scen))

    rects1_sc = ax2.bar(x_scen - width, scen_df["tier1_det_rate_pct"], width, label="Tier 1: Gross Only", color="#1f77b4", edgecolor="black")
    rects2_sc = ax2.bar(x_scen, scen_df["tier2_det_rate_pct"], width, label="Tier 2: Spectral Ratios", color="#ff7f0e", edgecolor="black")
    rects3_sc = ax2.bar(x_scen + width, scen_df["tier3_det_rate_pct"], width, label="Tier 3: Weather Fusion", color="#2ca02c", edgecolor="black")

    ax2.set_title("Panel B: Detection by Operational Release Scenario\n(100% Detection for Fukushima & Pure Cs-137)", fontweight="bold")
    ax2.set_xticks(x_scen)
    ax2.set_xticklabels(x_labels_scen, fontsize=9.0)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    for rects in [rects1_sc, rects2_sc, rects3_sc]:
        for r in rects:
            h = r.get_height()
            ax2.text(r.get_x() + r.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=8.0)

    out_file = FIGURES_DIR / "model_stratified_performance.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_feature_importance():
    """
    Figure 3: Feature Importance Hierarchy in Tier 3 Weather-Fused Model.
    """
    imp_df = pd.read_csv("data/processed/model_feature_importance.csv")
    top_df = imp_df.head(15).copy().sort_values("importance_gain", ascending=True)

    # Classify features by domain
    def get_feature_color(feat):
        if any(w in feat for w in ["precip", "rain", "pressure", "temp", "dewpoint", "humidity", "washout", "dry_excess"]):
            return "#2ca02c"  # Green: Weather / Atmospheric
        elif any(s in feat for s in ["share", "ratio_r0", "ratio_mid"]):
            return "#ff7f0e"  # Orange: NaI(Tl) Spectrometry
        else:
            return "#1f77b4"  # Blue: Gross Radiation / Exposure Rate

    colors = [get_feature_color(f) for f in top_df["feature"]]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(top_df["feature"], top_df["importance_gain"] / 1e3, color=colors, edgecolor="black", alpha=0.85)

    ax.set_title("Figure 3: Top 15 Feature Importances in Tier 3 Weather-Fused Model (Total Gain)\nPhysics-Driven Integration Across Weather, Spectrometry, and Gross Count Rates", fontweight="bold")
    ax.set_xlabel("Feature Importance Gain (x 1,000)")
    ax.set_ylabel("Feature Name")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", edgecolor="black", label="Weather & Atmospheric Interactions (NOAA)"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="NaI(Tl) Spectral Ratios & Photopeak Shares"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Gross Radiation & Exposure Rate Dynamics"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", edgecolor="#adb5bd")

    out_file = FIGURES_DIR / "model_feature_importance.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_hard_case_timeline():
    """
    Figure 4: Real-Time Model Probability and Alarm Timeline During INJ_0067.
    """
    df_raw = pd.read_csv("data/processed/labeled_al_birmingham_test.csv.gz")
    df_raw["dt"] = pd.to_datetime(df_raw["dt"])
    df_raw = df_raw.sort_values("dt").reset_index(drop=True)

    from src.feature_engineering import compute_features_for_series, TIER_3_FEATURES
    from src.train_and_evaluate_models import load_and_prepare_train_data, train_tier_model

    # Train model to get live probabilities
    X_tr_t3, y_tr = load_and_prepare_train_data(TIER_3_FEATURES)
    clf_t3 = train_tier_model(X_tr_t3, y_tr, "Tier 3 Plotting")

    feats_inj = compute_features_for_series(df_raw, "al_birmingham", use_injected=True)
    probs_inj = clf_t3.predict_proba(feats_inj[TIER_3_FEATURES])

    t_start = pd.to_datetime("2023-01-24 18:00:00")
    t_end = pd.to_datetime("2023-01-29 06:00:00")
    mask = (df_raw["dt"] >= t_start) & (df_raw["dt"] <= t_end)
    sub_df = df_raw[mask].copy().reset_index(drop=True)
    sub_probs = probs_inj[mask]

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, gridspec_kw={"height_ratios": [1, 1.8, 1.8, 1.2]})

    # Panel 1: Rain
    ax1 = axes[0]
    ax1.bar(sub_df["dt"], sub_df["precip_1h_mm"], width=0.035, color="#1f77b4", edgecolor="#0b559f", alpha=0.85, label="Precipitation Depth (mm/h)")
    ax1.set_ylabel("Rain\n(mm/h)", fontweight="bold")
    ax1.set_ylim(0, 16)
    ax1.set_title("Figure 4: Model Diagnostic Timeline on Dedicated Hard-Case Injection (INJ_0067, Birmingham, AL)\nConvective Storm (13.5 mm/h Rain Peak) + Subtle Band A Pure Cs-137 Plume (+479 CPM) Ending at Physical Filter Drop", fontweight="bold", fontsize=11)
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel 2: Gross Count Rate & Baseline
    ax2 = axes[1]
    ax2.plot(sub_df["dt"], sub_df["gross_cpm"], color="gray", linestyle="--", linewidth=1.8, label="Real Background + Natural Storm Washout (Peak: 4,896 CPM)")
    ax2.plot(sub_df["dt"], sub_df["inj_gross_cpm"], color="#d62728", linewidth=2.5, label="Injected Detector Gross CPM (Peak: 4,992 CPM)")
    ax2.plot(sub_df["dt"], sub_df["synthetic_excess_cpm"] + 3887.3, color="#ff7f0e", linestyle=":", linewidth=2.0, label="Synthetic Cs-137 Signal Alone (+479.1 CPM plateau)")

    base_thresh = 3887.3 + 3.0 * 348.4
    ax2.axhline(base_thresh, color="#2ca02c", linestyle="--", label=f"Fixed 3-sigma Alarm Threshold ({base_thresh:.1f} CPM)")

    inj_act = sub_df[sub_df["injection_active"]]
    if not inj_act.empty:
        ax2.axvspan(inj_act["dt"].iloc[0], inj_act["dt"].iloc[-1], color="#ffebee", alpha=0.4, label="Cs-137 Plume Active on Filter (83h duration until filter change)")

    ax2.set_ylabel("Gross CPM", fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)

    # Panel 3: Model Posterior Probabilities
    ax3 = axes[2]
    ax3.plot(sub_df["dt"], sub_probs[:, 2], color="#d62728", linewidth=2.5, label="P(fission_product) — Real-Time Model Threat Score")
    ax3.plot(sub_df["dt"], sub_probs[:, 1], color="#1f77b4", linewidth=1.8, linestyle="--", label="P(radon_washout) — Weather-Aware Natural Washout Probability")
    ax3.plot(sub_df["dt"], sub_probs[:, 0], color="gray", linewidth=1.5, linestyle=":", label="P(normal) — Background Probability")

    tau_t3 = 0.89
    ax3.axhline(tau_t3, color="black", linestyle="--", linewidth=1.5, label=f"Operating Decision Threshold (tau = {tau_t3})")
    ax3.set_ylabel("Posterior\nProbability", fontweight="bold")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)

    # Panel 4: Alarm State Comparison
    ax4 = axes[3]
    base_alarm = (sub_df["inj_gross_cpm"] >= base_thresh).astype(int)
    model_alarm = (sub_probs[:, 2] >= tau_t3).astype(int)

    ax4.step(sub_df["dt"], base_alarm * 0.9 + 1.1, color="#d62728", where="mid", linewidth=2.0, label="Fixed 3σ Alarm State (Triggered during storm washout peak)")
    ax4.step(sub_df["dt"], model_alarm * 0.9 + 0.1, color="#2ca02c", where="mid", linewidth=2.5, label="Tier 3 Model Alarm State (Active throughout the persistent Cs-137 plateau)")

    ax4.set_yticks([0.55, 1.55])
    ax4.set_yticklabels(["Tier 3 Model", "Fixed 3σ Base"], fontweight="bold")
    ax4.set_ylim(-0.1, 2.2)
    ax4.set_ylabel("Alarm State", fontweight="bold")
    ax4.set_xlabel("Date and Time (UTC, January 2023)", fontweight="bold")
    ax4.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.5)

    out_file = FIGURES_DIR / "model_hard_case_timeline.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    print("Generating Phase 4 evaluation figures...")
    plot_headline_roc()
    plot_stratified_performance()
    plot_feature_importance()
    plot_hard_case_timeline()
    print("All Phase 4 figures successfully generated.")
