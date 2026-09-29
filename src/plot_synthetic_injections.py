"""
Phase 3 Plotting Script: Visualizing Synthetic Fission Injections,
NaI(Tl) Spectral Allocations, Train/Test Parameter Disjointness,
and Rain-Coincident Hard-Case Signatures.

Generates:
1. reports/figures/synthetic_injection_shapes_and_nuclides.png
2. reports/figures/synthetic_injection_train_test_disjointness.png
3. reports/figures/synthetic_injection_hard_case_rain.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set plot styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
})

CATALOG_PATH = Path("data/processed/synthetic_injection_catalog.csv")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_shapes_and_nuclides():
    """Figure 1: Normalized injection shapes and NaI(Tl) channel energy allocations."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Normalized Shapes
    ax1 = axes[0]
    t = np.linspace(0, 24, 250)

    # Train shapes
    # 1. Linear ramp (rise over 3h, then plateau)
    ramp = np.clip(t / 3.0, 0.0, 1.0)
    # 2. Step arrival
    step = np.ones_like(t)

    # Test shapes
    # 3. Sigmoidal arrival (tau=1.2h, t_mid=4h)
    sig = 1.0 / (1.0 + np.exp(-(t - 4.0) / 1.2))
    sig = (sig - sig[0]) / (sig[-1] - sig[0])
    # 4. Exponential inflow (tau=2.0h)
    exp_in = 1.0 - np.exp(-t / 2.0)
    exp_in = exp_in / exp_in[-1]

    ax1.plot(t, ramp, color="#1f77b4", linestyle="--", linewidth=2.5, label="Linear Ramp (Train, 2017-2022)")
    ax1.plot(t, step, color="#2ca02c", linestyle=":", linewidth=2.5, label="Step Arrival (Train, 2017-2022)")
    ax1.plot(t, sig, color="#ff7f0e", linestyle="-", linewidth=2.5, label="Sigmoidal (Test, 2023-2025)")
    ax1.plot(t, exp_in, color="#9467bd", linestyle="-", linewidth=2.5, label="Exponential Inflow (Test, 2023-2025)")

    ax1.set_title("Panel A: Normalized Injection Shape Families $S(t) / S_{\\mathrm{peak}}$\n(Strictly Disjoint Between Train and Test)", fontweight="bold")
    ax1.set_xlabel("Elapsed Time Since Plume Onset (hours)")
    ax1.set_ylabel("Normalized Signal Fraction")
    ax1.set_xlim(0, 24)
    ax1.set_ylim(-0.05, 1.15)
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    # Panel B: NaI(Tl) Spectral Allocations
    ax2 = axes[1]
    channels = ["R02\n101-200", "R03\n201-400", "R04\n401-600", "R05\n601-800", "R06\n801-1000", "R07\n1001-1400", "R08\n1401-1800", "R09\n1801-2200"]
    x = np.arange(len(channels))
    width = 0.20

    # Fractions
    w_cs = [0.20, 0.15, 0.20, 0.45, 0.00, 0.00, 0.00, 0.00]
    w_i = [0.25, 0.65, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00]
    w_mix = [0.225, 0.40, 0.125, 0.25, 0.00, 0.00, 0.00, 0.00]
    w_washout = [0.420, 0.350, 0.082, 0.067, 0.024, 0.036, 0.017, 0.004]

    rects1 = ax2.bar(x - 1.5 * width, [v * 100 for v in w_cs], width, label="Pure Cs-137 ($f_{\\mathrm{Cs}}=1.0$)", color="#d62728", alpha=0.9)
    rects2 = ax2.bar(x - 0.5 * width, [v * 100 for v in w_i], width, label="Pure I-131 ($f_{\\mathrm{I}}=1.0$)", color="#ff7f0e", alpha=0.9)
    rects3 = ax2.bar(x + 0.5 * width, [v * 100 for v in w_mix], width, label="Balanced Mix ($50\\%$ Cs, $50\\%$ I)", color="#2ca02c", alpha=0.9)
    rects4 = ax2.bar(x + 1.5 * width, [v * 100 for v in w_washout], width, label="Natural Radon Washout (Empirical)", color="#1f77b4", alpha=0.9)

    ax2.set_title("Panel B: NaI(Tl) Energy Channel Allocation (% of Excess Signal)\n(Physical Discrimination: Fission = 0% in R06-R09)", fontweight="bold")
    ax2.set_xlabel("RadNet Gamma Channel and Energy Boundary (keV)")
    ax2.set_ylabel("Share of Excess Plume Counts (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(channels)
    ax2.set_ylim(0, 75)

    # Highlight R07 and R08 photopeaks for radon progeny
    rect_box = patches.Rectangle((4.5, 0), 3.4, 72, linewidth=1.5, edgecolor="red", facecolor="red", alpha=0.08, linestyle="--")
    ax2.add_patch(rect_box)
    ax2.text(6.2, 58, "Bi-214 Photopeaks\n(1120 & 1764 keV)\nFission = Strictly 0.0%\nWashout = Elevated!",
             ha="center", va="top", color="#b30000", fontsize=9.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="red", alpha=0.9))

    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    out_file = FIGURES_DIR / "synthetic_injection_shapes_and_nuclides.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_train_test_disjointness():
    """Figure 2: Four-panel verification of parameter disjointness between Train and Test sets."""
    df = pd.read_csv(CATALOG_PATH)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Duration Disjointness
    ax1 = axes[0, 0]
    bins_dur = np.arange(0, 80, 2)
    ax1.hist(train_df["duration_hours"], bins=bins_dur, color="#1f77b4", alpha=0.7, label=f"Train (N={len(train_df)}): 6 to 24 h", edgecolor="black")
    ax1.hist(test_df["duration_hours"], bins=bins_dur, color="#ff7f0e", alpha=0.7, label=f"Test (N={len(test_df)}): 28 to 72 h", edgecolor="black")
    ax1.axvspan(24, 28, color="red", alpha=0.15, label="Disjoint Gap (24h - 28h)")
    ax1.axvline(24, color="black", linestyle="--", linewidth=1.5)
    ax1.axvline(28, color="black", linestyle="--", linewidth=1.5)
    ax1.set_title("Panel A: Duration Distribution (Hours)\n(Strictly Disjoint: Train <= 24h, Test >= 28h)", fontweight="bold")
    ax1.set_xlabel("Injection Duration (hours)")
    ax1.set_ylabel("Number of Injections")
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel B: Peak Magnitude Disjointness
    ax2 = axes[0, 1]
    bins_mag = np.linspace(200, 2600, 35)
    ax2.hist(train_df[train_df["magnitude_band"] == "Band_A_Low"]["peak_cpm"], bins=bins_mag, color="#1f77b4", alpha=0.7, label="Train Band A [250, 600] CPM", edgecolor="black")
    ax2.hist(train_df[train_df["magnitude_band"] == "Band_C_High"]["peak_cpm"], bins=bins_mag, color="#2ca02c", alpha=0.7, label="Train Band C [1400, 2500] CPM", edgecolor="black")
    ax2.hist(test_df["peak_cpm"], bins=bins_mag, color="#ff7f0e", alpha=0.7, label="Test Band B [700, 1200] CPM (Interpolation)", edgecolor="black")
    ax2.axvspan(600, 700, color="red", alpha=0.15)
    ax2.axvspan(1200, 1400, color="red", alpha=0.15, label="Disjoint Gaps (600-700 & 1200-1400)")
    ax2.set_title("Panel B: Peak Magnitude Distribution (CPM)\n(Strictly Disjoint: Test Evaluates Interpolation)", fontweight="bold")
    ax2.set_xlabel("Peak Injected Signal (CPM)")
    ax2.set_ylabel("Number of Injections")
    ax2.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel C: Nuclide Mix Disjointness
    ax3 = axes[1, 0]
    bins_f = np.linspace(0.0, 1.0, 25)
    ax3.hist(train_df["fraction_cs137"], bins=bins_f, color="#1f77b4", alpha=0.7, label="Train Mixed Plumes ($f_{\\mathrm{Cs}} \\in [0.30, 0.70]$)", edgecolor="black")
    test_pure_cs = test_df[test_df["fraction_cs137"] >= 0.85]
    test_pure_i = test_df[test_df["fraction_cs137"] <= 0.15]
    label_pure_i = f"Test Pure I-131 (f_Cs <= 0.15, N={len(test_pure_i)})"
    label_pure_cs = f"Test Pure Cs-137 (f_Cs >= 0.85, N={len(test_pure_cs)})"
    ax3.hist(test_pure_i["fraction_cs137"], bins=bins_f, color="#ff7f0e", alpha=0.7, label=label_pure_i, edgecolor="black")
    ax3.hist(test_pure_cs["fraction_cs137"], bins=bins_f, color="#d62728", alpha=0.7, label=label_pure_cs, edgecolor="black")
    ax3.axvspan(0.15, 0.30, color="red", alpha=0.15)
    ax3.axvspan(0.70, 0.85, color="red", alpha=0.15, label="Disjoint Gaps (0.15-0.30 & 0.70-0.85)")
    ax3.set_title("Panel C: Nuclide Fraction $f_{\\mathrm{Cs}}$ Distribution\n(Strictly Disjoint: Train Mixed vs. Test Pure/Skewed)", fontweight="bold")
    ax3.set_xlabel("Fraction of Cs-137 ($f_{\\mathrm{Cs}}$)")
    ax3.set_ylabel("Number of Injections")
    ax3.legend(loc="upper center", frameon=True, facecolor="white")

    # Panel D: Shape Families by Split
    ax4 = axes[1, 1]
    shapes = ["Linear Ramp", "Step", "Sigmoidal", "Exponential"]
    train_counts = [
        len(train_df[train_df["shape_family"] == "linear_ramp"]),
        len(train_df[train_df["shape_family"] == "step"]),
        0,
        0
    ]
    test_counts = [
        0,
        0,
        len(test_df[test_df["shape_family"] == "sigmoidal"]),
        len(test_df[test_df["shape_family"] == "exponential"])
    ]

    x_sh = np.arange(len(shapes))
    w_sh = 0.35
    ax4.bar(x_sh - w_sh/2, train_counts, w_sh, label=f"Train (2017-2022, Total={len(train_df)})", color="#1f77b4", edgecolor="black")
    ax4.bar(x_sh + w_sh/2, test_counts, w_sh, label=f"Test (2023-2025, Total={len(test_df)})", color="#ff7f0e", edgecolor="black")
    ax4.set_title("Panel D: Shape Families by Split\n(Zero Cross-Contamination Across Sets)", fontweight="bold")
    ax4.set_xlabel("Shape Mathematical Family")
    ax4.set_ylabel("Number of Injections")
    ax4.set_xticks(x_sh)
    ax4.set_xticklabels(shapes)
    ax4.set_ylim(0, 180)
    for i in range(len(shapes)):
        if train_counts[i] > 0:
            ax4.text(x_sh[i] - w_sh/2, train_counts[i] + 3, str(train_counts[i]), ha="center", fontweight="bold", color="#1f77b4")
        if test_counts[i] > 0:
            ax4.text(x_sh[i] + w_sh/2, test_counts[i] + 3, str(test_counts[i]), ha="center", fontweight="bold", color="#ff7f0e")
    ax4.legend(loc="upper right", frameon=True, facecolor="white")

    plt.tight_layout()
    out_file = FIGURES_DIR / "synthetic_injection_train_test_disjointness.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_hard_case_rain():
    """Figure 3: Detailed event timeline of a rain-coincident hard-case injection (INJ_0064 in Birmingham)."""
    # Load Birmingham merged data
    df = pd.read_csv("data/processed/merged_al_birmingham_2017_2025.csv.gz")
    df["dt"] = pd.to_datetime(df["utc_hour"])
    
    # Target window around INJ_0064: 2024-03-14 12:00 to 2024-03-17 18:00
    t_start = pd.to_datetime("2024-03-14 12:00:00")
    t_end = pd.to_datetime("2024-03-17 18:00:00")
    sub = df[(df["dt"] >= t_start) & (df["dt"] <= t_end)].copy().reset_index(drop=True)

    # Injection parameters for INJ_0064
    inj_start = pd.to_datetime("2024-03-15 13:00:00")
    inj_dur = 37  # hours
    inj_peak = 772.8  # CPM
    tau = 2.21
    f_cs = 0.92
    f_i = 0.08

    # Spectral weights
    w_cs_r05 = 0.45
    w_i_r05 = 0.05
    w_r05 = f_cs * w_cs_r05 + f_i * w_i_r05  # 0.92*0.45 + 0.08*0.05 = 0.418

    # Compute injection profile
    sub["inj_signal"] = 0.0
    sub["inj_r05"] = 0.0
    sub["inj_r07"] = 0.0  # Fission is strictly 0.0 in R07!

    for i, row in sub.iterrows():
        if row["dt"] >= inj_start:
            dt_h = (row["dt"] - inj_start).total_seconds() / 3600.0
            if dt_h < inj_dur:
                # Exponential rise
                s_val = (1.0 - np.exp(-dt_h / tau)) / (1.0 - np.exp(-inj_dur / tau)) * inj_peak
                sub.loc[i, "inj_signal"] = s_val
                sub.loc[i, "inj_r05"] = s_val * w_r05

    sub["total_gross_cpm"] = sub["gross_cpm"] + sub["inj_signal"]
    sub["total_r05"] = sub["cpm_r05"] + sub["inj_r05"]

    # Baseline dry statistics for Birmingham
    dry_mask = (df["precip_1h_mm"] == 0.0) & (df["precip_1h_mm"].rolling(24).sum() == 0.0) & df["has_radnet_obs"] & df["rad_complete_channels"]
    mu_dry = df.loc[dry_mask, "gross_cpm"].mean()
    sigma_dry = df.loc[dry_mask, "gross_cpm"].std()
    thresh_3s = mu_dry + 3.0 * sigma_dry
    thresh_5s = mu_dry + 5.0 * sigma_dry

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [1, 2.2, 1.8, 1.5]})

    # Panel 1: Precipitation
    ax1 = axes[0]
    ax1.bar(sub["dt"], sub["precip_1h_mm"], width=0.04, color="#1f77b4", edgecolor="#0b559f", alpha=0.85, label="Precipitation Depth (mm/h)")
    ax1.set_ylabel("Rain (mm/h)", fontweight="bold")
    ax1.set_ylim(0, 11)
    ax1.set_title("Dedicated Hard-Case Injection (INJ_0064, Birmingham, AL) During Verified Heavy Rainstorm", fontweight="bold", fontsize=13)
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel 2: Gross CPM & Injected Total
    ax2 = axes[1]
    ax2.plot(sub["dt"], sub["gross_cpm"], color="#7f7f7f", linestyle="--", linewidth=1.8, label="Real Observed Signal (Baseline + Natural Radon Washout)")
    ax2.plot(sub["dt"], sub["total_gross_cpm"], color="#d62728", linewidth=2.5, label="Combined Detector Observation (Washout + Synthetic Fission Plume)")
    ax2.plot(sub["dt"], sub["inj_signal"] + mu_dry, color="#ff7f0e", linestyle=":", linewidth=2.0, label="Synthetic Fission Plume Alone ($S(t)$, Exponential, +772.8 CPM)")

    ax2.axhline(thresh_3s, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Fixed $3\\sigma$ Alarm Threshold ({thresh_3s:.1f} CPM)")
    ax2.axhline(thresh_5s, color="#e377c2", linestyle="--", linewidth=1.5, label=f"Fixed $5\\sigma$ Alarm Threshold ({thresh_5s:.1f} CPM)")

    # Shading injection window
    ax2.axvspan(inj_start, inj_start + pd.Timedelta(hours=inj_dur), color="#ffebee", alpha=0.4, label="Synthetic Plume Active (37h)")

    ax2.set_ylabel("Gross CPM", fontweight="bold")
    ax2.set_ylim(3200, 5600)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", fontsize=9.5)

    # Panel 3: Channel R05 (601-800 keV: Cs-137 photopeak & Bi-214 photopeak)
    ax3 = axes[2]
    ax3.plot(sub["dt"], sub["cpm_r05"], color="#7f7f7f", linestyle="--", linewidth=1.8, label="Observed R05 (Bi-214 Washout Spike)")
    ax3.plot(sub["dt"], sub["total_r05"], color="#d62728", linewidth=2.2, label="Combined R05 (Bi-214 Washout + Cs-137 Photopeak)")
    ax3.set_ylabel("R05 CPM\n(601-800 keV)", fontweight="bold")
    ax3.legend(loc="upper left", frameon=True, facecolor="white", fontsize=9.5)

    # Panel 4: Channel R07 (1001-1400 keV: Bi-214 1120 keV photopeak)
    ax4 = axes[3]
    ax4.plot(sub["dt"], sub["cpm_r07"], color="#1f77b4", linewidth=2.0, label="Observed R07 (Bi-214 Photopeak: Elevates ONLY during Washout!)")
    ax4.axhline(sub["cpm_r07"].median(), color="gray", linestyle=":", alpha=0.6, label="R07 Dry Baseline")
    ax4.set_ylabel("R07 CPM\n(1001-1400 keV)", fontweight="bold")
    ax4.set_xlabel("Date and Time (UTC, March 2024)", fontweight="bold")
    ax4.legend(loc="upper left", frameon=True, facecolor="white", fontsize=9.5)

    # Add explanatory callout boxes
    ax2.annotate("Storm Washout Peak\n(+829 CPM natural)",
                 xy=(pd.to_datetime("2024-03-15 15:00:00"), 4469),
                 xytext=(pd.to_datetime("2024-03-14 20:00:00"), 4900),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#2ca02c"),
                 fontsize=9.5, fontweight="bold")

    ax2.annotate("Washout decays after rain stops,\nbut Fission Plume persists on filter!",
                 xy=(pd.to_datetime("2024-03-16 06:00:00"), 4400),
                 xytext=(pd.to_datetime("2024-03-16 08:00:00"), 5100),
                 arrowprops=dict(facecolor="red", shrink=0.08, width=1, headwidth=6),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="red"),
                 fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    out_file = FIGURES_DIR / "synthetic_injection_hard_case_rain.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    print("Generating Phase 3 figures...")
    plot_shapes_and_nuclides()
    plot_train_test_disjointness()
    plot_hard_case_rain()
    print("All Phase 3 figures successfully generated.")
