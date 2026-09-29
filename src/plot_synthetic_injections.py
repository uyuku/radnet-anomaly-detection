"""
Phase 3 Plotting Script (Revised): Visualizing Physically Grounded Synthetic Injections,
NaI(Tl) Multi-Nuclide Spectroscopy, Parameter Disjointness, and Rain-Coincident Hard-Case Signatures.

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

# Plot styling
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
WASHOUT_SHARES_PATH = Path("data/processed/rain_washout_spectral_shares.csv")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_shapes_and_nuclides():
    """Figure 1: Physical accumulation/retention profiles and NaI(Tl) multi-nuclide spectrometry."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Physical Profile S(t)
    ax1 = axes[0]
    t_pass = 14
    t_ret = 36
    d = t_pass + t_ret
    t = np.arange(d)

    # Inflow profiles
    # Train: linear ramp
    u = np.arange(t_pass)
    c_ramp = np.clip(u / 3.0, 0.0, 1.0)
    acc_ramp = np.cumsum(c_ramp)
    acc_ramp = acc_ramp / acc_ramp[-1]
    prof_ramp = np.zeros(d)
    prof_ramp[:t_pass] = acc_ramp
    prof_ramp[t_pass:] = 1.0  # Cs-137 plateau

    # Train: step arrival
    c_step = np.ones(t_pass)
    acc_step = np.cumsum(c_step)
    acc_step = acc_step / acc_step[-1]
    prof_step = np.zeros(d)
    prof_step[:t_pass] = acc_step
    prof_step[t_pass:] = 1.0

    # Test: sigmoidal
    c_sig = 1.0 / (1.0 + np.exp(-(u - 4.0) / 1.2))
    c_sig = (c_sig - c_sig[0]) / (c_sig[-1] - c_sig[0])
    acc_sig = np.cumsum(c_sig)
    acc_sig = acc_sig / acc_sig[-1]
    prof_sig = np.zeros(d)
    prof_sig[:t_pass] = acc_sig
    prof_sig[t_pass:] = 1.0

    # Test: exponential with I-131 decay (lambda = 0.00360/h)
    lam = 0.00360
    c_exp = 1.0 - np.exp(-u / 2.0)
    c_exp = c_exp / c_exp[-1]
    acc_exp = np.zeros(t_pass)
    for i in range(t_pass):
        decay_w = np.exp(-lam * (i - np.arange(i + 1)))
        acc_exp[i] = np.sum(c_exp[:i + 1] * decay_w)
    acc_exp = acc_exp / acc_exp[-1]
    prof_exp = np.zeros(d)
    prof_exp[:t_pass] = acc_exp
    for i in range(t_pass, d):
        prof_exp[i] = 1.0 * np.exp(-lam * (i - (t_pass - 1)))

    ax1.plot(t, prof_ramp, color="#1f77b4", linestyle="--", linewidth=2.5, label="Linear Inflow + Plateau (Train, Cs-137)")
    ax1.plot(t, prof_step, color="#2ca02c", linestyle=":", linewidth=2.5, label="Step Inflow + Plateau (Train, Cs-137)")
    ax1.plot(t, prof_sig, color="#ff7f0e", linestyle="-", linewidth=2.5, label="Sigmoidal Inflow + Plateau (Test, Cs-137)")
    ax1.plot(t, prof_exp, color="#9467bd", linestyle="-", linewidth=2.5, label="Exp Inflow + I-131 Decay (-12% over 36h) (Test)")

    # Mark plume passage vs retention vs filter change
    ax1.axvline(t_pass, color="gray", linestyle="-.", alpha=0.7)
    ax1.text(t_pass / 2, 0.15, "Plume Passage Phase\n(Cumulative Inflow)", ha="center", fontsize=9.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8f5e9", edgecolor="#2ca02c", alpha=0.8))
    ax1.text(t_pass + t_ret / 2, 0.15, "Retention Phase\n(Plateau / Radiological Decay)", ha="center", fontsize=9.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff3e0", edgecolor="#ff7f0e", alpha=0.8))
    
    # Filter change drop indicator
    ax1.annotate("Filter Replaced:\nActivity drops to 0",
                 xy=(d - 1, 1.0), xytext=(d - 10, 0.65),
                 arrowprops=dict(facecolor="red", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="red"))

    ax1.set_title("Panel A: Physical Filter Accumulation & Retention Profile $S(t) / S_{\\mathrm{peak}}$\n(Plume Inflow -> Filter Retention -> Filter Replacement)", fontweight="bold")
    ax1.set_xlabel("Elapsed Monitoring Time (hours)")
    ax1.set_ylabel("Normalized Filter Excess Signal")
    ax1.set_xlim(0, d + 2)
    ax1.set_ylim(-0.05, 1.25)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)

    # Panel B: NaI(Tl) Multi-Nuclide Energy Allocations
    ax2 = axes[1]
    channels = ["R02\n101-200", "R03\n201-400", "R04\n401-600", "R05\n601-800", "R06\n801-1000", "R07\n1001-1400", "R08\n1401-1800", "R09\n1801-2200"]
    x = np.arange(len(channels))
    width = 0.20

    # Radionuclide spectra (nominal)
    w_cs137 = [20.0, 15.0, 20.0, 43.0, 1.5, 0.5, 0.0, 0.0]
    w_i131 =  [25.0, 63.0, 5.0,  5.0,  1.5, 0.5, 0.0, 0.0]
    w_co60 =  [15.0, 15.0, 15.0, 15.0, 10.0, 28.0, 2.0, 0.0]

    # Empirical pooled washout shares from data/processed/rain_washout_spectral_shares.csv
    df_wash = pd.read_csv(WASHOUT_SHARES_PATH)
    pooled = df_wash[df_wash["Scope"].str.contains("Pooled")].iloc[0]
    w_washout = [
        pooled["cpm_r02_mean_pct"],
        pooled["cpm_r03_mean_pct"],
        pooled["cpm_r04_mean_pct"],
        pooled["cpm_r05_mean_pct"],
        pooled["cpm_r06_mean_pct"],
        pooled["cpm_r07_mean_pct"],
        pooled["cpm_r08_mean_pct"],
        pooled["cpm_r09_mean_pct"],
    ]

    rects1 = ax2.bar(x - 1.5 * width, w_cs137, width, label="Cs-137 (662 keV photopeak in R05)", color="#d62728", alpha=0.9)
    rects2 = ax2.bar(x - 0.5 * width, w_i131,  width, label="I-131 (365 keV photopeak in R03)", color="#ff7f0e", alpha=0.9)
    rects3 = ax2.bar(x + 0.5 * width, w_co60,  width, label="Co-60 (1173 & 1332 keV in R07!)", color="#9467bd", alpha=0.9)
    rects4 = ax2.bar(x + 1.5 * width, w_washout, width, label="Natural Radon Washout (Empirical, N=5,479h)", color="#1f77b4", alpha=0.9)

    ax2.set_title("Panel B: NaI(Tl) Multi-Nuclide Energy Allocations (% of Excess)\n(Co-60 Photopeaks in R07 Eliminate Artificial Zero Shortcut)", fontweight="bold")
    ax2.set_xlabel("RadNet Gamma Channel and Energy Boundary (keV)")
    ax2.set_ylabel("Share of Excess Plume Counts (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(channels)
    ax2.set_ylim(0, 75)

    # Highlight R07
    rect_box = patches.Rectangle((4.5, 0), 1.0, 72, linewidth=1.5, edgecolor="purple", facecolor="purple", alpha=0.08, linestyle="--")
    ax2.add_patch(rect_box)
    ax2.text(5.0, 62, "Channel R07 (1001-1400 keV):\nCo-60 = 28.0% (Photopeaks!)\nWashout = 3.8% (Bi-214)\nR07 > 0 is NOT unique to rain!",
             ha="center", va="top", color="#4a148c", fontsize=9.0, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3e5f5", edgecolor="purple", alpha=0.9))

    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.0)

    plt.tight_layout()
    out_file = FIGURES_DIR / "synthetic_injection_shapes_and_nuclides.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_train_test_disjointness():
    """Figure 2: Four-panel verification of parameter distributions and stress-test coverage."""
    df = pd.read_csv(CATALOG_PATH)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    test_std = test_df[test_df["environment"].str.contains("standard")]
    test_stress = test_df[test_df["environment"].str.contains("stress")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Duration
    ax1 = axes[0, 0]
    bins_dur = np.arange(0, 90, 4)
    ax1.hist(train_df["duration_hours"], bins=bins_dur, color="#1f77b4", alpha=0.7, label=f"Train (N={len(train_df)}): 10 to 34 h", edgecolor="black")
    ax1.hist(test_std["duration_hours"], bins=bins_dur, color="#ff7f0e", alpha=0.7, label=f"Test Standard (N={len(test_std)}): 36 to 80 h", edgecolor="black")
    ax1.hist(test_stress["duration_hours"], bins=bins_dur, color="#d62728", alpha=0.7, label=f"Test Stress Hard-Regime (N={len(test_stress)}): 8 to 20 h", edgecolor="black")
    ax1.set_title("Panel A: Duration Distribution (Hours)\n(Standard Evaluates Long Retention; Stress Tests Short Plumes)", fontweight="bold")
    ax1.set_xlabel("Injection Duration on Filter (hours)")
    ax1.set_ylabel("Number of Injections")
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel B: Peak Magnitude
    ax2 = axes[0, 1]
    bins_mag = np.linspace(200, 2600, 35)
    ax2.hist(train_df[train_df["magnitude_band"] == "Band_A_Low"]["peak_cpm"], bins=bins_mag, color="#1f77b4", alpha=0.7, label="Train Band A [250, 600] CPM", edgecolor="black")
    ax2.hist(train_df[train_df["magnitude_band"] == "Band_C_High"]["peak_cpm"], bins=bins_mag, color="#2ca02c", alpha=0.7, label="Train Band C [1400, 2500] CPM", edgecolor="black")
    ax2.hist(test_std["peak_cpm"], bins=bins_mag, color="#ff7f0e", alpha=0.7, label="Test Band B [700, 1200] CPM (Interpolation)", edgecolor="black")
    ax2.hist(test_stress["peak_cpm"], bins=bins_mag, color="#d62728", alpha=0.7, label="Test Stress Band A [250, 600] CPM (Subtle Plumes)", edgecolor="black")
    ax2.axvspan(600, 700, color="gray", alpha=0.15)
    ax2.axvspan(1200, 1400, color="gray", alpha=0.15, label="Disjoint Gaps (600-700 & 1200-1400)")
    ax2.set_title("Panel B: Peak Magnitude Distribution (CPM)\n(Interpolation Band B + Subtle Stress Band A)", fontweight="bold")
    ax2.set_xlabel("Peak Injected Signal (CPM)")
    ax2.set_ylabel("Number of Injections")
    ax2.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel C: Nuclide Fractions
    ax3 = axes[1, 0]
    bins_f = np.linspace(0.0, 1.0, 25)
    ax3.hist(train_df["fraction_cs137"], bins=bins_f, color="#1f77b4", alpha=0.7, label="Train Multi-Nuclide Balanced Mixes", edgecolor="black")
    ax3.hist(test_df[test_df["fraction_cs137"] >= 0.85]["fraction_cs137"], bins=bins_f, color="#d62728", alpha=0.7, label="Test Pure Cs-137 (f_Cs >= 0.85)", edgecolor="black")
    ax3.hist(test_df[test_df["fraction_co60"] >= 0.85]["fraction_co60"], bins=bins_f, color="#9467bd", alpha=0.7, label="Test Pure Co-60 (f_Co >= 0.85)", edgecolor="black")
    ax3.set_title("Panel C: Radionuclide Inventory Composition\n(Train Balanced Mixtures vs. Test Pure/Skewed Profiles)", fontweight="bold")
    ax3.set_xlabel("Radionuclide Activity Fraction")
    ax3.set_ylabel("Number of Injections")
    ax3.legend(loc="upper center", frameon=True, facecolor="white")

    # Panel D: Environment Breakdown
    ax4 = axes[1, 1]
    env_labels = [
        "Train Strictly Dry\n(N=175)",
        "Train Rain Onset\n(N=75)",
        "Test Std Dry\n(N=50)",
        "Test Std Rain\n(N=50)",
        "Test Stress Dry\n(N=50)",
        "Test Stress Rain\n(N=50)",
    ]
    env_counts = [
        len(train_df[train_df["environment"] == "train_strictly_dry"]),
        len(train_df[train_df["environment"] == "train_rain_coincident"]),
        len(test_df[test_df["environment"] == "test_standard_strictly_dry"]),
        len(test_df[test_df["environment"] == "test_standard_rain_onset"]),
        len(test_df[test_df["environment"] == "test_stress_strictly_dry"]),
        len(test_df[test_df["environment"] == "test_stress_hard_case_rain"]),
    ]
    colors = ["#1f77b4", "#0b559f", "#ff7f0e", "#d95f02", "#e377c2", "#d62728"]
    bars = ax4.bar(range(len(env_labels)), env_counts, color=colors, edgecolor="black")
    ax4.set_title("Panel D: Injection Partitioning by Environmental Regime\n(30% Train Rain, 50% Test Rain including Stress Set)", fontweight="bold")
    ax4.set_ylabel("Number of Injections")
    ax4.set_xticks(range(len(env_labels)))
    ax4.set_xticklabels(env_labels, rotation=20, ha="right", fontsize=9.0)
    ax4.set_ylim(0, 200)
    for bar, count in zip(bars, env_counts):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, str(count), ha="center", fontweight="bold")

    plt.tight_layout()
    out_file = FIGURES_DIR / "synthetic_injection_train_test_disjointness.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_hard_case_rain():
    """Figure 3: Detailed event timeline of INJ_0057 in Birmingham during a severe convective storm."""
    df = pd.read_csv("data/processed/labeled_al_birmingham_test.csv.gz")
    df["dt"] = pd.to_datetime(df["dt"])

    # Target window: 2024-05-09 18:00 to 2024-05-12 18:00 (72 hours around INJ_0057)
    t_start = pd.to_datetime("2024-05-09 18:00:00")
    t_end = pd.to_datetime("2024-05-12 18:00:00")
    sub = df[(df["dt"] >= t_start) & (df["dt"] <= t_end)].copy().reset_index(drop=True)

    # Thresholds for Birmingham
    dry_mask = (df["precip_24h"] == 0.0) & (df["precip_1h_mm"] == 0.0) & df["has_radnet_obs"] & df["rad_complete_channels"]
    mu_dry = df.loc[dry_mask, "gross_cpm"].mean()
    sigma_dry = df.loc[dry_mask, "gross_cpm"].std()
    thresh_3s = mu_dry + 3.0 * sigma_dry
    thresh_5s = mu_dry + 5.0 * sigma_dry

    mu_dose = df.loc[dry_mask, "dose_rate_nsvh"].mean()
    sigma_dose = df.loc[dry_mask, "dose_rate_nsvh"].std()
    thresh_dose_3s = mu_dose + 3.0 * sigma_dose

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [1, 2.2, 1.8, 1.8]})

    # Panel 1: Precipitation
    ax1 = axes[0]
    ax1.bar(sub["dt"], sub["precip_1h_mm"], width=0.04, color="#1f77b4", edgecolor="#0b559f", alpha=0.85, label="Hourly Precipitation Depth (mm/h)")
    ax1.set_ylabel("Rain\n(mm/h)", fontweight="bold")
    ax1.set_ylim(0, 22)
    ax1.set_title("Dedicated Hard-Case Injection (INJ_0057, Birmingham, AL) During Severe Convective Storm", fontweight="bold", fontsize=13)
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel 2: Gross CPM
    ax2 = axes[1]
    ax2.plot(sub["dt"], sub["gross_cpm"], color="#7f7f7f", linestyle="--", linewidth=1.8, label="Real Background + Natural Storm Washout")
    ax2.plot(sub["dt"], sub["inj_gross_cpm"], color="#d62728", linewidth=2.5, label="Combined Detector Signal (Washout + Synthetic Fission Plume)")
    ax2.plot(sub["dt"], sub["synthetic_excess_cpm"] + mu_dry, color="#ff7f0e", linestyle=":", linewidth=2.0, label="Synthetic Fission Plume Alone (S(t) + Baseline)")

    ax2.axhline(thresh_3s, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Fixed 3-sigma Alarm Threshold ({thresh_3s:.1f} CPM)")
    ax2.axhline(thresh_5s, color="#e377c2", linestyle="--", linewidth=1.5, label=f"Fixed 5-sigma Alarm Threshold ({thresh_5s:.1f} CPM)")

    # Highlight injection window
    inj_active = sub[sub["injection_active"]]
    if not inj_active.empty:
        ax2.axvspan(inj_active["dt"].iloc[0], inj_active["dt"].iloc[-1], color="#ffebee", alpha=0.4, label="Fission Plume Active (55h on filter)")

    ax2.set_ylabel("Gross CPM", fontweight="bold")
    ax2.set_ylim(3200, 6200)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", fontsize=9.0)

    # Panel 3: Ambient Dose Rate
    ax3 = axes[2]
    ax3.plot(sub["dt"], sub["dose_rate_nsvh"], color="#7f7f7f", linestyle="--", linewidth=1.8, label="Real Background Dose Rate (Washout Peak: 89 nSv/h)")
    ax3.plot(sub["dt"], sub["inj_dose_rate_nsvh"], color="#d62728", linewidth=2.2, label="Combined Injected Dose Rate (Calibrated k_dose = 0.0134 nSv/h per CPM)")
    ax3.axhline(thresh_dose_3s, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Baseline 3-sigma Dose Rate Threshold ({thresh_dose_3s:.1f} nSv/h)")
    ax3.set_ylabel("Dose Rate\n(nSv/h)", fontweight="bold")
    ax3.legend(loc="upper left", frameon=True, facecolor="white", fontsize=9.0)

    # Panel 4: Channels R05 (Cs-137) and R07 (Co-60 / Bi-214)
    ax4 = axes[3]
    ax4.plot(sub["dt"], sub["inj_cpm_r05"], color="#d62728", linewidth=2.0, label="Combined R05 (601-800 keV: Cs-137 Photopeak + Bi-214 Washout)")
    ax4.plot(sub["dt"], sub["inj_cpm_r07"], color="#1f77b4", linewidth=2.0, label="Combined R07 (1001-1400 keV: Bi-214 Washout + Co-60 Photopeaks)")
    ax4.plot(sub["dt"], sub["cpm_r07"], color="gray", linestyle=":", alpha=0.8, label="Raw R07 (Natural Washout Spike Alone)")
    ax4.set_ylabel("Channel CPM", fontweight="bold")
    ax4.set_xlabel("Date and Time (UTC, May 2024)", fontweight="bold")
    ax4.legend(loc="upper left", frameon=True, facecolor="white", fontsize=9.0)

    # Annotations
    ax2.annotate("Severe Convective Storm Peak\n(19.8 mm rain, +1,775 CPM natural)",
                 xy=(pd.to_datetime("2024-05-10 02:00:00"), 5592),
                 xytext=(pd.to_datetime("2024-05-09 20:00:00"), 5800),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#2ca02c"),
                 fontsize=9.0, fontweight="bold")

    ax2.annotate("Storm washout decays within 3 hours,\nbut Fission Plume accumulates & persists!",
                 xy=(pd.to_datetime("2024-05-10 14:00:00"), 4473),
                 xytext=(pd.to_datetime("2024-05-10 16:00:00"), 5200),
                 arrowprops=dict(facecolor="red", shrink=0.08, width=1, headwidth=6),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="red"),
                 fontsize=9.0, fontweight="bold")

    plt.tight_layout()
    out_file = FIGURES_DIR / "synthetic_injection_hard_case_rain.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    print("Generating revised Phase 3 figures...")
    plot_shapes_and_nuclides()
    plot_train_test_disjointness()
    plot_hard_case_rain()
    print("All revised Phase 3 figures successfully generated.")
