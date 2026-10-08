"""
Generates publication-quality figures for Phase 3 (Synthetic Injection Design):
1. reports/figures/synthetic_injection_shapes_and_nuclides.png:
   - Panel A: Filter accumulation inflow, retention plateau, radiological decay, and filter change drop.
   - Panel B: NaI(Tl) channel energy shares comparing Cs-137, I-131, Co-60, Cs-134, and empirical radon washout ratio of sums.
2. reports/figures/synthetic_injection_train_test_disjointness.png:
   - Four-panel validation of injection durations, peak magnitudes, operational nuclide scenarios, and environmental regimes.
3. reports/figures/synthetic_injection_hard_case_rain.png:
   - Event timeline of INJ_0067 in Birmingham, AL during a severe convective storm (13.5 mm/h rain),
     demonstrating concurrent rain onset, pure Cs-137 plume (no R07 photopeak signature),
     calibrated dose-rate response, persistent particulate filter retention, and synchronization with a real physical filter change drop.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_PATH = Path("data/processed/synthetic_injection_catalog.csv")
WASHOUT_SHARES_PATH = Path("data/processed/rain_washout_spectral_shares.csv")


def plot_shapes_and_nuclides():
    """Figure 1: Physical Filter Accumulation Profiles & Multi-Nuclide Spectroscopy."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Physical Filter Profile S(t)
    ax1 = axes[0]
    t_pass = 14
    t_ret = 36
    d = t_pass + t_ret
    t = np.arange(d)
    u = np.arange(t_pass)

    # Train: linear ramp arrival
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

    ax1.axvline(t_pass, color="gray", linestyle="-.", alpha=0.7)
    ax1.text(t_pass / 2, 0.15, "Plume Passage Phase\n(Cumulative Inflow)", ha="center", fontsize=9.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8f5e9", edgecolor="#2ca02c", alpha=0.8))
    ax1.text(t_pass + t_ret / 2, 0.15, "Retention Phase\n(Plateau / Radiological Decay)", ha="center", fontsize=9.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff3e0", edgecolor="#ff7f0e", alpha=0.8))
    
    ax1.plot([d, d], [1.0, 0.0], color="#ff7f0e", linestyle="--", linewidth=2.0)
    ax1.plot([d, d + 4], [0.0, 0.0], color="#ff7f0e", linestyle="-", linewidth=2.0)
    ax1.axvline(d, color="red", linestyle=":", alpha=0.7, linewidth=1.5)
    
    ax1.annotate("Real Filter Replacement Drop:\nDeposited activity clears to 0",
                 xy=(d, 0.05), xytext=(d - 18, 0.38),
                 arrowprops=dict(facecolor="red", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="red"))

    ax1.set_title("Panel A: Physical Filter Accumulation & Retention Profile $S(t) / S_{\\mathrm{peak}}$\n(Plume Inflow -> Filter Retention -> Filter Replacement Drop)", fontweight="bold")
    ax1.set_xlabel("Elapsed Monitoring Time (hours)")
    ax1.set_ylabel("Normalized Filter Excess Signal")
    ax1.set_xlim(0, d + 4)
    ax1.set_ylim(-0.05, 1.25)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.0)

    # Panel B: NaI(Tl) Multi-Nuclide Energy Allocations
    ax2 = axes[1]
    channels = ["R02\n101-200", "R03\n201-400", "R04\n401-600", "R05\n601-800", "R06\n801-1000", "R07\n1001-1400", "R08\n1401-1800", "R09\n1801-2200"]
    x = np.arange(len(channels))
    width = 0.16

    # Radionuclide spectra (operational templates)
    w_cs137 = [20.0, 15.0, 20.0, 43.0, 1.5, 0.9, 0.4, 0.0]
    w_i131 =  [25.0, 63.0, 5.0,  5.0,  1.5, 0.8, 0.5, 0.0]
    w_co60 =  [15.0, 15.0, 15.0, 15.0, 10.0, 28.0, 2.0, 0.0]
    w_cs134 = [18.0, 20.0, 15.0, 35.0, 6.0,  4.5, 0.5, 0.0]

    # Empirical pooled washout shares (ratio of sums)
    df_wash = pd.read_csv(WASHOUT_SHARES_PATH)
    pooled = df_wash[df_wash["Scope"].str.contains("Pooled")].iloc[0]
    w_washout = [
        pooled["cpm_r02_ratio_of_sums_pct"],
        pooled["cpm_r03_ratio_of_sums_pct"],
        pooled["cpm_r04_ratio_of_sums_pct"],
        pooled["cpm_r05_ratio_of_sums_pct"],
        pooled["cpm_r06_ratio_of_sums_pct"],
        pooled["cpm_r07_ratio_of_sums_pct"],
        pooled["cpm_r08_ratio_of_sums_pct"],
        pooled["cpm_r09_ratio_of_sums_pct"],
    ]

    rects1 = ax2.bar(x - 2.0 * width, w_cs137, width, label="Pure Cs-137 (662 keV photopeak in R05)", color="#d62728", alpha=0.9)
    rects2 = ax2.bar(x - 1.0 * width, w_i131,  width, label="Pure I-131 (364 keV photopeak in R03)", color="#ff7f0e", alpha=0.9)
    rects3 = ax2.bar(x,               w_cs134, width, label="Fukushima Reactor (Cs-134 lines in R07)", color="#2ca02c", alpha=0.9)
    rects4 = ax2.bar(x + 1.0 * width, w_co60,  width, label="Orphan Co-60 (1173 & 1332 keV in R07)", color="#9467bd", alpha=0.9)
    rects5 = ax2.bar(x + 2.0 * width, w_washout, width, label=f"Radon Washout (Ratio-of-Sums, R07+R08={pooled['r07_plus_r08_ratio_of_sums_pct']}%)", color="#1f77b4", alpha=0.9)

    ax2.set_title("Panel B: Gamma Channel Branching Shares (% of Excess Counts)\n(Empirical Washout Ratio-of-Sums vs. Operational Anthropogenic Scenarios)", fontweight="bold")
    ax2.set_xlabel("RadNet Energy Channel (keV Window)")
    ax2.set_ylabel("Share of Excess Net Gross Counts (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(channels, fontsize=8.5)
    ax2.set_ylim(0, 80)

    # Highlight R07
    rect_box = patches.Rectangle((4.5, 0), 1.0, 78, linewidth=1.5, edgecolor="purple", facecolor="purple", alpha=0.08, linestyle="--")
    ax2.add_patch(rect_box)
    ax2.annotate(
        f"Channel R07 (1001-1400 keV):\n• Co-60: 28.0% (Photopeaks)\n• Washout: {pooled['cpm_r07_ratio_of_sums_pct']:.1f}% (Bi-214)\n• Fukushima: 4.5% (Cs-134)\n• Cs-137 / I-131: < 1.0%",
        xy=(5.0, 28),
        xytext=(2.6, 58),
        arrowprops=dict(facecolor="purple", edgecolor="purple", shrink=0.08, width=1.5, headwidth=6),
        ha="center", va="center", color="#4a148c", fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f3e5f5", edgecolor="purple", alpha=0.92)
    )

    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)

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
    bins_dur = np.arange(0, 180, 6)
    ax1.hist(train_df["duration_hours"], bins=bins_dur, color="#1f77b4", alpha=0.7, label=f"Train (N={len(train_df)}): 10 to 94 h", edgecolor="black")
    ax1.hist(test_std["duration_hours"], bins=bins_dur, color="#ff7f0e", alpha=0.7, label=f"Test Standard (N={len(test_std)}): 36 to 164 h", edgecolor="black")
    ax1.hist(test_stress["duration_hours"], bins=bins_dur, color="#d62728", alpha=0.7, label=f"Test Stress Hard-Regime (N={len(test_stress)}): 8 to 140 h", edgecolor="black")
    ax1.set_title("Panel A: Realized Duration Distribution (Hours)\n(All Injections Synchronized to Real Physical Filter Drops)", fontweight="bold")
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
    ax2.axvspan(1200, 1400, color="gray", alpha=0.15, label="Disjoint Magnitude Gaps")
    ax2.set_title("Panel B: Peak Magnitude Distribution (CPM)\n(Interpolation Band B + Subtle Stress Band A)", fontweight="bold")
    ax2.set_xlabel("Peak Injected Signal (CPM)")
    ax2.set_ylabel("Number of Injections")
    ax2.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel C: Operational Nuclide Scenarios
    ax3 = axes[1, 0]
    scen_counts = test_df["nuclide_scenario"].value_counts()
    scen_names = {
        "fission_reactor_fukushima": "Reactor Fission\n(Cs-137+134+I-131)",
        "fission_pure_cs137": "Legacy Source\n(Pure Cs-137)",
        "fission_pure_i131": "Radiopharma\n(Pure I-131)",
        "activation_orphan_co60": "Orphan Source\n(Co-60)",
        "mixed_fission_activation": "Core Excursion\n(Mixed All 4)",
    }
    scen_labels = [scen_names.get(k, k) for k in scen_counts.index]
    colors_scen = ["#ff7f0e", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    bars_sc = ax3.bar(range(len(scen_counts)), scen_counts.values, color=colors_scen, edgecolor="black")
    ax3.set_title("Panel C: Operational Release Scenarios in Test Set (N=200)\n(Exactly Balanced at 20.0% Across All Environments)", fontweight="bold")
    ax3.set_ylabel("Number of Injections")
    ax3.set_xticks(range(len(scen_counts)))
    ax3.set_xticklabels(scen_labels, fontsize=9.0)
    ax3.set_ylim(0, 60)
    for bar, count in zip(bars_sc, scen_counts.values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, str(count), ha="center", fontweight="bold")

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
        len(test_df[test_df["environment"] == "test_standard_rain"]),
        len(test_df[test_df["environment"] == "test_stress_strictly_dry"]),
        len(test_df[test_df["environment"] == "test_stress_rain"]),
    ]
    colors = ["#1f77b4", "#0b559f", "#ff7f0e", "#d95f02", "#e377c2", "#d62728"]
    bars = ax4.bar(range(len(env_labels)), env_counts, color=colors, edgecolor="black")
    ax4.set_title("Panel D: Injection Partitioning by Environmental Regime\n(30% Train Rain, 50% Test Rain with Strict 48h Separation)", fontweight="bold")
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
    """
    Figure 3: Detailed event timeline of INJ_0067 in Birmingham, AL during a severe convective storm.
    Demonstrates the challenging regime requested by Claude Review (Item 2.3):
    A subtle (Band A: 479 CPM) pure Cs-137 plume rising concurrently during a 13.5 mm/h rainstorm.
    Features:
    - Pure Cs-137: Prominent 662 keV photopeak in R05, ZERO photopeak in R07.
    - Concurrent rise with natural radon washout surge.
    - Persistent filter retention after rain ceases (+479 CPM plateau).
    - Synchronized drop at 2023-01-28 17:00 UTC at a verified physical filter replacement drop.
    """
    df = pd.read_csv("data/processed/labeled_al_birmingham_test.csv.gz")
    df["dt"] = pd.to_datetime(df["dt"])

    # Target window: 2023-01-24 18:00 to 2023-01-29 06:00 (108 hours around INJ_0067)
    t_start = pd.to_datetime("2023-01-24 18:00:00")
    t_end = pd.to_datetime("2023-01-29 06:00:00")
    sub = df[(df["dt"] >= t_start) & (df["dt"] <= t_end)].copy().reset_index(drop=True)

    # Baseline thresholds for Birmingham
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
    ax1.bar(sub["dt"], sub["precip_1h_mm"], width=0.035, color="#1f77b4", edgecolor="#0b559f", alpha=0.85, label="Hourly Precipitation Depth (mm/h)")
    ax1.set_ylabel("Rain\n(mm/h)", fontweight="bold")
    ax1.set_ylim(0, 16)
    ax1.set_title("Dedicated Hard-Case Injection (INJ_0067, Birmingham, AL) During Severe Convective Storm\n(Concurrent Rain Onset, 13.5 mm/h Storm Peak, Pure Cs-137 Plume without R07 Marker, and Real Filter Drop Reset)", fontweight="bold", fontsize=12)
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel 2: Gross CPM
    ax2 = axes[1]
    ax2.plot(sub["dt"], sub["gross_cpm"], color="#7f7f7f", linestyle="--", linewidth=1.8, label="Real Background + Natural Storm Washout (Peak: 4,896.0 CPM)")
    ax2.plot(sub["dt"], sub["inj_gross_cpm"], color="#d62728", linewidth=2.5, label="Combined Detector Signal (Peak: 4,992.4 CPM)")
    ax2.plot(sub["dt"], sub["synthetic_excess_cpm"] + mu_dry, color="#ff7f0e", linestyle=":", linewidth=2.0, label="Synthetic Cs-137 Plume Alone (Plateau: +479.1 CPM)")

    ax2.axhline(thresh_3s, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Fixed 3-sigma Alarm Threshold ({thresh_3s:.1f} CPM)")
    ax2.axhline(thresh_5s, color="#e377c2", linestyle="--", linewidth=1.5, label=f"Fixed 5-sigma Alarm Threshold ({thresh_5s:.1f} CPM)")

    # Highlight injection window
    inj_active = sub[sub["injection_active"]]
    if not inj_active.empty:
        ax2.axvspan(inj_active["dt"].iloc[0], inj_active["dt"].iloc[-1], color="#ffebee", alpha=0.4, label="Cs-137 Plume Active on Filter (83h duration until filter change)")

    ax2.set_ylabel("Gross CPM", fontweight="bold")
    ax2.set_ylim(3200, 5600)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)

    # Panel 3: Ambient Dose Rate
    ax3 = axes[2]
    ax3.plot(sub["dt"], sub["dose_rate_nsvh"], color="#7f7f7f", linestyle="--", linewidth=1.8, label="Real Background Dose Rate (Storm Washout Peak: 80.0 nSv/h)")
    ax3.plot(sub["dt"], sub["inj_dose_rate_nsvh"], color="#d62728", linewidth=2.2, label="Combined Injected Dose Rate (Calibrated k_dose = 0.0145 nSv/h per CPM, Peak: 82.2 nSv/h)")
    ax3.axhline(thresh_dose_3s, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Baseline 3-sigma Dose Rate Threshold ({thresh_dose_3s:.1f} nSv/h)")
    ax3.set_ylabel("Dose Rate\n(nSv/h)", fontweight="bold")
    ax3.set_ylim(45, 95)
    ax3.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)

    # Panel 4: Channels R05 and R07 (Cs-137 662 keV photopeak in R05 vs lack of R07 marker)
    ax4 = axes[3]
    ax4.plot(sub["dt"], sub["inj_cpm_r05"], color="#d62728", linewidth=2.2, label="Combined R05 (601-800 keV: Cs-137 662 keV Photopeak + Compton)")
    ax4.plot(sub["dt"], sub["cpm_r05"], color="#ff7f0e", linestyle=":", linewidth=1.8, label="Raw R05 (Natural Storm Washout Alone)")
    ax4.plot(sub["dt"], sub["inj_cpm_r07"], color="#9467bd", linewidth=1.8, linestyle="--", label="Combined R07 (1001-1400 keV: Natural Bi-214 Washout ONLY; No Cs-137 Photopeak)")
    ax4.plot(sub["dt"], sub["cpm_r07"], color="gray", linestyle=":", linewidth=1.5, label="Raw R07 (Natural Storm Washout Alone)")
    ax4.set_ylabel("Channel CPM", fontweight="bold")
    ax4.set_xlabel("Date and Time (UTC, January 2023)", fontweight="bold")
    ax4.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)

    # Annotations
    ax2.annotate("Severe Convective Storm Peak\n(13.5 mm/h rain, 4,896 CPM gross washout)",
                 xy=(pd.to_datetime("2023-01-25 09:00:00"), 4896),
                 xytext=(pd.to_datetime("2023-01-25 01:00:00"), 5350),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#2ca02c"),
                 fontsize=8.5, fontweight="bold")

    ax2.annotate("Storm rain ceases;\nWashout decays back to baseline, but particulate\nCs-137 persists on filter (+479.1 CPM)!",
                 xy=(pd.to_datetime("2023-01-26 12:00:00"), 4280),
                 xytext=(pd.to_datetime("2023-01-26 16:00:00"), 4800),
                 arrowprops=dict(facecolor="#ff7f0e", shrink=0.08, width=1, headwidth=6),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0", edgecolor="#ff7f0e"),
                 fontsize=8.5, fontweight="bold")

    ax2.annotate("Verified Real Filter Replacement Drop:\nActivity clears as background steps from 3885 to 3578 CPM!\n(Zero synthetic-only drop cliff)",
                 xy=(pd.to_datetime("2023-01-28 17:00:00"), 4057),
                 xytext=(pd.to_datetime("2023-01-27 12:00:00"), 3500),
                 arrowprops=dict(facecolor="red", shrink=0.08, width=1.5, headwidth=6),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="red"),
                 fontsize=8.5, fontweight="bold")

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
