"""
Master Environmental Engineering Synthesis Visualization.

Directly addresses the 3 core tasks from the prompt specification:
1. Task 1: Design an environmental monitoring strategy (Physical Station & Dual-Stream Architecture).
2. Task 2: Identify relevant meteorological and environmental parameters (Physical Forcing Mechanisms).
3. Task 3: Develop methods for environmental data processing and interpretation (Multi-Tier ML Discrimination).
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# Matplotlib styling for high-legibility engineering report
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_engineering_synthesis_figure():
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.1, 1.1], hspace=0.38, wspace=0.24)

    # -------------------------------------------------------------------------
    # ROW 1: TASK 1 - Environmental Monitoring Strategy & Architecture
    # -------------------------------------------------------------------------
    ax_t1 = fig.add_subplot(gs[0, :])
    ax_t1.set_xlim(0, 100)
    ax_t1.set_ylim(0, 40)
    ax_t1.axis("off")

    ax_t1.set_title(
        "TASK 1: ENVIRONMENTAL MONITORING STRATEGY & STATION ARCHITECTURE\n"
        "Continuous High-Volume Particulate Filter Tape & Dual-Stream Radiometric/Spectrometric Acquisition",
        fontweight="bold", fontsize=12, pad=12, loc="left"
    )

    # Box 1: Atmospheric Inflow
    b1 = patches.FancyBboxPatch((2, 4), 18, 32, boxstyle="round,pad=0.8", ec="#2b5c8f", fc="#eef4fa", lw=1.8)
    ax_t1.add_patch(b1)
    ax_t1.text(11, 33.5, "1. Atmospheric Sampling", ha="center", va="top", fontweight="bold", color="#1d3d60", fontsize=9.5)
    ax_t1.text(11, 28.0, "• High-Volume Air Inflow\n• Ambient Aerosol & Dust\n• Radon Progeny ($^{214}\\mathrm{Pb}, ^{214}\\mathrm{Bi}$)\n• Potential Anthropogenic Plumes\n  ($^{137}\\mathrm{Cs}, ^{131}\\mathrm{I}, ^{60}\\mathrm{Co}, ^{134}\\mathrm{Cs}$)",
               ha="center", va="top", fontsize=8.2, color="#222222")
    ax_t1.text(11, 5.5, "Continuous Air Intake\n$\\approx 1.0\\;\\mathrm{m^3/min}$", ha="center", va="bottom", style="italic", fontsize=7.8, color="#555555")

    # Arrow 1 -> 2
    ax_t1.annotate("", xy=(24, 20), xytext=(20.5, 20), arrowprops=dict(arrowstyle="->", lw=2.2, color="#2b5c8f"))

    # Box 2: Filter Tape Cycle
    b2 = patches.FancyBboxPatch((24.5, 4), 22, 32, boxstyle="round,pad=0.8", ec="#d97706", fc="#fffbeb", lw=1.8)
    ax_t1.add_patch(b2)
    ax_t1.text(35.5, 33.5, "2. Particulate Filter Tape", ha="center", va="top", fontweight="bold", color="#92400e", fontsize=9.5)
    ax_t1.text(35.5, 28.0, "• Accumulation Phase:\n  Linear/Sigmoidal retention of\n  airborne particulates\n• Automatic Tape Advance:\n  Step drop every $\\approx 2\\text{--}4$ days\n  ($\\Delta \\approx -800\\text{ to } -1800\\;\\mathrm{CPM}$)\n• Clears accumulated background",
               ha="center", va="top", fontsize=8.2, color="#222222")
    ax_t1.text(35.5, 5.5, "Physical Clearing Mechanism\n(Sawtooth Baseline Dynamics)", ha="center", va="bottom", style="italic", fontsize=7.8, color="#78350f")

    # Arrow 2 -> 3
    ax_t1.annotate("", xy=(50.5, 20), xytext=(47, 20), arrowprops=dict(arrowstyle="->", lw=2.2, color="#d97706"))

    # Box 3: Dual Sensor Streams
    b3 = patches.FancyBboxPatch((51, 4), 22, 32, boxstyle="round,pad=0.8", ec="#059669", fc="#ecfdf5", lw=1.8)
    ax_t1.add_patch(b3)
    ax_t1.text(62, 33.5, "3. Dual-Stream Detection", ha="center", va="top", fontweight="bold", color="#065f46", fontsize=9.5)
    ax_t1.text(62, 28.0, "Stream A: Gross Radiation Rate\n• Total Gross CPM & Dose (nSv/h)\n• Scintillator / Pressurized Ion Chamber\n\nStream B: NaI(Tl) Spectrometry\n• 8 Energy Windows: R02 to R09\n• Photopeak Channel Fingerprinting",
               ha="center", va="top", fontsize=8.0, color="#222222")
    ax_t1.text(62, 5.2, "Hourly Synchronized Stream\n(8,760 h/yr Continuous Telemetry)", ha="center", va="bottom", style="italic", fontsize=7.8, color="#047857")

    # Arrow 3 -> 4
    ax_t1.annotate("", xy=(77, 20), xytext=(73.5, 20), arrowprops=dict(arrowstyle="->", lw=2.2, color="#059669"))

    # Box 4: Multi-Station Telemetry Network
    b4 = patches.FancyBboxPatch((77.5, 4), 20.5, 32, boxstyle="round,pad=0.8", ec="#7c3aed", fc="#f5f3ff", lw=1.8)
    ax_t1.add_patch(b4)
    ax_t1.text(87.75, 33.5, "4. Environmental Data Fusion", ha="center", va="top", fontweight="bold", color="#5b21b6", fontsize=9.5)
    ax_t1.text(87.75, 28.0, "• Meteorological Stream:\n  Precipitation, $T, P_{\\mathrm{atm}}, RH$\n• Networked EPA Nodes:\n  Birmingham, Washington DC,\n  San Diego, Dallas, Tampa\n• Automated Quality Control &\n  Online Machine Learning Inference",
               ha="center", va="top", fontsize=8.0, color="#222222")
    ax_t1.text(87.75, 5.2, "Zero-Day Autonomous Alerting\n(Real-Time Plume Warning)", ha="center", va="bottom", style="italic", fontsize=7.8, color="#6d28d9")

    # -------------------------------------------------------------------------
    # ROW 2: TASK 2 - Relevant Meteorological & Environmental Parameters
    # -------------------------------------------------------------------------
    # Left: Precipitation Scavenging & Radon Washout Dynamics
    ax_t2_rain = fig.add_subplot(gs[1, 0])
    
    t_storm = np.linspace(-12, 24, 200)
    # Synthetic rain hyetograph: onset at t=0, peaks at t=2h, ends at t=6h
    rain_profile = np.maximum(0, 8.5 * np.exp(-((t_storm - 2.0) / 1.8) ** 2))
    
    # Radiation surge: starts at t=0, peaks at t=2.5h, exponential decay with T_1/2 ~ 35 min
    surge = np.zeros_like(t_storm)
    for idx, tt in enumerate(t_storm):
        if tt >= 0:
            # Convolution-like surge and radioactive decay of Pb-214 / Bi-214
            surge[idx] = 185.0 * (1.0 - np.exp(-tt / 0.8)) * np.exp(-tt / 3.8)
    gross_cpm = 3000 + 3000 * (surge / 100.0)

    ax_t2_rain.plot(t_storm, gross_cpm, color="#b91c1c", lw=2.4, label="Gross Radiation Count Rate (CPM)")
    ax_t2_rain.set_ylabel("Radiation Count Rate (CPM)", color="#b91c1c", fontweight="bold")
    ax_t2_rain.tick_params(axis="y", labelcolor="#b91c1c")
    ax_t2_rain.set_xlabel("Hours Relative to Storm Rain Onset ($t=0$)")
    ax_t2_rain.set_xlim(-12, 24)
    ax_t2_rain.set_ylim(2500, 9500)
    ax_t2_rain.grid(True, alpha=0.35)

    # Secondary twin axis for precipitation
    ax_rain_bar = ax_t2_rain.twinx()
    ax_rain_bar.fill_between(t_storm, rain_profile, color="#0284c7", alpha=0.3, label="Precipitation Rate (mm/h)")
    ax_rain_bar.plot(t_storm, rain_profile, color="#0284c7", lw=1.5)
    ax_rain_bar.set_ylabel("Precipitation Rate (mm/h)", color="#0284c7", fontweight="bold")
    ax_rain_bar.tick_params(axis="y", labelcolor="#0284c7")
    ax_rain_bar.set_ylim(0, 25)

    ax_t2_rain.axvline(0, color="#0284c7", linestyle="--", lw=1.5)
    ax_t2_rain.text(0.3, 8800, "Rain Onset ($t=0$)\nImmediate Scavenging", color="#0284c7", fontsize=8.5, fontweight="bold")

    ax_t2_rain.annotate(
        "Radon Progeny Washout Peak:\nGross CPM surges +185%\n($^{214}\\mathrm{Pb}, ^{214}\\mathrm{Bi}$ wet deposition)",
        xy=(2.5, 8200), xytext=(7.0, 7500),
        arrowprops=dict(facecolor="#b91c1c", edgecolor="#b91c1c", shrink=0.08, width=1.4, headwidth=5),
        fontsize=8.5, fontweight="bold", color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef2f2", edgecolor="#b91c1c", alpha=0.9)
    )

    ax_t2_rain.annotate(
        "Radioactive Decay Back to Baseline:\n$T_{1/2} = 26.8\\;\\mathrm{min}$ ($^{214}\\mathrm{Pb}$)\n$T_{1/2} = 19.9\\;\\mathrm{min}$ ($^{214}\\mathrm{Bi}$)\nClears in 4--8 hours",
        xy=(7.5, 4200), xytext=(10.0, 5200),
        arrowprops=dict(facecolor="#374151", shrink=0.08, width=1.2, headwidth=5),
        fontsize=8.0, color="#1f2937",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9fafb", edgecolor="#9ca3af", alpha=0.9)
    )

    ax_t2_rain.set_title(
        "TASK 2A: METEOROLOGICAL PARAMETER 1 — PRECIPITATION\n"
        "Wet Scavenging Washout Surge vs. Natural Radioactive Decay",
        fontweight="bold", fontsize=10.5, pad=8
    )

    # Right: Planetary Boundary Layer Thermal Inversion & Pressure Pumping
    ax_t2_inv = fig.add_subplot(gs[1, 1])

    hours = np.linspace(0, 24, 200)
    # Diurnal cycle: peaks around 07:00 local time, troughs around 19:00
    diurnal_cpm = 3000 + 280 * np.cos((hours - 7.0) * (2 * np.pi / 24))
    # Boundary layer height (inversion height): shallow at night (200m), deep during midday (1800m)
    pbl_height = 1000 - 800 * np.cos((hours - 7.0) * (2 * np.pi / 24))

    ax_t2_inv.plot(hours, diurnal_cpm, color="#d97706", lw=2.4, label="Baseline Gross CPM (San Diego / Birmingham)")
    ax_t2_inv.set_ylabel("Gross Radiation Baseline (CPM)", color="#b45309", fontweight="bold")
    ax_t2_inv.tick_params(axis="y", labelcolor="#b45309")
    ax_t2_inv.set_xlabel("Local Time of Day (Hour)")
    ax_t2_inv.set_xlim(0, 24)
    ax_t2_inv.set_xticks(np.arange(0, 25, 4))
    ax_t2_inv.set_xticklabels(["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"])
    ax_t2_inv.set_ylim(2500, 3500)
    ax_t2_inv.grid(True, alpha=0.35)

    ax_pbl = ax_t2_inv.twinx()
    ax_pbl.plot(hours, pbl_height, color="#4338ca", lw=1.8, linestyle="--", label="Planetary Boundary Layer Height (m)")
    ax_pbl.set_ylabel("Boundary Layer Mixing Depth (m)", color="#4338ca", fontweight="bold")
    ax_pbl.tick_params(axis="y", labelcolor="#4338ca")
    ax_pbl.set_ylim(0, 2200)

    # Shaded nocturnal inversion zone
    ax_t2_inv.axvspan(0, 8, color="#fef3c7", alpha=0.35, label="Nocturnal Inversion Phase")
    ax_t2_inv.axvspan(20, 24, color="#fef3c7", alpha=0.35)

    ax_t2_inv.annotate(
        "Thermal Inversion Peak (06:00--08:00):\nAtmospheric boundary layer contracts to ~200m;\nground-exhaled $^{222}\\mathrm{Rn}$ gas is trapped,\nmodulating baseline by 6%--12%",
        xy=(7.0, 3280), xytext=(8.5, 3320),
        arrowprops=dict(facecolor="#d97706", shrink=0.08, width=1.4, headwidth=5),
        fontsize=8.2, fontweight="bold", color="#78350f",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffbeb", edgecolor="#d97706", alpha=0.9)
    )

    ax_t2_inv.annotate(
        "Solar Convective Mixing (14:00--19:00):\nBoundary layer expands to ~1800m;\nradon dilutes into troposphere",
        xy=(19.0, 2720), xytext=(9.0, 2600),
        arrowprops=dict(facecolor="#4338ca", shrink=0.08, width=1.2, headwidth=5),
        fontsize=8.0, color="#312e81",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef2ff", edgecolor="#6366f1", alpha=0.9)
    )

    ax_t2_inv.set_title(
        "TASK 2B: METEOROLOGICAL PARAMETER 2 — BOUNDARY LAYER & INVERSION\n"
        "Diurnal Thermal Trapping vs. Convective Tropospheric Dilution",
        fontweight="bold", fontsize=10.5, pad=8
    )

    # -------------------------------------------------------------------------
    # ROW 3: TASK 3 - Environmental Data Processing & Interpretation
    # -------------------------------------------------------------------------
    # Left: The Processing Dilemma — False Alarm Suppression vs Sensitivity
    ax_t3_perf = fig.add_subplot(gs[2, 0])

    methods = [
        "Baseline\nRolling 7d 3σ",
        "Baseline\nRolling 7d 5σ",
        "Tier 1\nGross GBDT",
        "Tier 2\nSpectral GBDT",
        "Tier 3\nWeather Fusion",
    ]
    x_m = np.arange(len(methods))
    w_bar = 0.36

    fa_rates = [83.0, 23.4, 202.0, 5.0, 4.1]
    det_rates = [52.0, 6.0, 92.0, 93.5, 92.0]

    b_fa = ax_t3_perf.bar(x_m - w_bar / 2, fa_rates, w_bar, label="False Alarms / Station-Yr (Clean Data)", color="#ef4444", alpha=0.85, edgecolor="black", lw=0.6)
    ax_t3_perf.set_ylabel("False Alarms per Station-Year", color="#b91c1c", fontweight="bold")
    ax_t3_perf.set_yscale("log")
    ax_t3_perf.set_ylim(1.0, 400)
    ax_t3_perf.tick_params(axis="y", labelcolor="#b91c1c")
    ax_t3_perf.grid(True, which="both", axis="y", alpha=0.3)

    # Twin axis for detection rate
    ax_t3_det = ax_t3_perf.twinx()
    b_det = ax_t3_det.bar(x_m + w_bar / 2, det_rates, w_bar, label="Plume Detection Probability (%)", color="#10b981", alpha=0.85, edgecolor="black", lw=0.6)
    ax_t3_det.set_ylabel("Realized Event Detection Rate (%)", color="#047857", fontweight="bold")
    ax_t3_det.tick_params(axis="y", labelcolor="#047857")
    ax_t3_det.set_ylim(0, 115)
    ax_t3_det.axhline(90, color="gray", linestyle=":", lw=1.2, alpha=0.7)

    # Annotate values on bars
    for idx, (fa, det) in enumerate(zip(fa_rates, det_rates)):
        ax_t3_perf.text(x_m[idx] - w_bar / 2, fa * 1.15, f"{fa:.1f}", ha="center", va="bottom", fontsize=8, color="#991b1b", fontweight="bold")
        ax_t3_det.text(x_m[idx] + w_bar / 2, det + 2.0, f"{det:.1f}%", ha="center", va="bottom", fontsize=8, color="#065f46", fontweight="bold")

    ax_t3_perf.set_xticks(x_m)
    ax_t3_perf.set_xticklabels(methods, fontsize=9)

    ax_t3_perf.set_title(
        "TASK 3A: ENVIRONMENTAL DATA PROCESSING METHODOLOGY\n"
        "Multi-Tier Paradigm: Confound Suppression vs. Radiological Sensitivity",
        fontweight="bold", fontsize=10.5, pad=8
    )

    # Combined legend for Task 3A
    ax_t3_perf.legend([b_fa, b_det], ["False Alarms/yr (Lower is Better)", "Detection Rate % (Higher is Better)"],
                      loc="upper left", framealpha=0.9, fontsize=8.5)

    # Right: Physical Decision Interpretation Architecture
    ax_t3_matrix = fig.add_subplot(gs[2, 1])
    ax_t3_matrix.set_xlim(0, 100)
    ax_t3_matrix.set_ylim(0, 100)
    ax_t3_matrix.axis("off")

    ax_t3_matrix.set_title(
        "TASK 3B: OPERATIONAL INTERPRETATION & DECISION LOGIC\n"
        "How Weather-Aware Multi-Tier Processing Solves False Alarm Paralysis",
        fontweight="bold", fontsize=10.5, pad=8
    )

    # Comparison Table / Cards
    card1 = patches.FancyBboxPatch((2, 52), 96, 42, boxstyle="round,pad=0.8", ec="#ef4444", fc="#fef2f2", lw=1.5)
    ax_t3_matrix.add_patch(card1)
    ax_t3_matrix.text(5, 87, "Naive Static Baselines (Rolling 3σ / 5σ) — FAILED IN PRACTICE", fontweight="bold", color="#991b1b", fontsize=9.5)
    ax_t3_matrix.text(5, 76,
        "• Root Cause: Treat all count elevations as anomalous without physical context.\n"
        "• Flaw 1: Rain scavenging creates massive gross surges -> 32% to 58% of all alarms are rain washout.\n"
        "• Flaw 2: Dialing threshold to 5σ to suppress alarms collapses detection to 6.0% (plumes missed).\n"
        "• Outcome: Severe alarm fatigue, operator blindness during storms, non-operational.",
        fontsize=8.5, color="#1f2937", va="top"
    )

    card2 = patches.FancyBboxPatch((2, 4), 96, 44, boxstyle="round,pad=0.8", ec="#10b981", fc="#ecfdf5", lw=1.5)
    ax_t3_matrix.add_patch(card2)
    ax_t3_matrix.text(5, 41, "Tier 3 Weather-Aware ML Fusion (GBDT & MLP) — OPERATIONAL SOLUTION", fontweight="bold", color="#065f46", fontsize=9.5)
    ax_t3_matrix.text(5, 30,
        "• Feature Fusion: Integrates NaI(Tl) R02--R09 spectral ratios + precipitation rate + 3h/6h rain sums.\n"
        "• Discrimination: Recognizes Bi-214 photopeaks coincident with rain as benign natural radon washout.\n"
        "• Performance Breakthrough: 0.0% rain-coincident false alarms; overall false alarms cut by 95% (4.1/yr).\n"
        "• Plume Sensitivity: Preserves 92.0% detection of Cs-137, I-131, Co-60 plumes with 3--4h response time.",
        fontsize=8.5, color="#1f2937", va="top"
    )

    out_file = FIGURES_DIR / "environmental_engineering_synthesis.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    generate_engineering_synthesis_figure()
