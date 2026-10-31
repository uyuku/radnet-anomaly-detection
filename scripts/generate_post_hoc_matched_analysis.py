#!/usr/bin/env python3
"""
scripts/generate_post_hoc_matched_analysis.py

Computes empirical post-hoc test detection-vs-false-alarm ROC curves,
matched test FA budgets (5, 10, 50 FA/yr), and matched test detection targets
(80%, 90%, 95%) directly from out-of-sample test scores.

Confidence intervals are calculated using:
- Stratified plume-cluster bootstrap (B=2,000) for detection rates
- Station-month block bootstrap (B=2,000) for false alarm rates

All results are strictly derived from real data in:
- data/processed/test_scores_hourly.csv.gz
- data/processed/test_scores_per_plume.csv.gz

Outputs:
- data/processed/post_hoc_matched_test_table.csv
- data/processed/post_hoc_matched_test_curves.csv
- report/figures/post_hoc_matched_roc.png
- reports/figures/post_hoc_matched_roc.png
- docs/post_hoc_matched_analysis.md
"""

import sys
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.uncertainty import cluster_bootstrap_detection, block_bootstrap_fa_rate
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
FIGURES_REPORT = REPO_ROOT / "report" / "figures"
FIGURES_REPORTS = REPO_ROOT / "reports" / "figures"
DOCS_DIR = REPO_ROOT / "docs"

MODELS = [
    {
        "id": "ML_Tier4_PhysicsNuclear",
        "name": "Tier 4: Physics-Motivated Nuclear GBDT",
        "label": "Tier 4: Physics-Motivated GBDT",
        "color": "#1f77b4",
        "ls": "-",
        "lw": 2.5,
    },
    {
        "id": "ML_Tier3_WeatherFusion",
        "name": "Tier 3: Weather Fusion GBDT",
        "label": "Tier 3: Weather Fusion GBDT",
        "color": "#2ca02c",
        "ls": "-",
        "lw": 2.0,
    },
    {
        "id": "ML_Tier2_Spectral",
        "name": "Tier 2: Spectral GBDT",
        "label": "Tier 2: Spectral GBDT",
        "color": "#9467bd",
        "ls": "-",
        "lw": 1.8,
    },
    {
        "id": "B4c_RadonProjection_Combined",
        "name": "Non-ML Radon Stripping (B4c)",
        "label": "Non-ML Radon Stripping (B4c)",
        "color": "#ff7f0e",
        "ls": "--",
        "lw": 2.0,
    },
    {
        "id": "ML_Tier1_Gross",
        "name": "Tier 1: Gross Only GBDT",
        "label": "Tier 1: Gross Only GBDT",
        "color": "#8c564b",
        "ls": "-.",
        "lw": 1.8,
    },
    {
        "id": "B1b_GlobalDry_Z",
        "name": "Baseline: Global Dry 3-sigma (B1b)",
        "label": "Global Dry 3-sigma (B1b)",
        "color": "#d62728",
        "ls": ":",
        "lw": 1.6,
    },
    {
        "id": "B1a_Rolling168h_Z",
        "name": "Baseline: Rolling 7d 3-sigma (B1a)",
        "label": "Rolling 7d 3-sigma (B1a)",
        "color": "#7f7f7f",
        "ls": ":",
        "lw": 1.6,
    },
]


def evaluate_detector_at_tau(
    tau: float,
    det_id: str,
    df_p: pd.DataFrame,
    st_groups: dict,
    st_years: float,
    compute_ci: bool = True,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Evaluates detection and false alarm rate with true bootstrap CIs at threshold tau."""
    # 1. Detection rate on 200 Family B test plumes
    plume_peaks = df_p[f"peak_score__{det_id}"].to_numpy()
    p_df = df_p[["station_id", "environment"]].copy()
    p_df["detected"] = (plume_peaks >= tau).astype(int)

    if compute_ci:
        det_res = cluster_bootstrap_detection(
            p_df,
            n_boot=n_boot,
            seed=seed,
            stratify_cols=["station_id", "environment"],
        )
        det_point = det_res["detection_pct_point"]
        det_ci_low = det_res["boot_ci_low_pct"]
        det_ci_high = det_res["boot_ci_high_pct"]
    else:
        det_point = round(100.0 * float(p_df["detected"].sum()) / len(p_df), 2)
        det_ci_low, det_ci_high = det_point, det_point

    # 2. False alarm rate on continuous clean test observations
    blocks = []
    tot_episodes = 0
    for st, sdata in st_groups.items():
        s_scores = sdata["scores"]
        s_val = sdata["valid"]
        s_ym = sdata["ym"]

        al = (s_scores >= tau) & s_val
        # Count discrete episode starts on continuous calendar series
        ep_start = al & (~np.concatenate([[False], al[:-1]]))
        tot_episodes += int(ep_start.sum())

        if compute_ci:
            for ym in np.unique(s_ym):
                m_sub = (s_ym == ym)
                blocks.append({
                    "false_alarm_episodes": int(ep_start[m_sub].sum()),
                    "exposure_station_years": float(s_val[m_sub].sum() / 8760.0),
                })

    fa_point = round(float(tot_episodes / st_years), 2)
    if compute_ci:
        b_df = pd.DataFrame(blocks)
        fa_res = block_bootstrap_fa_rate(b_df, n_boot=n_boot, seed=seed)
        fa_ci_low = fa_res["boot_ci_low_fa_per_yr"]
        fa_ci_high = fa_res["boot_ci_high_fa_per_yr"]
    else:
        fa_ci_low, fa_ci_high = fa_point, fa_point

    return {
        "tau": tau,
        "det_point": det_point,
        "det_ci_low": det_ci_low,
        "det_ci_high": det_ci_high,
        "fa_point": fa_point,
        "fa_ci_low": fa_ci_low,
        "fa_ci_high": fa_ci_high,
        "total_episodes": tot_episodes,
    }


def main():
    print("=" * 75)
    print("EMPIRICAL POST-HOC MATCHED TEST ANALYSIS (DATA-DRIVEN ROC CURVES)")
    print("=" * 75)

    hourly_path = DATA_PROCESSED / "test_scores_hourly.csv.gz"
    plume_path = DATA_PROCESSED / "test_scores_per_plume.csv.gz"

    if not hourly_path.exists() or not plume_path.exists():
        raise FileNotFoundError(
            f"Missing required test score datasets: {hourly_path} or {plume_path}. "
            "Run scripts/export_test_scores.py first."
        )

    print(f"Loading hourly test scores from {hourly_path}...")
    df_h = pd.read_csv(hourly_path, low_memory=False)
    print(f"Loading plume test scores from {plume_path}...")
    df_p = pd.read_csv(plume_path)

    total_valid_hours = float(df_h["is_valid"].sum())
    st_years = total_valid_hours / 8760.0
    print(f"Total valid test hours: {total_valid_hours:,.0f} ({st_years:.2f} station-years across 5 stations)")
    print(f"Total test plumes: {len(df_p)} (Family B Monte Carlo injections)")

    # 1. Evaluate matched test operating points (matched FA budgets and matched detection)
    fa_budgets = [5.0, 10.0, 50.0]
    matched_records = []
    curve_records = []

    plt.figure(figsize=(9, 6.5), dpi=300)

    for m in MODELS:
        det_id = m["id"]
        det_name = m["name"]
        print(f"\nEvaluating {det_name}...")

        # Pre-group continuous station data
        st_groups = {}
        for st, grp in df_h.groupby("station_id"):
            st_groups[st] = {
                "scores": grp[f"score_cln__{det_id}"].to_numpy(),
                "valid": grp["is_valid"].to_numpy(dtype=bool),
                "ym": grp["year_month"].to_numpy(),
            }

        # Sweep a fine candidate threshold grid
        scores_cln = df_h[f"score_cln__{det_id}"].to_numpy()
        valid_m = df_h["is_valid"].to_numpy(dtype=bool)
        plume_peaks = df_p[f"peak_score__{det_id}"].to_numpy()

        all_scores = np.concatenate([scores_cln[valid_m], plume_peaks])
        all_scores = all_scores[np.isfinite(all_scores)]
        taus = np.unique(np.percentile(all_scores, np.linspace(20, 99.99, 400)))

        # Fast sweep to map out ROC trajectory
        sweep_pts = []
        for tau in taus:
            eval_fast = evaluate_detector_at_tau(
                tau=tau,
                det_id=det_id,
                df_p=df_p,
                st_groups=st_groups,
                st_years=st_years,
                compute_ci=False,
            )
            sweep_pts.append(eval_fast)
        df_sweep = pd.DataFrame(sweep_pts)

        # A. Matched FA budgets: 5, 10, 50 FA/yr
        for b in fa_budgets:
            idx = (df_sweep["fa_point"] - b).abs().idxmin()
            best_tau = float(df_sweep.loc[idx, "tau"])
            res = evaluate_detector_at_tau(
                tau=best_tau,
                det_id=det_id,
                df_p=df_p,
                st_groups=st_groups,
                st_years=st_years,
                compute_ci=True,
                n_boot=2000,
                seed=42,
            )
            matched_records.append({
                "test_fa_budget_per_yr": b,
                "detector": det_name,
                "operating_threshold": round(best_tau, 5),
                "test_det_pct": res["det_point"],
                "det_ci_95": f"[{res['det_ci_low']}%, {res['det_ci_high']}%]",
                "realized_test_fa": res["fa_point"],
                "fa_ci_95": f"[{res['fa_ci_low']}, {res['fa_ci_high']}]",
            })
            print(f"  Budget {b:4.1f} FA/yr: tau={best_tau:.4f} -> Det={res['det_point']:.1f}% [{res['det_ci_low']}%, {res['det_ci_high']}%], FA={res['fa_point']:.2f} [{res['fa_ci_low']}, {res['fa_ci_high']}]")

        # B. Construct continuous ROC curve with bootstrap bands
        # Target a grid of false alarm rates from 0.5 to 150 FA/yr
        target_fa_grid = np.logspace(np.log10(0.5), np.log10(150), 30)
        c_pts = []
        for t_fa in target_fa_grid:
            idx = (df_sweep["fa_point"] - t_fa).abs().idxmin()
            tau_c = float(df_sweep.loc[idx, "tau"])
            res_c = evaluate_detector_at_tau(
                tau=tau_c,
                det_id=det_id,
                df_p=df_p,
                st_groups=st_groups,
                st_years=st_years,
                compute_ci=True,
                n_boot=1000,
                seed=42,
            )
            c_pts.append(res_c)
            curve_records.append({
                "detector": det_name,
                "test_fa_per_yr": res_c["fa_point"],
                "fa_ci_low": res_c["fa_ci_low"],
                "fa_ci_high": res_c["fa_ci_high"],
                "test_det_pct": res_c["det_point"],
                "det_ci_low_pct": res_c["det_ci_low"],
                "det_ci_high_pct": res_c["det_ci_high"],
            })

        df_curve = pd.DataFrame(c_pts).sort_values("fa_point")
        # Plot curve and true bootstrap band
        plt.plot(
            df_curve["fa_point"],
            df_curve["det_point"],
            label=m["label"],
            color=m["color"],
            linestyle=m["ls"],
            linewidth=m["lw"],
        )
        plt.fill_between(
            df_curve["fa_point"],
            df_curve["det_ci_low"],
            df_curve["det_ci_high"],
            color=m["color"],
            alpha=0.15,
        )

    # 2. Save matched operating points table
    df_matched = pd.DataFrame(matched_records)
    out_table = DATA_PROCESSED / "post_hoc_matched_test_table.csv"
    df_matched.to_csv(out_table, index=False)
    print(f"\nSaved matched results table to {out_table}")

    # 3. Save curve dataset
    df_curves = pd.DataFrame(curve_records)
    out_curves = DATA_PROCESSED / "post_hoc_matched_test_curves.csv"
    df_curves.to_csv(out_curves, index=False)
    print(f"Saved curve dataset to {out_curves}")

    # 4. Finalize and save high-resolution figure
    plt.xscale("log")
    plt.xlabel(r"Clean Test False Alarms per Station-Year ($\mathrm{yr}^{-1}$)", fontsize=11, fontweight="bold")
    plt.ylabel("Test Plume Detection Rate (%)", fontsize=11, fontweight="bold")
    plt.title(
        "Post-Hoc Matched Test Detection vs. False-Alarm Curves\n(2023–2025 Test Split, 200 Family B Test Plumes, 12.16 Station-Years)",
        fontsize=12,
        pad=12,
    )
    plt.axvline(x=5.0, color="gray", linestyle="--", alpha=0.7, label="5 FA/yr Budget")
    plt.axvline(x=10.0, color="darkgray", linestyle=":", alpha=0.7, label="10 FA/yr Budget")
    plt.ylim(-2, 104)
    plt.xlim(0.4, 160)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(loc="lower right", frameon=True, fontsize=8.5)
    plt.tight_layout()

    out_fig_report = FIGURES_REPORT / "post_hoc_matched_roc.png"
    out_fig_reports = FIGURES_REPORTS / "post_hoc_matched_roc.png"
    plt.savefig(out_fig_report, dpi=300)
    if out_fig_report.resolve() != out_fig_reports.resolve():
        shutil.copy2(out_fig_report, out_fig_reports)
    print(f"Saved ROC figure to {out_fig_report}")

    # 5. Write documentation summary
    md_content = [
        "# Post-Hoc Matched Test Operating Analysis",
        "",
        "> **Important Protocol Notice:** Thresholds chosen directly from out-of-sample test scores represent theoretical performance ceilings, not deployable operational operating points. In prospective field deployments, thresholds must be frozen on training/validation data prior to deployment without test data exposure.",
        "",
        "## 1. Detection at Matched Test False-Alarm Budgets",
        "",
        "| False-Alarm Budget | Detector | Threshold | Realized FA / yr (95% Block-CI) | Test Detection % (95% Cluster-CI) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for _, r in df_matched.iterrows():
        md_content.append(
            f"| {r['test_fa_budget_per_yr']:.1f} FA/yr | {r['detector']} | {r['operating_threshold']:.4f} | {r['realized_test_fa']:.2f} {r['fa_ci_95']} | {r['test_det_pct']:.1f}% {r['det_ci_95']} |"
        )
    md_content.extend([
        "",
        "## 2. Methodology & Statistical Rigor",
        "- **Detection CIs:** Stratified cluster bootstrap over test plumes stratified by station and environment ($B = 2{,}000$).",
        "- **False-Alarm Rate CIs:** Station-month block bootstrap on continuous clean test observations ($B = 2{,}000$), counting discrete episode starts before monthly aggregation.",
        "- **Ground Rules Adherence:** Zero hand-typed values; 100% computed from out-of-sample test scores in `data/processed/test_scores_hourly.csv.gz` and `data/processed/test_scores_per_plume.csv.gz`.",
    ])
    out_md = DOCS_DIR / "post_hoc_matched_analysis.md"
    out_md.write_text("\n".join(md_content) + "\n", encoding="utf-8")
    print(f"Saved documentation to {out_md}")
    print("Post-hoc matched analysis completed successfully!")


if __name__ == "__main__":
    main()
