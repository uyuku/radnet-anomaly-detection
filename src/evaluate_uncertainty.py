"""
Statistical Uncertainty Driver: Plume-Level Cluster Bootstrap, Station-Month/Week Block Bootstrap,
Exact Binomial/Poisson Intervals, Matched-Operating-Point Threshold Resampling,
10-Catalog x 3-Seed Variance Decomposition, and Coverage Simulation Verification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import sys
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.artifact_tests import (
    _build_station_features_for_config,
    _enrich_catalog_with_snr_and_resets,
)
from src.baselines import score_b4c_radon_projection_combined
from src.evaluate_baselines import (
    HOURS_PER_YEAR,
    _build_adaptive_threshold_grid,
    _prepare_all_station_data,
)
from src.feature_engineering import (
    TIER_1_FEATURES,
    TIER_1B_FEATURES,
    TIER_2_FEATURES,
    TIER_3_FEATURES,
    TIER_4_NUCLEAR_FEATURES,
    compute_features_for_series,
)
from src.train_and_evaluate_models import LABEL_MAP, STATIONS
from src.uncertainty import (
    block_bootstrap_fa_rate,
    clopper_pearson_ci,
    cluster_bootstrap_detection,
    compute_variance_components_anova,
    exact_poisson_rate_ci,
    holm_adjust_pvalues,
    matched_operating_point_bootstrap,
    paired_detector_bootstrap,
    verify_interval_coverage_simulation,
)


def _compute_single_detector_intervals(
    plume_df: pd.DataFrame,
    month_df: pd.DataFrame,
    n_boot: int = 10000,
) -> pd.DataFrame:
    """
    Computes Step 1a (Stratified Plume Cluster Bootstrap + Clopper-Pearson) and
    Step 1b (Station-Month Block Bootstrap + Exact Garwood Poisson) intervals for
    every `(detector_id, split, operating_point_type, operating_target)` in the
    per-plume and per-month tables.
    """
    rows = []
    keys = ["detector_id", "detector_name", "split", "operating_point_type", "operating_target"]
    for key_vals, p_grp in plume_df.groupby(keys, sort=False):
        det_id, det_name, split_name, op_type, op_target = key_vals
        if split_name != "test":
            continue
        m_grp = month_df[
            (month_df["detector_id"] == det_id)
            & (month_df["split"] == split_name)
            & (month_df["operating_point_type"] == op_type)
            & (month_df["operating_target"] == op_target)
        ]
        if m_grp.empty:
            continue

        tau = float(p_grp["threshold_tau"].iloc[0])
        val_met = bool(p_grp["val_target_met"].iloc[0]) if "val_target_met" in p_grp.columns else True

        det_res = cluster_bootstrap_detection(p_grp, n_boot=n_boot, alpha=0.05, seed=42)
        fa_res = block_bootstrap_fa_rate(m_grp, n_boot=n_boot, alpha=0.05, seed=42)

        rows.append({
            "detector_id": det_id,
            "detector_name": det_name,
            "operating_point_type": op_type,
            "operating_target": op_target,
            "threshold_tau": round(tau, 6),
            "val_target_met": val_met,
            "n_test_plumes": det_res["n_plumes"],
            "n_detected": det_res["n_detected"],
            "det_pct_point": det_res["detection_pct_point"],
            "det_boot_se_pct": det_res["boot_se_pct"],
            "det_boot_ci_low_pct": det_res["boot_ci_low_pct"],
            "det_boot_ci_high_pct": det_res["boot_ci_high_pct"],
            "det_clopper_pearson_low_pct": det_res["clopper_pearson_low_pct"],
            "det_clopper_pearson_high_pct": det_res["clopper_pearson_high_pct"],
            "n_test_station_months": fa_res["n_blocks"],
            "total_test_episodes": fa_res["total_episodes"],
            "exposure_station_years": fa_res["exposure_station_years"],
            "fa_per_yr_point": fa_res["fa_per_yr_point"],
            "fa_boot_se_per_yr": fa_res["boot_se_fa_per_yr"],
            "fa_boot_ci_low_per_yr": fa_res["boot_ci_low_fa_per_yr"],
            "fa_boot_ci_high_per_yr": fa_res["boot_ci_high_fa_per_yr"],
            "fa_poisson_ci_low_per_yr": fa_res["poisson_ci_low_fa_per_yr"],
            "fa_poisson_ci_high_per_yr": fa_res["poisson_ci_high_fa_per_yr"],
        })
    return pd.DataFrame(rows)


def _compute_paired_comparisons_with_holm(
    plume_df: pd.DataFrame,
    month_df: pd.DataFrame,
    n_boot: int = 10000,
) -> pd.DataFrame:
    """
    Evaluates the pre-specified Primary Comparison (`Tier 4` vs. `B4c` at 90% detection target)
    and the two Exploratory Families (`Exploratory_Family_A_Matched_Detection` and
    `Exploratory_Family_B_Matched_FA_Budget`) with Holm-Bonferroni correction.
    """
    specs = [
        # Primary Pre-Specified Comparison
        (
            "Primary_PreSpecified",
            "Primary: Tier 4 vs B4c (Non-ML) at 90% Val Detection Target",
            "ML_Tier4_PhysicsNuclear",
            "B4c_RadonProjection_Combined",
            "matched_detection_pct",
            90.0,
        ),
        # Exploratory Family A: Matched Detection Targets (90% and 95%)
        (
            "Exploratory_Family_A_Matched_Detection",
            "Exploratory A1: Tier 4 vs Tier 3 at 90% Val Detection Target",
            "ML_Tier4_PhysicsNuclear",
            "ML_Tier3_WeatherFusion",
            "matched_detection_pct",
            90.0,
        ),
        (
            "Exploratory_Family_A_Matched_Detection",
            "Exploratory A2: Tier 4 vs Tier 3 at 95% Val Detection Target",
            "ML_Tier4_PhysicsNuclear",
            "ML_Tier3_WeatherFusion",
            "matched_detection_pct",
            95.0,
        ),
        (
            "Exploratory_Family_A_Matched_Detection",
            "Exploratory A3: Tier 4 vs B4c (Non-ML) at 95% Val Detection Target",
            "ML_Tier4_PhysicsNuclear",
            "B4c_RadonProjection_Combined",
            "matched_detection_pct",
            95.0,
        ),
        (
            "Exploratory_Family_A_Matched_Detection",
            "Exploratory A4: Tier 4 vs Tier 2 (Spectral) at 95% Val Detection Target",
            "ML_Tier4_PhysicsNuclear",
            "ML_Tier2_Spectral",
            "matched_detection_pct",
            95.0,
        ),
        # Exploratory Family B: Matched Validation FA Budgets (5 and 10 FA/yr)
        (
            "Exploratory_Family_B_Matched_FA_Budget",
            "Exploratory B1: Tier 4 vs B4c (Non-ML) at 5 Val FA/yr Budget",
            "ML_Tier4_PhysicsNuclear",
            "B4c_RadonProjection_Combined",
            "matched_fa_per_yr",
            5.0,
        ),
        (
            "Exploratory_Family_B_Matched_FA_Budget",
            "Exploratory B2: Tier 4 vs B4c (Non-ML) at 10 Val FA/yr Budget",
            "ML_Tier4_PhysicsNuclear",
            "B4c_RadonProjection_Combined",
            "matched_fa_per_yr",
            10.0,
        ),
        (
            "Exploratory_Family_B_Matched_FA_Budget",
            "Exploratory B3: Tier 4 vs Tier 3 at 5 Val FA/yr Budget (Matched Max Det)",
            "ML_Tier4_PhysicsNuclear",
            "ML_Tier3_WeatherFusion",
            "matched_fa_per_yr",
            5.0,
        ),
        (
            "Exploratory_Family_B_Matched_FA_Budget",
            "Exploratory B4: Tier 4 vs Tier 3 at 10 Val FA/yr Budget (Matched Max Det)",
            "ML_Tier4_PhysicsNuclear",
            "ML_Tier3_WeatherFusion",
            "matched_fa_per_yr",
            10.0,
        ),
        (
            "Exploratory_Family_B_Matched_FA_Budget",
            "Exploratory B5: Tier 4 vs Tier 3 at 5 Val FA/yr Clean-Quantile Budget",
            "ML_Tier4_PhysicsNuclear",
            "ML_Tier3_WeatherFusion",
            "fa_budget_per_yr",
            5.0,
        ),
        (
            "Exploratory_Family_B_Matched_FA_Budget",
            "Exploratory B6: Tier 4 vs Tier 3 at 10 Val FA/yr Clean-Quantile Budget",
            "ML_Tier4_PhysicsNuclear",
            "ML_Tier3_WeatherFusion",
            "fa_budget_per_yr",
            10.0,
        ),
    ]

    rows = []
    for idx, (family, label, det_a, det_b, op_type, op_target) in enumerate(specs):
        pa = plume_df[
            (plume_df["detector_id"] == det_a)
            & (plume_df["split"] == "test")
            & (plume_df["operating_point_type"] == op_type)
            & (plume_df["operating_target"] == op_target)
        ]
        pb = plume_df[
            (plume_df["detector_id"] == det_b)
            & (plume_df["split"] == "test")
            & (plume_df["operating_point_type"] == op_type)
            & (plume_df["operating_target"] == op_target)
        ]
        ma = month_df[
            (month_df["detector_id"] == det_a)
            & (month_df["split"] == "test")
            & (month_df["operating_point_type"] == op_type)
            & (month_df["operating_target"] == op_target)
        ]
        mb = month_df[
            (month_df["detector_id"] == det_b)
            & (month_df["split"] == "test")
            & (month_df["operating_point_type"] == op_type)
            & (month_df["operating_target"] == op_target)
        ]
        if pa.empty or pb.empty or ma.empty or mb.empty:
            continue

        res = paired_detector_bootstrap(pa, pb, ma, mb, n_boot=n_boot, alpha=0.05, seed=42 + idx)
        rows.append({
            "comparison_family": family,
            "comparison_label": label,
            "detector_a": det_a,
            "detector_b": det_b,
            "operating_point_type": op_type,
            "operating_target": op_target,
            **res,
        })

    df_out = pd.DataFrame(rows)
    df_out["holm_adj_pvalue_det"] = df_out["delta_det_boot_pvalue"]
    df_out["holm_reject_det_alpha005"] = df_out["ci_excludes_zero_det"]
    df_out["holm_adj_pvalue_fa"] = df_out["delta_fa_boot_pvalue"]
    df_out["holm_reject_fa_alpha005"] = df_out["ci_excludes_zero_fa"]

    for fam, grp in df_out.groupby("comparison_family", sort=False):
        if fam == "Primary_PreSpecified":
            continue
        adj_d, rej_d = holm_adjust_pvalues(grp["delta_det_boot_pvalue"].tolist(), alpha=0.05)
        adj_f, rej_f = holm_adjust_pvalues(grp["delta_fa_boot_pvalue"].tolist(), alpha=0.05)
        df_out.loc[grp.index, "holm_adj_pvalue_det"] = [round(x, 5) for x in adj_d]
        df_out.loc[grp.index, "holm_reject_det_alpha005"] = rej_d
        df_out.loc[grp.index, "holm_adj_pvalue_fa"] = [round(x, 5) for x in adj_f]
        df_out.loc[grp.index, "holm_reject_fa_alpha005"] = rej_f

    return df_out


def _build_grid_evaluation_matrices(
    val_data: dict[str, dict[str, Any]],
    test_data: dict[str, dict[str, Any]],
    val_plumes: list[dict[str, Any]],
    test_plumes: list[dict[str, Any]],
    val_cln: dict[str, np.ndarray],
    val_inj: dict[str, np.ndarray],
    test_cln: dict[str, np.ndarray],
    test_inj: dict[str, np.ndarray],
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs `(val_plume_det_matrix, val_block_ep_matrix, val_block_exp_years,
    test_plume_det_matrix, test_block_ep_matrix, test_block_exp_years)` across `grid`.
    """
    n_g = len(grid)
    vp_mat = np.zeros((len(val_plumes), n_g), dtype=float)
    for i, sl in enumerate(val_plumes):
        st_id = sl["station_id"]
        idx = sl["indices"]
        p_i = val_inj[st_id][idx]
        p_c = val_cln[st_id][idx]
        vp_mat[i] = ((p_i[:, None] >= grid[None, :]) & (p_c[:, None] < grid[None, :])).any(axis=0).astype(float)

    tp_mat = np.zeros((len(test_plumes), n_g), dtype=float)
    for i, sl in enumerate(test_plumes):
        st_id = sl["station_id"]
        idx = sl["indices"]
        p_i = test_inj[st_id][idx]
        p_c = test_cln[st_id][idx]
        tp_mat[i] = ((p_i[:, None] >= grid[None, :]) & (p_c[:, None] < grid[None, :])).any(axis=0).astype(float)

    vb_rows = []
    vb_exp = []
    for st_id, vd in val_data.items():
        m_cal = vd["calendar_mask"]
        val_m = vd["valid_mask"]
        ym_arr = vd["year_month"]
        p_c = val_cln[st_id][m_cal]
        is_on = (p_c[:, None] >= grid[None, :]) & val_m[:, None]
        prior = np.vstack([np.zeros((1, n_g), dtype=bool), is_on[:-1]])
        ep_starts = is_on & (~prior)
        for ym in pd.unique(ym_arr):
            ym_m = ym_arr == ym
            obs_h = int(val_m[ym_m].sum())
            if obs_h > 0:
                vb_rows.append(ep_starts[ym_m].sum(axis=0).astype(float))
                vb_exp.append(obs_h / HOURS_PER_YEAR)

    tb_rows = []
    tb_exp = []
    for st_id, td in test_data.items():
        m_cal = td["calendar_mask"]
        val_m = td["valid_mask"]
        ym_arr = td["year_month"]
        p_c = test_cln[st_id][m_cal]
        is_on = (p_c[:, None] >= grid[None, :]) & val_m[:, None]
        prior = np.vstack([np.zeros((1, n_g), dtype=bool), is_on[:-1]])
        ep_starts = is_on & (~prior)
        for ym in pd.unique(ym_arr):
            ym_m = ym_arr == ym
            obs_h = int(val_m[ym_m].sum())
            if obs_h > 0:
                tb_rows.append(ep_starts[ym_m].sum(axis=0).astype(float))
                tb_exp.append(obs_h / HOURS_PER_YEAR)

    return (
        vp_mat,
        np.vstack(vb_rows),
        np.asarray(vb_exp, dtype=float),
        tp_mat,
        np.vstack(tb_rows),
        np.asarray(tb_exp, dtype=float),
    )


def _generate_perturbed_catalog(base_catalog: pd.DataFrame, catalog_seed: int) -> pd.DataFrame:
    """
    Generates an independent synthetic plume catalog with RNG seed `catalog_seed`
    preserving the exact split, station, environment, and 48-hour buffer constraints,
    while resampling plume start times, nuclide scenarios, passage/retention durations,
    peak CPMs, and rise parameters.
    """
    rng = np.random.default_rng(catalog_seed)
    cat = base_catalog.copy()

    train_scenarios = [
        "fission_pure_cs137",
        "fission_pure_i131",
        "activation_orphan_co60",
        "fission_reactor_fukushima",
        "mixed_fission_activation",
    ]
    test_scenarios = train_scenarios + [
        "fission_pure_ru103",
        "fission_zr95_nb95",
        "fission_ba140_la140",
    ]
    test_probs = [0.16, 0.16, 0.16, 0.17, 0.17, 0.06, 0.06, 0.06]

    for i, row in cat.iterrows():
        split = str(row["split"])
        # Perturb peak_cpm by log-normal factor (+/- 25%) and durations within realistic bounds
        scale = float(np.exp(rng.normal(0.0, 0.22)))
        new_peak = float(np.clip(float(row["peak_cpm"]) * scale, 18.0, 650.0))
        t_pass = int(np.clip(int(row["t_passage_hours"]) + int(rng.integers(-3, 4)), 3, 28))
        t_ret = int(np.clip(int(row["t_retention_hours"]) + int(rng.integers(-8, 9)), 5, 136))
        dur = t_pass + t_ret

        if split == "train":
            scen = str(rng.choice(train_scenarios))
            shape = str(rng.choice(["linear_ramp", "step"]))
        else:
            scen = str(rng.choice(test_scenarios, p=test_probs))
            shape = str(rng.choice(["sigmoidal", "exponential"]))

        cat.loc[i, "peak_cpm"] = round(new_peak, 2)
        cat.loc[i, "t_passage_hours"] = t_pass
        cat.loc[i, "t_retention_hours"] = t_ret
        cat.loc[i, "duration_hours"] = dur
        cat.loc[i, "nuclide_scenario"] = scen
        cat.loc[i, "shape_family"] = shape
        cat.loc[i, "rise_param"] = round(float(rng.uniform(0.18, 0.55)), 4)

        s_dt = pd.to_datetime(row["start_utc"])
        cat.loc[i, "end_utc"] = (s_dt + pd.Timedelta(hours=dur - 1)).strftime("%Y-%m-%d %H:%M:%S")

    return cat


def run_uncertainty_pipeline() -> None:
    print("=== Uncertainty Evaluation Step 1a-c & Step 3: Single-Detector Intervals and Paired Comparisons ===")
    b_plume = pd.read_csv("data/processed/baseline_per_plume.csv")
    b_month = pd.read_csv("data/processed/baseline_per_month.csv")
    t_plume = pd.read_csv("data/processed/tier_per_plume.csv")
    t_month = pd.read_csv("data/processed/tier_per_month.csv")

    # Combine baseline_per_plume (which also has ML_Tier1..4 at 80/90/95% and 5/10/50 FA/yr)
    # with tier_per_plume (which has 1, 2, 20 FA/yr) deduplicating on key columns
    all_plume = pd.concat([b_plume, t_plume], ignore_index=True).drop_duplicates(
        subset=["detector_id", "split", "operating_point_type", "operating_target", "plume_id"]
    )
    all_month = pd.concat([b_month, t_month], ignore_index=True).drop_duplicates(
        subset=["detector_id", "split", "operating_point_type", "operating_target", "station_id", "year_month"]
    )

    df_single = _compute_single_detector_intervals(all_plume, all_month, n_boot=10000)
    df_single.to_csv("data/processed/uncertainty_single_detector_intervals.csv", index=False)
    print(f"Wrote data/processed/uncertainty_single_detector_intervals.csv ({len(df_single)} rows)")

    df_paired = _compute_paired_comparisons_with_holm(all_plume, all_month, n_boot=10000)
    df_paired.to_csv("data/processed/uncertainty_paired_comparisons.csv", index=False)
    print(f"Wrote data/processed/uncertainty_paired_comparisons.csv ({len(df_paired)} rows)")

    # Step 1b Sensitivity Check + Step 1d Matched-Operating-Point Threshold Resampling
    print("=== Uncertainty Evaluation Step 1b (Week vs Month Blocks) & Step 1d (Matched-Operating-Point Resampling) ===")
    fit_data, val_data, test_data, val_plumes, test_plumes = _prepare_all_station_data()

    X_fit = pd.concat(
        [fd["feats_inj"].loc[fd["fit_labeled_mask"], TIER_4_NUCLEAR_FEATURES] for fd in fit_data.values()],
        ignore_index=True,
    )
    y_fit = np.concatenate([
        fd["feats_inj"].loc[fd["fit_labeled_mask"], "label"].map(LABEL_MAP).to_numpy(dtype=int)
        for fd in fit_data.values()
    ])
    fit_clean_by_station = {st_id: fd["fit_clean_df"] for st_id, fd in fit_data.items()}

    det_scores: dict[str, dict[str, Any]] = {}

    # Score B4c
    det_scores["B4c_RadonProjection_Combined"] = {
        "name": "B4c: Radon Projection + Stripped Channel Peak Z",
        "val_cln": {st: score_b4c_radon_projection_combined(val_data[st]["feats_cln"]) for st in fit_data},
        "val_inj": {st: score_b4c_radon_projection_combined(val_data[st]["feats_inj"]) for st in fit_data},
        "test_cln": {st: score_b4c_radon_projection_combined(test_data[st]["feats_cln"]) for st in fit_data},
        "test_inj": {st: score_b4c_radon_projection_combined(test_data[st]["feats_inj"]) for st in fit_data},
    }

    for t_name, det_id, f_cols in [
        ("Tier 1 (Gross)", "ML_Tier1_Gross", TIER_1_FEATURES),
        ("Tier 1b (Gross+Weather)", "ML_Tier1b_GrossWeather", TIER_1B_FEATURES),
        ("Tier 2 (Spectral)", "ML_Tier2_Spectral", TIER_2_FEATURES),
        ("Tier 3 (Weather Fusion)", "ML_Tier3_WeatherFusion", TIER_3_FEATURES),
        ("Tier 4 (Physics-Motivated Nuclear)", "ML_Tier4_PhysicsNuclear", TIER_4_NUCLEAR_FEATURES),
    ]:
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=4,
            verbose=-1,
        )
        clf.fit(X_fit[f_cols], y_fit)
        det_scores[det_id] = {
            "name": t_name,
            "val_cln": {st: clf.predict_proba(val_data[st]["feats_cln"][f_cols])[:, 2] for st in fit_data},
            "val_inj": {st: clf.predict_proba(val_data[st]["feats_inj"][f_cols])[:, 2] for st in fit_data},
            "test_cln": {st: clf.predict_proba(test_data[st]["feats_cln"][f_cols])[:, 2] for st in fit_data},
            "test_inj": {st: clf.predict_proba(test_data[st]["feats_inj"][f_cols])[:, 2] for st in fit_data},
        }

    # Station-week vs station-month block bootstrap sensitivity check
    week_month_rows = []
    for det_id, sc in det_scores.items():
        # Get 95% target threshold from df_single
        sub_s = df_single[
            (df_single["detector_id"] == det_id)
            & (df_single["operating_point_type"] == "matched_detection_pct")
            & (df_single["operating_target"] == 95.0)
        ]
        if sub_s.empty:
            continue
        tau_95 = float(sub_s["threshold_tau"].iloc[0])

        # Build station-month and station-week blocks on test split counting episode starts BEFORE binning
        m_blocks = []
        w_blocks = []
        for st_id, td in test_data.items():
            val_m = td["valid_mask"]
            p_c = sc["test_cln"][st_id]
            is_on = (p_c >= tau_95) & val_m
            prior = np.concatenate([[False], is_on[:-1]])
            ep_starts = is_on & (~prior)

            dts = pd.to_datetime(td["feats_cln"]["dt"])
            ym_arr = dts.dt.strftime("%Y-%m").to_numpy()
            yw_arr = dts.dt.strftime("%Y-W%W").to_numpy()

            for ym in pd.unique(ym_arr):
                mask = ym_arr == ym
                obs_h = int(val_m[mask].sum())
                if obs_h > 0:
                    m_blocks.append({
                        "station_id": st_id,
                        "block_id": str(ym),
                        "false_alarm_episodes": int(ep_starts[mask].sum()),
                        "exposure_station_years": obs_h / HOURS_PER_YEAR,
                    })
            for yw in pd.unique(yw_arr):
                mask = yw_arr == yw
                obs_h = int(val_m[mask].sum())
                if obs_h > 0:
                    w_blocks.append({
                        "station_id": st_id,
                        "block_id": str(yw),
                        "false_alarm_episodes": int(ep_starts[mask].sum()),
                        "exposure_station_years": obs_h / HOURS_PER_YEAR,
                    })

        res_m = block_bootstrap_fa_rate(pd.DataFrame(m_blocks), n_boot=10000, seed=42)
        res_w = block_bootstrap_fa_rate(pd.DataFrame(w_blocks), n_boot=10000, seed=42)
        week_month_rows.append({
            "detector_id": det_id,
            "detector_name": sc["name"],
            "threshold_tau_95pct_target": tau_95,
            "total_episodes": res_m["total_episodes"],
            "exposure_station_years": res_m["exposure_station_years"],
            "fa_per_yr_point": res_m["fa_per_yr_point"],
            "month_block_n": res_m["n_blocks"],
            "month_boot_se": res_m["boot_se_fa_per_yr"],
            "month_boot_ci_low": res_m["boot_ci_low_fa_per_yr"],
            "month_boot_ci_high": res_m["boot_ci_high_fa_per_yr"],
            "week_block_n": res_w["n_blocks"],
            "week_boot_se": res_w["boot_se_fa_per_yr"],
            "week_boot_ci_low": res_w["boot_ci_low_fa_per_yr"],
            "week_boot_ci_high": res_w["boot_ci_high_fa_per_yr"],
            "exact_poisson_ci_low": res_m["poisson_ci_low_fa_per_yr"],
            "exact_poisson_ci_high": res_m["poisson_ci_high_fa_per_yr"],
        })
    pd.DataFrame(week_month_rows).to_csv("data/processed/uncertainty_week_vs_month_block_sensitivity.csv", index=False)
    print("Wrote data/processed/uncertainty_week_vs_month_block_sensitivity.csv")

    # Step 1d: Matched-operating-point threshold resampling (B = 5,000 for speed + precision)
    grid_mats: dict[str, tuple] = {}
    for det_id in ("ML_Tier4_PhysicsNuclear", "ML_Tier3_WeatherFusion", "B4c_RadonProjection_Combined"):
        sc = det_scores[det_id]
        all_vc = np.concatenate([sc["val_cln"][st][val_data[st]["calendar_mask"]][val_data[st]["valid_mask"]] for st in fit_data])
        all_vi = np.concatenate([sc["val_inj"][st][sl["indices"]] for sl in val_plumes for st in [sl["station_id"]]])
        grid = _build_adaptive_threshold_grid(all_vc, all_vi, n_grid=400)
        mats = _build_grid_evaluation_matrices(
            val_data, test_data, val_plumes, test_plumes,
            sc["val_cln"], sc["val_inj"], sc["test_cln"], sc["test_inj"], grid
        )
        grid_mats[det_id] = (grid, *mats)

    mop_rows = []
    for det_b, comp_name in [
        ("B4c_RadonProjection_Combined", "Tier 4 vs B4c (Non-ML)"),
        ("ML_Tier3_WeatherFusion", "Tier 4 vs Tier 3 (Weather Fusion)"),
    ]:
        g_a, vp_a, vb_a, v_exp, tp_a, tb_a, t_exp = grid_mats["ML_Tier4_PhysicsNuclear"]
        g_b, vp_b, vb_b, _, tp_b, tb_b, _ = grid_mats[det_b]

        for mode, tgt_val, op_label in [
            ("detection_target", 0.90, "Val_Detection_Target_90pct"),
            ("detection_target", 0.95, "Val_Detection_Target_95pct"),
            ("fa_budget", 5.0, "Val_FA_Budget_5_per_yr"),
            ("fa_budget", 10.0, "Val_FA_Budget_10_per_yr"),
        ]:
            res_mop = matched_operating_point_bootstrap(
                g_a, vp_a, vb_a, v_exp, tp_a, tb_a, t_exp,
                g_b, vp_b, vb_b, tp_b, tb_b,
                mode=mode, target_value=tgt_val, n_boot=5000, alpha=0.05, seed=42
            )
            mop_rows.append({
                "comparison": comp_name,
                "operating_point": op_label,
                "detector_a": "ML_Tier4_PhysicsNuclear",
                "detector_b": det_b,
                **res_mop,
            })
    pd.DataFrame(mop_rows).to_csv("data/processed/uncertainty_matched_operating_point_resampled.csv", index=False)
    print("Wrote data/processed/uncertainty_matched_operating_point_resampled.csv")

    # -----------------------------------------------------------------------
    # Step 2: Benchmark-Design Variance (10 Plume Catalogs x 3 Model Seeds)
    # -----------------------------------------------------------------------
    print("=== Uncertainty Evaluation Step 2: Running 10 Catalog Seeds x 3 Model Seeds Variance Decomposition ===")
    legacy_path = Path("data/processed/synthetic_injection_catalog_legacy_v1.csv")
    if not legacy_path.exists():
        legacy_path = Path("data/processed/synthetic_injection_catalog.csv")
    base_catalog = pd.read_csv(legacy_path)

    with open("data/processed/station_dry_baselines_train_only.json", "r") as f:
        train_baselines = json.load(f)

    merged_by_station = {}
    clean_feats_cache = {}
    for st in STATIONS:
        st_id = st["id"]
        df_m = pd.read_csv(f"data/processed/merged_{st_id}_2017_2025.csv.gz")
        df_m["dt"] = pd.to_datetime(df_m["utc_hour"])
        df_m = df_m.sort_values("dt").reset_index(drop=True)
        merged_by_station[st_id] = df_m
        clean_feats_cache[(st_id, "train")] = fit_data[st_id]["feats_cln"]
        clean_feats_cache[(st_id, "test")] = test_data[st_id]["feats_cln"]

    catalog_seeds = list(range(100, 110))
    model_seeds = [42, 43, 44]
    cat_run_rows = []

    cat_runs_path = Path("data/processed/uncertainty_catalog_seed_runs.csv")
    if cat_runs_path.exists() and "--recompute-catalogs" not in sys.argv:
        df_cat_runs = pd.read_csv(cat_runs_path)
        print(f"Loaded existing {cat_runs_path} ({len(df_cat_runs)} rows)")
    else:
        for c_seed in catalog_seeds:
            print(f"  Evaluating catalog_seed={c_seed} across 3 model seeds...")
            cat_pert = _generate_perturbed_catalog(base_catalog, catalog_seed=c_seed)
            cat_c = _enrich_catalog_with_snr_and_resets(
                cat_pert, timing_mode="continuous_arbitrary_end", use_derived_templates=True, seed=c_seed
            )
            tr_c, te_c = _build_station_features_for_config(
                cat_c,
                merged_by_station,
                clean_feats_cache,
                train_baselines,
                add_counting_noise=True,
                independent_dose=True,
                seed=c_seed,
            )

            # Build fit arrays and validation/test slices for catalog c
            x_tr_c = pd.concat([tr_c[st]["feats_inj"].loc[tr_c[st]["m_tr"], TIER_4_NUCLEAR_FEATURES] for st in tr_c], ignore_index=True)
            y_tr_c = np.concatenate([tr_c[st]["feats_inj"].loc[tr_c[st]["m_tr"], "label"].map(LABEL_MAP).to_numpy(dtype=int) for st in tr_c])

            val_obs_h = sum(int(tr_c[st]["val_valid"].sum()) for st in tr_c)
            val_st_yrs = max(1e-6, val_obs_h / HOURS_PER_YEAR)
            test_obs_h = sum(int(te_c[st]["valid"].sum()) for st in te_c)
            test_st_yrs = max(1e-6, test_obs_h / HOURS_PER_YEAR)

            val_slices_c = [sl for st in tr_c for sl in tr_c[st]["val_inj_slices"]]
            test_slices_c = [sl for st in te_c for sl in te_c[st]["test_inj_slices"]]

            for m_seed in model_seeds:
                for t_name, det_id, f_cols in [
                    ("Tier 1 (Gross)", "ML_Tier1_Gross", TIER_1_FEATURES),
                    ("Tier 2 (Spectral)", "ML_Tier2_Spectral", TIER_2_FEATURES),
                    ("Tier 3 (Weather Fusion)", "ML_Tier3_WeatherFusion", TIER_3_FEATURES),
                    ("Tier 4 (Physics-Motivated Nuclear)", "ML_Tier4_PhysicsNuclear", TIER_4_NUCLEAR_FEATURES),
                ]:
                    clf_c = lgb.LGBMClassifier(
                        objective="multiclass",
                        num_class=3,
                        n_estimators=150,
                        learning_rate=0.05,
                        num_leaves=31,
                        max_depth=6,
                        min_child_samples=50,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        class_weight="balanced",
                        random_state=m_seed,
                        n_jobs=4,
                        verbose=-1,
                    )
                    clf_c.fit(x_tr_c[f_cols], y_tr_c)

                    vp_inj = {st: clf_c.predict_proba(tr_c[st]["feats_inj"][f_cols])[:, 2] for st in tr_c}
                    vp_cln = {st: clf_c.predict_proba(tr_c[st]["feats_cln"][f_cols])[:, 2] for st in tr_c}
                    tp_inj = {st: clf_c.predict_proba(te_c[st]["feats_inj"][f_cols])[:, 2] for st in te_c}
                    tp_cln = {st: clf_c.predict_proba(te_c[st]["feats_cln"][f_cols])[:, 2] for st in te_c}

                    # Select threshold on 2021-2022 validation for 95% target
                    grid_c = np.unique(np.concatenate([np.linspace(0.50, 0.98, 49), np.linspace(0.981, 0.999, 37)]))
                    v_det = np.zeros(len(grid_c))
                    for st_id, idx, _ in val_slices_c:
                        pi = vp_inj[st_id][idx]
                        pc = vp_cln[st_id][idx]
                        v_det += ((pi[:, None] >= grid_c[None, :]) & (pc[:, None] < grid_c[None, :])).any(axis=0).astype(float)
                    v_det /= max(1, len(val_slices_c))

                    v_fa = np.zeros(len(grid_c))
                    for st_id in tr_c:
                        m_cal = tr_c[st_id]["m_val_calendar"]
                        vm = tr_c[st_id]["val_valid"]
                        pc = vp_cln[st_id][m_cal]
                        on = (pc[:, None] >= grid_c[None, :]) & vm[:, None]
                        pr = np.vstack([np.zeros((1, len(grid_c)), dtype=bool), on[:-1]])
                        v_fa += (on & (~pr)).sum(axis=0)
                    v_fa /= val_st_yrs

                    el = np.flatnonzero(v_det >= 0.95)
                    if len(el) == 0:
                        el = np.flatnonzero(v_det == v_det.max())
                    best_i = int(el[np.argmin(v_fa[el])])
                    tau_c = float(grid_c[best_i])

                    # Also select threshold for 5 FA/yr validation budget
                    el_b = np.flatnonzero(v_fa <= 5.0 + 1e-9)
                    tau_5fa = float(grid_c[el_b[0]]) if len(el_b) > 0 else float(grid_c[-1])

                    for op_lbl, tau_eval in [("Val_Target_95pct", tau_c), ("Val_FA_Budget_5_per_yr", tau_5fa)]:
                        det_cnt = 0
                        for st_id, idx, *_ in test_slices_c:
                            pi = tp_inj[st_id][idx]
                            pc = tp_cln[st_id][idx]
                            if ((pi >= tau_eval) & (pc < tau_eval)).any():
                                det_cnt += 1
                        test_det_pct = 100.0 * det_cnt / max(1, len(test_slices_c))

                        tot_ep = 0
                        for st_id in te_c:
                            vm = te_c[st_id]["valid"].to_numpy(dtype=bool)
                            pc = tp_cln[st_id]
                            on = (pc >= tau_eval) & vm
                            pr = np.concatenate([[False], on[:-1]])
                            tot_ep += int((on & (~pr)).sum())
                        test_fa_yr = tot_ep / test_st_yrs

                        cat_run_rows.append({
                            "tier_name": t_name,
                            "detector_id": det_id,
                            "operating_point": op_lbl,
                            "catalog_seed": c_seed,
                            "model_seed": m_seed,
                            "threshold_tau": round(tau_eval, 5),
                            "test_detection_pct": round(test_det_pct, 2),
                            "test_fa_per_yr": round(test_fa_yr, 2),
                        })

        df_cat_runs = pd.DataFrame(cat_run_rows)
        df_cat_runs.to_csv(cat_runs_path, index=False)

    var_rows = []
    for (t_name, det_id, op_lbl), grp in df_cat_runs.groupby(["tier_name", "detector_id", "operating_point"], sort=False):
        vc_det = compute_variance_components_anova(grp, "test_detection_pct")
        vc_fa = compute_variance_components_anova(grp, "test_fa_per_yr")
        var_rows.append({
            "tier_name": t_name,
            "detector_id": det_id,
            "operating_point": op_lbl,
            "metric": "test_detection_pct",
            **vc_det,
        })
        var_rows.append({
            "tier_name": t_name,
            "detector_id": det_id,
            "operating_point": op_lbl,
            "metric": "test_fa_per_yr",
            **vc_fa,
        })
    df_var = pd.DataFrame(var_rows)
    df_var.to_csv("data/processed/uncertainty_catalog_seed_variance.csv", index=False)
    print("Wrote data/processed/uncertainty_catalog_seed_variance.csv")

    # -----------------------------------------------------------------------
    # Step 4: Annotate `rigorous_seed_statistical_tests.csv` with explicit scope warning
    # -----------------------------------------------------------------------
    seed_test_path = Path("data/processed/rigorous_seed_statistical_tests.csv")
    if seed_test_path.exists():
        st_df = pd.read_csv(seed_test_path)
        st_df["inference_scope_warning"] = (
            "MEASURES_LIGHTGBM_SOLVER_SEED_RANDOMNESS_ON_ONE_FIXED_DATASET_ONLY_DO_NOT_USE_FOR_POPULATION_INFERENCE"
        )
        st_df.to_csv(seed_test_path, index=False)

    # -----------------------------------------------------------------------
    # Step 5: Coverage Simulation Verification
    # -----------------------------------------------------------------------
    print("=== Uncertainty Evaluation Step 5: Running Coverage Simulation Verification ===")
    df_cov = verify_interval_coverage_simulation(n_trials=500, n_boot=1000, alpha=0.05, seed=42)
    df_cov.to_csv("data/processed/uncertainty_coverage_simulation.csv", index=False)
    print(df_cov.to_string(index=False))

    # -----------------------------------------------------------------------
    # Step 6: Figures
    # -----------------------------------------------------------------------
    _plot_uncertainty_figures(df_single, df_var)
    print("Completed statistical uncertainty evaluation and figures.")


def _plot_uncertainty_figures(df_single: pd.DataFrame, df_var: pd.DataFrame) -> None:
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Regenerate `eval_bootstrap_uncertainty.png` showing 95% Plume Cluster Bootstrap + Station-Month Block Bootstrap CIs
    target_ids = [
        ("B1a_Rolling168h_Z", "B1a: Gross 168h Z"),
        ("B2a_RainMasked_Holdoff6h", "B2a: Rain-Masked Z"),
        ("B3b_WeatherRegression_SplineGAM", "B3b: Weather GAM"),
        ("B4c_RadonProjection_Combined", "B4c: Radon Strip (Non-ML)"),
        ("B6c_OneClassSVM_Tier4_BgOnly", "B6c: One-Class SVM"),
        ("ML_Tier1b_GrossWeather", "Tier 1b: Gross+Weather"),
        ("ML_Tier2_Spectral", "Tier 2: Spectral GBDT"),
        ("ML_Tier3_WeatherFusion", "Tier 3: Weather Fusion"),
        ("ML_Tier4_PhysicsNuclear", "Tier 4: Nuclear GBDT"),
    ]
    sub_95 = df_single[
        (df_single["operating_point_type"] == "matched_detection_pct")
        & (df_single["operating_target"] == 95.0)
    ].set_index("detector_id")

    labels, det_pt, det_lo, det_hi, fa_pt, fa_lo, fa_hi = [], [], [], [], [], [], []
    for did, lbl in target_ids:
        if did not in sub_95.index:
            continue
        r = sub_95.loc[did]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        labels.append(lbl)
        det_pt.append(float(r["det_pct_point"]))
        det_lo.append(float(r["det_boot_ci_low_pct"]))
        det_hi.append(float(r["det_boot_ci_high_pct"]))
        fa_pt.append(float(r["fa_per_yr_point"]))
        fa_lo.append(float(r["fa_boot_ci_low_per_yr"]))
        fa_hi.append(float(r["fa_boot_ci_high_per_yr"]))

    y_pos = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)

    det_err = [np.maximum(0.0, np.array(det_pt) - np.array(det_lo)), np.maximum(0.0, np.array(det_hi) - np.array(det_pt))]
    axes[0].barh(y_pos, det_pt, xerr=det_err, color="#0072B2", alpha=0.85, capsize=4, height=0.6)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=9.5)
    axes[0].set_xlabel("Test Plume Detection Rate (%) [95% Stratified Plume Bootstrap CI]", fontsize=10)
    axes[0].set_title("A. Out-of-Sample Detection (200 Test Plumes, B=10,000)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlim(50, 102)
    axes[0].grid(True, axis="x", alpha=0.3)

    fa_err = [np.maximum(0.0, np.array(fa_pt) - np.array(fa_lo)), np.maximum(0.0, np.array(fa_hi) - np.array(fa_pt))]
    axes[1].barh(y_pos, fa_pt, xerr=fa_err, color="#D55E00", alpha=0.85, capsize=4, height=0.6)
    axes[1].set_xscale("symlog", linthresh=5.0)
    axes[1].set_xlabel("Test False Alarms / Station-Year [95% Station-Month Block Bootstrap CI]", fontsize=10)
    axes[1].set_title("B. Out-of-Sample False Alarms (12.16 Station-Years, B=10,000)", fontsize=10.5, fontweight="bold")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "eval_bootstrap_uncertainty.png", dpi=300)
    plt.close(fig)

    # 2. Plot Variance Components (Catalog vs Model Seed)
    sub_v = df_var[df_var["operating_point"] == "Val_Target_95pct"]
    if not sub_v.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
        for ax_i, (metric, title_str) in enumerate([
            ("test_detection_pct", "A. Test Detection Rate Variance (%)"),
            ("test_fa_per_yr", "B. Test False-Alarm Rate Variance (%)"),
        ]):
            m_df = sub_v[sub_v["metric"] == metric]
            x = np.arange(len(m_df))
            cat_pct = m_df["pct_var_catalog"].to_numpy(dtype=float)
            seed_pct = m_df["pct_var_model_seed"].to_numpy(dtype=float)
            axes[ax_i].bar(x, cat_pct, label="Plume Catalog Variance (10 Catalogs)", color="#009E73", width=0.55)
            axes[ax_i].bar(x, seed_pct, bottom=cat_pct, label="Model Seed Variance (3 Seeds)", color="#CC79A7", width=0.55)
            axes[ax_i].set_xticks(x)
            axes[ax_i].set_xticklabels(m_df["tier_name"], rotation=20, ha="right", fontsize=9)
            axes[ax_i].set_ylabel("Share of Total Variance (%)", fontsize=10)
            axes[ax_i].set_ylim(0, 105)
            axes[ax_i].set_title(title_str, fontsize=10.5, fontweight="bold")
            axes[ax_i].legend(loc="upper right", frameon=True, fontsize=8.5)
            axes[ax_i].grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "uncertainty_variance_components.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    run_uncertainty_pipeline()
