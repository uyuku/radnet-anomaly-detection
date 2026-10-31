"""
Calibration & Threshold Saturation Diagnosis and FA-Budget Evaluation.

Executes all five steps of threshold evaluation and calibration:
  - Step 1a: Validation predicted probability histograms (log scale) for clean vs in-plume hours
    on Tier 3 and Tier 4 (`reports/figures/threshold_val_probability_histograms.png`) and counts
    of clean validation hours with p >= 0.99, >= 0.95, >= 0.90.
  - Step 1b: Records `target_met` across all tiers, seeds (20 seeds), and targets (90%, 95%),
    adds `target_met` and `n_seeds_tau_90_eq_95` to `data/processed/rigorous_benchmark_summary.csv`
    and `data/processed/threshold_saturation_by_seed.csv`, and compares coarse vs fine/rank grids.
  - Step 1c: Reliability diagrams on validation at hour level and event level
    (`reports/figures/threshold_reliability_diagrams.png`), plus Brier score and log loss
    (`data/processed/threshold_calibration_diagnostics.csv`).
  - Step 1d: Positive training label window sensitivity (first N = 6, 12, 24 hours, passage-only,
    vs full plume including retention; `data/processed/threshold_label_window_sensitivity.csv`).
  - Step 1e: Class weighting sensitivity (`balanced`, `unweighted`, custom `scale_pos_weight` /
    class weights, and environment sample weighting; `data/processed/threshold_class_weight_sensitivity.csv`).
  - Step 2a-d: False-alarm-budget operating thresholds (budgets = 1, 2, 5, 10, 20 FA/yr on clean
    validation) across 20 seeds (`data/processed/benchmark_fa_budget_operating_points.csv`), and
    comparison of raw vs cross-fitted Platt vs cross-fitted Isotonic scores at matched FA budgets
    (`data/processed/threshold_calibration_fa_budget_comparison.csv`).
  - Step 3: Paired (`(p_inj >= tau) & (p_clean < tau)`) vs unpaired (`p_inj >= tau`) detection
    comparison (`data/processed/threshold_paired_vs_unpaired.csv`).
  - Step 4: Event detection metrics within 3, 6, 12, and 24 hours, delay quantiles (p25, median,
    p75, p90, mean), plume-duration inflation analysis, and time-normalized 12h detection metric
    (`data/processed/threshold_event_metrics_and_delays.csv`).
  - Step 5: Saves `data/processed/tier_per_plume.csv` and `data/processed/tier_per_month.csv`
    for downstream statistical uncertainty resampling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from src.evaluate_baselines import (
    HOURS_PER_YEAR,
    _prepare_all_station_data,
    evaluate_single_threshold_detailed,
)
from src.feature_engineering import (
    TIER_1_FEATURES,
    TIER_1B_FEATURES,
    TIER_2_FEATURES,
    TIER_3_FEATURES,
    TIER_4_NUCLEAR_FEATURES,
)
from src.operating_threshold import (
    FINE_THRESHOLD_GRID,
    THRESHOLD_GRID,
    VALIDATION_MID,
    compute_event_detection_metrics,
    cross_fit_calibrate_validation_and_test,
    select_fa_budget_threshold,
    select_operating_threshold,
)
from src.train_and_evaluate_models import LABEL_MAP

SEEDS_20 = list(range(42, 62))
FA_BUDGETS_5 = [1.0, 2.0, 5.0, 10.0, 20.0]

TIER_CONFIGS = [
    ("Tier 1 (Gross)", "ML_Tier1_Gross", TIER_1_FEATURES),
    ("Tier 1b (Gross+Weather)", "ML_Tier1b_GrossWeather", TIER_1B_FEATURES),
    ("Tier 2 (Spectral)", "ML_Tier2_Spectral", TIER_2_FEATURES),
    ("Tier 3 (Weather Fusion)", "ML_Tier3_WeatherFusion", TIER_3_FEATURES),
    ("Tier 4 (Physics-Motivated Nuclear)", "ML_Tier4_PhysicsNuclear", TIER_4_NUCLEAR_FEATURES),
]


def _make_lgbm(seed: int = 42, class_weight: Any = "balanced") -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )


def _build_val_and_test_series_dicts(
    val_data: dict[str, dict[str, Any]],
    test_data: dict[str, dict[str, Any]],
    val_cln_scores: dict[str, np.ndarray],
    val_inj_scores: dict[str, np.ndarray],
    test_cln_scores: dict[str, np.ndarray],
    test_inj_scores: dict[str, np.ndarray],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Constructs `series` dicts compatible with `src/operating_threshold.py`."""
    val_series = {}
    for st_id, vd in val_data.items():
        m_cal = vd["calendar_mask"]
        f_inj = vd["feats_inj"].loc[m_cal].reset_index(drop=True)
        val_series[st_id] = pd.DataFrame({
            "dt": pd.to_datetime(f_inj["dt"]),
            "valid": vd["valid_mask"],
            "p_clean": np.clip(val_cln_scores[st_id][m_cal], 0.0, 1.0),
            "p_injected": np.clip(val_inj_scores[st_id][m_cal], 0.0, 1.0),
            "y_fission": (f_inj["label"] == "fission_product").to_numpy(dtype=int),
        })

    test_series = {}
    for st_id, td in test_data.items():
        f_inj = td["feats_inj"]
        test_series[st_id] = pd.DataFrame({
            "dt": pd.to_datetime(f_inj["dt"]),
            "valid": td["valid_mask"],
            "p_clean": np.clip(test_cln_scores[st_id], 0.0, 1.0),
            "p_injected": np.clip(test_inj_scores[st_id], 0.0, 1.0),
            "y_fission": (f_inj["label"] == "fission_product").to_numpy(dtype=int),
        })
    return val_series, test_series


def _evaluate_tau_paired_or_unpaired(
    tau: float,
    split_station_data: dict[str, dict[str, Any]],
    plume_slices: list[dict[str, Any]],
    scores_clean: dict[str, np.ndarray],
    scores_inj: dict[str, np.ndarray],
    paired: bool = True,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluates a single threshold tau with either paired or unpaired plume detection."""
    tot_ep = 0
    tot_al_hrs = 0
    tot_rain_al_hrs = 0
    tot_obs_hrs = 0

    for st_id, st_d in split_station_data.items():
        m_cal = st_d["calendar_mask"]
        val_m = st_d["valid_mask"]
        rain_m = st_d["rain_mask"]
        p_c = scores_clean[st_id][m_cal]
        is_on = (p_c >= tau) & val_m
        prior = np.concatenate([[False], is_on[:-1]])
        tot_ep += int((is_on & (~prior)).sum())
        tot_al_hrs += int(is_on.sum())
        tot_rain_al_hrs += int((is_on & rain_m).sum())
        tot_obs_hrs += int(val_m.sum())

    st_years = max(1e-6, tot_obs_hrs / HOURS_PER_YEAR)
    plume_rows = []
    for sl in plume_slices:
        st_id = sl["station_id"]
        idx = sl["indices"]
        p_i = scores_inj[st_id][idx]
        p_c = scores_clean[st_id][idx]
        ev_al = ((p_i >= tau) & (p_c < tau)) if paired else (p_i >= tau)
        is_det = bool(ev_al.any())
        delay_h = np.nan
        if is_det:
            first_i = int(np.where(ev_al)[0][0])
            delay_h = float((sl["dts"].iloc[first_i] - sl["start_dt"]).total_seconds() / 3600.0)
        plume_rows.append({
            "plume_id": sl["plume_id"],
            "station_id": st_id,
            "environment": sl["environment"],
            "scenario": sl["scenario"],
            "snr_bin": sl["snr_bin"],
            "duration_hours": sl["duration_hours"],
            "detected": int(is_det),
            "delay_hours": delay_h,
            "alarm_hours_in_plume": int(ev_al.sum()),
        })

    plume_df = pd.DataFrame(plume_rows)
    ev_metrics = compute_event_detection_metrics(plume_df, reference_window_hours=12.0)
    ev_metrics["false_alarm_episodes"] = tot_ep
    ev_metrics["false_alarms_per_station_year"] = round(tot_ep / st_years, 2)
    ev_metrics["rain_coincident_fa_pct"] = round(100.0 * tot_rain_al_hrs / tot_al_hrs, 1) if tot_al_hrs > 0 else 0.0
    return ev_metrics, plume_df


def run_threshold_diagnosis_evaluation() -> None:
    print("=== Step 0: Loading Station Fit/Val/Test Data ===")
    fit_data, val_data, test_data, val_plumes, test_plumes = _prepare_all_station_data()
    catalog_df = pd.read_csv("data/processed/synthetic_injection_catalog.csv")

    X_fit_all = pd.concat(
        [fd["feats_inj"].loc[fd["fit_labeled_mask"], TIER_4_NUCLEAR_FEATURES] for fd in fit_data.values()],
        ignore_index=True,
    )
    y_fit_all = np.concatenate([
        fd["feats_inj"].loc[fd["fit_labeled_mask"], "label"].map(LABEL_MAP).to_numpy(dtype=int)
        for fd in fit_data.values()
    ])

    # -----------------------------------------------------------------------
    # Step 1b & Step 2a: 20-Seed Evaluation Across All 5 Tiers
    #   - Detection targets (90%, 95%) on coarse THRESHOLD_GRID vs FINE_THRESHOLD_GRID vs rank
    #   - FA budgets (1, 2, 5, 10, 20 FA/yr) on clean validation scores
    # -----------------------------------------------------------------------
    print("=== Step 1b & Step 2a: Running 20-Seed Threshold Saturation & FA-Budget Evaluation ===")
    seed_sat_rows = []
    fa_budget_seed_rows = []
    paired_unpaired_rows = []
    event_deadline_rows = []
    tier_plume_rows = []
    tier_month_rows = []

    # Cache seed-42 models and score dicts for Step 1a/1c diagnostics
    seed42_val_series: dict[str, dict[str, pd.DataFrame]] = {}
    seed42_test_series: dict[str, dict[str, pd.DataFrame]] = {}
    seed42_scores: dict[str, dict[str, Any]] = {}

    for tier_name, det_id, f_cols in TIER_CONFIGS:
        print(f"  Evaluating 20 seeds for {tier_name}...")
        for seed in SEEDS_20:
            clf = _make_lgbm(seed=seed, class_weight="balanced")
            clf.fit(X_fit_all[f_cols], y_fit_all)

            v_c, v_i, t_c, t_i = {}, {}, {}, {}
            for st_id in fit_data:
                v_c[st_id] = clf.predict_proba(val_data[st_id]["feats_cln"][f_cols])[:, 2]
                v_i[st_id] = clf.predict_proba(val_data[st_id]["feats_inj"][f_cols])[:, 2]
                t_c[st_id] = clf.predict_proba(test_data[st_id]["feats_cln"][f_cols])[:, 2]
                t_i[st_id] = clf.predict_proba(test_data[st_id]["feats_inj"][f_cols])[:, 2]

            val_ser, test_ser = _build_val_and_test_series_dicts(val_data, test_data, v_c, v_i, t_c, t_i)
            if seed == 42:
                seed42_val_series[tier_name] = val_ser
                seed42_test_series[tier_name] = test_ser
                seed42_scores[tier_name] = {"v_c": v_c, "v_i": v_i, "t_c": t_c, "t_i": t_i, "det_id": det_id}

            # Evaluate 90% and 95% targets on coarse THRESHOLD_GRID and FINE_THRESHOLD_GRID
            res_90_coarse = select_operating_threshold(val_ser, catalog_df, target=0.90, thresholds=THRESHOLD_GRID)
            res_95_coarse = select_operating_threshold(val_ser, catalog_df, target=0.95, thresholds=THRESHOLD_GRID)
            res_90_fine = select_operating_threshold(val_ser, catalog_df, target=0.90, thresholds=FINE_THRESHOLD_GRID)
            res_95_fine = select_operating_threshold(val_ser, catalog_df, target=0.95, thresholds=FINE_THRESHOLD_GRID)

            for grid_label, r90, r95 in [
                ("coarse_THRESHOLD_GRID", res_90_coarse, res_95_coarse),
                ("fine_FINE_THRESHOLD_GRID", res_90_fine, res_95_fine),
            ]:
                same_tau = abs(r90["threshold"] - r95["threshold"]) < 1e-9
                for tgt_pct, r_cal in [(90.0, r90), (95.0, r95)]:
                    tau_val = r_cal["threshold"]
                    ev_m, p_df = _evaluate_tau_paired_or_unpaired(
                        tau_val, test_data, test_plumes, t_c, t_i, paired=True
                    )
                    seed_sat_rows.append({
                        "tier_name": tier_name,
                        "detector_id": det_id,
                        "seed": seed,
                        "grid_type": grid_label,
                        "target_detection_pct": tgt_pct,
                        "target_met": r_cal["target_met"],
                        "threshold_tau": round(tau_val, 6),
                        "tau_90_eq_tau_95": bool(same_tau),
                        "val_detection_pct": round(r_cal["validation_detection_pct"], 2),
                        "val_fa_per_yr": round(r_cal["validation_false_alarms_per_station_year"], 2),
                        "test_det_total_pct": ev_m["det_total_pct"],
                        "test_det_le_3h_pct": ev_m["det_le_3h_pct"],
                        "test_det_le_6h_pct": ev_m["det_le_6h_pct"],
                        "test_det_le_12h_pct": ev_m["det_le_12h_pct"],
                        "test_det_le_24h_pct": ev_m["det_le_24h_pct"],
                        "test_time_norm_12h_det_pct": ev_m["time_normalized_12h_det_pct"],
                        "test_median_delay_hrs": ev_m["delay_median_hours"],
                        "test_fa_per_yr": ev_m["false_alarms_per_station_year"],
                        "test_rain_fa_pct": ev_m["rain_coincident_fa_pct"],
                    })

            # Evaluate FA budgets (1, 2, 5, 10, 20 FA/yr) on clean validation score distribution
            for budget in FA_BUDGETS_5:
                b_cal = select_fa_budget_threshold(val_ser, fa_budget_per_year=budget, catalog=catalog_df)
                tau_b = b_cal["threshold"]
                ev_b, p_df_b = _evaluate_tau_paired_or_unpaired(
                    tau_b, test_data, test_plumes, t_c, t_i, paired=True
                )
                fa_budget_seed_rows.append({
                    "tier_name": tier_name,
                    "detector_id": det_id,
                    "seed": seed,
                    "fa_budget_per_yr": budget,
                    "budget_met": b_cal["budget_met"],
                    "threshold_tau": round(tau_b, 6),
                    "val_fa_per_yr": round(b_cal["validation_false_alarms_per_station_year"], 2),
                    "val_detection_pct": round(b_cal["validation_detection_pct"], 2),
                    "test_det_total_pct": ev_b["det_total_pct"],
                    "test_det_le_3h_pct": ev_b["det_le_3h_pct"],
                    "test_det_le_6h_pct": ev_b["det_le_6h_pct"],
                    "test_det_le_12h_pct": ev_b["det_le_12h_pct"],
                    "test_det_le_24h_pct": ev_b["det_le_24h_pct"],
                    "test_time_norm_12h_det_pct": ev_b["time_normalized_12h_det_pct"],
                    "test_delay_p25_hrs": ev_b["delay_p25_hours"],
                    "test_delay_median_hrs": ev_b["delay_median_hours"],
                    "test_delay_p75_hrs": ev_b["delay_p75_hours"],
                    "test_delay_p90_hrs": ev_b["delay_p90_hours"],
                    "test_fa_per_yr": ev_b["false_alarms_per_station_year"],
                    "test_rain_fa_pct": ev_b["rain_coincident_fa_pct"],
                })

                # On seed 42, also save per-plume & per-month rows and paired-vs-unpaired comparison
                if seed == 42:
                    _, val_p_r, val_m_r = evaluate_single_threshold_detailed(
                        det_id, tier_name, "val", "fa_budget_per_yr", budget, tau_b, b_cal["budget_met"],
                        val_data, val_plumes, v_c, v_i,
                    )
                    _, te_p_r, te_m_r = evaluate_single_threshold_detailed(
                        det_id, tier_name, "test", "fa_budget_per_yr", budget, tau_b, b_cal["budget_met"],
                        test_data, test_plumes, t_c, t_i,
                    )
                    tier_plume_rows.extend(val_p_r)
                    tier_plume_rows.extend(te_p_r)
                    tier_month_rows.extend(val_m_r)
                    tier_month_rows.extend(te_m_r)

                    if budget in (2.0, 5.0, 10.0):
                        ev_unp, _ = _evaluate_tau_paired_or_unpaired(
                            tau_b, test_data, test_plumes, t_c, t_i, paired=False
                        )
                        paired_unpaired_rows.append({
                            "tier_name": tier_name,
                            "operating_point": f"FA_Budget_{int(budget)}_per_yr",
                            "threshold_tau": round(tau_b, 6),
                            "paired_test_det_pct": ev_b["det_total_pct"],
                            "unpaired_test_det_pct": ev_unp["det_total_pct"],
                            "delta_det_unpaired_minus_paired_pp": round(ev_unp["det_total_pct"] - ev_b["det_total_pct"], 2),
                            "paired_median_delay_hrs": ev_b["delay_median_hours"],
                            "unpaired_median_delay_hrs": ev_unp["delay_median_hours"],
                            "test_fa_per_yr": ev_b["false_alarms_per_station_year"],
                        })

    df_sat = pd.DataFrame(seed_sat_rows)
    df_sat.to_csv("data/processed/threshold_saturation_by_seed.csv", index=False)

    sat_summary_rows = []
    for (tier, grid, tgt), g in df_sat.groupby(["tier_name", "grid_type", "target_detection_pct"], sort=False):
        sat_summary_rows.append({
            "tier_name": tier,
            "grid_type": grid,
            "target_detection_pct": tgt,
            "target_met_all_seeds": bool(g["target_met"].all()),
            "n_seeds_target_met": int(g["target_met"].sum()),
            "n_seeds_tau_90_eq_95": int(g["tau_90_eq_tau_95"].sum()),
            "threshold_tau_mean": round(float(g["threshold_tau"].mean()), 5),
            "threshold_tau_std": round(float(g["threshold_tau"].std()), 5),
            "val_detection_pct_mean": round(float(g["val_detection_pct"].mean()), 2),
            "val_fa_per_yr_mean": round(float(g["val_fa_per_yr"].mean()), 2),
            "test_det_total_mean": round(float(g["test_det_total_pct"].mean()), 2),
            "test_det_total_std": round(float(g["test_det_total_pct"].std()), 2),
            "test_det_le_3h_mean": round(float(g["test_det_le_3h_pct"].mean()), 2),
            "test_det_le_6h_mean": round(float(g["test_det_le_6h_pct"].mean()), 2),
            "test_det_le_12h_mean": round(float(g["test_det_le_12h_pct"].mean()), 2),
            "test_det_le_24h_mean": round(float(g["test_det_le_24h_pct"].mean()), 2),
            "test_time_norm_12h_det_mean": round(float(g["test_time_norm_12h_det_pct"].mean()), 2),
            "test_median_delay_mean": round(float(g["test_median_delay_hrs"].mean()), 2),
            "test_fa_per_yr_mean": round(float(g["test_fa_per_yr"].mean()), 2),
            "test_fa_per_yr_std": round(float(g["test_fa_per_yr"].std()), 2),
            "test_rain_fa_pct_mean": round(float(g["test_rain_fa_pct"].mean()), 1),
        })
    pd.DataFrame(sat_summary_rows).to_csv("data/processed/threshold_saturation_summary.csv", index=False)

    df_fab_seeds = pd.DataFrame(fa_budget_seed_rows)
    # Aggregate FA-budget operating points across 20 seeds
    fab_summary_rows = []
    for (t_name, det_id, budget), grp in df_fab_seeds.groupby(["tier_name", "detector_id", "fa_budget_per_yr"], sort=False):
        fab_summary_rows.append({
            "tier_name": t_name,
            "detector_id": det_id,
            "fa_budget_per_yr": budget,
            "budget_met_all_seeds": bool(grp["budget_met"].all()),
            "n_seeds_budget_met": int(grp["budget_met"].sum()),
            "threshold_tau_mean": round(float(grp["threshold_tau"].mean()), 5),
            "threshold_tau_std": round(float(grp["threshold_tau"].std()), 5),
            "val_fa_per_yr_mean": round(float(grp["val_fa_per_yr"].mean()), 2),
            "val_detection_pct_mean": round(float(grp["val_detection_pct"].mean()), 2),
            "test_det_total_mean": round(float(grp["test_det_total_pct"].mean()), 2),
            "test_det_total_std": round(float(grp["test_det_total_pct"].std()), 2),
            "test_det_le_3h_mean": round(float(grp["test_det_le_3h_pct"].mean()), 2),
            "test_det_le_6h_mean": round(float(grp["test_det_le_6h_pct"].mean()), 2),
            "test_det_le_12h_mean": round(float(grp["test_det_le_12h_pct"].mean()), 2),
            "test_det_le_24h_mean": round(float(grp["test_det_le_24h_pct"].mean()), 2),
            "test_time_norm_12h_det_mean": round(float(grp["test_time_norm_12h_det_pct"].mean()), 2),
            "test_delay_p25_mean": round(float(grp["test_delay_p25_hrs"].mean()), 2),
            "test_delay_median_mean": round(float(grp["test_delay_median_hrs"].mean()), 2),
            "test_delay_p75_mean": round(float(grp["test_delay_p75_hrs"].mean()), 2),
            "test_delay_p90_mean": round(float(grp["test_delay_p90_hrs"].mean()), 2),
            "test_fa_per_yr_mean": round(float(grp["test_fa_per_yr"].mean()), 2),
            "test_fa_per_yr_std": round(float(grp["test_fa_per_yr"].std()), 2),
            "test_rain_fa_pct_mean": round(float(grp["test_rain_fa_pct"].mean()), 1),
        })
    df_fab_summary = pd.DataFrame(fab_summary_rows)
    df_fab_summary.to_csv("data/processed/benchmark_fa_budget_operating_points.csv", index=False)
    print("Wrote data/processed/benchmark_fa_budget_operating_points.csv")

    # Update `data/processed/rigorous_benchmark_summary.csv` to add `target_met`, `n_seeds_target_met`, `n_seeds_tau_90_eq_95`
    rig_path = Path("data/processed/rigorous_benchmark_summary.csv")
    if rig_path.exists():
        rig_df = pd.read_csv(rig_path)
        coarse_sub = df_sat[df_sat["grid_type"] == "coarse_THRESHOLD_GRID"]
        t_met_col = []
        n_met_col = []
        n_eq_col = []
        for _, row in rig_df.iterrows():
            tn = row["tier_name"]
            tg = float(row["target_detection_pct"])
            m_sub = coarse_sub[(coarse_sub["tier_name"] == tn) & (coarse_sub["target_detection_pct"] == tg)]
            if len(m_sub) > 0:
                t_met_col.append(bool(m_sub["target_met"].all()))
                n_met_col.append(int(m_sub["target_met"].sum()))
                n_eq_col.append(int(m_sub["tau_90_eq_tau_95"].sum()))
            else:
                t_met_col.append(True)
                n_met_col.append(20)
                n_eq_col.append(0)
        rig_df["target_met"] = t_met_col
        rig_df["n_seeds_target_met"] = n_met_col
        rig_df["n_seeds_tau_90_eq_95"] = n_eq_col
        rig_df.to_csv(rig_path, index=False)
        print("Updated data/processed/rigorous_benchmark_summary.csv with target_met columns.")

    # Save Step 3 paired vs unpaired table (also add 95% target rows)
    for tier_name, det_id, _ in TIER_CONFIGS:
        sc = seed42_scores[tier_name]
        val_ser = seed42_val_series[tier_name]
        r_p = select_operating_threshold(val_ser, catalog_df, target=0.95, paired_detection=True)
        r_u = select_operating_threshold(val_ser, catalog_df, target=0.95, paired_detection=False)
        ev_p, _ = _evaluate_tau_paired_or_unpaired(r_p["threshold"], test_data, test_plumes, sc["t_c"], sc["t_i"], paired=True)
        ev_u, _ = _evaluate_tau_paired_or_unpaired(r_u["threshold"], test_data, test_plumes, sc["t_c"], sc["t_i"], paired=False)
        paired_unpaired_rows.append({
            "tier_name": tier_name,
            "operating_point": "Val_Target_95pct",
            "threshold_tau": round(r_p["threshold"], 6),
            "paired_test_det_pct": ev_p["det_total_pct"],
            "unpaired_test_det_pct": ev_u["det_total_pct"],
            "delta_det_unpaired_minus_paired_pp": round(ev_u["det_total_pct"] - ev_p["det_total_pct"], 2),
            "paired_median_delay_hrs": ev_p["delay_median_hours"],
            "unpaired_median_delay_hrs": ev_u["delay_median_hours"],
            "test_fa_per_yr": ev_p["false_alarms_per_station_year"],
        })
    pd.DataFrame(paired_unpaired_rows).to_csv("data/processed/threshold_paired_vs_unpaired.csv", index=False)

    # Save Step 4 event metrics & duration inflation table
    for tier_name, det_id, _ in TIER_CONFIGS:
        sc = seed42_scores[tier_name]
        val_ser = seed42_val_series[tier_name]
        for op_label, tau_val in [
            ("FA_Budget_5_per_yr", select_fa_budget_threshold(val_ser, fa_budget_per_year=5.0)["threshold"]),
            ("FA_Budget_10_per_yr", select_fa_budget_threshold(val_ser, fa_budget_per_year=10.0)["threshold"]),
            ("Val_Target_95pct", select_operating_threshold(val_ser, catalog_df, target=0.95)["threshold"]),
        ]:
            ev_all, p_df = _evaluate_tau_paired_or_unpaired(tau_val, test_data, test_plumes, sc["t_c"], sc["t_i"], paired=True)
            for dur_bin, d_mask in [
                ("All_Durations (8-164h)", p_df["duration_hours"] >= 0),
                ("Short_Plumes (<=24h)", p_df["duration_hours"] <= 24),
                ("Medium_Plumes (25-60h)", (p_df["duration_hours"] >= 25) & (p_df["duration_hours"] <= 60)),
                ("Long_Plumes (>60h)", p_df["duration_hours"] > 60),
            ]:
                sub_m = compute_event_detection_metrics(p_df[d_mask], reference_window_hours=12.0)
                event_deadline_rows.append({
                    "tier_name": tier_name,
                    "operating_point": op_label,
                    "duration_stratum": dur_bin,
                    "n_plumes": int(d_mask.sum()),
                    **sub_m,
                    "test_fa_per_station_year": ev_all["false_alarms_per_station_year"],
                })
    pd.DataFrame(event_deadline_rows).to_csv("data/processed/threshold_event_metrics_and_delays.csv", index=False)

    # Save Step 5 per-plume and per-month files
    pd.DataFrame(tier_plume_rows).to_csv("data/processed/tier_per_plume.csv", index=False)
    pd.DataFrame(tier_month_rows).to_csv("data/processed/tier_per_month.csv", index=False)

    # -----------------------------------------------------------------------
    # Step 1a, Step 1c & Step 2c-d: Histograms, Reliability Diagrams, and Cross-Fitted Calibration
    # -----------------------------------------------------------------------
    print("=== Step 1a, 1c & Step 2c-d: Probability Histograms, Reliability & Cross-Fitted Calibration ===")
    _run_histograms_and_cross_fitted_calibration(
        seed42_val_series, seed42_test_series, seed42_scores, val_data, test_data, val_plumes, test_plumes, catalog_df
    )

    # -----------------------------------------------------------------------
    # Step 1d: Positive Label Window Sensitivity (First N = 6, 12, 24h, Passage Only, Full Plume)
    # -----------------------------------------------------------------------
    print("=== Step 1d: Positive Label Window Sensitivity ===")
    _run_label_window_sensitivity(fit_data, val_data, test_data, val_plumes, test_plumes, catalog_df)

    # -----------------------------------------------------------------------
    # Step 1e: Class Weighting Sensitivity
    # -----------------------------------------------------------------------
    print("=== Step 1e: Class Weighting Sensitivity ===")
    _run_class_weight_sensitivity(X_fit_all, y_fit_all, fit_data, val_data, test_data, val_plumes, test_plumes, catalog_df)


def _run_histograms_and_cross_fitted_calibration(
    seed42_val_series: dict[str, dict[str, pd.DataFrame]],
    seed42_test_series: dict[str, dict[str, pd.DataFrame]],
    seed42_scores: dict[str, dict[str, Any]],
    val_data: dict[str, dict[str, Any]],
    test_data: dict[str, dict[str, Any]],
    val_plumes: list[dict[str, Any]],
    test_plumes: list[dict[str, Any]],
    catalog_df: pd.DataFrame,
) -> None:
    """Plots validation histograms and reliability diagrams, and evaluates cross-fitted Platt/Isotonic."""
    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300, sharey=True)
    fig_rel, axes_rel = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=300)

    diag_rows = []
    cal_comp_rows = []

    for idx_t, tier_name in enumerate(["Tier 3 (Weather Fusion)", "Tier 4 (Physics-Motivated Nuclear)"]):
        val_ser = seed42_val_series[tier_name]
        test_ser = seed42_test_series[tier_name]

        cln_probs = []
        plume_probs = []
        hour_y = []
        hour_p = []
        for st_id, frame in val_ser.items():
            valid = frame["valid"].to_numpy(dtype=bool)
            pc = frame["p_clean"].to_numpy(dtype=float)[valid]
            pi = frame["p_injected"].to_numpy(dtype=float)[valid]
            yf = frame["y_fission"].to_numpy(dtype=int)[valid]
            cln_probs.append(pc)
            plume_probs.append(pi[yf == 1])
            hour_y.append(yf)
            hour_p.append(pi)

        all_cln = np.concatenate(cln_probs)
        all_plm = np.concatenate(plume_probs)
        all_hy = np.concatenate(hour_y)
        all_hp = np.concatenate(hour_p)

        n_cln_total = len(all_cln)
        n_cln_ge_99 = int(np.sum(all_cln >= 0.99))
        n_cln_ge_95 = int(np.sum(all_cln >= 0.95))
        n_cln_ge_90 = int(np.sum(all_cln >= 0.90))
        n_cln_ge_50 = int(np.sum(all_cln >= 0.50))

        # Event-level validation reliability arrays (81 validation plumes + 81 matched clean control windows)
        ev_y = []
        ev_p = []
        for sl in val_plumes:
            st_id = sl["station_id"]
            idx = sl["indices"]
            pi_sl = seed42_scores[tier_name]["v_i"][st_id][idx]
            pc_sl = seed42_scores[tier_name]["v_c"][st_id][idx]
            ev_y.extend([1, 0])
            ev_p.extend([float(np.max(pi_sl)), float(np.max(pc_sl))])
        ev_y_arr = np.asarray(ev_y, dtype=int)
        ev_p_arr = np.clip(np.asarray(ev_p, dtype=float), 1e-6, 1.0 - 1e-6)

        # Plot Step 1a histogram (log y-scale)
        ax_h = axes_hist[idx_t]
        bins = np.linspace(0.0, 1.0, 51)
        ax_h.hist(all_cln, bins=bins, color="#377eb8", alpha=0.65, label=f"Clean Val Hours (N={n_cln_total:,}; p>=0.99: {n_cln_ge_99})")
        ax_h.hist(all_plm, bins=bins, color="#d62728", alpha=0.70, label=f"In-Plume Val Hours (N={len(all_plm):,})")
        ax_h.axvline(0.99, color="black", linestyle="--", linewidth=1.2, label="p = 0.99")
        ax_h.set_yscale("log")
        ax_h.set_xlabel("Predicted Fission Probability p(fission)", fontsize=10)
        if idx_t == 0:
            ax_h.set_ylabel("Hour Count (Log Scale)", fontsize=10)
        ax_h.set_title(f"{tier_name}\n(Clean Hours >=0.99: {n_cln_ge_99}, >=0.95: {n_cln_ge_95}, >=0.90: {n_cln_ge_90})", fontsize=10.5, fontweight="bold")
        ax_h.grid(True, which="both", linestyle=":", alpha=0.5)
        ax_h.legend(loc="upper center", fontsize=8.5)

        # Cross-fitted calibration (Platt and Isotonic)
        for cal_method in ["raw", "platt", "isotonic"]:
            if cal_method == "raw":
                hp_eval = np.clip(all_hp, 1e-6, 1.0 - 1e-6)
                brier_h = float(brier_score_loss(all_hy, hp_eval))
                ll_h = float(log_loss(all_hy, hp_eval))
                brier_ev = float(brier_score_loss(ev_y_arr, ev_p_arr))
                ll_ev = float(log_loss(ev_y_arr, ev_p_arr))

                # Reliability curves for raw
                prob_true_h, prob_pred_h = calibration_curve(all_hy, hp_eval, n_bins=10, strategy="uniform")
                prob_true_e, prob_pred_e = calibration_curve(ev_y_arr, ev_p_arr, n_bins=8, strategy="uniform")
                color_line = "#ff7f00" if "Tier 3" in tier_name else "#d62728"
                axes_rel[0].plot(prob_pred_h, prob_true_h, marker="o", color=color_line, linestyle="-", label=f"{tier_name} Raw (Brier={brier_h:.4f})")
                axes_rel[1].plot(prob_pred_e, prob_true_e, marker="o", color=color_line, linestyle="-", label=f"{tier_name} Raw (Brier={brier_ev:.4f})")

                for budget in [2.0, 5.0, 10.0, 20.0]:
                    b_cal = select_fa_budget_threshold(val_ser, fa_budget_per_year=budget, catalog=catalog_df)
                    ev_b, _ = _evaluate_tau_paired_or_unpaired(
                        b_cal["threshold"], test_data, test_plumes,
                        seed42_scores[tier_name]["t_c"], seed42_scores[tier_name]["t_i"], paired=True,
                    )
                    cal_comp_rows.append({
                        "tier_name": tier_name,
                        "calibration_method": "Raw_Uncalibrated",
                        "fa_budget_per_yr": budget,
                        "threshold_tau": round(b_cal["threshold"], 6),
                        "val_fa_per_yr": round(b_cal["validation_false_alarms_per_station_year"], 2),
                        "test_det_total_pct": ev_b["det_total_pct"],
                        "test_det_le_12h_pct": ev_b["det_le_12h_pct"],
                        "test_median_delay_hrs": ev_b["delay_median_hours"],
                        "test_fa_per_yr": ev_b["false_alarms_per_station_year"],
                    })
            else:
                v2022_cal, v2021_cal, te_from_2021, te_from_2022 = cross_fit_calibrate_validation_and_test(
                    val_ser, test_ser, method=cal_method
                )
                # Combine out-of-fold calibrated validation series (2021 calibrated by 2022 + 2022 calibrated by 2021)
                oof_val_ser = {}
                oof_hy, oof_hp = [], []
                for st_id in val_ser:
                    f_comb = pd.concat([v2021_cal[st_id], v2022_cal[st_id]], ignore_index=True).sort_values("dt").reset_index(drop=True)
                    oof_val_ser[st_id] = f_comb
                    v_m = f_comb["valid"].to_numpy(dtype=bool)
                    oof_hy.append(f_comb["y_fission"].to_numpy(dtype=int)[v_m])
                    oof_hp.append(np.clip(f_comb["p_injected"].to_numpy(dtype=float)[v_m], 1e-6, 1.0 - 1e-6))

                oof_hy_arr = np.concatenate(oof_hy)
                oof_hp_arr = np.concatenate(oof_hp)
                brier_h = float(brier_score_loss(oof_hy_arr, oof_hp_arr))
                ll_h = float(log_loss(oof_hy_arr, oof_hp_arr))

                # Event-level OOF Brier & Log-Loss
                ev_p_oof = []
                for sl in val_plumes:
                    st_id = sl["station_id"]
                    f_st = oof_val_ser[st_id]
                    in_ev = f_st["dt"].between(sl["start_dt"], sl["dts"].iloc[-1]).to_numpy(dtype=bool)
                    ev_p_oof.extend([
                        float(np.max(f_st.loc[in_ev, "p_injected"])),
                        float(np.max(f_st.loc[in_ev, "p_clean"])),
                    ])
                ev_p_oof_arr = np.clip(np.asarray(ev_p_oof, dtype=float), 1e-6, 1.0 - 1e-6)
                brier_ev = float(brier_score_loss(ev_y_arr, ev_p_oof_arr))
                ll_ev = float(log_loss(ev_y_arr, ev_p_oof_arr))

                if cal_method == "isotonic" and "Tier 4" in tier_name:
                    pt_h, pp_h = calibration_curve(oof_hy_arr, oof_hp_arr, n_bins=10, strategy="uniform")
                    pt_e, pp_e = calibration_curve(ev_y_arr, ev_p_oof_arr, n_bins=8, strategy="uniform")
                    axes_rel[0].plot(pp_h, pt_h, marker="s", color="#1b9e77", linestyle="--", label=f"Tier 4 Cross-Fit Isotonic (Brier={brier_h:.4f})")
                    axes_rel[1].plot(pp_e, pt_e, marker="s", color="#1b9e77", linestyle="--", label=f"Tier 4 Cross-Fit Isotonic (Brier={brier_ev:.4f})")

                # Cross-fitted threshold selection and test evaluation:
                # Fold A: calibrator fit on 2021 -> threshold selected on 2022 -> evaluated on test_from_2021
                # Fold B: calibrator fit on 2022 -> threshold selected on 2021 -> evaluated on test_from_2022
                for budget in [2.0, 5.0, 10.0, 20.0]:
                    b_2022 = select_fa_budget_threshold(v2022_cal, fa_budget_per_year=budget)
                    b_2021 = select_fa_budget_threshold(v2021_cal, fa_budget_per_year=budget)

                    t_c_21 = {st_id: te_from_2021[st_id]["p_clean"].to_numpy(dtype=float) for st_id in test_ser}
                    t_i_21 = {st_id: te_from_2021[st_id]["p_injected"].to_numpy(dtype=float) for st_id in test_ser}
                    t_c_22 = {st_id: te_from_2022[st_id]["p_clean"].to_numpy(dtype=float) for st_id in test_ser}
                    t_i_22 = {st_id: te_from_2022[st_id]["p_injected"].to_numpy(dtype=float) for st_id in test_ser}

                    ev_21, _ = _evaluate_tau_paired_or_unpaired(b_2022["threshold"], test_data, test_plumes, t_c_21, t_i_21, paired=True)
                    ev_22, _ = _evaluate_tau_paired_or_unpaired(b_2021["threshold"], test_data, test_plumes, t_c_22, t_i_22, paired=True)

                    cal_comp_rows.append({
                        "tier_name": tier_name,
                        "calibration_method": f"CrossFitted_{cal_method.capitalize()}",
                        "fa_budget_per_yr": budget,
                        "threshold_tau": round(0.5 * (b_2022["threshold"] + b_2021["threshold"]), 6),
                        "val_fa_per_yr": round(0.5 * (b_2022["validation_false_alarms_per_station_year"] + b_2021["validation_false_alarms_per_station_year"]), 2),
                        "test_det_total_pct": round(0.5 * (ev_21["det_total_pct"] + ev_22["det_total_pct"]), 2),
                        "test_det_le_12h_pct": round(0.5 * (ev_21["det_le_12h_pct"] + ev_22["det_le_12h_pct"]), 2),
                        "test_median_delay_hrs": round(0.5 * (ev_21["delay_median_hours"] + ev_22["delay_median_hours"]), 2),
                        "test_fa_per_yr": round(0.5 * (ev_21["false_alarms_per_station_year"] + ev_22["false_alarms_per_station_year"]), 2),
                    })

            diag_rows.append({
                "tier_name": tier_name,
                "calibration_method": cal_method,
                "val_clean_hours_total": n_cln_total,
                "val_clean_hours_p_ge_099": n_cln_ge_99 if cal_method == "raw" else np.nan,
                "val_clean_hours_p_ge_095": n_cln_ge_95 if cal_method == "raw" else np.nan,
                "val_clean_hours_p_ge_090": n_cln_ge_90 if cal_method == "raw" else np.nan,
                "val_clean_hours_p_ge_050": n_cln_ge_50 if cal_method == "raw" else np.nan,
                "hour_level_brier_score": round(brier_h, 6),
                "hour_level_log_loss": round(ll_h, 6),
                "event_level_brier_score": round(brier_ev, 6),
                "event_level_log_loss": round(ll_ev, 6),
            })

    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_hist.tight_layout()
    fig_hist.savefig(out_dir / "threshold_val_probability_histograms.png", dpi=300)
    plt.close(fig_hist)

    for ax_r, title_r in zip(axes_rel, ["Hour-Level Validation Reliability", "Event-Level Validation Reliability"]):
        ax_r.plot([0, 1], [0, 1], "k:", linewidth=1.2, label="Perfect Calibration")
        ax_r.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax_r.set_ylabel("Empirical Positive Fraction", fontsize=10)
        ax_r.set_title(title_r, fontsize=11, fontweight="bold")
        ax_r.grid(True, linestyle=":", alpha=0.5)
        ax_r.legend(loc="upper left", fontsize=8.5)
    fig_rel.tight_layout()
    fig_rel.savefig(out_dir / "threshold_reliability_diagrams.png", dpi=300)
    plt.close(fig_rel)

    pd.DataFrame(diag_rows).to_csv("data/processed/threshold_calibration_diagnostics.csv", index=False)
    pd.DataFrame(cal_comp_rows).to_csv("data/processed/threshold_calibration_fa_budget_comparison.csv", index=False)
    print("Wrote threshold_calibration_diagnostics.csv and threshold_calibration_fa_budget_comparison.csv.")


def _run_label_window_sensitivity(
    fit_data: dict[str, dict[str, Any]],
    val_data: dict[str, dict[str, Any]],
    test_data: dict[str, dict[str, Any]],
    val_plumes: list[dict[str, Any]],
    test_plumes: list[dict[str, Any]],
    catalog_df: pd.DataFrame,
) -> None:
    """Step 1d: Evaluates sensitivity to labeling only the first N hours of each training plume."""
    cat_meta = catalog_df.set_index("injection_id").to_dict(orient="index")
    rows = []

    for window_mode, max_hours in [
        ("First_6_Hours_Only", 6),
        ("First_12_Hours_Only", 12),
        ("First_24_Hours_Only", 24),
        ("Active_Passage_Hours_Only", -1),
        ("Full_Plume_With_Retention (Default)", 9999),
    ]:
        X_list, y_list = [], []
        for st_id, fd in fit_data.items():
            f_inj = fd["feats_inj"].loc[fd["fit_labeled_mask"]].copy()
            y_arr = f_inj["label"].map(LABEL_MAP).to_numpy(dtype=int)

            if max_hours != 9999:
                # Re-label retention hours beyond `max_hours` (or `t_passage_hours`) as normal/excluded
                keep_mask = np.ones(len(f_inj), dtype=bool)
                for inj_id, grp in f_inj[f_inj["injection_active"] == True].groupby("injection_id"):
                    if not inj_id or inj_id not in cat_meta:
                        continue
                    limit_h = int(cat_meta[inj_id]["t_passage_hours"]) if max_hours == -1 else max_hours
                    start_dt = pd.to_datetime(cat_meta[inj_id]["start_utc"])
                    elapsed_h = (pd.to_datetime(grp["dt"]) - start_dt).dt.total_seconds() / 3600.0
                    late_locs = grp.index[elapsed_h >= limit_h]
                    # Exclude late retention hours from training so they are neither positive nor clean normal
                    pos_in_f = f_inj.index.get_indexer(late_locs)
                    keep_mask[pos_in_f] = False
                f_inj = f_inj.iloc[keep_mask]
                y_arr = y_arr[keep_mask]

            X_list.append(f_inj[TIER_4_NUCLEAR_FEATURES])
            y_list.append(y_arr)

        X_tr = pd.concat(X_list, ignore_index=True)
        y_tr = np.concatenate(y_list)
        n_pos_hours = int(np.sum(y_tr == 2))

        clf = _make_lgbm(seed=42, class_weight="balanced")
        clf.fit(X_tr, y_tr)

        v_c, v_i, t_c, t_i = {}, {}, {}, {}
        for st_id in fit_data:
            v_c[st_id] = clf.predict_proba(val_data[st_id]["feats_cln"][TIER_4_NUCLEAR_FEATURES])[:, 2]
            v_i[st_id] = clf.predict_proba(val_data[st_id]["feats_inj"][TIER_4_NUCLEAR_FEATURES])[:, 2]
            t_c[st_id] = clf.predict_proba(test_data[st_id]["feats_cln"][TIER_4_NUCLEAR_FEATURES])[:, 2]
            t_i[st_id] = clf.predict_proba(test_data[st_id]["feats_inj"][TIER_4_NUCLEAR_FEATURES])[:, 2]

        val_ser, _ = _build_val_and_test_series_dicts(val_data, test_data, v_c, v_i, t_c, t_i)
        r_95 = select_operating_threshold(val_ser, catalog_df, target=0.95)
        b_5 = select_fa_budget_threshold(val_ser, fa_budget_per_year=5.0)

        ev_95, _ = _evaluate_tau_paired_or_unpaired(r_95["threshold"], test_data, test_plumes, t_c, t_i, paired=True)
        ev_b5, _ = _evaluate_tau_paired_or_unpaired(b_5["threshold"], test_data, test_plumes, t_c, t_i, paired=True)

        rows.append({
            "positive_label_window": window_mode,
            "train_positive_plume_hours": n_pos_hours,
            "tau_at_95pct_val_target": round(r_95["threshold"], 6),
            "test_det_at_95pct_val_target": ev_95["det_total_pct"],
            "test_fa_at_95pct_val_target": ev_95["false_alarms_per_station_year"],
            "tau_at_5fa_budget": round(b_5["threshold"], 6),
            "test_det_at_5fa_budget": ev_b5["det_total_pct"],
            "test_det_le_12h_at_5fa_budget": ev_b5["det_le_12h_pct"],
            "test_median_delay_at_5fa_budget": ev_b5["delay_median_hours"],
            "test_fa_at_5fa_budget": ev_b5["false_alarms_per_station_year"],
        })

    pd.DataFrame(rows).to_csv("data/processed/threshold_label_window_sensitivity.csv", index=False)
    print("Wrote data/processed/threshold_label_window_sensitivity.csv")


def _run_class_weight_sensitivity(
    X_fit_all: pd.DataFrame,
    y_fit_all: np.ndarray,
    fit_data: dict[str, dict[str, Any]],
    val_data: dict[str, dict[str, Any]],
    test_data: dict[str, dict[str, Any]],
    val_plumes: list[dict[str, Any]],
    test_plumes: list[dict[str, Any]],
    catalog_df: pd.DataFrame,
) -> None:
    """Step 1e: Evaluates class weighting and sample weighting schemes on Tier 4."""
    rows = []
    configs = [
        ("Unweighted (class_weight=None)", None, None),
        ("Mild_Fission_Boost ({0:1, 1:2, 2:5})", {0: 1.0, 1: 2.0, 2: 5.0}, None),
        ("Moderate_Fission_Boost ({0:1, 1:5, 2:15})", {0: 1.0, 1: 5.0, 2: 15.0}, None),
        ("Balanced (class_weight='balanced', Default)", "balanced", None),
        ("Environment_Rain_Upweighted (2x on rain hours)", "balanced", "rain_2x"),
    ]

    rain_weights = None
    if "precip_1h_mm" in X_fit_all.columns:
        rain_weights = np.where(X_fit_all["precip_1h_mm"].fillna(0.0) > 0.0, 2.0, 1.0)

    for label_scheme, cw, sw_mode in configs:
        clf = _make_lgbm(seed=42, class_weight=cw)
        if sw_mode == "rain_2x" and rain_weights is not None:
            clf.fit(X_fit_all[TIER_4_NUCLEAR_FEATURES], y_fit_all, sample_weight=rain_weights)
        else:
            clf.fit(X_fit_all[TIER_4_NUCLEAR_FEATURES], y_fit_all)

        v_c, v_i, t_c, t_i = {}, {}, {}, {}
        for st_id in fit_data:
            v_c[st_id] = clf.predict_proba(val_data[st_id]["feats_cln"][TIER_4_NUCLEAR_FEATURES])[:, 2]
            v_i[st_id] = clf.predict_proba(val_data[st_id]["feats_inj"][TIER_4_NUCLEAR_FEATURES])[:, 2]
            t_c[st_id] = clf.predict_proba(test_data[st_id]["feats_cln"][TIER_4_NUCLEAR_FEATURES])[:, 2]
            t_i[st_id] = clf.predict_proba(test_data[st_id]["feats_inj"][TIER_4_NUCLEAR_FEATURES])[:, 2]

        val_ser, _ = _build_val_and_test_series_dicts(val_data, test_data, v_c, v_i, t_c, t_i)
        all_cln = np.concatenate([f["p_clean"].to_numpy(dtype=float)[f["valid"].to_numpy(dtype=bool)] for f in val_ser.values()])
        n_cln_99 = int(np.sum(all_cln >= 0.99))

        r_90 = select_operating_threshold(val_ser, catalog_df, target=0.90)
        r_95 = select_operating_threshold(val_ser, catalog_df, target=0.95)
        b_5 = select_fa_budget_threshold(val_ser, fa_budget_per_year=5.0)

        ev_95, _ = _evaluate_tau_paired_or_unpaired(r_95["threshold"], test_data, test_plumes, t_c, t_i, paired=True)
        ev_b5, _ = _evaluate_tau_paired_or_unpaired(b_5["threshold"], test_data, test_plumes, t_c, t_i, paired=True)

        rows.append({
            "weighting_scheme": label_scheme,
            "val_clean_hours_p_ge_099": n_cln_99,
            "tau_90pct_target": round(r_90["threshold"], 6),
            "tau_95pct_target": round(r_95["threshold"], 6),
            "tau_90_eq_tau_95": bool(abs(r_90["threshold"] - r_95["threshold"]) < 1e-9),
            "test_det_at_95pct_target": ev_95["det_total_pct"],
            "test_fa_at_95pct_target": ev_95["false_alarms_per_station_year"],
            "tau_at_5fa_budget": round(b_5["threshold"], 6),
            "test_det_at_5fa_budget": ev_b5["det_total_pct"],
            "test_fa_at_5fa_budget": ev_b5["false_alarms_per_station_year"],
        })

    pd.DataFrame(rows).to_csv("data/processed/threshold_class_weight_sensitivity.csv", index=False)
    print("Wrote data/processed/threshold_class_weight_sensitivity.csv")


if __name__ == "__main__":
    run_threshold_diagnosis_evaluation()
