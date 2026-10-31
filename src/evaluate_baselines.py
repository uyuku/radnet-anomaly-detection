"""
Evaluation Driver for Stronger Baselines & ML Feature Ablations.

Fits all baseline detectors (B1-B8), ML tiers (Tiers 1, 1b, 2, 3, 4), and ML feature
ablations strictly on the 2017-2020 fit fold (`dt < 2021-01-01`), selects operating
thresholds on the 2021-2022 validation fold under both:
  1. Validation-matched detection targets (80%, 90%, 95%)
  2. Validation-matched false-alarm budgets (5, 10, 50 FA/station-year)
and evaluates out-of-sample performance on the 2023-2025 test split.

Outputs:
  - `data/processed/baseline_per_plume.csv`
  - `data/processed/baseline_per_month.csv`
  - `data/processed/baseline_matched_detection_table.csv`
  - `data/processed/baseline_matched_fa_table.csv`
  - `data/processed/ml_ablation_summary.csv`
  - `data/processed/baseline_roc_curve_points.csv`
  - `reports/figures/baseline_matched_roc_curves.png`
  - `reports/figures/rigorous_matched_roc_curves.png`
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

from src.baselines import (
    EPA_RADNET_ALERT_DOCUMENTATION_NOTE,
    fit_b1_b2_b8_station_params,
    fit_b3_weather_regression,
    fit_b5a_pca_subspace,
    fit_b5b_nmf_subspace,
    fit_b6_unsupervised_background,
    fit_b7_simple_supervised,
    score_b1a_rolling168h_z,
    score_b1b_global_dry_z,
    score_b2_rain_masked_z,
    score_b3_weather_regression,
    score_b4a_radon_projection_global,
    score_b4b_radon_projection_local168h,
    score_b4c_radon_projection_combined,
    score_b5a_pca_subspace,
    score_b5b_nmf_subspace,
    score_b6_unsupervised,
    score_b7_simple_supervised,
    score_b8_operational_screening_approx,
)
from src.feature_engineering import (
    NUCLEAR_PHYSICS_FEATURES,
    TIER_1_FEATURES,
    TIER_1B_FEATURES,
    TIER_2_FEATURES,
    TIER_3_FEATURES,
    TIER_4_NUCLEAR_FEATURES,
    WEATHER_FEATURES,
    compute_features_for_series,
)
from src.train_and_evaluate_models import LABEL_MAP, STATIONS

INTERACTION_FEATURES = [
    "rain_high_energy_interaction",
    "dry_excess_interaction",
    "washout_expected_ratio",
]
NUCLEAR_STRIPPING_FEATURES = [
    "radon_stripping_residual_l2",
    "stripped_excess_r03_z",
    "stripped_excess_r05_z",
    "stripped_excess_r07_z",
    "local_stripped_r03_z",
    "local_stripped_r05_z",
    "local_stripped_r07_z",
    "local_radon_stripping_residual_l2",
    "dose_spectral_residual_z",
]
NUCLEAR_BATEMAN_WASHOUT_FEATURES = [
    "bateman_washout_residual_ratio",
    "post_rain_persistence_anomaly",
]

HOURS_PER_YEAR = 8766.0
DETECTION_TARGETS = [80.0, 90.0, 95.0]
FA_BUDGETS = [5.0, 10.0, 50.0]


def _build_adaptive_threshold_grid(
    clean_val_scores: np.ndarray,
    inj_val_scores: np.ndarray,
    n_grid: int = 1200,
) -> np.ndarray:
    """
    Constructs a dense adaptive threshold grid spanning the full range of clean and
    injected validation scores, with high density in the upper tail of clean scores.
    """
    finite_cln = clean_val_scores[np.isfinite(clean_val_scores)]
    finite_inj = inj_val_scores[np.isfinite(inj_val_scores)]
    if len(finite_cln) == 0:
        return np.linspace(0.0, 1.0, n_grid)

    q_low = np.quantile(finite_cln, np.linspace(0.01, 0.95, 300))
    q_high = np.quantile(finite_cln, np.linspace(0.95, 0.99995, 600))
    q_inj = np.quantile(finite_inj, np.linspace(0.01, 0.999, 300)) if len(finite_inj) > 0 else q_low
    hi_max = max(float(np.max(finite_cln)), float(np.max(finite_inj)) if len(finite_inj) > 0 else 1.0)
    extra = np.linspace(float(np.quantile(finite_cln, 0.99)), hi_max + 1e-6, 150)

    grid = np.unique(np.concatenate([q_low, q_high, q_inj, extra]))
    return np.sort(grid)


def _evaluate_grid_on_split(
    split_station_data: dict[str, dict[str, Any]],
    plume_slices: list[dict[str, Any]],
    scores_clean: dict[str, np.ndarray],
    scores_inj: dict[str, np.ndarray],
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized computation of (detection_fraction_grid, fa_per_year_grid) across `grid`
    for a given split ('val' or 'test').
    """
    n_g = len(grid)
    det_counts = np.zeros(n_g, dtype=float)
    for sl in plume_slices:
        st_id = sl["station_id"]
        idx = sl["indices"]
        p_i = scores_inj[st_id][idx]
        p_c = scores_clean[st_id][idx]
        det_any = ((p_i[:, None] >= grid[None, :]) & (p_c[:, None] < grid[None, :])).any(axis=0)
        det_counts += det_any.astype(float)
    det_frac = det_counts / max(1, len(plume_slices))

    tot_ep = np.zeros(n_g, dtype=float)
    tot_obs_hours = 0
    for st_id, st_d in split_station_data.items():
        m_cal = st_d["calendar_mask"]
        val_m = st_d["valid_mask"]
        tot_obs_hours += int(val_m.sum())
        p_c = scores_clean[st_id][m_cal]
        is_on = (p_c[:, None] >= grid[None, :]) & val_m[:, None]
        prior = np.vstack([np.zeros((1, n_g), dtype=bool), is_on[:-1]])
        tot_ep += (is_on & (~prior)).sum(axis=0)

    st_years = max(1e-6, tot_obs_hours / HOURS_PER_YEAR)
    fa_per_yr = tot_ep / st_years
    return det_frac, fa_per_yr


def select_threshold_for_detection_target(
    grid: np.ndarray,
    val_det_frac: np.ndarray,
    val_fa_per_yr: np.ndarray,
    target_det_pct: float,
) -> tuple[float, bool, float, float]:
    """
    Selects threshold tau on validation to meet `target_det_pct` (e.g. 90.0%) while
    minimizing validation false alarms/yr (matching `src/operating_threshold.py`).
    Returns (tau, target_met, val_det_pct, val_fa_yr).
    """
    target_frac = target_det_pct / 100.0
    eligible = np.where(val_det_frac >= target_frac - 1e-9)[0]
    if len(eligible) > 0:
        best_idx = int(eligible[np.argmin(val_fa_per_yr[eligible])])
        target_met = True
    else:
        max_det = np.max(val_det_frac)
        max_idxs = np.where(val_det_frac >= max_det - 1e-9)[0]
        best_idx = int(max_idxs[np.argmin(val_fa_per_yr[max_idxs])])
        target_met = False
    return (
        float(grid[best_idx]),
        bool(target_met),
        round(100.0 * float(val_det_frac[best_idx]), 2),
        round(float(val_fa_per_yr[best_idx]), 2),
    )


def select_threshold_for_fa_budget(
    grid: np.ndarray,
    val_det_frac: np.ndarray,
    val_fa_per_yr: np.ndarray,
    budget_fa_per_yr: float,
) -> tuple[float, bool, float, float]:
    """
    Selects threshold tau on validation with validation FA/yr <= `budget_fa_per_yr`
    that maximizes validation detection, breaking ties by lowest validation FA/yr.
    Returns (tau, budget_met, val_det_pct, val_fa_yr).
    """
    eligible = np.where(val_fa_per_yr <= budget_fa_per_yr + 1e-9)[0]
    if len(eligible) > 0:
        max_det = np.max(val_det_frac[eligible])
        best_candidates = eligible[np.where(val_det_frac[eligible] >= max_det - 1e-9)[0]]
        min_fa = np.min(val_fa_per_yr[best_candidates])
        min_fa_cands = best_candidates[np.where(val_fa_per_yr[best_candidates] <= min_fa + 1e-9)[0]]
        best_idx = int(min_fa_cands[0])
        budget_met = True
    else:
        best_idx = int(np.argmin(val_fa_per_yr))
        budget_met = False
    return (
        float(grid[best_idx]),
        bool(budget_met),
        round(100.0 * float(val_det_frac[best_idx]), 2),
        round(float(val_fa_per_yr[best_idx]), 2),
    )


def evaluate_single_threshold_detailed(
    detector_id: str,
    detector_name: str,
    split_name: str,
    op_type: str,
    op_target: float,
    tau: float,
    val_target_met: bool,
    split_station_data: dict[str, dict[str, Any]],
    plume_slices: list[dict[str, Any]],
    scores_clean: dict[str, np.ndarray],
    scores_inj: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Computes aggregate metrics, per-plume outcomes, and per-station-month false-alarm
    episode counts (counting episode starts on the full station series before binning
    into YYYY-MM blocks so episodes crossing month edges are never double-counted).
    """
    plume_rows = []
    det_count = 0
    det_6h_count = 0
    det_12h_count = 0
    det_24h_count = 0
    delays = []

    for sl in plume_slices:
        st_id = sl["station_id"]
        idx = sl["indices"]
        p_i = scores_inj[st_id][idx]
        p_c = scores_clean[st_id][idx]
        net_al = (p_i >= tau) & (p_c < tau)
        is_det = bool(net_al.any())
        delay_h = np.nan
        if is_det:
            det_count += 1
            first_i = int(np.where(net_al)[0][0])
            delay_h = float((sl["dts"].iloc[first_i] - sl["start_dt"]).total_seconds() / 3600.0)
            delays.append(delay_h)
            if delay_h <= 6.0:
                det_6h_count += 1
            if delay_h <= 12.0:
                det_12h_count += 1
            if delay_h <= 24.0:
                det_24h_count += 1

        plume_rows.append({
            "detector_id": detector_id,
            "detector_name": detector_name,
            "split": split_name,
            "operating_point_type": op_type,
            "operating_target": op_target,
            "threshold_tau": round(tau, 6),
            "val_target_met": bool(val_target_met),
            "plume_id": sl["plume_id"],
            "station_id": st_id,
            "environment": sl["environment"],
            "scenario": sl["scenario"],
            "snr_bin": sl["snr_bin"],
            "peak_snr_168h": sl["peak_snr_168h"],
            "duration_hours": sl["duration_hours"],
            "detected": int(is_det),
            "delay_hours": round(delay_h, 2) if np.isfinite(delay_h) else np.nan,
            "peak_score_in_plume": round(float(np.max(p_i)), 6),
            "max_clean_score_in_plume": round(float(np.max(p_c)), 6),
        })

    month_rows = []
    tot_episodes = 0
    tot_alarm_hours = 0
    tot_rain_alarm_hours = 0
    tot_obs_hours = 0

    for st_id, st_d in split_station_data.items():
        m_cal = st_d["calendar_mask"]
        val_m = st_d["valid_mask"]
        rain_m = st_d["rain_mask"]
        ym_arr = st_d["year_month"]

        p_c = scores_clean[st_id][m_cal]
        is_on = (p_c >= tau) & val_m
        # Episode starts computed on full continuous station series before month binning
        prior = np.concatenate([[False], is_on[:-1]])
        ep_starts = is_on & (~prior)
        rain_al = is_on & rain_m

        tot_episodes += int(ep_starts.sum())
        tot_alarm_hours += int(is_on.sum())
        tot_rain_alarm_hours += int(rain_al.sum())
        tot_obs_hours += int(val_m.sum())

        for ym in pd.unique(ym_arr):
            ym_mask = ym_arr == ym
            obs_h = int(val_m[ym_mask].sum())
            month_rows.append({
                "detector_id": detector_id,
                "detector_name": detector_name,
                "split": split_name,
                "operating_point_type": op_type,
                "operating_target": op_target,
                "threshold_tau": round(tau, 6),
                "station_id": st_id,
                "year_month": str(ym),
                "false_alarm_episodes": int(ep_starts[ym_mask].sum()),
                "false_alarm_hours": int(is_on[ym_mask].sum()),
                "rain_coincident_alarm_hours": int(rain_al[ym_mask].sum()),
                "observed_exposure_hours": obs_h,
                "exposure_station_years": round(obs_h / HOURS_PER_YEAR, 6),
            })

    n_plumes = max(1, len(plume_slices))
    st_years = max(1e-6, tot_obs_hours / HOURS_PER_YEAR)
    summary = {
        "detector_id": detector_id,
        "detector_name": detector_name,
        "split": split_name,
        "operating_point_type": op_type,
        "operating_target": op_target,
        "threshold_tau": round(tau, 6),
        "val_target_met": bool(val_target_met),
        "realized_detection_pct": round(100.0 * det_count / n_plumes, 2),
        "detection_le_6h_pct": round(100.0 * det_6h_count / n_plumes, 2),
        "detection_le_12h_pct": round(100.0 * det_12h_count / n_plumes, 2),
        "detection_le_24h_pct": round(100.0 * det_24h_count / n_plumes, 2),
        "median_delay_hours": round(float(np.median(delays)), 2) if delays else np.nan,
        "false_alarm_episodes": tot_episodes,
        "false_alarms_per_station_year": round(tot_episodes / st_years, 2),
        "rain_coincident_fa_pct": round(100.0 * tot_rain_alarm_hours / tot_alarm_hours, 1) if tot_alarm_hours > 0 else 0.0,
        "observed_station_years": round(st_years, 4),
    }
    return summary, plume_rows, month_rows


def _prepare_all_station_data() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Loads labeled train (2017-2022) and test (2023-2025) datasets across all 5 stations,
    computes causal features (with 250-hour warmup prepended to test), and organizes
    fit (2017-2020), validation (2021-2022), and test (2023-2025) structures.
    """
    cat_df = pd.read_csv("data/processed/synthetic_injection_catalog.csv")
    cat_meta = cat_df.set_index("injection_id").to_dict(orient="index")

    fit_data: dict[str, dict[str, Any]] = {}
    val_data: dict[str, dict[str, Any]] = {}
    test_data: dict[str, dict[str, Any]] = {}
    val_plume_slices: list[dict[str, Any]] = []
    test_plume_slices: list[dict[str, Any]] = []

    for st in STATIONS:
        st_id = st["id"]
        df_tr = pd.read_csv(f"data/processed/labeled_{st_id}_train.csv.gz")
        df_te = pd.read_csv(f"data/processed/labeled_{st_id}_test.csv.gz")
        df_tr["dt"] = pd.to_datetime(df_tr["dt"])
        df_te["dt"] = pd.to_datetime(df_te["dt"])

        feats_tr_inj = compute_features_for_series(df_tr, st_id, use_injected=True)
        feats_tr_cln = compute_features_for_series(df_tr, st_id, use_injected=False)

        warmup_tr = df_tr.tail(250).copy()
        df_te_warm = pd.concat([warmup_tr, df_te], ignore_index=True)
        feats_te_inj = compute_features_for_series(df_te_warm, st_id, use_injected=True).iloc[250:].reset_index(drop=True)
        feats_te_cln = compute_features_for_series(df_te_warm, st_id, use_injected=False).iloc[250:].reset_index(drop=True)

        raw_cols = ["gross_cpm", "dose_rate_nsvh", "inj_gross_cpm", "inj_dose_rate_nsvh"] + [
            f"cpm_r0{k}" for k in range(2, 10)
        ] + [f"inj_cpm_r0{k}" for k in range(2, 10)]
        for rc in raw_cols:
            if rc in df_tr.columns:
                feats_tr_inj[rc] = df_tr[rc].to_numpy(dtype=float)
                feats_tr_cln[rc] = df_tr[rc].to_numpy(dtype=float)
            if rc in df_te.columns:
                feats_te_inj[rc] = df_te[rc].to_numpy(dtype=float)
                feats_te_cln[rc] = df_te[rc].to_numpy(dtype=float)

        valid_tr = (feats_tr_inj["has_radnet_obs"].astype(bool) & feats_tr_inj["rad_complete_channels"].astype(bool)).to_numpy(dtype=bool)
        valid_te = (feats_te_inj["has_radnet_obs"].astype(bool) & feats_te_inj["rad_complete_channels"].astype(bool)).to_numpy(dtype=bool)

        m_fit_cal = (feats_tr_inj["dt"] < "2021-01-01").to_numpy(dtype=bool)
        m_val_cal = (feats_tr_inj["dt"] >= "2021-01-01").to_numpy(dtype=bool)

        fit_data[st_id] = {
            "feats_inj": feats_tr_inj,
            "feats_cln": feats_tr_cln,
            "fit_clean_df": feats_tr_cln.loc[m_fit_cal].reset_index(drop=True),
            "fit_labeled_mask": m_fit_cal & valid_tr & feats_tr_inj["label"].isin(LABEL_MAP.keys()).to_numpy(dtype=bool),
        }

        val_data[st_id] = {
            "feats_inj": feats_tr_inj,
            "feats_cln": feats_tr_cln,
            "calendar_mask": m_val_cal,
            "valid_mask": valid_tr[m_val_cal],
            "rain_mask": (feats_tr_cln.loc[m_val_cal, "precip_1h_mm"].fillna(0.0) > 0.0).to_numpy(dtype=bool),
            "year_month": feats_tr_cln.loc[m_val_cal, "dt"].dt.strftime("%Y-%m").to_numpy(),
        }

        test_data[st_id] = {
            "feats_inj": feats_te_inj,
            "feats_cln": feats_te_cln,
            "calendar_mask": np.ones(len(feats_te_inj), dtype=bool),
            "valid_mask": valid_te,
            "rain_mask": (feats_te_cln["precip_1h_mm"].fillna(0.0) > 0.0).to_numpy(dtype=bool),
            "year_month": feats_te_cln["dt"].dt.strftime("%Y-%m").to_numpy(),
        }

        # Validation plume slices (2021-2022)
        val_sub = feats_tr_inj.loc[m_val_cal & valid_tr]
        for inj_id, grp in val_sub[val_sub["injection_active"] == True].groupby("injection_id"):
            if not inj_id or inj_id not in cat_meta:
                continue
            meta = cat_meta[inj_id]
            val_plume_slices.append({
                "plume_id": str(inj_id),
                "station_id": st_id,
                "indices": grp.index.to_numpy(),
                "dts": grp["dt"],
                "start_dt": pd.to_datetime(meta["start_utc"]),
                "environment": str(meta["environment"]),
                "scenario": str(meta["nuclide_scenario"]),
                "snr_bin": str(meta.get("snr_bin", "")),
                "peak_snr_168h": float(meta.get("peak_snr_168h", np.nan)),
                "duration_hours": int(meta["duration_hours"]),
            })

        # Test plume slices (2023-2025)
        te_sub = feats_te_inj.loc[valid_te]
        for inj_id, grp in te_sub[te_sub["injection_active"] == True].groupby("injection_id"):
            if not inj_id or inj_id not in cat_meta:
                continue
            meta = cat_meta[inj_id]
            test_plume_slices.append({
                "plume_id": str(inj_id),
                "station_id": st_id,
                "indices": grp.index.to_numpy(),
                "dts": grp["dt"],
                "start_dt": pd.to_datetime(meta["start_utc"]),
                "environment": str(meta["environment"]),
                "scenario": str(meta["nuclide_scenario"]),
                "snr_bin": str(meta.get("snr_bin", "")),
                "peak_snr_168h": float(meta.get("peak_snr_168h", np.nan)),
                "duration_hours": int(meta["duration_hours"]),
            })

    return fit_data, val_data, test_data, val_plume_slices, test_plume_slices


def _fit_and_score_all_detectors(
    fit_data: dict[str, dict[str, Any]],
    val_data: dict[str, dict[str, Any]],
    test_data: dict[str, dict[str, Any]],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Fits all baselines B1-B8, ML Tiers 1-4, and ML ablations on 2017-2020 and computes
    hourly clean and injected score arrays on validation (2021-2022) and test (2023-2025).
    """
    # 1. Fit per-station models on 2017-2020 clean background
    b1b2_stats = {}
    b3_lin_models = {}
    b3_gam_models = {}
    b5_pca_models = {}
    b5_nmf_models = {}

    for st_id, fd in fit_data.items():
        cln_fit = fd["fit_clean_df"]
        b1b2_stats[st_id] = fit_b1_b2_b8_station_params(cln_fit)
        b3_lin_models[st_id] = fit_b3_weather_regression(cln_fit, variant="linear")
        b3_gam_models[st_id] = fit_b3_weather_regression(cln_fit, variant="spline_gam")
        b5_pca_models[st_id] = fit_b5a_pca_subspace(cln_fit, n_components=3)
        b5_nmf_models[st_id] = fit_b5b_nmf_subspace(cln_fit, n_components=2)

    # Pooled 2017-2020 clean background for B6 unsupervised ML models
    pooled_clean_fit = pd.concat([fd["fit_clean_df"] for fd in fit_data.values()], ignore_index=True)
    b6a_model = fit_b6_unsupervised_background(pooled_clean_fit, method="iforest_tier3", seed=seed)
    b6b_model = fit_b6_unsupervised_background(pooled_clean_fit, method="iforest_tier4", seed=seed)
    b6c_model = fit_b6_unsupervised_background(pooled_clean_fit, method="ocsvm_tier4", seed=seed)

    # Pooled 2017-2020 labeled dataset for B7 simple supervised models & ML tiers/ablations
    X_fit_all = pd.concat(
        [fd["feats_inj"].loc[fd["fit_labeled_mask"], TIER_4_NUCLEAR_FEATURES] for fd in fit_data.values()],
        ignore_index=True,
    )
    y_fit_all = np.concatenate([
        fd["feats_inj"].loc[fd["fit_labeled_mask"], "label"].map(LABEL_MAP).to_numpy(dtype=int)
        for fd in fit_data.values()
    ])

    b7a_model = fit_b7_simple_supervised(X_fit_all, y_fit_all, method="logreg_tier3", feature_cols=TIER_3_FEATURES, seed=seed)
    b7b_model = fit_b7_simple_supervised(X_fit_all, y_fit_all, method="dtree_depth3_tier3", feature_cols=TIER_3_FEATURES, seed=seed)

    def _score_station_rule(fn_station) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        v_c, v_i, t_c, t_i = {}, {}, {}, {}
        for st_id in fit_data:
            v_c[st_id] = fn_station(val_data[st_id]["feats_cln"], st_id, False)
            v_i[st_id] = fn_station(val_data[st_id]["feats_inj"], st_id, True)
            t_c[st_id] = fn_station(test_data[st_id]["feats_cln"], st_id, False)
            t_i[st_id] = fn_station(test_data[st_id]["feats_inj"], st_id, True)
        return v_c, v_i, t_c, t_i

    detectors = []

    def _add_detector(det_id: str, det_name: str, category: str, uses_plumes: bool, fn_station) -> None:
        v_c, v_i, t_c, t_i = _score_station_rule(fn_station)
        detectors.append({
            "detector_id": det_id,
            "detector_name": det_name,
            "category": category,
            "uses_synthetic_plumes_in_training": uses_plumes,
            "val_clean": v_c,
            "val_inj": v_i,
            "test_clean": t_c,
            "test_inj": t_i,
        })

    # B1-B8 Baseline Detectors
    _add_detector(
        "B1a_Rolling168h_Z", "B1a: Rolling 168h Gross Z-Score", "B1_Gross_Sigma", False,
        lambda df, st_id, is_inj: score_b1a_rolling168h_z(df),
    )
    _add_detector(
        "B1b_GlobalDry_Z", "B1b: Global Dry Gross Z-Score (2017-2020)", "B1_Gross_Sigma", False,
        lambda df, st_id, is_inj: score_b1b_global_dry_z(df, b1b2_stats[st_id], use_injected=is_inj),
    )
    _add_detector(
        "B2a_RainMasked_Holdoff6h", "B2a: Rain-Masked Dry Z (H=6h Holdoff)", "B2_Rain_Masked", False,
        lambda df, st_id, is_inj: score_b2_rain_masked_z(df, b1b2_stats[st_id], holdoff_hours=6.0, use_injected=is_inj),
    )
    _add_detector(
        "B2b_RainMasked_Holdoff12h", "B2b: Rain-Masked Dry Z (H=12h Holdoff)", "B2_Rain_Masked", False,
        lambda df, st_id, is_inj: score_b2_rain_masked_z(df, b1b2_stats[st_id], holdoff_hours=12.0, use_injected=is_inj),
    )
    _add_detector(
        "B3a_WeatherRegression_Linear", "B3a: Weather Regression Residual (Linear)", "B3_Weather_Regression", False,
        lambda df, st_id, is_inj: score_b3_weather_regression(df, b3_lin_models[st_id], use_injected=is_inj),
    )
    _add_detector(
        "B3b_WeatherRegression_SplineGAM", "B3b: Weather Regression Residual (Spline GAM)", "B3_Weather_Regression", False,
        lambda df, st_id, is_inj: score_b3_weather_regression(df, b3_gam_models[st_id], use_injected=is_inj),
    )
    _add_detector(
        "B4a_RadonProjection_Global", "B4a: Radon-Direction Projection (Global Dry Basis)", "B4_Radon_Projection", False,
        lambda df, st_id, is_inj: score_b4a_radon_projection_global(df),
    )
    _add_detector(
        "B4b_RadonProjection_Local168h", "B4b: Radon-Direction Projection (Local 168h Basis)", "B4_Radon_Projection", False,
        lambda df, st_id, is_inj: score_b4b_radon_projection_local168h(df),
    )
    _add_detector(
        "B4c_RadonProjection_Combined", "B4c: Radon Projection + Stripped Channel Peak Z", "B4_Radon_Projection", False,
        lambda df, st_id, is_inj: score_b4c_radon_projection_combined(df),
    )
    _add_detector(
        "B5a_PCA_Subspace_Mahalanobis", "B5a: Background PCA Subspace (Mahalanobis + Recon)", "B5_Background_Subspace", False,
        lambda df, st_id, is_inj: score_b5a_pca_subspace(df, b5_pca_models[st_id], use_injected=is_inj),
    )
    _add_detector(
        "B5b_NMF_Background_Reconstruction", "B5b: Background NMF 2-Component Reconstruction", "B5_Background_Subspace", False,
        lambda df, st_id, is_inj: score_b5b_nmf_subspace(df, b5_nmf_models[st_id], use_injected=is_inj),
    )
    _add_detector(
        "B6a_IsolationForest_Tier3_BgOnly", "B6a: Unsupervised Isolation Forest (Tier 3, Bg Only)", "B6_Unsupervised_ML", False,
        lambda df, st_id, is_inj: score_b6_unsupervised(df, b6a_model),
    )
    _add_detector(
        "B6b_IsolationForest_Tier4_BgOnly", "B6b: Unsupervised Isolation Forest (Tier 4, Bg Only)", "B6_Unsupervised_ML", False,
        lambda df, st_id, is_inj: score_b6_unsupervised(df, b6b_model),
    )
    _add_detector(
        "B6c_OneClassSVM_Tier4_BgOnly", "B6c: Unsupervised One-Class SVM (Tier 4, Bg Only)", "B6_Unsupervised_ML", False,
        lambda df, st_id, is_inj: score_b6_unsupervised(df, b6c_model),
    )
    _add_detector(
        "B7a_LogisticRegression_Tier3", "B7a: Supervised Logistic Regression (Tier 3)", "B7_Simple_Supervised", True,
        lambda df, st_id, is_inj: score_b7_simple_supervised(df, b7a_model),
    )
    _add_detector(
        "B7b_DecisionTreeDepth3_Tier3", "B7b: Supervised Depth-3 Decision Tree (Tier 3)", "B7_Simple_Supervised", True,
        lambda df, st_id, is_inj: score_b7_simple_supervised(df, b7b_model),
    )
    _add_detector(
        "B8_Operational_Screening_Approx_UNVERIFIED",
        "B8: Operational Screening Approximation (Approximate EPA Threshold)",
        "B8_Operational_Rule",
        False,
        lambda df, st_id, is_inj: score_b8_operational_screening_approx(df, b1b2_stats[st_id], use_injected=is_inj),
    )

    # Supervised LightGBM Tiers & Feature Ablations
    tier_4_no_weather = [
        c for c in TIER_4_NUCLEAR_FEATURES
        if c not in WEATHER_FEATURES and c not in INTERACTION_FEATURES and c not in NUCLEAR_BATEMAN_WASHOUT_FEATURES
    ]
    tier_3_plus_stripping_only = TIER_3_FEATURES + NUCLEAR_STRIPPING_FEATURES
    tier_3_plus_bateman_only = TIER_3_FEATURES + NUCLEAR_BATEMAN_WASHOUT_FEATURES

    lgb_configs = [
        ("ML_Tier1_Gross", "Tier 1: Gross GBDT (12 features)", "ML_Main_Tier", TIER_1_FEATURES),
        ("ML_Tier1b_GrossWeather", "Tier 1b: Gross + Weather GBDT (29 features)", "ML_Main_Tier", TIER_1B_FEATURES),
        ("ML_Tier2_Spectral", "Tier 2: Spectral GBDT (30 features)", "ML_Main_Tier", TIER_2_FEATURES),
        ("ML_Tier3_WeatherFusion", "Tier 3: Weather Fusion GBDT (48 features)", "ML_Main_Tier", TIER_3_FEATURES),
        ("ML_Tier4_PhysicsNuclear", "Tier 4: Physics-Motivated Nuclear GBDT (59 features)", "ML_Main_Tier", TIER_4_NUCLEAR_FEATURES),
        ("Ablation_Tier4_NoWeather", "Ablation: Tier 4 without Weather/Bateman (39 features)", "ML_Ablation", tier_4_no_weather),
        ("Ablation_Tier3_PlusStrippingOnly", "Ablation: Tier 3 + Radon Stripping Only (57 features)", "ML_Ablation", tier_3_plus_stripping_only),
        ("Ablation_Tier3_PlusBatemanOnly", "Ablation: Tier 3 + Bateman Washout ODE Only (50 features)", "ML_Ablation", tier_3_plus_bateman_only),
    ]

    for det_id, det_name, cat_name, f_cols in lgb_configs:
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
            random_state=seed,
            n_jobs=4,
            verbose=-1,
        )
        clf.fit(X_fit_all[f_cols], y_fit_all)
        _add_detector(
            det_id, det_name, cat_name, True,
            lambda df, st_id, is_inj, _clf=clf, _cols=f_cols: _clf.predict_proba(df[_cols])[:, 2],
        )

    return detectors


def run_baseline_evaluation() -> None:
    """Executes the full baseline and ML ablation evaluation."""
    print("=== Step 1: Loading and Preparing 2017-2020 Fit, 2021-2022 Val, and 2023-2025 Test Data ===")
    fit_data, val_data, test_data, val_plumes, test_plumes = _prepare_all_station_data()
    print(f"Prepared {len(val_plumes)} validation plumes and {len(test_plumes)} test plumes across 5 stations.")

    print("=== Step 2: Fitting Baselines B1-B8, ML Tiers 1-4, and Feature Ablations on 2017-2020 ===")
    detectors = _fit_and_score_all_detectors(fit_data, val_data, test_data, seed=42)

    all_plume_rows: list[dict[str, Any]] = []
    all_month_rows: list[dict[str, Any]] = []
    matched_det_rows: list[dict[str, Any]] = []
    matched_fa_rows: list[dict[str, Any]] = []
    roc_curve_rows: list[dict[str, Any]] = []

    print("=== Step 3: Evaluating Operating Thresholds & ROC Curves ===")
    for det in detectors:
        det_id = det["detector_id"]
        det_name = det["detector_name"]
        cat_name = det["category"]
        uses_plumes = det["uses_synthetic_plumes_in_training"]

        # Pool validation clean & injected scores to construct adaptive grid
        val_cln_concat = np.concatenate([
            det["val_clean"][st_id][val_data[st_id]["calendar_mask"]][val_data[st_id]["valid_mask"]]
            for st_id in val_data
        ])
        val_inj_concat = np.concatenate([
            det["val_inj"][sl["station_id"]][sl["indices"]]
            for sl in val_plumes
        ])
        grid = _build_adaptive_threshold_grid(val_cln_concat, val_inj_concat, n_grid=1200)
        val_det_frac, val_fa_yr = _evaluate_grid_on_split(
            val_data, val_plumes, det["val_clean"], det["val_inj"], grid
        )
        test_det_frac, test_fa_yr = _evaluate_grid_on_split(
            test_data, test_plumes, det["test_clean"], det["test_inj"], grid
        )

        # Save 60 representative ROC curve points per detector
        roc_idxs = np.unique(np.linspace(0, len(grid) - 1, 60, dtype=int))
        for idx_r in roc_idxs:
            roc_curve_rows.append({
                "detector_id": det_id,
                "detector_name": det_name,
                "category": cat_name,
                "threshold_tau": round(float(grid[idx_r]), 6),
                "val_detection_pct": round(100.0 * float(val_det_frac[idx_r]), 2),
                "val_fa_per_yr": round(float(val_fa_yr[idx_r]), 2),
                "test_detection_pct": round(100.0 * float(test_det_frac[idx_r]), 2),
                "test_fa_per_yr": round(float(test_fa_yr[idx_r]), 2),
            })

        # 1. Validation-Matched Detection Targets (80%, 90%, 95%)
        for target_pct in DETECTION_TARGETS:
            tau, t_met, v_det, v_fa = select_threshold_for_detection_target(
                grid, val_det_frac, val_fa_yr, target_pct
            )
            _, val_p_rows, val_m_rows = evaluate_single_threshold_detailed(
                det_id, det_name, "val", "matched_detection_pct", target_pct, tau, t_met,
                val_data, val_plumes, det["val_clean"], det["val_inj"],
            )
            te_sum, te_p_rows, te_m_rows = evaluate_single_threshold_detailed(
                det_id, det_name, "test", "matched_detection_pct", target_pct, tau, t_met,
                test_data, test_plumes, det["test_clean"], det["test_inj"],
            )
            all_plume_rows.extend(val_p_rows)
            all_plume_rows.extend(te_p_rows)
            all_month_rows.extend(val_m_rows)
            all_month_rows.extend(te_m_rows)

            matched_det_rows.append({
                "detector_id": det_id,
                "detector_name": det_name,
                "category": cat_name,
                "uses_synthetic_plumes": uses_plumes,
                "val_target_detection_pct": target_pct,
                "val_target_met": t_met,
                "val_target_flag": "MET" if t_met else f"UNREACHABLE (max val det {v_det:.1f}%)",
                "threshold_tau": round(tau, 6),
                "val_realized_detection_pct": v_det,
                "val_fa_per_station_year": v_fa,
                "test_realized_detection_pct": te_sum["realized_detection_pct"],
                "test_detection_le_12h_pct": te_sum["detection_le_12h_pct"],
                "test_median_delay_hours": te_sum["median_delay_hours"],
                "test_false_alarm_episodes": te_sum["false_alarm_episodes"],
                "test_fa_per_station_year": te_sum["false_alarms_per_station_year"],
                "test_rain_coincident_fa_pct": te_sum["rain_coincident_fa_pct"],
            })

        # 2. Validation-Matched False-Alarm Budgets (5, 10, 50 FA/yr)
        for budget_fa in FA_BUDGETS:
            tau, b_met, v_det, v_fa = select_threshold_for_fa_budget(
                grid, val_det_frac, val_fa_yr, budget_fa
            )
            _, val_p_rows, val_m_rows = evaluate_single_threshold_detailed(
                det_id, det_name, "val", "matched_fa_budget", budget_fa, tau, b_met,
                val_data, val_plumes, det["val_clean"], det["val_inj"],
            )
            te_sum, te_p_rows, te_m_rows = evaluate_single_threshold_detailed(
                det_id, det_name, "test", "matched_fa_budget", budget_fa, tau, b_met,
                test_data, test_plumes, det["test_clean"], det["test_inj"],
            )
            all_plume_rows.extend(val_p_rows)
            all_plume_rows.extend(te_p_rows)
            all_month_rows.extend(val_m_rows)
            all_month_rows.extend(te_m_rows)

            matched_fa_rows.append({
                "detector_id": det_id,
                "detector_name": det_name,
                "category": cat_name,
                "uses_synthetic_plumes": uses_plumes,
                "val_fa_budget_per_yr": budget_fa,
                "val_budget_met": b_met,
                "val_budget_flag": "MET" if b_met else f"EXCEEDED (min val FA {v_fa:.2f}/yr)",
                "threshold_tau": round(tau, 6),
                "val_realized_detection_pct": v_det,
                "val_fa_per_station_year": v_fa,
                "test_realized_detection_pct": te_sum["realized_detection_pct"],
                "test_detection_le_12h_pct": te_sum["detection_le_12h_pct"],
                "test_median_delay_hours": te_sum["median_delay_hours"],
                "test_false_alarm_episodes": te_sum["false_alarm_episodes"],
                "test_fa_per_station_year": te_sum["false_alarms_per_station_year"],
                "test_rain_coincident_fa_pct": te_sum["rain_coincident_fa_pct"],
            })

        # Print summary at 90% detection and 10 FA/yr budget
        r90 = [r for r in matched_det_rows if r["detector_id"] == det_id and r["val_target_detection_pct"] == 90.0][0]
        b10 = [r for r in matched_fa_rows if r["detector_id"] == det_id and r["val_fa_budget_per_yr"] == 10.0][0]
        print(
            f"  {det_name:<54} | @90%ValDet: TestDet={r90['test_realized_detection_pct']:5.1f}%, "
            f"TestFA={r90['test_fa_per_station_year']:6.2f}/yr ({r90['val_target_flag']}) | "
            f"@10ValFA: TestDet={b10['test_realized_detection_pct']:5.1f}%, TestFA={b10['test_fa_per_station_year']:6.2f}/yr"
        )

    # Save CSV outputs
    df_plume = pd.DataFrame(all_plume_rows)
    df_month = pd.DataFrame(all_month_rows)
    df_mdet = pd.DataFrame(matched_det_rows)
    df_mfa = pd.DataFrame(matched_fa_rows)
    df_roc = pd.DataFrame(roc_curve_rows)

    df_plume.to_csv("data/processed/baseline_per_plume.csv", index=False)
    df_month.to_csv("data/processed/baseline_per_month.csv", index=False)
    df_mdet.to_csv("data/processed/baseline_matched_detection_table.csv", index=False)
    df_mfa.to_csv("data/processed/baseline_matched_fa_table.csv", index=False)
    df_roc.to_csv("data/processed/baseline_roc_curve_points.csv", index=False)

    # Build dedicated ML ablation table (`data/processed/ml_ablation_summary.csv`)
    ablation_ids = [
        "B6a_IsolationForest_Tier3_BgOnly",
        "B6b_IsolationForest_Tier4_BgOnly",
        "B6c_OneClassSVM_Tier4_BgOnly",
        "B4b_RadonProjection_Local168h",
        "B4c_RadonProjection_Combined",
        "B5b_NMF_Background_Reconstruction",
        "B7a_LogisticRegression_Tier3",
        "B7b_DecisionTreeDepth3_Tier3",
        "ML_Tier1_Gross",
        "ML_Tier1b_GrossWeather",
        "ML_Tier2_Spectral",
        "ML_Tier3_WeatherFusion",
        "Ablation_Tier4_NoWeather",
        "Ablation_Tier3_PlusStrippingOnly",
        "Ablation_Tier3_PlusBatemanOnly",
        "ML_Tier4_PhysicsNuclear",
    ]
    df_abl_det = df_mdet[df_mdet["detector_id"].isin(ablation_ids)].copy()
    df_abl_fa = df_mfa[df_mfa["detector_id"].isin(ablation_ids) & (df_mfa["val_fa_budget_per_yr"] == 10.0)][
        ["detector_id", "test_realized_detection_pct", "test_fa_per_station_year"]
    ].rename(columns={
        "test_realized_detection_pct": "test_det_at_10fa_budget_pct",
        "test_fa_per_station_year": "test_fa_at_10fa_budget_per_yr",
    })
    df_abl = df_abl_det.merge(df_abl_fa, on="detector_id", how="left")
    df_abl.to_csv("data/processed/ml_ablation_summary.csv", index=False)

    print("Saved baseline_per_plume.csv, baseline_per_month.csv, baseline_matched_detection_table.csv, baseline_matched_fa_table.csv, ml_ablation_summary.csv.")
    _plot_baseline_roc_curves(df_roc)


def _plot_baseline_roc_curves(df_roc: pd.DataFrame) -> None:
    """Plots ROC-style curves (Realized Test Detection % vs False Alarms/Station-Year)."""
    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=300)

    plot_specs = [
        ("B1a_Rolling168h_Z", "B1a: Rolling 168h Z-Score", "#7f7f7f", "--", 1.6),
        ("B2a_RainMasked_Holdoff6h", "B2a: Rain-Masked Z (H=6h)", "#a6761d", "--", 1.6),
        ("B3b_WeatherRegression_SplineGAM", "B3b: Weather Spline GAM Residual", "#e6ab02", "-.", 1.8),
        ("B4b_RadonProjection_Local168h", "B4b: Radon Projection Residual (Non-ML)", "#1b9e77", "-", 2.2),
        ("B4c_RadonProjection_Combined", "B4c: Radon Projection + Peak Channel Z (Non-ML)", "#66a61e", "-", 2.2),
        ("B5b_NMF_Background_Reconstruction", "B5b: NMF 2-Comp Background Subspace (No Plumes)", "#7570b3", "-.", 2.0),
        ("B6b_IsolationForest_Tier4_BgOnly", "B6b: Unsupervised Isolation Forest (Tier 4, Bg Only)", "#e7298a", ":", 2.0),
        ("B7a_LogisticRegression_Tier3", "B7a: Logistic Regression (Tier 3)", "#d95f02", "--", 1.8),
        ("ML_Tier2_Spectral", "Tier 2: Spectral GBDT", "#377eb8", "-.", 2.0),
        ("ML_Tier3_WeatherFusion", "Tier 3: Weather Fusion GBDT", "#ff7f00", "-", 2.4),
        ("ML_Tier4_PhysicsNuclear", "Tier 4: Physics-Motivated Nuclear GBDT", "#d62728", "-", 2.8),
    ]

    for det_id, label, color, ls, lw in plot_specs:
        sub = df_roc[df_roc["detector_id"] == det_id].sort_values("test_fa_per_yr")
        if sub.empty:
            continue
        x = np.clip(sub["test_fa_per_yr"].to_numpy(dtype=float), 0.08, 300.0)
        y = sub["test_detection_pct"].to_numpy(dtype=float)
        ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.92)

    ax.set_xscale("log")
    ax.set_xlim(0.08, 250.0)
    ax.set_ylim(0.0, 100.5)
    ax.set_xlabel("Clean-Background False Alarm Episodes per Station-Year (2023–2025 Test Split, Log Scale)", fontsize=11)
    ax.set_ylabel("Realized Synthetic Plume Detection Rate (%)", fontsize=11)
    ax.set_title("Detector Comparison: Classical, Weather-Regression, Subspace, Unsupervised & GBDT Tiers", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.8, framealpha=0.95)
    fig.tight_layout()

    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "baseline_matched_roc_curves.png", dpi=300)
    fig.savefig(out_dir / "rigorous_matched_roc_curves.png", dpi=300)
    plt.close(fig)
    print("Saved ROC figures to reports/figures/baseline_matched_roc_curves.png and rigorous_matched_roc_curves.png")


if __name__ == "__main__":
    run_baseline_evaluation()
