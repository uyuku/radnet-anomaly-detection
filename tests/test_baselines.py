"""Unit tests for Stronger Baselines & ML Feature Ablations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
from src.evaluate_baselines import (
    select_threshold_for_detection_target,
    select_threshold_for_fa_budget,
)
from src.feature_engineering import TIER_3_FEATURES, TIER_4_NUCLEAR_FEATURES, compute_features_for_series

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"


def _make_synthetic_station_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2019-01-01", periods=n, freq="h")
    rain = np.where(rng.uniform(size=n) < 0.1, rng.uniform(1.0, 10.0, size=n), 0.0)
    gross = 3500.0 + 80.0 * rain + rng.normal(0.0, 60.0, size=n)
    dose = 75.0 + 2.0 * rain + rng.normal(0.0, 2.0, size=n)

    df = pd.DataFrame({
        "dt": dt,
        "has_radnet_obs": True,
        "rad_complete_channels": True,
        "gross_cpm": gross,
        "dose_rate_nsvh": dose,
        "inj_gross_cpm": gross.copy(),
        "inj_dose_rate_nsvh": dose.copy(),
        "precip_1h_mm": rain,
        "temp_c": rng.normal(18.0, 4.0, size=n),
        "dewpoint_c": rng.normal(11.0, 3.0, size=n),
        "pressure_hpa": rng.normal(1014.0, 5.0, size=n),
        "rel_humidity_pct": rng.uniform(45.0, 85.0, size=n),
        "label": np.where(rain > 0.0, "radon_washout", "normal"),
        "injection_active": False,
    })
    shares = np.array([0.32, 0.27, 0.18, 0.10, 0.06, 0.04, 0.02, 0.01])
    for idx, ch in enumerate(range(2, 10)):
        c_val = np.maximum(1.0, gross * shares[idx] + rng.normal(0.0, 8.0, size=n))
        df[f"cpm_r0{ch}"] = c_val
        df[f"inj_cpm_r0{ch}"] = c_val.copy()

    # Add an injected Cs-137 spike in R05 at indices 250..270
    df.loc[250:270, "inj_cpm_r05"] += 800.0
    df.loc[250:270, "inj_gross_cpm"] += 800.0
    df.loc[250:270, "inj_dose_rate_nsvh"] += 15.0
    df.loc[250:270, "label"] = "fission_product"
    df.loc[250:270, "injection_active"] = True

    feats_cln = compute_features_for_series(df, "al_birmingham", use_injected=False)
    feats_inj = compute_features_for_series(df, "al_birmingham", use_injected=True)
    for col in ["gross_cpm", "dose_rate_nsvh", "inj_gross_cpm", "inj_dose_rate_nsvh"] + [
        f"cpm_r0{k}" for k in range(2, 10)
    ] + [f"inj_cpm_r0{k}" for k in range(2, 10)]:
        feats_cln[col] = df[col].to_numpy(dtype=float)
        feats_inj[col] = df[col].to_numpy(dtype=float)
    return feats_cln, feats_inj


def test_all_baseline_detectors_b1_through_b8_score_higher_on_plume() -> None:
    feats_cln, feats_inj = _make_synthetic_station_df(n=300, seed=42)
    fit_cln = feats_cln.iloc[:200].reset_index(drop=True)
    plume_slice = slice(255, 270)

    # B1 & B2 & B8
    stats = fit_b1_b2_b8_station_params(fit_cln)
    s_b1a_c = score_b1a_rolling168h_z(feats_cln)
    s_b1a_i = score_b1a_rolling168h_z(feats_inj)
    assert np.mean(s_b1a_i[plume_slice]) > np.mean(s_b1a_c[plume_slice]) + 2.0

    s_b1b_c = score_b1b_global_dry_z(feats_cln, stats, use_injected=False)
    s_b1b_i = score_b1b_global_dry_z(feats_inj, stats, use_injected=True)
    assert np.mean(s_b1b_i[plume_slice]) > np.mean(s_b1b_c[plume_slice]) + 2.0

    s_b2_c = score_b2_rain_masked_z(feats_cln, stats, holdoff_hours=6.0, use_injected=False)
    assert np.all(np.isfinite(s_b2_c))

    s_b8_c = score_b8_operational_screening_approx(feats_cln, stats, use_injected=False)
    s_b8_i = score_b8_operational_screening_approx(feats_inj, stats, use_injected=True)
    assert np.mean(s_b8_i[plume_slice]) > np.mean(s_b8_c[plume_slice]) + 1.5
    assert "EPA RadNet" in EPA_RADNET_ALERT_DOCUMENTATION_NOTE

    # B3: Weather regression (linear & spline GAM)
    for var in ["linear", "spline_gam"]:
        m_b3 = fit_b3_weather_regression(fit_cln, variant=var)
        s_c = score_b3_weather_regression(feats_cln, m_b3, use_injected=False)
        s_i = score_b3_weather_regression(feats_inj, m_b3, use_injected=True)
        assert np.mean(s_i[plume_slice]) > np.mean(s_c[plume_slice]) + 2.0

    # B4: Radon-direction projection (B4a and B4b are normalized in [0, 1]; B4c combines with channel z-score)
    assert np.mean(score_b4a_radon_projection_global(feats_inj)[plume_slice]) > np.mean(score_b4a_radon_projection_global(feats_cln)[plume_slice]) + 0.05
    assert np.mean(score_b4b_radon_projection_local168h(feats_inj)[plume_slice]) > np.mean(score_b4b_radon_projection_local168h(feats_cln)[plume_slice]) + 0.05
    assert np.mean(score_b4c_radon_projection_combined(feats_inj)[plume_slice]) > np.mean(score_b4c_radon_projection_combined(feats_cln)[plume_slice]) + 2.0

    # B5: Background PCA & NMF subspace models
    m_pca = fit_b5a_pca_subspace(fit_cln, n_components=3)
    assert np.mean(score_b5a_pca_subspace(feats_inj, m_pca, use_injected=True)[plume_slice]) > np.mean(score_b5a_pca_subspace(feats_cln, m_pca, use_injected=False)[plume_slice]) + 1.0

    m_nmf = fit_b5b_nmf_subspace(fit_cln, n_components=2)
    assert np.mean(score_b5b_nmf_subspace(feats_inj, m_nmf, use_injected=True)[plume_slice]) > np.mean(score_b5b_nmf_subspace(feats_cln, m_nmf, use_injected=False)[plume_slice]) + 1.0

    # B6: Unsupervised ML on background only
    for method in ["iforest_tier3", "iforest_tier4", "ocsvm_tier4"]:
        m_b6 = fit_b6_unsupervised_background(fit_cln, method=method, seed=42)
        s6_c = score_b6_unsupervised(feats_cln, m_b6)
        s6_i = score_b6_unsupervised(feats_inj, m_b6)
        assert s6_c.shape == (300,) and s6_i.shape == (300,)
        assert np.mean(s6_i[plume_slice]) > np.mean(s6_c[plume_slice])

    # B7: Simple supervised models
    y = feats_inj["label"].map({"normal": 0, "radon_washout": 1, "fission_product": 2}).to_numpy(dtype=int)
    for method in ["logreg_tier3", "dtree_depth3_tier3"]:
        m_b7 = fit_b7_simple_supervised(feats_inj, y, method=method, feature_cols=TIER_3_FEATURES, seed=42)
        s7_c = score_b7_simple_supervised(feats_cln, m_b7)
        s7_i = score_b7_simple_supervised(feats_inj, m_b7)
        assert np.mean(s7_i[plume_slice]) > np.mean(s7_c[plume_slice]) + 0.2


def test_threshold_selection_helpers() -> None:
    grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    det = np.array([0.98, 0.95, 0.91, 0.82, 0.60])
    fa = np.array([80.0, 25.0, 8.0, 3.0, 0.5])

    tau_90, met_90, v_det_90, v_fa_90 = select_threshold_for_detection_target(grid, det, fa, 90.0)
    assert met_90 is True
    assert tau_90 == pytest.approx(0.5)
    assert v_det_90 == pytest.approx(91.0)
    assert v_fa_90 == pytest.approx(8.0)

    # Tie-breaking when multiple thresholds achieve minimum FA=0.0: picks first index achieving min FA
    det_tie = np.array([0.98, 0.96, 0.91, 0.82, 0.60])
    fa_tie = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
    tau_80_tie, met_80_tie, _, _ = select_threshold_for_detection_target(grid, det_tie, fa_tie, 80.0)
    assert met_80_tie is True
    assert tau_80_tie == pytest.approx(0.3)

    tau_99, met_99, _, _ = select_threshold_for_detection_target(grid, det, fa, 99.0)
    assert met_99 is False
    assert tau_99 == pytest.approx(0.1)

    tau_b10, bmet_10, v_det_b10, v_fa_b10 = select_threshold_for_fa_budget(grid, det, fa, 10.0)
    assert bmet_10 is True
    assert tau_b10 == pytest.approx(0.5)
    assert v_det_b10 == pytest.approx(91.0)
    assert v_fa_b10 == pytest.approx(8.0)


def test_baseline_generated_csv_files_and_schema() -> None:
    mdet = pd.read_csv(DATA_DIR / "baseline_matched_detection_table.csv")
    assert set(mdet["val_target_detection_pct"]) == {80.0, 90.0, 95.0}
    assert "B4b_RadonProjection_Local168h" in set(mdet["detector_id"])
    assert "B5b_NMF_Background_Reconstruction" in set(mdet["detector_id"])
    assert "B6b_IsolationForest_Tier4_BgOnly" in set(mdet["detector_id"])
    assert "B8_Operational_Screening_Approx_UNVERIFIED" in set(mdet["detector_id"])
    assert "ML_Tier4_PhysicsNuclear" in set(mdet["detector_id"])

    mfa = pd.read_csv(DATA_DIR / "baseline_matched_fa_table.csv")
    assert set(mfa["val_fa_budget_per_yr"]) == {5.0, 10.0, 50.0}

    abl = pd.read_csv(DATA_DIR / "ml_ablation_summary.csv")
    assert len(abl) >= 30

    pp = pd.read_csv(DATA_DIR / "baseline_per_plume.csv")
    assert {"detector_id", "split", "operating_point_type", "operating_target", "plume_id", "detected", "delay_hours"}.issubset(set(pp.columns))
    assert set(pp["split"]) == {"val", "test"}

    pm = pd.read_csv(DATA_DIR / "baseline_per_month.csv")
    assert {"detector_id", "split", "station_id", "year_month", "false_alarm_episodes", "observed_exposure_hours"}.issubset(set(pm.columns))
