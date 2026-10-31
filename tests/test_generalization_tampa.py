"""Unit tests for src/generalization_tampa.py (Generalization and Tampa diagnosis)."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.generalization_tampa import (
    apply_normalization_variant,
    audit_station_raw_data_quality,
    compute_clean_feature_stats,
    detect_filter_change_indices,
    evaluate_station_test_metrics_with_exact_cis,
    extract_loso_false_alarm_episodes_with_shap,
    select_local_calibration_threshold,
)


def test_detect_filter_change_indices_and_raw_audit():
    n = 300
    dts = pd.date_range("2023-01-01", periods=n, freq="h")
    gross = np.full(n, 2500.0)
    # Insert filter advance drop at hour 100
    gross[100:] = 2100.0
    # Insert stuck run at hours 150..155
    gross[150:156] = 2155.0

    df = pd.DataFrame({
        "utc_hour": dts.strftime("%Y-%m-%d %H:%M:%S"),
        "gross_cpm": gross,
        "precip_mm": np.zeros(n),
    })
    for k in range(2, 10):
        df[f"cpm_r0{k}"] = gross * (0.45 if k == 2 else 0.07)

    vm = np.ones(n, dtype=bool)
    f_idx = detect_filter_change_indices(gross, vm)
    assert 100 in f_idx

    summary, yr_df = audit_station_raw_data_quality(df, "fl_tampa")
    assert summary["valid_complete_channel_hours"] == n
    assert summary["detected_filter_drop_hours"] == 1
    assert summary["stuck_gross_cpm_hours_ge3"] >= 3
    assert len(yr_df) == 1


def test_extract_loso_false_alarm_episodes_with_shap():
    rng = np.random.default_rng(42)
    n = 120
    dts = pd.date_range("2023-06-01", periods=n, freq="h")
    f_cols = ["share_r02", "share_r05", "z_score_global_dry", "hours_since_rain", "precip_1h"]
    feats = pd.DataFrame({
        "share_r02": rng.normal(0.415, 0.005, size=n),
        "share_r05": rng.normal(0.052, 0.002, size=n),
        "z_score_global_dry": rng.normal(0.5, 0.5, size=n),
        "z_score_168h": rng.normal(0.0, 0.5, size=n),
        "hours_since_rain": np.full(n, 72.0),
        "precip_1h": np.zeros(n),
        "precip_6h": np.zeros(n),
        "diff_1h_rel": np.zeros(n),
        "local_radon_stripping_residual_l2": rng.uniform(0.1, 1.0, size=n),
    })
    # Make one rain episode at hour 20 and one dry drift episode at hour 60
    feats.loc[20, "precip_1h"] = 4.5
    feats.loc[20, "hours_since_rain"] = 0.0

    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    clf = lgb.LGBMClassifier(n_estimators=12, num_class=3, objective="multiclass", random_state=42, verbose=-1)
    clf.fit(feats[f_cols], y)

    scores = np.full(n, 0.1)
    scores[20:22] = 0.99
    scores[60:63] = 0.985

    df_cln = pd.DataFrame({
        "utc_hour": dts.strftime("%Y-%m-%d %H:%M:%S"),
        "gross_cpm": np.full(n, 2100.0),
    })
    ep_df = extract_loso_false_alarm_episodes_with_shap(
        df_cln,
        feats,
        np.ones(n, dtype=bool),
        scores,
        threshold_tau=0.95,
        clf=clf,
        feature_cols=f_cols,
        tier_name="Tier 4",
        station_id="fl_tampa",
        train_share_stats={"share_r02": (0.50, 0.015), "share_r05": (0.043, 0.002)},
    )
    assert len(ep_df) == 2
    assert ep_df.iloc[0]["episode_category"] == "rain-coincident"
    assert ep_df.iloc[1]["episode_category"] == "baseline or gain drift"
    assert ep_df.iloc[0]["top_contrib_feature_1"] in f_cols


def test_apply_normalization_variants_and_local_calibration():
    n = 400
    feats = pd.DataFrame({
        "share_r02": np.full(n, 0.415),
        "share_r02_mean_6h": np.full(n, 0.415),
        "share_r05": np.full(n, 0.052),
        "share_r05_mean_6h": np.full(n, 0.052),
        "share_high_energy": np.full(n, 0.025),
        "ratio_r05_r03": np.full(n, 0.18),
        "z_score_global_dry": np.full(n, 2.5),
        "z_score_168h": np.zeros(n),
        "radon_stripping_residual_l2": np.full(n, 3.0),
        "local_radon_stripping_residual_l2": np.full(n, 0.4),
        "stripped_excess_r05_z": np.full(n, 2.8),
        "local_stripped_r05_z": np.full(n, 0.2),
        "hours_since_rain": np.full(n, 48.0),
        "dry_excess_interaction": np.full(n, 9.5),
    })
    loc_df = apply_normalization_variant(feats, "local_168h_baseline")
    assert np.allclose(loc_df["share_r02"].iloc[200:], 0.0, atol=1e-6)
    assert np.allclose(loc_df["z_score_global_dry"], 0.0, atol=1e-6)
    assert np.allclose(loc_df["radon_stripping_residual_l2"], 0.4, atol=1e-6)

    st_stats = compute_clean_feature_stats(feats, np.ones(n, dtype=bool), ["share_r02", "share_r05"])
    tp_stats = {"share_r02": (0.50, 0.01), "share_r05": (0.043, 0.002)}
    std_df = apply_normalization_variant(feats, "station_standardized", st_stats, tp_stats)
    assert np.allclose(std_df["share_r02"], 0.50, atol=1e-4)

    rng = np.random.default_rng(42)
    loc_scores = rng.beta(1.0, 20.0, size=2000)
    tau_5 = select_local_calibration_threshold(loc_scores, np.ones(2000, dtype=bool), fa_budget_per_yr=5.0, fallback_tau=0.95)
    tau_0 = select_local_calibration_threshold(loc_scores, np.ones(2000, dtype=bool), fa_budget_per_yr=0.0, fallback_tau=0.95)
    assert tau_0 >= tau_5

    slices = [{"indices": np.arange(10, 20)}, {"indices": np.arange(50, 60)}]
    sc_inj = loc_scores.copy()
    sc_inj[12] = 0.9999
    metrics = evaluate_station_test_metrics_with_exact_cis(loc_scores, sc_inj, np.ones(2000, dtype=bool), slices, tau_0)
    assert metrics["false_alarm_episodes"] == 0
    assert metrics["n_detected"] == 1


def test_generalization_artifacts_exist_and_schema():
    required_csvs = [
        "data/processed/extended_station_coverage_audit.csv",
        "data/processed/loso_tampa_false_alarms.csv",
        "data/processed/loso_tampa_episode_category_summary.csv",
        "data/processed/tampa_channel_share_comparison.csv",
        "data/processed/tampa_gain_drift_hypothesis_test.csv",
        "data/processed/tampa_raw_data_quality_audit.csv",
        "data/processed/all_stations_yearly_baseline_drift.csv",
        "data/processed/loso_local_calibration_results.csv",
        "data/processed/loso_normalization_variants.csv",
        "data/processed/extended_loso_11station_summary.csv",
        "data/processed/extended_loco_climate_summary.csv",
        "data/processed/loso_high_fa_stations_diagnosis.csv",
        "data/processed/temporal_generalization_by_year.csv",
        "data/processed/temporal_generalization_by_station_year.csv",
        "data/processed/generalization_loso_per_plume.csv",
        "data/processed/generalization_loso_per_month.csv",
    ]
    for csv_path in required_csvs:
        p = Path(csv_path)
        assert p.exists(), f"Missing generalization output CSV: {csv_path}"
        df = pd.read_csv(p)
        assert not df.empty, f"Empty generalization output CSV: {csv_path}"

    # Verify all 11 pre-registered stations are present in 11-station LOSO
    df_11 = pd.read_csv("data/processed/extended_loso_11station_summary.csv")
    assert df_11["held_out_station"].nunique() == 11
    assert {"ZeroShot_Uncalibrated", "ZeroShot_Local168h_Normalized", "LocalCalibrated_12m_Clean"}.issubset(
        set(df_11["transfer_mode"].unique())
    )

    # Verify figures and documentation exist
    assert Path("docs/station_selection.md").exists()
    assert Path("docs/phase6_generalization_summary.md").exists()
    assert Path("reports/figures/tampa_loso_diagnostic_episodes.png").exists()
    assert Path("reports/figures/tampa_spectral_drift_and_calibration.png").exists()
