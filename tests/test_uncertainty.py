"""Unit tests for src/uncertainty.py (Plume-level and block uncertainty)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_clopper_pearson_ci_bounds():
    lo0, hi0 = clopper_pearson_ci(0, 100, alpha=0.05)
    assert lo0 == 0.0
    assert 3.0 < hi0 < 4.0

    lo_full, hi_full = clopper_pearson_ci(100, 100, alpha=0.05)
    assert 96.0 < lo_full < 97.0
    assert hi_full == 100.0

    lo_mid, hi_mid = clopper_pearson_ci(90, 100, alpha=0.05)
    assert lo_mid < 90.0 < hi_mid


def test_exact_poisson_rate_ci_including_zero():
    lo0, hi0 = exact_poisson_rate_ci(0, 12.1638, alpha=0.05)
    assert lo0 == 0.0
    assert hi0 == pytest.approx(3.688879 / 12.1638, rel=1e-4)

    lo10, hi10 = exact_poisson_rate_ci(10, 10.0, alpha=0.05)
    assert lo10 < 1.0 < hi10


def test_cluster_bootstrap_detection_and_block_bootstrap_fa():
    rng = np.random.default_rng(42)
    plume_df = pd.DataFrame({
        "plume_id": [f"p_{i}" for i in range(100)],
        "station_id": ["st1", "st2"] * 50,
        "environment": ["dry", "rain"] * 50,
        "detected": rng.binomial(1, 0.85, size=100),
    })
    res_d = cluster_bootstrap_detection(plume_df, n_boot=2000, seed=42)
    assert res_d["n_plumes"] == 100
    assert res_d["boot_ci_low_pct"] <= res_d["detection_pct_point"] <= res_d["boot_ci_high_pct"]
    assert res_d["clopper_pearson_low_pct"] <= res_d["detection_pct_point"] <= res_d["clopper_pearson_high_pct"]

    block_df = pd.DataFrame({
        "station_id": ["st1"] * 36 + ["st2"] * 36,
        "year_month": [f"2023-{m:02d}" for m in range(1, 13)] * 6,
        "false_alarm_episodes": rng.poisson(0.5, size=72),
        "exposure_station_years": np.full(72, 1.0 / 12.0),
    })
    res_b = block_bootstrap_fa_rate(block_df, n_boot=2000, seed=42)
    assert res_b["n_blocks"] == 72
    assert res_b["boot_ci_low_fa_per_yr"] <= res_b["fa_per_yr_point"] <= res_b["boot_ci_high_fa_per_yr"]
    assert res_b["poisson_ci_low_fa_per_yr"] <= res_b["fa_per_yr_point"] <= res_b["poisson_ci_high_fa_per_yr"]


def test_paired_detector_bootstrap_and_holm():
    plume_a = pd.DataFrame({
        "plume_id": [f"p_{i}" for i in range(60)],
        "station_id": ["st1", "st2", "st3"] * 20,
        "environment": ["dry", "rain"] * 30,
        "detected": [1] * 56 + [0] * 4,
    })
    plume_b = plume_a.copy()
    plume_b["detected"] = [1] * 44 + [0] * 16

    block_a = pd.DataFrame({
        "station_id": ["st1"] * 24,
        "year_month": [f"2023-{m:02d}" for m in range(1, 25)],
        "false_alarm_episodes": [0] * 20 + [1] * 4,
        "exposure_station_years": np.full(24, 1.0 / 12.0),
    })
    block_b = block_a.copy()
    block_b["false_alarm_episodes"] = [2] * 24

    paired = paired_detector_bootstrap(plume_a, plume_b, block_a, block_b, n_boot=2000, seed=42)
    assert paired["delta_det_pct_point"] == pytest.approx(20.0)
    assert paired["ci_excludes_zero_det"] is True
    assert paired["delta_fa_per_yr_point"] < 0.0
    assert paired["ci_excludes_zero_fa"] is True

    adj, rej = holm_adjust_pvalues([0.001, 0.02, 0.04, 0.20], alpha=0.05)
    assert adj[0] == pytest.approx(0.004)
    assert rej[0] is True
    assert rej[3] is False


def test_matched_operating_point_bootstrap_and_anova():
    rng = np.random.default_rng(42)
    grid = np.linspace(0.5, 0.99, 20)
    vp_a = (rng.uniform(0.6, 1.0, size=(40, 1)) >= grid[None, :]).astype(float)
    vb_a = (rng.uniform(0.0, 0.96, size=(36, 1)) >= grid[None, :]).astype(float)
    tp_a = (rng.uniform(0.6, 1.0, size=(50, 1)) >= grid[None, :]).astype(float)
    tb_a = (rng.uniform(0.0, 0.97, size=(36, 1)) >= grid[None, :]).astype(float)
    v_exp = np.full(36, 1.0 / 12.0)
    t_exp = np.full(36, 1.0 / 12.0)

    res = matched_operating_point_bootstrap(
        grid, vp_a, vb_a, v_exp, tp_a, tb_a, t_exp,
        grid, vp_a, vb_a, tp_a, tb_a,
        mode="detection_target", target_value=0.90, n_boot=500, seed=42
    )
    assert res["delta_det_mean_pct"] == pytest.approx(0.0)
    assert res["delta_fa_mean_per_yr"] == pytest.approx(0.0)

    runs = []
    for c in range(10):
        c_eff = (c - 4.5) * 2.0
        for s in range(3):
            s_eff = (s - 1.0) * 0.2
            runs.append({"catalog_seed": c, "model_seed": s, "det_pct": 90.0 + c_eff + s_eff})
    vc = compute_variance_components_anova(pd.DataFrame(runs), "det_pct")
    assert vc["pct_var_catalog"] > 95.0
    assert vc["sd_catalog"] > vc["sd_model_seed"]


def test_coverage_simulation_passes_nominal_band():
    cov_df = verify_interval_coverage_simulation(n_trials=200, n_boot=400, seed=42)
    assert not cov_df.empty
    assert (cov_df["empirical_coverage_pct"] >= 91.0).all()
    assert (cov_df["empirical_coverage_pct"] <= 99.0).all()


def test_uncertainty_committed_outputs_exist_and_valid():
    from pathlib import Path

    single_df = pd.read_csv("data/processed/uncertainty_single_detector_intervals.csv")
    assert len(single_df) >= 150
    assert (single_df["det_boot_ci_low_pct"] <= single_df["det_pct_point"]).all()
    assert (single_df["det_pct_point"] <= single_df["det_boot_ci_high_pct"]).all()

    paired_df = pd.read_csv("data/processed/uncertainty_paired_comparisons.csv")
    assert len(paired_df) == 7
    assert "holm_adj_pvalue_det" in paired_df.columns
    assert "holm_adj_pvalue_fa" in paired_df.columns

    wm_df = pd.read_csv("data/processed/uncertainty_week_vs_month_block_sensitivity.csv")
    assert len(wm_df) == 6

    mop_df = pd.read_csv("data/processed/uncertainty_matched_operating_point_resampled.csv")
    assert len(mop_df) == 8

    var_df = pd.read_csv("data/processed/uncertainty_catalog_seed_variance.csv")
    assert len(var_df) == 16
    assert np.allclose(var_df["pct_var_catalog"] + var_df["pct_var_model_seed"], 100.0, atol=0.2)

    cov_file_df = pd.read_csv("data/processed/uncertainty_coverage_simulation.csv")
    assert cov_file_df["passes_nominal_band"].all()

    seed_tests = pd.read_csv("data/processed/rigorous_seed_statistical_tests.csv")
    assert "inference_scope_warning" in seed_tests.columns

    assert Path("docs/analysis_plan.md").exists()
    assert Path("reports/figures/eval_bootstrap_uncertainty.png").exists()
    assert Path("reports/figures/uncertainty_variance_components.png").exists()

