"""
Unit Tests for Nuclear Engineering & Physics-Informed Spectrometry.

Verifies:
1. Analytical Bateman 2-step (Te-132 -> I-132) and 3-step (Po-218 -> Pb-214 -> Bi-214)
   atom conservation and strict non-negativity across all t >= 0.
2. Channel-resolved energy-weighted dose response vector k_dose_channel:
   - Exact reproduction of empirical station scalar radon washout slope when applied to s_Rn.
   - Higher dose-to-gross ratio for high-energy Co-60 (R07) than low-energy I-131 (R03).
3. Multi-nuclide time-evolving filter spectra:
   - Dynamic evolution of channel shares during filter retention (t >= t_passage)
     for mixed/reactor releases as short-lived Te-132/I-132 and I-131 decay relative to Cs-137.
4. Exact orthogonality / zero stripping residual when a pure s_Rn vector is passed
   to the spectral stripping function.
5. Strictly causal behavior (perturbing t+1 never alters any Tier 4 feature at t).
6. Strict NaN propagation when weather inputs (precip_1h_mm) are NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FEATURE_SCHEMA_VERSION,
    NUCLEAR_PHYSICS_FEATURES,
    TIER_3_FEATURES,
    TIER_4_NUCLEAR_FEATURES,
    compute_features_for_series,
)
from src.nuclear_physics import (
    DEFAULT_RADON_BASIS,
    LAMBDA_BI214,
    LAMBDA_I132,
    LAMBDA_PB214,
    LAMBDA_PO218,
    LAMBDA_TE132,
    TEMPLATE_CO60,
    TEMPLATE_CS137,
    TEMPLATE_I131,
    bateman_te132_i132_activity_kernel,
    bateman_three_step_atoms,
    bateman_two_step_atoms,
    compute_channel_dose_response_vector,
    compute_injected_dose_rate,
    compute_multinuclide_injection_timeseries,
    compute_spectral_stripping,
    solve_bateman_washout_ode,
)


def test_bateman_two_step_and_three_step_conservation_and_non_negativity():
    """Analytical Bateman solutions must conserve total atoms and remain strictly non-negative."""
    dt = np.linspace(0.0, 240.0, 500)
    n1_0, n2_0 = 1250.0, 320.0
    n1, n2, n3 = bateman_two_step_atoms(n1_0, n2_0, LAMBDA_TE132, LAMBDA_I132, dt)

    assert np.all(n1 >= 0.0)
    assert np.all(n2 >= 0.0)
    assert np.all(n3 >= 0.0)
    np.testing.assert_allclose(n1 + n2 + n3, n1_0 + n2_0, rtol=1e-12, atol=1e-10)

    # 3-step radon progeny chain (Po-218 -> Pb-214 -> Bi-214 -> Pb-210)
    dt_rn = np.linspace(0.0, 12.0, 250)
    m1_0, m2_0, m3_0 = 500.0, 200.0, 100.0
    m1, m2, m3, m4 = bateman_three_step_atoms(
        m1_0, m2_0, m3_0, LAMBDA_PO218, LAMBDA_PB214, LAMBDA_BI214, dt_rn
    )
    assert np.all(m1 >= 0.0)
    assert np.all(m2 >= 0.0)
    assert np.all(m3 >= 0.0)
    assert np.all(m4 >= 0.0)
    np.testing.assert_allclose(m1 + m2 + m3 + m4, m1_0 + m2_0 + m3_0, rtol=1e-12, atol=1e-10)

    # Te-132 -> I-132 activity kernel non-negativity and transient equilibrium behavior
    a_te, a_i = bateman_te132_i132_activity_kernel(dt, initial_i132_to_te132_activity_ratio=0.0)
    assert np.all(a_te >= 0.0)
    assert np.all(a_i >= 0.0)
    assert a_i[0] == pytest.approx(0.0, abs=1e-12)
    # In transient equilibrium (t >> T_1/2(I-132)), A_I(t) / A_Te(t) -> lam_I / (lam_I - lam_Te) > 1
    eq_ratio = LAMBDA_I132 / (LAMBDA_I132 - LAMBDA_TE132)
    np.testing.assert_allclose(a_i[-1] / a_te[-1], eq_ratio, rtol=1e-3)


def test_channel_resolved_dose_coupling_anchors_radon_and_orders_energies():
    """k_dose_channel must reproduce scalar radon slope on s_Rn and weight Co-60 > I-131."""
    slope_rn = 0.01389
    k_vec = compute_channel_dose_response_vector(slope_rn, DEFAULT_RADON_BASIS)
    assert k_vec.shape == (8,)
    assert np.all(k_vec > 0.0)
    # Monotonically increasing with channel centroid energy (R02..R09)
    assert np.all(np.diff(k_vec) > 0.0)

    # 1000 CPM of pure radon washout spectrum must yield exactly 1000 * slope_rn nSv/h
    delta_rn = 1000.0 * DEFAULT_RADON_BASIS
    dose_rn = compute_injected_dose_rate(delta_rn, slope_rn, DEFAULT_RADON_BASIS)
    assert float(dose_rn) == pytest.approx(1000.0 * slope_rn, rel=1e-12)

    # High-energy Co-60 (R07) must yield higher dose per gross CPM than low-energy I-131 (R03)
    dose_i131 = float(compute_injected_dose_rate(1000.0 * TEMPLATE_I131, slope_rn, DEFAULT_RADON_BASIS))
    dose_co60 = float(compute_injected_dose_rate(1000.0 * TEMPLATE_CO60, slope_rn, DEFAULT_RADON_BASIS))
    assert dose_co60 > 1.5 * dose_i131


def test_multinuclide_time_evolving_spectra_retention_kinetics():
    """Reactor release spectrum must evolve during retention as Te-132/I-132 & I-131 decay faster than Cs-137."""
    spec_peak = np.array([0.20, 0.30, 0.10, 0.30, 0.05, 0.04, 0.008, 0.002], dtype=float)
    gross_prof, ch_prof = compute_multinuclide_injection_timeseries(
        shape_family="sigmoidal",
        t_passage=10,
        t_retention=60,
        peak_cpm=1000.0,
        rise_param=1.2,
        scenario="fission_reactor_fukushima",
        spectrum_at_peak=spec_peak,
        f_cs137=0.40,
        f_i131=0.30,
        f_cs134=0.30,
    )
    assert gross_prof.shape == (70,)
    assert ch_prof.shape == (70, 8)
    assert np.all(ch_prof >= 0.0)
    np.testing.assert_allclose(np.sum(ch_prof, axis=1), gross_prof, rtol=1e-12)

    # At peak passage (t = t_passage - 1 = 9), channel shares match spec_peak and sum to peak_cpm
    assert gross_prof[9] == pytest.approx(1000.0, rel=1e-6)
    np.testing.assert_allclose(ch_prof[9] / gross_prof[9], spec_peak, rtol=1e-6)

    # During retention (t = 9 -> t = 69), short-lived R03 (I-131 + Te-132) share decreases
    # relative to long-lived R05 (Cs-137 + Cs-134)
    shares_t9 = ch_prof[9] / gross_prof[9]
    shares_t69 = ch_prof[69] / gross_prof[69]
    ratio_r03_r05_peak = shares_t9[1] / shares_t9[3]
    ratio_r03_r05_end = shares_t69[1] / shares_t69[3]
    assert ratio_r03_r05_end < ratio_r03_r05_peak


def test_spectral_stripping_exact_orthogonality_on_pure_radon_basis():
    """Passing a pure s_Rn excess vector to compute_spectral_stripping must produce zero stripping residual."""
    amplitudes = np.array([0.0, 50.0, 250.0, 1200.0, 5000.0], dtype=float)
    pure_rn_matrix = np.outer(amplitudes, DEFAULT_RADON_BASIS)

    res = compute_spectral_stripping(pure_rn_matrix, DEFAULT_RADON_BASIS)
    np.testing.assert_allclose(res["alpha_rn"], amplitudes, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res["residual_vec"], 0.0, atol=1e-12)
    np.testing.assert_allclose(res["radon_stripping_residual_l2"], 0.0, atol=1e-14)
    np.testing.assert_allclose(res["stripped_excess_r03_z"], 0.0, atol=1e-12)
    np.testing.assert_allclose(res["stripped_excess_r05_z"], 0.0, atol=1e-12)
    np.testing.assert_allclose(res["stripped_excess_r07_z"], 0.0, atol=1e-12)

    # Non-radon fission spectra (I-131, Cs-137, Co-60) must yield positive stripping residuals
    fission_mat = 1000.0 * np.vstack([TEMPLATE_I131, TEMPLATE_CS137, TEMPLATE_CO60])
    res_fiss = compute_spectral_stripping(fission_mat, DEFAULT_RADON_BASIS)
    assert np.all(res_fiss["radon_stripping_residual_l2"] > 0.15)
    assert res_fiss["stripped_excess_r03_z"][0] > 2.0  # I-131 spikes R03
    assert res_fiss["stripped_excess_r05_z"][1] > 5.0  # Cs-137 spikes R05
    assert res_fiss["stripped_excess_r07_z"][2] > 5.0  # Co-60 spikes R07


def test_tier4_strict_causality_no_future_lookahead(hourly_frame):
    """Perturbing future observations at t > 200 must never alter any Tier 4 feature at t <= 200."""
    assert FEATURE_SCHEMA_VERSION == 3
    assert len(TIER_3_FEATURES) == 48
    assert len(TIER_4_NUCLEAR_FEATURES) == len(TIER_3_FEATURES) + len(NUCLEAR_PHYSICS_FEATURES)

    base_df = hourly_frame(320)
    # Add realistic rain event before t=200 to exercise Bateman washout and post-rain persistence
    base_df.loc[180:183, "precip_1h_mm"] = 4.5

    pert_df = base_df.copy()
    pert_df.loc[201:, "gross_cpm"] += 50000.0
    pert_df.loc[201:, "dose_rate_nsvh"] += 1500.0
    pert_df.loc[201:, "precip_1h_mm"] = 85.0
    for c in range(2, 10):
        pert_df.loc[201:, f"cpm_r0{c}"] *= 20.0
    for col in ["gross_cpm", "dose_rate_nsvh"] + [f"cpm_r0{c}" for c in range(2, 10)]:
        pert_df[f"inj_{col}"] = pert_df[col]

    f_base = compute_features_for_series(base_df, "al_birmingham", use_injected=True)
    f_pert = compute_features_for_series(pert_df, "al_birmingham", use_injected=True)

    for col in TIER_4_NUCLEAR_FEATURES:
        a = f_base.loc[:200, col].to_numpy(dtype=float)
        b = f_pert.loc[:200, col].to_numpy(dtype=float)
        assert np.allclose(a, b, atol=0.0, rtol=0.0, equal_nan=True), (
            f"Causality violation (future leakage) in feature '{col}' at or before t=200!"
        )


def test_bateman_ode_and_tier4_strict_nan_propagation(hourly_frame):
    """Missing precipitation (NaN) must strictly propagate NaN to Bateman washout and persistence features."""
    frame = hourly_frame(320)
    frame.loc[185, "precip_1h_mm"] = 3.0
    frame.loc[200, "precip_1h_mm"] = np.nan

    ode_out = solve_bateman_washout_ode(frame["precip_1h_mm"], beta=0.65, memory_hours=6)
    # At t=200..205 (6-hour causal horizon), Bateman ODE states must be strictly NaN
    assert np.isnan(ode_out["bateman_expected_index"][200:206]).all()
    # Before t=200 and after t=206, Bateman ODE state is finite
    assert np.isfinite(ode_out["bateman_expected_index"][199])
    assert np.isfinite(ode_out["bateman_expected_index"][206])

    feats = compute_features_for_series(frame, "al_birmingham", use_injected=False)
    for col in ["precip_1h_mm", "hours_since_rain", "bateman_washout_residual_ratio", "post_rain_persistence_anomaly"]:
        assert pd.isna(feats.loc[200, col]), f"Expected NaN at t=200 for '{col}'"
    assert feats.loc[200:205, "bateman_washout_residual_ratio"].isna().all()
    assert pd.notna(feats.loc[206, "bateman_washout_residual_ratio"])
