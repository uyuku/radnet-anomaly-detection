"""Unit tests for Injection Realism & Generator Artifact Tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.artifact_tests import (
    compute_hours_since_last_filter_drop,
    evaluate_counting_noise_artifact,
    evaluate_filter_age_leakage,
    evaluate_snr_stratification,
)
from src.detector_response import (
    DetectorGeometryParams,
    NUCLIDE_GAMMA_LINES,
    al_window_transmission,
    apply_counting_noise,
    compute_analytic_line_response,
    compute_independent_dose_response_vector,
    compute_independent_injected_dose_rate,
    compute_monte_carlo_line_response,
    compute_nuclide_template,
    disk_to_cylinder_solid_angle_fraction,
    klein_nishina_recoil_pdf,
    nai_linear_attenuation_cm1,
    sample_derived_spectrum,
)
from src.nuclear_physics import compute_multinuclide_injection_timeseries
from src.synthetic_injection import classify_snr_bin, sample_operational_spectrum

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"


def test_detector_response_physics_and_families() -> None:
    # Attenuation decreases monotonically with photon energy from 100 to 2000 keV
    mu_pe, mu_comp, mu_tot = nai_linear_attenuation_cm1(np.array([150.0, 662.0, 1332.0]))
    assert np.all(np.diff(mu_pe) < 0)
    assert np.all(np.diff(mu_tot) < 0)

    # Aluminum transmission increases with energy
    t_al = al_window_transmission(np.array([100.0, 662.0, 1500.0]), np.array([1.0, 1.0, 1.0]))
    assert np.all((t_al > 0.0) & (t_al < 1.0))
    assert np.all(np.diff(t_al) > 0)

    # Solid angle decreases with standoff distance
    om_near = disk_to_cylinder_solid_angle_fraction(1.0, 2.0, 2.54)
    om_far = disk_to_cylinder_solid_angle_fraction(3.0, 2.0, 2.54)
    assert 0.0 < om_far < om_near < 0.5

    # Klein-Nishina recoil PDF integrates to 1.0
    grid = np.linspace(1.0, 1000.0, 500)
    pdf = klein_nishina_recoil_pdf(661.657, grid)
    area = float(np.trapezoid(pdf, grid)) if hasattr(np, "trapezoid") else float(np.trapz(pdf, grid))
    assert area == pytest.approx(1.0, abs=0.02)

    # Family A (Analytic) vs Family B (Monte Carlo) templates for Cs-137 and Co-60
    params = DetectorGeometryParams()
    s9_a, s8_a, eff_a = compute_nuclide_template("Cs-137", family="A", params=params)
    s9_b, s8_b, eff_b = compute_nuclide_template("Cs-137", family="B", params=params, n_histories_per_line=4000, seed=42)
    assert s9_a.shape == (9,) and s8_a.shape == (8,)
    assert s9_b.shape == (9,) and s8_b.shape == (8,)
    assert np.sum(s8_a) == pytest.approx(1.0, abs=1e-6)
    assert np.sum(s8_b) == pytest.approx(1.0, abs=1e-6)
    assert eff_a > 0.0 and eff_b > 0.0
    # Cs-137 662 keV photopeak falls in R05 (index 3 of R02..R09)
    assert s8_a[3] > 0.30 and s8_b[3] > 0.25
    # Family A and Family B must differ in method/shares
    assert not np.allclose(s8_a, s8_b, atol=0.02)


def test_sample_derived_and_holdout_spectra() -> None:
    rng = np.random.RandomState(42)
    for scen in [
        "fission_pure_cs137",
        "fission_pure_i131",
        "fission_reactor_fukushima",
        "mixed_fission_activation",
        "activation_orphan_co60",
        "fission_pure_ru103",
        "fission_zr95_nb95",
        "fission_ba140_la140",
    ]:
        s_a, nuc_a = sample_derived_spectrum(rng, scen, family="A")
        s_b, nuc_b = sample_derived_spectrum(rng, scen, family="B")
        assert len(s_a) == 8 and len(s_b) == 8
        assert np.sum(s_a) == pytest.approx(1.0, abs=1e-6)
        assert np.sum(s_b) == pytest.approx(1.0, abs=1e-6)
        assert sum(nuc_a.values()) == pytest.approx(1.0, abs=1e-3)
        assert sum(nuc_b.values()) == pytest.approx(1.0, abs=1e-3)


def test_continuous_plume_filter_reset_and_counting_noise() -> None:
    rng = np.random.RandomState(42)
    w, _ = sample_operational_spectrum(rng, "fission_pure_cs137", family="A")
    gross_no_reset, ch_no_reset = compute_multinuclide_injection_timeseries(
        shape_family="linear_ramp",
        t_passage=10,
        t_retention=20,
        peak_cpm=1000.0,
        rise_param=3.0,
        scenario="fission_pure_cs137",
        spectrum_at_peak=w,
        f_cs137=1.0,
        filter_reset_steps=None,
    )
    gross_reset, ch_reset = compute_multinuclide_injection_timeseries(
        shape_family="linear_ramp",
        t_passage=10,
        t_retention=20,
        peak_cpm=1000.0,
        rise_param=3.0,
        scenario="fission_pure_cs137",
        spectrum_at_peak=w,
        f_cs137=1.0,
        filter_reset_steps=[15],
    )
    # During passage (steps 0..9), trajectories are identical; at step 15, filter reset drops trapped activity
    np.testing.assert_allclose(gross_no_reset[:10], gross_reset[:10], rtol=1e-5)
    assert gross_reset[15] < 0.25 * gross_no_reset[15]
    assert gross_reset[16] > gross_reset[15]  # Re-accumulates from ongoing dispersion tail

    # Counting noise adds realistic positive variance
    noisy_gross, noisy_ch = apply_counting_noise(ch_no_reset, rng=rng, include_overdispersion=True)
    assert noisy_ch.shape == ch_no_reset.shape
    assert np.var(noisy_gross - gross_no_reset) > 1.0


def test_independent_dose_coupling_non_circular() -> None:
    k_indep = compute_independent_dose_response_vector()
    assert k_indep.shape == (8,)
    assert np.all(k_indep > 0.0)
    # Dose per count increases monotonically with photon energy
    assert np.all(np.diff(k_indep) > 0.0)

    rng = np.random.RandomState(42)
    ch = np.ones((10, 8), dtype=float) * 50.0
    d = compute_independent_injected_dose_rate(ch, rng=rng)
    assert d.shape == (10,)
    assert np.all(d > 0.0)


def test_snr_bin_classification_and_filter_age_helper() -> None:
    assert classify_snr_bin(0.6) == "<1.0_sigma"
    assert classify_snr_bin(1.5) == "1.0_2.0_sigma"
    assert classify_snr_bin(2.5) == "2.0_3.0_sigma"
    assert classify_snr_bin(4.0) == "3.0_5.0_sigma"
    assert classify_snr_bin(6.2) == ">=5.0_sigma"


def test_benchmark_generated_csv_artifacts_exist_and_consistent() -> None:
    resp_df = pd.read_csv(DATA_DIR / "response_templates.csv")
    assert len(resp_df) == 42
    assert {"A_analytic_approximation", "B_monte_carlo_3d_transport"}.issubset(set(resp_df["family"]))

    hold_df = pd.read_csv(DATA_DIR / "benchmark_holdout_templates.csv")
    assert len(hold_df) == 38
    assert {
        "i_leave_one_scenario_out",
        "ii_unseen_test_radionuclide",
        "iii_test_time_gain_or_resolution_perturbation",
    } == set(hold_df["holdout_category"])

    pt_df = pd.read_csv(DATA_DIR / "plume_timing_comparison.csv")
    assert len(pt_df) == 5
    assert "filter_terminated_det_pct" in pt_df.columns
    assert "continuous_arbitrary_end_det_pct" in pt_df.columns

    fa_df = pd.read_csv(DATA_DIR / "artifact_filter_age_leakage.csv")
    assert len(fa_df) == 2
    old_auc = float(fa_df[fa_df["timing_variant"] == "legacy_filter_terminated"]["val_auc_filter_age_only"].iloc[0])
    new_auc = float(fa_df[fa_df["timing_variant"] == "realistic_continuous_arbitrary_end"]["val_auc_filter_age_only"].iloc[0])
    assert old_auc > 0.55
    assert new_auc <= 0.55

    cn_df = pd.read_csv(DATA_DIR / "artifact_counting_noise.csv")
    assert len(cn_df) == 8
    assert cn_df["new_verdict"].str.startswith("PASS").all()

    dc_df = pd.read_csv(DATA_DIR / "artifact_dose_circularity.csv")
    assert len(dc_df) == 3
    assert {"radon_anchored_k_dose_channel", "independent_icrp74_crystal_efficiency"} == set(dc_df["dose_generator_model"])

    snr_df = pd.read_csv(DATA_DIR / "artifact_snr_stratification.csv")
    assert len(snr_df) == 50
    assert "<1.0_sigma" in set(snr_df["snr_bin"])

    brc_df = pd.read_csv(DATA_DIR / "benchmark_realism_comparison.csv")
    assert len(brc_df) == 5
    assert (brc_df["new_det_pct_mean"] > 75.0).all()

