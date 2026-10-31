"""
Nuclear Engineering & Physics-Motivated Spectrometry Core Module.

Provides strictly causal, physically grounded models for:
1. 8-channel NaI(Tl) energy-weighted ambient dose equivalent response vector
   k_dose_channel (nSv/h per CPM across R02..R09), grounded in ICRP-74 photon
   conversion coefficients h*(10, E_gamma) and 2"x2" NaI(Tl) intrinsic detection
   efficiency, normalized to reproduce empirical station radon washout slopes.
2. Analytical Bateman equations for:
   - Multi-nuclide particulate filter deposition and decay, including the
     short-lived reactor tellurium-iodine pair Te-132 -> I-132 (T_1/2 = 76.9 h -> 2.295 h)
     alongside I-131 (8.025 d), Cs-134 (2.065 y), Co-60 (5.27 y), and Cs-137 (30.17 y).
   - Filter-trapped natural radon progeny washout kinetics:
     Po-218 (3.10 min) -> Pb-214 (26.8 min) -> Bi-214 (19.9 min) driven by
     causal precipitation scavenging W(t) ~ precip_1h_mm(t)^beta.
3. Causal Spectral Stripping (Unmixing) of the 8-channel radon progeny basis
   spectrum s_Rn from observed NaI(Tl) channel excesses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. NaI(Tl) Channel Energy Windows & ICRP-74 Dose Coupling Physics
# ---------------------------------------------------------------------------
CHANNELS = [f"cpm_r0{i}" for i in range(2, 10)]

# Centroid energies (MeV) for R02 (101-200 keV) through R09 (1801-2200 keV)
CHANNEL_CENTROID_MEV = np.array([0.15, 0.30, 0.50, 0.70, 0.90, 1.20, 1.60, 2.00], dtype=float)

# ICRP Publication 74 ambient dose equivalent conversion h*(10, E_gamma) in pSv * cm^2
ICRP74_H10_PSV_CM2 = np.array([0.89, 1.80, 2.93, 3.92, 4.79, 5.85, 7.08, 8.15], dtype=float)

# 2"x2" NaI(Tl) crystal intrinsic interaction efficiency at channel centroids
NAI_INTRINSIC_EFFICIENCY = np.array([0.92, 0.68, 0.48, 0.38, 0.32, 0.26, 0.21, 0.18], dtype=float)

# Relative dose per detected count in channel k: proportional to h*(10, E_k) / epsilon_int(E_k)
RAW_CHANNEL_DOSE_WEIGHTS = ICRP74_H10_PSV_CM2 / NAI_INTRINSIC_EFFICIENCY

# Default pooled train-only radon washout basis spectrum s_Rn across R02..R09 (sums to 1.0)
DEFAULT_RADON_BASIS = np.array(
    [0.4227, 0.3375, 0.0916, 0.0605, 0.0249, 0.0388, 0.0173, 0.0067],
    dtype=float,
)
DEFAULT_RADON_BASIS = DEFAULT_RADON_BASIS / np.sum(DEFAULT_RADON_BASIS)

# Characteristic unit spectral templates for NNLS / spectral unmixing (R02..R09)
TEMPLATE_I131 = np.array([0.23, 0.62, 0.06, 0.06, 0.015, 0.008, 0.005, 0.002], dtype=float)
TEMPLATE_I131 /= np.sum(TEMPLATE_I131)

TEMPLATE_CS137 = np.array([0.21, 0.16, 0.14, 0.46, 0.015, 0.008, 0.005, 0.002], dtype=float)
TEMPLATE_CS137 /= np.sum(TEMPLATE_CS137)

TEMPLATE_CO60 = np.array([0.16, 0.15, 0.15, 0.14, 0.10, 0.27, 0.025, 0.005], dtype=float)
TEMPLATE_CO60 /= np.sum(TEMPLATE_CO60)


def compute_channel_dose_response_vector(
    scalar_radon_slope: float,
    radon_basis_spectrum: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """
    Computes the 8-channel energy-weighted dose response vector k_dose_channel
    (nSv/h per CPM for R02..R09) anchored so that an excess spectrum matching
    the radon washout basis s_Rn reproduces scalar_radon_slope:
        sum_{k=2..9} k_dose_channel[k] * s_Rn[k] == scalar_radon_slope.

    Parameters
    ----------
    scalar_radon_slope : float
        Empirical station or event dose-to-gross slope (nSv/h per CPM) for radon washout.
    radon_basis_spectrum : array-like of shape (8,), optional
        Normalized 8-channel radon washout basis spectrum s_Rn. Defaults to DEFAULT_RADON_BASIS.

    Returns
    -------
    np.ndarray of shape (8,)
        Channel-resolved dose conversion factors (nSv/h per CPM) for R02..R09.
    """
    if scalar_radon_slope < 0:
        raise ValueError("scalar_radon_slope must be non-negative.")
    s_rn = (
        np.asarray(DEFAULT_RADON_BASIS, dtype=float)
        if radon_basis_spectrum is None
        else np.asarray(radon_basis_spectrum, dtype=float)
    )
    if s_rn.shape != (8,):
        raise ValueError(f"radon_basis_spectrum must have shape (8,), got {s_rn.shape}.")
    s_rn = np.maximum(0.0, s_rn)
    s_sum = float(np.sum(s_rn))
    if s_sum <= 0.0:
        s_rn = DEFAULT_RADON_BASIS.copy()
    else:
        s_rn = s_rn / s_sum

    denom = float(np.dot(s_rn, RAW_CHANNEL_DOSE_WEIGHTS))
    return float(scalar_radon_slope) * (RAW_CHANNEL_DOSE_WEIGHTS / denom)


def compute_injected_dose_rate(
    delta_channels: np.ndarray,
    scalar_radon_slope: float,
    radon_basis_spectrum: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """
    Computes channel-resolved injected ambient dose equivalent rate (nSv/h)
    from 8-channel injected count excesses Delta C_k(t).

    Parameters
    ----------
    delta_channels : np.ndarray
        Array of shape (..., 8) containing channel count excesses for R02..R09.
    scalar_radon_slope : float
        Station/event empirical radon washout coupling slope (nSv/h per CPM).
    radon_basis_spectrum : array-like of shape (8,), optional
        Station radon washout basis spectrum s_Rn.

    Returns
    -------
    np.ndarray
        Injected dose rate excess in nSv/h.
    """
    k_vec = compute_channel_dose_response_vector(scalar_radon_slope, radon_basis_spectrum)
    arr = np.asarray(delta_channels, dtype=float)
    return np.tensordot(arr, k_vec, axes=([-1], [0]))


# ---------------------------------------------------------------------------
# 2. Radioactive Decay Constants & Analytical Bateman Kinetics
# ---------------------------------------------------------------------------
# Anthropogenic nuclide decay constants (h^-1)
LAMBDA_I131 = np.log(2.0) / (8.025 * 24.0)      # T_1/2 = 8.025 d  (192.6 h)
LAMBDA_CS137 = np.log(2.0) / (30.17 * 8766.0)   # T_1/2 = 30.17 y
LAMBDA_CS134 = np.log(2.0) / (2.065 * 8766.0)   # T_1/2 = 2.065 y
LAMBDA_CO60 = np.log(2.0) / (5.27 * 8766.0)     # T_1/2 = 5.27 y
LAMBDA_TE132 = np.log(2.0) / 76.9               # T_1/2 = 76.9 h
LAMBDA_I132 = np.log(2.0) / 2.295               # T_1/2 = 2.295 h
LAMBDA_RU103 = np.log(2.0) / (39.247 * 24.0)    # T_1/2 = 39.247 d
LAMBDA_ZR95 = np.log(2.0) / (64.032 * 24.0)     # T_1/2 = 64.032 d
LAMBDA_NB95 = np.log(2.0) / (34.991 * 24.0)     # T_1/2 = 34.991 d
LAMBDA_BA140 = np.log(2.0) / (12.752 * 24.0)    # T_1/2 = 12.752 d
LAMBDA_LA140 = np.log(2.0) / (1.678 * 24.0)     # T_1/2 = 40.27 h

# Natural Rn-222 short-lived particulate progeny decay constants (h^-1)
LAMBDA_PO218 = np.log(2.0) / (3.10 / 60.0)      # T_1/2 = 3.10 min (0.05167 h)
LAMBDA_PB214 = np.log(2.0) / (26.8 / 60.0)      # T_1/2 = 26.8 min (0.44667 h)
LAMBDA_BI214 = np.log(2.0) / (19.9 / 60.0)      # T_1/2 = 19.9 min (0.33167 h)


def bateman_two_step_atoms(
    n1_0: float,
    n2_0: float,
    lam1: float,
    lam2: float,
    dt_hours: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact analytical Bateman solution for a 2-step decay chain:
        Parent (1) --lam1--> Daughter (2) --lam2--> Granddaughter (3, stable)

    Satisfies strict non-negativity (N_i(t) >= 0) and exact total atom conservation:
        N_1(t) + N_2(t) + N_3(t) == n1_0 + n2_0  for all t >= 0.

    Parameters
    ----------
    n1_0 : float
        Initial parent atoms at t = 0 (must be >= 0).
    n2_0 : float
        Initial daughter atoms at t = 0 (must be >= 0).
    lam1, lam2 : float
        Decay constants (h^-1, must be > 0 and distinct).
    dt_hours : float or array-like
        Elapsed time(s) t >= 0 in hours.

    Returns
    -------
    tuple of (n1, n2, n3) : np.ndarray
        Parent, daughter, and stable granddaughter atom counts at dt_hours.
    """
    t = np.maximum(0.0, np.asarray(dt_hours, dtype=float))
    e1 = np.exp(-lam1 * t)
    e2 = np.exp(-lam2 * t)

    n1 = float(n1_0) * e1
    if abs(lam2 - lam1) < 1e-12:
        n2 = float(n1_0) * lam1 * t * e1 + float(n2_0) * e2
    else:
        n2 = float(n1_0) * (lam1 / (lam2 - lam1)) * (e1 - e2) + float(n2_0) * e2
    n2 = np.maximum(0.0, n2)
    n3 = np.maximum(0.0, (float(n1_0) + float(n2_0)) - (n1 + n2))
    return n1, n2, n3


def bateman_three_step_atoms(
    n1_0: float,
    n2_0: float,
    n3_0: float,
    lam1: float,
    lam2: float,
    lam3: float,
    dt_hours: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact analytical Bateman solution for a 3-step decay chain:
        Po-218 (1) --lam1--> Pb-214 (2) --lam2--> Bi-214 (3) --lam3--> Pb-210 (4, stable on hourly scale)

    Satisfies strict non-negativity and exact conservation:
        N_1(t) + N_2(t) + N_3(t) + N_4(t) == n1_0 + n2_0 + n3_0.
    """
    t = np.maximum(0.0, np.asarray(dt_hours, dtype=float))
    e1 = np.exp(-lam1 * t)
    e2 = np.exp(-lam2 * t)
    e3 = np.exp(-lam3 * t)

    n1 = float(n1_0) * e1
    n2 = float(n1_0) * (lam1 / (lam2 - lam1)) * (e1 - e2) + float(n2_0) * e2

    # Bateman 3rd member from N1_0 + 2nd member from N2_0 + decay of N3_0
    c1 = lam1 * lam2 / ((lam2 - lam1) * (lam3 - lam1))
    c2 = lam1 * lam2 / ((lam1 - lam2) * (lam3 - lam2))
    c3 = lam1 * lam2 / ((lam1 - lam3) * (lam2 - lam3))
    n3_from_1 = float(n1_0) * (c1 * e1 + c2 * e2 + c3 * e3)
    n3_from_2 = float(n2_0) * (lam2 / (lam3 - lam2)) * (e2 - e3)
    n3 = np.maximum(0.0, n3_from_1 + n3_from_2 + float(n3_0) * e3)
    n2 = np.maximum(0.0, n2)
    n1 = np.maximum(0.0, n1)
    n4 = np.maximum(0.0, (float(n1_0) + float(n2_0) + float(n3_0)) - (n1 + n2 + n3))
    return n1, n2, n3, n4


def bateman_te132_i132_activity_kernel(
    dt_hours: np.ndarray | float,
    initial_i132_to_te132_activity_ratio: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the relative parent Te-132 and progeny I-132 activities A_Te(t), A_I(t)
    at elapsed time dt_hours >= 0 after filter deposition of unit Te-132 activity.

    Uses the analytical Bateman equation for Te-132 (T_1/2 = 76.9 h) -> I-132 (T_1/2 = 2.295 h):
        A_Te(t) = A_Te(0) * exp(-lam_Te * t)
        A_I(t)  = A_Te(0) * [lam_I / (lam_I - lam_Te)] * (exp(-lam_Te * t) - exp(-lam_I * t))
                  + A_I(0) * exp(-lam_I * t)
    """
    t = np.maximum(0.0, np.asarray(dt_hours, dtype=float))
    e_te = np.exp(-LAMBDA_TE132 * t)
    e_i = np.exp(-LAMBDA_I132 * t)

    a_te = e_te
    branch = LAMBDA_I132 / (LAMBDA_I132 - LAMBDA_TE132)
    a_i = branch * (e_te - e_i) + float(initial_i132_to_te132_activity_ratio) * e_i
    return np.maximum(0.0, a_te), np.maximum(0.0, a_i)


def _single_nuclide_filter_accumulation(
    c_deposition: np.ndarray,
    t_retention: int,
    lam: float,
    filter_reset_steps: list[int] | None = None,
) -> np.ndarray:
    """
    Computes filter-trapped activity trajectory over D = t_passage + t_retention hours
    for a single decay constant lam (h^-1), normalized to 1.0 at peak passage.
    If `filter_reset_steps` is provided (continuous timing variant),
    the deposited filter activity resets at each real filter advance step r_step and
    continues accumulating from ongoing airborne concentration (with an atmospheric
    dispersion tail after t_passage).
    """
    t_passage = len(c_deposition)
    d = t_passage + t_retention
    # Construct full airborne concentration curve c_full(u) over 0..d-1
    c_full = np.zeros(d, dtype=float)
    c_full[:t_passage] = c_deposition
    if filter_reset_steps:
        # In the continuous plume variant, airborne concentration has a dispersion tail
        # so that if a filter change occurs mid-plume, fresh activity still deposits
        tail_idx = np.arange(t_retention, dtype=float) + 1.0
        c_full[t_passage:] = c_deposition[-1] * np.exp(-tail_idx / max(3.0, 0.35 * t_passage))

    resets = sorted([int(r) for r in (filter_reset_steps or []) if 0 < int(r) < d])

    # First compute unreset peak at t_passage - 1 for consistent peak_cpm scaling
    u_peak = np.arange(t_passage)
    peak_unreset = float(np.sum(c_deposition * np.exp(-lam * ((t_passage - 1) - u_peak))))
    if peak_unreset <= 0.0:
        peak_unreset = 1.0

    prof = np.zeros(d, dtype=float)
    last_reset = 0
    for i in range(d):
        for r in resets:
            if r <= i:
                last_reset = max(last_reset, r)
        u = np.arange(last_reset, i + 1)
        prof[i] = float(np.sum(c_full[last_reset : i + 1] * np.exp(-lam * (i - u)))) / peak_unreset
    return prof


def _te132_i132_filter_accumulation(
    c_deposition: np.ndarray,
    t_retention: int,
    filter_reset_steps: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes filter-trapped Te-132 and daughter I-132 activity trajectories via
    analytical Bateman convolution over D = t_passage + t_retention hours,
    normalized to 1.0 at peak passage, supporting intermediate filter resets.
    """
    t_passage = len(c_deposition)
    d = t_passage + t_retention
    c_full = np.zeros(d, dtype=float)
    c_full[:t_passage] = c_deposition
    if filter_reset_steps:
        tail_idx = np.arange(t_retention, dtype=float) + 1.0
        c_full[t_passage:] = c_deposition[-1] * np.exp(-tail_idx / max(3.0, 0.35 * t_passage))

    resets = sorted([int(r) for r in (filter_reset_steps or []) if 0 < int(r) < d])

    u_peak = np.arange(t_passage)
    a_te_pk, a_i_pk = bateman_te132_i132_activity_kernel(float(t_passage - 1) - u_peak, initial_i132_to_te132_activity_ratio=0.80)
    pk_te = max(1e-12, float(np.sum(c_deposition * a_te_pk)))
    pk_i = max(1e-12, float(np.sum(c_deposition * a_i_pk)))

    prof_te = np.zeros(d, dtype=float)
    prof_i = np.zeros(d, dtype=float)

    last_reset = 0
    for i in range(d):
        for r in resets:
            if r <= i:
                last_reset = max(last_reset, r)
        u = np.arange(last_reset, i + 1)
        dt = float(i) - u
        a_te, a_i = bateman_te132_i132_activity_kernel(dt, initial_i132_to_te132_activity_ratio=0.80)
        prof_te[i] = float(np.sum(c_full[last_reset : i + 1] * a_te)) / pk_te
        prof_i[i] = float(np.sum(c_full[last_reset : i + 1] * a_i)) / pk_i

    return prof_te, prof_i


def compute_deposition_rate_curve(shape_family: str, t_passage: int, rise_param: float) -> np.ndarray:
    """Computes dimensionless atmospheric plume deposition rate c(u) for u in 0..t_passage-1."""
    u = np.arange(t_passage, dtype=float)
    if shape_family == "linear_ramp":
        c = np.clip(u / max(1.0, float(rise_param)), 0.0, 1.0)
    elif shape_family == "step":
        c = np.ones(t_passage, dtype=float)
    elif shape_family == "sigmoidal":
        tau = max(0.5, float(rise_param))
        t_mid = min(4.0, t_passage / 3.0)
        c = 1.0 / (1.0 + np.exp(-(u - t_mid) / tau))
        c = (c - c[0]) / (c[-1] - c[0] + 1e-6)
    elif shape_family == "exponential":
        tau = max(0.8, float(rise_param))
        c = 1.0 - np.exp(-u / tau)
        c = c / (c[-1] + 1e-6)
    else:
        c = np.ones(t_passage, dtype=float)
    return np.maximum(0.0, c)


def compute_multinuclide_injection_timeseries(
    shape_family: str,
    t_passage: int,
    t_retention: int,
    peak_cpm: float,
    rise_param: float,
    scenario: str,
    spectrum_at_peak: np.ndarray,
    f_cs137: float = 0.0,
    f_i131: float = 0.0,
    f_co60: float = 0.0,
    f_cs134: float = 0.0,
    filter_reset_steps: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes multi-nuclide, time-evolving 8-channel count excesses Delta C_k(t)
    and total gross count excess S(t) over t in 0..D-1 (D = t_passage + t_retention).

    Models nuclide-specific filter accumulation and radioactive decay for:
      - I-131 (T_1/2 = 8.025 d, primary photopeak in R03 [364 keV])
      - Te-132 -> I-132 via analytical Bateman kinetics (T_1/2 = 76.9 h -> 2.295 h,
        emitting 228 keV in R03 and 667/773/955 keV in R05/R06) in reactor releases
      - Cs-134 (T_1/2 = 2.065 y, photopeaks in R05 [605 keV], R06 [796 keV], R07 [1168/1365 keV])
      - Co-60 (T_1/2 = 5.27 y, photopeaks in R07 [1173 & 1332 keV] and Compton in R02..R06)
      - Cs-137 (T_1/2 = 30.17 y, photopeak in R05 [662 keV])
      - Plus hold-out test nuclides: Ru-103, Zr-95/Nb-95, Ba-140/La-140.

    Returns
    -------
    tuple of (gross_profile, channel_profiles)
        gross_profile : np.ndarray of shape (D,)
        channel_profiles : np.ndarray of shape (D, 8) for R02..R09
    """
    w = np.asarray(spectrum_at_peak, dtype=float)
    w = np.maximum(0.0, w)
    w = w / np.sum(w)

    c_dep = compute_deposition_rate_curve(shape_family, t_passage, rise_param)

    # Nuclide-specific normalized time profiles (all equal 1.0 at t = t_passage - 1 when unreset)
    p_i131 = _single_nuclide_filter_accumulation(c_dep, t_retention, LAMBDA_I131, filter_reset_steps)
    p_cs137 = _single_nuclide_filter_accumulation(c_dep, t_retention, LAMBDA_CS137, filter_reset_steps)
    p_cs134 = _single_nuclide_filter_accumulation(c_dep, t_retention, LAMBDA_CS134, filter_reset_steps)
    p_co60 = _single_nuclide_filter_accumulation(c_dep, t_retention, LAMBDA_CO60, filter_reset_steps)
    p_te132, p_i132 = _te132_i132_filter_accumulation(c_dep, t_retention, filter_reset_steps)

    if scenario == "fission_pure_cs137":
        channel_profiles = np.outer(p_cs137, w) * float(peak_cpm)
    elif scenario == "fission_pure_i131":
        channel_profiles = np.outer(p_i131, w) * float(peak_cpm)
    elif scenario == "activation_orphan_co60":
        channel_profiles = np.outer(p_co60, w) * float(peak_cpm)
    elif scenario == "fission_pure_ru103":
        p_ru103 = _single_nuclide_filter_accumulation(c_dep, t_retention, LAMBDA_RU103, filter_reset_steps)
        channel_profiles = np.outer(p_ru103, w) * float(peak_cpm)
    elif scenario == "fission_zr95_nb95":
        p_zr95 = _single_nuclide_filter_accumulation(c_dep, t_retention, LAMBDA_ZR95, filter_reset_steps)
        p_nb95 = _single_nuclide_filter_accumulation(c_dep, t_retention, LAMBDA_NB95, filter_reset_steps)
        channel_profiles = np.outer(0.5 * (p_zr95 + p_nb95), w) * float(peak_cpm)
    elif scenario == "fission_ba140_la140":
        p_ba140 = _single_nuclide_filter_accumulation(c_dep, t_retention, LAMBDA_BA140, filter_reset_steps)
        p_la140 = _single_nuclide_filter_accumulation(c_dep, t_retention, LAMBDA_LA140, filter_reset_steps)
        channel_profiles = np.outer(0.45 * p_ba140 + 0.55 * p_la140, w) * float(peak_cpm)
    elif scenario == "fission_reactor_fukushima":
        # Fresh reactor core release: includes short-lived Te-132 -> I-132 pair via Bateman kinetics,
        # I-131 (8.025 d), Cs-134 (2.065 y), and Cs-137 (30.17 y).
        # Partition each channel's peak share w[k] into (I131, Te132, I132, Cs134, Cs137) so sum_m w_m[k] == w[k]:
        alloc = np.array([
            # I131, Te132, I132, Cs134, Cs137
            [0.30, 0.15, 0.10, 0.20, 0.25],  # R02: Compton + backscatter
            [0.62, 0.23, 0.03, 0.05, 0.07],  # R03: I-131 (364 keV) + Te-132 (228 keV)
            [0.20, 0.05, 0.25, 0.25, 0.25],  # R04: Compton + I-132 (523 keV)
            [0.05, 0.00, 0.22, 0.33, 0.40],  # R05: Cs-137 (662) + Cs-134 (605) + I-132 (667/773)
            [0.04, 0.00, 0.26, 0.60, 0.10],  # R06: Cs-134 (796/802) + I-132 (955)
            [0.02, 0.00, 0.13, 0.80, 0.05],  # R07: Cs-134 (1168/1365 keV) + I-132 (1399 keV)
            [0.05, 0.00, 0.15, 0.65, 0.15],  # R08: high-energy tail
            [0.05, 0.00, 0.15, 0.65, 0.15],  # R09: high-energy tail
        ], dtype=float)
        # Scale by nuclide presence if custom fractions passed
        time_basis = np.column_stack([p_i131, p_te132, p_i132, p_cs134, p_cs137])  # (D, 5)
        w_comp = alloc * w[:, None]  # (8, 5), sums over axis=1 to w
        channel_profiles = (time_basis @ w_comp.T) * float(peak_cpm)
    else:  # mixed_fission_activation (Cs-137, I-131, Co-60, Cs-134)
        alloc = np.array([
            # I131, Cs137, Cs134, Co60
            [0.30, 0.30, 0.20, 0.20],  # R02
            [0.72, 0.10, 0.08, 0.10],  # R03: I-131 364 keV photopeak
            [0.18, 0.32, 0.25, 0.25],  # R04
            [0.06, 0.54, 0.30, 0.10],  # R05: Cs-137 662 keV & Cs-134 605 keV
            [0.04, 0.12, 0.54, 0.30],  # R06: Cs-134 796 keV & Co-60 Compton
            [0.02, 0.03, 0.25, 0.70],  # R07: Co-60 1173/1332 keV & Cs-134 1168/1365 keV
            [0.02, 0.03, 0.20, 0.75],  # R08: Co-60 high-energy tail
            [0.02, 0.03, 0.20, 0.75],  # R09
        ], dtype=float)
        time_basis = np.column_stack([p_i131, p_cs137, p_cs134, p_co60])  # (D, 4)
        w_comp = alloc * w[:, None]  # (8, 4), sums over axis=1 to w
        channel_profiles = (time_basis @ w_comp.T) * float(peak_cpm)

    gross_profile = np.sum(channel_profiles, axis=1)
    return gross_profile, channel_profiles


# ---------------------------------------------------------------------------
# 3. Causal Spectral Stripping (Unmixing) & Bateman Washout ODE Solver
# ---------------------------------------------------------------------------
def compute_spectral_stripping(
    delta_channels: np.ndarray,
    radon_basis_spectrum: np.ndarray | list[float] | None = None,
    channel_sigma_dry: np.ndarray | list[float] | None = None,
) -> dict[str, np.ndarray]:
    """
    Strictly causal spectral unmixing (stripping) of the natural radon washout
    basis spectrum s_Rn from observed non-negative channel excesses Delta C(t).

    For each timestamp t (row):
      1. Optimal non-negative projection amplitude onto s_Rn:
         alpha_hat(t) = max(0, (Delta C(t) . s_Rn) / ||s_Rn||_2^2)
      2. Relative L2 residual norm:
         radon_stripping_residual_l2(t) = ||Delta C(t) - alpha_hat(t) * s_Rn||_2 / (Delta Gross(t) + 10.0)
         (Note: when Delta C(t) == c * s_Rn for any c >= 0, residual is identically 0.0.)
      3. Channel-specific radon-stripped excesses predicted from high-energy Bi-214 lines (R07+R08):
         - stripped_excess_r03_z: R03 excess minus (s_Rn[R03] / (s_Rn[R07]+s_Rn[R08])) * (Delta R07 + Delta R08)
         - stripped_excess_r05_z: R05 excess minus (s_Rn[R05] / (s_Rn[R07]+s_Rn[R08])) * (Delta R07 + Delta R08)
         - stripped_excess_r07_z: R07 excess minus (s_Rn[R07] / (s_Rn[R02]+s_Rn[R03])) * (Delta R02 + Delta R03)

    Parameters
    ----------
    delta_channels : np.ndarray of shape (N, 8) or (8,)
        Non-negative channel count excesses above baseline for R02..R09.
    radon_basis_spectrum : array-like of shape (8,), optional
        Normalized 8-channel radon washout basis spectrum s_Rn.
    channel_sigma_dry : array-like of shape (8,), optional
        Dry baseline standard deviations for R02..R09.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing `alpha_rn`, `residual_vec`, `radon_stripping_residual_l2`,
        `stripped_excess_r03_z`, `stripped_excess_r05_z`, `stripped_excess_r07_z`.
    """
    arr = np.asarray(delta_channels, dtype=float)
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr[None, :]
    if arr.shape[1] != 8:
        raise ValueError(f"delta_channels must have 8 columns (R02..R09), got shape {arr.shape}.")

    s_rn = (
        np.asarray(DEFAULT_RADON_BASIS, dtype=float)
        if radon_basis_spectrum is None
        else np.asarray(radon_basis_spectrum, dtype=float)
    )
    s_rn = np.maximum(0.0, s_rn)
    s_rn = s_rn / np.sum(s_rn)

    if channel_sigma_dry is None:
        sig = np.array([120.0, 95.0, 35.0, 25.0, 15.0, 15.0, 10.0, 8.0], dtype=float)
    else:
        sig = np.maximum(1.0, np.asarray(channel_sigma_dry, dtype=float))

    delta_pos = np.maximum(0.0, arr)
    delta_gross = np.sum(delta_pos, axis=1)

    s_norm_sq = float(np.dot(s_rn, s_rn))
    alpha_rn = np.maximum(0.0, (delta_pos @ s_rn) / s_norm_sq)
    residual_vec = delta_pos - np.outer(alpha_rn, s_rn)
    res_l2 = np.linalg.norm(residual_vec, axis=1) / (delta_gross + 10.0)

    # Bi-214 high-energy reference (R07 + R08: indices 5 and 6)
    r07_08_exc = delta_pos[:, 5] + delta_pos[:, 6]
    s_r07_08 = max(1e-6, float(s_rn[5] + s_rn[6]))
    ratio_3_to_78 = float(s_rn[1]) / s_r07_08
    ratio_5_to_78 = float(s_rn[3]) / s_r07_08

    stripped_r03_z = (delta_pos[:, 1] - ratio_3_to_78 * r07_08_exc) / sig[1]
    stripped_r05_z = (delta_pos[:, 3] - ratio_5_to_78 * r07_08_exc) / sig[3]

    # Low-energy Pb-214 reference (R02 + R03: indices 0 and 1) for stripping R07 (Co-60 / Cs-134)
    r02_03_exc = delta_pos[:, 0] + delta_pos[:, 1]
    s_r02_03 = max(1e-6, float(s_rn[0] + s_rn[1]))
    ratio_7_to_23 = float(s_rn[5]) / s_r02_03
    stripped_r07_z = (delta_pos[:, 5] - ratio_7_to_23 * r02_03_exc) / sig[5]

    out = {
        "alpha_rn": alpha_rn,
        "residual_vec": residual_vec,
        "radon_stripping_residual_l2": res_l2,
        "stripped_excess_r03_z": stripped_r03_z,
        "stripped_excess_r05_z": stripped_r05_z,
        "stripped_excess_r07_z": stripped_r07_z,
    }
    if is_1d:
        return {k: (v[0] if v.ndim == 1 else v[0, :]) for k, v in out.items()}
    return out


def solve_bateman_washout_ode(
    precip_1h_mm: pd.Series | np.ndarray,
    beta: float = 0.65,
    memory_hours: int = 6,
) -> dict[str, np.ndarray]:
    """
    Solves the coupled 3-stage Bateman differential equations for filter-trapped
    short-lived radon progeny driven by causal precipitation scavenging:
        Po-218 (T_1/2 = 3.10 min) -> Pb-214 (T_1/2 = 26.8 min) -> Bi-214 (T_1/2 = 19.9 min)
    on the continuous 1-hour grid with strict NaN propagation.

    Over each 1-hour step with scavenging rate W(t) = max(0, precip_1h_mm(t))^beta,
    continuous scavenging injection is integrated exactly using the analytical
    Bateman propagator across 12 sub-steps (5-minute sub-grid) to build the exact
    causal hourly impulse-response kernels for Po-218, Pb-214, and Bi-214 activities.

    Strict NaN Propagation:
    Any missing (`NaN`) precipitation value at `t - k` (`0 <= k < memory_hours`)
    strictly propagates `NaN` to all ODE outputs at hour `t`.

    Parameters
    ----------
    precip_1h_mm : pd.Series or np.ndarray
        Hourly precipitation accumulation (mm) on the continuous hourly UTC grid.
    beta : float, default 0.65
        Power-law scavenging exponent W(t) = precip(t)^beta.
    memory_hours : int, default 6
        Causal horizon in hours over which the analytical Bateman response is integrated
        (at 6 hours, >99.98% of filter-trapped radon progeny have decayed).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:
          - `a_po218`: normalized Po-218 activity state
          - `a_pb214`: normalized Pb-214 activity state
          - `a_bi214`: normalized Bi-214 activity state
          - `bateman_expected_index`: combined NaI(Tl) gamma response index (0.36 Pb-214 + 0.64 Bi-214)
    """
    p = pd.Series(precip_1h_mm, dtype=float).reset_index(drop=True)
    w = np.power(np.maximum(0.0, p.to_numpy(dtype=float)), float(beta))
    w = np.where(p.isna().to_numpy(), np.nan, w)

    # Precompute exact hourly impulse response kernels for a 1-hour constant scavenging pulse
    # using 12 sub-intervals of 5 minutes (1/12 hour) via analytical 3-step Bateman solution
    n_sub = 12
    dt_sub = 1.0 / n_sub
    k_po = np.zeros(memory_hours, dtype=float)
    k_pb = np.zeros(memory_hours, dtype=float)
    k_bi = np.zeros(memory_hours, dtype=float)

    # Wet deposition injects dissolved Rn-222 progeny (Po-218:Pb-214:Bi-214 activity shares ~ 1.0 : 0.8 : 0.6)
    n1_pulse = (1.0 / LAMBDA_PO218) * dt_sub
    n2_pulse = (0.8 / LAMBDA_PB214) * dt_sub
    n3_pulse = (0.6 / LAMBDA_BI214) * dt_sub

    for lag_h in range(memory_hours):
        # Sum contributions from each 5-minute sub-step inside the pulse hour (lag_h hours ago)
        sub_offsets = lag_h + (np.arange(n_sub) + 0.5) * dt_sub
        n1, n2, n3, _ = bateman_three_step_atoms(
            n1_pulse, n2_pulse, n3_pulse,
            LAMBDA_PO218, LAMBDA_PB214, LAMBDA_BI214,
            sub_offsets,
        )
        k_po[lag_h] = float(np.sum(LAMBDA_PO218 * n1))
        k_pb[lag_h] = float(np.sum(LAMBDA_PB214 * n2))
        k_bi[lag_h] = float(np.sum(LAMBDA_BI214 * n3))

    # Normalize kernels so unit steady rain produces unit combined gamma response at lag 0
    norm_factor = 0.36 * k_pb[0] + 0.64 * k_bi[0]
    k_po /= norm_factor
    k_pb /= norm_factor
    k_bi /= norm_factor
    k_comb = 0.36 * k_pb + 0.64 * k_bi

    # Causal convolution on continuous hourly grid with strict NaN propagation
    n = len(w)
    a_po = np.zeros(n, dtype=float)
    a_pb = np.zeros(n, dtype=float)
    a_bi = np.zeros(n, dtype=float)
    comb = np.zeros(n, dtype=float)

    w_series = pd.Series(w)
    for lag_h in range(memory_hours):
        w_lag = w_series.shift(lag_h).to_numpy(dtype=float)
        # At the very start of the series (index < lag_h), shift introduces NaN;
        # if all observed hours up to index i are non-NaN, treat pre-series warmup as 0.0
        # while strictly propagating any actual NaN present in w[:i+1].
        if lag_h > 0:
            known_start = ~w_series.isna().rolling(lag_h + 1, min_periods=1).max().astype(bool).to_numpy()
            pre_series_mask = (np.arange(n) < lag_h) & known_start
            w_lag = np.where(pre_series_mask, 0.0, w_lag)
        a_po = a_po + k_po[lag_h] * w_lag
        a_pb = a_pb + k_pb[lag_h] * w_lag
        a_bi = a_bi + k_bi[lag_h] * w_lag
        comb = comb + k_comb[lag_h] * w_lag

    return {
        "a_po218": a_po,
        "a_pb214": a_pb,
        "a_bi214": a_bi,
        "bateman_expected_index": comb,
    }
