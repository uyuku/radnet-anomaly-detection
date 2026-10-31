"""
Derived Detector-Response Spectral Templates & Independent Dose Coupling Module.

Replaces hand-written uniform/Dirichlet channel shares in `src/synthetic_injection.py`
with physical detector-response templates for particulate activity deposited on an
EPA RadNet glass-fiber air filter viewed by a 2"x2" (5.08 cm x 5.08 cm) NaI(Tl)
scintillation spectrometer.

Provenance & External Data Sources:
1. EPA RadNet Stationary Air Monitor Configuration:
   - Source URL: https://www.epa.gov/radnet/about-radnet-air-monitoring-data
   - Retrieval Date: 2026-10-31
   - Documented: 2"x2" (5.08 cm diameter x 5.08 cm height) NaI(Tl) scintillation detector
     monitoring airborne particulates collected on a continuous/replaceable filter, binned
     into 9 energy windows (R01: 0-100 keV, R02: 101-200 keV, R03: 201-400 keV,
     R04: 401-600 keV, R05: 601-800 keV, R06: 801-1000 keV, R07: 1001-1400 keV,
     R08: 1401-1800 keV, R09: 1801-2200 keV; see configs/config.yaml).
   - Unpublished Geometric Parameters (not published in EPA public portal; treated as
     uncertain random parameters drawn from stated distributions):
     * Filter-to-crystal face standoff distance d_standoff_cm: nominal 1.5 cm,
       uncertainty Uniform(0.8, 2.5) cm [Estimated]
     * Active filter deposition radius r_filter_cm: nominal 2.0 cm (47 mm filter holder),
       uncertainty Uniform(1.6, 2.4) cm [Estimated]
     * Aluminum detector housing window thickness al_window_mm: nominal 0.8 mm,
       uncertainty Uniform(0.5, 1.5) mm [Estimated]
     * Shield/housing backscatter fraction f_backscatter: nominal 0.08,
       uncertainty Uniform(0.04, 0.14) [Estimated]
2. Nuclear Decay Data (Gamma Energies E_gamma and Absolute Intensities I_gamma):
   - Source: National Nuclear Data Center (NNDC), Brookhaven National Laboratory,
     Evaluated Nuclear Structure Data File (ENSDF) / NuDat 3.0.
   - Source URL: https://www.nndc.bnl.gov/nudat3/
   - Retrieval Date: 2026-10-31
3. Photon Attenuation Cross Sections in NaI (rho = 3.67 g/cm^3) and Al (rho = 2.70 g/cm^3):
   - Source: NIST XCOM Photon Cross Sections Database (Berger et al., NBSIR 87-3597).
   - Source URL: https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html
   - Retrieval Date: 2026-10-31

Template Families (Part B.1):
- Family A (`A_analytic`, used for training plumes):
  Analytic disk-to-cylinder solid-angle integration + NIST XCOM photoelectric/Compton
  interaction model + single-scatter Klein-Nishina Compton continuum + Gaussian energy
  resolution broadening. Explicitly documented as an analytic approximation.
- Family B (`B_monte_carlo`, used for test plumes):
  3D Monte Carlo multi-scatter photon ray-tracing in a finite right-circular cylindrical
  NaI(Tl) crystal (tracking up to 4 sequential Compton scatters before photoelectric
  capture or escape from crystal boundaries) with independent parameter distributions
  for energy resolution, gain drift, and shield backscatter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ME_KEV = 510.99895  # Electron rest mass energy (keV)
NAI_DENSITY_G_CM3 = 3.67
AL_DENSITY_G_CM3 = 2.70

EPA_RADNET_DOC_URL = "https://www.epa.gov/radnet/about-radnet-air-monitoring-data"
NNDC_ENSDF_URL = "https://www.nndc.bnl.gov/nudat3/"
NIST_XCOM_URL = "https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html"
RETRIEVAL_DATE = "2026-10-31"
GEOMETRY_STATUS = (
    "Estimated: Crystal 5.08x5.08 cm documented; filter standoff (0.8-2.5 cm), "
    "filter active radius (1.6-2.4 cm), Al window (0.5-1.5 mm), and housing backscatter "
    "(0.04-0.14) are estimated and sampled as uncertain parameters."
)

# 9 RadNet energy windows from configs/config.yaml (keV)
RADNET_WINDOWS_9: list[tuple[str, float, float]] = [
    ("R01", 0.0, 100.0),
    ("R02", 100.0, 200.0),
    ("R03", 200.0, 400.0),
    ("R04", 400.0, 600.0),
    ("R05", 600.0, 800.0),
    ("R06", 800.0, 1000.0),
    ("R07", 1000.0, 1400.0),
    ("R08", 1400.0, 1800.0),
    ("R09", 1800.0, 2200.0),
]
CHANNELS_8 = [f"cpm_r0{i}" for i in range(2, 10)]

# Published NNDC / ENSDF gamma lines: list of (energy_keV, photons_per_decay)
NUCLIDE_GAMMA_LINES: dict[str, list[tuple[float, float]]] = {
    "Cs-137": [
        (661.657, 0.8510),  # via Ba-137m secular equilibrium
    ],
    "Cs-134": [
        (475.365, 0.01479),
        (563.246, 0.08338),
        (569.331, 0.1537),
        (604.721, 0.9762),
        (795.864, 0.8546),
        (801.953, 0.08688),
        (1038.610, 0.00990),
        (1167.968, 0.01790),
        (1365.185, 0.03017),
    ],
    "I-131": [
        (80.185, 0.0262),
        (284.305, 0.0612),
        (364.489, 0.8150),
        (636.989, 0.0716),
        (722.911, 0.0177),
    ],
    "Co-60": [
        (1173.228, 0.9985),
        (1332.492, 0.9998),
    ],
    "Te-132": [
        (49.720, 0.1500),
        (228.160, 0.8800),
    ],
    "I-132": [
        (522.650, 0.1600),
        (630.190, 0.1330),
        (667.714, 0.9870),
        (772.600, 0.7560),
        (954.550, 0.1760),
        (1398.570, 0.0710),
    ],
    # Test-only hold-out radionuclides (Hold-out Realism B.2.ii)
    "Ru-103": [
        (497.085, 0.9100),
        (610.333, 0.0576),
    ],
    "Zr-95": [
        (724.192, 0.4427),
        (756.725, 0.5438),
    ],
    "Nb-95": [
        (765.803, 0.9981),
    ],
    "Ba-140": [
        (162.660, 0.0622),
        (304.849, 0.0429),
        (423.722, 0.0315),
        (437.575, 0.0193),
        (537.261, 0.2439),
    ],
    "La-140": [
        (328.762, 0.2030),
        (487.021, 0.4550),
        (815.772, 0.2328),
        (867.846, 0.0550),
        (925.189, 0.0690),
        (1596.210, 0.9540),
    ],
    # Natural radon progeny for verification
    "Pb-214": [
        (241.995, 0.0725),
        (295.223, 0.1842),
        (351.932, 0.3560),
        (785.960, 0.0106),
    ],
    "Bi-214": [
        (609.320, 0.4549),
        (768.360, 0.0489),
        (934.056, 0.0310),
        (1120.294, 0.1492),
        (1238.122, 0.0583),
        (1377.669, 0.0399),
        (1407.988, 0.0239),
        (1509.210, 0.0213),
        (1729.595, 0.0288),
        (1764.491, 0.1530),
        (2204.059, 0.0492),
    ],
}

# NIST XCOM tabulated mass attenuation coefficients (cm^2/g) for NaI
_XCOM_ENERGIES_KEV = np.array(
    [50.0, 80.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0, 662.0, 800.0, 1000.0, 1250.0, 1500.0, 2000.0, 2500.0],
    dtype=float,
)
_XCOM_NAI_PHOTO_CM2_G = np.array(
    [10.20, 2.82, 1.50, 0.470, 0.205, 0.0630, 0.0278, 0.0150, 0.0093, 0.0072, 0.0045, 0.0026, 0.0016, 0.0011, 0.00060, 0.00038],
    dtype=float,
)
_XCOM_NAI_COMPTON_CM2_G = np.array(
    [0.125, 0.122, 0.118, 0.109, 0.101, 0.0886, 0.0796, 0.0726, 0.0670, 0.0641, 0.0586, 0.0524, 0.0466, 0.0422, 0.0358, 0.0313],
    dtype=float,
)
_XCOM_AL_TOTAL_CM2_G = np.array(
    [0.368, 0.202, 0.170, 0.138, 0.122, 0.104, 0.0928, 0.0845, 0.0780, 0.0747, 0.0684, 0.0615, 0.0550, 0.0500, 0.0432, 0.0385],
    dtype=float,
)


@dataclass(frozen=True)
class DetectorGeometryParams:
    """Physical detector and filter geometry parameters (with estimated bounds documented)."""

    crystal_radius_cm: float = 2.54      # 2" diameter NaI(Tl) crystal (documented)
    crystal_length_cm: float = 5.08      # 2" height NaI(Tl) crystal (documented)
    standoff_cm: float = 1.50            # Filter-to-crystal face distance (estimated, nominal 1.5 cm)
    filter_radius_cm: float = 2.00       # Active particulate deposit radius (estimated, nominal 2.0 cm)
    al_window_mm: float = 0.80           # Aluminum housing window thickness (estimated, nominal 0.8 mm)
    fwhm_662_pct: float = 7.50           # Energy resolution FWHM at 662 keV (%)
    gain_shift_frac: float = 0.0         # Fractional gain shift delta_g
    backscatter_fraction: float = 0.08   # Housing/shield backscatter albedo (estimated)


def nai_linear_attenuation_cm1(energy_kev: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns linear attenuation coefficients (mu_photo, mu_compton, mu_total) in cm^-1
    for NaI (rho = 3.67 g/cm^3) interpolated in log-log space from NIST XCOM.
    """
    e = np.clip(np.asarray(energy_kev, dtype=float), 50.0, 2500.0)
    log_e = np.log(e)
    log_grid = np.log(_XCOM_ENERGIES_KEV)
    mu_pe = NAI_DENSITY_G_CM3 * np.exp(np.interp(log_e, log_grid, np.log(_XCOM_NAI_PHOTO_CM2_G)))
    mu_comp = NAI_DENSITY_G_CM3 * np.exp(np.interp(log_e, log_grid, np.log(_XCOM_NAI_COMPTON_CM2_G)))
    return mu_pe, mu_comp, mu_pe + mu_comp


def al_window_transmission(energy_kev: np.ndarray | float, al_window_mm: np.ndarray | float) -> np.ndarray:
    """Returns photon transmission factor through aluminum entrance window."""
    e = np.clip(np.asarray(energy_kev, dtype=float), 50.0, 2500.0)
    mu_al = AL_DENSITY_G_CM3 * np.exp(
        np.interp(np.log(e), np.log(_XCOM_ENERGIES_KEV), np.log(_XCOM_AL_TOTAL_CM2_G))
    )
    t_cm = np.maximum(0.0, np.asarray(al_window_mm, dtype=float)) * 0.1
    return np.exp(-mu_al * t_cm)


def disk_to_cylinder_solid_angle_fraction(
    standoff_cm: float,
    filter_radius_cm: float,
    crystal_radius_cm: float,
) -> float:
    """
    Computes average fractional solid angle Omega / (4*pi) subtended by a coaxial
    circular crystal face of radius R_c from a circular filter deposit of radius R_s
    at distance d.
    """
    r_grid = np.linspace(0.0, filter_radius_cm, 16)
    d = max(0.1, float(standoff_cm))
    rc = float(crystal_radius_cm)
    # Ruby / Paxton off-axis circular disk solid angle approximation
    dist_eff = np.sqrt(d**2 + 0.5 * r_grid**2 + rc**2)
    omega_r = 0.5 * (1.0 - d / dist_eff)
    weights = r_grid + 1e-6
    return float(np.sum(omega_r * weights) / np.sum(weights))


def klein_nishina_recoil_pdf(energy_kev: float, recoil_grid_kev: np.ndarray) -> np.ndarray:
    """
    Computes normalized Klein-Nishina differential probability density dsigma/dT
    for Compton recoil electron energy T in [0, E_compton_edge].
    """
    alpha = float(energy_kev) / ME_KEV
    e_edge = float(energy_kev) * (2.0 * alpha) / (1.0 + 2.0 * alpha)
    pdf = np.zeros_like(recoil_grid_kev, dtype=float)
    mask = (recoil_grid_kev > 0.0) & (recoil_grid_kev <= e_edge)
    if not np.any(mask):
        return pdf
    t = recoil_grid_kev[mask]
    s = t / float(energy_kev)  # fractional energy transfer
    # Klein-Nishina differential formula wrt recoil energy T
    one_minus_s = np.maximum(1e-8, 1.0 - s)
    term1 = 1.0 + (s / one_minus_s) ** 2
    cos_theta = 1.0 - s / (alpha * one_minus_s)
    sin2_theta = np.clip(1.0 - cos_theta**2, 0.0, 1.0)
    val = np.maximum(0.0, (1.0 / one_minus_s) * (1.0 + one_minus_s**2 - one_minus_s * sin2_theta))
    pdf[mask] = val
    area = float(np.trapezoid(pdf, recoil_grid_kev)) if hasattr(np, "trapezoid") else float(np.trapz(pdf, recoil_grid_kev))
    if area > 0.0:
        pdf /= area
    return pdf


def compute_analytic_line_response(
    energy_kev: float,
    params: DetectorGeometryParams = DetectorGeometryParams(),
) -> tuple[np.ndarray, float]:
    """
    Family A (Analytic Approximation):
    Computes 9-channel deposited energy distribution across R01..R09 and total
    detection efficiency per emitted photon for a monoenergetic gamma ray of `energy_kev`.
    """
    e_grid = np.linspace(2.0, 2300.0, 1150)
    de = e_grid[1] - e_grid[0]

    geom_frac = disk_to_cylinder_solid_angle_fraction(
        params.standoff_cm, params.filter_radius_cm, params.crystal_radius_cm
    )
    t_al = float(al_window_transmission(energy_kev, params.al_window_mm))
    mu_pe, mu_comp, mu_tot = nai_linear_attenuation_cm1(energy_kev)
    mu_pe_f, mu_comp_f, mu_tot_f = float(mu_pe), float(mu_comp), float(mu_tot)

    # Mean chord length in coaxial right-circular cylinder viewed from standoff
    mean_chord_cm = 0.82 * params.crystal_length_cm
    p_interact = 1.0 - np.exp(-mu_tot_f * mean_chord_cm)
    total_eff = geom_frac * t_al * p_interact

    # Multi-scatter photofraction in 2"x2" NaI(Tl): primary PE + re-absorbed Compton scatters
    primary_pe_frac = mu_pe_f / mu_tot_f
    avg_scattered_e = max(60.0, energy_kev / (1.0 + energy_kev / ME_KEV))
    mu_pe_scat, _, mu_tot_scat = nai_linear_attenuation_cm1(avg_scattered_e)
    p_reabsorb = (1.0 - np.exp(-float(mu_tot_scat) * (0.55 * params.crystal_radius_cm))) * (
        float(mu_pe_scat) / float(mu_tot_scat) + 0.38
    )
    p_reabsorb = float(np.clip(p_reabsorb, 0.0, 0.85))

    fep_frac = primary_pe_frac + (1.0 - primary_pe_frac) * p_reabsorb
    fep_frac = float(np.clip(fep_frac, 0.05, 0.98))
    compton_frac = max(0.0, 1.0 - fep_frac - params.backscatter_fraction)
    backscatter_frac = 1.0 - fep_frac - compton_frac

    # Unbroadened spectrum on e_grid
    spec = np.zeros_like(e_grid)

    # 1. Full-energy photopeak at E_gamma
    fep_idx = int(np.argmin(np.abs(e_grid - energy_kev)))
    spec[fep_idx] += fep_frac / de

    # 2. Compton continuum + partial multiple-scatter fill between Compton edge and photopeak
    kn_pdf = klein_nishina_recoil_pdf(energy_kev, e_grid)
    alpha = energy_kev / ME_KEV
    e_edge = energy_kev * (2.0 * alpha) / (1.0 + 2.0 * alpha)
    multi_fill = np.where((e_grid > e_edge) & (e_grid < energy_kev), 1.0, 0.0)
    if np.any(multi_fill > 0):
        multi_fill = multi_fill / (np.sum(multi_fill) * de)
        spec += compton_frac * (0.82 * kn_pdf + 0.18 * multi_fill)
    else:
        spec += compton_frac * kn_pdf

    # 3. Housing/shield backscatter peak at E_backscatter = E / (1 + 2*alpha)
    e_back = energy_kev / (1.0 + 2.0 * alpha)
    back_idx = int(np.argmin(np.abs(e_grid - e_back)))
    spec[back_idx] += backscatter_frac / de

    # Apply gain shift and energy-dependent Gaussian resolution broadening:
    # FWHM(E) = FWHM_662 * sqrt(662 * E)
    e_shifted = e_grid * (1.0 + params.gain_shift_frac)
    fwhm_662_frac = params.fwhm_662_pct / 100.0
    sigma_grid = np.maximum(4.0, (fwhm_662_frac / 2.35482) * np.sqrt(662.0 * np.maximum(20.0, e_grid)))

    broadened = np.zeros_like(e_grid)
    active = np.where(spec > 1e-10)[0]
    for idx in active:
        e_c = e_shifted[idx]
        sig = sigma_grid[idx]
        kernel = np.exp(-0.5 * ((e_grid - e_c) / sig) ** 2) / (np.sqrt(2.0 * np.pi) * sig)
        broadened += spec[idx] * de * kernel

    # Integrate into the 9 RadNet energy windows
    counts_9 = np.zeros(9, dtype=float)
    for w_i, (_, e_min, e_max) in enumerate(RADNET_WINDOWS_9):
        m = (e_grid >= e_min) & (e_grid <= e_max)
        counts_9[w_i] = float(np.sum(broadened[m]) * de)

    total_in_windows = float(np.sum(counts_9))
    if total_in_windows > 0.0:
        counts_9 /= total_in_windows
    return counts_9, float(total_eff)


def _sample_klein_nishina_cos_theta(energy_kev: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
     Kahn / Butcher-Messel rejection sampling for Klein-Nishina Compton scattering cosine
    for an array of photon energies `energy_kev`.
    """
    n = len(energy_kev)
    alpha = np.maximum(1e-4, energy_kev / ME_KEV)
    cos_theta = np.zeros(n, dtype=float)
    remaining = np.arange(n)

    for _ in range(30):
        if len(remaining) == 0:
            break
        a = alpha[remaining]
        r1 = rng.uniform(0.0, 1.0, size=len(remaining))
        r2 = rng.uniform(0.0, 1.0, size=len(remaining))
        r3 = rng.uniform(0.0, 1.0, size=len(remaining))

        p1 = (1.0 + 2.0 * a) / (9.0 + 2.0 * a)
        branch1 = r1 <= p1
        x = np.where(branch1, 1.0 + 2.0 * a * r2, (1.0 + 2.0 * a) / (1.0 + 2.0 * a * r2))
        mu = 1.0 - (x - 1.0) / a
        sin2 = np.clip(1.0 - mu**2, 0.0, 1.0)
        weight = np.where(branch1, 0.5 * ((a + 1.0 - x) / a) ** 2 + 0.5, 0.5 * (1.0 - sin2 + 1.0 / x))
        accept = r3 <= weight
        cos_theta[remaining[accept]] = np.clip(mu[accept], -1.0, 1.0)
        remaining = remaining[~accept]

    if len(remaining) > 0:
        cos_theta[remaining] = rng.uniform(-1.0, 1.0, size=len(remaining))
    return cos_theta


def compute_monte_carlo_line_response(
    energy_kev: float,
    params: DetectorGeometryParams = DetectorGeometryParams(),
    n_histories: int = 25000,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """
    Family B (3D Monte Carlo Multi-Scatter Photon Transport in Cylindrical NaI(Tl) Crystal):
    Simulates 3D photon emission from the particulate filter disk, entrance window transmission,
    and up to 4 sequential photoelectric/Compton interactions inside the 2"x2" NaI(Tl) cylinder.
    """
    rng = np.random.RandomState(seed + int(round(energy_kev * 10)))

    # 1. Sample emission positions uniformly on circular filter at z = -standoff_cm
    r_s = params.filter_radius_cm * np.sqrt(rng.uniform(0.0, 1.0, size=n_histories))
    phi_s = rng.uniform(0.0, 2.0 * np.pi, size=n_histories)
    x0 = r_s * np.cos(phi_s)
    y0 = r_s * np.sin(phi_s)
    z0 = np.full(n_histories, -params.standoff_cm, dtype=float)

    # 2. Sample forward hemisphere directions (cosine cone covering crystal face)
    max_cos = params.standoff_cm / np.sqrt(
        params.standoff_cm**2 + (params.crystal_radius_cm + params.filter_radius_cm) ** 2
    )
    cone_solid_angle_frac = 0.5 * (1.0 - max_cos)
    cos_th = rng.uniform(max_cos, 1.0, size=n_histories)
    sin_th = np.sqrt(np.clip(1.0 - cos_th**2, 0.0, 1.0))
    phi_d = rng.uniform(0.0, 2.0 * np.pi, size=n_histories)
    ux = sin_th * np.cos(phi_d)
    uy = sin_th * np.sin(phi_d)
    uz = cos_th

    # Propagate to crystal entrance face z = 0
    t_entry = -z0 / uz
    x = x0 + ux * t_entry
    y = y0 + uy * t_entry
    z = np.zeros(n_histories, dtype=float)

    hit_face = (x**2 + y**2) <= params.crystal_radius_cm**2
    if not np.any(hit_face):
        return np.zeros(9, dtype=float), 0.0

    x = x[hit_face]
    y = y[hit_face]
    z = z[hit_face]
    ux = ux[hit_face]
    uy = uy[hit_face]
    uz = uz[hit_face]
    n_hit = len(x)

    # Include housing/shield backscattered photons entering crystal
    is_backscatter = rng.uniform(0.0, 1.0, size=n_hit) < params.backscatter_fraction
    alpha0 = energy_kev / ME_KEV
    e_back = energy_kev / (1.0 + 1.8 * alpha0)
    e_phot = np.where(is_backscatter, e_back, float(energy_kev))

    # Aluminum entrance window survival
    t_al = al_window_transmission(e_phot, params.al_window_mm / np.maximum(0.2, uz))
    survive_al = rng.uniform(0.0, 1.0, size=n_hit) < t_al

    x = x[survive_al]
    y = y[survive_al]
    z = z[survive_al]
    ux = ux[survive_al]
    uy = uy[survive_al]
    uz = uz[survive_al]
    e_phot = e_phot[survive_al]

    n_entered = len(e_phot)
    if n_entered == 0:
        return np.zeros(9, dtype=float), 0.0

    e_dep = np.zeros(n_entered, dtype=float)
    alive = np.ones(n_entered, dtype=bool)
    rc2 = params.crystal_radius_cm**2
    lc = params.crystal_length_cm

    # Track up to 4 interactions inside the cylindrical crystal
    for _step in range(4):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        mu_pe, mu_comp, mu_tot = nai_linear_attenuation_cm1(e_phot[idx])
        step_len = rng.exponential(scale=1.0 / np.maximum(1e-4, mu_tot))

        xn = x[idx] + ux[idx] * step_len
        yn = y[idx] + uy[idx] * step_len
        zn = z[idx] + uz[idx] * step_len

        inside = (xn**2 + yn**2 <= rc2) & (zn >= 0.0) & (zn <= lc)
        escaped_idx = idx[~inside]
        alive[escaped_idx] = False

        in_idx = idx[inside]
        if len(in_idx) == 0:
            break

        x[in_idx] = xn[inside]
        y[in_idx] = yn[inside]
        z[in_idx] = zn[inside]

        mu_pe_in = mu_pe[inside]
        mu_tot_in = mu_tot[inside]
        is_pe = rng.uniform(0.0, 1.0, size=len(in_idx)) < (mu_pe_in / mu_tot_in)

        # Photoelectric absorption deposits remaining photon energy
        pe_idx = in_idx[is_pe]
        e_dep[pe_idx] += e_phot[pe_idx]
        e_phot[pe_idx] = 0.0
        alive[pe_idx] = False

        # Compton scattering deposits recoil electron energy and deflects photon
        comp_idx = in_idx[~is_pe]
        if len(comp_idx) > 0:
            e_old = e_phot[comp_idx]
            cos_sc = _sample_klein_nishina_cos_theta(e_old, rng)
            e_new = e_old / (1.0 + (e_old / ME_KEV) * (1.0 - cos_sc))
            e_dep[comp_idx] += e_old - e_new
            e_phot[comp_idx] = e_new

            sin_sc = np.sqrt(np.clip(1.0 - cos_sc**2, 0.0, 1.0))
            phi_sc = rng.uniform(0.0, 2.0 * np.pi, size=len(comp_idx))
            # Rotate direction vector
            ux_o, uy_o, uz_o = ux[comp_idx], uy[comp_idx], uz[comp_idx]
            perp = np.sqrt(np.maximum(1e-8, ux_o**2 + uy_o**2))
            ux_n = ux_o * cos_sc + (sin_sc / perp) * (ux_o * uz_o * np.cos(phi_sc) - uy_o * np.sin(phi_sc))
            uy_n = uy_o * cos_sc + (sin_sc / perp) * (uy_o * uz_o * np.cos(phi_sc) + ux_o * np.sin(phi_sc))
            uz_n = uz_o * cos_sc - perp * sin_sc * np.cos(phi_sc)
            norm = np.sqrt(ux_n**2 + uy_n**2 + uz_n**2)
            ux[comp_idx] = ux_n / norm
            uy[comp_idx] = uy_n / norm
            uz[comp_idx] = uz_n / norm

    interacted = e_dep > 1.0
    n_det = int(np.sum(interacted))
    if n_det == 0:
        return np.zeros(9, dtype=float), 0.0

    total_eff = cone_solid_angle_frac * (n_det / float(n_histories))
    e_detected = e_dep[interacted] * (1.0 + params.gain_shift_frac)

    # Apply energy resolution broadening
    fwhm_662_frac = params.fwhm_662_pct / 100.0
    sigma_e = np.maximum(4.0, (fwhm_662_frac / 2.35482) * np.sqrt(662.0 * np.maximum(20.0, e_detected)))
    e_measured = e_detected + rng.normal(0.0, sigma_e)

    counts_9 = np.zeros(9, dtype=float)
    for w_i, (_, e_min, e_max) in enumerate(RADNET_WINDOWS_9):
        counts_9[w_i] = float(np.sum((e_measured >= e_min) & (e_measured <= e_max)))

    tot_w = float(np.sum(counts_9))
    if tot_w > 0.0:
        counts_9 /= tot_w
    return counts_9, float(total_eff)


def compute_nuclide_template(
    nuclide: str,
    family: str = "A",
    params: DetectorGeometryParams = DetectorGeometryParams(),
    n_histories_per_line: int = 20000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes the 9-channel (R01..R09) and 8-channel (R02..R09) fractional spectra
    and total detected counts per decay for a given radionuclide under Family A
    (analytic) or Family B (3D Monte Carlo).
    """
    if nuclide not in NUCLIDE_GAMMA_LINES:
        raise ValueError(f"Unknown nuclide {nuclide!r}. Available: {sorted(NUCLIDE_GAMMA_LINES.keys())}")

    lines = NUCLIDE_GAMMA_LINES[nuclide]
    weighted_9 = np.zeros(9, dtype=float)
    total_yield_eff = 0.0

    for line_idx, (e_kev, intensity) in enumerate(lines):
        if family.upper().startswith("A"):
            c9, eff = compute_analytic_line_response(e_kev, params=params)
        elif family.upper().startswith("B"):
            c9, eff = compute_monte_carlo_line_response(
                e_kev, params=params, n_histories=n_histories_per_line, seed=seed + line_idx * 97
            )
        else:
            raise ValueError(f"Unknown template family {family!r}; expected 'A' or 'B'.")

        w_line = float(intensity) * eff
        weighted_9 += w_line * c9
        total_yield_eff += w_line

    sum_9 = float(np.sum(weighted_9))
    shares_9 = weighted_9 / sum_9 if sum_9 > 0 else np.zeros(9, dtype=float)
    shares_8 = shares_9[1:].copy()
    sum_8 = float(np.sum(shares_8))
    if sum_8 > 0:
        shares_8 /= sum_8
    return shares_9, shares_8, float(total_yield_eff)


# Precompute base nuclide templates on import / first use so sampling 450 plumes is fast
_BASE_NUCLIDE_CACHE: dict[tuple[str, str, int, int], np.ndarray] = {}


def _get_cached_nuclide_shares8(
    nuclide: str,
    family: str,
    fwhm_bucket: float,
    gain_bucket: float,
    standoff_bucket: float,
) -> np.ndarray:
    fwhm_q = round(float(fwhm_bucket) * 2.0) / 2.0  # 0.5% steps
    gain_q = round(float(gain_bucket) / 0.015) * 0.015  # 1.5% steps
    key = (
        nuclide,
        "A" if family.upper().startswith("A") else "B",
        int(round(fwhm_q * 10)),
        int(round(gain_q * 1000)),
    )
    if key not in _BASE_NUCLIDE_CACHE:
        params = DetectorGeometryParams(
            standoff_cm=standoff_bucket,
            filter_radius_cm=2.0 if family.upper().startswith("A") else 2.15,
            al_window_mm=0.8 if family.upper().startswith("A") else 1.0,
            fwhm_662_pct=fwhm_q,
            gain_shift_frac=gain_q,
            backscatter_fraction=0.07 if family.upper().startswith("A") else 0.10,
        )
        _, s8, _ = compute_nuclide_template(
            nuclide=nuclide,
            family=family,
            params=params,
            n_histories_per_line=6000,
            seed=42 + abs(hash(key)) % 1000,
        )
        _BASE_NUCLIDE_CACHE[key] = s8
    return _BASE_NUCLIDE_CACHE[key].copy()


def sample_derived_spectrum(
    rng: np.random.RandomState,
    scenario: str,
    family: str = "A",
    gain_shift_override: float | None = None,
    fwhm_662_override: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Samples an 8-channel (R02..R09) spectrum and nuclide mixture dictionary for a
    given radiological scenario from either:
      - Family A ('A'): Analytic detector-response model (used for training plumes)
      - Family B ('B'): 3D Monte Carlo photon transport model (used for test plumes)

    Physical spread (Part A.3) is generated entirely from physical detector parameters
    (energy resolution FWHM, PMT gain shift, standoff geometry, and radionuclide
    activity mixture fractions), NOT from hand-written channel uniform/Dirichlet draws.
    """
    is_family_a = family.upper().startswith("A")

    if fwhm_662_override is not None:
        fwhm = float(fwhm_662_override)
    else:
        # Family A uses 6.8-8.8% FWHM; Family B uses 7.5-10.0% FWHM (field NaI aging)
        fwhm = round(float(rng.uniform(6.8, 8.8) if is_family_a else rng.uniform(7.5, 10.0)), 1)

    if gain_shift_override is not None:
        gain = float(gain_shift_override)
    else:
        # Family A uses +/-1.5% gain stability; Family B uses +/-3.0% thermal gain drift
        gain_sd = 0.015 if is_family_a else 0.030
        gain = round(float(np.clip(rng.normal(0.0, gain_sd), -0.06, 0.06)), 3)

    standoff = 1.4 if is_family_a else 1.8

    # Define scenario nuclide activity fractions with physical mixture spread
    if scenario == "fission_pure_cs137":
        nuc = {"fraction_cs137": 1.0, "fraction_i131": 0.0, "fraction_co60": 0.0, "fraction_cs134": 0.0}
        s8 = _get_cached_nuclide_shares8("Cs-137", family, fwhm, gain, standoff)

    elif scenario == "fission_pure_i131":
        nuc = {"fraction_cs137": 0.0, "fraction_i131": 1.0, "fraction_co60": 0.0, "fraction_cs134": 0.0}
        s8 = _get_cached_nuclide_shares8("I-131", family, fwhm, gain, standoff)

    elif scenario == "fission_reactor_fukushima":
        # Reactor volatile release: Cs-137, Cs-134, I-131, and Te-132/I-132
        alpha_mix = [4.0, 3.0, 3.0] if is_family_a else [3.2, 2.8, 4.0]
        draws = rng.dirichlet(alpha_mix)
        f_cs137 = round(float(draws[0]), 4)
        f_cs134 = round(float(draws[1]), 4)
        f_i131 = round(float(1.0 - f_cs137 - f_cs134), 4)
        nuc = {"fraction_cs137": f_cs137, "fraction_i131": f_i131, "fraction_co60": 0.0, "fraction_cs134": f_cs134}

        s_cs137 = _get_cached_nuclide_shares8("Cs-137", family, fwhm, gain, standoff)
        s_cs134 = _get_cached_nuclide_shares8("Cs-134", family, fwhm, gain, standoff)
        s_i131 = _get_cached_nuclide_shares8("I-131", family, fwhm, gain, standoff)
        s_te132 = _get_cached_nuclide_shares8("Te-132", family, fwhm, gain, standoff)
        s_i132 = _get_cached_nuclide_shares8("I-132", family, fwhm, gain, standoff)
        s_iod_tel = 0.50 * s_i131 + 0.22 * s_te132 + 0.28 * s_i132
        s8 = f_cs137 * s_cs137 + f_cs134 * s_cs134 + f_i131 * s_iod_tel

    elif scenario == "activation_orphan_co60":
        nuc = {"fraction_cs137": 0.0, "fraction_i131": 0.0, "fraction_co60": 1.0, "fraction_cs134": 0.0}
        s8 = _get_cached_nuclide_shares8("Co-60", family, fwhm, gain, standoff)

    elif scenario == "mixed_fission_activation":
        alpha_mix = [3.5, 2.5, 2.0, 2.0] if is_family_a else [3.0, 2.2, 2.6, 2.2]
        draws = rng.dirichlet(alpha_mix)
        f_cs137 = round(float(draws[0]), 4)
        f_i131 = round(float(draws[1]), 4)
        f_co60 = round(float(draws[2]), 4)
        f_cs134 = round(float(1.0 - f_cs137 - f_i131 - f_co60), 4)
        nuc = {
            "fraction_cs137": f_cs137,
            "fraction_i131": f_i131,
            "fraction_co60": f_co60,
            "fraction_cs134": f_cs134,
        }
        s8 = (
            f_cs137 * _get_cached_nuclide_shares8("Cs-137", family, fwhm, gain, standoff)
            + f_i131 * _get_cached_nuclide_shares8("I-131", family, fwhm, gain, standoff)
            + f_co60 * _get_cached_nuclide_shares8("Co-60", family, fwhm, gain, standoff)
            + f_cs134 * _get_cached_nuclide_shares8("Cs-134", family, fwhm, gain, standoff)
        )

    # Test-only hold-out radionuclides (Hold-out Realism B.2.ii)
    elif scenario == "fission_pure_ru103":
        nuc = {"fraction_cs137": 0.5, "fraction_i131": 0.5, "fraction_co60": 0.0, "fraction_cs134": 0.0}
        s8 = _get_cached_nuclide_shares8("Ru-103", family, fwhm, gain, standoff)

    elif scenario == "fission_zr95_nb95":
        nuc = {"fraction_cs137": 0.6, "fraction_i131": 0.4, "fraction_co60": 0.0, "fraction_cs134": 0.0}
        r_nb = float(rng.uniform(0.35, 0.65))
        s8 = (1.0 - r_nb) * _get_cached_nuclide_shares8("Zr-95", family, fwhm, gain, standoff) + r_nb * _get_cached_nuclide_shares8("Nb-95", family, fwhm, gain, standoff)

    elif scenario == "fission_ba140_la140":
        nuc = {"fraction_cs137": 0.2, "fraction_i131": 0.5, "fraction_co60": 0.3, "fraction_cs134": 0.0}
        r_la = float(rng.uniform(0.45, 0.65))
        s8 = (1.0 - r_la) * _get_cached_nuclide_shares8("Ba-140", family, fwhm, gain, standoff) + r_la * _get_cached_nuclide_shares8("La-140", family, fwhm, gain, standoff)

    else:
        raise ValueError(f"Unsupported scenario: {scenario!r}")

    # Small physical filter-thickness / self-absorption variation on low-energy R02/R03 (+/- 4%)
    atten_jitter = np.exp(-np.array([0.04, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.001]) * rng.normal(0.0, 1.0))
    s8 = np.maximum(1e-5, s8 * atten_jitter)
    s8 = s8 / np.sum(s8)
    return s8, nuc


def compute_independent_dose_response_vector(
    standoff_cm: float = 1.65,
    al_window_mm: float = 0.90,
) -> np.ndarray:
    """
    Computes an independent 8-channel dose conversion vector (nSv/h per CPM for R02..R09)
    derived purely from ICRP-74 h*(10, E) and the geometric + attenuation efficiency of
    the 2"x2" NaI(Tl) crystal, WITHOUT anchoring to `scalar_radon_slope` or `s_Rn`
    (resolving the dose circularity tested in Realism Audit D.3).
    """
    centroid_kev = np.array([150.0, 300.0, 500.0, 700.0, 900.0, 1200.0, 1600.0, 2000.0], dtype=float)
    icrp74_psv_cm2 = np.array([0.89, 1.80, 2.93, 3.92, 4.79, 5.85, 7.08, 8.15], dtype=float)

    t_al = al_window_transmission(centroid_kev, al_window_mm)
    _, _, mu_tot = nai_linear_attenuation_cm1(centroid_kev)
    # Effective path length in 5.08 cm crystal at oblique angles (independently parameterized)
    p_int = 1.0 - np.exp(-mu_tot * 3.80)
    det_eff = t_al * p_int

    # Convert pSv*cm^2 per photon and CPM (counts/min -> counts/h = 60) over effective crystal area
    # Area_crystal = pi * (2.54 cm)^2 = 20.268 cm^2; 1 pSv = 1e-3 nSv
    # Rate (nSv/h) per CPM = (60 / Area_crystal) * 1e-3 * h*(10) / det_eff
    area_cm2 = np.pi * (2.54**2)
    k_indep = (60.0 / area_cm2) * 1e-3 * (icrp74_psv_cm2 / np.maximum(0.05, det_eff))
    return k_indep


def compute_independent_injected_dose_rate(
    delta_channels: np.ndarray,
    rng: np.random.RandomState | None = None,
    calibration_jitter_sd: float = 0.08,
) -> np.ndarray:
    """
    Computes injected ambient dose equivalent rate (nSv/h) using the independent
    detector-geometry efficiency curve (not anchored to station radon regressions).
    """
    k_vec = compute_independent_dose_response_vector()
    if rng is not None and calibration_jitter_sd > 0.0:
        k_vec = k_vec * float(np.exp(rng.normal(0.0, calibration_jitter_sd)))
    arr = np.asarray(delta_channels, dtype=float)
    return np.tensordot(arr, k_vec, axes=([-1], [0]))


def apply_counting_noise(
    channel_excess_cpm: np.ndarray,
    rng: np.random.RandomState,
    integration_minutes: float = 60.0,
    include_overdispersion: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies realistic Poisson counting statistics and empirical RadNet channel
    overdispersion (Realism Audit D.2) to an array of 8-channel count excesses
    of shape (T, 8) in CPM.

    Returns (noisy_gross_cpm, noisy_channel_cpm).
    """
    arr = np.maximum(0.0, np.asarray(channel_excess_cpm, dtype=float))
    # Empirical channel Fano factors Var(Delta C_k) / mean_CPM from 2017-2020 dry RadNet data
    empirical_fano_cpm = np.array([0.402, 0.680, 0.223, 0.170, 0.061, 0.140, 0.061, 0.027], dtype=float)

    if include_overdispersion:
        # Gamma-Poisson mixture: sample latent rate with variance matching empirical Fano factor
        excess_fano = np.maximum(0.0, empirical_fano_cpm - (1.0 / integration_minutes))
        # Var(rate) = excess_fano * mean_cpm => Gamma shape k = mean_cpm / excess_fano, scale = excess_fano
        latent_cpm = np.zeros_like(arr)
        pos = arr > 1e-6
        if np.any(pos):
            fano_broad = np.broadcast_to(excess_fano, arr.shape)[pos]
            mean_pos = arr[pos]
            shape_k = np.maximum(0.5, mean_pos / np.maximum(1e-4, fano_broad))
            scale_theta = mean_pos / shape_k
            latent_cpm[pos] = rng.gamma(shape=shape_k, scale=scale_theta)
    else:
        latent_cpm = arr

    expected_counts = np.maximum(0.0, latent_cpm * float(integration_minutes))
    sampled_counts = rng.poisson(expected_counts)
    noisy_ch = sampled_counts.astype(float) / float(integration_minutes)
    noisy_gross = np.sum(noisy_ch, axis=-1)
    return noisy_gross, noisy_ch


def export_response_templates_csv(
    out_path: Path = Path("data/processed/response_templates.csv"),
    n_draws: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates and writes `data/processed/response_templates.csv` containing nominal and
    Monte-Carlo-sampled channel fraction vectors (R01..R09 and R02..R09) across both
    Family A (Analytic) and Family B (3D Monte Carlo) for all individual radionuclides
    and composite release scenarios, with full provenance metadata.
    """
    rng = np.random.RandomState(seed)
    rows: list[dict] = []

    # 1. Pure individual radionuclides (R01..R09 and R02..R09)
    for fam_code, fam_label in [("A", "A_analytic_approximation"), ("B", "B_monte_carlo_3d_transport")]:
        params = DetectorGeometryParams(
            standoff_cm=1.4 if fam_code == "A" else 1.8,
            filter_radius_cm=2.0 if fam_code == "A" else 2.15,
            al_window_mm=0.8 if fam_code == "A" else 1.0,
            fwhm_662_pct=7.5 if fam_code == "A" else 8.5,
            gain_shift_frac=0.0,
            backscatter_fraction=0.07 if fam_code == "A" else 0.10,
        )
        for nuc, lines in NUCLIDE_GAMMA_LINES.items():
            s9, s8, eff = compute_nuclide_template(
                nuclide=nuc, family=fam_code, params=params, n_histories_per_line=20000, seed=seed
            )
            primary_lines_str = "; ".join(f"{e:.1f} keV ({i*100:.1f}%)" for e, i in lines[:4])
            row = {
                "template_type": "pure_nuclide",
                "identifier": nuc,
                "family": fam_label,
                "total_det_eff_per_decay": round(eff, 6),
                "primary_gamma_lines_ensdf": primary_lines_str,
                "standoff_cm": params.standoff_cm,
                "filter_radius_cm": params.filter_radius_cm,
                "al_window_mm": params.al_window_mm,
                "fwhm_662_pct": params.fwhm_662_pct,
                "geometry_verification_status": GEOMETRY_STATUS,
                "epa_doc_url": EPA_RADNET_DOC_URL,
                "nuclear_data_url": NNDC_ENSDF_URL,
                "attenuation_data_url": NIST_XCOM_URL,
                "retrieval_date": RETRIEVAL_DATE,
            }
            for i in range(9):
                row[f"share9_r0{i+1}"] = round(float(s9[i]), 6)
            for i in range(8):
                row[f"share_r0{i+2}"] = round(float(s8[i]), 6)
                row[f"share_r0{i+2}_sd"] = 0.0
            rows.append(row)

    # 2. Scenario ensemble distributions (mean and SD across physical parameter draws)
    all_scenarios = [
        "fission_pure_cs137",
        "fission_pure_i131",
        "fission_reactor_fukushima",
        "mixed_fission_activation",
        "activation_orphan_co60",
        "fission_pure_ru103",
        "fission_zr95_nb95",
        "fission_ba140_la140",
    ]
    for fam_code, fam_label in [("A", "A_analytic_approximation"), ("B", "B_monte_carlo_3d_transport")]:
        for scen in all_scenarios:
            draws_8 = []
            for _ in range(n_draws):
                s8, _ = sample_derived_spectrum(rng, scen, family=fam_code)
                draws_8.append(s8)
            arr8 = np.asarray(draws_8)
            mean8 = np.mean(arr8, axis=0)
            mean8 = mean8 / np.sum(mean8)
            sd8 = np.std(arr8, axis=0)
            row = {
                "template_type": "scenario_ensemble",
                "identifier": scen,
                "family": fam_label,
                "total_det_eff_per_decay": np.nan,
                "primary_gamma_lines_ensdf": "Composite scenario from NNDC/ENSDF lines",
                "standoff_cm": 1.4 if fam_code == "A" else 1.8,
                "filter_radius_cm": 2.0 if fam_code == "A" else 2.15,
                "al_window_mm": 0.8 if fam_code == "A" else 1.0,
                "fwhm_662_pct": 7.8 if fam_code == "A" else 8.8,
                "geometry_verification_status": GEOMETRY_STATUS,
                "epa_doc_url": EPA_RADNET_DOC_URL,
                "nuclear_data_url": NNDC_ENSDF_URL,
                "attenuation_data_url": NIST_XCOM_URL,
                "retrieval_date": RETRIEVAL_DATE,
                "share9_r01": 0.0,
            }
            for i in range(8):
                row[f"share9_r0{i+2}"] = round(float(mean8[i]), 6)
                row[f"share_r0{i+2}"] = round(float(mean8[i]), 6)
                row[f"share_r0{i+2}_sd"] = round(float(sd8[i]), 6)
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    df_tpl = export_response_templates_csv()
    print(f"Wrote {len(df_tpl)} detector response templates to data/processed/response_templates.csv")
    scen_sub = df_tpl[df_tpl["template_type"] == "scenario_ensemble"]
    cols = ["identifier", "family"] + [f"share_r0{i}" for i in range(2, 10)]
    print(scen_sub[cols].to_string(index=False))
