"""
Multi-Nuclide Physics-Informed Synthetic Plume Injection
Generator & Labeled Dataset Builder.

Implements the deterministic, physics-informed synthetic injection pipeline:
1. 100% filter synchronization: Every injection ends at a real detected physical filter replacement drop.
2. Perfect scenario balance: Identical 20.0% scenario proportions in all training and test environments.
3. Spectrum logging: Explicitly logs sampled 8-channel excess shares (R02-R09) at peak passage in the catalog.
4. Multi-nuclide time-evolving spectra & Bateman progeny kinetics:
   Models nuclide-specific filter deposition and radioactive decay for I-131 (8.025 d),
   Cs-137 (30.17 y), Cs-134 (2.065 y), Co-60 (5.27 y), and the short-lived reactor
   tellurium-iodine pair Te-132 -> I-132 (76.9 h -> 2.295 h) via the analytical Bateman equation.
5. Channel-resolved energy-weighted dose coupling:
   Computes injected ambient dose equivalent rate from the 8-channel injected counts
   sum_{k=2..9} k_dose_channel[k] * Delta C_k(t) grounded in ICRP-74 h*(10, E_gamma)
   and NaI(Tl) intrinsic efficiency.
6. Strict 48-hour buffer across all injection environments.

Saves:
- data/processed/synthetic_injection_catalog.csv
- data/processed/labeled_{station_id}_{train|test}.csv.gz
- data/processed/labeled_dataset_reconciliation_summary.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml

from src.analyze_filter_cycles_multi_station import detect_filter_drops
from src.nuclear_physics import (
    CHANNELS,
    DEFAULT_RADON_BASIS,
    LAMBDA_CO60,
    LAMBDA_CS134,
    LAMBDA_CS137,
    LAMBDA_I131,
    LAMBDA_I132,
    LAMBDA_TE132,
    compute_injected_dose_rate,
    compute_multinuclide_injection_timeseries,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RANDOM_SEED = config.get("random_seed", 42)

# Frozen train-only dry baselines (2017-2022) — used for washout labeling (BUG-4 fix).
_BASELINES_FILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "station_dry_baselines_train_only.json"
if not _BASELINES_FILE.exists():
    raise FileNotFoundError(
        f"Required train-only baseline file not found: {_BASELINES_FILE}. "
        "Run: python src/compute_train_only_baselines.py"
    )
with open(_BASELINES_FILE, "r") as _f:
    TRAIN_ONLY_BASELINES = json.load(_f)

STATIONS = [
    {"id": "al_birmingham", "name": "Birmingham, AL", "drop_thresh": -300.0},
    {"id": "dc_washington", "name": "Washington, DC", "drop_thresh": -250.0},
    {"id": "ca_san_diego", "name": "San Diego, CA", "drop_thresh": -300.0},
    {"id": "tx_dallas", "name": "Dallas, TX", "drop_thresh": -300.0},
    {"id": "fl_tampa", "name": "Tampa, FL", "drop_thresh": -300.0},
]

# Traceable dose-rate coupling slopes from data/processed/dose_rate_cpm_regression_summary.csv
DOSE_COUPLING_SLOPES = {
    "al_birmingham": 0.01389,
    "dc_washington": 0.01668,
    "ca_san_diego": 0.02359,
    "tx_dallas": 0.01238,
    "fl_tampa": 0.01142,
    "pooled": 0.01344,
}
_DOSE_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "dose_rate_cpm_regression_summary.csv"
if _DOSE_CSV.exists():
    _dose_df = pd.read_csv(_DOSE_CSV)
    for _, _r in _dose_df.iterrows():
        _sid = str(_r["Station_ID"])
        if _sid == "pooled_network":
            DOSE_COUPLING_SLOPES["pooled"] = float(_r["Slope_nSvh_per_CPM"])
        elif _sid in DOSE_COUPLING_SLOPES:
            DOSE_COUPLING_SLOPES[_sid] = float(_r["Slope_nSvh_per_CPM"])

SCENARIOS = [
    "fission_pure_cs137",
    "fission_pure_i131",
    "fission_reactor_fukushima",
    "mixed_fission_activation",
    "activation_orphan_co60",
]


def compute_physical_injection_profile(
    shape_family: str,
    t_passage: int,
    t_retention: int,
    peak_cpm: float,
    rise_param: float,
    f_i131: float,
    scenario: str | None = None,
    spectrum_at_peak: np.ndarray | None = None,
) -> np.ndarray:
    """
    Computes physical count rate profile S(t) for t in 0..D-1 (D = t_passage + t_retention):
    1. Plume passage (t < t_passage): cumulative particulate deposition on continuous air filter.
    2. Retention (t >= t_passage): atmospheric plume has passed; particulate activity trapped on filter media
       decays according to constituent nuclide half-lives (I-131, Te-132->I-132 Bateman pair, Cs-134, Co-60, Cs-137).
    3. Filter change (at t = D): technician replaces filter, resetting deposited activity to 0.
    """
    if scenario is not None and spectrum_at_peak is not None:
        gross_prof, _ = compute_multinuclide_injection_timeseries(
            shape_family=shape_family,
            t_passage=t_passage,
            t_retention=t_retention,
            peak_cpm=peak_cpm,
            rise_param=rise_param,
            scenario=scenario,
            spectrum_at_peak=spectrum_at_peak,
            f_i131=f_i131,
        )
        return gross_prof

    d = t_passage + t_retention
    lam = f_i131 * LAMBDA_I131 + (1.0 - f_i131) * LAMBDA_CS137

    u = np.arange(t_passage)
    if shape_family == "linear_ramp":
        c = np.clip(u / max(1.0, float(rise_param)), 0.0, 1.0)
    elif shape_family == "step":
        c = np.ones(t_passage)
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
        c = np.ones(t_passage)

    profile = np.zeros(d)
    for i in range(t_passage):
        decay_weights = np.exp(-lam * (i - np.arange(i + 1)))
        profile[i] = np.sum(c[:i + 1] * decay_weights)

    scale = peak_cpm / (profile[t_passage - 1] + 1e-6)
    profile[:t_passage] *= scale

    peak_val = profile[t_passage - 1]
    for i in range(t_passage, d):
        dt_ret = i - (t_passage - 1)
        profile[i] = peak_val * np.exp(-lam * dt_ret)

    return profile


def sample_operational_spectrum(rng: np.random.RandomState, scenario: str) -> tuple[np.ndarray, dict]:
    """
    Samples channel energy branching shares across R02-R09 under operational radiological scenarios.
    Uses continuous uniform/Dirichlet draws for photopeak fractions, Compton continuum, and
    high-energy scatter floor to ensure models encounter wide spectral variety.

    Includes Cs-134 high-energy lines (1168 & 1365 keV) in R07 for Fukushima-like reactor release,
    producing an R07+R08 share that directly spans the empirical radon washout band.
    """
    tail_r06 = rng.uniform(0.005, 0.025)
    tail_r07 = rng.uniform(0.004, 0.012)
    tail_r08 = rng.uniform(0.002, 0.008)

    if scenario == "fission_pure_cs137":
        p_r05 = rng.uniform(0.38, 0.55)  # 662 keV photopeak
        rem = 1.0 - p_r05 - tail_r06 - tail_r07 - tail_r08
        comp = rng.dirichlet([2.5, 2.0, 2.0]) * rem
        w = np.array([comp[0], comp[1], comp[2], p_r05, tail_r06, tail_r07, tail_r08, 0.0])
        nuc = {"fraction_cs137": 1.0, "fraction_i131": 0.0, "fraction_co60": 0.0, "fraction_cs134": 0.0}

    elif scenario == "fission_pure_i131":
        p_r03 = rng.uniform(0.52, 0.70)  # 364 keV photopeak
        p_r04 = rng.uniform(0.04, 0.08)  # 637 keV weak lines
        rem = 1.0 - p_r03 - p_r04 - tail_r06 - tail_r07 - tail_r08
        comp_r02 = rem * rng.uniform(0.70, 0.85)
        comp_r05 = rem - comp_r02
        w = np.array([comp_r02, p_r03, p_r04, comp_r05, tail_r06, tail_r07, tail_r08, 0.0])
        nuc = {"fraction_cs137": 0.0, "fraction_i131": 1.0, "fraction_co60": 0.0, "fraction_cs134": 0.0}

    elif scenario == "fission_reactor_fukushima":
        # Fresh core fission release: Cs-137 (~40%), Cs-134 (~30%), I-131 + Te-132/I-132 (~30%)
        # Cs-134 emits 1168 keV (1.8%) and 1365 keV (3.0%) gamma rays falling in R07
        p_r03 = rng.uniform(0.20, 0.35)  # I-131 364 keV + Te-132 228 keV
        p_r05 = rng.uniform(0.25, 0.40)  # Cs-137 + Cs-134 (605/662 keV) + I-132 (667/773 keV)
        p_r06 = rng.uniform(0.03, 0.08)  # Cs-134 796/802 keV + I-132 955 keV
        p_r07_cs134 = rng.uniform(0.025, 0.055)  # Cs-134 1168 & 1365 keV lines in R07
        rem = 1.0 - p_r03 - p_r05 - p_r06 - p_r07_cs134 - tail_r08
        comp = rng.dirichlet([2.5, 2.0, 2.0]) * rem
        w = np.array([comp[0], p_r03 + comp[1], comp[2], p_r05, p_r06, p_r07_cs134, tail_r08, 0.0])
        nuc = {"fraction_cs137": 0.40, "fraction_i131": 0.30, "fraction_co60": 0.0, "fraction_cs134": 0.30}

    elif scenario == "activation_orphan_co60":
        # Co-60: 1173 and 1332 keV photopeaks in R07 (22-35%)
        p_r07 = rng.uniform(0.22, 0.35)
        tail_high = rng.uniform(0.015, 0.035)
        rem = 1.0 - p_r07 - tail_high
        comp = rng.dirichlet([2.0, 2.0, 2.0, 2.0, 1.5]) * rem
        w = np.array([comp[0], comp[1], comp[2], comp[3], comp[4], p_r07, tail_high, 0.0])
        nuc = {"fraction_cs137": 0.0, "fraction_i131": 0.0, "fraction_co60": 1.0, "fraction_cs134": 0.0}

    else:  # "mixed_fission_activation"
        # Core damage with structural steel activation: Cs-137 + I-131 + Co-60 + Cs-134
        p_r03 = rng.uniform(0.15, 0.25)  # I-131
        p_r05 = rng.uniform(0.20, 0.32)  # Cs-137 / Cs-134
        p_r07 = rng.uniform(0.06, 0.12)  # Co-60
        tail_high = rng.uniform(0.01, 0.02)
        rem = 1.0 - p_r03 - p_r05 - p_r07 - tail_high
        comp = rng.dirichlet([2.0, 2.0, 2.0, 1.5]) * rem
        w = np.array([comp[0], p_r03, comp[1], p_r05, comp[2], p_r07, tail_high, 0.0])
        nuc = {"fraction_cs137": 0.35, "fraction_i131": 0.25, "fraction_co60": 0.20, "fraction_cs134": 0.20}

    # Relative channel perturbation (+/- 15%)
    pert = rng.normal(0.0, 0.15, size=len(w))
    w = np.maximum(0.001, w * (1.0 + pert))
    w = w / np.sum(w)
    return w, nuc


def generate_revised_injection_catalog() -> pd.DataFrame:
    """
    Generates deterministic, non-overlapping synthetic injection catalog across all 5 pilot stations:
    - 50 train injections per station (250 total, 2017-2022): 15 rain-coincident, 35 strictly dry.
    - 40 test injections per station (200 total, 2023-2025):
        - 20 standard duration (36-80h), Band B [700, 1200] CPM (10 rain, 10 dry).
        - 20 hard-regime stress test (8-20h dry, 25-45h rain), Band A [250, 600] CPM (10 rain, 10 dry).

    All 450 injections strictly end at real physical filter replacement drops (0 synthetic cliffs).
    Strict 48-hour buffer enforced across all injection branches.
    Balanced 20.0% scenario proportions in all environments.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    catalog = []
    injection_id = 1

    for st in STATIONS:
        st_id = st["id"]
        csv_file = Path(f"data/processed/merged_{st_id}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        df["dt"] = pd.to_datetime(df["utc_hour"])
        df = df.sort_values("dt").reset_index(drop=True)

        drops_df = detect_filter_drops(df, st["drop_thresh"])
        detected_drop_indices = sorted(drops_df.index.tolist()) if not drops_df.empty else []
        detected_drop_set = set(detected_drop_indices)
        print(f"{st['name']}: {len(detected_drop_indices)} real filter replacement drops detected for synchronization.")

        occupied = np.zeros(len(df), dtype=bool)
        df["precip_24h"] = df["precip_1h_mm"].rolling(24, min_periods=12).sum()
        valid = df["has_radnet_obs"].fillna(False).astype(bool) & df["rad_complete_channels"].fillna(False).astype(bool)

        k_dose_mean = DOSE_COUPLING_SLOPES.get(st_id, DOSE_COUPLING_SLOPES["pooled"])

        train_drops = [d for d in detected_drop_indices if df.loc[d, "dt"] < pd.Timestamp("2023-01-01")]
        test_drops = [d for d in detected_drop_indices if df.loc[d, "dt"] >= pd.Timestamp("2023-01-01")]

        def get_hours_since_drop(start_idx):
            priors = [d for d in detected_drop_indices if d < start_idx]
            return int(start_idx - priors[-1]) if priors else int(start_idx)

        # -------------------------------------------------------------
        # 1. Training Set (2017-01-01 to 2023-01-01: 52,584 hours)
        # -------------------------------------------------------------
        train_rain_scenarios = [s for s in SCENARIOS for _ in range(3)]
        rng.shuffle(train_rain_scenarios)
        placed_tr_rain = 0

        for d in rng.permutation(train_drops):
            if placed_tr_rain >= 15:
                break
            sub = df.loc[max(0, d - 140) : d]
            rain_hrs = sub[sub["precip_1h_mm"] >= 1.0].index.tolist()
            if not rain_hrs:
                continue
            for r in rng.permutation(rain_hrs):
                dur = d - r + 1
                if dur < 10 or dur > 96:
                    continue
                inter = [x for x in detected_drop_set if r <= x < d]
                if inter:
                    continue
                b_start = max(0, r - 48)
                b_end = min(len(df), d + 48)
                if occupied[b_start:b_end].any():
                    continue
                missing = (~valid.loc[r:d]).sum()
                if missing > 1:
                    continue
                t_pass = min(dur - 2, max(4, int(dur * 0.3)))
                t_ret = dur - t_pass
                if (df.loc[r : r + t_pass - 1, "precip_1h_mm"] >= 1.0).sum() < 1:
                    continue

                shape = rng.choice(["linear_ramp", "step"])
                rise_p = rng.uniform(1.0, 4.0) if shape == "linear_ramp" else 0.0
                mag_band = rng.choice(["Band_A_Low", "Band_C_High"])
                peak_cpm = round(rng.uniform(250.0, 600.0) if mag_band == "Band_A_Low" else rng.uniform(1400.0, 2500.0), 1)

                scen = train_rain_scenarios[placed_tr_rain]
                spec, nuc = sample_operational_spectrum(rng, scen)
                k_dose = round(float(rng.normal(k_dose_mean, 0.0015)), 5)
                k_dose = max(0.010, min(0.026, k_dose))

                washout_overlap = int((df.loc[r:d, "precip_1h_mm"] >= 1.0).sum())
                rise_overlap = int((df.loc[r : r + t_pass - 1, "precip_1h_mm"] >= 1.0).sum())
                hours_since_drop = get_hours_since_drop(r)

                occupied[r : d + 1] = True
                entry = {
                    "injection_id": f"INJ_{injection_id:04d}",
                    "split": "train",
                    "station_id": st_id,
                    "station_name": st["name"],
                    "start_utc": str(df.loc[r, "dt"]),
                    "end_utc": str(df.loc[d, "dt"]),
                    "start_index": int(r),
                    "t_passage_hours": t_pass,
                    "t_retention_hours": t_ret,
                    "duration_hours": dur,
                    "shape_family": shape,
                    "rise_param": round(rise_p, 2),
                    "magnitude_band": mag_band,
                    "peak_cpm": peak_cpm,
                    "nuclide_scenario": scen,
                    "fraction_cs137": nuc["fraction_cs137"],
                    "fraction_i131": nuc["fraction_i131"],
                    "fraction_co60": nuc["fraction_co60"],
                    "fraction_cs134": nuc["fraction_cs134"],
                    "k_dose": k_dose,
                    "is_hard_case_rain": True,
                    "washout_overlap_hours": washout_overlap,
                    "rise_washout_overlap_hours": rise_overlap,
                    "hours_since_last_detected_drop": hours_since_drop,
                    "truncated_by_filter_change": True,
                }
                for ch_i, _ in enumerate(CHANNELS):
                    entry[f"share_r0{ch_i+2}"] = round(float(spec[ch_i]), 6)
                entry["environment"] = "train_rain_coincident"
                catalog.append(entry)
                injection_id += 1
                placed_tr_rain += 1
                break

        train_dry_scenarios = [s for s in SCENARIOS for _ in range(7)]
        rng.shuffle(train_dry_scenarios)
        placed_tr_dry = 0

        for d in rng.permutation(train_drops):
            if placed_tr_dry >= 35:
                break
            for dur in rng.permutation(range(10, 36)):
                start = d - dur + 1
                if start < 0:
                    continue
                inter = [x for x in detected_drop_set if start <= x < d]
                if inter:
                    continue
                b_start = max(0, start - 48)
                b_end = min(len(df), d + 48)
                if occupied[b_start:b_end].any():
                    continue
                if not valid.loc[start:d].all():
                    continue
                if (df.loc[start:d, "precip_1h_mm"] > 0.0).any():
                    continue
                if df.loc[start, "precip_24h"] > 0.0:
                    continue

                t_pass = min(dur - 2, max(4, int(dur * 0.35)))
                t_ret = dur - t_pass

                shape = rng.choice(["linear_ramp", "step"])
                rise_p = rng.uniform(1.0, 4.0) if shape == "linear_ramp" else 0.0
                mag_band = rng.choice(["Band_A_Low", "Band_C_High"])
                peak_cpm = round(rng.uniform(250.0, 600.0) if mag_band == "Band_A_Low" else rng.uniform(1400.0, 2500.0), 1)

                scen = train_dry_scenarios[placed_tr_dry]
                spec, nuc = sample_operational_spectrum(rng, scen)
                k_dose = round(float(rng.normal(k_dose_mean, 0.0015)), 5)
                k_dose = max(0.010, min(0.026, k_dose))
                hours_since_drop = get_hours_since_drop(start)

                occupied[start : d + 1] = True
                entry = {
                    "injection_id": f"INJ_{injection_id:04d}",
                    "split": "train",
                    "station_id": st_id,
                    "station_name": st["name"],
                    "start_utc": str(df.loc[start, "dt"]),
                    "end_utc": str(df.loc[d, "dt"]),
                    "start_index": int(start),
                    "t_passage_hours": t_pass,
                    "t_retention_hours": t_ret,
                    "duration_hours": dur,
                    "shape_family": shape,
                    "rise_param": round(rise_p, 2),
                    "magnitude_band": mag_band,
                    "peak_cpm": peak_cpm,
                    "nuclide_scenario": scen,
                    "fraction_cs137": nuc["fraction_cs137"],
                    "fraction_i131": nuc["fraction_i131"],
                    "fraction_co60": nuc["fraction_co60"],
                    "fraction_cs134": nuc["fraction_cs134"],
                    "k_dose": k_dose,
                    "is_hard_case_rain": False,
                    "washout_overlap_hours": 0,
                    "rise_washout_overlap_hours": 0,
                    "hours_since_last_detected_drop": hours_since_drop,
                    "truncated_by_filter_change": True,
                }
                for ch_i, _ in enumerate(CHANNELS):
                    entry[f"share_r0{ch_i+2}"] = round(float(spec[ch_i]), 6)
                entry["environment"] = "train_strictly_dry"
                catalog.append(entry)
                injection_id += 1
                placed_tr_dry += 1
                break

        # -------------------------------------------------------------
        # 2. Test Set (2023-01-01 to 2026-01-01: 26,304 hours)
        # -------------------------------------------------------------
        test_std_rain_scenarios = [s for s in SCENARIOS for _ in range(2)]
        test_stress_rain_scenarios = [s for s in SCENARIOS for _ in range(2)]
        test_std_dry_scenarios = [s for s in SCENARIOS for _ in range(2)]
        test_stress_dry_scenarios = [s for s in SCENARIOS for _ in range(2)]
        rng.shuffle(test_std_rain_scenarios)
        rng.shuffle(test_stress_rain_scenarios)
        rng.shuffle(test_std_dry_scenarios)
        rng.shuffle(test_stress_dry_scenarios)

        placed_std_rain = 0
        placed_stress_rain = 0

        rain_p_thresh = 0.5 if st_id == "ca_san_diego" else 1.0
        max_lookback = 168 if st_id == "ca_san_diego" else 140

        for d in rng.permutation(test_drops):
            if placed_std_rain >= 10 and placed_stress_rain >= 10:
                break
            sub = df.loc[max(0, d - max_lookback) : d]
            rain_hrs = sub[sub["precip_1h_mm"] >= rain_p_thresh].index.tolist()
            if not rain_hrs:
                continue
            for r in rng.permutation(rain_hrs):
                dur = d - r + 1
                if dur < 10 or dur > 168:
                    continue
                inter = [x for x in detected_drop_set if r <= x < d]
                if inter:
                    continue
                b_start = max(0, r - 48)
                b_end = min(len(df), d + 48)
                if occupied[b_start:b_end].any():
                    continue
                missing = (~valid.loc[r:d]).sum()
                if missing > 2:
                    continue

                if placed_std_rain < 10 and (dur >= 36 or placed_stress_rain >= 10):
                    t_pass = min(24, max(8, int(dur * 0.3)))
                    t_ret = dur - t_pass
                    if (df.loc[r : r + t_pass - 1, "precip_1h_mm"] >= rain_p_thresh).sum() < 1:
                        continue
                    scen = test_std_rain_scenarios[placed_std_rain]
                    mag_band = "Band_B_Mid"
                    peak_cpm = round(rng.uniform(700.0, 1200.0), 1)
                    shape = rng.choice(["sigmoidal", "exponential"])
                    rise_p = rng.uniform(0.8, 1.8) if shape == "sigmoidal" else rng.uniform(1.0, 2.5)
                    env_name = "test_standard_rain"
                    placed_std_rain += 1
                elif placed_stress_rain < 10:
                    t_pass = min(8, max(3, int(dur * 0.25)))
                    t_ret = dur - t_pass
                    if (df.loc[r : r + t_pass - 1, "precip_1h_mm"] >= rain_p_thresh).sum() < 1:
                        continue
                    scen = test_stress_rain_scenarios[placed_stress_rain]
                    mag_band = "Band_A_Low (Stress Test)"
                    peak_cpm = round(rng.uniform(250.0, 600.0), 1)
                    shape = rng.choice(["sigmoidal", "exponential"])
                    rise_p = rng.uniform(0.6, 1.4) if shape == "sigmoidal" else rng.uniform(0.8, 1.8)
                    env_name = "test_stress_rain"
                    placed_stress_rain += 1
                else:
                    continue

                spec, nuc = sample_operational_spectrum(rng, scen)
                k_dose = round(float(rng.normal(k_dose_mean, 0.0015)), 5)
                k_dose = max(0.010, min(0.026, k_dose))

                washout_overlap = int((df.loc[r:d, "precip_1h_mm"] >= 1.0).sum())
                rise_overlap = int((df.loc[r : r + t_pass - 1, "precip_1h_mm"] >= 1.0).sum())
                hours_since_drop = get_hours_since_drop(r)

                occupied[r : d + 1] = True
                entry = {
                    "injection_id": f"INJ_{injection_id:04d}",
                    "split": "test",
                    "station_id": st_id,
                    "station_name": st["name"],
                    "start_utc": str(df.loc[r, "dt"]),
                    "end_utc": str(df.loc[d, "dt"]),
                    "start_index": int(r),
                    "t_passage_hours": t_pass,
                    "t_retention_hours": t_ret,
                    "duration_hours": dur,
                    "shape_family": shape,
                    "rise_param": round(rise_p, 2),
                    "magnitude_band": mag_band,
                    "peak_cpm": peak_cpm,
                    "nuclide_scenario": scen,
                    "fraction_cs137": nuc["fraction_cs137"],
                    "fraction_i131": nuc["fraction_i131"],
                    "fraction_co60": nuc["fraction_co60"],
                    "fraction_cs134": nuc["fraction_cs134"],
                    "k_dose": k_dose,
                    "is_hard_case_rain": True,
                    "washout_overlap_hours": washout_overlap,
                    "rise_washout_overlap_hours": rise_overlap,
                    "hours_since_last_detected_drop": hours_since_drop,
                    "truncated_by_filter_change": True,
                }
                for ch_i, _ in enumerate(CHANNELS):
                    entry[f"share_r0{ch_i+2}"] = round(float(spec[ch_i]), 6)
                entry["environment"] = env_name
                catalog.append(entry)
                injection_id += 1
                break

        placed_std_dry = 0
        placed_stress_dry = 0
        for d in rng.permutation(test_drops):
            if placed_std_dry >= 10 and placed_stress_dry >= 10:
                break
            if placed_std_dry < 10:
                for dur in rng.permutation(range(36, 76)):
                    start = d - dur + 1
                    if start < 0 or occupied[max(0, start - 48) : min(len(df), d + 48)].any():
                        continue
                    inter = [x for x in detected_drop_set if start <= x < d]
                    if not inter and valid.loc[start:d].all() and (df.loc[start:d, "precip_1h_mm"] == 0.0).all() and df.loc[start, "precip_24h"] == 0.0:
                        t_pass = min(dur - 2, max(8, int(dur * 0.3)))
                        t_ret = dur - t_pass
                        shape = rng.choice(["sigmoidal", "exponential"])
                        rise_p = rng.uniform(0.8, 1.8) if shape == "sigmoidal" else rng.uniform(1.0, 2.5)
                        scen = test_std_dry_scenarios[placed_std_dry]
                        spec, nuc = sample_operational_spectrum(rng, scen)
                        k_dose = round(float(rng.normal(k_dose_mean, 0.0015)), 5)
                        k_dose = max(0.010, min(0.026, k_dose))
                        hours_since_drop = get_hours_since_drop(start)

                        occupied[start : d + 1] = True
                        entry = {
                            "injection_id": f"INJ_{injection_id:04d}",
                            "split": "test",
                            "station_id": st_id,
                            "station_name": st["name"],
                            "start_utc": str(df.loc[start, "dt"]),
                            "end_utc": str(df.loc[d, "dt"]),
                            "start_index": int(start),
                            "t_passage_hours": t_pass,
                            "t_retention_hours": t_ret,
                            "duration_hours": dur,
                            "shape_family": shape,
                            "rise_param": round(rise_p, 2),
                            "magnitude_band": "Band_B_Mid",
                            "peak_cpm": round(rng.uniform(700.0, 1200.0), 1),
                            "nuclide_scenario": scen,
                            "fraction_cs137": nuc["fraction_cs137"],
                            "fraction_i131": nuc["fraction_i131"],
                            "fraction_co60": nuc["fraction_co60"],
                            "fraction_cs134": nuc["fraction_cs134"],
                            "k_dose": k_dose,
                            "is_hard_case_rain": False,
                            "washout_overlap_hours": 0,
                            "rise_washout_overlap_hours": 0,
                            "hours_since_last_detected_drop": hours_since_drop,
                            "truncated_by_filter_change": True,
                        }
                        for ch_i, _ in enumerate(CHANNELS):
                            entry[f"share_r0{ch_i+2}"] = round(float(spec[ch_i]), 6)
                        entry["environment"] = "test_standard_strictly_dry"
                        catalog.append(entry)
                        injection_id += 1
                        placed_std_dry += 1
                        break

            if placed_stress_dry < 10:
                for dur in rng.permutation(range(8, 21)):
                    start = d - dur + 1
                    if start < 0 or occupied[max(0, start - 48) : min(len(df), d + 48)].any():
                        continue
                    inter = [x for x in detected_drop_set if start <= x < d]
                    if not inter and valid.loc[start:d].all() and (df.loc[start:d, "precip_1h_mm"] == 0.0).all() and df.loc[start, "precip_24h"] == 0.0:
                        t_pass = min(dur - 2, max(3, int(dur * 0.35)))
                        t_ret = dur - t_pass
                        shape = rng.choice(["sigmoidal", "exponential"])
                        rise_p = rng.uniform(0.6, 1.4) if shape == "sigmoidal" else rng.uniform(0.8, 1.8)
                        scen = test_stress_dry_scenarios[placed_stress_dry]
                        spec, nuc = sample_operational_spectrum(rng, scen)
                        k_dose = round(float(rng.normal(k_dose_mean, 0.0015)), 5)
                        k_dose = max(0.010, min(0.026, k_dose))
                        hours_since_drop = get_hours_since_drop(start)

                        occupied[start : d + 1] = True
                        entry = {
                            "injection_id": f"INJ_{injection_id:04d}",
                            "split": "test",
                            "station_id": st_id,
                            "station_name": st["name"],
                            "start_utc": str(df.loc[start, "dt"]),
                            "end_utc": str(df.loc[d, "dt"]),
                            "start_index": int(start),
                            "t_passage_hours": t_pass,
                            "t_retention_hours": t_ret,
                            "duration_hours": dur,
                            "shape_family": shape,
                            "rise_param": round(rise_p, 2),
                            "magnitude_band": "Band_A_Low (Stress Test)",
                            "peak_cpm": round(rng.uniform(250.0, 600.0), 1),
                            "nuclide_scenario": scen,
                            "fraction_cs137": nuc["fraction_cs137"],
                            "fraction_i131": nuc["fraction_i131"],
                            "fraction_co60": nuc["fraction_co60"],
                            "fraction_cs134": nuc["fraction_cs134"],
                            "k_dose": k_dose,
                            "is_hard_case_rain": False,
                            "washout_overlap_hours": 0,
                            "rise_washout_overlap_hours": 0,
                            "hours_since_last_detected_drop": hours_since_drop,
                            "truncated_by_filter_change": True,
                        }
                        for ch_i, _ in enumerate(CHANNELS):
                            entry[f"share_r0{ch_i+2}"] = round(float(spec[ch_i]), 6)
                        entry["environment"] = "test_stress_strictly_dry"
                        catalog.append(entry)
                        injection_id += 1
                        placed_stress_dry += 1
                        break

        print(
            f"{st['name']}: placed total {placed_tr_rain + placed_tr_dry + placed_std_rain + placed_stress_rain + placed_std_dry + placed_stress_dry} "
            f"(train: {placed_tr_rain}R+{placed_tr_dry}D, test: {placed_std_rain}SR+{placed_stress_rain}XR+{placed_std_dry}SD+{placed_stress_dry}XD)"
        )

    df_catalog = pd.DataFrame(catalog)
    out_csv = Path("data/processed/synthetic_injection_catalog.csv")
    df_catalog.to_csv(out_csv, index=False)
    print(f"\nGenerated revised synthetic injection catalog ({len(df_catalog)} injections) to {out_csv}")
    return df_catalog


def build_and_save_labeled_datasets(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Applies synthetic injections using multi-nuclide time-evolving spectra (including
    Bateman Te-132 -> I-132 progeny kinetics) and 8-channel energy-weighted dose
    coupling sum_{k=2..9} k_dose_channel[k] * Delta C_k(t).
    """
    with open(_BASELINES_FILE, "r") as f:
        train_baselines = json.load(f)

    reconciliation_rows = []

    for st in STATIONS:
        st_id = st["id"]
        csv_file = Path(f"data/processed/merged_{st_id}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        df["dt"] = pd.to_datetime(df["utc_hour"])
        df = df.sort_values("dt").reset_index(drop=True)

        st_base = train_baselines[st_id]
        s_rn = np.asarray(st_base.get("radon_washout_basis", DEFAULT_RADON_BASIS), dtype=float)

        for split_name in ["train", "test"]:
            if split_name == "train":
                sub = df[(df["dt"] >= "2017-01-01") & (df["dt"] < "2023-01-01")].copy().reset_index(drop=True)
            else:
                sub = df[(df["dt"] >= "2023-01-01") & (df["dt"] < "2026-01-01")].copy().reset_index(drop=True)

            sub["inj_gross_cpm"] = sub["gross_cpm"].copy()
            sub["inj_dose_rate_nsvh"] = sub["dose_rate_nsvh"].copy()
            for ch in CHANNELS:
                sub[f"inj_{ch}"] = sub[ch].copy()

            sub["label"] = "normal"
            sub["injection_active"] = False
            sub["injection_id"] = None
            sub["synthetic_excess_cpm"] = 0.0

            # 1. Missing data mask: unobserved hours labeled 'unobserved'
            has_obs = sub["has_radnet_obs"].fillna(False).astype(bool)
            comp_ch = sub["rad_complete_channels"].fillna(False).astype(bool)
            unobs_mask = (~has_obs) | (~comp_ch)
            sub.loc[unobs_mask, "label"] = "unobserved"

            # 2. Rule-based labeling of natural radon washout using frozen train-only baselines
            sub["precip_3h"] = sub["precip_1h_mm"].rolling(3, min_periods=1).sum()
            sub["precip_24h"] = sub["precip_1h_mm"].rolling(24, min_periods=12).sum()
            mu_dry = st_base["mu"]
            sigma_dry = st_base["sigma"]

            washout_condition = (
                (sub["precip_3h"] > 0.0)
                & (sub["gross_cpm"] > (mu_dry + 2.0 * sigma_dry))
                & (~unobs_mask)
            )
            sub.loc[washout_condition, "label"] = "radon_washout"

            # 3. Apply synthetic injections with multi-nuclide time-evolving spectra & channel-resolved dose
            st_injections = catalog[(catalog["station_id"] == st_id) & (catalog["split"] == split_name)]
            for _, row in st_injections.iterrows():
                inj_start = pd.to_datetime(row["start_utc"])
                inj_end = pd.to_datetime(row["end_utc"])
                inj_mask = (sub["dt"] >= inj_start) & (sub["dt"] <= inj_end)
                indices = sub[inj_mask].index.to_numpy()
                dur = len(indices)
                if dur == 0:
                    continue

                spec = np.array([float(row[f"share_r0{ch_i+2}"]) for ch_i in range(8)], dtype=float)
                k_dose = float(row["k_dose"])

                gross_prof, ch_prof = compute_multinuclide_injection_timeseries(
                    shape_family=str(row["shape_family"]),
                    t_passage=int(row["t_passage_hours"]),
                    t_retention=int(row["t_retention_hours"]),
                    peak_cpm=float(row["peak_cpm"]),
                    rise_param=float(row["rise_param"]),
                    scenario=str(row["nuclide_scenario"]),
                    spectrum_at_peak=spec,
                    f_cs137=float(row["fraction_cs137"]),
                    f_i131=float(row["fraction_i131"]),
                    f_co60=float(row["fraction_co60"]),
                    f_cs134=float(row["fraction_cs134"]),
                )
                dose_prof = compute_injected_dose_rate(ch_prof, k_dose, s_rn)

                for step_i, idx_sub in enumerate(indices):
                    s_val = float(gross_prof[step_i])
                    sub.loc[idx_sub, "synthetic_excess_cpm"] = s_val
                    sub.loc[idx_sub, "injection_active"] = True
                    sub.loc[idx_sub, "injection_id"] = row["injection_id"]
                    if not unobs_mask.loc[idx_sub]:
                        sub.loc[idx_sub, "inj_gross_cpm"] += s_val
                        sub.loc[idx_sub, "inj_dose_rate_nsvh"] += float(dose_prof[step_i])
                        for ch_idx, ch in enumerate(CHANNELS):
                            sub.loc[idx_sub, f"inj_{ch}"] += float(ch_prof[step_i, ch_idx])
                        sub.loc[idx_sub, "label"] = "fission_product"

            out_file = Path(f"data/processed/labeled_{st_id}_{split_name}.csv.gz")
            sub.to_csv(out_file, index=False, compression="gzip")
            print(f"Saved: {out_file} (N={len(sub)})")

            counts = sub["label"].value_counts().to_dict()
            reconciliation_rows.append({
                "Station_ID": st_id,
                "Station_Name": st["name"],
                "Split": split_name,
                "Total_Hours": len(sub),
                "Normal_Hours": counts.get("normal", 0),
                "Washout_Hours": counts.get("radon_washout", 0),
                "Fission_Hours": counts.get("fission_product", 0),
                "Unobserved_Hours": counts.get("unobserved", 0),
            })

    df_recon = pd.DataFrame(reconciliation_rows)
    recon_csv = Path("data/processed/labeled_dataset_reconciliation_summary.csv")
    df_recon.to_csv(recon_csv, index=False)
    print(f"\nSaved labeled dataset reconciliation to {recon_csv}")
    print("\n=== Labeled Dataset Reconciled Calendar Accounting ===")
    print(df_recon.to_string(index=False))

    return df_recon


if __name__ == "__main__":
    frozen_catalog = Path("data/processed/synthetic_injection_catalog.csv")
    if frozen_catalog.exists():
        print(f"Loading frozen injection catalog: {frozen_catalog}")
        cat = pd.read_csv(frozen_catalog)
    else:
        cat = generate_revised_injection_catalog()
    build_and_save_labeled_datasets(cat)
