"""
Phase 3 (Revised): Physically Grounded Synthetic Fission Injection Generator.

Implements all Gate 3 physical specifications:
1. Multi-nuclide inventory: Cs-137 (662 keV), I-131 (365 keV), Co-60 (1173 & 1332 keV), Cs-134 (605, 796, 802 keV).
   - Co-60 lines in R07 eliminate the artificial "R07 is only radon washout" shortcut.
   - Non-zero continuum / pileup floor across all channels.
   - Spectral perturbation (+/-10% per channel) per injection so models cannot memorize fixed ratios.
2. Ambient Dose Rate Coupling:
   - Injects delta_Dose = k_dose * delta_Gross_CPM (k_dose ~ 0.016 nSv/h per CPM, calibrated to empirical rain regression).
3. Filter Accumulation & Replacement Physics:
   - Inflow/Passage Phase (T_passage): Plume accumulates cumulatively on continuous filter.
   - Retention Phase (T_retention): Plume has passed; activity remains on filter, decaying according to radiological half-life.
   - I-131 decay: lambda = ln(2)/(8.025 d * 24h) = 0.00360 /h applied during accumulation and retention.
   - Particulate fraction: Models particulate-bound radioiodine captured on glass fiber filter.
   - Filter Replacement (T_change): Activity drops to 0 at filter replacement.
4. Strict Non-Overlap Constraint:
   - 48-hour buffer before and after each injection. Exactly 0 overlapping injections across entire catalog.
5. True Dry vs Forced Rain Partitioning:
   - Standard Dry: Precip == 0.0 mm across the entire injection window and preceding 24h.
   - Forced Rain: Precip > 0.0 mm at plume onset. Included in BOTH Train (30%) and Test (50%).
6. Hard-Regime Stress Test Set:
   - Dedicated test subset of short (8-20h) plumes with subtle magnitudes (250-600 CPM) during active rain.
7. Exact Calendar Accounting:
   - Train: 2017-01-01 to 2023-01-01 (52,584 hours per station).
   - Test: 2023-01-01 to 2026-01-01 (26,304 hours per station).
   - Missing data labeled 'unobserved', not 'normal'.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import yaml


CONFIG_PATH = Path("configs/config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RANDOM_SEED = config.get("random_seed", 42)

STATIONS = [
    {"id": "al_birmingham", "name": "Birmingham, AL"},
    {"id": "dc_washington", "name": "Washington, DC"},
    {"id": "ca_san_diego", "name": "San Diego, CA"},
    {"id": "tx_dallas", "name": "Dallas, TX"},
    {"id": "fl_tampa", "name": "Tampa, FL"},
]

CHANNELS = [f"cpm_r0{i}" for i in range(2, 10)]

# Nominal NaI(Tl) channel energy branching fractions for pure radionuclides
# Normalized over reported channels R02-R09 (R01 <= 100 keV omitted by EPA as noise threshold)
# Grounded in ENSDF / IAEA nuclear data and standard NaI(Tl) response (Knoll 2010; Heath 1964)
NOMINAL_SPECTRA = {
    # Cs-137: 661.7 keV photopeak in R05 (43%), Compton in R02-R04 (55%), high-energy tail in R06-R07 (2%)
    "cs137": np.array([0.200, 0.150, 0.200, 0.430, 0.015, 0.005, 0.000, 0.000]),
    # I-131: 364.5 keV photopeak in R03 (63%), Compton in R02 (25%), weak lines in R04-R05, tail in R06-R07
    "i131":  np.array([0.250, 0.630, 0.050, 0.050, 0.015, 0.005, 0.000, 0.000]),
    # Co-60: 1173.2 and 1332.5 keV photopeaks in R07 (28%), Compton continuum across R02-R06 (70%), tail in R08 (2%)
    "co60":  np.array([0.150, 0.150, 0.150, 0.150, 0.100, 0.280, 0.020, 0.000]),
    # Cs-134: 604.7 keV (R05), 795.9 keV (R05/R06), 802.0 keV (R06), 1365.2 keV (R07)
    "cs134": np.array([0.180, 0.200, 0.150, 0.350, 0.080, 0.030, 0.010, 0.000]),
}

LAMBDA_I131 = np.log(2.0) / (8.025 * 24.0)  # 0.003600 h^-1


def compute_physical_injection_profile(shape_family: str, t_passage: int, t_retention: int, peak_cpm: float, rise_param: float, f_i131: float) -> np.ndarray:
    """
    Computes physical count rate profile S(t) for t in 0..D-1 (D = t_passage + t_retention):
    1. During plume passage (t < t_passage): cumulative accumulation of sampled airborne particulates on filter.
    2. During retention (t >= t_passage): atmospheric plume has passed; activity on filter decays with radiological half-life.
    3. At t = D: filter replacement removes loaded activity (signal drops to zero).
    """
    d = t_passage + t_retention
    lam = f_i131 * LAMBDA_I131
    
    # Atmospheric concentration profile C(u) during plume passage
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

    # Filter accumulation during passage
    profile = np.zeros(d)
    for i in range(t_passage):
        decay_weights = np.exp(-lam * (i - np.arange(i + 1)))
        profile[i] = np.sum(c[:i + 1] * decay_weights)
        
    scale = peak_cpm / (profile[t_passage - 1] + 1e-6)
    profile[:t_passage] *= scale
    
    # Retention phase: exponential radiological decay until filter replacement
    peak_val = profile[t_passage - 1]
    for i in range(t_passage, d):
        dt_ret = i - (t_passage - 1)
        profile[i] = peak_val * np.exp(-lam * dt_ret)
        
    return profile


def sample_perturbed_spectrum(rng: np.random.RandomState, f_cs137: float, f_i131: float, f_co60: float, f_cs134: float) -> np.ndarray:
    """
    Computes composite base spectrum from nuclide shares, then applies random perturbation (+/-10% per channel)
    to model detector gain drift, temperature shifts, and particulate mass shielding.
    """
    base = (
        f_cs137 * NOMINAL_SPECTRA["cs137"] +
        f_i131  * NOMINAL_SPECTRA["i131"] +
        f_co60  * NOMINAL_SPECTRA["co60"] +
        f_cs134 * NOMINAL_SPECTRA["cs134"]
    )
    # Relative Gaussian perturbation per channel (sigma = 10%)
    pert = rng.normal(0.0, 0.10, size=len(base))
    w = np.maximum(0.001, base * (1.0 + pert))
    return w / np.sum(w)


def generate_revised_injection_catalog():
    """
    Generates deterministic, non-overlapping synthetic injection catalog across all 5 pilot stations.
    - 50 train injections per station (250 total, 2017-2022)
    - 40 test injections per station (200 total, 2023-2025):
        - 20 standard duration (36-80h), Band B [700, 1200] CPM
        - 20 short duration hard-regime stress test (8-20h), Band A [250, 600] CPM
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

        occupied = np.zeros(len(df), dtype=bool)

        # -------------------------------------------------------------
        # 1. Training Set (2017-01-01 to 2023-01-01: 52,584 hours)
        # -------------------------------------------------------------
        train_mask = (df["dt"] >= "2017-01-01") & (df["dt"] < "2023-01-01")
        
        # Candidate start pools in train
        rain_starts_train = df[train_mask & df["has_radnet_obs"] & df["rad_complete_channels"] & (df["precip_1h_mm"] >= 1.0)].index.to_numpy()
        
        # 24h rolling dry mask for clean dry starts
        df["precip_24h"] = df["precip_1h_mm"].rolling(24, min_periods=12).sum()
        dry_starts_train = df[train_mask & df["has_radnet_obs"] & df["rad_complete_channels"] & (df["precip_1h_mm"] == 0.0) & (df["precip_24h"] == 0.0)].index.to_numpy()

        n_train_rain = 15  # 30% forced rain onset in train
        n_train_dry = 35   # 70% clean dry in train
        
        # A. Train forced rain-onset injections
        placed_rain_tr = 0
        shuffled_rain = rng.permutation(rain_starts_train)
        for idx in shuffled_rain:
            if placed_rain_tr >= n_train_rain:
                break
            t_pass = int(rng.randint(4, 15))
            t_ret = int(rng.randint(6, 21))
            dur = t_pass + t_ret
            if idx + dur >= len(df):
                continue
            # Non-overlap check with 48h buffer
            b_start = max(0, idx - 48)
            b_end = min(len(df), idx + dur + 48)
            if occupied[b_start:b_end].any():
                continue
            window = df.iloc[idx:idx + dur]
            if not (window["has_radnet_obs"] & window["rad_complete_channels"]).all():
                continue

            # Shape: linear_ramp or step
            shape = rng.choice(["linear_ramp", "step"])
            rise_p = rng.uniform(1.0, 4.0) if shape == "linear_ramp" else 0.0
            
            # Magnitude: Band A [250, 600] or Band C [1400, 2500]
            mag_band = rng.choice(["Band_A_Low", "Band_C_High"])
            peak_cpm = round(rng.uniform(250.0, 600.0) if mag_band == "Band_A_Low" else rng.uniform(1400.0, 2500.0), 1)

            # Balanced multi-nuclide mix in train
            f_cs = round(rng.uniform(0.30, 0.50), 2)
            f_i = round(rng.uniform(0.20, 0.40), 2)
            f_co = round(rng.uniform(0.10, 0.25), 2)
            f_cs134 = round(1.0 - (f_cs + f_i + f_co), 2)
            if f_cs134 < 0:
                f_cs134 = 0.0
                f_i = round(1.0 - (f_cs + f_co), 2)

            k_dose = round(rng.normal(0.016, 0.002), 5)
            k_dose = max(0.012, min(0.020, k_dose))

            occupied[idx:idx + dur] = True
            catalog.append({
                "injection_id": f"INJ_{injection_id:04d}",
                "split": "train",
                "station_id": st_id,
                "station_name": st["name"],
                "start_utc": str(df.loc[idx, "dt"]),
                "end_utc": str(df.loc[idx + dur - 1, "dt"]),
                "start_index": int(idx),
                "t_passage_hours": t_pass,
                "t_retention_hours": t_ret,
                "duration_hours": dur,
                "shape_family": shape,
                "rise_param": round(rise_p, 2),
                "magnitude_band": mag_band,
                "peak_cpm": peak_cpm,
                "fraction_cs137": f_cs,
                "fraction_i131": f_i,
                "fraction_co60": f_co,
                "fraction_cs134": f_cs134,
                "k_dose": k_dose,
                "is_hard_case_rain": True,
                "environment": "train_rain_coincident",
            })
            injection_id += 1
            placed_rain_tr += 1

        # B. Train strictly dry injections
        placed_dry_tr = 0
        shuffled_dry = rng.permutation(dry_starts_train)
        for idx in shuffled_dry:
            if placed_dry_tr >= n_train_dry:
                break
            t_pass = int(rng.randint(4, 15))
            t_ret = int(rng.randint(6, 21))
            dur = t_pass + t_ret
            if idx + dur >= len(df):
                continue
            b_start = max(0, idx - 48)
            b_end = min(len(df), idx + dur + 48)
            if occupied[b_start:b_end].any():
                continue
            window = df.iloc[idx:idx + dur]
            if not (window["has_radnet_obs"] & window["rad_complete_channels"]).all():
                continue
            # Strictly dry across entire window
            if (window["precip_1h_mm"] > 0.0).any():
                continue

            shape = rng.choice(["linear_ramp", "step"])
            rise_p = rng.uniform(1.0, 4.0) if shape == "linear_ramp" else 0.0
            mag_band = rng.choice(["Band_A_Low", "Band_C_High"])
            peak_cpm = round(rng.uniform(250.0, 600.0) if mag_band == "Band_A_Low" else rng.uniform(1400.0, 2500.0), 1)

            f_cs = round(rng.uniform(0.30, 0.50), 2)
            f_i = round(rng.uniform(0.20, 0.40), 2)
            f_co = round(rng.uniform(0.10, 0.25), 2)
            f_cs134 = round(1.0 - (f_cs + f_i + f_co), 2)
            if f_cs134 < 0:
                f_cs134 = 0.0
                f_i = round(1.0 - (f_cs + f_co), 2)

            k_dose = round(rng.normal(0.016, 0.002), 5)
            k_dose = max(0.012, min(0.020, k_dose))

            occupied[idx:idx + dur] = True
            catalog.append({
                "injection_id": f"INJ_{injection_id:04d}",
                "split": "train",
                "station_id": st_id,
                "station_name": st["name"],
                "start_utc": str(df.loc[idx, "dt"]),
                "end_utc": str(df.loc[idx + dur - 1, "dt"]),
                "start_index": int(idx),
                "t_passage_hours": t_pass,
                "t_retention_hours": t_ret,
                "duration_hours": dur,
                "shape_family": shape,
                "rise_param": round(rise_p, 2),
                "magnitude_band": mag_band,
                "peak_cpm": peak_cpm,
                "fraction_cs137": f_cs,
                "fraction_i131": f_i,
                "fraction_co60": f_co,
                "fraction_cs134": f_cs134,
                "k_dose": k_dose,
                "is_hard_case_rain": False,
                "environment": "train_strictly_dry",
            })
            injection_id += 1
            placed_dry_tr += 1

        # -------------------------------------------------------------
        # 2. Test Set (2023-01-01 to 2026-01-01: 26,304 hours)
        # -------------------------------------------------------------
        test_mask = (df["dt"] >= "2023-01-01") & (df["dt"] < "2026-01-01")
        rain_starts_test = df[test_mask & df["has_radnet_obs"] & df["rad_complete_channels"] & (df["precip_1h_mm"] >= 1.0)].index.to_numpy()
        dry_starts_test = df[test_mask & df["has_radnet_obs"] & df["rad_complete_channels"] & (df["precip_1h_mm"] == 0.0) & (df["precip_24h"] == 0.0)].index.to_numpy()

        # Subset 2A: Standard Test Cases (20 per station) - Band B Interpolation [700, 1200] CPM, medium/long duration
        # 10 dry, 10 rain-onset
        n_std_test_dry = 10
        n_std_test_rain = 10

        # Subset 2B: Short-Duration Hard-Regime Stress Cases (20 per station) - Band A Subtle [250, 600] CPM, short duration (8-20h)
        # 10 dry, 10 rain-onset
        n_stress_test_dry = 10
        n_stress_test_rain = 10

        # Helper function for test nuclide selection (pure or binary skewed)
        def sample_test_nuclides():
            choice = rng.choice(["pure_cs137", "pure_i131", "pure_co60", "skewed_cs_co"])
            if choice == "pure_cs137":
                return 0.90, 0.05, 0.05, 0.0
            elif choice == "pure_i131":
                return 0.05, 0.90, 0.05, 0.0
            elif choice == "pure_co60":
                return 0.05, 0.05, 0.90, 0.0
            else:
                return 0.50, 0.00, 0.50, 0.0

        # 2A-1: Standard Test Forced Rain
        placed = 0
        for idx in rng.permutation(rain_starts_test):
            if placed >= n_std_test_rain:
                break
            t_pass = int(rng.randint(10, 21))
            t_ret = int(rng.randint(24, 61))
            dur = t_pass + t_ret
            if idx + dur >= len(df):
                continue
            b_start = max(0, idx - 48)
            b_end = min(len(df), idx + dur + 48)
            if occupied[b_start:b_end].any():
                continue
            window = df.iloc[idx:idx + dur]
            if not (window["has_radnet_obs"] & window["rad_complete_channels"]).all():
                continue

            shape = rng.choice(["sigmoidal", "exponential"])
            rise_p = rng.uniform(0.8, 1.8) if shape == "sigmoidal" else rng.uniform(1.0, 2.5)
            peak_cpm = round(rng.uniform(700.0, 1200.0), 1)  # Band B
            f_cs, f_i, f_co, f_cs134 = sample_test_nuclides()
            k_dose = round(rng.normal(0.016, 0.002), 5)
            k_dose = max(0.012, min(0.020, k_dose))

            occupied[idx:idx + dur] = True
            catalog.append({
                "injection_id": f"INJ_{injection_id:04d}",
                "split": "test",
                "station_id": st_id,
                "station_name": st["name"],
                "start_utc": str(df.loc[idx, "dt"]),
                "end_utc": str(df.loc[idx + dur - 1, "dt"]),
                "start_index": int(idx),
                "t_passage_hours": t_pass,
                "t_retention_hours": t_ret,
                "duration_hours": dur,
                "shape_family": shape,
                "rise_param": round(rise_p, 2),
                "magnitude_band": "Band_B_Mid (Interpolation)",
                "peak_cpm": peak_cpm,
                "fraction_cs137": f_cs,
                "fraction_i131": f_i,
                "fraction_co60": f_co,
                "fraction_cs134": f_cs134,
                "k_dose": k_dose,
                "is_hard_case_rain": True,
                "environment": "test_standard_rain_onset",
            })
            injection_id += 1
            placed += 1

        # 2A-2: Standard Test Strictly Dry
        placed = 0
        for idx in rng.permutation(dry_starts_test):
            if placed >= n_std_test_dry:
                break
            t_pass = int(rng.randint(10, 21))
            t_ret = int(rng.randint(24, 61))
            dur = t_pass + t_ret
            if idx + dur >= len(df):
                continue
            b_start = max(0, idx - 48)
            b_end = min(len(df), idx + dur + 48)
            if occupied[b_start:b_end].any():
                continue
            window = df.iloc[idx:idx + dur]
            if not (window["has_radnet_obs"] & window["rad_complete_channels"]).all():
                continue
            if (window["precip_1h_mm"] > 0.0).any():
                continue

            shape = rng.choice(["sigmoidal", "exponential"])
            rise_p = rng.uniform(0.8, 1.8) if shape == "sigmoidal" else rng.uniform(1.0, 2.5)
            peak_cpm = round(rng.uniform(700.0, 1200.0), 1)  # Band B
            f_cs, f_i, f_co, f_cs134 = sample_test_nuclides()
            k_dose = round(rng.normal(0.016, 0.002), 5)
            k_dose = max(0.012, min(0.020, k_dose))

            occupied[idx:idx + dur] = True
            catalog.append({
                "injection_id": f"INJ_{injection_id:04d}",
                "split": "test",
                "station_id": st_id,
                "station_name": st["name"],
                "start_utc": str(df.loc[idx, "dt"]),
                "end_utc": str(df.loc[idx + dur - 1, "dt"]),
                "start_index": int(idx),
                "t_passage_hours": t_pass,
                "t_retention_hours": t_ret,
                "duration_hours": dur,
                "shape_family": shape,
                "rise_param": round(rise_p, 2),
                "magnitude_band": "Band_B_Mid (Interpolation)",
                "peak_cpm": peak_cpm,
                "fraction_cs137": f_cs,
                "fraction_i131": f_i,
                "fraction_co60": f_co,
                "fraction_cs134": f_cs134,
                "k_dose": k_dose,
                "is_hard_case_rain": False,
                "environment": "test_standard_strictly_dry",
            })
            injection_id += 1
            placed += 1

        # 2B-1: Short-Duration Hard-Regime Stress Test Forced Rain
        placed = 0
        for idx in rng.permutation(rain_starts_test):
            if placed >= n_stress_test_rain:
                break
            t_pass = int(rng.randint(3, 9))
            t_ret = int(rng.randint(4, 12))
            dur = t_pass + t_ret
            if idx + dur >= len(df):
                continue
            b_start = max(0, idx - 48)
            b_end = min(len(df), idx + dur + 48)
            if occupied[b_start:b_end].any():
                continue
            window = df.iloc[idx:idx + dur]
            if not (window["has_radnet_obs"] & window["rad_complete_channels"]).all():
                continue

            shape = rng.choice(["sigmoidal", "exponential"])
            rise_p = rng.uniform(0.6, 1.4) if shape == "sigmoidal" else rng.uniform(0.8, 1.8)
            peak_cpm = round(rng.uniform(250.0, 600.0), 1)  # Band A Subtle
            f_cs, f_i, f_co, f_cs134 = sample_test_nuclides()
            k_dose = round(rng.normal(0.016, 0.002), 5)
            k_dose = max(0.012, min(0.020, k_dose))

            occupied[idx:idx + dur] = True
            catalog.append({
                "injection_id": f"INJ_{injection_id:04d}",
                "split": "test",
                "station_id": st_id,
                "station_name": st["name"],
                "start_utc": str(df.loc[idx, "dt"]),
                "end_utc": str(df.loc[idx + dur - 1, "dt"]),
                "start_index": int(idx),
                "t_passage_hours": t_pass,
                "t_retention_hours": t_ret,
                "duration_hours": dur,
                "shape_family": shape,
                "rise_param": round(rise_p, 2),
                "magnitude_band": "Band_A_Low (Stress Test)",
                "peak_cpm": peak_cpm,
                "fraction_cs137": f_cs,
                "fraction_i131": f_i,
                "fraction_co60": f_co,
                "fraction_cs134": f_cs134,
                "k_dose": k_dose,
                "is_hard_case_rain": True,
                "environment": "test_stress_hard_case_rain",
            })
            injection_id += 1
            placed += 1

        # 2B-2: Short-Duration Hard-Regime Stress Test Strictly Dry
        placed = 0
        for idx in rng.permutation(dry_starts_test):
            if placed >= n_stress_test_dry:
                break
            t_pass = int(rng.randint(3, 9))
            t_ret = int(rng.randint(4, 12))
            dur = t_pass + t_ret
            if idx + dur >= len(df):
                continue
            b_start = max(0, idx - 48)
            b_end = min(len(df), idx + dur + 48)
            if occupied[b_start:b_end].any():
                continue
            window = df.iloc[idx:idx + dur]
            if not (window["has_radnet_obs"] & window["rad_complete_channels"]).all():
                continue
            if (window["precip_1h_mm"] > 0.0).any():
                continue

            shape = rng.choice(["sigmoidal", "exponential"])
            rise_p = rng.uniform(0.6, 1.4) if shape == "sigmoidal" else rng.uniform(0.8, 1.8)
            peak_cpm = round(rng.uniform(250.0, 600.0), 1)  # Band A Subtle
            f_cs, f_i, f_co, f_cs134 = sample_test_nuclides()
            k_dose = round(rng.normal(0.016, 0.002), 5)
            k_dose = max(0.012, min(0.020, k_dose))

            occupied[idx:idx + dur] = True
            catalog.append({
                "injection_id": f"INJ_{injection_id:04d}",
                "split": "test",
                "station_id": st_id,
                "station_name": st["name"],
                "start_utc": str(df.loc[idx, "dt"]),
                "end_utc": str(df.loc[idx + dur - 1, "dt"]),
                "start_index": int(idx),
                "t_passage_hours": t_pass,
                "t_retention_hours": t_ret,
                "duration_hours": dur,
                "shape_family": shape,
                "rise_param": round(rise_p, 2),
                "magnitude_band": "Band_A_Low (Stress Test)",
                "peak_cpm": peak_cpm,
                "fraction_cs137": f_cs,
                "fraction_i131": f_i,
                "fraction_co60": f_co,
                "fraction_cs134": f_cs134,
                "k_dose": k_dose,
                "is_hard_case_rain": False,
                "environment": "test_stress_strictly_dry",
            })
            injection_id += 1
            placed += 1

    df_catalog = pd.DataFrame(catalog)
    out_csv = Path("data/processed/synthetic_injection_catalog.csv")
    df_catalog.to_csv(out_csv, index=False)
    print(f"Generated revised synthetic injection catalog ({len(df_catalog)} injections) to {out_csv}")
    
    # Audit for overlaps
    overlap_count = 0
    for st_id in df_catalog["station_id"].unique():
        st_inj = df_catalog[df_catalog["station_id"] == st_id].sort_values("start_index").reset_index(drop=True)
        for i in range(len(st_inj) - 1):
            cur_end = st_inj.loc[i, "start_index"] + st_inj.loc[i, "duration_hours"]
            nxt_start = st_inj.loc[i + 1, "start_index"]
            if cur_end > nxt_start:
                print(f"OVERLAP ERROR at {st_id}: {st_inj.loc[i, 'injection_id']} ends at {cur_end} but {st_inj.loc[i+1, 'injection_id']} starts at {nxt_start}!")
                overlap_count += 1
    print(f"Total overlapping injection pairs detected: {overlap_count}")

    print("\n=== Revised Synthetic Injection Catalog Summary ===")
    summary = df_catalog.groupby(["split", "environment", "magnitude_band"]).agg(
        Count=("injection_id", "count"),
        Mean_Duration_h=("duration_hours", "mean"),
        Mean_Peak_CPM=("peak_cpm", "mean"),
        Mean_f_Cs=("fraction_cs137", "mean"),
        Mean_f_Co=("fraction_co60", "mean"),
        Rain_Coincident=("is_hard_case_rain", "sum"),
    ).reset_index()
    print(summary.to_string(index=False))

    return df_catalog


def build_and_save_labeled_datasets(catalog: pd.DataFrame):
    """
    Applies synthetic injections with perturbed spectra and physical accumulation/retention/decay,
    and saves labeled datasets and audit summaries.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    reconciliation_rows = []

    for st in STATIONS:
        st_id = st["id"]
        csv_file = Path(f"data/processed/merged_{st_id}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        df["dt"] = pd.to_datetime(df["utc_hour"])
        df = df.sort_values("dt").reset_index(drop=True)

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

            # 2. Rule-based labeling of natural radon washout (operational heuristic, partial circularity acknowledged):
            sub["precip_3h"] = sub["precip_1h_mm"].rolling(3, min_periods=1).sum()
            sub["precip_24h"] = sub["precip_1h_mm"].rolling(24, min_periods=12).sum()
            dry_mask = (sub["precip_24h"] == 0.0) & (sub["precip_1h_mm"] == 0.0) & has_obs & comp_ch
            mu_dry = sub.loc[dry_mask, "gross_cpm"].mean()
            sigma_dry = sub.loc[dry_mask, "gross_cpm"].std()

            washout_condition = (
                (sub["precip_3h"] > 0.0) &
                (sub["gross_cpm"] > (mu_dry + 2.0 * sigma_dry)) &
                (~unobs_mask)
            )
            sub.loc[washout_condition, "label"] = "radon_washout"

            # 3. Apply synthetic injections from catalog
            st_injections = catalog[(catalog["station_id"] == st_id) & (catalog["split"] == split_name)]
            for _, row in st_injections.iterrows():
                inj_start = pd.to_datetime(row["start_utc"])
                inj_end = pd.to_datetime(row["end_utc"])
                inj_mask = (sub["dt"] >= inj_start) & (sub["dt"] <= inj_end)
                indices = sub[inj_mask].index.to_numpy()
                dur = len(indices)
                if dur == 0:
                    continue

                # Physical profile
                prof = compute_physical_injection_profile(
                    shape_family=row["shape_family"],
                    t_passage=int(row["t_passage_hours"]),
                    t_retention=int(row["t_retention_hours"]),
                    peak_cpm=float(row["peak_cpm"]),
                    rise_param=float(row["rise_param"]),
                    f_i131=float(row["fraction_i131"])
                )

                # Sample perturbed spectrum for this injection
                spec = sample_perturbed_spectrum(
                    rng=rng,
                    f_cs137=float(row["fraction_cs137"]),
                    f_i131=float(row["fraction_i131"]),
                    f_co60=float(row["fraction_co60"]),
                    f_cs134=float(row["fraction_cs134"])
                )

                k_dose = float(row["k_dose"])

                for idx_sub, s_val in zip(indices, prof):
                    sub.loc[idx_sub, "inj_gross_cpm"] += s_val
                    sub.loc[idx_sub, "synthetic_excess_cpm"] = s_val
                    sub.loc[idx_sub, "inj_dose_rate_nsvh"] += s_val * k_dose
                    for ch_idx, ch in enumerate(CHANNELS):
                        sub.loc[idx_sub, f"inj_{ch}"] += s_val * spec[ch_idx]
                    sub.loc[idx_sub, "label"] = "fission_product"
                    sub.loc[idx_sub, "injection_active"] = True
                    sub.loc[idx_sub, "injection_id"] = row["injection_id"]

            # Save labeled dataset
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
    cat = generate_revised_injection_catalog()
    recon = build_and_save_labeled_datasets(cat)
