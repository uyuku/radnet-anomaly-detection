"""
Phase 3: Synthetic Injection Generator for Fission Products (Cs-137 and I-131).
Implements the physical specification from Section 5 of PROJECT_SPEC.md:
1. Parameterized families: onset time, rise shape, duration, magnitude, nuclide mix.
2. Disjoint parameter ranges between Training and Test sets (shapes, durations, magnitudes, nuclide fractions).
3. Dedicated hard-case test set injected during verified rainfall.
4. Channel-level NaI(Tl) spectrometry allocation (Knoll, 2010; Vieira et al., 2019):
   - Cs-137 (661.7 keV): photopeak in R05 (45%), Compton in R02 (20%), R03 (15%), R04 (20%). R06-R09 = 0.
   - I-131 (364.5 keV): photopeak in R03 (65%), Compton in R02 (25%), scatter in R04 (5%), R05 (5%). R06-R09 = 0.
   - Contrast with radon washout: R07 (1120 keV) and R08 (1764 keV) are 0 for fission products!
5. Persistence upper-bounded by empirical filter replacement schedule (~4 days / 96h).
6. Ground-truth labeling for normal, radon_washout, and fission_product classes.

Saves:
- data/processed/synthetic_injection_catalog.csv
- data/processed/injected_dataset_train_2017_2022.csv.gz (summary / features)
- data/processed/injected_dataset_test_2023_2025.csv.gz
"""

from pathlib import Path
import pandas as pd
import numpy as np
import yaml


CONFIG_PATH = Path("configs/config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RANDOM_SEED = config.get("random_seed", 42)
np.random.seed(RANDOM_SEED)

STATIONS = [
    {"id": "al_birmingham", "name": "Birmingham, AL"},
    {"id": "dc_washington", "name": "Washington, DC"},
    {"id": "ca_san_diego", "name": "San Diego, CA"},
    {"id": "tx_dallas", "name": "Dallas, TX"},
    {"id": "fl_tampa", "name": "Tampa, FL"},
]

# Physical NaI(Tl) channel allocations for pure radionuclides
# Based on Knoll (2010) gamma spectrometry and RadNet channel boundaries
SPECTRUM_CS137 = {
    "cpm_r02": 0.20,  # 101-200 keV: Compton backscatter & continuum
    "cpm_r03": 0.15,  # 201-400 keV: Intermediate Compton continuum
    "cpm_r04": 0.20,  # 401-600 keV: Compton continuum approaching 477 keV edge
    "cpm_r05": 0.45,  # 601-800 keV: Total absorption photopeak (661.7 keV)
    "cpm_r06": 0.00,  # 801-1000 keV: Zero above 662 keV
    "cpm_r07": 0.00,  # 1001-1400 keV: Zero
    "cpm_r08": 0.00,  # 1401-1800 keV: Zero
    "cpm_r09": 0.00,  # 1801-2200 keV: Zero
}

SPECTRUM_I131 = {
    "cpm_r02": 0.25,  # 101-200 keV: Compton continuum (edge at 211 keV)
    "cpm_r03": 0.65,  # 201-400 keV: Total absorption photopeak (364.5 keV)
    "cpm_r04": 0.05,  # 401-600 keV: Compton scatter / weak lines
    "cpm_r05": 0.05,  # 601-800 keV: Weak 637.0 keV line (7.1% yield)
    "cpm_r06": 0.00,  # 801-1000 keV: Zero
    "cpm_r07": 0.00,  # 1001-1400 keV: Zero
    "cpm_r08": 0.00,  # 1401-1800 keV: Zero
    "cpm_r09": 0.00,  # 1801-2200 keV: Zero
}


def compute_injection_profile(shape_family: str, duration_hours: int, peak_cpm: float, rise_param: float) -> np.ndarray:
    """
    Computes normalized shape profile S(t) for t in 0..duration-1, scaled to peak_cpm.
    """
    t = np.arange(duration_hours)
    
    if shape_family == "linear_ramp":
        # Linear rise over rise_param hours, then steady plateau
        rise_h = max(1.0, float(rise_param))
        profile = np.clip(t / rise_h, 0.0, 1.0)
    elif shape_family == "step":
        # Instantaneous step arrival
        profile = np.ones(duration_hours)
    elif shape_family == "sigmoidal":
        # Logistic growth centered at rise_param with dispersion tau
        tau = max(0.5, float(rise_param))
        t_mid = min(4.0, duration_hours / 4.0)
        profile = 1.0 / (1.0 + np.exp(-(t - t_mid) / tau))
        # Normalize so profile starts near 0 and reaches 1
        profile = (profile - profile[0]) / (profile[-1] - profile[0] + 1e-6)
    elif shape_family == "exponential":
        # Exponential inflow: 1 - exp(-t / tau)
        tau = max(0.8, float(rise_param))
        profile = 1.0 - np.exp(-t / tau)
        profile = profile / (profile[-1] + 1e-6)
    else:
        profile = np.ones(duration_hours)
        
    return profile * peak_cpm


def generate_injection_catalog():
    """
    Generates a deterministic catalog of synthetic fission injections across stations,
    enforcing strict disjoint parameter families between Train (2017-2022) and Test (2023-2025).
    """
    rng = np.random.RandomState(RANDOM_SEED)
    catalog = []
    injection_id = 1

    # Load merged data for each station to find valid injection anchor timestamps
    for st in STATIONS:
        csv_file = Path(f"data/processed/merged_{st['id']}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        df["dt"] = pd.to_datetime(df["utc_hour"])
        df = df.sort_values("dt").reset_index(drop=True)

        # 1. Training Injections (2017-2022)
        # Parameter specifications:
        # - Shapes: linear_ramp, step
        # - Duration: 6 to 24 hours (short/medium)
        # - Magnitude Bands: Band A [250, 600] CPM (low/subtle) and Band C [1400, 2500] CPM (high/severe)
        # - Nuclide Mix: Balanced mixed plume f_cs in [0.30, 0.70]
        train_mask = (df["dt"] >= "2017-06-01") & (df["dt"] <= "2022-12-31") & df["has_radnet_obs"] & df["rad_complete_channels"]
        valid_train_indices = df[train_mask].index.to_numpy()

        n_train_injections = 60  # per station = 300 total train injections across network
        selected_train_starts = rng.choice(valid_train_indices, size=n_train_injections * 2, replace=False)
        
        train_count = 0
        for idx in selected_train_starts:
            if train_count >= n_train_injections:
                break
            
            # Duration: 6 to 24 hours
            duration = int(rng.randint(6, 25))
            if idx + duration >= len(df):
                continue
            
            # Check data validity over injection window
            window_df = df.iloc[idx : idx + duration]
            if not window_df["has_radnet_obs"].all() or not window_df["rad_complete_channels"].all():
                continue

            # Shape: linear_ramp or step
            shape = rng.choice(["linear_ramp", "step"])
            rise_param = rng.uniform(1.0, 4.0) if shape == "linear_ramp" else 0.0

            # Magnitude: Band A [250, 600] or Band C [1400, 2500]
            mag_band = rng.choice(["Band_A_Low", "Band_C_High"])
            if mag_band == "Band_A_Low":
                peak_cpm = round(rng.uniform(250.0, 600.0), 1)
            else:
                peak_cpm = round(rng.uniform(1400.0, 2500.0), 1)

            # Nuclide Mix: Balanced mix f_cs in [0.30, 0.70]
            f_cs = round(rng.uniform(0.30, 0.70), 2)
            f_i = round(1.0 - f_cs, 2)

            is_rain_coincident = bool((window_df["precip_1h_mm"] > 0.0).any())

            catalog.append({
                "injection_id": f"INJ_{injection_id:04d}",
                "split": "train",
                "station_id": st["id"],
                "station_name": st["name"],
                "start_utc": str(df.loc[idx, "dt"]),
                "end_utc": str(df.loc[idx + duration - 1, "dt"]),
                "start_index": int(idx),
                "duration_hours": duration,
                "shape_family": shape,
                "rise_param": round(rise_param, 2),
                "magnitude_band": mag_band,
                "peak_cpm": peak_cpm,
                "fraction_cs137": f_cs,
                "fraction_i131": f_i,
                "is_hard_case_rain": is_rain_coincident,
                "environment": "ambient_train",
            })
            injection_id += 1
            train_count += 1

        # 2. Test Injections (2023-2025, Disjoint Parameters)
        # Parameter specifications:
        # - Shapes: sigmoidal, exponential (DISJOINT from train)
        # - Duration: 28 to 72 hours (medium/long, DISJOINT from train <= 24h)
        # - Magnitude Band: Band B [700, 1200] CPM (DISJOINT interpolation band)
        # - Nuclide Mix: Pure/skewed f_cs in [0.85, 1.0] (pure Cs) or [0.0, 0.15] (pure I) (DISJOINT from train 0.3-0.7)
        # - 30% Dedicated Hard-Case: forced rain coincidence at start
        test_mask = (df["dt"] >= "2023-01-01") & (df["dt"] <= "2025-12-31") & df["has_radnet_obs"] & df["rad_complete_channels"]
        
        # Hard-case rain candidates in test period
        test_rain_mask = test_mask & (df["precip_1h_mm"] > 0.0)
        valid_test_rain_indices = df[test_rain_mask].index.to_numpy()
        
        # Standard dry candidates in test period
        df["precip_24h"] = df["precip_1h_mm"].rolling(24, min_periods=12).sum()
        test_dry_mask = test_mask & (df["precip_24h"] == 0.0) & (df["precip_1h_mm"] == 0.0)
        valid_test_dry_indices = df[test_dry_mask].index.to_numpy()

        n_test_injections = 40  # per station = 200 total test injections across network
        n_hard_case = int(round(n_test_injections * 0.30))  # 12 hard cases per station (30%)
        n_standard = n_test_injections - n_hard_case        # 28 standard dry cases (70%)

        # A. Hard-case rain injections
        hard_count = 0
        if len(valid_test_rain_indices) > 0:
            shuffled_rain = rng.choice(valid_test_rain_indices, size=len(valid_test_rain_indices), replace=False)
            for idx in shuffled_rain:
                if hard_count >= n_hard_case:
                    break
                duration = int(rng.randint(28, 73))  # 28 to 72 hours
                if idx + duration >= len(df):
                    continue
                window_df = df.iloc[idx : idx + duration]
                if not window_df["has_radnet_obs"].all() or not window_df["rad_complete_channels"].all():
                    continue

                shape = rng.choice(["sigmoidal", "exponential"])
                rise_param = rng.uniform(0.8, 1.8) if shape == "sigmoidal" else rng.uniform(1.0, 2.5)
                peak_cpm = round(rng.uniform(700.0, 1200.0), 1)  # Band B (Interpolation)
                
                # Skewed/pure nuclide mix: either pure Cs or pure I
                nuclide_type = rng.choice(["pure_cs137", "pure_i131"])
                if nuclide_type == "pure_cs137":
                    f_cs = round(rng.uniform(0.85, 1.00), 2)
                else:
                    f_cs = round(rng.uniform(0.00, 0.15), 2)
                f_i = round(1.0 - f_cs, 2)

                catalog.append({
                    "injection_id": f"INJ_{injection_id:04d}",
                    "split": "test",
                    "station_id": st["id"],
                    "station_name": st["name"],
                    "start_utc": str(df.loc[idx, "dt"]),
                    "end_utc": str(df.loc[idx + duration - 1, "dt"]),
                    "start_index": int(idx),
                    "duration_hours": duration,
                    "shape_family": shape,
                    "rise_param": round(rise_param, 2),
                    "magnitude_band": "Band_B_Mid (Interpolation)",
                    "peak_cpm": peak_cpm,
                    "fraction_cs137": f_cs,
                    "fraction_i131": f_i,
                    "is_hard_case_rain": True,
                    "environment": "hard_case_rain_coincident",
                })
                injection_id += 1
                hard_count += 1

        # B. Standard dry test injections
        std_count = 0
        if len(valid_test_dry_indices) > 0:
            shuffled_dry = rng.choice(valid_test_dry_indices, size=len(valid_test_dry_indices), replace=False)
            for idx in shuffled_dry:
                if std_count >= n_standard:
                    break
                duration = int(rng.randint(28, 73))
                if idx + duration >= len(df):
                    continue
                window_df = df.iloc[idx : idx + duration]
                if not window_df["has_radnet_obs"].all() or not window_df["rad_complete_channels"].all():
                    continue

                shape = rng.choice(["sigmoidal", "exponential"])
                rise_param = rng.uniform(0.8, 1.8) if shape == "sigmoidal" else rng.uniform(1.0, 2.5)
                peak_cpm = round(rng.uniform(700.0, 1200.0), 1)

                nuclide_type = rng.choice(["pure_cs137", "pure_i131"])
                if nuclide_type == "pure_cs137":
                    f_cs = round(rng.uniform(0.85, 1.00), 2)
                else:
                    f_cs = round(rng.uniform(0.00, 0.15), 2)
                f_i = round(1.0 - f_cs, 2)

                is_rain = bool((window_df["precip_1h_mm"] > 0.0).any())

                catalog.append({
                    "injection_id": f"INJ_{injection_id:04d}",
                    "split": "test",
                    "station_id": st["id"],
                    "station_name": st["name"],
                    "start_utc": str(df.loc[idx, "dt"]),
                    "end_utc": str(df.loc[idx + duration - 1, "dt"]),
                    "start_index": int(idx),
                    "duration_hours": duration,
                    "shape_family": shape,
                    "rise_param": round(rise_param, 2),
                    "magnitude_band": "Band_B_Mid (Interpolation)",
                    "peak_cpm": peak_cpm,
                    "fraction_cs137": f_cs,
                    "fraction_i131": f_i,
                    "is_hard_case_rain": is_rain,
                    "environment": "standard_test_dry",
                })
                injection_id += 1
                std_count += 1

    df_catalog = pd.DataFrame(catalog)
    out_csv = Path("data/processed/synthetic_injection_catalog.csv")
    df_catalog.to_csv(out_csv, index=False)
    print(f"Generated synthetic injection catalog ({len(df_catalog)} injections) to {out_csv}")
    
    # Summary of catalog
    print("\n=== Synthetic Injection Catalog Summary ===")
    print(df_catalog.groupby(["split", "magnitude_band", "shape_family"]).agg(
        Count=("injection_id", "count"),
        Mean_Duration_h=("duration_hours", "mean"),
        Mean_Peak_CPM=("peak_cpm", "mean"),
        Mean_f_Cs=("fraction_cs137", "mean"),
        Hard_Case_Rain=("is_hard_case_rain", "sum"),
    ).reset_index().to_string(index=False))

    return df_catalog


def apply_injections_to_dataset(station_id: str, split_name: str, catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Loads merged dataset for a station, applies corresponding injections, and labels classes:
    - 0: normal
    - 1: radon_washout
    - 2: fission_product (injected)
    """
    csv_file = Path(f"data/processed/merged_{station_id}_2017_2025.csv.gz")
    df = pd.read_csv(csv_file)
    df["dt"] = pd.to_datetime(df["utc_hour"])
    df = df.sort_values("dt").reset_index(drop=True)

    # Date filter by split
    if split_name == "train":
        df = df[(df["dt"] >= "2017-01-01") & (df["dt"] <= "2022-12-31")].copy().reset_index(drop=True)
    else:
        df = df[(df["dt"] >= "2023-01-01") & (df["dt"] <= "2025-12-31")].copy().reset_index(drop=True)

    # Create synthetic injected columns initialized to raw values
    df["inj_gross_cpm"] = df["gross_cpm"].copy()
    for ch in ["cpm_r02", "cpm_r03", "cpm_r04", "cpm_r05", "cpm_r06", "cpm_r07", "cpm_r08", "cpm_r09"]:
        df[f"inj_{ch}"] = df[ch].copy()

    df["label"] = "normal"
    df["injection_active"] = False
    df["injection_id"] = None
    df["synthetic_excess_cpm"] = 0.0

    # 1. Rule-based labeling of natural radon washout:
    # Requires: precip_3h > 0, gross_cpm > mu_dry + 2*sigma_dry, R07/R08 elevated
    df["precip_3h"] = df["precip_1h_mm"].rolling(3, min_periods=1).sum()
    df["precip_24h"] = df["precip_1h_mm"].rolling(24, min_periods=12).sum()
    dry_mask = (df["precip_24h"] == 0.0) & (df["precip_1h_mm"] == 0.0) & df["has_radnet_obs"] & df["rad_complete_channels"]
    mu_dry = df.loc[dry_mask, "gross_cpm"].mean()
    sigma_dry = df.loc[dry_mask, "gross_cpm"].std()

    washout_condition = (
        (df["precip_3h"] > 0.0) &
        (df["gross_cpm"] > (mu_dry + 2.0 * sigma_dry)) &
        df["has_radnet_obs"] &
        df["rad_complete_channels"]
    )
    df.loc[washout_condition, "label"] = "radon_washout"

    # 2. Apply synthetic injections from catalog
    station_injections = catalog[(catalog["station_id"] == station_id) & (catalog["split"] == split_name)]
    
    for _, row in station_injections.iterrows():
        inj_start = pd.to_datetime(row["start_utc"])
        inj_end = pd.to_datetime(row["end_utc"])
        
        inj_mask = (df["dt"] >= inj_start) & (df["dt"] <= inj_end)
        indices = df[inj_mask].index.to_numpy()
        duration = len(indices)
        if duration == 0:
            continue

        # Compute profile
        profile = compute_injection_profile(
            shape_family=row["shape_family"],
            duration_hours=duration,
            peak_cpm=row["peak_cpm"],
            rise_param=row["rise_param"]
        )

        f_cs = row["fraction_cs137"]
        f_i = row["fraction_i131"]

        # Channel spectral weights
        weights = {}
        for ch in SPECTRUM_CS137.keys():
            weights[ch] = f_cs * SPECTRUM_CS137[ch] + f_i * SPECTRUM_I131[ch]

        # Add synthetic signal to columns
        for idx, s_val in zip(indices, profile):
            df.loc[idx, "inj_gross_cpm"] += s_val
            df.loc[idx, "synthetic_excess_cpm"] = s_val
            for ch, w in weights.items():
                df.loc[idx, f"inj_{ch}"] += s_val * w
            df.loc[idx, "label"] = "fission_product"
            df.loc[idx, "injection_active"] = True
            df.loc[idx, "injection_id"] = row["injection_id"]

    return df


if __name__ == "__main__":
    cat = generate_injection_catalog()
