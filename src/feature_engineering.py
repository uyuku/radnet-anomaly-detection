"""
Phase 4 Feature Engineering Pipeline: Three-Tier Hierarchy for Anomaly Detection.

Constructs strictly causal (no forward-looking leakage) feature representations
on the continuous hourly calendar grid for both clean and injected series:

- Tier 1: Gross radiation features (rolling means, diffs, Z-scores, dose rate).
- Tier 2: Radiation + NaI(Tl) spectral energy shares & physical photopeak ratios.
- Tier 3: Radiation + spectral + multi-scale NOAA weather fusion & cross-domain interactions.

All rolling and differential operations look strictly backward in time (t-k, k >= 0).
"""

from pathlib import Path
import pandas as pd
import numpy as np


TIER_1_FEATURES = [
    "z_score_168h",
    "z_score_24h",
    "z_score_global_dry",
    "ratio_to_168h",
    "ratio_to_24h",
    "rel_diff_1h",
    "rel_diff_3h",
    "rel_diff_6h",
    "std_ratio_24_to_168",
    "dose_present",
    "dose_z_score_168h",
    "dose_to_gross_ratio",
]

WEATHER_FEATURES = [
    "precip_1h_mm",
    "precip_3h_mm",
    "precip_6h_mm",
    "precip_24h_mm",
    "is_raining",
    "rain_recent_3h",
    "rain_recent_6h",
    "rain_recent_24h",
    "hours_since_rain",
    "pressure_hpa",
    "pressure_diff_3h",
    "pressure_diff_24h",
    "temp_c",
    "dewpoint_c",
    "rel_humidity_pct",
]

TIER_1B_FEATURES = TIER_1_FEATURES + WEATHER_FEATURES + [
    "washout_expected_ratio",
    "dry_excess_interaction",
]

TIER_2_FEATURES = TIER_1_FEATURES + [
    "share_r02",
    "share_r03",
    "share_r04",
    "share_r05",
    "share_r06",
    "share_r07",
    "share_r08",
    "share_r09",
    "share_high_energy",
    "ratio_r05_r07",
    "ratio_r03_r07",
    "ratio_r05_r03",
    "ratio_r02_r03",
    "share_r05_roll_6h",
    "share_r07_roll_6h",
    "share_high_roll_6h",
    "diff_share_r05_3h",
    "diff_share_r07_3h",
]

TIER_3_FEATURES = TIER_2_FEATURES + WEATHER_FEATURES + [
    "rain_high_energy_interaction",
    "dry_excess_interaction",
    "washout_expected_ratio",
]

# Load training-only dry baselines (2017-2022: training + validation years) to eliminate test leakage.
# Computed strictly on dry background hours prior to the 2023-2025 test split.
import json

_BASELINES_FILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "station_dry_baselines_train_only.json"
if not _BASELINES_FILE.exists():
    raise FileNotFoundError(
        f"Required training-only baseline file not found: {_BASELINES_FILE}. "
        "Run compute_train_only_dry_baselines() to generate."
    )

with open(_BASELINES_FILE, "r") as _f:
    STATION_DRY_BASELINES = json.load(_f)


def compute_features_for_series(df_input: pd.DataFrame, station_id: str, use_injected: bool = True) -> pd.DataFrame:
    """
    Computes all Tier 1, 2, and 3 features for a given station dataframe.
    
    Parameters
    ----------
    df_input : pd.DataFrame
        Continuous calendar grid dataframe loaded from labeled_{station}_{split}.csv.gz.
    station_id : str
        Station identifier (e.g. 'al_birmingham').
    use_injected : bool
        If True, features are computed on injected radiation columns ('inj_gross_cpm', etc.).
        If False, features are computed on clean background columns ('gross_cpm', etc.).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with computed feature columns, preserved labels, and metadata.
    """
    df = df_input.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["dt"]):
        df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    # 1. Select Radiation Sources (Clean vs Injected)
    if use_injected:
        gross = df["inj_gross_cpm"].astype(float)
        dose = df["inj_dose_rate_nsvh"].astype(float)
        r02 = df["inj_cpm_r02"].astype(float)
        r03 = df["inj_cpm_r03"].astype(float)
        r04 = df["inj_cpm_r04"].astype(float)
        r05 = df["inj_cpm_r05"].astype(float)
        r06 = df["inj_cpm_r06"].astype(float)
        r07 = df["inj_cpm_r07"].astype(float)
        r08 = df["inj_cpm_r08"].astype(float)
        r09 = df["inj_cpm_r09"].astype(float)
    else:
        gross = df["gross_cpm"].astype(float)
        dose = df["dose_rate_nsvh"].astype(float)
        r02 = df["cpm_r02"].astype(float)
        r03 = df["cpm_r03"].astype(float)
        r04 = df["cpm_r04"].astype(float)
        r05 = df["cpm_r05"].astype(float)
        r06 = df["cpm_r06"].astype(float)
        r07 = df["cpm_r07"].astype(float)
        r08 = df["cpm_r08"].astype(float)
        r09 = df["cpm_r09"].astype(float)

    try:
        st_base = STATION_DRY_BASELINES[station_id.lower()]
    except KeyError:
        # BUG-11 fix (2026-10-06): never fabricate baselines silently.
        raise KeyError(
            f"No train-only dry baseline for station '{station_id}'. "
            f"Known stations: {sorted(STATION_DRY_BASELINES)}"
        )
    mu_dry = st_base["mu"]
    sigma_dry = st_base["sigma"]
    mu_dose_dry = st_base["mu_dose"]
    sigma_dose_dry = st_base["sigma_dose"]

    # -------------------------------------------------------------
    # Tier 1: Gross Radiation & Dose Features
    # -------------------------------------------------------------
    roll_168_mean = gross.rolling(168, min_periods=24).mean()
    roll_168_std = gross.rolling(168, min_periods=24).std()
    roll_24_mean = gross.rolling(24, min_periods=6).mean()
    roll_24_std = gross.rolling(24, min_periods=6).std()

    feats = pd.DataFrame(index=df.index)
    feats["z_score_168h"] = (gross - roll_168_mean) / (roll_168_std + 1e-4)
    feats["z_score_24h"] = (gross - roll_24_mean) / (roll_24_std + 1e-4)
    feats["z_score_global_dry"] = (gross - mu_dry) / sigma_dry
    feats["ratio_to_168h"] = gross / (roll_168_mean + 1e-4)
    feats["ratio_to_24h"] = gross / (roll_24_mean + 1e-4)
    feats["rel_diff_1h"] = (gross - gross.shift(1)) / (roll_168_mean + 1e-4)
    feats["rel_diff_3h"] = (gross - gross.shift(3)) / (roll_168_mean + 1e-4)
    feats["rel_diff_6h"] = (gross - gross.shift(6)) / (roll_168_mean + 1e-4)
    feats["std_ratio_24_to_168"] = roll_24_std / (roll_168_std + 1e-4)

    # Ambient Dose Rate Features
    has_dose = dose.notna() & (dose > 0)
    feats["dose_present"] = has_dose.astype(float)
    roll_dose_168_mean = dose.rolling(168, min_periods=24).mean()
    roll_dose_168_std = dose.rolling(168, min_periods=24).std()
    feats["dose_z_score_168h"] = np.where(
        has_dose,
        (dose - roll_dose_168_mean) / (roll_dose_168_std + 1e-4),
        0.0
    )
    feats["dose_to_gross_ratio"] = np.where(
        has_dose,
        dose / (gross + 1e-4),
        mu_dose_dry / mu_dry
    )

    # -------------------------------------------------------------
    # Tier 2: Spectral Shares & Photopeak Ratios
    # -------------------------------------------------------------
    gross_denom = gross.replace(0.0, np.nan)
    feats["share_r02"] = r02 / gross_denom
    feats["share_r03"] = r03 / gross_denom
    feats["share_r04"] = r04 / gross_denom
    feats["share_r05"] = r05 / gross_denom
    feats["share_r06"] = r06 / gross_denom
    feats["share_r07"] = r07 / gross_denom
    feats["share_r08"] = r08 / gross_denom
    feats["share_r09"] = r09 / gross_denom
    feats["share_high_energy"] = (r07 + r08) / gross_denom

    feats["ratio_r05_r07"] = r05 / (r07 + 1e-4)
    feats["ratio_r03_r07"] = r03 / (r07 + 1e-4)
    feats["ratio_r05_r03"] = r05 / (r03 + 1e-4)
    feats["ratio_r02_r03"] = r02 / (r03 + 1e-4)

    feats["share_r05_roll_6h"] = feats["share_r05"].rolling(6, min_periods=2).mean()
    feats["share_r07_roll_6h"] = feats["share_r07"].rolling(6, min_periods=2).mean()
    feats["share_high_roll_6h"] = feats["share_high_energy"].rolling(6, min_periods=2).mean()
    feats["diff_share_r05_3h"] = feats["share_r05"] - feats["share_r05"].shift(3)
    feats["diff_share_r07_3h"] = feats["share_r07"] - feats["share_r07"].shift(3)

    # -------------------------------------------------------------
    # Tier 3: Weather Fusion & Interaction Features
    # -------------------------------------------------------------
    precip_1h = df["precip_1h_mm"].fillna(0.0).astype(float)
    feats["precip_1h_mm"] = precip_1h
    feats["precip_3h_mm"] = precip_1h.rolling(3, min_periods=1).sum()
    feats["precip_6h_mm"] = precip_1h.rolling(6, min_periods=1).sum()
    feats["precip_24h_mm"] = precip_1h.rolling(24, min_periods=1).sum()

    feats["is_raining"] = (precip_1h >= 0.1).astype(float)
    feats["rain_recent_3h"] = (feats["precip_3h_mm"] >= 0.1).astype(float)
    feats["rain_recent_6h"] = (feats["precip_6h_mm"] >= 0.1).astype(float)
    feats["rain_recent_24h"] = (feats["precip_24h_mm"] >= 0.1).astype(float)

    # Vectorized hours since last precipitation
    is_rain = precip_1h >= 0.1
    last_rain_dt = df["dt"].where(is_rain).ffill()
    hrs_since_rain = (df["dt"] - last_rain_dt).dt.total_seconds() / 3600.0
    feats["hours_since_rain"] = hrs_since_rain.clip(lower=0.0, upper=168.0).fillna(168.0)

    # Pressure & tendencies
    pressure = df["pressure_hpa"].astype(float)
    feats["pressure_hpa"] = pressure
    feats["pressure_diff_3h"] = pressure - pressure.shift(3)
    feats["pressure_diff_24h"] = pressure - pressure.shift(24)

    # Thermodynamics
    feats["temp_c"] = df["temp_c"].astype(float)
    feats["dewpoint_c"] = df["dewpoint_c"].astype(float)
    feats["rel_humidity_pct"] = df["rel_humidity_pct"].astype(float)

    # Physical Cross-Domain Interactions
    feats["rain_high_energy_interaction"] = feats["precip_3h_mm"] * feats["share_high_energy"]  # NaN-propagating (BUG-11: removed magic 0.05 fill)
    feats["dry_excess_interaction"] = (feats["precip_6h_mm"] == 0.0).astype(float) * feats["z_score_global_dry"].fillna(0.0)
    feats["washout_expected_ratio"] = feats["z_score_global_dry"] / (np.sqrt(np.maximum(0.0, feats["precip_3h_mm"])) + 0.1)

    # Attach labels and metadata
    feats["label"] = df["label"]
    feats["dt"] = df["dt"]
    feats["station_id"] = station_id
    feats["has_radnet_obs"] = df["has_radnet_obs"].fillna(False).astype(bool)
    feats["rad_complete_channels"] = df["rad_complete_channels"].fillna(False).astype(bool)
    feats["injection_active"] = df["injection_active"].fillna(False).astype(bool)
    if "injection_id" in df.columns:
        feats["injection_id"] = df["injection_id"]
    if "synthetic_excess_cpm" in df.columns:
        feats["synthetic_excess_cpm"] = df["synthetic_excess_cpm"].fillna(0.0)

    return feats
