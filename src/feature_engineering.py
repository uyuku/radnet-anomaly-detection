"""
Feature Engineering Pipeline: Physics-Informed Hierarchy for Anomaly Detection.

Constructs strictly causal (no forward-looking leakage) feature representations
on the continuous hourly calendar grid for both clean and injected series:

- Tier 1: Gross radiation features (rolling means, diffs, Z-scores, dose rate).
- Tier 2: Radiation + NaI(Tl) spectral energy shares & physical photopeak ratios.
- Tier 3 (Schema 2): Radiation + spectral + multi-scale NOAA weather fusion & cross-domain interactions (48 features).
- Tier 4 / Physics-Informed Nuclear (Schema 3): Extends Tier 3 with causal 8-channel
  Radon Spectral Stripping (unmixing s_Rn from global dry and local 168h rolling baselines),
  coupled 3-stage Bateman radon progeny washout ODE residuals (Po-218 -> Pb-214 -> Bi-214),
  post-rain persistence anomalies, and channel-resolved energy-weighted dose consistency.

All rolling, differential, and ODE operations look strictly backward in time (t-k, k >= 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.nuclear_physics import (
    DEFAULT_RADON_BASIS,
    compute_channel_dose_response_vector,
    compute_spectral_stripping,
    solve_bateman_washout_ode,
)

FEATURE_SCHEMA_VERSION = 3


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

# Preserved at 48 features for Schema 2 backward compatibility and head-to-head benchmarking
TIER_3_FEATURES = TIER_2_FEATURES + WEATHER_FEATURES + [
    "rain_high_energy_interaction",
    "dry_excess_interaction",
    "washout_expected_ratio",
]

NUCLEAR_PHYSICS_FEATURES = [
    "radon_stripping_residual_l2",
    "stripped_excess_r03_z",
    "stripped_excess_r05_z",
    "stripped_excess_r07_z",
    "local_stripped_r03_z",
    "local_stripped_r05_z",
    "local_stripped_r07_z",
    "local_radon_stripping_residual_l2",
    "bateman_washout_residual_ratio",
    "post_rain_persistence_anomaly",
    "dose_spectral_residual_z",
]

TIER_4_NUCLEAR_FEATURES = TIER_3_FEATURES + NUCLEAR_PHYSICS_FEATURES

_BASELINES_FILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "station_dry_baselines_train_only.json"
if not _BASELINES_FILE.exists():
    raise FileNotFoundError(
        f"Required training-only baseline file not found: {_BASELINES_FILE}. "
        "Run: python src/compute_train_only_baselines.py"
    )

with open(_BASELINES_FILE, "r") as _f:
    STATION_DRY_BASELINES = json.load(_f)


def compute_features_for_series(df_input: pd.DataFrame, station_id: str, use_injected: bool = True) -> pd.DataFrame:
    """
    Computes all Tier 1, 2, 3, and 4 (Physics-Informed Nuclear) features for a given station dataframe.

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
    df = df_input.copy().reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_localize(None)
    if df.empty or df["dt"].isna().any():
        raise ValueError("Feature input must contain non-missing hourly UTC timestamps.")
    if not df["dt"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Feature input must be a sorted, unique, continuous hourly UTC grid. "
            "Reindex missing hours before computing features; do not drop outage rows."
        )

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
        raise KeyError(
            f"No train-only dry baseline for station '{station_id}'. "
            f"Known stations: {sorted(STATION_DRY_BASELINES)}"
        )
    mu_dry = float(st_base["mu"])
    sigma_dry = float(st_base["sigma"])
    mu_dose_dry = float(st_base["mu_dose"])
    sigma_dose_dry = float(st_base.get("sigma_dose", 2.5))

    s_rn = np.asarray(st_base.get("radon_washout_basis", DEFAULT_RADON_BASIS), dtype=float)
    s_rn = np.maximum(1e-6, s_rn)
    s_rn = s_rn / np.sum(s_rn)

    if "channel_mu_dry" in st_base:
        ch_mu_dry = np.asarray(st_base["channel_mu_dry"], dtype=float)
    else:
        ch_mu_dry = mu_dry * s_rn

    if "channel_sigma_dry" in st_base:
        ch_sigma_dry = np.maximum(1.0, np.asarray(st_base["channel_sigma_dry"], dtype=float))
    else:
        ch_sigma_dry = np.maximum(1.0, sigma_dry * s_rn)

    k_dose_slope = float(st_base.get("k_dose_slope", 0.01344))
    if "k_dose_channel" in st_base:
        k_dose_vec = np.asarray(st_base["k_dose_channel"], dtype=float)
    else:
        k_dose_vec = compute_channel_dose_response_vector(k_dose_slope, s_rn)

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
        0.0,
    )
    feats["dose_to_gross_ratio"] = np.where(
        has_dose,
        dose / (gross + 1e-4),
        mu_dose_dry / mu_dry,
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
    precip_1h = df["precip_1h_mm"].astype(float)
    feats["precip_1h_mm"] = precip_1h
    for hours in (3, 6, 24):
        feats[f"precip_{hours}h_mm"] = precip_1h.rolling(hours, min_periods=hours).sum()

    feats["is_raining"] = precip_1h.ge(0.1).astype(float).where(precip_1h.notna())
    for hours in (3, 6, 24):
        total = feats[f"precip_{hours}h_mm"]
        feats[f"rain_recent_{hours}h"] = total.ge(0.1).astype(float).where(total.notna())

    # Vectorized hours since last precipitation
    is_rain = precip_1h >= 0.1
    last_rain_dt = df["dt"].where(is_rain).ffill()
    last_unknown_dt = df["dt"].where(precip_1h.isna()).ffill()
    hrs_since_rain = (df["dt"] - last_rain_dt).dt.total_seconds() / 3600.0
    rain_time_known = last_rain_dt.notna() & (
        last_unknown_dt.isna() | last_rain_dt.ge(last_unknown_dt)
    )
    dry_168h = precip_1h.rolling(168, min_periods=168).sum().eq(0.0)
    feats["hours_since_rain"] = hrs_since_rain.clip(0.0, 168.0).where(rain_time_known)
    feats.loc[dry_168h, "hours_since_rain"] = 168.0

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
    feats["rain_high_energy_interaction"] = feats["precip_3h_mm"] * feats["share_high_energy"]
    dry_6h = feats["precip_6h_mm"].eq(0.0).astype(float).where(feats["precip_6h_mm"].notna())
    feats["dry_excess_interaction"] = dry_6h * feats["z_score_global_dry"]
    feats["washout_expected_ratio"] = feats["z_score_global_dry"] / (
        np.sqrt(np.maximum(0.0, feats["precip_3h_mm"])) + 0.1
    )

    # -------------------------------------------------------------
    # Tier 4 (Schema 3): Physics-Informed Spectral Stripping & Bateman Washout ODE
    # -------------------------------------------------------------
    ch_series = [r02, r03, r04, r05, r06, r07, r08, r09]
    c_mat = np.column_stack([s.to_numpy(dtype=float) for s in ch_series])

    # 4A. Global dry-baseline-subtracted spectral stripping
    delta_c_dry = np.maximum(0.0, c_mat - ch_mu_dry[None, :])
    strip_dry = compute_spectral_stripping(delta_c_dry, s_rn, ch_sigma_dry)
    feats["radon_stripping_residual_l2"] = strip_dry["radon_stripping_residual_l2"]
    feats["stripped_excess_r03_z"] = strip_dry["stripped_excess_r03_z"]
    feats["stripped_excess_r05_z"] = strip_dry["stripped_excess_r05_z"]
    feats["stripped_excess_r07_z"] = strip_dry["stripped_excess_r07_z"]

    # 4B. Causal 168h local-baseline-subtracted spectral stripping (station-scale invariant)
    local_means = np.column_stack([
        s.rolling(168, min_periods=24).mean().fillna(ch_mu_dry[i]).to_numpy(dtype=float)
        for i, s in enumerate(ch_series)
    ])
    local_stds = np.column_stack([
        s.rolling(168, min_periods=24).std().fillna(ch_sigma_dry[i]).clip(lower=1.0).to_numpy(dtype=float)
        for i, s in enumerate(ch_series)
    ])
    delta_c_local = np.maximum(0.0, c_mat - local_means)
    strip_local = compute_spectral_stripping(delta_c_local, s_rn, ch_sigma_dry)

    # Normalize local stripped excesses by local 168h channel variability and smooth over 3h causal window
    s_r07_08 = max(1e-6, float(s_rn[5] + s_rn[6]))
    s_r02_03 = max(1e-6, float(s_rn[0] + s_rn[1]))
    ratio_3_to_78 = float(s_rn[1]) / s_r07_08
    ratio_5_to_78 = float(s_rn[3]) / s_r07_08
    ratio_7_to_23 = float(s_rn[5]) / s_r02_03

    r07_08_loc = delta_c_local[:, 5] + delta_c_local[:, 6]
    r02_03_loc = delta_c_local[:, 0] + delta_c_local[:, 1]

    loc_r03_z_raw = pd.Series((delta_c_local[:, 1] - ratio_3_to_78 * r07_08_loc) / (local_stds[:, 1] + 1.0), index=df.index)
    loc_r05_z_raw = pd.Series((delta_c_local[:, 3] - ratio_5_to_78 * r07_08_loc) / (local_stds[:, 3] + 1.0), index=df.index)
    loc_r07_z_raw = pd.Series((delta_c_local[:, 5] - ratio_7_to_23 * r02_03_loc) / (local_stds[:, 5] + 1.0), index=df.index)

    feats["local_stripped_r03_z"] = loc_r03_z_raw.rolling(3, min_periods=1).mean()
    feats["local_stripped_r05_z"] = loc_r05_z_raw.rolling(3, min_periods=1).mean()
    feats["local_stripped_r07_z"] = loc_r07_z_raw.rolling(3, min_periods=1).mean()
    feats["local_radon_stripping_residual_l2"] = pd.Series(
        strip_local["radon_stripping_residual_l2"], index=df.index
    ).rolling(3, min_periods=1).mean()

    # 4C. Coupled 3-stage Bateman Radon Progeny Washout ODE Residuals (strictly NaN-propagating)
    bateman_ode = solve_bateman_washout_ode(precip_1h, beta=0.65, memory_hours=6)
    bateman_idx = pd.Series(bateman_ode["bateman_expected_index"], index=df.index)

    # Ratio of local excess Z-score to theoretical Bateman washout index
    feats["bateman_washout_residual_ratio"] = feats["z_score_168h"] / (bateman_idx + 0.1)

    # Post-rain persistence anomaly:
    # After rain ceases (hours_since_rain in [1, 12]), filter-trapped radon progeny
    # (Pb-214 T_1/2=26.8m, Bi-214 T_1/2=19.9m) decay with effective lifetime tau_eff ~ 0.65 h:
    #   B_decay(dt_dry) = (1 + dt_dry / 0.65) * exp(-dt_dry / 0.65)
    # A persistent positive excess when B_decay -> 0 signals a long-lived fission plume trapped on the filter.
    hrs_rain = feats["hours_since_rain"]
    tau_eff = 0.65
    bateman_clearance_envelope = (1.0 + hrs_rain / tau_eff) * np.exp(-hrs_rain / tau_eff)
    in_post_rain_window = (hrs_rain >= 1.0) & (hrs_rain <= 12.0)
    persistence_weight = np.where(in_post_rain_window, 1.0 - bateman_clearance_envelope, 0.0)
    persistence_anom = persistence_weight * np.maximum(0.0, feats["z_score_168h"]) * feats["local_radon_stripping_residual_l2"]
    feats["post_rain_persistence_anomaly"] = pd.Series(persistence_anom, index=df.index).where(
        hrs_rain.notna() & bateman_idx.notna()
    )

    # 4D. Channel-resolved energy-weighted dose consistency Z-score
    expected_dose_excess_local = delta_c_local @ k_dose_vec
    obs_dose_excess_local = np.maximum(0.0, (dose - roll_dose_168_mean.fillna(mu_dose_dry)).to_numpy(dtype=float))
    dose_denom = roll_dose_168_std.fillna(sigma_dose_dry).clip(lower=0.5).to_numpy(dtype=float)
    feats["dose_spectral_residual_z"] = np.where(
        has_dose,
        (obs_dose_excess_local - expected_dose_excess_local) / dose_denom,
        0.0,
    )

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
