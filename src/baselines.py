"""
Stronger Non-ML, Subspace, Unsupervised, and Simple Supervised Baselines.

Implements eight baseline detector families (B1 through B8) under the exact same causal
protocol as the GBDT tiers:
  - Per-station parameters fitted strictly on the 2017-2020 fit fold (`dt < 2021-01-01`).
  - Single scalar anomaly score per hour so each detector can be thresholded on the
    2021-2022 validation split (either by target detection rate or by false-alarm budget)
    and evaluated on the held-out 2023-2025 test split.

Detector Families:
  - B1a (`B1a_Rolling168h_Z`): Existing local 168-hour rolling gross count z-score (`z_score_168h`).
  - B1b (`B1b_GlobalDry_Z`): Gross count z-score against 2017-2020 station dry baseline.
  - B2a (`B2a_RainMasked_Holdoff6h`): Global dry z-score suppressed during rain and for H=6h after rain.
  - B2b (`B2b_RainMasked_Holdoff12h`): Global dry z-score suppressed during rain and for H=12h after rain.
  - B3a (`B3a_WeatherRegression_Linear`): Per-station linear regression of gross CPM & dose rate on
    precipitation history (1, 3, 6, 24h sums, hours_since_rain), pressure, temperature, and cyclical
    hour-of-day fitted on 2017-2020 clean background (rain + dry); score = standardized residual.
  - B3b (`B3b_WeatherRegression_SplineGAM`): Per-station GAM-style B-spline regression (`SplineTransformer`
    + Ridge) of gross CPM & dose rate on precipitation history, pressure, temperature, and hour-of-day
    fitted on 2017-2020 clean background; score = standardized residual.
  - B4a (`B4a_RadonProjection_Global`): Non-ML orthogonal residual norm from projecting 8-channel
    global-dry excess onto the 2017-2020 station radon washout basis (`radon_stripping_residual_l2`).
  - B4b (`B4b_RadonProjection_Local168h`): Non-ML orthogonal residual norm from projecting 8-channel
    local-168h excess onto the 2017-2020 station radon washout basis (`local_radon_stripping_residual_l2`).
  - B5a (`B5a_PCA_Subspace_Mahalanobis`): Per-station PCA subspace model fitted on 2017-2020 clean
    background 8-channel counts and spectral shares (zero synthetic plumes); score = combined
    standardized reconstruction error + subspace Mahalanobis distance.
  - B5b (`B5b_NMF_Background_Reconstruction`): Per-station 2-component Non-Negative Matrix Factorization
    (NMF) fitted on 2017-2020 clean background 8-channel count rates (capturing dry background + natural
    radon washout); score = channel-noise-weighted L2 reconstruction residual.
  - B6a (`B6a_IsolationForest_Tier3_BgOnly`): Unsupervised Isolation Forest trained strictly on 2017-2020
    clean background using Tier 3 features (zero synthetic plumes).
  - B6b (`B6b_IsolationForest_Tier4_BgOnly`): Unsupervised Isolation Forest trained strictly on 2017-2020
    clean background using Tier 4 features (zero synthetic plumes).
  - B6c (`B6c_OneClassSVM_Tier4_BgOnly`): Unsupervised One-Class SVM (RBF kernel via Nystroem + SGDOneClassSVM)
    trained strictly on 2017-2020 clean background using Tier 4 features (zero synthetic plumes).
  - B7a (`B7a_LogisticRegression_Tier3`): Supervised L2-regularized Logistic Regression trained on
    2017-2020 labeled data (with synthetic training plumes) using Tier 3 features.
  - B7b (`B7b_DecisionTreeDepth3_Tier3`): Supervised depth-3 Decision Tree trained on 2017-2020
    labeled data (with synthetic training plumes) using Tier 3 features.
  - B8 (`B8_Operational_Screening_Approx_UNVERIFIED`): Generic approximation of EPA RadNet near-real-time
    baseline deviation screening combining 2017-2020 station-standardized gross CPM and dose rate z-scores
    minus radon high-energy share excess. Exact numerical EPA RadNet alerting thresholds are not publicly
    specified (unpublished; see https://www.epa.gov/radnet/radnet-near-real-time-air-monitoring-results,
    retrieved 2026-10-31).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge, SGDOneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.feature_engineering import TIER_3_FEATURES, TIER_4_NUCLEAR_FEATURES

CHANNELS = [f"cpm_r0{i}" for i in range(2, 10)]
SHARE_COLS = [f"share_r0{i}" for i in range(2, 10)]

WEATHER_REG_COLS = [
    "precip_1h_mm",
    "precip_3h_mm",
    "precip_6h_mm",
    "precip_24h_mm",
    "hours_since_rain",
    "pressure_hpa",
    "temp_c",
]


def _extract_raw_channel_matrix(df: pd.DataFrame, use_injected: bool) -> np.ndarray:
    """Extracts (N, 8) matrix of channel count rates R02..R09 (non-negative)."""
    prefix = "inj_" if use_injected else ""
    cols = [f"{prefix}{ch}" if f"{prefix}{ch}" in df.columns else ch for ch in CHANNELS]
    mat = df[cols].to_numpy(dtype=float)
    return np.nan_to_num(np.maximum(0.0, mat), nan=0.0)


def _extract_gross_and_dose(df: pd.DataFrame, use_injected: bool) -> tuple[np.ndarray, np.ndarray]:
    """Extracts gross_cpm and dose_rate_nsvh arrays."""
    g_col = "inj_gross_cpm" if (use_injected and "inj_gross_cpm" in df.columns) else "gross_cpm"
    d_col = "inj_dose_rate_nsvh" if (use_injected and "inj_dose_rate_nsvh" in df.columns) else "dose_rate_nsvh"
    gross = np.nan_to_num(df[g_col].to_numpy(dtype=float), nan=0.0)
    dose = np.nan_to_num(df[d_col].to_numpy(dtype=float), nan=0.0)
    return gross, dose


def _build_weather_design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Builds weather + cyclical hour-of-day design matrix for B3 regression baselines."""
    dt = pd.to_datetime(df["dt"])
    hour = dt.dt.hour.to_numpy(dtype=float)
    sin_h = np.sin(2.0 * np.pi * hour / 24.0)
    cos_h = np.cos(2.0 * np.pi * hour / 24.0)
    sin_2h = np.sin(4.0 * np.pi * hour / 24.0)
    cos_2h = np.cos(4.0 * np.pi * hour / 24.0)

    w_cols = []
    for c in WEATHER_REG_COLS:
        arr = df[c].to_numpy(dtype=float) if c in df.columns else np.zeros(len(df), dtype=float)
        if c == "hours_since_rain":
            arr = np.minimum(arr, 168.0)
            w_cols.append(np.exp(-arr / 3.0))
            w_cols.append(np.exp(-arr / 12.0))
        w_cols.append(arr)
    w_cols.extend([sin_h, cos_h, sin_2h, cos_2h])
    return np.column_stack(w_cols)


# ---------------------------------------------------------------------------
# B1 & B2: Rolling-168h Z, Global Dry Z (2017-2020 fit), and Rain-Masked Rules
# ---------------------------------------------------------------------------
@dataclass
class B1B2StationStats:
    mu_dry_gross: float
    sigma_dry_gross: float
    mu_all_gross: float
    sigma_all_gross: float
    mu_all_dose: float
    sigma_all_dose: float
    mu_hi_share: float
    sigma_hi_share: float


def fit_b1_b2_b8_station_params(fit_clean_df: pd.DataFrame) -> B1B2StationStats:
    """Fits per-station gross, dose, and high-energy share parameters strictly on 2017-2020 clean rows."""
    valid = (fit_clean_df["has_radnet_obs"].astype(bool)) & (fit_clean_df["rad_complete_channels"].astype(bool))
    sub = fit_clean_df.loc[valid]
    p24_col = "precip_24h_mm" if "precip_24h_mm" in sub.columns else "precip_24h"
    p1_col = "precip_1h_mm" if "precip_1h_mm" in sub.columns else "precip_1h"
    dry = sub[(sub[p24_col].fillna(0.0) == 0.0) & (sub[p1_col].fillna(0.0) == 0.0)]
    if len(dry) < 50:
        dry = sub

    gross_all = sub["gross_cpm"].to_numpy(dtype=float)
    gross_dry = dry["gross_cpm"].to_numpy(dtype=float)
    dose_all = sub["dose_rate_nsvh"].to_numpy(dtype=float)
    hi_share = sub["share_high_energy"].to_numpy(dtype=float)

    return B1B2StationStats(
        mu_dry_gross=float(np.nanmean(gross_dry)),
        sigma_dry_gross=max(1.0, float(np.nanstd(gross_dry))),
        mu_all_gross=float(np.nanmean(gross_all)),
        sigma_all_gross=max(1.0, float(np.nanstd(gross_all))),
        mu_all_dose=float(np.nanmean(dose_all)),
        sigma_all_dose=max(0.5, float(np.nanstd(dose_all))),
        mu_hi_share=float(np.nanmean(hi_share)),
        sigma_hi_share=max(1e-4, float(np.nanstd(hi_share))),
    )


def score_b1a_rolling168h_z(df: pd.DataFrame) -> np.ndarray:
    """B1a: Local 168-hour rolling gross count z-score."""
    return np.nan_to_num(df["z_score_168h"].to_numpy(dtype=float), nan=-10.0)


def score_b1b_global_dry_z(df: pd.DataFrame, stats: B1B2StationStats, use_injected: bool = False) -> np.ndarray:
    """B1b: Station dry z-score fitted strictly on 2017-2020 dry background."""
    gross, _ = _extract_gross_and_dose(df, use_injected=use_injected)
    return (gross - stats.mu_dry_gross) / stats.sigma_dry_gross


def score_b2_rain_masked_z(
    df: pd.DataFrame,
    stats: B1B2StationStats,
    holdoff_hours: float = 6.0,
    use_injected: bool = False,
) -> np.ndarray:
    """B2: Global dry z-score suppressed (-10.0) whenever hours_since_rain <= holdoff_hours."""
    z = score_b1b_global_dry_z(df, stats, use_injected=use_injected)
    hsr = df["hours_since_rain"].to_numpy(dtype=float)
    p1_col = "precip_1h_mm" if "precip_1h_mm" in df.columns else "precip_1h"
    p1 = df[p1_col].fillna(0.0).to_numpy(dtype=float)
    masked = (p1 >= 0.1) | (hsr <= float(holdoff_hours))
    out = z.copy()
    out[masked] = -10.0
    return out


# ---------------------------------------------------------------------------
# B3: Per-Station Weather-Regression Residual (Linear & GAM-Style Spline)
# ---------------------------------------------------------------------------
@dataclass
class B3WeatherRegressionModel:
    variant: str  # "linear" or "spline_gam"
    gross_pipeline: Pipeline
    dose_pipeline: Pipeline
    gross_res_std: float
    dose_res_std: float


def fit_b3_weather_regression(
    fit_clean_df: pd.DataFrame,
    variant: str = "linear",
) -> B3WeatherRegressionModel:
    """
    Fits a per-station weather-regression model predicting gross_cpm and dose_rate_nsvh
    from precipitation history (1, 3, 6, 24h sums, hours_since_rain), pressure, temperature,
    and hour-of-day on 2017-2020 clean background (both rain and dry hours).
    """
    valid = (fit_clean_df["has_radnet_obs"].astype(bool)) & (fit_clean_df["rad_complete_channels"].astype(bool))
    sub = fit_clean_df.loc[valid]
    X = _build_weather_design_matrix(sub)
    y_gross, y_dose = _extract_gross_and_dose(sub, use_injected=False)

    if variant == "spline_gam":
        gross_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
            ("ridge", Ridge(alpha=10.0)),
        ])
        dose_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
            ("ridge", Ridge(alpha=10.0)),
        ])
    else:
        gross_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        dose_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])

    gross_pipe.fit(X, y_gross)
    dose_pipe.fit(X, y_dose)

    res_g = y_gross - gross_pipe.predict(X)
    res_d = y_dose - dose_pipe.predict(X)
    return B3WeatherRegressionModel(
        variant=variant,
        gross_pipeline=gross_pipe,
        dose_pipeline=dose_pipe,
        gross_res_std=max(1.0, float(np.std(res_g))),
        dose_res_std=max(0.5, float(np.std(res_d))),
    )


def score_b3_weather_regression(
    df: pd.DataFrame,
    model: B3WeatherRegressionModel,
    use_injected: bool = False,
) -> np.ndarray:
    """Computes standardized positive weather-regression residual score per hour."""
    X = _build_weather_design_matrix(df)
    y_gross, y_dose = _extract_gross_and_dose(df, use_injected=use_injected)
    z_gross = (y_gross - model.gross_pipeline.predict(X)) / model.gross_res_std
    z_dose = (y_dose - model.dose_pipeline.predict(X)) / model.dose_res_std
    return 0.7 * z_gross + 0.3 * z_dose


# ---------------------------------------------------------------------------
# B4: Radon-Direction Projection (Non-ML Orthogonal Stripping Residual Norm)
# ---------------------------------------------------------------------------
def score_b4a_radon_projection_global(df: pd.DataFrame) -> np.ndarray:
    """
    B4a: Non-ML radon-direction projection residual norm against 2017-2020 global dry
    channel baseline and 2017-2020 station radon washout basis (`radon_stripping_residual_l2`).
    """
    return np.nan_to_num(df["radon_stripping_residual_l2"].to_numpy(dtype=float), nan=0.0)


def score_b4b_radon_projection_local168h(df: pd.DataFrame) -> np.ndarray:
    """
    B4b: Non-ML radon-direction projection residual norm against causal 168-hour local
    channel baseline and 2017-2020 station radon washout basis (`local_radon_stripping_residual_l2`).
    """
    return np.nan_to_num(df["local_radon_stripping_residual_l2"].to_numpy(dtype=float), nan=0.0)


def score_b4c_radon_projection_combined(df: pd.DataFrame) -> np.ndarray:
    """
    B4c: Non-ML radon-direction projection combining the 6-hour smoothed local-168h
    orthogonal residual norm with the peak stripped photopeak channel z-score
    (`max(local_stripped_r03_z, local_stripped_r05_z, local_stripped_r07_z)`).
    """
    l2_raw = pd.Series(np.nan_to_num(df["local_radon_stripping_residual_l2"].to_numpy(dtype=float), nan=0.0))
    l2_6h = l2_raw.rolling(6, min_periods=1).mean().to_numpy(dtype=float)
    r3 = np.nan_to_num(df["local_stripped_r03_z"].to_numpy(dtype=float), nan=0.0)
    r5 = np.nan_to_num(df["local_stripped_r05_z"].to_numpy(dtype=float), nan=0.0)
    r7 = np.nan_to_num(df["local_stripped_r07_z"].to_numpy(dtype=float), nan=0.0)
    pk = pd.Series(np.maximum.reduce([r3, r5, r7])).rolling(6, min_periods=1).mean().to_numpy(dtype=float)
    return 0.5 * l2_6h + 0.5 * np.maximum(0.0, pk)


# ---------------------------------------------------------------------------
# B5: Background-Only Subspace Models (PCA Mahalanobis & NMF Reconstruction)
# ---------------------------------------------------------------------------
@dataclass
class B5PCASubspaceModel:
    scaler: StandardScaler
    pca: PCA
    comp_std: np.ndarray
    recon_err_mean: float
    recon_err_std: float


def fit_b5a_pca_subspace(fit_clean_df: pd.DataFrame, n_components: int = 3) -> B5PCASubspaceModel:
    """
    Fits a per-station PCA subspace model strictly on 2017-2020 clean background
    8-channel count excesses (local-168h subtracted) and spectral shares (16 inputs).
    No synthetic plumes are used.
    """
    valid = (fit_clean_df["has_radnet_obs"].astype(bool)) & (fit_clean_df["rad_complete_channels"].astype(bool))
    sub = fit_clean_df.loc[valid]
    ch_mat = _extract_raw_channel_matrix(sub, use_injected=False)
    sh_mat = np.nan_to_num(sub[SHARE_COLS].to_numpy(dtype=float), nan=0.0)
    X = np.hstack([ch_mat, sh_mat])

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X_std)
    X_hat = pca.inverse_transform(Z)

    # Positive orthogonal reconstruction error (penalizes only excesses above reconstruction)
    pos_err = np.maximum(0.0, X_std - X_hat)
    recon_err = np.sqrt(np.sum(pos_err ** 2, axis=1))
    comp_std = np.maximum(1e-4, np.std(Z, axis=0))

    return B5PCASubspaceModel(
        scaler=scaler,
        pca=pca,
        comp_std=comp_std,
        recon_err_mean=float(np.mean(recon_err)),
        recon_err_std=max(1e-4, float(np.std(recon_err))),
    )


def score_b5a_pca_subspace(
    df: pd.DataFrame,
    model: B5PCASubspaceModel,
    use_injected: bool = False,
) -> np.ndarray:
    """Scores each hour by standardized PCA reconstruction error + subspace Mahalanobis norm."""
    ch_mat = _extract_raw_channel_matrix(df, use_injected=use_injected)
    sh_mat = np.nan_to_num(df[SHARE_COLS].to_numpy(dtype=float), nan=0.0)
    X = np.hstack([ch_mat, sh_mat])
    X_std = model.scaler.transform(X)
    Z = model.pca.transform(X_std)
    X_hat = model.pca.inverse_transform(Z)

    pos_err = np.maximum(0.0, X_std - X_hat)
    recon_err = np.sqrt(np.sum(pos_err ** 2, axis=1))
    z_recon = (recon_err - model.recon_err_mean) / model.recon_err_std
    mahal = np.sqrt(np.sum((Z / model.comp_std[None, :]) ** 2, axis=1))
    return 0.85 * z_recon + 0.15 * mahal


@dataclass
class B5NMFSubspaceModel:
    nmf: NMF
    ch_scale: np.ndarray
    res_mean: float
    res_std: float


def fit_b5b_nmf_subspace(fit_clean_df: pd.DataFrame, n_components: int = 2) -> B5NMFSubspaceModel:
    """
    Fits a 2-component Non-Negative Matrix Factorization (NMF) model strictly on
    2017-2020 clean background 8-channel count rates R02..R09 (capturing the dry
    background spectrum and natural radon washout spectrum without synthetic plumes).
    """
    valid = (fit_clean_df["has_radnet_obs"].astype(bool)) & (fit_clean_df["rad_complete_channels"].astype(bool))
    sub = fit_clean_df.loc[valid]
    ch_mat = _extract_raw_channel_matrix(sub, use_injected=False)
    ch_scale = np.maximum(1.0, np.std(ch_mat, axis=0))
    X_pos = np.maximum(0.0, ch_mat / ch_scale[None, :])

    nmf = NMF(n_components=n_components, init="nndsvda", max_iter=400, random_state=42)
    W = nmf.fit_transform(X_pos)
    X_hat = W @ nmf.components_
    pos_res = np.maximum(0.0, X_pos - X_hat)
    res_norm = np.sqrt(np.sum(pos_res ** 2, axis=1))

    return B5NMFSubspaceModel(
        nmf=nmf,
        ch_scale=ch_scale,
        res_mean=float(np.mean(res_norm)),
        res_std=max(1e-4, float(np.std(res_norm))),
    )


def score_b5b_nmf_subspace(
    df: pd.DataFrame,
    model: B5NMFSubspaceModel,
    use_injected: bool = False,
) -> np.ndarray:
    """Scores each hour by standardized positive NMF channel reconstruction residual."""
    ch_mat = _extract_raw_channel_matrix(df, use_injected=use_injected)
    X_pos = np.maximum(0.0, ch_mat / model.ch_scale[None, :])
    W = model.nmf.transform(X_pos)
    X_hat = W @ model.nmf.components_
    pos_res = np.maximum(0.0, X_pos - X_hat)
    res_norm = np.sqrt(np.sum(pos_res ** 2, axis=1))
    return (res_norm - model.res_mean) / model.res_std


# ---------------------------------------------------------------------------
# B6: Unsupervised ML on Clean Background Only (2017-2020, Zero Plumes)
# ---------------------------------------------------------------------------
@dataclass
class B6UnsupervisedModel:
    method: str  # "iforest_tier3", "iforest_tier4", "ocsvm_tier4"
    feature_cols: list[str]
    pipeline: Pipeline


def fit_b6_unsupervised_background(
    fit_clean_df: pd.DataFrame,
    method: str = "iforest_tier4",
    seed: int = 42,
) -> B6UnsupervisedModel:
    """
    Fits an unsupervised anomaly detector (Isolation Forest or One-Class SVM)
    strictly on 2017-2020 clean background hours (zero synthetic plumes).
    """
    f_cols = list(TIER_3_FEATURES) if "tier3" in method else list(TIER_4_NUCLEAR_FEATURES)
    valid = (fit_clean_df["has_radnet_obs"].astype(bool)) & (fit_clean_df["rad_complete_channels"].astype(bool))
    sub = fit_clean_df.loc[valid]
    X = sub[f_cols].to_numpy(dtype=float)

    if method.startswith("iforest"):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("iforest", IsolationForest(
                n_estimators=150,
                max_samples=2048,
                contamination="auto",
                random_state=seed,
                n_jobs=4,
            )),
        ])
    else:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("nystroem", Nystroem(kernel="rbf", gamma=1.0 / len(f_cols), n_components=150, random_state=seed)),
            ("ocsvm", SGDOneClassSVM(nu=0.02, random_state=seed, max_iter=1000, tol=1e-3)),
        ])

    pipe.fit(X)
    return B6UnsupervisedModel(method=method, feature_cols=f_cols, pipeline=pipe)


def score_b6_unsupervised(df: pd.DataFrame, model: B6UnsupervisedModel) -> np.ndarray:
    """Returns continuous anomaly score (higher = more anomalous) for B6 models."""
    X = df[model.feature_cols].to_numpy(dtype=float)
    # decision_function returns positive for inliers and negative for outliers; negate it
    return -model.pipeline.decision_function(X)


# ---------------------------------------------------------------------------
# B7: Simple Supervised Models on Same 2017-2020 Training Plumes (Tier 3)
# ---------------------------------------------------------------------------
@dataclass
class B7SimpleSupervisedModel:
    method: str  # "logreg_tier3" or "dtree_depth3_tier3"
    feature_cols: list[str]
    pipeline: Pipeline
    fission_class_idx: int


def fit_b7_simple_supervised(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    method: str = "logreg_tier3",
    feature_cols: list[str] | None = None,
    seed: int = 42,
) -> B7SimpleSupervisedModel:
    """
    Fits a simple supervised baseline (L2 Logistic Regression or Depth-3 Decision Tree)
    on the 2017-2020 fit fold with the same training plumes as the GBDT tiers.
    """
    f_cols = list(feature_cols) if feature_cols is not None else list(TIER_3_FEATURES)
    X = X_train[f_cols].to_numpy(dtype=float)

    if method == "logreg_tier3":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=500,
                random_state=seed,
            )),
        ])
    else:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=50,
                class_weight="balanced",
                random_state=seed,
            )),
        ])

    pipe.fit(X, y_train)
    classes = list(pipe.named_steps["clf"].classes_)
    fiss_idx = classes.index(2) if 2 in classes else len(classes) - 1
    return B7SimpleSupervisedModel(
        method=method,
        feature_cols=f_cols,
        pipeline=pipe,
        fission_class_idx=fiss_idx,
    )


def score_b7_simple_supervised(df: pd.DataFrame, model: B7SimpleSupervisedModel) -> np.ndarray:
    """Returns predicted probability of fission_product (class 2) for B7 models."""
    X = df[model.feature_cols].to_numpy(dtype=float)
    probs = model.pipeline.predict_proba(X)
    return probs[:, model.fission_class_idx]


# ---------------------------------------------------------------------------
# B8: Operational Screening Rule Approximation (approximate exact EPA threshold)
# ---------------------------------------------------------------------------
EPA_RADNET_ALERT_DOCUMENTATION_NOTE = (
    "Public EPA RadNet documentation (https://www.epa.gov/radnet/radnet-near-real-time-air-monitoring-results "
    "and https://www.epa.gov/radnet/about-radnet-air-monitoring-data, retrieved 2026-10-31) states that near-real-time "
    "gross gamma count rates and gamma dose rates are automatically screened for deviations outside expected station "
    "baseline fluctuations followed by analyst review for weather/radon washout, but does not publish a numerical "
    "sigma or count-rate threshold. B8 implements the closest documented generic baseline-deviation screening approximation."
)


def score_b8_operational_screening_approx(
    df: pd.DataFrame,
    stats: B1B2StationStats,
    use_injected: bool = False,
) -> np.ndarray:
    """
    B8: Approximate operational screening rule (uncalibrated exact EPA numerical threshold).
    Combines 2017-2020 station baseline gross count z-score and ambient dose rate z-score,
    penalized by positive high-energy (R07+R08) natural radon share excess.
    """
    gross, dose = _extract_gross_and_dose(df, use_injected=use_injected)
    z_gross = (gross - stats.mu_all_gross) / stats.sigma_all_gross
    z_dose = (dose - stats.mu_all_dose) / stats.sigma_all_dose
    hi_share = np.nan_to_num(df["share_high_energy"].to_numpy(dtype=float), nan=stats.mu_hi_share)
    z_hi_radon = np.maximum(0.0, (hi_share - stats.mu_hi_share) / stats.sigma_hi_share)
    return 0.5 * z_gross + 0.5 * z_dose - 0.5 * z_hi_radon
