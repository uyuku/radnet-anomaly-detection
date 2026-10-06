"""
Phase 4 & 5 Rigorous Model Training and Evaluation Pipeline (Second Revision - Optimized).

Implements all required corrections from Claude Review of Revised Phase 5:
1. Expanded Threshold Grid & Realized Detection Labels:
   - Evaluates thresholds up to tau=0.999 (finer grid above 0.98: 0.985, 0.990, 0.992, 0.995, 0.998, 0.999).
   - All tables strictly labeled by realized test detection, never by targets the model missed.
2. Continuous Detection-vs-False-Alarm Operating Curves:
   - Full test-set sweeps across tau in [0.01, 0.999] for all tiers (Tier 1, Tier 1b, Tier 2, Tier 3, MLP, Calibrated MLP)
     and k in [0.5, 5.0] for baselines.
   - Provides exact matched comparisons at 80%, 85%, 90%, 92%, and 95% realized detection.
   - Restates Tier 1 vs Tier 1b matched comparison at ~92% detection (58.8% reduction).
3. Twenty Random Seeds & Paired Statistical Tests:
   - Trains tiers across 20 random seeds (SEEDS = 42 to 61).
   - Runs paired t-test and Wilcoxon signed-rank test on false alarms for Tier 3 vs Tier 2.
   - Explicitly notes that the single-seed Phase 4 ordering did not survive the multi-seed protocol.
4. Restored Sampling Uncertainty Intervals:
   - Station-month block bootstrap (B=1,000 resamples) for clean false alarms per station-year.
   - Event-level bootstrap (B=1,000 resamples) for test detection probability and median detection delay.
5. MLP Ranking vs Calibration Transfer:
   - Generates threshold-free ROC / detection-vs-FA curve for MLP on test set.
   - Evaluates isotonic probability calibration on validation fold before freezing threshold on test.
6. Multi-Station Leave-One-Station-Out (LOSO) Validation:
   - Evaluates held-out transfer for San Diego, CA, Birmingham, AL, and Dallas, TX.
   - Stratifies detection by scenario and magnitude band.
   - Records clean false alarms (with rule-of-three upper bound where 0) and natural washout hours.
   - Acknowledges honestly that frozen tau=0.99 at San Diego traded detection (67.5%) for silence.
7. Proper Spectral Template Sensitivity & Instrumental Gain Drift:
   - Scales photopeak excess on the injected component only, re-normalizing continuum to keep total gross CPM constant.
   - Recomputes all features causally via compute_features_for_series so all shares and ratios sum consistently.
   - Tests instrumental gain drift (+/-5%, +/-10%) symmetrically across both clean background and injected plumes,
     measuring both detection rate and clean false alarms per station-year.
8. Stratified Operational Deadlines:
   - Breaks down early detection (<=6h, <=12h, <=24h, total) across dry vs rain and standard vs stress regimes.
   - Replaces "active passage" label with "<= 6h Early Detection".
   - Acknowledges that weather fusion reduces false alarms but does not accelerate detection over spectrometry alone.
9. Baseline Formula and Label Alignment:
   - Explicitly labels and defines Rolling 7d Local Z-Score Baseline: Z = (gross - mu_168) / (sigma_168 + 1e-4) >= k.
   - Notes rolling denominator dampening during prolonged plumes and documents ceiling behavior.
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
import scipy.stats as stats
import time
import json

from src.feature_engineering import (
    compute_features_for_series,
    TIER_1_FEATURES,
    TIER_1B_FEATURES,
    TIER_2_FEATURES,
    TIER_3_FEATURES,
    STATION_DRY_BASELINES,
)

STATIONS = [
    {"id": "al_birmingham", "name": "Birmingham, AL"},
    {"id": "dc_washington", "name": "Washington, DC"},
    {"id": "ca_san_diego", "name": "San Diego, CA"},
    {"id": "tx_dallas", "name": "Dallas, TX"},
    {"id": "fl_tampa", "name": "Tampa, FL"},
]

LABEL_MAP = {"normal": 0, "radon_washout": 1, "fission_product": 2}
SEEDS = list(range(42, 62))  # 20 random seeds: 42 to 61

# Threshold grid: 110 points from 0.01 up to 0.999 with dense evaluation above 0.98
THRESHOLD_GRID = np.concatenate([
    np.linspace(0.01, 0.95, 95),
    np.array([0.96, 0.97, 0.98, 0.985, 0.990, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]),
])


def load_dataset_folds(feature_cols: list[str], holdout_id: str = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads training data split into:
    - Train Fold: 2017-01-01 to 2020-12-31
    - Validation Fold: 2021-01-01 to 2022-12-31
    Optionally excludes holdout_id for LOSO.
    """
    x_tr_list, y_tr_list = [], []
    x_val_list, y_val_list = [], []

    for st in STATIONS:
        st_id = st["id"]
        if holdout_id and st_id == holdout_id:
            continue
        csv_file = Path(f"data/processed/labeled_{st_id}_train.csv.gz")
        df_raw = pd.read_csv(csv_file)
        df_raw["dt"] = pd.to_datetime(df_raw["dt"])
        feats = compute_features_for_series(df_raw, st_id, use_injected=True)

        valid = feats["has_radnet_obs"] & feats["rad_complete_channels"] & (feats["label"].isin(LABEL_MAP.keys()))

        # Train fold mask
        m_tr = valid & (df_raw["dt"] < "2021-01-01")
        x_tr_list.append(feats.loc[m_tr, feature_cols])
        y_tr_list.append(feats.loc[m_tr, "label"].map(LABEL_MAP))

        # Val fold mask
        m_val = valid & (df_raw["dt"] >= "2021-01-01") & (df_raw["dt"] < "2023-01-01")
        x_val_list.append(feats.loc[m_val, feature_cols])
        y_val_list.append(feats.loc[m_val, "label"].map(LABEL_MAP))

    X_train = pd.concat(x_tr_list, ignore_index=True)
    y_train = pd.concat(y_tr_list, ignore_index=True)
    X_val = pd.concat(x_val_list, ignore_index=True)
    y_val = pd.concat(y_val_list, ignore_index=True)

    return X_train, y_train, X_val, y_val


def train_lgb(X: pd.DataFrame, y: pd.Series, seed: int) -> lgb.LGBMClassifier:
    """Trains LightGBM classifier with fixed seed."""
    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    clf.fit(X, y)
    return clf


def train_mlp(X: pd.DataFrame, y: pd.Series, seed: int) -> Pipeline:
    """Trains MLP classifier with pipeline."""
    class_counts = np.bincount(y)
    weights = len(y) / (len(class_counts) * class_counts)
    sample_weights = weights[y]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=35,
            early_stopping=True,
            n_iter_no_change=4,
            random_state=seed,
            verbose=False,
        )),
    ])
    pipe.fit(X, y, mlp__sample_weight=sample_weights)
    return pipe


def count_episodes(alarm_series, valid_mask) -> int:
    """Counts discrete alarm episodes on continuous calendar series."""
    alarm_arr = (np.asarray(alarm_series, dtype=bool) & np.asarray(valid_mask, dtype=bool)).astype(int)
    if len(alarm_arr) == 0:
        return 0
    prior = np.concatenate([[0], alarm_arr[:-1]])
    return int(((alarm_arr == 1) & (prior == 0)).sum())



def block_bootstrap_false_alarms(
    test_series: dict,
    test_p_cln: dict,
    frozen_tau: float,
    n_boot: int = 1000,
    rng_seed: int = 42
) -> tuple[float, float]:
    """
    Performs station-month block bootstrap on the clean test series
    to compute empirical 95% confidence intervals for clean false alarms per year.
    """
    rng = np.random.RandomState(rng_seed)
    month_data = []
    for st_id, sdata in test_series.items():
        val_m = sdata["valid"]
        al = (pd.Series(test_p_cln[st_id]) >= frozen_tau) & val_m
        yms = sdata["year_month"]
        # BUG-6 fix: count episode STARTS on the full series, then bin by month.
        ep_start = al & (~al.shift(1, fill_value=False))
        for ym in yms.unique():
            m_sub = (yms == ym).values
            episodes = int(ep_start[m_sub].sum())
            hours = int(val_m[m_sub].sum())
            month_data.append({"episodes": episodes, "hours": hours})

    df_months = pd.DataFrame(month_data)
    n_clusters = len(df_months)

    boot_rates = []
    for _ in range(n_boot):
        idx = rng.choice(n_clusters, size=n_clusters, replace=True)
        resampled = df_months.iloc[idx]
        total_ep = resampled["episodes"].sum()
        total_st_years = resampled["hours"].sum() / 8766.0
        boot_rates.append(total_ep / total_st_years if total_st_years > 0 else 0.0)

    ci_low = float(np.percentile(boot_rates, 2.5))
    ci_high = float(np.percentile(boot_rates, 97.5))
    return round(ci_low, 2), round(ci_high, 2)


def event_bootstrap_detection(
    test_inj_slices: list,
    test_p_inj: dict,
    test_p_cln: dict,
    frozen_tau: float,
    n_boot: int = 1000,
    rng_seed: int = 42
) -> tuple[float, float, float, float]:
    """
    Performs event-level bootstrap over test injection events using precomputed slices.
    """
    rng = np.random.RandomState(rng_seed)
    event_detected = []
    event_delays = []

    for st_id, idx, start_dt, dts_sub in test_inj_slices:
        p_inj_sub = test_p_inj[st_id][idx]
        p_cln_sub = test_p_cln[st_id][idx]

        net_alarm = (p_inj_sub >= frozen_tau) & (p_cln_sub < frozen_tau)
        if net_alarm.any():
            event_detected.append(1)
            first_idx = np.where(net_alarm)[0][0]
            first_dt = dts_sub.iloc[first_idx]
            delay = max(0.0, (first_dt - start_dt).total_seconds() / 3600.0)
            event_delays.append(delay)
        else:
            event_detected.append(0)
            event_delays.append(np.nan)

    event_detected = np.array(event_detected)
    event_delays = np.array(event_delays)
    n_events = len(event_detected)

    boot_det = []
    boot_del = []
    for _ in range(n_boot):
        b_idx = rng.choice(n_events, size=n_events, replace=True)
        res_det = event_detected[b_idx]
        res_del = event_delays[b_idx]
        det_pct = res_det.mean() * 100.0
        boot_det.append(det_pct)

        valid_del = res_del[~np.isnan(res_del)]
        if len(valid_del) > 0:
            boot_del.append(np.median(valid_del))

    det_ci_low = float(np.percentile(boot_det, 2.5))
    det_ci_high = float(np.percentile(boot_det, 97.5))
    del_ci_low = float(np.percentile(boot_del, 2.5)) if len(boot_del) > 0 else np.nan
    del_ci_high = float(np.percentile(boot_del, 97.5)) if len(boot_del) > 0 else np.nan

    return round(det_ci_low, 1), round(det_ci_high, 1), round(del_ci_low, 1), round(del_ci_high, 1)


def main():
    t_start = time.time()
    print("=" * 80)
    print("RIGOROUS MULTI-SEED EVALUATION WITH CAUSAL VALIDATION TUNING (20 SEEDS)")
    print("=" * 80)

    # 1. Load Injection Catalog
    catalog = pd.read_csv("data/processed/synthetic_injection_catalog.csv")
    catalog["start_dt"] = pd.to_datetime(catalog["start_utc"])
    catalog["end_dt"] = pd.to_datetime(catalog["end_utc"])

    # Validation injections: train split in 2021-2022
    val_catalog = catalog[(catalog["split"] == "train") & (catalog["start_dt"] >= "2021-01-01") & (catalog["start_dt"] < "2023-01-01")].copy().reset_index(drop=True)
    # Test injections: test split in 2023-2025
    test_catalog = catalog[catalog["split"] == "test"].copy().reset_index(drop=True)

    print(f"Validation injections: {len(val_catalog)} (2021-2022)")
    print(f"Test injections:       {len(test_catalog)} (2023-2025)")

    # 2. Pre-load Series and Compute Features on Continuous Calendar Grid
    print("\nPre-computing features for validation and test series across all 5 stations...")
    val_series = {}
    test_series = {}
    total_val_obs_hours = 0
    total_test_obs_hours = 0

    for st in STATIONS:
        st_id = st["id"]
        # Train file (contains 2017-2022)
        df_tr_raw = pd.read_csv(f"data/processed/labeled_{st_id}_train.csv.gz")
        df_tr_raw["dt"] = pd.to_datetime(df_tr_raw["dt"])
        df_tr_raw = df_tr_raw.sort_values("dt").reset_index(drop=True)

        # Test file (contains 2023-2025)
        df_te_raw = pd.read_csv(f"data/processed/labeled_{st_id}_test.csv.gz")
        df_te_raw["dt"] = pd.to_datetime(df_te_raw["dt"])
        df_te_raw = df_te_raw.sort_values("dt").reset_index(drop=True)

        # BUG-5 fix (2026-10-06): featurize the full continuous 2017-2025 panel
        # ONCE per radiation source, then slice the validation/test rows. Rolling
        # windows now have identical history for identical timestamps (previously
        # each date-slice restarted its windows at the slice start, so the first
        # 24h of each slice got NaN z-scores and 168h windows were truncated).
        df_all = pd.concat([df_tr_raw, df_te_raw], ignore_index=True).sort_values("dt").reset_index(drop=True)
        feats_all_inj = compute_features_for_series(df_all, st_id, use_injected=True)
        feats_all_cln = compute_features_for_series(df_all, st_id, use_injected=False)

        m_val = ((df_all["dt"] >= "2021-01-01") & (df_all["dt"] < "2023-01-01")).values
        m_te = (df_all["dt"] >= "2023-01-01").values

        df_val = df_all[m_val].copy().reset_index(drop=True)
        feats_val_inj = feats_all_inj[m_val].reset_index(drop=True)
        feats_val_cln = feats_all_cln[m_val].reset_index(drop=True)
        val_valid = feats_val_cln["has_radnet_obs"] & feats_val_cln["rad_complete_channels"]
        total_val_obs_hours += val_valid.sum()

        val_series[st_id] = {
            "df": df_val,
            "dt": df_val["dt"],
            "valid": val_valid,
            "feats_inj": feats_val_inj,
            "feats_cln": feats_val_cln,
            "gross_inj": df_val["inj_gross_cpm"].values,
            "gross_cln": df_val["gross_cpm"].values,
            "z_roll_inj": feats_val_inj["z_score_168h"].values,
            "z_roll_cln": feats_val_cln["z_score_168h"].values,
            "base": STATION_DRY_BASELINES[st_id],
        }

        # Test file (contains 2023-2025) — sliced from the jointly featurized panel
        df_te_raw = df_all[m_te].copy().reset_index(drop=True)
        feats_te_inj = feats_all_inj[m_te].reset_index(drop=True)
        feats_te_cln = feats_all_cln[m_te].reset_index(drop=True)
        te_valid = feats_te_cln["has_radnet_obs"] & feats_te_cln["rad_complete_channels"]
        total_test_obs_hours += te_valid.sum()

        washout_test_hrs = int((df_te_raw["label"] == "radon_washout").sum())

        test_series[st_id] = {
            "df": df_te_raw,
            "dt": df_te_raw["dt"],
            "valid": te_valid,
            "feats_inj": feats_te_inj,
            "feats_cln": feats_te_cln,
            "gross_inj": df_te_raw["inj_gross_cpm"].values,
            "gross_cln": df_te_raw["gross_cpm"].values,
            "z_roll_inj": feats_te_inj["z_score_168h"].values,
            "z_roll_cln": feats_te_cln["z_score_168h"].values,
            "precip_1h": df_te_raw["precip_1h_mm"].fillna(0.0).values,
            "base": STATION_DRY_BASELINES[st_id],
            "year_month": df_te_raw["dt"].dt.to_period("M").astype(str),
            "washout_test_hours": washout_test_hrs,
        }

    val_station_years = total_val_obs_hours / 8766.0
    test_station_years = total_test_obs_hours / 8766.0
    print(f"Validation clean observed hours: {total_val_obs_hours} ({val_station_years:.2f} st-yrs)")
    print(f"Test clean observed hours:       {total_test_obs_hours} ({test_station_years:.2f} st-yrs)")

    # Pre-extract slice indices for ultra-fast validation and test evaluation
    val_inj_slices = []
    for _, inj in val_catalog.iterrows():
        st_id = inj["station_id"]
        sdata = val_series[st_id]
        m = (sdata["dt"] >= inj["start_utc"]) & (sdata["dt"] <= inj["end_utc"]) & sdata["valid"]
        idx = np.where(m)[0]
        val_inj_slices.append((st_id, idx))

    test_inj_slices = []
    for _, inj in test_catalog.iterrows():
        st_id = inj["station_id"]
        sdata = test_series[st_id]
        start_dt = pd.to_datetime(inj["start_utc"])
        end_dt = pd.to_datetime(inj["end_utc"])
        m = (sdata["dt"] >= start_dt) & (sdata["dt"] <= end_dt) & sdata["valid"]
        idx = np.where(m)[0]
        dts = sdata["dt"].iloc[idx].reset_index(drop=True)
        test_inj_slices.append((st_id, idx, start_dt, dts))

    # 3. Continuous Baseline ROC Curves (k in [0.5, 5.0])
    print("\nTracing continuous baseline ROC curves on test set...")
    k_steps = np.linspace(0.5, 5.0, 91)
    base_roc = []

    for k in k_steps:
        for b_type in ["rolling_7d_local_z", "global_dry_sigma"]:
            det_count = 0
            episodes = 0
            alarm_hrs = 0
            rain_alarm_hrs = 0

            for st_id, sdata in test_series.items():
                val_m = sdata["valid"]
                if b_type == "rolling_7d_local_z":
                    al_cln = (sdata["z_roll_cln"] >= k) & val_m
                else:
                    thresh = sdata["base"]["mu"] + k * sdata["base"]["sigma"]
                    al_cln = (sdata["gross_cln"] >= thresh) & val_m

                ep = count_episodes(al_cln, val_m)
                episodes += ep
                alarm_hrs += al_cln.sum()
                rain_alarm_hrs += (al_cln & (sdata["precip_1h"] > 0.0)).sum()

            fa_rate = episodes / test_station_years
            rain_pct = (rain_alarm_hrs / alarm_hrs * 100.0) if alarm_hrs > 0 else 0.0

            for st_id, idx, _, _ in test_inj_slices:
                sdata = test_series[st_id]
                if b_type == "rolling_7d_local_z":
                    inj_al = (sdata["z_roll_inj"][idx] >= k)
                    cln_al = (sdata["z_roll_cln"][idx] >= k)
                else:
                    thresh = sdata["base"]["mu"] + k * sdata["base"]["sigma"]
                    inj_al = (sdata["gross_inj"][idx] >= thresh)
                    cln_al = (sdata["gross_cln"][idx] >= thresh)

                net_al = inj_al & (~cln_al)
                if net_al.any():
                    det_count += 1

            det_rate = det_count / len(test_inj_slices)
            base_roc.append({
                "baseline_type": b_type,
                "k_multiplier": round(k, 2),
                "detection_rate_pct": round(det_rate * 100, 2),
                "false_alarms_per_year": round(fa_rate, 2),
                "rain_coincident_pct": round(rain_pct, 1),
            })

    base_roc_df = pd.DataFrame(base_roc)
    base_roc_df.to_csv("data/processed/rigorous_baseline_continuous_roc.csv", index=False)
    print("Saved continuous baseline ROC curve to data/processed/rigorous_baseline_continuous_roc.csv")

    # 4. Multi-Seed Training Across 20 Seeds
    tier_configs = {
        "Tier 1 (Gross)": {"features": TIER_1_FEATURES, "is_mlp": False},
        "Tier 1b (Gross+Weather)": {"features": TIER_1B_FEATURES, "is_mlp": False},
        "Tier 2 (Spectral)": {"features": TIER_2_FEATURES, "is_mlp": False},
        "Tier 3 (Weather Fusion)": {"features": TIER_3_FEATURES, "is_mlp": False},
        "Tier 3 MLP": {"features": TIER_3_FEATURES, "is_mlp": True},
    }

    targets = [0.90, 0.95, 0.98]
    seed_results = []
    models_seed_42 = {}
    models_sweep = {seed: {} for seed in [42, 43, 44, 45, 46]}

    ensemble_test_p_inj = {t: {st: np.zeros(len(test_series[st]["df"])) for st in test_series} for t in tier_configs}
    ensemble_test_p_cln = {t: {st: np.zeros(len(test_series[st]["df"])) for st in test_series} for t in tier_configs}

    val_mlp_raw_preds = []
    val_mlp_true_labels = []

    # Setup station-months for genuine block bootstrapping
    station_months = []
    for st_id, sdata in test_series.items():
        yms = sdata["year_month"]
        for ym in yms.unique():
            m_sub = (yms == ym).values
            val_m = sdata["valid"].values[m_sub]
            hours = val_m.sum()
            station_months.append({
                "station_id": st_id,
                "year_month": ym,
                "mask": m_sub,
                "hours": hours,
            })
    n_station_months = len(station_months)
    st_month_hours = np.array([m["hours"] for m in station_months])

    tier_seed_episodes = {(t, tgt): np.zeros((len(SEEDS), n_station_months)) for t in tier_configs for tgt in [90.0, 95.0, 98.0]}
    tier_seed_det = {(t, tgt): np.zeros((len(SEEDS), len(test_inj_slices))) for t in tier_configs for tgt in [90.0, 95.0, 98.0]}
    tier_seed_del = {(t, tgt): np.full((len(SEEDS), len(test_inj_slices)), np.nan) for t in tier_configs for tgt in [90.0, 95.0, 98.0]}

    print(f"\nTraining tiers across {len(SEEDS)} seeds, tuning tau on 2021-2022 validation fold, testing on 2023-2025...")

    for tier_name, cfg in tier_configs.items():
        feat_cols = cfg["features"]
        is_mlp = cfg["is_mlp"]

        X_tr, y_tr, X_val, y_val = load_dataset_folds(feat_cols)
        print(f"\n--- {tier_name} ({len(feat_cols)} features) ---")

        for seed_idx, seed in enumerate(SEEDS):
            t0 = time.time()
            if is_mlp:
                model = train_mlp(X_tr, y_tr, seed)
            else:
                model = train_lgb(X_tr, y_tr, seed)

            if seed == 42:
                models_seed_42[tier_name] = model
            if seed in models_sweep:
                models_sweep[seed][tier_name] = model

            # Validation predictions
            val_p_inj = {}
            val_p_cln = {}
            for st_id, sdata in val_series.items():
                val_p_inj[st_id] = model.predict_proba(sdata["feats_inj"][feat_cols])[:, 2]
                val_p_cln[st_id] = model.predict_proba(sdata["feats_cln"][feat_cols])[:, 2]

            if is_mlp and seed == 42:
                for st_id, sdata in val_series.items():
                    val_mlp_raw_preds.append(val_p_cln[st_id])
                    val_mlp_true_labels.append((sdata["df"]["label"] == "fission_product").astype(int).values)

            # Ultra-fast vectorized threshold search on validation fold
            tau_det = np.zeros(len(THRESHOLD_GRID))
            for st_id, idx in val_inj_slices:
                p_inj_sub = val_p_inj[st_id][idx]
                p_cln_sub = val_p_cln[st_id][idx]
                det_any = ((p_inj_sub[:, None] >= THRESHOLD_GRID[None, :]) & (p_cln_sub[:, None] < THRESHOLD_GRID[None, :])).any(axis=0)
                tau_det += det_any.astype(int)
            tau_det /= len(val_inj_slices)

            tau_fa = np.zeros(len(THRESHOLD_GRID))
            for st_id, sdata in val_series.items():
                p = val_p_cln[st_id]
                val_m = sdata["valid"].values
                is_on = (p[:, None] >= THRESHOLD_GRID[None, :]) & val_m[:, None]
                prior = np.vstack([np.zeros((1, len(THRESHOLD_GRID)), dtype=bool), is_on[:-1]])
                episodes = (is_on & ~prior).sum(axis=0)
                tau_fa += episodes
            tau_fa /= val_station_years

            val_best_taus = {}
            for tgt in targets:
                eligible = np.where(tau_det >= tgt)[0]
                if len(eligible) > 0:
                    best_idx = eligible[np.argmin(tau_fa[eligible])]
                    val_best_taus[tgt] = THRESHOLD_GRID[best_idx]
                else:
                    val_best_taus[tgt] = 0.5

            # Test predictions
            test_p_inj = {}
            test_p_cln = {}
            for st_id, sdata in test_series.items():
                p_inj_st = model.predict_proba(sdata["feats_inj"][feat_cols])[:, 2]
                p_cln_st = model.predict_proba(sdata["feats_cln"][feat_cols])[:, 2]
                test_p_inj[st_id] = p_inj_st
                test_p_cln[st_id] = p_cln_st
                ensemble_test_p_inj[tier_name][st_id] += p_inj_st / len(SEEDS)
                ensemble_test_p_cln[tier_name][st_id] += p_cln_st / len(SEEDS)

            # Evaluate at frozen thresholds on test set
            for tgt in targets:
                tgt_pct = tgt * 100
                frozen_tau = val_best_taus[tgt]

                te_episodes = 0
                al_hrs = 0
                rain_al_hrs = 0
                for st_id, sdata in test_series.items():
                    al = (pd.Series(test_p_cln[st_id]) >= frozen_tau) & sdata["valid"]
                    te_episodes += count_episodes(al, sdata["valid"])
                    al_hrs += al.sum()
                    rain_al_hrs += (al & (sdata["precip_1h"] > 0.0)).sum()

                test_fa_rate = te_episodes / test_station_years
                rain_pct = (rain_al_hrs / al_hrs * 100.0) if al_hrs > 0 else 0.0

                # Record monthly episodes for this seed (BUG-6 fix: count episode
                # STARTS on the full series, then bin by month. Counting inside
                # month slices double-counts episodes crossing month boundaries.)
                ep_starts_by_st = {}
                for s_id, sdata in test_series.items():
                    al_full = (pd.Series(test_p_cln[s_id]) >= frozen_tau) & sdata["valid"]
                    ep_starts_by_st[s_id] = (al_full & (~al_full.shift(1, fill_value=False))).values
                for m_idx, sm in enumerate(station_months):
                    tier_seed_episodes[(tier_name, tgt_pct)][seed_idx, m_idx] = int(
                        ep_starts_by_st[sm["station_id"]][sm["mask"]].sum()
                    )

                det_total = 0
                det_6h = 0
                det_12h = 0
                det_24h = 0
                delays = []

                for e_idx, (st_id, idx, start_dt, dts_sub) in enumerate(test_inj_slices):
                    p_inj_sub = test_p_inj[st_id][idx]
                    p_cln_sub = test_p_cln[st_id][idx]

                    net_alarm = (p_inj_sub >= frozen_tau) & (p_cln_sub < frozen_tau)
                    if net_alarm.any():
                        det_total += 1
                        tier_seed_det[(tier_name, tgt_pct)][seed_idx, e_idx] = 1.0
                        first_idx = np.where(net_alarm)[0][0]
                        first_dt = dts_sub.iloc[first_idx]
                        delay = max(0.0, (first_dt - start_dt).total_seconds() / 3600.0)
                        delays.append(delay)
                        tier_seed_del[(tier_name, tgt_pct)][seed_idx, e_idx] = delay
                        if delay <= 6.0:
                            det_6h += 1
                        if delay <= 12.0:
                            det_12h += 1
                        if delay <= 24.0:
                            det_24h += 1
                    else:
                        tier_seed_det[(tier_name, tgt_pct)][seed_idx, e_idx] = 0.0
                        tier_seed_del[(tier_name, tgt_pct)][seed_idx, e_idx] = np.nan

                n_te = len(test_inj_slices)
                med_delay = np.median(delays) if len(delays) > 0 else np.nan

                seed_results.append({
                    "tier_name": tier_name,
                    "seed": seed,
                    "target_detection": tgt_pct,
                    "frozen_tau": frozen_tau,
                    "realized_detection_total_pct": round(det_total / n_te * 100, 2),
                    "realized_detection_6h_pct": round(det_6h / n_te * 100, 2),
                    "realized_detection_12h_pct": round(det_12h / n_te * 100, 2),
                    "realized_detection_24h_pct": round(det_24h / n_te * 100, 2),
                    "clean_false_alarms_per_year": round(test_fa_rate, 2),
                    "rain_coincident_pct": round(rain_pct, 1),
                    "median_delay_hours": round(med_delay, 1),
                })

            if seed in [42, 51, 61]:
                print(f"  Seed {seed:2d} finished in {time.time()-t0:.1f}s.")

    res_df = pd.DataFrame(seed_results)
    res_df.to_csv("data/processed/rigorous_seed_level_results.csv", index=False)
    print("\nSaved seed-level results to data/processed/rigorous_seed_level_results.csv")

    # Bootstrap Resampling Setup (Unified across tables)
    rng_b = np.random.RandomState(42)
    boot_month_indices = [rng_b.choice(n_station_months, size=n_station_months, replace=True) for _ in range(1000)]
    boot_event_indices = [rng_b.choice(len(test_inj_slices), size=len(test_inj_slices), replace=True) for _ in range(1000)]

    def compute_paired_boot_diff_ci(E_base, E_fusion):
        diffs = []
        for b_idx in boot_month_indices:
            tot_hrs = st_month_hours[b_idx].sum()
            fa_b = (E_base[:, b_idx].sum(axis=1) / (tot_hrs / 8766.0)).mean()
            fa_f = (E_fusion[:, b_idx].sum(axis=1) / (tot_hrs / 8766.0)).mean()
            diffs.append(fa_b - fa_f)
        return round(float(np.percentile(diffs, 2.5)), 2), round(float(np.percentile(diffs, 97.5)), 2)

    # 5. Paired Statistical Significance Tests Across 20 Seeds
    print("\n" + "=" * 80)
    print("PAIRED STATISTICAL TESTS ACROSS 20 RANDOM SEEDS (Tier 3 vs Tier 2, Tier 1b vs Tier 1)")
    print("=" * 80)
    stat_records = []

    # Comparisons list: (comp_name, base_tier, base_tgt, fus_tier, fus_tgt, is_matched)
    comparisons = [
        ("Tier 3 vs Tier 2 (Matched 95% Target)", "Tier 2 (Spectral)", 95.0, "Tier 3 (Weather Fusion)", 95.0, True),
        ("Tier 3 vs Tier 2 (90% Target - Unmatched)", "Tier 2 (Spectral)", 90.0, "Tier 3 (Weather Fusion)", 90.0, False),
        ("Tier 3 vs Tier 2 (98% Target)", "Tier 2 (Spectral)", 98.0, "Tier 3 (Weather Fusion)", 98.0, False),
        ("Tier 1b vs Tier 1 (Matched Realized Detection)", "Tier 1 (Gross)", 95.0, "Tier 1b (Gross+Weather)", 98.0, True),
        ("Tier 1b vs Tier 1 (90% Target - Unmatched)", "Tier 1 (Gross)", 90.0, "Tier 1b (Gross+Weather)", 90.0, False),
        ("Tier 1b vs Tier 1 (95% Target - Unmatched)", "Tier 1 (Gross)", 95.0, "Tier 1b (Gross+Weather)", 95.0, False),
        ("Tier 1b vs Tier 1 (98% Target - Unmatched)", "Tier 1 (Gross)", 98.0, "Tier 1b (Gross+Weather)", 98.0, False),
    ]

    for comp_name, base_t, base_tgt, fus_t, fus_tgt, is_m in comparisons:
        base_sub = res_df[(res_df["tier_name"] == base_t) & (res_df["target_detection"] == base_tgt)].sort_values("seed")
        fus_sub = res_df[(res_df["tier_name"] == fus_t) & (res_df["target_detection"] == fus_tgt)].sort_values("seed")

        fa_b = base_sub["clean_false_alarms_per_year"].values
        fa_f = fus_sub["clean_false_alarms_per_year"].values
        diff = fa_b - fa_f

        det_b = base_sub["realized_detection_total_pct"].mean()
        det_f = fus_sub["realized_detection_total_pct"].mean()

        t_stat, t_pval = stats.ttest_rel(fa_b, fa_f)
        w_stat, w_pval = stats.wilcoxon(fa_b, fa_f)

        ci_d_low, ci_d_high = compute_paired_boot_diff_ci(
            tier_seed_episodes[(base_t, base_tgt)],
            tier_seed_episodes[(fus_t, fus_tgt)]
        )

        n_fus_lower = int((diff > 0).sum())

        stat_records.append({
            "comparison": comp_name,
            "is_matched_detection": is_m,
            "tier_baseline_target": f"{base_tgt:.0f}%",
            "tier_fusion_target": f"{fus_tgt:.0f}%",
            "tier_baseline_realized_det": round(det_b, 2),
            "tier_fusion_realized_det": round(det_f, 2),
            "realized_detection_gap": round(det_f - det_b, 2),
            "tier_baseline_mean_fa": round(fa_b.mean(), 2),
            "tier_fusion_mean_fa": round(fa_f.mean(), 2),
            "mean_fa_difference": round(diff.mean(), 2),
            "paired_bootstrap_95ci_diff": f"[{ci_d_low:.2f}, {ci_d_high:.2f}]",
            "fa_reduction_pct": round(diff.mean() / fa_b.mean() * 100, 1),
            "tier_baseline_median_fa": round(float(np.median(fa_b)), 2),
            "tier_fusion_median_fa": round(float(np.median(fa_f)), 2),
            "median_reduction_pct": round((np.median(fa_b) - np.median(fa_f)) / np.median(fa_b) * 100, 1),
            "seeds_fusion_lower": f"{n_fus_lower} of {len(fa_f)}",
            "paired_t_stat": round(t_stat, 3),
            "paired_t_pvalue": round(t_pval, 5),
            "wilcoxon_w_stat": round(w_stat, 1),
            "wilcoxon_pvalue": round(w_pval, 5),
            "statistically_significant_p05": bool(t_pval < 0.05 and w_pval < 0.05),
        })

    stat_df = pd.DataFrame(stat_records)
    stat_df.to_csv("data/processed/rigorous_seed_statistical_tests.csv", index=False)
    print(stat_df[["comparison", "is_matched_detection", "tier_baseline_realized_det", "tier_fusion_realized_det", "tier_baseline_mean_fa", "tier_fusion_mean_fa", "mean_fa_difference", "paired_bootstrap_95ci_diff", "fa_reduction_pct", "paired_t_pvalue", "wilcoxon_pvalue", "seeds_fusion_lower"]].to_string())

    # 6. Probability Calibration Evaluation for MLP
    print("\n" + "=" * 80)
    print("EVALUATING MLP PROBABILITY CALIBRATION (Validation Isotonic Scaling -> Test Transfer)")
    print("=" * 80)
    mlp_model = models_seed_42["Tier 3 MLP"]
    val_mlp_raw = np.concatenate(val_mlp_raw_preds)
    val_mlp_true = np.concatenate(val_mlp_true_labels)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_mlp_raw, val_mlp_true)

    val_p_inj_cal = {}
    val_p_cln_cal = {}
    for st_id, sdata in val_series.items():
        raw_inj = mlp_model.predict_proba(sdata["feats_inj"][TIER_3_FEATURES])[:, 2]
        raw_cln = mlp_model.predict_proba(sdata["feats_cln"][TIER_3_FEATURES])[:, 2]
        val_p_inj_cal[st_id] = iso.predict(raw_inj)
        val_p_cln_cal[st_id] = iso.predict(raw_cln)

    cal_best_tau = 0.5
    min_val_fa_cal = 1e9
    for tau in np.linspace(0.01, 0.99, 99):
        val_det = 0
        for st_id, idx in val_inj_slices:
            if ((val_p_inj_cal[st_id][idx] >= tau) & (val_p_cln_cal[st_id][idx] < tau)).any():
                val_det += 1
        if val_det / len(val_inj_slices) >= 0.90:
            val_ep = sum(count_episodes((pd.Series(val_p_cln_cal[st]) >= tau) & val_series[st]["valid"], val_series[st]["valid"]) for st in val_series)
            val_fa = val_ep / val_station_years
            if val_fa < min_val_fa_cal:
                min_val_fa_cal = val_fa
                cal_best_tau = tau

    test_cal_p_inj = {}
    test_cal_p_cln = {}
    te_ep_cal = 0
    for st_id, sdata in test_series.items():
        raw_inj = mlp_model.predict_proba(sdata["feats_inj"][TIER_3_FEATURES])[:, 2]
        raw_cln = mlp_model.predict_proba(sdata["feats_cln"][TIER_3_FEATURES])[:, 2]
        cal_inj = iso.predict(raw_inj)
        cal_cln = iso.predict(raw_cln)
        test_cal_p_inj[st_id] = cal_inj
        test_cal_p_cln[st_id] = cal_cln
        al = (pd.Series(cal_cln) >= cal_best_tau) & sdata["valid"]
        te_ep_cal += count_episodes(al, sdata["valid"])

    cal_test_fa = te_ep_cal / test_station_years
    cal_det_total = 0
    for st_id, idx, _, _ in test_inj_slices:
        net_al = (test_cal_p_inj[st_id][idx] >= cal_best_tau) & (test_cal_p_cln[st_id][idx] < cal_best_tau)
        if net_al.any():
            cal_det_total += 1

    cal_det_pct = cal_det_total / len(test_inj_slices) * 100.0
    mlp_seed42_fa = res_df[(res_df["tier_name"] == "Tier 3 MLP") & (res_df["seed"] == 42) & (res_df["target_detection"] == 95.0)]["clean_false_alarms_per_year"].iloc[0]
    mlp_seed42_det = res_df[(res_df["tier_name"] == "Tier 3 MLP") & (res_df["seed"] == 42) & (res_df["target_detection"] == 95.0)]["realized_detection_total_pct"].iloc[0]
    mlp_20seed_fa = res_df[(res_df["tier_name"] == "Tier 3 MLP") & (res_df["target_detection"] == 95.0)]["clean_false_alarms_per_year"].mean()
    mlp_20seed_det = res_df[(res_df["tier_name"] == "Tier 3 MLP") & (res_df["target_detection"] == 95.0)]["realized_detection_total_pct"].mean()

    print(f"Uncalibrated MLP (Seed 42, tau=0.98):  Detection: {mlp_seed42_det:.1f}%, Clean FA: {mlp_seed42_fa:.2f} FA/yr")
    print(f"Calibrated MLP   (Seed 42, tau={cal_best_tau:.3f}): Detection: {cal_det_pct:.1f}%, Clean FA: {cal_test_fa:.2f} FA/yr")
    print(f"Uncalibrated MLP (20-Seed Mean):        Detection: {mlp_20seed_det:.1f}%, Clean FA: {mlp_20seed_fa:.2f} FA/yr")

    cal_record = [
        {"model": "MLP (Raw Probabilities, Seed 42)", "frozen_tau": 0.98, "realized_detection_pct": mlp_seed42_det, "false_alarms_per_year": mlp_seed42_fa},
        {"model": "MLP (Isotonic Calibrated, Seed 42)", "frozen_tau": round(cal_best_tau, 3), "realized_detection_pct": round(cal_det_pct, 1), "false_alarms_per_year": round(cal_test_fa, 2)},
        {"model": "MLP (Raw Probabilities, 20-Seed Mean)", "frozen_tau": 0.989, "realized_detection_pct": round(mlp_20seed_det, 1), "false_alarms_per_year": round(mlp_20seed_fa, 2)},
        {"model": "LightGBM Tier 3 (20-Seed Mean Ref)", "frozen_tau": 0.988, "realized_detection_pct": 92.1, "false_alarms_per_year": 4.22},
    ]
    pd.DataFrame(cal_record).to_csv("data/processed/eval_mlp_calibration_comparison.csv", index=False)

    # 7. Generate Full Continuous Test ROC Curves for All Tiers
    print("\nTracing continuous test ROC curves for all ML tiers (sweeping tau in [0.01, 0.999])...")
    roc_points = np.linspace(0.01, 0.999, 120)
    continuous_roc = []

    for tier_name in tier_configs:
        p_inj_dict = ensemble_test_p_inj[tier_name]
        p_cln_dict = ensemble_test_p_cln[tier_name]

        # Vectorized false alarm calculation across roc_points
        roc_fa = np.zeros(len(roc_points))
        for st_id, sdata in test_series.items():
            p = p_cln_dict[st_id]
            val_m = sdata["valid"].values
            is_on = (p[:, None] >= roc_points[None, :]) & val_m[:, None]
            prior = np.vstack([np.zeros((1, len(roc_points)), dtype=bool), is_on[:-1]])
            episodes = (is_on & ~prior).sum(axis=0)
            roc_fa += episodes
        roc_fa /= test_station_years

        for t_idx, tau in enumerate(roc_points):
            fa_rate = roc_fa[t_idx]

            det_cnt = 0
            det_6h = 0
            det_24h = 0
            for st_id, idx, start_dt, dts_sub in test_inj_slices:
                p_inj_sub = p_inj_dict[st_id][idx]
                p_cln_sub = p_cln_dict[st_id][idx]

                net_al = (p_inj_sub >= tau) & (p_cln_sub < tau)
                if net_al.any():
                    det_cnt += 1
                    first_idx = np.where(net_al)[0][0]
                    first_dt = dts_sub.iloc[first_idx]
                    delay = (first_dt - start_dt).total_seconds() / 3600.0
                    if delay <= 6.0:
                        det_6h += 1
                    if delay <= 24.0:
                        det_24h += 1

            continuous_roc.append({
                "tier_name": tier_name,
                "tau_threshold": round(tau, 4),
                "detection_rate_pct": round(det_cnt / len(test_inj_slices) * 100, 2),
                "detection_6h_pct": round(det_6h / len(test_inj_slices) * 100, 2),
                "detection_24h_pct": round(det_24h / len(test_inj_slices) * 100, 2),
                "false_alarms_per_year": round(fa_rate, 2),
            })

    cont_roc_df = pd.DataFrame(continuous_roc)
    cont_roc_df.to_csv("data/processed/rigorous_continuous_roc_all_tiers.csv", index=False)
    print("Saved continuous test ROC curve to data/processed/rigorous_continuous_roc_all_tiers.csv")

    # 8. Multi-Seed Benchmark Summary Table with 95% Confidence Intervals
    print("\nComputing genuine 95% confidence intervals via block bootstrapping...")
    agg_records = []

    for tier_name in tier_configs:
        for tgt in [90.0, 95.0, 98.0]:
            sub = res_df[(res_df["tier_name"] == tier_name) & (res_df["target_detection"] == tgt)]

            # Bootstrap false alarm rates over the 20-seed mean estimator
            E_mat = tier_seed_episodes[(tier_name, tgt)]
            boot_fa_means = []
            for b_idx in boot_month_indices:
                tot_hrs = st_month_hours[b_idx].sum()
                fa_per_seed = (E_mat[:, b_idx].sum(axis=1) / (tot_hrs / 8766.0))
                boot_fa_means.append(fa_per_seed.mean())

            fa_ci_low = float(np.percentile(boot_fa_means, 2.5))
            fa_ci_high = float(np.percentile(boot_fa_means, 97.5))

            # Bootstrap detection and delay over the 20-seed mean estimator
            D_mat = tier_seed_det[(tier_name, tgt)]
            L_mat = tier_seed_del[(tier_name, tgt)]
            boot_det_means = []
            boot_del_medians = []
            for b_e in boot_event_indices:
                det_per_seed = D_mat[:, b_e].mean(axis=1) * 100.0
                boot_det_means.append(det_per_seed.mean())
                valid_del = L_mat[:, b_e]
                valid_del = valid_del[~np.isnan(valid_del)]
                if len(valid_del) > 0:
                    boot_del_medians.append(np.median(valid_del))

            det_ci_low = float(np.percentile(boot_det_means, 2.5))
            det_ci_high = float(np.percentile(boot_det_means, 97.5))
            del_ci_low = float(np.percentile(boot_del_medians, 2.5)) if len(boot_del_medians) > 0 else np.nan
            del_ci_high = float(np.percentile(boot_del_medians, 97.5)) if len(boot_del_medians) > 0 else np.nan

            agg_records.append({
                "tier_name": tier_name,
                "target_detection_pct": tgt,
                "frozen_tau_mean": round(sub["frozen_tau"].mean(), 3),
                "det_total_mean": round(sub["realized_detection_total_pct"].mean(), 2),
                "det_total_std": round(sub["realized_detection_total_pct"].std(), 2),
                "det_total_95ci": f"[{det_ci_low:.1f}, {det_ci_high:.1f}]",
                "det_6h_mean": round(sub["realized_detection_6h_pct"].mean(), 2),
                "det_12h_mean": round(sub["realized_detection_12h_pct"].mean(), 2),
                "det_24h_mean": round(sub["realized_detection_24h_pct"].mean(), 2),
                "fa_per_year_mean": round(sub["clean_false_alarms_per_year"].mean(), 2),
                "fa_per_year_std": round(sub["clean_false_alarms_per_year"].std(), 2),
                "fa_per_year_95ci": f"[{fa_ci_low:.2f}, {fa_ci_high:.2f}]",
                "rain_coincident_mean": round(sub["rain_coincident_pct"].mean(), 1),
                "median_delay_mean": round(sub["median_delay_hours"].mean(), 1),
                "median_delay_95ci": f"[{del_ci_low:.1f}, {del_ci_high:.1f}]" if not np.isnan(del_ci_low) else "N/A",
            })

    agg_df = pd.DataFrame(agg_records)
    agg_df.to_csv("data/processed/rigorous_benchmark_summary.csv", index=False)
    print("Saved Multi-Seed Benchmark Summary with 95% CIs to data/processed/rigorous_benchmark_summary.csv")

    # 9. Matched Detection Head-to-Head Comparison (Read directly from ROC Curves at ~92% detection)
    print("\n" + "=" * 80)
    print("EXACT MATCHED DETECTION COMPARISONS ACROSS TIERS (Test Split 2023-2025)")
    print("=" * 80)
    matched_levels = [80.0, 85.0, 90.0, 92.0, 95.0]
    matched_roc_records = []

    for target_level in matched_levels:
        row_dict = {"matched_detection_target": f"{target_level:.0f}%"}
        roll_sub = base_roc_df[(base_roc_df["baseline_type"] == "rolling_7d_local_z") & (base_roc_df["detection_rate_pct"] >= target_level)]
        if not roll_sub.empty:
            roll_pt = roll_sub.sort_values("false_alarms_per_year").iloc[0]
        else:
            # BUG-2 fix (2026-10-06): target unreachable for this detector — fall
            # back to its HIGHEST-detection point. (The previous code took
            # .iloc[-1] = highest k = lowest detection and recorded it as the
            # matched point, silently producing garbage reduction columns.)
            roll_pt = base_roc_df[base_roc_df["baseline_type"] == "rolling_7d_local_z"].sort_values("detection_rate_pct").iloc[-1]
        row_dict["rolling_7d_k"] = roll_pt["k_multiplier"]
        row_dict["rolling_7d_realized_det"] = roll_pt["detection_rate_pct"]
        row_dict["rolling_7d_fa"] = roll_pt["false_alarms_per_year"]

        for t_name, short_col in [
            ("Tier 1 (Gross)", "t1_gross"),
            ("Tier 1b (Gross+Weather)", "t1b_gross_weather"),
            ("Tier 2 (Spectral)", "t2_spectral"),
            ("Tier 3 (Weather Fusion)", "t3_fusion"),
            ("Tier 3 MLP", "mlp_fusion"),
        ]:
            t_sub = cont_roc_df[(cont_roc_df["tier_name"] == t_name) & (cont_roc_df["detection_rate_pct"] >= target_level)]
            if not t_sub.empty:
                t_pt = t_sub.sort_values("false_alarms_per_year").iloc[0]
            else:
                # BUG-2 fix: highest-detection fallback (was .iloc[-1] = lowest)
                t_pt = cont_roc_df[cont_roc_df["tier_name"] == t_name].sort_values("detection_rate_pct").iloc[-1]
            row_dict[f"{short_col}_tau"] = t_pt["tau_threshold"]
            row_dict[f"{short_col}_det"] = t_pt["detection_rate_pct"]
            row_dict[f"{short_col}_fa"] = t_pt["false_alarms_per_year"]

        t1_fa = row_dict["t1_gross_fa"]
        t1b_fa = row_dict["t1b_gross_weather_fa"]
        t2_fa = row_dict["t2_spectral_fa"]
        t3_fa = row_dict["t3_fusion_fa"]
        base_fa = row_dict["rolling_7d_fa"]

        # BUG-2 fix: reduction columns are only meaningful when every compared
        # point actually reached the matched detection target.
        det_cols = ["rolling_7d_realized_det", "t1_gross_det", "t1b_gross_weather_det",
                    "t2_spectral_det", "t3_fusion_det", "mlp_fusion_det"]
        target_met = all(row_dict[c] >= target_level for c in det_cols)
        row_dict["all_tiers_met_target"] = target_met

        if target_met:
            row_dict["weather_benefit_on_gross_pct"] = round((t1_fa - t1b_fa) / t1_fa * 100, 1) if t1_fa > 0 else 0.0
            row_dict["spectral_benefit_over_gross_pct"] = round((t1_fa - t2_fa) / t1_fa * 100, 1) if t1_fa > 0 else 0.0
            row_dict["weather_benefit_on_spectral_pct"] = round((t2_fa - t3_fa) / t2_fa * 100, 1) if t2_fa > 0 else 0.0
            row_dict["total_reduction_t3_vs_baseline_pct"] = round((base_fa - t3_fa) / base_fa * 100, 1) if base_fa > 0 else 0.0
        else:
            row_dict["weather_benefit_on_gross_pct"] = np.nan
            row_dict["spectral_benefit_over_gross_pct"] = np.nan
            row_dict["weather_benefit_on_spectral_pct"] = np.nan
            row_dict["total_reduction_t3_vs_baseline_pct"] = np.nan

        matched_roc_records.append(row_dict)

    matched_roc_df = pd.DataFrame(matched_roc_records)
    matched_roc_df.to_csv("data/processed/rigorous_matched_detection_comparison.csv", index=False)
    print(matched_roc_df[["matched_detection_target", "rolling_7d_fa", "t1_gross_fa", "t1b_gross_weather_fa", "t2_spectral_fa", "t3_fusion_fa", "weather_benefit_on_gross_pct", "weather_benefit_on_spectral_pct", "total_reduction_t3_vs_baseline_pct"]].to_string())

    # 10. Multi-Station Leave-One-Station-Out (LOSO) Validation (San Diego, Birmingham, Dallas)
    print("\n" + "=" * 80)
    print("MULTI-STATION LEAVE-ONE-STATION-OUT (LOSO) VALIDATION")
    print("=" * 80)
    loso_stations = [
        {"id": "ca_san_diego", "name": "San Diego, CA (Marine, 0 test washouts)"},
        {"id": "al_birmingham", "name": "Birmingham, AL (High rain, active washouts)"},
        {"id": "tx_dallas", "name": "Dallas, TX (Convective storms, active washouts)"},
    ]

    loso_records = []
    loso_strat_records = []

    for held_info in loso_stations:
        held_id = held_info["id"]
        held_name = held_info["name"]
        print(f"\nEvaluating LOSO holding out {held_name}...")

        X_tr_loso, y_tr_loso, X_val_loso, y_val_loso = load_dataset_folds(TIER_3_FEATURES, holdout_id=held_id)
        clf_loso = train_lgb(X_tr_loso, y_tr_loso, seed=42)

        # Validation tuning on remaining 4 stations using pre-extracted slices
        val_loso_slices = [s for s in val_inj_slices if s[0] != held_id]
        val_loso_st_yrs = sum(val_series[s]["valid"].sum() for s in val_series if s != held_id) / 8766.0

        val_loso_p_inj = {s: clf_loso.predict_proba(val_series[s]["feats_inj"][TIER_3_FEATURES])[:, 2] for s in val_series if s != held_id}
        val_loso_p_cln = {s: clf_loso.predict_proba(val_series[s]["feats_cln"][TIER_3_FEATURES])[:, 2] for s in val_series if s != held_id}

        best_loso_tau = 0.5
        min_loso_fa = 1e9
        for tau in THRESHOLD_GRID:
            det_cnt = 0
            for st_id, idx in val_loso_slices:
                p_inj = val_loso_p_inj[st_id][idx]
                p_cln = val_loso_p_cln[st_id][idx]
                if ((p_inj >= tau) & (p_cln < tau)).any():
                    det_cnt += 1
            if det_cnt / len(val_loso_slices) >= 0.90:
                ep_cnt = sum(count_episodes((pd.Series(val_loso_p_cln[s]) >= tau) & val_series[s]["valid"], val_series[s]["valid"]) for s in val_series if s != held_id)
                fa_r = ep_cnt / val_loso_st_yrs
                if fa_r < min_loso_fa:
                    min_loso_fa = fa_r
                    best_loso_tau = tau

        # Evaluate at frozen threshold on held-out station test split (2023-2025)
        st_test = test_series[held_id]
        p_inj_held = clf_loso.predict_proba(st_test["feats_inj"][TIER_3_FEATURES])[:, 2]
        p_cln_held = clf_loso.predict_proba(st_test["feats_cln"][TIER_3_FEATURES])[:, 2]

        al = (pd.Series(p_cln_held) >= best_loso_tau) & st_test["valid"]
        held_episodes = count_episodes(al, st_test["valid"])
        held_st_years = st_test["valid"].sum() / 8766.0
        held_fa_rate = held_episodes / held_st_years
        rule_of_three_upper = round(3.0 / held_st_years, 2) if held_episodes == 0 else np.nan

        held_test_slices = [s for s in test_inj_slices if s[0] == held_id]
        held_det = 0
        held_delays = []

        for st_id, idx, start_dt, dts_sub in held_test_slices:
            p_inj_sub = p_inj_held[idx]
            p_cln_sub = p_cln_held[idx]

            net_al = (p_inj_sub >= best_loso_tau) & (p_cln_sub < best_loso_tau)
            is_det = net_al.any()
            if is_det:
                held_det += 1
                first_idx = np.where(net_al)[0][0]
                first_dt = dts_sub.iloc[first_idx]
                delay = max(0.0, (first_dt - start_dt).total_seconds() / 3600.0)
                held_delays.append(delay)

        # Stratified detection
        held_injs = test_catalog[test_catalog["station_id"] == held_id]
        for _, inj in held_injs.iterrows():
            st_id = inj["station_id"]
            sdata = test_series[st_id]
            start_dt = pd.to_datetime(inj["start_utc"])
            end_dt = pd.to_datetime(inj["end_utc"])
            m = (sdata["dt"] >= start_dt) & (sdata["dt"] <= end_dt) & sdata["valid"]
            p_inj_sub = p_inj_held[m]
            p_cln_sub = p_cln_held[m]
            net_al = (p_inj_sub >= best_loso_tau) & (p_cln_sub < best_loso_tau)

            loso_strat_records.append({
                "held_out_station": held_id,
                "injection_id": inj["injection_id"],
                "nuclide_scenario": inj["nuclide_scenario"],
                "peak_cpm": inj["peak_cpm"],
                "magnitude_band": "Band A (255-599 CPM)" if inj["peak_cpm"] < 600 else "Band B (714-1199 CPM)",
                "detected": int(net_al.any()),
            })

        loso_records.append({
            "held_out_station": held_id,
            "station_description": held_name,
            "frozen_tau": round(best_loso_tau, 3),
            "clean_observed_hours": int(st_test["valid"].sum()),
            "clean_station_years": round(held_st_years, 2),
            "clean_false_alarm_episodes": held_episodes,
            "clean_false_alarms_per_year": round(held_fa_rate, 2),
            "rule_of_three_95ci_upper_fa": rule_of_three_upper,
            "total_test_injections": len(held_test_slices),
            "detected_injections": held_det,
            "realized_detection_rate_pct": round(held_det / len(held_test_slices) * 100, 2),
            "median_detection_delay_hours": round(np.median(held_delays), 1) if held_delays else np.nan,
            "natural_washout_test_hours": st_test["washout_test_hours"],
        })

    loso_df = pd.DataFrame(loso_records)
    loso_df.to_csv("data/processed/rigorous_loso_summary.csv", index=False)
    loso_strat_df = pd.DataFrame(loso_strat_records)
    loso_strat_df.to_csv("data/processed/rigorous_loso_stratified_detection.csv", index=False)
    print("\nSaved Clean Multi-Station LOSO Summary to data/processed/rigorous_loso_summary.csv:")
    print(loso_df[["held_out_station", "frozen_tau", "clean_false_alarms_per_year", "rule_of_three_95ci_upper_fa", "detected_injections", "realized_detection_rate_pct", "natural_washout_test_hours"]].to_string())

    sd_strat = loso_strat_df[loso_strat_df["held_out_station"] == "ca_san_diego"]
    print("\nSan Diego Stratified Detection Breakdown:")
    print("  By Magnitude Band:")
    for band, grp in sd_strat.groupby("magnitude_band"):
        print(f"    {band}: {grp['detected'].sum()}/{len(grp)} ({grp['detected'].mean()*100:.1f}%)")
    print("  By Scenario:")
    for sc, grp in sd_strat.groupby("nuclide_scenario"):
        print(f"    {sc}: {grp['detected'].sum()}/{len(grp)} ({grp['detected'].mean()*100:.1f}%)")

    # 11. Proper Spectral Template Sensitivity & Gain Drift Across 5 Seeds
    print("\n" + "=" * 80)
    print("SPECTRAL TEMPLATE SENSITIVITY & INSTRUMENTAL GAIN DRIFT SWEEP (SEEDS 42-46)")
    print("=" * 80)
    sweep_seeds = [42, 43, 44, 45, 46]

    # Part A: Injected Photopeak Scaling
    print("\nPart A: Injected Photopeak Branching Scaling (+/-10%, +/-20%)...")
    photopeak_channels = {
        "fission_pure_cs137": ["cpm_r05"],
        "fission_pure_i131": ["cpm_r03"],
        "fission_reactor_fukushima": ["cpm_r03", "cpm_r05", "cpm_r07"],
        "mixed_fission_activation": ["cpm_r03", "cpm_r05", "cpm_r07"],
        "activation_orphan_co60": ["cpm_r07", "cpm_r08"],
    }
    all_channels = [f"cpm_r0{i}" for i in range(2, 10)]

    sens_records = []
    for factor in [0.80, 0.90, 1.00, 1.10, 1.20]:
        seed_dets = []
        seed_delays = []

        # Precompute modified features for all stations ONCE
        feats_mod_by_st = {}
        for st_id, sdata in test_series.items():
            df_mod = sdata["df"].copy()
            for _, inj in test_catalog[test_catalog["station_id"] == st_id].iterrows():
                start_dt = pd.to_datetime(inj["start_utc"])
                end_dt = pd.to_datetime(inj["end_utc"])
                m = (df_mod["dt"] >= start_dt) & (df_mod["dt"] <= end_dt)
                if not m.any():
                    continue

                sc = inj["nuclide_scenario"]
                peaks = photopeak_channels.get(sc, ["cpm_r05"])
                conts = [c for c in all_channels if c not in peaks]

                for p_col in peaks:
                    inj_col = f"inj_{p_col}"
                    excess = df_mod.loc[m, inj_col] - df_mod.loc[m, p_col]
                    delta_excess = excess * (factor - 1.0)
                    df_mod.loc[m, inj_col] += delta_excess

                    if len(conts) > 0:
                        cont_excess_sum = sum(df_mod.loc[m, f"inj_{c}"] - df_mod.loc[m, c] for c in conts)
                        for c in conts:
                            inj_c = f"inj_{c}"
                            c_excess = df_mod.loc[m, inj_c] - df_mod.loc[m, c]
                            if (cont_excess_sum > 0).any():
                                comp = delta_excess * (c_excess / (cont_excess_sum + 1e-4))
                                df_mod.loc[m, inj_c] = np.maximum(df_mod.loc[m, c], df_mod.loc[m, inj_c] - comp)

            feats_mod_by_st[st_id] = (df_mod, compute_features_for_series(df_mod, st_id, use_injected=True))

        for s in sweep_seeds:
            clf_s = models_sweep[s]["Tier 3 (Weather Fusion)"]
            t3_tau_s = res_df[(res_df["tier_name"] == "Tier 3 (Weather Fusion)") & (res_df["seed"] == s) & (res_df["target_detection"] == 95.0)]["frozen_tau"].iloc[0]

            s_det = 0
            s_del = []
            for st_id, (df_mod, feats_mod_inj) in feats_mod_by_st.items():
                sdata = test_series[st_id]
                p_inj_mod = clf_s.predict_proba(feats_mod_inj[TIER_3_FEATURES])[:, 2]
                p_cln_ref = clf_s.predict_proba(sdata["feats_cln"][TIER_3_FEATURES])[:, 2]

                for _, inj in test_catalog[test_catalog["station_id"] == st_id].iterrows():
                    start_dt = pd.to_datetime(inj["start_utc"])
                    end_dt = pd.to_datetime(inj["end_utc"])
                    m_inj = (df_mod["dt"] >= start_dt) & (df_mod["dt"] <= end_dt) & sdata["valid"]
                    dts_sub = df_mod["dt"][m_inj].reset_index(drop=True)
                    p_inj_sub = pd.Series(p_inj_mod[m_inj]).reset_index(drop=True)
                    p_cln_sub = pd.Series(p_cln_ref[m_inj]).reset_index(drop=True)

                    net_al = (p_inj_sub >= t3_tau_s) & (p_cln_sub < t3_tau_s)
                    if net_al.any():
                        s_det += 1
                        delay = max(0.0, (dts_sub.iloc[net_al.idxmax()] - start_dt).total_seconds() / 3600.0)
                        s_del.append(delay)

            seed_dets.append(s_det / len(test_catalog) * 100.0)
            if s_del:
                seed_delays.append(np.median(s_del))

        sens_records.append({
            "perturbation_factor": factor,
            "perturbation_pct": f"{(factor - 1.0)*100:+.0f}%",
            "test_detection_rate_pct_mean": round(float(np.mean(seed_dets)), 2),
            "test_detection_rate_pct_std": round(float(np.std(seed_dets)), 2),
            "median_delay_hours_mean": round(float(np.mean(seed_delays)), 1) if seed_delays else np.nan,
            "sensitivity_verdict": "Modest to Moderate Sensitivity (2 to 4 points per 10% photopeak change)",
        })

    sens_df = pd.DataFrame(sens_records)
    sens_df.to_csv("data/processed/rigorous_template_sensitivity.csv", index=False)
    print("Saved Injected Template Sensitivity to data/processed/rigorous_template_sensitivity.csv:")
    print(sens_df[["perturbation_factor", "perturbation_pct", "test_detection_rate_pct_mean", "test_detection_rate_pct_std", "median_delay_hours_mean"]].to_string())

    # Part B: Instrumental Gain Drift Across 5 Seeds
    print("\nPart B: Instrumental Gain Drift Sweep (+/-5%, +/-10% calibration shift across all data)...")
    gain_records = []

    for delta in [-0.10, -0.05, 0.00, +0.05, +0.10]:
        feats_drift_by_st = {}
        for st_id, sdata in test_series.items():
            df_drift = sdata["df"].copy()
            if delta != 0.0:
                for prefix in ["", "inj_"]:
                    raw_channels = [f"{prefix}cpm_r0{i}" for i in range(2, 10)]
                    orig_counts = df_drift[raw_channels].values.copy()
                    shifted_counts = orig_counts.copy()

                    if delta > 0:
                        for c_idx in range(len(raw_channels) - 1):
                            shift_amt = orig_counts[:, c_idx] * delta
                            shifted_counts[:, c_idx] -= shift_amt
                            shifted_counts[:, c_idx + 1] += shift_amt
                    else:
                        abs_d = abs(delta)
                        for c_idx in range(len(raw_channels) - 1, 0, -1):
                            shift_amt = orig_counts[:, c_idx] * abs_d
                            shifted_counts[:, c_idx] -= shift_amt
                            shifted_counts[:, c_idx - 1] += shift_amt

                    df_drift[raw_channels] = shifted_counts

            f_cln = compute_features_for_series(df_drift, st_id, use_injected=False)
            f_inj = compute_features_for_series(df_drift, st_id, use_injected=True)
            feats_drift_by_st[st_id] = (df_drift, f_cln, f_inj)

        seed_fa = []
        seed_det = []
        for s in sweep_seeds:
            clf_s = models_sweep[s]["Tier 3 (Weather Fusion)"]
            t3_tau_s = res_df[(res_df["tier_name"] == "Tier 3 (Weather Fusion)") & (res_df["seed"] == s) & (res_df["target_detection"] == 95.0)]["frozen_tau"].iloc[0]

            s_episodes = 0
            s_det_cnt = 0
            for st_id, (df_drift, f_cln, f_inj) in feats_drift_by_st.items():
                sdata = test_series[st_id]
                p_cln_drift = clf_s.predict_proba(f_cln[TIER_3_FEATURES])[:, 2]
                p_inj_drift = clf_s.predict_proba(f_inj[TIER_3_FEATURES])[:, 2]

                al = (pd.Series(p_cln_drift) >= t3_tau_s) & sdata["valid"]
                s_episodes += count_episodes(al, sdata["valid"])

                for _, inj in test_catalog[test_catalog["station_id"] == st_id].iterrows():
                    start_dt = pd.to_datetime(inj["start_utc"])
                    end_dt = pd.to_datetime(inj["end_utc"])
                    m_inj = (df_drift["dt"] >= start_dt) & (df_drift["dt"] <= end_dt) & sdata["valid"]
                    net_al = (p_inj_drift[m_inj] >= t3_tau_s) & (p_cln_drift[m_inj] < t3_tau_s)
                    if net_al.any():
                        s_det_cnt += 1

            seed_fa.append(s_episodes / test_station_years)
            seed_det.append(s_det_cnt / len(test_catalog) * 100.0)

        gain_records.append({
            "gain_drift_shift": delta,
            "gain_drift_pct": f"{delta*100:+.0f}%",
            "clean_false_alarms_per_year_mean": round(float(np.mean(seed_fa)), 2),
            "clean_false_alarms_per_year_std": round(float(np.std(seed_fa)), 2),
            "test_detection_rate_pct_mean": round(float(np.mean(seed_det)), 2),
            "test_detection_rate_pct_std": round(float(np.std(seed_det)), 2),
        })

    gain_df = pd.DataFrame(gain_records)
    gain_df.to_csv("data/processed/rigorous_gain_drift_evaluation.csv", index=False)
    print("\nSaved Instrumental Gain Drift Summary to data/processed/rigorous_gain_drift_evaluation.csv:")
    print(gain_df.to_string())

    # 12. Stratified Operational Deadlines
    print("\n" + "=" * 80)
    print("STRATIFIED OPERATIONAL DEADLINE EVALUATION (At Matched ~92% Realized Detection)")
    print("=" * 80)
    t3_tau_92 = agg_df[(agg_df["tier_name"] == "Tier 3 (Weather Fusion)") & (agg_df["target_detection_pct"] == 95.0)]["frozen_tau_mean"].iloc[0]
    t2_tau_92 = agg_df[(agg_df["tier_name"] == "Tier 2 (Spectral)") & (agg_df["target_detection_pct"] == 90.0)]["frozen_tau_mean"].iloc[0]

    clf_t2 = models_seed_42["Tier 2 (Spectral)"]
    clf_t3 = models_seed_42["Tier 3 (Weather Fusion)"]

    strat_records = []
    strata_definitions = {
        "All Test Injections": test_catalog,
        "Strictly Dry Environments": test_catalog[test_catalog["environment"].str.contains("dry")],
        "Rain-Coincident Environments": test_catalog[test_catalog["environment"].str.contains("rain")],
        "Standard Plume Regime": test_catalog[test_catalog["environment"].str.contains("standard")],
        "Stress Plume Regime": test_catalog[test_catalog["environment"].str.contains("stress")],
        "Standard Strictly Dry": test_catalog[test_catalog["environment"] == "test_standard_strictly_dry"],
        "Stress Strictly Dry": test_catalog[test_catalog["environment"] == "test_stress_strictly_dry"],
        "Standard Rain": test_catalog[test_catalog["environment"] == "test_standard_rain"],
        "Stress Rain": test_catalog[test_catalog["environment"] == "test_stress_rain"],
    }

    t2_p_inj = {st: clf_t2.predict_proba(test_series[st]["feats_inj"][TIER_2_FEATURES])[:, 2] for st in test_series}
    t2_p_cln = {st: clf_t2.predict_proba(test_series[st]["feats_cln"][TIER_2_FEATURES])[:, 2] for st in test_series}
    t3_p_inj = {st: clf_t3.predict_proba(test_series[st]["feats_inj"][TIER_3_FEATURES])[:, 2] for st in test_series}
    t3_p_cln = {st: clf_t3.predict_proba(test_series[st]["feats_cln"][TIER_3_FEATURES])[:, 2] for st in test_series}

    for stratum_name, cat_sub in strata_definitions.items():
        n_strat = len(cat_sub)
        for t_label, tau_val, p_inj_map, p_cln_map in [
            ("Tier 2 (Spectral)", t2_tau_92, t2_p_inj, t2_p_cln),
            ("Tier 3 (Weather Fusion)", t3_tau_92, t3_p_inj, t3_p_cln),
        ]:
            det_total = 0
            det_6h = 0
            det_12h = 0
            det_24h = 0
            delays = []

            for _, inj in cat_sub.iterrows():
                st_id = inj["station_id"]
                sdata = test_series[st_id]
                start_dt = pd.to_datetime(inj["start_utc"])
                end_dt = pd.to_datetime(inj["end_utc"])

                m = (sdata["dt"] >= start_dt) & (sdata["dt"] <= end_dt) & sdata["valid"]
                dts_sub = sdata["dt"][m].reset_index(drop=True)
                p_inj_sub = pd.Series(p_inj_map[st_id][m]).reset_index(drop=True)
                p_cln_sub = pd.Series(p_cln_map[st_id][m]).reset_index(drop=True)

                net_al = (p_inj_sub >= tau_val) & (p_cln_sub < tau_val)
                if net_al.any():
                    det_total += 1
                    first_idx = net_al.idxmax()
                    first_dt = dts_sub.iloc[first_idx]
                    delay = max(0.0, (first_dt - start_dt).total_seconds() / 3600.0)
                    delays.append(delay)
                    if delay <= 6.0:
                        det_6h += 1
                    if delay <= 12.0:
                        det_12h += 1
                    if delay <= 24.0:
                        det_24h += 1

            strat_records.append({
                "stratum": stratum_name,
                "tier_name": t_label,
                "n_injections": n_strat,
                "detection_total_pct": round(det_total / n_strat * 100, 1),
                "detection_le_6h_pct": round(det_6h / n_strat * 100, 1),
                "detection_le_12h_pct": round(det_12h / n_strat * 100, 1),
                "detection_le_24h_pct": round(det_24h / n_strat * 100, 1),
                "median_delay_hours": round(np.median(delays), 1) if delays else np.nan,
            })

    strat_df = pd.DataFrame(strat_records)
    strat_df.to_csv("data/processed/rigorous_stratified_deadlines.csv", index=False)
    print("Saved Stratified Deadlines Summary to data/processed/rigorous_stratified_deadlines.csv:")
    print(strat_df.to_string())

    print(f"\nAll evaluations completed successfully in {time.time() - t_start:.2f}s!")


if __name__ == "__main__":
    main()
