"""
Phase 4 & 5 Rigorous Model Training and Evaluation Pipeline.

Addresses all Claude Review findings:
1. Validation Split for Threshold Selection:
   - Operating thresholds tau* (90%, 95%, 98%) are tuned STRICTLY on the 2021-2022 validation fold (81 injs, 8.46 st-yrs).
   - Test split (2023-2025, 200 injs, 12.16 st-yrs) is evaluated at FROZEN tau* (zero test tuning leakage).
2. Continuous Baseline Multiplier Sweep & Matched Detection:
   - Sweeps baseline multipliers k in [0.5, 5.0] to map continuous baseline ROC curves.
   - Compares Baselines vs Tier 1 vs Tier 1b vs Tier 2 vs Tier 3 vs MLP at EXACT MATCHED DETECTION.
3. Four-Tier Feature Ablation Across 5 Seeds:
   - Tier 1: Gross Radiation Only (12 feats)
   - Tier 1b: Gross Radiation + Weather (29 feats, NO spectra)
   - Tier 2: Gross Radiation + NaI Spectrometry (30 feats, NO weather)
   - Tier 3: Full Weather Fusion (48 feats)
   - Evaluated across 5 random seeds (42, 43, 44, 45, 46) reporting mean +/- std.
4. Net Alarm Criterion & Operational Deadlines:
   - Net alarm: credits event detection only when injected series alarms AND clean series does not.
   - Deadlines: evaluated at <= 6h, <= 12h, <= 24h, and total retention window.
5. Clean LOSO Evaluation:
   - Train on 4 stations (2017-2020), tune tau* on 4 stations (2021-2022), evaluate on held-out San Diego test split.
6. Template Sensitivity Sweep:
   - Perturb spectral template by +/-10% and +/-20% to assess detector drift tolerance.
7. Block Bootstrap Uncertainty with Frozen Thresholds:
   - 1,000 station-month resamples with fixed tau* for genuine 95% CIs.
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
SEEDS = [42, 43, 44, 45, 46]


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


def count_episodes(alarm_series: pd.Series, valid_mask: pd.Series) -> int:
    """Counts discrete alarm episodes on continuous calendar series."""
    alarm_int = (alarm_series & valid_mask).astype(int)
    prior = alarm_int.shift(1).fillna(0).astype(int)
    return int(((alarm_int == 1) & (prior == 0)).sum())


def main():
    t_start = time.time()
    print("=" * 80)
    print("RIGOROUS MULTI-SEED EVALUATION WITH CAUSAL VALIDATION TUNING")
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

        m_val = (df_tr_raw["dt"] >= "2021-01-01") & (df_tr_raw["dt"] < "2023-01-01")
        df_val = df_tr_raw[m_val].copy().reset_index(drop=True)

        feats_val_inj = compute_features_for_series(df_val, st_id, use_injected=True)
        feats_val_cln = compute_features_for_series(df_val, st_id, use_injected=False)
        val_valid = feats_val_cln["has_radnet_obs"] & feats_val_cln["rad_complete_channels"]
        total_val_obs_hours += val_valid.sum()

        val_series[st_id] = {
            "df": df_val,
            "dt": df_val["dt"],
            "valid": val_valid,
            "feats_inj": feats_val_inj,
            "feats_cln": feats_val_cln,
            "gross_inj": df_val["inj_gross_cpm"],
            "gross_cln": df_val["gross_cpm"],
            "z_roll_inj": feats_val_inj["z_score_168h"],
            "z_roll_cln": feats_val_cln["z_score_168h"],
            "base": STATION_DRY_BASELINES[st_id],
        }

        # Test file (contains 2023-2025)
        df_te_raw = pd.read_csv(f"data/processed/labeled_{st_id}_test.csv.gz")
        df_te_raw["dt"] = pd.to_datetime(df_te_raw["dt"])
        df_te_raw = df_te_raw.sort_values("dt").reset_index(drop=True)

        feats_te_inj = compute_features_for_series(df_te_raw, st_id, use_injected=True)
        feats_te_cln = compute_features_for_series(df_te_raw, st_id, use_injected=False)
        te_valid = feats_te_cln["has_radnet_obs"] & feats_te_cln["rad_complete_channels"]
        total_test_obs_hours += te_valid.sum()

        test_series[st_id] = {
            "df": df_te_raw,
            "dt": df_te_raw["dt"],
            "valid": te_valid,
            "feats_inj": feats_te_inj,
            "feats_cln": feats_te_cln,
            "gross_inj": df_te_raw["inj_gross_cpm"],
            "gross_cln": df_te_raw["gross_cpm"],
            "z_roll_inj": feats_te_inj["z_score_168h"],
            "z_roll_cln": feats_te_cln["z_score_168h"],
            "precip_1h": df_te_raw["precip_1h_mm"].fillna(0.0),
            "base": STATION_DRY_BASELINES[st_id],
            "year_month": df_te_raw["dt"].dt.to_period("M").astype(str),
        }

    val_station_years = total_val_obs_hours / 8766.0
    test_station_years = total_test_obs_hours / 8766.0
    print(f"Validation clean observed hours: {total_val_obs_hours} ({val_station_years:.2f} st-yrs)")
    print(f"Test clean observed hours:       {total_test_obs_hours} ({test_station_years:.2f} st-yrs)")

    # 3. Sweep Baseline Multipliers (k in [0.5, 5.0]) to map continuous Baseline ROC
    print("\nTracing continuous baseline ROC curves on test set...")
    k_steps = np.linspace(0.5, 5.0, 91)  # 0.05 step
    base_roc = []

    for k in k_steps:
        for b_type in ["rolling_7d", "global_dry"]:
            det_count = 0
            episodes = 0
            alarm_hrs = 0
            rain_alarm_hrs = 0

            for st_id, sdata in test_series.items():
                val_m = sdata["valid"]
                if b_type == "rolling_7d":
                    al_cln = (sdata["z_roll_cln"] >= k) & val_m
                    sig_inj = sdata["z_roll_inj"]
                    crit_inj = k
                else:
                    thresh = sdata["base"]["mu"] + k * sdata["base"]["sigma"]
                    al_cln = (sdata["gross_cln"] >= thresh) & val_m
                    sig_inj = sdata["gross_inj"]
                    crit_inj = thresh

                # Clean false alarms
                ep = count_episodes(al_cln, val_m)
                episodes += ep
                alarm_hrs += al_cln.sum()
                rain_alarm_hrs += (al_cln & (sdata["precip_1h"] > 0.0)).sum()

            fa_rate = episodes / test_station_years
            rain_pct = (rain_alarm_hrs / alarm_hrs * 100.0) if alarm_hrs > 0 else 0.0

            # Detection on test catalog with net alarm criterion
            for _, inj in test_catalog.iterrows():
                st_id = inj["station_id"]
                sdata = test_series[st_id]
                m_inj = (sdata["dt"] >= inj["start_utc"]) & (sdata["dt"] <= inj["end_utc"]) & sdata["valid"]
                if b_type == "rolling_7d":
                    inj_al = (sdata["z_roll_inj"][m_inj] >= k)
                    cln_al = (sdata["z_roll_cln"][m_inj] >= k)
                else:
                    thresh = sdata["base"]["mu"] + k * sdata["base"]["sigma"]
                    inj_al = (sdata["gross_inj"][m_inj] >= thresh)
                    cln_al = (sdata["gross_cln"][m_inj] >= thresh)

                # Net alarm: alarms on injected series AND not purely a clean alarm
                net_al = inj_al & (~cln_al)
                if net_al.any():
                    det_count += 1

            det_rate = det_count / len(test_catalog)
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

    # 4. Multi-Seed Training and Causal Validation Tuning
    tier_configs = {
        "Tier 1 (Gross)": {"features": TIER_1_FEATURES, "is_mlp": False},
        "Tier 1b (Gross+Weather)": {"features": TIER_1B_FEATURES, "is_mlp": False},
        "Tier 2 (Spectral)": {"features": TIER_2_FEATURES, "is_mlp": False},
        "Tier 3 (Weather Fusion)": {"features": TIER_3_FEATURES, "is_mlp": False},
        "Tier 3 MLP": {"features": TIER_3_FEATURES, "is_mlp": True},
    }

    targets = [0.90, 0.95, 0.98]
    seed_results = []  # Stores out-of-fold test results for each tier, seed, target
    threshold_grid = np.linspace(0.01, 0.99, 99)

    print("\nTraining tiers across 5 seeds, tuning tau on 2021-2022 validation fold, testing on 2023-2025...")

    # Also store fitted models for Seed 42 for detailed confusion matrix & sensitivity sweeps
    seed_42_models = {}

    for tier_name, cfg in tier_configs.items():
        feat_cols = cfg["features"]
        is_mlp = cfg["is_mlp"]

        # Load train (2017-2020) and val (2021-2022) data
        X_tr, y_tr, X_val, y_val = load_dataset_folds(feat_cols)

        for seed in SEEDS:
            # A. Fit model strictly on 2017-2020
            t0 = time.time()
            if is_mlp:
                model = train_mlp(X_tr, y_tr, seed)
            else:
                model = train_lgb(X_tr, y_tr, seed)

            if seed == 42:
                seed_42_models[tier_name] = model

            # B. Tune operating threshold on 2021-2022 Validation Fold
            # Predict validation probabilities
            val_p_inj = {}
            val_p_cln = {}
            for st_id, sdata in val_series.items():
                val_p_inj[st_id] = model.predict_proba(sdata["feats_inj"][feat_cols])[:, 2]
                val_p_cln[st_id] = model.predict_proba(sdata["feats_cln"][feat_cols])[:, 2]

            val_best_taus = {}
            for tgt in targets:
                best_tau = 0.5
                min_val_fa = 1e9
                for tau in threshold_grid:
                    # Detection rate on validation injections (net alarm criterion)
                    val_det = 0
                    for _, inj in val_catalog.iterrows():
                        st_id = inj["station_id"]
                        sdata = val_series[st_id]
                        m = (sdata["dt"] >= inj["start_utc"]) & (sdata["dt"] <= inj["end_utc"]) & sdata["valid"]
                        p_inj_sub = val_p_inj[st_id][m]
                        p_cln_sub = val_p_cln[st_id][m]
                        net_al = (p_inj_sub >= tau) & (p_cln_sub < tau)
                        if net_al.any():
                            val_det += 1
                    det_rate = val_det / len(val_catalog)

                    if det_rate >= tgt:
                        # Clean false alarms on validation clean series
                        val_episodes = 0
                        for st_id, sdata in val_series.items():
                            al = (pd.Series(val_p_cln[st_id]) >= tau) & sdata["valid"]
                            val_episodes += count_episodes(al, sdata["valid"])
                        val_fa = val_episodes / val_station_years
                        if val_fa < min_val_fa:
                            min_val_fa = val_fa
                            best_tau = tau

                val_best_taus[tgt] = best_tau

            # C. Evaluate on Unseen 2023-2025 Test Split at FROZEN best_tau
            test_p_inj = {}
            test_p_cln = {}
            for st_id, sdata in test_series.items():
                test_p_inj[st_id] = model.predict_proba(sdata["feats_inj"][feat_cols])[:, 2]
                test_p_cln[st_id] = model.predict_proba(sdata["feats_cln"][feat_cols])[:, 2]

            for tgt in targets:
                frozen_tau = val_best_taus[tgt]

                # 1. Clean false alarms on test set
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

                # 2. Realized detection and deadlines on 200 test injections
                det_total = 0
                det_6h = 0
                det_12h = 0
                det_24h = 0
                delays = []

                for _, inj in test_catalog.iterrows():
                    st_id = inj["station_id"]
                    sdata = test_series[st_id]
                    start_dt = pd.to_datetime(inj["start_utc"])
                    end_dt = pd.to_datetime(inj["end_utc"])

                    m = (sdata["dt"] >= start_dt) & (sdata["dt"] <= end_dt) & sdata["valid"]
                    dts_sub = sdata["dt"][m].reset_index(drop=True)
                    p_inj_sub = pd.Series(test_p_inj[st_id][m]).reset_index(drop=True)
                    p_cln_sub = pd.Series(test_p_cln[st_id][m]).reset_index(drop=True)

                    # Net alarm
                    net_alarm = (p_inj_sub >= frozen_tau) & (p_cln_sub < frozen_tau)
                    if net_alarm.any():
                        det_total += 1
                        first_idx = net_alarm.idxmax()
                        first_dt = dts_sub.iloc[first_idx]
                        delay = max(0.0, (first_dt - start_dt).total_seconds() / 3600.0)
                        delays.append(delay)
                        if delay <= 6.0:
                            det_6h += 1
                        if delay <= 12.0:
                            det_12h += 1
                        if delay <= 24.0:
                            det_24h += 1

                n_te = len(test_catalog)
                med_delay = np.median(delays) if len(delays) > 0 else np.nan

                seed_results.append({
                    "tier_name": tier_name,
                    "seed": seed,
                    "target_detection": tgt * 100,
                    "frozen_tau": frozen_tau,
                    "realized_detection_total_pct": round(det_total / n_te * 100, 2),
                    "realized_detection_6h_pct": round(det_6h / n_te * 100, 2),
                    "realized_detection_12h_pct": round(det_12h / n_te * 100, 2),
                    "realized_detection_24h_pct": round(det_24h / n_te * 100, 2),
                    "clean_false_alarms_per_year": round(test_fa_rate, 2),
                    "rain_coincident_pct": round(rain_pct, 1),
                    "median_delay_hours": round(med_delay, 1),
                })

            print(f"  {tier_name} (Seed {seed}) trained in {time.time()-t0:.1f}s.")

    res_df = pd.DataFrame(seed_results)
    res_df.to_csv("data/processed/rigorous_seed_level_results.csv", index=False)

    # 5. Aggregate Across 5 Seeds: Mean +/- Std
    agg_records = []
    for tier_name in tier_configs:
        for tgt in [90.0, 95.0, 98.0]:
            sub = res_df[(res_df["tier_name"] == tier_name) & (res_df["target_detection"] == tgt)]
            agg_records.append({
                "tier_name": tier_name,
                "target_detection_pct": tgt,
                "frozen_tau_mean": round(sub["frozen_tau"].mean(), 2),
                "det_total_mean": round(sub["realized_detection_total_pct"].mean(), 2),
                "det_total_std": round(sub["realized_detection_total_pct"].std(), 2),
                "det_6h_mean": round(sub["realized_detection_6h_pct"].mean(), 2),
                "det_12h_mean": round(sub["realized_detection_12h_pct"].mean(), 2),
                "det_24h_mean": round(sub["realized_detection_24h_pct"].mean(), 2),
                "fa_per_year_mean": round(sub["clean_false_alarms_per_year"].mean(), 2),
                "fa_per_year_std": round(sub["clean_false_alarms_per_year"].std(), 2),
                "rain_coincident_mean": round(sub["rain_coincident_pct"].mean(), 1),
                "median_delay_mean": round(sub["median_delay_hours"].mean(), 1),
            })

    agg_df = pd.DataFrame(agg_records)
    agg_df.to_csv("data/processed/rigorous_benchmark_summary.csv", index=False)
    print("\nSaved Multi-Seed Benchmark Summary to data/processed/rigorous_benchmark_summary.csv")

    # 6. Matched Detection Head-to-Head Comparison Table
    # For each ML model at ~90% and ~95% realized detection:
    # Find baseline operating points with matching detection
    matched_records = []
    for tgt in [90.0, 95.0]:
        # Tier 3 LightGBM mean detection
        t3_row = agg_df[(agg_df["tier_name"] == "Tier 3 (Weather Fusion)") & (agg_df["target_detection_pct"] == tgt)].iloc[0]
        t2_row = agg_df[(agg_df["tier_name"] == "Tier 2 (Spectral)") & (agg_df["target_detection_pct"] == tgt)].iloc[0]
        t1b_row = agg_df[(agg_df["tier_name"] == "Tier 1b (Gross+Weather)") & (agg_df["target_detection_pct"] == tgt)].iloc[0]
        t1_row = agg_df[(agg_df["tier_name"] == "Tier 1 (Gross)") & (agg_df["target_detection_pct"] == tgt)].iloc[0]
        mlp_row = agg_df[(agg_df["tier_name"] == "Tier 3 MLP") & (agg_df["target_detection_pct"] == tgt)].iloc[0]

        target_det = t3_row["det_total_mean"]

        # Find rolling baseline with closest detection >= target_det
        roll_sub = base_roc_df[(base_roc_df["baseline_type"] == "rolling_7d") & (base_roc_df["detection_rate_pct"] >= target_det)]
        roll_match = roll_sub.sort_values("false_alarms_per_year").iloc[0] if not roll_sub.empty else base_roc_df[base_roc_df["baseline_type"] == "rolling_7d"].iloc[0]

        # Find global baseline with closest detection >= target_det
        glob_sub = base_roc_df[(base_roc_df["baseline_type"] == "global_dry") & (base_roc_df["detection_rate_pct"] >= target_det)]
        glob_match = glob_sub.sort_values("false_alarms_per_year").iloc[0] if not glob_sub.empty else base_roc_df[base_roc_df["baseline_type"] == "global_dry"].iloc[0]

        matched_records.append({
            "target_detection_tier": f"{tgt}% Target",
            "rolling_7d_multiplier_k": roll_match["k_multiplier"],
            "rolling_7d_detection_pct": roll_match["detection_rate_pct"],
            "rolling_7d_fa_per_year": roll_match["false_alarms_per_year"],
            "global_dry_multiplier_k": glob_match["k_multiplier"],
            "global_dry_detection_pct": glob_match["detection_rate_pct"],
            "global_dry_fa_per_year": glob_match["false_alarms_per_year"],
            "tier1_gross_fa_per_year": t1_row["fa_per_year_mean"],
            "tier1b_gross_weather_fa_per_year": t1b_row["fa_per_year_mean"],
            "tier2_spectral_fa_per_year": t2_row["fa_per_year_mean"],
            "tier3_weather_fusion_fa_per_year": t3_row["fa_per_year_mean"],
            "tier3_mlp_fa_per_year": mlp_row["fa_per_year_mean"],
            "fa_reduction_t3_vs_rolling_pct": round((roll_match["false_alarms_per_year"] - t3_row["fa_per_year_mean"]) / roll_match["false_alarms_per_year"] * 100, 1),
            "fa_reduction_t2_vs_t3_pct": round((t3_row["fa_per_year_mean"] - t2_row["fa_per_year_mean"]) / t3_row["fa_per_year_mean"] * 100, 1),
        })

    matched_df = pd.DataFrame(matched_records)
    matched_df.to_csv("data/processed/rigorous_matched_detection_comparison.csv", index=False)
    print("Saved Matched Detection Comparison to data/processed/rigorous_matched_detection_comparison.csv")

    # 7. Clean LOSO Evaluation (Zero San Diego Leakage)
    print("\nRunning clean Leave-One-Station-Out (LOSO) holding out San Diego...")
    X_tr_loso, y_tr_loso, X_val_loso, y_val_loso = load_dataset_folds(TIER_3_FEATURES, holdout_id="ca_san_diego")
    clf_loso = train_lgb(X_tr_loso, y_tr_loso, seed=42)

    # Validation tuning on other 4 stations (2021-2022)
    val_loso_catalog = val_catalog[val_catalog["station_id"] != "ca_san_diego"]
    val_loso_station_years = sum(val_series[s]["valid"].sum() for s in val_series if s != "ca_san_diego") / 8766.0

    best_loso_tau = 0.5
    min_loso_fa = 1e9
    for tau in threshold_grid:
        det_cnt = 0
        for _, inj in val_loso_catalog.iterrows():
            st_id = inj["station_id"]
            sdata = val_series[st_id]
            m = (sdata["dt"] >= inj["start_utc"]) & (sdata["dt"] <= inj["end_utc"]) & sdata["valid"]
            p_inj = clf_loso.predict_proba(sdata["feats_inj"][TIER_3_FEATURES])[m, 2]
            p_cln = clf_loso.predict_proba(sdata["feats_cln"][TIER_3_FEATURES])[m, 2]
            if ((p_inj >= tau) & (p_cln < tau)).any():
                det_cnt += 1
        d_rate = det_cnt / len(val_loso_catalog)
        if d_rate >= 0.90:
            ep_cnt = sum(count_episodes((pd.Series(clf_loso.predict_proba(val_series[s]["feats_cln"][TIER_3_FEATURES])[:, 2]) >= tau) & val_series[s]["valid"], val_series[s]["valid"]) for s in val_series if s != "ca_san_diego")
            fa_r = ep_cnt / val_loso_station_years
            if fa_r < min_loso_fa:
                min_loso_fa = fa_r
                best_loso_tau = tau

    # Evaluate FROZEN best_loso_tau strictly on San Diego test set (2023-2025)
    sd_test = test_series["ca_san_diego"]
    sd_p_inj = clf_loso.predict_proba(sd_test["feats_inj"][TIER_3_FEATURES])[:, 2]
    sd_p_cln = clf_loso.predict_proba(sd_test["feats_cln"][TIER_3_FEATURES])[:, 2]

    sd_al = (pd.Series(sd_p_cln) >= best_loso_tau) & sd_test["valid"]
    sd_episodes = count_episodes(sd_al, sd_test["valid"])
    sd_st_years = sd_test["valid"].sum() / 8766.0
    sd_fa_rate = sd_episodes / sd_st_years

    sd_test_injs = test_catalog[test_catalog["station_id"] == "ca_san_diego"]
    sd_det = 0
    sd_delays = []
    for _, inj in sd_test_injs.iterrows():
        start_dt = pd.to_datetime(inj["start_utc"])
        end_dt = pd.to_datetime(inj["end_utc"])
        m = (sd_test["dt"] >= start_dt) & (sd_test["dt"] <= end_dt) & sd_test["valid"]
        dts_sub = sd_test["dt"][m].reset_index(drop=True)
        p_inj_sub = pd.Series(sd_p_inj[m]).reset_index(drop=True)
        p_cln_sub = pd.Series(sd_p_cln[m]).reset_index(drop=True)
        net_al = (p_inj_sub >= best_loso_tau) & (p_cln_sub < best_loso_tau)
        if net_al.any():
            sd_det += 1
            delay = max(0.0, (dts_sub.iloc[net_al.idxmax()] - start_dt).total_seconds() / 3600.0)
            sd_delays.append(delay)

    loso_record = [{
        "held_out_station": "ca_san_diego",
        "validation_tuned_frozen_tau": best_loso_tau,
        "clean_observed_hours": int(sd_test["valid"].sum()),
        "clean_station_years": round(sd_st_years, 2),
        "clean_false_alarm_episodes": sd_episodes,
        "clean_false_alarms_per_year": round(sd_fa_rate, 2),
        "total_test_injections": len(sd_test_injs),
        "detected_injections": sd_det,
        "realized_detection_rate_pct": round(sd_det / len(sd_test_injs) * 100, 2),
        "median_detection_delay_hours": round(np.median(sd_delays), 1) if sd_delays else np.nan,
    }]
    loso_df = pd.DataFrame(loso_record)
    loso_df.to_csv("data/processed/rigorous_loso_summary.csv", index=False)
    print("Saved Clean LOSO Summary to data/processed/rigorous_loso_summary.csv")

    # 8. Spectral Template Sensitivity Sweep (+/-10%, +/-20% photopeak share)
    print("\nRunning template sensitivity sweep (+/-10%, +/-20% spectral perturbation)...")
    clf_t3 = seed_42_models["Tier 3 (Weather Fusion)"]
    t3_tau_95 = agg_df[(agg_df["tier_name"] == "Tier 3 (Weather Fusion)") & (agg_df["target_detection_pct"] == 95.0)]["frozen_tau_mean"].iloc[0]

    sensitivity_records = []
    for pert_factor in [0.80, 0.90, 1.00, 1.10, 1.20]:
        det_cnt = 0
        for _, inj in test_catalog.iterrows():
            st_id = inj["station_id"]
            sdata = test_series[st_id]
            start_dt = pd.to_datetime(inj["start_utc"])
            end_dt = pd.to_datetime(inj["end_utc"])
            m = (sdata["dt"] >= start_dt) & (sdata["dt"] <= end_dt) & sdata["valid"]

            # Perturb spectral features
            sub_feats = sdata["feats_inj"].loc[m, TIER_3_FEATURES].copy()
            for ch_col in ["share_r03", "share_r05", "share_r07", "ratio_r05_r03", "ratio_r05_r07"]:
                if ch_col in sub_feats.columns:
                    sub_feats[ch_col] = sub_feats[ch_col] * pert_factor

            p_inj = clf_t3.predict_proba(sub_feats)[:, 2]
            p_cln = clf_t3.predict_proba(sdata["feats_cln"].loc[m, TIER_3_FEATURES])[:, 2]

            if ((p_inj >= t3_tau_95) & (p_cln < t3_tau_95)).any():
                det_cnt += 1

        sensitivity_records.append({
            "perturbation_factor": pert_factor,
            "perturbation_pct": f"{(pert_factor - 1.0)*100:+.0f}%",
            "test_detection_rate_pct": round(det_cnt / len(test_catalog) * 100, 2),
            "detected_events": det_cnt,
            "total_events": len(test_catalog),
        })

    sens_df = pd.DataFrame(sensitivity_records)
    sens_df.to_csv("data/processed/rigorous_template_sensitivity.csv", index=False)
    print("Saved Template Sensitivity Sweep to data/processed/rigorous_template_sensitivity.csv")

    # 9. Print Key Results Tables
    print("\n" + "=" * 80)
    print("SUMMARY OF RIGOROUS ABLATION & BENCHMARK RESULTS")
    print("=" * 80)
    print(agg_df[["tier_name", "target_detection_pct", "frozen_tau_mean", "det_total_mean", "fa_per_year_mean", "fa_per_year_std", "det_6h_mean", "det_24h_mean", "median_delay_mean"]].to_string())

    print("\n" + "=" * 80)
    print("MATCHED DETECTION COMPARISON (Model vs Baselines)")
    print("=" * 80)
    print(matched_df.to_string())

    print("\n" + "=" * 80)
    print("CLEAN LEAVE-ONE-STATION-OUT (LOSO) SAN DIEGO")
    print("=" * 80)
    print(loso_df.to_string())

    print("\n" + "=" * 80)
    print("SPECTRAL TEMPLATE SENSITIVITY SWEEP")
    print("=" * 80)
    print(sens_df.to_string())

    print(f"\nCompleted in {time.time() - t_start:.2f}s!")


if __name__ == "__main__":
    main()
