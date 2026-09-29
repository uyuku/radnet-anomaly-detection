"""
Phase 4 Model Training and Evaluation Pipeline.

Implements:
1. Dual-series feature generation across all 5 pilot stations (2017-2022 train, 2023-2025 test).
2. Three-tier feature ablation hierarchy:
   - Tier 1: Gross Radiation Only
   - Tier 2: Radiation + Spectral Ratios
   - Tier 3: Full Weather Fusion
3. Multi-class LightGBM modeling (normal vs radon_washout vs fission_product).
4. Continuous threshold sweep on P(fission) to benchmark against Phase 2 fixed-threshold baselines:
   - Event-level detection probability on injected test series.
   - Clean operational false alarms per station-year on unmodified clean background series.
5. Stratified performance breakdown:
   - By environmental regime (dry vs rain onset).
   - By test set (standard interpolation vs hard-regime subtle stress).
   - By operational scenario (Cs-137, I-131, Fukushima, Mixed, Co-60).
6. Leave-One-Station-Out (LOSO) cross-validation holding out San Diego.
7. Saves summary CSV tables in data/processed/.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
import time

from src.feature_engineering import (
    compute_features_for_series,
    TIER_1_FEATURES,
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
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_and_prepare_train_data(feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads training data across all 5 stations (2017-2022) with injected features.
    Filters out unobserved records.
    """
    x_list = []
    y_list = []

    for st in STATIONS:
        st_id = st["id"]
        csv_file = Path(f"data/processed/labeled_{st_id}_train.csv.gz")
        df_raw = pd.read_csv(csv_file)
        feats = compute_features_for_series(df_raw, st_id, use_injected=True)

        valid = feats["has_radnet_obs"] & feats["rad_complete_channels"] & (feats["label"].isin(LABEL_MAP.keys()))
        sub = feats[valid].copy()

        x_list.append(sub[feature_cols])
        y_list.append(sub["label"].map(LABEL_MAP))

    X = pd.concat(x_list, ignore_index=True)
    y = pd.concat(y_list, ignore_index=True)
    return X, y


def load_and_prepare_train_data_loso(feature_cols: list[str], holdout_id: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads training data excluding the holdout station for LOSO evaluation.
    """
    x_list = []
    y_list = []

    for st in STATIONS:
        if st["id"] == holdout_id:
            continue
        st_id = st["id"]
        csv_file = Path(f"data/processed/labeled_{st_id}_train.csv.gz")
        df_raw = pd.read_csv(csv_file)
        feats = compute_features_for_series(df_raw, st_id, use_injected=True)

        valid = feats["has_radnet_obs"] & feats["rad_complete_channels"] & (feats["label"].isin(LABEL_MAP.keys()))
        sub = feats[valid].copy()

        x_list.append(sub[feature_cols])
        y_list.append(sub["label"].map(LABEL_MAP))

    X = pd.concat(x_list, ignore_index=True)
    y = pd.concat(y_list, ignore_index=True)
    return X, y


def train_tier_model(X_train: pd.DataFrame, y_train: pd.Series, tier_name: str) -> lgb.LGBMClassifier:
    """
    Trains a multi-class LightGBM classifier with balanced class weighting.
    """
    print(f"Training {tier_name} model on {len(X_train)} samples with {X_train.shape[1]} features...")
    t0 = time.time()
    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    print(f"Finished {tier_name} training in {time.time() - t0:.2f}s.")
    return clf


def count_clean_alarm_episodes(p_fission: pd.Series, valid_mask: pd.Series, threshold: float) -> tuple[int, int]:
    """
    Counts discrete alarm episodes on continuous calendar grid for clean series.
    Returns (total_episodes, total_alarm_hours).
    """
    alarm_mask = (p_fission >= threshold) & valid_mask
    alarm_int = alarm_mask.astype(int)
    # Episode start: current is alarm, prior was not (or was not valid)
    prior_alarm = alarm_int.shift(1).fillna(0).astype(int)
    episode_starts = (alarm_int == 1) & (prior_alarm == 0)
    return int(episode_starts.sum()), int(alarm_mask.sum())


def evaluate_tier_pipeline():
    """
    Main evaluation routine:
    1. Trains Tier 1, Tier 2, Tier 3, and LOSO models.
    2. Runs continuous threshold sweep across test datasets.
    3. Evaluates event detection rate on catalog injections.
    4. Evaluates clean false alarms per station-year on clean background data.
    5. Produces summary CSVs and benchmarks against Phase 2 baselines.
    """
    print("=" * 80)
    print("PHASE 4: MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 80)

    # 1. Train Tier 1 Model
    X_tr_t1, y_tr = load_and_prepare_train_data(TIER_1_FEATURES)
    clf_t1 = train_tier_model(X_tr_t1, y_tr, "Tier 1 (Gross Only)")

    # 2. Train Tier 2 Model
    X_tr_t2, _ = load_and_prepare_train_data(TIER_2_FEATURES)
    clf_t2 = train_tier_model(X_tr_t2, y_tr, "Tier 2 (Spectral)")

    # 3. Train Tier 3 Model
    X_tr_t3, _ = load_and_prepare_train_data(TIER_3_FEATURES)
    clf_t3 = train_tier_model(X_tr_t3, y_tr, "Tier 3 (Weather Fusion)")

    # 4. Train LOSO Model (Holdout San Diego)
    X_tr_loso, y_tr_loso = load_and_prepare_train_data_loso(TIER_3_FEATURES, "ca_san_diego")
    clf_loso_sd = train_tier_model(X_tr_loso, y_tr_loso, "Tier 3 LOSO (Held-out San Diego)")

    # 5. Extract Feature Importance
    importance_df = pd.DataFrame({
        "feature": TIER_3_FEATURES,
        "importance_gain": clf_t3.booster_.feature_importance(importance_type="gain"),
        "importance_split": clf_t3.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    importance_df.to_csv("data/processed/model_feature_importance.csv", index=False)
    print("Saved feature importance to data/processed/model_feature_importance.csv")

    # 6. Load Test Data and Prepare Dual Series
    catalog = pd.read_csv("data/processed/synthetic_injection_catalog.csv")
    test_catalog = catalog[catalog["split"] == "test"].copy().reset_index(drop=True)
    print(f"Total test injections to evaluate: {len(test_catalog)}")

    test_series = {}
    total_clean_observed_hours = 0
    clean_obs_hours_per_st = {}

    for st in STATIONS:
        st_id = st["id"]
        csv_file = Path(f"data/processed/labeled_{st_id}_test.csv.gz")
        df_raw = pd.read_csv(csv_file)
        df_raw["dt"] = pd.to_datetime(df_raw["dt"])
        df_raw = df_raw.sort_values("dt").reset_index(drop=True)

        feats_inj = compute_features_for_series(df_raw, st_id, use_injected=True)
        feats_clean = compute_features_for_series(df_raw, st_id, use_injected=False)

        valid = feats_clean["has_radnet_obs"] & feats_clean["rad_complete_channels"]
        n_obs = valid.sum()
        total_clean_observed_hours += n_obs
        clean_obs_hours_per_st[st_id] = n_obs

        # Predictions on injected series (prob of fission class = 2)
        p_inj_t1 = clf_t1.predict_proba(feats_inj[TIER_1_FEATURES])[:, 2]
        p_inj_t2 = clf_t2.predict_proba(feats_inj[TIER_2_FEATURES])[:, 2]
        p_inj_t3 = clf_t3.predict_proba(feats_inj[TIER_3_FEATURES])[:, 2]
        p_inj_loso = clf_loso_sd.predict_proba(feats_inj[TIER_3_FEATURES])[:, 2]

        # Predictions on clean background series (prob of fission class = 2)
        p_clean_t1 = clf_t1.predict_proba(feats_clean[TIER_1_FEATURES])[:, 2]
        p_clean_t2 = clf_t2.predict_proba(feats_clean[TIER_2_FEATURES])[:, 2]
        p_clean_t3 = clf_t3.predict_proba(feats_clean[TIER_3_FEATURES])[:, 2]
        p_clean_loso = clf_loso_sd.predict_proba(feats_clean[TIER_3_FEATURES])[:, 2]

        # Baseline alarms on clean and injected series (Rolling 7d 3-sigma and Global 3-sigma)
        st_base = STATION_DRY_BASELINES.get(st_id, {"mu": 3000.0, "sigma": 400.0})
        base_3s_thresh = st_base["mu"] + 3.0 * st_base["sigma"]
        base_4s_thresh = st_base["mu"] + 4.0 * st_base["sigma"]
        base_5s_thresh = st_base["mu"] + 5.0 * st_base["sigma"]

        roll_168_clean = feats_clean["z_score_168h"]
        roll_168_inj = feats_inj["z_score_168h"]

        test_series[st_id] = {
            "dt": df_raw["dt"],
            "valid": valid,
            "feats_inj": feats_inj,
            "feats_clean": feats_clean,
            "p_inj": {"t1": p_inj_t1, "t2": p_inj_t2, "t3": p_inj_t3, "loso": p_inj_loso},
            "p_clean": {"t1": p_clean_t1, "t2": p_clean_t2, "t3": p_clean_t3, "loso": p_clean_loso},
            "gross_clean": df_raw["gross_cpm"],
            "gross_inj": df_raw["inj_gross_cpm"],
            "base_thresholds": {
                "global_3s": base_3s_thresh,
                "global_4s": base_4s_thresh,
                "global_5s": base_5s_thresh,
            },
            "z_roll_clean": roll_168_clean,
            "z_roll_inj": roll_168_inj,
        }

    total_station_years = total_clean_observed_hours / 8766.0
    print(f"Total network clean observed hours: {total_clean_observed_hours} ({total_station_years:.2f} station-years)")

    # 7. Sweep Thresholds to Map ROC Trade-Off Curve
    thresholds = np.linspace(0.01, 0.99, 99)
    roc_records = []

    for tau in thresholds:
        record = {"threshold": tau}
        for tier_key in ["t1", "t2", "t3", "loso"]:
            # A. Clean false alarms per station-year
            tot_episodes = 0
            for st_id in test_series:
                p_cl = pd.Series(test_series[st_id]["p_clean"][tier_key])
                val = test_series[st_id]["valid"]
                episodes, _ = count_clean_alarm_episodes(p_cl, val, tau)
                tot_episodes += episodes

            fa_per_year = tot_episodes / total_station_years
            record[f"{tier_key}_clean_fa_per_year"] = fa_per_year

            # B. Event detection rate on test catalog
            detected_events = 0
            for _, inj in test_catalog.iterrows():
                st_id = inj["station_id"]
                st_data = test_series[st_id]
                mask = (st_data["dt"] >= inj["start_utc"]) & (st_data["dt"] <= inj["end_utc"]) & st_data["valid"]
                p_event = st_data["p_inj"][tier_key][mask]
                if len(p_event) > 0 and np.max(p_event) >= tau:
                    detected_events += 1

            det_rate = detected_events / len(test_catalog)
            record[f"{tier_key}_event_det_rate"] = det_rate

        roc_records.append(record)

    roc_df = pd.DataFrame(roc_records)
    roc_df.to_csv("data/processed/model_roc_curve_data.csv", index=False)
    print("Saved ROC curve data to data/processed/model_roc_curve_data.csv")

    # 8. Benchmark Fixed-Threshold Baselines on Test Data
    # Global 3s, 4s, 5s and Rolling 3s, 4s, 5s
    baseline_benchmarks = {}
    for base_rule, k_val in [
        ("global_3s", 3.0), ("global_4s", 4.0), ("global_5s", 5.0),
        ("rolling_3s", 3.0), ("rolling_4s", 4.0), ("rolling_5s", 5.0)
    ]:
        tot_episodes = 0
        detected_events = 0

        # False alarms on clean data
        for st_id in test_series:
            val = test_series[st_id]["valid"]
            if base_rule.startswith("global"):
                thresh = test_series[st_id]["base_thresholds"][base_rule]
                alarm_mask = (test_series[st_id]["gross_clean"] >= thresh) & val
            else:
                alarm_mask = (test_series[st_id]["z_roll_clean"] >= k_val) & val

            alarm_int = alarm_mask.astype(int)
            prior = alarm_int.shift(1).fillna(0).astype(int)
            episodes = int(((alarm_int == 1) & (prior == 0)).sum())
            tot_episodes += episodes

        fa_per_year = tot_episodes / total_station_years

        # Detection rate on injected test catalog
        for _, inj in test_catalog.iterrows():
            st_id = inj["station_id"]
            st_data = test_series[st_id]
            mask = (st_data["dt"] >= inj["start_utc"]) & (st_data["dt"] <= inj["end_utc"]) & st_data["valid"]
            if base_rule.startswith("global"):
                thresh = st_data["base_thresholds"][base_rule]
                signal = st_data["gross_inj"][mask]
                if len(signal) > 0 and np.max(signal) >= thresh:
                    detected_events += 1
            else:
                signal = st_data["z_roll_inj"][mask]
                if len(signal) > 0 and np.max(signal) >= k_val:
                    detected_events += 1

        det_rate = detected_events / len(test_catalog)
        baseline_benchmarks[base_rule] = {
            "detection_rate_pct": round(det_rate * 100, 2),
            "false_alarms_per_year": round(fa_per_year, 2),
        }

    print("\n--- Baseline Benchmarks on Test Split (2023-2025) ---")
    for k, v in baseline_benchmarks.items():
        print(f"{k}: Det Rate = {v['detection_rate_pct']}%, FA/yr = {v['false_alarms_per_year']}")

    # 9. Extract Operating Points at Fixed Detection Rates (90%, 95%, 98%)
    target_det_rates = [0.90, 0.95, 0.98]
    op_records = []

    for target_det in target_det_rates:
        row = {"target_detection_rate_pct": target_det * 100}
        for tier_key, tier_label in [
            ("t1", "Tier 1 (Gross Only)"),
            ("t2", "Tier 2 (Spectral)"),
            ("t3", "Tier 3 (Weather Fusion)"),
            ("loso", "Tier 3 LOSO (Held-out SD)"),
        ]:
            # Find the threshold giving >= target_det with minimal false alarms
            sub = roc_df[roc_df[f"{tier_key}_event_det_rate"] >= target_det]
            if not sub.empty:
                # Highest threshold meeting the target detection rate has lowest FA rate
                best_match = sub.iloc[-1]
                tau = best_match["threshold"]
                realized_det = best_match[f"{tier_key}_event_det_rate"]
                fa_rate = best_match[f"{tier_key}_clean_fa_per_year"]
            else:
                best_match = roc_df.sort_values(f"{tier_key}_event_det_rate", ascending=False).iloc[0]
                tau = best_match["threshold"]
                realized_det = best_match[f"{tier_key}_event_det_rate"]
                fa_rate = best_match[f"{tier_key}_clean_fa_per_year"]

            row[f"{tier_key}_threshold"] = round(tau, 3)
            row[f"{tier_key}_realized_det_pct"] = round(realized_det * 100, 2)
            row[f"{tier_key}_fa_per_year"] = round(fa_rate, 2)

        op_records.append(row)

    op_df = pd.DataFrame(op_records)
    op_df.to_csv("data/processed/model_operating_points_summary.csv", index=False)
    print("\nSaved operating points to data/processed/model_operating_points_summary.csv")

    # 10. Stratified Evaluation Breakdown for Tier 3 at 95% Overall Operating Point
    # Find Tier 3 threshold for ~95% overall detection
    t3_95_row = roc_df[roc_df["t3_event_det_rate"] >= 0.95].iloc[-1]
    tau_t3 = t3_95_row["threshold"]
    print(f"\nOperating Tier 3 model at threshold tau = {tau_t3:.3f} (Overall Det Rate = {t3_95_row['t3_event_det_rate']*100:.2f}%)")

    strat_records = []
    # Stratify by environment, scenario, magnitude band, and station
    for group_col in ["environment", "nuclide_scenario", "magnitude_band", "station_name"]:
        for group_val, grp in test_catalog.groupby(group_col):
            tot = len(grp)
            det_t1 = 0
            det_t2 = 0
            det_t3 = 0
            delays_t3 = []

            for _, inj in grp.iterrows():
                st_id = inj["station_id"]
                st_data = test_series[st_id]
                mask = (st_data["dt"] >= inj["start_utc"]) & (st_data["dt"] <= inj["end_utc"]) & st_data["valid"]
                dt_event = st_data["dt"][mask].reset_index(drop=True)

                p1 = st_data["p_inj"]["t1"][mask]
                p2 = st_data["p_inj"]["t2"][mask]
                p3 = st_data["p_inj"]["t3"][mask]

                # Using Tier 1 & Tier 2 matching thresholds at ~95% overall
                t1_thresh = roc_df[roc_df["t1_event_det_rate"] >= 0.95].iloc[-1]["threshold"]
                t2_thresh = roc_df[roc_df["t2_event_det_rate"] >= 0.95].iloc[-1]["threshold"]

                if len(p1) > 0 and np.max(p1) >= t1_thresh:
                    det_t1 += 1
                if len(p2) > 0 and np.max(p2) >= t2_thresh:
                    det_t2 += 1
                if len(p3) > 0 and np.max(p3) >= tau_t3:
                    det_t3 += 1
                    # Compute delay (hours from start_utc to first alarm)
                    alarm_indices = np.where(p3 >= tau_t3)[0]
                    first_alarm_dt = dt_event.iloc[alarm_indices[0]]
                    delay_hours = (first_alarm_dt - pd.to_datetime(inj["start_utc"])).total_seconds() / 3600.0
                    delays_t3.append(max(0.0, delay_hours))

            med_delay = np.median(delays_t3) if delays_t3 else np.nan
            strat_records.append({
                "group_category": group_col,
                "group_value": group_val,
                "n_events": tot,
                "tier1_det_rate_pct": round(det_t1 / tot * 100, 2),
                "tier2_det_rate_pct": round(det_t2 / tot * 100, 2),
                "tier3_det_rate_pct": round(det_t3 / tot * 100, 2),
                "tier3_median_delay_hours": round(med_delay, 1) if not np.isnan(med_delay) else "N/A",
            })

    strat_df = pd.DataFrame(strat_records)
    strat_df.to_csv("data/processed/model_stratified_evaluation_summary.csv", index=False)
    print("Saved stratified summary to data/processed/model_stratified_evaluation_summary.csv")

    # 11. Leave-One-Station-Out (LOSO) Summary for San Diego
    sd_obs_years = clean_obs_hours_per_st["ca_san_diego"] / 8766.0
    sd_catalog = test_catalog[test_catalog["station_id"] == "ca_san_diego"]
    sd_detected = 0
    for _, inj in sd_catalog.iterrows():
        st_data = test_series["ca_san_diego"]
        mask = (st_data["dt"] >= inj["start_utc"]) & (st_data["dt"] <= inj["end_utc"]) & st_data["valid"]
        p_loso = st_data["p_inj"]["loso"][mask]
        if len(p_loso) > 0 and np.max(p_loso) >= tau_t3:
            sd_detected += 1

    sd_p_clean = pd.Series(test_series["ca_san_diego"]["p_clean"]["loso"])
    sd_val = test_series["ca_san_diego"]["valid"]
    sd_episodes, _ = count_clean_alarm_episodes(sd_p_clean, sd_val, tau_t3)
    sd_fa_rate = sd_episodes / sd_obs_years

    loso_df = pd.DataFrame([{
        "heldout_station": "San Diego, CA (ca_san_diego)",
        "climate": "West Coast Mediterranean / Coastal (Negative Rain Correlation)",
        "test_events": len(sd_catalog),
        "detected_events": sd_detected,
        "detection_rate_pct": round(sd_detected / len(sd_catalog) * 100, 2),
        "clean_observed_hours": clean_obs_hours_per_st["ca_san_diego"],
        "clean_false_alarm_episodes": sd_episodes,
        "false_alarms_per_station_year": round(sd_fa_rate, 2),
    }])
    loso_df.to_csv("data/processed/model_loso_evaluation_summary.csv", index=False)
    print("Saved LOSO summary to data/processed/model_loso_evaluation_summary.csv")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE - SUMMARY OF HEADLINE RESULTS")
    print("=" * 80)
    print(op_df.to_string(index=False))


if __name__ == "__main__":
    evaluate_tier_pipeline()
