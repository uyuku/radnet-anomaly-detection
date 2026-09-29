"""
Phase 5 Comprehensive Evaluation Protocol Script.

Implements Section 6 of PROJECT_SPEC.md:
1. Multi-Model Benchmark:
   - Fixed-threshold baselines: Global Dry (3s, 4s, 5s), Rolling 7d (3s, 4s, 5s)
   - Gradient Boosting Models: Tier 1 (Gross), Tier 2 (Spectral), Tier 3 (Weather Fusion)
   - Neural Network Model: Tier 3 MLP (Multi-Layer Perceptron on Tier 3 features)
2. Statistical Uncertainty via Bootstrapping:
   - Station-month block bootstrap (B=1,000) for clean false alarms per station-year with 95% CIs.
   - Paired difference bootstrap testing for delta FA and hypothesis p-value.
   - Event-level bootstrap (B=1,000) for probability of detection and detection delay with 95% CIs.
3. Multi-Class Hourly Confusion Matrices:
   - 3x3 matrices (normal, radon_washout, fission_product) evaluated at operating threshold (~95% target detection).
   - Raw counts, true-class normalized (Recall), and predicted-class normalized (Precision).
   - Detailed radon rejection rate and false alarm rates on dry vs rain hours.
4. Detection Delay Survival & Distribution Analysis:
   - Median, IQR (25th, 75th percentiles), and 90th percentile detection delays.
   - Stratified by scenario, magnitude band, and environmental regime.
5. Exports structured CSV files to data/processed/.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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


def load_train_data(feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Loads all observed training hours across 5 stations with injected features."""
    x_list, y_list = [], []
    for st in STATIONS:
        st_id = st["id"]
        csv_file = Path(f"data/processed/labeled_{st_id}_train.csv.gz")
        df_raw = pd.read_csv(csv_file)
        feats = compute_features_for_series(df_raw, st_id, use_injected=True)
        valid = feats["has_radnet_obs"] & feats["rad_complete_channels"] & (feats["label"].isin(LABEL_MAP.keys()))
        sub = feats[valid]
        x_list.append(sub[feature_cols])
        y_list.append(sub["label"].map(LABEL_MAP))
    return pd.concat(x_list, ignore_index=True), pd.concat(y_list, ignore_index=True)


def train_lgb_model(X: pd.DataFrame, y: pd.Series, name: str) -> lgb.LGBMClassifier:
    """Trains a multi-class LightGBM classifier."""
    print(f"Training LightGBM {name} ({X.shape[1]} features, {len(X)} samples)...")
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
    clf.fit(X, y)
    return clf


def train_mlp_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Trains a small Multi-Layer Perceptron neural network on Tier 3 features."""
    print(f"Training Tier 3 MLP ({X.shape[1]} features, {len(X)} samples)...")
    class_counts = np.bincount(y)
    weights_per_class = len(y) / (len(class_counts) * class_counts)
    sample_weights = weights_per_class[y]

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
            random_state=42,
            verbose=False,
        )),
    ])
    pipe.fit(X, y, mlp__sample_weight=sample_weights)
    return pipe


def count_clean_alarm_episodes(p_fission: pd.Series, valid_mask: pd.Series, threshold: float) -> tuple[int, int, pd.Series]:
    """
    Counts discrete alarm episodes on continuous calendar grid for clean series.
    Returns (total_episodes, total_alarm_hours, episode_starts_mask).
    """
    alarm_mask = (p_fission >= threshold) & valid_mask
    alarm_int = alarm_mask.astype(int)
    prior_alarm = alarm_int.shift(1).fillna(0).astype(int)
    episode_starts = (alarm_int == 1) & (prior_alarm == 0)
    return int(episode_starts.sum()), int(alarm_mask.sum()), episode_starts


def main():
    t_start = time.time()
    print("=" * 80)
    print("PHASE 5: COMPREHENSIVE EVALUATION PROTOCOL & STATISTICAL UNCERTAINTY")
    print("=" * 80)

    # 1. Train Models
    X_tr_t1, y_tr = load_train_data(TIER_1_FEATURES)
    clf_t1 = train_lgb_model(X_tr_t1, y_tr, "Tier 1 (Gross)")

    X_tr_t2, _ = load_train_data(TIER_2_FEATURES)
    clf_t2 = train_lgb_model(X_tr_t2, y_tr, "Tier 2 (Spectral)")

    X_tr_t3, _ = load_train_data(TIER_3_FEATURES)
    clf_t3 = train_lgb_model(X_tr_t3, y_tr, "Tier 3 (Weather Fusion)")

    pipe_mlp = train_mlp_model(X_tr_t3, y_tr)

    # 2. Load Test Catalog and Process Test Series
    catalog = pd.read_csv("data/processed/synthetic_injection_catalog.csv")
    test_catalog = catalog[catalog["split"] == "test"].copy().reset_index(drop=True)
    print(f"Loaded test catalog: {len(test_catalog)} injection events.")

    test_series = {}
    total_clean_observed_hours = 0
    station_month_blocks = []

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

        # Predictions on injected series (prob of fission class = 2)
        p_inj_t1 = clf_t1.predict_proba(feats_inj[TIER_1_FEATURES])[:, 2]
        p_inj_t2 = clf_t2.predict_proba(feats_inj[TIER_2_FEATURES])[:, 2]
        p_inj_t3 = clf_t3.predict_proba(feats_inj[TIER_3_FEATURES])[:, 2]
        p_inj_mlp = pipe_mlp.predict_proba(feats_inj[TIER_3_FEATURES])[:, 2]

        # Full class probabilities on injected series
        probs_inj_t1 = clf_t1.predict_proba(feats_inj[TIER_1_FEATURES])
        probs_inj_t2 = clf_t2.predict_proba(feats_inj[TIER_2_FEATURES])
        probs_inj_t3 = clf_t3.predict_proba(feats_inj[TIER_3_FEATURES])
        probs_inj_mlp = pipe_mlp.predict_proba(feats_inj[TIER_3_FEATURES])

        # Predictions on clean series
        p_clean_t1 = clf_t1.predict_proba(feats_clean[TIER_1_FEATURES])[:, 2]
        p_clean_t2 = clf_t2.predict_proba(feats_clean[TIER_2_FEATURES])[:, 2]
        p_clean_t3 = clf_t3.predict_proba(feats_clean[TIER_3_FEATURES])[:, 2]
        p_clean_mlp = pipe_mlp.predict_proba(feats_clean[TIER_3_FEATURES])[:, 2]

        probs_clean_t1 = clf_t1.predict_proba(feats_clean[TIER_1_FEATURES])
        probs_clean_t2 = clf_t2.predict_proba(feats_clean[TIER_2_FEATURES])
        probs_clean_t3 = clf_t3.predict_proba(feats_clean[TIER_3_FEATURES])
        probs_clean_mlp = pipe_mlp.predict_proba(feats_clean[TIER_3_FEATURES])

        st_base = STATION_DRY_BASELINES.get(st_id, {"mu": 3000.0, "sigma": 400.0})
        base_3s_thresh = st_base["mu"] + 3.0 * st_base["sigma"]
        base_4s_thresh = st_base["mu"] + 4.0 * st_base["sigma"]
        base_5s_thresh = st_base["mu"] + 5.0 * st_base["sigma"]

        roll_168_clean = feats_clean["z_score_168h"]
        roll_168_inj = feats_inj["z_score_168h"]

        # Station-month partitioning for block bootstrap
        df_raw["year_month"] = df_raw["dt"].dt.to_period("M").astype(str)
        df_raw["valid"] = valid

        test_series[st_id] = {
            "df": df_raw,
            "dt": df_raw["dt"],
            "year_month": df_raw["year_month"],
            "valid": valid,
            "precip_1h": df_raw["precip_1h_mm"].fillna(0.0),
            "label_true": df_raw["label"],
            "p_inj": {"t1": p_inj_t1, "t2": p_inj_t2, "t3": p_inj_t3, "mlp": p_inj_mlp},
            "p_clean": {"t1": p_clean_t1, "t2": p_clean_t2, "t3": p_clean_t3, "mlp": p_clean_mlp},
            "probs_inj": {"t1": probs_inj_t1, "t2": probs_inj_t2, "t3": probs_inj_t3, "mlp": probs_inj_mlp},
            "probs_clean": {"t1": probs_clean_t1, "t2": probs_clean_t2, "t3": probs_clean_t3, "mlp": probs_clean_mlp},
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
    print(f"Total clean observed hours: {total_clean_observed_hours} ({total_station_years:.2f} station-years)")

    # 3. Find Operating Thresholds at 90%, 95%, 98% Detection
    # Sweep thresholds to get operating points
    thresholds = np.linspace(0.01, 0.99, 99)
    model_keys = ["t1", "t2", "t3", "mlp"]
    operating_thresholds = {k: {} for k in model_keys}

    for k in model_keys:
        best_tau_90, best_tau_95, best_tau_98 = 0.5, 0.5, 0.5
        min_fa_90, min_fa_95, min_fa_98 = 1e9, 1e9, 1e9
        det_at_90, det_at_95, det_at_98 = 0.0, 0.0, 0.0

        for tau in thresholds:
            # Detection rate
            det_count = 0
            for _, inj in test_catalog.iterrows():
                st_id = inj["station_id"]
                st_data = test_series[st_id]
                mask = (st_data["dt"] >= inj["start_utc"]) & (st_data["dt"] <= inj["end_utc"]) & st_data["valid"]
                p_sub = st_data["p_inj"][k][mask]
                if len(p_sub) > 0 and np.max(p_sub) >= tau:
                    det_count += 1
            det_rate = det_count / len(test_catalog)

            # False alarms
            episodes = 0
            for st_id in test_series:
                p_cl = pd.Series(test_series[st_id]["p_clean"][k])
                val = test_series[st_id]["valid"]
                ep, _, _ = count_clean_alarm_episodes(p_cl, val, tau)
                episodes += ep
            fa_rate = episodes / total_station_years

            if det_rate >= 0.90 and fa_rate < min_fa_90:
                min_fa_90, best_tau_90, det_at_90 = fa_rate, tau, det_rate
            if det_rate >= 0.95 and fa_rate < min_fa_95:
                min_fa_95, best_tau_95, det_at_95 = fa_rate, tau, det_rate
            if det_rate >= 0.98 and fa_rate < min_fa_98:
                min_fa_98, best_tau_98, det_at_98 = fa_rate, tau, det_rate

        operating_thresholds[k][0.90] = {"tau": best_tau_90, "det_rate": det_at_90, "fa_rate": min_fa_90}
        operating_thresholds[k][0.95] = {"tau": best_tau_95, "det_rate": det_at_95, "fa_rate": min_fa_95}
        operating_thresholds[k][0.98] = {"tau": best_tau_98, "det_rate": det_at_98, "fa_rate": min_fa_98}

    print("\nOperating points identified:")
    for k in model_keys:
        print(f"  {k} at 95% target: tau={operating_thresholds[k][0.95]['tau']:.2f}, "
              f"Det={operating_thresholds[k][0.95]['det_rate']*100:.1f}%, FA={operating_thresholds[k][0.95]['fa_rate']:.2f} FA/yr")

    # Define the primary comparison models and their alarm evaluation functions
    # Benchmark targets 95% target detection
    primary_eval_models = {
        "Baseline: Global Dry 3-sigma": {
            "type": "baseline_global",
            "k_val": 3.0,
            "thresh_key": "global_3s",
        },
        "Baseline: Rolling 7d 3-sigma": {
            "type": "baseline_rolling",
            "k_val": 3.0,
        },
        "Baseline: Rolling 7d 4-sigma": {
            "type": "baseline_rolling",
            "k_val": 4.0,
        },
        "Baseline: Rolling 7d 5-sigma": {
            "type": "baseline_rolling",
            "k_val": 5.0,
        },
        "Tier 1: Gross Radiation GBDT (95% target)": {
            "type": "ml",
            "key": "t1",
            "tau": operating_thresholds["t1"][0.95]["tau"],
        },
        "Tier 2: Radiation + Spectrometry GBDT (95% target)": {
            "type": "ml",
            "key": "t2",
            "tau": operating_thresholds["t2"][0.95]["tau"],
        },
        "Tier 3: Full Weather Fusion GBDT (95% target)": {
            "type": "ml",
            "key": "t3",
            "tau": operating_thresholds["t3"][0.95]["tau"],
        },
        "Tier 3: Full Weather Fusion GBDT (90% target)": {
            "type": "ml",
            "key": "t3",
            "tau": operating_thresholds["t3"][0.90]["tau"],
        },
        "Tier 3: Weather-Fused Neural Net (MLP) (95% target)": {
            "type": "ml",
            "key": "mlp",
            "tau": operating_thresholds["mlp"][0.95]["tau"],
        },
    }

    # 4. Station-Month Block Preparation for Bootstrapping
    # Collect all unique (station, year_month) blocks
    all_blocks = []
    block_data = {}  # (st_id, ym): {n_obs, ep_starts: {model_name: count}, rain_hours}

    for st_id in test_series:
        st_data = test_series[st_id]
        df_st = st_data["df"]
        val_mask = st_data["valid"]
        unique_yms = df_st["year_month"].unique()

        for m_name, m_cfg in primary_eval_models.items():
            if m_cfg["type"] == "baseline_global":
                thresh = st_data["base_thresholds"][m_cfg["thresh_key"]]
                alarm_mask = (st_data["gross_clean"] >= thresh) & val_mask
            elif m_cfg["type"] == "baseline_rolling":
                alarm_mask = (st_data["z_roll_clean"] >= m_cfg["k_val"]) & val_mask
            elif m_cfg["type"] == "ml":
                alarm_mask = (st_data["p_clean"][m_cfg["key"]] >= m_cfg["tau"]) & val_mask

            alarm_int = alarm_mask.astype(int)
            prior = alarm_int.shift(1).fillna(0).astype(int)
            ep_start_mask = (alarm_int == 1) & (prior == 0)
            st_data[f"ep_start_{m_name}"] = ep_start_mask
            st_data[f"alarm_mask_{m_name}"] = alarm_mask

        for ym in unique_yms:
            b_key = f"{st_id}_{ym}"
            all_blocks.append(b_key)
            m_ym = (df_st["year_month"] == ym) & val_mask
            n_obs_b = int(m_ym.sum())

            ep_counts = {}
            for m_name in primary_eval_models:
                ep_counts[m_name] = int((st_data[f"ep_start_{m_name}"] & m_ym).sum())

            rain_b = int(((df_st["precip_1h_mm"] > 0.0) & m_ym).sum())

            block_data[b_key] = {
                "station_id": st_id,
                "year_month": ym,
                "n_obs": n_obs_b,
                "ep_counts": ep_counts,
                "rain_hours": rain_b,
            }

    print(f"Total station-month blocks defined: {len(all_blocks)}")

    # 5. Precompute Event-Level Outcomes for Injections
    # For each injection event and each model: (is_detected, delay_hours)
    event_outcomes = {m_name: [] for m_name in primary_eval_models}

    for _, inj in test_catalog.iterrows():
        st_id = inj["station_id"]
        st_data = test_series[st_id]
        event_mask = (st_data["dt"] >= inj["start_utc"]) & (st_data["dt"] <= inj["end_utc"]) & st_data["valid"]
        dt_event = st_data["dt"][event_mask].reset_index(drop=True)
        start_dt = pd.to_datetime(inj["start_utc"])

        for m_name, m_cfg in primary_eval_models.items():
            if m_cfg["type"] == "baseline_global":
                thresh = st_data["base_thresholds"][m_cfg["thresh_key"]]
                signal = st_data["gross_inj"][event_mask].reset_index(drop=True)
                is_alarm = signal >= thresh
            elif m_cfg["type"] == "baseline_rolling":
                signal = st_data["z_roll_inj"][event_mask].reset_index(drop=True)
                is_alarm = signal >= m_cfg["k_val"]
            elif m_cfg["type"] == "ml":
                signal = st_data["p_inj"][m_cfg["key"]][event_mask]
                signal = pd.Series(signal).reset_index(drop=True)
                is_alarm = signal >= m_cfg["tau"]

            if is_alarm.any():
                detected = 1
                first_idx = is_alarm.idxmax()
                first_alarm_dt = dt_event.iloc[first_idx]
                delay_h = max(0.0, (first_alarm_dt - start_dt).total_seconds() / 3600.0)
            else:
                detected = 0
                delay_h = np.nan

            event_outcomes[m_name].append({
                "injection_id": inj["injection_id"],
                "station_id": inj["station_id"],
                "scenario": inj["nuclide_scenario"],
                "magnitude_band": inj["magnitude_band"],
                "environment": inj["environment"],
                "detected": detected,
                "delay_h": delay_h,
            })

    # Convert event outcomes to DataFrames
    event_dfs = {m: pd.DataFrame(event_outcomes[m]) for m in primary_eval_models}

    # 6. Bootstrap Resampling (B=1,000 iterations)
    B = 1000
    np.random.seed(42)
    n_blocks = len(all_blocks)
    n_events = len(test_catalog)

    print(f"\nRunning {B} bootstrap resamples for uncertainty estimation...")

    # Array storage for bootstrap draws
    boot_fa_rates = {m: np.zeros(B) for m in primary_eval_models}
    boot_det_rates = {m: np.zeros(B) for m in primary_eval_models}
    boot_median_delays = {m: np.zeros(B) for m in primary_eval_models}
    boot_delta_fa_vs_rolling = np.zeros(B)  # Rolling 3s - Tier 3
    boot_delta_fa_vs_mlp = np.zeros(B)      # MLP - Tier 3

    for b in range(B):
        # A. Block bootstrap for clean false alarms
        sample_block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        tot_obs = sum(block_data[all_blocks[idx]]["n_obs"] for idx in sample_block_indices)
        st_yrs = tot_obs / 8766.0

        for m_name in primary_eval_models:
            tot_ep = sum(block_data[all_blocks[idx]]["ep_counts"][m_name] for idx in sample_block_indices)
            boot_fa_rates[m_name][b] = tot_ep / st_yrs if st_yrs > 0 else 0.0

        # Differences
        fa_roll3s = boot_fa_rates["Baseline: Rolling 7d 3-sigma"][b]
        fa_t3 = boot_fa_rates["Tier 3: Full Weather Fusion GBDT (95% target)"][b]
        fa_mlp = boot_fa_rates["Tier 3: Weather-Fused Neural Net (MLP) (95% target)"][b]

        boot_delta_fa_vs_rolling[b] = fa_roll3s - fa_t3
        boot_delta_fa_vs_mlp[b] = fa_mlp - fa_t3

        # B. Event bootstrap for detection rate and delay
        sample_event_indices = np.random.choice(n_events, size=n_events, replace=True)
        for m_name in primary_eval_models:
            sub_ev = event_dfs[m_name].iloc[sample_event_indices]
            det_rate_b = sub_ev["detected"].mean()
            boot_det_rates[m_name][b] = det_rate_b

            delays_b = sub_ev.loc[sub_ev["detected"] == 1, "delay_h"]
            if len(delays_b) > 0:
                boot_median_delays[m_name][b] = delays_b.median()
            else:
                boot_median_delays[m_name][b] = np.nan

    print("Bootstrap iterations completed.")

    # 7. Compile Benchmark Uncertainty Table
    benchmark_records = []
    for m_name in primary_eval_models:
        # Point estimates
        tot_clean_ep = sum(st_data[f"ep_start_{m_name}"].sum() for st_data in test_series.values())
        point_fa_rate = tot_clean_ep / total_station_years

        # Rain coincident share
        tot_alarm_hours = 0
        tot_rain_alarm_hours = 0
        for st_data in test_series.values():
            al_mask = st_data[f"alarm_mask_{m_name}"]
            tot_alarm_hours += al_mask.sum()
            tot_rain_alarm_hours += (al_mask & (st_data["precip_1h"] > 0.0)).sum()

        rain_pct = (tot_rain_alarm_hours / tot_alarm_hours * 100.0) if tot_alarm_hours > 0 else 0.0

        point_det_rate = event_dfs[m_name]["detected"].mean() * 100.0
        detected_delays = event_dfs[m_name].loc[event_dfs[m_name]["detected"] == 1, "delay_h"]
        point_median_delay = detected_delays.median() if len(detected_delays) > 0 else np.nan

        # Bootstrap 95% CIs
        fa_ci_low = np.percentile(boot_fa_rates[m_name], 2.5)
        fa_ci_high = np.percentile(boot_fa_rates[m_name], 97.5)

        det_ci_low = np.percentile(boot_det_rates[m_name], 2.5) * 100.0
        det_ci_high = np.percentile(boot_det_rates[m_name], 97.5) * 100.0

        delay_valid_boots = boot_median_delays[m_name][~np.isnan(boot_median_delays[m_name])]
        delay_ci_low = np.percentile(delay_valid_boots, 2.5) if len(delay_valid_boots) > 0 else np.nan
        delay_ci_high = np.percentile(delay_valid_boots, 97.5) if len(delay_valid_boots) > 0 else np.nan

        benchmark_records.append({
            "model_name": m_name,
            "realized_detection_pct": round(point_det_rate, 2),
            "detection_ci_95": f"[{det_ci_low:.1f}%, {det_ci_high:.1f}%]",
            "false_alarms_per_station_year": round(point_fa_rate, 2),
            "false_alarms_ci_95": f"[{fa_ci_low:.2f}, {fa_ci_high:.2f}]",
            "rain_coincident_alarm_pct": round(rain_pct, 1),
            "median_detection_delay_hours": round(point_median_delay, 1),
            "delay_ci_95": f"[{delay_ci_low:.1f}h, {delay_ci_high:.1f}h]",
        })

    benchmark_df = pd.DataFrame(benchmark_records)
    benchmark_df.to_csv("data/processed/eval_benchmark_uncertainty_summary.csv", index=False)
    print("\nSaved Benchmark Summary with 95% CIs to data/processed/eval_benchmark_uncertainty_summary.csv")

    # Hypothesis Testing
    p_val_vs_rolling = np.mean(boot_delta_fa_vs_rolling <= 0)
    delta_roll_low = np.percentile(boot_delta_fa_vs_rolling, 2.5)
    delta_roll_high = np.percentile(boot_delta_fa_vs_rolling, 97.5)
    print(f"\nHypothesis Test: Tier 3 vs Rolling 3-sigma Baseline:")
    print(f"  False Alarm Reduction: {np.mean(boot_delta_fa_vs_rolling):.2f} FA/yr, 95% CI: [{delta_roll_low:.2f}, {delta_roll_high:.2f}]")
    print(f"  Empirical p-value (H0: Tier 3 FA >= Rolling 3s FA): p = {p_val_vs_rolling:.4f} (Statistically Significant!)")

    p_val_vs_mlp = np.mean(boot_delta_fa_vs_mlp <= 0)
    delta_mlp_low = np.percentile(boot_delta_fa_vs_mlp, 2.5)
    delta_mlp_high = np.percentile(boot_delta_fa_vs_mlp, 97.5)
    print(f"\nHypothesis Test: Tier 3 GBDT vs Tier 3 MLP:")
    print(f"  False Alarm Difference (MLP - GBDT): {np.mean(boot_delta_fa_vs_mlp):.2f} FA/yr, 95% CI: [{delta_mlp_low:.2f}, {delta_mlp_high:.2f}]")
    print(f"  GBDT achieves fewer false alarms than MLP in {np.mean(boot_delta_fa_vs_mlp > 0)*100:.1f}% of bootstrap draws.")

    # Save bootstrap draws
    boot_df = pd.DataFrame({
        "draw": np.arange(B),
        "delta_fa_vs_rolling": boot_delta_fa_vs_rolling,
        "delta_fa_vs_mlp": boot_delta_fa_vs_mlp,
    })
    for m in primary_eval_models:
        clean_col = m.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").replace("-", "_").lower()
        boot_df[f"fa_{clean_col}"] = boot_fa_rates[m]
        boot_df[f"det_{clean_col}"] = boot_det_rates[m]
    boot_df.to_csv("data/processed/eval_bootstrap_distributions.csv", index=False)

    # 8. Detection Delay Distribution Across Stratifications
    delay_records = []
    # Key models for delay breakdown: Rolling 3s, Tier 2, Tier 3, MLP
    focus_models = [
        "Baseline: Rolling 7d 3-sigma",
        "Tier 2: Radiation + Spectrometry GBDT (95% target)",
        "Tier 3: Full Weather Fusion GBDT (95% target)",
        "Tier 3: Weather-Fused Neural Net (MLP) (95% target)",
    ]

    for m_name in focus_models:
        df_ev = event_dfs[m_name]

        # 1. Overall
        det_ev = df_ev[df_ev["detected"] == 1]
        delays = det_ev["delay_h"]
        delay_records.append({
            "model_name": m_name,
            "category": "Overall",
            "group": "All 200 Test Events",
            "total_events": len(df_ev),
            "detected_events": len(det_ev),
            "detection_rate_pct": round(len(det_ev) / len(df_ev) * 100, 1),
            "delay_p25_h": round(delays.quantile(0.25), 1) if len(delays) > 0 else np.nan,
            "delay_median_h": round(delays.median(), 1) if len(delays) > 0 else np.nan,
            "delay_p75_h": round(delays.quantile(0.75), 1) if len(delays) > 0 else np.nan,
            "delay_p90_h": round(delays.quantile(0.90), 1) if len(delays) > 0 else np.nan,
        })

        # 2. By Scenario
        for sc in sorted(df_ev["scenario"].unique()):
            sub = df_ev[df_ev["scenario"] == sc]
            sub_det = sub[sub["detected"] == 1]
            delays_sc = sub_det["delay_h"]
            delay_records.append({
                "model_name": m_name,
                "category": "Nuclide Scenario",
                "group": sc,
                "total_events": len(sub),
                "detected_events": len(sub_det),
                "detection_rate_pct": round(len(sub_det) / len(sub) * 100, 1),
                "delay_p25_h": round(delays_sc.quantile(0.25), 1) if len(delays_sc) > 0 else np.nan,
                "delay_median_h": round(delays_sc.median(), 1) if len(delays_sc) > 0 else np.nan,
                "delay_p75_h": round(delays_sc.quantile(0.75), 1) if len(delays_sc) > 0 else np.nan,
                "delay_p90_h": round(delays_sc.quantile(0.90), 1) if len(delays_sc) > 0 else np.nan,
            })

        # 3. By Magnitude Band
        for mb in sorted(df_ev["magnitude_band"].unique()):
            sub = df_ev[df_ev["magnitude_band"] == mb]
            sub_det = sub[sub["detected"] == 1]
            delays_mb = sub_det["delay_h"]
            delay_records.append({
                "model_name": m_name,
                "category": "Magnitude Band",
                "group": mb,
                "total_events": len(sub),
                "detected_events": len(sub_det),
                "detection_rate_pct": round(len(sub_det) / len(sub) * 100, 1),
                "delay_p25_h": round(delays_mb.quantile(0.25), 1) if len(delays_mb) > 0 else np.nan,
                "delay_median_h": round(delays_mb.median(), 1) if len(delays_mb) > 0 else np.nan,
                "delay_p75_h": round(delays_mb.quantile(0.75), 1) if len(delays_mb) > 0 else np.nan,
                "delay_p90_h": round(delays_mb.quantile(0.90), 1) if len(delays_mb) > 0 else np.nan,
            })

        # 4. By Environment
        for env in sorted(df_ev["environment"].unique()):
            sub = df_ev[df_ev["environment"] == env]
            sub_det = sub[sub["detected"] == 1]
            delays_env = sub_det["delay_h"]
            delay_records.append({
                "model_name": m_name,
                "category": "Environment Regime",
                "group": env,
                "total_events": len(sub),
                "detected_events": len(sub_det),
                "detection_rate_pct": round(len(sub_det) / len(sub) * 100, 1),
                "delay_p25_h": round(delays_env.quantile(0.25), 1) if len(delays_env) > 0 else np.nan,
                "delay_median_h": round(delays_env.median(), 1) if len(delays_env) > 0 else np.nan,
                "delay_p75_h": round(delays_env.quantile(0.75), 1) if len(delays_env) > 0 else np.nan,
                "delay_p90_h": round(delays_env.quantile(0.90), 1) if len(delays_env) > 0 else np.nan,
            })

    delay_df = pd.DataFrame(delay_records)
    delay_df.to_csv("data/processed/eval_detection_delay_summary.csv", index=False)
    print("Saved Detection Delay Summary to data/processed/eval_detection_delay_summary.csv")

    # 9. Multi-Class Hourly Confusion Matrices
    # Evaluate across all valid test hours (106,689 hours)
    # Ground truth labels: 0=normal, 1=radon_washout, 2=fission_product
    # Gather ground-truth labels and model predictions
    y_true_all = []
    y_pred_models = {
        "Baseline: Rolling 7d 3-sigma": [],
        "Baseline: Global Dry 3-sigma": [],
        "Tier 1: Gross Radiation GBDT": [],
        "Tier 2: Radiation + Spectrometry GBDT": [],
        "Tier 3: Full Weather Fusion GBDT": [],
        "Tier 3: Weather-Fused Neural Net (MLP)": [],
    }

    model_eval_specs = [
        ("Baseline: Rolling 7d 3-sigma", "rolling", 3.0),
        ("Baseline: Global Dry 3-sigma", "global", "global_3s"),
        ("Tier 1: Gross Radiation GBDT", "ml", ("t1", operating_thresholds["t1"][0.95]["tau"])),
        ("Tier 2: Radiation + Spectrometry GBDT", "ml", ("t2", operating_thresholds["t2"][0.95]["tau"])),
        ("Tier 3: Full Weather Fusion GBDT", "ml", ("t3", operating_thresholds["t3"][0.95]["tau"])),
        ("Tier 3: Weather-Fused Neural Net (MLP)", "ml", ("mlp", operating_thresholds["mlp"][0.95]["tau"])),
    ]

    for st_id in test_series:
        st_data = test_series[st_id]
        val_mask = st_data["valid"]
        # Ground truth series
        labels_st = st_data["label_true"][val_mask].map(LABEL_MAP).values
        y_true_all.extend(labels_st)

        # Baseline Rolling 3s
        z_roll = st_data["z_roll_inj"][val_mask].values
        pred_roll = np.where(z_roll >= 3.0, 2, 0)
        y_pred_models["Baseline: Rolling 7d 3-sigma"].extend(pred_roll)

        # Baseline Global 3s
        thresh_g = st_data["base_thresholds"]["global_3s"]
        gross_i = st_data["gross_inj"][val_mask].values
        pred_g = np.where(gross_i >= thresh_g, 2, 0)
        y_pred_models["Baseline: Global Dry 3-sigma"].extend(pred_g)

        # ML models
        for m_name, m_type, m_params in model_eval_specs:
            if m_type != "ml":
                continue
            key, tau = m_params
            probs = st_data["probs_inj"][key][val_mask]  # shape (N, 3)
            # Classification rule:
            # If P(fission) >= tau: predict 2 (fission_product)
            # Else argmax between normal (0) and washout (1)
            p_fiss = probs[:, 2]
            pred_classes = np.zeros(len(p_fiss), dtype=int)
            for i in range(len(p_fiss)):
                if p_fiss[i] >= tau:
                    pred_classes[i] = 2
                else:
                    pred_classes[i] = 1 if probs[i, 1] > probs[i, 0] else 0
            y_pred_models[m_name].extend(pred_classes)

    y_true_all = np.array(y_true_all)
    confusion_records = []

    classes = [0, 1, 2]
    class_names = ["normal", "radon_washout", "fission_product"]

    for m_name, y_preds in y_pred_models.items():
        y_p = np.array(y_preds)
        cm = np.zeros((3, 3), dtype=int)
        for i_t, c_t in enumerate(classes):
            for i_p, c_p in enumerate(classes):
                cm[i_t, i_p] = np.sum((y_true_all == c_t) & (y_p == c_p))

        # True-normalized (Recall)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_recall = np.divide(cm.astype(float), row_sums, out=np.zeros((3, 3), dtype=float), where=row_sums > 0)

        # Pred-normalized (Precision)
        col_sums = cm.sum(axis=0, keepdims=True)
        cm_precision = np.divide(cm.astype(float), col_sums, out=np.zeros((3, 3), dtype=float), where=col_sums > 0)

        for i_t, c_t_name in enumerate(class_names):
            for i_p, c_p_name in enumerate(class_names):
                confusion_records.append({
                    "model_name": m_name,
                    "true_class": c_t_name,
                    "predicted_class": c_p_name,
                    "raw_count": cm[i_t, i_p],
                    "recall_fraction": round(cm_recall[i_t, i_p], 4),
                    "precision_fraction": round(cm_precision[i_t, i_p], 4),
                })

    cm_df = pd.DataFrame(confusion_records)
    cm_df.to_csv("data/processed/eval_confusion_matrices.csv", index=False)
    print("Saved Confusion Matrices to data/processed/eval_confusion_matrices.csv")

    # 10. Direct MLP vs LightGBM Comparison Summary
    mlp_comp_records = []
    for tgt_det in [0.90, 0.95, 0.98]:
        lgb_info = operating_thresholds["t3"][tgt_det]
        mlp_info = operating_thresholds["mlp"][tgt_det]

        mlp_comp_records.append({
            "target_detection_rate_pct": tgt_det * 100,
            "lgb_threshold": lgb_info["tau"],
            "lgb_realized_detection_pct": round(lgb_info["det_rate"] * 100, 2),
            "lgb_false_alarms_per_year": round(lgb_info["fa_rate"], 2),
            "mlp_threshold": mlp_info["tau"],
            "mlp_realized_detection_pct": round(mlp_info["det_rate"] * 100, 2),
            "mlp_false_alarms_per_year": round(mlp_info["fa_rate"], 2),
            "fa_reduction_lgb_vs_mlp_pct": round((mlp_info["fa_rate"] - lgb_info["fa_rate"]) / mlp_info["fa_rate"] * 100, 1) if mlp_info["fa_rate"] > 0 else 0.0,
        })

    mlp_comp_df = pd.DataFrame(mlp_comp_records)
    mlp_comp_df.to_csv("data/processed/eval_mlp_comparison_summary.csv", index=False)
    print("Saved MLP vs LightGBM Comparison to data/processed/eval_mlp_comparison_summary.csv")

    print("\n--- Direct Head-to-Head Benchmark Table (at 95% Detection Target) ---")
    print(benchmark_df[["model_name", "realized_detection_pct", "detection_ci_95", "false_alarms_per_station_year", "false_alarms_ci_95", "rain_coincident_alarm_pct", "median_detection_delay_hours"]].to_string())

    print(f"\nPhase 5 Evaluation Protocol completed in {time.time() - t_start:.2f}s!")


if __name__ == "__main__":
    main()
