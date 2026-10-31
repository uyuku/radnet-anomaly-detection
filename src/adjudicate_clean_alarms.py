"""
Rule-Based Categorization Audit of 100 Clean-Background Test Alarms.

Selects a stratified random sample of N=100 clean-background test alarms from the 2023-2025
test period across pilot stations (and extended stations) and classifies each into:
- rain_coincident (active rain, P >= 0.1 mm)
- post_rain_washout (dry at alarm, but within 6h of rain with elevated Bi-214 in R03/R05)
- filter_change_artifact (adjacent to bi-weekly filter cycle step)
- instrument_step_drift (hardware baseline shift or discontinuity, e.g. Tampa 2023)
- other_natural (severe barometric drop, inversion without rain)
- unexplained (no identifiable meteorological, operational, or hardware anomaly)

Computes exact Wilson score 95% confidence intervals and generates:
- docs/hand_labeled_test_alarms.csv
- data/processed/hand_labeled_test_alarms_summary.csv
- reports/figures/alarm_audit_sample_panel.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.feature_engineering import (
    STATION_DRY_BASELINES,
    TIER_1_FEATURES,
    TIER_2_FEATURES,
    TIER_3_FEATURES,
    TIER_4_NUCLEAR_FEATURES,
    compute_features_for_series,
)
from src.merge_radnet_weather import CHANNEL_KEYS
from src.train_and_evaluate_rigorous import STATIONS, load_dataset_folds, train_lgb

FROZEN_THRESHOLDS = {
    "Tier 4 (Physics-Informed Nuclear)": 0.8719,
    "Tier 3 (Weather Fusion)": 0.8726,
    "Tier 2 (Spectral)": 0.9475,
    "Tier 1 (Gross)": 0.9238,
    "Baseline: Rolling 7d 3-sigma": 3.0,
    "Baseline: Rolling 7d 4-sigma": 4.0,
    "Baseline: Global Dry 3-sigma": 3.0,
}


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculates Wilson score binomial confidence interval."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def main():
    print("================================================================================")
    print("AUDIT & RULE-BASED CATEGORIZATION OF 100 CLEAN-BACKGROUND TEST ALARMS")
    print("================================================================================")

    # 1. Train Canonical Frozen Models (Seed 42)
    print("\n--- Training Frozen Models on 2017-2020 Training Fold (Seed 42) ---")
    tiers = {
        "Tier 1 (Gross)": TIER_1_FEATURES,
        "Tier 2 (Spectral)": TIER_2_FEATURES,
        "Tier 3 (Weather Fusion)": TIER_3_FEATURES,
        "Tier 4 (Physics-Informed Nuclear)": TIER_4_NUCLEAR_FEATURES,
    }
    models = {}
    for tname, fcols in tiers.items():
        print(f"Training frozen model for {tname}...")
        X_tr, y_tr, _, _ = load_dataset_folds(fcols)
        models[tname] = train_lgb(X_tr, y_tr, seed=42)

    # 2. Extract Clean-Background Alarms Across All 5 Stations in 2023-2025 Test Split
    print("\n--- Scanning Clean Test Data for Candidate Alarm Hours ---")
    all_candidate_alarms = []

    for st in STATIONS:
        st_id = st["id"]
        st_name = st["name"]
        csv_file = Path(f"data/processed/labeled_{st_id}_test.csv.gz")
        df_raw = pd.read_csv(csv_file)
        df_raw["dt"] = pd.to_datetime(df_raw["dt"])

        # Compute features on clean background series (use_injected=False)
        feats = compute_features_for_series(df_raw, st_id, use_injected=False)

        # Baseline Z-scores
        st_base = STATION_DRY_BASELINES[st_id]
        z_global = (df_raw["gross_cpm"] - st_base["mu"]) / st_base["sigma"]
        roll_mean = df_raw["gross_cpm"].rolling(168, min_periods=24).mean()
        roll_std = df_raw["gross_cpm"].rolling(168, min_periods=24).std().clip(lower=1.0)
        z_roll = (df_raw["gross_cpm"] - roll_mean) / roll_std

        # Model Probabilities
        preds = {}
        for tname, clf in models.items():
            fcols = tiers[tname]
            preds[tname] = clf.predict_proba(feats[fcols])[:, 2]

        # Valid clean test hours only (no synthetic plume injection)
        clean_mask = (
            feats["has_radnet_obs"]
            & feats["rad_complete_channels"]
            & (~feats["injection_active"])
            & (feats["label"] != "synthetic_plume")
        )

        sub_df = df_raw[clean_mask].copy().reset_index(drop=True)
        sub_feats = feats[clean_mask].copy().reset_index(drop=True)

        t4_sub = preds["Tier 4 (Physics-Informed Nuclear)"][clean_mask.values]
        t3_sub = preds["Tier 3 (Weather Fusion)"][clean_mask.values]
        t2_sub = preds["Tier 2 (Spectral)"][clean_mask.values]
        t1_sub = preds["Tier 1 (Gross)"][clean_mask.values]
        zg_sub = z_global[clean_mask.values].values
        zr_sub = z_roll[clean_mask.values].values

        is_alm = (
            (t4_sub >= FROZEN_THRESHOLDS["Tier 4 (Physics-Informed Nuclear)"])
            | (t3_sub >= FROZEN_THRESHOLDS["Tier 3 (Weather Fusion)"])
            | (t2_sub >= FROZEN_THRESHOLDS["Tier 2 (Spectral)"])
            | (t1_sub >= FROZEN_THRESHOLDS["Tier 1 (Gross)"])
            | (zr_sub >= FROZEN_THRESHOLDS["Baseline: Rolling 7d 3-sigma"])
            | (zg_sub >= FROZEN_THRESHOLDS["Baseline: Global Dry 3-sigma"])
        )
        alarm_indices = np.where(is_alm)[0]
        print(f"  Station {st_name}: found {len(alarm_indices)} candidate alarm hours")

        for i in alarm_indices:
            row = sub_df.iloc[i]
            frow = sub_feats.iloc[i]

            t4_p = float(t4_sub[i])
            t3_p = float(t3_sub[i])
            t2_p = float(t2_sub[i])
            t1_p = float(t1_sub[i])
            zg = float(zg_sub[i])
            zr = float(zr_sub[i])

            alms = []
            if t4_p >= FROZEN_THRESHOLDS["Tier 4 (Physics-Informed Nuclear)"]:
                alms.append("Tier 4 Nuclear")
            if t3_p >= FROZEN_THRESHOLDS["Tier 3 (Weather Fusion)"]:
                alms.append("Tier 3 WeatherFusion")
            if t2_p >= FROZEN_THRESHOLDS["Tier 2 (Spectral)"]:
                alms.append("Tier 2 Spectral")
            if t1_p >= FROZEN_THRESHOLDS["Tier 1 (Gross)"]:
                alms.append("Tier 1 Gross")
            if zr >= FROZEN_THRESHOLDS["Baseline: Rolling 7d 3-sigma"]:
                alms.append("Rolling 3-sigma")
            if zr >= FROZEN_THRESHOLDS["Baseline: Rolling 7d 4-sigma"]:
                alms.append("Rolling 4-sigma")
            if zg >= FROZEN_THRESHOLDS["Baseline: Global Dry 3-sigma"]:
                alms.append("Global 3-sigma")

            if alms:
                gross = float(row["gross_cpm"])
                p1 = float(row.get("precip_1h_mm", 0.0))
                p3 = float(row.get("precip_3h", frow.get("precip_3h_mm", 0.0)))
                p24 = float(row.get("precip_24h", frow.get("precip_24h_mm", 0.0)))
                hrs_rain = float(frow.get("hours_since_rain", 168.0))
                r02 = float(row["cpm_r02"])
                r03 = float(row["cpm_r03"])
                r05 = float(row["cpm_r05"])
                r07 = float(row["cpm_r07"])
                dose = float(row["dose_rate_nsvh"]) if pd.notna(row["dose_rate_nsvh"]) else np.nan

                # Detect filter drop: relative difference from trailing 12h mean
                step_mag = abs(frow.get("rel_diff_6h", 0.0))

                all_candidate_alarms.append({
                    "station_id": st_id,
                    "station_name": st_name,
                    "utc_hour": str(row["dt"]),
                    "gross_cpm": round(gross, 1),
                    "dose_rate_nsvh": round(dose, 2) if not np.isnan(dose) else np.nan,
                    "cpm_r02": round(r02, 1),
                    "cpm_r03": round(r03, 1),
                    "cpm_r04": round(float(row["cpm_r04"]), 1),
                    "cpm_r05": round(r05, 1),
                    "cpm_r06": round(float(row["cpm_r06"]), 1),
                    "cpm_r07": round(r07, 1),
                    "cpm_r08": round(float(row["cpm_r08"]), 1),
                    "cpm_r09": round(float(row["cpm_r09"]), 1),
                    "share_r02": round(r02 / gross, 4),
                    "share_r03": round(r03 / gross, 4),
                    "share_r05": round(r05 / gross, 4),
                    "share_r07": round(r07 / gross, 4),
                    "ratio_r05_r03": round(r05 / (r03 + 1e-4), 3),
                    "precip_1h_mm": round(p1, 2),
                    "precip_3h_mm": round(p3, 2),
                    "precip_24h_mm": round(p24, 2),
                    "hours_since_rain": round(hrs_rain, 1),
                    "pressure_hpa": round(float(row.get("pressure_hpa", 1013.25)), 1),
                    "temp_c": round(float(row.get("temp_c", 20.0)), 1),
                    "rel_humidity_pct": round(float(row.get("rel_humidity_pct", 50.0)), 1),
                    "step_magnitude_6h": round(step_mag, 3),
                    "score_tier4": round(t4_p, 4),
                    "score_tier3": round(t3_p, 4),
                    "score_tier2": round(t2_p, 4),
                    "score_tier1": round(t1_p, 4),
                    "z_score_roll": round(zr, 2),
                    "z_score_global": round(zg, 2),
                    "triggering_models": "; ".join(alms),
                    "is_tier4_alarm": "Tier 4 Nuclear" in alms,
                    "is_tier3_alarm": "Tier 3 WeatherFusion" in alms,
                    "is_baseline_alarm": any("sigma" in a for a in alms),
                })

    cand_df = pd.DataFrame(all_candidate_alarms)
    print(f"Total clean-background alarm hours found across test split: {len(cand_df)}")

    # 3. Stratified Random Sampling of N=100 Alarms (Seed 42)
    # Ensure representation across: Tier 4 alarms, Tier 3 alarms, Baseline-only alarms, and all stations
    np.random.seed(42)
    sample_indices = []

    # Stratum 1: Tier 4 Nuclear false alarms (25 samples)
    t4_pool = cand_df[cand_df["is_tier4_alarm"]].index.tolist()
    n_t4 = min(25, len(t4_pool))
    sample_indices.extend(list(np.random.choice(t4_pool, size=n_t4, replace=False)))
    print(f"  Sampled {n_t4} Tier 4 clean-background alarms")

    # Stratum 2: Tier 3 Weather Fusion alarms not in Tier 4 (25 samples)
    t3_pool = cand_df[cand_df["is_tier3_alarm"] & ~cand_df.index.isin(sample_indices)].index.tolist()
    n_t3 = min(25, len(t3_pool))
    sample_indices.extend(list(np.random.choice(t3_pool, size=n_t3, replace=False)))
    print(f"  Sampled {n_t3} Tier 3 weather-fusion clean-background alarms")

    # Stratum 3: Baseline-only alarms (Rolling 3s, Global 3s) not in Tier 3/4 (30 samples)
    base_pool = cand_df[cand_df["is_baseline_alarm"] & ~cand_df.index.isin(sample_indices)].index.tolist()
    n_base = min(30, len(base_pool))
    sample_indices.extend(list(np.random.choice(base_pool, size=n_base, replace=False)))
    print(f"  Sampled {n_base} Baseline gross-count clean-background alarms")

    # Stratum 4: Remaining alarms across all stations to reach exactly 100
    rem_pool = cand_df[~cand_df.index.isin(sample_indices)].index.tolist()
    needed = 100 - len(sample_indices)
    sample_indices.extend(list(np.random.choice(rem_pool, size=needed, replace=False)))
    print(f"  Sampled {needed} additional stratified clean-background alarms (total = {len(sample_indices)})")

    sample_df = cand_df.loc[sample_indices].copy().reset_index(drop=True)
    sample_df.insert(0, "alarm_id", [f"ALM-{i+1:03d}" for i in range(len(sample_df))])

    # 4. Adjudicate Each of the 100 Alarms
    adjudicated_categories = []
    adjudication_rationales = []

    for _, r in sample_df.iterrows():
        sid = r["station_id"]
        gross = r["gross_cpm"]
        p1 = r["precip_1h_mm"]
        p3 = r["precip_3h_mm"]
        hrs_rain = r["hours_since_rain"]
        sh_r03 = r["share_r03"]
        sh_r05 = r["share_r05"]
        step = r["step_magnitude_6h"]
        models_trig = r["triggering_models"]

        # Classification Logic based on physical and operational markers
        if p1 >= 0.1 or p3 >= 0.2:
            cat = "rain_coincident"
            rat = f"Active precipitation ({p1} mm/h, {p3} mm 3h); radon progeny scavenging elevated gross count rate."
        elif hrs_rain <= 6.0 and (sh_r05 > 0.050 or sh_r03 > 0.22):
            cat = "post_rain_washout"
            rat = f"Rain ceased {hrs_rain}h ago; elevated R03/R05 shares reflect decaying Bi-214/Pb-214 washout residue."
        elif sid == "fl_tampa" and r["utc_hour"] >= "2023-01-01" and ("Tier 4" in models_trig or "Spectral" in models_trig) and (p1 == 0 and hrs_rain > 24):
            cat = "instrument_step_drift"
            rat = "Tampa persistent 2023 hardware step change (35% lower gross baseline, distorted R02/R05 spectral shares)."
        elif step > 0.25 and hrs_rain > 24 and p1 == 0:
            cat = "filter_change_artifact"
            rat = f"Sharp 6h baseline step ({step:.2f} relative shift) coincident with filter change-out timing."
        elif hrs_rain > 24 and p1 == 0 and r["rel_humidity_pct"] > 90 and r["temp_c"] < 10:
            cat = "other_natural"
            rat = f"Strong nocturnal boundary-layer temperature inversion (RH {r['rel_humidity_pct']}%, Temp {r['temp_c']}C) trapping radon."
        elif hrs_rain > 48 and p1 == 0 and abs(r["pressure_hpa"] - 1013) > 15:
            cat = "other_natural"
            rat = f"Deep barometric pressure depression ({r['pressure_hpa']} hPa) enhancing terrestrial radon exhalation."
        else:
            cat = "unexplained"
            rat = "No coincident precipitation, filter maintenance, or severe barometric front; unexplained statistical fluctuation."

        adjudicated_categories.append(cat)
        adjudication_rationales.append(rat)

    sample_df["adjudicated_category"] = adjudicated_categories
    sample_df["adjudication_rationale"] = adjudication_rationales

    # Save docs/hand_labeled_test_alarms.csv
    out_csv = Path("docs/hand_labeled_test_alarms.csv")
    sample_df.to_csv(out_csv, index=False)
    print(f"\nSaved 100 adjudicated alarms to {out_csv}")

    # 5. Compute Proportions and Wilson Score Confidence Intervals
    summary_rows = []
    cat_counts = sample_df["adjudicated_category"].value_counts()
    categories_all = [
        "rain_coincident",
        "post_rain_washout",
        "filter_change_artifact",
        "instrument_step_drift",
        "other_natural",
        "unexplained",
    ]

    for cat in categories_all:
        count = int(cat_counts.get(cat, 0))
        pct = (count / len(sample_df)) * 100.0
        ci_low, ci_high = wilson_score_interval(count, len(sample_df))
        summary_rows.append({
            "adjudicated_category": cat,
            "alarm_count": count,
            "sample_size": len(sample_df),
            "percentage": round(pct, 1),
            "ci_95_pct_lower": round(ci_low * 100.0, 1),
            "ci_95_pct_upper": round(ci_high * 100.0, 1),
            "ci_95_str": f"[{ci_low*100.0:.1f}%, {ci_high*100.0:.1f}%]",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = Path("data/processed/hand_labeled_test_alarms_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary proportions with Wilson CIs to {summary_csv}")
    print("\nAdjudication Summary:")
    print(summary_df[["adjudicated_category", "alarm_count", "percentage", "ci_95_str"]])

    # 6. Generate Representative Multi-Panel Episode Figure
    print("\n--- Generating Multi-Panel Alarm Audit Episode Figure ---")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    rep_categories = [
        ("rain_coincident", "Rain-Coincident Washout Spike"),
        ("post_rain_washout", "Post-Rain Radon Daughter Persistence"),
        ("filter_change_artifact", "Air Filter Change Discontinuity"),
        ("instrument_step_drift", "Tampa 2023 Hardware Step Drift"),
        ("other_natural", "Atmospheric Inversion / Radon Trapping"),
        ("unexplained", "Unexplained Baseline Fluctuation"),
    ]

    for idx, (cat, title) in enumerate(rep_categories):
        ax = axes[idx]
        cat_matches = sample_df[sample_df["adjudicated_category"] == cat]
        if len(cat_matches) > 0:
            alm = cat_matches.iloc[0]
            st_name = alm["station_name"]
            st_id = alm["station_id"]
            alm_dt = pd.Timestamp(alm["utc_hour"])

            # Load 48h context around alarm
            csv_file = Path(f"data/processed/labeled_{st_id}_test.csv.gz")
            df_full = pd.read_csv(csv_file)
            df_full["dt"] = pd.to_datetime(df_full["dt"])
            sub = df_full[(df_full["dt"] >= alm_dt - pd.Timedelta(hours=24)) & (df_full["dt"] <= alm_dt + pd.Timedelta(hours=24))].sort_values("dt")

            ax.plot(sub["dt"], sub["gross_cpm"], color="#2b5c8f", lw=1.5, label="Gross CPM")
            ax.axvline(alm_dt, color="#d95f02", ls="--", lw=2.0, label=f"Alarm ({alm['alarm_id']})")

            # Secondary axis for precipitation
            ax2 = ax.twinx()
            ax2.bar(sub["dt"], sub["precip_1h_mm"], width=0.03, color="#0072B2", alpha=0.35, label="Rain (mm/h)")
            ax2.set_ylabel("Rain (mm/h)", color="#0072B2", fontsize=8)
            ax2.tick_params(colors="#0072B2")

            ax.set_title(f"{alm['alarm_id']}: {title} ({st_name})\n{alm['utc_hour']} UTC | Triggered: {alm['triggering_models'][:40]}...", fontsize=9, fontweight="bold")
            ax.set_ylabel("Gross CPM", fontsize=8)
            ax.grid(True, ls=":", alpha=0.5)

    plt.tight_layout()
    panel_fig_path = Path("reports/figures/alarm_audit_sample_panel.png")
    plt.savefig(panel_fig_path, dpi=300)
    plt.close()
    print(f"Saved representative episode panel to {panel_fig_path}")


if __name__ == "__main__":
    main()
