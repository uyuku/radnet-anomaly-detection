"""
Rigorous Automated Test Suite for the Weather-Aware RadNet Anomaly Detection Pipeline.

Approaches feature engineering, data hygiene, synthetic injection physics, and
evaluation tables with strict invariance and non-leakage checks:
  1. Strict temporal causality (zero future lookahead in all 48 Tier 3 features).
  2. Physical conservation laws (NaI(Tl) spectral shares sum to 1.0, template normalization).
  3. Zero test-split leakage into STATION_DRY_BASELINES.
  4. NOAA ISD hourly precipitation deduplication and reconciliation with merged datasets.
  5. Unobserved detector outage hours never mislabeled as fission_product/washout/normal.
  6. Validation-vs-test split separation and cross-table consistency across all CSV artifacts.
  7. End-to-end verification of the operational CLI diagnostic utility.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.detect_anomalies_cli import run_station_diagnostics
from src.feature_engineering import (
    STATION_DRY_BASELINES,
    TIER_3_FEATURES,
    compute_features_for_series,
)
from src.synthetic_injection import SCENARIOS, sample_operational_spectrum
from src.train_and_evaluate_models import STATIONS


def test_causal_feature_engineering_no_lookahead():
    """Perturbing future timestamps (t > 200) must have zero effect on features at t <= 200."""
    rng = np.random.default_rng(12345)
    n = 320
    dt = pd.date_range("2023-01-01", periods=n, freq="h")

    base_df = pd.DataFrame({
        "dt": dt,
        "has_radnet_obs": True,
        "rad_complete_channels": True,
        "gross_cpm": rng.normal(3800.0, 120.0, size=n),
        "dose_rate_nsvh": rng.normal(80.0, 4.0, size=n),
        "precip_1h_mm": np.zeros(n),
        "temp_c": rng.normal(18.0, 5.0, size=n),
        "dewpoint_c": rng.normal(10.0, 4.0, size=n),
        "pressure_hpa": rng.normal(1015.0, 6.0, size=n),
        "rel_humidity_pct": rng.uniform(40.0, 85.0, size=n),
        "label": "normal",
        "injection_active": False,
    })
    for ch in range(2, 10):
        base_df[f"cpm_r0{ch}"] = rng.uniform(100.0, 600.0, size=n)

    perturbed_df = base_df.copy()
    # Inject a massive future anomaly after index 200
    perturbed_df.loc[201:, "gross_cpm"] += 50000.0
    perturbed_df.loc[201:, "dose_rate_nsvh"] += 1500.0
    perturbed_df.loc[201:, "precip_1h_mm"] = 120.0
    perturbed_df.loc[201:, "temp_c"] += 30.0
    perturbed_df.loc[201:, "pressure_hpa"] -= 45.0
    for ch in range(2, 10):
        perturbed_df.loc[201:, f"cpm_r0{ch}"] *= 15.0

    feats_base = compute_features_for_series(base_df, "al_birmingham", use_injected=False)
    feats_pert = compute_features_for_series(perturbed_df, "al_birmingham", use_injected=False)

    for col in TIER_3_FEATURES:
        a = feats_base.loc[:200, col].to_numpy(dtype=float)
        b = feats_pert.loc[:200, col].to_numpy(dtype=float)
        assert np.allclose(a, b, atol=0.0, rtol=0.0, equal_nan=True), (
            f"Future lookahead leakage detected in feature '{col}' at or before t=200!"
        )


def test_spectral_shares_sum_to_one_and_templates_normalized():
    """Spectral channel shares R02..R09 must sum to 1.0 and all synthetic templates must sum to 1.0."""
    rng = np.random.RandomState(42)
    for name in SCENARIOS:
        w, nuc = sample_operational_spectrum(rng, name)
        assert len(w) == 8
        assert abs(float(np.sum(w)) - 1.0) < 1e-6, f"Nuclide template {name} sums to {np.sum(w)}, expected 1.0"
        assert np.all(w >= 0.0), f"Negative channel weight in template {name}"
        assert abs(sum(nuc.values()) - 1.0) < 1e-6

    df_te = pd.read_csv(PROJECT_ROOT / "data/processed/labeled_al_birmingham_test.csv.gz").head(500)
    feats = compute_features_for_series(df_te, "al_birmingham", use_injected=True)
    valid = feats["has_radnet_obs"] & feats["rad_complete_channels"]
    share_cols = [f"share_r0{i}" for i in range(2, 10)]
    share_sums = feats.loc[valid, share_cols].sum(axis=1).to_numpy()
    assert np.allclose(share_sums, 1.0, atol=1e-4), "Spectral channel shares R02..R09 do not sum to 1.0"


def test_station_dry_baselines_strictly_train_only_no_test_leakage():
    """STATION_DRY_BASELINES must match pre-2023 training dry-weather statistics with zero test leakage."""
    for st in STATIONS:
        st_id = st["id"]
        df_tr = pd.read_csv(PROJECT_ROOT / f"data/processed/labeled_{st_id}_train.csv.gz")
        df_tr["dt"] = pd.to_datetime(df_tr["dt"])
        df_tr = df_tr.sort_values("dt").reset_index(drop=True)
        precip_24h = df_tr["precip_1h_mm"].rolling(24, min_periods=12).sum()
        valid_dry = (
            df_tr["has_radnet_obs"].fillna(False).astype(bool)
            & df_tr["rad_complete_channels"].fillna(False).astype(bool)
            & (precip_24h == 0.0)
            & (df_tr["precip_1h_mm"] == 0.0)
        )
        emp_mu = round(float(df_tr.loc[valid_dry, "gross_cpm"].mean()), 1)
        emp_sigma = round(float(df_tr.loc[valid_dry, "gross_cpm"].std()), 1)

        cfg = STATION_DRY_BASELINES[st_id]
        assert abs(cfg["mu"] - emp_mu) <= 0.1, f"{st_id} mu={cfg['mu']} != train dry mu={emp_mu:.2f}"
        assert abs(cfg["sigma"] - emp_sigma) <= 0.1, f"{st_id} sigma={cfg['sigma']} != train dry sigma={emp_sigma:.2f}"


def test_noaa_precipitation_audit_matches_merge_audit():
    """NOAA precipitation audit CSVs must deduplicate hourly rain and match merge_audit_summary.csv."""
    audit_df = pd.read_csv(PROJECT_ROOT / "data/processed/noaa_precipitation_audit.csv")
    summary_df = pd.read_csv(PROJECT_ROOT / "data/processed/noaa_station_summary.csv")
    merge_df = pd.read_csv(PROJECT_ROOT / "data/processed/merge_audit_summary.csv")

    for _, m_row in merge_df.iterrows():
        city = m_row["Station"].split(",")[0].strip()
        a_rows = audit_df[audit_df["station_name"].str.contains(city, case=False)]
        s_row = summary_df[summary_df["City"].str.contains(city, case=False)].iloc[0]

        merge_ann = float(m_row["Total Precip (mm)"]) / 9.0
        audit_ann = float(a_rows["total_precip_mm"].sum() / 9.0)
        summ_ann = float(s_row["Mean Annual Precip (mm)"])

        assert abs(audit_ann - merge_ann) <= 0.5, (
            f"NOAA audit annual rain ({audit_ann}) disagrees with merge audit ({merge_ann}) for {city}"
        )
        assert abs(summ_ann - merge_ann) <= 0.5, (
            f"NOAA summary annual rain ({summ_ann}) disagrees with merge audit ({merge_ann}) for {city}"
        )
        assert float(a_rows["max_1hr_precip_mm"].max()) <= 105.0, (
            f"Unphysical 1h precipitation ({a_rows['max_1hr_precip_mm'].max()} mm) suggests multi-period AA1 leakage"
        )


def test_labeled_dataset_reconciliation_and_no_unobserved_label_leakage():
    """Unobserved detector outage hours must never be labeled as fission_product, radon_washout, or normal."""
    rec_df = pd.read_csv(PROJECT_ROOT / "data/processed/labeled_dataset_reconciliation_summary.csv")
    merge_df = pd.read_csv(PROJECT_ROOT / "data/processed/merge_audit_summary.csv")

    total_test_obs = 0
    for st in STATIONS:
        st_id = st["id"]
        st_unobs = 0
        for split in ["train", "test"]:
            df = pd.read_csv(PROJECT_ROOT / f"data/processed/labeled_{st_id}_{split}.csv.gz")
            valid = df["has_radnet_obs"] & df["rad_complete_channels"] & df["gross_cpm"].notna()
            unobs_labels = df.loc[~valid, "label"].unique().tolist()
            assert unobs_labels == ["unobserved"] or len(unobs_labels) == 0, (
                f"Found non-unobserved labels {unobs_labels} on invalid/outage rows in {st_id} ({split})"
            )
            obs_labels = set(df.loc[valid, "label"].unique().tolist())
            assert "unobserved" not in obs_labels, (
                f"Found 'unobserved' label on valid observed rows in {st_id} ({split})"
            )
            split_unobs = int((~valid).sum())
            rec_split_row = rec_df[(rec_df["Station_ID"] == st_id) & (rec_df["Split"] == split)].iloc[0]
            assert int(rec_split_row["Unobserved_Hours"]) == split_unobs
            st_unobs += split_unobs
            if split == "test":
                total_test_obs += int(valid.sum())

    # Verify total observed hours across all 5 stations match merge_audit_summary.csv
    rec_obs_total = int(
        (rec_df["Normal_Hours"] + rec_df["Washout_Hours"] + rec_df["Fission_Hours"]).sum()
    )
    assert rec_obs_total == int(merge_df["RadNet Valid Hours"].sum())
    assert total_test_obs == 106628


def test_synthetic_injection_catalog_physical_verification():
    """All empirical checks in injection_catalog_audit_verification.csv must PASSED."""
    ver_df = pd.read_csv(PROJECT_ROOT / "data/processed/injection_catalog_audit_verification.csv")
    assert (ver_df["status"] == "PASSED").all(), f"Failed catalog verification checks:\n{ver_df[ver_df['status'] != 'PASSED']}"

    cat_df = pd.read_csv(PROJECT_ROOT / "data/processed/synthetic_injection_catalog.csv")
    assert len(cat_df) == 450
    assert int((cat_df["split"] == "train").sum()) == 250
    assert int((cat_df["split"] == "test").sum()) == 200


def test_baseline_and_filter_cycle_summaries():
    """Filter cycle summary must cover all 5 stations and baseline evaluation must use 8766.0 h/yr."""
    fc_df = pd.read_csv(PROJECT_ROOT / "data/processed/multi_station_filter_cycle_summary.csv")
    assert len(fc_df) == 5, f"Expected 5 stations in multi_station_filter_cycle_summary.csv, got {len(fc_df)}"
    assert (fc_df["Median_Interval_Days"] >= 1.5).all() and (fc_df["Median_Interval_Days"] <= 5.5).all()

    base_df = pd.read_csv(PROJECT_ROOT / "data/processed/baseline_threshold_evaluation.csv")
    merge_df = pd.read_csv(PROJECT_ROOT / "data/processed/merge_audit_summary.csv")
    valid_hours_by_station = dict(zip(merge_df["Station"], merge_df["Synchronous Both Hours"]))
    assert "Episode_Rain_3h_Pct" in base_df.columns
    for _, row in base_df.iterrows():
        exact_years = float(valid_hours_by_station[row["Station"]]) / 8766.0
        expected_fa = round(float(row["Total_Alarm_Episodes"]) / exact_years, 2)
        assert abs(float(row["Episodes_Per_Year"]) - expected_fa) <= 0.02


def test_validation_vs_test_separation_and_cross_table_consistency():
    """Cross-checks evaluation summary CSVs to prevent validation/test mixups or degenerate calibrations."""
    mlp_comp = pd.read_csv(PROJECT_ROOT / "data/processed/eval_mlp_comparison_summary.csv")
    bench_unc = pd.read_csv(PROJECT_ROOT / "data/processed/eval_benchmark_uncertainty_summary.csv")

    row_95 = mlp_comp[mlp_comp["target_detection_rate_pct"] == 95.0].iloc[0]
    t3_bench = bench_unc[bench_unc["model_name"] == "Tier 3: Full Weather Fusion GBDT (95% target)"].iloc[0]
    mlp_bench = bench_unc[bench_unc["model_name"] == "Tier 3: Weather-Fused Neural Net (MLP) (95% target)"].iloc[0]

    # Must match realized test metrics, never validation fold metrics (0.12 FA/yr)
    assert abs(float(row_95["lgb_realized_detection_pct"]) - float(t3_bench["realized_detection_pct"])) < 1e-6
    assert abs(float(row_95["lgb_false_alarms_per_year"]) - float(t3_bench["false_alarms_per_station_year"])) < 1e-6
    assert abs(float(row_95["mlp_realized_detection_pct"]) - float(mlp_bench["realized_detection_pct"])) < 1e-6
    assert abs(float(row_95["mlp_false_alarms_per_year"]) - float(mlp_bench["false_alarms_per_station_year"])) < 1e-6

    # Check isotonic calibration table is non-degenerate and consistent with rigorous benchmark
    cal_df = pd.read_csv(PROJECT_ROOT / "data/processed/eval_mlp_calibration_comparison.csv")
    iso_row = cal_df[cal_df["model"].str.contains("Isotonic")].iloc[0]
    assert float(iso_row["realized_detection_pct"]) >= 80.0, "Isotonic calibration collapsed to degenerate 0% detection!"
    assert float(iso_row["false_alarms_per_year"]) > 0.0

    # Check rigorous_stratified_deadlines.csv All Test Injections matches rigorous_benchmark_summary.csv
    rig_bench = pd.read_csv(PROJECT_ROOT / "data/processed/rigorous_benchmark_summary.csv")
    strat_dead = pd.read_csv(PROJECT_ROOT / "data/processed/rigorous_stratified_deadlines.csv")
    for tier_name in ["Tier 2 (Spectral)", "Tier 3 (Weather Fusion)"]:
        b_row = rig_bench[(rig_bench["tier_name"] == tier_name) & (rig_bench["target_detection_pct"] == 95.0)].iloc[0]
        s_row = strat_dead[(strat_dead["stratum"] == "All Test Injections") & (strat_dead["tier_name"] == tier_name)].iloc[0]
        assert abs(float(b_row["det_total_mean"]) - float(s_row["detection_total_pct"])) <= 0.05
        assert abs(float(b_row["det_6h_mean"]) - float(s_row["detection_le_6h_pct"])) <= 0.05
        assert abs(float(b_row["det_12h_mean"]) - float(s_row["detection_le_12h_pct"])) <= 0.05
        assert abs(float(b_row["det_24h_mean"]) - float(s_row["detection_le_24h_pct"])) <= 0.05
        assert abs(float(b_row["median_delay_mean"]) - float(s_row["median_delay_hours"])) <= 0.05


def test_cli_diagnostic_utility_detects_hard_case_plume():
    """Operational CLI diagnostic utility must detect the Birmingham Jan 2023 storm + Cs-137 plume (INJ_0067)."""
    res = run_station_diagnostics(
        station_id="al_birmingham",
        split="test",
        tier="tier3",
        use_injected=True,
        start_utc="2023-01-24 18:00:00",
        end_utc="2023-01-29 06:00:00",
    )
    assert res["ml_alarm_episodes"] >= 1
    assert res["top_episodes"][0]["peak_p_fission"] >= 0.99
    assert "fission_product" in res["top_episodes"][0]["ground_truth_labels"]
