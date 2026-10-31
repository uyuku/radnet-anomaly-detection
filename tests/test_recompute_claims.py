"""Unit tests for recompute_claims.py (Claim Audit)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recompute_claims import (
    recompute_baseline_rates,
    recompute_benchmark_metrics,
    recompute_dataset_accounting,
    recompute_filter_cycles,
    recompute_washout_surges,
    verify_documents_and_audit,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
README_PATH = ROOT_DIR / "README.md"
AUDIT_CSV_PATH = ROOT_DIR / "docs" / "claims_audit.csv"


def test_recompute_dataset_accounting() -> None:
    acc = recompute_dataset_accounting(DATA_DIR)
    assert acc["valid_radnet_hours"] == 334567
    assert acc["synchronous_hours"] == 324550
    assert acc["test_observed_hours"] == 106628
    assert acc["test_station_years"] == pytest.approx(12.16, abs=0.01)
    assert acc["total_plumes"] == 450
    assert acc["train_plumes"] == 250
    assert acc["test_plumes"] == 200
    assert acc["passage_hours_min"] == 3
    assert acc["passage_hours_max"] == 28
    assert acc["duration_hours_min"] == 8
    assert acc["duration_hours_max"] == 164


def test_recompute_washout_surges() -> None:
    wash = recompute_washout_surges(DATA_DIR)
    assert wash["rain_hours"] == 15602
    assert wash["pooled_median_pct"] == pytest.approx(6.9, abs=0.05)
    assert wash["pooled_p90_pct"] == pytest.approx(41.4, abs=0.05)
    assert wash["pooled_max_pct"] == pytest.approx(196.72, abs=0.05)
    assert wash["san_diego_median_pct"] == pytest.approx(-8.65, abs=0.05)


def test_recompute_benchmark_metrics() -> None:
    bm = recompute_benchmark_metrics(DATA_DIR)
    t4_95 = bm["tiers"]["Tier 4 (Physics-Motivated Nuclear)@95"]
    assert t4_95["fa_per_year_mean"] == pytest.approx(2.71, abs=0.01)
    assert t4_95["fa_per_year_std"] == pytest.approx(1.39, abs=0.01)
    assert t4_95["det_total_mean"] == pytest.approx(94.00, abs=0.01)
    assert t4_95["rain_coincident_mean"] == pytest.approx(1.2, abs=0.05)

    t1_95 = bm["tiers"]["Tier 1 (Gross)@95"]
    assert t1_95["fa_per_year_mean"] == pytest.approx(102.83, abs=0.01)
    assert t1_95["det_total_mean"] == pytest.approx(89.98, abs=0.01)

    t1b_95 = bm["tiers"]["Tier 1b (Gross+Weather)@95"]
    assert t1b_95["fa_per_year_mean"] == pytest.approx(10.39, abs=0.01)
    assert t1b_95["det_total_mean"] == pytest.approx(79.62, abs=0.01)

    assert bm["loso"]["Tier 4 (Physics-Motivated Nuclear):fl_tampa"]["fa_per_year"] == pytest.approx(169.42, abs=0.01)
    assert bm["loso"]["Tier 3 (Weather Fusion):fl_tampa"]["fa_per_year"] == pytest.approx(1151.30, abs=0.01)


def test_recompute_filter_cycles_and_baseline_summaries(tmp_path: Path) -> None:
    # Copy lightweight CSVs to tmp_path so unit test does not scan all .csv.gz files
    for fname in [
        "multi_station_filter_cycle_summary.csv",
        "baseline_threshold_evaluation.csv",
        "rigorous_baseline_continuous_roc.csv",
    ]:
        (tmp_path / fname).write_text((DATA_DIR / fname).read_text(encoding="utf-8"), encoding="utf-8")

    fc = recompute_filter_cycles(tmp_path)
    w = fc["window_2021_2024"]
    assert w["median_interval_days_min"] == pytest.approx(2.92, abs=0.01)
    assert w["median_interval_days_max"] == pytest.approx(4.54, abs=0.01)
    assert w["mean_interval_days_min"] == pytest.approx(3.97, abs=0.01)
    assert w["mean_interval_days_max"] == pytest.approx(5.43, abs=0.01)

    br = recompute_baseline_rates(tmp_path)
    assert br["pooled_test_2023_2025_rolling_7d_local_3s"] == pytest.approx(82.62, abs=0.01)
    assert br["pooled_test_2023_2025_global_dry_3s"] == pytest.approx(52.86, abs=0.01)
    assert br["per_station_2017_2025_rolling_cpm_3s"]["fl_tampa"]["episodes_per_year"] == pytest.approx(6.59, abs=0.01)
    assert br["per_station_2017_2025_rolling_cpm_3s"]["tx_dallas"]["episodes_per_year"] == pytest.approx(62.91, abs=0.01)


def test_verify_documents_and_audit_passes() -> None:
    errors = verify_documents_and_audit(README_PATH, AUDIT_CSV_PATH, {})
    assert errors == [], f"Document verification failed with errors: {errors}"

    audit_df = pd.read_csv(AUDIT_CSV_PATH)
    assert len(audit_df) >= 35
    required_cols = {"claim_id", "location", "claim_text", "type", "evidence", "status", "proposed_fix"}
    assert required_cols.issubset(set(audit_df.columns))
