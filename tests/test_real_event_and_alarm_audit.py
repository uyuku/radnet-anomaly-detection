"""
Unit tests for Real-Event Validation and Alarm Adjudication.
"""

from pathlib import Path
import pandas as pd
import pytest


def test_real_event_and_alarm_audit_deliverables_exist():
    required_files = [
        Path("docs/real_event_feasibility.md"),
        Path("docs/real_event_protocol.md"),
        Path("docs/real_event_limitation.md"),
        Path("docs/published_radnet_anomalies_2017_2025.md"),
        Path("docs/hand_labeled_test_alarms.csv"),
        Path("data/processed/real_event_fukushima_evaluation.csv"),
        Path("data/processed/real_event_fukushima_station_summary.csv"),
        Path("data/processed/hand_labeled_test_alarms_summary.csv"),
        Path("reports/figures/real_event_fukushima_timeline.png"),
        Path("reports/figures/alarm_audit_sample_panel.png"),
    ]
    for p in required_files:
        assert p.exists(), f"Missing required file: {p}"
        assert p.stat().st_size > 0, f"File {p} is empty"


def test_hand_labeled_test_alarms_schema_and_count():
    df = pd.read_csv("docs/hand_labeled_test_alarms.csv")
    assert len(df) == 100, f"Expected exactly 100 audited alarms, got {len(df)}"

    required_cols = [
        "alarm_id",
        "station_id",
        "station_name",
        "utc_hour",
        "gross_cpm",
        "cpm_r02",
        "cpm_r03",
        "cpm_r05",
        "cpm_r07",
        "precip_1h_mm",
        "hours_since_rain",
        "adjudicated_category",
        "adjudication_rationale",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col} in docs/hand_labeled_test_alarms.csv"

    valid_categories = {
        "rain_coincident",
        "post_rain_washout",
        "filter_change_artifact",
        "instrument_step_drift",
        "other_natural",
        "unexplained",
    }
    assert set(df["adjudicated_category"]).issubset(valid_categories), "Found invalid category"


def test_alarm_summary_proportions_and_confidence_intervals():
    df = pd.read_csv("data/processed/hand_labeled_test_alarms_summary.csv")
    assert df["alarm_count"].sum() == 100
    assert abs(df["percentage"].sum() - 100.0) < 1e-3

    for _, row in df.iterrows():
        assert 0.0 <= row["ci_95_pct_lower"] <= row["percentage"]
        assert row["percentage"] <= row["ci_95_pct_upper"] <= 100.0


def test_real_event_fukushima_station_summary():
    df = pd.read_csv("data/processed/real_event_fukushima_station_summary.csv")
    assert len(df) >= 32  # 4 stations * 8 detectors
    expected_stations = {"hi_honolulu", "ca_san_diego", "ak_anchorage", "tx_dallas"}
    assert set(df["station_id"]) == expected_stations

    required_cols = [
        "station_id",
        "station_name",
        "detector",
        "pre_event_hours",
        "event_hours",
        "post_event_hours",
        "pre_event_fa_per_yr",
        "event_fa_per_yr",
        "post_event_fa_per_yr",
        "detected_any_alarm_in_event",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing {col} in fukushima summary"
