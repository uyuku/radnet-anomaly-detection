import numpy as np
import pandas as pd
import pytest

from src.operating_threshold import (
    FINE_THRESHOLD_GRID,
    THRESHOLD_GRID,
    build_rank_threshold_grid,
    compute_event_detection_metrics,
    cross_fit_calibrate_validation_and_test,
    select_fa_budget_threshold,
    select_operating_threshold,
)


@pytest.fixture
def validation_data():
    dt = pd.date_range("2021-02-01", periods=8, freq="h")
    series = {"station": pd.DataFrame({
        "dt": dt, "valid": True,
        "p_clean": [0.1, 0.3, 0.3, 0.1, 0.1, 0.1, 0.9, 0.1],
        "p_injected": [0.1, 0.7, 0.7, 0.1, 0.9, 0.9, 0.95, 0.1],
    })}
    catalog = pd.DataFrame({
        "station_id": ["station", "station"], "split": ["train", "train"],
        "start_utc": [dt[1], dt[4]], "end_utc": [dt[2], dt[5]],
        "peak_utc": [dt[1], dt[5]],
    })
    return series, catalog


def test_uses_net_detection_and_validation_false_alarm_episodes(validation_data):
    result = select_operating_threshold(*validation_data, thresholds=[0.2, 0.5, 0.8])
    assert result["threshold"] == 0.5
    assert result["target_met"]
    assert result["validation_detection_pct"] == 100
    assert result["validation_false_alarms_per_station_year"] == pytest.approx(8766 / 8)


def test_paired_vs_unpaired_detection(validation_data):
    series, catalog = validation_data
    # Suppose during the first event (hours 1-2), clean background also has p_clean = 0.75
    series["station"].loc[1:2, "p_clean"] = 0.75
    series["station"].loc[1:2, "p_injected"] = 0.80

    # At tau=0.6:
    # With paired_detection=True, hours 1-2 have c_alarm=True, so event 1 is masked out (only 50% detected)
    res_paired = select_operating_threshold(series, catalog, target=0.95, thresholds=[0.6], paired_detection=True)
    assert res_paired["validation_detection_pct"] == 50.0
    assert res_paired["target_met"] is False

    # With paired_detection=False, event 1 is still counted because p_injected >= 0.6 (100% detected)
    res_unpaired = select_operating_threshold(series, catalog, target=0.95, thresholds=[0.6], paired_detection=False)
    assert res_unpaired["validation_detection_pct"] == 100.0
    assert res_unpaired["target_met"] is True


def test_fine_and_rank_threshold_grids(validation_data):
    series, catalog = validation_data
    assert len(FINE_THRESHOLD_GRID) > len(THRESHOLD_GRID)
    rank_grid = build_rank_threshold_grid(series, n_quantiles=50)
    assert rank_grid.ndim == 1
    assert ((rank_grid >= 0.0) & (rank_grid <= 1.0)).all()

    res_fine = select_operating_threshold(series, catalog, target=0.95, thresholds="fine")
    res_rank = select_operating_threshold(series, catalog, target=0.95, thresholds="rank")
    assert res_fine["target_met"] is True
    assert res_rank["target_met"] is True


def test_select_fa_budget_threshold_and_monotone_invariance(validation_data):
    series, catalog = validation_data
    # Total valid hours = 8 => 1 episode = 8766/8 = 1095.75 FA/yr
    res_zero = select_fa_budget_threshold(series, fa_budget_per_year=0.0, catalog=catalog, thresholds=[0.2, 0.5, 0.95])
    assert res_zero["threshold"] == 0.95
    assert res_zero["validation_false_alarm_episodes"] == 0
    assert res_zero["budget_met"] is True

    res_one = select_fa_budget_threshold(series, fa_budget_per_year=1200.0, catalog=catalog, thresholds=[0.2, 0.5, 0.95])
    assert res_one["threshold"] == 0.5
    assert res_one["validation_false_alarm_episodes"] == 1
    assert res_one["validation_detection_pct"] == 100.0

    # Monotone transformation invariance (rank grid)
    series_sq = {
        k: df.assign(p_clean=df["p_clean"] ** 2, p_injected=df["p_injected"] ** 2)
        for k, df in series.items()
    }
    res_orig = select_fa_budget_threshold(series, fa_budget_per_year=1200.0, catalog=catalog, thresholds="rank")
    res_sq = select_fa_budget_threshold(series_sq, fa_budget_per_year=1200.0, catalog=catalog, thresholds="rank")
    assert res_orig["validation_false_alarm_episodes"] == res_sq["validation_false_alarm_episodes"]
    assert res_orig["validation_detection_pct"] == res_sq["validation_detection_pct"]


def test_cross_fit_calibrate_validation_and_test():
    dt_2021 = pd.date_range("2021-06-01", periods=24, freq="h")
    dt_2022 = pd.date_range("2022-06-01", periods=24, freq="h")
    dt_val = dt_2021.append(dt_2022)
    dt_test = pd.date_range("2023-06-01", periods=24, freq="h")

    rng = np.random.default_rng(42)
    p_clean_val = rng.uniform(0.01, 0.3, size=len(dt_val))
    p_inj_val = p_clean_val.copy()
    p_inj_val[5:10] = 0.85
    p_inj_val[30:35] = 0.90
    y_val = np.zeros(len(dt_val), dtype=int)
    y_val[5:10] = 1
    y_val[30:35] = 1

    val_series = {
        "st1": pd.DataFrame({
            "dt": dt_val,
            "valid": True,
            "p_clean": p_clean_val,
            "p_injected": p_inj_val,
            "y_fission": y_val,
        })
    }
    test_series = {
        "st1": pd.DataFrame({
            "dt": dt_test,
            "valid": True,
            "p_clean": rng.uniform(0.01, 0.3, size=len(dt_test)),
            "p_injected": rng.uniform(0.1, 0.9, size=len(dt_test)),
        })
    }

    for method in ("isotonic", "platt"):
        v22, v21, t21, t22 = cross_fit_calibrate_validation_and_test(val_series, test_series, method=method)
        assert len(v22["st1"]) == 24
        assert len(v21["st1"]) == 24
        assert len(t21["st1"]) == 24
        assert ((t21["st1"]["p_clean"] >= 0.0) & (t21["st1"]["p_clean"] <= 1.0)).all()


def test_compute_event_detection_metrics():
    plume_records = pd.DataFrame({
        "detected": [True, True, False, True],
        "delay_hours": [2.0, 5.0, np.nan, 18.0],
        "alarm_hours_in_plume": [8.0, 6.0, 0.0, 4.0],
        "duration_hours": [12.0, 12.0, 24.0, 24.0],
    })
    m = compute_event_detection_metrics(plume_records, reference_window_hours=12.0)
    assert m["det_total_pct"] == 75.0
    assert m["det_le_3h_pct"] == 25.0
    assert m["det_le_6h_pct"] == 50.0
    assert m["det_le_12h_pct"] == 50.0
    assert m["det_le_24h_pct"] == 75.0
    assert m["delay_median_hours"] == 5.0
    assert 0.0 < m["time_normalized_12h_det_pct"] <= 100.0
    assert 0.0 < m["mean_active_hour_recall_pct"] <= 100.0


def test_test_catalog_does_not_affect_calibration(validation_data):
    series, catalog = validation_data
    expected = select_operating_threshold(series, catalog)
    extra = catalog.copy()
    extra["split"] = "test"
    extra["start_utc"] += pd.DateOffset(years=2)
    extra["end_utc"] += pd.DateOffset(years=2)
    assert select_operating_threshold(series, pd.concat([catalog, extra])) == expected


def test_unreachable_target_is_reported(validation_data):
    result = select_operating_threshold(*validation_data, thresholds=[0.8, 0.9])
    assert result["target_met"] is False
    assert result["validation_detection_pct"] == 50


def test_outages_do_not_count_as_event_detections(validation_data):
    series, catalog = validation_data
    series["station"].loc[1:2, "valid"] = False
    result = select_operating_threshold(series, catalog, thresholds=[0.5])
    assert result["validation_detection_pct"] == 50
    assert result["validation_observed_hours"] == 6


@pytest.mark.parametrize("defect", ["test_hours", "gap", "no_observations", "nan_prediction", "no_events"])
def test_rejects_invalid_calibration_inputs(validation_data, defect):
    series, catalog = validation_data
    if defect == "test_hours":
        series["station"]["dt"] += pd.DateOffset(years=2)
    elif defect == "gap":
        series["station"] = series["station"].drop(index=3)
    elif defect == "no_observations":
        series["station"]["valid"] = False
    elif defect == "nan_prediction":
        series["station"].loc[1, "p_clean"] = np.nan
    else:
        catalog = catalog.iloc[:0]
    with pytest.raises(ValueError):
        select_operating_threshold(series, catalog)


@pytest.mark.parametrize("thresholds", [[], [np.nan], [-0.1], [1.1]])
def test_rejects_invalid_thresholds(validation_data, thresholds):
    with pytest.raises(ValueError, match="Thresholds"):
        select_operating_threshold(*validation_data, thresholds=thresholds)


def test_threshold_diagnosis_output_tables_and_flags():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "processed"
    for fname in [
        "threshold_saturation_by_seed.csv",
        "threshold_saturation_summary.csv",
        "benchmark_fa_budget_operating_points.csv",
        "rigorous_benchmark_summary.csv",
        "threshold_paired_vs_unpaired.csv",
        "threshold_event_metrics_and_delays.csv",
        "threshold_calibration_diagnostics.csv",
        "threshold_calibration_fa_budget_comparison.csv",
        "threshold_label_window_sensitivity.csv",
        "threshold_class_weight_sensitivity.csv",
        "tier_per_plume.csv",
        "tier_per_month.csv",
    ]:
        p = data_dir / fname
        assert p.exists(), f"Missing threshold diagnosis output CSV: {fname}"
        df = pd.read_csv(p)
        assert not df.empty, f"Empty threshold diagnosis output CSV: {fname}"

    rig = pd.read_csv(data_dir / "rigorous_benchmark_summary.csv")
    assert "target_met" in rig.columns
    assert "n_seeds_tau_90_eq_95" in rig.columns

    fab = pd.read_csv(data_dir / "benchmark_fa_budget_operating_points.csv")
    assert set(fab["fa_budget_per_yr"].unique()) == {1.0, 2.0, 5.0, 10.0, 20.0}
    assert fab["budget_met_all_seeds"].all()

