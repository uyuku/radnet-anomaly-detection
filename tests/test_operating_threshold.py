import numpy as np
import pandas as pd
import pytest

from src.operating_threshold import select_operating_threshold


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
    })
    return series, catalog


def test_uses_net_detection_and_validation_false_alarm_episodes(validation_data):
    result = select_operating_threshold(*validation_data, thresholds=[0.2, 0.5, 0.8])
    assert result["threshold"] == 0.5
    assert result["target_met"]
    assert result["validation_detection_pct"] == 100
    assert result["validation_false_alarms_per_station_year"] == pytest.approx(8766 / 8)


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
