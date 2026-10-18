import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src import detect_anomalies_cli as cli


def episode_frame():
    return pd.DataFrame({
        "dt": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 04:00", "2023-01-01 05:00", "2023-01-01 06:00"]),
        "is_alarm": True, "valid_obs": [True, True, True, False, True],
        "p_fission": 0.99, "p_washout": 0.005, "eval_gross_cpm": 4000.0,
        "z_score_168h": np.nan, "precip_1h_mm": np.nan, "label": pd.NA,
    })


def test_episodes_split_on_time_gaps_and_outages():
    episodes = cli.extract_alarm_episodes(episode_frame())
    assert [ep["duration_hours"] for ep in episodes] == [2, 1, 1]
    assert episodes[0]["precip_missing_hours"] == 2
    assert episodes[0]["total_precip_mm"] is None
    assert episodes[0]["peak_z_score_168h"] is None
    assert episodes[0]["ground_truth_labels"] == []
    json.dumps(episodes, allow_nan=False)


def test_duplicate_alarm_timestamps_are_rejected():
    frame = episode_frame()
    frame.loc[1, "dt"] = frame.loc[0, "dt"]
    with pytest.raises(ValueError, match="unique"):
        cli.extract_alarm_episodes(frame)


@pytest.mark.parametrize("kwargs", [
    {"threshold": float("nan")}, {"threshold": -0.01}, {"threshold": 1.01},
    {"split": "typo"}, {"tier": "typo"}, {"explain_top": -1},
    {"start_utc": "2023-02-01", "end_utc": "2023-01-01"}, {"start_utc": "NaT"},
])
def test_invalid_arguments_fail_before_training(kwargs, monkeypatch):
    monkeypatch.setattr(cli, "train_tier_model", lambda *args, **kw: pytest.fail("Training must not start"))
    with pytest.raises(ValueError):
        cli.run_station_diagnostics("al_birmingham", **kwargs)


@pytest.fixture
def fake_diagnostics(tmp_path, hourly_frame, monkeypatch):
    train = hourly_frame(192, "2022-12-24")
    target = hourly_frame(48, "2023-01-01")
    target["label"] = "fission_product"
    for station in cli.VALID_STATION_IDS:
        train.to_csv(tmp_path / f"labeled_{station}_train.csv.gz", index=False)
    target.to_csv(tmp_path / "labeled_al_birmingham_test.csv.gz", index=False)
    (tmp_path / "synthetic_injection_catalog.csv").write_text("fixture", encoding="utf-8")

    class Model:
        # Deliberately nonstandard class order catches positional probability bugs.
        classes_ = np.array([2, 0, 1])

        def predict_proba(self, x):
            return np.tile([0.99, 0.005, 0.005], (len(x), 1))

    model = Model()
    monkeypatch.setattr(cli, "load_and_prepare_train_data", lambda *args, **kw: (pd.DataFrame(), pd.Series(dtype=int)))
    monkeypatch.setattr(cli, "train_tier_model", lambda *args: model)
    calls = []

    def calibrate(fitted, columns, stations, directory):
        assert fitted is model
        calls.append(directory)
        return {"threshold": 0.98, "source": "validation_predictions_of_fitted_model", "target_met": True}

    monkeypatch.setattr(cli, "calibrate_model_threshold", calibrate)
    return tmp_path, target, calls


def test_clean_diagnostics_do_not_reuse_synthetic_labels(fake_diagnostics):
    directory, _, calls = fake_diagnostics
    result = cli.run_station_diagnostics("al_birmingham", data_dir=directory)
    assert calls == [directory]
    assert result["decision_threshold_tau"] == 0.98
    assert result["ml_alarm_episodes"] == 1
    assert result["top_episodes"][0]["peak_p_fission"] == 0.99
    assert result["top_episodes"][0]["ground_truth_labels"] == []
    assert result["label_source"] == "unlabeled_background"


def test_custom_threshold_skips_calibration_and_normalizes_utc(fake_diagnostics):
    directory, _, calls = fake_diagnostics
    result = cli.run_station_diagnostics(
        "al_birmingham", data_dir=directory, threshold=1.0,
        start_utc="2023-01-01T03:00:00+03:00", end_utc="2023-01-01T05:00:00+03:00",
    )
    assert calls == []
    assert result["total_hours"] == 3
    assert result["window_start_utc"] == "2023-01-01 00:00:00"
    assert result["ml_alarm_episodes"] == 0
    assert result["calibration"]["source"] == "user_override"


def test_empty_observation_window_reports_null_rates(fake_diagnostics):
    directory, target, _ = fake_diagnostics
    target["has_radnet_obs"] = False
    target["gross_cpm"] = np.nan
    target.to_csv(directory / "labeled_al_birmingham_test.csv.gz", index=False)
    result = cli.run_station_diagnostics("al_birmingham", data_dir=directory)
    assert result["valid_observed_hours"] == 0
    assert result["ml_alarms_per_station_year"] is None
    assert result["rolling_3sigma_alarms_per_station_year"] is None
    json.dumps(result, allow_nan=False)


def test_json_errors_and_help_work_outside_repository(tmp_path):
    script = Path(cli.__file__)
    result = subprocess.run([sys.executable, str(script), "--threshold", "nan", "--json"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert "threshold" in json.loads(result.stdout)["error"]
    assert "Traceback" not in result.stderr
    help_result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "--data-dir" in help_result.stdout
