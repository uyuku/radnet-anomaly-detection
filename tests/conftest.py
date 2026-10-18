from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def pytest_addoption(parser):
    parser.addoption("--require-data", action="store_true", help="Fail instead of skipping integration tests when station datasets are absent")


@pytest.fixture(autouse=True)
def check_integration_inputs(request):
    if request.node.get_closest_marker("integration") is None:
        return
    from src.train_and_evaluate_models import STATIONS
    root = Path(__file__).resolve().parents[1] / "data" / "processed"
    missing = [f"labeled_{station['id']}_{split}.csv.gz"
               for station in STATIONS for split in ("train", "test")
               if not (root / f"labeled_{station['id']}_{split}.csv.gz").is_file()]
    if missing:
        reason = "Run pipeline phases 0-3 for integration inputs: " + ", ".join(missing)
        if request.config.getoption("--require-data"):
            pytest.fail(reason)
        pytest.skip(reason)


@pytest.fixture
def hourly_frame():
    """Small physical panel; gross counts equal the sum of eight channels."""
    def make(n=320, start="2023-01-01"):
        rng = np.random.default_rng(123)
        frame = pd.DataFrame({
            "dt": pd.date_range(start, periods=n, freq="h"),
            "has_radnet_obs": True, "rad_complete_channels": True,
            "dose_rate_nsvh": rng.normal(50, 2, n),
            "precip_1h_mm": np.zeros(n), "temp_c": 18.0,
            "dewpoint_c": 10.0, "pressure_hpa": 1015.0,
            "rel_humidity_pct": 65.0, "label": "normal", "injection_active": False,
        })
        for channel in range(2, 10):
            frame[f"cpm_r0{channel}"] = rng.uniform(100, 600, n)
        frame["gross_cpm"] = frame[[f"cpm_r0{c}" for c in range(2, 10)]].sum(axis=1)
        for col in ["gross_cpm", "dose_rate_nsvh"] + [f"cpm_r0{c}" for c in range(2, 10)]:
            frame[f"inj_{col}"] = frame[col]
        return frame
    return make
