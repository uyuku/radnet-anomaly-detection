import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import TIER_3_FEATURES, compute_features_for_series
from src.synthetic_injection import SCENARIOS, sample_operational_spectrum


def features(frame):
    return compute_features_for_series(frame, "al_birmingham", use_injected=False)


def test_missing_rain_does_not_become_dry(hourly_frame):
    frame = hourly_frame()
    frame.loc[190, "precip_1h_mm"] = 2.0
    frame.loc[200, "precip_1h_mm"] = np.nan
    result = features(frame)
    for column in ["precip_1h_mm", "precip_3h_mm", "precip_6h_mm", "precip_24h_mm",
                   "is_raining", "rain_recent_3h", "rain_recent_6h", "rain_recent_24h",
                   "dry_excess_interaction", "washout_expected_ratio", "hours_since_rain"]:
        assert pd.isna(result.loc[200, column]), column
    assert result.loc[199, "hours_since_rain"] == 9
    assert result.loc[200:202, "precip_3h_mm"].isna().all()
    assert result.loc[203, "precip_3h_mm"] == 0
    assert pd.isna(result.loc[203, "dry_excess_interaction"])
    assert pd.isna(result.loc[223, "precip_24h_mm"])
    assert result.loc[224, "precip_24h_mm"] == 0


def test_hours_since_rain_requires_known_history_and_recovers(hourly_frame):
    frame = hourly_frame(400)
    frame.loc[180, "precip_1h_mm"] = np.nan
    frame.loc[190, "precip_1h_mm"] = 1.0
    frame.loc[200, "precip_1h_mm"] = np.nan
    result = features(frame)
    assert result.loc[:166, "hours_since_rain"].isna().all()
    assert result.loc[167, "hours_since_rain"] == 168
    assert pd.isna(result.loc[181, "hours_since_rain"])
    assert result.loc[190, "hours_since_rain"] == 0
    assert result.loc[191, "hours_since_rain"] == 1
    assert pd.isna(result.loc[367, "hours_since_rain"])
    assert result.loc[368, "hours_since_rain"] == 168


@pytest.mark.parametrize("defect", ["missing", "duplicate", "reversed", "nat", "empty"])
def test_rejects_invalid_hourly_grid(hourly_frame, defect):
    frame = hourly_frame(30)
    if defect == "missing":
        frame = frame.drop(index=5)
    elif defect == "duplicate":
        frame.loc[5, "dt"] = frame.loc[4, "dt"]
    elif defect == "reversed":
        frame = frame.iloc[::-1]
    elif defect == "nat":
        frame.loc[5, "dt"] = pd.NaT
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError, match="hourly UTC"):
        features(frame)


def test_equivalent_timezone_representation(hourly_frame):
    frame = hourly_frame()
    utc = features(frame)
    frame["dt"] = frame["dt"].dt.tz_localize("UTC").dt.tz_convert("Europe/Istanbul")
    pd.testing.assert_frame_equal(utc, features(frame))


def test_physical_conservation_without_downloads(hourly_frame):
    result = features(hourly_frame())
    shares = result[[f"share_r0{channel}" for channel in range(2, 10)]]
    np.testing.assert_allclose(shares.sum(axis=1), 1.0)
    assert len(TIER_3_FEATURES) == len(set(TIER_3_FEATURES)) == 48
    for scenario in SCENARIOS:
        spectrum, nuclides = sample_operational_spectrum(np.random.RandomState(42), scenario)
        assert np.all(spectrum >= 0)
        assert np.sum(spectrum) == pytest.approx(1)
        assert sum(nuclides.values()) == pytest.approx(1)
