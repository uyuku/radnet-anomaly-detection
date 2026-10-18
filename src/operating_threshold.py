"""Select an operating threshold for the fitted model using validation data only."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import compute_features_for_series


VALIDATION_START = pd.Timestamp("2021-01-01")
TEST_START = pd.Timestamp("2023-01-01")
HOURS_PER_YEAR = 8766.0
THRESHOLD_GRID = np.concatenate([
    np.linspace(0.01, 0.95, 95),
    [0.96, 0.97, 0.98, 0.985, 0.990, 0.991, 0.992, 0.993, 0.994,
     0.995, 0.996, 0.997, 0.998, 0.999],
])


def select_operating_threshold(
    series: dict[str, pd.DataFrame],
    catalog: pd.DataFrame,
    target: float = 0.95,
    thresholds=None,
) -> dict:
    """Minimize clean alarm episodes subject to event-level net detection.

    Each frame contains dt, valid, p_injected and p_clean on a continuous hourly
    validation grid. An event is detected only where injected alarms exceed the
    threshold while the paired clean series does not. Unreachable targets are
    reported explicitly, using best detection then fewest false alarms.
    """
    grid = np.asarray(THRESHOLD_GRID if thresholds is None else thresholds, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or ((grid < 0) | (grid > 1)).any():
        raise ValueError("Thresholds must be finite probabilities in [0, 1].")
    if not np.isfinite(target) or not 0 < target <= 1:
        raise ValueError("Detection target must be in (0, 1].")
    grid = np.unique(grid)
    cat = catalog.copy()
    starts = pd.to_datetime(cat["start_utc"], utc=True).dt.tz_localize(None)
    ends = pd.to_datetime(cat["end_utc"], utc=True).dt.tz_localize(None)
    cat["start_utc"], cat["end_utc"] = starts, ends
    cat = cat.loc[(cat["split"] == "train") & (starts >= VALIDATION_START) & (ends < TEST_START)]
    if cat.empty:
        raise ValueError("No 2021-2022 validation injections are available for calibration.")
    if (cat["end_utc"] < cat["start_utc"]).any():
        raise ValueError("Validation injection ends before it starts.")
    if not set(cat["station_id"]).issubset(series):
        raise ValueError("Validation predictions are missing a station in the catalog.")

    detections = np.zeros(len(grid), dtype=int)
    episodes = np.zeros(len(grid), dtype=int)
    observed_hours = 0
    for station_id, frame in series.items():
        dt = pd.to_datetime(frame["dt"], utc=True).dt.tz_localize(None)
        if dt.empty or dt.isna().any() or not dt.between(VALIDATION_START, TEST_START, inclusive="left").all():
            raise ValueError("Calibration predictions must contain only 2021-2022 validation hours.")
        if not dt.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError("Calibration requires a continuous, sorted hourly grid.")
        valid = frame["valid"].fillna(False).to_numpy(dtype=bool)
        p_clean = frame["p_clean"].to_numpy(dtype=float)
        p_injected = frame["p_injected"].to_numpy(dtype=float)
        for probabilities in (p_clean[valid], p_injected[valid]):
            if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError("Valid observations must have finite probabilities in [0, 1].")
        alarm = (p_clean[:, None] >= grid) & valid[:, None]
        prior = np.vstack([np.zeros((1, len(grid)), dtype=bool), alarm[:-1]])
        episodes += (alarm & ~prior).sum(axis=0)
        observed_hours += int(valid.sum())
        net_alarm = (p_injected[:, None] >= grid) & ~alarm & valid[:, None]
        for event in cat.loc[cat["station_id"] == station_id].itertuples():
            in_event = dt.between(event.start_utc, event.end_utc).to_numpy()
            detections += net_alarm[in_event].any(axis=0)

    if not observed_hours:
        raise ValueError("No valid observations are available for threshold calibration.")
    detection = detections / len(cat)
    eligible = np.flatnonzero(detection >= target)
    target_met = bool(len(eligible))
    if not target_met:
        eligible = np.flatnonzero(detection == detection.max())
    # Stable tie-breaking matches the research threshold sweep (ascending tau).
    best = eligible[np.argmin(episodes[eligible])]
    return {
        "threshold": float(grid[best]),
        "target_detection_pct": float(target * 100),
        "target_met": target_met,
        "validation_detection_pct": float(detection[best] * 100),
        "validation_false_alarms_per_station_year": float(episodes[best] / (observed_hours / HOURS_PER_YEAR)),
        "validation_observed_hours": observed_hours,
        "validation_events": int(len(cat)),
        "source": "validation_predictions_of_fitted_model",
    }


def calibrate_model_threshold(model, feature_cols: list[str], station_ids: list[str], data_dir: Path) -> dict:
    """Predict paired validation panels with the exact model being deployed."""
    fission_column = list(model.classes_).index(2)
    series = {}
    for station_id in station_ids:
        raw = pd.read_csv(data_dir / f"labeled_{station_id}_train.csv.gz")
        injected = compute_features_for_series(raw, station_id, use_injected=True)
        clean = compute_features_for_series(raw, station_id, use_injected=False)
        mask = injected["dt"].between(VALIDATION_START, TEST_START, inclusive="left")
        injected = injected.loc[mask].reset_index(drop=True)
        clean = clean.loc[mask].reset_index(drop=True)
        series[station_id] = pd.DataFrame({
            "dt": clean["dt"],
            "valid": clean["has_radnet_obs"] & clean["rad_complete_channels"] & clean["z_score_global_dry"].notna(),
            "p_clean": model.predict_proba(clean[feature_cols])[:, fission_column],
            "p_injected": model.predict_proba(injected[feature_cols])[:, fission_column],
        })
    catalog = pd.read_csv(data_dir / "synthetic_injection_catalog.csv")
    return select_operating_threshold(series, catalog)
