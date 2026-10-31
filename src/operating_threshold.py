"""
Select operating thresholds for fitted detectors using 2021-2022 validation data only.

Supports:
  1. False-alarm-budget operating thresholds (`select_fa_budget_threshold`): selects tau
     directly from the clean-validation score distribution so that validation false-alarm
     episodes per station-year match a stated budget (e.g. 1, 2, 5, 10, 20 FA/yr),
     removing dependence on probability calibration and validation plumes.
  2. Detection-target operating thresholds (`select_operating_threshold`): minimizes
     validation false-alarm episodes subject to meeting a target event detection rate,
     using either a fixed grid, a fine high-probability grid (`FINE_THRESHOLD_GRID`),
     or a rank-based search over empirical validation scores (`build_rank_threshold_grid`).
  3. Paired vs. unpaired detection (`paired_detection=True` vs `False`): compares counting
     a plume as detected only when `(p_injected >= tau) & (p_clean < tau)` vs. `p_injected >= tau`.
  4. Cross-fitted validation probability calibration (`cross_fit_calibrate_validation_and_test`):
     fits Isotonic or Platt (sigmoid) calibration on 2021 and selects the threshold on 2022,
     then swaps (fit on 2022, select threshold on 2021), never fitting the calibrator and
     picking the threshold on the same rows.
  5. Event-level deadline and time-normalized detection metrics (`compute_event_detection_metrics`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.feature_engineering import compute_features_for_series

VALIDATION_START = pd.Timestamp("2021-01-01")
VALIDATION_MID = pd.Timestamp("2022-01-01")
TEST_START = pd.Timestamp("2023-01-01")
HOURS_PER_YEAR = 8766.0

THRESHOLD_GRID = np.concatenate([
    np.linspace(0.01, 0.95, 95),
    [0.96, 0.97, 0.98, 0.985, 0.990, 0.991, 0.992, 0.993, 0.994,
     0.995, 0.996, 0.997, 0.998, 0.999],
])

FINE_THRESHOLD_GRID = np.unique(np.concatenate([
    np.linspace(0.01, 0.90, 90),
    np.linspace(0.901, 0.989, 89),
    np.linspace(0.9900, 0.99995, 400),
]))


def build_rank_threshold_grid(
    series: dict[str, pd.DataFrame],
    n_quantiles: int = 1500,
    probability_bounds: bool = True,
) -> np.ndarray:
    """
    Constructs a rank-based threshold grid from the empirical clean and injected
    validation score distributions, concentrating resolution in the upper tail.
    """
    clean_vals = []
    inj_vals = []
    for frame in series.values():
        valid = frame["valid"].fillna(False).to_numpy(dtype=bool)
        pc = frame["p_clean"].to_numpy(dtype=float)[valid]
        clean_vals.append(pc[np.isfinite(pc)])
        if "p_injected" in frame.columns:
            pi = frame["p_injected"].to_numpy(dtype=float)[valid]
            inj_vals.append(pi[np.isfinite(pi)])
    all_cln = np.concatenate(clean_vals) if clean_vals else np.linspace(0.01, 0.99, 100)
    all_inj = np.concatenate(inj_vals) if inj_vals else all_cln

    q_cln_body = np.quantile(all_cln, np.linspace(0.01, 0.95, n_quantiles // 3))
    q_cln_tail = np.quantile(all_cln, np.linspace(0.95, 1.0, n_quantiles // 2))
    q_inj = np.quantile(all_inj, np.linspace(0.01, 0.999, n_quantiles // 6))
    # Also include exact sorted top-500 clean scores so every discrete episode step is addressable
    top_cln = np.sort(all_cln)[-min(500, len(all_cln)):]
    grid = np.unique(np.concatenate([q_cln_body, q_cln_tail, q_inj, top_cln]))
    if probability_bounds:
        grid = np.clip(grid, 0.0, 1.0)
    return np.sort(grid)


def select_operating_threshold(
    series: dict[str, pd.DataFrame],
    catalog: pd.DataFrame,
    target: float = 0.95,
    thresholds=None,
    paired_detection: bool = True,
) -> dict:
    """Minimize clean alarm episodes subject to event-level detection.

    Each frame contains dt, valid, p_injected and p_clean on a continuous hourly
    validation grid. When `paired_detection=True` (default), an event is detected
    only where injected alarms exceed the threshold while the paired clean series
    does not. When `paired_detection=False`, an event is detected whenever
    `p_injected >= threshold` during a valid event hour. Unreachable targets are
    reported explicitly, using best detection then fewest false alarms.
    """
    if isinstance(thresholds, str) and thresholds == "rank":
        grid = build_rank_threshold_grid(series, probability_bounds=True)
    elif isinstance(thresholds, str) and thresholds == "fine":
        grid = FINE_THRESHOLD_GRID.copy()
    else:
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
        if paired_detection:
            ev_alarm = (p_injected[:, None] >= grid) & ~alarm & valid[:, None]
        else:
            ev_alarm = (p_injected[:, None] >= grid) & valid[:, None]
        for event in cat.loc[cat["station_id"] == station_id].itertuples():
            in_event = dt.between(event.start_utc, event.end_utc).to_numpy()
            detections += ev_alarm[in_event].any(axis=0)

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


def select_fa_budget_threshold(
    series: dict[str, pd.DataFrame],
    fa_budget_per_year: float = 5.0,
    catalog: pd.DataFrame | None = None,
    thresholds=None,
    paired_detection: bool = True,
    probability_bounds: bool = True,
) -> dict[str, Any]:
    """
    Selects the operating threshold tau purely from the clean-validation score
    distribution so that validation false-alarm episodes per station-year are
    <= `fa_budget_per_year` and as close as possible to the budget (the lowest
    threshold tau whose clean validation FA/yr <= `fa_budget_per_year`).

    Because tau is determined solely by clean validation scores and the target
    false-alarm rate, it is invariant to monotone probability transformations
    and never peeks at validation plumes.
    """
    if fa_budget_per_year < 0 or not np.isfinite(fa_budget_per_year):
        raise ValueError("fa_budget_per_year must be a non-negative finite number.")

    if thresholds is None or (isinstance(thresholds, str) and thresholds == "rank"):
        grid = build_rank_threshold_grid(series, n_quantiles=1800, probability_bounds=probability_bounds)
    elif isinstance(thresholds, str) and thresholds == "fine":
        grid = FINE_THRESHOLD_GRID.copy()
    else:
        grid = np.unique(np.asarray(thresholds, dtype=float))

    episodes = np.zeros(len(grid), dtype=int)
    observed_hours = 0
    for _, frame in series.items():
        valid = frame["valid"].fillna(False).to_numpy(dtype=bool)
        p_clean = frame["p_clean"].to_numpy(dtype=float)
        alarm = (p_clean[:, None] >= grid[None, :]) & valid[:, None]
        prior = np.vstack([np.zeros((1, len(grid)), dtype=bool), alarm[:-1]])
        episodes += (alarm & ~prior).sum(axis=0)
        observed_hours += int(valid.sum())

    if observed_hours == 0:
        raise ValueError("No valid observations are available for FA-budget calibration.")

    val_years = observed_hours / HOURS_PER_YEAR
    fa_rates = episodes / val_years

    eligible = np.flatnonzero(fa_rates <= fa_budget_per_year + 1e-9)
    budget_met = bool(len(eligible) > 0)
    if budget_met:
        # Lowest threshold tau that satisfies clean validation FA/yr <= fa_budget_per_year
        best = int(eligible[0])
    else:
        best = int(np.argmin(fa_rates))

    tau = float(grid[best])
    val_det_pct = np.nan
    n_events = 0

    if catalog is not None:
        cat = catalog.copy()
        starts = pd.to_datetime(cat["start_utc"], utc=True).dt.tz_localize(None)
        ends = pd.to_datetime(cat["end_utc"], utc=True).dt.tz_localize(None)
        cat["start_utc"], cat["end_utc"] = starts, ends
        cat = cat.loc[(cat["split"] == "train") & (starts >= VALIDATION_START) & (ends < TEST_START)]
        n_events = int(len(cat))
        if n_events > 0:
            det_count = 0
            for station_id, frame in series.items():
                if "p_injected" not in frame.columns:
                    continue
                dt = pd.to_datetime(frame["dt"], utc=True).dt.tz_localize(None)
                valid = frame["valid"].fillna(False).to_numpy(dtype=bool)
                p_clean = frame["p_clean"].to_numpy(dtype=float)
                p_injected = frame["p_injected"].to_numpy(dtype=float)
                c_alarm = (p_clean >= tau) & valid
                ev_alarm = ((p_injected >= tau) & (~c_alarm) & valid) if paired_detection else ((p_injected >= tau) & valid)
                for event in cat.loc[cat["station_id"] == station_id].itertuples():
                    in_event = dt.between(event.start_utc, event.end_utc).to_numpy()
                    det_count += int(ev_alarm[in_event].any())
            val_det_pct = float(100.0 * det_count / n_events)

    return {
        "threshold": tau,
        "fa_budget_per_station_year": float(fa_budget_per_year),
        "budget_met": budget_met,
        "validation_false_alarms_per_station_year": float(fa_rates[best]),
        "validation_false_alarm_episodes": int(episodes[best]),
        "validation_detection_pct": val_det_pct,
        "validation_observed_hours": observed_hours,
        "validation_events": n_events,
        "paired_detection": paired_detection,
        "source": "clean_validation_fa_budget_quantile",
    }


def _fit_1d_calibrator(raw_scores: np.ndarray, y_binary: np.ndarray, method: str = "isotonic"):
    """Fits either an IsotonicRegression or Platt (sigmoid on logit) calibrator."""
    s = np.clip(np.asarray(raw_scores, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y_binary, dtype=int)
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(s, y)
        return lambda x: np.clip(iso.predict(np.clip(np.asarray(x, dtype=float), 1e-6, 1.0 - 1e-6)), 0.0, 1.0)
    elif method == "platt":
        logit = np.log(s / (1.0 - s)).reshape(-1, 1)
        lr = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
        lr.fit(logit, y)
        return lambda x: lr.predict_proba(
            np.log(np.clip(np.asarray(x, dtype=float), 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(np.asarray(x, dtype=float), 1e-6, 1.0 - 1e-6))).reshape(-1, 1)
        )[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def cross_fit_calibrate_validation_and_test(
    val_series: dict[str, pd.DataFrame],
    test_series: dict[str, pd.DataFrame],
    method: str = "isotonic",
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Performs two-fold temporal cross-fitting inside the 2021-2022 validation period:
      - Fold A: Fit calibrator on 2021 (`dt < 2022-01-01`), calibrate 2022 validation and 2023-2025 test.
      - Fold B: Fit calibrator on 2022 (`dt >= 2022-01-01`), calibrate 2021 validation and 2023-2025 test.
    Never fits the calibrator and selects the threshold on the same validation rows.

    Returns:
      (val_fold_2022_cal, val_fold_2021_cal, test_from_2021_cal, test_from_2022_cal)
    """
    scores_2021, labels_2021 = [], []
    scores_2022, labels_2022 = [], []

    for frame in val_series.values():
        dt = pd.to_datetime(frame["dt"], utc=True).dt.tz_localize(None)
        valid = frame["valid"].fillna(False).to_numpy(dtype=bool)
        is_2021 = (dt < VALIDATION_MID).to_numpy(dtype=bool) & valid
        is_2022 = (dt >= VALIDATION_MID).to_numpy(dtype=bool) & valid
        y_bin = frame["y_fission"].to_numpy(dtype=int) if "y_fission" in frame.columns else np.zeros(len(frame), dtype=int)
        p_inj = frame["p_injected"].to_numpy(dtype=float)

        scores_2021.append(p_inj[is_2021])
        labels_2021.append(y_bin[is_2021])
        scores_2022.append(p_inj[is_2022])
        labels_2022.append(y_bin[is_2022])

    cal_from_2021 = _fit_1d_calibrator(np.concatenate(scores_2021), np.concatenate(labels_2021), method=method)
    cal_from_2022 = _fit_1d_calibrator(np.concatenate(scores_2022), np.concatenate(labels_2022), method=method)

    val_2022_cal = {}
    val_2021_cal = {}
    for st_id, frame in val_series.items():
        dt = pd.to_datetime(frame["dt"], utc=True).dt.tz_localize(None)
        f_2022 = frame.loc[dt >= VALIDATION_MID].copy().reset_index(drop=True)
        f_2022["p_clean"] = cal_from_2021(f_2022["p_clean"].to_numpy(dtype=float))
        f_2022["p_injected"] = cal_from_2021(f_2022["p_injected"].to_numpy(dtype=float))
        val_2022_cal[st_id] = f_2022

        f_2021 = frame.loc[dt < VALIDATION_MID].copy().reset_index(drop=True)
        f_2021["p_clean"] = cal_from_2022(f_2021["p_clean"].to_numpy(dtype=float))
        f_2021["p_injected"] = cal_from_2022(f_2021["p_injected"].to_numpy(dtype=float))
        val_2021_cal[st_id] = f_2021

    test_from_2021 = {}
    test_from_2022 = {}
    for st_id, frame in test_series.items():
        t1 = frame.copy()
        t1["p_clean"] = cal_from_2021(t1["p_clean"].to_numpy(dtype=float))
        t1["p_injected"] = cal_from_2021(t1["p_injected"].to_numpy(dtype=float))
        test_from_2021[st_id] = t1

        t2 = frame.copy()
        t2["p_clean"] = cal_from_2022(t2["p_clean"].to_numpy(dtype=float))
        t2["p_injected"] = cal_from_2022(t2["p_injected"].to_numpy(dtype=float))
        test_from_2022[st_id] = t2

    return val_2022_cal, val_2021_cal, test_from_2021, test_from_2022


def compute_event_detection_metrics(
    plume_records: pd.DataFrame,
    reference_window_hours: float = 12.0,
) -> dict[str, float]:
    """
    Computes comprehensive event-level detection metrics from per-plume outcomes:
      - fraction of plumes detected within 3, 6, 12, and 24 hours, plus any-hour total detection
      - delay distribution quantiles (25th, 50th/median, 75th, 90th percentiles and mean)
      - active-hour recall across plumes (`mean_active_hour_recall_pct`)
      - time-normalized detection probability per `reference_window_hours` (12h) of exposure:
        `1 - (1 - n_alarm_hours / duration_hours) ** (reference_window_hours / duration_hours_unit)`,
        which removes the duration inflation whereby 100-hour plumes have ~10x more opportunities
        to trigger an any-hour alarm than 10-hour plumes.
    """
    if plume_records.empty:
        return {
            "det_total_pct": 0.0,
            "det_le_3h_pct": 0.0,
            "det_le_6h_pct": 0.0,
            "det_le_12h_pct": 0.0,
            "det_le_24h_pct": 0.0,
            "time_normalized_12h_det_pct": 0.0,
            "mean_active_hour_recall_pct": 0.0,
            "delay_p25_hours": np.nan,
            "delay_median_hours": np.nan,
            "delay_p75_hours": np.nan,
            "delay_p90_hours": np.nan,
            "delay_mean_hours": np.nan,
        }

    n = len(plume_records)
    det = plume_records["detected"].to_numpy(dtype=bool)
    delays = plume_records["delay_hours"].to_numpy(dtype=float)
    valid_delays = delays[det & np.isfinite(delays)]

    det_3h = np.sum(det & (delays <= 3.0)) / n * 100.0
    det_6h = np.sum(det & (delays <= 6.0)) / n * 100.0
    det_12h = np.sum(det & (delays <= 12.0)) / n * 100.0
    det_24h = np.sum(det & (delays <= 24.0)) / n * 100.0
    det_tot = np.sum(det) / n * 100.0

    if "alarm_hours_in_plume" in plume_records.columns and "duration_hours" in plume_records.columns:
        al_h = np.maximum(0.0, plume_records["alarm_hours_in_plume"].to_numpy(dtype=float))
        dur_h = np.maximum(1.0, plume_records["duration_hours"].to_numpy(dtype=float))
        p_hour = np.clip(al_h / dur_h, 0.0, 1.0)
        # Probability of >=1 alarm in a standardized `reference_window_hours` window
        p_norm = 1.0 - np.power(1.0 - p_hour, reference_window_hours)
        time_norm_pct = float(np.mean(p_norm) * 100.0)
        active_recall_pct = float(np.mean(p_hour) * 100.0)
    else:
        time_norm_pct = float(det_12h)
        active_recall_pct = np.nan

    if len(valid_delays) > 0:
        p25, p50, p75, p90 = np.percentile(valid_delays, [25, 50, 75, 90])
        d_mean = float(np.mean(valid_delays))
    else:
        p25 = p50 = p75 = p90 = d_mean = np.nan

    return {
        "det_total_pct": round(float(det_tot), 2),
        "det_le_3h_pct": round(float(det_3h), 2),
        "det_le_6h_pct": round(float(det_6h), 2),
        "det_le_12h_pct": round(float(det_12h), 2),
        "det_le_24h_pct": round(float(det_24h), 2),
        "time_normalized_12h_det_pct": round(float(time_norm_pct), 2),
        "mean_active_hour_recall_pct": round(float(active_recall_pct), 2) if np.isfinite(active_recall_pct) else np.nan,
        "delay_p25_hours": round(float(p25), 2) if np.isfinite(p25) else np.nan,
        "delay_median_hours": round(float(p50), 2) if np.isfinite(p50) else np.nan,
        "delay_p75_hours": round(float(p75), 2) if np.isfinite(p75) else np.nan,
        "delay_p90_hours": round(float(p90), 2) if np.isfinite(p90) else np.nan,
        "delay_mean_hours": round(float(d_mean), 2) if np.isfinite(d_mean) else np.nan,
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
