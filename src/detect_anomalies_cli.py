"""
Research CLI diagnostics for Weather-Aware RadNet Anomaly Detection.

Provides command-line inference, episode-level alarm summarization, and physical
attribution (radiation excess, NaI(Tl) spectral channel shares, and NOAA weather
context) on station telemetry.

Usage examples:
  # Inspect Birmingham test split around the Jan 2023 storm + Cs-137 hard case (injected)
  python src/detect_anomalies_cli.py --station al_birmingham --start 2023-01-24 --end 2023-01-30 --use-injected

  # Audit clean operational false-alarm episodes on Washington DC (2023-2025)
  python src/detect_anomalies_cli.py --station dc_washington --split test

  # Output machine-readable JSON summary
  python src/detect_anomalies_cli.py --station al_birmingham --start 2023-01-24 --end 2023-01-30 --use-injected --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.feature_engineering import (
    FEATURE_SCHEMA_VERSION,
    TIER_1_FEATURES,
    TIER_1B_FEATURES,
    TIER_2_FEATURES,
    TIER_3_FEATURES,
    compute_features_for_series,
)
from src.operating_threshold import calibrate_model_threshold
from src.train_and_evaluate_models import STATIONS, load_and_prepare_train_data, train_tier_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_STATION_IDS = [s["id"] for s in STATIONS]
TIER_FEATURE_MAP = {
    "tier1": (TIER_1_FEATURES, "Tier 1 (Gross Only)"),
    "tier1b": (TIER_1B_FEATURES, "Tier 1b (Gross + Weather)"),
    "tier2": (TIER_2_FEATURES, "Tier 2 (Spectral)"),
    "tier3": (TIER_3_FEATURES, "Tier 3 (Weather Fusion)"),
}


def extract_alarm_episodes(df_eval: pd.DataFrame) -> list[dict]:
    """Group hourly alarms; missing hours and detector outages break episodes."""
    if df_eval.empty:
        return []
    frame = df_eval.copy().reset_index(drop=True)
    frame["dt"] = pd.to_datetime(frame["dt"], utc=True).dt.tz_localize(None)
    if frame["dt"].isna().any() or frame["dt"].duplicated().any() or not frame["dt"].is_monotonic_increasing:
        raise ValueError("Alarm timestamps must be non-missing, sorted and unique.")
    alarm = frame["is_alarm"].fillna(False) & frame["valid_obs"].fillna(False)
    contiguous = frame["dt"].diff().eq(pd.Timedelta(hours=1))
    starts = alarm & (~alarm.shift(1, fill_value=False) | ~contiguous)
    return [_summarize_episode(group) for _, group in frame.loc[alarm].groupby(starts.cumsum()[alarm])]


def _finite_round(value, digits=2):
    return round(float(value), digits) if pd.notna(value) and np.isfinite(value) else None


def _summarize_episode(ep_df: pd.DataFrame) -> dict:
    peak_idx = ep_df["p_fission"].idxmax()
    peak_row = ep_df.loc[peak_idx]

    # Largest observed channel share; this is not a radionuclide attribution.
    shares = {f"R0{i}": float(peak_row[f"share_r0{i}"]) for i in range(2, 10)
              if f"share_r0{i}" in peak_row and pd.notna(peak_row[f"share_r0{i}"])}
    dom_channel = max(shares, key=shares.get) if shares else "N/A"

    return {
        "start_utc": str(ep_df["dt"].iloc[0]),
        "end_utc": str(ep_df["dt"].iloc[-1]),
        "duration_hours": int(len(ep_df)),
        "peak_p_fission": round(float(ep_df["p_fission"].max()), 4),
        "peak_p_washout": round(float(ep_df["p_washout"].max()), 4),
        "peak_gross_cpm": round(float(ep_df["eval_gross_cpm"].max()), 1),
        "peak_z_score_168h": _finite_round(ep_df["z_score_168h"].max()),
        "total_precip_mm": _finite_round(ep_df["precip_1h_mm"].sum(min_count=1)),
        "max_precip_1h_mm": _finite_round(ep_df["precip_1h_mm"].max()),
        "precip_observed_hours": int(ep_df["precip_1h_mm"].notna().sum()),
        "precip_missing_hours": int(ep_df["precip_1h_mm"].isna().sum()),
        "dominant_channel": dom_channel,
        "ground_truth_labels": sorted(ep_df["label"].dropna().astype(str).unique().tolist()),
    }


def run_station_diagnostics(
    station_id: str,
    split: str = "test",
    tier: str = "tier3",
    threshold: float | None = None,
    use_injected: bool = False,
    start_utc: str | None = None,
    end_utc: str | None = None,
    explain_top: int = 5,
    data_dir: Path | None = None,
) -> dict:
    """Fit on 2017-2020, calibrate on 2021-2022, then inspect a station window."""
    if station_id not in VALID_STATION_IDS:
        raise ValueError(f"Unknown station_id '{station_id}'. Valid stations: {VALID_STATION_IDS}")
    if tier not in TIER_FEATURE_MAP:
        raise ValueError(f"Unknown tier '{tier}'. Valid tiers: {list(TIER_FEATURE_MAP)}")
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold must be a finite probability in [0, 1].")
    if not isinstance(explain_top, int) or explain_top < 0:
        raise ValueError("explain_top must be a non-negative integer.")
    start = pd.to_datetime(start_utc, utc=True).tz_localize(None) if start_utc is not None else None
    end = pd.to_datetime(end_utc, utc=True).tz_localize(None) if end_utc is not None else None
    if (start is not None and pd.isna(start)) or (end is not None and pd.isna(end)):
        raise ValueError("Window timestamps must not be NaT.")
    if start is not None and end is not None and start > end:
        raise ValueError("start must be at or before end.")
    data_dir = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "data" / "processed"
    required = [data_dir / f"labeled_{sid}_train.csv.gz" for sid in VALID_STATION_IDS]
    required.append(data_dir / f"labeled_{station_id}_{split}.csv.gz")
    if threshold is None:
        required.append(data_dir / "synthetic_injection_catalog.csv")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing diagnostic inputs; run pipeline phases 0-3 first:\n" + "\n".join(missing))

    feat_cols, tier_label = TIER_FEATURE_MAP[tier]

    df_tr_raw = pd.read_csv(data_dir / f"labeled_{station_id}_train.csv.gz")
    df_tr_raw["dt"] = pd.to_datetime(df_tr_raw["dt"])
    df_tr_raw = df_tr_raw.sort_values("dt").reset_index(drop=True)

    if split == "test":
        df_target = pd.read_csv(data_dir / f"labeled_{station_id}_test.csv.gz")
        df_target["dt"] = pd.to_datetime(df_target["dt"])
        df_target = df_target.sort_values("dt").reset_index(drop=True)

        df_warm = df_tr_raw.iloc[-250:].copy().reset_index(drop=True)
        df_warm["inj_gross_cpm"] = df_warm["gross_cpm"]
        df_warm["inj_dose_rate_nsvh"] = df_warm["dose_rate_nsvh"]
        for ch_idx in range(2, 10):
            df_warm[f"inj_cpm_r0{ch_idx}"] = df_warm[f"cpm_r0{ch_idx}"]
        df_panel = pd.concat([df_warm, df_target], ignore_index=True)
        feats = compute_features_for_series(df_panel, station_id, use_injected=use_injected).iloc[len(df_warm):].reset_index(drop=True)
    else:
        df_target = df_tr_raw
        feats = compute_features_for_series(df_target, station_id, use_injected=use_injected)

    window = pd.Series(True, index=feats.index)
    if start is not None:
        window &= feats["dt"].ge(start)
    if end is not None:
        window &= feats["dt"].le(end)
    if not window.any():
        raise ValueError("The requested window contains no data rows.")

    X_tr, y_tr = load_and_prepare_train_data(feat_cols, data_dir=data_dir)
    clf = train_tier_model(X_tr, y_tr, tier_label)
    calibration = (
        calibrate_model_threshold(clf, feat_cols, VALID_STATION_IDS, data_dir)
        if threshold is None else {"source": "user_override", "threshold": float(threshold)}
    )
    tau = calibration["threshold"]
    feats = feats.loc[window].reset_index(drop=True)
    df_target = df_target.loc[window].reset_index(drop=True)
    gross_col = "inj_gross_cpm" if use_injected else "gross_cpm"
    valid_obs = (feats["has_radnet_obs"] & feats["rad_complete_channels"] & df_target[gross_col].notna()).to_numpy()
    probs = np.full((len(feats), 3), np.nan)
    if valid_obs.any():
        predicted = clf.predict_proba(feats.loc[valid_obs, feat_cols])
        for column, label in enumerate(clf.classes_):
            probs[valid_obs, int(label)] = predicted[:, column]

    df_eval = pd.DataFrame({
        "dt": df_target["dt"],
        "valid_obs": valid_obs,
        "label": df_target["label"] if use_injected else pd.Series(pd.NA, index=df_target.index),
        "eval_gross_cpm": df_target["inj_gross_cpm"] if use_injected else df_target["gross_cpm"],
        "clean_gross_cpm": df_target["gross_cpm"],
        "precip_1h_mm": df_target["precip_1h_mm"],
        "z_score_168h": feats["z_score_168h"],
        "p_normal": probs[:, 0],
        "p_washout": probs[:, 1],
        "p_fission": probs[:, 2],
    })
    for i in range(2, 10):
        df_eval[f"share_r0{i}"] = feats[f"share_r0{i}"]

    df_eval["is_alarm"] = (df_eval["p_fission"] >= tau) & df_eval["valid_obs"]

    episodes = extract_alarm_episodes(df_eval)
    n_obs = int(df_eval["valid_obs"].sum())
    stn_years = n_obs / 8766.0

    # Baseline 3-sigma episodes for direct comparison
    df_base_tmp = df_eval.copy()
    df_base_tmp["is_alarm"] = df_eval["z_score_168h"].ge(3.0) & df_eval["valid_obs"]
    base_rolling_episodes = extract_alarm_episodes(df_base_tmp)

    return {
        "station_id": station_id,
        "split": split,
        "tier": tier,
        "tier_label": tier_label,
        "use_injected": bool(use_injected),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_seed": 42,
        "model_training_period": "2017-2020",
        "label_source": "synthetic_catalog_and_washout_heuristic" if use_injected else "unlabeled_background",
        "decision_threshold_tau": float(tau),
        "calibration": calibration,
        "window_start_utc": str(df_eval["dt"].iloc[0]) if not df_eval.empty else None,
        "window_end_utc": str(df_eval["dt"].iloc[-1]) if not df_eval.empty else None,
        "total_hours": int(len(df_eval)),
        "valid_observed_hours": n_obs,
        "observed_station_years": round(stn_years, 4),
        "ml_alarm_hours": int(df_eval["is_alarm"].sum()),
        "ml_alarm_episodes": int(len(episodes)),
        "ml_alarms_per_station_year": round(len(episodes) / stn_years, 2) if stn_years > 0 else None,
        "rolling_3sigma_alarm_episodes": int(len(base_rolling_episodes)),
        "rolling_3sigma_alarms_per_station_year": round(len(base_rolling_episodes) / stn_years, 2) if stn_years > 0 else None,
        "top_episodes": sorted(episodes, key=lambda x: x["peak_p_fission"], reverse=True)[:explain_top],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Research diagnostics with a threshold calibrated for the fitted RadNet model."
    )
    parser.add_argument("--station", default="al_birmingham", choices=VALID_STATION_IDS, help="Pilot station ID")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split to inspect")
    parser.add_argument("--tier", default="tier3", choices=list(TIER_FEATURE_MAP), help="Model feature tier")
    parser.add_argument("--threshold", type=float, default=None, help="Probability in [0, 1]; defaults to calibration on 2021-2022 for this fitted model")
    parser.add_argument(
        "--use-injected",
        "--injected",
        dest="use_injected",
        action="store_true",
        help="Evaluate on synthetic-plume-injected series instead of clean background",
    )
    parser.add_argument("--start", default=None, help="Optional UTC start timestamp (e.g. 2023-01-24)")
    parser.add_argument("--end", default=None, help="Optional UTC end timestamp (e.g. 2023-01-30)")
    parser.add_argument("--explain-top", type=int, default=5, help="Number of top alarm episodes to display")
    parser.add_argument("--json", action="store_true", help="Emit JSON output only")
    parser.add_argument("--data-dir", type=Path, default=None, help="Processed data directory (defaults to this repository's data/processed)")
    args = parser.parse_args()

    from contextlib import nullcontext, redirect_stdout

    ctx = redirect_stdout(sys.stderr) if args.json else nullcontext()
    try:
        with ctx:
            res = run_station_diagnostics(
                station_id=args.station,
                split=args.split,
                tier=args.tier,
                threshold=args.threshold,
                use_injected=args.use_injected,
                start_utc=args.start,
                end_utc=args.end,
                explain_top=args.explain_top,
                data_dir=args.data_dir,
            )
    except (ValueError, FileNotFoundError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, allow_nan=False))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(res, indent=2, allow_nan=False))
        return 0

    def fmt(value):
        return "unknown" if value is None else f"{value:.2f}"

    print("=" * 80)
    print(f"RADNET WEATHER-AWARE ANOMALY DIAGNOSTICS — {res['station_id'].upper()}")
    print("=" * 80)
    print(f"Model Tier:            {res['tier_label']} (tau = {res['decision_threshold_tau']})")
    print(f"Series Mode:           {'Injected (Synthetic Plumes Active)' if res['use_injected'] else 'Clean Background Telemetry'}")
    print(f"Evaluation Window:     {res['window_start_utc']} to {res['window_end_utc']}")
    print(f"Valid Observed Hours:  {res['valid_observed_hours']:,} / {res['total_hours']:,} ({res['observed_station_years']:.2f} stn-yr)")
    print(f"ML Alarm Episodes:     {res['ml_alarm_episodes']} ({fmt(res['ml_alarms_per_station_year'])} / stn-yr, {res['ml_alarm_hours']} alarm hrs)")
    print(f"Rolling 3-sigma:       {res['rolling_3sigma_alarm_episodes']} episodes ({fmt(res['rolling_3sigma_alarms_per_station_year'])} / stn-yr)")
    print(f"Threshold source:      {res['calibration']['source']}")
    if "target_met" in res["calibration"]:
        print(f"Validation detection:  {res['calibration']['validation_detection_pct']:.2f}% (95% target met: {res['calibration']['target_met']})")
    if res["top_episodes"]:
        print(f"\nTop {len(res['top_episodes'])} Alarm Episodes (by peak P(fission_product)):")
        for idx, ep in enumerate(res["top_episodes"], 1):
            print(
                f"  [{idx}] {ep['start_utc']} -> {ep['end_utc']} ({ep['duration_hours']}h) | "
                f"Peak P(fiss)={ep['peak_p_fission']:.3f}, P(wash)={ep['peak_p_washout']:.3f} | "
                f"Peak CPM={ep['peak_gross_cpm']:.1f} (z={fmt(ep['peak_z_score_168h'])}) | "
                f"Observed rain={fmt(ep['total_precip_mm'])}mm ({ep['precip_missing_hours']} missing hours) | Labels={ep['ground_truth_labels']}"
            )
    else:
        print("\nNo alarm episodes triggered in the selected window.")


if __name__ == "__main__":
    raise SystemExit(main())
