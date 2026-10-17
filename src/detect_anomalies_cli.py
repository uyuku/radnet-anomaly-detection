"""
Operational CLI & Diagnostic Utility for Weather-Aware RadNet Anomaly Detection.

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
    STATION_DRY_BASELINES,
    TIER_1_FEATURES,
    TIER_1B_FEATURES,
    TIER_2_FEATURES,
    TIER_3_FEATURES,
    compute_features_for_series,
)
from src.train_and_evaluate_models import STATIONS, load_and_prepare_train_data, train_tier_model

VALID_STATION_IDS = [s["id"] for s in STATIONS]
TIER_FEATURE_MAP = {
    "tier1": (TIER_1_FEATURES, "Tier 1 (Gross Only)"),
    "tier1b": (TIER_1B_FEATURES, "Tier 1b (Gross + Weather)"),
    "tier2": (TIER_2_FEATURES, "Tier 2 (Spectral)"),
    "tier3": (TIER_3_FEATURES, "Tier 3 (Weather Fusion)"),
}


def get_default_frozen_threshold(tier_key: str) -> float:
    """Loads the validation-frozen operating threshold (95% target) from CSV."""
    rig_path = Path("data/processed/rigorous_benchmark_summary.csv")
    tier_name_map = {
        "tier1": "Tier 1 (Gross)",
        "tier1b": "Tier 1b (Gross+Weather)",
        "tier2": "Tier 2 (Spectral)",
        "tier3": "Tier 3 (Weather Fusion)",
    }
    if rig_path.exists() and tier_key in tier_name_map:
        df = pd.read_csv(rig_path)
        sub = df[(df["tier_name"] == tier_name_map[tier_key]) & (df["target_detection_pct"] == 95.0)]
        if not sub.empty:
            return float(sub["frozen_tau_mean"].iloc[0])

    op_path = Path("data/processed/model_operating_points_summary.csv")
    col_map = {"tier1": "t1_threshold", "tier2": "t2_threshold", "tier3": "t3_threshold"}
    if op_path.exists() and tier_key in col_map:
        df = pd.read_csv(op_path)
        sub = df[df["target_detection_rate_pct"] == 95.0]
        if not sub.empty:
            return float(sub[col_map[tier_key]].iloc[0])
    return 0.988


def extract_alarm_episodes(df_eval: pd.DataFrame) -> list[dict]:
    """Groups contiguous hourly alarms into discrete operational alarm episodes."""
    episodes: list[dict] = []
    in_ep = False
    ep_rows: list[pd.Series] = []

    for _, row in df_eval.iterrows():
        if not bool(row["valid_obs"]):
            if in_ep and ep_rows:
                episodes.append(_summarize_episode(ep_rows))
                ep_rows = []
                in_ep = False
            continue
        if bool(row["is_alarm"]):
            in_ep = True
            ep_rows.append(row)
        else:
            if in_ep and ep_rows:
                episodes.append(_summarize_episode(ep_rows))
                ep_rows = []
                in_ep = False
    if in_ep and ep_rows:
        episodes.append(_summarize_episode(ep_rows))
    return episodes


def _summarize_episode(rows: list[pd.Series]) -> dict:
    ep_df = pd.DataFrame(rows)
    peak_idx = ep_df["p_fission"].idxmax()
    peak_row = ep_df.loc[peak_idx]

    # Identify dominant spectral channel share elevation vs typical background
    shares = {f"R0{i}": float(peak_row[f"share_r0{i}"]) for i in range(2, 10) if f"share_r0{i}" in peak_row}
    dom_channel = max(shares, key=shares.get) if shares else "N/A"

    return {
        "start_utc": str(ep_df["dt"].iloc[0]),
        "end_utc": str(ep_df["dt"].iloc[-1]),
        "duration_hours": int(len(ep_df)),
        "peak_p_fission": round(float(ep_df["p_fission"].max()), 4),
        "peak_p_washout": round(float(ep_df["p_washout"].max()), 4),
        "peak_gross_cpm": round(float(ep_df["eval_gross_cpm"].max()), 1),
        "peak_z_score_168h": round(float(ep_df["z_score_168h"].max()), 2),
        "total_precip_mm": round(float(ep_df["precip_1h_mm"].fillna(0.0).sum()), 2),
        "max_precip_1h_mm": round(float(ep_df["precip_1h_mm"].fillna(0.0).max()), 2),
        "dominant_channel": dom_channel,
        "ground_truth_labels": sorted(ep_df["label"].astype(str).unique().tolist()),
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
) -> dict:
    """Trains the specified model tier on 2017-2022 training data and evaluates a station window."""
    if station_id not in VALID_STATION_IDS:
        raise ValueError(f"Unknown station_id '{station_id}'. Valid stations: {VALID_STATION_IDS}")
    if tier not in TIER_FEATURE_MAP:
        raise ValueError(f"Unknown tier '{tier}'. Valid tiers: {list(TIER_FEATURE_MAP)}")

    feat_cols, tier_label = TIER_FEATURE_MAP[tier]
    tau = threshold if threshold is not None else get_default_frozen_threshold(tier)

    X_tr, y_tr = load_and_prepare_train_data(feat_cols)
    clf = train_tier_model(X_tr, y_tr, tier_label)

    df_tr_raw = pd.read_csv(f"data/processed/labeled_{station_id}_train.csv.gz")
    df_tr_raw["dt"] = pd.to_datetime(df_tr_raw["dt"])
    df_tr_raw = df_tr_raw.sort_values("dt").reset_index(drop=True)

    if split == "test":
        df_target = pd.read_csv(f"data/processed/labeled_{station_id}_test.csv.gz")
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

    valid_obs = (feats["has_radnet_obs"] & feats["rad_complete_channels"]).to_numpy()
    probs = clf.predict_proba(feats[feat_cols])

    df_eval = pd.DataFrame({
        "dt": df_target["dt"],
        "valid_obs": valid_obs,
        "label": df_target["label"],
        "eval_gross_cpm": df_target["inj_gross_cpm"] if use_injected else df_target["gross_cpm"],
        "clean_gross_cpm": df_target["gross_cpm"],
        "precip_1h_mm": df_target["precip_1h_mm"].fillna(0.0),
        "z_score_168h": feats["z_score_168h"],
        "p_normal": probs[:, 0],
        "p_washout": probs[:, 1],
        "p_fission": probs[:, 2],
    })
    for i in range(2, 10):
        df_eval[f"share_r0{i}"] = feats[f"share_r0{i}"]

    if start_utc is not None:
        df_eval = df_eval[df_eval["dt"] >= pd.to_datetime(start_utc)].copy()
    if end_utc is not None:
        df_eval = df_eval[df_eval["dt"] <= pd.to_datetime(end_utc)].copy()
    df_eval = df_eval.reset_index(drop=True)

    df_eval["is_alarm"] = (df_eval["p_fission"] >= tau) & df_eval["valid_obs"]

    st_base = STATION_DRY_BASELINES[station_id]
    base_3s = st_base["mu"] + 3.0 * st_base["sigma"]
    df_eval["is_baseline_3s_alarm"] = (df_eval["eval_gross_cpm"] >= base_3s) & df_eval["valid_obs"]

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
        "decision_threshold_tau": round(float(tau), 4),
        "window_start_utc": str(df_eval["dt"].iloc[0]) if not df_eval.empty else None,
        "window_end_utc": str(df_eval["dt"].iloc[-1]) if not df_eval.empty else None,
        "total_hours": int(len(df_eval)),
        "valid_observed_hours": n_obs,
        "observed_station_years": round(stn_years, 4),
        "ml_alarm_hours": int(df_eval["is_alarm"].sum()),
        "ml_alarm_episodes": int(len(episodes)),
        "ml_alarms_per_station_year": round(len(episodes) / stn_years, 2) if stn_years > 0 else 0.0,
        "rolling_3sigma_alarm_episodes": int(len(base_rolling_episodes)),
        "rolling_3sigma_alarms_per_station_year": round(len(base_rolling_episodes) / stn_years, 2) if stn_years > 0 else 0.0,
        "top_episodes": sorted(episodes, key=lambda x: x["peak_p_fission"], reverse=True)[:explain_top],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Operational CLI & Diagnostic Utility for Weather-Aware RadNet Anomaly Detection."
    )
    parser.add_argument("--station", default="al_birmingham", choices=VALID_STATION_IDS, help="Pilot station ID")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split to inspect")
    parser.add_argument("--tier", default="tier3", choices=list(TIER_FEATURE_MAP), help="Model feature tier")
    parser.add_argument("--threshold", type=float, default=None, help="Custom probability threshold tau (defaults to frozen validation threshold)")
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
    args = parser.parse_args()

    from contextlib import nullcontext, redirect_stdout

    ctx = redirect_stdout(sys.stderr) if args.json else nullcontext()
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
        )

    if args.json:
        print(json.dumps(res, indent=2))
        return

    print("=" * 80)
    print(f"RADNET WEATHER-AWARE ANOMALY DIAGNOSTICS — {res['station_id'].upper()}")
    print("=" * 80)
    print(f"Model Tier:            {res['tier_label']} (tau = {res['decision_threshold_tau']})")
    print(f"Series Mode:           {'Injected (Synthetic Plumes Active)' if res['use_injected'] else 'Clean Background Telemetry'}")
    print(f"Evaluation Window:     {res['window_start_utc']} to {res['window_end_utc']}")
    print(f"Valid Observed Hours:  {res['valid_observed_hours']:,} / {res['total_hours']:,} ({res['observed_station_years']:.2f} stn-yr)")
    print(f"ML Alarm Episodes:     {res['ml_alarm_episodes']} ({res['ml_alarms_per_station_year']:.2f} / stn-yr, {res['ml_alarm_hours']} alarm hrs)")
    print(f"Rolling 3σ Episodes:   {res['rolling_3sigma_alarm_episodes']} ({res['rolling_3sigma_alarms_per_station_year']:.2f} / stn-yr)")
    if res["top_episodes"]:
        print(f"\nTop {len(res['top_episodes'])} Alarm Episodes (by peak P(fission_product)):")
        for idx, ep in enumerate(res["top_episodes"], 1):
            print(
                f"  [{idx}] {ep['start_utc']} -> {ep['end_utc']} ({ep['duration_hours']}h) | "
                f"Peak P(fiss)={ep['peak_p_fission']:.3f}, P(wash)={ep['peak_p_washout']:.3f} | "
                f"Peak CPM={ep['peak_gross_cpm']:.1f} (z={ep['peak_z_score_168h']:+.2f}) | "
                f"Rain={ep['total_precip_mm']:.1f}mm | Labels={ep['ground_truth_labels']}"
            )
    else:
        print("\nNo alarm episodes triggered in the selected window.")


if __name__ == "__main__":
    main()
