"""
Real-Event Validation on Fukushima Daiichi Trans-Pacific Plume (2011).

Executes the pre-registered protocol in docs/real_event_protocol.md:
- Prepares continuous hourly 2011 dataset for Honolulu, San Diego, Anchorage, and Dallas.
- Merges RadNet 8-channel gamma data and NOAA ISD hourly weather.
- Sets pre-event local baselines causally from Jan 1 to Mar 10, 2011.
- Evaluates frozen LightGBM models (Tier 1-4) and non-ML baselines (B1, B2, B3, B4c).
- Reports primary and secondary metrics across Pre-Event, Active Event, and Post-Event windows.
- Generates publication figures and summary tables.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from src.feature_engineering import (
    STATION_DRY_BASELINES,
    TIER_1_FEATURES,
    TIER_2_FEATURES,
    TIER_3_FEATURES,
    TIER_4_NUCLEAR_FEATURES,
    compute_features_for_series,
)
from src.merge_radnet_weather import (
    CHANNEL_KEYS,
    calculate_relative_humidity,
    parse_noaa_precip,
    parse_noaa_value,
)
from src.train_and_evaluate_rigorous import load_dataset_folds, train_lgb

EVENT_STATIONS = [
    {
        "id": "hi_honolulu",
        "name": "Honolulu, HI",
        "noaa_call": "PHNL",
        "noaa_id": "91182022521",
        "radnet_zip": "data/raw/hi_honolulu_2025-2008.zip",
        "city_prefix": "HI_HONOLULU",
        "is_pacific": True,
    },
    {
        "id": "ca_san_diego",
        "name": "San Diego, CA",
        "noaa_call": "KSAN",
        "noaa_id": "72290023188",
        "radnet_zip": "data/raw/ca_san_diego_2025-2006.zip",
        "city_prefix": "CA_SAN_DIEGO",
        "is_pacific": True,
    },
    {
        "id": "ak_anchorage",
        "name": "Anchorage, AK",
        "noaa_call": "PANC",
        "noaa_id": "70273026451",
        "radnet_zip": "data/raw/ak_anchorage_2025-2008.zip",
        "city_prefix": "AK_ANCHORAGE",
        "is_pacific": True,
    },
    {
        "id": "tx_dallas",
        "name": "Dallas, TX",
        "noaa_call": "KDFW",
        "noaa_id": "72259003927",
        "radnet_zip": "data/raw/tx_dallas_2025-2007.zip",
        "city_prefix": "TX_DALLAS",
        "is_pacific": False,
    },
]

# Pre-registered temporal windows (UTC)
WINDOW_PRE = ("2011-01-01 00:00:00", "2011-03-11 05:46:00")
WINDOW_EVENT = ("2011-03-11 05:47:00", "2011-04-30 23:59:00")
WINDOW_POST = ("2011-05-01 00:00:00", "2011-06-30 23:59:00")

# Frozen operating thresholds (from validation calibration, 5 FA/yr budget operating points)
FROZEN_THRESHOLDS = {
    "Tier 4 (Physics-Motivated Nuclear)": 0.8719,
    "Tier 3 (Weather Fusion)": 0.8726,
    "Tier 2 (Spectral)": 0.9475,
    "Tier 1 (Gross)": 0.9238,
    "Baseline: Global Dry 3-sigma": 3.0,
    "Baseline: Rolling 7d 3-sigma": 3.0,
    "Baseline: Rolling 7d 4-sigma": 4.0,
    "Baseline: Radon Projection 3-sigma": 3.0,
}


def _download_noaa_2011(station: dict) -> Path:
    target_path = Path(f"data/raw/noaa_{station['noaa_call'].lower()}_2011.csv")
    if target_path.exists() and target_path.stat().st_size > 1000:
        return target_path

    url = f"https://noaa-global-hourly-pds.s3.amazonaws.com/2011/{station['noaa_id']}.csv"
    print(f"Downloading NOAA 2011 weather for {station['name']} from {url}...")
    headers = {"User-Agent": "RadNet-Research/1.0"}
    resp = requests.get(url, headers=headers, stream=True, timeout=60)
    resp.raise_for_status()

    tmp = target_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
    tmp.rename(target_path)
    os.chmod(target_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(f"  Saved {target_path} ({target_path.stat().st_size / (1024*1024):.2f} MB)")
    return target_path


def load_radnet_2011(station: dict) -> pd.DataFrame:
    zip_path = Path(station["radnet_zip"])
    with zipfile.ZipFile(zip_path, "r") as z:
        matched = [n for n in z.namelist() if "2011" in n]
        if not matched:
            raise ValueError(f"No 2011 file in {zip_path}")
        with z.open(matched[0]) as f:
            df = pd.read_csv(f, low_memory=False)

    df["dt_raw"] = pd.to_datetime(df["SAMPLE COLLECTION TIME"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["dt_raw"])
    df["utc_hour"] = df["dt_raw"].dt.round("h")
    df = df.drop_duplicates(subset=["utc_hour"]).copy()

    chan_cols_map = {k: [c for c in df.columns if k in c][0] for k in CHANNEL_KEYS}
    chan_cols = [chan_cols_map[k] for k in CHANNEL_KEYS]
    for c in chan_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["rad_complete_channels"] = df[chan_cols].notna().all(axis=1)
    df["gross_cpm"] = np.nan
    df.loc[df["rad_complete_channels"], "gross_cpm"] = df.loc[df["rad_complete_channels"], chan_cols].sum(axis=1)
    df["dose_rate_nsvh"] = np.nan  # Absent in 2011

    rename_dict = {chan_cols_map[k]: f"cpm_{k.lower()}" for k in CHANNEL_KEYS}
    df = df.rename(columns=rename_dict)
    keep_cols = [
        "utc_hour",
        "rad_complete_channels",
        "gross_cpm",
        "dose_rate_nsvh",
        "cpm_r02",
        "cpm_r03",
        "cpm_r04",
        "cpm_r05",
        "cpm_r06",
        "cpm_r07",
        "cpm_r08",
        "cpm_r09",
    ]
    return df[keep_cols].sort_values("utc_hour").reset_index(drop=True)


def load_noaa_2011(station: dict) -> pd.DataFrame:
    file_path = _download_noaa_2011(station)
    cols = ["DATE", "TMP", "DEW", "SLP", "AA1"]
    df = pd.read_csv(file_path, usecols=lambda c: c in cols, low_memory=False)
    df["dt_raw"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["dt_raw"])
    df["utc_hour"] = df["dt_raw"].dt.round("h")

    df["precip_1h_mm"] = df["AA1"].apply(parse_noaa_precip)
    df["pressure_hpa"] = df["SLP"].apply(lambda v: parse_noaa_value(v, missing_val="99999", divisor=10.0))
    df["temp_c"] = df["TMP"].apply(lambda v: parse_noaa_value(v, missing_val="+9999", divisor=10.0))
    df["dewpoint_c"] = df["DEW"].apply(lambda v: parse_noaa_value(v, missing_val="+9999", divisor=10.0))
    df["rel_humidity_pct"] = [
        calculate_relative_humidity(t, d) for t, d in zip(df["temp_c"], df["dewpoint_c"])
    ]

    hourly = df.groupby("utc_hour").agg(
        {
            "precip_1h_mm": "max",
            "pressure_hpa": "mean",
            "temp_c": "mean",
            "dewpoint_c": "mean",
            "rel_humidity_pct": "mean",
        }
    ).reset_index()

    hourly["pressure_hpa"] = hourly["pressure_hpa"].round(1)
    hourly["temp_c"] = hourly["temp_c"].round(1)
    hourly["dewpoint_c"] = hourly["dewpoint_c"].round(1)
    hourly["rel_humidity_pct"] = hourly["rel_humidity_pct"].round(1)
    return hourly


def prepare_merged_2011(station: dict) -> pd.DataFrame:
    df_rad = load_radnet_2011(station)
    df_weather = load_noaa_2011(station)

    grid = pd.date_range("2011-01-01 00:00:00", "2011-06-30 23:00:00", freq="h", name="utc_hour")
    df_grid = pd.DataFrame(index=grid).reset_index()

    merged = pd.merge(df_grid, df_rad, on="utc_hour", how="left")
    merged = pd.merge(merged, df_weather, on="utc_hour", how="left")

    merged["station_id"] = station["id"]
    merged["station_name"] = station["name"]
    merged["has_radnet_obs"] = merged["gross_cpm"].notna()
    merged["dt"] = merged["utc_hour"]
    merged["label"] = "clean_dry"
    merged["injection_active"] = False
    merged["synthetic_excess_cpm"] = 0.0
    merged["rad_complete_channels"] = merged["rad_complete_channels"].fillna(False)

    # Fill weather gaps forward/backward
    for col in ["pressure_hpa", "temp_c", "dewpoint_c", "rel_humidity_pct"]:
        merged[col] = merged[col].ffill().bfill()
    merged["precip_1h_mm"] = merged["precip_1h_mm"].fillna(0.0)

    # Impute missing channels with linear interpolation for causal rolling windows
    for k in CHANNEL_KEYS:
        c = f"cpm_{k.lower()}"
        merged[c] = merged[c].interpolate(method="linear", limit=6).ffill().bfill()
    merged["gross_cpm"] = merged[[f"cpm_{k.lower()}" for k in CHANNEL_KEYS]].sum(axis=1)

    return merged


def compute_pre_event_baseline(df_merged: pd.DataFrame) -> dict:
    pre_mask = (df_merged["dt"] >= WINDOW_PRE[0]) & (df_merged["dt"] <= "2011-03-10 23:00:00")
    dry_mask = pre_mask & (df_merged["precip_1h_mm"] == 0.0) & df_merged["has_radnet_obs"]
    if dry_mask.sum() < 48:
        dry_mask = pre_mask & df_merged["has_radnet_obs"]

    df_dry = df_merged[dry_mask]
    mu_dry = float(df_dry["gross_cpm"].mean())
    sigma_dry = float(df_dry["gross_cpm"].std())

    chan_mu = [float(df_dry[f"cpm_{k.lower()}"].mean()) for k in CHANNEL_KEYS]
    chan_sigma = [max(1.0, float(df_dry[f"cpm_{k.lower()}"].std())) for k in CHANNEL_KEYS]

    return {
        "mu": mu_dry,
        "sigma": max(1.0, sigma_dry),
        "mu_dose": 0.0,
        "sigma_dose": 2.5,
        "radon_washout_basis": [0.48, 0.22, 0.12, 0.08, 0.04, 0.03, 0.02, 0.01],
        "channel_mu_dry": chan_mu,
        "channel_sigma_dry": chan_sigma,
        "k_dose_slope": 0.01344,
    }


def main():
    print("================================================================================")
    print("REAL-EVENT VALIDATION — FUKUSHIMA DAIICHI PLUME (2011)")
    print("================================================================================")

    # 1. Train Canonical Frozen Models (2017-2020 train fold, seed 42)
    print("\n--- Training Frozen Models on 2017-2020 Training Fold (Seed 42) ---")
    tiers = {
        "Tier 1 (Gross)": TIER_1_FEATURES,
        "Tier 2 (Spectral)": TIER_2_FEATURES,
        "Tier 3 (Weather Fusion)": TIER_3_FEATURES,
        "Tier 4 (Physics-Motivated Nuclear)": TIER_4_NUCLEAR_FEATURES,
    }

    models = {}
    for tname, fcols in tiers.items():
        print(f"Training frozen model for {tname} ({len(fcols)} features)...")
        X_tr, y_tr, _, _ = load_dataset_folds(fcols)
        models[tname] = train_lgb(X_tr, y_tr, seed=42)

    all_station_evals = []
    summary_rows = []

    # 2. Process Each Station
    for st in EVENT_STATIONS:
        st_id = st["id"]
        st_name = st["name"]
        print(f"\n=== Evaluating Station {st_name} ({st_id}) on 2011 Data ===")

        merged = prepare_merged_2011(st)
        pre_base = compute_pre_event_baseline(merged)
        STATION_DRY_BASELINES[st_id] = pre_base

        feats = compute_features_for_series(merged, st_id, use_injected=False)

        # Baseline Detectors
        # B1: Global Dry 3-sigma
        z_global = (merged["gross_cpm"] - pre_base["mu"]) / pre_base["sigma"]
        # B2: Rolling 7d 3-sigma
        roll_mean = merged["gross_cpm"].rolling(168, min_periods=24).mean()
        roll_std = merged["gross_cpm"].rolling(168, min_periods=24).std().clip(lower=1.0)
        z_roll = (merged["gross_cpm"] - roll_mean) / roll_std

        # B4c: Radon Projection Residual
        s_rn = np.array([0.48, 0.22, 0.12, 0.08, 0.04, 0.03, 0.02, 0.01])
        s_rn = s_rn / np.sum(s_rn)
        chan_data = merged[[f"cpm_{k.lower()}" for k in CHANNEL_KEYS]].values
        proj = np.dot(chan_data, s_rn)[:, None] * s_rn[None, :]
        res = chan_data - proj
        res_l2 = np.linalg.norm(res, axis=1)
        res_roll_mean = pd.Series(res_l2).rolling(168, min_periods=24).mean()
        res_roll_std = pd.Series(res_l2).rolling(168, min_periods=24).std().clip(lower=0.1)
        z_radon_proj = (pd.Series(res_l2) - res_roll_mean) / res_roll_std

        # Model Probabilities
        preds = {}
        for tname, clf in models.items():
            fcols = tiers[tname]
            p_anom = clf.predict_proba(feats[fcols])[:, 2]  # class 2 = anomaly
            preds[tname] = p_anom

        # Store Time Series
        ts_df = pd.DataFrame({
            "station_id": st_id,
            "station_name": st_name,
            "dt": merged["dt"],
            "gross_cpm": merged["gross_cpm"],
            "has_radnet_obs": merged["has_radnet_obs"],
            "precip_1h_mm": merged["precip_1h_mm"],
            "hours_since_rain": feats["hours_since_rain"],
            "score_t1": preds["Tier 1 (Gross)"],
            "score_t2": preds["Tier 2 (Spectral)"],
            "score_t3": preds["Tier 3 (Weather Fusion)"],
            "score_t4": preds["Tier 4 (Physics-Motivated Nuclear)"],
            "score_b1": z_global,
            "score_b2": z_roll,
            "score_b4c": z_radon_proj,
            "alarm_t1": (preds["Tier 1 (Gross)"] >= FROZEN_THRESHOLDS["Tier 1 (Gross)"]).astype(int),
            "alarm_t2": (preds["Tier 2 (Spectral)"] >= FROZEN_THRESHOLDS["Tier 2 (Spectral)"]).astype(int),
            "alarm_t3": (preds["Tier 3 (Weather Fusion)"] >= FROZEN_THRESHOLDS["Tier 3 (Weather Fusion)"]).astype(int),
            "alarm_t4": (preds["Tier 4 (Physics-Motivated Nuclear)"] >= FROZEN_THRESHOLDS["Tier 4 (Physics-Motivated Nuclear)"]).astype(int),
            "alarm_b1": (z_global >= FROZEN_THRESHOLDS["Baseline: Global Dry 3-sigma"]).astype(int),
            "alarm_b2": (z_roll >= FROZEN_THRESHOLDS["Baseline: Rolling 7d 3-sigma"]).astype(int),
            "alarm_b3": (z_roll >= FROZEN_THRESHOLDS["Baseline: Rolling 7d 4-sigma"]).astype(int),
            "alarm_b4c": (z_radon_proj >= FROZEN_THRESHOLDS["Baseline: Radon Projection 3-sigma"]).astype(int),
        })

        # Windows
        m_pre = (ts_df["dt"] >= WINDOW_PRE[0]) & (ts_df["dt"] <= WINDOW_PRE[1]) & ts_df["has_radnet_obs"]
        m_evt = (ts_df["dt"] >= WINDOW_EVENT[0]) & (ts_df["dt"] <= WINDOW_EVENT[1]) & ts_df["has_radnet_obs"]
        m_post = (ts_df["dt"] >= WINDOW_POST[0]) & (ts_df["dt"] <= WINDOW_POST[1]) & ts_df["has_radnet_obs"]

        hrs_pre = m_pre.sum()
        hrs_evt = m_evt.sum()
        hrs_post = m_post.sum()

        detector_keys = [
            ("Tier 4 (Physics-Motivated Nuclear)", "alarm_t4", "score_t4"),
            ("Tier 3 (Weather Fusion)", "alarm_t3", "score_t3"),
            ("Tier 2 (Spectral)", "alarm_t2", "score_t2"),
            ("Tier 1 (Gross)", "alarm_t1", "score_t1"),
            ("Baseline: Global Dry 3-sigma", "alarm_b1", "score_b1"),
            ("Baseline: Rolling 7d 3-sigma", "alarm_b2", "score_b2"),
            ("Baseline: Rolling 7d 4-sigma", "alarm_b3", "score_b2"),
            ("Baseline: Radon Projection 3-sigma", "alarm_b4c", "score_b4c"),
        ]

        for det_label, col_alm, col_score in detector_keys:
            alms_pre = ts_df.loc[m_pre, col_alm].sum()
            alms_evt = ts_df.loc[m_evt, col_alm].sum()
            alms_post = ts_df.loc[m_post, col_alm].sum()

            rate_pre = (alms_pre / max(1, hrs_pre)) * 8766.0
            rate_evt = (alms_evt / max(1, hrs_evt)) * 8766.0
            rate_post = (alms_post / max(1, hrs_post)) * 8766.0

            # First alarm time in event window relative to 2011-03-18 (earliest documented detection)
            evt_alms = ts_df.loc[m_evt & (ts_df[col_alm] == 1)]
            if len(evt_alms) > 0:
                first_alm_dt = evt_alms["dt"].min()
                lag_hours = (first_alm_dt - pd.Timestamp("2011-03-18 00:00:00")).total_seconds() / 3600.0
                detected_event = True
            else:
                first_alm_dt = None
                lag_hours = np.nan
                detected_event = False

            # Rain coincidence of alarms during event
            if alms_evt > 0:
                rain_coinc = (ts_df.loc[m_evt & (ts_df[col_alm] == 1), "precip_1h_mm"] > 0).mean() * 100.0
            else:
                rain_coinc = np.nan

            summary_rows.append({
                "station_id": st_id,
                "station_name": st_name,
                "is_pacific": st["is_pacific"],
                "detector": det_label,
                "pre_event_hours": int(hrs_pre),
                "event_hours": int(hrs_evt),
                "post_event_hours": int(hrs_post),
                "pre_event_alarms": int(alms_pre),
                "event_alarms": int(alms_evt),
                "post_event_alarms": int(alms_post),
                "pre_event_fa_per_yr": round(rate_pre, 2),
                "event_fa_per_yr": round(rate_evt, 2),
                "post_event_fa_per_yr": round(rate_post, 2),
                "detected_any_alarm_in_event": detected_event,
                "first_alarm_utc": str(first_alm_dt) if first_alm_dt else "None",
                "lead_lag_hours_vs_mar18": round(lag_hours, 1) if not np.isnan(lag_hours) else np.nan,
                "event_rain_coincident_pct": round(rain_coinc, 1) if not np.isnan(rain_coinc) else np.nan,
            })

        all_station_evals.append(ts_df)

    # 3. Export Deliverables
    eval_df = pd.concat(all_station_evals, ignore_index=True)
    eval_csv = Path("data/processed/real_event_fukushima_evaluation.csv")
    eval_df.to_csv(eval_csv, index=False)
    print(f"\nSaved full hourly evaluation to {eval_csv} ({len(eval_df)} rows)")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = Path("data/processed/real_event_fukushima_station_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved station summary table to {summary_csv}")

    # 4. Generate Publication Figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    station_plot_order = ["hi_honolulu", "ca_san_diego", "ak_anchorage", "tx_dallas"]

    for i, sid in enumerate(station_plot_order):
        ax = axes[i]
        sdf = eval_df[eval_df["station_id"] == sid].sort_values("dt").reset_index(drop=True)
        st_name = sdf["station_name"].iloc[0]

        # Radiation timeline
        ax.plot(sdf["dt"], sdf["gross_cpm"], color="#2b5c8f", lw=1.0, alpha=0.8, label="Gross Count Rate (CPM)")

        # Highlight Event Window
        ax.axvspan(pd.Timestamp(WINDOW_EVENT[0]), pd.Timestamp(WINDOW_EVENT[1]), color="#fffae6", alpha=0.6, zorder=0, label="Fukushima Active Window (Mar 11 – Apr 30)")
        # Mark Documented First Detection
        ax.axvline(pd.Timestamp("2011-03-18 00:00:00"), color="#d95f02", ls="--", lw=1.5, label="Documented Filter Detection (Mar 18)")

        # Mark Alarms for Tier 4 and Rolling 3-sigma
        t4_alms = sdf[sdf["alarm_t4"] == 1]
        b2_alms = sdf[sdf["alarm_b2"] == 1]

        if len(b2_alms) > 0:
            ax.scatter(b2_alms["dt"], b2_alms["gross_cpm"], color="#7570b3", s=25, marker="x", zorder=4, label=f"Rolling 3-sigma Alarm (n={len(b2_alms)})")
        if len(t4_alms) > 0:
            ax.scatter(t4_alms["dt"], t4_alms["gross_cpm"], color="#e7298a", s=35, marker="o", facecolors="none", edgecolors="#e7298a", lw=1.5, zorder=5, label=f"Tier 4 Nuclear Alarm (n={len(t4_alms)})")

        ax.set_ylabel(f"{st_name}\nGross CPM", fontsize=10)
        ax.grid(True, ls=":", alpha=0.5)

        # Annotate precipitation spikes
        rain_pts = sdf[sdf["precip_1h_mm"] > 5.0]
        if len(rain_pts) > 0:
            for _, r in rain_pts.iloc[:2].iterrows():
                ax.annotate("Rain Washout", (r["dt"], r["gross_cpm"]), xytext=(0, 15), textcoords="offset points", fontsize=7.5, color="#0072B2", arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.8))

        if i == 0:
            ax.legend(loc="upper right", fontsize=8.5, ncol=3)

    axes[-1].set_xlabel("Date (UTC, 2011)", fontsize=11)
    fig.suptitle("RadNet Real-Event Evaluation: Fukushima Daiichi Trans-Pacific Plume (January–June 2011)\nPacific Gateway Stations (Honolulu, San Diego, Anchorage) vs Continental Control (Dallas)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    fig_path = Path("reports/figures/real_event_fukushima_timeline.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved figure to {fig_path}")

    # Print summary highlights
    print("\n--- SUMMARY OF PRE-REGISTERED REAL EVENT EVALUATION ---")
    t4_summary = summary_df[summary_df["detector"] == "Tier 4 (Physics-Motivated Nuclear)"]
    print(t4_summary[["station_name", "pre_event_fa_per_yr", "event_fa_per_yr", "post_event_fa_per_yr", "detected_any_alarm_in_event", "event_rain_coincident_pct"]])


if __name__ == "__main__":
    main()
