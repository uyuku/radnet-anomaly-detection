"""
Calibrates the physical dose-rate coupling coefficient (k_dose) and the
8-channel energy-weighted dose response vector (k_dose_channel, nSv/h per CPM
for R02..R09) grounded in ICRP-74 ambient dose equivalent conversion h*(10, E_gamma)
and NaI(Tl) intrinsic efficiency, anchored against the empirical radon washout
regression for each pilot station and the pooled network.

Saves: data/processed/dose_rate_cpm_regression_summary.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.nuclear_physics import (
    DEFAULT_RADON_BASIS,
    compute_channel_dose_response_vector,
)

STATIONS = [
    {"id": "al_birmingham", "name": "Birmingham, AL"},
    {"id": "dc_washington", "name": "Washington, DC"},
    {"id": "ca_san_diego", "name": "San Diego, CA"},
    {"id": "tx_dallas", "name": "Dallas, TX"},
    {"id": "fl_tampa", "name": "Tampa, FL"},
]


def _load_station_radon_basis(station_id: str) -> np.ndarray:
    base_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "station_dry_baselines_train_only.json"
    if base_path.exists():
        with open(base_path, "r") as f:
            data = json.load(f)
        if station_id in data and "radon_washout_basis" in data[station_id]:
            return np.asarray(data[station_id]["radon_washout_basis"], dtype=float)
    return DEFAULT_RADON_BASIS.copy()


def run_dose_calibration() -> pd.DataFrame:
    results = []
    pooled_cpm_excess = []
    pooled_dose_excess = []

    for st in STATIONS:
        csv_file = Path(f"data/processed/merged_{st['id']}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        df["dt"] = pd.to_datetime(df["utc_hour"])
        df = df.sort_values("dt").reset_index(drop=True)

        # Baseline: 168-hour trailing (causal) rolling median on continuous calendar grid
        df["cpm_base"] = df["gross_cpm"].rolling(168, min_periods=72).median()
        df["dose_base"] = df["dose_rate_nsvh"].rolling(168, min_periods=72).median()

        df["cpm_excess"] = df["gross_cpm"] - df["cpm_base"]
        df["dose_excess"] = df["dose_rate_nsvh"] - df["dose_base"]

        # Selection criteria on pre-test hours (or full verified rain hours):
        # 1. Active precipitation: precip_1h_mm >= 1.0 mm
        # 2. Significant gross excess: cpm_excess >= 100 CPM
        # 3. Positive dose excess
        # 4. Valid RadNet observations
        mask = (
            (df["precip_1h_mm"] >= 1.0)
            & (df["cpm_excess"] >= 100.0)
            & (df["dose_excess"] > 0.0)
            & df["has_radnet_obs"]
            & df["rad_complete_channels"]
            & df["dose_rate_nsvh"].notna()
        )
        sub = df[mask].dropna(subset=["cpm_excess", "dose_excess"])

        if len(sub) > 10:
            reg = stats.linregress(sub["cpm_excess"], sub["dose_excess"])
            ci_95 = 1.96 * reg.stderr
            slope = round(float(reg.slope), 5)
            s_rn = _load_station_radon_basis(st["id"])
            k_vec = compute_channel_dose_response_vector(slope, s_rn)

            row = {
                "Station_ID": st["id"],
                "Station_Name": st["name"],
                "Rain_Hours_Analyzed": len(sub),
                "Slope_nSvh_per_CPM": slope,
                "Intercept_nSvh": round(float(reg.intercept), 2),
                "R_Squared": round(float(reg.rvalue ** 2), 3),
                "Std_Err": round(float(reg.stderr), 6),
                "CI95_Lower": round(float(reg.slope - ci_95), 5),
                "CI95_Upper": round(float(reg.slope + ci_95), 5),
                "P_Value": float(reg.pvalue),
            }
            for ch_idx in range(8):
                row[f"k_dose_r0{ch_idx+2}"] = round(float(k_vec[ch_idx]), 6)
            results.append(row)
            pooled_cpm_excess.extend(sub["cpm_excess"].tolist())
            pooled_dose_excess.extend(sub["dose_excess"].tolist())

    # Pooled network regression
    reg_pool = stats.linregress(pooled_cpm_excess, pooled_dose_excess)
    ci_95_pool = 1.96 * reg_pool.stderr
    pool_slope = round(float(reg_pool.slope), 5)
    k_vec_pool = compute_channel_dose_response_vector(pool_slope, DEFAULT_RADON_BASIS)
    pool_row = {
        "Station_ID": "pooled_network",
        "Station_Name": "Pooled Network (5 Stations)",
        "Rain_Hours_Analyzed": len(pooled_cpm_excess),
        "Slope_nSvh_per_CPM": pool_slope,
        "Intercept_nSvh": round(float(reg_pool.intercept), 2),
        "R_Squared": round(float(reg_pool.rvalue ** 2), 3),
        "Std_Err": round(float(reg_pool.stderr), 6),
        "CI95_Lower": round(float(reg_pool.slope - ci_95_pool), 5),
        "CI95_Upper": round(float(reg_pool.slope + ci_95_pool), 5),
        "P_Value": float(reg_pool.pvalue),
    }
    for ch_idx in range(8):
        pool_row[f"k_dose_r0{ch_idx+2}"] = round(float(k_vec_pool[ch_idx]), 6)
    results.append(pool_row)

    out_df = pd.DataFrame(results)
    out_csv = Path("data/processed/dose_rate_cpm_regression_summary.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved channel-resolved dose-rate coupling calibration to {out_csv}")
    print(out_df.to_string(index=False))
    return out_df


if __name__ == "__main__":
    run_dose_calibration()
