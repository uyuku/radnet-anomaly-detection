"""
Computes FROZEN train-only dry baselines (2017-2022) and fit-fold (2017-2020)
8-channel Radon Washout Basis Spectra per station, writing
data/processed/station_dry_baselines_train_only.json.

These baselines are used by:
  - feature_engineering.py  (z_score_global_dry, dose_to_gross_ratio,
                             spectral stripping, and Bateman washout residuals)
  - synthetic_injection.py  (washout labeling thresholds and channel-resolved dose coupling)

Dry hour definition (project standard): current hour precip == 0.0 mm AND
trailing 24h precip sum == 0.0 mm, with a complete-channel RadNet observation.
Hours with unknown precipitation (NaN) are NOT dry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.nuclear_physics import (
    CHANNELS,
    DEFAULT_RADON_BASIS,
    compute_channel_dose_response_vector,
)

STATIONS = ["al_birmingham", "dc_washington", "ca_san_diego", "tx_dallas", "fl_tampa"]
FIT_FOLD_END = "2021-01-01"
TRAIN_END = "2023-01-01"
OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "station_dry_baselines_train_only.json"

DEFAULT_SLOPES = {
    "al_birmingham": 0.01389,
    "dc_washington": 0.01668,
    "ca_san_diego": 0.02359,
    "tx_dallas": 0.01238,
    "fl_tampa": 0.01142,
}


def main() -> None:
    baselines: dict[str, dict] = {}
    pooled_washout_excess = np.zeros(8, dtype=float)
    station_washout_excess: dict[str, np.ndarray] = {}
    station_washout_hours: dict[str, int] = {}

    # Pass 1: Compute dry baselines and collect fit-fold (2017-2020) radon washout channel excesses
    for st_id in STATIONS:
        csv_file = Path(f"data/processed/merged_{st_id}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        df["dt"] = pd.to_datetime(df["utc_hour"])
        df = df.sort_values("dt").reset_index(drop=True)

        df["precip_24h"] = df["precip_1h_mm"].rolling(24, min_periods=12).sum()

        # Pre-test freeze (2017-2022) for gross/dose dry baseline normalizers
        train = df[df["dt"] < TRAIN_END]
        valid = (
            train["has_radnet_obs"].fillna(False).astype(bool)
            & train["rad_complete_channels"].fillna(False).astype(bool)
        )
        dry_mask = (train["precip_1h_mm"] == 0.0) & (train["precip_24h"] == 0.0) & valid

        dry = train[dry_mask]
        dose = dry["dose_rate_nsvh"]
        dose = dose[dose.notna() & (dose > 0)]

        # Strictly fit-fold (2017-2020) channel dry baselines and radon washout spectrum
        fit_df = df[df["dt"] < FIT_FOLD_END]
        fit_valid = (
            fit_df["has_radnet_obs"].fillna(False).astype(bool)
            & fit_df["rad_complete_channels"].fillna(False).astype(bool)
        )
        fit_dry_mask = (fit_df["precip_1h_mm"] == 0.0) & (fit_df["precip_24h"] == 0.0) & fit_valid
        fit_dry = fit_df[fit_dry_mask]

        ch_mu_dry = [round(float(fit_dry[c].mean()), 2) for c in CHANNELS]
        ch_sigma_dry = [round(max(1.0, float(fit_dry[c].std())), 2) for c in CHANNELS]

        # Fit-fold rain washout hours: precip >= 1.0 mm/h and gross excess > 100 CPM above dry mean
        rain_mask = (fit_df["precip_1h_mm"] >= 1.0) & fit_valid
        rain_df = fit_df[rain_mask]
        if not rain_df.empty:
            exc_mat = rain_df[CHANNELS].to_numpy(dtype=float) - np.asarray(ch_mu_dry, dtype=float)[None, :]
            exc_gross = np.sum(exc_mat, axis=1)
            surge_mask = exc_gross > 100.0
            if np.any(surge_mask):
                pos_exc = np.maximum(0.0, exc_mat[surge_mask])
                sum_exc = np.sum(pos_exc, axis=0)
                station_washout_excess[st_id] = sum_exc
                station_washout_hours[st_id] = int(np.sum(surge_mask))
                pooled_washout_excess += sum_exc
            else:
                station_washout_excess[st_id] = np.zeros(8, dtype=float)
                station_washout_hours[st_id] = 0
        else:
            station_washout_excess[st_id] = np.zeros(8, dtype=float)
            station_washout_hours[st_id] = 0

        baselines[st_id] = {
            "mu": round(float(dry["gross_cpm"].mean()), 1),
            "sigma": round(float(dry["gross_cpm"].std()), 1),
            "mu_dose": round(float(dose.mean()), 1),
            "sigma_dose": round(float(dose.std()), 1),
            "dry_hours_train": int(dry_mask.sum()),
            "channel_mu_dry": ch_mu_dry,
            "channel_sigma_dry": ch_sigma_dry,
            "washout_hours_fit_fold": station_washout_hours[st_id],
        }

    # Normalize pooled 2017-2020 radon washout basis spectrum s_Rn
    if np.sum(pooled_washout_excess) > 0:
        pooled_basis = pooled_washout_excess / np.sum(pooled_washout_excess)
    else:
        pooled_basis = DEFAULT_RADON_BASIS.copy()

    # Read empirical slopes if available
    dose_csv = Path(__file__).resolve().parent.parent / "data" / "processed" / "dose_rate_cpm_regression_summary.csv"
    slopes = dict(DEFAULT_SLOPES)
    if dose_csv.exists():
        dose_df = pd.read_csv(dose_csv)
        for _, r in dose_df.iterrows():
            sid = str(r["Station_ID"])
            if sid in slopes:
                slopes[sid] = float(r["Slope_nSvh_per_CPM"])

    # Pass 2: Finalize station radon washout basis s_Rn (shrunk with pooled prior for small-N stability)
    for st_id in STATIONS:
        n_w = station_washout_hours[st_id]
        raw_sum = station_washout_excess[st_id]
        if n_w > 0 and np.sum(raw_sum) > 0:
            st_basis = raw_sum / np.sum(raw_sum)
            # Empirical Bayes shrinkage weight (n_w / (n_w + 100)) so sparse rain stations
            # like San Diego retain physical stability without losing local calibration
            w_shrink = float(n_w) / float(n_w + 100.0)
            s_rn = w_shrink * st_basis + (1.0 - w_shrink) * pooled_basis
        else:
            s_rn = pooled_basis.copy()
        s_rn = np.maximum(1e-4, s_rn)
        s_rn = s_rn / np.sum(s_rn)

        k_vec = compute_channel_dose_response_vector(slopes[st_id], s_rn)
        baselines[st_id]["radon_washout_basis"] = [round(float(x), 6) for x in s_rn]
        # Renormalize rounded basis so sum is exact 1.0
        rounded_s = np.array(baselines[st_id]["radon_washout_basis"], dtype=float)
        rounded_s[-1] = round(1.0 - float(np.sum(rounded_s[:-1])), 6)
        baselines[st_id]["radon_washout_basis"] = [float(x) for x in rounded_s]
        baselines[st_id]["k_dose_slope"] = round(float(slopes[st_id]), 5)
        baselines[st_id]["k_dose_channel"] = [round(float(x), 6) for x in k_vec]

        print(
            f"{st_id}: mu={baselines[st_id]['mu']}, sigma={baselines[st_id]['sigma']}, "
            f"mu_dose={baselines[st_id]['mu_dose']}, sigma_dose={baselines[st_id]['sigma_dose']}, "
            f"dry_hours={baselines[st_id]['dry_hours_train']}, "
            f"fit_washout_hrs={n_w}, s_Rn_R02_R03={s_rn[0]+s_rn[1]:.3f}, s_Rn_R07_R08={s_rn[5]+s_rn[6]:.4f}"
        )

    with open(OUT_FILE, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()
