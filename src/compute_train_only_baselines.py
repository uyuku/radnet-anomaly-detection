"""
Computes FROZEN train-only dry baselines (2017-2022) per station and writes
data/processed/station_dry_baselines_train_only.json.

These baselines (mu/sigma of gross CPM and dose rate on verified dry hours)
are used by:
  - feature_engineering.py  (z_score_global_dry, dose_to_gross_ratio)
  - synthetic_injection.py  (washout labeling thresholds; BUG-4 fix)

Train-only = strictly before the 2023-2025 test split. No test-period
statistics enter the label or feature normalizers.

Dry hour definition (project standard): current hour precip == 0.0 mm AND
trailing 24h precip sum == 0.0 mm, with a complete-channel RadNet observation.
Hours with unknown precipitation (NaN) are NOT dry.
"""

import json
from pathlib import Path

import pandas as pd

STATIONS = ["al_birmingham", "dc_washington", "ca_san_diego", "tx_dallas", "fl_tampa"]
TRAIN_END = "2023-01-01"
OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "station_dry_baselines_train_only.json"


def main():
    baselines = {}
    for st_id in STATIONS:
        csv_file = Path(f"data/processed/merged_{st_id}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        df["dt"] = pd.to_datetime(df["utc_hour"])
        df = df.sort_values("dt").reset_index(drop=True)

        df["precip_24h"] = df["precip_1h_mm"].rolling(24, min_periods=12).sum()

        train = df[df["dt"] < TRAIN_END]
        valid = train["has_radnet_obs"].fillna(False).astype(bool) & train["rad_complete_channels"].fillna(False).astype(bool)
        dry_mask = (train["precip_1h_mm"] == 0.0) & (train["precip_24h"] == 0.0) & valid

        dry = train[dry_mask]
        dose = dry["dose_rate_nsvh"]
        dose = dose[dose.notna() & (dose > 0)]

        baselines[st_id] = {
            "mu": round(float(dry["gross_cpm"].mean()), 1),
            "sigma": round(float(dry["gross_cpm"].std()), 1),
            "mu_dose": round(float(dose.mean()), 1),
            "sigma_dose": round(float(dose.std()), 1),
            "dry_hours_train": int(dry_mask.sum()),
        }
        print(f"{st_id}: mu={baselines[st_id]['mu']}, sigma={baselines[st_id]['sigma']}, "
              f"mu_dose={baselines[st_id]['mu_dose']}, sigma_dose={baselines[st_id]['sigma_dose']}, "
              f"dry_hours={baselines[st_id]['dry_hours_train']}")

    with open(OUT_FILE, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()
