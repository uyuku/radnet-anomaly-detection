"""Control experiment: does a trivial rain mask on the rolling-7d z-score baseline
already remove most false alarms?

The rolling 7-day local z-score rule (z_score_168h >= k) is evaluated on the
2023-2025 test split, exactly as in train_and_evaluate_rigorous.py (step 3,
"Continuous Baseline ROC Curves"), except that alarms are suppressed while it is
raining or within H hours after rain (H in MASK_HOURS). H = None means "no
masking" and must reproduce the existing rolling_7d_local_z rows of
data/processed/rigorous_baseline_continuous_roc.csv.

Outputs:
- data/processed/rain_masked_baseline_roc.csv

Run from the repository root:  python src/evaluate_rain_masked_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.feature_engineering import compute_features_for_series
from src.train_and_evaluate_rigorous import STATIONS, count_episodes

HOURS_PER_YEAR = 8766.0
K_STEPS = np.linspace(0.5, 5.0, 91)
MASK_HOURS = [None, 0, 3, 6, 12]  # None = no masking
OUT_CSV = Path("data/processed/rain_masked_baseline_roc.csv")


def load_test_series():
    catalog = pd.read_csv("data/processed/synthetic_injection_catalog.csv")
    test_catalog = catalog[catalog["split"] == "test"].copy().reset_index(drop=True)

    series = {}
    total_valid_hours = 0
    for st in STATIONS:
        st_id = st["id"]
        df_tr = pd.read_csv(f"data/processed/labeled_{st_id}_train.csv.gz")
        df_tr["dt"] = pd.to_datetime(df_tr["dt"])
        df_tr = df_tr.sort_values("dt").reset_index(drop=True)
        df_te = pd.read_csv(f"data/processed/labeled_{st_id}_test.csv.gz")
        df_te["dt"] = pd.to_datetime(df_te["dt"])
        df_te = df_te.sort_values("dt").reset_index(drop=True)

        # Clean pre-2023 warm-up history (no injections) so 2023-01-01 has a full 168h window.
        df_warm = df_tr.copy()
        df_warm["inj_gross_cpm"] = df_warm["gross_cpm"]
        df_warm["inj_dose_rate_nsvh"] = df_warm["dose_rate_nsvh"]
        for ch in range(2, 10):
            df_warm[f"inj_cpm_r0{ch}"] = df_warm[f"cpm_r0{ch}"]

        panel = pd.concat([df_warm, df_te], ignore_index=True).sort_values("dt").reset_index(drop=True)
        m_te = (panel["dt"] >= "2023-01-01").values
        f_inj = compute_features_for_series(panel, st_id, use_injected=True)[m_te].reset_index(drop=True)
        f_cln = compute_features_for_series(panel, st_id, use_injected=False)[m_te].reset_index(drop=True)

        valid = (f_cln["has_radnet_obs"] & f_cln["rad_complete_channels"]).values.astype(bool)
        total_valid_hours += int(valid.sum())
        series[st_id] = {
            "dt": df_te["dt"],
            "valid": valid,
            "z_inj": f_inj["z_score_168h"].values,
            "z_cln": f_cln["z_score_168h"].values,
            "is_raining": f_cln["is_raining"].values,
            "hours_since_rain": f_cln["hours_since_rain"].values,
            "precip_1h": df_te["precip_1h_mm"].fillna(0.0).values,
        }

    # Pre-compute the observed-hour indices of every test injection.
    inj_slices = []
    for _, inj in test_catalog.iterrows():
        s = series[inj["station_id"]]
        start_dt = pd.to_datetime(inj["start_utc"])
        end_dt = pd.to_datetime(inj["end_utc"])
        m = ((s["dt"] >= start_dt) & (s["dt"] <= end_dt)).values & s["valid"]
        inj_slices.append((inj["station_id"], np.where(m)[0]))

    return series, inj_slices, total_valid_hours / HOURS_PER_YEAR


def rain_mask(s, hours):
    """True where an alarm must be suppressed. Unknown weather (NaN) is NOT masked."""
    n = len(s["valid"])
    if hours is None:
        return np.zeros(n, dtype=bool)
    raining = np.nan_to_num(s["is_raining"].astype(float), nan=0.0) > 0
    recent = np.nan_to_num(s["hours_since_rain"].astype(float), nan=np.inf) <= hours
    return raining | recent


def main():
    series, inj_slices, test_station_years = load_test_series()
    print(f"Test station-years: {test_station_years:.2f}; test injections: {len(inj_slices)}")

    rows = []
    for hours in MASK_HOURS:
        masks = {st_id: rain_mask(s, hours) for st_id, s in series.items()}
        label = "none" if hours is None else f"{hours}h"
        for k in K_STEPS:
            episodes = 0
            alarm_hrs = 0
            rain_alarm_hrs = 0
            for st_id, s in series.items():
                al = (s["z_cln"] >= k) & s["valid"] & ~masks[st_id]
                episodes += count_episodes(al, s["valid"])
                alarm_hrs += int(al.sum())
                rain_alarm_hrs += int((al & (s["precip_1h"] > 0.0)).sum())

            det = 0
            for st_id, idx in inj_slices:
                s = series[st_id]
                keep = ~masks[st_id][idx]
                inj_al = (s["z_inj"][idx] >= k) & keep
                cln_al = (s["z_cln"][idx] >= k) & keep
                if (inj_al & ~cln_al).any():
                    det += 1

            rows.append({
                "rain_mask_hours": label,
                "k_multiplier": round(float(k), 2),
                "detection_rate_pct": round(det / len(inj_slices) * 100.0, 2),
                "false_alarms_per_year": round(episodes / test_station_years, 2),
                "rain_coincident_pct": round(rain_alarm_hrs / alarm_hrs * 100.0, 1) if alarm_hrs else 0.0,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

    # Sanity check: "none" must reproduce the existing rolling baseline ROC.
    ref = pd.read_csv("data/processed/rigorous_baseline_continuous_roc.csv")
    ref = ref[ref["baseline_type"] == "rolling_7d_local_z"].set_index("k_multiplier")
    mine = out[out["rain_mask_hours"] == "none"].set_index("k_multiplier")
    diff_fa = (mine["false_alarms_per_year"] - ref["false_alarms_per_year"]).abs().max()
    diff_det = (mine["detection_rate_pct"] - ref["detection_rate_pct"]).abs().max()
    print(f"Reproduction check vs rigorous_baseline_continuous_roc.csv: max |dFA|={diff_fa:.3f}, max |dDet|={diff_det:.3f}")
    if diff_fa > 0.05 or diff_det > 0.05:
        print("WARNING: unmasked baseline does NOT reproduce the existing table. Do not trust the masked results.")
        sys.exit(1)


if __name__ == "__main__":
    main()
