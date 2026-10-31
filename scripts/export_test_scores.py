#!/usr/bin/env python3
"""
scripts/export_test_scores.py

Exports out-of-sample test scores for all baseline detectors (B1-B8)
and ML tiers (Tiers 1-4) on the 2023-2025 test split (5 pilot stations).

Outputs:
- data/processed/test_scores_hourly.csv.gz: Continuous hourly test scores (clean & injected)
- data/processed/test_scores_per_plume.csv.gz: Per-plume peak test scores (200 Family B test plumes)
- data/processed/test_scores_manifest.json: Metadata, model versions, seeds, and split documentation
"""

import gzip
import json
import time
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.evaluate_baselines import _prepare_all_station_data, _fit_and_score_all_detectors

DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 75)
    print("EXPORTING TEST SCORES FOR ALL DETECTORS (2023-2025 TEST SPLIT)")
    print("=" * 75)

    # 1. Prepare station datasets
    print("Loading preprocessed station datasets and synthetic injection catalogs...")
    fit_data, val_data, test_data, val_plumes, test_plumes = _prepare_all_station_data()
    print(f"Data loaded in {time.time() - t0:.1f}s: {len(test_plumes)} test plumes across 5 stations.")

    # 2. Fit all baseline detectors (B1-B8) and ML tiers (Tiers 1-4) on 2017-2020 training split
    t_fit = time.time()
    print("Fitting baseline detectors B1-B8 and ML Tiers 1-4 strictly on 2017-2020 background/train plumes (seed=42)...")
    detectors = _fit_and_score_all_detectors(fit_data, val_data, test_data, seed=42)
    print(f"Fitted and scored {len(detectors)} detectors in {time.time() - t_fit:.1f}s.")

    # 3. Build continuous hourly test dataframe across all 5 pilot stations
    hourly_rows = []
    print("Assembling continuous hourly test score series...")

    for st_id, sdata in test_data.items():
        feats_cln = sdata["feats_cln"]
        feats_inj = sdata["feats_inj"]
        valid_m = sdata["valid_mask"]
        cal_m = sdata["calendar_mask"]

        # Only evaluate on calendar test split (2023-2025)
        eval_mask = np.asarray(cal_m, dtype=bool)
        dts = feats_cln["dt"].to_numpy()[eval_mask]
        dt_series = pd.to_datetime(feats_cln["dt"])
        yms = dt_series.dt.to_period("M").astype(str).to_numpy()[eval_mask]
        val_arr = np.asarray(valid_m, dtype=bool)[eval_mask]
        precip = feats_cln["precip_1h_mm"].fillna(0.0).to_numpy()[eval_mask]
        is_inj_active = feats_inj["injection_active"].fillna(False).to_numpy(dtype=bool)[eval_mask]
        inj_ids = feats_inj["injection_id"].fillna("").to_numpy(dtype=str)[eval_mask]

        st_df = pd.DataFrame({
            "station_id": st_id,
            "dt": dts,
            "year_month": yms,
            "is_valid": val_arr,
            "precip_1h_mm": np.round(precip, 3),
            "is_plume_active": is_inj_active,
            "plume_id": inj_ids,
        })

        for det in detectors:
            det_id = det["detector_id"]
            # Clean background scores (for false alarm rate evaluation)
            cln_scores = det["test_clean"][st_id][eval_mask]
            # Injected scores (for detection evaluation)
            inj_scores = det["test_inj"][st_id][eval_mask]

            st_df[f"score_cln__{det_id}"] = np.round(cln_scores, 6)
            st_df[f"score_inj__{det_id}"] = np.round(inj_scores, 6)

        hourly_rows.append(st_df)

    df_hourly = pd.concat(hourly_rows, ignore_index=True)
    out_hourly = DATA_PROCESSED / "test_scores_hourly.csv.gz"
    print(f"Writing {out_hourly} ({len(df_hourly):,} hourly records)...")
    df_hourly.to_csv(out_hourly, index=False, compression="gzip")
    print(f"Saved {out_hourly} ({out_hourly.stat().st_size / (1024*1024):.2f} MB)")

    # 4. Build per-plume test dataframe (200 test plumes)
    plume_rows = []
    print("Assembling per-plume test scores...")

    for pl in test_plumes:
        inj_id = pl["plume_id"]
        st_id = pl["station_id"]
        indices = pl["indices"]

        p_row = {
            "plume_id": inj_id,
            "station_id": st_id,
            "environment": pl["environment"],
            "scenario": pl["scenario"],
            "snr_bin": pl["snr_bin"],
            "peak_snr_168h": pl["peak_snr_168h"],
            "duration_hours": pl["duration_hours"],
            "start_dt": str(pl["start_dt"]),
        }

        for det in detectors:
            det_id = det["detector_id"]
            inj_sub = det["test_inj"][st_id][indices]
            cln_sub = det["test_clean"][st_id][indices]

            # Peak score during plume passage
            p_row[f"peak_score__{det_id}"] = round(float(np.nanmax(inj_sub)), 6)
            p_row[f"max_clean_score__{det_id}"] = round(float(np.nanmax(cln_sub)), 6)

        plume_rows.append(p_row)

    df_plumes = pd.DataFrame(plume_rows)
    out_plumes = DATA_PROCESSED / "test_scores_per_plume.csv.gz"
    print(f"Writing {out_plumes} ({len(df_plumes)} test plumes)...")
    df_plumes.to_csv(out_plumes, index=False, compression="gzip")
    print(f"Saved {out_plumes} ({out_plumes.stat().st_size / 1024:.1f} KB)")

    # 5. Build and write manifest
    manifest = {
        "dataset_name": "RadNet Out-of-Sample Test Detector Scores",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "description": "Continuous hourly and per-plume out-of-sample detector scores on 2023-2025 RadNet test observations",
        "splits": {
            "training_period": "2017-01-01 to 2020-12-31 (4 years)",
            "validation_period": "2021-01-01 to 2022-12-31 (2 years)",
            "test_period": "2023-01-01 to 2025-12-31 (3 years, 12.16 station-years observed across 5 pilot stations)",
        },
        "model_parameters": {
            "random_seed": 42,
            "training_plume_family": "Family A (Analytic detector response templates)",
            "test_plume_family": "Family B (Independent MCNP/Monte Carlo response templates)",
            "total_test_plumes": len(df_plumes),
            "tuning_on_test": False,
        },
        "files": {
            "hourly_scores": "data/processed/test_scores_hourly.csv.gz",
            "per_plume_scores": "data/processed/test_scores_per_plume.csv.gz",
        },
        "detectors": [
            {
                "detector_id": det["detector_id"],
                "detector_name": det["detector_name"],
                "category": det["category"],
                "uses_synthetic_plumes_in_training": det["uses_synthetic_plumes_in_training"],
            }
            for det in detectors
        ],
    }

    out_manifest = DATA_PROCESSED / "test_scores_manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {out_manifest}")
    print(f"All test score exports completed successfully in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
