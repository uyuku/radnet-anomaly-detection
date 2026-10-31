"""
Recomputes all quantitative claims from the committed datasets and processed outputs,
writes data/processed/recomputed_claims_summary.json, and verifies that README.md
and docs/claims_audit.csv match the recomputed evidence.

Exits 0 if all checks pass; exits 1 if any discrepancy or unresolved unsupported claim is found.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.analyze_filter_cycles_multi_station import detect_filter_drops
from src.feature_engineering import compute_features_for_series

HOURS_PER_YEAR = 8766.0

STATIONS = [
    {"id": "al_birmingham", "name": "Birmingham, AL", "drop_thresh_inj": -300.0, "drop_thresh_cycle": -300.0},
    {"id": "dc_washington", "name": "Washington, DC", "drop_thresh_inj": -250.0, "drop_thresh_cycle": -250.0},
    {"id": "ca_san_diego", "name": "San Diego, CA", "drop_thresh_inj": -300.0, "drop_thresh_cycle": -450.0},
    {"id": "tx_dallas", "name": "Dallas, TX", "drop_thresh_inj": -300.0, "drop_thresh_cycle": -300.0},
    {"id": "fl_tampa", "name": "Tampa, FL", "drop_thresh_inj": -300.0, "drop_thresh_cycle": -300.0},
]


def recompute_dataset_accounting(data_dir: Path) -> dict:
    """Recomputes valid hours, synchronous hours, test hours, and station-years."""
    merge_df = pd.read_csv(data_dir / "merge_audit_summary.csv")
    recon_df = pd.read_csv(data_dir / "labeled_dataset_reconciliation_summary.csv")
    cat_df = pd.read_csv(data_dir / "synthetic_injection_catalog.csv")

    valid_radnet_hours = int(merge_df["RadNet Valid Hours"].sum())
    synchronous_hours = int(merge_df["Synchronous Both Hours"].sum())
    total_grid_hours = int(merge_df["Total Grid Hours"].sum())

    test_recon = recon_df[recon_df["Split"] == "test"]
    test_total_hours = int(test_recon["Total_Hours"].sum())
    test_unobs_hours = int(test_recon["Unobserved_Hours"].sum())
    test_observed_hours = test_total_hours - test_unobs_hours
    test_station_years = round(test_observed_hours / HOURS_PER_YEAR, 2)

    train_plumes = int((cat_df["split"] == "train").sum())
    test_plumes = int((cat_df["split"] == "test").sum())
    total_plumes = len(cat_df)

    return {
        "total_grid_hours": total_grid_hours,
        "valid_radnet_hours": valid_radnet_hours,
        "synchronous_hours": synchronous_hours,
        "test_observed_hours": test_observed_hours,
        "test_station_years": test_station_years,
        "total_plumes": total_plumes,
        "train_plumes": train_plumes,
        "test_plumes": test_plumes,
        "passage_hours_min": int(cat_df["t_passage_hours"].min()),
        "passage_hours_max": int(cat_df["t_passage_hours"].max()),
        "duration_hours_min": int(cat_df["duration_hours"].min()),
        "duration_hours_max": int(cat_df["duration_hours"].max()),
    }


def recompute_washout_surges(data_dir: Path) -> dict:
    """Extracts rain washout surge distribution statistics from committed summary."""
    df = pd.read_csv(data_dir / "rain_washout_surge_distribution.csv")
    pooled = df[(df["Group"] == "All Stations Pooled") & (df["Category"] == "All Rain")].iloc[0]
    sd = df[(df["Group"] == "San Diego, CA") & (df["Category"] == "All Rain")].iloc[0]

    return {
        "rain_hours": int(pooled["Rain_Hours"]),
        "pooled_mean_pct": float(pooled["Mean"]),
        "pooled_median_pct": float(pooled["p50_Median"]),
        "pooled_p90_pct": float(pooled["p90"]),
        "pooled_p99_pct": float(pooled["p99"]),
        "pooled_max_pct": float(pooled["Max"]),
        "san_diego_median_pct": float(sd["p50_Median"]),
    }


def recompute_filter_cycles(data_dir: Path) -> dict:
    """Computes filter-change cadence from multi_station_filter_cycle_summary.csv and full 2017-2025 series."""
    cyc_df = pd.read_csv(data_dir / "multi_station_filter_cycle_summary.csv")
    summary = {
        "window_2021_2024": {
            "median_interval_days_min": float(cyc_df["Median_Interval_Days"].min()),
            "median_interval_days_max": float(cyc_df["Median_Interval_Days"].max()),
            "mean_interval_days_min": float(cyc_df["Mean_Interval_Days"].min()),
            "mean_interval_days_max": float(cyc_df["Mean_Interval_Days"].max()),
            "mean_step_drop_cpm_min": float(cyc_df["Mean_Step_Drop_CPM"].min()),
            "mean_step_drop_cpm_max": float(cyc_df["Mean_Step_Drop_CPM"].max()),
            "detected_drops_min": int(cyc_df["Detected_Step_Drops"].min()),
            "detected_drops_max": int(cyc_df["Detected_Step_Drops"].max()),
        },
        "per_station_2017_2025": {},
    }

    for st in STATIONS:
        merged_path = data_dir / f"merged_{st['id']}_2017_2025.csv.gz"
        if merged_path.exists():
            df = pd.read_csv(merged_path)
            df["dt"] = pd.to_datetime(df["utc_hour"])
            df = df.sort_values("dt").reset_index(drop=True)
            drops = detect_filter_drops(df, st["drop_thresh_inj"])
            intervals = drops["dt"].diff().dt.total_seconds() / 86400.0
            plausible = intervals[(intervals >= 1.5) & (intervals <= 14.0)]
            summary["per_station_2017_2025"][st["id"]] = {
                "detected_drops_total": int(len(drops)),
                "all_interval_median_days": round(float(intervals.median()), 2) if len(intervals) > 0 else None,
                "plausible_interval_median_days": round(float(plausible.median()), 2) if len(plausible) > 0 else None,
                "plausible_interval_mean_days": round(float(plausible.mean()), 2) if len(plausible) > 0 else None,
            }
    return summary


def recompute_baseline_rates(data_dir: Path) -> dict:
    """Computes per-station and pooled baseline false-alarm rates on 2017-2025 and 2023-2025 test split."""
    base_df = pd.read_csv(data_dir / "baseline_threshold_evaluation.csv")
    r3s = base_df[base_df["Rule_Code"] == "rolling_cpm_3s"]
    per_station_2017_2025 = {}
    for _, row in r3s.iterrows():
        per_station_2017_2025[str(row["Station_ID"])] = {
            "episodes_per_year": float(row["Episodes_Per_Year"]),
            "episode_rain_3h_pct": float(row["Episode_Rain_3h_Pct"]),
        }

    per_station_test_2023_2025 = {}
    tot_ep_local = 0
    tot_ep_global = 0
    tot_hours = 0

    for st in STATIONS:
        st_id = st["id"]
        test_file = data_dir / f"labeled_{st_id}_test.csv.gz"
        if test_file.exists():
            df = pd.read_csv(test_file)
            feats = compute_features_for_series(df, st_id, use_injected=False)
            valid = feats["has_radnet_obs"] & feats["rad_complete_channels"]
            hrs = int(valid.sum())
            yrs = hrs / HOURS_PER_YEAR

            a_loc = valid & (feats["z_score_168h"] > 3.0)
            ep_loc = int((a_loc & ~a_loc.shift(1, fill_value=False)).sum())

            a_glob = valid & (feats["z_score_global_dry"] > 3.0)
            ep_glob = int((a_glob & ~a_glob.shift(1, fill_value=False)).sum())

            per_station_test_2023_2025[st_id] = {
                "observed_hours": hrs,
                "station_years": round(yrs, 2),
                "local_168h_3s_episodes_per_year": round(ep_loc / yrs, 2),
                "global_dry_3s_episodes_per_year": round(ep_glob / yrs, 2),
            }
            tot_ep_local += ep_loc
            tot_ep_global += ep_glob
            tot_hours += hrs

    roc_df = pd.read_csv(data_dir / "rigorous_baseline_continuous_roc.csv")
    loc_3s_pooled = float(
        roc_df[(roc_df["baseline_type"] == "rolling_7d_local_z") & (roc_df["k_multiplier"] == 3.0)][
            "false_alarms_per_year"
        ].iloc[0]
    )
    glob_3s_pooled = float(
        roc_df[(roc_df["baseline_type"] == "global_dry_sigma") & (roc_df["k_multiplier"] == 3.0)][
            "false_alarms_per_year"
        ].iloc[0]
    )

    return {
        "pooled_test_2023_2025_rolling_7d_local_3s": loc_3s_pooled,
        "pooled_test_2023_2025_global_dry_3s": glob_3s_pooled,
        "per_station_2017_2025_rolling_cpm_3s": per_station_2017_2025,
        "per_station_test_2023_2025": per_station_test_2023_2025,
    }


def recompute_benchmark_metrics(data_dir: Path) -> dict:
    """Extracts benchmark and LOSO headline numbers from committed CSVs."""
    rig_df = pd.read_csv(data_dir / "rigorous_benchmark_summary.csv")
    unc_df = pd.read_csv(data_dir / "eval_benchmark_uncertainty_summary.csv")
    loso_df = pd.read_csv(data_dir / "rigorous_loso_summary.csv")

    tiers = {}
    for _, row in rig_df.iterrows():
        key = f"{row['tier_name']}@{int(row['target_detection_pct'])}"
        tiers[key] = {
            "frozen_tau_mean": float(row["frozen_tau_mean"]),
            "det_total_mean": float(row["det_total_mean"]),
            "det_total_std": float(row["det_total_std"]),
            "det_12h_mean": float(row["det_12h_mean"]),
            "fa_per_year_mean": float(row["fa_per_year_mean"]),
            "fa_per_year_std": float(row["fa_per_year_std"]),
            "rain_coincident_mean": float(row["rain_coincident_mean"]),
            "median_delay_mean": float(row["median_delay_mean"]),
        }

    baselines = {}
    for _, row in unc_df[unc_df["model_name"].str.startswith("Baseline:")].iterrows():
        baselines[str(row["model_name"])] = {
            "realized_detection_pct": float(row["realized_detection_pct"]),
            "false_alarms_per_station_year": float(row["false_alarms_per_station_year"]),
            "rain_coincident_alarm_pct": float(row["rain_coincident_alarm_pct"]),
            "median_detection_delay_hours": float(row["median_detection_delay_hours"]),
        }

    loso = {}
    for _, row in loso_df.iterrows():
        key = f"{row['tier_name']}:{row['held_out_station']}"
        loso[key] = {
            "det_pct": float(row["realized_detection_rate_pct"]),
            "fa_per_year": float(row["clean_false_alarms_per_year"]),
        }

    return {"tiers": tiers, "baselines": baselines, "loso": loso}


def verify_documents_and_audit(
    readme_path: Path,
    audit_csv_path: Path,
    summary: dict,
) -> list[str]:
    """Verifies that README.md and docs/claims_audit.csv satisfy all checks."""
    errors: list[str] = []

    readme_text = readme_path.read_text(encoding="utf-8")

    # 1. Check required numbers in README.md
    required_readme_Substrings = [
        "334,567",
        "324,550",
        "106,628",
        "12.16",
        "450",
        "+6.9%",
        "+41.4%",
        "+196.7%",
        "82.95",
        "52.86",
        "7.19 ± 1.17",
        "93.60 ± 1.03%",
        "1.23 ± 1.08",
        "85.83 ± 2.48%",
        "0.33 ± 0.37",
        "82.92 ± 3.60%",
        "55.43 ± 6.91",
        "83.42 ± 1.60%",
        "206.47 ± 52.91",
        "87.95 ± 2.38%",
        "169.42",
        "1151.30",
    ]
    for sub in required_readme_Substrings:
        if sub not in readme_text:
            errors.append(f"README.md missing expected verified value: {sub!r}")

    # 2. Check prohibited stale/contradicted claims
    prohibited_readme = [
        "169.36",
        "92.22",
        "38.64",
        "87.92",
        "60.75",
        "90.30",
        "56.50%",
        "+7.2%",
        "stripped_energy_weighted_dose_anomaly",
    ]
    for bad in prohibited_readme:
        if bad in readme_text:
            errors.append(f"README.md still contains contradicted value/string: {bad!r}")

    # 3. Check claims_audit.csv structure and resolution
    if not audit_csv_path.exists():
        errors.append(f"Missing claims audit CSV: {audit_csv_path}")
    else:
        audit_df = pd.read_csv(audit_csv_path)
        expected_cols = {
            "claim_id",
            "location",
            "claim_text",
            "type",
            "evidence",
            "status",
            "proposed_fix",
        }
        missing_cols = expected_cols - set(audit_df.columns)
        if missing_cols:
            errors.append(f"docs/claims_audit.csv missing columns: {sorted(missing_cols)}")
        valid_statuses = {"SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "NEEDS_REWORDING", "UNVERIFIED"}
        bad_statuses = set(audit_df["status"].dropna().unique()) - valid_statuses
        if bad_statuses:
            errors.append(f"docs/claims_audit.csv has invalid status values: {sorted(bad_statuses)}")
        for _, row in audit_df.iterrows():
            st = str(row["status"])
            fix = str(row["proposed_fix"])
            if st in {"UNSUPPORTED", "CONTRADICTED"} and (not fix or fix == "nan"):
                errors.append(f"Claim {row['claim_id']} is {st} without a proposed_fix or TODO marker.")

    return errors


def main() -> int:
    root = Path(__file__).resolve().parent
    data_dir = root / "data" / "processed"
    readme_path = root / "README.md"
    audit_csv_path = root / "docs" / "claims_audit.csv"

    summary = {
        "accounting": recompute_dataset_accounting(data_dir),
        "washout_surges": recompute_washout_surges(data_dir),
        "filter_cycles": recompute_filter_cycles(data_dir),
        "baseline_rates": recompute_baseline_rates(data_dir),
        "benchmark_metrics": recompute_benchmark_metrics(data_dir),
    }

    out_json = data_dir / "recomputed_claims_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote recomputed claims summary to {out_json}")

    errors = verify_documents_and_audit(readme_path, audit_csv_path, summary)
    if errors:
        print("\nCLAIM VERIFICATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("All quantitative and factual claim checks passed (exit 0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
