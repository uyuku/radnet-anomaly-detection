"""Portable end-to-end pipeline runner with prerequisite checks and failure propagation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STATION_IDS = ("al_birmingham", "dc_washington", "ca_san_diego", "tx_dallas", "fl_tampa")


def processed(*names):
    return tuple(f"data/processed/{name}" for name in names)


MERGED = processed(*(f"merged_{s}_2017_2025.csv.gz" for s in STATION_IDS))
LABELED = processed(*(f"labeled_{s}_{split}.csv.gz" for s in STATION_IDS for split in ("train", "test")))
BASELINES = processed("station_dry_baselines_train_only.json")
CATALOG = processed("synthetic_injection_catalog.csv")
RADNET = tuple(f"data/raw/{name}.zip" for name in (
    "al_birmingham_2025-2007", "dc_washington_2025-2006", "ca_san_diego_2025-2006",
    "tx_dallas_2025-2007", "fl_tampa_2025-2008",
))
CANDIDATES = RADNET + tuple(f"data/raw/{name}.zip" for name in (
    "al_montgomery_2025-2006", "il_chicago_2025-2006", "tx_austin_2025-2007",
))
NOAA = tuple(f"data/raw/noaa_{s}_{y}.csv" for s in ("kbhm", "kdca", "ksan", "kdfw", "ktpa") for y in range(2017, 2026))
MODEL_TABLES = processed(*(f"model_{name}.csv" for name in (
    "feature_importance", "roc_curve_data", "operating_points_summary",
    "stratified_evaluation_summary", "loso_evaluation_summary",
)))
RIGOROUS_TABLES = processed(*(f"rigorous_{name}.csv" for name in (
    "baseline_continuous_roc", "seed_level_results", "seed_statistical_tests",
    "continuous_roc_all_tiers", "benchmark_summary", "matched_detection_comparison",
    "loso_summary", "loso_stratified_detection", "template_sensitivity",
    "gain_drift_evaluation", "stratified_deadlines",
)))
EVAL_TABLES = processed(*(f"eval_{name}.csv" for name in (
    "benchmark_uncertainty_summary", "bootstrap_distributions", "detection_delay_summary",
    "empirical_delays", "confusion_matrices", "mlp_comparison_summary",
)))


@dataclass(frozen=True)
class Step:
    script: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()


def build_plan(script_filter: str | None = None) -> list[Step]:
    steps = [
        # A fresh checkout must download the index before parsing it.
        Step("download_pilot_data.py", outputs=CANDIDATES + ("data/raw/radnet_csv_file_downloads.html",)),
        Step("parse_stations.py", ("data/raw/radnet_csv_file_downloads.html",), processed("radnet_station_inventory.csv")),
        Step("analyze_pilot_stations.py", CANDIDATES, processed("pilot_station_metrics.csv", "pilot_station_channel_missingness.csv", "candidate_pilot_stations_comparison.csv")),
        Step("download_noaa_data.py", outputs=NOAA),
        Step("audit_noaa_precipitation.py", NOAA, processed("noaa_precipitation_audit.csv", "noaa_station_summary.csv")),
        Step("merge_radnet_weather.py", RADNET + NOAA, MERGED + processed("merge_audit_summary.csv")),
        Step("plot_rain_events.py", MERGED),
        Step("characterize_baseline_cycles.py", MERGED, processed("diurnal_cycle_summary.csv", "filter_cycle_summary.csv")),
        Step("analyze_filter_cycles_multi_station.py", MERGED, processed("multi_station_filter_cycle_summary.csv")),
        Step("analyze_timestamp_lags.py", MERGED, processed("lag_cross_correlation.csv")),
        Step("analyze_rain_washout_distribution.py", MERGED, processed("rain_washout_surge_distribution.csv")),
        Step("evaluate_baseline.py", MERGED, processed("baseline_threshold_evaluation.csv")),
        Step("plot_baseline_timeline.py", MERGED),
        Step("compute_train_only_baselines.py", MERGED, BASELINES),
        Step("calibrate_dose_coupling.py", MERGED, processed("dose_rate_cpm_regression_summary.csv")),
        Step("analyze_washout_spectrum.py", MERGED, processed("rain_washout_spectral_shares.csv")),
        Step("synthetic_injection.py", MERGED + BASELINES + processed("dose_rate_cpm_regression_summary.csv") + ("configs/config.yaml",), LABELED + CATALOG + processed("labeled_dataset_reconciliation_summary.csv")),
        Step("audit_injection_catalog.py", MERGED + LABELED + CATALOG, processed("injection_catalog_audit_summary.csv", "injection_catalog_audit_verification.csv")),
        Step("plot_synthetic_injections.py", LABELED + CATALOG + processed("rain_washout_spectral_shares.csv")),
        Step("train_and_evaluate_models.py", LABELED + CATALOG + BASELINES, MODEL_TABLES),
        Step("train_and_evaluate_rigorous.py", LABELED + CATALOG + BASELINES, RIGOROUS_TABLES + processed("eval_mlp_calibration_comparison.csv")),
        Step("plot_model_evaluation.py", MODEL_TABLES + LABELED + BASELINES + processed("rigorous_baseline_continuous_roc.csv")),
        Step("plot_rigorous_evaluation.py", RIGOROUS_TABLES),
        Step("evaluate_uncertainty_protocol.py", LABELED + CATALOG + BASELINES, EVAL_TABLES),
        Step("plot_uncertainty_evaluation.py", EVAL_TABLES),
        Step("plot_environmental_engineering_synthesis.py", LABELED + processed("diurnal_cycle_summary.csv", "rigorous_baseline_continuous_roc.csv", "rigorous_benchmark_summary.csv")),
    ]
    return [step for step in steps if script_filter is None or step.script == script_filter]


def missing_inputs(steps: list[Step], root: Path = ROOT) -> dict[str, list[str]]:
    """Check prerequisites, accounting for outputs of earlier planned steps."""
    produced = set()
    missing = {}
    for step in steps:
        absent = [name for name in step.inputs if name not in produced and not (root / name).is_file()]
        if absent:
            missing[step.script] = absent
        produced.update(step.outputs)
    return missing


def execute_plan(steps: list[Step], root: Path = ROOT) -> int:
    missing = missing_inputs(steps, root)
    if missing:
        print("Missing prerequisites. Run earlier pipeline steps first:", file=sys.stderr)
        for script, names in missing.items():
            print(f"  {script}: {', '.join(names)}", file=sys.stderr)
        return 2
    env = dict(os.environ, PYTHONUTF8="1")
    for idx, step in enumerate(steps, start=1):
        print(f"[{idx}/{len(steps)}] {step.script}", flush=True)
        result = subprocess.run([sys.executable, str(root / "src" / step.script)], cwd=root, env=env)
        if result.returncode:
            print(f"FAILED: {step.script} (exit {result.returncode})", file=sys.stderr)
            return result.returncode if result.returncode > 0 else 1
        absent = [name for name in step.outputs if not (root / name).is_file()]
        if absent:
            print(f"FAILED: {step.script} did not produce {', '.join(absent)}", file=sys.stderr)
            return 1
    print(f"Pipeline complete: {len(steps)} steps succeeded.")
    return 0


def main(argv=None) -> int:
    valid_scripts = [step.script for step in build_plan()]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", choices=valid_scripts, help="Run only a specific script step (all steps by default)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Show every command without executing or downloading")
    mode.add_argument("--check", action="store_true", help="Check prerequisites for the selected plan without running it")
    args = parser.parse_args(argv)
    steps = build_plan(args.script)
    if args.dry_run:
        for idx, step in enumerate(steps, start=1):
            command = subprocess.list2cmdline([sys.executable, str(ROOT / "src" / step.script)])
            print(f"[{idx}/{len(steps)}] {command}")
        print("Plan only; no scripts were run. Use --check to validate prerequisites.")
        return 0
    if args.check:
        missing = missing_inputs(steps)
        for script, names in missing.items():
            print(f"MISSING {script}: {', '.join(names)}")
        if not missing:
            print(f"Prerequisites satisfied for {len(steps)} steps (including outputs of preceding planned steps).")
        return 2 if missing else 0
    try:
        return execute_plan(steps)
    except KeyboardInterrupt:
        print("Pipeline interrupted.", file=sys.stderr)
        return 130
    except OSError as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
