#!/usr/bin/env bash
# run_pipeline.sh — master sequential runner for the RadNet weather-aware
# anomaly detection pipeline (Phases 0-5), per HANDOFF.md section 3.
#
# Usage:
#   bash run_pipeline.sh              # run everything that has its inputs present
#   bash run_pipeline.sh --phase 4    # run only one phase block
#   bash run_pipeline.sh --dry-run    # print commands without executing
#
# Notes:
#   - Phases 0-1 require live network access (EPA RadNet + NOAA NCEI).
#   - Phase 3+ require the merged station files from Phase 1 and the labeled
#     datasets produced by synthetic_injection.py. If those intermediate files
#     are absent (they are not shipped in this archive), later phases are
#     skipped with an explicit SKIP message rather than failing.
#   - All commands run from the repository root.

set -uo pipefail
cd "$(dirname "$0")"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON="./.venv/bin/python"
  elif [[ -x "../.venv/bin/python" ]]; then
    PYTHON="../.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi
DRY_RUN=0
ONLY_PHASE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --phase) ONLY_PHASE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

run() {
  echo ""
  echo ">>> $*"
  if [[ $DRY_RUN -eq 1 ]]; then return 0; fi
  if ! "$PYTHON" "$@"; then
    echo "!!! FAILED: $*" >&2
    exit 1
  fi
}

# run_if <required-file-glob> <script...> : skip when required inputs are missing
run_if() {
  local pattern="$1"; shift
  if ! compgen -G "$pattern" > /dev/null; then
    echo ""
    echo ">>> SKIP $* (missing required input: $pattern)"
    return 0
  fi
  run "$@"
}

phase() {
  local n="$1"
  if [[ -n "$ONLY_PHASE" && "$ONLY_PHASE" != "$n" ]]; then return 1; fi
  echo ""
  echo "=================== PHASE $n ==================="
  return 0
}

phase 0 && {
  run src/parse_stations.py
  run src/download_pilot_data.py
  run src/analyze_pilot_stations.py
}

phase 1 && {
  run src/download_noaa_data.py
  run src/audit_noaa_precipitation.py
  run src/merge_radnet_weather.py
  run_if "data/processed/merged_*.csv.gz" src/plot_rain_events.py
}

phase 2 && {
  run_if "data/processed/merged_*.csv.gz" src/characterize_baseline_cycles.py
  run_if "data/processed/merged_*.csv.gz" src/analyze_filter_cycles_multi_station.py
  run_if "data/processed/merged_*.csv.gz" src/analyze_timestamp_lags.py
  run_if "data/processed/merged_*.csv.gz" src/analyze_rain_washout_distribution.py
  run_if "data/processed/merged_*.csv.gz" src/evaluate_baseline.py
  run_if "data/processed/merged_*.csv.gz" src/plot_baseline_timeline.py
}

phase 3 && {
  run_if "data/processed/merged_*.csv.gz" src/calibrate_dose_coupling.py
  run_if "data/processed/merged_*.csv.gz" src/analyze_washout_spectrum.py
  run_if "data/processed/merged_*.csv.gz" src/synthetic_injection.py
  run_if "data/processed/labeled_*_test.csv.gz" src/audit_injection_catalog.py
  run_if "data/processed/labeled_*_test.csv.gz" src/plot_synthetic_injections.py
}

phase 4 && {
  run_if "data/processed/labeled_*_test.csv.gz" src/train_and_evaluate_models.py
  run_if "data/processed/labeled_*_test.csv.gz" src/plot_model_evaluation.py
}

phase 5 && {
  run_if "data/processed/labeled_*_test.csv.gz" src/train_and_evaluate_rigorous.py
  run_if "data/processed/rigorous_benchmark_summary.csv" src/plot_rigorous_evaluation.py
  run_if "data/processed/labeled_*_test.csv.gz" src/evaluate_phase5_protocol.py
  run_if "data/processed/eval_benchmark_uncertainty_summary.csv" src/plot_phase5_evaluation.py
}

echo ""
echo "Pipeline run complete."
