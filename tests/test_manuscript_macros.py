"""
tests/test_manuscript_macros.py
Tests that manuscript macros exist, match the underlying CSV values,
and ensure zero number drift between data and manuscript.
"""

from pathlib import Path
import re
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "report"
MACROS_PATH = REPORT_DIR / "manuscript_macros.tex"
TABLES_DIR = REPORT_DIR / "tables"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def test_macros_file_exists():
    assert MACROS_PATH.exists(), "report/manuscript_macros.tex must exist"
    content = MACROS_PATH.read_text(encoding="utf-8")
    assert len(content) > 100, "manuscript_macros.tex must not be empty"


def test_tables_exist():
    expected_tables = [
        "table1_benchmark_summary.tex",
        "table2_fa_budget_operating_points.tex",
        "table3_loso_11stations.tex",
        "table4_alarm_adjudication.tex",
        "table5_real_event_fukushima.tex",
        "table6_ablation_summary.tex",
    ]
    for tbl in expected_tables:
        path = TABLES_DIR / tbl
        assert path.exists(), f"Expected table {tbl} does not exist in report/tables/"
        assert path.stat().st_size > 50, f"Table {tbl} is suspiciously empty"


def test_macro_values_match_underlying_csvs():
    """Verify that macros match the exact data in benchmark_realism_comparison.csv."""
    macros_content = MACROS_PATH.read_text(encoding="utf-8")
    macro_map = {}
    for line in macros_content.splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.+)\}", line.strip())
        if m:
            macro_map[m.group(1)] = m.group(2)

    comp_csv = DATA_PROCESSED / "benchmark_realism_comparison.csv"
    assert comp_csv.exists(), "benchmark_realism_comparison.csv must exist"
    df_comp = pd.read_csv(comp_csv)

    r_t4 = df_comp[df_comp["tier_name"] == "Tier 4 (Physics-Informed Nuclear)"].iloc[0]
    assert macro_map["TierFourDet"] == f"{r_t4['new_det_pct_mean']:.1f}\\%"
    assert macro_map["TierFourFA"] == f"{r_t4['new_fa_per_yr_mean']:.2f}"

    r_t3 = df_comp[df_comp["tier_name"] == "Tier 3 (Weather Fusion)"].iloc[0]
    assert macro_map["TierThreeDet"] == f"{r_t3['new_det_pct_mean']:.1f}\\%"
    assert macro_map["TierThreeFA"] == f"{r_t3['new_fa_per_yr_mean']:.2f}"

    r_t2 = df_comp[df_comp["tier_name"] == "Tier 2 (Spectral)"].iloc[0]
    assert macro_map["TierTwoDet"] == f"{r_t2['new_det_pct_mean']:.1f}\\%"
    assert macro_map["TierTwoFA"] == f"{r_t2['new_fa_per_yr_mean']:.2f}"

    r_t1 = df_comp[df_comp["tier_name"] == "Tier 1 (Gross)"].iloc[0]
    assert macro_map["TierOneDet"] == f"{r_t1['new_det_pct_mean']:.1f}\\%"
    assert macro_map["TierOneFA"] == f"{r_t1['new_fa_per_yr_mean']:.2f}"

    # Plume counts
    assert macro_map["TotalBenchmarkPlumes"] == "450"
    assert macro_map["BenchmarkPlumesTrain"] == "250"
    assert macro_map["BenchmarkPlumesTest"] == "200"
    assert macro_map["BenchmarkPlumeInjections"] == "450"

    adj_csv = DATA_PROCESSED / "hand_labeled_test_alarms_summary.csv"
    if adj_csv.exists():
        df_adj = pd.read_csv(adj_csv)
        r_rain = df_adj[df_adj["adjudicated_category"] == "rain_coincident"].iloc[0]
        assert macro_map["AdjudicatedRainPct"] == f"{r_rain['percentage']:.1f}\\%"


def test_no_number_drift_between_macros_and_documents():
    """Verify that numbers in the abstract, README, and Results match the macros exactly."""
    macros_content = MACROS_PATH.read_text(encoding="utf-8")
    macro_map = {}
    for line in macros_content.splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.+)\}", line.strip())
        if m:
            macro_map[m.group(1)] = m.group(2)

    # 1. Total plume count must be 450 (250 train, 200 test)
    assert macro_map["BenchmarkPlumeInjections"] == "450"
    assert macro_map["TotalBenchmarkPlumes"] == "450"
    assert macro_map["BenchmarkPlumesTrain"] == "250"
    assert macro_map["BenchmarkPlumesTest"] == "200"

    # 2. Manuscript Abstract check
    manuscript_text = (REPORT_DIR / "manuscript_mdpi.tex").read_text(encoding="utf-8")
    start = manuscript_text.find(r"\abstract{") + len(r"\abstract{")
    assert start > len(r"\abstract{"), "Abstract not found in manuscript"
    depth = 1
    end = start
    while end < len(manuscript_text) and depth > 0:
        if manuscript_text[end] == "{":
            depth += 1
        elif manuscript_text[end] == "}":
            depth -= 1
        end += 1
    abstract_text = manuscript_text[start:end-1]

    # Abstract must NOT quote 200 as if it were the total
    assert "200 synthetic atmospheric plume injections" not in abstract_text
    assert r"\BenchmarkPlumeInjections{} synthetic atmospheric plume injections" in abstract_text or "450 synthetic" in abstract_text

    # Abstract must use TierFourDet and TierFourFA macros
    assert r"\TierFourDet{}" in abstract_text or r"\TierFourDet" in abstract_text
    assert r"\TierFourFA{}" in abstract_text or r"\TierFourFA" in abstract_text
    assert "94.5%" not in abstract_text
    assert "2.71" not in abstract_text

    # 3. README check
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    t4_fa_val = macro_map["TierFourFA"]
    # Main README headline must quote realistic benchmark numbers
    assert f"{float(t4_fa_val):.2f} ±" in readme_text or t4_fa_val in readme_text
    assert "93.60 ±" in readme_text
    # Legacy headline numbers must not appear uncontextualized
    assert "2.71 ±" not in readme_text
    assert "94.00 ±" not in readme_text
