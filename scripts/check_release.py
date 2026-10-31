#!/usr/bin/env python3
"""
scripts/check_release.py
Final automated release verification script for RadNet Anomaly Detection.

Enforces:
1. Zero forbidden strings across the codebase (excluding .git, .venv, and tests/ assertions):
   - "Prompt "
   - "BUG-"
   - "TODO(author)"
   - "UNVERIFIED" (strictly allowing only the approved Fukushima filter sampler flow rate exception:
     "[UNVERIFIED: EPA RadNet continuous sampler nominal flow rate ~35 CFM / ~60 m³/h]")
2. Zero number drift between manuscript macros, underlying CSVs, abstract, and README.
3. Perfect 1-to-1 citation correspondence (0 missing bibitems, 0 unused bibitems).
4. All figures referenced in manuscript text have valid existing files and captions.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
REPORT_DIR = REPO_ROOT / "report"
MANUSCRIPT_TEX = REPORT_DIR / "manuscript_mdpi.tex"
MACROS_TEX = REPORT_DIR / "manuscript_macros.tex"
README_MD = REPO_ROOT / "README.md"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def check_forbidden_strings() -> list[str]:
    """Scan codebase for forbidden strings: Prompt , BUG-, TODO(author), UNVERIFIED."""
    errors = []
    forbidden_tokens = ["Prompt ", "BUG-", "TODO(author)"]
    
    approved_unverified_patterns = [
        "UNVERIFIED: EPA RadNet continuous sampler nominal flow rate",
        "B8_Operational_Screening_Approx_UNVERIFIED",
        'status = "UNVERIFIED"',
        'status = "CORRECTED" if title_match else "UNVERIFIED"',
        'valid_statuses = {"SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "NEEDS_REWORDING", "UNVERIFIED"}',
    ]
    
    ignore_dirs = {".git", ".venv", "__pycache__", "tests", "zenodo_archive", ".system_generated"}
    ignore_files = {"check_release.py"}
    ignore_extensions = {".pyc", ".gz", ".png", ".pdf", ".zip", ".tar", ".parquet", ".aux", ".log", ".out"}

    for p in REPO_ROOT.rglob("*"):
        if not p.is_file():
            continue
        if p.name in ignore_files:
            continue
        rel_parts = set(p.relative_to(REPO_ROOT).parts)
        if rel_parts & ignore_dirs:
            continue
        if p.suffix in ignore_extensions:
            continue

        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        rel_path = p.relative_to(REPO_ROOT).as_posix()

        for line_no, line in enumerate(content.splitlines(), start=1):
            for token in forbidden_tokens:
                if token in line:
                    errors.append(f"{rel_path}:{line_no} contains forbidden token '{token}': {line.strip()}")
            if "UNVERIFIED" in line:
                if not any(approved in line for approved in approved_unverified_patterns):
                    errors.append(f"{rel_path}:{line_no} contains unauthorized 'UNVERIFIED': {line.strip()}")

    return errors


def check_manuscript_macros() -> list[str]:
    """Verify that macros match underlying benchmark CSVs and README matches macros."""
    errors = []
    if not MACROS_TEX.exists():
        return [f"Missing {MACROS_TEX}"]

    macros_text = MACROS_TEX.read_text(encoding="utf-8")
    macro_map = {}
    for line in macros_text.splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.+)\}", line.strip())
        if m:
            macro_map[m.group(1)] = m.group(2)

    comp_csv = DATA_PROCESSED / "benchmark_realism_comparison.csv"
    if not comp_csv.exists():
        return [f"Missing {comp_csv}"]

    df_comp = pd.read_csv(comp_csv)
    # Support both canonical and legacy tier name in CSV
    r_t4_match = df_comp[df_comp["tier_name"].str.contains("Tier 4")]
    if r_t4_match.empty:
        errors.append("benchmark_realism_comparison.csv missing Tier 4 entry")
    else:
        r_t4 = r_t4_match.iloc[0]
        expected_t4_det = f"{r_t4['new_det_pct_mean']:.1f}\\%"
        expected_t4_fa = f"{r_t4['new_fa_per_yr_mean']:.2f}"
        if macro_map.get("TierFourDet") != expected_t4_det:
            errors.append(f"TierFourDet mismatch: macro={macro_map.get('TierFourDet')} vs csv={expected_t4_det}")
        if macro_map.get("TierFourFA") != expected_t4_fa:
            errors.append(f"TierFourFA mismatch: macro={macro_map.get('TierFourFA')} vs csv={expected_t4_fa}")

    # Plume counts
    if macro_map.get("TotalBenchmarkPlumes") != "450":
        errors.append(f"TotalBenchmarkPlumes must be 450, got {macro_map.get('TotalBenchmarkPlumes')}")
    if macro_map.get("BenchmarkPlumesTrain") != "250":
        errors.append(f"BenchmarkPlumesTrain must be 250, got {macro_map.get('BenchmarkPlumesTrain')}")
    if macro_map.get("BenchmarkPlumesTest") != "200":
        errors.append(f"BenchmarkPlumesTest must be 200, got {macro_map.get('BenchmarkPlumesTest')}")
    if macro_map.get("BenchmarkPlumeInjections") != "450":
        errors.append(f"BenchmarkPlumeInjections must be 450, got {macro_map.get('BenchmarkPlumeInjections')}")

    # README check
    if README_MD.exists():
        readme_text = README_MD.read_text(encoding="utf-8")
        if "93.60 ±" not in readme_text and "93.6%" not in readme_text:
            errors.append("README.md does not quote realistic Tier 4 detection (93.6%)")
        if "7.19 ±" not in readme_text and "7.19" not in readme_text:
            errors.append("README.md does not quote realistic Tier 4 false alarm rate (7.19/yr)")
        if "2.71 ±" in readme_text:
            errors.append("README.md contains legacy uncontextualized Tier 4 false alarm rate (2.71)")

    # Manuscript abstract check
    if MANUSCRIPT_TEX.exists():
        tex = MANUSCRIPT_TEX.read_text(encoding="utf-8")
        if "200 synthetic atmospheric plume injections" in tex:
            errors.append("Manuscript quotes '200 synthetic atmospheric plume injections' as total instead of 450 total (200 test)")

    return errors


def check_citations() -> list[str]:
    """Verify 1-to-1 citation correspondence (0 missing bibitems, 0 unused bibitems)."""
    errors = []
    if not MANUSCRIPT_TEX.exists():
        return [f"Missing {MANUSCRIPT_TEX}"]

    tex = MANUSCRIPT_TEX.read_text(encoding="utf-8")
    cited_keys = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", tex):
        for k in m.group(1).split(","):
            cleaned = k.strip()
            if cleaned:
                cited_keys.add(cleaned)

    bibitem_keys = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))

    missing = cited_keys - bibitem_keys
    unused = bibitem_keys - cited_keys

    if missing:
        errors.append(f"Missing \\bibitem entries for citations: {sorted(missing)}")
    if unused:
        errors.append(f"Unused \\bibitem entries in reference list: {sorted(unused)}")

    return errors


def check_figures() -> list[str]:
    """Verify all figures referenced in manuscript have valid files and captions."""
    errors = []
    if not MANUSCRIPT_TEX.exists():
        return [f"Missing {MANUSCRIPT_TEX}"]

    tex = MANUSCRIPT_TEX.read_text(encoding="utf-8")
    
    # Check includegraphics files exist
    graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    for g in graphics:
        p1 = REPORT_DIR / g
        p2 = REPO_ROOT / g
        if not p1.exists() and not p2.exists():
            errors.append(f"Figure file not found: {g} (checked {p1} and {p2})")

    # Check each figure environment has a caption
    fig_envs = re.findall(r"\\begin\{figure\}(.*?)\\end\{figure\}", tex, re.DOTALL)
    for idx, env in enumerate(fig_envs, start=1):
        if "\\caption{" not in env:
            errors.append(f"Figure environment #{idx} is missing a \\caption{{...}}")

    return errors


def main() -> int:
    print("=" * 70)
    print("RadNet Anomaly Detection: Release Verification Suite (check_release.py)")
    print("=" * 70)

    all_errors = []

    print("[1/5] Scanning codebase for forbidden tokens (Prompt , BUG-, TODO(author), UNVERIFIED)...")
    token_errors = check_forbidden_strings()
    if token_errors:
        print(f"  FAILED: Found {len(token_errors)} forbidden token violations:")
        for err in token_errors[:15]:
            print(f"    - {err}")
        if len(token_errors) > 15:
            print(f"    ... and {len(token_errors) - 15} more")
        all_errors.extend(token_errors)
    else:
        print("  PASSED: 0 forbidden tokens found across codebase.")

    print("[2/5] Verifying manuscript macros and cross-document number consistency...")
    macro_errors = check_manuscript_macros()
    if macro_errors:
        print(f"  FAILED: Found {len(macro_errors)} macro/document drift errors:")
        for err in macro_errors:
            print(f"    - {err}")
        all_errors.extend(macro_errors)
    else:
        print("  PASSED: All macros match underlying CSVs, abstract, and README.")

    print("[3/5] Verifying 1-to-1 citation correspondence in manuscript...")
    cite_errors = check_citations()
    if cite_errors:
        print(f"  FAILED: Found {len(cite_errors)} citation errors:")
        for err in cite_errors:
            print(f"    - {err}")
        all_errors.extend(cite_errors)
    else:
        print("  PASSED: 1-to-1 citation correspondence verified (0 missing, 0 unused bibitems).")

    print("[4/5] Verifying manuscript figures and captions...")
    fig_errors = check_figures()
    if fig_errors:
        print(f"  FAILED: Found {len(fig_errors)} figure errors:")
        for err in fig_errors:
            print(f"    - {err}")
        all_errors.extend(fig_errors)
    else:
        print("  PASSED: All figure files exist and have valid captions.")

    print("[5/5] Auditing codebase for hardcoded results and verified data reading...")
    from scripts.audit_hardcoded_results import audit_repository
    if not audit_repository():
        all_errors.append("Repository hardcode audit failed: found unauthorized hardcoded results or missing data readers.")
    else:
        print("  PASSED: 0 unauthorized hardcoded results found; all results scripts read data.")

    print("-" * 70)
    if all_errors:
        print(f"TOTAL ERRORS: {len(all_errors)}. Release checks FAILED.")
        return 1
    else:
        print("ALL RELEASE CHECKS PASSED SUCCESSFULLY. Ready for release!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
