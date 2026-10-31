#!/usr/bin/env python3
"""
scripts/check_release.py
Automated release verification script for RadNet Anomaly Detection.

Enforces:
1. Zero forbidden strings across the codebase (excluding .git, .venv, and tests/ assertions):
   - "Prompt "
   - "BUG-"
   - "TODO(author)"
   - "UNVERIFIED" (strictly allowing only approved operational exceptions)
2. Zero number drift between benchmark CSVs and README.md.
3. Zero unauthorized hardcoded results across all codebase scripts.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

    ignore_dirs = {".git", ".venv", "__pycache__", "tests", ".system_generated"}
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


def check_readme_consistency() -> list[str]:
    """Verify that README.md has no stale numbers and matches benchmark CSVs."""
    errors = []
    if not README_MD.exists():
        return [f"Missing {README_MD}"]

    readme_text = README_MD.read_text(encoding="utf-8")

    comp_csv = DATA_PROCESSED / "benchmark_realism_comparison.csv"
    if not comp_csv.exists():
        return [f"Missing {comp_csv}"]

    df_comp = pd.read_csv(comp_csv)
    r_t4_match = df_comp[df_comp["tier_name"].str.contains("Tier 4")]
    if r_t4_match.empty:
        errors.append("benchmark_realism_comparison.csv missing Tier 4 entry")
    else:
        r_t4 = r_t4_match.iloc[0]
        expected_t4_det = f"{r_t4['new_det_pct_mean']:.1f}%"
        expected_t4_fa = f"{r_t4['new_fa_per_yr_mean']:.2f}"
        if expected_t4_det not in readme_text:
            errors.append(f"README missing verified Tier 4 detection: {expected_t4_det}")
        if expected_t4_fa not in readme_text:
            errors.append(f"README missing verified Tier 4 FA rate: {expected_t4_fa}")

    # Check for prohibited stale figures in README
    if "0.0002 CPM" in readme_text or "0.0002~CPM" in readme_text:
        errors.append("README still contains contradicted 0.0002 CPM excess claim")

    return errors


def main() -> int:
    print("=" * 70)
    print("RadNet Anomaly Detection: Release Verification Suite (check_release.py)")
    print("=" * 70)

    all_errors = []

    print("[1/3] Scanning codebase for forbidden tokens (Prompt , BUG-, TODO(author), UNVERIFIED)...")
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

    print("[2/3] Verifying README numbers and benchmark consistency...")
    readme_errors = check_readme_consistency()
    if readme_errors:
        print(f"  FAILED: Found {len(readme_errors)} consistency errors:")
        for err in readme_errors:
            print(f"    - {err}")
        all_errors.extend(readme_errors)
    else:
        print("  PASSED: README numbers match underlying benchmark CSVs.")

    print("[3/3] Auditing codebase for hardcoded results and verified data reading...")
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
