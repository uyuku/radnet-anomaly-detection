"""
Unit tests for Reproducibility, Data Manifest, and Archive Completeness.
"""

import csv
import subprocess
import sys
from pathlib import Path
import pytest


def test_readme_claims_match_generated_csvs():
    """Ensures all numbers quoted in README match generated CSVs via recompute_claims.py."""
    result = subprocess.run(
        [sys.executable, "recompute_claims.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"recompute_claims.py failed:\n{result.stderr}\n{result.stdout}"


def test_data_manifest_integrity():
    """Verifies that DATA_MANIFEST.csv exists and catalogs processed datasets."""
    manifest_path = Path("data/DATA_MANIFEST.csv")
    assert manifest_path.exists(), "data/DATA_MANIFEST.csv is missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 50, f"Expected >50 cataloged files, got {len(rows)}"

    file_names = {r["file_name"] for r in rows}
    assert "data/processed/eval_benchmark_uncertainty_summary.csv" in file_names
    assert "data/processed/synthetic_injection_catalog.csv" in file_names
    assert "data/processed/extended_loso_11station_summary.csv" in file_names
    assert "data/processed/real_event_fukushima_station_summary.csv" in file_names
    assert "data/processed/hand_labeled_test_alarms_summary.csv" in file_names

    # Check that sha256 hashes are valid 64-char hex strings
    for r in rows:
        assert len(r["sha256"]) == 64, f"Invalid SHA-256 in manifest row {r['file_name']}"


def test_zenodo_archive_completeness():
    """Verifies that zenodo_archive/ directory contains required archive files."""
    zenodo_dir = Path("zenodo_archive")
    assert zenodo_dir.exists(), "zenodo_archive directory missing"
    assert (zenodo_dir / "README.md").exists()
    assert (zenodo_dir / "data/DATA_MANIFEST.csv").exists()
    assert (zenodo_dir / "data/processed/eval_benchmark_uncertainty_summary.csv").exists()
    assert (zenodo_dir / "data/processed/synthetic_injection_catalog.csv").exists()
    assert (zenodo_dir / "docs/hand_labeled_test_alarms.csv").exists()


def test_reproducibility_documentation_and_license_deliverables():
    """Verifies LICENSE, CITATION.cff, CHANGELOG, and open science deliverables exist."""
    required = [
        Path("LICENSE"),
        Path("CITATION.cff"),
        Path("CHANGELOG.md"),
        Path("Dockerfile"),
        Path("environment.yml"),
        Path(".github/workflows/ci.yml"),
        Path("docs/data_licenses_and_terms.md"),
        Path("docs/zenodo_upload_instructions.md"),
    ]
    for p in required:
        assert p.exists(), f"Missing required file: {p}"
        assert p.stat().st_size > 0, f"File {p} is empty"
