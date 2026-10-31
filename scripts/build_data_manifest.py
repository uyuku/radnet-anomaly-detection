"""
Builds data/DATA_MANIFEST.csv listing every input and derived dataset in the repository.
Computes sha256, row count, date range, columns, and source metadata.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import os
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_SOURCES = {
    "al_birmingham": {
        "url": "https://www.epa.gov/system/files/other-files/2026-02/al_birmingham_2025-2007.zip",
        "agency": "US EPA RadNet",
        "retrieval_date": "2026-09-28",
    },
    "dc_washington": {
        "url": "https://www.epa.gov/system/files/other-files/2026-02/dc_washington_2025-2006.zip",
        "agency": "US EPA RadNet",
        "retrieval_date": "2026-09-28",
    },
    "ca_san_diego": {
        "url": "https://www.epa.gov/system/files/other-files/2026-02/ca_san_diego_2025-2006.zip",
        "agency": "US EPA RadNet",
        "retrieval_date": "2026-09-28",
    },
    "tx_dallas": {
        "url": "https://www.epa.gov/system/files/other-files/2026-02/tx_dallas_2025-2007.zip",
        "agency": "US EPA RadNet",
        "retrieval_date": "2026-09-28",
    },
    "fl_tampa": {
        "url": "https://www.epa.gov/system/files/other-files/2026-02/fl_tampa_2025-2008.zip",
        "agency": "US EPA RadNet",
        "retrieval_date": "2026-09-28",
    },
    "az_phoenix": {
        "url": "https://www.epa.gov/system/files/other-files/2026-02/az_phoenix_2025-2007.zip",
        "agency": "US EPA RadNet",
        "retrieval_date": "2026-10-31",
    },
    "ne_omaha": {
        "url": "https://www.epa.gov/system/files/other-files/2026-02/ne_omaha_2025-2008.zip",
        "agency": "US EPA RadNet",
        "retrieval_date": "2026-10-31",
    },
    "hi_honolulu": {
        "url": "https://www.epa.gov/system/files/other-files/2026-02/hi_honolulu_2025-2008.zip",
        "agency": "US EPA RadNet",
        "retrieval_date": "2026-10-31",
    },
    "ak_anchorage": {
        "url": "https://www.epa.gov/system/files/other-files/2026-02/ak_anchorage_2025-2008.zip",
        "agency": "US EPA RadNet",
        "retrieval_date": "2026-10-31",
    },
    "la_baton_rouge": {
        "url": "https://www.epa.gov/system/files/other-files/2026-02/la_baton_rouge_2025-2008.zip",
        "agency": "US EPA RadNet",
        "retrieval_date": "2026-10-31",
    },
    "vt_burlington": {
        "url": "https://www.epa.gov/system/files/other-files/2026-02/vt_burlington_2025-2009.zip",
        "agency": "US EPA RadNet",
        "retrieval_date": "2026-10-31",
    },
}


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def inspect_dataset(path: Path) -> dict:
    fname = path.name
    sha256 = hash_file(path)
    size_bytes = path.stat().st_size

    row_count = "N/A"
    date_range = "N/A"
    columns = "N/A"
    col_count = 0

    try:
        if fname.endswith(".csv.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                df = pd.read_csv(f, nrows=100)
                # count rows
                f.seek(0)
                total_lines = sum(1 for _ in f) - 1
                row_count = str(total_lines)
                col_count = len(df.columns)
                columns = "; ".join(df.columns.tolist()[:10]) + ("..." if len(df.columns) > 10 else "")
                dt_col = next((c for c in ["dt", "utc_hour", "SAMPLE COLLECTION TIME", "DATE"] if c in df.columns), None)
                if dt_col:
                    f.seek(0)
                    full_df = pd.read_csv(f, usecols=[dt_col])
                    dts = pd.to_datetime(full_df[dt_col], errors="coerce").dropna()
                    if len(dts) > 0:
                        date_range = f"{dts.min().strftime('%Y-%m-%d')} to {dts.max().strftime('%Y-%m-%d')}"
        elif fname.endswith(".csv"):
            with open(path, "rt", encoding="utf-8", errors="replace") as f:
                df = pd.read_csv(f, nrows=100)
                f.seek(0)
                total_lines = sum(1 for _ in f) - 1
                row_count = str(total_lines)
                col_count = len(df.columns)
                columns = "; ".join(df.columns.tolist()[:10]) + ("..." if len(df.columns) > 10 else "")
                dt_col = next((c for c in ["dt", "utc_hour", "SAMPLE COLLECTION TIME", "DATE"] if c in df.columns), None)
                if dt_col:
                    f.seek(0)
                    full_df = pd.read_csv(f, usecols=[dt_col])
                    dts = pd.to_datetime(full_df[dt_col], errors="coerce").dropna()
                    if len(dts) > 0:
                        date_range = f"{dts.min().strftime('%Y-%m-%d')} to {dts.max().strftime('%Y-%m-%d')}"
        elif fname.endswith(".zip"):
            row_count = "Archived ZIP"
            date_range = "Multi-year annual archives"
            columns = "Raw EPA CSV members"
        elif fname.endswith(".json"):
            row_count = "JSON Dictionary"
            date_range = "N/A"
            columns = "Key-value schema"
    except Exception as exc:
        columns = f"Error reading: {exc}"

    return {
        "file_name": str(path.relative_to(REPO_ROOT)),
        "file_size_bytes": size_bytes,
        "row_count": row_count,
        "column_count": col_count,
        "date_range": date_range,
        "columns_preview": columns,
        "sha256": sha256,
    }


def main():
    print("Building data/DATA_MANIFEST.csv...")
    records = []

    # 1. Raw Files
    raw_dir = REPO_ROOT / "data" / "raw"
    if raw_dir.exists():
        for p in sorted(raw_dir.glob("*")):
            if p.is_file() and not p.name.startswith("."):
                info = inspect_dataset(p)
                info["dataset_tier"] = "raw_external"

                # Assign agency and source url
                if "noaa_" in p.name:
                    info["source_agency"] = "NOAA NCEI"
                    info["source_url"] = "https://noaa-global-hourly-pds.s3.amazonaws.com/"
                    info["retrieval_date"] = "2026-09-29 / 2026-10-31"
                    info["license"] = "Public Domain (17 U.S.C. § 105)"
                elif p.name.endswith(".zip"):
                    st_id = p.name.split("_2025")[0]
                    meta = RAW_SOURCES.get(st_id, {})
                    info["source_agency"] = meta.get("agency", "US EPA RadNet")
                    info["source_url"] = meta.get("url", "https://www.epa.gov/radnet/radnet-csv-file-downloads")
                    info["retrieval_date"] = meta.get("retrieval_date", "2026-09-28")
                    info["license"] = "Public Domain (17 U.S.C. § 105)"
                else:
                    info["source_agency"] = "US EPA RadNet"
                    info["source_url"] = "https://www.epa.gov/radnet/radnet-csv-file-downloads"
                    info["retrieval_date"] = "2026-09-28"
                    info["license"] = "Public Domain (17 U.S.C. § 105)"

                records.append(info)

    # 2. Processed Datasets
    proc_dir = REPO_ROOT / "data" / "processed"
    if proc_dir.exists():
        for p in sorted(proc_dir.glob("*")):
            if p.is_file() and not p.name.startswith("."):
                info = inspect_dataset(p)
                info["dataset_tier"] = "derived_processed"
                info["source_agency"] = "Generated by Project Pipeline"
                info["source_url"] = "https://github.com/TODO_AUTHOR/radnet-anomaly-detection"
                info["retrieval_date"] = "2026-10-31"
                info["license"] = "MIT / CC-BY-4.0"
                records.append(info)

    # Export to data/DATA_MANIFEST.csv
    out_csv = REPO_ROOT / "data" / "DATA_MANIFEST.csv"
    fieldnames = [
        "file_name",
        "dataset_tier",
        "source_agency",
        "source_url",
        "retrieval_date",
        "license",
        "file_size_bytes",
        "row_count",
        "column_count",
        "date_range",
        "sha256",
        "columns_preview",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"Generated {out_csv} with {len(records)} cataloged datasets.")


if __name__ == "__main__":
    main()
