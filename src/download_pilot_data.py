"""
Download and verification utility for RadNet and NOAA data.
Ensures files are saved to data/raw/, made read-only (chmod 444),
hashed (full SHA-256), recorded in data/raw/MANIFEST.sha256,
and logged in DATA_LOG.md per Section 2 Operating Rules.
"""

import hashlib
import os
import stat
from pathlib import Path
from datetime import datetime
import requests


# Master inventory of raw data targets
DOWNLOAD_TARGETS = [
    # Station index HTML
    {
        "station": "All Stations (EPA Page)",
        "url": "https://www.epa.gov/radnet/radnet-csv-file-downloads",
        "filename": "radnet_csv_file_downloads.html",
        "notes": "RadNet official CSV downloads index page with all stations, start dates, and ZIP/CSV links",
    },
    # 2026 YTD CSV for Birmingham
    {
        "station": "AL: BIRMINGHAM",
        "url": "https://radnet.epa.gov/cdx-radnet-rest/api/rest/csv/2026/fixed/AL/BIRMINGHAM",
        "filename": "AL_BIRMINGHAM_2026.csv",
        "notes": "2026 YTD hourly near-real-time observations",
    },
    # Historical archives for pilot / candidate stations
    {
        "station": "Birmingham, AL",
        "url": "https://www.epa.gov/system/files/other-files/2026-02/al_birmingham_2025-2007.zip",
        "filename": "al_birmingham_2025-2007.zip",
        "notes": "Complete historical archive 2007-2025 (19 annual CSV files)",
    },
    {
        "station": "Washington, DC",
        "url": "https://www.epa.gov/system/files/other-files/2026-02/dc_washington_2025-2006.zip",
        "filename": "dc_washington_2025-2006.zip",
        "notes": "Historical archive 2006-2025 (20 annual CSVs)",
    },
    {
        "station": "San Diego, CA",
        "url": "https://www.epa.gov/system/files/other-files/2026-02/ca_san_diego_2025-2006.zip",
        "filename": "ca_san_diego_2025-2006.zip",
        "notes": "Historical archive 2006-2025 (20 annual CSVs)",
    },
    {
        "station": "Dallas, TX",
        "url": "https://www.epa.gov/system/files/other-files/2026-02/tx_dallas_2025-2007.zip",
        "filename": "tx_dallas_2025-2007.zip",
        "notes": "Historical archive 2007-2025 (19 annual CSVs)",
    },
    {
        "station": "Tampa, FL",
        "url": "https://www.epa.gov/system/files/other-files/2026-02/fl_tampa_2025-2008.zip",
        "filename": "fl_tampa_2025-2008.zip",
        "notes": "Historical archive 2008-2025 (18 annual CSVs)",
    },
    {
        "station": "Montgomery, AL",
        "url": "https://www.epa.gov/system/files/other-files/2026-02/al_montgomery_2025-2006.zip",
        "filename": "al_montgomery_2025-2006.zip",
        "notes": "Historical archive 2006-2025 (20 annual CSVs)",
    },
    {
        "station": "Chicago, IL",
        "url": "https://www.epa.gov/system/files/other-files/2026-02/il_chicago_2025-2006.zip",
        "filename": "il_chicago_2025-2006.zip",
        "notes": "Historical archive 2006-2025 (20 annual CSVs)",
    },
    {
        "station": "Austin, TX",
        "url": "https://www.epa.gov/system/files/other-files/2026-02/tx_austin_2025-2007.zip",
        "filename": "tx_austin_2025-2007.zip",
        "notes": "Historical archive 2007-2025 (19 annual CSVs)",
    },
    # NOAA verification file
    {
        "station": "Birmingham, AL (NOAA KBHM)",
        "url": "https://www.ncei.noaa.gov/data/global-hourly/access/2024/72228013876.csv",
        "filename": "noaa_kbhm_2024.csv",
        "notes": "NOAA NCEI Global-Hourly 2024 for KBHM (72228013876)",
    },
]


def update_manifest(filename: str, sha256_hash: str):
    manifest_path = Path("data/raw/MANIFEST.sha256")
    manifest = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    manifest[parts[1]] = parts[0]

    manifest[filename] = sha256_hash
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("# SHA256 Manifest of Raw Downloaded Files\n")
        for fn in sorted(manifest.keys()):
            f.write(f"{manifest[fn]}  {fn}\n")


def log_download(station: str, url: str, filename: str, file_hash: str, file_size_str: str, notes: str):
    log_file = Path("DATA_LOG.md")
    today = datetime.now().strftime("%Y-%m-%d")
    short_hash = f"{file_hash[:8]}..."
    entry = f"| {today} | {station} | {url} | {filename} | {short_hash} ({file_size_str}) | {notes} |\n"

    if log_file.exists():
        content = log_file.read_text(encoding="utf-8")
        if filename in content:
            return
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry)
    else:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("# Data Download Log\n\nFull SHA256 hashes are recorded in `data/raw/MANIFEST.sha256`.\n\n| Date Retrieved | Station | URL | File Name | SHA256 Prefix (Size) | Notes |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            f.write(entry)


def download_raw_file(station: str, url: str, target_filename: str, notes: str) -> Path:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = raw_dir / target_filename

    if target_path.exists():
        os.chmod(target_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
        size_mb = target_path.stat().st_size / (1024 * 1024)
        size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{target_path.stat().st_size / 1024:.1f} KB"
        update_manifest(target_filename, sha256)
        log_download(station, url, target_filename, sha256, size_str, notes)
        return target_path

    print(f"Downloading {url} to {target_path}...")
    headers = {"User-Agent": "RadNet-Research/1.0 (Mozilla/5.0 compatible)"}
    resp = requests.get(url, headers=headers, stream=True, timeout=60)
    resp.raise_for_status()

    temp_path = target_path.with_suffix(".tmp")
    hasher = hashlib.sha256()
    with open(temp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                hasher.update(chunk)

    temp_path.rename(target_path)
    os.chmod(target_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    file_hash = hasher.hexdigest()
    size_mb = target_path.stat().st_size / (1024 * 1024)
    size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{target_path.stat().st_size / 1024:.1f} KB"

    update_manifest(target_filename, file_hash)
    log_download(station, url, target_filename, file_hash, size_str, notes)
    return target_path


def download_all_targets():
    for target in DOWNLOAD_TARGETS:
        download_raw_file(target["station"], target["url"], target["filename"], target["notes"])


if __name__ == "__main__":
    download_all_targets()
    print("All raw targets verified, hashed, and logged.")
