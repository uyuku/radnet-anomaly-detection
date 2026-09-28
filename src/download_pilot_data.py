"""
Download utility for RadNet historical archives and observations.
Ensures files are saved to data/raw/, made read-only (chmod 444),
hashed (SHA256), and logged in DATA_LOG.md per Section 2 Operating Rules.
"""

import hashlib
import os
import stat
from pathlib import Path
from datetime import datetime
import requests


def log_download(station: str, url: str, filename: str, file_hash: str, file_size_str: str, notes: str):
    log_file = Path("DATA_LOG.md")
    today = datetime.now().strftime("%Y-%m-%d")
    entry = f"| {today} | {station} | {url} | {filename} | SHA256: {file_hash[:8]}... ({file_size_str}) | {notes} |\n"

    # Check if already logged
    if log_file.exists():
        content = log_file.read_text(encoding="utf-8")
        if filename in content:
            print(f"Already logged {filename} in DATA_LOG.md")
            return
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry)
    else:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("# Data Download Log\n\n| Date Retrieved | Station | URL | File Name | File Hash / Size | Notes |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            f.write(entry)
    print(f"Logged {filename} in DATA_LOG.md")


def download_raw_file(station: str, url: str, target_filename: str, notes: str) -> Path:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = raw_dir / target_filename

    if target_path.exists():
        print(f"File {target_path} already exists. Verifying permissions and logging.")
        # Ensure read-only
        os.chmod(target_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        # Compute hash
        sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
        size_mb = target_path.stat().st_size / (1024 * 1024)
        size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{target_path.stat().st_size / 1024:.1f} KB"
        log_download(station, url, target_filename, sha256, size_str, notes)
        return target_path

    print(f"Downloading {url} to {target_path}...")
    headers = {"User-Agent": "RadNet-Research/1.0 (Mozilla/5.0 compatible)"}
    resp = requests.get(url, headers=headers, stream=True, timeout=60)
    resp.raise_for_status()

    # Write temporary first
    temp_path = target_path.with_suffix(".tmp")
    hasher = hashlib.sha256()
    with open(temp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                hasher.update(chunk)

    temp_path.rename(target_path)
    # Set read-only (chmod 444)
    os.chmod(target_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    file_hash = hasher.hexdigest()
    size_mb = target_path.stat().st_size / (1024 * 1024)
    size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{target_path.stat().st_size / 1024:.1f} KB"

    log_download(station, url, target_filename, file_hash, size_str, notes)
    return target_path


if __name__ == "__main__":
    pilot_targets = [
        ("San Diego, CA", "https://www.epa.gov/system/files/other-files/2026-02/ca_san_diego_2025-2006.zip", "ca_san_diego_2025-2006.zip", "Historical archive 2006-2025 (20 annual CSVs)"),
        ("Tampa, FL", "https://www.epa.gov/system/files/other-files/2026-02/fl_tampa_2025-2008.zip", "fl_tampa_2025-2008.zip", "Historical archive 2008-2025 (18 annual CSVs)"),
        ("Chicago, IL", "https://www.epa.gov/system/files/other-files/2026-02/il_chicago_2025-2006.zip", "il_chicago_2025-2006.zip", "Historical archive 2006-2025 (20 annual CSVs)"),
        ("Austin, TX", "https://www.epa.gov/system/files/other-files/2026-02/tx_austin_2025-2007.zip", "tx_austin_2025-2007.zip", "Historical archive 2007-2025 (19 annual CSVs)"),
    ]

    for station, url, filename, notes in pilot_targets:
        download_raw_file(station, url, filename, notes)
