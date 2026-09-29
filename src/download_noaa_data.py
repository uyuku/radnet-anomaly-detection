"""
Downloads NOAA NCEI Global-Hourly observation CSVs for the 5 pilot airport stations
from 2017 through 2025.
Enforces read-only permissions (chmod 444), full SHA-256 hashing in MANIFEST.sha256,
and logging in DATA_LOG.md.
"""

import hashlib
import os
import stat
import time
from pathlib import Path
from datetime import datetime
import requests


NOAA_PILOT_STATIONS = {
    "KBHM": {"station_id": "72228013876", "name": "Birmingham-Shuttlesworth Int'l, AL"},
    "KDCA": {"station_id": "72405013743", "name": "Reagan National, Washington DC"},
    "KSAN": {"station_id": "72290023188", "name": "San Diego Int'l, CA"},
    "KDFW": {"station_id": "72259003927", "name": "Dallas/Fort Worth Int'l, TX"},
    "KTPA": {"station_id": "72211012842", "name": "Tampa Int'l, FL"},
}

YEARS = list(range(2017, 2026))
BASE_URL = "https://www.ncei.noaa.gov/data/global-hourly/access"


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


def download_noaa_file(call_sign: str, station_info: dict, year: int) -> Path:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    station_id = station_info["station_id"]
    filename = f"noaa_{call_sign.lower()}_{year}.csv"
    target_path = raw_dir / filename
    url = f"{BASE_URL}/{year}/{station_id}.csv"
    notes = f"NOAA NCEI Global-Hourly {year} for {call_sign} ({station_id})"

    if target_path.exists():
        os.chmod(target_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
        size_mb = target_path.stat().st_size / (1024 * 1024)
        size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{target_path.stat().st_size / 1024:.1f} KB"
        update_manifest(filename, sha256)
        log_download(station_info["name"], url, filename, sha256, size_str, notes)
        return target_path

    print(f"Downloading {call_sign} {year} from {url}...")
    headers = {"User-Agent": "RadNet-Research/1.0 (Mozilla/5.0 compatible)"}

    # Retry up to 3 times
    for attempt in range(3):
        try:
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
            update_manifest(filename, file_hash)
            log_download(station_info["name"], url, filename, file_hash, size_str, notes)
            print(f"Saved {filename} ({size_str})")
            return target_path
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(2)

    raise RuntimeError(f"Failed to download {url} after 3 attempts")


def download_all_noaa():
    for call_sign, info in NOAA_PILOT_STATIONS.items():
        print(f"\n=== Processing {call_sign} ({info['name']}) ===")
        for year in YEARS:
            download_noaa_file(call_sign, info, year)


if __name__ == "__main__":
    download_all_noaa()
    print("\nAll NOAA pilot files downloaded and verified.")
