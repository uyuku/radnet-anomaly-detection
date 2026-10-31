"""
Downloads, merges, and audits the 6 pre-registered extended RadNet and NOAA
stations from `docs/station_selection.md`.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests

from src.download_pilot_data import log_download, update_manifest
from src.generalization_tampa import audit_station_raw_data_quality
from src.merge_radnet_weather import CHANNEL_KEYS, YEARS, load_and_clean_noaa

EXTENDED_CANDIDATES = [
    {
        "id": "az_phoenix",
        "name": "Phoenix, AZ",
        "climate_regime": "Arid Interior (BWh)",
        "radnet_url": "https://www.epa.gov/system/files/other-files/2026-02/az_phoenix_2025-2007.zip",
        "radnet_zip": "data/raw/az_phoenix_2025-2007.zip",
        "city_prefix": "AZ_PHOENIX",
        "noaa_call": "KPHX",
        "noaa_id": "72278023183",
        "role": "primary",
    },
    {
        "id": "ne_omaha",
        "name": "Omaha, NE",
        "climate_regime": "Cold Continental (Dfa)",
        "radnet_url": "https://www.epa.gov/system/files/other-files/2026-02/ne_omaha_2025-2008.zip",
        "radnet_zip": "data/raw/ne_omaha_2025-2008.zip",
        "city_prefix": "NE_OMAHA",
        "noaa_call": "KOMA",
        "noaa_id": "72550014942",
        "role": "primary",
    },
    {
        "id": "hi_honolulu",
        "name": "Honolulu, HI",
        "climate_regime": "Tropical Trade-Wind (As)",
        "radnet_url": "https://www.epa.gov/system/files/other-files/2026-02/hi_honolulu_2025-2008.zip",
        "radnet_zip": "data/raw/hi_honolulu_2025-2008.zip",
        "city_prefix": "HI_HONOLULU",
        "noaa_call": "PHNL",
        "noaa_id": "91182022521",
        "role": "primary",
    },
    {
        "id": "ak_anchorage",
        "name": "Anchorage, AK",
        "climate_regime": "High-Latitude Subarctic (Dfc)",
        "radnet_url": "https://www.epa.gov/system/files/other-files/2026-02/ak_anchorage_2025-2008.zip",
        "radnet_zip": "data/raw/ak_anchorage_2025-2008.zip",
        "city_prefix": "AK_ANCHORAGE",
        "noaa_call": "PANC",
        "noaa_id": "70273026451",
        "role": "primary",
    },
    {
        "id": "la_baton_rouge",
        "name": "Baton Rouge, LA",
        "climate_regime": "High-Rainfall Coastal (Cfa)",
        "radnet_url": "https://www.epa.gov/system/files/other-files/2026-02/la_baton_rouge_2025-2008.zip",
        "radnet_zip": "data/raw/la_baton_rouge_2025-2008.zip",
        "city_prefix": "LA_BATON_ROUGE",
        "noaa_call": "KBTR",
        "noaa_id": "72232013970",
        "role": "primary",
    },
    {
        "id": "vt_burlington",
        "name": "Burlington, VT",
        "climate_regime": "Northern Cold Continental (Dfb)",
        "radnet_url": "https://www.epa.gov/system/files/other-files/2026-02/vt_burlington_2025-2009.zip",
        "radnet_zip": "data/raw/vt_burlington_2025-2009.zip",
        "city_prefix": "VT_BURLINGTON",
        "noaa_call": "KBTV",
        "noaa_id": "72617014742",
        "role": "primary",
    },
]


def _download_file(url: str, target_path: Path, station_label: str, notes: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 100:
        # Guard against HTML redirect pages saved as .csv
        is_bad_html = False
        if target_path.suffix == ".csv":
            with open(target_path, "rb") as f:
                head_bytes = f.read(64)
            if b"<!DOCTYPE" in head_bytes or b"<html" in head_bytes:
                is_bad_html = True
                os.chmod(target_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
                target_path.unlink()
        if not is_bad_html:
            sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
            size_mb = target_path.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{target_path.stat().st_size / 1024:.1f} KB"
            update_manifest(target_path.name, sha256)
            log_download(station_label, url, target_path.name, sha256, size_str, notes)
            return

    headers = {"User-Agent": "RadNet-Research/1.0"}
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=60)
            resp.raise_for_status()
            tmp = target_path.with_suffix(".tmp")
            hasher = hashlib.sha256()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        hasher.update(chunk)
            tmp.rename(target_path)
            os.chmod(target_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            file_hash = hasher.hexdigest()
            size_mb = target_path.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{target_path.stat().st_size / 1024:.1f} KB"
            update_manifest(target_path.name, file_hash)
            log_download(station_label, url, target_path.name, file_hash, size_str, notes)
            print(f"  Downloaded {target_path.name} ({size_str})")
            return
        except Exception as exc:
            print(f"  Attempt {attempt + 1} failed for {url}: {exc}")
            time.sleep(2)
    raise RuntimeError(f"Failed to download {url}")


def _load_and_clean_radnet_flexible(station: dict) -> pd.DataFrame:
    dfs = []
    zip_path = Path(station["radnet_zip"])
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        for yr in YEARS:
            yr_str = f"_{yr}.csv"
            matched = [n for n in names if n.lower().endswith(yr_str)]
            if matched:
                with z.open(matched[0]) as f:
                    dfs.append(pd.read_csv(f, low_memory=False))
    if not dfs:
        raise ValueError(f"No annual 2017-2025 CSVs found inside {zip_path}")
    df = pd.concat(dfs, ignore_index=True)

    df["dt_raw"] = pd.to_datetime(df["SAMPLE COLLECTION TIME"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["dt_raw"])
    df["utc_hour"] = df["dt_raw"].dt.round("h")
    df = df.drop_duplicates(subset=["utc_hour"]).copy()

    chan_cols_map = {k: [c for c in df.columns if k in c][0] for k in CHANNEL_KEYS}
    chan_cols = [chan_cols_map[k] for k in CHANNEL_KEYS]
    for c in chan_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["rad_complete_channels"] = df[chan_cols].notna().all(axis=1)
    df["gross_cpm"] = np.nan
    df.loc[df["rad_complete_channels"], "gross_cpm"] = df.loc[df["rad_complete_channels"], chan_cols].sum(axis=1)

    dose_col = "DOSE EQUIVALENT RATE (nSv/h)"
    df["dose_rate_nsvh"] = pd.to_numeric(df[dose_col], errors="coerce") if dose_col in df.columns else np.nan

    rename_dict = {chan_cols_map[k]: f"cpm_{k.lower()}" for k in CHANNEL_KEYS}
    df = df.rename(columns=rename_dict)
    keep_cols = [
        "utc_hour",
        "rad_complete_channels",
        "gross_cpm",
        "dose_rate_nsvh",
        "cpm_r02",
        "cpm_r03",
        "cpm_r04",
        "cpm_r05",
        "cpm_r06",
        "cpm_r07",
        "cpm_r08",
        "cpm_r09",
    ]
    return df[keep_cols]


def prepare_all_extended_stations() -> None:
    audit_rows = []
    for st in EXTENDED_CANDIDATES:
        st_id = st["id"]
        print(f"=== Preparing extended station {st['name']} ({st_id}, {st['climate_regime']}) ===")
        _download_file(
            st["radnet_url"],
            Path(st["radnet_zip"]),
            st["name"],
            f"Extended RadNet historical archive 2017-2025 for {st['name']}",
        )
        for yr in YEARS:
            noaa_url = f"https://noaa-global-hourly-pds.s3.amazonaws.com/{yr}/{st['noaa_id']}.csv"
            noaa_path = Path(f"data/raw/noaa_{st['noaa_call'].lower()}_{yr}.csv")
            _download_file(
                noaa_url,
                noaa_path,
                f"{st['name']} (NOAA {st['noaa_call']})",
                f"NOAA NCEI Global-Hourly {yr} for {st['noaa_call']} ({st['noaa_id']})",
            )

        out_merged = Path(f"data/processed/merged_{st_id}_2017_2025.csv.gz")
        if out_merged.exists():
            merged = pd.read_csv(out_merged)
            merged["utc_hour"] = pd.to_datetime(merged["utc_hour"])
        else:
            df_rad = _load_and_clean_radnet_flexible(st)
            df_weather = load_and_clean_noaa(st)
            grid = pd.date_range("2017-01-01 00:00:00", "2025-12-31 23:00:00", freq="h", name="utc_hour")
            df_grid = pd.DataFrame(index=grid).reset_index()
            merged = pd.merge(df_grid, df_rad, on="utc_hour", how="left")
            merged = pd.merge(merged, df_weather, on="utc_hour", how="left")
            merged["station_id"] = st_id.upper()
            merged["station_name"] = st["name"]
            merged["year"] = merged["utc_hour"].dt.year
            merged["month"] = merged["utc_hour"].dt.month
            merged["day"] = merged["utc_hour"].dt.day
            merged["hour_utc"] = merged["utc_hour"].dt.hour
            merged["dayofweek"] = merged["utc_hour"].dt.dayofweek
            merged["has_radnet_obs"] = merged["gross_cpm"].notna()
            merged["has_weather_obs"] = merged["temp_c"].notna()
            merged["is_raining"] = merged["precip_1h_mm"] > 0.0
            merged.to_csv(out_merged, compression="gzip", index=False)
            print(f"  Saved {out_merged} ({len(merged):,} rows)")

        qual_summary, _ = audit_station_raw_data_quality(merged, st_id)
        dts = pd.to_datetime(merged["utc_hour"])
        valid_rad = merged["has_radnet_obs"].fillna(False) & merged["rad_complete_channels"].fillna(False)
        fit_h = int((valid_rad & (dts < "2021-01-01")).sum())
        val_h = int((valid_rad & (dts >= "2021-01-01") & (dts < "2023-01-01")).sum())
        test_h = int((valid_rad & (dts >= "2023-01-01")).sum())
        passes_gate = bool(
            qual_summary["valid_complete_channel_hours"] >= 50000
            and fit_h > 0
            and val_h > 0
            and test_h > 0
        )

        audit_rows.append({
            "station_id": st_id,
            "station_name": st["name"],
            "climate_regime": st["climate_regime"],
            "noaa_call": st["noaa_call"],
            "noaa_id": st["noaa_id"],
            "role": st["role"],
            "fit_valid_hours_2017_2020": fit_h,
            "val_valid_hours_2021_2022": val_h,
            "test_valid_hours_2023_2025": test_h,
            "passes_completeness_gate": passes_gate,
            **qual_summary,
        })

    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv("data/processed/extended_station_coverage_audit.csv", index=False)
    print("\n=== Extended Station Coverage & Quality Audit ===")
    print(df_audit[["station_id", "climate_regime", "valid_complete_channel_hours", "completeness_pct", "fit_valid_hours_2017_2020", "val_valid_hours_2021_2022", "test_valid_hours_2023_2025", "passes_completeness_gate", "dry_share_r02_mean", "dry_share_r05_mean"]].to_string(index=False))


if __name__ == "__main__":
    prepare_all_extended_stations()
