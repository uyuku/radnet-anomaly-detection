"""
Parses the RadNet CSV downloads HTML page to extract station metadata,
start dates for gamma count rate and exposure rate, and data URLs.
"""

import re
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path


def parse_stations(html_path: str) -> pd.DataFrame:
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    stations = []
    # Find all station paragraphs
    # Each station has a link to /radnet/radnet-near-real-time-air-data-...
    for a in soup.find_all("a", href=re.compile(r"/radnet/radnet-near-real-time-air-data-[a-z0-9-]+")):
        p = a.parent
        if not p or p.name != "p":
            continue

        station_name = a.get_text(strip=True)
        station_url = f"https://www.epa.gov{a['href']}"

        # Text in paragraph
        p_text = p.get_text(" ", strip=True)

        # Match start dates
        # e.g., "Start date: April 19, 2007 (gamma gross count rate) | July 5, 2016 (exposure rate)"
        # or "Start date: December 9, 2015 (gamma gross count rate) | Not currently available (exposure rate)"
        gamma_start = None
        exposure_start = None

        gamma_match = re.search(r"Start date:?\s*(.*?)\s*\(gamma gross count rate\)", p_text, re.IGNORECASE)
        if gamma_match:
            gamma_start = gamma_match.group(1).strip()
            # Clean up potential leading "Start date:"
            gamma_start = re.sub(r"^Start date:?\s*", "", gamma_start, flags=re.IGNORECASE)

        exp_match = re.search(r"\|\s*(.*?)\s*\(exposure rate\)", p_text, re.IGNORECASE)
        if exp_match:
            exp_text = exp_match.group(1).strip()
            if "not" not in exp_text.lower() and "n/a" not in exp_text.lower():
                exposure_start = exp_text

        # 2026 CSV URL
        csv_2026_url = None
        csv_2026_link = p.find("a", href=re.compile(r"api/rest/csv/2026/fixed"))
        if csv_2026_link:
            csv_2026_url = csv_2026_link["href"]

        # Historical ZIP URL
        zip_url = None
        zip_link = p.find("a", href=re.compile(r"\.zip$", re.IGNORECASE))
        if zip_link:
            zip_url = zip_link["href"]
            if not zip_url.startswith("http"):
                zip_url = f"https://www.epa.gov{zip_url}"

        # State and City extraction
        parts = [part.strip() for part in station_name.split(",")]
        city = parts[0] if len(parts) > 0 else ""
        state = parts[1] if len(parts) > 1 else ""

        stations.append({
            "station_name": station_name,
            "city": city,
            "state": state,
            "station_url": station_url,
            "gamma_start": gamma_start,
            "exposure_start": exposure_start,
            "has_exposure_rate": exposure_start is not None,
            "csv_2026_url": csv_2026_url,
            "historical_zip_url": zip_url,
        })

    df = pd.DataFrame(stations).drop_duplicates(subset=["station_name"])
    return df


if __name__ == "__main__":
    html_file = Path("data/raw/radnet_csv_file_downloads.html")
    df = parse_stations(str(html_file))
    print(f"Total stations found: {len(df)}")
    print(f"Stations with exposure rate: {df['has_exposure_rate'].sum()}")
    out_file = Path("data/processed/radnet_station_inventory.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved inventory to {out_file}")
