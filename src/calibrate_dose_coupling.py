"""
Calibrates the physical dose-rate coupling coefficient (k_dose) by regressing
ambient dose rate excess (Delta Dose Rate, nSv/h) on gross count rate excess (Delta Gross CPM)
across verified precipitation washout hours for each pilot station and pooled network.

Saves: data/processed/dose_rate_cpm_regression_summary.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

STATIONS = [
    {"id": "al_birmingham", "name": "Birmingham, AL"},
    {"id": "dc_washington", "name": "Washington, DC"},
    {"id": "ca_san_diego", "name": "San Diego, CA"},
    {"id": "tx_dallas", "name": "Dallas, TX"},
    {"id": "fl_tampa", "name": "Tampa, FL"},
]


def run_dose_calibration():
    results = []
    pooled_cpm_excess = []
    pooled_dose_excess = []

    for st in STATIONS:
        csv_file = Path(f"data/processed/merged_{st['id']}_2017_2025.csv.gz")
        df = pd.read_csv(csv_file)
        df["dt"] = pd.to_datetime(df["utc_hour"])
        df = df.sort_values("dt").reset_index(drop=True)

        # Baseline: 168-hour rolling median on continuous calendar grid
        # Trailing (causal) baseline window. BUG-7 fix (2026-10-06): the previous
        # center=True window used up to +84h of future data in the k_dose regression.
        df["cpm_base"] = df["gross_cpm"].rolling(168, min_periods=72).median()
        df["dose_base"] = df["dose_rate_nsvh"].rolling(168, min_periods=72).median()

        df["cpm_excess"] = df["gross_cpm"] - df["cpm_base"]
        df["dose_excess"] = df["dose_rate_nsvh"] - df["dose_base"]

        # Selection criteria:
        # 1. Active precipitation: precip_1h_mm >= 1.0 mm
        # 2. Significant gross excess: cpm_excess >= 100 CPM
        # 3. Positive dose excess
        # 4. Valid RadNet observations
        mask = (
            (df["precip_1h_mm"] >= 1.0)
            & (df["cpm_excess"] >= 100.0)
            & (df["dose_excess"] > 0.0)
            & df["has_radnet_obs"]
            & df["rad_complete_channels"]
            & df["dose_rate_nsvh"].notna()
        )
        sub = df[mask].dropna(subset=["cpm_excess", "dose_excess"])

        if len(sub) > 10:
            reg = stats.linregress(sub["cpm_excess"], sub["dose_excess"])
            ci_95 = 1.96 * reg.stderr
            results.append({
                "Station_ID": st["id"],
                "Station_Name": st["name"],
                "Rain_Hours_Analyzed": len(sub),
                "Slope_nSvh_per_CPM": round(reg.slope, 5),
                "Intercept_nSvh": round(reg.intercept, 2),
                "R_Squared": round(reg.rvalue ** 2, 3),
                "Std_Err": round(reg.stderr, 6),
                "CI95_Lower": round(reg.slope - ci_95, 5),
                "CI95_Upper": round(reg.slope + ci_95, 5),
                "P_Value": reg.pvalue,
            })
            pooled_cpm_excess.extend(sub["cpm_excess"].tolist())
            pooled_dose_excess.extend(sub["dose_excess"].tolist())

    # Pooled network regression
    reg_pool = stats.linregress(pooled_cpm_excess, pooled_dose_excess)
    ci_95_pool = 1.96 * reg_pool.stderr
    results.append({
        "Station_ID": "pooled_network",
        "Station_Name": "Pooled Network (5 Stations)",
        "Rain_Hours_Analyzed": len(pooled_cpm_excess),
        "Slope_nSvh_per_CPM": round(reg_pool.slope, 5),
        "Intercept_nSvh": round(reg_pool.intercept, 2),
        "R_Squared": round(reg_pool.rvalue ** 2, 3),
        "Std_Err": round(reg_pool.stderr, 6),
        "CI95_Lower": round(reg_pool.slope - ci_95_pool, 5),
        "CI95_Upper": round(reg_pool.slope + ci_95_pool, 5),
        "P_Value": reg_pool.pvalue,
    })

    out_df = pd.DataFrame(results)
    out_csv = Path("data/processed/dose_rate_cpm_regression_summary.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved dose-rate coupling calibration to {out_csv}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    run_dose_calibration()
