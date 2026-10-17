# RadNet Anomaly Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Data: EPA RadNet + NOAA ISD](https://img.shields.io/badge/Data-EPA%20RadNet%20%2B%20NOAA%20ISD-orange.svg)](https://www.epa.gov/radnet)

When it rains, natural radon daughters ($^{214}\text{Pb}$, $^{214}\text{Bi}$) wash out of the atmosphere onto EPA RadNet particulate filters. This produces sudden gamma count spikes of +40% to +180% that mimic radiological releases.

Conventional fixed and rolling $3\sigma$ thresholds fail under precipitation: across humid U.S. stations they trigger **53 to 83 false alarm episodes per station-year**, with up to 89% of alarm episodes in humid cities coinciding with rain.

This project fuses **8-channel EPA RadNet NaI(Tl) gamma spectrometry** with synchronous **NOAA ISD surface weather telemetry** using gradient-boosted decision trees (LightGBM). Evaluated across 20 random seeds under a strictly causal validation-frozen protocol (2017–2020 train, 2021–2022 validation, 2023–2025 out-of-sample test), full weather-spectral fusion reduces false alarms to **4.32 ± 1.29 per station-year** (**0.0% rain-coincident**) while detecting **92.65 ± 1.37%** of synthetic fission/activation plumes.

---

## How It Works

![Environmental Engineering Overview](reports/figures/environmental_engineering_synthesis.png)

1. **Continuous Air Monitoring**: EPA RadNet monitors draw ambient air through glass-fiber filters and record hourly gross counts (CPM), ambient dose rate (nSv/h), and 8 NaI(Tl) gamma energy channels (`R02`: 101–200 keV through `R09`: 1801–2200 keV).
2. **Environmental Confounders**:
   - **Precipitation Washout**: Rain scavenges airborne $^{214}\text{Pb}$ ($T_{1/2} = 26.8\text{ min}$) and $^{214}\text{Bi}$ ($T_{1/2} = 19.9\text{ min}$), causing immediate count surges (+40% to +180% at lag 0–1 h) that decay over 3–6 hours.
   - **Nocturnal Boundary-Layer Inversions**: Morning stability traps terrestrial radon progeny near the surface, creating a +5% to +15% daily oscillation peaking near 06:00–08:00 local time.
   - **Filter Tape Advances**: Automated filter replacements every 3–5 days cause abrupt step-drops of 340–720 CPM.
3. **Physics-Informed ML Discrimination**: LightGBM evaluates 48 strictly causal features combining spectral photopeak ratios (`R05/R07`, `R03/R07`, `R05/R03`), high-energy channel shares (`R07+R08`), multi-scale rolling statistics, and NOAA precipitation/barometric dynamics.

---

## Results at a Glance (2023–2025 Out-of-Sample Test Split)

Evaluated across **334,567 valid RadNet hours** (**324,550 synchronous radiation-weather hours**; **106,628 observed test hours** = 12.16 station-years) across five U.S. stations (Birmingham AL, Washington DC, San Diego CA, Dallas TX, Tampa FL) and **450 filter-synchronized synthetic plumes** (250 train, 200 test across $^{137}\text{Cs}$, $^{131}\text{I}$, $^{60}\text{Co}$, and $^{134}\text{Cs}$ mixtures):

| Model / Rule (95% Validation Target) | False Alarms / Station-Year | Rain-Coincident FA | Realized Test Detection | Median Detection Delay |
| :--- | :---: | :---: | :---: | :---: |
| **Standard Baseline** (Rolling 7d $3\sigma$) | 82.62 | 32.5% | 52.50% | 10.0 hrs |
| **Global Dry Baseline** (Global $3\sigma_{\text{dry}}$) | 52.86 | 24.9% | 57.00% | 9.0 hrs |
| **Conservative Baseline** (Rolling 7d $5\sigma$) | 23.51 | 57.6% | 6.00% | 8.5 hrs |
| **Tier 1: Gross GBDT** (12 features) | 166.78 ± 19.69 | 17.3% | 92.00 ± 0.69% | 6.7 hrs |
| **Tier 1b: Gross + Weather GBDT** (29 features) | 29.27 ± 3.93 | 0.0% | 85.58 ± 1.59% | 9.5 hrs |
| **Tier 2: Spectral GBDT** (30 features) | 7.21 ± 3.93 | 4.7% | 92.98 ± 0.92% | 4.0 hrs |
| **Tier 3: Weather Fusion GBDT** (48 features) | **4.32 ± 1.29** | **0.0%** | **92.65 ± 1.37%** | **4.0 hrs** |
| **Tier 3: Weather-Fused MLP** (Neural Net) | 58.30 ± 23.63 | 3.7% | 90.18 ± 2.81% | 3.6 hrs |

*Source: [`data/processed/rigorous_benchmark_summary.csv`](file:///Users/o/Projects/radnet-anomaly-detection/data/processed/rigorous_benchmark_summary.csv) (20 random seeds, thresholds frozen on 2021–2022 validation fold, net-alarm criterion).*

![ROC Performance Curves](reports/figures/rigorous_matched_roc_curves.png)

---

## Key Visualizations

### 1. Rain Washout Spikes
Precipitation leads gamma count surges by 0–1 hours across continental and coastal stations, followed by exponential decay of short-lived radon progeny:

![Rain Washout Surges](reports/figures/rain_washout_multi_station_comparison.png)

### 2. Particulate Filter Sawtooth
Particulate filters accumulate aerosol background over 3–5 days before dropping abruptly by 340–720 CPM when the filter tape advances:

![Filter Tape Advance Sawtooth](reports/figures/filter_cycle_sawtooth.png)

### 3. Model Benchmark & Uncertainty
Station-month block bootstrap 95% confidence intervals showing false alarm rates collapsing from 82.6 down to ~4.3/yr while maintaining >92% plume detection:

![Bootstrap Benchmark Uncertainty](reports/figures/eval_bootstrap_uncertainty.png)

---

## Quickstart, Operational CLI & Automated Tests

### 1. Install Dependencies
Python ≥ 3.10:
```bash
pip install -r requirements.txt
```

### 2. Run the Automated Verification Test Suite
The test suite in [`tests/test_radnet_pipeline.py`](file:///Users/o/Projects/radnet-anomaly-detection/tests/test_radnet_pipeline.py) verifies strict temporal causality (zero future lookahead across all 48 features), spectral conservation laws, zero test leakage into dry baselines, NOAA hourly precipitation deduplication, unobserved detector-outage labeling, and cross-table CSV consistency:
```bash
.venv/bin/pytest -v
```

### 3. Run the Operational CLI Diagnostic Utility
Use [`src/detect_anomalies_cli.py`](file:///Users/o/Projects/radnet-anomaly-detection/src/detect_anomalies_cli.py) to run model inference, compare against rolling $3\sigma$ alarms, and inspect episode-level spectral and weather attribution on any station and time window:
```bash
# Inspect clean background during the Jan 25, 2023 Birmingham rainstorm (Rolling 3σ false-alarms; Tier 3 stays silent)
.venv/bin/python src/detect_anomalies_cli.py --station al_birmingham --start "2023-01-24 18:00:00" --end "2023-01-29 06:00:00"

# Inspect the same storm window with the synthetic Cs-137 hard-case plume active (INJ_0067)
.venv/bin/python src/detect_anomalies_cli.py --station al_birmingham --use-injected --start "2023-01-24 18:00:00" --end "2023-01-29 06:00:00"

# Emit machine-readable JSON diagnostics
.venv/bin/python src/detect_anomalies_cli.py --station dc_washington --split test --json
```

### 4. Run the Full Pipeline
```bash
# Preview the execution plan
bash run_pipeline.sh --dry-run

# Run all phases sequentially
bash run_pipeline.sh

# Or run a single phase (e.g. Phase 5 evaluation)
bash run_pipeline.sh --phase 5
```

---

## License
MIT
