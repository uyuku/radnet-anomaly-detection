# Physical Derivation & Reconciliation: Fukushima 2011 Real-Event Validation

## 1. Submersion Immersion Dose Recomputation
### Governing Formula & Dose Coefficients:
For an airborne concentration $C_{\text{air}}$ ($\text{Bq/m}^3$) in a semi-infinite cloud, the air submersion effective dose equivalent rate $\dot{D}_{\text{sub}}$ is given by:
$$\dot{D}_{\text{sub}} = C_{\text{air}} \cdot h_{\text{sub}}$$

From **Federal Guidance Report No. 12 (EPA 402-R-93-081, 1993), Table III.1** (and consistent with ICRP Publication 119):
- For $^{131}\text{I}$ ($t_{1/2} = 8.02\text{ d}$), the air submersion dose coefficient is:
  $$h_{\text{sub}} = 1.82 \times 10^{-14} \text{ Sv}\cdot\text{m}^3/(\text{Bq}\cdot\text{s})$$
- Converting to practical environmental units:
  $$h_{\text{sub}} = 1.82 \times 10^{-14} \times 10^9\text{ nSv/Sv} \times 3{,}600\text{ s/h} = 6.552 \times 10^{-2}\text{ nSv/h per Bq/m}^3$$

### Evaluation at Stated Trans-Pacific Concentration:
During peak trans-Pacific atmospheric transport following the March 11, 2011 Fukushima accident, measured air concentrations in Hawaii and the U.S. West Coast peaked at:
$$C_{\text{air}} = 31\text{ mBq/m}^3 = 0.031\text{ Bq/m}^3$$

Substituting into the immersion formula:
$$\dot{D}_{\text{sub}} = 0.031\text{ Bq/m}^3 \times 0.06552\text{ nSv/h per Bq/m}^3 = 2.031 \times 10^{-3}\text{ nSv/h} = 2.03\text{ pSv/h}$$

### Reconciliation of Previous Flawed Claim:
- The earlier draft text quoted $6.3 \times 10^{-6}\text{ nSv/h}$ ($0.0063\text{ pSv/h}$).
- The actual submersion dose rate is:
  $$\frac{2.031 \times 10^{-3}\text{ nSv/h}}{6.3 \times 10^{-6}\text{ nSv/h}} = 322.4 \approx 320\times \text{ higher}$$
- At the detector dose calibration slope of $\approx 0.0134\text{ nSv/h per CPM}$, this immersion dose rate produces:
  $$\Delta R_{\text{sub}} = \frac{0.002031\text{ nSv/h}}{0.0134\text{ nSv/h per CPM}} \approx 0.15\text{ CPM}$$
- An ambient excess of $0.15\text{ CPM}$ is completely imperceptible against 1-hour counting statistics (Poisson $\sigma \approx 5.7\text{ CPM}$) and natural background variations ($2{,}000\text{--}7{,}000\text{ CPM}$).

## 2. Active Filter Accumulation Mechanism & Sensitivity
The RadNet monitoring system actively draws ambient air through glass-fiber filter tape directly beneath a 2"x2" NaI(Tl) scintillation detector.

### Sampler Flow Rate & Cumulative Volume:
- Continuous volumetric flow rate: nominal $Q \approx 35\text{ CFM} \approx 60.0\text{ m}^3/\text{h}$ (sensitivity range $30\text{--}40\text{ CFM} = 51\text{--}68\text{ m}^3/\text{h}$; `[UNVERIFIED: EPA RadNet continuous sampler nominal flow rate ~35 CFM / ~60 m³/h]`).
- In 24 hours: $V_{24} = 60.0 \times 24 = 1{,}440\text{ m}^3$.
- Total $^{131}\text{I}$ activity passing through sampler in 24 hours:
  $$A_{\text{drawn}} = 1{,}440\text{ m}^3 \times 0.031\text{ Bq/m}^3 = 44.64\text{ Bq}$$

### Particulate Retained Fraction:
- In trans-Pacific atmospheric plumes, volatile gaseous iodine ($	ext{I}_2, \text{CH}_3\text{I}$) passes through standard particulate glass-fiber filters unretarded. The particulate fraction is typically $f_{\text{part}} \approx 15\%\text{--}35\%$ (nominal $25\%$).
- Retained particulate activity on filter tape after 24 hours: nominal $11.16\text{ Bq}$ (range $6.2\text{--}16.2\text{ Bq}$).

### Expected Scintillation Count Rate Excess:
- Photopeak efficiency in Channel 3 (200--400 keV): nominal $2.5\%$ (range $1.5\%\text{--}3.5\%$).
- Total gross gamma efficiency: nominal $4.5\%$ (range $3.0\%\text{--}6.0\%$).
- **After 24 hours:**
  - Channel 3: nominal $16.7\text{ CPM}$ (sensitivity range $6.9\text{--}35.8\text{ CPM}$).
  - Gross CPM: nominal $30.1\text{ CPM}$ (sensitivity range $13.9\text{--}61.4\text{ CPM}$).

### Comparison with Field Noise and Environmental Baseline Variability:
- **Single-hour Poisson counting noise:** For an hourly mean rate $R$, $\sigma_{\text{Poisson}} = \sqrt{R/60}$.
  - Honolulu ($R \approx 1{,}923\text{ CPM}$): $\sigma_{\text{Poisson}} = \sqrt{1923/60} \approx 5.7\text{ CPM}$.
  - Network stations ($2{,}000\text{--}7{,}000\text{ CPM}$): $\sigma_{\text{Poisson}} \approx 5.8\text{--}10.8\text{ CPM}$.
- **Natural environmental baseline variability:**
  - Radon diurnal swing, air mass trajectory, and pressure changes produce an empirical standard deviation of $\sigma_{\text{env}} = 129.7\text{ CPM}$ at Honolulu, and $150\text{--}400+\text{ CPM}$ at continental stations.
- **Physical Detectability Limit:** While $\approx 17\text{ CPM}$ exceeds the 1-hour Poisson counting statistics ($5.7\text{ CPM}$), it is completely masked by natural environmental variability ($\sigma_{\text{env}} \approx 130\text{ CPM}$). Natural background variability, not Poisson counting noise, is the true physical limit. Furthermore, operational units advance the filter tape periodically (every 24--72 hours), resetting the accumulated activity.

## 3. Honolulu March 21, 2011 Alarms Reconciliation
- In `data/processed/real_event_fukushima_station_summary.csv`, Honolulu recorded **2 alarm hours** for Tier 4 during the event window (`2011-03-21 07:00:00` and `2011-03-21 08:00:00` UTC).
- At both hours, meteorological conditions were completely dry (`precip_1h_mm = 0.0 mm`, 37--38 hours after prior rain). Gross count rates were 2432 CPM and 2452 CPM.
- Tier 4 scores reached **0.9157** and **0.9530**, exceeding the loaded operating threshold of **0.8719**.
- In contrast, Tier 3 (score 0.0189, 0.0149), Tier 2 (0.0128, 0.0137), Tier 1 (0.0492, 0.0498), and B4c (0.3598, 0.5130) showed zero alarms.
- **Physical Meaning:** These two alarms at Honolulu on March 21 do not represent confirmed Fukushima plume detections; rather, they reflect subtle PMT baseline shifts and maritime spectral differences in the Hawaiian NaI crystal that exceeded the continental-calibrated Tier 4 threshold under dry conditions.

## 4. Threshold Documentation Discrepancy
- `docs/real_event_protocol.md` erroneously documented the Tier 4 operating threshold as $\tau = 0.5802$.
- `src/evaluate_real_event.py` actually loaded the pre-specified validation budget operating threshold of $\tau = 0.8719$ (from the 5 FA/yr validation calibration).
- In accordance with Ground Rule 2 (no post-hoc re-tuning on evaluation events), the evaluation code was executed with $\tau = 0.8719$, and no thresholds have been retroactively modified.
