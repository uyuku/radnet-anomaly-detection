# Open Questions & Unknown Parameters

This file tracks parameters and questions that are not yet known from data or cited sources, per Section 2 of `PROJECT_SPEC.md`.

## Resolved in Phase 0

1. **Exact gamma channel energy boundaries (in keV)**:
   - **Resolution**: Identified from peer-reviewed literature using EPA RadNet airborne spectrometry:
     - Channel 1 (γ1 / R01): $\le 100\text{ keV}$ (noise threshold, omitted in public near-real-time CSVs)
     - Channel 2 (γ2 / R02): $101 - 200\text{ keV}$
     - Channel 3 (γ3 / R03): $201 - 400\text{ keV}$ (Contains Pb-214 352 keV and I-131 364 keV)
     - Channel 4 (γ4 / R04): $401 - 600\text{ keV}$
     - Channel 5 (γ5 / R05): $601 - 800\text{ keV}$ (Contains Bi-214 609 keV and Cs-137 662 keV)
     - Channel 6 (γ6 / R06): $801 - 1000\text{ keV}$
     - Channel 7 (γ7 / R07): $1001 - 1400\text{ keV}$ (Contains Bi-214 1120 keV)
     - Channel 8 (γ8 / R08): $1401 - 1800\text{ keV}$ (Contains Bi-214 1764 keV)
     - Channel 9 (γ9 / R09): $1801 - 2200\text{ keV}$
   - **Citation**: Vieira et al. (2019), *Environmental Research*, 175, 221–227. https://doi.org/10.1016/j.envres.2019.05.021.
2. **Exposure rate start dates and sensor models across all stations**:
   - **Resolution**: Scraped from official EPA downloads page into `data/processed/radnet_station_inventory.csv`. 127 stations have exposure rate capability, 20 of which have records dating back to mid-2016.
3. **Missingness baseline across candidate pilot stations**:
   - **Resolution**: Quantified across 2017–2025 (9 full calendar years, 78,888 expected hourly records per station). The top 5 stations show 80.6% to 94.3% overall completeness and 93.9% to 100.0% exposure rate presence.

---

## Active Open Questions for Phase 1 & Later

1. **Rooftop site vs. Airport distance offset**:
   - EPA does not publish exact street addresses or coordinates for stationary monitors for security reasons; monitors are located in urban/metropolitan centers (often on state/local agency roofs). The airport ASOS stations are located 5–25 km from city centers.
   - Question: Does localized convective precipitation introduce spatial mismatches between airport rain gauges and RadNet particulate filters?
   - Plan in Phase 1: Test correlation between NOAA rain reports and RadNet count rate surges to quantify lag and detection probability.
2. **Operational ground-truth labeling for radon washout class**:
   - Per Section 5 of Project Spec: Real data lacks ground-truth labels. Any rule-based labeling (rain + count rate rise + half-life decay) is partly circular.
   - Plan for Phase 3: Formalize the non-circular operational criteria and document all assumptions and limitations in `DECISIONS.md`.
