# Phase 9 Summary: Manuscript Rewrite and Reference Audit

## Overview
Phase 9 converted the preliminary undergraduate project report into a publication-ready journal manuscript formatted in the official MDPI Sensors template (`report/manuscript_mdpi.tex` and `report/manuscript_mdpi.pdf`). The revision enforces zero number drift by generating all tables and manuscript macros directly from backing CSVs and audits every reference against Crossref publisher records.

## Key Deliverables

1. **MDPI Sensors Manuscript (`report/manuscript_mdpi.tex`, `report/manuscript_mdpi.pdf`)**:
   - Official MDPI document class (`mdpi.cls`, `mdpi.bst`, `journalnames.tex`, vector logos).
   - Removed all course boilerplate (MUH401 headers, framed boxes, student ID).
   - Title explicitly declares synthetic benchmark scope: *"Anomaly Detection in Environmental Radiation Sensor Networks under Precipitation-Induced Radon Washout: Evaluation on Synthetic Atmospheric Plumes"*.
   - Structured in 7 formal sections:
     1. Introduction
     2. Related Work (accurate representation of prior work)
     3. Materials and Methods (3.1 Data, 3.2 Features, 3.3 Synthetic Plumes, 3.4 Models & Baselines, 3.5 Operating Points, 3.6 Statistics)
     4. Results (4.1 Benchmark Detection/FA, 4.2 Operating-Point Budgets, 4.3 Ablations, 4.4 11-Station LOSO, 4.5 Fukushima Real-Event Limits, 4.6 Alarm Audit)
     5. Discussion
     6. Limitations (Honest, non-defensive coverage of benchmark dependence, photopeak shifting, ambient MDA limits, hardware step-drift, and unadjudicated alarms)
     7. Conclusions
     - MDPI Back Matter: Data Availability, Code Availability, Generative AI Use Declaration, Author Contributions, Funding, Conflicts of Interest, Acknowledgments, References.
   - Compiled with `pdflatex` into a 20-page camera-ready PDF (`report/manuscript_mdpi.pdf`).

2. **Automated Table & Macro Generator (`scripts/make_manuscript_tables.py`, `report/manuscript_macros.tex`)**:
   - Reads final processed CSVs and emits 6 LaTeX tables in `report/tables/`:
     * `table1_benchmark_summary.tex`: Benchmark detection and false alarm rates with 95% bootstrap CIs.
     * `table2_fa_budget_operating_points.tex`: Performance under fixed validation FA budgets ($\alpha \in \{1, 2, 5, 10\}$ yr$^{-1}$).
     * `table3_loso_11stations.tex`: 11-station LOSO cross-validation (Zero-Shot vs Local Calibration).
     * `table4_alarm_adjudication.tex`: Diagnostic breakdown of $N=100$ clean-background alarms.
     * `table5_real_event_fukushima.tex`: Real-event evaluation during 2011 Fukushima plume.
     * `table6_ablation_summary.tex`: Feature ablations and B1--B8 comparators at 95% validation target.
   - Generates 58 LaTeX macros in `report/manuscript_macros.tex` guaranteeing that every quantitative number in the abstract and results is dynamically linked to backing CSV data.
   - Automated unit test in `tests/test_manuscript_macros.py` verifies macro existence and numerical consistency against CSVs.

3. **Reference Audit (`scripts/audit_references.py`, `docs/reference_audit.csv`)**:
   - 38 references audited and verified against Crossref / publisher official repositories (Status: `VERIFIED`).
   - Mapped each reference to its specific in-text section and claim.
   - Corrected citations:
     * Breitkreutz et al. (2023): Described as AI for environmental gamma dose-rate time series anomaly detection (not spectral decomposition).
     * Frankemölle et al. (2025): Described as Bayesian baseline prediction in local monitoring networks (not spectral decomposition).
     * Thakur et al. (2012): Added verified DOI (`10.1039/c2em11011c`) documenting dilute trans-Pacific Fukushima activity concentrations ($31\text{ mBq/m}^3$).
     * Cited required foundational sources: US EPA RadNet Documentation (2021), ICRP Publication 74 (1996), NOAA ISD (Smith et al., 2011), LightGBM (Ke et al., 2017), and NuDat 3.0 ENSDF decay data (2023).

4. **Verification**:
   - `python -m pytest -q -m "not integration"`: 97 passed in 9.55s.
   - `python run_pipeline.py --check`: All 26 prerequisites satisfied.
   - `python recompute_claims.py`: Exits 0.
