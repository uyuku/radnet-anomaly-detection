# Instructions for Author: Depositing the Research Archive on Zenodo

**Revision Step**: Phase 8, Part A.3  
**Date**: October 31, 2026  
**Archive Folder**: `zenodo_archive/`

---

## 1. Prerequisites
1. Create or log in to your account at [Zenodo](https://zenodo.org) (or [Zenodo Sandbox](https://sandbox.zenodo.org) for testing).
2. Connect your ORCID ID to your Zenodo account to link publication metadata automatically.

---

## 2. Deposit Steps

1. **Navigate to Uploads**:
   - Click the green **"New upload"** button in Zenodo.

2. **Upload Files**:
   - Drag and drop the contents of the `zenodo_archive/` directory:
     - `DATA_MANIFEST.csv`
     - `README.md`
     - All CSV and JSON datasets from `zenodo_archive/data/processed/` and `zenodo_archive/docs/`.
   - Alternatively, create a single zip file:
     ```bash
     cd zenodo_archive && zip -r ../radnet_benchmark_zenodo_v1.0.0.zip . && cd ..
     ```
     and upload `radnet_benchmark_zenodo_v1.0.0.zip`.

3. **Fill Deposit Metadata**:
   - **Upload type**: Dataset.
   - **Publication date**: 2026-10-31 (or submission date).
   - **Title**: Curated Benchmark Datasets, Monte Carlo Spectrometry Templates, and Machine Learning Evaluation Results for RadNet Ambient Radiological Anomaly Detection.
   - **Authors**: Fill in your full name, affiliation, and ORCID ID.
   - **Description**: Copy and paste the text from `zenodo_archive/README.md`.
   - **License**: Creative Commons Attribution 4.0 International (`CC-BY-4.0`).
   - **Keywords**: Radiation anomaly detection; RadNet; Environmental radioactivity; Gamma spectrometry; Machine learning; LightGBM; Weather fusion; Radon washout.
   - **Related identifiers**: Add GitHub repository link:
     - Relation: `isSupplementTo` -> `https://github.com/TODO_AUTHOR/radnet-anomaly-detection`

4. **Reserve DOI**:
   - Click **"Reserve DOI"** at the top right of the upload page.
   - Copy the generated DOI string (e.g., `10.5281/zenodo.1234567`).
   - Replace the `[DOI PLACEHOLDER]` in `docs/statements_draft.md` and the manuscript `report/interim_report.tex`.

5. **Publish**:
   - When ready, click **"Publish"** to mint the persistent DOI.
