# Mandatory Manuscript Statements & Declarations (MDPI Format)

**Revision Step**: Phase 8, Part D  
**Target Venue**: MDPI Journal Submission  
**Date**: October 31, 2026

---

## 1. Data Availability Statement

The raw environmental radiation monitoring data analyzed in this study are publicly available from the United States Environmental Protection Agency (US EPA) RadNet repository at `https://www.epa.gov/radnet/radnet-csv-file-downloads`. Surface meteorological observations are publicly available from the National Oceanic and Atmospheric Administration (NOAA) National Centers for Environmental Information (NCEI) Integrated Surface Database (ISD) via the NOAA Open Data Dissemination (NODD) program at `https://noaa-global-hourly-pds.s3.amazonaws.com/`. 

All derived and processed benchmark datasets—including the pre-computed training and test time series grids, Monte Carlo NaI(Tl) detector response templates, synthetic plume injection catalogs, per-plume and per-month benchmark outputs, and the complete data manifest (`DATA_MANIFEST.csv`)—are deposited in the Zenodo open research repository:
> **Zenodo Repository**: `[DOI PLACEHOLDER UPON DEPOSIT: https://doi.org/10.5281/zenodo.XXXXXXX]`  
> **License**: Creative Commons Attribution 4.0 International (CC-BY-4.0).

---

## 2. Code Availability Statement

The underlying source code, feature engineering algorithms, baseline models, statistical evaluation pipelines, and figure generation scripts are publicly available in the GitHub repository:
> **Repository**: `https://github.com/TODO_AUTHOR/radnet-anomaly-detection`  
> **Archived Version**: `v1.0.0` (Software Heritage / Zenodo DOI: `[DOI PLACEHOLDER]`)  
> **Software License**: MIT License  
> **Containerized Environment**: Full reproduction Dockerfile and locked conda environment specification (`requirements-lock.txt`, `environment.yml`) are included in the repository root.

---

## 3. Declaration of Generative AI and AI-Assisted Technologies in the Writing Process

### 3.1 Journal Policy Grounding
This declaration complies with MDPI's official Policy on the Use of Generative AI and AI-Assisted Technologies (`https://www.mdpi.com/ethics#_bookmark28`):
> *"Generative AI tools and Large Language Models (such as ChatGPT, Copilot, or similar) do not meet the criteria for authorship. Authors are fully responsible for the originality, validity, and integrity of their manuscript, including any parts produced or assisted by AI tools. Authors must disclose the use of generative AI in a dedicated section."*

### 3.2 Author Disclosure Frame
During the preparation of this manuscript, the authors utilized advanced agentic AI coding assistants ([Author to specify tool: e.g., Google DeepMind Antigravity / Gemini 3.8 Flash]) exclusively for:
1. Drafting, refactoring, and linting computational Python pipeline scripts and automated statistical testing harnesses.
2. Generating reproducible data manifests and hashing scripts (`DATA_MANIFEST.csv`).
3. Assisting with LaTeX table formatting and technical English grammar polishing.

The authors independently verified all mathematical derivations, executed all data pipelines, audited all experimental tables against committed raw outputs, reviewed and revised all manuscript text, and take full institutional and scholarly responsibility for the scientific integrity, accuracy, and conclusions of the work. Generative AI tools are not listed as authors and have no intellectual ownership of this study.

---

## 4. Author Contributions (CRediT Taxonomy)

- **Conceptualization**: Necdet Ömer Barut, Şule Ergün;
- **Methodology**: Necdet Ömer Barut, Şule Ergün;
- **Software**: Necdet Ömer Barut;
- **Validation**: Necdet Ömer Barut, Şule Ergün;
- **Formal Analysis**: Necdet Ömer Barut;
- **Investigation**: Necdet Ömer Barut;
- **Data Curation**: Necdet Ömer Barut;
- **Writing – Original Draft Preparation**: Necdet Ömer Barut;
- **Writing – Review & Editing**: Şule Ergün, Necdet Ömer Barut;
- **Visualization**: Necdet Ömer Barut;
- **Supervision**: Şule Ergün;
- **Project Administration**: Şule Ergün.

All authors have read and agreed to the published version of the manuscript.

---

## 5. Funding Statement
> `[FUNDING PLACEHOLDER: This research received no external funding / This work was supported by [Grant Provider Name, Grant Number XXXXXXX].]`

---

## 6. Institutional Review Board (IRB) and Ethical Approval
Not applicable. This study analyzes solely publicly available environmental sensor telemetry and meteorological data and does not involve human subjects, live vertebrates, or private patient records.

---

## 7. Informed Consent Statement
Not applicable.

---

## 8. Conflicts of Interest
The authors declare no conflict of interest. The funders had no role in the design of the study; in the collection, analyses, or interpretation of data; in the writing of the manuscript; or in the decision to publish the results.

---

## 9. Acknowledgments
The authors gratefully acknowledge the United States Environmental Protection Agency (EPA) for continuous public operation and archival of the nationwide RadNet radiation monitoring network, and the National Oceanic and Atmospheric Administration (NOAA) National Centers for Environmental Information (NCEI) for maintaining the Integrated Surface Database (ISD).
