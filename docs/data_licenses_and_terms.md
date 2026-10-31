# Data Licenses, Terms of Use, and Redistribution Rights

**Revision Step**: Phase 8, Part A.2  
**Date**: October 31, 2026  
**Scope**: Legal status, copyright disclaimers, access policies, and attribution guidelines for public data sources utilized in this project.

---

## 1. U.S. Environmental Protection Agency (EPA) RadNet Data

### 1.1 Legal Status and Copyright
- **Statutory Authority**: Under Section 105 of the United States Copyright Act (17 U.S.C. § 105), works produced by officers and employees of the United States Government in the performance of their official duties are not subject to copyright protection within the United States.
- **Official Policy Statement**: 
  - EPA Privacy and Security Notice: `https://www.epa.gov/privacy/privacy-and-security-notice`
  - "EPA information is in the public domain unless otherwise indicated. You may freely copy and distribute EPA information, with appropriate credit given."
- **RadNet Data Access Portal**: 
  - URL: `https://www.epa.gov/radnet/radnet-csv-file-downloads`
  - Access Date: September 28, 2026 (pilot archives) / October 31, 2026 (extended & 2011 archives).
  - Terms: Open access, free of charge, with no API key or contractual licensing agreement required.

### 1.2 Redistribution and Attribution Requirements
- **Redistribution Rights**: Fully permitted without royalty or copyright restriction.
- **Disclaimer**: RadNet near-real-time data are screening measurements subject to preliminary automated quality control. Formal laboratory verification occurs on physical filter media at NAREL.
- **Recommended Attribution**: 
  > "RadNet near-real-time monitoring data provided courtesy of the United States Environmental Protection Agency (US EPA, https://www.epa.gov/radnet)."

---

## 2. NOAA National Centers for Environmental Information (NCEI) ISD Data

### 2.1 Legal Status and Open Data Policy
- **Statutory Authority**: As a work of the U.S. Department of Commerce / National Oceanic and Atmospheric Administration, data compiled by NOAA NCEI are in the worldwide public domain (17 U.S.C. § 105).
- **Official Policy Statement**:
  - NOAA Open Data Policy: `https://www.noaa.gov/information-technology/open-data`
  - "NOAA data and products are free and openly accessible to all users worldwide without restriction."
- **Data Source**: Integrated Surface Database (ISD) / Global Hourly dataset (Smith et al., 2011, *BAMS*, DOI: 10.1175/2011BAMS3015.1).
- **Access Endpoints**:
  - NOAA AWS Open Data Registry: `https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{usaf_wban}.csv`
  - NCEI Web Archive: `https://www.ncei.noaa.gov/data/global-hourly/access/`
  - Terms: Fully open access, no rate limits, public bucket access.

### 2.2 Recommended Attribution
- **Recommended Attribution**:
  > "Meteorological observations retrieved from the NOAA National Centers for Environmental Information (NCEI) Integrated Surface Database (ISD), accessed via the NOAA Open Data Dissemination (NODD) Program on AWS."

---

## 3. Project Derived Datasets and Code

- **Software Code**: Licensed under the MIT License (`LICENSE`).
- **Curated Datasets and Synthetic Catalogs**: Released under Creative Commons Attribution 4.0 International (CC-BY-4.0).
