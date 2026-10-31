# Phase 3: Stronger Baselines & ML Feature Ablation Summary

## 1. One-Paragraph Statement of the Strongest Competing Baseline
Among all non-ML and unsupervised baselines fitted strictly on 2017–2020 clean background without synthetic plumes (`src/baselines.py`, `data/processed/baseline_matched_detection_table.csv`, `data/processed/baseline_matched_fa_table.csv`), **B4c (`B4c: Radon Projection + Stripped Channel Peak Z`)** — which combines the 6-hour smoothed local-168h radon-direction projection residual norm (`local_radon_stripping_residual_l2`) with the peak stripped photopeak channel $z$-score ($\max(\text{local\_stripped\_r03\_z}, \text{local\_stripped\_r05\_z}, \text{local\_stripped\_r07\_z})$) — is by far the strongest competing detector. At the 5 and 10 false-alarms/station-year validation budgets, **B4c achieves `89.00%` realized test detection (`81.50%` within $\le 12\text{ h}$, median delay `7.0 h`) at `6.49` false alarm episodes per station-year (`0.0%` rain-coincident)** without using machine learning or synthetic training plumes, and at the 95% validation detection target it achieves **`80.00%` test detection at `2.30` FA/station-year**. Compared against single-seed Tier 4 GBDT (`96.50%` test detection at `9.87` FA/yr; 20-seed mean `93.60 ± 1.03%` at `7.19 ± 1.17` FA/yr), **non-ML spectral radon stripping (`B4c`) accounts for the vast majority of the false-alarm reduction over gross-count sigma rules (`227.07` FA/yr $\to$ `6.49` FA/yr), while supervised GBDT training on synthetic templates contributes the remaining `+4.6 to +7.5 pp` detection gain (`89.00% -> 93.60%–96.50%`)**. Among unsupervised ML models trained on clean background only (`B6`), One-Class SVM on Tier 4 features (`B6c`) is strongest (`72.00%` test detection at `9.29` FA/yr at the 10 FA/yr budget; `78.50%` at `41.43` FA/yr at the 95% validation target).

---

## 2. Matched Detection Table (80%, 90%, 95% Validation Targets)
*Source: `data/processed/baseline_matched_detection_table.csv` (fitted on 2017–2020, thresholded on 2021–2022 validation, evaluated on 2023–2025 test split = 12.16 station-years).*

| Detector | Plumes in Train? | 80% Target: Test Det / FA/yr | 90% Target: Test Det / FA/yr | 95% Target: Test Det / FA/yr | Target Flag (95%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B1a: Rolling 168h Gross Z-Score** | No | `73.0%` / `155.95` | `84.0%` / `227.07` | `86.5%` / `254.61` | `MET` |
| **B1b: Global Dry Gross Z-Score** | No | `57.0%` / `61.41` | `82.0%` / `161.63` | `82.0%` / `161.63` | `UNREACHABLE (87.7%)` |
| **B2a: Rain-Masked Dry Z ($H=6\text{h}$)** | No | `51.0%` / `36.01` | `79.5%` / `118.14` | `79.5%` / `118.14` | `UNREACHABLE (87.7%)` |
| **B2b: Rain-Masked Dry Z ($H=12\text{h}$)** | No | `50.0%` / `35.76` | `79.0%` / `115.26` | `79.0%` / `115.26` | `UNREACHABLE (86.4%)` |
| **B3a: Weather Regression (Linear)** | No | `53.5%` / `31.90` | `66.0%` / `70.45` | `72.0%` / `84.43` | `UNREACHABLE (92.6%)` |
| **B3b: Weather Regression (Spline GAM)** | No | `53.5%` / `35.43` | `65.5%` / `72.26` | `76.5%` / `101.45` | `UNREACHABLE (93.8%)` |
| **B4a: Radon Projection (Global Dry)** | No | `59.0%` / `400.53` | `68.5%` / `434.57` | `72.0%` / `446.49` | `UNREACHABLE (93.8%)` |
| **B4b: Radon Projection (Local 168h)** | No | `66.5%` / `32.88` | `77.0%` / `47.35` | `82.0%` / `59.36` | `MET` |
| **B4c: Radon Projection + Peak Channel Z** | No | **`66.5%` / `0.66`** | **`70.0%` / `0.74`** | **`80.0%` / `2.30`** | `MET` |
| **B5a: Background PCA Subspace** | No | `51.0%` / `57.47` | `65.5%` / `133.43` | `69.5%` / `141.73` | `MET` |
| **B5b: Background NMF (2-Component)** | No | `56.0%` / `50.48` | `79.5%` / `129.89` | `80.0%` / `276.80` | `UNREACHABLE (93.8%)` |
| **B6a: Isolation Forest (Tier 3, Bg Only)** | No | `50.5%` / `45.87` | `69.5%` / `86.90` | `78.0%` / `268.99` | `UNREACHABLE (93.8%)` |
| **B6b: Isolation Forest (Tier 4, Bg Only)** | No | `68.5%` / `55.41` | `85.0%` / `132.11` | `81.0%` / `95.53` | `MET` |
| **B6c: One-Class SVM (Tier 4, Bg Only)** | No | `57.5%` / `2.22` | `86.0%` / `103.01` | `78.5%` / `41.43` | `MET` |
| **B7a: Logistic Regression (Tier 3)** | Yes | `61.5%` / `24.42` | `61.5%` / `24.42` | `70.5%` / `23.59` | `MET` |
| **B7b: Depth-3 Decision Tree (Tier 3)** | Yes | `76.0%` / `59.52` | `76.0%` / `59.52` | `76.0%` / `59.52` | `UNREACHABLE (88.9%)` |
| **B8: Operational Approx. (`Uncalibrated`)** | No | `62.0%` / `203.47` | `62.0%` / `203.47` | `62.0%` / `203.47` | `UNREACHABLE (74.1%)` |
| **Tier 1: Gross GBDT (12 feats)** | Yes | `76.0%` / `44.48` | `82.5%` / `108.85` | `86.5%` / `179.14` | `MET` |
| **Tier 1b: Gross + Weather GBDT (29 feats)** | Yes | `61.5%` / `6.17` | `76.5%` / `24.83` | `85.0%` / `69.96` | `MET` |
| **Tier 2: Spectral GBDT (30 feats)** | Yes | `86.5%` / `1.23` | `86.5%` / `1.23` | `86.5%` / `1.23` | `MET` |
| **Tier 3: Weather Fusion GBDT (48 feats)** | Yes | `93.5%` / `12.99` | `93.5%` / `12.99` | `93.5%` / `12.99` | `MET` |
| **Tier 4: Physics-Motivated Nuclear GBDT** | Yes | **`96.5%` / `9.87`** | **`96.5%` / `9.87`** | **`96.5%` / `9.87`** | `MET` |

---

## 3. Matched False-Alarm Budget Table (5, 10, 50 Validation FA/Station-Year)
*Source: `data/processed/baseline_matched_fa_table.csv`.*

| Detector | 5 Val FA/yr Budget: Test Det / Test FA/yr | 10 Val FA/yr Budget: Test Det / Test FA/yr | 50 Val FA/yr Budget: Test Det / Test FA/yr |
| :--- | :---: | :---: | :---: |
| **B1a: Rolling 168h Gross Z-Score** | `2.0%` / `5.59` | `5.0%` / `11.18` | `27.5%` / `41.85` |
| **B1b: Global Dry Gross Z-Score** | `10.0%` / `3.37` | `13.5%` / `5.84` | `42.5%` / `28.36` |
| **B2a: Rain-Masked Dry Z ($H=6\text{h}$)** | `16.5%` / `3.45` | `19.5%` / `5.01` | `48.0%` / `28.86` |
| **B2b: Rain-Masked Dry Z ($H=12\text{h}$)** | `18.0%` / `4.11` | `25.0%` / `6.74` | `48.5%` / `31.49` |
| **B3a: Weather Regression (Linear)** | `12.0%` / `4.44` | `18.0%` / `8.47` | `53.5%` / `31.90` |
| **B3b: Weather Regression (Spline GAM)** | `12.5%` / `4.11` | `19.0%` / `8.55` | `52.0%` / `27.46` |
| **B4b: Radon Projection (Local 168h)** | `17.5%` / `9.45` | `31.0%` / `14.88` | `77.0%` / `48.26` |
| **B4c: Radon Projection + Peak Channel Z** | **`89.0%` / `6.49`** | **`89.0%` / `6.49`** | **`95.5%` / `36.17`** |
| **B5b: Background NMF (2-Component)** | `46.5%` / `2.30` | `57.0%` / `110.74` | `72.5%` / `101.28` |
| **B6b: Isolation Forest (Tier 4, Bg Only)** | `28.5%` / `7.56` | `41.0%` / `16.85` | `76.0%` / `88.54` |
| **B6c: One-Class SVM (Tier 4, Bg Only)** | `57.5%` / `2.22` | `72.0%` / `9.29` | `78.5%` / `41.43` |
| **B7a: Logistic Regression (Tier 3)** | `70.5%` / `23.59` | `71.5%` / `24.17` | `71.5%` / `24.17` |
| **B7b: Depth-3 Decision Tree (Tier 3)** | `62.5%` / `6.74` | `62.5%` / `6.74` | `62.5%` / `6.74` |
| **B8: Operational Approx. (`Uncalibrated`)** | `3.5%` / `3.54` | `11.0%` / `7.81` | `36.5%` / `33.54` |
| **Tier 1: Gross GBDT (12 feats)** | `52.0%` / `4.27` | `61.5%` / `8.47` | `75.0%` / `39.87` |
| **Tier 1b: Gross + Weather GBDT (29 feats)** | `47.5%` / `1.97` | `56.0%` / `4.69` | `76.5%` / `27.62` |
| **Tier 4: Physics-Motivated Nuclear GBDT** | **`96.5%` / `9.87`** | **`96.5%` / `9.87`** | **`96.5%` / `9.87`** |

---

## 4. ML Feature Ablations (`data/processed/ml_ablation_summary.csv`)
| Ablation Variant (Seed 42, 95% Val Target) | Features | Uses Plumes? | Realized Test Det (%) | Det $\le 12\text{h}$ (%) | Test FA/Station-Yr |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unsupervised One-Class SVM (`B6c`, Tier 4)** | 59 | No | `78.50%` (`72.00%` @10FA) | `72.50%` | `41.43` (`9.29` @10FA) |
| **Non-ML Radon Projection + Peak Channel Z (`B4c`)** | 4 (non-ML) | No | `80.00%` (`89.00%` @10FA) | `74.50%` (`81.50%`) | `2.30` (`6.49` @10FA) |
| **Tier 4 without Weather/Bateman (`Ablation_Tier4_NoWeather`)** | 39 | Yes | `93.00%` | `89.50%` | `6.33` |
| **Tier 3 + Bateman Washout ODE Only (`Ablation_Tier3_PlusBatemanOnly`)** | 50 | Yes | `91.50%` | `85.50%` | `3.86` |
| **Tier 3 + Radon Stripping Only (`Ablation_Tier3_PlusStrippingOnly`)** | 57 | Yes | `95.50%` | `91.00%` | `7.48` |
| **Full Tier 4: Physics-Motivated Nuclear GBDT** | 59 | Yes | `96.50%` | `94.00%` | `9.87` |
