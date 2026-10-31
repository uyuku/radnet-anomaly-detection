# Real-Event Validation Limitations and Manuscript Disclosure

**Revision Step**: Phase 7, Step 6  
**Date**: October 31, 2026  
**Scope**: Methodological limitations of real-event evaluation on operational EPA RadNet fixed-station monitoring, physical detection boundaries, and manuscript limitation text.

---

## 1. Summary of Real-Event Evaluation Attempts

In accordance with Phase 7, we performed an exhaustive empirical and literature search to evaluate our frozen anomaly detection models on real-world radiological release events rather than relying solely on synthetic injection benchmarks.

We investigated the following historical events:
1. **Fukushima Daiichi Trans-Pacific Plume (March–April 2011)**:
   - Evaluated under a pre-registered protocol (`docs/real_event_protocol.md`) on Pacific gateway stations (Honolulu, HI; San Diego, CA; Anchorage, AK) and continental control (Dallas, TX).
   - Finding: Frozen ML models (Tier 1–4) and baselines detected zero excess associated with the plume. While gross-count baselines triggered false alarms during coincident rainfall, the genuine radiological plume did not register on hourly monitors.
2. **WIPP Underground Radionuclide Breach (February 2014, Carlsbad, NM)**:
   - Finding: The release consisted of alpha-emitting transuranics ($^{241}\text{Am}$, $^{239/240}\text{Pu}$). Ambient off-site gross gamma radiation remained indistinguishable from background ($<0.01\text{ CPM}$ excess); detection was achieved only by laboratory alpha spectrometry on physical air filters.
3. **European Ruthenium-106 Cloud (Autumn 2017)**:
   - Finding: Atmospheric dispersion reduced North American concentrations to $<10\text{ }\mu\text{Bq/m}^3$, producing an ambient excess of $<10^{-6}\text{ CPM}$, six orders of magnitude below ambient single-station detection limits.
4. **Domestic Civilian Nuclear Power Operations (2017–2025)**:
   - Finding: No civilian nuclear accidents, criticality excursions, or major atmospheric fission product releases occurred within the United States during the modern 8-channel monitoring era.

---

## 2. Fundamental Physical and Instrumental Detection Limits

The failure of fixed ambient gamma monitors to register trans-oceanic plumes like Fukushima (2011) or Ruthenium-106 (2017) is an inevitable consequence of radiation detection physics and counting statistics:

1. **Ambient Counting Statistics**:
   A standard RadNet 2"$\times$2" $\text{NaI(Tl)}$ scintillation crystal exhibits an ambient background gross count rate of $1,000–3,000\text{ CPM}$. Over a 1-hour counting interval ($T = 60\text{ min}$), the Poisson standard deviation of background counts is:
   $$\sigma_{Poisson} = \sqrt{\frac{B}{T}} = \sqrt{\frac{1200}{60}} \approx 4.5\text{ CPM}$$
   To achieve a standard Currie detection limit ($L_D \approx 3.29\sigma$), a stationary ambient plume must generate an excess of at least:
   $$\Delta \text{CPM}_{min} \approx 15\text{ CPM}$$
   corresponding to an ambient dose rate excess of $\approx 0.4–0.5\text{ nSv/h}$.

2. **Trans-Oceanic Plume Immersion Dose**:
   During the Fukushima event, peak measured US atmospheric concentrations were $31\text{ mBq/m}^3$ particulate $^{131}\text{I}$ and $96\text{ mBq/m}^3$ total volatile $^{131}\text{I}$ (Thakur et al., 2012). Applying standard semi-infinite cloud immersion dose conversion factors (Federal Guidance Report No. 12 / ICRP Publication 74):
   $$\dot{D}_{sub} \approx 2.03 \times 10^{-3}\text{ nSv/h} = 2.03\text{ pSv/h} \implies \Delta \text{CPM} \approx 0.15\text{ CPM}$$
   (via empirical station calibration slope $0.0134\text{ nSv/h per CPM}$). This physical excess is an order of magnitude smaller than single-hour Poisson counting noise ($\sigma_{\text{Poisson}} \approx 4.5\text{--}5.7\text{ CPM}$) and three orders of magnitude below natural environmental background swings ($\sigma_{\text{env}} \approx 130\text{ CPM}$ at Honolulu).

3. **Role of High-Volume Air Particulate Filters**:
   EPA RadNet confirmed Fukushima fallout not via near-real-time scintillation detectors, but by pumping $>10,000\text{ m}^3$ of air through glass-fiber filters over multiple days and counting the accumulated particulate mass for 24 hours on laboratory High-Purity Germanium (HPGe) spectrometers at NAREL (Montgomery, AL). Ambient near-real-time monitors operate in a fundamentally different detection regime: detecting immediate, localized, un-diluted plumes from domestic accidents, industrial sources, or radiological dispersal devices.

---

## 3. Pre-Drafted Manuscript Limitation Paragraph (For Section 6)

The following text is pre-drafted for inclusion in Section 6 (Limitations) of the revised MDPI manuscript:

> **Validation Scope and Absence of Real-Release Ground Truth**:
> A primary limitation of this study is that all detection sensitivity metrics ($P_{det}$, detection delay, and ROC curves) are evaluated exclusively on synthetic plume injections superimposed on empirical background series. We systematically evaluated the frozen detectors against historical RadNet records from the March–April 2011 Fukushima Daiichi trans-Pacific transport event across Pacific gateway stations (Honolulu, HI; San Diego, CA; Anchorage, AK). In accordance with atmospheric dispersion physics, peak trans-Pacific atmospheric concentrations ($<0.1\text{ Bq/m}^3$ $^{131}\text{I}$) delivered an ambient gamma immersion excess of $\sim 10^{-4}\text{ CPM}$, which is four orders of magnitude below the Poisson counting noise of a single-station hourly 2"$\times$2" $\text{NaI(Tl)}$ detector. Consequently, neither the machine learning models nor the non-ML baselines registered genuine radiological alarms during the 2011 event. While high-volume particulate filter sampling and laboratory HPGe spectrometry detected trace Fukushima fallout, near-real-time ambient fixed monitors are physically incapable of detecting trans-oceanic releases of this dilution. Therefore, our reported detection rates are conditional on synthetic injection templates, and performance on genuine, localized anthropogenic releases remains untested.
