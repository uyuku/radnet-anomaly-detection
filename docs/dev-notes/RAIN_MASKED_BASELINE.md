# Control Experiment: Rain-Masked Baseline vs. Machine Learning

This control evaluates whether a trivial heuristic—suppressing rolling 7-day $z$-score alarms ($z_{\text{168h}} \ge k$) during precipitation or within $H$ hours after rain ($H \in \{0, 3, 6, 12\}$ hours)—can match the performance of the multimodal machine learning model without gamma spectrometry.

## Matched-Detection Comparison Table

Evaluation on the 2023–2025 test split (12.16 station-years, 200 synthetic test plumes):

| Model / Baseline Variant | Lowest False Alarms/yr at $\ge 87.12\%$ Detection | Lowest False Alarms/yr at $\ge 91.88\%$ Detection | Maximum Detection Rate |
| :--- | :---: | :---: | :---: |
| **Unmasked Baseline** ($H = \text{none}$) | 215.48 (at 87.5%, $k=1.40$) | 303.03 (at 92.5%, $k=0.75$) | 92.50% |
| **Rain Mask: Rain only** ($H = 0\text{h}$) | 197.47 (at 87.5%, $k=1.35$) | 285.77 (at 92.5%, $k=0.75$) | 92.50% |
| **Rain Mask: Rain + 3h** ($H = 3\text{h}$) | 179.80 (at 87.5%, $k=1.30$) | 251.65 (at 92.5%, $k=0.75$) | 92.50% |
| **Rain Mask: Rain + 6h** ($H = 6\text{h}$) | 181.77 (at 88.0%, $k=1.25$) | 245.98 (at 92.5%, $k=0.75$) | 92.50% |
| **Rain Mask: Rain + 12h** ($H = 12\text{h}$) | 177.49 (at 87.5%, $k=1.25$) | 238.66 (at 92.0%, $k=0.75$) | 92.00% |
| **Tier 3 ML (Weather Fusion GBDT)** | **2.01** (at 87.12% target) | **5.60** (at 91.88% target) | 97.90% |

*Source: `data/processed/rain_masked_baseline_roc.csv` and `data/processed/rigorous_benchmark_summary.csv`.*

## Interpretation

1. **False alarm reduction:** The rain mask does eliminate rain-coincident false alarms at fixed threshold levels (at $k=3.0$, false alarms decrease from 82.62 to 47.60 episodes/year), but it does not eliminate baseline false alarms caused by diurnal radon oscillations and non-precipitation fluctuations.
2. **Cost in detection:** Suppressing alarms during and after precipitation blinds the system to genuine radiological plumes that arrive during rainfall; at standard $k=3.0$, detection drops from 52.5% down to 41.0% under a 12-hour mask.
3. **High-detection operating regime:** To reach high detection rates ($\ge 87\%$ or $\ge 92\%$), the single-channel $z$-score threshold must be lowered to $k = 0.75\text{--}1.40\sigma$, triggering 177.49 to 238.66 false alarms per year even with the 12-hour rain mask active.
4. **Comparison with ML:** At matched detection ($\sim 92\%$), the Tier 3 weather-fused GBDT achieves 5.60 false alarms per year compared to 238.66 for the best rain-masked baseline (a 97.6% reduction in false alarms), demonstrating that 8-channel gamma spectrometry and multi-scale feature fusion provide discrimination that cannot be replicated by a simple precipitation muting rule.

> **Caveat:** This is a simple control on one split with synthetic plumes; it does not prove how either method performs on real releases.
