"""
Valid Plume-Level, Block-Bootstrap, Exact, and Multi-Catalog Uncertainty Estimation.

Replaces 20-seed solver-randomness p-values with statistically valid confidence intervals
that resample the actual sampling units of the benchmark:
  1. Detection rate (`cluster_bootstrap_detection`, `clopper_pearson_ci`):
     - Stratified cluster bootstrap over test plumes stratified by `(station_id, environment)`,
       B = 10,000 replicates.
     - Exact Clopper-Pearson binomial confidence intervals for single proportions.
  2. False-alarm rate (`block_bootstrap_fa_rate`, `exact_poisson_rate_ci`):
     - Block bootstrap over station-months (and station-weeks as a sensitivity check),
       counting discrete episode starts on the full continuous series before binning so
       episodes crossing block edges are never split or double-counted, B = 10,000.
     - Exact Garwood chi-square Poisson confidence intervals with exact exposure in
       station-years (including zero-episode counts).
  3. Paired comparisons (`paired_detector_bootstrap`):
     - Joint resampling of the same plume indices and the same station-month block indices
       for both detectors, reporting 95% CIs of `Delta FA/yr` and `Delta Detection %`.
  4. Matched-operating-point uncertainty (`matched_operating_point_bootstrap`):
     - Jointly resamples validation plumes and validation station-months, re-derives each
       detector's threshold tau*(b) on every bootstrap replicate so threshold-selection noise
       is included, and evaluates tau*(b) on the jointly resampled test fold.
  5. Multiplicity control (`holm_adjust_pvalues`):
     - Holm-Bonferroni step-down procedure for exploratory comparison families.
  6. Variance components (`compute_variance_components_anova`):
     - Decomposes benchmark spread across independent synthetic plume catalogs vs. LightGBM
       model random seeds.
  7. Coverage simulation (`verify_interval_coverage_simulation`):
     - Monte Carlo coverage check with known Bernoulli detection truth and block-clustered
       Poisson episode rate truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as sp_beta
from scipy.stats import chi2 as sp_chi2


def clopper_pearson_ci(
    k: int,
    n: int,
    alpha: float = 0.05,
    as_percent: bool = True,
) -> tuple[float, float]:
    """
    Computes exact two-sided (1 - alpha) Clopper-Pearson confidence interval for a
    binomial proportion k / n.
    """
    if n <= 0 or k < 0 or k > n:
        raise ValueError(f"Invalid binomial counts k={k}, n={n}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    lo = 0.0 if k == 0 else float(sp_beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(sp_beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    scale = 100.0 if as_percent else 1.0
    return (lo * scale, hi * scale)


def exact_poisson_rate_ci(
    episodes: int,
    exposure_years: float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Computes exact two-sided (1 - alpha) Garwood chi-square confidence interval for a
    Poisson rate `episodes / exposure_years` (including when `episodes == 0`).
    """
    if episodes < 0:
        raise ValueError(f"Episode count must be non-negative, got {episodes}.")
    if exposure_years <= 0 or not np.isfinite(exposure_years):
        raise ValueError(f"Exposure must be positive and finite, got {exposure_years}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    if episodes == 0:
        lo_count = 0.0
    else:
        lo_count = 0.5 * float(sp_chi2.ppf(alpha / 2.0, 2 * episodes))
    hi_count = 0.5 * float(sp_chi2.ppf(1.0 - alpha / 2.0, 2 * (episodes + 1)))
    return (lo_count / exposure_years, hi_count / exposure_years)


def cluster_bootstrap_detection(
    plume_df: pd.DataFrame,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
    stratify_cols: list[str] | None = None,
    detected_col: str = "detected",
) -> dict[str, float]:
    """
    Stratified cluster bootstrap over plumes (default strata: `station_id` x `environment`),
    plus exact Clopper-Pearson confidence intervals.
    """
    if plume_df.empty:
        raise ValueError("plume_df cannot be empty.")
    if stratify_cols is None:
        stratify_cols = [c for c in ["station_id", "environment"] if c in plume_df.columns]

    det_arr = plume_df[detected_col].to_numpy(dtype=int)
    n_total = len(det_arr)
    k_total = int(det_arr.sum())
    point_pct = 100.0 * k_total / n_total

    rng = np.random.default_rng(seed)
    boot_totals = np.zeros(n_boot, dtype=float)

    if stratify_cols:
        for _, grp in plume_df.groupby(stratify_cols, sort=False):
            vals = grp[detected_col].to_numpy(dtype=float)
            n_g = len(vals)
            idx = rng.integers(0, n_g, size=(n_boot, n_g))
            boot_totals += vals[idx].sum(axis=1)
    else:
        idx = rng.integers(0, n_total, size=(n_boot, n_total))
        boot_totals = det_arr[idx].sum(axis=1)

    boot_pcts = 100.0 * boot_totals / n_total
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    boot_lo, boot_hi = np.percentile(boot_pcts, [lo_q, hi_q])
    cp_lo, cp_hi = clopper_pearson_ci(k_total, n_total, alpha=alpha, as_percent=True)

    return {
        "n_plumes": int(n_total),
        "n_detected": int(k_total),
        "detection_pct_point": round(float(point_pct), 2),
        "boot_se_pct": round(float(np.std(boot_pcts, ddof=1)), 2),
        "boot_ci_low_pct": round(float(boot_lo), 2),
        "boot_ci_high_pct": round(float(boot_hi), 2),
        "clopper_pearson_low_pct": round(float(cp_lo), 2),
        "clopper_pearson_high_pct": round(float(cp_hi), 2),
    }


def block_bootstrap_fa_rate(
    block_df: pd.DataFrame,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
    episodes_col: str = "false_alarm_episodes",
    exposure_col: str = "exposure_station_years",
) -> dict[str, float]:
    """
    Block bootstrap over station-months (or station-weeks) where episode starts were
    counted on the continuous series prior to binning, plus exact Poisson rate intervals.
    """
    if block_df.empty:
        raise ValueError("block_df cannot be empty.")

    ep_arr = block_df[episodes_col].to_numpy(dtype=float)
    exp_arr = block_df[exposure_col].to_numpy(dtype=float)
    valid_blocks = exp_arr > 0
    ep_arr = ep_arr[valid_blocks]
    exp_arr = exp_arr[valid_blocks]

    n_blocks = len(ep_arr)
    if n_blocks == 0:
        raise ValueError("No blocks with positive exposure.")

    tot_ep = int(round(float(ep_arr.sum())))
    tot_exp = float(exp_arr.sum())
    point_rate = tot_ep / tot_exp

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    boot_ep = ep_arr[idx].sum(axis=1)
    boot_exp = np.maximum(1e-9, exp_arr[idx].sum(axis=1))
    boot_rates = boot_ep / boot_exp

    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    boot_lo, boot_hi = np.percentile(boot_rates, [lo_q, hi_q])
    pois_lo, pois_hi = exact_poisson_rate_ci(tot_ep, tot_exp, alpha=alpha)

    return {
        "n_blocks": int(n_blocks),
        "total_episodes": int(tot_ep),
        "exposure_station_years": round(float(tot_exp), 4),
        "fa_per_yr_point": round(float(point_rate), 2),
        "boot_se_fa_per_yr": round(float(np.std(boot_rates, ddof=1)), 2),
        "boot_ci_low_fa_per_yr": round(float(boot_lo), 2),
        "boot_ci_high_fa_per_yr": round(float(boot_hi), 2),
        "poisson_ci_low_fa_per_yr": round(float(pois_lo), 2),
        "poisson_ci_high_fa_per_yr": round(float(pois_hi), 2),
    }


def _two_sided_bootstrap_pvalue(boot_diffs: np.ndarray) -> float:
    """
    Computes two-sided tail probability `2 * min(P(diff <= 0), P(diff >= 0))` with
    plus-one finite-sample correction `(count + 1) / (B + 1)`.
    """
    b = len(boot_diffs)
    p_le = (np.sum(boot_diffs <= 0.0) + 1.0) / (b + 1.0)
    p_ge = (np.sum(boot_diffs >= 0.0) + 1.0) / (b + 1.0)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def holm_adjust_pvalues(
    pvalues: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> tuple[list[float], list[bool]]:
    """
    Applies the Holm-Bonferroni step-down procedure to a family of p-values.
    Returns `(adjusted_pvalues, reject_flags)`.
    """
    p_arr = np.asarray(pvalues, dtype=float)
    m = len(p_arr)
    if m == 0:
        return [], []
    order = np.argsort(p_arr)
    adj_sorted = np.zeros(m, dtype=float)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        mult = m - rank_idx
        val = min(1.0, p_arr[orig_idx] * mult)
        running_max = max(running_max, val)
        adj_sorted[rank_idx] = running_max

    adj = np.zeros(m, dtype=float)
    for rank_idx, orig_idx in enumerate(order):
        adj[orig_idx] = adj_sorted[rank_idx]
    reject = [bool(p < alpha) for p in adj]
    return [float(x) for x in adj], reject


def paired_detector_bootstrap(
    plume_a: pd.DataFrame,
    plume_b: pd.DataFrame,
    block_a: pd.DataFrame,
    block_b: pd.DataFrame,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Jointly resamples the exact same stratified test plumes and the exact same test
    station-month blocks for Detector A and Detector B (`A - B`), returning paired
    95% bootstrap confidence intervals and two-sided bootstrap p-values for both
    `Delta Detection (%)` and `Delta FA/yr`.
    """
    merged_p = pd.merge(
        plume_a[["plume_id", "station_id", "environment", "detected"]],
        plume_b[["plume_id", "detected"]],
        on="plume_id",
        suffixes=("_a", "_b"),
    )
    if merged_p.empty:
        raise ValueError("No overlapping plumes between Detector A and Detector B.")

    block_key = "year_month" if "year_month" in block_a.columns else "block_id"
    merged_b = pd.merge(
        block_a[["station_id", block_key, "false_alarm_episodes", "exposure_station_years"]],
        block_b[["station_id", block_key, "false_alarm_episodes"]],
        on=["station_id", block_key],
        suffixes=("_a", "_b"),
    )
    merged_b = merged_b[merged_b["exposure_station_years"] > 0].reset_index(drop=True)
    if merged_b.empty:
        raise ValueError("No overlapping exposure blocks between Detector A and Detector B.")

    n_plumes = len(merged_p)
    det_a_pt = 100.0 * float(merged_p["detected_a"].mean())
    det_b_pt = 100.0 * float(merged_p["detected_b"].mean())
    delta_det_pt = det_a_pt - det_b_pt

    tot_exp = float(merged_b["exposure_station_years"].sum())
    fa_a_pt = float(merged_b["false_alarm_episodes_a"].sum()) / tot_exp
    fa_b_pt = float(merged_b["false_alarm_episodes_b"].sum()) / tot_exp
    delta_fa_pt = fa_a_pt - fa_b_pt

    rng = np.random.default_rng(seed)

    # Joint stratified plume bootstrap
    boot_det_a = np.zeros(n_boot, dtype=float)
    boot_det_b = np.zeros(n_boot, dtype=float)
    for _, grp in merged_p.groupby(["station_id", "environment"], sort=False):
        da = grp["detected_a"].to_numpy(dtype=float)
        db = grp["detected_b"].to_numpy(dtype=float)
        n_g = len(da)
        idx = rng.integers(0, n_g, size=(n_boot, n_g))
        boot_det_a += da[idx].sum(axis=1)
        boot_det_b += db[idx].sum(axis=1)
    boot_delta_det = 100.0 * (boot_det_a - boot_det_b) / n_plumes

    # Joint station-month block bootstrap
    n_blocks = len(merged_b)
    ep_a = merged_b["false_alarm_episodes_a"].to_numpy(dtype=float)
    ep_b = merged_b["false_alarm_episodes_b"].to_numpy(dtype=float)
    exp_v = merged_b["exposure_station_years"].to_numpy(dtype=float)

    b_idx = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    boot_exp = np.maximum(1e-9, exp_v[b_idx].sum(axis=1))
    boot_fa_a = ep_a[b_idx].sum(axis=1) / boot_exp
    boot_fa_b = ep_b[b_idx].sum(axis=1) / boot_exp
    boot_delta_fa = boot_fa_a - boot_fa_b

    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    det_lo, det_hi = np.percentile(boot_delta_det, [lo_q, hi_q])
    fa_lo, fa_hi = np.percentile(boot_delta_fa, [lo_q, hi_q])

    p_det = _two_sided_bootstrap_pvalue(boot_delta_det)
    p_fa = _two_sided_bootstrap_pvalue(boot_delta_fa)

    return {
        "det_a_pct": round(det_a_pt, 2),
        "det_b_pct": round(det_b_pt, 2),
        "delta_det_pct_point": round(delta_det_pt, 2),
        "delta_det_ci_low_pct": round(float(det_lo), 2),
        "delta_det_ci_high_pct": round(float(det_hi), 2),
        "delta_det_boot_pvalue": round(float(p_det), 5),
        "ci_excludes_zero_det": bool(det_lo > 0.0 or det_hi < 0.0),
        "fa_a_per_yr": round(fa_a_pt, 2),
        "fa_b_per_yr": round(fa_b_pt, 2),
        "delta_fa_per_yr_point": round(delta_fa_pt, 2),
        "delta_fa_ci_low_per_yr": round(float(fa_lo), 2),
        "delta_fa_ci_high_per_yr": round(float(fa_hi), 2),
        "delta_fa_boot_pvalue": round(float(p_fa), 5),
        "ci_excludes_zero_fa": bool(fa_lo > 0.0 or fa_hi < 0.0),
    }


def matched_operating_point_bootstrap(
    grid_a: np.ndarray,
    val_plume_det_matrix_a: np.ndarray,
    val_block_ep_matrix_a: np.ndarray,
    val_block_exp_years: np.ndarray,
    test_plume_det_matrix_a: np.ndarray,
    test_block_ep_matrix_a: np.ndarray,
    test_block_exp_years: np.ndarray,
    grid_b: np.ndarray | None = None,
    val_plume_det_matrix_b: np.ndarray | None = None,
    val_block_ep_matrix_b: np.ndarray | None = None,
    test_plume_det_matrix_b: np.ndarray | None = None,
    test_block_ep_matrix_b: np.ndarray | None = None,
    mode: str = "detection_target",
    target_value: float = 0.90,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Step 1d: Resamples validation plumes and validation station-month blocks, re-derives
    each detector's threshold tau*(b) from the resampled validation fold, and evaluates
    tau*(b) on the jointly resampled test plumes and test station-month blocks.

    Matrix shapes:
      - `val_plume_det_matrix`: `(n_val_plumes, n_grid)` binary/float indicator of whether
        plume `i` is detected at grid threshold `g`.
      - `val_block_ep_matrix`: `(n_val_blocks, n_grid)` episode starts in block `m` at grid `g`.
      - `test_plume_det_matrix`: `(n_test_plumes, n_grid)`
      - `test_block_ep_matrix`: `(n_test_blocks, n_grid)`
    """
    rng = np.random.default_rng(seed)
    n_vp = val_plume_det_matrix_a.shape[0]
    n_vb = val_block_ep_matrix_a.shape[0]
    n_tp = test_plume_det_matrix_a.shape[0]
    n_tb = test_block_ep_matrix_a.shape[0]

    vp_idx = rng.integers(0, n_vp, size=(n_boot, n_vp))
    vb_idx = rng.integers(0, n_vb, size=(n_boot, n_vb))
    tp_idx = rng.integers(0, n_tp, size=(n_boot, n_tp))
    tb_idx = rng.integers(0, n_tb, size=(n_boot, n_tb))

    def _select_and_eval_single_detector(
        grid: np.ndarray,
        vp_mat: np.ndarray,
        vb_mat: np.ndarray,
        tp_mat: np.ndarray,
        tb_mat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Vectorized in chunks of 500 bootstrap replicates for low memory footprint
        taus_out = np.zeros(n_boot, dtype=float)
        test_det_out = np.zeros(n_boot, dtype=float)
        test_fa_out = np.zeros(n_boot, dtype=float)

        chunk = 500
        for s in range(0, n_boot, chunk):
            e = min(n_boot, s + chunk)
            c_vp = vp_idx[s:e]
            c_vb = vb_idx[s:e]
            c_tp = tp_idx[s:e]
            c_tb = tb_idx[s:e]

            # (chunk, n_grid)
            val_det_b = vp_mat[c_vp].mean(axis=1)
            val_exp_b = np.maximum(1e-9, val_block_exp_years[c_vb].sum(axis=1))[:, None]
            val_fa_b = vb_mat[c_vb].sum(axis=1) / val_exp_b

            test_det_b = tp_mat[c_tp].mean(axis=1) * 100.0
            test_exp_b = np.maximum(1e-9, test_block_exp_years[c_tb].sum(axis=1))[:, None]
            test_fa_b = tb_mat[c_tb].sum(axis=1) / test_exp_b

            for r in range(e - s):
                vd = val_det_b[r]
                vf = val_fa_b[r]
                if mode == "detection_target":
                    el = np.flatnonzero(vd >= target_value - 1e-9)
                    if len(el) == 0:
                        el = np.flatnonzero(vd >= vd.max() - 1e-9)
                    best_g = int(el[np.argmin(vf[el])])
                elif mode == "fa_budget":
                    el = np.flatnonzero(vf <= target_value + 1e-9)
                    if len(el) > 0:
                        # Lowest threshold tau whose validation FA/yr <= target_value (matching select_threshold_by_fa_budget)
                        best_g = int(el[0])
                    else:
                        best_g = int(np.argmin(vf))
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                taus_out[s + r] = float(grid[best_g])
                test_det_out[s + r] = float(test_det_b[r, best_g])
                test_fa_out[s + r] = float(test_fa_b[r, best_g])

        return taus_out, test_det_out, test_fa_out

    taus_a, det_a, fa_a = _select_and_eval_single_detector(
        grid_a, val_plume_det_matrix_a, val_block_ep_matrix_a, test_plume_det_matrix_a, test_block_ep_matrix_a
    )
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)

    res: dict[str, Any] = {
        "tau_a_mean": round(float(np.mean(taus_a)), 5),
        "tau_a_std": round(float(np.std(taus_a, ddof=1)), 5),
        "test_det_a_mean": round(float(np.mean(det_a)), 2),
        "test_det_a_ci_low": round(float(np.percentile(det_a, lo_q)), 2),
        "test_det_a_ci_high": round(float(np.percentile(det_a, hi_q)), 2),
        "test_fa_a_mean": round(float(np.mean(fa_a)), 2),
        "test_fa_a_ci_low": round(float(np.percentile(fa_a, lo_q)), 2),
        "test_fa_a_ci_high": round(float(np.percentile(fa_a, hi_q)), 2),
    }

    if (
        grid_b is not None
        and val_plume_det_matrix_b is not None
        and val_block_ep_matrix_b is not None
        and test_plume_det_matrix_b is not None
        and test_block_ep_matrix_b is not None
    ):
        taus_b, det_b, fa_b = _select_and_eval_single_detector(
            grid_b, val_plume_det_matrix_b, val_block_ep_matrix_b, test_plume_det_matrix_b, test_block_ep_matrix_b
        )
        d_det = det_a - det_b
        d_fa = fa_a - fa_b
        d_det_lo, d_det_hi = np.percentile(d_det, [lo_q, hi_q])
        d_fa_lo, d_fa_hi = np.percentile(d_fa, [lo_q, hi_q])
        res.update({
            "tau_b_mean": round(float(np.mean(taus_b)), 5),
            "tau_b_std": round(float(np.std(taus_b, ddof=1)), 5),
            "test_det_b_mean": round(float(np.mean(det_b)), 2),
            "test_det_b_ci_low": round(float(np.percentile(det_b, lo_q)), 2),
            "test_det_b_ci_high": round(float(np.percentile(det_b, hi_q)), 2),
            "test_fa_b_mean": round(float(np.mean(fa_b)), 2),
            "test_fa_b_ci_low": round(float(np.percentile(fa_b, lo_q)), 2),
            "test_fa_b_ci_high": round(float(np.percentile(fa_b, hi_q)), 2),
            "delta_det_mean_pct": round(float(np.mean(d_det)), 2),
            "delta_det_ci_low_pct": round(float(d_det_lo), 2),
            "delta_det_ci_high_pct": round(float(d_det_hi), 2),
            "delta_det_boot_pvalue": round(_two_sided_bootstrap_pvalue(d_det), 5),
            "ci_excludes_zero_det": bool(d_det_lo > 0.0 or d_det_hi < 0.0),
            "delta_fa_mean_per_yr": round(float(np.mean(d_fa)), 2),
            "delta_fa_ci_low_per_yr": round(float(d_fa_lo), 2),
            "delta_fa_ci_high_per_yr": round(float(d_fa_hi), 2),
            "delta_fa_boot_pvalue": round(_two_sided_bootstrap_pvalue(d_fa), 5),
            "ci_excludes_zero_fa": bool(d_fa_lo > 0.0 or d_fa_hi < 0.0),
        })

    return res


def compute_variance_components_anova(
    runs_df: pd.DataFrame,
    metric_col: str,
    catalog_col: str = "catalog_seed",
    model_seed_col: str = "model_seed",
) -> dict[str, float]:
    """
    One-way random-effects ANOVA variance decomposition for a balanced C catalogs x S model seeds
    experiment:
      Y_{cs} = mu + alpha_c + eps_{cs},
      where Var(alpha_c) = max(0, (MS_between - MS_within) / S) is the between-catalog variance
      and Var(eps_{cs}) = MS_within is the within-catalog (model seed + residual) variance.
    """
    piv = runs_df.pivot(index=catalog_col, columns=model_seed_col, values=metric_col)
    vals = piv.to_numpy(dtype=float)
    c_num, s_num = vals.shape
    grand_mean = float(np.mean(vals))
    cat_means = np.mean(vals, axis=1)

    ss_between = float(s_num * np.sum((cat_means - grand_mean) ** 2))
    ss_within = float(np.sum((vals - cat_means[:, None]) ** 2))

    ms_between = ss_between / max(1, c_num - 1)
    ms_within = ss_within / max(1, c_num * (s_num - 1))

    var_catalog = max(0.0, (ms_between - ms_within) / s_num)
    var_seed = ms_within
    var_total = var_catalog + var_seed

    pct_cat = 100.0 * var_catalog / var_total if var_total > 0 else 0.0
    pct_seed = 100.0 * var_seed / var_total if var_total > 0 else 0.0

    return {
        "n_catalogs": int(c_num),
        "n_model_seeds": int(s_num),
        "grand_mean": round(grand_mean, 3),
        "var_catalog": round(var_catalog, 5),
        "var_model_seed": round(var_seed, 5),
        "var_total": round(var_total, 5),
        "sd_catalog": round(float(np.sqrt(var_catalog)), 3),
        "sd_model_seed": round(float(np.sqrt(var_seed)), 3),
        "sd_total": round(float(np.sqrt(var_total)), 3),
        "pct_var_catalog": round(pct_cat, 1),
        "pct_var_model_seed": round(pct_seed, 1),
        "catalog_min_mean": round(float(np.min(cat_means)), 2),
        "catalog_max_mean": round(float(np.max(cat_means)), 2),
    }


def verify_interval_coverage_simulation(
    n_trials: int = 500,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Step 5: Monte Carlo coverage verification with known ground truth:
      1. Synthetic Bernoulli detections across 200 plumes stratified into 10 strata
         (5 stations x 2 environments, 20 plumes per stratum) with known true detection
         probabilities `p_true in {0.80, 0.90}`.
      2. Synthetic Poisson episode counts across 180 station-months (each with exposure
         1/12 station-year, total 15.0 station-years) with known true rates
         `lambda_true in {5.0, 10.0}` episodes/station-year, plus an overdispersed
         Gamma-Poisson block-clustered setting (`dispersion_k = 20.0`).
    Returns a DataFrame reporting empirical 95% CI coverage and mean interval width.
    """
    rng = np.random.default_rng(seed)
    rows = []

    # 1. Detection rate coverage simulation
    stations = [f"st_{i}" for i in range(5)]
    envs = ["dry", "rain"]
    strata_meta = []
    for st in stations:
        for ev in envs:
            for _ in range(20):
                strata_meta.append((st, ev))
    base_plume_df = pd.DataFrame(strata_meta, columns=["station_id", "environment"])
    n_plumes = len(base_plume_df)

    for p_true in (0.80, 0.90):
        true_pct = 100.0 * p_true
        cov_boot = 0
        cov_cp = 0
        widths_boot = []
        widths_cp = []

        for t in range(n_trials):
            det_sim = rng.binomial(1, p_true, size=n_plumes)
            df_sim = base_plume_df.copy()
            df_sim["detected"] = det_sim
            res = cluster_bootstrap_detection(
                df_sim, n_boot=n_boot, alpha=alpha, seed=seed + t * 13 + 1
            )
            if res["boot_ci_low_pct"] <= true_pct <= res["boot_ci_high_pct"]:
                cov_boot += 1
            if res["clopper_pearson_low_pct"] <= true_pct <= res["clopper_pearson_high_pct"]:
                cov_cp += 1
            widths_boot.append(res["boot_ci_high_pct"] - res["boot_ci_low_pct"])
            widths_cp.append(res["clopper_pearson_high_pct"] - res["clopper_pearson_low_pct"])

        rows.append({
            "metric_family": "Detection_Proportion",
            "method": "Stratified_Plume_Cluster_Bootstrap",
            "true_parameter": true_pct,
            "n_units": n_plumes,
            "n_trials": n_trials,
            "nominal_coverage_pct": round(100.0 * (1.0 - alpha), 1),
            "empirical_coverage_pct": round(100.0 * cov_boot / n_trials, 2),
            "mean_interval_width": round(float(np.mean(widths_boot)), 2),
            "passes_nominal_band": bool(92.5 <= 100.0 * cov_boot / n_trials <= 98.0),
        })
        rows.append({
            "metric_family": "Detection_Proportion",
            "method": "Exact_Clopper_Pearson",
            "true_parameter": true_pct,
            "n_units": n_plumes,
            "n_trials": n_trials,
            "nominal_coverage_pct": round(100.0 * (1.0 - alpha), 1),
            "empirical_coverage_pct": round(100.0 * cov_cp / n_trials, 2),
            "mean_interval_width": round(float(np.mean(widths_cp)), 2),
            "passes_nominal_band": bool(92.5 <= 100.0 * cov_cp / n_trials <= 98.5),
        })

    # 2. False-alarm rate coverage simulation (180 station-months, exposure = 1/12 yr each)
    n_months = 180
    exp_per_month = 1.0 / 12.0
    base_block_df = pd.DataFrame({
        "exposure_station_years": np.full(n_months, exp_per_month, dtype=float)
    })

    for lam_true in (5.0, 10.0):
        cov_boot = 0
        cov_pois = 0
        widths_boot = []
        widths_pois = []

        for t in range(n_trials):
            ep_sim = rng.poisson(lam_true * exp_per_month, size=n_months)
            df_b = base_block_df.copy()
            df_b["false_alarm_episodes"] = ep_sim
            res = block_bootstrap_fa_rate(
                df_b, n_boot=n_boot, alpha=alpha, seed=seed + t * 17 + 7
            )
            if res["boot_ci_low_fa_per_yr"] <= lam_true <= res["boot_ci_high_fa_per_yr"]:
                cov_boot += 1
            if res["poisson_ci_low_fa_per_yr"] <= lam_true <= res["poisson_ci_high_fa_per_yr"]:
                cov_pois += 1
            widths_boot.append(res["boot_ci_high_fa_per_yr"] - res["boot_ci_low_fa_per_yr"])
            widths_pois.append(res["poisson_ci_high_fa_per_yr"] - res["poisson_ci_low_fa_per_yr"])

        rows.append({
            "metric_family": "False_Alarm_Rate_Poisson",
            "method": "Station_Month_Block_Bootstrap",
            "true_parameter": lam_true,
            "n_units": n_months,
            "n_trials": n_trials,
            "nominal_coverage_pct": round(100.0 * (1.0 - alpha), 1),
            "empirical_coverage_pct": round(100.0 * cov_boot / n_trials, 2),
            "mean_interval_width": round(float(np.mean(widths_boot)), 2),
            "passes_nominal_band": bool(92.5 <= 100.0 * cov_boot / n_trials <= 98.0),
        })
        rows.append({
            "metric_family": "False_Alarm_Rate_Poisson",
            "method": "Exact_Garwood_Poisson",
            "true_parameter": lam_true,
            "n_units": n_months,
            "n_trials": n_trials,
            "nominal_coverage_pct": round(100.0 * (1.0 - alpha), 1),
            "empirical_coverage_pct": round(100.0 * cov_pois / n_trials, 2),
            "mean_interval_width": round(float(np.mean(widths_pois)), 2),
            "passes_nominal_band": bool(92.5 <= 100.0 * cov_pois / n_trials <= 98.5),
        })

    # 3. Block-clustered (overdispersed Gamma-Poisson) false-alarm rate coverage check
    lam_clust = 10.0
    k_disp = 15.0
    cov_boot_cl = 0
    widths_boot_cl = []
    for t in range(n_trials):
        g_rates = rng.gamma(shape=k_disp, scale=(lam_clust * exp_per_month) / k_disp, size=n_months)
        ep_sim = rng.poisson(g_rates)
        df_b = base_block_df.copy()
        df_b["false_alarm_episodes"] = ep_sim
        res = block_bootstrap_fa_rate(df_b, n_boot=n_boot, alpha=alpha, seed=seed + t * 23 + 11)
        if res["boot_ci_low_fa_per_yr"] <= lam_clust <= res["boot_ci_high_fa_per_yr"]:
            cov_boot_cl += 1
        widths_boot_cl.append(res["boot_ci_high_fa_per_yr"] - res["boot_ci_low_fa_per_yr"])

    rows.append({
        "metric_family": "False_Alarm_Rate_Block_Clustered_Overdispersed",
        "method": "Station_Month_Block_Bootstrap",
        "true_parameter": lam_clust,
        "n_units": n_months,
        "n_trials": n_trials,
        "nominal_coverage_pct": round(100.0 * (1.0 - alpha), 1),
        "empirical_coverage_pct": round(100.0 * cov_boot_cl / n_trials, 2),
        "mean_interval_width": round(float(np.mean(widths_boot_cl)), 2),
        "passes_nominal_band": bool(92.5 <= 100.0 * cov_boot_cl / n_trials <= 98.0),
    })

    return pd.DataFrame(rows)
