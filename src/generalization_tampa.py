r"""
Spatial and temporal generalization diagnostics, Tampa LOSO false-alarm SHAP
decomposition, local background calibration ($N \in \{3, 6, 12\}$ months),
normalization variants (`global_baseline`, `local_168h_baseline`,
`station_standardized`), and year-by-year drift analysis.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.uncertainty import clopper_pearson_ci, exact_poisson_rate_ci

HOURS_PER_YEAR = 8766.0
SPECTRAL_SHARE_COLS = [f"share_r0{k}" for k in range(2, 10)]
SPECTRAL_RATIO_COLS = ["share_high_energy", "ratio_r05_r07", "ratio_r03_r07", "ratio_r05_r03", "ratio_r02_r03"]
SPECTRAL_MEAN6H_COLS = [f"share_r0{k}_mean_6h" for k in range(2, 10)]
DRIFT_SENSITIVE_FEATURES = set(
    SPECTRAL_SHARE_COLS
    + SPECTRAL_RATIO_COLS
    + SPECTRAL_MEAN6H_COLS
    + [
        "z_score_global_dry",
        "dose_z_global_dry",
        "dose_to_gross_ratio",
        "radon_stripping_residual_l2",
        "stripped_excess_r03_z",
        "stripped_excess_r05_z",
        "stripped_excess_r07_z",
        "dose_spectral_residual_z",
        "dry_excess_interaction",
    ]
)


def detect_filter_change_indices(gross_cpm: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Detects hourly indices where the particulate filter tape advances (sharp negative
    step-drop <= -180 CPM across 1 to 2 valid hours).
    """
    g = np.asarray(gross_cpm, dtype=float)
    vm = np.asarray(valid_mask, dtype=bool)
    diff1 = np.zeros(len(g), dtype=float)
    diff1[1:] = g[1:] - g[:-1]
    is_drop = vm & np.isfinite(diff1) & (diff1 <= -180.0)
    return np.flatnonzero(is_drop)


def compute_hours_to_nearest_event(n_hours: int, event_indices: np.ndarray) -> np.ndarray:
    """Returns array of distance in hours from each index `0..n_hours-1` to the nearest index in `event_indices`."""
    if len(event_indices) == 0:
        return np.full(n_hours, 9999.0, dtype=float)
    ev = np.sort(np.asarray(event_indices, dtype=int))
    idx = np.arange(n_hours, dtype=int)
    pos = np.searchsorted(ev, idx)
    pos_left = np.clip(pos - 1, 0, len(ev) - 1)
    pos_right = np.clip(pos, 0, len(ev) - 1)
    dist = np.minimum(np.abs(idx - ev[pos_left]), np.abs(idx - ev[pos_right]))
    return dist.astype(float)


def extract_loso_false_alarm_episodes_with_shap(
    df_test_clean: pd.DataFrame,
    feats_test_clean: pd.DataFrame,
    valid_mask: np.ndarray,
    scores_clean: np.ndarray,
    threshold_tau: float,
    clf: Any,
    feature_cols: list[str],
    tier_name: str,
    station_id: str,
    train_share_stats: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """
    Lists every discrete false-alarm episode on a held-out station for a given model tier,
    extracting exact LightGBM tree SHAP values (`pred_contrib=True`) at each episode's peak
    score hour and classifying the episode into:
      - `rain-coincident`
      - `post-rain`
      - `filter-change adjacent`
      - `missing-data adjacent`
      - `instrument glitch`
      - `baseline or gain drift`
      - `unexplained`
    """
    vm = np.asarray(valid_mask, dtype=bool)
    p = np.asarray(scores_clean, dtype=float)
    is_on = (p >= float(threshold_tau)) & vm
    if not is_on.any():
        return pd.DataFrame()

    n_h = len(p)
    gross = pd.to_numeric(df_test_clean["gross_cpm"], errors="coerce").to_numpy(dtype=float)
    filter_idx = detect_filter_change_indices(gross, vm)
    missing_idx = np.flatnonzero(~vm)
    dist_filter = compute_hours_to_nearest_event(n_h, filter_idx)
    dist_missing = compute_hours_to_nearest_event(n_h, missing_idx)

    # Identify contiguous episode [start_idx, end_idx] spans
    starts = np.flatnonzero(is_on & (~np.concatenate([[False], is_on[:-1]])))
    ends = np.flatnonzero(is_on & (~np.concatenate([is_on[1:], [False]])))

    peak_indices = [int(s + np.argmax(p[s : e + 1])) for s, e in zip(starts, ends)]
    X_peaks = feats_test_clean.iloc[peak_indices][feature_cols]
    n_f = len(feature_cols)

    # LightGBM multiclass pred_contrib has shape (n_episodes, 3 * (n_f + 1))
    contrib_all = clf.predict(X_peaks, pred_contrib=True)
    # Contrast class 2 (fission) vs class 0 (background) log-odds contributions
    shap_fission = contrib_all[:, 2 * (n_f + 1) : 2 * (n_f + 1) + n_f] - contrib_all[:, 0:n_f]

    dts = pd.to_datetime(df_test_clean["utc_hour"] if "utc_hour" in df_test_clean.columns else df_test_clean["dt"])
    precip_1h = pd.to_numeric(feats_test_clean.get("precip_1h", pd.Series(np.zeros(n_h))), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    precip_6h = pd.to_numeric(feats_test_clean.get("precip_6h", pd.Series(np.zeros(n_h))), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    hsr = pd.to_numeric(feats_test_clean.get("hours_since_rain", pd.Series(np.full(n_h, 999.0))), errors="coerce").fillna(999.0).to_numpy(dtype=float)
    sh_r02 = pd.to_numeric(feats_test_clean.get("share_r02", pd.Series(np.nan, index=range(n_h))), errors="coerce").to_numpy(dtype=float)
    sh_r05 = pd.to_numeric(feats_test_clean.get("share_r05", pd.Series(np.nan, index=range(n_h))), errors="coerce").to_numpy(dtype=float)
    z_glob = pd.to_numeric(feats_test_clean.get("z_score_global_dry", pd.Series(np.nan, index=range(n_h))), errors="coerce").to_numpy(dtype=float)
    z_168 = pd.to_numeric(feats_test_clean.get("z_score_168h", pd.Series(np.nan, index=range(n_h))), errors="coerce").to_numpy(dtype=float)
    diff_1h = np.abs(pd.to_numeric(feats_test_clean.get("diff_1h_rel", pd.Series(np.zeros(n_h))), errors="coerce").fillna(0.0).to_numpy(dtype=float))
    loc_strip = pd.to_numeric(
        feats_test_clean.get("local_radon_stripping_residual_l2", pd.Series(np.nan, index=range(n_h))),
        errors="coerce",
    ).to_numpy(dtype=float)

    rows = []
    for ep_i, (s, e, pk) in enumerate(zip(starts, ends, peak_indices)):
        dur = int(e - s + 1)
        ep_p1 = float(np.max(precip_1h[s : e + 1]))
        ep_p6 = float(np.max(precip_6h[s : e + 1]))
        ep_hsr_start = float(hsr[s])
        ep_hsr_min = float(np.min(hsr[s : e + 1]))
        d_filt = float(np.min(dist_filter[s : e + 1]))
        d_miss = float(np.min(dist_missing[s : e + 1]))

        if ep_p1 > 0.0 or ep_hsr_min == 0.0:
            rain_state = "rain_coincident"
        elif ep_hsr_min <= 6.0:
            rain_state = "post_rain_1_to_6h"
        elif ep_hsr_min <= 24.0:
            rain_state = "post_rain_6_to_24h"
        else:
            rain_state = "dry"

        sh_vec = shap_fission[ep_i]
        top_order = np.argsort(-sh_vec)
        top1_f, top1_v = feature_cols[int(top_order[0])], float(sh_vec[int(top_order[0])])
        top2_f, top2_v = feature_cols[int(top_order[1])], float(sh_vec[int(top_order[1])])
        top3_f, top3_v = feature_cols[int(top_order[2])], float(sh_vec[int(top_order[2])])

        # Check spectral / baseline shift relative to training stations
        is_spectral_shifted = False
        if train_share_stats is not None:
            m2, sd2 = train_share_stats.get("share_r02", (0.50, 0.02))
            m5, sd5 = train_share_stats.get("share_r05", (0.043, 0.003))
            if abs(sh_r02[pk] - m2) > 1.5 * max(sd2, 1e-4) or abs(sh_r05[pk] - m5) > 1.5 * max(sd5, 1e-4):
                is_spectral_shifted = True
        if top1_f in DRIFT_SENSITIVE_FEATURES or top2_f in DRIFT_SENSITIVE_FEATURES:
            is_spectral_shifted = True

        # Hierarchical episode classification
        if rain_state == "rain_coincident":
            category = "rain-coincident"
        elif rain_state in ("post_rain_1_to_6h", "post_rain_6_to_24h"):
            category = "post-rain"
        elif d_filt <= 3.0:
            category = "filter-change adjacent"
        elif d_miss <= 3.0:
            category = "missing-data adjacent"
        elif dur == 1 and (diff_1h[pk] > 0.25 or abs(z_168[pk]) > 4.5):
            category = "instrument glitch"
        elif is_spectral_shifted:
            category = "baseline or gain drift"
        else:
            category = "unexplained"

        rows.append({
            "tier_name": tier_name,
            "held_out_station": station_id,
            "episode_id": f"{station_id}_{tier_name[:6].replace(' ', '')}_ep{ep_i + 1:04d}",
            "start_idx": int(s),
            "end_idx": int(e),
            "peak_idx": int(pk),
            "start_utc": str(dts.iloc[s]),
            "end_utc": str(dts.iloc[e]),
            "peak_utc": str(dts.iloc[pk]),
            "duration_hours": dur,
            "peak_score": round(float(p[pk]), 5),
            "mean_score": round(float(np.mean(p[s : e + 1])), 5),
            "rain_state": rain_state,
            "hours_since_rain_at_start": round(ep_hsr_start, 1),
            "precip_1h_mm_at_start": round(float(precip_1h[s]), 2),
            "precip_6h_mm_max": round(ep_p6, 2),
            "hours_to_nearest_filter_change": round(d_filt, 1),
            "hours_to_nearest_missing_gap": round(d_miss, 1),
            "gross_cpm_at_peak": round(float(gross[pk]), 1),
            "share_r02_at_peak": round(float(sh_r02[pk]), 4),
            "share_r05_at_peak": round(float(sh_r05[pk]), 4),
            "z_score_global_dry_at_peak": round(float(z_glob[pk]), 3),
            "z_score_168h_at_peak": round(float(z_168[pk]), 3),
            "local_radon_stripping_residual_l2_at_peak": round(float(loc_strip[pk]), 3) if np.isfinite(loc_strip[pk]) else np.nan,
            "top_contrib_feature_1": top1_f,
            "top_contrib_shap_1": round(top1_v, 4),
            "top_contrib_feature_2": top2_f,
            "top_contrib_shap_2": round(top2_v, 4),
            "top_contrib_feature_3": top3_f,
            "top_contrib_shap_3": round(top3_v, 4),
            "episode_category": category,
        })

    return pd.DataFrame(rows)


def audit_station_raw_data_quality(df_merged: pd.DataFrame, station_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Audits a station's 2017-2025 merged series for missingness, stuck values (>= 3 consecutive
    identical gross_cpm values), abrupt non-filter step changes (> 4*sigma 1-hour positive jumps),
    and year-by-year spectral/gross baseline drift without deleting any data.
    """
    df = df_merged.copy()
    dts = pd.to_datetime(df["utc_hour"] if "utc_hour" in df.columns else df["dt"])
    df["year"] = dts.dt.year

    chan_cols = [f"cpm_r0{k}" for k in range(2, 10)]
    valid_rad = df[chan_cols].notna().all(axis=1) & df["gross_cpm"].notna() & (df["gross_cpm"] > 0)
    gross = pd.to_numeric(df["gross_cpm"], errors="coerce").to_numpy(dtype=float)

    # Stuck-value runs: >= 3 consecutive valid hours with identical gross_cpm
    valid_gross = gross[valid_rad.to_numpy()]
    if len(valid_gross) >= 3:
        same1 = valid_gross[1:] == valid_gross[:-1]
        same2 = valid_gross[2:] == valid_gross[:-2]
        stuck_hours = int(np.sum(same1[1:] & same2))
    else:
        stuck_hours = 0

    # Non-filter abrupt step changes: 1-hour positive jump > 4 * dry_std while precip == 0
    precip = pd.to_numeric(df.get("precip_mm", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dry_mask = valid_rad.to_numpy() & (precip == 0.0)
    dry_std = float(np.nanstd(gross[dry_mask])) if np.sum(dry_mask) > 10 else 300.0
    diff1 = np.zeros(len(gross), dtype=float)
    diff1[1:] = gross[1:] - gross[:-1]
    abrupt_up_jumps = int(np.sum(dry_mask & (diff1 > 4.0 * max(dry_std, 50.0))))
    filter_drops = int(np.sum(valid_rad.to_numpy() & (diff1 <= -180.0)))

    # Missing data stretches
    not_valid = ~valid_rad.to_numpy()
    gap_starts = int(np.sum(not_valid & (~np.concatenate([[False], not_valid[:-1]]))))

    summary = {
        "station_id": station_id,
        "total_grid_hours": int(len(df)),
        "valid_complete_channel_hours": int(valid_rad.sum()),
        "completeness_pct": round(100.0 * float(valid_rad.mean()), 2),
        "missing_gap_episodes": gap_starts,
        "stuck_gross_cpm_hours_ge3": stuck_hours,
        "abrupt_dry_positive_jumps_gt4sigma": abrupt_up_jumps,
        "detected_filter_drop_hours": filter_drops,
        "dry_gross_mean_cpm": round(float(np.nanmean(gross[dry_mask])), 1) if np.any(dry_mask) else np.nan,
        "dry_gross_sd_cpm": round(dry_std, 1),
        "dry_share_r02_mean": round(float(np.nanmean(df.loc[dry_mask, "cpm_r02"].to_numpy(dtype=float) / gross[dry_mask])), 4) if np.any(dry_mask) else np.nan,
        "dry_share_r03_mean": round(float(np.nanmean(df.loc[dry_mask, "cpm_r03"].to_numpy(dtype=float) / gross[dry_mask])), 4) if np.any(dry_mask) else np.nan,
        "dry_share_r05_mean": round(float(np.nanmean(df.loc[dry_mask, "cpm_r05"].to_numpy(dtype=float) / gross[dry_mask])), 4) if np.any(dry_mask) else np.nan,
        "dry_share_r07_mean": round(float(np.nanmean(df.loc[dry_mask, "cpm_r07"].to_numpy(dtype=float) / gross[dry_mask])), 4) if np.any(dry_mask) else np.nan,
    }

    yr_rows = []
    for yr, grp in df.groupby("year"):
        g_v = grp[chan_cols].notna().all(axis=1) & grp["gross_cpm"].notna() & (grp["gross_cpm"] > 0)
        if g_v.sum() == 0:
            continue
        sub = grp.loc[g_v]
        g_arr = sub["gross_cpm"].to_numpy(dtype=float)
        yr_rows.append({
            "station_id": station_id,
            "year": int(yr),
            "valid_hours": int(len(sub)),
            "gross_mean_cpm": round(float(np.mean(g_arr)), 1),
            "gross_sd_cpm": round(float(np.std(g_arr)), 1),
            "share_r02_mean": round(float(np.mean(sub["cpm_r02"].to_numpy(dtype=float) / g_arr)), 4),
            "share_r03_mean": round(float(np.mean(sub["cpm_r03"].to_numpy(dtype=float) / g_arr)), 4),
            "share_r05_mean": round(float(np.mean(sub["cpm_r05"].to_numpy(dtype=float) / g_arr)), 4),
            "share_r07_mean": round(float(np.mean(sub["cpm_r07"].to_numpy(dtype=float) / g_arr)), 4),
        })

    return summary, pd.DataFrame(yr_rows)


def apply_normalization_variant(
    feats_df: pd.DataFrame,
    variant: str,
    station_clean_stats: dict[str, tuple[float, float]] | None = None,
    target_pool_stats: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """
    Transforms a feature DataFrame according to one of three normalization variants:
      1. `global_baseline`: unmodified features (uses raw channel shares and global dry z-scores).
      2. `local_168h_baseline`: replaces static station-level spectral shares, ratios, and global
         dry baselines with causal trailing 168-hour local deviations so the detector is invariant
         to static inter-station spectral/gain offsets:
           - `share_r0k` -> `share_r0k - rolling_168h_mean(share_r0k)`
           - `share_r0k_mean_6h` -> `share_r0k_mean_6h - rolling_168h_mean(share_r0k)`
           - `share_high_energy` -> `share_high_energy - rolling_168h_mean(share_high_energy)`
           - pairwise ratios `ratio_*` -> `(ratio - rolling_168h_mean(ratio)) / (rolling_168h_std(ratio) + 1e-4)`
           - `z_score_global_dry` -> `z_score_168h`
           - `radon_stripping_residual_l2` -> `local_radon_stripping_residual_l2`
           - `stripped_excess_r0{3,5,7}_z` -> `local_stripped_r0{3,5,7}_z`
      3. `station_standardized`: aligns each station's radiation/spectral features using its own
         pre-test clean dry mean and SD (`station_clean_stats`) mapped onto the training pool's
         clean dry mean and SD (`target_pool_stats`).
    """
    if variant == "global_baseline":
        return feats_df

    out = feats_df.copy()
    if variant == "local_168h_baseline":
        for k in range(2, 10):
            col = f"share_r0{k}"
            m6_col = f"share_r0{k}_mean_6h"
            if col in out.columns:
                s = pd.to_numeric(out[col], errors="coerce")
                loc_mean = s.rolling(window=168, min_periods=24).mean()
                out[col] = (s - loc_mean).fillna(0.0)
                if m6_col in out.columns:
                    s6 = pd.to_numeric(out[m6_col], errors="coerce")
                    out[m6_col] = (s6 - loc_mean).fillna(0.0)

        for r_col in SPECTRAL_RATIO_COLS:
            if r_col in out.columns:
                s = pd.to_numeric(out[r_col], errors="coerce")
                rm = s.rolling(window=168, min_periods=24).mean()
                rs = s.rolling(window=168, min_periods=24).std().clip(lower=1e-4)
                out[r_col] = ((s - rm) / rs).fillna(0.0)

        for d_col in ["dose_z_global_dry", "dose_to_gross_ratio", "dose_spectral_residual_z"]:
            if d_col in out.columns:
                s = pd.to_numeric(out[d_col], errors="coerce")
                rm = s.rolling(window=168, min_periods=24).mean()
                rs = s.rolling(window=168, min_periods=24).std().clip(lower=1e-4)
                out[d_col] = ((s - rm) / rs).fillna(0.0)

        if "z_score_168h" in out.columns and "z_score_global_dry" in out.columns:
            out["z_score_global_dry"] = out["z_score_168h"]
        if "local_radon_stripping_residual_l2" in out.columns and "radon_stripping_residual_l2" in out.columns:
            out["radon_stripping_residual_l2"] = out["local_radon_stripping_residual_l2"]
        for k in (3, 5, 7):
            g_c = f"stripped_excess_r0{k}_z"
            l_c = f"local_stripped_r0{k}_z"
            if l_c in out.columns and g_c in out.columns:
                out[g_c] = out[l_c]
        if "dry_excess_interaction" in out.columns and "z_score_168h" in out.columns and "hours_since_rain" in out.columns:
            z168 = pd.to_numeric(out["z_score_168h"], errors="coerce").fillna(0.0)
            hsr = pd.to_numeric(out["hours_since_rain"], errors="coerce").fillna(0.0)
            out["dry_excess_interaction"] = np.maximum(0.0, z168) * np.log1p(np.maximum(0.0, hsr))
        return out

    if variant == "station_standardized":
        if station_clean_stats is None or target_pool_stats is None:
            raise ValueError("station_standardized requires station_clean_stats and target_pool_stats")
        for col, (st_mu, st_sd) in station_clean_stats.items():
            if col in out.columns and col in target_pool_stats:
                tp_mu, tp_sd = target_pool_stats[col]
                s = pd.to_numeric(out[col], errors="coerce")
                out[col] = (s - st_mu) / max(st_sd, 1e-5) * max(tp_sd, 1e-5) + tp_mu
        return out

    raise ValueError(f"Unknown normalization variant: {variant!r}")


def compute_clean_feature_stats(
    feats_df: pd.DataFrame,
    valid_mask: np.ndarray,
    feature_cols: list[str],
) -> dict[str, tuple[float, float]]:
    """Computes (mean, std) for each feature column on valid clean hours."""
    vm = np.asarray(valid_mask, dtype=bool)
    sub = feats_df.loc[vm]
    stats: dict[str, tuple[float, float]] = {}
    for col in feature_cols:
        if col in sub.columns:
            arr = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]
            if len(finite) > 5:
                stats[col] = (float(np.mean(finite)), float(max(np.std(finite, ddof=1), 1e-5)))
            else:
                stats[col] = (0.0, 1.0)
    return stats


def select_local_calibration_threshold(
    local_clean_scores: np.ndarray,
    local_valid_mask: np.ndarray,
    fa_budget_per_yr: float,
    fallback_tau: float,
) -> float:
    """
    Selects decision threshold tau from an N-month local clean-background window
    so that the realized episode rate in that local clean window is <= `fa_budget_per_yr`.
    When `fa_budget_per_yr == 0.0` (zero-local-FA ceiling), sets tau slightly above the
    maximum observed clean score in the calibration window.
    """
    vm = np.asarray(local_valid_mask, dtype=bool)
    p = np.asarray(local_clean_scores, dtype=float)
    n_obs = int(vm.sum())
    if n_obs < 48:
        return float(fallback_tau)

    exp_yrs = n_obs / HOURS_PER_YEAR
    p_valid = p[vm & np.isfinite(p)]
    if len(p_valid) == 0:
        return float(fallback_tau)

    if fa_budget_per_yr <= 0.0:
        # Zero-local-FA ceiling with a small safety margin toward 1.0
        max_p = float(np.max(p_valid))
        return float(min(0.9995, max_p + 0.002 * (1.0 - max_p)))

    grid = np.unique(
        np.concatenate([
            np.linspace(0.30, 0.99, 140),
            np.quantile(p_valid, np.linspace(0.90, 1.0, 120)),
            [float(np.max(p_valid)) + 1e-5],
        ])
    )
    grid = np.sort(grid)

    on = (p[:, None] >= grid[None, :]) & vm[:, None]
    pr = np.vstack([np.zeros((1, len(grid)), dtype=bool), on[:-1]])
    ep_rates = (on & (~pr)).sum(axis=0) / exp_yrs

    el = np.flatnonzero(ep_rates <= fa_budget_per_yr + 1e-9)
    if len(el) > 0:
        return float(grid[int(el[0])])
    return float(grid[-1])


def evaluate_station_test_metrics_with_exact_cis(
    scores_clean: np.ndarray,
    scores_inj: np.ndarray,
    valid_mask: np.ndarray,
    test_plume_slices: list[dict[str, Any]],
    threshold_tau: float,
) -> dict[str, Any]:
    """
    Evaluates false-alarm episodes, exposure (station-years), Garwood exact 95% Poisson CI,
    plume detections, and Clopper-Pearson exact 95% binomial CI on a held-out station.
    """
    vm = np.asarray(valid_mask, dtype=bool)
    pc = np.asarray(scores_clean, dtype=float)
    pi = np.asarray(scores_inj, dtype=float)
    tau = float(threshold_tau)

    obs_h = int(vm.sum())
    exp_yrs = max(1e-6, obs_h / HOURS_PER_YEAR)
    on = (pc >= tau) & vm
    pr = np.concatenate([[False], on[:-1]])
    n_ep = int((on & (~pr)).sum())
    fa_yr = n_ep / exp_yrs
    fa_lo, fa_hi = exact_poisson_rate_ci(n_ep, exp_yrs, alpha=0.05)

    n_plumes = len(test_plume_slices)
    n_det = 0
    delays = []
    for sl in test_plume_slices:
        idx = sl["indices"]
        net = (pi[idx] >= tau) & (pc[idx] < tau)
        if np.any(net):
            n_det += 1
            first_i = int(np.flatnonzero(net)[0])
            delays.append(float(first_i))

    det_pct = 100.0 * n_det / max(1, n_plumes)
    det_lo, det_hi = clopper_pearson_ci(n_det, max(1, n_plumes), alpha=0.05)
    med_delay = float(np.median(delays)) if delays else np.nan

    return {
        "threshold_tau": round(tau, 5),
        "observed_hours": obs_h,
        "exposure_station_years": round(exp_yrs, 4),
        "false_alarm_episodes": n_ep,
        "fa_per_yr": round(fa_yr, 2),
        "fa_exact_poisson_ci_low": round(fa_lo, 2),
        "fa_exact_poisson_ci_high": round(fa_hi, 2),
        "n_plumes": n_plumes,
        "n_detected": n_det,
        "detection_pct": round(det_pct, 2),
        "det_clopper_pearson_ci_low": round(det_lo, 2),
        "det_clopper_pearson_ci_high": round(det_hi, 2),
        "median_delay_hours": round(med_delay, 2) if np.isfinite(med_delay) else np.nan,
    }
