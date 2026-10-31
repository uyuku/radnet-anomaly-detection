"""
Executes the complete spatial, climatic, and temporal generalization
evaluation and Tampa LOSO diagnostic suite (`Part A` through `Part F`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baselines import score_b4c_radon_projection_combined
from src.detector_response import (
    apply_counting_noise,
    compute_independent_injected_dose_rate,
    sample_derived_spectrum,
)
from src.feature_engineering import (
    STATION_DRY_BASELINES,
    TIER_2_FEATURES,
    TIER_3_FEATURES,
    TIER_4_NUCLEAR_FEATURES,
    compute_features_for_series,
)
from src.generalization_tampa import (
    SPECTRAL_SHARE_COLS,
    apply_normalization_variant,
    audit_station_raw_data_quality,
    compute_clean_feature_stats,
    evaluate_station_test_metrics_with_exact_cis,
    extract_loso_false_alarm_episodes_with_shap,
    select_local_calibration_threshold,
)
from src.nuclear_physics import (
    CHANNELS,
    DEFAULT_RADON_BASIS,
    compute_channel_dose_response_vector,
    compute_multinuclide_injection_timeseries,
)
from src.synthetic_injection import classify_snr_bin
from src.uncertainty import clopper_pearson_ci, exact_poisson_rate_ci

HOURS_PER_YEAR = 8766.0
LABEL_MAP = {"normal": 0, "radon_washout": 1, "fission_product": 2}

PILOT_STATIONS = [
    {"id": "al_birmingham", "name": "Birmingham, AL", "climate_group": "Humid_Subtropical_SE", "climate_desc": "Humid Subtropical (Cfa)"},
    {"id": "dc_washington", "name": "Washington, DC", "climate_group": "Temperate_MidAtlantic", "climate_desc": "Mid-Atlantic Temperate (Cfa)"},
    {"id": "ca_san_diego", "name": "San Diego, CA", "climate_group": "Coastal_Mediterranean", "climate_desc": "Coastal Mediterranean (CSa)"},
    {"id": "tx_dallas", "name": "Dallas, TX", "climate_group": "Convective_Plains", "climate_desc": "Southern Plains Convective (Cfa)"},
    {"id": "fl_tampa", "name": "Tampa, FL", "climate_group": "Subtropical_Gulf_Coastal", "climate_desc": "Gulf Subtropical Coastal (Cfa)"},
]

EXTENDED_STATIONS = [
    {"id": "az_phoenix", "name": "Phoenix, AZ", "climate_group": "Arid_Interior", "climate_desc": "Arid Interior Desert (BWh)"},
    {"id": "ne_omaha", "name": "Omaha, NE", "climate_group": "Cold_Continental", "climate_desc": "Cold Continental (Dfa)"},
    {"id": "hi_honolulu", "name": "Honolulu, HI", "climate_group": "Tropical_Maritime", "climate_desc": "Tropical Trade-Wind (As)"},
    {"id": "ak_anchorage", "name": "Anchorage, AK", "climate_group": "High_Latitude_Subarctic", "climate_desc": "High-Latitude Subarctic (Dfc)"},
    {"id": "la_baton_rouge", "name": "Baton Rouge, LA", "climate_group": "Subtropical_Gulf_Coastal", "climate_desc": "High-Rainfall Gulf Coastal (Cfa)"},
    {"id": "vt_burlington", "name": "Burlington, VT", "climate_group": "Cold_Continental", "climate_desc": "Northern Cold Continental (Dfb)"},
]

ALL_11_STATIONS = PILOT_STATIONS + EXTENDED_STATIONS


def _compute_station_dry_baseline_from_window(
    df_merged: pd.DataFrame,
    start_dt: str,
    end_dt: str,
    fallback_basis: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Computes dry baseline parameters (`mu`, `sigma`, `mu_dose`, `sigma_dose`,
    `channel_mu_dry`, `channel_sigma_dry`, `radon_washout_basis`, `k_dose_channel`)
    strictly from clean observations within `[start_dt, end_dt)`.
    """
    dts = pd.to_datetime(df_merged["utc_hour"] if "utc_hour" in df_merged.columns else df_merged["dt"])
    sub = df_merged[(dts >= start_dt) & (dts < end_dt)].copy()
    sub["precip_24h"] = sub["precip_1h_mm"].rolling(24, min_periods=12).sum()

    valid = (
        sub["has_radnet_obs"].fillna(False).astype(bool)
        & sub["rad_complete_channels"].fillna(False).astype(bool)
        & (sub["gross_cpm"] > 0)
    )
    dry_mask = valid & (sub["precip_1h_mm"] == 0.0) & (sub["precip_24h"] == 0.0)
    dry = sub[dry_mask]
    if len(dry) < 24:
        dry = sub[valid]

    gross = dry["gross_cpm"].to_numpy(dtype=float)
    mu = float(np.nanmean(gross)) if len(gross) > 0 else 2500.0
    sigma = float(max(20.0, np.nanstd(gross))) if len(gross) > 1 else 300.0

    dose = pd.to_numeric(dry["dose_rate_nsvh"], errors="coerce")
    dose = dose[dose.notna() & (dose > 0)]
    mu_dose = float(dose.mean()) if len(dose) > 0 else 45.0
    sigma_dose = float(max(1.0, dose.std())) if len(dose) > 1 else 5.0

    ch_mu = [round(float(dry[c].mean()), 2) if len(dry) > 0 else 100.0 for c in CHANNELS]
    ch_sd = [round(max(1.0, float(dry[c].std())), 2) if len(dry) > 1 else 15.0 for c in CHANNELS]

    # Estimate radon washout basis from rain hours in window
    rain_df = sub[valid & (sub["precip_1h_mm"] >= 1.0)]
    basis = fallback_basis.copy() if fallback_basis is not None else DEFAULT_RADON_BASIS.copy()
    n_w = 0
    if not rain_df.empty:
        exc = rain_df[CHANNELS].to_numpy(dtype=float) - np.asarray(ch_mu, dtype=float)[None, :]
        surge = np.sum(exc, axis=1) > 80.0
        if np.any(surge):
            pos = np.maximum(0.0, exc[surge])
            raw_s = np.sum(pos, axis=0)
            n_w = int(np.sum(surge))
            if np.sum(raw_s) > 0:
                st_b = raw_s / np.sum(raw_s)
                w = float(n_w) / float(n_w + 60.0)
                basis = w * st_b + (1.0 - w) * basis
    basis = np.maximum(1e-4, basis)
    basis = basis / np.sum(basis)

    slope = mu_dose / max(mu, 100.0) * 0.95
    k_vec = compute_channel_dose_response_vector(slope, basis)

    return {
        "mu": round(mu, 1),
        "sigma": round(sigma, 1),
        "mu_dose": round(mu_dose, 1),
        "sigma_dose": round(sigma_dose, 1),
        "dry_hours_train": int(len(dry)),
        "channel_mu_dry": ch_mu,
        "channel_sigma_dry": ch_sd,
        "washout_hours_fit_fold": n_w,
        "radon_washout_basis": [round(float(x), 6) for x in basis],
        "k_dose_channel": [round(float(x), 6) for x in k_vec],
    }


def _generate_extended_station_catalog(
    merged_by_station: dict[str, pd.DataFrame],
    base_catalog: pd.DataFrame,
    all_baselines: dict[str, Any],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Appends synthetic plumes for each of the 6 extended stations by re-using the
    exact plume parameter sets of one pilot station's catalog (scenario, passage /
    retention hours, shape, rise parameter, peak CPM, nuclide fractions, Family A/B
    spectral shares, environment class). Extended station k reuses the rows of pilot
    station k mod 5, so the parameter distribution is identical by construction.

    Each plume is re-placed at the extended station within the same split and, where
    possible, the same calendar year as the template row, under the pilot placement
    constraints (synthetic_injection.py): valid complete-channel hours, 48 h buffer
    between plumes, strictly-dry plumes have no precipitation in the window or the
    preceding 24 h, rain plumes have at least one hour >= 1 mm during passage.

    Known differences from the pilot catalog (documented in
    docs/phase6_generalization_summary.md): no filter-reset steps are modelled for
    extended stations, and `hours_since_last_detected_drop` is not computed (NaN).
    """
    rng = np.random.default_rng(seed)
    pilot_ids = [s["id"] for s in PILOT_STATIONS]
    ext_rows = []
    next_id = len(base_catalog) + 1
    n_unplaced = 0

    for k, st in enumerate(EXTENDED_STATIONS):
        st_id = st["id"]
        template_st = pilot_ids[k % len(pilot_ids)]
        df_m = merged_by_station[st_id].reset_index(drop=True)
        dts = pd.to_datetime(df_m["utc_hour"])
        years = dts.dt.year.to_numpy()
        valid = (df_m["has_radnet_obs"].fillna(False).astype(bool) & df_m["rad_complete_channels"].fillna(False).astype(bool)).to_numpy()
        precip = df_m["precip_1h_mm"].fillna(0.0).to_numpy(dtype=float)
        precip_24 = pd.Series(precip).rolling(24, min_periods=1).sum().to_numpy()
        sigma_168h = df_m["gross_cpm"].rolling(168, min_periods=24).std().bfill().fillna(250.0).to_numpy()
        n = len(df_m)
        occupied = np.zeros(n, dtype=bool)
        st_b = all_baselines[st_id]
        k_dose_st = round(float(st_b["mu_dose"]) / max(float(st_b["mu"]), 100.0) * 0.95, 5)
        test_start = int(np.flatnonzero(dts >= "2023-01-01")[0])

        tmpl = base_catalog[base_catalog["station_id"] == template_st].sort_values("start_utc")
        for _, r in tmpl.iterrows():
            split = str(r["split"])
            dur = int(r["duration_hours"])
            t_pass = int(r["t_passage_hours"])
            env = str(r["environment"])
            is_dry = "strictly_dry" in env
            lo, hi = (168, test_start - 1) if split == "train" else (test_start, n - 1)
            tmpl_year = int(pd.to_datetime(r["start_utc"]).year)

            def _ok(s: int) -> bool:
                e = s + dur - 1
                if s < lo or e > hi:
                    return False
                if occupied[max(0, s - 48): min(n, e + 49)].any():
                    return False
                if is_dry:
                    return bool(valid[s:e + 1].all() and precip[s:e + 1].max() <= 0.0 and precip_24[s] <= 0.0)
                max_miss = 1 if split == "train" else 2
                if int((~valid[s:e + 1]).sum()) > max_miss:
                    return False
                return bool((precip[s:s + t_pass] >= 1.0).any())

            chosen = None
            for year_pref in (True, False):
                cand = np.arange(lo, hi - dur + 2)
                if year_pref:
                    cand = cand[years[cand] == tmpl_year]
                cand = cand[valid[cand]]
                for s in rng.permutation(cand)[:4000]:
                    if _ok(int(s)):
                        chosen = int(s)
                        break
                if chosen is not None:
                    break
            if chosen is None:
                n_unplaced += 1
                continue

            e = chosen + dur - 1
            occupied[chosen:e + 1] = True
            bg_sig = max(10.0, float(sigma_168h[chosen]))
            snr_val = round(float(r["peak_cpm"]) / bg_sig, 3)
            row = r.to_dict()
            row.update({
                "injection_id": f"INJ_{next_id:04d}",
                "station_id": st_id,
                "station_name": st["name"],
                "start_index": chosen,
                "start_utc": dts.iloc[chosen].strftime("%Y-%m-%d %H:%M:%S"),
                "end_utc": dts.iloc[e].strftime("%Y-%m-%d %H:%M:%S"),
                "k_dose": k_dose_st,
                "washout_overlap_hours": int((precip[chosen:e + 1] >= 1.0).sum()),
                "rise_washout_overlap_hours": int((precip[chosen:chosen + t_pass] >= 1.0).sum()),
                "hours_since_last_detected_drop": np.nan,
                "truncated_by_filter_change": np.nan,
                "filter_reset_steps": "",
                "n_filter_resets_crossed": 0,
                "bg_sigma_168h_cpm": round(bg_sig, 2),
                "peak_snr_168h": snr_val,
                "snr_bin": classify_snr_bin(snr_val),
                "template_source_injection_id": str(r["injection_id"]),
            })
            ext_rows.append(row)
            next_id += 1

    if n_unplaced:
        print(f"WARNING: {n_unplaced} extended-station plumes could not be placed under pilot constraints")
    ext_df = pd.DataFrame(ext_rows)
    return pd.concat([base_catalog, ext_df], ignore_index=True)


def _build_11station_features(
    catalog_df: pd.DataFrame,
    merged_by_station: dict[str, pd.DataFrame],
    all_baselines: dict[str, Any],
    seed: int = 42,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Constructs train/val and test feature packages across all 11 stations.
    For stations with existing `labeled_{st_id}_{train|test}.csv.gz` on disk, loads
    those directly; for newly added extended stations, injects the realistic plumes
    and caches `labeled_{st_id}_{train|test}.csv.gz`.
    """
    STATION_DRY_BASELINES.update(all_baselines)
    rng = np.random.RandomState(seed + 777)
    train_data: dict[str, dict[str, Any]] = {}
    test_data: dict[str, dict[str, Any]] = {}

    raw_cols = ["gross_cpm", "dose_rate_nsvh", "inj_gross_cpm", "inj_dose_rate_nsvh"] + [
        f"cpm_r0{k}" for k in range(2, 10)
    ] + [f"inj_cpm_r0{k}" for k in range(2, 10)]

    for st in ALL_11_STATIONS:
        st_id = st["id"]
        tr_path = Path(f"data/processed/labeled_{st_id}_train.csv.gz")
        te_path = Path(f"data/processed/labeled_{st_id}_test.csv.gz")

        if tr_path.exists() and te_path.exists():
            df_tr = pd.read_csv(tr_path)
            df_te = pd.read_csv(te_path)
            df_tr["dt"] = pd.to_datetime(df_tr["dt"])
            df_te["dt"] = pd.to_datetime(df_te["dt"])
        else:
            df = merged_by_station[st_id]
            st_base = all_baselines[st_id]
            mu_dry = float(st_base["mu"])
            sigma_dry = float(st_base["sigma"])
            split_dfs = {}
            for split_name, s_dt_str, e_dt_str in [
                ("train", "2017-01-01", "2023-01-01"),
                ("test", "2023-01-01", "2026-01-01"),
            ]:
                sub = df[(df["dt"] >= s_dt_str) & (df["dt"] < e_dt_str)].copy().reset_index(drop=True)
                sub["inj_gross_cpm"] = sub["gross_cpm"].copy()
                sub["inj_dose_rate_nsvh"] = sub["dose_rate_nsvh"].copy()
                for ch in CHANNELS:
                    sub[f"inj_{ch}"] = sub[ch].copy()

                sub["label"] = "normal"
                sub["injection_active"] = False
                sub["injection_id"] = None
                sub["synthetic_excess_cpm"] = 0.0

                has_obs = sub["has_radnet_obs"].fillna(False).astype(bool)
                comp_ch = sub["rad_complete_channels"].fillna(False).astype(bool)
                unobs_mask = (~has_obs) | (~comp_ch)
                sub.loc[unobs_mask, "label"] = "unobserved"

                sub["precip_3h"] = sub["precip_1h_mm"].rolling(3, min_periods=1).sum()
                washout_condition = (
                    (sub["precip_3h"] > 0.0)
                    & (sub["gross_cpm"] > (mu_dry + 2.0 * sigma_dry))
                    & (~unobs_mask)
                )
                sub.loc[washout_condition, "label"] = "radon_washout"

                st_injections = catalog_df[(catalog_df["station_id"] == st_id) & (catalog_df["split"] == split_name)]
                for _, row in st_injections.iterrows():
                    inj_start = pd.to_datetime(row["start_utc"])
                    inj_end = pd.to_datetime(row["end_utc"])
                    indices = sub[(sub["dt"] >= inj_start) & (sub["dt"] <= inj_end)].index.to_numpy()
                    if len(indices) == 0:
                        continue
                    spec = np.array([float(row[f"share_r0{ch_i+2}"]) for ch_i in range(8)], dtype=float)
                    gross_prof, ch_prof = compute_multinuclide_injection_timeseries(
                        shape_family=str(row["shape_family"]),
                        t_passage=int(row["t_passage_hours"]),
                        t_retention=int(row["t_retention_hours"]),
                        peak_cpm=float(row["peak_cpm"]),
                        rise_param=float(row["rise_param"]),
                        scenario=str(row["nuclide_scenario"]),
                        spectrum_at_peak=spec,
                        f_cs137=float(row["fraction_cs137"]),
                        f_i131=float(row["fraction_i131"]),
                        f_co60=float(row["fraction_co60"]),
                        f_cs134=float(row["fraction_cs134"]),
                        filter_reset_steps=None,
                    )
                    gross_prof, ch_prof = apply_counting_noise(ch_prof, rng=rng, include_overdispersion=True)
                    dose_prof = compute_independent_injected_dose_rate(ch_prof, rng=rng)

                    for step_i, idx_sub in enumerate(indices):
                        s_val = float(gross_prof[step_i])
                        sub.loc[idx_sub, "synthetic_excess_cpm"] = s_val
                        sub.loc[idx_sub, "injection_active"] = True
                        sub.loc[idx_sub, "injection_id"] = row["injection_id"]
                        if not unobs_mask.loc[idx_sub]:
                            sub.loc[idx_sub, "inj_gross_cpm"] += s_val
                            sub.loc[idx_sub, "inj_dose_rate_nsvh"] += float(dose_prof[step_i])
                            for ch_idx, ch in enumerate(CHANNELS):
                                sub.loc[idx_sub, f"inj_{ch}"] += float(ch_prof[step_i, ch_idx])
                            sub.loc[idx_sub, "label"] = "fission_product"
                split_dfs[split_name] = sub

            df_tr = split_dfs["train"]
            df_te = split_dfs["test"]
            df_tr.to_csv(tr_path, index=False, compression="gzip")
            df_te.to_csv(te_path, index=False, compression="gzip")

        feats_tr_inj = compute_features_for_series(df_tr, st_id, use_injected=True)
        feats_tr_cln = compute_features_for_series(df_tr, st_id, use_injected=False)

        warmup = df_tr.iloc[-250:].copy()
        warmup["inj_gross_cpm"] = warmup["gross_cpm"]
        warmup["inj_dose_rate_nsvh"] = warmup["dose_rate_nsvh"]
        for ch_name in CHANNELS:
            warmup[f"inj_{ch_name}"] = warmup[ch_name]
        panel_te = pd.concat([warmup, df_te], ignore_index=True).sort_values("dt").reset_index(drop=True)
        m_te_panel = (panel_te["dt"] >= "2023-01-01").to_numpy(dtype=bool)
        feats_te_inj = compute_features_for_series(panel_te, st_id, use_injected=True)[m_te_panel].reset_index(drop=True)
        feats_te_cln = compute_features_for_series(panel_te, st_id, use_injected=False)[m_te_panel].reset_index(drop=True)

        for rc in raw_cols:
            if rc in df_tr.columns:
                feats_tr_inj[rc] = df_tr[rc].to_numpy(dtype=float)
                feats_tr_cln[rc] = df_tr[rc].to_numpy(dtype=float)
            if rc in df_te.columns:
                feats_te_inj[rc] = df_te[rc].to_numpy(dtype=float)
                feats_te_cln[rc] = df_te[rc].to_numpy(dtype=float)

        valid_tr = (feats_tr_inj["has_radnet_obs"].astype(bool) & feats_tr_inj["rad_complete_channels"].astype(bool))
        m_tr = (valid_tr & (df_tr["dt"] < "2021-01-01") & feats_tr_inj["label"].isin(LABEL_MAP.keys())).to_numpy(dtype=bool)
        m_val_calendar = ((df_tr["dt"] >= "2021-01-01") & (df_tr["dt"] < "2023-01-01")).to_numpy(dtype=bool)
        val_valid = valid_tr.to_numpy(dtype=bool)[m_val_calendar]

        st_tr_cat = catalog_df[(catalog_df["station_id"] == st_id) & (catalog_df["split"] == "train")]
        val_cat = st_tr_cat[(pd.to_datetime(st_tr_cat["start_utc"]) >= "2021-01-01") & (pd.to_datetime(st_tr_cat["start_utc"]) < "2023-01-01")]
        val_inj_slices = []
        for _, r in val_cat.iterrows():
            s_dt = pd.to_datetime(r["start_utc"])
            e_dt = pd.to_datetime(r["end_utc"])
            idx = df_tr[(df_tr["dt"] >= s_dt) & (df_tr["dt"] <= e_dt) & valid_tr].index.to_numpy()
            if len(idx) > 0:
                val_inj_slices.append((st_id, idx, str(r["injection_id"])))

        train_data[st_id] = {
            "df": df_tr,
            "feats_inj": feats_tr_inj,
            "feats_cln": feats_tr_cln,
            "m_tr": m_tr,
            "m_val_calendar": m_val_calendar,
            "val_valid": val_valid,
            "val_inj_slices": val_inj_slices,
        }

        valid_te = (feats_te_cln["has_radnet_obs"].astype(bool) & feats_te_cln["rad_complete_channels"].astype(bool))
        st_te_cat = catalog_df[(catalog_df["station_id"] == st_id) & (catalog_df["split"] == "test")]
        test_inj_slices = []
        for _, r in st_te_cat.iterrows():
            s_dt = pd.to_datetime(r["start_utc"])
            e_dt = pd.to_datetime(r["end_utc"])
            idx = df_te[(df_te["dt"] >= s_dt) & (df_te["dt"] <= e_dt) & valid_te].index.to_numpy()
            if len(idx) > 0:
                test_inj_slices.append((
                    st_id,
                    idx,
                    s_dt,
                    df_te.loc[idx, "dt"],
                    str(r["injection_id"]),
                    str(r["nuclide_scenario"]),
                    str(r.get("environment", "dry")),
                    float(r["peak_cpm"]),
                    int(r["duration_hours"]),
                    float(r.get("peak_snr_168h", 2.0)),
                ))

        test_data[st_id] = {
            "df": df_te,
            "feats_inj": feats_te_inj,
            "feats_cln": feats_te_cln,
            "valid": valid_te,
            "rain_1h": (df_te["precip_1h_mm"].fillna(0.0) > 0.0),
            "test_inj_slices": test_inj_slices,
        }

    return train_data, test_data


def _train_lgbm(X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    clf.fit(X, y)
    return clf


def _select_val_threshold(
    val_p_cln: dict[str, np.ndarray],
    val_p_inj: dict[str, np.ndarray],
    val_valid_masks: dict[str, np.ndarray],
    val_slices: list[tuple],
    target_det: float = 0.90,
) -> float:
    grid = np.unique(np.concatenate([np.linspace(0.50, 0.98, 49), np.linspace(0.981, 0.999, 37)]))
    val_st_yrs = max(1e-6, sum(int(vm.sum()) for vm in val_valid_masks.values()) / HOURS_PER_YEAR)

    v_det = np.zeros(len(grid))
    for st_id, idx, *_ in val_slices:
        pi_arr = val_p_inj[st_id]
        pc_arr = val_p_cln[st_id]
        offset = len(pi_arr) - len(pc_arr)
        pi = pi_arr[idx]
        pc = pc_arr[idx - offset]
        v_det += ((pi[:, None] >= grid[None, :]) & (pc[:, None] < grid[None, :])).any(axis=0).astype(float)
    v_det /= max(1, len(val_slices))

    v_fa = np.zeros(len(grid))
    for st_id, vm in val_valid_masks.items():
        pc = val_p_cln[st_id]
        on = (pc[:, None] >= grid[None, :]) & vm[:, None]
        pr = np.vstack([np.zeros((1, len(grid)), dtype=bool), on[:-1]])
        v_fa += (on & (~pr)).sum(axis=0)
    v_fa /= val_st_yrs

    el = np.flatnonzero(v_det >= target_det - 1e-9)
    if len(el) == 0:
        el = np.flatnonzero(v_det >= v_det.max() - 1e-9)
    best_i = int(el[np.argmin(v_fa[el])])
    return float(grid[best_i])


def run_generalization_pipeline() -> None:
    print("=== Loading 11 Stations (5 Pilot + 6 Extended) & Computing Baselines/Features ===")
    with open("data/processed/station_dry_baselines_train_only.json", "r") as f:
        all_baselines = json.load(f)

    merged_by_station: dict[str, pd.DataFrame] = {}
    raw_audit_rows = []
    yearly_drift_dfs = []

    for st in ALL_11_STATIONS:
        st_id = st["id"]
        df_m = pd.read_csv(f"data/processed/merged_{st_id}_2017_2025.csv.gz")
        df_m["dt"] = pd.to_datetime(df_m["utc_hour"])
        df_m = df_m.sort_values("dt").reset_index(drop=True)
        merged_by_station[st_id] = df_m

        if st_id not in all_baselines:
            all_baselines[st_id] = _compute_station_dry_baseline_from_window(df_m, "2017-01-01", "2021-01-01")

        q_sum, yr_df = audit_station_raw_data_quality(df_m, st_id)
        q_sum["station_name"] = st["name"]
        q_sum["climate_group"] = st["climate_group"]
        raw_audit_rows.append(q_sum)
        yearly_drift_dfs.append(yr_df)

    pd.DataFrame(raw_audit_rows).to_csv("data/processed/tampa_raw_data_quality_audit.csv", index=False)
    ext_baselines = {s["id"]: all_baselines[s["id"]] for s in EXTENDED_STATIONS}
    with open("data/processed/extended_station_dry_baselines_2017_2020.json", "w") as f_ext:
        json.dump(ext_baselines, f_ext, indent=2)
    df_yearly_drift = pd.concat(yearly_drift_dfs, ignore_index=True)
    df_yearly_drift.to_csv("data/processed/all_stations_yearly_baseline_drift.csv", index=False)
    print("Wrote data/processed/tampa_raw_data_quality_audit.csv and all_stations_yearly_baseline_drift.csv")

    base_catalog = pd.read_csv("data/processed/synthetic_injection_catalog.csv")
    ext_cat_path = Path("data/processed/extended_synthetic_injection_catalog.csv")
    if ext_cat_path.exists():
        full_catalog = pd.read_csv(ext_cat_path)
    else:
        full_catalog = _generate_extended_station_catalog(merged_by_station, base_catalog, all_baselines, seed=42)
        full_catalog.to_csv(ext_cat_path, index=False)
    print(f"Full 11-station catalog ready ({len(full_catalog)} plumes)")

    tr_all, te_all = _build_11station_features(
        full_catalog,
        merged_by_station,
        all_baselines,
        seed=42,
    )

    # Precompute clean feature stats per station on 2017-2020 fit fold + cache 168h local-normalized feature frames
    station_fit_clean_stats: dict[str, dict[str, tuple[float, float]]] = {}
    tr_168h_inj: dict[str, pd.DataFrame] = {}
    tr_168h_cln: dict[str, pd.DataFrame] = {}
    te_168h_inj: dict[str, pd.DataFrame] = {}
    te_168h_cln: dict[str, pd.DataFrame] = {}
    for st in ALL_11_STATIONS:
        st_id = st["id"]
        f_cln = tr_all[st_id]["feats_cln"]
        m_fit = tr_all[st_id]["m_tr"] & f_cln["gross_cpm"].notna().to_numpy()
        station_fit_clean_stats[st_id] = compute_clean_feature_stats(f_cln, m_fit, TIER_4_NUCLEAR_FEATURES)
        tr_168h_inj[st_id] = apply_normalization_variant(tr_all[st_id]["feats_inj"], "local_168h_baseline")
        tr_168h_cln[st_id] = apply_normalization_variant(tr_all[st_id]["feats_cln"], "local_168h_baseline")
        te_168h_inj[st_id] = apply_normalization_variant(te_all[st_id]["feats_inj"], "local_168h_baseline")
        te_168h_cln[st_id] = apply_normalization_variant(te_all[st_id]["feats_cln"], "local_168h_baseline")

    # -------------------------------------------------------------------------
    # Part A: Diagnose Held-Out Tampa (Tier 3 and Tier 4) + Gain-Drift Hypothesis
    # -------------------------------------------------------------------------
    print("=== Generalization Part A: Diagnosing Held-Out Tampa False Alarms & SHAP Contributions ===")
    pilot_ids = [s["id"] for s in PILOT_STATIONS]
    non_tampa_ids = [s for s in pilot_ids if s != "fl_tampa"]

    # Pool dry share stats across the 4 non-Tampa pilot training stations
    pool_r02 = np.concatenate([
        tr_all[s]["feats_cln"].loc[tr_all[s]["m_tr"], "share_r02"].dropna().to_numpy(dtype=float)
        for s in non_tampa_ids
    ])
    pool_r05 = np.concatenate([
        tr_all[s]["feats_cln"].loc[tr_all[s]["m_tr"], "share_r05"].dropna().to_numpy(dtype=float)
        for s in non_tampa_ids
    ])
    train_share_stats = {
        "share_r02": (float(np.mean(pool_r02)), float(np.std(pool_r02))),
        "share_r05": (float(np.mean(pool_r05)), float(np.std(pool_r05))),
    }

    x_tr_no_tampa = pd.concat([tr_all[s]["feats_inj"].loc[tr_all[s]["m_tr"], TIER_4_NUCLEAR_FEATURES] for s in non_tampa_ids], ignore_index=True)
    y_tr_no_tampa = np.concatenate([tr_all[s]["feats_inj"].loc[tr_all[s]["m_tr"], "label"].map(LABEL_MAP).to_numpy(dtype=int) for s in non_tampa_ids])

    val_slices_no_tampa = [sl for s in non_tampa_ids for sl in tr_all[s]["val_inj_slices"]]
    val_vm_no_tampa = {s: tr_all[s]["val_valid"] for s in non_tampa_ids}

    tampa_ep_dfs = []
    tampa_models_and_scores = {}

    for t_name, f_cols in [
        ("Tier 3 (Weather Fusion)", TIER_3_FEATURES),
        ("Tier 4 (Physics-Informed Nuclear)", TIER_4_NUCLEAR_FEATURES),
    ]:
        clf = _train_lgbm(x_tr_no_tampa[f_cols], y_tr_no_tampa, seed=42)
        vp_cln = {s: clf.predict_proba(tr_all[s]["feats_cln"][f_cols])[tr_all[s]["m_val_calendar"], 2] for s in non_tampa_ids}
        vp_inj = {s: clf.predict_proba(tr_all[s]["feats_inj"][f_cols])[:, 2] for s in non_tampa_ids}
        tau_loso = _select_val_threshold(vp_cln, vp_inj, val_vm_no_tampa, val_slices_no_tampa, target_det=0.90)

        p_tampa_cln = clf.predict_proba(te_all["fl_tampa"]["feats_cln"][f_cols])[:, 2]
        p_tampa_inj = clf.predict_proba(te_all["fl_tampa"]["feats_inj"][f_cols])[:, 2]
        tampa_models_and_scores[t_name] = (clf, f_cols, tau_loso, p_tampa_cln, p_tampa_inj)

        ep_df = extract_loso_false_alarm_episodes_with_shap(
            te_all["fl_tampa"]["df"],
            te_all["fl_tampa"]["feats_cln"],
            te_all["fl_tampa"]["valid"].to_numpy(dtype=bool),
            p_tampa_cln,
            threshold_tau=tau_loso,
            clf=clf,
            feature_cols=f_cols,
            tier_name=t_name,
            station_id="fl_tampa",
            train_share_stats=train_share_stats,
        )
        tampa_ep_dfs.append(ep_df)

    df_tampa_eps = pd.concat(tampa_ep_dfs, ignore_index=True)
    df_tampa_eps.to_csv("data/processed/loso_tampa_false_alarms.csv", index=False)

    cat_summary_rows = []
    for t_name, grp in df_tampa_eps.groupby("tier_name", sort=False):
        n_tot = len(grp)
        for cat, c_grp in grp.groupby("episode_category"):
            cat_summary_rows.append({
                "tier_name": t_name,
                "episode_category": cat,
                "episode_count": int(len(c_grp)),
                "share_of_episodes_pct": round(100.0 * len(c_grp) / max(1, n_tot), 2),
                "median_duration_hours": round(float(c_grp["duration_hours"].median()), 1),
                "mean_peak_score": round(float(c_grp["peak_score"].mean()), 4),
                "modal_top1_feature": str(c_grp["top_contrib_feature_1"].mode().iloc[0]),
            })
    pd.DataFrame(cat_summary_rows).to_csv("data/processed/loso_tampa_episode_category_summary.csv", index=False)
    print(f"Wrote data/processed/loso_tampa_false_alarms.csv ({len(df_tampa_eps)} episodes)")

    # Part A.3: Channel-share distribution comparison + Gain-drift / spectral-shift hypothesis test
    ch_comp_rows = []
    for st in ALL_11_STATIONS:
        st_id = st["id"]
        f_c = te_all[st_id]["feats_cln"]
        vm = te_all[st_id]["valid"].to_numpy(dtype=bool)
        sub = f_c.loc[vm]
        row = {"station_id": st_id, "group": st["climate_group"], "condition": "test_clean_2023_2025"}
        for col in SPECTRAL_SHARE_COLS + ["ratio_r05_r03", "ratio_r05_r07", "ratio_r02_r03"]:
            arr = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{col}_mean"] = round(float(np.mean(arr)), 4)
            row[f"{col}_sd"] = round(float(np.std(arr)), 4)
        ch_comp_rows.append(row)

    # Also add injected plume hours on the 4 non-Tampa pilot stations for comparison
    inj_rows_4st = []
    for s in non_tampa_ids:
        f_i = te_all[s]["feats_inj"]
        for sl in te_all[s]["test_inj_slices"]:
            inj_rows_4st.append(f_i.iloc[sl[1]])
    df_inj_4st = pd.concat(inj_rows_4st, ignore_index=True)
    inj_row = {"station_id": "4_Pilot_Train_Stations", "group": "Injected_Plumes", "condition": "test_injected_plume_hours"}
    for col in SPECTRAL_SHARE_COLS + ["ratio_r05_r03", "ratio_r05_r07", "ratio_r02_r03"]:
        arr = pd.to_numeric(df_inj_4st[col], errors="coerce").dropna().to_numpy(dtype=float)
        inj_row[f"{col}_mean"] = round(float(np.mean(arr)), 4)
        inj_row[f"{col}_sd"] = round(float(np.std(arr)), 4)
    ch_comp_rows.append(inj_row)
    pd.DataFrame(ch_comp_rows).to_csv("data/processed/tampa_channel_share_comparison.csv", index=False)

    # Direct counterfactual test of spectral-share / gain shift on Held-Out Tampa
    tampa_test_slices = [{"indices": sl[1]} for sl in te_all["fl_tampa"]["test_inj_slices"]]
    tampa_vm = te_all["fl_tampa"]["valid"].to_numpy(dtype=bool)
    pool_clean_stats_4st = compute_clean_feature_stats(
        pd.concat([tr_all[s]["feats_cln"].loc[tr_all[s]["m_tr"]] for s in non_tampa_ids], ignore_index=True),
        np.ones(sum(int(tr_all[s]["m_tr"].sum()) for s in non_tampa_ids), dtype=bool),
        TIER_4_NUCLEAR_FEATURES,
    )

    hyp_rows = []
    for t_name in ("Tier 3 (Weather Fusion)", "Tier 4 (Physics-Informed Nuclear)"):
        clf, f_cols, tau_loso, p_cln_raw, p_inj_raw = tampa_models_and_scores[t_name]
        m_raw = evaluate_station_test_metrics_with_exact_cis(p_cln_raw, p_inj_raw, tampa_vm, tampa_test_slices, tau_loso)
        hyp_rows.append({"tier_name": t_name, "intervention": "1_Unmodified_HeldOut_Tampa", **m_raw})

        # Counterfactual 2: Align ONLY Tampa's static spectral shares/ratios to 4-station dry mean/SD using Tampa's 2017-2020 dry stats
        f_cln_sh = te_all["fl_tampa"]["feats_cln"].copy()
        f_inj_sh = te_all["fl_tampa"]["feats_inj"].copy()
        for col in SPECTRAL_SHARE_COLS + ["share_high_energy", "ratio_r05_r07", "ratio_r03_r07", "ratio_r05_r03", "ratio_r02_r03"] + [f"share_r0{k}_mean_6h" for k in range(2, 10)]:
            if col in station_fit_clean_stats["fl_tampa"] and col in pool_clean_stats_4st:
                st_mu, st_sd = station_fit_clean_stats["fl_tampa"][col]
                tp_mu, tp_sd = pool_clean_stats_4st[col]
                f_cln_sh[col] = (f_cln_sh[col] - st_mu) / max(st_sd, 1e-5) * max(tp_sd, 1e-5) + tp_mu
                f_inj_sh[col] = (f_inj_sh[col] - st_mu) / max(st_sd, 1e-5) * max(tp_sd, 1e-5) + tp_mu
        m_sh = evaluate_station_test_metrics_with_exact_cis(
            clf.predict_proba(f_cln_sh[f_cols])[:, 2],
            clf.predict_proba(f_inj_sh[f_cols])[:, 2],
            tampa_vm,
            tampa_test_slices,
            tau_loso,
        )
        hyp_rows.append({"tier_name": t_name, "intervention": "2_Align_Tampa_2017_2020_Spectral_Shares_Only", **m_sh})

        # Counterfactual 3: Align using Tampa's 2022 pre-test spectral shares (accounting for 2017->2022 Tampa gain drift)
        f_2022 = tr_all["fl_tampa"]["feats_cln"].loc[tr_all["fl_tampa"]["m_val_calendar"]]
        stats_2022 = compute_clean_feature_stats(f_2022, tr_all["fl_tampa"]["val_valid"], TIER_4_NUCLEAR_FEATURES)
        f_cln_22 = apply_normalization_variant(te_all["fl_tampa"]["feats_cln"], "station_standardized", stats_2022, pool_clean_stats_4st)
        f_inj_22 = apply_normalization_variant(te_all["fl_tampa"]["feats_inj"], "station_standardized", stats_2022, pool_clean_stats_4st)
        m_22 = evaluate_station_test_metrics_with_exact_cis(
            clf.predict_proba(f_cln_22[f_cols])[:, 2],
            clf.predict_proba(f_inj_22[f_cols])[:, 2],
            tampa_vm,
            tampa_test_slices,
            tau_loso,
        )
        hyp_rows.append({"tier_name": t_name, "intervention": "3_Align_Tampa_2022_PreTest_Clean_Stats", **m_22})

    pd.DataFrame(hyp_rows).to_csv("data/processed/tampa_gain_drift_hypothesis_test.csv", index=False)
    print("Wrote data/processed/tampa_gain_drift_hypothesis_test.csv")

    # -------------------------------------------------------------------------
    # Part B & Part C & Part E & Part F:
    #   - Part B: Local Calibration (N = 0, 3, 6, 12 months of held-out clean background)
    #   - Part C: Normalization Variants (global_baseline, local_168h_baseline, station_standardized)
    #   - Part E: 11-Station LOSO and 7-Climate LOCO
    #   - Part F: Per-plume and per-month outputs
    # -------------------------------------------------------------------------
    print("=== Generalization Parts B, C, E, F: Running Local Calibration, Normalization Variants, 11-Station LOSO, and LOCO ===")
    local_cal_rows = []
    norm_var_rows = []
    loso_11_rows = []
    high_fa_diag_dfs = []
    per_plume_rows = []
    per_month_rows = []

    # Helper to append per-plume and per-month records (Part F)
    def _record_plume_and_month(
        det_id: str,
        op_type: str,
        op_target: float,
        st_id: str,
        tau: float,
        pc: np.ndarray,
        pi: np.ndarray,
        vm: np.ndarray,
        dts_series: pd.Series,
        slices_full: list[tuple],
    ) -> None:
        for _, idx, start_dt, dts_sub, pid, scen, env, pk_cpm, dur_h, snr_g in slices_full:
            net = (pi[idx] >= tau) & (pc[idx] < tau)
            is_d = int(np.any(net))
            delay_h = float((dts_sub.iloc[int(np.flatnonzero(net)[0])] - start_dt).total_seconds() / 3600.0) if is_d else np.nan
            per_plume_rows.append({
                "detector_id": det_id,
                "split": "test",
                "operating_point_type": op_type,
                "operating_target": op_target,
                "plume_id": pid,
                "station_id": st_id,
                "nuclide_scenario": scen,
                "environment": env,
                "peak_cpm": pk_cpm,
                "duration_hours": dur_h,
                "gross_snr": snr_g,
                "detected": is_d,
                "detection_delay_hours": delay_h,
            })

        on = (pc >= tau) & vm
        pr = np.concatenate([[False], on[:-1]])
        ep_st = on & (~pr)
        ym_arr = pd.to_datetime(dts_series).dt.strftime("%Y-%m").to_numpy()
        for ym in pd.unique(ym_arr):
            m = ym_arr == ym
            obs_h = int(vm[m].sum())
            if obs_h > 0:
                per_month_rows.append({
                    "detector_id": det_id,
                    "split": "test",
                    "operating_point_type": op_type,
                    "operating_target": op_target,
                    "station_id": st_id,
                    "year_month": str(ym),
                    "false_alarm_episodes": int(ep_st[m].sum()),
                    "exposure_station_years": round(obs_h / HOURS_PER_YEAR, 6),
                })

    all_11_ids = [s["id"] for s in ALL_11_STATIONS]

    # 1. Normalization Variants under 5-Station Pilot LOSO (Part C)
    for variant in ("global_baseline", "local_168h_baseline", "station_standardized"):
        for t_name, f_cols in [
            ("Tier 3 (Weather Fusion)", TIER_3_FEATURES),
            ("Tier 4 (Physics-Informed Nuclear)", TIER_4_NUCLEAR_FEATURES),
        ]:
            for held_st in PILOT_STATIONS:
                held_id = held_st["id"]
                train_ids = [s for s in pilot_ids if s != held_id]

                # Compute pool stats on the 4 training stations' 2017-2020 clean fit hours
                pool_df = pd.concat([tr_all[s]["feats_cln"].loc[tr_all[s]["m_tr"]] for s in train_ids], ignore_index=True)
                pool_stats = compute_clean_feature_stats(pool_df, np.ones(len(pool_df), dtype=bool), TIER_4_NUCLEAR_FEATURES)

                x_tr_list, y_tr_list = [], []
                val_cln_v, val_inj_v, val_vm_v = {}, {}, {}
                for s in train_ids:
                    f_inj_s = apply_normalization_variant(tr_all[s]["feats_inj"], variant, station_fit_clean_stats[s], pool_stats)
                    f_cln_s = apply_normalization_variant(tr_all[s]["feats_cln"], variant, station_fit_clean_stats[s], pool_stats)
                    m_tr_s = tr_all[s]["m_tr"]
                    x_tr_list.append(f_inj_s.loc[m_tr_s, f_cols])
                    y_tr_list.append(f_inj_s.loc[m_tr_s, "label"].map(LABEL_MAP).to_numpy(dtype=int))
                    val_cln_v[s] = f_cln_s.loc[tr_all[s]["m_val_calendar"], f_cols]
                    val_inj_v[s] = f_inj_s[f_cols]
                    val_vm_v[s] = tr_all[s]["val_valid"]

                clf_v = _train_lgbm(pd.concat(x_tr_list, ignore_index=True), np.concatenate(y_tr_list), seed=42)
                vp_cln = {s: clf_v.predict_proba(val_cln_v[s])[:, 2] for s in train_ids}
                vp_inj = {s: clf_v.predict_proba(val_inj_v[s])[:, 2] for s in train_ids}
                v_slices = [sl for s in train_ids for sl in tr_all[s]["val_inj_slices"]]
                tau_v = _select_val_threshold(vp_cln, vp_inj, val_vm_v, v_slices, target_det=0.90)

                # For station_standardized on held-out station, use held-out station's pre-test clean stats (first 12m of 2017 or pre-test 2022)
                f_te_cln = apply_normalization_variant(te_all[held_id]["feats_cln"], variant, station_fit_clean_stats[held_id], pool_stats)
                f_te_inj = apply_normalization_variant(te_all[held_id]["feats_inj"], variant, station_fit_clean_stats[held_id], pool_stats)
                p_te_cln = clf_v.predict_proba(f_te_cln[f_cols])[:, 2]
                p_te_inj = clf_v.predict_proba(f_te_inj[f_cols])[:, 2]

                t_slices = [{"indices": sl[1]} for sl in te_all[held_id]["test_inj_slices"]]
                vm_h = te_all[held_id]["valid"].to_numpy(dtype=bool)
                metrics_v = evaluate_station_test_metrics_with_exact_cis(p_te_cln, p_te_inj, vm_h, t_slices, tau_v)
                norm_var_rows.append({
                    "normalization_variant": variant,
                    "tier_name": t_name,
                    "held_out_station": held_id,
                    "station_name": held_st["name"],
                    **metrics_v,
                })

    pd.DataFrame(norm_var_rows).to_csv("data/processed/loso_normalization_variants.csv", index=False)
    print("Wrote data/processed/loso_normalization_variants.csv")

    # 2. Part B: Direct Local-Calibration Test (N = 0, 3, 6, 12 months of held-out station's own pre-test clean background)
    for held_st in PILOT_STATIONS:
        held_id = held_st["id"]
        train_ids = [s for s in pilot_ids if s != held_id]
        pool_df = pd.concat([tr_all[s]["feats_cln"].loc[tr_all[s]["m_tr"]] for s in train_ids], ignore_index=True)
        pool_stats = compute_clean_feature_stats(pool_df, np.ones(len(pool_df), dtype=bool), TIER_4_NUCLEAR_FEATURES)

        x_tr_b = pd.concat([tr_all[s]["feats_inj"].loc[tr_all[s]["m_tr"], TIER_4_NUCLEAR_FEATURES] for s in train_ids], ignore_index=True)
        y_tr_b = np.concatenate([tr_all[s]["feats_inj"].loc[tr_all[s]["m_tr"], "label"].map(LABEL_MAP).to_numpy(dtype=int) for s in train_ids])
        v_slices = [sl for s in train_ids for sl in tr_all[s]["val_inj_slices"]]
        val_vm_b = {s: tr_all[s]["val_valid"] for s in train_ids}

        for t_name, f_cols in [
            ("Tier 3 (Weather Fusion)", TIER_3_FEATURES),
            ("Tier 4 (Physics-Informed Nuclear)", TIER_4_NUCLEAR_FEATURES),
        ]:
            clf_b = _train_lgbm(x_tr_b[f_cols], y_tr_b, seed=42)
            vp_cln = {s: clf_b.predict_proba(tr_all[s]["feats_cln"][f_cols])[tr_all[s]["m_val_calendar"], 2] for s in train_ids}
            vp_inj = {s: clf_b.predict_proba(tr_all[s]["feats_inj"][f_cols])[:, 2] for s in train_ids}
            tau_zero_shot = _select_val_threshold(vp_cln, vp_inj, val_vm_b, v_slices, target_det=0.90)

            t_slices = [{"indices": sl[1]} for sl in te_all[held_id]["test_inj_slices"]]
            vm_te = te_all[held_id]["valid"].to_numpy(dtype=bool)

            # N = 0 (Uncalibrated Zero-Shot)
            p_zs_cln = clf_b.predict_proba(te_all[held_id]["feats_cln"][f_cols])[:, 2]
            p_zs_inj = clf_b.predict_proba(te_all[held_id]["feats_inj"][f_cols])[:, 2]
            m_zs = evaluate_station_test_metrics_with_exact_cis(p_zs_cln, p_zs_inj, vm_te, t_slices, tau_zero_shot)
            local_cal_rows.append({
                "tier_name": t_name,
                "held_out_station": held_id,
                "station_name": held_st["name"],
                "calibration_window": "0m_ZeroShot_Uncalibrated",
                "calibration_months_N": 0,
                "fa_budget_rule": "ZeroShot_Val90pct",
                **m_zs,
            })

            # N in {3, 6, 12} months of held-out station's own pre-test clean background:
            # Test both: (i) Initial 2017 N months [2017-01-01, 2017-01-01 + N months) and
            #            (ii) Trailing pre-test N months ending 2022-12-31 [2023-01-01 - N months, 2023-01-01)
            df_tr_cln = tr_all[held_id]["feats_cln"]
            dts_tr = pd.to_datetime(tr_all[held_id]["df"]["dt"])
            valid_tr = df_tr_cln["gross_cpm"].notna().to_numpy(dtype=bool)

            for cal_origin, win_fn in [
                ("Initial_2017_Clean", lambda n_m: (dts_tr >= "2017-01-01") & (dts_tr < pd.Timestamp("2017-01-01") + pd.DateOffset(months=n_m))),
                ("Trailing_PreTest_2022_Clean", lambda n_m: (dts_tr >= pd.Timestamp("2023-01-01") - pd.DateOffset(months=n_m)) & (dts_tr < "2023-01-01")),
            ]:
                for n_months in (3, 6, 12):
                    cal_mask = (win_fn(n_months) & valid_tr).to_numpy(dtype=bool)
                    loc_stats = compute_clean_feature_stats(df_tr_cln, cal_mask, TIER_4_NUCLEAR_FEATURES)

                    # Align held-out station features using only its N-month local clean background stats
                    f_cal_aligned = apply_normalization_variant(df_tr_cln.loc[cal_mask], "station_standardized", loc_stats, pool_stats)
                    p_cal_cln = clf_b.predict_proba(f_cal_aligned[f_cols])[:, 2]

                    f_te_cln_al = apply_normalization_variant(te_all[held_id]["feats_cln"], "station_standardized", loc_stats, pool_stats)
                    f_te_inj_al = apply_normalization_variant(te_all[held_id]["feats_inj"], "station_standardized", loc_stats, pool_stats)
                    p_te_cln_al = clf_b.predict_proba(f_te_cln_al[f_cols])[:, 2]
                    p_te_inj_al = clf_b.predict_proba(f_te_inj_al[f_cols])[:, 2]

                    for b_lbl, b_val in [("Zero_Local_FA_Ceiling", 0.0), ("Local_FA_Budget_1_per_yr", 1.0), ("Local_FA_Budget_5_per_yr", 5.0)]:
                        tau_loc = select_local_calibration_threshold(
                            p_cal_cln,
                            np.ones(len(p_cal_cln), dtype=bool),
                            fa_budget_per_yr=b_val,
                            fallback_tau=tau_zero_shot,
                        )
                        m_loc = evaluate_station_test_metrics_with_exact_cis(p_te_cln_al, p_te_inj_al, vm_te, t_slices, tau_loc)
                        local_cal_rows.append({
                            "tier_name": t_name,
                            "held_out_station": held_id,
                            "station_name": held_st["name"],
                            "calibration_window": f"{cal_origin}_{n_months}m",
                            "calibration_months_N": n_months,
                            "fa_budget_rule": b_lbl,
                            **m_loc,
                        })

    pd.DataFrame(local_cal_rows).to_csv("data/processed/loso_local_calibration_results.csv", index=False)
    print("Wrote data/processed/loso_local_calibration_results.csv")

    # 3. Part E: Full 11-Station LOSO and 7-Climate LOCO (Zero-Shot Global vs Local-168h Normalized vs 12m Pre-Test Local Calibration)
    print("=== Generalization Part E: Running 11-Station LOSO & 7-Climate LOCO ===")
    for held_st in ALL_11_STATIONS:
        held_id = held_st["id"]
        train_ids = [s for s in all_11_ids if s != held_id]

        pool_df = pd.concat([tr_all[s]["feats_cln"].loc[tr_all[s]["m_tr"]] for s in train_ids], ignore_index=True)
        pool_stats = compute_clean_feature_stats(pool_df, np.ones(len(pool_df), dtype=bool), TIER_4_NUCLEAR_FEATURES)

        x_tr_11 = pd.concat([tr_all[s]["feats_inj"].loc[tr_all[s]["m_tr"], TIER_4_NUCLEAR_FEATURES] for s in train_ids], ignore_index=True)
        y_tr_11 = np.concatenate([tr_all[s]["feats_inj"].loc[tr_all[s]["m_tr"], "label"].map(LABEL_MAP).to_numpy(dtype=int) for s in train_ids])
        v_slices_11 = [sl for s in train_ids for sl in tr_all[s]["val_inj_slices"]]
        val_vm_11 = {s: tr_all[s]["val_valid"] for s in train_ids}

        t_slices_11 = [{"indices": sl[1]} for sl in te_all[held_id]["test_inj_slices"]]
        vm_te = te_all[held_id]["valid"].to_numpy(dtype=bool)
        dts_te = te_all[held_id]["df"]["dt"]

        # Evaluate Non-ML B4c on 11-station LOSO
        vp_cln_b4c = {s: score_b4c_radon_projection_combined(tr_all[s]["feats_cln"])[tr_all[s]["m_val_calendar"]] for s in train_ids}
        vp_inj_b4c = {s: score_b4c_radon_projection_combined(tr_all[s]["feats_inj"]) for s in train_ids}
        # Adaptive grid for B4c
        b4c_grid = np.linspace(0.5, 3.5, 120)
        val_st_yrs_11 = max(1e-6, sum(int(vm.sum()) for vm in val_vm_11.values()) / HOURS_PER_YEAR)
        vd_b4c = np.zeros(len(b4c_grid))
        for st_s, idx, *_ in v_slices_11:
            pi_arr = vp_inj_b4c[st_s]
            pc_arr = vp_cln_b4c[st_s]
            offset = len(pi_arr) - len(pc_arr)
            pi = pi_arr[idx]
            pc = pc_arr[idx - offset]
            vd_b4c += ((pi[:, None] >= b4c_grid[None, :]) & (pc[:, None] < b4c_grid[None, :])).any(axis=0).astype(float)
        vd_b4c /= max(1, len(v_slices_11))
        vf_b4c = np.zeros(len(b4c_grid))
        for st_s, vm_s in val_vm_11.items():
            pc = vp_cln_b4c[st_s]
            on = (pc[:, None] >= b4c_grid[None, :]) & vm_s[:, None]
            pr = np.vstack([np.zeros((1, len(b4c_grid)), dtype=bool), on[:-1]])
            vf_b4c += (on & (~pr)).sum(axis=0)
        vf_b4c /= val_st_yrs_11
        el_b4c = np.flatnonzero(vd_b4c >= 0.90)
        tau_b4c = float(b4c_grid[el_b4c[np.argmin(vf_b4c[el_b4c])]]) if len(el_b4c) > 0 else 1.2

        p_b4c_cln = score_b4c_radon_projection_combined(te_all[held_id]["feats_cln"])
        p_b4c_inj = score_b4c_radon_projection_combined(te_all[held_id]["feats_inj"])
        m_b4c = evaluate_station_test_metrics_with_exact_cis(p_b4c_cln, p_b4c_inj, vm_te, t_slices_11, tau_b4c)
        loso_11_rows.append({
            "detector_name": "B4c: Radon Projection + Peak Channel Z (Non-ML)",
            "transfer_mode": "ZeroShot_Local168h_Intrinsic",
            "held_out_station": held_id,
            "station_name": held_st["name"],
            "climate_group": held_st["climate_group"],
            **m_b4c,
        })
        _record_plume_and_month(
            "LOSO11_B4c_ZeroShot", "loso_val_target_pct", 90.0, held_id, tau_b4c,
            p_b4c_cln, p_b4c_inj, vm_te, dts_te, te_all[held_id]["test_inj_slices"]
        )

        for t_name, det_short, f_cols in [
            ("Tier 3 (Weather Fusion)", "Tier3", TIER_3_FEATURES),
            ("Tier 4 (Physics-Informed Nuclear)", "Tier4", TIER_4_NUCLEAR_FEATURES),
        ]:
            clf_11 = _train_lgbm(x_tr_11[f_cols], y_tr_11, seed=42)
            vp_cln = {s: clf_11.predict_proba(tr_all[s]["feats_cln"][f_cols])[tr_all[s]["m_val_calendar"], 2] for s in train_ids}
            vp_inj = {s: clf_11.predict_proba(tr_all[s]["feats_inj"][f_cols])[:, 2] for s in train_ids}
            tau_11 = _select_val_threshold(vp_cln, vp_inj, val_vm_11, v_slices_11, target_det=0.90)

            p_cln_h = clf_11.predict_proba(te_all[held_id]["feats_cln"][f_cols])[:, 2]
            p_inj_h = clf_11.predict_proba(te_all[held_id]["feats_inj"][f_cols])[:, 2]
            m_zs11 = evaluate_station_test_metrics_with_exact_cis(p_cln_h, p_inj_h, vm_te, t_slices_11, tau_11)
            loso_11_rows.append({
                "detector_name": t_name,
                "transfer_mode": "ZeroShot_Uncalibrated",
                "held_out_station": held_id,
                "station_name": held_st["name"],
                "climate_group": held_st["climate_group"],
                **m_zs11,
            })
            _record_plume_and_month(
                f"LOSO11_{det_short}_ZeroShot", "loso_val_target_pct", 90.0, held_id, tau_11,
                p_cln_h, p_inj_h, vm_te, dts_te, te_all[held_id]["test_inj_slices"]
            )

            # Diagnose any held-out station with FA > 5.0/yr under Zero-Shot
            if m_zs11["fa_per_yr"] > 5.0:
                ep_h = extract_loso_false_alarm_episodes_with_shap(
                    te_all[held_id]["df"],
                    te_all[held_id]["feats_cln"],
                    vm_te,
                    p_cln_h,
                    threshold_tau=tau_11,
                    clf=clf_11,
                    feature_cols=f_cols,
                    tier_name=t_name,
                    station_id=held_id,
                    train_share_stats=train_share_stats,
                )
                if not ep_h.empty:
                    high_fa_diag_dfs.append(ep_h)

            # Also evaluate Zero-Shot Local-168h Normalized transfer on 11-station LOSO
            x_tr_168 = pd.concat([tr_168h_inj[s].loc[tr_all[s]["m_tr"], f_cols] for s in train_ids], ignore_index=True)
            clf_168 = _train_lgbm(x_tr_168, y_tr_11, seed=42)
            vp_cln_168 = {s: clf_168.predict_proba(tr_168h_cln[s][f_cols])[tr_all[s]["m_val_calendar"], 2] for s in train_ids}
            vp_inj_168 = {s: clf_168.predict_proba(tr_168h_inj[s][f_cols])[:, 2] for s in train_ids}
            tau_168 = _select_val_threshold(vp_cln_168, vp_inj_168, val_vm_11, v_slices_11, target_det=0.90)
            p_cln_168 = clf_168.predict_proba(te_168h_cln[held_id][f_cols])[:, 2]
            p_inj_168 = clf_168.predict_proba(te_168h_inj[held_id][f_cols])[:, 2]
            m_168_11 = evaluate_station_test_metrics_with_exact_cis(p_cln_168, p_inj_168, vm_te, t_slices_11, tau_168)
            loso_11_rows.append({
                "detector_name": t_name,
                "transfer_mode": "ZeroShot_Local168h_Normalized",
                "held_out_station": held_id,
                "station_name": held_st["name"],
                "climate_group": held_st["climate_group"],
                **m_168_11,
            })
            _record_plume_and_month(
                f"LOSO11_{det_short}_Local168h", "loso_val_target_pct", 90.0, held_id, tau_168,
                p_cln_168, p_inj_168, vm_te, dts_te, te_all[held_id]["test_inj_slices"]
            )

            # Also evaluate 12-month pre-test local calibration on 11-station LOSO
            f_2022_h = tr_all[held_id]["feats_cln"].loc[tr_all[held_id]["m_val_calendar"]]
            dts_val_h = pd.to_datetime(tr_all[held_id]["df"].loc[tr_all[held_id]["m_val_calendar"], "dt"])
            m_12m = ((dts_val_h >= "2022-01-01") & tr_all[held_id]["val_valid"]).to_numpy(dtype=bool)
            if m_12m.sum() < 100:
                m_12m = tr_all[held_id]["val_valid"]
            loc_stats_h = compute_clean_feature_stats(f_2022_h, m_12m, TIER_4_NUCLEAR_FEATURES)

            f_cal_al = apply_normalization_variant(f_2022_h.loc[m_12m], "station_standardized", loc_stats_h, pool_stats)
            p_cal_h = clf_11.predict_proba(f_cal_al[f_cols])[:, 2]
            tau_loc_h = select_local_calibration_threshold(p_cal_h, np.ones(len(p_cal_h), dtype=bool), fa_budget_per_yr=0.0, fallback_tau=tau_11)

            f_te_cln_al = apply_normalization_variant(te_all[held_id]["feats_cln"], "station_standardized", loc_stats_h, pool_stats)
            f_te_inj_al = apply_normalization_variant(te_all[held_id]["feats_inj"], "station_standardized", loc_stats_h, pool_stats)
            p_te_cln_al = clf_11.predict_proba(f_te_cln_al[f_cols])[:, 2]
            p_te_inj_al = clf_11.predict_proba(f_te_inj_al[f_cols])[:, 2]
            m_loc11 = evaluate_station_test_metrics_with_exact_cis(p_te_cln_al, p_te_inj_al, vm_te, t_slices_11, tau_loc_h)
            loso_11_rows.append({
                "detector_name": t_name,
                "transfer_mode": "LocalCalibrated_12m_Clean",
                "held_out_station": held_id,
                "station_name": held_st["name"],
                "climate_group": held_st["climate_group"],
                **m_loc11,
            })
            _record_plume_and_month(
                f"LOSO11_{det_short}_LocalCal12m", "loso_local_cal", 0.0, held_id, tau_loc_h,
                p_te_cln_al, p_te_inj_al, vm_te, dts_te, te_all[held_id]["test_inj_slices"]
            )

    pd.DataFrame(loso_11_rows).to_csv("data/processed/extended_loso_11station_summary.csv", index=False)
    print("Wrote data/processed/extended_loso_11station_summary.csv")

    if high_fa_diag_dfs:
        df_high_fa = pd.concat(high_fa_diag_dfs, ignore_index=True)
        hfa_rows = []
        for (t_name, st_id), grp in df_high_fa.groupby(["tier_name", "held_out_station"], sort=False):
            n_tot = len(grp)
            for cat, c_grp in grp.groupby("episode_category"):
                hfa_rows.append({
                    "tier_name": t_name,
                    "held_out_station": st_id,
                    "episode_category": cat,
                    "episode_count": int(len(c_grp)),
                    "share_pct": round(100.0 * len(c_grp) / max(1, n_tot), 2),
                    "modal_top1_feature": str(c_grp["top_contrib_feature_1"].mode().iloc[0]),
                })
        pd.DataFrame(hfa_rows).to_csv("data/processed/loso_high_fa_stations_diagnosis.csv", index=False)

    # Leave-One-Climate-Out (LOCO) across the 7 climate groups
    climate_groups = sorted(list({s["climate_group"] for s in ALL_11_STATIONS}))
    loco_rows = []
    for c_grp in climate_groups:
        held_ids = [s["id"] for s in ALL_11_STATIONS if s["climate_group"] == c_grp]
        train_ids = [s["id"] for s in ALL_11_STATIONS if s["climate_group"] != c_grp]

        pool_df = pd.concat([tr_all[s]["feats_cln"].loc[tr_all[s]["m_tr"]] for s in train_ids], ignore_index=True)
        pool_stats = compute_clean_feature_stats(pool_df, np.ones(len(pool_df), dtype=bool), TIER_4_NUCLEAR_FEATURES)
        x_tr_c = pd.concat([tr_all[s]["feats_inj"].loc[tr_all[s]["m_tr"], TIER_4_NUCLEAR_FEATURES] for s in train_ids], ignore_index=True)
        y_tr_c = np.concatenate([tr_all[s]["feats_inj"].loc[tr_all[s]["m_tr"], "label"].map(LABEL_MAP).to_numpy(dtype=int) for s in train_ids])
        v_slices_c = [sl for s in train_ids for sl in tr_all[s]["val_inj_slices"]]
        val_vm_c = {s: tr_all[s]["val_valid"] for s in train_ids}

        for t_name, f_cols in [
            ("Tier 3 (Weather Fusion)", TIER_3_FEATURES),
            ("Tier 4 (Physics-Informed Nuclear)", TIER_4_NUCLEAR_FEATURES),
        ]:
            clf_c = _train_lgbm(x_tr_c[f_cols], y_tr_c, seed=42)
            vp_cln = {s: clf_c.predict_proba(tr_all[s]["feats_cln"][f_cols])[tr_all[s]["m_val_calendar"], 2] for s in train_ids}
            vp_inj = {s: clf_c.predict_proba(tr_all[s]["feats_inj"][f_cols])[:, 2] for s in train_ids}
            tau_c = _select_val_threshold(vp_cln, vp_inj, val_vm_c, v_slices_c, target_det=0.90)

            x_tr_c_168 = pd.concat([tr_168h_inj[s].loc[tr_all[s]["m_tr"], f_cols] for s in train_ids], ignore_index=True)
            clf_c_168 = _train_lgbm(x_tr_c_168, y_tr_c, seed=42)
            vp_cln_c168 = {s: clf_c_168.predict_proba(tr_168h_cln[s][f_cols])[tr_all[s]["m_val_calendar"], 2] for s in train_ids}
            vp_inj_c168 = {s: clf_c_168.predict_proba(tr_168h_inj[s][f_cols])[:, 2] for s in train_ids}
            tau_c_168 = _select_val_threshold(vp_cln_c168, vp_inj_c168, val_vm_c, v_slices_c, target_det=0.90)

            for transfer_mode in ("ZeroShot_Uncalibrated", "ZeroShot_Local168h_Normalized", "LocalCalibrated_12m_Clean"):
                tot_obs = 0
                tot_ep = 0
                tot_plumes = 0
                tot_det = 0
                for h_id in held_ids:
                    vm_h = te_all[h_id]["valid"].to_numpy(dtype=bool)
                    t_sl = [{"indices": sl[1]} for sl in te_all[h_id]["test_inj_slices"]]
                    if transfer_mode == "ZeroShot_Uncalibrated":
                        pc = clf_c.predict_proba(te_all[h_id]["feats_cln"][f_cols])[:, 2]
                        pi = clf_c.predict_proba(te_all[h_id]["feats_inj"][f_cols])[:, 2]
                        tau_eval = tau_c
                    elif transfer_mode == "ZeroShot_Local168h_Normalized":
                        pc = clf_c_168.predict_proba(te_168h_cln[h_id][f_cols])[:, 2]
                        pi = clf_c_168.predict_proba(te_168h_inj[h_id][f_cols])[:, 2]
                        tau_eval = tau_c_168
                    else:
                        f_22 = tr_all[h_id]["feats_cln"].loc[tr_all[h_id]["m_val_calendar"]]
                        loc_st = compute_clean_feature_stats(f_22, tr_all[h_id]["val_valid"], TIER_4_NUCLEAR_FEATURES)
                        f_cal = apply_normalization_variant(f_22.loc[tr_all[h_id]["val_valid"]], "station_standardized", loc_st, pool_stats)
                        p_cal = clf_c.predict_proba(f_cal[f_cols])[:, 2]
                        tau_eval = select_local_calibration_threshold(p_cal, np.ones(len(p_cal), dtype=bool), fa_budget_per_yr=0.0, fallback_tau=tau_c)
                        pc = clf_c.predict_proba(apply_normalization_variant(te_all[h_id]["feats_cln"], "station_standardized", loc_st, pool_stats)[f_cols])[:, 2]
                        pi = clf_c.predict_proba(apply_normalization_variant(te_all[h_id]["feats_inj"], "station_standardized", loc_st, pool_stats)[f_cols])[:, 2]

                    m_h = evaluate_station_test_metrics_with_exact_cis(pc, pi, vm_h, t_sl, tau_eval)
                    tot_obs += m_h["observed_hours"]
                    tot_ep += m_h["false_alarm_episodes"]
                    tot_plumes += m_h["n_plumes"]
                    tot_det += m_h["n_detected"]

                exp_c = max(1e-6, tot_obs / HOURS_PER_YEAR)
                fa_c = tot_ep / exp_c
                fa_lo, fa_hi = exact_poisson_rate_ci(tot_ep, exp_c, alpha=0.05)
                det_c = 100.0 * tot_det / max(1, tot_plumes)
                det_lo, det_hi = clopper_pearson_ci(tot_det, max(1, tot_plumes), alpha=0.05)
                loco_rows.append({
                    "held_out_climate_group": c_grp,
                    "held_out_stations": ",".join(held_ids),
                    "n_stations": len(held_ids),
                    "tier_name": t_name,
                    "transfer_mode": transfer_mode,
                    "observed_hours": tot_obs,
                    "exposure_station_years": round(exp_c, 4),
                    "false_alarm_episodes": tot_ep,
                    "fa_per_yr": round(fa_c, 2),
                    "fa_exact_poisson_ci_low": round(fa_lo, 2),
                    "fa_exact_poisson_ci_high": round(fa_hi, 2),
                    "n_plumes": tot_plumes,
                    "n_detected": tot_det,
                    "detection_pct": round(det_c, 2),
                    "det_clopper_pearson_ci_low": round(det_lo, 2),
                    "det_clopper_pearson_ci_high": round(det_hi, 2),
                })

    pd.DataFrame(loco_rows).to_csv("data/processed/extended_loco_climate_summary.csv", index=False)
    print("Wrote data/processed/extended_loco_climate_summary.csv")

    pd.DataFrame(per_plume_rows).to_csv("data/processed/generalization_loso_per_plume.csv", index=False)
    pd.DataFrame(per_month_rows).to_csv("data/processed/generalization_loso_per_month.csv", index=False)
    print(f"Wrote data/processed/generalization_loso_per_plume.csv ({len(per_plume_rows)} rows) and generalization_loso_per_month.csv ({len(per_month_rows)} rows)")

    # -------------------------------------------------------------------------
    # Part D: Temporal Generalization (Train Once on 2017-2020; Report 2021, 2022, 2023, 2024, 2025)
    # -------------------------------------------------------------------------
    print("=== Generalization Part D: Evaluating Year-by-Year Temporal Generalization (2021-2025) ===")
    x_tr_5 = pd.concat([tr_all[s]["feats_inj"].loc[tr_all[s]["m_tr"], TIER_4_NUCLEAR_FEATURES] for s in pilot_ids], ignore_index=True)
    y_tr_5 = np.concatenate([tr_all[s]["feats_inj"].loc[tr_all[s]["m_tr"], "label"].map(LABEL_MAP).to_numpy(dtype=int) for s in pilot_ids])
    val_slices_5 = [sl for s in pilot_ids for sl in tr_all[s]["val_inj_slices"]]
    val_vm_5 = {s: tr_all[s]["val_valid"] for s in pilot_ids}

    temp_year_rows = []
    temp_st_year_rows = []

    detectors_temporal: list[tuple[str, Any, list[str], float, float]] = []
    # Add B4c
    detectors_temporal.append(("B4c: Radon Projection + Peak Channel Z (Non-ML)", None, [], 1.155291, 0.888908))
    for t_name, f_cols in [
        ("Tier 2 (Spectral)", TIER_2_FEATURES),
        ("Tier 3 (Weather Fusion)", TIER_3_FEATURES),
        ("Tier 4 (Physics-Informed Nuclear)", TIER_4_NUCLEAR_FEATURES),
    ]:
        clf_5 = _train_lgbm(x_tr_5[f_cols], y_tr_5, seed=42)
        vp_c = {s: clf_5.predict_proba(tr_all[s]["feats_cln"][f_cols])[tr_all[s]["m_val_calendar"], 2] for s in pilot_ids}
        vp_i = {s: clf_5.predict_proba(tr_all[s]["feats_inj"][f_cols])[:, 2] for s in pilot_ids}
        tau_95 = _select_val_threshold(vp_c, vp_i, val_vm_5, val_slices_5, target_det=0.95)
        # Also 5 FA/yr validation budget
        p_all_val = np.concatenate([vp_c[s][val_vm_5[s]] for s in pilot_ids])
        tau_5fa = select_local_calibration_threshold(
            np.concatenate([vp_c[s] for s in pilot_ids]),
            np.concatenate([val_vm_5[s] for s in pilot_ids]),
            fa_budget_per_yr=5.0,
            fallback_tau=tau_95,
        )
        detectors_temporal.append((t_name, clf_5, f_cols, tau_95, tau_5fa))

    for det_name, clf_5, f_cols, tau_95, tau_5fa in detectors_temporal:
        for op_lbl, tau_eval in [("Val_Target_95pct", tau_95), ("Val_FA_Budget_5_per_yr", tau_5fa)]:
            for yr in (2021, 2022, 2023, 2024, 2025):
                is_val_yr = yr in (2021, 2022)
                tot_obs = 0
                tot_ep = 0
                tot_pl = 0
                tot_det = 0

                for s in pilot_ids:
                    pack = tr_all[s] if is_val_yr else te_all[s]
                    dts_s = pd.to_datetime(pack["df"]["dt"])
                    yr_m = (dts_s.dt.year == yr).to_numpy(dtype=bool)
                    vm_s = (pack["feats_cln"]["gross_cpm"].notna().to_numpy(dtype=bool) if is_val_yr else pack["valid"].to_numpy(dtype=bool)) & yr_m

                    if clf_5 is None:
                        pc_s = score_b4c_radon_projection_combined(pack["feats_cln"])
                        pi_s = score_b4c_radon_projection_combined(pack["feats_inj"])
                    else:
                        pc_s = clf_5.predict_proba(pack["feats_cln"][f_cols])[:, 2]
                        pi_s = clf_5.predict_proba(pack["feats_inj"][f_cols])[:, 2]

                    on_s = (pc_s >= tau_eval) & vm_s
                    pr_s = np.concatenate([[False], on_s[:-1]])
                    ep_s = int((on_s & (~pr_s)).sum())
                    obs_s = int(vm_s.sum())

                    # Plumes starting in year `yr` at station `s`
                    slices_s = pack["val_inj_slices"] if is_val_yr else pack["test_inj_slices"]
                    pl_s = 0
                    det_s = 0
                    for sl in slices_s:
                        idx = sl[1]
                        s_dt = pd.to_datetime( pack["df"]["dt"].iloc[idx[0]] )
                        if s_dt.year == yr:
                            pl_s += 1
                            if ((pi_s[idx] >= tau_eval) & (pc_s[idx] < tau_eval)).any():
                                det_s += 1

                    tot_obs += obs_s
                    tot_ep += ep_s
                    tot_pl += pl_s
                    tot_det += det_s

                    if op_lbl == "Val_Target_95pct":
                        exp_st = max(1e-6, obs_s / HOURS_PER_YEAR)
                        fa_lo_s, fa_hi_s = exact_poisson_rate_ci(ep_s, exp_st, alpha=0.05)
                        temp_st_year_rows.append({
                            "detector_name": det_name,
                            "station_id": s,
                            "year": yr,
                            "split_role": "validation" if is_val_yr else "test",
                            "observed_hours": obs_s,
                            "exposure_station_years": round(exp_st, 4),
                            "false_alarm_episodes": ep_s,
                            "fa_per_yr": round(ep_s / exp_st, 2),
                            "fa_exact_poisson_ci_low": round(fa_lo_s, 2),
                            "fa_exact_poisson_ci_high": round(fa_hi_s, 2),
                            "n_plumes": pl_s,
                            "n_detected": det_s,
                            "detection_pct": round(100.0 * det_s / max(1, pl_s), 2),
                        })

                exp_yr = max(1e-6, tot_obs / HOURS_PER_YEAR)
                fa_yr = tot_ep / exp_yr
                fa_lo, fa_hi = exact_poisson_rate_ci(tot_ep, exp_yr, alpha=0.05)
                det_pct = 100.0 * tot_det / max(1, tot_pl)
                det_lo, det_hi = clopper_pearson_ci(tot_det, max(1, tot_pl), alpha=0.05)

                temp_year_rows.append({
                    "detector_name": det_name,
                    "operating_point": op_lbl,
                    "threshold_tau": round(tau_eval, 5),
                    "year": yr,
                    "split_role": "validation" if is_val_yr else "test",
                    "observed_hours": tot_obs,
                    "exposure_station_years": round(exp_yr, 4),
                    "false_alarm_episodes": tot_ep,
                    "fa_per_yr": round(fa_yr, 2),
                    "fa_exact_poisson_ci_low": round(fa_lo, 2),
                    "fa_exact_poisson_ci_high": round(fa_hi, 2),
                    "n_plumes": tot_pl,
                    "n_detected": tot_det,
                    "detection_pct": round(det_pct, 2),
                    "det_clopper_pearson_ci_low": round(det_lo, 2),
                    "det_clopper_pearson_ci_high": round(det_hi, 2),
                })

    pd.DataFrame(temp_year_rows).to_csv("data/processed/temporal_generalization_by_year.csv", index=False)
    pd.DataFrame(temp_st_year_rows).to_csv("data/processed/temporal_generalization_by_station_year.csv", index=False)
    print("Wrote data/processed/temporal_generalization_by_year.csv and temporal_generalization_by_station_year.csv")

    # -------------------------------------------------------------------------
    # Plot Diagnostic Figures (6 Representative Tampa Episodes + Spectral Drift & Calibration)
    # -------------------------------------------------------------------------
    _plot_tampa_diagnostic_episodes(te_all["fl_tampa"], df_tampa_eps, tampa_models_and_scores)
    _plot_tampa_spectral_drift_and_calibration(
        pd.DataFrame(ch_comp_rows),
        df_yearly_drift,
        pd.DataFrame(local_cal_rows),
        pd.DataFrame(loso_11_rows),
    )
    print("Completed generalization evaluation and figures.")


def _plot_tampa_diagnostic_episodes(
    tampa_test_pack: dict[str, Any],
    df_tampa_eps: pd.DataFrame,
    tampa_models_and_scores: dict[str, tuple],
) -> None:
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = tampa_test_pack["df"]
    feats = tampa_test_pack["feats_cln"]
    dts = pd.to_datetime(df["dt"])
    _, _, tau_t3, p_t3, _ = tampa_models_and_scores["Tier 3 (Weather Fusion)"]
    _, _, tau_t4, p_t4, _ = tampa_models_and_scores["Tier 4 (Physics-Informed Nuclear)"]

    t4_eps = df_tampa_eps[df_tampa_eps["tier_name"] == "Tier 4 (Physics-Informed Nuclear)"].copy()
    # Select 6 representative episodes spanning distinct categories/dates
    selected = []
    for cat in ("rain-coincident", "post-rain", "filter-change adjacent", "missing-data adjacent", "baseline or gain drift"):
        sub = t4_eps[t4_eps["episode_category"] == cat]
        if not sub.empty:
            selected.append(sub.sort_values("peak_score", ascending=False).iloc[0])
    while len(selected) < 6 and len(t4_eps) >= 6:
        for _, r in t4_eps.sort_values("duration_hours", ascending=False).iterrows():
            if all(r["episode_id"] != s["episode_id"] for s in selected):
                selected.append(r)
                if len(selected) == 6:
                    break

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 11.0))
    axes_flat = axes.flatten()

    for i, ep in enumerate(selected[:6]):
        ax = axes_flat[i]
        pk = int(ep["peak_idx"])
        lo = max(0, pk - 36)
        hi = min(len(df) - 1, pk + 36)
        t_sub = dts.iloc[lo : hi + 1]

        r02 = pd.to_numeric(df["cpm_r02"].iloc[lo : hi + 1], errors="coerce")
        r03 = pd.to_numeric(df["cpm_r03"].iloc[lo : hi + 1], errors="coerce")
        r05 = pd.to_numeric(df["cpm_r05"].iloc[lo : hi + 1], errors="coerce") * 5.0
        r07 = pd.to_numeric(df["cpm_r07"].iloc[lo : hi + 1], errors="coerce") * 10.0

        ax.plot(t_sub, r02, label="R02 (100-200 keV)", color="#0072B2", lw=1.2)
        ax.plot(t_sub, r03, label="R03 (200-400 keV)", color="#009E73", lw=1.2)
        ax.plot(t_sub, r05, label="5x R05 (600-800 keV)", color="#D55E00", lw=1.4)
        ax.plot(t_sub, r07, label="10x R07 (1000-1400 keV)", color="#CC79A7", lw=1.1)
        ax.axvspan(dts.iloc[int(ep["start_idx"])], dts.iloc[int(ep["end_idx"])], color="#D55E00", alpha=0.18, label="Tier 4 Alarm Window")

        ax2 = ax.twinx()
        ax2.plot(t_sub, p_t4[lo : hi + 1], color="black", ls="--", lw=1.2, label="Tier 4 Score")
        ax2.axhline(tau_t4, color="black", ls=":", lw=0.9)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("P(Fission)", fontsize=8.5)

        title = (
            f"Ep {i + 1} ({ep['episode_category']}): {ep['peak_utc'][:13]}h | "
            f"dur={ep['duration_hours']}h, peak={ep['peak_score']:.3f}\n"
            f"Top SHAP: {ep['top_contrib_feature_1']} (+{ep['top_contrib_shap_1']:.2f}), "
            f"{ep['top_contrib_feature_2']} (+{ep['top_contrib_shap_2']:.2f})"
        )
        ax.set_title(title, fontsize=8.8, fontweight="bold")
        ax.set_ylabel("Channel Count Rate (CPM)", fontsize=8.5)
        ax.tick_params(axis="x", labelsize=7.5, rotation=15)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7.2, frameon=True)

    fig.tight_layout()
    fig.savefig(fig_dir / "tampa_loso_diagnostic_episodes.png", dpi=300)
    plt.close(fig)


def _plot_tampa_spectral_drift_and_calibration(
    df_ch_comp: pd.DataFrame,
    df_yearly_drift: pd.DataFrame,
    df_local_cal: pd.DataFrame,
    df_loso_11: pd.DataFrame,
) -> None:
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # Panel A: share_r02 vs share_r05 across all 11 stations + injected plumes
    for _, r in df_ch_comp.iterrows():
        sid = str(r["station_id"])
        x = float(r["share_r02_mean"]) * 100.0
        y = float(r["share_r05_mean"]) * 100.0
        if sid == "fl_tampa":
            axes[0].scatter(x, y, color="#D55E00", s=95, zorder=5, label="Tampa, FL (Clean)")
            axes[0].annotate("Tampa, FL", (x + 0.2, y + 0.03), fontsize=8.5, fontweight="bold", color="#D55E00")
        elif sid == "4_Pilot_Train_Stations":
            axes[0].scatter(x, y, color="#CC79A7", marker="^", s=95, zorder=5, label="Injected Fission Plumes")
            axes[0].annotate("Injected Plumes", (x - 4.8, y - 0.16), fontsize=8.5, fontweight="bold", color="#CC79A7")
        elif sid == "hi_honolulu":
            axes[0].scatter(x, y, color="#0072B2", s=65, alpha=0.9)
            axes[0].annotate("Honolulu, HI", (x + 0.3, y + 0.04), fontsize=8.0, color="#0072B2")
        else:
            axes[0].scatter(x, y, color="#0072B2", s=55, alpha=0.85)
    axes[0].set_xlabel("Mean R02 Energy Share (100-200 keV, %)", fontsize=9.5)
    axes[0].set_ylabel("Mean R05 Energy Share (600-800 keV, %)", fontsize=9.5)
    axes[0].set_title("A. Spectral Baseline Offset (11 Stations)", fontsize=10, fontweight="bold")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel B: Year-by-year share_r05 drift 2017-2025 for Tampa vs other pilot stations
    for sid, color, lw, lbl in [
        ("fl_tampa", "#D55E00", 2.4, "Tampa, FL"),
        ("al_birmingham", "#0072B2", 1.3, "Birmingham, AL"),
        ("dc_washington", "#009E73", 1.3, "Washington, DC"),
        ("ca_san_diego", "#56B4E9", 1.3, "San Diego, CA"),
        ("tx_dallas", "#E69F00", 1.3, "Dallas, TX"),
    ]:
        sub = df_yearly_drift[df_yearly_drift["station_id"] == sid].sort_values("year")
        axes[1].plot(sub["year"], sub["share_r05_mean"] * 100.0, marker="o", color=color, lw=lw, label=lbl)
    axes[1].set_xlabel("Calendar Year", fontsize=9.5)
    axes[1].set_ylabel("Mean R05 Share (600-800 keV, %)", fontsize=9.5)
    axes[1].set_title("B. Tampa Spectral Drift (2017-2025)", fontsize=10, fontweight="bold")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel C: 11-Station Tier 4 LOSO False-Alarm Rate (Zero-Shot vs Local-168h vs 12m Local Calibration)
    t4_zs = df_loso_11[(df_loso_11["detector_name"] == "Tier 4 (Physics-Informed Nuclear)") & (df_loso_11["transfer_mode"] == "ZeroShot_Uncalibrated")].set_index("held_out_station")
    t4_168 = df_loso_11[(df_loso_11["detector_name"] == "Tier 4 (Physics-Informed Nuclear)") & (df_loso_11["transfer_mode"] == "ZeroShot_Local168h_Normalized")].set_index("held_out_station")
    t4_lc = df_loso_11[(df_loso_11["detector_name"] == "Tier 4 (Physics-Informed Nuclear)") & (df_loso_11["transfer_mode"] == "LocalCalibrated_12m_Clean")].set_index("held_out_station")
    st_order = list(t4_zs.index)
    y_pos = np.arange(len(st_order))
    axes[2].barh(y_pos - 0.25, t4_zs["fa_per_yr"], height=0.25, color="#D55E00", label="Zero-Shot Uncalibrated")
    axes[2].barh(y_pos, t4_168.loc[st_order, "fa_per_yr"], height=0.25, color="#0072B2", label="Zero-Shot Local-168h Norm")
    axes[2].barh(y_pos + 0.25, t4_lc.loc[st_order, "fa_per_yr"], height=0.25, color="#009E73", label="12m Local Clean Calibrated")
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(st_order, fontsize=8)
    axes[2].set_xscale("symlog", linthresh=2.0)
    axes[2].set_xlabel("Test False Alarms / Station-Year", fontsize=9.5)
    axes[2].set_title("C. 11-Station Tier 4 LOSO Transfer", fontsize=10, fontweight="bold")
    axes[2].legend(loc="lower right", fontsize=7.5)
    axes[2].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "tampa_spectral_drift_and_calibration.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    run_generalization_pipeline()
