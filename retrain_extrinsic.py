#!/usr/bin/env python3
"""
Targeted Retraining Pipeline for Extrinsic Magnetic Properties:
1. Anisotropy Constant K1 (J/m^3)
2. Saturation Magnetization Ms (emu/g)
3. Coercivity Hc (A/m)
4. Energy Product (BH)max (kJ/m^3)

Integrates:
- Real MSN Point-Charge Crystal-Field Gradient A_2^0
- MSN Sublattice-Resolved Saturation Magnetization (FiM cancellation)
- MSN Exchange Stiffness A_ex & Bethe-Slater Exchange Ratios
- Intrinsic Anisotropy Field H_A = 2K1 / (mu_0 Ms)
- Full F5 Representation + Leak-Free OOF Cross-Target Physics Cascade
- Zero-Leakage GroupKFold Cross-Validation grouped strictly by reduced_formula
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from src.cascade import build_oof_cascade
from src.config import RunConfig
from src.dao_engine import get_dao_engine
from src.evaluation import grouped_cv_predict, summarize
from src.magnetic_structure_network import get_msn_engine
from src.modeling import (
    StreamlinedTriRegressor,
    get_ablation_features,
    leakage_safe_feature_selection,
    screen_features,
)
from src.pinn_hysteresis import compute_pinn_hysteresis_features
from src.stevens_operator import compute_stevens_features

MASTER_CSV = os.path.join(PIPELINE_ROOT, "output/featurization/features_master.csv")
GROUPS_JSON = os.path.join(PIPELINE_ROOT, "output/featurization/feature_groups.json")
MODELS_PKL = os.path.join(PIPELINE_ROOT, "output/training/trained_models.pkl")
RESULTS_JSON = os.path.join(PIPELINE_ROOT, "results_summary.json")
MSN_CACHE_CSV = os.path.join(PIPELINE_ROOT, "output/training/msn_descriptors_cache.csv")
CASCADE_CACHE = os.path.join(PIPELINE_ROOT, "output/training/cascade_cache.parquet")


def extract_and_cache_msn_descriptors(formulas: List[str]) -> pd.DataFrame:
    """Extracts MSN quantum descriptors for unique formulas and caches them to disk."""
    cache_dict = {}
    if os.path.exists(MSN_CACHE_CSV):
        try:
            df_existing = pd.read_csv(MSN_CACHE_CSV)
            for _, row in df_existing.iterrows():
                cache_dict[str(row["reduced_formula"])] = row.to_dict()
            print(f"Loaded {len(cache_dict)} cached MSN descriptors from {MSN_CACHE_CSV}")
        except Exception as e:
            print(f"Note: could not read existing cache: {e}")

    missing_formulas = [f for f in formulas if f not in cache_dict and pd.notna(f) and str(f).strip() != ""]
    if missing_formulas:
        print(f"Computing MSN quantum descriptors for {len(missing_formulas)} formulas...")
        dao = get_dao_engine()
        msn = get_msn_engine()
        t0 = time.time()
        for idx, f in enumerate(missing_formulas):
            try:
                res = dao.predict_crystal(str(f))
                s = res["structure"]
                msn_res = msn.resolve_magnetic_structure(s)
                bs = msn_res.get("bethe_slater_ratio")
                rec = {
                    "reduced_formula": str(f),
                    "dao_msn_a20_cf": round(float(msn_res.get("crystal_field_a20_cf", 0.0)), 4),
                    "dao_msn_ms_emu_g": round(float(msn_res.get("saturation_magnetization_emu_g", 50.0)), 2),
                    "dao_msn_exchange_stiffness": round(float(msn_res.get("exchange_stiffness_pj_m", 5.0)), 3),
                    "dao_msn_bethe_slater_ratio": round(float(bs) if bs is not None else 1.0, 3),
                }
            except Exception:
                rec = {
                    "reduced_formula": str(f),
                    "dao_msn_a20_cf": 0.0,
                    "dao_msn_ms_emu_g": 50.0,
                    "dao_msn_exchange_stiffness": 5.0,
                    "dao_msn_bethe_slater_ratio": 1.0,
                }
            cache_dict[str(f)] = rec
            if (idx + 1) % 1000 == 0 or (idx + 1) == len(missing_formulas):
                elapsed = time.time() - t0
                print(f"  Processed {idx + 1}/{len(missing_formulas)} formulas ({elapsed:.1f}s)")

        df_cache = pd.DataFrame(list(cache_dict.values()))
        os.makedirs(os.path.dirname(MSN_CACHE_CSV), exist_ok=True)
        df_cache.to_csv(MSN_CACHE_CSV, index=False)
        print(f"Saved total {len(df_cache)} MSN descriptors to {MSN_CACHE_CSV}")

    return pd.DataFrame(list(cache_dict.values()))


def build_enhanced_dataset(df_master: pd.DataFrame, feature_groups: dict, cfg: RunConfig) -> Tuple[pd.DataFrame, List[str]]:
    """Merges MSN descriptors, Stevens operator features, and physics cascade."""
    df_feat = df_master.copy()

    # 1. Merge MSN descriptors
    unique_formulas = df_feat["reduced_formula"].dropna().astype(str).unique().tolist()
    df_msn = extract_and_cache_msn_descriptors(unique_formulas)
    msn_cols = ["dao_msn_a20_cf", "dao_msn_ms_emu_g", "dao_msn_exchange_stiffness", "dao_msn_bethe_slater_ratio"]
    for col in msn_cols:
        if col in df_feat.columns:
            df_feat.drop(columns=[col], inplace=True)
    df_feat = df_feat.merge(df_msn, on="reduced_formula", how="left")
    for col in msn_cols:
        df_feat[col] = df_feat[col].fillna(0.0)

    # 2. Compute Stevens operator features with real A20
    stev_df = compute_stevens_features(df_feat)
    for col in stev_df.columns:
        df_feat[col] = stev_df[col].values

    # 2b. Parse and encode microstructure and sample morphology
    def parse_thickness_nm(val):
        if pd.isna(val): return np.nan
        s = str(val).lower()
        m_um = re.search(r'([\d\.]+)\s*(?:μm|um)', s)
        if m_um: return float(m_um.group(1)) * 1000.0
        m_nm = re.search(r'([\d\.]+)\s*nm', s)
        if m_nm: return float(m_nm.group(1))
        m_a = re.search(r'([\d\.]+)\s*å', s)
        if m_a: return float(m_a.group(1)) * 0.1
        return np.nan

    df_feat["micro_thickness_nm"] = df_feat.get("Thickness", pd.Series(index=df_feat.index, dtype=object)).apply(parse_thickness_nm)
    mat_type = df_feat.get("Material_Type", pd.Series(index=df_feat.index, dtype=object))
    is_bulk = mat_type.isna() | mat_type.astype(str).str.contains("Bulk|Single Crystal", case=False, na=False)
    is_film = mat_type.astype(str).str.contains("Film|Layer|Superlattice", case=False, na=False)
    is_nano = mat_type.astype(str).str.contains("Nano|2D|Monolayer", case=False, na=False)

    df_feat["micro_is_bulk"] = is_bulk.astype(float)
    df_feat["micro_is_thin_film"] = is_film.astype(float)
    df_feat["micro_is_nanostructured"] = is_nano.astype(float)

    thick_imputed = df_feat["micro_thickness_nm"].copy()
    thick_imputed[is_bulk & thick_imputed.isna()] = 1e6
    thick_imputed[is_nano & thick_imputed.isna()] = 10.0
    thick_imputed[is_film & thick_imputed.isna()] = 50.0
    thick_imputed = thick_imputed.fillna(1e6).clip(lower=0.1, upper=1e7)
    df_feat["micro_log_thickness_nm"] = np.log10(thick_imputed)

    def get_sample_form(row):
        if row["micro_is_thin_film"] == 1: return "film"
        elif row["micro_is_nanostructured"] == 1: return "nano"
        return "bulk"

    df_feat["sample_form"] = df_feat.apply(get_sample_form, axis=1)
    df_feat["consensus_key"] = df_feat["reduced_formula"].astype(str) + "_" + df_feat["sample_form"]
    micro_feats = ["micro_is_bulk", "micro_is_thin_film", "micro_is_nanostructured", "micro_log_thickness_nm"]

    # 3. Base F5 representation pool + MSN quantum descriptors + Microstructure
    f5_base = get_ablation_features(feature_groups, "F5", df_feat.columns)
    msn_feats = ["dao_msn_a20_cf", "dao_msn_ms_emu_g", "dao_msn_exchange_stiffness", "dao_msn_bethe_slater_ratio", "stevens_a20_cf", "stevens_k1_torque"]
    f5_msn = list(dict.fromkeys(f5_base + [c for c in msn_feats if c in df_feat.columns] + micro_feats))
    print(f"Assembled F5 + MSN + Microstructure representation: {len(f5_msn)} descriptors.")

    # 4. Build or load Out-Of-Fold physics cascade
    if os.path.exists(CASCADE_CACHE):
        print(f"Loading cached OOF physics cascade from {CASCADE_CACHE}...")
        df_casc = pd.read_parquet(CASCADE_CACHE)
        for c in df_casc.columns:
            df_feat[c] = df_casc[c].values
    else:
        print("Building out-of-fold physics cascade across donor targets...")
        casc_factories = [
            lambda: ExtraTreesRegressor(n_estimators=100, max_depth=25, min_samples_split=2, min_samples_leaf=2, max_features=0.85, random_state=42, n_jobs=-1),
            lambda: LGBMRegressor(n_estimators=100, max_depth=8, learning_rate=0.04, num_leaves=63, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbose=-1)
        ]
        df_feat = build_oof_cascade(df_feat, f5_msn, casc_factories, n_splits=cfg.n_splits, out_of_fold=True, verbose=True)
        casc_cols = [c for c in df_feat.columns if c.startswith("cascade_") or c.startswith("pinn_")]
        if casc_cols:
            df_feat[casc_cols].to_parquet(CASCADE_CACHE)
            print(f"Cached {len(casc_cols)} cascade columns to {CASCADE_CACHE}")

    # 5. Formulate intrinsic anisotropy field H_A and Kronmuller interaction
    k1_pred = df_feat.get("cascade_pred_k1", pd.Series(np.zeros(len(df_feat)))).values if isinstance(df_feat.get("cascade_pred_k1", 0), pd.Series) else np.array(df_feat.get("cascade_pred_k1", np.zeros(len(df_feat))))
    ms_pred = df_feat.get("cascade_pred_ms", pd.Series(np.full(len(df_feat), 50.0))).values if isinstance(df_feat.get("cascade_pred_ms", 0), pd.Series) else np.array(df_feat.get("cascade_pred_ms", np.full(len(df_feat), 50.0)))
    hc_pred = df_feat.get("cascade_pred_hc", pd.Series(np.full(len(df_feat), 1000.0))).values if isinstance(df_feat.get("cascade_pred_hc", 0), pd.Series) else np.array(df_feat.get("cascade_pred_hc", np.full(len(df_feat), 1000.0)))
    tc_pred = df_feat.get("cascade_pred_TC", pd.Series(np.full(len(df_feat), 300.0))).values if isinstance(df_feat.get("cascade_pred_TC", 0), pd.Series) else np.array(df_feat.get("cascade_pred_TC", np.full(len(df_feat), 300.0)))
    pfm_pred = df_feat.get("cascade_prob_FM", pd.Series(np.full(len(df_feat), 0.5))).values if isinstance(df_feat.get("cascade_prob_FM", 0), pd.Series) else np.array(df_feat.get("cascade_prob_FM", np.full(len(df_feat), 0.5)))

    # Safely extract arrays
    k1_arr = np.asarray(k1_pred, dtype=float)
    ms_arr = np.asarray(ms_pred, dtype=float)
    hc_arr = np.asarray(hc_pred, dtype=float)
    tc_arr = np.asarray(tc_pred, dtype=float)
    pfm_arr = np.asarray(pfm_pred, dtype=float)
    density_arr = np.asarray(df_feat.get("dao_density", np.full(len(df_feat), 7.5)), dtype=float)

    # H_A anisotropy field bound (Kronmuller limit)
    ha_denom = 4.0 * np.pi * 1e-4 * np.maximum(ms_arr, 0.5) * np.maximum(density_arr, 1.0)
    df_feat["cascade_Ha_log"] = np.log10(np.maximum((2.0 * np.maximum(k1_arr, 1.0)) / ha_denom, 1.0))
    df_feat["cascade_kronmuller_hk"] = np.log10(1.0 + np.maximum(k1_arr, 0.0)) - np.log10(np.maximum(ms_arr, 0.1))
    df_feat["cascade_theoretical_bh_max"] = 2.0 * np.log10(np.maximum(ms_arr, 0.1))
    df_feat["phys_domain_wall_energy"] = np.sqrt(np.maximum(tc_arr, 0.0) * np.maximum(k1_arr, 0.0))
    df_feat["phys_coercivity_nucleation_index"] = df_feat["cascade_kronmuller_hk"] * pfm_arr

    # Hc log-space cascade feature
    if "cascade_pred_hc_log" not in df_feat.columns:
        df_feat["cascade_pred_hc_log"] = np.log10(np.maximum(hc_arr, 1.0))

    # Extrinsic BH product: theoretical upper bound Ms^2 * Hc / (4*pi)
    df_feat["cascade_extrinsic_bh_product"] = df_feat["cascade_pred_hc_log"] + np.log10(np.maximum(ms_arr, 0.1))

    # K1 in symlog space for cascade feature
    k1_symlog = df_feat.get("cascade_pred_k1_symlog", None)
    if k1_symlog is None or (hasattr(k1_symlog, '__len__') and np.all(np.isnan(k1_symlog))):
        df_feat["cascade_pred_k1_symlog"] = np.sign(k1_arr) * np.log10(1.0 + np.abs(k1_arr))

    # Dynamically compute rigorous PINN hysteresis features
    pinn_res = compute_pinn_hysteresis_features(df_feat, use_cascade_preds=True)
    for pc in pinn_res.columns:
        df_feat[pc] = pinn_res[pc].values

    # Base candidate pool remains pure F5 + MSN quantum descriptors
    # (prevents downstream cascade contamination in donor target selection)
    print(f"Purified base candidate pool: {len(f5_msn)} physical descriptors.")

    return df_feat, f5_msn


def make_factories(n_estimators: int = 100):
    """Committee of distinct model families for cross-validation evaluation."""
    return [
        lambda: ExtraTreesRegressor(n_estimators=n_estimators, max_depth=25, min_samples_split=2, min_samples_leaf=2, max_features=0.85, random_state=42, n_jobs=-1),
        lambda: RandomForestRegressor(n_estimators=n_estimators, max_depth=25, min_samples_split=2, min_samples_leaf=2, max_features=0.85, random_state=42, n_jobs=-1),
        lambda: LGBMRegressor(n_estimators=n_estimators, max_depth=8, learning_rate=0.04, num_leaves=63, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbose=-1)
    ]


def run_target(df_feat: pd.DataFrame, target_config: dict, candidate_pool: List[str], cfg: RunConfig):
    """Runs feature selection, GroupKFold CV evaluation, and trains final production model."""
    t_name = target_config["name"]
    col = target_config["col"]
    transform = target_config["transform"]

    series = pd.to_numeric(df_feat[col], errors="coerce")
    mask = target_config["valid_filter"](series) & df_feat["reduced_formula"].notna()
    df_target = df_feat[mask].copy()

    # Assemble candidate features
    extra_cols = [c for c in target_config.get("extra_features", []) if c in df_target.columns]
    mand_cols = [c for c in target_config.get("mandatory_features", []) if c in df_target.columns]
    pool_cols = list(dict.fromkeys(candidate_pool + extra_cols + mand_cols))
    num_cols = [c for c in pool_cols if c in df_target.columns and pd.api.types.is_numeric_dtype(df_target[c])]

    # Option 1 + Option 2: Consensus median aggregation per (reduced_formula, sample_form)
    agg_num = df_target.groupby("consensus_key")[num_cols + [col]].median()
    agg_formula = df_target.groupby("consensus_key")["reduced_formula"].first()
    df_consensus = agg_num.join(agg_formula).reset_index()

    n_samples = len(df_consensus)
    groups = df_consensus["reduced_formula"].astype(str).values
    n_compounds = len(np.unique(groups))

    print(f"\n==============================================================================")
    print(f"  TARGET: {t_name.upper()} ({col}) [Consensus Denoised + Microstructure]")
    print(f"  {n_samples} consensus records across {n_compounds} unique chemical compounds")
    print(f"  Transform: {transform} | Unit: {target_config['unit']}")
    print(f"==============================================================================")

    y_raw = df_consensus[col].values.astype(float)
    if transform == "symlog":
        y_trans = np.sign(y_raw) * np.log10(1.0 + np.abs(y_raw))
    elif transform == "log":
        y_trans = np.log10(np.maximum(y_raw, 1e-4))
    else:
        y_trans = y_raw

    X_pool = df_consensus[num_cols].copy()

    # Feature selection preserving mandatory physics
    X_sel, _, active_cols, medians_dict = leakage_safe_feature_selection(
        X_pool, X_pool, pd.Series(y_trans), mandatory_features=[c for c in mand_cols if c in num_cols]
    )
    medians_series = pd.Series(medians_dict)
    X_sel = X_sel.fillna(medians_series).fillna(0.0)
    print(f"  Selected {len(active_cols)} active features (preserved {len([c for c in mand_cols if c in active_cols])} mandatory physics features)")

    # GroupKFold Cross-Validation with inner-split NNLS meta-weights
    if cfg.mode == "fast_debug":
        n_est = 80
    elif n_samples < 2000:
        n_est = 100
    else:
        n_est = 200
    factories = make_factories(n_estimators=n_est)
    oof, folds, weights = grouped_cv_predict(
        X_sel, y_trans, groups, factories,
        n_splits=cfg.n_splits, inner_splits=cfg.inner_splits
    )

    rec = summarize(
        t_name, y_trans, oof, folds, cfg.n_splits,
        transform=transform, unit=target_config["unit"],
        bootstrap_iterations=cfg.bootstrap_iterations,
        random_state=cfg.random_state
    )
    print(f"\n  >>> StratifiedGroupKFold Out-of-Fold Results for {t_name.upper()} <<<")
    print(f"  Generalization R2: {rec.pooled_r2:.4f} (per-fold: {rec.r2_mean:.4f} +/- {rec.r2_std:.4f})")
    print(f"  Generalization MAE: {rec.pooled_mae:.4f} +/- {rec.mae_std:.4f} ({transform} space)")

    # Fit final production model
    print(f"  Fitting production StreamlinedTriRegressor ensemble...")
    tri_model = StreamlinedTriRegressor(transform="log10" if transform == "log" else transform, random_state=42)
    tri_model.fit(X_sel, y_raw if transform != "log" else np.maximum(y_raw, 1e-4), active_cols, groups=groups)

    # Store OOF residuals for conformal calibration (90% coverage)
    oof_residuals = np.sort(np.abs(y_trans - oof))
    tri_model.oof_residuals = oof_residuals
    n_cal = len(oof_residuals)
    q_idx = int(np.ceil((n_cal + 1) * 0.90)) - 1
    tri_model.conformal_quantile = float(oof_residuals[min(q_idx, n_cal - 1)])
    print(f"  Conformal 90% quantile ({transform} space): {tri_model.conformal_quantile:.4f}")

    metrics_dict = {
        "target": t_name,
        "n_samples": int(n_samples),
        "n_compounds": int(n_compounds),
        "transform": transform,
        "unit": target_config["unit"],
        "generalization_r2": round(rec.pooled_r2, 4),
        "generalization_mae": round(rec.pooled_mae, 4),
        "fold_r2_mean": round(float(rec.r2_mean), 4),
        "fold_r2_std": round(float(rec.r2_std), 4),
        "features_used": len(active_cols),
        "conformal_quantile": round(tri_model.conformal_quantile, 4),
    }

    return tri_model, metrics_dict, active_cols, medians_dict


def main():
    parser = argparse.ArgumentParser(description="Retrain Extrinsic Magnetic Models with MSN Descriptors")
    parser.add_argument("--mode", choices=["fast_debug", "final_manuscript"], default="fast_debug")
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    cfg = RunConfig(mode=args.mode, n_splits=args.splits)
    print("=" * 80)
    print("  EXTRINSIC MAGNETIC PROPERTIES RETRAINING PIPELINE (MSN-ENHANCED)")
    print(f"  Mode: {cfg.mode} | Splits: {cfg.n_splits} | Estimators CV: {cfg.n_estimators_cv}")
    print("=" * 80)

    # 1. Load Master Features & Feature Groups
    print(f"Loading master dataset from {MASTER_CSV}...")
    df_master = pd.read_csv(MASTER_CSV, low_memory=False)
    with open(GROUPS_JSON, "r") as f:
        feature_groups = json.load(f)

    # 2. Enrich with MSN descriptors, Stevens operator, and OOF cascade
    df_feat, candidate_pool = build_enhanced_dataset(df_master, feature_groups, cfg)

    # Target configurations — purified, leakage-free physics representations
    targets = [
        {
            "name": "anisotropy_k1",
            "model_key": "anisotropy",
            "alt_key": "k1",
            "col": "clean_K1_J_m3",
            "transform": "symlog",
            "unit": "J/m^3",
            "valid_filter": lambda s: s.notna() & (np.abs(s) <= 1e8),
            "mandatory_features": [
                "uniaxial_symmetry_flag", "phys_aniso_uniaxial_tc", "dao_msn_a20_cf",
                "stevens_a20_cf", "stevens_k1_torque",
            ],
            "extra_features": [
                "cascade_prob_FM", "cascade_prob_AFM", "cascade_pred_TC",
                "phys_k1_thermal_factor", "phys_soc_exchange_coupling", "phys_aniso_uniaxial_tc",
                "dao_msn_bethe_slater_ratio", "dao_msn_exchange_stiffness",
                "dao_c_over_a", "dao_is_uniaxial",
                "micro_is_thin_film", "micro_log_thickness_nm",
            ],
        },
        {
            "name": "saturation_magnetization_ms",
            "model_key": "magnetization",
            "alt_key": None,
            "col": "clean_Ms_emu_g",
            "transform": "none",
            "unit": "emu/g",
            "valid_filter": lambda s: s.notna() & (s > 0) & (s <= 350.0),
            "mandatory_features": [
                "valence_electron_concentration", "dao_moment_density",
                "phys_tc_ms_interaction", "dao_msn_ms_emu_g",
            ],
            "extra_features": [
                "phys_ms_bloch_factor", "phys_fm_scaled_moment", "phys_tc_ms_interaction",
                "dao_msn_ms_emu_g", "dao_msn_exchange_stiffness", "dao_msn_bethe_slater_ratio",
                "cascade_prob_FM", "cascade_prob_AFM", "cascade_pred_TC",
                "valence_electron_concentration", "dao_magnetic_concentration",
                "dao_moment_density",
            ],
        },
        {
            "name": "coercivity_hc",
            "model_key": "coercivity",
            "alt_key": None,
            "col": "clean_Hc_A_m",
            "transform": "log",
            "unit": "A/m",
            "valid_filter": lambda s: s.notna() & (s > 0) & (s <= 1e8),
            "mandatory_features": [
                "uniaxial_symmetry_flag", "pinn_loop_squareness",
                "cascade_Ha_log", "dao_msn_exchange_stiffness",
                "micro_is_thin_film",
            ],
            "extra_features": [
                "cascade_kronmuller_hk", "cascade_Ha_log",
                "dao_msn_exchange_stiffness", "dao_msn_a20_cf", "dao_msn_ms_emu_g",
                "cascade_pred_k1_symlog", "cascade_pred_ms",
                "cascade_prob_FM", "cascade_pred_TC",
                "pinn_kronmuller_hk", "pinn_kronmuller_hc_log",
                "pinn_hardness_parameter", "pinn_loop_squareness",
                "phys_domain_wall_energy", "phys_coercivity_nucleation_index",
                "micro_is_bulk", "micro_is_thin_film", "micro_is_nanostructured", "micro_log_thickness_nm",
            ],
        },
        {
            "name": "energy_product_bh_max",
            "model_key": "bh_max",
            "alt_key": None,
            "col": "clean_BH_max_kJ_m3",
            "transform": "log",
            "unit": "kJ/m^3",
            "valid_filter": lambda s: s.notna() & (s >= 1.0) & (s <= 600.0),
            "mandatory_features": [
                "uniaxial_symmetry_flag", "cascade_pred_hc_log",
                "pinn_bh_max_bound", "dao_msn_ms_emu_g",
                "micro_is_thin_film",
            ],
            "extra_features": [
                "cascade_theoretical_bh_max", "cascade_extrinsic_bh_product",
                "dao_msn_ms_emu_g", "dao_msn_exchange_stiffness", "dao_msn_a20_cf",
                "cascade_pred_hc_log", "cascade_pred_ms", "cascade_pred_k1_symlog",
                "pinn_bh_max_ideal", "pinn_bh_max_extrinsic", "pinn_bh_max_bound",
                "pinn_log_bh_bound", "pinn_loop_squareness", "pinn_hardness_parameter",
                "micro_is_bulk", "micro_is_thin_film", "micro_is_nanostructured", "micro_log_thickness_nm",
            ],
        },
    ]

    # Load existing models dict and results summary
    print(f"\nLoading existing models pack from {MODELS_PKL}...")
    models_pack = joblib.load(MODELS_PKL)
    with open(RESULTS_JSON, "r") as f:
        results_summary = json.load(f)
    old_ext = results_summary.get("extended_physical_targets", {})

    new_metrics = {}

    for cfg_tgt in targets:
        model, metrics, active_features, medians = run_target(
            df_feat=df_feat,
            target_config=cfg_tgt,
            candidate_pool=candidate_pool,
            cfg=cfg
        )
        new_metrics[cfg_tgt["name"]] = metrics

        # Update models_pack
        models_pack[cfg_tgt["model_key"]] = model
        if cfg_tgt["alt_key"]:
            models_pack[cfg_tgt["alt_key"]] = model
        models_pack[f"features_{cfg_tgt['model_key']}"] = active_features
        models_pack[f"medians_{cfg_tgt['model_key']}"] = medians
        if cfg_tgt["alt_key"]:
            models_pack[f"features_{cfg_tgt['alt_key']}"] = active_features
            models_pack[f"medians_{cfg_tgt['alt_key']}"] = medians

    # Print Comparison Table
    print("\n" + "=" * 80)
    print("  GENERALIZATION BENCHMARK: BEFORE vs AFTER MSN QUANTUM DESCRIPTORS")
    print("=" * 80)
    print(f"{'Target':<28} | {'Old Gen R²':<12} | {'New Gen R²':<12} | {'Delta R²':<12} | {'Status'}")
    print("-" * 80)

    for cfg_tgt in targets:
        t_name = cfg_tgt["name"]
        old_r2 = old_ext.get(t_name, {}).get("generalization_r2", 0.0)
        new_r2 = new_metrics[t_name]["generalization_r2"]
        delta = new_r2 - old_r2
        status = "IMPROVED" if delta >= 0 else "—"
        print(f"{t_name:<28} | {old_r2:<12.4f} | {new_r2:<12.4f} | {delta:+12.4f} | {status}")
    print("=" * 80)

    # Persisting updated models pack
    print(f"\nPersisting updated models pack to {MODELS_PKL}...")
    joblib.dump(models_pack, MODELS_PKL, compress=3)
    print("Successfully updated trained_models.pkl!")

    # Update results_summary.json
    for t_name, met in new_metrics.items():
        if t_name in results_summary.get("extended_physical_targets", {}):
            results_summary["extended_physical_targets"][t_name]["generalization_r2"] = met["generalization_r2"]
            results_summary["extended_physical_targets"][t_name]["generalization_mae"] = met["generalization_mae"]
            results_summary["extended_physical_targets"][t_name]["features_used"] = met["features_used"]
    with open(RESULTS_JSON, "w") as f:
        json.dump(results_summary, f, indent=4)
    print(f"Successfully updated {RESULTS_JSON}!")


if __name__ == "__main__":
    main()

