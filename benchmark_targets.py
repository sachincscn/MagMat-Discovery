#!/usr/bin/env python3
"""
Honest benchmark for the extended physical targets (K1, Ms, Hc, (BH)max, theta_p).

Produces the artifact the pipeline previously lacked: a persisted, reproducible
record of what each representation tier is actually worth, evaluated with

  * GroupKFold on reduced_formula (no formula crosses the split),
  * blend weights fitted on inner out-of-fold predictions only,
  * per-fold mean +/- std and a bootstrap CI alongside the pooled value,
  * an out-of-fold physics cascade.

It also evaluates three physically motivated changes:

  K1 unit filter    13% of clean_K1_J_m3 lies below 1 J/m^3, down to 1e-24 --
                    impossible for a magnetocrystalline anisotropy constant and
                    almost certainly unconverted units. Restrict to a physical
                    window.
  K1 two-stage      Sign (easy-axis vs easy-plane) is a discrete symmetry
                    outcome, not a smooth function; regressing symlog(K1) across
                    a sign change is ill-posed. Classify sign, regress |K1|.
  Ms as moment      emu/g is mass-normalised. The Slater-Pauling relation is
                    linear in moment per atom, n_muB = sigma * M_bar / 5585.
                    Predict that, convert back for reporting.

Usage:
    python benchmark_targets.py --mode final_manuscript --out output/benchmarks
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import r2_score, mean_absolute_error, f1_score
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor, LGBMClassifier

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from src.config import RunConfig
from src.evaluation import grouped_cv_predict, summarize, MetricRecord
from src.cascade import build_oof_cascade, cascade_feature_names

MASTER = os.path.join(PIPELINE_ROOT, "output/featurization/features_master.csv")
GROUPS_JSON = os.path.join(PIPELINE_ROOT, "output/featurization/feature_groups.json")

# Representation tiers, mirroring get_ablation_features in modeling.py.
TIERS = {
    "F1": ["composition_basic", "element_fractions"],
    "F3": ["composition_basic", "element_fractions", "magpie", "magpie_magnetic_relevant",
           "valence", "magnetic_chemistry", "magpieex_magnetic", "magpieex_lite",
           "magpieex_cation_anion", "oxidation_states", "formula_family",
           "ionic_and_electronegativity", "boundary_interaction_indices",
           "gka_and_exchange", "anisotropy_and_soc", "mean_field_priors",
           "frustration_and_competition", "structural_tolerance"],
    "F5": None,  # F3 + proxies + MACE + symmetry; filled in below
}

TARGETS = {
    "anisotropy_k1": dict(col="clean_K1_J_m3", transform="symlog", unit="J/m^3",
                          valid=lambda s: s.notna()),
    "saturation_magnetization_ms": dict(col="clean_Ms_emu_g", transform="none", unit="emu/g",
                                        valid=lambda s: s.notna() & (s > 0)),
    "coercivity_hc": dict(col="clean_Hc_A_m", transform="log", unit="A/m",
                          valid=lambda s: s.notna() & (s > 0) & (s <= 1e8)),
    "energy_product_bh_max": dict(col="clean_BH_max_kJ_m3", transform="log", unit="kJ/m^3",
                                  valid=lambda s: s.notna() & (s >= 1.0) & (s <= 600.0)),
    "curie_weiss_theta_p": dict(col="clean_theta_p_K", transform="none", unit="K",
                                valid=lambda s: s.notna() & (s >= -2000) & (s <= 2000)),
}

# Physically defensible window for a magnetocrystalline anisotropy constant.
K1_FLOOR, K1_CEIL = 1.0e0, 1.0e8


def forward(y, kind):
    if kind == "symlog":
        return np.sign(y) * np.log10(1.0 + np.abs(y))
    if kind == "log":
        return np.log10(y)
    return y


def make_factories(cfg: RunConfig):
    n = cfg.n_estimators_cv
    return [
        lambda: ExtraTreesRegressor(n_estimators=n, max_depth=25, min_samples_split=2,
                                    min_samples_leaf=2, max_features=0.85,
                                    random_state=cfg.random_state, n_jobs=-1),
        lambda: LGBMRegressor(n_estimators=n, max_depth=8, learning_rate=0.04, num_leaves=63,
                              subsample=0.85, colsample_bytree=0.85,
                              random_state=cfg.random_state, n_jobs=-1, verbose=-1),
    ]


def prep(df, cols):
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    X = df.loc[:, ~df.columns.duplicated()][cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0.0)
    return X.loc[:, X.var() > 1e-8]


def evaluate(df, name, col, kind, valid, features, cfg, unit="", label=""):
    s = pd.to_numeric(df[col], errors="coerce")
    mask = valid(s) & df["reduced_formula"].notna()
    d = df[mask]
    if len(d) < 50:
        return None
    y = forward(s[mask].values.astype(float), kind)
    X = prep(d, features)
    groups = d["reduced_formula"].values

    oof, folds, weights = grouped_cv_predict(
        X, y, groups, make_factories(cfg),
        n_splits=cfg.n_splits, inner_splits=cfg.inner_splits)
    rec = summarize(label or name, y, oof, folds, cfg.n_splits, transform=kind, unit=unit,
                    bootstrap_iterations=cfg.bootstrap_iterations,
                    random_state=cfg.random_state)
    print(f"    {rec}")
    return rec


# ── Physically motivated variants ─────────────────────────────────────────────
def eval_k1_filtered(df, features, cfg):
    """K1 restricted to a physical window."""
    return evaluate(
        df, "anisotropy_k1", "clean_K1_J_m3", "symlog",
        lambda s: s.notna() & (s.abs() >= K1_FLOOR) & (s.abs() <= K1_CEIL),
        features, cfg, unit="J/m^3", label="K1 [physical window]")


def eval_k1_magnitude(df, features, cfg):
    """Magnitude only: log10|K1| within the physical window.

    Reported separately from sign because the two are different physical
    questions -- |K1| is a spin-orbit energy scale, sign is a symmetry outcome --
    and a single symlog R2 conflates them.
    """
    s = pd.to_numeric(df["clean_K1_J_m3"], errors="coerce")
    mask = (s.notna() & (s.abs() >= K1_FLOOR) & (s.abs() <= K1_CEIL)
            & df["reduced_formula"].notna())
    d = df[mask]
    y = np.log10(np.abs(s[mask].values.astype(float)))
    X = prep(d, features)
    groups = d["reduced_formula"].values
    oof, folds, _ = grouped_cv_predict(X, y, groups, make_factories(cfg),
                                       n_splits=cfg.n_splits, inner_splits=cfg.inner_splits)
    rec = summarize("|K1| [log10, physical window]", y, oof, folds, cfg.n_splits,
                    transform="log10_abs", unit="J/m^3",
                    bootstrap_iterations=cfg.bootstrap_iterations,
                    random_state=cfg.random_state)
    print(f"    {rec}")
    return rec


def eval_k1_two_stage(df, features, cfg):
    """Sign classification + magnitude regression, recombined."""
    s = pd.to_numeric(df["clean_K1_J_m3"], errors="coerce")
    mask = s.notna() & (s.abs() >= K1_FLOOR) & (s.abs() <= K1_CEIL) & df["reduced_formula"].notna()
    d = df[mask]
    raw = s[mask].values.astype(float)
    y_sign = (raw > 0).astype(int)
    y_mag = np.log10(np.abs(raw))
    X = prep(d, features)
    groups = d["reduced_formula"].values

    n = cfg.n_estimators_cv
    oof_sign = np.zeros(len(d))
    oof_mag = np.zeros(len(d))
    folds = []
    for tr, va in GroupKFold(n_splits=cfg.n_splits).split(X, y_mag, groups=groups):
        clf_e = ExtraTreesClassifier(n_estimators=n, max_depth=25, min_samples_leaf=2,
                                     max_features=0.85, class_weight="balanced",
                                     random_state=cfg.random_state, n_jobs=-1)
        clf_l = LGBMClassifier(n_estimators=n, max_depth=8, learning_rate=0.04, num_leaves=63,
                               subsample=0.85, colsample_bytree=0.85, class_weight="balanced",
                               random_state=cfg.random_state, n_jobs=-1, verbose=-1)
        clf_e.fit(X.iloc[tr], y_sign[tr]); clf_l.fit(X.iloc[tr], y_sign[tr])
        p = 0.5 * (clf_e.predict_proba(X.iloc[va])[:, 1] + clf_l.predict_proba(X.iloc[va])[:, 1])
        oof_sign[va] = p

        regs = make_factories(cfg)
        preds = []
        for f in regs:
            m = f(); m.fit(X.iloc[tr], y_mag[tr]); preds.append(m.predict(X.iloc[va]))
        oof_mag[va] = np.mean(preds, axis=0)
        folds.append((tr, va))

    sign_pred = np.where(oof_sign >= 0.5, 1.0, -1.0)
    k1_hat = sign_pred * (np.power(10.0, oof_mag))
    y_symlog_true = np.sign(raw) * np.log10(1.0 + np.abs(raw))
    y_symlog_pred = np.sign(k1_hat) * np.log10(1.0 + np.abs(k1_hat))

    rec = summarize("K1 [window + two-stage]", y_symlog_true, y_symlog_pred, folds,
                    cfg.n_splits, transform="symlog", unit="J/m^3",
                    bootstrap_iterations=cfg.bootstrap_iterations,
                    random_state=cfg.random_state)
    sign_f1 = f1_score(y_sign, (oof_sign >= 0.5).astype(int), average="macro")
    mag_r2 = r2_score(y_mag, oof_mag)
    print(f"    {rec}")
    print(f"      sign macro-F1 = {sign_f1:.4f} | log10|K1| R2 = {mag_r2:.4f}")
    out = rec.to_dict()
    out["sign_macro_f1"] = float(sign_f1)
    out["magnitude_log10_r2"] = float(mag_r2)
    return out


def _mean_atomic_mass(formulas):
    """Mean atomic mass per formula, from pymatgen. Average_Weight in the master
    CSV is zero across the entire Ms subset, so it cannot be used here."""
    from pymatgen.core import Composition
    out = np.full(len(formulas), np.nan)
    cache = {}
    for i, f in enumerate(formulas):
        if not isinstance(f, str) or not f:
            continue
        if f not in cache:
            try:
                c = Composition(f)
                cache[f] = c.weight / c.num_atoms
            except Exception:
                cache[f] = np.nan
        out[i] = cache[f]
    return out


def eval_ms_moment(df, features, cfg):
    """Predict moment per atom (Slater-Pauling-linear), convert back to emu/g."""
    s = pd.to_numeric(df["clean_Ms_emu_g"], errors="coerce")
    base = s.notna() & (s > 0) & df["reduced_formula"].notna()
    mbar_all = pd.Series(_mean_atomic_mass(df["reduced_formula"].values), index=df.index)
    mask = base & mbar_all.notna() & (mbar_all > 0)
    d = df[mask]
    sigma = s[mask].values.astype(float)
    mbar = mbar_all[mask].values.astype(float)
    n_mub = sigma * mbar / 5585.0          # muB per atom

    X = prep(d, features)
    groups = d["reduced_formula"].values
    oof_mub, folds, _ = grouped_cv_predict(X, n_mub, groups, make_factories(cfg),
                                           n_splits=cfg.n_splits, inner_splits=cfg.inner_splits)
    sigma_hat = oof_mub * 5585.0 / mbar    # back to emu/g for a fair comparison

    rec = summarize("Ms [via muB/atom]", sigma, sigma_hat, folds, cfg.n_splits,
                    transform="none", unit="emu/g",
                    bootstrap_iterations=cfg.bootstrap_iterations,
                    random_state=cfg.random_state)
    print(f"    {rec}")
    out = rec.to_dict()
    out["moment_space_r2"] = float(r2_score(n_mub, oof_mub))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="final_manuscript")
    ap.add_argument("--out", default=os.path.join(PIPELINE_ROOT, "output/benchmarks"))
    ap.add_argument("--skip-cascade", action="store_true")
    ap.add_argument("--n-splits", type=int, default=None,
                    help="Override fold count (statistical validity is independent of capacity)")
    ap.add_argument("--bootstrap", type=int, default=None, help="Override bootstrap iterations")
    args = ap.parse_args()

    overrides = {}
    if args.n_splits is not None:
        overrides["n_splits"] = args.n_splits
    if args.bootstrap is not None:
        overrides["bootstrap_iterations"] = args.bootstrap
    cfg = RunConfig.preset(args.mode, **overrides)
    print("=" * 78)
    print(f"  EXTENDED TARGET BENCHMARK  |  mode={cfg.mode}  n_splits={cfg.n_splits}  "
          f"n_est={cfg.n_estimators_cv}")
    print("=" * 78)

    df = pd.read_csv(MASTER, low_memory=False)
    groups_json = json.load(open(GROUPS_JSON))

    tiers = dict(TIERS)
    tiers["F5"] = tiers["F3"] + ["mace_quantum", "mp_symmetry",
                                 "crystal_system_dummies"]
    pools = {}
    for tier, names in tiers.items():
        cols = []
        for g in names:
            cols.extend(groups_json.get(g, []))
        pools[tier] = [c for c in dict.fromkeys(cols) if c in df.columns]
        print(f"  pool {tier}: {len(pools[tier])} features")

    results = {"config": cfg.provenance(), "n_master_rows": int(len(df)), "targets": {}}
    t0 = time.time()

    # ── Representation ablation, cascade-free ────────────────────────────────
    for tier in ["F1", "F3", "F5"]:
        print(f"\n--- Representation {tier} (no cascade) ---")
        for name, spec in TARGETS.items():
            rec = evaluate(df, name, spec["col"], spec["transform"], spec["valid"],
                           pools[tier], cfg, unit=spec["unit"], label=f"{name} [{tier}]")
            if rec:
                results["targets"].setdefault(name, {})[tier] = rec.to_dict()

    # ── F5 + out-of-fold cascade ─────────────────────────────────────────────
    if not args.skip_cascade:
        print("\n--- Representation F5 + out-of-fold physics cascade ---")
        df_c = build_oof_cascade(df, pools["F5"], make_factories(cfg),
                                 n_splits=cfg.n_splits, out_of_fold=True)
        casc_pool = pools["F5"] + cascade_feature_names()
        for name, spec in TARGETS.items():
            rec = evaluate(df_c, name, spec["col"], spec["transform"], spec["valid"],
                           casc_pool, cfg, unit=spec["unit"], label=f"{name} [F5+cascade]")
            if rec:
                results["targets"].setdefault(name, {})["F5_cascade_oof"] = rec.to_dict()

    # ── Physics-motivated variants ───────────────────────────────────────────
    print("\n--- Physically motivated reformulations (F5 pool) ---")
    results["variants"] = {}
    r = eval_k1_filtered(df, pools["F5"], cfg)
    if r:
        results["variants"]["k1_physical_window"] = r.to_dict()
    r = eval_k1_magnitude(df, pools["F5"], cfg)
    if r:
        results["variants"]["k1_magnitude_only"] = r.to_dict()
    results["variants"]["k1_window_two_stage"] = eval_k1_two_stage(df, pools["F5"], cfg)
    results["variants"]["ms_via_moment_per_atom"] = eval_ms_moment(df, pools["F5"], cfg)

    results["elapsed_seconds"] = round(time.time() - t0, 1)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "extended_targets_benchmark.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {path}  ({results['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
