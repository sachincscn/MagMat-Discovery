"""
Out-of-fold construction of the cross-target physics cascade.

The extended targets are deliberately coupled: coercivity is predicted with the
help of a predicted Ms and K1, and (BH)max with the help of a predicted Hc and
Ms, because the governing relations (Kronmueller nucleation, the Stoner-Wohlfarth
demagnetisation envelope) are written in those variables. The coupling is sound
physics; the previous implementation just built it unsafely.

Previously each donor model was fitted on its *entire* dataset and its in-sample
predictions were injected as mandatory features into the next target's
cross-validation. A compound sitting in (BH)max's validation fold that also
carried an experimental Ms therefore received a near-oracle Ms, and the PINN
bounds -- analytic functions of Ms and Hc -- inherited it. Measured inflation on
(BH)max was +0.011 R2.

Here every donor prediction consumed by a downstream target is out-of-fold with
respect to that compound's own label. Rows that are not donors for a given
property (no experimental value) are filled from a full-fit donor model, which
introduces no leakage because their labels never entered any fit.
"""

from typing import Dict, List, Optional, Sequence, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    from src.pinn_hysteresis import compute_pinn_hysteresis_features
except ImportError:  # direct module execution
    from pinn_hysteresis import compute_pinn_hysteresis_features


# Donor properties, in the order the physics requires them.
DONORS = [
    {
        "name": "k1",
        "column": "clean_K1_J_m3",
        "forward": lambda v: np.sign(v) * np.log10(1.0 + np.abs(v)),
        "inverse": lambda v: np.sign(v) * (np.power(10.0, np.abs(v)) - 1.0),
        "clip": (-1e8, 1e8),
        "valid": lambda s: s.notna(),
    },
    {
        "name": "ms",
        "column": "clean_Ms_emu_g",
        "forward": lambda v: v,
        "inverse": lambda v: v,
        "clip": (0.0, 350.0),
        "valid": lambda s: s.notna() & (s > 0),
    },
    {
        "name": "hc",
        "column": "clean_Hc_A_m",
        "forward": np.log10,
        "inverse": lambda v: np.power(10.0, v),
        "clip": (0.0, 1e8),
        "valid": lambda s: s.notna() & (s > 0) & (s <= 1e8),
    },
]


def _prepare(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    X = df.loc[:, ~df.columns.duplicated()][cols].apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.median()).fillna(0.0)


def _donor_oof(df: pd.DataFrame, spec: Dict, base_features: Sequence[str],
               model_factories: Sequence[Callable], n_splits: int,
               group_column: str, verbose: bool) -> Optional[np.ndarray]:
    """OOF predictions on donor rows, full-fit predictions elsewhere."""
    col = spec["column"]
    if col not in df.columns:
        return None

    series = pd.to_numeric(df[col], errors="coerce")
    mask = spec["valid"](series) & df[group_column].notna()
    if mask.sum() < 50:
        return None

    donors = df[mask]
    y = spec["forward"](series[mask].values.astype(float))
    groups = donors[group_column].values

    X_donor = _prepare(donors, base_features)
    X_all = _prepare(df, X_donor.columns)

    def blend(X_tr, y_tr, X_target):
        preds = []
        for factory in model_factories:
            est = factory()
            est.fit(X_tr, y_tr)
            preds.append(est.predict(X_target))
        return np.mean(preds, axis=0)

    # Out-of-fold over donor rows.
    oof = np.zeros(len(donors))
    for tr, va in GroupKFold(n_splits=n_splits).split(X_donor, y, groups=groups):
        oof[va] = blend(X_donor.iloc[tr], y[tr], X_donor.iloc[va])

    # Full fit only to reach non-donor rows.
    out = blend(X_donor, y, X_all)
    out[np.where(mask.values)[0]] = oof

    lo, hi = spec["clip"]
    out = np.clip(spec["inverse"](out), lo, hi)
    if verbose:
        print(f"    cascade donor {spec['name']:<4} n_donor={int(mask.sum()):<6} "
              f"range=[{np.nanmin(out):.3g}, {np.nanmax(out):.3g}]")
    return out


def build_oof_cascade(df: pd.DataFrame, base_features: Sequence[str],
                      model_factories: Sequence[Callable], n_splits: int = 5,
                      group_column: str = "reduced_formula",
                      out_of_fold: bool = True,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Return ``df`` with cascade_* / pinn_* columns attached.

    Set ``out_of_fold=False`` only to reproduce the historical (leaky) behaviour
    for comparison; it must never be used for reported metrics.
    """
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if verbose:
        mode = "out-of-fold" if out_of_fold else "IN-SAMPLE (leaky, comparison only)"
        print(f"  Building physics cascade [{mode}] ...")

    for spec in DONORS:
        if out_of_fold:
            values = _donor_oof(df, spec, base_features, model_factories,
                                n_splits, group_column, verbose)
        else:
            values = _donor_insample(df, spec, base_features, model_factories, group_column)
        if values is None:
            continue
        df[f"cascade_pred_{spec['name']}"] = values

    # Derived couplings, mirroring the relations used downstream.
    if "cascade_pred_k1" in df:
        k1 = df["cascade_pred_k1"].values
        df["cascade_pred_k1_symlog"] = np.sign(k1) * np.log10(1.0 + np.abs(k1))
    if "cascade_pred_hc" in df:
        df["cascade_pred_hc_log"] = np.log10(np.maximum(df["cascade_pred_hc"].values, 1.0))
    if "cascade_pred_ms" in df:
        ms = np.maximum(df["cascade_pred_ms"].values, 0.1)
        df["cascade_theoretical_bh_max"] = 2.0 * np.log10(ms)
        if "cascade_pred_hc_log" in df:
            df["cascade_extrinsic_bh_product"] = df["cascade_pred_hc_log"].values + np.log10(ms)
        if "cascade_pred_k1" in df:
            k1_pos = np.maximum(df["cascade_pred_k1"].values, 0.0)
            df["cascade_kronmuller_hk"] = np.log10(1.0 + k1_pos) - np.log10(ms)

    # PINN hysteresis bounds recomputed from the (now safe) cascade.
    pinn = compute_pinn_hysteresis_features(df, use_cascade_preds=True)
    for c in pinn.columns:
        df[c] = pinn[c]

    return df


def _donor_insample(df, spec, base_features, model_factories, group_column):
    """Historical behaviour: fit on everything, predict everything. Leaky."""
    col = spec["column"]
    if col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors="coerce")
    mask = spec["valid"](series) & df[group_column].notna()
    if mask.sum() < 50:
        return None
    donors = df[mask]
    y = spec["forward"](series[mask].values.astype(float))
    X_donor = _prepare(donors, base_features)
    X_all = _prepare(df, X_donor.columns)
    preds = []
    for factory in model_factories:
        est = factory()
        est.fit(X_donor, y)
        preds.append(est.predict(X_all))
    lo, hi = spec["clip"]
    return np.clip(spec["inverse"](np.mean(preds, axis=0)), lo, hi)


def cascade_feature_names() -> List[str]:
    """Names produced by build_oof_cascade, for feature-group registration."""
    return [
        "cascade_pred_k1", "cascade_pred_k1_symlog", "cascade_pred_ms",
        "cascade_pred_hc", "cascade_pred_hc_log", "cascade_theoretical_bh_max",
        "cascade_extrinsic_bh_product", "cascade_kronmuller_hk",
        "pinn_loop_squareness", "pinn_bh_max_ideal", "pinn_bh_max_extrinsic",
        "pinn_bh_max_bound", "pinn_log_bh_bound", "pinn_kronmuller_hk",
        "pinn_kronmuller_hc_log", "pinn_kronmuller_margin", "pinn_hardness_parameter",
    ]
