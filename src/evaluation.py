"""
Honest cross-validation and metric reporting.

Two defects in the previous evaluation path motivated this module:

1. Blend weights were obtained with ``nnls(P_val, y_val)`` and then applied to
   that same validation fold, so the meta-weights saw the labels they were
   scored against. Measured inflation was small (~+0.004 R2 on (BH)max) but it
   is not defensible in review. ``fit_blend_weights`` fits on inner
   out-of-fold predictions from the training fold only.

2. A single pooled R2 was reported with no fold variance. For the smaller
   targets ((BH)max, n=1379) a pooled point estimate hides most of the
   uncertainty. ``summarize`` reports per-fold mean +/- std and a bootstrap CI
   alongside the pooled value.
"""

from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Sequence, Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error


# ──────────────────────────────────────────────────────────────────────────────
#  Metric records
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class MetricRecord:
    """Everything needed to quote a number in a paper, including its spread."""

    target: str
    n_samples: int
    n_splits: int
    transform: str = "none"
    unit: str = ""

    pooled_r2: float = float("nan")
    pooled_mae: float = float("nan")

    fold_r2: List[float] = field(default_factory=list)
    fold_mae: List[float] = field(default_factory=list)

    r2_ci95: Optional[List[float]] = None
    mae_ci95: Optional[List[float]] = None

    @property
    def r2_mean(self) -> float:
        return float(np.mean(self.fold_r2)) if self.fold_r2 else float("nan")

    @property
    def r2_std(self) -> float:
        return float(np.std(self.fold_r2, ddof=1)) if len(self.fold_r2) > 1 else 0.0

    @property
    def mae_mean(self) -> float:
        return float(np.mean(self.fold_mae)) if self.fold_mae else float("nan")

    @property
    def mae_std(self) -> float:
        return float(np.std(self.fold_mae, ddof=1)) if len(self.fold_mae) > 1 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(
            r2_mean=self.r2_mean, r2_std=self.r2_std,
            mae_mean=self.mae_mean, mae_std=self.mae_std,
        )
        return d

    def __str__(self) -> str:
        ci = ""
        if self.r2_ci95:
            ci = f"  95% CI [{self.r2_ci95[0]:.3f}, {self.r2_ci95[1]:.3f}]"
        return (
            f"{self.target:<28} n={self.n_samples:<6} "
            f"R2 = {self.pooled_r2:.4f} (folds {self.r2_mean:.4f} +/- {self.r2_std:.4f}){ci}  "
            f"MAE = {self.pooled_mae:.4f} +/- {self.mae_std:.4f}"
        )


def bootstrap_ci(y_true, y_pred, metric: Callable, n_iter: int = 1000,
                 alpha: float = 0.05, random_state: int = 42):
    """Percentile bootstrap CI over paired (truth, prediction) samples."""
    if n_iter <= 0:
        return None
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    stats = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        # A resample can be degenerate for R2; skip those draws.
        if np.ptp(y_true[idx]) < 1e-12:
            stats[i] = np.nan
            continue
        stats[i] = metric(y_true[idx], y_pred[idx])
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return None
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def summarize(target: str, y_true, oof_pred, fold_indices, n_splits: int,
              transform: str = "none", unit: str = "",
              bootstrap_iterations: int = 1000,
              random_state: int = 42) -> MetricRecord:
    """Build a MetricRecord from OOF predictions and the folds that made them."""
    y_true = np.asarray(y_true, dtype=float)
    oof_pred = np.asarray(oof_pred, dtype=float)

    rec = MetricRecord(
        target=target, n_samples=int(len(y_true)), n_splits=n_splits,
        transform=transform, unit=unit,
        pooled_r2=float(r2_score(y_true, oof_pred)),
        pooled_mae=float(mean_absolute_error(y_true, oof_pred)),
    )
    for _, va in fold_indices:
        if len(va) < 2 or np.ptp(y_true[va]) < 1e-12:
            continue
        rec.fold_r2.append(float(r2_score(y_true[va], oof_pred[va])))
        rec.fold_mae.append(float(mean_absolute_error(y_true[va], oof_pred[va])))

    rec.r2_ci95 = bootstrap_ci(y_true, oof_pred, r2_score,
                               bootstrap_iterations, random_state=random_state)
    rec.mae_ci95 = bootstrap_ci(y_true, oof_pred, mean_absolute_error,
                                bootstrap_iterations, random_state=random_state)
    return rec


# ──────────────────────────────────────────────────────────────────────────────
#  Leak-free ensemble blending
# ──────────────────────────────────────────────────────────────────────────────
def fit_blend_weights(models: Sequence, X_train: pd.DataFrame, y_train: np.ndarray,
                      groups: Optional[np.ndarray] = None, inner_splits: int = 3,
                      fallback: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Non-negative blend weights fitted on *inner* out-of-fold predictions.

    ``models`` is a sequence of zero-argument factories, so a fresh unfitted
    estimator is built for every inner fold. Weights are normalised to sum to 1;
    if NNLS degenerates, ``fallback`` (default: uniform) is returned.
    """
    n_models = len(models)
    if fallback is None:
        fallback = np.full(n_models, 1.0 / n_models)

    n = len(X_train)
    if n < inner_splits * 2:
        return fallback

    P_inner = np.zeros((n, n_models))
    if groups is not None:
        splitter = GroupKFold(n_splits=inner_splits).split(X_train, y_train, groups=groups)
    else:
        from sklearn.model_selection import KFold
        splitter = KFold(n_splits=inner_splits, shuffle=True, random_state=42).split(X_train)

    for itr, iva in splitter:
        for j, factory in enumerate(models):
            est = factory()
            est.fit(X_train.iloc[itr], y_train[itr])
            P_inner[iva, j] = est.predict(X_train.iloc[iva])

    w, _ = nnls(P_inner, y_train)
    if w.sum() <= 1e-6:
        return fallback
    return w / w.sum()


def grouped_cv_predict(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                       model_factories: Sequence[Callable], n_splits: int = 5,
                       inner_splits: int = 3, verbose: bool = False):
    """
    StratifiedGroupKFold OOF predictions from a committee blended without label leakage.
    Stratifies by target quantile bins for balanced fold distributions.

    Returns ``(oof_pred, fold_indices, fold_weights)``.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    oof = np.zeros(len(X))
    fold_indices = []
    fold_weights = []

    # Stratify by target quantile bins
    try:
        y_bins = pd.qcut(y, q=min(n_splits, 5), labels=False, duplicates='drop')
    except Exception:
        y_bins = np.zeros(len(y), dtype=int)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for k, (tr, va) in enumerate(sgkf.split(X, y_bins, groups=groups)):
        X_tr, y_tr = X.iloc[tr], y[tr]
        X_va = X.iloc[va]

        w = fit_blend_weights(model_factories, X_tr, y_tr,
                              groups=np.asarray(groups)[tr], inner_splits=inner_splits)

        P_va = np.zeros((len(va), len(model_factories)))
        for j, factory in enumerate(model_factories):
            est = factory()
            est.fit(X_tr, y_tr)
            P_va[:, j] = est.predict(X_va)

        oof[va] = P_va @ w
        fold_indices.append((tr, va))
        fold_weights.append(w.tolist())
        if verbose:
            print(f"    fold {k + 1}/{n_splits}: weights={np.round(w, 3).tolist()} "
                  f"R2={r2_score(y[va], oof[va]):.4f}")

    return oof, fold_indices, fold_weights
