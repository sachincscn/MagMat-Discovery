"""
Single source of truth for pipeline run configuration.

Replaces the module-level flags that were previously scattered through
modeling.py (TRAINING_MODE, RUN_ABLATION, HIGH_ACCURACY_TABULAR, ...), where a
value set for a quick debug run silently propagated into the metrics written to
stage3_modeling.json.

Three presets:
  fast_debug        smoke-testing only; small forests, short GNN, no ablation.
  development       intermediate; use while iterating on features.
  final_manuscript  full capacity, real ablation, per-fold statistics.

Any run that writes manuscript-grade numbers must pass `assert_publishable()`,
which refuses to certify metrics produced under a reduced-capacity preset.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import json
import os

VALID_MODES = ("fast_debug", "development", "final_manuscript")


class NonPublishableConfigError(RuntimeError):
    """Raised when reduced-capacity metrics are about to be treated as final."""


@dataclass
class RunConfig:
    """Immutable-by-convention description of how a training run is configured."""

    mode: str = "final_manuscript"

    # ── Cross-validation ────────────────────────────────────────────────────
    n_splits: int = 5
    random_state: int = 42
    # Blend weights are fitted on inner out-of-fold predictions, never on the
    # outer validation labels. See evaluation.fit_blend_weights.
    inner_splits: int = 3

    # ── Representation ablation ─────────────────────────────────────────────
    # When False, only F5 is trained and F1-F4 are NOT reported at all.
    # They are never back-filled from F5.
    run_ablation: bool = True
    ablation_versions: tuple = ("F1", "F3", "F5")

    # ── Cascade construction ────────────────────────────────────────────────
    # Cross-target cascade features (cascade_pred_ms, cascade_pred_hc, ...) must
    # be generated out-of-fold. In-sample cascades inflate downstream targets.
    cascade_out_of_fold: bool = True

    # ── Reporting ───────────────────────────────────────────────────────────
    report_per_fold: bool = True
    bootstrap_iterations: int = 1000

    # ── Feature handling ────────────────────────────────────────────────────
    force_feature_reselect: bool = True
    use_structure_features: bool = True
    variance_threshold: float = 1e-5
    correlation_threshold: float = 0.95

    # ── GNN ─────────────────────────────────────────────────────────────────
    force_gnn_retrain: bool = False

    # ── Plotting ────────────────────────────────────────────────────────────
    heavy_plots: bool = True

    extra: Dict[str, Any] = field(default_factory=dict)

    # ── Capacity, derived from mode ─────────────────────────────────────────
    @property
    def gnn_epochs(self) -> int:
        return {"fast_debug": 10, "development": 15, "final_manuscript": 40}[self.mode]

    @property
    def n_estimators(self) -> int:
        """Forest / booster size for the main tabular committees."""
        return {"fast_debug": 120, "development": 300, "final_manuscript": 600}[self.mode]

    @property
    def n_estimators_cv(self) -> int:
        """Slightly smaller inside CV folds, where many fits are made."""
        return {"fast_debug": 80, "development": 200, "final_manuscript": 400}[self.mode]

    @property
    def n_ensemble_members(self) -> int:
        return {"fast_debug": 2, "development": 4, "final_manuscript": 8}[self.mode]

    @property
    def is_publishable(self) -> bool:
        """Whether metrics from this configuration may be quoted as final."""
        return (
            self.mode == "final_manuscript"
            and self.cascade_out_of_fold
            and self.report_per_fold
            and self.n_splits >= 5
        )

    # ── Guards & serialisation ──────────────────────────────────────────────
    def assert_publishable(self) -> None:
        if self.is_publishable:
            return
        reasons = []
        if self.mode != "final_manuscript":
            reasons.append(f"mode={self.mode!r} (reduced capacity)")
        if not self.cascade_out_of_fold:
            reasons.append("cascade_out_of_fold=False (cross-target leakage)")
        if not self.report_per_fold:
            reasons.append("report_per_fold=False (no fold variance)")
        if self.n_splits < 5:
            reasons.append(f"n_splits={self.n_splits} (<5)")
        raise NonPublishableConfigError(
            "Refusing to certify these metrics as manuscript-grade: "
            + "; ".join(reasons)
            + ". Use RunConfig.preset('final_manuscript')."
        )

    def provenance(self) -> Dict[str, Any]:
        """Block embedded in every metrics file so numbers carry their config."""
        d = asdict(self)
        d.update(
            gnn_epochs=self.gnn_epochs,
            n_estimators=self.n_estimators,
            n_estimators_cv=self.n_estimators_cv,
            n_ensemble_members=self.n_ensemble_members,
            is_publishable=self.is_publishable,
        )
        d["ablation_versions"] = list(self.ablation_versions)
        return d

    @classmethod
    def preset(cls, mode: str, **overrides) -> "RunConfig":
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        base: Dict[str, Any] = {"mode": mode}
        if mode == "fast_debug":
            base.update(
                n_splits=3, run_ablation=False, heavy_plots=False,
                report_per_fold=True, bootstrap_iterations=0,
            )
        elif mode == "development":
            base.update(n_splits=3, run_ablation=False, heavy_plots=False)
        base.update(overrides)
        return cls(**base)

    @classmethod
    def from_env(cls) -> "RunConfig":
        """Read MAGPIPE_MODE so a run's capacity is explicit at the call site."""
        return cls.preset(os.environ.get("MAGPIPE_MODE", "final_manuscript"))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.provenance(), fh, indent=2)


# Default used when a module is imported without an explicit config.
# Deliberately the full-capacity preset: a debug run must be opted into.
DEFAULT_CONFIG = RunConfig.preset("final_manuscript")


# ──────────────────────────────────────────────────────────────────────────────
#  Forbidden feature columns
# ──────────────────────────────────────────────────────────────────────────────
# features_master.csv still carries the raw, pre-cleaning target columns
# alongside their clean_* counterparts, at Spearman rho = 1.0:
#
#   Saturation_Magnetization_emu_g  <-> clean_Ms_emu_g
#   Coercivity_A_m                  <-> clean_Hc_A_m
#   K1_J_m3                         <-> clean_K1_J_m3
#   Hard_Magnet                     <-  thresholded coercivity (rho = 0.82)
#
# The curated feature_groups.json does not reference them, so the pipeline as
# written is not affected. But any code path that selects features by dtype
# (`df.select_dtypes("number")`) picks them up and yields R2 ~ 0.99 -- a trap
# that is easy to fall into and hard to notice, because the result looks like a
# triumph rather than a bug. Selection helpers screen against this list.
FORBIDDEN_FEATURE_COLUMNS = frozenset({
    # Raw pre-cleaning duplicates of regression targets
    "Saturation_Magnetization_emu_g", "Coercivity_A_m", "K1_J_m3",
    "BH_Max", "BH_max_kJ_m3", "Anisotropy_Field", "Curie_Weiss_Theta",
    "Theta_p_K", "Magnetic_Moment_emu_g",
    # Direct target columns
    "clean_Ms_emu_g", "clean_Hc_A_m", "clean_K1_J_m3",
    "clean_BH_max_kJ_m3", "clean_theta_p_K",
    "Mean_TC_K", "Mean_TN_K", "Type",
    # Thresholded / derived views of a target
    "Hard_Magnet", "Soft_Magnet",
})


def screen_features(columns, context: str = "", verbose: bool = True):
    """Drop any forbidden target-derived column from a candidate feature list."""
    kept, dropped = [], []
    for c in columns:
        (dropped if c in FORBIDDEN_FEATURE_COLUMNS else kept).append(c)
    if dropped and verbose:
        where = f" in {context}" if context else ""
        print(f"  [leakage guard] dropped {len(dropped)} target-derived column(s)"
              f"{where}: {sorted(dropped)}")
    return kept
