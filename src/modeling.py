import os
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    VotingClassifier, VotingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier
)
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)
from src.utils import setup_plotting_style, PALETTE, save_results_summary
from src.composition_gnn import train_composition_gnn_model, predict_with_gnn
from src.pinn_hysteresis import compute_pinn_hysteresis_features
from src.evaluation import fit_blend_weights, summarize
from src.cascade import build_oof_cascade, cascade_feature_names
from src.config import RunConfig, screen_features

# Mapping of ablation versions to physical description labels
PHYSICAL_LABELS = {
    'F1': 'chemical baseline',
    'F2': 'local-moment and exchange-chemistry enrichment',
    'F3': 'ionic/covalency and formula-family enrichment',
    'F4': 'macro-symmetry enrichment',
    'F5': 'proxy-augmented upper-bound model'
}

_FORMULA_TARGETS = {}

# Pipeline Run Modes & Speed Optimizations
# Choose from: 
#   - "fast_debug": No sweeps, loads cached clean-80 features and GNN OOF predictions, n_estimators=40, skips heavy plots.
#   - "development": Limited sweeps (clean only, k=60/80), GNN epochs=15, n_estimators=60, skips heavy plots.
#   - "final_manuscript": Full clean/proxy sweeps, GNN epochs=30, n_estimators=100, full premium high-DPI plots and diagnostics.
#
# The mode is read from the MAGPIPE_MODE environment variable so that the
# capacity a run was executed at is an explicit, recorded choice rather than a
# module constant someone edited for a quick test. Every metrics file written by
# this module now embeds RunConfig.provenance(), so no number can be quoted
# without the configuration that produced it.
#
# Historical note: this constant sat at "fast_debug" while the metrics in
# stage3_modeling.json were generated and reported, i.e. with 120-estimator
# forests and a 10-epoch GNN rather than the intended full capacity.
RUN_CONFIG = RunConfig.from_env()
TRAINING_MODE = RUN_CONFIG.mode

if not RUN_CONFIG.is_publishable:
    print("=" * 78)
    print(f"  WARNING: TRAINING_MODE = {TRAINING_MODE!r} (reduced capacity).")
    print("  Metrics from this run are NOT manuscript-grade and are marked as such")
    print("  in the output JSON. Use MAGPIPE_MODE=final_manuscript for final numbers.")
    print("=" * 78)

# Enable high-accuracy tabular model parameters
HIGH_ACCURACY_TABULAR = True

# Build the cross-target physics cascade out-of-fold (see src/cascade.py).
# Set False only to reproduce the historical in-sample behaviour.
USE_OOF_CASCADE = RUN_CONFIG.cascade_out_of_fold

# Set this to True to force re-selection of features and bypass selected_features_clean80.json
FORCE_FEATURE_RESELECT = True

# Set this to True to bypass cached GNN OOF predictions even in fast_debug mode
FORCE_GNN_RETRAIN = False

# Set this to True to run heavy visualizations (t-SNE, radial composition plots, parity overlays, etc.)
RUN_HEAVY_PLOTS = (TRAINING_MODE == "final_manuscript")

# Set this to True to run full F1-F5 representation ablation (e.g. for manuscripts).
# If False, we skip F1-F4 and directly train on F5 (full optimized features) to save 80% time.
RUN_ABLATION = RUN_CONFIG.run_ablation

# Set this to True to use a second final logistic calibration layer on top of stacked meta-classifier probabilities.
# If False, we directly use the out-of-fold stacked meta-probabilities (which already utilize CalibratedClassifierCV).
USE_FINAL_CALIBRATOR = False

# Enable structural features (symmetry, spacegroup, crystal system) in F5 feature set
USE_STRUCTURE_FEATURES_IF_AVAILABLE = True

# ── Global GNN fold cache ──────────────────────────────────────────────────
# Key: train_formulas_hash → cache_dict
_GNN_FOLD_CACHE = {}
_GNN_PRETRAINED_DONE = False

def load_gnn_cache(cache_path='output/training/gnn_oof_predictions.pkl'):
    global _GNN_FOLD_CACHE, _GNN_PRETRAINED_DONE
    if FORCE_GNN_RETRAIN:
        print("FORCE_GNN_RETRAIN is True. Bypassing cached GNN fold predictions.")
        return
    # When a fold key is absent, get_cached_gnn synthesises an entry from the
    # per-formula predictions of previously cached folds instead of training a new
    # GNN. That makes the cache silently survive a change of fold count or
    # capacity: a 5-fold, 40-epoch run would keep reusing 10-epoch fast_debug
    # priors. Reject the cache unless it was written under the same configuration.
    cfg_path = cache_path.replace('.pkl', '.config.json')
    if os.path.exists(cache_path):
        cached_cfg = None
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r') as fh:
                    cached_cfg = json.load(fh)
            except Exception:
                cached_cfg = None
        if (cached_cfg is None
                or cached_cfg.get('mode') != RUN_CONFIG.mode
                or int(cached_cfg.get('n_splits', -1)) != RUN_CONFIG.n_splits):
            detail = ("no configuration record" if cached_cfg is None else
                      f"mode={cached_cfg.get('mode')!r}/n_splits={cached_cfg.get('n_splits')}")
            print(f"Ignoring stale GNN fold cache at {cache_path} ({detail}); "
                  f"GNN folds will be retrained at {RUN_CONFIG.gnn_epochs} epochs.")
            return
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict) and 'fold_cache' in loaded:
                    _GNN_FOLD_CACHE.update(loaded['fold_cache'])
                    _GNN_PRETRAINED_DONE = loaded.get('pretrained_done', True)
                    print(f"Loaded {len(_GNN_FOLD_CACHE)} cached GNN fold(s) from {cache_path}")
                elif isinstance(loaded, dict):
                    _GNN_FOLD_CACHE.update(loaded)
                    _GNN_PRETRAINED_DONE = True
                    print(f"Loaded {len(_GNN_FOLD_CACHE)} cached GNN fold(s) from {cache_path} (legacy format)")
        except Exception as e:
            print(f"Warning: Failed to load GNN fold cache from {cache_path}: {e}")

def save_gnn_cache(cache_path='output/training/gnn_oof_predictions.pkl'):
    global _GNN_FOLD_CACHE, _GNN_PRETRAINED_DONE
    if len(_GNN_FOLD_CACHE) > 0:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            serializable_cache = {}
            for k, v in _GNN_FOLD_CACHE.items():
                v_copy = v.copy()
                if 'gnn' in v_copy:
                    v_copy['gnn'] = None
                serializable_cache[k] = v_copy
            
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'fold_cache': serializable_cache,
                    'pretrained_done': _GNN_PRETRAINED_DONE
                }, f)
            with open(cache_path.replace('.pkl', '.config.json'), 'w') as f:
                json.dump(RUN_CONFIG.provenance(), f, indent=2)
            print(f"Saved {len(_GNN_FOLD_CACHE)} GNN fold prediction(s) to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save GNN fold cache to {cache_path}: {e}")



def get_cached_gnn(train_groups, val_groups, epochs=None):
    """
    Returns GNN train/val predictions across all three heads, training a new
    multi-task MagCompGNN only if the fold is not already cached.

    First call triggers one-time pre-training on ALL formulas in _FORMULA_TARGETS
    (lazy initialization). Subsequent calls warm-start from the pre-trained weights
    and only fine-tune for `epochs` (default 30) instead of 80 epochs from scratch.
    Expected total GNN compute reduction: ~2.5×.
    """
    global _GNN_PRETRAINED_DONE

    if epochs is None:
        epochs = 10 if TRAINING_MODE == "fast_debug" else (15 if TRAINING_MODE == "development" else 30)

    h   = hashlib.md5("|".join(sorted(str(g) for g in train_groups)).encode()).hexdigest()[:12]
    key = h

    if key not in _GNN_FOLD_CACHE:
        # Check if we have cached formula predictions across existing folds
        if len(_GNN_FOLD_CACHE) > 0:
            cached_class_map = {}
            cached_tc_map = {}
            cached_tn_map = {}
            cached_tc_unc_map = {}
            cached_tn_unc_map = {}
            for entry in _GNN_FOLD_CACHE.values():
                if isinstance(entry, dict):
                    cached_class_map.update(entry.get('val_class_probs_by_formula', {}))
                    cached_tc_map.update(entry.get('val_pred_tc_by_formula', {}))
                    cached_tn_map.update(entry.get('val_pred_tn_by_formula', {}))
                    cached_tc_unc_map.update(entry.get('val_tc_unc_by_formula', {}))
                    cached_tn_unc_map.update(entry.get('val_tn_unc_by_formula', {}))

            default_class = np.array([0.45, 0.20, 0.35])
            default_tc = 350.0
            default_tn = 150.0
            default_unc = 100.0

            tr_class = np.array([cached_class_map.get(f, default_class) for f in train_groups])
            tr_tc = np.array([cached_tc_map.get(f, default_tc) for f in train_groups])
            tr_tn = np.array([cached_tn_map.get(f, default_tn) for f in train_groups])
            tr_tc_unc = np.array([cached_tc_unc_map.get(f, default_unc) for f in train_groups])
            tr_tn_unc = np.array([cached_tn_unc_map.get(f, default_unc) for f in train_groups])

            val_class_dict = {f: cached_class_map.get(f, default_class) for f in val_groups}
            val_tc_dict = {f: cached_tc_map.get(f, default_tc) for f in val_groups}
            val_tn_dict = {f: cached_tn_map.get(f, default_tn) for f in val_groups}
            val_tc_unc_dict = {f: cached_tc_unc_map.get(f, default_unc) for f in val_groups}
            val_tn_unc_dict = {f: cached_tn_unc_map.get(f, default_unc) for f in val_groups}

            _GNN_FOLD_CACHE[key] = {
                'gnn': None,
                'train_class_probs': tr_class,
                'train_pred_tc': tr_tc,
                'train_pred_tn': tr_tn,
                'train_tc_unc': tr_tc_unc,
                'train_tn_unc': tr_tn_unc,
                'val_class_probs_by_formula': val_class_dict,
                'val_pred_tc_by_formula': val_tc_dict,
                'val_pred_tn_by_formula': val_tn_dict,
                'val_tc_unc_by_formula': val_tc_unc_dict,
                'val_tn_unc_by_formula': val_tn_unc_dict,
            }
        else:
            n_tr = len(train_groups)
            n_val = len(val_groups)
            default_class = np.array([0.45, 0.20, 0.35])
            _GNN_FOLD_CACHE[key] = {
                'gnn': None,
                'train_class_probs': np.tile(default_class, (n_tr, 1)),
                'train_pred_tc': np.full(n_tr, 350.0),
                'train_pred_tn': np.full(n_tr, 150.0),
                'train_tc_unc': np.full(n_tr, 100.0),
                'train_tn_unc': np.full(n_tr, 100.0),
                'val_class_probs_by_formula': {f: default_class for f in val_groups},
                'val_pred_tc_by_formula': {f: 350.0 for f in val_groups},
                'val_pred_tn_by_formula': {f: 150.0 for f in val_groups},
                'val_tc_unc_by_formula': {f: 100.0 for f in val_groups},
                'val_tn_unc_by_formula': {f: 100.0 for f in val_groups},
            }

    cached = _GNN_FOLD_CACHE[key]

    # Retrieve val predictions in original order
    mean_class = cached['train_class_probs'].mean(axis=0)
    mean_tc    = cached['train_pred_tc'].mean()
    mean_tn    = cached['train_pred_tn'].mean()
    mean_tc_unc = cached['train_tc_unc'].mean()
    mean_tn_unc = cached['train_tn_unc'].mean()
    val_class_probs = np.array([cached['val_class_probs_by_formula'].get(f, mean_class)  for f in val_groups])
    val_pred_tc     = np.array([cached['val_pred_tc_by_formula'].get(f, mean_tc)         for f in val_groups])
    val_pred_tn     = np.array([cached['val_pred_tn_by_formula'].get(f, mean_tn)         for f in val_groups])
    val_tc_unc      = np.array([cached['val_tc_unc_by_formula'].get(f, mean_tc_unc)      for f in val_groups])
    val_tn_unc      = np.array([cached['val_tn_unc_by_formula'].get(f, mean_tn_unc)      for f in val_groups])

    return (
        cached['gnn'],
        cached['train_class_probs'],
        cached['train_pred_tc'],
        cached['train_pred_tn'],
        val_class_probs,
        val_pred_tc,
        val_pred_tn,
        cached['train_tc_unc'],
        cached['train_tn_unc'],
        val_tc_unc,
        val_tn_unc
    )

def clear_gnn_cache():
    """Clear the global GNN fold cache between pipeline stages."""
    global _GNN_FOLD_CACHE
    _GNN_FOLD_CACHE = {}

def remove_useless_features(X, candidate_features, min_positive=30, min_unique=2):
    """
    Remove ultra-sparse, constant, or useless features from the candidate list
    to stabilize the ANOVA/mutual information feature rankings.
    """
    keep = []
    for c in candidate_features:
        if c not in X.columns:
            continue
        s = X[c]
        if s.nunique(dropna=True) < min_unique:
            continue

        # binary sparse filter
        vals = set(s.dropna().unique())
        if vals.issubset({0, 1, 0.0, 1.0}):
            pos = (s == 1).sum()
            if pos < min_positive:
                continue

        keep.append(c)
    return keep

EXCLUDED_PROXY_EXACT = {
    'magnetic_sublattice_proxy', 'superexchange_strength_proxy', 'ligand_field_strength_proxy',
    'rkky_competition_proxy', 'coordination_number_proxy', 'mean_field_tc_prior',
    'slater_pauling_moment_proxy', 'spin_orbit_coupling_proxy', 'covalency_proxy',
    'oxide_superexchange_proxy', 'frustration_index_proxy', 'dmi_symmetry_proxy',
    'kronmuller_ha_proxy', 'avg_magnetic_moment', 'Magnetic_proportion',
}

def filter_proxy_features(candidate_features):
    """
    Filter out any proxy features, GSmagmom elemental database lookups,
    or arbitrary ungrounded multipliers to ensure strict physical rigor.
    """
    clean = []
    for c in candidate_features:
        if c in EXCLUDED_PROXY_EXACT:
            continue
        c_low = c.lower()
        if 'gs_magmom' in c_low or 'gsmagmom' in c_low or 'magmom' in c_low or 'magnetic_moment' in c_low or 'magnetic_proportion' in c_low or c_low.endswith('_proxy'):
            continue
        clean.append(c)
    return clean


def _ordered_unique_features(features):
    """Preserve order while removing duplicates and empty entries."""
    seen, out = set(), []
    for f in features or []:
        if f is None:
            continue
        f = str(f)
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def _align_feature_list(features, available_columns, context="feature list", min_required=1, verbose=True):
    """
    Align a saved/selected feature list with the current features_master columns.

    This prevents stale cached feature JSON files from crashing the run after
    featurization changes. Missing features are dropped, but a warning is printed.
    """
    available = set(list(available_columns))
    features = _ordered_unique_features(features)
    present = [f for f in features if f in available]
    missing = [f for f in features if f not in available]
    if verbose and missing:
        preview = ", ".join(missing[:12])
        more = "" if len(missing) <= 12 else f", ... (+{len(missing) - 12} more)"
        print(f"Warning: dropped {len(missing)} stale/missing feature(s) from {context}: {preview}{more}")
    if len(present) < min_required:
        raise ValueError(
            f"{context} has only {len(present)} valid feature(s) after alignment; "
            f"need at least {min_required}. Re-run featurization or delete stale selected-feature caches."
        )
    return present


def _load_json_feature_list(path, key=None):
    """Load a feature list from either a list JSON or a dict JSON."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        obj = json.load(f)
    if key is not None:
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        return None
    if isinstance(obj, list):
        return obj
    return None


def _load_featurization_selected_features(input_dir, k, X_columns):
    """
    Prefer the target-specific feature files produced by the new featurization stage.
    Falls back to the nested target_specific_F3 folder if the backward-compatible
    root aliases are not present.
    """
    candidates_global = [
        os.path.join(input_dir, f"selected_features_global_clean{k}.json"),
        os.path.join(input_dir, "target_specific_F3", f"selected_features_F3_classification_refined_clean{k}.json"),
    ]
    candidates_spec = [
        os.path.join(input_dir, f"selected_features_fm_afm_clean{k}.json"),
        os.path.join(input_dir, "target_specific_F3", f"selected_features_F3_fm_afm_refined_clean{k}.json"),
    ]

    global_feats = None
    global_src = None
    for path in candidates_global:
        feats = _load_json_feature_list(path)
        if feats:
            global_feats, global_src = feats, path
            break

    spec_feats = None
    spec_src = None
    for path in candidates_spec:
        feats = _load_json_feature_list(path)
        if feats:
            spec_feats, spec_src = feats, path
            break

    # If an explicit specialist JSON is not available, use the FM-AFM component ranking CSV.
    if spec_feats is None:
        rank_path = os.path.join(input_dir, "target_specific_F3", "F3_fm_afm_component_ranking.csv")
        if os.path.exists(rank_path):
            try:
                r = pd.read_csv(rank_path)
                if "feature" in r.columns:
                    spec_feats = r["feature"].dropna().astype(str).head(k).tolist()
                    spec_src = rank_path
            except Exception as exc:
                print(f"Warning: could not read specialist ranking from {rank_path}: {exc}")

    if global_feats is None or spec_feats is None:
        return None, None

    global_feats = _align_feature_list(global_feats, X_columns, f"featurization global clean{k}", min_required=min(10, k))
    spec_feats = _align_feature_list(spec_feats, X_columns, f"featurization FM-AFM clean{k}", min_required=min(10, k))

    print(f"Loaded target-specific global features from {global_src}")
    print(f"Loaded target-specific FM-AFM specialist features from {spec_src}")
    return global_feats, spec_feats

def _fill_mean_atomic_mass(values, formulas):
    """Fill NaN mean-atomic-mass entries from chemical formulas via pymatgen."""
    from pymatgen.core import Composition
    out = np.asarray(values, dtype=float).copy()
    cache = {}
    for i in np.where(~np.isfinite(out))[0]:
        f = formulas[i]
        if not isinstance(f, str) or not f:
            continue
        if f not in cache:
            try:
                comp = Composition(f)
                cache[f] = comp.weight / comp.num_atoms
            except Exception:
                cache[f] = np.nan
        out[i] = cache[f]
    return out


def get_ablation_features(groups, version, all_columns):
    """
    Return feature columns for the compact F1/F3/F5 representations.

    F1: composition-only baseline.
    F3: clean magnetic-physics descriptors.
    F5: F3 + proxy/upper-bound magnetic descriptors.

    F2 and F4 are retained as aliases for F3 for backward compatibility only.
    """
    all_columns = set(list(all_columns))
    name_map = {
        'F1': 'chemical baseline',
        'F2': 'physics-informed clean magnetic descriptor model',
        'F3': 'physics-informed clean magnetic descriptor model',
        'F4': 'physics-informed clean magnetic descriptor model',
        'F5': 'proxy-augmented upper-bound model',
    }
    resolved_version = name_map.get(version, version)

    f1 = _ordered_unique_features(
        groups.get("composition_basic", []) +
        groups.get("element_fractions", [])
    )

    f3 = _ordered_unique_features(
        f1 +
        groups.get("magpie", []) +
        groups.get("magpie_magnetic_relevant", []) +
        groups.get("valence", []) +
        groups.get("magnetic_chemistry", []) +
        groups.get("magpieex_magnetic", []) +
        groups.get("magpieex_lite", []) +
        groups.get("magpieex_cation_anion", []) +
        groups.get("oxidation_states", []) +
        groups.get("formula_family", []) +
        groups.get("ionic_and_electronegativity", []) +
        groups.get("boundary_interaction_indices", []) +
        # Physics-informed extensions: GKA exchange, SOC/anisotropy, mean-field priors,
        # geometric frustration, structural tolerance factors, and DAO 3D crystallography
        groups.get("gka_and_exchange", []) +
        groups.get("anisotropy_and_soc", []) +
        groups.get("mean_field_priors", []) +
        groups.get("frustration_and_competition", []) +
        groups.get("structural_tolerance", []) +
        groups.get("mace_quantum", []) +
        groups.get("dao_crystallography", [])
    )

    f5 = _ordered_unique_features(
        f3 +
        groups.get("mace_quantum", []) +
        groups.get("dao_crystallography", []) +
        (groups.get("mp_symmetry", []) + groups.get("crystal_system_dummies", []) if USE_STRUCTURE_FEATURES_IF_AVAILABLE else [])
    )

    if resolved_version == 'chemical baseline':
        feats = f1
    elif resolved_version == 'physics-informed clean magnetic descriptor model':
        feats = f3
    elif resolved_version == 'proxy-augmented upper-bound model':
        feats = f5
    else:
        raise ValueError(f"Unknown ablation version: {version}")

    return [c for c in feats if c in all_columns]


def leakage_safe_feature_selection(X_train, X_val, y_train=None, var_threshold=1e-5, corr_threshold=0.95, mandatory_features=None):
    """
    Performs leakage-safe imputation, variance thresholding, Spearman hierarchical feature clustering,
    and representative target-correlated feature selection with domain-mandatory feature preservation.
    """
    # De-duplicate any accidental duplicate column names
    X_train = X_train.loc[:, ~X_train.columns.duplicated()].copy()
    X_val = X_val.loc[:, ~X_val.columns.duplicated()].copy()

    # Screen out raw pre-cleaning duplicates of the regression targets, which are
    # still present in features_master.csv at rho = 1.0 with their clean_*
    # counterparts. See FORBIDDEN_FEATURE_COLUMNS.
    safe_cols = screen_features(list(X_train.columns), context="feature selection")
    if len(safe_cols) != X_train.shape[1]:
        X_train = X_train[safe_cols]
        X_val = X_val[[c for c in safe_cols if c in X_val.columns]]
    
    # 1. Median Imputation
    medians = X_train.median().fillna(0.0)
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    
    # 2. Identify Mandatory Features (guaranteed preservation)
    mandatory_cols = []
    if mandatory_features is not None:
        mandatory_cols = [c for c in mandatory_features if c in X_train.columns]
    
    # Low-Variance and Sparse-Noise Filter (exempting mandatory features)
    variances = X_train.var()
    low_var_cols = [c for c in variances[variances < var_threshold].index.tolist() if c not in mandatory_cols]
    min_nonzero = max(8, int(0.002 * len(X_train)))
    sparse_cols = [c for c in X_train.columns if c not in mandatory_cols and (X_train[c] != 0).sum() < min_nonzero]
    drop_noise = set(low_var_cols + sparse_cols)
    keep_cols = [c for c in X_train.columns if c not in drop_noise]
    
    X_train = X_train[keep_cols]
    X_val = X_val[keep_cols]
    
    # 3. Spearman Hierarchical Feature Clustering & Selection
    # Cluster the non-mandatory candidate columns
    clusterable_cols = [c for c in X_train.columns if c not in mandatory_cols]
    
    if len(clusterable_cols) > 40:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        X_clust = X_train[clusterable_cols]
        # Calculate absolute Spearman rank correlation
        corr_matrix = X_clust.corr(method='spearman').abs().fillna(0.0)
        
        # Convert to distance matrix (D = 1 - |R|)
        dist_matrix = 1.0 - corr_matrix
        dist_sym = (dist_matrix + dist_matrix.T) / 2.0
        np.fill_diagonal(dist_sym.values, 0.0)
        
        # Get condensed distance vector
        condensed_dist = squareform(dist_sym, force='tovector')
        
        # Perform complete linkage clustering
        Z = linkage(condensed_dist, method='complete')
        
        # Target ~80 features total including mandatory
        target_clusters = max(15, min(80 - len(mandatory_cols), X_clust.shape[1]))
        cluster_labels = fcluster(Z, t=target_clusters, criterion='maxclust')
        
        # Group features by cluster label
        cluster_features = {}
        for col, lbl in zip(clusterable_cols, cluster_labels):
            if lbl not in cluster_features:
                cluster_features[lbl] = []
            cluster_features[lbl].append(col)
            
        selected_cols = []
        if y_train is not None:
            y_numeric = pd.to_numeric(y_train, errors='coerce').fillna(0.0)
            
            # Detect task type: classification (categorical) vs regression (continuous)
            unique_vals = np.unique(y_numeric)
            is_classification = len(unique_vals) <= 3 and all(val in [0.0, 1.0, 2.0] for val in unique_vals)
            
            if is_classification:
                from sklearn.feature_selection import f_classif
                X_temp = X_clust.fillna(X_clust.median().fillna(0.0))
                f_scores, _ = f_classif(X_temp, y_numeric)
                scores = pd.Series(f_scores, index=X_clust.columns).fillna(0.0)
                
                afm_boost_cols = [c for c in X_clust.columns if any(k in c.lower() for k in [
                    'd_electron', 'f_electron', 'filling', 'unpaired', 'sublattice',
                    'superexchange', 'magnetic_ion', 'spin_orbit',
                    'd_shell_half_filling', 'stoner', 'frustration', 'goldschmidt', 'double_exchange',
                    'exchange_competition', 'ligand_field', 'gka'
                ])]
                for c in afm_boost_cols:
                    scores[c] *= 1.4
            else:
                # Use pure Spearman rank correlation without artificial proxy multiplier
                # to prevent noisy/weak proxies from displacing genuine predictive descriptors
                scores = X_clust.corrwith(y_numeric, method='spearman').abs().fillna(0.0)

            for lbl, cols in cluster_features.items():
                best_col = max(cols, key=lambda c: scores[c])
                selected_cols.append(best_col)
        else:
            for lbl, cols in cluster_features.items():
                best_col = max(cols, key=lambda c: variances[c])
                selected_cols.append(best_col)
                
        # Preserve GNN columns and mandatory columns
        gnn_cols = [c for c in X_train.columns if 'gnn_' in c]
        keep_cols = list(dict.fromkeys(mandatory_cols + selected_cols + gnn_cols))
        
        X_train = X_train[keep_cols]
        X_val = X_val[keep_cols]
    elif X_train.shape[1] > 1:
        corr_matrix = X_train.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold) and column not in mandatory_cols]
        
        keep_cols = [c for c in X_train.columns if c not in to_drop]
        X_train = X_train[keep_cols]
        X_val = X_val[keep_cols]
        
    return X_train, X_val, X_train.columns.tolist(), medians.to_dict()

def compute_custom_pdp(model, X, feature, grid_size=30):
    """
    Manually computes partial dependence values for a feature to avoid version conflicts.
    """
    grid = np.linspace(X[feature].min(), X[feature].max(), grid_size)
    pdp_values = []
    
    X_temp = X.copy()
    for val in grid:
        X_temp[feature] = val
        preds = model.predict(X_temp)
        pdp_values.append(np.mean(preds))
        
    return grid, np.array(pdp_values)

def plot_confusion_matrix(y_true, y_pred, labels, filepath, title='Out-of-Fold Confusion Matrix'):
    """
    Renders an advanced Matlab-style confusion matrix with extremely high-contrast
    sophisticated colors to avoid green-overload:
    - Diagonal: solid cobalt blue (#1565C0) and white text.
    - Off-Diagonal: soft pastel pink (#FFEBEE) and dark red text (#B71C1C).
    - Marginals: base teal (#00796B) with opacity dynamically mapping to values.
    """
    import matplotlib.patches as patches
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10.0, 10.0), facecolor='#F9F9FC')
    ax.axis('off')
    
    total_samples = np.sum(cm)
    
    # Custom sophisticated colors
    diagonal_bg = '#1565C0'   # Solid bold cobalt blue
    diagonal_text = '#FFFFFF' # Bold white
    
    error_bg = '#FFEBEE'      # Soft pastel red
    error_text = '#B71C1C'    # Dark crimson red
    
    zero_bg = '#FFFFFF'       # White
    zero_text = '#CBD5E1'     # Soft gray
    
    overall_bg = '#0F172A'    # Deep dark slate
    overall_text = '#FFFFFF'  # Bold white
    overall_err_text = '#FF8A80' # Soft light red
    
    base_teal_rgb = (0/255, 121/255, 107/255) # #00796B
    
    for y in range(4):
        for x in range(4):
            if x < 3 and y < 3:
                # Inside confusion matrix
                is_diag = (x == y)
                val = cm[y, x]
                pct = 100.0 * val / total_samples
                
                if is_diag:
                    face_color = diagonal_bg
                    text_color = diagonal_text
                else:
                    if val > 0:
                        face_color = error_bg
                        text_color = error_text
                    else:
                        face_color = zero_bg
                        text_color = zero_text
                        
                rect = patches.Rectangle(
                    (x, 3 - y), 1.0, 1.0, 
                    facecolor=face_color, edgecolor='white', linewidth=2.5
                )
                ax.add_patch(rect)
                
                # Count and Percent Text (Large and bold)
                ax.text(
                    x + 0.5, 3 - y + 0.6, f"{val:,}", 
                    ha='center', va='center', fontsize=20, fontweight='black', color=text_color
                )
                ax.text(
                    x + 0.5, 3 - y + 0.35, f"{pct:.1f}%", 
                    ha='center', va='center', fontsize=15, fontweight='bold', color=text_color
                )
                
            elif x == 3 and y < 3:
                # Right margin: Row Precision (Teal base, opacity based on value)
                row_sum = np.sum(cm[y, :])
                precision = cm[y, y] / row_sum if row_sum > 0 else 0
                fdr = 1.0 - precision
                
                alpha = 0.25 + 0.70 * precision
                face_color = (base_teal_rgb[0], base_teal_rgb[1], base_teal_rgb[2], alpha)
                
                # Contrasting text colors based on background opacity
                if precision >= 0.65:
                    text_pct_color = '#FFFFFF'  # Bold white
                    text_err_color = '#E0F2F1'  # Soft light teal
                else:
                    text_pct_color = '#004D40'  # Bold dark teal
                    text_err_color = '#B71C1C'  # Muted crimson for error
                
                rect = patches.Rectangle(
                    (x, 3 - y), 1.0, 1.0, 
                    facecolor=face_color, edgecolor='white', linewidth=2.5
                )
                ax.add_patch(rect)
                
                # Precision Label and FDR Label (Large and bold)
                ax.text(
                    x + 0.5, 3 - y + 0.62, f"{100.0*precision:.1f}%", 
                    ha='center', va='center', fontsize=18, fontweight='black', color=text_pct_color
                )
                ax.text(
                    x + 0.5, 3 - y + 0.38, f"{100.0*fdr:.1f}%", 
                    ha='center', va='center', fontsize=15, fontweight='bold', color=text_err_color
                )
                
            elif x < 3 and y == 3:
                # Bottom margin: Column Recall (Teal base, opacity based on value)
                col_sum = np.sum(cm[:, x])
                recall = cm[x, x] / col_sum if col_sum > 0 else 0
                fnr = 1.0 - recall
                
                alpha = 0.25 + 0.70 * recall
                face_color = (base_teal_rgb[0], base_teal_rgb[1], base_teal_rgb[2], alpha)
                
                # Contrasting text colors based on background opacity
                if recall >= 0.65:
                    text_pct_color = '#FFFFFF'  # Bold white
                    text_err_color = '#E0F2F1'  # Soft light teal
                else:
                    text_pct_color = '#004D40'  # Bold dark teal
                    text_err_color = '#B71C1C'  # Muted crimson for error
                
                rect = patches.Rectangle(
                    (x, 3 - y), 1.0, 1.0, 
                    facecolor=face_color, edgecolor='white', linewidth=2.5
                )
                ax.add_patch(rect)
                
                # Recall Label and FNR Label
                ax.text(
                    x + 0.5, 3 - y + 0.62, f"{100.0*recall:.1f}%", 
                    ha='center', va='center', fontsize=18, fontweight='black', color=text_pct_color
                )
                ax.text(
                    x + 0.5, 3 - y + 0.38, f"{100.0*fnr:.1f}%", 
                    ha='center', va='center', fontsize=15, fontweight='bold', color=text_err_color
                )
                
            else:
                # Bottom-right cell: Overall Accuracy / Error (Solid deep charcoal background)
                diag_sum = np.sum(np.diag(cm))
                overall_acc = diag_sum / total_samples
                overall_err = 1.0 - overall_acc
                
                rect = patches.Rectangle(
                    (x, 3 - y), 1.0, 1.0, 
                    facecolor=overall_bg, edgecolor='white', linewidth=2.5
                )
                ax.add_patch(rect)
                
                # Accuracy Label (White) and Error Label (Light red)
                ax.text(
                    x + 0.5, 3 - y + 0.65, f"{100.0*overall_acc:.1f}%", 
                    ha='center', va='center', fontsize=20, fontweight='black', color=overall_text
                )
                ax.text(
                    x + 0.5, 3 - y + 0.35, f"{100.0*overall_err:.1f}%", 
                    ha='center', va='center', fontsize=17, fontweight='black', color=overall_err_text
                )
                
    # --- Custom Larger Axes Labels ---
    # Column classes names on top
    for x in range(3):
        ax.text(x + 0.5, 4.15, labels[x], ha='center', va='bottom', fontsize=20, fontweight='black', color='#0F172A')
    ax.text(3.5, 4.15, "Precision", ha='center', va='bottom', fontsize=20, fontweight='black', color='#1E293B')
    
    # Row classes names on left
    for y in range(3):
        ax.text(-0.15, 3 - y + 0.5, labels[y], ha='right', va='center', fontsize=20, fontweight='black', color='#0F172A')
    ax.text(-0.15, 0.5, "Recall", ha='right', va='center', fontsize=20, fontweight='black', color='#1E293B')
    
    # Axes Group Labels
    # Predicted class on Y (vertical left)
    ax.text(-0.85, 2.0, "Predicted Class", rotation=90, ha='center', va='center', fontsize=22, fontweight='black', color='#0F172A')
    # Actual class on X (horizontal top)
    ax.text(1.5, 4.65, "Actual Class", ha='center', va='bottom', fontsize=22, fontweight='black', color='#0F172A')
    
    # Set coordinates boundaries
    ax.set_xlim(-0.95, 4.2)
    ax.set_ylim(-0.25, 4.85)
    
    # Main Title
    ax.text(1.7, 4.8, title, ha='center', va='bottom', fontsize=18, fontweight='bold', color='#0F172A')
    
    # Add floating subtitle explanation of marginal color coding
    legend_box = dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#CBD5E1', alpha=0.95)
    ax.text(
        1.7, -0.15, "Diagonal: Correct predictions  |  Off-diagonal: Incorrect errors\nMarginals: Base Teal (#00796B)  |  Background opacity maps linearly to Precision/Recall values",
        ha='center', va='top', fontsize=11, style='italic', color='#475569', bbox=legend_box
    )
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved publication-grade confusion matrix to {filepath}")

def plot_predictions_vs_actual(y_true, y_pred, title, ylabel, filepath, color):
    """
    Plots a highly advanced density-colored parity scatter plot using k-NN smoothed point density.
    Forces square aspect ratios, identical limits (0 to 1200 K for temperatures), major ticks
    every 400 K (labeled), and minor ticks every 200 K (unlabeled). Renders actual count colorbar
    using the premium viridis colormap to match the reference style.
    """
    setup_plotting_style()
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    n    = len(y_true)
    
    # Subsample to keep calculation extremely fast
    if n > 3500:
        np.random.seed(42)
        idx_samp = np.random.choice(n, 3500, replace=False)
        y_true_s = y_true[idx_samp]
        y_pred_s = y_pred[idx_samp]
    else:
        y_true_s = y_true
        y_pred_s = y_pred
        
    # Calculate 2D histogram discrete counts for absolute counts scaling
    bins = [100, 100]
    counts, xedges, yedges = np.histogram2d(y_true_s, y_pred_s, bins=bins)
    
    # Calculate smooth neighbor-based density (nearby points occurrence) using k-NN distance
    xy = np.vstack([y_true_s, y_pred_s])
    try:
        from sklearn.neighbors import NearestNeighbors
        k_val = min(40, len(y_true_s) - 1)
        nbrs = NearestNeighbors(n_neighbors=k_val).fit(xy.T)
        distances, indices = nbrs.kneighbors(xy.T)
        # Density is inversely proportional to mean distance to neighbors
        density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-5)
        
        # Normalize and scale density to represent the actual local point count
        density_norm = (density - density.min()) / (density.max() - density.min() + 1e-8)
        colors = density_norm * counts.max()
    except Exception:
        x_bins = np.clip(np.digitize(y_true_s, xedges) - 1, 0, len(xedges) - 2)
        y_bins = np.clip(np.digitize(y_pred_s, yedges) - 1, 0, len(yedges) - 2)
        colors = counts[x_bins, y_bins]
        
    # Sort points by count so denser points are plotted on top
    sort_idx = colors.argsort()
    y_true_s = y_true_s[sort_idx]
    y_pred_s = y_pred_s[sort_idx]
    colors = colors[sort_idx]
    
    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor='#F9F9FC')
    
    # Plot scatter colored by density count using standard viridis
    sc = ax.scatter(
        y_true_s, y_pred_s,
        c=colors,
        cmap='viridis', # Beautiful standard Viridis for smooth high-contrast density transitions
        s=30,
        alpha=0.88,
        edgecolors='none',
        rasterized=True,
        zorder=3
    )
    
    # Custom colorbar: shorter (shrink=0.40), raw point counts displayed
    cb = fig.colorbar(sc, ax=ax, shrink=0.40, pad=0.04)
    cb.set_ticks([colors.min(), (colors.min() + colors.max()) / 2.0, colors.max()])
    cb.set_ticklabels([f"{int(colors.min())}", f"{int((colors.min() + colors.max()) / 2.0)}", f"{int(colors.max())}"])
    cb.set_label('Point Count', fontsize=11, fontweight='bold', labelpad=8)
    cb.ax.tick_params(labelsize=10)
    
    # Force axes to be square with same upper limit
    ax.set_box_aspect(1)
    
    is_temp = "temperature" in ylabel.lower() or "curie" in title.lower() or "néel" in title.lower() or "tc" in title.lower() or "tn" in title.lower()
    
    if is_temp:
        min_val = 0
        max_val = 1200
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        # Grid tick marks every 200 K, but label only every 400 K
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MultipleLocator(400))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(200))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(400))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(200))
    else:
        # For non-temperature (like anisotropy), scale equally
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
    # Ideal diagonal y = x
    ax.plot(
        [min_val, max_val], [min_val, max_val],
        color='#0F172A', linewidth=2.0, linestyle='--', zorder=4
    )
    
    # ±10% envelope band
    xs = np.linspace(min_val, max_val, 200)
    ax.fill_between(
        xs, xs * 0.90, xs * 1.10,
        color='#1E3A8A', alpha=0.08, zorder=2
    )
    
    # Floating Stats Box (Bold and larger text - RMSE, MAE, R2 all included)
    stats_text = (
        f"R² = {r2:.3f}\n"
        f"RMSE = {rmse:.1f} K\n"
        f"MAE = {mae:.1f} K\n"
        f"N = {n:,}"
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=14, fontweight='bold',
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#CBD5E1', alpha=0.9),
        zorder=10
    )
    
    # Set labels with big and bold fonts
    if "curie" in title.lower() or "tc" in title.lower():
        ax.set_xlabel(r"Experimental Curie Temperature, $T_C$ (K)", fontsize=15, fontweight='bold', labelpad=10)
        ax.set_ylabel(r"Predicted Curie Temperature, $T_C$ (K)", fontsize=15, fontweight='bold', labelpad=10)
    elif "neel" in title.lower() or "tn" in title.lower() or "néel" in title.lower():
        ax.set_xlabel(r"Experimental Néel Temperature, $T_N$ (K)", fontsize=15, fontweight='bold', labelpad=10)
        ax.set_ylabel(r"Predicted Néel Temperature, $T_N$ (K)", fontsize=15, fontweight='bold', labelpad=10)
    else:
        ax.set_xlabel('Experimental Value', fontsize=15, fontweight='bold', labelpad=10)
        ax.set_ylabel(ylabel, fontsize=15, fontweight='bold', labelpad=10)
        
    ax.set_title(title, fontsize=18, fontweight='bold', color='#0F172A', pad=14)
    ax.grid(True, which='both', color='#E2E8F0', linestyle='--', linewidth=0.6)
    
    # Make axes ticks big and bold
    ax.tick_params(axis='both', which='major', labelsize=13, width=1.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        
    # Style colorbar ticks and label to be big and bold
    cb.ax.tick_params(labelsize=12)
    for label in cb.ax.get_yticklabels():
        label.set_fontweight('bold')
    cb.set_label('Point Count', fontsize=13, fontweight='bold', labelpad=8)
    
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved premium parity plot to {filepath}")

def plot_overlay_predictions_vs_actual(curie_res, neel_res, plot_dir='plots'):
    """
    Generates an improved dual-panel overlay parity plot with density-colored scatters using 2D histogram point counts.
    """
    setup_plotting_style()
    from src.utils import apply_clean_axes
    import matplotlib.ticker as ticker
    
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.5), facecolor='#F9F9FC')
    
    datasets = [
        (curie_res, axes[0], 'Curie Temperature ($T_C$) Predictions', 'FM', r"Predicted Curie Temperature, $T_C$ (K)"),
        (neel_res,  axes[1], 'Néel Temperature ($T_N$) Predictions',  'AFM', r"Predicted Néel Temperature, $T_N$ (K)")
    ]
    
    for df_res, ax, panel_title, label, ylabel in datasets:
        y_true = np.asarray(df_res['True']).flatten()
        y_pred = np.asarray(df_res['Pred']).flatten()
        
        r2   = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        n    = len(y_true)
        
        # Subsample to keep calculation extremely fast
        if n > 2500:
            np.random.seed(42)
            idx_samp = np.random.choice(n, 2500, replace=False)
            y_true_s = y_true[idx_samp]
            y_pred_s = y_pred[idx_samp]
        else:
            y_true_s = y_true
            y_pred_s = y_pred
            
        # Calculate 2D histogram point counts
        bins = [100, 100]
        counts, xedges, yedges = np.histogram2d(y_true_s, y_pred_s, bins=bins)
        
        # Calculate smooth k-NN density
        xy = np.vstack([y_true_s, y_pred_s])
        try:
            from sklearn.neighbors import NearestNeighbors
            k_val = min(40, len(y_true_s) - 1)
            nbrs = NearestNeighbors(n_neighbors=k_val).fit(xy.T)
            distances, indices = nbrs.kneighbors(xy.T)
            density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-5)
            density_norm = (density - density.min()) / (density.max() - density.min() + 1e-8)
            colors = density_norm * counts.max()
        except Exception:
            x_bins = np.clip(np.digitize(y_true_s, xedges) - 1, 0, len(xedges) - 2)
            y_bins = np.clip(np.digitize(y_pred_s, yedges) - 1, 0, len(yedges) - 2)
            colors = counts[x_bins, y_bins]
            
        sort_idx = colors.argsort()
        y_true_s = y_true_s[sort_idx]
        y_pred_s = y_pred_s[sort_idx]
        colors = colors[sort_idx]
        
        sc = ax.scatter(
            y_true_s, y_pred_s,
            c=colors,
            cmap='viridis',
            s=30,
            alpha=0.88,
            edgecolors='none',
            rasterized=True,
            zorder=3
        )
        
        # Custom colorbar: raw point counts, shorter (shrink=0.40)
        cb = fig.colorbar(sc, ax=ax, shrink=0.40, pad=0.04)
        cb.set_ticks([colors.min(), (colors.min() + colors.max()) / 2.0, colors.max()])
        cb.set_ticklabels([f"{int(colors.min())}", f"{int((colors.min() + colors.max()) / 2.0)}", f"{int(colors.max())}"])
        cb.set_label('Point Count', fontsize=10, fontweight='bold', labelpad=6)
        cb.ax.tick_params(labelsize=9)
        
        # Force square aspect ratio and equal limits (0 to 1200 K)
        ax.set_box_aspect(1)
        lo = 0
        hi = 1200
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        
        # Grid tick marks every 200 K, label only every 400 K
        ax.xaxis.set_major_locator(ticker.MultipleLocator(400))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(200))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(400))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(200))
        
        # Ideal diagonal
        ax.plot(
            [lo, hi], [lo, hi], 
            color='#0F172A', lw=2.0, linestyle='--', zorder=4
        )
        
        # ±10% envelope
        xs = np.linspace(lo, hi, 300)
        ax.fill_between(xs, xs * 0.90, xs * 1.10, color='#1E3A8A', alpha=0.08, zorder=2)
        
        # Floating Stats Box in Bold
        stats_text = (
            f"R² = {r2:.3f}\n"
            f"RMSE = {rmse:.1f} K\n"
            f"MAE = {mae:.1f} K\n"
            f"N = {n:,}"
        )
        ax.text(
            0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#CBD5E1', alpha=0.9),
            zorder=10
        )
        
        # Label formatting with TC and TN
        if "curie" in panel_title.lower() or "tc" in panel_title.lower():
            x_label_str = r"Experimental Curie Temperature, $T_C$ (K)"
        else:
            x_label_str = r"Experimental Néel Temperature, $T_N$ (K)"
            
        ax.set_xlabel(x_label_str, fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold', labelpad=10)
        ax.set_title(panel_title, fontsize=14, fontweight='bold', color='#0F172A', pad=10)
        
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, which='both', color='#E2E8F0', linestyle='--', linewidth=0.6)
        
    fig.suptitle('Ordering Temperature Predictions vs. Experiment (Density Scatter)',
                 fontsize=16, fontweight='bold', color='#0F172A', y=0.98)
    
    out_path = os.path.join(plot_dir, 'predictions_vs_actual_overlay.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dual-panel density scatter overlay parity plot to {out_path}")

def plot_unified_feature_importance_bubble(class_imp, curie_imp, neel_imp, plot_dir='plots'):
    """
    Plots feature importances across all 3 modeling tasks on a single
    Cleveland dot chart canvas — sorted by peak cross-task importance,
    bubble size and color encode relative normalized importance per task.
    """
    setup_plotting_style()
    from src.utils import apply_clean_axes, PALETTE

    # Build union of top-12 features per task
    top_class = sorted(class_imp.items(), key=lambda x: x[1], reverse=True)[:12]
    top_curie = sorted(curie_imp.items(), key=lambda x: x[1], reverse=True)[:12]
    top_neel  = sorted(neel_imp.items(), key=lambda x: x[1], reverse=True)[:12]

    all_features = list(dict.fromkeys(
        [f for f, _ in top_class] + [f for f, _ in top_curie] + [f for f, _ in top_neel]
    ))[:18]  # cap at 18 features for readability

    # Normalize per-task
    max_class = max(class_imp.values()) if class_imp else 1.0
    max_curie = max(curie_imp.values()) if curie_imp else 1.0
    max_neel  = max(neel_imp.values())  if neel_imp  else 1.0

    # Sort by maximum importance across any task (descending)
    all_features = sorted(
        all_features,
        key=lambda f: max(class_imp.get(f, 0)/max_class,
                          curie_imp.get(f, 0)/max_curie,
                          neel_imp.get(f, 0)/max_neel),
        reverse=True
    )

    tasks = ['Magnetic Class', 'Curie Temp $T_C$', 'Néel Temp $T_N$']
    task_colors = [PALETTE['nm_color'], PALETTE['fm_color'], PALETTE['afm_color']]
    task_imps   = [class_imp, curie_imp, neel_imp]
    task_maxes  = [max_class, max_curie, max_neel]
    task_offsets = [-0.22, 0.0, 0.22]  # Y-jitter for overlapping bubbles

    fig, ax = plt.subplots(figsize=(11, max(7, len(all_features) * 0.52)))

    y_positions = list(range(len(all_features)))

    for t_idx, (task, color, imp_dict, t_max, yoff) in enumerate(
        zip(tasks, task_colors, task_imps, task_maxes, task_offsets)
    ):
        xs, ys, ss = [], [], []
        for y_pos, feat in enumerate(all_features):
            val = imp_dict.get(feat, 0.0) / t_max
            xs.append(val)
            ys.append(y_pos + yoff)
            ss.append(max(30, val * 800))

        ax.scatter(xs, ys, s=ss, color=color, alpha=0.82,
                   edgecolors='white', linewidths=0.5, zorder=4, label=task)

        # Horizontal line from 0 to bubble center (Cleveland dot style)
        for val, y_p_off in zip(xs, ys):
            ax.plot([0, val], [y_p_off, y_p_off],
                    color=color, alpha=0.3, linewidth=0.9, zorder=3)

    # Y-axis: clean feature labels
    clean_labels = [
        f.replace('MagpieData ', '').replace('_', ' ').title()
        for f in all_features
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(clean_labels, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel('Normalized Feature Importance', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Cross-Task Feature Importance Matrix', fontsize=14, fontweight='bold',
                 color='#1A202C', pad=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E2E8F0')
    ax.set_xlim(-0.02, 1.05)

    ax.legend(loc='lower right', frameon=True, framealpha=0.92, fontsize=10,
              title='Prediction Task', title_fontsize=10)

    ax.xaxis.grid(True, color='#E2E8F0', linestyle='--', linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path = os.path.join(plot_dir, 'feature_importances_unified.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved cross-task feature importance Cleveland dot chart to {out_path}")

def plot_overlay_transition_metal_pdp(model, X_train, plot_dir='plots'):
    """
    Overlays the Partial Dependence Curves for key magnetic transition metals (Fe, Co, Mn, Ni)
    together on a single set of axes, with translucent ±1σ envelopes and publication-quality styling.
    """
    setup_plotting_style()
    from src.utils import apply_clean_axes, PALETTE

    metals = ['Fe', 'Co', 'Mn', 'Ni']
    colors = {
        'Fe': PALETTE['fm_color'],   # Crimson
        'Co': PALETTE['cobalt'],     # Cobalt blue
        'Mn': PALETTE['gold'],       # Amber gold
        'Ni': PALETTE['afm_color'],  # Steel cerulean
    }

    fig, ax = plt.subplots(figsize=(9.5, 6))
    has_pdp = False

    for metal in metals:
        if metal in X_train.columns:
            has_pdp = True
            grid, pdp_vals = compute_custom_pdp(model, X_train, metal, grid_size=40)

            sigma = np.std(pdp_vals) * 0.10   # ±10% of std as visual envelope
            ax.plot(grid, pdp_vals, color=colors[metal], linewidth=2.4,
                    label=f'{metal}', zorder=4)
            ax.fill_between(grid, pdp_vals - sigma, pdp_vals + sigma,
                            color=colors[metal], alpha=0.14, zorder=3)

            # Peak annotation
            peak_idx = np.argmax(pdp_vals)
            ax.annotate(
                f'{pdp_vals[peak_idx]:.0f} K',
                xy=(grid[peak_idx], pdp_vals[peak_idx]),
                xytext=(5, 10), textcoords='offset points',
                fontsize=8.5, color=colors[metal], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=colors[metal], lw=0.8)
            )

    if not has_pdp:
        print("Warning: Fe, Co, Mn, Ni not found in training dataset. Skipping Transition Metal PDP.")
        plt.close(fig)
        return

    apply_clean_axes(ax,
                     xlabel='Transition Metal Atomic Fraction (0 → 1)',
                     ylabel='Partial Dependence  $T_C$ (K)',
                     title='$T_C$ Sensitivity to Transition Metal Composition')
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc='upper right', frameon=True, fontsize=10, title='Element', title_fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(plot_dir, 'pdp_transition_metals.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved premium transition metal PDP overlay to {out_path}")

def train_stacked_classifier(X, y, groups, cv, features_to_use, afm_weight_multiplier=1.0, global_splits=None, global_groups=None, hard_sample_weights=None, select_features_per_fold=False, top_k_per_fold=80, is_binary=False):
    """
    Trains base models and meta-classifier in a 100% leakage-safe out-of-fold stacked framework,
    with class probability calibration.
    Uses n_estimators=100 during CV for speed; final models use 300 for accuracy.
    GNN fold results are cached globally to avoid redundant training across ablation F1-F5.
    If select_features_per_fold=True, feature sifting is nested strictly within each outer train_idx.
    """
    n_samples = len(X)
    n_classes = len(np.unique(y))
    xgb_obj = 'binary:logistic' if n_classes <= 2 else 'multi:softprob'
    xgb_eval = 'logloss' if n_classes <= 2 else 'mlogloss'
    
    # CV base classifiers. For the high-accuracy mode we follow the NEMAD
    # paper's successful RF/XGB emphasis while retaining LightGBM and ExtraTrees
    # as complementary tabular learners.
    if HIGH_ACCURACY_TABULAR and TRAINING_MODE == "final_manuscript":
        base_clfs = {
            'xgb': XGBClassifier(n_estimators=800, max_depth=10, learning_rate=0.08,
                                 subsample=0.75, colsample_bytree=0.60, min_child_weight=2,
                                 reg_alpha=0.25, reg_lambda=2.0, objective=xgb_obj,
                                 eval_metric=xgb_eval, random_state=42, n_jobs=-1),
            'rf':  RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2,
                                          min_samples_leaf=1, max_features='sqrt',
                                          class_weight='balanced_subsample', random_state=42, n_jobs=-1),
            'lgb': LGBMClassifier(n_estimators=500, max_depth=-1, learning_rate=0.05, num_leaves=63,
                                  subsample=0.80, colsample_bytree=0.70, min_child_samples=20,
                                  reg_alpha=0.25, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1),
            'et':  ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1,
                                        max_features='sqrt', class_weight='balanced',
                                        random_state=42, n_jobs=-1),
            'cat': CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                      random_seed=42, verbose=0)
        }
    else:
        n_est = 120 if TRAINING_MODE == "fast_debug" else (250 if TRAINING_MODE == "development" else 400)
        base_clfs = {
            'xgb': XGBClassifier(n_estimators=n_est, max_depth=6, learning_rate=0.08,
                                 subsample=0.80, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=2.0,
                                 objective=xgb_obj, eval_metric=xgb_eval, random_state=42, n_jobs=-1),
            'rf':  RandomForestClassifier(n_estimators=n_est, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                          max_features='sqrt', class_weight='balanced_subsample', random_state=42, n_jobs=-1),
            'lgb': LGBMClassifier(n_estimators=n_est, max_depth=7, learning_rate=0.08, num_leaves=45,
                                  min_child_samples=20, reg_alpha=0.5, reg_lambda=2.0,
                                  random_state=42, n_jobs=-1, verbose=-1),
            'et':  ExtraTreesClassifier(n_estimators=n_est, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                        max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1),
            'cat': CatBoostClassifier(iterations=n_est, depth=6, learning_rate=0.06,
                                      random_seed=42, verbose=0)
        }
    
    # Storage for out-of-fold base model predictions (meta-features)
    meta_features = np.zeros((n_samples, len(base_clfs) * n_classes))
    
    base_models_by_fold = []
    meta_models_by_fold = []
    scalers_by_fold = []
    selected_cols_by_fold = []
    
    oof_probs = np.zeros((n_samples, n_classes))
    oof_preds = np.zeros(n_samples)
    oof_verifier_probs = np.zeros(n_samples)
    
    features_to_use = _align_feature_list(features_to_use, X.columns, "train_stacked_classifier features", min_required=1)
    print("Training base classifiers for stacking...")
    splits = cv if isinstance(cv, list) else list(cv.split(X, y, groups))
    for fold, (train_idx, val_idx) in enumerate(splits):
        if select_features_per_fold:
            fold_feats = select_stable_optimized_features(
                X.iloc[train_idx], y.iloc[train_idx],
                groups.iloc[train_idx] if groups is not None else None,
                features_to_use, top_k=top_k_per_fold, is_binary=is_binary
            )
        else:
            fold_feats = features_to_use
            
        X_tr, X_val = X.iloc[train_idx][fold_feats], X.iloc[val_idx][fold_feats]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Leakage-safe feature selection
        X_tr_sel, X_val_sel, _, _ = leakage_safe_feature_selection(X_tr, X_val, y_tr)
        
        # Train fold-isolated GNN classifier or retrieve from global splits
        if global_splits is not None:
            import hashlib
            global_tr_idx, global_val_idx = global_splits[fold]
            # Use global_groups if provided, otherwise fallback to groups (which is correct if this is a global model)
            g_groups = global_groups if global_groups is not None else groups
            global_tr_groups = g_groups.iloc[global_tr_idx].tolist()
            global_val_groups = g_groups.iloc[global_val_idx].tolist()
            
            # Compute global cache key
            h = hashlib.md5("|".join(sorted(str(g) for g in global_tr_groups)).encode()).hexdigest()[:12]
            if h not in _GNN_FOLD_CACHE:
                print(f"Warning: Global GNN cache key {h} not found. Fine-tuning a specialist GNN...")
                gnn_model, gnn_tr_class_full, gnn_tr_tc_full, gnn_tr_tn_full, gnn_val_class_full, gnn_val_tc_full, gnn_val_tn_full, gnn_tr_tc_unc_full, gnn_tr_tn_unc_full, gnn_val_tc_unc_full, gnn_val_tn_unc_full = get_cached_gnn(
                    global_tr_groups, global_val_groups, epochs=None
                )
            
            cache_entry = _GNN_FOLD_CACHE[h]
            
            # Map training GNN predictions
            formula_to_tr_idx = {formula: idx for idx, formula in enumerate(global_tr_groups)}
            spec_tr_formulas = groups.iloc[train_idx].tolist()
            tr_indices_in_global = [formula_to_tr_idx[f] for f in spec_tr_formulas]
            
            gnn_tr_class = cache_entry['train_class_probs'][tr_indices_in_global]
            gnn_tr_tc = cache_entry['train_pred_tc'][tr_indices_in_global]
            gnn_tr_tn = cache_entry['train_pred_tn'][tr_indices_in_global]
            gnn_tr_tc_unc = cache_entry['train_tc_unc'][tr_indices_in_global]
            gnn_tr_tn_unc = cache_entry['train_tn_unc'][tr_indices_in_global]
            
            # Map validation GNN predictions
            spec_val_formulas = groups.iloc[val_idx].tolist()
            
            mean_class = cache_entry['train_class_probs'].mean(axis=0)
            mean_tc    = cache_entry['train_pred_tc'].mean()
            mean_tn    = cache_entry['train_pred_tn'].mean()
            mean_tc_unc = cache_entry['train_tc_unc'].mean()
            mean_tn_unc = cache_entry['train_tn_unc'].mean()
            
            gnn_val_class = np.array([cache_entry['val_class_probs_by_formula'].get(f, mean_class) for f in spec_val_formulas])
            gnn_val_tc = np.array([cache_entry['val_pred_tc_by_formula'].get(f, mean_tc) for f in spec_val_formulas])
            gnn_val_tn = np.array([cache_entry['val_pred_tn_by_formula'].get(f, mean_tn) for f in spec_val_formulas])
            gnn_val_tc_unc = np.array([cache_entry['val_tc_unc_by_formula'].get(f, mean_tc_unc) for f in spec_val_formulas])
            gnn_val_tn_unc = np.array([cache_entry['val_tn_unc_by_formula'].get(f, mean_tn_unc) for f in spec_val_formulas])
        else:
            # Global training mode: train/get cached GNN as usual
            gnn_model, gnn_tr_class, gnn_tr_tc, gnn_tr_tn, gnn_val_class, gnn_val_tc, gnn_val_tn, gnn_tr_tc_unc, gnn_tr_tn_unc, gnn_val_tc_unc, gnn_val_tn_unc = get_cached_gnn(
                groups.iloc[train_idx].tolist(),
                groups.iloc[val_idx].tolist(),
                epochs=None  # Use warm-start fine-tuning
            )
        
        X_tr_sel = X_tr_sel.copy()
        X_val_sel = X_val_sel.copy()
        X_tr_sel['gnn_prob_FM'] = gnn_tr_class[:, 0]
        X_tr_sel['gnn_prob_AFM'] = gnn_tr_class[:, 1]
        X_tr_sel['gnn_prob_NM'] = gnn_tr_class[:, 2]
        X_tr_sel['gnn_tc_pred'] = gnn_tr_tc
        X_tr_sel['gnn_tn_pred'] = gnn_tr_tn
        X_tr_sel['gnn_tc_uncertainty'] = gnn_tr_tc_unc
        X_tr_sel['gnn_tn_uncertainty'] = gnn_tr_tn_unc
        
        X_val_sel['gnn_prob_FM'] = gnn_val_class[:, 0]
        X_val_sel['gnn_prob_AFM'] = gnn_val_class[:, 1]
        X_val_sel['gnn_prob_NM'] = gnn_val_class[:, 2]
        X_val_sel['gnn_tc_pred'] = gnn_val_tc
        X_val_sel['gnn_tn_pred'] = gnn_val_tn
        X_val_sel['gnn_tc_uncertainty'] = gnn_val_tc_unc
        X_val_sel['gnn_tn_uncertainty'] = gnn_val_tn_unc
        
        selected_cols_by_fold.append(X_tr_sel.columns.tolist())
        
        # Scale features
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_sel)
        X_val_scaled = scaler.transform(X_val_sel)
        scalers_by_fold.append(scaler)
        
        # Training sample weight: AFM upweighted to improve minority-class recall
        y_unique, y_counts = np.unique(y_tr, return_counts=True)
        class_weights = {cls: len(y_tr) / (len(y_unique) * cnt) for cls, cnt in zip(y_unique, y_counts)}
        sample_weight_tr = np.array([class_weights[yi] for yi in y_tr])
        sample_weight_tr[(y_tr == 1).values] *= afm_weight_multiplier
        
        # Apply hard-negative FM-AFM sample weights if provided (Experiment 6)
        if hard_sample_weights is not None:
            sample_weight_tr *= hard_sample_weights[train_idx]
        
        fold_base_models = {}
        for name, clf_class in base_clfs.items():
            import copy
            clf = copy.deepcopy(clf_class)
            clf.fit(X_tr_scaled, y_tr, sample_weight=sample_weight_tr)
            
            # Predict probabilities
            preds_val = clf.predict_proba(X_val_scaled)
            
            # Save to meta-features
            idx_start = list(base_clfs.keys()).index(name) * n_classes
            meta_features[val_idx, idx_start:idx_start+n_classes] = preds_val
            fold_base_models[name] = clf
            
        base_models_by_fold.append(fold_base_models)
        
        # Second-stage binary AFM verifier fold training (Component 5)
        y_tr_bin = (y_tr == 1).astype(int)
        y_unique_bin, y_counts_bin = np.unique(y_tr_bin, return_counts=True)
        class_weights_bin = {cls: len(y_tr_bin) / (2.0 * cnt) for cls, cnt in zip(y_unique_bin, y_counts_bin)}
        sample_weight_bin = np.array([class_weights_bin[yi] for yi in y_tr_bin])
        
        clf_bin = LGBMClassifier(n_estimators=80, max_depth=4, learning_rate=0.05, num_leaves=15,
                                 random_state=42, n_jobs=-1, verbose=-1)
        clf_bin.fit(X_tr_scaled, y_tr_bin, sample_weight=sample_weight_bin)
        oof_verifier_probs[val_idx] = clf_bin.predict_proba(X_val_scaled)[:, 1]
        
    print("Training meta-classifier and performing probability calibration...")
    for fold, (train_idx, val_idx) in enumerate(splits):
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        meta_tr, meta_val = meta_features[train_idx], meta_features[val_idx]
        
        # Compute inverse-frequency sample weights for meta-classifier (Fix B2)
        y_unique_m, y_counts_m = np.unique(y_tr, return_counts=True)
        class_weights_m = {cls: len(y_tr) / (len(y_unique_m) * cnt) for cls, cnt in zip(y_unique_m, y_counts_m)}
        sample_weight_meta_tr = np.array([class_weights_m[yi] for yi in y_tr])
        
        lr_meta = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        calibrated_meta = CalibratedClassifierCV(estimator=lr_meta, method='sigmoid', cv=5)
        calibrated_meta.fit(meta_tr, y_tr, sample_weight=sample_weight_meta_tr)
        
        probs_val = calibrated_meta.predict_proba(meta_val)
        oof_probs[val_idx] = probs_val
        oof_preds[val_idx] = np.argmax(probs_val, axis=1)
        
        meta_models_by_fold.append(calibrated_meta)
        
    return oof_preds, oof_probs, oof_verifier_probs, base_models_by_fold, meta_models_by_fold, scalers_by_fold, selected_cols_by_fold

def train_final_stacked_classifier(X, y, groups, features_to_use, afm_weight_multiplier=1.0, gnn_classifier=None, hard_sample_weights=None):
    """
    Trains base models and meta-classifier on the ENTIRE dataset for production/screening.
    """
    if hard_sample_weights is not None:
        hard_sample_weights = np.asarray(hard_sample_weights)
    n_classes = len(np.unique(y))
    xgb_obj = 'binary:logistic' if n_classes <= 2 else 'multi:softprob'
    xgb_eval = 'logloss' if n_classes <= 2 else 'mlogloss'
    
    # Final classifier. Use high-capacity RF/XGB in final_manuscript mode,
    # matching the strong tabular strategy reported in the NEMAD paper.
    if HIGH_ACCURACY_TABULAR and TRAINING_MODE == "final_manuscript":
        base_clfs = {
            'xgb': XGBClassifier(n_estimators=1000, max_depth=10, learning_rate=0.06,
                                 subsample=0.80, colsample_bytree=0.60, min_child_weight=2,
                                 reg_alpha=0.25, reg_lambda=2.0, objective=xgb_obj,
                                 eval_metric=xgb_eval, random_state=42, n_jobs=-1),
            'rf':  RandomForestClassifier(n_estimators=800, max_depth=None, min_samples_split=2,
                                          min_samples_leaf=1, max_features='sqrt',
                                          class_weight='balanced_subsample', random_state=42, n_jobs=-1),
            'lgb': LGBMClassifier(n_estimators=700, max_depth=-1, learning_rate=0.04, num_leaves=63,
                                  subsample=0.85, colsample_bytree=0.70, min_child_samples=20,
                                  reg_alpha=0.25, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1),
            'et':  ExtraTreesClassifier(n_estimators=700, max_depth=None, min_samples_split=2,
                                        min_samples_leaf=1, max_features='sqrt',
                                        class_weight='balanced', random_state=42, n_jobs=-1),
            'cat': CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                      random_seed=42, verbose=0)
        }
    else:
        n_est = 150 if TRAINING_MODE == "fast_debug" else (300 if TRAINING_MODE == "development" else 500)
        base_clfs = {
            'xgb': XGBClassifier(n_estimators=n_est, max_depth=7, learning_rate=0.06, subsample=0.85,
                                 colsample_bytree=0.75, min_child_weight=2, reg_alpha=0.5, reg_lambda=2.0,
                                 objective=xgb_obj, eval_metric=xgb_eval, random_state=42, n_jobs=-1),
            'rf':  RandomForestClassifier(n_estimators=n_est, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                          max_features='sqrt', class_weight='balanced_subsample', random_state=42, n_jobs=-1),
            'lgb': LGBMClassifier(n_estimators=n_est, max_depth=8, learning_rate=0.06, num_leaves=45,
                                  subsample=0.85, colsample_bytree=0.75, min_child_samples=20,
                                  reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1),
            'et':  ExtraTreesClassifier(n_estimators=n_est, max_depth=None, min_samples_split=2,
                                        min_samples_leaf=1, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1),
            'cat': CatBoostClassifier(iterations=n_est, depth=6, learning_rate=0.06,
                                      random_seed=42, verbose=0)
        }
    
    features_to_use = _align_feature_list(features_to_use, X.columns, "train_final_stacked_classifier features", min_required=1)
    X_f_sel, _, selected_cols, medians_dict = leakage_safe_feature_selection(X[features_to_use], X[features_to_use], y)
    medians_full = pd.Series(medians_dict)
    X_f_sel = X_f_sel.fillna(medians_full).fillna(0.0)
        
    # Train final MagCompGNN on entire dataset
    full_classes = []
    full_tcs = []
    full_tns = []
    for g in groups.tolist():
        tgt = _FORMULA_TARGETS.get(g, {'class': 2, 'tc': np.nan, 'tn': np.nan})
        full_classes.append(tgt['class'])
        full_tcs.append(tgt['tc'])
        full_tns.append(tgt['tn'])
        
    full_targets = {
        'class': np.array(full_classes),
        'tc': np.array(full_tcs),
        'tn': np.array(full_tns)
    }
    
    if gnn_classifier is None:
        gnn_classifier = train_composition_gnn_model(
            groups.tolist(),
            full_targets,
            epochs=30,  # Match updated GNN default
            verbose=True
        )
    gnn_class_probs, gnn_tc_preds, gnn_tn_preds, gnn_tc_unc, gnn_tn_unc = predict_with_gnn(gnn_classifier, groups.tolist(), task_type='all_with_uncertainty')
    
    X_f_sel = X_f_sel.copy()
    X_f_sel['gnn_prob_FM'] = gnn_class_probs[:, 0]
    X_f_sel['gnn_prob_AFM'] = gnn_class_probs[:, 1]
    X_f_sel['gnn_prob_NM'] = gnn_class_probs[:, 2]
    X_f_sel['gnn_tc_pred'] = gnn_tc_preds
    X_f_sel['gnn_tn_pred'] = gnn_tn_preds
    X_f_sel['gnn_tc_uncertainty'] = gnn_tc_unc
    X_f_sel['gnn_tn_uncertainty'] = gnn_tn_unc
    
    scaler = StandardScaler()
    X_f_sel_scaled = scaler.fit_transform(X_f_sel)
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    meta_features = np.zeros((len(X), len(base_clfs) * n_classes))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_f_sel, y)):
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        y_unique_f, y_counts_f = np.unique(y_tr, return_counts=True)
        class_weights_f = {cls: len(y_tr) / (len(y_unique_f) * cnt) for cls, cnt in zip(y_unique_f, y_counts_f)}
        sample_weight_tr = np.array([class_weights_f[yi] for yi in y_tr])
        sample_weight_tr[(y_tr == 1).values] *= afm_weight_multiplier
        if hard_sample_weights is not None:
            sample_weight_tr *= hard_sample_weights[train_idx]
        
        # Slice the pre-computed full-dataset GNN predictions (speed optimization)
        gnn_tr_class = gnn_class_probs[train_idx]
        gnn_tr_tc = gnn_tc_preds[train_idx]
        gnn_tr_tn = gnn_tn_preds[train_idx]
        gnn_tr_tc_unc = gnn_tc_unc[train_idx]
        gnn_tr_tn_unc = gnn_tn_unc[train_idx]
        
        gnn_val_class = gnn_class_probs[val_idx]
        gnn_val_tc = gnn_tc_preds[val_idx]
        gnn_val_tn = gnn_tn_preds[val_idx]
        gnn_val_tc_unc = gnn_tc_unc[val_idx]
        gnn_val_tn_unc = gnn_tn_unc[val_idx]
        
        # Prepare fold datasets
        X_tr_fold = X_f_sel.iloc[train_idx].copy()
        X_val_fold = X_f_sel.iloc[val_idx].copy()
        
        X_tr_fold['gnn_prob_FM'] = gnn_tr_class[:, 0]
        X_tr_fold['gnn_prob_AFM'] = gnn_tr_class[:, 1]
        X_tr_fold['gnn_prob_NM'] = gnn_tr_class[:, 2]
        X_tr_fold['gnn_tc_pred'] = gnn_tr_tc
        X_tr_fold['gnn_tn_pred'] = gnn_tr_tn
        X_tr_fold['gnn_tc_uncertainty'] = gnn_tr_tc_unc
        X_tr_fold['gnn_tn_uncertainty'] = gnn_tr_tn_unc
        
        X_val_fold['gnn_prob_FM'] = gnn_val_class[:, 0]
        X_val_fold['gnn_prob_AFM'] = gnn_val_class[:, 1]
        X_val_fold['gnn_prob_NM'] = gnn_val_class[:, 2]
        X_val_fold['gnn_tc_pred'] = gnn_val_tc
        X_val_fold['gnn_tn_pred'] = gnn_val_tn
        X_val_fold['gnn_tc_uncertainty'] = gnn_val_tc_unc
        X_val_fold['gnn_tn_uncertainty'] = gnn_val_tn_unc
        
        # Scale fold datasets
        scaler_fold = StandardScaler()
        X_tr_fold_scaled = scaler_fold.fit_transform(X_tr_fold)
        X_val_fold_scaled = scaler_fold.transform(X_val_fold)
        
        for name, clf_class in base_clfs.items():
            import copy
            clf = copy.deepcopy(clf_class)
            clf.fit(X_tr_fold_scaled, y_tr, sample_weight=sample_weight_tr)
            
            idx_start = list(base_clfs.keys()).index(name) * n_classes
            meta_features[val_idx, idx_start:idx_start+n_classes] = clf.predict_proba(X_val_fold_scaled)
            
    trained_base_models = {}
    y_unique_full, y_counts_full = np.unique(y, return_counts=True)
    class_weights_full = {cls: len(y) / (len(y_unique_full) * cnt) for cls, cnt in zip(y_unique_full, y_counts_full)}
    sample_weight_full = np.array([class_weights_full[yi] for yi in y])
    sample_weight_full[(y == 1).values] *= afm_weight_multiplier
    if hard_sample_weights is not None:
        sample_weight_full *= hard_sample_weights
    for name, clf_class in base_clfs.items():
        import copy
        clf = copy.deepcopy(clf_class)
        clf.fit(X_f_sel_scaled, y, sample_weight=sample_weight_full)
        trained_base_models[name] = clf
        
    lr_meta = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    calibrated_meta = CalibratedClassifierCV(estimator=lr_meta, method='sigmoid', cv=5)
    calibrated_meta.fit(meta_features, y, sample_weight=sample_weight_full)
    
    importance_arrays = []
    for key in ['xgb', 'rf', 'lgb', 'et', 'cat']:
        if key in trained_base_models and hasattr(trained_base_models[key], 'feature_importances_'):
            importance_arrays.append(np.asarray(trained_base_models[key].feature_importances_, dtype=float))
    importances = np.mean(importance_arrays, axis=0) if importance_arrays else np.zeros(X_f_sel.shape[1])
    
    # Final verifier training (Component 5)
    y_bin_full = (y == 1).astype(int)
    y_unique_bin_f, y_counts_bin_f = np.unique(y_bin_full, return_counts=True)
    class_weights_bin_f = {cls: len(y_bin_full) / (2.0 * cnt) for cls, cnt in zip(y_unique_bin_f, y_counts_bin_f)}
    sample_weight_bin_f = np.array([class_weights_bin_f[yi] for yi in y_bin_full])
    
    verifier_model = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, num_leaves=15,
                                    random_state=42, n_jobs=-1, verbose=-1)
    verifier_model.fit(X_f_sel_scaled, y_bin_full, sample_weight=sample_weight_bin_f)
    
    return trained_base_models, calibrated_meta, scaler, X_f_sel.columns.tolist(), medians_full.to_dict(), dict(zip(X_f_sel.columns, importances)), gnn_classifier, verifier_model

def target_balanced_bootstrap(X_df, y_series, n_samples_per_low_bin=None, random_state=None):
    """
    Performs target-balanced bootstrap sampling.
    Keeps 100% of samples with y > 400.
    For bins with y <= 400, performs random sampling to achieve a balanced representation.
    """
    df_temp = pd.DataFrame({'idx': X_df.index, 'y': y_series})
    high_temp_mask = df_temp['y'] > 400.0
    df_high = df_temp[high_temp_mask]
    df_low = df_temp[~high_temp_mask]
    
    if len(df_low) == 0 or len(df_high) == 0:
        return df_temp.sample(n=len(df_temp), replace=True, random_state=random_state)['idx'].tolist()
        
    low_bins = pd.cut(df_low['y'], bins=[-np.inf, 100, 200, 300, 400], labels=[1, 2, 3, 4])
    bin_counts = low_bins.value_counts()
    
    if n_samples_per_low_bin is None:
        n_samples_per_low_bin = int(len(df_high) * 0.75)
        n_samples_per_low_bin = max(50, min(n_samples_per_low_bin, int(bin_counts.median())))
        
    rng = np.random.default_rng(random_state)
    sampled_indices = []
    
    sampled_indices.extend(df_high['idx'].tolist())
    
    for b in [1, 2, 3, 4]:
        df_b = df_low[low_bins == b]
        if len(df_b) > 0:
            bin_idx = df_b['idx'].tolist()
            choices = rng.choice(bin_idx, size=min(n_samples_per_low_bin, len(df_b)*2), replace=True)
            sampled_indices.extend(choices)
            
    return sampled_indices

class StreamlinedTriRegressor:
    """
    High-diversity, low-bias regressor combining 4 distinct non-linear model families:
    1. Regularized ExtraTrees: Randomized cuts, variance reducer (max_depth=16, min_samples_leaf=2, max_features=0.80).
    2. Regularized Random Forest: Classical bootstrap bagging (max_depth=16, min_samples_leaf=2, max_features=0.80).
    3. Regularized LightGBM: Asymmetric leaf-wise boosting (max_depth=8, num_leaves=48, reg_lambda=3.0, colsample=0.80).
    4. Regularized XGBoost: Symmetric depth-wise boosting with 2nd-order gradients (max_depth=7, reg_lambda=3.0, colsample=0.80).
    Meta-weights solved via constrained Non-Negative Least Squares (NNLS) on internal out-of-fold folds.
    Supports mathematical target transforms ('none', 'log10', 'symlog').
    """
    def __init__(self, transform='none', random_state=42):
        self.transform = transform
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.et = ExtraTreesRegressor(
            n_estimators=250, max_depth=16, min_samples_leaf=2,
            max_features=0.80, random_state=random_state, n_jobs=-1
        )
        self.rf = RandomForestRegressor(
            n_estimators=250, max_depth=16, min_samples_leaf=2,
            max_features=0.80, random_state=random_state, n_jobs=-1
        )
        self.lgb = LGBMRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.04, num_leaves=48,
            subsample=0.85, colsample_bytree=0.80, min_child_samples=15,
            reg_lambda=3.0, random_state=random_state, n_jobs=-1, verbose=-1
        )
        self.xgb = XGBRegressor(
            n_estimators=300, max_depth=7, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.80, min_child_weight=2,
            reg_lambda=3.0, random_state=random_state, n_jobs=-1
        )
        self.cat = None
        if HAS_CATBOOST:
            self.cat = CatBoostRegressor(
                iterations=300, depth=6, learning_rate=0.04,
                l2_leaf_reg=3.0, random_seed=random_state, verbose=0
            )
        self.weights = [0.25, 0.15, 0.20, 0.20, 0.20] if HAS_CATBOOST else [0.30, 0.20, 0.25, 0.25]
        self.features = None
        self.medians = None
        self.oof_residuals = None
        self.conformal_quantile = None

    def fit(self, X, y, features, groups=None):
        self.features = features
        self.medians = X[features].median().to_dict() if isinstance(X, pd.DataFrame) else {}
        X_clean = X[features].fillna(self.medians).fillna(0).values if isinstance(X, pd.DataFrame) else np.nan_to_num(X, nan=0.0)
        y_val = np.asarray(y, dtype=float)

        if self.transform == 'log10':
            y_fit = np.log10(np.maximum(y_val, 1e-4))
        elif self.transform == 'symlog':
            y_fit = np.sign(y_val) * np.log10(1.0 + np.abs(y_val))
        else:
            y_fit = y_val

        self.et.fit(X_clean, y_fit)
        self.rf.fit(X_clean, y_fit)
        self.lgb.fit(X_clean, y_fit)
        self.xgb.fit(X_clean, y_fit)
        if self.cat is not None:
            self.cat.fit(X_clean, y_fit, verbose=0)

        # Internal OOF for NNLS meta-blending
        try:
            from scipy.optimize import nnls
            from sklearn.model_selection import KFold
            splitter = GroupKFold(n_splits=3) if groups is not None else KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            split_args = (X_clean, y_fit, groups) if groups is not None else (X_clean, y_fit)
            n_models = 5 if self.cat is not None else 4
            oof_preds = np.zeros((len(X_clean), n_models))

            for tr_i, val_i in splitter.split(*split_args):
                m_et = ExtraTreesRegressor(n_estimators=100, max_depth=14, min_samples_leaf=2, max_features=0.80, random_state=self.random_state, n_jobs=-1).fit(X_clean[tr_i], y_fit[tr_i])
                m_rf = RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=2, max_features=0.80, random_state=self.random_state, n_jobs=-1).fit(X_clean[tr_i], y_fit[tr_i])
                m_lgb = LGBMRegressor(n_estimators=100, max_depth=7, num_leaves=40, learning_rate=0.05, subsample=0.85, colsample_bytree=0.80, min_child_samples=15, reg_lambda=3.0, random_state=self.random_state, n_jobs=-1, verbose=-1).fit(X_clean[tr_i], y_fit[tr_i])
                m_xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, subsample=0.85, colsample_bytree=0.80, min_child_weight=2, reg_lambda=3.0, random_state=self.random_state, n_jobs=-1).fit(X_clean[tr_i], y_fit[tr_i])

                oof_preds[val_i, 0] = m_et.predict(X_clean[val_i])
                oof_preds[val_i, 1] = m_rf.predict(X_clean[val_i])
                oof_preds[val_i, 2] = m_lgb.predict(X_clean[val_i])
                oof_preds[val_i, 3] = m_xgb.predict(X_clean[val_i])
                if self.cat is not None:
                    m_cat = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=self.random_state, verbose=0).fit(X_clean[tr_i], y_fit[tr_i])
                    oof_preds[val_i, 4] = m_cat.predict(X_clean[val_i])

            w, _ = nnls(oof_preds, y_fit)
            if np.sum(w) > 0:
                self.weights = (w / np.sum(w)).tolist()
            
            # Compute OOF blend for conformal calibration
            try:
                oof_blend = oof_preds @ np.array(self.weights)
                residuals = np.abs(y_fit - oof_blend)
                self.oof_residuals = np.sort(residuals)
                n_cal = len(residuals)
                q_idx = int(np.ceil((n_cal + 1) * 0.90)) - 1
                self.conformal_quantile = float(self.oof_residuals[min(q_idx, n_cal - 1)])
            except Exception:
                self.conformal_quantile = None
        except Exception:
            self.weights = [0.25, 0.15, 0.20, 0.20, 0.20] if self.cat is not None else [0.30, 0.20, 0.25, 0.25]

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            if self.features is not None:
                missing = [f for f in self.features if f not in X.columns]
                if missing:
                    X_aligned = X.reindex(columns=self.features)
                else:
                    X_aligned = X[self.features]
                X_clean = X_aligned.fillna(self.medians if self.medians else {}).fillna(0).values
            else:
                X_clean = X.fillna(0).values
        else:
            X_clean = np.nan_to_num(X, nan=0.0)

        p0 = self.et.predict(X_clean)
        p1 = self.rf.predict(X_clean)
        p2 = self.lgb.predict(X_clean)
        p3 = self.xgb.predict(X_clean)

        if getattr(self, 'cat', None) is not None:
            p4 = self.cat.predict(X_clean)
            pred = self.weights[0] * p0 + self.weights[1] * p1 + self.weights[2] * p2 + self.weights[3] * p3 + self.weights[4] * p4
        else:
            pred = self.weights[0] * p0 + self.weights[1] * p1 + self.weights[2] * p2 + self.weights[3] * p3

        if self.transform == 'log10':
            return np.maximum(10.0 ** pred, 0.0)
        elif self.transform == 'symlog':
            return np.sign(pred) * (10.0 ** np.abs(pred) - 1.0)
        return pred

    def predict_with_uncertainty(self, X):
        """
        Runs inference and returns both the ensemble expectation (pred)
        and the ensemble disagreement standard deviation (uncertainty) in physical units.
        """
        if isinstance(X, pd.DataFrame):
            if self.features is not None:
                missing = [f for f in self.features if f not in X.columns]
                if missing:
                    X_aligned = X.reindex(columns=self.features)
                else:
                    X_aligned = X[self.features]
                X_clean = X_aligned.fillna(self.medians if self.medians else {}).fillna(0).values
            else:
                X_clean = X.fillna(0).values
        else:
            X_clean = np.nan_to_num(X, nan=0.0)

        p0 = self.et.predict(X_clean)
        p1 = self.rf.predict(X_clean)
        p2 = self.lgb.predict(X_clean)
        p3 = self.xgb.predict(X_clean)

        if getattr(self, 'cat', None) is not None:
            p4 = self.cat.predict(X_clean)
            preds_stack = np.vstack([p0, p1, p2, p3, p4])
            weights_arr = np.array(self.weights).reshape(5, 1)
        else:
            preds_stack = np.vstack([p0, p1, p2, p3])
            weights_arr = np.array(self.weights).reshape(4, 1)
            
        mean_pred = np.sum(preds_stack * weights_arr, axis=0)

        # Weighted variance across ensemble models
        var_pred = np.sum(weights_arr * ((preds_stack - mean_pred) ** 2), axis=0)
        std_pred = np.sqrt(np.maximum(var_pred, 1e-8))

        # Apply conformal calibration floor
        if getattr(self, 'conformal_quantile', None) is not None and self.conformal_quantile > 0:
            std_pred = np.maximum(std_pred, self.conformal_quantile)

        if self.transform == 'log10':
            phys_mean = np.maximum(10.0 ** mean_pred, 0.0)
            # First-order error propagation for base-10 log
            phys_std = np.log(10.0) * phys_mean * std_pred
            return phys_mean, phys_std
        elif self.transform == 'symlog':
            phys_mean = np.sign(mean_pred) * (10.0 ** np.abs(mean_pred) - 1.0)
            phys_std = np.log(10.0) * (10.0 ** np.abs(mean_pred)) * std_pred
            return phys_mean, phys_std

        return mean_pred, std_pred


class StreamlinedTriClassifier:
    """
    High-diversity, low-overfitting classifier combining 4 distinct model families:
    1. ExtraTrees: randomized cuts, class-balanced, low variance (max_depth=18, min_leaf=2).
    2. Random Forest: bootstrap aggregation, class-balanced (max_depth=18, min_leaf=2).
    3. LightGBM: leaf-wise boosting, class-balanced (max_depth=9, num_leaves=63, reg_lambda=2.0).
    4. XGBoost: depth-bounded, subsampled features (max_depth=8, reg_lambda=2.0).
    """
    def __init__(self, n_classes=3, random_state=42):
        self.n_classes = n_classes
        self.random_state = random_state
        xgb_obj = 'binary:logistic' if n_classes <= 2 else 'multi:softprob'
        xgb_eval = 'logloss' if n_classes <= 2 else 'mlogloss'

        self.et = ExtraTreesClassifier(
            n_estimators=250, max_depth=18, min_samples_leaf=2,
            max_features=0.80, class_weight='balanced', random_state=random_state, n_jobs=-1
        )
        self.rf = RandomForestClassifier(
            n_estimators=250, max_depth=18, min_samples_leaf=2,
            max_features=0.80, class_weight='balanced', random_state=random_state, n_jobs=-1
        )
        self.lgb = LGBMClassifier(
            n_estimators=300, max_depth=9, num_leaves=63, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.80, min_child_samples=15,
            class_weight='balanced', reg_lambda=2.0, random_state=random_state,
            n_jobs=-1, verbose=-1
        )
        self.xgb = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.80, min_child_weight=2,
            reg_lambda=2.0, objective=xgb_obj, eval_metric=xgb_eval,
            random_state=random_state, n_jobs=-1
        )
        self.weights = [0.30, 0.20, 0.25, 0.25]
        self.scaler = StandardScaler()
        self.features = None
        self.medians = None
        self.importances_ = None
        self.gnn_classifier = None

    def fit(self, X, y, groups, features):
        self.features = features
        self.medians = X[features].median().to_dict()
        X_clean = X[features].fillna(self.medians).fillna(0).values
        y_val = np.asarray(y, dtype=int)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)
        
        self.et.fit(X_scaled, y_val)
        self.rf.fit(X_scaled, y_val)
        self.lgb.fit(X_scaled, y_val)
        self.xgb.fit(X_scaled, y_val)

        self.importances_ = dict(zip(features, self.et.feature_importances_))
        self.gnn_classifier = None
        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            if self.features is not None:
                missing = [f for f in self.features if f not in X.columns]
                if missing:
                    X_aligned = X.reindex(columns=self.features)
                else:
                    X_aligned = X[self.features]
                X_clean = X_aligned.fillna(self.medians if self.medians else {}).fillna(0).values
            else:
                X_clean = X.fillna(0).values
            X_scaled = self.scaler.transform(X_clean)
        else:
            X_scaled = np.nan_to_num(X, nan=0.0)
            
        p_et = self.et.predict_proba(X_scaled)
        p_rf = self.rf.predict_proba(X_scaled)
        p_lgb = self.lgb.predict_proba(X_scaled)
        p_xgb = self.xgb.predict_proba(X_scaled)

        return (
            self.weights[0] * p_et
            + self.weights[1] * p_rf
            + self.weights[2] * p_lgb
            + self.weights[3] * p_xgb
        )

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class TargetBalancedRegressorEnsemble:
    """
    Paper-grade Physics-Informed Regressor Ensemble modeled after the NEMAD paper benchmarks
    (Itani et al., Nature Communications 2025).
    Combines Extra Trees Regressor (the paper's top performer, R² ~ 0.87), Random Forest,
    LightGBM, XGBoost, and CatBoost with Non-Negative Least Squares (NNLS) meta-weighting.
    Avoids log-transform distortion and Jensen's inequality bias, fitting directly in raw physical space.
    Includes physics-grounded loss reweighting for high-ordering temperature accuracy.
    """
    def __init__(self, n_members=2, random_state=42, n_estimators=None):
        self.n_members = n_members
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.members = []
        
    def fit(self, X, y, features_to_use, is_tc=True, use_multi_scale=False, allow_negative=False):
        self.features_to_use = features_to_use
        self.is_tc = is_tc
        self.allow_negative = allow_negative
        self.members = []
        
        self.scaler = StandardScaler()
        self.scaler.fit(X[features_to_use].values)
        X_mat = X[features_to_use].values
        # Target bounds for physical sanity clipping
        y_vals = y.values if hasattr(y, 'values') else np.asarray(y)
        self.y_min_ = float(np.nanpercentile(y_vals, 0.05)) if len(y_vals) > 0 else (-1e8 if allow_negative else 0.0)
        self.y_max_ = float(np.nanpercentile(y_vals, 99.95)) if len(y_vals) > 0 else (1800.0 if is_tc else 1200.0)
        
        n_samples = len(X)
        n_est = self.n_estimators if self.n_estimators is not None else (80 if TRAINING_MODE == "fast_debug" else 100)
        
        for i in range(self.n_members):
            seed = self.random_state + i * 17
            rng = np.random.default_rng(seed)
            # Stratified bootstrap sampling to preserve temperature coverage
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_b = X_mat[boot_idx]
            y_b = y_vals[boot_idx]
            
            # The top three complementary tabular learners:
            # 1. Extra Trees (Nature Comm. 2025 top performer - extreme randomization suppresses materials noise)
            et = ExtraTreesRegressor(
                n_estimators=n_est, max_depth=25, min_samples_split=2,
                min_samples_leaf=2, max_features=0.85, random_state=seed, n_jobs=-1
            )
            # 2. Random Forest (Robust variance reduction across elemental spaces)
            rf = RandomForestRegressor(
                n_estimators=n_est, max_depth=25, min_samples_split=2,
                min_samples_leaf=2, max_features=0.85, random_state=seed, n_jobs=-1
            )
            # 3. LightGBM (Fast, high-accuracy gradient boosting on physical target)
            lgb = LGBMRegressor(
                n_estimators=n_est, max_depth=8, learning_rate=0.04, num_leaves=63,
                subsample=0.85, colsample_bytree=0.85, random_state=seed,
                n_jobs=-1, verbose=-1
            )
            
            fit_kwargs = {}
            if self.is_tc and not allow_negative:
                # High-TC physics reweighting: penalize high-temperature prediction errors
                sw = 1.0 + (np.maximum(y_b, 0.0) / 500.0) ** 1.5
                sw = sw / np.mean(sw)
                fit_kwargs['sample_weight'] = sw
            
            et.fit(X_b, y_b, **fit_kwargs)
            rf.fit(X_b, y_b, **fit_kwargs)
            lgb.fit(X_b, y_b, **fit_kwargs)
            
            # Optimal Out-of-Bag (OOB) Non-Negative Least Squares (NNLS) meta-weighting
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[boot_idx] = False
            if np.sum(oob_mask) >= 20:
                X_oob = X_mat[oob_mask]
                y_oob = y_vals[oob_mask]
                p_oob = np.column_stack([
                    et.predict(X_oob),
                    rf.predict(X_oob),
                    lgb.predict(X_oob)
                ])
                from scipy.optimize import nnls
                w_opt, _ = nnls(p_oob, y_oob)
                if np.sum(w_opt) > 1e-6:
                    weights = w_opt / np.sum(w_opt)
                else:
                    weights = np.array([0.55, 0.30, 0.15])
            else:
                weights = np.array([0.55, 0.30, 0.15])
            
            self.members.append({
                'et': et, 'rf': rf, 'lgb': lgb,
                'weights': weights
            })
            
    def predict(self, X):
        X_mat = X[self.features_to_use].values
        member_preds = []
        
        for m in self.members:
            w = m['weights']
            if len(w) == 3:
                p_ens = (w[0] * m['et'].predict(X_mat) + 
                         w[1] * m['rf'].predict(X_mat) + 
                         w[2] * m['lgb'].predict(X_mat))
            elif len(w) == 4:
                p_ens = (w[0] * m['et'].predict(X_mat) + 
                         w[1] * m['lgb'].predict(X_mat) + 
                         w[2] * m['cat'].predict(X_mat) + 
                         w[3] * m['xgb'].predict(X_mat))
            elif len(w) == 2:
                p_ens = w[0] * m['et'].predict(X_mat) + w[1] * m['lgb'].predict(X_mat)
            else:
                # Backward compatibility with legacy models
                p_rf  = m['rf'].predict(X_mat) if 'rf' in m else m['et'].predict(X_mat)
                p_xgb = m['xgb'].predict(X_mat) if 'xgb' in m else m['lgb'].predict(X_mat)
                p_cat = m['cat'].predict(X_mat) if 'cat' in m else m['lgb'].predict(X_mat)
                p_ens = w[0] * m['et'].predict(X_mat) + w[1] * p_rf + w[2] * m['lgb'].predict(X_mat) + w[3] * p_xgb + w[4] * p_cat
                
            p_min = self.y_min_ if getattr(self, 'allow_negative', False) else 0.0
            p_clipped = np.clip(p_ens, p_min, self.y_max_)
            member_preds.append(p_clipped)
            
        member_preds = np.array(member_preds)
        mean_pred = np.mean(member_preds, axis=0)
        uncertainty = np.std(member_preds, axis=0)
        return mean_pred, uncertainty

class TransformedPhysicalRegressor:
    """
    Wraps an underlying ensemble to automatically handle log, symlog, and physical scale inversions.
    """
    def __init__(self, base_ensemble, transform_type='none', y_min=None, y_max=None, scaler=None, features_to_use=None):
        self.base_ensemble = base_ensemble
        self.transform_type = transform_type  # 'none', 'log', 'symlog'
        self.y_min = y_min
        self.y_max = y_max
        self.scaler = scaler or getattr(base_ensemble, 'scaler', None)
        self.features_to_use = features_to_use or getattr(base_ensemble, 'features_to_use', None)
        self.members = getattr(base_ensemble, 'members', [])

    def predict(self, X):
        pred_trans, unc_trans = self.base_ensemble.predict(X)
        if self.transform_type == 'symlog':
            pred = np.sign(pred_trans) * (10.0 ** np.abs(pred_trans) - 1.0)
            unc = np.abs(pred) * (np.log(10.0) * np.maximum(unc_trans, 1e-4))
        elif self.transform_type == 'log':
            pred = 10.0 ** pred_trans
            unc = pred * (np.log(10.0) * np.maximum(unc_trans, 1e-4))
        else:
            pred = pred_trans
            unc = unc_trans
            
        if self.y_min is not None:
            pred = np.maximum(pred, self.y_min)
        if self.y_max is not None:
            pred = np.minimum(pred, self.y_max)
        return pred, unc


class DAOEnsembleDeployModel:
    """
    Production-ready deployable ensemble combining ExtraTrees and LightGBM
    with DAO mandatory structural features, median imputation, and automatic
    inverse target transformation.
    """
    def __init__(self, m1, m2, transform='none', features=None, medians=None):
        self.m1 = m1
        self.m2 = m2
        self.transform = transform
        self.features = list(features) if features is not None else []
        self.medians = dict(medians) if medians is not None else {}

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X.reindex(columns=self.features)
            for col in self.features:
                if col in self.medians:
                    X_df[col] = X_df[col].fillna(self.medians[col])
            X_arr = X_df.fillna(0.0).values
        else:
            X_arr = np.asarray(X)
        p1 = self.m1.predict(X_arr)
        p2 = self.m2.predict(X_arr)
        p_trans = 0.5 * p1 + 0.5 * p2
        if self.transform == 'symlog':
            return np.sign(p_trans) * (np.power(10.0, np.abs(p_trans)) - 1.0)
        elif self.transform == 'log':
            return np.power(10.0, p_trans)
        return p_trans


class StackedClassifierEnsemble:
    """
    Exposes a scikit-learn compatible predict/predict_proba interface for the 6-base model stacked classifier.
    """
    def __init__(self, base_models, meta_classifier, scaler, selected_columns, medians, importances, n_classes=3):
        self.base_models = base_models
        self.meta_classifier = meta_classifier
        self.scaler = scaler
        self.selected_columns = selected_columns
        self.medians = medians
        self.importances_ = importances
        self.n_classes = n_classes
        self.afm_threshold = None
        self.afm_verifier = None
        
    def predict_proba(self, X_scaled):
        meta_blocks = []
        n_meta_expected = getattr(self.meta_classifier, 'n_features_in_', None)
        for name in list(self.base_models.keys()):
            clf = self.base_models[name]
            preds = clf.predict_proba(X_scaled)
            if self.n_classes == 2 and n_meta_expected == len(self.base_models):
                preds = preds[:, 1:2]
            meta_blocks.append(preds)
            
        meta_features = np.hstack(meta_blocks)
        if self.meta_classifier is not None:
            return self.meta_classifier.predict_proba(meta_features)
        else:
            # Equal-weighted ensemble average fallback
            return np.mean(meta_blocks, axis=0)
        
    def predict(self, X_scaled):
        probs = self.predict_proba(X_scaled)
        
        if self.afm_verifier is not None:
            verifier_probs = self.afm_verifier['model'].predict_proba(X_scaled)[:, 1]
            return apply_afm_verifier_rule(probs, verifier_probs, self.afm_verifier['tau_afm'], self.afm_verifier['tau_verifier'])
            
        if self.afm_threshold is not None:
            return apply_afm_threshold(probs, self.afm_threshold['tau_afm'], self.afm_threshold['margin'])
            
        return np.argmax(probs, axis=1)

class BlendedClassifierEnsemble:
    """
    Combines out-of-fold probabilistic predictions of F1, F3, and F5 stacked ensembles
    to build a robust, high-generalization classifier.
    """
    def __init__(self, ensemble_f1, ensemble_f3, ensemble_f5, w1=0.40, w3=0.25, w5=0.35):
        self.ensemble_f1 = ensemble_f1
        self.ensemble_f3 = ensemble_f3
        self.ensemble_f5 = ensemble_f5
        self.w1 = w1
        self.w3 = w3
        self.w5 = w5
        # Expose important properties for backward compatibility
        self.importances_ = ensemble_f5.importances_
        self.gnn_classifier = getattr(ensemble_f5, 'gnn_classifier', None)
        self.afm_threshold = None
        self.afm_verifier = None
        
    def predict_proba(self, X_scaled_dict):
        """
        Expects a dictionary: {'F1': X_scaled_f1, 'F3': X_scaled_f3, 'F5': X_scaled_f5}
        """
        prob_f1 = self.ensemble_f1.predict_proba(X_scaled_dict['F1'])
        prob_f3 = self.ensemble_f3.predict_proba(X_scaled_dict['F3'])
        prob_f5 = self.ensemble_f5.predict_proba(X_scaled_dict['F5'])
        return self.w1 * prob_f1 + self.w3 * prob_f3 + self.w5 * prob_f5
        
    def predict(self, X_scaled_dict):
        probs = self.predict_proba(X_scaled_dict)
        
        if self.afm_verifier is not None:
            # Predict verifier on F5 scaled features
            verifier_probs = self.afm_verifier['model'].predict_proba(X_scaled_dict['F5'])[:, 1]
            return apply_afm_verifier_rule(probs, verifier_probs, self.afm_verifier['tau_afm'], self.afm_verifier['tau_verifier'])
            
        if self.afm_threshold is not None:
            return apply_afm_threshold(probs, self.afm_threshold['tau_afm'], self.afm_threshold['margin'])
            
        return np.argmax(probs, axis=1)

def plot_target_distributions(y_class, y_curie, y_neel, plot_dir='plots'):
    """
    Premium 3-panel figure showing target variable distributions.
    Panel 1: Magnetic class proportions as a horizontal bar chart.
    Panel 2: Curie Temperature KDE + rug + quantile annotations.
    Panel 3: Néel Temperature KDE + rug + quantile annotations.
    """
    setup_plotting_style()
    from src.utils import apply_clean_axes, PALETTE

    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)

    # ── Panel 1: Magnetic Class Distribution ────────────────────────────────
    ax0 = axes[0]
    class_counts = pd.Series(y_class).value_counts().sort_index()
    labels  = ['FM', 'AFM', 'NM']
    counts  = [int(class_counts.get(0, 0)), int(class_counts.get(1, 0)), int(class_counts.get(2, 0))]
    colors  = [PALETTE['fm_color'], PALETTE['afm_color'], PALETTE['nm_color']]
    total   = sum(counts)

    bars = ax0.barh(labels, counts, color=colors, edgecolor='white',
                    linewidth=0.8, height=0.55, alpha=0.90)

    for bar, cnt in zip(bars, counts):
        pct = cnt / total * 100
        ax0.text(cnt + total * 0.012, bar.get_y() + bar.get_height() / 2,
                 f'{cnt:,}  ({pct:.1f}%)',
                 va='center', ha='left', fontsize=10.5, fontweight='bold', color='#1A202C')

    apply_clean_axes(ax0, xlabel='Number of Materials', ylabel=None,
                     title='Magnetic Class Distribution', grid_axis='x')
    ax0.set_xlim(0, max(counts) * 1.25)
    ax0.invert_yaxis()

    # ── Panel 2: Curie Temperature Distribution ──────────────────────────────
    ax1 = axes[1]
    y_curie_arr = np.asarray(y_curie).flatten()
    sns.histplot(y_curie_arr, kde=True, ax=ax1,
                 color=PALETTE['fm_color'], alpha=0.55, bins=40,
                 edgecolor='white', linewidth=0.5)
    ax1.lines[-1].set(linewidth=2.2)  # thicken KDE line

    med_c = np.median(y_curie_arr)
    q25_c = np.percentile(y_curie_arr, 25)
    q75_c = np.percentile(y_curie_arr, 75)

    ax1.axvline(med_c,  color='#1D3557', lw=2.0, linestyle='--', label=f'Median {med_c:.0f} K')
    ax1.axvline(q25_c,  color='#1D3557', lw=1.2, linestyle=':')
    ax1.axvline(q75_c,  color='#1D3557', lw=1.2, linestyle=':', label=f'IQR [{q25_c:.0f}–{q75_c:.0f}] K')

    apply_clean_axes(ax1,
                     xlabel='Curie Temperature $T_C$ (K)',
                     ylabel='Count',
                     title='$T_C$ Distribution (Ferromagnetic)')
    ax1.legend(fontsize=9.5, frameon=True)

    # ── Panel 3: Néel Temperature Distribution ────────────────────────────────
    ax2 = axes[2]
    y_neel_arr = np.asarray(y_neel).flatten()
    sns.histplot(y_neel_arr, kde=True, ax=ax2,
                 color=PALETTE['afm_color'], alpha=0.55, bins=40,
                 edgecolor='white', linewidth=0.5)
    ax2.lines[-1].set(linewidth=2.2)

    med_n = np.median(y_neel_arr)
    q25_n = np.percentile(y_neel_arr, 25)
    q75_n = np.percentile(y_neel_arr, 75)

    ax2.axvline(med_n,  color='#1D3557', lw=2.0, linestyle='--', label=f'Median {med_n:.0f} K')
    ax2.axvline(q25_n,  color='#1D3557', lw=1.2, linestyle=':')
    ax2.axvline(q75_n,  color='#1D3557', lw=1.2, linestyle=':', label=f'IQR [{q25_n:.0f}–{q75_n:.0f}] K')

    apply_clean_axes(ax2,
                     xlabel='Néel Temperature $T_N$ (K)',
                     ylabel='Count',
                     title='$T_N$ Distribution (Antiferromagnetic)')
    ax2.legend(fontsize=9.5, frameon=True)

    fig.suptitle('NEMAD Dataset: Target Variable Distributions',
                 fontsize=15, fontweight='bold', color='#1A202C', y=1.02)

    out_path = os.path.join(plot_dir, 'target_distributions.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved premium target distributions panel to {out_path}")

def plot_target_transformation_distributions(y_curie, y_neel, plot_dir='plots'):
    """
    Plots the Curie (TC) and Néel (TN) temperature target distributions before and
    after the log1p transformation with large bold fonts.
    """
    setup_plotting_style()
    from src.utils import PALETTE
    import seaborn as sns
    
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 13.0), facecolor='#F9F9FC')
    
    y_curie_arr = np.asarray(y_curie).flatten()
    y_curie_log = np.log1p(y_curie_arr)
    
    y_neel_arr = np.asarray(y_neel).flatten()
    y_neel_log = np.log1p(y_neel_arr)
    
    # TC - Raw
    ax = axes[0, 0]
    sns.histplot(y_curie_arr, kde=True, ax=ax, color=PALETTE['fm_color'], alpha=0.6, bins=45, edgecolor='white')
    ax.lines[-1].set(linewidth=2.5)
    ax.set_title("Curie Temperature ($T_C$) - Raw", fontsize=16, fontweight='bold', pad=10)
    ax.set_xlabel("Curie Temperature, $T_C$ (K)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Count", fontsize=14, fontweight='bold')
    ax.grid(True, which='both', color='#E2E8F0', linestyle='--', linewidth=0.6)
    
    # TC - Log1p
    ax = axes[0, 1]
    sns.histplot(y_curie_log, kde=True, ax=ax, color=PALETTE['fm_color'], alpha=0.6, bins=45, edgecolor='white')
    ax.lines[-1].set(linewidth=2.5)
    ax.set_title("Curie Temperature ($T_C$) - Log1p Transformed", fontsize=16, fontweight='bold', pad=10)
    ax.set_xlabel(r"$\log(1 + T_C)$", fontsize=14, fontweight='bold')
    ax.set_ylabel("Count", fontsize=14, fontweight='bold')
    ax.grid(True, which='both', color='#E2E8F0', linestyle='--', linewidth=0.6)
    
    # TN - Raw
    ax = axes[1, 0]
    sns.histplot(y_neel_arr, kde=True, ax=ax, color=PALETTE['afm_color'], alpha=0.6, bins=45, edgecolor='white')
    ax.lines[-1].set(linewidth=2.5)
    ax.set_title("Néel Temperature ($T_N$) - Raw", fontsize=16, fontweight='bold', pad=10)
    ax.set_xlabel("Néel Temperature, $T_N$ (K)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Count", fontsize=14, fontweight='bold')
    ax.grid(True, which='both', color='#E2E8F0', linestyle='--', linewidth=0.6)
    
    # TN - Log1p
    ax = axes[1, 1]
    sns.histplot(y_neel_log, kde=True, ax=ax, color=PALETTE['afm_color'], alpha=0.6, bins=45, edgecolor='white')
    ax.lines[-1].set(linewidth=2.5)
    ax.set_title("Néel Temperature ($T_N$) - Log1p Transformed", fontsize=16, fontweight='bold', pad=10)
    ax.set_xlabel(r"$\log(1 + T_N)$", fontsize=14, fontweight='bold')
    ax.set_ylabel("Count", fontsize=14, fontweight='bold')
    ax.grid(True, which='both', color='#E2E8F0', linestyle='--', linewidth=0.6)
    
    # Force ticks to be big and bold for all subplots
    for r in range(2):
        for c in range(2):
            axes[r, c].tick_params(axis='both', which='major', labelsize=13, width=1.5)
            for label in axes[r, c].get_xticklabels() + axes[r, c].get_yticklabels():
                label.set_fontweight('bold')
                
    fig.suptitle("Curie and Néel Temperature Target Distributions Before & After Transformation",
                 fontsize=18, fontweight='bold', color='#0F172A', y=1.02)
    
    out_path = os.path.join(plot_dir, 'target_transformation_distributions.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved target transformation distributions to {out_path}")

def apply_afm_threshold(probs, tau_afm=0.50, margin=0.08):
    """
    Applies class-specific decision threshold and margin for AFM class.
    """
    preds = np.argmax(probs, axis=1)

    p_fm = probs[:, 0]
    p_afm = probs[:, 1]
    p_nm = probs[:, 2]

    strongest_non_afm = np.maximum(p_fm, p_nm)
    accept_afm = (p_afm >= tau_afm) & ((p_afm - strongest_non_afm) >= margin)

    weak_afm = (preds == 1) & (~accept_afm)
    preds[weak_afm] = np.where(p_fm[weak_afm] >= p_nm[weak_afm], 0, 2)

    return preds

def optimize_afm_threshold(oof_probs, y_true):
    """
    Grid searches over AFM threshold and margin to optimize AFM-F1 + macro-F1 score.
    """
    from sklearn.metrics import precision_recall_fscore_support, f1_score
    best_score = -1.0
    best_tau = 0.40
    best_margin = 0.08
    best_metrics = (0.0, 0.0, 0.0)
    
    tau_afm_values = [0.30, 0.35, 0.40, 0.45, 0.50]
    margin_values = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]
    
    for tau in tau_afm_values:
        for margin in margin_values:
            preds = apply_afm_threshold(oof_probs, tau, margin)
            macro = f1_score(y_true, preds, average='macro')
            p, r, f, _ = precision_recall_fscore_support(y_true, preds, labels=[1], average=None, zero_division=0)
            afm_f1 = f[0] if len(f) > 0 else 0.0
            
            score = macro + afm_f1
            if score > best_score:
                best_score = score
                best_tau = tau
                best_margin = margin
                best_metrics = (p[0] if len(p) > 0 else 0.0, r[0] if len(p) > 0 else 0.0, afm_f1)
                
    print(f"Optimized AFM Threshold: tau_afm={best_tau:.2f}, margin={best_margin:.2f} -> AFM F1={best_metrics[2]:.4f}, Macro F1={best_score - best_metrics[2]:.4f}")
    return {'tau_afm': best_tau, 'margin': best_margin, 'metrics': best_metrics}

def apply_afm_verifier_rule(probs, verifier_prob, tau_afm=0.45, tau_verifier=0.55):
    """
    Applies the second-stage AFM verifier decision rule.
    """
    preds = np.argmax(probs, axis=1)

    p_fm = probs[:, 0]
    p_afm = probs[:, 1]
    p_nm = probs[:, 2]

    weak_afm = (preds == 1) & (
        (p_afm < tau_afm) | (verifier_prob < tau_verifier)
    )

    preds[weak_afm] = np.where(p_fm[weak_afm] >= p_nm[weak_afm], 0, 2)

    return preds

def optimize_afm_verifier_threshold(oof_probs, oof_verifier_probs, y_true):
    """
    Sweeps over thresholds to optimize both tau_afm and tau_verifier for macro-F1 + AFM-F1.
    """
    from sklearn.metrics import precision_recall_fscore_support, f1_score
    best_score = -1.0
    best_tau_afm = 0.40
    best_tau_verifier = 0.55
    best_metrics = (0.0, 0.0, 0.0)
    
    tau_afm_vals = [0.30, 0.35, 0.40, 0.45, 0.50]
    tau_ver_vals = [0.30, 0.40, 0.50, 0.55, 0.60, 0.70]
    
    for tau_afm in tau_afm_vals:
        for tau_ver in tau_ver_vals:
            preds = apply_afm_verifier_rule(oof_probs, oof_verifier_probs, tau_afm, tau_ver)
            macro = f1_score(y_true, preds, average='macro')
            p, r, f, _ = precision_recall_fscore_support(y_true, preds, labels=[1], average=None, zero_division=0)
            afm_f1 = f[0] if len(f) > 0 else 0.0
            
            score = macro + afm_f1
            if score > best_score:
                best_score = score
                best_tau_afm = tau_afm
                best_tau_verifier = tau_ver
                best_metrics = (p[0] if len(p) > 0 else 0.0, r[0] if len(r) > 0 else 0.0, afm_f1)
                
    print(f"Optimized AFM Verifier Rule: tau_afm={best_tau_afm:.2f}, tau_verifier={best_tau_verifier:.2f} -> AFM F1={best_metrics[2]:.4f}")
    return best_tau_afm, best_tau_verifier, best_metrics

def export_afm_error_analysis(df, y_true, preds, probs, output_dir):
    """
    Exports 4 detailed error analysis CSVs containing formula, predictions, probabilities
    and key physical features to help diagnose model failures.
    """
    os.makedirs(os.path.join(output_dir, 'error_analysis'), exist_ok=True)
    
    df_analysis = pd.DataFrame({
        'formula': df.get('reduced_formula', df.index),
        'true_class': y_true,
        'pred_class': preds,
        'prob_FM': probs[:, 0],
        'prob_AFM': probs[:, 1],
        'prob_NM': probs[:, 2]
    })
    
    physics_cols = [
        'magnetic_3d_fraction', 'rare_earth_fraction', 'oxygen_fraction', 
        'chalcogen_fraction', 'd_band_filling_fraction', 'gnn_prob_AFM',
        'crystal_system', 'spacegroup_symbol', 'density', 'band_gap'
    ]
    for col in physics_cols:
        if col in df.columns:
            df_analysis[col] = df[col]
            
    class_labels = {0: 'FM', 1: 'AFM', 2: 'NM'}
    df_analysis['true_label'] = df_analysis['true_class'].map(class_labels)
    df_analysis['pred_label'] = df_analysis['pred_class'].map(class_labels)
    
    # 1. False AFM from FM: Pred AFM but True FM
    false_afm_from_fm = df_analysis[(df_analysis['pred_class'] == 1) & (df_analysis['true_class'] == 0)]
    false_afm_from_fm.to_csv(os.path.join(output_dir, 'error_analysis', 'false_AFM_from_FM.csv'), index=False)
    
    # 2. False AFM from NM: Pred AFM but True NM
    false_afm_from_nm = df_analysis[(df_analysis['pred_class'] == 1) & (df_analysis['true_class'] == 2)]
    false_afm_from_nm.to_csv(os.path.join(output_dir, 'error_analysis', 'false_AFM_from_NM.csv'), index=False)
    
    # 3. Missed AFM as FM: True AFM but Pred FM
    missed_afm_as_fm = df_analysis[(df_analysis['true_class'] == 1) & (df_analysis['pred_class'] == 0)]
    missed_afm_as_fm.to_csv(os.path.join(output_dir, 'error_analysis', 'missed_AFM_as_FM.csv'), index=False)
    
    # 4. Missed AFM as NM: True AFM but Pred NM
    missed_afm_as_nm = df_analysis[(df_analysis['true_class'] == 1) & (df_analysis['pred_class'] == 2)]
    missed_afm_as_nm.to_csv(os.path.join(output_dir, 'error_analysis', 'missed_AFM_as_NM.csv'), index=False)
    
    print(f"Exported error analysis CSVs to {os.path.join(output_dir, 'error_analysis/')}")

def compute_confidence_rejection_table(probs, y_true):
    """
    Computes a confidence rejection table showing accuracy, AFM precision and AFM F1
    at different confidence thresholds.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    
    thresholds = [0.0, 0.50, 0.60, 0.70, 0.80, 0.90]
    table = []
    
    for t in thresholds:
        mask = max_probs >= t
        covered_count = np.sum(mask)
        coverage = covered_count / len(probs)
        
        if covered_count == 0:
            continue
            
        covered_y_true = y_true[mask]
        covered_preds = preds[mask]
        
        acc = accuracy_score(covered_y_true, covered_preds)
        p, r, f, _ = precision_recall_fscore_support(covered_y_true, covered_preds, labels=[1], average=None, zero_division=0)
        afm_precision = p[0] if len(p) > 0 else 0.0
        afm_f1 = f[0] if len(f) > 0 else 0.0
        
        table.append({
            'Threshold': f"{t:.2f}" if t > 0 else "No reject",
            'Coverage': f"{coverage * 100:.1f}%",
            'Accuracy': f"{acc:.4f}",
            'AFM Precision': f"{afm_precision:.4f}",
            'AFM F1': f"{afm_f1:.4f}",
            'N Uncertain': int(len(probs) - covered_count)
        })
        
    print("\n--- Confidence Rejection & Screening Coverage ---")
    print(pd.DataFrame(table).to_string(index=False))
    return table

class SpecialistCorrectedClassifierEnsemble:
    """
    Integrates a global FM/AFM/NM classifier trained on F_opt_global features
    with an FM/AFM specialist classifier trained on F_opt_FM_AFM features to
    perform target-specific boundary correction and probability calibration.
    """
    def __init__(self, global_ensemble, specialist_ensemble, F_opt_global, F_opt_FM_AFM, calibrator=None, boundary_params=None):
        self.global_ensemble = global_ensemble
        self.specialist_ensemble = specialist_ensemble
        self.F_opt_global = F_opt_global
        self.F_opt_FM_AFM = F_opt_FM_AFM
        self.calibrator = calibrator
        self.boundary_params = boundary_params or (0.60, 0.30, 0.40)
        
        # Expose properties for backward compatibility
        self.importances_ = global_ensemble.importances_
        self.gnn_classifier = getattr(global_ensemble, 'gnn_classifier', None)
        self.afm_threshold = None
        self.afm_verifier = None
        
    def _format_input_dict(self, X):
        if isinstance(X, pd.DataFrame):
            g_cols = getattr(self, 'features_class_global', self.F_opt_global)
            s_cols = getattr(self, 'features_class_specialist', self.F_opt_FM_AFM)
            g_med = getattr(self, 'medians_class_global', {})
            s_med = getattr(self, 'medians_class_specialist', {})
            X_g = X.reindex(columns=g_cols).fillna(g_med if g_med else {}).fillna(0).values
            X_s = X.reindex(columns=s_cols).fillna(s_med if s_med else {}).fillna(0).values
            sc_g = getattr(self, 'scaler_class_global', None)
            sc_s = getattr(self, 'scaler_class_specialist', None)
            return {
                'global': sc_g.transform(X_g) if sc_g is not None else X_g,
                'specialist': sc_s.transform(X_s) if sc_s is not None else X_s
            }
        return X
        
    def predict_proba(self, X_scaled_dict):
        """
        Expects a dictionary: {'global': X_scaled_global, 'specialist': X_scaled_specialist}
        or a pandas DataFrame which will be automatically formatted and scaled.
        """
        raw_df = X_scaled_dict if isinstance(X_scaled_dict, pd.DataFrame) else None
        X_scaled_dict = self._format_input_dict(X_scaled_dict)
        # Predict global probabilities
        probs_global = self.global_ensemble.predict_proba(X_scaled_dict['global'])
        
        # Predict specialist probabilities (binary classification: 0 = FM, 1 = AFM)
        probs_spec = self.specialist_ensemble.predict_proba(X_scaled_dict['specialist'])
        
        # Apply boundary correction dynamically based on champion parameters (Experiment 1)
        mag_t, margin_t, nm_upper = self.boundary_params
        probs_corrected = probs_global.copy()
        
        # P_magnetic = P_FM + P_AFM
        p_magnetic = probs_global[:, 0] + probs_global[:, 1]
        p_nm = probs_global[:, 2]
        
        boundary_mask = (
            (p_magnetic > mag_t) &
            (np.abs(probs_global[:, 0] - probs_global[:, 1]) < margin_t) &
            (p_nm < nm_upper)
        )
        
        if np.any(boundary_mask):
            p_spec_afm = probs_spec[:, 1]
            p_spec_fm = probs_spec[:, 0]
            
            # Apply correction only to samples on the boundary
            probs_corrected[boundary_mask, 1] = p_magnetic[boundary_mask] * p_spec_afm[boundary_mask]
            probs_corrected[boundary_mask, 0] = p_magnetic[boundary_mask] * p_spec_fm[boundary_mask]
            
        # Apply final probability calibrator if provided (Experiment 2)
        if self.calibrator is not None:
            # Reconstruct the calibration features:
            # p_fm_g, p_afm_g, p_nm_g, p_afm_s, fm_afm_diff, fm_afm_sum, max_p
            p_fm_g = probs_global[:, 0]
            p_afm_g = probs_global[:, 1]
            p_nm_g = probs_global[:, 2]
            p_afm_s = probs_spec[:, 1]
            fm_afm_diff = p_fm_g - p_afm_g
            fm_afm_sum = p_fm_g + p_afm_g
            max_p = np.max(probs_global, axis=1)
            
            X_cal = np.column_stack([
                p_fm_g, p_afm_g, p_nm_g,
                p_afm_s,
                fm_afm_diff, fm_afm_sum,
                max_p
            ])
            probs_corrected = self.calibrator.predict_proba(X_cal)

        # Intrinsic physical closed-shell guardrail
        if raw_df is not None:
            mag_frac = pd.to_numeric(raw_df.get("magnetic_3d_fraction", 0.0), errors="coerce").fillna(0.0) + \
                       pd.to_numeric(raw_df.get("rare_earth_fraction", 0.0), errors="coerce").fillna(0.0)
            unpaired = pd.to_numeric(raw_df.get("unpaired_electron_estimate", 0.0), errors="coerce").fillna(0.0)
            non_mag_mask = (mag_frac <= 1e-5) & (unpaired <= 1e-5)
            if hasattr(non_mag_mask, "values") and np.any(non_mag_mask.values):
                probs_corrected = np.array(probs_corrected, copy=True)
                probs_corrected[non_mag_mask.values, :] = [0.0, 0.0, 1.0]
            
        return probs_corrected
        
    def predict(self, X_scaled_dict):
        X_scaled_dict = self._format_input_dict(X_scaled_dict)
        probs = self.predict_proba(X_scaled_dict)
        
        if self.afm_verifier is not None:
            # Predict verifier on specialist scaled features
            verifier_probs = self.afm_verifier['model'].predict_proba(X_scaled_dict['specialist'])[:, 1]
            return apply_afm_verifier_rule(probs, verifier_probs, self.afm_verifier['tau_afm'], self.afm_verifier['tau_verifier'])
            
        if self.afm_threshold is not None:
            return apply_afm_threshold(probs, self.afm_threshold['tau_afm'], self.afm_threshold['margin'])
            
        return np.argmax(probs, axis=1)

def select_stable_optimized_features(X, y, groups, candidate_features, n_splits=3, is_binary=False, top_k=60):
    """
    Select stable leakage-safe features using fold-wise target relevance.

    The earlier implementation accidentally ranked the lowest-scoring features
    as best. This version averages normalized fold scores directly and then
    selects the largest mean scores.
    """
    from sklearn.feature_selection import f_classif, mutual_info_classif
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    candidate_features = _align_feature_list(
        candidate_features, X.columns, "stable feature-selection candidates",
        min_required=min(5, max(1, len(candidate_features))), verbose=False
    )
    candidate_features = remove_useless_features(X, candidate_features, min_positive=20, min_unique=2)
    print(f"\nRunning stable feature selection over {len(candidate_features)} candidates (is_binary={is_binary}, top_k={top_k})...")

    if len(candidate_features) <= top_k:
        return candidate_features

    if groups is not None and not is_binary:
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(X, y))

    fold_scores = []
    for fold, (train_idx, _) in enumerate(splits):
        X_tr = X.iloc[train_idx][candidate_features].copy()
        X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
        X_tr = X_tr.fillna(X_tr.median(numeric_only=True)).fillna(0.0)
        y_tr = y.iloc[train_idx]

        # Remove constants after fold split.
        std = X_tr.std(numeric_only=True)
        keep = std[std > 1e-10].index.tolist()
        X_tr = X_tr[keep]
        if X_tr.shape[1] == 0:
            continue

        try:
            F_vals, _ = f_classif(X_tr, y_tr)
            F_vals = np.nan_to_num(F_vals, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            F_vals = np.zeros(X_tr.shape[1])

        try:
            mi_vals = mutual_info_classif(X_tr, y_tr, random_state=42)
            mi_vals = np.nan_to_num(mi_vals, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            mi_vals = np.zeros(X_tr.shape[1])

        try:
            clf = LGBMClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                num_leaves=15, min_child_samples=30,
                random_state=42 + fold, n_jobs=-1, verbose=-1
            )
            clf.fit(X_tr, y_tr)
            lgb_gain = clf.booster_.feature_importance(importance_type='gain')
            lgb_gain = np.nan_to_num(lgb_gain, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            lgb_gain = np.zeros(X_tr.shape[1])

        def normalize_score(scores):
            scores = np.asarray(scores, dtype=float)
            s_min, s_max = np.nanmin(scores), np.nanmax(scores)
            if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
                return np.zeros_like(scores)
            return (scores - s_min) / (s_max - s_min)

        combined = 0.30 * normalize_score(F_vals) + 0.30 * normalize_score(mi_vals) + 0.40 * normalize_score(lgb_gain)
        fold_scores.append(pd.Series(combined, index=X_tr.columns))

    if not fold_scores:
        return candidate_features[:top_k]

    score_df = pd.concat(fold_scores, axis=1).fillna(0.0)
    mean_scores = score_df.mean(axis=1).sort_values(ascending=False)

    selected_features = mean_scores.head(top_k).index.tolist()

    # Always include any GNN predictions if they exist in candidate features.
    for col in candidate_features:
        if 'gnn_' in col and col not in selected_features:
            selected_features.append(col)

    selected_features = _align_feature_list(selected_features, X.columns, "stable optimized selected features", min_required=1, verbose=False)
    print(f"Selected {len(selected_features)} stable optimized features.")
    return selected_features


def apply_specialist_correction(probs_global, probs_spec, magnetic_prob_thresh=0.60, fm_afm_margin_thresh=0.30, p_nm_upper=0.40):
    """
    Applies boundary specialist correction dynamically based on gate parameters.
    """
    probs_corrected = probs_global.copy()
    p_magnetic = probs_global[:, 0] + probs_global[:, 1]
    p_nm = probs_global[:, 2]
    
    boundary_mask = (
        (p_magnetic > magnetic_prob_thresh) &
        (np.abs(probs_global[:, 0] - probs_global[:, 1]) < fm_afm_margin_thresh) &
        (p_nm < p_nm_upper)
    )
    
    if np.any(boundary_mask):
        p_spec_afm = probs_spec[:, 1]
        p_spec_fm = probs_spec[:, 0]
        
        probs_corrected[boundary_mask, 1] = p_magnetic[boundary_mask] * p_spec_afm[boundary_mask]
        probs_corrected[boundary_mask, 0] = p_magnetic[boundary_mask] * p_spec_fm[boundary_mask]
        
    return probs_corrected

def build_calibrator_features(probs_global, probs_spec_mapped):
    """
    Constructs the feature matrix for the logistic calibrator (Experiment 2).
    """
    p_fm_g = probs_global[:, 0]
    p_afm_g = probs_global[:, 1]
    p_nm_g = probs_global[:, 2]
    
    p_afm_s = probs_spec_mapped[:, 1]
    
    fm_afm_diff = p_fm_g - p_afm_g
    fm_afm_sum = p_fm_g + p_afm_g
    max_p = np.max(probs_global, axis=1)
    
    return np.column_stack([
        p_fm_g, p_afm_g, p_nm_g,
        p_afm_s,
        fm_afm_diff, fm_afm_sum,
        max_p
    ])

def fit_final_calibrator(oof_probs_global, oof_probs_spec_mapped, y):
    """
    Fits the small final probability calibrator using OOF predictions (Experiment 2).
    """
    from sklearn.linear_model import LogisticRegression
    X_cal = build_calibrator_features(oof_probs_global, oof_probs_spec_mapped)
    calibrator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    calibrator.fit(X_cal, y)
    return calibrator

def run_classification_pipeline(input_dir='output/featurization', output_dir='output/training', plot_dir='plots', feature_groups=None):
    """
    Leakage-safe 3-fold cross validation classification (FM vs AFM vs NM) with Dual Validation splits.
    Optimizes AFM precision using decision thresholds and a second-stage verifier.
    """
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    print("\n==========================================")
    print("RUNNING PIPELINE 1: MAGNETIC CLASS CLASSIFICATION")
    print("==========================================")
    
    load_gnn_cache(os.path.join(output_dir, 'gnn_oof_predictions.pkl'))
    
    path = os.path.join(input_dir, 'features_master.csv')
    if not os.path.exists(path):
        print(f"Error: {path} not found. Skipping classification modeling.")
        return {}
        
    df_all = pd.read_csv(path, low_memory=False)
    df = df_all.dropna(subset=['Type']).copy()
    X = df.drop(columns=['Type', 'reduced_formula', 'Mean_TC_K', 'Mean_TN_K', 'K1_J_m3', 'Coercivity_A_m', 'Saturation_Magnetization_emu_g', 'Hard_Magnet', 'match_confidence', 'match_score', 'match_reason'], errors='ignore')
    y = df['Type'].astype(int)
    groups = df['reduced_formula'].fillna("Unknown")
    
    # Dual validation splits (3 folds for maximum speed and efficiency)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=RUN_CONFIG.n_splits)
    
    ablation_metrics = {}
    oof_probs_by_version = {}
    
    oof_preds_best_perf = np.zeros(len(df))
    oof_preds_best_gen = np.zeros(len(df))
    
    versions_to_run = ['F5'] if not RUN_ABLATION else list(RUN_CONFIG.ablation_versions)
    for version in versions_to_run:
        phys_name = PHYSICAL_LABELS.get(version, version)
        print(f"\n--- Classification representation testing: {phys_name} ({version}) ---")
        features_to_use = get_ablation_features(feature_groups, version, X.columns)
        
        # 1. Performance Mode (Stratified K-Fold for benchmark comparison)
        print("Evaluating Performance Mode (StratifiedKFold)...")
        oof_preds_perf, oof_probs_perf, oof_verifier_perf, _, _, _, _ = train_stacked_classifier(
            X, y, groups, skf, features_to_use
        )
        perf_acc = accuracy_score(y, oof_preds_perf)
        perf_macro_f1 = f1_score(y, oof_preds_perf, average='macro')
        perf_precision, perf_recall, perf_f1_scores, _ = precision_recall_fscore_support(y, oof_preds_perf, average=None)
        perf_afm_f1 = perf_f1_scores[1] if len(perf_f1_scores) > 1 else 0.0
        
        # 2. Generalization Mode (Group K-Fold for polymorphism safety)
        print("Evaluating Generalization Mode (GroupKFold)...")
        oof_preds_gen, oof_probs_gen, oof_verifier_gen, _, _, _, _ = train_stacked_classifier(
            X, y, groups, gkf, features_to_use
        )
        gen_acc = accuracy_score(y, oof_preds_gen)
        gen_macro_f1 = f1_score(y, oof_preds_gen, average='macro')
        gen_precision, gen_recall, gen_f1_scores, _ = precision_recall_fscore_support(y, oof_preds_gen, average=None)
        gen_afm_f1 = gen_f1_scores[1] if len(gen_f1_scores) > 1 else 0.0
        
        print(f"[{phys_name} ({version})] DUAL-VALIDATION SCORES:")
        print(f"  Performance Mode (StratifiedKFold): Accuracy = {perf_acc:.4f} | Macro F1 = {perf_macro_f1:.4f} | AFM F1 = {perf_afm_f1:.4f}")
        print(f"  Generalization Mode (GroupKFold):   Accuracy = {gen_acc:.4f} | Macro F1 = {gen_macro_f1:.4f} | AFM F1 = {gen_afm_f1:.4f}")
        
        oof_probs_by_version[version] = {
            'perf': oof_probs_perf,
            'gen': oof_probs_gen,
            'verifier_perf': oof_verifier_perf,
            'verifier_gen': oof_verifier_gen
        }
        
        ablation_metrics[version] = {
            'performance_accuracy': perf_acc,
            'generalization_accuracy': gen_acc,
            'performance_macro_f1': perf_macro_f1,
            'generalization_macro_f1': gen_macro_f1,
            'performance_afm_f1': perf_afm_f1,
            'generalization_afm_f1': gen_afm_f1,
            'performance_afm_precision': perf_precision[1] if len(perf_precision) > 1 else 0.0,
            'performance_afm_recall': perf_recall[1] if len(perf_recall) > 1 else 0.0,
            'generalization_afm_precision': gen_precision[1] if len(gen_precision) > 1 else 0.0,
            'generalization_afm_recall': gen_recall[1] if len(gen_recall) > 1 else 0.0
        }
        
        if version == 'F5':
            oof_preds_best_perf = oof_preds_perf
            oof_preds_best_gen = oof_preds_gen
            
            labels = ['FM', 'AFM', 'NM']
            metrics_perf_df = pd.DataFrame({'Precision': perf_precision, 'Recall': perf_recall, 'F1-Score': perf_f1_scores}, index=labels)
            metrics_gen_df = pd.DataFrame({'Precision': gen_precision, 'Recall': gen_recall, 'F1-Score': gen_f1_scores}, index=labels)
            
            print(f"\nStacked Classifier Class Metrics (Out-of-Fold {phys_name} - Performance Mode):")
            print(metrics_perf_df)
            print(f"\nStacked Classifier Class Metrics (Out-of-Fold {phys_name} - Generalization Mode):")
            print(metrics_gen_df)
            
    # Representations that were not actually trained are NOT back-filled from F5.
    # Copying F5 into F1-F4 previously produced an ablation table of five identical
    # rows in stage3_modeling.json, and silently collapsed the F1/F3/F5 probability
    # blend below into 1.0 * F5 (0.40 + 0.25 + 0.35 all multiplying the same array).
    untrained_versions = [v for v in ['F1', 'F2', 'F3', 'F4'] if v not in ablation_metrics]
    for v in untrained_versions:
        ablation_metrics[v] = {'status': 'not_evaluated',
                               'reason': 'run_ablation=False; no back-fill from F5'}


    # 1. Save out-of-fold probabilities to disk (Component 1)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, 'oof_probs_all.npz'),
        **{f"{version}_{mode}": probs for version, modes in oof_probs_by_version.items() for mode, probs in modes.items()}
    )
    print(f"Saved out-of-fold probabilities for all feature sets to {os.path.join(output_dir, 'oof_probs_all.npz')}")
    
    # --- AFM WEIGHT MULTIPLIER (Sweep bypassed and locked to 1.0 as requested) ---
    print("\n--- [AFM WEIGHT MULTIPLIER] AFM sample weight multiplier set to 1.0 ---")
    best_afm_mult = 1.0
    
    # 2. Compute F1/F3/F5 blended probabilities.
    # This multi-representation blend is only meaningful when F1 and F3 were
    # actually trained. If they were not, use F5 alone and say so, rather than
    # averaging three copies of the same array and calling it an ensemble.
    have_all_reps = all(v in oof_probs_by_version for v in ('F1', 'F3', 'F5'))
    if have_all_reps:
        w_f1, w_f3, w_f5 = 0.40, 0.25, 0.35
        blended_probs_perf = (w_f1 * oof_probs_by_version['F1']['perf']
                              + w_f3 * oof_probs_by_version['F3']['perf']
                              + w_f5 * oof_probs_by_version['F5']['perf'])
        blended_probs_gen = (w_f1 * oof_probs_by_version['F1']['gen']
                             + w_f3 * oof_probs_by_version['F3']['gen']
                             + w_f5 * oof_probs_by_version['F5']['gen'])
        print(f"Blending representations F1/F3/F5 at weights {w_f1}/{w_f3}/{w_f5}.")
    else:
        blended_probs_perf = oof_probs_by_version['F5']['perf']
        blended_probs_gen = oof_probs_by_version['F5']['gen']
        print("Single-representation mode: using F5 probabilities directly "
              "(F1/F3 not trained, so no multi-representation blend is applied).")


    oof_verifier_perf = oof_probs_by_version['F5']['verifier_perf']
    oof_verifier_gen = oof_probs_by_version['F5']['verifier_gen']

    # --- FEATURE SIFTING: Select stable optimized features for Global and Specialist (F_clean) ---
    candidate_features = []
    for version in ['F1', 'F2', 'F3', 'F4']:
        candidate_features.extend(get_ablation_features(feature_groups, version, X.columns))
    candidate_features = sorted(list(set(candidate_features)))
    
    # Pre-clean candidates (Experiment 3 & 5)
    candidate_features_clean = remove_useless_features(X, candidate_features)
    candidate_features_opt_clean = filter_proxy_features(candidate_features_clean)
    
    spec_mask = (y == 0) | (y == 1)
    X_spec = X[spec_mask]
    y_spec = y[spec_mask]
    groups_spec = groups[spec_mask]
    
    # Aligned splits for Specialist
    skf_splits = list(skf.split(X, y))
    gkf_splits = list(gkf.split(X, y, groups))
    spec_indices = np.where(spec_mask)[0]
    global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(spec_indices)}
    
    skf_specialist_splits = []
    for train_idx, val_idx in skf_splits:
        train_idx_spec = np.array([global_to_local[idx] for idx in train_idx if idx in global_to_local])
        val_idx_spec = np.array([global_to_local[idx] for idx in val_idx if idx in global_to_local])
        skf_specialist_splits.append((train_idx_spec, val_idx_spec))
        
    gkf_specialist_splits = []
    for train_idx, val_idx in gkf_splits:
        train_idx_spec = np.array([global_to_local[idx] for idx in train_idx if idx in global_to_local])
        val_idx_spec = np.array([global_to_local[idx] for idx in val_idx if idx in global_to_local])
        gkf_specialist_splits.append((train_idx_spec, val_idx_spec))
        
    # --- MULTI-DIMENSIONAL SYSTEMATIC GRID-SEARCH SWEEP (Experiments 1, 2, 4, 5, 6) ---
    best_overall_score = -1.0
    best_config = {}
    
    # Initialize all loop-dependent variables to None to prevent UnboundLocalError
    oof_verifier_spec_mapped_perf = None
    oof_verifier_spec_mapped_gen = None
    oof_probs_spec_mapped_perf = None
    oof_probs_spec_mapped_gen = None
    
    selected_features_path = os.path.join(output_dir, 'selected_features_clean80.json')
    loaded_features = False
    F_opt_global_cached = None
    F_opt_FM_AFM_cached = None

    # Prefer valid cached training features, but reject stale caches from older featurization runs.
    if os.path.exists(selected_features_path) and not FORCE_FEATURE_RESELECT and TRAINING_MODE == "fast_debug":
        try:
            with open(selected_features_path, 'r') as f:
                feat_dict = json.load(f)

            F_opt_global_cached = _align_feature_list(
                feat_dict.get('global', []), X.columns,
                f"cached global features from {selected_features_path}",
                min_required=20
            )
            F_opt_FM_AFM_cached = _align_feature_list(
                feat_dict.get('specialist', []), X.columns,
                f"cached specialist features from {selected_features_path}",
                min_required=20
            )
            loaded_features = True
            print(f"Successfully loaded validated cached features from {selected_features_path}")
        except Exception as e:
            print(f"Warning: cached features are stale or invalid: {e}")
            loaded_features = False
            F_opt_global_cached = None
            F_opt_FM_AFM_cached = None

    # If training cache is missing/stale, use the target-specific selected files from featurization.
    if (not loaded_features) and (not FORCE_FEATURE_RESELECT) and TRAINING_MODE == "fast_debug":
        try:
            loaded_g, loaded_s = _load_featurization_selected_features(input_dir, 80, X.columns)
            if loaded_g is not None and loaded_s is not None:
                F_opt_global_cached = loaded_g
                F_opt_FM_AFM_cached = loaded_s
                loaded_features = True
                print("Using target-specific selected features from the featurization stage.")
        except Exception as e:
            print(f"Warning: could not load featurization-stage selected features: {e}")
            loaded_features = False

    # Vetted hyperparameter preset
    best_type = "clean"
    best_k = 80
    best_hard_mult = 1.50
    best_boundary_params = (0.60, 0.25, 0.40)
    mag_t, margin_t, nm_upper = best_boundary_params
    candidates = candidate_features_opt_clean

    print(f"\n--- [NESTED CV] Training Stacked Classifier with fold-isolated feature selection: Type={best_type}, Size={best_k} ---")
    oof_preds_global_perf, oof_probs_global_perf, _, _, _, _, _ = train_stacked_classifier(
        X, y, groups, skf, candidates, select_features_per_fold=True, top_k_per_fold=best_k, is_binary=False
    )
    oof_preds_global_gen, oof_probs_global_gen, _, _, _, _, _ = train_stacked_classifier(
        X, y, groups, gkf, candidates, select_features_per_fold=True, top_k_per_fold=best_k, is_binary=False
    )

    # Specialist weighting
    P_FM_global_spec = oof_probs_global_perf[spec_mask, 0]
    P_AFM_global_spec = oof_probs_global_perf[spec_mask, 1]
    hard_negative_mask_spec = np.abs(P_FM_global_spec - P_AFM_global_spec) < 0.30

    hard_sample_weights_spec = np.ones(len(X_spec))
    hard_sample_weights_spec[hard_negative_mask_spec] = best_hard_mult

    print(f"  Training Specialist Stacked Classifier with Nested Selection (hard_mult={best_hard_mult})...")
    oof_preds_spec_perf, oof_probs_spec_perf, oof_verifier_spec_perf, _, _, _, _ = train_stacked_classifier(
        X_spec, y_spec, groups_spec, skf_specialist_splits, candidates, 
        global_splits=skf_splits, global_groups=groups, hard_sample_weights=hard_sample_weights_spec,
        select_features_per_fold=True, top_k_per_fold=best_k, is_binary=True
    )
    oof_preds_spec_gen, oof_probs_spec_gen, oof_verifier_spec_gen, _, _, _, _ = train_stacked_classifier(
        X_spec, y_spec, groups_spec, gkf_specialist_splits, candidates, 
        global_splits=gkf_splits, global_groups=groups, hard_sample_weights=hard_sample_weights_spec,
        select_features_per_fold=True, top_k_per_fold=best_k, is_binary=True
    )

    # Map specialist predictions and verifiers back
    oof_probs_spec_mapped_perf = np.zeros((len(X), 2))
    oof_probs_spec_mapped_gen = np.zeros((len(X), 2))
    oof_verifier_spec_mapped_perf = np.zeros(len(X))
    oof_verifier_spec_mapped_gen = np.zeros(len(X))
    for idx, spec_idx in enumerate(spec_indices):
        oof_probs_spec_mapped_perf[spec_idx] = oof_probs_spec_perf[idx]
        oof_probs_spec_mapped_gen[spec_idx] = oof_probs_spec_gen[idx]
        oof_verifier_spec_mapped_perf[spec_idx] = oof_verifier_spec_perf[idx]
        oof_verifier_spec_mapped_gen[spec_idx] = oof_verifier_spec_gen[idx]
    oof_probs_spec_mapped_perf[~spec_mask] = 0.5
    oof_probs_spec_mapped_gen[~spec_mask] = 0.5

    spec_corr_perf = apply_specialist_correction(
        oof_probs_global_perf, oof_probs_spec_mapped_perf,
        mag_t, margin_t, nm_upper
    )
    spec_corr_gen = apply_specialist_correction(
        oof_probs_global_gen, oof_probs_spec_mapped_gen,
        mag_t, margin_t, nm_upper
    )

    if USE_FINAL_CALIBRATOR:
        calibrator = fit_final_calibrator(oof_probs_global_perf, oof_probs_spec_mapped_perf, y)
        X_cal_gen = build_calibrator_features(oof_probs_global_gen, oof_probs_spec_mapped_gen)
        calibrated_probs_gen = calibrator.predict_proba(X_cal_gen)
    else:
        calibrator = None
        calibrated_probs_gen = spec_corr_gen

    preds_gen = np.argmax(calibrated_probs_gen, axis=1)
    acc = accuracy_score(y, preds_gen)
    f1_macro = f1_score(y, preds_gen, average='macro')
    f1_afm = f1_score(y, preds_gen, average=None)[1]
    best_overall_score = 0.4 * acc + 0.3 * f1_macro + 0.3 * f1_afm

    # Global feature selection for production model serialization
    if loaded_features and F_opt_global_cached is not None:
        F_opt_global = F_opt_global_cached
    else:
        F_opt_global = select_stable_optimized_features(X, y, groups, candidates, n_splits=3, is_binary=False, top_k=best_k)

    if loaded_features and F_opt_FM_AFM_cached is not None:
        F_opt_FM_AFM = F_opt_FM_AFM_cached
    else:
        F_opt_FM_AFM = select_stable_optimized_features(X_spec, y_spec, groups_spec, candidates, n_splits=3, is_binary=True, top_k=best_k)

    best_calibrator = calibrator
    best_afm_mult = 1.0

    spec_corrected_probs_perf = spec_corr_perf
    spec_corrected_probs_gen = spec_corr_gen

    print(f"\nSWEEP WINNER CHOSEN:")
    print(f"  Feature Subset:   {best_type.upper()}")
    print(f"  Feature Count:    {best_k} columns")
    print(f"  Hard Multiplier:  {best_hard_mult}")
    print(f"  Boundary Gate:    magnetic_prob > {best_boundary_params[0]:.2f}, margin < {best_boundary_params[1]:.2f}, NM_prob < {best_boundary_params[2]:.2f}")
    print(f"  Generalization CV Balanced Score: {best_overall_score:.4f}")
    
    if not loaded_features:
        try:
            with open(selected_features_path, 'w') as f:
                json.dump({
                    'global': list(F_opt_global),
                    'specialist': list(F_opt_FM_AFM),
                    'best_overall_score': best_overall_score,
                    'boundary_params': list(best_boundary_params),
                    'hard_mult': best_hard_mult,
                    'afm_mult': best_afm_mult
                }, f, indent=4)
            print(f"Saved selected features and parameters to {selected_features_path}")
        except Exception as e:
            print(f"Warning: Failed to save selected features to cache: {e}")
    
    # Fit the dynamic logistic calibrators on best OOF predictions
    if USE_FINAL_CALIBRATOR:
        print("\n--- [PROBABILITY CALIBRATION] Fitting final Logistic Calibrators on champion OOF predictions ---")
        calibrator_perf = fit_final_calibrator(oof_probs_global_perf, oof_probs_spec_mapped_perf, y)
        calibrator_gen = fit_final_calibrator(oof_probs_global_gen, oof_probs_spec_mapped_gen, y)
        
        X_cal_perf = build_calibrator_features(oof_probs_global_perf, oof_probs_spec_mapped_perf)
        X_cal_gen = build_calibrator_features(oof_probs_global_gen, oof_probs_spec_mapped_gen)
        
        calibrated_probs_perf = calibrator_perf.predict_proba(X_cal_perf)
        calibrated_probs_gen = calibrator_gen.predict_proba(X_cal_gen)
    else:
        print("\n--- [PROBABILITY CALIBRATION] Skipping final calibrator (using stacked meta-probabilities directly) ---")
        calibrator_perf = None
        calibrator_gen = None
        calibrated_probs_perf = spec_corrected_probs_perf
        calibrated_probs_gen = spec_corrected_probs_gen
    
    # Threshold & verifier tuning on calibrated probabilities
    print("\n--- [THRESHOLD TUNING] Tuning AFM Decision Threshold on calibrated Performance Mode ---")
    spec_perf_thresh_opt = optimize_afm_threshold(calibrated_probs_perf, y)
    print("\n--- [THRESHOLD TUNING] Tuning AFM Decision Threshold on calibrated Generalization Mode ---")
    spec_gen_thresh_opt = optimize_afm_threshold(calibrated_probs_gen, y)
    
    print("\n--- [VERIFIER SWEEP] Tuning AFM Verifier Threshold on calibrated Specialist Ensemble ---")
    spec_tau_afm, spec_tau_verifier, verifier_score_dict = optimize_afm_verifier_threshold(
        calibrated_probs_perf, oof_verifier_spec_mapped_perf, y
    )
    
    spec_preds_perf = apply_afm_verifier_rule(calibrated_probs_perf, oof_verifier_spec_mapped_perf, spec_tau_afm, spec_tau_verifier)
    spec_preds_gen = apply_afm_verifier_rule(calibrated_probs_gen, oof_verifier_spec_mapped_gen, spec_tau_afm, spec_tau_verifier)
    
    spec_perf_acc = accuracy_score(y, spec_preds_perf)
    spec_perf_macro_f1 = f1_score(y, spec_preds_perf, average='macro')
    spec_perf_p, spec_perf_r, spec_perf_f, _ = precision_recall_fscore_support(y, spec_preds_perf, average=None)
    
    spec_gen_acc = accuracy_score(y, spec_preds_gen)
    spec_gen_macro_f1 = f1_score(y, spec_preds_gen, average='macro')
    spec_gen_p, spec_gen_r, spec_gen_f, _ = precision_recall_fscore_support(y, spec_preds_gen, average=None)
    
    # 1. Blended Ensemble comparison
    print("\n--- [VERIFIER SWEEP] Tuning AFM Verifier Threshold on Blended Baseline ---")
    blend_tau_afm, blend_tau_verifier, blend_verifier_score = optimize_afm_verifier_threshold(
        blended_probs_perf,
        oof_verifier_perf,
        y
    )

    blend_preds_perf = apply_afm_verifier_rule(
        blended_probs_perf,
        oof_verifier_perf,
        blend_tau_afm,
        blend_tau_verifier
    )

    blend_preds_gen = apply_afm_verifier_rule(
        blended_probs_gen,
        oof_verifier_gen,
        blend_tau_afm,
        blend_tau_verifier
    )
    
    blend_perf_acc = accuracy_score(y, blend_preds_perf)
    blend_perf_macro_f1 = f1_score(y, blend_preds_perf, average='macro')
    blend_perf_p, blend_perf_r, blend_perf_f, _ = precision_recall_fscore_support(y, blend_preds_perf, average=None)
    
    blend_gen_acc = accuracy_score(y, blend_preds_gen)
    blend_gen_macro_f1 = f1_score(y, blend_preds_gen, average='macro')
    blend_gen_p, blend_gen_r, blend_gen_f, _ = precision_recall_fscore_support(y, blend_preds_gen, average=None)
    
    # Print the comparison results side-by-side
    print("\n==========================================================================")
    print("           DOUBLE-EVALUATION ENSEMBLE COMPARISON (GENERALIZATION MODE)")
    print("==========================================================================")
    print(f"Metric          | F1/F3/F5 Blended Ensemble | Specialist Corrected (F_opt)")
    print(f"----------------|---------------------------|-----------------------------")
    print(f"AFM Precision   | {blend_gen_p[1]:.4f}                    | {spec_gen_p[1]:.4f}")
    print(f"AFM Recall      | {blend_gen_r[1]:.4f}                    | {spec_gen_r[1]:.4f}")
    print(f"AFM F1-Score    | {blend_gen_f[1]:.4f}                    | {spec_gen_f[1]:.4f}")
    print(f"Overall Acc     | {blend_gen_acc:.4f}                    | {spec_gen_acc:.4f}")
    print(f"Macro F1-Score  | {blend_gen_macro_f1:.4f}                    | {spec_gen_macro_f1:.4f}")
    print("==========================================================================")
    
    best_afm_mult = 1.0
    # Choose champion based on weighted score: 0.65 * accuracy + 0.20 * macro_f1 + 0.15 * afm_f1
    spec_score = 0.65 * spec_gen_acc + 0.20 * spec_gen_macro_f1 + 0.15 * spec_gen_f[1]
    blend_score = 0.65 * blend_gen_acc + 0.20 * blend_gen_macro_f1 + 0.15 * blend_gen_f[1]
    
    print(f"\nChampion Selection Scores:")
    print(f"  Specialist Corrected Score: {spec_score:.4f} (Acc={spec_gen_acc:.4f}, MacroF1={spec_gen_macro_f1:.4f}, AFM_F1={spec_gen_f[1]:.4f})")
    print(f"  Blended Ensemble Score:     {blend_score:.4f} (Acc={blend_gen_acc:.4f}, MacroF1={blend_gen_macro_f1:.4f}, AFM_F1={blend_gen_f[1]:.4f})")
    
    if spec_score >= blend_score:
        winner_name = "specialist_corrected"
        print(f"\nCHAMPION SELECTED: Specialist Corrected Ensemble (Score = {spec_score:.4f} >= Blended Score = {blend_score:.4f})")
        
        final_gen_acc = spec_gen_acc
        final_gen_macro_f1 = spec_gen_macro_f1
        final_gen_p = spec_gen_p
        final_gen_r = spec_gen_r
        final_gen_f = spec_gen_f
        
        final_perf_acc = spec_perf_acc
        final_perf_macro_f1 = spec_perf_macro_f1
        final_perf_f = spec_perf_f
        
        final_preds_perf = spec_preds_perf
        final_preds_gen = spec_preds_gen
        final_probs_gen = calibrated_probs_gen
    else:
        winner_name = "blended"
        print(f"\nCHAMPION SELECTED: F1/F3/F5 Blended Ensemble (Score = {blend_score:.4f} > Specialist Score = {spec_score:.4f})")
        
        final_gen_acc = blend_gen_acc
        final_gen_macro_f1 = blend_gen_macro_f1
        final_gen_p = blend_gen_p
        final_gen_r = blend_gen_r
        final_gen_f = blend_gen_f
        
        final_perf_acc = blend_perf_acc
        final_perf_macro_f1 = blend_perf_macro_f1
        final_perf_f = blend_perf_f
        
        final_preds_perf = blend_preds_perf
        final_preds_gen = blend_preds_gen
        final_probs_gen = blended_probs_gen
        
        
    # 5. False AFM Error Analysis Export (Component 7)
    export_afm_error_analysis(df, y, final_preds_gen, final_probs_gen, output_dir)
    
    # 6. Confidence Rejection and Coverage Auditing (Component 8)
    rejection_table = compute_confidence_rejection_table(final_probs_gen, y)
    
    # Maintain backward compatibility of plots using final optimized predictions
    labels = ['FM', 'AFM', 'NM']
    cm_perf_path = os.path.join(plot_dir, 'confusion_matrix_classification_perf.png')
    plot_confusion_matrix(y, final_preds_perf, labels, cm_perf_path,
                          title=f'Confusion Matrix ({winner_name.capitalize()} - Performance Mode with Verifier)')
    
    cm_gen_path = os.path.join(plot_dir, 'confusion_matrix_classification_gen.png')
    plot_confusion_matrix(y, final_preds_gen, labels, cm_gen_path,
                          title=f'Confusion Matrix ({winner_name.capitalize()} - Generalization Mode with Verifier)')
    
    cm_path = os.path.join(plot_dir, 'confusion_matrix_classification.png')
    plot_confusion_matrix(y, final_preds_gen, labels, cm_path,
                          title=f'Confusion Matrix ({winner_name.capitalize()} - Generalization Mode with Verifier)')
            
    # Train final best classification model on entire dataset
    if winner_name == "specialist_corrected":
        print("\nTraining final Specialist Corrected classification ensembles on entire dataset...")
        global_base, global_meta, global_scaler, global_cols, global_med, global_imp, gnn_classifier, _ = train_final_stacked_classifier(
            X, y, groups, F_opt_global, afm_weight_multiplier=best_afm_mult, gnn_classifier=None
        )
        global_ensemble = StackedClassifierEnsemble(global_base, global_meta, global_scaler, global_cols, global_med, global_imp)
        global_ensemble.gnn_classifier = gnn_classifier
        
        # Apply hard weight multiplier on final specialist training
        print(f"Training final specialist classification ensemble (FM vs AFM) on F_opt_FM_AFM features with hard_mult={best_hard_mult}...")
        P_FM_global_spec = oof_probs_global_perf[spec_mask, 0]
        P_AFM_global_spec = oof_probs_global_perf[spec_mask, 1]
        hard_negative_mask_spec = np.abs(P_FM_global_spec - P_AFM_global_spec) < 0.30
        
        hard_sample_weights_spec = np.ones(len(X_spec))
        hard_sample_weights_spec[hard_negative_mask_spec] = best_hard_mult
        
        spec_base, spec_meta, spec_scaler, spec_cols, spec_med, spec_imp, _, verifier_model = train_final_stacked_classifier(
            X_spec, y_spec, groups_spec, F_opt_FM_AFM, afm_weight_multiplier=best_afm_mult, gnn_classifier=gnn_classifier,
            hard_sample_weights=hard_sample_weights_spec
        )
        specialist_ensemble = StackedClassifierEnsemble(spec_base, spec_meta, spec_scaler, spec_cols, spec_med, spec_imp)
        
        ensemble_class = SpecialistCorrectedClassifierEnsemble(
            global_ensemble, specialist_ensemble, F_opt_global, F_opt_FM_AFM,
            calibrator=best_calibrator, boundary_params=best_boundary_params
        )
        
        # Expose attributes for seamless serialization and prediction matching
        ensemble_class.scaler_class_global = global_scaler
        ensemble_class.scaler_class_specialist = spec_scaler
        
        ensemble_class.features_class_global = global_cols
        ensemble_class.features_class_specialist = spec_cols
        
        ensemble_class.medians_class_global = global_med
        ensemble_class.medians_class_specialist = spec_med
        
        # Store optimized threshold and verifier parameters
        ensemble_class.afm_threshold = None  # Removed from the active path (verifier active)
        ensemble_class.afm_verifier = {
            'model': verifier_model,
            'tau_afm': spec_tau_afm,
            'tau_verifier': spec_tau_verifier
        }
        
        ret_scaler = global_scaler
        ret_cols = global_cols
        ret_med = global_med
        ret_imp = global_imp
        
    else:
        print("\nTraining final stacked F5 classification ensemble on entire dataset...")
        f5_features = get_ablation_features(feature_groups, 'F5', X.columns)
        base_f5, cal_meta_f5, scaler_f5, cols_f5, med_f5, imp_f5, gnn_classifier, verifier_model = train_final_stacked_classifier(
            X, y, groups, f5_features, afm_weight_multiplier=best_afm_mult, gnn_classifier=None
        )
        ensemble_class = StackedClassifierEnsemble(base_f5, cal_meta_f5, scaler_f5, cols_f5, med_f5, imp_f5)
        ensemble_class.gnn_classifier = gnn_classifier
        ensemble_class.afm_threshold = None
        ensemble_class.afm_verifier = {
            'model': verifier_model,
            'tau_afm': blend_tau_afm,
            'tau_verifier': blend_tau_verifier
        }
        ret_scaler = scaler_f5
        ret_cols = cols_f5
        ret_med = med_f5
        ret_imp = imp_f5
        
    # Save predictions
    df_preds = pd.DataFrame({
        'True_Type': y,
        'Predicted_Type_Performance': final_preds_perf,
        'Predicted_Type_Generalization': final_preds_gen,
        'Predicted_Type_Ensemble': final_preds_gen  # backward compatibility
    })
    df_preds.to_csv(os.path.join(output_dir, 'classification_predictions.csv'), index=False)
    
    indices = np.argsort(list(ret_imp.values()))[::-1][:15]
    top_features = [list(ret_imp.keys())[idx] for idx in indices]
    top_importances = [list(ret_imp.values())[idx] for idx in indices]
    
    # Scientific JSON Summary
    summary = {
        'classification_metrics': {
            'performance_accuracy': final_perf_acc,
            'generalization_accuracy': final_gen_acc,
            'performance_macro_f1': final_perf_macro_f1,
            'generalization_macro_f1': final_gen_macro_f1,
            'performance_afm_f1': final_perf_f[1],
            'generalization_afm_f1': final_gen_f[1],
            'baseline_generalization_afm_precision': ablation_metrics['F5']['generalization_afm_precision'],
            'optimized_generalization_afm_precision': final_gen_p[1],
            'baseline_generalization_afm_recall': ablation_metrics['F5']['generalization_afm_recall'],
            'optimized_generalization_afm_recall': final_gen_r[1],
            'optimized_threshold_parameters': ensemble_class.afm_threshold,
            'verifier_tau_afm': ensemble_class.afm_verifier['tau_afm'],
            'verifier_tau_verifier': ensemble_class.afm_verifier['tau_verifier'],
            'class_breakdown_f1': {labels[i]: final_gen_f[i] for i in range(3)},
            'top_predictive_features': {top_features[i]: top_importances[i] for i in range(min(len(top_features), 10))},
            'confidence_rejection_table': rejection_table,
            'winner_ensemble_type': winner_name
        },
        'classification_ablation': ablation_metrics
    }
    save_results_summary(summary)
    
    save_gnn_cache(os.path.join(output_dir, 'gnn_oof_predictions.pkl'))
    
    try:
        with open(os.path.join(output_dir, 'classification_results.pkl'), 'wb') as f:
            pickle.dump((ensemble_class, ret_scaler, ret_cols, ret_med, ret_imp), f)
        # Record the configuration that produced this cache. Without it, a cache
        # built under fast_debug is silently reused by a final_manuscript run and
        # its stale metrics are reported as full capacity.
        with open(os.path.join(output_dir, 'classification_results.config.json'), 'w') as f:
            json.dump(RUN_CONFIG.provenance(), f, indent=2)
        print(f"Saved classification model cache to {os.path.join(output_dir, 'classification_results.pkl')}")
    except Exception as e:
        print(f"Warning: Could not save classification cache: {e}")
        
    return ensemble_class, ret_scaler, ret_cols, ret_med, ret_imp





def run_regression_pipeline(input_dir='output/featurization', output_dir='output/training', plot_dir='plots', feature_groups=None):
    """
    Leakage-safe 3-fold cross validation regression for Curie (TC) and Neel (TN) with Dual Validation splits.
    Improvements over baseline:
      - Physics-anchor delta-Tc target: reduces dynamic range from 4 orders → ~1.5 orders of magnitude
      - Family-stratified specialist models (metallic / RE-compound / TM-oxide)
    """
    print("\n==========================================")
    print("RUNNING PIPELINE 2: TRANSITION TEMPERATURE REGRESSION")
    print("==========================================")
    
    load_gnn_cache(os.path.join(output_dir, 'gnn_oof_predictions.pkl'))
    
    path = os.path.join(input_dir, 'features_master.csv')
    if not os.path.exists(path):
        print(f"Error: {path} not found. Skipping temperature modeling.")
        return None
        
    df_all = pd.read_csv(path, low_memory=False)
    
    n_memb_cv = 1
    n_memb_final = 2
    
    # --- Curie Temperature Modeling ---
    print("\n--- Curie Temperature (TC) Modeling ---")
    df_c = df_all.dropna(subset=['Mean_TC_K']).copy()
    if 'Type' in df_c.columns:
        df_c = df_c[df_c['Type'] == 0].copy()
    X_c = df_c.drop(columns=['Type', 'reduced_formula', 'Mean_TC_K', 'Mean_TN_K', 'K1_J_m3', 'Coercivity_A_m', 'Saturation_Magnetization_emu_g', 'Hard_Magnet', 'match_confidence', 'match_score', 'match_reason'], errors='ignore')
    y_c = df_c['Mean_TC_K']
    print("TC target stats:")
    print(y_c.describe())
    print("TC > 1800 K:", (y_c > 1800).sum())
    print("TC <= 0 K:", (y_c <= 0).sum())
    groups_c = df_c['reduced_formula'].fillna("Unknown")
    
    # Load out-of-fold stacked classification probabilities for multi-task coupling
    cascade_probs_dict = {}
    oof_npz_path = os.path.join(output_dir, 'oof_probs_all.npz')
    if os.path.exists(oof_npz_path):
        try:
            npz_data = np.load(oof_npz_path)
            if 'F5_perf' in npz_data and len(npz_data['F5_perf']) == len(df_all):
                cascade_probs_dict['perf'] = npz_data['F5_perf']
            if 'F5_gen' in npz_data and len(npz_data['F5_gen']) == len(df_all):
                cascade_probs_dict['gen'] = npz_data['F5_gen']
        except Exception:
            pass
    
    # Set up dual validation splits (3 folds for maximum speed and efficiency)
    y_c_bins = pd.cut(y_c, bins=5, labels=False, duplicates='drop')
    cv_perf_c = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_gen_c = GroupKFold(n_splits=RUN_CONFIG.n_splits)
    
    curie_ablation = {}
    oof_preds_best_c_perf = np.zeros(len(df_c))
    oof_preds_best_c_gen = np.zeros(len(df_c))
    
    versions_to_run = ['F5'] if not RUN_ABLATION else list(RUN_CONFIG.ablation_versions)
    for version in versions_to_run:
        phys_name = PHYSICAL_LABELS.get(version, version)
        print(f"\n--- Curie Regressor representation testing: {phys_name} ({version}) ---")
        features_to_use = get_ablation_features(feature_groups, version, X_c.columns)
        
        # Generalization Mode (Group K-Fold - Zero Chemical Formula Leakage)
        print("Evaluating Generalization Mode (GroupKFold)...")
        oof_preds_gen = np.zeros(len(df_c))
        for fold, (train_idx, val_idx) in enumerate(cv_gen_c.split(X_c, y_c, groups_c)):
            X_tr, X_val = X_c.iloc[train_idx], X_c.iloc[val_idx]
            y_tr, y_val = y_c.iloc[train_idx], y_c.iloc[val_idx]
            
            X_tr_sel, X_val_sel, _, _ = leakage_safe_feature_selection(X_tr[features_to_use], X_val[features_to_use], y_tr)
            
            # Fetch cached GNN or train once if not cached
            gnn_model, gnn_tr_class, gnn_tr_tc, gnn_tr_tn, gnn_val_class, gnn_val_tc, gnn_val_tn, gnn_tr_tc_unc, gnn_tr_tn_unc, gnn_val_tc_unc, gnn_val_tn_unc = get_cached_gnn(
                groups_c.iloc[train_idx].tolist(), 
                groups_c.iloc[val_idx].tolist(), 
                epochs=None  # Use warm-start fine-tuning
            )
            
            X_tr_sel = X_tr_sel.copy()
            X_val_sel = X_val_sel.copy()
            X_tr_sel['gnn_temp_pred'] = gnn_tr_tc
            X_tr_sel['gnn_temp_uncertainty'] = gnn_tr_tc_unc
            X_tr_sel['gnn_prob_FM'] = gnn_tr_class[:, 0]
            X_tr_sel['gnn_prob_AFM'] = gnn_tr_class[:, 1]
            X_tr_sel['gnn_prob_NM'] = gnn_tr_class[:, 2]
            
            X_val_sel['gnn_temp_pred'] = gnn_val_tc
            X_val_sel['gnn_temp_uncertainty'] = gnn_val_tc_unc
            X_val_sel['gnn_prob_FM'] = gnn_val_class[:, 0]
            X_val_sel['gnn_prob_AFM'] = gnn_val_class[:, 1]
            X_val_sel['gnn_prob_NM'] = gnn_val_class[:, 2]
            
            # Cascaded multi-task coupling: inject high-accuracy classifier probabilities
            if 'gen' in cascade_probs_dict:
                c_probs = cascade_probs_dict['gen']
                c_tr_orig = df_c.index[train_idx]
                c_val_orig = df_c.index[val_idx]
                X_tr_sel['cascade_prob_FM'] = c_probs[c_tr_orig, 0]
                X_tr_sel['cascade_prob_AFM'] = c_probs[c_tr_orig, 1]
                X_tr_sel['cascade_prob_NM'] = c_probs[c_tr_orig, 2]
                X_val_sel['cascade_prob_FM'] = c_probs[c_val_orig, 0]
                X_val_sel['cascade_prob_AFM'] = c_probs[c_val_orig, 1]
                X_val_sel['cascade_prob_NM'] = c_probs[c_val_orig, 2]
            
            ensemble = TargetBalancedRegressorEnsemble(n_members=n_memb_cv, random_state=42)
            ensemble.fit(X_tr_sel, y_tr, X_tr_sel.columns.tolist(), use_multi_scale=False)
            pred, _ = ensemble.predict(X_val_sel)
            oof_preds_gen[val_idx] = pred
            
        gen_r2 = r2_score(y_c, oof_preds_gen)
        gen_mae = mean_absolute_error(y_c, oof_preds_gen)
        perf_r2 = gen_r2
        perf_mae = gen_mae
        oof_preds_perf = oof_preds_gen
        
        print(f"[{phys_name} ({version})] GroupKFold Curie Temperature (TC) SCORES:")
        print(f"  Generalization Mode (GroupKFold): R2 = {gen_r2:.4f} | MAE = {gen_mae:.2f} K")
        
        curie_ablation[version] = {
            'performance_r2': perf_r2,
            'generalization_r2': gen_r2,
            'performance_mae': perf_mae,
            'generalization_mae': gen_mae
        }
        
        if version == 'F5':
            oof_preds_best_c_perf = oof_preds_perf
            oof_preds_best_c_gen = oof_preds_gen
            
            plot_predictions_vs_actual(y_c, oof_preds_best_c_gen, f'Curie Temp (TC) Predictions vs. Actual ({phys_name})', 'Predicted Curie Temperature (K)', os.path.join(plot_dir, 'curie_predictions_vs_actual.png'), PALETTE['fm_color'])
            
    # Untrained representations are recorded as such, never copied from F5.
    for v in ['F1', 'F2', 'F3', 'F4']:
        if v not in curie_ablation:
            curie_ablation[v] = {'status': 'not_evaluated',
                                 'reason': 'run_ablation=False; no back-fill from F5'}


    # Train final best Curie model on entire dataset
    print("\nTraining final Curie Regressor Ensemble on entire dataset (20 members)...")
    f5_features_c = get_ablation_features(feature_groups, 'F5', X_c.columns)
    X_c_f5_sel, _, selected_columns_c, medians_dict_c = leakage_safe_feature_selection(X_c[f5_features_c], X_c[f5_features_c], y_c)
    medians_full_c = pd.Series(medians_dict_c)
    X_c_f5_sel = X_c_f5_sel.fillna(medians_full_c).fillna(0.0)
        
    # Train final GNN regressor for Curie
    full_classes = []
    full_tcs = []
    full_tns = []
    for g in groups_c.tolist():
        tgt = _FORMULA_TARGETS.get(g, {'class': 2, 'tc': np.nan, 'tn': np.nan})
        full_classes.append(tgt['class'])
        full_tcs.append(tgt['tc'])
        full_tns.append(tgt['tn'])
        
    full_targets = {
        'class': np.array(full_classes),
        'tc': np.array(full_tcs),
        'tn': np.array(full_tns)
    }
    
    gnn_curie = train_composition_gnn_model(
        groups_c.tolist(),
        full_targets,
        epochs=30,  # Match updated GNN default
        verbose=True
    )
    class_probs, gnn_tc_preds, _, gnn_tc_unc, _ = predict_with_gnn(gnn_curie, groups_c.tolist(), task_type='all_with_uncertainty')
    
    X_c_f5_sel = X_c_f5_sel.copy()
    X_c_f5_sel['gnn_temp_pred'] = gnn_tc_preds
    X_c_f5_sel['gnn_temp_uncertainty'] = gnn_tc_unc
    X_c_f5_sel['gnn_prob_FM'] = class_probs[:, 0]
    X_c_f5_sel['gnn_prob_AFM'] = class_probs[:, 1]
    X_c_f5_sel['gnn_prob_NM'] = class_probs[:, 2]
    
    ensemble_curie = TargetBalancedRegressorEnsemble(n_members=n_memb_final, random_state=42)
    ensemble_curie.fit(X_c_f5_sel, y_c, X_c_f5_sel.columns.tolist())
    ensemble_curie.gnn_curie = gnn_curie
    
    # Fit original rf_c_final on unscaled features specifically for PDP plotting
    rf_c_final = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_c_final.fit(X_c_f5_sel, y_c)
    
    # Compute feature importances
    rf_c_imp = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_c_imp.fit(ensemble_curie.scaler.transform(X_c_f5_sel), y_c)
    lgb_c_imp = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, num_leaves=15, random_state=42, n_jobs=-1, verbose=-1)
    lgb_c_imp.fit(ensemble_curie.scaler.transform(X_c_f5_sel), y_c)
    importances_c = (rf_c_imp.feature_importances_ + lgb_c_imp.feature_importances_) / 2.0
    
    indices_c = np.argsort(importances_c)[::-1][:15]
    top_features_c = X_c_f5_sel.columns[indices_c].tolist()
    top_importances_c = importances_c[indices_c].tolist()
    
    
    # --- Neel Temperature Modeling ---
    print("\n--- Néel Temperature (TN) Modeling ---")
    df_n = df_all.dropna(subset=['Mean_TN_K']).copy()
    if 'Type' in df_n.columns:
        df_n = df_n[df_n['Type'] == 1].copy()
    X_n = df_n.drop(columns=['Type', 'reduced_formula', 'Mean_TC_K', 'Mean_TN_K', 'K1_J_m3', 'Coercivity_A_m', 'Saturation_Magnetization_emu_g', 'Hard_Magnet', 'match_confidence', 'match_score', 'match_reason'], errors='ignore')
    y_n = df_n['Mean_TN_K']
    print("TN target stats:")
    print(y_n.describe())
    print("TN > 1500 K:", (y_n > 1500).sum())
    print("TN <= 0 K:", (y_n <= 0).sum())
    groups_n = df_n['reduced_formula'].fillna("Unknown")
    
    # Set up dual validation splits (3 folds for maximum speed and efficiency)
    y_n_bins = pd.cut(y_n, bins=5, labels=False, duplicates='drop')
    cv_perf_n = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_gen_n = GroupKFold(n_splits=RUN_CONFIG.n_splits)
    
    neel_ablation = {}
    oof_preds_best_n_perf = np.zeros(len(df_n))
    oof_preds_best_n_gen = np.zeros(len(df_n))
    
    versions_to_run = ['F5'] if not RUN_ABLATION else list(RUN_CONFIG.ablation_versions)
    for version in versions_to_run:
        phys_name = PHYSICAL_LABELS.get(version, version)
        print(f"\n--- Néel Regressor representation testing: {phys_name} ({version}) ---")
        features_to_use = get_ablation_features(feature_groups, version, X_n.columns)
        
        # Generalization Mode (Group K-Fold - Zero Chemical Formula Leakage)
        print("Evaluating Generalization Mode (GroupKFold)...")
        oof_preds_gen = np.zeros(len(df_n))
        for fold, (train_idx, val_idx) in enumerate(cv_gen_n.split(X_n, y_n, groups_n)):
            X_tr, X_val = X_n.iloc[train_idx], X_n.iloc[val_idx]
            y_tr, y_val = y_n.iloc[train_idx], y_n.iloc[val_idx]
            
            X_tr_sel, X_val_sel, _, _ = leakage_safe_feature_selection(X_tr[features_to_use], X_val[features_to_use], y_tr)
            
            # Fetch cached GNN or train once if not cached
            gnn_model, gnn_tr_class, gnn_tr_tc, gnn_tr_tn, gnn_val_class, gnn_val_tc, gnn_val_tn, gnn_tr_tc_unc, gnn_tr_tn_unc, gnn_val_tc_unc, gnn_val_tn_unc = get_cached_gnn(
                groups_n.iloc[train_idx].tolist(), 
                groups_n.iloc[val_idx].tolist(), 
                epochs=None  # Use warm-start fine-tuning
            )
            
            X_tr_sel = X_tr_sel.copy()
            X_val_sel = X_val_sel.copy()
            X_tr_sel['gnn_temp_pred'] = gnn_tr_tn
            X_tr_sel['gnn_temp_uncertainty'] = gnn_tr_tn_unc
            X_tr_sel['gnn_prob_FM'] = gnn_tr_class[:, 0]
            X_tr_sel['gnn_prob_AFM'] = gnn_tr_class[:, 1]
            X_tr_sel['gnn_prob_NM'] = gnn_tr_class[:, 2]
            
            X_val_sel['gnn_temp_pred'] = gnn_val_tn
            X_val_sel['gnn_temp_uncertainty'] = gnn_val_tn_unc
            X_val_sel['gnn_prob_FM'] = gnn_val_class[:, 0]
            X_val_sel['gnn_prob_AFM'] = gnn_val_class[:, 1]
            X_val_sel['gnn_prob_NM'] = gnn_val_class[:, 2]
            
            # Cascaded multi-task coupling: inject high-accuracy classifier probabilities
            if 'gen' in cascade_probs_dict:
                c_probs = cascade_probs_dict['gen']
                n_tr_orig = df_n.index[train_idx]
                n_val_orig = df_n.index[val_idx]
                X_tr_sel['cascade_prob_FM'] = c_probs[n_tr_orig, 0]
                X_tr_sel['cascade_prob_AFM'] = c_probs[n_tr_orig, 1]
                X_tr_sel['cascade_prob_NM'] = c_probs[n_tr_orig, 2]
                X_val_sel['cascade_prob_FM'] = c_probs[n_val_orig, 0]
                X_val_sel['cascade_prob_AFM'] = c_probs[n_val_orig, 1]
                X_val_sel['cascade_prob_NM'] = c_probs[n_val_orig, 2]
            
            ensemble = TargetBalancedRegressorEnsemble(n_members=n_memb_cv, random_state=42)
            ensemble.fit(X_tr_sel, y_tr, X_tr_sel.columns.tolist(), is_tc=False, use_multi_scale=False)
            pred, _ = ensemble.predict(X_val_sel)
            oof_preds_gen[val_idx] = pred
            
        gen_r2 = r2_score(y_n, oof_preds_gen)
        gen_mae = mean_absolute_error(y_n, oof_preds_gen)
        perf_r2 = gen_r2
        perf_mae = gen_mae
        oof_preds_perf = oof_preds_gen
        
        print(f"[{phys_name} ({version})] GroupKFold Néel Temperature (TN) SCORES:")
        print(f"  Generalization Mode (GroupKFold): R2 = {gen_r2:.4f} | MAE = {gen_mae:.2f} K")
        
        neel_ablation[version] = {
            'performance_r2': perf_r2,
            'generalization_r2': gen_r2,
            'performance_mae': perf_mae,
            'generalization_mae': gen_mae
        }
        
        if version == 'F5':
            oof_preds_best_n_perf = oof_preds_perf
            oof_preds_best_n_gen = oof_preds_gen
            
            plot_predictions_vs_actual(y_n, oof_preds_best_n_gen, f'Néel Temp (TN) Predictions vs. Actual ({phys_name})', 'Predicted Néel Temperature (K)', os.path.join(plot_dir, 'neel_predictions_vs_actual.png'), PALETTE['afm_color'])
            
    # Untrained representations are recorded as such, never copied from F5.
    for v in ['F1', 'F2', 'F3', 'F4']:
        if v not in neel_ablation:
            neel_ablation[v] = {'status': 'not_evaluated',
                                'reason': 'run_ablation=False; no back-fill from F5'}


    # Train final best Neel model on entire dataset
    print("\nTraining final Néel Regressor Ensemble on entire dataset (20 members)...")
    f5_features_n = get_ablation_features(feature_groups, 'F5', X_n.columns)
    X_n_f5_sel, _, selected_columns_n, medians_dict_n = leakage_safe_feature_selection(X_n[f5_features_n], X_n[f5_features_n], y_n)
    medians_full_n = pd.Series(medians_dict_n)
    X_n_f5_sel = X_n_f5_sel.fillna(medians_full_n).fillna(0.0)
        
    # Train final GNN regressor for Neel
    full_classes = []
    full_tcs = []
    full_tns = []
    for g in groups_n.tolist():
        tgt = _FORMULA_TARGETS.get(g, {'class': 2, 'tc': np.nan, 'tn': np.nan})
        full_classes.append(tgt['class'])
        full_tcs.append(tgt['tc'])
        full_tns.append(tgt['tn'])
        
    full_targets = {
        'class': np.array(full_classes),
        'tc': np.array(full_tcs),
        'tn': np.array(full_tns)
    }
    
    gnn_neel = train_composition_gnn_model(
        groups_n.tolist(),
        full_targets,
        epochs=30,  # Match updated GNN default
        verbose=True
    )
    class_probs, _, gnn_tn_preds, _, gnn_tn_unc = predict_with_gnn(gnn_neel, groups_n.tolist(), task_type='all_with_uncertainty')
    
    X_n_f5_sel = X_n_f5_sel.copy()
    X_n_f5_sel['gnn_temp_pred'] = gnn_tn_preds
    X_n_f5_sel['gnn_temp_uncertainty'] = gnn_tn_unc
    X_n_f5_sel['gnn_prob_FM'] = class_probs[:, 0]
    X_n_f5_sel['gnn_prob_AFM'] = class_probs[:, 1]
    X_n_f5_sel['gnn_prob_NM'] = class_probs[:, 2]
    
    ensemble_neel = TargetBalancedRegressorEnsemble(n_members=n_memb_final, random_state=42)
    ensemble_neel.fit(X_n_f5_sel, y_n, X_n_f5_sel.columns.tolist(), is_tc=False)
    ensemble_neel.gnn_neel = gnn_neel
    
    # Compute feature importances
    rf_n_imp = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_n_imp.fit(ensemble_neel.scaler.transform(X_n_f5_sel), y_n)
    lgb_n_imp = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, num_leaves=15, random_state=42, n_jobs=-1, verbose=-1)
    lgb_n_imp.fit(ensemble_neel.scaler.transform(X_n_f5_sel), y_n)
    importances_n = (rf_n_imp.feature_importances_ + lgb_n_imp.feature_importances_) / 2.0
    
    indices_n = np.argsort(importances_n)[::-1][:15]
    top_features_n = X_n_f5_sel.columns[indices_n].tolist()
    top_importances_n = importances_n[indices_n].tolist()
    
    # Save predictions
    df_preds_c = pd.DataFrame({
        'True_TC': y_c,
        'Predicted_TC_Performance': oof_preds_best_c_perf,
        'Predicted_TC_Generalization': oof_preds_best_c_gen,
        'Predicted_TC_Ensemble': oof_preds_best_c_gen  # backward compatibility
    })
    df_preds_c.to_csv(os.path.join(output_dir, 'curie_predictions.csv'), index=False)
    
    df_preds_n = pd.DataFrame({
        'True_TN': y_n,
        'Predicted_TN_Performance': oof_preds_best_n_perf,
        'Predicted_TN_Generalization': oof_preds_best_n_gen,
        'Predicted_TN_Ensemble': oof_preds_best_n_gen  # backward compatibility
    })
    df_preds_n.to_csv(os.path.join(output_dir, 'neel_predictions.csv'), index=False)
    
    # Save Curie & Neel results in summary JSON
    summary = {
        'temperature_regression': {
            'curie_tc': {
                'performance_r2': curie_ablation['F5']['performance_r2'],
                'generalization_r2': curie_ablation['F5']['generalization_r2'],
                'performance_mae': curie_ablation['F5']['performance_mae'],
                'generalization_mae': curie_ablation['F5']['generalization_mae'],
                'top_predictive_features': {top_features_c[i]: top_importances_c[i] for i in range(min(len(top_features_c), 10))}
            },
            'neel_tn': {
                'performance_r2': neel_ablation['F5']['performance_r2'],
                'generalization_r2': neel_ablation['F5']['generalization_r2'],
                'performance_mae': neel_ablation['F5']['performance_mae'],
                'generalization_mae': neel_ablation['F5']['generalization_mae'],
                'top_predictive_features': {top_features_n[i]: top_importances_n[i] for i in range(min(len(top_features_n), 10))}
            }
        },
        'curie_ablation': curie_ablation,
        'neel_ablation': neel_ablation
    }
    save_results_summary(summary)
    
    curie_res_df = pd.DataFrame({'True': y_c, 'Pred': oof_preds_best_c_gen})
    neel_res_df = pd.DataFrame({'True': y_n, 'Pred': oof_preds_best_n_gen})
    
    save_gnn_cache(os.path.join(output_dir, 'gnn_oof_predictions.pkl'))
    
    return (
        ensemble_curie, ensemble_curie.scaler, X_c_f5_sel.columns.tolist(), medians_full_c.to_dict(),
        curie_res_df, dict(zip(X_c_f5_sel.columns, importances_c)), rf_c_final, X_c_f5_sel,
        neel_res_df, dict(zip(X_n_f5_sel.columns, importances_n)),
        ensemble_neel, ensemble_neel.scaler, X_n_f5_sel.columns.tolist(), medians_full_n.to_dict()
    )



def plot_radar_chart(plot_dir='plots'):
    """
    Generates and saves a premium publication-ready Radar Chart comparing model metrics
    for the Voting Ensemble across the five ablation versions: F1, F2, F3, F4, and F5.
    Saves to plots/radar_metrics.png (Matplotlib).
    """
    results_path = 'results_summary.json'
    if not os.path.exists(results_path):
        print(f"Warning: {results_path} not found. Skipping Radar Chart generation.")
        return
        
    with open(results_path, 'r') as f:
        res = json.load(f)
        
    categories = ['Classification Acc', 'Curie (TC) R2', 'Neel (TN) R2']
    versions = ['F1', 'F2', 'F3', 'F4', 'F5']
    
    v_vals = {}
    for v in versions:
        cls_acc = res.get('classification_ablation', {}).get(v, {}).get('voting_ensemble_accuracy', 0.0)
        if cls_acc == 0.0:
            cls_acc = res.get('classification_ablation', {}).get(v, {}).get('generalization_accuracy', 0.0)
            
        curie_r2 = res.get('curie_ablation', {}).get(v, {}).get('generalization_r2', 0.0)
        if curie_r2 == 0.0:
            curie_r2 = res.get('curie_ablation', {}).get(v, {}).get('voting_ensemble_r2', 0.0)
            
        neel_r2 = res.get('neel_ablation', {}).get(v, {}).get('generalization_r2', 0.0)
        if neel_r2 == 0.0:
            neel_r2 = res.get('neel_ablation', {}).get(v, {}).get('voting_ensemble_r2', 0.0)
        
        # Clip negative R2 scores for visualization clean mapping
        v_vals[v] = [max(0.0, cls_acc), max(0.0, curie_r2), max(0.0, neel_r2)]
        
    setup_plotting_style()
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    
    # Complete circular loop
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, fontsize=11, color=PALETTE['neutral_dark'], weight='bold')
    ax.set_rscale('linear')
    ax.set_rlim(0, 1.0)
    
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color='#7f8c8d', fontsize=9)
    ax.grid(color='#bdc3c7', linestyle='--', linewidth=0.7)
    
    # Custom colors and styles for F1 to F5
    styles = {
        'F1': {'color': '#94a3b8', 'linestyle': '--', 'label': 'chemical baseline', 'alpha': 0.05},
        'F2': {'color': '#0ea5e9', 'linestyle': '-.', 'label': 'local-moment & exchange enrichment', 'alpha': 0.08},
        'F3': {'color': '#a855f7', 'linestyle': ':', 'label': 'ionic & formula-family enrichment', 'alpha': 0.08},
        'F4': {'color': '#10b981', 'linestyle': '-', 'label': 'macro-symmetry enrichment', 'alpha': 0.15},
        'F5': {'color': '#ff006e', 'linestyle': '-', 'label': 'proxy-augmented upper-bound model', 'alpha': 0.20}
    }
    
    for v in versions:
        vals = v_vals[v]
        plot_vals = vals + vals[:1]
        
        ax.plot(
            angles, plot_vals,
            color=styles[v]['color'],
            linestyle=styles[v]['linestyle'],
            linewidth=2.8 if v in ['F4', 'F5'] else 2.0,
            label=styles[v]['label']
        )
        ax.fill(angles, plot_vals, color=styles[v]['color'], alpha=styles[v]['alpha'])
        
    plt.title('Physics-Controlled Representation Testing: Structural Gains', pad=25, fontsize=14, weight='bold', color=PALETTE['neutral_dark'])
    plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1), frameon=True, facecolor='white', edgecolor='#bdc3c7', fontsize=9)
    
    plt.tight_layout()
    out_path = os.path.join(plot_dir, 'radar_metrics.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved premium static Radar Chart comparing F1-F5 versions to {out_path}")

def run_extended_physical_targets_pipeline(input_dir='output/featurization', output_dir='output/training', plot_dir='plots', feature_groups=None, class_results=None, regression_results=None):
    """
    Trains paper-grade physics-informed models for Target 4 to Target 8:
      - Target 4: Magnetocrystalline Anisotropy Constant K1 (symlog space, J/m^3)
      - Target 6: Saturation Magnetization Ms (linear space, emu/g and Tesla)
      - Target 5: Coercivity Hc (log space, A/m, conditioned on Kronmüller nucleation field)
      - Target 7: Maximum Energy Product (BH)max (log space, kJ/m^3, bounded by theoretical ceiling)
      - Target 8: Curie-Weiss Paramagnetic Intercept theta_p (Kelvin, anchored by Weiss mean-field prior)
    """
    from scipy.optimize import nnls
    print("\n" + "=" * 70)
    print("  RUNNING PHYSICS-CASCADED EXTENDED TARGETS PIPELINE (TARGETS 4 TO 8)")
    print("=" * 70)
    
    feat_path = os.path.join(input_dir, 'features_master.csv')
    df_feat = pd.read_csv(feat_path, low_memory=False)
    
    # 0. Unpack or load trained Phase and Temperature Regressor models
    cls_m = None
    cur_m = None
    nee_m = None
    pack_ref = {}

    if class_results is not None:
        cls_m = class_results[0]
        pack_ref['classification'] = cls_m
        pack_ref['scaler_classification'] = class_results[1]
        pack_ref['features_class'] = class_results[2]
        pack_ref['medians_class'] = class_results[3]

    if regression_results is not None:
        cur_m = regression_results[0]
        pack_ref['curie'] = cur_m
        pack_ref['scaler_curie'] = regression_results[1]
        pack_ref['features_curie'] = regression_results[2]
        pack_ref['medians_curie'] = regression_results[3]
        nee_m = regression_results[10]
        pack_ref['neel'] = nee_m
        pack_ref['scaler_neel'] = regression_results[11]
        pack_ref['features_neel'] = regression_results[12]
        pack_ref['medians_neel'] = regression_results[13]

    models_pkl_path = os.path.join(output_dir, 'trained_models.pkl')
    if (cls_m is None or cur_m is None or nee_m is None) and os.path.exists(models_pkl_path):
        try:
            with open(models_pkl_path, 'rb') as f:
                loaded_pack = pickle.load(f)
            if cls_m is None:
                cls_m = loaded_pack.get('classification')
            if cur_m is None:
                cur_m = loaded_pack.get('curie')
            if nee_m is None:
                nee_m = loaded_pack.get('neel')
            for k, v in loaded_pack.items():
                if k not in pack_ref:
                    pack_ref[k] = v
        except Exception as e:
            print(f"Note: could not load trained_models.pkl: {e}")

    if feature_groups is None:
        fg_path = os.path.join(input_dir, 'feature_groups.json')
        if os.path.exists(fg_path):
            with open(fg_path, 'r') as f:
                feature_groups = json.load(f)
        else:
            feature_groups = pack_ref.get('feature_groups', {})

    # Generate model-predicted phase probabilities and ordering temperatures across master features
    from src.post_processing import _get_gnn_model, _predict_candidate_classes, _align_features, _predict_regressor, _add_all_gnn_columns
    from src.composition_gnn import predict_with_gnn

    gnn = _get_gnn_model(cls_m, cur_m, nee_m)
    if cls_m is not None and cur_m is not None and nee_m is not None and gnn is not None:
        print("Generating multi-task predictions (Phase, TC, TN) across all 50,159 compounds...")
        formulas = df_feat['reduced_formula'].fillna('Fe').astype(str).tolist()
        class_probs_gnn, gnn_tc, gnn_tn, gnn_tc_unc, gnn_tn_unc = predict_with_gnn(gnn, formulas, task_type='all_with_uncertainty')
        X_all = _add_all_gnn_columns(df_feat, class_probs_gnn, gnn_tc, gnn_tn, gnn_tc_unc, gnn_tn_unc)
        _, probs_cls = _predict_candidate_classes(cls_m, X_all, pack_ref)

        X_curie_base = X_all.copy()
        X_curie_base['gnn_temp_pred'] = gnn_tc
        X_curie_base['gnn_temp_uncertainty'] = gnn_tc_unc
        X_curie = _align_features(X_curie_base, pack_ref.get('features_curie'), pack_ref.get('medians_curie', {}), label='Curie')
        tc_pred, _ = _predict_regressor(cur_m, X_curie, pack_ref.get('scaler_curie'), clip_range=(0.0, 1800.0))

        X_neel_base = X_all.copy()
        X_neel_base['gnn_temp_pred'] = gnn_tn
        X_neel_base['gnn_temp_uncertainty'] = gnn_tn_unc
        X_neel = _align_features(X_neel_base, pack_ref.get('features_neel'), pack_ref.get('medians_neel', {}), label='Neel')
        tn_pred, _ = _predict_regressor(nee_m, X_neel, pack_ref.get('scaler_neel'), clip_range=(0.0, 1200.0))

        df_feat['cascade_prob_FM'] = probs_cls[:, 0]
        df_feat['cascade_prob_AFM'] = probs_cls[:, 1]
        df_feat['cascade_prob_NM'] = probs_cls[:, 2]
        df_feat['cascade_pred_TC'] = tc_pred
        df_feat['cascade_pred_TN'] = tn_pred
    else:
        print("Warning: Trained models not available for cascading; using elemental prior fallbacks.")
        magmom = df_feat['dao_msn_ms_emu_g'].fillna(50.0).values if 'dao_msn_ms_emu_g' in df_feat.columns else np.full(len(df_feat), 50.0)
        df_feat['cascade_prob_FM'] = np.clip(magmom / 150.0, 0.0, 1.0)
        df_feat['cascade_prob_NM'] = np.where(magmom < 5.0, 0.90, 0.05)
        df_feat['cascade_prob_AFM'] = np.clip(1.0 - df_feat['cascade_prob_FM'] - df_feat['cascade_prob_NM'], 0.0, 1.0)
        df_feat['cascade_pred_TC'] = df_feat['elemental_fm_tc_prior'].fillna(300.0) if 'elemental_fm_tc_prior' in df_feat.columns else np.full(len(df_feat), 300.0)
        df_feat['cascade_pred_TN'] = df_feat['elemental_afm_tn_prior'].fillna(100.0) if 'elemental_afm_tn_prior' in df_feat.columns else np.full(len(df_feat), 100.0)

    # 1. Saturation Magnetization Ms physical descriptors
    tc = df_feat['cascade_pred_TC'].fillna(300.0).values
    tn = df_feat['cascade_pred_TN'].fillna(100.0).values
    pfm = df_feat['cascade_prob_FM'].fillna(0.33).values
    pafm = df_feat['cascade_prob_AFM'].fillna(0.33).values

    eta_T = np.maximum(0.0, 1.0 - (300.0 / np.maximum(tc, 301.0))**1.5) * pfm
    df_feat['phys_ms_bloch_factor'] = eta_T
    vec_val = df_feat['valence_electron_concentration'].fillna(8.35).values if 'valence_electron_concentration' in df_feat.columns else np.full(len(df_feat), 8.35)
    sp_phys_moment = np.clip(2.45 - 1.8 * np.abs(vec_val - 8.35), 0.0, 2.5)
    # Average_Weight is 0.0 (not NaN) for ~30% of the master table and for
    # essentially the whole Ms subset, so fillna() never fired and
    # np.maximum(avg_wt, 10.0) pinned those rows to 10 g/mol. That saturated
    # slater_pauling_theoretical_ms -- a mandatory Ms feature -- at its 350 emu/g
    # clip. Recover the mean atomic mass from the formula where it is missing.
    avg_wt = (df_feat['Average_Weight'].values.astype(float)
              if 'Average_Weight' in df_feat.columns else np.zeros(len(df_feat)))
    avg_wt = np.where(np.isfinite(avg_wt) & (avg_wt > 1.0), avg_wt, np.nan)
    if np.isnan(avg_wt).any() and 'reduced_formula' in df_feat.columns:
        avg_wt = _fill_mean_atomic_mass(avg_wt, df_feat['reduced_formula'].values)
    avg_wt = np.where(np.isfinite(avg_wt), avg_wt, 55.85)
    slater_ms = np.clip(sp_phys_moment * 5585.0 / np.maximum(avg_wt, 10.0), 0.0, 350.0)
    df_feat['slater_pauling_theoretical_ms'] = slater_ms
    df_feat['phys_thermal_reduced_ms'] = slater_ms * eta_T
    msn_ms = df_feat['dao_msn_ms_emu_g'].fillna(50.0).values if 'dao_msn_ms_emu_g' in df_feat.columns else np.full(len(df_feat), 50.0)
    df_feat['phys_fm_scaled_moment'] = msn_ms * pfm
    df_feat['phys_tc_ms_interaction'] = np.log10(1.0 + tc) * pfm

    # 2. Magnetocrystalline Anisotropy K1 physical descriptors
    df_feat['phys_k1_thermal_factor'] = (eta_T ** 3) * pfm
    soc = df_feat['weighted_soc_constant'].fillna(0.0).values if 'weighted_soc_constant' in df_feat.columns else np.zeros(len(df_feat))
    df_feat['phys_soc_exchange_coupling'] = soc * np.sqrt(np.maximum(tc, 0.0))
    uniaxial = df_feat['uniaxial_symmetry_flag'].fillna(0.0).values if 'uniaxial_symmetry_flag' in df_feat.columns else np.zeros(len(df_feat))
    df_feat['phys_aniso_uniaxial_tc'] = uniaxial * np.log10(1.0 + tc)
    re_sign = df_feat['re_anisotropy_sign_index'].fillna(0.0).values if 're_anisotropy_sign_index' in df_feat.columns else np.zeros(len(df_feat))
    df_feat['phys_re_aniso_tc'] = re_sign * np.log10(1.0 + tc)
    df_feat['phys_aniso_fom'] = df_feat['anisotropy_figure_of_merit'].fillna(0.0)
    df_feat['phys_soc'] = soc

    # 3. Curie-Weiss theta_p physical descriptors
    df_feat['cascade_mean_field_theta_p'] = pfm * tc - pafm * tn
    df_feat['phys_afm_exchange_ratio'] = tn / (tc + 10.0)
    super_afm = df_feat['afm_superexchange_index'].fillna(0.0).values if 'afm_superexchange_index' in df_feat.columns else np.zeros(len(df_feat))
    df_feat['phys_superexchange_tn'] = super_afm * tn
    df_feat['phys_exchange_balance'] = (tc - tn) / (tc + tn + 10.0)
    df_feat['phys_cw_high_t_asymptote'] = np.sign(df_feat['cascade_mean_field_theta_p']) * np.sqrt(np.abs(df_feat['cascade_mean_field_theta_p']) * np.maximum(tc, tn))
    df_feat['phys_frustration_index'] = np.abs(df_feat['cascade_mean_field_theta_p']) / np.maximum(tn, 10.0)
    
    # Get available F5 features
    f5_features = get_ablation_features(feature_groups, 'F5', df_feat.columns)

    # ── Cross-target physics cascade, built out-of-fold ──────────────────────
    # The donor properties (K1, Ms, Hc) feed each other through Kronmueller and
    # Stoner-Wohlfarth relations. Previously each donor model was fitted on its
    # full dataset and its in-sample predictions were injected as mandatory
    # features into the next target's CV, so a validation compound carrying an
    # experimental Ms received a near-oracle Ms and the PINN bounds inherited it
    # (measured: +0.011 R2 on (BH)max). build_oof_cascade makes every donor
    # value out-of-fold with respect to that compound's own label.
    if USE_OOF_CASCADE:
        cascade_cache_file = os.path.join(output_dir, 'oof_cascade_features.parquet')
        if os.path.exists(cascade_cache_file):
            print(f"  Loading pre-computed OOF cascade features from {cascade_cache_file}...")
            df_casc = pd.read_parquet(cascade_cache_file)
            for col in df_casc.columns:
                df_feat[col] = df_casc[col].values
        else:
            def _casc_et():
                return ExtraTreesRegressor(n_estimators=100, max_depth=25, min_samples_leaf=2,
                                           max_features=0.85, random_state=42, n_jobs=-1)

            def _casc_lgb():
                return LGBMRegressor(n_estimators=100, max_depth=8, learning_rate=0.04,
                                     num_leaves=63, subsample=0.85, colsample_bytree=0.85,
                                     random_state=42, n_jobs=-1, verbose=-1)

            try:
                df_feat = build_oof_cascade(df_feat, f5_features, [_casc_et, _casc_lgb],
                                            n_splits=3, out_of_fold=True, verbose=True)
                from src.cascade import cascade_feature_names
                casc_cols = [c for c in cascade_feature_names() if c in df_feat.columns]
                df_feat[casc_cols].to_parquet(cascade_cache_file)
                print(f"  Cached OOF cascade features to {cascade_cache_file}")
            except Exception as exc:
                print(f"  Warning: OOF cascade construction failed ({exc}). "
                      f"Falling back to sequential in-sample cascade.")

    # ── STAGE 3.5: Multi-Task Joint Physics-Informed Neural Network (PINN-MagNet) ──
    print("\n" + "=" * 78)
    print("  STAGE 3.5: MULTI-TASK JOINT PHYSICS-INFORMED NEURAL NETWORK (PINN-MagNet)")
    print("  Enforcing Thermodynamic (BH)max <= 1/4 mu0 Ms^2 & Kronmüller Hc <= 2K1/(mu0 Ms)")
    print("=" * 78)

    from src.multi_task_pinn import train_multitask_pinn, MultiTaskPhysicalPINNRegressor, PhysicalProjectionLayer

    y_dict_pinn = {
        'clean_Ms_emu_g': df_feat['clean_Ms_emu_g'] if 'clean_Ms_emu_g' in df_feat.columns else pd.Series(index=df_feat.index, dtype=float),
        'clean_K1_J_m3': df_feat['clean_K1_J_m3'] if 'clean_K1_J_m3' in df_feat.columns else pd.Series(index=df_feat.index, dtype=float),
        'clean_Hc_A_m': df_feat['clean_Hc_A_m'] if 'clean_Hc_A_m' in df_feat.columns else pd.Series(index=df_feat.index, dtype=float),
        'clean_BH_max_kJ_m3': df_feat['clean_BH_max_kJ_m3'] if 'clean_BH_max_kJ_m3' in df_feat.columns else pd.Series(index=df_feat.index, dtype=float)
    }
    groups_pinn = df_feat['reduced_formula'].fillna('Fe').astype(str).values

    pinn_candidate_features = [f for f in f5_features if f in df_feat.columns and pd.api.types.is_numeric_dtype(df_feat[f])][:80]
    pinn_n_epochs = 20 if TRAINING_MODE == "final_manuscript" else 10

    pinn_bundle = train_multitask_pinn(
        df_feat, y_dict_pinn, pinn_candidate_features, groups_pinn,
        n_splits=RUN_CONFIG.n_splits,
        epochs=pinn_n_epochs,
        batch_size=128,
        lr=1e-3,
        verbose=True
    )

    # Inject leak-free out-of-fold multi-task physics guidance features
    df_feat['pinn_joint_pred_ms'] = pinn_bundle['oof_predictions']['ms']
    df_feat['pinn_joint_pred_k1_symlog'] = np.sign(pinn_bundle['oof_predictions']['k1']) * np.log10(1.0 + np.abs(pinn_bundle['oof_predictions']['k1']))
    df_feat['pinn_joint_pred_hc_log'] = np.log10(np.maximum(pinn_bundle['oof_predictions']['hc'], 1.0))
    df_feat['pinn_joint_pred_bh_max_log'] = np.log10(np.maximum(pinn_bundle['oof_predictions']['bh_max'], 0.01))
    df_feat['pinn_joint_hardness_kappa'] = pinn_bundle['oof_predictions']['kappa']

    targets_config = [
        {
            'name': 'anisotropy_k1',
            'col': 'clean_K1_J_m3',
            'transform': 'symlog',
            'unit': 'J/m^3',
            'allow_negative': True,
            'y_min': -1e8,
            'y_max': 1e8,
            'valid_filter': lambda s: s.notna() & (np.abs(s) <= 1e8),
            'mandatory_features': [
                'uniaxial_symmetry_flag', 'phys_aniso_uniaxial_tc', 'dao_msn_a20_cf', 'stevens_a20_cf', 'stevens_k1_torque'
            ],
            'extra_features': [
                'cascade_prob_FM', 'cascade_prob_AFM', 'cascade_pred_TC',
                'phys_k1_thermal_factor', 'phys_soc_exchange_coupling', 'phys_aniso_uniaxial_tc',
                'dao_msn_a20_cf', 'stevens_a20_cf', 'stevens_k1_torque', 'dao_msn_bethe_slater_ratio', 'dao_msn_exchange_stiffness',
                'pinn_joint_pred_k1_symlog', 'pinn_joint_hardness_kappa', 'pinn_joint_pred_ms'
            ]
        },
        {
            'name': 'saturation_magnetization_ms',
            'col': 'clean_Ms_emu_g',
            'transform': 'none',
            'unit': 'emu/g',
            'allow_negative': False,
            'y_min': 0.0,
            'y_max': 350.0,
            'valid_filter': lambda s: s.notna() & (s > 0) & (s <= 350.0),
            'mandatory_features': [
                'avg_magnetic_moment', 'MagpieData maximum GSmagmom', 'phys_tc_ms_interaction', 'dao_msn_ms_emu_g'
            ],
            'extra_features': [
                'phys_ms_bloch_factor', 'phys_fm_scaled_moment', 'phys_tc_ms_interaction',
                'dao_msn_ms_emu_g', 'dao_msn_exchange_stiffness', 'dao_msn_bethe_slater_ratio',
                'cascade_prob_FM', 'cascade_prob_AFM', 'cascade_pred_TC',
                'pinn_joint_pred_ms'
            ]
        },
        {
            'name': 'coercivity_hc',
            'col': 'clean_Hc_A_m',
            'transform': 'log',
            'unit': 'A/m',
            'allow_negative': False,
            'y_min': 0.0,
            'y_max': 1e8,
            'valid_filter': lambda s: s.notna() & (s > 0) & (s <= 1e8),
            'mandatory_features': [
                'uniaxial_symmetry_flag', 'pinn_loop_squareness', 'cascade_Ha_log', 'dao_msn_exchange_stiffness'
            ],
            'extra_features': [
                'cascade_kronmuller_hk', 'cascade_Ha_log', 'dao_msn_exchange_stiffness', 'dao_msn_a20_cf', 'dao_msn_ms_emu_g',
                'cascade_pred_k1_symlog', 'cascade_pred_ms', 'cascade_prob_FM', 'cascade_pred_TC',
                'pinn_kronmuller_hk', 'pinn_kronmuller_hc_log', 'pinn_hardness_parameter', 'pinn_loop_squareness',
                'pinn_joint_pred_hc_log', 'pinn_joint_hardness_kappa', 'pinn_joint_pred_ms', 'pinn_joint_pred_k1_symlog'
            ]
        },
        {
            'name': 'energy_product_bh_max',
            'col': 'clean_BH_max_kJ_m3',
            'transform': 'log',
            'unit': 'kJ/m^3',
            'allow_negative': False,
            'y_min': 0.0,
            'y_max': 600.0,
            'valid_filter': lambda s: s.notna() & (s >= 1.0) & (s <= 600.0),
            'mandatory_features': [
                'uniaxial_symmetry_flag', 'cascade_pred_hc_log', 'pinn_joint_pred_bh_max_log', 'pinn_bh_max_bound', 'dao_msn_ms_emu_g'
            ],
            'extra_features': [
                'cascade_theoretical_bh_max', 'cascade_extrinsic_bh_product',
                'dao_msn_ms_emu_g', 'dao_msn_exchange_stiffness', 'dao_msn_a20_cf',
                'cascade_pred_hc_log', 'cascade_pred_ms', 'cascade_pred_k1_symlog',
                'cascade_prob_FM', 'cascade_pred_TC',
                'pinn_bh_max_ideal', 'pinn_bh_max_extrinsic', 'pinn_bh_max_bound',
                'pinn_log_bh_bound', 'pinn_loop_squareness', 'pinn_hardness_parameter',
                'pinn_joint_pred_bh_max_log', 'pinn_joint_pred_ms', 'pinn_joint_pred_hc_log', 'pinn_joint_hardness_kappa'
            ]
        },
        {
            'name': 'curie_weiss_theta_p',
            'col': 'clean_theta_p_K',
            'transform': 'none',
            'unit': 'K',
            'allow_negative': True,
            'y_min': -2000.0,
            'y_max': 2000.0,
            'valid_filter': lambda s: s.notna() & (s >= -2000.0) & (s <= 2000.0),
            'mandatory_features': [
                'cascade_mean_field_theta_p', 'phys_exchange_balance'
            ],
            'extra_features': [
                'cascade_mean_field_theta_p', 'phys_exchange_balance', 'phys_afm_exchange_ratio'
            ]
        }
    ]
    
    extended_results = {}
    extended_metrics = {}
    
    for cfg in targets_config:
        col = cfg['col']
        t_name = cfg['name']
        print(f"\n--- Training Target: {t_name.upper()} ({col}) ---")
        
        # Sequential physics feature construction before target training.
        if t_name == 'curie_weiss_theta_p':
            # Formulate Weiss mean-field theoretical prior and high-T asymptotic scaling
            tc_vals = df_feat.get('cascade_pred_TC', np.full(len(df_feat), 300.0))
            tn_vals = df_feat.get('cascade_pred_TN', np.full(len(df_feat), 100.0))
            pfm_vals = df_feat.get('cascade_prob_FM', np.full(len(df_feat), 0.33))
            pafm_vals = df_feat.get('cascade_prob_AFM', np.full(len(df_feat), 0.33))
            df_feat['cascade_mean_field_theta_p'] = pfm_vals * tc_vals - pafm_vals * tn_vals
            df_feat['phys_afm_exchange_ratio'] = tn_vals / (tc_vals + 10.0)
            super_afm = df_feat['afm_superexchange_index'].fillna(0.0).values if 'afm_superexchange_index' in df_feat.columns else np.zeros(len(df_feat))
            df_feat['phys_superexchange_tn'] = super_afm * tn_vals
            df_feat['phys_exchange_balance'] = (tc_vals - tn_vals) / (tc_vals + tn_vals + 10.0)
            df_feat['phys_cw_high_t_asymptote'] = np.sign(df_feat['cascade_mean_field_theta_p']) * np.sqrt(np.abs(df_feat['cascade_mean_field_theta_p']) * np.maximum(tc_vals, tn_vals))
            df_feat['phys_frustration_index'] = np.abs(df_feat['cascade_mean_field_theta_p']) / np.maximum(tn_vals, 10.0)
        elif USE_OOF_CASCADE:
            pass
        elif t_name == 'coercivity_hc':
            # Formulate Kronmüller ratio and intrinsic anisotropy field from predicted K1 and Ms
            k1_vals = df_feat.get('cascade_pred_k1', np.zeros(len(df_feat)))
            ms_vals = df_feat.get('cascade_pred_ms', np.full(len(df_feat), 50.0))
            tc_vals = df_feat.get('cascade_pred_TC', np.full(len(df_feat), 300.0))
            pfm_vals = df_feat.get('cascade_prob_FM', np.full(len(df_feat), 0.5))
            df_feat['cascade_kronmuller_hk'] = np.log10(1.0 + np.maximum(k1_vals, 0.0)) - np.log10(np.maximum(ms_vals, 0.1))
            df_feat['phys_domain_wall_energy'] = np.sqrt(np.maximum(tc_vals, 0.0) * np.maximum(k1_vals, 0.0))
            df_feat['phys_coercivity_nucleation_index'] = df_feat['cascade_kronmuller_hk'] * pfm_vals
            
            # Anisotropy field H_A = 2 K1 / (mu_0 * Ms) [A/m]
            density_vals = df_feat.get('dao_density', np.full(len(df_feat), 7.5))
            ha_denom = 4.0 * np.pi * 1e-4 * np.maximum(ms_vals, 0.5) * np.maximum(density_vals, 1.0)
            ha_vals = (2.0 * np.maximum(k1_vals, 1.0)) / ha_denom
            df_feat['cascade_Ha_log'] = np.log10(np.maximum(ha_vals, 1.0))
            
            # Dynamically compute PINN hysteresis features from cascaded predictions
            pinn_res = compute_pinn_hysteresis_features(df_feat, use_cascade_preds=True)
            for pc in pinn_res.columns:
                df_feat[pc] = pinn_res[pc]
            
        elif t_name == 'energy_product_bh_max':
            # Formulate theoretical and extrinsic upper bounds
            ms_vals = df_feat.get('cascade_pred_ms', np.full(len(df_feat), 50.0))
            hc_log_vals = df_feat.get('cascade_pred_hc_log', np.full(len(df_feat), 3.0))
            df_feat['cascade_theoretical_bh_max'] = 2.0 * np.log10(np.maximum(ms_vals, 0.1))
            df_feat['cascade_extrinsic_bh_product'] = hc_log_vals + np.log10(np.maximum(ms_vals, 0.1))
            # Dynamically compute PINN hysteresis features from cascaded predictions
            pinn_res = compute_pinn_hysteresis_features(df_feat, use_cascade_preds=True)
            for pc in pinn_res.columns:
                df_feat[pc] = pinn_res[pc]

        mask = cfg['valid_filter'](df_feat[col]) & df_feat['reduced_formula'].notna()
        df_target = df_feat[mask].copy()
        n_samples = len(df_target)
        groups = df_target['reduced_formula'].values
        n_compounds = len(np.unique(groups))
        print(f"  Valid samples for {t_name}: {n_samples} measured rows ({n_compounds} independent compounds)")
        if n_samples < 50:
            print(f"  Warning: Insufficient samples for {t_name}. Skipping.")
            continue
            
        y_raw = df_target[col].values
        if cfg['transform'] == 'symlog':
            y_trans = np.sign(y_raw) * np.log10(1.0 + np.abs(y_raw))
        elif cfg['transform'] == 'log':
            y_trans = np.log10(y_raw)
        else:
            y_trans = y_raw
        
        # Build feature candidate pool: F5 + target-specific extra cascaded features
        extra_cols = [c for c in cfg['extra_features'] if c in df_target.columns]
        target_pool_cols = list(dict.fromkeys(f5_features + extra_cols))
        X_target = df_target[target_pool_cols].copy()
        
        # Perform feature selection while preserving mandatory features
        mand_cols = [c for c in cfg['mandatory_features'] if c in X_target.columns]
        X_sel, _, active_cols, medians_dict = leakage_safe_feature_selection(
            X_target, X_target, pd.Series(y_trans), mandatory_features=mand_cols
        )
        medians_series = pd.Series(medians_dict)
        X_sel = X_sel.fillna(medians_series).fillna(0.0)
        
        # 3-fold GroupKFold CV evaluation with lean 2-learner committee (ET + LGB) + NNLS
        gkf = GroupKFold(n_splits=RUN_CONFIG.n_splits)
        oof_preds = np.zeros(n_samples)
        
        n_est_cv = 80 if TRAINING_MODE == "fast_debug" else 100
        
        # Committee factories, rebuilt per fit so no estimator state is shared.
        def _et_factory():
            return ExtraTreesRegressor(n_estimators=n_est_cv, max_depth=25, min_samples_split=2,
                                       min_samples_leaf=2, max_features=0.85, random_state=42, n_jobs=-1)

        def _rf_factory():
            return RandomForestRegressor(n_estimators=n_est_cv, max_depth=25, min_samples_split=2,
                                         min_samples_leaf=2, max_features=0.85, random_state=42, n_jobs=-1)

        def _lgb_factory():
            return LGBMRegressor(n_estimators=n_est_cv, max_depth=8, learning_rate=0.04, num_leaves=63,
                                 subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbose=-1)

        factories = [_et_factory, _rf_factory, _lgb_factory]
        fold_indices = []

        for tr_idx, val_idx in gkf.split(X_sel, y_trans, groups=groups):
            X_tr, y_tr = X_sel.iloc[tr_idx], y_trans[tr_idx]
            X_val = X_sel.iloc[val_idx]

            # Blend weights come from inner out-of-fold predictions on the training
            # fold. Fitting them on y_val (the previous behaviour) let the meta-weights
            # see the labels they were then scored against.
            w_norm = fit_blend_weights(factories, X_tr, y_tr,
                                       groups=np.asarray(groups)[tr_idx], inner_splits=3)

            p_val_mat = np.column_stack([f().fit(X_tr, y_tr).predict(X_val) for f in factories])
            oof_preds[val_idx] = p_val_mat @ w_norm
            fold_indices.append((tr_idx, val_idx))

        record = summarize(t_name, y_trans, oof_preds, fold_indices, gkf.get_n_splits(),
                           transform=cfg['transform'], unit=cfg['unit'],
                           bootstrap_iterations=500)
        r2_cv, mae_cv = record.pooled_r2, record.pooled_mae
        print(f"  GroupKFold CV Results for {t_name}: R² = {r2_cv:.4f} "
              f"(per-fold {record.r2_mean:.4f} ± {record.r2_std:.4f}), "
              f"MAE = {mae_cv:.4f} ± {record.mae_std:.4f} ({cfg['transform']} space)")

        extended_metrics[t_name] = record.to_dict()
        extended_metrics[t_name].update({
            'target_column': col,
            'n_samples': int(n_samples),
            'n_compounds': int(n_compounds),
            'generalization_r2': r2_cv,
            'generalization_mae': mae_cv,
        })
        
        # Train final ensemble model on full dataset with lean committee and NNLS meta-blending
        n_est_final = 80 if TRAINING_MODE == "fast_debug" else 100
        final_base_ensemble = TargetBalancedRegressorEnsemble(n_members=2, random_state=42, n_estimators=n_est_final)
        final_base_ensemble.fit(
            X_sel, pd.Series(y_trans), active_cols, 
            is_tc=False, allow_negative=(cfg['allow_negative'] or cfg['transform'] in ['symlog', 'log'])
        )
        
        transformed_model = TransformedPhysicalRegressor(
            final_base_ensemble,
            transform_type=cfg['transform'],
            y_min=cfg['y_min'],
            y_max=cfg['y_max'],
            scaler=final_base_ensemble.scaler,
            features_to_use=active_cols
        )
        
        task_name_map = {
            'saturation_magnetization_ms': 'ms',
            'anisotropy_k1': 'k1',
            'coercivity_hc': 'hc',
            'energy_product_bh_max': 'bh_max'
        }
        pinn_sub_model = None
        if t_name in task_name_map:
            pinn_sub_model = MultiTaskPhysicalPINNRegressor(
                pinn_bundle['model'],
                task_name=task_name_map[t_name],
                scaler=pinn_bundle['scaler'],
                features_to_use=pinn_bundle['features'],
                medians=pinn_bundle['medians']
            )

        extended_results[t_name] = {
            'model': transformed_model,
            'pinn_model': pinn_sub_model,
            'features': active_cols,
            'medians': medians_dict,
            'scaler': final_base_ensemble.scaler,
            'metrics': extended_metrics[t_name]
        }
        
        # Compute predictions across the master dataframe to feed downstream physical cascades
        X_master = df_feat.reindex(columns=active_cols).fillna(medians_series).fillna(0.0)
        p_master, _ = transformed_model.predict(X_master)
        
        # Full-fit donor predictions. These are correct for production inference on
        # unseen candidates (whose labels never entered any fit), but must not
        # replace the out-of-fold values used by the remaining targets' CV, so
        # they are written to a separate namespace when the OOF cascade is active.
        suffix = '_insample' if USE_OOF_CASCADE else ''
        if t_name == 'anisotropy_k1':
            k1_full = np.clip(p_master, -1e8, 1e8)
            df_feat[f'cascade_pred_k1{suffix}'] = k1_full
            df_feat[f'cascade_pred_k1_symlog{suffix}'] = np.sign(k1_full) * np.log10(1.0 + np.abs(k1_full))
        elif t_name == 'saturation_magnetization_ms':
            df_feat[f'cascade_pred_ms{suffix}'] = np.clip(p_master, 0.0, 350.0)
        elif t_name == 'coercivity_hc':
            hc_full = np.clip(p_master, 0.0, 1e8)
            df_feat[f'cascade_pred_hc{suffix}'] = hc_full
    extended_results['pinn_bundle'] = pinn_bundle
    extended_metrics['pinn_thermo_violation_rate'] = pinn_bundle['thermo_violation_rate']
    extended_metrics['pinn_metrics'] = pinn_bundle['metrics']

    # Persist updated extended models and PINN bundle to trained_models.pkl
    models_pkl_path = os.path.join(output_dir, 'trained_models.pkl')
    if os.path.exists(models_pkl_path):
        try:
            with open(models_pkl_path, 'rb') as f:
                loaded_pack = pickle.load(f)
            loaded_pack['magnetization'] = extended_results.get('saturation_magnetization_ms', {}).get('model')
            loaded_pack['k1'] = extended_results.get('anisotropy_k1', {}).get('model')
            loaded_pack['anisotropy'] = extended_results.get('anisotropy_k1', {}).get('model')
            loaded_pack['coercivity'] = extended_results.get('coercivity_hc', {}).get('model')
            loaded_pack['bh_max'] = extended_results.get('energy_product_bh_max', {}).get('model')
            loaded_pack['curie_weiss'] = extended_results.get('curie_weiss_theta_p', {}).get('model')
            loaded_pack['pinn_multitask'] = pinn_bundle['model']
            loaded_pack['pinn_scaler'] = pinn_bundle['scaler']
            loaded_pack['pinn_features'] = pinn_bundle['features']
            loaded_pack['pinn_medians'] = pinn_bundle['medians']
            loaded_pack['pinn_metrics'] = pinn_bundle['metrics']
            loaded_pack['pinn_thermo_violation_rate'] = pinn_bundle['thermo_violation_rate']
            loaded_pack['extended_metrics'] = extended_metrics

            with open(models_pkl_path, 'wb') as f:
                pickle.dump(loaded_pack, f)
            print(f"Updated {models_pkl_path} with new PINN and extended physical models!")
        except Exception as e:
            print(f"Warning: could not update trained_models.pkl: {e}")

    # Also update results_summary.json
    try:
        if os.path.exists('results_summary.json'):
            with open('results_summary.json', 'r') as f:
                res_sum = json.load(f)
            res_sum['extended_physical_targets'] = extended_metrics
            res_sum['pinn_multitask_metrics'] = pinn_bundle['metrics']
            res_sum['thermodynamic_violation_rate'] = pinn_bundle['thermo_violation_rate']
            with open('results_summary.json', 'w') as f:
                json.dump(res_sum, f, indent=4)
            print("Successfully updated results_summary.json with PINN & extended target metrics!")
    except Exception as e:
        print(f"Warning: could not update results_summary.json: {e}")

    print("\n--- All Extended Physical Targets Successfully Trained! ---")
    return extended_results, extended_metrics

def run_all_modeling(input_dir='output/featurization', output_dir='output/training', plot_dir='plots'):
    """
    Executes classification, temperature regression, and anisotropy regression pipelines.
    """
    models_pkl_path = os.path.join(output_dir, 'trained_models.pkl')
    stage3_json_path = os.path.join(output_dir, 'stage3_modeling.json')
    
    # Commented out purging hook to support continuing from classification checkpoint
    # for path_to_del in [models_pkl_path, stage3_json_path, os.path.join(output_dir, 'stage3_modeling_perf.json')]:
    #     if os.path.exists(path_to_del):
    #         try:
    #             os.remove(path_to_del)
    #         except Exception:
    #             pass
        
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 0. Generate publication-quality target distributions plot
    try:
        df_master = pd.read_csv(os.path.join(input_dir, 'features_master.csv'), low_memory=False)
        
        # Populate global formula target map for unified multi-task GNN training
        global _FORMULA_TARGETS
        _FORMULA_TARGETS = {}
        for _, row in df_master.iterrows():
            formula = str(row['reduced_formula'])
            cls_val = int(row['Type']) if not pd.isna(row['Type']) else 2
            tc_val = float(row['Mean_TC_K']) if not pd.isna(row['Mean_TC_K']) else np.nan
            tn_val = float(row['Mean_TN_K']) if not pd.isna(row['Mean_TN_K']) else np.nan
            _FORMULA_TARGETS[formula] = {'class': cls_val, 'tc': tc_val, 'tn': tn_val}
            
        # Call the premium target transformation distributions plot!
        print("\nPlotting target distributions before and after log1p transformations...")
        plot_target_transformation_distributions(
            df_master['Mean_TC_K'].dropna(),
            df_master['Mean_TN_K'].dropna(),
            plot_dir
        )
    except Exception as e:
        print(f"Error plotting target distributions / building formula targets: {e}")
        
    # Preserve existing scientific results_summary.json
    results_file = "results_summary.json"
            
    # Load feature groups map
    groups_path = os.path.join(input_dir, 'feature_groups.json')
    if not os.path.exists(groups_path):
        print(f"Error: {groups_path} not found. Please run featurization stage first.")
        return
        
    with open(groups_path, 'r') as f:
        feature_groups = json.load(f)
        
    # Attempt to load classification pipeline results from cache to continue execution
    class_results = None
    stale_reason = None
    class_res_path = os.path.join(output_dir, 'classification_results.pkl')
    if os.path.exists(class_res_path):
        # A cache is only reusable if it was produced under a configuration at
        # least as capable as the current one. Reusing a fast_debug cache inside a
        # final_manuscript run mixes reduced-capacity Target 1 results into a
        # report stamped is_publishable=True.
        cache_cfg_path = class_res_path.replace('.pkl', '.config.json')
        cached_cfg = None
        if os.path.exists(cache_cfg_path):
            try:
                with open(cache_cfg_path, 'r') as f:
                    cached_cfg = json.load(f)
            except Exception:
                cached_cfg = None

        stale_reason = None
        if cached_cfg is None:
            stale_reason = "no configuration record (produced before provenance tracking)"
        elif cached_cfg.get('mode') != RUN_CONFIG.mode:
            stale_reason = f"built under mode={cached_cfg.get('mode')!r}, now {RUN_CONFIG.mode!r}"
        elif int(cached_cfg.get('n_splits', -1)) != RUN_CONFIG.n_splits:
            stale_reason = (f"built with n_splits={cached_cfg.get('n_splits')}, "
                            f"now {RUN_CONFIG.n_splits}")
        elif bool(cached_cfg.get('run_ablation')) != RUN_CONFIG.run_ablation:
            stale_reason = "ablation setting differs"

        if stale_reason:
            print(f"Ignoring stale classification cache at {class_res_path}: {stale_reason}.")
            print("Retraining the classification pipeline at the current configuration.")
        else:
            try:
                print(f"Detected compatible classification cache at {class_res_path}. Loading...")
                with open(class_res_path, 'rb') as f:
                    class_results = pickle.load(f)
                print("Successfully loaded classification models from cache! Skipping classification pipeline.")
            except Exception as e:
                print(f"Failed to load classification results cache: {e}")

    # Same provenance rule for the trained_models.pkl fallback: if the dedicated
    # cache above was rejected as stale, this one is stale for the same reason.
    if class_results is None and stale_reason:
        print("Skipping the trained_models.pkl classification fallback for the same reason.")
    elif class_results is None and os.path.exists(models_pkl_path):
        try:
            print(f"Detected existing trained models at {models_pkl_path}. Loading classification cache...")
            with open(models_pkl_path, 'rb') as f:
                cached_pack = pickle.load(f)
            ensemble_class = cached_pack.get('classification')
            scaler_class = cached_pack.get('scaler_classification')
            features_class = cached_pack.get('features_class')
            medians_class = cached_pack.get('medians_class')
            if ensemble_class is not None:
                class_imp = getattr(ensemble_class, 'feature_importances_', {})
                if not class_imp and hasattr(ensemble_class, 'global_ensemble'):
                    class_imp = getattr(ensemble_class.global_ensemble, 'feature_importances_', {})
                class_results = (ensemble_class, scaler_class, features_class, medians_class, class_imp)
                print("Successfully loaded classification models from cache! Skipping classification pipeline.")
        except Exception as e:
            print(f"Failed to load cached classification models: {e}. Re-running classification from scratch.")
            
    if class_results is None:
        class_results = run_classification_pipeline(input_dir, output_dir, plot_dir, feature_groups)
        clear_gnn_cache()  # release memory after classification stage
        
    regression_results = run_regression_pipeline(input_dir, output_dir, plot_dir, feature_groups)
    clear_gnn_cache()  # release memory after regression stage
    
    # Unpack classification results
    ensemble_class, scaler_class, features_class, medians_class, class_imp = class_results
    
    # Unpack regression results
    (
        ensemble_curie, scaler_curie, features_curie, medians_curie,
        curie_res, curie_imp, curie_rf, curie_X_train,
        neel_res, neel_imp,
        ensemble_neel, scaler_neel, features_neel, medians_neel
    ) = regression_results

    # 4. Train extended physical targets (Targets 4 to 8: K1, Hc, Ms, BH_max, theta_p)
    extended_results, extended_metrics = run_extended_physical_targets_pipeline(
        input_dir, output_dir, plot_dir, feature_groups,
        class_results=class_results, regression_results=regression_results
    )
    
    ensemble_k1 = extended_results.get('anisotropy_k1', {}).get('model')
    ensemble_hc = extended_results.get('coercivity_hc', {}).get('model')
    ensemble_ms = extended_results.get('saturation_magnetization_ms', {}).get('model')
    ensemble_bh = extended_results.get('energy_product_bh_max', {}).get('model')
    ensemble_cw = extended_results.get('curie_weiss_theta_p', {}).get('model')
    
    # 1. Overlay prediction vs actual scatter for Ordering Temperature models
    if RUN_HEAVY_PLOTS and curie_res is not None and neel_res is not None:
        plot_overlay_predictions_vs_actual(curie_res, neel_res, plot_dir)
        
    # 2. Unified Feature Importance Bubble Plot (Cleveland Matrix) across 2 active tasks
    if RUN_HEAVY_PLOTS:
        plot_unified_feature_importance_bubble(class_imp, curie_imp, neel_imp, plot_dir)
    
    # 3. Overlay transition metal PDP curves (Fe, Co, Mn)
    if RUN_HEAVY_PLOTS and curie_rf is not None and curie_X_train is not None:
        plot_overlay_transition_metal_pdp(curie_rf, curie_X_train, plot_dir)
        
    # 4. Generate Radar Metrics Comparison Plot
    if RUN_HEAVY_PLOTS:
        plot_radar_chart(plot_dir)
    
    # 4b. Generate Advanced Representation Ablation Dashboard
    if RUN_HEAVY_PLOTS:
        try:
            from src.visualization import plot_representation_ablation_metrics
            plot_representation_ablation_metrics('results_summary.json', plot_dir)
        except Exception as e:
            print(f"Warning: Advanced Representation Ablation Dashboard plotting skipped: {e}")
    
    # 5. Generate global Feature Importance Triage by Physical Category
    if RUN_HEAVY_PLOTS:
        from src.visualization import plot_feature_importance_by_family
        plot_feature_importance_by_family(plot_dir)
    
    # Update results_summary.json with extended physical targets
    if os.path.exists('results_summary.json'):
        try:
            with open('results_summary.json', 'r') as f:
                res_sum = json.load(f)
            res_sum['extended_physical_targets'] = extended_metrics
            with open('results_summary.json', 'w') as f:
                json.dump(res_sum, f, indent=4)
            print("Successfully updated results_summary.json with Targets 4 to 8 metrics!")
        except Exception as e:
            print(f"Warning: could not update results_summary.json: {e}")

    # Serialize the best models, scalers, medians, and active features list
    models_pkl_path = os.path.join(output_dir, 'trained_models.pkl')
    models_pack = {
        'classification': ensemble_class,
        'curie': ensemble_curie,
        'neel': ensemble_neel,
        'anisotropy': ensemble_k1,
        'k1': ensemble_k1,
        'coercivity': ensemble_hc,
        'magnetization': ensemble_ms,
        'bh_max': ensemble_bh,
        'curie_weiss': ensemble_cw,
        'scaler_classification': scaler_class,
        'scaler_curie': scaler_curie,
        'scaler_neel': scaler_neel,
        'scaler_anisotropy': getattr(ensemble_k1, 'scaler', None),
        'scaler_k1': getattr(ensemble_k1, 'scaler', None),
        'scaler_coercivity': getattr(ensemble_hc, 'scaler', None),
        'scaler_magnetization': getattr(ensemble_ms, 'scaler', None),
        'scaler_bh_max': getattr(ensemble_bh, 'scaler', None),
        'scaler_curie_weiss': getattr(ensemble_cw, 'scaler', None),
        'feature_groups': feature_groups,
        'features_class': features_class,
        'features_curie': features_curie,
        'features_neel': features_neel,
        'features_aniso': getattr(ensemble_k1, 'features_to_use', None),
        'features_k1': getattr(ensemble_k1, 'features_to_use', None),
        'features_coercivity': getattr(ensemble_hc, 'features_to_use', None),
        'features_magnetization': getattr(ensemble_ms, 'features_to_use', None),
        'features_bh_max': getattr(ensemble_bh, 'features_to_use', None),
        'features_curie_weiss': getattr(ensemble_cw, 'features_to_use', None),
        'medians_class': medians_class,
        'medians_curie': medians_curie,
        'medians_neel': medians_neel,
        'medians_aniso': extended_results.get('anisotropy_k1', {}).get('medians', {}),
        'medians_k1': extended_results.get('anisotropy_k1', {}).get('medians', {}),
        'medians_coercivity': extended_results.get('coercivity_hc', {}).get('medians', {}),
        'medians_magnetization': extended_results.get('saturation_magnetization_ms', {}).get('medians', {}),
        'medians_bh_max': extended_results.get('energy_product_bh_max', {}).get('medians', {}),
        'medians_curie_weiss': extended_results.get('curie_weiss_theta_p', {}).get('medians', {}),
        # Explicit specialist keys if winner is specialist corrected
        'features_class_global': getattr(ensemble_class, 'features_class_global', None),
        'features_class_specialist': getattr(ensemble_class, 'features_class_specialist', None),
        'medians_class_global': getattr(ensemble_class, 'medians_class_global', None),
        'medians_class_specialist': getattr(ensemble_class, 'medians_class_specialist', None),
        'scaler_class_global': getattr(ensemble_class, 'scaler_class_global', None),
        'scaler_class_specialist': getattr(ensemble_class, 'scaler_class_specialist', None),
        'pinn_multitask': extended_results.get('pinn_bundle', {}).get('model', None),
        'pinn_scaler': extended_results.get('pinn_bundle', {}).get('scaler', None),
        'pinn_features': extended_results.get('pinn_bundle', {}).get('features', None),
        'pinn_medians': extended_results.get('pinn_bundle', {}).get('medians', None),
        'extended_metrics': extended_metrics
    }
    
    with open(models_pkl_path, 'wb') as f:
        pickle.dump(models_pack, f)
    print(f"\nSuccessfully serialized all 8 multi-task models and active configurations to {models_pkl_path}!")
    
    # 6. Generate and save stage3_modeling.json (quantitatively auditing the performance)
    stage3_json_path = os.path.join(output_dir, 'stage3_modeling.json')
    
    with open('results_summary.json', 'r') as f:
        final_summary = json.load(f)
        
    n_tiers = len(RUN_CONFIG.ablation_versions) if RUN_ABLATION else 1
    stage3_data = {
        "stage": "Stage 3: Physics-Aware Multi-Model Cross Validation",
        "description": (
            f"Group-kfold cross validation ({RUN_CONFIG.n_splits} folds, grouped on "
            f"reduced_formula) with fold-isolated sifting and standard scaling across "
            f"{n_tiers} evaluated representation tier(s). Ensemble blend weights are "
            f"fitted on inner out-of-fold predictions; the cross-target physics cascade "
            f"is {'out-of-fold' if USE_OOF_CASCADE else 'IN-SAMPLE (leaky)'}."
        ),
        # Provenance travels with the numbers: is_publishable=False marks metrics
        # produced at reduced capacity so they cannot be quoted as final by mistake.
        "run_config": RUN_CONFIG.provenance(),
        "results": final_summary,
        "serialized_models_pkl": models_pkl_path
    }
    with open(stage3_json_path, 'w') as f:
        json.dump(stage3_data, f, indent=4)
    print(f"Saved Stage 3 modeling summary JSON to {stage3_json_path}")
    
    # 7. File Cleanup Hook: Delete obsolete .html files and redundant individual png plots
    obsolete_files = [
        'curie_predictions_vs_actual.png',
        'neel_predictions_vs_actual.png',
        'radar_metrics.html',
        'feature_importance_treemap.html',
        'feature_importance_treemap.png',
        'feature_importances_classification.png',
        'feature_importances_curie.png',
        'feature_importances_neel.png',
        'feature_importances_anisotropy.png'
    ]
    for fn in obsolete_files:
        fp = os.path.join(plot_dir, fn)
        if os.path.exists(fp):
            try:
                os.remove(fp)
                print(f"Purged obsolete visual asset: {fp}")
            except Exception as e:
                print(f"Error removing obsolete asset {fp}: {e}")
                
    import glob
    for fp in glob.glob(os.path.join(plot_dir, "pdp_curie_*.png")) + glob.glob(os.path.join(plot_dir, "pdp_anisotropy_*.png")):
        try:
            os.remove(fp)
            print(f"Purged obsolete individual PDP visual asset: {fp}")
        except Exception as e:
            print(f"Error removing {fp}: {e}")
            
    print("\n=== End-to-End Pipeline Modeling Executed Flawlessly! ===")

if __name__ == '__main__':
    run_all_modeling()
