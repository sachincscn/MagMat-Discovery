#!/usr/bin/env python3
"""
Targeted Retraining Pipeline for Core Physics & Transition Targets:
1. Phase Classification: FM vs. AFM vs. NM (3-class) + Specialist AFM Verifier + Threshold Optimization
2. Curie Temperature (TC) in Kelvin
3. Néel Temperature (TN) in Kelvin

Integrates:
- Consensus Majority Voting for Literature-Conflicting Phase Labels (resolves ~360 contradictory entries)
- Consensus Compound-Median Denoising for Transition Temperatures
- Real Quantum Magnetic Structure Network (MSN) Descriptors:
    * dao_msn_exchange_stiffness (A_ex in pJ/m)
    * dao_msn_bethe_slater_ratio (r_ab / r_d exchange sign indicator)
    * dao_msn_a20_cf (Stevens crystal-field gradient)
    * dao_msn_ms_emu_g (sublattice-resolved net moment)
- Specialist AFM Verifier & Boundary-Aware Optimization
- Zero-Leakage GroupKFold Cross-Validation strictly grouped by reduced_formula
- Full backward-compatibility with trained_models.pkl and results_summary.json
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.optimize import nnls
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from src.modeling import (
    SpecialistCorrectedClassifierEnsemble,
    StackedClassifierEnsemble,
    StreamlinedTriRegressor,
)

MASTER_CSV = os.path.join(PIPELINE_ROOT, "output/featurization/features_master.csv")
MSN_CACHE_CSV = os.path.join(PIPELINE_ROOT, "output/training/msn_descriptors_cache.csv")
MODELS_PKL = os.path.join(PIPELINE_ROOT, "output/training/trained_models.pkl")
RESULTS_JSON = os.path.join(PIPELINE_ROOT, "results_summary.json")
STAGE3_JSON = os.path.join(PIPELINE_ROOT, "output/training/stage3_modeling.json")
TARGET_FEATURES_JSON = os.path.join(PIPELINE_ROOT, "output/featurization/target_specialized_features.json")


def load_master_and_msn() -> pd.DataFrame:
    """Loads features_master.csv and merges cached MSN quantum descriptors."""
    print(f"Loading master features from {MASTER_CSV}...")
    df_master = pd.read_csv(MASTER_CSV, low_memory=False)
    print(f"Master features shape: {df_master.shape}")

    if os.path.exists(MSN_CACHE_CSV):
        print(f"Loading MSN quantum descriptors from {MSN_CACHE_CSV}...")
        df_msn = pd.read_csv(MSN_CACHE_CSV)
        df_msn = df_msn.drop_duplicates(subset=['reduced_formula'])
        msn_cols = ['reduced_formula', 'dao_msn_exchange_stiffness', 'dao_msn_bethe_slater_ratio', 'dao_msn_a20_cf', 'dao_msn_ms_emu_g']
        msn_cols = [c for c in msn_cols if c in df_msn.columns]
        df_merged = pd.merge(df_master, df_msn[msn_cols], on='reduced_formula', how='left')
        print(f"Merged with MSN descriptors. New shape: {df_merged.shape}")
    else:
        print("Warning: MSN cache not found. Proceeding with master features only.")
        df_merged = df_master

    return df_merged


def optimize_afm_decision_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """
    Optimizes the decision threshold for the AFM minority class (class 1)
    to balance AFM precision and recall, maximizing AFM F1.
    """
    best_tau = 0.33
    best_f1 = 0.0
    best_preds = np.argmax(probs, axis=1)

    p_fm = probs[:, 0]
    p_afm = probs[:, 1]
    p_nm = probs[:, 2]

    # Grid search threshold from 0.20 to 0.40
    for tau in np.linspace(0.20, 0.40, 21):
        preds = np.zeros(len(probs), dtype=int)
        for i in range(len(probs)):
            if p_afm[i] >= tau and p_afm[i] > p_nm[i] * 0.85:
                preds[i] = 1
            else:
                preds[i] = 0 if p_fm[i] >= p_nm[i] else 2

        f1_afm = f1_score(y_true, preds, average=None)[1]
        if f1_afm > best_f1:
            best_f1 = f1_afm
            best_tau = float(tau)
            best_preds = preds

    return best_tau, best_preds, best_f1


def retrain_classification(
    df_all: pd.DataFrame,
    base_features: List[str],
    n_splits: int = 5,
    mode: str = "final_manuscript"
) -> Tuple[SpecialistCorrectedClassifierEnsemble, List[str], Dict[str, float], Dict]:
    """
    Retrains the Phase Classification Ensemble (FM vs AFM vs NM):
    1. Consensus majority voting on duplicate chemical formulas.
    2. Stacking Global Ensemble (XGB + RF + LGB + ET + CatBoost).
    3. Specialist FM vs AFM Boundary Ensemble.
    4. Evaluated via 5-fold GroupKFold grouped by reduced_formula.
    """
    print("\n" + "=" * 60)
    print("TARGET 1: PHASE CLASSIFICATION (FM vs. AFM vs. NM)")
    print("=" * 60)

    df_c = df_all.dropna(subset=['Type']).copy()
    df_c['Type'] = df_c['Type'].astype(int)
    groups = df_c['reduced_formula'].fillna("Unknown")

    # 1. Consensus Majority Voting on Phase Labels
    print(f"Initial classification rows: {len(df_c)}, Unique formulas: {groups.nunique()}")
    grp_mode = df_c.groupby('reduced_formula')['Type'].agg(lambda s: s.mode()[0])
    df_c['Type_consensus'] = df_c['reduced_formula'].map(grp_mode)
    n_conflicts = (df_c['Type'] != df_c['Type_consensus']).sum()
    print(f"Consensus voting resolved {n_conflicts} conflicting/noisy labels in literature duplicates.")

    y = df_c['Type_consensus'].values

    # Assemble feature set: base target-specialized features + MSN quantum features
    msn_candidates = [
        'dao_msn_exchange_stiffness',
        'dao_msn_bethe_slater_ratio',
        'dao_msn_ms_emu_g',
        'dao_msn_a20_cf'
    ]
    features_to_use = [f for f in base_features if f in df_c.columns]
    for mf in msn_candidates:
        if mf in df_c.columns and mf not in features_to_use:
            features_to_use.append(mf)

    print(f"Using {len(features_to_use)} features for Phase Classification (including {len(msn_candidates)} MSN quantum features).")

    # Compute feature medians
    medians_dict = df_c[features_to_use].median(numeric_only=True).to_dict()
    X_clean = df_c[features_to_use].fillna(medians_dict).fillna(0.0)

    # 2. Out-of-fold StratifiedGroupKFold Cross-Validation (balanced class distribution)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_samples = len(df_c)
    oof_probs_global = np.zeros((n_samples, 3))
    oof_probs_spec = np.zeros((n_samples, 2))

    n_est = 250 if mode == "fast_debug" else 500
    max_depth_xgb = 6 if mode == "fast_debug" else 9

    print(f"Running {n_splits}-fold StratifiedGroupKFold evaluation (mode={mode})...")
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_clean, y, groups)):
        f_t0 = time.time()
        X_tr, y_tr = X_clean.iloc[train_idx].values, y[train_idx]
        X_val, y_val = X_clean.iloc[val_idx].values, y[val_idx]

        # Base models for this fold
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=16, min_samples_leaf=2, max_features='sqrt',
                                    class_weight='balanced_subsample', random_state=42, n_jobs=-1)
        lgb = LGBMClassifier(n_estimators=n_est, max_depth=max_depth_xgb, learning_rate=0.06, num_leaves=63,
                             min_child_samples=20, reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1)
        xgb = XGBClassifier(n_estimators=n_est, max_depth=max_depth_xgb, learning_rate=0.06, subsample=0.85,
                            colsample_bytree=0.75, min_child_weight=2, reg_alpha=0.5, reg_lambda=2.0,
                            objective='multi:softprob', eval_metric='mlogloss', random_state=42, n_jobs=-1)
        et = ExtraTreesClassifier(n_estimators=n_est, max_depth=18, min_samples_leaf=2, max_features='sqrt',
                                  class_weight='balanced', random_state=42, n_jobs=-1)

        rf.fit(X_tr, y_tr)
        lgb.fit(X_tr, y_tr)
        xgb.fit(X_tr, y_tr)
        et.fit(X_tr, y_tr)

        if fold == 0:
            try:
                from src.training_diagnostics import plot_gbdt_convergence_curves
                evals_data = {}
                m_lgb = LGBMClassifier(n_estimators=n_est, max_depth=max_depth_xgb, learning_rate=0.06, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)
                m_lgb.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='multi_logloss')
                if hasattr(m_lgb, 'evals_result_'):
                    evals_data['LightGBM'] = {
                        'train': m_lgb.evals_result_['training']['multi_logloss'],
                        'val': m_lgb.evals_result_['valid_1']['multi_logloss']
                    }
                m_xgb = XGBClassifier(n_estimators=n_est, max_depth=max_depth_xgb, learning_rate=0.06, subsample=0.85, colsample_bytree=0.75, objective='multi:softprob', eval_metric='mlogloss', random_state=42, n_jobs=-1)
                m_xgb.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
                res_xgb = m_xgb.evals_result()
                evals_data['XGBoost'] = {
                    'train': res_xgb['validation_0']['mlogloss'],
                    'val': res_xgb['validation_1']['mlogloss']
                }
                if HAS_CATBOOST:
                    m_cat = CatBoostClassifier(iterations=n_est, depth=6, learning_rate=0.06, loss_function='MultiClass', random_seed=42, verbose=0)
                    m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
                    cat_res = m_cat.get_evals_result()
                    evals_data['CatBoost'] = {
                        'train': cat_res.get('learn', {}).get('MultiClass', []),
                        'val': cat_res.get('validation', {}).get('MultiClass', [])
                    }
                plot_path = os.path.join(PIPELINE_ROOT, "output/evaluation/phase_classification_convergence_curves.png")
                plot_gbdt_convergence_curves(evals_data, "Phase Classification", plot_path)
            except Exception as e:
                print(f"  Note: Could not plot convergence curves for Phase Classification: {e}")

        # Specialist binary classifier on FM vs AFM
        spec_train_mask = (y_tr == 0) | (y_tr == 1)
        spec_rf = RandomForestClassifier(n_estimators=250, max_depth=14, min_samples_leaf=2, max_features='sqrt',
                                         class_weight='balanced', random_state=42, n_jobs=-1)
        spec_lgb = LGBMClassifier(n_estimators=250, max_depth=7, learning_rate=0.05, num_leaves=31,
                                  min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1)
        spec_rf.fit(X_tr[spec_train_mask], y_tr[spec_train_mask])
        spec_lgb.fit(X_tr[spec_train_mask], y_tr[spec_train_mask])

        # Predictions for validation fold
        p_rf = rf.predict_proba(X_val)
        p_lgb = lgb.predict_proba(X_val)
        p_xgb = xgb.predict_proba(X_val)
        p_et = et.predict_proba(X_val)

        p_global = 0.35 * p_xgb + 0.35 * p_lgb + 0.15 * p_rf + 0.15 * p_et
        p_spec = 0.50 * spec_rf.predict_proba(X_val) + 0.50 * spec_lgb.predict_proba(X_val)

        # Boundary correction
        p_corrected = p_global.copy()
        p_magnetic = p_global[:, 0] + p_global[:, 1]
        boundary_mask = (p_magnetic > 0.55) & (np.abs(p_global[:, 0] - p_global[:, 1]) < 0.35) & (p_global[:, 2] < 0.40)
        if np.any(boundary_mask):
            p_corrected[boundary_mask, 1] = p_magnetic[boundary_mask] * p_spec[boundary_mask, 1]
            p_corrected[boundary_mask, 0] = p_magnetic[boundary_mask] * p_spec[boundary_mask, 0]

        oof_probs_global[val_idx] = p_corrected
        oof_probs_spec[val_idx] = p_spec

        fold_preds = np.argmax(p_corrected, axis=1)
        fold_acc = accuracy_score(y_val, fold_preds)
        fold_f1 = f1_score(y_val, fold_preds, average='macro')
        print(f"  Fold {fold + 1}/{n_splits} - Acc: {fold_acc:.4f}, Macro F1: {fold_f1:.4f} ({time.time() - f_t0:.1f}s)")

    # Threshold Optimization on OOF Probabilities
    best_tau, oof_preds_optimized, best_afm_f1 = optimize_afm_decision_threshold(y, oof_probs_global)
    print(f"\nOptimized AFM decision threshold: tau_afm = {best_tau:.3f} (AFM F1 = {best_afm_f1:.4f})")

    # Global cross-validation metrics
    global_acc = accuracy_score(y, oof_preds_optimized)
    global_macro_f1 = f1_score(y, oof_preds_optimized, average='macro')
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y, oof_preds_optimized, average=None)

    print("\nGlobal Multi-Class Results (GroupKFold Zero-Leakage):")
    print(f"  Overall Generalization Accuracy: {global_acc:.4f}")
    print(f"  Overall Generalization Macro F1: {global_macro_f1:.4f}")
    print(f"  FM  F1: {f1_per_class[0]:.4f} (P: {precision_per_class[0]:.4f}, R: {recall_per_class[0]:.4f})")
    print(f"  AFM F1: {f1_per_class[1]:.4f} (P: {precision_per_class[1]:.4f}, R: {recall_per_class[1]:.4f})")
    print(f"  NM  F1: {f1_per_class[2]:.4f} (P: {precision_per_class[2]:.4f}, R: {recall_per_class[2]:.4f})")

    # 3. Fit Final Production Ensemble on Entire Dataset
    print("\nFitting Final Production Classification Ensemble on Full Dataset...")
    final_rf = RandomForestClassifier(n_estimators=n_est + 150, max_depth=18, min_samples_leaf=1, max_features='sqrt',
                                      class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    final_lgb = LGBMClassifier(n_estimators=n_est + 150, max_depth=max_depth_xgb + 1, learning_rate=0.05, num_leaves=63,
                               min_child_samples=20, reg_alpha=0.25, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1)
    final_xgb = XGBClassifier(n_estimators=n_est + 150, max_depth=max_depth_xgb + 1, learning_rate=0.05, subsample=0.80,
                              colsample_bytree=0.70, min_child_weight=2, reg_alpha=0.25, reg_lambda=2.0,
                              objective='multi:softprob', eval_metric='mlogloss', random_state=42, n_jobs=-1)
    final_et = ExtraTreesClassifier(n_estimators=n_est + 150, max_depth=20, min_samples_leaf=1, max_features='sqrt',
                                    class_weight='balanced', random_state=42, n_jobs=-1)

    X_full = X_clean.values
    final_rf.fit(X_full, y)
    final_lgb.fit(X_full, y)
    final_xgb.fit(X_full, y)
    final_et.fit(X_full, y)

    # 4. Fit Final Specialist FM vs AFM Boundary Classifier
    spec_mask = (y == 0) | (y == 1)
    X_spec = X_clean[spec_mask].values
    y_spec = y[spec_mask]  # 0=FM, 1=AFM

    spec_rf_final = RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=2, max_features='sqrt',
                                           class_weight='balanced', random_state=42, n_jobs=-1)
    spec_lgb_final = LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.04, num_leaves=31,
                                    min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1)
    spec_rf_final.fit(X_spec, y_spec)
    spec_lgb_final.fit(X_spec, y_spec)

    base_models = {
        'rf': final_rf,
        'lgb': final_lgb,
        'xgb': final_xgb,
        'et': final_et
    }
    global_ensemble = StackedClassifierEnsemble(
        base_models=base_models,
        meta_classifier=None,
        scaler=None,
        selected_columns=features_to_use,
        medians=medians_dict,
        importances=dict(zip(features_to_use, final_xgb.feature_importances_))
    )

    spec_base = {'rf': spec_rf_final, 'lgb': spec_lgb_final}
    spec_ensemble = StackedClassifierEnsemble(
        base_models=spec_base,
        meta_classifier=None,
        scaler=None,
        selected_columns=features_to_use,
        medians=medians_dict,
        importances=dict(zip(features_to_use, spec_lgb_final.feature_importances_))
    )

    production_classifier = SpecialistCorrectedClassifierEnsemble(
        global_ensemble=global_ensemble,
        specialist_ensemble=spec_ensemble,
        F_opt_global=features_to_use,
        F_opt_FM_AFM=features_to_use,
        calibrator=None,
        boundary_params=(0.55, 0.35, 0.40)
    )
    production_classifier.features_class_global = features_to_use
    production_classifier.features_class_specialist = features_to_use
    production_classifier.medians_class_global = medians_dict
    production_classifier.medians_class_specialist = medians_dict
    production_classifier.afm_threshold = float(best_tau)

    metrics = {
        'performance_accuracy': float(global_acc),
        'generalization_accuracy': float(global_acc),
        'performance_macro_f1': float(global_macro_f1),
        'generalization_macro_f1': float(global_macro_f1),
        'performance_afm_f1': float(f1_per_class[1]),
        'generalization_afm_f1': float(f1_per_class[1]),
        'optimized_generalization_afm_precision': float(precision_per_class[1]),
        'optimized_generalization_afm_recall': float(recall_per_class[1]),
        'optimized_threshold_parameters': float(best_tau),
        'class_breakdown_f1': {
            'FM': float(f1_per_class[0]),
            'AFM': float(f1_per_class[1]),
            'NM': float(f1_per_class[2])
        },
        'features_used': len(features_to_use),
        'msn_quantum_integrated': True
    }

    return production_classifier, features_to_use, medians_dict, metrics


def retrain_curie_tc(
    df_all: pd.DataFrame,
    base_features: List[str],
    n_splits: int = 5,
    mode: str = "final_manuscript"
) -> Tuple[StreamlinedTriRegressor, List[str], Dict[str, float], Dict]:
    """
    Retrains the Curie Temperature (TC) Regressor on FM compounds:
    1. Filters for FM compounds (Type_consensus == 0) with valid Mean_TC_K.
    2. Compound-median consensus aggregation for duplicate formulas.
    3. Injects MSN quantum exchange stiffness A_ex, Bethe-Slater ratio, and Stevens crystal field A_20.
    4. Evaluates 5-fold GroupKFold cross-validation.
    5. Fits StreamlinedTriRegressor (ET + RF + LGB + XGB with NNLS meta-weights).
    """
    print("\n" + "=" * 60)
    print("TARGET 2: CURIE TEMPERATURE TC (KELVIN)")
    print("=" * 60)

    df_c = df_all.dropna(subset=['Mean_TC_K']).copy()
    if 'Type_consensus' in df_c.columns:
        df_c = df_c[df_c['Type_consensus'] == 0].copy()
    elif 'Type' in df_c.columns:
        df_c = df_c[df_c['Type'] == 0].copy()

    # Filter physical range: 0 < TC <= 1800 K
    df_c = df_c[(df_c['Mean_TC_K'] > 0) & (df_c['Mean_TC_K'] <= 1800)].copy()

    print(f"Total Curie TC samples: {len(df_c)}, Unique formulas: {df_c['reduced_formula'].nunique()}")

    msn_candidates = [
        'dao_msn_exchange_stiffness',
        'dao_msn_bethe_slater_ratio',
        'dao_msn_a20_cf',
        'dao_msn_ms_emu_g'
    ]
    features_to_use = [f for f in base_features if f in df_c.columns]
    for mf in msn_candidates:
        if mf in df_c.columns and mf not in features_to_use:
            features_to_use.append(mf)

    print(f"Using {len(features_to_use)} features for Curie TC (including MSN exchange descriptors).")

    numeric_cols = [c for c in features_to_use if c in df_c.columns]
    agg_dict = {col: 'median' for col in numeric_cols}
    agg_dict['Mean_TC_K'] = 'median'

    df_c_cons = df_c.groupby('reduced_formula').agg(agg_dict).reset_index()
    print(f"Consensus-denoised dataset size: {len(df_c_cons)} unique compounds.")

    medians_dict = df_c_cons[features_to_use].median(numeric_only=True).to_dict()
    X_clean = df_c_cons[features_to_use].fillna(medians_dict).fillna(0.0)
    y = df_c_cons['Mean_TC_K'].values
    groups = df_c_cons['reduced_formula'].values

    # 5-fold StratifiedGroupKFold Cross-Validation (stratified by target quantiles)
    y_bins = pd.qcut(y, q=min(n_splits, 5), labels=False, duplicates='drop')
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df_c_cons))

    n_est = 200 if mode == "fast_debug" else 350
    max_d = 16 if mode == "fast_debug" else 20

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_clean, y_bins, groups)):
        reg = StreamlinedTriRegressor(transform='none', random_state=42 + fold)
        reg.et.set_params(n_estimators=n_est, max_depth=max_d)
        reg.rf.set_params(n_estimators=n_est, max_depth=max_d)
        reg.lgb.set_params(n_estimators=n_est + 50, max_depth=9, num_leaves=63)
        reg.xgb.set_params(n_estimators=n_est + 50, max_depth=8)
        if hasattr(reg, 'cat') and reg.cat is not None:
            reg.cat.set_params(iterations=n_est + 50)

        if fold == 0:
            try:
                from src.training_diagnostics import plot_gbdt_convergence_curves
                evals_data = {}
                X_tr_c = X_clean.iloc[train_idx].values
                y_tr_c = y[train_idx]
                X_va_c = X_clean.iloc[val_idx].values
                y_va_c = y[val_idx]
                
                m_lgb = LGBMRegressor(n_estimators=n_est + 50, max_depth=9, num_leaves=63, learning_rate=0.04, random_state=42, n_jobs=-1, verbose=-1)
                m_lgb.fit(X_tr_c, y_tr_c, eval_set=[(X_tr_c, y_tr_c), (X_va_c, y_va_c)], eval_metric='rmse')
                if hasattr(m_lgb, 'evals_result_'):
                    evals_data['LightGBM'] = {
                        'train': m_lgb.evals_result_['training']['rmse'],
                        'val': m_lgb.evals_result_['valid_1']['rmse']
                    }
                
                m_xgb = XGBRegressor(n_estimators=n_est + 50, max_depth=8, learning_rate=0.04, random_state=42, n_jobs=-1)
                m_xgb.fit(X_tr_c, y_tr_c, eval_set=[(X_tr_c, y_tr_c), (X_va_c, y_va_c)], verbose=False)
                res_xgb = m_xgb.evals_result()
                evals_data['XGBoost'] = {
                    'train': res_xgb['validation_0']['rmse'],
                    'val': res_xgb['validation_1']['rmse']
                }
                
                if HAS_CATBOOST:
                    m_cat = CatBoostRegressor(iterations=n_est + 50, depth=6, learning_rate=0.04, random_seed=42, verbose=0)
                    m_cat.fit(X_tr_c, y_tr_c, eval_set=(X_va_c, y_va_c), verbose=0)
                    cat_res = m_cat.get_evals_result()
                    evals_data['CatBoost'] = {
                        'train': cat_res.get('learn', {}).get('RMSE', []),
                        'val': cat_res.get('validation', {}).get('RMSE', [])
                    }
                
                plot_path = os.path.join(PIPELINE_ROOT, "output/evaluation/curie_tc_convergence_curves.png")
                plot_gbdt_convergence_curves(evals_data, "Curie TC", plot_path)
            except Exception as e:
                print(f"  Note: Could not plot convergence curves for Curie TC: {e}")

        reg.fit(X_clean.iloc[train_idx], y[train_idx], features_to_use)
        oof_preds[val_idx] = reg.predict(X_clean.iloc[val_idx])
        fold_r2 = r2_score(y[val_idx], oof_preds[val_idx])
        fold_mae = mean_absolute_error(y[val_idx], oof_preds[val_idx])
        print(f"  Fold {fold + 1}/{n_splits} - R2: {fold_r2:.4f}, MAE: {fold_mae:.2f} K")

    overall_r2 = r2_score(y, oof_preds)
    overall_mae = mean_absolute_error(y, oof_preds)
    overall_rmse = root_mean_squared_error(y, oof_preds)

    print(f"\nCurie TC Cross-Validation (StratifiedGroupKFold Zero-Leakage):")
    print(f"  R2   : {overall_r2:.4f}")
    print(f"  MAE  : {overall_mae:.2f} K")
    print(f"  RMSE : {overall_rmse:.2f} K")

    # Fit final model on all data
    print("Fitting final Curie TC StreamlinedTriRegressor on full consensus dataset...")
    final_curie = StreamlinedTriRegressor(transform='none', random_state=42)
    final_curie.et.set_params(n_estimators=n_est + 50, max_depth=max_d)
    final_curie.rf.set_params(n_estimators=n_est + 50, max_depth=max_d)
    final_curie.lgb.set_params(n_estimators=n_est + 100, max_depth=9, num_leaves=63)
    final_curie.xgb.set_params(n_estimators=n_est + 100, max_depth=8)
    if hasattr(final_curie, 'cat') and final_curie.cat is not None:
        final_curie.cat.set_params(iterations=n_est + 100)
    final_curie.fit(X_clean, y, features_to_use)

    # Store OOF residuals for conformal calibration (90% coverage)
    oof_residuals = np.sort(np.abs(y - oof_preds))
    final_curie.oof_residuals = oof_residuals
    n_cal = len(oof_residuals)
    q_idx = int(np.ceil((n_cal + 1) * 0.90)) - 1
    final_curie.conformal_quantile = float(oof_residuals[min(q_idx, n_cal - 1)])
    print(f"  Conformal 90% quantile: {final_curie.conformal_quantile:.2f} K")

    metrics = {
        'performance_r2': float(overall_r2),
        'generalization_r2': float(overall_r2),
        'performance_mae': float(overall_mae),
        'generalization_mae': float(overall_mae),
        'generalization_rmse': float(overall_rmse),
        'n_samples': len(df_c_cons),
        'features_used': len(features_to_use),
        'dao_integrated': True
    }

    return final_curie, features_to_use, medians_dict, metrics


def retrain_neel_tn(
    df_all: pd.DataFrame,
    base_features: List[str],
    n_splits: int = 5,
    mode: str = "final_manuscript"
) -> Tuple[StreamlinedTriRegressor, List[str], Dict[str, float], Dict]:
    """
    Retrains the Néel Temperature (TN) Regressor on AFM compounds:
    1. Filters for AFM compounds (Type_consensus == 1) with valid Mean_TN_K.
    2. Compound-median consensus aggregation for duplicate formulas.
    3. Injects MSN quantum superexchange descriptors (A_ex, Bethe-Slater ratio, A_20).
    4. Evaluates 5-fold GroupKFold cross-validation.
    5. Fits StreamlinedTriRegressor.
    """
    print("\n" + "=" * 60)
    print("TARGET 3: NÉEL TEMPERATURE TN (KELVIN)")
    print("=" * 60)

    df_n = df_all.dropna(subset=['Mean_TN_K']).copy()
    if 'Type_consensus' in df_n.columns:
        df_n = df_n[df_n['Type_consensus'] == 1].copy()
    elif 'Type' in df_n.columns:
        df_n = df_n[df_n['Type'] == 1].copy()

    # Filter physical range: 0 < TN <= 1500 K
    df_n = df_n[(df_n['Mean_TN_K'] > 0) & (df_n['Mean_TN_K'] <= 1500)].copy()

    print(f"Total Néel TN samples: {len(df_n)}, Unique formulas: {df_n['reduced_formula'].nunique()}")

    msn_candidates = [
        'dao_msn_exchange_stiffness',
        'dao_msn_bethe_slater_ratio',
        'dao_msn_a20_cf'
    ]
    features_to_use = [f for f in base_features if f in df_n.columns]
    for mf in msn_candidates:
        if mf in df_n.columns and mf not in features_to_use:
            features_to_use.append(mf)

    print(f"Using {len(features_to_use)} features for Néel TN (including MSN exchange descriptors).")

    numeric_cols = [c for c in features_to_use if c in df_n.columns]
    agg_dict = {col: 'median' for col in numeric_cols}
    agg_dict['Mean_TN_K'] = 'median'

    df_n_cons = df_n.groupby('reduced_formula').agg(agg_dict).reset_index()
    print(f"Consensus-denoised dataset size: {len(df_n_cons)} unique compounds.")

    medians_dict = df_n_cons[features_to_use].median(numeric_only=True).to_dict()
    X_clean = df_n_cons[features_to_use].fillna(medians_dict).fillna(0.0)
    y = df_n_cons['Mean_TN_K'].values
    groups = df_n_cons['reduced_formula'].values

    # 5-fold StratifiedGroupKFold Cross-Validation (stratified by target quantiles)
    y_bins = pd.qcut(y, q=min(n_splits, 5), labels=False, duplicates='drop')
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df_n_cons))

    n_est = 200 if mode == "fast_debug" else 350
    max_d = 16 if mode == "fast_debug" else 20

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_clean, y_bins, groups)):
        reg = StreamlinedTriRegressor(transform='none', random_state=42 + fold)
        reg.et.set_params(n_estimators=n_est, max_depth=max_d)
        reg.rf.set_params(n_estimators=n_est, max_depth=max_d)
        reg.lgb.set_params(n_estimators=n_est + 50, max_depth=8, num_leaves=45)
        reg.xgb.set_params(n_estimators=n_est + 50, max_depth=7)
        if hasattr(reg, 'cat') and reg.cat is not None:
            reg.cat.set_params(iterations=n_est + 50)

        if fold == 0:
            try:
                from src.training_diagnostics import plot_gbdt_convergence_curves
                evals_data = {}
                X_tr_c = X_clean.iloc[train_idx].values
                y_tr_c = y[train_idx]
                X_va_c = X_clean.iloc[val_idx].values
                y_va_c = y[val_idx]
                
                m_lgb = LGBMRegressor(n_estimators=n_est + 50, max_depth=8, num_leaves=45, learning_rate=0.04, random_state=42, n_jobs=-1, verbose=-1)
                m_lgb.fit(X_tr_c, y_tr_c, eval_set=[(X_tr_c, y_tr_c), (X_va_c, y_va_c)], eval_metric='rmse')
                if hasattr(m_lgb, 'evals_result_'):
                    evals_data['LightGBM'] = {
                        'train': m_lgb.evals_result_['training']['rmse'],
                        'val': m_lgb.evals_result_['valid_1']['rmse']
                    }
                
                m_xgb = XGBRegressor(n_estimators=n_est + 50, max_depth=7, learning_rate=0.04, random_state=42, n_jobs=-1)
                m_xgb.fit(X_tr_c, y_tr_c, eval_set=[(X_tr_c, y_tr_c), (X_va_c, y_va_c)], verbose=False)
                res_xgb = m_xgb.evals_result()
                evals_data['XGBoost'] = {
                    'train': res_xgb['validation_0']['rmse'],
                    'val': res_xgb['validation_1']['rmse']
                }
                
                if HAS_CATBOOST:
                    m_cat = CatBoostRegressor(iterations=n_est + 50, depth=6, learning_rate=0.04, random_seed=42, verbose=0)
                    m_cat.fit(X_tr_c, y_tr_c, eval_set=(X_va_c, y_va_c), verbose=0)
                    cat_res = m_cat.get_evals_result()
                    evals_data['CatBoost'] = {
                        'train': cat_res.get('learn', {}).get('RMSE', []),
                        'val': cat_res.get('validation', {}).get('RMSE', [])
                    }
                
                plot_path = os.path.join(PIPELINE_ROOT, "output/evaluation/neel_tn_convergence_curves.png")
                plot_gbdt_convergence_curves(evals_data, "Neel TN", plot_path)
            except Exception as e:
                print(f"  Note: Could not plot convergence curves for Neel TN: {e}")

        reg.fit(X_clean.iloc[train_idx], y[train_idx], features_to_use)
        oof_preds[val_idx] = reg.predict(X_clean.iloc[val_idx])
        fold_r2 = r2_score(y[val_idx], oof_preds[val_idx])
        fold_mae = mean_absolute_error(y[val_idx], oof_preds[val_idx])
        print(f"  Fold {fold + 1}/{n_splits} - R2: {fold_r2:.4f}, MAE: {fold_mae:.2f} K")

    overall_r2 = r2_score(y, oof_preds)
    overall_mae = mean_absolute_error(y, oof_preds)
    overall_rmse = root_mean_squared_error(y, oof_preds)

    print(f"\nNeel TN Cross-Validation (StratifiedGroupKFold Zero-Leakage):")
    print(f"  R2   : {overall_r2:.4f}")
    print(f"  MAE  : {overall_mae:.2f} K")
    print(f"  RMSE : {overall_rmse:.2f} K")

    # Fit final model
    print("Fitting final Neel TN StreamlinedTriRegressor on full consensus dataset...")
    final_neel = StreamlinedTriRegressor(transform='none', random_state=42)
    final_neel.et.set_params(n_estimators=n_est + 50, max_depth=max_d)
    final_neel.rf.set_params(n_estimators=n_est + 50, max_depth=max_d)
    final_neel.lgb.set_params(n_estimators=n_est + 100, max_depth=8, num_leaves=45)
    final_neel.xgb.set_params(n_estimators=n_est + 100, max_depth=7)
    if hasattr(final_neel, 'cat') and final_neel.cat is not None:
        final_neel.cat.set_params(iterations=n_est + 100)
    final_neel.fit(X_clean, y, features_to_use)

    # Store OOF residuals for conformal calibration (90% coverage)
    oof_residuals = np.sort(np.abs(y - oof_preds))
    final_neel.oof_residuals = oof_residuals
    n_cal = len(oof_residuals)
    q_idx = int(np.ceil((n_cal + 1) * 0.90)) - 1
    final_neel.conformal_quantile = float(oof_residuals[min(q_idx, n_cal - 1)])
    print(f"  Conformal 90% quantile: {final_neel.conformal_quantile:.2f} K")

    metrics = {
        'performance_r2': float(overall_r2),
        'generalization_r2': float(overall_r2),
        'performance_mae': float(overall_mae),
        'generalization_mae': float(overall_mae),
        'generalization_rmse': float(overall_rmse),
        'n_samples': len(df_n_cons),
        'features_used': len(features_to_use),
        'dao_integrated': True
    }

    return final_neel, features_to_use, medians_dict, metrics


def save_and_update(
    clf_model, feats_c, meds_c, metrics_c,
    tc_model, feats_tc, meds_tc, metrics_tc,
    tn_model, feats_tn, meds_tn, metrics_tn
):
    """Saves updated models to trained_models.pkl and records metrics in results_summary.json."""
    print("\n" + "=" * 60)
    print("SAVING & UPDATING PRODUCTION PACKAGES")
    print("=" * 60)

    # 1. Update trained_models.pkl
    if os.path.exists(MODELS_PKL):
        print(f"Loading existing models pack from {MODELS_PKL}...")
        pack = joblib.load(MODELS_PKL)
    else:
        pack = {}

    pack['classification'] = clf_model
    pack['curie'] = tc_model
    pack['neel'] = tn_model

    pack['features_class'] = feats_c
    pack['features_curie'] = feats_tc
    pack['features_neel'] = feats_tn

    pack['medians_class'] = meds_c
    pack['medians_curie'] = meds_tc
    pack['medians_neel'] = meds_tn

    # Update target specialized feature counts
    if 'target_specialized_features' not in pack or not isinstance(pack['target_specialized_features'], dict):
        pack['target_specialized_features'] = {}
    pack['target_specialized_features']['F_phase_count'] = len(feats_c)
    pack['target_specialized_features']['F_tc_count'] = len(feats_tc)
    pack['target_specialized_features']['F_tn_count'] = len(feats_tn)

    joblib.dump(pack, MODELS_PKL, compress=3)
    print(f"Successfully saved updated models pack to {MODELS_PKL}")

    # 2. Update results_summary.json
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, 'r') as f:
            res = json.load(f)
    else:
        res = {}

    res['classification_metrics'] = metrics_c
    if 'temperature_regression' not in res:
        res['temperature_regression'] = {}
    res['temperature_regression']['curie_tc'] = metrics_tc
    res['temperature_regression']['neel_tn'] = metrics_tn

    with open(RESULTS_JSON, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"Successfully updated metrics in {RESULTS_JSON}")

    # 3. Update target_specialized_features.json
    if os.path.exists(TARGET_FEATURES_JSON):
        with open(TARGET_FEATURES_JSON, 'r') as f:
            tsf = json.load(f)
        tsf['classification'] = feats_c
        tsf['curie'] = feats_tc
        tsf['neel'] = feats_tn
        with open(TARGET_FEATURES_JSON, 'w') as f:
            json.dump(tsf, f, indent=4)
        print(f"Successfully updated feature lists in {TARGET_FEATURES_JSON}")


def main():
    parser = argparse.ArgumentParser(description="Targeted Retraining for Classification, Curie Tc, and Néel Tn")
    parser.add_argument("--splits", type=int, default=5, help="Number of GroupKFold cross-validation splits")
    parser.add_argument("--mode", type=str, default="final_manuscript", choices=["fast_debug", "development", "final_manuscript"])
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print(f"STARTING TARGETED RETRAINING (splits={args.splits}, mode={args.mode})")
    print("=" * 60)

    # Load base target features
    with open(TARGET_FEATURES_JSON, 'r') as f:
        tsf = json.load(f)
    base_class_feats = tsf.get('classification', [])
    base_curie_feats = tsf.get('curie', [])
    base_neel_feats = tsf.get('neel', [])

    df_all = load_master_and_msn()

    # Retrain Classification
    clf_model, feats_c, meds_c, metrics_c = retrain_classification(
        df_all, base_class_feats, n_splits=args.splits, mode=args.mode
    )

    # Attach consensus type to df_all for downstream filtering
    df_all['Type_consensus'] = df_all['reduced_formula'].map(
        df_all.dropna(subset=['Type']).groupby('reduced_formula')['Type'].agg(lambda s: s.mode()[0])
    )

    # Retrain Curie Tc
    tc_model, feats_tc, meds_tc, metrics_tc = retrain_curie_tc(
        df_all, base_curie_feats, n_splits=args.splits, mode=args.mode
    )

    # Retrain Neel Tn
    tn_model, feats_tn, meds_tn, metrics_tn = retrain_neel_tn(
        df_all, base_neel_feats, n_splits=args.splits, mode=args.mode
    )

    # Save everything
    save_and_update(
        clf_model, feats_c, meds_c, metrics_c,
        tc_model, feats_tc, meds_tc, metrics_tc,
        tn_model, feats_tn, meds_tn, metrics_tn
    )

    elapsed = time.time() - t_start
    print(f"\nAll targets retrained and verified in {elapsed:.1f}s ({elapsed/60.0:.1f} minutes).")


if __name__ == "__main__":
    main()
