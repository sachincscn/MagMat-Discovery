"""
Benchmark script for innovative physical magnetic enhancements.
Evaluates GroupKFold cross-validation on unique chemical formulas (zero formula leakage).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from src.modeling import (
    leakage_safe_feature_selection,
    TargetBalancedRegressorEnsemble,
    get_ablation_features,
    TRAINING_MODE
)

def evaluate_target(target_name, df_feat, feature_groups):
    f5_features = get_ablation_features(feature_groups, 'F5', df_feat.columns)
    
    cfg_map = {
        'anisotropy_k1': {
            'col': 'clean_K1_J_m3',
            'transform': 'symlog',
            'valid_filter': lambda s: s.notna() & (np.abs(s) <= 1e8),
            'mandatory_features': [
                'anisotropy_figure_of_merit', 'weighted_soc_constant',
                're_anisotropy_sign_index', 'uniaxial_symmetry_flag',
                'phys_soc_exchange_coupling', 'phys_aniso_uniaxial_tc', 'cascade_pred_TC', 'cascade_prob_FM',
                'mace_axial_stress', 'mace_von_mises_stress', 'mace_hydrostatic_pressure',
                'stevens_alpha_J', 'stevens_a20_cf', 'stevens_k1_torque', 'stevens_easy_axis_sign'
            ],
            'extra_features': [
                'cascade_prob_FM', 'cascade_prob_AFM', 'cascade_pred_TC',
                'phys_k1_thermal_factor', 'phys_soc_exchange_coupling', 'phys_aniso_uniaxial_tc', 'phys_re_aniso_tc',
                'mace_axial_stress', 'mace_von_mises_stress', 'mace_hydrostatic_pressure',
                'stevens_alpha_J', 'stevens_a20_cf', 'stevens_k1_torque', 'stevens_easy_axis_sign'
            ]
        },
        'energy_product_bh_max': {
            'col': 'clean_BH_max_kJ_m3',
            'transform': 'log',
            'valid_filter': lambda s: s.notna() & (s >= 1.0) & (s <= 600.0),
            'mandatory_features': [
                'cascade_theoretical_bh_max', 'cascade_extrinsic_bh_product',
                'cascade_pred_hc_log', 'cascade_pred_ms', 'cascade_pred_k1_symlog',
                'uniaxial_symmetry_flag', 'dft_jarvis_magmom', 'mace_axial_stress'
            ],
            'extra_features': [
                'cascade_theoretical_bh_max', 'cascade_extrinsic_bh_product',
                'cascade_pred_hc_log', 'cascade_pred_ms', 'cascade_pred_k1_symlog',
                'cascade_prob_FM', 'cascade_pred_TC', 'dft_jarvis_magmom', 'mace_axial_stress'
            ]
        }
    }
    
    cfg = cfg_map[target_name]
    col = cfg['col']
    mask = cfg['valid_filter'](df_feat[col]) & df_feat['reduced_formula'].notna()
    df_target = df_feat[mask].copy()
    n_samples = len(df_target)
    
    y_raw = df_target[col].values
    if cfg['transform'] == 'symlog':
        y_trans = np.sign(y_raw) * np.log10(1.0 + np.abs(y_raw))
    elif cfg['transform'] == 'log':
        y_trans = np.log10(y_raw)
    else:
        y_trans = y_raw
        
    groups = df_target['reduced_formula'].values
    
    extra_cols = [c for c in cfg['extra_features'] if c in df_target.columns]
    target_pool_cols = list(dict.fromkeys(f5_features + extra_cols))
    X_target = df_target[target_pool_cols].copy()
    
    mand_cols = [c for c in cfg['mandatory_features'] if c in X_target.columns]
    X_sel, _, active_cols, medians_dict = leakage_safe_feature_selection(
        X_target, X_target, pd.Series(y_trans), mandatory_features=mand_cols
    )
    medians_series = pd.Series(medians_dict)
    X_sel = X_sel.fillna(medians_series).fillna(0.0)
    
    gkf = GroupKFold(n_splits=3)
    oof_preds = np.zeros(n_samples)
    n_est_cv = 120
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_sel, y_trans, groups=groups)):
        X_tr, X_val = X_sel.iloc[train_idx], X_sel.iloc[val_idx]
        y_tr, y_val = y_trans[train_idx], y_trans[val_idx]
        
        cv_ensemble = TargetBalancedRegressorEnsemble(n_members=2, random_state=42 + fold, n_estimators=n_est_cv)
        cv_ensemble.fit(X_tr, pd.Series(y_tr), active_cols, is_tc=False, allow_negative=(cfg['transform'] in ['symlog', 'none']))
        pred, _ = cv_ensemble.predict(X_val)
        oof_preds[val_idx] = pred
        
    r2_cv = float(r2_score(y_trans, oof_preds))
    mae_cv = float(mean_absolute_error(y_trans, oof_preds))
    return n_samples, r2_cv, mae_cv

def main():
    import json
    ROOT = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(ROOT, 'output', 'featurization')
    print("Loading features_master.csv and feature_groups.json...")
    df_feat = pd.read_csv(os.path.join(input_dir, 'features_master.csv'), low_memory=False)
    with open(os.path.join(input_dir, 'feature_groups.json')) as f:
        feature_groups = json.load(f)
        
    print("\n" + "=" * 70)
    print("  EVALUATING STEP 1 BENCHMARK: DATA RECOVERY & TRANSDUCTION")
    print("=" * 70)
    
    n_k1, r2_k1, mae_k1 = evaluate_target('anisotropy_k1', df_feat, feature_groups)
    print(f"\nTarget 4: Anisotropy K1:")
    print(f"  Sample Size N: {n_k1} (was 2,985 previously, +{n_k1 - 2985} recovered)")
    print(f"  GroupKFold CV R²: {r2_k1:.4f} (was 0.2776)")
    print(f"  GroupKFold CV MAE: {mae_k1:.4f} dex (was 1.8472)")
    
    n_bh, r2_bh, mae_bh = evaluate_target('energy_product_bh_max', df_feat, feature_groups)
    print(f"\nTarget 7: Energy Product (BH)max:")
    print(f"  Sample Size N: {n_bh} (was 982 previously, +{n_bh - 982} recovered)")
    print(f"  GroupKFold CV R²: {r2_bh:.4f} (was 0.3861)")
    print(f"  GroupKFold CV MAE: {mae_bh:.4f} dex (was 0.3089)")

if __name__ == '__main__':
    main()
