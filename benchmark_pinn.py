"""
Benchmark script for Step 3: Differentiable PINN Hysteresis Operator.
Evaluates GroupKFold cross-validation on unique chemical formulas (zero formula leakage)
for:
  - Target 5: Coercivity Hc
  - Target 7: Energy Product (BH)max
Comparing performance Before vs After adding the PINN Hysteresis Operator.
"""

import os
import sys
import json
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
    get_ablation_features
)

def evaluate_target(target_name, df_feat, feature_groups, use_pinn=True):
    f5_features = get_ablation_features(feature_groups, 'F5', df_feat.columns)
    
    pinn_features = [
        'pinn_loop_squareness', 'pinn_bh_max_ideal', 'pinn_bh_max_extrinsic',
        'pinn_bh_max_bound', 'pinn_log_bh_bound', 'pinn_kronmuller_hk',
        'pinn_kronmuller_hc_log', 'pinn_hardness_parameter'
    ]
    
    cfg_map = {
        'coercivity_hc': {
            'col': 'clean_Hc_A_m',
            'transform': 'log',
            'valid_filter': lambda s: s.notna() & (s > 0) & (s <= 1e8),
            'base_mandatory': [
                'anisotropy_figure_of_merit', 'weighted_soc_constant',
                'uniaxial_symmetry_flag', 'mace_von_mises_stress', 'mace_axial_stress',
                'stevens_alpha_J', 'stevens_k1_torque'
            ],
            'pinn_mandatory': [
                'pinn_kronmuller_hk', 'pinn_kronmuller_hc_log', 'pinn_hardness_parameter', 'pinn_loop_squareness'
            ]
        },
        'energy_product_bh_max': {
            'col': 'clean_BH_max_kJ_m3',
            'transform': 'log',
            'valid_filter': lambda s: s.notna() & (s >= 1.0) & (s <= 600.0),
            'base_mandatory': [
                'uniaxial_symmetry_flag', 'dft_jarvis_magmom', 'mace_axial_stress',
                'stevens_alpha_J', 'stevens_k1_torque'
            ],
            'pinn_mandatory': [
                'pinn_bh_max_ideal', 'pinn_bh_max_extrinsic', 'pinn_bh_max_bound',
                'pinn_log_bh_bound', 'pinn_loop_squareness', 'pinn_hardness_parameter'
            ]
        }
    }
    
    cfg = cfg_map[target_name]
    col = cfg['col']
    mask = cfg['valid_filter'](df_feat[col]) & df_feat['reduced_formula'].notna()
    df_target = df_feat[mask].copy()
    n_samples = len(df_target)
    
    y_raw = df_target[col].values
    y_trans = np.log10(y_raw)
    groups = df_target['reduced_formula'].values
    
    mand_cols = list(cfg['base_mandatory'])
    if use_pinn:
        mand_cols += [c for c in cfg['pinn_mandatory'] if c in df_target.columns]
        target_pool = list(dict.fromkeys(f5_features + mand_cols))
    else:
        # Exclude PINN features for baseline comparison
        target_pool = [c for c in f5_features if c not in pinn_features]
        mand_cols = [c for c in mand_cols if c not in pinn_features]
        
    X_target = df_target[[c for c in target_pool if c in df_target.columns]].copy()
    
    X_sel, _, active_cols, medians_dict = leakage_safe_feature_selection(
        X_target, X_target, pd.Series(y_trans), mandatory_features=mand_cols
    )
    medians_series = pd.Series(medians_dict)
    X_sel = X_sel.fillna(medians_series).fillna(0.0)
    
    gkf = GroupKFold(n_splits=3)
    oof_preds = np.zeros(n_samples)
    n_est_cv = 80
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_sel, y_trans, groups=groups)):
        X_tr, X_val = X_sel.iloc[train_idx], X_sel.iloc[val_idx]
        y_tr, y_val = y_trans[train_idx], y_trans[val_idx]
        
        cv_ensemble = TargetBalancedRegressorEnsemble(n_members=1, random_state=42 + fold, n_estimators=n_est_cv)
        cv_ensemble.fit(X_tr, pd.Series(y_tr), active_cols, is_tc=False, allow_negative=False)
        pred, _ = cv_ensemble.predict(X_val)
        oof_preds[val_idx] = pred
        
    r2_cv = float(r2_score(y_trans, oof_preds))
    mae_cv = float(mean_absolute_error(y_trans, oof_preds))
    return n_samples, r2_cv, mae_cv

def main():
    feat_path = 'ML_pipeline/output/featurization/features_master.csv'
    groups_path = 'ML_pipeline/output/featurization/feature_groups.json'
    
    print("Loading features_master.csv...", flush=True)
    df_feat = pd.read_csv(feat_path, low_memory=False)
    with open(groups_path) as f:
        feature_groups = json.load(f)
        
    print("\n" + "=" * 75, flush=True)
    print("  STEP 3 QUANTITATIVE BENCHMARK: DIFFERENTIABLE PINN HYSTERESIS OPERATOR", flush=True)
    print("  (Strict 3-Fold GroupKFold Cross-Validation on Unique Chemical Formulas)", flush=True)
    print("=" * 75, flush=True)
    
    # 1. Target 7: (BH)max
    print("\nEvaluating Target 7: Energy Product (BH)max...", flush=True)
    n_bh_base, r2_bh_base, mae_bh_base = evaluate_target('energy_product_bh_max', df_feat, feature_groups, use_pinn=False)
    n_bh_pinn, r2_bh_pinn, mae_bh_pinn = evaluate_target('energy_product_bh_max', df_feat, feature_groups, use_pinn=True)
    
    print(f"\n--- Target 7: Energy Product (BH)max (N = {n_bh_pinn}) ---", flush=True)
    print(f"  Baseline (Without PINN) : R² = {r2_bh_base:.4f} | MAE = {mae_bh_base:.4f} dex", flush=True)
    print(f"  With PINN Operator      : R² = {r2_bh_pinn:.4f} | MAE = {mae_bh_pinn:.4f} dex", flush=True)
    delta_r2_bh = r2_bh_pinn - r2_bh_base
    delta_mae_bh = mae_bh_pinn - mae_bh_base
    print(f"  --> PINN Impact         : ΔR² = {delta_r2_bh:+.4f} | ΔMAE = {delta_mae_bh:+.4f} dex", flush=True)
    
    # 2. Target 5: Coercivity Hc
    print("\nEvaluating Target 5: Coercivity Hc...", flush=True)
    n_hc_base, r2_hc_base, mae_hc_base = evaluate_target('coercivity_hc', df_feat, feature_groups, use_pinn=False)
    n_hc_pinn, r2_hc_pinn, mae_hc_pinn = evaluate_target('coercivity_hc', df_feat, feature_groups, use_pinn=True)
    
    print(f"\n--- Target 5: Coercivity Hc (N = {n_hc_pinn}) ---", flush=True)
    print(f"  Baseline (Without PINN) : R² = {r2_hc_base:.4f} | MAE = {mae_hc_base:.4f} dex", flush=True)
    print(f"  With PINN Operator      : R² = {r2_hc_pinn:.4f} | MAE = {mae_hc_pinn:.4f} dex", flush=True)
    delta_r2_hc = r2_hc_pinn - r2_hc_base
    delta_mae_hc = mae_hc_pinn - mae_hc_base
    print(f"  --> PINN Impact         : ΔR² = {delta_r2_hc:+.4f} | ΔMAE = {delta_mae_hc:+.4f} dex", flush=True)
    
    print("\n" + "=" * 75, flush=True)
    print("BENCHMARK COMPLETE.", flush=True)
    print("=" * 75, flush=True)

if __name__ == '__main__':
    main()
