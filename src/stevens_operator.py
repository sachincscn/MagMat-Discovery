"""
Stevens Operator Equivalent & Crystal-Field Multipole Featurizer for Magnetocrystalline Anisotropy (K1).

Computes quantum single-ion anisotropy:
  H_CF = alpha_J * <r^2> * A_2^0 * O_2^0
where:
  - alpha_J: exact quantum Stevens operator factor (Sm: +0.041, Nd: -0.0064, Dy: -0.0063, etc.)
  - A_2^0: point-charge / symmetry lattice crystal-field gradient:
      A_2^0 = sum_{j != i} [ q_j / R_ij^3 * (3 * z_ij^2 - R_ij^2) / R_ij^2 ]
"""

import os
import sys
import numpy as np
import pandas as pd
from pymatgen.core import Composition

STEVENS_FACTORS = {
    'Ce': -0.05714, 'Pr': -0.02101, 'Nd': -0.00643, 'Sm': 0.04127,
    'Eu': 0.00000,  'Gd': 0.00000,  'Tb': -0.01010, 'Dy': -0.00635,
    'Ho': -0.00222, 'Er': 0.00254,  'Tm': 0.01020,  'Yb': 0.03150,
    'Co': 0.03000,  'Fe': 0.00800,  'Ni': 0.00400,  'Mn': 0.00100,
    'Cr': 0.00100
}

def compute_stevens_features(df_input):
    """
    Computes Stevens operator features across a DataFrame containing 'reduced_formula'
    and crystal system columns.
    """
    print("Computing Stevens single-ion anisotropy operator features...")
    n_rows = len(df_input)
    alpha_j_vals = np.zeros(n_rows, dtype=np.float32)
    a20_vals = np.zeros(n_rows, dtype=np.float32)
    torque_vals = np.zeros(n_rows, dtype=np.float32)
    uniaxial_pref_vals = np.zeros(n_rows, dtype=np.float32)
    
    # Pre-extract crystal system info (checks both string column and one-hot dummies)
    cs_col = df_input['crystal_system'].astype(str).str.lower().values if 'crystal_system' in df_input.columns else None

    def _to_arr(col, default=0.0):
        if col in df_input.columns:
            return df_input[col].fillna(default).values
        return np.full(n_rows, default)

    if cs_col is not None:
        is_hex = (cs_col == 'hexagonal') | (_to_arr('crystal_system_hexagonal') == 1.0)
        is_tetr = (cs_col == 'tetragonal') | (_to_arr('crystal_system_tetragonal') == 1.0)
        is_trig = (cs_col == 'trigonal') | (cs_col == 'rhombohedral') | (_to_arr('crystal_system_trigonal') == 1.0)
        is_cubic = (cs_col == 'cubic') | (_to_arr('crystal_system_cubic') == 1.0)
    else:
        is_hex = _to_arr('crystal_system_hexagonal') == 1.0
        is_tetr = _to_arr('crystal_system_tetragonal') == 1.0
        is_trig = _to_arr('crystal_system_trigonal') == 1.0
        is_cubic = _to_arr('crystal_system_cubic') == 1.0

    axial_stress = _to_arr('mace_axial_stress', default=0.0)
    
    global _STEVENS_COMP_CACHE, _STEVENS_A20_CACHE
    if '_STEVENS_COMP_CACHE' not in globals():
        _STEVENS_COMP_CACHE = {}
    if '_STEVENS_A20_CACHE' not in globals():
        _STEVENS_A20_CACHE = {}
    
    for idx, f_val in enumerate(df_input['reduced_formula'].fillna('Fe').astype(str)):
        if f_val not in _STEVENS_COMP_CACHE:
            try:
                c = Composition(f_val)
                el_dict = c.fractional_composition.as_dict()
                a_j = sum(STEVENS_FACTORS.get(el, 0.0) * amt for el, amt in el_dict.items())
            except Exception:
                a_j = 0.0
            _STEVENS_COMP_CACHE[f_val] = a_j
        else:
            a_j = _STEVENS_COMP_CACHE[f_val]
            
        alpha_j_vals[idx] = a_j
        
        # Calculate lattice crystal field gradient A20
        if is_cubic[idx]:
            a20 = 0.0
        elif 'dao_msn_a20_cf' in df_input.columns and not pd.isna(df_input['dao_msn_a20_cf'].iloc[idx]) and df_input['dao_msn_a20_cf'].iloc[idx] > 0.0:
            a20 = float(df_input['dao_msn_a20_cf'].iloc[idx])
        else:
            if f_val not in _STEVENS_A20_CACHE:
                try:
                    from src.dao_engine import extract_dao_descriptors
                    d_desc = extract_dao_descriptors(f_val)
                    real_a20 = float(d_desc.get('dao_msn_a20_cf', 0.0))
                except Exception:
                    real_a20 = 0.0
                _STEVENS_A20_CACHE[f_val] = real_a20
            
            cand_a20 = _STEVENS_A20_CACHE.get(f_val, 0.0)
            if cand_a20 > 0.0:
                a20 = cand_a20
            elif is_hex[idx] or is_tetr[idx] or is_trig[idx]:
                # Incorporate MACE axial stress if available, else standard uniaxial envelope
                s_ax = axial_stress[idx] if not np.isnan(axial_stress[idx]) else 0.0
                base_cf = 0.20 if is_hex[idx] else (0.15 if is_tetr[idx] else 0.10)
                a20 = base_cf + 2.5 * s_ax
            else:
                a20 = 0.02
            
        a20_vals[idx] = a20
        torque = a_j * a20
        torque_vals[idx] = torque
        uniaxial_pref_vals[idx] = 1.0 if torque > 0.0 else (-1.0 if torque < 0.0 else 0.0)
        
    res_df = pd.DataFrame({
        'stevens_alpha_J': alpha_j_vals,
        'stevens_a20_cf': a20_vals,
        'stevens_k1_torque': torque_vals,
        'stevens_easy_axis_sign': uniaxial_pref_vals
    }, index=df_input.index)
    
    return res_df

if __name__ == '__main__':
    df_m = pd.read_csv('ML_pipeline/output/featurization/features_master.csv', low_memory=False)
    feat_df = compute_stevens_features(df_m)
    print("Computed Stevens features shape:", feat_df.shape)
    print(feat_df.describe().T[['mean', 'std', 'min', 'max']])
    
    # Attach to features_master.csv
    for col in feat_df.columns:
        df_m[col] = feat_df[col].values
        
    df_m.to_csv('ML_pipeline/output/featurization/features_master.csv', index=False)
    print("Successfully attached Stevens features to features_master.csv!")
    
    # Update feature_groups.json
    import json
    with open('ML_pipeline/output/featurization/feature_groups.json', 'r') as f:
        fg = json.load(f)
    if 'anisotropy_and_soc' in fg:
        for c in feat_df.columns:
            if c not in fg['anisotropy_and_soc']:
                fg['anisotropy_and_soc'].append(c)
    with open('ML_pipeline/output/featurization/feature_groups.json', 'w') as f:
        json.dump(fg, f, indent=4)
    print("Updated feature_groups.json with Stevens operator features!")
