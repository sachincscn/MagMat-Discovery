"""
Data Recovery and Relational Transduction for Magnetic Properties Pipeline.
Recovers unharvested experimental records from NEMAD:
- Sanitizes complex chemical formula strings (GBD, suffixes, balance flags, solid solutions).
- Parses raw (BH_Max) into standardized clean_BH_max_kJ_m3.
- Parses Anisotropy_Field (H_A) and applies relational transduction: K1 = 0.5 * mu0 * H_A * M_s.
- Updates target columns in features_master.csv.
"""

import os
import sys
import re
import ast
import numpy as np
import pandas as pd
from pymatgen.core import Composition

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

ARCHETYPES = {
    'ND-FE-B': 'Nd2Fe14B',
    'NDFEB': 'Nd2Fe14B',
    'SM-CO': 'SmCo5',
    'SMCO': 'SmCo5',
    'SM2CO17': 'Sm2Co17',
    'SMCO5': 'SmCo5',
    'ALNICO': 'Fe35Co35Ni15Al8Ti5',
    'BARIUM HEXAFERRITE': 'BaFe12O19',
    'STRONTIUM HEXAFERRITE': 'SrFe12O19',
    'BAM': 'BaFe12O19',
    'SRM': 'SrFe12O19',
}

def sanitize_formula(raw):
    """Sanitizes experimental chemical formula strings to valid reduced formulas."""
    if not raw or str(raw).strip() in ['', 'nan', 'NA', 'None']:
        return None
    s = str(raw).strip()
    
    # 1. Direct archetype lookup
    s_upper = s.upper().replace(' MAGNET', '').replace(' ALLOY', '').strip()
    for arch, rep in ARCHETYPES.items():
        if arch in s_upper:
            return rep
            
    # 2. Strip additives and processing prefixes/suffixes
    s = re.split(r'\s+(?:with|\+|\/|\&)\s+', s, flags=re.IGNORECASE)[0]
    s = re.sub(r'\s*\+\s*[\d\.]+\s*(?:wt%|at%|vol%).*$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+(?:GBD|diffused|sintered|annealed|ribbon|film|bulk|powder).*$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^[M\-]-(Sr|Ba)', r'\1', s)
    
    # 3. Clean non-stoichiometric delta, -x, +x, (1), (2)
    s = re.sub(r'[−\-]\s*[\delta\δxXyYzZ]', '', s)
    s = re.sub(r'\+\s*[\delta\δxXyYzZ]', '', s)
    s = re.sub(r'\([0-9]+\)$', '', s)
    
    # 4. Handle balanced iron: e.g. Fe_bal, FeBal, Fe(balance) -> Fe
    s = re.sub(r'Fe[_]?bal(?:ance)?', 'Fe', s, flags=re.IGNORECASE)
    
    # 5. Handle solid solution parentheses without numbers: (Pr,Nd) -> (Pr0.5Nd0.5)
    def resolve_commas(match):
        elements = [e.strip() for e in match.group(1).split(',')]
        frac = 1.0 / len(elements)
        return ''.join(f'{el}{frac:.3f}' for el in elements)
    s = re.sub(r'\(([A-Z][a-z]?(?:\s*,\s*[A-Z][a-z]?)+)\)', resolve_commas, s)
    
    try:
        c = Composition(s)
        return c.reduced_formula
    except Exception:
        return None

def parse_dict_entry(val_raw):
    """Safely parses raw dictionary string from NEMAD."""
    if not val_raw or pd.isna(val_raw):
        return None, None
    s = str(val_raw).strip()
    try:
        d = ast.literal_eval(s)
        val = str(d.get('Value', '')).replace(',', '').strip()
        unit = str(d.get('Units', '')).strip()
    except Exception:
        m_val = re.search(r'[\'\"]Value[\'\"]:\s*[\'\"]([^\'\"]+)[\'\"]', s)
        m_unit = re.search(r'[\'\"]Units[\'\"]:\s*[\'\"]([^\'\"]+)[\'\"]', s)
        val = m_val.group(1).replace(',', '').strip() if m_val else ''
        unit = m_unit.group(1).strip() if m_unit else ''
        
    if not val:
        return None, None
    # If range like '300-320' or multiple values, take mean or first
    if any(sep in val for sep in ['-', 'to', '/', ',']):
        parts = re.split(r'[-/,\s]|to', val)
        nums = []
        for p in parts:
            try:
                nums.append(float(re.sub(r'[^\d\.\-]', '', p)))
            except Exception:
                pass
        if nums:
            return float(np.mean(nums)), unit
        return None, None
        
    try:
        return float(re.sub(r'[^\d\.\-]', '', val)), unit
    except Exception:
        return None, None

def parse_bh_max_entry(val_raw):
    """Standardizes (BH_Max) to kJ/m^3."""
    val, unit = parse_dict_entry(val_raw)
    if val is None or val <= 0:
        return None
    unit_l = str(unit).lower()
    
    # 1 MGOe = 1 MGsOe = 7.9577 kJ/m^3
    if 'mgoe' in unit_l or 'mgsoe' in unit_l or 'mg' in unit_l:
        val_kj = val * 7.9577
    elif 'kj/m' in unit_l or 'kj m' in unit_l or 'kj/m3' in unit_l:
        val_kj = val
    elif 'j/m' in unit_l:
        val_kj = val / 1000.0
    else:
        # If unit missing or unparsed, check range
        if val < 80.0:  # clearly MGOe
            val_kj = val * 7.9577
        else:
            val_kj = val
            
    if 0.5 <= val_kj <= 650.0:
        return float(val_kj)
    return None

def parse_ha_entry(val_raw):
    """Standardizes Anisotropy Field to Tesla (mu0 * H_A)."""
    val, unit = parse_dict_entry(val_raw)
    if val is None or val <= 0:
        return None
    unit_l = str(unit).lower()
    
    # 1 T = 10 kOe = 10,000 Oe = 795.77 kA/m
    if 'koe' in unit_l:
        val_t = val / 10.0
    elif 'oe' in unit_l:
        val_t = val / 10000.0
    elif 't' in unit_l and 'mt' not in unit_l:
        val_t = val
    elif 'mt' in unit_l:
        val_t = val / 1000.0
    elif 'ka/m' in unit_l:
        val_t = val * 4.0 * np.pi * 1e-4
    elif 'a/m' in unit_l:
        val_t = val * 4.0 * np.pi * 1e-7
    elif 'ma/m' in unit_l:
        val_t = val * 4.0 * np.pi * 1e-1
    else:
        if val < 50.0:
            val_t = val  # likely Tesla
        else:
            val_t = val / 10000.0  # likely Oe
            
    if 0.001 <= val_t <= 120.0:
        return float(val_t)
    return None

def run_recovery():
    print("=" * 70)
    print("  STEP 1: UNHARVESTED DATA RECOVERY & RELATIONAL TRANSDUCTION")
    print("=" * 70)
    
    raw_path = os.path.join(PIPELINE_ROOT, 'Dataset', 'magnetic_anisotropy_materials.csv')
    master_path = os.path.join(PIPELINE_ROOT, 'output', 'featurization', 'features_master.csv')
    
    print(f"Reading raw anisotropy dataset from {raw_path}...")
    df_raw = pd.read_csv(raw_path, low_memory=False)
    print(f"Total raw records: {len(df_raw)}")
    
    print("Sanitizing chemical formulas across raw dataset...")
    df_raw['clean_formula'] = df_raw['Material_Name'].apply(sanitize_formula)
    valid_formula_count = df_raw['clean_formula'].notna().sum()
    print(f"Successfully sanitized valid formulas: {valid_formula_count} / {len(df_raw)} ({valid_formula_count/len(df_raw)*100:.1f}%)")
    
    # 1. Recover (BH_Max)
    print("\nRecovering (BH_Max) entries...")
    df_raw['recovered_bh_max'] = df_raw['(BH_Max)'].apply(parse_bh_max_entry)
    valid_bh = df_raw['recovered_bh_max'].notna() & df_raw['clean_formula'].notna()
    print(f"Valid recovered (BH_Max) with verified formulas: {valid_bh.sum()}")
    
    # 2. Recover Anisotropy Field H_A
    print("\nRecovering Anisotropy Field (H_A) entries...")
    df_raw['recovered_ha_tesla'] = df_raw['Anisotropy_Field'].apply(parse_ha_entry)
    valid_ha = df_raw['recovered_ha_tesla'].notna() & df_raw['clean_formula'].notna()
    print(f"Valid recovered H_A with verified formulas: {valid_ha.sum()}")
    
    # 3. Parse Saturation Magnetization Ms to Tesla: mu0 * Ms
    from src.preprocessing import preprocess_saturation_magnetization
    print("Standardizing Saturation Magnetization for relational transduction...")
    df_raw['recovered_ms_emu_g'] = df_raw.apply(
        lambda r: preprocess_saturation_magnetization(r['Saturation_Magnetization'], 7.5), axis=1
    )
    
    # Relational transduction: K1 = 0.5 * mu0 * H_A * Ms
    # In SI: K1 (J/m^3) = 0.5 * B_A (Tesla) * M_s (A/m)
    # where M_s (A/m) = M_s (emu/g) * density (g/cm^3) * 1000
    # Density default = 7.5 g/cm^3
    ms_a_m = df_raw['recovered_ms_emu_g'] * 7.5 * 1000.0
    df_raw['transduced_k1_j_m3'] = 0.5 * df_raw['recovered_ha_tesla'] * ms_a_m
    valid_trans_k1 = df_raw['transduced_k1_j_m3'].notna() & (df_raw['transduced_k1_j_m3'] > 10.0) & df_raw['clean_formula'].notna()
    print(f"Relational Transduction: Successfully deduced {valid_trans_k1.sum()} K1 values from (H_A, Ms) pairs!")
    
    # Load features_master.csv to update target columns
    print(f"\nLoading master features dataset from {master_path}...")
    df_master = pd.read_csv(master_path, low_memory=False)
    print(f"Master features shape before recovery: {df_master.shape}")
    
    # Create lookup dictionaries for the recovered targets by reduced_formula
    bh_lookup = df_raw[valid_bh].groupby('clean_formula')['recovered_bh_max'].median().to_dict()
    trans_k1_lookup = df_raw[valid_trans_k1].groupby('clean_formula')['transduced_k1_j_m3'].median().to_dict()
    
    # Update clean_BH_max_kJ_m3: fill missing using recovered values
    prev_bh_count = df_master['clean_BH_max_kJ_m3'].dropna().shape[0] if 'clean_BH_max_kJ_m3' in df_master.columns else 0
    if 'clean_BH_max_kJ_m3' not in df_master.columns:
        df_master['clean_BH_max_kJ_m3'] = np.nan
        
    for f, v in bh_lookup.items():
        mask = (df_master['reduced_formula'] == f)
        if mask.any():
            # Update where clean_BH_max is missing or fill
            df_master.loc[mask & df_master['clean_BH_max_kJ_m3'].isna(), 'clean_BH_max_kJ_m3'] = v
            
    new_bh_count = df_master['clean_BH_max_kJ_m3'].dropna().shape[0]
    print(f"(BH_Max) training samples in master dataset: {prev_bh_count} -> {new_bh_count} (+{new_bh_count - prev_bh_count} recovered samples!)")
    
    # Update clean_K1_J_m3: fill missing using transduced K1
    prev_k1_count = df_master['clean_K1_J_m3'].dropna().shape[0] if 'clean_K1_J_m3' in df_master.columns else 0
    if 'clean_K1_J_m3' not in df_master.columns:
        df_master['clean_K1_J_m3'] = np.nan
        
    for f, v in trans_k1_lookup.items():
        mask = (df_master['reduced_formula'] == f)
        if mask.any():
            df_master.loc[mask & df_master['clean_K1_J_m3'].isna(), 'clean_K1_J_m3'] = v
            
    new_k1_count = df_master['clean_K1_J_m3'].dropna().shape[0]
    print(f"K1 Anisotropy training samples in master dataset: {prev_k1_count} -> {new_k1_count} (+{new_k1_count - prev_k1_count} recovered samples!)")
    
    # Save updated master features
    print(f"Saving updated master features dataset to {master_path}...")
    df_master.to_csv(master_path, index=False)
    print("Master dataset successfully updated!")
    
    return {
        'bh_count_before': prev_bh_count,
        'bh_count_after': new_bh_count,
        'k1_count_before': prev_k1_count,
        'k1_count_after': new_k1_count
    }

if __name__ == '__main__':
    run_recovery()
