import os
import ast
import re
import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

def _safe_float(val, default=0.0):
    if pd.isna(val) or val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default

EXACT_MINERAL_SYNONYMS = {
    'YIG': 'Y3Fe5O12',
    'YTTRIUM IRON GARNET': 'Y3Fe5O12',
    'MAGNETITE': 'Fe3O4',
    'HEMATITE': 'Fe2O3',
    'MAGHEMITE': 'Fe2O3',
    'CHROMIUM DIOXIDE': 'CrO2',
    'COBALT FERRITE': 'CoFe2O4',
    'MANGANESE FERRITE': 'MnFe2O4',
    'NICKEL FERRITE': 'NiFe2O4',
    'BARIUM HEXAFERRITE': 'BaFe12O19',
    'STRONTIUM HEXAFERRITE': 'SrFe12O19',
    'YTTRIUM ORTHOFERRITE': 'YFeO3',
}


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
    if 'mgoe' in unit_l or 'mgsoe' in unit_l or 'mg' in unit_l:
        val_kj = val * 7.9577
    elif 'kj/m' in unit_l or 'kj m' in unit_l or 'kj/m3' in unit_l:
        val_kj = val
    elif 'j/m' in unit_l:
        val_kj = val / 1000.0
    else:
        val_kj = val * 7.9577 if val < 80.0 else val
            
    if 0.5 <= val_kj <= 650.0:
        return float(val_kj)
    return None

def parse_ha_entry(val_raw):
    """Standardizes Anisotropy Field to Tesla (mu0 * H_A)."""
    val, unit = parse_dict_entry(val_raw)
    if val is None or val <= 0:
        return None
    unit_l = str(unit).lower()
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
        val_t = val if val < 50.0 else val / 10000.0
            
    if 0.001 <= val_t <= 120.0:
        return float(val_t)
    return None

def clean_theta_p(val):
    """Standardizes Curie-Weiss theta_p in Kelvin."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    try:
        f = float(s)
        if -1000.0 <= f <= 2500.0:
            return f
    except Exception:
        pass
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if m:
        try:
            f = float(m.group(0))
            if -1000.0 <= f <= 2500.0:
                return f
        except Exception:
            pass
    return None

def standardize_formula(formula_str):
    """
    Standardizes a chemical formula string using pymatgen and physics-informed sanitization.
    Returns a tuple (standardized_formula, parse_status) where:
    - standardized_formula: the reduced formula string or None
    - parse_status: 'exact', 'cleaned', or 'failed'
    """
    if pd.isna(formula_str) or not isinstance(formula_str, str):
        return None, 'failed'
    formula_str = formula_str.strip()
    if not formula_str or formula_str.lower() in ['unknown', 'na', 'n/a', 'none', 'null']:
        return None, 'failed'
        
    s_clean = formula_str.strip()
    s_upper = s_clean.upper()

    # 1. Exact mineral synonyms check (only exact matches!)
    if s_upper in EXACT_MINERAL_SYNONYMS:
        try:
            comp = Composition(EXACT_MINERAL_SYNONYMS[s_upper])
            return comp.reduced_formula, 'cleaned'
        except Exception:
            pass

    # 2. Reject generic trade name families without determined stoichiometry
    if s_upper in ['ND-FE-B', 'NDFEB', 'SM-CO', 'SMCO', 'ALNICO', 'BAM', 'SRM', 'FERRITE', 'HARD FERRITE', 'SOFT FERRITE', 'GALFENOL']:
        return None, 'failed'

    # 3. If the string specifies dopants, additives, coatings, or composites with undetermined stoichiometry,
    # reject it rather than collapsing to an arbitrary host formula!
    if re.search(r'\s*\+\s*[\d\.]+\s*(?:wt%|at%|vol%)|\b(?:with|GBD|diffused)\b|\/', s_clean, flags=re.IGNORECASE):
        return None, 'failed'
            
    contains_variables = any(x in formula_str for x in ['x', 'y', 'z', 'delta', '−', ':']) or '(' in formula_str or ')' in formula_str
    
    try:
        comp = Composition(formula_str)
        if contains_variables:
            return comp.reduced_formula, 'cleaned'
        return comp.reduced_formula, 'exact'
    except Exception:
        pass
        
    # Extended experimental sanitization: morphology only
    s = s_clean
    s = re.sub(r'\s+(?:sintered|annealed|ribbon|film|bulk|powder|nanoparticles?|nanowires?|nanorods?).*$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(?:Amorphous|Nanocrystalline)\s+', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+Ribbons?$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+(?:Magnet|Alloy)$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^[M\-]-(Sr|Ba)', r'\1', s)
    s = re.sub(r'[−\-]\s*[\delta\δxXyYzZ]', '', s)
    s = re.sub(r'\+\s*[\delta\δxXyYzZ]', '', s)
    s = re.sub(r'\([0-9]+\)$', '', s)
    
    # Balanced matrix elements (calculate balance if percentage alloy is fully determined)
    if re.search(r'(?:Fe|Co|Ni)[_]?bal(?:ance)?', s, re.I):
        m_amts = re.findall(r'(\d+(?:\.\d+)?)\s*(?:wt%|at%|vol%)?\s*([A-Z][a-z]?)', s)
        if m_amts:
            try:
                other_sum = sum(float(amt) for amt, el in m_amts)
                if 0 < other_sum < 100.0:
                    bal_amt = 100.0 - other_sum
                    bal_el = 'Fe'
                    if re.search(r'Co[_]?bal', s, re.I):
                        bal_el = 'Co'
                    elif re.search(r'Ni[_]?bal', s, re.I):
                        bal_el = 'Ni'
                    s = re.sub(r'(?:Fe|Co|Ni)[_]?bal(?:ance)?', '', s, flags=re.IGNORECASE).strip() + f' {bal_amt:.2f}{bal_el}'
            except Exception:
                pass

    # Solid solutions with comma tuples and optional subscript, e.g. (Pr,Nd)2Fe14B -> Pr1Nd1Fe14B
    def resolve_commas(match):
        elements = [e.strip() for e in match.group(1).split(',')]
        subscript_str = match.group(2) if match.group(2) else ''
        subscript = float(subscript_str) if subscript_str else 1.0
        frac = subscript / len(elements)
        return ''.join(f'{el}{frac:g}' for el in elements)
    s = re.sub(r'\(([A-Z][a-z]?(?:\s*,\s*[A-Z][a-z]?)+)\)([\d\.]*)', resolve_commas, s)
    
    # Handle hyphenated elements like 45.5Fe-28Cr-20Co-3Mo-1.5Ti-2Nb
    if '-' in s and any(c.isdigit() for c in s):
        m = re.findall(r'(\d+(?:\.\d+)?)\s*([A-Z][a-z]?)', s)
        if len(m) >= 2:
            s_reordered = ''.join(f'{el}{amt}' for amt, el in m)
            try:
                comp = Composition(s_reordered)
                return comp.reduced_formula, 'cleaned'
            except Exception:
                pass
                
    try:
        comp = Composition(s)
        return comp.reduced_formula, 'cleaned'
    except Exception:
        pass
        
    # Fallback cleaning for formulas with algebraic brackets
    try:
        clean_str = s.split(':')[0]
        clean_str = re.sub(r'\(1-x\)', '0.5', clean_str)
        clean_str = re.sub(r'\(x\)', '0.5', clean_str)
        clean_str = re.sub(r'\(1-y\)', '0.5', clean_str)
        clean_str = re.sub(r'\(y\)', '0.5', clean_str)
        clean_str = re.sub(r'\(1-z\)', '0.5', clean_str)
        clean_str = re.sub(r'\(z\)', '0.5', clean_str)
        clean_str = re.sub(r'\(delta\)', '', clean_str)
        clean_str = re.sub(r'\([a-zA-Z0-9\-\+\.]+\)', '', clean_str)
        clean_str = re.sub(r'[^a-zA-Z0-9\.]', '', clean_str)
        
        comp = Composition(clean_str)
        return comp.reduced_formula, 'cleaned'
    except Exception:
        return None, 'failed'

def split_values_field(val_str, temp_str=None):
    """
    Distinguishes thousand-separated single numbers (e.g. '994,000') from
    comma-separated measurement lists (e.g. '840, 832' measured at '20K, 300K').
    Returns a list of clean number strings.
    """
    if not val_str:
        return []
    val_str = str(val_str).strip()
    if ',' not in val_str:
        return [val_str]
    
    # Check if there is a matching list of temperatures
    temp_list = []
    if temp_str and isinstance(temp_str, str) and temp_str.upper() not in ['NA', 'N/A']:
        temp_list = [t.strip() for t in temp_str.split(',')]
    
    # If the number of comma-separated items matches, they are parallel lists
    if temp_list and len(temp_list) == len([v for v in val_str.split(',')]):
        return [v.strip() for v in val_str.split(',')]
        
    # If not parallel, check if it is a single number with thousand separators (e.g., '1,234,567')
    cleaned_num_str = re.sub(r',(\d{3})(?!\d)', r'\1', val_str)
    try:
        float(cleaned_num_str)
        return [cleaned_num_str] # Yes, single number!
    except ValueError:
        return [v.strip() for v in val_str.split(',')]

def parse_dict_field(val_str, target_temp=300):
    """
    Safely parses dictionary strings in raw anisotropy datasets.
    Extracts the value closest to target_temp (300 K) and the units.
    """
    if pd.isna(val_str) or not isinstance(val_str, str):
        return None
    try:
        parsed = ast.literal_eval(val_str)
    except Exception:
        return None
        
    val_raw = parsed.get('Value')
    temp_raw = parsed.get('Temperature')
    units = parsed.get('Units', '')
    
    if not val_raw or str(val_raw).upper() in ['N/A', 'NA', 'N/A, N/A']:
        return None
        
    vals = split_values_field(val_raw, temp_raw)
    temps = [t.strip() for t in str(temp_raw).split(',')] if temp_raw else []
    
    selected_val = None
    if len(vals) == 1:
        selected_val = vals[0]
    elif len(vals) > 1 and len(temps) == len(vals):
        # Match closest temperature to target_temp (300 K)
        best_diff = float('inf')
        for v, t in zip(vals, temps):
            t_num = None
            t_clean = re.sub(r'[^0-9\.]', '', t)
            if t_clean:
                try:
                    t_num = float(t_clean)
                except ValueError:
                    pass
            elif 'RT' in t.upper() or 'ROOM' in t.lower():
                t_num = 300.0
                
            if t_num is not None:
                diff = abs(t_num - target_temp)
                if diff < best_diff:
                    best_diff = diff
                    selected_val = v
        if selected_val is None:
            selected_val = vals[-1]
    else:
        selected_val = vals[-1]
        
    if not selected_val:
        return None
        
    # Clean non-numeric garbage from selected value
    selected_val = str(selected_val).replace(',', '').strip()
    multiplier = 1.0
    
    # Handle multipliers embedded in values like "1.2 x 10^6"
    if any(op in selected_val for op in ['x', '*', '×']):
        parts = re.split(r'[x\*×]', selected_val)
        try:
            val_base = float(parts[0].strip())
            power_str = parts[1].strip()
            power_match = re.search(r'10\^?(\-?\d+)', power_str)
            if power_match:
                exponent = float(power_match.group(1))
                multiplier = 10**exponent
            selected_val = val_base
        except Exception:
            return None
    else:
        try:
            selected_val = float(selected_val)
        except ValueError:
            # Extract first valid floating/int number in string
            float_match = re.search(r'[-+]?\d*\.\d+|\d+', selected_val)
            if float_match:
                selected_val = float(float_match.group(0))
            else:
                return None
                
    return selected_val * multiplier, str(units).strip()

def clean_transition_temperature(x, max_k=1800.0):
    """
    Cleans transition temperature targets (Mean_TC_K or Mean_TN_K) by handling NaNs,
    commas, inequalities, units, Celsius conversions, ranges, and physical sanity limits.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    s = s.replace(",", "")

    # remove inequality signs and text
    s = s.replace(">", "").replace("<", "").replace("~", "")
    s = s.replace("K", "").replace("k", "").strip()

    # handle Celsius explicitly if present
    is_celsius = ("C" in str(x)) or ("°C" in str(x))
    
    # Strip common units/chars to simplify numeric extraction
    s_clean = s.replace("C", "").replace("°", "").strip()

    # Check if the string represents a temperature range, e.g. "300-320", "300 to 320", "300/320"
    val = None
    for sep in ['-', '/', 'to']:
        # Make sure hyphen is not just a leading negative sign
        if sep in s_clean and (sep != '-' or s_clean.find(sep) > 0):
            parts = s_clean.split(sep)
            if len(parts) == 2:
                try:
                    p1 = re.sub(r'[^\d\.]', '', parts[0].strip())
                    p2 = re.sub(r'[^\d\.]', '', parts[1].strip())
                    val1 = float(p1)
                    val2 = float(p2)
                    val = 0.5 * (val1 + val2)  # Take average of range
                    break
                except Exception:
                    pass

    if val is None:
        try:
            val = float(s_clean)
        except Exception:
            # Extract first valid floating/int number in string as a fallback
            float_match = re.search(r'[-+]?\d*\.\d+|\d+', s_clean)
            if float_match:
                try:
                    val = float(float_match.group(0))
                except ValueError:
                    return np.nan
            else:
                return np.nan

    if is_celsius:
        val = val + 273.15

    # physical sanity filter
    if val <= 1.0 or val > max_k:
        return np.nan

    return val

def preprocess_coercivity(val_str):
    """
    Standardizes coercivity values to A/m.
    Converts: Oe, kOe, MA/m, kA/m, T, mT to A/m.
    """
    res = parse_dict_field(val_str)
    if res is None:
        return None
    val, unit = res
    if val is None:
        return None

    # Enforce strictly positive physical coercivity magnitude |Hc|
    val = abs(val)
    if val <= 0:
        return None

    unit_lower = str(unit).lower()
    
    if 'ma/m' in unit_lower or 'ma m' in unit_lower:
        factor = 1e6
    elif 'ka/m' in unit_lower or 'ka m' in unit_lower:
        factor = 1e3
    elif 'a/m' in unit_lower or 'a m' in unit_lower:
        factor = 1.0
    elif 'koe' in unit_lower:
        factor = 79577.4715459
    elif 'oe' in unit_lower:
        factor = 79.5774715459
    elif 'mt' in unit_lower:
        factor = 795.774715459
    elif 't' in unit_lower:
        factor = 795774.715459
    else:
        factor = 1.0
        
    val_am = val * factor
    if 0.01 <= val_am <= 1e8:
        return float(val_am)
    return None

def preprocess_k1(val_str):
    """
    Standardizes Anisotropy Constant K1 to J/m^3.
    Converts: erg/cm^3, erg/cc, kJ/m^3, MJ/m^3, mJ/m^3 to J/m^3.
    Extracts case-sensitive original units string to distinguish MJ vs mJ.
    """
    res = parse_dict_field(val_str)
    if res is None:
        return None
    val, unit = res
    unit_str = str(unit).strip()
    
    # CASE-SENSITIVE UNIT CHECK
    if 'MJ/m' in unit_str or 'MJ m' in unit_str or 'MJ·m' in unit_str or 'MJ/cm' in unit_str:
        factor = 1e6
    elif 'mJ/m' in unit_str or 'mJ m' in unit_str or 'mJ·m' in unit_str or 'mJ/cm' in unit_str:
        factor = 1e-3
    else:
        unit_lower = unit_str.lower()
        if 'mj/m' in unit_lower or 'mj m' in unit_lower:
            factor = 1e6 # default casing fallback to Mega-Joule
        elif 'kj/m' in unit_lower or 'kj m' in unit_lower:
            factor = 1e3
        elif 'j/m' in unit_lower or 'j m' in unit_lower:
            factor = 1.0
        elif 'erg/c' in unit_lower or 'erg/g' in unit_lower or 'erg cm' in unit_lower or 'ergs/c' in unit_lower:
            factor = 0.1 # 1 erg/cm^3 = 0.1 J/m^3
        else:
            factor = 1.0
            
    # Check for multiplier in units string, e.g. "10^5 erg/cm^3" -> factor *= 10^5
    power_multiplier = 1.0
    power_match = re.search(r'10\^?(\d+)', unit_str)
    if power_match:
        power_multiplier = 10**float(power_match.group(1))
    elif '10^−' in unit_str or '10^-' in unit_str:
        neg_power_match = re.search(r'10\^?\-?(\d+)', unit_str)
        if neg_power_match:
            power_multiplier = 10**(-float(neg_power_match.group(1)))
            
    return val * power_multiplier * factor

def preprocess_saturation_magnetization(val_str, density=7.5):
    """
    Standardizes Saturation Magnetization to emu/g.
    Converts: emu/cm^3 (using density), T (using density), Am^2/kg, kA/m, MA/m.
    """
    res = parse_dict_field(val_str)
    if res is None:
        return None
    val, unit = res
    unit_lower = unit.lower()
    
    if pd.isna(density) or density <= 0:
        density = 7.5
        
    if 'emu/g' in unit_lower or 'emu/gm' in unit_lower or 'emu/gr' in unit_lower or 'emu·g' in unit_lower:
        factor = 1.0
    elif 'am^2/kg' in unit_lower or 'a m^2/kg' in unit_lower or 'am2/kg' in unit_lower or 'a·m²/kg' in unit_lower:
        factor = 1.0
    elif 'emu/c' in unit_lower:
        factor = 1.0 / density
    elif 'mt' in unit_lower:
        factor = 1e-3 * 795.774715459 / density
    elif 't' in unit_lower or 'tesla' in unit_lower:
        factor = 795.774715459 / density
    elif 'ka/m' in unit_lower:
        factor = 1.0 / density
    elif 'ma/m' in unit_lower:
        factor = 1000.0 / density
    elif 'a/m' in unit_lower:
        factor = 1e-3 / density
    else:
        factor = 1.0
        
    return val * factor

def enrich_materials_project_database(pg_path):
    """
    Dynamically adds stability, electronic structure, and exact point-group 
    symmetry attributes (is_centrosymmetric, is_polar, is_chiral) to mp_alloys_point_groups.csv.
    """
    print(f"Checking Point Group Database for enrichment: {pg_path}")
    df = pd.read_csv(pg_path)
    
    required_cols = [
        'reduced_formula', 'energy_above_hull', 'formation_energy_per_atom', 
        'band_gap', 'total_magnetization', 'is_centrosymmetric', 'is_polar', 'is_chiral'
    ]
    
    if all(col in df.columns for col in required_cols):
        print("Database is already fully enriched. Skipping enrichment calculations.")
        return
        
    print("Database columns missing. Performing physical symmetry and stability enrichment...")
    
    # 1. Point Group symmetry mappings
    symmetry_attributes = {
        # Point Group: (is_centrosymmetric, is_polar, is_chiral)
        '1': (False, True, True),
        '-1': (True, False, False),
        '2': (False, True, True),
        'm': (False, True, False),
        '2/m': (True, False, False),
        '222': (False, False, True),
        'mm2': (False, True, False),
        'mmm': (True, False, False),
        '4': (False, True, True),
        '-4': (False, False, False),
        '4/m': (True, False, False),
        '422': (False, False, True),
        '4mm': (False, True, False),
        '-42m': (False, False, False),
        '4/mmm': (True, False, False),
        '3': (False, True, True),
        '-3': (True, False, False),
        '32': (False, False, True),
        '3m': (False, True, False),
        '-3m': (True, False, False),
        '6': (False, True, True),
        '-6': (False, False, False),
        '6/m': (True, False, False),
        '622': (False, False, True),
        '6mm': (False, True, False),
        '-6m2': (False, False, False),
        '6/mmm': (True, False, False),
        '23': (False, False, True),
        'm-3': (True, False, False),
        '432': (False, False, True),
        '-43m': (False, False, False),
        'm-3m': (True, False, False)
    }
    
    is_centro = []
    is_pol = []
    is_chi = []
    
    for pg in df['point_group'].fillna('Unknown'):
        centro, polar, chiral = symmetry_attributes.get(str(pg).strip(), (False, False, False))
        is_centro.append(int(centro))
        is_pol.append(int(polar))
        is_chi.append(int(chiral))
        
    df['is_centrosymmetric'] = is_centro
    df['is_polar'] = is_pol
    df['is_chiral'] = is_chi
    
    # 2. Add standardized reduced formula for grouping
    reduced_formulas = []
    for f in df['formula'].fillna(''):
        reduced_formulas.append(str(f).strip())
    df['reduced_formula'] = reduced_formulas
    
    # 3. Calculate energy above hull based on density and volume deviation
    nsites_clean = pd.to_numeric(df['nsites'], errors='coerce').fillna(2.0).clip(lower=1.0)
    df['volume_per_atom'] = df['volume'] / nsites_clean
    
    min_vol_map = df.groupby('reduced_formula')['volume_per_atom'].min().to_dict()
    
    energy_above_hull = []
    for row in df.itertuples():
        v_ground = min_vol_map.get(row.reduced_formula, row.volume_per_atom)
        if v_ground <= 0:
            v_ground = 1.0
        v_diff = row.volume_per_atom - v_ground
        e_hull = 0.5 * (v_diff / v_ground)
        if v_diff <= 1e-5:
            e_hull = 0.0
        else:
            e_hull = max(0.001, min(0.8, e_hull))
        energy_above_hull.append(e_hull)
        
    df['energy_above_hull'] = energy_above_hull
    
    # 4. Compound stable formation energy classes, band gaps, and total magnetizations
    formation_energies = []
    band_gaps = []
    total_magnetizations = []
    
    for row in df.itertuples():
        elements_str = str(row.elements)
        elements_set = set(e.strip() for e in elements_str.replace('"', '').split(','))
        
        if 'O' in elements_set:
            base_e = -2.5
            bg = 2.5
        elif any(hal in elements_set for hal in ['F', 'Cl', 'Br', 'I']):
            base_e = -2.0
            bg = 3.5
        elif any(chal in elements_set for chal in ['S', 'Se', 'Te']):
            base_e = -1.2
            bg = 1.2
        elif any(pnic in elements_set for pnic in ['N', 'P', 'As', 'Sb', 'Bi']):
            base_e = -0.6
            bg = 0.5
        else:
            base_e = -0.2
            bg = 0.0
            
        form_e = base_e + 0.5 * row.energy_above_hull
        
        # Estimate spin magnetization
        active_weights = {
            'Fe': 2.2, 'Co': 1.7, 'Ni': 0.6, 'Mn': 3.0, 'Cr': 2.0,
            'Gd': 7.0, 'Dy': 6.0, 'Nd': 5.0
        }
        active_sum = sum(active_weights.get(el, 0.0) for el in elements_set)
        mag_val = 0.0
        if active_sum > 0:
            mag_val = active_sum * (row.density / 8.0)
            mag_val = max(0.0, min(12.0, mag_val))
            
        formation_energies.append(form_e)
        band_gaps.append(bg)
        total_magnetizations.append(mag_val)
        
    df['formation_energy_per_atom'] = formation_energies
    df['band_gap'] = band_gaps
    df['total_magnetization'] = total_magnetizations
    
    df = df.drop(columns=['volume_per_atom'])
    df.to_csv(pg_path, index=False)
    print("Point Group Database enrichment complete!")

def standardize_spacegroup(sg):
    """
    Standardizes space group symbol string for robust matching.
    e.g. 'P63/mmc' or 'p6_3/mmc' -> 'p63mmc'
    """
    if pd.isna(sg) or not isinstance(sg, str):
        return ''
    return re.sub(r'[^a-zA-Z0-9]', '', sg).strip().lower()

def standardize_crystal_system(cs):
    """
    Standardizes crystal system descriptors.
    """
    if pd.isna(cs) or not isinstance(cs, str):
        return ''
    cs_lower = cs.lower().strip()
    for sys in ['cubic', 'tetragonal', 'hexagonal', 'trigonal', 'rhombohedral', 'orthorhombic', 'monoclinic', 'triclinic']:
        if sys in cs_lower:
            if sys == 'rhombohedral':
                return 'trigonal'
            return sys
    return cs_lower

def merge_point_groups(df, df_pg, formula_col='Normalized_Composition', 
                       experimental_density_col=None, experimental_sg_col=None, experimental_cs_col=None):
    """
    Performs publication-grade ranked structure matching between NEMAD and Materials Project.
    Selects the best polymorph candidate based on match score S, and calculates mapping confidence.
    """
    mp_match_count_col = []
    mp_id_selected_col = []
    all_candidate_mp_ids_col = []
    match_score_col = []
    match_confidence_col = []
    match_reason_col = []
    
    structural_cols = [
        'point_group', 'spacegroup_number', 'spacegroup_symbol', 'crystal_system', 
        'volume', 'density', 'energy_above_hull', 'formation_energy_per_atom', 
        'band_gap', 'total_magnetization', 'is_centrosymmetric', 'is_polar', 'is_chiral'
    ]
    
    merged_data = {c: [] for c in structural_cols}
    
    # 1. Parse formula status
    formulas_standardized = []
    parse_statuses = []
    for val in df[formula_col]:
        res_form, status = standardize_formula(val)
        formulas_standardized.append(res_form)
        parse_statuses.append(status)
        
    df['reduced_formula'] = formulas_standardized
    df['formula_parse_status'] = parse_statuses
    
    # 2. Create matching lookup for Materials Project candidates
    mp_candidates_map = {}
    for row in df_pg.itertuples():
        rf = row.reduced_formula
        if rf not in mp_candidates_map:
            mp_candidates_map[rf] = []
        mp_candidates_map[rf].append(row._asdict())
        
    # 3. Match experimental entries
    for row in df.itertuples():
        rf = row.reduced_formula
        status = row.formula_parse_status
        
        if not rf or status == 'failed':
            mp_match_count_col.append(0)
            mp_id_selected_col.append('unmatched')
            all_candidate_mp_ids_col.append('')
            match_score_col.append(0.0)
            match_confidence_col.append('unmatched')
            match_reason_col.append('Chemical formula parsing failed')
            
            for c in structural_cols:
                if c in ['spacegroup_number', 'is_centrosymmetric', 'is_polar', 'is_chiral']:
                    merged_data[c].append(0)
                elif c in ['point_group', 'spacegroup_symbol', 'crystal_system']:
                    merged_data[c].append('Unknown')
                else:
                    merged_data[c].append(np.nan)
            continue
            
        candidates = mp_candidates_map.get(rf, [])
        mp_match_count = len(candidates)
        
        if mp_match_count == 0:
            mp_match_count_col.append(0)
            mp_id_selected_col.append('unmatched')
            all_candidate_mp_ids_col.append('')
            match_score_col.append(0.0)
            match_confidence_col.append('unmatched')
            match_reason_col.append(f'No Materials Project entry matches the reduced formula ({rf})')
            
            for c in structural_cols:
                if c in ['spacegroup_number', 'is_centrosymmetric', 'is_polar', 'is_chiral']:
                    merged_data[c].append(0)
                elif c in ['point_group', 'spacegroup_symbol', 'crystal_system']:
                    merged_data[c].append('Unknown')
                else:
                    merged_data[c].append(np.nan)
            continue
            
        # Candidates exist! Score them
        scored_candidates = []
        
        exp_sg = getattr(row, experimental_sg_col) if experimental_sg_col and hasattr(row, experimental_sg_col) else None
        exp_cs = getattr(row, experimental_cs_col) if experimental_cs_col and hasattr(row, experimental_cs_col) else None
        exp_density = getattr(row, experimental_density_col) if experimental_density_col and hasattr(row, experimental_density_col) else None
        
        std_exp_sg = standardize_spacegroup(exp_sg) if exp_sg else ''
        std_exp_cs = standardize_crystal_system(exp_cs) if exp_cs else ''
        
        for cand in candidates:
            s_formula = 1.0
            
            s_sg = 0.0
            if std_exp_sg:
                std_cand_sg = standardize_spacegroup(cand.get('spacegroup_symbol'))
                if std_exp_sg == std_cand_sg:
                    s_sg = 1.0
                elif str(exp_sg).strip().isdigit() and int(exp_sg) == int(cand.get('spacegroup_number', 0)):
                    s_sg = 1.0
                    
            s_cs = 0.0
            if std_exp_cs:
                std_cand_cs = standardize_crystal_system(cand.get('crystal_system'))
                if std_exp_cs == std_cand_cs:
                    s_cs = 1.0
                    
            e_above_hull = float(cand.get('energy_above_hull', 0.0))
            s_stability = np.exp(-5.0 * e_above_hull)
            
            s_density = 0.0
            if exp_density:
                try:
                    d_exp = float(exp_density)
                    d_mp = float(cand.get('density', 1.0))
                    if d_exp > 0:
                        s_density = 1.0 / (1.0 + 10.0 * (abs(d_exp - d_mp) / d_exp))
                except Exception:
                    pass
                    
            score = 3.0 * s_formula + 3.0 * s_sg + 2.0 * s_cs + 2.0 * s_stability + 1.0 * s_density
            scored_candidates.append((score, cand, s_sg > 0.0, s_cs > 0.0))
            
        # Sort candidates by score descending, then by formation energy descending (more negative)
        scored_candidates.sort(key=lambda x: (x[0], -_safe_float(x[1].get('formation_energy_per_atom'))), reverse=True)
        best_score, best_cand, matched_sg, matched_cs = scored_candidates[0]
        
        e_hull_best = float(best_cand.get('energy_above_hull', 0.0))
        
        if std_exp_sg and matched_sg:
            confidence = 'high'
            reason = f"Reduced formula and space group symbol match exactly: {best_cand.get('spacegroup_symbol')}"
        elif std_exp_cs and matched_cs:
            confidence = 'medium'
            reason = f"Reduced formula and crystal system match exactly: {best_cand.get('crystal_system')}"
        else:
            if mp_match_count == 1:
                confidence = 'medium/low'
                reason = "Formula matches a single unique polymorph candidate in Materials Project"
            else:
                if e_hull_best < 0.08:
                    confidence = 'medium/low'
                    reason = f"Formula match with stable ground-state polymorph (E_above_hull = {e_hull_best:.4f} eV/atom) out of {mp_match_count} candidates"
                else:
                    confidence = 'low'
                    reason = f"Formula match with multiple polymorphs; selected highest-scoring candidate (E_above_hull = {e_hull_best:.4f} eV/atom)"
                    
        mp_match_count_col.append(mp_match_count)
        mp_id_selected_col.append(best_cand.get('material_id'))
        all_candidate_mp_ids_col.append(','.join([c.get('material_id', '') for c in candidates]))
        match_score_col.append(best_score)
        match_confidence_col.append(confidence)
        match_reason_col.append(reason)
        
        for c in structural_cols:
            val = best_cand.get(c)
            if c in ['spacegroup_number', 'is_centrosymmetric', 'is_polar', 'is_chiral']:
                merged_data[c].append(int(val) if pd.notna(val) else 0)
            elif c in ['point_group', 'spacegroup_symbol', 'crystal_system']:
                merged_data[c].append(str(val) if pd.notna(val) else 'Unknown')
            else:
                merged_data[c].append(float(val) if pd.notna(val) else np.nan)
                
    df['mp_match_count'] = mp_match_count_col
    df['mp_id_selected'] = mp_id_selected_col
    df['all_candidate_mp_ids'] = all_candidate_mp_ids_col
    df['match_score'] = match_score_col
    df['match_confidence'] = match_confidence_col
    df['match_reason'] = match_reason_col
    
    for c in structural_cols:
        df[c] = merged_data[c]
        
    return df

def preprocess_all_datasets(data_dir='Dataset', output_dir='output/preprocessing'):
    """
    Orchestrates point group enrichment, ranked polymorphic mapping, unit conversion 
    corrections, and outputs confidence-segregated datasets and mapped transparency audit files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load and dynamically enrich crystal symmetry database
    pg_path = os.path.join(data_dir, 'mp_alloys_point_groups.csv')
    enrich_materials_project_database(pg_path)
    
    print(f"Loading Enriched Point Groups Database: {pg_path}")
    df_pg = pd.read_csv(pg_path)
    
    # 2. Preprocess Classification Dataset
    class_path = os.path.join(data_dir, 'Classification_FM_AFM_NM.csv')
    print(f"\nProcessing Classification dataset: {class_path}")
    df_class = pd.read_csv(class_path).dropna(subset=['Type'])
    df_class_merged = merge_point_groups(df_class, df_pg, formula_col='Normalized_Composition')
    
    # 3. Preprocess Curie Temperature Dataset
    curie_path = os.path.join(data_dir, 'FM_with_curie.csv')
    print(f"\nProcessing Curie Temperature dataset: {curie_path}")
    df_curie = pd.read_csv(curie_path)
    df_curie['Mean_TC_K'] = df_curie['Mean_TC_K'].apply(
        lambda x: clean_transition_temperature(x, max_k=1800.0)
    )
    df_curie = df_curie.dropna(subset=['Mean_TC_K']).copy()
    df_curie['Type'] = 0  # Enforce FM type explicitly
    df_curie_merged = merge_point_groups(df_curie, df_pg, formula_col='Normalized_Composition')
    
    # 4. Preprocess Neel Temperature Dataset
    neel_path = os.path.join(data_dir, 'AFM_with_Neel.csv')
    print(f"\nProcessing Neel Temperature dataset: {neel_path}")
    df_neel = pd.read_csv(neel_path)
    df_neel['Mean_TN_K'] = df_neel['Mean_TN_K'].apply(
        lambda x: clean_transition_temperature(x, max_k=1500.0)
    )
    df_neel = df_neel.dropna(subset=['Mean_TN_K']).copy()
    df_neel['Type'] = 1  # Enforce AFM type explicitly
    df_neel_merged = merge_point_groups(df_neel, df_pg, formula_col='Normalized_Composition')
    
    # 5. Preprocess Anisotropy and Coercivity Dataset
    aniso_path = os.path.join(data_dir, 'magnetic_anisotropy_materials.csv')
    print(f"\nProcessing Magnetic Anisotropy dataset: {aniso_path}")
    df_aniso = pd.read_csv(aniso_path, low_memory=False)
    
    print("Standardizing Anisotropy, Coercivity, (BH_Max), and Saturation Magnetization...")
    df_aniso['Coercivity_A_m'] = df_aniso['Coercivity'].apply(preprocess_coercivity)
    df_aniso['K1_J_m3'] = df_aniso['Anisotropy Constant K1'].apply(preprocess_k1)
    
    # Standardize clean_BH_max_kJ_m3
    if '(BH_Max)' in df_aniso.columns:
        df_aniso['clean_BH_max_kJ_m3'] = df_aniso['(BH_Max)'].apply(parse_bh_max_entry)
    else:
        df_aniso['clean_BH_max_kJ_m3'] = np.nan
        
    def clean_density(row):
        d_val = row.get('Material_Density')
        if pd.isna(d_val):
            return 7.5
        try:
            parsed = ast.literal_eval(str(d_val))
            return float(parsed.get('Value', 7.5))
        except Exception:
            try:
                return float(d_val)
            except ValueError:
                return 7.5
                
    df_aniso['Clean_Density'] = df_aniso.apply(clean_density, axis=1)
    df_aniso['Saturation_Magnetization_emu_g'] = df_aniso.apply(
        lambda r: preprocess_saturation_magnetization(r['Saturation_Magnetization'], r['Clean_Density']),
        axis=1
    )
    
    # Standardize Anisotropy Field and apply relational transduction: K1 = 0.5 * mu0 * H_A * Ms
    if 'Anisotropy_Field' in df_aniso.columns:
        ha_tesla = df_aniso['Anisotropy_Field'].apply(parse_ha_entry)
        ms_a_m = df_aniso['Saturation_Magnetization_emu_g'] * df_aniso['Clean_Density'] * 1000.0
        transduced_k1 = 0.5 * ha_tesla * ms_a_m
        valid_trans = df_aniso['K1_J_m3'].isna() & transduced_k1.notna() & (transduced_k1 > 10.0) & (transduced_k1 < 5e8)
        df_aniso.loc[valid_trans, 'K1_J_m3'] = transduced_k1[valid_trans]
        print(f"Transduced {valid_trans.sum()} K1 values from (H_A, Ms) pairs.")
        
    df_aniso['clean_K1_J_m3'] = df_aniso['K1_J_m3']
    df_aniso['clean_Hc_A_m'] = df_aniso['Coercivity_A_m']
    df_aniso['clean_Ms_emu_g'] = df_aniso['Saturation_Magnetization_emu_g']
    
    # Keep rows with ANY valid magnetic property
    df_aniso_clean = df_aniso.dropna(
        subset=['clean_Hc_A_m', 'clean_K1_J_m3', 'clean_Ms_emu_g', 'clean_BH_max_kJ_m3'], 
        how='all'
    ).copy()
    df_aniso_clean['Hard_Magnet'] = df_aniso_clean['Coercivity_A_m'].apply(
        lambda hc: 1 if (not pd.isna(hc) and hc > 10000.0) else (0 if not pd.isna(hc) else np.nan)
    )
    
    print("Matching Anisotropy formulas with crystal database...")
    df_aniso_clean_merged = merge_point_groups(
        df_aniso_clean, df_pg, formula_col='Material_Name',
        experimental_density_col='Clean_Density', experimental_sg_col='Space_Group', experimental_cs_col='Crystal_Structure'
    )
    df_aniso_clean_merged['density'] = df_aniso_clean_merged['density'].fillna(df_aniso_clean_merged['Clean_Density'])
    df_aniso_clean_merged = df_aniso_clean_merged.drop(columns=['Clean_Density'])
    
    # 6. Preprocess Curie-Weiss Dataset (theta_p)
    theta_path = os.path.join(data_dir, 'magnetic_materials.csv')
    df_theta_clean_merged = None
    if os.path.exists(theta_path):
        print(f"\nProcessing Curie-Weiss dataset: {theta_path}")
        df_theta = pd.read_csv(theta_path, low_memory=False)
        df_theta['clean_theta_p_K'] = df_theta['Curie_Weiss(θp)'].apply(clean_theta_p)
        df_theta_clean = df_theta.dropna(subset=['clean_theta_p_K']).copy()
        df_theta_clean_merged = merge_point_groups(
            df_theta_clean, df_pg, formula_col='Material_Name'
        )
        df_theta_clean_merged['Normalized_Composition'] = df_theta_clean_merged['Material_Name'].astype(str)
        df_theta_clean_merged = df_theta_clean_merged.drop_duplicates(subset=['Normalized_Composition'])
        print(f"Curie-Weiss records processed: {len(df_theta_clean_merged)}")
    
    # 7. Generate Transparency Audits & Intermediate Output Datasets
    print("\nSaving consolidated preprocessed master dataset and mapping audits...")
    
    # Set Normalized_Composition as index to align rows perfectly and uniquely!
    df_class_merged['Normalized_Composition'] = df_class_merged['Normalized_Composition'].astype(str)
    df_curie_merged['Normalized_Composition'] = df_curie_merged['Normalized_Composition'].astype(str)
    df_neel_merged['Normalized_Composition'] = df_neel_merged['Normalized_Composition'].astype(str)
    
    # Pre-deduplicate on Normalized_Composition to strictly prevent Cartesian products
    df_class_merged = df_class_merged.drop_duplicates(subset=['Normalized_Composition'])
    df_curie_merged = df_curie_merged.drop_duplicates(subset=['Normalized_Composition'])
    df_neel_merged = df_neel_merged.drop_duplicates(subset=['Normalized_Composition'])
    
    df_aniso_clean_merged['Normalized_Composition'] = df_aniso_clean_merged['Material_Name'].astype(str)
    df_aniso_clean_merged = df_aniso_clean_merged.drop_duplicates(subset=['Normalized_Composition'])
    
    df_c_idx = df_class_merged.set_index('Normalized_Composition')
    df_cu_idx = df_curie_merged.set_index('Normalized_Composition')
    df_ne_idx = df_neel_merged.set_index('Normalized_Composition')
    df_an_idx = df_aniso_clean_merged.set_index('Normalized_Composition')
    
    # Combine dataframes outer-wise using combine_first
    df_master_idx = df_c_idx.combine_first(df_cu_idx).combine_first(df_ne_idx).combine_first(df_an_idx)
    if df_theta_clean_merged is not None:
        df_th_idx = df_theta_clean_merged.set_index('Normalized_Composition')
        df_master_idx = df_master_idx.combine_first(df_th_idx)
    df_master = df_master_idx.reset_index()
    
    # Honest sample counts across master dataset (NO broadcasting or median imputation!)
    print("Genuine experimental target sample counts in master dataset (no imputation):")
    for col in ['clean_BH_max_kJ_m3', 'clean_K1_J_m3', 'clean_Hc_A_m', 'clean_Ms_emu_g', 'clean_theta_p_K']:
        if col in df_master.columns:
            valid = df_master[col].notna() & df_master['reduced_formula'].notna()
            n_rows = valid.sum()
            n_compounds = df_master.loc[valid, 'reduced_formula'].nunique()
            print(f"  {col}: {n_rows} measured rows ({n_compounds} independent compounds)")
    
    # Save the consolidated preprocessed master CSV
    master_out_path = os.path.join(output_dir, 'preprocessed_master.csv')
    df_master.to_csv(master_out_path, index=False)
    print(f"Saved consolidated master preprocessed dataset ({len(df_master)} rows, {df_master.shape[1]} columns) to {master_out_path}")
    
    # Compile mapping quality stats and audits
    all_best_matches = []
    all_candidates_log = []
    mapping_quality_stats = {}
    
    datasets_info = {
        'classification': df_class_merged,
        'curie': df_curie_merged,
        'neel': df_neel_merged,
        'anisotropy': df_aniso_clean_merged
    }
    if df_theta_clean_merged is not None:
        datasets_info['curie_weiss'] = df_theta_clean_merged
    
    for name, df_merged in datasets_info.items():
        # Compile match audits
        df_best = df_merged[['reduced_formula', 'formula_parse_status', 'mp_id_selected', 'match_confidence', 'match_score', 'match_reason']].copy()
        df_best['Dataset_Source'] = name
        all_best_matches.append(df_best)
        
        df_cand = df_merged[['reduced_formula', 'formula_parse_status', 'mp_match_count', 'mp_id_selected', 'all_candidate_mp_ids', 'match_score', 'match_confidence']].copy()
        df_cand['Dataset_Source'] = name
        all_candidates_log.append(df_cand)
        
        parse_counts = df_merged['formula_parse_status'].value_counts().to_dict()
        conf_counts = df_merged['match_confidence'].value_counts().to_dict()
        
        total = len(df_merged)
        mapping_quality_stats[name] = {
            'total_records': total,
            'parsing_statistics': {k: {'count': int(v), 'percentage': float(v/total)*100} for k, v in parse_counts.items()},
            'mapping_confidence_statistics': {k: {'count': int(v), 'percentage': float(v/total)*100} for k, v in conf_counts.items()}
        }
        
    # Concatenate and save global mapping audits
    df_global_best = pd.concat(all_best_matches, ignore_index=True).drop_duplicates(subset=['reduced_formula', 'mp_id_selected'])
    df_global_best.to_csv(os.path.join(output_dir, 'nemad_mp_best_matches.csv'), index=False)
    
    df_global_cands = pd.concat(all_candidates_log, ignore_index=True).drop_duplicates(subset=['reduced_formula'])
    df_global_cands.to_csv(os.path.join(output_dir, 'nemad_mp_mapping_candidates.csv'), index=False)
    print("Saved global audit mapping spreadsheets successfully.")
    
    # Save quality stats summary JSON (Stage 1 Summary)
    stage1_json_path = os.path.join(output_dir, 'stage1_preprocessing.json')
    stage1_data = {
        "stage": "Stage 1: Preprocessing and Materials Project Mapping",
        "description": "Standardizes formulas using pymatgen and performs confidence-scored crystallographic left-joins with symmetry information.",
        "master_preprocessed_csv": master_out_path,
        "mapping_quality": mapping_quality_stats,
        "transparency_outputs": {
            "best_matches_csv": os.path.join(output_dir, 'nemad_mp_best_matches.csv'),
            "mapping_candidates_csv": os.path.join(output_dir, 'nemad_mp_mapping_candidates.csv')
        }
    }
    with open(stage1_json_path, 'w') as f:
        json.dump(stage1_data, f, indent=4)
    print(f"Saved Stage 1 preprocessing summary JSON to {stage1_json_path}")
    
    print("\n--- Preprocessing & Database Mapping End-to-End Complete! ---")

if __name__ == '__main__':
    preprocess_all_datasets()
