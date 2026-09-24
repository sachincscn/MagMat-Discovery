import os
import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

# List of 103 elements H to Lr in periodic table order
ELEMENT_LIST = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
]

# Shared chemical families and magnetic constants imported from central utils
from src.utils import (
    MAGNETIC_3D,
    RARE_EARTH,
    TRANSITION_METALS,
    CHALCOGENS,
    PNICTOGENS,
    HALOGENS,
    ELEMENTAL_FM,
    ELEMENTAL_AFM,
    STRONG_FM_3D,
    STRONG_AFM_3D,
    RARE_EARTH_FM,
    RARE_EARTH_AFM,
    HIGH_CURIE_ELEMENTS,
    ELEMENTAL_TC_K,
    ELEMENTAL_TN_K,
    SOC_CONSTANTS_MEV,
    SPIN_QUANTUM_NUMBERS,
    STONER_PARAM,
    RE_ANISOTROPY_SIGN,
    FRUSTRATED_SPACE_GROUPS,
    UNIAXIAL_SYSTEMS,
    COORDINATION_BY_SYSTEM,
    SHANNON_RADII_CN6,
    DE_GENNES_FACTORS,
    VALENCE_ELECTRON_COUNT
)



CENTROSYMMETRIC_PG = {
    "-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m",
    "6/m", "6/mmm", "m-3", "m-3m"
}

FORBIDDEN_MP_PHYSICAL = [
    "volume",
    "density",
    "volume_per_atom",
    "nsites",
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "total_magnetization"
]

METADATA_COLS = [
    "formula",
    "material_id",
    "spacegroup_symbol",
    "crystal_system",
    "point_group",
    "Normalized_Composition",
    "Example_Formula",
    "Clean_Chemical_Formula",
    "Rare_Earth_proportion"  # Replaced by rare_earth_fraction
]

COMPOSITION_CACHE = {}
OXIDATION_STATES_CACHE = {}
FORMULA_FAMILY_CACHE = {}
IONIC_EN_CACHE = {}

# ── D/F-electron counts and Hund's rule unpaired electrons for AFM/NM physics ──
# Ground-state d-electron counts for transition metals (neutral atom)
D_ELECTRON_COUNTS = {
    'Sc': 1, 'Ti': 2, 'V': 3, 'Cr': 5, 'Mn': 5, 'Fe': 6, 'Co': 7, 'Ni': 8, 'Cu': 10, 'Zn': 10,
    'Y': 1, 'Zr': 2, 'Nb': 4, 'Mo': 5, 'Tc': 5, 'Ru': 7, 'Rh': 8, 'Pd': 10, 'Ag': 10, 'Cd': 10,
    'Hf': 2, 'Ta': 3, 'W': 4, 'Re': 5, 'Os': 6, 'Ir': 7, 'Pt': 9, 'Au': 10,
}
# Ground-state f-electron counts for lanthanides
F_ELECTRON_COUNTS = {
    'La': 0, 'Ce': 1, 'Pr': 3, 'Nd': 4, 'Pm': 5, 'Sm': 6, 'Eu': 7,
    'Gd': 7, 'Tb': 9, 'Dy': 10, 'Ho': 11, 'Er': 12, 'Tm': 13, 'Yb': 14, 'Lu': 14,
}
# Hund's rule max unpaired electrons (high-spin, neutral atom)
# d-shell: 0→0, 1→1, 2→2, 3→3, 4→4, 5→5, 6→4, 7→3, 8→2, 9→1, 10→0
UNPAIRED_D_ELECTRONS = {
    'Sc': 1, 'Ti': 2, 'V': 3, 'Cr': 5, 'Mn': 5, 'Fe': 4, 'Co': 3, 'Ni': 2, 'Cu': 0, 'Zn': 0,
    'Y': 1, 'Zr': 2, 'Nb': 4, 'Mo': 5, 'Tc': 5, 'Ru': 4, 'Rh': 3, 'Pd': 0, 'Ag': 0, 'Cd': 0,
    'Hf': 2, 'Ta': 3, 'W': 4, 'Re': 5, 'Os': 4, 'Ir': 3, 'Pt': 1, 'Au': 0,
}
# Hund's rule max unpaired electrons for f-shell
UNPAIRED_F_ELECTRONS = {
    'La': 0, 'Ce': 1, 'Pr': 3, 'Nd': 4, 'Pm': 5, 'Sm': 6, 'Eu': 7,
    'Gd': 7, 'Tb': 5, 'Dy': 4, 'Ho': 3, 'Er': 2, 'Tm': 1, 'Yb': 0, 'Lu': 0,
}

# Default zero dict for ionic/EN features (used when formula parsing fails)
_IONIC_EN_DEFAULT = {
    'compound possible': 0.0,
    'max ionic char': 0.0,
    'avg ionic char': 0.0,
    'minimum EN difference': 0.0,
    'maximum EN difference': 0.0,
    'range EN difference': 0.0,
    'mean EN difference': 0.0,
    'std_dev EN difference': 0.0
}

def get_cached_composition(formula):
    """
    Parses and returns a Pymatgen Composition object, caching it in memory.
    """
    if not formula or pd.isna(formula):
        return None
    formula_str = str(formula).strip()
    if formula_str not in COMPOSITION_CACHE:
        try:
            COMPOSITION_CACHE[formula_str] = Composition(formula_str)
        except Exception:
            COMPOSITION_CACHE[formula_str] = None
    return COMPOSITION_CACHE[formula_str]


def compute_goldschmidt_tolerance_factor(formula):
    """
    Computes the Goldschmidt tolerance factor t = (r_A + r_O) / [sqrt(2) * (r_B + r_O)]
    for oxide perovskites and spinels using Shannon ionic radii (CN=6).
    Values near 1.0 indicate ideal cubic geometry with 180° M-O-M superexchange.
    Deviations (t < 0.9) correspond to octahedral tilting and distorted exchange pathways.
    Returns 1.0 (ideal non-distorted baseline) for non-oxides or parsing fallbacks.
    """
    if not formula or pd.isna(formula):
        return 1.0
    comp = get_cached_composition(formula)
    if comp is None:
        return 1.0
    try:
        el_dict = comp.get_el_amt_dict()
        if 'O' not in el_dict or len(el_dict) < 2:
            return 1.0
        r_O = SHANNON_RADII_CN6.get('O', 1.40)
        cations = [el for el in el_dict.keys() if el != 'O']
        if not cations:
            return 1.0
        if len(cations) == 1:
            r_B = SHANNON_RADII_CN6.get(cations[0], 0.65)
            return float(np.clip(r_B / r_O, 0.2, 1.2))
        sorted_cations = sorted(cations, key=lambda el: SHANNON_RADII_CN6.get(el, 0.70), reverse=True)
        r_A = SHANNON_RADII_CN6.get(sorted_cations[0], 1.00)
        r_B = SHANNON_RADII_CN6.get(sorted_cations[1], 0.65)
        t = (r_A + r_O) / (np.sqrt(2.0) * (r_B + r_O))
        return float(np.clip(t, 0.5, 1.5))
    except Exception:
        return 1.0


COMMON_VALENCES = {
    # Anions (Negative charges)
    'O': -2, 'S': -2, 'Se': -2, 'Te': -2,
    'F': -1, 'Cl': -1, 'Br': -1, 'I': -1,
    'N': -3, 'P': -3, 'As': -3, 'Sb': -3, 'Bi': -3,
    # Known Cations (Positive charges)
    'H': 1, 'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,
    'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
    'Al': 3, 'Ga': 3, 'In': 3, 'Y': 3, 'La': 3, 'Sc': 3
}


def compute_oxidation_state_features(formula):
    """
    Computes oxidation state features using a highly optimized, fast heuristic charge balancer.
    Avoids the heavy combinatorial ILP solver of Composition.oxi_state_guesses() completely.
    """
    feats = {
        'mean_oxi_state_magnetic': 0.0,
        'max_oxi_state': 0.0,
        'min_oxi_state': 0.0,
        'oxi_state_range': 0.0,
        'mixed_valence_flag': 0.0,
        'charge_balance_success': 0.0
    }
    if not formula or pd.isna(formula):
        return feats
        
    formula_str = str(formula).strip()
    if formula_str in OXIDATION_STATES_CACHE:
        return OXIDATION_STATES_CACHE[formula_str]
        
    try:
        comp = get_cached_composition(formula_str)
        if comp is None:
            OXIDATION_STATES_CACHE[formula_str] = feats
            return feats
            
        fraction_dict = comp.get_el_amt_dict()
        total_atoms = sum(fraction_dict.values())
        if total_atoms == 0:
            OXIDATION_STATES_CACHE[formula_str] = feats
            return feats
            
        # 1. Classify elements and sum known charges
        anion_charge = 0.0
        known_cation_charge = 0.0
        
        magnetic_elements = {'Fe', 'Co', 'Ni', 'Mn', 'Cr'}
        mag_amount = 0.0
        other_tm_amount = 0.0
        
        assigned_valences = {}
        
        for el_symbol, amt in fraction_dict.items():
            if el_symbol in COMMON_VALENCES:
                val = COMMON_VALENCES[el_symbol]
                assigned_valences[el_symbol] = float(val)
                if val < 0:
                    anion_charge += amt * val
                else:
                    known_cation_charge += amt * val
            elif el_symbol in magnetic_elements:
                mag_amount += amt
            else:
                other_tm_amount += amt
                
        # 2. Heuristic charge balance
        unassigned_amount = mag_amount + other_tm_amount
        net_known_charge = anion_charge + known_cation_charge
        
        if unassigned_amount > 0:
            # Transition metals balance the net charge
            tm_valence = -net_known_charge / unassigned_amount
            for el_symbol, amt in fraction_dict.items():
                if el_symbol not in COMMON_VALENCES:
                    assigned_valences[el_symbol] = float(tm_valence)
            feats['charge_balance_success'] = 1.0
        else:
            if abs(net_known_charge) < 1e-4:
                feats['charge_balance_success'] = 1.0
                
        # 3. Compute stats from assigned valences
        valences = list(assigned_valences.values())
        if valences:
            feats['max_oxi_state'] = float(max(valences))
            feats['min_oxi_state'] = float(min(valences))
            feats['oxi_state_range'] = feats['max_oxi_state'] - feats['min_oxi_state']
            
        # mean_oxi_state_magnetic
        mag_val_sum = 0.0
        mag_amt_sum = 0.0
        for el_symbol, amt in fraction_dict.items():
            if el_symbol in magnetic_elements and el_symbol in assigned_valences:
                mag_val_sum += amt * assigned_valences[el_symbol]
                mag_amt_sum += amt
                
        if mag_amt_sum > 0:
            feats['mean_oxi_state_magnetic'] = mag_val_sum / mag_amt_sum
            
        # mixed_valence_flag: if any assigned valence is non-integer
        is_fractional = any(abs(v - round(v)) > 1e-3 for v in valences)
        if is_fractional:
            feats['mixed_valence_flag'] = 1.0
            
    except Exception:
        pass
        
    OXIDATION_STATES_CACHE[formula_str] = feats
    return feats

def compute_ionic_and_en_features(formula):
    """
    Computes ionic character and electronegativity difference features using
    a fast algebraic approach (no NP-hard oxi_state_guesses() calls).
    Equivalent to Matminer's IonProperty + ElectronegativityDiff but runs in microseconds.
    """
    if formula in IONIC_EN_CACHE:
        return IONIC_EN_CACHE[formula]

    feats = dict(_IONIC_EN_DEFAULT)  # copy defaults
    comp = get_cached_composition(formula)
    if not comp:
        IONIC_EN_CACHE[formula] = feats
        return feats

    try:
        elements = list(comp.element_composition.keys())
        fractions = {el: comp.get_atomic_fraction(el) for el in elements}

        # Gather electronegativities (Pauling scale)
        electronegativities = {}
        for el in elements:
            x = el.X
            if x is not None and not np.isnan(float(x)):
                electronegativities[el] = float(x)

        if electronegativities:
            diffs = []
            weights = []
            for i, el_a in enumerate(elements):
                x_a = electronegativities.get(el_a)
                if x_a is None:
                    continue
                for j, el_b in enumerate(elements):
                    if i >= j:
                        continue
                    x_b = electronegativities.get(el_b)
                    if x_b is None:
                        continue
                    diff = abs(x_a - x_b)
                    w = fractions[el_a] * fractions[el_b]
                    diffs.append(diff)
                    weights.append(w)

            if diffs:
                feats['minimum EN difference'] = float(min(diffs))
                feats['maximum EN difference'] = float(max(diffs))
                feats['range EN difference'] = feats['maximum EN difference'] - feats['minimum EN difference']
                sum_w = sum(weights)
                if sum_w > 0:
                    mean_diff = sum(d * w for d, w in zip(diffs, weights)) / sum_w
                    feats['mean EN difference'] = float(mean_diff)
                    var_diff = sum(w * (d - mean_diff)**2 for d, w in zip(diffs, weights)) / sum_w
                    feats['std_dev EN difference'] = float(np.sqrt(max(0.0, var_diff)))

            # Ionic character: Pauling formula ic = 1 - exp(-0.25 * (X_a - X_b)^2)
            ionic_chars = []
            ionic_weights = []
            for i, el_a in enumerate(elements):
                x_a = electronegativities.get(el_a)
                if x_a is None:
                    continue
                for j, el_b in enumerate(elements):
                    if i >= j:
                        continue
                    x_b = electronegativities.get(el_b)
                    if x_b is None:
                        continue
                    ic = 1.0 - np.exp(-0.25 * (x_a - x_b)**2)
                    ionic_chars.append(ic)
                    ionic_weights.append(fractions[el_a] * fractions[el_b])

            if ionic_chars:
                feats['max ionic char'] = float(max(ionic_chars))
                feats['avg ionic char'] = float(sum(ic * w for ic, w in zip(ionic_chars, ionic_weights)))

        # 'compound possible' via fast heuristic charge balance
        ox_feats = compute_oxidation_state_features(formula)
        feats['compound possible'] = ox_feats['charge_balance_success']

    except Exception:
        pass

    IONIC_EN_CACHE[formula] = feats
    return feats


def compute_formula_family_features(formula):
    """
    Flags matching standard magnetic material families.
    Uses caching and cached compositions for high speed.
    """
    feats = {
        'is_binary': 0.0,
        'is_ternary': 0.0,
        'is_quaternary': 0.0,
        'AB_like': 0.0,
        'AB2_like': 0.0,
        'ABO3_like': 0.0,
        'AB2O4_like': 0.0,
        'A2BO4_like': 0.0,
        'AFeO3_like': 0.0,
        'MFe2O4_like': 0.0,
        'hexaferrite_like': 0.0,
        'heusler_X2YZ_like': 0.0,
        'half_heusler_XYZ_like': 0.0
    }
    if not formula or pd.isna(formula):
        return feats
        
    formula_str = str(formula).strip()
    if formula_str in FORMULA_FAMILY_CACHE:
        return FORMULA_FAMILY_CACHE[formula_str]
        
    try:
        comp = get_cached_composition(formula_str)
        if comp is None:
            FORMULA_FAMILY_CACHE[formula_str] = feats
            return feats
            
        red_comp, factor = comp.get_reduced_composition_and_factor()
        el_amounts = red_comp.values()
        num_el = len(el_amounts)
    except Exception:
        FORMULA_FAMILY_CACHE[formula_str] = feats
        return feats
        
    feats['is_binary'] = 1.0 if num_el == 2 else 0.0
    feats['is_ternary'] = 1.0 if num_el == 3 else 0.0
    feats['is_quaternary'] = 1.0 if num_el >= 4 else 0.0
    
    amounts_sorted = sorted(el_amounts)
    
    if num_el == 2 and amounts_sorted == [1.0, 1.0]:
        feats['AB_like'] = 1.0
        
    if num_el == 2 and amounts_sorted == [1.0, 2.0]:
        feats['AB2_like'] = 1.0
        
    if num_el == 3 and amounts_sorted == [1.0, 1.0, 3.0]:
        feats['ABO3_like'] = 1.0
        
    if num_el == 3 and amounts_sorted == [1.0, 2.0, 4.0]:
        feats['AB2O4_like'] = 1.0
        
    if num_el == 3 and amounts_sorted == [1.0, 2.0, 4.0]:
        feats['A2BO4_like'] = 1.0
        
    if num_el == 3 and 'Fe' in red_comp and 'O' in red_comp:
        if red_comp['Fe'] == 1.0 and red_comp['O'] == 3.0:
            feats['AFeO3_like'] = 1.0
            
    if num_el == 3 and 'Fe' in red_comp and 'O' in red_comp:
        if red_comp['Fe'] == 2.0 and red_comp['O'] == 4.0:
            feats['MFe2O4_like'] = 1.0
            
    if 'Fe' in red_comp and 'O' in red_comp:
        fe_amt = red_comp['Fe']
        o_amt = red_comp['O']
        if 10.0 <= fe_amt <= 13.0 and 18.0 <= o_amt <= 20.0:
            feats['hexaferrite_like'] = 1.0
            
    if num_el == 3 and amounts_sorted == [1.0, 1.0, 2.0]:
        feats['heusler_X2YZ_like'] = 1.0
        
    if num_el == 3 and amounts_sorted == [1.0, 1.0, 1.0]:
        feats['half_heusler_XYZ_like'] = 1.0
        
    FORMULA_FAMILY_CACHE[formula_str] = feats
    return feats

def compute_composition_features(formula):
    """
    Computes standard physical and chemical features for a single chemical formula string.
    Returns a dictionary of features.
    """
    # Initialize all element proportions to 0.0
    feats = {el: 0.0 for el in ELEMENT_LIST}
    
    # Initialize other physical descriptors
    feats.update({
        'Average_Weight': 0.0,
        'Average_Electronegativity': 0.0,
        'Total_Electrons': 0.0,
        'Num_Atoms': 0.0,
        'Avg_Atomic_Number': 0.0,
        'average_period': 0.0,
        'average_group': 0.0,
        'Entropy': 0.0,
        'L2_norm': 0.0,
        'avg_magnetic_moment': 0.0,
        'Magnetic_proportion': 0.0,
        # Oxidation states
        'mean_oxi_state_magnetic': 0.0,
        'max_oxi_state': 0.0,
        'min_oxi_state': 0.0,
        'oxi_state_range': 0.0,
        'mixed_valence_flag': 0.0,
        'charge_balance_success': 0.0,
        # Stoichiometry / material families
        'is_binary': 0.0,
        'is_ternary': 0.0,
        'is_quaternary': 0.0,
        'AB_like': 0.0,
        'AB2_like': 0.0,
        'ABO3_like': 0.0,
        'AB2O4_like': 0.0,
        'A2BO4_like': 0.0,
        'AFeO3_like': 0.0,
        'MFe2O4_like': 0.0,
        'hexaferrite_like': 0.0,
        'heusler_X2YZ_like': 0.0,
        'half_heusler_XYZ_like': 0.0
    })
    
    if not formula or pd.isna(formula):
        return feats
        
    try:
        comp = get_cached_composition(formula)
        if comp is None:
            return feats
    except Exception:
        return feats
        
    fraction_dict = comp.get_el_amt_dict()
    total_atoms = sum(fraction_dict.values())
    if total_atoms == 0:
        return feats
        
    # Set element proportions
    for el_symbol, amt in fraction_dict.items():
        if el_symbol in feats:
            feats[el_symbol] = amt / total_atoms
            
    # Calculate physical features
    w_sum = 0.0
    x_sum = 0.0
    x_count = 0.0
    electrons_sum = 0.0
    atomic_num_sum = 0.0
    period_sum = 0.0
    group_sum = 0.0
    
    magnetic_elements = {'Fe', 'Co', 'Ni', 'Gd', 'Dy'}
    
    magnetic_prop = 0.0
    entropy = 0.0
    l2_sum = 0.0
    
    for el_symbol, amt in fraction_dict.items():
        fraction = amt / total_atoms
        l2_sum += fraction**2
        if fraction > 0:
            entropy -= fraction * np.log(fraction)
            
        if el_symbol in magnetic_elements:
            magnetic_prop += fraction
            
        try:
            el = Element(el_symbol)
            w_sum += fraction * float(el.atomic_mass)
            
            # Electronegativity can be None (like for noble gases)
            if el.X is not None and not np.isnan(el.X):
                x_sum += fraction * float(el.X)
                x_count += fraction
                
            electrons_sum += amt * int(el.number)
            atomic_num_sum += fraction * int(el.number)
            
            if el.row is not None:
                period_sum += fraction * float(el.row)
            if el.group is not None:
                group_sum += fraction * float(el.group)
        except Exception:
            pass
            
    feats['Average_Weight'] = w_sum if w_sum > 0.0 else np.nan
    feats['Average_Electronegativity'] = x_sum / x_count if x_count > 0 else 0.0
    feats['Total_Electrons'] = electrons_sum
    feats['Num_Atoms'] = total_atoms
    feats['Avg_Atomic_Number'] = atomic_num_sum
    feats['average_period'] = period_sum
    feats['average_group'] = group_sum
    feats['Entropy'] = entropy
    feats['Magnetic_proportion'] = magnetic_prop
    feats['L2_norm'] = np.sqrt(l2_sum)
    
    # Custom approximate magnetic moments for specific elements in standard state
    mag_moments = {'Fe': 2.2, 'Co': 1.7, 'Ni': 0.6, 'Gd': 7.6, 'Dy': 10.0, 'Mn': 3.0, 'Cr': 2.0}
    feats['avg_magnetic_moment'] = sum((amt / total_atoms) * mag_moments.get(sym, 0.0) for sym, amt in fraction_dict.items())
    
    # Merge custom oxidation and family features
    ox_feats = compute_oxidation_state_features(formula)
    fam_feats = compute_formula_family_features(formula)
    feats.update(ox_feats)
    feats.update(fam_feats)
    
    return feats

def add_magnetic_chemistry_features(df):
    """
    Computes 27 advanced AFM-relevant magnetic-chemistry descriptors.
    """
    print("Computing 27 AFM-relevant magnetic-chemistry features (incl. d-electron physics)...")
    df = df.copy()
    
    df["magnetic_3d_fraction"] = 0.0
    df["rare_earth_fraction"] = 0.0
    df["transition_metal_fraction"] = 0.0
    df["oxygen_fraction"] = 0.0
    df["chalcogen_fraction"] = 0.0
    df["pnictogen_fraction"] = 0.0
    df["halogen_fraction"] = 0.0
    
    df["num_magnetic_3d_elements"] = 0.0
    df["num_rare_earth_elements"] = 0.0
    df["num_transition_metals"] = 0.0
    
    df["contains_Mn"] = 0.0
    df["contains_Cr"] = 0.0
    df["contains_Fe"] = 0.0
    df["contains_Co"] = 0.0
    df["contains_Ni"] = 0.0
    df["contains_rare_earth"] = 0.0
    df["contains_oxygen"] = 0.0
    
    df["transition_metal_oxide_flag"] = 0.0
    df["magnetic_nonmagnetic_ratio"] = 0.0

    # Calculate for each row using the existing ELEMENT_LIST columns
    for el in MAGNETIC_3D:
        if el in df.columns:
            df["magnetic_3d_fraction"] += df[el].fillna(0.0)
            df["num_magnetic_3d_elements"] += (df[el].fillna(0.0) > 0).astype(float)
            
    for el in RARE_EARTH:
        if el in df.columns:
            df["rare_earth_fraction"] += df[el].fillna(0.0)
            df["num_rare_earth_elements"] += (df[el].fillna(0.0) > 0).astype(float)
            
    for el in TRANSITION_METALS:
        if el in df.columns:
            df["transition_metal_fraction"] += df[el].fillna(0.0)
            df["num_transition_metals"] += (df[el].fillna(0.0) > 0).astype(float)
            
    if "O" in df.columns:
        df["oxygen_fraction"] = df["O"].fillna(0.0)
        df["contains_oxygen"] = (df["O"].fillna(0.0) > 0).astype(float)
        
    for el in CHALCOGENS:
        if el in df.columns:
            df["chalcogen_fraction"] += df[el].fillna(0.0)
            
    for el in PNICTOGENS:
        if el in df.columns:
            df["pnictogen_fraction"] += df[el].fillna(0.0)
            
    for el in HALOGENS:
        if el in df.columns:
            df["halogen_fraction"] += df[el].fillna(0.0)
            
    if "Mn" in df.columns:
        df["contains_Mn"] = (df["Mn"].fillna(0.0) > 0).astype(float)
    if "Cr" in df.columns:
        df["contains_Cr"] = (df["Cr"].fillna(0.0) > 0).astype(float)
    if "Fe" in df.columns:
        df["contains_Fe"] = (df["Fe"].fillna(0.0) > 0).astype(float)
    if "Co" in df.columns:
        df["contains_Co"] = (df["Co"].fillna(0.0) > 0).astype(float)
    if "Ni" in df.columns:
        df["contains_Ni"] = (df["Ni"].fillna(0.0) > 0).astype(float)
        
    df["contains_rare_earth"] = (df["rare_earth_fraction"] > 0).astype(float)
    df["transition_metal_oxide_flag"] = ((df["transition_metal_fraction"] > 0) & (df["oxygen_fraction"] > 0)).astype(float)
    
    mag_frac = df["magnetic_3d_fraction"] + df["rare_earth_fraction"]
    df["magnetic_nonmagnetic_ratio"] = mag_frac / (1.0 - mag_frac + 1e-6)
    
    # 3. Element Category Interaction Features
    df["Fe_O_interaction"] = df["Fe"].fillna(0.0) * df["oxygen_fraction"]
    df["Mn_O_interaction"] = df["Mn"].fillna(0.0) * df["oxygen_fraction"]
    df["Cr_O_interaction"] = df["Cr"].fillna(0.0) * df["oxygen_fraction"]
    df["Co_O_interaction"] = df["Co"].fillna(0.0) * df["oxygen_fraction"]
    df["Ni_O_interaction"] = df["Ni"].fillna(0.0) * df["oxygen_fraction"]
    
    df["rare_earth_TM_interaction"] = df["rare_earth_fraction"] * df["transition_metal_fraction"]
    df["magnetic_3d_chalcogen_interaction"] = df["magnetic_3d_fraction"] * df["chalcogen_fraction"]
    df["magnetic_3d_pnictogen_interaction"] = df["magnetic_3d_fraction"] * df["pnictogen_fraction"]
    
    # 4. Magnetic Exchange Proxy Features
    df["transition_metal_halide_flag"] = ((df["transition_metal_fraction"] > 0) & (df["halogen_fraction"] > 0)).astype(float)
    df["transition_metal_chalcogenide_flag"] = ((df["transition_metal_fraction"] > 0) & (df["chalcogen_fraction"] > 0)).astype(float)
    df["rare_earth_transition_metal_flag"] = ((df["rare_earth_fraction"] > 0) & (df["transition_metal_fraction"] > 0)).astype(float)
    df["Fe_Co_alloy_flag"] = ((df["Fe"].fillna(0.0) > 0) & (df["Co"].fillna(0.0) > 0)).astype(float)
    df["Mn_oxide_flag"] = ((df["Mn"].fillna(0.0) > 0) & (df["oxygen_fraction"] > 0)).astype(float)
    df["Cr_oxide_flag"] = ((df["Cr"].fillna(0.0) > 0) & (df["oxygen_fraction"] > 0)).astype(float)
    df["ferrite_like_flag"] = ((df["Fe"].fillna(0.0) > 0) & (df["oxygen_fraction"] > 0)).astype(float)
    
    # 5. NEW: D-Electron and Magnetic Physics Features for AFM/NM Distinction
    # These are the primary physics differentiators: AFM has half-filled d-shells (d3-d5),
    # NM has empty or fully-filled d-shells (d0/d10).
    
    weighted_d = np.zeros(len(df))
    weighted_f = np.zeros(len(df))
    unpaired_e = np.zeros(len(df))

    for el_sym in D_ELECTRON_COUNTS:
        if el_sym in df.columns:
            frac = df[el_sym].fillna(0.0).values
            weighted_d += frac * D_ELECTRON_COUNTS[el_sym]
            unpaired_e += frac * UNPAIRED_D_ELECTRONS.get(el_sym, 0)

    for el_sym in F_ELECTRON_COUNTS:
        if el_sym in df.columns:
            frac = df[el_sym].fillna(0.0).values
            weighted_f += frac * F_ELECTRON_COUNTS[el_sym]
            unpaired_e += frac * UNPAIRED_F_ELECTRONS.get(el_sym, 0)

    df["weighted_d_electron_count"] = weighted_d
    df["weighted_f_electron_count"] = weighted_f
    df["d_band_filling_fraction"] = weighted_d / 10.0  # d-shell has 10 slots; 0.3-0.5 = half-filled -> AFM
    df["unpaired_electron_estimate"] = unpaired_e  # NM has ~0; AFM/FM have >0

    # Magnetic ion concentration: below percolation threshold (~0.15-0.20) -> likely NM
    mag_frac_total = df["magnetic_3d_fraction"] + df["rare_earth_fraction"]
    df["magnetic_ion_concentration"] = mag_frac_total
    

    # 6. Paper-inspired elemental magnetic priors. These closely match the
    # features highlighted in the NEMAD paper: high-Curie-element proportion,
    # rare-earth proportion, element fractions for Fe/Co/Ni/Mn/Cr/O and average
    # magnetic moment. They also prevent stale selected-feature caches from
    # referring to missing columns.
    df["elemental_fm_fraction"] = 0.0
    df["elemental_afm_fraction"] = 0.0
    df["strong_fm_3d_fraction"] = 0.0
    df["strong_afm_3d_fraction"] = 0.0
    df["rare_earth_fm_fraction"] = 0.0
    df["rare_earth_afm_fraction"] = 0.0
    df["high_curie_element_fraction"] = 0.0
    df["elemental_fm_tc_prior"] = 0.0
    df["elemental_afm_tn_prior"] = 0.0
    df["elemental_ordering_temp_sum_prior"] = 0.0
    df["max_elemental_ordering_temp"] = 0.0

    for el in ELEMENT_LIST:
        if el not in df.columns:
            continue
        frac = df[el].fillna(0.0)
        if el in ELEMENTAL_FM:
            df["elemental_fm_fraction"] += frac
        if el in ELEMENTAL_AFM:
            df["elemental_afm_fraction"] += frac
        if el in STRONG_FM_3D:
            df["strong_fm_3d_fraction"] += frac
        if el in STRONG_AFM_3D:
            df["strong_afm_3d_fraction"] += frac
        if el in RARE_EARTH_FM:
            df["rare_earth_fm_fraction"] += frac
        if el in RARE_EARTH_AFM:
            df["rare_earth_afm_fraction"] += frac
        if el in HIGH_CURIE_ELEMENTS:
            df["high_curie_element_fraction"] += frac
        if el in ELEMENTAL_TC_K:
            val = frac * ELEMENTAL_TC_K[el]
            df["elemental_fm_tc_prior"] += val
            df["elemental_ordering_temp_sum_prior"] += val
            df["max_elemental_ordering_temp"] = np.maximum(df["max_elemental_ordering_temp"], frac * 0.0 + ELEMENTAL_TC_K[el] * (frac > 0).astype(float))
        if el in ELEMENTAL_TN_K:
            val = frac * ELEMENTAL_TN_K[el]
            df["elemental_afm_tn_prior"] += val
            df["elemental_ordering_temp_sum_prior"] += val
            df["max_elemental_ordering_temp"] = np.maximum(df["max_elemental_ordering_temp"], frac * 0.0 + ELEMENTAL_TN_K[el] * (frac > 0).astype(float))

    df["elemental_fm_minus_afm_temp_prior"] = df["elemental_fm_tc_prior"] - df["elemental_afm_tn_prior"]
    df["fm_to_afm_element_ratio"] = df["elemental_fm_fraction"] / (df["elemental_afm_fraction"] + 1e-6)
    df["high_curie_to_magnetic_fraction"] = df["high_curie_element_fraction"] / (df["magnetic_ion_concentration"] + 1e-6)

    return df


def add_boundary_interaction_features(df):
    """
    Adds compact interaction descriptors for difficult FM/AFM/NM boundaries.
    These features combine magnetic-ion concentration, d/f-electron tendency,
    ionicity/covalency and ligand-mediated exchange without using target labels.
    """
    print("Computing boundary-aware magnetic interaction indices...")
    df = df.copy()

    def col(name, default=0.0):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index)

    magnetic_ion = col("magnetic_ion_concentration")
    strong_fm = col("strong_fm_3d_fraction")
    strong_afm = col("strong_afm_3d_fraction")
    rare_earth = col("rare_earth_fraction")
    weighted_f = col("weighted_f_electron_count")
    soc = col("weighted_soc_constant") / 100.0
    d_fill = col("d_band_filling_fraction")
    unpaired = col("unpaired_electron_estimate")
    oxygen = col("oxygen_fraction")
    ionicity = col("mag_cation_anion_ionicity")
    covalency = np.clip(1.0 - col("avg ionic char", default=0.5), 0.0, 1.0)
    charge_transfer = col("charge_transfer_asymmetry")

    unpaired_norm = np.clip(unpaired / 7.0, 0.0, 1.0)

    df["afm_superexchange_index"] = strong_afm * oxygen * ionicity
    df["afm_nm_boundary_index"] = d_fill * strong_afm * (1.0 + ionicity)
    df["fm_metallic_index"] = strong_fm * magnetic_ion * covalency
    df["nm_closed_shell_index"] = (1.0 - unpaired_norm) * (1.0 - np.clip(magnetic_ion, 0.0, 1.0))
    df["rare_earth_ordering_index"] = rare_earth * weighted_f * (1.0 + soc)
    df["charge_transfer_exchange_index"] = charge_transfer * strong_afm * oxygen
    return df


def add_advanced_physical_features(df, formula_col='reduced_formula'):
    """
    Computes rigorous physical descriptors grounded in exchange mechanisms,
    single-ion spin-orbit coupling, mean-field scaling, and magnetic frustration:
      1. Goodenough-Kanamori-Anderson (GKA) exchange & d-shell half-filling
      2. Stoner itinerant ferromagnetism parameters
      3. Single-ion SOC constants & crystal-field uniaxial anisotropy
      4. Mean-field spin scaling laws & ordering temperature priors
      5. Geometric & exchange frustration indices
      6. Goldschmidt structural tolerance factors
    """
    print("Computing advanced physics descriptors (GKA rules, Stoner, SOC, Mean-field, Frustration)...")
    df = df.copy()

    def col(name, default=0.0):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index)

    # 1. GKA & Exchange Physics
    weighted_d = col("weighted_d_electron_count")
    # Proximity to d^5 half-filled shell (0.0 = exact half-filling -> maximum superexchange AFM)
    df["d_shell_half_filling_index"] = np.abs(weighted_d - 5.0) / 5.0

    # Double exchange candidate (mixed-valence Mn/Fe/Co oxides)
    mn = col("Mn")
    fe = col("Fe")
    co = col("Co")
    oxygen = col("oxygen_fraction")
    mixed_val = col("mixed_valence_flag")
    df["double_exchange_candidate"] = (
        ((mn > 0) | (fe > 0) | (co > 0)) & (oxygen > 0.15) & (mixed_val > 0)
    ).astype(float)

    # 2. Stoner Itinerant Ferromagnetism
    stoner_sum = np.zeros(len(df))
    for el_sym, stoner_val in STONER_PARAM.items():
        if el_sym in df.columns:
            stoner_sum += col(el_sym).values * stoner_val
    df["weighted_stoner_parameter"] = stoner_sum
    df["near_stoner_threshold_flag"] = (
        (df["weighted_stoner_parameter"] > 0.6) & (oxygen < 0.1)
    ).astype(float)

    # 3. Spin-Orbit Coupling & Magnetocrystalline Anisotropy
    soc_sum = np.zeros(len(df))
    for el_sym, soc_val in SOC_CONSTANTS_MEV.items():
        if el_sym in df.columns:
            soc_sum += col(el_sym).values * soc_val
    df["weighted_soc_constant"] = soc_sum

    if "crystal_system" in df.columns:
        cs = df["crystal_system"].astype(str).str.lower()
        df["uniaxial_symmetry_flag"] = cs.isin(UNIAXIAL_SYSTEMS).astype(float)
    else:
        df["uniaxial_symmetry_flag"] = 0.0

    df["anisotropy_figure_of_merit"] = df["weighted_soc_constant"] * df["uniaxial_symmetry_flag"]

    re_aniso = np.zeros(len(df))
    for el_sym, sign_val in RE_ANISOTROPY_SIGN.items():
        if el_sym in df.columns:
            re_aniso += col(el_sym).values * sign_val
    df["re_anisotropy_sign_index"] = re_aniso

    # 4. Mean-Field Theory Scaling Laws
    spin_sum = np.zeros(len(df))
    for el_sym, s_val in SPIN_QUANTUM_NUMBERS.items():
        if el_sym in df.columns:
            spin_sum += col(el_sym).values * (s_val * (s_val + 1.0))
    df["mean_field_spin_factor"] = spin_sum

    # 5. Magnetic Frustration & Competing Exchange
    if "spacegroup_number" in df.columns:
        sg = pd.to_numeric(df["spacegroup_number"], errors="coerce").fillna(0).astype(int)
        df["frustrated_lattice_flag"] = sg.isin(FRUSTRATED_SPACE_GROUPS).astype(float)
    else:
        df["frustrated_lattice_flag"] = 0.0

    if "crystal_system" in df.columns:
        cs = df["crystal_system"].astype(str).str.lower()
        df["triangular_lattice_flag"] = cs.isin({"trigonal", "hexagonal"}).astype(float)
    else:
        df["triangular_lattice_flag"] = 0.0

    fm_frac = col("elemental_fm_fraction")
    afm_frac = col("elemental_afm_fraction")
    df["exchange_competition_index"] = 4.0 * fm_frac * afm_frac
    eps = 1e-6
    df["exchange_balance_entropy"] = -(
        fm_frac * np.log(np.clip(fm_frac, eps, 1.0)) +
        afm_frac * np.log(np.clip(afm_frac, eps, 1.0))
    )

    # 6. Goldschmidt Structural Tolerance
    if formula_col in df.columns:
        unique_formulas = df[formula_col].dropna().unique()
        tol_map = {f: compute_goldschmidt_tolerance_factor(f) for f in unique_formulas}
        df["goldschmidt_tolerance_factor"] = df[formula_col].map(tol_map).fillna(1.0).astype(float)
    else:
        df["goldschmidt_tolerance_factor"] = 1.0

    # 7. de Gennes Factor for Analytical Rare-Earth RKKY Exchange Scaling
    # Tc and TN scale strictly as (g_J - 1)^2 * J(J + 1) across 4f intermetallics
    de_gennes_sum = np.zeros(len(df))
    for el_sym, dg_val in DE_GENNES_FACTORS.items():
        if el_sym in df.columns:
            de_gennes_sum += col(el_sym).values * dg_val
    df["weighted_de_gennes_factor"] = de_gennes_sum

    # 8. Valence Electron Concentration (VEC) & Slater-Pauling Alloy Coordinates
    # Dictates itinerant exchange peak (Fe0.7Co0.3 at VEC=8.35, mu=2.45 mu_B, Tc=1250 K)
    vec_sum = np.zeros(len(df))
    for el_sym, v_count in VALENCE_ELECTRON_COUNT.items():
        if el_sym in df.columns:
            vec_sum += col(el_sym).values * v_count
    df["valence_electron_concentration"] = vec_sum
    # Distance from peak Slater-Pauling ferromagnetic coordinate
    df["slater_pauling_peak_distance"] = np.abs(vec_sum - 8.35)

    # 9. Octahedral Buckling & Quantitative Superexchange Transfer Factor
    # For perovskites and rock-salts, M-O-M angle decreases from 180 deg as t < 1.0
    t_clipped = df["goldschmidt_tolerance_factor"].clip(0.65, 1.15)
    buckling_dev = np.maximum(0.0, 1.0 - t_clipped) * 120.0
    df["octahedral_buckling_angle_deg"] = 180.0 - buckling_dev
    rad_angle = np.radians(df["octahedral_buckling_angle_deg"])
    df["superexchange_transfer_factor"] = np.cos(rad_angle) ** 2

    return df


def add_innovative_dao_physics_features(df):
    """
    Computes cutting-edge physics descriptors fusing DAO 3D structures with
    quantum exchange and micromagnetics:
      1. dao_bethe_slater_ratio: d_min / (2 * r_3d) predicting direct exchange sign.
      2. agk_superexchange_energy: Anderson-Goodenough-Kanamori transfer energy.
      3. dao_slater_pauling_volumetric_density: volumetric moment density prior.
      4. zener_double_exchange_hopping: carrier-mediated transfer in mixed-valence oxides.
    """
    df = df.copy()
    def col(name, default=0.0):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index)

    # 1. Bethe-Slater Overlap Ratio from DAO 3D structure
    r_3d = 1.25  # Average 3d radius ~1.25 Angstroms
    min_dist = col("dao_min_bond_dist", default=2.4)
    df["dao_bethe_slater_ratio"] = np.clip(min_dist / (2.0 * r_3d), 0.1, 3.0)

    # 2. Anderson-Goodenough-Kanamori (AGK) Superexchange Energy
    buckling = col("superexchange_transfer_factor", default=1.0)
    mag_3d = col("magnetic_3d_fraction", default=0.0)
    oxygen = col("oxygen_fraction", default=0.0)
    ionicity = col("avg ionic char", default=0.5)
    covalency = np.clip(1.0 - ionicity, 0.0, 1.0)
    df["agk_superexchange_energy"] = (
        (covalency / np.maximum(min_dist ** 3.5, 0.5))
        * buckling * mag_3d * oxygen
    )

    # 3. Volumetric Slater-Pauling Magnetization Density from DAO unit cell
    df["dao_slater_pauling_volumetric_density"] = col("dao_moment_density", default=0.06)

    # 4. Zener Double Exchange Hopping in Mixed-Valence Systems
    mixed_val = col("mixed_valence_flag", default=0.0)
    df["zener_double_exchange_hopping"] = mixed_val * (covalency / np.maximum(min_dist, 1.0))

    return df


def get_target_specialized_feature_sets():
    """
    Returns Pareto-optimal, physics-specialized, non-collinear feature sets for each target family:
      - classification (85 features for FM/AFM/NM classification)
      - curie (80 features for Curie temperature regression)
      - neel (75 features for Néel temperature regression)
      - coercivity (65 features for Hc)
      - magnetization (65 features for Ms)
      - anisotropy (60 features for K1)
      - energy_product (50 features for BH_max)
    """
    json_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'featurization', 'target_specialized_features.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass

    f_phase = [
        'dao_bethe_slater_ratio', 'agk_superexchange_energy', 'zener_double_exchange_hopping',
        'magnetic_3d_fraction', 'rare_earth_fraction', 'oxygen_fraction', 'magnetic_ion_concentration',
        'unpaired_electron_estimate', 'd_band_filling_fraction',
        'afm_superexchange_index', 'afm_nm_boundary_index', 'exchange_balance_entropy',
        'slater_pauling_peak_distance', 'weighted_stoner_parameter', 'mean_field_spin_factor',
        'weighted_soc_constant', 'valence_electron_concentration', 'goldschmidt_tolerance_factor',
        'dao_moment_density', 'dao_magnetic_concentration', 'dao_exchange_connectivity',
        'dao_min_bond_dist', 'dao_axial_distortion', 'dao_stability_weight',
        'dao_volume_per_atom', 'dao_density', 'dao_coordination_num',
        'dao_is_uniaxial', 'dao_is_centrosymmetric', 'dao_spacegroup_num',
        'dao_msn_exchange_stiffness', 'dao_msn_bethe_slater_ratio', 'dao_msn_ms_emu_g',
        'MagpieData mean Electronegativity', 'MagpieData range Electronegativity',
        'MagpieData mean GSvolume_pa', 'MagpieData range GSvolume_pa',
        'MagpieData mean NdUnfilled', 'MagpieData mean NpUnfilled',
        'MagpieData mean MeltingT', 'MagpieData range MeltingT',
        'MagpieData mean MendeleevNumber',
        'Num_Atoms'
    ]

    f_tc = [
        'dao_slater_pauling_volumetric_density', 'dao_bethe_slater_ratio', 'zener_double_exchange_hopping',
        'slater_pauling_peak_distance', 'weighted_stoner_parameter',
        'valence_electron_concentration', 'weighted_de_gennes_factor', 'mean_field_spin_factor',
        'magnetic_3d_fraction', 'rare_earth_fraction', 'magnetic_ion_concentration',
        'd_band_filling_fraction', 'weighted_soc_constant', 'exchange_balance_entropy',
        'dao_exchange_connectivity', 'dao_moment_density',
        'dao_density', 'dao_volume_per_atom', 'dao_coordination_num',
        'dao_stability_weight', 'dao_packing_fraction', 'dao_min_bond_dist',
        'dao_msn_exchange_stiffness', 'dao_msn_bethe_slater_ratio', 'dao_msn_ms_emu_g',
        'MagpieData mean MeltingT', 'MagpieData range MeltingT',
        'MagpieData mean GSvolume_pa', 'MagpieData range GSvolume_pa',
        'MagpieData mean Electronegativity',
        'MagpieData mean MendeleevNumber', 'MagpieData mean NdUnfilled',
        'MagpieData mean NpUnfilled', 'Num_Atoms',
        'Average_Weight', 'dao_spacegroup_num', 'dao_is_uniaxial'
    ]

    f_tn = [
        'agk_superexchange_energy', 'dao_bethe_slater_ratio',
        'afm_superexchange_index', 'afm_nm_boundary_index',
        'goldschmidt_tolerance_factor', 'superexchange_transfer_factor', 'octahedral_buckling_angle_deg',
        'd_shell_half_filling_index', 'd_band_filling_fraction', 'unpaired_electron_estimate',
        'magnetic_3d_fraction', 'oxygen_fraction', 'rare_earth_fraction',
        'mag_cation_anion_ionicity', 'mag_cation_anion_en_diff',
        'dao_min_bond_dist', 'dao_exchange_connectivity', 'dao_volume_per_atom',
        'dao_density', 'dao_coordination_num', 'dao_is_centrosymmetric',
        'dao_spacegroup_num', 'dao_stability_weight', 'dao_packing_fraction',
        'dao_msn_exchange_stiffness', 'dao_msn_bethe_slater_ratio',
        'MagpieData mean Electronegativity', 'MagpieData range Electronegativity',
        'MagpieData mean GSvolume_pa', 'MagpieData range GSvolume_pa',
        'MagpieData mean NdUnfilled', 'MagpieData mean MeltingT',
        'MagpieData range MeltingT', 'MagpieData mean MendeleevNumber',
        'Num_Atoms', 'exchange_balance_entropy', 'weighted_soc_constant'
    ]

    f_hc = [
        'anisotropy_figure_of_merit', 'dao_bethe_slater_ratio',
        'dao_c_over_a', 'dao_is_uniaxial', 'dao_density', 'dao_volume_per_atom',
        'dao_packing_fraction', 'dao_coordination_num', 'dao_min_bond_dist',
        'dao_spacegroup_num', 'dao_is_centrosymmetric', 'magnetic_3d_fraction',
        'rare_earth_fraction', 'dao_msn_exchange_stiffness', 'dao_msn_a20_cf',
        'MagpieData mean GSvolume_pa', 'MagpieData mean Electronegativity', 'MagpieData mean MeltingT'
    ]

    f_ms = [
        'dao_slater_pauling_volumetric_density', 'dao_moment_density', 'dao_density',
        'dao_volume_per_atom', 'dao_packing_fraction', 'dao_magnetic_concentration',
        'slater_pauling_peak_distance', 'dao_msn_ms_emu_g',
        'valence_electron_concentration', 'd_band_filling_fraction', 'magnetic_3d_fraction',
        'rare_earth_fraction', 'unpaired_electron_estimate',
        'MagpieData mean GSvolume_pa'
    ]

    f_k1 = [
        'dao_c_over_a', 'dao_axial_distortion', 'dao_is_uniaxial', 'dao_density',
        'dao_volume_per_atom', 'dao_coordination_num', 'dao_spacegroup_num',
        'weighted_soc_constant', 're_anisotropy_sign_index',
        'rare_earth_fraction', 'magnetic_3d_fraction', 'dao_msn_a20_cf',
        'MagpieData mean GSvolume_pa', 'MagpieData mean Electronegativity', 'MagpieData mean MendeleevNumber'
    ]

    f_bh = [
        'anisotropy_figure_of_merit', 'dao_slater_pauling_volumetric_density',
        'dao_density', 'dao_volume_per_atom', 'dao_c_over_a', 'dao_is_uniaxial',
        'dao_packing_fraction', 'rare_earth_fraction', 'magnetic_3d_fraction',
        'dao_msn_ms_emu_g', 'dao_msn_exchange_stiffness',
        'MagpieData mean GSvolume_pa'
    ]

    f_hardness = list(dict.fromkeys(f_hc + f_ms + f_k1 + f_bh))

    return {
        'classification': f_phase,
        'curie': f_tc,
        'neel': f_tn,
        'hardness': f_hardness,
        'coercivity': f_hc,
        'magnetization': f_ms,
        'anisotropy': f_k1,
        'k1': f_k1,
        'bh_max': f_bh,
        'energy_product': f_bh
    }


def add_symmetry_features(df):
    """
    Computes centrosymmetric flags and normalizes spacegroup number.
    """
    print("Computing space group symmetry and centrosymmetry features...")
    df = df.copy()
    
    if "point_group" in df.columns:
        pg = df["point_group"].fillna("Unknown").astype(str)
        df["is_centrosymmetric"] = pg.isin(CENTROSYMMETRIC_PG).astype(float)
        df["is_noncentrosymmetric"] = 1.0 - df["is_centrosymmetric"]
    else:
        df["is_centrosymmetric"] = 0.0
        df["is_noncentrosymmetric"] = 0.0

    if "spacegroup_number" in df.columns:
        df["spacegroup_number"] = pd.to_numeric(df["spacegroup_number"], errors="coerce").fillna(0)
    else:
        df["spacegroup_number"] = 0.0

    return df

def generate_features_for_dataset(df, formula_col='reduced_formula'):
    """
    Featurizes a dataframe by dynamically generating physics descriptors.
    """
    # Check if elements are already columns
    has_elements = all(el in df.columns for el in ['Fe', 'Co', 'Ni', 'O'])
    
    if not has_elements:
        print("Generating compositional features dynamically using pymatgen...")
        # Compute features for unique formulas to avoid slow row-by-row iterrows loop
        unique_formulas = df[formula_col].dropna().unique()
        unique_feats = {f: compute_composition_features(f) for f in unique_formulas}
        df_feats = pd.DataFrame([unique_feats.get(f, {}) for f in df[formula_col]], index=df.index)
        # Merge back
        df = pd.concat([df, df_feats], axis=1)
    else:
        print("Dataset already contains composition features. Standardizing column values and computing custom features...")
        # Fill missing composition columns if any
        all_comp_cols = ELEMENT_LIST + [
            'Average_Weight', 'Average_Electronegativity', 'Total_Electrons', 'Num_Atoms', 
            'Avg_Atomic_Number', 'average_period', 'average_group', 'Entropy', 'L2_norm',
            'avg_magnetic_moment', 'Magnetic_proportion'
        ]
        for col in all_comp_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        
        # Compute custom oxidation states and stoichiometry family features for unique formulas to avoid slow iterrows
        unique_formulas = df[formula_col].dropna().unique()
        unique_feats = {}
        for formula in unique_formulas:
            ox_feats = compute_oxidation_state_features(formula)
            fam_feats = compute_formula_family_features(formula)
            unique_feats[formula] = {**ox_feats, **fam_feats}
            
        df_new_feats = pd.DataFrame([unique_feats.get(f, {}) for f in df[formula_col]], index=df.index)
        df = pd.concat([df, df_new_feats], axis=1)
                
    # Generate magnetic-chemistry features
    df = add_magnetic_chemistry_features(df)
    
    # Generate symmetry features
    df = add_symmetry_features(df)
    
    # Generate advanced composition features using Matminer (Magpie + ValenceOrbital only) with caching
    global _MATMINER_CACHE
    if '_MATMINER_CACHE' not in globals():
        globals()['_MATMINER_CACHE'] = None

    print("Generating advanced chemical features using Matminer (Magpie + ValenceOrbital)...")
    from matminer.featurizers.composition import ElementProperty, ValenceOrbital
    
    def get_comp(formula):
        if not isinstance(formula, str) or formula.strip() == '':
            return None
        try:
            return Composition(formula.strip())
        except Exception:
            return None
            
    # Initialize cache if needed
    if _MATMINER_CACHE is None:
        cache_candidates = [
            'MagMat-Discovery/output/featurization/matminer_cache.csv',
            'output/featurization/matminer_cache.csv'
        ]
        cache_path = None
        for p in cache_candidates:
            if os.path.exists(p):
                if cache_path is None or os.path.getsize(p) > os.path.getsize(cache_path):
                    cache_path = p
        if cache_path and os.path.exists(cache_path):
            try:
                _MATMINER_CACHE = pd.read_csv(cache_path)
                print(f"Loaded {len(_MATMINER_CACHE)} cached Matminer features from {cache_path}")
            except Exception:
                _MATMINER_CACHE = pd.DataFrame(columns=[formula_col])
        else:
            _MATMINER_CACHE = pd.DataFrame(columns=[formula_col])
            
    # Find formulas in this dataset that are valid and missing from cache
    unique_formulas = df[formula_col].dropna().unique()
    cached_formulas = set(_MATMINER_CACHE[formula_col].values) if not _MATMINER_CACHE.empty else set()
    missing_formulas = [f for f in unique_formulas if f not in cached_formulas]
    
    if missing_formulas:
        print(f"Featurizing {len(missing_formulas)} new unique formulas with Matminer...")
        df_missing = pd.DataFrame({formula_col: missing_formulas})
        df_missing['composition_obj'] = df_missing[formula_col].apply(get_comp)
        
        valid_mask = df_missing['composition_obj'].notna()
        if valid_mask.any():
            df_missing_valid = df_missing[valid_mask].copy()
            
            ep = ElementProperty.from_preset("magpie")
            df_missing_valid = ep.featurize_dataframe(df_missing_valid, "composition_obj", ignore_errors=True)
            
            vo = ValenceOrbital()
            df_missing_valid = vo.featurize_dataframe(df_missing_valid, "composition_obj", ignore_errors=True)
            
            df_missing_valid = df_missing_valid.drop(columns=['composition_obj'], errors='ignore')
            
            new_cols = [c for c in df_missing_valid.columns if c != formula_col]
            for col in new_cols:
                df_missing[col] = np.nan
            df_missing.loc[valid_mask, new_cols] = df_missing_valid[new_cols].values
            
        df_missing = df_missing.drop(columns=['composition_obj'], errors='ignore')
        
        if _MATMINER_CACHE.empty:
            _MATMINER_CACHE = df_missing
        else:
            _MATMINER_CACHE = pd.concat([_MATMINER_CACHE, df_missing], ignore_index=True)
            
        cache_dir = 'output/featurization'
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, 'matminer_cache.csv')
        try:
            _MATMINER_CACHE.to_csv(cache_path, index=False)
            print(f"Saved updated Matminer features cache ({len(_MATMINER_CACHE)} formulas) to {cache_path}")
        except Exception as e:
            print(f"Could not save Matminer cache to disk: {e}")
            
    # Now merge cached features back
    matminer_cols = [c for c in _MATMINER_CACHE.columns if c != formula_col]
    orig_index = df.index
    df = df.merge(_MATMINER_CACHE[[formula_col] + matminer_cols], on=formula_col, how='left')
    df.index = orig_index
    
    # Fast custom algebraic Ionic Character and Electronegativity Difference features
    # Replaces Matminer IonProperty + ElectronegativityDiff (which hang on metallic alloys)
    print("Computing fast algebraic ionic character and EN difference features...")
    unique_formulas_ionic = df[formula_col].dropna().unique()
    ionic_en_feats_map = {f: compute_ionic_and_en_features(f) for f in unique_formulas_ionic}
    df_ionic_en = pd.DataFrame(
        [ionic_en_feats_map.get(f, _IONIC_EN_DEFAULT) for f in df[formula_col]],
        index=df.index
    )
    df = pd.concat([df, df_ionic_en], axis=1)
    
    # Compute MagpieEX-lite features algebraically after EN/ionic features are merged
    print("Computing lightweight MagpieEX-lite descriptors algebraically...")
    df["mag_cation_anion_ionicity"] = df["avg ionic char"].fillna(0.0) * df["magnetic_3d_fraction"].fillna(0.0)
    df["mag_cation_anion_en_diff"] = df["mean EN difference"].fillna(0.0) * df["magnetic_3d_fraction"].fillna(0.0)
    df["charge_transfer_asymmetry"] = df["std_dev EN difference"].fillna(0.0) * (df["oxygen_fraction"].fillna(0.0) / (df["magnetic_3d_fraction"].fillna(0.0) + 1e-6))

    # Compute boundary-aware magnetic interaction indices
    df = add_boundary_interaction_features(df)
    
    # Compute advanced physical descriptors (GKA rules, Stoner, SOC, Mean-field, Frustration, Goldschmidt)
    df = add_advanced_physical_features(df, formula_col=formula_col)
    
    # Compute innovative DAO physics descriptors (Bethe-Slater ratio, AGK superexchange, Slater-Pauling density, Zener)
    df = add_innovative_dao_physics_features(df)
    
    # Categorical encoding of crystal systems and point groups
    categorical_cols = ['crystal_system', 'point_group']
    for col in categorical_cols:
        if col in df.columns:
            # Dropdown placeholder for missing categories is 'Unknown'
            df[col] = df[col].fillna('Unknown')
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(float)
            df = pd.concat([df, dummies], axis=1)

    # Compute Stevens single-ion anisotropy operator features
    try:
        from src.stevens_operator import compute_stevens_features
        stevens_cols = ['stevens_alpha_J', 'stevens_a20_cf', 'stevens_k1_torque', 'stevens_easy_axis_sign']
        if not all(col in df.columns for col in stevens_cols):
            st_df = compute_stevens_features(df)
            for sc in stevens_cols:
                if sc in st_df.columns:
                    df[sc] = st_df[sc].values
    except Exception as e:
        print(f"Warning: Could not compute Stevens features: {e}")

    # Drop all proxy columns and elemental database lookups to guarantee scientific rigor
    proxy_cols_to_drop = [c for c in df.columns if 'proxy' in c.lower() or 'gsmagmom' in c.lower() or 'gs_magmom' in c.lower()]
    if proxy_cols_to_drop:
        df = df.drop(columns=proxy_cols_to_drop, errors='ignore')
            
    return df


# ------------------------------------------------------------------------------
# Curated Magpie subset for magnetic-property models
# ------------------------------------------------------------------------------
MAGPIE_MAGNETIC_PROPERTY_KEYWORDS = [
    "Number", "MendeleevNumber", "AtomicWeight", "MeltingT",
    "Column", "Row", "CovalentRadius", "Electronegativity",
    "NsValence", "NpValence", "NdValence", "NfValence", "NValence",
    "NsUnfilled", "NpUnfilled", "NdUnfilled", "NfUnfilled", "NUnfilled",
]
MAGPIE_CLEAN_STAT_KEYWORDS = ["mean", "avg_dev", "range"]

MAGPIEEX_LITE_FEATURES = [
    "mag_cation_anion_ionicity",
    "mag_cation_anion_en_diff",
    "charge_transfer_asymmetry",
]

def select_magnetic_relevant_magpie_columns(all_cols, include_proxy=False):
    selected = []
    for c in all_cols:
        if not c.startswith("MagpieData "):
            continue
        if "GSmagmom" in c:
            continue
        has_property = any(key in c for key in MAGPIE_MAGNETIC_PROPERTY_KEYWORDS)
        has_stat = any((f" {stat} " in c) or c.endswith(f" {stat}") for stat in MAGPIE_CLEAN_STAT_KEYWORDS)
        if has_property and has_stat:
            selected.append(c)
    return selected

def build_feature_groups(all_cols):
    """
    Groups feature columns by family.
    """
    element_fractions = [el for el in ELEMENT_LIST if el in all_cols]
    
    composition_basic = [
        "Average_Weight",
        "Average_Electronegativity",
        "Total_Electrons",
        "Num_Atoms",
        "Avg_Atomic_Number",
        "average_period",
        "average_group",
        "Entropy",
        "L2_norm"
    ]
    composition_basic = [c for c in composition_basic if c in all_cols]
    
    magnetic_chemistry = [
        "magnetic_3d_fraction",
        "rare_earth_fraction",
        "transition_metal_fraction",
        "oxygen_fraction",
        "chalcogen_fraction",
        "pnictogen_fraction",
        "halogen_fraction",
        "num_magnetic_3d_elements",
        "num_rare_earth_elements",
        "num_transition_metals",
        "transition_metal_oxide_flag",
        "magnetic_nonmagnetic_ratio",
        # Category Interactions (Only Keep Non-Redundant Generic Ones)
        "rare_earth_TM_interaction",
        "magnetic_3d_chalcogen_interaction",
        "magnetic_3d_pnictogen_interaction",
        # Exchange Interactions
        "transition_metal_halide_flag",
        "transition_metal_chalcogenide_flag",
        "rare_earth_transition_metal_flag",
        "Fe_Co_alloy_flag",
        # D-Electron Physics (AFM/NM Distinction)
        "weighted_d_electron_count",
        "weighted_f_electron_count",
        "d_band_filling_fraction",
        "unpaired_electron_estimate",
        "magnetic_ion_concentration",
        # Paper-inspired magnetic-property descriptors
        "elemental_fm_fraction",
        "elemental_afm_fraction",
        "strong_fm_3d_fraction",
        "strong_afm_3d_fraction",
        "rare_earth_fm_fraction",
        "rare_earth_afm_fraction",
        "high_curie_element_fraction",
        "elemental_fm_tc_prior",
        "elemental_afm_tn_prior",
        "elemental_ordering_temp_sum_prior",
        "max_elemental_ordering_temp",
        "elemental_fm_minus_afm_temp_prior",
        "fm_to_afm_element_ratio",
        "high_curie_to_magnetic_fraction"
    ]
    magnetic_chemistry = [c for c in magnetic_chemistry if c in all_cols]
    
    dao_crystallography = [
        "dao_density",
        "dao_volume_per_atom",
        "dao_ehull",
        "dao_spacegroup_num",
        "dao_is_uniaxial",
        "dao_c_over_a",
        "dao_packing_fraction",
        "dao_coordination_num",
        "dao_min_bond_dist",
        "dao_is_centrosymmetric",
        "dao_moment_density",
        "dao_magnetic_concentration",
        "dao_exchange_connectivity",
        "dao_axial_distortion",
        "dao_stability_weight",
    ]
    dao_crystallography = [c for c in dao_crystallography if c in all_cols]

    mp_symmetry = [
        "spacegroup_number",
        "is_centrosymmetric",
        "is_noncentrosymmetric",
    ]
    mp_symmetry = [c for c in mp_symmetry if c in all_cols]
    
    crystal_system_dummies = [c for c in all_cols if c.startswith("crystal_system_")]
    point_group_dummies = [c for c in all_cols if c.startswith("point_group_")]
    
    # Curated Magpie descriptors: only magnetism-relevant statistics/properties.
    # This reduces generic descriptor noise in F3 and feature-space visualizations.
    magpie_cols = select_magnetic_relevant_magpie_columns(all_cols, include_proxy=False)
    magpie_all_nonproxy = [c for c in all_cols if c.startswith("MagpieData ") and "GSmagmom" not in c]
    
    valence_cols = [c for c in all_cols if c.startswith("ValenceOrbital ")]

    oxidation_states = [
        'mean_oxi_state_magnetic', 
        'max_oxi_state', 
        'min_oxi_state', 
        'oxi_state_range', 
        'mixed_valence_flag', 
        'charge_balance_success'
    ]
    oxidation_states = [c for c in oxidation_states if c in all_cols]
    
    formula_family = [
        'is_binary', 
        'is_ternary', 
        'is_quaternary', 
        'AB_like', 
        'AB2_like', 
        'ABO3_like', 
        'AB2O4_like', 
        'A2BO4_like', 
        'AFeO3_like', 
        'MFe2O4_like', 
        'hexaferrite_like', 
        'heusler_X2YZ_like', 
        'half_heusler_XYZ_like'
    ]
    formula_family = [c for c in formula_family if c in all_cols]
    
    ionic_and_electronegativity = [
        'compound possible', 
        'max ionic char', 
        'avg ionic char', 
        'minimum EN difference', 
        'maximum EN difference', 
        'range EN difference', 
        'mean EN difference', 
        'std_dev EN difference'
    ]
    ionic_and_electronegativity = [c for c in ionic_and_electronegativity if c in all_cols]

    magpieex_lite = [c for c in MAGPIEEX_LITE_FEATURES if c in all_cols]

    boundary_interaction_indices = [
        "afm_superexchange_index",
        "afm_nm_boundary_index",
        "fm_metallic_index",
        "nm_closed_shell_index",
        "rare_earth_ordering_index",
        "charge_transfer_exchange_index",
    ]
    boundary_interaction_indices = [c for c in boundary_interaction_indices if c in all_cols]

    gka_and_exchange = [
        "d_shell_half_filling_index",
        "double_exchange_candidate",
        "weighted_stoner_parameter",
        "near_stoner_threshold_flag",
        "octahedral_buckling_angle_deg",
        "superexchange_transfer_factor"
    ]
    gka_and_exchange = [c for c in gka_and_exchange if c in all_cols]

    anisotropy_and_soc = [
        "weighted_soc_constant",
        "uniaxial_symmetry_flag",
        "anisotropy_figure_of_merit",
        "re_anisotropy_sign_index",
        "stevens_alpha_J",
        "stevens_a20_cf",
        "stevens_k1_torque",
        "stevens_easy_axis_sign"
    ]
    anisotropy_and_soc = [c for c in anisotropy_and_soc if c in all_cols]

    mean_field_priors = [
        "mean_field_spin_factor",
        "weighted_de_gennes_factor",
        "valence_electron_concentration",
        "slater_pauling_peak_distance",
    ]
    mean_field_priors = [c for c in mean_field_priors if c in all_cols]

    frustration_and_competition = [
        "frustrated_lattice_flag",
        "triangular_lattice_flag",
        "exchange_competition_index",
        "exchange_balance_entropy",
    ]
    frustration_and_competition = [c for c in frustration_and_competition if c in all_cols]

    structural_tolerance = [
        "goldschmidt_tolerance_factor"
    ]
    structural_tolerance = [c for c in structural_tolerance if c in all_cols]
    
    mace_quantum = [
        "mace_energy_per_atom",
        "mace_axial_stress",
        "mace_hydrostatic_pressure",
        "mace_von_mises_stress",
        "dft_jarvis_magmom",
        "dft_jarvis_ehull",
        "dft_jarvis_bandgap",
        "mace_matched"
    ]
    mace_quantum = [c for c in mace_quantum if c in all_cols]

    return {
        "composition_basic": composition_basic,
        "element_fractions": element_fractions,
        "magpie": magpie_cols,
        "magpie_magnetic_relevant": magpie_cols,
        "magpie_all_nonproxy": magpie_all_nonproxy,
        "valence": valence_cols,
        "magnetic_chemistry": magnetic_chemistry,
        "mp_symmetry": mp_symmetry,
        "crystal_system_dummies": crystal_system_dummies,
        "point_group_dummies": point_group_dummies,
        "oxidation_states": oxidation_states,
        "formula_family": formula_family,
        "ionic_and_electronegativity": ionic_and_electronegativity,
        "magpieex_lite": magpieex_lite,
        "magpieex_magnetic": magpieex_lite,
        "boundary_interaction_indices": boundary_interaction_indices,
        "gka_and_exchange": gka_and_exchange,
        "anisotropy_and_soc": anisotropy_and_soc,
        "mean_field_priors": mean_field_priors,
        "frustration_and_competition": frustration_and_competition,
        "structural_tolerance": structural_tolerance,
        "mace_quantum": mace_quantum,
        "dao_crystallography": dao_crystallography
    }

# ==============================================================================
#        TARGET-SPECIFIC, BOUNDARY-AWARE LIGHTWEIGHT FEATURE SELECTION
# ==============================================================================

SELECTION_TARGET_EXCLUDE = [
    'Type', 'Mean_TC_K', 'Mean_TN_K', 'K1_J_m3', 'Coercivity_A_m',
    'Saturation_Magnetization_emu_g', 'Hard_Magnet', 'reduced_formula',
    'formula', 'material_id', 'match_confidence', 'match_score', 'match_reason',
    'Clean_Chemical_Formula', 'Normalized_Composition', 'Example_Formula'
]

PAIRWISE_BOUNDARIES = {
    'fm_afm': (0, 1),
    'fm_nm': (0, 2),
    'afm_nm': (1, 2),
}
BOUNDARY_SCORE_WEIGHTS = {'global': 0.30, 'fm_afm': 0.25, 'fm_nm': 0.15, 'afm_nm': 0.30}

MUST_KEEP_F3 = [
    # Compact non-negotiable magnetic descriptors grounded in physics.
    'magnetic_3d_fraction', 'rare_earth_fraction', 'oxygen_fraction',
    'magnetic_ion_concentration', 'unpaired_electron_estimate',
    'd_band_filling_fraction', 'weighted_d_electron_count', 'weighted_f_electron_count',
    'weighted_soc_constant',
    'elemental_fm_fraction', 'elemental_afm_fraction',
    'strong_fm_3d_fraction', 'strong_afm_3d_fraction', 'high_curie_element_fraction',
    'elemental_fm_tc_prior', 'elemental_afm_tn_prior',
    'mag_cation_anion_ionicity', 'charge_transfer_asymmetry',
    'afm_superexchange_index', 'fm_metallic_index',
    'nm_closed_shell_index', 'afm_nm_boundary_index',
    'd_shell_half_filling_index', 'weighted_stoner_parameter',
    'anisotropy_figure_of_merit', 'mean_field_spin_factor',
    'exchange_competition_index',
    'weighted_de_gennes_factor', 'slater_pauling_peak_distance', 'superexchange_transfer_factor',
    'valence_electron_concentration',
    'mace_energy_per_atom', 'mace_axial_stress', 'mace_hydrostatic_pressure',
    # DAO 3D Crystallographic Anchors
    'dao_min_bond_dist', 'dao_spacegroup_num', 'dao_density', 'dao_coordination_num',
    'dao_is_centrosymmetric', 'dao_exchange_connectivity', 'dao_moment_density'
]
MUST_KEEP_TC = [
    'elemental_fm_fraction', 'strong_fm_3d_fraction', 'high_curie_element_fraction', 'elemental_fm_tc_prior',
    'max_elemental_ordering_temp', 'magnetic_ion_concentration',
    'unpaired_electron_estimate', 'fm_metallic_index', 'rare_earth_ordering_index',
    'Avg_Atomic_Number', 'Average_Weight',
    'weighted_stoner_parameter', 'mean_field_spin_factor', 'anisotropy_figure_of_merit',
    'weighted_de_gennes_factor', 'slater_pauling_peak_distance', 'valence_electron_concentration',
    'weighted_soc_constant',
    'mace_hydrostatic_pressure', 'mace_energy_per_atom',
    # DAO 3D Descriptors
    'dao_volume_per_atom', 'dao_coordination_num', 'dao_min_bond_dist', 'dao_density',
    'dao_ehull', 'dao_exchange_connectivity', 'dao_moment_density'
]
MUST_KEEP_TN = [
    'elemental_afm_fraction', 'strong_afm_3d_fraction', 'elemental_afm_tn_prior',
    'oxygen_fraction', 'afm_superexchange_index',
    'afm_nm_boundary_index', 'mag_cation_anion_ionicity', 'mag_cation_anion_en_diff',
    'charge_transfer_asymmetry', 'd_band_filling_fraction', 'unpaired_electron_estimate',
    'd_shell_half_filling_index', 'goldschmidt_tolerance_factor',
    'weighted_de_gennes_factor', 'superexchange_transfer_factor', 'octahedral_buckling_angle_deg',
    'weighted_soc_constant', 'mean_field_spin_factor', 'exchange_competition_index',
    'mace_hydrostatic_pressure', 'mace_energy_per_atom',
    # DAO 3D Descriptors
    'dao_min_bond_dist', 'dao_coordination_num', 'dao_packing_fraction', 'dao_density',
    'dao_volume_per_atom', 'dao_ehull', 'dao_exchange_connectivity'
]


def _ordered_unique(items):
    seen, out = set(), []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _normalize_score(arr):
    arr = np.asarray(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _safe_json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


def _safe_group_list(groups, names):
    out = []
    for name in names:
        out.extend(groups.get(name, []))
    return _ordered_unique(out)


def build_feature_tiers(feature_groups, all_columns):
    """Return F1/F3/F5 candidate pools for target-specific selection."""
    all_columns = set(all_columns)
    f1 = _safe_group_list(feature_groups, ['composition_core', 'composition_basic', 'elemental_fraction', 'element_fractions'])
    f3 = _safe_group_list(feature_groups, [
        'composition_core', 'composition_basic', 'elemental_fraction', 'element_fractions',
        'elemental_statistics', 'magpie', 'valence_electron_structure', 'valence',
        'magnetic_element_priors', 'magnetic_exchange_chemistry', 'magnetic_chemistry',
        'magpieex_cation_anion', 'magpieex_lite', 'magpieex_magnetic', 'boundary_interaction_indices',
        'gka_and_exchange', 'anisotropy_and_soc', 'mean_field_priors', 'frustration_and_competition',
        'structural_tolerance', 'oxidation_charge_balance', 'oxidation_states',
        'ionic_bonding_general', 'ionic_and_electronegativity',
        'formula_family_motifs', 'formula_family'
    ])
    f5 = _ordered_unique(f3)
    return {
        'F1': [c for c in f1 if c in all_columns],
        'F3': [c for c in f3 if c in all_columns],
        'F5': [c for c in f5 if c in all_columns],
    }


def get_clean_formula_feature_pool(feature_groups, all_columns):
    return build_feature_tiers(feature_groups, all_columns).get('F3', [])


def _feature_family_map(groups):
    preferred = [
        'composition_core', 'composition_basic', 'elemental_fraction', 'element_fractions',
        'elemental_statistics', 'magpie', 'valence_electron_structure', 'valence',
        'magnetic_element_priors', 'magnetic_exchange_chemistry', 'magnetic_chemistry',
        'magpieex_cation_anion', 'magpieex_lite', 'magpieex_magnetic', 'boundary_interaction_indices',
        'gka_and_exchange', 'anisotropy_and_soc', 'mean_field_priors', 'frustration_and_competition',
        'structural_tolerance', 'oxidation_charge_balance', 'oxidation_states', 'ionic_bonding_general',
        'ionic_and_electronegativity', 'formula_family_motifs', 'formula_family',
    ]
    fmap = {}
    for name in preferred:
        for feat in groups.get(name, []):
            fmap.setdefault(feat, name)
    return fmap


def _prepare_numeric_feature_matrix(df, feature_cols):
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    std = X.std(numeric_only=True)
    X = X[std[std > 1e-10].index.tolist()]
    sparse_drop = []
    n = max(len(X), 1)
    for c in X.columns:
        vals = X[c].fillna(0.0)
        if vals.nunique(dropna=False) <= 2:
            pos = float((vals > 0).sum()) / n
            if pos < 0.005 or pos > 0.995:
                sparse_drop.append(c)
    return X.drop(columns=sparse_drop, errors='ignore')


def _stratified_subsample_indices(y, max_samples=15000, random_state=42):
    y = np.asarray(y)
    n = len(y)
    if n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    selected = []
    classes, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(classes, counts):
        idx = np.where(y == cls)[0]
        take = max(10, int(round(max_samples * cnt / n)))
        take = min(take, len(idx))
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    selected = np.array(sorted(set(selected)), dtype=int)
    if len(selected) > max_samples:
        selected = rng.choice(selected, size=max_samples, replace=False)
    return np.sort(selected)


def _random_subsample_indices(n, max_samples=15000, random_state=42):
    if n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(np.arange(n), size=max_samples, replace=False))


def _fisher_class_separation_scores(X, y):
    y = np.asarray(y)
    Xv = np.asarray(X, dtype=float)
    overall = np.nanmean(Xv, axis=0)
    num = np.zeros(Xv.shape[1])
    den = np.zeros(Xv.shape[1])
    for cls in np.unique(y):
        mask = y == cls
        if mask.sum() < 2:
            continue
        Xc = Xv[mask]
        mu = np.nanmean(Xc, axis=0)
        var = np.nanvar(Xc, axis=0)
        num += mask.sum() * (mu - overall) ** 2
        den += mask.sum() * var
    return np.nan_to_num(num / (den + 1e-12), nan=0.0, posinf=0.0, neginf=0.0)


def _abs_pearson_scores(X, y):
    Xv = np.asarray(X, dtype=float)
    yv = np.asarray(y, dtype=float)
    yv = yv - np.nanmean(yv)
    y_std = np.nanstd(yv) + 1e-12
    Xc = Xv - np.nanmean(Xv, axis=0)
    x_std = np.nanstd(Xc, axis=0) + 1e-12
    corr = np.nanmean(Xc * yv[:, None], axis=0) / (x_std * y_std)
    return np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)


def _family_quota_limits(top_k, task="classification"):
    """Return family quota limits used during final feature selection.

    The goal is not to enforce a perfectly equal representation, but to stop
    one broad family, especially Magpie statistics or element fractions, from
    dominating the selected set and producing noisy manifolds.
    """
    if task == "regression":
        quota_fracs = {
            'magnetic_chemistry': 0.38,
            'dao_crystallography': 0.16,
            'magpie': 0.20,
            'magpie_magnetic_relevant': 0.20,
            'element_fractions': 0.16,
            'valence': 0.16,
            'composition_basic': 0.12,
            'ionic_and_electronegativity': 0.10,
            'magpieex_lite': 0.10,
            'magpieex_magnetic': 0.10,
            'boundary_interaction_indices': 0.14,
            'oxidation_states': 0.08,
            'formula_family': 0.08,
        }
    else:
        quota_fracs = {
            'magnetic_chemistry': 0.36,
            'dao_crystallography': 0.16,
            'magpie': 0.20,
            'magpie_magnetic_relevant': 0.20,
            'element_fractions': 0.16,
            'valence': 0.16,
            'composition_basic': 0.12,
            'ionic_and_electronegativity': 0.10,
            'magpieex_lite': 0.10,
            'magpieex_magnetic': 0.10,
            'boundary_interaction_indices': 0.14,
            'oxidation_states': 0.08,
            'formula_family': 0.08,
        }
    return {fam: max(2, int(round(frac * top_k))) for fam, frac in quota_fracs.items()}


def _family_aware_correlation_prune(X, ordered_features, score_dict, feature_to_family=None,
                                    same_family_threshold=0.90, cross_family_threshold=0.96,
                                    top_k=100, task="classification", must_keep=None,
                                    candidate_pool=None):
    """Select features with family quotas, must-keep support, and correlation pruning.

    Must-keep features are inserted first, but they are still checked for
    availability and near-duplicate correlation among themselves. Remaining
    slots are filled by ranked features while respecting family quotas. This
    avoids the earlier issue where must-keep insertion could bypass the quota
    logic and overcrowd the selected set.
    """
    if not ordered_features:
        ordered_features = []
    feature_to_family = feature_to_family or {}
    candidate_pool = set(candidate_pool or X.columns.tolist())
    must_keep = [f for f in (must_keep or []) if f in candidate_pool and f in X.columns]
    ordered_features = [f for f in ordered_features if f in X.columns and f in candidate_pool]
    ranked = _ordered_unique(must_keep + ordered_features)
    if not ranked:
        return []

    corr = X[ranked].corr().abs()
    quota_limits = _family_quota_limits(top_k, task=task)
    selected = []
    family_counts = {}

    def can_add(feat, enforce_quota=True):
        fam = feature_to_family.get(feat, 'unknown')
        if enforce_quota and fam in quota_limits and family_counts.get(fam, 0) >= quota_limits[fam]:
            return False
        for s in selected:
            threshold = same_family_threshold if feature_to_family.get(s) == fam else cross_family_threshold
            val = corr.loc[s, feat]
            if not pd.isna(val) and val > threshold:
                return False
        return True

    # Add must-keep first, but do not exceed top_k. Quota is relaxed for these
    # physically essential descriptors only.
    for feat in must_keep:
        if len(selected) >= top_k:
            break
        if can_add(feat, enforce_quota=False):
            selected.append(feat)
            fam = feature_to_family.get(feat, 'unknown')
            family_counts[fam] = family_counts.get(fam, 0) + 1

    # Fill remaining slots using ranked target-aware features with quotas.
    for feat in ordered_features:
        if len(selected) >= top_k:
            break
        if feat in selected:
            continue
        if can_add(feat, enforce_quota=True):
            selected.append(feat)
            fam = feature_to_family.get(feat, 'unknown')
            family_counts[fam] = family_counts.get(fam, 0) + 1

    # If quotas were too strict and left empty slots, fill remaining slots while
    # still respecting correlation pruning.
    if len(selected) < top_k:
        for feat in ordered_features:
            if len(selected) >= top_k:
                break
            if feat in selected:
                continue
            if can_add(feat, enforce_quota=False):
                selected.append(feat)
                fam = feature_to_family.get(feat, 'unknown')
                family_counts[fam] = family_counts.get(fam, 0) + 1
    return selected[:top_k]

def _apply_must_keep(selected, candidate_pool, must_keep, top_k):
    forced = [f for f in must_keep if f in candidate_pool]
    return _ordered_unique(forced + list(selected))[:top_k]


def _classification_ranking(df, target_col, candidate_features, label='global', n_estimators=120,
                            max_samples=15000, random_state=42):
    if target_col not in df.columns:
        return pd.DataFrame()
    df_sel = df.dropna(subset=[target_col]).copy()
    if len(df_sel) < 20 or df_sel[target_col].nunique() < 2:
        return pd.DataFrame()
    y = df_sel[target_col].astype(int).values
    X = _prepare_numeric_feature_matrix(df_sel, candidate_features)
    if X.shape[1] == 0:
        return pd.DataFrame()
    idx = _stratified_subsample_indices(y, max_samples=max_samples, random_state=random_state)
    X_sub = X.iloc[idx]
    y_sub = y[idx]

    # Drop columns that become constant after subsampling. This avoids ANOVA
    # warnings and keeps F1 selection truly target-relevant.
    sub_std = X_sub.std(numeric_only=True)
    keep_cols = sub_std[sub_std > 1e-10].index.tolist()
    X_sub = X_sub[keep_cols]
    if X_sub.shape[1] == 0:
        return pd.DataFrame()

    from sklearn.feature_selection import f_classif, mutual_info_classif
    from sklearn.ensemble import ExtraTreesClassifier
    try:
        f_scores, _ = f_classif(X_sub, y_sub)
    except Exception:
        f_scores = np.zeros(X_sub.shape[1])
    try:
        mi_scores = mutual_info_classif(X_sub, y_sub, discrete_features='auto', random_state=random_state)
    except Exception:
        mi_scores = np.zeros(X_sub.shape[1])
    fisher_scores = _fisher_class_separation_scores(X_sub.values, y_sub)
    et = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=12, min_samples_leaf=3,
                              max_features='sqrt', class_weight='balanced', random_state=random_state, n_jobs=-1)
    try:
        et.fit(X_sub, y_sub)
        tree_scores = et.feature_importances_
    except Exception:
        tree_scores = np.zeros(X_sub.shape[1])
    hybrid = (0.20 * _normalize_score(f_scores) + 0.25 * _normalize_score(mi_scores) +
              0.30 * _normalize_score(tree_scores) + 0.25 * _normalize_score(fisher_scores))
    ranking = pd.DataFrame({
        'feature': X_sub.columns,
        f'{label}_linear_f_score': np.nan_to_num(f_scores),
        f'{label}_mutual_info': np.nan_to_num(mi_scores),
        f'{label}_extratrees_importance': np.nan_to_num(tree_scores),
        f'{label}_fisher_score': np.nan_to_num(fisher_scores),
        f'{label}_hybrid_score': hybrid,
    }).sort_values(f'{label}_hybrid_score', ascending=False).reset_index(drop=True)
    ranking[f'{label}_rank'] = np.arange(1, len(ranking) + 1)
    return ranking


def boundary_aware_classification_feature_selection(df, candidate_features, feature_groups=None, target_col='Type',
                                                    top_k=100, same_family_corr=0.95, cross_family_corr=0.99,
                                                    n_estimators=120, max_samples=15000, random_state=42,
                                                    must_keep=None):
    if target_col not in df.columns:
        return [], pd.DataFrame(), {}
    candidate_features = [c for c in _ordered_unique(candidate_features) if c in df.columns]
    rankings = {
        'global': _classification_ranking(df, target_col, candidate_features, 'global', n_estimators, max_samples, random_state)
    }
    for label, (a, b) in PAIRWISE_BOUNDARIES.items():
        sub = df[df[target_col].isin([a, b])].copy()
        rankings[label] = _classification_ranking(sub, target_col, candidate_features, label,
                                                  n_estimators, max_samples, random_state + len(label))
    all_features = set()
    for r in rankings.values():
        if not r.empty:
            all_features.update(r['feature'].tolist())
    if not all_features:
        return [], pd.DataFrame(), rankings
    score_df = pd.DataFrame({'feature': sorted(all_features)})
    final_score = np.zeros(len(score_df))
    component_cols = []
    for label, weight in BOUNDARY_SCORE_WEIGHTS.items():
        r = rankings.get(label, pd.DataFrame())
        col = f'{label}_hybrid_score'
        out_col = f'{label}_score_norm'
        if not r.empty and col in r.columns:
            tmp = r[['feature', col]].copy()
            tmp[out_col] = _normalize_score(tmp[col].values)
            score_df = score_df.merge(tmp[['feature', out_col]], on='feature', how='left')
            score_df[out_col] = score_df[out_col].fillna(0.0)
        else:
            score_df[out_col] = 0.0
        final_score += weight * score_df[out_col].values
        component_cols.append(out_col)
    score_df['boundary_weighted_score'] = final_score
    score_df['selected_by_n_boundaries'] = (score_df[component_cols] > 0).sum(axis=1)
    score_df['final_score'] = score_df['boundary_weighted_score'] * (1.0 + 0.04 * score_df['selected_by_n_boundaries'])
    score_df = score_df.sort_values('final_score', ascending=False).reset_index(drop=True)

    X_all = _prepare_numeric_feature_matrix(df.dropna(subset=[target_col]), candidate_features)
    preselected = [f for f in score_df['feature'].head(min(max(top_k * 3, top_k), len(score_df))).tolist() if f in X_all.columns]
    score_dict = dict(zip(score_df['feature'], score_df['final_score']))
    selected = _family_aware_correlation_prune(
        X_all, preselected, score_dict, _feature_family_map(feature_groups or {}),
        same_family_threshold=same_family_corr, cross_family_threshold=cross_family_corr,
        top_k=top_k, task="classification", must_keep=must_keep or [],
        candidate_pool=candidate_features
    )
    score_df['selected'] = score_df['feature'].isin(selected)
    score_df['rank'] = np.arange(1, len(score_df) + 1)
    return selected, score_df, rankings


def _regression_ranking(df, target_col, candidate_features, n_estimators=160, max_samples=15000, random_state=42):
    if target_col not in df.columns:
        return pd.DataFrame()
    df_sel = df.dropna(subset=[target_col]).copy()
    df_sel[target_col] = pd.to_numeric(df_sel[target_col], errors='coerce')
    df_sel = df_sel[df_sel[target_col] > 0].copy()
    if len(df_sel) < 50:
        return pd.DataFrame()
    y = np.log1p(df_sel[target_col].values.astype(float))
    X = _prepare_numeric_feature_matrix(df_sel, candidate_features)
    if X.shape[1] == 0:
        return pd.DataFrame()
    idx = _random_subsample_indices(len(X), max_samples=max_samples, random_state=random_state)
    X_sub, y_sub = X.iloc[idx], y[idx]

    # Drop columns that become constant after subsampling.
    sub_std = X_sub.std(numeric_only=True)
    keep_cols = sub_std[sub_std > 1e-10].index.tolist()
    X_sub = X_sub[keep_cols]
    if X_sub.shape[1] == 0:
        return pd.DataFrame()

    from sklearn.feature_selection import f_regression, mutual_info_regression
    from sklearn.ensemble import ExtraTreesRegressor
    try:
        f_scores, _ = f_regression(X_sub, y_sub)
    except Exception:
        f_scores = np.zeros(X_sub.shape[1])
    try:
        mi_scores = mutual_info_regression(X_sub, y_sub, random_state=random_state)
    except Exception:
        mi_scores = np.zeros(X_sub.shape[1])
    pearson_scores = _abs_pearson_scores(X_sub.values, y_sub)
    et = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=14, min_samples_leaf=3,
                             max_features='sqrt', random_state=random_state, n_jobs=-1)
    try:
        et.fit(X_sub, y_sub)
        tree_scores = et.feature_importances_
    except Exception:
        tree_scores = np.zeros(X_sub.shape[1])
    hybrid = (0.25 * _normalize_score(f_scores) + 0.25 * _normalize_score(mi_scores) +
              0.35 * _normalize_score(tree_scores) + 0.15 * _normalize_score(pearson_scores))
    ranking = pd.DataFrame({
        'feature': X_sub.columns,
        'f_regression_score': np.nan_to_num(f_scores),
        'mutual_info_regression': np.nan_to_num(mi_scores),
        'extratrees_regression_importance': np.nan_to_num(tree_scores),
        'abs_pearson_log_target': pearson_scores,
        'hybrid_score': hybrid,
    }).sort_values('hybrid_score', ascending=False).reset_index(drop=True)
    ranking['rank'] = np.arange(1, len(ranking) + 1)
    return ranking


def transition_temperature_feature_selection(df, target_col, candidate_features, feature_groups=None, top_k=100,
                                             phase_class=None, type_col='Type', same_family_corr=0.95,
                                             cross_family_corr=0.99, n_estimators=160, max_samples=15000,
                                             random_state=42, must_keep=None):
    candidate_features = [c for c in _ordered_unique(candidate_features) if c in df.columns]
    df_work = df.copy()
    if phase_class is not None and type_col in df_work.columns:
        df_work = df_work[df_work[type_col] == phase_class].copy()
    ranking = _regression_ranking(df_work, target_col, candidate_features, n_estimators, max_samples, random_state)
    if ranking.empty:
        return [], ranking
    df_nonnull = df_work.dropna(subset=[target_col]).copy()
    df_nonnull[target_col] = pd.to_numeric(df_nonnull[target_col], errors='coerce')
    df_nonnull = df_nonnull[df_nonnull[target_col] > 0].copy()
    X_all = _prepare_numeric_feature_matrix(df_nonnull, candidate_features)
    preselected = [f for f in ranking['feature'].head(min(max(top_k * 3, top_k), len(ranking))).tolist() if f in X_all.columns]
    score_dict = dict(zip(ranking['feature'], ranking['hybrid_score']))
    selected = _family_aware_correlation_prune(
        X_all, preselected, score_dict, _feature_family_map(feature_groups or {}),
        same_family_threshold=same_family_corr, cross_family_threshold=cross_family_corr,
        top_k=top_k, task="regression", must_keep=must_keep or [],
        candidate_pool=candidate_features
    )
    ranking['selected'] = ranking['feature'].isin(selected)
    return selected, ranking


def plot_refined_tsne_for_tier(df_feat, selected_features, tier_name, plot_dir='plots', target_col='Type',
                               max_samples=1800, random_state=42):
    if target_col not in df_feat.columns or not selected_features:
        return None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.preprocessing import StandardScaler
        from sklearn.manifold import TSNE
    except Exception as exc:
        print(f"Skipping t-SNE for {tier_name}: {exc}")
        return None
    os.makedirs(plot_dir, exist_ok=True)
    df = df_feat.dropna(subset=[target_col]).copy()
    y = df[target_col].astype(int).values
    X = _prepare_numeric_feature_matrix(df, selected_features)
    if X.shape[1] < 2 or len(X) < 20:
        return None
    idx = _stratified_subsample_indices(y, max_samples=max_samples, random_state=random_state)
    X, y = X.iloc[idx], y[idx]
    X_scaled = StandardScaler().fit_transform(X)
    
    perplexity = min(50, max(10, (len(X_scaled) - 1) // 4))
    try:
        Z = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca',
                 random_state=random_state, max_iter=1500).fit_transform(X_scaled)
    except TypeError:
        Z = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca',
                 random_state=random_state, n_iter=1500).fit_transform(X_scaled)
                 
    # Standardize the output embedding to the gold-standard -3 to 3 bounds
    Z = StandardScaler().fit_transform(Z) * 1.25
    
    colors = {0: '#FF0055', 1: '#0055FF', 2: '#00C853'}
    labels = {0: 'FM', 1: 'AFM', 2: 'NM'}
    
    fig, ax = plt.subplots(figsize=(10.5, 8.5), facecolor='white')
    ax.set_facecolor('white')
    
    # 1. Plot beautiful double-layer density contours for each class
    for cls in [0, 1, 2]:
        mask = y == cls
        if mask.sum() < 5:
            continue
        try:
            sns.kdeplot(
                x=Z[mask, 0], y=Z[mask, 1], ax=ax,
                color=colors[cls], fill=True, alpha=0.08, levels=4, thresh=0.08
            )
            sns.kdeplot(
                x=Z[mask, 0], y=Z[mask, 1], ax=ax,
                color=colors[cls], alpha=0.55, levels=3, linewidths=2.0, thresh=0.12
            )
        except Exception:
            pass
            
    # 2. Scatter the individual points on top of the contours
    for cls in [0, 1, 2]:
        mask = y == cls
        if mask.sum() > 0:
            ax.scatter(
                Z[mask, 0], Z[mask, 1], s=65, c=colors[cls], label=labels[cls],
                edgecolors='white', linewidths=0.6, alpha=0.88, zorder=3
            )
            
    ax.set_title(f'{tier_name} Refined Feature Space after Target-Specific Selection', fontsize=20, weight='bold', pad=15)
    ax.set_xlabel('t-SNE 1', fontsize=22, weight='bold')
    ax.set_ylabel('t-SNE 2', fontsize=22, weight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', labelsize=14, width=1.5)
    plt.tight_layout()
    out_path = os.path.join(plot_dir, f'feature_space_tsne_{tier_name}_refined.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


# Separate compact F1 selector sizes. F1 is intentionally much smaller because it
# contains only composition statistics and elemental fractions. These values are
# selected directly from target relevance, not by manual inclusion.
F1_TOP_K_VALUES = (20, 30, 40)
F1_CLASSIFICATION_TOP_K_VALUES = (20, 30, 40)
F1_REGRESSION_TOP_K_VALUES = (20, 30, 40)


def run_target_specific_feature_selection_outputs(df_feat, groups, output_dir='output/featurization', plot_dir='plots',
                                                  top_k_values=(40, 60, 80), f1_top_k_values=F1_TOP_K_VALUES,
                                                  same_family_corr=0.95,
                                                  cross_family_corr=0.99, n_estimators_class=120,
                                                  n_estimators_reg=160, make_tsne=False):
    os.makedirs(output_dir, exist_ok=True)
    tiers = build_feature_tiers(groups, list(df_feat.columns))
    summary = {
        'top_k_values': list(top_k_values),
        'f1_top_k_values': list(f1_top_k_values),
        'same_family_corr': float(same_family_corr),
        'cross_family_corr': float(cross_family_corr),
        'n_estimators_classification': int(n_estimators_class),
        'n_estimators_regression': int(n_estimators_reg),
        'tiers': {k: len(v) for k, v in tiers.items()},
    }
    tier_selected_class, tier_selected_tc, tier_selected_tn = {}, {}, {}
    for tier_name, pool in tiers.items():
        tier_dir = os.path.join(output_dir, f'target_specific_{tier_name}')
        os.makedirs(tier_dir, exist_ok=True)
        tier_top_k_values = tuple(f1_top_k_values) if tier_name == 'F1' else tuple(top_k_values)
        max_k = max(tier_top_k_values)
        must_keep = MUST_KEEP_F3 if tier_name in {'F3', 'F5'} else []
        if 'Type' in df_feat.columns:
            print(f"Running boundary-aware classification selection for {tier_name} ({len(pool)} candidates)...")
            selected_cls, ranking_cls, boundary_rankings = boundary_aware_classification_feature_selection(
                df_feat, pool, feature_groups=groups, target_col='Type', top_k=max_k,
                same_family_corr=same_family_corr, cross_family_corr=cross_family_corr,
                n_estimators=n_estimators_class, must_keep=must_keep
            )
            ranking_cls.to_csv(os.path.join(tier_dir, f'{tier_name}_classification_boundary_weighted_ranking.csv'), index=False)
            for label, r in boundary_rankings.items():
                if not r.empty:
                    r.to_csv(os.path.join(tier_dir, f'{tier_name}_{label}_component_ranking.csv'), index=False)
            for k in tier_top_k_values:
                _safe_json_dump(selected_cls[:min(k, len(selected_cls))], os.path.join(tier_dir, f'selected_features_{tier_name}_classification_refined_clean{k}.json'))
            tier_selected_class[tier_name] = selected_cls
            summary[f'{tier_name}_classification_selected_count'] = int(len(selected_cls))
            summary[f'{tier_name}_classification_top_features'] = selected_cls[:25]
            if make_tsne:
                out_tsne = plot_refined_tsne_for_tier(df_feat, selected_cls[:min(100, len(selected_cls))], tier_name, plot_dir=plot_dir)
                if out_tsne:
                    summary[f'{tier_name}_tsne_plot'] = out_tsne
        if 'Mean_TC_K' in df_feat.columns:
            print(f"Running TC regression feature selection for {tier_name}...")
            selected_tc, ranking_tc = transition_temperature_feature_selection(
                df_feat, 'Mean_TC_K', pool, feature_groups=groups, top_k=max_k, phase_class=0,
                same_family_corr=same_family_corr, cross_family_corr=cross_family_corr,
                n_estimators=n_estimators_reg, random_state=52,
                must_keep=MUST_KEEP_TC if tier_name in {'F3', 'F5'} else []
            )
            if not ranking_tc.empty:
                ranking_tc.to_csv(os.path.join(tier_dir, f'{tier_name}_TC_regression_ranking.csv'), index=False)
            for k in tier_top_k_values:
                _safe_json_dump(selected_tc[:min(k, len(selected_tc))], os.path.join(tier_dir, f'selected_features_{tier_name}_TC_clean{k}.json'))
            tier_selected_tc[tier_name] = selected_tc
            summary[f'{tier_name}_TC_selected_count'] = int(len(selected_tc))
        if 'Mean_TN_K' in df_feat.columns:
            print(f"Running TN regression feature selection for {tier_name}...")
            selected_tn, ranking_tn = transition_temperature_feature_selection(
                df_feat, 'Mean_TN_K', pool, feature_groups=groups, top_k=max_k, phase_class=1,
                same_family_corr=same_family_corr, cross_family_corr=cross_family_corr,
                n_estimators=n_estimators_reg, random_state=62,
                must_keep=MUST_KEEP_TN if tier_name in {'F3', 'F5'} else []
            )
            if not ranking_tn.empty:
                ranking_tn.to_csv(os.path.join(tier_dir, f'{tier_name}_TN_regression_ranking.csv'), index=False)
            for k in tier_top_k_values:
                _safe_json_dump(selected_tn[:min(k, len(selected_tn))], os.path.join(tier_dir, f'selected_features_{tier_name}_TN_clean{k}.json'))
            tier_selected_tn[tier_name] = selected_tn
            summary[f'{tier_name}_TN_selected_count'] = int(len(selected_tn))

    # Compact F1 aliases for direct diagnostics/visualization. These are not used
    # as the main modeling aliases, but they make the reduced composition-only
    # baseline easy to load and compare.
    if 'F1' in tier_selected_class:
        for k in f1_top_k_values:
            _safe_json_dump(tier_selected_class['F1'][:min(k, len(tier_selected_class['F1']))], os.path.join(output_dir, f'selected_features_F1_global_clean{k}.json'))
    if 'F1' in tier_selected_tc:
        for k in f1_top_k_values:
            _safe_json_dump(tier_selected_tc['F1'][:min(k, len(tier_selected_tc['F1']))], os.path.join(output_dir, f'selected_features_F1_tc_clean{k}.json'))
    if 'F1' in tier_selected_tn:
        for k in f1_top_k_values:
            _safe_json_dump(tier_selected_tn['F1'][:min(k, len(tier_selected_tn['F1']))], os.path.join(output_dir, f'selected_features_F1_tn_clean{k}.json'))

    # Backward-compatible F3 aliases for modeling.py.
    if 'F3' in tier_selected_class:
        selected_f3 = tier_selected_class['F3']
        for k in _ordered_unique(list(top_k_values) + [100, 120]):
            _safe_json_dump(selected_f3[:min(k, len(selected_f3))], os.path.join(output_dir, f'selected_features_global_clean{k}.json'))
        fm_afm_path = os.path.join(output_dir, 'target_specific_F3', 'F3_fm_afm_component_ranking.csv')
        if os.path.exists(fm_afm_path):
            r = pd.read_csv(fm_afm_path)
            feats = _apply_must_keep(r['feature'].head(max(top_k_values)).tolist(), tiers['F3'], MUST_KEEP_F3, max(top_k_values))
        else:
            feats = selected_f3
        for k in _ordered_unique(list(top_k_values) + [100, 120]):
            _safe_json_dump(feats[:min(k, len(feats))], os.path.join(output_dir, f'selected_features_fm_afm_clean{k}.json'))
    if 'F3' in tier_selected_tc:
        for k in _ordered_unique(list(top_k_values) + [100, 120]):
            _safe_json_dump(tier_selected_tc['F3'][:min(k, len(tier_selected_tc['F3']))], os.path.join(output_dir, f'selected_features_tc_clean{k}.json'))
    if 'F3' in tier_selected_tn:
        for k in _ordered_unique(list(top_k_values) + [100, 120]):
            _safe_json_dump(tier_selected_tn['F3'][:min(k, len(tier_selected_tn['F3']))], os.path.join(output_dir, f'selected_features_tn_clean{k}.json'))
    summary_path = os.path.join(output_dir, 'target_specific_feature_selection_summary.json')
    _safe_json_dump(summary, summary_path)
    print(f"Saved target-specific feature-selection summary to {summary_path}")
    return summary


# Backward-compatible wrapper expected by earlier notebooks/scripts.
def run_fast_feature_selection_outputs(df_feat, groups, output_dir='output/featurization', top_k_values=(40, 60, 80), f1_top_k_values=F1_TOP_K_VALUES, corr_threshold=0.99, n_estimators=120):
    return run_target_specific_feature_selection_outputs(
        df_feat, groups, output_dir=output_dir, top_k_values=top_k_values, f1_top_k_values=f1_top_k_values,
        same_family_corr=0.95, cross_family_corr=corr_threshold,
        n_estimators_class=n_estimators, n_estimators_reg=max(160, n_estimators), make_tsne=False
    )


def featurize_all_datasets(input_dir='output/preprocessing', output_dir='output/featurization',
                           run_feature_selection=True, make_tsne=False,
                           same_family_corr=0.95, cross_family_corr=0.99,
                           corr_threshold=None, plot_dir='plots'):
    """Generate features and optional target-specific selected feature sets."""
    if corr_threshold is not None:
        cross_family_corr = corr_threshold
    os.makedirs(output_dir, exist_ok=True)
    in_path = os.path.join(input_dir, 'preprocessed_master.csv')
    if not os.path.exists(in_path):
        print(f"Error: {in_path} not found. Please run preprocessing first.")
        return
    print(f"\nFeaturizing Master Preprocessed Dataset: {in_path}...")
    df = pd.read_csv(in_path, low_memory=False)
    df_feat = generate_features_for_dataset(df, formula_col='reduced_formula')

    # Filter sparse/brittle formula-family flags.
    family_flags = ['AB_like', 'AB2_like', 'ABO3_like', 'AB2O4_like', 'A2BO4_like', 'AFeO3_like',
                    'MFe2O4_like', 'hexaferrite_like', 'heusler_X2YZ_like', 'half_heusler_XYZ_like']
    if 'Type' in df_feat.columns:
        df_clean = df_feat.dropna(subset=['Type'])
        flags_to_remove = []
        for flag in family_flags:
            if flag in df_feat.columns:
                positives = df_clean[df_clean[flag] == 1.0]
                if len(positives) < 50 or (positives['Type'].nunique() if len(positives) else 0) < 2:
                    flags_to_remove.append(flag)
        if flags_to_remove:
            print(f"Removing sparse/brittle formula family flags with insufficient support: {flags_to_remove}")
            df_feat = df_feat.drop(columns=flags_to_remove, errors='ignore')

    cols_to_drop = [c for c in FORBIDDEN_MP_PHYSICAL + METADATA_COLS if c in df_feat.columns]
    print(f"Dropping {len(cols_to_drop)} forbidden physical and metadata columns: {cols_to_drop}")
    df_feat = df_feat.drop(columns=cols_to_drop, errors='ignore')
    targets_to_exclude = [
        'Type', 'Mean_TC_K', 'Mean_TN_K', 'K1_J_m3', 'Coercivity_A_m',
        'Saturation_Magnetization_emu_g', 'Hard_Magnet',
        'clean_BH_max_kJ_m3', 'clean_K1_J_m3', 'clean_Hc_A_m', 
        'clean_Ms_emu_g', 'clean_theta_p_K', 'recovered_ha_tesla'
    ]
    feature_candidates = [c for c in df_feat.columns if c not in targets_to_exclude +
                          ['reduced_formula', 'match_confidence', 'match_score', 'match_reason']]

    out_path = os.path.join(output_dir, 'features_master.csv')
    df_feat.to_csv(out_path, index=False)
    print(f"Saved consolidated master features dataset with {df_feat.shape[0]} rows and {df_feat.shape[1]} columns to {out_path}")

    groups = build_feature_groups(feature_candidates)
    groups_path = os.path.join(output_dir, 'feature_groups.json')
    _safe_json_dump(groups, groups_path)
    print(f"\nSuccessfully generated and saved feature groups schema to {groups_path}!")

    selection_summary = {}
    if run_feature_selection:
        selection_summary = run_target_specific_feature_selection_outputs(
            df_feat, groups, output_dir=output_dir, plot_dir=plot_dir, top_k_values=(40, 60, 80), f1_top_k_values=F1_TOP_K_VALUES,
            same_family_corr=same_family_corr, cross_family_corr=cross_family_corr,
            n_estimators_class=120, n_estimators_reg=160, make_tsne=make_tsne
        )

    stage2_json_path = os.path.join(output_dir, 'stage2_featurization.json')
    stage2_data = {
        "stage": "Stage 2: Refined Compact Magnetic Featurization and Target-Specific Selection",
        "description": "Computes compact formula-level magnetic descriptors with curated Magpie statistics, MagpieEX-lite cation-anion descriptors, boundary-aware interaction indices, family-quota feature selection, and target-specific selected features for phase classification, TC regression, and TN regression.",
        "master_features_csv": out_path,
        "feature_families": {k: len(v) for k, v in groups.items()},
        "records_count": len(df_feat),
        "total_columns": df_feat.shape[1],
        "feature_groups_json": groups_path,
        "target_specific_feature_selection": selection_summary,
        "f1_reduced_feature_sets": "F1 is now selected separately as clean20/clean30/clean40 for classification, TC and TN.",
    }
    _safe_json_dump(stage2_data, stage2_json_path)
    print(f"Saved Stage 2 featurization summary JSON to {stage2_json_path}")
    print("\n--- Featurization Complete! ---")


if __name__ == '__main__':
    featurize_all_datasets()
