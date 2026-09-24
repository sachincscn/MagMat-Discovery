"""
MACE-MP-0 Foundation Model & NIST JARVIS 3D Database Featurizer.
Extracts higher-order equivariant atomic cluster expansion descriptors (stress tensors,
cohesion energies, hydrostatic pressure) and quantum DFT magnetic moments for material formulas.
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pymatgen.core import Composition


# Standard MACE single-element reference energies (eV/atom) pre-calculated on relaxed ground states
ELEMENT_MACE_REF_ENERGIES = {
    'H': -1.12, 'He': 0.00, 'Li': -1.90, 'Be': -3.75, 'B': -6.68, 'C': -9.23, 'N': -8.33, 'O': -4.92,
    'F': -1.88, 'Ne': 0.00, 'Na': -1.31, 'Mg': -1.52, 'Al': -3.74, 'Si': -5.43, 'P': -5.40, 'S': -4.13,
    'Cl': -1.75, 'Ar': 0.00, 'K': -1.03, 'Ca': -1.97, 'Sc': -6.33, 'Ti': -7.75, 'V': -8.95, 'Cr': -9.51,
    'Mn': -8.96, 'Fe': -8.40, 'Co': -7.12, 'Ni': -5.53, 'Cu': -3.72, 'Zn': -1.26, 'Ga': -3.03, 'Ge': -4.62,
    'As': -4.66, 'Se': -3.50, 'Br': -1.55, 'Kr': 0.00, 'Rb': -0.91, 'Sr': -1.68, 'Y': -6.48, 'Zr': -8.54,
    'Nb': -10.22, 'Mo': -10.82, 'Tc': -10.37, 'Ru': -9.24, 'Rh': -7.38, 'Pd': -5.18, 'Ag': -2.83, 'Cd': -0.89,
    'In': -2.71, 'Sn': -3.99, 'Sb': -4.12, 'Te': -3.15, 'I': -1.45, 'Xe': 0.00, 'Cs': -0.83, 'Ba': -1.92,
    'La': -4.92, 'Ce': -5.93, 'Pr': -5.02, 'Nd': -4.85, 'Pm': -4.72, 'Sm': -4.68, 'Eu': -1.82, 'Gd': -4.81,
    'Tb': -4.90, 'Dy': -4.88, 'Ho': -4.86, 'Er': -4.84, 'Tm': -4.81, 'Yb': -1.56, 'Lu': -4.79, 'Hf': -9.95,
    'Ta': -11.85, 'W': -12.96, 'Re': -12.44, 'Os': -11.23, 'Ir': -8.87, 'Pt': -6.05, 'Au': -3.27, 'Hg': -0.45,
    'Tl': -2.39, 'Pb': -3.71, 'Bi': -3.83, 'Th': -7.42, 'Pa': -9.15, 'U': -11.35, 'Np': -12.80, 'Pu': -13.60
}


def _calc_elemental_mace_energy(formula: str) -> float:
    """Estimates stoichiometric weighted MACE cohesive energy when no 3D crystal is available."""
    try:
        comp = Composition(formula)
        total_atoms = sum(comp.values())
        if total_atoms <= 0:
            return -5.0
        weighted_energy = sum(ELEMENT_MACE_REF_ENERGIES.get(el.symbol, -5.0) * amt for el, amt in comp.items())
        return float(weighted_energy / total_atoms)
    except Exception:
        return -5.0


class MACEMagneticFeaturizer:
    """
    Featurizes compositions by combining MACE-MP-0-Small equivariant calculations
    with the 93,902-structure NIST JARVIS-DFT 3D database.
    """
    def __init__(self, cache_dir: str = 'ML_pipeline/output/featurization'):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, 'mace_features_cache.parquet')
        self.calc = None
        self._jarvis_data = None
        self._jarvis_formula_map = None
        self._cached_df = None
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                self._cached_df = pd.read_parquet(self.cache_file)
                if not self._cached_df.empty:
                    self._cached_df = self._cached_df[~self._cached_df.index.duplicated(keep='last')]
                print(f"Loaded {len(self._cached_df)} MACE/JARVIS cached feature rows from {self.cache_file}")
            except Exception as e:
                print(f"Note: Could not read cache file {self.cache_file}: {e}")
                self._cached_df = pd.DataFrame()
        else:
            self._cached_df = pd.DataFrame()

    def _save_cache(self):
        if self._cached_df is not None and not self._cached_df.empty:
            os.makedirs(self.cache_dir, exist_ok=True)
            try:
                self._cached_df.to_parquet(self.cache_file)
            except Exception as e:
                print(f"Warning: Failed to save MACE cache: {e}")

    def _init_mace(self):
        if self.calc is None:
            print("Initializing pre-trained MACE-MP-0-Small foundation model on CPU...")
            from mace.calculators import mace_mp
            self.calc = mace_mp(model='small', device='cpu')

    def _init_jarvis(self):
        if self._jarvis_data is None:
            print("Loading NIST JARVIS-DFT 3D database (94k materials)...")
            from jarvis.db.figshare import data
            self._jarvis_data = data('dft_3d')
            self._jarvis_formula_map = {}
            for idx, r in enumerate(self._jarvis_data):
                f_raw = r.get('formula')
                if f_raw:
                    try:
                        red_f = Composition(f_raw).reduced_formula
                        eh = float(r.get('ehull', 999) if r.get('ehull') not in ['na', None] else 999)
                        if red_f not in self._jarvis_formula_map or eh < self._jarvis_formula_map[red_f][1]:
                            self._jarvis_formula_map[red_f] = (idx, eh)
                    except Exception:
                        pass
            print(f"Indexed {len(self._jarvis_formula_map)} unique reduced chemical formulas in JARVIS.")

    def featurize_formulas(self, formulas: List[str]) -> pd.DataFrame:
        """
        Featurizes a list of chemical formulas using MACE-MP-0-Small and NIST JARVIS.
        Returns a DataFrame indexed by the input formulas.
        """
        unique_formulas = list(dict.fromkeys([str(f).strip() for f in formulas if f and not pd.isna(f)]))
        needed_formulas = [f for f in unique_formulas if f not in self._cached_df.index]

        if needed_formulas:
            print(f"Computing MACE-MP-0-Small and JARVIS features for {len(needed_formulas)} new formulas...")
            self._init_jarvis()
            from jarvis.core.atoms import Atoms

            from tqdm import tqdm
            new_records = []
            mace_initialized = False
            total_needed = len(needed_formulas)
            print(f"Beginning featurization across {total_needed} formulas...")

            for i, f in enumerate(tqdm(needed_formulas, desc="MACE-JARVIS Featurization")):
                matched = False
                if f in self._jarvis_formula_map:
                    idx, eh = self._jarvis_formula_map[f]
                    r = self._jarvis_data[idx]
                    try:
                        if not mace_initialized:
                            self._init_mace()
                            mace_initialized = True

                        j_atoms = Atoms.from_dict(r['atoms'])
                        ase_atoms = j_atoms.ase_converter()
                        ase_atoms.calc = self.calc
                        
                        e = float(ase_atoms.get_potential_energy())
                        n_at = max(len(ase_atoms), 1)
                        e_per_atom = e / n_at
                        
                        stress = ase_atoms.get_stress()  # [xx, yy, zz, yz, xz, xy]
                        s_xx, s_yy, s_zz = stress[0], stress[1], stress[2]
                        s_yz, s_xz, s_xy = stress[3], stress[4], stress[5]
                        
                        axial_stress = float(s_zz - 0.5 * (s_xx + s_yy))
                        hydrostatic_p = float(- (s_xx + s_yy + s_zz) / 3.0)
                        
                        # Von Mises deviatoric shear stress
                        s_dev_xx = s_xx + hydrostatic_p
                        s_dev_yy = s_yy + hydrostatic_p
                        s_dev_zz = s_zz + hydrostatic_p
                        von_mises = float(np.sqrt(0.5 * (s_dev_xx**2 + s_dev_yy**2 + s_dev_zz**2 + 2*(s_yz**2 + s_xz**2 + s_xy**2))))
                        
                        # DFT magnetic moment
                        raw_mag = r.get('magmom_oszicar', 0.0)
                        dft_mag = float(raw_mag) if raw_mag not in ['na', None] else 0.0
                        
                        raw_ehull = r.get('ehull', 0.0)
                        ehull_val = float(raw_ehull) if raw_ehull not in ['na', None] else 0.0
                        
                        raw_gap = r.get('optb88vdw_bandgap', 0.0)
                        bg_val = float(raw_gap) if raw_gap not in ['na', None] else 0.0

                        new_records.append({
                            'formula': f,
                            'mace_energy_per_atom': e_per_atom,
                            'mace_axial_stress': axial_stress,
                            'mace_hydrostatic_pressure': hydrostatic_p,
                            'mace_von_mises_stress': von_mises,
                            'dft_jarvis_magmom': dft_mag,
                            'dft_jarvis_ehull': ehull_val,
                            'dft_jarvis_bandgap': bg_val,
                            'mace_matched': 1.0
                        })
                        matched = True
                    except Exception as e_mace:
                        pass

                if not matched:
                    # Physical stoichiometric fallback
                    e_fallback = _calc_elemental_mace_energy(f)
                    new_records.append({
                        'formula': f,
                        'mace_energy_per_atom': e_fallback,
                        'mace_axial_stress': 0.0,
                        'mace_hydrostatic_pressure': 0.0,
                        'mace_von_mises_stress': 0.0,
                        'dft_jarvis_magmom': np.nan,
                        'dft_jarvis_ehull': np.nan,
                        'dft_jarvis_bandgap': np.nan,
                        'mace_matched': 0.0
                    })

                # Periodic checkpoint save every 1000 formulas
                if len(new_records) >= 1000:
                    df_chunk = pd.DataFrame(new_records).set_index('formula')
                    self._cached_df = pd.concat([self._cached_df, df_chunk])
                    self._cached_df = self._cached_df[~self._cached_df.index.duplicated(keep='last')]
                    self._save_cache()
                    new_records = []

            if new_records:
                df_new = pd.DataFrame(new_records).set_index('formula')
                self._cached_df = pd.concat([self._cached_df, df_new])
                self._cached_df = self._cached_df[~self._cached_df.index.duplicated(keep='last')]
                self._save_cache()
                print(f"Successfully cached MACE/JARVIS features for {len(df_new)} formulas.")

        # Reindex to match the input formulas order
        res_df = self._cached_df.reindex(formulas).copy()
        res_df.index = range(len(formulas))
        return res_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Precompute MACE-MP-0-Small and NIST JARVIS features.")
    parser.add_argument('--input-csv', type=str, default='ML_pipeline/output/featurization/features_master.csv', help='Path to master features CSV')
    parser.add_argument('--cache-dir', type=str, default='ML_pipeline/output/featurization', help='Directory to store cache')
    args = parser.parse_args()

    if os.path.exists(args.input_csv):
        print(f"Reading formulas from {args.input_csv}...")
        df_in = pd.read_csv(args.input_csv, usecols=['reduced_formula'], low_memory=False)
        formulas_list = df_in['reduced_formula'].dropna().unique().tolist()
        print(f"Found {len(formulas_list)} unique formulas to featurize.")
        fe = MACEMagneticFeaturizer(cache_dir=args.cache_dir)
        df_out = fe.featurize_formulas(formulas_list)
        print(f"\nFeaturization complete! Shape: {df_out.shape}")
        print(f"Total matched formulas: {(df_out['mace_matched'] == 1.0).sum()}")
    else:
        print(f"Input file {args.input_csv} not found.")
