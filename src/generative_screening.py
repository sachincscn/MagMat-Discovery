"""
Generative AI Crystal Screening Engine.
Provides an inverse-design screening pipeline to ingest candidate 3D crystals generated
by inorganic crystal diffusion models (e.g., MatterGen, CDVAE, DiffCSP) and evaluate them
against our forward multi-task magnetic models.

Screens:
1. Magnetic ground state: P(FM), P(AFM), P(NM)
2. Ordering temperatures: Curie (TC) and Néel (TN) in Kelvin
3. Magnetocrystalline Anisotropy Constant K1 (MJ/m^3)
4. Permanent Magnet Hardness Parameter: kappa = sqrt(K1 / (mu0 * Ms^2))
5. Critical Raw Material (CRM) Sustainability Index & Rare-Earth Independence
"""

import os
import re
import math
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union

# Physics Constants
MU_0 = 4.0 * math.pi * 1e-7  # Vacuum permeability (T*m/A)
BOHR_MAGNETON_TO_A_M = 9.274010e-24  # J/T

try:
    from src.multi_task_pinn import PhysicalProjectionLayer, IDEAL_BH_MAX_FACTOR
except ImportError:
    class PhysicalProjectionLayer:
        @staticmethod
        def project(ms_emu_g, k1_j_m3, hc_a_m, bh_max_kj_m3, density=7.5):
            rho = np.clip(np.asarray(density), 3.0, 15.0)
            ms_proj = np.clip(ms_emu_g, 0.1, 350.0)
            mu0_ms = np.maximum(4.0 * np.pi * 1e-4 * rho * ms_proj, 1e-5)
            ms_a_m = mu0_ms / MU_0
            k1_proj = np.clip(k1_j_m3, -1e8, 1e8)
            k1_pos = np.maximum(k1_proj, 10.0)
            ha_a_m = 2.0 * k1_pos / mu0_ms
            hc_proj = np.clip(hc_a_m, 1.0, np.minimum(ha_a_m, 1e8))
            bh_ideal = 198.94367886486917 * (mu0_ms ** 2)
            bh_extrinsic = np.maximum(1e-3 * MU_0 * hc_proj * ms_a_m * 0.35, 1e-3)
            bh_max_ceiling = np.minimum(bh_ideal, bh_extrinsic)
            bh_proj = np.minimum(np.maximum(bh_max_kj_m3, 0.01), bh_max_ceiling)
            bh_proj = np.minimum(bh_proj, bh_ideal)
            kappa = np.sqrt(np.maximum(k1_proj, 0.0) / (MU_0 * (ms_a_m ** 2) + 1e-6))
            return ms_proj, k1_proj, hc_proj, bh_proj, kappa
    IDEAL_BH_MAX_FACTOR = 198.94367886486917

CRITICAL_RAW_MATERIALS = {
    'Nd': 1.0, 'Dy': 1.0, 'Tb': 1.0, 'Sm': 0.8, 'Pr': 0.8, 
    'Eu': 0.7, 'Co': 0.4, 'Ni': 0.2, 'Li': 0.3
}


class GenerativeCrystalScreeningEngine:
    """
    Inverse-design screening engine connecting generative inorganic crystal models
    with the forward magnetic property evaluation pipeline.
    """
    def __init__(self, models_dir: str = "output/training", models_path: Optional[str] = None):
        if models_path is not None and os.path.isfile(models_path):
            self.models_dir = os.path.dirname(models_path)
            self.models_path = models_path
        else:
            self.models_dir = models_dir
            self.models_path = os.path.join(models_dir, "trained_models.pkl")
            
        self.clf_model = None
        self.tc_model = None
        self.tn_model = None
        self.k1_model = None
        self.ms_model = None
        self.hc_model = None
        self.bh_model = None
        self.pinn_model = None
        self.magformer_model = None
        self.models_pack = None
        self.feature_cols = None
        self._load_models_if_available()

    def screen_crystal(self, formula: str, crystal_system: str = "cubic", spacegroup_number: int = 0, sample_form: str = "bulk", **kwargs) -> Dict[str, Any]:
        """Convenience method to screen a single crystal formula."""
        df = self.screen_candidates([{
            "formula": str(formula).strip(),
            "crystal_system": crystal_system,
            "spacegroup_number": spacegroup_number,
            "sample_form": sample_form
        }])
        return df.iloc[0].to_dict()

    def _load_models_if_available(self):
        """Loads serialized models if present on disk."""
        # 1. First try unified models package (trained_models.pkl)
        if self.models_path and os.path.exists(self.models_path):
            try:
                self.models_pack = joblib.load(self.models_path)
                if isinstance(self.models_pack, dict):
                    self.clf_model = self.models_pack.get('classification')
                    self.tc_model = self.models_pack.get('curie')
                    self.tn_model = self.models_pack.get('neel')
                    self.k1_model = self.models_pack.get('k1') or self.models_pack.get('anisotropy')
                    self.ms_model = self.models_pack.get('magnetization')
                    self.hc_model = self.models_pack.get('coercivity')
                    self.bh_model = self.models_pack.get('bh_max')
                    self.pinn_model = self.models_pack.get('pinn_multitask')
                    self.magformer_model = self.models_pack.get('magformer')
            except Exception as e:
                print(f"Warning: Could not load unified models pack from {self.models_path}: {e}")

        # Check standalone MagFormer checkpoint if not in pack
        if self.magformer_model is None:
            mf_path = os.path.join(self.models_dir, "magformer_model.pt")
            if os.path.exists(mf_path):
                try:
                    from src.composition_gnn import load_magformer_model
                    self.magformer_model = load_magformer_model(mf_path)
                except Exception as e:
                    print(f"Warning: Could not load MagFormer checkpoint from {mf_path}: {e}")

        # 2. Check standalone pickle fallbacks if needed
        clf_path = os.path.join(self.models_dir, "classification_model.pkl")
        tc_path = os.path.join(self.models_dir, "tc_model.pkl")
        tn_path = os.path.join(self.models_dir, "tn_model.pkl")
        
        if self.clf_model is None and os.path.exists(clf_path):
            try:
                self.clf_model = joblib.load(clf_path)
            except Exception as e:
                print(f"Warning: Could not load classification model: {e}")
        if self.tc_model is None and os.path.exists(tc_path):
            try:
                self.tc_model = joblib.load(tc_path)
            except Exception as e:
                print(f"Warning: Could not load TC model: {e}")
        if self.tn_model is None and os.path.exists(tn_path):
            try:
                self.tn_model = joblib.load(tn_path)
            except Exception as e:
                print(f"Warning: Could not load TN model: {e}")

    @staticmethod
    def parse_cif_metadata(cif_text: str) -> Dict[str, Any]:
        """
        Extracts formula, space group, and lattice parameters from a CIF string.
        """
        meta = {
            "formula": "Fe",
            "spacegroup_number": 0,
            "crystal_system": "cubic",
            "volume": 0.0,
            "nsites": 1,
            "is_centrosymmetric": 1,
            "is_polar": 0,
            "is_chiral": 0,
        }
        
        for line in cif_text.splitlines():
            line = line.strip()
            if line.startswith("_chemical_formula_sum"):
                f = line.split(maxsplit=1)[-1].replace("'", "").replace('"', '').strip()
                meta["formula"] = f
            elif line.startswith("_symmetry_Int_Tables_number") or line.startswith("_space_group_IT_number"):
                try:
                    meta["spacegroup_number"] = int(line.split()[-1])
                except ValueError:
                    pass
            elif line.startswith("_cell_volume"):
                try:
                    meta["volume"] = float(line.split()[-1].split('(')[0])
                except ValueError:
                    pass
                    
        return meta

    @staticmethod
    def compute_sustainability_penalty(element_fractions: Dict[str, float]) -> float:
        """
        Computes Critical Raw Material (CRM) penalty score in [0.0, 1.0].
        0.0 = completely earth-abundant (Fe, Co, Mn, Ti, B, C, N).
        1.0 = heavily dependent on scarce/critical rare earths (Nd, Dy, Tb).
        """
        penalty = 0.0
        for el, frac in element_fractions.items():
            cost = CRITICAL_RAW_MATERIALS.get(el, 0.0)
            penalty += cost * frac
        return float(np.clip(penalty, 0.0, 1.0))

    @staticmethod
    def parse_chemical_formula(formula: str) -> Dict[str, float]:
        """
        Parses chemical formula into element amounts dictionary.
        """
        if not formula or pd.isna(formula):
            return {'Fe': 1.0}
        s = str(formula).strip()
        try:
            from pymatgen.core import Composition
            c = Composition(s)
            return {str(k): float(v) for k, v in c.get_el_amt_dict().items()}
        except Exception:
            pass
        matches = re.findall(r'([A-Z][a-z]*)(\d*\.?\d*)', s)
        comp = {}
        for el, amt_str in matches:
            amt = float(amt_str) if amt_str else 1.0
            comp[el] = comp.get(el, 0.0) + amt
        return comp if comp else {'Fe': 1.0}

    @classmethod
    def estimate_saturation_magnetization(cls, formula: str, volume_per_atom_a3: float = 18.0) -> float:
        """
        Estimates Ms (in A/m) from quantum spin alignments via Magnetic Structure Network (MSN).
        Returns 0.0 if quantum resolution is unavailable (never fabricates ungrounded elemental moments).
        """
        try:
            from src.dao_engine import extract_dao_descriptors
            d_desc = extract_dao_descriptors(formula)
            if "dao_msn_ms_emu_g" in d_desc and d_desc["dao_msn_ms_emu_g"] > 0.0:
                density = float(d_desc.get("dao_density", 7.5))
                # emu/g to A/m: Ms_Am = Ms_emug * density * 1000
                return float(d_desc["dao_msn_ms_emu_g"] * density * 1000.0)
        except Exception:
            pass

        return 0.0

    @staticmethod
    def compute_magnetic_hardness_parameter(k1_j_m3: float, ms_a_m: float) -> float:
        """
        Computes the dimensionless magnetic hardness parameter:
        kappa = sqrt(K1 / (mu_0 * Ms^2))
        kappa > 1.0: Hard permanent magnet (Nd2Fe14B ~ 1.5, SmCo5 ~ 3.0)
        0.1 <= kappa <= 1.0: Semi-hard magnet (Alnico, MnBi ~ 1.2)
        kappa < 0.1: Soft magnetic material (Fe-Si, permalloy)
        """
        if ms_a_m <= 1e-4 or k1_j_m3 <= 0:
            return 0.0
        denom = MU_0 * (ms_a_m ** 2)
        if denom <= 0:
            return 0.0
        return float(math.sqrt(k1_j_m3 / denom))

    def screen_candidates(
        self, 
        candidates: List[Dict[str, Any]], 
        min_tc_kelvin: float = 350.0,
        require_earth_abundant: bool = True
    ) -> pd.DataFrame:
        """
        Screens a batch of generated candidates through the full physical filter.
        """
        if not candidates:
            return pd.DataFrame()

        from src.featurization import generate_features_for_dataset
        from src.dao_engine import extract_dao_descriptors

        INTRINSIC_MAGNETIC_ELEMENTS = {
            'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'V',
            'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Ru', 'Rh', 'Os', 'Ir'
        }

        formulas = [str(cand.get("formula", "Fe")).strip() for cand in candidates]
        cs_list = [cand.get("crystal_system", "cubic") for cand in candidates]
        sg_list = [cand.get("spacegroup_number", 0) for cand in candidates]

        # 1. Batch extract compositional and physical features
        df_batch = pd.DataFrame({
            "reduced_formula": formulas,
            "crystal_system": cs_list,
            "spacegroup_number": sg_list
        })
        try:
            df_physics = generate_features_for_dataset(df_batch, formula_col='reduced_formula')
        except Exception:
            df_physics = df_batch.copy()

        # 2. Extract DAO 3D Foundation Model Descriptors
        dao_rows = []
        for formula, cs in zip(formulas, cs_list):
            try:
                cs_hint = None if (not cs or cs in ["Not specified", "unknown", "cubic"]) else cs
                dao_desc = extract_dao_descriptors(formula, crystal_system=cs_hint)
            except Exception:
                dao_desc = {}
            dao_rows.append(dao_desc)
        df_dao = pd.DataFrame(dao_rows)

        # Merge features
        df_full = pd.concat([df_physics, df_dao], axis=1)

        # Microstructure & morphology defaults (bulk default for alloy screening, with optional cand override)
        sample_forms = [cand.get("sample_form", "bulk") for cand in candidates]
        df_full["micro_is_bulk"] = [1.0 if f == "bulk" else 0.0 for f in sample_forms]
        df_full["micro_is_thin_film"] = [1.0 if f == "film" else 0.0 for f in sample_forms]
        df_full["micro_is_nanostructured"] = [1.0 if f == "nano" else 0.0 for f in sample_forms]
        df_full["micro_log_thickness_nm"] = [6.0 if f == "bulk" else (1.7 if f == "film" else 1.0) for f in sample_forms]

        # Purge any ungrounded proxy columns to maintain strict physical validity
        proxy_cols = [c for c in df_full.columns if 'proxy' in c.lower() or 'gsmagmom' in c.lower() or 'gs_magmom' in c.lower()]
        if proxy_cols:
            df_full = df_full.drop(columns=proxy_cols, errors='ignore')

        # Batch predict with MagFormer Transformer if available
        tr_probs = None
        tr_tc = None
        tr_tn = None
        tr_tc_std = None
        tr_tn_std = None
        if self.magformer_model is not None:
            try:
                from src.composition_gnn import predict_with_gnn
                tr_probs, tr_tc, tr_tn, tr_tc_std, tr_tn_std = predict_with_gnn(
                    self.magformer_model, formulas, task_type='all_with_uncertainty'
                )
            except Exception as e:
                print(f"Warning: MagFormer batch inference failed: {e}")

        results = []
        phase_map = {0: "FM", 1: "AFM", 2: "NM"}

        for i, cand in enumerate(candidates):
            formula = formulas[i]
            sg = sg_list[i]
            cs = cs_list[i]
            dao_desc = dao_rows[i]
            feat_df = df_full.iloc[[i]].copy()

            # Default initializations for all prediction and uncertainty metrics
            pred_phase = "NM"
            tc_k = 0.0
            tc_unc = 0.0
            tn_k = 0.0
            tn_unc = 0.0
            ms_val = 0.0
            mu0_ms_tesla = 0.0
            k1_val = 0.0
            hc_val = 0.0
            bh_val = 0.0
            kappa_val = 0.0
            ms_unc_t = 0.0
            k1_unc_mj = 0.0
            hc_unc_ka = 0.0
            bh_unc_kj = 0.0
            pm_score = 0.0
            is_viable_pm = False
            bh_ideal_ceiling = 0.0
            ha_limit_ka_m = 0.0

            try:
                comp = self.parse_chemical_formula(formula)
                total_atoms = sum(comp.values())
                fracs = {el: amt / max(1.0, total_atoms) for el, amt in comp.items()}
            except Exception:
                fracs = {'Fe': 1.0}
                total_atoms = 1.0

            dao_density = float(dao_desc.get("dao_density", 7.5))
            dao_v_atom = float(dao_desc.get("dao_volume_per_atom", 18.0))
            if "density" not in cand or cand["density"] == 7.5 or cand["density"] is None:
                density_est = dao_density
            else:
                density_est = float(cand["density"])
            density_est = float(np.clip(density_est, 3.0, 22.0))

            if "volume_per_atom" not in cand or cand["volume_per_atom"] == 18.0 or cand["volume_per_atom"] is None:
                v_atom = dao_v_atom
            else:
                v_atom = float(cand["volume_per_atom"])

            crm_penalty = self.compute_sustainability_penalty(fracs)
            has_intrinsic = any(el in fracs for el in INTRINSIC_MAGNETIC_ELEMENTS)

            if not has_intrinsic:
                # Strictly Non-Magnetic Closed-Shell / Main-Group System
                pred_phase = "NM"
                tc_k = 0.0
                tc_unc = 0.0
                tn_k = 0.0
                tn_unc = 0.0
                ms_val = 0.0
                mu0_ms_tesla = 0.0
                k1_val = 0.0
                hc_val = 0.0
                bh_val = 0.0
                kappa_val = 0.0
                ms_unc_t = 0.0
                k1_unc_mj = 0.0
                hc_unc_ka = 0.0
                bh_unc_kj = 0.0
                pm_score = 0.0
                is_viable_pm = False
                bh_ideal_ceiling = 0.0
                ha_limit_ka_m = 0.0
            else:
                ms_a_m = 0.0
                ms_emu_g = 0.0

                is_uniaxial = (
                    str(cs).lower() in ['tetragonal', 'hexagonal', 'trigonal', 'rhombohedral'] 
                    or (75 <= int(sg) <= 194)
                    or (dao_desc.get("dao_is_uniaxial", 0.0) > 0.5)
                )

                k1_j_m3 = 0.0
                k1_unc_j = float('nan')

                # Multi-Task PINN predictions as cascade features
                if self.pinn_model is not None:
                    try:
                        pinn_p, _ = self.pinn_model.predict(feat_df)
                        if isinstance(pinn_p, pd.DataFrame):
                            feat_df['pinn_joint_pred_ms'] = pinn_p['ms'].values
                            feat_df['pinn_joint_pred_k1_symlog'] = np.sign(pinn_p['k1'].values) * np.log10(1.0 + np.abs(pinn_p['k1'].values))
                            feat_df['pinn_joint_pred_hc_log'] = np.log10(np.maximum(pinn_p['hc'].values, 1.0))
                            feat_df['pinn_joint_pred_bh_max_log'] = np.log10(np.maximum(pinn_p['bh_max'].values, 0.01))
                            feat_df['pinn_joint_hardness_kappa'] = pinn_p['kappa'].values
                    except Exception:
                        pass

                # K1: ML ensemble prediction with calibrated uncertainty
                if self.k1_model is not None:
                    try:
                        if hasattr(self.k1_model, 'predict_with_uncertainty'):
                            k1_m_arr, k1_u_arr = self.k1_model.predict_with_uncertainty(feat_df)
                            k1_j_m3 = float(k1_m_arr[0])
                            k1_unc_j = float(k1_u_arr[0])
                        else:
                            k1_j_m3 = float(self.k1_model.predict(feat_df)[0])
                            k1_unc_j = float('nan')
                    except Exception:
                        pass

                ms_unc_emug = float('nan')
                if self.ms_model is not None:
                    try:
                        if hasattr(self.ms_model, 'predict_with_uncertainty'):
                            ms_m_arr, ms_u_arr = self.ms_model.predict_with_uncertainty(feat_df)
                            ms_emu_g = float(ms_m_arr[0])
                            ms_unc_emug = float(ms_u_arr[0])
                        else:
                            ms_emu_g = float(self.ms_model.predict(feat_df)[0])
                            ms_unc_emug = float('nan')
                        ms_a_m = (ms_emu_g * density_est) * 1000.0
                    except Exception:
                        pass

                # Hc: ML ensemble prediction with calibrated uncertainty
                hc_raw_a_m = 0.0
                hc_unc_a_m = float('nan')
                if self.hc_model is not None:
                    try:
                        if hasattr(self.hc_model, 'predict_with_uncertainty'):
                            hc_m_arr, hc_u_arr = self.hc_model.predict_with_uncertainty(feat_df)
                            hc_raw_a_m = float(hc_m_arr[0])
                            hc_unc_a_m = float(hc_u_arr[0])
                        else:
                            hc_raw_a_m = float(self.hc_model.predict(feat_df)[0])
                            hc_unc_a_m = float('nan')
                    except Exception:
                        pass

                # Multi-Task Joint Physical Projection
                ms_proj, k1_proj, hc_proj, _, kappa_proj = PhysicalProjectionLayer.project(
                    np.array([ms_emu_g]),
                    np.array([k1_j_m3]),
                    np.array([hc_raw_a_m]),
                    np.array([10.0]),
                    density=np.array([density_est])
                )
                ms_val = float(ms_proj[0])
                k1_val = float(k1_proj[0])
                hc_val = float(hc_proj[0])
                kappa_val = float(kappa_proj[0])

                mu0_ms_tesla = 4.0 * math.pi * 1e-4 * density_est * ms_val
                bh_ideal_ceiling = IDEAL_BH_MAX_FACTOR * (mu0_ms_tesla ** 2)
                ha_limit_ka_m = (2.0 * max(k1_val, 10.0) / max(mu0_ms_tesla, 1e-4)) / 1e3

                # (BH)max from thermodynamic ceiling: (BH)max <= mu0*Ms^2 / (4*mu0)
                # Bound by Stoner-Wohlfarth demagnetization: operating point limited by Hc vs Ms
                mu0_hc_t = MU_0 * hc_val
                if mu0_ms_tesla > 1e-3 and hc_val > 0:
                    if mu0_hc_t < 0.5 * mu0_ms_tesla:
                        bh_val = float(np.clip(
                            (mu0_hc_t * mu0_ms_tesla * (1.0 - mu0_hc_t / (2.0 * mu0_ms_tesla))) / (MU_0 * 1e3),
                            0.0, bh_ideal_ceiling
                        ))
                    else:
                        bh_val = float(np.clip(bh_ideal_ceiling, 0.0, bh_ideal_ceiling))
                else:
                    bh_val = 0.0

                # Uncertainty in physical units (NaN-safe: propagated from ensemble disagreement)
                ms_unc_t = float(4.0 * math.pi * 1e-4 * density_est * ms_unc_emug) if not math.isnan(ms_unc_emug) else float('nan')
                k1_unc_mj = float(k1_unc_j / 1e6) if not math.isnan(k1_unc_j) else float('nan')
                hc_unc_ka = float(hc_unc_a_m / 1e3) if not math.isnan(hc_unc_a_m) else float('nan')

                # Gaussian error propagation for (BH)max uncertainty
                if not math.isnan(ms_unc_t) and not math.isnan(hc_unc_ka) and mu0_ms_tesla > 1e-3 and hc_val > 1.0:
                    rel_ms = ms_unc_t / max(1e-3, mu0_ms_tesla)
                    rel_hc = hc_unc_ka / max(1.0, hc_val / 1e3)
                    rel_bh = math.sqrt((2.0 * rel_ms) ** 2 + rel_hc ** 2)
                    bh_unc_kj = float(bh_val * rel_bh)
                else:
                    bh_unc_kj = float('nan')

                # Magnetic phase classification via ML ensemble with physics refinement and Transformer blending
                tab_probs = None
                if self.clf_model is not None:
                    try:
                        tab_probs = self.clf_model.predict_proba(feat_df)[0]
                    except Exception:
                        p_cls = self.clf_model.predict(feat_df)
                        pred_idx = int(p_cls[0] if isinstance(p_cls, (list, np.ndarray)) else p_cls)
                        tab_probs = np.zeros(3)
                        tab_probs[min(2, pred_idx)] = 1.0

                if tab_probs is not None and tr_probs is not None and i < len(tr_probs):
                    # Entropy-weighted fusion: weight each model inversely by its Shannon entropy
                    # (a more confident model contributes more to the blend)
                    p_tr = np.array(tr_probs[i])
                    h_tab = -np.sum(tab_probs * np.log(np.clip(tab_probs, 1e-8, 1.0)))
                    h_tr = -np.sum(p_tr * np.log(np.clip(p_tr, 1e-8, 1.0)))
                    inv_h_tab = 1.0 / max(h_tab, 1e-6)
                    inv_h_tr = 1.0 / max(h_tr, 1e-6)
                    w_tab = inv_h_tab / (inv_h_tab + inv_h_tr)
                    w_tr = 1.0 - w_tab
                    blended_probs = w_tab * tab_probs + w_tr * p_tr
                    pred_phase = phase_map.get(int(np.argmax(blended_probs)), "FM")
                elif tab_probs is not None:
                    pred_phase = phase_map.get(int(np.argmax(tab_probs)), "FM")
                elif tr_probs is not None and i < len(tr_probs):
                    pred_phase = phase_map.get(int(np.argmax(tr_probs[i])), "FM")
                else:
                    pred_phase = "NM"  # No model available: default to non-magnetic (safest assumption)

                tc_k = 0.0
                tc_unc = 0.0
                tn_k = 0.0
                tn_unc = 0.0
                if pred_phase == "FM":
                    tc_k = 0.0
                    tc_unc = float('nan')
                    has_tab_tc = False
                    if self.tc_model is not None:
                        try:
                            if hasattr(self.tc_model, 'predict_with_uncertainty'):
                                p_tc_arr, u_tc_arr = self.tc_model.predict_with_uncertainty(feat_df)
                                tc_k = float(p_tc_arr[0])
                                tc_unc = float(u_tc_arr[0])
                            else:
                                p_tc = self.tc_model.predict(feat_df)
                                tc_k = float(p_tc[0] if isinstance(p_tc, (list, np.ndarray)) else p_tc)
                                tc_unc = float('nan')
                            has_tab_tc = True
                        except Exception:
                            pass

                    # Bayesian precision-weighted fusion with MagFormer heteroscedastic prediction
                    if tr_tc is not None and i < len(tr_tc) and tr_tc[i] > 10.0:
                        tc_tr_val = float(tr_tc[i])
                        tc_tr_unc = float(tr_tc_std[i]) if (tr_tc_std is not None and i < len(tr_tc_std)) else float('nan')
                        if has_tab_tc and not math.isnan(tc_unc) and not math.isnan(tc_tr_unc) and tc_unc > 0 and tc_tr_unc > 0:
                            # Inverse-variance weighting (optimal for Gaussian posteriors)
                            tau_tab = 1.0 / (tc_unc ** 2)
                            tau_tr = 1.0 / (tc_tr_unc ** 2)
                            w_tab = tau_tab / (tau_tab + tau_tr)
                            w_tr = tau_tr / (tau_tab + tau_tr)
                            tc_k = w_tab * tc_k + w_tr * tc_tr_val
                            tc_unc = math.sqrt(1.0 / (tau_tab + tau_tr))
                        elif not has_tab_tc:
                            tc_k = tc_tr_val
                            tc_unc = tc_tr_unc
                    tc_k = max(10.0, min(1600.0, tc_k))

                elif pred_phase == "AFM":
                    ms_val = 0.0
                    mu0_ms_tesla = 0.0
                    bh_val = 0.0
                    kappa_val = 0.0
                    ms_unc_t = float('nan')
                    k1_unc_mj = float('nan')
                    hc_unc_ka = float('nan')
                    bh_unc_kj = float('nan')
                    tn_k = 0.0
                    tn_unc = float('nan')
                    has_tab_tn = False
                    if self.tn_model is not None:
                        try:
                            if hasattr(self.tn_model, 'predict_with_uncertainty'):
                                p_tn_arr, u_tn_arr = self.tn_model.predict_with_uncertainty(feat_df)
                                tn_k = float(p_tn_arr[0])
                                tn_unc = float(u_tn_arr[0])
                            else:
                                p_tn = self.tn_model.predict(feat_df)
                                tn_k = float(p_tn[0] if isinstance(p_tn, (list, np.ndarray)) else p_tn)
                                tn_unc = float('nan')
                            has_tab_tn = True
                        except Exception:
                            pass

                    # Bayesian precision-weighted fusion with MagFormer heteroscedastic prediction
                    if tr_tn is not None and i < len(tr_tn) and tr_tn[i] > 10.0:
                        tn_tr_val = float(tr_tn[i])
                        tn_tr_unc = float(tr_tn_std[i]) if (tr_tn_std is not None and i < len(tr_tn_std)) else float('nan')
                        if has_tab_tn and not math.isnan(tn_unc) and not math.isnan(tn_tr_unc) and tn_unc > 0 and tn_tr_unc > 0:
                            tau_tab = 1.0 / (tn_unc ** 2)
                            tau_tr = 1.0 / (tn_tr_unc ** 2)
                            w_tab = tau_tab / (tau_tab + tau_tr)
                            w_tr = tau_tr / (tau_tab + tau_tr)
                            tn_k = w_tab * tn_k + w_tr * tn_tr_val
                            tn_unc = math.sqrt(1.0 / (tau_tab + tau_tr))
                        elif not has_tab_tn:
                            tn_k = tn_tr_val
                            tn_unc = tn_tr_unc
                    tn_k = max(10.0, min(1200.0, tn_k))

                else:  # NM
                    ms_val = 0.0
                    mu0_ms_tesla = 0.0
                    k1_val = 0.0
                    hc_val = 0.0
                    bh_val = 0.0
                    kappa_val = 0.0
                    ms_unc_t = float('nan')
                    k1_unc_mj = float('nan')
                    hc_unc_ka = float('nan')
                    bh_unc_kj = float('nan')

                # Permanent magnet figure of merit: Maximum Energy Product (BH)max [kJ/m3]
                # Scaled by critical raw material sustainability fraction if earth-abundance is required
                sustainability_factor = max(0.0, 1.0 - crm_penalty) if require_earth_abundant else 1.0
                pm_score = float(bh_val * sustainability_factor)

                # Physical viability criteria:
                # 1. Ferromagnetic ordering
                # 2. Curie temperature above application minimum (e.g. 350 K)
                # 3. Magnetic hardness parameter kappa >= 0.5 (semi-hard / hard magnet micromagnetic regime)
                # 4. Coercivity Hc >= 10 kA/m
                is_viable_pm = bool(
                    (pred_phase == "FM") and 
                    (tc_k >= min_tc_kelvin) and 
                    (kappa_val >= 0.5) and 
                    (hc_val >= 10.0e3)
                )
                if require_earth_abundant and crm_penalty > 0.3:
                    is_viable_pm = False

            # MagFormer self-attention element weights
            attn_weights = {}
            if self.magformer_model is not None:
                try:
                    from src.composition_gnn import extract_magformer_attention_weights
                    attn_weights = extract_magformer_attention_weights(self.magformer_model, formula)
                except Exception:
                    attn_weights = {}

            def _safe_round(val, ndigits=1):
                """NaN-safe rounding: returns None if NaN for clean JSON serialization."""
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    return None
                return round(val, ndigits)

            results.append({
                "formula": formula,
                "spacegroup_number": sg,
                "crystal_system": cs,
                "predicted_phase": pred_phase,
                "predicted_TC_K": round(tc_k, 1),
                "predicted_TC_uncertainty_K": _safe_round(tc_unc, 1),
                "predicted_TN_K": round(tn_k, 1),
                "predicted_TN_uncertainty_K": _safe_round(tn_unc, 1),
                "mu0_Ms_Tesla": round(mu0_ms_tesla, 2),
                "mu0_Ms_uncertainty_Tesla": _safe_round(ms_unc_t, 3),
                "K1_MJ_m3": round(k1_val / 1e6, 3),
                "K1_uncertainty_MJ_m3": _safe_round(k1_unc_mj, 3),
                "hardness_kappa": round(kappa_val, 2),
                "coercivity_Hc_kA_m": round(hc_val / 1e3, 1),
                "coercivity_Hc_uncertainty_kA_m": _safe_round(hc_unc_ka, 1),
                "coercivity_Hc_kOe": round((hc_val / 79.57747) / 1e3, 2),
                "energy_product_BH_max_kJ_m3": round(bh_val, 1),
                "energy_product_BH_max_uncertainty_kJ_m3": _safe_round(bh_unc_kj, 1),
                "energy_product_BH_max_MGOe": round(bh_val / 7.957747, 2),
                "thermo_BH_max_limit_kJ_m3": round(bh_ideal_ceiling, 1),
                "stoner_wohlfarth_limit_kA_m": round(ha_limit_ka_m, 1),
                "physical_validity_status": "ML ensemble prediction",
                "crm_sustainability_penalty": round(crm_penalty, 2),
                "permanent_magnet_score": round(pm_score, 2),
                "is_viable_sustainable_magnet": is_viable_pm,
                "dao_msn_exchange_stiffness": round(float(dao_desc.get("dao_msn_exchange_stiffness", 5.0)), 2),
                "dao_msn_bethe_slater_ratio": round(float(dao_desc.get("dao_msn_bethe_slater_ratio", 1.0)), 2),
                "dao_msn_a20_cf": round(float(dao_desc.get("dao_msn_a20_cf", 0.0)), 2),
                "dao_msn_ms_emu_g": round(float(dao_desc.get("dao_msn_ms_emu_g", 50.0)), 1),
                "sample_form": cand.get("sample_form", "bulk"),
                "magformer_attention_weights": attn_weights,
                "transformer_phase_confidence": float(round(float(np.max(tr_probs[i])) * 100.0, 1)) if (tr_probs is not None and i < len(tr_probs)) else None,
                "transformer_tc_k": float(round(float(tr_tc[i]), 1)) if (tr_tc is not None and i < len(tr_tc)) else None,
                "transformer_tn_k": float(round(float(tr_tn[i]), 1)) if (tr_tn is not None and i < len(tr_tn)) else None
            })

        df_out = pd.DataFrame(results)
        df_out = df_out.sort_values(by="permanent_magnet_score", ascending=False).reset_index(drop=True)
        return df_out

