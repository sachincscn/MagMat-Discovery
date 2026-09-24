"""Cached data access for MagMat Discovery. All paths resolve to MagMat-Discovery/."""

import json
import math
import os
import re
import sys
from typing import Tuple, Optional, Dict, Any, List

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TRAINING = os.path.join(ROOT, "output", "training")
POST = os.path.join(ROOT, "output", "post_processing")
BENCH = os.path.join(ROOT, "output", "benchmarks")

CRYSTAL_SYSTEMS = ["Not specified", "cubic", "tetragonal", "hexagonal", "trigonal",
                   "orthorhombic", "monoclinic", "triclinic"]

# Valid International Space Group Ranges by Crystal System (ITA Vol A)
# 1-2: Triclinic | 3-15: Monoclinic | 16-74: Orthorhombic | 75-142: Tetragonal
# 143-167: Trigonal | 168-194: Hexagonal | 195-230: Cubic
CRYSTAL_SYSTEM_SPACE_GROUPS = {
    "Not specified": [0],
    "triclinic": [0] + list(range(1, 3)),
    "monoclinic": [0] + list(range(3, 16)),
    "orthorhombic": [0] + list(range(16, 75)),
    "tetragonal": [0] + list(range(75, 143)),
    "trigonal": [0] + list(range(143, 168)),
    "hexagonal": [0] + list(range(168, 195)),
    "cubic": [0] + list(range(195, 231)),
}


def get_valid_space_groups(crystal_system: str) -> list:
    """Return strictly valid space group numbers for a crystal system."""
    cs = (crystal_system or "").strip().lower()
    return CRYSTAL_SYSTEM_SPACE_GROUPS.get(cs, [0])


def get_crystal_system_for_space_group(space_group: int) -> str:
    """Infer crystal system from a space group number (1-230)."""
    sg = int(space_group or 0)
    if 1 <= sg <= 2:
        return "triclinic"
    elif 3 <= sg <= 15:
        return "monoclinic"
    elif 16 <= sg <= 74:
        return "orthorhombic"
    elif 75 <= sg <= 142:
        return "tetragonal"
    elif 143 <= sg <= 167:
        return "trigonal"
    elif 168 <= sg <= 194:
        return "hexagonal"
    elif 195 <= sg <= 230:
        return "cubic"
    return "Not specified"

# Common substituents in magnetic alloy design — 3d partners, light interstitials
# and the rare earths that carry single-ion anisotropy.
SUBSTITUENTS = ["Co", "Fe", "Ni", "Mn", "Cr", "V", "Ti", "Al", "Si", "Ga", "Ge",
                "Sn", "Sb", "B", "C", "N", "O", "Cu", "Zn", "Mo", "Nb", "Ta", "W",
                "Nd", "Sm", "Pr", "Dy", "Tb", "Gd", "Ce", "La", "Y", "Zr", "Hf",
                "Pt", "Pd", "Bi", "Se", "Te", "P", "S"]

# Well-characterised magnets with their real symmetry and archetype descriptions.
ARCHETYPES = {
    "Nd2Fe14B": {
        "formula": "Nd2Fe14B",
        "name": "Nd₂Fe₁₄B",
        "title": "Champion Permanent Magnet",
        "crystal_system": "tetragonal",
        "space_group": 136,
        "symbol": "P4₂/mnm",
        "highlight": "Record energy product (BH)max ≈ 470 kJ/m³, TC = 588 K, modern EV powertrain standard.",
        "amounts": {"Nd": 2.0, "Fe": 14.0, "B": 1.0}
    },
    "SmCo5": {
        "formula": "SmCo5",
        "name": "SmCo₅",
        "title": "Extreme Uniaxial Anisotropy",
        "crystal_system": "hexagonal",
        "space_group": 191,
        "symbol": "P6/mmm",
        "highlight": "Huge anisotropy field μ₀Ha > 25 T (K₁ = 17.2 MJ/m³), TC = 1020 K, aerospace-grade.",
        "amounts": {"Sm": 1.0, "Co": 5.0}
    },
    "Sm2Co17": {
        "formula": "Sm2Co17",
        "name": "Sm₂Co₁₇",
        "title": "High-Temp Power Magnet",
        "crystal_system": "hexagonal",
        "space_group": 194,
        "symbol": "P6₃/mmc",
        "highlight": "Stable up to 550°C with near-zero thermal drift coefficient, TC = 1190 K.",
        "amounts": {"Sm": 2.0, "Co": 17.0}
    },
    "MnBi": {
        "formula": "MnBi",
        "name": "MnBi",
        "title": "Positive Coercivity Temp-Coefficient (RE-Free)",
        "crystal_system": "hexagonal",
        "space_group": 194,
        "symbol": "P6₃/mmc",
        "highlight": "Unique anomaly: coercivity Hc INCREASES upon heating to 540 K; 100% rare-earth free.",
        "amounts": {"Mn": 1.0, "Bi": 1.0}
    },
    "Fe16N2": {
        "formula": "Fe16N2",
        "name": "Fe₁₆N₂",
        "title": "Giant Saturation Polarization",
        "crystal_system": "tetragonal",
        "space_group": 139,
        "symbol": "I4/mmm",
        "highlight": "Theoretical giant magnetic polarization (μ₀Ms ≈ 2.8 T); sustainable earth-abundant Fe-N.",
        "amounts": {"Fe": 16.0, "N": 2.0}
    },
    "FePt": {
        "formula": "FePt",
        "name": "FePt (L1₀)",
        "title": "Ultra-High Density HAMR Storage",
        "crystal_system": "tetragonal",
        "space_group": 123,
        "symbol": "P4/mmm",
        "highlight": "Enormous K₁ = 6.6 MJ/m³ suppresses superparamagnetism in sub-3 nm recording media.",
        "amounts": {"Fe": 1.0, "Pt": 1.0}
    },
    "Fe3Sn": {
        "formula": "Fe3Sn",
        "name": "Fe₃Sn",
        "title": "Kagome Topological Magnet",
        "crystal_system": "hexagonal",
        "space_group": 194,
        "symbol": "P6₃/mmc",
        "highlight": "Topological Kagome lattice exhibiting massive Berry curvature and giant anomalous Hall effect.",
        "amounts": {"Fe": 3.0, "Sn": 1.0}
    },
    "CrO2": {
        "formula": "CrO2",
        "name": "CrO₂",
        "title": "100% Spin-Polarized Half-Metal",
        "crystal_system": "tetragonal",
        "space_group": 136,
        "symbol": "P4₂/mnm",
        "highlight": "True half-metallic ferromagnet with 100% Fermi spin polarization for spin-valve devices.",
        "amounts": {"Cr": 1.0, "O": 2.0}
    },
    "Cr2O3": {
        "formula": "Cr2O3",
        "name": "Cr₂O₃",
        "title": "Archetypal Antiferromagnetic Insulator",
        "crystal_system": "trigonal",
        "space_group": 167,
        "symbol": "R-3c",
        "highlight": "Prototypical collinear antiferromagnet (TN ≈ 308 K) with zero net moment and giant magnetoelectric effect.",
        "amounts": {"Cr": 2.0, "O": 3.0}
    },
    "Al2O3": {
        "formula": "Al2O3",
        "name": "Al₂O₃",
        "title": "Closed-Shell Non-Magnetic Insulator",
        "crystal_system": "trigonal",
        "space_group": 167,
        "symbol": "R-3c",
        "highlight": "Classic wide-bandgap sapphire insulator with completely closed electronic shells (strictly non-magnetic).",
        "amounts": {"Al": 2.0, "O": 3.0}
    },
}

REFERENCE_COMPOUNDS = [
    (k, v["crystal_system"], v["space_group"], v["title"])
    for k, v in ARCHETYPES.items()
]

def to_subscript(formula: str) -> str:
    """Format chemical formula numbers as clean Unicode subscripts."""
    sub_map = str.maketrans("0123456789.", "₀₁₂₃₄₅₆₇₈₉·")
    return re.sub(r'(\d+\.?\d*)', lambda m: m.group(1).translate(sub_map), str(formula))


def format_composition_subscript(amounts: dict) -> str:
    """Format an element-to-amount dict into a clean chemical formula with unicode subscripts."""
    if not amounts:
        return ""
    parts = []
    sub_map = str.maketrans("0123456789.", "₀₁₂₃₄₅₆₇₈₉.")
    for el, amt in amounts.items():
        if amt <= 0:
            continue
        if abs(amt - 1.0) < 1e-6:
            parts.append(str(el))
        else:
            if abs(amt - round(amt)) < 1e-4:
                amt_str = f"{int(round(amt))}"
            else:
                amt_str = f"{amt:.2f}".rstrip("0").rstrip(".")
            parts.append(f"{el}{amt_str.translate(sub_map)}")
    return "".join(parts)


@st.cache_data(show_spinner=False)
def load_mp_ground_states() -> dict:
    """Load thermodynamic ground-state crystal structures indexed by reduced formula."""
    p = os.path.join(ROOT, "Dataset", "mp_alloys_point_groups.csv")
    if not os.path.exists(p):
        return {}
    try:
        df = pd.read_csv(
            p,
            usecols=["reduced_formula", "crystal_system", "spacegroup_number", "spacegroup_symbol", "energy_above_hull"],
            low_memory=False
        )
        df_gs = df.sort_values("energy_above_hull").groupby("reduced_formula").first()
        return df_gs[["crystal_system", "spacegroup_number", "spacegroup_symbol", "energy_above_hull"]].to_dict("index")
    except Exception:
        return {}


def format_sg_symbol(symbol: str) -> str:
    """Format space group symbol with unicode subscripts."""
    if not symbol or symbol == "None":
        return ""
    sub_map = {"_0": "₀", "_1": "₁", "_2": "₂", "_3": "₃", "_4": "₄", "_5": "₅", "_6": "₆", "_7": "₇", "_8": "₈", "_9": "₉"}
    res = str(symbol)
    for k, v in sub_map.items():
        res = res.replace(k, v)
    return res


def infer_crystal_structure(formula: str) -> dict:
    """Infer or predict ground-state crystal structure (crystal system and space group).

    Checks curated ARCHETYPES first, then queries the Materials Project
    thermodynamic ground-state database (132,336 structures).
    """
    if not formula or not str(formula).strip():
        return {"crystal_system": None, "spacegroup_number": 0, "spacegroup_symbol": "", "source": "None"}

    clean_f = str(formula).strip()

    # 1. Curated Archetypes check
    for k, arch in ARCHETYPES.items():
        if clean_f.lower() == k.lower() or clean_f.lower() == arch.get("formula", "").lower():
            return {
                "crystal_system": arch["crystal_system"].capitalize(),
                "spacegroup_number": int(arch["space_group"]),
                "spacegroup_symbol": format_sg_symbol(arch.get("symbol", "")),
                "source": "Curated Archetype",
                "energy_above_hull": 0.0,
            }

    # 2. Materials Project Ground States check via pymatgen Composition
    try:
        from pymatgen.core import Composition
        comp = Composition(clean_f)
        rf = comp.reduced_formula

        # Also check Archetypes with reduced formula
        for k, arch in ARCHETYPES.items():
            try:
                if Composition(k).reduced_formula == rf:
                    return {
                        "crystal_system": arch["crystal_system"].capitalize(),
                        "spacegroup_number": int(arch["space_group"]),
                        "spacegroup_symbol": format_sg_symbol(arch.get("symbol", "")),
                        "source": "Curated Archetype",
                        "energy_above_hull": 0.0,
                    }
            except Exception:
                pass

        gs_dict = load_mp_ground_states()
        if rf in gs_dict:
            entry = gs_dict[rf]
            return {
                "crystal_system": str(entry["crystal_system"]).capitalize(),
                "spacegroup_number": int(entry["spacegroup_number"]),
                "spacegroup_symbol": format_sg_symbol(entry["spacegroup_symbol"]),
                "source": "Materials Project DFT Ground-State",
                "energy_above_hull": float(entry["energy_above_hull"]),
            }

        if clean_f in gs_dict:
            entry = gs_dict[clean_f]
            return {
                "crystal_system": str(entry["crystal_system"]).capitalize(),
                "spacegroup_number": int(entry["spacegroup_number"]),
                "spacegroup_symbol": format_sg_symbol(entry["spacegroup_symbol"]),
                "source": "Materials Project DFT Ground-State",
                "energy_above_hull": float(entry["energy_above_hull"]),
            }
    except Exception:
        pass

    # 3. DAO 3D Foundation Model Ground-State Inference
    try:
        from src.dao_engine import DAOCrystalEngine
        dao = DAOCrystalEngine()
        pred = dao.predict_crystal(clean_f)
        if pred and pred.get("spacegroup_number"):
            return {
                "crystal_system": str(pred["crystal_system"]).capitalize(),
                "spacegroup_number": int(pred["spacegroup_number"]),
                "spacegroup_symbol": format_sg_symbol(pred.get("spacegroup_symbol", "")),
                "source": f"DAO 3D Foundation Model ({pred.get('prototype', 'Generative')})",
                "energy_above_hull": float(pred.get("energy_above_hull", 0.0)),
                "density": float(pred.get("density", 7.5)),
                "volume": float(pred.get("volume", 0.0)),
            }
    except Exception:
        pass

    return {
        "crystal_system": None,
        "spacegroup_number": 0,
        "spacegroup_symbol": "",
        "source": "Unknown (Physical Symmetry Envelope)",
        "energy_above_hull": None,
    }


# Reliability threshold for what the app is willing to present as a prediction.
# R² ≥ 0.70 out-of-fold (or ≥ 0.90 accuracy for the classifier). Below that the
# quantity is a screening signal, not a number to act on, and is confined to the
# Documentation tab where the full validation record lives.
RELIABLE_R2 = 0.70
RELIABLE_ACC = 0.90

# Display metadata for the eight targets. Metrics are the 5-fold
# final_manuscript values recorded in stage3_modeling.json.
TARGETS = [
    dict(key="phase", n=1, name="Magnetic phase", unit="FM / AFM / NM", n_samples="35,045",
         metric="0.9097", metric_label="accuracy", extra="Macro F₁ 0.897 · AFM F₁ 0.793",
         tier="ordering", score=0.9097, reliable=True),
    dict(key="curie_tc", n=2, name="Curie temperature Tᴄ", unit="K", n_samples="15,479",
         metric="0.8467", metric_label="R²", extra="MAE 63.7 K · q₉₀ 148.6 K",
         tier="ordering", score=0.8467, reliable=True),
    dict(key="neel_tn", n=3, name="Néel temperature Tₙ", unit="K", n_samples="6,402",
         metric="0.8071", metric_label="R²", extra="MAE 40.1 K · q₉₀ 97.0 K",
         tier="ordering", score=0.8071, reliable=True),
    dict(key="curie_weiss_theta_p", n=4, name="Curie-Weiss θₚ", unit="K", n_samples="6,820",
         metric="0.6075", metric_label="R²", extra="MAE 54.9 K",
         tier="hysteresis", score=0.6075, reliable=False),
    dict(key="coercivity_hc", n=5, name="Coercivity Hᴄ", unit="A/m", n_samples="6,468",
         metric="0.6172", metric_label="R²", extra="MAE 0.627 dex · q₉₀ 1.52 dex",
         tier="hysteresis", score=0.6172, reliable=False),
    dict(key="saturation_magnetization_ms", n=6, name="Saturation Mₛ", unit="emu/g", n_samples="7,165",
         metric="0.5658", metric_label="R²", extra="MAE 25.0 emu/g · q₉₀ 59.4 emu/g",
         tier="hysteresis", score=0.5658, reliable=False),
    dict(key="energy_product_bh_max", n=7, name="Energy product (BH)ₘₐₓ", unit="kJ/m³", n_samples="1,022",
         metric="0.4794", metric_label="R²", extra="MAE 0.221 dex · q₉₀ 0.51 dex",
         tier="hysteresis", score=0.4794, reliable=False),
    dict(key="anisotropy_k1", n=8, name="Anisotropy K₁", unit="J/m³", n_samples="3,036",
         metric="0.3453", metric_label="R²", extra="MAE 1.606 symlog · q₉₀ 3.67 symlog",
         tier="hysteresis", score=0.3453, reliable=False),
]

RELIABLE = [t for t in TARGETS if t["reliable"]]
EXCLUDED = [t for t in TARGETS if not t["reliable"]]

# Prediction columns the app is willing to surface, keyed to the reliable set.
RELIABLE_COLUMNS = [
    "formula", "crystal_system", "Predicted_Phase", "max_probability",
    "Predicted_Curie_K", "Predicted_Neel_K", "Expected_ordering_temperature_K",
    "Expected_temperature_uncertainty_K", "Material_Criticality_Index",
    "Rare_Earth_Free",
]

# Measured F1 / F3 / F5 representation ablation (5-fold GroupKFold).
ABLATION = [
    ("Magnetic phase (acc)", 0.8984, 0.9016, 0.8992, "ordering"),
    ("Curie Tc", 0.7778, 0.7930, 0.8157, "ordering"),
    ("Néel TN", 0.7406, 0.7407, 0.7301, "ordering"),
    ("Curie-Weiss θₚ", 0.513, 0.607, 0.596, "hysteresis"),
    ("Anisotropy K₁", 0.054, 0.309, 0.310, "hysteresis"),
    ("Energy product (BH)ₘₐₓ", 0.080, 0.478, 0.494, "hysteresis"),
    ("Saturation Ms", -0.001, 0.536, 0.545, "hysteresis"),
    ("Coercivity Hc", -0.000, 0.620, 0.622, "hysteresis"),
]


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    p = os.path.join(TRAINING, "stage3_modeling.json")
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_benchmark() -> dict:
    p = os.path.join(BENCH, "extended_targets_benchmark.json")
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_results_summary() -> dict:
    """Load latest verified production benchmark metrics from results_summary.json."""
    p = os.path.join(ROOT, "results_summary.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_candidates(name: str) -> pd.DataFrame:
    p = os.path.join(POST, name)
    if not os.path.exists(p):
        return pd.DataFrame()
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def catalogue() -> dict:
    """Row counts for each generated candidate set."""
    files = {
        "All MP candidates": "magnetic_predictions_full.csv",
        "High confidence (P≥0.80)": "high_confidence_predictions_Pmax_080.csv",
        "Permanent magnets": "high_performance_permanent_magnets.csv",
        "Sustainable / RE-free": "screened_sustainable_magnets.csv",
        "Frustrated lattices": "frustrated_magnetic_candidates.csv",
        "Confident FM": "high_confidence_FM_candidates.csv",
        "Confident AFM": "high_confidence_AFM_candidates.csv",
    }
    out = {}
    for label, fn in files.items():
        p = os.path.join(POST, fn)
        if os.path.exists(p):
            try:
                out[label] = (sum(1 for _ in open(p)) - 1, fn)
            except Exception:
                out[label] = (0, fn)
    return out


@st.cache_resource(show_spinner=False)
def get_engine():
    """Lazy-load the screening engine; returns (engine, error_message)."""
    try:
        from src.generative_screening import GenerativeCrystalScreeningEngine
        path = os.path.join(TRAINING, "trained_models.pkl")
        if not os.path.exists(path):
            return None, "trained_models.pkl not found — run stage 4 first."
        return GenerativeCrystalScreeningEngine(models_path=path), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


UNKNOWN = "Not specified"

# The screening engine keys anisotropy off a single binary test:
#   is_uniaxial = crystal_system in {tetragonal, hexagonal, trigonal, rhombohedral}
#                 or 75 <= spacegroup <= 194
# Phase, T_C, T_N and M_s are entirely symmetry-independent (verified across all
# seven systems). So when symmetry is unknown there are exactly two possible
# answers, and both can be reported rather than silently assuming one.
UNIAXIAL_PROBE = "hexagonal"
NONUNIAXIAL_PROBE = "cubic"


@st.cache_data(show_spinner=False)
def predict(formula: str, crystal_system: str = "cubic", spacegroup: int = 0, sample_form: str = "bulk"):
    engine, err = get_engine()
    if engine is None:
        return None, err
    try:
        return engine.screen_crystal(formula.strip(), crystal_system, int(spacegroup), sample_form=sample_form), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False)
def predict_auto(formula: str, crystal_system: str = UNKNOWN, spacegroup: int = 0, sample_form: str = "bulk"):
    """Predict with symmetry optional.

    Returns (result, alternative, err). When the crystal system is given,
    alternative is None. When it is not, result is the non-uniaxial
    branch and alternative the uniaxial one.
    """
    known = crystal_system and crystal_system != UNKNOWN
    if known or (spacegroup and int(spacegroup) > 0):
        cs = crystal_system if known else NONUNIAXIAL_PROBE
        res, err = predict(formula, cs, spacegroup, sample_form=sample_form)
        return res, None, err

    engine, err = get_engine()
    if engine is None:
        return None, None, err
    try:
        cands = [
            {"formula": str(formula).strip(), "crystal_system": NONUNIAXIAL_PROBE, "spacegroup_number": 0, "sample_form": sample_form},
            {"formula": str(formula).strip(), "crystal_system": UNIAXIAL_PROBE, "spacegroup_number": 0, "sample_form": sample_form},
        ]
        df = engine.screen_candidates(cands)
        if len(df) >= 2:
            return df.iloc[0].to_dict(), df.iloc[1].to_dict(), None
        elif len(df) == 1:
            return df.iloc[0].to_dict(), None, None
        return None, None, "No screening results produced."
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def provenance() -> dict:
    m = load_metrics()
    return m.get("run_config", {}) if m else {}


#  Composition entry
def parse_formula(text: str):
    """Validate a formula. Returns (pretty, element_amounts, error)."""
    text = (text or "").strip()
    if not text:
        return None, None, "Enter a formula."
    try:
        from pymatgen.core import Composition
        comp = Composition(text)
        if len(comp) == 0:
            return None, None, "No elements recognised."
        amounts = {str(el): float(amt) for el, amt in comp.get_el_amt_dict().items()}
        return comp.reduced_formula, amounts, None
    except Exception:
        return None, None, f"'{text}' is not a valid chemical formula."


def build_formula(rows) -> str:
    """Compose a formula string from (element, amount) rows."""
    parts = []
    for el, amt in rows:
        el = str(el or "").strip().capitalize()
        try:
            amt = float(amt)
        except (TypeError, ValueError):
            continue
        if not el or amt <= 0:
            continue
        amt_str = "" if abs(amt - 1.0) < 1e-9 else f"{amt:g}"
        parts.append(f"{el}{amt_str}")
    return "".join(parts)


def substitution_series(formula: str, host: str, guest: str, steps: int = 11):
    """Substitute guest for host across the full range, preserving site count.

    Nd2Fe14B with Fe → Co yields Nd2Fe14B, Nd2Fe12.6Co1.4B, … Nd2Co14B, which is
    how a substitution study is actually set up.
    """
    pretty, amounts, err = parse_formula(formula)
    if err:
        return [], err
    if host not in amounts:
        return [], f"{host} is not in {pretty}."
    total = amounts[host]
    out = []
    for i in range(steps):
        x = i / (steps - 1) if steps > 1 else 0.0
        a = dict(amounts)
        a[host] = total * (1 - x)
        a[guest] = a.get(guest, 0.0) + total * x
        parts = [f"{el}{amt:g}" for el, amt in a.items() if amt > 1e-9]
        out.append((round(x, 4), "".join(parts)))
    return out, None


def predict_series(pairs, crystal_system=None, spacegroup=0):
    """Run the screen across a substitution series; returns a list of dicts."""
    cs = crystal_system or UNKNOWN
    rows = []
    for x, f in pairs:
        res, _alt, err = predict_auto(f, cs, spacegroup)
        if err or not res:
            continue
        tc = float(res.get("predicted_TC_K", 0) or 0)
        tn = float(res.get("predicted_TN_K", 0) or 0)
        phase = str(res.get("predicted_phase", "—"))
        rows.append(dict(
            x=x, formula=f, phase=phase,
            ordering_K=tn if phase == "AFM" else tc,
            criticality=float(res.get("crm_sustainability_penalty", 0) or 0),
        ))
    return rows


#  Reverse Design: Binary, Ternary & Multi-Element Systems
def generate_multicomponent_grid(elements: list, step: float = 0.05, n_dirichlet: int = 240) -> list:
    """Generate stoichiometric composition grid for 2, 3, 4, or 5+ elements."""
    k = len(elements)
    if k < 2:
        return []

    pts = []
    if k == 2:
        # Binary alloy sweep (21 points from 0% to 100% in 5% intervals)
        elA, elB = elements[0], elements[1]
        for i in range(21):
            b = round(i * 0.05, 3)
            a = round(1.0 - b, 3)
            parts = []
            if a > 1e-4:
                parts.append(f"{elA}{round(a * 100):g}")
            if b > 1e-4:
                parts.append(f"{elB}{round(b * 100):g}")
            f = "".join(parts) if parts else elA
            pts.append({elA: a, elB: b, "formula": f, "a_frac": a, "b_frac": b, "c_frac": 0.0})

    elif k == 3:
        # Standard ternary simplex grid (231 points at 5% step)
        elA, elB, elC = elements[0], elements[1], elements[2]
        n = 20
        for i in range(n + 1):
            for j in range(n + 1 - i):
                m = n - i - j
                a = round(i * 0.05, 3)
                b = round(j * 0.05, 3)
                c = round(m * 0.05, 3)
                parts = []
                if a > 1e-4:
                    parts.append(f"{elA}{round(a * 100):g}")
                if b > 1e-4:
                    parts.append(f"{elB}{round(b * 100):g}")
                if c > 1e-4:
                    parts.append(f"{elC}{round(c * 100):g}")
                f = "".join(parts) if parts else elA
                pts.append({elA: a, elB: b, elC: c, "formula": f, "a_frac": a, "b_frac": b, "c_frac": c})

    elif k == 4:
        # Quaternary simplex grid (step = 0.10, 286 points)
        elA, elB, elC, elD = elements[0], elements[1], elements[2], elements[3]
        n = 10
        for i in range(n + 1):
            for j in range(n + 1 - i):
                for m in range(n + 1 - i - j):
                    p = n - i - j - m
                    a = round(i * 0.10, 3)
                    b = round(j * 0.10, 3)
                    c = round(m * 0.10, 3)
                    d_val = round(p * 0.10, 3)
                    parts = []
                    if a > 1e-4:
                        parts.append(f"{elA}{round(a * 100):g}")
                    if b > 1e-4:
                        parts.append(f"{elB}{round(b * 100):g}")
                    if c > 1e-4:
                        parts.append(f"{elC}{round(c * 100):g}")
                    if d_val > 1e-4:
                        parts.append(f"{elD}{round(d_val * 100):g}")
                    f = "".join(parts) if parts else elA
                    pts.append({elA: a, elB: b, elC: c, elD: d_val, "formula": f,
                                "a_frac": a, "b_frac": b, "c_frac": c})

    else:
        # 5+ elements: Dirichlet uniform simplex sampling
        import numpy as np
        rng = np.random.RandomState(42)
        samples = rng.dirichlet(np.ones(k), size=n_dirichlet)
        for s in samples:
            parts = []
            d = {}
            for el, v in zip(elements, s):
                v_rnd = round(float(v), 3)
                d[el] = v_rnd
                if v_rnd > 1e-4:
                    parts.append(f"{el}{round(v_rnd * 100):g}")
            d["formula"] = "".join(parts) if parts else elements[0]
            d["a_frac"] = d[elements[0]]
            d["b_frac"] = d[elements[1]]
            d["c_frac"] = d[elements[2]] if k >= 3 else 0.0
            pts.append(d)

    return pts


def generate_ternary_grid(elA: str, elB: str, elC: str, step: float = 0.05):
    """Backward-compatible wrapper for ternary grid generation."""
    return generate_multicomponent_grid([elA, elB, elC], step=step)


@st.cache_data(show_spinner=False)
def screen_multicomponent_system(elements: tuple,
                                  crystal_system: str = "tetragonal",
                                  spacegroup: int = 136,
                                  step: float = 0.05) -> pd.DataFrame:
    """Screen an entire binary, ternary, or multi-component alloy space."""
    engine, err = get_engine()
    if engine is None:
        return pd.DataFrame()
    raw_grid = generate_multicomponent_grid(list(elements), step=step)
    if not raw_grid:
        return pd.DataFrame()
    eval_payload = [{
        "formula": r["formula"],
        "crystal_system": crystal_system,
        "spacegroup_number": spacegroup,
    } for r in raw_grid]
    df = engine.screen_candidates(eval_payload)
    # Merge element fractional coordinates
    for el in elements:
        df[f"{el}_frac"] = [r.get(el, 0.0) for r in raw_grid]
    df["a_frac"] = [r.get("a_frac", 0.0) for r in raw_grid]
    df["b_frac"] = [r.get("b_frac", 0.0) for r in raw_grid]
    df["c_frac"] = [r.get("c_frac", 0.0) for r in raw_grid]
    return df


@st.cache_data(show_spinner=False)
def screen_ternary_system(elA: str, elB: str, elC: str,
                          crystal_system: str = "tetragonal",
                          spacegroup: int = 136, step: float = 0.05) -> pd.DataFrame:
    """Backward-compatible wrapper for screening 3-element ternary systems."""
    return screen_multicomponent_system((elA, elB, elC), crystal_system=crystal_system, spacegroup=spacegroup)


#  Physical Schematic Generators
def make_hysteresis_fig(ms_tesla: float = 1.4, hc_ka_m: float = 600.0, kappa: float = 1.0, loop_mode: str = "Major Loop",
                        ideal_ha_ka_m: float = None):
    """
    Generate dynamic physical M-H magnetic hysteresis loop via Physics-Informed Neural Hysteresis Operator (DeepONet).
    Supported loop_modes: 'Major Loop', 'Minor Loops', 'Virgin Curve'.
    Optionally overlays single-crystal theoretical upper bound (ideal_ha_ka_m).
    """
    import numpy as np
    import plotly.graph_objects as go
    from src.neural_hysteresis import get_hysteresis_operator

    op = get_hysteresis_operator()
    h_pts, m_desc, m_asc = op.predict_major_loop(ms_tesla, hc_ka_m, kappa, n_points=160)
    hc = max(5.0, float(hc_ka_m))
    ms = max(0.05, float(ms_tesla))

    fig = go.Figure()

    # Optional Theoretical Single-Crystal Ceiling Curve
    if ideal_ha_ka_m and ideal_ha_ka_m > hc * 1.08:
        h_id, m_id_desc, _ = op.predict_major_loop(ms, ideal_ha_ka_m, kappa, n_points=120)
        fig.add_trace(go.Scatter(
            x=h_id, y=m_id_desc, name=f"Single-Crystal Ceiling (HA={ideal_ha_ka_m:.0f} kA/m)",
            line=dict(color="rgba(255, 255, 255, 0.45)", width=2.0, dash="dashdot")
        ))

    if loop_mode == "Virgin Curve":
        h_virg, m_virg = op.predict_virgin_curve(ms_tesla, hc_ka_m, kappa, n_points=100)
        fig.add_trace(go.Scatter(
            x=h_pts, y=m_desc, name="Major Descending Envelope",
            line=dict(color="rgba(0, 210, 255, 0.40)", width=2.0, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=h_pts, y=m_asc, name="Major Ascending Envelope",
            line=dict(color="rgba(255, 107, 0, 0.40)", width=2.0, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=h_virg, y=m_virg, name="Virgin Path M_init(H)",
            line=dict(color="#00F59B", width=4.0)
        ))
    elif loop_mode == "Minor Loops":
        fig.add_trace(go.Scatter(
            x=h_pts, y=m_desc, name="Major Descending",
            line=dict(color="#00D2FF", width=3.0)
        ))
        fig.add_trace(go.Scatter(
            x=h_pts, y=m_asc, name="Major Ascending",
            line=dict(color="#FF6B00", width=3.0)
        ))
        minor_loops = op.predict_minor_loops(ms_tesla, hc_ka_m, kappa, reversal_fractions=(0.25, 0.50, 0.75))
        minor_colors = ["#FFB703", "#A855F7", "#00F59B"]
        for idx, m_data in enumerate(minor_loops):
            col = minor_colors[idx % len(minor_colors)]
            fig.add_trace(go.Scatter(
                x=m_data["h"], y=m_data["m"],
                name=f"Minor Loop ({m_data['fraction_hc']*100:.0f}% Hc, Hrev={m_data['reversal_field_ka_m']:.0f} kA/m)",
                line=dict(color=col, width=2.5, dash="dot")
            ))
    else: # Major Loop
        fig.add_trace(go.Scatter(
            x=h_pts, y=m_desc, name="Demagnetization (Descending)",
            line=dict(color="#00D2FF", width=3.5)
        ))
        fig.add_trace(go.Scatter(
            x=h_pts, y=m_asc, name="Magnetization (Ascending)",
            line=dict(color="#FF6B00", width=3.5)
        ))
        # Second quadrant energy product region
        q2_mask = (h_pts >= -hc) & (h_pts <= 0)
        q2_h = h_pts[q2_mask]
        q2_m = m_desc[q2_mask]
        fig.add_trace(go.Scatter(
            x=q2_h, y=q2_m, name="(BH)ₘₐₓ Envelope",
            fill="tozeroy", fillcolor="rgba(255, 107, 0, 0.18)",
            line=dict(color="#FFB703", dash="dot", width=2)
        ))

    fig.add_vline(x=-hc, line_dash="dash", line_color="rgba(255, 51, 102, 0.45)")
    fig.add_vline(x=hc, line_dash="dash", line_color="#FF3366",
                  annotation_text=f"+Hc: {hc:.0f} kA/m", annotation_position="bottom right", annotation_font_color="#FF3366")
    fig.add_hline(y=-ms, line_dash="dot", line_color="rgba(0, 245, 155, 0.45)")
    fig.add_hline(y=ms, line_dash="dot", line_color="#00F59B",
                  annotation_text=f"Ms: {ms:.2f} T", annotation_position="top left", annotation_font_color="#00F59B")

    fig.update_layout(
        title="",
        xaxis=dict(
            title=dict(text="<b>Applied Magnetic Field H (kA/m)</b>", font=dict(size=13.5, color="#FFFFFF")),
            gridcolor="rgba(255, 255, 255, 0.14)",
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.28)",
            linecolor="rgba(255, 255, 255, 0.40)",
            tickfont=dict(size=12, color="#E2E8F0")
        ),
        yaxis=dict(
            title=dict(text="<b>Polarization J = μ₀M (Tesla)</b>", font=dict(size=13.5, color="#FFFFFF")),
            gridcolor="rgba(255, 255, 255, 0.14)",
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.28)",
            linecolor="rgba(255, 255, 255, 0.40)",
            tickfont=dict(size=12, color="#E2E8F0")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.90)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, color="#CBD5E1")
        ),
        margin=dict(l=55, r=25, t=18, b=68),
        height=390
    )
    return fig


def make_thermal_decay_fig(tc_k: float = 770.0, ms_tesla: float = 1.4, phase: str = "FM",
                           hc_ka_m: float = 600.0, thermal_mode: str = "Spontaneous Ms(T)"):
    """
    Generate thermal demagnetization and temperature-dependent magnetic order via PINO-Thermal Operator.
    Supported thermal_modes: 'Spontaneous Ms(T)', 'Reduced Curve m(t)', 'Thermal Coercivity Hc(T)'.
    """
    import numpy as np
    import plotly.graph_objects as go
    from src.neural_thermal import get_thermal_operator

    op = get_thermal_operator()
    prof = op.predict_thermal_profile(tc_k, ms_tesla, phase=phase, hc_ka_m=hc_ka_m, n_points=160)
    t_pts = prof["temperature_k"]
    tc = max(15.0, float(tc_k))

    fig = go.Figure()

    if thermal_mode == "Reduced Curve m(t)":
        fig.add_trace(go.Scatter(
            x=prof["reduced_t"], y=prof["reduced_m"], mode="lines",
            name="Reduced Magnetization m(t) = Ms(T)/Ms(0)",
            line=dict(color="#00D2FF", width=3.5),
            fill="tozeroy", fillcolor="rgba(0, 210, 255, 0.15)"
        ))
        x_title = "<b>Reduced Temperature t = T / Tᴄ</b>"
        y_title = "<b>Reduced Order Parameter m(t)</b>"
        fig.add_vline(x=1.0, line_dash="dash", line_color="#FF3366",
                      annotation_text="Transition Point (t = 1.0)", annotation_position="top right", annotation_font_color="#FF3366")
        if tc > 0:
            fig.add_vline(x=293.15 / tc, line_dash="dot", line_color="#00F59B",
                          annotation_text=f"RT (t = {293.15/tc:.2f})", annotation_position="bottom left", annotation_font_color="#00F59B")
    elif thermal_mode == "Thermal Coercivity Hc(T)":
        fig.add_trace(go.Scatter(
            x=t_pts, y=prof["hc_t"], mode="lines",
            name="Thermal Coercivity Decay Hc(T)",
            line=dict(color="#FF3366", width=3.5),
            fill="tozeroy", fillcolor="rgba(255, 51, 102, 0.15)"
        ))
        x_title = "<b>Temperature T (Kelvin)</b>"
        y_title = "<b>Intrinsic Coercivity Hc (kA/m)</b>"
        fig.add_vline(x=tc, line_dash="dash", line_color="#FF3366",
                      annotation_text=f"Curie Point: {tc:.0f} K", annotation_position="top right", annotation_font_color="#FF3366")
        fig.add_vline(x=293.15, line_dash="dot", line_color="#00D2FF",
                      annotation_text="RT (293 K)", annotation_position="bottom left", annotation_font_color="#00D2FF")
        fig.add_vline(x=400.0, line_dash="dot", line_color="#FFB703",
                      annotation_text="EV Target (400 K)", annotation_position="bottom right", annotation_font_color="#FFB703")
    else: # Spontaneous Ms(T)
        fig.add_trace(go.Scatter(
            x=t_pts, y=prof["ms_t"], mode="lines",
            name="Spontaneous Polarization Ms(T)",
            line=dict(color="#FFB703", width=3.5),
            fill="tozeroy", fillcolor="rgba(255, 183, 3, 0.15)"
        ))
        x_title = "<b>Temperature T (Kelvin)</b>"
        y_title = "<b>Sublattice Polarization (Tesla)</b>" if phase == "AFM" else "<b>Spontaneous Polarization μ₀Mₛ (Tesla)</b>"
        lbl_pt = f"Néel Point: {tc:.0f} K" if phase == "AFM" else f"Curie Point: {tc:.0f} K"
        fig.add_vline(x=tc, line_dash="dash", line_color="#FF3366",
                      annotation_text=lbl_pt, annotation_position="top right", annotation_font_color="#FF3366")
        fig.add_vline(x=293.15, line_dash="dot", line_color="#00D2FF",
                      annotation_text="RT (293 K)", annotation_position="bottom left", annotation_font_color="#00D2FF")
        fig.add_vline(x=400.0, line_dash="dot", line_color="#00F59B",
                      annotation_text="EV Target (400 K)", annotation_position="bottom right", annotation_font_color="#00F59B")

    fig.update_layout(
        title="",
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=13.5, color="#FFFFFF")),
            gridcolor="rgba(255, 255, 255, 0.14)",
            zeroline=False,
            linecolor="rgba(255, 255, 255, 0.40)",
            tickfont=dict(size=12, color="#E2E8F0")
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=13.5, color="#FFFFFF")),
            gridcolor="rgba(255, 255, 255, 0.14)",
            zeroline=False,
            linecolor="rgba(255, 255, 255, 0.40)",
            tickfont=dict(size=12, color="#E2E8F0")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.90)",
        showlegend=False,
        margin=dict(l=55, r=25, t=18, b=68),
        height=390
    )
    return fig


def make_spin_alignment_fig(phase: str = "FM"):
    """Generate visual schematic of microscopic spin lattice orientation."""
    import plotly.graph_objects as go
    fig = go.Figure()
    p = str(phase).upper()

    for r in range(3):
        for c in range(5):
            x0 = c * 1.5
            y0 = r * 1.5
            if p == "FM":
                dy = 0.85
                color = "#FF3366"
                fig.add_annotation(
                    x=x0, y=y0 + dy, ax=x0, ay=y0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=3, arrowsize=1.6, arrowwidth=3.2,
                    arrowcolor=color
                )
            elif p == "AFM":
                dy = 0.85 if (r + c) % 2 == 0 else -0.85
                color = "#00D2FF"
                fig.add_annotation(
                    x=x0, y=y0 + dy, ax=x0, ay=y0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=3, arrowsize=1.6, arrowwidth=3.2,
                    arrowcolor=color
                )
            else:
                color = "#A855F7"
                fig.add_shape(
                    type="circle", x0=x0 - 0.35, y0=y0 - 0.35, x1=x0 + 0.35, y1=y0 + 0.35,
                    fillcolor=color, line_color="#FFFFFF", line_width=1.5
                )

    sub_title = (
        "Ferromagnetic Exchange Coupling (Parallel Spin Alignment)" if p == "FM" else (
            "Antiferromagnetic Superexchange (Antiparallel Alternating Sublattices)" if p == "AFM" else (
                "Non-Magnetic Ground State (Paired Valence Shells, Net Zero Moment)"
            )
        )
    )
    fig.update_layout(
        title=f"<b>Microscopic Spin Configuration: {p} Phase</b><br><sup>{sub_title}</sup>",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 7]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 4]),
        height=220,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def make_voronoi_microstructure_fig(d_g_nm: float = 200.0, delta_gb_nm: float = 1.5,
                                    sigma_theta_deg: float = 12.0, seed: int = 42):
    """
    Generate an interactive 2D Voronoi polycrystal microstructure plot.
    Grains are rendered with their Inverse Pole Figure (IPF) orientation colors,
    and grain boundaries are drawn with thickness proportional to delta_gb.
    """
    import plotly.graph_objects as go
    from src.microstructure import VoronoiMicrostructureGenerator

    gen = VoronoiMicrostructureGenerator(n_grains=38, lloyd_iterations=2, seed=seed)
    micro = gen.generate(d_g_nm=d_g_nm, delta_gb_nm=delta_gb_nm, sigma_theta_deg=sigma_theta_deg)

    fig = go.Figure()

    # 1. Add polygon trace for each grain
    for g in micro["grains"]:
        poly = g["polygon"]
        x_pts = [p[0] * micro["box_size_nm"] for p in poly] + [poly[0][0] * micro["box_size_nm"]]
        y_pts = [p[1] * micro["box_size_nm"] for p in poly] + [poly[0][1] * micro["box_size_nm"]]
        c_x = g["centroid"][0] * micro["box_size_nm"]
        c_y = g["centroid"][1] * micro["box_size_nm"]
        theta = g["misorientation_deg"]

        fig.add_trace(go.Scatter(
            x=x_pts, y=y_pts, fill="toself",
            fillcolor=g["hex_color"],
            mode="lines",
            line=dict(color="#07090E", width=max(1.5, delta_gb_nm * 1.8)),
            name=f"Grain #{g['id']}",
            hoverinfo="text",
            hovertext=f"<b>Grain #{g['id']}</b><br>c-axis Misorientation: {theta:.1f}°<br>Area: {g['area_norm'] * (micro['box_size_nm']**2):.0f} nm²",
            showlegend=False
        ))

        # Grain orientation vector arrow
        arrow_len = 0.045 * micro["box_size_nm"]
        dx = arrow_len * math.sin(g["theta_rad"]) * math.cos(g["phi_rad"])
        dy = arrow_len * math.cos(g["theta_rad"])
        fig.add_annotation(
            x=c_x + dx, y=c_y + dy, ax=c_x, ay=c_y,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.8,
            arrowcolor="#FFFFFF", opacity=0.85
        )

    fig.update_layout(
        title=dict(
            text=f"<b>Polycrystalline Grain Morphology (2D Voronoi)</b><br><sup>N = {micro['num_grains']} grains | Mean dg = {d_g_nm:.0f} nm | GB Shell δGB = {delta_gb_nm:.1f} nm | Texture Spread σθ = {sigma_theta_deg:.1f}°</sup>",
            font=dict(color="#FFFFFF", size=13.5)
        ),
        xaxis=dict(
            title=dict(text="<b>X Position (nm)</b>", font=dict(size=12, color="#FFFFFF")),
            gridcolor="rgba(255, 255, 255, 0.10)",
            zeroline=False,
            scaleanchor="y", scaleratio=1,
            range=[0, micro["box_size_nm"]],
            tickfont=dict(color="#CBD5E1", size=11)
        ),
        yaxis=dict(
            title=dict(text="<b>Y Position (nm)</b>", font=dict(size=12, color="#FFFFFF")),
            gridcolor="rgba(255, 255, 255, 0.10)",
            zeroline=False,
            range=[0, micro["box_size_nm"]],
            tickfont=dict(color="#CBD5E1", size=11)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#07090E",
        margin=dict(l=50, r=20, t=60, b=45),
        height=380
    )
    return fig, micro


def compute_microstructure_derating(ms_tesla: float, k1_j_m3: float,
                                    d_g_nm: float = 200.0, delta_gb_nm: float = 1.5,
                                    sigma_theta_deg: float = 12.0) -> dict:
    """Compute Kronmüller derating metrics linking microstructure to magnetic properties."""
    from src.microstructure import MicromagneticKronmullerDerater
    return MicromagneticKronmullerDerater.derate(
        intrinsic_ms_tesla=ms_tesla,
        intrinsic_k1_j_m3=k1_j_m3,
        d_g_nm=d_g_nm,
        delta_gb_nm=delta_gb_nm,
        sigma_theta_deg=sigma_theta_deg
    )


# ══════════════════════════════════════════════════════════════════════
#  POLYNOMIAL RESPONSE SURFACE MODELING & INVERSE DESIGN
# ══════════════════════════════════════════════════════════════════════

def fit_polynomial_surface(
    df: pd.DataFrame,
    target_col: str,
    degree: int = 3,
    optimize_min: bool = False,
    grid_resolution: int = 120
) -> dict:
    """
    Fit a continuous bivariate response surface polynomial over composition coordinates.
    Z(x_A, x_B) = sum_{i+j <= degree} c_{ij} * (x_A)^i * (x_B)^j
    Evaluates on dense regular grid and identifies the global optimum peak (or minimum).
    """
    import numpy as np
    x = df["a_frac"].values.astype(float)
    y = df["b_frac"].values.astype(float)
    z = df[target_col].values.astype(float)

    # Build polynomial Vandermonde design matrix
    terms = []
    power_pairs = []
    for deg in range(degree + 1):
        for i in range(deg + 1):
            j = deg - i
            terms.append((x ** i) * (y ** j))
            power_pairs.append((i, j))
    X = np.column_stack(terms)

    # Least-squares fit
    coeffs, residuals, rank, s = np.linalg.lstsq(X, z, rcond=None)
    pred = X @ coeffs
    ss_tot = np.sum((z - np.mean(z)) ** 2)
    ss_res = np.sum((z - pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(np.mean((z - pred) ** 2)))

    # Regular Cartesian meshgrid
    gx = np.linspace(0.0, 1.0, grid_resolution)
    gy = np.linspace(0.0, 1.0, grid_resolution)
    GX, GY = np.meshgrid(gx, gy)
    simplex_mask = (GX + GY <= 1.0001)

    eval_terms = []
    for i, j in power_pairs:
        eval_terms.append((GX ** i) * (GY ** j))
    X_dense = np.stack(eval_terms, axis=-1)
    Z_dense = np.tensordot(X_dense, coeffs, axes=(-1, 0))
    Z_dense[~simplex_mask] = np.nan

    # Peak finding within valid simplex region
    valid = simplex_mask & (~np.isnan(Z_dense))
    if optimize_min:
        opt_idx = np.nanargmin(np.where(valid, Z_dense, np.inf))
    else:
        opt_idx = np.nanargmax(np.where(valid, Z_dense, -np.inf))
    opt_y, opt_x = np.unravel_index(opt_idx, Z_dense.shape)

    opt_a = float(gx[opt_x])
    opt_b = float(gy[opt_y])
    opt_c = max(0.0, float(1.0 - opt_a - opt_b))
    opt_z = float(Z_dense[opt_y, opt_x])

    return {
        "degree": degree,
        "r2": float(r2),
        "rmse": rmse,
        "coeffs": coeffs,
        "power_pairs": power_pairs,
        "gx": gx,
        "gy": gy,
        "Z_grid": Z_dense,
        "opt_a": opt_a,
        "opt_b": opt_b,
        "opt_c": opt_c,
        "opt_z": opt_z,
        "target_col": target_col,
        "optimize_min": optimize_min
    }


def make_ternary_phase_property_fig(
    tern_df: pd.DataFrame,
    poly_data: dict,
    el_A: str,
    el_B: str,
    el_C: str,
    target_label: str,
    target_col: str,
    unit: str,
    colorscale: str = "Turbo"
):
    """
    Generate an interactive Ternary Simplex displaying both magnetic ground state
    phase boundaries (FM vs AFM vs NM) and quantitative property distributions.
    """
    import plotly.graph_objects as go
    fig = go.Figure()

    is_phase_view = (target_col == "predicted_phase")

    # Phase styling configurations
    phase_cfg = {
        "FM": dict(symbol="triangle-up", size=13, color="#FF3366", name="Ferromagnetic (FM)", border="#FFFFFF"),
        "AFM": dict(symbol="diamond", size=12, color="#00D2FF", name="Antiferromagnetic (AFM)", border="#000000"),
        "NM": dict(symbol="circle", size=10, color="#A855F7", name="Non-Magnetic (NM)", border="#FFFFFF"),
    }

    cmin, cmax = 0.0, 1.0
    if not is_phase_view and target_col in tern_df:
        vals = tern_df[target_col].dropna()
        if len(vals):
            cmin = float(vals.min())
            cmax = float(vals.max())
            if abs(cmax - cmin) < 1e-4:
                cmax = cmin + 1.0

    has_colorbar = False
    for phase_key in ["FM", "AFM", "NM"]:
        pdf = tern_df[tern_df["predicted_phase"] == phase_key]
        if pdf.empty:
            continue

        cfg = phase_cfg[phase_key]
        n_pts = len(pdf)

        htext = []
        for _, r in pdf.iterrows():
            f_sub = to_subscript(str(r["formula"]))
            t_ord = float(r.get("predicted_TN_K", 0) if phase_key == "AFM" else r.get("predicted_TC_K", 0) or 0)
            ms_v = float(r.get("mu0_Ms_Tesla", 0) or 0)
            kp_v = float(r.get("hardness_kappa", 0) or 0)
            a_p = r["a_frac"] * 100
            b_p = r["b_frac"] * 100
            c_p = r["c_frac"] * 100
            htext.append(
                f"<b>Formula:</b> {f_sub}<br>"
                f"<b>Magnetic Phase:</b> {phase_key}<br>"
                f"<b>Ordering Temp:</b> {t_ord:.0f} K ({t_ord - 273.15:.0f} °C)<br>"
                f"<b>Saturation μ₀Mₛ:</b> {ms_v:.2f} T<br>"
                f"<b>Hardness κ:</b> {kp_v:.2f}<br>"
                f"<b>Composition:</b> {el_A}: {a_p:.1f}% | {el_B}: {b_p:.1f}% | {el_C}: {c_p:.1f}"
            )

        if is_phase_view:
            marker_dict = dict(
                symbol=cfg["symbol"],
                size=cfg["size"],
                color=cfg["color"],
                line=dict(color=cfg["border"], width=1.4)
            )
        else:
            show_scale = not has_colorbar
            has_colorbar = True
            marker_dict = dict(
                symbol=cfg["symbol"],
                size=cfg["size"],
                color=pdf[target_col],
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=show_scale,
                line=dict(color=cfg["color"], width=2.0)
            )
            if show_scale:
                marker_dict["colorbar"] = dict(
                    title=dict(text=f"<b>{unit}</b>", font=dict(color="#FFFFFF", size=13)),
                    tickfont=dict(color="#FFFFFF", size=11),
                    thickness=18,
                    len=0.88
                )

        fig.add_trace(go.Scatterternary(
            a=pdf["a_frac"],
            b=pdf["b_frac"],
            c=pdf["c_frac"],
            mode="markers",
            marker=marker_dict,
            name=f"{cfg['name']} ({n_pts})",
            hovertext=htext,
            hovertemplate="%{hovertext}<extra></extra>"
        ))

    # Global Peak / Optimum Marker
    if poly_data and not is_phase_view:
        oa = poly_data["opt_a"]
        ob = poly_data["opt_b"]
        oc = poly_data["opt_c"]
        oz = poly_data["opt_z"]
        opt_formula = f"{el_A}{round(oa*100):02d}{el_B}{round(ob*100):02d}{el_C}{round(oc*100):02d}"
        sub_opt = to_subscript(opt_formula)

        fig.add_trace(go.Scatterternary(
            a=[oa],
            b=[ob],
            c=[oc],
            mode="markers+text",
            marker=dict(symbol="star", size=24, color="#FFFF00", line=dict(color="#000000", width=2.5)),
            text=[f"<b>Peak: {sub_opt} ({oz:.2f} {unit})</b>"],
            textposition="top center",
            textfont=dict(color="#FFFF00", size=13, family="sans-serif"),
            name=f"Peak: {sub_opt}"
        ))

    title_text = (
        f"<b>Ternary Magnetic Phase Diagram ({el_A}–{el_B}–{el_C})</b><br><sup>Phase boundaries & magnetic ground state stability across 231 alloy compositions</sup>"
        if is_phase_view else
        f"<b>Ternary Composition Simplex ({el_A}–{el_B}–{el_C})</b><br><sup>Continuous {target_label} distribution with magnetic phase transitions (FM · AFM · NM)</sup>"
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(color="#FFFFFF", size=16.5)),
        ternary=dict(
            sum=1.0,
            aaxis=dict(title=f"<b>{el_A}</b>", min=0.0, linewidth=2.5, linecolor="#FF6B00", ticks="outside", tickfont=dict(color="#CBD5E1", size=11)),
            baxis=dict(title=f"<b>{el_B}</b>", min=0.0, linewidth=2.5, linecolor="#00D2FF", ticks="outside", tickfont=dict(color="#CBD5E1", size=11)),
            caxis=dict(title=f"<b>{el_C}</b>", min=0.0, linewidth=2.5, linecolor="#00F59B", ticks="outside", tickfont=dict(color="#CBD5E1", size=11)),
            bgcolor="rgba(15, 23, 42, 0.90)"
        ),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.08,
            xanchor="center", x=0.5,
            font=dict(color="#FFFFFF", size=12),
            bgcolor="rgba(17, 24, 39, 0.85)",
            bordercolor="rgba(255, 255, 255, 0.25)",
            borderwidth=1
        ),
        height=610,
        margin=dict(l=30, r=30, t=65, b=60),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def make_polynomial_3d_fig(
    poly_data: dict,
    tern_df: pd.DataFrame,
    el_A: str,
    el_B: str,
    el_C: str,
    target_label: str,
    target_col: str,
    unit: str,
    colorscale: str = "Turbo"
):
    """
    Generate an interactive 3D Topographical Property Landscape Surface with Phase Markers.
    """
    import plotly.graph_objects as go
    gx = poly_data["gx"]
    gy = poly_data["gy"]
    Z = poly_data["Z_grid"]
    oa = poly_data["opt_a"]
    ob = poly_data["opt_b"]
    oc = poly_data["opt_c"]
    oz = poly_data["opt_z"]
    r2 = poly_data["r2"]
    rmse = poly_data["rmse"]
    opt_formula = f"{el_A}{round(oa*100):02d}{el_B}{round(ob*100):02d}{el_C}{round(oc*100):02d}"
    sub_opt = to_subscript(opt_formula)

    fig = go.Figure()

    # 1. 3D Continuous Polynomial Response Surface
    fig.add_trace(go.Surface(
        x=gx, y=gy, z=Z,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title=dict(text=f"<b>{unit}</b>", font=dict(color="#FFFFFF", size=12)),
            tickfont=dict(color="#FFFFFF", size=10),
            thickness=15, len=0.75, x=1.02
        ),
        opacity=0.92,
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="#FFFFFF", project_z=True)
        ),
        hoverinfo="none",
        name="Polynomial Response Surface"
    ))

    # 2. Discrete Empirical Data Points Formatted in 3D Space
    phase_cfg = {
        "FM": dict(symbol="diamond", size=5.5, color="#FF3366", name="FM Data"),
        "AFM": dict(symbol="circle", size=5.0, color="#00D2FF", name="AFM Data"),
        "NM": dict(symbol="cross", size=4.5, color="#A855F7", name="NM Data"),
    }
    for p_key in ["FM", "AFM", "NM"]:
        pdf = tern_df[tern_df["predicted_phase"] == p_key]
        if pdf.empty:
            continue
        cfg = phase_cfg[p_key]
        htext = [
            f"<b>{to_subscript(str(r['formula']))}</b><br>{target_label}: {float(r[target_col]):.2f} {unit}<br>Phase: {p_key}"
            for _, r in pdf.iterrows()
        ]
        fig.add_trace(go.Scatter3d(
            x=pdf["a_frac"],
            y=pdf["b_frac"],
            z=pdf[target_col],
            mode="markers",
            marker=dict(size=cfg["size"], color=cfg["color"], line=dict(color="#FFFFFF", width=1)),
            name=cfg["name"],
            hovertext=htext,
            hovertemplate="%{hovertext}<extra></extra>"
        ))

    # 3. Peak / Optimum Marker in 3D
    fig.add_trace(go.Scatter3d(
        x=[oa], y=[ob], z=[oz],
        mode="markers+text",
        marker=dict(symbol="diamond", size=12, color="#FFFF00", line=dict(color="#000000", width=2)),
        text=[f"Peak: {sub_opt}"],
        textposition="top center",
        textfont=dict(color="#FFFF00", size=13),
        name=f"Peak: {sub_opt}"
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>3D Topographical Landscape ({el_A}–{el_B}–{el_C})</b><br><sup>Bivariate Degree-3 Response Surface (R² = {r2:.3f}, RMSE = {rmse:.2f}) with Empirical Phase Markers</sup>",
            font=dict(color="#FFFFFF", size=16.5)
        ),
        scene=dict(
            xaxis=dict(title=f"x ({el_A})", backgroundcolor="#07090E", gridcolor="rgba(255, 255, 255, 0.18)", color="#FFFFFF"),
            yaxis=dict(title=f"y ({el_B})", backgroundcolor="#07090E", gridcolor="rgba(255, 255, 255, 0.18)", color="#FFFFFF"),
            zaxis=dict(title=f"{target_label} ({unit})", backgroundcolor="#07090E", gridcolor="rgba(255, 255, 255, 0.18)", color="#FFFFFF"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=560,
        margin=dict(l=20, r=20, t=55, b=20)
    )
    return fig


def make_binary_phase_property_fig(
    b_df: pd.DataFrame,
    el_A: str,
    el_B: str,
    target_label: str,
    target_col: str,
    unit: str
):
    """
    Interactive 1D Binary Alloy Phase & Property Transition Diagram.
    """
    import plotly.graph_objects as go
    fig = go.Figure()

    is_phase_mode = (target_col == "predicted_phase")
    phase_order = {"NM": 0, "AFM": 1, "FM": 2}
    y_vals = [phase_order.get(p, 0) for p in b_df["predicted_phase"]] if is_phase_mode else b_df[target_col]

    # 1. Continuous smooth trend curve
    fig.add_trace(go.Scatter(
        x=b_df["b_frac"],
        y=y_vals,
        mode="lines",
        line=dict(color="rgba(255, 107, 0, 0.7)", width=3.5, shape="spline"),
        hoverinfo="none",
        name="Continuous Profile",
        showlegend=False
    ))

    # 2. Phase-delineated scatter points
    phase_cfg = {
        "FM": dict(symbol="triangle-up", size=14, color="#FF3366", name="Ferromagnetic (FM)"),
        "AFM": dict(symbol="diamond", size=13, color="#00D2FF", name="Antiferromagnetic (AFM)"),
        "NM": dict(symbol="circle", size=11, color="#A855F7", name="Non-Magnetic (NM)"),
    }

    for p_key in ["FM", "AFM", "NM"]:
        pdf = b_df[b_df["predicted_phase"] == p_key]
        if pdf.empty:
            continue
        cfg = phase_cfg[p_key]
        py = [phase_order[p_key]] * len(pdf) if is_phase_mode else pdf[target_col]

        htext = []
        for _, r in pdf.iterrows():
            f_sub = to_subscript(str(r["formula"]))
            t_ord = float(r.get("predicted_TN_K", 0) if p_key == "AFM" else r.get("predicted_TC_K", 0) or 0)
            ms_v = float(r.get("mu0_Ms_Tesla", 0) or 0)
            kp_v = float(r.get("hardness_kappa", 0) or 0)
            b_pct = r["b_frac"] * 100
            a_pct = 100.0 - b_pct
            htext.append(
                f"<b>Formula:</b> {f_sub}<br>"
                f"<b>Magnetic Phase:</b> {p_key}<br>"
                f"<b>Ordering Temp:</b> {t_ord:.0f} K ({t_ord - 273.15:.0f} °C)<br>"
                f"<b>Saturation μ₀Mₛ:</b> {ms_v:.2f} T<br>"
                f"<b>Hardness κ:</b> {kp_v:.2f}<br>"
                f"<b>Composition:</b> {el_A}: {a_pct:.1f}% | {el_B}: {b_pct:.1f}%"
            )

        fig.add_trace(go.Scatter(
            x=pdf["b_frac"],
            y=py,
            mode="markers",
            marker=dict(
                symbol=cfg["symbol"],
                size=cfg["size"],
                color=cfg["color"],
                line=dict(color="#FFFFFF", width=1.5)
            ),
            name=f"{cfg['name']} ({len(pdf)})",
            hovertext=htext,
            hovertemplate="%{hovertext}<extra></extra>"
        ))

    # 3. Peak optimum marker (when not phase mode)
    if not is_phase_mode and len(b_df):
        best_idx = b_df[target_col].idxmax() if target_col != "crm_sustainability_penalty" else b_df[target_col].idxmin()
        best_row = b_df.loc[best_idx]
        best_f = to_subscript(str(best_row["formula"]))
        best_val = float(best_row[target_col])
        fig.add_trace(go.Scatter(
            x=[best_row["b_frac"]],
            y=[best_val],
            mode="markers+text",
            marker=dict(symbol="star", size=24, color="#FFFF00", line=dict(color="#000000", width=2.5)),
            text=[f"<b>Peak: {best_f} ({best_val:.2f} {unit})</b>"],
            textposition="top center",
            textfont=dict(color="#FFFF00", size=13),
            name="Peak Optimum"
        ))

    y_axis_dict = dict(
        title=dict(text=f"<b>{target_label} ({unit})</b>", font=dict(color="#FFFFFF", size=14)),
        gridcolor="rgba(255, 255, 255, 0.15)",
        tickfont=dict(color="#CBD5E1", size=12)
    )
    if is_phase_mode:
        y_axis_dict["tickmode"] = "array"
        y_axis_dict["tickvals"] = [0, 1, 2]
        y_axis_dict["ticktext"] = ["Non-Magnetic (NM)", "Antiferromagnetic (AFM)", "Ferromagnetic (FM)"]
        y_axis_dict["range"] = [-0.3, 2.3]

    fig.update_layout(
        title=dict(
            text=f"<b>Binary Alloy Phase & Property Profile ({el_A}₁₋ₓ {el_B}ₓ)</b><br><sup>Compositional phase transition across x = 0.0 ({el_A}) to 1.0 ({el_B})</sup>",
            font=dict(color="#FFFFFF", size=16.5)
        ),
        xaxis=dict(
            title=dict(text=f"<b>Atomic Fraction x of {el_B} ({el_A}₁₋ₓ {el_B}ₓ)</b>", font=dict(color="#FFFFFF", size=14)),
            tickformat=".0%",
            range=[-0.02, 1.02],
            gridcolor="rgba(255, 255, 255, 0.15)",
            tickfont=dict(color="#CBD5E1", size=12)
        ),
        yaxis=y_axis_dict,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.03,
            xanchor="right", x=1.0,
            font=dict(color="#FFFFFF", size=12),
            bgcolor="rgba(17, 24, 39, 0.85)",
            bordercolor="rgba(255, 255, 255, 0.25)",
            borderwidth=1
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.90)",
        height=490,
        margin=dict(l=55, r=30, t=75, b=45)
    )
    return fig


def make_multicomponent_parcoords_fig(
    df: pd.DataFrame,
    elements: list,
    target_label: str,
    target_col: str,
    unit: str,
    colorscale: str = "Turbo"
):
    """
    Interactive Multi-Component Alloy Parallel Coordinates Space (4+ elements).
    Allows real-time multidimensional brushing across element concentrations.
    """
    import plotly.graph_objects as go

    is_phase_mode = (target_col == "predicted_phase")
    phase_order = {"NM": 0, "AFM": 1, "FM": 2}
    line_values = [phase_order.get(p, 0) for p in df["predicted_phase"]] if is_phase_mode else df[target_col]

    dims = []
    # Element dimensions
    for el in elements:
        cname = f"{el}_frac" if f"{el}_frac" in df else el
        vals = df[cname] if cname in df else [0.0] * len(df)
        max_val = max(0.2, float(vals.max()))
        dims.append(dict(
            range=[0.0, max_val],
            label=f"<b>{el} Fraction</b>",
            values=vals,
            tickformat=".0%"
        ))

    # Target property dimension
    if is_phase_mode:
        dims.append(dict(
            range=[0, 2],
            label="<b>Magnetic Phase</b>",
            values=line_values,
            tickvals=[0, 1, 2],
            ticktext=["NM", "AFM", "FM"]
        ))
        line_dict = dict(
            color=line_values,
            colorscale=[[0, "#A855F7"], [0.5, "#00D2FF"], [1.0, "#FF3366"]],
            cmin=0, cmax=2,
            showscale=True,
            colorbar=dict(
                title=dict(text="<b>Phase</b>", font=dict(color="#FFFFFF", size=13)),
                tickvals=[0, 1, 2],
                ticktext=["NM", "AFM", "FM"],
                tickfont=dict(color="#FFFFFF", size=11),
                thickness=18, len=0.85
            )
        )
    else:
        prop_vals = df[target_col]
        p_min = float(prop_vals.min())
        p_max = float(prop_vals.max())
        if abs(p_max - p_min) < 1e-4:
            p_max = p_min + 1.0
        dims.append(dict(
            range=[p_min, p_max],
            label=f"<b>{target_label} ({unit})</b>",
            values=prop_vals
        ))
        line_dict = dict(
            color=prop_vals,
            colorscale=colorscale,
            cmin=p_min, cmax=p_max,
            showscale=True,
            colorbar=dict(
                title=dict(text=f"<b>{unit}</b>", font=dict(color="#FFFFFF", size=13)),
                tickfont=dict(color="#FFFFFF", size=11),
                thickness=18, len=0.85
            )
        )

    fig = go.Figure(data=go.Parcoords(
        line=line_dict,
        dimensions=dims,
        labelangle=0,
        labelside="top"
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Multi-Component Alloy Parallel Coordinates ({'–'.join(elements)})</b><br><sup>Drag range sliders on any axis to filter high-performance compositions in multidimensional space</sup>",
            font=dict(color="#FFFFFF", size=15)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#07090E",
        height=520,
        margin=dict(l=60, r=40, t=85, b=40)
    )
    return fig


def make_polygonal_phase_property_fig(
    df: pd.DataFrame,
    elements: list,
    target_label: str,
    target_col: str,
    unit: str,
    colorscale: str = "Turbo"
):
    """
    Generate an interactive 2D Regular Polygon Composition Simplex:
    - 4 elements: Quadrilateral (Diamond / Square)
    - 5 elements: Regular Pentagon
    - 6 elements: Regular Hexagon
    - k elements: Regular k-gon
    Maps compositional coordinates via radial barycentric projection:
    P = sum_{i=0}^{k-1} x_i * [cos(theta_i), sin(theta_i)]
    Displays continuous property gradients and phase boundaries (FM / AFM / NM).
    """
    import numpy as np
    import plotly.graph_objects as go

    k = len(elements)
    is_phase_mode = (target_col == "predicted_phase")

    # Vertex angular coordinates (clockwise from top)
    angles = [np.pi / 2.0 - i * (2.0 * np.pi / k) for i in range(k)]
    vx = [float(np.cos(a)) for a in angles]
    vy = [float(np.sin(a)) for a in angles]

    # Pre-calculate projected (X, Y) coordinates for all points in df
    frac_cols = [f"{el}_frac" if f"{el}_frac" in df else el for el in elements]
    frac_matrix = df[frac_cols].values.astype(float)
    row_sums = np.sum(frac_matrix, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    frac_norm = frac_matrix / row_sums

    proj_x = np.dot(frac_norm, vx)
    proj_y = np.dot(frac_norm, vy)

    fig = go.Figure()

    # 1. Background radial spokes from center (0,0) to each vertex
    for i in range(k):
        fig.add_trace(go.Scatter(
            x=[0.0, vx[i]],
            y=[0.0, vy[i]],
            mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.16)", width=1.5, dash="dot"),
            hoverinfo="none",
            showlegend=False
        ))

    # 2. Concentric reference polygons (e.g. 25%, 50%, 75% scale)
    for r_scale in [0.25, 0.50, 0.75]:
        cx = [r_scale * x for x in vx] + [r_scale * vx[0]]
        cy = [r_scale * y for y in vy] + [r_scale * vy[0]]
        fig.add_trace(go.Scatter(
            x=cx, y=cy,
            mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.08)", width=1.0, dash="dash"),
            hoverinfo="none",
            showlegend=False
        ))

    # 3. Outer boundary polygon wireframe
    poly_wire_x = vx + [vx[0]]
    poly_wire_y = vy + [vy[0]]
    fig.add_trace(go.Scatter(
        x=poly_wire_x,
        y=poly_wire_y,
        mode="lines",
        line=dict(color="rgba(255, 255, 255, 0.45)", width=2.8),
        hoverinfo="none",
        name="Composition Boundary",
        showlegend=False
    ))

    # 4. Vertex elemental annotations
    vertex_colors = ["#FF6B00", "#00D2FF", "#00F59B", "#FF3366", "#A855F7", "#FFB703", "#E2E8F0"]
    for i, el in enumerate(elements):
        v_col = vertex_colors[i % len(vertex_colors)]
        tx = 1.18 * vx[i]
        ty = 1.18 * vy[i]
        fig.add_trace(go.Scatter(
            x=[vx[i]], y=[vy[i]],
            mode="markers",
            marker=dict(symbol="circle", size=14, color=v_col, line=dict(color="#FFFFFF", width=2)),
            hoverinfo="none",
            showlegend=False
        ))
        fig.add_annotation(
            x=tx, y=ty,
            text=f"<b>{el}</b>",
            showarrow=False,
            font=dict(size=15, color="#FFFFFF", family="sans-serif"),
            bgcolor="#111827",
            bordercolor=v_col,
            borderwidth=1.5,
            borderpad=4
        )

    # 5. Phase-delineated scatter points
    phase_cfg = {
        "FM": dict(symbol="triangle-up", size=12, color="#FF3366", name="Ferromagnetic (FM)", border="#FFFFFF"),
        "AFM": dict(symbol="diamond", size=11, color="#00D2FF", name="Antiferromagnetic (AFM)", border="#000000"),
        "NM": dict(symbol="circle", size=9, color="#A855F7", name="Non-Magnetic (NM)", border="#FFFFFF"),
    }

    cmin, cmax = 0.0, 1.0
    if not is_phase_mode and target_col in df:
        vals = df[target_col].dropna()
        if len(vals):
            cmin = float(vals.min())
            cmax = float(vals.max())
            if abs(cmax - cmin) < 1e-4:
                cmax = cmin + 1.0

    has_colorbar = False
    for p_key in ["FM", "AFM", "NM"]:
        mask = (df["predicted_phase"] == p_key).values
        if not np.any(mask):
            continue
        cfg = phase_cfg[p_key]
        n_pts = int(np.sum(mask))

        sub_df = df.iloc[mask]
        sub_x = proj_x[mask]
        sub_y = proj_y[mask]

        htext = []
        for _, r in sub_df.iterrows():
            f_sub = to_subscript(str(r["formula"]))
            t_ord = float(r.get("predicted_TN_K", 0) if p_key == "AFM" else r.get("predicted_TC_K", 0) or 0)
            ms_v = float(r.get("mu0_Ms_Tesla", 0) or 0)
            kp_v = float(r.get("hardness_kappa", 0) or 0)
            comp_strs = [f"{el}: {float(r.get(f'{el}_frac', r.get(el, 0.0))) * 100:.1f}%" for el in elements]
            htext.append(
                f"<b>Formula:</b> {f_sub}<br>"
                f"<b>Magnetic Phase:</b> {p_key}<br>"
                f"<b>Ordering Temp:</b> {t_ord:.0f} K ({t_ord - 273.15:.0f} °C)<br>"
                f"<b>Saturation μ₀Mₛ:</b> {ms_v:.2f} T<br>"
                f"<b>Hardness κ:</b> {kp_v:.2f}<br>"
                f"<b>Composition:</b> {' | '.join(comp_strs)}"
            )

        if is_phase_mode:
            marker_dict = dict(
                symbol=cfg["symbol"],
                size=cfg["size"],
                color=cfg["color"],
                line=dict(color=cfg["border"], width=1.4)
            )
        else:
            show_scale = not has_colorbar
            has_colorbar = True
            marker_dict = dict(
                symbol=cfg["symbol"],
                size=cfg["size"],
                color=sub_df[target_col],
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=show_scale,
                line=dict(color=cfg["color"], width=1.8)
            )
            if show_scale:
                marker_dict["colorbar"] = dict(
                    title=dict(text=f"<b>{unit}</b>", font=dict(color="#FFFFFF", size=13)),
                    tickfont=dict(color="#FFFFFF", size=11),
                    thickness=18,
                    len=0.88
                )

        fig.add_trace(go.Scatter(
            x=sub_x,
            y=sub_y,
            mode="markers",
            marker=marker_dict,
            name=f"{cfg['name']} ({n_pts})",
            hovertext=htext,
            hovertemplate="%{hovertext}<extra></extra>"
        ))

    # 6. Peak optimum marker (when not phase mode)
    if not is_phase_mode and len(df) and target_col in df:
        best_idx = df[target_col].idxmin() if target_col == "crm_sustainability_penalty" else df[target_col].idxmax()
        best_pos = df.index.get_loc(best_idx)
        bx = proj_x[best_pos]
        by = proj_y[best_pos]
        best_row = df.loc[best_idx]
        best_f = to_subscript(str(best_row["formula"]))
        best_val = float(best_row[target_col])
        fig.add_trace(go.Scatter(
            x=[bx],
            y=[by],
            mode="markers+text",
            marker=dict(symbol="star", size=24, color="#FFFF00", line=dict(color="#000000", width=2.5)),
            text=[f"<b>Peak: {best_f} ({best_val:.2f} {unit})</b>"],
            textposition="top center",
            textfont=dict(color="#FFFF00", size=13, family="sans-serif"),
            name=f"Peak: {best_f}"
        ))

    poly_name = "Quadrilateral" if k == 4 else ("Pentagon" if k == 5 else ("Hexagon" if k == 6 else f"{k}-Sided Polygon"))
    title_text = (
        f"<b>{poly_name} Magnetic Phase Diagram ({'–'.join(elements)})</b><br><sup>Regular {k}-gon barycentric projection showing magnetic phase stability across alloy composition space</sup>"
        if is_phase_mode else
        f"<b>{poly_name} Composition Simplex ({'–'.join(elements)})</b><br><sup>Barycentric {poly_name.lower()} projection with {target_label} gradient & magnetic phase transitions (FM · AFM · NM)</sup>"
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(color="#FFFFFF", size=15)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.45, 1.45]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.45, 1.45], scaleanchor="x", scaleratio=1),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.03,
            xanchor="right", x=1.0,
            font=dict(color="#FFFFFF", size=11),
            bgcolor="rgba(17, 24, 39, 0.85)",
            bordercolor="rgba(255, 255, 255, 0.25)",
            borderwidth=1
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#07090E",
        height=580,
        margin=dict(l=30, r=30, t=75, b=30)
    )
    return fig


def make_quaternary_tetrahedron_fig(
    df: pd.DataFrame,
    elements: list,
    target_label: str,
    target_col: str,
    unit: str,
    colorscale: str = "Turbo"
):
    """
    Generate an interactive 3D Quaternary Tetrahedron Simplex (Delta^3) for 4 elements:
    V_0 = (0, 0, 1)                      [Element 0]
    V_1 = (sqrt(8/9), 0, -1/3)           [Element 1]
    V_2 = (-sqrt(2/9),  sqrt(2/3), -1/3) [Element 2]
    V_3 = (-sqrt(2/9), -sqrt(2/3), -1/3) [Element 3]
    Renders the 3D regular tetrahedron wireframe and embeds alloy compositions in 3D volume.
    """
    import numpy as np
    import plotly.graph_objects as go

    if len(elements) != 4:
        return go.Figure()

    is_phase_mode = (target_col == "predicted_phase")

    v_tetra = np.array([
        [0.0, 0.0, 1.0],
        [np.sqrt(8.0 / 9.0), 0.0, -1.0 / 3.0],
        [-np.sqrt(2.0 / 9.0), np.sqrt(2.0 / 3.0), -1.0 / 3.0],
        [-np.sqrt(2.0 / 9.0), -np.sqrt(2.0 / 3.0), -1.0 / 3.0]
    ])

    frac_cols = [f"{el}_frac" if f"{el}_frac" in df else el for el in elements]
    frac_matrix = df[frac_cols].values.astype(float)
    row_sums = np.sum(frac_matrix, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    frac_norm = frac_matrix / row_sums

    tx = np.dot(frac_norm, v_tetra[:, 0])
    ty = np.dot(frac_norm, v_tetra[:, 1])
    tz = np.dot(frac_norm, v_tetra[:, 2])

    fig = go.Figure()

    # 1. 6 Wireframe edges of the regular tetrahedron
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1)]
    wire_x, wire_y, wire_z = [], [], []
    for u, v in edges:
        wire_x.extend([v_tetra[u, 0], v_tetra[v, 0], None])
        wire_y.extend([v_tetra[u, 1], v_tetra[v, 1], None])
        wire_z.extend([v_tetra[u, 2], v_tetra[v, 2], None])

    fig.add_trace(go.Scatter3d(
        x=wire_x, y=wire_y, z=wire_z,
        mode="lines",
        line=dict(color="rgba(255, 255, 255, 0.45)", width=4),
        hoverinfo="none",
        showlegend=False
    ))

    # 2. Apex corner markers & text labels
    vertex_colors = ["#FF6B00", "#00D2FF", "#00F59B", "#FF3366"]
    for i, el in enumerate(elements):
        fig.add_trace(go.Scatter3d(
            x=[v_tetra[i, 0]], y=[v_tetra[i, 1]], z=[v_tetra[i, 2]],
            mode="markers+text",
            marker=dict(symbol="diamond", size=9, color=vertex_colors[i], line=dict(color="#FFFFFF", width=2)),
            text=[f"<b>{el}</b>"],
            textposition="top center",
            textfont=dict(color="#FFFFFF", size=15),
            hoverinfo="none",
            showlegend=False
        ))

    # 3. 3D phase-delineated scatter points
    phase_3d_cfg = {
        "FM": dict(symbol="diamond", color="#FF3366", name="FM Phase"),
        "AFM": dict(symbol="square", color="#00D2FF", name="AFM Phase"),
        "NM": dict(symbol="circle", color="#A855F7", name="NM Phase"),
    }

    cmin, cmax = 0.0, 1.0
    if not is_phase_mode and target_col in df:
        vals = df[target_col].dropna()
        if len(vals):
            cmin = float(vals.min())
            cmax = float(vals.max())
            if abs(cmax - cmin) < 1e-4:
                cmax = cmin + 1.0

    has_colorbar = False
    for p_key in ["FM", "AFM", "NM"]:
        mask = (df["predicted_phase"] == p_key).values
        if not np.any(mask):
            continue
        p_cfg = phase_3d_cfg[p_key]
        n_pts = int(np.sum(mask))

        sub_df = df.iloc[mask]
        sub_x = tx[mask]
        sub_y = ty[mask]
        sub_z = tz[mask]

        htext = []
        for _, r in sub_df.iterrows():
            f_sub = to_subscript(str(r["formula"]))
            t_ord = float(r.get("predicted_TN_K", 0) if p_key == "AFM" else r.get("predicted_TC_K", 0) or 0)
            ms_v = float(r.get("mu0_Ms_Tesla", 0) or 0)
            kp_v = float(r.get("hardness_kappa", 0) or 0)
            comp_strs = [f"{el}: {float(r.get(f'{el}_frac', r.get(el, 0.0))) * 100:.1f}%" for el in elements]
            htext.append(
                f"<b>Formula:</b> {f_sub}<br>"
                f"<b>Magnetic Phase:</b> {p_key}<br>"
                f"<b>Ordering Temp:</b> {t_ord:.0f} K ({t_ord - 273.15:.0f} °C)<br>"
                f"<b>Saturation μ₀Mₛ:</b> {ms_v:.2f} T<br>"
                f"<b>Hardness κ:</b> {kp_v:.2f}<br>"
                f"<b>Composition:</b> {' | '.join(comp_strs)}"
            )

        if is_phase_mode:
            marker_dict = dict(
                symbol=p_cfg["symbol"],
                size=5.0,
                color=p_cfg["color"],
                line=dict(color="#000000", width=1.0)
            )
        else:
            show_scale = not has_colorbar
            has_colorbar = True
            marker_dict = dict(
                symbol=p_cfg["symbol"],
                size=5.5,
                color=sub_df[target_col],
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=show_scale,
                line=dict(color=p_cfg["color"], width=1.5)
            )
            if show_scale:
                marker_dict["colorbar"] = dict(
                    title=dict(text=f"<b>{unit}</b>", font=dict(color="#FFFFFF", size=13)),
                    tickfont=dict(color="#FFFFFF", size=11),
                    thickness=18,
                    len=0.85
                )

        fig.add_trace(go.Scatter3d(
            x=sub_x, y=sub_y, z=sub_z,
            mode="markers",
            marker=marker_dict,
            name=f"{p_cfg['name']} ({n_pts})",
            hovertext=htext,
            hovertemplate="%{hovertext}<extra></extra>"
        ))

    # 4. Peak optimum marker (when not phase mode)
    if not is_phase_mode and len(df) and target_col in df:
        best_idx = df[target_col].idxmin() if target_col == "crm_sustainability_penalty" else df[target_col].idxmax()
        best_pos = df.index.get_loc(best_idx)
        bx, by, bz = tx[best_pos], ty[best_pos], tz[best_pos]
        best_row = df.loc[best_idx]
        best_f = to_subscript(str(best_row["formula"]))
        best_val = float(best_row[target_col])
        fig.add_trace(go.Scatter3d(
            x=[bx], y=[by], z=[bz],
            mode="markers+text",
            marker=dict(symbol="diamond", size=10, color="#FFFF00", line=dict(color="#000000", width=2.5)),
            text=[f"Peak: {best_f} ({best_val:.2f} {unit})"],
            textposition="top center",
            textfont=dict(color="#FFFF00", size=13),
            name=f"Peak: {best_f}"
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>3D Quaternary Tetrahedron Simplex ({'–'.join(elements)})</b><br><sup>3D regular tetrahedron simplex Δ³ mapping {target_label} & magnetic phase transitions</sup>",
            font=dict(color="#FFFFFF", size=15)
        ),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor="#07090E", color="#FFFFFF"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor="#07090E", color="#FFFFFF"),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor="#07090E", color="#FFFFFF"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=560,
        margin=dict(l=20, r=20, t=65, b=20)
    )
    return fig


# 18. DAO Foundation Model Integration (Nature Communications 2026)

@st.cache_resource
def get_dao_engine():
    """Cache the DAOCrystalEngine singleton."""
    from src.dao_engine import DAOCrystalEngine
    return DAOCrystalEngine()


@st.cache_data(show_spinner=False)
def predict_dao_crystal(formula: str, crystal_system: str = None, predicted_phase: str = "FM") -> dict:
    """Predict 3D crystal structure, symmetry, Ehull, and quantum spin moments using DAO Foundation Model."""
    engine = get_dao_engine()
    return engine.predict_crystal(formula, crystal_system=crystal_system, predicted_phase=predicted_phase)


def make_3d_crystal_fig(structure, title: str = "3D Crystal Unit Cell"):
    """
    Generate an interactive Plotly 3D visualization of a crystal unit cell.
    Draws the real bounding box wireframe and atomic spheres with physical radii and quantum spin annotations.
    """
    import numpy as np
    import plotly.graph_objects as go
    from src.dao_engine import get_radius

    ELEMENT_PALETTE = {
        "Fe": "#E06666", "Co": "#3D85C6", "Ni": "#559955", "Nd": "#FF9900",
        "Sm": "#990099", "Pr": "#CC00FF", "Ce": "#E6B800", "Dy": "#FF007F",
        "B": "#00DDDD", "Pt": "#D0D0E0", "Mn": "#993366", "Al": "#AFAFAF",
        "Cu": "#C88033", "Ti": "#888888", "Cr": "#8A99C7", "V": "#A6A6AB",
        "Bi": "#9E51BA", "La": "#70D4FF", "Y": "#94FFFF", "Zr": "#00FF99",
        "Nb": "#73C2C9", "Mo": "#54B5B5", "Si": "#F0C8A0", "C": "#606060",
        "O": "#FF4D4D", "N": "#4D79FF", "S": "#FFCC00", "P": "#FF8000"
    }
    DEFAULT_COLORS = ["#00F59B", "#38BDF8", "#F472B6", "#A78BFA", "#FBBF24", "#34D399"]

    lat = structure.lattice
    va, vb, vc = lat.matrix[0], lat.matrix[1], lat.matrix[2]
    corners = [
        np.array([0, 0, 0]),
        va, va + vb, vb,
        vc, va + vc, va + vb + vc, vb + vc
    ]

    idx_seq = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, None, 1, 5, None, 2, 6, None, 3, 7]
    cx, cy, cz = [], [], []
    for idx in idx_seq:
        if idx is None:
            cx.append(None); cy.append(None); cz.append(None)
        else:
            cx.append(corners[idx][0])
            cy.append(corners[idx][1])
            cz.append(corners[idx][2])

    fig = go.Figure()

    # Wireframe unit cell
    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=cz,
        mode="lines",
        line=dict(color="#475569", width=3.5),
        name="Unit Cell Boundary",
        hoverinfo="none"
    ))

    # Group sites by species
    species_dict = {}
    has_mag = "magmom" in structure.site_properties
    mag_props = structure.site_properties.get("magmom", [])

    for i, site in enumerate(structure):
        sym = site.specie.symbol
        if sym not in species_dict:
            species_dict[sym] = {"x": [], "y": [], "z": [], "text": []}
        species_dict[sym]["x"].append(site.coords[0])
        species_dict[sym]["y"].append(site.coords[1])
        species_dict[sym]["z"].append(site.coords[2])
        fx, fy, fz = site.frac_coords[0], site.frac_coords[1], site.frac_coords[2]

        mag_str = ""
        if has_mag and i < len(mag_props):
            mv = mag_props[i]
            spin_dir = "+z" if mv > 0.05 else ("-z" if mv < -0.05 else "0")
            mag_str = f"<br>Quantum Spin Moment: <b>{mv:+.2f} μB</b> ({spin_dir})"

        species_dict[sym]["text"].append(
            f"<b>{sym}</b> (Site #{i+1}){mag_str}<br>"
            f"Cartesian: ({site.coords[0]:.2f}, {site.coords[1]:.2f}, {site.coords[2]:.2f}) Å<br>"
            f"Fractional: ({fx:.3f}, {fy:.3f}, {fz:.3f})"
        )

    for idx, (sym, data) in enumerate(species_dict.items()):
        color = ELEMENT_PALETTE.get(sym, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
        rad = get_radius(sym)
        marker_size = max(10, min(24, int(rad * 10)))
        fig.add_trace(go.Scatter3d(
            x=data["x"], y=data["y"], z=data["z"],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=color,
                opacity=0.92,
                line=dict(color="#000000", width=1.5)
            ),
            name=f"{sym} ({len(data['x'])} sites)",
            text=data["text"],
            hovertemplate="%{text}<extra></extra>"
        ))

    # 3D Quantum Spin Vectors (MSN Layer)
    if has_mag:
        spin_line_x, spin_line_y, spin_line_z = [], [], []
        spin_arrow_x, spin_arrow_y, spin_arrow_z = [], [], []
        spin_arrow_u, spin_arrow_v, spin_arrow_w = [], [], []

        for i, site in enumerate(structure):
            if i < len(mag_props):
                mv = float(mag_props[i])
                if abs(mv) > 0.05:
                    x0, y0, z0 = site.coords[0], site.coords[1], site.coords[2]
                    arrow_len = 0.85 + min(0.9, abs(mv) / 6.0)
                    dz = arrow_len if mv > 0 else -arrow_len
                    spin_line_x.extend([x0, x0, None])
                    spin_line_y.extend([y0, y0, None])
                    spin_line_z.extend([z0, z0 + dz, None])
                    spin_arrow_x.append(x0)
                    spin_arrow_y.append(y0)
                    spin_arrow_z.append(z0 + dz)
                    spin_arrow_u.append(0.0)
                    spin_arrow_v.append(0.0)
                    spin_arrow_w.append(0.4 if mv > 0 else -0.4)

        if spin_line_x:
            fig.add_trace(go.Scatter3d(
                x=spin_line_x, y=spin_line_y, z=spin_line_z,
                mode="lines",
                line=dict(color="#38BDF8", width=4.5),
                name="MSN Spin Vector",
                hoverinfo="none",
                showlegend=False
            ))
            fig.add_trace(go.Cone(
                x=spin_arrow_x, y=spin_arrow_y, z=spin_arrow_z,
                u=spin_arrow_u, v=spin_arrow_v, w=spin_arrow_w,
                sizemode="absolute",
                sizeref=0.45,
                anchor="tip",
                colorscale=[[0, "#38BDF8"], [1, "#00F59B"]],
                showscale=False,
                name="MSN Spin Vectors"
            ))


    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sup>Lattice: a={lat.a:.2f}Å, b={lat.b:.2f}Å, c={lat.c:.2f}Å | α={lat.alpha:.1f}°, β={lat.beta:.1f}°, γ={lat.gamma:.1f}°</sup>",
            font=dict(color="#FFFFFF", size=14)
        ),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X (Å)", showgrid=True, gridcolor="#1E293B", zeroline=False, color="#94A3B8"),
            yaxis=dict(title="Y (Å)", showgrid=True, gridcolor="#1E293B", zeroline=False, color="#94A3B8"),
            zaxis=dict(title="Z (Å)", showgrid=True, gridcolor="#1E293B", zeroline=False, color="#94A3B8"),
            bgcolor="#07090E"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(
            font=dict(color="#E2E8F0", size=11),
            bgcolor="rgba(15,23,42,0.7)",
            bordercolor="#334155",
            borderwidth=1,
            x=0.02, y=0.98
        )
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENTAL DATABASE & DATA EXPLORER ACCESS
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_experimental_master_table() -> pd.DataFrame:
    """Load and preprocess the unified master experimental dataset (51,394 compounds)."""
    p = os.path.join(ROOT, "output", "featurization", "features_master.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, low_memory=False)
        df_clean = df.dropna(subset=["reduced_formula"]).copy()

        # Standardize phase
        phase_map = {0.0: "FM", 1.0: "AFM", 2.0: "NM"}
        df_clean["Phase"] = df_clean["Type"].map(phase_map).fillna("Unassigned")

        # Standardize crystal system
        systems = ["cubic", "tetragonal", "hexagonal", "trigonal", "orthorhombic", "monoclinic", "triclinic"]
        def extract_cs(r):
            for s in systems:
                if r.get(f"crystal_system_{s}", 0) == 1:
                    return s.capitalize()
            raw = str(r.get("Crystal_Structure", "")).strip()
            if raw and raw.lower() != "nan":
                return raw.capitalize()
            return "Unspecified"

        df_clean["Crystal_System"] = [extract_cs(r) for _, r in df_clean.iterrows()]

        # Key physical properties with clean numeric parsing
        df_clean["Formula"] = df_clean["reduced_formula"]
        df_clean["Formula_Sub"] = df_clean["reduced_formula"].apply(to_subscript)
        df_clean["TC_K"] = pd.to_numeric(df_clean["Mean_TC_K"], errors="coerce")
        df_clean["TN_K"] = pd.to_numeric(df_clean["Mean_TN_K"], errors="coerce")
        df_clean["Hc_kA_m"] = pd.to_numeric(df_clean["clean_Hc_A_m"], errors="coerce") / 1000.0
        df_clean["Ms_emu_g"] = pd.to_numeric(df_clean["clean_Ms_emu_g"], errors="coerce")
        df_clean["K1_MJ_m3"] = pd.to_numeric(df_clean["clean_K1_J_m3"], errors="coerce") / 1e6
        df_clean["BH_max_kJ_m3"] = pd.to_numeric(df_clean["clean_BH_max_kJ_m3"], errors="coerce")
        df_clean["Theta_p_K"] = pd.to_numeric(df_clean["clean_theta_p_K"], errors="coerce")
        df_clean["SpaceGroup"] = pd.to_numeric(df_clean.get("spacegroup_number", 0), errors="coerce").fillna(0).astype(int)

        # Descriptors
        df_clean["Slater_Pauling"] = pd.to_numeric(df_clean.get("valence_electron_concentration", 0), errors="coerce")
        df_clean["Aniso_FOM"] = pd.to_numeric(df_clean.get("anisotropy_figure_of_merit", 0), errors="coerce")
        df_clean["Vol_per_Atom"] = pd.to_numeric(df_clean.get("dao_volume_per_atom", 0), errors="coerce")

        return df_clean
    except Exception as e:
        print(f"Error loading experimental master: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_mp_structures_dataframe() -> pd.DataFrame:
    """Load Materials Project DFT crystal structures catalog."""
    p = os.path.join(ROOT, "Dataset", "mp_alloys_point_groups.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    try:
        df = pd.read_csv(
            p,
            usecols=[
                "material_id", "formula", "reduced_formula", "crystal_system",
                "spacegroup_number", "spacegroup_symbol", "energy_above_hull",
                "formation_energy_per_atom", "band_gap", "total_magnetization"
            ],
            low_memory=False
        )
        df["Formula"] = df["formula"]
        df["Formula_Sub"] = df["formula"].apply(to_subscript)
        df["crystal_system"] = df["crystal_system"].str.capitalize()
        df["Phase"] = df["total_magnetization"].apply(
            lambda m: "FM" if m > 0.05 else ("NM" if m == 0.0 else "Weak/AFM")
        )
        return df
    except Exception as e:
        print(f"Error loading MP structures: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_experimental_catalogue_dict() -> dict:
    """Dictionary of available curated experimental datasets with descriptions."""
    return {
        "Master Experimental Dataset (56.8k)": {
            "key": "master",
            "desc": "Unified repository of 56,874 inorganics with experimental phase, transitions, coercivity & anisotropy",
            "count": 56874,
            "unit_col": "Phase & Multiple Targets"
        },
        "NovoMag RE-Free Permanent Magnets (124)": {
            "key": "novomag",
            "desc": "124 high-anisotropy rare-earth-free permanent magnets from Ames Lab (Fe16N2, Fe-Co-N, MnBi, FePt, Co-Zr)",
            "count": 124,
            "unit_col": "Hc, K1, (BH)max, Ms, TC"
        },
        "NEMAD Ground State Phase (35.1k)": {
            "key": "phase",
            "desc": "35,138 experimental compounds classified into FM (16,934), AFM (7,068), NM (11,136)",
            "count": 35138,
            "unit_col": "FM / AFM / NM"
        },
        "NEMAD Curie Temperature Tc (15.6k)": {
            "key": "curie",
            "desc": "15,611 ferromagnetic Curie transitions measured via SQUID, VSM and Faraday balance",
            "count": 15611,
            "unit_col": "Kelvin (K)"
        },
        "NEMAD Néel Temperature Tn (7.7k)": {
            "key": "neel",
            "desc": "7,734 antiferromagnetic Néel transitions verified by neutron scattering and specific heat",
            "count": 7734,
            "unit_col": "Kelvin (K)"
        },
        "Coercivity & Magnetic Hardness (10.1k)": {
            "key": "coercivity",
            "desc": "10,075 experimental coercivity measurements across soft, semi-hard, and permanent magnets",
            "count": 10075,
            "unit_col": "kA/m (log₁₀)"
        },
        "Saturation Magnetization Ms (10.6k)": {
            "key": "magnetization",
            "desc": "10,580 experimental saturation magnetization values measured at 4.2 K and 300 K",
            "count": 10580,
            "unit_col": "emu/g"
        },
        "Magnetocrystalline Anisotropy K1 (4.7k)": {
            "key": "anisotropy",
            "desc": "4,687 single-crystal torque magnetometry, SPD and DFT anisotropy measurements",
            "count": 4687,
            "unit_col": "MJ/m³"
        },
        "Maximum Energy Product (BH)max (1.75k)": {
            "key": "bh_max",
            "desc": "1,751 second-quadrant B-H demagnetization energy products for permanent magnets",
            "count": 1751,
            "unit_col": "kJ/m³"
        },
        "Materials Project DFT Structures (132k)": {
            "key": "mp_structures",
            "desc": "132,336 DFT-relaxed crystal structures with 230 space groups, point groups, and E_hull",
            "count": 132336,
            "unit_col": "Space Groups (1–230)"
        }
    }


def query_experimental_slice(dataset_key: str) -> pd.DataFrame:
    """Retrieve filtered dataframe for a specific experimental target slice."""
    if dataset_key == "mp_structures":
        return load_mp_structures_dataframe()

    if dataset_key == "novomag":
        novomag_path = os.path.join(ROOT, "Dataset", "novomag_refree_magnets.csv")
        if os.path.exists(novomag_path):
            nov_df = pd.read_csv(novomag_path)
            nov_df["Formula"] = nov_df["reduced_formula"]
            nov_df["Formula_Sub"] = nov_df["reduced_formula"].apply(to_subscript)
            nov_df["TC_K"] = pd.to_numeric(nov_df["Mean_TC_K"], errors="coerce")
            nov_df["TN_K"] = np.nan
            nov_df["Hc_kA_m"] = pd.to_numeric(nov_df["clean_Hc_A_m"], errors="coerce") / 1000.0
            nov_df["Ms_emu_g"] = pd.to_numeric(nov_df["clean_Ms_emu_g"], errors="coerce")
            nov_df["K1_MJ_m3"] = pd.to_numeric(nov_df["clean_K1_J_m3"], errors="coerce") / 1e6
            nov_df["BH_max_kJ_m3"] = pd.to_numeric(nov_df["clean_BH_max_kJ_m3"], errors="coerce")
            nov_df["Phase"] = "FM"
            nov_df["Crystal_System"] = nov_df["crystal_system"].astype(str).str.capitalize()
            nov_df["SpaceGroup"] = pd.to_numeric(nov_df["spacegroup_number"], errors="coerce").fillna(0).astype(int)
            return nov_df

    df = load_experimental_master_table()
    if df.empty:
        return df

    if dataset_key == "phase":
        return df[df["Phase"].isin(["FM", "AFM", "NM"])].copy()
    elif dataset_key == "curie":
        return df[df["TC_K"].notna()].copy()
    elif dataset_key == "neel":
        return df[df["TN_K"].notna()].copy()
    elif dataset_key == "coercivity":
        return df[df["Hc_kA_m"].notna()].copy()
    elif dataset_key == "magnetization":
        return df[df["Ms_emu_g"].notna()].copy()
    elif dataset_key == "anisotropy":
        return df[df["K1_MJ_m3"].notna()].copy()
    elif dataset_key == "bh_max":
        return df[df["BH_max_kJ_m3"].notna()].copy()
    return df


def screen_batch_candidates(candidates_list: list) -> pd.DataFrame:
    """
    Screens a batch of candidates using GenerativeCrystalScreeningEngine.
    """
    engine, err = get_engine()
    if engine is None or not candidates_list:
        return pd.DataFrame()
    try:
        return engine.screen_candidates(candidates_list)
    except Exception as e:
        print(f"Batch screening error: {e}")
        return pd.DataFrame()


def generate_vasp_poscar_and_incar(formula: str, crystal_system: str = None, dao_res: dict = None) -> Tuple[str, str]:
    """
    Generates VASP POSCAR atomic structure and initialized magnetic INCAR template
    for high-throughput DFT validation.
    """
    try:
        from pymatgen.io.vasp import Poscar
        from pymatgen.core import Composition
        if dao_res is None:
            dao_res = predict_dao_crystal(formula, crystal_system=crystal_system if crystal_system != UNKNOWN else None)
        struct = dao_res.get("structure")
        if struct is None:
            return "", ""

        poscar_str = Poscar(struct, comment=f"MagMat_{formula}_{dao_res.get('spacegroup_symbol', '')}").get_str()

        # Build site-specific MAGMOM string according to POSCAR element order
        SPIN_MOMENTS = {
            'Fe': 2.22, 'Co': 1.72, 'Ni': 0.61, 'Mn': 3.00, 'Cr': 2.50,
            'Nd': 3.20, 'Sm': 1.50, 'Dy': 10.0, 'Tb': 9.00, 'Gd': 7.00,
            'Pr': 3.20, 'Ho': 10.0, 'Er': 9.00, 'Tm': 7.00, 'Yb': 4.00,
            'Pt': 0.10, 'Pd': 0.05, 'V': 0.50, 'Ti': 0.00
        }
        
        # Order of species in POSCAR
        species_counts = []
        curr_spec = None
        curr_cnt = 0
        for site in struct:
            sym = site.specie.symbol
            if sym == curr_spec:
                curr_cnt += 1
            else:
                if curr_spec is not None:
                    species_counts.append((curr_spec, curr_cnt))
                curr_spec = sym
                curr_cnt = 1
        if curr_spec is not None:
            species_counts.append((curr_spec, curr_cnt))

        magmom_parts = []
        for sym, count in species_counts:
            mom = SPIN_MOMENTS.get(sym, 0.0)
            magmom_parts.append(f"{count}*{mom:.2f}")
        magmom_str = " ".join(magmom_parts)

        cs_str = dao_res.get('crystal_system', 'unspecified').title()
        sg_str = dao_res.get('spacegroup_symbol', '')
        phase_str = dao_res.get('predicted_phase', 'FM')

        incar_str = f"""# VASP INCAR Template for Magnetic Ground State & Anisotropy Calculation
# Generated by MagMat Discovery Platform
# Formula: {formula} | Symmetry: {cs_str} ({sg_str}) | Phase: {phase_str}
SYSTEM = MagMat_{formula}

# Electronic Relaxation
ENCUT    = 520
PREC     = Accurate
EDIFF    = 1E-6
ISMEAR   = 0
SIGMA    = 0.05
ALGO     = Fast
LREAL    = Auto
LWAVE    = .FALSE.
LCHARG   = .TRUE.

# Collinear Spin Polarization
ISPIN    = 2
MAGMOM   = {magmom_str}

# Optional Spin-Orbit Coupling & Magnetocrystalline Anisotropy (MAE):
# Step 1: Converge collinear charge density with above settings.
# Step 2: Uncomment below lines and rerun with ICHARG = 11 to evaluate MAE:
# LNONCOLLINEAR = .TRUE.
# LSORBIT       = .TRUE.
# SAXIS         = 0 0 1   # Easy axis candidate (e.g. c-axis [001] vs in-plane [100])
# NBANDS        = {len(struct) * 16}

# Ionic Relaxation (Structure Optimization)
IBRION   = 2
NSW      = 100
EDIFFG   = -0.01
ISIF     = 3
"""
        return poscar_str, incar_str
    except Exception as e:
        print(f"Error generating POSCAR/INCAR: {e}")
        return "", ""
