"""
MagMat-Concept: Physics-Grounded Screening & Rapid Inverse Discovery Studio.

Combines fundamental condensed-matter boundaries (Slater-Pauling atomic band ceiling,
Stoner quantum exchange, Kronmüller micromagnetic scaling, and Maxwell energy limits)
with live inference from pretrained 3D foundation models (DAO + MACE) and in-house
trained multi-task machine learning regressors.

Author: Sachin Poudel
Run:  streamlit run app/concept_app.py
"""

import base64
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Direct local imports within MagMat-Discovery/app/
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import data as D
import theme as T

# Physical Constants
MU0 = 4.0 * math.pi * 1e-7          # H/m
BOHR_MAGNETON_PER_MOL = 5585.0      # emu*g / (mol*mu_B)

# Standard Valence Electron Count (VEC) and Atomic Masses for 3d & key elements
ELEMENT_VEC = {
    "H": 1, "He": 2, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Ne": 8,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 8,
    "K": 1, "Ca": 2, "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10,
    "Cu": 11, "Zn": 12, "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7, "Kr": 8,
    "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7, "Ru": 8, "Rh": 9, "Pd": 10,
    "Ag": 11, "Cd": 12, "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7, "Xe": 8,
    "Cs": 1, "Ba": 2, "La": 3, "Ce": 3, "Pr": 3, "Nd": 3, "Pm": 3, "Sm": 3, "Eu": 2, "Gd": 3,
    "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 2, "Lu": 3, "Hf": 4, "Ta": 5, "W": 6,
    "Re": 7, "Os": 8, "Ir": 9, "Pt": 10, "Au": 11, "Hg": 12, "Tl": 3, "Pb": 4, "Bi": 5
}

ELEMENT_MASS = {
    "H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "Al": 26.982, "Si": 28.085,
    "P": 30.974, "S": 32.06, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.63,
    "Zr": 91.224, "Nb": 92.906, "Mo": 95.95, "Pd": 106.42, "Ag": 107.87, "Sn": 118.71,
    "Nd": 144.24, "Sm": 150.36, "Dy": 162.50, "Tb": 158.93, "Pr": 140.91, "Ce": 140.12,
    "Pt": 195.08, "Au": 196.97, "Bi": 208.98
}

MAGNETIC_3D_ELEMENTS = {"Fe", "Co", "Ni", "Mn", "Cr", "V"}
RARE_EARTH_ELEMENTS = {
    "Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"
}


# =========================================================================
# Classical Condensed-Matter Physics Functions
# =========================================================================

def slater_pauling_moment(vec: float) -> float:
    """Calculates theoretical Slater-Pauling magnetic moment (mu_B/atom)."""
    if vec is None or math.isnan(vec) or vec <= 0:
        return 0.0
    if vec >= 8.3:
        return max(0.0, 10.7 - vec)
    elif vec >= 5.5:
        return max(0.0, 2.45 - (8.3 - vec) * 0.875)
    return 0.0


def slater_pauling_ms(vec: float, avg_mass: float) -> float:
    """Converts Slater-Pauling moment to saturation magnetization (emu/g)."""
    if avg_mass <= 0:
        return 0.0
    mu = slater_pauling_moment(vec)
    return (mu * BOHR_MAGNETON_PER_MOL) / max(avg_mass, 10.0)


def emu_to_tesla(ms_emu_g: float, density_g_cm3: float = 7.8) -> float:
    """Converts Ms in emu/g to SI magnetic polarization mu0*Ms in Tesla."""
    if ms_emu_g <= 0 or density_g_cm3 <= 0:
        return 0.0
    return (4.0 * math.pi * density_g_cm3 * ms_emu_g) / 10000.0


def kronmuller_ideal_coercivity(k1_j_m3: float, ms_emu_g: float, density_g_cm3: float = 7.8) -> float:
    """Calculates theoretical Brown-Kronmüller nucleation limit Hk = 2*K1 / (mu0*Ms) in A/m."""
    if k1_j_m3 <= 0 or ms_emu_g <= 0:
        return 0.0
    mu0_ms = emu_to_tesla(ms_emu_g, density_g_cm3)
    if mu0_ms <= 0:
        return 0.0
    return (2.0 * k1_j_m3) / mu0_ms


def bh_max_theoretical_limit(ms_emu_g: float, density_g_cm3: float = 7.8) -> float:
    """Calculates thermodynamic energy ceiling (BH)max <= 1/4 * mu0 * Ms^2 in kJ/m^3."""
    mu0_ms = emu_to_tesla(ms_emu_g, density_g_cm3)
    if mu0_ms <= 0:
        return 0.0
    return (mu0_ms ** 2) / (4.0 * MU0) * 1e-3


def evaluate_stoichiometry_physics(amounts: Dict[str, float], density_g_cm3: float = 7.8) -> Dict[str, Any]:
    """Calculates analytical condensed-matter physical boundary metrics for a composition dict."""
    tot_atoms = sum(amounts.values())
    if tot_atoms <= 0:
        return {}

    vec_sum = sum(ELEMENT_VEC.get(el, 4.0) * amt for el, amt in amounts.items())
    mass_sum = sum(ELEMENT_MASS.get(el, 55.0) * amt for el, amt in amounts.items())
    mag_3d_atoms = sum(amt for el, amt in amounts.items() if el in MAGNETIC_3D_ELEMENTS)
    re_atoms = sum(amt for el, amt in amounts.items() if el in RARE_EARTH_ELEMENTS)

    vec = vec_sum / tot_atoms
    avg_mass = mass_sum / tot_atoms
    f_3d = mag_3d_atoms / tot_atoms
    f_re = re_atoms / tot_atoms

    sp_mu = slater_pauling_moment(vec)
    sp_ms_emu = slater_pauling_ms(vec, avg_mass)
    sp_ms_t = emu_to_tesla(sp_ms_emu, density_g_cm3)
    bh_ceiling_kj = bh_max_theoretical_limit(sp_ms_emu, density_g_cm3)

    # Stoner exchange score
    vec_penalty = math.exp(-((vec - 8.3) ** 2) / 4.0)
    stoner_score = float(np.clip(f_3d * vec_penalty, 0.0, 1.0))

    return {
        "vec": vec,
        "avg_mass": avg_mass,
        "f_3d": f_3d,
        "f_re": f_re,
        "is_re_free": f_re == 0.0,
        "sp_moment_mu_b": sp_mu,
        "sp_ms_emu_g": sp_ms_emu,
        "sp_ms_tesla": sp_ms_t,
        "bh_max_ceiling_kj_m3": bh_ceiling_kj,
        "stoner_score": stoner_score,
    }


# =========================================================================
# Reality Gap Statistics from Harmonized Experimental Data
# =========================================================================

@st.cache_data(show_spinner=False)
def get_reality_gap_dataset_stats() -> Dict[str, Any]:
    """Computes Kronmüller alpha distribution and Maxwell bounds over experimental records."""
    df = D.load_experimental_master_table()
    if df.empty:
        return {
            "total_records": 51487, "alpha_median": 0.110, "alpha_n": 1064,
            "alpha_values": np.array([0.11]), "bh_within_bound_pct": 71.7,
            "tc_count": 18860, "ms_count": 10131, "hc_count": 9935,
            "bh_count": 1658, "re_free_count": 8572
        }

    tc_n = int(df["TC_K"].dropna().between(1, 1500).sum())
    ms_n = int(df["Ms_emu_g"].dropna().between(0.01, 350).sum())
    hc_n = int(df["Hc_kA_m"].dropna().between(0.001, 1e5).sum())
    bh_n = int(df["BH_max_kJ_m3"].dropna().between(0.1, 650).sum())
    re_free_n = int((df.get("rare_earth_fraction", 1.0) == 0.0).sum())

    # Kronmuller alpha: Hc / (2*K1 / mu0*Ms)
    sub_k = df[
        (df["K1_MJ_m3"] > 0.01) &
        (df["Ms_emu_g"] > 5.0) & (df["Ms_emu_g"] <= 350) &
        (df["Hc_kA_m"] > 0.01)
    ].copy()

    if not sub_k.empty:
        # K1 in MJ/m^3 -> J/m^3 (* 1e6). Hc in kA/m -> A/m (* 1e3).
        sub_k["Hk_A_m"] = [
            kronmuller_ideal_coercivity(k1 * 1e6, ms)
            for k1, ms in zip(sub_k["K1_MJ_m3"], sub_k["Ms_emu_g"])
        ]
        sub_k["alpha"] = (sub_k["Hc_kA_m"] * 1e3) / sub_k["Hk_A_m"].replace(0, np.nan)
        clean_alpha = sub_k.loc[sub_k["alpha"].between(0.001, 1.2), "alpha"].dropna().to_numpy()
        alpha_med = float(np.median(clean_alpha)) if len(clean_alpha) else 0.110
        alpha_count = len(clean_alpha)
    else:
        alpha_med = 0.110
        alpha_count = 1064
        clean_alpha = np.array([0.11])

    # Maxwell energy product bound
    sub_bh = df[
        (df["Ms_emu_g"] > 5.0) & (df["Ms_emu_g"] <= 350) &
        (df["BH_max_kJ_m3"] > 0.1) & (df["BH_max_kJ_m3"] <= 650)
    ].copy()
    if not sub_bh.empty:
        sub_bh["bh_ceil"] = [bh_max_theoretical_limit(ms) for ms in sub_bh["Ms_emu_g"]]
        bh_ratio = sub_bh["BH_max_kJ_m3"] / sub_bh["bh_ceil"].replace(0, np.nan)
        bh_within_pct = float((bh_ratio <= 1.05).mean() * 100.0)
    else:
        bh_within_pct = 71.7

    return {
        "total_records": len(df),
        "tc_count": tc_n,
        "ms_count": ms_n,
        "hc_count": hc_n,
        "bh_count": bh_n,
        "re_free_count": re_free_n,
        "alpha_median": alpha_med,
        "alpha_n": alpha_count,
        "alpha_values": clean_alpha,
        "bh_within_bound_pct": bh_within_pct,
    }


# =========================================================================
# Animated Vector Visualizations (SMIL SVG & Plotly)
# =========================================================================

def _svg_wrap(body: str, w: int, h: int, alt: str) -> str:
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">{body}</svg>'
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<div style="background:#FFFFFF;border:1.5px solid #CBD5E1;border-radius:12px;padding:8px;margin:0.5rem 0 1.0rem 0;box-shadow:0 2px 8px rgba(0,0,0,0.04);"><img src="data:image/svg+xml;base64,{b64}" alt="{alt}" style="width:100%;max-width:{w}px;display:block;margin:0 auto;"/></div>'


def sp_curve_svg() -> str:
    """Slater-Pauling rigid-band ceiling with animated indicator running to Fe-Co peak."""
    w, h, base, top = 480, 110, 88, 14
    x0, x1, peak = 5.4, 11.0, 2.45

    def px(v):
        return 24 + (v - x0) / (x1 - x0) * (w - 48)

    def py(m):
        return base - (m / peak) * (base - top)

    pts = " ".join(f"{px(x0 + (x1 - x0) * i / 120):.1f},{py(slater_pauling_moment(x0 + (x1 - x0) * i / 120)):.1f}"
                   for i in range(121))
    marks = "".join(
        f'<circle cx="{px(v):.1f}" cy="{py(slater_pauling_moment(v)):.1f}" r="4.0" fill="#0284C7"/>'
        f'<text x="{px(v)+8:.1f}" y="{py(slater_pauling_moment(v))+14:.1f}" font-size="11.5" '
        f'font-weight="900" fill="#0F172A">{lbl}</text>'
        for v, lbl in ((8.0, "Fe"), (9.0, "Co"), (10.0, "Ni"))
    )
    body = (
        f'<line x1="20" y1="{base}" x2="{w-20}" y2="{base}" stroke="#94A3B8" stroke-width="1.5"/>'
        f'<polyline points="{pts}" fill="none" stroke="#059669" stroke-width="3.5" stroke-linejoin="round"/>'
        f'<line x1="{px(8.3):.1f}" y1="{py(peak):.1f}" x2="{px(8.3):.1f}" y2="{base}" '
        f'stroke="#D97706" stroke-width="1.6" stroke-dasharray="3 3"/>'
        f'<text x="{px(8.3):.1f}" y="{py(peak)-6:.1f}" font-size="12" font-weight="950" fill="#B45309" text-anchor="middle">Fe₇₀Co₃₀ (2.45 μB)</text>'
        + marks +
        f'<circle r="5.5" fill="#D97706">'
        f'<animate attributeName="cx" values="{px(5.5):.1f};{px(8.3):.1f};{px(8.3):.1f}" keyTimes="0;0.45;1" dur="4.5s" repeatCount="indefinite"/>'
        f'<animate attributeName="cy" values="{py(0):.1f};{py(peak):.1f};{py(peak):.1f}" keyTimes="0;0.45;1" dur="4.5s" repeatCount="indefinite"/>'
        f'</circle>'
    )
    return _svg_wrap(body, w, h, "Slater-Pauling Curve")


def spin_alignment_svg() -> str:
    """Stoner quantum exchange: atomic moments aligning ferromagnetically."""
    angles = [-36, 28, -20, 39, -42, 18, -29, 32, -24, 38]
    arrows = []
    for i, a in enumerate(angles):
        x = 24 + i * 36
        arrows.append(
            f'<g transform="translate({x},60)">'
            f'<g><animateTransform attributeName="transform" type="rotate" '
            f'values="{a};{a};0;0" keyTimes="0;0.08;0.32;1" dur="4.0s" begin="{i * 0.05:.2f}s" repeatCount="indefinite"/>'
            f'<line x1="0" y1="0" x2="0" y2="-36" stroke="#059669" stroke-width="2.8" stroke-linecap="round"/>'
            f'<polygon points="-4,-32 4,-32 0,-44" fill="#059669"/></g>'
            f'<circle cx="0" cy="4" r="2.8" fill="#0284C7"/></g>'
        )
    body = "".join(arrows) + '<text x="190" y="78" text-anchor="middle" font-size="11.5" font-weight="900" fill="#0369A1">I · D(EF) > 1 (Stoner Exchange Alignment)</text>'
    return _svg_wrap(body, 390, 85, "Stoner Spin Exchange Alignment")


def hysteresis_svg(alpha: float) -> str:
    """Ideal single-crystal square loop against real polycrystal showing the coercivity deficit."""
    w, h = 420, 160
    cx, cy, sx, sy = w / 2, h / 2, 130, 48
    ideal, real = 0.82, max(0.82 * alpha, 0.08)

    def loop(f, colour, width):
        p = f"M{cx+sx},{cy-sy} L{cx-f*sx},{cy-sy} L{cx-f*sx},{cy+sy} L{cx-sx},{cy+sy} L{cx+f*sx},{cy+sy} L{cx+f*sx},{cy-sy} Z"
        return f'<path d="{p}" fill="none" stroke="{colour}" stroke-width="{width}" stroke-linejoin="round"/>'

    body = (
        f'<line x1="16" y1="{cy}" x2="{w-16}" y2="{cy}" stroke="#94A3B8" stroke-width="1.4"/>'
        f'<line x1="{cx}" y1="14" x2="{cx}" y2="{h-14}" stroke="#94A3B8" stroke-width="1.4"/>'
        + loop(ideal, "#059669", 2.6)
        + loop(real, "#D97706", 3.8)
        + f'<text x="{cx-ideal*sx-8:.0f}" y="{cy+sy+18:.0f}" font-size="11.5" font-weight="900" fill="#047857" text-anchor="middle">Ideal Hk (100%)</text>'
        + f'<text x="{cx+real*sx+10:.0f}" y="{cy-sy-8:.0f}" font-size="11.5" font-weight="900" fill="#B45309" text-anchor="start">Real Hc ({alpha*100:.0f}%)</text>'
        + f'<text x="{cx}" y="{cy+sy+30:.0f}" font-size="11.5" font-weight="950" fill="#DC2626" text-anchor="middle">89% Coercivity Deficit Gap</text>'
    )
    return _svg_wrap(body, w, h, "Hysteresis Loops")


def animate_loop(fig, n_frames: int = 54):
    """Adds an animated magnetic tracing marker traversing the M-H hysteresis envelope."""
    try:
        branches = sorted(
            (t for t in fig.data
             if "lines" in (getattr(t, "mode", "") or "lines")
             and getattr(t, "x", None) is not None and len(t.x) > 20),
            key=lambda t: len(t.x), reverse=True)[:2]
        if not branches:
            return fig
        px_ = np.concatenate([np.asarray(t.x, dtype=float) for t in branches])
        py_ = np.concatenate([np.asarray(t.y, dtype=float) for t in branches])
        idx = np.linspace(0, len(px_) - 1, n_frames).astype(int)

        dot = dict(mode="markers", marker=dict(size=12, color="#FF6B00", line=dict(color="#FFFFFF", width=2)),
                   hoverinfo="skip", showlegend=False)
        fig.add_trace(go.Scatter(x=[px_[0]], y=[py_[0]], **dot))
        m_idx = len(fig.data) - 1

        fig.frames = [
            go.Frame(data=[go.Scatter(x=[px_[i]], y=[py_[i]], **dot)], traces=[m_idx], name=str(k))
            for k, i in enumerate(idx)
        ]
        fig.update_layout(updatemenus=[dict(
            type="buttons", showactive=False,
            x=-10.0, y=-10.0,  # Positioned off-screen so no visible trace button
            buttons=[dict(label="", method="animate",
                          args=[None, dict(frame=dict(duration=35, redraw=False),
                                           transition=dict(duration=0),
                                           fromcurrent=True, mode="immediate",
                                           loop=True)])])])
    except Exception:
        return fig
    return fig


# =========================================================================
# Streamlit Application Bootstrap & Page Configuration
# =========================================================================

st.set_page_config(
    page_title="MagMat",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

T.inject("Hybrid (Dark & Light)")

# Larger, higher-contrast navigation tabs (overrides the theme defaults).
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] { gap: 0.6rem; border-bottom: 2px solid #CBD5E1; }
.stTabs [data-baseweb="tab"] {
    font-size: 1.35rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.015em;
    color: #64748B !important;
    padding: 0.85rem 1.6rem !important;
    border-radius: 10px 10px 0 0;
}
.stTabs [data-baseweb="tab"]:hover { color: #0F172A !important; background: rgba(2,132,199,0.07); }
.stTabs [aria-selected="true"] {
    color: #0F172A !important;
    background: rgba(255,107,0,0.10);
    border-bottom: 4px solid #FF6B00 !important;
}
.stTabs [data-baseweb="tab"] p { font-size: 1.35rem !important; font-weight: 900 !important; }

/* Dropdown menus render in a portal outside .stApp, so the app theme does not
   reach them and the dark base leaves white text on a white sheet. Pin both. */
div[data-baseweb="popover"] ul[role="listbox"],
div[data-baseweb="popover"] div[role="listbox"],
div[data-baseweb="menu"] { background: #FFFFFF !important; }
div[data-baseweb="popover"] li[role="option"],
div[data-baseweb="popover"] div[role="option"],
div[data-baseweb="menu"] li {
    color: #0F172A !important;
    background: #FFFFFF !important;
    font-size: 1.02rem !important;
    font-weight: 700 !important;
}
div[data-baseweb="popover"] li[role="option"]:hover,
div[data-baseweb="popover"] div[role="option"]:hover,
div[data-baseweb="popover"] li[aria-selected="true"],
div[data-baseweb="menu"] li:hover {
    background: #FFE8D6 !important;
    color: #0F172A !important;
}

/* selected element chips */
[data-testid="stMultiSelect"] [data-baseweb="tag"] { background: #0F172A !important; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] span,
[data-testid="stMultiSelect"] [data-baseweb="tag"] div {
    color: #FFFFFF !important; font-weight: 800 !important; font-size: 0.95rem !important;
}

/* closed control values and typed text */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stSelectbox"] input,
[data-testid="stMultiSelect"] input {
    color: #0F172A !important; font-weight: 700 !important;
}

/* widget labels */
[data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {
    color: #1E293B !important; font-weight: 800 !important; font-size: 1.0rem !important;
}
</style>
""", unsafe_allow_html=True)
gap_stats = get_reality_gap_dataset_stats()
ALPHA_MEDIAN = gap_stats["alpha_median"]



def orbit_crystal(fig, n_frames: int = 60, radius: float = 1.85, height: float = 0.72):
    """Add a camera orbit around the unit cell.

    The frames carry only a layout change (scene.camera.eye), so the geometry is
    never re-sent; Plotly needs redraw=True for a 3D scene to pick the new camera
    up. Streamlit cannot autoplay, so this is wired to a button.
    """
    try:
        frames = []
        for k in range(n_frames):
            a = 2.0 * math.pi * k / n_frames
            frames.append(go.Frame(
                name=str(k),
                layout=dict(scene=dict(camera=dict(
                    eye=dict(x=radius * math.cos(a),
                             y=radius * math.sin(a),
                             z=height))))))
        fig.frames = frames
        fig.update_layout(
            scene_camera=dict(eye=dict(x=radius, y=0.0, z=height)),
            updatemenus=[dict(
                type="buttons", showactive=False,
                x=0.99, y=0.99, xanchor="right", yanchor="top",
                bgcolor="rgba(255,107,0,0.16)", bordercolor="#FF6B00", borderwidth=1,
                font=dict(color="#FF6B00", size=12),
                buttons=[dict(label="Rotate", method="animate",
                              args=[None, dict(frame=dict(duration=70, redraw=True),
                                               transition=dict(duration=0),
                                               fromcurrent=True, mode="immediate")])])])
    except Exception:
        return fig
    return fig

# Header Banner (Clean, No Ceilings Badge)
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;padding:0.4rem 0 0.8rem 0;border-bottom:1.5px solid #CBD5E1;margin-bottom:1.0rem;">
  <div style="display:flex;align-items:baseline;gap:1.2rem;flex-wrap:wrap;">
    <span style="font-size:2.55rem;font-weight:950;background:linear-gradient(135deg, #0F172A 0%, #0284C7 50%, #059669 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.035em;">MagMat</span>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ℹ️ About MagMat-Concept")
    st.info(
        "**MagMat-Concept** couples fundamental condensed-matter boundaries "
        "(Slater-Pauling, Stoner, Kronmüller, Maxwell) with trained multi-task "
        "machine learning models (Phase classification, $T_C$, $T_N$, $\\mu_0 M_s$, $H_c$, $(BH)_{\\max}$) "
        "and pretrained 3D foundation AI (DAO + MACE) "
        "for sustainable permanent magnet discovery.\n\n"
        "**Author:** Sachin Poudel"
    )


# Application Navigation Tabs (2 Tabs Only)
tab_screener, tab_concept = st.tabs([
    "Forward Screener",
    "Physics Principles & Reality Gap"
])


# =========================================================================
# TAB 1: FORWARD SCREENER & PHYSICS BOUNDS
# =========================================================================
with tab_screener:
    # Open on a blank custom formulation so the user enters their own elements
    # and stoichiometry; the archetypes stay one dropdown away. An empty element
    # list is handled downstream by the "select at least one element" guard.
    if "comp_elements" not in st.session_state:
        st.session_state.comp_elements = []
    if "comp_amounts" not in st.session_state:
        st.session_state.comp_amounts = {}
    if "comp_cs" not in st.session_state:
        st.session_state.comp_cs = "Not specified"
    if "comp_sg" not in st.session_state:
        st.session_state.comp_sg = 0
    if "_active_arch" not in st.session_state:
        st.session_state._active_arch = "Custom Formulation"
    if "_arch_version" not in st.session_state:
        st.session_state._arch_version = 0

    col_input, col_display = st.columns([1.14, 2.16], gap="large")

    with col_input:
        st.markdown("""
        <div style="margin-bottom:1.1rem;padding-bottom:0.65rem;border-bottom:1.5px solid #E2E8F0;">
          <div style="font-size:1.25rem;font-weight:950;color:#0F172A;letter-spacing:-0.02em;">Compound Formulation</div>
        </div>
        """, unsafe_allow_html=True)

        # 1. Material Archetype Preset Selector
        arch_keys = list(D.ARCHETYPES.keys())
        arch_opts = ["Custom Formulation"] + arch_keys
        cur_v = st.session_state.get("_arch_version", 0)

        def on_arch_change():
            v = st.session_state.get("_arch_version", 0)
            sel = st.session_state.get(f"arch_select_{v}", "Custom Formulation")
            st.session_state._active_arch = sel
            if sel in D.ARCHETYPES:
                arch_meta = D.ARCHETYPES[sel]
                st.session_state.comp_elements = list(arch_meta["amounts"].keys())
                st.session_state.comp_amounts = dict(arch_meta["amounts"])
                st.session_state.comp_cs = arch_meta["crystal_system"]
                st.session_state.comp_sg = arch_meta["space_group"]
                st.session_state.input_csys = arch_meta["crystal_system"]
                st.session_state.comp_elements_multiselect = list(arch_meta["amounts"].keys())
                for k in list(st.session_state.keys()):
                    if k.startswith("amt_") or k.startswith("input_sg_"):
                        del st.session_state[k]
            else:
                st.session_state.comp_cs = "Not specified"
                st.session_state.comp_sg = 0
                st.session_state.input_csys = "Not specified"
                for k in list(st.session_state.keys()):
                    if k.startswith("input_sg_"):
                        del st.session_state[k]

        active_idx = arch_opts.index(st.session_state._active_arch) if st.session_state.get("_active_arch") in arch_opts else 0
        chosen_arch = st.selectbox(
            "Material Archetype",
            arch_opts,
            index=active_idx,
            key=f"arch_select_{cur_v}",
            on_change=on_arch_change,
            format_func=lambda k: f"{D.ARCHETYPES[k]['name']} — {D.ARCHETYPES[k]['title']}" if k in D.ARCHETYPES else "Custom Formulation"
        )

        st.markdown('<div style="margin-bottom:0.85rem;"></div>', unsafe_allow_html=True)

        # 2. Constituent Elements Selector
        elem_options = list(D.SUBSTITUENTS)
        for el in st.session_state.comp_elements:
            if el not in elem_options:
                elem_options.append(el)

        def on_elements_change():
            sel_els = st.session_state.comp_elements_multiselect
            new_amounts = {}
            for el in sel_els:
                new_amounts[el] = st.session_state.comp_amounts.get(el, 1.0)
            st.session_state.comp_elements = sel_els
            st.session_state.comp_amounts = new_amounts
            st.session_state._active_arch = "Custom Formulation"
            st.session_state.comp_cs = "Not specified"
            st.session_state.comp_sg = 0
            st.session_state.input_csys = "Not specified"
            st.session_state._arch_version = st.session_state.get("_arch_version", 0) + 1
            for k in list(st.session_state.keys()):
                if k.startswith("amt_") or k.startswith("input_sg_"):
                    del st.session_state[k]

        if "comp_elements_multiselect" not in st.session_state:
            st.session_state.comp_elements_multiselect = st.session_state.comp_elements

        st.multiselect(
            "Constituent Elements",
            options=elem_options,
            key="comp_elements_multiselect",
            on_change=on_elements_change
        )

        # 3. Stoichiometry Inputs
        if not st.session_state.comp_elements:
            st.warning("Please select at least one constituent element.")
            formula = ""
            pretty = ""
            amounts = {}
            perr = "No elements selected."
        else:
            tot_amt = sum(st.session_state.comp_amounts.get(el, 0.0) for el in st.session_state.comp_elements) or 1.0
            formula = D.build_formula([(el, st.session_state.comp_amounts.get(el, 0)) for el in st.session_state.comp_elements])
            pretty = D.format_composition_subscript(st.session_state.comp_amounts)
            amounts = dict(st.session_state.comp_amounts)
            perr = None if formula else "Empty formulation."

            def on_amt_change(elem):
                val = st.session_state.get(f"amt_{elem}", 1.0)
                st.session_state.comp_amounts[elem] = val
                if st.session_state.get("_active_arch") != "Custom Formulation":
                    st.session_state._active_arch = "Custom Formulation"
                    st.session_state.comp_cs = "Not specified"
                    st.session_state.comp_sg = 0
                    st.session_state.input_csys = "Not specified"
                    for k in list(st.session_state.keys()):
                        if k.startswith("input_sg_"):
                            del st.session_state[k]
                st.session_state._arch_version = st.session_state.get("_arch_version", 0) + 1

            n_els = len(st.session_state.comp_elements)
            chunk_size = 3 if n_els >= 3 else n_els

            for i in range(0, n_els, chunk_size):
                chunk = st.session_state.comp_elements[i:i + chunk_size]
                cols = st.columns(len(chunk), gap="small")
                for c, el in zip(cols, chunk):
                    cur_amt = float(st.session_state.comp_amounts.get(el, 1.0))
                    at_pct = (cur_amt / tot_amt) * 100.0 if tot_amt > 0 else 0.0
                    c.number_input(
                        f"{el} ({at_pct:.1f}%)",
                        min_value=0.01,
                        max_value=999.0,
                        value=cur_amt,
                        step=0.1,
                        format="%.2f",
                        key=f"amt_{el}",
                        on_change=on_amt_change,
                        args=(el,)
                    )

        st.markdown('<div style="margin-bottom:0.85rem;"></div>', unsafe_allow_html=True)

        # 4. Crystal System & Space Group
        def on_csys_change():
            new_cs = st.session_state.get("input_csys", "Not specified")
            st.session_state.comp_cs = new_cs
            valid_for_new = D.get_valid_space_groups(new_cs)
            if st.session_state.get("comp_sg", 0) not in valid_for_new:
                st.session_state.comp_sg = 0
                for k in list(st.session_state.keys()):
                    if k.startswith("input_sg_"):
                        del st.session_state[k]

        c_cs, c_sg = st.columns([1.0, 1.0], gap="medium")
        with c_cs:
            cs_idx = D.CRYSTAL_SYSTEMS.index(st.session_state.comp_cs) if st.session_state.comp_cs in D.CRYSTAL_SYSTEMS else 0
            csys = st.selectbox("Crystal System", D.CRYSTAL_SYSTEMS, index=cs_idx, key="input_csys", on_change=on_csys_change)
            st.session_state.comp_cs = csys

        with c_sg:
            valid_sgs = D.get_valid_space_groups(csys)
            curr_sg = int(st.session_state.get("comp_sg", 0))
            if curr_sg not in valid_sgs:
                curr_sg = 0
                st.session_state.comp_sg = 0
            sg_idx = valid_sgs.index(curr_sg) if curr_sg in valid_sgs else 0
            sg = st.selectbox(
                "Space Group",
                valid_sgs,
                index=sg_idx,
                format_func=lambda x: "Not specified" if x == 0 else f"SG {x}",
                key=f"input_sg_{csys}"
            )
            st.session_state.comp_sg = int(sg)

        # 5. Sample Form Conditioning
        st.markdown("""
        <div style="margin:0.85rem 0 0.45rem 0;padding-top:0.65rem;border-top:1.5px solid #E2E8F0;">
          <div style="font-size:0.95rem;font-weight:900;color:#0F172A;letter-spacing:-0.01em;">Microstructural Sample Form</div>
        </div>
        """, unsafe_allow_html=True)

        form_opts = ["bulk", "film", "nano"]
        form_labels = {"bulk": "Bulk PM", "film": "Thin Film", "nano": "Nanoparticles"}
        chosen_form = st.segmented_control(
            "Processing Form",
            options=form_opts,
            format_func=lambda x: form_labels[x],
            default=st.session_state.get("comp_sample_form", "bulk"),
            key="comp_sample_form_seg",
            label_visibility="collapsed"
        ) or "bulk"
        st.session_state.comp_sample_form = chosen_form

        st.markdown('<div style="margin-bottom:0.75rem;"></div>', unsafe_allow_html=True)
        st.button("Screen & Evaluate", type="primary", width="stretch", key="btn_screen_compound")

    with col_display:
        if formula.strip() and not perr:
            with st.spinner("Executing trained multi-task inference and computing physics bounds..."):
                res, alt, err = D.predict_auto(formula, csys, sg, sample_form=chosen_form)

            if err:
                st.error(f"Prediction Error: {err}")
            elif res:
                # 1. Properties predicted by the trained ML models
                phase = str(res.get("predicted_phase", "—"))
                tc = float(res.get("predicted_TC_K", 0) or 0)
                tn = float(res.get("predicted_TN_K", 0) or 0)
                ms_t = float(res.get("mu0_Ms_Tesla", 0) or 0)
                k1_mj = float(res.get("K1_MJ_m3", 0) or 0)
                kappa = float(res.get("hardness_kappa", 0) or 0)
                hc_ka_m = float(res.get("coercivity_Hc_kA_m", 0) or 0)
                hc_kOe = float(res.get("coercivity_Hc_kOe", 0) or 0)
                bh_max_kj = float(res.get("energy_product_BH_max_kJ_m3", 0) or 0)
                t_ord = tn if phase == "AFM" else tc
                t_lab = "Néel Point Tₙ" if phase == "AFM" else "Curie Point Tᴄ"

                # Uncertainties & Conformal bounds
                tc_u = float(res.get("predicted_TC_uncertainty_K", 0) or 0)
                tn_u = float(res.get("predicted_TN_uncertainty_K", 0) or 0)
                ms_u = float(res.get("mu0_Ms_uncertainty_Tesla", 0) or 0)
                hc_u = float(res.get("coercivity_Hc_uncertainty_kA_m", 0) or 0)
                bh_u = float(res.get("energy_product_BH_max_uncertainty_kJ_m3", 0) or 0)

                # 2. Deterministic Condensed-Matter Physics Bounds
                phys_bounds = evaluate_stoichiometry_physics(amounts)
                sp_ms_t = phys_bounds.get("sp_ms_tesla", 0.0)
                bh_max_ceiling = phys_bounds.get("bh_max_ceiling_kj_m3", 0.0)
                stoner_score = phys_bounds.get("stoner_score", 0.0)

                # Theoretical nucleation ceiling Hk = 2*K1 / (mu0*Ms)
                hk_ideal_ka_m = (2.0 * (k1_mj * 1e6) / max(ms_t, 1e-4)) / 1000.0 if k1_mj > 0 and ms_t > 0 else 0.0
                alpha_realized = (hc_ka_m / hk_ideal_ka_m) if hk_ideal_ka_m > 0 else 0.0
                defect_gap_pct = max(0.0, (1.0 - alpha_realized) * 100.0) if hk_ideal_ka_m > 0 else 0.0

                # Synthesize 3D periodic lattice & quantum spin configuration (DAO + MACE)
                with st.spinner("Synthesizing 3D lattice & quantum spin configuration (DAO diffusion)..."):
                    dao_res = D.predict_dao_crystal(formula, crystal_system=csys if csys != "Not specified" else None,
                                                    predicted_phase=phase if phase != "—" else "FM")

                sp = dao_res.get('spin_polarization', {})
                spin_flag = sp.get('spin_alignment_flag', phase if phase != "—" else "FM")
                spin_map = {
                    "FM": "Collinear FM", "AFM_COLLINEAR": "Collinear AFM",
                    "FERRIMAGNETIC": "Ferrimagnetic", "CANTED_NONCOLLINEAR": "Non-Collinear Canting",
                    "NON_MAGNETIC": "Non-Magnetic"
                }
                spin_label = spin_map.get(spin_flag, "Collinear FM" if phase == "FM" else "Non-Magnetic")

                # Banner
                tot_at = sum(amounts.values()) or 1.0
                stoich_chips = "".join([
                    f'<span style="background:#1E293B;color:#FFFFFF;border:1.5px solid rgba(255,255,255,0.25);'
                    f'border-radius:8px;padding:0.25rem 0.6rem;font-weight:900;font-size:0.9rem;margin-left:0.3rem;">'
                    f'{el} {(amt/tot_at)*100:.1f}%</span>'
                    for el, amt in amounts.items()
                ])

                cs_txt = csys.title() if csys != "Not specified" else "Unspecified Symmetry"
                sg_txt = f" · SG {sg}" if sg > 0 else ""
                sym_badge = (
                    f'<span style="background:rgba(56,189,248,0.15);border:1.5px solid #38BDF8;'
                    f'color:#38BDF8;border-radius:10px;padding:0.4rem 0.9rem;font-size:1.0rem;font-weight:900;">'
                    f'{cs_txt}{sg_txt}</span>'
                )
                spin_badge = (
                    f'<span style="background:rgba(0,245,155,0.15);border:1.5px solid #00F59B;'
                    f'color:#00F59B;border-radius:10px;padding:0.4rem 0.9rem;font-size:1.0rem;font-weight:900;">'
                    f'Spin: {spin_label}</span>'
                )
                p_col = "#EF4444" if phase == "FM" else ("#A855F7" if phase == "AFM" else "#64748B")
                p_bg = "rgba(239,68,68,0.18)" if phase == "FM" else ("rgba(168,85,247,0.18)" if phase == "AFM" else "rgba(100,116,139,0.18)")
                phase_title = "Ferromagnetic (FM)" if phase == "FM" else ("Antiferromagnetic (AFM)" if phase == "AFM" else "Non-Magnetic (NM)")
                phase_badge = (
                    f'<span style="background:{p_bg};border:1.5px solid {p_col};'
                    f'color:{p_col};border-radius:8px;padding:0.4rem 0.95rem;font-size:1.05rem;font-weight:900;">'
                    f'{phase_title}</span>'
                )

                # Clean Compound Header (Big Fonts, Crisp Badge)
                st.markdown(f"""
                <div style="background:linear-gradient(180deg, rgba(30, 41, 59, 0.90) 0%, rgba(15, 23, 42, 0.98) 100%);
                            border:1.5px solid rgba(255,255,255,0.20);border-radius:14px;padding:0.85rem 1.4rem;
                            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.8rem;margin-bottom:1.0rem;">
                  <div style="display:flex;align-items:center;gap:1.0rem;flex-wrap:wrap;">
                    <span style="font-size:2.35rem;font-weight:950;color:#FFFFFF;letter-spacing:-0.03em;">{pretty}</span>
                    {phase_badge}
                    {sym_badge}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # 4 Primary Figures of Merit Cards (Clean, Large Fonts, Compact)
                q_tc = tc_u if tc_u > 0 else 45.0
                q_ms = ms_u if ms_u > 0 else 0.08
                hard_label = "Hard Magnet" if kappa >= 1.0 else ("Semi-Hard" if kappa >= 0.1 else "Soft")
                sp_pct = (ms_t / sp_ms_t * 100.0) if sp_ms_t > 0 else 0.0
                bh_eff = (bh_max_kj / bh_max_ceiling * 100.0) if bh_max_ceiling > 0 else 0.0

                m1, m2, m3, m4 = st.columns(4, gap="medium")
                with m1:
                    st.markdown(f"""
                    <div style="background:#111827;border:1px solid rgba(255,255,255,0.18);border-top:4px solid {T.TEMP};
                                border-radius:14px;padding:16px 18px;box-shadow:0 8px 24px -4px rgba(0,0,0,0.5);">
                      <div style="font-size:0.85rem;font-weight:850;color:#CBD5E1;letter-spacing:0.05em;text-transform:uppercase;">{t_lab}</div>
                      <div style="font-size:2.25rem;font-weight:950;color:{T.TEMP};line-height:1.15;margin:4px 0;">
                        {t_ord:,.0f} K <span style="font-size:1.0rem;font-weight:600;color:#CBD5E1;">± {q_tc:.0f} K</span>
                      </div>
                      <div style="font-size:0.92rem;font-weight:800;color:#E2E8F0;">{t_ord - 273.15:.0f} °C</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div style="background:#111827;border:1px solid rgba(255,255,255,0.18);border-top:4px solid {T.MOMENT};
                                border-radius:14px;padding:16px 18px;box-shadow:0 8px 24px -4px rgba(0,0,0,0.5);">
                      <div style="font-size:0.85rem;font-weight:850;color:#CBD5E1;letter-spacing:0.05em;text-transform:uppercase;">Saturation μ₀Mₛ</div>
                      <div style="font-size:2.25rem;font-weight:950;color:{T.MOMENT};line-height:1.15;margin:4px 0;">
                        {ms_t:.2f} T <span style="font-size:1.0rem;font-weight:600;color:#CBD5E1;">± {q_ms:.2f} T</span>
                      </div>
                      <div style="font-size:0.92rem;font-weight:800;color:#E2E8F0;">Ceiling: <b>{sp_ms_t:.2f} T</b> ({sp_pct:.0f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div style="background:#111827;border:1px solid rgba(255,255,255,0.18);border-top:4px solid {T.HARD};
                                border-radius:14px;padding:16px 18px;box-shadow:0 8px 24px -4px rgba(0,0,0,0.5);">
                      <div style="font-size:0.85rem;font-weight:850;color:#CBD5E1;letter-spacing:0.05em;text-transform:uppercase;">Coercivity Hᴄ</div>
                      <div style="font-size:2.25rem;font-weight:950;color:{T.HARD};line-height:1.15;margin:4px 0;">
                        {hc_ka_m:,.0f} <span style="font-size:1.1rem;font-weight:800;">kA/m</span>
                      </div>
                      <div style="font-size:0.92rem;font-weight:800;color:#E2E8F0;">{hc_kOe:.1f} kOe · Realized α = <b>{alpha_realized:.2f}</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div style="background:#111827;border:1px solid rgba(255,255,255,0.18);border-top:4px solid {T.ACCENT};
                                border-radius:14px;padding:16px 18px;box-shadow:0 8px 24px -4px rgba(0,0,0,0.5);">
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:0.85rem;font-weight:850;color:#CBD5E1;letter-spacing:0.05em;text-transform:uppercase;">Energy Product</span>
                        <span style="background:rgba(255,107,0,0.22);border:1.5px solid #FF8C00;color:#FFA000;padding:2px 8px;border-radius:6px;font-size:0.75rem;font-weight:900;">{hard_label}</span>
                      </div>
                      <div style="font-size:2.25rem;font-weight:950;color:{T.ACCENT};line-height:1.15;margin:4px 0;">
                        {bh_max_kj:.0f} <span style="font-size:1.1rem;font-weight:800;">kJ/m³</span>
                      </div>
                      <div style="font-size:0.92rem;font-weight:800;color:#E2E8F0;">{bh_max_kj / 7.96:.1f} MGOe · Limit <b>{bh_max_ceiling:.0f} kJ/m³</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div style="margin-bottom: 1.25rem;"></div>', unsafe_allow_html=True)

                # Side-by-Side: Dynamic M-H Hysteresis & 3D Crystal Structure
                col_hyst, col_3d = st.columns(2, gap="medium")
                with col_hyst:
                    st.markdown('<div style="font-size:1.18rem;font-weight:950;color:#0F172A;margin-bottom:0.35rem;">Dynamic M-H Hysteresis Response</div>', unsafe_allow_html=True)
                    hyst_fig = animate_loop(D.make_hysteresis_fig(
                        ms_tesla=ms_t,
                        hc_ka_m=hc_ka_m if hc_ka_m > 0 else 750.0,
                        kappa=kappa,
                        loop_mode="Major Loop"
                    ))
                    hyst_fig.update_layout(
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#F8FAFC",
                        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#0F172A"),
                        xaxis=dict(
                            title=dict(text="<b>Applied Magnetic Field H (kA/m)</b>", font=dict(size=12.5, color="#0F172A")),
                            tickfont=dict(size=11, color="#0F172A"),
                            gridcolor="#CBD5E1",
                            zeroline=True,
                            zerolinecolor="#94A3B8",
                            linecolor="#64748B"
                        ),
                        yaxis=dict(
                            title=dict(text="<b>Polarization J = μ₀M (Tesla)</b>", font=dict(size=12.5, color="#0F172A")),
                            tickfont=dict(size=11, color="#0F172A"),
                            gridcolor="#CBD5E1",
                            zeroline=True,
                            zerolinecolor="#94A3B8",
                            linecolor="#64748B"
                        ),
                        legend=dict(
                            font=dict(size=11, color="#0F172A"),
                            bgcolor="rgba(255, 255, 255, 0.95)",
                            bordercolor="#CBD5E1",
                            borderwidth=1
                        ),
                        height=395,
                        margin=dict(l=60, r=20, t=25, b=65)
                    )
                    st.plotly_chart(hyst_fig, width="stretch", config={"displayModeBar": False})
                    st.markdown(f'<div style="font-size:0.88rem;font-weight:750;color:#334155;margin-top:0.35rem;">Saturation μ₀Ms = {ms_t:.2f} T · Coercivity Hc = {hc_ka_m:,.0f} kA/m · α = {alpha_realized:.2f}</div>', unsafe_allow_html=True)

                with col_3d:
                    struct = dao_res['structure']
                    lat = struct.lattice
                    sg_label = dao_res.get('spacegroup_symbol') or (csys.title() if csys != "Not specified" else "")
                    sg_title_sub = f" ({sg_label})" if sg_label else ""
                    st.markdown(f"""
                    <div style="font-size:1.18rem;font-weight:950;color:#0F172A;margin-bottom:0.35rem;">3D Crystal Structure{sg_title_sub}</div>
                    <div style="background:#FFFFFF;border:1.5px solid #CBD5E1;border-radius:10px;padding:6px 12px;margin-bottom:8px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
                      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                        <div>
                          <span style="font-size:0.80rem;font-weight:900;color:#0F172A;text-transform:uppercase;letter-spacing:0.04em;">Lattice:</span>
                          <span style="font-size:0.92rem;font-weight:900;color:#B45309;margin-left:5px;">a = {lat.a:.3f} Å</span>
                          <span style="font-size:0.92rem;font-weight:900;color:#0369A1;margin-left:8px;">b = {lat.b:.3f} Å</span>
                          <span style="font-size:0.92rem;font-weight:900;color:#047857;margin-left:8px;">c = {lat.c:.3f} Å</span>
                        </div>
                        <div>
                          <span style="font-size:0.86rem;font-weight:800;color:#334155;">α={lat.alpha:.1f}° β={lat.beta:.1f}° γ={lat.gamma:.1f}°</span>
                          <span style="font-size:0.82rem;font-weight:750;color:#64748B;margin-left:6px;">({lat.volume:.1f} Å³)</span>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    fig_3d = D.make_3d_crystal_fig(struct, f"{pretty} Unit Cell")
                    for trace in fig_3d.data:
                        if getattr(trace, "name", "") == "Unit Cell Boundary":
                            trace.showlegend = False
                    fig_3d.update_layout(title="", height=355, margin=dict(l=10, r=10, t=10, b=15))
                    fig_3d = orbit_crystal(fig_3d)
                    st.plotly_chart(fig_3d, width="stretch", config={"displayModeBar": False})

                    cif_content = dao_res.get('magnetic_cif') or dao_res.get('cif', '')
                    if cif_content:
                        st.download_button("📥 Download Crystallographic CIF (.cif)", data=cif_content, file_name=f"{pretty}.cif", mime="chemical/x-cif", width="stretch", key="btn_dl_cif")

                # Auto-play Trigger for Hysteresis Loop Animation
                import streamlit.components.v1 as components
                components.html("""
                <script>
                function triggerHysteresisLoop() {
                    try {
                        const pDoc = window.parent.document;
                        const plots = pDoc.querySelectorAll('.js-plotly-plot');
                        plots.forEach(plot => {
                            if (window.parent.Plotly && plot.data && plot._fullLayout) {
                                window.parent.Plotly.animate(plot, null, {
                                    frame: { duration: 35, redraw: false },
                                    transition: { duration: 0 },
                                    mode: 'immediate',
                                    loop: true
                                });
                            }
                        });
                        const btns = pDoc.querySelectorAll('g.updatemenu-button, .updatemenu-button');
                        btns.forEach(b => {
                            b.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
                        });
                    } catch(e) {}
                }
                setTimeout(triggerHysteresisLoop, 350);
                setTimeout(triggerHysteresisLoop, 850);
                setTimeout(triggerHysteresisLoop, 1600);
                </script>
                """, height=0)

        else:
            st.markdown("""
            <div style="text-align:center;padding:3.5rem 1.5rem;">
              <div style="font-size:1.25rem;font-weight:950;color:#0F172A;margin-bottom:0.4rem;">Awaiting Compound Formulation</div>
              <div style="font-size:1.0rem;font-weight:700;color:#475569;">Select constituent elements and specify stoichiometry on the left control panel to trigger live multi-task screening.</div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================================
# TAB 2: PHYSICS CONCEPT & REALITY GAP
# =========================================================================
with tab_concept:
    T.section("Physics Principles & Reality Gap Analysis")
    st.markdown(
        f'<p style="font-size:1.05rem;font-weight:700;color:#1E293B;margin:-0.2rem 0 1.2rem 0;line-height:1.5;">'
        f'Benchmarking against condensed-matter boundaries across {gap_stats["total_records"]:,} experimental records. '
        f'Real polycrystals achieve only ~11% of their single-crystal theoretical limit, exposing an <b>89% microstructural defect gap</b>.'
        f'</p>',
        unsafe_allow_html=True,
    )

    T.stats([
        ("TOTAL RECORDS", f"{gap_stats['total_records']:,}", "Harmonized Inorganics", "#1E293B"),
        ("CURIE POINT Tᴄ", f"{gap_stats['tc_count']:,}", "Measured Alloys", "#D97706"),
        ("SATURATION Mₛ", f"{gap_stats['ms_count']:,}", "Measured Moments", "#047857"),
        ("COERCIVITY Hᴄ", f"{gap_stats['hc_count']:,}", "Measured Loops", "#DC2626"),
    ])

    T.section("Condensed-Matter Boundaries")
    b1, b2 = st.columns(2, gap="large")
    with b1:
        st.markdown(T.metric_card(
            "01 · SLATER-PAULING LIMIT", "2.45 μB", "Max spin moment at VEC ≈ 8.3",
            accent="#047857",
            meta="3d band filling caps net magnetization at 8.3 e⁻/atom (Fe-Co peak); exceeding this fills minority states, lowering moment."
        ), unsafe_allow_html=True)
        st.markdown(sp_curve_svg(), unsafe_allow_html=True)

        st.markdown(T.metric_card(
            "02 · STONER EXCHANGE CRITERION", "I · D(EF) > 1", "Spontaneous ferromagnetic ordering",
            accent="#0284C7",
            meta="Spontaneous ferromagnetism emerges when intra-atomic exchange overcomes kinetic band-polarization penalty."
        ), unsafe_allow_html=True)
        st.markdown(spin_alignment_svg(), unsafe_allow_html=True)

    with b2:
        st.markdown(T.metric_card(
            "03 · KRONMÜLLER MICROMAGNETIC LIMIT", f"α = {ALPHA_MEDIAN:.2f}", f"Median of {gap_stats['alpha_n']:,} measured alloys",
            accent="#EA580C", status_badge="89% DEFICIT GAP",
            meta="Polycrystals realize ~11% of theoretical nucleation field (2K₁/μ₀Mₛ) due to premature grain-boundary demagnetization."
        ), unsafe_allow_html=True)
        st.markdown(hysteresis_svg(ALPHA_MEDIAN), unsafe_allow_html=True)

        st.markdown(T.metric_card(
            "04 · MAXWELL THERMODYNAMIC LIMIT", f"{gap_stats['bh_within_bound_pct']:.0f}%", "Experimental records obey bound",
            accent="#7C3AED",
            meta="Demagnetization energetics strictly enforce a fundamental thermodynamic ceiling of (BH)ₘₐₓ ≤ ¼μ₀Mₛ²."
        ), unsafe_allow_html=True)

    T.section("The Coercivity Realization Gap", tag=f"α = {ALPHA_MEDIAN:.2f} · 89% SHORTFALL")
    g1, g2 = st.columns([1, 1.3], gap="large")
    with g1:
        st.markdown(T.metric_card(
            "IDEAL SINGLE CRYSTAL", "100%", "Theoretical limit (2K₁ / μ₀Mₛ)",
            accent="#047857",
            meta="Upper limit permitted by intrinsic magnetocrystalline anisotropy."
        ), unsafe_allow_html=True)
        st.markdown(T.metric_card(
            "REAL POLYCRYSTALLINE ALLOY", f"{ALPHA_MEDIAN*100:.0f}%", "Empirically realized in practice",
            accent="#DC2626",
            meta="89% deficit lost to grain boundaries, misorientation, and microstructural defect sites."
        ), unsafe_allow_html=True)
        st.markdown(T.metric_card(
            "CRYSTAL STRUCTURE SENSITIVITY", "τ ≠ β", "Same chemistry (MnAl), distinct phases",
            accent="#0284C7",
            meta="Polymorph distinction: ferromagnetic τ-MnAl vs. paramagnetic β-MnAl at identical composition."
        ), unsafe_allow_html=True)

    with g2:
        alpha_vals = gap_stats["alpha_values"]
        if len(alpha_vals):
            fig_a = px.histogram(x=np.clip(alpha_vals, 0, 0.8), nbins=48, title="Experimental Distribution of Microstructural Factor α")
            fig_a.update_traces(marker_color="#0284C7")
            fig_a.add_vline(x=ALPHA_MEDIAN, line_dash="dash", line_color="#DC2626",
                            annotation_text=f"Median α = {ALPHA_MEDIAN:.2f}",
                            annotation_font=dict(color="#DC2626", size=12, family="Inter, sans-serif"))
            cfg_a = dict(T.get_plot_layout("Hybrid (Dark & Light)"))
            cfg_a["margin"] = dict(l=60, r=20, t=44, b=55)
            fig_a.update_layout(**cfg_a, height=360)
            fig_a.update_xaxes(
                title=dict(text="<b>Microstructural Factor α = Hc / Hk</b>", font=dict(color="#0F172A", size=13)),
                tickfont=dict(color="#1E293B", size=11),
                gridcolor="#CBD5E1",
                zerolinecolor="#94A3B8"
            )
            fig_a.update_yaxes(
                title=dict(text="<b>Experimental Count</b>", font=dict(color="#0F172A", size=13)),
                tickfont=dict(color="#1E293B", size=11),
                gridcolor="#CBD5E1",
                zerolinecolor="#94A3B8"
            )
            st.plotly_chart(fig_a, width="stretch")





    # Harmonized Experimental Database as Collapsible Expander in Tab 2
    with st.expander("📂 Harmonized Experimental Database (51,487 Records)", expanded=False):
        T.section("Harmonized Experimental Database", tag="51,487 UNIFIED EXPERIMENTAL RECORDS")
        master_df = D.load_experimental_master_table()
        if master_df is not None and not master_df.empty:
            f_col1, f_col2 = st.columns([1.6, 1.0], gap="medium")
            with f_col1:
                q_search = st.text_input("Search Chemical Formula (e.g. Fe, Nd2Fe14B, MnAl, SmCo5):", value="", key="search_cat_formula")
            with f_col2:
                phase_opts = ["FM", "AFM", "NM"]
                sel_phases = st.multiselect("Filter Ground-State Phase:", options=phase_opts, default=phase_opts, key="filter_cat_phase")

            view_df = master_df.copy()
            if q_search.strip():
                view_df = view_df[view_df["Formula"].str.contains(q_search.strip(), case=False, na=False)]
            if sel_phases and "Phase" in view_df.columns:
                view_df = view_df[view_df["Phase"].isin(sel_phases)]

            col_c1, col_c2 = st.columns([1, 1], gap="medium")
            with col_c1:
                st.caption(f"Showing {len(view_df):,} records")
            with col_c2:
                csv_data = view_df[["Formula", "Crystal_System", "Phase", "TC_K", "Ms_emu_g", "Hc_kA_m", "BH_max_kJ_m3"]].head(500).to_csv(index=False)
                st.download_button("Export Screened Results (.csv)", data=csv_data, file_name="screened_magnetic_materials.csv", mime="text/csv")

            disp_cols = ["Formula", "Crystal_System", "Phase", "TC_K", "TN_K", "Ms_emu_g", "Hc_kA_m", "BH_max_kJ_m3"]
            disp_avail = [c for c in disp_cols if c in view_df.columns]
            st.dataframe(view_df[disp_avail].head(100), width="stretch")
        else:
            st.info("Experimental records database is currently loading or unavailable.")


