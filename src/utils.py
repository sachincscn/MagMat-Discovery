import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np

# ── Publication-grade Color Palette ───────────────────────────────────────────
# Curated for magnetic materials science — tested for contrast, color-blindness
# safety, and print reproduction quality.
PALETTE = {
    # Magnetic class identity colors
    'fm_color':       '#E63946',   # Vivid crimson — Ferromagnetic
    'afm_color':      '#457B9D',   # Steel cerulean — Antiferromagnetic
    'nm_color':       '#8338EC',   # Electric violet — Non-magnetic

    # Structural/semantic tones
    'primary':        '#1D3557',   # Deep navy blue
    'secondary':      '#A8DADC',   # Pale teal accent
    'accent':         '#F4A261',   # Amber gold highlight
    'highlight':      '#E63946',   # Emergency red

    # Neutral tones
    'neutral_dark':   '#0D1117',   # Very dark slate
    'neutral_mid':    '#1F2937',   # Dark slate
    'neutral_light':  '#F8F9FA',   # Off-white paper
    'grid_color':     '#2D3748',   # Subtle dark grid
    'spine_color':    '#4A5568',   # Muted spine

    # Accent metals
    'gold':           '#FFB703',   # Deep amber gold
    'cobalt':         '#4361EE',   # Classic cobalt
    'teal':           '#06B6D4',   # Bright cyan
}

# Categorical palette list for multi-group seaborn plots
CAT_PALETTE = [
    PALETTE['fm_color'],
    PALETTE['afm_color'],
    PALETTE['nm_color'],
    PALETTE['cobalt'],
    PALETTE['gold'],
    PALETTE['teal'],
    PALETTE['accent'],
]

# Sequential gradient for heatmaps (magnetic field density feel)
CMAP_DIV    = 'RdBu_r'       # Diverging: red–blue for correlations
CMAP_SEQ    = 'magma'        # Sequential: dark→bright for temps
CMAP_CLASS  = 'viridis'      # Multi-class probability maps
CMAP_PLASMA = 'plasma'       # Feature importance


def setup_plotting_style():
    """
    Configures Matplotlib/Seaborn to produce premium, reproducible, publication-
    quality figures.  Targets 300 DPI output on a clean white background with
    tightly controlled typography, line weights, and grid visibility.

    This function is idempotent — calling it multiple times is safe.
    """
    # ── Font configuration ───────────────────────────────────────────────────
    # Fall back gracefully through an ordered priority list of professional fonts
    preferred_fonts = [
        'Helvetica Neue', 'Helvetica', 'Arial', 'Liberation Sans',
        'DejaVu Sans', 'Nimbus Sans', 'FreeSans', 'sans-serif'
    ]

    # ── Global rcParams ──────────────────────────────────────────────────────
    plt.rcParams.update({
        # Figure
        'figure.facecolor':      '#F9F9FC',  # light aesthetic background
        'figure.edgecolor':      '#F9F9FC',
        'figure.dpi':            150,        # screen preview DPI (save at 300)
        'figure.autolayout':     False,      # manual tight_layout/constrained

        # Fonts
        'font.family':           'sans-serif',
        'font.sans-serif':       preferred_fonts,
        'font.size':             14,
        'mathtext.fontset':      'dejavusans',

        # Axes
        'axes.facecolor':        '#F9F9FC',
        'axes.edgecolor':        '#334155',
        'axes.linewidth':        1.5,
        'axes.labelsize':        16,
        'axes.labelweight':      'bold',
        'axes.labelcolor':       '#0f172a',
        'axes.titlesize':        18,
        'axes.titleweight':      'bold',
        'axes.titlecolor':       '#0f172a',
        'axes.titlepad':         20,
        'axes.spines.top':       False,
        'axes.spines.right':     False,
        'axes.prop_cycle':       plt.cycler('color', CAT_PALETTE),

        # Ticks
        'xtick.labelsize':       14,
        'ytick.labelsize':       14,
        'xtick.color':           '#334155',
        'ytick.color':           '#334155',
        'xtick.major.size':      6,
        'ytick.major.size':      6,
        'xtick.major.width':     1.5,
        'ytick.major.width':     1.5,
        'xtick.minor.visible':   False,
        'ytick.minor.visible':   False,

        # Grid
        'axes.grid':             True,
        'grid.color':            '#cbd5e1',
        'grid.linestyle':        '--',
        'grid.linewidth':        0.8,
        'grid.alpha':            0.7,

        # Legend
        'legend.fontsize':       14,
        'legend.title_fontsize': 16,
        'legend.framealpha':     0.95,
        'legend.edgecolor':      '#94a3b8',
        'legend.facecolor':      'white',
        'legend.shadow':         True,
        'legend.frameon':        True,

        # Lines & Markers
        'lines.linewidth':       2.5,
        'lines.markersize':      8.0,
        'patch.linewidth':       1.2,

        # Save
        'savefig.dpi':           300,
        'savefig.bbox':          'tight',
        'savefig.facecolor':     '#F9F9FC',
        'savefig.edgecolor':     '#F9F9FC',
        'savefig.transparent':   False,

        # PDF embedding
        'pdf.fonttype':          42,
        'ps.fonttype':           42,
    })

    # Set seaborn theme to whitegrid (compatible with our rcParams)
    sns.set_theme(style='ticks', context='talk', font_scale=1.2)
    sns.set_palette(CAT_PALETTE)


def apply_clean_axes(ax, xlabel=None, ylabel=None, title=None,
                     grid_axis='both', remove_top=True, remove_right=True):
    """
    Applies consistent formatting to a single Axes object.
    Call this after plotting to ensure all axes share the same look.
    """
    if remove_top:
        ax.spines['top'].set_visible(False)
    if remove_right:
        ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', color='#1A202C', labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', color='#1A202C', labelpad=10)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', color='#1A202C', pad=14)

    ax.tick_params(axis='both', labelsize=10, colors='#4A5568')
    ax.set_axisbelow(True)

    if grid_axis in ('x', 'both'):
        ax.xaxis.grid(True, color='#E2E8F0', linestyle='--', linewidth=0.65, alpha=0.8)
    if grid_axis in ('y', 'both'):
        ax.yaxis.grid(True, color='#E2E8F0', linestyle='--', linewidth=0.65, alpha=0.8)

    return ax


def add_stat_box(ax, text, loc='upper left', fontsize=10, alpha=0.92):
    """
    Adds a floating statistics annotation box on an axes.
    loc: 'upper left', 'upper right', 'lower left', 'lower right'
    """
    loc_map = {
        'upper left':  (0.03, 0.97),
        'upper right': (0.97, 0.97),
        'lower left':  (0.03, 0.03),
        'lower right': (0.97, 0.03),
    }
    va_map = {
        'upper left': 'top', 'upper right': 'top',
        'lower left': 'bottom', 'lower right': 'bottom'
    }
    ha_map = {
        'upper left': 'left', 'upper right': 'right',
        'lower left': 'left', 'lower right': 'right'
    }
    x, y = loc_map.get(loc, (0.03, 0.97))
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        verticalalignment=va_map.get(loc, 'top'),
        horizontalalignment=ha_map.get(loc, 'left'),
        fontsize=fontsize,
        color='#1A202C',
        bbox=dict(
            boxstyle='round,pad=0.45', facecolor='white',
            edgecolor='#CBD5E0', linewidth=0.8, alpha=alpha
        ),
        family='monospace'
    )


def save_results_summary(summary_dict, filepath="results_summary.json"):
    """
    Consolidates scientific findings and model performance metrics into a clean,
    structured JSON. If the file exists, merges the new data with existing content.
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                existing = json.load(f)
            existing.update(summary_dict)
            summary_dict = existing

        with open(filepath, 'w') as f:
            json.dump(summary_dict, f, indent=4, default=str)
        print(f"Successfully saved scientific summary to {filepath}")
    except Exception as e:
        print(f"Error saving scientific summary JSON: {e}")


# ── Shared Chemical & Magnetic Constants ───────────────────────────────────────
# Centralized definitions used across featurization, GNN embeddings, and modeling.

MAGNETIC_3D = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"}
RARE_EARTH = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"
}
CHALCOGENS = {"O", "S", "Se", "Te"}
PNICTOGENS = {"N", "P", "As", "Sb", "Bi"}
HALOGENS = {"F", "Cl", "Br", "I"}

ELEMENTAL_FM = {"Fe", "Co", "Ni", "Gd", "Tb", "Dy", "Ho", "Er", "Tm"}
ELEMENTAL_AFM = {"Cr", "Mn", "O", "Nd", "Sm", "Eu"}
STRONG_FM_3D = {"Fe", "Co", "Ni"}
STRONG_AFM_3D = {"Cr", "Mn"}
RARE_EARTH_FM = {"Gd", "Tb", "Dy", "Ho", "Er", "Tm"}
RARE_EARTH_AFM = {"Nd", "Sm", "Eu"}
COMPLEX_RARE_EARTH_MAGNETS = {"Tb", "Dy", "Ho", "Er", "Tm", "Nd", "Sm", "Eu"}
MOLECULAR_AFM_ELEMENTS = {"O"}
ALL_MAGNETIC_ELEMENTS = ELEMENTAL_FM | ELEMENTAL_AFM | COMPLEX_RARE_EARTH_MAGNETS | RARE_EARTH_AFM | RARE_EARTH_FM
HIGH_CURIE_ELEMENTS = {"Fe", "Co", "Ni"}

# Approximate elemental magnetic ordering temperatures in kelvin (soft priors)
ELEMENTAL_TC_K = {
    "Co": 1388.0, "Fe": 1043.0, "Ni": 627.0,
    "Gd": 292.0, "Tb": 222.0, "Dy": 89.0,
    "Tm": 32.0, "Ho": 20.0, "Er": 19.0,
}
ELEMENTAL_TN_K = {
    "Cr": 311.0, "Mn": 100.0, "O": 23.8,
    "Nd": 19.9, "Sm": 106.0, "Eu": 90.0,
}

# ── Advanced Magnetic Physics & Anisotropy Constants ───────────────────────────

# Single-ion spin-orbit coupling (SOC) parameters \zeta (in meV)
SOC_CONSTANTS_MEV = {
    # 3d transition metals
    'Sc': 10.0, 'Ti': 20.0, 'V': 30.0, 'Cr': 35.0, 'Mn': 45.0,
    'Fe': 50.0, 'Co': 70.0, 'Ni': 100.0, 'Cu': 105.0, 'Zn': 40.0,
    # 4d transition metals
    'Y': 40.0, 'Zr': 60.0, 'Nb': 90.0, 'Mo': 100.0, 'Ru': 140.0, 'Rh': 160.0, 'Pd': 190.0,
    # 5d transition metals (strong SOC → magnetocrystalline anisotropy / DMI drivers)
    'Hf': 200.0, 'Ta': 240.0, 'W': 270.0, 'Re': 350.0, 'Os': 450.0,
    'Ir': 550.0, 'Pt': 600.0, 'Au': 650.0,
    # 4f rare earths (massive unquenched orbital moments & single-ion anisotropy)
    'La': 0.0, 'Ce': 80.0, 'Pr': 150.0, 'Nd': 250.0, 'Sm': 350.0, 'Eu': 400.0,
    'Gd': 170.0, 'Tb': 1100.0, 'Dy': 1400.0, 'Ho': 1200.0, 'Er': 1000.0,
    'Tm': 800.0, 'Yb': 400.0, 'Lu': 0.0
}

# Nominal high-spin spin quantum numbers S for magnetic ions
SPIN_QUANTUM_NUMBERS = {
    'Cr': 1.5, 'Mn': 2.5, 'Fe': 2.0, 'Co': 1.5, 'Ni': 1.0, 'Cu': 0.5,
    'Gd': 3.5, 'Tb': 3.0, 'Dy': 2.5, 'Ho': 2.0, 'Er': 1.5, 'Tm': 1.0,
    'Nd': 1.5, 'Sm': 0.5, 'Eu': 3.5, 'V': 1.5, 'Ti': 0.5
}

# Stoner exchange parameter I (eV) for itinerant 3d transition metals
STONER_PARAM = {
    'Fe': 0.93, 'Co': 0.99, 'Ni': 1.01, 'Mn': 0.77, 'Cr': 0.74, 'V': 0.65, 'Ti': 0.55
}

# Rare-earth Stevens factor sign (+1 = oblate / easy-axis preference, -1 = prolate / easy-plane)
RE_ANISOTROPY_SIGN = {
    'Nd': 1.0, 'Sm': 1.0, 'Er': 1.0, 'Tm': 1.0, 'Yb': 1.0,
    'Pr': -1.0, 'Tb': -1.0, 'Dy': -1.0, 'Ho': -1.0, 'Gd': 0.0
}

# Frustrated space group numbers (triangular, kagome, pyrochlore, and frustrated FCC)
FRUSTRATED_SPACE_GROUPS = {166, 167, 194, 227, 225, 216, 221}

# Uniaxial crystal systems favoring magnetocrystalline anisotropy (K1)
UNIAXIAL_SYSTEMS = {'hexagonal', 'tetragonal', 'trigonal'}

# Coordination number proxy by crystal system for mean-field exchange scaling
COORDINATION_BY_SYSTEM = {
    'cubic': 12.0,
    'hexagonal': 12.0,
    'tetragonal': 8.0,
    'trigonal': 8.0,
    'orthorhombic': 6.0,
    'monoclinic': 6.0,
    'triclinic': 4.0
}

# Shannon effective ionic radii (Å) for octahedral coordination (CN=6)
SHANNON_RADII_CN6 = {
    'Fe': 0.65, 'Co': 0.65, 'Ni': 0.69, 'Mn': 0.65, 'Cr': 0.62, 'Ti': 0.61, 'V': 0.64,
    'La': 1.03, 'Ba': 1.35, 'Sr': 1.18, 'Ca': 1.00, 'Y': 0.90, 'Nd': 0.98, 'Sm': 0.96,
    'Bi': 1.03, 'Al': 0.54, 'Mg': 0.72, 'Zn': 0.74, 'Cu': 0.73, 'Nb': 0.64, 'Ta': 0.64,
    'O': 1.40
}

# Analytical de Gennes factor (g_J - 1)^2 * J(J + 1) for 4f rare-earth ions
# Directly scales Curie and Néel temperatures in RKKY-coupled systems
DE_GENNES_FACTORS = {
    'La': 0.0,
    'Ce': 0.18,
    'Pr': 0.80,
    'Nd': 1.84,
    'Pm': 2.25,
    'Sm': 4.46,
    'Eu': 7.88,  # Weighted mean between Eu2+ (15.75) and Eu3+ (0.0)
    'Gd': 15.75,
    'Tb': 10.50,
    'Dy': 7.08,
    'Ho': 4.50,
    'Er': 2.55,
    'Tm': 1.17,
    'Yb': 0.32,
    'Lu': 0.0,
    'Y': 0.0,
    'Sc': 0.0
}

# Valence electron counts for computing exact Valence Electron Concentration (VEC)
# and Slater-Pauling curve coordinates for transition metal alloys
VALENCE_ELECTRON_COUNT = {
    'H': 1, 'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,
    'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
    'Sc': 3, 'Y': 3, 'La': 3, 'Lu': 3,
    'Ti': 4, 'Zr': 4, 'Hf': 4,
    'V': 5, 'Nb': 5, 'Ta': 5,
    'Cr': 6, 'Mo': 6, 'W': 6,
    'Mn': 7, 'Tc': 7, 'Re': 7,
    'Fe': 8, 'Ru': 8, 'Os': 8,
    'Co': 9, 'Rh': 9, 'Ir': 9,
    'Ni': 10, 'Pd': 10, 'Pt': 10,
    'Cu': 11, 'Ag': 11, 'Au': 11,
    'Zn': 12, 'Cd': 12, 'Hg': 12,
    'B': 3, 'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3,
    'C': 4, 'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4,
    'N': 5, 'P': 5, 'As': 5, 'Sb': 5, 'Bi': 5,
    'O': 6, 'S': 6, 'Se': 6, 'Te': 6,
    'F': 7, 'Cl': 7, 'Br': 7, 'I': 7,
    'Ce': 4, 'Pr': 3, 'Nd': 3, 'Pm': 3, 'Sm': 3, 'Eu': 2, 'Gd': 3,
    'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3, 'Yb': 2
}



