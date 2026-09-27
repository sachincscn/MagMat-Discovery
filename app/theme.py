"""High-Contrast, Ultra-Thick Visual System for MagMat Discovery.

Focused on data-dense visual dashboards, big thick typography,
minimal text noise, and physical schematics.
"""

import streamlit as st

# ── High-Contrast Electric Color Palette ──────────────────────────────
FM = "#FF3366"          # Vivid Crimson — Ferromagnetic
AFM = "#00D2FF"         # Electric Azure — Antiferromagnetic
NM = "#A855F7"          # Electric Purple — Non-Magnetic

ACCENT = "#FF6B00"      # Blazing Neon Orange
TEMP = "#FFB703"        # Electric Amber
MOMENT = "#00F59B"      # Neon Mint / Emerald
HARD = "#10B981"        # Radiant Emerald
CRITICAL = "#F43F5E"    # Electric Rose

INK_PURE = "#FFFFFF"    # Pure White
INK_LIGHT = "#F1F5F9"   # Light Silver
MUTED = "#CBD5E1"       # Bold Silver
FAINT = "#94A3B8"       # Secondary annotations
PAGE_BG = "#07090E"     # Deep obsidian canvas
CARD_BG = "#111827"     # Deep solid card
CARD_BORDER = "rgba(255, 255, 255, 0.28)"

SEQ = [ACCENT, AFM, MOMENT, NM, TEMP, "#EC4899", "#38BDF8"]
PHASE_COLORS = {"FM": FM, "AFM": AFM, "NM": NM}

# ── Plotly Ultra-High Contrast Styling ────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15, 23, 42, 0.85)",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color="#FFFFFF"),
    margin=dict(l=16, r=16, t=32, b=16),
    hoverlabel=dict(
        bgcolor="#0B0F17",
        font_size=13,
        font_family="Inter, sans-serif",
        font_color="#FFFFFF",
        bordercolor=ACCENT
    ),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, x=0,
        font=dict(size=12, family="Inter, sans-serif", color="#FFFFFF"),
        bgcolor="rgba(0,0,0,0)"
    ),
)

# ── CSS with Ultra-Thick Fonts & Zero Clutter (Hybrid Theme by Default) ─
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@600;700;800;900&family=JetBrains+Mono:wght@600;700;800;900&display=swap');

/* Force hybrid dark-light atmospheric canvas across all browser environments */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
    background: radial-gradient(1200px circle at 50% -80px, #1E293B 0%, #0F172A 50%, #080C14 100%) fixed !important;
    color: #F8FAFC !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    letter-spacing: -0.015em !important;
}

#MainMenu, footer, header { visibility: hidden; height: 0; }
.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 3.0rem !important;
    max-width: 1560px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.90) 100%) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 16px !important;
    padding: 1.5rem 1.4rem !important;
    box-shadow: 0 16px 36px -6px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
}

/* Ultra-thick headings */
h1, h2, h3, h4 {
    color: #FFFFFF !important;
    font-weight: 900 !important;
    letter-spacing: -0.04em !important;
}

p, span, label, div {
    color: #FFFFFF;
    font-weight: 600;
}

/* Monospace numbers & formulas */
.num, .v, table, .stDataFrame, code {
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum" 1;
}
.chem-code, .formula-display {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    font-weight: 900 !important;
    letter-spacing: -0.01em;
}

/* ── Hero Banner (Compact & Bold) ──────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, rgba(255, 107, 0, 0.20) 0%, #111827 50%, rgba(0, 210, 255, 0.14) 100%);
    border: 2px solid rgba(255, 255, 255, 0.25);
    border-radius: 12px;
    padding: 0.75rem 1.25rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.7);
}
.hero-banner h1 {
    font-size: 2.0rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.035em !important;
    margin: 0 0 0.2rem 0 !important;
    line-height: 1.15 !important;
    color: #FFFFFF !important;
}
.hero-banner p {
    color: #CBD5E1 !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
    font-weight: 600 !important;
    line-height: 1.35 !important;
}

/* ── Section Dividers & Headers ────────────────────────────────── */
.sec-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1.1rem 0 0.5rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid rgba(255, 255, 255, 0.2);
}
.sec-title {
    font-size: 1.02rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #FFFFFF;
    font-weight: 900;
}
.sec-tag {
    font-size: 0.82rem;
    color: #FFB703;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── High-Contrast Metric Deck ─────────────────────────────────── */
.metric-deck {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 1.0rem;
    margin: 0.8rem 0 1.4rem 0;
}
.metric-card {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.88) 0%, rgba(15, 23, 42, 0.96) 100%);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 14px;
    padding: 0.95rem 1.05rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 28px -4px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.14);
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: var(--accent-bar, #FF6B00);
    box-shadow: 0 16px 36px -4px rgba(0, 0, 0, 0.7), 0 0 22px rgba(255, 107, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.25);
}
.metric-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--accent-bar, #FF6B00);
    box-shadow: none;
}
.metric-card .k {
    font-size: 0.78rem;
    color: #E2E8F0;
    font-weight: 850;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.30rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.metric-card .v {
    font-size: 2.15rem;
    font-weight: 950;
    letter-spacing: -0.04em;
    line-height: 1.1;
    color: var(--accent-bar, #FFFFFF);
    margin-bottom: 0.20rem;
    text-shadow: none;
}
.metric-card .u {
    font-size: 0.82rem;
    color: #CBD5E1;
    font-weight: 800;
}
.metric-card .meta {
    margin-top: 0.45rem;
    font-size: 0.80rem;
    color: #E2E8F0;
    line-height: 1.35;
    font-weight: 650;
    border-top: 1px solid rgba(255, 255, 255, 0.16);
    padding-top: 0.35rem;
}

/* ── Phase Badges & Status Chips ───────────────────────────────── */
.phase-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.35rem 0.95rem;
    border-radius: 9999px;
    font-weight: 900;
    font-size: 0.90rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.phase-fm {
    background: #FF3366;
    color: #FFFFFF;
    box-shadow: 0 0 16px rgba(255, 51, 102, 0.7);
}
.phase-afm {
    background: #00D2FF;
    color: #000000;
    box-shadow: 0 0 16px rgba(0, 210, 255, 0.7);
}
.phase-nm {
    background: #A855F7;
    color: #FFFFFF;
    box-shadow: 0 0 16px rgba(168, 85, 247, 0.7);
}

/* ── Probability Distribution ─────────────────────────────────── */
.prob-bar-wrap {
    margin: 0.7rem 0 1.2rem 0;
    background: #111827;
    border: 2px solid rgba(255, 255, 255, 0.25);
    border-radius: 14px;
    padding: 0.9rem 1.2rem;
}
.prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.86rem;
    color: #FFFFFF;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.5rem;
}
.prob-track {
    display: flex;
    height: 14px;
    border-radius: 9999px;
    overflow: hidden;
    background: #000000;
    border: 1px solid rgba(255, 255, 255, 0.25);
}
.prob-seg {
    height: 100%;
}
.prob-legend {
    display: flex;
    gap: 1.5rem;
    margin-top: 0.55rem;
    font-size: 0.86rem;
    color: #FFFFFF;
    font-weight: 800;
}
.prob-legend-item {
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.prob-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}

/* ── Hardness Scale Gauge ──────────────────────────────────────── */
.scale-box {
    background: #111827;
    border: 2px solid rgba(255, 255, 255, 0.25);
    border-radius: 14px;
    padding: 1.0rem 1.2rem;
    margin: 0.8rem 0;
}
.scale-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.88rem;
    color: #FFFFFF;
    font-weight: 850;
    margin-bottom: 0.55rem;
}
.scale-header b {
    font-size: 1.1rem;
    font-weight: 900;
}
.scale-track {
    position: relative;
    height: 12px;
    border-radius: 9999px;
    background: linear-gradient(90deg, #00D2FF 0%, #00D2FF 22%, #FFB703 22%, #FFB703 55%, #00F59B 55%, #00F59B 100%);
}
.scale-mark {
    position: absolute;
    top: -6px;
    width: 4px;
    height: 24px;
    background: #FFFFFF;
    border-radius: 2px;
    box-shadow: 0 0 12px rgba(255, 255, 255, 1.0);
}
.scale-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 0.5rem;
    font-size: 0.80rem;
    color: #FFFFFF;
    font-weight: 800;
}

/* ── Tabs & Form Inputs ────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2.2rem;
    border-bottom: 2.5px solid rgba(255, 255, 255, 0.25);
    margin-bottom: 1.2rem;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding: 0 0.6rem;
    background: transparent;
    font-weight: 850;
    font-size: 1.15rem;
    color: #94A3B8 !important;
}
.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    border-bottom: 3.5px solid #FF6B00 !important;
    text-shadow: 0 0 14px rgba(255, 107, 0, 0.8);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* Buttons with thick bold typography */
.stButton button, .stDownloadButton button {
    border-radius: 10px !important;
    font-weight: 850 !important;
    font-size: 1.0rem !important;
    background: #1E293B !important;
    color: #FFFFFF !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.6) !important;
    transition: all 0.16s ease !important;
}
.stButton button:hover, .stDownloadButton button:hover {
    background: #334155 !important;
    border-color: #FF6B00 !important;
    box-shadow: 0 6px 20px rgba(255, 107, 0, 0.45) !important;
    transform: translateY(-1px);
}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
    background: linear-gradient(135deg, #FF6B00 0%, #EA580C 100%) !important;
    color: #000000 !important;
    font-weight: 950 !important;
    border: none !important;
    box-shadow: 0 6px 22px rgba(255, 107, 0, 0.6) !important;
}
.stButton button[kind="primary"]:hover, .stDownloadButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, #FF8800 0%, #FF6B00 100%) !important;
    box-shadow: 0 8px 28px rgba(255, 107, 0, 0.8) !important;
}

/* Inputs, selects & multiselect tags */
.stTextInput input, .stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    background: #111827 !important;
    border: 2px solid rgba(255, 255, 255, 0.32) !important;
    color: #FFFFFF !important;
    font-size: 1.05rem !important;
    font-weight: 750 !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #FF6B00 !important;
    box-shadow: 0 0 14px rgba(255, 107, 0, 0.5) !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 9999px !important;
    padding: 0.15rem 0.65rem !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35) !important;
}
.stMultiSelect [data-baseweb="tag"] span {
    color: #0F172A !important;
    font-weight: 900 !important;
    font-size: 0.92rem !important;
}
.stMultiSelect [data-baseweb="tag"] svg {
    fill: #0F172A !important;
}

.stDataFrame, [data-testid="stDataEditor"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 2px solid rgba(255, 255, 255, 0.25) !important;
    background-color: #111827 !important;
}

/* Pills & Segmented Control System (Ultra-High Contrast) */
[data-testid="stPills"], [data-testid="stSegmentedControl"] {
    background: #0B0F19 !important;
    border: 1.5px solid rgba(255, 255, 255, 0.20) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    margin: 0.35rem 0 0.85rem 0 !important;
}
[data-testid="stSegmentedControl"] > div, [data-testid="stPills"] > div {
    background: transparent !important;
}
[data-testid="stPills"] button, [data-testid="stSegmentedControl"] button,
[data-testid="stButtonGroup"] button, div[data-baseweb="button-group"] button {
    background-color: #1E293B !important;
    border: 1.5px solid #334155 !important;
    color: #F8FAFC !important;
    font-weight: 850 !important;
    font-size: 0.90rem !important;
    border-radius: 8px !important;
    padding: 0.40rem 0.65rem !important;
    transition: all 0.16s ease-in-out !important;
    box-shadow: none !important;
}
[data-testid="stPills"] button *, [data-testid="stSegmentedControl"] button *,
[data-testid="stButtonGroup"] button *, div[data-baseweb="button-group"] button * {
    color: #F8FAFC !important;
    font-weight: 850 !important;
}
[data-testid="stPills"] button:hover, [data-testid="stSegmentedControl"] button:hover,
[data-testid="stButtonGroup"] button:hover, div[data-baseweb="button-group"] button:hover {
    border-color: #38BDF8 !important;
    background-color: #334155 !important;
    color: #FFFFFF !important;
}
[data-testid="stPills"] button:hover *, [data-testid="stSegmentedControl"] button:hover *,
[data-testid="stButtonGroup"] button:hover *, div[data-baseweb="button-group"] button:hover * {
    color: #FFFFFF !important;
}
[data-testid="stPills"] button[aria-checked="true"], [data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[aria-selected="true"], [data-testid="stSegmentedControl"] button[data-checked="true"],
[data-testid="stButtonGroup"] button[aria-checked="true"], div[data-baseweb="button-group"] button[aria-checked="true"],
[data-testid="stPills"] button[aria-pressed="true"], [data-testid="stSegmentedControl"] button[aria-pressed="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"], div[data-baseweb="button-group"] button[aria-pressed="true"] {
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
    border: 1.5px solid #38BDF8 !important;
    color: #FFFFFF !important;
    font-weight: 950 !important;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.45) !important;
}
[data-testid="stPills"] button[aria-checked="true"] *, [data-testid="stSegmentedControl"] button[aria-checked="true"] *,
[data-testid="stSegmentedControl"] button[aria-selected="true"] *, [data-testid="stSegmentedControl"] button[data-checked="true"] *,
[data-testid="stButtonGroup"] button[aria-checked="true"] *, div[data-baseweb="button-group"] button[aria-checked="true"] *,
[data-testid="stPills"] button[aria-pressed="true"] *, [data-testid="stSegmentedControl"] button[aria-pressed="true"] *,
[data-testid="stButtonGroup"] button[aria-pressed="true"] *, div[data-baseweb="button-group"] button[aria-pressed="true"] * {
    color: #FFFFFF !important;
    font-weight: 950 !important;
}

/* Popover Dark System */
[data-testid="stPopover"] button, [data-testid="stPopoverButton"] button {
    background: #1E293B !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    color: #FFFFFF !important;
    font-weight: 850 !important;
    border-radius: 10px !important;
}
[data-testid="stPopover"] button *, [data-testid="stPopoverButton"] button * {
    color: #FFFFFF !important;
}

/* Architecture Flow Steps */
.flow {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.0rem;
    margin: 1.2rem 0;
}
.fs {
    background: #111827;
    border: 2px solid rgba(255, 255, 255, 0.25);
    border-radius: 16px;
    padding: 1.2rem 1.3rem;
}
.fs .n {
    display: inline-block;
    font-size: 0.76rem;
    color: #000000;
    font-weight: 950;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: #FF6B00;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    margin-bottom: 0.4rem;
}
.fs .t {
    font-weight: 900;
    color: #FFFFFF;
    font-size: 1.1rem;
    margin-bottom: 0.35rem;
}
.fs .d {
    font-size: 0.90rem;
    color: #E2E8F0;
    line-height: 1.5;
    font-weight: 600;
}
</style>
"""


CSS_DARK_OVERRIDE = """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
    background: #07090E !important;
    color: #FFFFFF !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    box-shadow: 0 16px 36px -6px rgba(0, 0, 0, 0.7) !important;
}
.metric-card {
    background: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background-color: #1E293B !important;
    border: 1.5px solid #FF6B00 !important;
}
.stMultiSelect [data-baseweb="tag"] span {
    color: #FFFFFF !important;
}
.stMultiSelect [data-baseweb="tag"] svg {
    fill: #FFFFFF !important;
}
</style>
"""

CSS_LIGHT_OVERRIDE = """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
    background: #F1F5F9 !important;
    color: #0F172A !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    box-shadow: 0 10px 25px -4px rgba(0, 0, 0, 0.08) !important;
}
h1, h2, h3, h4, .sec-title {
    color: #0F172A !important;
}
p, span, label, div {
    color: #1E293B;
}
.metric-card {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    box-shadow: 0 4px 14px -2px rgba(0, 0, 0, 0.05) !important;
}
.metric-card::before {
    box-shadow: none !important;
}
.metric-card .k {
    color: #475569 !important;
    font-weight: 850 !important;
}
.metric-card .v {
    text-shadow: none !important;
    font-weight: 950 !important;
}
.metric-card .meta {
    color: #334155 !important;
    border-top: 1px solid #E2E8F0 !important;
}
.stTextInput input, .stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    color: #0F172A !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background-color: #0F172A !important;
    border: 1.5px solid #0F172A !important;
}
.stMultiSelect [data-baseweb="tag"] span {
    color: #FFFFFF !important;
}
.stMultiSelect [data-baseweb="tag"] svg {
    fill: #FFFFFF !important;
}
.stTabs [data-baseweb="tab"] {
    color: #64748B !important;
}
.stTabs [aria-selected="true"] {
    color: #0F172A !important;
    border-bottom: 3.5px solid #FF6B00 !important;
}
.prob-bar-wrap, .scale-box {
    background: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
}
.prob-bar-label, .scale-header {
    color: #0F172A !important;
}
.stButton button, .stDownloadButton button {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1.5px solid #CBD5E1 !important;
}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
    background: linear-gradient(135deg, #FF6B00 0%, #EA580C 100%) !important;
    color: #FFFFFF !important;
}
</style>
"""


CSS_HYBRID_OVERRIDE = """
<style>
/* ── HYBRID PALETTE: Light Canvas & Inputs, Luxurious Dark Output Panel ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
    background: #F1F5F9 !important;
    color: #0F172A !important;
}

/* Universal High Contrast for Text & Headings on Light Canvas */
p, span, label, div, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span {
    color: #0F172A;
}
.sec-header {
    border-bottom: 2.5px solid #CBD5E1 !important;
}
.sec-title {
    color: #0F172A !important;
    font-size: 1.15rem !important;
    font-weight: 950 !important;
    letter-spacing: 0.03em !important;
}
.sec-tag {
    color: #B45309 !important;
    font-weight: 900 !important;
}
.stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color: #475569 !important;
    font-weight: 750 !important;
    font-size: 0.90rem !important;
}
.metric-card {
    background: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 14px -2px rgba(0, 0, 0, 0.05) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.metric-card:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px -2px rgba(0, 0, 0, 0.10) !important;
}
.metric-card::before {
    box-shadow: none !important;
}
.metric-card .k, .metric-card .k span {
    color: #475569 !important;
    font-size: 0.82rem !important;
    font-weight: 850 !important;
    letter-spacing: 0.03em !important;
}
.metric-card .v {
    text-shadow: none !important;
    font-weight: 950 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
}
.metric-card .u {
    color: #64748B !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
}
.metric-card .meta {
    color: #334155 !important;
    font-size: 0.86rem !important;
    line-height: 1.45 !important;
    font-weight: 550 !important;
    border-top: 1.5px solid #E2E8F0 !important;
    padding-top: 0.45rem !important;
    margin-top: 0.45rem !important;
}

/* ── White Physical Cards for Bordered Containers on Light Canvas ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 16px !important;
    padding: 1.4rem 1.4rem !important;
    box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.06) !important;
}

/* ── Top Header Banner on Light Canvas ────────────────────────────── */
.hero-banner {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.3rem 0 !important;
    margin-bottom: 0.5rem !important;
}
.hero-banner h1 {
    color: #0F172A !important;
    font-size: 2.45rem !important;
    font-weight: 950 !important;
    letter-spacing: -0.04em !important;
    margin: 0 !important;
}
.hero-banner p {
    color: #475569 !important;
    font-size: 1.12rem !important;
    font-weight: 750 !important;
    margin-top: 0.25rem !important;
}
.hero-eyebrow {
    color: #FF6B00 !important;
    font-weight: 850 !important;
}

/* Top Right Theme Selector & Global Selectboxes */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 2px solid #CBD5E1 !important;
    border-radius: 12px !important;
    min-height: 48px !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #0F172A !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
    fill: #0F172A !important;
}

/* BaseWeb Popovers & Listboxes (Dropdown menus) */
div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
ul[role="listbox"] {
    background: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15) !important;
}
li[role="option"],
div[data-baseweb="menu"] li {
    background: #FFFFFF !important;
    color: #0F172A !important;
    font-size: 1.05rem !important;
    font-weight: 750 !important;
    padding: 0.65rem 1.0rem !important;
}
li[role="option"] *,
div[data-baseweb="menu"] li * {
    color: #0F172A !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background: #F1F5F9 !important;
}
li[role="option"]:hover *,
li[role="option"][aria-selected="true"] * {
    color: #FF6B00 !important;
    font-weight: 850 !important;
}

/* ── Navigation Tabs: Crisp Dark Text on Light Canvas ─────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2.2rem !important;
    border-bottom: 2.5px solid #CBD5E1 !important;
    margin-bottom: 2.0rem !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"],
.stTabs button[role="tab"] {
    background: transparent !important;
    border: none !important;
    padding: 0.85rem 0.5rem !important;
}
.stTabs [data-baseweb="tab"] *,
.stTabs button[role="tab"] * {
    color: #475569 !important;
    font-weight: 850 !important;
    font-size: 1.20rem !important;
    letter-spacing: -0.01em !important;
}
.stTabs [data-baseweb="tab"]:hover *,
.stTabs button[role="tab"]:hover * {
    color: #0F172A !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs button[role="tab"][aria-selected="true"] {
    border-bottom: 3.5px solid #FF6B00 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] *,
.stTabs button[role="tab"][aria-selected="true"] * {
    color: #0F172A !important;
    font-weight: 950 !important;
}

/* ══════════════════════════════════════════════════════════════════════
   LEFT INPUT PANEL (Crisp White Physical Card on Light Canvas)
   ══════════════════════════════════════════════════════════════════════ */
div[data-testid="stColumn"]:has(.magmat-input-marker),
.stColumn:has(.magmat-input-marker) {
    background: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 20px !important;
    padding: 1.6rem 1.6rem !important;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.08) !important;
}

/* Left panel headers and titles */
div[data-testid="stColumn"]:has(.magmat-input-marker) .sec-title,
.stColumn:has(.magmat-input-marker) .sec-title {
    color: #0F172A !important;
    font-size: 1.25rem !important;
    font-weight: 950 !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 1.1rem !important;
}

/* ── Universal High-Contrast Labels across Light Canvas ────────────── */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
label,
label p,
label span {
    color: #0F172A !important;
    font-size: 0.96rem !important;
    font-weight: 850 !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 0.35rem !important;
}

/* ── Universal Selectboxes on Light Canvas ─────────────────────────── */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 2px solid #94A3B8 !important;
    border-radius: 10px !important;
    min-height: 46px !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #0F172A !important;
    font-size: 0.98rem !important;
    font-weight: 800 !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
    fill: #0F172A !important;
}

/* ── Universal Multiselect (Filter Phase, Elements, etc.) ──────────── */
.stMultiSelect div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 2px solid #94A3B8 !important;
    border-radius: 10px !important;
    min-height: 48px !important;
}
.stMultiSelect input {
    color: #0F172A !important;
    font-size: 0.98rem !important;
    font-weight: 750 !important;
}
.stMultiSelect input::placeholder {
    color: #64748B !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background-color: #0F172A !important;
    border: 1.5px solid #0F172A !important;
    border-radius: 9999px !important;
    padding: 0.22rem 0.75rem !important;
}
.stMultiSelect [data-baseweb="tag"] span,
.stMultiSelect [data-baseweb="tag"] div,
.stMultiSelect [data-baseweb="tag"] p {
    color: #FFFFFF !important;
    font-weight: 900 !important;
    font-size: 0.96rem !important;
}
.stMultiSelect [data-baseweb="tag"] svg,
.stMultiSelect [data-baseweb="tag"] path {
    fill: #CBD5E1 !important;
}

/* ── Universal Text & Number Inputs (Formula, Element Queries) ─────── */
.stTextInput input,
.stNumberInput input {
    background: #FFFFFF !important;
    border: 2px solid #94A3B8 !important;
    color: #0F172A !important;
    font-size: 1.06rem !important;
    font-weight: 800 !important;
    border-radius: 10px !important;
    min-height: 46px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}
.stTextInput input::placeholder,
.stNumberInput input::placeholder {
    color: #64748B !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}
.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #0284C7 !important;
    box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.25) !important;
}
.stNumberInput button {
    background: #F8FAFC !important;
    border-color: #CBD5E1 !important;
    color: #0F172A !important;
}
.stNumberInput button svg {
    fill: #0F172A !important;
}

/* ── Universal Sliders on Light Canvas ─────────────────────────────── */
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] label p {
    color: #0F172A !important;
    font-size: 0.96rem !important;
    font-weight: 850 !important;
}
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #475569 !important;
    font-weight: 800 !important;
}

/* ── Radios & Checkboxes on Light Canvas ───────────────────────────── */
div[data-testid="stRadio"] label p,
div[data-testid="stCheckbox"] label p {
    color: #0F172A !important;
    font-size: 0.98rem !important;
    font-weight: 800 !important;
}

/* Primary Action Button */
div[data-testid="stColumn"]:has(.magmat-input-marker) .stButton button,
div[data-testid="stColumn"]:has(.magmat-input-marker) .stButton button[kind="primary"] {
    background: linear-gradient(135deg, #FF6B00 0%, #EA580C 100%) !important;
    color: #FFFFFF !important;
    font-size: 1.22rem !important;
    font-weight: 950 !important;
    height: 58px !important;
    border-radius: 14px !important;
    border: none !important;
    box-shadow: 0 10px 26px -2px rgba(255, 107, 0, 0.50) !important;
    letter-spacing: 0.02em !important;
}
div[data-testid="stColumn"]:has(.magmat-input-marker) .stButton button:hover {
    box-shadow: 0 14px 34px -2px rgba(255, 107, 0, 0.65) !important;
    transform: translateY(-1px) !important;
}

/* Universal High-Contrast Download Buttons across Hybrid Mode */
.stDownloadButton button,
div[data-testid="stDownloadButton"] button {
    background: #0F172A !important;
    border: 1.5px solid #334155 !important;
    border-radius: 12px !important;
    min-height: 48px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12) !important;
    transition: all 0.15s ease !important;
}
.stDownloadButton button *,
.stDownloadButton button p,
.stDownloadButton button span,
.stDownloadButton button div,
div[data-testid="stDownloadButton"] button * {
    color: #FFFFFF !important;
    font-weight: 850 !important;
    font-size: 1.02rem !important;
}
.stDownloadButton button:hover,
div[data-testid="stDownloadButton"] button:hover {
    background: #1E293B !important;
    border-color: #0284C7 !important;
}
.stDownloadButton button:hover *,
.stDownloadButton button:hover p,
.stDownloadButton button:hover span,
div[data-testid="stDownloadButton"] button:hover * {
    color: #38BDF8 !important;
}

/* ══════════════════════════════════════════════════════════════════════
   RIGHT OUTPUT PANEL (Luxurious Elevated Dark Card)
   ══════════════════════════════════════════════════════════════════════ */
div[data-testid="stColumn"]:has(.magmat-output-marker),
.stColumn:has(.magmat-output-marker) {
    background: linear-gradient(180deg, #0F172A 0%, #080D1A 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 24px !important;
    padding: 2.4rem 2.2rem !important;
    box-shadow: 0 24px 60px -10px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
    color: #FFFFFF !important;
}

/* All text & labels inside right output panel: WHITE */
div[data-testid="stColumn"]:has(.magmat-output-marker) h1,
div[data-testid="stColumn"]:has(.magmat-output-marker) h2,
div[data-testid="stColumn"]:has(.magmat-output-marker) h3,
div[data-testid="stColumn"]:has(.magmat-output-marker) h4,
div[data-testid="stColumn"]:has(.magmat-output-marker) p,
div[data-testid="stColumn"]:has(.magmat-output-marker) span,
div[data-testid="stColumn"]:has(.magmat-output-marker) div,
div[data-testid="stColumn"]:has(.magmat-output-marker) label,
div[data-testid="stColumn"]:has(.magmat-output-marker) [data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
}

/* Metric Cards inside Dark Output Panel */
div[data-testid="stColumn"]:has(.magmat-output-marker) .metric-card {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.92) 0%, rgba(15, 23, 42, 0.98) 100%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 16px !important;
    padding: 1.10rem 0.70rem !important;
    box-shadow: 0 12px 28px -4px rgba(0, 0, 0, 0.60), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
}
div[data-testid="stColumn"]:has(.magmat-output-marker) .metric-card .k {
    color: #94A3B8 !important;
    font-size: 0.88rem !important;
    font-weight: 850 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
div[data-testid="stColumn"]:has(.magmat-output-marker) .metric-card .v {
    font-size: clamp(0.92rem, 1.15vw, 1.42rem) !important;
    font-weight: 950 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.15 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
div[data-testid="stColumn"]:has(.magmat-output-marker) .metric-card .u {
    color: #CBD5E1 !important;
    font-size: 0.92rem !important;
    font-weight: 800 !important;
}
div[data-testid="stColumn"]:has(.magmat-output-marker) .metric-card .meta {
    color: #E2E8F0 !important;
    font-size: 0.88rem !important;
    font-weight: 800 !important;
    border-top: 1px solid rgba(255, 255, 255, 0.18) !important;
    padding-top: 0.5rem !important;
    margin-top: 0.6rem !important;
}

/* Secondary Buttons inside Dark Output Panel (e.g. Load to Screener, Export) */
div[data-testid="stColumn"]:has(.magmat-output-marker) .stButton button,
div[data-testid="stColumn"]:has(.magmat-output-marker) .stDownloadButton button {
    background: rgba(30, 41, 59, 0.90) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.30) !important;
    color: #FFFFFF !important;
    font-size: 1.05rem !important;
    font-weight: 850 !important;
    border-radius: 12px !important;
    min-height: 50px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stColumn"]:has(.magmat-output-marker) .stButton button:hover,
div[data-testid="stColumn"]:has(.magmat-output-marker) .stDownloadButton button:hover {
    background: rgba(51, 65, 85, 0.95) !important;
    border-color: #FF6B00 !important;
    color: #FF6B00 !important;
}

/* Dataframe in Discovery Catalog */
.stDataFrame, [data-testid="stDataEditor"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1.5px solid rgba(255, 255, 255, 0.20) !important;
}

/* Tab 4 General Elements in Hybrid Mode */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span,
div[data-testid="stCheckbox"] label,
div[data-testid="stCheckbox"] label p,
div[data-testid="stCheckbox"] label span {
    color: #0F172A !important;
    font-weight: 850 !important;
    font-size: 1.05rem !important;
}

.sec-title {
    color: #0F172A !important;
    font-weight: 950 !important;
}
div[data-testid="stColumn"]:has(.magmat-output-marker) .sec-title {
    color: #FFFFFF !important;
}

table {
    color: #0F172A !important;
    background: #FFFFFF !important;
}
th {
    background: #F8FAFC !important;
    color: #0F172A !important;
    font-weight: 850 !important;
}
td {
    color: #334155 !important;
    font-weight: 600 !important;
}
</style>
"""


def get_plot_layout(theme_mode: str = "Hybrid (Dark & Light)"):
    """Return Plotly layout settings matching the active theme mode."""
    if theme_mode in ("Crisp Light", "Hybrid (Dark & Light)"):
        return dict(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(family="Inter, system-ui, sans-serif", size=13, color="#0F172A"),
            margin=dict(l=16, r=16, t=32, b=16),
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                font_size=13,
                font_family="Inter, sans-serif",
                font_color="#0F172A",
                bordercolor=ACCENT
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, x=0,
                font=dict(size=12, family="Inter, sans-serif", color="#0F172A"),
                bgcolor="rgba(255, 255, 255, 0.90)"
            ),
        )
    return PLOT_LAYOUT


def inject(theme_mode: str = "Hybrid (Dark & Light)"):
    """Inject the ultra-thick visual styles with chosen theme mode."""
    st.markdown(CSS, unsafe_allow_html=True)
    if theme_mode == "Crisp Light":
        st.markdown(CSS_LIGHT_OVERRIDE, unsafe_allow_html=True)
    elif theme_mode == "Deep Obsidian":
        st.markdown(CSS_DARK_OVERRIDE, unsafe_allow_html=True)
    else:
        st.markdown(CSS_HYBRID_OVERRIDE, unsafe_allow_html=True)


def hero(title: str, subtitle: str = "", eyebrow: str = "", badge: str = ""):
    """Render a clean header banner without unnecessary clutter."""
    b_html = f'<div class="hero-badge-pill">{badge}</div>' if badge else ""
    e_html = f'<div class="hero-eyebrow">{eyebrow}</div>' if eyebrow else ""
    top = f'<div class="hero-top-row">{e_html}{b_html}</div>' if (eyebrow or badge) else ""
    p_html = f'<p>{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="hero-banner">{top}<h1>{title}</h1>{p_html}</div>',
        unsafe_allow_html=True
    )


def section(text: str, tag: str = ""):
    """Section header with thick font and optional right tag."""
    t_html = f'<span class="sec-tag">{tag}</span>' if tag else ""
    st.markdown(
        f'<div class="sec-header"><span class="sec-title">{text}</span>{t_html}</div>',
        unsafe_allow_html=True
    )


def metric_card(label: str, value: str, unit: str = "",
                accent: str = ACCENT, meta: str = "",
                status_badge: str = None, uncertainty: str = "",
                conformal_ci: str = "") -> str:
    """Generate HTML for an elevated high-contrast metric tile with thick numbers, UQ, and conformal intervals."""
    b_html = f'<span style="font-size:0.80rem;font-weight:900;color:{accent}">{status_badge}</span>' if status_badge else ""
    meta_html = f'<div class="meta">{meta}</div>' if meta else ""
    unc_html = f' <span style="font-size:0.92rem;font-weight:700;color:#475569;letter-spacing:0;">± {uncertainty}</span>' if uncertainty else ""
    ci_html = f'<div style="font-size:0.80rem;font-weight:850;color:#0284C7;margin-top:0.30rem;letter-spacing:0.01em;">{conformal_ci}</div>' if conformal_ci else ""
    v_style = f"color:{accent};"
    if len(str(value)) >= 8 or "–" in str(value):
        v_style += "font-size:1.62rem;letter-spacing:-0.02em;"
    return (
        f'<div class="metric-card" style="--accent-bar:{accent}">'
        f'<div class="k"><span>{label}</span>{b_html}</div>'
        f'<div class="v" style="{v_style}">{value}{unc_html}</div>'
        f'<div class="u">{unit}</div>'
        f'{ci_html}'
        f'{meta_html}'
        f'</div>'
    )


def phase_pill(phase: str) -> str:
    """Return an ultra-thick pill badge for FM / AFM / NM."""
    p = str(phase).upper()
    cls = "phase-fm" if p == "FM" else ("phase-afm" if p == "AFM" else "phase-nm")
    symbol = "FM" if p == "FM" else ("AFM" if p == "AFM" else "NM")
    return f'<span class="phase-pill {cls}">{symbol}</span>'


def phase_distribution(p_fm: float, p_afm: float, p_nm: float):
    """Render a stacked probability bar for predicted ground states."""
    tot = max(1e-6, p_fm + p_afm + p_nm)
    w_fm = (p_fm / tot) * 100
    w_afm = (p_afm / tot) * 100
    w_nm = (p_nm / tot) * 100

    st.markdown(f"""
<div class="prob-bar-wrap">
  <div class="prob-bar-label">
    <span>Ground-State Ordering Confidence</span>
    <span style="color:#00F59B;">{max(w_fm, w_afm, w_nm):.0f}% CONFIDENT</span>
  </div>
  <div class="prob-track">
    <div class="prob-seg" style="width:{w_fm:.1f}%;background:{FM};" title="FM: {w_fm:.1f}%"></div>
    <div class="prob-seg" style="width:{w_afm:.1f}%;background:{AFM};" title="AFM: {w_afm:.1f}%"></div>
    <div class="prob-seg" style="width:{w_nm:.1f}%;background:{NM};" title="NM: {w_nm:.1f}%"></div>
  </div>
  <div class="prob-legend">
    <div class="prob-legend-item"><span class="prob-dot" style="background:{FM}"></span>FM: {w_fm:.0f}%</div>
    <div class="prob-legend-item"><span class="prob-dot" style="background:{AFM}"></span>AFM: {w_afm:.0f}%</div>
    <div class="prob-legend-item"><span class="prob-dot" style="background:{NM}"></span>NM: {w_nm:.0f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)


def stats(items):
    """Render stats cards in a responsive grid."""
    cards = []
    for item in items:
        label = item[0]
        value = item[1]
        unit = item[2]
        accent = item[3] if len(item) > 3 else ACCENT
        cards.append(metric_card(label, value, unit, accent=accent))
    st.markdown(f'<div class="metric-deck">{"".join(cards)}</div>', unsafe_allow_html=True)


def hardness_scale(kappa: float, kappa_upper: float = None):
    """Interactive hardness gauge: soft / semi-hard / hard band."""
    def pos(v):
        return max(0.0, min(1.0, v / 2.0)) * 100

    marks = f'<div class="scale-mark" style="left:{pos(kappa):.1f}%"></div>'
    span = ""
    if kappa_upper is not None and abs(kappa_upper - kappa) > 1e-9:
        lo, hi = sorted((pos(kappa), pos(kappa_upper)))
        span = (f'<div style="position:absolute;top:0;left:{lo:.1f}%;'
                f'width:{hi - lo:.1f}%;height:12px;border-radius:9999px;'
                f'background:rgba(255,107,0,0.75)"></div>')
        marks += f'<div class="scale-mark" style="left:{pos(kappa_upper):.1f}%"></div>'

    regime = "Hard Magnet (κ ≥ 1.0)" if kappa >= 1.0 else ("Semi-Hard (0.1 ≤ κ < 1.0)" if kappa >= 0.1 else "Soft Magnet (κ < 0.1)")
    regime_color = HARD if kappa >= 1.0 else (TEMP if kappa >= 0.1 else AFM)

    st.markdown(f"""
<div class="scale-box">
  <div class="scale-header">
    <span>Magnetic Hardness Parameter κ = √(K₁ / μ₀Mₛ²)</span>
    <b style="color:{regime_color}">{kappa:.2f} · {regime}</b>
  </div>
  <div class="scale-track">{span}{marks}</div>
  <div class="scale-labels">
    <span>0 · Soft</span>
    <span>0.1 · Semi-Hard</span>
    <span>1.0 · Permanent Magnet</span>
    <span>2.0+ (Nd₂Fe₁₄B, SmCo₅)</span>
  </div>
</div>
""", unsafe_allow_html=True)
