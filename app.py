"""
LearnWise AI - Multimodal Source-Grounded Study Companion & Adaptive Tutor
Track D: Personalized Tutoring & Adaptive Learning
==========================================================================
UI/UX Redesign: Premium purple/mauve/white visual identity
PANTONE P 95-16 C → #32004B (Primary Deep Purple)
PANTONE P 90-14 C → #784B79 (Secondary Mauve)
PANTONE P 1-1 C  → #FFFEF9 (Warm Off-White / White)
"""

import io
import re
import random
import json
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Local Helper Modules
from bkt_model import LearnerProfileManager, DEFAULT_P_L0
from multimodal_loader import (
    process_pdf,
    process_pptx,
    process_video_or_transcript,
    split_into_sentences,
    format_seconds
)
from evaluator import (
    run_system_benchmark,
    run_simulated_student_evaluation,
    DEFAULT_BENCHMARK_TEST_SET
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LearnWise AI — Your Personal Learning Universe",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PANTONE-BASED COLOUR SYSTEM & GLOBAL CSS
# ============================================================
# PANTONE P 95-16 C  → #32004B  Deep Primary Purple
# PANTONE P 90-14 C  → #784B79  Secondary Mauve-Purple
# PANTONE P 1-1 C    → #FFFEF9  Warm Off-White
#
# Derived tints and shades kept strictly within the purple/mauve/white family:
#   Purple 80%  → #5A1A6B   mid-tone for gradients
#   Purple 50%  → #9835B5   accent buttons
#   Purple 20%  → #EFD6FF   very light lavender for card backgrounds
#   Purple 10%  → #F7EAFF   page background tint
#   Mauve 80%   → #9A6A9B   lighter mauve
#   Mauve 20%   → #EDD9EE   soft mauve card accent
#   Deep Dark   → #1A0030   near-black for text on light surfaces

COLORS = {
    "primary":       "#32004B",  # PANTONE P 95-16 C
    "primary_mid":   "#5A1A6B",  # 80% primary tint
    "primary_light": "#9835B5",  # vivid purple for accents
    "primary_pale":  "#EFD6FF",  # card bg lavender
    "primary_ghost": "#F7EAFF",  # page background
    "secondary":     "#784B79",  # PANTONE P 90-14 C
    "secondary_mid": "#9A6A9B",  # lighter mauve
    "secondary_pale":"#EDD9EE",  # soft mauve
    "white":         "#FFFEF9",  # PANTONE P 1-1 C
    "text_dark":     "#1A0030",  # near-black for body text
    "text_mid":      "#4A2060",  # secondary text
    "text_muted":    "#7B5A8A",  # muted captions
    # Semantic – kept subtle so they don't clash with the purple palette
    "success":       "#2D6A4F",  # dark green
    "success_bg":    "#D8F3DC",
    "warning":       "#7B4D00",  # dark amber
    "warning_bg":    "#FFF3CD",
    "error":         "#6B1A1A",  # dark red
    "error_bg":      "#FADADD",
    "info_bg":       "#EDD9EE",  # mauve for info
}

_CSS = f"""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Global reset ── */
*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: {COLORS['primary_ghost']};
    color: {COLORS['text_dark']};
}}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {COLORS['primary']} 0%, {COLORS['primary_mid']} 60%, {COLORS['secondary']} 100%);
    border-right: none;
    box-shadow: 4px 0 24px rgba(50,0,75,0.18);
}}
[data-testid="stSidebar"] * {{ color: {COLORS['white']} !important; }}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {{
    color: {COLORS['white']} !important;
    font-weight: 700;
}}
[data-testid="stSidebar"] .stCaption {{
    color: rgba(255,254,249,0.70) !important;
    font-size: 0.78rem;
}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {{
    background: rgba(255,254,249,0.10);
    border: 1.5px dashed rgba(255,254,249,0.45);
    border-radius: 12px;
    padding: 8px;
}}
[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {{
    border-color: rgba(255,254,249,0.80);
    background: rgba(255,254,249,0.15);
}}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextArea label,
[data-testid="stSidebar"] .stCheckbox label span {{
    color: rgba(255,254,249,0.90) !important;
    font-size: 0.82rem;
}}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
[data-testid="stSidebar"] textarea {{
    background: rgba(255,254,249,0.12) !important;
    border: 1px solid rgba(255,254,249,0.35) !important;
    border-radius: 8px;
    color: {COLORS['white']} !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(255,254,249,0.20) !important;
}}
/* Sidebar expander */
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background: rgba(255,254,249,0.08);
    border: 1px solid rgba(255,254,249,0.20);
    border-radius: 10px;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
    color: {COLORS['white']} !important;
    font-size: 0.82rem;
}}

/* ── Primary button ── */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {COLORS['primary_light']}, {COLORS['secondary']});
    color: {COLORS['white']} !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.90rem;
    padding: 0.55rem 1.4rem;
    box-shadow: 0 4px 14px rgba(120,75,121,0.35);
    transition: all 0.22s ease;
    letter-spacing: 0.01em;
}}
.stButton > button[kind="primary"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(120,75,121,0.50);
    background: linear-gradient(135deg, {COLORS['secondary']}, {COLORS['primary_light']});
}}
.stButton > button[kind="primary"]:active {{
    transform: translateY(0);
}}

/* Secondary button */
.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {{
    background: {COLORS['white']};
    color: {COLORS['primary']} !important;
    border: 1.5px solid {COLORS['secondary']};
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.88rem;
    transition: all 0.20s ease;
}}
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {{
    background: {COLORS['primary_pale']};
    border-color: {COLORS['primary']};
    transform: translateY(-1px);
}}

/* Sidebar button override */
[data-testid="stSidebar"] .stButton > button {{
    background: rgba(255,254,249,0.18) !important;
    color: {COLORS['white']} !important;
    border: 1.5px solid rgba(255,254,249,0.40) !important;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.20s ease;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, rgba(152,53,181,0.85), rgba(120,75,121,0.85)) !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(50,0,75,0.40) !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,254,249,0.30) !important;
}}

/* ── Main area header strip ── */
[data-testid="stAppViewContainer"] > .main > div:first-child {{
    padding-top: 0;
}}

/* ── Streamlit tabs ── */
[data-testid="stTabs"] [role="tablist"] {{
    background: {COLORS['white']};
    border-radius: 14px;
    padding: 4px 6px;
    gap: 4px;
    border: 1.5px solid {COLORS['primary_pale']};
    box-shadow: 0 2px 8px rgba(50,0,75,0.07);
}}
[data-testid="stTabs"] [role="tab"] {{
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.85rem;
    color: {COLORS['text_mid']};
    padding: 0.45rem 1.1rem;
    transition: all 0.18s ease;
    border: none !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
    color: {COLORS['white']} !important;
    font-weight: 700;
    box-shadow: 0 3px 10px rgba(50,0,75,0.28);
}}
[data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]) {{
    background: {COLORS['primary_pale']};
    color: {COLORS['primary']};
}}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: {COLORS['white']};
    border: 1.5px solid {COLORS['primary_pale']};
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 12px rgba(50,0,75,0.07);
    transition: transform 0.20s ease, box-shadow 0.20s ease;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(50,0,75,0.13);
}}
[data-testid="stMetricLabel"] {{
    color: {COLORS['text_muted']} !important;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetricValue"] {{
    color: {COLORS['primary']} !important;
    font-size: 1.75rem;
    font-weight: 800;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.80rem;
    font-weight: 600;
}}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    background: {COLORS['white']};
    border: 1.5px solid {COLORS['primary_pale']};
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(50,0,75,0.05);
    overflow: hidden;
}}
[data-testid="stExpander"] summary {{
    font-weight: 600;
    color: {COLORS['primary']} !important;
    padding: 0.75rem 1rem;
}}
[data-testid="stExpander"] summary:hover {{
    background: {COLORS['primary_ghost']};
}}

/* ── Form inputs ── */
.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {{
    border: 1.5px solid {COLORS['secondary_pale']} !important;
    border-radius: 10px !important;
    background: {COLORS['white']} !important;
    color: {COLORS['text_dark']} !important;
    font-size: 0.90rem;
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
    border-color: {COLORS['secondary']} !important;
    box-shadow: 0 0 0 3px rgba(120,75,121,0.18) !important;
    outline: none !important;
}}
.stSelectbox [data-baseweb="select"]:focus-within {{
    border-color: {COLORS['secondary']} !important;
    box-shadow: 0 0 0 3px rgba(120,75,121,0.18) !important;
}}
.stSelectbox label, .stTextInput label, .stTextArea label,
.stSlider label, .stRadio label > div:first-child,
.stCheckbox label span:first-child {{
    color: {COLORS['text_mid']} !important;
    font-weight: 600;
    font-size: 0.85rem;
}}

/* ── Radio buttons ── */
.stRadio [data-testid="stWidgetLabel"] {{
    color: {COLORS['text_mid']} !important;
    font-weight: 600;
    font-size: 0.85rem;
}}
.stRadio label {{
    background: {COLORS['white']};
    border: 1.5px solid {COLORS['secondary_pale']};
    border-radius: 10px;
    padding: 0.65rem 1.0rem;
    margin: 3px 0;
    transition: all 0.18s ease;
    cursor: pointer;
}}
.stRadio label:hover {{
    border-color: {COLORS['secondary']};
    background: {COLORS['primary_ghost']};
}}
.stRadio [aria-checked="true"] label {{
    border-color: {COLORS['secondary']};
    background: {COLORS['primary_pale']};
    font-weight: 600;
}}

/* ── Alert / status boxes ── */
[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
[data-testid="stInfo"] {{
    background: {COLORS['info_bg']} !important;
    border-left: 4px solid {COLORS['secondary']} !important;
    color: {COLORS['text_dark']} !important;
    border-radius: 0 10px 10px 0;
}}
[data-testid="stAlert"][data-baseweb="notification"][kind="success"],
[data-testid="stSuccess"] {{
    background: {COLORS['success_bg']} !important;
    border-left: 4px solid {COLORS['success']} !important;
    color: {COLORS['success']} !important;
    border-radius: 0 10px 10px 0;
}}
[data-testid="stAlert"][data-baseweb="notification"][kind="warning"],
[data-testid="stWarning"] {{
    background: {COLORS['warning_bg']} !important;
    border-left: 4px solid {COLORS['warning']} !important;
    color: {COLORS['warning']} !important;
    border-radius: 0 10px 10px 0;
}}
[data-testid="stAlert"][data-baseweb="notification"][kind="error"],
[data-testid="stError"] {{
    background: {COLORS['error_bg']} !important;
    border-left: 4px solid {COLORS['error']} !important;
    color: {COLORS['error']} !important;
    border-radius: 0 10px 10px 0;
}}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {{
    border: 1.5px solid {COLORS['primary_pale']};
    border-radius: 12px;
    overflow: hidden;
}}
[data-testid="stDataFrame"] th {{
    background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']}) !important;
    color: {COLORS['white']} !important;
    font-weight: 700;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
[data-testid="stDataFrame"] tr:hover td {{
    background: {COLORS['primary_ghost']} !important;
}}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stTickBarMax"] {{
    color: {COLORS['text_muted']};
    font-size: 0.80rem;
}}
[data-testid="stSlider"] [role="slider"] {{
    background: {COLORS['secondary']} !important;
}}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {{
    border-top-color: {COLORS['secondary']} !important;
}}

/* ── Divider ── */
hr {{
    border: none;
    height: 1.5px;
    background: linear-gradient(90deg, transparent, {COLORS['primary_pale']}, transparent);
    margin: 1.5rem 0;
}}

/* ── Caption / small text ── */
.stCaption, [data-testid="stCaptionContainer"] {{
    color: {COLORS['text_muted']} !important;
    font-size: 0.80rem;
}}

/* ── Checkbox ── */
[data-testid="stCheckbox"] {{
    accent-color: {COLORS['secondary']};
}}

/* ── Form border ── */
[data-testid="stForm"] {{
    background: {COLORS['white']};
    border: 1.5px solid {COLORS['primary_pale']};
    border-radius: 16px;
    padding: 1.5rem 1.5rem 1rem;
    box-shadow: 0 2px 12px rgba(50,0,75,0.06);
}}

/* ── Graphviz chart ── */
[data-testid="stGraphVizChart"] svg {{
    border-radius: 14px;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {COLORS['primary_ghost']}; }}
::-webkit-scrollbar-thumb {{ background: {COLORS['secondary_mid']}; border-radius: 6px; }}
::-webkit-scrollbar-thumb:hover {{ background: {COLORS['secondary']}; }}
</style>
"""

# ── Helper HTML card builder ──────────────────────────────────

def card_html(title: str, body: str, icon: str = "", accent: str = None,
              padding: str = "1.2rem 1.4rem") -> str:
    border_color = accent or COLORS["primary_pale"]
    icon_html = f'<span style="font-size:1.5rem;margin-right:0.6rem;">{icon}</span>' if icon else ""
    return f"""
    <div style="
        background:{COLORS['white']};
        border:1.5px solid {border_color};
        border-radius:16px;
        padding:{padding};
        box-shadow:0 2px 14px rgba(50,0,75,0.07);
        margin-bottom:1rem;
        transition:box-shadow 0.20s ease;
    ">
        <div style="display:flex;align-items:flex-start;gap:0.4rem;">
            {icon_html}
            <div style="flex:1;">
                <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.06em;color:{COLORS['text_muted']};margin-bottom:0.35rem;">
                    {title}
                </div>
                <div style="color:{COLORS['text_dark']};font-size:0.92rem;line-height:1.55;">
                    {body}
                </div>
            </div>
        </div>
    </div>"""

def badge_html(text: str, color: str = None) -> str:
    bg = color or COLORS["primary_pale"]
    fg = COLORS["primary"]
    return (f'<span style="background:{bg};color:{fg};font-size:0.73rem;font-weight:700;'
            f'padding:0.2rem 0.65rem;border-radius:20px;letter-spacing:0.04em;">{text}</span>')

def section_heading(title: str, subtitle: str = "") -> str:
    sub = (f'<p style="color:{COLORS["text_muted"]};font-size:0.84rem;margin:0.2rem 0 0;">{subtitle}</p>'
           if subtitle else "")
    return f"""
    <div style="margin-bottom:1.4rem;">
        <h2 style="font-size:1.35rem;font-weight:800;color:{COLORS['primary']};
            margin:0;letter-spacing:-0.01em;">{title}</h2>
        {sub}
    </div>"""

def hero_banner(title: str, subtitle: str, emoji: str = "🔮") -> str:
    return f"""
    <div style="
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_mid']} 55%, {COLORS['secondary']} 100%);
        border-radius: 20px;
        padding: 2.4rem 2.4rem 2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(50,0,75,0.22);
        position: relative;
        overflow: hidden;
    ">
        <div style="position:absolute;top:-30px;right:-30px;font-size:9rem;opacity:0.08;
            user-select:none;pointer-events:none;">{emoji}</div>
        <div style="position:relative;z-index:1;">
            <p style="color:rgba(255,254,249,0.72);font-size:0.80rem;font-weight:700;
                text-transform:uppercase;letter-spacing:0.12em;margin:0 0 0.6rem;">
                LearnWise AI  ✦  Your Personal Learning Universe
            </p>
            <h1 style="color:{COLORS['white']};font-size:2.1rem;font-weight:800;
                margin:0 0 0.6rem;line-height:1.2;letter-spacing:-0.02em;">
                {title}
            </h1>
            <p style="color:rgba(255,254,249,0.82);font-size:1.0rem;
                margin:0;line-height:1.55;max-width:640px;">
                {subtitle}
            </p>
        </div>
    </div>"""

def stat_pill(label: str, value: str, color: str = None) -> str:
    bg = color or COLORS["primary_pale"]
    return (f'<div style="display:inline-flex;flex-direction:column;align-items:center;'
            f'background:{bg};border-radius:12px;padding:0.6rem 1.1rem;margin:0.3rem;">'
            f'<span style="font-size:1.4rem;font-weight:800;color:{COLORS["primary"]}">{value}</span>'
            f'<span style="font-size:0.72rem;font-weight:600;color:{COLORS["text_muted"]};'
            f'text-transform:uppercase;letter-spacing:0.05em;margin-top:2px;">{label}</span></div>')

def chat_bubble(role: str, text: str, ts: str = "") -> str:
    is_user = (role == "student")
    bg = COLORS["primary_pale"] if is_user else COLORS["white"]
    border = COLORS["secondary"] if is_user else COLORS["primary_pale"]
    align = "flex-end" if is_user else "flex-start"
    avatar = "🧑‍🎓" if is_user else "🤖"
    label = f"You  {ts}" if is_user else "LearnWise AI Tutor"
    label_color = COLORS["secondary"] if is_user else COLORS["primary"]
    return f"""
    <div style="display:flex;flex-direction:column;align-items:{align};margin:0.85rem 0;">
        <div style="font-size:0.72rem;font-weight:700;color:{label_color};
            margin-bottom:0.3rem;letter-spacing:0.04em;">
            {avatar}  {label}
        </div>
        <div style="
            background:{bg};
            border:1.5px solid {border};
            border-radius:{'14px 4px 14px 14px' if is_user else '4px 14px 14px 14px'};
            padding:0.85rem 1.1rem;
            max-width:82%;
            font-size:0.90rem;
            line-height:1.60;
            color:{COLORS['text_dark']};
            box-shadow:0 2px 8px rgba(50,0,75,0.07);
            white-space:pre-wrap;
        ">{text}</div>
    </div>"""

def difficulty_badge(level: str) -> str:
    colors = {
        "Easy":   (COLORS["success_bg"],  COLORS["success"]),
        "Medium": (COLORS["warning_bg"],  COLORS["warning"]),
        "Hard":   (COLORS["error_bg"],    COLORS["error"]),
    }
    bg, fg = colors.get(level, (COLORS["primary_pale"], COLORS["primary"]))
    return (f'<span style="background:{bg};color:{fg};font-size:0.72rem;font-weight:700;'
            f'padding:0.18rem 0.6rem;border-radius:20px;">{level}</span>')

def mastery_pill(pct: int) -> str:
    if pct >= 75:
        bg, fg, label = COLORS["success_bg"], COLORS["success"], "Mastered"
    elif pct >= 40:
        bg, fg, label = COLORS["warning_bg"], COLORS["warning"], "Developing"
    else:
        bg, fg, label = COLORS["error_bg"],   COLORS["error"],   "Needs Practice"
    return (f'<span style="background:{bg};color:{fg};font-size:0.75rem;font-weight:700;'
            f'padding:0.22rem 0.65rem;border-radius:20px;">{pct}%  {label}</span>')

def plotly_theme(fig, title: str = ""):
    """Apply consistent purple/mauve/white theme to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=COLORS["primary"], family="Inter")),
        paper_bgcolor=COLORS["white"],
        plot_bgcolor=COLORS["primary_ghost"],
        font=dict(family="Inter", color=COLORS["text_dark"]),
        legend=dict(bgcolor=COLORS["white"], bordercolor=COLORS["primary_pale"],
                    borderwidth=1, font=dict(size=11)),
        margin=dict(l=12, r=12, t=45, b=12),
    )
    fig.update_xaxes(gridcolor=COLORS["primary_pale"], zerolinecolor=COLORS["primary_pale"])
    fig.update_yaxes(gridcolor=COLORS["primary_pale"], zerolinecolor=COLORS["primary_pale"])
    return fig


# ============================================================
# CONSTANTS & CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

STOP_WORDS = set("""
a an the and or but if then than so because as of at by for from in into on onto to with
without is am are was were be been being it its this that these those they them their he
she we you i my our your have has had do does did can could should would will may might
not no yes which who whom what when where why how all any each every both few more most
other some such only same also very just about over under between through during before after
use used using based according include includes including example examples figure table
chapter section page method methods result results one two three first second third new
different important following called known make made many much several type types
""".split())

METADATA_PHRASES = [
    "isbn", "all rights reserved", "printed in", "published by",
    "publisher", "copyright", "table of contents", "acknowledgements",
    "acknowledgments", "preface", "foreword", "about the author"
]


# ============================================================
# SESSION STATE & LEARNER PROFILE
# ============================================================

if "profile_manager" not in st.session_state:
    st.session_state.profile_manager = LearnerProfileManager()

profile_mgr: LearnerProfileManager = st.session_state.profile_manager

DEFAULTS = {
    "pages": [],
    "chunks": [],
    "sentences": [],
    "media_type": "pdf",
    "media_bytes": None,
    "course_meta": {"topics": ["General"], "concepts": {}, "prerequisites": []},
    "document_name": "",
    "document_ready": False,
    "chat_history": [],
    "quiz": [],
    "quiz_topic": "",
    "quiz_difficulty": "Medium",
    "quiz_submitted": False,
    "quiz_score": None,
    "quiz_feedbacks": [],
    "quiz_misconceptions": [],
    "used_quiz_sentences": [],
    "tutor_language": "English",
    "allow_supplemental": False,
    "active_flashcard_idx": 0,
    "show_flashcard_answer": False,
    "benchmark_results": None,
    "simulation_results": None,
    "active_tab": "dashboard",
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def reset_learning_session():
    """Clears current in-memory quiz states while preserving student history."""
    st.session_state.chat_history = []
    st.session_state.quiz = []
    st.session_state.quiz_topic = ""
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = None
    st.session_state.quiz_feedbacks = []
    st.session_state.quiz_misconceptions = []
    st.session_state.used_quiz_sentences = []
    for key in list(st.session_state.keys()):
        if str(key).startswith("quiz_answer_"):
            del st.session_state[key]


# ============================================================
# SOURCE RETRIEVAL (TF-IDF & KEYWORD MATCHING)
# ============================================================

def retrieve_passages(question: str, topic: str = None, top_k: int = 4):
    chunks = st.session_state.chunks
    if topic and topic != "All topics":
        chunks = [c for c in chunks if c.get("topic") == topic]
    if not chunks:
        return []
    question_words = set(re.findall(r"\b[a-zA-Z]{2,}\b", question.lower())) - STOP_WORDS
    if not question_words:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([c["text"] for c in chunks])
        q_vec = vectorizer.transform([question])
        scores = cosine_similarity(q_vec, matrix).flatten()
    except Exception:
        return []
    results = []
    for idx in scores.argsort()[::-1]:
        chunk = chunks[idx]
        chunk_words = set(re.findall(r"\b[a-zA-Z]{2,}\b", chunk["text"].lower())) - STOP_WORDS
        matched = question_words & chunk_words
        if float(scores[idx]) > 0.05 and matched:
            results.append({
                **chunk,
                "similarity": float(scores[idx]),
                "overlap_count": len(matched),
                "matched_terms": sorted(matched)
            })
        if len(results) >= top_k:
            break
    return results


# ============================================================
# SOURCE-GROUNDED AI TUTOR (WITH OLLAMA & MULTILINGUAL SUPPORT)
# ============================================================

def answer_question_grounded(question: str, topic: str = "All topics", language: str = "English"):
    if len(question.strip()) < 3:
        return {"supported": False, "message": "Please enter a specific question about your course material.",
                "sources": [], "explanation": "", "key_points": []}

    results = retrieve_passages(question, topic, top_k=4)
    question_terms = set(re.findall(r"\b[a-zA-Z]{2,}\b", question.lower())) - STOP_WORDS

    if not results or results[0]["similarity"] < 0.08 or results[0]["overlap_count"] < min(1, len(question_terms)):
        if not st.session_state.allow_supplemental:
            return {
                "supported": False,
                "message": (
                    "🛑 **Not Covered in Course Material:** I couldn't find matching evidence in your uploaded "
                    "textbook, slides, or lecture videos. To ensure academic trustworthiness, LearnWise AI only "
                    "answers queries grounded in your course material. (You can enable Supplemental AI Knowledge in the sidebar if desired)."
                ),
                "sources": [], "explanation": "", "key_points": []
            }
        else:
            flagged_msg = "⚠️ **Supplemental AI Knowledge (Not Backed by Uploaded Course Material):**\n\n"
            try:
                res = requests.post(OLLAMA_URL,
                    json={"model": OLLAMA_MODEL, "prompt": f"Explain this concept briefly for a student: {question}",
                          "stream": False, "options": {"temperature": 0.2, "num_ctx": 2048}}, timeout=25)
                supp_text = res.json().get("response", "")
            except Exception:
                supp_text = "Could not reach local AI model for supplemental response."
            return {"supported": False, "message": flagged_msg + supp_text,
                    "sources": [], "explanation": supp_text, "key_points": []}

    context_blocks = []
    for idx, s in enumerate(results, 1):
        origin = s.get("origin_label", f"Page {s.get('page', '?')}")
        context_blocks.append(f"[{origin}] {s.get('text', '')}")
    context_str = "\n\n".join(context_blocks)

    lang_instruction = "Respond in clear English."
    if language == "Hindi":
        lang_instruction = "Explain the concepts in clean Hindi (हिंदी), keeping technical English terminology in brackets or English script."
    elif language == "Hinglish":
        lang_instruction = "Explain the concepts in natural Hinglish (conversational Hindi-English blend), keeping technical terms in English."

    prompt = f"""
You are LearnWise AI, an expert, source-grounded academic tutor.
Your task is to explain concepts clearly to a student using ONLY the supplied course material excerpts.

GROUNDING RULES:
1. Base your answer STRICTLY on the facts stated in the excerpts below.
2. Every major claim or step must include an inline bracket citation referencing its source, e.g., [Page 3], [Slide 5], or [Timestamp 02:15].
3. Quote at least one direct short excerpt from the text to verify the source claim.
4. {lang_instruction}
5. Do not invent facts or extrapolate beyond what is documented in the source text.

STUDENT QUESTION:
{question}

COURSE MATERIAL EXCERPTS:
{context_str}
"""

    explanation = ""
    try:
        response = requests.post(OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.15, "num_ctx": 4096}}, timeout=45)
        if response.status_code == 200:
            explanation = response.json().get("response", "").strip()
    except Exception:
        explanation = " ".join([s["text"] for s in results[:2]])

    if not explanation:
        explanation = " ".join([s["text"] for s in results[:2]])

    primary_topic = results[0].get("topic", "General")
    profile_mgr.record_conversation_engagement(primary_topic)

    return {"supported": True, "message": f"Verified Grounded Response ({language}):",
            "sources": results, "explanation": explanation, "key_points": results[:3]}


# ============================================================
# ADAPTIVE QUESTION GENERATION (MCQ, SHORT ANSWER, NUMERICAL)
# ============================================================

def generate_adaptive_questions(topic: str, question_type: str = "MCQ",
                                 difficulty: str = "Medium", count: int = 5) -> List[Dict[str, Any]]:
    items = [item for item in st.session_state.sentences
             if topic == "All topics" or item.get("topic") == topic]
    used = set(st.session_state.used_quiz_sentences)
    available = [item for item in items
                 if item["text"].casefold() not in used
                 and not any(phrase in item["text"].casefold() for phrase in METADATA_PHRASES)
                 and len(item["text"].split()) >= 6]
    random.shuffle(available)
    if not available:
        return []

    sample_size = min(len(available), max(4, count * 2))
    batch = available[:sample_size]
    source_payload = [{"source_id": idx, "origin": s.get("origin_label", f"Page {s.get('page', 1)}"),
                       "topic": s.get("topic", "General"), "text": s.get("text", "")}
                      for idx, s in enumerate(batch)]

    if question_type == "MCQ":
        type_req = f"Generate {count} high-quality Multiple Choice Questions (Difficulty: {difficulty}). Each question must have exactly 4 distinct options, exactly 1 correct answer, and a clear explanation."
        example_format = '{"questions":[{"source_id":0,"question":"Q?","answer":"Correct option","options":["Correct option","Wrong 1","Wrong 2","Wrong 3"],"difficulty":"Medium","explanation":"Explanation."}]}'
    elif question_type == "Numerical":
        type_req = f"Generate {count} Numerical calculation problems (Difficulty: {difficulty}). Return empty options, the exact numerical answer, and step-by-step derivation in explanation."
        example_format = '{"questions":[{"source_id":0,"question":"Calculate X if...?","answer":"1.20","options":[],"difficulty":"Medium","explanation":"Formula: X = ..."}]}'
    else:
        type_req = f"Generate {count} Conceptual Short Answer Questions (Difficulty: {difficulty}). Return empty options, a concise expected answer, and grading criteria in explanation."
        example_format = '{"questions":[{"source_id":0,"question":"Why does...?","answer":"Because...","options":[],"difficulty":"Medium","explanation":"Key grading points."}]}'

    prompt = f"""You are an expert assessment designer for LearnWise AI.
Create educational questions strictly derived from the provided course material.
DIFFICULTY LEVEL: {difficulty}
QUESTION TYPE: {question_type}
RULES: 1. Every question must be directly answerable from its assigned source_id. 2. Questions must be grammatically complete. 3. Output valid JSON only, without markdown backticks.
REQUIREMENTS: {type_req}
OUTPUT FORMAT: {example_format}
SOURCE EXCERPTS: {json.dumps(source_payload, ensure_ascii=False)}"""

    accepted_questions = []
    try:
        res = requests.post(OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "format": "json", "options": {"temperature": 0.15, "num_ctx": 4096}}, timeout=90)
        if res.status_code == 200:
            parsed = json.loads(res.json().get("response", "{}"))
            candidates = parsed.get("questions", [])
            for cand in candidates:
                sid = cand.get("source_id", 0)
                if isinstance(sid, int) and 0 <= sid < len(batch):
                    src = batch[sid]
                    q_text = str(cand.get("question", "")).strip()
                    ans_text = str(cand.get("answer", "")).strip()
                    opts = cand.get("options", [])
                    expl = str(cand.get("explanation", "")).strip()
                    if q_text and ans_text and q_text.endswith("?"):
                        if question_type == "MCQ":
                            if not isinstance(opts, list) or len(opts) != 4 or ans_text not in opts:
                                continue
                        accepted_questions.append({
                            "question": q_text, "question_type": question_type,
                            "difficulty": difficulty, "options": opts, "answer": ans_text,
                            "explanation": expl,
                            "origin_label": src.get("origin_label", f"Page {src.get('page', 1)}"),
                            "origin_type": src.get("origin_type", "page"),
                            "unit_id": src.get("unit_id", 1),
                            "topic": src.get("topic", "General"),
                            "source_sentence": src.get("text", "")
                        })
                        st.session_state.used_quiz_sentences.append(src["text"].casefold())
                        if len(accepted_questions) >= count:
                            break
    except Exception:
        pass
    return accepted_questions


# ============================================================
# AI SEMANTIC GRADING & MISCONCEPTION DETECTION
# ============================================================

def grade_student_answer(question: str, expected_answer: str, student_answer: str,
                          question_type: str, source_context: str) -> Dict[str, Any]:
    cleaned_student = str(student_answer or "").strip()
    if not cleaned_student:
        return {"is_correct": False, "feedback": "No answer was provided.",
                "likely_misconception": "Unattempted question."}

    if question_type == "MCQ":
        is_corr = cleaned_student.casefold() == expected_answer.strip().casefold()
        feedback = "Correct! Your selection matches the verified course definition." if is_corr else (
            f"Incorrect. The course material establishes that: '{expected_answer}'.")
        misconception = "None" if is_corr else f"Confused '{cleaned_student}' with the true principle: '{expected_answer}'"
        return {"is_correct": is_corr, "feedback": feedback, "likely_misconception": misconception}

    if question_type == "Numerical":
        stud_nums = re.findall(r"[-+]?\d*\.?\d+", cleaned_student)
        exp_nums = re.findall(r"[-+]?\d*\.?\d+", expected_answer)
        if stud_nums and exp_nums:
            try:
                diff_pct = abs(float(stud_nums[0]) - float(exp_nums[0])) / max(abs(float(exp_nums[0])), 1e-6)
                if diff_pct <= 0.05:
                    return {"is_correct": True,
                            "feedback": f"Correct numerical calculation! (Got: {stud_nums[0]}, Expected: {exp_nums[0]}).",
                            "likely_misconception": "None"}
            except ValueError:
                pass
        return {"is_correct": False, "feedback": f"Numerical result mismatch. Expected: {expected_answer}.",
                "likely_misconception": "Likely formula application or unit conversion error."}

    prompt = f"""You are an expert academic assessment grader.
QUESTION: {question}
EXPECTED ANSWER: {expected_answer}
SOURCE CONTEXT: {source_context}
STUDENT ANSWER: {cleaned_student}
Evaluate if the student demonstrates accurate understanding. Accept valid alternative phrasings.
OUTPUT FORMAT (JSON only): {{"is_correct": true or false, "feedback": "1-2 sentence cited feedback", "likely_misconception": "None or 1 sentence"}}"""
    try:
        res = requests.post(OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "format": "json", "options": {"temperature": 0.1, "num_ctx": 2048}}, timeout=25)
        if res.status_code == 200:
            parsed = json.loads(res.json().get("response", "{}"))
            return {"is_correct": bool(parsed.get("is_correct", False)),
                    "feedback": parsed.get("feedback", f"Expected: {expected_answer}"),
                    "likely_misconception": parsed.get("likely_misconception", "None")}
    except Exception:
        pass

    s_words = set(re.findall(r"\b[a-z0-9]+\b", cleaned_student.lower())) - STOP_WORDS
    e_words = set(re.findall(r"\b[a-z0-9]+\b", expected_answer.lower())) - STOP_WORDS
    overlap = len(s_words & e_words) / max(1, len(e_words))
    is_corr = overlap >= 0.65
    return {"is_correct": is_corr,
            "feedback": f"Keyword overlap: {int(overlap * 100)}%. Expected: {expected_answer}",
            "likely_misconception": "None" if is_corr else "Incomplete explanation of key components."}


# ============================================================
# INJECT CSS
# ============================================================
st.markdown(_CSS, unsafe_allow_html=True)


# ============================================================
# SIDEBAR: MULTIMODAL INGESTION & LEARNER INTAKE
# ============================================================

with st.sidebar:
    # Brand identity
    st.markdown(f"""
    <div style="padding:1.2rem 0.4rem 1rem;text-align:center;">
        <div style="font-size:2.8rem;margin-bottom:0.3rem;">🔮</div>
        <div style="font-size:1.15rem;font-weight:800;color:{COLORS['white']};
            letter-spacing:-0.01em;">LearnWise AI</div>
        <div style="font-size:0.72rem;color:rgba(255,254,249,0.65);
            text-transform:uppercase;letter-spacing:0.10em;margin-top:2px;">
            Your Personal Learning Universe
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(255,254,249,0.10);border-radius:10px;
        padding:0.5rem 0.85rem;margin-bottom:0.8rem;">
        <div style="font-size:0.70rem;font-weight:700;text-transform:uppercase;
            letter-spacing:0.08em;color:rgba(255,254,249,0.60);margin-bottom:4px;">
            📚 Course Material
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF, PPTX, SRT, VTT, or TXT",
        type=["pdf", "pptx", "mp4", "mov", "mp3", "wav", "srt", "vtt", "txt"],
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    companion_transcript = ""
    if uploaded_file and uploaded_file.name.lower().endswith((".mp4", ".mov", ".mp3", ".wav")):
        st.info("📹 Video/Audio detected! Paste transcript or subtitles below for timestamp citations.")
        companion_transcript = st.text_area(
            "Video Transcript / Subtitles (Optional)",
            placeholder="[00:15] Introduction to Layering\n[01:30] TCP Handshake Mechanism",
            height=90
        )

    with st.expander("⚙️ Manual Topic Mapping (optional)"):
        mapping_text = st.text_area(
            "Assign topics to page ranges (leave blank to auto-discover):",
            placeholder="1-3: Fundamentals\n4-6: Protocols\n7-10: Security",
            height=75,
            label_visibility="collapsed"
        )

    if st.button("🚀 Process & Ingest Material", type="primary", use_container_width=True):
        if not uploaded_file:
            st.error("Please select a course file first.")
        else:
            with st.spinner("Analysing material, extracting diagrams & indexing topics…"):
                raw_bytes = uploaded_file.getvalue()
                fname = uploaded_file.name
                try:
                    if fname.lower().endswith(".pdf"):
                        res_data = process_pdf(raw_bytes, fname, mapping_text if 'mapping_text' in dir() else "")
                    elif fname.lower().endswith(".pptx"):
                        res_data = process_pptx(raw_bytes, fname, mapping_text if 'mapping_text' in dir() else "")
                    else:
                        res_data = process_video_or_transcript(raw_bytes, fname, companion_transcript)
                    st.session_state.pages = res_data["pages"]
                    st.session_state.chunks = res_data["chunks"]
                    st.session_state.sentences = res_data["sentences"]
                    st.session_state.course_meta = res_data["course_meta"]
                    st.session_state.media_type = res_data["media_type"]
                    st.session_state.media_bytes = res_data.get("media_bytes")
                    st.session_state.document_name = fname
                    st.session_state.document_ready = True
                    reset_learning_session()
                    st.success(f"✅ Indexed: {fname}")
                except Exception as err:
                    st.error(f"Processing error: {err}")

    if st.session_state.document_ready:
        fname = st.session_state.document_name
        mtype = st.session_state.media_type
        units_label = "Slides" if mtype == "pptx" else ("Video Cues" if mtype == "video" else "Pages")
        n_pages = len(st.session_state.pages)
        n_chunks = len(st.session_state.chunks)
        n_imgs = sum(len(p.get("images", [])) for p in st.session_state.pages)
        n_topics = len(st.session_state.course_meta.get("topics", []))
        st.markdown(f"""
        <div style="background:rgba(255,254,249,0.12);border-radius:12px;
            padding:0.85rem 1rem;margin:0.8rem 0;
            border:1px solid rgba(255,254,249,0.20);">
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.08em;color:rgba(255,254,249,0.60);margin-bottom:6px;">
                ✦ Active Course
            </div>
            <div style="font-size:0.83rem;font-weight:600;color:{COLORS['white']};
                margin-bottom:6px;word-break:break-all;">{fname}</div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;">
                <span style="background:rgba(255,254,249,0.18);border-radius:8px;
                    padding:2px 8px;font-size:0.73rem;color:{COLORS['white']};">
                    📄 {n_pages} {units_label}
                </span>
                <span style="background:rgba(255,254,249,0.18);border-radius:8px;
                    padding:2px 8px;font-size:0.73rem;color:{COLORS['white']};">
                    🔍 {n_chunks} Chunks
                </span>
                <span style="background:rgba(255,254,249,0.18);border-radius:8px;
                    padding:2px 8px;font-size:0.73rem;color:{COLORS['white']};">
                    🖼️ {n_imgs} Diagrams
                </span>
                <span style="background:rgba(255,254,249,0.18);border-radius:8px;
                    padding:2px 8px;font-size:0.73rem;color:{COLORS['white']};">
                    🏷️ {n_topics} Topics
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Learner Calibration
    st.markdown(f"""
    <div style="font-size:0.70rem;font-weight:700;text-transform:uppercase;
        letter-spacing:0.08em;color:rgba(255,254,249,0.60);
        padding:0.4rem 0 0.3rem;">👤 Learner Calibration</div>
    """, unsafe_allow_html=True)

    intake_level = st.selectbox(
        "Experience Level",
        ["Beginner", "Intermediate", "Advanced"],
        index=["Beginner", "Intermediate", "Advanced"].index(
            profile_mgr.profile.get("experience_level", "Beginner")),
        label_visibility="collapsed"
    )
    if st.button("Set BKT Baseline", use_container_width=True):
        profile_mgr.set_intake_level(intake_level)
        st.success(f"✅ Prior P(L₀) set for {intake_level}")

    st.checkbox(
        "Allow Supplemental AI Knowledge",
        value=st.session_state.allow_supplemental,
        key="allow_supplemental",
        help="When enabled, off-material queries receive a flagged supplemental AI answer instead of strict refusal."
    )

    # Footer
    st.markdown(f"""
    <div style="margin-top:2rem;padding:0.8rem 0;border-top:1px solid rgba(255,254,249,0.15);
        text-align:center;">
        <div style="font-size:0.68rem;color:rgba(255,254,249,0.45);">
            Hackathon 2026 · Track D<br/>
            Powered by Ollama llama3.2 · BKT
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN CONTENT — WELCOME SCREEN (no material loaded)
# ============================================================

if not st.session_state.document_ready:
    st.markdown(hero_banner(
        "Learn smarter. Grow every day.",
        "Your personal AI learning companion helps you understand concepts, practise what you learn, "
        "and discover where to improve — all grounded in your own course material.",
        "🔮"
    ), unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(card_html(
            "Multimodal Knowledge Base", 
            "Upload PDF textbooks, PowerPoint slides, or lecture transcripts (SRT/VTT/TXT). "
            "Topics, concepts, and prerequisites are discovered automatically.",
            "📚", COLORS["secondary_pale"]
        ), unsafe_allow_html=True)
    with col_b:
        st.markdown(card_html(
            "Source-Grounded AI Tutor",
            "Ask questions in English, Hindi, or Hinglish. Every answer cites the exact page, "
            "slide, or video timestamp — no hallucinations, no guesswork.",
            "🤖", COLORS["primary_pale"]
        ), unsafe_allow_html=True)
    with col_c:
        st.markdown(card_html(
            "Adaptive Assessments & BKT",
            "MCQ, Short Answer, and Numerical quizzes adapt to your weakest topics using "
            "Bayesian Knowledge Tracing. Misconceptions are identified after every attempt.",
            "🎯", COLORS["secondary_pale"]
        ), unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;padding:2rem 0 1rem;">
        <p style="color:{COLORS['text_muted']};font-size:0.90rem;">
            ← Upload your course material in the sidebar to begin your personalised learning journey.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ============================================================
# DERIVE RUNTIME STATE
# ============================================================

available_topics = st.session_state.course_meta.get("topics", [])
if not available_topics:
    available_topics = sorted(list({c.get("topic", "General") for c in st.session_state.chunks})) or ["General"]

all_mastery = profile_mgr.get_all_mastery()
weakest_topic = min(all_mastery, key=all_mastery.get) if all_mastery else available_topics[0]
p_l_weak = profile_mgr.get_topic_mastery(weakest_topic)


# ============================================================
# MAIN TABS
# ============================================================

tab_home, tab_tutor, tab_quiz, tab_dashboard, tab_flow, tab_eval = st.tabs([
    "🏠 Dashboard",
    "💬 AI Tutor",
    "📝 Practice Quiz",
    "📊 Learning Progress",
    "🗺️ Course Flow & Revision",
    "🧪 System Evaluation",
])


# ════════════════════════════════════════════════════════════
# TAB 0: DASHBOARD
# ════════════════════════════════════════════════════════════

with tab_home:
    mtype = st.session_state.media_type
    units_label = "Slides" if mtype == "pptx" else ("Video Cues" if mtype == "video" else "Pages")

    st.markdown(hero_banner(
        "Welcome back to your learning universe! ✦",
        f"Currently studying: <strong>{st.session_state.document_name}</strong>  "
        "— Ask questions, practise quizzes, and track your mastery.",
        "📚"
    ), unsafe_allow_html=True)

    # ── Quick Stats Row ──────────────────────────────────────
    total_attempts = sum(d.get("attempts", 0) for d in profile_mgr.profile.get("topic_mastery", {}).values())
    total_correct = sum(d.get("correct", 0) for d in profile_mgr.profile.get("topic_mastery", {}).values())
    avg_mastery = sum(all_mastery.values()) / max(1, len(all_mastery))

    qc1, qc2, qc3, qc4, qc5 = st.columns(5)
    qc1.metric(units_label, len(st.session_state.pages))
    qc2.metric("Search Chunks", len(st.session_state.chunks))
    qc3.metric("Topics", len(available_topics))
    qc4.metric("Questions Attempted", total_attempts)
    qc5.metric("Mean Mastery P(L)", f"{int(avg_mastery * 100)}%")

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown(section_heading("Topic Mastery Overview",
            "Estimated mastery probability P(L) per topic — updated via Bayesian Knowledge Tracing after every question."),
            unsafe_allow_html=True)

        chart_data = [{"Topic": t, "Mastery P(L) %": round(all_mastery.get(t, DEFAULT_P_L0) * 100, 1),
                       "Status": "Mastered (>75%)" if all_mastery.get(t, DEFAULT_P_L0) >= 0.75
                       else ("Developing (40-75%)" if all_mastery.get(t, DEFAULT_P_L0) >= 0.40
                             else "Needs Practice (<40%)")}
                      for t in available_topics]

        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            fig_m = px.bar(df_chart, x="Topic", y="Mastery P(L) %", color="Status",
                color_discrete_map={"Mastered (>75%)":      COLORS["success"],
                                    "Developing (40-75%)":  COLORS["secondary"],
                                    "Needs Practice (<40%)":COLORS["primary"]},
                range_y=[0, 100], text_auto=".1f")
            fig_m = plotly_theme(fig_m, "BKT Mastery by Topic")
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.markdown(card_html("No Mastery Data Yet",
                "Complete your first adaptive quiz to populate the mastery chart.", "📊",
                COLORS["secondary_pale"]), unsafe_allow_html=True)

    with col_right:
        st.markdown(section_heading("Revision Priority"), unsafe_allow_html=True)

        if all_mastery:
            sorted_by_mastery = sorted(available_topics, key=lambda t: all_mastery.get(t, DEFAULT_P_L0))
            for t in sorted_by_mastery[:5]:
                p_l = all_mastery.get(t, DEFAULT_P_L0)
                pct = int(p_l * 100)
                bar_color = COLORS["success"] if pct >= 75 else (COLORS["secondary"] if pct >= 40 else COLORS["primary_light"])
                st.markdown(f"""
                <div style="background:{COLORS['white']};border:1.5px solid {COLORS['primary_pale']};
                    border-radius:12px;padding:0.75rem 1rem;margin-bottom:0.6rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                        <span style="font-size:0.85rem;font-weight:600;color:{COLORS['text_dark']};
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:70%">{t}</span>
                        {mastery_pill(pct)}
                    </div>
                    <div style="background:{COLORS['primary_ghost']};border-radius:4px;height:6px;">
                        <div style="background:{bar_color};width:{pct}%;height:100%;border-radius:4px;
                            transition:width 0.4s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(card_html("No Data Yet",
                "Take your first quiz to see personalised revision priorities here.", "🎯",
                COLORS["secondary_pale"]), unsafe_allow_html=True)

        st.markdown(section_heading("Recommended Next Step"), unsafe_allow_html=True)
        rec_color = COLORS["error_bg"] if p_l_weak < 0.40 else COLORS["warning_bg"]
        rec_border = COLORS["primary_light"] if p_l_weak < 0.40 else COLORS["secondary"]
        rec_action = "Start a focused quiz on this topic to build your foundation." if p_l_weak < 0.40 else "Continue practising to deepen your understanding."
        st.markdown(f"""
        <div style="background:{rec_color};border:1.5px solid {rec_border};
            border-radius:14px;padding:1rem 1.2rem;">
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.06em;color:{rec_border};margin-bottom:4px;">
                🎯 Focus Area
            </div>
            <div style="font-size:1.0rem;font-weight:700;color:{COLORS['text_dark']};margin-bottom:4px;">
                {weakest_topic}
            </div>
            <div style="font-size:0.82rem;color:{COLORS['text_mid']};">{rec_action}</div>
            <div style="margin-top:8px;">{mastery_pill(int(p_l_weak * 100))}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Recent Quiz History ───────────────────────────────────
    attempts_history = profile_mgr.profile.get("quiz_attempts", [])
    if attempts_history:
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        st.markdown(section_heading("Recent Quiz Activity"), unsafe_allow_html=True)
        recent = attempts_history[-6:][::-1]
        hist_df = pd.DataFrame(recent)
        hist_df["Date"] = hist_df["timestamp"].apply(lambda t: datetime.fromtimestamp(t).strftime("%b %d, %H:%M"))
        hist_df["Accuracy (%)"] = hist_df["accuracy"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(
            hist_df[["Date", "topic", "correct", "total", "Accuracy (%)"]].rename(
                columns={"topic": "Topic", "correct": "✅ Correct", "total": "Total"}),
            use_container_width=True, hide_index=True
        )


# ════════════════════════════════════════════════════════════
# TAB 1: AI TUTOR
# ════════════════════════════════════════════════════════════

with tab_tutor:
    st.markdown(hero_banner(
        "Your Source-Grounded AI Tutor",
        "Ask any question about your uploaded course material. "
        "Every answer is cited with exact page, slide, or timestamp references.",
        "🤖"
    ), unsafe_allow_html=True)

    # Controls
    ctrl1, ctrl2, ctrl3 = st.columns([3, 1, 1])
    with ctrl1:
        tutor_scope = st.selectbox("Search within topic:", ["All topics"] + available_topics,
                                    label_visibility="visible")
    with ctrl2:
        st.session_state.tutor_language = st.selectbox(
            "Response language:",
            ["English", "Hindi", "Hinglish"],
            index=["English", "Hindi", "Hinglish"].index(st.session_state.tutor_language)
        )
    with ctrl3:
        lang_icon = {"English": "🇬🇧", "Hindi": "🇮🇳", "Hinglish": "🤝"}
        st.markdown(f"""
        <div style="padding-top:1.6rem;font-size:1.8rem;text-align:center;">
            {lang_icon.get(st.session_state.tutor_language, '🌐')}
        </div>""", unsafe_allow_html=True)

    # Suggested questions
    suggested = [
        "What are the main topics covered in this material?",
        "Explain the most important concept from page 1.",
        "What are the key principles discussed?"
    ]
    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin:0.5rem 0 1rem;">
        <span style="font-size:0.78rem;color:{COLORS['text_muted']};padding-top:4px;">
            💡 Suggested:</span>
    """, unsafe_allow_html=True)
    s_c1, s_c2, s_c3 = st.columns(3)
    for col, q in zip([s_c1, s_c2, s_c3], suggested):
        if col.button(q, use_container_width=True):
            with st.spinner("Searching source material…"):
                resp = answer_question_grounded(q, tutor_scope, st.session_state.tutor_language)
                st.session_state.chat_history.append({
                    "query": q, "response": resp,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Question form
    with st.form("tutor_chat_form", clear_on_submit=True):
        student_query = st.text_input(
            "Ask your course material:",
            placeholder="e.g., How does the TCP handshake establish a connection?",
            label_visibility="visible"
        )
        cols_f = st.columns([5, 1])
        ask_btn = cols_f[1].form_submit_button("Send ➤", type="primary", use_container_width=True)

    if ask_btn and student_query:
        with st.spinner("Searching source material and generating cited explanation…"):
            resp = answer_question_grounded(student_query, tutor_scope, st.session_state.tutor_language)
            st.session_state.chat_history.append({
                "query": student_query, "response": resp,
                "timestamp": datetime.now().strftime("%H:%M")
            })

    # Chat history render
    if not st.session_state.chat_history:
        st.markdown(f"""
        <div style="text-align:center;padding:2.5rem 1rem;color:{COLORS['text_muted']};">
            <div style="font-size:3rem;margin-bottom:0.6rem;">💬</div>
            <div style="font-size:0.95rem;">Ask your first question above to begin learning.</div>
        </div>""", unsafe_allow_html=True)

    for chat in reversed(st.session_state.chat_history):
        ans = chat["response"]
        st.markdown(chat_bubble("student", chat["query"], chat["timestamp"]), unsafe_allow_html=True)

        if ans["supported"]:
            st.markdown(chat_bubble("tutor", ans["explanation"]), unsafe_allow_html=True)

            with st.expander(f"📎 {len(ans['sources'])} Source Citation(s)"):
                for s in ans["sources"]:
                    origin_tag = s.get("origin_label", f"Page {s.get('page', 1)}")
                    sim = s.get("similarity", 0.0)
                    st.markdown(f"""
                    <div style="background:{COLORS['primary_ghost']};border-left:3px solid {COLORS['secondary']};
                        border-radius:0 10px 10px 0;padding:0.7rem 0.9rem;margin-bottom:0.6rem;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="font-size:0.78rem;font-weight:700;color:{COLORS['primary']};">
                                📍 {origin_tag}
                            </span>
                            <span style="font-size:0.72rem;color:{COLORS['text_muted']};">
                                match {sim:.2f}
                            </span>
                        </div>
                        <div style="font-size:0.85rem;color:{COLORS['text_dark']};font-style:italic;">
                            "{s.get('text', '')[:280]}…"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if s.get("origin_type") == "video" and st.session_state.media_bytes:
                        st.video(st.session_state.media_bytes, start_time=s.get("seconds", 0))

            # Diagrams
            all_page_imgs = []
            for s in ans["sources"]:
                p_num = s.get("unit_id") or s.get("page", 1)
                for p in st.session_state.pages:
                    if p.get("unit_id") == p_num or p.get("page") == p_num:
                        all_page_imgs.extend(p.get("images", []))

            if all_page_imgs:
                with st.expander(f"🖼️ {len(all_page_imgs)} Extracted Diagram(s)"):
                    img_cols = st.columns(min(3, len(all_page_imgs)))
                    for idx_i, img in enumerate(all_page_imgs):
                        with img_cols[idx_i % len(img_cols)]:
                            st.image(img["data"], caption=img.get("caption", "Course Diagram"),
                                     use_container_width=True)
        else:
            st.markdown(chat_bubble("tutor", ans["message"]), unsafe_allow_html=True)

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Conversation", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()


# ════════════════════════════════════════════════════════════
# TAB 2: ADAPTIVE ASSESSMENT
# ════════════════════════════════════════════════════════════

with tab_quiz:
    st.markdown(hero_banner(
        "Adaptive Practice Quiz",
        "Questions are generated from your course material, difficulty-tiered, and graded by AI with "
        "misconception detection. Your mastery is updated after every answer via Bayesian Knowledge Tracing.",
        "📝"
    ), unsafe_allow_html=True)

    # Recommendation banner
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{COLORS['primary_pale']},{COLORS['secondary_pale']});
        border:1.5px solid {COLORS['secondary']};border-radius:14px;
        padding:0.85rem 1.2rem;margin-bottom:1rem;display:flex;align-items:center;gap:1rem;">
        <span style="font-size:1.6rem;">🎯</span>
        <div>
            <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.06em;color:{COLORS['secondary']};margin-bottom:2px;">BKT Recommendation</div>
            <div style="font-size:0.92rem;font-weight:600;color:{COLORS['text_dark']};">
                Focus on <strong>{weakest_topic}</strong>
                — Estimated Mastery P(L) = {int(p_l_weak * 100)}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        quiz_topic = st.selectbox("Assessment Topic:",
            available_topics,
            index=available_topics.index(weakest_topic) if weakest_topic in available_topics else 0)
    with col_q2:
        q_type = st.selectbox("Question Format:", ["MCQ", "Short Answer", "Numerical"])
    with col_q3:
        q_diff = st.selectbox("Difficulty Tier:", ["Easy", "Medium", "Hard"], index=1)

    if st.button("✨ Generate Questions", type="primary", use_container_width=True):
        with st.spinner("Generating verified, source-grounded questions…"):
            for k in list(st.session_state.keys()):
                if str(k).startswith("quiz_answer_"):
                    del st.session_state[k]
            st.session_state.quiz = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = None
            st.session_state.quiz_feedbacks = []
            st.session_state.quiz_misconceptions = []
            generated = generate_adaptive_questions(quiz_topic, q_type, q_diff, count=4)
            st.session_state.quiz = generated
            st.session_state.quiz_topic = quiz_topic
            st.session_state.quiz_difficulty = q_diff
            if not generated:
                st.warning("Could not generate enough unique questions for this topic. Try another difficulty or format.")
            else:
                st.rerun()

    # ── Quiz Form ─────────────────────────────────────────────
    if st.session_state.quiz:
        qcount = len(st.session_state.quiz)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;margin:1rem 0 0.5rem;">
            <h3 style="margin:0;font-size:1.1rem;font-weight:800;color:{COLORS['primary']};">
                {st.session_state.quiz_topic}
            </h3>
            {difficulty_badge(st.session_state.quiz_difficulty)}
            <span style="background:{COLORS['primary_pale']};color:{COLORS['primary']};
                font-size:0.78rem;font-weight:700;padding:0.2rem 0.7rem;border-radius:20px;">
                {qcount} Question{'s' if qcount != 1 else ''}
            </span>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.quiz_submitted:
            with st.form("quiz_submission_form"):
                for idx, item in enumerate(st.session_state.quiz):
                    st.markdown(f"""
                    <div style="background:{COLORS['white']};border:1.5px solid {COLORS['primary_pale']};
                        border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:0.8rem;">
                        <div style="display:flex;align-items:flex-start;gap:0.5rem;margin-bottom:0.7rem;">
                            <span style="background:linear-gradient(135deg,{COLORS['primary']},{COLORS['secondary']});
                                color:{COLORS['white']};font-size:0.78rem;font-weight:800;
                                padding:0.2rem 0.6rem;border-radius:8px;white-space:nowrap;">
                                Q {idx + 1}
                            </span>
                            {difficulty_badge(item.get('difficulty', 'Medium'))}
                            <span style="background:{COLORS['secondary_pale']};color:{COLORS['secondary']};
                                font-size:0.72rem;font-weight:600;padding:0.18rem 0.55rem;border-radius:8px;">
                                {item.get('origin_label','Source')}
                            </span>
                        </div>
                        <div style="font-size:0.95rem;font-weight:600;color:{COLORS['text_dark']};
                            margin-bottom:0.8rem;line-height:1.50;">
                            {item['question']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if item["question_type"] == "MCQ":
                        st.radio("Choose one:", item["options"],
                                 key=f"quiz_answer_{idx}", index=None,
                                 label_visibility="collapsed")
                    elif item["question_type"] == "Numerical":
                        st.text_input("Numerical answer:", placeholder="e.g., 1.20",
                                      key=f"quiz_answer_{idx}", label_visibility="collapsed")
                    else:
                        st.text_area("Your explanation:", placeholder="Write your answer here…",
                                     key=f"quiz_answer_{idx}", height=80, label_visibility="collapsed")

                    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

                submit_btn = st.form_submit_button("Submit Assessment ✓", type="primary")

            if submit_btn:
                unanswered = [i for i in range(len(st.session_state.quiz))
                              if not str(st.session_state.get(f"quiz_answer_{i}", "")).strip()]
                if unanswered:
                    st.warning(f"Please answer question(s): {', '.join(str(i+1) for i in unanswered)}")
                else:
                    with st.spinner("AI grading submissions, detecting misconceptions, updating BKT model…"):
                        correct_count = 0
                        feedbacks, misconceptions = [], []
                        for idx, item in enumerate(st.session_state.quiz):
                            user_ans = st.session_state.get(f"quiz_answer_{idx}", "")
                            res = grade_student_answer(item["question"], item["answer"], user_ans,
                                                       item["question_type"], item.get("source_sentence", ""))
                            if res["is_correct"]:
                                correct_count += 1
                            else:
                                if res.get("likely_misconception") and res["likely_misconception"] != "None":
                                    misconceptions.append(f"Q{idx+1}: {res['likely_misconception']}")
                            profile_mgr.record_question_result(item.get("topic", "General"),
                                                               res["is_correct"], item["question_type"])
                            feedbacks.append({"student_answer": user_ans, "is_correct": res["is_correct"],
                                              "feedback": res["feedback"],
                                              "misconception": res.get("likely_misconception", "None")})
                        profile_mgr.record_quiz_summary(st.session_state.quiz_topic, correct_count,
                                                        len(st.session_state.quiz), misconceptions)
                        st.session_state.quiz_score = (correct_count, len(st.session_state.quiz))
                        st.session_state.quiz_feedbacks = feedbacks
                        st.session_state.quiz_misconceptions = misconceptions
                        st.session_state.quiz_submitted = True
                        st.rerun()

        # ── Results Display ────────────────────────────────────
        else:
            corr, tot = st.session_state.quiz_score
            acc_pct = int((corr / tot) * 100) if tot else 0
            new_p_l = profile_mgr.get_topic_mastery(st.session_state.quiz_topic)

            # Score banner
            score_color = COLORS["success"] if acc_pct >= 70 else (COLORS["warning"] if acc_pct >= 40 else COLORS["primary_light"])
            score_bg = COLORS["success_bg"] if acc_pct >= 70 else (COLORS["warning_bg"] if acc_pct >= 40 else COLORS["error_bg"])
            st.markdown(f"""
            <div style="background:{score_bg};border:2px solid {score_color};border-radius:18px;
                padding:1.4rem 1.8rem;margin-bottom:1.2rem;text-align:center;">
                <div style="font-size:3rem;font-weight:900;color:{score_color};">{corr}/{tot}</div>
                <div style="font-size:1.0rem;font-weight:600;color:{score_color};margin-bottom:6px;">
                    {acc_pct}% Accuracy
                </div>
                <div style="display:flex;justify-content:center;gap:0.8rem;flex-wrap:wrap;margin-top:8px;">
                    <span style="background:white;padding:0.3rem 0.9rem;border-radius:20px;
                        font-size:0.80rem;font-weight:600;color:{COLORS['text_mid']};">
                        Updated BKT Mastery: {int(new_p_l * 100)}%
                    </span>
                    <span style="background:white;padding:0.3rem 0.9rem;border-radius:20px;
                        font-size:0.80rem;font-weight:600;color:{COLORS['text_mid']};">
                        Difficulty: {st.session_state.quiz_difficulty}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"<h3 style='color:{COLORS['primary']};font-weight:800;margin-bottom:0.8rem;'>Post-Assessment Feedback</h3>",
                        unsafe_allow_html=True)

            for idx, item in enumerate(st.session_state.quiz):
                fb = st.session_state.quiz_feedbacks[idx]
                is_c = fb["is_correct"]
                result_color = COLORS["success"] if is_c else COLORS["error"]
                result_bg = COLORS["success_bg"] if is_c else COLORS["error_bg"]
                result_icon = "✅" if is_c else "❌"

                with st.expander(f"{result_icon} Q{idx+1}: {item['question'][:80]}{'…' if len(item['question'])>80 else ''}",
                                 expanded=not is_c):
                    st.markdown(f"""
                    <div style="border-left:3px solid {result_color};padding-left:0.8rem;margin-bottom:0.8rem;">
                        <div style="font-size:0.78rem;font-weight:700;color:{result_color};margin-bottom:4px;">
                            {'CORRECT' if is_c else 'NEEDS REVIEW'}
                        </div>
                        <div style="font-size:0.88rem;"><strong>Your answer:</strong> {fb['student_answer']}</div>
                        <div style="font-size:0.88rem;margin-top:3px;">
                            <strong>Expected:</strong> {item['answer']}
                        </div>
                    </div>
                    <div style="background:{COLORS['info_bg']};border-radius:10px;
                        padding:0.7rem 0.9rem;font-size:0.85rem;color:{COLORS['text_dark']};
                        border-left:3px solid {COLORS['secondary']};margin-bottom:0.5rem;">
                        💬 {fb['feedback']}
                    </div>
                    """, unsafe_allow_html=True)

                    if not is_c and fb["misconception"] != "None":
                        st.markdown(f"""
                        <div style="background:{COLORS['error_bg']};border-radius:10px;
                            padding:0.65rem 0.9rem;font-size:0.83rem;color:{COLORS['error']};
                            border-left:3px solid {COLORS['error']};">
                            ⚠️ <strong>Identified Misconception:</strong> {fb['misconception']}
                        </div>
                        """, unsafe_allow_html=True)

                    st.caption(f"📖 Source: {item.get('origin_label')} • Topic: {item.get('topic')}")
                    st.caption(f"Excerpt: \"{item.get('source_sentence', '')[:160]}…\"")

            if st.session_state.quiz_misconceptions:
                st.markdown(f"""
                <div style="background:{COLORS['warning_bg']};border:1.5px solid {COLORS['warning']};
                    border-radius:14px;padding:1rem 1.2rem;margin-top:0.8rem;">
                    <div style="font-weight:700;color:{COLORS['warning']};margin-bottom:6px;">
                        💡 Action Items for Identified Misconceptions
                    </div>
                """, unsafe_allow_html=True)
                for m in st.session_state.quiz_misconceptions:
                    st.markdown(f"• {m}")
                st.markdown("</div>", unsafe_allow_html=True)

            if st.button("🔄 Take Another Assessment", type="primary"):
                st.session_state.quiz_submitted = False
                st.session_state.quiz = []
                st.rerun()


# ════════════════════════════════════════════════════════════
# TAB 3: LEARNING PROGRESS DASHBOARD (BKT)
# ════════════════════════════════════════════════════════════

with tab_dashboard:
    st.markdown(hero_banner(
        "Learning Progress — Bayesian Knowledge Tracing",
        "Per-topic mastery estimates (P(L)) updated probabilistically after every quiz and tutoring interaction. "
        "This is an estimated probability, not a guaranteed measure of knowledge.",
        "📊"
    ), unsafe_allow_html=True)

    mastery_map = profile_mgr.get_all_mastery()
    attempts_history = profile_mgr.profile.get("quiz_attempts", [])
    total_attempts = sum(d.get("attempts", 0) for d in profile_mgr.profile.get("topic_mastery", {}).values())
    total_correct = sum(d.get("correct", 0) for d in profile_mgr.profile.get("topic_mastery", {}).values())
    avg_mastery = sum(mastery_map.values()) / max(1, len(mastery_map))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Topics", len(mastery_map))
    m2.metric("Questions Attempted", total_attempts)
    m3.metric("Overall Accuracy", f"{int((total_correct / total_attempts) * 100)}%" if total_attempts else "0%")
    m4.metric("Mean Mastery P(L)", f"{int(avg_mastery * 100)}%")

    if mastery_map:
        chart_data = [{"Topic": t,
                       "Mastery P(L) %": round(p_l * 100, 1),
                       "Status": "Mastered (>75%)" if p_l >= 0.75 else ("Developing (40-75%)" if p_l >= 0.40 else "Needs Practice (<40%)")}
                      for t, p_l in mastery_map.items()]
        df_chart = pd.DataFrame(chart_data)
        fig_mastery = px.bar(df_chart, x="Topic", y="Mastery P(L) %", color="Status",
            color_discrete_map={
                "Mastered (>75%)":      COLORS["success"],
                "Developing (40-75%)":  COLORS["secondary"],
                "Needs Practice (<40%)":COLORS["primary_light"]
            },
            range_y=[0, 100], text_auto=".1f")
        fig_mastery = plotly_theme(fig_mastery, "Bayesian Knowledge Tracing — Mastery Probability by Topic")
        st.plotly_chart(fig_mastery, use_container_width=True)

    # Topic-by-topic mastery history
    topic_mastery_raw = profile_mgr.profile.get("topic_mastery", {})
    if topic_mastery_raw:
        st.markdown(section_heading("Per-Topic Mastery Details"), unsafe_allow_html=True)
        tm_cols = st.columns(min(3, len(topic_mastery_raw)))
        for col_idx, (t, data) in enumerate(topic_mastery_raw.items()):
            p_l = data.get("p_l", DEFAULT_P_L0)
            pct = int(p_l * 100)
            att = data.get("attempts", 0)
            corr = data.get("correct", 0)
            acc = int((corr / att) * 100) if att else 0
            bar_c = COLORS["success"] if pct >= 75 else (COLORS["secondary"] if pct >= 40 else COLORS["primary_light"])
            with tm_cols[col_idx % len(tm_cols)]:
                st.markdown(f"""
                <div style="background:{COLORS['white']};border:1.5px solid {COLORS['primary_pale']};
                    border-radius:14px;padding:1rem;margin-bottom:0.8rem;">
                    <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.05em;color:{COLORS['text_muted']};margin-bottom:6px;">
                        {t[:30]}{'…' if len(t)>30 else ''}
                    </div>
                    <div style="font-size:1.6rem;font-weight:800;color:{bar_c};">{pct}%</div>
                    <div style="font-size:0.72rem;color:{COLORS['text_muted']};margin:3px 0 8px;">
                        Estimated Mastery P(L)
                    </div>
                    <div style="background:{COLORS['primary_ghost']};border-radius:4px;height:6px;margin-bottom:8px;">
                        <div style="background:{bar_c};width:{pct}%;height:100%;border-radius:4px;"></div>
                    </div>
                    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;">
                        <span style="font-size:0.72rem;color:{COLORS['text_muted']};">
                            {att} attempt{'s' if att != 1 else ''}
                        </span>
                        <span style="font-size:0.72rem;color:{COLORS['text_muted']};">·</span>
                        <span style="font-size:0.72rem;color:{COLORS['text_muted']};">{acc}% quiz accuracy</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    if attempts_history:
        st.markdown(section_heading("Assessment History"), unsafe_allow_html=True)
        hist_df = pd.DataFrame(attempts_history)
        hist_df["Date"] = hist_df["timestamp"].apply(lambda t: datetime.fromtimestamp(t).strftime("%b %d, %H:%M"))
        hist_df["Accuracy (%)"] = hist_df["accuracy"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(
            hist_df[["Date", "topic", "correct", "total", "Accuracy (%)"]].rename(
                columns={"topic": "Topic", "correct": "✅ Correct", "total": "Total"}),
            use_container_width=True, hide_index=True
        )

    with st.expander("ℹ️ How the BKT Learner Model Works"):
        st.markdown(r"""
**Bayesian Knowledge Tracing (BKT)** models the student's latent knowledge as a probability $P(L_t) \in [0,1]$.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Prior $P(L_0)$ | Calibrated at intake | Beginner: 0.15 · Intermediate: 0.40 · Advanced: 0.70 |
| Slip $P(S)$ | 0.10 | Probability of a careless error despite mastery |
| Guess $P(G)$ | 0.20 (MCQ) / 0.05 (Open) | Probability of a lucky correct answer without mastery |
| Transit $P(T)$ | 0.15 | Probability of learning during a practice opportunity |

$$P(L_t \mid \text{Correct}) = \frac{P(L_t)(1 - P(S))}{P(L_t)(1 - P(S)) + (1 - P(L_t))P(G)}$$

$$P(L_{t+1}) = P(L_t \mid \text{Obs}) + \bigl(1 - P(L_t \mid \text{Obs})\bigr) \cdot P(T)$$

> **Note:** BKT mastery is an *estimated probability*, not a scientifically certified measure of guaranteed knowledge.
""")


# ════════════════════════════════════════════════════════════
# TAB 4: COURSE FLOW, FLASHCARDS & STUDY SCHEDULE
# ════════════════════════════════════════════════════════════

with tab_flow:
    st.markdown(hero_banner(
        "Course Flow, Flashcards & Study Schedule",
        "Explore topic prerequisites, review concepts with active-recall flashcards, "
        "and follow a spaced-repetition study schedule personalised to your mastery gaps.",
        "🗺️"
    ), unsafe_allow_html=True)

    subtab_flow, subtab_cards, subtab_sched = st.tabs([
        "🗺️ Topic Prerequisite Map",
        "🗂️ Revision Flashcards",
        "📅 Study Schedule"
    ])

    with subtab_flow:
        st.markdown(section_heading("Visual Course Prerequisite Map",
            "Topics and their prerequisite relationships extracted from your uploaded material. "
            "Node colour indicates current BKT mastery estimate."), unsafe_allow_html=True)

        prereqs = st.session_state.course_meta.get("prerequisites", [])
        topics_list = st.session_state.course_meta.get("topics", [])
        concepts_map = st.session_state.course_meta.get("concepts", {})

        # Graphviz DAG with brand colours
        dot_lines = [
            "digraph CourseFlow {",
            "rankdir=LR;",
            f'graph [bgcolor="{COLORS["primary_ghost"]}", pad="0.4"];',
            f'node [shape=box, style="filled,rounded", fontname="sans-serif", fontsize=11, '
            f'fontcolor="{COLORS["text_dark"]}", color="{COLORS["secondary"]}", penwidth=1.5];',
            f'edge [color="{COLORS["secondary_mid"]}", penwidth=1.2, fontname="sans-serif", fontsize=9];'
        ]
        for t in topics_list:
            clean = re.sub(r'[^a-zA-Z0-9_ ]', '', t)
            p_l_v = profile_mgr.get_topic_mastery(t)
            fill = COLORS["success_bg"] if p_l_v >= 0.75 else (COLORS["warning_bg"] if p_l_v >= 0.40 else COLORS["error_bg"])
            dot_lines.append(f'"{clean}" [label="{t}\\n({int(p_l_v * 100)}%)", fillcolor="{fill}"];')

        for edge in prereqs:
            s_c = re.sub(r'[^a-zA-Z0-9_ ]', '', edge.get("prerequisite", ""))
            d_c = re.sub(r'[^a-zA-Z0-9_ ]', '', edge.get("topic", ""))
            if s_c and d_c:
                dot_lines.append(f'"{s_c}" -> "{d_c}" [label="requires"];')

        dot_lines.append("}")
        st.graphviz_chart("\n".join(dot_lines))

        # Topic-concept grid
        if concepts_map:
            st.markdown(section_heading("Key Concepts by Topic"), unsafe_allow_html=True)
            concept_cols = st.columns(min(3, len(concepts_map)))
            for ci, (topic, concepts) in enumerate(concepts_map.items()):
                if concepts:
                    with concept_cols[ci % len(concept_cols)]:
                        concepts_html = "".join(
                            f'<span style="background:{COLORS["secondary_pale"]};color:{COLORS["secondary"]};'
                            f'font-size:0.75rem;font-weight:600;padding:0.18rem 0.6rem;'
                            f'border-radius:20px;display:inline-block;margin:2px;">{c}</span>'
                            for c in concepts[:6]
                        )
                        st.markdown(card_html(topic, concepts_html, "🏷️", COLORS["secondary_pale"]),
                                    unsafe_allow_html=True)

    with subtab_cards:
        st.markdown(section_heading("Active-Recall Flashcards",
            f"Cards drawn from your weakest topic: {weakest_topic}"), unsafe_allow_html=True)

        weak_sentences = [s for s in st.session_state.sentences
                          if s.get("topic") == weakest_topic]
        if not weak_sentences:
            weak_sentences = st.session_state.sentences[:10]

        if weak_sentences:
            fc_idx = st.session_state.active_flashcard_idx % len(weak_sentences)
            card_item = weak_sentences[fc_idx]
            words = card_item["text"].split()
            masked = " ".join(words[:max(3, len(words) // 2)]) + " …"

            st.markdown(f"""
            <div style="text-align:right;margin-bottom:0.5rem;">
                <span style="font-size:0.78rem;color:{COLORS['text_muted']};">
                    Card {fc_idx + 1} of {len(weak_sentences)}
                </span>
            </div>
            <div style="background:linear-gradient(135deg,{COLORS['primary']},{COLORS['secondary']});
                border-radius:20px;padding:2.2rem 2.4rem;text-align:center;
                box-shadow:0 8px 28px rgba(50,0,75,0.22);margin-bottom:1rem;position:relative;">
                <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.10em;color:rgba(255,254,249,0.65);margin-bottom:1rem;">
                    ❓ Concept Prompt  ·  {card_item.get('topic','')}
                </div>
                <div style="font-size:1.05rem;font-weight:600;color:{COLORS['white']};
                    line-height:1.55;margin-bottom:0.6rem;">
                    What key principle is described here?
                </div>
                <div style="font-size:0.90rem;color:rgba(255,254,249,0.80);
                    font-style:italic;line-height:1.5;">
                    "{masked}"
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.show_flashcard_answer:
                st.markdown(f"""
                <div style="background:{COLORS['success_bg']};border:2px solid {COLORS['success']};
                    border-radius:16px;padding:1.3rem 1.5rem;margin-bottom:1rem;">
                    <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.06em;color:{COLORS['success']};margin-bottom:6px;">
                        ✅ Full Concept
                    </div>
                    <div style="font-size:0.95rem;color:{COLORS['text_dark']};line-height:1.55;">
                        {card_item['text']}
                    </div>
                    <div style="font-size:0.75rem;color:{COLORS['text_muted']};margin-top:8px;">
                        📍 {card_item.get('origin_label', 'Source')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            fc1, fc2, fc3 = st.columns(3)
            if fc1.button("👁️ Flip Card", use_container_width=True):
                st.session_state.show_flashcard_answer = not st.session_state.show_flashcard_answer
                st.rerun()
            if fc2.button("⬅️ Previous", use_container_width=True):
                st.session_state.active_flashcard_idx = max(0, st.session_state.active_flashcard_idx - 1)
                st.session_state.show_flashcard_answer = False
                st.rerun()
            if fc3.button("➡️ Next", use_container_width=True):
                st.session_state.active_flashcard_idx += 1
                st.session_state.show_flashcard_answer = False
                st.rerun()

            # Progress dots
            total_cards = len(weak_sentences)
            dots_per_row = min(total_cards, 20)
            dots_html = "".join(
                f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                f'background:{"" + COLORS["secondary"] if i == fc_idx else COLORS["primary_pale"]};'
                f'margin:0 3px;"></span>'
                for i in range(dots_per_row)
            )
            st.markdown(f'<div style="text-align:center;padding:0.5rem;">{dots_html}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(card_html("No Flashcards Available",
                "Upload course material to generate revision flashcards.", "🗂️"), unsafe_allow_html=True)

    with subtab_sched:
        st.markdown(section_heading("Spaced-Repetition Study Schedule",
            "Generated using Ebbinghaus forgetting curve intervals and BKT mastery gaps."),
            unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            target_exam_days = st.slider("Days until exam:", min_value=3, max_value=30, value=14)
        with sc2:
            daily_hours = st.slider("Daily study time (hours):", min_value=1, max_value=6, value=2)

        today = datetime.now()
        sorted_topics_sched = sorted(available_topics, key=lambda t: profile_mgr.get_topic_mastery(t))
        schedule_rows = []
        for day in range(1, target_exam_days + 1):
            tdate = today + timedelta(days=day)
            t_assigned = sorted_topics_sched[(day - 1) % len(sorted_topics_sched)]
            mastery_v = profile_mgr.get_topic_mastery(t_assigned)
            if mastery_v < 0.40:
                activity = "📖 Foundational Review + Tutor Deep Dive"
            elif mastery_v < 0.75:
                activity = "📝 Targeted Adaptive Quiz Practice"
            else:
                activity = "⚡ Spaced Recall & Speed Drill"

            schedule_rows.append({
                "Day": f"Day {day}", "Date": tdate.strftime("%a, %b %d"),
                "Priority Topic": t_assigned[:40],
                "Mastery": f"{int(mastery_v * 100)}%",
                "Recommended Activity": activity,
                "Hours": f"{daily_hours}h"
            })

        df_sched = pd.DataFrame(schedule_rows)
        st.dataframe(df_sched, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# TAB 5: SYSTEM EVALUATION & BENCHMARKING
# ════════════════════════════════════════════════════════════

with tab_eval:
    st.markdown(hero_banner(
        "System Evaluation & Benchmarking",
        "Standardised RAG metrics (Faithfulness, Answer Relevancy, Context Precision, Recall, Refusal Honesty) "
        "and multi-session simulated student personalisation evaluation.",
        "🧪"
    ), unsafe_allow_html=True)

    eval_tab_rag, eval_tab_sim = st.tabs(["📊 RAG Benchmark Metrics", "👥 Student Personalisation Simulation"])

    with eval_tab_rag:
        st.markdown(f"""
        <div style="background:{COLORS['primary_ghost']};border:1.5px solid {COLORS['primary_pale']};
            border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:1.2rem;font-size:0.88rem;
            color:{COLORS['text_mid']};line-height:1.60;">
            Evaluates the retrieval and generation pipeline on a <strong>team-built test set</strong> containing:<br/>
            <strong>10 Source-Grounded Queries</strong> — verifying citation accuracy, precision, and recall.<br/>
            <strong>5 Adversarial Off-Material Queries</strong> — verifying strict refusal of unsupported claims.<br/>
            <em>Note: Benchmark does not run automatically on every page load to avoid unnecessary computation.</em>
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶️ Run RAG Benchmark Evaluation", type="primary"):
            with st.spinner("Executing benchmark queries through the retrieval & grounding pipeline…"):
                def pipeline_runner(q):
                    return answer_question_grounded(q, topic="All topics", language="English")
                st.session_state.benchmark_results = run_system_benchmark(pipeline_runner)

        if st.session_state.benchmark_results:
            b_res = st.session_state.benchmark_results
            summary = b_res["summary"]

            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Faithfulness", f"{summary['avg_faithfulness_pct']}%")
            b2.metric("Answer Relevancy", f"{summary['avg_relevancy_pct']}%")
            b3.metric("Context Precision", f"{summary['avg_context_precision_pct']}%")
            b4.metric("Context Recall", f"{summary['avg_context_recall_pct']}%")
            b5.metric("Refusal Accuracy", f"{summary['refusal_accuracy_pct']}%")

            # Radar chart
            cats = ["Faithfulness", "Relevancy", "Precision", "Recall", "Refusal"]
            vals = [summary["avg_faithfulness_pct"], summary["avg_relevancy_pct"],
                    summary["avg_context_precision_pct"], summary["avg_context_recall_pct"],
                    summary["refusal_accuracy_pct"]]
            fig_radar = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor=f"rgba(120,75,121,0.18)",
                line=dict(color=COLORS["secondary"], width=2.5),
                marker=dict(color=COLORS["primary"], size=7)
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor=COLORS["white"],
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9),
                                   gridcolor=COLORS["primary_pale"]),
                    angularaxis=dict(tickfont=dict(size=11, color=COLORS["text_dark"]),
                                     gridcolor=COLORS["primary_pale"])
                ),
                paper_bgcolor=COLORS["white"],
                font=dict(family="Inter"),
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                height=300
            )
            col_r1, col_r2 = st.columns([1, 1])
            with col_r1:
                st.plotly_chart(fig_radar, use_container_width=True)
            with col_r2:
                st.markdown(section_heading("Benchmark Interpretation"), unsafe_allow_html=True)
                st.markdown(f"""
                <div style="font-size:0.84rem;color:{COLORS['text_mid']};line-height:1.65;">
                    <strong>Faithfulness:</strong> Proportion of claims in generated answers that are grounded in retrieved context.<br/>
                    <strong>Answer Relevancy:</strong> Semantic alignment between the query and the response.<br/>
                    <strong>Context Precision:</strong> Signal-to-noise ratio of passage ranking.<br/>
                    <strong>Context Recall:</strong> Coverage of ground-truth concepts in retrieved passages.<br/>
                    <strong>Refusal Accuracy:</strong> Proportion of off-material queries correctly declined.
                </div>
                """, unsafe_allow_html=True)

            st.markdown(section_heading("Itemised Query Scorecard"), unsafe_allow_html=True)
            item_df = pd.DataFrame(b_res["itemized"])
            st.dataframe(
                item_df.rename(columns={"query": "Query", "type": "Type", "status": "Outcome",
                                        "faithfulness": "Faith.", "relevancy": "Relev.",
                                        "precision": "Prec.", "recall": "Recall", "passed": "Passed"}),
                use_container_width=True, hide_index=True
            )

    with eval_tab_sim:
        st.markdown(f"""
        <div style="background:{COLORS['primary_ghost']};border:1.5px solid {COLORS['primary_pale']};
            border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:1.2rem;font-size:0.88rem;
            color:{COLORS['text_mid']};line-height:1.60;">
            Evaluates <strong>personalisation effectiveness</strong> using 3 simulated student personas
            across 5 study sessions — reporting mastery gains and question repetition rate.<br/>
            <strong>Struggling Learner</strong> · <strong>Steady Learner</strong> · <strong>Advanced Learner</strong>
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶️ Run Multi-Session Student Simulation", type="primary"):
            with st.spinner("Simulating multi-session adaptive study trajectories…"):
                sim_res = run_simulated_student_evaluation(available_topics, sessions_count=5, questions_per_session=4)
                st.session_state.simulation_results = sim_res

        if st.session_state.simulation_results:
            sim = st.session_state.simulation_results

            g1, g2, g3 = st.columns(3)
            g1.metric("Simulated Sessions", "5 per student")
            g2.metric("Total Simulated Questions", sim["total_simulated_questions"])
            g3.metric("Question Repetition Rate", f"{sim['question_repetition_rate_pct']}%",
                      "0% = Fully Novel" if sim["question_repetition_rate_pct"] == 0 else None)

            st.markdown(section_heading("Mastery Gains Over 5 Sessions"), unsafe_allow_html=True)
            gain_cols = st.columns(len(sim["gains"]))
            for idx_g, (p_name, g_val) in enumerate(sim["gains"].items()):
                gain_cols[idx_g].metric(p_name, f"+{g_val}%", f"gain over 5 sessions")

            sim_df = pd.DataFrame(sim["curves"])
            fig_sim = px.line(sim_df, x="session", y="average_mastery_pct",
                color="persona", markers=True,
                labels={"session": "Study Session",
                        "average_mastery_pct": "Average Topic Mastery P(L) (%)",
                        "persona": "Student Persona"},
                color_discrete_sequence=[COLORS["primary_light"], COLORS["secondary"], COLORS["success"]]
            )
            fig_sim = plotly_theme(fig_sim, "Adaptive Mastery Progression — BKT P(L) Across 5 Sessions")
            fig_sim.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_sim, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(f"""
<div style="text-align:center;padding:1.5rem 0 2rem;margin-top:1rem;">
    <div style="background:linear-gradient(135deg,{COLORS['primary']},{COLORS['secondary']});
        display:inline-block;border-radius:50px;padding:0.5rem 2rem;">
        <span style="color:{COLORS['white']};font-size:0.78rem;font-weight:600;">
            🔮 LearnWise AI  ·  Multimodal AI Hackathon 2026  ·  Track D  ·  Powered by Ollama llama3.2 + Bayesian Knowledge Tracing
        </span>
    </div>
</div>
""", unsafe_allow_html=True)