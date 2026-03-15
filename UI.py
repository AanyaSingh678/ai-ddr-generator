"""
ui.py

UrbanRoof DDR Generator — Streamlit UI.

Design: matches reference screenshot — terminal status bar, centered hero,
        sidebar with upload + form fields, bottom footer bar.
Fonts:  DM Serif Display (headings/wordmark) + DM Sans (body/labels/buttons).
Logic:  real AI pipeline via Groq/OpenAI, no mocked data.

Usage:
    streamlit run ui.py
"""

import logging
import time
from datetime import date, datetime
from pathlib import Path

import streamlit as st

import config
from app import (
    step_build_empty_extraction_result,
    step_extract_images,
    step_extract_observations,
    step_parse_document,
)
from parser.image_extractor import ExtractionResult
from reasoning.conflict_detector import ConflictDetector
from reasoning.observation_image_mapper import ObservationImageMapper
from reasoning.severity_assessor import SeverityAssessor
from report.ddr_generator import DDRGenerator


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="UrbanRoof DDR Generator",
    page_icon="U",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# CSS — matches reference screenshot exactly, DM fonts preserved
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=DM+Serif+Display:ital@0;1&display=swap');

    :root {
        --bg:           #0d0d0f;
        --bg-2:         #111114;
        --card:         #161618;
        --card-2:       #1c1c1f;
        --border:       #242428;
        --border-2:     #2e2e33;
        --fg:           #e8e8ea;
        --fg-2:         #9a9a9f;
        --fg-3:         #55555a;
        --amber:        #c8923a;
        --amber-bright: #e0a84a;
        --amber-glow:   rgba(200,146,58,0.14);
        --amber-border: rgba(200,146,58,0.35);
        --success:      #3dba6e;
        --error:        #e05555;
        --info:         #4a9edd;
        --mono:         'DM Sans', 'SF Mono', 'Fira Code', monospace;
        --serif:        'DM Serif Display', Georgia, serif;
        --sans:         'DM Sans', -apple-system, sans-serif;
    }

    /* ── Base ── */
    .stApp { background: var(--bg) !important; }
    html, body, [class*="css"] {
        font-family: var(--sans) !important;
        color: var(--fg) !important;
    }
    #MainMenu, footer, header,
    div[data-testid="stDecoration"],
    [data-testid="stToolbar"] { display: none !important; visibility: hidden !important; }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 3px; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-2) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

    /* ── Main block ── */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    /* Remove Streamlit top padding from main area */
    .main > div:first-child { padding-top: 0 !important; }
    div[data-testid="stAppViewContainer"] > section.main { padding-top: 0 !important; }
    div[data-testid="block-container"] { padding-top: 0 !important; gap: 0 !important; }
    /* Remove auto-anchor link icons on headings */
    h1 a, h2 a, h3 a { display: none !important; }
    .hero-title a { display: none !important; }
    /* Force subtitle centering regardless of Streamlit wrapper */
    .hero-wrap p, .hero-wrap .hero-sub { text-align: center !important; }
    div[data-testid="stMarkdownContainer"] p { text-align: inherit; }
    .hero-wrap div[data-testid="stMarkdownContainer"] p { text-align: center !important; }

    /* ── Top status bar ── */
    .status-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 32px;
        background: var(--bg);
        border-bottom: 1px solid var(--border);
        font-family: var(--mono);
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--fg-2);
        position: sticky; top: 0; z-index: 100;
    }
    .status-dot {
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        background: var(--success);
        margin-right: 8px;
        box-shadow: 0 0 6px var(--success);
        animation: pulse-dot 2.5s ease-in-out infinite;
    }
    .status-dot.running { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
    .status-dot.idle    { background: var(--fg-3);  box-shadow: none; animation: none; }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.35; }
    }

    /* ── Content wrapper ── */
    .content-wrap {
        padding: 0 48px 100px;
        max-width: 1280px;
        margin: 0 auto;
    }
    /* Kill any Streamlit-injected top margin on first child */
    .content-wrap > div:first-child,
    .content-wrap > div:first-child > div:first-child { margin-top: 0 !important; padding-top: 0 !important; }

    /* ── Hero ── */
    .hero-wrap { text-align: center; padding: 36px 0 24px; }
    .hero-pill {
        display: inline-flex; align-items: center; gap: 7px;
        background: var(--amber-glow);
        border: 1px solid var(--amber-border);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--amber-bright);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 18px;
    }
    .hero-pill-dot {
        width: 5px; height: 5px;
        border-radius: 50%;
        background: var(--amber-bright);
    }
    .hero-title {
        font-family: var(--serif) !important;
        font-size: 3.5rem !important;
        font-weight: 400 !important;
        color: var(--fg) !important;
        letter-spacing: -0.03em;
        line-height: 1.06;
        margin-bottom: 18px !important;
    }
    .hero-sub {
        font-size: 0.975rem;
        color: var(--fg-2);
        max-width: 560px;
        margin: 0 auto 28px !important;
        line-height: 1.7;
        font-weight: 300;
        text-align: center !important;
        display: block;
    }

    /* ── Upload placeholder card (idle, no file) ── */
    .upload-placeholder {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 48px 32px;
        text-align: center;
        width: 100%;
        max-width: 720px;
        margin: 0 auto;
    }
    .upload-placeholder-icon {
        width: 60px; height: 60px;
        background: var(--card-2);
        border: 1px solid var(--border-2);
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 18px;
        font-size: 1.5rem; color: var(--fg-3);
    }
    .upload-placeholder-text {
        font-size: 0.9rem; color: var(--fg-2);
        font-weight: 400; line-height: 1.5;
    }

    /* ── Sidebar brand block ── */
    .sidebar-brand {
        display: flex; align-items: center; gap: 10px;
        padding: 20px 18px 16px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 4px;
    }
    .sidebar-logo {
        width: 34px; height: 34px;
        background: var(--amber);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-family: var(--serif);
        font-size: 17px; color: var(--bg-2); font-weight: bold;
        flex-shrink: 0;
    }
    .sidebar-brand-name {
        font-family: var(--serif) !important;
        font-size: 1.1rem !important;
        color: var(--fg) !important;
        letter-spacing: -0.01em; line-height: 1;
    }
    .sidebar-brand-sub {
        font-size: 0.58rem;
        color: var(--fg-3);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-top: 2px;
    }

    /* ── Sidebar section label ── */
    .sidebar-label {
        font-size: 0.62rem !important;
        font-weight: 600 !important;
        color: var(--fg-3) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.14em !important;
        padding: 10px 18px 6px;
        display: block;
    }
    /* Tighten Streamlit default gaps inside sidebar */
    section[data-testid="stSidebar"] .stVerticalBlock { gap: 0.3rem !important; }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: 0.3rem !important; }
    section[data-testid="stSidebar"] .element-container { margin-bottom: 0 !important; }
    section[data-testid="stSidebar"] hr { margin: 8px 0 !important; border-color: var(--border) !important; }
    /* Tiny breathing room between each sidebar section block */
    section[data-testid="stSidebar"] .stVerticalBlock > div { padding-bottom: 2px; }

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    div[data-testid="stDateInput"] input {
        font-family: var(--sans) !important;
        background: var(--card) !important;
        border: 1px solid var(--border-2) !important;
        border-radius: 7px !important;
        color: var(--fg) !important;
        font-size: 0.875rem !important;
        padding: 9px 12px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--amber) !important;
        box-shadow: 0 0 0 2px var(--amber-glow) !important;
        outline: none !important;
    }
    .stTextInput > label, .stSelectbox > label,
    .stDateInput > label, .stFileUploader > label {
        font-family: var(--sans) !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: var(--fg) !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 4px !important;
    }
    /* Selectbox dropdown popup — z-index fix for glitch */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div {
        z-index: 99999 !important;
        background: var(--card-2) !important;
        border: 1px solid var(--border-2) !important;
        border-radius: 8px !important;
        overflow: visible !important;
    }
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] > ul {
        background: var(--card-2) !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }
    div[role="listbox"] {
        background: var(--card-2) !important;
        border: 1px solid var(--border-2) !important;
        border-radius: 8px !important;
        z-index: 99999 !important;
    }
    div[role="option"],
    li[role="option"] {
        background: var(--card-2) !important;
        color: var(--fg) !important;
        font-family: var(--sans) !important;
        font-size: 0.875rem !important;
        padding: 8px 12px !important;
        border-radius: 5px !important;
        cursor: pointer !important;
    }
    div[role="option"]:hover,
    li[role="option"]:hover,
    div[aria-selected="true"],
    li[aria-selected="true"] {
        background: var(--border-2) !important;
        color: var(--fg) !important;
    }
    /* Selected option text — force visible and centered */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stSelectbox"] > div > div > div,
    [data-testid="stSelectbox"] span,
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] span {
        color: var(--fg) !important;
        line-height: normal !important;
        display: flex !important;
        align-items: center !important;
    }
    /* Dropdown arrow */
    [data-testid="stSelectbox"] svg { fill: var(--fg-2) !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: var(--card) !important;
        border: 1px dashed var(--border-2) !important;
        border-radius: 10px !important;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover { border-color: var(--amber) !important; }
    [data-testid="stFileUploader"] section { background: transparent !important; }
    [data-testid="stFileUploader"] button {
        background: transparent !important;
        border: 1px solid var(--border-2) !important;
        color: var(--fg-2) !important;
        font-size: 0.75rem !important;
        border-radius: 6px !important;
    }

    /* ── File pill ── */
    .file-pill {
        display: flex; align-items: center; gap: 10px;
        padding: 8px 12px;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px; margin-top: 6px;
    }
    .file-pill-icon {
        width: 28px; height: 28px;
        background: var(--amber-glow);
        border-radius: 5px;
        display: flex; align-items: center; justify-content: center;
        color: var(--amber); font-size: 0.75rem; flex-shrink: 0;
    }
    .file-pill-name {
        flex: 1; font-size: 0.78rem; color: var(--fg);
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .file-pill-size { font-size: 0.68rem; color: var(--fg-3); }

    /* ── Buttons ── */
    .stButton > button {
        font-family: var(--sans) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        transition: all 0.18s ease !important;
        width: 100%;
    }
    .stButton > button[kind="primary"] {
        background: var(--amber) !important;
        color: var(--bg-2) !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--amber-bright) !important;
        box-shadow: 0 0 16px var(--amber-glow) !important;
    }
    .stButton > button:not([kind="primary"]) {
        background: var(--card-2) !important;
        color: var(--fg-2) !important;
        border: 1px solid var(--border-2) !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: var(--amber) !important;
        color: var(--amber) !important;
    }
    .stButton > button:disabled {
        background: rgba(200, 146, 58, 0.35) !important;
        color: rgba(200, 146, 58, 0.7) !important;
        border: 1px solid rgba(200, 146, 58, 0.2) !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
    }
    .stDownloadButton > button {
        font-family: var(--sans) !important;
        font-weight: 600 !important;
        background: var(--amber) !important;
        color: var(--bg-2) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        background: var(--amber-bright) !important;
        box-shadow: 0 0 16px var(--amber-glow) !important;
    }
    .sidebar-helper {
        font-size: 0.7rem;
        color: var(--fg-3);
        padding: 6px 18px 0;
        line-height: 1.5;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div { background: var(--amber) !important; border-radius: 2px !important; }
    .stProgress > div > div { background: var(--border) !important; border-radius: 2px !important; height: 3px !important; }

    /* ── Pipeline steps ── */
    .pipeline-section-label {
        font-size: 0.62rem;
        font-weight: 600;
        color: var(--fg-3);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 14px;
    }
    .pipeline-step {
        display: flex; align-items: flex-start; gap: 14px;
        padding: 13px 16px;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 9px; margin-bottom: 7px;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .pipeline-step.active  { border-color: var(--amber); box-shadow: 0 0 14px var(--amber-glow); }
    .pipeline-step.done    { border-color: rgba(61,186,110,0.3); background: rgba(61,186,110,0.04); }
    .pipeline-step.fail    { border-color: rgba(224,85,85,0.4);  background: rgba(224,85,85,0.04); }
    .step-num {
        width: 28px; height: 28px; border-radius: 7px;
        display: flex; align-items: center; justify-content: center;
        font-family: var(--serif); font-size: 0.82rem; flex-shrink: 0;
    }
    .step-num.pending { background: var(--border);              color: var(--fg-3); }
    .step-num.active  { background: var(--amber);               color: var(--bg-2); }
    .step-num.done    { background: rgba(61,186,110,0.2);       color: var(--success); }
    .step-num.fail    { background: rgba(224,85,85,0.2);        color: var(--error); }
    .step-body { flex: 1; min-width: 0; }
    .step-title { font-size: 0.875rem; font-weight: 500; color: var(--fg); margin-bottom: 3px; }
    .step-desc  { font-size: 0.75rem;  color: var(--fg-2); line-height: 1.45; }
    .step-tag {
        font-size: 0.65rem; padding: 3px 9px; border-radius: 4px;
        white-space: nowrap; align-self: flex-start; margin-top: 3px;
        font-weight: 600;
    }
    .step-tag.amber  { color: var(--amber);   background: var(--amber-glow); }
    .step-tag.green  { color: var(--success); background: rgba(61,186,110,0.12); }
    .step-tag.red    { color: var(--error);   background: rgba(224,85,85,0.12); }

    /* ── Results ── */
    .results-header {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 24px; padding-bottom: 16px;
        border-bottom: 1px solid var(--border);
    }
    .results-title {
        font-family: var(--serif) !important;
        font-size: 1.65rem !important;
        color: var(--fg) !important;
        letter-spacing: -0.02em;
    }
    .results-badge {
        font-size: 0.62rem; font-weight: 600;
        color: var(--success); background: rgba(61,186,110,0.12);
        padding: 5px 12px; border-radius: 20px;
        text-transform: uppercase; letter-spacing: 0.1em;
    }
    .metric-card {
        background: var(--card); border: 1px solid var(--border);
        border-radius: 10px; padding: 20px 16px; text-align: center;
    }
    .metric-val {
        font-family: var(--serif) !important;
        font-size: 2.4rem !important; color: var(--amber) !important;
        line-height: 1 !important; letter-spacing: -0.03em;
        margin-bottom: 6px !important;
    }
    .metric-lbl {
        font-size: 0.65rem; color: var(--fg-3);
        text-transform: uppercase; letter-spacing: 0.1em;
    }
    .sev-wrap { margin-bottom: 14px; }
    .sev-top {
        display: flex; justify-content: space-between;
        font-size: 0.82rem; margin-bottom: 6px;
        text-transform: capitalize;
    }
    .sev-top-name { color: var(--fg); }
    .sev-top-count { color: var(--fg-2); }
    .sev-bar { height: 6px; border-radius: 3px; background: var(--border); overflow: hidden; }
    .sev-fill { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
    .sev-critical { background: #e05555; }
    .sev-high     { background: #e07a35; }
    .sev-medium   { background: var(--amber); }
    .sev-low      { background: var(--success); }
    .timeline-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 0; border-bottom: 1px solid var(--border);
        font-size: 0.78rem;
    }
    .timeline-row:last-child { border-bottom: none; }
    .timeline-name { color: var(--fg-2); }
    .timeline-time { color: var(--amber); font-weight: 500; }

    /* ── Bottom footer bar ── */
    .footer-bar {
        position: fixed; bottom: 0; left: 0; right: 0; z-index: 200;
        display: flex; align-items: center; justify-content: space-between;
        padding: 9px 32px;
        background: var(--bg-2);
        border-top: 1px solid var(--border);
        font-family: var(--mono);
        font-size: 0.62rem;
        letter-spacing: 0.07em;
        color: var(--fg-3);
    }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid var(--border); margin: 14px 0; }

    /* ── Streamlit alert overrides ── */
    [data-testid="stAlert"] {
        background: var(--card) !important;
        border-radius: 7px !important;
        font-size: 0.825rem !important;
        border: 1px solid var(--border-2) !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict = {
        "pipeline_running":  False,
        "pipeline_complete": False,
        "step_times":        {},
        "report":            None,
        "step_results":      {},
        "severity_summary":  {},
        "user_config":       None,
        "status_text":       "SYSTEM READY",
        "status_state":      "ready",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------------------------
# Pipeline step definitions
# ---------------------------------------------------------------------------

PIPELINE_STEPS: list[tuple[str, str]] = [
    ("Document Parsing",          "Parsing PDF structure and extracting raw text content"),
    ("Image Extraction",          "Extracting and deduplicating embedded images from both PDFs"),
    ("AI Observation Extraction", "Identifying key findings using LLM analysis"),
    ("Image–Observation Mapping", "Linking extracted images to corresponding observations"),
    ("Conflict Detection",        "Cross-referencing inspection and thermal findings"),
    ("Severity Assessment",       "Classifying issues by urgency and impact level"),
    ("Report Generation",         "Compiling structured DDR HTML document"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_uploaded_file(uploaded_file, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        f.write(uploaded_file.getbuffer())


def fmt_size(n: int) -> str:
    if n < 1024:       return f"{n} B"
    if n < 1024**2:    return f"{n/1024:.1f} KB"
    return f"{n/(1024**2):.1f} MB"


def file_pill(name: str, size: int) -> str:
    return (
        f'<div class="file-pill">'
        f'<div class="file-pill-icon">PDF</div>'
        f'<span class="file-pill-name">{name}</span>'
        f'<span class="file-pill-size">{fmt_size(size)}</span>'
        f'</div>'
    )


def render_steps(
    completed: list[tuple[str, str]],
    active_idx: int | None = None,
    error_idx:  int | None = None,
    placeholder=None,
) -> None:
    html = ""
    for i, (name, default_desc) in enumerate(PIPELINE_STEPS):
        if error_idx is not None and i == error_idx:
            css, num_css = "pipeline-step fail", "step-num fail"
            num_content  = "✕"
            tag          = '<span class="step-tag red">Failed</span>'
            desc         = completed[i][1] if i < len(completed) else default_desc
        elif i < len(completed):
            css, num_css = "pipeline-step done", "step-num done"
            num_content  = "✓"
            tag          = f'<span class="step-tag green">{completed[i][0]}</span>'
            desc         = completed[i][1]
        elif active_idx is not None and i == active_idx:
            css, num_css = "pipeline-step active", "step-num active"
            num_content  = str(i + 1)
            tag          = '<span class="step-tag amber">Processing...</span>'
            desc         = default_desc
        else:
            css, num_css = "pipeline-step", "step-num pending"
            num_content  = str(i + 1)
            tag, desc    = "", default_desc

        html += (
            f'<div class="{css}">'
            f'<div class="{num_css}">{num_content}</div>'
            f'<div class="step-body">'
            f'<div class="step-title">{name}</div>'
            f'<div class="step-desc">{desc}</div>'
            f'</div>{tag}</div>'
        )

    target = placeholder if placeholder else st
    target.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Status bar + footer bar
# ---------------------------------------------------------------------------

def status_bar(text: str, state: str = "ready") -> None:
    """Renders the top status bar. state: ready | running | idle"""
    dot_cls = {"ready": "", "running": " running", "idle": " idle"}.get(state, "")
    now = datetime.now().strftime("%a, %b %d, %Y")
    st.markdown(
        f'<div class="status-bar">'
        f'<span><span class="status-dot{dot_cls}"></span>{text}</span>'
        f'<span>{now}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def footer_bar() -> None:
    st.markdown(
        '<div class="footer-bar">'
        '<span>UrbanRoof DDR Generator &nbsp;v1.0.0</span>'
        '<span>Powered by advanced AI analysis</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    with st.sidebar:
        # Brand
        st.markdown(
            '<div class="sidebar-brand">'
            '<div class="sidebar-logo">U</div>'
            '<div><div class="sidebar-brand-name">UrbanRoof</div>'
            '<div class="sidebar-brand-sub">DDR Generator</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Upload
        st.markdown('<span class="sidebar-label">Document Upload</span>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding:0 14px">', unsafe_allow_html=True)
            inspection_pdf = st.file_uploader(
                "Inspection Report",
                type=["pdf"], key="insp_up",
                label_visibility="collapsed",
            )
            if inspection_pdf:
                st.markdown(file_pill(inspection_pdf.name, inspection_pdf.size),
                            unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            thermal_pdf = st.file_uploader(
                "Thermal Report (optional)",
                type=["pdf"], key="therm_up",
                label_visibility="collapsed",
            )
            if thermal_pdf:
                st.markdown(file_pill(thermal_pdf.name, thermal_pdf.size),
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Inspection details
        st.markdown('<span class="sidebar-label">Inspection Details</span>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding:0 14px">', unsafe_allow_html=True)
            property_address = st.text_input(
                "Property Address",
                value=config.PROPERTY_ADDRESS,
                placeholder="123 Main Street, City, State",
            )
            inspector_name = st.text_input(
                "Inspector Name",
                value=config.INSPECTED_BY,
                placeholder="John Smith",
            )
            inspection_date = st.date_input(
                "Inspection Date",
                value=date.today(),
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # AI config
        st.markdown('<span class="sidebar-label">AI Configuration</span>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding:0 14px">', unsafe_allow_html=True)
            ai_model = st.selectbox(
                "Analysis Model",
                options=[
                    "llama-3.1-8b-instant",
                    "llama-3.3-70b-versatile",
                    "gpt-4o-mini",
                    "gpt-4o",
                ],
                index=0,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Generate button
        can_run = (
            inspection_pdf is not None
            and bool(property_address.strip())
            and bool(inspector_name.strip())
            and not st.session_state.pipeline_running
        )

        with st.container():
            st.markdown('<div style="padding:0 14px">', unsafe_allow_html=True)
            if st.button(
                "Generate Report",
                type="primary",
                disabled=not can_run,
                use_container_width=True,
                key="gen_btn",
            ):
                st.session_state.pipeline_running  = True
                st.session_state.pipeline_complete = False
                st.session_state.step_times        = {}
                st.session_state.report            = None
                st.session_state.step_results      = {}
                st.session_state.severity_summary  = {}
                st.session_state.status_text       = "PIPELINE RUNNING"
                st.session_state.status_state      = "running"
                st.session_state.user_config = {
                    "inspection_pdf":  inspection_pdf,
                    "thermal_pdf":     thermal_pdf,
                    "property_address": property_address,
                    "inspected_by":    inspector_name,
                    "inspection_date": str(inspection_date),
                    "ai_model":        ai_model,
                }
                st.rerun()

            st.markdown(
                '<div class="sidebar-helper">'
                + ("Upload PDFs and fill all fields to continue."
                   if not can_run else
                   "Ready. Click Generate Report to begin.")
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    return {
        "inspection_pdf":   inspection_pdf,
        "thermal_pdf":      thermal_pdf,
        "property_address": property_address,
        "inspected_by":     inspector_name,
        "inspection_date":  str(inspection_date),
        "ai_model":         ai_model,
    }


# ---------------------------------------------------------------------------
# Real pipeline
# ---------------------------------------------------------------------------

def run_pipeline(user_config: dict) -> None:
    progress_bar     = st.progress(0)
    steps_placeholder = st.empty()
    TOTAL = len(PIPELINE_STEPS)
    completed: list[tuple[str, str]] = []
    step_times: dict[int, float]     = {}
    thermal_images: ExtractionResult | None = None

    def done(idx: int, badge: str, detail: str, elapsed: float) -> None:
        completed.append((badge, detail))
        step_times[idx] = elapsed
        st.session_state.step_times = dict(step_times)
        progress_bar.progress(
            int(((idx + 1) / TOTAL) * 100),
            text=f"Step {idx+1} of {TOTAL} complete",
        )
        next_idx = idx + 1 if idx + 1 < TOTAL else None
        render_steps(completed, active_idx=next_idx, placeholder=steps_placeholder)

    def fail(idx: int, detail: str) -> None:
        completed.append(("Failed", detail))
        render_steps(completed, error_idx=idx, placeholder=steps_placeholder)
        st.error(detail)

    # Initial render
    render_steps([], active_idx=0, placeholder=steps_placeholder)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    inspection_path = config.DATA_DIR / "inspection_report.pdf"
    save_uploaded_file(user_config["inspection_pdf"], inspection_path)
    thermal_path: Path | None = None
    if user_config["thermal_pdf"] is not None:
        thermal_path = config.DATA_DIR / "thermal_report.pdf"
        save_uploaded_file(user_config["thermal_pdf"], thermal_path)

    # ── 1: Parse ──
    t0 = time.perf_counter()
    try:
        inspection_doc = step_parse_document(inspection_path)
        elapsed = time.perf_counter() - t0
        done(0, f"{elapsed:.1f}s",
             f"Inspection: {inspection_doc.total_pages} pages "
             f"({len(inspection_doc.non_empty_pages)} with content)",
             elapsed)
    except (FileNotFoundError, RuntimeError) as e:
        fail(0, str(e)); return

    thermal_doc = None
    if thermal_path:
        try:
            thermal_doc = step_parse_document(thermal_path)
        except (FileNotFoundError, RuntimeError) as e:
            st.warning(f"Thermal report could not be parsed: {e}")

    # ── 2: Images ──
    t0 = time.perf_counter()
    try:
        inspection_images = step_extract_images(inspection_path, config.INSPECTION_IMAGES_DIR)
        thermal_note = ""
        if thermal_doc:
            try:
                thermal_images = step_extract_images(thermal_path, config.THERMAL_IMAGES_DIR)
                thermal_note = f"  Thermal: {thermal_images.unique_count} images."
            except (FileNotFoundError, RuntimeError) as e:
                st.warning(f"Thermal image extraction failed: {e}")
        elapsed = time.perf_counter() - t0
        done(1, f"{elapsed:.1f}s",
             f"Inspection: {inspection_images.unique_count} unique "
             f"({inspection_images.duplicates_skipped} duplicates removed).{thermal_note}",
             elapsed)
    except (FileNotFoundError, RuntimeError) as e:
        fail(1, str(e)); return

    # ── 3: AI extraction ──
    config.AI_MODEL = user_config["ai_model"]
    t0 = time.perf_counter()
    try:
        inspection_result = step_extract_observations(inspection_doc)
        elapsed = time.perf_counter() - t0
        done(2, f"{elapsed:.1f}s",
             f"{inspection_result.count} observations extracted using "
             f"{user_config['ai_model']} — "
             f"{inspection_result.duplicates_removed} duplicates removed",
             elapsed)
    except EnvironmentError as e:
        fail(2, str(e)); return
    except (RuntimeError, ValueError) as e:
        fail(2, str(e)); return

    thermal_result = step_build_empty_extraction_result("thermal_report.pdf")
    if thermal_doc:
        try:
            thermal_result = step_extract_observations(thermal_doc)
        except (EnvironmentError, RuntimeError, ValueError) as e:
            st.warning(f"Thermal observation extraction failed: {e}")

    # ── 4: Image mapping ──
    t0 = time.perf_counter()
    mapping_result = ObservationImageMapper(
        observations=inspection_result.observations,
        image_result=inspection_images,
    ).build_map()
    elapsed = time.perf_counter() - t0
    done(3, f"{elapsed:.1f}s",
         f"{mapping_result.observations_matched}/{inspection_result.count} "
         f"observations matched ({mapping_result.match_rate:.0f}%)",
         elapsed)

    # ── 5: Conflict detection ──
    t0 = time.perf_counter()
    conflict_result = ConflictDetector(
        inspection_result=inspection_result,
        thermal_result=thermal_result,
    ).detect()
    elapsed = time.perf_counter() - t0
    done(4, f"{elapsed:.1f}s",
         f"{conflict_result.count} conflicts across "
         f"{conflict_result.areas_compared} areas — "
         f"{len(conflict_result.high_severity)} high severity",
         elapsed)

    # ── 6: Severity ──
    t0 = time.perf_counter()
    severity_result = SeverityAssessor(
        observations=inspection_result.observations,
        source_name=inspection_doc.source_path.name,
    ).assess()
    summary = severity_result.summary()
    elapsed = time.perf_counter() - t0
    done(5, f"{elapsed:.1f}s",
         f"Critical: {summary.get('critical',0)}  "
         f"High: {summary.get('high',0)}  "
         f"Medium: {summary.get('medium',0)}  "
         f"Low: {summary.get('low',0)}",
         elapsed)

    # ── 7: Generate report ──
    t0 = time.perf_counter()
    combined_images = inspection_images
    if thermal_images:
        combined_images = ExtractionResult(
            images=inspection_images.images + thermal_images.images,
            total_found=inspection_images.total_found + thermal_images.total_found,
            duplicates_skipped=(inspection_images.duplicates_skipped
                                + thermal_images.duplicates_skipped),
            source_pdf=inspection_images.source_pdf,
        )
    try:
        report = DDRGenerator(
            observation_result=inspection_result,
            severity_result=severity_result,
            conflict_result=conflict_result,
            image_result=combined_images,
            property_address=user_config["property_address"],
            inspected_by=user_config["inspected_by"],
            inspection_date=user_config["inspection_date"],
            output_dir=config.OUTPUT_DIR,
        ).generate()
    except Exception as e:
        fail(6, str(e)); return

    elapsed = time.perf_counter() - t0
    done(6, f"{elapsed:.1f}s",
         f"HTML report saved — {report.observation_count} observations, "
         f"{report.conflict_count} conflicts documented",
         elapsed)

    progress_bar.progress(100, text="Pipeline complete")

    st.session_state.report           = report
    st.session_state.step_results     = {
        "inspection_images": inspection_images.unique_count,
        "images_matched":    mapping_result.observations_matched,
    }
    st.session_state.severity_summary  = summary
    st.session_state.pipeline_running  = False
    st.session_state.pipeline_complete = True
    st.session_state.status_text       = "ANALYSIS COMPLETE"
    st.session_state.status_state      = "ready"
    st.rerun()


# ---------------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------------

def render_results() -> None:
    report   = st.session_state.report
    sr       = st.session_state.step_results
    summary  = st.session_state.severity_summary
    times    = {k: v for k, v in st.session_state.step_times.items()
                if isinstance(k, int)}
    total_t  = sum(times.values())

    st.markdown(
        '<div class="results-header">'
        '<span class="results-title">Analysis Complete</span>'
        '<span class="results-badge">Ready for Export</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, report.observation_count,       "Observations"),
        (c2, report.conflict_count,          "Conflicts"),
        (c3, sr.get("inspection_images", 0), "Images"),
        (c4, f"{total_t:.1f}s",              "Total Time"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-val">{val}</div>'
            f'<div class="metric-lbl">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<p class="pipeline-section-label">Severity Distribution</p>',
                    unsafe_allow_html=True)
        total_issues = sum(summary.values()) or 1
        for level in ["critical", "high", "medium", "low"]:
            count = summary.get(level, 0)
            pct   = (count / total_issues) * 100
            st.markdown(
                f'<div class="sev-wrap">'
                f'<div class="sev-top">'
                f'<span class="sev-top-name">{level.capitalize()}</span>'
                f'<span class="sev-top-count">{count}</span>'
                f'</div>'
                f'<div class="sev-bar">'
                f'<div class="sev-fill sev-{level}" style="width:{pct:.1f}%"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    with col_r:
        st.markdown('<p class="pipeline-section-label">Processing Timeline</p>',
                    unsafe_allow_html=True)
        for i, (name, _) in enumerate(PIPELINE_STEPS):
            st.markdown(
                f'<div class="timeline-row">'
                f'<span class="timeline-name">{name}</span>'
                f'<span class="timeline-time">{times.get(i, 0.0):.2f}s</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="pipeline-section-label">Export Report</p>', unsafe_allow_html=True)

    report_html = report.output_path.read_text(encoding="utf-8")
    timestamp   = report.generated_at.strftime("%Y%m%d_%H%M%S")

    col_dl, col_new = st.columns([1, 2])
    with col_dl:
        st.download_button(
            label="Download DDR Report",
            data=report_html,
            file_name=f"DDR_Report_{timestamp}.html",
            mime="text/html",
            type="primary",
            use_container_width=True,
            key="dl_btn",
        )
    with col_new:
        if st.button("Start New Analysis", use_container_width=True, key="new_btn"):
            for k in ["pipeline_running", "pipeline_complete", "step_times",
                      "report", "step_results", "severity_summary", "user_config"]:
                st.session_state.pop(k, None)
            st.session_state.status_text  = "SYSTEM READY"
            st.session_state.status_state = "ready"
            _init_state()
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    user_cfg = render_sidebar()

    # Status bar
    status_bar(
        st.session_state.get("status_text", "SYSTEM READY"),
        st.session_state.get("status_state", "ready"),
    )

    # Content wrapper open
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

    # ── IDLE ──
    if not st.session_state.pipeline_running and not st.session_state.pipeline_complete:
        st.markdown(
            '<div class="hero-wrap">'
            '<div class="hero-pill"><div class="hero-pill-dot"></div>AI-Powered Analysis</div>'
            '<div class="hero-title">Detailed Diagnostic <em style="color:var(--amber-bright);font-style:italic;">Report Generator</em></div>'
            '<div class="hero-sub">Upload inspection documents and thermal imaging PDFs. '
            'Our multi-step AI pipeline extracts observations, detects conflicts, '
            'assesses severity, and generates client-ready reports.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Show upload placeholder or ready state
        if user_cfg["inspection_pdf"] is None:
            st.markdown(
                '<div class="upload-placeholder">'
                '<div class="upload-placeholder-icon">&#128196;</div>'
                '<div class="upload-placeholder-text">'
                'Upload inspection PDFs using the sidebar to begin analysis'
                '</div></div>',
                unsafe_allow_html=True,
            )
        else:
            # Feature cards
            c1, c2, c3 = st.columns(3)
            cards = [
                ("Multi-Document Support",
                 "Process both inspection reports and thermal imaging PDFs "
                 "in a single pipeline run."),
                ("Conflict Detection",
                 "AI automatically identifies contradictions between visual "
                 "and thermal findings and flags them for review."),
                ("Severity Classification",
                 "Each finding is classified as critical, high, medium, or low "
                 "with full reasoning provided in the report."),
            ]
            for col, (title, body) in zip([c1, c2, c3], cards):
                col.markdown(
                    f'<div style="background:var(--card);border:1px solid var(--border);'
                    f'border-radius:10px;padding:20px;">'
                    f'<p style="font-size:0.62rem;font-weight:600;color:var(--fg-3);'
                    f'text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px;">'
                    f'{title}</p>'
                    f'<p style="font-size:0.85rem;color:var(--fg);line-height:1.65;margin:0;">'
                    f'{body}</p></div>',
                    unsafe_allow_html=True,
                )

    # ── RUNNING ──
    elif st.session_state.pipeline_running:
        st.markdown(
            '<p class="pipeline-section-label">Pipeline Execution</p>',
            unsafe_allow_html=True,
        )
        uc = st.session_state.get("user_config")
        if uc:
            run_pipeline(uc)
        else:
            st.error("Session expired. Please upload files and try again.")
            st.session_state.pipeline_running = False

    # ── RESULTS ──
    else:
        render_results()

    # Content wrapper close
    st.markdown("</div>", unsafe_allow_html=True)

    # Fixed footer bar
    footer_bar()


if __name__ == "__main__":
    main()