"""Visual identity for the dashboard: palette, typography and Streamlit overrides.

Direction: a private-bank deal book. Ink navy on cool paper, white panels,
one restrained oxblood accent, a brass hairline under the masthead. Serif
(Source Serif 4) for headings and key figures; IBM Plex Sans for interface text.
"""

from __future__ import annotations

import streamlit as st

INK = "#172033"
SLATE = "#4F5B6E"
MUTED = "#7A8496"
PAPER = "#F3F4F6"
SURFACE = "#FFFFFF"
LINE = "#DCE0E6"
OXBLOOD = "#7B2D3A"
BRASS = "#A08446"

SERIF = "'Source Serif 4', 'Source Serif Pro', Georgia, 'Times New Roman', serif"
SANS = "'IBM Plex Sans', -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');

:root {{
  --ink: {INK}; --slate: {SLATE}; --muted: {MUTED}; --paper: {PAPER};
  --surface: {SURFACE}; --line: {LINE}; --oxblood: {OXBLOOD}; --brass: {BRASS};
}}

/* ---------- canvas & chrome ---------- */
html, body, [data-testid="stAppViewContainer"], .stApp {{ background: var(--paper); color: var(--ink); }}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp [data-testid="stMarkdownContainer"] {{ font-family: {SANS}; }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer,
[data-testid="stStatusWidget"], [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
section[data-testid="stSidebar"] {{ display: none; }}
.block-container {{ max-width: 1180px; padding-top: 1.6rem; padding-bottom: 4rem; }}

h1, h2, h3, h4 {{ font-family: {SERIF} !important; color: var(--ink); letter-spacing: -0.005em; }}
h1 {{ font-weight: 600; }} h2, h3 {{ font-weight: 600; }}
.stApp p, .stApp li {{ line-height: 1.6; }}
a {{ color: var(--oxblood); }}
:focus-visible {{ outline: 2px solid var(--oxblood) !important; outline-offset: 2px; }}

/* ---------- masthead ---------- */
.fp-masthead {{ display: flex; align-items: flex-end; justify-content: space-between;
  border-bottom: 1px solid var(--brass); padding-bottom: .7rem; margin-bottom: 1.6rem; }}
.fp-brand {{ font-family: {SERIF}; font-size: 1.65rem; font-weight: 700; color: var(--ink); line-height: 1; }}
.fp-brand small {{ display: block; font-family: {SANS}; font-size: .82rem; font-weight: 400;
  color: var(--slate); margin-top: .35rem; letter-spacing: .01em; }}

/* ---------- landing ---------- */
.fp-hero h1 {{ font-size: 2.6rem; line-height: 1.12; margin: .2rem 0 .8rem; max-width: 18ch; }}
.fp-hero p {{ color: var(--slate); font-size: 1.05rem; max-width: 52ch; }}
.fp-capabilities {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.8rem; margin-top: 2.6rem; }}
.fp-capabilities > div {{ border-top: 1px solid var(--line); padding-top: .8rem; }}
.fp-capabilities h4 {{ font-size: 1.05rem; margin: 0 0 .3rem; }}
.fp-capabilities p {{ color: var(--slate); font-size: .9rem; margin: 0; }}
@media (max-width: 760px) {{ .fp-capabilities {{ grid-template-columns: 1fr; }} .fp-hero h1 {{ font-size: 2rem; }} }}

/* ---------- panels ---------- */
.fp-panel {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 1.2rem 1.4rem; }}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] .fp-panel-marker) {{
  background: var(--surface); }}

/* ---------- the seal (the one bold element) ---------- */
.fp-verdict {{ display: flex; gap: 1.8rem; align-items: center; background: var(--surface);
  border: 1px solid var(--line); border-radius: 6px; padding: 1.4rem 1.6rem; }}
.fp-seal {{ flex: 0 0 auto; width: 132px; height: 132px; border-radius: 50%;
  border: 2px solid var(--seal); box-shadow: inset 0 0 0 5px var(--surface), inset 0 0 0 6px var(--seal);
  display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--seal); }}
.fp-seal b {{ font-family: {SERIF}; font-size: 3rem; font-weight: 600; line-height: 1; font-variant-numeric: lining-nums; }}
.fp-seal span {{ font-size: .72rem; color: var(--slate); margin-top: .25rem; }}
.fp-verdict h2 {{ margin: 0 0 .35rem; font-size: 1.5rem; }}
.fp-verdict p {{ margin: 0; color: var(--slate); max-width: 62ch; }}

/* ---------- figures strip ---------- */
.fp-figures {{ display: grid; grid-template-columns: repeat(4, 1fr); background: var(--surface);
  border: 1px solid var(--line); border-radius: 6px; margin: 1rem 0 1.4rem; }}
.fp-figures div {{ padding: .95rem 1.2rem; border-left: 1px solid var(--line); }}
.fp-figures div:first-child {{ border-left: 0; }}
.fp-figures b {{ display: block; font-family: {SERIF}; font-size: 1.55rem; font-weight: 600; color: var(--ink);
  font-variant-numeric: tabular-nums lining-nums; }}
.fp-figures span {{ font-size: .82rem; color: var(--slate); }}
@media (max-width: 760px) {{ .fp-figures {{ grid-template-columns: 1fr 1fr; }}
  .fp-figures div:nth-child(3) {{ border-left: 0; }} .fp-figures div {{ border-top: 1px solid var(--line); }} }}

/* ---------- severity & tags ---------- */
.fp-sev {{ display: inline-flex; align-items: center; gap: .4rem; font-size: .8rem; font-weight: 500; color: var(--ink); white-space: nowrap; }}
.fp-sev i {{ width: .62rem; height: .62rem; border-radius: 50%; display: inline-block; }}
.fp-tag {{ display: inline-block; font-size: .75rem; color: var(--slate); border: 1px solid var(--line);
  border-radius: 3px; padding: .05rem .4rem; margin-right: .3rem; background: var(--paper); }}
.fp-quote {{ border-left: 2px solid var(--brass); padding: .2rem 0 .2rem .9rem; color: var(--slate);
  font-family: {SERIF}; font-size: .98rem; margin: .6rem 0; }}
.fp-note {{ font-size: .85rem; color: var(--muted); }}
.fp-item-title {{ font-family: {SERIF}; font-size: 1.08rem; font-weight: 600; color: var(--ink); }}
.fp-figure-sm {{ font-family: {SERIF}; font-size: 1.15rem; }}
.fp-section-title {{ font-family: {SERIF}; font-size: 1.25rem; font-weight: 600; margin: 1.6rem 0 .6rem; }}

/* ---------- rubric matrix ---------- */
.fp-matrix {{ width: 100%; border-collapse: collapse; background: var(--surface); border: 1px solid var(--line);
  border-radius: 6px; overflow: hidden; font-size: .9rem; }}
.fp-matrix th {{ text-align: left; font-weight: 500; color: var(--slate); padding: .6rem .8rem; border-bottom: 1px solid var(--line); }}
.fp-matrix td {{ padding: .6rem .8rem; border-bottom: 1px solid var(--line); vertical-align: top; }}
.fp-matrix .num {{ text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
.fp-matrix tr.fp-mx-cat td {{ background: var(--paper); font-family: {SERIF}; font-weight: 600; font-size: 1rem; color: var(--ink); }}
.fp-bar {{ height: 4px; background: var(--line); border-radius: 2px; margin: .25rem 0 .35rem; max-width: 220px; }}
.fp-bar i {{ display: block; height: 4px; background: #2a78d6; border-radius: 2px; }}

/* ---------- segmented controls styled as quiet tabs ---------- */
div[data-testid="stButtonGroup"]:has(button[data-variant="segmented_control"]),
div[data-testid="stButtonGroup"]:has(button[kind^="segmented_control"]) {{ border-bottom: 1px solid var(--line); margin-bottom: .6rem; }}
button[data-variant="segmented_control"], button[kind^="segmented_control"] {{
  background: transparent !important; border: none !important; border-radius: 0 !important; box-shadow: none !important;
  border-bottom: 2px solid transparent !important; padding: .45rem .1rem !important; margin-right: 1.4rem !important;
  min-height: 0 !important; }}
button[data-variant="segmented_control"] p, button[kind^="segmented_control"] p {{ font-size: .95rem; color: var(--slate); }}
button[data-variant="segmented_control"][data-selected="true"], button[kind="segmented_controlActive"] {{
  border-bottom-color: var(--oxblood) !important; }}
button[data-variant="segmented_control"][data-selected="true"] p, button[kind="segmented_controlActive"] p {{
  color: var(--ink) !important; font-weight: 600; }}
button[data-variant="segmented_control"]:hover p, button[kind^="segmented_control"]:hover p {{ color: var(--oxblood); }}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ padding: .6rem 0; background: transparent; }}
.stTabs [data-baseweb="tab"] p {{ font-family: {SANS}; font-size: .95rem; color: var(--slate); }}
.stTabs [aria-selected="true"] p {{ color: var(--ink); font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--oxblood); height: 2px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.4rem; }}

/* ---------- widgets ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius: 4px; border: 1px solid var(--ink); color: var(--ink); background: var(--surface);
  font-weight: 500; padding: .45rem 1.1rem; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: var(--oxblood); color: var(--oxblood); }}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {{
  background: var(--oxblood); border-color: var(--oxblood); color: #fff; }}
.stButton > button[kind="primary"]:hover {{ background: #63222E; color: #fff; }}
[data-testid="stExpander"] details {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }}
[data-testid="stExpander"] summary p {{ font-weight: 500; }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--surface); border: 1px dashed #B9C0CB; border-radius: 6px; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: 6px; }}
[data-testid="stMetricValue"] {{ font-family: {SERIF}; }}
[data-testid="stPlotlyChart"] {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: .3rem; }}
.fp-doc-title {{ font-family: {SERIF}; font-size: 2rem; font-weight: 600; line-height: 1.15; margin: 0 0 .3rem; }}
[data-testid="stChatMessage"] {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }}
[data-testid="stAlert"] {{ border-radius: 6px; }}
div[data-testid="stForm"] {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }}
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
