"""Fine Print: M&A agreement review and deal simulation (Streamlit app).

Run with:  streamlit run app.py
The Gemini API key is read from utils.py; users of the app never enter one.
"""

from __future__ import annotations

import logging
from html import escape

import streamlit as st

from ma_engine.llm.base import LLMError
from ma_engine.reporting.markdown_report import report_to_markdown
from ma_engine.ui import landing, state
from ma_engine.ui.components import display_title, html, masthead, note
from ma_engine.ui.theme import apply_theme
from ma_engine.ui.views import advice, ask, document, financials, growth, overview, risks, scorecard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Fine Print - M&A agreement review", page_icon="📑", layout="wide",
                   initial_sidebar_state="collapsed")
apply_theme()
state.init_state()
engine = state.get_engine()
status = state.connection_status()

# The seven product features, in order, plus Q&A over the agreement.
SECTIONS = [
    "Document & RAG",            # 1 ingestion, preprocessing and RAG
    "Summary",                   # 2 executive summary
    "Cash flow",                 # 3 financial impact and cash-flow simulation
    "Score",                     # 4 confidence scoring and rubric matrix
    "Growth",                    # 5 growth and profitability trajectory
    "Risks",                     # 6 fine print risk flags and anomaly detection
    "Mitigation",                # 7 mitigation and strategy advisor
    "Ask",                       # Q&A over the agreement
]
STATUS_NOTE = {"live": "", "offline": "Offline preview", "error": "Gemini connection problem"}


def show_status() -> None:
    if status.detail:
        note(status.detail)
    if status.state == "error":
        where = "the Ollama settings" if engine.settings.llm.is_local else "the key"
        st.error(f"{status.message} The app reads {where} from utils.py in the project folder; save the file and "
                 "reload this page.")
    elif status.state == "offline":
        note(f"{status.message} Pattern checks, simulations and scoring still run in full.")


def run_analysis(raw_doc) -> None:
    state.reset_results()
    with st.status("Reviewing the agreement", expanded=True) as box:
        bar = st.progress(0.0, text="Reading the agreement")
        try:
            ws = engine.build_workspace([raw_doc])
            report = engine.analyze(ws, ws.doc_ids[0], progress=lambda msg, frac: bar.progress(frac, text=msg))
        except (LLMError, ValueError) as exc:
            box.update(label="The review could not be completed", state="error")
            st.session_state.error = str(exc)
            return
        box.update(label="Review complete", state="complete", expanded=False)
    st.session_state.workspace = ws
    st.session_state.report = report
    st.session_state.section = "Summary"
    st.rerun()


report = st.session_state.report
ws = st.session_state.workspace

# ------------------------------------------------------------------ landing
if report is None or ws is None:
    masthead(status.message if status.ok else STATUS_NOTE[status.state])
    show_status()
    if status.ok and engine.settings.llm.is_local:
        note("Running on your own GPU: nothing leaves this machine. A full review of a long agreement takes "
             "roughly 8 to 15 minutes on an NVIDIA T4; progress is shown as each step finishes.")
    doc = landing.render(status)
    if st.session_state.error:
        st.error(f"The review stopped: {st.session_state.error}")
    if doc is not None:
        run_analysis(doc)
    st.stop()

# ------------------------------------------------------------------- report
masthead(status.message if status.ok else STATUS_NOTE[status.state])
title_col, action_col = st.columns([2.6, 1.4], vertical_alignment="bottom")
with title_col:
    html(f'<div class="fp-doc-title">{escape(display_title(report.document.title, report.document.clean_text))}</div>')
    note(f"{report.document.word_count:,} words reviewed in {report.num_chunks} passages.")
with action_col:
    b1, b2 = st.columns(2)
    b1.download_button("Download memo", report_to_markdown(report), width="stretch",
                       file_name=f"{report.document.title}-review.md", mime="text/markdown")
    if b2.button("New review", width="stretch"):
        state.reset_results()
        st.rerun()
if report.demo_mode:
    note("Offline preview: AI-written sections are placeholders until a working Gemini key is in utils.py.")
for w in report.warnings:
    if not w.startswith("Illustrative defaults"):
        st.warning(w)

st.session_state.setdefault("section", "Summary")
section = st.segmented_control("Section", SECTIONS, key="section", label_visibility="collapsed") or "Summary"

if section == "Document & RAG":
    document.render(report, ws, engine)
elif section == "Summary":
    overview.render(report)
elif section == "Cash flow":
    overrides = financials.render(report)
    if overrides is not None:
        with st.spinner("Updating the simulation"):
            st.session_state.report = engine.recompute_financials(report, overrides)
        st.session_state.advice_stale = report.advice is not None and not report.demo_mode
        st.rerun()
elif section == "Score":
    scorecard.render(report)
elif section == "Growth":
    growth.render(report)
elif section == "Risks":
    risks.render(report, ws.index)
elif section == "Mitigation":
    if advice.render(report, stale=bool(st.session_state.advice_stale)):
        with st.spinner("Rewriting the recommendations"):
            try:
                st.session_state.report = engine.refresh_advice(report)
                st.session_state.advice_stale = False
            except LLMError as exc:
                st.error(str(exc))
        st.rerun()
else:
    ask.render(engine, ws, report.document.doc_id)
