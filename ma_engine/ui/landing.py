"""Landing screen: explain the product in one breath, then pick, upload or browse agreements."""

from __future__ import annotations

import streamlit as st

from ..ingestion.archives import unpack
from ..ingestion.loader import RawDocument, load_uploads
from ..ingestion.metadata import describe
from ..llm.connection import ConnectionStatus
from .components import html, note
from .library import library_items

SOURCES = ["Library", "Files or .zip", "Folder"]


@st.cache_data(show_spinner="Reading the dataset", max_entries=4)
def _read_uploads(payload: tuple[tuple[str, bytes], ...]) -> tuple[list[RawDocument], list[str]]:
    result = unpack(list(payload))
    return load_uploads(result.files), result.skipped


def _label(doc: RawDocument) -> str:
    words = len(doc.text.split())
    who = describe(doc.text, "")
    name = doc.title.replace("_", " ")
    return f"{name}: {who} ({words:,} words)" if who else f"{name} ({words:,} words)"


def _pick(docs: list[RawDocument], skipped: list[str]) -> RawDocument | None:
    if skipped:
        shown = ", ".join(skipped[:4]) + (f" and {len(skipped) - 4} more" if len(skipped) > 4 else "")
        note(f"Skipped {len(skipped)} non-text files ({shown}). Only .txt agreements are analysed.")
    if not docs:
        st.warning("No readable .txt agreements were found in the upload.")
        return None
    if len(docs) == 1:
        note(f"Ready: {_label(docs[0])}.")
        return docs[0]
    return st.selectbox(f"{len(docs)} agreements found. Type to search, then choose one to review:",
                        docs, format_func=_label)


def render(status: ConnectionStatus) -> RawDocument | None:
    """Return the document to analyse once the user presses the button."""
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        html("""<div class="fp-hero">
          <h1>Read the fine print before you sign.</h1>
          <p>Upload a merger agreement, letter of intent or filing, or a whole dataset of them. You get an
          executive summary, the clauses that put the buyer at risk, cash-flow stress tests, a scored verdict
          and a negotiating plan, each traced back to the exact wording in the document.</p>
        </div>""")

    with right:
        with st.container(border=True):
            st.markdown("#### Choose an agreement")
            source = st.segmented_control("Source", SOURCES, default=SOURCES[0], label_visibility="collapsed",
                                          key="landing_source") or SOURCES[0]
            doc: RawDocument | None = None
            if source == "Library":
                items = library_items()
                if items:
                    doc = st.selectbox(f"{len(items)} agreements available. Type to search.", items,
                                       format_func=lambda i: i.label).doc
                else:
                    note("The library is empty. Add .txt agreements to data/raw, or upload them.")
            elif source == "Files or .zip":
                ups = st.file_uploader("Text files or a .zip of them (split documents are joined automatically)",
                                       type=["txt", "zip"], accept_multiple_files=True, key="up_files")
                if ups:
                    doc = _pick(*_read_uploads(tuple((u.name, u.getvalue()) for u in ups)))
            else:
                ups = st.file_uploader("A folder of .txt agreements", accept_multiple_files="directory",
                                       key="up_folder")
                if ups:
                    doc = _pick(*_read_uploads(tuple((u.name, u.getvalue()) for u in ups)))

            blocked = status.state == "error"
            clicked = st.button("Analyze agreement", type="primary", disabled=doc is None or blocked, width="stretch")
            if blocked:
                note("Analysis is paused until the Gemini connection problem above is fixed.")

    html("""<div class="fp-capabilities">
      <div><h4>Executive summary</h4><p>Parties, price, consideration, conditions and timeline, sized to the document.</p></div>
      <div><h4>Clause-level risk flags</h4><p>Predatory or off-market terms and inconsistent figures, each with the quoted clause.</p></div>
      <div><h4>Cash-flow stress tests</h4><p>Five scenarios and 2,000 simulations of post-merger liquidity, with a scored verdict.</p></div>
    </div>""")
    return doc if clicked else None
