"""Feature 6 - Fine print risk flagging and anomaly detection."""

from __future__ import annotations

from html import escape

import streamlit as st

from ...models import AnalysisReport
from ...retrieval.index_store import IndexStore
from ...utils.formatting import money
from ...visualization.charts import flags_chart
from ..components import html, note, quote, section_title, severity, tag

SOURCE_LABEL = {"llm": "AI review", "rule": "Pattern check", "anomaly": "Number check"}
ORDER = ["critical", "high", "medium", "low"]


def render(r: AnalysisReport, index: IndexStore | None) -> None:
    if not r.flags:
        st.success("No risk flags were raised for this agreement.")
        return

    st.plotly_chart(flags_chart(r.flags), width="stretch", config={"displayModeBar": False})

    f1, f2 = st.columns([1, 2])
    sev = f1.pills("Severity", ORDER, selection_mode="multi", default=ORDER, format_func=str.capitalize) or []
    cats = f2.multiselect("Category", sorted({f.category for f in r.flags}), placeholder="All categories")
    flags = [f for f in r.flags if f.severity in sev and (not cats or f.category in cats)]
    note(f"Showing {len(flags)} of {len(r.flags)} flags, most severe first.")

    clause_flags = [f for f in flags if f.source != "anomaly"]
    anomalies = [f for f in flags if f.source == "anomaly"]
    section_title(f"Clause risks ({len(clause_flags)})")
    if not clause_flags:
        note("No clause-level risks match the filters.")
    for f in clause_flags:
        _flag_card(f, index)
    section_title(f"Number anomalies ({len(anomalies)})")
    note("Figures checked for internal consistency, e.g. the same metric quoted with different values, "
         "or recurring revenue far above reported revenue.")
    if not anomalies:
        note("No inconsistent figures were found.")
    for f in anomalies:
        _flag_card(f, index)


def _flag_card(f, index: IndexStore | None) -> None:
    with st.container(border=True):
        exposure = f" Estimated exposure {money(f.estimated_exposure_usd)}." if f.estimated_exposure_usd else ""
        html(f"""<div style="display:flex;justify-content:space-between;gap:1rem;align-items:baseline">
          <div class="fp-item-title">{escape(f.title)}</div>
          {severity(f.severity)}</div>
          <div style="margin:.35rem 0 .2rem">{tag(f.category)}{tag(SOURCE_LABEL.get(f.source, f.source))}
          <span class="fp-note">{escape(f.flag_id)}.{escape(exposure)}</span></div>""")
        st.markdown(f.description)
        quote(f.clause_excerpt)
        if f.why_nonstandard:
            html(f'<p class="fp-note"><b>Market practice.</b> {escape(f.why_nonstandard)}</p>')
        if index is not None and f.chunk_ids:
            chunk = index.get(f.chunk_ids[0])
            if chunk is not None:
                with st.expander("Read the full passage"):
                    if chunk.section:
                        note(chunk.section)
                    st.text(chunk.text)
