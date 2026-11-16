"""Feature 1 - Document ingestion, preprocessing and RAG pipeline.

Shows how the agreement was processed, lets the user query the retrieval index
directly (no AI call), and lists the terms extracted with their evidence.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from ...models import AnalysisReport
from ...pipeline import MAEngine, Workspace
from ..components import figures, html, note, section_title

FIELD_LABELS = {
    "acquirer_name": "Acquirer", "target_name": "Target", "deal_value_usd": "Deal value",
    "cash_consideration_usd": "Cash consideration", "stock_consideration_usd": "Stock consideration",
    "per_share_price_usd": "Price per share", "termination_fee_usd": "Termination fee (paid by target)",
    "reverse_termination_fee_usd": "Reverse termination fee (paid by acquirer)", "escrow_usd": "Escrow",
}


def _fmt(key: str, v) -> str:
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, list):
        return ", ".join(map(str, v))
    if isinstance(v, float) and key.endswith("_usd"):
        return f"${v:,.0f}"
    if isinstance(v, float) and (key.endswith("_rate") or key.endswith("_pct") or key.endswith("_premium")):
        return f"{v:.2%}"
    if isinstance(v, float) and key.endswith("_months"):
        return f"{v:g} months"
    return str(v)


def render(r: AnalysisReport, ws: Workspace, engine: MAEngine) -> None:
    d = r.document
    cfg = engine.settings
    figures([("Words after cleaning", f"{d.word_count:,}"), ("Passages indexed", f"{r.num_chunks}"),
             ("Sections detected", f"{len(d.sections)}"), ("Embedding method", ws.index.embedder.name)])

    html(f"""<div class="fp-panel"><div class="fp-item-title" style="margin-bottom:.4rem">How the agreement was processed</div>
      <ol style="margin:0;padding-left:1.2rem;color:var(--slate)">
        <li>Formatting stripped: page numbers, running headers, filing tags and hard line breaks{escape(", split pieces joined" if "pieces" in d.source_path else "")}.</li>
        <li>Split into passages of about {cfg.chunking.chunk_words} words that overlap by {cfg.chunking.overlap_words} words, cut at paragraph or sentence ends and labelled with their article and section.</li>
        <li>Indexed twice: semantic vectors ({escape(ws.index.embedder.name)}) and keyword search (BM25), combined by reciprocal rank fusion.</li>
        <li>Each analysis step retrieves only the passages it needs, so long agreements never exceed the model's context.</li>
      </ol></div>""")

    section_title("Search the index")
    note("This runs the same retrieval the analysis uses, without calling the AI model.")
    query = st.text_input("Search the agreement", placeholder="e.g. termination fee payable by the company",
                          label_visibility="collapsed", key="rag_query")
    if query.strip():
        hits = ws.retriever.search(query, k=5, doc_id=d.doc_id)
        if not hits:
            note("No passages matched.")
        for h in hits:
            ranks = ", ".join(x for x in (f"semantic rank {h.dense_rank}" if h.dense_rank else "",
                                          f"keyword rank {h.sparse_rank}" if h.sparse_rank else "") if x)
            with st.expander(f"{h.chunk.section or 'Opening pages'} ({ranks})"):
                st.text(h.chunk.text)

    if r.deal_terms:
        section_title("Terms extracted from the agreement")
        terms = r.deal_terms.model_dump(exclude={"evidence", "notes"})
        rows = [{"Term": FIELD_LABELS.get(k, k.replace("_usd", "").replace("_", " ").capitalize()), "Value": _fmt(k, v)}
                for k, v in terms.items() if v not in (None, [])]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            note("No terms could be extracted.")
        if r.deal_terms.evidence:
            with st.expander("Where each figure was found"):
                st.dataframe(pd.DataFrame([{"Term": e.field_name, "Passage": e.chunk_id, "Quote": e.quote}
                                           for e in r.deal_terms.evidence]), hide_index=True, width="stretch")
        for n in r.deal_terms.notes:
            note(n)

    with st.expander("Document structure"):
        st.markdown("\n".join(f"- {s}" for s in d.sections) or "No headings detected.")
    with st.expander(f"All {r.num_chunks} passages"):
        for ch in ws.index.chunks_for(d.doc_id):
            st.markdown(f"**{ch.chunk_id}**, {ch.section or 'Opening pages'}")
            st.text(ch.text[:900] + ("..." if len(ch.text) > 900 else ""))

    u = r.usage
    section_title("Run details")
    note(f"{u.calls} model calls, {u.input_tokens:,} input and {u.output_tokens:,} output tokens, "
         f"{u.cached_responses} answers reused from cache.")
    st.download_button("Download full results (JSON)", r.model_dump_json(indent=2),
                       file_name=f"{d.title}-review.json", mime="application/json")
