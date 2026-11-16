"""Ask tab: questions answered only from retrieved passages, with citations."""

from __future__ import annotations

import streamlit as st

from ...llm.base import LLMError
from ...pipeline import MAEngine, Workspace
from ..components import note

SUGGESTIONS = [
    "What happens to the target's debt at closing?",
    "Who pays a termination fee, and when?",
    "Which regulatory approvals are required?",
]


def render(engine: MAEngine, ws: Workspace, doc_id: str) -> None:
    note("Answers come only from passages of this agreement, and cite the passages used.")
    if not st.session_state.qa_history:
        cols = st.columns(len(SUGGESTIONS))
        for col, q in zip(cols, SUGGESTIONS):
            if col.button(q, key=f"sugg_{q}", width="stretch"):
                _ask(engine, ws, doc_id, q)

    for q, ans, hits in st.session_state.qa_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(ans.answer)
            if ans.not_found:
                note("The agreement does not appear to address this.")
            with st.expander(f"Passages used ({len(hits)})"):
                for h in hits:
                    st.markdown(f"**{h.chunk.section or 'Opening pages'}**")
                    st.text(h.chunk.text[:1500])

    question = st.chat_input("Ask about this agreement")
    if question:
        _ask(engine, ws, doc_id, question)


def _ask(engine: MAEngine, ws: Workspace, doc_id: str, question: str) -> None:
    with st.spinner("Searching the agreement"):
        try:
            ans, hits = engine.ask(ws, question, doc_id=doc_id)
        except (LLMError, ValueError) as exc:
            st.error(str(exc))
            return
    st.session_state.qa_history.append((question, ans, hits))
    st.rerun()
