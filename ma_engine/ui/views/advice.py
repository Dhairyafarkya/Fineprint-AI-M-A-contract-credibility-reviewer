"""Feature 7 - Actionable financial mitigation and strategy advisor."""

from __future__ import annotations

from html import escape

import streamlit as st

from ...models import AnalysisReport
from ...utils.formatting import money, pct
from ..components import figures, html, note, section_title, tag

STANCE = {
    "proceed": "Proceed",
    "proceed_with_conditions": "Proceed with conditions",
    "renegotiate": "Renegotiate before signing",
    "walk_away": "Walk away",
}
PRIORITY = {"P1": "First priority", "P2": "Second priority", "P3": "Third priority"}
ACTION = {
    "renegotiate_terms": "Renegotiate terms", "adjust_escrow": "Escrow and indemnity",
    "price_adjustment": "Price", "structural_change": "Deal structure", "additional_diligence": "Diligence",
    "integration_plan": "Integration", "walk_away": "Walk away",
}


def render(r: AnalysisReport, stale: bool = False) -> bool:
    """Render the section; return True when the user asks to refresh the recommendations."""
    a, mm = r.advice, r.mitigation_math
    refresh = False
    if stale:
        with st.container(border=True):
            note("You changed the financial assumptions after these recommendations were written. The sizing "
                 "figures below are current; the written advice is not.")
            refresh = st.button("Refresh recommendations", type="primary")
    if a is not None:
        html(f"""<div class="fp-panel"><h3 style="margin:0 0 .4rem">{escape(STANCE.get(a.overall_stance, a.overall_stance))}</h3>
             <p style="margin:0;color:var(--slate)">{escape(a.stance_rationale)}</p></div>""")
    if mm:
        be = mm.break_even_synergy_realization
        figures([
            ("Recommended escrow", money(mm.recommended_escrow_usd)),
            ("Escrow shortfall", money(mm.escrow_gap_usd)),
            ("Cash headroom above covenant", money(mm.price_cushion_usd)),
            ("Synergies needed to stay liquid", "None" if be == 0 else "Over 200%" if be is None else pct(be, 0)),
        ])
        note("These sizing figures are calculated from the simulation, not generated.")

    if a is None:
        st.info("Recommendations could not be produced for this document.")
        return refresh

    section_title("Actions")
    for rec in a.recommendations:
        with st.container(border=True):
            impact = f"Value protected about {money(rec.estimated_financial_impact_usd)}" if rec.estimated_financial_impact_usd else ""
            html(f"""<div style="display:flex;justify-content:space-between;gap:1rem;align-items:baseline">
              <div class="fp-item-title">{escape(rec.title)}</div>
              <span class="fp-note">{escape(PRIORITY.get(rec.priority, rec.priority))}</span></div>
              <div style="margin:.35rem 0">{tag(ACTION.get(rec.action_type, rec.action_type))}
              {''.join(tag(fid) for fid in rec.addresses_flag_ids)}<span class="fp-note">{escape(impact)}</span></div>""")
            st.markdown(rec.rationale)
            if rec.proposed_terms:
                html(f'<div class="fp-quote">{escape(rec.proposed_terms)}</div>')

    if a.negotiation_sequence:
        section_title("Order of negotiation")
        st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(a.negotiation_sequence, 1)))
    if a.walk_away_triggers:
        section_title("Walk away if")
        st.markdown("\n".join(f"- {t}" for t in a.walk_away_triggers))
    return refresh
