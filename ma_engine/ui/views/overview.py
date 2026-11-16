"""Overview tab: the verdict, headline figures and the executive summary."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from ...models import AnalysisReport
from ...utils.formatting import money, pct
from ..components import figures, html, section_title, verdict


def render(r: AnalysisReport) -> None:
    sc, s = r.scorecard, r.summary
    base = r.simulations[0] if r.simulations else None
    if sc:
        verdict(sc.total, sc.rating, sc.verdict, s.headline if s else "")

    serious = sum(f.severity in ("critical", "high") for f in r.flags)
    items = [("Risk flags", f"{len(r.flags)}"), ("Critical or high", f"{serious}")]
    if base:
        items.append(("Lowest cash, base case", money(base.metrics.minimum_cash)))
    if r.monte_carlo:
        items.append(("Chance of covenant breach", pct(r.monte_carlo.prob_covenant_breach, 0)))
    figures(items)

    if s is None:
        st.info("The executive summary could not be produced for this document. The other tabs are complete.")
        return

    left, right = st.columns([1.6, 1], gap="large")
    with left:
        section_title("Executive summary")
        if s.deal_type:
            html(f'<p class="fp-note" style="margin-top:-.4rem">{escape(s.deal_type)}</p>')
        st.markdown(s.summary)
        if s.top_concerns:
            section_title("What leadership should know first")
            st.markdown("\n".join(f"- {c}" for c in s.top_concerns))
        if s.valuation_terms:
            section_title("Price and consideration")
            st.markdown("\n".join(f"- {v}" for v in s.valuation_terms))
    with right:
        if s.key_metrics:
            section_title("Key terms")
            st.dataframe(pd.DataFrame([{"Term": m.label, "Value": m.value} for m in s.key_metrics]),
                         hide_index=True, width="stretch")
        if s.stakeholders:
            section_title("Parties")
            st.dataframe(pd.DataFrame([{"Party": p.name, "Role": p.role} for p in s.stakeholders]),
                         hide_index=True, width="stretch")
        if s.timeline:
            section_title("Timeline")
            st.dataframe(pd.DataFrame([{"Event": t.event, "When": t.timing} for t in s.timeline]),
                         hide_index=True, width="stretch")
