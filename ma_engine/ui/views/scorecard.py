"""Feature 4 - Multi-tiered confidence scoring with a visible rubric matrix."""

from __future__ import annotations

from html import escape

import streamlit as st

from ...models import AnalysisReport
from ...visualization.charts import scorecard_chart
from ..components import html, note, section_title, verdict


def _matrix(r: AnalysisReport) -> str:
    rows = []
    for c in r.scorecard.categories:
        rows.append(f"""<tr class="fp-mx-cat"><td colspan="4">{escape(c.name)}</td>
          <td class="num">{c.awarded:.1f} / {c.max_points:.0f}</td></tr>""")
        for x in c.criteria:
            share = 0 if x.max_points == 0 else x.awarded / x.max_points
            rows.append(f"""<tr><td>{escape(x.name)}</td><td class="num">{x.max_points:.1f}</td>
              <td class="num">{x.awarded:.1f}</td><td class="num">{x.max_points - x.awarded:.1f}</td>
              <td><div class="fp-bar"><i style="width:{share * 100:.0f}%"></i></div>
              <div class="fp-note">{escape(x.rationale)}</div></td></tr>""")
    return f"""<table class="fp-matrix"><thead><tr><th>Criterion</th><th class="num">Weight</th>
      <th class="num">Awarded</th><th class="num">Docked</th><th>Why</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"""


def render(r: AnalysisReport) -> None:
    sc = r.scorecard
    if sc is None:
        st.info("No score is available for this document.")
        return
    verdict(sc.total, sc.rating, sc.verdict)
    st.plotly_chart(scorecard_chart(sc), width="stretch", config={"displayModeBar": False})
    section_title("Rubric matrix")
    note("Every criterion is scored by a fixed rule over the extracted terms, the risk flags and the simulation. "
         "Weights are the maximum points; docked points show exactly where confidence was lost.")
    html(_matrix(r))
    flagged = [(c.name, x) for c in sc.categories for x in c.criteria if x.evidence]
    if flagged:
        with st.expander("Risk flags behind the legal and regulatory scores"):
            for cat, x in flagged:
                st.markdown(f"**{x.name}** ({cat})")
                st.markdown("\n".join(f"- {e}" for e in x.evidence))
