"""Feature 5 - Visual growth and profitability trajectory analytics."""

from __future__ import annotations

import streamlit as st

from ...models import AnalysisReport, SimulationResult
from ...utils.formatting import money, pct
from ...visualization.charts import (
    cumulative_fcf_chart,
    ebitda_margin_chart,
    growth_vs_liabilities_chart,
    revenue_trajectory_chart,
)
from ..components import figures, html, note, section_title

CHART_CONFIG = {"displayModeBar": False}


def _annual(s: SimulationResult, series: list[float], last: bool) -> float:
    window = series[-12:] if last else series[1:13]
    return sum(window)


def outlook(base: SimulationResult, downside: SimulationResult) -> tuple[str, str]:
    """One-line verdict on long-term growth versus capital drain."""
    b, d = base.metrics, downside.metrics
    if b.npv_free_cash_flow > 0 and b.npv_synergies > 0 and d.first_negative_cash_month is None:
        return "Positive long-term growth", "Synergies more than cover integration costs and cash stays positive even in the severe downside."
    if b.npv_free_cash_flow > 0 and b.npv_synergies > 0:
        return "Growth in the base case, fragile in the downside", (
            f"The plan creates value, but the severe downside runs out of cash in month {d.first_negative_cash_month}.")
    if b.npv_synergies <= 0:
        return "Synergies do not pay back", "Integration costs exceed the value of synergies within three years: a capital drain unless the plan improves."
    return "Capital drain", "Free cash flow after closing is negative in the base case."


def render(r: AnalysisReport) -> None:
    if not r.simulations:
        st.info("No projections are available for this document.")
        return
    base = r.simulations[0]
    downside = next((s for s in r.simulations if s.scenario.name == "Severe Downside"), r.simulations[-1])

    title, text = outlook(base, downside)
    html(f'<div class="fp-panel"><h3 style="margin:0 0 .35rem">{title}</h3><p style="margin:0;color:var(--slate)">{text}</p></div>')

    rev_y1, rev_y3 = _annual(base, base.revenue, False), _annual(base, base.revenue, True)
    ebitda_y1, ebitda_y3 = _annual(base, base.ebitda, False), _annual(base, base.ebitda, True)
    figures([
        ("Revenue growth, year 1 to year 3", pct(rev_y3 / rev_y1 - 1 if rev_y1 else 0)),
        ("EBITDA margin, year 3", pct(ebitda_y3 / rev_y3 if rev_y3 else 0)),
        ("Margin change since year 1", f"{(ebitda_y3 / rev_y3 - ebitda_y1 / rev_y1) * 100:+.1f} pts" if rev_y1 and rev_y3 else "n/a"),
        ("Synergy NPV net of integration", money(base.metrics.npv_synergies)),
    ])
    note("Base case unless stated. Projections use the assumptions in the Cash flow section.")

    st.plotly_chart(revenue_trajectory_chart(r.simulations), width="stretch", config=CHART_CONFIG)
    left, right = st.columns(2)
    left.plotly_chart(ebitda_margin_chart(r.simulations), width="stretch", config=CHART_CONFIG)
    right.plotly_chart(cumulative_fcf_chart(r.simulations), width="stretch", config=CHART_CONFIG)

    section_title("Synergy value against worst-case liabilities")
    st.plotly_chart(growth_vs_liabilities_chart(base, downside), width="stretch", config=CHART_CONFIG)
    note("Blue: value created by synergies in the base case. Orange: integration costs, hidden liabilities, "
         "accelerated debt and earnouts in the severe downside. Where orange stays above blue, the deal drains capital.")
