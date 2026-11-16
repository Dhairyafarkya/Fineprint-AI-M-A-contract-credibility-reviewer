"""Feature 3 - Financial impact and cash-flow simulation: scenarios, Monte Carlo, closing cash, assumptions."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ...models import AnalysisReport
from ...utils.formatting import money, month, pct
from ...visualization.charts import cash_trajectory_chart, closing_waterfall_chart, monte_carlo_chart
from .. import assumptions
from ..components import figures, note, section_title

CHART_CONFIG = {"displayModeBar": False}


def scenario_table(r: AnalysisReport) -> pd.DataFrame:
    return pd.DataFrame([{
        "Scenario": s.scenario.name,
        "Cash at closing": money(s.cash[0]),
        "Lowest cash": f"{money(s.metrics.minimum_cash)} (month {s.metrics.minimum_cash_month})",
        "Cash after 3 years": money(s.metrics.ending_cash),
        "Covenant breached": month(s.metrics.first_covenant_breach_month),
        "Runs out of cash": month(s.metrics.first_negative_cash_month),
        "NPV of free cash flow": money(s.metrics.npv_free_cash_flow),
        "Synergy NPV (net)": money(s.metrics.npv_synergies),
    } for s in r.simulations])


def render(r: AnalysisReport) -> dict[str, Any] | None:
    """Render the tab; return assumption overrides if the user submitted the form."""
    fi = r.financial_inputs
    if not r.simulations or fi is None:
        st.info("No simulation is available for this document.")
        return None

    st.plotly_chart(cash_trajectory_chart(r.simulations, fi.minimum_cash_covenant), width="stretch", config=CHART_CONFIG)
    st.dataframe(scenario_table(r), hide_index=True, width="stretch")
    with st.expander("How each scenario is defined"):
        for s in r.simulations:
            st.markdown(f"**{s.scenario.name}.** {s.scenario.description}")

    if r.monte_carlo:
        mc = r.monte_carlo
        section_title("Range of outcomes")
        st.plotly_chart(monte_carlo_chart(mc, fi.minimum_cash_covenant), width="stretch", config=CHART_CONFIG)
        figures([
            ("Chance cash falls below covenant", pct(mc.prob_covenant_breach)),
            ("Chance cash runs out", pct(mc.prob_negative_cash)),
            ("Cash after 3 years, median", money(mc.ending_cash_p50)),
            ("Cash after 3 years, 5th to 95th pct", f"{money(mc.ending_cash_p5, 0)} to {money(mc.ending_cash_p95, 0)}"),
        ])

    section_title("Cash at closing")
    st.plotly_chart(closing_waterfall_chart(fi, r.simulations[0]), width="stretch", config=CHART_CONFIG)

    with st.expander("Month-by-month base case"):
        b = r.simulations[0]
        st.dataframe(pd.DataFrame({
            "Month": b.months, "Revenue ($M)": [x / 1e6 for x in b.revenue], "EBITDA ($M)": [x / 1e6 for x in b.ebitda],
            "Synergies ($M)": [x / 1e6 for x in b.synergies], "Free cash flow ($M)": [x / 1e6 for x in b.free_cash_flow],
            "Debt ($M)": [x / 1e6 for x in b.debt_balance], "Cash ($M)": [x / 1e6 for x in b.cash],
        }).round(2), hide_index=True, width="stretch")

    section_title("Assumptions")
    note("Figures found in the agreement are filled in automatically. Change any value and update the simulation; "
         "the scorecard updates with it.")
    return assumptions.render(fi)
