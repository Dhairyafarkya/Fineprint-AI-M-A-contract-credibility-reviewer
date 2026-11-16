"""Interactive Plotly charts for the growth, liquidity and risk dashboard (no LLM).

Every chart has one y-axis, hover tooltips, and a legend whenever there is more
than one series.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..models import FinancialInputs, MonteCarloResult, RiskFlag, ScoreCard, SimulationResult
from .theme import REFERENCE, SEQ_LIGHT, SERIES, STATUS, TEXT_SECONDARY, base_layout

M = 1e6


def _covenant_line(fig: go.Figure, covenant: float) -> None:
    fig.add_hline(
        y=covenant / M, line={"color": REFERENCE, "dash": "dash", "width": 1.5},
        annotation_text=f"Minimum cash covenant ${covenant / M:,.0f}M", annotation_position="top right",
        annotation_font_color=TEXT_SECONDARY,
    )


def cash_trajectory_chart(sims: list[SimulationResult], covenant: float) -> go.Figure:
    fig = go.Figure()
    for i, s in enumerate(sims):
        fig.add_trace(go.Scatter(
            x=s.months, y=[c / M for c in s.cash], name=s.scenario.name, mode="lines",
            line={"color": SERIES[i % len(SERIES)], "width": 2},
            hovertemplate="%{y:$,.1f}M",
        ))
    _covenant_line(fig, covenant)
    fig.update_layout(**base_layout("Post-merger cash by scenario", "Cash ($M)"))
    fig.update_xaxes(title_text="Months after closing")
    return fig


def monte_carlo_chart(mc: MonteCarloResult, covenant: float) -> go.Figure:
    x = mc.months
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[v / M for v in mc.cash_p95], name="95th percentile", mode="lines",
                             line={"width": 0}, showlegend=False, hovertemplate="P95 %{y:$,.1f}M"))
    fig.add_trace(go.Scatter(x=x, y=[v / M for v in mc.cash_p5], name="5th-95th percentile", mode="lines",
                             line={"width": 0}, fill="tonexty", fillcolor=SEQ_LIGHT, hovertemplate="P5 %{y:$,.1f}M"))
    fig.add_trace(go.Scatter(x=x, y=[v / M for v in mc.cash_p50], name="Median", mode="lines",
                             line={"color": SERIES[0], "width": 2}, hovertemplate="Median %{y:$,.1f}M"))
    _covenant_line(fig, covenant)
    fig.update_layout(**base_layout(
        f"Monte Carlo cash range ({mc.runs:,} runs) - {mc.prob_covenant_breach:.0%} breach the covenant", "Cash ($M)"))
    fig.update_xaxes(title_text="Months after closing")
    return fig


def growth_vs_liabilities_chart(base: SimulationResult, downside: SimulationResult) -> go.Figure:
    """Cumulative value created by synergies vs cumulative one-off drains in the worst case."""
    syn = np.cumsum(base.synergies) / M
    drains = np.cumsum(downside.one_off_outflows) / M + downside.upfront_outflow / M - base.upfront_outflow / M
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base.months, y=syn, name="Cumulative synergy EBITDA (base)", mode="lines",
                             line={"color": SERIES[0], "width": 2}, hovertemplate="%{y:$,.1f}M"))
    fig.add_trace(go.Scatter(x=downside.months, y=drains, name=f"Cumulative extra outflows ({downside.scenario.name})",
                             mode="lines", line={"color": SERIES[1], "width": 2}, hovertemplate="%{y:$,.1f}M"))
    fig.update_layout(**base_layout("Synergy value vs worst-case liabilities & integration drains", "Cumulative ($M)"))
    fig.update_xaxes(title_text="Months after closing")
    return fig


def ebitda_margin_chart(sims: list[SimulationResult]) -> go.Figure:
    fig = go.Figure()
    for i, s in enumerate(sims):
        margin = [None] + [e / r * 100 if r else None for e, r in zip(s.ebitda[1:], s.revenue[1:])]
        fig.add_trace(go.Scatter(x=s.months[1:], y=margin[1:], name=s.scenario.name, mode="lines",
                                 line={"color": SERIES[i % len(SERIES)], "width": 2}, hovertemplate="%{y:.1f}%"))
    fig.update_layout(**base_layout("Combined EBITDA margin trajectory", "EBITDA margin (%)"))
    fig.update_xaxes(title_text="Months after closing")
    return fig


def closing_waterfall_chart(fi: FinancialInputs, base: SimulationResult) -> go.Figure:
    accel = fi.debt_accelerates_on_change_of_control
    items = [
        ("Acquirer cash", fi.acquirer_cash, "absolute"),
        ("New acquisition debt", fi.new_acquisition_debt, "relative"),
        ("Target cash acquired", fi.target_cash, "relative"),
        ("Cash consideration", -fi.cash_consideration, "relative"),
        ("Transaction costs", -fi.transaction_costs, "relative"),
        ("Change-of-control payments", -fi.change_of_control_payments, "relative"),
        ("Target debt payoff", -(fi.target_debt * (1 + fi.target_debt_prepayment_premium)) if accel else 0.0, "relative"),
        ("Cash at closing", base.cash[0], "total"),
    ]
    items = [it for it in items if it[2] != "relative" or abs(it[1]) > 0]
    fig = go.Figure(go.Waterfall(
        x=[i[0] for i in items], y=[i[1] / M for i in items], measure=[i[2] for i in items],
        increasing={"marker": {"color": SERIES[2]}}, decreasing={"marker": {"color": SERIES[1]}},
        totals={"marker": {"color": SERIES[0]}}, connector={"line": {"color": "#C5CBD3", "width": 1}},
        texttemplate="%{y:$,.0f}M", textposition="outside", hovertemplate="%{x}: %{y:$,.1f}M<extra></extra>",
    ))
    layout = base_layout("Sources and uses of cash at closing", "$M")
    layout["hovermode"] = "closest"
    fig.update_layout(**layout, showlegend=False)
    return fig


def scorecard_chart(card: ScoreCard) -> go.Figure:
    names = [c.name for c in card.categories][::-1]
    pct = [c.confidence_pct for c in card.categories][::-1]
    fig = go.Figure(go.Bar(
        x=pct, y=names, orientation="h", marker={"color": SERIES[0]}, width=0.5,
        text=[f"{p:.0f}%  ({c.awarded:.1f}/{c.max_points:.0f})" for p, c in zip(pct, card.categories[::-1])],
        textposition="outside", hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    layout = base_layout("Confidence by category", "", height=280)
    layout["hovermode"] = "closest"
    fig.update_layout(**layout, showlegend=False)
    fig.update_xaxes(range=[0, 115], title_text="Confidence (%)")
    return fig


def flags_chart(flags: list[RiskFlag]) -> go.Figure:
    cats = sorted({f.category for f in flags})
    fig = go.Figure()
    for sev in ("critical", "high", "medium", "low"):
        counts = [sum(1 for f in flags if f.category == c and f.severity == sev) for c in cats]
        if any(counts):
            fig.add_trace(go.Bar(y=cats, x=counts, name=sev.capitalize(), orientation="h",
                                 marker={"color": STATUS[sev], "line": {"color": "#FFFFFF", "width": 2}},
                                 hovertemplate=f"{sev.capitalize()}: %{{x}}<extra>%{{y}}</extra>"))
    layout = base_layout("Risk flags by category and severity", "", height=max(220, 34 * len(cats) + 120))
    layout["hovermode"] = "closest"
    fig.update_layout(**layout, barmode="stack", bargap=0.5)
    fig.update_xaxes(title_text="Number of flags", dtick=1)
    return fig


def revenue_trajectory_chart(sims: list[SimulationResult]) -> go.Figure:
    """Annualised combined revenue (including revenue synergies) by scenario."""
    fig = go.Figure()
    for i, s in enumerate(sims):
        fig.add_trace(go.Scatter(x=s.months[1:], y=[r * 12 / M for r in s.revenue[1:]], name=s.scenario.name,
                                 mode="lines", line={"color": SERIES[i % len(SERIES)], "width": 2},
                                 hovertemplate="%{y:$,.0f}M a year"))
    fig.update_layout(**base_layout("Combined revenue, annualised", "Revenue ($M a year)"))
    fig.update_xaxes(title_text="Months after closing")
    return fig


def cumulative_fcf_chart(sims: list[SimulationResult]) -> go.Figure:
    """Cumulative free cash flow by scenario: rising lines create value, falling lines drain capital."""
    fig = go.Figure()
    for i, s in enumerate(sims):
        cum = np.cumsum(s.free_cash_flow) / M
        fig.add_trace(go.Scatter(x=s.months, y=cum, name=s.scenario.name, mode="lines",
                                 line={"color": SERIES[i % len(SERIES)], "width": 2}, hovertemplate="%{y:$,.1f}M"))
    fig.add_hline(y=0, line={"color": REFERENCE, "width": 1, "dash": "dot"})
    fig.update_layout(**base_layout("Cumulative free cash flow after closing", "Cumulative FCF ($M)"))
    fig.update_xaxes(title_text="Months after closing")
    return fig
