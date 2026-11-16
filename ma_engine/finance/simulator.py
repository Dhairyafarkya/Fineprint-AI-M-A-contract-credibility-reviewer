"""Deterministic post-merger cash-flow simulator (monthly, vectorised with NumPy).

Timeline: month 0 is closing day, months 1..H follow.

Closing day (month 0)
    cash = acquirer cash + target cash + new acquisition debt
         - cash consideration (escrow is funded at closing, so it is included)
         - transaction costs - change-of-control payments
         - target debt (+ prepayment premium) if it accelerates on change of control

Each month t
    revenue      = acquirer + target revenue on their growth paths, x (1 + revenue shock)
    synergies    = run-rate synergies x linear ramp (starts after any integration delay) x realisation
    EBITDA       = revenue - operating costs + cost synergies + revenue synergies x margin
    integration  = integration budget x (1 + overrun), spread evenly over the integration period
    interest     = opening debt balance x rate / 12
    taxes        = max(0, EBITDA - interest - integration) x tax rate
    one-offs     = hidden liability (net of escrow recovery) + earnout
    FCF          = EBITDA - interest - taxes - integration - capex - one-offs
    cash_t       = cash_{t-1} + FCF - scheduled principal repayment
"""

from __future__ import annotations

import numpy as np

from ..models import FinancialInputs, Scenario, SimulationResult
from .metrics import compute_metrics


def upfront_outflow(fi: FinancialInputs, sc: Scenario) -> float:
    """Cash leaving the combined company at closing, net of target cash acquired."""
    accelerate = fi.debt_accelerates_on_change_of_control or sc.force_debt_acceleration
    debt_payoff = fi.target_debt * (1 + fi.target_debt_prepayment_premium) if accelerate else 0.0
    return (
        fi.cash_consideration + fi.transaction_costs + fi.change_of_control_payments
        + sc.extra_change_of_control_cost + debt_payoff - fi.target_cash
    )


def simulate(fi: FinancialInputs, sc: Scenario, horizon: int = 36, annual_discount_rate: float = 0.10) -> SimulationResult:
    if horizon < 1:
        raise ValueError("horizon must be >= 1 month")
    t = np.arange(1, horizon + 1, dtype=float)
    years = t / 12.0

    # ---------------- revenue & costs
    acq_growth = (1 + fi.acquirer_revenue_growth) ** years
    tgt_growth = (1 + fi.target_revenue_growth) ** years
    base_revenue = (fi.acquirer_annual_revenue / 12) * acq_growth + (fi.target_annual_revenue / 12) * tgt_growth
    revenue = base_revenue * (1 + sc.revenue_shock)
    op_costs = (fi.acquirer_annual_operating_costs / 12) * acq_growth + (fi.target_annual_operating_costs / 12) * tgt_growth

    # ---------------- synergies
    delay = max(0, sc.integration_delay_months)
    ramp_len = max(1, fi.synergy_ramp_months)
    ramp = np.clip((t - delay) / ramp_len, 0.0, 1.0) * max(0.0, sc.synergy_realization)
    cost_syn = (fi.annual_cost_synergies / 12) * ramp
    rev_syn = (fi.annual_revenue_synergies / 12) * ramp
    synergy_ebitda = cost_syn + rev_syn * fi.revenue_synergy_margin
    revenue = revenue + rev_syn
    ebitda = base_revenue * (1 + sc.revenue_shock) - op_costs + synergy_ebitda

    # ---------------- integration costs
    integ = np.zeros(horizon)
    n_int = max(1, fi.integration_months)
    start = min(delay, horizon)
    end = min(delay + n_int, horizon)
    if end > start:
        integ[start:end] = fi.integration_cost * (1 + sc.integration_cost_overrun) / n_int

    # ---------------- debt
    accelerate = fi.debt_accelerates_on_change_of_control or sc.force_debt_acceleration
    amort_months = max(1.0, fi.acquisition_debt_amortization_years * 12)
    principal_pm = fi.new_acquisition_debt / amort_months
    acq_open = np.maximum(0.0, fi.new_acquisition_debt - principal_pm * (t - 1))
    principal = np.minimum(principal_pm, acq_open)
    acq_close = acq_open - principal
    tgt_balance = np.zeros(horizon) if accelerate else np.full(horizon, fi.target_debt)
    interest = (
        acq_open * (fi.acquisition_debt_interest_rate + sc.interest_rate_shock) / 12
        + tgt_balance * (fi.target_debt_interest_rate + sc.interest_rate_shock) / 12
    )
    debt_balance = acq_close + tgt_balance

    # ---------------- one-off items
    one_offs = np.zeros(horizon)
    if sc.hidden_liability > 0 and 1 <= sc.hidden_liability_month <= horizon:
        recoverable = 0.0
        if sc.hidden_liability_month <= fi.escrow_release_month:
            recoverable = min(fi.escrow_amount, sc.hidden_liability * sc.hidden_liability_indemnifiable)
        one_offs[sc.hidden_liability_month - 1] += sc.hidden_liability - recoverable
    if fi.earnout_amount > 0 and 1 <= fi.earnout_month <= horizon:
        if sc.earnout_paid is None:
            one_offs[fi.earnout_month - 1] += fi.earnout_amount * fi.earnout_probability
        elif sc.earnout_paid:
            one_offs[fi.earnout_month - 1] += fi.earnout_amount

    # ---------------- cash
    taxes = np.maximum(0.0, ebitda - interest - integ) * fi.tax_rate
    capex = revenue * fi.capex_pct_revenue
    fcf = ebitda - interest - taxes - integ - capex - one_offs

    outflow = upfront_outflow(fi, sc)
    cash0 = fi.acquirer_cash + fi.new_acquisition_debt - outflow
    cash = cash0 + np.cumsum(fcf - principal)

    # Acquirer stand-alone FCF (for deal payback): its own EBITDA after tax and capex, no deal items.
    acq_rev = (fi.acquirer_annual_revenue / 12) * acq_growth
    acq_ebitda = acq_rev - (fi.acquirer_annual_operating_costs / 12) * acq_growth
    standalone_fcf = acq_ebitda - np.maximum(0.0, acq_ebitda) * fi.tax_rate - acq_rev * fi.capex_pct_revenue

    months = [0] + t.astype(int).tolist()
    pad = lambda arr: [0.0] + [float(x) for x in arr]  # noqa: E731
    cash_series = [float(cash0)] + [float(x) for x in cash]
    debt_series = [float(fi.new_acquisition_debt + (0.0 if accelerate else fi.target_debt))] + [float(x) for x in debt_balance]
    synergy_net = synergy_ebitda - integ

    metrics = compute_metrics(
        cash=np.array(cash_series),
        debt=np.array(debt_series),
        fcf=fcf,
        synergy_net=synergy_net,
        incremental_fcf=fcf - standalone_fcf,
        upfront=outflow,
        covenant=fi.minimum_cash_covenant,
        annual_discount_rate=annual_discount_rate,
    )
    return SimulationResult(
        scenario=sc,
        months=months,
        cash=cash_series,
        revenue=pad(revenue),
        operating_costs=pad(op_costs),
        synergies=pad(synergy_ebitda),
        ebitda=pad(ebitda),
        free_cash_flow=pad(fcf),
        debt_balance=debt_series,
        one_off_outflows=pad(one_offs + integ),
        upfront_outflow=float(outflow),
        metrics=metrics,
    )
