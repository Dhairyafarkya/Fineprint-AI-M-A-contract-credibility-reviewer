"""Deterministic sizing numbers that ground the mitigation advisor.

The advisor (Gemini) proposes countermeasures; these figures make sure the
escrow, price and synergy asks are tied to the simulation rather than guessed.
"""

from __future__ import annotations

from ..models import FinancialInputs, MitigationMath, RiskFlag, Scenario, SimulationResult
from .scenarios import flagged_exposure
from .simulator import simulate

MARKET_ESCROW_PCT = 0.10  # typical private-target indemnity escrow


def break_even_synergy_realization(fi: FinancialInputs, horizon: int, rate: float) -> float | None:
    """Lowest synergy realisation that keeps the base case above the cash covenant.

    0.0 means liquidity is safe even with no synergies; None means even 200% is not enough.
    """
    def ok(r: float) -> bool:
        return simulate(fi, Scenario(name="be", synergy_realization=r), horizon, rate).metrics.liquidity_ok

    if ok(0.0):
        return 0.0
    if not ok(2.0):
        return None
    lo, hi = 0.0, 2.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return round(hi, 3)


def compute_mitigation_math(
    fi: FinancialInputs,
    base: SimulationResult,
    flags: list[RiskFlag],
    horizon: int = 36,
    annual_discount_rate: float = 0.10,
) -> MitigationMath:
    deal_value = fi.deal_value or (fi.cash_consideration + fi.stock_consideration)
    exposure = flagged_exposure(flags)
    recommended = max(MARKET_ESCROW_PCT * deal_value, exposure)
    headroom = base.metrics.minimum_cash - fi.minimum_cash_covenant
    run_rate = fi.annual_cost_synergies + fi.annual_revenue_synergies * fi.revenue_synergy_margin
    return MitigationMath(
        recommended_escrow_usd=round(recommended, 2),
        escrow_gap_usd=round(recommended - fi.escrow_amount, 2),
        max_cash_price_for_covenant_usd=round(fi.cash_consideration + headroom, 2),
        price_cushion_usd=round(headroom, 2),
        synergy_npv_usd=round(base.metrics.npv_synergies, 2),
        integration_cost_to_synergy_ratio=round(fi.integration_cost / run_rate, 3) if run_rate > 0 else None,
        break_even_synergy_realization=break_even_synergy_realization(fi, horizon, annual_discount_rate),
    )
