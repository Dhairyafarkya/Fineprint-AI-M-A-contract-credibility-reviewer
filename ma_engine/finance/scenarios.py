"""Standard stress scenarios, sized from the deal and the risk flags found."""

from __future__ import annotations

from ..models import FinancialInputs, RiskFlag, Scenario

_SEVERE = {"high", "critical"}


def flagged_exposure(flags: list[RiskFlag]) -> float:
    """Sum of dollar exposures on high/critical flags (de-duplicated by rounded amount)."""
    seen: set[int] = set()
    total = 0.0
    for f in flags:
        if f.severity in _SEVERE and f.estimated_exposure_usd and f.estimated_exposure_usd > 0:
            key = round(f.estimated_exposure_usd / 1e5)
            if key not in seen:
                seen.add(key)
                total += f.estimated_exposure_usd
    return total


def default_scenarios(fi: FinancialInputs, flags: list[RiskFlag] | None = None) -> list[Scenario]:
    flags = flags or []
    deal = fi.deal_value or (fi.cash_consideration + fi.stock_consideration)
    liability = flagged_exposure(flags) or 0.05 * deal
    return [
        Scenario(name="Base Case", description="Deal terms as signed; synergies on plan."),
        Scenario(
            name="Delayed Integration",
            description="Integration starts 6 months late, 75% of synergies realised, 25% cost overrun.",
            integration_delay_months=6, synergy_realization=0.75, integration_cost_overrun=0.25,
        ),
        Scenario(
            name="Hidden Liabilities",
            description=f"Undisclosed/excluded liabilities of ${liability / 1e6:,.1f}M crystallise in month 6; "
            "only 30% recoverable from escrow.",
            hidden_liability=liability, hidden_liability_month=6, hidden_liability_indemnifiable=0.3,
        ),
        Scenario(
            name="Debt Acceleration + Rate Shock",
            description="Target debt accelerates at closing and rates rise 200 bps.",
            force_debt_acceleration=True, interest_rate_shock=0.02,
        ),
        Scenario(
            name="Severe Downside",
            description="9-month delay, 50% synergies, 50% integration overrun, hidden liabilities, "
            "debt acceleration, +300 bps, 8% revenue dip, earnout paid in full.",
            integration_delay_months=9, synergy_realization=0.5, integration_cost_overrun=0.5,
            hidden_liability=liability, hidden_liability_month=6, hidden_liability_indemnifiable=0.3,
            force_debt_acceleration=True, interest_rate_shock=0.03, revenue_shock=-0.08, earnout_paid=True,
        ),
    ]
