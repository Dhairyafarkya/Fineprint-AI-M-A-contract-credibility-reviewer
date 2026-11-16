"""Inputs and outputs of the deterministic cash-flow simulator.

All money is in US dollars. Annual figures are converted to monthly inside the
simulator. Rates are decimals (0.08 = 8%).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

InputSource = Literal["extracted", "user", "default"]


class FinancialInputs(BaseModel):
    # --- Acquirer stand-alone (normally supplied by the user) ---
    acquirer_cash: float = 500_000_000
    acquirer_annual_revenue: float = 1_200_000_000
    acquirer_annual_operating_costs: float = 1_020_000_000
    acquirer_revenue_growth: float = 0.06

    # --- Target stand-alone ---
    target_annual_revenue: float = 240_000_000
    target_annual_operating_costs: float = 225_000_000
    target_revenue_growth: float = 0.08
    target_cash: float = 30_000_000

    # --- Deal consideration & financing ---
    cash_consideration: float = 600_000_000
    stock_consideration: float = 0.0
    transaction_costs: float = 18_000_000
    new_acquisition_debt: float = 300_000_000
    acquisition_debt_interest_rate: float = 0.075
    acquisition_debt_amortization_years: float = 7.0

    # --- Target liabilities that come with the deal ---
    target_debt: float = 80_000_000
    target_debt_interest_rate: float = 0.065
    debt_accelerates_on_change_of_control: bool = True
    target_debt_prepayment_premium: float = 0.0
    change_of_control_payments: float = 12_000_000

    # --- Integration & synergies ---
    integration_cost: float = 45_000_000
    integration_months: int = 18
    annual_cost_synergies: float = 40_000_000
    annual_revenue_synergies: float = 25_000_000
    synergy_ramp_months: int = 24
    revenue_synergy_margin: float = Field(0.6, description="Share of revenue synergies that reaches EBITDA")

    # --- Contingent consideration ---
    escrow_amount: float = 60_000_000
    escrow_release_month: int = 18
    earnout_amount: float = 0.0
    earnout_month: int = 24
    earnout_probability: float = 0.5

    # --- Other ---
    tax_rate: float = 0.21
    capex_pct_revenue: float = 0.03
    minimum_cash_covenant: float = 100_000_000
    deal_value: float = 0.0

    sources: dict[str, InputSource] = Field(
        default_factory=dict, description="Where each field's value came from"
    )


class Scenario(BaseModel):
    """Stress applied on top of FinancialInputs."""

    name: str
    description: str = ""
    integration_delay_months: int = 0
    synergy_realization: float = 1.0  # 1.0 = synergies fully achieved
    integration_cost_overrun: float = 0.0  # 0.3 = 30% over budget
    hidden_liability: float = 0.0
    hidden_liability_month: int = 6
    hidden_liability_indemnifiable: float = Field(0.5, description="Share recoverable from escrow")
    earnout_paid: bool | None = Field(None, description="None = expected value; True/False = outcome")
    force_debt_acceleration: bool = False
    interest_rate_shock: float = 0.0  # +0.02 = +200 bps
    revenue_shock: float = 0.0  # -0.05 = combined revenue 5% lower
    extra_change_of_control_cost: float = 0.0


class ScenarioMetrics(BaseModel):
    ending_cash: float
    minimum_cash: float
    minimum_cash_month: int
    months_below_covenant: int
    first_covenant_breach_month: int | None
    first_negative_cash_month: int | None
    cumulative_free_cash_flow: float
    npv_free_cash_flow: float
    npv_synergies: float
    payback_month: int | None = Field(None, description="Month cumulative post-close FCF covers upfront cash outlay")
    peak_net_debt: float
    liquidity_ok: bool


class SimulationResult(BaseModel):
    scenario: Scenario
    months: list[int]
    cash: list[float]
    revenue: list[float]
    operating_costs: list[float]
    synergies: list[float]
    ebitda: list[float]
    free_cash_flow: list[float]
    debt_balance: list[float]
    one_off_outflows: list[float]
    upfront_outflow: float
    metrics: ScenarioMetrics


class MonteCarloResult(BaseModel):
    runs: int
    months: list[int]
    cash_p5: list[float]
    cash_p50: list[float]
    cash_p95: list[float]
    prob_covenant_breach: float
    prob_negative_cash: float
    ending_cash_p5: float
    ending_cash_p50: float
    ending_cash_p95: float
    expected_npv_fcf: float


class MitigationMath(BaseModel):
    """Deterministic numbers the advisor uses to size its recommendations."""

    recommended_escrow_usd: float
    escrow_gap_usd: float
    max_cash_price_for_covenant_usd: float
    price_cushion_usd: float = Field(description="Positive = room to pay more; negative = overpaying vs liquidity floor")
    synergy_npv_usd: float
    integration_cost_to_synergy_ratio: float | None = Field(None, description="None when no synergies")
    break_even_synergy_realization: float | None
