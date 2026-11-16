"""Structured outputs that Gemini must return.

Each model here is turned into a JSON schema and passed to Gemini as its
response schema, so the response is always machine-readable and validated by Pydantic.
Field descriptions double as instructions to the model, so keep them precise.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high", "critical"]

RISK_CATEGORIES = (
    "Termination & Break-up Fees",
    "Material Adverse Effect",
    "Hidden & Contingent Liabilities",
    "Revenue & Financial Metrics",
    "Debt & Change of Control",
    "Regulatory & Antitrust",
    "Indemnification & Escrow",
    "Deal Protection",
    "Financing",
    "Other",
)


# --------------------------------------------------------------------------
# Executive summary
# --------------------------------------------------------------------------
class KeyMetric(BaseModel):
    label: str = Field(description="Metric name, e.g. 'Enterprise value' or 'Per-share price'")
    value: str = Field(description="Value as written, with units, e.g. '$1.2 billion'")
    chunk_ids: list[str] = Field(default_factory=list, description="Chunk ids supporting this metric")


class Stakeholder(BaseModel):
    name: str
    role: str = Field(description="e.g. Acquirer, Target, Merger Sub, Financial advisor, Lender")


class TimelineEvent(BaseModel):
    event: str
    timing: str = Field(description="Date or period as stated, e.g. 'Outside Date: June 30, 2027'")


class SectionSummary(BaseModel):
    """Intermediate 'map' output used when summarising long documents."""

    summary: str = Field(description="Dense factual summary of this part of the agreement")
    key_points: list[str] = Field(default_factory=list)


class ExecutiveSummary(BaseModel):
    headline: str = Field(description="One-sentence description of the transaction")
    deal_type: str = Field(description="e.g. 'Reverse triangular merger', 'Asset purchase', 'Stock purchase'")
    summary: str = Field(description="Executive summary in Markdown, sized as instructed")
    key_metrics: list[KeyMetric] = Field(default_factory=list)
    stakeholders: list[Stakeholder] = Field(default_factory=list)
    valuation_terms: list[str] = Field(default_factory=list, description="Price, consideration mix, adjustments")
    timeline: list[TimelineEvent] = Field(default_factory=list)
    top_concerns: list[str] = Field(default_factory=list, description="Up to 5 issues leadership should know first")


# --------------------------------------------------------------------------
# Deal-term extraction (feeds the simulator and the scorecard)
# --------------------------------------------------------------------------
class Evidence(BaseModel):
    field_name: str = Field(description="Name of the extracted field this supports")
    chunk_id: str
    quote: str = Field(description="Short verbatim excerpt (under 40 words) from that chunk")


class DealTerms(BaseModel):
    """Facts extracted from the agreement. Use null when the document is silent.

    All money values are in US dollars (convert 'millions'/'billions' to full numbers).
    Percentages are decimals (3.5% -> 0.035). Durations are months from signing.
    """

    acquirer_name: str | None = None
    target_name: str | None = None
    deal_value_usd: float | None = Field(None, description="Total purchase price / equity value")
    cash_consideration_usd: float | None = None
    stock_consideration_usd: float | None = None
    per_share_price_usd: float | None = None
    termination_fee_usd: float | None = Field(None, description="Fee the TARGET pays the acquirer")
    reverse_termination_fee_usd: float | None = Field(None, description="Fee the ACQUIRER pays the target")
    escrow_usd: float | None = Field(None, description="Indemnity escrow / holdback amount")
    escrow_release_months: float | None = None
    earnout_max_usd: float | None = None
    earnout_period_months: float | None = None
    change_of_control_payments_usd: float | None = Field(
        None, description="Golden parachutes, retention, CoC bonuses, consent fees triggered by closing"
    )
    target_debt_usd: float | None = Field(None, description="Target indebtedness outstanding")
    debt_accelerates_on_change_of_control: bool | None = None
    new_acquisition_debt_usd: float | None = Field(None, description="New financing raised by the acquirer for the deal")
    acquisition_debt_interest_rate: float | None = None
    expected_annual_cost_synergies_usd: float | None = None
    expected_annual_revenue_synergies_usd: float | None = None
    integration_cost_usd: float | None = None
    expected_close_months: float | None = Field(None, description="Expected months from signing to closing")
    outside_date_months: float | None = Field(None, description="Months from signing to the outside/end date")
    target_debt_prepayment_premium: float | None = Field(None, description="Premium on accelerated repayment, decimal")
    integration_period_months: float | None = Field(None, description="Months over which integration costs are incurred")
    synergy_ramp_months: float | None = Field(None, description="Months after closing to reach full run-rate synergies")
    target_annual_revenue_usd: float | None = None
    target_annual_operating_costs_usd: float | None = None
    target_cash_usd: float | None = None
    acquirer_cash_usd: float | None = None
    acquirer_annual_revenue_usd: float | None = None
    acquirer_annual_operating_costs_usd: float | None = None
    transaction_costs_usd: float | None = Field(None, description="Acquirer's advisory, legal, financing and filing fees")
    regulatory_approvals_required: list[str] = Field(default_factory=list)
    hell_or_high_water: bool | None = Field(
        None, description="True if acquirer must accept any remedy/divestiture to get approval"
    )
    financing_condition: bool | None = Field(None, description="True if closing is conditioned on acquirer financing")
    mae_has_standard_carveouts: bool | None = Field(
        None,
        description="True if MAE definition excludes general economic, industry, law and pandemic changes "
        "with a disproportionate-effect qualifier",
    )
    no_shop_has_fiduciary_out: bool | None = None
    indemnification_cap_pct: float | None = Field(None, description="Indemnity cap as fraction of deal value")
    indemnification_survival_months: float | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list, description="Ambiguities or assumptions made")


# --------------------------------------------------------------------------
# Risk flags
# --------------------------------------------------------------------------
class RiskFlag(BaseModel):
    flag_id: str = ""
    category: str = Field(description="One of: " + "; ".join(RISK_CATEGORIES))
    title: str
    severity: Severity
    description: str = Field(description="What the clause does and why it matters to the acquirer")
    why_nonstandard: str = Field("", description="How it deviates from market norms for comparable deals")
    clause_excerpt: str = Field("", description="Short verbatim excerpt (under 60 words)")
    chunk_ids: list[str] = Field(default_factory=list)
    source: Literal["rule", "llm", "anomaly"] = "llm"
    confidence: float = Field(0.7, ge=0.0, le=1.0)
    estimated_exposure_usd: float | None = None


class RiskReview(BaseModel):
    """Gemini's review of one risk category."""

    flags: list[RiskFlag] = Field(default_factory=list)
    category_assessment: str = Field("", description="One or two sentences on this category overall")


# --------------------------------------------------------------------------
# Mitigation advice
# --------------------------------------------------------------------------
ActionType = Literal[
    "renegotiate_terms",
    "adjust_escrow",
    "price_adjustment",
    "structural_change",
    "additional_diligence",
    "integration_plan",
    "walk_away",
]


class Recommendation(BaseModel):
    title: str
    action_type: ActionType
    priority: Literal["P1", "P2", "P3"]
    addresses_flag_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(description="Corporate-finance reasoning, citing the numbers provided")
    proposed_terms: str = Field("", description="Concrete counter-proposal or clause language")
    estimated_financial_impact_usd: float | None = Field(
        None, description="Positive = value protected or recovered for the acquirer"
    )


class AdvisorReport(BaseModel):
    overall_stance: Literal["proceed", "proceed_with_conditions", "renegotiate", "walk_away"]
    stance_rationale: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    walk_away_triggers: list[str] = Field(default_factory=list)
    negotiation_sequence: list[str] = Field(default_factory=list, description="Order to raise the asks")


# --------------------------------------------------------------------------
# Question answering over the agreement
# --------------------------------------------------------------------------
class QAAnswer(BaseModel):
    answer: str = Field(description="Answer in Markdown. Cite chunks inline like [doc-c0007].")
    citations: list[str] = Field(default_factory=list, description="Chunk ids used")
    confidence: Literal["low", "medium", "high"] = "medium"
    not_found: bool = Field(False, description="True if the excerpts do not contain the answer")
