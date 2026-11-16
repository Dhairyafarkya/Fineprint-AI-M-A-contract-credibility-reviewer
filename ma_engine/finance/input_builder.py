"""Map extracted DealTerms (+ user overrides) onto simulator FinancialInputs.

Every field records where its value came from: 'extracted' (from the
document by Gemini), 'user' (typed into the dashboard) or 'default'.
"""

from __future__ import annotations

from typing import Any

from ..models import DealTerms, FinancialInputs

# FinancialInputs field  <-  DealTerms field
FIELD_MAP: dict[str, str] = {
    "acquirer_cash": "acquirer_cash_usd",
    "acquirer_annual_revenue": "acquirer_annual_revenue_usd",
    "acquirer_annual_operating_costs": "acquirer_annual_operating_costs_usd",
    "target_annual_revenue": "target_annual_revenue_usd",
    "target_annual_operating_costs": "target_annual_operating_costs_usd",
    "target_cash": "target_cash_usd",
    "cash_consideration": "cash_consideration_usd",
    "stock_consideration": "stock_consideration_usd",
    "transaction_costs": "transaction_costs_usd",
    "new_acquisition_debt": "new_acquisition_debt_usd",
    "acquisition_debt_interest_rate": "acquisition_debt_interest_rate",
    "target_debt": "target_debt_usd",
    "debt_accelerates_on_change_of_control": "debt_accelerates_on_change_of_control",
    "target_debt_prepayment_premium": "target_debt_prepayment_premium",
    "change_of_control_payments": "change_of_control_payments_usd",
    "integration_cost": "integration_cost_usd",
    "integration_months": "integration_period_months",
    "annual_cost_synergies": "expected_annual_cost_synergies_usd",
    "annual_revenue_synergies": "expected_annual_revenue_synergies_usd",
    "synergy_ramp_months": "synergy_ramp_months",
    "escrow_amount": "escrow_usd",
    "escrow_release_month": "escrow_release_months",
    "earnout_amount": "earnout_max_usd",
    "earnout_month": "earnout_period_months",
    "deal_value": "deal_value_usd",
}

# Fields that, when unknown, should default to zero rather than to an illustrative number.
ZERO_IF_UNKNOWN = {
    "stock_consideration", "target_debt", "change_of_control_payments", "earnout_amount",
    "escrow_amount", "new_acquisition_debt", "annual_cost_synergies", "annual_revenue_synergies",
    "integration_cost", "target_cash",
}

INT_FIELDS = {"integration_months", "synergy_ramp_months", "escrow_release_month", "earnout_month"}


def _coerce(name: str, value: Any) -> Any:
    if name in INT_FIELDS:
        return max(1, int(round(float(value))))
    return value


def _scale_estimates_to_deal(values: dict[str, Any], sources: dict[str, str]) -> None:
    """Size illustrative estimates to the deal when the agreement doesn't state them.

    Without this, a $4bn deal would be simulated against a $500M acquirer and look
    insolvent purely because of placeholder numbers. Ratios are deliberately neutral:
    target at ~4x revenue with a thin margin, acquirer ~4x the deal with a 15% margin,
    and closing cash of roughly a quarter of the deal value above the cash paid out.
    """
    deal = values.get("deal_value") or (values.get("cash_consideration", 0.0) + values.get("stock_consideration", 0.0))
    if not deal:
        return
    cash_out = values.get("cash_consideration", deal)
    estimates = {
        "acquirer_annual_revenue": 4.0 * deal,
        "acquirer_annual_operating_costs": 0.85 * 4.0 * deal,
        "target_annual_revenue": deal / 4.0,
        "target_annual_operating_costs": 0.9 * deal / 4.0,
        "acquirer_cash": max(0.0, cash_out - values.get("new_acquisition_debt", 0.0)) + 0.25 * deal,
        "minimum_cash_covenant": 0.10 * deal,
    }
    for name, value in estimates.items():
        if name not in values:
            values[name] = value
            sources[name] = "default"


def build_financial_inputs(terms: DealTerms | None, overrides: dict[str, Any] | None = None) -> tuple[FinancialInputs, list[str]]:
    """Return inputs plus human-readable warnings about defaulted values."""
    defaults = FinancialInputs()
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    terms = terms or DealTerms()
    has_any_deal_data = any(getattr(terms, f) is not None for f in FIELD_MAP.values())

    for fi_field, dt_field in FIELD_MAP.items():
        v = getattr(terms, dt_field)
        if v is not None:
            values[fi_field] = _coerce(fi_field, v)
            sources[fi_field] = "extracted"

    # Derive cash consideration from deal value if only the headline price is known.
    if "cash_consideration" not in values and terms.deal_value_usd:
        stock = terms.stock_consideration_usd or 0.0
        values["cash_consideration"] = max(0.0, terms.deal_value_usd - stock)
        sources["cash_consideration"] = "extracted"
    if "deal_value" not in values and ("cash_consideration" in values or "stock_consideration" in values):
        values["deal_value"] = values.get("cash_consideration", 0.0) + values.get("stock_consideration", 0.0)
        sources["deal_value"] = "extracted"

    if has_any_deal_data:
        for name in ZERO_IF_UNKNOWN:
            if name not in values:
                values[name] = 0.0
                sources[name] = "default"

    _scale_estimates_to_deal(values, sources)

    for name, v in (overrides or {}).items():
        if name in FinancialInputs.model_fields and name != "sources" and v is not None:
            values[name] = _coerce(name, v)
            sources[name] = "user"

    for name in FinancialInputs.model_fields:
        if name != "sources" and name not in sources:
            sources[name] = "default"

    fi = FinancialInputs(**values, sources=sources)
    if not fi.deal_value:
        fi.deal_value = fi.cash_consideration + fi.stock_consideration

    important = [
        "acquirer_cash", "acquirer_annual_revenue", "acquirer_annual_operating_costs",
        "target_annual_revenue", "target_annual_operating_costs", "cash_consideration", "minimum_cash_covenant",
    ]
    defaulted = [n for n in important if sources.get(n) == "default"]
    warnings = []
    if defaulted:
        warnings.append(
            "Illustrative defaults are in use for: " + ", ".join(defaulted)
            + ". They are sized to the deal; replace them in the Cash flow section."
        )
    return fi, warnings
