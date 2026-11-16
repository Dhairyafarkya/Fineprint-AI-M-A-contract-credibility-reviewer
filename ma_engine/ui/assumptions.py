"""Editable financial assumptions behind the cash-flow simulation."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ..models import FinancialInputs
from .components import note

# (label, field, kind) - kind: "M" = dollars shown in millions, "pct", "int", "bool"
GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Acquirer", [
        ("Cash on hand ($M)", "acquirer_cash", "M"),
        ("Annual revenue ($M)", "acquirer_annual_revenue", "M"),
        ("Annual operating costs ($M)", "acquirer_annual_operating_costs", "M"),
        ("Revenue growth (% a year)", "acquirer_revenue_growth", "pct"),
        ("Minimum cash covenant ($M)", "minimum_cash_covenant", "M"),
    ]),
    ("Target", [
        ("Annual revenue ($M)", "target_annual_revenue", "M"),
        ("Annual operating costs ($M)", "target_annual_operating_costs", "M"),
        ("Revenue growth (% a year)", "target_revenue_growth", "pct"),
        ("Cash acquired ($M)", "target_cash", "M"),
        ("Debt outstanding ($M)", "target_debt", "M"),
    ]),
    ("Price and financing", [
        ("Cash consideration ($M)", "cash_consideration", "M"),
        ("Stock consideration ($M)", "stock_consideration", "M"),
        ("Transaction costs ($M)", "transaction_costs", "M"),
        ("New acquisition debt ($M)", "new_acquisition_debt", "M"),
        ("Interest rate on new debt (%)", "acquisition_debt_interest_rate", "pct"),
    ]),
    ("Integration and synergies", [
        ("Integration cost ($M)", "integration_cost", "M"),
        ("Integration period (months)", "integration_months", "int"),
        ("Annual cost synergies ($M)", "annual_cost_synergies", "M"),
        ("Annual revenue synergies ($M)", "annual_revenue_synergies", "M"),
        ("Months to full synergies", "synergy_ramp_months", "int"),
    ]),
    ("Contingent payments", [
        ("Change-of-control payments ($M)", "change_of_control_payments", "M"),
        ("Escrow ($M)", "escrow_amount", "M"),
        ("Escrow claim window (months)", "escrow_release_month", "int"),
        ("Earnout maximum ($M)", "earnout_amount", "M"),
        ("Earnout probability (%)", "earnout_probability", "pct"),
    ]),
]
FLAGS = [("Target debt is repaid at closing (change of control)", "debt_accelerates_on_change_of_control")]
SOURCE_HELP = {"extracted": "Taken from the agreement.", "user": "Entered by you.",
               "default": "Illustrative estimate. Replace with your own figure."}


def _shown(fi: FinancialInputs, field: str, kind: str) -> Any:
    v = getattr(fi, field)
    return float(v) / 1e6 if kind == "M" else float(v) * 100 if kind == "pct" else v


def render(fi: FinancialInputs) -> dict[str, Any] | None:
    """Show the form; return overrides when the user submits, else None."""
    estimated = [f for _g, fields in GROUPS for _l, f, _k in fields if fi.sources.get(f, "default") == "default"]
    if estimated:
        note(f"{len(estimated)} figures are illustrative estimates (marked with *) because agreements rarely "
             "state them. Replace them with real numbers for a meaningful result.")
    with st.form("assumptions", border=False):
        values: dict[str, Any] = {}
        cols = st.columns(len(GROUPS))
        for col, (group, fields) in zip(cols, GROUPS):
            with col:
                st.markdown(f"**{group}**")
                for label, field, kind in fields:
                    src = fi.sources.get(field, "default")
                    shown_label = label + (" *" if src == "default" else "")
                    cur = _shown(fi, field, kind)
                    if kind == "int":
                        values[field] = st.number_input(shown_label, value=int(cur), min_value=0, step=1,
                                                        help=SOURCE_HELP[src], key=f"as_{field}")
                    else:
                        values[field] = st.number_input(shown_label, value=round(float(cur), 2), format="%.2f",
                                                        step=1.0 if kind == "M" else 0.5, help=SOURCE_HELP[src],
                                                        key=f"as_{field}")
        for label, field in FLAGS:
            values[field] = st.checkbox(label, value=bool(getattr(fi, field)), help=SOURCE_HELP[fi.sources.get(field, "default")],
                                        key=f"as_{field}")
        submitted = st.form_submit_button("Update simulation", type="primary")
    if not submitted:
        return None

    overrides: dict[str, Any] = {}
    kinds = {f: k for _g, fields in GROUPS for _l, f, k in fields} | {f: "bool" for _l, f in FLAGS}
    for field, new in values.items():
        kind = kinds[field]
        old = _shown(fi, field, kind) if kind != "bool" else bool(getattr(fi, field))
        changed = (new != old) if kind in ("bool", "int") else abs(float(new) - float(old)) > 1e-9
        if changed or fi.sources.get(field) == "user":
            overrides[field] = new * 1e6 if kind == "M" else new / 100 if kind == "pct" else new
    return overrides
