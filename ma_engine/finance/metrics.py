"""Liquidity and value metrics computed from a simulated cash path."""

from __future__ import annotations

import numpy as np

from ..models import ScenarioMetrics


def monthly_discount_factors(n: int, annual_rate: float) -> np.ndarray:
    monthly = (1 + annual_rate) ** (1 / 12) - 1
    return 1 / (1 + monthly) ** np.arange(1, n + 1)


def _first(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def compute_metrics(
    *,
    cash: np.ndarray,
    debt: np.ndarray,
    fcf: np.ndarray,
    synergy_net: np.ndarray,
    incremental_fcf: np.ndarray,
    upfront: float,
    covenant: float,
    annual_discount_rate: float,
) -> ScenarioMetrics:
    """`cash` and `debt` include month 0; flow arrays cover months 1..H."""
    disc = monthly_discount_factors(len(fcf), annual_discount_rate)
    below = cash < covenant
    cumulative_incremental = np.cumsum(incremental_fcf)
    payback = _first(cumulative_incremental >= upfront)
    return ScenarioMetrics(
        ending_cash=float(cash[-1]),
        minimum_cash=float(cash.min()),
        minimum_cash_month=int(cash.argmin()),
        months_below_covenant=int(below.sum()),
        first_covenant_breach_month=_first(below),
        first_negative_cash_month=_first(cash < 0),
        cumulative_free_cash_flow=float(fcf.sum()),
        npv_free_cash_flow=float((fcf * disc).sum()),
        npv_synergies=float((synergy_net * disc).sum()),
        payback_month=None if payback is None else payback + 1,
        peak_net_debt=float((debt - cash).max()),
        liquidity_ok=not bool(below.any()),
    )
