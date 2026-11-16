"""Human-friendly number formatting shared by the UI and reports."""

from __future__ import annotations


def money(x: float | None, decimals: int = 1) -> str:
    if x is None:
        return "n/a"
    sign = "-" if x < 0 else ""
    v = abs(x)
    if v >= 1e9:
        return f"{sign}${v / 1e9:,.{decimals + 1}f}B"
    if v >= 1e6:
        return f"{sign}${v / 1e6:,.{decimals}f}M"
    if v >= 1e3:
        return f"{sign}${v / 1e3:,.{decimals}f}K"
    return f"{sign}${v:,.0f}"


def pct(x: float | None, decimals: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{decimals}f}%"


def month(m: int | None) -> str:
    return "never" if m is None else f"month {m}"


SEVERITY_ICON = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
