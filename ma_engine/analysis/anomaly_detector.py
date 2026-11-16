"""Numeric anomaly detection across the document.

Finds the same financial metric quoted with materially different values, and
non-GAAP metrics that run far ahead of reported GAAP revenue - typical signs of
overstated or cherry-picked numbers.
"""

from __future__ import annotations

import re
from collections import defaultdict

from ..models import Chunk, RiskFlag

_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(billion|million|thousand|bn|mm|m)?\b", re.I)
_SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "mm": 1e6, "m": 1e6, "thousand": 1e3}

METRICS: dict[str, str] = {
    "ARR": r"\b(?:ARR|annual recurring revenue)\b",
    "Adjusted EBITDA": r"\badjusted ebitda\b",
    "EBITDA": r"(?<!adjusted )\bebitda\b",
    "Bookings": r"\bbookings\b",
    "Backlog": r"\bbacklog\b",
}
_GAAP_REVENUE = re.compile(r"\b(?:total|recorded|net)\s+revenue(?:s)?\s+of\s+\$", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")


def _amount(m: re.Match) -> float:
    return float(m.group(1).replace(",", "")) * _SCALE.get((m.group(2) or "").lower(), 1.0)


def _metric_mentions(chunks: list[Chunk]) -> dict[str, list[tuple[float, str, str]]]:
    """metric -> [(value, chunk_id, sentence)] using the first $ amount after the metric in a sentence."""
    found: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
    seen: set[tuple[str, float]] = set()
    for c in chunks:
        for sentence in _SENTENCE_SPLIT.split(c.text):
            if re.search(r"synerg|target|threshold|equals or exceeds", sentence, re.I):
                continue
            for metric, pat in METRICS.items():
                mm = re.search(pat, sentence, re.I)
                if not mm:
                    continue
                money = _MONEY.search(sentence, mm.end())
                if money and money.start() - mm.end() < 120:
                    val = _amount(money)
                    if (metric, val) not in seen:
                        seen.add((metric, val))
                        found[metric].append((val, c.chunk_id, re.sub(r"\s+", " ", sentence)[:300]))
    return found


def _gaap_revenue(chunks: list[Chunk]) -> tuple[float, str] | None:
    for c in chunks:
        m = _GAAP_REVENUE.search(c.text)
        if m:
            money = _MONEY.search(c.text, m.end() - 1)
            if money:
                return _amount(money), c.chunk_id
    return None


def detect_anomalies(chunks: list[Chunk], tolerance: float = 0.10) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    mentions = _metric_mentions(chunks)
    for metric, vals in mentions.items():
        if len(vals) < 2:
            continue
        lo, hi = min(vals, key=lambda v: v[0]), max(vals, key=lambda v: v[0])
        if lo[0] > 0 and (hi[0] - lo[0]) / lo[0] > tolerance:
            spread = (hi[0] - lo[0]) / lo[0]
            flags.append(RiskFlag(
                flag_id=f"ANOMALY_{metric.upper().replace(' ', '_')}",
                category="Revenue & Financial Metrics",
                title=f"Inconsistent {metric} figures ({spread:.0%} apart)",
                severity="high" if spread > 0.2 else "medium",
                description=f"{metric} is stated as ${lo[0] / 1e6:,.1f}M and ${hi[0] / 1e6:,.1f}M in different places. "
                "Reconcile the dates and definitions before relying on either number.",
                why_nonstandard="The same KPI should reconcile across the agreement and its schedules.",
                clause_excerpt=f"{lo[2]} ... {hi[2]}"[:600],
                chunk_ids=sorted({lo[1], hi[1]}),
                source="anomaly",
                confidence=0.75,
                estimated_exposure_usd=None,
            ))
    gaap = _gaap_revenue(chunks)
    if gaap and mentions.get("ARR"):
        top = max(mentions["ARR"], key=lambda v: v[0])
        ratio = top[0] / gaap[0] if gaap[0] else 0
        if ratio > 1.3:
            flags.append(RiskFlag(
                flag_id="ANOMALY_ARR_VS_GAAP",
                category="Revenue & Financial Metrics",
                title=f"ARR is {ratio:.1f}x reported GAAP revenue",
                severity="medium",
                description="Recurring revenue far above recognised revenue suggests ARR counts contracts not yet earning.",
                why_nonstandard="For a subscription business ARR and trailing revenue are usually close.",
                clause_excerpt=top[2],
                chunk_ids=sorted({top[1], gaap[1]}),
                source="anomaly",
                confidence=0.6,
            ))
    return flags
