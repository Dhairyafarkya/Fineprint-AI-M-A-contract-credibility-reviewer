"""Prescriptive mitigation advice from Gemini, grounded in the computed numbers."""

from __future__ import annotations

import json
import math

from ..llm.base import LLMClient
from ..llm.prompts import ADVISOR_PROMPT, SYSTEM_ANALYST
from ..models import (
    AdvisorReport,
    DealTerms,
    FinancialInputs,
    MitigationMath,
    MonteCarloResult,
    RiskFlag,
    ScoreCard,
    SimulationResult,
)


def _round(obj):
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return round(obj, 4) if abs(obj) < 10 else round(obj)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def build_advisor_context(
    terms: DealTerms, fi: FinancialInputs, flags: list[RiskFlag], card: ScoreCard,
    sims: list[SimulationResult], mc: MonteCarloResult | None, mm: MitigationMath | None,
    compact: bool = False,
) -> str:
    """JSON context for the advisor. `compact` trims it for small context windows (local models)."""
    if compact:
        flags = flags[:18]  # already sorted most severe first
        flags = [f.model_copy(update={"clause_excerpt": f.clause_excerpt[:220], "description": f.description[:280],
                                      "why_nonstandard": f.why_nonstandard[:200]}) for f in flags]
    payload = {
        "deal_terms": terms.model_dump(exclude={"evidence", "notes"}, exclude_none=True),
        "financial_inputs": fi.model_dump(exclude={"sources"}),
        "risk_flags": [
            f.model_dump(include={"flag_id", "category", "title", "severity", "description", "why_nonstandard",
                                  "clause_excerpt", "chunk_ids", "estimated_exposure_usd", "confidence"})
            for f in flags
        ],
        "scorecard": {
            "total": card.total, "rating": card.rating,
            "categories": [
                {"name": c.name, "awarded": c.awarded, "max": c.max_points,
                 "criteria": [{"name": x.name, "awarded": x.awarded, "max": x.max_points, "rationale": x.rationale}
                              for x in c.criteria]}
                for c in card.categories
            ],
        },
        "scenarios": [
            {"name": s.scenario.name, "description": s.scenario.description,
             "upfront_outflow": s.upfront_outflow, **s.metrics.model_dump()}
            for s in sims
        ],
        "monte_carlo": mc.model_dump(exclude={"months", "cash_p5", "cash_p50", "cash_p95"}) if mc else None,
        "mitigation_math": mm.model_dump() if mm else None,
    }
    return "<analysis_results>\n" + json.dumps(_round(payload), indent=1, default=str) + "\n</analysis_results>"


def advise(
    llm: LLMClient, terms: DealTerms, fi: FinancialInputs, flags: list[RiskFlag], card: ScoreCard,
    sims: list[SimulationResult], mc: MonteCarloResult | None, mm: MitigationMath | None,
    compact: bool = False,
) -> AdvisorReport:
    report = llm.structured(
        system=SYSTEM_ANALYST,
        prompt=ADVISOR_PROMPT,
        context=build_advisor_context(terms, fi, flags, card, sims, mc, mm, compact=compact),
        output_model=AdvisorReport,
        max_tokens=8000,
        task="advisor",
    )
    order = {"P1": 0, "P2": 1, "P3": 2}
    report.recommendations.sort(key=lambda r: order.get(r.priority, 9))
    return report
