"""Offline stand-in for Gemini, used by tests and the dashboard's demo mode.

It never calls the network. Responses are either supplied by the caller
(``responses={ModelClass: instance_or_callable}``) or produced by simple,
clearly-labelled heuristics so every screen of the app has something to show.
Demo output is NOT analysis - it only exercises the plumbing.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from pydantic import BaseModel

from ..models import (
    AdvisorReport,
    DealTerms,
    Evidence,
    ExecutiveSummary,
    KeyMetric,
    QAAnswer,
    Recommendation,
    RiskReview,
    SectionSummary,
)
from .base import LLMClient, T

_EXCERPT = re.compile(r'<excerpt id="([^"]+)"[^>]*>\n(.*?)\n</excerpt>', re.S)
_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(billion|million|thousand)?", re.I)
_SCALE = {"billion": 1e9, "million": 1e6, "thousand": 1e3}

Responder = BaseModel | Callable[[str, str, str], BaseModel]


def _money(text: str) -> float | None:
    m = _MONEY.search(text)
    if not m:
        return None
    return float(m.group(1).replace(",", "")) * _SCALE.get((m.group(2) or "").lower(), 1.0)


def _find_amount(excerpts: list[tuple[str, str]], patterns: list[str], window: int = 160) -> tuple[float, str, str] | None:
    """First dollar amount appearing shortly after any of the keyword patterns."""
    for cid, text in excerpts:
        for pat in patterns:
            for m in re.finditer(pat, text, re.I):
                seg = text[m.start(): m.end() + window]
                val = _money(seg)
                if val:
                    return val, cid, re.sub(r"\s+", " ", seg)[:160]
    return None


class MockLLMClient(LLMClient):
    is_mock = True

    def __init__(self, responses: dict[type[BaseModel], Responder] | None = None):
        super().__init__()
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []

    def structured(self, *, system: str, prompt: str, output_model: type[T], context: str | None = None,
                   max_tokens: int | None = None, task: str = "") -> T:
        ctx = context or ""
        self.calls.append({"task": task, "model": output_model.__name__, "prompt": prompt, "context_chars": len(ctx)})
        self._record(inp=len(ctx) // 4 + len(prompt) // 4, out=200)
        if output_model in self.responses:
            r = self.responses[output_model]
            out = r(task, prompt, ctx) if callable(r) and not isinstance(r, BaseModel) else r
            return output_model.model_validate(out.model_dump() if isinstance(out, BaseModel) else out)
        handler = getattr(self, f"_mock_{output_model.__name__}", None)
        if handler is None:
            return output_model.model_validate({})
        return handler(task, prompt, ctx)

    # ------------------------------------------------------------- heuristics
    @staticmethod
    def _excerpts(ctx: str) -> list[tuple[str, str]]:
        return _EXCERPT.findall(ctx)

    def _mock_SectionSummary(self, task: str, prompt: str, ctx: str) -> SectionSummary:
        ex = self._excerpts(ctx)
        points = []
        for cid, t in ex[:6]:
            first_sentence = re.split(r"(?<=[.;])\s", t.strip())[0][:200]
            points.append(f"{first_sentence} [{cid}]")
        return SectionSummary(summary="[DEMO] " + " ".join(points[:2]), key_points=points)

    def _mock_ExecutiveSummary(self, task: str, prompt: str, ctx: str) -> ExecutiveSummary:
        ex = self._excerpts(ctx)
        amounts = []
        for cid, text in ex:
            for m in _MONEY.finditer(text):
                if (_money(m.group(0)) or 0) < 1e5:
                    continue
                amounts.append(KeyMetric(label="Amount mentioned", value=m.group(0).strip(), chunk_ids=[cid]))
                if len(amounts) >= 6:
                    break
        return ExecutiveSummary(
            headline="[Offline preview] Placeholder summary - configure GEMINI_API_KEY in utils.py for real analysis.",
            deal_type="Not analysed (demo)",
            summary=(
                "**Offline preview.** The engine ran end to end without calling Gemini. Clauses were chunked, "
                f"indexed and retrieved ({len(ex)} excerpts in context); rule-based risk checks, the "
                "cash-flow simulator, scorecard and charts below are fully functional. Summaries, "
                "LLM risk review and advice need a live API key."
            ),
            key_metrics=amounts,
        )

    def _mock_DealTerms(self, task: str, prompt: str, ctx: str) -> DealTerms:
        ex = self._excerpts(ctx)
        fields = {
            "deal_value_usd": [r"aggregate (?:equity|purchase|transaction) (?:value|price)", r"purchase price"],
            "termination_fee_usd": [r"Company Termination Fee", r"(?<!reverse )termination fee"],
            "reverse_termination_fee_usd": [r"Parent Termination Fee", r"reverse termination fee"],
            "escrow_usd": [r"Escrow Amount", r"escrow"],
            "target_debt_usd": [r"principal amount outstanding", r"Indebtedness"],
            "change_of_control_payments_usd": [r"aggregate cash payments", r"change of control payment"],
            "integration_cost_usd": [r"integration costs?"],
        }
        data: dict[str, Any] = {"notes": ["[DEMO] Values found by keyword heuristics, not by Gemini."], "evidence": []}
        for name, pats in fields.items():
            hit = _find_amount(ex, pats)
            if hit:
                data[name] = hit[0]
                data["evidence"].append(Evidence(field_name=name, chunk_id=hit[1], quote=hit[2]))
        return DealTerms.model_validate(data)

    def _mock_RiskReview(self, task: str, prompt: str, ctx: str) -> RiskReview:
        return RiskReview(flags=[], category_assessment="[DEMO] LLM review skipped; rule-based flags only.")

    def _mock_AdvisorReport(self, task: str, prompt: str, ctx: str) -> AdvisorReport:
        return AdvisorReport(
            overall_stance="proceed_with_conditions",
            stance_rationale="[Offline preview] Placeholder stance. Add GEMINI_API_KEY to utils.py for real recommendations.",
            recommendations=[
                Recommendation(
                    title="[DEMO] Resize indemnity escrow to the computed recommendation",
                    action_type="adjust_escrow", priority="P1",
                    rationale="Demo placeholder based on the deterministic mitigation math shown on this page.",
                )
            ],
            walk_away_triggers=["[DEMO] Example trigger"],
        )

    def _mock_QAAnswer(self, task: str, prompt: str, ctx: str) -> QAAnswer:
        ex = self._excerpts(ctx)
        if not ex:
            return QAAnswer(answer="[DEMO] No excerpts retrieved.", not_found=True, confidence="low")
        cid, text = ex[0]
        snippet = re.sub(r"\s+", " ", text)[:300]
        return QAAnswer(answer=f"[DEMO] Most relevant excerpt [{cid}]: {snippet}...", citations=[cid], confidence="low")
