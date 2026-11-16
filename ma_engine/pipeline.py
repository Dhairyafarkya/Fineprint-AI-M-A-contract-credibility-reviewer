"""End-to-end orchestration of the seven modules.

Where Gemini is (and isn't) used:

    1. Ingestion & RAG index ............ no LLM (cleaning, chunking, embeddings, BM25)
    2. Executive summary ................ Gemini (full text or map-reduce)
    3. Cash-flow simulation ............. no LLM (deterministic); inputs come from step 3a
       3a. Deal-term extraction ......... Gemini over RAG-retrieved excerpts
    4. Confidence scorecard ............. no LLM (transparent rubric over steps 3, 6)
    5. Charts ........................... no LLM
    6. Fine-print risk flags ............ rules + anomaly detection + Gemini per category (RAG)
    7. Mitigation advisor ............... Gemini, grounded in deterministic mitigation math
    +  Ask-the-agreement Q&A ............ Gemini over RAG
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analysis.advisor import advise
from .analysis.extractor import extract_deal_terms
from .analysis.qa import answer_question
from .analysis.risk_flagger import flag_risks
from .analysis.scoring import score_deal
from .analysis.summarizer import summarize_document
from .config import Settings, get_settings
from .finance.input_builder import build_financial_inputs
from .finance.mitigation_math import compute_mitigation_math
from .finance.monte_carlo import run_monte_carlo
from .finance.scenarios import default_scenarios, flagged_exposure
from .finance.simulator import simulate
from .ingestion.loader import RawDocument
from .llm.base import LLMClient, LLMError
from .models import AnalysisReport, DealTerms, QAAnswer, RetrievedChunk
from .retrieval.index_store import IndexStore
from .retrieval.indexer import build_index
from .retrieval.retriever import HybridRetriever

log = logging.getLogger(__name__)
Progress = Callable[[str, float], None]


@dataclass
class Workspace:
    """An index plus its retriever. One workspace may hold one or many documents."""

    index: IndexStore
    retriever: HybridRetriever

    @property
    def doc_ids(self) -> list[str]:
        return list(self.index.documents)


class MAEngine:
    def __init__(self, llm: LLMClient, settings: Settings | None = None):
        self.llm = llm
        self.settings = settings or get_settings()

    # ---------------------------------------------------------------- budgets
    @property
    def _local(self) -> bool:
        """True when the model is local (small context window): prompts are budgeted."""
        return self.settings.llm.is_local and not self.llm.is_mock

    def _budget_chars(self) -> int:
        return self.settings.llm.prompt_budget_chars() if self._local else 0

    # ------------------------------------------------------------ workspaces
    def build_workspace(self, raw_docs: list[RawDocument]) -> Workspace:
        index = build_index(raw_docs, self.settings)
        return Workspace(index, HybridRetriever(index, self.settings.retrieval))

    def open_workspace(self, directory: str | Path) -> Workspace:
        index = IndexStore.load(directory)
        return Workspace(index, HybridRetriever(index, self.settings.retrieval))

    # -------------------------------------------------------------- analysis
    def analyze(
        self,
        ws: Workspace,
        doc_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        progress: Progress | None = None,
    ) -> AnalysisReport:
        doc_id = doc_id or ws.doc_ids[0]
        doc = ws.index.documents[doc_id]
        chunks = ws.index.chunks_for(doc_id)
        report = AnalysisReport(document=doc, num_chunks=len(chunks), demo_mode=self.llm.is_mock)
        step = progress or (lambda name, frac: None)

        step("Writing executive summary", 0.05)
        try:
            report.summary = summarize_document(
                self.llm, doc, chunks, self.settings.summary, max_workers=self.settings.llm.max_parallel,
                retriever=ws.retriever, max_chars=self._budget_chars())
        except LLMError as exc:
            report.warnings.append(f"Executive summary failed: {exc}")

        step("Extracting deal terms", 0.25)
        try:
            report.deal_terms = extract_deal_terms(
                self.llm, ws.retriever, doc_id, full_text_max_words=self.settings.summary.single_pass_max_words,
                max_chunks=max(6, self.settings.llm.prompt_budget_words() // 500) if self._local else 24)
        except LLMError as exc:
            report.warnings.append(f"Deal-term extraction failed: {exc}. Using defaults.")
            report.deal_terms = DealTerms()

        step("Reviewing the fine print", 0.45)
        flags, _assessments, warns = flag_risks(
            self.llm, ws.retriever, doc_id, top_k=self.settings.retrieval.top_k,
            max_workers=self.settings.llm.max_parallel, max_chars=self._budget_chars(),
            on_progress=lambda done, total: step(f"Reviewing the fine print: {done} of {total} risk areas",
                                                 0.45 + 0.25 * done / total),
        )
        report.flags = flags
        report.warnings.extend(warns)

        step("Simulating cash flows", 0.7)
        self._run_financials(report, overrides)

        step("Drafting mitigation strategy", 0.85)
        self._run_advisor(report)

        report.usage = self.llm.usage.model_copy()
        step("Done", 1.0)
        return report

    def _run_financials(self, report: AnalysisReport, overrides: dict[str, Any] | None) -> None:
        sim_cfg = self.settings.simulation
        fi, warns = build_financial_inputs(report.deal_terms, overrides)
        report.financial_inputs = fi
        report.warnings = [w for w in report.warnings if not w.startswith("Illustrative defaults")] + warns
        report.simulations = [
            simulate(fi, sc, sim_cfg.horizon_months, sim_cfg.annual_discount_rate)
            for sc in default_scenarios(fi, report.flags)
        ]
        exposure = flagged_exposure(report.flags) or 0.05 * (fi.deal_value or fi.cash_consideration)
        report.monte_carlo = run_monte_carlo(
            fi, runs=sim_cfg.monte_carlo_runs, horizon=sim_cfg.horizon_months, seed=sim_cfg.random_seed,
            liability_exposure=exposure, annual_discount_rate=sim_cfg.annual_discount_rate,
        )
        report.mitigation_math = compute_mitigation_math(
            fi, report.simulations[0], report.flags, sim_cfg.horizon_months, sim_cfg.annual_discount_rate
        )
        report.scorecard = score_deal(
            report.deal_terms or DealTerms(), fi, report.flags, report.simulations,
            report.monte_carlo, report.mitigation_math, self.settings.weights,
        )

    def _run_advisor(self, report: AnalysisReport) -> None:
        try:
            report.advice = advise(
                self.llm, report.deal_terms or DealTerms(), report.financial_inputs, report.flags,
                report.scorecard, report.simulations, report.monte_carlo, report.mitigation_math,
                compact=self._local,
            )
        except LLMError as exc:
            report.warnings.append(f"Mitigation advisor failed: {exc}")

    def recompute_financials(
        self, report: AnalysisReport, overrides: dict[str, Any] | None, rerun_advisor: bool = False
    ) -> AnalysisReport:
        """Re-run simulation, scoring (and optionally advice) after the user edits inputs. No re-reading."""
        new = report.model_copy(deep=True)
        self._run_financials(new, overrides)
        if rerun_advisor:
            self._run_advisor(new)
        new.usage = self.llm.usage.model_copy()
        return new

    def refresh_advice(self, report: AnalysisReport) -> AnalysisReport:
        """Re-run only the mitigation advisor (after the user changed assumptions)."""
        new = report.model_copy(deep=True)
        new.warnings = [w for w in new.warnings if not w.startswith("Mitigation advisor failed")]
        self._run_advisor(new)
        new.usage = self.llm.usage.model_copy()
        return new

    # -------------------------------------------------------------------- Q&A
    def ask(self, ws: Workspace, question: str, doc_id: str | None = None) -> tuple[QAAnswer, list[RetrievedChunk]]:
        return answer_question(self.llm, ws.retriever, question, doc_id=doc_id,
                               k=self.settings.retrieval.top_k if self._local else 8, max_chars=self._budget_chars())
