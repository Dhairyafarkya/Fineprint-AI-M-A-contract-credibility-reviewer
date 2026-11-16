"""The full analysis report produced by the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .documents import Document
from .finance import FinancialInputs, MitigationMath, MonteCarloResult, SimulationResult
from .llm_outputs import AdvisorReport, DealTerms, ExecutiveSummary, RiskFlag
from .scoring import ScoreCard


class UsageStats(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cached_responses: int = 0


class AnalysisReport(BaseModel):
    document: Document
    num_chunks: int
    summary: ExecutiveSummary | None = None
    deal_terms: DealTerms | None = None
    flags: list[RiskFlag] = Field(default_factory=list)
    financial_inputs: FinancialInputs | None = None
    simulations: list[SimulationResult] = Field(default_factory=list)
    monte_carlo: MonteCarloResult | None = None
    mitigation_math: MitigationMath | None = None
    scorecard: ScoreCard | None = None
    advice: AdvisorReport | None = None
    usage: UsageStats = Field(default_factory=UsageStats)
    warnings: list[str] = Field(default_factory=list)
    demo_mode: bool = False
