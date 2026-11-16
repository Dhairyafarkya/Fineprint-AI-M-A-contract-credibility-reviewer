"""Confidence scorecard models (100-point rubric matrix)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CriterionScore(BaseModel):
    name: str
    max_points: float
    awarded: float
    rationale: str
    evidence: list[str] = Field(default_factory=list)

    @property
    def docked(self) -> float:
        return round(self.max_points - self.awarded, 2)


class CategoryScore(BaseModel):
    name: str
    max_points: float
    awarded: float
    criteria: list[CriterionScore]

    @property
    def confidence_pct(self) -> float:
        return 0.0 if self.max_points == 0 else round(100 * self.awarded / self.max_points, 1)


class ScoreCard(BaseModel):
    total: float
    rating: str
    verdict: str
    categories: list[CategoryScore]
