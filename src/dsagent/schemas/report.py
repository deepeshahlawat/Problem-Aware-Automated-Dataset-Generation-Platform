"""Feasibility, relevance and run-summary contracts (README sections 8, 15, 16, 17)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

RelevanceTier = Literal["high", "medium", "low", "metadata", "leakage", "dropped"]


class FeasibilityItem(BaseModel):
    feature: str
    status: Literal["found", "partial", "derivable", "not_found"]
    sources: list[str]
    coverage: float = Field(ge=0, le=1, description="Fraction of rows with a value")
    newest_date: Optional[datetime] = None
    trust: Optional[float] = Field(default=None, ge=0, le=1)
    note: str = ""


class ColumnRelevance(BaseModel):
    column: str
    stat_score: float = Field(ge=0, le=1)
    model_score: float = Field(ge=0, le=1)
    semantic_score: float = Field(ge=0, le=1)
    fused_score: float = Field(ge=0, le=1)
    tier: RelevanceTier
    reason: str


class RunSummary(BaseModel):
    job_id: str
    status: Literal["running", "succeeded", "failed", "aborted_budget", "needs_review"]
    started_at: datetime
    finished_at: Optional[datetime] = None
    llm_calls: int = 0
    llm_cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    pages_fetched: int = 0
    notes: list[str] = Field(default_factory=list)
