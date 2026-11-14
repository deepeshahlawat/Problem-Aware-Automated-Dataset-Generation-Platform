"""Semantic relevance signal (design spec 16.1): one batched cheap-model call.

For each candidate column the model sees its name, type and a few real sample values,
plus the problem statement, and returns a relevance score and a role. Roles drive the
metadata / leakage overrides; the score enters the fusion. Answers for columns that do
not exist are ignored; columns the model skipped get the neutral score (reported).
"""

from __future__ import annotations

from typing import Literal

import polars as pl
from pydantic import BaseModel, Field

from dsagent.llm.client import LLMClient, LLMError, dumps_for_prompt
from dsagent.schemas.spec import DatasetSpec

TEMPLATE_VERSION = "semantic-relevance-v1"
Role = Literal["predictive", "identifier", "metadata", "target_derived", "future_info"]


class ColumnAssessment(BaseModel):
    column: str
    relevance: float = Field(ge=0, le=1, description="How useful for predicting the target, 0-1")
    role: Role
    reason: str


class SemanticAssessment(BaseModel):
    columns: list[ColumnAssessment]


SYSTEM = """You review the columns of a tabular dataset for a machine-learning problem.
For each column give:
- relevance: 0-1, how useful it is for predicting the target at prediction time,
  using domain knowledge (a column can be meaningful even if sparse);
- role: predictive | identifier (IDs, URLs, names of individual records) |
  metadata (scrape timestamps, page numbers, source names) |
  target_derived (computed from the target, e.g. price per square foot when predicting price) |
  future_info (only known after the outcome);
- reason: one short sentence.
Judge from the name, type and sample values. Return every listed column exactly once."""


def assess(
    table: pl.DataFrame, spec: DatasetSpec, columns: list[str], llm: LLMClient
) -> tuple[dict[str, ColumnAssessment], list[str]]:
    payload = {
        "problem": spec.problem_text,
        "target": {"name": spec.target.name, "description": spec.target.description},
        "columns": [
            {"name": c, "dtype": str(table[c].dtype),
             "samples": [str(v)[:40] for v in table[c].drop_nulls().unique(maintain_order=True).head(6).to_list()]}
            for c in columns
        ],
    }
    notes: list[str] = []
    try:
        res = llm.structured(role="cheap", system=SYSTEM, user=dumps_for_prompt(payload),
                             schema=SemanticAssessment, template_version=TEMPLATE_VERSION, max_tokens=8000)
    except LLMError as exc:
        return {}, [f"semantic check unavailable: {exc}"]
    out = {}
    for a in res.columns:
        if a.column in columns and a.column not in out:
            out[a.column] = a
    missing = [c for c in columns if c not in out]
    if missing:
        notes.append(f"semantic check returned nothing for {missing}; neutral score used")
    return out, notes
