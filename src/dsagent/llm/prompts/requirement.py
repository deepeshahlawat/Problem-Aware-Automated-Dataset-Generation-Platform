"""Prompt and output schema for Stage 1 (requirement analyser).

Bump TEMPLATE_VERSION whenever SYSTEM, the user template or SpecDraft changes:
it is part of the LLM call-cache key, so old cached answers are not reused.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from dsagent.schemas.spec import Feature, TaskType

TEMPLATE_VERSION = "requirement-v1"

Volatility = Literal["fast", "medium", "slow"]


class SpecDraft(BaseModel):
    """What the LLM returns. Converted to a DatasetSpec by the stage, which adds
    the problem text and maps `volatility` to a configured half-life."""

    task_type: TaskType
    target: Feature
    features: list[Feature] = Field(min_length=1)
    region: Optional[str] = Field(default=None, description="Geographic scope, if any")
    volatility: Volatility = Field(
        description="How fast the target's relationship to the features changes: "
        "fast (prices, ~2y), medium (~5y), slow (structural, ~15y)"
    )
    time_window_start: Optional[int] = Field(
        default=None, description="Only if the user explicitly asked for a year range"
    )
    time_window_end: Optional[int] = None
    min_rows: int = Field(default=1000, ge=50)
    excluded_candidates: list[str] = Field(
        default_factory=list,
        description="Columns deliberately left out, each as 'name: reason' "
        "(leakage, identifier, unavailable at prediction time)",
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Interpretations made where the problem was ambiguous"
    )
    needs_clarification: Optional[str] = Field(
        default=None,
        description="A question for the user if the problem is too vague to specify; else null",
    )


SYSTEM = """You are the requirement analyser of an automated dataset-building system. \
You turn a plain-English machine-learning problem into a feature specification for a \
tabular supervised-learning dataset that will be assembled from public sources.

Rules:
- Choose one target column and the task type (regression or classification; ranking only if asked).
- List the features a domain expert would want: usually 6-15. Prefer features that are \
plausibly available in public datasets, open APIs, or derivable from open data such as \
OpenStreetMap; mark rarely available ones required=false.
- Mark required=true only for features without which the model would be meaningless.
- Never include the target, anything computed from the target (e.g. price_per_sqft when \
predicting price), identifiers (listing IDs, URLs, usernames), or information that would \
not exist at prediction time. Put such tempting candidates in excluded_candidates with a reason.
- Names are lower snake_case. Give dtype, unit where relevant, a one-line description, \
and expected_range or allowed_values only when you are confident.
- Set region if the problem names a place. Set the time window only if the user asked for one.
- Record any interpretation you had to make in assumptions. If the problem is too vague \
to choose a target, fill needs_clarification with one short question."""


def render_user(problem_text: str, sample_columns: list[str] | None) -> str:
    parts = [f"Problem:\n{problem_text.strip()}"]
    if sample_columns:
        cols = ", ".join(sample_columns)
        parts.append(
            "The user also supplied a sample file with these columns. Reuse their meaning "
            f"where relevant, but rename to snake_case:\n{cols}"
        )
    return "\n\n".join(parts)
