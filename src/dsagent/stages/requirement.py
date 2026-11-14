"""Stage 1: requirement analyser (design spec section 10, Stage 1).

problem text (+ optional sample columns) -> DatasetSpec, shown to the user for approval.

The LLM proposes a draft; deterministic guards then remove obvious target leakage and
identifier columns that slipped through, so the spec never depends on the LLM alone
for those rules. Everything removed is reported, never silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dsagent.config import Settings, get_settings
from dsagent.llm.client import LLMClient, LLMInvalidOutput
from dsagent.llm.prompts import requirement as prompt
from dsagent.schemas.spec import DatasetSpec, Feature, TemporalPolicy


class RequirementError(RuntimeError):
    """The problem could not be turned into a valid spec; ask the user to rephrase."""


class NeedsClarification(RequirementError):
    def __init__(self, question: str) -> None:
        super().__init__(question)
        self.question = question


@dataclass
class RequirementResult:
    spec: DatasetSpec
    volatility: str
    assumptions: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)  # LLM's own exclusions
    guard_removed: list[str] = field(default_factory=list)  # removed by deterministic guards


# Identifier-like names carry no predictive meaning (spec 16.1). Kept deliberately narrow:
# a false removal is reported, so the user can restore the column at approval time.
_ID_PATTERN = re.compile(
    r"(^id$|_id$|^id_|uuid|url|link|username|user_name|email|phone|listing_no|serial)"
)


def _guard_reason(feature: Feature, target: Feature) -> str | None:
    name, t = feature.name, target.name
    if name == t:
        return "same as target"
    tokens = name.split("_")
    if t in tokens or name.startswith(f"{t}_") or name.endswith(f"_{t}"):
        if "per" in tokens or "log" in tokens or name.startswith(f"{t}_"):
            return f"derived from target '{t}' (leakage)"
    if _ID_PATTERN.search(name):
        return "identifier-like column"
    return None


def _temporal(draft: prompt.SpecDraft, settings: Settings) -> TemporalPolicy:
    half_life = {
        "fast": settings.recency.half_life_fast,
        "medium": settings.recency.half_life_medium,
        "slow": settings.recency.half_life_slow,
    }[draft.volatility]
    if draft.time_window_start is not None and draft.time_window_end is not None:
        return TemporalPolicy(
            mode="window",
            half_life_years=half_life,
            window_start=draft.time_window_start,
            window_end=draft.time_window_end,
        )
    return TemporalPolicy(mode="recent_weighted", half_life_years=half_life)


def analyse_problem(
    problem_text: str,
    llm: LLMClient,
    sample_columns: list[str] | None = None,
    settings: Settings | None = None,
) -> RequirementResult:
    settings = settings or get_settings()
    if len(problem_text.strip()) < 10:
        raise NeedsClarification("Please describe what you want to predict, and for where/when.")

    try:
        draft = llm.structured(
            role="planner",
            system=prompt.SYSTEM,
            user=prompt.render_user(problem_text, sample_columns),
            schema=prompt.SpecDraft,
            template_version=prompt.TEMPLATE_VERSION,
        )
    except LLMInvalidOutput as exc:
        raise RequirementError(
            "Could not produce a valid specification; please rephrase the problem."
        ) from exc

    if draft.needs_clarification:
        raise NeedsClarification(draft.needs_clarification)

    kept: list[Feature] = []
    removed: list[str] = []
    seen: set[str] = set()
    for f in draft.features:
        reason = _guard_reason(f, draft.target)
        if reason:
            removed.append(f"{f.name}: {reason}")
        elif f.name in seen:
            removed.append(f"{f.name}: duplicate")
        else:
            seen.add(f.name)
            kept.append(f)

    if not kept:
        raise RequirementError("No usable features remained after leakage/identifier checks.")

    spec = DatasetSpec(
        problem_text=problem_text.strip(),
        task_type=draft.task_type,
        target=draft.target,
        features=kept,
        region=draft.region,
        temporal=_temporal(draft, settings),
        min_rows=draft.min_rows,
    )
    return RequirementResult(
        spec=spec,
        volatility=draft.volatility,
        assumptions=draft.assumptions,
        excluded=draft.excluded_candidates,
        guard_removed=removed,
    )
