"""Experiment E1: feature-spec quality against expert-written feature lists.

A predicted feature matches a gold feature if its normalised name equals the gold
name or one of its aliases. Unmatched predictions are listed for manual review,
because a correct feature under an unexpected name counts as a false positive here
(the reported precision is therefore a lower bound).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from dsagent.schemas.spec import DatasetSpec


class GoldFeature(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)


class GoldSpec(BaseModel):
    id: str
    problem: str
    task_type: str
    target: GoldFeature
    features: list[GoldFeature]
    must_not_include: list[str] = Field(
        default_factory=list, description="Leakage/identifier columns that must not appear"
    )


class SpecScore(BaseModel):
    id: str
    task_type_correct: bool
    target_correct: bool
    precision: float
    recall: float
    f1: float
    matched: list[str]
    missed: list[str]
    unmatched_predictions: list[str]
    forbidden_included: list[str]


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _names(g: GoldFeature) -> set[str]:
    return {norm(g.name), *(norm(a) for a in g.aliases)}


# Words that carry no meaning for matching: counts, articles, and qualifiers like
# "nearest station" (distance_to_nearest_metro_station_km ~ distance_to_metro_km).
_FILLER = {"num", "number", "no", "of", "the", "to", "from", "total", "nearest", "station",
           "count", "is", "has", "in"}


def _tokens(name: str) -> frozenset[str]:
    return frozenset(t for t in norm(name).split("_") if t and t not in _FILLER)


def _matches(pred: str, g: GoldFeature) -> int:
    """0 = no match, 2 = exact name/alias, 1 = token match (every content word of
    some gold name/alias appears in the prediction, e.g. super_built_up_area_sqft
    contains built_up_area_sqft). Rule fixed on 2026-11-14 after the first smoke test
    showed exact matching under-counted correct but longer names."""
    if pred in _names(g):
        return 2
    pt = _tokens(pred)
    return 1 if any(_tokens(n) and _tokens(n) <= pt for n in _names(g)) else 0


def score_spec(spec: DatasetSpec, gold: GoldSpec) -> SpecScore:
    predicted = [norm(f.name) for f in spec.features]
    matched_gold: list[str] = []
    used_pred: set[str] = set()
    # Exact matches first, then token matches, so a loose match can't steal
    # a prediction that exactly matches another gold feature.
    for level in (2, 1):
        for g in gold.features:
            if g.name in matched_gold:
                continue
            hit = next(
                (p for p in predicted if p not in used_pred and _matches(p, g) == level), None
            )
            if hit:
                matched_gold.append(g.name)
                used_pred.add(hit)
    matched_gold.sort(key=[g.name for g in gold.features].index)

    tp = len(matched_gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold.features) if gold.features else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    forbidden = {norm(x) for x in gold.must_not_include}

    return SpecScore(
        id=gold.id,
        task_type_correct=spec.task_type == gold.task_type,
        target_correct=_matches(norm(spec.target.name), gold.target) > 0,
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        matched=matched_gold,
        missed=[g.name for g in gold.features if g.name not in matched_gold],
        unmatched_predictions=[p for p in predicted if p not in used_pred],
        forbidden_included=sorted(set(predicted) & forbidden),
    )


def load_gold(directory: Path) -> list[GoldSpec]:
    return [
        GoldSpec.model_validate(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(directory.glob("*.json"))
    ]
