"""Map spec features to a source's real columns (decision D3, discovery-time use).

1. Deterministic name matching on normalised tokens (one-to-one, best score first).
   Ties are left unresolved rather than guessed.
2. Optional: one batched cheap-model call for the leftovers, shown column names plus
   a few real sample values. Answers naming a column that does not exist are dropped.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from dsagent.llm.client import LLMClient, LLMError, dumps_for_prompt
from dsagent.schemas.spec import DatasetSpec, Feature
from dsagent.text import jaccard, tokens

TEMPLATE_VERSION = "column-match-v1"
_MISSING = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?"}


class ColumnMatch(BaseModel):
    feature: str
    column: Optional[str] = Field(description="Exact column name from the list, or null")
    reason: str


class ColumnMatches(BaseModel):
    matches: list[ColumnMatch]


SYSTEM = """You match the columns of a data source to the features of a dataset specification.
For each listed feature, return the single source column that holds the same quantity, \
or null if none does. Use the sample values to check meaning and units. A column that is \
derived from another quantity (e.g. price per square foot) does NOT match the base \
quantity (price). Only use column names exactly as listed."""


def name_score(feature: str, column: str) -> float:
    f, c = tokens(feature), tokens(column)
    if not f or not c:
        return 0.0
    if f == c:
        return 1.0
    if c <= f or f <= c:
        return 0.5 + 0.5 * jaccard(f, c)
    return 0.0


def match_by_name(features: list[str], columns: list[str]) -> dict[str, str]:
    def score(f: str, c: str) -> float:
        s = name_score(f, c)
        if 0 < s < 1:
            # A generic column ('Area') contained in several features
            # (built_up_area_sqft, plot_area_sqft) is ambiguous: don't guess.
            ct = tokens(c)
            if sum(1 for g in features if ct <= tokens(g)) > 1:
                return 0.0
        return s

    pairs = sorted(
        ((score(f, c), f, c) for f in features for c in columns),
        key=lambda x: -x[0],
    )
    best_for_col: dict[str, list[float]] = {}
    for s, f, c in pairs:
        if s > 0:
            best_for_col.setdefault(c, []).append(s)
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for s, f, c in pairs:
        if s <= 0 or f in mapping or c in used:
            continue
        scores = best_for_col[c]
        if len(scores) > 1 and scores[0] == scores[1]:
            continue  # tie: ambiguous, leave to the LLM / user
        mapping[f] = c
        used.add(c)
    return mapping


def coverage(rows: list[dict[str, Any]], column: str) -> float:
    if not rows:
        return 0.0
    ok = sum(1 for r in rows if r.get(column) is not None and str(r.get(column)).strip().lower() not in _MISSING)
    return round(ok / len(rows), 4)


def _samples(rows: list[dict[str, Any]], column: str, k: int = 6) -> list[str]:
    vals: list[str] = []
    for r in rows:
        v = r.get(column)
        if v is not None and str(v).strip() and str(v) not in vals:
            vals.append(str(v)[:40])
        if len(vals) >= k:
            break
    return vals


def match_columns(
    spec: DatasetSpec,
    columns: list[str],
    rows: list[dict[str, Any]],
    llm: LLMClient | None = None,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Returns (feature_map, method_by_feature, notes). Includes the target."""
    feats: list[Feature] = [spec.target, *spec.features]
    names = [f.name for f in feats]
    mapping = match_by_name(names, columns)
    method = {f: "name" for f in mapping}
    notes: list[str] = []

    left_f = [f for f in feats if f.name not in mapping]
    left_c = [c for c in columns if c not in mapping.values()]
    if llm is not None and left_f and left_c:
        payload = {
            "features": [{"name": f.name, "dtype": f.dtype, "unit": f.unit, "description": f.description} for f in left_f],
            "columns": [{"name": c, "samples": _samples(rows, c)} for c in left_c],
        }
        try:
            res = llm.structured(
                role="cheap",
                system=SYSTEM,
                user=dumps_for_prompt(payload),
                schema=ColumnMatches,
                template_version=TEMPLATE_VERSION,
                max_tokens=4000,
            )
        except LLMError as exc:
            notes.append(f"LLM column matching skipped: {exc}")
        else:
            valid_f = {f.name for f in left_f}
            used = set(mapping.values())
            # Tolerate whitespace/case slips, but only resolve to a column that exists.
            by_key = {c.strip().lower(): c for c in left_c}
            for m in res.matches:
                if m.column is None or m.feature not in valid_f or m.feature in mapping:
                    continue
                col = by_key.get(m.column.strip().lower())
                if col is None or col in used:
                    notes.append(f"ignored LLM match {m.feature}->{m.column!r} (not an available column)")
                    continue
                mapping[m.feature] = col
                method[m.feature] = "llm"
                used.add(col)
    return mapping, method, notes
