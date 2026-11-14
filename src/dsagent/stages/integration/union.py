"""Schema mapping + vertical union of entity sources (Stage 4 steps 2-3, decision D2).

Each entity source is rewritten onto the canonical schema (spec feature names, spec
dtypes and units) using the feature_map found in Stage 2, then all sources are stacked.
Unmapped source columns are kept as `x_<name>` so the "all collected attributes"
baseline (experiment E4, variant A) remains possible; the relevance engine decides later.

Every row carries provenance columns:
  _row_ref  "<source_id>:<row index>"   _source_id   _method   _source_trust   _retrieved_at
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import polars as pl

from dsagent.config import IntegrationConfig
from dsagent.schemas.source import SourceCandidate
from dsagent.schemas.spec import DatasetSpec, Feature
from dsagent.stages.integration.normalise import ColumnReport, is_missing, normalise_column, parse_number
from dsagent.text import norm

PROVENANCE_COLS = ["_row_ref", "_source_id", "_method", "_source_trust", "_retrieved_at"]
_DTYPE = {"int": pl.Int64, "float": pl.Float64, "bool": pl.Boolean, "date": pl.Date,
          "category": pl.Utf8, "text": pl.Utf8}


@dataclass
class SourceTable:
    candidate: SourceCandidate
    df: pl.DataFrame
    column_units: dict[str, str] = field(default_factory=dict)  # declared units, e.g. {"Price": "EUR"}
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _method(kind: str) -> str:
    return {"api": "api", "html_table": "html_table", "jsonld": "jsonld"}.get(kind, "file")


def _extra(values: list) -> tuple[list, pl.DataType]:
    """Unmapped column: numeric if >= 90% of non-missing values parse as numbers."""
    present = [v for v in values if not is_missing(v)]
    nums = [parse_number(v)[0] for v in present]
    if present and sum(n is not None for n in nums) / len(present) >= 0.9:
        return [parse_number(v)[0] if not is_missing(v) else None for v in values], pl.Float64
    return [None if is_missing(v) else str(v).strip() for v in values], pl.Utf8


def to_canonical(
    spec: DatasetSpec, src: SourceTable, cfg: IntegrationConfig
) -> tuple[pl.DataFrame, list[ColumnReport]]:
    cand, df = src.candidate, src.df
    feats: list[Feature] = [spec.target, *spec.features]
    data: dict[str, pl.Series] = {}
    reports: list[ColumnReport] = []
    for f in feats:
        col = cand.feature_map.get(f.name)
        if col is None or col not in df.columns:
            data[f.name] = pl.Series(f.name, [None] * df.height, dtype=_DTYPE[f.dtype])
            continue
        values, rep = normalise_column(
            df[col].to_list(), f.dtype, f.unit, col, f.name, cfg, src.column_units.get(col)
        )
        reports.append(rep)
        data[f.name] = pl.Series(f.name, values, dtype=_DTYPE[f.dtype], strict=False)

    mapped = set(cand.feature_map.values())
    for col in df.columns:
        if col in mapped:
            continue
        name = f"x_{norm(col)}"
        if name in data:
            continue
        vals, dt = _extra(df[col].to_list())
        data[name] = pl.Series(name, vals, dtype=dt, strict=False)

    n = df.height
    data["_row_ref"] = pl.Series("_row_ref", [f"{cand.source_id}:{i}" for i in range(n)])
    data["_source_id"] = pl.Series("_source_id", [cand.source_id] * n)
    data["_method"] = pl.Series("_method", [_method(cand.kind)] * n)
    data["_source_trust"] = pl.Series("_source_trust", [cand.trust_score] * n, dtype=pl.Float64)
    data["_retrieved_at"] = pl.Series("_retrieved_at", [src.retrieved_at.isoformat()] * n)
    return pl.DataFrame(data), reports


def union_sources(
    spec: DatasetSpec, sources: list[SourceTable], cfg: IntegrationConfig
) -> tuple[pl.DataFrame, list[ColumnReport]]:
    frames, reports = [], []
    for s in sources:
        f, r = to_canonical(spec, s, cfg)
        frames.append(f)
        reports += r
    if not frames:
        raise ValueError("no entity sources to integrate")
    table = pl.concat(frames, how="diagonal_relaxed")
    feature_cols = [spec.target.name, *spec.feature_names()]
    extras = sorted(c for c in table.columns if c.startswith("x_"))
    return table.select(feature_cols + extras + PROVENANCE_COLS), reports


def source_unit_conflicts(reports: list[ColumnReport]) -> list[str]:
    """Same feature arriving in different units from different sources, unconverted."""
    by_feature: dict[str, set[Optional[str]]] = {}
    for r in reports:
        by_feature.setdefault(r.feature, set()).add(r.unit_to)
    return [f"{f}: sources disagree on unit {sorted(map(str, u))}" for f, u in by_feature.items() if len(u) > 1]
