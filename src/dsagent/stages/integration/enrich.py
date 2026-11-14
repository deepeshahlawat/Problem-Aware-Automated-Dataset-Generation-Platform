"""Horizontal enrichment from context sources (Stage 4 step 5, decision D2).

Two join types, both exact and auditable (Retrieve-Merge-Predict found simple
exact-match joins the best cost/benefit option):

* key join      - context table keyed by a normalised value (e.g. locality -> circle-rate
                  category). An optional alias table maps spellings to the key.
                  Unmatched keys are reported, never fuzzily guessed.
* nearest point - for each row with coordinates, distance (km, haversine) to the nearest
                  point of a context table (e.g. metro stations). Marked method=derived.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import polars as pl

from dsagent.stages.integration.normalise import normalise_category


@dataclass
class EnrichReport:
    source_id: str
    kind: str
    features: list[str]
    rows_matched: int = 0
    rows_total: int = 0
    unmatched_keys: list[str] = field(default_factory=list)


def key_join(
    table: pl.DataFrame,
    context: pl.DataFrame,
    *,
    source_id: str,
    table_key: str,
    context_key: str,
    columns: dict[str, str],          # context column -> new feature name
    aliases: Optional[dict[str, str]] = None,  # spelling -> canonical key
) -> tuple[pl.DataFrame, EnrichReport]:
    alias = {normalise_category(k): normalise_category(v) for k, v in (aliases or {}).items()}
    ctx = context.with_columns(
        pl.col(context_key).map_elements(normalise_category, return_dtype=pl.Utf8).alias("__k")
    ).select(["__k", *columns]).unique(subset="__k", keep="first").rename(columns)
    keyed = table.with_columns(
        pl.col(table_key).map_elements(lambda v: alias.get(v, v), return_dtype=pl.Utf8).alias("__k")
    )
    out = keyed.join(ctx, on="__k", how="left", maintain_order="left").drop("__k")
    feats = list(columns.values())
    matched = out.filter(pl.col(feats[0]).is_not_null()).height
    known = set(ctx["__k"].to_list())
    unmatched = sorted({k for k in keyed["__k"].to_list() if k is not None and k not in known})
    return out, EnrichReport(source_id, "key_join", feats, matched, table.height, unmatched[:50])


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_distance(
    table: pl.DataFrame,
    points: pl.DataFrame,
    *,
    source_id: str,
    feature: str,
    lat: str = "latitude",
    lon: str = "longitude",
    point_lat: str = "lat",
    point_lon: str = "lon",
) -> tuple[pl.DataFrame, EnrichReport]:
    pts = [(a, b) for a, b in zip(points[point_lat].to_list(), points[point_lon].to_list())
           if a is not None and b is not None]
    dists = []
    for la, lo in zip(table[lat].to_list(), table[lon].to_list()):
        if la is None or lo is None or not pts:
            dists.append(None)
        else:
            dists.append(round(min(haversine_km(la, lo, a, b) for a, b in pts), 3))
    out = table.with_columns(pl.Series(feature, dists, dtype=pl.Float64))
    matched = sum(d is not None for d in dists)
    return out, EnrichReport(source_id, "nearest_point", [feature], matched, table.height)
