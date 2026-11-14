"""Stage 6: recency weighting (design spec section 14).

apply_recency(table, spec, date_column=None, source_dates=None, now=None) -> (table, report)

Observation date per row, in order of preference:
  1. a date column in the table (e.g. listing date), if given
  2. the source's last-updated date (from Stage 2)
  3. unknown -> neutral weight 1.0 (counted in the report, never guessed)

Policy (TemporalPolicy.mode):
  recent_weighted  weight = max(min_weight, 0.5 ** (age_years / half_life))
  window           rows outside [window_start, window_end] removed (reported); weight 1
  uniform          weight 1 for every row
sample_weight = recency_weight * row_confidence  (passed to LightGBM, exported in the dataset)
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

import polars as pl
from pydantic import BaseModel

from dsagent.config import Settings, get_settings
from dsagent.schemas.spec import DatasetSpec
from dsagent.trust.recency import age_years, recency_weight


class RecencyReport(BaseModel):
    mode: str
    half_life_years: float
    rows_in: int
    rows_out: int
    dated_from_column: int
    dated_from_source: int
    undated: int
    removed_outside_window: int = 0
    mean_weight: Optional[float] = None


def _to_dt(v) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day, tzinfo=timezone.utc)
    return None


def apply_recency(
    table: pl.DataFrame,
    spec: DatasetSpec,
    date_column: Optional[str] = None,
    source_dates: Optional[dict[str, datetime]] = None,
    settings: Optional[Settings] = None,
    now: Optional[datetime] = None,
) -> tuple[pl.DataFrame, RecencyReport]:
    settings = settings or get_settings()
    pol = spec.temporal
    n = table.height
    col_vals = table[date_column].to_list() if date_column and date_column in table.columns else [None] * n
    src_ids = table["_source_id"].to_list() if "_source_id" in table.columns else [None] * n
    source_dates = source_dates or {}

    obs, from_col, from_src = [], 0, 0
    for v, s in zip(col_vals, src_ids):
        d = _to_dt(v)
        if d is not None:
            from_col += 1
        else:
            d = _to_dt(source_dates.get(s)) if s else None
            from_src += d is not None
        obs.append(d)

    if pol.mode == "recent_weighted":
        w = [1.0 if d is None else recency_weight(age_years(d, now), pol.half_life_years,
                                                  settings.recency.min_weight) for d in obs]
    else:
        w = [1.0] * n

    out = table.with_columns(
        pl.Series("_obs_date", [d.date() if d else None for d in obs], dtype=pl.Date),
        pl.Series("_recency_weight", [round(x, 6) for x in w], dtype=pl.Float64),
    )
    conf = pl.col("_row_confidence") if "_row_confidence" in out.columns else pl.lit(1.0)
    out = out.with_columns((pl.col("_recency_weight") * conf).alias("_sample_weight"))

    removed = 0
    if pol.mode == "window":
        keep = pl.col("_obs_date").dt.year().is_between(pol.window_start, pol.window_end)
        # Undated rows cannot be shown to be inside the window: they are removed and counted.
        before = out.height
        out = out.filter(keep.fill_null(False))
        removed = before - out.height

    rep = RecencyReport(
        mode=pol.mode, half_life_years=pol.half_life_years, rows_in=n, rows_out=out.height,
        dated_from_column=from_col, dated_from_source=from_src, undated=n - from_col - from_src,
        removed_outside_window=removed,
        mean_weight=round(out["_recency_weight"].mean(), 4) if out.height else None,
    )
    return out, rep
