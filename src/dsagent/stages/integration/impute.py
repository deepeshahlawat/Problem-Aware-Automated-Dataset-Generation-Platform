"""Per-column missing-value handling (Stage 4 step 6; design principle 3).

Stage 4 only *plans* the strategy per column. Statistics are fitted later on the
training split alone (`apply_plan(df, plan, fit_on=train_df)`), because imputing
with full-table medians would leak test information into training.

Rules (documented in the data card, never one global rule):
  target                  -> never imputed; rows with a missing target are dropped
  0% missing              -> none
  > max_impute_share      -> leave_null (too sparse to fill honestly)
  numeric, group column   -> group_median (e.g. median area within the same locality)
  numeric                 -> median
  category / text         -> explicit '__missing__' category (no invented value)
  bool / date             -> leave_null
Every filled cell gets a companion flag column `<col>__imputed`.
"""

from __future__ import annotations

from typing import Literal, Optional

import polars as pl
from pydantic import BaseModel

from dsagent.config import IntegrationConfig
from dsagent.schemas.spec import DatasetSpec

Strategy = Literal["none", "drop_rows", "median", "group_median", "missing_category", "leave_null"]


class ColumnPlan(BaseModel):
    column: str
    missing_share: float
    strategy: Strategy
    group_by: Optional[str] = None
    reason: str


def plan_imputation(
    table: pl.DataFrame, spec: DatasetSpec, cfg: IntegrationConfig, group_by: Optional[str] = None
) -> list[ColumnPlan]:
    dtypes = {spec.target.name: spec.target.dtype, **{f.name: f.dtype for f in spec.features}}
    n = max(table.height, 1)
    plans = []
    for col in [c for c in table.columns if not c.startswith("_")]:
        share = round(table[col].null_count() / n, 4)
        dt = dtypes.get(col) or ("float" if table[col].dtype.is_numeric() else "category")
        if col == spec.target.name:
            s, why = ("drop_rows", "target is never imputed") if share else ("none", "complete")
        elif share == 0:
            s, why = "none", "complete"
        elif share > cfg.max_impute_share:
            s, why = "leave_null", f"{share:.0%} missing exceeds {cfg.max_impute_share:.0%}; filling would invent most values"
        elif dt in ("int", "float"):
            if group_by and group_by in table.columns and group_by != col:
                s, why = "group_median", f"median within the same {group_by}, overall median as fallback"
            else:
                s, why = "median", "numeric; median is robust to outliers"
        elif dt in ("category", "text"):
            s, why = "missing_category", "explicit missing category instead of a guessed value"
        else:
            s, why = "leave_null", f"{dt} values are not imputed"
        plans.append(ColumnPlan(column=col, missing_share=share, strategy=s,
                                group_by=group_by if s == "group_median" else None, reason=why))
    return plans


def apply_plan(
    df: pl.DataFrame, plans: list[ColumnPlan], cfg: IntegrationConfig, fit_on: Optional[pl.DataFrame] = None
) -> pl.DataFrame:
    """Apply a plan; statistics come from `fit_on` (the training split), default `df`."""
    fit = df if fit_on is None else fit_on
    out = df
    for p in plans:
        c = p.column
        if p.strategy == "drop_rows":
            out = out.filter(pl.col(c).is_not_null())
            continue
        if p.strategy in ("none", "leave_null"):
            continue
        flag = f"{c}__imputed"
        out = out.with_columns(pl.col(c).is_null().alias(flag))
        if p.strategy == "median":
            out = out.with_columns(pl.col(c).fill_null(fit[c].median()))
        elif p.strategy == "group_median":
            g = p.group_by
            med = fit.group_by(g).agg(pl.col(c).median().alias("__m"))
            out = (out.join(med, on=g, how="left", maintain_order="left")
                   .with_columns(pl.col(c).fill_null(pl.col("__m")).fill_null(fit[c].median()))
                   .drop("__m"))
        elif p.strategy == "missing_category":
            out = out.with_columns(pl.col(c).fill_null(cfg.missing_category))
    return out
