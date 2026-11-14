"""Target-leakage detection (design spec 16.3), on training rows only.

Three independent rules produce candidates:
  name     the column is named as a transform of the target (price_per_sqft, log_price)
  single   a model on this column alone predicts the target near-perfectly
           (R^2 >= 0.98 or AUC >= 0.99)
  pair     log(target) = a*log(col) + b*log(other) with R^2 >= 0.98 and |a|, |b| ~ 1:
           the target is (almost exactly) col x other or col / other. This is what catches
           Per_Sqft = Price / Area, which no single-column test can see.
The semantic check (LLM) later confirms or clears 'single' candidates (spec 16.3);
'name' and 'pair' findings are structural and stand on their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import polars as pl

from dsagent.config import Settings
from dsagent.schemas.spec import DatasetSpec
from dsagent.stages.relevance.signals import ColumnStats
from dsagent.text import norm


@dataclass
class LeakFinding:
    column: str
    rule: str  # name | single | pair
    detail: str


def _name_leak(col: str, target: str) -> bool:
    toks = norm(col.removeprefix("x_")).split("_")
    t = norm(target).split("_")
    has_target = all(x in toks for x in t)
    return has_target and ("per" in toks or "log" in toks or norm(col.removeprefix("x_")).startswith(norm(target) + "_"))


def find_leakage(
    train: pl.DataFrame, spec: DatasetSpec, stats: dict[str, ColumnStats], numeric_cols: list[str],
    settings: Settings,
) -> list[LeakFinding]:
    rc = settings.relevance
    target = spec.target.name
    out: list[LeakFinding] = []

    for c, s in stats.items():
        if _name_leak(c, target):
            out.append(LeakFinding(c, "name", f"named as a transform of '{target}'"))
        thr = rc.leakage_r2 if spec.task_type == "regression" else rc.leakage_auc
        if np.isfinite(s.single_score) and s.single_score >= thr:
            out.append(LeakFinding(c, "single", f"alone predicts the target (score {s.single_score:.3f})"))

    if spec.task_type != "regression":
        return out
    y = train[target].cast(pl.Float64, strict=False).to_numpy()
    feats = {c: train[c].cast(pl.Float64, strict=False).to_numpy() for c in numeric_cols if c in train.columns}
    spec_features = set(spec.feature_names())
    for a, b in combinations(sorted(feats), 2):
        xa, xb = feats[a], feats[b]
        ok = (y > 0) & (xa > 0) & (xb > 0) & np.isfinite(y) & np.isfinite(xa) & np.isfinite(xb)
        if ok.sum() < 20 or ok.mean() < 0.5:
            continue
        A = np.column_stack([np.log(xa[ok]), np.log(xb[ok]), np.ones(ok.sum())])
        ly = np.log(y[ok])
        coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
        resid = ly - A @ coef
        ss_tot = ((ly - ly.mean()) ** 2).sum()
        r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
        if r2 >= rc.pair_leakage_r2 and all(0.9 <= abs(k) <= 1.1 for k in coef[:2]):
            # Blame the column that is not part of the problem spec (e.g. an unmapped
            # extra like x_per_sqft); if both or neither are, flag both for review.
            blamed = [c for c in (a, b) if c not in spec_features] or [a, b]
            other = {a: b, b: a}
            for c in blamed:
                out.append(LeakFinding(c, "pair", f"target ~ {c} x/÷ {other[c]} (log R^2 {r2:.3f})"))
    return out
