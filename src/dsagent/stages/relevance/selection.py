"""Model-guided source selection: three strategies under nested CV (design spec 16.4, D5).

Column groups:  base = entity-source columns;  one group per context source.
  S-base   : base columns only
  S-full   : base + every context group
  S-greedy : METAM-style loop *inside each outer training fold*: start from base,
             add context groups in the given (trust) order, keep a group only if the
             inner-CV score improves by > epsilon (relative); stop after
             `max_no_improvement` misses or `max_candidate_sources` groups.
Every strategy is scored on the same outer folds, so selection never sees the fold
it is scored on. The final held-out split is not used here at all.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import BaseModel

from dsagent.config import Settings
from dsagent.schemas.spec import DatasetSpec
from dsagent.stages.relevance.modeling import cat_indices, fit_encoder, fit_score, kfold, target_array


class StrategyResult(BaseModel):
    strategy: str
    mean: float
    sd: float
    fold_scores: list[float]
    groups_selected_per_fold: list[list[str]] = []


def _cv_score(df: pl.DataFrame, spec: DatasetSpec, cols: list[str], settings: Settings, k: int, seed: int) -> float:
    if not cols:
        return float("nan")
    y, classes = target_array(df, spec)
    w = df["_sample_weight"].to_numpy() if "_sample_weight" in df.columns else None
    scores = []
    for tr, te in kfold(spec, y, k, seed):
        enc = fit_encoder(df[tr], cols)
        Xtr, Xte = enc.transform(df[tr], cols), enc.transform(df[te], cols)
        s, _ = fit_score(spec, settings.relevance.lgbm_params, seed, Xtr, y[tr], Xte, y[te],
                         None if w is None else w[tr], cat_indices(enc, cols))
        scores.append(s)
    return float(np.nanmean(scores))


def _greedy(train: pl.DataFrame, spec, base: list[str], groups: dict[str, list[str]], settings: Settings) -> list[str]:
    sel = settings.selection
    cols = list(base)
    chosen: list[str] = []
    s0 = _cv_score(train, spec, cols, settings, sel.inner_folds, sel.random_seed)
    misses = 0
    for gid, gcols in list(groups.items())[: sel.max_candidate_sources]:
        s1 = _cv_score(train, spec, cols + gcols, settings, sel.inner_folds, sel.random_seed)
        gain = (s1 - s0) / abs(s0) if s0 and np.isfinite(s0) else (s1 - s0)
        if np.isfinite(s1) and gain > sel.epsilon_relative:
            cols += gcols
            chosen.append(gid)
            s0, misses = s1, 0
        else:
            misses += 1
            if misses >= sel.max_no_improvement:
                break
    return chosen


def compare_strategies(
    train: pl.DataFrame, spec: DatasetSpec, base: list[str], groups: dict[str, list[str]], settings: Settings
) -> list[StrategyResult]:
    sel = settings.selection
    y, _ = target_array(train, spec)
    w = train["_sample_weight"].to_numpy() if "_sample_weight" in train.columns else None
    full = base + [c for g in groups.values() for c in g]
    res = {"S-base": [], "S-full": [], "S-greedy": []}
    picked: list[list[str]] = []
    for tr, te in kfold(spec, y, sel.outer_folds, sel.random_seed):
        otr, ote = train[tr], train[te]
        chosen = _greedy(otr, spec, base, groups, settings) if groups else []
        picked.append(chosen)
        greedy_cols = base + [c for g in chosen for c in groups[g]]
        for name, cols in (("S-base", base), ("S-full", full), ("S-greedy", greedy_cols)):
            enc = fit_encoder(otr, cols)
            s, _ = fit_score(spec, settings.relevance.lgbm_params, sel.random_seed,
                             enc.transform(otr, cols), y[tr], enc.transform(ote, cols), y[te],
                             None if w is None else w[tr], cat_indices(enc, cols))
            res[name].append(s)
    out = []
    for name, scores in res.items():
        arr = np.array(scores, dtype=float)
        out.append(StrategyResult(
            strategy=name, mean=round(float(np.nanmean(arr)), 4),
            sd=round(float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
            fold_scores=[round(float(s), 4) for s in arr],
            groups_selected_per_fold=picked if name == "S-greedy" else [],
        ))
    return out
