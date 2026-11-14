"""Stage 7: feature relevance engine + model-guided source selection (design spec 16).

run_relevance(table, spec, lineage, llm=None) -> RelevanceResult

1. Rows without a target are dropped; a holdout split (20%) is set aside and touched
   exactly once, at the end.
2. On the training split only:
     statistical signal (MI) and diagnostics -> leakage rules (name / single / pair)
     -> semantic signal (one cheap-LLM call; neutral if unavailable)
     -> overrides: near-constant -> dropped; identifier/metadata -> metadata;
        leakage -> leakage (excluded, kept in provenance)
     -> model signal (LightGBM permutation + TreeSHAP) on the remaining columns
     -> fused = 0.4*model + 0.3*stat + 0.3*semantic -> tiers
3. S-base / S-full / S-greedy compared under nested CV on the training split.
4. The chosen configuration is fitted on the whole training split and scored once on
   the holdout, alongside the 'all collected columns' configuration (E4 variant A).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from dsagent.config import Settings, get_settings
from dsagent.llm.client import LLMClient
from dsagent.schemas.report import ColumnRelevance
from dsagent.schemas.spec import DatasetSpec
from dsagent.stages.relevance.leakage import LeakFinding, find_leakage
from dsagent.stages.relevance.modeling import (
    cat_indices, fit_encoder, fit_score, is_feature_column, split_holdout, target_array,
)
from dsagent.stages.relevance.selection import StrategyResult, _greedy, compare_strategies
from dsagent.stages.relevance.semantic import assess
from dsagent.stages.relevance.signals import model_based, statistical


@dataclass
class RelevanceResult:
    columns: list[ColumnRelevance]
    selected_columns: list[str]
    leakage: list[LeakFinding]
    strategies: list[StrategyResult]
    chosen_strategy: str
    chosen_groups: list[str]
    holdout_scores: dict[str, float]
    metric: str
    train_rows: int
    holdout_rows: int
    holdout_refs: list[str]
    notes: list[str] = field(default_factory=list)


def _tier(fused: float, settings: Settings) -> str:
    r = settings.relevance
    if fused >= r.tier_high:
        return "high"
    if fused >= r.tier_medium:
        return "medium"
    if fused >= r.tier_low:
        return "low"
    return "dropped"


def _groups(cols: list[str], lineage: dict[str, list[str]]) -> tuple[list[str], dict[str, list[str]]]:
    base, groups = [], {}
    for c in cols:
        lin = lineage.get(c, [])
        derived = [x for x in lin if x.startswith("derived:")]
        if lin and len(derived) == len(lin):
            sid = derived[0].split(":")[1]
            groups.setdefault(sid, []).append(c)
        else:
            base.append(c)
    return base, groups


def run_relevance(
    table: pl.DataFrame,
    spec: DatasetSpec,
    lineage: dict[str, list[str]] | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    group_order: list[str] | None = None,
) -> RelevanceResult:
    settings = settings or get_settings()
    rc, seed = settings.relevance, settings.selection.random_seed
    lineage = lineage or {}
    notes: list[str] = []
    t = spec.target.name

    data = table.filter(pl.col(t).is_not_null())
    if data.height < table.height:
        notes.append(f"{table.height - data.height} rows without a target dropped")
    if data.height < 30:
        raise ValueError(f"relevance analysis needs at least 30 rows with a target; got {data.height}")
    train, holdout = split_holdout(data, spec, rc.holdout_share, seed)

    candidates = [c for c in data.columns if is_feature_column(c) and c != t]
    enc = fit_encoder(train, candidates)
    stats = statistical(train, spec, enc, settings)
    leaks = find_leakage(train, spec, stats, enc.numeric, settings)

    sem, sem_notes = ({}, ["no LLM: semantic score neutral, single-feature leakage not semantically confirmed"]) \
        if llm is None else assess(train, spec, candidates, llm)
    notes += sem_notes

    # --- overrides -----------------------------------------------------------------
    override: dict[str, tuple[str, str]] = {}
    by_col: dict[str, list[LeakFinding]] = {}
    for f in leaks:
        by_col.setdefault(f.column, []).append(f)
    for c in candidates:
        s, a = stats[c], sem.get(c)
        structural = [f for f in by_col.get(c, []) if f.rule in ("name", "pair")]
        single = [f for f in by_col.get(c, []) if f.rule == "single"]
        if train[c].null_count() == train.height:
            override[c] = ("dropped", "no values: not available from any selected source")
        elif structural:
            override[c] = ("leakage", structural[0].detail)
        elif a and a.role in ("target_derived", "future_info"):
            override[c] = ("leakage", f"semantic: {a.role} ({a.reason})")
        elif single:
            if a and a.role == "predictive":
                notes.append(f"{c}: near-perfect alone but semantically predictive; kept ({single[0].detail})")
            else:
                override[c] = ("leakage", single[0].detail + ("" if a else " (not semantically confirmed)"))
        elif s.top_share >= rc.near_constant_share:
            override[c] = ("dropped", f"near-constant ({s.top_share:.1%} one value)")
        elif (c in enc.categorical and s.unique_ratio > rc.id_unique_ratio) or (a and a.role == "identifier"):
            override[c] = ("metadata", "identifier-like" + (f": {a.reason}" if a else f" ({s.unique_ratio:.0%} unique)"))
        elif a and a.role == "metadata":
            override[c] = ("metadata", f"semantic: {a.reason}")

    usable = [c for c in candidates if c not in override]
    model_based(train, spec, enc, stats, settings, usable)

    cols_out: list[ColumnRelevance] = []
    for c in candidates:
        s, a = stats[c], sem.get(c)
        sem_score = a.relevance if a else rc.semantic_neutral
        if a:
            fused = rc.w_model * s.model_score + rc.w_stat * s.stat_score + rc.w_semantic * sem_score
        else:
            # No semantic evidence: re-weight the two available signals instead of letting a
            # neutral 0.5 lift every column (including pure noise) above the drop threshold.
            fused = (rc.w_model * s.model_score + rc.w_stat * s.stat_score) / (rc.w_model + rc.w_stat)
        fused = round(fused, 4)
        if c in override:
            tier, reason = override[c]
        else:
            tier = _tier(fused, settings)
            reason = (f"model {s.model_score:.2f}, MI {s.stat_score:.2f}, semantic {sem_score:.2f}"
                      + (f" ({a.reason})" if a else ""))
        cols_out.append(ColumnRelevance(column=c, stat_score=s.stat_score, model_score=s.model_score,
                                        semantic_score=round(sem_score, 4), fused_score=min(fused, 1.0),
                                        tier=tier, reason=reason))
    kept = [r.column for r in cols_out if r.tier in ("high", "medium", "low")]

    # --- strategies (nested CV on train) ---------------------------------------------
    base, groups = _groups(kept, lineage)
    if group_order:
        groups = {g: groups[g] for g in group_order if g in groups} | {g: v for g, v in groups.items() if g not in group_order}
    if not base:
        notes.append("no entity-source columns survived; strategies use context columns as base")
        base, groups = kept, {}
    strategies = compare_strategies(train, spec, base, groups, settings) if kept else []
    order = {"S-base": 0, "S-greedy": 1, "S-full": 2}  # ties -> simpler configuration
    best = min(strategies, key=lambda r: (-r.mean if np.isfinite(r.mean) else np.inf, order[r.strategy])) if strategies else None
    chosen = best.strategy if best else "S-base"
    if chosen == "S-full":
        chosen_groups = list(groups)
    elif chosen == "S-greedy":
        chosen_groups = _greedy(train, spec, base, groups, settings)
    else:
        chosen_groups = []
    selected = base + [c for g in chosen_groups for c in groups[g]]

    # --- single final holdout evaluation --------------------------------------------------
    y_tr, classes = target_array(train, spec)
    y_ho, _ = target_array(holdout, spec, classes)
    w = train["_sample_weight"].to_numpy() if "_sample_weight" in train.columns else None
    holdout_scores = {}
    for name, cols in (("selected", selected), ("all_columns", candidates)):
        if not cols:
            continue
        e = fit_encoder(train, cols)
        s, _ = fit_score(spec, rc.lgbm_params, seed, e.transform(train, cols), y_tr,
                         e.transform(holdout, cols), y_ho, w, cat_indices(e, cols))
        holdout_scores[name] = round(s, 4)
    if any(r.tier == "leakage" for r in cols_out):
        notes.append("'all_columns' includes leaking columns, so its holdout score is optimistic by construction")

    metric = "r2" if spec.task_type == "regression" else ("roc_auc" if classes and len(classes) == 2 else "f1_macro")
    return RelevanceResult(
        columns=cols_out, selected_columns=selected, leakage=leaks, strategies=strategies,
        chosen_strategy=chosen, chosen_groups=chosen_groups, holdout_scores=holdout_scores, metric=metric,
        train_rows=train.height, holdout_rows=holdout.height,
        holdout_refs=holdout["_row_ref"].to_list() if "_row_ref" in holdout.columns else [],
        notes=notes,
    )
