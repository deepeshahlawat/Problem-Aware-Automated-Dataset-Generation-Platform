"""Duplicate removal (Stage 4 step 4).

1. Exact duplicates on all collected columns (provenance ignored).
2. Near-duplicate *candidates* via MinHash LSH over a record signature built from every
   collected column except the target (categorical values as words, numbers rounded to
   N significant digits). Extras such as a model name are included on purpose.
3. A candidate is merged only if its target agrees with the kept row (within tolerance).
   Candidates whose targets disagree are KEPT: identical-looking records at different
   prices are often genuinely different items (found on real laptop data, where 192 rows
   were wrongly merged by a spec-columns-only signature). If they come from different
   sources, both rows are marked 'disagree' and the conflict is reported.

Within a merged group the row from the most trusted source wins (design spec 13).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl
from datasketch import MinHash, MinHashLSH

from dsagent.config import IntegrationConfig


@dataclass
class DedupeReport:
    rows_in: int = 0
    exact_removed: int = 0
    near_removed: int = 0
    groups: int = 0
    target_conflicts: list[dict] = field(default_factory=list)
    # kept _row_ref -> "confirmed" (same record from >= 2 independent sources, targets
    # agree) or "disagree" (targets differ beyond tolerance). Feeds row confidence (13.3).
    agreement: dict[str, str] = field(default_factory=dict)


def _round_sig(x: float, sig: int) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def signature(row: dict, cols: list[str], sig: int) -> set[str]:
    toks: set[str] = set()
    for c in cols:
        v = row.get(c)
        if v is None:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            toks.add(f"{c}={_round_sig(float(v), sig):g}")
        else:
            for w in str(v).split():
                toks.add(f"{c}~{w}")
    return toks


def _targets_agree(a, b, rel_tol: float) -> bool:
    if a is None or b is None:
        return True  # nothing to contradict
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        lo, hi = min(a, b), max(a, b)
        return hi == lo or (lo > 0 and (hi - lo) / lo <= rel_tol)
    return a == b


def _keep_best(group: list[int], trust: list[float]) -> int:
    return max(group, key=lambda i: (trust[i], -i))


def dedupe(
    table: pl.DataFrame, feature_cols: list[str], target: str, cfg: IntegrationConfig
) -> tuple[pl.DataFrame, DedupeReport]:
    rep = DedupeReport(rows_in=table.height)
    if table.height == 0:
        return table, rep
    extras = [c for c in table.columns if not c.startswith("_") and c not in feature_cols]
    # An extra column only one source has would make that source's rows look different
    # from everyone else's; use only extras present in >= 2 sources (or all, if 1 source).
    n_src = table["_source_id"].n_unique()
    shared_extras = [c for c in extras
                     if n_src == 1 or table.filter(pl.col(c).is_not_null())["_source_id"].n_unique() >= 2]
    feature_cols = list(feature_cols) + shared_extras
    rows = table.select(feature_cols).to_dicts()
    trust = table["_source_trust"].to_list()
    refs = table["_row_ref"].to_list()
    srcs = table["_source_id"].to_list()
    support: dict[int, set[str]] = {}  # kept row -> sources that contained this record
    disagree: set[int] = set()

    # 1. exact duplicates
    first: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        first.setdefault(tuple(r[c] for c in feature_cols), []).append(i)
    keep = set()
    for idxs in first.values():
        best = _keep_best(idxs, trust)
        keep.add(best)
        support[best] = {srcs[i] for i in idxs}
        rep.exact_removed += len(idxs) - 1

    # 2. near duplicates among the survivors (signature excludes the target so that
    # the same listing re-posted at a different price is still caught and reported)
    sig_cols = [c for c in feature_cols if c != target]
    survivors = sorted(keep)
    lsh = MinHashLSH(threshold=cfg.near_dup_threshold, num_perm=cfg.minhash_perm)
    mh: dict[int, MinHash] = {}
    for i in survivors:
        toks = signature(rows[i], sig_cols, cfg.signature_round_sig)
        if len(toks) < 2:
            continue  # too little information to call anything a duplicate
        m = MinHash(num_perm=cfg.minhash_perm)
        for t in toks:
            m.update(t.encode("utf-8"))
        mh[i] = m
        lsh.insert(str(i), m)

    seen: set[int] = set()
    for i in survivors:
        if i in seen or i not in mh:
            continue
        group = sorted({int(k) for k in lsh.query(mh[i])} - seen)
        seen.update(group)
        if len(group) < 2:
            continue
        rep.groups += 1
        best = _keep_best(group, trust)
        merged = set(support.get(best, {srcs[best]}))
        for j in group:
            if j == best:
                continue
            if _targets_agree(rows[best][target], rows[j][target], cfg.target_conflict_rel_tol):
                merged |= support.get(j, {srcs[j]})
                keep.discard(j)
                rep.near_removed += 1
            elif srcs[j] != srcs[best]:
                # Same-looking record, different target, different sources: keep both, flag both.
                rep.target_conflicts.append({"rows": [refs[best], refs[j]],
                                             "target_values": [rows[best][target], rows[j][target]]})
                disagree.update((best, j))
        support[best] = merged

    for i in keep:
        if i in disagree:
            rep.agreement[refs[i]] = "disagree"
        elif len(support.get(i, ())) >= 2:
            rep.agreement[refs[i]] = "confirmed"

    mask = pl.Series([i in keep for i in range(table.height)])
    return table.filter(mask), rep
