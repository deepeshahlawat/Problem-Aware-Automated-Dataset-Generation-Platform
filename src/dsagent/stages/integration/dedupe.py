"""Duplicate removal (Stage 4 step 4).

1. Exact duplicates on the canonical feature columns (provenance ignored).
2. Near-duplicates (re-posted listings) via MinHash LSH over a canonical record
   signature: categorical values as words, numbers rounded to N significant digits.

Within a duplicate group the row from the most trusted source wins (design spec 13:
source-trust ordering). Groups whose target values disagree beyond a tolerance are
reported, never silently averaged.
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


def _keep_best(group: list[int], trust: list[float]) -> int:
    return max(group, key=lambda i: (trust[i], -i))


def dedupe(
    table: pl.DataFrame, feature_cols: list[str], target: str, cfg: IntegrationConfig
) -> tuple[pl.DataFrame, DedupeReport]:
    rep = DedupeReport(rows_in=table.height)
    if table.height == 0:
        return table, rep
    rows = table.select(feature_cols).to_dicts()
    trust = table["_source_trust"].to_list()
    refs = table["_row_ref"].to_list()

    # 1. exact duplicates
    first: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        first.setdefault(tuple(r[c] for c in feature_cols), []).append(i)
    keep = set()
    for idxs in first.values():
        keep.add(_keep_best(idxs, trust))
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
        targets = [rows[j][target] for j in group if rows[j][target] is not None]
        if len(targets) > 1 and isinstance(targets[0], (int, float)):
            lo, hi = min(targets), max(targets)
            if lo > 0 and (hi - lo) / lo > cfg.target_conflict_rel_tol:
                rep.target_conflicts.append({
                    "rows": [refs[j] for j in group], "target_values": targets, "kept": refs[best]})
        for j in group:
            if j != best:
                keep.discard(j)
                rep.near_removed += 1

    mask = pl.Series([i in keep for i in range(table.height)])
    return table.filter(mask), rep
