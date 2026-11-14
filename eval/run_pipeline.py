"""End-to-end run on real data: Stage 1 -> 2 -> 4 -> 5 -> 6 -> 7.

    .venv/Scripts/python eval/run_pipeline.py "Predict the price of a laptop from its specifications"

Sources: the local manifest (data/sources.json) and the curated context registry.
Stage 3 (live acquisition) is not built yet, so entity tables are read from the
registered local files. Outputs go to outputs/<slug>/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from dsagent.config import Settings
from dsagent.llm import Budget, LLMCallCache, LLMClient
from dsagent.stages.discovery import discover_sources
from dsagent.stages.discovery.catalogs import CuratedContextCatalog, LocalManifestCatalog
from dsagent.stages.integration import integrate
from dsagent.stages.integration.union import SourceTable
from dsagent.stages.recency import apply_recency
from dsagent.stages.relevance import run_relevance
from dsagent.stages.requirement import analyse_problem
from dsagent.stages.validation import validate

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]


def hr(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("--max-cost", type=float, default=0.10)
    ap.add_argument("--no-llm-semantic", action="store_true")
    ap.add_argument("--continue-on-review", action="store_true",
                    help="experiments only: run Stages 6-7 even if validation needs review")
    args = ap.parse_args()

    s = Settings()
    budget = Budget(s.max_tokens_per_run, args.max_cost, s.max_pages_per_run)
    llm = LLMClient(budget, LLMCallCache(s.cache_db_path), settings=s)
    t0 = time.time()

    hr("Stage 1: requirement analysis")
    req = analyse_problem(args.problem, llm, settings=s)
    spec = req.spec
    print(f"target={spec.target.name} ({spec.target.unit}) task={spec.task_type} region={spec.region}")
    print("features:", spec.feature_names())

    hr("Stage 2: discovery")
    cats = [LocalManifestCatalog(ROOT / s.source_manifest_path, s.discovery.sample_rows),
            CuratedContextCatalog(ROOT / s.context_registry_path)]
    disc = discover_sources(spec, cats, llm, s)
    for sid, why in disc.rejected:
        print(f"  rejected {sid}: {why}")
    print(disc.feasibility_markdown())
    print(f"deliverable={disc.deliverable} blocking={disc.blocking}")
    entities = [c for c in disc.selected_candidates() if c.role == "entity"]
    if not entities:
        print("No entity source selected; stopping (nothing is fabricated).")
        return 1
    for c in entities:
        print(f"  {c.source_id}: trust={c.trust_score} map={c.feature_map}")

    hr("Stage 4: integration")
    manifest = {e["source_id"]: e for e in json.loads((ROOT / s.source_manifest_path).read_text(encoding="utf-8"))}
    tables = []
    for c in entities:
        path = (ROOT / s.source_manifest_path).parent / manifest[c.source_id]["path"]
        df = pl.read_csv(path, infer_schema=False, encoding="utf8-lossy")
        tables.append(SourceTable(c, df))
    integ = integrate(spec, tables, settings=s,
                      impute_group_by="locality" if "locality" in spec.feature_names() else None)
    r4 = integ.report
    print(f"rows per source={r4.rows_per_source} after union={r4.rows_after_union} "
          f"after dedupe={r4.rows_after_dedupe} (exact -{r4.dedupe.exact_removed}, near -{r4.dedupe.near_removed}, "
          f"target conflicts {len(r4.dedupe.target_conflicts)})")
    for w in r4.warnings[:15]:
        print("  !", w)

    hr("Stage 5: validation")
    dup_rate = round(1 - r4.rows_after_dedupe / r4.rows_after_union, 4) if r4.rows_after_union else None
    if integ.spec != spec:
        print("  spec corrected from data:", [w for w in r4.warnings if w.startswith("dtype corrected")])
    spec = integ.spec
    val = validate(integ.table, spec, r4.dedupe.agreement, s, dup_rate)
    q = val.report
    print(f"status={q.status} valid={q.rows_valid} excluded={q.rows_excluded} critical={q.critical}")
    for f in q.failed_checks:
        print(f"  {f.severity:4} {f.column}: {f.check} x{f.count}  e.g. {f.examples[:3]}")
    print(f"outliers={q.outliers} trust_bands={q.trust_bands} mean_conf={q.mean_row_confidence}")
    if q.status == "needs_review":
        if not args.continue_on_review:
            print("\nSTOPPED: validation needs review (see critical issues above). Nothing delivered.\n"
                  "Re-run with --continue-on-review to analyse anyway (experiments only).")
            return 2
        print("\n!! continuing despite needs_review (--continue-on-review); results are provisional")

    hr("Stage 6: recency")
    src_dates = {c.source_id: c.last_updated for c in entities if c.last_updated}
    rec_table, rec = apply_recency(val.table, spec, None, src_dates, s)
    print(rec.model_dump())

    hr("Stage 7: relevance + source selection")
    rel = run_relevance(rec_table, spec, r4.lineage, None if args.no_llm_semantic else llm, s)
    for c in sorted(rel.columns, key=lambda c: -c.fused_score):
        print(f"  {c.column:28} {c.tier:9} fused={c.fused_score:.2f} model={c.model_score:.2f} "
              f"mi={c.stat_score:.2f} sem={c.semantic_score:.2f} | {c.reason[:80]}")
    for st in rel.strategies:
        print(f"  {st.strategy:9} {rel.metric}={st.mean:.4f} ± {st.sd:.4f}")
    print(f"chosen={rel.chosen_strategy} holdout={rel.holdout_scores} train={rel.train_rows} holdout_rows={rel.holdout_rows}")
    print("leakage:", [(f.column, f.rule, f.detail) for f in rel.leakage])
    for n in rel.notes:
        print("  note:", n)

    hr("Run summary")
    print(f"LLM calls={budget.llm_calls} cache hits={budget.cache_hits} cost=${budget.cost_used:.5f} "
          f"runtime={time.time() - t0:.1f}s")

    slug = re.sub(r"[^a-z0-9]+", "_", args.problem.lower())[:40].strip("_")
    out = ROOT / "outputs" / slug
    paths = integ.write(out)
    rec_table.write_parquet(out / "validated.parquet")
    val.excluded.write_parquet(out / "excluded_rows.parquet")
    (out / "quality_report.json").write_text(q.model_dump_json(indent=2), encoding="utf-8")
    (out / "relevance.json").write_text(json.dumps({
        "columns": [c.model_dump() for c in rel.columns], "selected": rel.selected_columns,
        "strategies": [st.model_dump() for st in rel.strategies], "chosen": rel.chosen_strategy,
        "holdout_scores": rel.holdout_scores, "metric": rel.metric,
        "leakage": [f.__dict__ for f in rel.leakage], "notes": rel.notes,
    }, indent=2), encoding="utf-8")
    (out / "run_summary.json").write_text(json.dumps({
        "problem": args.problem, "spec": spec.model_dump(mode="json"),
        "validation_status": q.status, "continued_despite_review": bool(args.continue_on_review and q.status != "passed"),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "llm_calls": budget.llm_calls, "cache_hits": budget.cache_hits, "cost_usd": budget.cost_used,
        "runtime_s": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")
    print(f"outputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
