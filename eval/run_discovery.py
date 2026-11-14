"""Stage 1 + Stage 2 dry run for any problem text (no data is downloaded).

    .venv/Scripts/python eval/run_discovery.py "I want to predict house prices in Delhi" --max-cost 0.05

Uses: local manifest (data/sources.json, if present), curated context registry,
and live Hugging Face search. Prints the feasibility report and saves JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dsagent.config import Settings
from dsagent.llm import Budget, LLMCallCache, LLMClient
from dsagent.stages.discovery import discover_sources
from dsagent.stages.discovery.catalogs import (
    CuratedContextCatalog, HuggingFaceCatalog, LocalManifestCatalog,
)
from dsagent.stages.requirement import analyse_problem

HERE = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("--max-cost", type=float, default=0.05)
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--no-llm-matching", action="store_true")
    args = ap.parse_args()

    s = Settings()
    budget = Budget(s.max_tokens_per_run, args.max_cost, args.max_pages)
    llm = LLMClient(budget, LLMCallCache(s.cache_db_path), settings=s)

    req = analyse_problem(args.problem, llm, settings=s)
    spec = req.spec
    print(f"SPEC: target={spec.target.name} task={spec.task_type} region={spec.region} "
          f"features={spec.feature_names()}")
    if req.guard_removed:
        print("guard removed:", req.guard_removed)

    catalogs = [
        LocalManifestCatalog(s.source_manifest_path, s.discovery.sample_rows),
        CuratedContextCatalog(s.context_registry_path),
        HuggingFaceCatalog(budget, s.user_agent, s.discovery.hf_search_limit),
    ]
    res = discover_sources(spec, catalogs, None if args.no_llm_matching else llm, s)

    print("\nCANDIDATES (trust):")
    for c in sorted(res.candidates, key=lambda c: -c.trust_score):
        mark = "*" if c.source_id in res.selected else " "
        print(f" {mark} {c.trust_score:.2f} {c.role:7} {c.source_id}  covers={c.covers_features}")
    print("\nREJECTED:")
    for sid, why in res.rejected:
        print(f"   {sid}: {why}")
    print("\nFEASIBILITY:\n" + res.feasibility_markdown())
    print(f"\ndeliverable={res.deliverable} blocking={res.blocking}")
    if res.warnings:
        print("warnings:", *res.warnings[:12], sep="\n  ")
    print(f"\nLLM calls={budget.llm_calls} cache_hits={budget.cache_hits} "
          f"cost=${budget.cost_used:.5f} http_pages={budget.pages_used}")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", args.problem.lower())[:40].strip("_")
    path = out / f"discovery_{slug}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps({
        "problem": args.problem,
        "spec": spec.model_dump(mode="json"),
        "candidates": [c.model_dump(mode="json") for c in res.candidates],
        "selected": res.selected,
        "rejected": res.rejected,
        "feasibility": [f.model_dump(mode="json") for f in res.feasibility],
        "deliverable": res.deliverable,
        "blocking": res.blocking,
        "warnings": res.warnings,
        "cost_usd": budget.cost_used,
        "http_pages": budget.pages_used,
    }, indent=2), encoding="utf-8")
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
