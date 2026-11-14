"""Experiment E1: requirement analyser vs expert feature lists.

Calls the real LLM (costs money; needs ANTHROPIC_API_KEY or `ant auth login`).
Each repeat uses its own cache namespace, so repeats are independent samples
(LLM output is non-deterministic), while re-running the script replays for free.

    .venv/Scripts/python eval/run_e1.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from dsagent.config import Settings
from dsagent.evaluation.spec_scoring import load_gold, score_spec
from dsagent.llm import Budget, BudgetExceeded, LLMCallCache, LLMClient
from dsagent.llm.client import LLMError
from dsagent.stages.requirement import RequirementError, analyse_problem

HERE = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--planner", help="override the planner model ID for this run")
    ap.add_argument("--only", nargs="*", help="gold spec ids to run (default: all)")
    ap.add_argument("--max-cost", type=float, help="USD cap for this run (default: from .env)")
    ap.add_argument("--gold", type=Path, default=HERE / "gold_specs")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()

    golds = [g for g in load_gold(args.gold) if not args.only or g.id in args.only]
    base = Settings()
    if args.planner:
        base = base.model_copy(update={"llm_model_planner": args.planner})
    cache = LLMCallCache(base.cache_db_path)
    rows = []
    budget = Budget(
        base.max_tokens_per_run, args.max_cost or base.max_cost_per_run, base.max_pages_per_run
    )
    print(f"planner={base.llm_model_planner} provider={base.llm_provider} "
          f"problems={len(golds)} repeats={args.repeats} cap=${budget.cost_max}")

    stopped = None
    for rep in range(args.repeats):
        settings = base.model_copy(update={"prompt_cache_namespace": f"e1-rep{rep}"})
        llm = LLMClient(budget, cache, settings=settings)
        for gold in golds:
            cost_before, tok_before = budget.cost_used, budget.tokens_used
            try:
                result = analyse_problem(gold.problem, llm, settings=settings)
            except RequirementError as exc:
                rows.append({"repeat": rep, "id": gold.id, "error": str(exc)})
                print(f"[rep {rep}] {gold.id}: ERROR {exc}")
                continue
            except (BudgetExceeded, LLMError) as exc:
                stopped = f"{type(exc).__name__}: {exc}"
                print(f"STOPPED: {stopped}")
                break
            s = score_spec(result.spec, gold)
            call_cost = budget.cost_used - cost_before
            rows.append({
                "repeat": rep, **s.model_dump(),
                "cost_usd": round(call_cost, 6),
                "tokens": budget.tokens_used - tok_before,
                "predicted_features": result.spec.feature_names(),
                "target": result.spec.target.name,
                "guard_removed": result.guard_removed,
                "excluded_by_llm": result.excluded,
            })
            print(f"[rep {rep}] {gold.id}: P={s.precision} R={s.recall} F1={s.f1} "
                  f"target_ok={s.target_correct} forbidden={s.forbidden_included} "
                  f"cost=${call_cost:.5f}")
        if stopped:
            break

    scored = [r for r in rows if "error" not in r]
    summary = {}
    for key in ("precision", "recall", "f1"):
        vals = [r[key] for r in scored]
        summary[key] = {
            "mean": round(statistics.mean(vals), 3) if vals else None,
            "sd": round(statistics.stdev(vals), 3) if len(vals) > 1 else None,
        }
    summary["target_accuracy"] = (
        round(sum(r["target_correct"] for r in scored) / len(scored), 3) if scored else None
    )
    summary["forbidden_rate"] = (
        round(sum(bool(r["forbidden_included"]) for r in scored) / len(scored), 3) if scored else None
    )
    summary["planner_model"] = base.llm_model_planner
    summary["stopped"] = stopped
    summary["errors"] = len(rows) - len(scored)
    summary["llm_calls"] = budget.llm_calls
    summary["cache_hits"] = budget.cache_hits
    summary["cost_usd"] = round(budget.cost_used, 4)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = base.llm_model_planner.replace("/", "_")
    path = args.out / f"e1_{tag}_{stamp}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
