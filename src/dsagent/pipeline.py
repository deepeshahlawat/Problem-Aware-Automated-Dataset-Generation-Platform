"""The pipeline as resumable steps with the two human checkpoints (design spec 7.6).

  start_job(problem)            Stage 1  -> status 'spec_ready'          (checkpoint 1)
  approve_spec(job, drop=[...]) Stage 2  -> status 'feasibility_ready'   (checkpoint 2)
  build(job, continue_on_review) Stages 4-7 -> 'done' | 'needs_review' | 'failed'

Used by the web API (dsagent.api) and usable from scripts. Stage 3 (live acquisition)
is not built yet, so entity tables come from files registered in data/sources.json.
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import polars as pl

from dsagent.config import Settings, get_settings
from dsagent.llm import Budget, LLMCallCache, LLMClient
from dsagent.schemas.spec import DatasetSpec
from dsagent.stages.discovery import DiscoveryResult, discover_sources
from dsagent.stages.discovery.catalogs import CuratedContextCatalog, LocalManifestCatalog
from dsagent.stages.integration import IntegrationResult, integrate
from dsagent.stages.integration.union import SourceTable
from dsagent.stages.recency import RecencyReport, apply_recency
from dsagent.stages.relevance import RelevanceResult, run_relevance
from dsagent.stages.requirement import RequirementResult, analyse_problem
from dsagent.stages.validation import ValidationResult, validate

ROOT = Path(__file__).resolve().parents[2]  # dataset-agent/
DOWNLOADS = ("dataset.csv", "dataset.parquet", "quality_report.json", "relevance.json",
             "integration_report.json", "run_summary.json", "excluded_rows.csv")


@dataclass
class Job:
    id: str
    problem: str
    status: str = "created"
    stage: Optional[str] = None  # current stage while building (progress view)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None
    log: list[str] = field(default_factory=list)
    requirement: Optional[RequirementResult] = None
    spec: Optional[DatasetSpec] = None
    discovery: Optional[DiscoveryResult] = None
    integration: Optional[IntegrationResult] = None
    validation: Optional[ValidationResult] = None
    recency: Optional[RecencyReport] = None
    relevance: Optional[RelevanceResult] = None
    dataset: Optional[pl.DataFrame] = None
    continued_despite_review: bool = False
    budget: Optional[Budget] = None
    out_dir: Optional[Path] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def note(self, msg: str) -> None:
        self.log.append(f"{datetime.now(timezone.utc):%H:%M:%S} {msg}")


def _llm(job: Job, settings: Settings) -> LLMClient:
    if job.budget is None:
        job.budget = Budget(settings.max_tokens_per_run, settings.max_cost_per_run, settings.max_pages_per_run)
    return LLMClient(job.budget, LLMCallCache(ROOT / settings.cache_db_path), settings=settings)


def start_job(problem: str, settings: Settings | None = None) -> Job:
    """Stage 1. Raises NeedsClarification / RequirementError for the caller to show."""
    settings = settings or get_settings()
    job = Job(id=uuid.uuid4().hex[:10], problem=problem.strip())
    job.note("Stage 1: requirement analysis")
    job.requirement = analyse_problem(job.problem, _llm(job, settings), settings=settings)
    job.spec = job.requirement.spec
    job.status = "spec_ready"
    job.note(f"spec ready: target {job.spec.target.name}, {len(job.spec.features)} features")
    return job


def approve_spec(job: Job, drop_features: list[str] | None = None, settings: Settings | None = None,
                 overrides: dict[str, dict] | None = None) -> Job:
    """Checkpoint 1 passed, then Stage 2. The user may remove features and correct
    a feature's type or required flag (overrides: {name: {"dtype": ..., "required": ...}})."""
    settings = settings or get_settings()
    drop = set(drop_features or [])
    if overrides:
        allowed = {"dtype", "required"}
        feats = []
        for f in job.spec.features:
            o = {k: v for k, v in (overrides.get(f.name) or {}).items() if k in allowed}
            if o.get("dtype") and o["dtype"] != f.dtype and o["dtype"] not in ("int", "float"):
                o.update(unit=None, expected_range=None)  # a category has no unit or numeric range
            if o:
                feats.append(f.model_validate({**f.model_dump(), **o}))
                job.note(f"user changed {f.name}: {o}")
            else:
                feats.append(f)
        job.spec = job.spec.model_copy(update={"features": feats})
    if drop:
        kept = [f for f in job.spec.features if f.name not in drop]
        if not kept:
            raise ValueError("at least one feature must remain")
        job.spec = job.spec.model_copy(update={"features": kept})
        job.note(f"user removed features: {sorted(drop)}")
    job.note("Stage 2: source discovery")
    cats = [LocalManifestCatalog(ROOT / settings.source_manifest_path, settings.discovery.sample_rows),
            CuratedContextCatalog(ROOT / settings.context_registry_path)]
    job.discovery = discover_sources(job.spec, cats, _llm(job, settings), settings)
    job.status = "feasibility_ready"
    job.note(f"feasibility ready: deliverable={job.discovery.deliverable}")
    return job


def _entity_tables(job: Job, settings: Settings) -> list[SourceTable]:
    manifest_path = ROOT / settings.source_manifest_path
    manifest = {e["source_id"]: e for e in json.loads(manifest_path.read_text(encoding="utf-8"))}
    tables = []
    for c in job.discovery.selected_candidates():
        if c.role != "entity" or c.source_id not in manifest:
            continue
        path = manifest_path.parent / manifest[c.source_id]["path"]
        tables.append(SourceTable(c, pl.read_csv(path, infer_schema=False, encoding="utf8-lossy")))
    return tables


def build(job: Job, continue_on_review: bool = False, settings: Settings | None = None) -> Job:
    """Checkpoint 2 passed, then Stages 4-7 and outputs. Never raises: errors -> 'failed'."""
    settings = settings or get_settings()
    with job.lock:
        try:
            warnings.filterwarnings("ignore", category=UserWarning)
            job.status, job.error, job.stage = "building", None, "Loading sources"
            tables = _entity_tables(job, settings)
            if not tables:
                raise ValueError("no entity source was selected; nothing to build (nothing is fabricated)")

            job.stage = "Integrating sources"
            job.note("Stage 4: integration")
            group_by = "locality" if "locality" in job.spec.feature_names() else None
            job.integration = integrate(job.spec, tables, settings=settings, impute_group_by=group_by)
            spec = job.integration.spec
            r4 = job.integration.report

            job.stage = "Validating"
            job.note("Stage 5: validation")
            dup = round(1 - r4.rows_after_dedupe / r4.rows_after_union, 4) if r4.rows_after_union else None
            job.validation = validate(job.integration.table, spec, r4.dedupe.agreement, settings, dup)
            if job.validation.report.status == "needs_review" and not continue_on_review:
                job.status, job.stage = "needs_review", None
                job.note("validation needs review: stopped before delivery")
                return job
            job.continued_despite_review = job.validation.report.status == "needs_review"

            job.stage = "Weighting by recency"
            job.note("Stage 6: recency weighting")
            src_dates = {c.source_id: c.last_updated for c in job.discovery.selected_candidates() if c.last_updated}
            table, job.recency = apply_recency(job.validation.table, spec, None, src_dates, settings)

            job.stage = "Scoring features"
            job.note("Stage 7: relevance + source selection")
            job.relevance = run_relevance(table, spec, r4.lineage, _llm(job, settings), settings)

            keep = [spec.target.name, *job.relevance.selected_columns]
            prov = [c for c in ("_row_ref", "_source_id", "_row_confidence", "_sample_weight") if c in table.columns]
            job.dataset = table.select(keep + prov)
            _write_outputs(job, spec, table, settings)
            job.status, job.stage = "done", None
            job.note(f"done: {job.dataset.height} rows x {len(keep)} columns")
        except Exception as exc:  # surfaced to the user, never silently swallowed
            job.status, job.stage = "failed", None
            job.error = f"{type(exc).__name__}: {exc}"
            job.note("failed: " + job.error)
            job.log.append(traceback.format_exc(limit=3))
    return job


def _write_outputs(job: Job, spec: DatasetSpec, full: pl.DataFrame, settings: Settings) -> None:
    out = ROOT / "outputs" / job.id
    out.mkdir(parents=True, exist_ok=True)
    job.dataset.write_parquet(out / "dataset.parquet")
    job.dataset.write_csv(out / "dataset.csv")
    ex = job.validation.excluded
    if ex.height:
        ex.with_columns(pl.col("_flags").list.join("; ")).write_csv(out / "excluded_rows.csv")
    (out / "quality_report.json").write_text(job.validation.report.model_dump_json(indent=2), encoding="utf-8")
    (out / "integration_report.json").write_text(job.integration.report.to_json(), encoding="utf-8")
    rel = job.relevance
    (out / "relevance.json").write_text(json.dumps({
        "columns": [c.model_dump() for c in rel.columns], "selected": rel.selected_columns,
        "strategies": [s.model_dump() for s in rel.strategies], "chosen": rel.chosen_strategy,
        "holdout_scores": rel.holdout_scores, "metric": rel.metric,
        "leakage": [f.__dict__ for f in rel.leakage], "notes": rel.notes,
    }, indent=2), encoding="utf-8")
    (out / "run_summary.json").write_text(json.dumps({
        "job_id": job.id, "problem": job.problem, "spec": spec.model_dump(mode="json"),
        "validation_status": job.validation.report.status,
        "continued_despite_review": job.continued_despite_review,
        "llm_calls": job.budget.llm_calls, "cache_hits": job.budget.cache_hits,
        "cost_usd": round(job.budget.cost_used, 6), "finished_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    job.out_dir = out


def job_view(job: Job) -> dict[str, Any]:
    """JSON-friendly summary for the UI."""
    v: dict[str, Any] = {"id": job.id, "problem": job.problem, "status": job.status, "stage": job.stage,
                         "error": job.error,
                         "log": job.log[-30:], "created_at": job.created_at}
    if job.budget:
        v["cost"] = {"usd": round(job.budget.cost_used, 5), "llm_calls": job.budget.llm_calls,
                     "cache_hits": job.budget.cache_hits}
    if job.spec:
        v["spec"] = job.spec.model_dump(mode="json")
    if job.requirement:
        v["requirement"] = {"assumptions": job.requirement.assumptions, "excluded": job.requirement.excluded,
                            "guard_removed": job.requirement.guard_removed, "volatility": job.requirement.volatility}
    if job.discovery:
        d = job.discovery
        v["discovery"] = {
            "deliverable": d.deliverable, "blocking": d.blocking, "warnings": d.warnings[:20],
            "selected": d.selected, "rejected": d.rejected,
            "feasibility": [f.model_dump(mode="json") for f in d.feasibility],
            "candidates": [{"source_id": c.source_id, "title": c.title, "role": c.role, "trust": c.trust_score,
                            "trust_components": c.trust_components, "covers": c.covers_features,
                            "feature_map": c.feature_map, "licence": c.licence, "url": c.url}
                           for c in d.candidates],
        }
    if job.integration:
        r = job.integration.report
        v["integration"] = {"rows_per_source": r.rows_per_source, "rows_after_union": r.rows_after_union,
                            "rows_after_dedupe": r.rows_after_dedupe, "exact_removed": r.dedupe.exact_removed,
                            "near_removed": r.dedupe.near_removed, "conflicts": len(r.dedupe.target_conflicts),
                            "warnings": r.warnings[:20]}
    if job.validation:
        v["quality"] = json.loads(job.validation.report.model_dump_json(exclude={"contract"}))
    if job.recency:
        v["recency"] = job.recency.model_dump()
    if job.relevance:
        rel = job.relevance
        v["relevance"] = {
            "columns": [c.model_dump() for c in sorted(rel.columns, key=lambda c: -c.fused_score)],
            "selected": rel.selected_columns, "chosen": rel.chosen_strategy, "metric": rel.metric,
            "strategies": [s.model_dump(exclude={"fold_scores", "groups_selected_per_fold"}) for s in rel.strategies],
            "holdout_scores": rel.holdout_scores, "train_rows": rel.train_rows, "holdout_rows": rel.holdout_rows,
            "leakage": [f.__dict__ for f in rel.leakage], "notes": rel.notes,
            # True when the semantic (AI) review failed: leakage protection then rests on rules only
            "ai_review_missing": any(n.startswith("semantic check unavailable") for n in rel.notes),
        }
    v["continued_despite_review"] = job.continued_despite_review
    if job.dataset is not None:
        v["dataset"] = {"rows": job.dataset.height, "columns": job.dataset.columns}
        v["downloads"] = [n for n in DOWNLOADS if job.out_dir and (job.out_dir / n).exists()]
    return v
