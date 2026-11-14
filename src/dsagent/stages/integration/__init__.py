"""Stage 4: integration and cleaning (design spec section 10, decisions D2/D3).

integrate(spec, entity_sources, context_joins) -> IntegrationResult

  normalise + map each entity source onto the spec schema
  -> vertical union (provenance on every row)
  -> exact + near-duplicate removal (source-trust ordering)
  -> context enrichment joins (key join / nearest point)
  -> per-column missing-value plan (applied later, fitted on the training split)

Output: unified table (Parquet) + integration report (JSON) with column lineage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

import polars as pl

from dsagent.config import Settings, get_settings
from dsagent.schemas.spec import DatasetSpec
from dsagent.stages.integration.dedupe import DedupeReport, dedupe
from dsagent.stages.integration.enrich import EnrichReport, key_join, nearest_distance
from dsagent.stages.integration.impute import ColumnPlan, plan_imputation
from dsagent.stages.integration.normalise import ColumnReport
from dsagent.stages.integration.union import PROVENANCE_COLS, SourceTable, source_unit_conflicts, union_sources


@dataclass
class ContextJoin:
    source_id: str
    kind: Literal["key_join", "nearest_point"]
    df: pl.DataFrame
    # key_join: table_key, context_key, columns {ctx_col: feature}, aliases
    # nearest_point: feature, lat, lon, point_lat, point_lon
    params: dict = field(default_factory=dict)


@dataclass
class IntegrationReport:
    rows_per_source: dict[str, int]
    rows_after_union: int
    dedupe: DedupeReport
    rows_after_dedupe: int
    columns: list[ColumnReport]
    unit_conflicts: list[str]
    enrichment: list[EnrichReport]
    imputation_plan: list[ColumnPlan]
    lineage: dict[str, list[str]]  # feature -> ["source_id.column", ...] or ["derived:..."]
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        d["imputation_plan"] = [p.model_dump() for p in self.imputation_plan]
        return json.dumps(d, indent=2, default=str)


@dataclass
class IntegrationResult:
    table: pl.DataFrame
    report: IntegrationReport

    def write(self, out_dir: Path) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {"table": out_dir / "unified.parquet", "report": out_dir / "integration_report.json"}
        self.table.write_parquet(paths["table"])
        paths["report"].write_text(self.report.to_json(), encoding="utf-8")
        return paths


def integrate(
    spec: DatasetSpec,
    entity_sources: list[SourceTable],
    context_joins: list[ContextJoin] | None = None,
    settings: Settings | None = None,
    impute_group_by: Optional[str] = None,
) -> IntegrationResult:
    settings = settings or get_settings()
    cfg = settings.integration
    warnings: list[str] = []

    table, col_reports = union_sources(spec, entity_sources, cfg)
    rows_after_union = table.height
    for r in col_reports:
        if r.failed:
            warnings.append(f"{r.feature} <- {r.source_column}: {r.failed} unparseable values set to null")
        warnings += [f"{r.feature} <- {r.source_column}: {n}" for n in r.notes]

    feature_cols = [spec.target.name, *spec.feature_names()]
    table, dd = dedupe(table, feature_cols, spec.target.name, cfg)
    if dd.target_conflicts:
        warnings.append(f"{len(dd.target_conflicts)} duplicate groups disagree on the target (kept most trusted row)")

    lineage: dict[str, list[str]] = {}
    for s in entity_sources:
        for f, c in s.candidate.feature_map.items():
            lineage.setdefault(f, []).append(f"{s.candidate.source_id}.{c}")

    enrich_reports: list[EnrichReport] = []
    for cj in context_joins or []:
        if cj.kind == "key_join":
            table, er = key_join(table, cj.df, source_id=cj.source_id, **cj.params)
        else:
            missing = [c for c in (cj.params.get("lat", "latitude"), cj.params.get("lon", "longitude")) if c not in table.columns]
            if missing:
                warnings.append(f"{cj.source_id}: skipped, table lacks {missing}")
                continue
            table, er = nearest_distance(table, cj.df, source_id=cj.source_id, **cj.params)
        enrich_reports.append(er)
        for f in er.features:
            lineage.setdefault(f, []).append(f"derived:{cj.source_id}:{cj.kind}")
        if er.rows_total and er.rows_matched < er.rows_total:
            warnings.append(f"{cj.source_id}: {er.rows_total - er.rows_matched}/{er.rows_total} rows not matched")

    # Keep provenance columns last and fold enrichment features in before extras.
    ordered = [c for c in table.columns if c not in PROVENANCE_COLS] + PROVENANCE_COLS
    table = table.select(ordered)

    group_by = impute_group_by if impute_group_by in table.columns else None
    plan = plan_imputation(table, spec, cfg, group_by=group_by)

    report = IntegrationReport(
        rows_per_source={s.candidate.source_id: s.df.height for s in entity_sources},
        rows_after_union=rows_after_union,
        dedupe=dd,
        rows_after_dedupe=table.height,
        columns=col_reports,
        unit_conflicts=source_unit_conflicts(col_reports),
        enrichment=enrich_reports,
        imputation_plan=plan,
        lineage=lineage,
        warnings=warnings,
    )
    return IntegrationResult(table=table, report=report)
