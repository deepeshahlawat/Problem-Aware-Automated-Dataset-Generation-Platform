"""Stage 5: validation (design spec sections 10, 13.2, 13.3).

validate(table, spec, agreement) -> ValidationResult

1. A **contract** is derived from the spec (never from the data it checks):
     hard (row excluded):  target present and in range; impossible negatives
     soft (row kept, confidence reduced): required feature missing,
           value outside the spec's expected range, category not allowed
   The same rules are run as Pandera schemas, whose failure cases give the rows.
2. **Outliers** (IQR, k=3) are flagged, never deleted: a luxury flat is real data.
3. **Row confidence** = source_trust * extraction_factor * agreement_factor * validation_factor.
4. **Gate**: critical problems (target column missing or empty, a required feature
   entirely absent, too few rows, too many exclusions) set status 'needs_review' so
   the run is surfaced to the user, never silently delivered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import pandera.errors
import pandera.polars as pa
import polars as pl
from pydantic import BaseModel, Field

from dsagent.config import Settings, get_settings
from dsagent.schemas.spec import DatasetSpec, Feature
from dsagent.stages.integration.normalise import normalise_category
from dsagent.text import norm

Severity = Literal["hard", "soft"]


class Rule(BaseModel):
    """One machine-checkable expectation. Exported as the dataset's contract."""

    column: str
    check: Literal["not_null", "ge", "in_range", "isin"]
    severity: Severity
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str

    def pandera_check(self) -> Optional[pa.Check]:
        if self.check == "ge":
            return pa.Check.ge(self.params["min"])
        if self.check == "in_range":
            return pa.Check.in_range(self.params["min"], self.params["max"])
        if self.check == "isin":
            return pa.Check.isin(self.params["values"])
        return None  # not_null is expressed through Column(nullable=False)


class CheckFailure(BaseModel):
    column: str
    check: str
    severity: Severity
    count: int
    examples: list[str]


class QualityReport(BaseModel):
    status: Literal["passed", "needs_review"]
    critical: list[str]
    rows_in: int
    rows_valid: int
    rows_excluded: int
    completeness: dict[str, float]
    failed_checks: list[CheckFailure]
    outliers: dict[str, int]
    trust_bands: dict[str, int]
    agreement: dict[str, int]
    llm_derived_share: float
    mean_row_confidence: Optional[float]
    duplication_rate: Optional[float] = None
    contract: list[Rule]


@dataclass
class ValidationResult:
    table: pl.DataFrame      # valid rows + _row_confidence, _flags
    excluded: pl.DataFrame   # hard failures + _exclusion_reason (kept for provenance)
    report: QualityReport

    @property
    def deliverable(self) -> bool:
        return self.report.status == "passed"


def _nonnegative(f: Feature, settings: Settings) -> bool:
    toks = set(norm(f.name).split("_")) | set(norm(f.unit or "").split("_"))
    return any(t in toks for t in settings.validation.nonnegative_tokens)


def build_contract(spec: DatasetSpec, settings: Settings | None = None) -> list[Rule]:
    settings = settings or get_settings()
    rules: list[Rule] = []
    t = spec.target
    rules.append(Rule(column=t.name, check="not_null", severity="hard", reason="target is required"))
    if t.dtype in ("int", "float"):
        if t.expected_range:
            rules.append(Rule(column=t.name, check="in_range", severity="hard",
                              params={"min": t.expected_range.min, "max": t.expected_range.max},
                              reason="target outside the spec's expected range"))
        elif _nonnegative(t, settings):
            rules.append(Rule(column=t.name, check="ge", severity="hard", params={"min": 0},
                              reason="target cannot be negative"))
    for f in spec.features:
        if f.required:
            rules.append(Rule(column=f.name, check="not_null", severity="soft",
                              reason="required feature missing"))
        if f.dtype in ("int", "float"):
            if _nonnegative(f, settings):
                lo = max(0.0, f.expected_range.min) if f.expected_range else 0.0
                rules.append(Rule(column=f.name, check="ge", severity="hard", params={"min": 0},
                                  reason="impossible negative value"))
                if f.expected_range:
                    rules.append(Rule(column=f.name, check="in_range", severity="soft",
                                      params={"min": lo, "max": f.expected_range.max},
                                      reason="outside the spec's expected range"))
            elif f.expected_range:
                rules.append(Rule(column=f.name, check="in_range", severity="soft",
                                  params={"min": f.expected_range.min, "max": f.expected_range.max},
                                  reason="outside the spec's expected range"))
        if f.dtype == "category" and f.allowed_values:
            vals = sorted({v for v in (normalise_category(x) for x in f.allowed_values) if v})
            rules.append(Rule(column=f.name, check="isin", severity="soft", params={"values": vals},
                              reason="category not in the spec's allowed values"))
    return rules


def _schema(rules: list[Rule], columns: list[str]) -> pa.DataFrameSchema:
    cols: dict[str, pa.Column] = {}
    for r in rules:
        if r.column not in columns:
            continue
        c = cols.setdefault(r.column, pa.Column(None, checks=[], nullable=True, required=True))
        if r.check == "not_null":
            c.nullable = False
        else:
            c.checks.append(r.pandera_check())
    return pa.DataFrameSchema(cols, strict=False)


def _run(schema: pa.DataFrameSchema, df: pl.DataFrame) -> pl.DataFrame:
    """Pandera failure cases (empty frame if everything passed)."""
    try:
        schema.validate(df, lazy=True)
    except pandera.errors.SchemaErrors as e:
        return e.failure_cases
    return pl.DataFrame(schema={"failure_case": pl.Utf8, "column": pl.Utf8, "check": pl.Utf8, "index": pl.Int32})


def _check_label(check: str) -> str:
    return "not_null" if check == "not_nullable" else check


def validate(
    table: pl.DataFrame,
    spec: DatasetSpec,
    agreement: dict[str, str] | None = None,
    settings: Settings | None = None,
    duplication_rate: Optional[float] = None,
) -> ValidationResult:
    settings = settings or get_settings()
    v = settings.validation
    agreement = agreement or {}
    rules = build_contract(spec, settings)
    critical: list[str] = []

    # --- column-level (critical) ------------------------------------------------------
    n = table.height
    if spec.target.name not in table.columns:
        critical.append(f"target column '{spec.target.name}' missing")
    elif n and table[spec.target.name].null_count() == n:
        critical.append(f"target column '{spec.target.name}' is entirely empty")
    for f in spec.features:
        if f.required and (f.name not in table.columns or (n and table[f.name].null_count() == n)):
            critical.append(f"required feature '{f.name}' has no values")

    # --- row-level via Pandera ----------------------------------------------------------
    hard_reasons: dict[int, list[str]] = {}
    soft_fails: dict[int, list[str]] = {}
    failures: dict[tuple[str, str, str], list[str]] = {}
    for severity in ("hard", "soft"):
        sub = [r for r in rules if r.severity == severity]
        fc = _run(_schema(sub, table.columns), table)
        for row in fc.iter_rows(named=True):
            idx, col, chk = row.get("index"), row.get("column"), _check_label(str(row.get("check")))
            if idx is None:
                critical.append(f"{col}: {chk} (column-level)")
                continue
            failures.setdefault((col, chk, severity), []).append(str(row.get("failure_case")))
            target = hard_reasons if severity == "hard" else soft_fails
            target.setdefault(int(idx), []).append(f"{col}:{chk}")

    # --- outliers (flag only) ------------------------------------------------------------
    flags: dict[int, list[str]] = {i: [f"soft:{s}" for s in soft_fails.get(i, [])] for i in range(n)}
    outliers: dict[str, int] = {}
    numeric = [f.name for f in [spec.target, *spec.features]
               if f.dtype in ("int", "float") and f.name in table.columns]
    passing = [i for i in range(n) if i not in hard_reasons]  # bounds from rows that passed hard checks
    for col in numeric:
        vals = table[col].to_list()
        s = pl.Series([vals[i] for i in passing], dtype=pl.Float64).drop_nulls()
        if s.len() < 8:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if not iqr:
            continue
        lo, hi = q1 - v.outlier_iqr_k * iqr, q3 + v.outlier_iqr_k * iqr
        hits = [i for i in passing if vals[i] is not None and (vals[i] < lo or vals[i] > hi)]
        outliers[col] = len(hits)
        for i in hits:
            flags[i].append(f"outlier:{col}")

    # --- row confidence -------------------------------------------------------------------
    ext = settings.extraction
    factor_by_method = {"api": ext.api, "file": ext.file, "html_table": ext.html_table,
                        "jsonld": ext.jsonld, "llm": ext.llm_verified, "derived": ext.derived}
    trust = table["_source_trust"].to_list() if "_source_trust" in table.columns else [1.0] * n
    method = table["_method"].to_list() if "_method" in table.columns else ["file"] * n
    refs = table["_row_ref"].to_list() if "_row_ref" in table.columns else [str(i) for i in range(n)]
    conf, agree_counts = [], {"confirmed": 0, "single": 0, "disagree": 0}
    for i in range(n):
        a = agreement.get(refs[i], "single")
        if i not in hard_reasons:
            agree_counts[a] += 1
        af = {"confirmed": v.agreement_confirmed, "disagree": v.agreement_disagree}.get(a, v.agreement_single)
        if a == "disagree":
            flags[i].append("sources_disagree")
        vf = 0.0 if i in hard_reasons else v.soft_penalty ** len(soft_fails.get(i, []))
        conf.append(round(trust[i] * factor_by_method.get(method[i], 1.0) * af * vf, 4))

    annotated = table.with_columns(
        pl.Series("_row_confidence", conf, dtype=pl.Float64),
        pl.Series("_flags", [flags[i] for i in range(n)], dtype=pl.List(pl.Utf8)),
        pl.Series("_exclusion_reason", ["; ".join(hard_reasons[i]) if i in hard_reasons else None for i in range(n)],
                  dtype=pl.Utf8),
    )
    is_bad = pl.col("_exclusion_reason").is_not_null()
    excluded = annotated.filter(is_bad)
    valid = annotated.filter(~is_bad).drop("_exclusion_reason")

    if valid.height < spec.min_rows:
        critical.append(f"only {valid.height} valid rows, spec requires at least {spec.min_rows}")
    if n and excluded.height / n > v.max_excluded_share:
        critical.append(f"{excluded.height}/{n} rows excluded by hard checks (> {v.max_excluded_share:.0%})")

    vc = valid["_row_confidence"].to_list()
    report = QualityReport(
        status="needs_review" if critical else "passed",
        critical=critical,
        rows_in=n,
        rows_valid=valid.height,
        rows_excluded=excluded.height,
        completeness={c: round(1 - valid[c].null_count() / valid.height, 4) if valid.height else 0.0
                      for c in valid.columns if not c.startswith("_")},
        failed_checks=[CheckFailure(column=c, check=k, severity=sev, count=len(ex), examples=ex[:5])
                       for (c, k, sev), ex in sorted(failures.items())],
        outliers=outliers,
        trust_bands={
            "high": sum(x >= v.trust_band_high for x in vc),
            "medium": sum(v.trust_band_medium <= x < v.trust_band_high for x in vc),
            "low": sum(x < v.trust_band_medium for x in vc),
        },
        agreement=agree_counts,
        llm_derived_share=round(sum(m == "llm" for m in method) / n, 4) if n else 0.0,
        mean_row_confidence=round(sum(vc) / len(vc), 4) if vc else None,
        duplication_rate=duplication_rate,
        contract=rules,
    )
    return ValidationResult(table=valid, excluded=excluded, report=report)
