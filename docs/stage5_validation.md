# Stage 5: Validation

**Code:** `src/dsagent/stages/validation.py` · agreement tracking: `stages/integration/dedupe.py` · tests: `tests/test_validation.py` (8) + 1 in `tests/test_integration.py`

`validate(table, spec, agreement, duplication_rate=None) -> ValidationResult(table, excluded, report)`

## 1. Contract derived from the spec, not from the data
Rules are built from the `DatasetSpec` alone. Deriving them from the data being checked would make every dataset pass. Each rule is a `Rule` object, exported in the report as the dataset's machine-checkable contract, and run through **Pandera** (polars backend). Pandera's failure cases give the exact failing rows.

| Rule | Severity | When |
|---|---|---|
| target not null | **hard** | always |
| target ≥ 0 | **hard** | numeric target whose name or unit implies a quantity that can't be negative (price, area, count, weight, ...) |
| target in expected range | soft | the spec gives `expected_range`. *Changed after the real-data run:* this rule was hard, and an LLM-guessed 100–5000 "USD" range excluded every laptop row whose currency was unstated. |

**Spec/data mismatch:** if more than 50% of non-null values break a soft range or category rule, the rule is treated as wrong (unit, currency or vocabulary mismatch), not the rows. No row is penalised, and the mismatch is listed as a critical issue (`needs_review`) for the user to resolve.
| feature ≥ 0 | **hard** | numeric feature that can't be negative (same token list, `config.validation.nonnegative_tokens`) |
| required feature not null | soft | `required=True` |
| feature in expected range | soft | the spec gives `expected_range` |
| category in allowed values | soft | the spec gives `allowed_values` (normalised like the data) |

- **Hard failure:** the row is **excluded**. It is kept in `excluded` with an `_exclusion_reason` for provenance, never silently dropped.
- **Soft failure:** the row is **kept**. A flag is added and its confidence is multiplied by 0.8.
- A feature like `temperature_change` gets no ≥ 0 rule, because nothing in its name says it can't be negative.

## 2. Outliers: flag, never delete
Bounds are Q1 − 3·IQR and Q3 + 3·IQR, computed only on rows that passed the hard checks, so an excluded −9e12 can't distort them. Flagged rows stay in the data, because a luxury flat is a real data point. Columns with fewer than 8 values are skipped.

## 3. Row confidence (design spec 13.3)
`row_confidence = source_trust × extraction_factor × agreement_factor × validation_factor`

| Factor | Values |
|---|---|
| extraction | api/file 1.0 · html_table 0.95 · jsonld 0.9 · llm (verified) 0.75 · derived 1.0 |
| agreement | confirmed by ≥ 2 sources 1.0 · single source 0.9 · sources disagree 0.6 (+ `sources_disagree` flag) |
| validation | 0.8 per soft failure; a hard failure means the row is excluded |

Agreement comes from Stage 4. `dedupe()` now records, for each kept row, whether the same record appeared in two or more independent sources with agreeing targets ("confirmed") or with conflicting targets ("disagree").

## 4. Gate: needs review, never silent delivery
`status = "needs_review"` if any of these hold:
- the target column is missing or empty;
- a required feature has no values at all;
- a column-level schema error occurs;
- valid rows < `spec.min_rows`;
- more than 30% of rows fail hard checks.

Otherwise the status is `"passed"`.

## Quality report (`QualityReport`, JSON)
- status and critical issues;
- rows in / valid / excluded;
- completeness per column;
- failed checks (column, check, severity, count, up to 5 example values);
- outlier counts;
- trust bands (high ≥ 0.7, medium ≥ 0.4, low);
- agreement counts;
- the LLM-derived share of rows;
- mean row confidence;
- duplication rate (passed in from Stage 4);
- the full contract.

## Definition of done
| Item | Status |
|---|---|
| Typed I/O | integrated table + spec → `ValidationResult` |
| Unit tests | 9 (contract derivation, hard/soft split, confidence formula, outliers, three gate conditions, Stage 4 → 5 hand-off) |
| Provenance | excluded rows kept with reasons; flags per row; contract exported |
| Documented | this file |

## Not yet done / next
- **Cross-source agreement at aggregate level** (e.g. median price per locality compared across sources) is not implemented. Only record-level agreement through duplicates is.
- **Extraction verification** (an LLM value must appear in the source text) belongs to Stage 3, which will set `_method = "llm"` only for verified values.
- **The ydata-profiling HTML report** is deferred to Stage 8 (outputs). Stage 5 produces the JSON report that page will be built from.
- **Experiment E6:** manually audit about 100 valid rows against the original source and report the observed error rate and how precise the flags are.
