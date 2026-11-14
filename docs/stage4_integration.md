# Stage 4: Integration and cleaning

**Code:** `src/dsagent/stages/integration/` (`normalise.py`, `union.py`, `dedupe.py`, `enrich.py`, `impute.py`, `__init__.py`) · tests: `tests/test_integration.py` (28 tests)

`integrate(spec, entity_sources, context_joins, impute_group_by=None) -> IntegrationResult`
Outputs `unified.parquet` and `integration_report.json`.

Built per the design spec build order (weeks 5–6): it works on hand-provided tables, before any acquisition agent exists. Inputs are the Stage 2 `SourceCandidate`s (with their `feature_map`) plus the raw tables.

## Steps
1. **Normalise** (`normalise.py`). Each value is parsed to the spec feature's dtype and unit:
   - Numbers with separators and multipliers: `6,500,000`, `65 Lac`, `1.2 Cr`, `2.5 mn`, `8GB`, `1,450 Sq. Ft.`. **Ambiguous single letters (`12 L`, `2.5 M`) are not multiplied**, because they could be lakh or litre, million or metre.
   - Units come from the value itself (`111.5 sqm`), a declared unit (`SourceTable.column_units`), or the column name (`Price_euros`, `Weight_kg`), in that order.
   - Physical units (area, mass, length, distance, storage) are converted with fixed constants.
   - **Currency is converted only with a configured, dated rate** (`integration.currency_rates`, `currency_rates_as_of`). Without one, values stay in the source currency and the report says so. A guessed exchange rate would be fabricated data.
   - Categories are trimmed, case-folded and given unified separators (`" Rohini  Sector-25"` → `rohini sector 25`).
   - Unparseable values become null and are counted per column.
2. **Schema mapping + union** (`union.py`, decision D2). Each entity source is rewritten onto the spec schema via its Stage 2 `feature_map`, then all sources are stacked.
   - Unmapped columns are kept as `x_<name>`, so the "all collected attributes" baseline (E4 variant A) remains possible.
   - Every row gets `_row_ref`, `_source_id`, `_method`, `_source_trust` and `_retrieved_at`.
3. **Deduplicate** (`dedupe.py`):
   - Exact duplicates are removed first.
   - Near-duplicates are found with MinHash LSH (threshold 0.9) over a record signature: categorical words plus numbers rounded to 2 significant digits. **The target is excluded from the signature**, so a listing re-posted at a different price is still caught.
   - The most trusted source's row is kept.
   - Groups whose targets differ by more than 5% are **reported**, never averaged.
4. **Enrich** (`enrich.py`). Exact, auditable joins:
   - **key join**: by normalised key, with an optional alias table. Unmatched keys are listed and never fuzzy-guessed.
   - **nearest point**: haversine distance in km to the nearest context point (e.g. metro stations). Lineage records it as `derived:<source>`.
5. **Missing values** (`impute.py`). Stage 4 only **plans** a strategy per column:

   | Column | Strategy |
   |---|---|
   | Target | never imputed; rows without it are dropped |
   | More than 40% missing | left null |
   | Numeric | median, or group median (e.g. within locality) |
   | Category | explicit `__missing__` value |
   | Bool / date | left null |

   `apply_plan(df, plan, fit_on=train_split)` fits the statistics on the **training split only**, so test rows never influence the fill values. Filled cells get a `<col>__imputed` flag.

## Report contents
- rows per source, after the union and after dedup;
- per column: values parsed, failed and missing, units from → to, notes;
- unit conflicts between sources;
- dedup counts and target conflicts;
- enrichment match rates and unmatched keys;
- the imputation plan with a reason for each column;
- column lineage (`feature → ["source.column", "derived:..."]`).

## Definition of done
| Item | Status |
|---|---|
| Typed I/O | `DatasetSpec` + `SourceTable`s + `ContextJoin`s → `IntegrationResult` |
| Unit tests | 28 tests on synthetic fixtures (parsing table, units, union, dedup, joins, imputation leakage check, end-to-end Parquet round trip) |
| Provenance | per row (5 columns) + per column (lineage) |
| Never invent data | unparseable → null; no guessed currency rates; no fuzzy key joins; no imputation of the target or of >40%-missing columns |
| Documented | this file |

## Not yet done / next
- **Run on real data.** This needs the hand-downloaded Kaggle files (`data/README.md`) registered in `data/sources.json`. Stage 3 (acquisition) will later fetch tables automatically.
- **Experiment E3** needs a small hand-labelled sample: about 100 row pairs marked duplicate / not duplicate, plus localities checked against the circle-rate join. The precision and recall of dedup and joining come from that sample.
- Locality alias tables (`Dwarka Sec 7` ↔ `Dwarka Sector 7`) are supplied by hand for now. A cheap-LLM alias proposer whose suggestions the user approves would fit the advisory-automation principle.
- Exchange rates must be entered in config with a date before any cross-currency union.
