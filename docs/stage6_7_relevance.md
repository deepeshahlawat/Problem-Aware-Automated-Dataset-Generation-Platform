# Stage 6 (recency) and Stage 7 (relevance engine + source selection)

**Code:** `src/dsagent/stages/recency.py` · `src/dsagent/stages/relevance/` (`modeling.py`, `signals.py`, `leakage.py`, `semantic.py`, `selection.py`, `__init__.py`) · tests: `tests/test_relevance.py` (10) on `tests/synthetic.py`

## Stage 6: recency weighting
`apply_recency(table, spec, date_column, source_dates) -> (table, RecencyReport)`

- **Observation date, in order of preference:** the row's own date column → the source's last-updated date → unknown. Unknown means a neutral weight of 1.0, counted in the report and never guessed.
- **`recent_weighted`:** `weight = max(0.05, 0.5 ** (age / half_life))`.
- **`window`:** rows outside the years are removed. Undated rows can't be shown to fall inside the window, so they are removed too; both counts are reported.
- **`uniform`:** weight 1.0 for every row.
- **`_sample_weight = recency weight × row confidence`** is passed to LightGBM and exported.

## Stage 7: relevance engine
`run_relevance(table, spec, lineage, llm=None) -> RelevanceResult`

**Data hygiene first.** Rows without a target are dropped. A **20% holdout** is set aside and used exactly once, at the end. Every signal, rule and selection decision uses the training split only.

| Signal | How | Notes |
|---|---|---|
| statistical | mutual information with the target, normalised to [0, 1] | high-cardinality categoricals (ID-like) get 0 |
| model-based | LightGBM baseline: permutation importance on an internal validation split + mean \|TreeSHAP\| (LightGBM `pred_contrib`), each normalised, averaged | computed after leaks and IDs are removed, so a leak can't soak up importance |
| semantic | one batched cheap-LLM call: relevance 0–1 + role (predictive / identifier / metadata / target_derived / future_info) | columns the model invents are ignored |

**Fusion:** `0.4·model + 0.3·stat + 0.3·semantic`. Tiers: high ≥ 0.7, medium ≥ 0.4, low ≥ 0.15, otherwise dropped. **Without an LLM, the two available signals are re-weighted (÷ 0.7).** A neutral 0.5 would have lifted pure noise above the drop line; this was found in the synthetic run and fixed.

**Overrides**, which always exclude the column but keep it in provenance:

| Override | Rule |
|---|---|
| leakage (name) | named as a transform of the target (`price_per_sqft`, `log_price`) |
| leakage (pair) | `log(target) = a·log(col) + b·log(other)`, R² ≥ 0.98, \|a\|,\|b\| ≈ 1: the target is col × other or col ÷ other. **This is the only rule that catches `Per_Sqft = Price ÷ Area`**, which no single-column test sees. The column outside the spec is blamed; if both are spec features, both are flagged. |
| leakage (single) | the column alone predicts the target with R² ≥ 0.98 or AUC ≥ 0.99. **Confirmed or cleared by the semantic check** (spec 16.3); without an LLM it is treated as leakage, conservatively. |
| leakage (semantic) | role `target_derived` or `future_info` |
| dropped | near-constant (≥ 99% one value) |
| metadata | categorical > 95% unique, or semantic role identifier / metadata |

## Source selection: three strategies under nested CV (decision D5)
- **Groups:** columns whose lineage is entirely `derived:<source>` form one group per context source. Everything else is the base.
- **Strategies:**
  - **S-base**: base columns only;
  - **S-full**: base + every context group;
  - **S-greedy**: inside each outer training fold, add groups in trust order and keep one only if the inner-CV score improves by > 0.5%. Stop after 2 misses.
- **Evaluation:** 5 outer × 3 inner folds, all on the training split, so selection never sees the fold it is scored on.
- **Choice:** the best mean wins. Ties go to the simpler configuration.

**Final:** the chosen columns are fitted on the full training split and scored **once** on the holdout, alongside "all collected columns" (E4 variant A).

## Planted-structure check (`tests/synthetic.py`, 600 rows, ~6 s)
| Planted | Result |
|---|---|
| `x_per_sqft = price / area` | **leakage (pair, log R² 1.000)** |
| unique listing ID | metadata |
| constant site name | dropped |
| pure noise column | dropped (fused 0.01) |
| useful context (metro distance) | kept by greedy in 5/5 folds |
| useless context (amenities) | dropped |
| strategies (R²) | base 0.916 · full 0.966 · greedy 0.966 → greedy (tie → simpler) |
| holdout R² | selected 0.964 · all columns 0.990 (inflated by the leak, which is what E4/E10 should show) |

## Definition of done
| Item | Status |
|---|---|
| Typed I/O | table + spec + lineage → `RelevanceResult` (per-column `ColumnRelevance`, leakage findings, strategy results, holdout scores, holdout row refs) |
| Unit tests | 10 (recency modes; tiers on planted data; pair leak; strategies; holdout disjointness; single-leak confirm/clear with a stub LLM; semantic overrides; classification path) |
| Leakage-safe | holdout untouched; selection nested; imputation not needed (LightGBM handles NaN) |
| Configurable | all weights, thresholds and LightGBM parameters in `config.relevance` / `config.selection` |
| Documented | this file |

## Not yet done / next
- **Fusion weights are defaults.** They are justified only once **E11** (stat only vs model only vs LLM only vs fused) runs on real data.
- **Real data:** all results above are synthetic. Running on the Kaggle Delhi and laptop files is the next real test, including E10 on the real `Per_Sqft` column.
- `holdout_refs` must be reused by Stage 8 so the exported train/test split is the one used here.
