# First run on real data (2026-11-14)

**Command:** `eval/run_pipeline.py "<problem>"` runs Stages 1 → 2 → 4 → 5 → 6 → 7 on registered local files. Stage 3 (live acquisition) is not built yet.

**Data** (downloaded with permission, git-ignored in `data/raw/`):

| Case | File(s) | Source | Licence |
|---|---|---|---|
| Laptops | `laptops_train.csv` (977 rows) + `laptops_test.csv` (325) | huggingface.co/datasets/Ammok/laptop_price_prediction | Apache-2.0 |
| Delhi houses | `delhi_magicbricks.csv` (1,259) | github.com/SUKHMAN-SINGH-1612/Data-Science-Projects, a re-upload of the Kaggle MagicBricks scrape | **none stated**: private testing only |

**Cost:** about $0.003 of OpenRouter credit for all runs. Re-runs are free from the LLM cache. Balance after the session: $3.637.

## Bugs that only real data exposed (all fixed, all generic)
| # | Symptom | Cause | Fix | Regression test |
|---|---|---|---|---|
| 1 | Laptops: **all 1,080 rows excluded** | Stage 1 guessed a price range of 100–5000 "USD". The real prices are in the millions, in an unstated currency, and the range was a **hard** rule. | Ranges from the spec are soft. If >50% of values break a soft range or category rule, it is reported as a spec/data mismatch (`needs_review`) and no row is penalised. | `test_spec_range_in_wrong_unit_is_reported_not_used_to_delete_rows` |
| 2 | Laptops: 716–953 rows flagged for categories ("notebook" vs "budget/gaming…") | The spec's category vocabulary differs from the data's | Same mismatch rule as #1 | same |
| 3 | Laptops: **192 rows wrongly merged** as near-duplicates | Different models with identical spec columns at different prices; the signature ignored the model name | Signature includes extras shared by ≥ 2 sources. Merge only if targets agree within 5%; otherwise keep both and flag | 4 dedup tests |
| 4 | `256GB SSD + 1TB HDD` read as 256 GB; "kg" vs "kgs" reported as mixed units | Parser read only the first number; no canonical unit spelling | `A + B` summed when units are compatible; canonical units | parser probes |
| 5 | Delhi: **locality, property type and furnishing became null** (1,254 values) | Stage 1 (Muse Ember) declared *every* feature as float in one call, a real LLM error | Stage 4 checks types against the data (`looks_numeric`: "Rohini Sector-25" contains a number but isn't one) and reports the correction | `test_spec_dtype_contradicted_by_data_is_corrected_and_reported` |
| 6 | Pipeline crashed in Stage 7 after "needs review" | No stop at the gate | The runner stops at `needs_review` unless `--continue-on-review` (recorded in `run_summary.json`); Stage 7 needs ≥ 30 rows | — |
| 7 | Empty, never-sourced features labelled "near-constant" | Wrong reason text | "no values: not available from any selected source" | — |

## Results after the fixes
| | Laptops | Delhi houses |
|---|---|---|
| Stage 2 | Delhi file rejected as off-topic (1/13 features) | 8 features mapped (`BHK` → bedrooms through the LLM) |
| Stage 4 | 1,302 → 1,236 rows (exact −29, merged −37, 23 conflicting pairs kept and flagged) | 1,259 → 1,155 (exact −83, merged −21); 3 types corrected from data |
| Stage 5 | **needs_review**: the price unit doesn't match the spec, and 2 category vocabularies don't. 0 rows excluded | **passed**: 0 excluded; outliers flagged, not deleted |
| Stage 7 top features | RAM (high), processor, category, GPU, weight | built-up area (high), bathrooms, locality |
| Leakage | none (correct: the data has no price-derived column) | **`Per_Sqft` → leakage**, caught by the semantic (LLM) rule |
| Holdout R², selected | **0.751** | **0.848** |
| Holdout R², all columns | 0.749 | 0.902 (inflated by the `Per_Sqft` leak) |
| CV R² (5-fold, train) | 0.817 ± 0.031 | 0.511 ± **0.445** |
| Strategies | all equal: no context data loaded | all equal: no context data loaded |

## What this shows for the evaluation
- **E10 (leakage):** confirmed on real data. Real `Per_Sqft` is *not* exactly Price ÷ Area: the ratio's middle 50% runs from 0.87 to 1.26, probably carpet vs super area. So the structural pair rule (log R² 0.69 < 0.98) correctly did not fire, and **the semantic rule caught it**. With one rule alone, this leak would have gone through. Including it raises holdout R² from 0.848 to 0.902, a false gain the system avoids.
- **The gate works on real problems:** the laptop run is held for review because the price currency is genuinely unknown. That is the correct outcome, not a failure.
- **LLM output needs data-side checks:** 1 of about 30 cached Stage 1 specs had every type wrong. Checking the output's *shape* (Pydantic) could not catch it; checking against the *data* did.

## Open issues found (not fixed yet)
1. **Heavy-tailed targets.** Delhi CV R² varies hugely (± 0.44) because a few very expensive properties dominate a fold. The standard remedy is to model log(target) for strictly positive skewed regression targets. That is a design decision, so it is listed here.
2. **Messy locality strings** ("J R Designers Floors, Rohini Sector 24") give high MI (1.00) but little model value (0.02): there are too many unique values. Extracting the locality part, or an alias table, would help, and is also needed for context joins.
3. **No context data yet.** Metro distance and circle rates are "derivable" in Stage 2 but have no data until Stage 3 acquisition (OSM / Nominatim) exists, so the three strategies currently tie.
4. **The Delhi file has no licence** and can't be redistributed or used for publishable results; get the Kaggle original with its licence.
5. **The laptop price currency** must be confirmed by a person, or set in config, before the data is delivered.
