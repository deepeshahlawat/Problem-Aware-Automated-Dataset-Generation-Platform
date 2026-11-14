# Stage 2: Source discovery, trust ranking, feasibility report

**Code:** `src/dsagent/stages/discovery/` (`catalogs.py`, `matching.py`, `__init__.py`) · trust: `src/dsagent/trust/` · tests: `tests/test_discovery.py` · live dry run: `eval/run_discovery.py "<any problem>"`

## The system is generic, not Delhi-specific
Nothing in the code knows about houses, Delhi or laptops. Everything is driven by the `DatasetSpec` from Stage 1:

| Step | Generic mechanism |
|---|---|
| Search | Queries come from the problem text: short noun phrases such as "house price" or "laptop price". |
| Region | Applied as a filter, not baked in. A source must match the spec's region, or the region must be unset. |
| Relevance | A source must contain the target **and** at least max(2, 25%) of the other spec features. A table with only a `price` column is rejected as off-topic. |
| Columns | Spec features are mapped to the source's *real* columns by name, then by one cheap LLM call for leftovers. The LLM cannot invent a column. |
| Context | Each context source declares which features it `provides` and which join keys it needs. It is used only for features the entity sources lack, and only if its join key is available. |

The only region-specific item is one **curated registry entry** (`configs/context_sources.json`): Delhi circle rates, an official local source. That is the intended pattern: local official sources are plug-in entries, while OSM transit, amenities and geocoding are region-agnostic and work for any city.

### Evidence from two unrelated live runs (2026-11-14)
| Problem | Outcome | Cost |
|---|---|---|
| "I want to predict house prices in Delhi" | Hugging Face has house-price data for Boston, Taiwan, Dublin and Multan, but none for Delhi. All are rejected as **region mismatch**. Result: **not deliverable**, target not available. No data fabricated. | $0.002 |
| "Predict the price of a laptop from its specifications" | Found `Ammok/laptop_price_prediction`, which covers the target + 10/13 features at 100%. Crude-oil, fuel, AI-compute and luxury-goods tables were rejected (target missing, or off-topic). Result: **deliverable**. | $0.004 (re-run: $0, cached) |

Raw outputs: `eval/results/discovery_*.json`.

The Delhi result is correct behaviour: public catalogs have no Delhi listing data. The Delhi case therefore relies on the hand-downloaded Kaggle file registered in `data/sources.json` (see `data/sources.example.json`). This is exactly the decision D4 path.

## Pipeline
1. **Catalogs** (`catalogs.py`):
   - `LocalManifestCatalog`: hand-downloaded files.
   - `CuratedContextCatalog`: known context sources.
   - `HuggingFaceCatalog`: live Hub search plus the first rows from datasets-server.

   Every HTTP request counts against the page budget. Catalog failures become warnings and never crash the run.
2. **Entity scoring:** region check → column matching → target present? → topic overlap → sample coverage → trust (spec 13.1). Sources whose robots rules or licence forbid reuse are **excluded**.
3. **Complementary selection:**
   - up to 3 entity sources with trust ≥ 0.35;
   - then context sources for uncovered features, with one level of join-key dependency (e.g. geocoding supplies coordinates for metro distance).
4. **Feasibility report:** per feature, found / partial / derivable / not_found, with sources, row-weighted coverage, newest date and trust. Blocking conditions:
   - target missing;
   - a required feature not found;
   - fewer rows than `min_rows`.

   This is the human approval point before acquisition.

## Definition of done
| Item | Status |
|---|---|
| Typed I/O | `DatasetSpec` → `DiscoveryResult` (candidates, selected, rejected with reasons, `FeasibilityItem`s) |
| Unit tests | 23 tests (trust, matching, each catalog with stubbed HTTP, end to end, blocking rules) |
| Fixture example | 3-row synthetic CSV in the tests (clearly a test fixture, never used as data) |
| Budget | LLM cost + HTTP page budget |
| Documented | this file |

## Known limitations / next steps
- **Units and currency are not checked here.** The laptop dataset's price may be in EUR while the spec says `price_usd`. That is Stage 4 normalisation; the feasibility report should flag it once Stage 4 exists.
- More catalogs: OpenML (global, free API, tabular ML datasets) is the obvious next generic one. Kaggle and data.gov.in need API keys.
- Context features are matched by name (`provides`). A context source whose names differ from the spec's (e.g. `amenities_count` vs `schools_nearby`) is missed. A cheap-LLM mapping step, like the one for columns, would close this.
- Hugging Face sometimes returns HTTP 500/401 for the column preview; those sources are skipped with a warning.
