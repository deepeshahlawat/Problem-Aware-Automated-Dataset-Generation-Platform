# Research Findings and Design Lessons (pre-build)

Date: 2026-11-14. Purpose: learn from published wins and failures before locking the design in `README (1).md`.
Method: about 20 sequential web searches and paper fetches. Claims marked **[judgement]** are our inference, not a published result. Claims marked **[unverified]** could not be confirmed from a primary source.

---

## Key findings (most important first)

1. **We misread AutoData's "hypergraph cache".** In AutoData (arXiv 2505.15859), the hypergraph is an *oriented message hypergraph between agents*: nodes are agents, and each hyperedge is (source agent → set of target agents). The "local cache" stores artifacts such as HTML and passes cache IDs around. It is **not** a cache of reusable parsers keyed by layout and schema, which is what README §12.3 describes. Our idea is still good, but it must not be presented as an adaptation of AutoData. A closer precedent is AutoCrawler's *reusable generated rules*.
2. **LLM-generated scrapers fail often.** AutoCrawler with GPT-4 produced a fully correct crawler for only **71.6%** of SWDE tasks; GPT-3.5 managed 54.8% and small open models less than 47%. The main failure causes are rules that don't generalise across pages of the same site, fragile XPath, and missed multi-valued fields. So every cached parser needs per-run validation and a fallback.
3. **Live scraping of Indian property portals is impractical for us.** WebFetch could not read either portal's robots.txt: MagicBricks refused the fetch and 99acres returned HTTP 403. AutoData's own limitations list anti-bot measures, CAPTCHAs and JS-rendered sites as out of reach, and it assumes the data is legally retrievable. **Drop portal scraping from the MVP.**
4. **Simple join and selection beats clever join and selection on cost** ("Retrieve, Merge, Predict", arXiv 2402.06282):
   - Exact-match retrieval is Pareto-optimal.
   - Full Join gives +2.1% at lower cost than greedy.
   - Stepwise greedy selection gives +1.6% at about 2× runtime.
   - Gains plateau at about 8 candidates.
   - Tree models (CatBoost) handle the missing values that joins create.

   **Simplify the METAM-style loop.**
5. **An LLM can choose features from names alone.** LLM-Select (Jeong et al., TMLR) finds GPT-4 feature selection competitive with LASSO **without seeing the data**. This is direct support for Stage 1 (deciding what to collect) and for the semantic relevance signal. CAAFE's gain is real but modest: AUC 0.798 → 0.822, with 11 wins out of 14 datasets. It depends strongly on the model: GPT-3.5 improved only 6 datasets.
6. **Leakage is the most common way projects like ours fool themselves.** Kapoor and Narayanan document leakage in **294 papers across 17 fields** and give an 8-type taxonomy. Our own primary dataset contains a ready-made leak: the Kaggle Delhi MagicBricks file has `Per_Sqft`, which is price ÷ area.
7. **Temperature 0 is not deterministic** (arXiv 2408.04667, 2506.09501). Reproducibility must come from replaying the LLM call cache, not from re-calling the model. The README already plans this, so keep it.
8. **Use the simplest thing that works.** Anthropic's "Building Effective Agents" says to prefer fixed *workflows* over autonomous agents and to start simple. It warns that frameworks hide prompts and make debugging harder. The MAST study (1,600+ traces, 14 failure modes) groups failures into system design, inter-agent misalignment and task verification. **Our state-machine design is already a workflow, which is the right call. Remove the "multi-agent" framing where it isn't real.**

---

## Details by sub-topic

### A. Closest existing systems
- **AutoData** (instruction → web dataset):
  - Architecture: 8 agents in two squads (plan, web, tool and blueprint agents; engineering, test and validation agents) plus a manager.
  - Instruct2DS F1: Academic 91.9, Stock 96.8, Sport 90.1, at $0.57 per task, beating Manus and OpenManus.
  - On SWDE its F1 (89.25) is essentially tied with Manus (89.22).
  - Reported failures: it mis-parses attributes when pages lack explicit labels, and it sometimes picks the less reliable route, crawling HTML instead of the cleaner API.

  → *Lessons:* (a) a **fixed cheapest-first ladder** (API before HTML) is better than letting an agent choose; (b) AutoData generates a *program* once, not per-page LLM extraction, which confirms "write the parser once".
- **METAM** (ICDE 2023): uses downstream-task utility as feedback for choosing augmentations. It assumes an existing data lake and a join key. A poster reports fewer than 50 iterations over more than 10K candidates **[unverified: poster, not paper]**.
- **CAAFE**: LLM feature engineering that keeps a feature only if CV improves. The gains are modest and depend on the model.
- **Novelty check [judgement]:**
  - AutoData = generic web collection, with no ML-target awareness.
  - METAM = task-guided augmentation, but no natural-language intake and no acquisition.
  - LLM-Select/CAAFE = operate on a given table.

  Combining them (NL problem → feature spec → open-source acquisition → task-guided selection → data card) still looks novel. **The defensible claim is the integration plus the evaluation, not any single component.** We did not do an exhaustive GitHub sweep for "NL → dataset" projects; do one quick manual check before writing the report.

### B. Acquisition and extraction
- AutoCrawler numbers are in finding 2. Its failure categories are worth reusing in our E7 evaluation: Correct, Precision-only, Recall-only, Unexecutable, Over-estimate.
- The cheapest-first ladder (file/API → HTML table → JSON-LD → LLM) is consistent with every source found. **Keep it.**
- A cached parser can go stale. Run validation on every reuse and fall back to regeneration (README §12.3 step 3 already says this). **Keep.**

### C. Schema matching and entity resolution
- **Parciak et al. (2407.11852):** LLM matching quality drops with too *little* **and** too *much* context. Newer models commit to match decisions more readily.
- **Magneto (2412.08194):** uses a cheap small-model retriever and an LLM reranker. It's designed for large lakes with thousands of columns.
- **Entity matching:** GPT-4 zero-shot beats fine-tuned matchers on unseen entities (Peeters and Bizer, 2310.11244). Batch prompting cuts cost, and GPT-3.5 is within 6.4 F1 at about 1/10 the cost (Fan et al., 2312.03987).
- **[judgement] Scale mismatch:** we will merge 2–4 sources with roughly 10–30 columns each, a few hundred column pairs at most. One batched LLM call with 5–10 sample values per column, plus a cheap string/embedding pre-filter, is enough. **FAISS and fine-tuned retrievers are overkill.**
- **[judgement] Entity resolution is probably ill-defined for our data.** Different listing datasets describe *different listings*, and there is no shared key. Trying to "link the same flat" across Kaggle files will mostly produce false merges. The realistic integration is:
  1. **Vertical union** of listing tables after schema matching, with exact plus MinHash near-duplicate removal for re-posted listings.
  2. **Horizontal enrichment** by joining on locality or coordinates: circle-rate category, distance to metro, amenity counts from OSM. This is exactly the "exact-match join" that Retrieve-Merge-Predict found best.

  → Splink becomes optional. Keep it only if a real overlapping-entity case appears.

### D. Feature relevance and the selection loop
- LLM-Select validates the semantic signal (finding 5).
- **[judgement]** Fusion weights of 0.4/0.3/0.3 have no published basis. Keep them as configurable defaults, but justify them through an **ablation** (stat only, model only, LLM only, fused) instead of asserting them.
- SHAP and permutation importance split credit between correlated features, and ID-like high-cardinality columns can look important. Keep the explicit ID rule (unique ratio > 0.95) and the leakage rules. **[general knowledge, not re-verified in this pass]**
- Greedy keep-if-CV-improves selection tunes itself to the validation folds. Mitigate with **nested CV or a held-out selection split**, plus an untouched final test split (README §16.4 already requires this).

### E. Orchestration, reproducibility, quality
- Workflow, not agents (Anthropic). Keep the LangGraph state machine thin, or start with plain Python functions and wrap them in LangGraph only when the stages are stable (README rule 7 already says not to wire stages before their tests pass).
- **Non-determinism:** cache-replay reproducibility; run 3–5 end-to-end repeats and report mean ± sd (already in README §18).
- **Pandera** has native Polars support since 0.19. Great Expectations did not, as of mid-2025. **→ Use Pandera only; drop GX.**

### F. Data sources for the case studies
| Use | Source | Notes |
|---|---|---|
| Delhi listings (primary) | Kaggle "MagicBricks.csv", Delhi, **1,259 rows × 11 cols**: Area, BHK, Bathroom, Furnishing, Locality, Parking, Price, Status, Transaction, Type, Per_Sqft | Already a MagicBricks scrape, so portal ToS still matter for redistribution. Licence not confirmed. `Per_Sqft` = leakage test case. |
| Delhi listings (second) | Other Kaggle Delhi/NCR MagicBricks sets; Makaan rentals (5,000 × 16, rental, not sale) | **[unverified]** schemas; several may be re-uploads of the same scrape, so check overlap before treating them as independent. |
| Metro distance | OpenStreetMap Overpass: `railway=station` + `station=subway` / `network=Delhi Metro` | Open (ODbL). Derived feature, `method=derived`. |
| Locality price context | Delhi circle rates, colony categories A–H (revenue.delhi.gov.in); 2014 notification reportedly still in force | Official, so authority 1.0. Probably PDF only, so it's a good test of the PDF/HTML-table extractor. Same colony name can sit in different categories in different districts. |
| Amenities | OSM POIs (schools, hospitals, parks) by locality | Open. |
| Laptop prices (case 2) | Kaggle laptop price dataset (Muhammet Varlı), **1,303 rows**, ~11–13 cols incl. Company, TypeName, Ram, Cpu, Gpu, Price_euros | Old (≈2018) prices in EUR, so the recency weighting and currency normalisation stages have real work to do. |
| Avoid in MVP | Live scraping of 99acres / MagicBricks / Housing.com | Bot-blocked; ToS not verified; legal risk. |

---

## Recommendations (prioritised)

**Change**
1. Rename the "hypergraph cache" to a **recipe cache** (parser recipes + field-mapping recipes). You may still store them as hyperedges, but cite AutoCrawler for reusable rules and describe AutoData accurately.
2. Replace "entity resolution across listings" with **union + near-duplicate removal + locality/geo enrichment joins**. Make Splink optional.
3. Simplify schema matching to **string/embedding pre-filter + one batched LLM call with sample values**. Drop FAISS.
4. Simplify the feedback loop. Compare (i) base table, (ii) full join of all candidate enrichments, and (iii) greedy keep-if-improves, all under nested CV. Report all three. This *is* a publishable ablation.
5. Re-frame acquisition: the MVP sources are **open files + OSM API + one government PDF/HTML table**. Show LLM extraction on one permissive page as a demonstration, not as the data backbone.

**Keep**
- Cheapest-first ladder; validate-on-reuse parser cache; Pydantic structured outputs.
- Human approval of spec and feasibility; never invent data; provenance on every row.
- Trust scoring; recency weighting; leakage and ID rules; LLM call cache for replay.
- Pandera validation; data card.

**Drop or defer (MVP)**
- Great Expectations, FAISS/Qdrant, Splink, Playwright, Crawl4AI.
- Celery+Redis (use FastAPI background tasks), PostgreSQL and MinIO (DuckDB + local Parquet), MLflow (use `run_summary.json`).
- MCP server: defer until the end, then wrap the existing tool functions.

**Evaluation additions**
- Use AutoCrawler's outcome categories for extraction runs.
- E4 should include *base table vs full join vs greedy* (Retrieve-Merge-Predict style).
- E1 can use LLM-Select's framing: LLM-chosen features vs LASSO/MI-chosen features.
- Add a **leakage test**: the system must flag `Per_Sqft` as leakage for target `Price`.
- Use CatBoost/LightGBM as the main baseline model (tree models handle missing join values).

---

## Open questions / lower confidence
- Licences of the Kaggle Delhi datasets, and whether several are the same scrape. Check on Kaggle directly.
- Exact current Delhi circle-rate notification and its format. Check revenue.delhi.gov.in.
- MagicBricks/99acres terms of use were not read directly (pages blocked automated fetch).
- AutoData cache ablation numbers appear only in a figure, so they're not quoted here.
- METAM "<50 iterations" figure comes from a poster.
- No exhaustive search for GitHub projects doing NL → ML dataset; a quick manual check is still advised.

## Sources
- AutoData: https://arxiv.org/abs/2505.15859 · full text https://arxiv.org/html/2505.15859
- AutoCrawler: https://arxiv.org/html/2404.12753v1
- METAM: https://arxiv.org/abs/2304.09068 · poster https://www.nitrd.gov/documents/2022/CiFellows-posters/CIFellows-2020-2021-Sainyam-Galhotra.pdf
- Retrieve, Merge, Predict: https://arxiv.org/abs/2402.06282 · https://arxiv.org/html/2402.06282
- LLM-Select: https://arxiv.org/abs/2407.02694
- CAAFE: https://arxiv.org/abs/2305.03403 · https://neurips.cc/virtual/2023/poster/72592
- Leakage and the Reproducibility Crisis: https://reproducible.cs.princeton.edu/ · https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856
- Schema matching with LLMs (Parciak et al.): https://arxiv.org/abs/2407.11852
- Magneto: https://arxiv.org/abs/2412.08194
- Entity matching with LLMs (Peeters & Bizer): https://arxiv.org/abs/2310.11244 · https://arxiv.org/abs/2305.03423v2
- Cost-effective in-context ER (batching): https://arxiv.org/pdf/2312.03987
- MAST, Why do multi-agent LLM systems fail: https://arxiv.org/abs/2503.13657
- Building Effective Agents: https://www.anthropic.com/engineering/building-effective-agents
- Non-determinism of deterministic LLM settings: https://arxiv.org/html/2408.04667v5 · https://arxiv.org/pdf/2506.09501v2
- Validation libraries 2025 (Pandera/GX Polars): https://opensource.posit.co/blog/2025-06-04_validation-libs-2025/ · https://github.com/unionai-oss/pandera/discussions/598
- Delhi MagicBricks dataset description: https://sukhman-singh-1612.github.io/data_science/dhouseprice.html
- Delhi circle rates: https://housing.com/news/new-delhi-circle-rate/?amp=1 · https://web.landeed.com/products/delhi/circle-rate
- OSM Delhi Metro: https://wiki.openstreetmap.org/wiki/Delhi_Metro
- Laptop dataset: https://scitepress.org/PublishedPapers/2024/128223 · https://github.com/Rishabhrv/Laptop-Price-Predictor
