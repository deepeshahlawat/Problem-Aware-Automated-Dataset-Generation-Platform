# Problem-Aware Automated Dataset Generation Platform Using Agentic AI

B.Tech Computer Science (2023-2027) | Minor Project
Team: Palak Mishra, Nitya Jamdagni, Deepesh Ahlawat | Section CS, F(14)
Supervisor: Ms. Tusha Gupta, Assistant Professor
Institute: Dr. Akhilesh Das Gupta Institute of Professional Studies, New Delhi

---

## 0. How to use this document

- **Teachers / reviewers:** read sections 1-4 (summary, synopsis, novelty, scope) and section 18 (evaluation).
- **Coding agents / developers:** read everything. Sections 5-17 are the build specification. Section 24 contains the rules you must follow. Section 22 gives the build order.
- Libraries are named without versions on purpose. Pin the current stable versions when you create `pyproject.toml`.
- Numeric defaults (weights, thresholds, half-lives) are starting values to be tuned during evaluation, not proven constants.
- Pre-build research behind the decisions below: `docs/RESEARCH_DESIGN_LESSONS.md`.

### 0.1 Design decisions log (locked 2026-11-14, after literature review)
| # | Decision | Why (evidence) | Sections |
|---|---|---|---|
| D1 | "Hypergraph cache" renamed **recipe cache**; it is our own design, not AutoData's | AutoData's hypergraph routes agent messages, not parsers. Reusable generated rules follow AutoCrawler, where only ~72% of GPT-4 crawlers were fully correct, so every reuse is validated | 12.3 |
| D2 | Integration = **stack entity sources + MinHash dedupe + locality/geo enrichment joins**; Splink optional | Listing datasets have no shared key and mostly different properties. Simple exact-match joins are the best cost/benefit option (Retrieve-Merge-Predict) | 10 (Stage 4) |
| D3 | Schema matching = string/embedding pre-filter + one batched LLM call; no FAISS | Our scale is tens of columns. Too much or too little context hurts LLM matching (Parciak et al.) | 10 (Stage 4), 6 |
| D4 | MVP acquisition = open data + OSM + government tables; live extraction shown on one permissive site; no portal scraping | Portals block bots and have unverified terms; AutoData also lists anti-bot and JS sites as out of reach | 10 (Stage 3) |
| D5 | Source selection compares **S-base / S-full / S-greedy** under nested CV | Full join ≈ greedy at lower cost (Retrieve-Merge-Predict); greedy selection overfits its folds | 16.4, 18 |
| D6 | Lean MVP stack; Celery/Redis, PostgreSQL/MinIO, MLflow, FAISS, Splink, Playwright, Crawl4AI and Great Expectations deferred or dropped; MCP added last | "Start with the simplest solution" (Anthropic, Building Effective Agents); Pandera supports Polars natively | 6 |
| D7 | Added leakage test (E10) and relevance-signal ablation (E11) | Leakage found in 294 papers (Kapoor & Narayanan). LLM-only feature selection is competitive with LASSO (LLM-Select) | 18 |
| D8 | Reproducibility = replay from the LLM call cache plus 3-5 repeated runs | Temperature 0 is not deterministic on hosted LLMs | 12.2, 18 |

---

## 1. Project summary

Machine learning practitioners spend most of their time on data preparation: deciding what data is needed, finding it, acquiring it from scattered sources, merging it, and cleaning it. Existing tools solve only fragments of this. Dataset search engines find existing datasets but do not build new ones. AutoML starts only after a clean table exists. Scrapers fetch raw content but do not know what modelling problem the data is for.

**This project builds an automated data engineer for the pre-training phase of ML.**

Input: a plain-English modelling problem, for example "I want to predict house prices in Delhi."

Output: a validated, documented, ML-ready dataset, plus a feature-documentation file, a column-level provenance record, a data card, and a quality report.

What makes it different: it is **problem-aware**. It decides which features the problem needs, collects only those, scores how trustworthy and how recent each source is, and uses the performance of a quick baseline model to decide which sources and columns are worth keeping.

---

## 2. Synopsis

### 2.1 Problem statement
Before training a model, a practitioner must manually determine required features, identify sources, extract data, reconcile schemas, remove duplicates and conflicts, handle missing values, and judge which attributes to keep. This is slow, repetitive, error-prone, and must be repeated for each new domain. The correct dataset is also problem-dependent: collecting everything produces bloated tables full of identifiers, usernames and URLs that carry no predictive value and can degrade models.

### 2.2 Aim
Design and develop an agentic, problem-aware platform that automates data collection, integration and preparation for ML workflows and generates a validated, documented, ready-to-use dataset from a natural-language problem description.

### 2.3 Objectives
1. Requirement analysis module: natural-language problem to structured spec (target, task type, features, constraints).
2. Source discovery module: identify and rank heterogeneous sources containing required features.
3. Tool-enabled acquisition agents: extract structured records from APIs, files, open repositories and web pages.
4. Integration engine: schema matching, normalisation, union of entity sources, de-duplication, enrichment joins from context sources, missing-value handling (entity resolution only where sources truly share entities).
5. Feature relevance engine: statistical, model-based and semantic evidence to separate predictive features from metadata and leakage.
6. Automated data-quality validation with machine-checkable expectations.
7. End-to-end provenance: every column and transformation traceable to its origin.
8. Final dataset in standard formats with data card, feature documentation and quality report.
9. Evaluation: compare models trained on generated datasets against models trained on manually prepared benchmarks.

### 2.4 Research questions
- Can a natural-language modelling problem be reliably translated into a correct and sufficient feature specification?
- To what extent can schema matching across heterogeneous sources be automated with embeddings plus LLM adjudication?
- Does problem-aware, model-guided feature and source selection beat indiscriminate collection?
- How much manual preparation effort (time, number of interventions) does the automation remove?
- What are the reliability, cost and correctness limits of agentic automation here, and where is human review still required?

---

## 3. What is novel (positioning)

| Existing work | What it does | Gap this project addresses |
|---|---|---|
| AutoData (multi-agent web data collection) | NL instruction to dataset from the web; oriented message hypergraph between agents; generates a collection program once | Not focused on tabular ML relevance, trust scoring, or model-guided selection |
| METAM (goal-oriented data discovery) | Uses the downstream task as feedback to steer discovery and augmentation | Starts from an existing data lake; no NL problem intake, no acquisition |
| Auctus (dataset search) | Join and union search over open datasets | Finds datasets; does not build or validate one for a stated problem |
| Harmonia (agentic data harmonization) | LLM-assisted harmonization pipelines | Harmonization only; no discovery or relevance engine |
| CAAFE (LLM feature engineering) | Generates features, keeps them only if CV improves | Works on an existing dataset; no sourcing |
| Magneto / ReMatch (LLM schema matching) | Cheap retrieval then LLM rerank for column matching | A component, not an end-to-end system |

**Core contribution (the claim to defend):** *goal-directed dataset construction.* A problem-aware pipeline that (a) turns NL into a feature spec, (b) acquires from complementary sources using cheapest-extractor-first, (c) scores source trust and record recency, and (d) uses baseline-model feedback to keep only sources and columns that improve the downstream task, with full provenance and a feasibility report when data is missing.

---

## 4. Scope

**In scope:** tabular supervised learning (regression/classification); publicly accessible sources; two or three case studies (house prices in Delhi, laptop prices, optionally employee attrition); a web UI for job submission and preview.

**Out of scope:** real-time streaming, licensed commercial data, very large distributed collection, image/text modalities (future work).

**Future work:** image/text modalities, scheduled refresh with drift detection, synthetic augmentation, AutoML handover, community dataset catalogue, bias auditing.

---

## 5. System overview

```
 User (React UI) --problem text, optional sample file--> FastAPI
                                                           |
                                                  background job       
                                                           |
        +----------------- LangGraph orchestrator (state machine) -----------------+
        |                                                                           |
        |  1 Requirement   2 Source        3 Acquisition   4 Integration            |
        |    analyser  ->    discovery ->    agents    ->    & cleaning             |
        |       |               |   ^            |               |                  |
        |   [human          [feasibility    cheapest-first   schema match,          |
        |    approves]        report]       extractor        union, dedupe, enrich  |
        |                                                        |                  |
        |  8 Output  <-  7 Relevance engine  <-  5 Validation  <-+                  |
        |                    |   (feedback loop: retrain, keep source if better)    |
        |                    +------> back to 2/3 when more data is needed          |
        +---------------------------------------------------------------------------+
                 |                         |                          |
          MCP tool server          Caches (LLM call cache,     Storage + provenance
          (web, API, files,         recipe cache)              (DuckDB, Parquet,       
           profile, validate)                                   PostgreSQL, lineage log)
```

---

## 6. Technology stack and why each tool is used

**Locked MVP stack (2026-11-14):** Python, LangGraph (kept thin), Pydantic, LLM SDK via wrapper, httpx, Polars, DuckDB + local Parquet, Pandera, datasketch, scikit-learn + LightGBM/CatBoost + SHAP, FastAPI (background tasks), React + Tailwind. Rows marked *(deferred)* below are post-MVP and added only if a concrete need appears.

| Layer | Tool | Why |
|---|---|---|
| Orchestration | **LangGraph** (Python) | Explicit state machine: typed state, retries, budgets, checkpoints, human-approval pauses. Auditable and reproducible, unlike a free-running agent loop. |
| Tool interface | **MCP** (use FastMCP / official Python SDK); *added late in the build* by wrapping the existing tool functions | Standard way for agents to call tools. New sources become new tools without changing agent logic; reusable by any MCP client; one place for permissions, rate limits, logging. Keep thin: it is the interface, not the product. |
| LLMs | Claude / GPT-class via a thin client wrapper (model name from env var) | Requirement analysis, extraction fallback, schema adjudication, semantic relevance, data-card text. Use structured outputs. Use a cheaper model for adjudication, a stronger one for planning. |
| Schemas / contracts | **Pydantic** | Fixes the shape of LLM output; validates and enables automatic retry on malformed output. |
| HTTP / APIs | **httpx** | Async API and file fetching with timeouts and headers. |
| Dynamic pages | **Playwright** (deferred) | JavaScript-rendered pages. |
| Clean text | **Trafilatura** (or readability) | Reduces HTML to clean text before any LLM call, saving tokens. |
| Structured markup | **extruct** | Reads JSON-LD / microdata deterministically, no LLM. |
| LLM-friendly crawling | **Crawl4AI** (deferred) | Convenience crawling for LLM pipelines. |
| Data processing | **Polars, DuckDB, PyArrow** | Fast in-process columnar analytics, no server needed. |
| Embeddings | **sentence-transformers** | Column and record embeddings for matching. |
| Vector search | **FAISS** (or Qdrant) *(deferred)* | Not needed at our scale: a few hundred column pairs fit one batched LLM call. |
| Dedupe / linkage | **datasketch** (MinHash LSH); **Splink** *(deferred, optional)* | Near-duplicate detection for re-posted listings; Splink only if a real cross-source entity case appears. |
| Validation | **Pandera** (Great Expectations dropped: no native Polars support as of 2025) | Declarative, machine-checkable data contracts. |
| Profiling | **ydata-profiling** | Quality report and statistics. |
| Feature analysis | **scikit-learn, LightGBM, SHAP** | Mutual information, baseline model, permutation importance, SHAP. |
| Backend | **FastAPI** (background tasks); Celery + Redis *(deferred)* | REST API; a single-user demo does not need a task queue. |
| Storage | **DuckDB + local Parquet**; PostgreSQL, MinIO/S3 *(deferred)* | Metadata, artifacts, columnar datasets. |
| Cache | **SQLite or DuckDB tables** | LLM call cache and recipe cache (section 12). |
| Provenance | OpenLineage-style JSON event log (simple append-only table is enough) | Column-level lineage. |
| Experiment tracking | `run_summary.json`; **MLflow** *(deferred)* | Log runs, scores, token cost. |
| Frontend | **React + Tailwind CSS** | Job submission, feasibility approval, progress, preview. |
| Deployment | **Docker, Docker Compose** | Reproducible local and cloud runs. |

---

## 7. Design principles

1. **Cheapest extractor first.** API/file, then HTML table, then JSON-LD, then LLM. LLM is the last resort.
2. **Schema as contract.** Every LLM output is constrained by a Pydantic schema and validated. Invalid output is retried, then rejected.
3. **Never invent data.** Missing means missing. Imputation is documented per column, never silent.
4. **Provenance on everything.** Every record carries source URL, retrieval timestamp, extraction method, and source trust.
5. **Explicit state machine.** No free-running loops. Every node has typed input/output, retry policy, budget, validation gate.
6. **Human checkpoints.** The user approves the feature spec and the feasibility report before expensive acquisition.
7. **Reproducibility.** All LLM calls cached; every run replayable from the cache and the provenance log.
8. **Budgeted.** Every run has caps on tokens, pages, and money. Exceeding a cap stops the run with a report.
9. **Advisory automation.** Relevance and trust scores are shown with reasons; the user keeps final judgement.

---

## 8. Data contracts (Pydantic models)

```python
from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime

class Feature(BaseModel):
    name: str
    dtype: Literal["int", "float", "category", "text", "date", "bool"]
    unit: Optional[str] = None
    description: str
    required: bool = True            # critical features block delivery if missing
    expected_range: Optional[tuple[float, float]] = None
    allowed_values: Optional[list[str]] = None

class TemporalPolicy(BaseModel):
    mode: Literal["recent_weighted", "window", "uniform"] = "recent_weighted"
    half_life_years: float = 2.0
    window_start: Optional[int] = None   # year, used when mode == "window"
    window_end: Optional[int] = None

class DatasetSpec(BaseModel):
    problem_text: str
    task_type: Literal["regression", "classification", "ranking"]
    target: Feature
    features: list[Feature]
    region: Optional[str] = None
    temporal: TemporalPolicy = TemporalPolicy()
    min_rows: int = 1000

class SourceCandidate(BaseModel):
    source_id: str
    url: str
    kind: Literal["api", "file", "open_repo", "html_table", "jsonld", "web_page"]
    covers_features: list[str]
    est_rows: Optional[int]
    last_updated: Optional[datetime]
    licence: Optional[str]
    robots_allowed: bool
    trust_score: float                 # section 13
    notes: str = ""

class Provenance(BaseModel):
    source_id: str
    source_url: str
    retrieved_at: datetime
    method: Literal["api", "file", "html_table", "jsonld", "llm", "derived"]
    llm_verified: Optional[bool] = None   # value found in source text?
    parser_id: Optional[str] = None       # recipe cache parser version

class FeasibilityItem(BaseModel):
    feature: str
    status: Literal["found", "partial", "derivable", "not_found"]
    sources: list[str]
    coverage: float                     # fraction of rows with a value
    newest_date: Optional[datetime]
    trust: float
    note: str = ""

class ColumnRelevance(BaseModel):
    column: str
    stat_score: float
    model_score: float
    semantic_score: float
    fused_score: float
    tier: Literal["high", "medium", "low", "metadata", "leakage", "dropped"]
    reason: str
```

---

## 9. Orchestration (LangGraph)

**State** (single typed object passed between nodes): `job_id, problem_text, spec, sources, feasibility, approvals, raw_artifacts, unified_table_id, validation_report, relevance, iteration, budget{tokens_used, tokens_max, pages_used, pages_max, cost_used, cost_max}, provenance_log, errors, status`.

**Nodes and edges**

```
parse_problem -> plan_sources -> feasibility_report -> [HUMAN APPROVAL: spec + sources]
   -> acquire (fan-out per source) -> integrate -> validate
   -> relevance_and_feedback --(improvement found / more sources needed, budget left)--> plan_sources
                              \--(done)--> generate_outputs -> END
Any node --(fails after retries)--> try_alternative_source | abort_with_report
```

**Per-node rules:** typed inputs/outputs; max 2 retries with backoff; per-node token/time budget; validation gate at exit (a node cannot pass bad output forward); checkpoint to storage so a crashed run resumes. Use LangGraph interrupts for human approval.

**Stop conditions for the feedback loop:** max sources reached, budget exhausted, or improvement below epsilon twice in a row.

---

## 10. Pipeline stages in detail

### Stage 1: Requirement analyser
- **Input:** problem text, optional sample file or schema.
- **Process:** LLM with structured output (`DatasetSpec`). Proposes target, task type, candidate features with dtype/unit/description, geographic and temporal constraints, and a recency half-life by domain (fast-changing prices: about 2 years; medium: about 5; slow structural data: about 15).
- **Output:** `DatasetSpec`, shown to the user for approval or edit.
- **Failure handling:** invalid JSON means retry; still invalid means ask the user to rephrase.
- **Test:** compare against 10-20 hand-written expert feature lists (precision/recall).
- **Example (house prices, Delhi):** locality, built-up area, bedrooms, bathrooms, property type, age, floor, furnishing, distance to nearest metro, amenities; target `price`; region Delhi.

### Stage 2: Source discovery and ranking
- **Input:** `DatasetSpec`.
- **Process:** search a **curated allowlist** first (Kaggle datasets, data.gov.in, OpenStreetMap/Overpass, World Bank/Census-style portals, Hugging Face datasets, documented public APIs); use general web search only as a secondary step. Score each candidate (section 13.1). Select a **complementary set** (different sources cover different columns), not one big source.
- **Output:** ranked `SourceCandidate` list and a **feasibility report** (section 15).
- **Tools:** `web_search`, `list_open_datasets`, `fetch_page` (metadata only), embeddings to match feature descriptions to dataset descriptions.
- **Do not** let the LLM freely crawl the whole web.

### Stage 3: Extraction and acquisition
Cheapest-first ladder per source:
1. **API / file** (CSV, JSON, Parquet): deterministic parse with `httpx` + Polars.
2. **HTML tables:** deterministic table parse.
3. **JSON-LD / microdata:** `extruct`.
4. **LLM extraction (last resort):** clean text via Trafilatura, then LLM constrained by target schema returns typed records.

Rules:
- Check the recipe cache first (section 12): reuse a stored parser for a known site layout.
- LLM writes a parser (selectors) **once per layout**, then it is cached and run deterministically.
- Verify every LLM-extracted value appears in the source text; reject otherwise.
- Stamp every record with `Provenance`.
- Obey robots.txt, terms of use, polite rate limiting, identifying User-Agent, domain allowlist.
- Playwright only for JavaScript-rendered pages (deferred, not in the MVP).
- Every source is classified as an **entity source** (stacked) or a **context source** (joined); see Stage 4.

**Acquisition scope (locked 2026-11-14):**
- **MVP data backbone (open data only):**
  - Kaggle/HF dataset files: entity sources.
  - OpenStreetMap Overpass API: metro stations, amenities.
  - Government HTML/PDF tables, e.g. Delhi circle-rate colony categories: context sources.
- **Live web extraction:** demonstrated on **one** site that permits automated access, to exercise steps 2-4 of the ladder and the recipe cache. It is not used as the data backbone.
- **Out of scope:**
  - Live scraping of commercial property portals (99acres, MagicBricks, Housing.com). They block automated access, and their terms are unverified.
  - Login-protected, CAPTCHA-protected or JavaScript-only sites. AutoData lists these as out of reach as well.

### Stage 4: Integration and cleaning
Pipeline: type inference and normalisation, schema matching, **vertical union** of entity tables, de-duplication, **horizontal enrichment** joins, missing-value handling, unified table.

**Design decision (2026-11-14):** different listing datasets mostly describe *different* properties with no shared key, so cross-source entity resolution of individual listings is ill-defined and risks false merges. Sources are therefore treated as one of two kinds:
- **Entity sources:** row-level records of the thing being predicted, e.g. listings. These are stacked.
- **Context sources:** attributes of a shared key such as locality, coordinates or model name, e.g. OSM metro stations, circle-rate categories. These are joined.

1. **Normalisation:** units (sq ft / sq m), currencies, date formats, categorical vocabularies, casing, whitespace.
2. **Schema matching:**
   - A cheap string and embedding pre-filter proposes candidate pairs.
   - **One batched LLM call** confirms or rejects them, seeing column names plus 5-10 sample values per column.
   - Too little or too much context hurts LLM matching (Parciak et al., arXiv 2407.11852), so the prompt stays compact.
   - Must distinguish `price` from `price_per_sqft`.
   - Decisions are cached as field-mapping recipes (12.3).
   - FAISS/Magneto-style retrieval is unnecessary at our scale (2-4 sources, tens of columns).
3. **Vertical union:** stack entity sources onto the canonical schema. Every row keeps its `source_id` and provenance.
4. **De-duplication:** exact duplicates, then near-duplicates via MinHash LSH (datasketch) over a canonical record signature such as normalised locality + rounded area + BHK + price band, to catch re-posted listings. Conflicts within a duplicate group are resolved by **source-trust ordering** (section 13); strong disagreement is flagged.
5. **Horizontal enrichment:** exact-match joins from context sources onto the unified table, by normalised locality name or nearest-neighbour coordinates. Examples: distance to nearest metro, circle-rate category, amenity counts. Locality names go through a normalisation and alias table first; unmatched localities are reported, never guessed. Simple exact-match joins were found to be the best cost/benefit option in Retrieve-Merge-Predict (arXiv 2402.06282).
6. *(Optional)* **Entity resolution with Splink:** only for a case that genuinely contains the same entity in several sources. Not in the MVP.
7. **Missing values:** documented strategy **per column** (median/mode, group-wise median, model-based, or leave null), never one global rule. Record imputation flags in a companion column or the data card.
- **Output:** unified feature table (Parquet) with per-row provenance and confidence.

### Stage 5: Validation
- **Declarative contracts (Pandera):** type, permitted range, nullability, uniqueness, allowed categories, derived from the spec and from profiling.
- **Authenticity checks:** section 13 (cross-source agreement, outliers, extraction verification).
- **Report:** completeness, duplication rate, outliers, failed checks.
- **Gate:** if **critical** expectations fail (target missing, required feature absent, too few rows), the run is surfaced to the user, never silently delivered.

### Stage 6: Recency weighting
Section 14. Applied after validation, before the relevance model is trained.

### Stage 7: Feature relevance engine
Section 16. Combines statistical, model-based and semantic evidence into tiers, removes identifiers/metadata/leakage, and drives the source-selection feedback loop.

### Stage 8: Dataset generator
- Materialise CSV / Parquet / JSON.
- Optional stratified train/validation/test split.
- **Data card** (section 17), **feature documentation**, **quality report**, **provenance/lineage file**.
- Log the run to `run_summary.json` (scores, tokens, cost, cache hit rate).

---

## 11. MCP tool layer

Expose these as MCP tools. The orchestrator and agents call only these, never raw libraries, so permissions, rate limits and logging live in one place.

| Tool | Purpose |
|---|---|
| `web_search(query)` | General search (secondary). |
| `list_open_datasets(query, catalogs[])` | Search Kaggle / data.gov.in / HF / etc. |
| `fetch_page(url)` | Polite fetch; checks robots.txt, allowlist, rate limit; returns HTML + clean text. |
| `fetch_api(url, params)` | Call a documented API. |
| `read_table(path_or_url, format)` | CSV / JSON / Parquet / HTML table to artifact. |
| `extract_structured(content, schema)` | Cheapest-first extraction ladder (consults the recipe cache). |
| `profile_table(artifact_id)` | Statistics, types, missingness. |
| `validate_table(artifact_id, contract)` | Pandera checks, returns report. |
| `store_artifact / get_artifact` | Object store access. |
| `cache_lookup / cache_put` | Recipe cache + LLM-call cache access. |

Benefits: add sources as new tools without touching agent logic; reusable by any MCP-compatible client; central place for permissions and logging. Keep MCP lightweight: five or six tools are enough for the MVP.

---

## 12. Token minimisation, LLM call cache, and recipe cache

LLMs are the main cost and the main source of non-determinism, so the design minimises calls.

### 12.1 Token-saving checklist
1. Tiered extraction: LLM only after API/table/JSON-LD fail.
2. **Generate a parser once per site layout**, cache it, reuse deterministically.
3. **Recipe cache** (12.3) so known layouts and known column mappings skip the LLM.
4. **LLM call cache** (12.2): identical request returns the stored answer.
5. Reduce HTML to clean text; strip scripts, styles, navigation; truncate to relevant regions.
6. Embedding pre-filter: use cheap embeddings to prune candidates; LLM sees only top-k uncertain cases.
7. **Batch prompting:** many column pairs or record pairs per call.
8. Send **sample values** (5-10 per column), never whole columns.
9. Use a small/cheap model for adjudication; a stronger model only for planning.
10. Short structured-output schemas; no free-text answers.
11. Per-run budgets (tokens, pages, cost) enforced in the graph state.
12. Checkpoint and resume so reruns never redo finished stages.
13. Later option: locally hosted small models fine-tuned for data tasks (Jellyfish-style) for schema/entity matching.

### 12.2 LLM call cache
- Key: hash of (model, prompt template version, rendered prompt, output schema, temperature).
- Value: raw response, tokens used, timestamp.
- Store: SQLite or a DuckDB table.
- Effects: reproducible runs, free reruns while debugging, accurate cost reporting. Temperature 0 where possible.

### 12.3 Recipe cache (reusable parsers and field mappings)
**Attribution (corrected 2026-11-14):** this is our own design. It is *not* AutoData's hypergraph: in AutoData (arXiv 2505.15859) the "oriented message hypergraph" routes messages between agents, and its local cache stores artifacts such as HTML by ID. The closest precedent for reusing LLM-generated extraction rules is **AutoCrawler** (arXiv 2404.12753). Even with GPT-4, only about 72% of AutoCrawler's generated crawlers were fully correct on SWDE, so every reuse below is validated. See `docs/RESEARCH_DESIGN_LESSONS.md`.

**Idea:** a normal graph edge links two nodes. A **hyperedge** links *several* nodes that were jointly involved in one operation, so one lookup returns a whole reusable "recipe".

**Node types:** `Domain`, `URLPattern`, `LayoutSignature` (hash of the page's DOM skeleton), `TargetSchema` (hash of the spec's schema), `Parser` (versioned selectors/code), `ValidationOutcome`, `FieldMapping` (source column signature to canonical column).

**Example hyperedges**
- Extraction recipe: `{Domain: example.com, LayoutSignature: a1f3, TargetSchema: house_v1, Parser: p7, ValidationOutcome: pass(0.97)}`
- Schema-match recipe: `{ColumnSignature: "Area (sq. ft.)" + numeric sample profile, Canonical: area_sqft, method: embedding+llm, decision: match}`

**Lookup flow**
1. Compute `LayoutSignature` for the incoming page and `TargetSchema` hash.
2. Query hyperedges containing both. If a **passing** edge exists, run the cached parser deterministically (zero LLM tokens).
3. If validation of the result falls below a threshold, mark the edge **stale**, call the LLM to regenerate the parser, store a new edge.
4. For schema matching, look up the column signature first; only unknown signatures reach the LLM.

**Storage (simple):** two tables, `hyperedges(edge_id, kind, status, score, created_at)` and `hyperedge_members(edge_id, node_type, node_id)`. Optionally load into NetworkX for inspection and diagrams. No graph database is needed.

**What to measure:** cache hit rate, tokens saved, extraction success with/without the cache (section 18, experiment E7).

---

## 13. Data trust and authenticity

You cannot prove data is true. The goal is to estimate trust and catch obvious errors, in layers.

### 13.1 Source trust score (0 to 1)
```
trust = 0.35*authority + 0.20*freshness + 0.20*coverage + 0.15*structure + 0.10*accessibility
```
- **authority** by source type (initial values): official/government 1.0; academic or established open repository 0.8; community datasets (e.g. Kaggle) 0.6; scraped commercial portal 0.5; unknown 0.3.
- **freshness:** based on last update date (use the recency formula in section 14 with the source's age).
- **coverage:** fraction of required features the source supplies.
- **structure:** regularity of schema/format (API and CSV high; free text low).
- **accessibility:** open licence and robots/terms permit automated access (not permitted means excluded, not just low-scored).

Used for: ranking sources, ordering conflict resolution during merging, and reporting in the quality report.

### 13.2 Layered authenticity checks
1. **Source trust** (above).
2. **Cross-source agreement:** if two independent sources give similar values for the same entity or locality aggregate, confidence rises; strong disagreement is **flagged**, not silently resolved.
3. **Rule validation (Pandera):** type, range, allowed values (negative price, 90,000 sq ft 2-BHK fail).
4. **Outlier detection:** z-score / IQR / isolation forest. **Flag, do not delete** (luxury properties are legitimate outliers).
5. **Extraction verification:** every LLM-extracted value must appear in the source text; otherwise reject.
6. **Provenance:** source URL, timestamp, method on every row so any value is traceable.
7. **Manual audit:** sample about 100 rows, check against the original source by hand, report the observed error rate.

### 13.3 Row confidence (0 to 1)
```
row_confidence = source_trust(winning source) * extraction_factor * agreement_factor * validation_factor
```
- `extraction_factor`: api/file 1.0; html_table 0.95; jsonld 0.9; llm and verified 0.75; llm and unverified: rejected.
- `agreement_factor`: confirmed by 2 or more independent sources 1.0; single source 0.9; sources disagree 0.6 (and flagged).
- `validation_factor`: 1.0 if all checks pass; reduced per soft-check failure; 0 if a hard check fails (row excluded).

The quality report summarises: share of rows by trust band, number flagged, share of values that came from the LLM.

---

## 14. Recency weighting

Default: recent data counts more, unless the user asks otherwise.

```
weight = max(min_weight, 0.5 ** (age_years / half_life_years))      # min_weight default 0.05
sample_weight = weight * row_confidence                              # optional combination
```
- **Half-life by domain** (proposed by the requirement analyser, shown to the user): about 2 years for fast-moving data such as prices; about 5 for medium; about 15 for slow structural data.
- **User override** through `TemporalPolicy.mode`:
  - `recent_weighted` (default): decay as above.
  - `window`: keep only `window_start`-`window_end` (e.g. "I want 2000 to 2010").
  - `uniform`: no weighting (long-term trend studies).
- **Where applied:** (a) at source ranking via the freshness term; (b) at row level as `sample_weight` passed to LightGBM and exported in the dataset, or as down-sampling of old rows.
- **Comparability:** adjust old monetary values for inflation where possible, or keep the observation date as a feature so the model can learn the trend.
- **Store** the observation date in the table and document the half-life in the data card.
- **Evaluate** with an ablation: with vs without recency weighting on the same task (experiment E5).

---

## 15. When data is not available

Never fail silently and never invent values. Degrade in steps and always report.

**Fallback ladder**
1. **Alternative sources:** query the next-ranked sources for the missing feature.
2. **Proxy or derived feature:** e.g. compute distance-to-metro from OpenStreetMap coordinates; mark `method = derived`.
3. **Partial coverage:** continue; impute with a documented strategy or leave null; report the coverage percentage.
4. **Drop the feature:** if it is non-critical, exclude and state so in the data card.
5. **Stop and notify:** if the target or a `required` feature is unavailable, or the data is too small or too old, do not deliver a dataset.

**Feasibility report (shown before expensive acquisition; the human approval point)**

| Feature | Status | Sources | Coverage | Newest data | Trust | Note |
|---|---|---|---|---|---|---|
| price (target) | found | 2 | 100% | 2025 | 0.78 | |
| built_up_area | found | 2 | 96% | 2025 | 0.78 | |
| distance_to_metro | derivable | OSM | 100% | 2026 | 0.80 | computed from coordinates |
| building_age | not_found | none | 0% | n/a | n/a | optional |

User options: approve, continue without a feature, change the problem, or upload their own file.

---

## 16. Feature relevance engine

Decides which columns to keep for **this** problem and drives source selection.

### 16.1 Three signals
1. **Statistical:** mutual information and correlation with the target; variance and cardinality checks; near-constant columns; ID-like columns (unique ratio above about 0.95); leakage detection.
2. **Model-based:** fast LightGBM baseline on the unified table; permutation importance and SHAP per feature.
3. **Semantic:** an LLM (or embeddings) compares each column's description with the problem description and domain knowledge. Keeps meaningful-but-sparse columns; discards identifiers that correlate by accident of ordering.

### 16.2 Fusion and tiers
Weighted rank aggregation (starting weights, tune on validation):
```
fused = 0.40*model_score + 0.30*stat_score + 0.30*semantic_score
```
Tiers (starting thresholds): **high** at 0.7 or above; **medium** 0.4-0.7; **low** 0.15-0.4; below 0.15 **dropped**. **Metadata** and **leakage** override the tiers and are always excluded from the final dataset but kept in the provenance record.

### 16.3 Leakage rules
- Column derived from the target (e.g. `price_per_sqft` when predicting `price`).
- Single-feature predictor near perfect (e.g. R2 above 0.98 or AUC above 0.99): candidate, confirmed or cleared by the semantic check.
- Information that would not exist at prediction time.

### 16.4 Model-guided source selection: three strategies compared (locked 2026-11-14)
Retrieve-Merge-Predict (arXiv 2402.06282) found that a simple full join is about as good as greedy result-based selection and cheaper: +2.1% vs +1.6% median, at about half the runtime. So the system runs and reports **three strategies** instead of assuming the greedy loop wins:

```
Strategy S-base   : base entity table only (no context sources)
Strategy S-full   : join ALL approved context sources, let the relevance engine prune columns
Strategy S-greedy : METAM-style loop
    S0 = CV score of baseline on base table
    for each candidate source (in trust-ranked order):
        merge source -> table'
        S1 = CV score on table'
        if S1 - S0 > epsilon (e.g. 0.5% relative): keep source, S0 = S1
        else: drop source, record reason
    stop when: max sources reached | budget exhausted | no improvement twice in a row
```
- All scores come from **nested CV** (inner folds select, outer folds score) on the train/validation portion. Greedy selection otherwise overfits the folds it selects on.
- The final held-out test split stays untouched until the single final comparison.
- The delivered dataset uses the strategy with the best outer-CV score. The data card reports all three, with candidate count capped at about 8; gains plateau beyond that, per Retrieve-Merge-Predict.
- Baseline model: LightGBM or CatBoost. Tree models tolerate the missing values that joins introduce.

### 16.5 Output
Selected feature set plus a written justification per column (tier, three scores, reason).

---

## 17. Outputs

- `dataset.parquet` (and CSV / JSON on request), optional stratified `train/val/test` splits.
- `data_card.md` per dataset: purpose, problem statement, sources and licences, per-column description / origin / transformations / relevance tier and score / imputation, recency policy, known limitations, bias-relevant distribution summaries, date generated, tool and model versions.
- `feature_documentation.json`: column metadata (name, dtype, unit, description, tier, reason).
- `quality_report.html/json`: completeness, duplication rate, outliers, failed checks, trust-band summary, LLM-derived share.
- `provenance.jsonl`: column- and row-level lineage and transformation history.
- `run_summary.json`: tokens, cost, pages fetched, cache hit rate, runtime, success status.

---

## 18. Evaluation plan

**Case studies:** (1) house prices in Delhi (primary), (2) laptop prices, (3) optional employee attrition or a public-health task. At least two independent domains.

**Baselines for the downstream comparison (same model, same split):** (A) all collected attributes, (B) problem-aware filtered dataset (ours), (C) manually prepared benchmark dataset (e.g. a Kaggle dataset for the same task), (D) optional LLM-only baseline.

| ID | Experiment | Metric |
|---|---|---|
| E1 | Feature-spec quality vs expert lists (10-20 problems) | precision, recall |
| E2 | Schema matching: string baseline vs embeddings vs embeddings+LLM, on labelled column pairs | precision, recall, F1 |
| E3 | Near-duplicate detection and locality-join accuracy on hand-labelled samples (Splink ER only if a real overlapping-entity case is used) | precision, recall, F1; % localities matched |
| E4 | Downstream models on A vs B vs C, plus source-selection strategies S-base vs S-full vs S-greedy (16.4) under nested CV | RMSE/MAE/R2 (regression), F1/AUC (classification) |
| E5 | Recency weighting on vs off | downstream score on recent test data |
| E6 | Trust and validation: manual audit of about 100 rows | observed error rate; precision of flagged rows |
| E7 | Cost and reliability, with vs without recipe and LLM caches | tokens and cost per dataset, cache hit rate, run success rate, failure modes |
| E8 | Missing-data robustness: deliberately remove a source | correct reporting, no fabricated values |
| E9 | Effort saved | time and number of manual interventions vs manual preparation |
| E10 | Leakage detection: known leak `Per_Sqft` (= Price / Area) in the Delhi MagicBricks dataset, plus injected ID-like and target-derived columns | leak flagged (yes/no); false-positive rate on legitimate columns |
| E11 | Relevance-signal ablation: stat only vs model only vs LLM only (LLM-Select style) vs fused | downstream score; overlap with expert feature list |

For extraction runs (E7 and the live demo), classify each generated parser with AutoCrawler's outcome categories: Correct, Precision-only, Recall-only, Unexecutable, Over-estimate, Else.

LLM output is non-deterministic: repeat each end-to-end run 3-5 times and report mean and standard deviation. Report failures honestly (the research question on reliability and limits).

---

## 19. Known difficulties (from the literature and practice) and mitigations

| Difficulty | Mitigation in this design |
|---|---|
| LLM scraping is costly, non-deterministic, hallucinates | Tiered extraction; parser generated once and cached; verify values against source text; schema validation |
| Agents struggle on real web pages (long HTML, access restrictions, trial-and-error loops, reproducibility, scalability) | Curated source allowlist; 2-3 well-behaved sources; strict budgets; state machine; clean-text reduction |
| LLM schema matching is costly and sensitive to how much context is given | Embedding retrieval first, LLM only on ambiguous pairs; sample values only; labelled test set |
| LLM entity matching is costly | Avoided: listings are stacked and deduplicated, context joined on locality; Splink only if a real cross-source entity case appears |
| Finding useful sources is hard | Complementary-set selection; METAM-style feedback loop |
| Relevance scoring can be circular or leak | Three independent signals; separate leakage rules; ablations; untouched test split |
| Data may be wrong, stale, or missing | Trust scores, layered validation, recency weighting, fallback ladder, feasibility report |
| Legal and ethical risk of scraping | Open sources first; robots/terms compliance; rate limiting; licence tracking |

---

## 20. Ethics, legality, limitations

- Collect only from sources that permit automated access; obey robots.txt and published terms; polite rate limiting; identifying request headers. Many commercial portals forbid scraping in their terms, so check before use and prefer open data.
- Exclude or pseudonymise personally identifiable information.
- Record licences and known limitations in every data card.
- Source data can encode social bias; the quality report includes distributional summaries to make it visible.
- Generated-dataset quality is bounded by source quality; the system measures and documents limits but cannot create information that does not exist.
- LLM extraction can misread ambiguous fields, hence validation gates and human checkpoints.
- Relevance and trust scores are advisory and always shown with reasons.

---

## 21. Repository structure

```
dataset-agent/
  README.md
  pyproject.toml
  .env.example                 # LLM keys, model names, budgets, allowlist path
  docker-compose.yml
  src/dsagent/
    config.py
    schemas/                   # spec.py, source.py, record.py, report.py (Pydantic)
    llm/                       # client.py (wrapper), cache.py (call cache), prompts/
    cache/                     # recipe_cache.py, layout_signature.py
    graph/                     # state.py, nodes.py, build.py (LangGraph)
    stages/
      requirement.py
      discovery.py
      acquisition/             # api.py, files.py, html_table.py, jsonld.py, llm_extract.py
      integration/             # normalise.py, schema_match.py, union.py, dedupe.py, enrich.py, impute.py
      validation.py
      relevance/               # statistical.py, model_based.py, semantic.py, fusion.py, feedback_loop.py
      output.py
    trust/                     # source_trust.py, row_confidence.py, recency.py
    mcp_server/                # server.py, tools/
    api/                       # main.py (FastAPI), routes, celery tasks
    storage/                   # duckdb, object store, lineage log
  frontend/                    # React + Tailwind
  tests/                       # unit + fixture data per stage
  eval/                        # benchmark data, labelled pairs, experiment scripts, notebooks
  docs/                        # diagrams, report drafts
```

---

## 22. Build order (about 12 weeks)

Always keep something runnable.

| Weeks | Work |
|---|---|
| 1-2 | Read key papers (section 25). Fix the single case study (Delhi houses). Write the final claim. Set up repo, config, Pydantic schemas, LLM wrapper + call cache. |
| 3-4 | Stage 1 requirement analyser; test against 5-10 hand-written specs. Basic MCP tool server (5-6 tools). |
| 5-6 | **No agents yet.** Hand-download 2-3 open datasets. Build integration (normalise, schema match, union, dedupe, locality/geo enrichment, impute) and validation (Pandera, profiling, trust scoring). |
| 7 | Relevance engine and the METAM-style feedback loop on the merged table. Recency weighting. |
| 8-9 | Discovery and acquisition agents with the cheapest-first ladder, recipe cache, feasibility report; wrap everything in LangGraph with human approval and budgets. |
| 10 | Second case study (laptop prices). Run experiments E1-E9. |
| 11 | FastAPI + thin React UI (submit, approve feasibility, preview, download). Docker Compose. |
| 12 | Results analysis, documentation, final report, demo. |

**Minimum viable demo:** problem text in, feature spec out, two sources acquired and merged, validated, filtered by relevance, data card out.

---

## 23. API and configuration

**Endpoints (FastAPI)**
- `POST /jobs` : submit problem (and optional file).
- `GET /jobs/{id}` : status and progress.
- `GET /jobs/{id}/spec` and `POST /jobs/{id}/spec/approve` : review and approve the feature spec.
- `GET /jobs/{id}/feasibility` and `POST /jobs/{id}/feasibility/approve` : review and approve sources.
- `GET /jobs/{id}/artifacts` : dataset, data card, quality report, provenance.

**Environment (`.env`)**: `LLM_PROVIDER`, `LLM_MODEL_PLANNER`, `LLM_MODEL_CHEAP`, API keys, `MAX_TOKENS_PER_RUN`, `MAX_PAGES_PER_RUN`, `MAX_COST_PER_RUN`, `SOURCE_ALLOWLIST_PATH`, `USER_AGENT`, `CACHE_DB_PATH`, storage and Redis URLs.

---

## 24. Rules for coding agents

1. Build the minimum viable stack first (LangGraph, DuckDB, Pandera, FastAPI, Polars, LightGBM). Add other tools only when needed.
2. Every LLM call goes through the wrapper in `llm/client.py`, uses a Pydantic output schema, is cached, and counts toward the run budget. No direct SDK calls elsewhere.
3. Never fabricate data. Missing stays missing and is reported.
4. Every record and every derived column carries provenance.
5. Extraction ladder order is fixed: API/file, HTML table, JSON-LD, LLM. LLM extraction must verify values against source text.
6. Respect robots.txt, the allowlist, rate limits and the User-Agent in `fetch_page`.
7. Each stage is a pure function of typed input to typed output, with unit tests on small fixture data. Do not wire stages into the graph before they pass tests.
8. Selection and tuning use train/validation only; keep the final test split untouched.
9. Thresholds, weights and half-lives live in config, not hard-coded.
10. Log tokens, cost, cache hits, and runtime per run to `run_summary.json` (MLflow optional).
11. Fail loudly: critical validation failures stop the run with a report.
12. **Definition of done for a stage:** typed I/O, unit tests, a fixture-based example, provenance emitted, budget enforced, documented in `docs/`.

---

## 25. Reading list

**Closest related systems and methods**
- AutoData: multi-agent system for automated web data collection (arXiv 2505.15859). Note: its "hypergraph" routes messages between agents; it is not a parser cache.
- METAM: goal-oriented data discovery with a downstream-task feedback loop.
- Auctus: dataset search engine for discovery and augmentation (PVLDB 2021, Castelo et al.).
- Harmonia: agentic data harmonization (Santos et al.).
- CAAFE: context-aware automated feature engineering with LLMs (arXiv 2305.03403, Hollmann et al.).
- AutoCrawler: LLM-generated reusable crawlers; ~72% fully correct with GPT-4 on SWDE (arXiv 2404.12753).
- Retrieve, Merge, Predict: simple exact-match joins and full join are near-best for augmentation (arXiv 2402.06282).
- LLM-Select: LLM feature selection from names alone is competitive with LASSO (arXiv 2407.02694, TMLR).
- Kapoor & Narayanan, Leakage and the Reproducibility Crisis in ML-based Science (Patterns, 2023).
- Why Do Multi-Agent LLM Systems Fail? (MAST taxonomy, arXiv 2503.13657).
- Non-Determinism of "Deterministic" LLM Settings (arXiv 2408.04667).
- Agentic Data Engineering for model specialization (EMNLP 2026 main, paper 2330).

**Schema matching, entity matching, integration**
- Magneto: combining small and large language models for schema matching (arXiv 2412.08194, PVLDB 2025).
- ReMatch: retrieval-enhanced schema matching with LLMs (Sheetrit et al.).
- Parciak et al., Schema Matching with LLMs: an Experimental Study (arXiv 2407.11852).
- Survey: LLM x DATA (arXiv 2505.18458); survey on language models for tabular data preparation (arXiv 2508.01556).
- Retrieve, Merge, Predict: augmenting tables with data lakes (arXiv 2402.06282).
- Fan et al., Table Discovery in Data Lakes (SIGMOD companion, 2023).
- Rahm & Bernstein, A Survey of Approaches to Automatic Schema Matching (VLDB Journal, 2001).
- Christen, Data Matching (Springer, 2012).

**Also from memory, verify titles and venues before citing:** Narayan et al., "Can Foundation Models Wrangle Your Data?" (VLDB 2022); Li et al., Ditto (entity matching); Peeters & Bizer on ChatGPT for entity matching; Fellegi-Sunter record linkage (the basis of Splink); Starmie and other table union search papers; Jellyfish (instruction-tuned data-processing LLMs); Anthropic, "Building Effective Agents".

**From the original synopsis:** Halevy et al. (Goods, SIGMOD 2016); Lundberg & Lee (SHAP, 2017); Gebru et al. (Datasheets for Datasets, 2021); Yao et al. (ReAct, ICLR 2023); Anthropic (Model Context Protocol specification, 2024); Polyzotis et al. (Data Management Challenges in Production ML, SIGMOD 2017).

---

## 26. Glossary

- **Agentic AI:** LLM-based components that choose and call tools to reach a goal.
- **MCP (Model Context Protocol):** a standard interface through which agents access tools.
- **LangGraph:** a library for building agent workflows as explicit state machines.
- **Schema (Pydantic):** a declared structure that an LLM output or table must satisfy.
- **Schema matching:** finding which columns in different sources mean the same thing.
- **Entity resolution:** finding records that refer to the same real-world entity.
- **Blocking:** grouping candidate records so only likely matches are compared.
- **LSH:** locality-sensitive hashing, for finding near-duplicates quickly.
- **Provenance / lineage:** the recorded origin and transformation history of every value.
- **Data card:** a document describing a dataset's origin, columns, and limitations.
- **Target leakage:** a feature that contains information about the target that would not be available at prediction time.
- **SHAP / permutation importance:** methods for measuring how much each feature contributes to a model.
- **Hyperedge:** a graph edge connecting more than two nodes; used here to store a reusable "recipe" in the cache.
- **Half-life (recency):** the age at which a record's weight falls to one half.
