# dsagent: Problem-Aware Automated Dataset Generation

Code for the B.Tech minor project. The full design specification is [`docs/DESIGN_SPEC.md`](docs/DESIGN_SPEC.md); the literature review behind the locked decisions is [`docs/RESEARCH_DESIGN_LESSONS.md`](docs/RESEARCH_DESIGN_LESSONS.md).

## Setup (Windows, Git Bash or PowerShell)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
cp .env.example .env        # then fill in ANTHROPIC_API_KEY (or use `ant auth login`)
.venv/Scripts/python -m pytest
```

Extra dependency groups (installed when the matching stage is built): `.[data]` (Polars, DuckDB, Pandera, datasketch) and `.[ml]` (scikit-learn, LightGBM, SHAP).

## Status (build order, README section 22)

| Part | Status |
|---|---|
| Repo skeleton, config (`src/dsagent/config.py`) | done |
| Pydantic contracts (`src/dsagent/schemas/`) | done, tested |
| LLM wrapper + call cache + budget (`src/dsagent/llm/`) | done, tested with a fake backend |
| Real case-study data (`data/sources.json`) | laptops (HF, Apache-2.0) + Delhi (GitHub re-upload, no licence) downloaded; see `docs/REAL_DATA_RUN_2026-11-14.md` |
| Stage 1 requirement analyser (`src/dsagent/stages/requirement.py`) | done, tested; E1 run with 2 models (F1 0.739 / 0.724, see `docs/MODEL_CHOICE.md`) |
| Stage 2 source discovery (`src/dsagent/stages/discovery/`) | done, tested; live dry runs on Delhi houses and laptops (`eval/run_discovery.py`) |
| Stage 4 integration (`src/dsagent/stages/integration/`) | done; run on real data (5 real-data bugs fixed) |
| Stage 5 validation (`src/dsagent/stages/validation.py`) | done; run on real data |
| Stage 6 recency + Stage 7 relevance engine (`src/dsagent/stages/recency.py`, `relevance/`) | done; run end-to-end on real laptop + Delhi data (111 tests total) |
| Stages 3 (acquisition agents) + LangGraph orchestration (weeks 8-9) | next |

## Rules that the code enforces
- All LLM calls go through `dsagent.llm.LLMClient.structured()`. It uses a Pydantic output schema, retries then rejects invalid output, and uses the SQLite call cache and the run budget.
- Thresholds, weights and half-lives live in `config.py`.
- Every stage gets unit tests on small fixture data before it is wired into the graph.

## Demo (MVP web app)

**Windows PowerShell** (run each line separately; PowerShell 5.1 has no `&&`):
```powershell
cd dataset-agent
.\.venv\Scripts\python.exe -m uvicorn dsagent.api.main:app --port 8000
```

**Git Bash:**
```bash
cd dataset-agent && .venv/Scripts/python -m uvicorn dsagent.api.main:app --port 8000
```

First time only, install the dependencies: `.\.venv\Scripts\python.exe -m pip install -e ".[data,ml,api,dev]"`.
If port 8000 is busy, add `--port 8001` and open that port instead.
Open http://localhost:8000. Flow: problem → review the spec (untick features) → review the feasibility report → build → results:
- rows delivered, columns kept, holdout score and LLM cost;
- leakage blocked;
- feature relevance with reasons;
- quality report;
- downloads and a dataset preview.

If validation needs review (e.g. an unknown currency), nothing is delivered until you press "Continue anyway (provisional)". The two example problems are served from the LLM cache (free). Data comes from the files registered in `data/sources.json`, because Stage 3 (live acquisition) is not built yet.
