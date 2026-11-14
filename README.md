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
| Hand-downloaded case-study data (`data/README.md`) | instructions written; files not yet downloaded |
| Stage 1 requirement analyser (`src/dsagent/stages/requirement.py`) | done, tested; E1 run with 2 models (F1 0.739 / 0.724, see `docs/MODEL_CHOICE.md`) |
| Stage 2 source discovery (`src/dsagent/stages/discovery/`) | done, tested; live dry runs on Delhi houses and laptops (`eval/run_discovery.py`) |
| Stage 4 integration (`src/dsagent/stages/integration/`) | done, tested on fixtures (88 tests total); not yet run on real data |
| Stage 5 validation (Pandera contracts, quality report) | next |

## Rules that the code enforces
- All LLM calls go through `dsagent.llm.LLMClient.structured()`. It uses a Pydantic output schema, retries then rejects invalid output, and uses the SQLite call cache and the run budget.
- Thresholds, weights and half-lives live in `config.py`.
- Every stage gets unit tests on small fixture data before it is wired into the graph.
