# Stage 1: Requirement analyser

**Code:** `src/dsagent/stages/requirement.py` · prompt and output schema: `src/dsagent/llm/prompts/requirement.py` · tests: `tests/test_requirement.py` · evaluation: `eval/run_e1.py`

## What it does
`analyse_problem(problem_text, llm, sample_columns=None) -> RequirementResult`

1. Rejects input shorter than 10 characters without calling the LLM.
2. One planner-model call returns a `SpecDraft`:
   - task type, target and features (dtype, unit, description, `required`, optional range or allowed values);
   - region and a volatility class (fast, medium or slow);
   - an optional year window;
   - the columns it chose to exclude, with reasons;
   - its assumptions;
   - an optional clarification question.
3. Bad output is retried twice by the LLM wrapper, then reported as `RequirementError` ("please rephrase").
4. If the LLM asks a clarification question, the stage raises `NeedsClarification` and the UI shows the question to the user.
5. **Deterministic guards** (they do not rely on the LLM). They remove:
   - features derived from the target (`price_per_sqft`, `log_price`, `price_band`);
   - identifier-like columns (`*_id`, URL, username, email, phone);
   - duplicates.

   Every removal is listed in `guard_removed`, so the user can restore a column at the approval step.
6. Volatility is converted to a half-life from `config.recency`:
   - fast → 2 years
   - medium → 5 years
   - slow → 15 years

   A year window is used only when the user gave both start and end.
7. The output is a validated `DatasetSpec`, which the user approves or edits. This is the first human checkpoint.

## Definition of done (spec section 24, rule 12)
| Item | Status |
|---|---|
| Typed I/O | `str` → `RequirementResult(spec: DatasetSpec, ...)` |
| Unit tests on fixtures | 16 tests, fake LLM backend, no network |
| Fixture-based example | `delhi_draft()` in the tests |
| Provenance | N/A for the spec itself; the LLM call is cached and replayable (template `requirement-v1`) |
| Budget enforced | through `LLMClient` (token, cost and call limits) |
| Documented | this file |

## Experiment E1
- `eval/gold_specs/` contains 5 expert-style feature lists, each with aliases and a `must_not_include` leakage list:
  - Delhi house prices
  - laptop prices
  - employee attrition
  - used-car prices (India)
  - crop yield (India)
- The design spec asks for 10–20 lists. **The team should review these 5 and add more, each written before looking at any LLM output, to avoid bias.**
- Run it (calls the real API and costs money):
  ```bash
  .venv/Scripts/python eval/run_e1.py --repeats 3
  ```
- Reports per-problem precision, recall and F1, target accuracy, the rate of forbidden (leakage) columns, and mean ± sd over repeats. Results are saved to `eval/results/`.
- Precision is a lower bound: a correct feature under an unexpected name is counted as a miss. Review the `unmatched_predictions` list by hand and add genuine synonyms as aliases.
- Planned comparison (LLM-Select framing): the LLM-chosen features vs features selected by LASSO/MI on a full benchmark dataset.

## Known limitations
- The guards are name-based. A leak with an innocent name (e.g. `valuation_amount` for price) is not caught here; the relevance engine's leakage rules (spec 16.3) are the second line of defence.
- Volatility is a coarse 3-way class, which is deliberate: the user can override the half-life at approval.
