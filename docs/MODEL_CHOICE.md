# LLM choice for development (2026-11-14)

**Goal:** the highest intelligence per dollar on OpenRouter, with structured-output (strict JSON schema) support. The team has about $3.5 of credit.

**Sources:**
- intelligence scores and cost per task: Artificial Analysis leaderboard (https://artificialanalysis.ai/leaderboards/models), Intelligence Index v5;
- prices and structured-output support: OpenRouter `/api/v1/models`, fetched 2026-11-14.

"Cost per task" is Artificial Analysis's cost to run one Intelligence Index task. It includes reasoning tokens, so it shows how verbose a model is, not just its per-token price.

| Model (OpenRouter ID) | AA Index | AA cost / task | $ in / out per 1M | Index per $ (per task) | Structured outputs |
|---|---|---|---|---|---|
| **Muse Ember** (`meta/muse-ember`) | 36 | $0.14 | 0.10 / 0.50 | **257** | yes |
| MiMo-V2.6-Flash (`xiaomi/mimo-v2.6-flash`) | 27 | $0.13 | 0.14 / 0.28 | 208 | yes |
| **GPT-6 Luna** (`openai/gpt-6-luna`) | 28 | $0.15 | 0.10 / 0.50 | 187 | yes |
| MiMo-V2.6-Pro (`xiaomi/mimo-v2.6-pro`) | 30 | $0.26 | 0.435 / 0.87 | 115 | yes |
| Claude Haiku 5.5 (`anthropic/claude-haiku-5.5`) | 36 | $0.41 | 0.10 / 0.50 | 88 | yes |
| DeepSeek V4.1 Flash (`deepseek/deepseek-v4.1-flash-1017`) | 33 | $0.60 | 0.038 / 0.50 | 55 | yes |
| GPT-6.1 Terra (`openai/gpt-6.1-terra`) | 51 | $0.96 | 1.00 / 5.00 | 53 | yes |
| Claude Opus 5.5 (direct Anthropic API) | 59 | $10.04 | 4.00 / 20.00 | 6 | yes |

## Decision
- **Planner role** (requirement analysis, data-card text): **`meta/muse-ember`**. It has the best index per dollar, and its score equals Claude Haiku 5.5's at about a third of the cost per task.
- **Cheap role** (schema-match adjudication, semantic relevance): **`openai/gpt-6-luna`**. It's the same per-token price but less verbose for short classification calls. `xiaomi/mimo-v2.6-flash` is a close alternative.
- **Optional quality check:** for the final E1 report, re-run the planner once with `openai/gpt-6.1-terra` (index 51). It's still cheap at our call sizes and shows whether a stronger model changes spec quality, which is a useful result in itself.

Both are configurable in `.env` (`DSAGENT_LLM_MODEL_PLANNER`, `DSAGENT_LLM_MODEL_CHEAP`), and the model ID is part of the cache key.

## Expected cost (estimates, not measured)
A Stage 1 call is about 1.5k input + 2–4k output tokens:
- **Muse Ember:** about $0.002 per call, so E1 (5 problems × 3 repeats) costs about $0.03.
- **GPT-6.1 Terra:** about $0.02 per call, so E1 costs about $0.30.

`DSAGENT_MAX_COST_PER_RUN=0.25` in `.env.example` caps any single run. OpenRouter reports the billed cost per call, and that figure is what gets recorded.

## Measured: E1 head-to-head (2026-11-14, 5 problems × 3 repeats)
| Planner | Precision | Recall | F1 | Target correct | Failed calls | Avg features | Billed cost (15 calls) |
|---|---|---|---|---|---|---|---|
| `meta/muse-ember` | 0.685 ± 0.085 | 0.812 ± 0.139 | **0.739 ± 0.104** | 100% | 1 (needless clarification question) | 13.4 | $0.037 |
| `deepseek/deepseek-v4.1-flash-1017` | 0.622 ± 0.093 | **0.876 ± 0.088** | 0.724 ± 0.083 | 93%* | 0 | 16.3 | $0.037 |

\*The one "wrong" target, `left_within_next_year` for attrition, is a scorer alias miss, not a real error.

Raw results: `eval/results/e1_*.json`.

**Reading:**
- **Quality is a statistical tie.** The F1 difference (0.015) is far inside one standard deviation, and n is only 15.
- **Different styles:** DeepSeek proposes more features (higher recall, lower precision); Ember is tighter.
- **Real cost was identical**, about $0.0025 per call. The 4× gap in Artificial Analysis cost per task did **not** carry over to our short structured calls, so benchmark cost figures should be checked against measurement.
- **Precision is a lower bound for both models.** Many "unmatched" predictions are reasonable features the gold lists lack (`total_working_years`, `body_type`, `release_year`).
- **Ember's failure:** once, it asked a clarification question for "house prices in Delhi". Ask-only-when-necessary wording in the prompt is a candidate fix, to be checked on *new* problems so the gold set isn't tuned on.

**Decision:** keep `meta/muse-ember` as the default planner, with DeepSeek V4.1 Flash as an equal-cost alternative. Re-check once the gold set has 10–20 team-written problems.

## Why not the free (`:free`) models
- Rate limits are tight below $10 of purchased credit: 20 requests/minute and **50 per day** (OpenRouter docs, `/docs/api-reference/limits`).
- Free endpoints may log prompts, and they change or disappear without notice (one listed today is scheduled for removal on 2026-11-28).

That makes them a poor base for reproducible evaluation. They stay usable for ad-hoc tries by setting a `:free` model ID.

## Caveats
- Several of these models are newer than the team's own knowledge. The scores come from one benchmark source and should be quoted as "Artificial Analysis Intelligence Index v5, retrieved 2026-11-14".
- Structured-output support is per provider endpoint. The backend sends `provider.require_parameters: true`, so OpenRouter routes only to endpoints that honour the schema.
