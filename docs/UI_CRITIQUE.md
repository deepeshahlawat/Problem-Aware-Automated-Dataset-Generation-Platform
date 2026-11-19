# UI critique of MVP v1 (2026-11-19)

Walked through every screen of the first demo UI with the Delhi example (landing, spec review, feasibility, build, results, preview), judged as a first-time user: a student, a teacher at the demo, or an analyst who knows the domain but not this codebase.

## What a user would dislike
| # | Problem | Where | Why it hurts |
|---|---|---|---|
| 1 | **One endless page.** Every step stays stacked; after a build the results start ~6 screens down | whole app | No sense of place; the important part (the dataset) is the hardest to reach |
| 2 | **Looks like a default template**: stock indigo on dark slate, `rounded-xl` cards everywhere, no brand, no visual hierarchy | whole app | Reads as "vibe coded"; nothing signals care or trust |
| 3 | **Jargon everywhere**: `float`, `snake_case` names, `MI`, `S-base`, `holdout R²`, `fused`, `semantic: target_derived`, `x_` prefixes, `dtype corrected` | spec, results | A domain user can't read the product's own explanations |
| 4 | **Tables clip and wrap badly.** Descriptions are cut off, the Trust column shows "0.", and reasons wrap one word per line | spec, feasibility, relevance | Broken-looking at normal window widths |
| 5 | **The spec can only be pruned, not corrected.** If the model gets a type wrong (it once called *locality* a number), the user sees "float" and can't fix it | spec review | The human checkpoint can't fix the most common LLM error |
| 6 | **Feasibility has no summary**: 14 rows plus raw trust maths ("coverage 0.3333") and no headline | feasibility | The user can't answer "is this going to work?" at a glance |
| 7 | **Misleading "selected"**: OSM context sources are marked selected, but no data arrives until Stage 3 exists | feasibility | Promises features that never show up |
| 8 | **The build is a raw timestamped log**: no progress, no current step, no sense of time | build | Feels like a terminal, not a product |
| 9 | **Results treat 17 columns equally.** Kept, dropped, empty, ID and leak columns are mixed in one long table with numbers-only reasons | results | The story (what was kept, and why) is buried |
| 10 | **KPIs aren't explained.** "Holdout R² 0.848" and "$0.0000, 0 calls, 3 cache hits" mean nothing to most users | results | The headline numbers don't persuade |
| 11 | **Downloads are file names** (`integration_report.json`). There's no primary "get my dataset" action | results | The main deliverable looks like a debug dump |
| 12 | **The preview shows raw values**: 6500000, 344.445, snake_case headers | preview | Hard to sanity-check the data |
| 13 | **Refresh loses everything**, there's no run history, and runs can't be shared | whole app | A demo or a teacher's question means starting over |
| 14 | **Errors appear at the top**, far from the button that caused them | all steps | The user misses them |
| 15 | **Status relies on colour alone**, with low-contrast grey text | badges | Accessibility; unreadable on projectors |
| 16 | **The not-deliverable case is a dead end** with no next step | feasibility | The user doesn't know what to do |

## What a user would demand
1. **Know where I am**: a clear step rail, one step on screen at a time, and the ability to look back.
2. **Plain language first, technical names second**: "Bedrooms" with `number_of_bedrooms` underneath; "Number" instead of `float`; explanations of what each score means.
3. **Fix the spec, not just prune it**: change a feature's type and whether it's required.
4. **A one-line verdict at each checkpoint**, e.g. "10 of 14 features available from 1 source. 2 can be derived later. 4 not found (all optional)."
5. **Honest status for context sources**: "Planned: needs live acquisition (not in this MVP)".
6. **Visible progress** through the stages.
7. **A results dashboard, not a wall**: an overview with the verdict, then tabs for Features, Quality, Sources, Data.
8. **One obvious "Download dataset" button**, with the reports secondary and described.
9. **A readable preview**: ₹65.0 L, 1,300 sq ft, friendly headers.
10. **Runs survive refresh** (run id in the URL) and a recent-runs list.
11. **Looks like a product**: a calm neutral palette, one accent colour, a real typeface, consistent spacing, light and dark themes.

## Redesign decisions
- Layout: a top bar (brand, run id, new run, recent runs), a sticky step rail on the left, and one focused panel per step.
- Visual system: Inter, a zinc neutral scale, one teal accent, hairline borders, a restrained radius, no gradients or glow; tabular numbers for data.
- Results: verdict banner, KPI strip with captions, tabs (Overview · Features · Quality · Sources · Data).
- Backend additions: spec overrides (type / required), a current-stage indicator for the progress view, a recent-runs list.

## What changed (v2, same day)
Every item in the table above was addressed:
- one step on screen at a time, with a step rail;
- plain-language labels with the technical names secondary;
- editable feature types and required flags (sent to the API as `overrides`);
- one-line verdicts with an availability bar;
- "Planned · needs live acquisition" for context sources;
- a stage-by-stage progress view;
- a results dashboard (Overview · Features · Quality · Sources & method · Data);
- a primary CSV download with the reports described;
- formatted values (₹65.0 L);
- the run id in the URL, plus a recent-runs menu;
- inline errors;
- text and icon alongside every colour;
- light and dark themes;
- no horizontal overflow at 375 px.

Two real bugs surfaced while testing the new UI, and both are fixed:
1. OpenRouter can answer **HTTP 200 with an error in the body** (provider overloaded). It wasn't retried, the AI review silently failed, and the Per_Sqft leak passed. Now retried.
2. With the AI review down, leak protection had no deterministic catch for `Per_Sqft`. Now the name rule flags unnamed rates ("per + unit"), and the UI warns, with a "Re-run scoring" button, whenever the AI review was unavailable.
