"""Stage 6 (recency) and Stage 7 (relevance) tests on synthetic tables with planted
structure (tests/synthetic.py) and a stubbed LLM. No network."""

import warnings
from datetime import date, datetime, timezone

import numpy as np
import polars as pl
import pytest

from dsagent.config import Settings
from dsagent.llm import Budget, LLMCallCache, LLMClient
from dsagent.llm.client import BackendResult
from dsagent.schemas import DatasetSpec, Feature
from dsagent.stages.recency import apply_recency
from dsagent.stages.relevance import run_relevance
from dsagent.stages.relevance.leakage import _name_leak
from dsagent.stages.relevance.semantic import ColumnAssessment, SemanticAssessment
from synthetic import LINEAGE, SPEC, make

warnings.filterwarnings("ignore", category=UserWarning)
S = Settings(_env_file=None)
NOW = datetime(2026, 11, 14, tzinfo=timezone.utc)


# --- Stage 6: recency ----------------------------------------------------------------

def _rec_table():
    return pl.DataFrame({
        "price": [1.0, 2.0, 3.0, 4.0],
        "listed": [date(2026, 11, 14), date(2024, 11, 14), None, None],
        "_source_id": ["a", "a", "a", "b"],
        "_row_confidence": [1.0, 0.5, 1.0, 1.0],
    })


def test_recency_weights_date_column_then_source_date_then_neutral():
    spec = SPEC.model_copy(update={"temporal": SPEC.temporal.model_copy(update={"half_life_years": 2.0})})
    out, rep = apply_recency(_rec_table(), spec, "listed", {"a": datetime(2022, 11, 14, tzinfo=timezone.utc)},
                             S, NOW)
    w = out["_recency_weight"].to_list()
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.5, abs=0.01)    # 2 years old, half-life 2
    assert w[2] == pytest.approx(0.25, abs=0.01)   # falls back to source a's date (4 years)
    assert w[3] == 1.0                             # source b undated: neutral, counted
    assert out["_sample_weight"].to_list()[1] == pytest.approx(0.25, abs=0.01)  # x row confidence 0.5
    assert (rep.dated_from_column, rep.dated_from_source, rep.undated) == (2, 1, 1)


def test_recency_window_and_uniform_modes():
    win = SPEC.model_copy(update={"temporal": SPEC.temporal.model_copy(
        update={"mode": "window", "window_start": 2025, "window_end": 2026})})
    out, rep = apply_recency(_rec_table(), win, "listed", None, S, NOW)
    assert out.height == 1 and rep.removed_outside_window == 3  # 2024 row + 2 undated rows
    uni = SPEC.model_copy(update={"temporal": SPEC.temporal.model_copy(update={"mode": "uniform"})})
    out, _ = apply_recency(_rec_table(), uni, "listed", None, S, NOW)
    assert set(out["_recency_weight"].to_list()) == {1.0}


# --- Stage 7: relevance -------------------------------------------------------------

@pytest.fixture(scope="module")
def result():
    return run_relevance(make(n=500), SPEC, LINEAGE, None, S)


def test_planted_columns_get_the_right_tiers(result):
    tiers = {c.column: c.tier for c in result.columns}
    assert tiers["x_per_sqft"] == "leakage"        # Per_Sqft = Price / Area (E10)
    assert tiers["x_listing_id"] == "metadata"
    assert tiers["x_site_name"] == "dropped"       # constant
    assert tiers["x_noise"] == "dropped"
    assert tiers["amenities_count"] == "dropped"   # planted with no effect
    assert tiers["locality"] in ("high", "medium") and tiers["area_sqft"] in ("high", "medium")


def test_leak_found_by_pair_rule_and_excluded_from_selection(result):
    assert ("x_per_sqft", "pair") in [(f.column, f.rule) for f in result.leakage]
    assert "area_sqft" not in [f.column for f in result.leakage]  # spec feature not blamed
    assert "x_per_sqft" not in result.selected_columns


def test_strategies_use_context_that_helps(result):
    by = {s.strategy: s for s in result.strategies}
    assert set(by) == {"S-base", "S-full", "S-greedy"}
    assert by["S-greedy"].mean > by["S-base"].mean           # metro distance carries signal
    assert result.chosen_strategy in ("S-greedy", "S-full")
    assert "distance_to_metro_km" in result.selected_columns
    assert all(len(s) <= 1 for s in by["S-greedy"].groups_selected_per_fold)


def test_holdout_is_disjoint_and_scored_once(result):
    assert result.train_rows + result.holdout_rows == 500
    assert len(result.holdout_refs) == result.holdout_rows
    assert result.metric == "r2"
    assert result.holdout_scores["selected"] > 0.8
    # the leaky 'all columns' set looks better on paper; that is the point of E4/E10
    assert result.holdout_scores["all_columns"] >= result.holdout_scores["selected"]
    assert any("optimistic" in n for n in result.notes)


def test_name_leak_rule():
    assert _name_leak("x_price_per_sqft", "price")
    assert _name_leak("log_price", "price")
    assert not _name_leak("area_sqft", "price")
    assert _name_leak("x_per_sqft", "price")  # an unnamed rate: almost always target per unit
    assert not _name_leak("x_sqft_per_room", "price")


class FakeSemantic:
    def __init__(self, assessments):
        self.assessments = assessments
        self.calls = 0

    def call(self, *, model, system, user, schema, max_tokens):
        self.calls += 1
        return BackendResult(SemanticAssessment(columns=self.assessments), "", model, 500, 200, "stop", 0.0002)


def _llm(assessments):
    b = FakeSemantic(assessments)
    return LLMClient(Budget(1_000_000, 1, 10), LLMCallCache(":memory:"), b, S), b


def _single_leak_table(n=300, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    return pl.DataFrame({
        "price": 3 * x + rng.normal(0, 0.1, n), "area_sqft": x,
        "bedrooms": rng.integers(1, 5, n), "_row_ref": [f"a:{i}" for i in range(n)],
    })


SPEC2 = DatasetSpec(problem_text="predict price from area", task_type="regression",
                    target=Feature(name="price", dtype="float", description="price"),
                    features=[Feature(name="area_sqft", dtype="float", description="area"),
                              Feature(name="bedrooms", dtype="int", description="bedrooms")])


def test_single_feature_leak_confirmed_without_llm_cleared_by_semantic():
    r = run_relevance(_single_leak_table(), SPEC2, {}, None, S)
    assert {c.column: c.tier for c in r.columns}["area_sqft"] == "leakage"
    llm, backend = _llm([
        ColumnAssessment(column="area_sqft", relevance=0.95, role="predictive", reason="core driver"),
        ColumnAssessment(column="bedrooms", relevance=0.4, role="predictive", reason="size proxy"),
        ColumnAssessment(column="made_up_col", relevance=1, role="predictive", reason="ignored"),
    ])
    r2 = run_relevance(_single_leak_table(), SPEC2, {}, llm, S)
    tiers = {c.column: c.tier for c in r2.columns}
    assert tiers["area_sqft"] in ("high", "medium")
    assert any("semantically predictive" in n for n in r2.notes)
    assert backend.calls == 1 and "made_up_col" not in tiers


def test_semantic_roles_override_tiers():
    llm, _ = _llm([
        ColumnAssessment(column="area_sqft", relevance=0.9, role="predictive", reason="x"),
        ColumnAssessment(column="bedrooms", relevance=0.2, role="future_info", reason="known after sale"),
    ])
    t = _single_leak_table().with_columns(pl.col("price") + pl.Series(np.random.default_rng(2).normal(0, 60, 300)))
    r = run_relevance(t, SPEC2, {}, llm, S)
    assert {c.column: c.tier for c in r.columns}["bedrooms"] == "leakage"


def test_classification_path_uses_auc():
    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(0, 1, n)
    y = (x + rng.normal(0, 0.7, n) > 0).astype(int)
    spec = DatasetSpec(problem_text="will the employee leave", task_type="classification",
                       target=Feature(name="attrition", dtype="category", description="left"),
                       features=[Feature(name="satisfaction", dtype="float", description="s"),
                                 Feature(name="dept", dtype="category", description="d")])
    t = pl.DataFrame({"attrition": [str(v) for v in y], "satisfaction": x,
                      "dept": rng.choice(["a", "b", "c"], n).tolist(), "_row_ref": [str(i) for i in range(n)]})
    r = run_relevance(t, spec, {}, None, S)
    assert r.metric == "roc_auc" and 0.7 < r.holdout_scores["selected"] <= 1.0
    assert {c.column: c.tier for c in r.columns}["satisfaction"] in ("high", "medium")
