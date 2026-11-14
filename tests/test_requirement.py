"""Stage 1 tests on fixture data with a fake LLM backend (no network)."""

from pathlib import Path

import pytest

from dsagent.config import Settings
from dsagent.evaluation.spec_scoring import load_gold, score_spec
from dsagent.llm import Budget, LLMCallCache, LLMClient
from dsagent.llm.client import BackendResult
from dsagent.llm.prompts.requirement import SpecDraft, render_user
from dsagent.schemas.spec import Feature
from dsagent.stages.requirement import NeedsClarification, RequirementError, analyse_problem

GOLD_DIR = Path(__file__).resolve().parents[1] / "eval" / "gold_specs"


def F(name, dtype="float", **kw):
    return Feature(name=name, dtype=dtype, description=name, **kw)


def delhi_draft(**overrides) -> SpecDraft:
    data = dict(
        task_type="regression",
        target=F("price", unit="INR"),
        features=[
            F("locality", "category"),
            F("built_up_area_sqft", unit="sq ft"),
            F("bhk", "int"),
            F("bathrooms", "int"),
            F("furnishing", "category"),
            F("distance_to_metro_km", required=False),
        ],
        region="Delhi",
        volatility="fast",
        excluded_candidates=["price_per_sqft: derived from target"],
        assumptions=["Sale prices, not rents"],
    )
    data.update(overrides)
    return SpecDraft(**data)


class FakeBackend:
    def __init__(self, draft):
        self.draft = draft
        self.prompts = []

    def call(self, *, model, system, user, schema, max_tokens):
        self.prompts.append(user)
        return BackendResult(self.draft, "", model, 800, 400, "end_turn")


def run(draft, problem="I want to predict house prices in Delhi", **kw):
    settings = Settings(_env_file=None)
    backend = FakeBackend(draft)
    llm = LLMClient(Budget(100_000, 10, 10), LLMCallCache(":memory:"), backend, settings)
    return analyse_problem(problem, llm, settings=settings, **kw), backend


def test_builds_spec_with_configured_half_life():
    result, _ = run(delhi_draft())
    spec = result.spec
    assert spec.problem_text == "I want to predict house prices in Delhi"
    assert spec.target.name == "price"
    assert spec.region == "Delhi"
    assert spec.temporal.mode == "recent_weighted"
    assert spec.temporal.half_life_years == Settings(_env_file=None).recency.half_life_fast
    assert result.assumptions == ["Sale prices, not rents"]
    assert result.excluded == ["price_per_sqft: derived from target"]
    assert result.guard_removed == []


@pytest.mark.parametrize(
    "bad, reason",
    [
        ("price_per_sqft", "leakage"),
        ("log_price", "leakage"),
        ("price_band", "leakage"),
        ("listing_id", "identifier"),
        ("property_url", "identifier"),
        ("seller_username", "identifier"),
    ],
)
def test_guards_remove_leakage_and_identifiers(bad, reason):
    draft = delhi_draft()
    draft.features.append(F(bad))
    result, _ = run(draft)
    assert bad not in result.spec.feature_names()
    assert any(r.startswith(bad) and reason in r for r in result.guard_removed)


def test_guard_keeps_legitimate_columns_mentioning_target_word():
    draft = delhi_draft(target=F("price"), features=[F("base_price_of_locality_index"), F("bhk", "int")])
    # 'price' appears mid-name without 'per'/'log' and not as a prefix: kept (reported nowhere)
    result, _ = run(draft)
    assert "base_price_of_locality_index" in result.spec.feature_names()


def test_duplicate_feature_names_dropped_and_reported():
    draft = delhi_draft(features=[F("bhk", "int"), F("bhk", "int"), F("locality", "category")])
    result, _ = run(draft)
    assert result.spec.feature_names() == ["bhk", "locality"]
    assert "bhk: duplicate" in result.guard_removed


def test_all_features_removed_is_an_error():
    with pytest.raises(RequirementError):
        run(delhi_draft(features=[F("listing_id"), F("price_per_sqft")]))


def test_window_used_only_when_both_years_given():
    result, _ = run(delhi_draft(volatility="slow", time_window_start=2000, time_window_end=2010))
    t = result.spec.temporal
    assert (t.mode, t.window_start, t.window_end) == ("window", 2000, 2010)
    result, _ = run(delhi_draft(time_window_start=2000))
    assert result.spec.temporal.mode == "recent_weighted"


def test_llm_clarification_request_is_raised():
    with pytest.raises(NeedsClarification, match="Which city"):
        run(delhi_draft(needs_clarification="Which city do you mean?"))


def test_too_short_problem_rejected_without_llm_call():
    with pytest.raises(NeedsClarification):
        run(delhi_draft(), problem="price")


def test_sample_columns_reach_the_prompt():
    _, backend = run(delhi_draft(), sample_columns=["Area", "BHK", "Locality"])
    assert "Area, BHK, Locality" in backend.prompts[0]
    assert "sample file" not in render_user("predict prices", None)


def test_gold_specs_load_and_score():
    golds = {g.id: g for g in load_gold(GOLD_DIR)}
    assert {"delhi_house_prices", "laptop_prices", "employee_attrition"} <= set(golds)

    result, _ = run(delhi_draft())
    s = score_spec(result.spec, golds["delhi_house_prices"])
    assert s.target_correct and s.task_type_correct
    # locality, built_up_area_sqft, bhk(alias), bathrooms, furnishing, distance_to_metro_km
    assert s.precision == 1.0
    assert s.recall == pytest.approx(6 / 12, abs=1e-3)
    assert s.forbidden_included == []
    assert "floor" in s.missed


def test_scorer_token_matching_on_real_smoke_test_names():
    golds = {g.id: g for g in load_gold(GOLD_DIR)}
    from dsagent.schemas.spec import DatasetSpec

    spec = DatasetSpec(
        problem_text="x" * 20, task_type="regression", target=F("house_price_inr"),
        features=[F("super_built_up_area_sqft"), F("distance_to_nearest_metro_station_km"),
                  F("num_parking_spaces"), F("num_balconies"), F("bathrooms", "int")],
    )
    s = score_spec(spec, golds["delhi_house_prices"])
    assert s.target_correct
    assert set(s.matched) == {"built_up_area_sqft", "distance_to_metro_km", "parking", "bathrooms"}
    assert s.unmatched_predictions == ["num_balconies"]


def test_scorer_flags_forbidden_columns():
    golds = {g.id: g for g in load_gold(GOLD_DIR)}
    from dsagent.schemas.spec import DatasetSpec

    spec = DatasetSpec(
        problem_text="x" * 20, task_type="regression", target=F("price"),
        features=[F("price_per_sqft"), F("bhk", "int"), F("random_thing")],
    )
    s = score_spec(spec, golds["delhi_house_prices"])
    assert s.forbidden_included == ["price_per_sqft"]
    assert s.unmatched_predictions == ["price_per_sqft", "random_thing"]
    assert s.precision == pytest.approx(1 / 3, abs=1e-3)
