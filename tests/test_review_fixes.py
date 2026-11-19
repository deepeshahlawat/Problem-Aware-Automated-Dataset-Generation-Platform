"""Regression tests for the code-review findings fixed on 2026-11-14."""
from datetime import date

import polars as pl
import pytest

from dsagent.config import IntegrationConfig, Settings
from dsagent.llm import Budget, BudgetExceeded, LLMCallCache, LLMClient
from dsagent.llm.client import BackendResult, LLMError, OpenRouterBackend
from dsagent.schemas import DatasetSpec, Feature
from dsagent.stages.discovery.catalogs import HuggingFaceCatalog
from dsagent.stages.integration.normalise import normalise_column, unit_hint_from_name
from dsagent.stages.integration.union import _extra
from dsagent.stages.relevance.leakage import _name_leak
from dsagent.stages.relevance.modeling import fit_encoder
from dsagent.stages.requirement import _guard_reason
from dsagent.text import named_as_target_transform, target_head
from pydantic import BaseModel


def F(n, d="float"):
    return Feature(name=n, dtype=d, description=n)


def test_date_categoricals_are_encoded():
    df = pl.DataFrame({"d": [date(2020, 1, 1), date(2021, 1, 1)]})
    assert not any(v != v for v in fit_encoder(df, ["d"]).transform(df).ravel())


def test_text_extras_with_digits_stay_text():
    assert _extra(["Rohini Sector 25", "Dwarka Sector 7"])[1] == pl.Utf8
    assert _extra(["65 Lac", "1.2 Cr"])[1] == pl.Float64


def test_name_leak_rules_handle_multi_token_targets():
    assert target_head("house_price_inr") == "price"
    assert target_head("crop_yield_per_hectare") == "yield"
    assert named_as_target_transform("price_per_sqft", "house_price_inr")
    assert _name_leak("x_price_per_sqft", "house_price_inr")
    assert _guard_reason(F("price_per_sqft"), F("house_price_inr"))
    assert not named_as_target_transform("area_sqft", "house_price_inr")
    assert not named_as_target_transform("rainfall_per_hectare", "crop_yield_per_hectare")


def test_mixed_units_converted_even_without_spec_unit():
    vals, rep = normalise_column(["1TB", "256GB", "512GB"], "float", None, "Storage", "storage", IntegrationConfig())
    assert vals == [1024.0, 256.0, 512.0]


def test_in_is_not_inches():
    assert unit_hint_from_name("Price_in_Lakhs") is None
    assert unit_hint_from_name("Screen_inches") == "inch"


def test_page_budget_stops_discovery():
    class HTTP:
        def get(self, url):
            raise AssertionError("must not fetch once the budget is spent")
    cat = HuggingFaceCatalog(Budget(1, 1, pages_max=0), "ua", http_client=HTTP())
    spec = DatasetSpec(problem_text="predict laptop prices", task_type="regression",
                       target=F("price"), features=[F("ram_gb")])
    with pytest.raises(BudgetExceeded):
        cat.find(spec)


class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._p, self.headers = status, payload or {}, headers or {}
        self.text = str(payload)

    def json(self):
        return self._p


class _A(BaseModel):
    x: int


def test_openrouter_retries_transient_errors(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    ok = _Resp(200, {"model": "m", "choices": [{"message": {"content": '{"x": 1}'}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    seq = [_Resp(429, headers={"retry-after": "1"}), _Resp(502), ok]

    class HTTP:
        def post(self, url, headers, json):
            return seq.pop(0)
    r = OpenRouterBackend("k", "https://x", "ua", HTTP()).call(model="m", system="", user="", schema=_A, max_tokens=5)
    assert r.raw_text == '{"x": 1}' and not seq


def test_truncation_retry_doubles_max_tokens():
    limits = []

    class B:
        def call(self, *, model, system, user, schema, max_tokens):
            limits.append(max_tokens)
            if len(limits) == 1:
                return BackendResult(None, "", model, 1, 1, "max_tokens")
            return BackendResult(_A(x=1), "", model, 1, 1, "stop")
    c = LLMClient(Budget(10_000, 1, 1), LLMCallCache(":memory:"), B(), Settings(_env_file=None))
    c.structured(role="cheap", system="", user="u", schema=_A, template_version="t", max_tokens=1000)
    assert limits == [1000, 2000]


def test_openrouter_retries_error_inside_http_200(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    overloaded = _Resp(200, {"error": {"message": "Our servers are currently overloaded.", "code": 502}})
    ok = _Resp(200, {"model": "m", "choices": [{"message": {"content": '{"x": 2}'}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    seq = [overloaded, ok]

    class HTTP:
        def post(self, url, headers, json):
            return seq.pop(0)
    r = OpenRouterBackend("k", "https://x", "ua", HTTP()).call(model="m", system="", user="", schema=_A, max_tokens=5)
    assert r.raw_text == '{"x": 2}'


def test_unnamed_rate_column_is_a_name_leak():
    assert named_as_target_transform("x_per_sqft", "house_price_inr")
    assert named_as_target_transform("per_kg", "price")
    assert not named_as_target_transform("rooms_per_floor", "house_price_inr")
