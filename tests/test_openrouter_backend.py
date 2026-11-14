"""OpenRouter backend tests with a stubbed HTTP client (no network, no key needed)."""

import json

import pytest
from pydantic import BaseModel

from dsagent.config import Settings
from dsagent.llm import Budget, LLMCallCache, LLMClient
from dsagent.llm.client import LLMError, OpenRouterBackend, make_backend, to_strict_json_schema
from dsagent.llm.prompts.requirement import SpecDraft


class Answer(BaseModel):
    label: str
    title: str | None = None  # field named like a stripped keyword must survive
    score: float = 0.5


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeHTTP:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies = []

    def post(self, url, headers, json):
        self.bodies.append(json)
        self.headers = headers
        return self.responses.pop(0)


def completion(content, finish="stop", cost=0.00042, model="meta/muse-ember"):
    return FakeResponse(200, {
        "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 300, "completion_tokens": 120, "cost": cost},
    })


def backend(responses):
    http = FakeHTTP(responses)
    return OpenRouterBackend("sk-or-test", "https://openrouter.ai/api/v1", "ua", http), http


def test_strict_schema_closes_objects_and_requires_all_fields():
    s = to_strict_json_schema(Answer.model_json_schema())
    assert s["additionalProperties"] is False
    assert set(s["required"]) == {"label", "title", "score"}
    assert "title" in s["properties"]  # property name kept
    assert "title" not in s  # schema-level 'title' keyword stripped
    assert "default" not in json.dumps(s)


def test_strict_schema_handles_nested_defs():
    s = to_strict_json_schema(SpecDraft.model_json_schema())
    for name, sub in s["$defs"].items():
        if sub.get("type") == "object":
            assert sub["additionalProperties"] is False, name
            assert set(sub["required"]) == set(sub["properties"]), name
    assert "minItems" not in json.dumps(s)


def test_request_shape_and_result_parsing():
    b, http = backend([completion('{"label": "a", "title": null, "score": 0.9}')])
    r = b.call(model="meta/muse-ember", system="sys", user="hi", schema=Answer, max_tokens=500)
    body = http.bodies[0]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["provider"] == {"require_parameters": True}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert http.headers["Authorization"] == "Bearer sk-or-test"
    assert (r.input_tokens, r.output_tokens, r.billed_cost) == (300, 120, 0.00042)
    assert r.stop_reason == "stop"


def test_length_finish_maps_to_max_tokens():
    b, _ = backend([completion('{"label":', finish="length")])
    assert b.call(model="m", system="", user="", schema=Answer, max_tokens=5).stop_reason == "max_tokens"


def test_http_errors_raise():
    b, _ = backend([FakeResponse(402, {"error": {"message": "no credits"}})])
    with pytest.raises(LLMError, match="402"):
        b.call(model="m", system="", user="", schema=Answer, max_tokens=5)


def test_missing_key_gives_clear_error():
    s = Settings(_env_file=None, llm_provider="openrouter")
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        make_backend(s)


def test_end_to_end_through_client_uses_billed_cost_and_cache():
    b, http = backend([completion('{"label": "x", "title": "t", "score": 1.0}', cost=0.0011)])
    budget = Budget(100_000, 1.0, 10)
    client = LLMClient(budget, LLMCallCache(":memory:"), b, Settings(_env_file=None))
    for _ in range(2):
        out = client.structured(role="planner", system="s", user="u", schema=Answer, template_version="t")
    assert out == Answer(label="x", title="t", score=1.0)
    assert len(http.bodies) == 1  # second call replayed from cache
    assert budget.cost_used == pytest.approx(0.0011)
    assert budget.cache_hits == 1
