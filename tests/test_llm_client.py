"""LLM wrapper tests use a fake backend: no network, no API key, no cost."""

import pytest
from pydantic import BaseModel

from dsagent.config import Settings
from dsagent.llm import Budget, BudgetExceeded, LLMCallCache, LLMClient, LLMInvalidOutput, LLMRefusal
from dsagent.llm.budget import cost_usd
from dsagent.llm.client import BackendResult


class Answer(BaseModel):
    label: str
    score: float


class FakeBackend:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def call(self, *, model, system, user, schema, max_tokens):
        self.calls.append(model)
        return self.results.pop(0)


def ok(label="a", score=0.5, model="claude-haiku-5-5", tin=100, tout=20):
    return BackendResult(Answer(label=label, score=score), "", model, tin, tout, "end_turn")


def make_client(results, *, tokens_max=10_000, cost_max=1.0):
    settings = Settings(_env_file=None)
    budget = Budget(tokens_max=tokens_max, cost_max=cost_max, pages_max=10)
    backend = FakeBackend(results)
    client = LLMClient(budget, LLMCallCache(":memory:"), backend, settings)
    return client, backend, budget


def ask(client, user="classify this", version="t1"):
    return client.structured(
        role="cheap", system="sys", user=user, schema=Answer, template_version=version
    )


def test_returns_parsed_and_records_usage():
    client, backend, budget = make_client([ok()])
    assert ask(client) == Answer(label="a", score=0.5)
    assert backend.calls == [client.settings.llm_model_cheap]
    assert budget.tokens_used == 120
    assert budget.cost_used == pytest.approx(cost_usd("claude-haiku-5-5", 100, 20))


def test_identical_request_served_from_cache():
    client, backend, budget = make_client([ok()])
    first = ask(client)
    second = ask(client)
    assert first == second
    assert len(backend.calls) == 1
    assert budget.cache_hits == 1 and budget.llm_calls == 1


def test_template_version_change_misses_cache():
    client, backend, _ = make_client([ok("a"), ok("b")])
    assert ask(client, version="t1").label == "a"
    assert ask(client, version="t2").label == "b"
    assert len(backend.calls) == 2


def test_invalid_output_retried_then_succeeds():
    bad = BackendResult(None, '{"label": "x"}', "claude-haiku-5-5", 50, 5, "end_turn")
    client, backend, budget = make_client([bad, ok()])
    assert ask(client).label == "a"
    assert len(backend.calls) == 2
    assert budget.llm_calls == 2  # failed attempts still cost money


def test_invalid_output_rejected_after_retries():
    bad = BackendResult(None, "not json", "claude-haiku-5-5", 50, 5, "end_turn")
    client, backend, _ = make_client([bad, bad, bad])
    with pytest.raises(LLMInvalidOutput):
        ask(client)
    assert len(backend.calls) == 3


def test_refusal_raises_without_retry():
    refusal = BackendResult(None, "", "claude-haiku-5-5", 10, 0, "refusal")
    client, backend, _ = make_client([refusal, ok()])
    with pytest.raises(LLMRefusal):
        ask(client)
    assert len(backend.calls) == 1


def test_budget_stops_before_paid_call():
    client, backend, _ = make_client([ok(tin=900, tout=200), ok()], tokens_max=1000)
    ask(client, user="first")
    with pytest.raises(BudgetExceeded):
        ask(client, user="second")
    assert len(backend.calls) == 1


def test_cache_hit_allowed_even_when_budget_exhausted():
    client, _, _ = make_client([ok(tin=900, tout=200)], tokens_max=1000)
    ask(client)
    assert ask(client).label == "a"  # replay is free


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "cache.sqlite"
    settings = Settings(_env_file=None)
    b1 = Budget(tokens_max=10_000, cost_max=1, pages_max=1)
    c1 = LLMClient(b1, LLMCallCache(path), FakeBackend([ok("persisted")]), settings)
    ask(c1)
    b2 = Budget(tokens_max=10_000, cost_max=1, pages_max=1)
    c2 = LLMClient(b2, LLMCallCache(path), FakeBackend([]), settings)
    assert ask(c2).label == "persisted"
    assert b2.cache_hits == 1


def test_page_budget():
    b = Budget(tokens_max=1, cost_max=1, pages_max=2)
    b.record_page()
    b.record_page()
    with pytest.raises(BudgetExceeded):
        b.record_page()
