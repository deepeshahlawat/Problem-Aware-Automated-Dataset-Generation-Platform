"""The single entry point for LLM calls (README rule 2).

Every LLM call in the project goes through `LLMClient.structured()`, which:
  * constrains output to a Pydantic schema (structured outputs) and validates it,
  * retries on invalid output, then rejects (design principle 2),
  * looks up / stores the LLM call cache (section 12.2),
  * counts tokens and cost against the run budget (principle 8).
No other module may import the Anthropic SDK directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from dsagent.config import Settings, get_settings
from dsagent.llm.budget import Budget
from dsagent.llm.cache import LLMCallCache, make_key

T = TypeVar("T", bound=BaseModel)
ModelRole = Literal["planner", "cheap"]


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


class LLMInvalidOutput(LLMError):
    pass


@dataclass
class BackendResult:
    parsed: BaseModel | None
    raw_text: str
    model_served: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None


class LLMBackend(Protocol):
    def call(
        self, *, model: str, system: str, user: str, schema: type[BaseModel], max_tokens: int
    ) -> BackendResult: ...


# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
# Claude Haiku 5.5 has no server-side fallback.
_SERVER_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5"}


class AnthropicBackend:
    """Claude via the official SDK, using `messages.parse` with a Pydantic output format.

    Refusal fallback is enabled by default for models that support it: if a safety
    classifier declines, the API retries on a fallback model server-side.
    """

    def __init__(self, client=None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client

    def call(
        self, *, model: str, system: str, user: str, schema: type[BaseModel], max_tokens: int
    ) -> BackendResult:
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if model in _SERVER_FALLBACK_MODELS:
            response = self._client.beta.messages.parse(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            response = self._client.messages.parse(**kwargs)

        raw_text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        return BackendResult(
            parsed=response.parsed_output,
            raw_text=raw_text,
            model_served=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )


class LLMClient:
    def __init__(
        self,
        budget: Budget,
        cache: LLMCallCache,
        backend: LLMBackend | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.budget = budget
        self.cache = cache
        self.backend = backend or AnthropicBackend()

    def model_for(self, role: ModelRole) -> str:
        return (
            self.settings.llm_model_planner if role == "planner" else self.settings.llm_model_cheap
        )

    def structured(
        self,
        *,
        role: ModelRole,
        system: str,
        user: str,
        schema: type[T],
        template_version: str,
        max_tokens: int | None = None,
    ) -> T:
        model = self.model_for(role)
        output_schema = schema.model_json_schema()
        key = make_key(
            model=model,
            template_version=f"{self.settings.prompt_cache_namespace}:{template_version}",
            system=system,
            user=user,
            output_schema=output_schema,
        )

        cached = self.cache.get(key)
        if cached is not None:
            self.budget.record_cache_hit()
            return schema.model_validate_json(cached.response_json)

        attempts = 1 + self.settings.llm_max_retries_invalid_output
        last_error: Exception | None = None
        for _ in range(attempts):
            self.budget.check_llm()
            result = self.backend.call(
                model=model,
                system=system,
                user=user,
                schema=schema,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
            )
            self.budget.record_llm(result.model_served, result.input_tokens, result.output_tokens)

            if result.stop_reason == "refusal":
                raise LLMRefusal(f"model declined the request ({model})")
            if result.stop_reason == "max_tokens":
                last_error = LLMInvalidOutput("output truncated at max_tokens")
                continue

            try:
                parsed = (
                    schema.model_validate(result.parsed.model_dump())
                    if result.parsed is not None
                    else schema.model_validate_json(result.raw_text)
                )
            except ValidationError as exc:
                last_error = exc
                continue

            self.cache.put(
                key,
                model_requested=model,
                model_served=result.model_served,
                template_version=template_version,
                response_json=parsed.model_dump_json(),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
            return parsed

        raise LLMInvalidOutput(
            f"no valid {schema.__name__} after {attempts} attempts: {last_error}"
        )


def dumps_for_prompt(obj) -> str:
    """Deterministic JSON for prompts (stable cache keys)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
