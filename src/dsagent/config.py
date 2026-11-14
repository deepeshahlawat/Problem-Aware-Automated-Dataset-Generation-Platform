"""Central configuration.

README rule 9: thresholds, weights and half-lives live here, not hard-coded in stages.
Values are starting defaults to be tuned during evaluation, not proven constants.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPrice(BaseModel):
    """USD per 1M tokens."""

    input: float
    output: float


# List prices (USD / 1M tokens). Fallback for cost reporting only: OpenRouter
# responses carry the actual billed cost, which takes precedence when present.
MODEL_PRICES: dict[str, ModelPrice] = {
    # Anthropic first-party, cached 2026-11-03
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00),
    "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50),
    # OpenRouter, from /api/v1/models on 2026-11-14
    "meta/muse-ember": ModelPrice(input=0.10, output=0.50),
    "openai/gpt-6-luna": ModelPrice(input=0.10, output=0.50),
    "xiaomi/mimo-v2.6-flash": ModelPrice(input=0.14, output=0.28),
    "anthropic/claude-haiku-5.5": ModelPrice(input=0.10, output=0.50),
    "deepseek/deepseek-v4.1-flash-1017": ModelPrice(input=0.038, output=0.50),
    "openai/gpt-6.1-terra": ModelPrice(input=1.00, output=5.00),
}


class TrustWeights(BaseModel):
    """README section 13.1."""

    authority: float = 0.35
    freshness: float = 0.20
    coverage: float = 0.20
    structure: float = 0.15
    accessibility: float = 0.10


class AuthorityScores(BaseModel):
    official: float = 1.0
    academic_or_open_repo: float = 0.8
    community: float = 0.6
    commercial_scrape: float = 0.5
    unknown: float = 0.3


class ExtractionFactors(BaseModel):
    """README section 13.3. LLM + unverified is rejected outright (no factor)."""

    api: float = 1.0
    file: float = 1.0
    html_table: float = 0.95
    jsonld: float = 0.9
    llm_verified: float = 0.75
    derived: float = 1.0


class RelevanceConfig(BaseModel):
    """README section 16. Fusion weights are justified by ablation E11, not assumed."""

    w_model: float = 0.40
    w_stat: float = 0.30
    w_semantic: float = 0.30
    tier_high: float = 0.70
    tier_medium: float = 0.40
    tier_low: float = 0.15
    id_unique_ratio: float = 0.95
    leakage_r2: float = 0.98
    leakage_auc: float = 0.99


class SelectionConfig(BaseModel):
    """README section 16.4 (three strategies, nested CV)."""

    epsilon_relative: float = 0.005
    max_candidate_sources: int = 8
    max_no_improvement: int = 2
    outer_folds: int = 5
    inner_folds: int = 3
    random_seed: int = 42


class RecencyConfig(BaseModel):
    """README section 14."""

    min_weight: float = 0.05
    half_life_fast: float = 2.0
    half_life_medium: float = 5.0
    half_life_slow: float = 15.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DSAGENT_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # "openrouter" (default: cheapest to develop on) or "anthropic".
    # Model choice rationale: docs/MODEL_CHOICE.md
    llm_provider: Literal["openrouter", "anthropic"] = "openrouter"
    llm_model_planner: str = "meta/muse-ember"
    llm_model_cheap: str = "openai/gpt-6-luna"
    openrouter_api_key: Optional[SecretStr] = Field(
        default=None, validation_alias=AliasChoices("OPENROUTER_API_KEY", "DSAGENT_OPENROUTER_API_KEY")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_max_tokens: int = 16000
    llm_max_retries_invalid_output: int = 2
    prompt_cache_namespace: str = "v1"

    max_tokens_per_run: int = 500_000
    max_cost_per_run: float = 2.00
    max_pages_per_run: int = 200

    cache_db_path: Path = Path(".cache/llm_cache.sqlite")
    source_allowlist_path: Path = Path("configs/allowlist.txt")
    user_agent: str = "dsagent-research-bot/0.1 (academic project)"

    trust: TrustWeights = Field(default_factory=TrustWeights)
    authority: AuthorityScores = Field(default_factory=AuthorityScores)
    extraction: ExtractionFactors = Field(default_factory=ExtractionFactors)
    relevance: RelevanceConfig = Field(default_factory=RelevanceConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    recency: RecencyConfig = Field(default_factory=RecencyConfig)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
