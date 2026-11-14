"""Per-run budget (README design principle 8 and section 12.1 item 11).

Exceeding a cap raises BudgetExceeded; the orchestrator turns that into an
`aborted_budget` run with a report instead of silently continuing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dsagent.config import MODEL_PRICES


class BudgetExceeded(RuntimeError):
    pass


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = MODEL_PRICES.get(model)
    if price is None:
        # Unknown model: count tokens but flag cost as unknown rather than guessing.
        return 0.0
    return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


@dataclass
class Budget:
    tokens_max: int
    cost_max: float
    pages_max: int
    tokens_used: int = 0
    cost_used: float = 0.0
    pages_used: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    unknown_price_models: set[str] = field(default_factory=set)

    def check_llm(self) -> None:
        """Call before every paid LLM request."""
        if self.tokens_used >= self.tokens_max:
            raise BudgetExceeded(f"token budget exhausted ({self.tokens_used}/{self.tokens_max})")
        if self.cost_used >= self.cost_max:
            raise BudgetExceeded(f"cost budget exhausted (${self.cost_used:.4f}/${self.cost_max:.2f})")

    def record_llm(
        self, model: str, input_tokens: int, output_tokens: int, billed_cost: float | None = None
    ) -> float:
        """`billed_cost` (e.g. OpenRouter's usage.cost) overrides the price-table estimate."""
        self.llm_calls += 1
        self.tokens_used += input_tokens + output_tokens
        if billed_cost is not None:
            c = billed_cost
        else:
            if model not in MODEL_PRICES:
                self.unknown_price_models.add(model)
            c = cost_usd(model, input_tokens, output_tokens)
        self.cost_used += c
        return c

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def record_page(self) -> None:
        if self.pages_used >= self.pages_max:
            raise BudgetExceeded(f"page budget exhausted ({self.pages_used}/{self.pages_max})")
        self.pages_used += 1
