"""Recency weighting (design spec section 14).

weight = max(min_weight, 0.5 ** (age_years / half_life_years))
Used for source freshness (Stage 2) and row sample weights (Stage 6).
"""

from __future__ import annotations

from datetime import datetime, timezone


def age_years(when: datetime, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (now - when).days / 365.25)


def recency_weight(age: float, half_life_years: float, min_weight: float = 0.05) -> float:
    if half_life_years <= 0:
        raise ValueError("half_life_years must be > 0")
    return max(min_weight, 0.5 ** (max(age, 0.0) / half_life_years))
