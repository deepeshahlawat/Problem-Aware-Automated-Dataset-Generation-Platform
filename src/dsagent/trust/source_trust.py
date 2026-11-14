"""Source trust score (design spec section 13.1).

trust = 0.35*authority + 0.20*freshness + 0.20*coverage + 0.15*structure + 0.10*accessibility

Sources that do not permit automated access are excluded, not merely down-scored.
"""

from __future__ import annotations

from datetime import datetime

from dsagent.config import Settings
from dsagent.schemas.source import SourceDescriptor
from dsagent.trust.recency import age_years, recency_weight


class SourceNotPermitted(ValueError):
    pass


def licence_is_open(licence: str | None, settings: Settings) -> bool | None:
    """True = recognised open licence, False = explicitly restrictive, None = unknown."""
    if not licence or licence.strip().lower() in {"unknown", "other", "n/a", ""}:
        return None
    low = licence.lower()
    if any(x in low for x in ("proprietary", "all rights reserved", "no redistribution")):
        return False
    return True if any(x in low for x in settings.discovery.open_licences) else None


def trust_components(
    src: SourceDescriptor,
    coverage: float,
    half_life_years: float,
    settings: Settings,
    now: datetime | None = None,
) -> dict[str, float]:
    if not src.robots_allowed:
        raise SourceNotPermitted(f"{src.source_id}: automated access not permitted")
    open_ = licence_is_open(src.licence, settings)
    if open_ is False:
        raise SourceNotPermitted(f"{src.source_id}: licence forbids reuse ({src.licence})")

    d = settings.discovery
    authority = getattr(settings.authority, src.authority_class)
    freshness = (
        recency_weight(age_years(src.last_updated, now), half_life_years, settings.recency.min_weight)
        if src.last_updated
        else d.freshness_unknown
    )
    structure = d.structure_by_kind.get(src.kind, 0.3)
    accessibility = d.accessibility_open if open_ else d.accessibility_unknown
    return {
        "authority": round(authority, 4),
        "freshness": round(freshness, 4),
        "coverage": round(max(0.0, min(1.0, coverage)), 4),
        "structure": round(structure, 4),
        "accessibility": round(accessibility, 4),
    }


def trust_score(components: dict[str, float], settings: Settings) -> float:
    w = settings.trust
    score = (
        w.authority * components["authority"]
        + w.freshness * components["freshness"]
        + w.coverage * components["coverage"]
        + w.structure * components["structure"]
        + w.accessibility * components["accessibility"]
    )
    return round(min(1.0, max(0.0, score)), 4)
