"""Name normalisation and token matching shared by discovery, schema matching and E1."""

from __future__ import annotations

import re

# Words that carry no meaning for matching names.
FILLER = {"num", "number", "no", "of", "the", "to", "from", "total", "nearest", "station",
          "count", "is", "has", "in", "a", "an", "and"}


def norm(name: str) -> str:
    """'Area (sq. ft.)' -> 'area_sq_ft'; 'Per_Sqft' -> 'per_sqft'."""
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", name.strip())
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _singular(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def tokens(name: str) -> frozenset[str]:
    return frozenset(_singular(t) for t in norm(name).split("_") if t and t not in FILLER)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0
