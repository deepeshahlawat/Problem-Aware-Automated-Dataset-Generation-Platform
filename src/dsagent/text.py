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


# Unit/currency qualifiers that end a target name without changing what it measures.
_QUALIFIERS = {"inr", "usd", "eur", "gbp", "rs", "rupees", "dollars", "euros", "lakh", "lakhs",
               "crore", "crores", "amount", "value", "in", "kg", "tonnes", "tons", "sqft", "sqm",
               "km", "hours", "years", "pct", "percent"}


def target_head(target: str) -> str:
    """The noun a target measures: 'house_price_inr' -> 'price',
    'crop_yield_per_hectare' -> 'yield', 'attrition' -> 'attrition'."""
    toks = norm(target).split("_")
    if "per" in toks:
        toks = toks[: toks.index("per")]
    core = [t for t in toks if t not in _QUALIFIERS] or toks
    return core[-1]


def named_as_target_transform(column: str, target: str) -> bool:
    """'price_per_sqft', 'log_price', 'house_price_inr_band' for target 'house_price_inr'."""
    col = norm(column.removeprefix("x_"))
    t = norm(target)
    if col == t or col.startswith(t + "_"):
        return True
    toks = col.split("_")
    return target_head(t) in toks and bool({"per", "log", "ratio"} & set(toks))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0
