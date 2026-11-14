"""Value normalisation (design spec Stage 4 step 1).

Turns raw strings into typed values in the spec's units:
  numbers  "6,500,000", "65 Lac", "1.2 Cr", "₹ 15,000", "1.37kg", "8GB"
  units    sq m -> sq ft, lb -> kg, cm -> inch, mi -> km  (fixed physical constants)
  currency only with a configured rate; otherwise kept as-is and reported
  category trimmed, case-folded, separators unified ("Sector-25 " -> "sector 25")
Unparseable values become null and are counted, never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

from dsagent.config import IntegrationConfig
from dsagent.text import norm

_MISSING = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "not available"}

# Word multipliers (Indian and western).
# Single letters like "L" (lakh or litre?) and "M" (million or metre?) are ambiguous
# and deliberately NOT treated as multipliers.
_MULT = {
    "k": 1e3, "thousand": 1e3,
    "lac": 1e5, "lacs": 1e5, "lakh": 1e5, "lakhs": 1e5,
    "cr": 1e7, "crore": 1e7, "crores": 1e7,
    "mn": 1e6, "million": 1e6,
    "bn": 1e9, "billion": 1e9,
}
_CURRENCY_SYMBOLS = {"₹": "INR", "rs": "INR", "inr": "INR", "€": "EUR", "eur": "EUR", "euro": "EUR",
                     "euros": "EUR", "$": "USD", "usd": "USD", "£": "GBP", "gbp": "GBP"}

# Physical unit families: unit token -> (family, factor to the family's base unit)
_UNITS: dict[str, tuple[str, float]] = {
    "sqft": ("area", 1.0), "sq_ft": ("area", 1.0), "sqfeet": ("area", 1.0), "ft2": ("area", 1.0),
    "sqm": ("area", 10.7639104), "sq_m": ("area", 10.7639104), "m2": ("area", 10.7639104),
    "sqyd": ("area", 9.0), "sq_yd": ("area", 9.0), "sqyard": ("area", 9.0),
    "kg": ("mass", 1.0), "kgs": ("mass", 1.0), "g": ("mass", 0.001), "lb": ("mass", 0.45359237),
    "lbs": ("mass", 0.45359237),
    "inch": ("length_small", 1.0), "inches": ("length_small", 1.0), "in": ("length_small", 1.0),
    "cm": ("length_small", 1 / 2.54), "mm": ("length_small", 1 / 25.4),
    "km": ("distance", 1.0), "m": ("distance", 0.001), "mi": ("distance", 1.609344),
    "miles": ("distance", 1.609344),
    "gb": ("storage", 1.0), "tb": ("storage", 1024.0), "mb": ("storage", 1 / 1024),
}


@dataclass
class ColumnReport:
    feature: str
    source_column: str
    parsed: int = 0
    failed: int = 0
    missing: int = 0
    unit_from: Optional[str] = None
    unit_to: Optional[str] = None
    converted: bool = False
    notes: list[str] = field(default_factory=list)


def is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and v != v) or str(v).strip().lower() in _MISSING


def unit_key(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    k = norm(u).replace("square_feet", "sqft").replace("square_meter", "sqm").replace("square_metre", "sqm")
    k = {"sq_feet": "sqft", "sq_meter": "sqm", "sq_metre": "sqm", "kilogram": "kg",
         "kilograms": "kg", "pounds": "lbs"}.get(k, k)
    return k


def currency_of(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    k = u.strip().lower()
    if k in _CURRENCY_SYMBOLS:
        return _CURRENCY_SYMBOLS[k]
    return u.strip().upper() if re.fullmatch(r"[A-Za-z]{3}", u.strip()) else None


def unit_hint_from_name(column: str) -> Optional[str]:
    """'Price_euros' -> 'EUR'; 'Area (sq. m.)' -> 'sqm'; 'Weight_kg' -> 'kg'."""
    toks = norm(column).split("_")
    joined = "_".join(toks)
    for multi in ("sq_ft", "sq_m", "sq_yd"):
        if multi in joined:
            return multi.replace("_", "")
    for t in reversed(toks):
        if t in _CURRENCY_SYMBOLS:
            return _CURRENCY_SYMBOLS[t]
        if t in _UNITS and len(t) > 1:
            return t
    return None


_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_number(v: Any) -> tuple[Optional[float], Optional[str]]:
    """Returns (value, unit-or-currency found in the text)."""
    if is_missing(v):
        return None, None
    if isinstance(v, bool):
        return float(v), None
    if isinstance(v, (int, float)):
        return float(v), None
    s = str(v).strip().lower()
    found_unit = None
    for sym, cur in _CURRENCY_SYMBOLS.items():
        if not sym.isalpha() and sym in s:
            found_unit = cur
            s = s.replace(sym, " ")
    s = s.replace(",", "")
    # "sq. ft." / "sq ft" / "square feet" -> "sqft" so multi-word units are recognised
    s = re.sub(r"\bsq(?:uare)?\.?\s*(?:ft|feet|foot)\.?", "sqft", s)
    s = re.sub(r"\bsq(?:uare)?\.?\s*(?:m|mt|mtr|metres?|meters?)\.?(?![a-z])", "sqm", s)
    s = re.sub(r"\bsq(?:uare)?\.?\s*(?:yd|yards?)\.?", "sqyd", s)
    m = _NUM.search(s)
    if not m:
        return None, found_unit
    value = float(m.group())
    words = re.findall(r"[a-z0-9]+", s[m.end():])
    if words and words[0] in _MULT:
        value *= _MULT[words[0]]
        words = words[1:]
    if words:
        w = words[0]
        if w in _CURRENCY_SYMBOLS:
            found_unit = _CURRENCY_SYMBOLS[w]
        elif unit_key(w) in _UNITS:
            found_unit = unit_key(w)
    pre = re.findall(r"[a-z]+", s[: m.start()])
    if found_unit is None and pre and pre[-1] in _CURRENCY_SYMBOLS:
        found_unit = _CURRENCY_SYMBOLS[pre[-1]]
    return value, found_unit


def convert(value: float, from_u: Optional[str], to_u: Optional[str], cfg: IntegrationConfig) -> tuple[Optional[float], str]:
    """Returns (converted value or None if impossible, status)."""
    if from_u is None or to_u is None:
        return value, "unknown_unit"
    fu, tu = unit_key(from_u), unit_key(to_u)
    if fu == tu:
        return value, "same"
    if fu in _UNITS and tu in _UNITS:
        ff, fx = _UNITS[fu]
        tf, tx = _UNITS[tu]
        if ff != tf:
            return None, f"incompatible units {from_u}->{to_u}"
        return value * fx / tx, "converted"
    fc, tc = currency_of(from_u), currency_of(to_u)
    if fc and tc:
        if fc == tc:
            return value, "same"
        rates = {cfg.currency_base.upper(): 1.0, **{k.upper(): r for k, r in cfg.currency_rates.items()}}
        if fc in rates and tc in rates:
            return value * rates[fc] / rates[tc], "converted"
        return None, f"no configured rate for {fc}->{tc}"
    return value, "unknown_unit"


def normalise_category(v: Any) -> Optional[str]:
    if is_missing(v):
        return None
    s = re.sub(r"[\s_\-/]+", " ", str(v)).strip().casefold()
    return s or None


_TRUE = {"yes", "y", "true", "t", "1", "available", "present"}
_FALSE = {"no", "n", "false", "f", "0", "not available", "absent"}


def parse_bool(v: Any) -> Optional[bool]:
    if is_missing(v):
        return None
    s = str(v).strip().lower()
    return True if s in _TRUE else False if s in _FALSE else None


def parse_date(v: Any) -> Optional[date]:
    if is_missing(v):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%b %Y", "%B %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    if re.fullmatch(r"(19|20)\d{2}", s):
        return date(int(s), 1, 1)
    return None


def normalise_column(
    values: list[Any], dtype: str, target_unit: Optional[str], column: str, feature: str,
    cfg: IntegrationConfig, declared_unit: Optional[str] = None,
) -> tuple[list[Any], ColumnReport]:
    """Normalise one source column to the spec feature's dtype and unit."""
    rep = ColumnReport(feature=feature, source_column=column)
    out: list[Any] = []
    if dtype in ("int", "float"):
        col_unit = declared_unit or unit_hint_from_name(column)
        seen_units: dict[str, int] = {}
        for v in values:
            if is_missing(v):
                out.append(None)
                rep.missing += 1
                continue
            x, u = parse_number(v)
            if x is None:
                out.append(None)
                rep.failed += 1
                continue
            u = u or col_unit
            if u:
                seen_units[u] = seen_units.get(u, 0) + 1
            if target_unit and u:
                y, status = convert(x, u, target_unit, cfg)
                if y is None:
                    out.append(x)  # keep original value; unit problem reported below
                    if status not in rep.notes:
                        rep.notes.append(status)
                    rep.parsed += 1
                    continue
                if status == "converted":
                    rep.converted = True
                x = y
            out.append(int(round(x)) if dtype == "int" else x)
            rep.parsed += 1
        if seen_units:
            rep.unit_from = max(seen_units, key=seen_units.get)
            if len(seen_units) > 1:
                rep.notes.append(f"mixed units in source: {sorted(seen_units)}")
        elif target_unit:
            rep.notes.append(f"unit not stated in source; assumed {target_unit}")
        if not seen_units:
            rep.unit_to = target_unit  # assumption recorded in notes above
        elif target_unit and convert(1, rep.unit_from, target_unit, cfg)[0] is not None:
            rep.unit_to = target_unit
        else:
            rep.unit_to = rep.unit_from  # could not convert: values stay in the source unit
    elif dtype == "bool":
        for v in values:
            b = parse_bool(v)
            out.append(b)
            rep.missing += is_missing(v)
            rep.failed += (b is None and not is_missing(v))
            rep.parsed += b is not None
    elif dtype == "date":
        for v in values:
            d = parse_date(v)
            out.append(d)
            rep.missing += is_missing(v)
            rep.failed += (d is None and not is_missing(v))
            rep.parsed += d is not None
    else:  # category / text
        for v in values:
            c = normalise_category(v) if dtype == "category" else (None if is_missing(v) else str(v).strip())
            out.append(c)
            rep.missing += c is None
            rep.parsed += c is not None
    return out, rep
