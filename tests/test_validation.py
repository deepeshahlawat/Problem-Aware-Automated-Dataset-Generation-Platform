"""Stage 5 tests on small synthetic tables (fixtures for code testing only)."""

import polars as pl
import pytest

from dsagent.config import Settings
from dsagent.schemas import DatasetSpec, Feature, SourceCandidate, ValueRange
from dsagent.stages.integration import integrate
from dsagent.stages.integration.union import SourceTable
from dsagent.stages.validation import build_contract, validate

S = Settings(_env_file=None)


def F(name, dtype="float", unit=None, required=True, **kw):
    return Feature(name=name, dtype=dtype, unit=unit, description=name, required=required, **kw)


SPEC = DatasetSpec(
    problem_text="I want to predict house prices in Delhi", task_type="regression",
    target=F("price", unit="INR"),
    features=[
        F("area_sqft", unit="sqft", expected_range=ValueRange(min=100, max=20000)),
        F("bedrooms", "int"),
        F("furnishing", "category", required=False,
          allowed_values=["Furnished", "Semi-Furnished", "Unfurnished"]),
        F("temperature_change", required=False, expected_range=ValueRange(min=-5, max=5)),
    ],
    min_rows=3,
)


def table(n_extra=8, **overrides):
    base = {
        "price": [6.5e6, None, -1.0, 7e6] + [6e6 + i * 1e5 for i in range(n_extra)],
        "area_sqft": [1200.0, 900, 800, 90000] + [1000.0 + i * 20 for i in range(n_extra)],
        "bedrooms": [2, 2, None, 3] + [2] * n_extra,
        "furnishing": ["furnished", "semi furnished", "luxury", None] + ["furnished"] * n_extra,
        "temperature_change": [-2.0, 0.0, 1.0, 9.0] + [0.5] * n_extra,
    }
    base.update(overrides)
    m = len(base["price"])
    base.update({
        "_row_ref": [f"a:{i}" for i in range(m)], "_source_id": ["a"] * m,
        "_method": ["file"] * m, "_source_trust": [0.7] * m, "_retrieved_at": ["t"] * m,
    })
    return pl.DataFrame(base)


def test_contract_is_derived_from_the_spec():
    rules = {(r.column, r.check): r for r in build_contract(SPEC, S)}
    assert rules[("price", "not_null")].severity == "hard"
    assert rules[("price", "ge")].severity == "hard"            # price can't be negative
    assert rules[("area_sqft", "ge")].severity == "hard"
    assert rules[("area_sqft", "in_range")].severity == "soft"
    assert rules[("bedrooms", "not_null")].severity == "soft"   # required feature
    assert ("furnishing", "not_null") not in rules              # optional feature
    assert rules[("furnishing", "isin")].params["values"] == ["furnished", "semi furnished", "unfurnished"]
    # a feature that may legitimately be negative gets no non-negativity rule
    assert ("temperature_change", "ge") not in rules
    assert rules[("temperature_change", "in_range")].params == {"min": -5, "max": 5}


def test_hard_failures_excluded_with_reasons_soft_failures_kept():
    res = validate(table(), SPEC, settings=S)
    ex = dict(zip(res.excluded["_row_ref"], res.excluded["_exclusion_reason"]))
    assert set(ex) == {"a:1", "a:2"}
    assert "price:not_null" in ex["a:1"] and "greater_than_or_equal_to" in ex["a:2"]
    kept = res.table.filter(pl.col("_row_ref") == "a:3").row(0, named=True)
    assert any("area_sqft:in_range" in f for f in kept["_flags"])
    assert any("temperature_change:in_range" in f for f in kept["_flags"])
    assert res.report.rows_valid == 10 and res.report.rows_excluded == 2


def test_row_confidence_formula():
    res = validate(table(), SPEC, agreement={"a:0": "confirmed", "a:4": "disagree"}, settings=S)
    conf = dict(zip(res.table["_row_ref"], res.table["_row_confidence"]))
    assert conf["a:0"] == pytest.approx(0.7 * 1.0 * 1.0 * 1.0)   # confirmed by 2 sources
    assert conf["a:5"] == pytest.approx(0.7 * 1.0 * 0.9 * 1.0)   # single source
    assert conf["a:4"] == pytest.approx(0.7 * 1.0 * 0.6 * 1.0)   # sources disagree
    assert conf["a:3"] == pytest.approx(0.7 * 1.0 * 0.9 * 0.8 ** 2)  # two soft failures
    assert "sources_disagree" in res.table.filter(pl.col("_row_ref") == "a:4")["_flags"][0].to_list()
    assert res.report.agreement == {"confirmed": 1, "single": 8, "disagree": 1}


def test_outliers_flagged_not_deleted_and_bounds_ignore_excluded_rows():
    t = table(price=[6.5e6, None, -9e12, 7e6] + [6e6 + i * 1e5 for i in range(7)] + [9e9])
    res = validate(t, SPEC, settings=S)
    refs = res.table["_row_ref"].to_list()
    assert "a:11" in refs  # the 9e9 luxury price is kept
    flagged = res.table.filter(pl.col("_row_ref") == "a:11")["_flags"][0].to_list()
    assert "outlier:price" in flagged
    assert res.report.outliers["price"] == 1  # -9e12 was excluded, so it neither counts nor skews bounds


def test_gate_target_missing_or_too_few_rows_needs_review():
    res = validate(table().drop("price"), SPEC, settings=S)
    assert res.report.status == "needs_review" and not res.deliverable
    assert any("target column 'price' missing" in c for c in res.report.critical)

    few = validate(table(n_extra=0), SPEC.model_copy(update={"min_rows": 50}), settings=S)
    assert any("valid rows" in c for c in few.report.critical)


def test_gate_too_many_exclusions():
    t = table(n_extra=0)  # 2 of 4 rows fail hard checks = 50% > 30%
    res = validate(t, SPEC.model_copy(update={"min_rows": 1}), settings=S)
    assert any("excluded by hard checks" in c for c in res.report.critical)


def test_clean_table_passes_and_report_fields():
    t = table(n_extra=6).filter(~pl.col("_row_ref").is_in(["a:1", "a:2", "a:3"]))
    res = validate(t, SPEC, settings=S, duplication_rate=0.1)
    r = res.report
    assert r.status == "passed" and r.critical == []
    assert r.completeness["price"] == 1.0
    assert r.trust_bands == {"high": 0, "medium": r.rows_valid, "low": 0}
    assert r.duplication_rate == 0.1 and r.llm_derived_share == 0.0
    assert r.contract and r.mean_row_confidence == pytest.approx(0.63)


def test_stage4_to_stage5_hand_off():
    cand = SourceCandidate(source_id="s", url="u", kind="file", role="entity", covers_features=[],
                           feature_map={"price": "Price", "area_sqft": "Area", "bedrooms": "BHK"},
                           robots_allowed=True, trust_score=0.6)
    raw = pl.DataFrame({"Price": ["65 Lac", "-5", "70 Lac", "72 Lac"], "Area": ["1200", "900", "1000", "1100"],
                        "BHK": ["2", "2", "3", "3"]})
    spec = SPEC.model_copy(update={"features": SPEC.features[:3], "min_rows": 2})
    integ = integrate(spec, [SourceTable(cand, raw)], settings=S)
    res = validate(integ.table, spec, integ.report.dedupe.agreement, S)
    assert res.excluded["_row_ref"].to_list() == ["s:1"]  # negative price from the raw file
    assert res.report.status == "passed"
