"""Stage 4 tests on tiny hand-written fixtures (synthetic, for code testing only)."""

import json

import polars as pl
import pytest

from dsagent.config import IntegrationConfig, Settings
from dsagent.schemas import DatasetSpec, Feature, SourceCandidate
from dsagent.stages.integration import ContextJoin, integrate
from dsagent.stages.integration.dedupe import dedupe
from dsagent.stages.integration.enrich import haversine_km, key_join, nearest_distance
from dsagent.stages.integration.impute import apply_plan, plan_imputation
from dsagent.stages.integration.normalise import (
    convert, normalise_category, normalise_column, parse_number, unit_hint_from_name,
)
from dsagent.stages.integration.union import SourceTable, union_sources

CFG = IntegrationConfig()
S = Settings(_env_file=None)


def F(name, dtype="float", unit=None, required=True):
    return Feature(name=name, dtype=dtype, unit=unit, description=name, required=required)


SPEC = DatasetSpec(
    problem_text="I want to predict house prices in Delhi", task_type="regression",
    target=F("price", unit="INR"),
    features=[F("area_sqft", unit="sqft"), F("bedrooms", "int"), F("locality", "category"),
              F("latitude", required=False), F("longitude", required=False)],
    region="Delhi",
)


def cand(sid, fmap, trust, kind="file"):
    return SourceCandidate(source_id=sid, url="u", kind=kind, role="entity", covers_features=list(fmap),
                           feature_map=fmap, robots_allowed=True, trust_score=trust)


def sources():
    a = SourceTable(cand("a", {"price": "Price", "area_sqft": "Area", "bedrooms": "BHK", "locality": "Locality",
                               "latitude": "lat", "longitude": "lon"}, 0.7), pl.DataFrame({
        "Price": ["65 Lac", "1.2 Cr", "52 Lac", "52 Lac"],
        "Area": ["1200 sq ft", "100 sqm", "650", "650"],
        "BHK": ["2", "3", "2", "2"],
        "Locality": ["Rohini Sector-25", "Lajpat Nagar", "Dwarka Sec 7", "Dwarka Sec 7"],
        "lat": ["28.70", "28.57", "28.58", "28.58"],
        "lon": ["77.10", "77.24", "77.05", "77.05"],
        "Per_Sqft": ["5416", "11148", "8000", "8000"],
    }))
    # b re-posts a's Rohini listing at a different price, plus one new listing
    b = SourceTable(cand("b", {"price": "cost", "area_sqft": "size", "bedrooms": "rooms", "locality": "area_name",
                               "latitude": "lat", "longitude": "lon"}, 0.5, kind="api"), pl.DataFrame({
        "cost": ["₹ 7,000,000", "₹ 9,500,000"],
        "size": ["1200", "1500"],
        "rooms": ["2", "3"],
        "area_name": ["rohini sector 25", "Saket"],
        "lat": ["28.70", "28.52"],
        "lon": ["77.10", "77.21"],
    }))
    return [a, b]


# --- normalisation ----------------------------------------------------------------

@pytest.mark.parametrize("raw, value, unit", [
    ("6,500,000", 6_500_000, None), ("65 Lac", 6_500_000, None), ("1.2 Cr", 12_000_000, None),
    ("₹ 15,000", 15_000, "INR"), ("Rs. 45 lakh", 4_500_000, "INR"), ("€1,339.69", 1339.69, "EUR"),
    ("1,450 Sq. Ft.", 1450, "sqft"), ("111.5 sqm", 111.5, "sqm"), ("1.37kg", 1.37, "kg"),
    ("8GB", 8, "gb"), ("2.5 mn", 2_500_000, None),
    ("12 L", 12, None),  # ambiguous (lakh? litre?) -> not multiplied
    ("2.5 M", 2.5, "m"),  # ambiguous (million? metre?) -> not multiplied
    ("", None, None), ("n/a", None, None), ("abc", None, None),
])
def test_parse_number(raw, value, unit):
    v, u = parse_number(raw)
    assert (v if v is None else pytest.approx(v)) == value and u == unit


def test_unit_conversion_and_currency_needs_a_rate():
    assert convert(100, "sqm", "sqft", CFG)[0] == pytest.approx(1076.39104)
    assert convert(1, "TB", "GB", CFG)[0] == 1024
    assert convert(1, "kg", "sqft", CFG)[0] is None
    assert convert(10, "EUR", "USD", CFG) == (None, "no configured rate for EUR->USD")
    rated = IntegrationConfig(currency_rates={"EUR": 1.1})
    assert convert(10, "EUR", "USD", rated)[0] == pytest.approx(11.0)


def test_unit_hints_from_column_names():
    assert unit_hint_from_name("Price_euros") == "EUR"
    assert unit_hint_from_name("Area (sq. m.)") == "sqm"
    assert unit_hint_from_name("Weight_kg") == "kg"
    assert unit_hint_from_name("Price") is None


def test_unconvertible_currency_keeps_values_and_reports():
    vals, rep = normalise_column(["1339.69", "898"], "float", "USD", "Price_euros", "price", CFG)
    assert vals == [1339.69, 898.0]
    assert rep.unit_to == "EUR" and any("no configured rate" in n for n in rep.notes)


def test_category_normalisation():
    assert normalise_category(" Rohini  Sector-25 ") == "rohini sector 25"
    assert normalise_category("Semi_Furnished") == "semi furnished"
    assert normalise_category("NA") is None


# --- union --------------------------------------------------------------------------

def test_union_maps_schema_converts_units_and_adds_provenance():
    t, reports = union_sources(SPEC, sources(), CFG)
    assert t.height == 6
    assert t.columns[:6] == ["price", "area_sqft", "bedrooms", "locality", "latitude", "longitude"]
    assert "x_per_sqft" in t.columns  # unmapped column kept for the 'all attributes' baseline
    assert t["price"].to_list()[:2] == [6_500_000, 12_000_000]
    assert t["area_sqft"][1] == pytest.approx(1076.39104)  # 100 sqm
    assert t["_row_ref"].to_list()[4] == "b:0" and t["_method"].to_list()[4] == "api"
    assert t["x_per_sqft"].dtype == pl.Float64 and t["x_per_sqft"][4] is None


# --- dedupe -------------------------------------------------------------------------

def test_dedupe_exact_and_near_duplicates_with_trust_ordering():
    t, _ = union_sources(SPEC, sources(), CFG)
    cols = [SPEC.target.name, *SPEC.feature_names()]
    out, rep = dedupe(t, cols, "price", CFG)
    assert rep.exact_removed == 1          # the two identical Dwarka rows in source a
    assert rep.near_removed == 1           # Rohini re-posted in b at another price
    assert out.height == 4
    assert "a:0" in out["_row_ref"].to_list() and "b:0" not in out["_row_ref"].to_list()  # a more trusted
    assert rep.target_conflicts and rep.target_conflicts[0]["kept"] == "a:0"


def test_distinct_rows_are_not_merged():
    t, _ = union_sources(SPEC, sources(), CFG)
    out, _ = dedupe(t, [SPEC.target.name, *SPEC.feature_names()], "price", CFG)
    assert {"lajpat nagar", "saket"} <= set(out["locality"].to_list())


# --- enrichment ---------------------------------------------------------------------

def test_key_join_with_aliases_reports_unmatched_and_keeps_order():
    t = pl.DataFrame({"locality": ["rohini sector 25", "dwarka sec 7", "unknown place"], "i": [0, 1, 2]})
    ctx = pl.DataFrame({"colony": ["Rohini Sector 25", "Dwarka Sector 7"], "cat": ["E", "D"]})
    out, rep = key_join(t, ctx, source_id="circle", table_key="locality", context_key="colony",
                        columns={"cat": "circle_rate_category"}, aliases={"Dwarka Sec 7": "Dwarka Sector 7"})
    assert out["i"].to_list() == [0, 1, 2]
    assert out["circle_rate_category"].to_list() == ["E", "D", None]  # context values kept as published
    assert rep.rows_matched == 2 and rep.unmatched_keys == ["unknown place"]


def test_haversine_and_nearest_distance():
    assert haversine_km(0, 0, 1, 0) == pytest.approx(111.19, abs=0.05)
    t = pl.DataFrame({"latitude": [28.70, None], "longitude": [77.10, 77.0]})
    pts = pl.DataFrame({"lat": [28.70, 30.0], "lon": [77.11, 77.0]})
    out, rep = nearest_distance(t, pts, source_id="osm", feature="distance_to_metro_km")
    assert out["distance_to_metro_km"][0] == pytest.approx(0.976, abs=0.01)
    assert out["distance_to_metro_km"][1] is None and rep.rows_matched == 1


# --- imputation -------------------------------------------------------------------------

def test_imputation_plan_per_column():
    t = pl.DataFrame({
        "price": [1.0, None, 3.0, 4.0, 5.0],
        "area_sqft": [100.0, None, 300.0, 400.0, 500.0],
        "bedrooms": [None, None, None, 2, 3],
        "locality": ["a", None, "b", "b", "a"],
        "_source_id": ["s"] * 5,
    })
    spec = SPEC.model_copy(update={"features": [F("area_sqft"), F("bedrooms", "int"), F("locality", "category")]})
    plan = {p.column: p for p in plan_imputation(t, spec, CFG, group_by="locality")}
    assert plan["price"].strategy == "drop_rows"
    assert plan["area_sqft"].strategy == "group_median"
    assert plan["bedrooms"].strategy == "leave_null"  # 60% missing
    assert plan["locality"].strategy == "missing_category"
    assert "_source_id" not in plan


def test_imputation_fits_on_training_split_only():
    spec = SPEC.model_copy(update={"features": [F("area_sqft")]})
    train = pl.DataFrame({"price": [1.0, 2.0, 3.0], "area_sqft": [100.0, 200.0, None]})
    test = pl.DataFrame({"price": [4.0], "area_sqft": [None]})
    plan = plan_imputation(train, spec, CFG)
    filled = apply_plan(test, plan, CFG, fit_on=train)
    assert filled["area_sqft"][0] == 150.0  # train median, not influenced by test
    assert filled["area_sqft__imputed"][0] is True


# --- end to end -------------------------------------------------------------------------

def test_integrate_end_to_end_writes_parquet_and_report(tmp_path):
    circle = pl.DataFrame({"colony": ["Rohini Sector 25", "Lajpat Nagar", "Saket"], "cat": ["E", "C", "C"]})
    stations = pl.DataFrame({"lat": [28.70, 28.57], "lon": [77.11, 77.24]})
    res = integrate(SPEC, sources(), [
        ContextJoin("delhi_circle_rates", "key_join", circle,
                    {"table_key": "locality", "context_key": "colony", "columns": {"cat": "circle_rate_category"}}),
        ContextJoin("osm_transit_stations", "nearest_point", stations, {"feature": "distance_to_metro_km"}),
    ], S, impute_group_by="locality")

    t, rep = res.table, res.report
    assert t.height == 4 and rep.rows_after_union == 6
    assert t.columns[-5:] == ["_row_ref", "_source_id", "_method", "_source_trust", "_retrieved_at"]
    assert rep.lineage["price"] == ["a.Price", "b.cost"]
    assert rep.lineage["distance_to_metro_km"] == ["derived:osm_transit_stations:nearest_point"]
    assert any("not matched" in w for w in rep.warnings)  # 'dwarka sec 7' has no circle rate
    assert rep.unit_conflicts == []

    paths = res.write(tmp_path)
    back = pl.read_parquet(paths["table"])
    assert back.equals(t)
    js = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert js["dedupe"]["near_removed"] == 1 and js["imputation_plan"]
