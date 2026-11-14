"""Stage 2 tests. Fixture CSVs are tiny hand-written test files (clearly synthetic,
used only to exercise the code), plus stubbed HTTP and LLM backends."""

import json
from datetime import datetime, timezone

import pytest

from dsagent.config import Settings
from dsagent.llm import Budget, LLMCallCache, LLMClient
from dsagent.llm.client import BackendResult
from dsagent.schemas.source import SourceDescriptor
from dsagent.schemas.spec import DatasetSpec, Feature
from dsagent.stages.discovery import discover_sources
from dsagent.stages.discovery.catalogs import (
    CuratedContextCatalog, HuggingFaceCatalog, LocalManifestCatalog, search_queries,
)
from dsagent.stages.discovery.matching import ColumnMatch, ColumnMatches, match_by_name, match_columns
from dsagent.trust.recency import recency_weight
from dsagent.trust.source_trust import SourceNotPermitted, licence_is_open, trust_components, trust_score

NOW = datetime(2026, 11, 14, tzinfo=timezone.utc)
S = Settings(_env_file=None)


def F(name, dtype="float", required=True):
    return Feature(name=name, dtype=dtype, description=name, required=required)


def delhi_spec(min_rows=3):
    return DatasetSpec(
        problem_text="I want to predict house prices in Delhi",
        task_type="regression",
        target=F("price"),
        features=[
            F("locality", "category"), F("built_up_area_sqft"), F("bedrooms", "int"),
            F("bathrooms", "int"), F("furnishing", "category", required=False),
            F("distance_to_metro_km", required=False),
        ],
        region="Delhi",
        min_rows=min_rows,
    )


FIXTURE_CSV = """Area,BHK,Bathroom,Furnishing,Locality,Price,Per_Sqft
800,2,2,Semi-Furnished,Rohini Sector 25,6500000,8125
1100,3,2,Furnished,Lajpat Nagar,15000000,13636
650,2,1,,Dwarka Sector 7,5200000,8000
"""


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "delhi.csv").write_text(FIXTURE_CSV, encoding="utf-8")
    manifest = [
        {"source_id": "local_delhi", "title": "Delhi listings (fixture)", "path": "raw/delhi.csv",
         "role": "entity", "authority_class": "community", "licence": "CC0-1.0",
         "last_updated": "2024-11-14T00:00:00Z", "region": "Delhi"},
        {"source_id": "local_mumbai", "title": "Mumbai listings (fixture)", "path": "raw/delhi.csv",
         "role": "entity", "authority_class": "community", "licence": "CC0-1.0", "region": "Mumbai"},
        {"source_id": "local_missing", "title": "missing", "path": "raw/nope.csv", "role": "entity"},
    ]
    (tmp_path / "sources.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


class FakeLLM:
    """Backend returning a fixed ColumnMatches answer."""

    def __init__(self, matches):
        self.matches = matches
        self.calls = 0

    def call(self, *, model, system, user, schema, max_tokens):
        self.calls += 1
        return BackendResult(ColumnMatches(matches=self.matches), "", model, 200, 50, "stop", 0.0001)


def llm_with(matches):
    backend = FakeLLM(matches)
    return LLMClient(Budget(100_000, 1, 10), LLMCallCache(":memory:"), backend, S), backend


# --- trust ----------------------------------------------------------------------------

def test_recency_weight_halves_per_half_life():
    assert recency_weight(0, 2) == 1.0
    assert recency_weight(2, 2) == pytest.approx(0.5)
    assert recency_weight(100, 2) == 0.05  # floor


def test_licence_classification():
    assert licence_is_open("CC-BY-4.0", S) is True
    assert licence_is_open("ODbL-1.0", S) is True
    assert licence_is_open("other", S) is None
    assert licence_is_open(None, S) is None
    assert licence_is_open("Proprietary - all rights reserved", S) is False


def test_trust_formula_and_exclusions():
    src = SourceDescriptor(source_id="g", title="t", url="u", kind="api", role="context",
                           authority_class="official", licence="GODL",
                           last_updated=datetime(2026, 11, 14, tzinfo=timezone.utc))
    comps = trust_components(src, coverage=1.0, half_life_years=2, settings=S, now=NOW)
    assert comps == {"authority": 1.0, "freshness": 1.0, "coverage": 1.0, "structure": 1.0, "accessibility": 1.0}
    assert trust_score(comps, S) == 1.0
    with pytest.raises(SourceNotPermitted):
        trust_components(src.model_copy(update={"robots_allowed": False}), 1, 2, S, NOW)
    with pytest.raises(SourceNotPermitted):
        trust_components(src.model_copy(update={"licence": "proprietary"}), 1, 2, S, NOW)


# --- matching -------------------------------------------------------------------------

def test_name_matching_and_ambiguity():
    feats = ["price", "built_up_area_sqft", "bathrooms", "furnishing", "locality", "bedrooms"]
    cols = ["Area", "BHK", "Bathroom", "Furnishing", "Locality", "Price", "Per_Sqft"]
    m = match_by_name(feats, cols)
    assert m == {"price": "Price", "bathrooms": "Bathroom", "furnishing": "Furnishing",
                 "locality": "Locality", "built_up_area_sqft": "Area"}
    # generic 'Area' becomes ambiguous once two area features exist
    m2 = match_by_name(feats + ["plot_area_sqft"], cols)
    assert "built_up_area_sqft" not in m2 and "plot_area_sqft" not in m2


def test_llm_fills_gaps_and_cannot_invent_columns():
    spec = delhi_spec()
    rows = [{"BHK": "2", "Area": "800"}]
    llm, backend = llm_with([
        ColumnMatch(feature="bedrooms", column="BHK", reason="BHK = bedrooms"),
        ColumnMatch(feature="distance_to_metro_km", column="MetroDist", reason="invented"),
    ])
    fmap, method, notes = match_columns(spec, ["BHK", "Area"], rows, llm)
    assert fmap["bedrooms"] == "BHK" and method["bedrooms"] == "llm"
    assert "distance_to_metro_km" not in fmap
    assert any("MetroDist" in n for n in notes)
    assert backend.calls == 1


# --- catalogs -------------------------------------------------------------------------

def test_local_manifest_reads_real_columns_and_reports_missing(workspace):
    cat = LocalManifestCatalog(workspace / "sources.json")
    found = cat.find(delhi_spec())
    assert [s.source_id for s in found] == ["local_delhi", "local_mumbai"]
    assert found[0].columns[:2] == ["Area", "BHK"] and found[0].est_rows == 3
    assert any("local_missing" in w for w in cat.warnings)


def test_curated_context_filtered_by_region():
    cat = CuratedContextCatalog(S.context_registry_path)
    ids = {s.source_id for s in cat.find(delhi_spec())}
    assert {"osm_transit_stations", "delhi_circle_rates", "nominatim_geocoding"} <= ids
    mumbai = delhi_spec().model_copy(update={"region": "Mumbai"})
    # region-agnostic sources apply anywhere; the Delhi-only official source does not
    assert {s.source_id for s in cat.find(mumbai)} == {"osm_transit_stations", "osm_amenities", "nominatim_geocoding"}


def test_search_queries_are_short_noun_phrases():
    assert search_queries(delhi_spec()) == ["house price", "house"]


class FakeHTTP:
    def __init__(self, routes):
        self.routes = routes
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        for prefix, payload in self.routes.items():
            if url.startswith(prefix):
                return type("R", (), {"status_code": 200, "json": lambda self, p=payload: p})()
        return type("R", (), {"status_code": 404, "json": lambda self: {}})()


def test_huggingface_catalog_describes_hits_and_counts_pages():
    http = FakeHTTP({
        "https://huggingface.co/api/datasets?search=house": [
            {"id": "someone/dublin-house-price", "lastModified": "2023-10-13T00:00:00.000Z",
             "cardData": {"license": "cc-by-4.0"}, "tags": [], "downloads": 13},
        ],
        "https://datasets-server.huggingface.co/splits": {"splits": [{"config": "default", "split": "train"}]},
        "https://datasets-server.huggingface.co/first-rows": {
            "features": [{"name": "price"}, {"name": "area"}],
            "rows": [{"row": {"price": 300000, "area": 90}}],
        },
    })
    budget = Budget(1, 1, pages_max=20)
    cat = HuggingFaceCatalog(budget, "ua", http_client=http)
    found = cat.find(delhi_spec())
    assert len(found) == 1 and found[0].licence == "cc-by-4.0" and found[0].columns == ["price", "area"]
    assert budget.pages_used == len(http.urls)


# --- end to end -----------------------------------------------------------------------

def test_discovery_end_to_end(workspace):
    llm, _ = llm_with([ColumnMatch(feature="bedrooms", column="BHK", reason="BHK")])
    catalogs = [LocalManifestCatalog(workspace / "sources.json"), CuratedContextCatalog(S.context_registry_path)]
    res = discover_sources(delhi_spec(), catalogs, llm, S, now=NOW)

    assert res.selected[0] == "local_delhi"
    assert ("local_mumbai", "region mismatch (wanted Delhi)") in res.rejected
    delhi = res.selected_candidates()[0]
    assert delhi.feature_map["bedrooms"] == "BHK"
    assert "Per_Sqft" not in delhi.feature_map.values()

    by_f = {it.feature: it for it in res.feasibility}
    assert by_f["price"].status == "found"
    assert by_f["furnishing"].status == "partial"  # 2 of 3 rows non-empty
    assert by_f["distance_to_metro_km"].status == "derivable"
    assert by_f["distance_to_metro_km"].coverage is None
    # metro needs coordinates or locality; locality is present, so no helper needed
    assert "osm_transit_stations" in res.selected
    assert res.deliverable, res.blocking
    assert "| price | found |" in res.feasibility_markdown()


def test_off_topic_source_with_only_a_price_column_is_rejected(tmp_path):
    (tmp_path / "oil.csv").write_text("Date,Price\n2024-01-01,80\n2024-01-02,81\n", encoding="utf-8")
    (tmp_path / "m.json").write_text(json.dumps([
        {"source_id": "crude_oil", "title": "Crude oil prices Delhi", "path": "oil.csv",
         "role": "entity", "licence": "CC0", "region": "Delhi"}]), encoding="utf-8")
    res = discover_sources(delhi_spec(), [LocalManifestCatalog(tmp_path / "m.json")], None, S, now=NOW)
    assert res.selected == []
    assert res.rejected[0][0] == "crude_oil" and "off-topic" in res.rejected[0][1]


def test_llm_column_name_whitespace_is_tolerated():
    llm, _ = llm_with([ColumnMatch(feature="bedrooms", column=" bhk ", reason="x")])
    fmap, _, _ = match_columns(delhi_spec(), ["BHK"], [{"BHK": "2"}], llm)
    assert fmap["bedrooms"] == "BHK"


def test_region_matching_uses_primary_place():
    from dsagent.stages.discovery import _region_ok

    hf = SourceDescriptor(source_id="hf:x/delhi_house_prices", title="x/delhi_house_prices",
                          url="u", kind="open_repo", role="entity")
    assert _region_ok(hf, "Delhi, India")
    assert not _region_ok(hf, "Mumbai, India")
    local = hf.model_copy(update={"region": "Delhi"})
    assert _region_ok(local, "Delhi, India")
    assert _region_ok(hf, None)


def test_missing_target_blocks_delivery(workspace):
    spec = delhi_spec().model_copy(update={"target": F("monthly_rent")})
    res = discover_sources(spec, [LocalManifestCatalog(workspace / "sources.json")], None, S, now=NOW)
    assert not res.deliverable
    assert any("monthly_rent" in b for b in res.blocking)


def test_too_few_rows_blocks(workspace):
    res = discover_sources(delhi_spec(min_rows=1000), [LocalManifestCatalog(workspace / "sources.json")],
                           None, S, now=NOW)
    assert any("rows" in b for b in res.blocking)
