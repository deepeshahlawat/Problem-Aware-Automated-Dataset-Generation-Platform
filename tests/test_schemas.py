import pytest
from pydantic import ValidationError

from dsagent.schemas import DatasetSpec, Feature, SourceCandidate, TemporalPolicy, ValueRange


def _feature(name, dtype="float", **kw):
    return Feature(name=name, dtype=dtype, description=f"{name} description", **kw)


def delhi_spec(**overrides):
    data = dict(
        problem_text="I want to predict house prices in Delhi",
        task_type="regression",
        target=_feature("price", unit="INR"),
        features=[
            _feature("locality", "category"),
            _feature("area_sqft", unit="sq ft", expected_range=ValueRange(min=100, max=20000)),
            _feature("bhk", "int"),
            _feature("distance_to_metro_km", required=False),
        ],
        region="Delhi",
    )
    data.update(overrides)
    return DatasetSpec(**data)


def test_valid_spec_roundtrips_through_json():
    spec = delhi_spec()
    again = DatasetSpec.model_validate_json(spec.model_dump_json())
    assert again == spec
    assert spec.feature_names() == ["locality", "area_sqft", "bhk", "distance_to_metro_km"]


def test_target_cannot_also_be_a_feature():
    with pytest.raises(ValidationError, match="target"):
        delhi_spec(features=[_feature("price"), _feature("bhk", "int")])


def test_duplicate_feature_names_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        delhi_spec(features=[_feature("bhk", "int"), _feature("bhk", "int")])


@pytest.mark.parametrize("bad", ["Area", "area sqft", "", "price-per-sqft"])
def test_feature_names_must_be_snake_case(bad):
    with pytest.raises(ValidationError):
        _feature(bad)


def test_value_range_ordered():
    with pytest.raises(ValidationError):
        ValueRange(min=10, max=1)


def test_window_mode_needs_both_years():
    with pytest.raises(ValidationError):
        TemporalPolicy(mode="window", window_start=2000)
    assert TemporalPolicy(mode="window", window_start=2000, window_end=2010).window_end == 2010


def test_spec_json_schema_has_no_tuple_constructs():
    # Structured outputs handle tuples (prefixItems) poorly; ranges use ValueRange instead.
    assert "prefixItems" not in DatasetSpec.model_json_schema().__repr__()


def test_source_candidate_requires_role_and_bounded_trust():
    base = dict(
        source_id="osm_metro",
        url="https://overpass-api.de/api/interpreter",
        kind="api",
        covers_features=["distance_to_metro_km"],
        robots_allowed=True,
        trust_score=0.8,
    )
    with pytest.raises(ValidationError):
        SourceCandidate(**base)  # role missing
    s = SourceCandidate(**base, role="context", join_key="locality")
    assert s.role == "context"
    with pytest.raises(ValidationError):
        SourceCandidate(**{**base, "trust_score": 1.5}, role="context")
