"""Synthetic table with planted structure, for testing the relevance engine only.
Not data: generated from a known formula so the right answers are known."""
import numpy as np
import polars as pl

from dsagent.schemas import DatasetSpec, Feature


def F(name, dtype="float", unit=None):
    return Feature(name=name, dtype=dtype, unit=unit, description=name)


SPEC = DatasetSpec(
    problem_text="I want to predict house prices in Delhi", task_type="regression",
    target=F("price", unit="INR"),
    features=[F("area_sqft", unit="sqft"), F("bedrooms", "int"), F("locality", "category"),
              F("distance_to_metro_km", unit="km"), F("amenities_count", "int")],
    region="Delhi", min_rows=50,
)
LINEAGE = {
    "price": ["a.Price"], "area_sqft": ["a.Area"], "bedrooms": ["a.BHK"], "locality": ["a.Locality"],
    "distance_to_metro_km": ["derived:osm_transit_stations:nearest_point"],
    "amenities_count": ["derived:osm_amenities:key_join"],
}


def make(n=600, seed=0):
    rng = np.random.default_rng(seed)
    loc = rng.integers(0, 10, n)
    rate = np.linspace(4000, 20000, 10)[loc]
    area = rng.uniform(400, 3000, n)
    beds = np.clip(np.round(area / 700 + rng.normal(0, 0.5, n)), 1, 5).astype(int)
    dist = rng.uniform(0.1, 10, n)
    amen = rng.integers(0, 30, n)  # no effect on price
    price = area * rate * (1 - 0.04 * dist) * rng.lognormal(0, 0.08, n)
    return pl.DataFrame({
        "price": price, "area_sqft": area, "bedrooms": beds,
        "locality": [f"loc {i}" for i in loc],
        "distance_to_metro_km": dist, "amenities_count": amen,
        "x_noise": rng.normal(0, 1, n),
        "x_listing_id": [f"MB{100000 + i}" for i in range(n)],
        "x_site_name": ["magicbricks"] * n,
        "x_per_sqft": price / area,
        "_row_ref": [f"a:{i}" for i in range(n)], "_source_id": ["a"] * n,
        "_source_trust": [0.7] * n, "_sample_weight": [1.0] * n,
    })
