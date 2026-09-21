import pytest

from geospatial.attribution.facility_match import (
    DEFAULT_TYPE_PRIOR,
    FACILITY_TYPE_PRIORS,
    attribute,
    attribution_is_confident,
    candidate_score,
    top_candidate,
    type_prior,
)
from geospatial.attribution.landcover_zonal import (
    buffer_radius_m,
    summarise_modis,
    summarise_worldcover,
)

SOURCE = (77.0, 28.0)


def _facility(fid, lon, lat, ftype="industrial", inside=False, name=None):
    return {
        "id": fid, "lon": lon, "lat": lat, "facility_type": ftype,
        "inside_polygon": inside, "name": name or f"Facility {fid}",
    }


# --- scoring ------------------------------------------------------------

def test_type_prior_falls_back_for_unknown_types():
    assert type_prior("refinery") == FACILITY_TYPE_PRIORS["refinery"]
    assert type_prior("something_else") == DEFAULT_TYPE_PRIOR
    assert type_prior(None) == DEFAULT_TYPE_PRIOR


def test_score_decays_with_distance():
    near = candidate_score(100, "refinery")
    far = candidate_score(3000, "refinery")
    assert near > far > 0


def test_flare_outscores_a_generic_warehouse_at_equal_distance():
    """Flare stacks burn by design; generic industrial footprints do not.
    An equidistant pair is not a 50/50 call."""
    assert candidate_score(300, "flare") > candidate_score(300, "industrial")


def test_containment_outranks_a_closer_point_source_outside():
    """A hotspot inside a sprawling plant belongs to that plant even if
    another facility's centroid is nearer."""
    inside_big_plant = candidate_score(1200, "refinery", inside_polygon=True)
    just_outside_small = candidate_score(50, "industrial", inside_polygon=False)
    assert inside_big_plant > just_outside_small


# --- attribution --------------------------------------------------------

def test_probabilities_sum_to_one():
    candidates = attribute(*SOURCE, [
        _facility(1, 77.001, 28.0, "refinery"),
        _facility(2, 77.004, 28.0, "industrial"),
        _facility(3, 77.010, 28.0, "power"),
    ])
    assert sum(c.probability for c in candidates) == pytest.approx(1.0)


def test_candidates_are_ranked_from_one():
    candidates = attribute(*SOURCE, [
        _facility(1, 77.010, 28.0), _facility(2, 77.001, 28.0),
    ])
    assert [c.rank for c in candidates] == [1, 2]


def test_nearest_plausible_facility_wins():
    candidates = attribute(*SOURCE, [
        _facility(1, 77.020, 28.0, "refinery"),
        _facility(2, 77.001, 28.0, "refinery"),
    ])
    assert top_candidate(candidates).facility_id == 2


def test_facilities_beyond_the_search_radius_are_excluded():
    candidates = attribute(*SOURCE, [_facility(1, 78.5, 28.0)], search_radius_m=5000)
    assert candidates == []


def test_a_contained_facility_is_kept_even_if_its_centroid_is_far():
    """Large plants have distant centroids; containment must survive the
    radius filter."""
    candidates = attribute(
        *SOURCE, [_facility(1, 77.2, 28.2, "refinery", inside=True)], search_radius_m=1000
    )
    assert len(candidates) == 1
    assert candidates[0].inside_polygon is True


def test_facilities_missing_coordinates_are_skipped():
    candidates = attribute(*SOURCE, [{"id": 1, "lon": None, "lat": None}])
    assert candidates == []


def test_a_supplied_polygon_distance_is_used_instead_of_centroid_distance():
    """Regression: the DB query measures ST_Distance to the facility's
    full geometry (zero at a polygon's fence line), but this used to be
    discarded and recomputed from the polygon's centroid — reporting a
    sprawling refinery as ~1.2 km away when the fire was at its edge.
    That under-scored the correct facility ~10x and fed a wrong
    facility_distance_m into M2's features and label thresholds."""
    row_from_sql = {
        "id": 1, "name": "Big Refinery", "facility_type": "refinery",
        "lon": 77.012, "lat": 28.0,     # ST_Centroid, ~1.2 km from the hotspot
        "distance_m": 0.0,               # ST_Distance to the polygon itself
        "inside_polygon": False,
    }
    candidate = attribute(*SOURCE, [row_from_sql])[0]
    assert candidate.distance_m == 0.0


def test_haversine_is_still_used_when_no_distance_is_supplied():
    """The pure/offline path (no DB) must keep working."""
    candidate = attribute(*SOURCE, [_facility(1, 77.012, 28.0)])[0]
    assert candidate.distance_m == pytest.approx(1178, rel=0.05)


def test_supplied_distance_drives_the_score_not_the_centroid():
    at_fence = attribute(*SOURCE, [{
        "id": 1, "facility_type": "refinery", "lon": 77.02, "lat": 28.0, "distance_m": 0.0,
    }])[0]
    far_away = attribute(*SOURCE, [{
        "id": 1, "facility_type": "refinery", "lon": 77.02, "lat": 28.0, "distance_m": 3000.0,
    }])[0]
    assert at_fence.score > far_away.score


def test_no_facilities_gives_no_candidates():
    assert attribute(*SOURCE, []) == []
    assert top_candidate([]) is None


def test_ranking_is_deterministic_for_tied_scores():
    tied = [_facility(2, 77.001, 28.0, "power"), _facility(1, 77.001, 28.0, "power")]
    first = [c.facility_id for c in attribute(*SOURCE, tied)]
    second = [c.facility_id for c in attribute(*SOURCE, list(reversed(tied)))]
    assert first == second


# --- confidence ---------------------------------------------------------

def test_a_clear_leader_is_confident():
    candidates = attribute(*SOURCE, [
        _facility(1, 77.0005, 28.0, "refinery"),
        _facility(2, 77.040, 28.0, "industrial"),
    ])
    assert attribution_is_confident(candidates)


def test_a_near_tie_is_not_confident():
    """Same as M2's label arbitration: a near-tie is information to
    surface, not noise to hide behind a single answer."""
    candidates = attribute(*SOURCE, [
        _facility(1, 77.001, 28.0, "power"),
        _facility(2, 77.0, 28.001, "power"),
    ])
    assert not attribution_is_confident(candidates)


def test_a_single_candidate_is_confident():
    candidates = attribute(*SOURCE, [_facility(1, 77.001, 28.0)])
    assert attribution_is_confident(candidates)


def test_no_candidates_is_not_confident():
    assert not attribution_is_confident([])


# --- land cover ---------------------------------------------------------

def test_buffer_scales_with_the_event_footprint():
    small = buffer_radius_m(0.5)
    large = buffer_radius_m(20.0)
    assert large > small


def test_buffer_is_clamped_to_sane_bounds():
    assert buffer_radius_m(0.0001) >= 500.0
    assert buffer_radius_m(10000.0) <= 20000.0


def test_buffer_falls_back_when_extent_is_unknown():
    assert buffer_radius_m(None) == 1000.0
    assert buffer_radius_m(-1.0) == 1000.0


def test_worldcover_histogram_becomes_fractions():
    summary = summarise_worldcover({10: 50, 40: 30, 50: 20})  # forest/cropland/urban
    assert summary["pct_forest"] == pytest.approx(0.5)
    assert summary["pct_cropland"] == pytest.approx(0.3)
    assert summary["pct_urban"] == pytest.approx(0.2)


def test_worldcover_handles_an_empty_histogram():
    summary = summarise_worldcover({})
    assert all(value == 0.0 for value in summary.values())


def test_modis_histogram_uses_the_igbp_scheme():
    """MCD12Q1 codes are a different scheme from WorldCover; 12 is
    cropland in IGBP but nothing in WorldCover."""
    summary = summarise_modis({1: 40, 12: 40, 13: 20})
    assert summary["pct_forest"] == pytest.approx(0.4)
    assert summary["pct_cropland"] == pytest.approx(0.4)
    assert summary["pct_urban"] == pytest.approx(0.2)


def test_the_two_schemes_disagree_on_the_same_code():
    """Guards against silently feeding a MODIS raster to the WorldCover
    summariser: code 10 is 'forest' in WorldCover, 'grassland' in IGBP."""
    assert summarise_worldcover({10: 10})["pct_forest"] == 1.0
    assert summarise_modis({10: 10})["pct_forest"] == 0.0
