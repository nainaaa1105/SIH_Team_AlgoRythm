import math

from shapely.geometry import Point, Polygon

from app.enrichment.osm_facilities import (
    build_overpass_query,
    facility_proximity_score,
    find_nearest_facility,
    haversine_m,
    normalize_elements,
)


def test_haversine_m_zero_distance_for_identical_points():
    assert haversine_m(77.1, 28.7, 77.1, 28.7) == 0.0


def test_haversine_m_known_distance_delhi_to_mumbai_roughly():
    # Delhi (77.1025, 28.7041) to Mumbai (72.8777, 19.0760) ~ 1150-1160 km great-circle
    d = haversine_m(77.1025, 28.7041, 72.8777, 19.0760)
    assert 1_100_000 < d < 1_200_000


def test_facility_proximity_score_decays_with_distance():
    at_zero = facility_proximity_score(0.0)
    at_decay_distance = facility_proximity_score(500.0)
    far = facility_proximity_score(5000.0)

    assert at_zero == 1.0
    assert math.isclose(at_decay_distance, math.exp(-1), rel_tol=1e-6)
    assert far < at_decay_distance < at_zero


def test_find_nearest_facility_picks_closest_within_radius():
    facilities = [
        {"name": "far", "lon": 80.0, "lat": 20.0, "prior_weight": 1.0},
        {"name": "near", "lon": 77.1005, "lat": 28.7005, "prior_weight": 1.0},
    ]
    result = find_nearest_facility(77.1000, 28.7000, facilities, max_distance_m=5000)
    assert result is not None
    assert result["name"] == "near"
    assert result["distance_m"] < 200


def test_find_nearest_facility_respects_max_distance():
    facilities = [{"name": "far", "lon": 90.0, "lat": 10.0, "prior_weight": 1.0}]
    result = find_nearest_facility(77.1, 28.7, facilities, max_distance_m=1000)
    assert result is None


def test_build_overpass_query_includes_all_tags_and_correct_bbox_order():
    query = build_overpass_query((68.0, 6.5, 97.5, 37.5))
    assert "6.5,68.0,37.5,97.5" in query  # south,west,north,east
    assert '["landuse"="industrial"]' in query
    assert '["power"="plant"]' in query


def test_build_overpass_query_requests_full_geometry_not_just_centroid():
    """Regression guard: `out center tags` discards a way's vertex list,
    so ST_Contains in the attribution query always evaluates false —
    `out geom` is what makes a real Polygon possible instead of a Point."""
    query = build_overpass_query((68.0, 6.5, 97.5, 37.5))
    assert "out geom;" in query
    assert "out center tags;" not in query


# --- normalize_elements: real polygons, not just centroids ----------------


def test_normalize_elements_builds_a_point_for_a_node():
    elements = [{"type": "node", "id": 1, "lat": 21.82, "lon": 83.99, "tags": {"landuse": "industrial"}}]
    facilities = normalize_elements(elements)
    assert len(facilities) == 1
    assert isinstance(facilities[0]["geom"], Point)
    assert facilities[0]["lon"] == 83.99
    assert facilities[0]["lat"] == 21.82


def test_normalize_elements_builds_a_real_polygon_for_a_closed_way():
    square = [
        {"lat": 21.0, "lon": 84.0}, {"lat": 21.0, "lon": 84.01},
        {"lat": 21.01, "lon": 84.01}, {"lat": 21.01, "lon": 84.0},
        {"lat": 21.0, "lon": 84.0},  # closed: first == last
    ]
    elements = [{"type": "way", "id": 2, "geometry": square, "tags": {"landuse": "industrial"}}]
    facilities = normalize_elements(elements)
    assert len(facilities) == 1
    geom = facilities[0]["geom"]
    assert isinstance(geom, Polygon)
    assert geom.area > 0
    # lon/lat kept in sync with the polygon's own centroid, not the first vertex
    assert facilities[0]["lon"] == geom.centroid.x
    assert facilities[0]["lat"] == geom.centroid.y


def test_normalize_elements_falls_back_to_a_point_for_an_open_way():
    """A pipeline or similar linear feature has no interior — must not be
    forced into a bogus Polygon, and must not be dropped either."""
    line = [{"lat": 21.0, "lon": 84.0}, {"lat": 21.0, "lon": 84.01}, {"lat": 21.02, "lon": 84.02}]
    elements = [{"type": "way", "id": 3, "geometry": line, "tags": {"man_made": "works"}}]
    facilities = normalize_elements(elements)
    assert len(facilities) == 1
    assert isinstance(facilities[0]["geom"], Point)


def test_normalize_elements_skips_a_way_with_no_geometry():
    elements = [{"type": "way", "id": 4, "tags": {"landuse": "industrial"}}]
    assert normalize_elements(elements) == []


def test_normalize_elements_sets_facility_type_from_tags():
    elements = [{"type": "node", "id": 5, "lat": 21.0, "lon": 84.0, "tags": {"power": "plant"}}]
    facilities = normalize_elements(elements)
    assert facilities[0]["facility_type"] == "power"


# --- nearest_named_industrial_feature / named_feature_is_same_site --------
# Regression coverage for a real gap: a fire's matched facility (99%
# confidence, 148.7m away — geospatial's own attribution engine got the
# proximity right) had no `name` at all, because it's a bare
# landuse=industrial polygon from a Maxar bulk import. The actual named
# company the map's own OSM tile label rendered ("SPS Steel and Power
# Limited") was never captured by the ingestion's tag list at all — a
# genuinely separate OSM element. named_feature_is_same_site() decides
# whether a live lookup's result is close enough to be that same site.

import app.enrichment.osm_facilities as osm_facilities
from app.enrichment.osm_facilities import (
    NAMED_FACILITY_MAX_EXCESS_M,
    named_feature_is_same_site,
    nearest_named_industrial_feature,
)


def test_named_feature_is_same_site_accepts_a_close_match():
    candidate = {"name": "SPS Steel and Power Limited", "distance_m": 150.0}
    assert named_feature_is_same_site(candidate, matched_distance_m=148.7) is True


def test_named_feature_is_same_site_rejects_a_distant_unrelated_feature():
    candidate = {"name": "Some Other Factory", "distance_m": 5000.0}
    assert named_feature_is_same_site(candidate, matched_distance_m=148.7) is False


def test_named_feature_is_same_site_boundary_is_inclusive():
    matched = 148.7
    at_boundary = {"name": "X", "distance_m": matched + NAMED_FACILITY_MAX_EXCESS_M}
    just_beyond = {"name": "X", "distance_m": matched + NAMED_FACILITY_MAX_EXCESS_M + 0.1}
    assert named_feature_is_same_site(at_boundary, matched) is True
    assert named_feature_is_same_site(just_beyond, matched) is False


def test_named_feature_is_same_site_fails_closed_on_missing_data():
    assert named_feature_is_same_site(None, 148.7) is False
    assert named_feature_is_same_site({"name": "X", "distance_m": None}, 148.7) is False
    assert named_feature_is_same_site({"name": "X", "distance_m": 10.0}, None) is False


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_nearest_named_industrial_feature_picks_the_closest_named_element(monkeypatch):
    payload = {
        "elements": [
            {"tags": {"name": "Far Factory"}, "lat": 20.0, "lon": 85.0},
            {"tags": {"name": "SPS Steel and Power Limited"}, "lat": 21.8219, "lon": 83.9957},
            {"tags": {}, "lat": 21.8212, "lon": 83.9968},  # unnamed — must be skipped
        ]
    }
    monkeypatch.setattr(osm_facilities.requests, "post", lambda *a, **k: _FakeResponse(payload))

    result = nearest_named_industrial_feature(83.9968, 21.8212)
    assert result is not None
    assert result["name"] == "SPS Steel and Power Limited"


def test_nearest_named_industrial_feature_returns_none_when_nothing_named_is_close(monkeypatch):
    monkeypatch.setattr(osm_facilities.requests, "post", lambda *a, **k: _FakeResponse({"elements": []}))

    assert nearest_named_industrial_feature(83.9968, 21.8212) is None


def test_nearest_named_industrial_feature_fails_closed_on_network_error(monkeypatch):
    def _raise(*a, **k):
        raise ConnectionError("no network")

    monkeypatch.setattr(osm_facilities.requests, "post", _raise)
    assert nearest_named_industrial_feature(83.9968, 21.8212) is None


def test_nearest_named_industrial_feature_reads_way_center_not_just_node_latlon(monkeypatch):
    """A `way` (e.g. the actual factory polygon) reports its
    representative point under `center`, not top-level lat/lon — the
    lookup must handle both element shapes, not just nodes."""
    payload = {"elements": [{"tags": {"name": "Way Factory"}, "center": {"lat": 21.8219, "lon": 83.9957}}]}
    monkeypatch.setattr(osm_facilities.requests, "post", lambda *a, **k: _FakeResponse(payload))

    result = nearest_named_industrial_feature(83.9968, 21.8212)
    assert result is not None
    assert result["name"] == "Way Factory"
