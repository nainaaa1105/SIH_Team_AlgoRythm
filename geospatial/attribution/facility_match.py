"""Probabilistic facility attribution.

Upgrades M1's version (`app/enrichment/osm_facilities.py`), which picks
the single nearest facility within 5 km using a point distance. Three
things that version cannot express:

  1. A hotspot *inside* a refinery polygon is not "0 m from" the
     refinery in any meaningful probabilistic sense — it is almost
     certainly that refinery. Containment deserves its own treatment.
  2. Nearest is not the same as most likely. A hotspot equidistant from
     a flare stack and a warehouse is not a 50/50 call, because flare
     stacks burn by design and warehouses do not.
  3. Responders need the runners-up. "Refinery X at 72%, tank farm Y at
     21%" is actionable in a way a single answer is not.

So this scores every candidate in range and returns a normalised
distribution. M1's simpler path stays as the fast inline fallback for
when M3's worker isn't running.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from geospatial.geometry import haversine_m

# How plausible each facility type is as the origin of a thermal anomaly,
# independent of distance. Flares and refineries produce heat as part of
# normal operation; a generic industrial footprint much less reliably so.
FACILITY_TYPE_PRIORS: Dict[str, float] = {
    "flare": 1.00,
    "refinery": 0.95,
    "oil_well": 0.85,
    "steel": 0.85,
    "power": 0.80,
    "mine": 0.75,
    "LNG": 0.70,
    "industrial": 0.50,
}
DEFAULT_TYPE_PRIOR = 0.40

# Distance decay constant, matching the exp(-d/500) form M1 already uses
# so both implementations agree on the shape of the falloff.
DISTANCE_DECAY_M = 500.0

DEFAULT_SEARCH_RADIUS_M = 5000.0

# Score given to a containment hit. Well above any distance-decayed
# score so an enclosing polygon always outranks a closer point source
# outside it.
CONTAINMENT_SCORE = 1.0


@dataclass
class AttributionCandidate:
    facility_id: int
    name: Optional[str]
    facility_type: Optional[str]
    distance_m: float
    inside_polygon: bool
    score: float
    probability: float = 0.0
    rank: int = 0


def type_prior(facility_type: Optional[str]) -> float:
    if not facility_type:
        return DEFAULT_TYPE_PRIOR
    return FACILITY_TYPE_PRIORS.get(facility_type, DEFAULT_TYPE_PRIOR)


def candidate_score(
    distance_m: float, facility_type: Optional[str], inside_polygon: bool = False
) -> float:
    """Unnormalised plausibility that this facility is the source."""
    prior = type_prior(facility_type)
    if inside_polygon:
        return CONTAINMENT_SCORE * prior
    return prior * math.exp(-max(distance_m, 0.0) / DISTANCE_DECAY_M)


def normalise(candidates: List[AttributionCandidate]) -> List[AttributionCandidate]:
    """Turn scores into probabilities and rank them.

    Ranking is by score, with distance as a deterministic tie-break so
    two equally-scored facilities don't swap places between runs.
    """
    total = sum(c.score for c in candidates)
    ordered = sorted(candidates, key=lambda c: (-c.score, c.distance_m, c.facility_id))

    for i, candidate in enumerate(ordered, start=1):
        candidate.probability = (candidate.score / total) if total > 0 else 0.0
        candidate.rank = i
    return ordered


def attribute(
    lon: float,
    lat: float,
    facilities: Sequence[Dict],
    search_radius_m: float = DEFAULT_SEARCH_RADIUS_M,
) -> List[AttributionCandidate]:
    """Rank nearby facilities by how likely each is to be the source.

    Each facility dict needs: id, lon, lat, optionally name,
    facility_type, and `inside_polygon` (set by the caller from a PostGIS
    ST_Contains test, since a point-based check can't do containment).
    """
    candidates: List[AttributionCandidate] = []

    for facility in facilities:
        # Prefer a distance the caller already measured. The DB query
        # computes ST_Distance against the facility's full geometry, which
        # for a polygon is the distance to its *boundary* (zero when the
        # hotspot is on or inside it). Recomputing here from the polygon's
        # centroid instead would report a sprawling refinery as ~1 km away
        # when the fire is actually at its fence line — under-scoring the
        # correct facility by an order of magnitude, and poisoning
        # `fingerprints.facility_distance_m`, which M2 snapshots as a model
        # feature and thresholds at 500 m in its label rules.
        distance_m = facility.get("distance_m")
        if distance_m is None:
            f_lon, f_lat = facility.get("lon"), facility.get("lat")
            if f_lon is None or f_lat is None:
                continue
            distance_m = haversine_m(lon, lat, f_lon, f_lat)
        distance_m = float(distance_m)

        inside = bool(facility.get("inside_polygon", False))

        # Containment wins regardless of centroid distance: a hotspot can
        # sit inside a sprawling plant whose centroid is over a km away.
        if not inside and distance_m > search_radius_m:
            continue

        candidates.append(AttributionCandidate(
            facility_id=facility["id"],
            name=facility.get("name"),
            facility_type=facility.get("facility_type"),
            distance_m=distance_m,
            inside_polygon=inside,
            score=candidate_score(distance_m, facility.get("facility_type"), inside),
        ))

    return normalise(candidates)


def top_candidate(candidates: Sequence[AttributionCandidate]) -> Optional[AttributionCandidate]:
    return candidates[0] if candidates else None


def attribution_is_confident(
    candidates: Sequence[AttributionCandidate], margin: float = 0.25
) -> bool:
    """Is the leading candidate a clear winner?

    Used to decide whether to present a single attributed facility or to
    show the alternatives. Mirrors the ambiguity handling in M2's label
    rules: a near-tie is information, not noise to be hidden.
    """
    if not candidates:
        return False
    if len(candidates) == 1:
        return True
    return (candidates[0].probability - candidates[1].probability) >= margin


# Containment + distance in one pass, at DB scale. `inside` short-circuits
# the radius filter so a hotspot inside a large plant is always a candidate.
CANDIDATE_FACILITIES_SQL = """
SELECT f.id,
       f.name,
       f.facility_type,
       ST_X(ST_Centroid(f.geom)) AS lon,
       ST_Y(ST_Centroid(f.geom)) AS lat,
       ST_Distance(f.geom::geography, c.centroid::geography) AS distance_m,
       ST_Contains(f.geom, c.centroid) AS inside_polygon
FROM facilities f, clusters c
WHERE c.id = :cluster_id
  AND (
        ST_DWithin(f.geom::geography, c.centroid::geography, :search_radius_m)
        OR ST_Contains(f.geom, c.centroid)
      )
ORDER BY inside_polygon DESC, distance_m ASC
LIMIT 25;
"""
