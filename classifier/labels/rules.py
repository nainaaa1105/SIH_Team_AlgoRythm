"""Weak-supervision label rules.

Context for why this module exists at all, from the project research
notes: there is no ready-made Indian ground-truth dataset that says
"this FIRMS hotspot was a gas flare". The US NIFC wildfire-perimeter
approach from the reference paper doesn't transfer. So labels are
derived from independent, authoritative-ish sources — GGFR's flare
catalogue, FSI's forest-fire alerts, GEM/OSM industrial footprints,
known coalfield boundaries, and the well-documented Punjab/Haryana
stubble-burning season.

These are *weak* labels. Each rule returns a confidence alongside its
label, and `resolve_label` arbitrates when several fire at once. Rules
that fire on the same cluster with similar confidence are a signal that
the cluster is genuinely ambiguous and belongs in manual review, not in
the training set — `LabelVote.is_ambiguous` marks those.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

# --- Region definitions -------------------------------------------------
# (west, south, east, north)

# Punjab + Haryana, the stubble-burning belt.
AGRI_BURN_REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    "punjab_haryana": (73.8, 29.5, 77.6, 32.6),
}

# Major coalfields with well-documented persistent coal-seam fires and
# opencast mining thermal activity.
MINING_REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    "jharia": (86.1, 23.6, 86.6, 23.9),
    "korba": (82.5, 22.2, 82.9, 22.5),
    "singrauli": (82.5, 23.9, 82.9, 24.3),
}

# Stubble burning is sharply seasonal: rabi residue around April-May,
# kharif (paddy) residue around October-November.
AGRI_BURN_MONTHS = (4, 5, 10, 11)


@dataclass
class LabelVote:
    label: str
    confidence: float
    source: str
    reason: str


def _in_bbox(lon: float, lat: float, bbox: Tuple[float, float, float, float]) -> bool:
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


def region_containing(
    lon: float, lat: float, regions: Dict[str, Tuple[float, float, float, float]]
) -> Optional[str]:
    for name, bbox in regions.items():
        if _in_bbox(lon, lat, bbox):
            return name
    return None


def flare_rule(
    nearest_facility_type: Optional[str],
    facility_distance_m: Optional[float],
    persistence_days: Optional[float],
    max_distance_m: float = 500.0,
) -> Optional[LabelVote]:
    """A persistent hotspot sitting on a known flare site is a gas flare.

    GGFR flare sites are loaded into `facilities` with
    facility_type='flare' by M1's bulk_load_gem script, so this is a
    direct catalogue match rather than an inference.
    """
    if nearest_facility_type != "flare" or facility_distance_m is None:
        return None
    if facility_distance_m > max_distance_m:
        return None

    # Flares burn continuously; a one-off detection on a flare site is
    # more likely a coincidence or an actual accident at the facility.
    confident = persistence_days is not None and persistence_days >= 3
    return LabelVote(
        label="gas_flare",
        confidence=0.90 if confident else 0.60,
        source="GGFR",
        reason=f"Within {facility_distance_m:.0f}m of a catalogued GGFR flare site",
    )


def mining_rule(
    lon: float, lat: float,
    nearest_facility_type: Optional[str] = None,
    facility_distance_m: Optional[float] = None,
    persistence_days: Optional[float] = None,
    max_facility_distance_m: float = 500.0,
    min_persistence_days: float = 3.0,
) -> Optional[LabelVote]:
    """A real mapped quarry/mine nearby AND persistent, or a known
    coalfield boundary.

    Same real-vs-regional split as flare_rule/industrial_rule: OSM tags
    real quarries (landuse=quarry -> facility_type='mine') all over
    India, not just the three named coalfields below, and a located
    match is much stronger evidence than "somewhere inside this box".
    This used to be region-only, which had two real consequences: real
    mining activity anywhere outside Jharia/Korba/Singrauli never got a
    mining vote at all (falling through to whatever else fired — often
    a cropland-percentage heuristic, indistinguishable from a genuine
    agricultural burn), and a coalfield's own land cover being partly
    cropland gave the model contradictory training signal for the same
    "mining" label. Grounding the common case in the real quarry
    footprint instead of a land-cover percentage fixes both.

    The persistence gate was added after a live retrain: firing on
    proximity alone pulled in ~1,260 real industrial_fire and ~1,470
    real agricultural_burning test cases as false "mining" (0.249
    precision) — a quarry sitting near an unrelated fire doesn't make
    that fire mining activity. Real mining/coal-seam fires are
    persistent by nature (this module's own docstring: "well-documented
    persistent coal-seam fires"); a one-off detection near a quarry is
    much more likely a different, transient event passing through, same
    reasoning flare_rule already applies to a one-off flare-site hit.
    A non-persistent quarry-adjacent detection falls through to the
    regional coalfield prior instead of confidently claiming mining on
    proximity alone.

    Confidence is deliberately lower for the regional fallback — the
    coalfield boxes are tens of km across and routinely contain pithead
    power plants, flare stacks and other industry, so "somewhere inside
    this box" is real but much weaker evidence than a located match.
    """
    near_quarry = nearest_facility_type == "mine" and facility_distance_m is not None \
        and facility_distance_m <= max_facility_distance_m
    persistent = persistence_days is not None and persistence_days >= min_persistence_days

    if near_quarry and persistent:
        return LabelVote(
            label="mining",
            confidence=0.85,
            source="OSM_QUARRY",
            reason=f"Within {facility_distance_m:.0f}m of a mapped quarry/mine, persistent {persistence_days:.0f}d",
        )

    region = region_containing(lon, lat, MINING_REGIONS)
    if region is None:
        return None
    return LabelVote(
        label="mining",
        confidence=0.70,
        source="MINING_REGION",
        reason=f"Inside the {region} coalfield boundary",
    )


def agri_burn_rule(
    lon: float,
    lat: float,
    acq_datetime: Optional[datetime],
    pct_cropland: Optional[float],
    persistence_days: Optional[float],
    facility_distance_m: Optional[float] = None,
    min_cropland: float = 0.6,
    max_facility_distance_m: float = 500.0,
) -> Optional[LabelVote]:
    """Stubble burning: cropland, in the belt, in season, and short-lived.

    The persistence check matters — a permanent thermal source sitting in
    the middle of Punjab farmland is a brick kiln or a small industrial
    unit, not stubble burning, and mislabelling those would teach the
    model that cropland implies agricultural burning.

    The facility-proximity check matters just as much and was missing:
    a brand-new, single-detection hotspot right next to a factory has
    persistence_days=0 and no FRP history to compute a z-score from, so
    it can never satisfy industrial_rule's persistence/z-score gate —
    but it sailed straight through this rule uncontested (cropland is
    common in industrial belts too), teaching the model that facility
    proximity plus high cropland% plus a short-lived detection means
    agricultural burning. Same 500m radius flare_rule/industrial_rule
    already use to mean "on a facility".
    """
    region = region_containing(lon, lat, AGRI_BURN_REGIONS)
    if region is None:
        return None
    if acq_datetime is None or acq_datetime.month not in AGRI_BURN_MONTHS:
        return None
    if pct_cropland is None or pct_cropland < min_cropland:
        return None
    if persistence_days is not None and persistence_days > 5:
        return None
    if facility_distance_m is not None and facility_distance_m <= max_facility_distance_m:
        return None

    return LabelVote(
        label="agricultural_burning",
        confidence=0.80,
        source="AGRI_RULE",
        reason=(
            f"Cropland ({pct_cropland:.0%}) in the {region} belt during "
            f"month {acq_datetime.month}, short-lived"
        ),
    )


def wildfire_rule(
    pct_forest: Optional[float],
    facility_distance_m: Optional[float],
    spatial_growth_rate: Optional[float],
    fsi_matched: bool = False,
    min_forest: float = 0.5,
) -> Optional[LabelVote]:
    """Forest cover, away from industry, and ideally spreading.

    An FSI alert match is a much stronger signal than the land-cover
    heuristic, so it gets a higher confidence.
    """
    if fsi_matched:
        return LabelVote(
            label="wildfire",
            confidence=0.92,
            source="FSI",
            reason="Matched a Forest Survey of India forest-fire alert",
        )

    if pct_forest is None or pct_forest < min_forest:
        return None
    # Don't call it a wildfire if it's sitting on top of a plant.
    if facility_distance_m is not None and facility_distance_m < 1000:
        return None

    spreading = spatial_growth_rate is not None and spatial_growth_rate > 0.05
    return LabelVote(
        label="wildfire",
        confidence=0.75 if spreading else 0.60,
        source="LANDCOVER_FOREST",
        reason=(
            f"Forest cover {pct_forest:.0%} away from industrial infrastructure"
            + (", footprint spreading" if spreading else "")
        ),
    )


def industrial_rule(
    nearest_facility_type: Optional[str],
    facility_distance_m: Optional[float],
    persistence_days: Optional[float],
    frp_zscore: Optional[float],
    max_distance_m: float = 500.0,
    min_persistence_days: int = 14,
) -> Optional[LabelVote]:
    """Persistent heat on an industrial footprint (excluding flare sites,
    which the flare rule owns).

    A high FRP z-score means the source is running far above its own
    norm — the "abnormal event" case the project brief highlights. It's
    still labelled industrial_fire; the normal-vs-abnormal distinction is
    a separate axis carried by the z-score feature itself and M5's
    escalation forecast, not a sixth class.
    """
    industrial_types = {"refinery", "industrial", "power", "oil_well", "steel", "LNG"}
    if nearest_facility_type not in industrial_types or facility_distance_m is None:
        return None
    if facility_distance_m > max_distance_m:
        return None

    persistent = persistence_days is not None and persistence_days >= min_persistence_days
    anomalous = frp_zscore is not None and frp_zscore >= 2.0

    if not (persistent or anomalous):
        return None

    return LabelVote(
        label="industrial_fire",
        confidence=0.85 if persistent else 0.70,
        source="ZSCORE_INDUSTRIAL",
        reason=(
            f"On a {nearest_facility_type} footprint, "
            + (f"persistent {persistence_days:.0f}d" if persistent else "")
            + (f" FRP z-score {frp_zscore:.1f}" if anomalous else "")
        ).strip(),
    )


def collect_votes(
    lon: float,
    lat: float,
    features: Dict[str, Optional[float]],
    acq_datetime: Optional[datetime] = None,
    nearest_facility_type: Optional[str] = None,
    fsi_matched: bool = False,
) -> List[LabelVote]:
    votes = [
        flare_rule(
            nearest_facility_type,
            features.get("facility_distance_m"),
            features.get("persistence_days"),
        ),
        mining_rule(
            lon, lat, nearest_facility_type,
            features.get("facility_distance_m"), features.get("persistence_days"),
        ),
        agri_burn_rule(
            lon, lat, acq_datetime, features.get("pct_cropland"), features.get("persistence_days"),
            features.get("facility_distance_m"),
        ),
        wildfire_rule(
            features.get("pct_forest"),
            features.get("facility_distance_m"),
            features.get("spatial_growth_rate"),
            fsi_matched=fsi_matched,
        ),
        industrial_rule(
            nearest_facility_type,
            features.get("facility_distance_m"),
            features.get("persistence_days"),
            features.get("frp_zscore"),
        ),
    ]
    return [v for v in votes if v is not None]


def resolve_label(votes: Sequence[LabelVote], ambiguity_margin: float = 0.10) -> Optional[LabelVote]:
    """Pick the winning vote, or None when the evidence is contradictory.

    Returning None is a feature: a cluster two rules disagree about at
    similar confidence is exactly the kind of sample that would teach the
    model noise. Those go to analyst review (M1's incident review
    service) instead of into the training set.
    """
    if not votes:
        return None

    ranked = sorted(votes, key=lambda v: v.confidence, reverse=True)
    best = ranked[0]

    contenders = [v for v in ranked[1:] if v.label != best.label]
    if contenders and (best.confidence - contenders[0].confidence) < ambiguity_margin:
        return None

    return best


def is_ambiguous(votes: Sequence[LabelVote], ambiguity_margin: float = 0.10) -> bool:
    return bool(votes) and resolve_label(votes, ambiguity_margin) is None
