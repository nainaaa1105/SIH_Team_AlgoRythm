from datetime import datetime, timezone

from classifier.labels.rules import (
    agri_burn_rule,
    collect_votes,
    flare_rule,
    industrial_rule,
    is_ambiguous,
    mining_rule,
    region_containing,
    resolve_label,
    wildfire_rule,
    AGRI_BURN_REGIONS,
    MINING_REGIONS,
)

OCT = datetime(2026, 10, 15, tzinfo=timezone.utc)
JUL = datetime(2026, 7, 15, tzinfo=timezone.utc)


def test_flare_rule_fires_on_a_catalogued_flare_site():
    vote = flare_rule("flare", facility_distance_m=120, persistence_days=30)
    assert vote is not None
    assert vote.label == "gas_flare"
    assert vote.source == "GGFR"
    assert vote.confidence >= 0.9


def test_flare_rule_less_confident_for_a_one_off_detection():
    persistent = flare_rule("flare", 120, persistence_days=30)
    transient = flare_rule("flare", 120, persistence_days=0.5)
    assert transient.confidence < persistent.confidence


def test_flare_rule_ignores_distant_or_non_flare_facilities():
    assert flare_rule("flare", facility_distance_m=5000, persistence_days=30) is None
    assert flare_rule("refinery", facility_distance_m=100, persistence_days=30) is None
    assert flare_rule(None, None, None) is None


def test_mining_rule_fires_inside_a_known_coalfield():
    vote = mining_rule(86.3, 23.75)  # Jharia
    assert vote is not None and vote.label == "mining"
    assert "jharia" in vote.reason.lower()


def test_mining_rule_silent_outside_coalfields():
    assert mining_rule(77.1, 28.7) is None


def test_mining_rule_fires_on_a_real_quarry_match_anywhere():
    """Regression: mining_rule used to only fire inside the three named
    coalfield boxes, so real quarrying activity everywhere else in
    India — 9,966 real OSM quarry/mine facilities, not just Jharia/
    Korba/Singrauli — never got a mining vote at all, and fell through
    to whatever else fired, often confusable with a cropland-percentage
    agricultural-burn heuristic. A located, PERSISTENT quarry match must
    win outside the coalfields too, same as flare_rule/industrial_rule
    already do."""
    vote = mining_rule(77.1, 28.7, nearest_facility_type="mine", facility_distance_m=120, persistence_days=5)
    assert vote is not None
    assert vote.label == "mining"
    assert vote.source == "OSM_QUARRY"
    assert vote.confidence >= 0.8


def test_mining_rule_requires_persistence_for_a_confident_quarry_match():
    """Regression for a real precision bug found in a live retrain:
    firing on quarry proximity alone (no persistence check) pulled in
    ~1,260 real industrial_fire and ~1,470 real agricultural_burning
    test cases as false "mining" (0.249 precision) — a quarry sitting
    near an unrelated, one-off fire doesn't make that fire mining
    activity. Real mining/coal-seam fires are persistent by nature; a
    single detection near a quarry must NOT get the confident OSM_QUARRY
    vote, same reasoning flare_rule already applies to a one-off flare
    hit."""
    single_detection = mining_rule(
        77.1, 28.7, nearest_facility_type="mine", facility_distance_m=120, persistence_days=0,
    )
    assert single_detection is None or single_detection.source != "OSM_QUARRY"


def test_mining_rule_real_quarry_match_beats_the_regional_prior():
    """A located, persistent quarry match 100m away is stronger evidence
    than the coarse regional box, same hierarchy the flare/coalfield
    test already protects for the reverse case."""
    inside_coalfield_no_quarry = mining_rule(86.3, 23.75)
    inside_coalfield_with_quarry = mining_rule(86.3, 23.75, "mine", 100, persistence_days=5)
    assert inside_coalfield_with_quarry.confidence > inside_coalfield_no_quarry.confidence
    assert inside_coalfield_with_quarry.source == "OSM_QUARRY"


def test_mining_rule_ignores_a_distant_or_non_mine_facility():
    assert mining_rule(77.1, 28.7, "mine", 5000, persistence_days=10) is None
    assert mining_rule(77.1, 28.7, "industrial", 100, persistence_days=10) is None


def test_agri_burn_rule_requires_region_season_and_cropland():
    vote = agri_burn_rule(75.5, 30.5, OCT, pct_cropland=0.85, persistence_days=1)
    assert vote is not None and vote.label == "agricultural_burning"

    # Right place and crop, wrong season
    assert agri_burn_rule(75.5, 30.5, JUL, 0.85, 1) is None
    # Right place and season, not cropland
    assert agri_burn_rule(75.5, 30.5, OCT, 0.1, 1) is None
    # Right season and crop, wrong region
    assert agri_burn_rule(85.0, 23.0, OCT, 0.85, 1) is None


def test_agri_burn_rule_rejects_persistent_sources_on_farmland():
    """A permanent hot spot in Punjab farmland is a brick kiln, not
    stubble burning — mislabelling it would teach the model that cropland
    implies agricultural burning."""
    assert agri_burn_rule(75.5, 30.5, OCT, pct_cropland=0.9, persistence_days=60) is None


def test_agri_burn_rule_rejects_sources_near_a_real_facility():
    """Regression for the live bug: a brand-new, single-detection hotspot
    (persistence_days=0, no FRP history) right next to a real factory has
    no way to satisfy industrial_rule's persistence/z-score gate, so
    without this check agri_burn_rule fired uncontested on high-cropland
    industrial-belt land — 5 real live clusters (92m-313m from a real
    facility) were misclassified agricultural_burning this way."""
    assert agri_burn_rule(
        75.5, 30.5, OCT, pct_cropland=0.85, persistence_days=0, facility_distance_m=92,
    ) is None
    # Right at the 500m radius flare_rule/industrial_rule already use —
    # inclusive, matching their own max_distance_m boundary.
    assert agri_burn_rule(
        75.5, 30.5, OCT, pct_cropland=0.85, persistence_days=0, facility_distance_m=500,
    ) is None
    # Comfortably past it, the rule fires exactly as before.
    vote = agri_burn_rule(
        75.5, 30.5, OCT, pct_cropland=0.85, persistence_days=0, facility_distance_m=501,
    )
    assert vote is not None and vote.label == "agricultural_burning"
    # No facility data at all must not silently suppress the rule.
    assert agri_burn_rule(
        75.5, 30.5, OCT, pct_cropland=0.85, persistence_days=0, facility_distance_m=None,
    ) is not None


def test_wildfire_rule_prefers_an_fsi_match_over_the_landcover_heuristic():
    fsi = wildfire_rule(None, None, None, fsi_matched=True)
    heuristic = wildfire_rule(0.8, 5000, 0.2, fsi_matched=False)
    assert fsi.source == "FSI" and heuristic.source == "LANDCOVER_FOREST"
    assert fsi.confidence > heuristic.confidence


def test_wildfire_rule_declines_when_sitting_on_a_facility():
    assert wildfire_rule(0.9, facility_distance_m=200, spatial_growth_rate=0.3) is None


def test_wildfire_rule_more_confident_when_footprint_is_spreading():
    spreading = wildfire_rule(0.8, 5000, spatial_growth_rate=0.5)
    static = wildfire_rule(0.8, 5000, spatial_growth_rate=0.0)
    assert spreading.confidence > static.confidence


def test_industrial_rule_fires_on_persistent_heat_at_a_plant():
    vote = industrial_rule("refinery", 200, persistence_days=40, frp_zscore=0.5)
    assert vote is not None and vote.label == "industrial_fire"


def test_industrial_rule_fires_on_an_anomalous_spike_even_if_not_persistent():
    vote = industrial_rule("power", 100, persistence_days=2, frp_zscore=5.0)
    assert vote is not None and vote.label == "industrial_fire"


def test_industrial_rule_silent_for_ordinary_short_lived_heat():
    assert industrial_rule("refinery", 200, persistence_days=1, frp_zscore=0.2) is None


def test_industrial_rule_does_not_claim_flare_sites():
    """Flare sites belong to the flare rule; overlapping ownership would
    make every flare ambiguous."""
    assert industrial_rule("flare", 100, persistence_days=40, frp_zscore=0.1) is None


def test_a_located_facility_match_outranks_a_coarse_regional_prior():
    """A catalogued flare site 100m away is much stronger evidence than
    'somewhere inside a 50km coalfield box'. If these two sat within the
    ambiguity margin, every flare stack and pithead plant inside a
    coalfield would be dropped from training as 'ambiguous'."""
    votes = collect_votes(
        86.3, 23.75,  # inside the Jharia coalfield
        {"facility_distance_m": 100, "persistence_days": 30, "pct_forest": 0.2},
        acq_datetime=OCT, nearest_facility_type="flare",
    )
    winner = resolve_label(votes)
    assert winner is not None, "a located flare match must beat the regional mining prior"
    assert winner.label == "gas_flare"


def test_industrial_facility_inside_a_coalfield_is_not_labelled_mining():
    """Pithead thermal power plants sit inside coalfields constantly."""
    votes = collect_votes(
        82.7, 22.35,  # inside Korba
        {"facility_distance_m": 150, "persistence_days": 40, "frp_zscore": 0.3},
        acq_datetime=OCT, nearest_facility_type="power",
    )
    winner = resolve_label(votes)
    assert winner is not None and winner.label == "industrial_fire"


def test_a_coalfield_cluster_with_no_facility_nearby_is_labelled_mining():
    votes = collect_votes(
        86.3, 23.75,
        {"facility_distance_m": 8000, "persistence_days": 30, "pct_forest": 0.1},
        acq_datetime=OCT, nearest_facility_type=None,
    )
    winner = resolve_label(votes)
    assert winner is not None and winner.label == "mining"


def test_spreading_forest_fire_inside_a_coalfield_stays_ambiguous():
    """Genuinely unclear — a spreading fire in forest inside a coalfield
    could be either, and belongs in analyst review rather than being
    guessed at in the training set."""
    votes = collect_votes(
        86.3, 23.75,
        {"facility_distance_m": 9000, "pct_forest": 0.85, "spatial_growth_rate": 0.5},
        acq_datetime=OCT, nearest_facility_type=None,
    )
    assert resolve_label(votes) is None
    assert is_ambiguous(votes)


def test_a_fresh_hotspot_next_to_a_facility_in_cropland_is_not_agri_burn():
    """End-to-end regression for the live bug via collect_votes(): a
    single-detection hotspot (persistence_days=0, no FRP history so no
    z-score) 92m from an industrial facility, in the Punjab/Haryana belt,
    in season, on high-cropland land. Before the fix, agri_burn_rule was
    the only vote and won by default; now it's excluded, leaving the
    cluster correctly ambiguous (industrial_rule also needs persistence
    or a z-score it doesn't have) rather than confidently mislabelled."""
    votes = collect_votes(
        75.5, 30.5,
        {"facility_distance_m": 92, "persistence_days": 0, "pct_cropland": 0.85},
        acq_datetime=OCT, nearest_facility_type="industrial",
    )
    assert not any(v.label == "agricultural_burning" for v in votes)


def test_resolve_label_returns_none_when_rules_contradict_closely():
    """Near-tied contradictory rules mean a genuinely ambiguous cluster;
    it belongs in analyst review, not the training set."""
    from classifier.labels.rules import LabelVote

    votes = [
        LabelVote("wildfire", 0.80, "LANDCOVER_FOREST", ""),
        LabelVote("mining", 0.85, "MINING_REGION", ""),
    ]
    assert resolve_label(votes) is None
    assert is_ambiguous(votes)


def test_resolve_label_accepts_a_clear_winner():
    from classifier.labels.rules import LabelVote

    votes = [
        LabelVote("wildfire", 0.60, "LANDCOVER_FOREST", ""),
        LabelVote("gas_flare", 0.90, "GGFR", ""),
    ]
    winner = resolve_label(votes)
    assert winner.label == "gas_flare"
    assert not is_ambiguous(votes)


def test_resolve_label_handles_no_votes():
    assert resolve_label([]) is None
    assert not is_ambiguous([])


def test_agreeing_votes_are_not_ambiguous():
    from classifier.labels.rules import LabelVote

    votes = [
        LabelVote("wildfire", 0.75, "FSI", ""),
        LabelVote("wildfire", 0.70, "LANDCOVER_FOREST", ""),
    ]
    assert resolve_label(votes).label == "wildfire"


def test_region_lookup_returns_the_containing_region():
    assert region_containing(86.3, 23.75, MINING_REGIONS) == "jharia"
    assert region_containing(75.5, 30.5, AGRI_BURN_REGIONS) == "punjab_haryana"
    assert region_containing(0.0, 0.0, MINING_REGIONS) is None
