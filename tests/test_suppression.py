"""Suppression resource estimation — doctrine-based, not measured.

Same split as decision_engine/wui_analysis: pure functions, no DB, no
network. Every rate is checked against the cited real standard
(geospatial/suppression.py's own docstring), and every result must
self-report `is_estimate: True` so nothing downstream can present it as
a measurement.

Real-world trigger for most of this file: 582 of 903 real clusters in
this system (64%) are single-detection fires with a measured
spatial_extent_km of 0 — mathematically correct (no second point to
measure a pairwise distance against yet), but the earlier version of
this module read that as "zero area" and reported zero material for
every one of them. Every test here checks that a fire is never reported
as needing zero material or "N/A" fleet just because it hasn't been
observed by more than one pass.
"""
from geospatial.suppression import (
    AIR_SUPPORT_AREA_HA,
    DUMP_TRUCK_CAPACITY_M3,
    FOAM_APPLICATION_MINUTES,
    FOAM_RATE_L_PER_MIN_PER_M2,
    FRP_INTENSITY_MAX_MULTIPLIER,
    FRP_INTENSITY_REFERENCE_MW,
    GAS_FLARE_COOLING_RATE_L_PER_HA,
    SENSOR_PIXEL_AREA_HA,
    SMOTHER_DEPTH_M,
    UNCLASSIFIED_FALLBACK_RATE_L_PER_HA,
    WATER_RATE_L_PER_HA,
    WATER_TANKER_CAPACITY_L,
    air_support_recommended,
    effective_area_ha,
    estimate_suppression,
    intensity_multiplier,
)


# --- every fire gets a positive, non-N/A result ----------------------------

def test_a_single_detection_cluster_still_gets_real_nonzero_material():
    """The exact real-world case that exposed the bug: a single-
    detection cluster (area_ha=0.0, the honest measurement) must not
    report zero water and zero tanker trips."""
    result = estimate_suppression(
        "agricultural_burning", frp_max_mw=3.0, area_ha=0.0,
        coa_type="MONITOR_ECOLOGICAL_BENEFIT", sensor="VIIRS",
    )
    assert result["primary_volume_l"] > 0
    assert result["tanker_trips"] >= 1


def test_every_class_returns_a_result_with_is_estimate_true():
    for cls in ("wildfire", "agricultural_burning", "industrial_fire", "gas_flare", "mining", None, "unknown_class"):
        result = estimate_suppression(cls, frp_max_mw=10.0, area_ha=5.0, coa_type="MONITOR_ECOLOGICAL_BENEFIT")
        assert result["is_estimate"] is True


def test_unclassified_fire_gets_a_real_provisional_estimate_not_none():
    """'None can be zero or N/A' — an unclassified fire still gets a
    real number (the conservative Class A default), loudly flagged as
    provisional rather than a fabricated confident answer."""
    result = estimate_suppression(None, frp_max_mw=10.0, area_ha=5.0, coa_type=None)
    assert result["primary_volume_l"] > 0
    assert result["tanker_trips"] >= 1
    assert any("provisional" in n.lower() or "not yet classified" in n.lower() for n in result["notes"])


def test_zero_area_never_produces_zero_material_regardless_of_class():
    for cls in ("wildfire", "agricultural_burning", "industrial_fire", "mining"):
        result = estimate_suppression(cls, frp_max_mw=5.0, area_ha=0.0, coa_type="MONITOR_ECOLOGICAL_BENEFIT", sensor="VIIRS")
        primary = result.get("primary_volume_l") or result.get("primary_volume_m3")
        assert primary is not None and primary > 0, f"{cls} reported no material for a real detection"


def test_gas_flare_gets_a_real_nonzero_cooling_volume_and_fleet():
    """Fuel isolation is still the primary doctrine, but 'no bulk
    material' must not collapse into a blank result — cooling water and
    a response-team count are both real, quantifiable needs."""
    result = estimate_suppression("gas_flare", frp_max_mw=20.0, area_ha=1.0, coa_type="MONITOR_ECOLOGICAL_BENEFIT")
    assert result["primary_volume_l"] > 0
    assert result["tanker_trips"] >= 1
    assert result["well_control_units"] >= 1


# --- the sensor-pixel-footprint floor --------------------------------------

def test_effective_area_floors_at_the_real_sensor_pixel_size():
    assert effective_area_ha(0.0, "VIIRS") == SENSOR_PIXEL_AREA_HA["VIIRS"]
    assert effective_area_ha(0.0, "MODIS") == SENSOR_PIXEL_AREA_HA["MODIS"]


def test_effective_area_never_shrinks_a_real_larger_measurement():
    """The floor only ever raises a too-small reading — a real,
    already-larger measured footprint must pass through unchanged."""
    measured = SENSOR_PIXEL_AREA_HA["VIIRS"] * 10
    assert effective_area_ha(measured, "VIIRS") == measured


def test_unknown_sensor_still_gets_a_conservative_floor():
    assert effective_area_ha(0.0, None) > 0
    assert effective_area_ha(0.0, "SOME_UNKNOWN_PRODUCT") > 0


def test_modis_floor_is_larger_than_viirs_matching_real_pixel_sizes():
    """MODIS's 1 km pixel is real and coarser than VIIRS's 375 m one —
    the floor must reflect that, not use one number for both."""
    assert SENSOR_PIXEL_AREA_HA["MODIS"] > SENSOR_PIXEL_AREA_HA["VIIRS"]


def test_the_estimate_reports_both_measured_and_effective_area():
    """The operator should be able to see when the floor kicked in, not
    have a silently-substituted number."""
    result = estimate_suppression("wildfire", frp_max_mw=5.0, area_ha=0.0, coa_type="MONITOR_ECOLOGICAL_BENEFIT", sensor="VIIRS")
    assert result["measured_area_ha"] == 0.0
    assert result["area_ha"] == round(SENSOR_PIXEL_AREA_HA["VIIRS"], 2)
    assert any("floor" in n.lower() or "pixel" in n.lower() for n in result["notes"])


# --- FRP-based intensity scaling --------------------------------------

def test_intensity_multiplier_is_one_at_zero_or_missing_frp():
    assert intensity_multiplier(None) == 1.0
    assert intensity_multiplier(0.0) == 1.0


def test_intensity_multiplier_increases_with_real_frp():
    assert intensity_multiplier(FRP_INTENSITY_REFERENCE_MW) > 1.0
    assert intensity_multiplier(FRP_INTENSITY_REFERENCE_MW * 2) > intensity_multiplier(FRP_INTENSITY_REFERENCE_MW)


def test_intensity_multiplier_is_capped_for_very_high_frp():
    assert intensity_multiplier(FRP_INTENSITY_REFERENCE_MW * 100) == FRP_INTENSITY_MAX_MULTIPLIER


def test_higher_frp_fire_needs_more_water_than_an_identical_lower_frp_one():
    low = estimate_suppression("wildfire", frp_max_mw=1.0, area_ha=5.0, coa_type="MONITOR_ECOLOGICAL_BENEFIT")
    high = estimate_suppression("wildfire", frp_max_mw=FRP_INTENSITY_REFERENCE_MW * 2, area_ha=5.0, coa_type="MONITOR_ECOLOGICAL_BENEFIT")
    assert high["primary_volume_l"] > low["primary_volume_l"]


# --- water-class fires (wildfire / agricultural_burning) ------------------

def test_water_volume_scales_with_effective_area_at_the_cited_rate():
    result = estimate_suppression("agricultural_burning", frp_max_mw=0.0, area_ha=2.0, coa_type="MONITOR_ECOLOGICAL_BENEFIT", sensor="VIIRS")
    expected = max(2.0, SENSOR_PIXEL_AREA_HA["VIIRS"]) * WATER_RATE_L_PER_HA["agricultural_burning"]
    assert result["primary_volume_l"] == round(expected, 0)


def test_wildfire_uses_the_heavier_fuel_rate_than_agricultural():
    assert WATER_RATE_L_PER_HA["wildfire"] > WATER_RATE_L_PER_HA["agricultural_burning"]


def test_tanker_trips_is_never_zero_for_a_real_positive_volume():
    result = estimate_suppression("agricultural_burning", frp_max_mw=0.0, area_ha=0.001, coa_type="MONITOR_ECOLOGICAL_BENEFIT", sensor="VIIRS")
    assert result["primary_volume_l"] > 0
    assert result["tanker_trips"] >= 1


# --- industrial fire (foam, not water) -------------------------------------

def test_industrial_fire_gets_foam_not_water():
    result = estimate_suppression("industrial_fire", frp_max_mw=50.0, area_ha=1.0, coa_type="MONITOR_ECOLOGICAL_BENEFIT")
    assert result["resource_kind"] == "foam"
    assert "water" not in result["resource_label"].lower()


def test_industrial_fire_warns_against_water_as_primary_agent():
    result = estimate_suppression("industrial_fire", frp_max_mw=50.0, area_ha=1.0, coa_type="MONITOR_ECOLOGICAL_BENEFIT")
    assert any("spread" in n.lower() for n in result["notes"])


def test_industrial_foam_volume_matches_the_cited_nfpa_11_rate_at_effective_area():
    area_ha = 0.5
    result = estimate_suppression("industrial_fire", frp_max_mw=0.0, area_ha=area_ha, coa_type="MONITOR_ECOLOGICAL_BENEFIT", sensor="VIIRS")
    effective_ha = max(area_ha, SENSOR_PIXEL_AREA_HA["VIIRS"])
    expected = FOAM_RATE_L_PER_MIN_PER_M2 * FOAM_APPLICATION_MINUTES * (effective_ha * 10_000.0)
    assert result["primary_volume_l"] == round(expected, 0)


# --- gas flare (fuel isolation + real cooling water, no direct extinguish) -

def test_gas_flare_gets_fuel_isolation_as_primary_doctrine():
    result = estimate_suppression("gas_flare", frp_max_mw=20.0, area_ha=1.0, coa_type="MONITOR_ECOLOGICAL_BENEFIT")
    assert result["resource_kind"] == "fuel_isolation"


def test_gas_flare_cooling_volume_uses_the_lower_cooling_rate_not_direct_attack():
    area_ha = 2.0
    result = estimate_suppression("gas_flare", frp_max_mw=0.0, area_ha=area_ha, coa_type="MONITOR_ECOLOGICAL_BENEFIT", sensor="VIIRS")
    effective_ha = max(area_ha, SENSOR_PIXEL_AREA_HA["VIIRS"])
    expected = GAS_FLARE_COOLING_RATE_L_PER_HA * effective_ha
    assert result["primary_volume_l"] == round(expected, 0)
    assert GAS_FLARE_COOLING_RATE_L_PER_HA < WATER_RATE_L_PER_HA["agricultural_burning"]


def test_gas_flare_warns_about_reignition_risk():
    result = estimate_suppression("gas_flare", frp_max_mw=20.0, area_ha=1.0, coa_type="MONITOR_ECOLOGICAL_BENEFIT")
    assert any("shut off" in n.lower() for n in result["notes"])


# --- mining (excavation + smothering, not water tankers) -------------------

def test_mining_fire_gets_smothering_volume_not_water():
    result = estimate_suppression("mining", frp_max_mw=15.0, area_ha=1.0, coa_type="MONITOR_ECOLOGICAL_BENEFIT")
    assert result["resource_kind"] == "excavation_and_smothering"
    assert result["primary_volume_l"] is None
    assert result["primary_volume_m3"] > 0
    assert result["tanker_trips"] == 0


def test_mining_smothering_volume_matches_the_cited_depth_at_effective_area():
    area_ha = 2.0
    result = estimate_suppression("mining", frp_max_mw=0.0, area_ha=area_ha, coa_type="MONITOR_ECOLOGICAL_BENEFIT", sensor="VIIRS")
    effective_ha = max(area_ha, SENSOR_PIXEL_AREA_HA["VIIRS"])
    expected_m3 = effective_ha * 10_000.0 * SMOTHER_DEPTH_M
    assert result["primary_volume_m3"] == round(expected_m3, 0)


def test_mining_truck_loads_is_never_zero_for_a_real_footprint():
    result = estimate_suppression("mining", frp_max_mw=0.0, area_ha=0.0, coa_type="MONITOR_ECOLOGICAL_BENEFIT", sensor="VIIRS")
    assert result["truck_loads"] >= 1


# --- air support trigger ----------------------------------------------------

def test_air_support_triggers_on_full_suppression_tier_regardless_of_class():
    assert air_support_recommended("gas_flare", frp_max_mw=1.0, area_ha=0.1, coa_type="FULL_SUPPRESSION_AIR_TANKERS") is True


def test_air_support_triggers_on_crown_fire_scale_frp():
    assert air_support_recommended("wildfire", frp_max_mw=100.0, area_ha=0.1, coa_type="MONITOR_ECOLOGICAL_BENEFIT") is True
    assert air_support_recommended("wildfire", frp_max_mw=99.9, area_ha=0.1, coa_type="MONITOR_ECOLOGICAL_BENEFIT") is False


def test_air_support_triggers_on_large_footprint_regardless_of_intensity():
    assert air_support_recommended("wildfire", frp_max_mw=1.0, area_ha=AIR_SUPPORT_AREA_HA, coa_type="MONITOR_ECOLOGICAL_BENEFIT") is True
    assert air_support_recommended("wildfire", frp_max_mw=1.0, area_ha=AIR_SUPPORT_AREA_HA - 0.1, coa_type="MONITOR_ECOLOGICAL_BENEFIT") is False


def test_small_low_intensity_fire_does_not_recommend_air_support():
    assert air_support_recommended("agricultural_burning", frp_max_mw=2.0, area_ha=0.5, coa_type="MONITOR_ECOLOGICAL_BENEFIT") is False


# --- priority (low/medium/high, reusing the real RDI tier) ----------------

def test_priority_passes_through_the_real_coa_tier_not_a_new_scale():
    result = estimate_suppression("wildfire", frp_max_mw=5.0, area_ha=1.0, coa_type="POINT_ZONE_PROTECTION")
    assert result["priority"] == "POINT_ZONE_PROTECTION"
    assert "POINT" in result["priority_label"].upper()


def test_priority_level_maps_full_suppression_to_high():
    result = estimate_suppression("wildfire", frp_max_mw=5.0, area_ha=1.0, coa_type="FULL_SUPPRESSION_AIR_TANKERS")
    assert result["priority_level"] == "HIGH"


def test_priority_level_maps_point_zone_to_medium():
    result = estimate_suppression("wildfire", frp_max_mw=5.0, area_ha=1.0, coa_type="POINT_ZONE_PROTECTION")
    assert result["priority_level"] == "MEDIUM"


def test_priority_level_maps_monitor_to_low():
    result = estimate_suppression("wildfire", frp_max_mw=5.0, area_ha=1.0, coa_type="MONITOR_ECOLOGICAL_BENEFIT")
    assert result["priority_level"] == "LOW"


def test_priority_level_defaults_to_low_for_an_unknown_tier():
    result = estimate_suppression("wildfire", frp_max_mw=5.0, area_ha=1.0, coa_type=None)
    assert result["priority_level"] == "LOW"
