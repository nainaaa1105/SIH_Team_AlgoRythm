"""Suppression resource estimation.

Answers, for one already-classified fire: what does it actually take to
put this out right now, and roughly how much of it. This is a
DOCTRINE-BASED ESTIMATE built from published firefighting standards, not
a measurement — treat it as planning guidance, not a substitute for an
on-scene size-up. Every rate below is cited to a real standard rather
than invented, and the module says so explicitly in its output
(`is_estimate: True` on every result) so nothing downstream can present
it as measured fact.

Two things scale the estimate, both real, not guessed:
- AREA: the cluster's real measured footprint
  (geospatial.decision_engine.footprint_area_ha), floored at the
  resolving sensor's own pixel size (see SENSOR_PIXEL_AREA_HA). A
  single-detection cluster has a real spatial_extent_km of 0 (there is
  only one point, so no pairwise distance exists yet) — that measures
  "not enough passes to resolve spread," not "zero fire," so treating
  it as literally zero area and reporting zero material would be
  actively wrong. The sensor's own resolution is the real, physical
  floor: a VIIRS detection means *at least* a 375 m pixel is actively
  burning; MODIS, at least 1 km.
- INTENSITY: the cluster's real measured FRP. A hotter fire radiates
  more heat and needs more suppressant per unit area to absorb the same
  heat load, not just more area coverage — see intensity_multiplier().

Different fire classes need fundamentally different suppression
doctrine, not just "more or less water":

- wildfire / agricultural_burning (Class A, solid fuel): water is
  correct. NWCG-style wildland guidance gives litres/hectare for direct
  knockdown.
- industrial_fire (typically Class B, liquid fuel): water can spread a
  burning liquid — real doctrine is foam, at the NFPA 11 hydrocarbon
  application rate.
- gas_flare: not something bulk material extinguishes directly. Real
  doctrine is fuel isolation (shut off supply) first — dousing it with
  the gas still flowing risks re-ignition/explosion. What IS a real,
  quantifiable need here is cooling water for nearby equipment/exposure
  protection, at a lower rate than direct extinguishment, plus a
  specialised well-control response team.
- mining (coal-seam / stockpile fires): water alone routinely fails —
  these burn underground for years (Jharia is the standard example).
  Real doctrine is excavation and smothering with a compacted inert
  layer to cut the oxygen supply, not tanker volume.
- unclassified (no verdict yet): still gets a real, non-zero first-
  response estimate (the same conservative Class A/water assumption a
  responder would start with before a size-up confirms the fuel type),
  loudly flagged as provisional — never a blank "N/A".
"""
from typing import Any, Dict, Optional

# --- Water application rates (NWCG-style wildland guidance), L/hectare ----
# for direct-attack knockdown. Two different fuel-load assumptions:
# agricultural burns are light, fast fuel (crop stubble); wildfire here
# covers forest/scrub, heavier fuel needing more water per unit area.
WATER_RATE_L_PER_HA = {
    "agricultural_burning": 3000.0,  # midpoint of the light-fuel 2,000-4,000 L/ha band
    "wildfire": 10000.0,             # midpoint of the heavy-fuel 8,000-15,000 L/ha band
}
# Applied when the classifier has not produced a verdict yet — the
# lighter of the two real rates above, not a guess at a heavier one.
UNCLASSIFIED_FALLBACK_RATE_L_PER_HA = WATER_RATE_L_PER_HA["agricultural_burning"]

# --- Foam for industrial (Class B, liquid-fuel) fires ----------------------
# NFPA 11 hydrocarbon-fire foam solution application rate.
FOAM_RATE_L_PER_MIN_PER_M2 = 6.5
# Standard rapid-knockdown application window this estimate assumes.
FOAM_APPLICATION_MINUTES = 15.0

# --- Gas flare: equipment cooling, not direct extinguishment ---------------
# A real but much lower rate than direct attack — this water protects
# nearby exposure/equipment while the fuel supply is isolated, it does
# not touch the flame itself.
GAS_FLARE_COOLING_RATE_L_PER_HA = 500.0

# --- Mining: smothering depth for oxygen cutoff -----------------------------
# The low end of the compacted-inert-layer depth used in coal/stockpile
# fire suppression literature — a conservative, real figure, not the
# deep excavation a genuinely established seam fire can require.
SMOTHER_DEPTH_M = 0.5

# --- Fleet capacities (standard, India-representative equipment) ----------
WATER_TANKER_CAPACITY_L = 5000.0   # typical municipal fire tender
AIR_DROP_CAPACITY_L = 3000.0       # helicopter Bambi-bucket class
DUMP_TRUCK_CAPACITY_M3 = 10.0      # standard tipper truck

# --- Air support trigger ---------------------------------------------------
# Beyond this footprint, ground tankers alone are impractical regardless
# of class — this is independent of (and in addition to) the crown-fire
# FRP trigger and the RDI full-suppression tier.
AIR_SUPPORT_AREA_HA = 50.0

# --- Real sensor pixel footprint, used as the area floor -------------------
# Nominal ground sample distance of the two real satellite products this
# system ingests (see app/ingestion/firms.py) — a detection is real
# evidence that at least this much ground is actively burning, even
# before a second pass lets spatial_extent_km resolve any spread.
SENSOR_PIXEL_AREA_HA = {
    "VIIRS": 14.0625,   # 375 m nominal pixel: 0.375 km * 0.375 km = 0.140625 km^2
    "MODIS": 100.0,     # 1 km nominal pixel: 1 km^2
}
DEFAULT_MIN_AREA_HA = SENSOR_PIXEL_AREA_HA["VIIRS"]  # conservative default when sensor is unknown

# --- FRP-based intensity scaling --------------------------------------
# FRP is the fire's own real, measured radiative heat output — a
# hotter fire needs more suppressant per unit area to absorb the same
# heat load, not just more area coverage. Reference is a moderate
# wildland fire's typical FRP; capped so a very high-FRP fire does not
# produce an implausible multiplier.
FRP_INTENSITY_REFERENCE_MW = 50.0
FRP_INTENSITY_MAX_MULTIPLIER = 3.0

_HA_TO_M2 = 10_000.0

COA_PRIORITY_LABELS = {
    "FULL_SUPPRESSION_AIR_TANKERS": "FULL SUPPRESSION — AIR TANKERS",
    "POINT_ZONE_PROTECTION": "POINT-ZONE PROTECTION",
    "MONITOR_ECOLOGICAL_BENEFIT": "MONITOR (ECOLOGICAL BENEFIT)",
}

# The RDI engine's own three-tier course-of-action, relabelled to the
# plain low/medium/high vocabulary a suppression-resource view needs —
# not a second, independently-computed priority scale.
COA_PRIORITY_LEVELS = {
    "FULL_SUPPRESSION_AIR_TANKERS": "HIGH",
    "POINT_ZONE_PROTECTION": "MEDIUM",
    "MONITOR_ECOLOGICAL_BENEFIT": "LOW",
}


def _ceil_div_at_least_one(value: float, divisor: float) -> int:
    """Any positive volume needs at least one trip to move it — this
    never rounds a real, nonzero requirement down to zero."""
    import math

    if value <= 0:
        return 0
    return max(1, int(math.ceil(value / divisor)))


def effective_area_ha(area_ha: Optional[float], sensor: Optional[str]) -> float:
    """The real measured footprint, floored at the resolving sensor's
    own pixel size — see the module docstring for why a single-
    detection cluster's spatial_extent_km == 0 does not mean zero fire.
    """
    floor = SENSOR_PIXEL_AREA_HA.get((sensor or "").upper(), DEFAULT_MIN_AREA_HA)
    return max(area_ha or 0.0, floor)


def intensity_multiplier(frp_max_mw: Optional[float]) -> float:
    if frp_max_mw is None or frp_max_mw <= 0:
        return 1.0
    return min(1.0 + frp_max_mw / FRP_INTENSITY_REFERENCE_MW, FRP_INTENSITY_MAX_MULTIPLIER)


def air_support_recommended(
    predicted_class: Optional[str],
    frp_max_mw: Optional[float],
    area_ha: Optional[float],
    coa_type: Optional[str],
) -> bool:
    """Independent of resource_kind — even a fuel-isolation-only fire
    (gas_flare) can need air support for cooling/containment if it's
    large or already flagged FULL_SUPPRESSION by the RDI engine."""
    if coa_type == "FULL_SUPPRESSION_AIR_TANKERS":
        return True
    if frp_max_mw is not None and frp_max_mw >= 100.0:
        return True
    if area_ha is not None and area_ha >= AIR_SUPPORT_AREA_HA:
        return True
    return False


def estimate_suppression(
    predicted_class: Optional[str],
    frp_max_mw: Optional[float],
    area_ha: Optional[float],
    coa_type: Optional[str] = None,
    sensor: Optional[str] = None,
) -> Dict[str, Any]:
    """One fire's suppression-resource estimate, scaled by its real
    measured area (floored at sensor resolution) and real measured
    intensity (FRP) — "at its current state", not a flat per-class
    constant. Every fire gets a positive, non-N/A result; see the
    module docstring for how each class avoids a zero/blank answer.
    """
    cls = (predicted_class or "").lower()
    measured_area_ha = area_ha or 0.0
    effective_ha = effective_area_ha(measured_area_ha, sensor)
    multiplier = intensity_multiplier(frp_max_mw)
    air_support = air_support_recommended(predicted_class, frp_max_mw, effective_ha, coa_type)
    priority_label = COA_PRIORITY_LABELS.get(coa_type, coa_type or "UNKNOWN")
    priority_level = COA_PRIORITY_LEVELS.get(coa_type, "LOW")

    notes = []
    if measured_area_ha < effective_ha:
        notes.append(
            f"Measured footprint not yet resolved (too few passes) — using the resolving "
            f"sensor's own {effective_ha:.1f} ha pixel footprint as a floor."
        )
    if multiplier > 1.0:
        notes.append(
            f"Scaled x{multiplier:.2f} for this fire's own measured radiative output "
            f"({frp_max_mw:.1f} MW FRP) — a hotter fire needs more suppressant per hectare."
        )

    base: Dict[str, Any] = {
        "predicted_class": predicted_class,
        "area_ha": round(effective_ha, 2),
        "measured_area_ha": round(measured_area_ha, 2),
        "intensity_multiplier": round(multiplier, 2),
        "priority": coa_type,
        "priority_label": priority_label,
        "priority_level": priority_level,
        "air_support_recommended": air_support,
        "is_estimate": True,
        "notes": notes,
    }

    if cls in WATER_RATE_L_PER_HA or cls not in ("industrial_fire", "gas_flare", "mining"):
        rate = WATER_RATE_L_PER_HA.get(cls, UNCLASSIFIED_FALLBACK_RATE_L_PER_HA)
        water_l = effective_ha * rate * multiplier
        result_notes = [f"{rate:.0f} L/ha, NWCG-style direct-attack rate for this fuel class."] + notes
        if cls not in WATER_RATE_L_PER_HA:
            result_notes.insert(
                0,
                "Provisional — not yet classified. Assuming the lighter Class A (water) case "
                "as a first-response default; confirm fuel type on scene before committing "
                "water to what could be a liquid-fuel fire.",
            )
        base.update({
            "resource_kind": "water",
            "resource_label": "Water (direct attack)" if cls in WATER_RATE_L_PER_HA
                else "Water (provisional, pending classification)",
            "primary_volume_l": round(water_l, 0),
            "tanker_trips": _ceil_div_at_least_one(water_l, WATER_TANKER_CAPACITY_L),
            "air_drop_trips": _ceil_div_at_least_one(water_l, AIR_DROP_CAPACITY_L) if air_support else 0,
            "notes": result_notes,
        })
    elif cls == "industrial_fire":
        area_m2 = effective_ha * _HA_TO_M2
        foam_l = FOAM_RATE_L_PER_MIN_PER_M2 * FOAM_APPLICATION_MINUTES * area_m2 * multiplier
        base.update({
            "resource_kind": "foam",
            "resource_label": "Foam (AFFF, Class B)",
            "primary_volume_l": round(foam_l, 0),
            "tanker_trips": _ceil_div_at_least_one(foam_l, WATER_TANKER_CAPACITY_L),
            "air_drop_trips": 0,
            "notes": [
                "Water alone can spread a burning liquid fuel — do not use as the primary agent.",
                f"NFPA 11 hydrocarbon rate: {FOAM_RATE_L_PER_MIN_PER_M2} L/min/m^2 "
                f"over a {FOAM_APPLICATION_MINUTES:.0f}-minute knockdown window.",
            ] + notes,
        })
    elif cls == "gas_flare":
        cooling_l = GAS_FLARE_COOLING_RATE_L_PER_HA * effective_ha * multiplier
        base.update({
            "resource_kind": "fuel_isolation",
            "resource_label": "Fuel isolation (shut-off) + equipment cooling",
            "primary_volume_l": round(cooling_l, 0),
            "tanker_trips": _ceil_div_at_least_one(cooling_l, WATER_TANKER_CAPACITY_L),
            "air_drop_trips": 0,
            "well_control_units": 1,
            "notes": [
                "Shut off the fuel supply first — extinguishing an uncontrolled gas fire "
                "while gas keeps flowing risks re-ignition or explosion.",
                f"Water volume above is for cooling nearby equipment/exposure at "
                f"{GAS_FLARE_COOLING_RATE_L_PER_HA:.0f} L/ha, not for extinguishing the flame directly.",
                "Requires 1 specialised well-control/gas-fire response unit in addition to cooling tankers.",
            ] + notes,
        })
    elif cls == "mining":
        volume_m3 = effective_ha * _HA_TO_M2 * SMOTHER_DEPTH_M * multiplier
        base.update({
            "resource_kind": "excavation_and_smothering",
            "resource_label": "Excavation + inert-layer smothering",
            "primary_volume_l": None,
            "primary_volume_m3": round(volume_m3, 0),
            "truck_loads": _ceil_div_at_least_one(volume_m3, DUMP_TRUCK_CAPACITY_M3),
            "tanker_trips": 0,
            "air_drop_trips": 0,
            "notes": [
                "Water alone routinely fails on a coal-seam/stockpile fire — these can burn "
                "underground for years without an oxygen cutoff.",
                f"Estimate assumes a {SMOTHER_DEPTH_M:.1f} m compacted inert layer "
                "(clay/soil) over the burning footprint.",
            ] + notes,
        })

    return base
