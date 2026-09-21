"""Expected emission profile by facility type.

This is a lookup of what a responder should expect to be in the smoke,
keyed on the facility type M1 stores on `facilities.facility_type` (set
by its OSM/GEM bulk loaders). It drives the "what is in this plume"
line on the event card.

These are indicative profiles for responder awareness, drawn from the
general process chemistry of each industry — they are NOT measured
emission factors for a specific plant, and nothing here should be
presented as a quantitative emission estimate.
"""
from typing import Dict, List, Optional

# facility_type values M1 actually writes (see app/enrichment/osm_facilities.py
# _FACILITY_TYPE_BY_TAG and scripts/bulk_load_gem.py): refinery, industrial,
# power, oil_well, mine, flare, steel, LNG.
_PROFILES: Dict[str, Dict[str, object]] = {
    "refinery": {
        "primary": ["SO2", "H2S", "VOC", "benzene", "particulates"],
        "hazard": "high",
        "note": "Sour-gas and hydrocarbon inventory; H2S is immediately dangerous to life at low ppm.",
    },
    "flare": {
        "primary": ["CO2", "CO", "NOx", "unburnt hydrocarbons", "soot"],
        "hazard": "moderate",
        "note": "Routine flaring is designed combustion; a sudden change in flare behaviour matters more than the flare itself.",
    },
    "oil_well": {
        "primary": ["CH4", "H2S", "VOC", "condensate aerosol"],
        "hazard": "high",
        "note": "Blowout risk; methane is an asphyxiant and explosion hazard before it is a toxicity hazard.",
    },
    "LNG": {
        "primary": ["CH4", "cryogenic vapour cloud"],
        "hazard": "high",
        "note": "Dense cold vapour hugs the ground initially; flammable envelope matters more than toxicity.",
    },
    "power": {
        "primary": ["SO2", "NOx", "fly ash", "particulates", "CO"],
        "hazard": "moderate",
        "note": "Coal-fired plants add heavy-metal-bearing fly ash to the particulate load.",
    },
    "steel": {
        "primary": ["CO", "particulates", "metal oxides", "SO2"],
        "hazard": "moderate",
        "note": "Coke and blast-furnace gas carry high CO; metal-oxide fume affects respiratory tracts.",
    },
    "mine": {
        "primary": ["CO", "particulates", "coal smoke", "SO2"],
        "hazard": "moderate",
        "note": "Coal-seam fires produce sustained CO and dense particulate smoke over long periods.",
    },
    "industrial": {
        "primary": ["particulates", "CO", "VOC"],
        "hazard": "moderate",
        "note": "Generic industrial footprint; the actual inventory depends on the specific process on site.",
    },
}

# Non-industrial classes: the source class from M2's classifier, not a
# facility type, determines what is burning.
_CLASS_PROFILES: Dict[str, Dict[str, object]] = {
    "wildfire": {
        "primary": ["PM2.5", "CO", "VOC", "NOx"],
        "hazard": "moderate",
        "note": "Vegetation smoke; PM2.5 is the dominant public-health concern over distance.",
    },
    "agricultural_burning": {
        "primary": ["PM2.5", "CO", "CH4", "NH3"],
        "hazard": "moderate",
        "note": "Crop-residue smoke; the regional air-quality burden comes from many simultaneous small fires.",
    },
}

_UNKNOWN: Dict[str, object] = {
    "primary": ["particulates", "CO"],
    "hazard": "unknown",
    "note": "Source type not identified; treat combustion products as the default expectation.",
}


def chemical_profile(
    facility_type: Optional[str] = None, predicted_class: Optional[str] = None
) -> Dict[str, object]:
    """Expected emissions for a plume.

    Facility type wins when we have one, because a located industrial
    footprint tells you far more about the inventory than the broad
    source class does. Falls back to the classifier's verdict, then to a
    generic combustion profile.
    """
    if facility_type:
        profile = _PROFILES.get(facility_type.lower())
        if profile:
            return {**profile, "basis": f"facility_type={facility_type}"}

    if predicted_class:
        profile = _CLASS_PROFILES.get(predicted_class.lower())
        if profile:
            return {**profile, "basis": f"predicted_class={predicted_class}"}
        # An industrial verdict with no located facility still implies
        # an industrial inventory, just an unidentified one.
        if predicted_class.lower() in ("industrial_fire", "gas_flare", "mining"):
            return {**_PROFILES["industrial"], "basis": f"predicted_class={predicted_class}"}

    return {**_UNKNOWN, "basis": "unknown"}


def known_facility_types() -> List[str]:
    return sorted(_PROFILES)
