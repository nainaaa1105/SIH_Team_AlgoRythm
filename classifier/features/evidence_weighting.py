"""Evidence-weighting engine.

The problem it solves: a cluster under heavy cloud has no usable
Sentinel-2 imagery, so every M4 feature is missing. A cluster M5 hasn't
processed yet has no rhythm features. Naively imputing zeros would tell
the model "NDVI is exactly 0 here", which is a lie that shifts the
prediction; dropping the cluster entirely would mean the system goes
blind exactly when cloud cover is worst.

What we do instead:
  1. Missing groups stay NaN. XGBoost has native missing-value handling
     (it learns a default split direction per node), so NaN genuinely
     means "unknown" to the model rather than "zero".
  2. The final *confidence* is discounted in proportion to how much
     evidence weight was unavailable, and the reason is recorded so the
     dashboard's XAI panel can say "no optical corroboration: heavy
     cloud" instead of silently showing a lower number.

Group weights reflect how much each block of evidence contributes to
separating the five classes, per the project research notes: land-cover
context and thermal/temporal behaviour are the strongest signals, with
imagery as corroboration and rhythm as a refinement.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from classifier.features.schema import FEATURE_COLUMNS, FeatureGroup, columns_in_group

# Must sum to 1.0 — asserted at import time below.
GROUP_WEIGHTS: Dict[FeatureGroup, float] = {
    FeatureGroup.CONTEXT: 0.30,
    FeatureGroup.THERMAL: 0.25,
    FeatureGroup.TEMPORAL: 0.20,
    FeatureGroup.SPATIAL: 0.10,
    FeatureGroup.IMAGERY: 0.10,
    FeatureGroup.RHYTHM: 0.05,
}

assert math.isclose(sum(GROUP_WEIGHTS.values()), 1.0), "GROUP_WEIGHTS must sum to 1.0"

# Never discount confidence below this, however much evidence is missing:
# the thermal detection itself is always real, we just know less about it.
MIN_CONFIDENCE_MULTIPLIER = 0.5


@dataclass
class EvidenceAssessment:
    """Result of weighing what evidence was actually available."""

    available_groups: List[FeatureGroup]
    missing_groups: List[FeatureGroup]
    available_weight: float
    confidence_multiplier: float
    notes: Dict[str, str] = field(default_factory=dict)

    def as_json(self) -> dict:
        return {
            "available_groups": [g.value for g in self.available_groups],
            "missing_groups": [g.value for g in self.missing_groups],
            "available_weight": round(self.available_weight, 4),
            "confidence_multiplier": round(self.confidence_multiplier, 4),
            "notes": self.notes,
        }


def group_is_present(features: Dict[str, Optional[float]], group: FeatureGroup) -> bool:
    """A group counts as present if ANY of its columns has a real value.

    Deliberately "any" rather than "all": M4 may deliver Dozier
    temperature but no NDVI for a night-time scene, and that partial
    evidence is still worth more than nothing.
    """
    for column in columns_in_group(group):
        value = features.get(column)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        return True
    return False


def assess_evidence(
    features: Dict[str, Optional[float]],
    optical_available: Optional[bool] = None,
    cloud_fraction: Optional[float] = None,
) -> EvidenceAssessment:
    """Decide which evidence groups are usable and how much to discount
    confidence as a result.

    `optical_available` and `cloud_fraction` come straight off M1's
    `clusters` row (its cloud gate already computed them), and are used
    to explain *why* imagery is missing rather than to decide whether it
    is — the features themselves are the authority on that.
    """
    available: List[FeatureGroup] = []
    missing: List[FeatureGroup] = []
    notes: Dict[str, str] = {}

    for group in FeatureGroup:
        if group_is_present(features, group):
            available.append(group)
        else:
            missing.append(group)

    for group in missing:
        if group is FeatureGroup.IMAGERY:
            if optical_available is False:
                pct = f"{cloud_fraction:.0%}" if cloud_fraction is not None else "unknown"
                notes[group.value] = (
                    f"No optical corroboration: cloud fraction {pct} exceeded the usable "
                    "threshold, so no Sentinel-2 patch was fetched for this cluster."
                )
            else:
                notes[group.value] = "Imagery features not yet computed for this cluster."
        elif group is FeatureGroup.RHYTHM:
            notes[group.value] = (
                "Rhythm/forecast features unavailable — the cluster has too short a "
                "history, or M5's analysis has not run for it yet."
            )
        else:
            notes[group.value] = f"{group.value} features unavailable for this cluster."

    available_weight = sum(GROUP_WEIGHTS[g] for g in available)

    # Discount confidence toward the floor in proportion to missing weight.
    confidence_multiplier = MIN_CONFIDENCE_MULTIPLIER + (
        (1.0 - MIN_CONFIDENCE_MULTIPLIER) * available_weight
    )

    return EvidenceAssessment(
        available_groups=available,
        missing_groups=missing,
        available_weight=available_weight,
        confidence_multiplier=confidence_multiplier,
        notes=notes,
    )


def _to_model_float(value: object) -> float:
    """Coerce one stored value into something the model can consume.

    Callers hand us whole database rows, which carry more than the model's
    numeric inputs: `facility_type` is a string, `updated_at` a datetime,
    `threat_corridor_present` a bool. Anything that isn't numerically
    meaningful becomes NaN ("unknown") rather than raising — a single
    unexpected column type must not take down every classification.
    """
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def apply_evidence_weighting(
    features: Dict[str, Optional[float]],
    optical_available: Optional[bool] = None,
    cloud_fraction: Optional[float] = None,
) -> tuple:
    """Normalise a feature dict for the model and assess its evidence.

    Returns (model_features, assessment). The returned dict contains
    exactly the model's FEATURE_COLUMNS — no more, no less — with missing
    values as NaN. Restricting to the declared contract matters because
    callers pass whole ORM rows: `cluster_id`, `facility_type` and
    `updated_at` are all present on a `cluster_features` row and none of
    them belong in the feature vector.
    """
    assessment = assess_evidence(features, optical_available, cloud_fraction)
    normalised = {
        column: _to_model_float(features.get(column)) for column in FEATURE_COLUMNS
    }
    return normalised, assessment
