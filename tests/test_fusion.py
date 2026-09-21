import pytest

from classifier.features.schema import CLASSES
from classifier.model.fusion import align_to_classes, confidence_score, fuse, top_class


def _probs(**overrides):
    base = {cls: 0.0 for cls in CLASSES}
    base.update(overrides)
    return base


def test_fusion_without_imagery_gives_tabular_full_weight():
    xgb = _probs(industrial_fire=0.8, gas_flare=0.2)
    fused, weight = fuse(xgb, None)
    assert weight == 1.0
    assert fused["industrial_fire"] == pytest.approx(0.8)


def test_fusion_blends_at_seventy_thirty():
    xgb = _probs(industrial_fire=1.0)
    image = _probs(wildfire=1.0)
    fused, weight = fuse(xgb, image, weight_xgb=0.70)

    assert weight == 0.70
    assert fused["industrial_fire"] == pytest.approx(0.70)
    assert fused["wildfire"] == pytest.approx(0.30)


def test_fused_probabilities_always_sum_to_one():
    xgb = _probs(industrial_fire=0.6, mining=0.4)
    image = _probs(wildfire=0.5, gas_flare=0.5)
    fused, _ = fuse(xgb, image)
    assert sum(fused.values()) == pytest.approx(1.0)


def test_align_handles_a_partial_class_set_from_m4():
    """M4's model may emit only the classes it was trained on; missing
    ones fill with zero and the rest renormalise."""
    aligned = align_to_classes({"wildfire": 0.6, "mining": 0.4})
    assert set(aligned) == set(CLASSES)
    assert sum(aligned.values()) == pytest.approx(1.0)
    assert aligned["wildfire"] == pytest.approx(0.6)


def test_align_drops_unknown_labels_from_m4():
    aligned = align_to_classes({"wildfire": 0.5, "volcano": 0.5})
    assert "volcano" not in aligned
    assert aligned["wildfire"] == pytest.approx(1.0)


def test_align_of_unnormalised_input_renormalises():
    aligned = align_to_classes({"wildfire": 60, "mining": 40})
    assert aligned["wildfire"] == pytest.approx(0.6)


def test_align_of_all_zeros_falls_back_to_uniform_not_division_by_zero():
    aligned = align_to_classes({cls: 0.0 for cls in CLASSES})
    assert sum(aligned.values()) == pytest.approx(1.0)
    assert aligned["wildfire"] == pytest.approx(1.0 / len(CLASSES))


def test_align_of_none_stays_none():
    assert align_to_classes(None) is None


def test_fuse_rejects_a_weight_outside_zero_to_one():
    with pytest.raises(ValueError):
        fuse(_probs(mining=1.0), _probs(mining=1.0), weight_xgb=1.5)


def test_top_class_picks_the_argmax():
    assert top_class(_probs(gas_flare=0.7, mining=0.3))[0] == "gas_flare"


def test_top_class_tie_break_is_deterministic():
    tied = _probs(industrial_fire=0.5, gas_flare=0.5)
    assert top_class(tied)[0] == top_class(tied)[0] == "industrial_fire"


def test_confidence_score_is_a_percentage():
    score = confidence_score(_probs(wildfire=0.85), confidence_multiplier=1.0)
    assert score == pytest.approx(85.0)


def test_confidence_score_is_discounted_when_evidence_is_missing():
    full = confidence_score(_probs(wildfire=0.9), confidence_multiplier=1.0)
    degraded = confidence_score(_probs(wildfire=0.9), confidence_multiplier=0.6)
    assert degraded < full
    assert degraded == pytest.approx(54.0)


def test_confidence_score_stays_within_zero_and_one_hundred():
    assert 0.0 <= confidence_score(_probs(mining=1.0), 1.0) <= 100.0
    assert confidence_score(_probs(mining=0.0), 0.5) >= 0.0
