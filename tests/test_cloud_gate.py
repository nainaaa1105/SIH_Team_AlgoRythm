from app.config import Settings
from app.ingestion.cloud_gate import confidence_decay, is_optically_available


def test_is_optically_available_below_threshold():
    settings = Settings(cloud_fraction_threshold=0.30)
    assert is_optically_available(0.10, settings) is True
    assert is_optically_available(0.30, settings) is True  # inclusive boundary
    assert is_optically_available(0.31, settings) is False


def test_is_optically_available_unknown_cloud_state_treated_as_unavailable():
    settings = Settings()
    assert is_optically_available(None, settings) is False


def test_confidence_decay_no_cloud_keeps_full_confidence():
    assert confidence_decay(0.8, 0.0) == 0.8


def test_confidence_decay_full_cloud_halves_confidence():
    assert confidence_decay(0.8, 1.0) == 0.4


def test_confidence_decay_partial_cloud_is_between():
    result = confidence_decay(1.0, 0.5)
    assert 0.5 < result < 1.0
    assert result == 0.75


def test_confidence_decay_unknown_cloud_state_applies_mild_penalty():
    result = confidence_decay(1.0, None)
    assert result == 0.85


def test_confidence_decay_never_goes_negative():
    assert confidence_decay(0.0, 1.0) == 0.0
