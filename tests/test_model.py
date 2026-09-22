"""EfficientNet-B0 adaptation, registry and inference.

Runs against real torch when it is installed and skips cleanly when it
is not — the same degradation the production code relies on.
"""
import numpy as np
import pytest

from imagery.model.registry import (
    BandOrderMismatch,
    ImageRegistryError,
    load_metadata,
    new_metadata,
    validate_band_order,
)

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("torchvision", reason="torchvision not installed")

from imagery.model.efficientnet import (  # noqa: E402
    adapt_first_conv,
    build_model,
    normalise_stack,
    predict_probabilities,
    torch_available,
)

CLASSES = ["industrial_fire", "gas_flare", "wildfire", "agricultural_burning", "mining"]
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]


# --- multispectral stem adaptation --------------------------------------

def test_torch_is_detected():
    assert torch_available()


def test_stem_is_widened_to_the_requested_channel_count():
    from torchvision.models import efficientnet_b0

    model = adapt_first_conv(efficientnet_b0(weights=None), 6)
    assert model.features[0][0].in_channels == 6


def test_widening_preserves_the_pretrained_rgb_kernels():
    """Re-initialising the stem randomly would throw away the ImageNet
    weights that make fine-tuning on a small dataset work at all."""
    from torchvision.models import efficientnet_b0

    model = efficientnet_b0(weights=None)
    original = model.features[0][0].weight.clone()
    adapted = adapt_first_conv(model, 6)

    # Scaled by 3/6 to keep activation magnitude where downstream layers expect it.
    assert torch.allclose(adapted.features[0][0].weight[:, :3], original * 0.5, atol=1e-6)


def test_extra_channels_are_seeded_from_the_rgb_mean():
    """Zero-init would waste the extra bands; random init would inject
    noise into a pretrained network."""
    from torchvision.models import efficientnet_b0

    model = efficientnet_b0(weights=None)
    original = model.features[0][0].weight.clone()
    adapted = adapt_first_conv(model, 6)

    expected = original.mean(dim=1, keepdim=True) * 0.5
    assert torch.allclose(adapted.features[0][0].weight[:, 3:4], expected, atol=1e-6)


def test_adapting_to_the_same_channel_count_is_a_no_op():
    from torchvision.models import efficientnet_b0

    model = efficientnet_b0(weights=None)
    original = model.features[0][0].weight.clone()
    adapted = adapt_first_conv(model, 3)
    assert torch.allclose(adapted.features[0][0].weight, original)


# --- model construction and inference -----------------------------------

def test_model_has_the_requested_number_of_classes():
    model = build_model(n_classes=5, in_channels=6, pretrained=False)
    assert model.classifier[1].out_features == 5


def test_normalise_stack_resizes_to_the_network_input():
    stack = np.random.rand(6, 40, 55) * 0.3
    assert tuple(normalise_stack(stack, 224).shape) == (1, 6, 224, 224)


def test_normalise_stack_standardises_per_band():
    """Reflectance statistics, not ImageNet RGB statistics — the latter
    do not describe surface reflectance at all."""
    stack = np.random.rand(6, 32, 32) * 0.3 + 0.1
    normalised = normalise_stack(stack, 64)
    assert abs(float(normalised.mean())) < 0.1


def test_normalise_stack_survives_a_constant_band():
    """A uniform patch has zero variance; dividing by it would give NaN."""
    stack = np.ones((6, 16, 16)) * 0.25
    assert torch.isfinite(normalise_stack(stack, 64)).all()


def test_predicted_probabilities_form_a_distribution():
    model = build_model(n_classes=5, in_channels=6, pretrained=False)
    probabilities = predict_probabilities(model, np.random.rand(6, 64, 64) * 0.3, CLASSES)

    assert set(probabilities) == set(CLASSES)
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= p <= 1.0 for p in probabilities.values())


def test_model_accepts_a_three_band_stack_too():
    model = build_model(n_classes=3, in_channels=3, pretrained=False)
    probabilities = predict_probabilities(
        model, np.random.rand(3, 64, 64) * 0.3, ["wildfire", "mining", "gas_flare"]
    )
    assert len(probabilities) == 3


# --- registry -----------------------------------------------------------

def test_band_order_contract_accepts_a_match():
    metadata = new_metadata("v1", BANDS, CLASSES, 224)
    validate_band_order(metadata, BANDS)


def test_reordered_bands_are_rejected():
    """A scrambled band order feeds the network a wrong tensor and it
    returns confident nonsense — exactly the silent failure M2's
    feature-order check exists to prevent."""
    metadata = new_metadata("v1", BANDS, CLASSES, 224)
    with pytest.raises(BandOrderMismatch, match="DIFFERENT"):
        validate_band_order(metadata, list(reversed(BANDS)))


def test_missing_and_extra_bands_are_named_in_the_error():
    metadata = new_metadata("v1", BANDS, CLASSES, 224)
    wrong = [b for b in BANDS if b != "B11"] + ["B99"]
    with pytest.raises(BandOrderMismatch) as exc:
        validate_band_order(metadata, wrong)
    assert "B11" in str(exc.value) and "B99" in str(exc.value)


def test_loading_a_missing_model_gives_an_actionable_error(tmp_path):
    with pytest.raises(ImageRegistryError, match="train one first"):
        load_metadata(str(tmp_path), "nonexistent")


def test_model_round_trips_through_the_registry(tmp_path):
    from imagery.model.registry import load_state_dict, save_model

    model = build_model(n_classes=5, in_channels=6, pretrained=False)
    metadata = new_metadata("rt", BANDS, CLASSES, 224)
    save_model(model.state_dict(), metadata, str(tmp_path))

    reloaded = build_model(n_classes=5, in_channels=6, pretrained=False)
    reloaded.load_state_dict(load_state_dict(str(tmp_path), "rt"))

    stack = np.random.rand(6, 64, 64) * 0.3
    original_probs = predict_probabilities(model, stack, CLASSES)
    reloaded_probs = predict_probabilities(reloaded, stack, CLASSES)

    for cls in CLASSES:
        assert original_probs[cls] == pytest.approx(reloaded_probs[cls], abs=1e-6)


def test_latest_pointer_is_written(tmp_path):
    from imagery.model.registry import save_model

    model = build_model(n_classes=5, in_channels=6, pretrained=False)
    save_model(model.state_dict(), new_metadata("v7", BANDS, CLASSES, 224), str(tmp_path))
    assert load_metadata(str(tmp_path), "latest").version == "v7"


# --- label encoding -----------------------------------------------------

def test_label_encoding_is_contiguous_for_a_class_subset():
    """Same constraint M2 hit: non-contiguous indices break the loss."""
    from imagery.model.train import encode_labels

    encoded, present = encode_labels(
        ["wildfire", "mining", "wildfire"], CLASSES
    )
    assert sorted(set(encoded.tolist())) == [0, 1]
    assert present == ["wildfire", "mining"]   # canonical order, not alphabetical


def test_label_encoding_rejects_an_unknown_class():
    from imagery.model.train import encode_labels

    with pytest.raises(ValueError, match="Unknown label"):
        encode_labels(["wildfire", "volcano"], CLASSES)


# --- prediction gating --------------------------------------------------

def test_low_confidence_prediction_is_withheld_from_fusion(tmp_path):
    """M2 fuses at 30% weight, so a near-uniform image guess would still
    move the final answer. Better to send nothing and let M2 run
    tabular-only, which its evidence engine already explains."""
    from imagery.model.predict import classify_patch
    from imagery.model.registry import save_model

    model = build_model(n_classes=5, in_channels=6, pretrained=False)
    metadata = new_metadata("gate", BANDS, CLASSES, 64)
    save_model(model.state_dict(), metadata, str(tmp_path))

    bands = {b: np.random.rand(32, 32) * 0.3 for b in BANDS}
    result = classify_patch(bands, model_dir=str(tmp_path), version="gate")

    assert result is not None
    # An untrained network on random input is near-uniform (~0.2 for five
    # classes), which is below the 0.35 fusion threshold.
    assert result.usable_for_fusion == (result.confidence >= 0.35)
    if not result.usable_for_fusion:
        assert "below" in result.reason


def test_classify_patch_returns_none_without_a_trained_model(tmp_path):
    from imagery.model.predict import classify_patch

    bands = {b: np.random.rand(16, 16) * 0.3 for b in BANDS}
    assert classify_patch(bands, model_dir=str(tmp_path), version="absent") is None
