"""EfficientNet-B0 for Sentinel-2 patch classification.

The architecture doc specifies EfficientNet-B0 with ImageNet weights,
fine-tuned on labelled 4+-band Sentinel-2 patches to a 5-class output.

Two adaptations the stock torchvision model needs:

  1. **More than three input channels.** ImageNet weights expect RGB, but
     the informative bands here include NIR and SWIR — the SWIR bands are
     precisely what separates smoke from cloud and burnt from unburnt.
     `adapt_first_conv` widens the stem convolution and seeds the new
     channels from the mean of the pretrained RGB weights, so the extra
     bands start from a sensible prior instead of noise.
  2. **A 5-class head**, replacing the 1000-class ImageNet classifier.

torch is imported lazily everywhere. Without it the rest of M4 — Dozier,
indices, smoke direction, the fetcher — still works, and M2 simply runs
tabular-only, which its evidence-weighting engine already handles.
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return True
    except ImportError:
        return False


def adapt_first_conv(model, in_channels: int):
    """Widen EfficientNet's stem convolution to accept `in_channels` bands.

    New channels are initialised with the mean of the pretrained RGB
    kernels and the whole layer rescaled by 3/in_channels, which keeps
    the output activation magnitude roughly where the pretrained
    downstream layers expect it. Initialising them to zero would waste
    the extra bands; random init would destroy the pretrained signal.
    """
    import torch
    import torch.nn as nn

    stem = model.features[0][0]
    if stem.in_channels == in_channels:
        return model

    new_conv = nn.Conv2d(
        in_channels,
        stem.out_channels,
        kernel_size=stem.kernel_size,
        stride=stem.stride,
        padding=stem.padding,
        bias=stem.bias is not None,
    )

    with torch.no_grad():
        pretrained = stem.weight  # (out, 3, kh, kw)
        mean_kernel = pretrained.mean(dim=1, keepdim=True)
        if in_channels <= pretrained.shape[1]:
            new_conv.weight.copy_(pretrained[:, :in_channels])
        else:
            extra = in_channels - pretrained.shape[1]
            new_conv.weight.copy_(
                torch.cat([pretrained, mean_kernel.repeat(1, extra, 1, 1)], dim=1)
            )
        new_conv.weight.mul_(pretrained.shape[1] / in_channels)
        if stem.bias is not None:
            new_conv.bias.copy_(stem.bias)

    model.features[0][0] = new_conv
    return model


def build_model(n_classes: int, in_channels: int = 6, pretrained: bool = True):
    """EfficientNet-B0 adapted to multispectral input and a 5-class head."""
    if not torch_available():
        raise ImportError(
            "torch and torchvision are required for the image classifier. "
            "Install with: pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cpu"
        )

    import torch.nn as nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = None
    if pretrained:
        try:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        except Exception:
            logger.warning("Could not resolve pretrained weights — training from scratch")

    model = efficientnet_b0(weights=weights)
    model = adapt_first_conv(model, in_channels)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, n_classes)
    return model


def normalise_stack(stack, image_size_px: int = 224):
    """Resize and standardise a (C, H, W) reflectance stack for the network.

    Reflectance is standardised per-band rather than with ImageNet's RGB
    statistics, which do not describe surface reflectance at all. Bands
    with no variance (a constant patch) are left centred rather than
    divided by ~zero.
    """
    import torch
    import torch.nn.functional as F

    tensor = torch.as_tensor(stack, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    tensor = F.interpolate(
        tensor, size=(image_size_px, image_size_px), mode="bilinear", align_corners=False
    )

    mean = tensor.mean(dim=(2, 3), keepdim=True)
    std = tensor.std(dim=(2, 3), keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (tensor - mean) / std


def predict_probabilities(model, stack, classes: List[str], image_size_px: int = 224) -> dict:
    """Run one patch through the model and return {class: probability}."""
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(normalise_stack(stack, image_size_px))
        probabilities = torch.softmax(logits, dim=1)[0]

    return {cls: float(probabilities[i]) for i, cls in enumerate(classes)}
