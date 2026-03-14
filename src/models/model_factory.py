"""
src/models/model_factory.py
───────────────────────────
Central factory for creating pretrained backbone models with a custom
classification head for the AID remote sensing dataset.

All three architectures (ResNet-50, EfficientNet-B0, ConvNeXt-Tiny) are
loaded with ImageNet-1k pretrained weights, their original classifiers are
replaced with a linear layer for ``num_classes`` outputs, and the resulting
``nn.Module`` is returned.

Supported model names
---------------------
``"resnet50"``         – torchvision ResNet-50
``"efficientnet_b0"``  – torchvision EfficientNet-B0
``"convnext_tiny"``    – torchvision ConvNeXt-Tiny

Usage
-----
>>> from src.models.model_factory import create_model
>>> model = create_model("resnet50", num_classes=30, pretrained=True)
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
)

# ---------------------------------------------------------------------------
# Registry: maps name → (loader_fn, weights_enum)
# ---------------------------------------------------------------------------
# Using the new torchvision v0.13+ weights API so deprecation warnings are
# avoided and weight provenance is explicit.
_MODEL_REGISTRY: dict = {
    "resnet50": {
        "loader":  models.resnet50,
        "weights": ResNet50_Weights.IMAGENET1K_V2,
    },
    "efficientnet_b0": {
        "loader":  models.efficientnet_b0,
        "weights": EfficientNet_B0_Weights.IMAGENET1K_V1,
    },
    "convnext_tiny": {
        "loader":  models.convnext_tiny,
        "weights": ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
    },
}

SUPPORTED_MODELS = sorted(_MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Head-replacement helpers (one per architecture family)
# ---------------------------------------------------------------------------

def _replace_resnet_head(backbone: nn.Module, num_classes: int) -> nn.Module:
    """Replace the ``fc`` layer on a ResNet model."""
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)
    return backbone


def _replace_efficientnet_head(backbone: nn.Module, num_classes: int) -> nn.Module:
    """Replace the final ``Linear`` inside EfficientNet's ``classifier``."""
    # torchvision EfficientNet: classifier = Sequential(Dropout, Linear)
    in_features = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Linear(in_features, num_classes)
    return backbone


def _replace_convnext_head(backbone: nn.Module, num_classes: int) -> nn.Module:
    """Replace the final ``Linear`` inside ConvNeXt's ``classifier``."""
    # torchvision ConvNeXt: classifier = Sequential(LayerNorm2d, Flatten, Linear)
    in_features = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Linear(in_features, num_classes)
    return backbone


_HEAD_REPLACERS = {
    "resnet50":        _replace_resnet_head,
    "efficientnet_b0": _replace_efficientnet_head,
    "convnext_tiny":   _replace_convnext_head,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_model(
    model_name:  str,
    num_classes: int  = 30,
    pretrained:  bool = True,
) -> nn.Module:
    """
    Instantiate a pretrained backbone with a custom classification head.

    Args:
        model_name:  One of ``"resnet50"``, ``"efficientnet_b0"``,
                     ``"convnext_tiny"``.
        num_classes: Number of output classes (default: 30 for AID).
        pretrained:  If ``True``, load ImageNet-1k pretrained weights.
                     If ``False``, weights are randomly initialised.

    Returns:
        ``nn.Module`` ready for training or feature extraction.

    Raises:
        ValueError: If *model_name* is not in the supported registry.

    Example:
        >>> model = create_model("efficientnet_b0", num_classes=30)
        >>> model.eval()
    """
    model_name = model_name.lower().strip()

    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported models: {SUPPORTED_MODELS}. "
            "Check the 'model_name' field in your config."
        )

    entry   = _MODEL_REGISTRY[model_name]
    loader  = entry["loader"]
    weights = entry["weights"] if pretrained else None

    backbone = loader(weights=weights)
    backbone = _HEAD_REPLACERS[model_name](backbone, num_classes)

    return backbone


def list_models() -> list[str]:
    """Return the list of supported model name strings."""
    return SUPPORTED_MODELS
