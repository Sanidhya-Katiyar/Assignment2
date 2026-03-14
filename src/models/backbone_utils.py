"""
src/models/backbone_utils.py
─────────────────────────────
Utilities for controlling which parts of a backbone are trainable.

Two primary use-cases in the transfer learning workflow:

1. **Linear probing** – freeze all backbone weights, train only the
   newly added classification head (``freeze_backbone``).
2. **Full fine-tuning** – unfreeze every parameter so the entire network
   is updated (``unfreeze_all``).

A helper ``get_classifier_params`` is also provided so optimisers can
apply different learning rates to the head vs. the backbone.

All functions mutate the model *in-place* and also return it, enabling
chaining:

>>> model = freeze_backbone(create_model("resnet50", 30))
"""

from __future__ import annotations

from typing import Iterator

import torch.nn as nn


# ---------------------------------------------------------------------------
# Architecture-to-classifier attribute mapping
# ---------------------------------------------------------------------------
# Each entry maps the model_name key (as returned by model_factory) to the
# dotted attribute path of its classification head.
_CLASSIFIER_ATTR: dict[str, str] = {
    "resnet50":        "fc",
    "efficientnet_b0": "classifier",
    "convnext_tiny":   "classifier",
}


def _get_classifier(model: nn.Module) -> nn.Module | None:
    """
    Return the classification head sub-module, or *None* if undetectable.

    The function first tries to read the ``model_name`` attribute that
    ``create_model`` stamps onto the module; failing that it falls back to
    common attribute names used by torchvision architectures.
    """
    # Prefer explicit model_name stamp set by create_model
    name = getattr(model, "model_name", None)
    if name and name in _CLASSIFIER_ATTR:
        attr = _CLASSIFIER_ATTR[name]
        return _getattr_nested(model, attr)

    # Fallback heuristic – try common head names
    for attr in ("fc", "classifier", "head"):
        if hasattr(model, attr):
            return getattr(model, attr)

    return None


def _getattr_nested(obj: object, dotted: str) -> object:
    """Resolve a dotted attribute path, e.g. 'classifier.1'."""
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    Freeze all backbone parameters; keep only the classifier head trainable.

    This is the setup for **linear probing**: the backbone acts as a fixed
    feature extractor and only the final linear layer is optimised.

    Args:
        model: Any ``nn.Module`` created by :func:`~src.models.model_factory.create_model`.

    Returns:
        The same model with frozen backbone weights (mutated in-place).

    Example:
        >>> model = create_model("resnet50", num_classes=30)
        >>> freeze_backbone(model)
        >>> trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    """
    # 1. Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # 2. Unfreeze the classification head
    classifier = _get_classifier(model)
    if classifier is None:
        raise RuntimeError(
            "freeze_backbone: could not locate the classification head. "
            "Set model.model_name to one of the supported architectures, "
            "or manually unfreeze the head after calling this function."
        )

    for param in classifier.parameters():
        param.requires_grad = True

    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """
    Unfreeze every parameter in the model for full fine-tuning.

    Args:
        model: Any ``nn.Module``.

    Returns:
        The same model with all parameters set to ``requires_grad=True``
        (mutated in-place).

    Example:
        >>> unfreeze_all(model)
        >>> trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    """
    for param in model.parameters():
        param.requires_grad = True
    return model


def get_classifier_params(model: nn.Module) -> list[nn.Parameter]:
    """
    Return the parameter list of the classification head only.

    Useful when constructing an optimiser with per-group learning rates,
    e.g. a higher LR for the head and a lower LR for the backbone:

        >>> optimizer = torch.optim.Adam([
        ...     {"params": get_backbone_params(model),    "lr": 1e-4},
        ...     {"params": get_classifier_params(model),  "lr": 1e-3},
        ... ])

    Args:
        model: Any ``nn.Module`` created by
               :func:`~src.models.model_factory.create_model`.

    Returns:
        List of ``nn.Parameter`` objects belonging to the classifier head.

    Raises:
        RuntimeError: If the classifier head cannot be located.
    """
    classifier = _get_classifier(model)
    if classifier is None:
        raise RuntimeError(
            "get_classifier_params: could not locate the classification head."
        )
    return list(classifier.parameters())


def get_backbone_params(model: nn.Module) -> list[nn.Parameter]:
    """
    Return parameters belonging to the backbone (i.e. everything except
    the classification head).

    Args:
        model: Any ``nn.Module`` created by
               :func:`~src.models.model_factory.create_model`.

    Returns:
        List of ``nn.Parameter`` objects NOT belonging to the classifier head.
    """
    classifier = _get_classifier(model)
    classifier_ids: set[int] = set()
    if classifier is not None:
        classifier_ids = {id(p) for p in classifier.parameters()}

    return [p for p in model.parameters() if id(p) not in classifier_ids]


def trainable_summary(model: nn.Module) -> str:
    """
    Return a one-line string showing total vs. trainable parameter counts.

    Args:
        model: Any ``nn.Module``.

    Returns:
        Formatted string, e.g.
        ``"Parameters: 25,557,032 total | 2,048,030 trainable (8.01%)"``
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct       = 100.0 * trainable / total if total > 0 else 0.0
    return (
        f"Parameters: {total:,} total | "
        f"{trainable:,} trainable ({pct:.2f}%)"
    )
