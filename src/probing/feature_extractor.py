"""
src/probing/feature_extractor.py
─────────────────────────────────
Extract intermediate activations from any named layer of a trained
backbone using PyTorch forward hooks.

Design goals
------------
* Zero model surgery — hooks attach and detach cleanly.
* Works for any layer whose output is a Tensor (including spatial maps,
  which are globally-average-pooled and flattened to 1-D vectors).
* Architecture-agnostic: the caller passes a *resolved* ``nn.Module``
  target layer, not a string path.

Public API
----------
``LAYER_REGISTRY``          – dict mapping model_name → {depth_tag: dotted_attr}.
``resolve_layer``           – walk dotted attribute path to retrieve an nn.Module.
``extract_layer_features``  – hook-based feature extraction over a DataLoader.
``extract_all_layers``      – extract features for every depth tag of one model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Layer registry
# ---------------------------------------------------------------------------

# Maps model_name → {depth_tag → dotted attribute path on the nn.Module}
# Attribute paths use dot-notation; integer indices become digit strings.
LAYER_REGISTRY: Dict[str, Dict[str, str]] = {
    "resnet50": {
        "early":  "layer1",
        "middle": "layer3",
        "final":  "layer4",
    },
    "efficientnet_b0": {
        "early":  "features.2",
        "middle": "features.5",
        "final":  "features.7",
    },
    "convnext_tiny": {
        "early":  "features.1",   # torchvision ConvNeXt: features is a Sequential
        "middle": "features.3",   # of (Downsample, Stage) pairs at indices 1,3,5,7
        "final":  "features.5",
    },
}

# Human-readable display names used in plot labels / CSV
DEPTH_ORDER = ["early", "middle", "final"]


# ---------------------------------------------------------------------------
# Attribute resolution
# ---------------------------------------------------------------------------

def resolve_layer(model: nn.Module, dotted_path: str) -> nn.Module:
    """
    Walk a dot-separated attribute path on *model* and return the sub-module.

    Supports integer indices for ``nn.Sequential`` children
    (e.g. ``"features.2"`` resolves ``model.features[2]``).

    Args:
        model:       Root ``nn.Module``.
        dotted_path: Dot-separated path, e.g. ``"layer1"`` or ``"features.5"``.

    Returns:
        The resolved ``nn.Module`` sub-layer.

    Raises:
        AttributeError: If any segment of the path cannot be resolved.

    Example::

        layer = resolve_layer(model, "features.2")
    """
    obj: object = model
    for part in dotted_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]  # type: ignore[index]
        else:
            obj = getattr(obj, part)
    return obj  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Single-layer extraction
# ---------------------------------------------------------------------------

def extract_layer_features(
    model:      nn.Module,
    dataloader: DataLoader,
    layer:      nn.Module,
    device:     torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract activations from *layer* for every sample in *dataloader*.

    A forward hook captures the output tensor after *layer* completes.
    Spatial feature maps ``(N, C, H, W)`` are reduced via global-average
    pooling to ``(N, C)`` before being stacked, so the returned feature
    matrix always has shape ``(total_samples, feature_dim)``.

    Args:
        model:      Trained ``nn.Module`` (backbone + head).  Weights must
                    already be loaded.  The model is run in ``eval()`` mode.
        dataloader: DataLoader for the probe dataset (usually the
                    visualization subset).
        layer:      The specific ``nn.Module`` sub-layer to hook.
        device:     Compute device to use.

    Returns:
        Tuple ``(features, labels)`` where

        * ``features`` – float32 NumPy array ``(N, D)``.
        * ``labels``   – int64 NumPy array  ``(N,)``.

    Raises:
        RuntimeError: If the hook never fires (layer not in forward graph).
    """
    model.eval()
    model.to(device)

    captured: List[torch.Tensor] = []

    def _hook(_module, _inp, output: torch.Tensor) -> None:
        # output may be a Tensor or a tuple (some torchvision blocks wrap output)
        if isinstance(output, (tuple, list)):
            output = output[0]
        # Global-average-pool spatial dimensions if present
        if output.dim() == 4:               # (N, C, H, W)
            output = output.mean(dim=(2, 3))
        elif output.dim() == 3:             # (N, L, C) — transformer tokens
            output = output.mean(dim=1)
        captured.append(output.detach().cpu().float())

    handle = layer.register_forward_hook(_hook)

    all_labels: List[torch.Tensor] = []

    try:
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                model(inputs)              # forward pass fires the hook
                all_labels.append(targets.cpu())
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(
            "Forward hook never fired.  The supplied layer may not be part "
            "of the model's forward graph for the given input shape."
        )

    features = torch.cat(captured, dim=0).numpy().astype(np.float32)
    labels   = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    return features, labels


# ---------------------------------------------------------------------------
# Convenience: extract all registered depth tags for one model
# ---------------------------------------------------------------------------

def extract_all_layers(
    model:      nn.Module,
    model_name: str,
    dataloader: DataLoader,
    device:     torch.device,
    depth_tags: Optional[List[str]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract features from every registered depth tag for *model_name*.

    Args:
        model:      Trained backbone (weights already loaded).
        model_name: Key into :data:`LAYER_REGISTRY`
                    (e.g. ``"resnet50"``).
        dataloader: DataLoader for the probe dataset.
        device:     Compute device.
        depth_tags: Optional subset of depth tags to extract
                    (default: all tags in LAYER_REGISTRY for *model_name*).

    Returns:
        Dict mapping depth tag → ``(features, labels)`` tuple.

    Raises:
        KeyError: If *model_name* is not in :data:`LAYER_REGISTRY`.
    """
    if model_name not in LAYER_REGISTRY:
        raise KeyError(
            f"'{model_name}' not in LAYER_REGISTRY.  "
            f"Supported: {sorted(LAYER_REGISTRY.keys())}."
        )

    registry   = LAYER_REGISTRY[model_name]
    tags       = depth_tags if depth_tags is not None else DEPTH_ORDER
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for tag in tags:
        if tag not in registry:
            print(f"  [warn] depth tag '{tag}' not registered for {model_name}, skipping.")
            continue

        dotted_path = registry[tag]
        try:
            layer = resolve_layer(model, dotted_path)
        except (AttributeError, IndexError, KeyError) as exc:
            print(f"  [warn] could not resolve '{dotted_path}' on {model_name}: {exc}")
            continue

        print(f"  Extracting {model_name}/{tag}  (layer: {dotted_path}) …", end="", flush=True)
        feats, labels = extract_layer_features(model, dataloader, layer, device)
        print(f"  shape={feats.shape}")
        results[tag] = (feats, labels)

    return results
