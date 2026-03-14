"""
src/train/linear_probe.py
──────────────────────────
Convenience wrapper that configures a ``Trainer`` specifically for the
**linear probing** regime:

* backbone is fully frozen
* only the classification head parameters are optimised
* feature snapshots for the visualization subset are saved after training

Public API
----------
``build_linear_probe_trainer``  – create a ready-to-use ``Trainer``.
``save_feature_snapshot``       – extract and persist features as ``.npz``.
``run_linear_probe``            – end-to-end: build, fit, snapshot, return summary.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.models.backbone_utils import freeze_backbone, get_classifier_params
from src.models.model_factory  import create_model
from src.train.engine          import extract_features
from src.train.trainer         import TrainConfig, Trainer


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_linear_probe_trainer(
    model_name:   str,
    tcfg:         TrainConfig,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    num_classes:  int = 30,
    resume_path:  Optional[str | Path] = None,
) -> Trainer:
    """
    Instantiate a model, freeze the backbone, and return a configured ``Trainer``.

    Args:
        model_name:   One of ``"resnet50"``, ``"efficientnet_b0"``,
                      ``"convnext_tiny"``.
        tcfg:         ``TrainConfig`` with at minimum ``lr``, ``epochs``,
                      ``seed``, and ``device`` set.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        num_classes:  Number of output classes (default 30 for AID).
        resume_path:  Optional path to a checkpoint to resume from.

    Returns:
        A ``Trainer`` instance ready to call ``.fit()`` on.
    """
    device = torch.device(tcfg.device)

    # ── Model ──────────────────────────────────────────────────────────
    model = create_model(model_name, num_classes=num_classes, pretrained=True)
    model.model_name = model_name           # needed by backbone_utils
    model = model.to(device)

    # ── Freeze backbone, optimise only head ────────────────────────────
    freeze_backbone(model)
    head_params = get_classifier_params(model)

    from src.metrics.metrics import make_optimizer, make_scheduler
    optimizer = make_optimizer(
        head_params,
        optimizer_name = tcfg.optimizer_name,
        lr             = tcfg.lr,
        weight_decay   = tcfg.weight_decay,
        momentum       = tcfg.momentum,
    )
    scheduler = make_scheduler(
        optimizer,
        scheduler_name = tcfg.scheduler_name,
        epochs         = tcfg.epochs,
        step_size      = tcfg.step_size,
        gamma          = tcfg.gamma,
    )

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        criterion    = criterion,
        tcfg         = dataclasses.replace(tcfg, mode="linearprobe"),
        optimizer    = optimizer,
        scheduler    = scheduler,
        resume_path  = resume_path,
    )
    return trainer


# ---------------------------------------------------------------------------
# Feature snapshot
# ---------------------------------------------------------------------------

def save_feature_snapshot(
    model:       nn.Module,
    loader:      DataLoader,
    image_paths: List[str],
    output_path: Path | str,
    device:      torch.device,
    amp_enabled: bool = False,
) -> Path:
    """
    Extract penultimate-layer features and save them as a ``.npz`` file.

    The head linear layer is temporarily replaced with ``nn.Identity`` so
    the function captures the feature vector rather than class logits.

    The ``.npz`` file contains:

    * ``"features"``  – float32 array ``(N, D)``
    * ``"labels"``    – int64 array ``(N,)``
    * ``"paths"``     – object array of image path strings ``(N,)``

    Args:
        model:       Model (any supported backbone).
        loader:      DataLoader for the snapshot images.
        image_paths: Image paths aligned with the loader's dataset order.
        output_path: Destination ``.npz`` file.
        device:      Compute device.
        amp_enabled: Whether AMP is active.

    Returns:
        Path of the written ``.npz`` file.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Swap head with Identity
    orig_head, head_attr = _get_head(model)
    setattr(model, head_attr, nn.Identity())

    try:
        feats, labels = extract_features(model, loader, device, amp_enabled=amp_enabled)
    finally:
        setattr(model, head_attr, orig_head)  # always restore

    np.savez(
        out_path,
        features = feats.numpy().astype(np.float32),
        labels   = labels.numpy().astype(np.int64),
        paths    = np.array(image_paths, dtype=object),
    )
    return out_path


def _get_head(model: nn.Module):
    """Return (head_module, attribute_name) for supported architectures."""
    for attr in ("fc", "classifier"):
        if hasattr(model, attr):
            return getattr(model, attr), attr
    raise RuntimeError(
        "save_feature_snapshot: cannot locate classification head. "
        "Expected 'model.fc' or 'model.classifier'."
    )


# ---------------------------------------------------------------------------
# End-to-end convenience runner
# ---------------------------------------------------------------------------

def run_linear_probe(
    model_name:      str,
    tcfg:            TrainConfig,
    train_loader:    DataLoader,
    val_loader:      DataLoader,
    viz_loader:      Optional[DataLoader] = None,
    viz_image_paths: Optional[List[str]]  = None,
    num_classes:     int = 30,
    resume_path:     Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Full linear probe pipeline: build → fit → snapshot → return summary.

    Args:
        model_name:      Backbone name (see :func:`~src.models.model_factory.create_model`).
        tcfg:            ``TrainConfig`` hyper-parameters.
        train_loader:    Training DataLoader.
        val_loader:      Validation DataLoader.
        viz_loader:      DataLoader for the visualization subset (optional).
        viz_image_paths: Image paths for the visualization subset (optional).
        num_classes:     Output class count.
        resume_path:     Optional checkpoint to resume from.

    Returns:
        Summary dict from ``Trainer.fit()`` augmented with
        ``"feature_snapshot_path"`` key (or ``None`` if skipped).
    """
    trainer = build_linear_probe_trainer(
        model_name   = model_name,
        tcfg         = dataclasses.replace(tcfg, model_name=model_name),
        train_loader = train_loader,
        val_loader   = val_loader,
        num_classes  = num_classes,
        resume_path  = resume_path,
    )

    summary = trainer.fit()

    # ── Feature snapshot ──────────────────────────────────────────────
    snap_path = None
    if viz_loader is not None and viz_image_paths is not None:
        snap_path = Path(tcfg.output_dir) / "features" / (
            f"{model_name}_linearprobe_{tcfg.seed}_"
            f"epoch{summary['best_epoch']}.npz"
        )
        save_feature_snapshot(
            model        = trainer.model,
            loader       = viz_loader,
            image_paths  = viz_image_paths,
            output_path  = snap_path,
            device       = torch.device(tcfg.device),
            amp_enabled  = tcfg.amp,
        )
        print(f"  Feature snapshot → {snap_path}")

    summary["feature_snapshot_path"] = str(snap_path) if snap_path else None
    return summary
