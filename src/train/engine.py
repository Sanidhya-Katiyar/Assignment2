"""
src/train/engine.py
────────────────────
Low-level, stateless training and validation step functions.

The engine layer contains *no* state; all state lives in the ``Trainer``
class (``trainer.py``).  Each function returns plain Python dicts so the
caller can log, accumulate, or inspect results freely.

Public API
----------
``train_one_epoch``   – run one full pass over the training DataLoader.
``validate_one_epoch`` – run one full pass over the validation DataLoader.
``extract_features``  – forward-pass a DataLoader and collect feature vectors.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.metrics.metrics import (
    AverageMeter,
    ConfusionMatrixAccumulator,
    accuracy,
    compute_gradient_norm,
)
from src.train.amp_utils import autocast_ctx, optimizer_step, scaled_backward


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    optimizer:   torch.optim.Optimizer,
    scaler,                                    # GradScaler (may be no-op)
    device:      torch.device,
    amp_enabled: bool = False,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """
    Execute one full training epoch.

    Args:
        model:         The network in ``train()`` mode.
        loader:        Training DataLoader.
        criterion:     Loss function (e.g. ``nn.CrossEntropyLoss()``).
        optimizer:     Optimizer (already scoped to the right parameters).
        scaler:        ``GradScaler`` for AMP; pass a disabled scaler for CPU.
        device:        Target compute device.
        amp_enabled:   Whether to use ``torch.autocast``.
        max_grad_norm: Gradient clipping threshold (L2 norm).

    Returns:
        Dict with keys:
        ``"loss"`` (float), ``"acc1"`` (float 0-100), ``"grad_norm"`` (float),
        ``"epoch_time"`` (seconds), ``"gpu_mem_max_mb"`` (float or 0).
    """
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter  = AverageMeter("acc1")
    grad_norms: List[float] = []
    t0         = time.time()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for inputs, targets in loader:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs      = inputs.size(0)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(amp_enabled, device_type=device.type):
            logits = model(inputs)
            loss   = criterion(logits, targets)

        scaled_backward(scaler, loss, optimizer)
        gn = optimizer_step(scaler, optimizer, model, max_norm=max_grad_norm)
        grad_norms.append(gn)

        loss_meter.update(loss.item(), bs)
        top1 = accuracy(logits.detach(), targets, top_k=(1,))[0]
        acc_meter.update(top1, bs)

    gpu_mem = 0.0
    if device.type == "cuda":
        gpu_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2  # MiB

    return {
        "loss":          loss_meter.avg,
        "acc1":          acc_meter.avg,
        "grad_norm":     float(sum(grad_norms) / max(len(grad_norms), 1)),
        "epoch_time":    time.time() - t0,
        "gpu_mem_max_mb": gpu_mem,
    }


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

def validate_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
    num_classes: int,
    amp_enabled: bool = False,
) -> Dict[str, object]:
    """
    Execute one full validation pass.

    Args:
        model:       The network; will be temporarily put in ``eval()`` mode.
        loader:      Validation (or test) DataLoader.
        criterion:   Loss function.
        device:      Target compute device.
        num_classes: Number of classes for the confusion matrix.
        amp_enabled: Whether to use ``torch.autocast``.

    Returns:
        Dict with keys:
        ``"loss"`` (float), ``"acc1"`` (float 0-100),
        ``"confusion_matrix"`` (numpy array ``(C, C)``).
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    acc_meter  = AverageMeter("val_acc1")
    cm_acc     = ConfusionMatrixAccumulator(num_classes)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            bs      = inputs.size(0)

            with autocast_ctx(amp_enabled, device_type=device.type):
                logits = model(inputs)
                loss   = criterion(logits, targets)

            loss_meter.update(loss.item(), bs)
            top1 = accuracy(logits, targets, top_k=(1,))[0]
            acc_meter.update(top1, bs)
            cm_acc.update(logits, targets)

    return {
        "loss":             loss_meter.avg,
        "acc1":             acc_meter.avg,
        "confusion_matrix": cm_acc.compute(),
    }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    model:         nn.Module,
    loader:        DataLoader,
    device:        torch.device,
    feature_hook:  Optional[Callable] = None,
    amp_enabled:   bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward-pass *loader* and collect feature vectors + labels.

    If *feature_hook* is provided it is called on each batch's hidden
    representation (register it as a forward hook on the desired layer
    before calling this function).  Otherwise, the penultimate layer
    output is approximated by removing the final linear head and using
    the model's second-to-last output.

    For the common use case in this project (ResNet, EfficientNet,
    ConvNeXt) callers replace the final linear with an ``nn.Identity``
    temporarily:

        >>> orig_head = model.fc
        >>> model.fc  = nn.Identity()
        >>> feats, lbls = extract_features(model, loader, device)
        >>> model.fc  = orig_head

    Args:
        model:        Model in eval mode.
        loader:       DataLoader to iterate.
        device:       Target compute device.
        feature_hook: Optional callable invoked as ``hook(batch_features)``
                      after each forward pass (for custom layer taps).
        amp_enabled:  Whether to use ``torch.autocast``.

    Returns:
        Tuple ``(features, labels)`` — CPU tensors of shape
        ``(N, D)`` and ``(N,)`` respectively.
    """
    model.eval()
    all_feats:  List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)

            with autocast_ctx(amp_enabled, device_type=device.type):
                out = model(inputs)

            if feature_hook is not None:
                feature_hook(out)

            all_feats.append(out.cpu().float())
            all_labels.append(targets.cpu())

    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)
