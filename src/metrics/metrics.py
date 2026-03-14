"""
src/metrics/metrics.py
───────────────────────
All loss, accuracy, and gradient-norm computation lives here so that
the training engine has no metric logic scattered through it.

Public API
----------
``accuracy``                 – top-1 and top-k accuracy from logits.
``AverageMeter``             – running mean tracker.
``ConfusionMatrixAccumulator`` – accumulate and retrieve confusion matrix.
``compute_gradient_norm``    – L2 norm of all active gradients.
``make_optimizer``           – factory for AdamW / SGD with param groups.
``make_scheduler``           – factory for cosine annealing / StepLR.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


# ---------------------------------------------------------------------------
# Running statistics
# ---------------------------------------------------------------------------

class AverageMeter:
    """
    Compute and store a running mean and sum of a scalar quantity.

    Typical usage::

        meter = AverageMeter("loss")
        for batch in loader:
            loss = criterion(...)
            meter.update(loss.item(), n=batch_size)
        print(meter.avg)
    """

    def __init__(self, name: str = "") -> None:
        self.name  = name
        self.reset()

    def reset(self) -> None:
        self.val   = 0.0
        self.sum   = 0.0
        self.count = 0
        self.avg   = 0.0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update with *val* observed *n* times.

        Args:
            val: Observed value (e.g. per-sample mean loss).
            n:   Number of samples the value represents.
        """
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter({self.name}: avg={self.avg:.4f}, n={self.count})"


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def accuracy(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    top_k:   Tuple[int, ...] = (1,),
) -> List[float]:
    """
    Compute top-k accuracy from un-normalised logits.

    Args:
        logits:  Float tensor of shape ``(N, C)``.
        targets: Long tensor of shape ``(N,)`` with class indices.
        top_k:   Which top-k values to compute (default: ``(1,)``).

    Returns:
        List of accuracy values (0–100) for each k in *top_k*.
    """
    with torch.no_grad():
        max_k    = max(top_k)
        n        = targets.size(0)
        _, pred  = logits.topk(max_k, dim=1, largest=True, sorted=True)
        pred     = pred.t()                                    # (max_k, N)
        correct  = pred.eq(targets.view(1, -1).expand_as(pred))

        results: List[float] = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append(float(correct_k.mul(100.0 / n).item()))
        return results


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

class ConfusionMatrixAccumulator:
    """
    Accumulate predictions and targets, then return a confusion matrix.

    Args:
        num_classes: Number of distinct classes.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self._matrix     = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update the matrix with a batch of predictions.

        Args:
            logits:  Float tensor ``(N, C)`` — raw model output.
            targets: Long tensor ``(N,)``    — ground-truth labels.
        """
        preds = logits.argmax(dim=1).cpu().numpy()
        tgts  = targets.cpu().numpy()
        for t, p in zip(tgts, preds):
            self._matrix[t, p] += 1

    def compute(self) -> np.ndarray:
        """Return the accumulated confusion matrix as a NumPy array."""
        return self._matrix.copy()

    def reset(self) -> None:
        self._matrix[:] = 0


# ---------------------------------------------------------------------------
# Gradient norm
# ---------------------------------------------------------------------------

def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute the total L2 norm of all gradients that exist on *model*.

    Call this *after* ``loss.backward()`` and *before* ``optimizer.step()``
    (or use :func:`src.train.amp_utils.optimizer_step` which does this
    internally).

    Args:
        model: Any ``nn.Module``.

    Returns:
        Total gradient L2 norm as a Python float, or ``0.0`` if no gradients
        are present.
    """
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().data.norm(2).item() ** 2
    return float(total_sq ** 0.5)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def make_optimizer(
    params,
    optimizer_name: str   = "adamw",
    lr:             float = 1e-3,
    weight_decay:   float = 1e-4,
    momentum:       float = 0.9,
) -> torch.optim.Optimizer:
    """
    Create an optimizer.

    Supports ``"adamw"`` and ``"sgd"``.  The *params* argument accepts
    anything accepted by PyTorch optimizers: a list of tensors, a list of
    param-group dicts (for differential learning rates), or an iterator.

    Args:
        params:         Parameters or param groups to optimise.
        optimizer_name: ``"adamw"`` (default) or ``"sgd"``.
        lr:             Base learning rate.
        weight_decay:   L2 regularisation coefficient.
        momentum:       SGD momentum (ignored for AdamW).

    Returns:
        Configured ``torch.optim.Optimizer``.

    Raises:
        ValueError: If *optimizer_name* is not supported.
    """
    name = optimizer_name.lower().strip()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(
        f"Unknown optimizer '{optimizer_name}'. Supported: 'adamw', 'sgd'."
    )


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def make_scheduler(
    optimizer:      torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    epochs:         int = 30,
    step_size:      int = 10,
    gamma:          float = 0.1,
    eta_min:        float = 1e-6,
):
    """
    Create a learning-rate scheduler.

    Supports ``"cosine"`` (``CosineAnnealingLR``) and ``"step"``
    (``StepLR``).

    Args:
        optimizer:      The optimizer to schedule.
        scheduler_name: ``"cosine"`` (default) or ``"step"``.
        epochs:         Total epochs (used as ``T_max`` for cosine).
        step_size:      Step size in epochs for StepLR.
        gamma:          Decay factor for StepLR.
        eta_min:        Minimum LR for cosine annealing.

    Returns:
        A PyTorch LR scheduler.

    Raises:
        ValueError: If *scheduler_name* is not supported.
    """
    name = scheduler_name.lower().strip()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    if name == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(
        f"Unknown scheduler '{scheduler_name}'. Supported: 'cosine', 'step'."
    )
