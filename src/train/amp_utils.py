"""
src/train/amp_utils.py
───────────────────────
Thin wrappers around ``torch.cuda.amp`` so that the rest of the training
code can stay identical whether AMP is enabled or not.

Design principle: every helper has a no-op CPU / non-AMP path so that
training on CPU or with ``amp=False`` requires zero conditional branches
in the caller.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# GradScaler factory
# ---------------------------------------------------------------------------

def make_scaler(enabled: bool) -> torch.cuda.amp.GradScaler:
    """
    Return a ``GradScaler`` (a no-op one when *enabled* is ``False``).

    On CPU or when AMP is disabled the scaler is created with
    ``enabled=False`` so all its methods become identity operations.

    Args:
        enabled: Whether AMP mixed-precision training is active.

    Returns:
        A ``torch.cuda.amp.GradScaler`` instance.
    """
    return torch.cuda.amp.GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# Autocast context manager
# ---------------------------------------------------------------------------

@contextmanager
def autocast_ctx(enabled: bool, device_type: str = "cuda") -> Generator:
    """
    Context manager that applies ``torch.autocast`` only when *enabled*.

    Args:
        enabled:     Whether to use autocast.
        device_type: ``"cuda"`` or ``"cpu"`` (autocast on CPU is a no-op).

    Yields:
        Nothing — used purely for its side-effects.

    Example::

        with autocast_ctx(cfg.amp):
            logits = model(inputs)
            loss   = criterion(logits, targets)
    """
    if enabled and device_type == "cuda":
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# Backward + step helpers
# ---------------------------------------------------------------------------

def scaled_backward(
    scaler:    torch.cuda.amp.GradScaler,
    loss:      torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    Run ``scaler.scale(loss).backward()`` and then ``scaler.step(optimizer)``.

    Separating these two operations lets callers insert gradient-norm
    computation between backward and the optimizer step.

    Note: caller must call ``scaler.update()`` separately after this.

    Args:
        scaler:    Active ``GradScaler`` (may be a no-op scaler).
        loss:      Scalar loss tensor.
        optimizer: Optimizer whose parameters will be updated.
    """
    scaler.scale(loss).backward()


def optimizer_step(
    scaler:    torch.cuda.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    model:     nn.Module,
    max_norm:  float = 1.0,
) -> float:
    """
    Unscale gradients, clip by norm, step the optimizer, and update the scaler.

    Args:
        scaler:    Active ``GradScaler``.
        optimizer: Optimizer to step.
        model:     Model whose parameters' gradients will be clipped.
        max_norm:  Maximum gradient L2 norm for clipping (default: 1.0).

    Returns:
        The total gradient norm (before clipping) as a Python float.
    """
    # Unscale so that gradient clipping operates on real-valued gradients
    scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.grad is not None],
        max_norm=max_norm,
    ).item()

    scaler.step(optimizer)
    scaler.update()

    return float(grad_norm)
