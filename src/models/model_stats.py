"""
src/models/model_stats.py
──────────────────────────
Utilities for computing and reporting model complexity metrics:

* **Total parameters** – all ``nn.Parameter`` elements in the model.
* **Trainable parameters** – parameters with ``requires_grad=True``.
* **MACs / FLOPs** – multiply-accumulate operations for one forward pass,
  computed via ``ptflops`` (preferred) with a pure-PyTorch hook-based
  fallback so the codebase degrades gracefully if ptflops is unavailable.

The primary public symbol is :func:`get_model_stats`, which returns a
:class:`ModelStats` dataclass.  A :func:`format_stats_table` helper
renders a human-readable comparison table for multiple models.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelStats:
    """
    Complexity metrics for a single model.

    Attributes:
        model_name:        Display name of the architecture.
        total_params:      Total parameter count (including frozen).
        trainable_params:  Parameter count with ``requires_grad=True``.
        flops:             Approximate MACs for a single forward pass.
                           ``None`` if computation was not possible.
        input_size:        ``(C, H, W)`` used for the FLOPs estimate.
        flops_source:      Which library / method produced the FLOPs figure.
    """

    model_name:       str
    total_params:     int
    trainable_params: int
    flops:            Optional[int]              = None
    input_size:       Tuple[int, int, int]       = (3, 224, 224)
    flops_source:     str                        = "unknown"
    extra:            Dict[str, object]          = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Formatted properties
    # ------------------------------------------------------------------

    @property
    def total_params_m(self) -> str:
        """Total parameters formatted as e.g. '25.56M'."""
        return _fmt_params(self.total_params)

    @property
    def trainable_params_m(self) -> str:
        """Trainable parameters formatted as e.g. '25.56M'."""
        return _fmt_params(self.trainable_params)

    @property
    def flops_str(self) -> str:
        """FLOPs formatted as e.g. '4.10G' or 'N/A'."""
        if self.flops is None:
            return "N/A"
        return _fmt_flops(self.flops)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_flops(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}G"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    return str(n)


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in *model*.

    Args:
        model: Any ``nn.Module``.

    Returns:
        ``(total_params, trainable_params)`` as plain integers.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# FLOPs / MACs computation
# ---------------------------------------------------------------------------

def _flops_via_ptflops(
    model: nn.Module,
    input_size: Tuple[int, int, int],
) -> Tuple[Optional[int], str]:
    """
    Attempt to compute MACs using ``ptflops``.

    Returns:
        ``(macs_int_or_None, source_string)``
    """
    try:
        from ptflops import get_model_complexity_info  # type: ignore

        # ptflops expects (C, H, W) without the batch dimension
        macs, _ = get_model_complexity_info(
            model,
            input_size,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        # ptflops returns MACs; multiply by 2 for FLOPs (optional convention)
        return int(macs), "ptflops (MACs)"
    except ImportError:
        return None, "ptflops not installed"
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"ptflops failed: {exc}. Falling back to hook-based counter.")
        return None, f"ptflops error: {exc}"


def _flops_via_hooks(
    model: nn.Module,
    input_size: Tuple[int, int, int],
) -> Tuple[Optional[int], str]:
    """
    Fallback FLOPs estimator using forward hooks.

    Counts MACs for ``Conv2d``, ``Linear``, and ``BatchNorm2d`` layers.
    This is a *lower-bound* estimate — activation functions and some exotic
    ops are not counted.

    Returns:
        ``(macs_int_or_None, source_string)``
    """
    macs_accumulator: List[int] = [0]
    hooks = []

    def _conv2d_hook(module: nn.Conv2d, inp, out):
        batch     = out.shape[0]
        out_h     = out.shape[2]
        out_w     = out.shape[3]
        k_h, k_w  = module.kernel_size if isinstance(module.kernel_size, tuple) \
                     else (module.kernel_size, module.kernel_size)
        in_ch_per_group = module.in_channels // module.groups
        macs = batch * module.out_channels * out_h * out_w * in_ch_per_group * k_h * k_w
        macs_accumulator[0] += macs

    def _linear_hook(module: nn.Linear, inp, out):
        batch = inp[0].shape[0] if inp[0].dim() > 1 else 1
        macs_accumulator[0] += batch * module.in_features * module.out_features

    def _bn_hook(module: nn.BatchNorm2d, inp, out):
        macs_accumulator[0] += inp[0].numel()

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv2d_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))
        elif isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_hook(_bn_hook))

    device = next(model.parameters()).device
    dummy  = torch.zeros(1, *input_size, device=device)
    training_state = model.training
    model.eval()

    try:
        with torch.no_grad():
            model(dummy)
        result = macs_accumulator[0]
        source = "hook-based (MACs, approximate)"
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Hook-based FLOPs counter failed: {exc}")
        result = None
        source = f"hook-based error: {exc}"
    finally:
        for h in hooks:
            h.remove()
        model.train(training_state)

    return result, source


def compute_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 224, 224),
) -> Tuple[Optional[int], str]:
    """
    Compute approximate MACs for one forward pass through *model*.

    Tries ``ptflops`` first; falls back to a hook-based counter if
    ``ptflops`` is not installed or raises an exception.

    Args:
        model:      Any ``nn.Module``.
        input_size: ``(C, H, W)`` tensor shape (no batch dimension).

    Returns:
        ``(macs_or_None, source_string)``
    """
    macs, source = _flops_via_ptflops(model, input_size)
    if macs is not None:
        return macs, source
    # Fallback
    return _flops_via_hooks(model, input_size)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def get_model_stats(
    model:      nn.Module,
    model_name: str                      = "",
    input_size: Tuple[int, int, int]     = (3, 224, 224),
) -> ModelStats:
    """
    Compute total params, trainable params, and FLOPs for *model*.

    Args:
        model:      Any ``nn.Module`` (typically from
                    :func:`~src.models.model_factory.create_model`).
        model_name: Human-readable label stored in the returned stats object.
        input_size: ``(C, H, W)`` input shape used for FLOPs estimation.

    Returns:
        :class:`ModelStats` dataclass with all metrics populated.

    Example:
        >>> stats = get_model_stats(model, model_name="ResNet-50")
        >>> print(stats.total_params_m, stats.flops_str)
    """
    total, trainable = count_parameters(model)
    flops, source    = compute_flops(model, input_size)

    return ModelStats(
        model_name       = model_name,
        total_params     = total,
        trainable_params = trainable,
        flops            = flops,
        input_size       = input_size,
        flops_source     = source,
    )


# ---------------------------------------------------------------------------
# Table formatter
# ---------------------------------------------------------------------------

def format_stats_table(stats_list: List[ModelStats]) -> str:
    """
    Render a human-readable comparison table for a list of :class:`ModelStats`.

    Args:
        stats_list: One entry per architecture.

    Returns:
        A multi-line string ready to be printed to stdout.

    Example output::

        ┌─────────────────┬────────────┬────────────┬────────────┐
        │ Model           │ Params     │ Trainable  │ FLOPs      │
        ├─────────────────┼────────────┼────────────┼────────────┤
        │ ResNet-50       │ 25.56M     │ 25.56M     │ 4.10G      │
        │ EfficientNet-B0 │  5.29M     │  5.29M     │ 0.39G      │
        │ ConvNeXt-Tiny   │ 28.59M     │ 28.59M     │ 4.47G      │
        └─────────────────┴────────────┴────────────┴────────────┘
    """
    col_model = max(len(s.model_name) for s in stats_list)
    col_model = max(col_model, len("Model"))
    col_w     = 12  # fixed width for numeric columns

    def _row(name: str, total: str, trainable: str, flops: str) -> str:
        return (
            f"│ {name:<{col_model}} │ {total:>{col_w-2}} │ "
            f"{trainable:>{col_w-2}} │ {flops:>{col_w-2}} │"
        )

    def _sep(left: str, mid: str, right: str, fill: str = "─") -> str:
        return (
            left
            + fill * (col_model + 2)
            + mid
            + fill * col_w
            + mid
            + fill * col_w
            + mid
            + fill * col_w
            + right
        )

    lines = [
        _sep("┌", "┬", "┐"),
        _row("Model", "Params", "Trainable", "FLOPs"),
        _sep("├", "┼", "┤"),
    ]
    for s in stats_list:
        lines.append(_row(s.model_name, s.total_params_m, s.trainable_params_m, s.flops_str))
    lines.append(_sep("└", "┴", "┘"))

    return "\n".join(lines)
