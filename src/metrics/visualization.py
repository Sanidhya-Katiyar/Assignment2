"""
src/metrics/visualization.py
─────────────────────────────
Plotting helpers for training diagnostics.  All functions write to disk
(PNG + PDF by default) and return the output path for logging.

Dependencies: matplotlib, seaborn (optional — falls back to plain
matplotlib for the confusion matrix if seaborn is absent).

Public API
----------
``plot_train_val_curves``  – loss and accuracy curves over epochs.
``plot_confusion_matrix``  – annotated heatmap from a confusion matrix array.
``plot_param_vs_epoch``    – arbitrary scalar vs. epoch (e.g. gradient norm, lr).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Internal: lazy matplotlib import
# ---------------------------------------------------------------------------

def _get_mpl():
    """Return (matplotlib, pyplot) — deferred import so CI can skip graphics."""
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend, safe on servers
    import matplotlib.pyplot as plt
    return matplotlib, plt


# ---------------------------------------------------------------------------
# Train / val curves
# ---------------------------------------------------------------------------

def plot_train_val_curves(
    epochs:       Sequence[int],
    train_losses: Sequence[float],
    val_losses:   Sequence[float],
    train_accs:   Sequence[float],
    val_accs:     Sequence[float],
    output_stem:  Path | str,
    title:        str = "",
    formats:      Sequence[str] = ("png", "pdf"),
) -> List[Path]:
    """
    Plot training and validation loss + accuracy on a 1×2 figure.

    Args:
        epochs:       Epoch indices (x-axis).
        train_losses: Per-epoch training loss.
        val_losses:   Per-epoch validation loss.
        train_accs:   Per-epoch training accuracy (0–100).
        val_accs:     Per-epoch validation accuracy (0–100).
        output_stem:  File path WITHOUT extension; format suffixes are appended.
        title:        Optional super-title for the figure.
        formats:      Tuple of file formats to save (default ``("png", "pdf")``).

    Returns:
        List of written file paths.
    """
    _, plt = _get_mpl()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, train_losses, label="Train loss", color="#2196F3")
    ax.plot(epochs, val_losses,   label="Val loss",   color="#F44336", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, train_accs, label="Train acc", color="#2196F3")
    ax.plot(epochs, val_accs,   label="Val acc",   color="#F44336", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()

    paths: List[Path] = []
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        p = stem.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        paths.append(p)

    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    matrix:       np.ndarray,
    class_names:  Optional[List[str]],
    output_path:  Path | str,
    title:        str = "Confusion Matrix",
    normalize:    bool = True,
    figsize_per:  float = 0.5,
) -> Path:
    """
    Save a confusion matrix heatmap as PNG.

    Args:
        matrix:       Square array of shape ``(C, C)`` — raw counts.
        class_names:  List of class name strings (optional).
        output_path:  Destination ``.png`` file path.
        title:        Figure title.
        normalize:    If ``True``, normalise each row to [0, 1].
        figsize_per:  Inches per class for auto-sizing.

    Returns:
        Path of the saved file.
    """
    _, plt = _get_mpl()

    cm = matrix.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1          # avoid div-by-zero for empty classes
        cm = cm / row_sums

    n       = cm.shape[0]
    figsize = max(8, n * figsize_per)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))

    try:
        import seaborn as sns
        sns.heatmap(
            cm, ax=ax,
            vmin=0, vmax=1 if normalize else None,
            cmap="Blues", square=True,
            xticklabels=class_names or "auto",
            yticklabels=class_names or "auto",
            annot=(n <= 30),   # annotations only if legible
            fmt=".2f" if normalize else "d",
            linewidths=0.4,
        )
    except ImportError:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=1 if normalize else None)
        plt.colorbar(im, ax=ax)
        if class_names and n <= 30:
            ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=90, fontsize=7)
            ax.set_yticks(range(n)); ax.set_yticklabels(class_names, fontsize=7)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Generic scalar-vs-epoch plot
# ---------------------------------------------------------------------------

def plot_scalar_curve(
    epochs:      Sequence[int],
    values:      Sequence[float],
    output_path: Path | str,
    ylabel:      str = "Value",
    title:       str = "",
    color:       str = "#4CAF50",
) -> Path:
    """
    Plot a single scalar time-series over training epochs.

    Useful for gradient norm, learning rate, GPU memory, etc.

    Args:
        epochs:      Epoch indices.
        values:      Scalar values aligned with *epochs*.
        output_path: Destination ``.png`` file.
        ylabel:      Y-axis label.
        title:       Figure title.
        color:       Line colour (hex or named).

    Returns:
        Path of the saved file.
    """
    _, plt = _get_mpl()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, values, color=color, linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
