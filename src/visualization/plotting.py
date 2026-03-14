"""
src/visualization/plotting.py
──────────────────────────────
Publication-quality multi-panel visualization functions.

All figures are saved at 300 DPI as both PNG and PDF.  A single
``CLASS_COLORMAP`` is used across every panel so that color↔class
associations are visually consistent throughout the report.

Public API
----------
``build_class_colormap``         – generate / load reproducible 30-class colors.
``save_colormap_json``           – persist color mapping to JSON.
``plot_embedding_grid``          – multi-panel (rows=depths, cols=models) scatter.
``plot_cluster_compactness``     – grouped bar chart of intra-class compactness.
``_save_fig``                    – helper to save PNG + PDF at 300 DPI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Matplotlib backend (always non-interactive on servers)
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba


# ---------------------------------------------------------------------------
# Class colormap
# ---------------------------------------------------------------------------

# 30 visually distinct colors (hand-tuned palette, then extended with HSV)
_BASE_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#ffffff", "#000000", "#e6beff", "#0075dc", "#993f00",
    "#4c005c", "#191919", "#ff0010", "#5ef1f2", "#9dcc00",
]


def build_class_colormap(
    class_names: Optional[List[str]] = None,
    n_classes:   int = 30,
) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Return a dict mapping class index → RGBA tuple.

    Uses a fixed palette so the same class always gets the same color
    across all figures in the report.

    Args:
        class_names: Optional list of 30 class name strings (for JSON export).
        n_classes:   Number of classes (default 30).

    Returns:
        Dict ``{class_idx: (r, g, b, a)}``.
    """
    colors = {}
    for i in range(n_classes):
        hex_c  = _BASE_PALETTE[i % len(_BASE_PALETTE)]
        colors[i] = to_rgba(hex_c)
    return colors


def save_colormap_json(
    colormap:    Dict[int, Tuple],
    output_path: Path | str,
    class_names: Optional[List[str]] = None,
) -> Path:
    """
    Persist the colormap to a JSON file.

    Args:
        colormap:    Dict from :func:`build_class_colormap`.
        output_path: Destination JSON path.
        class_names: Optional list of class name strings.

    Returns:
        Path of the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for idx, rgba in colormap.items():
        entry: Dict = {"rgba": list(rgba)}
        if class_names and int(idx) < len(class_names):
            entry["name"] = class_names[int(idx)]
        data[str(idx)] = entry
    with out.open("w") as fh:
        json.dump(data, fh, indent=2)
    return out


# ---------------------------------------------------------------------------
# Figure save helper
# ---------------------------------------------------------------------------

def _save_fig(
    fig,
    stem:    Path | str,
    formats: Sequence[str] = ("png", "pdf"),
    dpi:     int = 300,
) -> List[Path]:
    """Save a matplotlib figure in one or more formats."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for fmt in formats:
        p = stem.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Single scatter panel
# ---------------------------------------------------------------------------

def _scatter_panel(
    ax,
    coords:      np.ndarray,
    labels:      np.ndarray,
    colormap:    Dict[int, Tuple],
    title:       str = "",
    alpha:       float = 0.55,
    point_size:  float = 8.0,
    show_axes:   bool = False,
) -> None:
    """Draw one scatter panel on *ax*."""
    unique_classes = sorted(np.unique(labels).tolist())
    for cls in unique_classes:
        mask = labels == cls
        color = colormap.get(cls, (0.5, 0.5, 0.5, 1.0))
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c      = [color],
            s      = point_size,
            alpha  = alpha,
            linewidths = 0,
            rasterized = True,
        )
    if title:
        ax.set_title(title, fontsize=9, pad=3)
    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


# ---------------------------------------------------------------------------
# Multi-panel embedding grid
# ---------------------------------------------------------------------------

def plot_embedding_grid(
    embeddings:  Dict[str, Dict[str, np.ndarray]],
    labels_dict: Dict[str, Dict[str, np.ndarray]],
    colormap:    Dict[int, Tuple],
    model_names: List[str],
    depth_tags:  List[str],
    method:      str = "PCA",
    output_stem: Path | str = "outputs/plots/embedding_grid",
    class_names: Optional[List[str]] = None,
    suptitle:    str = "",
    formats:     Sequence[str] = ("png", "pdf"),
    run_id:      Optional[int] = None,
) -> List[Path]:
    """
    Create a (rows=depths) × (cols=models) grid of scatter plots.

    Args:
        embeddings:  ``{model_name: {depth_tag: coords_2d}}``
        labels_dict: ``{model_name: {depth_tag: label_array}}``
        colormap:    Class index → RGBA from :func:`build_class_colormap`.
        model_names: Column order of model names.
        depth_tags:  Row order of depth tags.
        method:      Reduction method label (``"PCA"``, ``"t-SNE"``, …).
        output_stem: File path stem (format suffix appended).
        class_names: Optional list of class name strings for the legend.
        suptitle:    Optional super-title for the full figure.
        formats:     File formats to save.
        run_id:      If provided, appended to filename (for t-SNE multi-run).

    Returns:
        List of saved file paths.
    """
    n_rows = len(depth_tags)
    n_cols = len(model_names)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize = (4.5 * n_cols, 3.5 * n_rows),
        squeeze = False,
    )

    for r, depth in enumerate(depth_tags):
        for c, model in enumerate(model_names):
            ax    = axes[r][c]
            coords= embeddings.get(model, {}).get(depth)
            lbls  = labels_dict.get(model, {}).get(depth)

            if coords is None or lbls is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_title(f"{model}\n{depth}", fontsize=8)
                continue

            col_title = model if r == 0 else ""
            row_label = depth.capitalize() if c == 0 else ""

            _scatter_panel(
                ax       = ax,
                coords   = coords,
                labels   = lbls,
                colormap = colormap,
                title    = f"{model} — {depth.capitalize()}" if (r == 0 or c == 0) else depth.capitalize(),
                alpha    = 0.5 if len(lbls) > 500 else 0.7,
            )

            # Row label on left-most column
            if c == 0:
                ax.set_ylabel(depth.capitalize(), fontsize=9, labelpad=4)

    # Column headers
    for c, model in enumerate(model_names):
        axes[0][c].set_title(model, fontsize=10, fontweight="bold", pad=6)

    # Legend (max 30 patches, small font)
    if class_names:
        patches = [
            mpatches.Patch(color=colormap.get(i, (0.5, 0.5, 0.5, 1.0)),
                           label=class_names[i] if i < len(class_names) else str(i))
            for i in range(min(30, len(colormap)))
        ]
        fig.legend(
            handles       = patches,
            loc           = "lower center",
            ncol          = 10,
            fontsize      = 6,
            frameon       = False,
            bbox_to_anchor= (0.5, -0.04),
            handlelength  = 1.0,
            handleheight  = 0.8,
        )
        fig.subplots_adjust(bottom=0.10)

    title = suptitle or f"{method} Embeddings — AID Dataset"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    stem = Path(output_stem)
    if run_id is not None:
        stem = stem.parent / f"{stem.name}_run{run_id}"

    paths = _save_fig(fig, stem, formats=formats)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Cluster compactness figure
# ---------------------------------------------------------------------------

def plot_cluster_compactness(
    compactness_data: Dict[str, Dict[str, Dict[int, float]]],
    model_names:      List[str],
    depth_tags:       List[str],
    output_path:      Path | str = "outputs/plots/cluster_compactness.png",
    class_names:      Optional[List[str]] = None,
    formats:          Sequence[str] = ("png", "pdf"),
) -> List[Path]:
    """
    Plot mean intra-class compactness scores per (model, depth) as grouped bars.

    Args:
        compactness_data: ``{model_name: {depth_tag: {class_idx: distance}}}``
        model_names:      Models to include.
        depth_tags:       Depth tags to include.
        output_path:      Destination PNG path (PDF also saved).
        class_names:      Optional list for x-axis class labels.
        formats:          File formats to save.

    Returns:
        List of saved file paths.
    """
    # Compute mean compactness across classes for each (model, depth)
    means: Dict[str, List[float]] = {m: [] for m in model_names}

    for model in model_names:
        for depth in depth_tags:
            cls_dict = compactness_data.get(model, {}).get(depth, {})
            if cls_dict:
                mean_val = float(np.mean(list(cls_dict.values())))
            else:
                mean_val = 0.0
            means[model].append(mean_val)

    # Grouped bar chart
    x        = np.arange(len(depth_tags))
    width    = 0.8 / max(len(model_names), 1)
    colors   = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    fig, ax  = plt.subplots(figsize=(9, 5))

    for i, model in enumerate(model_names):
        offsets = x + (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(offsets, means[model], width * 0.9,
                      label=model, color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, means[model]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in depth_tags], fontsize=11)
    ax.set_xlabel("Layer Depth", fontsize=12)
    ax.set_ylabel("Mean Intra-class Distance (2D embedding)", fontsize=11)
    ax.set_title("Cluster Compactness by Model and Layer Depth", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10, frameon=True)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    stem = Path(output_path)
    stem = stem.parent / stem.stem
    paths = _save_fig(fig, stem, formats=formats)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Perplexity sensitivity figure
# ---------------------------------------------------------------------------

def plot_tsne_kl_sensitivity(
    kl_data:     Dict[str, Dict[float, float]],
    output_path: Path | str = "outputs/plots/tsne_kl_sensitivity.png",
    formats:     Sequence[str] = ("png",),
) -> List[Path]:
    """
    Plot t-SNE KL divergence vs. perplexity for each model.

    Args:
        kl_data:     ``{model_name: {perplexity_value: kl_divergence}}``.
        output_path: Destination file stem.
        formats:     File formats.

    Returns:
        List of saved file paths.
    """
    fig, ax   = plt.subplots(figsize=(7, 4))
    colors    = ["#2196F3", "#F44336", "#4CAF50"]
    for i, (model, perp_kl) in enumerate(sorted(kl_data.items())):
        perps = sorted(perp_kl.keys())
        kls   = [perp_kl[p] for p in perps]
        ax.plot(perps, kls, marker="o", label=model,
                color=colors[i % len(colors)], linewidth=1.8)

    ax.set_xlabel("Perplexity", fontsize=11)
    ax.set_ylabel("KL Divergence", fontsize=11)
    ax.set_title("t-SNE KL Divergence vs Perplexity", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    stem  = Path(output_path)
    stem  = stem.parent / stem.stem
    paths = _save_fig(fig, stem, formats=formats)
    plt.close(fig)
    return paths
