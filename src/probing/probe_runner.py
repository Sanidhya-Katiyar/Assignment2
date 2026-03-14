"""
src/probing/probe_runner.py
────────────────────────────
High-level orchestrator that:

1. Loads a trained checkpoint into the model.
2. Extracts features from the three registered depth layers using the
   visualization subset.
3. Trains a linear classifier on each layer's features.
4. Records and returns accuracy per layer.

Also handles CSV persistence and (optionally) feature caching so that the
expensive extraction step can be skipped on repeated runs.

Public API
----------
``ProbeResult``     – typed result container for one (model, layer) pair.
``run_probes``      – end-to-end probe run for one model; returns list of ProbeResult.
``save_probe_csv``  – write results to ``outputs/probing/layer_probe_results.csv``.
``load_probe_csv``  – read previously saved results for plotting.
``plot_depth_accuracy`` – bar + line chart of accuracy vs. depth, one curve per model.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.probing.feature_extractor  import (
    DEPTH_ORDER,
    LAYER_REGISTRY,
    extract_all_layers,
)
from src.probing.linear_probe_layer import probe_features
from src.train.utils_checkpoint     import load_checkpoint


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ProbeResult:
    """
    Accuracy of a linear probe at one (model, layer-depth) point.

    Attributes:
        model_name:  Architecture name (e.g. ``"resnet50"``).
        layer_tag:   Depth tag: ``"early"``, ``"middle"``, or ``"final"``.
        layer_path:  Dotted attribute path of the probed layer.
        accuracy:    Top-1 accuracy in ``[0, 1]``.
        feature_dim: Dimensionality of the extracted feature vector.
        n_train:     Number of training samples used for the probe.
        n_eval:      Number of evaluation samples used for the probe.
    """

    model_name:  str
    layer_tag:   str
    layer_path:  str
    accuracy:    float
    feature_dim: int
    n_train:     int
    n_eval:      int

    def __repr__(self) -> str:
        return (
            f"ProbeResult({self.model_name}/{self.layer_tag}: "
            f"acc={self.accuracy:.4f}, dim={self.feature_dim})"
        )


# ---------------------------------------------------------------------------
# Main probe runner
# ---------------------------------------------------------------------------

def run_probes(
    model:           nn.Module,
    model_name:      str,
    dataloader:      DataLoader,
    num_classes:     int,
    device:          torch.device,
    checkpoint_path: Optional[str | Path] = None,
    depth_tags:      Optional[List[str]]  = None,
    probe_epochs:    int   = 10,
    probe_lr:        float = 1e-3,
    probe_batch:     int   = 64,
    train_frac:      float = 0.8,
    seed:            int   = 42,
    verbose:         bool  = False,
    feature_cache_dir: Optional[Path] = None,
) -> List[ProbeResult]:
    """
    Run linear probes at all registered depth layers for *model_name*.

    Steps
    -----
    1. Optionally load *checkpoint_path* into *model*.
    2. Extract features from each registered layer using *dataloader*.
    3. Train a :class:`~src.probing.linear_probe_layer.LinearProbeClassifier`
       on a stratified 80 / 20 split of the extracted features.
    4. Record accuracy in a :class:`ProbeResult`.

    Args:
        model:             Backbone ``nn.Module`` (head may be present; it is
                           not used during extraction).
        model_name:        Key into ``LAYER_REGISTRY`` (e.g. ``"resnet50"``).
        dataloader:        DataLoader for the probe dataset (visualization subset).
        num_classes:       Number of AID classes (30).
        device:            Compute device.
        checkpoint_path:   Optional path to a ``.pth`` checkpoint.  If ``None``,
                           the model's current weights are used as-is.
        depth_tags:        Subset of depth tags to probe (default: all three).
        probe_epochs:      Linear probe training epochs (default 10).
        probe_lr:          Adam learning rate for the probe (default 1e-3).
        probe_batch:       Mini-batch size for probe training (default 64).
        train_frac:        Fraction of features for training (default 0.8).
        seed:              Random seed for the train/eval split.
        verbose:           Print per-epoch probe loss.
        feature_cache_dir: If provided, save/load extracted features as ``.npz``
                           files so extraction can be skipped on reruns.

    Returns:
        List of :class:`ProbeResult`, one per successfully probed layer,
        in depth order (early → middle → final).

    Raises:
        KeyError: If *model_name* not in LAYER_REGISTRY.
    """
    if model_name not in LAYER_REGISTRY:
        raise KeyError(
            f"'{model_name}' not in LAYER_REGISTRY.  "
            f"Supported: {sorted(LAYER_REGISTRY.keys())}."
        )

    # ── 1. Load checkpoint ────────────────────────────────────────────
    if checkpoint_path is not None:
        ckpt_info = load_checkpoint(
            path   = Path(checkpoint_path),
            model  = model,
            device = str(device),
            strict = False,          # allow missing head params if probing backbone only
        )
        print(f"  Loaded checkpoint from '{checkpoint_path}' "
              f"(epoch={ckpt_info.get('epoch', '?')}, "
              f"best_val_acc={ckpt_info.get('best_val_acc', '?'):.2f}%)")

    model.eval()
    model.to(device)

    tags = depth_tags if depth_tags is not None else DEPTH_ORDER
    registry = LAYER_REGISTRY[model_name]

    # ── 2. Extract features (with optional caching) ───────────────────
    layer_features: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for tag in tags:
        if tag not in registry:
            continue

        cache_path: Optional[Path] = None
        if feature_cache_dir is not None:
            cache_path = Path(feature_cache_dir) / f"{model_name}_{tag}_features.npz"

        if cache_path is not None and cache_path.exists():
            print(f"  Loading cached features for {model_name}/{tag} …")
            npz = np.load(cache_path, allow_pickle=False)
            layer_features[tag] = (npz["features"], npz["labels"])
        else:
            # extract_all_layers handles resolve + hook + GAP + flatten
            extracted = extract_all_layers(
                model       = model,
                model_name  = model_name,
                dataloader  = dataloader,
                device      = device,
                depth_tags  = [tag],
            )
            if tag not in extracted:
                print(f"  [warn] extraction failed for {model_name}/{tag}, skipping.")
                continue

            layer_features[tag] = extracted[tag]

            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    features = layer_features[tag][0],
                    labels   = layer_features[tag][1],
                )

    # ── 3. Train probes and record results ────────────────────────────
    results: List[ProbeResult] = []

    for tag in tags:
        if tag not in layer_features:
            continue

        feats, labels = layer_features[tag]
        n_total       = len(labels)
        n_train_est   = max(1, int(n_total * train_frac))
        n_eval_est    = max(1, n_total - n_train_est)

        print(f"  Probing {model_name}/{tag}  "
              f"(dim={feats.shape[1]}, n={n_total}) …", end="", flush=True)

        acc, _ = probe_features(
            features    = feats,
            labels      = labels,
            num_classes = num_classes,
            train_frac  = train_frac,
            epochs      = probe_epochs,
            lr          = probe_lr,
            batch_size  = probe_batch,
            seed        = seed,
            device      = device,
            verbose     = verbose,
        )

        print(f"  acc={acc:.4f}")

        results.append(ProbeResult(
            model_name  = model_name,
            layer_tag   = tag,
            layer_path  = registry[tag],
            accuracy    = acc,
            feature_dim = feats.shape[1],
            n_train     = n_train_est,
            n_eval      = n_eval_est,
        ))

    return results


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "model", "layer", "layer_path", "accuracy",
    "feature_dim", "n_train", "n_eval",
]


def save_probe_csv(
    results:     List[ProbeResult],
    output_path: Path | str,
    append:      bool = True,
) -> Path:
    """
    Write probe results to a CSV file.

    If *append* is ``True`` and the file already exists, new rows are
    appended (no duplicate header is written).  Set *append=False* to
    overwrite.

    Args:
        results:     List of :class:`ProbeResult` objects.
        output_path: Destination CSV path.
        append:      Append to existing file (default ``True``).

    Returns:
        Path of the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    write_header = (not out.exists()) or (not append)
    mode         = "a" if append else "w"

    with out.open(mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "model":       r.model_name,
                "layer":       r.layer_tag,
                "layer_path":  r.layer_path,
                "accuracy":    f"{r.accuracy:.6f}",
                "feature_dim": r.feature_dim,
                "n_train":     r.n_train,
                "n_eval":      r.n_eval,
            })

    return out


def load_probe_csv(csv_path: Path | str) -> List[ProbeResult]:
    """
    Load probe results from a previously saved CSV.

    Args:
        csv_path: Path to the CSV file written by :func:`save_probe_csv`.

    Returns:
        List of :class:`ProbeResult` objects.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Probe CSV not found: '{p}'")

    results: List[ProbeResult] = []
    with p.open() as fh:
        for row in csv.DictReader(fh):
            results.append(ProbeResult(
                model_name  = row["model"],
                layer_tag   = row["layer"],
                layer_path  = row["layer_path"],
                accuracy    = float(row["accuracy"]),
                feature_dim = int(row["feature_dim"]),
                n_train     = int(row["n_train"]),
                n_eval      = int(row["n_eval"]),
            ))
    return results


# ---------------------------------------------------------------------------
# Depth-accuracy plot
# ---------------------------------------------------------------------------

def plot_depth_accuracy(
    results:     List[ProbeResult],
    output_stem: Path | str,
    title:       str = "Layer-wise Probe Accuracy",
    formats:     tuple = ("png", "pdf"),
) -> List[Path]:
    """
    Plot probe accuracy vs. layer depth for each model.

    Generates one line per model with markers at each depth tag
    (early / middle / final mapped to x = 0 / 1 / 2).

    Args:
        results:     List of :class:`ProbeResult` (may cover multiple models).
        output_stem: File path stem (extension is appended per format).
        title:       Figure title.
        formats:     Iterable of file format strings (default ``("png", "pdf")``).

    Returns:
        List of written file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group by model
    model_data: Dict[str, Dict[str, float]] = {}
    for r in results:
        model_data.setdefault(r.model_name, {})[r.layer_tag] = r.accuracy

    depth_x   = {tag: i for i, tag in enumerate(DEPTH_ORDER)}
    x_ticks   = list(range(len(DEPTH_ORDER)))
    colors    = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
    markers   = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (model_name, tag_accs) in enumerate(sorted(model_data.items())):
        xs: list = []
        ys: list = []
        for tag in DEPTH_ORDER:
            if tag in tag_accs:
                xs.append(depth_x[tag])
                ys.append(tag_accs[tag] * 100.0)   # display as %

        col = colors[idx % len(colors)]
        mrk = markers[idx % len(markers)]
        ax.plot(xs, ys, color=col, marker=mrk, linewidth=2,
                markersize=9, label=model_name)

        # Annotate each point with its value
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}%", xy=(x, y),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=col)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([t.capitalize() for t in DEPTH_ORDER], fontsize=11)
    ax.set_xlabel("Layer Depth", fontsize=12)
    ax.set_ylabel("Probe Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 105)
    plt.tight_layout()

    stem  = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for fmt in formats:
        p = stem.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        paths.append(p)

    plt.close(fig)
    return paths
