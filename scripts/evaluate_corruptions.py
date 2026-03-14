#!/usr/bin/env python3
"""
scripts/evaluate_corruptions.py
────────────────────────────────
Load a trained checkpoint and evaluate it under Gaussian noise, motion
blur, and brightness shift at multiple severity levels.  Saves a CSV
results table and a robustness-curve plot.

Usage
-----
    python scripts/evaluate_corruptions.py \\
        --config  configs/baseline.yaml \\
        --model   resnet50 \\
        --checkpoint outputs/checkpoints/resnet50_linearprobe_42_best.pth

    # Evaluate multiple checkpoints in one run:
    python scripts/evaluate_corruptions.py \\
        --config configs/baseline.yaml \\
        --model resnet50 efficientnet_b0 \\
        --checkpoint path/to/resnet50.pth path/to/effnet.pth

    # Restrict to specific corruptions:
    python scripts/evaluate_corruptions.py \\
        --config configs/baseline.yaml \\
        --model resnet50 \\
        --checkpoint path/to/ckpt.pth \\
        --corruptions gaussian_noise motion_blur

Output files
------------
    outputs/robustness/corruption_results.csv
    outputs/plots/robustness_curves.png
    outputs/plots/robustness_curves.pdf
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.aid_dataset      import AIDDataset, discover_dataset, get_transforms
from src.datasets.split_utils      import stratified_split
from src.evaluation.corruptions    import CORRUPTION_REGISTRY, CORRUPTION_DISPLAY
from src.evaluation.robustness_eval import (
    RobustnessResult,
    evaluate_clean,
    evaluate_model_on_corruptions,
)
from src.models.model_factory      import create_model
from src.train.utils_checkpoint    import load_checkpoint
from src.utils.config              import load_config
from src.utils.seed                import set_seed

from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate trained models under image corruptions."
    )
    p.add_argument("--config",      required=True,
                   help="Path to YAML config (e.g. configs/baseline.yaml).")
    p.add_argument("--model",       nargs="+", required=True,
                   choices=["resnet50", "efficientnet_b0", "convnext_tiny"],
                   help="One or more backbone names.")
    p.add_argument("--checkpoint",  nargs="+", required=True,
                   help="Path(s) to .pth checkpoint(s), one per model.")
    p.add_argument("--corruptions", nargs="+", default=None,
                   choices=list(CORRUPTION_REGISTRY.keys()),
                   help="Subset of corruptions to evaluate (default: all).")
    p.add_argument("--batch-size",  type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--split",       default="val",
                   choices=["train", "val", "test"],
                   help="Dataset split to evaluate on (default: val).")
    p.add_argument("--output-dir",  default="outputs",
                   help="Root output directory (default: outputs).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset builder (validation split, no shuffle)
# ---------------------------------------------------------------------------

def build_val_dataset(config, split: str = "val") -> AIDDataset:
    """
    Rebuild the validation (or test) split from the dataset discovery +
    stratified split utilities so that no pre-existing CSV is required.
    """
    image_paths, labels, class_to_idx = discover_dataset(config.dataset_path)

    (train_paths, train_labels), \
    (val_paths,   val_labels),   \
    (test_paths,  test_labels) = stratified_split(
        image_paths = image_paths,
        labels      = labels,
        train_frac  = config.train_split,
        val_frac    = config.val_split,
        test_frac   = config.test_split,
        seed        = config.seed,
    )

    if split == "val":
        paths, lbls = val_paths, val_labels
    elif split == "test":
        paths, lbls = test_paths, test_labels
    else:
        paths, lbls = train_paths, train_labels

    # Use standard eval transform; robustness_eval will replace it per-severity
    transform = get_transforms("test", config.image_size)
    return AIDDataset(paths, lbls, class_to_idx, transform=transform)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "model", "corruption_type", "severity",
    "accuracy", "corruption_error", "relative_robustness",
    "clean_accuracy", "n_samples",
]


def save_results_csv(output_path: Path, results: List[RobustnessResult]) -> None:
    """Append robustness results to the output CSV (creates with header if absent)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()

    with output_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "model":               r.model_name,
                "corruption_type":     r.corruption_type,
                "severity":            r.severity,
                "accuracy":            f"{r.accuracy:.4f}",
                "corruption_error":    f"{r.corruption_error:.6f}",
                "relative_robustness": f"{r.relative_robustness:.6f}",
                "clean_accuracy":      f"{r.clean_accuracy:.4f}",
                "n_samples":           r.n_samples,
            })


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _get_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt


# Colour palette — one per model
_PALETTE = [
    "#2196F3",   # blue
    "#F44336",   # red
    "#4CAF50",   # green
    "#FF9800",   # orange
    "#9C27B0",   # purple
]

# Line style per corruption
_LINESTYLES = {
    "gaussian_noise":   "-",
    "motion_blur":      "--",
    "brightness_shift": "-.",
}

# Marker per corruption
_MARKERS = {
    "gaussian_noise":   "o",
    "motion_blur":      "s",
    "brightness_shift": "^",
}


def plot_robustness_curves(
    all_results:  Dict[str, List[RobustnessResult]],   # model_name → results
    output_stem:  Path,
    clean_accs:   Dict[str, float],
    formats:      tuple = ("png", "pdf"),
) -> List[Path]:
    """
    Generate robustness accuracy-vs-severity curves for all models.

    Creates one subplot per corruption type; each subplot shows one line
    per model.  A horizontal dashed line marks each model's clean accuracy.

    Args:
        all_results: Dict mapping model name to its list of RobustnessResult.
        output_stem: Path without extension; format suffixes are appended.
        clean_accs:  Dict mapping model name to its clean baseline accuracy.
        formats:     File formats to save (default: png + pdf).

    Returns:
        List of written file paths.
    """
    _, plt = _get_mpl()

    # Determine corruption types present in the data
    corruption_types: List[str] = []
    for results in all_results.values():
        for r in results:
            if r.corruption_type not in corruption_types:
                corruption_types.append(r.corruption_type)

    n_cols   = len(corruption_types)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
    fig.suptitle("Robustness Evaluation — AID Dataset", fontsize=14, fontweight="bold")

    model_names = list(all_results.keys())

    for col, ctype in enumerate(corruption_types):
        ax          = axes[0][col]
        display     = CORRUPTION_DISPLAY.get(ctype, ctype)
        linestyle   = _LINESTYLES.get(ctype, "-")
        marker      = _MARKERS.get(ctype, "o")

        for m_idx, model_name in enumerate(model_names):
            color   = _PALETTE[m_idx % len(_PALETTE)]
            results = all_results[model_name]

            # Filter to this corruption, sort by severity
            sub = sorted(
                [r for r in results if r.corruption_type == ctype],
                key=lambda r: r.severity,
            )
            if not sub:
                continue

            severities = [str(r.severity) for r in sub]
            accs       = [r.accuracy      for r in sub]

            ax.plot(
                range(len(severities)), accs,
                color=color, linestyle=linestyle, marker=marker,
                linewidth=2.0, markersize=7,
                label=model_name,
            )

            # Clean accuracy horizontal line
            if model_name in clean_accs:
                ax.axhline(
                    clean_accs[model_name],
                    color=color, linestyle=":", linewidth=1.0, alpha=0.6,
                )

        ax.set_xticks(range(len(sub) if sub else 0))
        ax.set_xticklabels(severities if sub else [], fontsize=9)
        ax.set_xlabel("Severity", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_title(display, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    for fmt in formats:
        p = output_stem.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        saved.append(p)

    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if len(args.model) != len(args.checkpoint):
        print(
            f"ERROR: --model ({len(args.model)}) and --checkpoint "
            f"({len(args.checkpoint)}) must have the same length."
        )
        sys.exit(1)

    config = load_config(args.config)
    set_seed(config.seed)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size  = args.batch_size  or config.batch_size
    num_workers = args.num_workers or config.num_workers
    output_dir  = _PROJECT_ROOT / args.output_dir

    print("=" * 65)
    print("  Corruption Robustness Evaluation")
    print("=" * 65)
    print(f"  Models      : {args.model}")
    print(f"  Split       : {args.split}")
    print(f"  Device      : {device}")
    print(f"  Corruptions : {args.corruptions or 'all'}")
    print()

    # ── Build validation dataset (shared across models) ───────────────
    print(f"  Loading {args.split} dataset from '{config.dataset_path}' …")
    try:
        val_dataset = build_val_dataset(config, split=args.split)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)
    print(f"  {len(val_dataset)} samples, {val_dataset.num_classes} classes.\n")

    class_names = val_dataset.classes

    # ── Clean eval DataLoader (standard transform, no corruption) ─────
    clean_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = (device.type == "cuda"),
    )

    # Results accumulation
    all_results: Dict[str, List[RobustnessResult]] = {}
    clean_accs:  Dict[str, float] = {}
    csv_path    = output_dir / "robustness" / "corruption_results.csv"

    for model_name, ckpt_path in zip(args.model, args.checkpoint):
        print(f"\n{'─'*55}")
        print(f"  Model: {model_name}  |  Checkpoint: {ckpt_path}")
        print(f"{'─'*55}")

        # ── Load model ────────────────────────────────────────────────
        model = create_model(
            model_name,
            num_classes = val_dataset.num_classes,
            pretrained  = False,     # weights come from checkpoint
        )
        model.model_name = model_name

        try:
            load_checkpoint(
                path    = ckpt_path,
                model   = model,
                device  = str(device),
                strict  = True,
            )
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}")
            continue

        model = model.to(device)
        model.eval()

        # ── Clean accuracy ────────────────────────────────────────────
        print("  Evaluating clean accuracy …", end="", flush=True)
        clean_acc, n_clean = evaluate_clean(model, clean_loader, device)
        clean_accs[model_name] = clean_acc
        print(f"  {clean_acc:.2f}%  (n={n_clean})")

        # ── Corruption evaluation ─────────────────────────────────────
        results = evaluate_model_on_corruptions(
            model            = model,
            dataset          = val_dataset,
            clean_accuracy   = clean_acc,
            model_name       = model_name,
            corruption_types = args.corruptions,     # None → all
            image_size       = config.image_size,
            batch_size       = batch_size,
            num_workers      = num_workers,
            device           = device,
        )

        all_results[model_name] = results

        # ── Save per-model CSV rows ───────────────────────────────────
        save_results_csv(csv_path, results)

    if not all_results:
        print("\n  No results produced.  Check model/checkpoint arguments.")
        sys.exit(1)

    # ── Print summary table ───────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Results Summary")
    print(f"{'='*65}")
    header = f"{'Model':<20} {'Corruption':<20} {'Severity':>9} {'Acc%':>7} {'RR':>7}"
    print(header)
    print("─" * len(header))
    for model_name, results in all_results.items():
        print(f"  {model_name:<18} (clean={clean_accs[model_name]:.2f}%)")
        for r in results:
            print(
                f"  {'':18} {r.corruption_type:<20} {r.severity:>9}  "
                f"{r.accuracy:>6.2f}%  {r.relative_robustness:>6.3f}"
            )

    print(f"\n  CSV saved → {csv_path}")

    # ── Robustness curves ─────────────────────────────────────────────
    plot_stem = output_dir / "plots" / "robustness_curves"
    saved_plots = plot_robustness_curves(
        all_results = all_results,
        output_stem = plot_stem,
        clean_accs  = clean_accs,
        formats     = ("png", "pdf"),
    )
    for p in saved_plots:
        print(f"  Plot saved  → {p}")

    print(f"\n{'='*65}")
    print("  Robustness evaluation complete.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
