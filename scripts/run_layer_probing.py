#!/usr/bin/env python3
"""
scripts/run_layer_probing.py
─────────────────────────────
Layer-wise linear probing: extract intermediate features at three
representative depths (early / middle / final) and measure how linearly
separable they are.

Usage
-----
    # Single model, single checkpoint:
    python scripts/run_layer_probing.py \\
        --config configs/baseline.yaml \\
        --model resnet50 \\
        --checkpoint outputs/checkpoints/resnet50_linearprobe_42_best.pth

    # All three architectures in one sweep:
    python scripts/run_layer_probing.py \\
        --config configs/baseline.yaml \\
        --model resnet50 efficientnet_b0 convnext_tiny \\
        --checkpoint \\
            outputs/checkpoints/resnet50_linearprobe_42_best.pth \\
            outputs/checkpoints/efficientnet_b0_linearprobe_42_best.pth \\
            outputs/checkpoints/convnext_tiny_linearprobe_42_best.pth

    # Use ImageNet weights only (no fine-tuned checkpoint):
    python scripts/run_layer_probing.py \\
        --config configs/baseline.yaml \\
        --model resnet50

Output files
------------
    outputs/probing/layer_probe_results.csv
    outputs/plots/layer_accuracy_vs_depth.png
    outputs/plots/layer_accuracy_vs_depth.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import csv
import torch
from torch.utils.data import DataLoader

from src.datasets.aid_dataset    import AIDDataset, discover_dataset, get_transforms
from src.models.model_factory    import create_model
from src.probing.feature_extractor import LAYER_REGISTRY, DEPTH_ORDER
from src.probing.probe_runner    import (
    ProbeResult,
    load_probe_csv,
    plot_depth_accuracy,
    run_probes,
    save_probe_csv,
)
from src.utils.config            import load_config
from src.utils.seed              import set_seed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Layer-wise linear probing of trained backbones."
    )
    p.add_argument(
        "--config", required=True, metavar="PATH",
        help="Path to YAML config (e.g. configs/baseline.yaml).",
    )
    p.add_argument(
        "--model", nargs="+",
        default=["resnet50"],
        choices=["resnet50", "efficientnet_b0", "convnext_tiny"],
        help="One or more backbone names to probe.",
    )
    p.add_argument(
        "--checkpoint", nargs="*", default=None, metavar="PATH",
        help=(
            "Checkpoint path(s) aligned with --model.  "
            "If omitted, ImageNet-pretrained weights are used.  "
            "Pass a single path to reuse it for all models."
        ),
    )
    p.add_argument(
        "--layers", nargs="+", default=None,
        choices=["early", "middle", "final"],
        help="Subset of depth tags to probe (default: all three).",
    )
    p.add_argument(
        "--probe-epochs",  type=int,   default=10,   help="Probe training epochs.")
    p.add_argument(
        "--probe-lr",      type=float, default=1e-3, help="Probe Adam learning rate.")
    p.add_argument(
        "--train-frac",    type=float, default=0.8,
        help="Fraction of visualization subset for probe training.",
    )
    p.add_argument(
        "--seed",          type=int,   default=None, help="Random seed.")
    p.add_argument(
        "--num-workers",   type=int,   default=None)
    p.add_argument(
        "--cache-features", action="store_true",
        help="Save/load extracted features to disk to speed up reruns.",
    )
    p.add_argument(
        "--overwrite",     action="store_true",
        help="Overwrite outputs/probing/layer_probe_results.csv instead of appending.",
    )
    p.add_argument(
        "--verbose",       action="store_true",
        help="Print per-epoch probe loss.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visualization-subset DataLoader
# ---------------------------------------------------------------------------

def build_viz_loader(
    config,
    num_workers: int = 0,
) -> Optional[DataLoader]:
    """
    Build a DataLoader from the pre-generated visualization_subset.csv.

    Returns ``None`` (with a warning) if the CSV does not exist.
    """
    viz_csv = _PROJECT_ROOT / "outputs" / "visualization_subset.csv"
    if not viz_csv.exists():
        print(
            "  [warn] outputs/visualization_subset.csv not found.\n"
            "         Run scripts/create_visualization_subset.py first.\n"
            "         Falling back to validation split …"
        )
        return None

    paths:  List[str] = []
    labels: List[int] = []
    with viz_csv.open() as fh:
        for row in csv.DictReader(fh):
            paths.append(row["path"])
            labels.append(int(row["label_idx"]))

    _, _, class_to_idx = discover_dataset(config.dataset_path)
    transform = get_transforms("test", config.image_size)
    dataset   = AIDDataset(paths, labels, class_to_idx, transform=transform)
    loader    = DataLoader(
        dataset,
        batch_size  = config.batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = False,
    )
    print(f"  Visualization subset: {len(dataset)} samples loaded.")
    return loader


def build_val_loader_fallback(config) -> DataLoader:
    """Fallback: use the validation DataLoader from the standard split."""
    from src.datasets.dataloader import get_dataloaders
    _, val_loader, _ = get_dataloaders(config)
    return val_loader


# ---------------------------------------------------------------------------
# Checkpoint alignment helper
# ---------------------------------------------------------------------------

def _resolve_checkpoints(
    models:       List[str],
    checkpoints:  Optional[List[str]],
) -> Dict[str, Optional[str]]:
    """Map each model name to its checkpoint path (or None)."""
    if checkpoints is None:
        return {m: None for m in models}

    if len(checkpoints) == 1:
        # Single checkpoint reused for all models
        return {m: checkpoints[0] for m in models}

    if len(checkpoints) != len(models):
        raise ValueError(
            f"Number of --checkpoint paths ({len(checkpoints)}) must be 1 "
            f"or equal to the number of --model names ({len(models)})."
        )

    return dict(zip(models, checkpoints))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    seed        = args.seed        if args.seed        is not None else config.seed
    num_workers = args.num_workers if args.num_workers is not None else config.num_workers
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    print("=" * 65)
    print("  Layer-wise Probing")
    print("=" * 65)
    print(f"  Models  : {args.model}")
    print(f"  Layers  : {args.layers or DEPTH_ORDER}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.probe_epochs}  LR: {args.probe_lr}")
    print()

    # ── Visualisation-subset loader ───────────────────────────────────
    viz_loader = build_viz_loader(config, num_workers=num_workers)
    if viz_loader is None:
        print("  Using validation split as fallback …")
        viz_loader = build_val_loader_fallback(config)

    # ── Discover class count ──────────────────────────────────────────
    _, _, class_to_idx = discover_dataset(config.dataset_path)
    num_classes        = len(class_to_idx)

    # ── Checkpoint alignment ──────────────────────────────────────────
    ckpt_map = _resolve_checkpoints(args.model, args.checkpoint)

    # ── Feature cache dir ─────────────────────────────────────────────
    cache_dir: Optional[Path] = None
    if args.cache_features:
        cache_dir = _PROJECT_ROOT / "outputs" / "probing" / "feature_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Run probes for each model ─────────────────────────────────────
    all_results: List[ProbeResult] = []

    for model_name in args.model:
        print(f"\n── {model_name.upper()} ──────────────────────────────────")

        model = create_model(model_name, num_classes=num_classes, pretrained=True)
        model.model_name = model_name  # needed by backbone_utils / checkpoint loader

        results = run_probes(
            model            = model,
            model_name       = model_name,
            dataloader       = viz_loader,
            num_classes      = num_classes,
            device           = device,
            checkpoint_path  = ckpt_map[model_name],
            depth_tags       = args.layers,
            probe_epochs     = args.probe_epochs,
            probe_lr         = args.probe_lr,
            train_frac       = args.train_frac,
            seed             = seed,
            verbose          = args.verbose,
            feature_cache_dir= cache_dir,
        )
        all_results.extend(results)

    if not all_results:
        print("\n  No probe results produced.  Check your config and checkpoints.")
        sys.exit(1)

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = _PROJECT_ROOT / "outputs" / "probing" / "layer_probe_results.csv"
    saved    = save_probe_csv(
        all_results, csv_path, append=(not args.overwrite)
    )
    print(f"\n  Results saved → {saved}")

    # ── Print summary table ───────────────────────────────────────────
    _print_summary_table(all_results)

    # ── Plot ──────────────────────────────────────────────────────────
    # Load full CSV (may include results from previous runs for multi-model plot)
    try:
        plot_data = load_probe_csv(csv_path)
    except FileNotFoundError:
        plot_data = all_results

    plot_stem  = _PROJECT_ROOT / "outputs" / "plots" / "layer_accuracy_vs_depth"
    plot_paths = plot_depth_accuracy(
        results     = plot_data,
        output_stem = plot_stem,
        title       = "Layer-wise Probe Accuracy — AID Dataset",
    )
    for pp in plot_paths:
        print(f"  Plot saved → {pp}")

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def _print_summary_table(results: List[ProbeResult]) -> None:
    """Print a formatted accuracy table to stdout."""
    print()
    header = f"{'Model':<22} {'Layer':<10} {'Accuracy':>10}  {'Feat dim':>10}"
    sep    = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"  {r.model_name:<20} {r.layer_tag:<10} "
            f"{r.accuracy*100:>8.2f}%  {r.feature_dim:>10}"
        )
    print(sep)


if __name__ == "__main__":
    main()
