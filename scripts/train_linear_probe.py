#!/usr/bin/env python3
"""
scripts/train_linear_probe.py
──────────────────────────────
Run a single linear-probe experiment: freeze the backbone, train only the
classification head, save all artifacts.

Usage
-----
    python scripts/train_linear_probe.py \\
        --config configs/baseline.yaml \\
        --model resnet50 \\
        --seed 42

CLI overrides
-------------
Most config values can be overridden directly:

    --epochs 30  --lr 1e-3  --batch-size 64  --amp  --resume path/to/ckpt.pth

Output (in outputs/)
--------------------
    checkpoints/<model>_linearprobe_<seed>_best.pth
    checkpoints/<model>_linearprobe_<seed>_last.pth
    logs/<model>_linearprobe_<seed>_epoch_metrics.csv
    plots/<model>_linearprobe_<seed>_train_val_curve.png/.pdf
    plots/<model>_linearprobe_<seed>_confusion_matrix.png
    features/<model>_linearprobe_<seed>_epoch<N>.npz
    best_results.csv  (appended)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.aid_dataset import discover_dataset
from src.datasets.dataloader  import get_dataloaders
from src.train.linear_probe   import TrainConfig, run_linear_probe
from src.utils.config         import load_config
from src.utils.seed           import set_seed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probe on AID dataset.")
    p.add_argument("--config",      required=True,   help="Path to YAML config.")
    p.add_argument("--model",       default="resnet50",
                   choices=["resnet50", "efficientnet_b0", "convnext_tiny"],
                   help="Backbone architecture.")
    p.add_argument("--seed",        type=int, default=None,  help="Random seed.")
    p.add_argument("--epochs",      type=int, default=None,  help="Override epochs.")
    p.add_argument("--lr",          type=float, default=None,help="Learning rate.")
    p.add_argument("--batch-size",  type=int, default=None,  help="Mini-batch size.")
    p.add_argument("--amp",         action="store_true",     help="Enable AMP.")
    p.add_argument("--resume",      default=None,            help="Checkpoint to resume.")
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visualization subset helper
# ---------------------------------------------------------------------------

def _load_viz_subset(config, num_classes: int):
    """
    Load the visualization subset CSV (if it exists) and return
    (viz_loader, viz_image_paths).  Returns (None, None) gracefully if the
    file is absent.
    """
    viz_csv = _PROJECT_ROOT / "outputs" / "visualization_subset.csv"
    if not viz_csv.exists():
        print("  [info] visualization_subset.csv not found — skipping feature snapshot.")
        return None, None

    import torch
    from torch.utils.data import DataLoader
    from src.datasets.aid_dataset import AIDDataset, get_transforms, discover_dataset

    # Read paths from CSV
    paths, labels = [], []
    with viz_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            paths.append(row["path"])
            labels.append(int(row["label_idx"]))

    _, _, class_to_idx = discover_dataset(config.dataset_path)
    transform = get_transforms("test", config.image_size)
    dataset   = AIDDataset(paths, labels, class_to_idx, transform=transform)
    loader    = DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=0, pin_memory=False)
    return loader, paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    # Apply CLI overrides to config
    seed       = args.seed       if args.seed       is not None else config.seed
    batch_size = args.batch_size if args.batch_size is not None else config.batch_size
    num_workers= args.num_workers if args.num_workers is not None else config.num_workers
    if args.batch_size:
        config.batch_size   = batch_size
    if args.num_workers:
        config.num_workers  = num_workers

    set_seed(seed)

    print("=" * 60)
    print(f"  Linear Probe | model={args.model} | seed={seed}")
    print("=" * 60)

    # ── Dataloaders ───────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(config)

    # ── Discover class names ──────────────────────────────────────────
    _, _, class_to_idx = discover_dataset(config.dataset_path)
    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

    # ── Visualization subset ──────────────────────────────────────────
    viz_loader, viz_paths = _load_viz_subset(config, num_classes=len(class_to_idx))

    # ── TrainConfig ───────────────────────────────────────────────────
    tcfg = TrainConfig.from_config(
        config,
        model_name  = args.model,
        mode        = "linearprobe",
        seed        = seed,
        amp         = args.amp or config.extra.get("amp", False),
        epochs      = args.epochs      or config.extra.get("epochs", 30),
        lr          = args.lr          or config.extra.get("lr_probe", 1e-3),
        num_classes = len(class_to_idx),
        class_names = class_names,
    )

    # ── Run ───────────────────────────────────────────────────────────
    summary = run_linear_probe(
        model_name      = args.model,
        tcfg            = tcfg,
        train_loader    = train_loader,
        val_loader      = val_loader,
        viz_loader      = viz_loader,
        viz_image_paths = viz_paths,
        num_classes     = len(class_to_idx),
        resume_path     = args.resume,
    )

    print("\n  === Summary ===")
    print(f"  Best val acc : {summary['best_val_acc']:.2f}%")
    print(f"  Best epoch   : {summary['best_epoch']}")
    print(f"  Runtime      : {summary['total_time']/60:.1f} min")
    if summary.get("feature_snapshot_path"):
        print(f"  Features     : {summary['feature_snapshot_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
