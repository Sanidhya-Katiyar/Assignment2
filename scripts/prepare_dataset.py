#!/usr/bin/env python3
"""
scripts/prepare_dataset.py
──────────────────────────
Verify the AID dataset, generate stratified train/val/test splits, print
dataset statistics, and save split manifests to the outputs/ directory.

Usage
-----
    python scripts/prepare_dataset.py --config configs/baseline.yaml

Output files (written to outputs/)
------------------------------------
    outputs/train_split.csv
    outputs/val_split.csv
    outputs/test_split.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.aid_dataset import discover_dataset
from src.datasets.split_utils  import split_summary, stratified_split
from src.utils.config           import load_config
from src.utils.seed             import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify dataset and generate reproducible train/val/test splits."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to a YAML config file (e.g. configs/baseline.yaml).",
    )
    return parser.parse_args()


def save_split_csv(
    output_path: Path,
    image_paths: list,
    labels: list,
    idx_to_class: dict,
) -> None:
    """Write a split manifest CSV with columns: path, label_idx, class_name."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "label_idx", "class_name"])
        for path, lbl in zip(image_paths, labels):
            writer.writerow([path, lbl, idx_to_class[lbl]])
    print(f"  Saved → {output_path}  ({len(image_paths)} samples)")


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    print("=" * 60)
    print("AID Dataset Preparation")
    print("=" * 60)
    print(config)
    print()

    # Set global seed
    set_seed(config.seed)

    # ── 1. Discover dataset ───────────────────────────────────────────
    print(f"[1/3]  Scanning dataset at: {config.dataset_path}")
    try:
        image_paths, labels, class_to_idx = discover_dataset(config.dataset_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"  Found {len(class_to_idx)} classes, {len(image_paths)} total images.")
    print()

    # ── 2. Stratified split ───────────────────────────────────────────
    print("[2/3]  Generating stratified splits …")
    (train_paths, train_labels), \
    (val_paths,   val_labels),   \
    (test_paths,  test_labels)  = stratified_split(
        image_paths = image_paths,
        labels      = labels,
        train_frac  = config.train_split,
        val_frac    = config.val_split,
        test_frac   = config.test_split,
        seed        = config.seed,
    )
    print()

    # ── 3. Print per-class statistics ────────────────────────────────
    print("[3/3]  Split statistics")
    print()
    print(split_summary(train_labels, val_labels, test_labels, idx_to_class))
    print()

    # ── 4. Save manifests ────────────────────────────────────────────
    output_dir = _PROJECT_ROOT / "outputs"
    print("Saving split manifests …")
    save_split_csv(output_dir / "train_split.csv", train_paths, train_labels, idx_to_class)
    save_split_csv(output_dir / "val_split.csv",   val_paths,   val_labels,   idx_to_class)
    save_split_csv(output_dir / "test_split.csv",  test_paths,  test_labels,  idx_to_class)
    print()
    print("Done.  Splits are fully reproducible with seed =", config.seed)
    print("=" * 60)


if __name__ == "__main__":
    main()
