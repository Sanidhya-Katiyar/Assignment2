#!/usr/bin/env python3
"""
scripts/create_visualization_subset.py
───────────────────────────────────────
Build a fixed 30 × 30 subset (30 classes × 30 images each) used for
PCA / t-SNE feature-space visualisations.

Because the same random seed is always used, every experiment in the
project will compare features over the *exact same* image set.

Usage
-----
    python scripts/create_visualization_subset.py --config configs/baseline.yaml

Output
------
    outputs/visualization_subset.csv
    Columns: path, label_idx, class_name
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.aid_dataset import discover_dataset
from src.utils.config          import load_config
from src.utils.seed            import set_seed

# ── Subset parameters ────────────────────────────────────────────────
NUM_CLASSES        = 30
IMAGES_PER_CLASS   = 30
OUTPUT_FILENAME    = "visualization_subset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible 30 × 30 visualization subset "
            "and save image paths to outputs/visualization_subset.csv."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to a YAML config file (e.g. configs/baseline.yaml).",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=IMAGES_PER_CLASS,
        metavar="N",
        help=f"Images to sample per class (default: {IMAGES_PER_CLASS}).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        metavar="N",
        help=f"Number of classes to include (default: {NUM_CLASSES}).",
    )
    return parser.parse_args()


def sample_subset(
    image_paths:       List[str],
    labels:            List[int],
    class_to_idx:      Dict[str, int],
    num_classes:       int,
    images_per_class:  int,
    seed:              int,
) -> List[dict]:
    """
    Sample *images_per_class* images from *num_classes* classes.

    If a class has fewer than *images_per_class* images, all available
    images for that class are included and a warning is printed.

    Args:
        image_paths:      All image paths in the full dataset.
        labels:           Corresponding integer labels.
        class_to_idx:     Class name → integer index mapping.
        num_classes:      How many classes to sample.
        images_per_class: How many images per class to include.
        seed:             Random seed.

    Returns:
        List of dicts with keys: 'path', 'label_idx', 'class_name'.
    """
    rng           = np.random.default_rng(seed)
    idx_to_class  = {v: k for k, v in class_to_idx.items()}

    # Group paths by class
    class_buckets: Dict[int, List[str]] = {}
    for path, lbl in zip(image_paths, labels):
        class_buckets.setdefault(lbl, []).append(path)

    all_class_ids = sorted(class_buckets.keys())

    if len(all_class_ids) < num_classes:
        raise ValueError(
            f"Requested {num_classes} classes but only "
            f"{len(all_class_ids)} classes are present in the dataset."
        )

    # Sample num_classes without replacement (sorted for determinism)
    selected_class_ids = sorted(
        rng.choice(all_class_ids, size=num_classes, replace=False).tolist()
    )

    records = []
    for cls_id in selected_class_ids:
        cls_name = idx_to_class[cls_id]
        paths    = sorted(class_buckets[cls_id])

        if len(paths) < images_per_class:
            print(
                f"  WARNING: Class '{cls_name}' has only {len(paths)} images "
                f"(requested {images_per_class}). Using all available."
            )
            sampled = paths
        else:
            indices = rng.choice(len(paths), size=images_per_class, replace=False)
            sampled = [paths[i] for i in sorted(indices)]

        for p in sampled:
            records.append({
                "path":       p,
                "label_idx":  cls_id,
                "class_name": cls_name,
            })

    return records


def save_subset_csv(output_path: Path, records: List[dict]) -> None:
    """Write subset records to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label_idx", "class_name"])
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    print("=" * 60)
    print("Visualization Subset Creator")
    print("=" * 60)
    print(f"  Dataset path  : {config.dataset_path}")
    print(f"  Classes       : {args.num_classes}")
    print(f"  Images/class  : {args.images_per_class}")
    print(f"  Seed          : {config.seed}")
    print()

    set_seed(config.seed)

    # ── Discover dataset ─────────────────────────────────────────────
    print(f"Scanning dataset at: {config.dataset_path}")
    try:
        image_paths, labels, class_to_idx = discover_dataset(config.dataset_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)

    print(f"  Found {len(class_to_idx)} classes, {len(image_paths)} images.\n")

    # ── Sample subset ────────────────────────────────────────────────
    print("Sampling subset …")
    try:
        records = sample_subset(
            image_paths      = image_paths,
            labels           = labels,
            class_to_idx     = class_to_idx,
            num_classes      = args.num_classes,
            images_per_class = args.images_per_class,
            seed             = config.seed,
        )
    except ValueError as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)

    # ── Save CSV ─────────────────────────────────────────────────────
    output_path = _PROJECT_ROOT / "outputs" / OUTPUT_FILENAME
    save_subset_csv(output_path, records)

    total = len(records)
    unique_classes = len({r["class_name"] for r in records})
    print(f"\n  Saved {total} image paths ({unique_classes} classes) → {output_path}")
    print()

    # ── Brief breakdown ──────────────────────────────────────────────
    print("Classes included in subset:")
    from collections import Counter
    counts = Counter(r["class_name"] for r in records)
    for cls_name in sorted(counts):
        print(f"  {cls_name:<25} {counts[cls_name]:>4} images")

    print()
    print("Done.  Use this CSV to ensure consistent image sets across all experiments.")
    print("=" * 60)


if __name__ == "__main__":
    main()
