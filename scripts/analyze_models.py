#!/usr/bin/env python3
"""
scripts/analyze_models.py
──────────────────────────
Instantiate every supported backbone, compute complexity metrics, print a
comparison table, and save results to ``outputs/model_stats.csv``.

Usage
-----
    # From the project root:
    python scripts/analyze_models.py --config configs/baseline.yaml

    # Override number of classes (e.g. for a different dataset):
    python scripts/analyze_models.py --config configs/baseline.yaml --num-classes 30

    # Also report stats after freezing the backbone (linear-probe setup):
    python scripts/analyze_models.py --config configs/baseline.yaml --show-frozen

Output files
------------
    outputs/model_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

# ── Path setup ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.backbone_utils import freeze_backbone, trainable_summary, unfreeze_all
from src.models.model_factory  import SUPPORTED_MODELS, create_model
from src.models.model_stats    import ModelStats, format_stats_table, get_model_stats
from src.utils.config          import load_config
from src.utils.seed            import set_seed

# ── Display names ─────────────────────────────────────────────────────────────
_DISPLAY_NAMES: dict[str, str] = {
    "resnet50":        "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny":   "ConvNeXt-Tiny",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load all supported backbones, compute complexity metrics, "
            "and save a comparison table to outputs/model_stats.csv."
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
        "--num-classes",
        type=int,
        default=None,
        metavar="N",
        help="Override the number of output classes (default: from config or 30).",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load ImageNet pretrained weights (default: True).",
    )
    parser.add_argument(
        "--show-frozen",
        action="store_true",
        default=False,
        help=(
            "Also show trainable parameter count after freezing the backbone "
            "(linear-probe setup)."
        ),
    )
    return parser.parse_args()


# ── CSV helpers ───────────────────────────────────────────────────────────────

def save_csv(output_path: Path, stats_list: List[ModelStats]) -> None:
    """Write model statistics to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "total_params",
        "total_params_fmt",
        "trainable_params",
        "trainable_params_fmt",
        "flops",
        "flops_fmt",
        "flops_source",
        "input_size",
    ]
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for s in stats_list:
            writer.writerow({
                "model_name":           s.model_name,
                "total_params":         s.total_params,
                "total_params_fmt":     s.total_params_m,
                "trainable_params":     s.trainable_params,
                "trainable_params_fmt": s.trainable_params_m,
                "flops":                s.flops if s.flops is not None else "",
                "flops_fmt":            s.flops_str,
                "flops_source":         s.flops_source,
                "input_size":           "x".join(str(d) for d in s.input_size),
            })


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    set_seed(config.seed)

    num_classes = args.num_classes if args.num_classes is not None else 30
    input_size  = (3, config.image_size, config.image_size)

    print("=" * 65)
    print("Model Analysis — AID Transfer Learning")
    print("=" * 65)
    print(f"  Input size   : {input_size}")
    print(f"  Num classes  : {num_classes}")
    print(f"  Pretrained   : {args.pretrained}")
    print(f"  Models       : {', '.join(SUPPORTED_MODELS)}")
    print()

    stats_list: List[ModelStats] = []

    for model_key in SUPPORTED_MODELS:
        display = _DISPLAY_NAMES.get(model_key, model_key)
        print(f"  Loading {display} …", end="", flush=True)

        try:
            model = create_model(model_key, num_classes=num_classes, pretrained=args.pretrained)
            # Stamp the name so backbone_utils can find the head
            model.model_name = model_key  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        stats = get_model_stats(model, model_name=display, input_size=input_size)
        stats_list.append(stats)
        print(f"  done  ({stats.total_params_m} params, {stats.flops_str} MACs)")

        # Optional frozen-backbone stats
        if args.show_frozen:
            freeze_backbone(model)
            print(f"    ↳ frozen:  {trainable_summary(model)}")
            unfreeze_all(model)

    if not stats_list:
        print("\nERROR: No models could be loaded. Check your environment.")
        sys.exit(1)

    # ── Print table ──────────────────────────────────────────────────────────
    print()
    print(format_stats_table(stats_list))
    print()
    print(f"  FLOPs source: {stats_list[0].flops_source}")
    print()

    # ── Save CSV ─────────────────────────────────────────────────────────────
    output_path = _PROJECT_ROOT / "outputs" / "model_stats.csv"
    save_csv(output_path, stats_list)
    print(f"  Saved → {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
