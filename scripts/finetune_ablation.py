#!/usr/bin/env python3
"""
scripts/finetune_ablation.py
─────────────────────────────
Run a single fine-tuning experiment with one of three unfreeze schemes:

``last-block``    – unfreeze the last N backbone blocks (default N=1).
``selective``     – unfreeze layers totalling ≤ X% of total parameters.
``full``          – unfreeze the entire network.

All schemes use the same hyperparameters (from config or CLI overrides)
so that only the unfreeze scheme differs between ablation runs.

Usage
-----
    # Last-block fine-tune (1 block):
    python scripts/finetune_ablation.py \\
        --config configs/baseline.yaml --model efficientnet_b0 --scheme last-block

    # Selective unfreeze up to 20 % of params (top-N strategy):
    python scripts/finetune_ablation.py \\
        --config configs/baseline.yaml --model efficientnet_b0 \\
        --scheme selective --param-fraction 0.2

    # Full fine-tune:
    python scripts/finetune_ablation.py \\
        --config configs/baseline.yaml --model resnet50 --scheme full

Output (in outputs/)
--------------------
Same artifact layout as train_linear_probe.py with mode tag replaced by
the unfreeze scheme label (e.g. ``resnet50_lastblock_42_best.pth``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn

from src.datasets.aid_dataset import discover_dataset
from src.datasets.dataloader  import get_dataloaders
from src.models.backbone_utils import unfreeze_all
from src.models.model_factory  import create_model
from src.train.trainer import (
    TrainConfig,
    Trainer,
    selective_unfreeze_by_fraction,
    unfreeze_last_n_blocks,
)
from src.utils.config import load_config
from src.utils.seed   import set_seed


SCHEMES = ["last-block", "selective", "full"]
SCHEME_MODES = {
    "last-block": "lastblock",
    "selective":  "selective",
    "full":       "finetune",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune ablation with configurable unfreeze scheme."
    )
    p.add_argument("--config",         required=True,  help="Path to YAML config.")
    p.add_argument("--model",          default="resnet50",
                   choices=["resnet50", "efficientnet_b0", "convnext_tiny"])
    p.add_argument("--scheme",         default="last-block", choices=SCHEMES,
                   help="Unfreeze scheme.")
    p.add_argument("--n-blocks",       type=int,   default=1,
                   help="Number of backbone blocks to unfreeze (last-block scheme).")
    p.add_argument("--param-fraction", type=float, default=0.2,
                   help="Fraction of params to unfreeze (selective scheme).")
    p.add_argument("--gradient-probe", action="store_true",
                   help="Use gradient-probe ranking for selective scheme.")
    p.add_argument("--seed",           type=int,   default=None)
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--lr-backbone",    type=float, default=None)
    p.add_argument("--batch-size",     type=int,   default=None)
    p.add_argument("--amp",            action="store_true")
    p.add_argument("--resume",         default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Unfreeze dispatcher
# ---------------------------------------------------------------------------

def apply_unfreeze_scheme(
    model:          nn.Module,
    scheme:         str,
    n_blocks:       int,
    param_fraction: float,
    use_gradient_probe: bool,
    probe_loader,
    criterion:      nn.Module,
    device:         torch.device,
) -> None:
    """Apply the requested unfreeze scheme to *model* in-place."""
    if scheme == "full":
        unfreeze_all(model)

    elif scheme == "last-block":
        unfreeze_last_n_blocks(model, n_blocks=n_blocks)

    elif scheme == "selective":
        strategy = "gradient_probe" if use_gradient_probe else "top_n"
        selective_unfreeze_by_fraction(
            model          = model,
            param_fraction = param_fraction,
            strategy       = strategy,
            probe_loader   = probe_loader if use_gradient_probe else None,
            criterion      = criterion    if use_gradient_probe else None,
            device         = device       if use_gradient_probe else None,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    seed       = args.seed       if args.seed       is not None else config.seed
    batch_size = args.batch_size if args.batch_size is not None else config.batch_size
    if args.batch_size:
        config.batch_size = batch_size

    set_seed(seed)

    mode = SCHEME_MODES[args.scheme]

    print("=" * 65)
    print(f"  Fine-tune Ablation | model={args.model} | scheme={args.scheme} | seed={seed}")
    print("=" * 65)

    # ── Dataloaders ───────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(config)

    # ── Discover classes ──────────────────────────────────────────────
    _, _, class_to_idx = discover_dataset(config.dataset_path)
    class_names  = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    num_classes  = len(class_to_idx)
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build model ───────────────────────────────────────────────────
    model = create_model(args.model, num_classes=num_classes, pretrained=True)
    model.model_name = args.model
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # ── Apply unfreeze scheme ─────────────────────────────────────────
    apply_unfreeze_scheme(
        model               = model,
        scheme              = args.scheme,
        n_blocks            = args.n_blocks,
        param_fraction      = args.param_fraction,
        use_gradient_probe  = args.gradient_probe,
        probe_loader        = train_loader,
        criterion           = criterion,
        device              = device,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)\n")

    # ── Build TrainConfig ─────────────────────────────────────────────
    lr_effective = args.lr or config.extra.get("lr_finetune", 1e-4)
    lr_backbone  = args.lr_backbone or config.extra.get("lr_backbone", 1e-5)

    tcfg = TrainConfig.from_config(
        config,
        model_name  = args.model,
        mode        = mode,
        seed        = seed,
        amp         = args.amp or config.extra.get("amp", False),
        epochs      = args.epochs or config.extra.get("epochs", 30),
        lr          = lr_effective,
        lr_backbone = lr_backbone,
        num_classes = num_classes,
        class_names = class_names,
    )

    # ── Differential learning rates for partial unfreeze ──────────────
    from src.models.backbone_utils import get_backbone_params, get_classifier_params
    from src.metrics.metrics        import make_optimizer, make_scheduler

    head_params     = [p for p in get_classifier_params(model) if p.requires_grad]
    backbone_params = [p for p in get_backbone_params(model)   if p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_effective})

    optimizer = make_optimizer(
        param_groups or [p for p in model.parameters() if p.requires_grad],
        optimizer_name = tcfg.optimizer_name,
        lr             = lr_effective,
        weight_decay   = tcfg.weight_decay,
    )
    scheduler = make_scheduler(
        optimizer,
        scheduler_name = tcfg.scheduler_name,
        epochs         = tcfg.epochs,
    )

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        criterion    = criterion,
        tcfg         = tcfg,
        optimizer    = optimizer,
        scheduler    = scheduler,
        resume_path  = args.resume,
    )

    summary = trainer.fit()

    print("\n  === Summary ===")
    print(f"  Scheme       : {args.scheme}")
    print(f"  Best val acc : {summary['best_val_acc']:.2f}%")
    print(f"  Best epoch   : {summary['best_epoch']}")
    print(f"  Runtime      : {summary['total_time']/60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    main()
