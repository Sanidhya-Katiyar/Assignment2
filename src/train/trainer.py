"""
src/train/trainer.py
─────────────────────
High-level ``Trainer`` class that orchestrates the full training lifecycle:

* epoch loop with early stopping
* per-epoch CSV logging
* best / last checkpoint saving
* optional fine-tune unfreeze schemes for ablation experiments
* selective-unfreeze helpers (gradient-probe importance ranking)

The Trainer itself is backend-agnostic — it delegates forward/backward
passes to :mod:`src.train.engine` and checkpoint I/O to
:mod:`src.train.utils_checkpoint`.
"""

from __future__ import annotations

import csv
import dataclasses
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.train.amp_utils    import make_scaler
from src.train.engine       import train_one_epoch, validate_one_epoch
from src.train.utils_checkpoint import (
    append_best_results,
    export_metadata,
    load_checkpoint,
    save_checkpoint,
)
from src.metrics.metrics        import make_optimizer, make_scheduler
from src.metrics.visualization  import plot_confusion_matrix, plot_train_val_curves


# ---------------------------------------------------------------------------
# TrainConfig: all hyper-parameters the Trainer needs
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainConfig:
    """
    Flat hyper-parameter bundle consumed by ``Trainer``.

    Every value has a sensible default so callers only need to override
    what they care about.

    Attributes:
        model_name:      Short name used for file naming (e.g. ``"resnet50"``).
        mode:            Experiment label (e.g. ``"linearprobe"``).
        seed:            Random seed for this run.
        epochs:          Maximum training epochs.
        lr:              Base / head learning rate.
        lr_backbone:     Backbone learning rate (used when backbone unfrozen).
        weight_decay:    L2 regularisation.
        momentum:        SGD momentum (ignored for AdamW).
        optimizer_name:  ``"adamw"`` or ``"sgd"``.
        scheduler_name:  ``"cosine"`` or ``"step"``.
        step_size:       StepLR step size (epochs).
        gamma:           StepLR decay factor.
        patience:        Early-stopping patience in epochs (0 = disabled).
        amp:             Enable mixed-precision (AMP).
        max_grad_norm:   Gradient clipping L2 norm.
        num_classes:     Number of output classes.
        output_dir:      Root output directory (``Path`` or ``str``).
        class_names:     Optional list of class-name strings for plots.
        device:          Torch device string (``"cuda"`` or ``"cpu"``).
    """

    model_name:     str   = "model"
    mode:           str   = "train"
    seed:           int   = 42
    epochs:         int   = 30
    lr:             float = 1e-3
    lr_backbone:    float = 1e-4
    weight_decay:   float = 1e-4
    momentum:       float = 0.9
    optimizer_name: str   = "adamw"
    scheduler_name: str   = "cosine"
    step_size:      int   = 10
    gamma:          float = 0.1
    patience:       int   = 10
    amp:            bool  = False
    max_grad_norm:  float = 1.0
    num_classes:    int   = 30
    output_dir:     str   = "outputs"
    class_names:    Optional[List[str]] = None
    device:         str   = "cpu"

    @classmethod
    def from_config(cls, cfg, **overrides) -> "TrainConfig":
        """
        Build a ``TrainConfig`` from a :class:`~src.utils.config.Config` object.

        Any *overrides* keyword arguments take precedence over config values.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        init = dict(
            seed          = cfg.seed,
            num_classes   = cfg.extra.get("num_classes", 30),
            amp           = cfg.extra.get("amp", False),
            epochs        = cfg.extra.get("epochs", 30),
            lr            = cfg.extra.get("lr", 1e-3),
            lr_backbone   = cfg.extra.get("lr_backbone", 1e-4),
            weight_decay  = cfg.extra.get("weight_decay", 1e-4),
            optimizer_name= cfg.extra.get("optimizer", "adamw"),
            scheduler_name= cfg.extra.get("scheduler", "cosine"),
            patience      = cfg.extra.get("patience", 10),
            max_grad_norm = cfg.extra.get("max_grad_norm", 1.0),
            output_dir    = "outputs",
            device        = device,
        )
        init.update(overrides)
        return cls(**init)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Orchestrates model training with logging, checkpointing, and early stopping.

    Args:
        model:       ``nn.Module`` to train (already on the correct device).
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        criterion:   Loss function.
        tcfg:        :class:`TrainConfig` hyper-parameter bundle.
        optimizer:   If ``None``, one is created from ``tcfg``.
        scheduler:   If ``None``, one is created from ``tcfg``.
        resume_path: Path to a checkpoint to resume from (optional).
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        criterion:    nn.Module,
        tcfg:         TrainConfig,
        optimizer:    Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        resume_path:  Optional[str | Path] = None,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.tcfg         = tcfg
        self.device       = torch.device(tcfg.device)

        # Optimizer
        if optimizer is None:
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = make_optimizer(
                trainable,
                optimizer_name = tcfg.optimizer_name,
                lr             = tcfg.lr,
                weight_decay   = tcfg.weight_decay,
                momentum       = tcfg.momentum,
            )
        self.optimizer = optimizer

        # Scheduler
        if scheduler is None:
            scheduler = make_scheduler(
                self.optimizer,
                scheduler_name = tcfg.scheduler_name,
                epochs         = tcfg.epochs,
                step_size      = tcfg.step_size,
                gamma          = tcfg.gamma,
            )
        self.scheduler = scheduler

        # AMP
        self.scaler = make_scaler(tcfg.amp and self.device.type == "cuda")

        # State
        self.start_epoch:   int   = 0
        self.best_val_acc:  float = 0.0
        self.best_epoch:    int   = 0
        self._no_improve:   int   = 0
        self._history:      List[Dict[str, Any]] = []

        # cudnn determinism for fixed input size
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Output paths
        root = Path(tcfg.output_dir)
        tag  = f"{tcfg.model_name}_{tcfg.mode}_{tcfg.seed}"
        self.ckpt_dir   = root / "checkpoints"
        self.log_dir    = root / "logs"
        self.plot_dir   = root / "plots"
        self.best_path  = self.ckpt_dir / f"{tag}_best.pth"
        self.last_path  = self.ckpt_dir / f"{tag}_last.pth"
        self.csv_path   = self.log_dir  / f"{tag}_epoch_metrics.csv"
        self.curve_stem = self.plot_dir / f"{tag}_train_val_curve"
        self.cm_path    = self.plot_dir / f"{tag}_confusion_matrix.png"
        for d in (self.ckpt_dir, self.log_dir, self.plot_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Optionally resume
        if resume_path is not None:
            self._resume(Path(resume_path))

        # Initialise CSV log
        self._init_csv()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resume(self, path: Path) -> None:
        ckpt = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler,
            device=str(self.device),
        )
        self.start_epoch  = ckpt["epoch"] + 1
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        self.best_epoch   = ckpt.get("epoch", 0)
        print(f"  Resumed from '{path}' (epoch {ckpt['epoch']}, "
              f"best_val_acc={self.best_val_acc:.2f}%)")

    def _config_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self.tcfg)

    def _init_csv(self) -> None:
        """Create CSV log with header (only if not already existing from resume)."""
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
                writer.writeheader()

    def _log_csv(self, row: Dict[str, Any]) -> None:
        with self.csv_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
            writer.writerow(row)

    def _save(self, path: Path, epoch: int) -> None:
        save_checkpoint(
            path        = path,
            model       = self.model,
            optimizer   = self.optimizer,
            scheduler   = self.scheduler,
            epoch       = epoch,
            best_val_acc= self.best_val_acc,
            seed        = self.tcfg.seed,
            config_dict = self._config_dict(),
        )

    # ------------------------------------------------------------------
    # Public: single-epoch methods
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch and return metrics dict."""
        return train_one_epoch(
            model         = self.model,
            loader        = self.train_loader,
            criterion     = self.criterion,
            optimizer     = self.optimizer,
            scaler        = self.scaler,
            device        = self.device,
            amp_enabled   = self.tcfg.amp,
            max_grad_norm = self.tcfg.max_grad_norm,
        )

    def validate_one_epoch(self) -> Dict[str, Any]:
        """Run one validation pass and return metrics dict."""
        return validate_one_epoch(
            model       = self.model,
            loader      = self.val_loader,
            criterion   = self.criterion,
            device      = self.device,
            num_classes = self.tcfg.num_classes,
            amp_enabled = self.tcfg.amp,
        )

    # ------------------------------------------------------------------
    # Public: full training run
    # ------------------------------------------------------------------

    def fit(self) -> Dict[str, Any]:
        """
        Run the full training loop from ``start_epoch`` to ``tcfg.epochs``.

        Returns a summary dict with keys:
        ``"best_val_acc"``, ``"best_epoch"``, ``"total_time"``,
        ``"history"`` (list of per-epoch dicts).
        """
        t_start = time.time()
        print(f"\n{'='*60}")
        print(f"  Training: {self.tcfg.model_name} | mode={self.tcfg.mode} | "
              f"seed={self.tcfg.seed} | device={self.device}")
        print(f"  epochs={self.tcfg.epochs}  lr={self.tcfg.lr}  "
              f"amp={self.tcfg.amp}  patience={self.tcfg.patience}")
        print(f"{'='*60}\n")

        last_cm = None

        for epoch in range(self.start_epoch, self.tcfg.epochs):
            # ── Train ──────────────────────────────────────────────────
            tr = self.train_one_epoch(epoch)
            va = self.validate_one_epoch()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            last_cm = va["confusion_matrix"]

            row: Dict[str, Any] = {
                "epoch":       epoch,
                "train_loss":  f"{tr['loss']:.6f}",
                "val_loss":    f"{va['loss']:.6f}",
                "train_acc":   f"{tr['acc1']:.4f}",
                "val_acc":     f"{va['acc1']:.4f}",
                "lr":          f"{current_lr:.8f}",
                "grad_norm":   f"{tr['grad_norm']:.6f}",
                "epoch_time":  f"{tr['epoch_time']:.1f}",
                "gpu_mem_max": f"{tr['gpu_mem_max_mb']:.0f}",
            }
            self._log_csv(row)
            self._history.append(row)

            improved = va["acc1"] > self.best_val_acc
            if improved:
                self.best_val_acc = va["acc1"]
                self.best_epoch   = epoch
                self._no_improve  = 0
                self._save(self.best_path, epoch)
            else:
                self._no_improve += 1

            self._save(self.last_path, epoch)

            # ── Console ───────────────────────────────────────────────
            flag = " ★" if improved else ""
            print(
                f"  Ep {epoch:3d}/{self.tcfg.epochs-1}  "
                f"loss={tr['loss']:.4f}/{va['loss']:.4f}  "
                f"acc={tr['acc1']:.2f}/{va['acc1']:.2f}%  "
                f"lr={current_lr:.2e}  gn={tr['grad_norm']:.3f}{flag}"
            )

            # ── Early stopping ─────────────────────────────────────────
            if self.tcfg.patience > 0 and self._no_improve >= self.tcfg.patience:
                print(f"\n  Early stop after {epoch+1} epochs "
                      f"(no improvement for {self.tcfg.patience} epochs).")
                break

        total_time = time.time() - t_start

        # ── Post-training artifacts ───────────────────────────────────
        self._write_final_artifacts(last_cm)

        # ── Append to global results CSV ──────────────────────────────
        append_best_results(
            csv_path        = Path(self.tcfg.output_dir) / "best_results.csv",
            model           = self.tcfg.model_name,
            mode            = self.tcfg.mode,
            seed            = self.tcfg.seed,
            best_val_acc    = self.best_val_acc,
            epoch_of_best   = self.best_epoch,
            runtime_seconds = total_time,
            checkpoint_path = str(self.best_path),
        )

        print(f"\n  Done.  best_val_acc={self.best_val_acc:.2f}% @ epoch {self.best_epoch}")
        print(f"  Total time: {total_time/60:.1f} min\n")

        return {
            "best_val_acc": self.best_val_acc,
            "best_epoch":   self.best_epoch,
            "total_time":   total_time,
            "history":      self._history,
        }

    # ------------------------------------------------------------------
    # Post-training artifact helpers
    # ------------------------------------------------------------------

    def _write_final_artifacts(self, last_cm) -> None:
        """Save training curves and confusion matrix after fit() completes."""
        if not self._history:
            return

        epochs     = [int(r["epoch"]) for r in self._history]
        tr_losses  = [float(r["train_loss"]) for r in self._history]
        val_losses = [float(r["val_loss"])   for r in self._history]
        tr_accs    = [float(r["train_acc"])  for r in self._history]
        val_accs   = [float(r["val_acc"])    for r in self._history]

        plot_train_val_curves(
            epochs, tr_losses, val_losses, tr_accs, val_accs,
            output_stem = self.curve_stem,
            title       = f"{self.tcfg.model_name} | {self.tcfg.mode} | seed={self.tcfg.seed}",
        )

        if last_cm is not None:
            plot_confusion_matrix(
                matrix      = last_cm,
                class_names = self.tcfg.class_names,
                output_path = self.cm_path,
                title       = (f"Confusion Matrix — {self.tcfg.model_name} "
                               f"({self.tcfg.mode})"),
            )


# ---------------------------------------------------------------------------
# CSV column order
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "epoch", "train_loss", "val_loss", "train_acc", "val_acc",
    "lr", "grad_norm", "epoch_time", "gpu_mem_max",
]


# ---------------------------------------------------------------------------
# Fine-tune unfreeze helpers
# ---------------------------------------------------------------------------

def unfreeze_last_n_blocks(model: nn.Module, n_blocks: int = 1) -> None:
    """
    Unfreeze the last *n_blocks* top-level children of *model*.

    The classification head (``model.fc`` / ``model.classifier``) is
    always unfrozen regardless.  This implements the "last-block" ablation.

    Args:
        model:    Any ``nn.Module`` (backbone + head).
        n_blocks: Number of top-level children to unfreeze from the end.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    children = list(model.named_children())
    # Always unfreeze the head (last child assumed to be classifier)
    head_name, head_module = children[-1]
    for p in head_module.parameters():
        p.requires_grad = True

    # Unfreeze additional blocks from the back of the backbone
    backbone_children = children[:-1]
    for name, module in backbone_children[-n_blocks:]:
        for p in module.parameters():
            p.requires_grad = True


def selective_unfreeze_by_fraction(
    model:          nn.Module,
    param_fraction: float = 0.2,
    strategy:       str   = "top_n",
    probe_loader:   Optional[object] = None,
    criterion:      Optional[nn.Module] = None,
    device:         Optional[torch.device] = None,
) -> None:
    """
    Unfreeze a subset of backbone parameters totalling at most *param_fraction*
    of the full parameter count.

    Two selection strategies:

    ``"top_n"``
        Select the fewest layers (by param count) from the *end* of the
        network that cumulatively reach *param_fraction*.  Efficient and
        deterministic.

    ``"gradient_probe"``
        Run a short forward/backward pass on *probe_loader*, rank layers by
        mean absolute gradient magnitude, and unfreeze the top-ranked layers
        up to *param_fraction*.  Requires *probe_loader*, *criterion*, and
        *device* to be provided.

    Args:
        model:          Model to mutate.
        param_fraction: Target fraction of total params to unfreeze (0–1).
        strategy:       ``"top_n"`` (default) or ``"gradient_probe"``.
        probe_loader:   DataLoader for gradient probe (only for ``"gradient_probe"``).
        criterion:      Loss function (only for ``"gradient_probe"``).
        device:         Compute device (only for ``"gradient_probe"``).
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    budget       = int(total_params * param_fraction)

    if strategy == "gradient_probe" and probe_loader is not None:
        _selective_by_gradient(model, budget, probe_loader, criterion, device)
    else:
        _selective_top_n(model, budget)


def _selective_top_n(model: nn.Module, budget: int) -> None:
    """Unfreeze layers from the end until *budget* params are active."""
    named_params = list(model.named_parameters())
    cumulative   = 0
    for name, param in reversed(named_params):
        if cumulative >= budget:
            break
        param.requires_grad = True
        cumulative += param.numel()


def _selective_by_gradient(
    model:        nn.Module,
    budget:       int,
    probe_loader: object,
    criterion:    nn.Module,
    device:       torch.device,
) -> None:
    """Unfreeze highest-gradient-magnitude layers up to *budget* params."""
    # Temporarily enable all gradients for the probe
    for p in model.parameters():
        p.requires_grad = True

    model.train()
    probe_batch = next(iter(probe_loader))
    inputs, targets = probe_batch
    inputs  = inputs.to(device)
    targets = targets.to(device)

    logits = model(inputs)
    loss   = criterion(logits, targets)
    loss.backward()

    # Rank parameters by mean absolute gradient
    ranked = sorted(
        [(name, p) for name, p in model.named_parameters() if p.grad is not None],
        key=lambda x: x[1].grad.abs().mean().item(),
        reverse=True,
    )

    # Re-freeze everything
    for p in model.parameters():
        p.requires_grad = False
        if p.grad is not None:
            p.grad = None

    # Unfreeze top-ranked within budget
    cumulative = 0
    for _, param in ranked:
        if cumulative >= budget:
            break
        param.requires_grad = True
        cumulative += param.numel()
