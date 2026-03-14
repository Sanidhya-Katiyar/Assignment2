"""
src/train/utils_checkpoint.py
──────────────────────────────
Save and load training checkpoints that capture the complete state needed
to resume an experiment identically, including:

* model weights
* optimizer state
* LR-scheduler state
* current epoch
* best validation accuracy so far
* seed used for this run
* a frozen copy of the config dict
* CPU / GPU RNG states for full reproducibility on resume

Public API
----------
``save_checkpoint``  – write a checkpoint ``.pth`` file.
``load_checkpoint``  – restore all states from a ``.pth`` file.
``export_metadata``  – write a human-readable ``.json`` summary alongside.
``append_best_results`` – append one row to ``outputs/best_results.csv``.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    path:           Path | str,
    model:          nn.Module,
    optimizer:      torch.optim.Optimizer,
    scheduler,                               # any LR scheduler or None
    epoch:          int,
    best_val_acc:   float,
    seed:           int,
    config_dict:    Dict[str, Any],
    extra:          Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persist a complete training checkpoint.

    Args:
        path:          File path to write (extension ``.pth``).
        model:         Model whose ``state_dict`` is saved.
        optimizer:     Optimizer whose ``state_dict`` is saved.
        scheduler:     LR scheduler (``None`` if not used).
        epoch:         Current epoch index (0-based).
        best_val_acc:  Best validation accuracy seen so far.
        seed:          Master random seed of this run.
        config_dict:   Serialisable copy of the config parameters.
        extra:         Optional dict of additional items to store.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "epoch":               epoch,
        "best_val_acc":        best_val_acc,
        "seed":                seed,
        "config":              config_dict,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        # RNG states for deterministic resume
        "rng_cpu":             torch.get_rng_state(),
        "rng_cuda":            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if extra:
        payload.update(extra)

    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    path:       Path | str,
    model:      nn.Module,
    optimizer:  Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device:     str = "cpu",
    strict:     bool = True,
) -> Dict[str, Any]:
    """
    Restore model (and optionally optimizer / scheduler) from a checkpoint.

    Args:
        path:       Path to the ``.pth`` checkpoint file.
        model:      Model to load weights into (mutated in-place).
        optimizer:  If provided, optimizer state is also restored.
        scheduler:  If provided, scheduler state is also restored.
        device:     Target device for weight loading.
        strict:     Whether to use strict ``load_state_dict`` (default True).

    Returns:
        The full checkpoint dict (contains epoch, best_val_acc, seed, config,
        and any extra keys that were stored).

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{path}'. "
            "Pass --resume with a valid checkpoint path."
        )

    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # Restore RNG states when resuming on the same hardware configuration
    if ckpt.get("rng_cpu") is not None:
        try:
            torch.set_rng_state(ckpt["rng_cpu"])
        except RuntimeError:
            pass  # Different hardware — silently skip

    if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        except RuntimeError:
            pass

    return ckpt


# ---------------------------------------------------------------------------
# JSON metadata sidecar
# ---------------------------------------------------------------------------

def export_metadata(
    checkpoint_path: Path | str,
    extra:           Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write a ``.json`` summary alongside a checkpoint file.

    Reads the checkpoint and extracts non-tensor fields (epoch, best_val_acc,
    seed, config) to produce a human-readable JSON sidecar.

    Args:
        checkpoint_path: Path to the existing ``.pth`` checkpoint.
        extra:           Any additional key/value pairs to include.

    Returns:
        Path of the written JSON file.
    """
    ckpt_path = Path(checkpoint_path)
    ckpt      = torch.load(ckpt_path, map_location="cpu")

    meta: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "epoch":      ckpt.get("epoch"),
        "best_val_acc": ckpt.get("best_val_acc"),
        "seed":       ckpt.get("seed"),
        "config":     ckpt.get("config"),
        "saved_at":   time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        meta.update(extra)

    json_path = ckpt_path.with_suffix(".json")
    with json_path.open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    return json_path


# ---------------------------------------------------------------------------
# best_results.csv aggregator
# ---------------------------------------------------------------------------

_RESULTS_COLUMNS = [
    "model", "mode", "seed", "best_val_acc", "epoch_of_best",
    "runtime_seconds", "checkpoint_path", "timestamp",
]


def append_best_results(
    csv_path:        Path | str,
    model:           str,
    mode:            str,
    seed:            int,
    best_val_acc:    float,
    epoch_of_best:   int,
    runtime_seconds: float,
    checkpoint_path: str,
) -> None:
    """
    Append a one-row summary to ``outputs/best_results.csv``.

    The file is created with a header row if it does not yet exist.

    Args:
        csv_path:        Path to the results aggregation CSV.
        model:           Model name string (e.g. ``"resnet50"``).
        mode:            Experiment mode (e.g. ``"linearprobe"``).
        seed:            Random seed used for this run.
        best_val_acc:    Best validation accuracy achieved.
        epoch_of_best:   Epoch index at which best accuracy was reached.
        runtime_seconds: Total wall-clock training time in seconds.
        checkpoint_path: Absolute path to the best checkpoint file.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "model":            model,
            "mode":             mode,
            "seed":             seed,
            "best_val_acc":     f"{best_val_acc:.6f}",
            "epoch_of_best":    epoch_of_best,
            "runtime_seconds":  f"{runtime_seconds:.1f}",
            "checkpoint_path":  checkpoint_path,
            "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
