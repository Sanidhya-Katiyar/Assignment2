"""
src/evaluation/robustness_eval.py
──────────────────────────────────
Evaluate a trained model under each supported image corruption at multiple
severity levels and collect per-level accuracy, corruption error, and
relative robustness.

Design
------
Corruption is applied on-the-fly at the PIL level, *before* the standard
ImageNet normalisation transform.  This means the existing
:class:`~src.datasets.aid_dataset.AIDDataset` can be reused with a custom
``transform`` that intercepts the raw PIL image, corrupts it, and then
runs the standard resize + normalise pipeline.

Public API
----------
``evaluate_model_on_corruptions``  – evaluate one model across all severities
                                     of one or all corruptions; returns a
                                     structured ``RobustnessResults`` dict.
``evaluate_clean``                 – evaluate without any corruption (baseline).
``build_corrupted_transform``      – compose corruption + normalisation pipeline.
``RobustnessResult``               – typed dataclass for a single (corruption,
                                     severity) measurement.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

from src.datasets.aid_dataset    import (
    AIDDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.evaluation.corruptions  import (
    CORRUPTION_REGISTRY,
    get_corruption_fn,
)
from src.metrics.metrics         import AverageMeter, accuracy


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RobustnessResult:
    """
    Measurement for one (model, corruption, severity) combination.

    Attributes:
        model_name:          Short model identifier.
        corruption_type:     Name of the corruption (e.g. ``"gaussian_noise"``).
        severity:            Severity parameter value (e.g. ``0.1`` or ``9``).
        accuracy:            Top-1 accuracy in % on the corrupted validation set.
        corruption_error:    ``1 - accuracy / 100`` (fraction mis-classified).
        relative_robustness: ``accuracy / clean_accuracy`` — ratio vs. clean baseline.
        clean_accuracy:      Clean baseline accuracy used to compute relative robustness.
        n_samples:           Number of validation samples evaluated.
        eval_time_s:         Wall-clock time for this evaluation pass in seconds.
    """

    model_name:          str
    corruption_type:     str
    severity:            float
    accuracy:            float
    corruption_error:    float
    relative_robustness: float
    clean_accuracy:      float
    n_samples:           int
    eval_time_s:         float


# ---------------------------------------------------------------------------
# Corrupted transform builder
# ---------------------------------------------------------------------------

def build_corrupted_transform(
    corruption_fn,
    severity:   float,
    image_size: int = 224,
) -> transforms.Compose:
    """
    Build a torchvision transform pipeline that applies a corruption first.

    The returned pipeline is:
        PIL → corrupt → Resize(image_size) → ToTensor → Normalize(ImageNet)

    Args:
        corruption_fn: Callable ``(PIL.Image, severity) → PIL.Image``.
        severity:      Severity parameter passed to *corruption_fn*.
        image_size:    Target spatial resolution (default 224).

    Returns:
        A ``transforms.Compose`` instance ready for use as a Dataset transform.
    """

    class _CorruptTransform:
        """Callable that applies the corruption then the standard eval pipeline."""

        def __init__(self, fn, sev: float, sz: int) -> None:
            self._fn  = fn
            self._sev = sev
            self._post = transforms.Compose([
                transforms.Resize((sz, sz)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        def __call__(self, pil_img):
            corrupted = self._fn(pil_img, self._sev)
            return self._post(corrupted)

    return _CorruptTransform(corruption_fn, severity, image_size)


# ---------------------------------------------------------------------------
# Clean evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_clean(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
) -> Tuple[float, int]:
    """
    Evaluate *model* on *loader* without any corruption.

    Args:
        model:  Model in eval mode (caller's responsibility).
        loader: DataLoader — should use the standard eval transform.
        device: Compute device.

    Returns:
        Tuple ``(top1_accuracy_percent, n_samples)``.
    """
    model.eval()
    meter = AverageMeter("clean_acc")

    for inputs, targets in loader:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits  = model(inputs)
        top1    = accuracy(logits, targets, top_k=(1,))[0]
        meter.update(top1, inputs.size(0))

    return meter.avg, meter.count


# ---------------------------------------------------------------------------
# Single-corruption evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_one_severity(
    model:        nn.Module,
    dataset_ref:  AIDDataset,
    corruption_fn,
    severity:     float,
    image_size:   int,
    batch_size:   int,
    num_workers:  int,
    device:       torch.device,
) -> Tuple[float, int, float]:
    """
    Evaluate *model* on *dataset_ref* with one (corruption, severity) pair.

    Creates a lightweight copy of the dataset with a replaced transform so
    no data is duplicated on disk.

    Returns:
        Tuple ``(top1_acc_percent, n_samples, eval_time_s)``.
    """
    corrupt_transform = build_corrupted_transform(
        corruption_fn, severity, image_size
    )

    # Shallow-copy the dataset and replace the transform only
    corrupted_dataset = AIDDataset(
        image_paths  = dataset_ref.image_paths,
        labels       = dataset_ref.labels,
        class_to_idx = dataset_ref.class_to_idx,
        transform    = corrupt_transform,
    )

    loader = DataLoader(
        corrupted_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = (device.type == "cuda"),
    )

    model.eval()
    meter = AverageMeter("acc")
    t0    = time.time()

    for inputs, targets in loader:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits  = model(inputs)
        top1    = accuracy(logits, targets, top_k=(1,))[0]
        meter.update(top1, inputs.size(0))

    return meter.avg, meter.count, time.time() - t0


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def evaluate_model_on_corruptions(
    model:            nn.Module,
    dataset:          AIDDataset,
    clean_accuracy:   float,
    model_name:       str                        = "model",
    corruption_types: Optional[Sequence[str]]    = None,
    severity_map:     Optional[Dict[str, List]]  = None,
    image_size:       int                        = 224,
    batch_size:       int                        = 32,
    num_workers:      int                        = 4,
    device:           Optional[torch.device]     = None,
) -> List[RobustnessResult]:
    """
    Evaluate *model* across corruptions and severity levels.

    For each (corruption_type, severity) pair the function:

    1. Builds a corrupted transform (corruption applied at PIL level).
    2. Wraps the existing *dataset* with that transform.
    3. Runs a full forward pass and records top-1 accuracy.
    4. Computes corruption error and relative robustness.

    Args:
        model:            Trained ``nn.Module`` (already on *device*).
        dataset:          Validation :class:`~src.datasets.aid_dataset.AIDDataset`
                          (used as the image source; its transform will be
                          replaced per severity — the original is never modified).
        clean_accuracy:   Baseline accuracy on the uncorrupted validation set
                          (used to compute relative robustness).
        model_name:       Short model identifier for result labelling.
        corruption_types: List of corruption names to evaluate.  Defaults to
                          all registered corruptions when ``None``.
        severity_map:     Dict mapping each corruption name to its list of
                          severity values.  Defaults to the canonical set
                          when ``None``::

                              {
                                "gaussian_noise":   [0.05, 0.1, 0.2],
                                "motion_blur":      [5, 9],
                                "brightness_shift": [0.5, 1.5],
                              }

        image_size:       Resize target for the evaluation transform.
        batch_size:       Batch size for the corrupted evaluation loader.
        num_workers:      DataLoader workers.
        device:           Torch device (defaults to ``cuda`` if available).

    Returns:
        List of :class:`RobustnessResult` — one entry per
        (corruption_type, severity) pair, sorted by corruption then severity.

    Example::

        results = evaluate_model_on_corruptions(
            model, val_dataset, clean_accuracy=91.3,
            model_name="resnet50",
        )
        for r in results:
            print(r.corruption_type, r.severity, f"{r.accuracy:.2f}%")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default severity map
    if severity_map is None:
        severity_map = {
            "gaussian_noise":   [0.05, 0.1, 0.2],
            "motion_blur":      [5, 9],
            "brightness_shift": [0.5, 1.5],
        }

    # Default corruption list
    if corruption_types is None:
        corruption_types = list(CORRUPTION_REGISTRY.keys())

    results: List[RobustnessResult] = []

    for ctype in corruption_types:
        corruption_fn = get_corruption_fn(ctype)
        severities    = severity_map.get(ctype, [])

        if not severities:
            print(f"  [warn] No severities configured for '{ctype}' — skipping.")
            continue

        print(f"\n  Corruption: {ctype}")
        for sev in severities:
            print(f"    severity={sev} …", end="", flush=True)

            acc, n_samples, t = _evaluate_one_severity(
                model        = model,
                dataset_ref  = dataset,
                corruption_fn= corruption_fn,
                severity     = sev,
                image_size   = image_size,
                batch_size   = batch_size,
                num_workers  = num_workers,
                device       = device,
            )

            corr_error    = 1.0 - acc / 100.0
            rel_robustness = acc / clean_accuracy if clean_accuracy > 0 else 0.0

            print(f"  acc={acc:.2f}%  CE={corr_error:.4f}  RR={rel_robustness:.4f}  [{t:.1f}s]")

            results.append(RobustnessResult(
                model_name          = model_name,
                corruption_type     = ctype,
                severity            = float(sev),
                accuracy            = acc,
                corruption_error    = corr_error,
                relative_robustness = rel_robustness,
                clean_accuracy      = clean_accuracy,
                n_samples           = n_samples,
                eval_time_s         = t,
            ))

    return results
