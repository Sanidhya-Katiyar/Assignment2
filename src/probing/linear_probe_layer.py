"""
src/probing/linear_probe_layer.py
──────────────────────────────────
Lightweight linear classifier trained on pre-extracted feature arrays.

No DataLoader or torchvision transforms are required here — callers
pass raw NumPy arrays produced by
:func:`~src.probing.feature_extractor.extract_layer_features`.

Architecture::

    nn.Linear(input_dim, num_classes)  →  CrossEntropyLoss

The classifier is intentionally minimal so that accuracy measures the
quality of the *frozen* feature representation, not the capacity of the
probe head.

Public API
----------
``LinearProbeClassifier``  – the ``nn.Module`` probe head.
``train_linear_probe``     – fit a probe on (features, labels) arrays.
``evaluate_probe``         – compute accuracy on a held-out split.
``probe_features``         – end-to-end: split → train → evaluate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LinearProbeClassifier(nn.Module):
    """
    Single linear layer probe classifier.

    Args:
        input_dim:   Dimensionality of the input feature vector.
        num_classes: Number of output classes.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_linear_probe(
    features:    np.ndarray,
    labels:      np.ndarray,
    num_classes: int,
    epochs:      int   = 10,
    lr:          float = 1e-3,
    batch_size:  int   = 64,
    device:      Optional[torch.device] = None,
    weight_decay: float = 1e-4,
    verbose:     bool  = False,
) -> LinearProbeClassifier:
    """
    Train a linear probe on pre-extracted feature arrays.

    Args:
        features:     Float32 array of shape ``(N, D)``.
        labels:       Int64 array of shape  ``(N,)``.
        num_classes:  Number of output classes.
        epochs:       Training epochs (default 10).
        lr:           Adam learning rate (default 1e-3).
        batch_size:   Mini-batch size (default 64).
        device:       Torch device; defaults to CPU.
        weight_decay: L2 regularisation for Adam.
        verbose:      Print per-epoch loss when ``True``.

    Returns:
        Trained :class:`LinearProbeClassifier` on *device*.
    """
    if device is None:
        device = torch.device("cpu")

    input_dim = features.shape[1]

    # Convert to tensors
    x = torch.from_numpy(features).float().to(device)
    y = torch.from_numpy(labels).long().to(device)

    dataset   = TensorDataset(x, y)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    probe     = LinearProbeClassifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches  = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        if verbose:
            avg = epoch_loss / max(n_batches, 1)
            print(f"    probe epoch {epoch+1:2d}/{epochs}  loss={avg:.4f}")

    return probe


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_probe(
    probe:    LinearProbeClassifier,
    features: np.ndarray,
    labels:   np.ndarray,
    batch_size: int = 256,
    device:   Optional[torch.device] = None,
) -> float:
    """
    Compute top-1 accuracy of a trained probe on a feature split.

    Args:
        probe:      Trained :class:`LinearProbeClassifier`.
        features:   Float32 array ``(N, D)``.
        labels:     Int64 array ``(N,)``.
        batch_size: Evaluation batch size (default 256).
        device:     Torch device; defaults to probe's device.

    Returns:
        Top-1 accuracy as a float in ``[0, 1]``.
    """
    if device is None:
        device = next(probe.parameters()).device

    x = torch.from_numpy(features).float().to(device)
    y = torch.from_numpy(labels).long().to(device)

    dataset = TensorDataset(x, y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probe.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for xb, yb in loader:
            preds    = probe(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# End-to-end convenience function
# ---------------------------------------------------------------------------

def probe_features(
    features:    np.ndarray,
    labels:      np.ndarray,
    num_classes: int,
    train_frac:  float = 0.8,
    epochs:      int   = 10,
    lr:          float = 1e-3,
    batch_size:  int   = 64,
    seed:        int   = 42,
    device:      Optional[torch.device] = None,
    verbose:     bool  = False,
) -> Tuple[float, LinearProbeClassifier]:
    """
    Split features into train / eval, train a linear probe, and return accuracy.

    Uses a deterministic stratified-like split: for each class, the first
    ``train_frac`` fraction of samples go to training and the rest to
    evaluation.  This is simple, fast, and reproducible given a fixed seed.

    Args:
        features:    Float32 array ``(N, D)``.
        labels:      Int64 array ``(N,)``.
        num_classes: Number of output classes.
        train_frac:  Fraction of samples for training (default 0.8).
        epochs:      Probe training epochs (default 10).
        lr:          Adam learning rate (default 1e-3).
        batch_size:  Mini-batch size.
        seed:        NumPy random seed for the split.
        device:      Torch device.
        verbose:     Print training loss when ``True``.

    Returns:
        Tuple ``(accuracy, probe)`` where *accuracy* is in ``[0, 1]``.
    """
    rng = np.random.default_rng(seed)

    train_idx: list = []
    eval_idx:  list = []

    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        split   = max(1, int(len(cls_idx) * train_frac))
        train_idx.extend(cls_idx[:split].tolist())
        eval_idx.extend( cls_idx[split:].tolist())

    if not eval_idx:
        # Degenerate case: very few samples, use all for both
        eval_idx = train_idx

    train_idx_arr = np.array(train_idx)
    eval_idx_arr  = np.array(eval_idx)

    probe = train_linear_probe(
        features    = features[train_idx_arr],
        labels      = labels[train_idx_arr],
        num_classes = num_classes,
        epochs      = epochs,
        lr          = lr,
        batch_size  = batch_size,
        device      = device,
        verbose     = verbose,
    )

    acc = evaluate_probe(
        probe    = probe,
        features = features[eval_idx_arr],
        labels   = labels[eval_idx_arr],
        device   = device,
    )

    return acc, probe
