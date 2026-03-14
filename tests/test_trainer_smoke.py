"""
tests/test_trainer_smoke.py
────────────────────────────
Fast, dependency-light smoke tests that verify the training pipeline
works end-to-end on a tiny synthetic dataset (no real images required).

Tests complete in < 30 s on CPU.

Run with:
    python -m pytest tests/test_trainer_smoke.py -v
or simply:
    python tests/test_trainer_smoke.py
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.metrics.metrics         import (
    AverageMeter,
    ConfusionMatrixAccumulator,
    accuracy,
    compute_gradient_norm,
    make_optimizer,
    make_scheduler,
)
from src.train.amp_utils         import make_scaler
from src.train.engine            import train_one_epoch, validate_one_epoch
from src.train.trainer           import TrainConfig, Trainer
from src.train.utils_checkpoint  import (
    append_best_results,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Documentation-only constants (not referenced inside function bodies —
# all functions use literals so they are immune to .pyc constant-table bugs).
NUM_CLASSES = 5    # output classes
INPUT_DIM   = 32   # synthetic feature width   (≠ NUM_CLASSES, ≠ HIDDEN)
HIDDEN      = 64   # MLP hidden width          (≠ NUM_CLASSES, ≠ INPUT_DIM)
N_TRAIN     = 40
N_VAL       = 10
BATCH_SIZE  = 8


def _make_tiny_model(num_classes: int = 5) -> nn.Module:
    """
    Minimal two-layer MLP for smoke testing.

    Architecture (literals, not module constants):
        Linear(32, 64) -> ReLU -> Linear(64, num_classes=5)

    Forward shapes:
        input  : (batch, 32)
        hidden : (batch, 64)
        output : (batch, 5)
    """
    model = nn.Sequential(
        nn.Linear(32, 64),          # in=32, out=64
        nn.ReLU(),
        nn.Linear(64, num_classes), # in=64, out=5
    )
    model.fc = model[-1]  # type: ignore[attr-defined]
    return model


def _make_loaders():
    """
    Synthetic DataLoaders with 2-D float tensors.

    Uses literals throughout (not module-level constants) so the function
    is unaffected by any cached bytecode that might have stale constant values.

    Shapes: x (N, 32)  y (N,) labels in [0, 5).
    """
    x_train = torch.randn(40, 32)        # 40 samples, 32 features
    y_train = torch.randint(0, 5, (40,)) # labels in [0, 5)
    x_val   = torch.randn(10, 32)
    y_val   = torch.randint(0, 5, (10,))

    train_ds = TensorDataset(x_train, y_train)
    val_ds   = TensorDataset(x_val,   y_val)
    train_ld = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=8, shuffle=False)
    return train_ld, val_ld


# ---------------------------------------------------------------------------
# Test: AverageMeter
# ---------------------------------------------------------------------------

def test_average_meter():
    m = AverageMeter("loss")
    m.update(1.0, 1)
    m.update(3.0, 1)
    assert abs(m.avg - 2.0) < 1e-6, f"Expected 2.0, got {m.avg}"
    m.update(0.0, 2)  # n=2 so weight 2
    expected = (1.0 + 3.0 + 0.0 * 2) / 4
    assert abs(m.avg - expected) < 1e-6
    print("  [PASS] test_average_meter")


# ---------------------------------------------------------------------------
# Test: accuracy
# ---------------------------------------------------------------------------

def test_accuracy():
    logits  = torch.tensor([[2.0, 0.5, 0.1], [0.1, 0.2, 3.0]])
    targets = torch.tensor([0, 2])
    top1 = accuracy(logits, targets, top_k=(1,))[0]
    assert abs(top1 - 100.0) < 1e-4, f"Expected 100.0, got {top1}"
    print("  [PASS] test_accuracy")


# ---------------------------------------------------------------------------
# Test: ConfusionMatrixAccumulator
# ---------------------------------------------------------------------------

def test_confusion_matrix():
    cm = ConfusionMatrixAccumulator(num_classes=3)
    logits  = torch.tensor([[5.0, 0.0, 0.0],
                             [0.0, 5.0, 0.0],
                             [0.0, 0.0, 5.0]])
    targets = torch.tensor([0, 1, 2])
    cm.update(logits, targets)
    mat = cm.compute()
    assert mat.shape == (3, 3)
    assert mat[0, 0] == 1 and mat[1, 1] == 1 and mat[2, 2] == 1
    print("  [PASS] test_confusion_matrix")


# ---------------------------------------------------------------------------
# Test: train/val engine functions
# ---------------------------------------------------------------------------

def test_engine_train_val():
    model     = _make_tiny_model()
    tr_ld, va_ld = _make_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(list(model.parameters()), lr=1e-3)
    scaler    = make_scaler(enabled=False)
    device    = torch.device("cpu")

    tr_stats = train_one_epoch(
        model, tr_ld, criterion, optimizer, scaler, device, amp_enabled=False
    )
    assert "loss"    in tr_stats
    assert "acc1"    in tr_stats
    assert "grad_norm" in tr_stats
    assert tr_stats["loss"] >= 0
    assert 0 <= tr_stats["acc1"] <= 100

    va_stats = validate_one_epoch(
        model, va_ld, criterion, device, num_classes=NUM_CLASSES, amp_enabled=False
    )
    assert "loss" in va_stats
    assert "confusion_matrix" in va_stats
    assert va_stats["confusion_matrix"].shape == (NUM_CLASSES, NUM_CLASSES)
    print("  [PASS] test_engine_train_val")


# ---------------------------------------------------------------------------
# Test: Trainer.fit for 1 epoch saves checkpoint and CSV log
# ---------------------------------------------------------------------------

def test_trainer_fit_saves_artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        model    = _make_tiny_model()
        tr_ld, va_ld = _make_loaders()

        tcfg = TrainConfig(
            model_name   = "tinymlp",
            mode         = "test",
            seed         = 0,
            epochs       = 2,
            lr           = 1e-3,
            patience     = 0,    # disable early stopping
            amp          = False,
            num_classes  = NUM_CLASSES,
            output_dir   = str(tmp),
            device       = "cpu",
        )

        trainer = Trainer(
            model        = model,
            train_loader = tr_ld,
            val_loader   = va_ld,
            criterion    = nn.CrossEntropyLoss(),
            tcfg         = tcfg,
        )

        summary = trainer.fit()

        # Check summary keys
        assert "best_val_acc"  in summary
        assert "best_epoch"    in summary
        assert "total_time"    in summary
        assert 0.0 <= summary["best_val_acc"] <= 100.0

        # Checkpoint exists
        assert trainer.best_path.exists(), f"Best checkpoint missing: {trainer.best_path}"
        assert trainer.last_path.exists(), f"Last checkpoint missing: {trainer.last_path}"

        # CSV log has rows
        assert trainer.csv_path.exists()
        with trainer.csv_path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2, f"Expected 2 epoch rows, got {len(rows)}"
        assert "val_acc" in rows[0]

        # Plot files exist
        png_path = trainer.curve_stem.with_suffix(".png")
        assert png_path.exists(), f"Curve PNG missing: {png_path}"

        print("  [PASS] test_trainer_fit_saves_artifacts")


# ---------------------------------------------------------------------------
# Test: checkpoint save / load round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test.pth"
        model     = _make_tiny_model()
        optimizer = make_optimizer(list(model.parameters()), lr=1e-3)

        save_checkpoint(
            path         = ckpt_path,
            model        = model,
            optimizer    = optimizer,
            scheduler    = None,
            epoch        = 5,
            best_val_acc = 73.5,
            seed         = 42,
            config_dict  = {"lr": 1e-3},
        )
        assert ckpt_path.exists()

        model2    = _make_tiny_model()
        optimizer2 = make_optimizer(list(model2.parameters()), lr=1e-3)
        ckpt = load_checkpoint(ckpt_path, model2, optimizer2, device="cpu")

        assert ckpt["epoch"]        == 5
        assert abs(ckpt["best_val_acc"] - 73.5) < 1e-6
        assert ckpt["seed"]         == 42

        # Weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Weight mismatch after load!"

        print("  [PASS] test_checkpoint_round_trip")


# ---------------------------------------------------------------------------
# Test: append_best_results writes CSV
# ---------------------------------------------------------------------------

def test_append_best_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "best_results.csv"
        append_best_results(csv_path, "resnet50", "linearprobe", 42, 87.3, 10, 120.0, "/tmp/ckpt.pth")
        append_best_results(csv_path, "resnet50", "linearprobe", 43, 88.1, 12, 118.0, "/tmp/ckpt2.pth")

        with csv_path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2
        assert float(rows[0]["best_val_acc"]) == pytest_approx(87.3, tol=1e-3)
        print("  [PASS] test_append_best_results")


def pytest_approx(val, tol=1e-3):
    """Simple approx helper for running without pytest."""
    class _Approx:
        def __init__(self, val, tol):
            self.val = val
            self.tol = tol
        def __eq__(self, other):
            return abs(other - self.val) < self.tol
    return _Approx(val, tol)


# ---------------------------------------------------------------------------
# Test runner (usable without pytest)
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        test_average_meter,
        test_accuracy,
        test_confusion_matrix,
        test_engine_train_val,
        test_trainer_fit_saves_artifacts,
        test_checkpoint_round_trip,
        test_append_best_results,
    ]
    print("=" * 55)
    print("  Smoke Tests — AID Transfer Learning Block 3")
    print("=" * 55)
    passed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {fn.__name__}: {exc}")

    print("=" * 55)
    print(f"  {passed}/{len(tests)} tests passed.")
    print("=" * 55)
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
