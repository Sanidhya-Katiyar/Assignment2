"""
Stratified train / val / test splitting utilities.

Splits are produced by stratified sampling so that each class is
proportionally represented in every partition.  All splits are fully
reproducible via a fixed random seed.
"""

from typing import Dict, List, Tuple

import numpy as np


def stratified_split(
    image_paths:  List[str],
    labels:       List[int],
    train_frac:   float = 0.70,
    val_frac:     float = 0.15,
    test_frac:    float = 0.15,
    seed:         int   = 42,
) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
]:
    """
    Split image paths and labels into stratified train / val / test sets.

    The function iterates over each class independently, shuffles the
    per-class samples with a fixed seed, and then slices according to the
    requested fractions.  This guarantees that every class is represented
    in every split proportionally.

    Args:
        image_paths: Flat list of image file paths (all classes).
        labels:      Integer label for each path (same order as image_paths).
        train_frac:  Fraction of data for training   (default 0.70).
        val_frac:    Fraction of data for validation (default 0.15).
        test_frac:   Fraction of data for test       (default 0.15).
        seed:        Random seed for reproducibility (default 42).

    Returns:
        Three (image_paths, labels) tuples for train, val, and test.

    Raises:
        ValueError: If the fractions do not sum to approximately 1.0.
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_frac + val_frac + test_frac must equal 1.0, got {total:.4f}."
        )

    rng = np.random.default_rng(seed)

    # Group indices by class label
    class_indices: Dict[int, List[int]] = {}
    for idx, lbl in enumerate(labels):
        class_indices.setdefault(lbl, []).append(idx)

    train_paths, train_labels = [], []
    val_paths,   val_labels   = [], []
    test_paths,  test_labels  = [], []

    for lbl, indices in sorted(class_indices.items()):
        indices_arr = np.array(indices)
        rng.shuffle(indices_arr)

        n        = len(indices_arr)
        n_train  = max(1, round(n * train_frac))
        n_val    = max(1, round(n * val_frac))
        # test gets the remainder to avoid rounding loss
        n_test   = n - n_train - n_val
        if n_test < 1:
            # Collapse: give 1 sample to test, adjust val
            n_test = 1
            n_val  = max(1, n - n_train - n_test)
            n_train = n - n_val - n_test

        train_idx = indices_arr[:n_train]
        val_idx   = indices_arr[n_train: n_train + n_val]
        test_idx  = indices_arr[n_train + n_val:]

        for i in train_idx:
            train_paths.append(image_paths[i])
            train_labels.append(labels[i])
        for i in val_idx:
            val_paths.append(image_paths[i])
            val_labels.append(labels[i])
        for i in test_idx:
            test_paths.append(image_paths[i])
            test_labels.append(labels[i])

    return (
        (train_paths, train_labels),
        (val_paths,   val_labels),
        (test_paths,  test_labels),
    )


def split_summary(
    train_labels: List[int],
    val_labels:   List[int],
    test_labels:  List[int],
    idx_to_class: Dict[int, str],
) -> str:
    """
    Build a human-readable summary of split sizes per class.

    Args:
        train_labels: Integer labels for training samples.
        val_labels:   Integer labels for validation samples.
        test_labels:  Integer labels for test samples.
        idx_to_class: Mapping from integer index to class name string.

    Returns:
        A formatted multi-line string suitable for printing to stdout.
    """
    all_classes = sorted(idx_to_class.keys())

    def _counts(lbls: List[int]) -> Dict[int, int]:
        c: Dict[int, int] = {k: 0 for k in all_classes}
        for l in lbls:
            c[l] += 1
        return c

    tr = _counts(train_labels)
    va = _counts(val_labels)
    te = _counts(test_labels)

    header = f"{'Class':<25} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}"
    sep    = "-" * len(header)
    lines  = [header, sep]

    for cls_idx in all_classes:
        cls_name = idx_to_class[cls_idx]
        t, v, e  = tr[cls_idx], va[cls_idx], te[cls_idx]
        lines.append(f"{cls_name:<25} {t:>6} {v:>6} {e:>6} {t+v+e:>7}")

    lines.append(sep)
    tot_tr = sum(tr.values())
    tot_va = sum(va.values())
    tot_te = sum(te.values())
    lines.append(
        f"{'TOTAL':<25} {tot_tr:>6} {tot_va:>6} {tot_te:>6} "
        f"{tot_tr + tot_va + tot_te:>7}"
    )
    return "\n".join(lines)
