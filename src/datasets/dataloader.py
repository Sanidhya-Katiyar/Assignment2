"""
DataLoader factory utilities.

Builds PyTorch DataLoader objects for the AID dataset from a
configuration object produced by :mod:`src.utils.config`.
"""

from typing import Tuple

from torch.utils.data import DataLoader

from src.datasets.aid_dataset import AIDDataset, discover_dataset, get_transforms
from src.datasets.split_utils  import stratified_split
from src.utils.seed             import set_seed


def get_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return train, val, and test DataLoaders for the AID dataset.

    The function:
      1. Discovers all images under ``config.dataset_path``.
      2. Performs a reproducible stratified split.
      3. Wraps each split in an :class:`~src.datasets.aid_dataset.AIDDataset`.
      4. Returns three configured :class:`~torch.utils.data.DataLoader` objects.

    Args:
        config: Configuration namespace / object with at minimum the
                following attributes:

                * ``dataset_path``  – path to AID root directory
                * ``image_size``    – spatial resolution (int)
                * ``batch_size``    – mini-batch size (int)
                * ``num_workers``   – DataLoader worker processes (int)
                * ``seed``          – random seed (int)
                * ``train_split``   – fraction for training (float)
                * ``val_split``     – fraction for validation (float)
                * ``test_split``    – fraction for test (float)

    Returns:
        Tuple ``(train_loader, val_loader, test_loader)``.
    """
    set_seed(config.seed)

    # ------------------------------------------------------------------
    # 1. Discover dataset
    # ------------------------------------------------------------------
    image_paths, labels, class_to_idx = discover_dataset(config.dataset_path)

    # ------------------------------------------------------------------
    # 2. Stratified split
    # ------------------------------------------------------------------
    (train_paths, train_labels), \
    (val_paths,   val_labels),   \
    (test_paths,  test_labels) = stratified_split(
        image_paths = image_paths,
        labels      = labels,
        train_frac  = config.train_split,
        val_frac    = config.val_split,
        test_frac   = config.test_split,
        seed        = config.seed,
    )

    # ------------------------------------------------------------------
    # 3. Build datasets
    # ------------------------------------------------------------------
    train_dataset = AIDDataset(
        image_paths  = train_paths,
        labels       = train_labels,
        class_to_idx = class_to_idx,
        transform    = get_transforms("train", config.image_size),
    )
    val_dataset = AIDDataset(
        image_paths  = val_paths,
        labels       = val_labels,
        class_to_idx = class_to_idx,
        transform    = get_transforms("val", config.image_size),
    )
    test_dataset = AIDDataset(
        image_paths  = test_paths,
        labels       = test_labels,
        class_to_idx = class_to_idx,
        transform    = get_transforms("test", config.image_size),
    )

    # ------------------------------------------------------------------
    # 4. Build DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config.batch_size,
        shuffle     = False,
        num_workers = config.num_workers,
        pin_memory  = True,
        drop_last   = False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = config.batch_size,
        shuffle     = False,
        num_workers = config.num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    return train_loader, val_loader, test_loader
