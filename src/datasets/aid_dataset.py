"""
AID Remote Sensing Dataset loader.

Provides a PyTorch Dataset class for the AID dataset with support for
train/val/test splits, stratified sampling, and standard ImageNet transforms.
"""

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """
    Return torchvision transforms for the given dataset split.

    Training split applies light augmentation; val/test are deterministic.

    Args:
        split:      One of 'train', 'val', or 'test'.
        image_size: Target spatial resolution (height == width).

    Returns:
        A composed torchvision transform pipeline.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:  # val or test — deterministic
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class AIDDataset(Dataset):
    """
    PyTorch Dataset for the AID (Aerial Image Dataset) remote sensing benchmark.

    Expected on-disk layout::

        <root>/
            airport/
                airport_001.jpg
                ...
            bareland/
                bareland_001.jpg
                ...
            ...   (30 classes total)

    Args:
        image_paths:  List of absolute paths to image files.
        labels:       Integer class label for each image (aligned with image_paths).
        class_to_idx: Mapping from class name string to integer index.
        transform:    Optional transform applied to each PIL image.
    """

    def __init__(
        self,
        image_paths:  List[str],
        labels:       List[int],
        class_to_idx: dict,
        transform:    Optional[Callable] = None,
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError(
                f"image_paths ({len(image_paths)}) and labels ({len(labels)}) "
                "must have the same length."
            )
        self.image_paths  = image_paths
        self.labels       = labels
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform    = transform

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple:
        """
        Load and return a (image_tensor, label) pair.

        Args:
            index: Sample index.

        Returns:
            Tuple of (transformed image tensor, integer label).
        """
        img_path = self.image_paths[index]
        label    = self.labels[index]

        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError) as exc:
            raise RuntimeError(
                f"Could not load image at '{img_path}'. "
                "Please verify the dataset path in your config."
            ) from exc

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def classes(self) -> List[str]:
        """Sorted list of class name strings."""
        return sorted(self.class_to_idx.keys())

    @property
    def num_classes(self) -> int:
        """Total number of distinct classes."""
        return len(self.class_to_idx)

    def class_counts(self) -> dict:
        """Return a dict mapping class name → sample count for this split."""
        counts: dict = {cls: 0 for cls in self.classes}
        for lbl in self.labels:
            counts[self.idx_to_class[lbl]] += 1
        return counts


# ------------------------------------------------------------------
# Dataset discovery
# ------------------------------------------------------------------

def discover_dataset(root: str) -> Tuple[List[str], List[int], dict]:
    """
    Walk the AID directory tree and collect (path, label) pairs.

    Args:
        root: Path to the top-level AID directory.

    Returns:
        Tuple of (image_paths, labels, class_to_idx).

    Raises:
        FileNotFoundError: If *root* does not exist.
        ValueError:        If no class sub-directories are found.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"Dataset root '{root}' does not exist. "
            "Please set 'dataset_path' correctly in your config file."
        )

    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(
            f"No class sub-directories found under '{root}'. "
            "Expected layout: <root>/<class_name>/<image_files>."
        )

    class_to_idx     = {d.name: idx for idx, d in enumerate(class_dirs)}
    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    image_paths: List[str] = []
    labels:      List[int] = []

    for class_dir in class_dirs:
        idx = class_to_idx[class_dir.name]
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in valid_extensions:
                image_paths.append(str(img_file))
                labels.append(idx)

    if not image_paths:
        raise ValueError(
            f"No images with extensions {valid_extensions} found under '{root}'."
        )

    return image_paths, labels, class_to_idx
