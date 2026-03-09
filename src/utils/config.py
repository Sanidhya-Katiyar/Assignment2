"""
YAML-based configuration system.

Loads a YAML config file and returns a typed :class:`Config` dataclass.
All experiment parameters flow through this object — no hardcoded values
elsewhere in the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    """
    Experiment configuration.

    All path values are stored as strings so they can be passed directly to
    :func:`open`, :class:`pathlib.Path`, etc. without further conversion.

    Attributes:
        dataset_path: Absolute or project-relative path to the AID root dir.
        image_size:   Spatial resolution used for resizing (height == width).
        batch_size:   Mini-batch size for all DataLoaders.
        num_workers:  Number of worker processes for data loading.
        train_split:  Fraction of data used for training.
        val_split:    Fraction of data used for validation.
        test_split:   Fraction of data used for testing.
        seed:         Master random seed for reproducibility.
        extra:        Any additional key/value pairs present in the YAML that
                      are not explicitly modelled above.
    """

    dataset_path: str   = "data/AID"
    image_size:   int   = 224
    batch_size:   int   = 32
    num_workers:  int   = 4
    train_split:  float = 0.70
    val_split:    float = 0.15
    test_split:   float = 0.15
    seed:         int   = 42
    extra:        Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that split fractions sum to 1.0."""
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_split + val_split + test_split must equal 1.0, "
                f"got {total:.4f} (check your config file)."
            )

    def __repr__(self) -> str:
        lines = ["Config("]
        for f_name in (
            "dataset_path", "image_size", "batch_size",
            "num_workers", "train_split", "val_split",
            "test_split", "seed",
        ):
            lines.append(f"  {f_name}={getattr(self, f_name)!r},")
        if self.extra:
            lines.append(f"  extra={self.extra!r},")
        lines.append(")")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Known config keys (maps YAML key → Config field name)
# ------------------------------------------------------------------
_KNOWN_KEYS = {
    "dataset_path", "image_size", "batch_size",
    "num_workers", "train_split", "val_split", "test_split", "seed",
}


def load_config(yaml_path: str) -> Config:
    """
    Load a YAML configuration file and return a :class:`Config` instance.

    Unknown keys in the YAML file are collected in ``Config.extra`` so that
    downstream code can access them without needing to modify this module.

    Args:
        yaml_path: Path to a ``.yaml`` / ``.yml`` config file.

    Returns:
        A fully validated :class:`Config` object.

    Raises:
        FileNotFoundError: If *yaml_path* does not exist.
        yaml.YAMLError:    If the file cannot be parsed.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file '{yaml_path}' not found. "
            "Please provide a valid path with --config."
        )

    with path.open("r") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    known  = {k: v for k, v in raw.items() if k in _KNOWN_KEYS}
    extras = {k: v for k, v in raw.items() if k not in _KNOWN_KEYS}

    cfg = Config(**known, extra=extras)
    return cfg
