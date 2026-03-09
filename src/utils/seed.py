"""
Global reproducibility utilities.

Call :func:`set_seed` once at the start of any script or experiment to
ensure deterministic behaviour across Python, NumPy, and PyTorch.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU and GPU).

    Args:
        seed: Integer seed value.  Defaults to 42.

    Note:
        For *full* CUDA determinism you may also need to set::

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False

        These are intentionally left opt-in as they can reduce GPU throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
