"""
src/evaluation/corruptions.py
──────────────────────────────
Pure image corruption functions for robustness evaluation.

Each function accepts a PIL ``Image`` (RGB) and returns a corrupted PIL
``Image`` (RGB) of identical dimensions.  All operations are implemented
with NumPy so the only hard dependency is ``numpy``; OpenCV is used
opportunistically for motion blur (with a clean NumPy fallback when OpenCV
is not installed).

Corruption catalogue
--------------------
``apply_gaussian_noise``   – additive zero-mean Gaussian noise.
``apply_motion_blur``      – horizontal motion blur via uniform kernel.
``apply_brightness_shift`` – multiplicative brightness scaling in LAB space.

Factory helpers
---------------
``get_corruption_fn``      – look up a corruption function by string name.
``CORRUPTION_REGISTRY``    – dict of all available corruptions.

Design notes
------------
* All functions are deterministic given the same NumPy RNG state (or use a
  fixed seed so results are reproducible across evaluation runs).
* No torchvision transforms are used here — corruptions are applied at the
  PIL level before the standard normalisation pipeline.
* Clipping to [0, 255] uint8 is always performed before returning.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip a float array to [0, 255] and cast to uint8."""
    return np.clip(arr, 0, 255).astype(np.uint8)


def _pil_to_float(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a float64 array in [0, 255]."""
    return np.array(image, dtype=np.float64)


def _float_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a float64 array to a PIL RGB image."""
    return Image.fromarray(_to_uint8(arr), mode="RGB")


# ---------------------------------------------------------------------------
# 1. Gaussian Noise
# ---------------------------------------------------------------------------

def apply_gaussian_noise(
    image: Image.Image,
    sigma: float,
    seed:  int = 0,
) -> Image.Image:
    """
    Add zero-mean Gaussian noise to a PIL image.

    Noise is added in the [0, 1] normalised domain; the image is rescaled
    to [0, 1], corrupted, then rescaled back to [0, 255].

    Args:
        image: Input PIL image (RGB).
        sigma: Standard deviation of the noise in the [0, 1] domain.
               Typical values: 0.05, 0.1, 0.2.
        seed:  NumPy random seed for reproducibility.

    Returns:
        Corrupted PIL image (RGB, same size as input).

    Example::

        noisy = apply_gaussian_noise(pil_img, sigma=0.1)
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}.")

    rng  = np.random.default_rng(seed)
    arr  = _pil_to_float(image) / 255.0                    # [0, 1]
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape)
    corrupted = (arr + noise) * 255.0
    return _float_to_pil(corrupted)


# ---------------------------------------------------------------------------
# 2. Motion Blur
# ---------------------------------------------------------------------------

def _motion_blur_numpy(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply a horizontal motion-blur kernel using 1-D convolution (pure NumPy).

    The kernel is a normalised box filter of length *kernel_size* applied
    along the horizontal (width) axis for each colour channel independently.

    Args:
        arr:         Float64 array ``(H, W, C)`` in [0, 255].
        kernel_size: Width of the motion-blur kernel (odd recommended).

    Returns:
        Blurred float64 array of the same shape.
    """
    kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
    out    = np.empty_like(arr)
    pad    = kernel_size // 2
    for c in range(arr.shape[2]):
        # Reflect-pad along width axis
        padded = np.pad(arr[:, :, c], ((0, 0), (pad, pad)), mode="reflect")
        # 1-D convolution row-by-row
        for row in range(arr.shape[0]):
            out[row, :, c] = np.convolve(padded[row], kernel, mode="valid")
    return out


def apply_motion_blur(
    image:       Image.Image,
    kernel_size: int,
) -> Image.Image:
    """
    Apply horizontal motion blur to a PIL image.

    Uses OpenCV's ``filter2D`` when available (fast) and falls back to a
    pure-NumPy 1-D convolution otherwise.

    Args:
        image:       Input PIL image (RGB).
        kernel_size: Length of the motion-blur kernel in pixels.
                     Typical values: 5, 9.  Must be a positive integer.

    Returns:
        Motion-blurred PIL image (RGB, same size as input).

    Raises:
        ValueError: If *kernel_size* is not a positive integer.

    Example::

        blurred = apply_motion_blur(pil_img, kernel_size=9)
    """
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be ≥ 1, got {kernel_size}.")

    arr = _pil_to_float(image)

    try:
        import cv2  # type: ignore
        # Horizontal motion blur kernel: 1×K row vector
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        blurred = cv2.filter2D(arr.astype(np.float32), -1, kernel).astype(np.float64)
    except ImportError:
        blurred = _motion_blur_numpy(arr, kernel_size)

    return _float_to_pil(blurred)


# ---------------------------------------------------------------------------
# 3. Brightness Shift
# ---------------------------------------------------------------------------

def apply_brightness_shift(
    image:  Image.Image,
    factor: float,
) -> Image.Image:
    """
    Scale image brightness multiplicatively.

    The image is converted to CIE LAB colour space; the L* (lightness)
    channel is multiplied by *factor* and clamped to [0, 100]; the image is
    then converted back to RGB.

    Using the LAB representation means that chromaticity (hue and
    saturation) is preserved and only perceived brightness changes — a more
    realistic model of natural illumination variation than naively scaling
    RGB values.

    Args:
        image:  Input PIL image (RGB).
        factor: Multiplicative brightness scaling factor.
                Values < 1 darken; values > 1 brighten.
                Typical values: 0.5 (dark), 1.5 (bright).

    Returns:
        Brightness-shifted PIL image (RGB, same size as input).

    Raises:
        ValueError: If *factor* is negative.

    Example::

        bright = apply_brightness_shift(pil_img, factor=1.5)
        dark   = apply_brightness_shift(pil_img, factor=0.5)
    """
    if factor < 0:
        raise ValueError(f"factor must be non-negative, got {factor}.")

    # Convert to LAB via PIL
    lab = image.convert("LAB")
    lab_arr = np.array(lab, dtype=np.float64)

    # PIL's LAB encoding: L ∈ [0, 255] maps to CIE L* ∈ [0, 100]
    # Scale L channel; keep a* and b* unchanged
    lab_arr[:, :, 0] = np.clip(lab_arr[:, :, 0] * factor, 0, 255)

    shifted_lab = Image.fromarray(lab_arr.astype(np.uint8), mode="LAB")
    return shifted_lab.convert("RGB")


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

CORRUPTION_REGISTRY: Dict[str, Callable] = {
    "gaussian_noise":    apply_gaussian_noise,
    "motion_blur":       apply_motion_blur,
    "brightness_shift":  apply_brightness_shift,
}

CORRUPTION_DISPLAY: Dict[str, str] = {
    "gaussian_noise":   "Gaussian Noise (σ)",
    "motion_blur":      "Motion Blur (kernel size)",
    "brightness_shift": "Brightness Shift (factor)",
}


def get_corruption_fn(name: str) -> Callable:
    """
    Return a corruption function by its registry key.

    Args:
        name: One of ``"gaussian_noise"``, ``"motion_blur"``,
              ``"brightness_shift"``.

    Returns:
        Callable ``(PIL.Image, severity) → PIL.Image``.

    Raises:
        ValueError: If *name* is not in the registry.
    """
    name = name.lower().strip()
    if name not in CORRUPTION_REGISTRY:
        raise ValueError(
            f"Unknown corruption '{name}'. "
            f"Available: {sorted(CORRUPTION_REGISTRY.keys())}."
        )
    return CORRUPTION_REGISTRY[name]
