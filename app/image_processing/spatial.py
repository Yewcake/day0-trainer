"""Spatial adjustments -- sharpening, vignette.

Ported from master-merlin/mrln-arcane-tuner (Apache-2.0). Lens correction
was dropped -- not part of this trainer's dataset-adjust scope.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def apply_sharpening(
    img: Image.Image,
    method: str = "unsharp_mask",
    params: dict | None = None,
) -> Image.Image:
    """Apply sharpening with selectable method."""
    params = params or {}
    rgb = img.convert("RGB")

    if method == "unsharp_mask":
        radius = params.get("radius", 2.0)
        percent = params.get("percent", 150)
        threshold = params.get("threshold", 3)
        return rgb.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

    if method == "kernel":
        strength = params.get("strength", 1.0)
        sharpened = rgb.filter(ImageFilter.SHARPEN)
        if strength == 1.0:
            return sharpened
        arr_o = np.array(rgb, dtype=np.float32)
        arr_s = np.array(sharpened, dtype=np.float32)
        blended = arr_o * (1 - strength) + arr_s * strength
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB")

    if method == "high_pass":
        radius = params.get("radius", 3.0)
        strength = params.get("strength", 0.5)
        arr = np.array(rgb, dtype=np.float32)
        blurred = np.array(rgb.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
        high_pass = arr - blurred + 128.0
        result = arr + (high_pass - 128.0) * strength
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGB")

    raise ValueError(f"Unknown sharpening method: {method}")


def apply_vignette(
    img: Image.Image,
    amount: float = 0.0,
    midpoint: float = 0.5,
    feather: float = 0.5,
) -> Image.Image:
    """Apply or remove vignette (radial brightness correction).

    Args:
        amount: -1.0 (remove/brighten corners) to +1.0 (darken corners). 0 = no change.
        midpoint: Radial distance (0-1) where effect begins. 0.5 = halfway from center.
        feather: Transition smoothness (0-1). 0 = hard, 1 = very gradual.
    """
    if amount == 0.0:
        return img

    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    y_coords = np.linspace(-1, 1, h)[:, np.newaxis]
    x_coords = np.linspace(-1, 1, w)[np.newaxis, :]
    aspect = w / max(h, 1)
    radius = np.sqrt((x_coords / max(aspect, 1.0)) ** 2 + (y_coords * min(aspect, 1.0)) ** 2)
    radius = radius / (radius.max() + 1e-6)

    feather_val = max(feather, 0.01)
    mask = np.clip((radius - midpoint) / feather_val, 0.0, 1.0)

    if amount > 0:
        multiplier = 1.0 - amount * mask
    else:
        multiplier = 1.0 + abs(amount) * mask

    arr *= multiplier[:, :, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
