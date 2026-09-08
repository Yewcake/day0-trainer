"""Cropping operations -- not part of mrln-arcane-tuner's block list, added
here specifically because Gemini's own checkpoint analysis flagged white
letterboxing/canvas padding in a real dataset as a likely overfitting
contributor (see minimax_h3_trainer_status / the Livia auto-analysis run).
Same numpy/PIL style as the ported modules, no new dependencies.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Never trim more than this fraction of a side, even if the border-detection
# heuristic thinks it should -- caps the damage from a false positive on a
# photo with a genuinely uniform-colored background near one edge.
_MAX_TRIM_FRACTION = 0.30


def trim_uniform_borders(img: Image.Image, tolerance: float = 10.0) -> Image.Image:
    """Auto-crop solid-color borders (letterboxing, pillarboxing, white/black
    frames added by an export tool) by scanning inward from each edge until a
    row/column with real variation is found.

    Args:
        img: Source image.
        tolerance: Max per-channel standard deviation for a row/column to
            still count as "uniform" (0-255 scale). Higher = more aggressive.
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    max_top = int(h * _MAX_TRIM_FRACTION)
    max_bottom = int(h * _MAX_TRIM_FRACTION)
    max_left = int(w * _MAX_TRIM_FRACTION)
    max_right = int(w * _MAX_TRIM_FRACTION)

    top = 0
    while top < max_top and arr[top, :, :].std() < tolerance:
        top += 1
    bottom = h
    while bottom > h - max_bottom and arr[bottom - 1, :, :].std() < tolerance:
        bottom -= 1
    left = 0
    while left < max_left and arr[:, left, :].std() < tolerance:
        left += 1
    right = w
    while right > w - max_right and arr[:, right - 1, :].std() < tolerance:
        right -= 1

    if top == 0 and bottom == h and left == 0 and right == w:
        return img  # nothing to trim
    if bottom - top < h * 0.5 or right - left < w * 0.5:
        return img  # heuristic ran away (e.g. a near-solid photo) -- refuse rather than gut the image

    return img.crop((left, top, right, bottom))


def crop_to_aspect(img: Image.Image, ratio_w: float, ratio_h: float, anchor: str = "center") -> Image.Image:
    """Center-crop (never upscale/pad) to a target aspect ratio, trimming the
    dimension that overshoots.

    Args:
        anchor: "center" (default), "top", or "bottom" -- which part of the
            trimmed dimension to keep when cropping height.
    """
    if ratio_w <= 0 or ratio_h <= 0:
        raise ValueError("ratio_w/ratio_h must be positive")

    w, h = img.size
    target_ratio = ratio_w / ratio_h
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 1e-6:
        return img

    if current_ratio > target_ratio:
        # Too wide -- trim left/right, always centered (no meaningful "anchor" on width).
        new_w = round(h * target_ratio)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))

    # Too tall -- trim top/bottom per anchor.
    new_h = round(w / target_ratio)
    if anchor == "top":
        top = 0
    elif anchor == "bottom":
        top = h - new_h
    else:
        top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))
