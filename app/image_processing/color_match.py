"""Color matching -- match one image's color distribution to a reference.

Ported from master-merlin/mrln-arcane-tuner (Apache-2.0), unchanged. Wavelet
mode needs PyWavelets; falls back to CDF matching if it isn't installed
rather than hard-requiring a new dependency for one optional mode.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def apply_color_match(
    img: Image.Image,
    reference: Image.Image,
    strength: float = 1.0,
    method: str = "cdf",
) -> Image.Image:
    """Match color distribution of img to reference image.

    Args:
        reference: The reference/target image whose colors to match.
        strength: Blend factor (0.0 = no change, 1.0 = full match).
        method: 'cdf' for histogram CDF matching, 'wavelet' for wavelet-based.
    """
    if strength <= 0.0:
        return img

    if method == "wavelet":
        return _color_match_wavelet(img, reference, strength)
    return _color_match_cdf(img, reference, strength)


def _color_match_cdf(
    img: Image.Image,
    reference: Image.Image,
    strength: float,
) -> Image.Image:
    """Per-channel CDF histogram specification."""
    src = np.array(img.convert("RGB"), dtype=np.uint8)
    ref = np.array(reference.convert("RGB"), dtype=np.uint8)
    result = src.copy()

    for ch in range(3):
        src_hist, _ = np.histogram(src[:, :, ch], bins=256, range=(0, 256))
        ref_hist, _ = np.histogram(ref[:, :, ch], bins=256, range=(0, 256))
        src_cdf = np.cumsum(src_hist).astype(np.float64)
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        src_cdf /= src_cdf[-1] + 1e-10
        ref_cdf /= ref_cdf[-1] + 1e-10

        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            j = np.searchsorted(ref_cdf, src_cdf[i])
            lut[i] = min(j, 255)

        matched = lut[src[:, :, ch]]
        result[:, :, ch] = np.clip(
            src[:, :, ch].astype(np.float32) * (1 - strength) + matched.astype(np.float32) * strength,
            0, 255,
        ).astype(np.uint8)

    return Image.fromarray(result, "RGB")


def _color_match_wavelet(
    img: Image.Image,
    reference: Image.Image,
    strength: float,
) -> Image.Image:
    """Wavelet-based color transfer preserving source detail."""
    try:
        import pywt
    except ImportError:
        return _color_match_cdf(img, reference, strength)

    src = np.array(img.convert("RGB"), dtype=np.float32)
    result = src.copy()

    ref_resized = np.array(reference.convert("RGB").resize(
        (src.shape[1], src.shape[0]), Image.Resampling.LANCZOS
    ), dtype=np.float32)

    for ch in range(3):
        src_coeffs = pywt.wavedec2(src[:, :, ch], 'db4', level=2)
        ref_coeffs = pywt.wavedec2(ref_resized[:, :, ch], 'db4', level=2)

        matched_coeffs = list(src_coeffs)
        src_approx = src_coeffs[0]
        ref_approx = ref_coeffs[0]
        src_mean, src_std = src_approx.mean(), src_approx.std() + 1e-6
        ref_mean, ref_std = ref_approx.mean(), ref_approx.std() + 1e-6
        transferred = (src_approx - src_mean) * (ref_std / src_std) + ref_mean
        matched_coeffs[0] = src_approx * (1 - strength) + transferred * strength

        result[:, :, ch] = pywt.waverec2(matched_coeffs, 'db4')[:src.shape[0], :src.shape[1]]

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGB")
