"""Adjustment pipeline orchestrator -- ordered list of typed blocks applied
in sequence. Structure adapted from master-merlin/mrln-arcane-tuner
(Apache-2.0); block set trimmed to this trainer's dataset-adjust scope
(no GPU restoration/upscale blocks -- those need separately-downloaded
model weights, out of scope for this pass) and extended with the two
crop blocks, which aren't in the source repo at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from PIL import Image

from app.image_processing.color import apply_contrast, apply_hue_saturation, apply_white_balance
from app.image_processing.color_match import apply_color_match
from app.image_processing.crop import crop_to_aspect, trim_uniform_borders
from app.image_processing.curves import CurvePoint, apply_curves, apply_lut_cube, parse_cube_file
from app.image_processing.hsl import apply_hsl_selective
from app.image_processing.spatial import apply_sharpening, apply_vignette

BlockType = Literal[
    "trim_borders",
    "crop_aspect",
    "white_balance",
    "curves",
    "cube_lut",
    "hsl_selective",
    "hue_saturation",
    "contrast",
    "vignette",
    "sharpening",
    "color_match",
]


@dataclass
class PipelineBlock:
    """A single operation block in the editing pipeline."""

    type: BlockType
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


def _handle_trim_borders(img: Image.Image, params: dict) -> Image.Image:
    if not params.get("enabled", True):
        return img
    return trim_uniform_borders(img, tolerance=params.get("tolerance", 10.0))


def _handle_crop_aspect(img: Image.Image, params: dict) -> Image.Image:
    ratio = params.get("ratio")
    if not ratio:
        return img
    ratio_w, ratio_h = ratio
    return crop_to_aspect(img, ratio_w, ratio_h, anchor=params.get("anchor", "center"))


def _handle_white_balance(img: Image.Image, params: dict) -> Image.Image:
    temperature = params.get("temperature", 6500.0)
    tint = params.get("tint", 0.0)
    if temperature != 6500.0 or tint != 0.0:
        return apply_white_balance(img, temperature, tint)
    return img


def _handle_curves(img: Image.Image, params: dict) -> Image.Image:
    def _to_points(pts: list[dict] | None) -> list[CurvePoint] | None:
        if not pts:
            return None
        return [CurvePoint(x=p["x"], y=p["y"]) for p in pts]

    return apply_curves(
        img,
        master=_to_points(params.get("master")),
        r=_to_points(params.get("r")),
        g=_to_points(params.get("g")),
        b=_to_points(params.get("b")),
    )


def _handle_cube_lut(img: Image.Image, params: dict) -> Image.Image:
    cube_lut_str: str | None = params.get("cube_lut")
    if cube_lut_str:
        lut_data = parse_cube_file(cube_lut_str)
        strength = params.get("cube_lut_strength", 1.0)
        return apply_lut_cube(img, lut_data, strength=strength)
    return img


def _handle_hsl_selective(img: Image.Image, params: dict) -> Image.Image:
    hsl = params.get("hsl_config", params)
    if not hsl:
        return img
    return apply_hsl_selective(img, hsl)


def _handle_hue_saturation(img: Image.Image, params: dict) -> Image.Image:
    hue_shift = params.get("hue_shift", 0.0)
    saturation = params.get("saturation", 1.0)
    if hue_shift != 0.0 or saturation != 1.0:
        return apply_hue_saturation(img, hue_shift, saturation)
    return img


def _handle_contrast(img: Image.Image, params: dict) -> Image.Image:
    contrast = params.get("contrast", 1.0)
    if contrast != 1.0:
        return apply_contrast(img, contrast)
    return img


def _handle_vignette(img: Image.Image, params: dict) -> Image.Image:
    amount = params.get("amount", 0.0)
    if amount != 0.0:
        return apply_vignette(img, amount, midpoint=params.get("midpoint", 0.5), feather=params.get("feather", 0.5))
    return img


def _handle_sharpening(img: Image.Image, params: dict) -> Image.Image:
    method = params.get("method", "none")
    if method != "none":
        return apply_sharpening(img, method=method, params=params.get("params"))
    return img


def _handle_color_match(img: Image.Image, params: dict) -> Image.Image:
    ref_path = params.get("reference_path")
    if ref_path and os.path.exists(ref_path):
        with Image.open(ref_path) as ref_img:
            return apply_color_match(
                img,
                ref_img.convert("RGB"),
                strength=params.get("strength", 1.0),
                method=params.get("method", "cdf"),
            )
    return img


BLOCK_HANDLERS: dict[str, "callable"] = {
    "trim_borders": _handle_trim_borders,
    "crop_aspect": _handle_crop_aspect,
    "white_balance": _handle_white_balance,
    "curves": _handle_curves,
    "cube_lut": _handle_cube_lut,
    "hsl_selective": _handle_hsl_selective,
    "hue_saturation": _handle_hue_saturation,
    "contrast": _handle_contrast,
    "vignette": _handle_vignette,
    "sharpening": _handle_sharpening,
    "color_match": _handle_color_match,
}

# Crop/trim first (so every downstream color op works on the final framing,
# not pixels about to be discarded), color-match next (it should react to the
# image's own real content, not a border), then the rest.
BLOCK_ORDER: list[BlockType] = [
    "trim_borders",
    "crop_aspect",
    "color_match",
    "white_balance",
    "curves",
    "cube_lut",
    "hsl_selective",
    "hue_saturation",
    "contrast",
    "vignette",
    "sharpening",
]


def execute_pipeline(img: Image.Image, blocks: list[PipelineBlock]) -> Image.Image:
    """Execute an ordered list of pipeline blocks on an image.

    Blocks run in BLOCK_ORDER regardless of the order they were given in,
    so a caller can't accidentally color-grade before cropping just by
    submitting the list in a different order. Disabled blocks are skipped.
    """
    by_type = {b.type: b for b in blocks if b.enabled}
    result = img.convert("RGB")
    for block_type in BLOCK_ORDER:
        block = by_type.get(block_type)
        if block is None:
            continue
        handler = BLOCK_HANDLERS.get(block_type)
        if handler:
            result = handler(result, block.params)
    return result


def blocks_from_dict(adjustments: dict) -> list[PipelineBlock]:
    """Build a PipelineBlock list from the flat dict the dataset-adjust panel
    sends (one key per block type, absent/falsy = disabled)."""
    blocks: list[PipelineBlock] = []
    for block_type in BLOCK_ORDER:
        params = adjustments.get(block_type)
        if params:
            blocks.append(PipelineBlock(type=block_type, params=params))
    return blocks
