#!/usr/bin/env python3
"""
Direct Krea 2 LoRA training with Diffusers + PEFT.

This is an experimental direct trainer for fresh Krea2Pipeline support in
Diffusers. It avoids AI-Toolkit and diffusion-pipe. It caches VAE latents and
text embeddings, unloads the VAE/text encoder, then trains transformer LoRA.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from automagic import Automagic


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_brand", default="Day0 Made by Yewcake")
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--trigger_word", default="")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--save_every_n_epochs", type=int, default=0)
    parser.add_argument("--sample_every_n_steps", type=int, default=0)
    parser.add_argument("--sample_prompts", default="")
    parser.add_argument("--sample_inference_model", default="")
    parser.add_argument("--sample_num_inference_steps", type=int, default=28)
    parser.add_argument("--sample_guidance_scale", type=float, default=4.0)
    parser.add_argument("--sample_lora_scale", type=float, default=1.35)
    parser.add_argument("--sample_compare_base", type=int, default=1)
    parser.add_argument("--sample_seed", type=int, default=1234)
    parser.add_argument("--allow_weight_mismatch", type=int, default=0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--network_type", default="lora", choices=["lora", "lokr"])
    parser.add_argument("--lokr_factor", type=int, default=-1)  # -1 = auto (PEFT's own default)
    parser.add_argument("--lokr_full_rank", type=int, default=0)
    parser.add_argument("--lokr_decompose_both", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler", default="cosine", choices=["cosine"])
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--lora_type", default="character", choices=["character", "style", "pose", "custom"])
    parser.add_argument("--target_modules", default="character")
    parser.add_argument("--optimizer", default="paged_adamw8bit", choices=["adamw_fused", "adamw8bit", "paged_adamw8bit", "adamw", "automagic"])
    parser.add_argument("--enable_buckets", type=int, default=1)
    parser.add_argument("--bucket_no_upscale", type=int, default=1)
    parser.add_argument("--bucket_step", type=int, default=16)
    parser.add_argument("--min_bucket_res", type=int, default=384)
    parser.add_argument("--max_bucket_area", type=int, default=0)
    parser.add_argument("--vae_tiling", type=int, default=1)
    parser.add_argument("--vae_slicing", type=int, default=1)
    parser.add_argument("--attention_backend", default="auto")
    parser.add_argument("--empty_cache_every_n_steps", type=int, default=0)
    parser.add_argument("--transformer_group_offload", type=int, default=0)
    parser.add_argument("--group_offload_blocks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--caption_extension", default=".txt")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--gradient_checkpointing", type=int, default=0)
    parser.add_argument("--enable_wandb", type=int, default=0)
    parser.add_argument("--wandb_project", default="krea2-lora")
    parser.add_argument("--fp8_base", type=int, default=0)
    parser.add_argument("--train_dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--lora_dtype", default="match", choices=["fp32", "match"])
    parser.add_argument("--save_dtype", default="bf16", choices=["bf16", "fp32"])
    return parser.parse_args()


def resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def cast_state_dict(state: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().cpu().to(dtype=dtype) for key, tensor in state.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_caption(path: Path, trigger_word: str) -> str:
    txt = path.with_suffix(".txt")
    js = path.with_suffix(".json")
    caption = ""
    if txt.exists():
        caption = txt.read_text(encoding="utf-8", errors="replace").strip()
        try:
            payload = json.loads(caption)
            caption = json_to_plain_caption(payload)
        except Exception:
            pass
    elif js.exists():
        try:
            caption = json_to_plain_caption(json.loads(js.read_text(encoding="utf-8")))
        except Exception:
            caption = ""

    caption = " ".join(caption.split())
    if trigger_word and trigger_word not in caption:
        caption = f"{trigger_word}, adult woman, {caption}".strip(", ")
    return caption or f"{trigger_word}, adult woman, realistic casual smartphone photo"


def json_to_plain_caption(payload: object) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    parts: list[str] = []
    high = payload.get("high_level_description")
    if isinstance(high, str):
        parts.append(high)
    comp = payload.get("compositional_deconstruction")
    if isinstance(comp, dict):
        bg = comp.get("background")
        if isinstance(bg, str):
            parts.append(bg)
        for elem in comp.get("elements") or []:
            if isinstance(elem, dict) and isinstance(elem.get("desc"), str):
                parts.append(elem["desc"])
                break
    return ", ".join(parts)


class CachedKreaDataset(Dataset):
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.latents = sorted((cache_dir / "latents").glob("*.safetensors"))
        self.embeds = sorted((cache_dir / "embeds").glob("*.safetensors"))
        self.manifest_path = cache_dir / "manifest.json"
        self.rows: list[dict] = []
        if self.manifest_path.exists():
            try:
                self.rows = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                self.rows = []
        if len(self.latents) != len(self.embeds):
            raise RuntimeError("Latent/text cache counts do not match.")
        if not self.latents:
            raise RuntimeError("No cached samples found.")
        if self.rows and len(self.rows) != len(self.latents):
            print("WARNING: manifest count does not match latent count; continuing with file order.")
            self.rows = []

        self.bucket_keys: list[tuple[int, int]] = []
        for idx, latent_path in enumerate(self.latents):
            if self.rows and "latent_shape" in self.rows[idx]:
                _b, _c, lh, lw = self.rows[idx]["latent_shape"]
                self.bucket_keys.append((int(lh), int(lw)))
            elif self.rows and "bucket_height" in self.rows[idx] and "bucket_width" in self.rows[idx]:
                self.bucket_keys.append((int(self.rows[idx]["bucket_height"]), int(self.rows[idx]["bucket_width"])))
            else:
                self.bucket_keys.append((-1, -1))

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        from safetensors.torch import load_file

        latent = load_file(self.latents[idx])["latents"]
        embed_pack = load_file(self.embeds[idx])
        return {
            "latents": latent,
            "prompt_embeds": embed_pack["prompt_embeds"],
            "prompt_attention_mask": embed_pack["prompt_attention_mask"],
        }


class BucketBatchSampler:
    """Batches cached samples by latent shape so variable-size buckets can train with batch > 1."""

    def __init__(self, bucket_keys: list[tuple[int, int]], batch_size: int, drop_last: bool = False, shuffle: bool = True):
        self.bucket_keys = bucket_keys
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.groups: dict[tuple[int, int], list[int]] = {}
        for idx, key in enumerate(bucket_keys):
            self.groups.setdefault(key, []).append(idx)

    def __iter__(self) -> Iterator[list[int]]:
        keys = list(self.groups)
        if self.shuffle:
            random.shuffle(keys)
        batches: list[list[int]] = []
        for key in keys:
            indices = list(self.groups[key])
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        total = 0
        for indices in self.groups.values():
            q, r = divmod(len(indices), self.batch_size)
            total += q + (0 if self.drop_last or r == 0 else 1)
        return total


def _round_to_multiple(value: int, multiple: int, mode: str = "floor") -> int:
    value = int(value)
    multiple = max(1, int(multiple))
    if mode == "ceil":
        return max(multiple, ((value + multiple - 1) // multiple) * multiple)
    return max(multiple, (value // multiple) * multiple)


def choose_bucket_size(image: Image.Image, args: argparse.Namespace) -> tuple[int, int]:
    """Return (width, height), preserving aspect ratio and never upscaling when requested.

    `resolution` is treated as the maximum side / max square area budget, not a mandatory square size.
    Width/height are snapped to bucket_step, which should be divisible by VAE scale * patch size.
    """
    src_w, src_h = image.size
    step = max(8, int(args.bucket_step))
    max_side = int(args.resolution)
    max_area = int(args.max_bucket_area) if int(args.max_bucket_area) > 0 else max_side * max_side

    scale_side = min(max_side / max(src_w, 1), max_side / max(src_h, 1))
    scale_area = math.sqrt(max_area / max(src_w * src_h, 1))
    scale = min(scale_side, scale_area)
    if int(args.bucket_no_upscale):
        scale = min(1.0, scale)

    raw_w = max(step, int(round(src_w * scale)))
    raw_h = max(step, int(round(src_h * scale)))

    # Floor to avoid accidental upscaling/area overflow. Tiny images are allowed below min_bucket_res
    # when no_upscale is active; no_upscale is more important than forcing a minimum.
    bucket_w = _round_to_multiple(raw_w, step, "floor")
    bucket_h = _round_to_multiple(raw_h, step, "floor")

    if not int(args.bucket_no_upscale):
        bucket_w = max(_round_to_multiple(int(args.min_bucket_res), step, "ceil"), bucket_w)
        bucket_h = max(_round_to_multiple(int(args.min_bucket_res), step, "ceil"), bucket_h)

    bucket_w = min(bucket_w, _round_to_multiple(max_side, step, "floor"))
    bucket_h = min(bucket_h, _round_to_multiple(max_side, step, "floor"))

    # Enforce area budget after snapping.
    while bucket_w * bucket_h > max_area and bucket_w > step and bucket_h > step:
        if bucket_w >= bucket_h:
            bucket_w -= step
        else:
            bucket_h -= step

    return int(bucket_w), int(bucket_h)


def resize_to_bucket_tensor(image: Image.Image, width: int, height: int) -> torch.Tensor:
    image = image.convert("RGB")
    # ImageOps.fit preserves the target canvas and only minimally crops after aspect-preserving resize.
    # Because choose_bucket_size never exceeds the scaled source dimensions when no_upscale=1, this does not
    # upsample small images in the normal path.
    image = ImageOps.fit(image, (int(width), int(height)), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transform(image)


def center_crop_resize(image: Image.Image, size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transform(image.convert("RGB"))


def normalize_vae_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    mean = getattr(vae.config, "latents_mean", None)
    std = getattr(vae.config, "latents_std", None)
    if mean is None or std is None:
        scaling = getattr(vae.config, "scaling_factor", 1.0)
        shift = getattr(vae.config, "shift_factor", 0.0)
        return (latents - shift) * scaling

    z_dim = latents.shape[1]
    mean_t = torch.tensor(mean, device=latents.device, dtype=latents.dtype).view(1, z_dim, 1, 1)
    # Krea/Qwen decode uses raw = normalized / (1/std) + mean, so encode is:
    inv_std_t = (1.0 / torch.tensor(std, device=latents.device, dtype=latents.dtype)).view(1, z_dim, 1, 1)
    return (latents - mean_t) * inv_std_t


def cache_dataset(
    args: argparse.Namespace,
    pipe,
    cache_dir: Path,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> None:
    from safetensors.torch import save_file

    latent_dir = cache_dir / "latents"
    embed_dir = cache_dir / "embeds"
    manifest = cache_dir / "manifest.json"
    cache_info = cache_dir / "cache_info.json"
    expected_info = {
        "cache_version": 4,
        "model": args.pretrained_model_name_or_path,
        "resolution": args.resolution,
        "caption_extension": args.caption_extension,
        "train_dtype": args.train_dtype,
        "enable_buckets": int(args.enable_buckets),
        "bucket_no_upscale": int(args.bucket_no_upscale),
        "bucket_step": int(args.bucket_step),
        "min_bucket_res": int(args.min_bucket_res),
        "max_bucket_area": int(args.max_bucket_area),
        "vae_tiling": int(args.vae_tiling),
        "vae_slicing": int(args.vae_slicing),
    }

    if cache_info.exists():
        try:
            existing_info = json.loads(cache_info.read_text(encoding="utf-8"))
        except Exception:
            existing_info = {}
    else:
        existing_info = {}

    cache_ready = (
        manifest.exists()
        and existing_info == expected_info
        and list(latent_dir.glob("*.safetensors"))
        and list(embed_dir.glob("*.safetensors"))
    )
    if cache_ready:
        print(f"Using existing cache: {cache_dir}")
        return

    if cache_dir.exists():
        print(f"Rebuilding cache because model/resolution metadata changed or cache is incomplete: {cache_dir}")
        shutil.rmtree(cache_dir)

    latent_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in Path(args.dataset_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise RuntimeError(f"No images found in {args.dataset_dir}")

    print(f"Caching {len(image_paths)} samples with buckets={int(args.enable_buckets)} max_res={args.resolution} no_upscale={int(args.bucket_no_upscale)}...")

    pipe.vae.to(device=device, dtype=compute_dtype)
    if int(args.vae_tiling) and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
        print("VAE tiling enabled for cache encode.")
    if int(args.vae_slicing) and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
        print("VAE slicing enabled for cache encode.")
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.to(device=device, dtype=compute_dtype)

    image_rows = []
    for idx, image_path in enumerate(tqdm(image_paths, desc="cache")):
        caption = read_caption(image_path, args.trigger_word)

        with torch.no_grad():
            raw_image = Image.open(image_path)
            if int(args.enable_buckets):
                bucket_w, bucket_h = choose_bucket_size(raw_image, args)
                pixels = resize_to_bucket_tensor(raw_image, bucket_w, bucket_h).unsqueeze(0)
            else:
                bucket_w = bucket_h = int(args.resolution)
                pixels = center_crop_resize(raw_image, args.resolution).unsqueeze(0)
            pixels = pixels.to(device=device, dtype=compute_dtype)
            try:
                encoded = pipe.vae.encode(pixels.unsqueeze(2))
                latents = encoded.latent_dist.sample().squeeze(2)
            except Exception:
                encoded = pipe.vae.encode(pixels)
                latents = encoded.latent_dist.sample()
            latents = normalize_vae_latents(pipe.vae, latents).to("cpu", dtype=compute_dtype)

            hidden, mask = pipe.get_text_hidden_states(caption, device=device)
            hidden = hidden.to("cpu", dtype=compute_dtype)
            mask = mask.to("cpu")

        stem = f"{idx:05d}"
        save_file({"latents": latents}, latent_dir / f"{stem}.safetensors")
        save_file({"prompt_embeds": hidden, "prompt_attention_mask": mask}, embed_dir / f"{stem}.safetensors")
        image_rows.append({
            "image": image_path.name,
            "caption": caption,
            "source_width": int(raw_image.size[0]),
            "source_height": int(raw_image.size[1]),
            "bucket_width": int(bucket_w),
            "bucket_height": int(bucket_h),
            "latent_shape": list(latents.shape),
        })

    manifest.write_text(json.dumps(image_rows, indent=2), encoding="utf-8")
    cache_info.write_text(json.dumps(expected_info, indent=2), encoding="utf-8")

    if hasattr(pipe.vae, "disable_tiling"):
        try:
            pipe.vae.disable_tiling()
        except Exception:
            pass
    if hasattr(pipe.vae, "disable_slicing"):
        try:
            pipe.vae.disable_slicing()
        except Exception:
            pass
    pipe.vae.to("cpu")
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()


def pack_latents(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Latent dimensions must be divisible by patch_size.")
    latents = latents.view(batch, channels, height // patch_size, patch_size, width // patch_size, patch_size)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch, (height // patch_size) * (width // patch_size), channels * patch_size * patch_size)


def make_position_ids(latent_h: int, latent_w: int, text_seq_len: int, device: torch.device) -> torch.Tensor:
    img_ids = torch.zeros(latent_h, latent_w, 3, device=device)
    img_ids[..., 1] = torch.arange(latent_h, device=device)[:, None]
    img_ids[..., 2] = torch.arange(latent_w, device=device)[None, :]
    img_ids = img_ids.reshape(latent_h * latent_w, 3)
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    return torch.cat([text_ids, img_ids], dim=0)


def collate_cached(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    latent_shapes = {tuple(item["latents"].shape) for item in batch}
    if len(latent_shapes) != 1:
        raise RuntimeError(
            "Mixed latent shapes in one batch. Use train_batch_size=1 or the bucket batch sampler. "
            f"Shapes: {sorted(latent_shapes)}"
        )
    return {
        "latents": torch.cat([item["latents"] for item in batch], dim=0),
        "prompt_embeds": torch.cat([item["prompt_embeds"] for item in batch], dim=0),
        "prompt_attention_mask": torch.cat([item["prompt_attention_mask"] for item in batch], dim=0),
    }


def maybe_enable_fp8_base(model, enabled: bool) -> None:
    if not enabled:
        return
    if not hasattr(model, "enable_layerwise_casting"):
        print("FP8 base casting requested, but this Diffusers model does not expose enable_layerwise_casting().")
        return
    print("Enabling experimental FP8 layerwise base casting...")
    model.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

def maybe_set_attention_backend(model, backend: str) -> None:
    backend = (backend or "").strip().lower()
    if not backend or backend == "default":
        return
    # Let PyTorch prefer memory-efficient SDPA kernels where possible.
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
    if backend == "auto":
        candidates = ["flash", "flash_attention_2", "sdpa"]
    else:
        candidates = [backend]
    if hasattr(model, "set_attention_backend"):
        for candidate in candidates:
            try:
                model.set_attention_backend(candidate)
                print(f"Attention backend set to: {candidate}")
                return
            except Exception as exc:
                print(f"Attention backend '{candidate}' unavailable: {type(exc).__name__}: {exc}")
    print("Using Diffusers/PyTorch default attention backend with SDPA kernels enabled where available.")


def maybe_enable_transformer_group_offload(model, enabled: bool, device: torch.device, blocks: int) -> None:
    if not enabled:
        return
    if not hasattr(model, "enable_group_offload"):
        print("Transformer group offload requested, but this model does not expose enable_group_offload().")
        return
    print("Enabling experimental Diffusers transformer group offload. If backward fails, disable this option.")
    try:
        model.enable_group_offload(
            onload_device=device,
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=max(1, int(blocks)),
            use_stream=True,
        )
    except TypeError:
        model.enable_group_offload(
            onload_device=device,
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=max(1, int(blocks)),
        )


def trainable_stats(model) -> dict[str, float]:
    total_sq = 0.0
    grad_sq = 0.0
    max_abs = 0.0
    count = 0
    grad_count = 0
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            data = param.detach().float()
            total_sq += float(torch.sum(data * data).cpu())
            max_abs = max(max_abs, float(torch.max(torch.abs(data)).cpu()))
            count += param.numel()
            if param.grad is not None:
                grad = param.grad.detach().float()
                grad_sq += float(torch.sum(grad * grad).cpu())
                grad_count += param.numel()
    return {
        "lora_param_count": float(count),
        "lora_param_l2": math.sqrt(total_sq) if total_sq else 0.0,
        "lora_param_max_abs": max_abs,
        "lora_grad_l2": math.sqrt(grad_sq) if grad_sq else 0.0,
        "lora_grad_count": float(grad_count),
    }


def find_meta_tensors(model) -> list[str]:
    meta_names: list[str] = []
    for name, param in model.named_parameters():
        if getattr(param, "is_meta", False):
            meta_names.append(name)
    for name, buffer in model.named_buffers():
        if getattr(buffer, "is_meta", False):
            meta_names.append(name)
    return meta_names


def resolve_target_modules(model, target_spec: str, lora_type: str = "custom") -> list[str]:
    spec = target_spec.strip()
    preset = spec.lower()
    lora_type = (lora_type or "custom").lower()
    if preset == "auto":
        preset = "character" if lora_type == "character" else "style"

    known_presets = {
        "character",
        "identity",
        "strong_identity",
        "auto_identity",
        "style",
        "pose",
        "blocks",
        "blocks_only",
        "attention",
        "attention_only",
        "character_attention",
    }
    if preset not in known_presets:
        return [x.strip() for x in spec.split(",") if x.strip()]

    attention_suffixes = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "to_gate",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
    )
    mlp_suffixes = (
        "ff.gate",
        "ff.up",
        "ff.down",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    )
    attention_only = preset in {"attention", "attention_only", "character_attention"}
    suffixes = attention_suffixes if attention_only else attention_suffixes + mlp_suffixes

    candidates: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.endswith(suffixes):
            candidates.append(name)

    block_targets = [name for name in candidates if name.startswith("transformer_blocks.")]
    txtfusion_targets = [
        name
        for name in candidates
        if name.startswith("text_fusion.") or name.startswith("txtfusion.") or name.startswith("text_fusion_transformer.")
    ]

    if preset in {"style", "pose", "blocks", "blocks_only", "attention", "attention_only"}:
        targets = block_targets or [name for name in candidates if not name.startswith(("text_fusion.", "txtfusion.", "text_fusion_transformer."))]
    else:
        # Character/identity: use the main denoising blocks plus Krea2 text-fusion modules when present.
        targets = list(block_targets)
        for name in txtfusion_targets:
            if name not in targets:
                targets.append(name)
        if not targets:
            targets = candidates

    if not targets:
        raise RuntimeError(
            "Could not resolve target modules from the Krea2 transformer. "
            "Pass --target_modules manually, for example: to_q,to_k,to_v,to_out.0"
        )

    print(f"Resolved target preset '{preset}' with lora_type='{lora_type}' to {len(targets)} Linear modules.")
    if txtfusion_targets:
        selected_txt = sum(1 for name in targets if name in txtfusion_targets)
        print(f"Text-fusion Linear modules available: {len(txtfusion_targets)}; selected: {selected_txt}.")
    print("Target preview: " + ", ".join(targets[:12]) + (" ..." if len(targets) > 12 else ""))
    return targets


LOKR_PARAM_NAMES = (
    "lokr_w1",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_w2",
    "lokr_w2_a",
    "lokr_w2_b",
    "lokr_t2",
)


def normalize_peft_lora_key(key: str) -> str:
    key = key.replace(".lora_A.default.", ".lora_A.")
    key = key.replace(".lora_B.default.", ".lora_B.")
    for param in LOKR_PARAM_NAMES:
        key = key.replace(f".{param}.default", f".{param}")
    for prefix in ("base_model.model.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    if key.startswith("transformer."):
        key = key[len("transformer."):]
    return key


def build_prefixed_lora_state(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        normalized = normalize_peft_lora_key(key)
        out[f"{prefix}.{normalized}"] = tensor.detach().cpu()
    return out


def native_krea2_base_key(normalized: str) -> str | None:
    replacements = (
        (".attn.to_q.", ".attn.wq."),
        (".attn.to_k.", ".attn.wk."),
        (".attn.to_v.", ".attn.wv."),
        (".attn.to_out.0.", ".attn.wo."),
        (".attn.to_gate.", ".attn.gate."),
        (".ff.gate.", ".mlp.gate."),
        (".ff.up.", ".mlp.up."),
        (".ff.down.", ".mlp.down."),
        (".ff.net.2.", ".mlp.down."),
    )
    if normalized.startswith("transformer_blocks."):
        base = "diffusion_model." + normalized.replace("transformer_blocks.", "blocks.")
    elif normalized.startswith("text_fusion."):
        base = "diffusion_model." + normalized.replace("text_fusion.", "txtfusion.")
    elif normalized.startswith("txtfusion."):
        base = "diffusion_model." + normalized
    elif normalized.startswith("text_fusion_transformer."):
        base = "diffusion_model." + normalized.replace("text_fusion_transformer.", "txtfusion.")
    else:
        return None

    for old, new in replacements:
        if old in base:
            return base.replace(old, new)
    return None


def build_native_krea2_lora_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Convert Diffusers PEFT module keys into the native Comfy/AI-Toolkit Krea2
    # key schema. Uploaded known-good Krea2 LoRAs use:
    # diffusion_model.blocks.N.attn.wq/wk/wv/wo and mlp.gate/up/down.
    normalized_state = {normalize_peft_lora_key(k): v.detach().cpu() for k, v in state.items()}
    out: dict[str, torch.Tensor] = {}

    for normalized, tensor in normalized_state.items():
        if ".ff.net.0.proj." in normalized and any(f".{p}" in normalized for p in LOKR_PARAM_NAMES):
            # Kronecker factors cannot be chunked into gate/up halves the way
            # lora_B rows can. LoKr training excludes fused GEGLU targets, so
            # hitting this means a manual target override; skip with a warning.
            print(f"Skipping fused GEGLU module for LoKr native export: {normalized}")
            continue
        if ".ff.net.0.proj." in normalized:
            # Diffusers GEGLU projection is usually gate+up fused on the output
            # dimension. Native Krea2 stores those as separate mlp.gate/mlp.up
            # modules that share the same input projection rank.
            if normalized.startswith("transformer_blocks."):
                native = "diffusion_model." + normalized.replace("transformer_blocks.", "blocks.")
            elif normalized.startswith("text_fusion."):
                native = "diffusion_model." + normalized.replace("text_fusion.", "txtfusion.")
            elif normalized.startswith("txtfusion."):
                native = "diffusion_model." + normalized
            elif normalized.startswith("text_fusion_transformer."):
                native = "diffusion_model." + normalized.replace("text_fusion_transformer.", "txtfusion.")
            else:
                continue
            if ".lora_A." in native:
                out[native.replace(".ff.net.0.proj.", ".mlp.gate.")] = tensor
                out[native.replace(".ff.net.0.proj.", ".mlp.up.")] = tensor.clone()
            elif ".lora_B." in native:
                if tensor.ndim == 2 and tensor.shape[0] % 2 == 0:
                    gate_b, up_b = tensor.chunk(2, dim=0)
                    out[native.replace(".ff.net.0.proj.", ".mlp.gate.")] = gate_b.contiguous()
                    out[native.replace(".ff.net.0.proj.", ".mlp.up.")] = up_b.contiguous()
                else:
                    print(f"Skipping native Krea2 split for unexpected ff.net.0.proj shape: {normalized} {tuple(tensor.shape)}")
            continue

        native_key = native_krea2_base_key(normalized)
        if native_key is not None:
            out[native_key] = tensor

    if not out:
        print("WARNING: native Krea2 LoRA conversion produced no keys. Check target_modules and key manifest.")
    return out


def write_lora_key_manifest(save_dir: Path, files: dict[str, dict[str, torch.Tensor]]) -> None:
    manifest = {}
    for filename, state in files.items():
        keys = sorted(state)
        manifest[filename] = {
            "key_count": len(keys),
            "first_40_keys": keys[:40],
        }
    (save_dir / "lora_key_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def activate_loaded_lora(pipe, weight: float) -> str:
    adapter_name = "default"
    try:
        adapters = pipe.get_list_adapters()
        if isinstance(adapters, dict):
            names = []
            for value in adapters.values():
                if isinstance(value, (list, tuple, set)):
                    names.extend(str(x) for x in value)
                elif value:
                    names.append(str(value))
            if names:
                adapter_name = names[-1]
        elif isinstance(adapters, (list, tuple, set)) and adapters:
            adapter_name = str(list(adapters)[-1])
    except Exception:
        pass

    try:
        pipe.set_adapters([adapter_name], adapter_weights=[weight])
    except Exception:
        # Some Diffusers builds auto-name the first adapter default_0.
        adapter_name = "default_0"
        pipe.set_adapters([adapter_name], adapter_weights=[weight])
    return adapter_name


def load_and_activate_sample_lora(pipe, lora_dir: Path, weight_name: str, weight: float) -> str:
    try:
        pipe.load_lora_weights(lora_dir, weight_name=weight_name, adapter_name="sample")
        pipe.set_adapters(["sample"], adapter_weights=[weight])
        return "sample"
    except TypeError:
        pipe.load_lora_weights(lora_dir, weight_name=weight_name)
        return activate_loaded_lora(pipe, weight)


LOKR_FULL_RANK_R = 9999999999  # mirrors AI-Toolkit: huge r disables w2 decomposition


def build_network_config(args: argparse.Namespace, target_modules: list[str]):
    """Build the PEFT adapter config for the selected network type.

    Used for both the training transformer and re-injection into the fresh
    sampling pipeline (Diffusers load_lora_weights cannot load LoKr).
    """
    if args.network_type == "lokr":
        from peft import LoKrConfig

        rank = LOKR_FULL_RANK_R if args.lokr_full_rank else args.rank
        # alpha == r keeps the PEFT runtime scale at 1.0, which matches what
        # ComfyUI computes when no alpha tensors are present in the file.
        return LoKrConfig(
            r=rank,
            alpha=rank,
            decompose_factor=args.lokr_factor,
            decompose_both=bool(args.lokr_decompose_both),
            target_modules=target_modules,
            init_weights=True,
        )
    from peft import LoraConfig

    return LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )


def filter_targets_for_network(args: argparse.Namespace, targets: list[str]) -> list[str]:
    if args.network_type != "lokr":
        return targets
    kept = [name for name in targets if not name.endswith("ff.net.0.proj")]
    dropped = len(targets) - len(kept)
    if dropped:
        print(
            f"LoKr: excluded {dropped} fused GEGLU module(s) (ff.net.0.proj); "
            "Kronecker factors cannot be split into native mlp.gate/mlp.up."
        )
    if not kept:
        raise RuntimeError("No valid target modules remain for LoKr training.")
    return kept


def apply_peft_scaling(model, multiplier: float) -> int:
    """Best-effort runtime scale for injected adapters (used for LoKr sampling)."""
    touched = 0
    if abs(multiplier - 1.0) < 1e-9:
        return touched
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            for adapter_name in scaling:
                scaling[adapter_name] = scaling[adapter_name] * multiplier
                touched += 1
    return touched


def train(args: argparse.Namespace) -> None:
    from diffusers import Krea2Pipeline
    from peft import LoraConfig, get_peft_model_state_dict
    from transformers import get_cosine_schedule_with_warmup

    try:
        import bitsandbytes as bnb
    except Exception:
        bnb = None

    device = torch.device("cuda")
    train_dtype = resolve_torch_dtype(args.train_dtype)
    save_dtype = resolve_torch_dtype(args.save_dtype)
    if args.fp8_base:
        raise RuntimeError(
            "fp8_base is disabled for this direct PEFT trainer. Diffusers layerwise FP8 casting "
            "also casts PEFT LoRA Linear weights, which crashes with addmm_cuda not implemented "
            "for Float8_e4m3fn. Use gradient_checkpointing=1, lower resolution, or a narrower "
            "target module set instead."
        )
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    print(f"{args.trainer_brand} | direct Krea2 LoRA trainer")
    print(
        f"Training compute dtype: {args.train_dtype}; "
        f"LoRA parameter dtype: {args.lora_dtype}; save dtype: {args.save_dtype}"
    )

    run_dir = Path(args.output_dir) / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    cache_dir = Path(args.cache_dir) if args.cache_dir else run_dir / f"cache_{args.resolution}_buckets{int(args.enable_buckets)}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Krea2Pipeline from {args.pretrained_model_name_or_path}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe = Krea2Pipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=train_dtype,
        )

    mismatch_warnings = [
        str(w.message)
        for w in caught
        if "were not used when initializing" in str(w.message)
        or "were not initialized from the model checkpoint" in str(w.message)
        or "newly initialized" in str(w.message)
    ]
    if mismatch_warnings and not args.allow_weight_mismatch:
        preview = "\n\n".join(msg[:1200] for msg in mismatch_warnings[:2])
        raise RuntimeError(
            "Krea2 model loaded with checkpoint/key mismatches. This usually means the selected "
            "repo is not compatible with the installed Diffusers Krea2 class, and training would "
            f"adapt a partially random model. Use krea/Krea-2-Raw or a known-good local Diffusers folder.\n\n{preview}"
        )
    meta_tensors = find_meta_tensors(pipe.transformer)
    if meta_tensors and not args.allow_weight_mismatch:
        preview = ", ".join(meta_tensors[:20])
        raise RuntimeError(
            "Krea2 transformer contains meta tensors after loading, meaning some weights were "
            "not actually materialized. This is usually an incompatible or incomplete Diffusers "
            "conversion. Do not train this run. Use official krea/Krea-2-Raw or a known-good local "
            f"Diffusers folder. First meta tensors: {preview}"
        )
    pipe.set_progress_bar_config(disable=True)

    cache_dataset(args, pipe, cache_dir, device, train_dtype)

    transformer = pipe.transformer
    del pipe.vae
    if getattr(pipe, "text_encoder", None) is not None:
        del pipe.text_encoder
    if getattr(pipe, "tokenizer", None) is not None:
        del pipe.tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    transformer.requires_grad_(False)
    transformer.to(device=device, dtype=train_dtype)
    maybe_set_attention_backend(transformer, args.attention_backend)
    maybe_enable_fp8_base(transformer, bool(args.fp8_base))
    if args.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled.")
    else:
        print("Gradient checkpointing disabled.")

    target_modules = filter_targets_for_network(
        args, resolve_target_modules(transformer, args.target_modules, args.lora_type)
    )
    network_config = build_network_config(args, target_modules)
    if args.network_type == "lokr":
        rank_desc = "full-rank" if args.lokr_full_rank else f"r={args.rank}"
        print(f"Network: LoKr ({rank_desc}, factor={args.lokr_factor}, decompose_both={bool(args.lokr_decompose_both)})")
    else:
        print(f"Network: LoRA (r={args.rank}, alpha={args.lora_alpha})")
    transformer.add_adapter(network_config)
    if args.lora_dtype == "fp32":
        for param in transformer.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        print("Converted trainable LoRA parameters to fp32.")
    maybe_enable_transformer_group_offload(
        transformer,
        bool(args.transformer_group_offload),
        device,
        args.group_offload_blocks,
    )
    transformer.train()

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    print(f"Trainable LoRA params: {sum(p.numel() for p in trainable):,}")
    initial_lora_l2 = trainable_stats(transformer)["lora_param_l2"]
    print(f"Initial LoRA param L2: {initial_lora_l2:.6f}")

    if args.optimizer == "paged_adamw8bit" and bnb is not None:
        optimizer = bnb.optim.PagedAdamW8bit(
            trainable,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
        print("Using bitsandbytes PagedAdamW8bit optimizer.")
    elif args.optimizer == "paged_adamw8bit" and bnb is None:
        print("bitsandbytes is unavailable; falling back to torch AdamW.")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw8bit" and bnb is not None:
        optimizer = bnb.optim.AdamW8bit(
            trainable,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw8bit" and bnb is None:
        print("bitsandbytes is unavailable; falling back to torch AdamW.")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw_fused":
        try:
            optimizer = torch.optim.AdamW(
                trainable,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_epsilon,
                weight_decay=args.weight_decay,
                fused=True,
            )
            print("Using torch AdamW fused optimizer.")
        except TypeError:
            print("Fused AdamW unavailable; falling back to torch AdamW.")
            optimizer = torch.optim.AdamW(
                trainable,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_epsilon,
                weight_decay=args.weight_decay,
            )
    elif args.optimizer == "automagic":
        optimizer = Automagic(
            trainable,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        print("Using Automagic optimizer (adaptive per-parameter learning rate).")
    else:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    dataset = CachedKreaDataset(cache_dir)
    if int(args.enable_buckets) and args.train_batch_size > 1:
        batch_sampler = BucketBatchSampler(dataset.bucket_keys, args.train_batch_size, drop_last=False, shuffle=True)
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=collate_cached,
        )
        print(f"Using bucket batch sampler for variable latent shapes: {len(batch_sampler)} batches/epoch.")
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_cached,
            drop_last=False,
        )

    use_wandb = bool(args.enable_wandb)
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    else:
        wandb = None

    patch_size = int(getattr(transformer.config, "patch_size", 2))
    global_step = 0
    epoch = 0
    last_saved_step = -1
    pbar = tqdm(total=args.max_train_steps, desc="steps")

    def save_samples(lora_dir: Path, step: int) -> None:
        prompts = [p.strip() for p in args.sample_prompts.split("||") if p.strip()]
        if not prompts:
            return
        sample_model = args.sample_inference_model or args.pretrained_model_name_or_path
        sample_dir = run_dir / "samples" / f"step-{step:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        print(f"Sampling {len(prompts)} prompts from {sample_model}...")
        torch.cuda.empty_cache()

        try:
            sample_pipe = Krea2Pipeline.from_pretrained(sample_model, torch_dtype=torch.bfloat16)
            sample_pipe.to(device)

            with torch.no_grad():
                for idx, prompt in enumerate(prompts):
                    seed = args.sample_seed + step + idx
                    if args.sample_compare_base:
                        base_generator = torch.Generator(device=device).manual_seed(seed)
                        base_image = sample_pipe(
                            prompt=prompt,
                            height=args.resolution,
                            width=args.resolution,
                            num_inference_steps=args.sample_num_inference_steps,
                            guidance_scale=args.sample_guidance_scale,
                            generator=base_generator,
                        ).images[0]
                        base_image.save(sample_dir / f"base_{idx + 1:02d}.png")
                        if use_wandb:
                            wandb.log({f"samples/base_{idx + 1:02d}": wandb.Image(base_image, caption=prompt)}, step=step)

                    if idx == 0:
                        if args.network_type == "lokr":
                            from peft import inject_adapter_in_model
                            from peft.utils import set_peft_model_state_dict
                            from safetensors.torch import load_file

                            lokr_state = load_file(lora_dir / "pytorch_lokr_weights.safetensors")
                            sample_config = build_network_config(args, target_modules)
                            inject_adapter_in_model(sample_config, sample_pipe.transformer)
                            set_peft_model_state_dict(sample_pipe.transformer, lokr_state)
                            sample_pipe.transformer.to(device=device, dtype=torch.bfloat16)
                            touched = apply_peft_scaling(sample_pipe.transformer, args.sample_lora_scale)
                            print(
                                f"Injected sample LoKr adapter at weight {args.sample_lora_scale}"
                                + (f" ({touched} modules rescaled)" if touched else "")
                            )
                        else:
                            adapter_name = load_and_activate_sample_lora(
                                sample_pipe,
                                lora_dir,
                                "pytorch_lora_weights.safetensors",
                                args.sample_lora_scale,
                            )
                            print(f"Activated sample LoRA adapter: {adapter_name} at weight {args.sample_lora_scale}")

                    lora_generator = torch.Generator(device=device).manual_seed(seed)
                    image = sample_pipe(
                        prompt=prompt,
                        height=args.resolution,
                        width=args.resolution,
                        num_inference_steps=args.sample_num_inference_steps,
                        guidance_scale=args.sample_guidance_scale,
                        generator=lora_generator,
                    ).images[0]
                    image.save(sample_dir / f"lora_{idx + 1:02d}.png")
                    if use_wandb:
                        wandb.log({f"samples/lora_{idx + 1:02d}": wandb.Image(image, caption=prompt)}, step=step)
            print(f"Saved samples: {sample_dir}")
        except Exception as exc:
            print(f"Sampling failed at step {step}: {type(exc).__name__}: {exc}")
        finally:
            try:
                del sample_pipe
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            transformer.train()

    def save_checkpoint(step: int, reason: str = "step") -> None:
        nonlocal last_saved_step
        if step == last_saved_step:
            return
        save_dir = checkpoint_dir / f"step-{step:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        state = cast_state_dict(get_peft_model_state_dict(transformer), save_dtype)
        saved_states: dict[str, dict[str, torch.Tensor]] = {}
        if args.network_type == "lokr":
            # Diffusers save/load_lora_weights only understands lora_A/lora_B,
            # so keep the raw PEFT LoKr state for sample-time re-injection.
            raw_state = {k: v.detach().cpu() for k, v in state.items()}
            save_file(raw_state, save_dir / "pytorch_lokr_weights.safetensors")
            saved_states["pytorch_lokr_weights.safetensors"] = raw_state
        else:
            try:
                Krea2Pipeline.save_lora_weights(save_dir, transformer_lora_layers=state)
                from safetensors.torch import load_file

                diffusers_file = save_dir / "pytorch_lora_weights.safetensors"
                if diffusers_file.exists():
                    saved_states[diffusers_file.name] = load_file(diffusers_file)
            except Exception as exc:
                print(f"save_lora_weights fallback because Diffusers raised: {exc}")
                diffusers_state = {k: v.detach().cpu() for k, v in state.items()}
                save_file(diffusers_state, save_dir / "pytorch_lora_weights.safetensors")
                saved_states["pytorch_lora_weights.safetensors"] = diffusers_state

        # Main ComfyUI/Krea2 export. This mirrors AI-Toolkit's native Krea2
        # schema: diffusion_model.blocks.* with attention/MLP names.
        comfy_native_state = build_native_krea2_lora_state(state)
        save_file(comfy_native_state, save_dir / "krea2_comfy_native_lora.safetensors")
        saved_states["krea2_comfy_native_lora.safetensors"] = comfy_native_state
        write_lora_key_manifest(save_dir, saved_states)

        metadata = vars(args).copy()
        metadata["trainer_brand"] = args.trainer_brand
        metadata["trainer_note"] = "Day0 direct Krea2 LoRA trainer. Made by Yewcake."
        if args.network_type == "lokr":
            metadata["comfy_lora_note"] = (
                "LoKr network. For ComfyUI, use krea2_comfy_native_lora.safetensors: it maps "
                "PEFT LoKr keys (lokr_w1/lokr_w2[_a/_b]) to the native Krea2 schema "
                "diffusion_model.blocks.* and loads through the standard LoRA loader. "
                "pytorch_lokr_weights.safetensors is kept only for sample re-injection during "
                "training. Recommended ComfyUI strength: 1.0."
            )
        else:
            metadata["comfy_lora_note"] = (
                "For ComfyUI Krea2 Turbo, use krea2_comfy_native_lora.safetensors. "
                "It maps Diffusers LoRA keys to the AI-Toolkit/Comfy native Krea2 schema "
                "diffusion_model.blocks.*. pytorch_lora_weights.safetensors is kept only "
                "for Diffusers sampling during training."
            )
        (save_dir / "training_args.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        last_saved_step = step
        print(f"Saved checkpoint ({reason}): {save_dir}")
        if args.sample_every_n_steps > 0 and step % args.sample_every_n_steps == 0:
            save_samples(save_dir, step)

    while global_step < args.max_train_steps:
        epoch += 1
        for batch in loader:
            latents = batch["latents"].to(device=device, dtype=train_dtype)
            prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=train_dtype)
            prompt_attention_mask = batch["prompt_attention_mask"].to(device=device)

            bsz, _channels, latent_h_raw, latent_w_raw = latents.shape
            noise = torch.randn_like(latents)
            t = torch.rand((bsz,), device=device, dtype=train_dtype)
            t = t.clamp(0.02, 0.98)
            sigmas = t.view(bsz, 1, 1, 1).to(dtype=latents.dtype)

            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            target = noise - latents

            noisy_packed = pack_latents(noisy_latents, patch_size=patch_size)
            target_packed = pack_latents(target, patch_size=patch_size)
            latent_h = latent_h_raw // patch_size
            latent_w = latent_w_raw // patch_size
            position_ids = make_position_ids(
                latent_h,
                latent_w,
                prompt_embeds.shape[1],
                device,
            )

            pred = transformer(
                hidden_states=noisy_packed,
                encoder_hidden_states=prompt_embeds,
                timestep=t,
                position_ids=position_ids,
                encoder_attention_mask=prompt_attention_mask,
                return_dict=False,
            )[0]

            loss = F.mse_loss(pred.float(), target_packed.float(), reduction="mean")
            loss.backward()
            stats_before_step = trainable_stats(transformer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if args.empty_cache_every_n_steps > 0 and global_step > 0 and global_step % args.empty_cache_every_n_steps == 0:
                gc.collect()
                torch.cuda.empty_cache()

            global_step += 1
            # Automagic manages its own per-parameter learning rates and ignores
            # the scheduler's param_group['lr'] entirely, so report its real
            # average rate instead of the (unused) cosine-decayed value.
            lr = optimizer.get_avg_learning_rate() if isinstance(optimizer, Automagic) else scheduler.get_last_lr()[0]
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}", epoch=epoch)

            # Local metrics stream (independent of wandb); the Day0 UI polls this.
            with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as metrics_file:
                metrics_file.write(
                    json.dumps(
                        {"step": global_step, "loss": round(loss.item(), 6), "lr": lr, "epoch": epoch}
                    )
                    + "\n"
                )

            if use_wandb:
                stats_after_step = trainable_stats(transformer)
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "epoch": epoch,
                        "lora/param_l2": stats_after_step["lora_param_l2"],
                        "lora/param_l2_delta_from_init": stats_after_step["lora_param_l2"] - initial_lora_l2,
                        "lora/param_max_abs": stats_after_step["lora_param_max_abs"],
                        "lora/grad_l2": stats_before_step["lora_grad_l2"],
                    },
                    step=global_step,
                )

            if global_step % args.save_every_n_steps == 0:
                save_checkpoint(global_step, reason="step")

            if global_step >= args.max_train_steps:
                break

        if (
            args.save_every_n_epochs > 0
            and epoch % args.save_every_n_epochs == 0
            and global_step < args.max_train_steps
        ):
            save_checkpoint(global_step, reason=f"epoch-{epoch}")

    save_checkpoint(global_step, reason="final")
    pbar.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
