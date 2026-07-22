"""Optional dataset-image enhancement via diffusers' native Flux2KleinPipeline.

Nothing downloads at import time or pod boot -- gated behind an explicit
/api/enhance/setup call the user confirms in the UI, since this is a real
~18GB one-time download. Flux2KleinPipeline accepts an `image=` kwarg
natively for edit-style generation (same from_pretrained pattern our own
trainer already uses for Krea2), so no separate ComfyUI install or custom
node is needed -- an earlier version of this file drove ComfyUI directly;
this is a straight diffusers call instead.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import requests
import torch
from PIL import Image

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"

DEFAULT_ENHANCE_PROMPT = (
    "Enhance skin details and make the image look more realistic, amateur "
    "smartphone snapshot vibe. Keep the same person, pose, outfit, and framing."
)

# SamsungCam UltraReal -- a realism LoRA trained specifically for Flux.2 Klein 9B-base
# (https://civitai.com/models/1551668), pushes output away from the smooth/plasticky
# default look toward phone-camera texture. Pairs directly with the enhance prompt's
# own "amateur snapshot" goal, so it's loaded alongside the base pipeline at setup time.
LORA_URL = "https://civitai.com/api/download/models/2777498"
LORA_FILENAME = "Samsung_fluxklein9b.safetensors"
LORA_ADAPTER_NAME = "samsungcam_ultrareal"
DEFAULT_LORA_WEIGHT = 0.8
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
LORA_CACHE_DIR = WORKSPACE / "enhance_lora"

_state_lock = threading.Lock()
_state = {"status": "not_started", "detail": "", "error": ""}
_pipe = None
_pipe_lock = threading.Lock()


def get_status() -> dict:
    with _state_lock:
        return dict(_state)


def _set_status(status: str, detail: str = "", error: str = "") -> None:
    with _state_lock:
        _state["status"] = status
        _state["detail"] = detail
        _state["error"] = error


def is_ready() -> bool:
    return get_status()["status"] == "ready"


def _download_lora(civitai_key: str) -> Path:
    LORA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = LORA_CACHE_DIR / LORA_FILENAME
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    headers = {"Authorization": f"Bearer {civitai_key}"} if civitai_key else {}
    tmp = dest.with_suffix(".tmp")
    with requests.get(LORA_URL, headers=headers, stream=True, timeout=120) as resp:
        if resp.status_code in (401, 403):
            raise RuntimeError(
                f"Civitai returned {resp.status_code} downloading the realism LoRA -- "
                "add a Civitai API key in Settings (civitai.com -> Account -> API Keys)."
            )
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.rename(dest)
    return dest


def setup_enhance(hf_token: str, civitai_key: str = "") -> None:
    """Runs in a background thread. Downloads (first time only) and loads the
    pipeline plus its realism LoRA, then keeps it resident in _pipe for
    subsequent enhance calls."""
    global _pipe
    try:
        _set_status("downloading", f"Downloading {MODEL_ID} (first time only, ~18GB)...")
        from diffusers import Flux2KleinPipeline

        pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, token=hf_token or None,
        )
        _set_status("downloading", "Downloading realism LoRA (SamsungCam UltraReal, ~80MB)...")
        lora_path = _download_lora(civitai_key)
        pipe.load_lora_weights(str(lora_path), adapter_name=LORA_ADAPTER_NAME)
        pipe.set_adapters([LORA_ADAPTER_NAME], adapter_weights=[DEFAULT_LORA_WEIGHT])
        _set_status("starting", "Moving model to GPU...")
        pipe.to("cuda")
        with _pipe_lock:
            _pipe = pipe
        _set_status("ready", "Ready.")
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI, this is best-effort
        msg = str(exc)
        if "gated" in msg.lower() or "403" in msg or "401" in msg:
            msg += (
                f" -- {MODEL_ID} is gated: log into huggingface.co with the account matching your "
                f"HF token, open {MODEL_ID}, and click Agree once. Also check your token has "
                "'Read access to contents of all public gated repos you can access' enabled."
            )
        _set_status("failed", "", msg)


def unload() -> None:
    """Frees the ~20GB+ of VRAM the pipeline holds. Call before training --
    enhance and training can't run at once on a single GPU."""
    global _pipe
    with _pipe_lock:
        if _pipe is not None:
            del _pipe
            _pipe = None
            torch.cuda.empty_cache()
    _set_status("not_started", "")


def _resize_for_flux(img: Image.Image, max_megapixels: float = 1.0) -> Image.Image:
    w, h = img.size
    scale = min(1.0, (max_megapixels * 1_000_000 / (w * h)) ** 0.5)
    new_w = max(16, round(w * scale / 16) * 16)
    new_h = max(16, round(h * scale / 16) * 16)
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def run_enhance(
    image_path: Path, prompt: str, count: int, out_dir: Path, lora_weight: float = DEFAULT_LORA_WEIGHT
) -> list[Path]:
    if not is_ready():
        raise RuntimeError("Enhance is not set up yet.")
    out_dir.mkdir(parents=True, exist_ok=True)
    img = _resize_for_flux(Image.open(image_path).convert("RGB"))
    width, height = img.size

    results: list[Path] = []
    with _pipe_lock:
        _pipe.set_adapters([LORA_ADAPTER_NAME], adapter_weights=[lora_weight])
        for i in range(count):
            seed = int.from_bytes(os.urandom(4), "big")
            generator = torch.Generator(device="cuda").manual_seed(seed)
            out_image = _pipe(
                prompt=prompt,
                image=img,
                num_inference_steps=8,
                guidance_scale=1.0,
                height=height,
                width=width,
                generator=generator,
            ).images[0]
            dest = out_dir / f"candidate_{i}.png"
            out_image.save(dest)
            results.append(dest)
    return results
