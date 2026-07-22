"""Optional dataset-image enhancement via a trimmed Flux.2 Klein ComfyUI graph.

Nothing here downloads or runs anything at import time or at pod boot -- it's
all gated behind an explicit /api/enhance/setup call the user confirms in the
UI, since ComfyUI + its models are a real (~15-20GB) one-time download.

Only one third-party custom node is required (TextEncodeEditAdvanced, for the
image+text conditioning Flux.2's edit/kontext mechanism needs) -- everything
else in the reference workflow (multi-lora UI, model-source switch, image
resize-to-megapixels, side-by-side preview) was ComfyUI-canvas convenience
that a headless single-image API call doesn't need, so it's replaced with
plain core nodes or precomputed Python (see build_enhance_graph / resize math
in run_enhance).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import requests
from PIL import Image

COMFY_DIR = Path(os.environ.get("WORKSPACE_DIR", "/workspace")) / "comfyui"
COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8189
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"

CUSTOM_NODE_REPO = "https://github.com/BigStationW/ComfyUi-TextEncodeEditAdvanced.git"
UNET_REPO, UNET_FILE = "black-forest-labs/FLUX.2-klein-9b-fp8", "flux-2-klein-9b-fp8.safetensors"
VAE_TEXTENC_REPO = "Comfy-Org/vae-text-encorder-for-flux-klein-9b"
VAE_FILE = "split_files/vae/flux2-vae.safetensors"
CLIP_FILE = "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"

DEFAULT_ENHANCE_PROMPT = (
    "Enhance skin details and make the image look more realistic, amateur "
    "smartphone snapshot vibe. Keep the same person, pose, outfit, and framing."
)

_state_lock = threading.Lock()
_state = {"status": "not_started", "detail": "", "error": ""}
_comfy_proc: subprocess.Popen | None = None


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


def _filter_out_torch(requirements_path: Path) -> Path:
    """ComfyUI's requirements.txt pins its own torch; we already have a tested
    torch 2.9.1+cu128 install and must not let this clobber it."""
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    kept = [l for l in lines if not l.strip().lower().startswith(("torch", "torchvision", "torchaudio"))]
    filtered = requirements_path.with_name("requirements.no-torch.txt")
    filtered.write_text("\n".join(kept), encoding="utf-8")
    return filtered


def _download_model(repo_id: str, filename: str, dest: Path, token: str) -> None:
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return
    _set_status("downloading", f"Downloading {filename}...")
    path = hf_hub_download(repo_id=repo_id, filename=filename, token=token or None)
    shutil.copy2(path, dest)


def setup_enhance(hf_token: str) -> None:
    """Runs in a background thread. Idempotent -- safe to call again after a
    partial failure, already-fetched pieces are skipped."""
    try:
        _set_status("cloning", "Cloning ComfyUI...")
        if not COMFY_DIR.is_dir():
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFY_DIR)],
                check=True, capture_output=True, text=True,
            )

        custom_node_dir = COMFY_DIR / "custom_nodes" / "ComfyUi-TextEncodeEditAdvanced"
        if not custom_node_dir.is_dir():
            _set_status("cloning", "Installing TextEncodeEditAdvanced custom node...")
            subprocess.run(
                ["git", "clone", "--depth", "1", CUSTOM_NODE_REPO, str(custom_node_dir)],
                check=True, capture_output=True, text=True,
            )
            req = custom_node_dir / "requirements.txt"
            if req.is_file():
                subprocess.run(["pip", "install", "-r", str(req)], check=True, capture_output=True, text=True)

        req = COMFY_DIR / "requirements.txt"
        if req.is_file():
            _set_status("installing", "Installing ComfyUI dependencies...")
            filtered = _filter_out_torch(req)
            subprocess.run(["pip", "install", "-r", str(filtered)], check=True, capture_output=True, text=True)

        _download_model(UNET_REPO, UNET_FILE, COMFY_DIR / "models" / "diffusion_models" / UNET_FILE, hf_token)
        _download_model(VAE_TEXTENC_REPO, VAE_FILE, COMFY_DIR / "models" / "vae" / "flux2-vae.safetensors", hf_token)
        _download_model(VAE_TEXTENC_REPO, CLIP_FILE, COMFY_DIR / "models" / "clip" / "qwen_3_8b_fp8mixed.safetensors", hf_token)

        _set_status("starting", "Starting ComfyUI...")
        _launch_comfy()
        _wait_for_comfy()
        _set_status("ready", "Ready.")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc))[-800:]
        _set_status("failed", "", detail)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI, this is a best-effort setup
        msg = str(exc)
        if "gated" in msg.lower() or "403" in msg or "401" in msg:
            msg += (
                " -- the FLUX.2 Klein model is gated: log into huggingface.co with the account matching your "
                "HF token, open black-forest-labs/FLUX.2-klein-9b-fp8, and click Agree to the license once."
            )
        _set_status("failed", "", msg)


def _launch_comfy() -> None:
    global _comfy_proc
    if _comfy_proc is not None and _comfy_proc.poll() is None:
        return
    log_path = COMFY_DIR / "day0_comfy.log"
    log_file = open(log_path, "w", encoding="utf-8")
    _comfy_proc = subprocess.Popen(
        ["python", "main.py", "--listen", COMFY_HOST, "--port", str(COMFY_PORT), "--disable-auto-launch"],
        cwd=str(COMFY_DIR), stdout=log_file, stderr=subprocess.STDOUT,
    )


def _wait_for_comfy(timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{COMFY_URL}/system_stats", timeout=3).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("ComfyUI did not respond in time after starting.")


def _resize_for_flux(image_path: Path, max_megapixels: float = 1.0) -> tuple[Path, int, int]:
    """Downscale to a megapixel budget, snapped to 16px (Flux2's latent factor).
    Replaces the reference workflow's custom scale-to-megapixels node."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, (max_megapixels * 1_000_000 / (w * h)) ** 0.5)
    new_w = max(16, round(w * scale / 16) * 16)
    new_h = max(16, round(h * scale / 16) * 16)
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    out_path = COMFY_DIR / "input" / f"day0_{uuid.uuid4().hex[:10]}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path, new_w, new_h


def build_enhance_graph(image_filename: str, prompt: str, seed: int, width: int, height: int, steps: int = 8) -> dict:
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET_FILE, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_8b_fp8mixed.safetensors", "type": "flux2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "load_image": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "encode": {
            "class_type": "TextEncodeEditAdvanced",
            "inputs": {
                "clip": ["clip", 0],
                "vae": ["vae", 0],
                "image1": ["load_image", 0],
                "prompt": prompt,
                "vl_megapixels": 0.5,
                "max_images_allowed": "1",
            },
        },
        "guider": {"class_type": "BasicGuider", "inputs": {"model": ["unet", 0], "conditioning": ["encode", 0]}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
        "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "sample": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                "sigmas": ["sigmas", 0], "latent_image": ["latent", 0],
            },
        },
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["decode", 0], "filename_prefix": f"day0_enhance/{seed}"},
        },
    }


def _submit_and_wait(graph: dict, timeout: int = 120) -> list[Path]:
    resp = requests.post(f"{COMFY_URL}/prompt", json={"prompt": graph}, timeout=10)
    resp.raise_for_status()
    prompt_id = resp.json()["prompt_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10).json()
        entry = hist.get(prompt_id)
        if entry and entry.get("status", {}).get("completed"):
            outputs = entry.get("outputs", {}).get("save", {}).get("images", [])
            return [COMFY_DIR / "output" / o["subfolder"] / o["filename"] for o in outputs]
        time.sleep(1)
    raise RuntimeError("ComfyUI did not finish generating in time.")


def run_enhance(image_path: Path, prompt: str, count: int, out_dir: Path) -> list[Path]:
    if not is_ready():
        raise RuntimeError("Enhance is not set up yet.")
    resized_path, width, height = _resize_for_flux(image_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []
    try:
        for i in range(count):
            seed = int.from_bytes(os.urandom(4), "big")
            graph = build_enhance_graph(resized_path.name, prompt, seed, width, height)
            produced = _submit_and_wait(graph)
            for j, src in enumerate(produced):
                if not src.is_file():
                    continue
                dest = out_dir / f"candidate_{i}_{j}.png"
                shutil.copy2(src, dest)
                results.append(dest)
    finally:
        resized_path.unlink(missing_ok=True)
    return results
