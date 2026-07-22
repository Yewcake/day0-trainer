"""Day0 Trainer UI backend v2. Made by Yewcake.

Adds over v1: settings (HF/Gemini keys), model registry, multi-format
dataset intake (zip/rar/7z/loose files/folders), Gemini dataset captioner
with batch progress, per-image caption editing, and Gemini checkpoint
analysis (loss curve + samples -> best candidate recommendation).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

import requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
JOBS_DIR = WORKSPACE / "jobs"
DATASETS_DIR = WORKSPACE / "datasets"
SETTINGS_FILE = WORKSPACE / ".day0" / "settings.json"
UI_PASSWORD = os.environ.get("UI_PASSWORD", "")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

for directory in (JOBS_DIR, DATASETS_DIR, SETTINGS_FILE.parent):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Day0 Trainer", docs_url=None, redoc_url=None)

_active: dict[str, subprocess.Popen] = {}
_caption_runs: dict[str, dict] = {}  # dataset -> progress state


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def require_auth(t: str = Query(default="")) -> None:
    if UI_PASSWORD and t != UI_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid or missing token.")


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Invalid name.")
    return cleaned


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_settings(data: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(data, indent=2))


def load_models() -> list[dict]:
    return json.loads((APP_DIR / "models.json").read_text())["models"]


def job_dir(job_id: str) -> Path:
    path = (JOBS_DIR / safe_name(job_id)).resolve()
    if not str(path).startswith(str(JOBS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid job id.")
    return path


def dataset_dir(name: str) -> Path:
    path = (DATASETS_DIR / safe_name(name)).resolve()
    if not str(path).startswith(str(DATASETS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid dataset name.")
    return path


def dataset_images(path: Path) -> list[Path]:
    return sorted(
        p for p in path.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and ".thumbs" not in p.relative_to(path).parts
    )


def job_status(job_id: str) -> str:
    proc = _active.get(job_id)
    if proc is not None:
        code = proc.poll()
        if code is None:
            return "running"
        _active.pop(job_id, None)
        return "finished" if code == 0 else "failed"
    status_file = job_dir(job_id) / "status.json"
    if status_file.exists():
        try:
            return json.loads(status_file.read_text())["status"]
        except Exception:
            pass
    return "unknown"


def set_status(job_id: str, status: str) -> None:
    (job_dir(job_id) / "status.json").write_text(json.dumps({"status": status, "ts": time.time()}))


def reap_finished() -> None:
    for job_id in list(_active):
        code = _active[job_id].poll()
        if code is not None:
            _active.pop(job_id, None)
            status = "finished" if code == 0 else "failed"
            set_status(job_id, status)
            if status == "finished":
                maybe_auto_analyze(job_id)


def maybe_auto_analyze(job_id: str) -> None:
    try:
        config = json.loads((job_dir(job_id) / "config.json").read_text())
    except Exception:
        return
    if not config.get("auto_analyze"):
        return
    if (job_dir(job_id) / "analysis.md").exists():
        return  # already analyzed; avoid duplicate calls if reap_finished runs again
    if not load_settings().get("gemini_api_key"):
        return  # no key configured, nothing to do

    def worker() -> None:
        try:
            run_analysis(job_id)
        except Exception as exc:
            print(f"[auto-analyze] job {job_id} failed: {exc}")

    threading.Thread(target=worker, daemon=True).start()


def gemini_key() -> str:
    key = load_settings().get("gemini_api_key", "")
    if not key:
        raise HTTPException(status_code=400, detail="No Gemini API key configured. Add it in Settings.")
    return key


def encode_image_for_gemini(path: Path, max_side: int = 1024) -> dict:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buffer.getvalue()).decode()}}


def gemini_generate(model: str, parts: list[dict], key: str, timeout: int = 120) -> str:
    response = requests.post(
        f"{GEMINI_BASE}/models/{model}:generateContent",
        params={"key": key},
        json={"contents": [{"parts": parts}]},
        timeout=timeout,
    )
    if response.status_code != 200:
        detail = response.json().get("error", {}).get("message", response.text[:300])
        raise RuntimeError(f"Gemini error {response.status_code}: {detail}")
    data = response.json()
    try:
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"]).strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {json.dumps(data)[:300]}") from exc


# --------------------------------------------------------------------------
# Frontend + settings
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (APP_DIR / "app" / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "auth_required": bool(UI_PASSWORD)}


@app.get("/api/settings", dependencies=[Depends(require_auth)])
def get_settings() -> dict:
    settings = load_settings()
    return {
        "hf_token_set": bool(settings.get("hf_token") or os.environ.get("HF_TOKEN")),
        "gemini_key_set": bool(settings.get("gemini_api_key")),
        "gemini_model": settings.get("gemini_model", ""),
    }


@app.put("/api/settings", dependencies=[Depends(require_auth)])
def put_settings(payload: dict) -> dict:
    settings = load_settings()
    for key in ("hf_token", "gemini_api_key", "gemini_model"):
        if key in payload and payload[key] is not None:
            value = str(payload[key]).strip()
            if value:
                settings[key] = value
            elif payload[key] == "":
                settings.pop(key, None)
    save_settings(settings)
    return get_settings()


@app.get("/api/models", dependencies=[Depends(require_auth)])
def list_models() -> list[dict]:
    return load_models()


@app.get("/api/gemini/models", dependencies=[Depends(require_auth)])
def gemini_models() -> list[dict]:
    key = gemini_key()
    response = requests.get(f"{GEMINI_BASE}/models", params={"key": key, "pageSize": 100}, timeout=30)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Gemini model list failed: {response.text[:200]}")
    models = []
    for model in response.json().get("models", []):
        if "generateContent" in model.get("supportedGenerationMethods", []):
            name = model["name"].removeprefix("models/")
            models.append({"id": name, "label": model.get("displayName", name)})
    return models


# --------------------------------------------------------------------------
# Datasets: multi-format intake, browsing, captions
# --------------------------------------------------------------------------
JUNK_NAMES = {".DS_Store", "Thumbs.db"}


def extract_archive(archive: Path, target: Path) -> None:
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        target_root = target.resolve()
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                if info.filename.startswith("__MACOSX/") or Path(info.filename).name in JUNK_NAMES:
                    continue
                dest = (target / info.filename).resolve()
                if dest != target_root and target_root not in dest.parents:
                    continue  # zip-slip guard: entry would land outside the dataset dir
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(info) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
        return
    seven_zip = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
    if seven_zip is None:
        raise HTTPException(status_code=400, detail=f"No extractor available for {suffix}. Use a .zip instead.")
    result = subprocess.run(
        [seven_zip, "x", "-y", f"-o{target}", str(archive)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail=f"Extraction failed: {result.stderr[-300:] or result.stdout[-300:]}")
    for junk in target.rglob("*"):
        if junk.is_file() and (junk.name in JUNK_NAMES or "__MACOSX" in junk.parts):
            junk.unlink(missing_ok=True)


def flatten_dataset(target: Path) -> None:
    """Move all images (and their caption txts) to the dataset root."""
    for image in dataset_images(target):
        if image.parent != target:
            destination = target / image.name
            if not destination.exists():
                shutil.move(str(image), destination)
            caption = image.with_suffix(".txt")
            if caption.exists() and not (target / caption.name).exists():
                shutil.move(str(caption), target / caption.name)
    for entry in list(target.iterdir()):
        if entry.is_dir():
            shutil.rmtree(entry)
        elif entry.suffix.lower() not in IMAGE_EXTS | {".txt"}:
            entry.unlink()


@app.get("/api/datasets", dependencies=[Depends(require_auth)])
def list_datasets() -> list[dict]:
    out = []
    for entry in sorted(DATASETS_DIR.iterdir()):
        if entry.is_dir():
            images = dataset_images(entry)
            captioned = sum(1 for img in images if img.with_suffix(".txt").exists())
            out.append({"name": entry.name, "images": len(images), "captioned": captioned})
    return out


@app.post("/api/datasets/{name}/files", dependencies=[Depends(require_auth)])
async def add_files(name: str, files: list[UploadFile] = File(...)) -> dict:
    """Universal intake: archives (.zip/.rar/.7z), loose images, caption txts.

    The frontend flattens dropped folders into individual files before upload,
    so a folder drop arrives here as loose images + txts.
    """
    target = dataset_dir(name)
    target.mkdir(parents=True, exist_ok=True)
    added_archives = 0
    for upload in files:
        filename = safe_name(Path(upload.filename or "file").name)
        suffix = Path(filename).suffix.lower()
        destination = target / filename
        with destination.open("wb") as handle:
            while chunk := await upload.read(1 << 20):
                handle.write(chunk)
        if suffix in {".zip", ".rar", ".7z"}:
            extract_archive(destination, target)
            destination.unlink()
            added_archives += 1
        elif suffix not in IMAGE_EXTS | {".txt"}:
            destination.unlink()
    flatten_dataset(target)
    images = dataset_images(target)
    return {"name": target.name, "images": len(images), "archives_extracted": added_archives}


@app.delete("/api/datasets/{name}", dependencies=[Depends(require_auth)])
def delete_dataset(name: str) -> dict:
    target = dataset_dir(name)
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="Dataset not found.")
    shutil.rmtree(target)
    return {"deleted": name}


@app.get("/api/datasets/{name}/items", dependencies=[Depends(require_auth)])
def dataset_items(name: str) -> list[dict]:
    target = dataset_dir(name)
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="Dataset not found.")
    out = []
    for image in dataset_images(target):
        caption_file = image.with_suffix(".txt")
        out.append({
            "image": image.name,
            "caption": caption_file.read_text(encoding="utf-8", errors="replace") if caption_file.exists() else "",
        })
    return out


@app.get("/api/datasets/{name}/thumb/{image}", dependencies=[Depends(require_auth)])
def dataset_thumb(name: str, image: str, size: int = 220) -> FileResponse:
    source = dataset_dir(name) / safe_name(image)
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    thumbs = dataset_dir(name) / ".thumbs"
    thumbs.mkdir(exist_ok=True)
    thumb = thumbs / f"{size}_{source.name}.jpg"
    if not thumb.exists() or thumb.stat().st_mtime < source.stat().st_mtime:
        img = Image.open(source).convert("RGB")
        img.thumbnail((size, size))
        img.save(thumb, format="JPEG", quality=85)
    return FileResponse(thumb)


@app.put("/api/datasets/{name}/caption/{image}", dependencies=[Depends(require_auth)])
def set_caption(name: str, image: str, payload: dict) -> dict:
    source = dataset_dir(name) / safe_name(image)
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    source.with_suffix(".txt").write_text(str(payload.get("caption", "")).strip(), encoding="utf-8")
    return {"ok": True}


# --------------------------------------------------------------------------
# Captioner (Gemini, background batch)
# --------------------------------------------------------------------------
DEFAULT_CAPTION_INSTRUCTION = (
    "You are a world-class AI model specialist for image and video generation. "
    "Caption the image for dataset creation. Mention only: expression, outfit, pose, "
    "hairstyle, camera angle, whether there is blur in the photo, jewelry, accessories, "
    "setting, background. Do not mention face shape or body type. Caption in concise "
    "but detailed natural language. Output only the caption, no preamble."
)


def _caption_worker(name: str, instruction: str, model: str, trigger: str, only_missing: bool, key: str) -> None:
    state = _caption_runs[name]
    target = dataset_dir(name)
    images = dataset_images(target)
    if only_missing:
        images = [img for img in images if not img.with_suffix(".txt").exists()]
    state.update({"total": len(images), "done": 0, "errors": [], "status": "running"})
    for image in images:
        if state.get("cancel"):
            state["status"] = "cancelled"
            return
        try:
            parts = [{"text": instruction}, encode_image_for_gemini(image)]
            caption = gemini_generate(model, parts, key).replace("\n", " ").strip()
            if trigger and trigger not in caption:
                caption = f"{trigger}, {caption}"
            image.with_suffix(".txt").write_text(caption, encoding="utf-8")
        except Exception as exc:
            state["errors"].append(f"{image.name}: {exc}")
        state["done"] += 1
    state["status"] = "finished"


@app.post("/api/datasets/{name}/caption-all", dependencies=[Depends(require_auth)])
def caption_all(name: str, payload: dict) -> dict:
    if _caption_runs.get(name, {}).get("status") == "running":
        raise HTTPException(status_code=409, detail="Captioning already running for this dataset.")
    key = gemini_key()
    model = str(payload.get("model") or load_settings().get("gemini_model") or "gemini-2.5-flash")
    instruction = str(payload.get("instruction") or DEFAULT_CAPTION_INSTRUCTION)
    trigger = str(payload.get("trigger_word", "")).strip()
    only_missing = bool(payload.get("only_missing", True))
    _caption_runs[name] = {"status": "starting", "cancel": False}
    thread = threading.Thread(
        target=_caption_worker, args=(name, instruction, model, trigger, only_missing, key), daemon=True
    )
    thread.start()
    return {"started": True}


@app.get("/api/datasets/{name}/caption-status", dependencies=[Depends(require_auth)])
def caption_status(name: str) -> dict:
    return _caption_runs.get(name, {"status": "idle"})


@app.post("/api/datasets/{name}/caption-cancel", dependencies=[Depends(require_auth)])
def caption_cancel(name: str) -> dict:
    if name in _caption_runs:
        _caption_runs[name]["cancel"] = True
    return {"ok": True}


@app.get("/api/caption-default-instruction", dependencies=[Depends(require_auth)])
def caption_default() -> dict:
    return {"instruction": DEFAULT_CAPTION_INSTRUCTION}


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------
@app.get("/api/jobs", dependencies=[Depends(require_auth)])
def list_jobs() -> list[dict]:
    reap_finished()
    jobs = []
    for entry in sorted(JOBS_DIR.iterdir(), reverse=True):
        config_file = entry / "config.json"
        if config_file.exists():
            jobs.append({
                "id": entry.name,
                "status": job_status(entry.name),
                "config": json.loads(config_file.read_text()),
            })
    return jobs


@app.post("/api/jobs", dependencies=[Depends(require_auth)])
def create_job(payload: dict) -> dict:
    reap_finished()
    if any(proc.poll() is None for proc in _active.values()):
        raise HTTPException(status_code=409, detail="A job is already running. One GPU, one job.")

    dataset = safe_name(str(payload.get("dataset", "")))
    dataset_path = DATASETS_DIR / dataset
    if not dataset_path.is_dir() or not dataset_images(dataset_path):
        raise HTTPException(status_code=400, detail=f"Dataset '{dataset}' not found or empty.")

    model_id = str(payload.get("model_id", ""))
    model_entry = next((m for m in load_models() if m["id"] == model_id and m.get("enabled")), None)
    if model_entry is None:
        raise HTTPException(status_code=400, detail=f"Unknown or disabled model '{model_id}'.")
    trainer_script = APP_DIR / model_entry["trainer"]
    if not trainer_script.is_file():
        raise HTTPException(status_code=500, detail="Trainer script for this model is missing.")

    network = payload.get("network_type", "lora")
    if network not in model_entry["networks"]:
        raise HTTPException(status_code=400, detail=f"{model_entry['label']} does not support '{network}'.")

    prompts = payload.get("sample_prompts", [])
    if isinstance(prompts, str):
        prompts = [p for p in prompts.split("||") if p.strip()]
    prompts = [str(p).strip() for p in prompts if str(p).strip()]

    trigger = str(payload.get("trigger_word", "")).strip()
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    directory = JOBS_DIR / job_id
    directory.mkdir(parents=True)

    defaults = {
        "model_path": model_entry["default_path"], "resolution": 1024, "steps": 2000,
        "save_every": 250, "sample_every": 500, "sample_steps": 12, "batch_size": 1,
        "rank": 32, "lokr_factor": -1, "lokr_full_rank": 0, "learning_rate": "1e-4",
        "warmup_steps": 100, "target_modules": "identity", "optimizer": "paged_adamw8bit",
        "gradient_checkpointing": 1, "transformer_group_offload": 0, "group_offload_blocks": 1,
        "weight_decay": 0.01, "lokr_decompose_both": 0,
        "validation_image": "", "validation_prompt": "", "auto_analyze": False,
        "seed": 42,
    }
    config = {**defaults, **{k: payload[k] for k in defaults if k in payload}}
    config.update({
        "dataset": dataset, "trigger_word": trigger, "network_type": network,
        "model_id": model_id, "model_label": model_entry["label"], "sample_prompts": prompts,
    })
    (directory / "config.json").write_text(json.dumps(config, indent=2))

    rank = int(config["rank"])
    cmd = [
        sys.executable, str(trainer_script),
        "--pretrained_model_name_or_path", str(config["model_path"]),
        "--dataset_dir", str(dataset_path),
        "--output_dir", str(directory),
        "--run_name", "run",
        "--trigger_word", trigger,
        "--resolution", str(config["resolution"]),
        "--train_batch_size", str(config["batch_size"]),
        "--max_train_steps", str(config["steps"]),
        "--save_every_n_steps", str(config["save_every"]),
        "--sample_every_n_steps", str(config["sample_every"]),
        "--sample_num_inference_steps", str(config["sample_steps"]),
        "--sample_lora_scale", "1.0" if network == "lokr" else "1.35",
        "--sample_prompts", "||".join(prompts),
        "--network_type", network,
        "--rank", str(rank), "--lora_alpha", str(rank),
        "--lokr_factor", str(config["lokr_factor"]),
        "--lokr_full_rank", str(config["lokr_full_rank"]),
        "--lokr_decompose_both", str(config["lokr_decompose_both"]),
        "--learning_rate", str(config["learning_rate"]),
        "--weight_decay", str(config["weight_decay"]),
        "--lr_scheduler", "cosine",
        "--lr_warmup_steps", str(config["warmup_steps"]),
        "--target_modules", str(config["target_modules"]),
        "--optimizer", str(config["optimizer"]),
        "--gradient_checkpointing", str(config["gradient_checkpointing"]),
        "--transformer_group_offload", str(config["transformer_group_offload"]),
        "--group_offload_blocks", str(config["group_offload_blocks"]),
        "--validation_image", safe_name(str(config["validation_image"])) if str(config["validation_image"]).strip() else "",
        "--validation_prompt", str(config["validation_prompt"]),
        "--seed", str(config["seed"]),
        "--enable_wandb", "0",
    ]

    env = os.environ.copy()
    hf_token = str(payload.get("hf_token", "")).strip() or load_settings().get("hf_token") or env.get("HF_TOKEN", "")
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if payload.get("hf_token") and payload.get("save_hf_token"):
        settings = load_settings()
        settings["hf_token"] = hf_token
        save_settings(settings)

    log_file = (directory / "log.txt").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                            cwd=str(APP_DIR), env=env, start_new_session=True)
    _active[job_id] = proc
    set_status(job_id, "running")
    return {"id": job_id, "status": "running"}


@app.post("/api/jobs/{job_id}/stop", dependencies=[Depends(require_auth)])
def stop_job(job_id: str) -> dict:
    proc = _active.get(job_id)
    if proc is None or proc.poll() is not None:
        raise HTTPException(status_code=400, detail="Job is not running.")
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    for _ in range(50):
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    _active.pop(job_id, None)
    set_status(job_id, "stopped")
    return {"id": job_id, "status": "stopped"}


@app.delete("/api/jobs/{job_id}", dependencies=[Depends(require_auth)])
def delete_job(job_id: str) -> dict:
    if _active.get(job_id) and _active[job_id].poll() is None:
        raise HTTPException(status_code=409, detail="Stop the job before deleting it.")
    directory = job_dir(job_id)
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Job not found.")
    shutil.rmtree(directory)
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/metrics", dependencies=[Depends(require_auth)])
def job_metrics(job_id: str, max_points: int = 1200) -> dict:
    metrics_file = job_dir(job_id) / "run" / "metrics.jsonl"
    points: list[dict] = []
    if metrics_file.exists():
        for line in metrics_file.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                points.append(json.loads(line))
            except Exception:
                continue
    if len(points) > max_points:
        stride = len(points) / max_points
        points = [points[int(i * stride)] for i in range(max_points - 1)] + [points[-1]]
    return {"status": job_status(job_id), "points": points}


@app.get("/api/jobs/{job_id}/log", dependencies=[Depends(require_auth)])
def job_log(job_id: str, tail: int = 120) -> JSONResponse:
    log_file = job_dir(job_id) / "log.txt"
    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:] if log_file.exists() else []
    return JSONResponse({"status": job_status(job_id), "lines": lines})


@app.get("/api/jobs/{job_id}/samples", dependencies=[Depends(require_auth)])
def job_samples(job_id: str) -> list[dict]:
    samples_root = job_dir(job_id) / "run" / "samples"
    groups = []
    if samples_root.is_dir():
        for step_folder in sorted(samples_root.iterdir()):
            if step_folder.is_dir():
                groups.append({"step": step_folder.name, "images": sorted(p.name for p in step_folder.glob("*.png"))})
    return groups


@app.get("/api/jobs/{job_id}/samples/{step}/{image}", dependencies=[Depends(require_auth)])
def sample_image(job_id: str, step: str, image: str) -> FileResponse:
    path = job_dir(job_id) / "run" / "samples" / safe_name(step) / safe_name(image)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Sample not found.")
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/checkpoints", dependencies=[Depends(require_auth)])
def job_checkpoints(job_id: str) -> list[dict]:
    checkpoints_root = job_dir(job_id) / "run" / "checkpoints"
    out = []
    if checkpoints_root.is_dir():
        for step_folder in sorted(checkpoints_root.iterdir()):
            native = step_folder / "krea2_comfy_native_lora.safetensors"
            if native.is_file():
                out.append({"step": step_folder.name, "size_mb": round(native.stat().st_size / 1e6, 1)})
    return out


@app.get("/api/jobs/{job_id}/checkpoints/{step}/download", dependencies=[Depends(require_auth)])
def download_checkpoint(job_id: str, step: str) -> FileResponse:
    path = job_dir(job_id) / "run" / "checkpoints" / safe_name(step) / "krea2_comfy_native_lora.safetensors"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Checkpoint not found.")
    config = json.loads((job_dir(job_id) / "config.json").read_text())
    step_num = step.removeprefix("step-").lstrip("0") or "0"
    filename = safe_name(f"{config['dataset']}_{config['model_id']}_{step_num}") + ".safetensors"
    return FileResponse(path, filename=filename)


# --------------------------------------------------------------------------
# Gemini checkpoint analysis: loss curve + samples -> best candidates
# --------------------------------------------------------------------------
@app.post("/api/jobs/{job_id}/analyze", dependencies=[Depends(require_auth)])
def analyze_job(job_id: str) -> dict:
    return run_analysis(job_id)


def run_analysis(job_id: str) -> dict:
    key = gemini_key()
    model = load_settings().get("gemini_model") or "gemini-2.5-flash"
    directory = job_dir(job_id) / "run"

    metrics = job_metrics(job_id, max_points=150)["points"]
    config = json.loads((job_dir(job_id) / "config.json").read_text())
    status = job_status(job_id)
    last_step = metrics[-1]["step"] if metrics else 0
    loss_summary = [
        {"step": p["step"], "loss": p["loss"]} | ({"val_loss": p["val_loss"]} if p.get("val_loss") is not None else {})
        for p in metrics
    ]

    run_state = (
        f"This run is still IN PROGRESS: {last_step}/{config.get('steps', '?')} steps so far. "
        "Don't assume it has finished or converged — judge only what's visible so far, and don't "
        "suggest changes to settings that already match the config below."
        if status == "running" else
        f"This run has ENDED (status: {status}) at step {last_step}/{config.get('steps', '?')}."
    )

    has_val_loss = any("val_loss" in p for p in loss_summary)
    val_note = (
        " Points with val_loss were measured on a held-out image never seen during training, at "
        "fixed noise levels/seed each time — that's the reliable convergence signal, trust it over "
        "the raw per-step loss, which is noisy by nature (single-image batches, random timesteps)."
        if has_val_loss else
        " No validation loss was configured for this run, so judge convergence from the loss EMA "
        "trend and the sample images rather than raw per-step loss, which is inherently noisy."
    )

    parts: list[dict] = [{
        "text": (
            "You are an expert LoRA/LoKr character-identity trainer. Below is the FULL training "
            "config actually used (do not assume or guess any setting not shown — everything "
            "relevant is included), the run's current state, the downsampled loss curve, and one "
            "sample image per saved checkpoint (labelled with its step). The goal is a character "
            "identity adapter: judge likeness stability, overfitting signs (plastic skin, rigid "
            "pose, artifacting, burned contrast) and pick the 1-2 best candidate checkpoints. "
            "Respond with: (1) best candidate step(s) and why, (2) over/underfitting verdict, "
            "(3) one concrete suggestion for the next run — only if the current config doesn't "
            f"already reflect it. Be concise.{val_note}\n\n"
            f"Run state: {run_state}\n"
            f"Full config: {json.dumps(config)}\n"
            f"Loss curve: {json.dumps(loss_summary)}"
        )
    }]

    samples_root = directory / "samples"
    attached = 0
    if samples_root.is_dir():
        for step_folder in sorted(samples_root.iterdir()):
            lora_images = sorted(step_folder.glob("lora_*.png")) or sorted(step_folder.glob("*.png"))
            if lora_images and attached < 10:
                parts.append({"text": f"Sample from {step_folder.name}:"})
                parts.append(encode_image_for_gemini(lora_images[0], max_side=768))
                attached += 1
    if attached == 0:
        parts.append({"text": "No sample images available; judge from the loss curve alone."})

    try:
        verdict = gemini_generate(model, parts, key, timeout=180)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    (job_dir(job_id) / "analysis.md").write_text(verdict, encoding="utf-8")
    return {"analysis": verdict, "samples_attached": attached, "model": model}


@app.get("/api/jobs/{job_id}/analysis", dependencies=[Depends(require_auth)])
def get_analysis(job_id: str) -> dict:
    path = job_dir(job_id) / "analysis.md"
    return {"analysis": path.read_text(encoding="utf-8") if path.exists() else ""}


# --------------------------------------------------------------------------
# Self-update
# --------------------------------------------------------------------------
@app.post("/api/update", dependencies=[Depends(require_auth)])
def self_update() -> dict:
    reap_finished()
    if any(proc.poll() is None for proc in _active.values()):
        raise HTTPException(status_code=409, detail="Stop the running job before updating.")
    os.execv("/start.sh", ["/start.sh"])
    return {"ok": True}
