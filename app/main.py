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
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

import requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.background import BackgroundTask

from . import enhance
from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
JOBS_DIR = WORKSPACE / "jobs"
DATASETS_DIR = WORKSPACE / "datasets"
SETTINGS_FILE = WORKSPACE / ".day0" / "settings.json"
UI_PASSWORD = os.environ.get("UI_PASSWORD", "")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS
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


DATASET_CACHE_DIRS = {".thumbs", ".enhance"}


def dataset_images(path: Path) -> list[Path]:
    return sorted(
        p for p in path.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and DATASET_CACHE_DIRS.isdisjoint(p.relative_to(path).parts)
    )


def dataset_videos(path: Path) -> list[Path]:
    return sorted(
        p for p in path.rglob("*")
        if p.suffix.lower() in VIDEO_EXTS and DATASET_CACHE_DIRS.isdisjoint(p.relative_to(path).parts)
    )


def dataset_media(path: Path) -> list[Path]:
    """Images and videos together, e.g. for listing/export -- most other call sites want just one
    or the other (the Gemini captioner runs per caption method -- image or video -- against just
    that subset; the enhance panel is image-only; MiniMax-H3 training data is video-only), so this
    is only for the places that genuinely need both."""
    return sorted(dataset_images(path) + dataset_videos(path), key=lambda p: p.name)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0: no-op, just checks whether the PID exists and is ours to signal
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else -- shouldn't happen here, but not "dead"
    return True


def job_status(job_id: str) -> str:
    proc = _active.get(job_id)
    if proc is not None:
        code = proc.poll()
        if code is None:
            return "running"
        _active.pop(job_id, None)
        status = "finished" if code == 0 else "failed"
        # This branch runs on every /api/jobs poll while the job is still tracked in `_active`, so
        # it's the one that actually determines what the UI shows first, before reap_finished()'s
        # own periodic sweep would otherwise catch the same transition -- needs the same exit-code
        # logging, or a "failed" status with a clean "Training complete." log and no traceback stays
        # just as unexplained as it was before (see reap_finished() for the full reasoning).
        set_status(job_id, status, exit_code=code)
        if status == "failed":
            detail = f"signal {-code}" if code < 0 else f"exit code {code}"
            try:
                with (job_dir(job_id) / "log.txt").open("a", encoding="utf-8") as fh:
                    fh.write(f"\n[day0] Process ended: {detail}.\n")
            except Exception:
                pass
        elif status == "finished":
            maybe_auto_analyze(job_id)
        return status
    status_file = job_dir(job_id) / "status.json"
    if not status_file.exists():
        return "unknown"
    try:
        data = json.loads(status_file.read_text())
    except Exception:
        return "unknown"
    status, pid = data.get("status"), data.get("pid")
    # `_active` is in-memory only and empties out on every server restart -- a job that was
    # "running" when the server went down (e.g. via the "Update trainer" restart) would otherwise
    # be stuck reporting "running" forever, dead or not, since nothing would ever re-check it. If
    # we have its PID, verify it's actually still alive rather than trusting the last-written status.
    if status == "running" and pid and not _pid_alive(pid):
        status = "failed"
        set_status(job_id, status)
    return status or "unknown"


def set_status(job_id: str, status: str, pid: int | None = None, exit_code: int | None = None) -> None:
    payload = {"status": status, "ts": time.time()}
    status_file = job_dir(job_id) / "status.json"
    if pid is not None:
        payload["pid"] = pid
    elif status_file.exists():
        # Preserve a previously-recorded pid (e.g. when reap_finished() updates status for a job
        # that's still tracked in `_active` this session) instead of dropping it.
        try:
            existing_pid = json.loads(status_file.read_text()).get("pid")
            if existing_pid is not None:
                payload["pid"] = existing_pid
        except Exception:
            pass
    if exit_code is not None:
        payload["exit_code"] = exit_code
    status_file.write_text(json.dumps(payload))


def reap_finished() -> None:
    for job_id in list(_active):
        code = _active[job_id].poll()
        if code is not None:
            _active.pop(job_id, None)
            status = "finished" if code == 0 else "failed"
            set_status(job_id, status, exit_code=code)
            if status == "failed":
                # Popen.returncode is negative-N when the process was killed by signal N (e.g. -11
                # SIGSEGV, -6 SIGABRT), positive for a normal nonzero sys.exit()/os._exit(). This is
                # the one piece of information "failed" alone doesn't carry -- a MiniMax-H3 run that
                # printed "Training complete." with no traceback and still failed needs exactly this
                # to tell a genuine training bug apart from a CUDA-teardown crash after the fact, and
                # it wasn't visible anywhere before (job_status() only ever exposed the string).
                detail = f"signal {-code}" if code < 0 else f"exit code {code}"
                try:
                    with (job_dir(job_id) / "log.txt").open("a", encoding="utf-8") as fh:
                        fh.write(f"\n[day0] Process ended: {detail}.\n")
                except Exception:
                    pass
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


VIDEO_MIME_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska", ".webm": "video/webm"}
GEMINI_INLINE_LIMIT_BYTES = 19 * 1024 * 1024  # Gemini's inline-request cap is ~20MB total; leave headroom for the prompt text.


def _compress_video_for_gemini(path: Path) -> Path:
    """Gemini only needs to see the motion clearly enough to caption/locate it, not the quality the
    trainer will actually use (which downscales to its own ~768px working resolution regardless of
    source anyway) -- so a large source clip gets a throwaway compressed copy instead of being
    rejected outright. The real dataset file on disk is never touched."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to compress this video for Gemini but was not found.")
    handle, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(handle)
    dest = Path(tmp_path)
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(path),
         "-vf", "scale='min(480,iw)':'min(480,ih)':force_original_aspect_ratio=decrease",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-an", str(dest)],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0 or not dest.is_file():
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Could not compress {path.name} for Gemini: {result.stderr[-300:]}")
    if dest.stat().st_size > GEMINI_INLINE_LIMIT_BYTES:
        size_mb = dest.stat().st_size / 1e6
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"{path.name} is still {size_mb:.1f}MB after compression (~20MB Gemini limit). "
            "Trim it shorter with the ✂ Trim tool first."
        )
    return dest


def encode_video_for_gemini(path: Path) -> dict:
    """Gemini reads actual video content natively (motion across frames, not just one still), which
    is the whole point for video training clips -- but only inline for files under its ~20MB request
    cap. Files under the cap go through as-is; larger ones get compressed first (see
    _compress_video_for_gemini)."""
    size = path.stat().st_size
    if size <= GEMINI_INLINE_LIMIT_BYTES:
        mime_type = VIDEO_MIME_TYPES.get(path.suffix.lower(), "video/mp4")
        data = base64.b64encode(path.read_bytes()).decode()
        return {"inline_data": {"mime_type": mime_type, "data": data}}

    compressed = _compress_video_for_gemini(path)
    try:
        data = base64.b64encode(compressed.read_bytes()).decode()
        return {"inline_data": {"mime_type": "video/mp4", "data": data}}
    finally:
        compressed.unlink(missing_ok=True)


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
        "civitai_key_set": bool(settings.get("civitai_api_key") or os.environ.get("CIVITAI_API_KEY")),
    }


@app.put("/api/settings", dependencies=[Depends(require_auth)])
def put_settings(payload: dict) -> dict:
    settings = load_settings()
    for key in ("hf_token", "gemini_api_key", "gemini_model", "civitai_api_key"):
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
    """Move all media files -- images and videos -- (and their caption txts) to the dataset root."""
    for item in dataset_media(target):
        if item.parent != target:
            destination = target / item.name
            if not destination.exists():
                shutil.move(str(item), destination)
            caption = item.with_suffix(".txt")
            if caption.exists() and not (target / caption.name).exists():
                shutil.move(str(caption), target / caption.name)
    for entry in list(target.iterdir()):
        if entry.name == ".trigger":
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        elif entry.suffix.lower() not in MEDIA_EXTS | {".txt"}:
            entry.unlink()


@app.get("/api/datasets", dependencies=[Depends(require_auth)])
def list_datasets() -> list[dict]:
    out = []
    for entry in sorted(DATASETS_DIR.iterdir()):
        if entry.is_dir():
            images = dataset_images(entry)
            videos = dataset_videos(entry)
            captioned = sum(1 for item in images + videos if item.with_suffix(".txt").exists())
            trigger_file = entry / ".trigger"
            trigger_word = trigger_file.read_text(encoding="utf-8").strip() if trigger_file.exists() else ""
            out.append({
                "name": entry.name, "images": len(images), "videos": len(videos), "captioned": captioned,
                "trigger_word": trigger_word,
            })
    return out


@app.post("/api/datasets/{name}/files", dependencies=[Depends(require_auth)])
async def add_files(name: str, files: list[UploadFile] = File(...)) -> dict:
    """Universal intake: archives (.zip/.rar/.7z), loose images, videos, caption txts.

    The frontend flattens dropped folders into individual files before upload,
    so a folder drop arrives here as loose images/videos + txts.
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
        elif suffix not in MEDIA_EXTS | {".txt"}:
            destination.unlink()
    flatten_dataset(target)
    images = dataset_images(target)
    videos = dataset_videos(target)
    return {"name": target.name, "images": len(images), "videos": len(videos), "archives_extracted": added_archives}


@app.delete("/api/datasets/{name}", dependencies=[Depends(require_auth)])
def delete_dataset(name: str) -> dict:
    target = dataset_dir(name)
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="Dataset not found.")
    shutil.rmtree(target)
    return {"deleted": name}


@app.get("/api/datasets/{name}/export", dependencies=[Depends(require_auth)])
def export_dataset(name: str) -> FileResponse:
    """Zip media (images/videos) + captions + trigger word so the dataset can be re-dragged into a fresh pod."""
    target = dataset_dir(name)
    media = dataset_media(target)
    if not media:
        raise HTTPException(status_code=404, detail="Dataset not found or has no media.")
    handle, tmp_path = tempfile.mkstemp(suffix=".zip")
    os.close(handle)
    zip_path = Path(tmp_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in media:
            zf.write(item, item.name)
            caption = item.with_suffix(".txt")
            if caption.exists():
                zf.write(caption, caption.name)
        trigger_file = target / ".trigger"
        if trigger_file.exists():
            zf.write(trigger_file, ".trigger")
    return FileResponse(
        zip_path, filename=f"{target.name}.zip", media_type="application/zip",
        background=BackgroundTask(zip_path.unlink, missing_ok=True),
    )


@app.get("/api/datasets/{name}/items", dependencies=[Depends(require_auth)])
def dataset_items(name: str) -> list[dict]:
    target = dataset_dir(name)
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="Dataset not found.")
    out = []
    for item in dataset_media(target):
        is_video = item.suffix.lower() in VIDEO_EXTS
        caption_file = item.with_suffix(".txt")
        width = height = None
        if not is_video:
            try:
                width, height = Image.open(item).size  # header-only read, no full decode
            except Exception:
                pass
        out.append({
            "image": item.name, "is_video": is_video,
            "caption": caption_file.read_text(encoding="utf-8", errors="replace") if caption_file.exists() else "",
            "width": width, "height": height,
        })
    return out


@app.get("/api/datasets/{name}/raw/{video}", dependencies=[Depends(require_auth)])
def dataset_raw(name: str, video: str) -> FileResponse:
    """Serves the actual video file (not a thumbnail) for the trim tool's <video> player."""
    source = dataset_dir(name) / safe_name(video)
    if not source.is_file() or source.suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(source)


@app.get("/api/datasets/{name}/thumb/{image}", dependencies=[Depends(require_auth)])
def dataset_thumb(name: str, image: str, size: int = 220) -> FileResponse:
    source = dataset_dir(name) / safe_name(image)
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Media not found.")
    thumbs = dataset_dir(name) / ".thumbs"
    thumbs.mkdir(exist_ok=True)
    thumb = thumbs / f"{size}_{source.name}.jpg"
    if not thumb.exists() or thumb.stat().st_mtime < source.stat().st_mtime:
        if source.suffix.lower() in VIDEO_EXTS:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise HTTPException(status_code=500, detail="ffmpeg is required for video thumbnails but was not found.")
            frame = thumbs / f".frame_{source.name}.jpg"
            result = subprocess.run(
                [ffmpeg, "-y", "-ss", "0.5", "-i", str(source), "-frames:v", "1", str(frame)],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0 or not frame.is_file():
                raise HTTPException(status_code=500, detail="Could not extract a video thumbnail frame.")
            img = Image.open(frame).convert("RGB")
            frame.unlink(missing_ok=True)
        else:
            img = Image.open(source).convert("RGB")
        img.thumbnail((size, size))
        img.save(thumb, format="JPEG", quality=85)
    return FileResponse(thumb)


def _probe_video(source: Path) -> dict:
    """width/height/duration via ffprobe -- shared by video_meta and autoclip.

    The per-stream `duration` field ffprobe reports is frequently missing or wrong (many mp4/webm/mkv
    muxings just don't tag it on the video stream itself) -- the container-level `format.duration` is
    the reliable one, so that's read too and preferred whenever the stream value looks bogus."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise HTTPException(status_code=500, detail="ffprobe is required but was not found.")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration:format=duration", "-of", "json", str(source)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail="Could not read video metadata.")
    info = json.loads(result.stdout)
    stream = info.get("streams", [{}])[0]
    stream_duration = float(stream.get("duration") or 0)
    format_duration = float(info.get("format", {}).get("duration") or 0)
    # Prefer the container-level duration whenever the stream-level one is missing or clearly wrong
    # (e.g. reporting a fraction of a second for a file that's actually much longer).
    duration = format_duration if format_duration > stream_duration else stream_duration
    return {
        "width": int(stream.get("width", 0)), "height": int(stream.get("height", 0)),
        "duration": duration,
    }


@app.get("/api/datasets/{name}/video-meta/{video}", dependencies=[Depends(require_auth)])
def video_meta(name: str, video: str) -> dict:
    """Duration/dimensions for the trim tool's timeline + canvas-size preview."""
    source = dataset_dir(name) / safe_name(video)
    if not source.is_file() or source.suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(status_code=404, detail="Video not found.")
    return _probe_video(source)


def _ffmpeg_cut(ffmpeg: str, source: Path, start: float, end: float, dest: Path) -> None:
    # -ss after -i (input seeking) is frame-accurate, unlike the fast-but-keyframe-snapped
    # -ss-before-i form -- worth the slower re-encode for a short curation clip.
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-ss", str(start), "-to", str(end),
         "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac", str(dest)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Trim failed: {result.stderr[-300:]}")


def _clear_thumb_cache(target: Path, filename: str) -> None:
    thumb_dir = target / ".thumbs"
    if thumb_dir.is_dir():
        for stale in thumb_dir.glob(f"*_{filename}.jpg"):
            stale.unlink(missing_ok=True)


def _split_into_chunks(ffmpeg: str, target: Path, source: Path, start: float, end: float, chunk_seconds: float) -> list[str]:
    """Cut [start, end] of source into consecutive chunk_seconds-long pieces -- shared by the
    single-video trim tool and the bulk autoclip endpoint. The first chunk overwrites source in
    place; later chunks are new files sharing its caption as a starting point (edit each afterward,
    the same caption rarely fits every chunk exactly). Any leftover shorter than chunk_seconds at
    the end is dropped, not padded into its own clip."""
    n_chunks = max(1, int((end - start) // chunk_seconds))
    caption_file = source.with_suffix(".txt")
    caption_text = caption_file.read_text(encoding="utf-8") if caption_file.exists() else ""
    stem = source.stem

    # Cut every chunk from the original file first, into temp files -- only once all of them exist
    # is `source` itself overwritten. Cutting chunk 0 in place before cutting chunk 1 would mean
    # chunk 1 gets cut from the already-truncated chunk-0 output instead of the original footage.
    tmp_paths = []
    try:
        for i in range(n_chunks):
            chunk_start = start + i * chunk_seconds
            chunk_end = chunk_start + chunk_seconds
            handle, tmp_path = tempfile.mkstemp(suffix=source.suffix)
            os.close(handle)
            _ffmpeg_cut(ffmpeg, source, chunk_start, chunk_end, Path(tmp_path))
            tmp_paths.append(Path(tmp_path))

        created = []
        shutil.move(str(tmp_paths[0]), source)
        _clear_thumb_cache(target, source.name)
        created.append(source.name)
        for i, tmp_path in enumerate(tmp_paths[1:], start=2):
            dest = target / f"{stem}_part{i}{source.suffix}"
            shutil.move(str(tmp_path), dest)
            if caption_text:
                dest.with_suffix(".txt").write_text(caption_text, encoding="utf-8")
            created.append(dest.name)
    finally:
        for tmp_path in tmp_paths:
            tmp_path.unlink(missing_ok=True)  # no-op for any already moved into place
    return created


@app.post("/api/datasets/{name}/trim/{video}", dependencies=[Depends(require_auth)])
def trim_video(name: str, video: str, payload: dict) -> dict:
    """Cut a dataset video down to [start, end] seconds -- for shaping a raw clip into the in/out
    range actually worth training on before it ever reaches the trainer.

    With chunk_seconds set, the selected range is instead split into consecutive same-length
    pieces (e.g. a 9s selection at chunk_seconds=3 becomes three 3s clips) -- one longer take
    turned into several independent training examples instead of one, each still short enough to
    land in the clip-length range that actually teaches a single motion well. The original file
    becomes the first chunk in place; later chunks are added as new dataset items sharing its
    caption as a starting point (edit each afterward -- the same caption rarely fits every chunk
    exactly)."""
    target = dataset_dir(name)
    source = target / safe_name(video)
    if not source.is_file() or source.suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(status_code=404, detail="Video not found.")
    start = float(payload.get("start", 0))
    end = float(payload.get("end", 0))
    if start < 0 or end <= start:
        raise HTTPException(status_code=400, detail="Invalid trim range.")
    chunk_seconds = float(payload.get("chunk_seconds") or 0)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(status_code=500, detail="ffmpeg is required but was not found.")

    if chunk_seconds <= 0 or chunk_seconds >= (end - start):
        handle, tmp_path = tempfile.mkstemp(suffix=source.suffix)
        os.close(handle)
        tmp_out = Path(tmp_path)
        try:
            _ffmpeg_cut(ffmpeg, source, start, end, tmp_out)
            shutil.move(str(tmp_out), source)
        finally:
            tmp_out.unlink(missing_ok=True)
        _clear_thumb_cache(target, source.name)
        return {"ok": True, "clips_created": 1}

    created = _split_into_chunks(ffmpeg, target, source, start, end, chunk_seconds)
    return {"ok": True, "clips_created": len(created), "clips": created}


@app.post("/api/datasets/{name}/autoclip", dependencies=[Depends(require_auth)])
def autoclip_videos(name: str, payload: dict) -> dict:
    """Bulk version of the trim tool's chunk-splitting: cut every selected video (or every video in
    the dataset, if none are named) into consecutive same-length clips covering its full duration --
    for turning a folder of raw takes into short training-length clips in one pass instead of doing
    each one by hand in the trim tool. Same _partN naming and caption-copying as a manual chunked
    trim. Videos shorter than clip_seconds are skipped outright, not padded or force-included."""
    target = dataset_dir(name)
    clip_seconds = float(payload.get("clip_seconds") or 0)
    if clip_seconds <= 0:
        raise HTTPException(status_code=400, detail="clip_seconds must be > 0.")
    requested = payload.get("videos")
    videos = [target / safe_name(v) for v in requested] if requested else dataset_videos(target)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(status_code=500, detail="ffmpeg is required but was not found.")

    videos_processed, clips_created, skipped = 0, 0, []
    for source in videos:
        if not source.is_file() or source.suffix.lower() not in VIDEO_EXTS:
            skipped.append(source.name)
            continue
        duration = _probe_video(source)["duration"]
        if duration < clip_seconds:
            skipped.append(source.name)
            continue
        clips_created += len(_split_into_chunks(ffmpeg, target, source, 0, duration, clip_seconds))
        videos_processed += 1
    return {"ok": True, "videos_processed": videos_processed, "clips_created": clips_created, "skipped": skipped}


# --------------------------------------------------------------------------
# Smart autoclip (Gemini-assisted): asks Gemini where the motion actually is
# instead of slicing at fixed intervals, and captions each clip in the same
# call -- reuses the video understanding already used for captioning.
# --------------------------------------------------------------------------
_smart_clip_runs: dict[str, dict] = {}

SMART_CLIP_INSTRUCTION_TEMPLATE = (
    "You are curating a video LoRA training dataset. Watch this clip and identify up to 3 short "
    "segments where clear physical motion/action happens (not static shots, pans, or dead time). "
    "Each segment should be a single continuous moment, roughly {clip_seconds} seconds long, that "
    "on its own teaches the motion well. For each segment also write a dataset caption for it -- but "
    "the model must learn the motion itself as an implicit concept, not tied to text, so the caption "
    "must describe everything EXCEPT the motion: expression, outfit, pose or starting body position, "
    "hairstyle, camera angle, jewelry, accessories, setting, background, in concise natural positive "
    "phrasing, with no verbs of motion or action and no mention of what physically happens or "
    "changes. Respond with ONLY a JSON array, no markdown fences, no other text, in this exact shape: "
    "[{{\"start\": <seconds float>, \"end\": <seconds float>, \"caption\": \"<text>\"}}, ...]. If "
    "there isn't enough distinct motion for multiple segments, return fewer entries. Times must fall "
    "within the clip's actual duration."
)


def _parse_gemini_json_array(text: str) -> list:
    # Gemini sometimes wraps JSON in ```json ... ``` fences, or adds a stray sentence before/after
    # the array, despite being told to return only JSON -- strip fences first, then fall back to
    # slicing out the first [...] substring rather than failing the whole video on a format slip.
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1 or end < start:
            raise
        return json.loads(text[start:end + 1])


def _write_smart_caption(path: Path, caption: str, trigger: str) -> None:
    caption = (caption or "").strip()
    if not caption:
        return
    if trigger and trigger not in caption:
        caption = f"{trigger}, {caption}"
    path.with_suffix(".txt").write_text(caption, encoding="utf-8")


def _smart_clip_worker(name: str, videos: list[Path], clip_seconds: float, model: str, key: str, trigger: str) -> None:
    state = _smart_clip_runs[name]
    target = dataset_dir(name)
    ffmpeg = shutil.which("ffmpeg")
    state.update({"total": len(videos), "done": 0, "errors": [], "clips_created": 0, "status": "running"})
    for source in videos:
        if state.get("cancel"):
            state["status"] = "cancelled"
            return
        try:
            duration = _probe_video(source)["duration"]
            parts = [
                {"text": SMART_CLIP_INSTRUCTION_TEMPLATE.format(clip_seconds=clip_seconds)},
                encode_video_for_gemini(source),
            ]
            raw = gemini_generate(model, parts, key, timeout=120)
            segments = _parse_gemini_json_array(raw)[:3]
            valid = [
                s for s in segments
                if isinstance(s, dict) and 0 <= float(s.get("start", -1)) < float(s.get("end", -1))
                and float(s.get("start", -1)) < duration
            ]
            if not valid:
                state["errors"].append(f"{source.name}: no valid motion segments returned")
                state["done"] += 1
                continue

            stem = source.stem
            tmp_paths = []
            try:
                for seg in valid:
                    handle, tmp_path = tempfile.mkstemp(suffix=source.suffix)
                    os.close(handle)
                    _ffmpeg_cut(ffmpeg, source, float(seg["start"]), min(float(seg["end"]), duration), Path(tmp_path))
                    tmp_paths.append(Path(tmp_path))

                shutil.move(str(tmp_paths[0]), source)
                _clear_thumb_cache(target, source.name)
                _write_smart_caption(source, valid[0].get("caption", ""), trigger)
                for i, tmp_path in enumerate(tmp_paths[1:], start=2):
                    dest = target / f"{stem}_part{i}{source.suffix}"
                    shutil.move(str(tmp_path), dest)
                    _write_smart_caption(dest, valid[i - 1].get("caption", ""), trigger)
            finally:
                for tmp_path in tmp_paths:
                    tmp_path.unlink(missing_ok=True)  # no-op for any already moved into place
            state["clips_created"] += len(valid)
        except Exception as exc:
            state["errors"].append(f"{source.name}: {exc}")
        state["done"] += 1
    state["status"] = "finished"


@app.post("/api/datasets/{name}/autoclip/smart", dependencies=[Depends(require_auth)])
def autoclip_smart(name: str, payload: dict) -> dict:
    """Gemini-assisted autoclip: instead of slicing every video at fixed intervals, ask Gemini
    (already used for captioning, so this is one more use of the same call) to find up to 3 segments
    per video where the actual motion happens, and caption each segment in the same pass. One Gemini
    call per video does both clip selection and captioning."""
    if _smart_clip_runs.get(name, {}).get("status") == "running":
        raise HTTPException(status_code=409, detail="Smart autoclip already running for this dataset.")
    target = dataset_dir(name)
    clip_seconds = float(payload.get("clip_seconds") or 3)
    key = gemini_key()
    model = str(payload.get("model") or load_settings().get("gemini_model") or "gemini-2.5-flash")
    trigger = str(payload.get("trigger_word", "")).strip()
    requested = payload.get("videos")
    videos = [target / safe_name(v) for v in requested] if requested else dataset_videos(target)
    videos = [v for v in videos if v.is_file() and v.suffix.lower() in VIDEO_EXTS]
    if not videos:
        raise HTTPException(status_code=400, detail="No videos to process.")
    if not shutil.which("ffmpeg"):
        raise HTTPException(status_code=500, detail="ffmpeg is required but was not found.")
    _smart_clip_runs[name] = {"status": "starting", "cancel": False}
    thread = threading.Thread(
        target=_smart_clip_worker, args=(name, videos, clip_seconds, model, key, trigger), daemon=True
    )
    thread.start()
    return {"started": True}


@app.get("/api/datasets/{name}/autoclip/smart-status", dependencies=[Depends(require_auth)])
def autoclip_smart_status(name: str) -> dict:
    return _smart_clip_runs.get(name, {"status": "idle"})


@app.post("/api/datasets/{name}/autoclip/smart-cancel", dependencies=[Depends(require_auth)])
def autoclip_smart_cancel(name: str) -> dict:
    if name in _smart_clip_runs:
        _smart_clip_runs[name]["cancel"] = True
    return {"ok": True}


@app.put("/api/datasets/{name}/caption/{image}", dependencies=[Depends(require_auth)])
def set_caption(name: str, image: str, payload: dict) -> dict:
    source = dataset_dir(name) / safe_name(image)
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    source.with_suffix(".txt").write_text(str(payload.get("caption", "")).strip(), encoding="utf-8")
    return {"ok": True}


@app.delete("/api/datasets/{name}/images/{image}", dependencies=[Depends(require_auth)])
def delete_dataset_image(name: str, image: str) -> dict:
    target = dataset_dir(name)
    source = target / safe_name(image)
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    source.unlink()
    source.with_suffix(".txt").unlink(missing_ok=True)
    thumb_dir = target / ".thumbs"
    if thumb_dir.is_dir():
        for stale in thumb_dir.glob(f"*_{source.name}.jpg"):
            stale.unlink(missing_ok=True)
    shutil.rmtree(target / ".enhance" / source.name, ignore_errors=True)
    with _enhance_queue_lock:
        _enhance_queue.pop(f"{name}::{image}", None)
    return {"ok": True}


# --------------------------------------------------------------------------
# Optional image enhancement (diffusers' native Flux2KleinPipeline).
# Nothing downloads until /api/enhance/setup is explicitly called. Shares the
# GPU with training, so both guard against the other being active.
#
# Generation is queued and processed by a single background worker rather
# than run inline in the request -- 4 candidates take 2-4 minutes, long
# enough that RunPod's HTTP proxy kills the connection mid-request. The
# client polls /enhance/queue instead of waiting on one long POST.
# --------------------------------------------------------------------------
def _training_active() -> bool:
    reap_finished()
    return any(proc.poll() is None for proc in _active.values())


_enhance_queue_lock = threading.Lock()
_enhance_queue: dict[str, dict] = {}  # f"{dataset}::{image}" -> row state
_enhance_worker_active = False
_enhance_order_counter = 0


def _enhance_run_worker() -> None:
    global _enhance_worker_active
    while True:
        with _enhance_queue_lock:
            pending = sorted(
                (v for v in _enhance_queue.values() if v["status"] == "queued"),
                key=lambda v: v["order"],
            )
            item = pending[0] if pending else None
            if item is None:
                _enhance_worker_active = False
                return
            item["status"] = "running"
        try:
            source = dataset_dir(item["dataset"]) / safe_name(item["image"])
            out_dir = dataset_dir(item["dataset"]) / ".enhance" / safe_name(item["image"])
            if out_dir.exists():
                shutil.rmtree(out_dir)
            candidates = enhance.run_enhance(
                source, item["prompt"], item.get("count", 1), out_dir,
                lora_weight=item.get("lora_weight", enhance.DEFAULT_LORA_WEIGHT),
                max_megapixels=item.get("max_megapixels", 1.0),
            )
            with _enhance_queue_lock:
                item["status"] = "done"
                item["candidates"] = [c.name for c in candidates]
        except Exception as exc:
            with _enhance_queue_lock:
                item["status"] = "error"
                item["error"] = str(exc)


def _enhance_ensure_worker() -> None:
    global _enhance_worker_active
    with _enhance_queue_lock:
        if _enhance_worker_active:
            return
        _enhance_worker_active = True
    threading.Thread(target=_enhance_run_worker, daemon=True).start()


@app.get("/api/enhance/status", dependencies=[Depends(require_auth)])
def enhance_status() -> dict:
    return enhance.get_status()


@app.post("/api/enhance/setup", dependencies=[Depends(require_auth)])
def enhance_setup() -> dict:
    if _training_active():
        raise HTTPException(status_code=409, detail="A training job is running -- stop it first, enhance needs the same GPU.")
    status = enhance.get_status()["status"]
    if status in ("downloading", "starting"):
        raise HTTPException(status_code=409, detail="Setup already in progress.")
    if status == "ready":
        return {"status": "ready"}
    settings = load_settings()
    hf_token = settings.get("hf_token") or os.environ.get("HF_TOKEN", "")
    civitai_key = settings.get("civitai_api_key") or os.environ.get("CIVITAI_API_KEY", "")
    threading.Thread(target=enhance.setup_enhance, args=(hf_token, civitai_key), daemon=True).start()
    return {"status": "starting"}


@app.get("/api/enhance/default_prompt", dependencies=[Depends(require_auth)])
def enhance_default_prompt() -> dict:
    return {"prompt": enhance.DEFAULT_ENHANCE_PROMPT, "lora_weight": enhance.DEFAULT_LORA_WEIGHT}


@app.post("/api/datasets/{name}/enhance/queue", dependencies=[Depends(require_auth)])
def enhance_queue_add(name: str, payload: dict) -> dict:
    if not enhance.is_ready():
        raise HTTPException(status_code=409, detail="Enhance isn't set up yet.")
    if _training_active():
        raise HTTPException(status_code=409, detail="A training job is running -- stop it first, enhance needs the same GPU.")
    images = payload.get("images") or []
    if not images:
        raise HTTPException(status_code=400, detail="No images given.")
    prompt = str(payload.get("prompt") or enhance.DEFAULT_ENHANCE_PROMPT)
    try:
        lora_weight = float(payload.get("lora_weight", enhance.DEFAULT_LORA_WEIGHT))
    except (TypeError, ValueError):
        lora_weight = enhance.DEFAULT_LORA_WEIGHT
    try:
        count = max(1, min(4, int(payload.get("count", 1))))
    except (TypeError, ValueError):
        count = 1
    try:
        max_megapixels = max(0.5, min(2.0, float(payload.get("max_megapixels", 1.0))))
    except (TypeError, ValueError):
        max_megapixels = 1.0
    global _enhance_order_counter
    queued = []
    with _enhance_queue_lock:
        for image in images:
            source = dataset_dir(name) / safe_name(image)
            if not source.is_file():
                continue
            key = f"{name}::{image}"
            _enhance_order_counter += 1
            _enhance_queue[key] = {
                "dataset": name, "image": image, "status": "queued", "count": count,
                "candidates": [], "prompt": prompt, "lora_weight": lora_weight,
                "max_megapixels": max_megapixels, "error": "",
                "order": _enhance_order_counter,
            }
            queued.append(key)
    _enhance_ensure_worker()
    return {"queued": queued}


@app.get("/api/datasets/{name}/enhance/queue", dependencies=[Depends(require_auth)])
def enhance_queue_status(name: str) -> dict:
    with _enhance_queue_lock:
        items = [dict(v) for v in _enhance_queue.values() if v["dataset"] == name]
    items.sort(key=lambda v: v["order"])
    return {"items": items}


@app.delete("/api/datasets/{name}/enhance/{image}", dependencies=[Depends(require_auth)])
def enhance_dismiss(name: str, image: str) -> dict:
    with _enhance_queue_lock:
        _enhance_queue.pop(f"{name}::{image}", None)
    shutil.rmtree(dataset_dir(name) / ".enhance" / safe_name(image), ignore_errors=True)
    return {"ok": True}


@app.get("/api/datasets/{name}/enhance/{image}/candidate/{filename}", dependencies=[Depends(require_auth)])
def enhance_candidate(name: str, image: str, filename: str) -> FileResponse:
    path = dataset_dir(name) / ".enhance" / safe_name(image) / safe_name(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Candidate not found.")
    return FileResponse(path)


@app.post("/api/datasets/{name}/enhance/{image}/accept", dependencies=[Depends(require_auth)])
def enhance_accept(name: str, image: str, payload: dict) -> dict:
    source = dataset_dir(name) / safe_name(image)
    candidate = dataset_dir(name) / ".enhance" / safe_name(image) / safe_name(str(payload.get("candidate", "")))
    if not source.is_file() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image or candidate not found.")
    shutil.copy2(candidate, source)
    thumb_dir = dataset_dir(name) / ".thumbs"
    if thumb_dir.is_dir():
        for stale in thumb_dir.glob(f"*_{source.name}.jpg"):
            stale.unlink(missing_ok=True)
    shutil.rmtree(candidate.parent, ignore_errors=True)
    with _enhance_queue_lock:
        _enhance_queue.pop(f"{name}::{image}", None)
    return {"ok": True}


@app.post("/api/datasets/{name}/enhance/{image}/promote", dependencies=[Depends(require_auth)])
def enhance_promote(name: str, image: str, payload: dict) -> dict:
    source_dir = dataset_dir(name)
    candidate = source_dir / ".enhance" / safe_name(image) / safe_name(str(payload.get("candidate", "")))
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Candidate not found.")
    stem, suffix = Path(safe_name(image)).stem, Path(safe_name(image)).suffix
    n = 1
    while (source_dir / f"{stem}_enh{n}{suffix}").exists():
        n += 1
    dest = source_dir / f"{stem}_enh{n}{suffix}"
    shutil.copy2(candidate, dest)
    orig_caption = source_dir / f"{stem}.txt"
    if orig_caption.is_file():
        shutil.copy2(orig_caption, source_dir / f"{stem}_enh{n}.txt")
    return {"ok": True, "filename": dest.name}


@app.post("/api/datasets/{name}/enhance/{image}/gemini_pick", dependencies=[Depends(require_auth)])
def enhance_gemini_pick(name: str, image: str) -> dict:
    key = gemini_key()
    model = load_settings().get("gemini_model") or "gemini-2.5-flash"
    source = dataset_dir(name) / safe_name(image)
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    cand_dir = dataset_dir(name) / ".enhance" / safe_name(image)
    candidates = sorted(cand_dir.glob("candidate_*.png")) if cand_dir.is_dir() else []
    if not candidates:
        raise HTTPException(status_code=400, detail="No candidates to analyze yet.")

    parts: list[dict] = [{
        "text": (
            "You are judging AI-enhanced versions of a training-dataset photo. The goal of the "
            "enhancement is to make skin/texture look more like a real amateur phone photo -- NOT "
            "smooth, plastic, or AI-looking -- while keeping the same person, pose, outfit, and "
            f"framing as the original. Below is the ORIGINAL image, followed by {len(candidates)} "
            "candidate(s), each labelled by filename. Pick the single best candidate, or say none "
            "are good and the original should be kept. Watch for identity drift, warped anatomy, "
            "changed pose/outfit, or over-smoothing that undoes the point of the enhancement. "
            "Respond with the winning candidate's exact filename (or 'none') on the first line, "
            "then one or two sentences why."
        )
    }]
    parts.append({"text": "ORIGINAL:"})
    parts.append(encode_image_for_gemini(source, max_side=768))
    for cand in candidates:
        parts.append({"text": f"{cand.name}:"})
        parts.append(encode_image_for_gemini(cand, max_side=768))

    try:
        verdict = gemini_generate(model, parts, key, timeout=90)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"analysis": verdict, "model": model}


# --------------------------------------------------------------------------
# Captioner (Gemini, background batch)
# --------------------------------------------------------------------------
DEFAULT_CAPTION_INSTRUCTION = (
    "You are a world-class AI model specialist for image generation. "
    "Caption the image for dataset creation. Mention only: expression, outfit, pose, "
    "hairstyle, camera angle, whether there is blur in the photo, jewelry, accessories, "
    "setting, background. Do not mention face shape or body type. Caption in concise "
    "but detailed natural language. Output only the caption, no preamble. If a tattoo is "
    "visible, caption it. Caption only what you see, in natural positive phrasing only -- "
    "do not use words like 'no', 'or', 'not'. For example, 'face out of frame' is allowed, "
    "but 'no visible jewelry' is not allowed -- just don't mention things that aren't there."
)

# Deliberately the opposite of the image instruction's completeness: the whole point of a video
# LoRA dataset is for the model to learn the motion itself as an implicit concept, not one tied to
# a text description, so the caption must describe everything EXCEPT what's moving or changing.
# Captioning the motion here would teach the model to only reproduce that motion when the exact
# caption text is present, rather than baking it in as the dataset's constant, unlabeled trait.
DEFAULT_VIDEO_CAPTION_INSTRUCTION = (
    "You are a world-class AI model specialist for video generation LoRA datasets. This is a short "
    "video clip. The model must learn the motion/action happening in it as an implicit concept, not "
    "tied to any text description, so caption everything EXCEPT the motion. Mention only the static, "
    "held-constant elements visible throughout the clip: expression, outfit, pose or starting body "
    "position, hairstyle, camera angle, whether there is blur, jewelry, accessories, setting, "
    "background. Do not mention face shape or body type. Do NOT describe what physically happens, "
    "moves, or changes over the clip -- no verbs of motion or action at all. Caption in concise but "
    "detailed natural language. Output only the caption, no preamble. If a tattoo is visible, caption "
    "it. Caption only what you see, in natural positive phrasing only -- do not use words like 'no', "
    "'or', 'not'."
)


def wrap_ideogram4_caption(text: str, trigger: str) -> dict:
    """Ports the caption JSON shape diffusion-pipe's Ideogram4 config expects (see
    IDEOGRAM/Train_Ideogram4_DiffusionPipe_FIXED.sh's wrap_plain_caption). Built
    deterministically in Python rather than asking Gemini to emit JSON directly --
    VLMs are unreliable at exact bbox coordinates and hex color codes."""
    text = text.strip()
    if trigger and trigger not in text:
        text = f"{trigger}, {text}"
    return {
        "high_level_description": f"A realistic casual lifestyle photograph of {trigger} as the main character. {text}",
        "style_description": {
            "aesthetics": "Realistic social-media lifestyle portrait, identity-focused character reference.",
            "lighting": "Natural available light with realistic smartphone-photo exposure.",
            "photo": "Sharp smartphone photograph with natural facial features and visible hair detail.",
            "medium": "Photograph.",
            "color_palette": ["#111111", "#6B5145", "#B98972", "#E8E0D4"],
        },
        "compositional_deconstruction": {
            "background": "Casual lifestyle environment, secondary to the main character.",
            "elements": [{"type": "obj", "bbox": [150, 80, 880, 980], "desc": text}],
        },
    }


def _caption_worker(
    name: str, instruction: str, model: str, trigger: str, only_missing: bool, key: str, caption_format: str, method: str
) -> None:
    state = _caption_runs[name]
    target = dataset_dir(name)
    items = dataset_videos(target) if method == "video" else dataset_images(target)
    if only_missing:
        items = [item for item in items if not item.with_suffix(".txt").exists()]
    state.update({"total": len(items), "done": 0, "errors": [], "status": "running"})
    for item in items:
        if state.get("cancel"):
            state["status"] = "cancelled"
            return
        try:
            if method == "video":
                parts = [{"text": instruction}, encode_video_for_gemini(item)]
            else:
                parts = [{"text": instruction}, encode_image_for_gemini(item)]
            caption = gemini_generate(model, parts, key).replace("\n", " ").strip()
            if trigger and trigger not in caption:
                caption = f"{trigger}, {caption}"
            if caption_format == "ideogram4_json":
                caption = json.dumps(
                    wrap_ideogram4_caption(caption, trigger), ensure_ascii=False, separators=(",", ":")
                )
            item.with_suffix(".txt").write_text(caption, encoding="utf-8")
        except Exception as exc:
            state["errors"].append(f"{item.name}: {exc}")
        state["done"] += 1
    state["status"] = "finished"


@app.post("/api/datasets/{name}/caption-all", dependencies=[Depends(require_auth)])
def caption_all(name: str, payload: dict) -> dict:
    if _caption_runs.get(name, {}).get("status") == "running":
        raise HTTPException(status_code=409, detail="Captioning already running for this dataset.")
    key = gemini_key()
    model = str(payload.get("model") or load_settings().get("gemini_model") or "gemini-2.5-flash")
    method = str(payload.get("method") or "image")
    default_instruction = DEFAULT_VIDEO_CAPTION_INSTRUCTION if method == "video" else DEFAULT_CAPTION_INSTRUCTION
    instruction = str(payload.get("instruction") or default_instruction)
    trigger = str(payload.get("trigger_word", "")).strip()
    only_missing = bool(payload.get("only_missing", True))
    caption_format = str(payload.get("caption_format") or "plain")
    if trigger:
        (dataset_dir(name) / ".trigger").write_text(trigger, encoding="utf-8")
    _caption_runs[name] = {"status": "starting", "cancel": False}
    thread = threading.Thread(
        target=_caption_worker, args=(name, instruction, model, trigger, only_missing, key, caption_format, method), daemon=True
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
def caption_default(method: str = "image") -> dict:
    return {"instruction": DEFAULT_VIDEO_CAPTION_INSTRUCTION if method == "video" else DEFAULT_CAPTION_INSTRUCTION}


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
    if enhance.is_ready():
        enhance.unload()  # frees its ~20GB+ before training claims the GPU

    model_id = str(payload.get("model_id", ""))
    model_entry = next((m for m in load_models() if m["id"] == model_id and m.get("enabled")), None)
    if model_entry is None:
        raise HTTPException(status_code=400, detail=f"Unknown or disabled model '{model_id}'.")
    trainer_script = APP_DIR / model_entry["trainer"]
    if not trainer_script.is_file():
        raise HTTPException(status_code=500, detail="Trainer script for this model is missing.")

    dataset = safe_name(str(payload.get("dataset", "")))
    dataset_path = DATASETS_DIR / dataset
    # MiniMax-H3 trains on video clips, everything else on stills -- a dataset with the wrong
    # media type for the selected model is empty as far as that trainer is concerned.
    if model_entry["arch"] == "minimax_h3":
        if not dataset_path.is_dir() or not dataset_videos(dataset_path):
            raise HTTPException(status_code=400, detail=f"Dataset '{dataset}' not found or has no videos.")
    elif not dataset_path.is_dir() or not dataset_images(dataset_path):
        raise HTTPException(status_code=400, detail=f"Dataset '{dataset}' not found or empty.")

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
        "sample_inference_model": "", "sample_guidance_scale": 4.0,
        "rank": 32, "lokr_factor": -1, "lokr_full_rank": 0, "learning_rate": "1e-4",
        "target_modules": "identity", "optimizer": "paged_adamw8bit",
        "gradient_checkpointing": 1, "transformer_group_offload": 0, "group_offload_blocks": 1,
        "weight_decay": 0.01, "lokr_decompose_both": 0,
        "validation_image": "", "validation_prompt": "", "auto_analyze": False,
        "masterchef_enabled": False, "include_text_fusion": False,
        "partition": "FL2VA", "num_frames": 73, "short_edge": 768,
        "seed": 42,
    }
    config = {**defaults, **{k: payload[k] for k in defaults if k in payload}}
    # 5% warmup matches the proven lora recipe (150 steps for a 3000-step run); scales with step count.
    config["warmup_steps"] = int(payload["warmup_steps"]) if "warmup_steps" in payload else max(1, round(int(config["steps"]) * 0.05))
    config.update({
        "dataset": dataset, "trigger_word": trigger, "network_type": network,
        "model_id": model_id, "model_label": model_entry["label"], "sample_prompts": prompts,
    })
    (directory / "config.json").write_text(json.dumps(config, indent=2))

    rank = int(config["rank"])
    if model_entry["arch"] == "ideogram4":
        # Different framework entirely (diffusion-pipe + DeepSpeed, not our own training
        # loop), so it gets its own small flag set instead of the ~30 Krea2-specific ones
        # below. --output_dir/--run_name follow the same convention as the Krea2 branch
        # (run_dir = output_dir / run_name = directory/"run") so job_metrics()/
        # job_checkpoints() work against this job unmodified.
        cmd = [
            sys.executable, str(trainer_script),
            "--pretrained_model_name_or_path", str(config["model_path"]),
            "--dataset_dir", str(dataset_path),
            "--output_dir", str(directory),
            "--run_name", "run",
            "--trigger_word", trigger,
            "--resolution", str(config["resolution"]),
            "--max_train_steps", str(config["steps"]),
            "--save_every_n_steps", str(config["save_every"]),
            "--rank", str(rank),
            "--learning_rate", str(config["learning_rate"]),
            "--seed", str(config["seed"]),
        ]
    elif model_entry["arch"] == "minimax_h3":
        # A third, distinct framework: our own direct diffusers+PEFT loop (like Krea2 below), but
        # against MiniMax-H3's own component layout -- --partition selects which of the two separate
        # ~33B checkpoints (FL2VA vs Ref2VA) this LoRA targets, see train_minimax_h3.py's own
        # docstring for why those aren't interchangeable. --output_dir/--run_name follow the same
        # run_dir = output_dir/run_name convention as the other two branches.
        cmd = [
            sys.executable, str(trainer_script),
            "--pretrained_model_name_or_path", str(config["model_path"]),
            "--partition", str(config["partition"]),
            "--dataset_dir", str(dataset_path),
            "--output_dir", str(directory),
            "--run_name", "run",
            "--trigger_word", trigger,
            "--num_frames", str(config["num_frames"]),
            "--short_edge", str(config["short_edge"]),
            "--max_train_steps", str(config["steps"]),
            "--save_every_n_steps", str(config["save_every"]),
            "--rank", str(rank), "--lora_alpha", str(rank),
            "--learning_rate", str(config["learning_rate"]),
            "--weight_decay", str(config["weight_decay"]),
            "--lr_warmup_steps", str(config["warmup_steps"]),
            "--train_batch_size", str(config["batch_size"]),
            "--gradient_checkpointing", str(config["gradient_checkpointing"]),
            "--seed", str(config["seed"]),
        ]
    else:
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
            "--sample_inference_model", str(config["sample_inference_model"]),
            "--sample_guidance_scale", str(config["sample_guidance_scale"]),
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
            "--include_text_fusion", "1" if config["include_text_fusion"] else "0",
            "--optimizer", str(config["optimizer"]),
            "--gradient_checkpointing", str(config["gradient_checkpointing"]),
            "--transformer_group_offload", str(config["transformer_group_offload"]),
            "--group_offload_blocks", str(config["group_offload_blocks"]),
            "--validation_image", safe_name(str(config["validation_image"])) if str(config["validation_image"]).strip() else "",
            "--validation_prompt", str(config["validation_prompt"]),
            "--seed", str(config["seed"]),
            "--enable_wandb", "0",
            "--masterchef_enabled", "1" if config["masterchef_enabled"] else "0",
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
    set_status(job_id, "running", pid=proc.pid)
    return {"id": job_id, "status": "running"}


@app.post("/api/jobs/{job_id}/stop", dependencies=[Depends(require_auth)])
def stop_job(job_id: str) -> dict:
    proc = _active.get(job_id)
    pid = proc.pid if proc else None
    if pid is None:
        # Not tracked this session (e.g. the server restarted since this job started) -- fall back
        # to the pid persisted in status.json rather than assuming a "running"-looking job that
        # isn't in `_active` must already be dead.
        try:
            pid = json.loads((job_dir(job_id) / "status.json").read_text()).get("pid")
        except Exception:
            pid = None
    if pid is None or not _pid_alive(pid):
        raise HTTPException(status_code=400, detail="Job is not running.")
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    for _ in range(50):
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    if _pid_alive(pid):
        os.killpg(os.getpgid(pid), signal.SIGKILL)
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
    directory = job_dir(job_id)
    metrics_file = directory / "run" / "metrics.jsonl"
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

    status = job_status(job_id)
    config_file = directory / "config.json"
    total_steps = None
    dataset_name = None
    started_at = None
    if config_file.exists():
        try:
            job_config = json.loads(config_file.read_text())
            total_steps = job_config.get("steps")
            dataset_name = job_config.get("dataset")
        except Exception:
            pass
        started_at = config_file.stat().st_mtime
    elapsed_seconds = None
    if started_at:
        # Freeze the clock at the last metrics write once the run isn't active anymore,
        # so reopening a finished job later doesn't show elapsed time still climbing.
        end_reference = time.time() if status == "running" else (
            metrics_file.stat().st_mtime if metrics_file.exists() else started_at
        )
        elapsed_seconds = max(0.0, end_reference - started_at)
    return {
        "status": status, "points": points,
        "total_steps": total_steps, "elapsed_seconds": elapsed_seconds,
        "dataset": dataset_name,
    }


@app.get("/api/jobs/{job_id}/masterchef", dependencies=[Depends(require_auth)])
def job_masterchef(job_id: str) -> dict:
    status_file = job_dir(job_id) / "run" / "masterchef_status.json"
    if not status_file.exists():
        return {"enabled": False, "images": []}
    try:
        return json.loads(status_file.read_text(encoding="utf-8"))
    except Exception:
        return {"enabled": False, "images": []}


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
