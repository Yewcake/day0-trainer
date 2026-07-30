"""Ideogram 4 LoRA trainer via diffusion-pipe + DeepSpeed.

This is NOT the same architecture as train_krea2_lora_direct.py -- diffusion-pipe
is a separate, third-party training framework (its own repo, venv, config format,
training loop). This script is a thin orchestration wrapper: it sets up
diffusion-pipe once (idempotent), stages+validates the dataset, writes the TOML
config diffusion-pipe wants, launches it, and translates its output into the
same run/metrics.jsonl + run/checkpoints/step-NNNNNN/<file>.safetensors shape
the rest of Day0 Trainer already expects (see app/main.py's job_metrics()/
job_checkpoints()) -- so none of the polling UI needed to change to support
this second trainer.

Ported from IDEOGRAM/Train_Ideogram4_DiffusionPipe_FIXED.sh (a working, manually
tested bash script) -- same setup steps and source patches, orchestrated from
Python and re-targeted at Day0's existing dataset/job-directory conventions
instead of a standalone zip-drop-in-workspace workflow.

Two things diffusion-pipe itself doesn't do at all, confirmed by reading its
source rather than assumed: it never prints loss to stdout (only to TensorBoard/
WandB), and it never generates sample images during a normal training run (only
via a separate, one-off --test_sample flag). So this wrapper adds one small
source patch for a greppable stdout metric line, and deliberately does not
attempt training-time sample generation -- matches what the reference script
itself already does (eval_every_n_steps = 0). No Masterchef Cooking hooks either
-- that's wired into train_krea2_lora_direct.py's own loop; diffusion-pipe runs
a loop we don't control the internals of, aside from these two source patches.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
REPO_DIR = WORKDIR / "diffusion-pipe"
MODEL_DIR = REPO_DIR / "models" / "ideogram4"
DIFFUSION_PIPE_URL = "https://github.com/tdrussell/diffusion-pipe"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

METRIC_PREFIX = "DAY0_METRIC"
METRIC_RE = re.compile(r"DAY0_METRIC step=(?P<step>\d+) loss=(?P<loss>[-\d.eE+]+) epoch=(?P<epoch>\d+)")
CHECKPOINT_STEP_RE = re.compile(r"step(\d+)")


def say(msg: str) -> None:
    print(f"[ideogram4-trainer] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[ideogram4-trainer] WARNING: {msg}", flush=True)


def run(cmd: list[str], **kwargs) -> None:
    say("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="run")
    parser.add_argument("--trigger_word", default="")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_train_steps", type=int, default=3000)
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--learning_rate", default="5e-5")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# --------------------------------------------------------------------------
# One-time (idempotent) diffusion-pipe setup
# --------------------------------------------------------------------------
def setup_diffusion_pipe() -> None:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    if not (REPO_DIR / ".git").is_dir():
        say("Cloning diffusion-pipe...")
        run(["git", "clone", "--filter=blob:none", "--recurse-submodules", DIFFUSION_PIPE_URL, str(REPO_DIR)])
    else:
        say(f"Using existing diffusion-pipe at {REPO_DIR}")
        if os.environ.get("IDEOGRAM4_UPDATE_REPO") == "1":
            say("IDEOGRAM4_UPDATE_REPO=1 set; updating diffusion-pipe checkout...")
            try:
                run(["git", "pull", "--ff-only"], cwd=str(REPO_DIR))
            except subprocess.CalledProcessError:
                warn("Could not fast-forward diffusion-pipe; continuing with existing checkout.")
        else:
            warn("Not updating existing diffusion-pipe checkout. Set IDEOGRAM4_UPDATE_REPO=1 for latest upstream.")
        try:
            run(["git", "submodule", "update", "--init", "--recursive"], cwd=str(REPO_DIR))
        except subprocess.CalledProcessError:
            warn("Submodule update had issues; continuing.")

    patch_ideogram4_dtype_callsite()
    patch_ideogram4_metric_print()

    venv_dir = REPO_DIR / "venv"
    if not venv_dir.is_dir():
        say("Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(venv_dir)])
    pip = str(venv_dir / "bin" / "pip")

    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    env.setdefault("MAX_JOBS", "2")
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "2")

    say("Installing dependencies...")
    run([pip, "install", "-U", "pip", "wheel", "packaging", "setuptools"], env=env)
    run(
        [pip, "install", "torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1",
         "--index-url", "https://download.pytorch.org/whl/cu128"],
        env=env,
    )
    run([pip, "install", "-U", "wandb", "hf_transfer", "huggingface_hub", "bitsandbytes"], env=env)
    run([pip, "install", "-r", "requirements.txt"], cwd=str(REPO_DIR), env=env)

    if os.environ.get("INSTALL_FLASH_ATTN") == "1":
        say("Installing flash-attn because INSTALL_FLASH_ATTN=1...")
        try:
            run([pip, "install", "flash-attn", "--no-build-isolation"], env=env)
        except subprocess.CalledProcessError:
            warn("flash-attn install failed; continuing.")
    else:
        warn("Skipping flash-attn install by default; set INSTALL_FLASH_ATTN=1 to compile/install it.")

    for sub in ("diffusion_models", "text_encoders", "vae"):
        (MODEL_DIR / sub).mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    downloads = [
        ("diffusion_models/ideogram4_fp8_scaled.safetensors", "Ideogram4 diffusion model"),
        ("text_encoders/qwen3vl_8b_fp8_scaled.safetensors", "Qwen3-VL text encoder"),
        ("vae/flux2-vae.safetensors", "Flux2 VAE"),
    ]
    for rel_path, label in downloads:
        dest = MODEL_DIR / rel_path
        if dest.is_file():
            continue
        say(f"Downloading {label}...")
        cmd = ["hf", "download", "Comfy-Org/Ideogram-4", rel_path, "--local-dir", str(MODEL_DIR)]
        if hf_token:
            cmd += ["--token", hf_token]
        run(cmd)


def patch_ideogram4_dtype_callsite() -> None:
    model_file = REPO_DIR / "models" / "ideogram4.py"
    if not model_file.is_file():
        raise RuntimeError(f"Cannot apply dtype patch; missing {model_file}")
    text = model_file.read_text()
    old = "t_cond = self.t_embedding(t)"
    new = "t_cond = self.t_embedding(t, dtype=t.dtype)"
    if new in text:
        say("dtype callsite patch already present.")
        return
    if old not in text:
        # The reference bash script only warns and continues here, which risks
        # silently training with an unpatched (possibly buggy) dtype path if
        # upstream ever changes this line. Fail loudly instead.
        raise RuntimeError(
            f"Ideogram4 dtype callsite patch target not found in {model_file} "
            "(upstream diffusion-pipe source may have changed). Refusing to train "
            "with an unverified dtype path -- update this patch before retrying."
        )
    backup = model_file.with_suffix(model_file.suffix + ".bak_before_dtype_callsite_patch")
    if not backup.exists():
        backup.write_text(text)
    model_file.write_text(text.replace(old, new, 1))
    say(f"Applied Ideogram4 dtype callsite patch: {old} -> {new}")


def patch_ideogram4_metric_print() -> None:
    """diffusion-pipe only logs loss to TensorBoard/WandB, never stdout -- insert
    one greppable print line so this wrapper can tail stdout for step/loss/epoch
    instead of parsing TensorBoard's binary event files mid-run."""
    train_file = REPO_DIR / "train.py"
    if not train_file.is_file():
        raise RuntimeError(f"Cannot apply metric print patch; missing {train_file}")
    text = train_file.read_text()
    marker = f'print(f"{METRIC_PREFIX} step={{x_axis}} loss={{loss}} epoch={{epoch}}", flush=True)'
    if marker in text:
        say("metric print patch already present.")
        return
    old = "tb_writer.add_scalar(f'train/loss', loss, x_axis)"
    if old not in text:
        raise RuntimeError(
            f"Ideogram4 metric print patch target not found in {train_file} "
            "(upstream diffusion-pipe source may have changed). Refusing to train "
            "without a way to read progress -- update this patch before retrying."
        )
    backup = train_file.with_suffix(train_file.suffix + ".bak_before_metric_print_patch")
    if not backup.exists():
        backup.write_text(text)
    train_file.write_text(text.replace(old, old + "\n            " + marker, 1))
    say("Applied Ideogram4 metric print patch.")


# --------------------------------------------------------------------------
# Dataset staging (never mutates the shared, reusable Day0 dataset directory)
# --------------------------------------------------------------------------
def wrap_plain_caption(text: str, trigger: str) -> dict:
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


def stage_and_validate_captions(source_dir: Path, staged_dir: Path, trigger: str) -> None:
    """Copies the dataset into a job-local staging directory and ensures every
    caption is in the JSON shape diffusion-pipe's Ideogram4 config expects --
    never touches the original, shared dataset directory. Plain-text captions
    (or ones written before Day0's "Caption format" option existed) get
    auto-wrapped here, so that Datasets-page choice is a convenience, not a
    strict requirement -- mirrors the reference bash script's own validation
    pass, which does the same unconditional wrap-if-not-already-JSON."""
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True)

    images = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise RuntimeError(f"No images found in {source_dir}.")

    missing: list[str] = []
    for image in images:
        shutil.copy2(image, staged_dir / image.name)
        caption_file = image.with_suffix(".txt")
        dest_caption = staged_dir / caption_file.name
        if not caption_file.is_file():
            missing.append(image.name)
            continue
        raw = caption_file.read_text(encoding="utf-8", errors="replace").strip()
        try:
            json.loads(raw)
            dest_caption.write_text(raw, encoding="utf-8")  # already JSON, leave as-is
        except json.JSONDecodeError:
            payload = wrap_plain_caption(raw, trigger)
            dest_caption.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    if missing:
        raise RuntimeError(f"Missing captions for {len(missing)} image(s): {', '.join(missing[:10])}")

    say(f"Staged {len(images)} image(s) with validated JSON captions at {staged_dir}.")


# --------------------------------------------------------------------------
# Config generation
# --------------------------------------------------------------------------
def prepare_dataset_toml(staged_dir: Path, resolution: int) -> Path:
    toml_path = staged_dir / "dataset.toml"
    toml_path.write_text(
        f"resolutions = [{resolution}]\n"
        "enable_ar_bucket = true\n"
        "min_ar = 0.5\n"
        "max_ar = 2.0\n"
        "num_ar_buckets = 9\n"
        "bucket_resolution_steps = 64\n\n"
        "[[directory]]\n"
        f'path = "{staged_dir}"\n'
        "num_repeats = 1\n"
    )
    return toml_path


def write_training_config(args: argparse.Namespace, dataset_toml: Path, dp_out_dir: Path) -> Path:
    config_path = REPO_DIR / f"config_{args.run_name}.toml"
    config_path.write_text(
        f'output_dir = "{dp_out_dir}"\n'
        f'output_name = "{args.run_name}"\n'
        f'dataset = "{dataset_toml}"\n\n'
        "epochs = 999\n"
        f"max_steps = {args.max_train_steps}\n"
        "micro_batch_size_per_gpu = 1\n"
        "gradient_accumulation_steps = 1\n"
        "pipeline_stages = 1\n"
        "gradient_clipping = 1.0\n"
        'lr_scheduler = "cosine"\n'
        "warmup_steps = 100\n"
        f"save_every_n_steps = {args.save_every_n_steps}\n"
        "checkpoint_every_n_minutes = 120\n"
        "activation_checkpointing = true\n"
        'save_dtype = "bfloat16"\n'
        "caching_batch_size = 1\n"
        "logging_steps = 1\n"
        "eval_before_first_step = false\n"
        "eval_every_n_steps = 0\n"
        "compile = false\n\n"
        "[model]\n"
        'type = "ideogram4"\n'
        f'diffusion_model = "{MODEL_DIR}/diffusion_models/ideogram4_fp8_scaled.safetensors"\n'
        f'vae = "{MODEL_DIR}/vae/flux2-vae.safetensors"\n'
        "text_encoders = [\n"
        f'  {{path = "{MODEL_DIR}/text_encoders/qwen3vl_8b_fp8_scaled.safetensors", type = "ideogram4"}}\n'
        "]\n"
        'dtype = "bfloat16"\n'
        'diffusion_model_dtype = "float8"\n'
        'timestep_sample_method = "logit_normal"\n'
        "shift = 3\n\n"
        "[adapter]\n"
        'type = "lora"\n'
        f"rank = {args.rank}\n"
        'dtype = "bfloat16"\n\n'
        "[optimizer]\n"
        'type = "adamw8bit"\n'
        f"lr = {args.learning_rate}\n"
        "betas = [0.9, 0.99]\n"
        "weight_decay = 0.01\n"
        "eps = 1e-8\n\n"
        "[monitoring]\n"
        "enable_wandb = false\n"
    )
    return config_path


# --------------------------------------------------------------------------
# Checkpoint discovery/copy -- reuses the reference script's own proven approach
# (latest timestamped run dir -> find adapter_model.safetensors), just called
# periodically during training instead of once at the end.
# --------------------------------------------------------------------------
def copy_new_checkpoints(dp_out_dir: Path, run_dir: Path, copied: set[str]) -> None:
    if not dp_out_dir.is_dir():
        return
    run_subdirs = sorted(
        (d for d in dp_out_dir.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True
    )
    if not run_subdirs:
        return
    latest_run = run_subdirs[0]
    for adapter_file in sorted(latest_run.glob("*/adapter_model.safetensors")):
        step_dir_name = adapter_file.parent.name
        if step_dir_name in copied:
            continue
        match = CHECKPOINT_STEP_RE.search(step_dir_name)
        step_num = int(match.group(1)) if match else 0
        dest_dir = run_dir / "checkpoints" / f"step-{step_num:06d}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapter_file, dest_dir / "krea2_comfy_native_lora.safetensors")
        copied.add(step_dir_name)
        say(f"Copied checkpoint: {step_dir_name} -> {dest_dir}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    out_root = Path(args.output_dir)
    run_dir = out_root / args.run_name  # matches job_metrics()/job_checkpoints()'s job_dir(id)/"run" expectation
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    dp_out_dir = out_root / "diffusion_pipe_raw"  # diffusion-pipe's own raw timestamped output, kept separate

    say("Day0 Ideogram4 trainer (diffusion-pipe wrapper) starting.")
    setup_diffusion_pipe()

    staged_dir = out_root / "dataset_staged"
    stage_and_validate_captions(Path(args.dataset_dir), staged_dir, args.trigger_word)
    if args.trigger_word:
        any_mentions = any(
            args.trigger_word in p.read_text(encoding="utf-8", errors="replace")
            for p in staged_dir.glob("*.txt")
        )
        if not any_mentions:
            warn(f"Trigger word '{args.trigger_word}' doesn't appear in any staged caption -- check your dataset.")

    dataset_toml = prepare_dataset_toml(staged_dir, args.resolution)
    config_path = write_training_config(args, dataset_toml, dp_out_dir)
    say(f"Wrote config: {config_path}")

    venv_dir = REPO_DIR / "venv"
    deepspeed_bin = str(venv_dir / "bin" / "deepspeed")
    cmd = [deepspeed_bin, "--num_gpus=1", "train.py", "--deepspeed", "--config", str(config_path)]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_DIR}:{REPO_DIR / 'submodules'}"
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    say("Starting Ideogram4 LoRA training (deepspeed)...")
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_DIR), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    copied_checkpoints: set[str] = set()
    with open(metrics_path, "a", encoding="utf-8") as metrics_file:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            print(line, flush=True)  # this process's own stdout is already redirected to the job's log.txt by main.py
            match = METRIC_RE.search(line)
            if match:
                row = {
                    "step": int(match.group("step")),
                    "loss": round(float(match.group("loss")), 6),
                    # diffusion-pipe's live per-step LR isn't exposed anywhere we can read --
                    # this is the configured base rate, not a live cosine-decayed value.
                    "lr": float(args.learning_rate),
                    "epoch": int(match.group("epoch")),
                }
                metrics_file.write(json.dumps(row) + "\n")
                metrics_file.flush()
            copy_new_checkpoints(dp_out_dir, run_dir, copied_checkpoints)

    return_code = proc.wait()
    copy_new_checkpoints(dp_out_dir, run_dir, copied_checkpoints)

    if return_code != 0:
        say(f"deepspeed exited with code {return_code}.")
        sys.exit(return_code)
    say("Training complete.")


if __name__ == "__main__":
    main()
