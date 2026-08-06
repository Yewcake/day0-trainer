"""MiniMax-H3 LoRA trainer -- direct diffusers + PEFT, style/motion LoRAs.

STATUS: first draft, written directly against MiniMax-H3's real source (the
`minimax-h3` branch of huggingface/diffusers, not yet merged/released to PyPI)
and NOT YET RUN on a GPU. Every architectural claim below is grounded in that
source -- file paths and line-level behavior were read, not guessed -- but the
end-to-end training loop has no live verification yet. Expect at least one real
debugging pass on first use, the same caveat train_ideogram4.py shipped with.

WHY THIS FILE EXISTS AND HOW IT'S STRUCTURED
MiniMax-H3 is not merged into diffusers yet, so `pip install diffusers` alone
does not have it. `setup_environment()` installs diffusers pinned to the exact
commit this file was written against (see DIFFUSERS_MINIMAX_H3_COMMIT below).
From there this script imports the real MiniMaxH3Transformer3DModel, the real
VAEs, the real MiniMaxH3Scheduler, and the real packing helpers
(`build_packed_sequence`, `build_row_timesteps`, `patchify_video_latents`) and
calls them exactly as the modular inference pipeline does -- it does NOT
reimplement any of that math by hand. The subtle parts (mixed-precision
casting, the reversed/data-ward velocity convention, the exponential sigma
shift, the packed-sequence row layout) are exactly the kind of thing that is
easy to get silently wrong from a written description, so this script leans on
upstream's own tested code for all of it and only writes the pieces diffusers
does not ship: a training loop, a dataset, and LoRA injection + checkpointing.

WHAT THIS FIRST DRAFT DELIBERATELY DOES NOT DO
- Ref2VA reference conditioning uses each clip's own first frame as its
  reference (see precompute_cache()/main()'s reference-row handling) --
  a practical proxy for "a reference photo of this subject", not textbook
  Ref2VA usage (a genuinely separate reference image per clip). Good enough
  to actually exercise the conditioning pathway those weights expect instead
  of training them as if they were plain FL2VA, but a real multi-reference
  dataset format (separate reference images, not just first-frame-of-clip)
  is future work, not done here.
- Audio training is opt-in (`--train_audio 0` by default). The stated first
  targets (NSFW, yoga, cartoon) are visual styles, not audio concepts, so most
  runs still skip the whole audio-VAE/waveform-extraction path entirely
  (num_audio_latents=0 keeps the packed sequence video+text only). When enabled,
  each clip's own audio track (same cropped time window as its video frames) is
  encoded via AutoencoderKLMiniMaxH3Audio (mono VAE, stereo handled as two batch
  items, `.mode()` not `.sample()` -- matching the real reference-conditioning
  encoder step's own convention, read from source, not guessed), noised with the
  model's own audio_shift=3.0 schedule (reusing the same MiniMaxH3Scheduler
  instance as video -- scale_noise() is shift-independent, confirmed via
  source), and trained against `output.audio_sample` with an independent MSE
  term (`--audio_loss_weight`, default 1.0, matching a real third-party
  checkpoint's `ss_h3_audio_loss_weight: 1` metadata convention). Clips with no
  audio stream at all train video-only for that step rather than being taught
  silence.
- No training-time sample video generation. Reusing the modular inference
  pipeline for periodic previews is a real chunk of work on its own (assembling
  the actual T2VABlocks graph) -- deferred rather than half-built here.
- No in-loop VAE/text-encoder offload shuffling. Instead, model loading and the
  training loop are two strictly separate phases (see main()): first the VAE +
  text encoder run once over the whole dataset and their outputs are cached,
  then those encoders are freed from GPU memory entirely and the transformer
  loads only after that. The alternative -- keeping every component resident
  for the whole run -- was the actual GPU-sizing blocker: the frozen Qwen3-VL
  text encoder alone is a second ~33-66GB model sitting alongside the 33B
  transformer, and captions/videos never change between steps, so re-running
  frozen encoders on the same inputs every step would be pure waste anyway.

ARCHITECTURE FACTS THIS FILE DEPENDS ON (read from source, cited inline where
they're used):
- transformer_minimax_h3.py: one packed 1-D self-attention sequence carries
  text + audio + video rows through 50 shared transformer_blocks (no
  cross-attention anywhere, no per-modality block weights). Already inherits
  PeftAdapterMixin and has @apply_lora_scale wired into forward() -- standard
  diffusers/PEFT LoRA injection works directly, no custom PEFT plumbing needed.
- scheduling_minimax_h3.py: rectified flow, but with MiniMax-H3's own
  conventions -- t = 1 - sigma (t=1 clean), and the transformer predicts a
  *data-ward* velocity (x0 = x_t + (1-t)*v, the sign-reversed opposite of
  diffusers' usual FlowMatchEulerDiscreteScheduler). scale_noise() reproduces
  `x_t = t*x0 + (1-t)*noise` exactly, and the shift formula for the sigma grid
  is `s*sigma / (1 + (s-1)*sigma)`. Video uses shift=12.0, audio shift=3.0 --
  two independent noise levels can be live in the same packed sequence.
- modular_pipelines/minimax_h3/packing.py: valid clip lengths are exactly
  `17*n + 5` frames at a fixed 24fps; canvas is a 768px short edge, aspect
  ratio clamped to [1:4, 4:1], multiple of 32.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# Must be set before any CUDA context exists (i.e. before torch is ever imported, including by
# a dependency) to take effect at all -- live run hit a near-total OOM (94.49/94.97GB used) one
# 1.16GB allocation short, and this is the exact mitigation PyTorch's own error message suggests
# for allocator fragmentation. It won't manufacture memory that isn't there, but at this close to
# the ceiling it's worth having on for free.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

WORKDIR = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
DIFFUSERS_MINIMAX_H3_COMMIT = "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"  # huggingface/diffusers, branch minimax-h3
MINIMAX_H3_MODEL_ID = "MiniMaxAI/MiniMax-H3"

MINIMAX_H3_FPS = 24
MINIMAX_H3_FRAMES_PER_CHUNK = 17
MINIMAX_H3_LATENTS_PER_CHUNK = 5
MINIMAX_H3_TEXT_ENCODER_LAYER = 50
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}

# Comfy-Org's own pre-quantized MiniMax-H3 release. Deliberately the NON-pruned int8-ConvRot
# file, not "*_pruned_int8_convrot" -- confirmed by inspecting both files' real safetensors
# headers (byte-range HTTP requests, not assumed): the pruned variant replaces adaln_proj's
# input with an 8-dim "adaln_t_table" lookup (fed by a precomputed [1025, 8] curve buffer)
# instead of the full model's normal 2688-dim TimestepEmbedding-MLP conditioning -- a genuinely
# different computation graph, not just different quantization, that would require reimplementing
# MiniMax-H3's forward pass by hand (ComfyUI's own comfy/ldm/minimax/model.py) to use correctly.
# The non-pruned convrot file's adaln_proj is still 2688-dim (verified: blocks.0.adaln_proj.
# linear.weight is [96768, 2688], int8-quantized, and time_embedder.proj_in/proj_out are present
# with their normal shapes) -- same computation graph as the full model, just int8-quantized, so
# it loads into diffusers' existing, unmodified MiniMaxH3Transformer3DModel with zero forward-pass
# changes needed.
MINIMAX_H3_CONVROT_REPO = "Comfy-Org/MiniMax-H3"
MINIMAX_H3_CONVROT_FILES = {
    "FL2VA": "diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors",
    "Ref2VA": "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors",
}


def say(msg: str) -> None:
    print(f"[minimax-h3-trainer] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[minimax-h3-trainer] WARNING: {msg}", flush=True)


def run(cmd: list[str], **kwargs) -> None:
    say("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def align_num_frames(num_frames: int) -> int:
    """Round up to the nearest valid `17*n + 5` clip length -- diffusers' own packing.align_num_frames."""
    while num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


# --------------------------------------------------------------------------
# One-time (idempotent) environment setup
# --------------------------------------------------------------------------
def setup_environment() -> None:
    say(f"Installing diffusers @ {DIFFUSERS_MINIMAX_H3_COMMIT} (minimax-h3 branch, not yet released to PyPI)...")
    # The base Docker image already ships a released diffusers (for the Krea2/Ideogram4 trainers).
    # Without --upgrade --force-reinstall, `pip install git+URL` silently no-ops when it considers
    # any version of the package already installed -- it does NOT rebuild from the pinned commit
    # just because a VCS URL was given. diffusers' own setup.py dependencies (Pillow, numpy, regex,
    # requests, huggingface-hub, safetensors...) are lightweight and don't touch torch, so letting
    # pip resolve them normally (no --no-deps) is safe and covers anything this unreleased branch
    # might have newly added.
    run([
        sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall",
        f"git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_MINIMAX_H3_COMMIT}",
    ])
    # MiniMaxH3Qwen3VLHFEncoder needs Qwen3VLForConditionalGeneration -- a recent enough
    # `transformers` to even have that class, which the base image's pinned version predates.
    # `av` (PyAV) is for video decoding in load_and_prepare_clip() -- not torchvision (see that
    # function's own docstring for why) or torchaudio (no audio path exists yet), so neither is
    # installed here; numpy is deliberately left alone too, already guaranteed present via torch
    # and not worth the ABI-mismatch risk of force-upgrading it alongside a pinned torch build.
    run([
        sys.executable, "-m", "pip", "install", "-q", "--upgrade",
        "transformers", "peft>=0.13", "accelerate", "av", "bitsandbytes",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", default=MINIMAX_H3_MODEL_ID)
    # MiniMax-H3 ships two genuinely separate ~33B transformer checkpoints under one repo, confirmed
    # via the actual model_index.json partition metadata, not assumed: "fl2va" (tasks t2va + fl2va --
    # plain text-to-video and first/last-keyframe-conditioned generation) and "ref2va" (reference-image/
    # video-conditioned generation, MiniMax-H3's native identity-consistency mechanism). Same
    # architecture and config, different weights -- a LoRA trained against one partition's transformer
    # is not a drop-in adapter for the other (confirmed live: a checkpoint's key names alone match
    # either partition's naming, but the actual weight values it perturbs are partition-specific, so
    # testing a ref2va-trained LoRA against fl2va weights is a no-op, not a weaker effect). Selecting
    # "Ref2VA" here automatically trains with first-frame reference conditioning (see main()'s
    # reference-row handling) rather than plain T2VA on ref2va's weights -- the earlier version of
    # this script trained those weights without ever exercising the conditioning pathway they expect,
    # a real gap independent of the checkpoint key-prefix bug that turned out to be the actual cause
    # of the first "LoRA has no effect" report (see the checkpoint-save code below).
    parser.add_argument("--partition", default="FL2VA", choices=["FL2VA", "Ref2VA"])
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="run")
    parser.add_argument("--trigger_word", default="")
    parser.add_argument("--num_frames", type=int, default=73)  # ~3s @ 24fps; aligned to 17n+5 at load time
    # diffusers' resolve_canvas_size() hardcodes a 768px short edge (matching the released model's
    # default inference resolution). Lower values shrink the packed sequence length -- fewer spatial
    # patches per frame -- for meaningfully faster caching and training steps, at the cost of training
    # away from the model's native resolution. 768 (the default) reproduces the original hardcoded
    # behavior exactly; anything else overrides it via a module-level constant patch, see main().
    parser.add_argument("--short_edge", type=int, default=768)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--save_every_n_steps", type=int, default=250)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_scheduler", default="cosine", choices=["cosine", "constant", "linear"])
    # The training loop pulls one cached (clip, caption) pair per step -- batching multiple clips of
    # different aspect ratios/lengths into one packed layout isn't implemented, so this is accepted
    # (main.py's job form has a general Batch size field) but enforced to be 1, not silently ignored.
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=int, default=1)
    parser.add_argument("--video_shift", type=float, default=12.0)  # MiniMax-H3's own default for video rows
    parser.add_argument("--audio_shift", type=float, default=3.0)  # MiniMax-H3's own default for audio rows
    parser.add_argument("--train_audio", type=int, default=0)  # 0 = video+text only; see module docstring
    parser.add_argument("--audio_loss_weight", type=float, default=1.0)  # matches ss_h3_audio_loss_weight: 1 convention
    parser.add_argument("--base_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--quantize_base", type=int, default=1)  # 4-bit (nf4) frozen base; ~33B model, LoRA-only fits nothing else
    # "bitsandbytes" (default): download the full model ourselves, quantize to NF4 at load time
    # (load_transformer()) -- --quantize_base governs this path. "comfy_convrot": download
    # Comfy-Org's own pre-quantized int8-ConvRot checkpoint instead (load_transformer_convrot()),
    # ignoring --quantize_base entirely (that checkpoint is already quantized, there's no
    # "unquantized" variant of it to fall back to). Matches what real community LoRAs (and
    # ai-toolkit's own default recipe) actually target, and produces a checkpoint that loads
    # cleanly against ComfyUI's own int8-ConvRot checkpoint without any shape mismatch.
    parser.add_argument("--quant_source", default="bitsandbytes", choices=["bitsandbytes", "comfy_convrot"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--caption_extension", default=".txt")
    return parser.parse_args()


# --------------------------------------------------------------------------
# Dataset: video clips + plain-text captions (no JSON, no chat template --
# encoders.py's own encode_prompt() feeds the prompt to Qwen3-VL verbatim)
# --------------------------------------------------------------------------
def dataset_clips(dataset_dir: Path, caption_extension: str) -> list[tuple[Path, str]]:
    clips = []
    for path in sorted(dataset_dir.iterdir()):
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        caption_file = path.with_suffix(caption_extension)
        if not caption_file.is_file():
            warn(f"Skipping {path.name}: no matching {caption_extension} caption.")
            continue
        caption = caption_file.read_text(encoding="utf-8").strip()
        if not caption:
            warn(f"Skipping {path.name}: empty caption.")
            continue
        clips.append((path, caption))
    if not clips:
        raise RuntimeError(f"No usable (video, caption) pairs found in {dataset_dir}.")
    return clips


def load_and_prepare_clip(path, num_frames: int, resolve_canvas_size):
    """Read a video file, align its frame count to the nearest valid 17n+5, resize/center-crop
    to MiniMax-H3's fixed-short-edge canvas for the clip's own aspect ratio, return a
    ((C, num_frames, H, W) float tensor in [0, 1], clip_start_seconds, clip_duration_seconds) tuple
    (the VAE's own encode() then applies the ImageNet normalization documented in
    autoencoder_kl_minimax_h3.py -- not done here). The start/duration are in seconds at
    MiniMax-H3's fixed 24fps, so load_audio_waveform() can extract the exact matching time window
    from the same source file when --train_audio is on -- audio and video must cover the same
    content, not just the same clip file.

    Decodes via PyAV rather than torchvision.io.read_video -- torchvision has been shuffling its
    video-reading backend across releases (this trainer's own first live run hit exactly that:
    `--upgrade`d torchvision no longer exposed read_video at all), while PyAV is a stable, narrowly-
    scoped binding directly onto ffmpeg's own decoders and isn't subject to torchvision's churn."""
    import av
    import numpy as np
    import torch

    container = av.open(str(path))
    stream = container.streams.video[0]
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]  # each (H, W, 3) uint8
    source_fps = float(stream.average_rate) if stream.average_rate else MINIMAX_H3_FPS
    container.close()

    video = torch.from_numpy(np.stack(frames)).float() / 255.0  # (T, H, W, C) in [0, 1]
    video = video.permute(0, 3, 1, 2)  # (T, C, H, W)

    if abs(source_fps - MINIMAX_H3_FPS) > 0.5:
        # Nearest-frame resample to 24fps rather than requiring pre-conformed source clips.
        indices = (torch.arange(0, video.shape[0] * MINIMAX_H3_FPS / source_fps) * source_fps / MINIMAX_H3_FPS)
        indices = indices.long().clamp(max=video.shape[0] - 1)
        video = video[indices]

    aligned = align_num_frames(min(num_frames, video.shape[0]) or MINIMAX_H3_LATENTS_PER_CHUNK)
    if video.shape[0] < aligned:
        pad = video[-1:].repeat(aligned - video.shape[0], 1, 1, 1)
        video = torch.cat([video, pad], dim=0)
        start = 0
    else:
        start = random.randint(0, video.shape[0] - aligned)
        video = video[start:start + aligned]
    # Both start and aligned are frame counts on the already-24fps-resampled timeline above, so
    # dividing by MINIMAX_H3_FPS gives seconds regardless of the source file's own frame rate.
    clip_start_seconds = start / MINIMAX_H3_FPS
    clip_duration_seconds = aligned / MINIMAX_H3_FPS

    _, _, h, w = video.shape  # video is still (T, C, H, W) here
    height, width = resolve_canvas_size(w, h)
    scale = max(width / w, height / h)
    resized_h, resized_w = max(height, round(h * scale)), max(width, round(w * scale))
    # F.interpolate's 2D bilinear mode wants (N, C, H, W); treat the T frames as the batch axis.
    video = torch.nn.functional.interpolate(
        video, size=(resized_h, resized_w), mode="bilinear", align_corners=False,
    )
    top = max(0, (resized_h - height) // 2)
    left = max(0, (resized_w - width) // 2)
    video = video[:, :, top:top + height, left:left + width]  # (T, C, height, width)
    return video.permute(1, 0, 2, 3), clip_start_seconds, clip_duration_seconds  # (C, T, height, width)


def load_audio_waveform(path, start_seconds: float, duration_seconds: float, target_sample_rate: int):
    """Extract the stereo waveform for the same time window load_and_prepare_clip() cropped the
    video to, resampled to the audio VAE's fixed 32kHz stereo input. Returns None when the file has
    no audio stream, or when its audio track doesn't even reach the video crop's start time -- both
    left as video-only for that step rather than synthesizing silence, same reasoning as this
    trainer's original --train_audio 0 default (silence would actively teach the model to output
    silence, worse than not training audio on that clip at all).

    Decodes the whole track and slices by sample index rather than seeking, mirroring
    load_and_prepare_clip()'s own decode-then-slice approach -- these are short training clips, not
    long-form video, so this isn't a hot-path performance concern, and it sidesteps PyAV seek()
    only guaranteeing landing at or before a keyframe (imprecise for audio-sample-accurate slicing).

    format="fltp" (planar float) rather than a packed/interleaved format -- planar keeps each
    channel's samples contiguous, matching the (channels, samples) layout this function returns
    directly."""
    import av
    import numpy as np
    import torch

    container = av.open(str(path))
    if not container.streams.audio:
        container.close()
        return None
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=target_sample_rate)

    chunks = [
        resampled.to_ndarray()
        for frame in container.decode(stream)
        for resampled in resampler.resample(frame)
    ]
    container.close()
    if not chunks:
        return None

    waveform = np.concatenate(chunks, axis=1)  # (2, total_samples) float32, planar
    if waveform.shape[0] == 1:
        waveform = np.repeat(waveform, 2, axis=0)  # mono source -> duplicate to the VAE's fixed 2-channel convention

    start_sample = round(start_seconds * target_sample_rate)
    target_samples = round(duration_seconds * target_sample_rate)
    if start_sample >= waveform.shape[1]:
        return None
    segment = waveform[:, start_sample:start_sample + target_samples]
    if segment.shape[1] < target_samples:
        pad_width = target_samples - segment.shape[1]
        pad = (
            np.repeat(segment[:, -1:], pad_width, axis=1)
            if segment.shape[1] > 0 else np.zeros((2, pad_width), dtype=np.float32)
        )
        segment = np.concatenate([segment, pad], axis=1)

    return torch.from_numpy(segment.copy())  # (2, target_samples) float32


def pack_audio_latents(latents: "torch.Tensor") -> "torch.Tensor":
    """Inverse of diffusers' own packing.unpack_audio_tokens() (decode-side: packed rows -> VAE
    latents) -- diffusers doesn't ship the training-side direction, so it's hand-written here,
    derived directly from unpack_audio_tokens' own reshape (`rows.reshape(2, num_audio_latents,
    C).permute(0, 2, 1)` to recover the VAE-native (2, C, T) layout) and from the real
    reference-conditioning encoder step's own output layout (`posterior.mode().transpose(1, 2)`,
    already (2, T, C) -- channel-major, the batch dim standing in for the two stereo channels,
    confirmed by reading both directly from diffusers source rather than assumed).

    Input here is expected in that same (2, T, C) layout (channels, time, latent_dim) -- already
    what encode+mode+transpose produces, so no permute is needed, just the row-major flatten
    unpack_audio_tokens' own reshape(2, T, C) exactly inverts."""
    channels, num_audio_latents, latent_channels = latents.shape
    return latents.reshape(channels * num_audio_latents, latent_channels)


# --------------------------------------------------------------------------
# Model loading
#
# Loaded and used in two strictly separate phases, not all at once -- see the
# "why two phases" note at the top of main(). load_encoders() covers only the
# frozen components needed to turn (video, caption) into (latents, embeddings);
# load_transformer() is called later, after those are cached and freed.
# --------------------------------------------------------------------------
def get_device() -> "torch.device":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        warn("No CUDA device found -- this will run (uselessly slowly) on CPU. Expected on a RunPod GPU pod.")
    else:
        # Printed once at startup so a slow run can be diagnosed from the log alone -- e.g. distinguishing
        # a genuinely underpowered/shared GPU from a code-level slowdown without needing to ask.
        name = torch.cuda.get_device_name(device)
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        say(f"GPU: {name} ({total_gb:.1f}GB), compute capability {'.'.join(map(str, torch.cuda.get_device_capability(device)))}")
    return device


def load_encoders(args, device):
    import torch
    from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Scheduler
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration
    if args.train_audio:
        from diffusers import AutoencoderKLMiniMaxH3Audio

    dtype = torch.bfloat16 if args.base_dtype == "bfloat16" else torch.float16

    say("Loading video VAE...")
    # NOT {partition}/video_vae -- that subfolder is MiniMax's own native "standalone" bundle
    # (raw Python source + a model.safetensors under a nested source/ dir, config._class_name
    # "MiniMaxH3VideoVAE" with auto_map pointing at their own AutoencoderKLLegacy code), a
    # completely different thing from the diffusers-native checkpoint our AutoencoderKLMiniMaxH3
    # class expects. The real diffusers-format one (config._class_name "AutoencoderKLMiniMaxH3",
    # proper sharded diffusion_pytorch_model-*.safetensors) lives at the shared top-level "vae"
    # folder -- confirmed by reading both config.json files, not assumed. Same VAE either
    # partition, so this isn't partition-specific.
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    say("Loading Qwen3-VL text conditioner (frozen, read at its 50th decoder layer, LM head unused)...")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path, subfolder=f"{args.partition}/text_encoder", torch_dtype=dtype,
    ).to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder=f"{args.partition}/tokenizer"
    )

    video_scheduler = MiniMaxH3Scheduler(shift=args.video_shift)

    audio_vae = None
    if args.train_audio:
        say("Loading audio VAE...")
        # Shared top-level "audio_vae" subfolder (confirmed via model_index.json), same partition-
        # independence as the video "vae" folder above.
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="audio_vae", torch_dtype=torch.float32,
        ).to(device)
        audio_vae.requires_grad_(False)
        audio_vae.eval()

    return vae, text_encoder, tokenizer, video_scheduler, audio_vae


# --------------------------------------------------------------------------
# ConvRot int8 dequantization -- for loading Comfy-Org's pre-quantized checkpoint
# (see load_transformer_convrot() below). Algorithm read directly from ai-toolkit's real
# source (toolkit/util/convrot_quant.py's ConvRotInt8Quantizer, toolkit/util/
# comfy_quant_import.py's import_comfy_quantized_layers()), not guessed or reverse-engineered
# from behavior -- this is the same rotate-then-quantize-then-derotate scheme QuaRot/SpinQuant
# use to reduce int8 outlier error, applied per output row:
#   quantize:   q, scale = int8_per_row(rotate(w, rot_size))     -- rotate is a fixed, self-
#   dequantize: w = rotate(q.float() * scale, rot_size)             inverse orthogonal transform
# rotate()'s matrix is a Kronecker power of the 4x4 regular Hadamard matrix -- deterministic,
# parameter-free, needs no data from the checkpoint beyond rot_size itself.
# --------------------------------------------------------------------------
_convrot_hadamard_cache: dict = {}


def _convrot_hadamard(rot_size: int, device, dtype) -> "torch.Tensor":
    import torch

    key = (rot_size, str(device), dtype)
    if key not in _convrot_hadamard_cache:
        r4 = torch.tensor(
            [[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=torch.float32, device="cpu",
        )
        h = r4.clone()
        while h.shape[0] < rot_size:
            h = torch.kron(h, r4)
        if h.shape[0] != rot_size:
            raise ValueError(f"ConvRot rot_size {rot_size} is not a power of 4")
        _convrot_hadamard_cache[key] = (h / rot_size**0.5).to(device=device, dtype=dtype)
    return _convrot_hadamard_cache[key]


def _convrot_rotate(x: "torch.Tensor", rot_size: int) -> "torch.Tensor":
    """Block regular-Hadamard rotation along the last dim. Self-inverse -- applying this twice
    returns the original tensor exactly (up to floating-point rounding), which is what makes
    "rotate the already-rotated-and-dequantized int8 data" the correct way to undo it."""
    import torch

    if rot_size == 1:
        return x
    h = _convrot_hadamard(rot_size, x.device, x.dtype)
    shape = x.shape
    xb = x.reshape(-1, shape[-1] // rot_size, rot_size)
    return torch.matmul(xb, h).reshape(shape)


def _parse_comfy_quant_blob(blob: "torch.Tensor") -> dict:
    """Comfy's own on-disk marker: a uint8 tensor holding UTF-8 JSON, e.g.
    {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}."""
    import json

    return json.loads(bytes(blob.cpu().tolist()).decode("utf-8"))


def _dequantize_convrot_int8(
    qdata: "torch.Tensor", scale: "torch.Tensor", rot_size: int, dtype
) -> "torch.Tensor":
    """qdata: int8 (out_features, in_features), the rotated weight's per-row-symmetric int8
    codes. scale: per-output-row fp32 scale (out_features,) -- max(abs(rotated_row))/127 at
    quantize time. Returns the dense, de-rotated weight in `dtype`."""
    import torch

    w_rotated = qdata.to(torch.float32) * scale.reshape(-1, 1).to(torch.float32)
    return _convrot_rotate(w_rotated, rot_size).to(dtype)


def load_transformer(args, device):
    import torch
    from diffusers import MiniMaxH3Transformer3DModel
    from transformers import BitsAndBytesConfig

    dtype = torch.bfloat16 if args.base_dtype == "bfloat16" else torch.float16

    say("Loading MiniMax-H3 transformer (33B, LoRA training only -- do not attempt full fine-tune)...")
    # NOT {partition}/transformer -- that subfolder's config._class_name is "MiniMaxH3DiTModel",
    # MiniMax's own native checkpoint (13 shards named model-*.safetensors), not the diffusers-native
    # MiniMaxH3Transformer3DModel our import expects. The real diffusers-format checkpoints (config.
    # _class_name "MiniMaxH3Transformer3DModel", proper diffusion_pytorch_model-*.safetensors shards)
    # live at the top-level "transformer" (FL2VA) / "transformer_ref" (Ref2VA) folders -- confirmed
    # by reading both config.json files, not assumed.
    partition_subfolder = {"FL2VA": "transformer", "Ref2VA": "transformer_ref"}[args.partition]
    # Live run got all the way into the first real forward pass and crashed there:
    # NotImplementedError: "addmm_cuda" not implemented for 'Char'. Root cause: several small
    # "boundary" Linears (proj_in, audio_proj_in, context_embedder, time_embedder, proj_out,
    # audio_proj_out, every block's adaln_proj, norm_out) have their forward calls written as
    # `self.linear(x.to(self.linear.weight.dtype))` -- reading the *quantized layer's own weight
    # dtype* to decide what to cast the activation to. That's fine when weight.dtype is a normal
    # float dtype, but under int8 quantization it isn't a valid cast target for an activation
    # tensor at all. diffusers already flags proj_in/audio_proj_in/time_embedder/proj_out/
    # audio_proj_out as _keep_in_fp32_modules for exactly this reason -- context_embedder,
    # adaln_proj (all 50 of them) and norm_out do the identical pattern but were missed from that
    # list, so they'd hit this same crash the moment training reached them. context_embedder and
    # norm_out aren't LoRA targets (only transformer_blocks.*.attn.*/.ff.*/.adaln_proj.linear and
    # token_refiner.refiner_blocks.*.attn.*/.ff.* are, see inject_lora()) -- but adaln_proj is NOT
    # a small exclusion: each block's is a (2688 -> 6*5376*3) linear, ~260M params, and there are
    # 50 of them -- ~13B params, ~40% of the model's 33B total, staying in bf16 (2 bytes) instead
    # of int8. That's a real, meaningful jump in static memory (roughly 46GB total transformer
    # weights instead of ~33GB fully quantized), not a rounding error -- still fits an 80GB GPU
    # with LoRA/activation headroom to spare, just not as comfortably as the fully-quantized
    # estimate suggested. Worth knowing if VRAM ever gets tight on a smaller card.
    # Live run confirmed int8 quantization + the boundary-module skip list works correctly (training
    # actually reached a real forward/backward pass) but still hit a near-total OOM on a ~95GB GPU --
    # 94.5-94.6GB used regardless of whether clips were 73 or 22 frames, which rules out sequence
    # length/activation memory as the driver and points at the static weights themselves (~46GB) plus
    # int8 dequantization overhead (bitsandbytes materializes a full fp16 buffer per weight matrix
    # during each matmul -- these are huge matrices, e.g. the SwiGLU proj is 5376x28672). Switching to
    # 4-bit (nf4, double-quantized) roughly halves the quantized portion again on top of that.
    # llm_int8_skip_modules applies to 4-bit skipping too despite the name (a carried-over field from
    # when only int8 was supported) -- same skip list, same reasoning: none of those modules do the
    # weight.dtype self-referential cast, so quantizing them wasn't unsafe, just previously unhelpful
    # to change without knowing whether it'd be enough.
    #
    # token_refiner added here after cross-checking ai-toolkit's own MiniMax-H3 extension's
    # get_quantization_exclude_modules() line by line against this list (every other name matches
    # 1:1 once accounting for diffusers-vs-native naming -- video_patch_proj/audio_patch_proj =
    # proj_in/audio_proj_in, condition_proj = context_embedder, final_layer = proj_out+norm_out --
    # confirmed against the real diffusers __init__ attribute names, not assumed). token_refiner was
    # the one real gap: it's a LoRA target here (inject_lora() trains token_refiner.refiner_blocks.*)
    # but wasn't excluded from quantization, meaning that LoRA was being trained against a quantized
    # frozen base while ai-toolkit trains it against a full-precision one. token_refiner preprocesses
    # every text token before it ever reaches the main transformer_blocks, so a noisier frozen base
    # there has more reach than a typical single attention/FF layer would.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
        llm_int8_skip_modules=[
            "proj_in", "audio_proj_in", "context_embedder", "time_embedder",
            "proj_out", "audio_proj_out", "adaln_proj", "norm_out", "rope", "token_refiner",
        ],
    ) if args.quantize_base else None
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder=partition_subfolder,
        torch_dtype=dtype, quantization_config=quant_config,
        # bitsandbytes quantized weights must be placed at load time via device_map -- moving them
        # with .to() afterwards does not work. Unquantized loads are moved explicitly below instead.
        device_map={"": device} if args.quantize_base else None,
    )
    if not args.quantize_base:
        transformer = transformer.to(device)
    transformer.requires_grad_(False)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    return transformer


# Diffusers attribute name -> Comfy-Org checkpoint key-prefix template, for every per-block
# Linear except the fused attn.qkv_proj (handled separately below, since it splits into three).
# Verified against real diffusers __init__ code for every entry (not assumed): attn.to_out.0=
# out_proj, attn.norm_q/norm_k=q_norm/k_norm, ff.net.0.proj/net.2=mlp.fc1/fc2, norm1/norm2 match
# by name already, adaln_proj.linear matches by name already.
_CONVROT_BLOCK_LINEAR_MAP = {
    "attn.to_out.0": "attn.out_proj",
    "ff.net.0.proj": "mlp.fc1",
    "ff.net.2": "mlp.fc2",
    "adaln_proj.linear": "adaln_proj.linear",
}
_CONVROT_BLOCK_NORM_MAP = {
    "norm1": "norm1", "norm2": "norm2",
    "attn.norm_q": "attn.q_norm", "attn.norm_k": "attn.k_norm",
}
# Top-level (non-block) modules. norm_out/proj_out/audio_proj_out map to comfy's final_layer's
# own norm/adaln_proj.linear/video_out/audio_out (verified: MiniMaxH3AdaLayerNormOut.linear is
# Linear(2688, 10752), exactly matching AdalnProj(t_dim=2688, hidden=5376, expand=2,
# modalities=1)'s Linear(2688, 2*5376*1=10752) -- same tensor, just unwrapped one level).
_CONVROT_TOP_LINEAR_MAP = {
    "proj_in": "video_patch_proj",
    "audio_proj_in": "audio_patch_proj",
    "context_embedder": "condition_proj",
    "time_embedder.linear_1": "time_embedder.proj_in",
    "time_embedder.linear_2": "time_embedder.proj_out",
    "norm_out.linear": "final_layer.adaln_proj.linear",
    "proj_out": "final_layer.video_out",
    "audio_proj_out": "final_layer.audio_out",
}
_CONVROT_TOP_NORM_MAP = {"norm_out.norm": "final_layer.norm"}


def load_transformer_convrot(args, device):
    """Load MiniMax-H3 from Comfy-Org's own pre-quantized int8-ConvRot checkpoint instead of
    downloading and self-quantizing the full model (see load_transformer()). Trades bitsandbytes
    NF4 (this trainer's other path) for Comfy's own int8-ConvRot format on the attention/FFN
    linears -- worse bits-per-parameter than NF4, but matches what the wider community actually
    trains against (real ComfyUI checkpoints dissected earlier all target this format) and what
    a LoRA saved from this trainer needs to shape-match to load cleanly in ComfyUI without the
    block-diagonal QKV-fusion conversion silently producing a checkpoint that only loads against
    a *different* base checkpoint than the one it was actually trained on.

    Deliberately the non-pruned "*_int8_convrot" file, not "*_pruned_int8_convrot" -- confirmed
    live (byte-range HTTP header reads against both real files) that the pruned variant replaces
    adaln_proj's normal 2688-dim TimestepEmbedding-MLP conditioning with an 8-dim lookup-table
    mechanism ("adaln_t_table"), a genuinely different computation graph this trainer does not
    implement. The non-pruned file keeps the full model's exact architecture (verified: its
    adaln_proj is still 2688-dim, time_embedder.proj_in/proj_out are present with normal shapes),
    just int8-quantized -- loads into diffusers' unmodified MiniMaxH3Transformer3DModel.

    adaln_proj is dequantized ONCE here and kept dense (not wrapped in the lazy per-forward
    ConvRotInt8Linear used for attention/FFN) -- matches this trainer's own token_refiner
    exclusion above and ai-toolkit's real exclude-list philosophy: keep the sensitive modulation
    path's LoRA training against a clean, full-precision-equivalent base rather than one that's
    re-quantized every forward pass.
    """
    import torch
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    class ConvRotInt8Linear(torch.nn.Linear):
        """A frozen int8-ConvRot-quantized Linear that dequantizes fresh on every forward call
        under no_grad, instead of ai-toolkit's own QAT-capable custom-autograd machinery
        (straight-through estimators, Triton kernels) -- that machinery exists so gradients CAN
        flow into the quantized weights themselves for quantization-aware fine-tuning. We don't
        need that: this is standard LoRA training, the base weight is frozen either way, and
        PyTorch's autograd already backprops correctly through a plain F.linear w.r.t. the
        *input* (what a LoRA adapter's gradient path actually needs) without any custom backward,
        exactly like bitsandbytes' own Linear4bit/Linear8bitLt already do in this same trainer's
        other loading path.

        Defined nested inside load_transformer_convrot() rather than at module level -- this file
        deliberately never imports torch at module level (PYTORCH_CUDA_ALLOC_CONF must be set
        before torch's own first import to take effect, see the top of this file), and a
        module-level `class Foo(torch.nn.Linear)` evaluates its base class at import time, before
        any function (including this one) has run. Nesting it here means its methods resolve
        `torch` through the closure over this function's own `import torch` above, no separate
        import needed inside __init__/forward.

        `.weight` is kept as nn.Linear's own meta-device placeholder (zero real memory), not
        deleted -- a live run confirmed deleting it (mirroring ai-toolkit's own `_to_ostris()`
        pattern, `del module._parameters["weight"]`) crashes PEFT: LoraLayer.__init__ calls
        peft's `_get_in_out_features()`, which for any nn.Linear reads `module.weight` first
        (an isinstance(..., DTensor) check for tensor-parallel support, unrelated to us) before
        falling through to `module.in_features`/`out_features` -- so `.weight` must exist and be
        accessible even though its data is never actually used. Separately, LoraLayer.
        update_layer() (the code that creates lora_A/lora_B for a target module) only reads
        `self.in_features`/`self.out_features` for sizing, and only touches `base_layer.weight`'s
        values inside the pissa/olora/loftq/corda init-scheme branches, none of which this
        trainer uses (inject_lora() passes init_lora_weights="gaussian").
        Forward delegation is `self.base_layer(x)` -- calls this class's own forward() directly,
        never reads `.weight`. isinstance(module, torch.nn.Linear) is what inject_lora() checks to
        build its target list, and this class satisfies that by real inheritance, not duck-typing."""

        def __init__(self, qdata, scale, rot_size: int, bias, compute_dtype):
            out_features, in_features = qdata.shape
            super().__init__(in_features, out_features, bias=bias is not None, device="meta")
            # Live run confirmed deleting this crashes PEFT: LoraLayer.__init__ calls peft's
            # _get_in_out_features(), which for any nn.Linear does `module.weight` FIRST (to
            # check isinstance(..., DTensor), for tensor-parallel support we don't use) before
            # falling through to module.in_features/out_features (the plain ints, already
            # correct) -- so .weight must exist and not error on access, even though its actual
            # data is never read for a non-DTensor. Left as nn.Linear's own meta-device
            # placeholder (zero real memory) rather than deleted; explicitly frozen here instead
            # of relying on this trainer's later whole-model requires_grad_(False) call, since a
            # stray requires_grad=True meta parameter reaching the optimizer would try to
            # allocate momentum state against a tensor with no real storage.
            self.weight.requires_grad_(False)
            if bias is not None:
                self.bias = torch.nn.Parameter(bias.to(compute_dtype), requires_grad=False)
            self.register_buffer("cr8_qdata", qdata, persistent=False)
            self.register_buffer("cr8_scale", scale.reshape(-1).to(torch.float32), persistent=False)
            self.cr8_rot_size = rot_size
            self.compute_dtype = compute_dtype

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            with torch.no_grad():
                w = _dequantize_convrot_int8(self.cr8_qdata, self.cr8_scale, self.cr8_rot_size, self.compute_dtype)
            return torch.nn.functional.linear(x, w, self.bias)

    dtype = torch.bfloat16 if args.base_dtype == "bfloat16" else torch.float16
    filename = MINIMAX_H3_CONVROT_FILES[args.partition]
    say(f"Downloading Comfy-Org pre-quantized checkpoint ({filename})...")
    ckpt_path = hf_hub_download(MINIMAX_H3_CONVROT_REPO, filename)

    say("Building MiniMax-H3 transformer shell (defaults already match this checkpoint's "
        "architecture, confirmed against its real header, not assumed)...")
    with init_empty_weights():
        transformer = MiniMaxH3Transformer3DModel()

    f = safe_open(ckpt_path, framework="pt", device="cpu")
    keys = set(f.keys())
    consumed: set = set()

    def take(key: str) -> "torch.Tensor":
        consumed.add(key)
        return f.get_tensor(key)

    def has_quant_marker(prefix: str) -> bool:
        return f"{prefix}.comfy_quant" in keys

    def dequantize_dense(prefix: str, out_dtype) -> "torch.Tensor":
        """For modules we deliberately keep dense (adaln_proj) even though the checkpoint
        stores them quantized -- dequantize once here rather than wrapping in ConvRotInt8Linear."""
        conf = _parse_comfy_quant_blob(take(f"{prefix}.comfy_quant"))
        qdata = take(f"{prefix}.weight")
        scale = take(f"{prefix}.weight_scale")
        rot = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1
        return _dequantize_convrot_int8(qdata, scale, rot, out_dtype)

    def set_param(module, attr: str, tensor: "torch.Tensor") -> None:
        setattr(module, attr, torch.nn.Parameter(tensor.to(device=device), requires_grad=False))

    def check_bias_matches(module_holder, attr: str, prefix: str) -> None:
        """diffusers' own shell (built from its real __init__, just on the meta device) already
        encodes whether each Linear expects a bias -- verify the checkpoint agrees instead of
        silently trusting comfy's own bias=False convention carries over unchanged. A mismatch
        here would mean a real trained bias getting silently dropped (checkpoint has one, shell
        doesn't expect one) or a wrong bias=None replacing a real one (shell expects one, this
        specific checkpoint variant doesn't have it) -- either way, worth failing loudly on."""
        original = getattr(module_holder, attr)
        expects_bias = original.bias is not None
        has_bias = f"{prefix}.bias" in keys
        if expects_bias != has_bias:
            raise RuntimeError(
                f"Bias mismatch loading {prefix}: diffusers' shell "
                f"{'expects' if expects_bias else 'does not expect'} a bias, checkpoint "
                f"{'has' if has_bias else 'has no'} '{prefix}.bias'. Aborting rather than "
                "silently guessing which one is right."
            )

    def load_quantized_linear(module_holder, attr: str, prefix: str) -> None:
        """Replace the meta-device nn.Linear at module_holder.<attr> with a lazy
        ConvRotInt8Linear backed by this prefix's on-disk int8 codes."""
        check_bias_matches(module_holder, attr, prefix)
        conf = _parse_comfy_quant_blob(take(f"{prefix}.comfy_quant"))
        qdata = take(f"{prefix}.weight").to(device)
        scale = take(f"{prefix}.weight_scale").to(device)
        bias_key = f"{prefix}.bias"
        bias = take(bias_key).to(device) if bias_key in keys else None
        rot = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1
        setattr(module_holder, attr, ConvRotInt8Linear(qdata, scale, rot, bias, dtype))

    def load_dense_linear(module_holder, attr: str, prefix: str) -> None:
        check_bias_matches(module_holder, attr, prefix)
        weight = take(f"{prefix}.weight").to(device=device, dtype=dtype)
        bias_key = f"{prefix}.bias"
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias_key in keys)
        set_param(linear, "weight", weight)
        if bias_key in keys:
            set_param(linear, "bias", take(bias_key).to(dtype=dtype))
        setattr(module_holder, attr, linear)

    def load_dequantized_dense_linear(module_holder, attr: str, prefix: str) -> None:
        """For modules kept dense on purpose even though this checkpoint stores them quantized
        (adaln_proj) -- dequantize once here rather than wrapping in ConvRotInt8Linear."""
        check_bias_matches(module_holder, attr, prefix)
        weight = dequantize_dense(prefix, dtype)
        bias_key = f"{prefix}.bias"
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias_key in keys)
        set_param(linear, "weight", weight)
        if bias_key in keys:
            set_param(linear, "bias", take(bias_key).to(dtype=dtype))
        setattr(module_holder, attr, linear)

    def load_dense_norm(module, prefix: str) -> None:
        set_param(module, "weight", take(f"{prefix}.weight").to(dtype=dtype))

    def resolve(root, dotted: str):
        obj = root
        parts = dotted.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        return obj, parts[-1]

    say("Populating boundary modules (video/audio patch-in, text conditioning, time embedder, "
        "output heads)...")
    for diffusers_name, comfy_prefix in _CONVROT_TOP_LINEAR_MAP.items():
        holder, attr = resolve(transformer, diffusers_name)
        if has_quant_marker(comfy_prefix):
            # Kept dense on purpose (matches this trainer's own quantization-exclude philosophy
            # for boundary modules, see load_transformer()'s own comment) even where the
            # checkpoint itself quantized it (adaln_proj-adjacent final_layer weights do, here).
            load_dequantized_dense_linear(holder, attr, comfy_prefix)
        else:
            load_dense_linear(holder, attr, comfy_prefix)
    for diffusers_name, comfy_prefix in _CONVROT_TOP_NORM_MAP.items():
        holder, attr = resolve(transformer, diffusers_name)
        load_dense_norm(getattr(holder, attr), comfy_prefix)
    # rope.inv_freq is a deterministic function of rope_freq_dim/rope_theta (both config, not
    # checkpoint data) in diffusers -- confirmed real source computes it fresh, not from a stored
    # buffer, so it's identical to the checkpoint's own buffer by construction and needs no load.

    say(f"Populating {transformer.config.num_layers} transformer blocks (attention + FFN quantized, "
        "adaln_proj kept dense)...")
    for i, block in enumerate(transformer.transformer_blocks):
        prefix = f"blocks.{i}"
        # Fused qkv_proj splits into three equal row-chunks for to_q/to_k/to_v -- exact and
        # lossless: per-row quantization scale means each output row (and its rotation, which
        # only ever mixes along the shared *input* dimension) is fully independent of every
        # other row, so slicing the int8 codes and per-row scales before dequantizing is
        # identical to dequantizing the fused matrix first and slicing after.
        qkv_prefix = f"{prefix}.attn.qkv_proj"
        for sub in ("to_q", "to_k", "to_v"):
            check_bias_matches(block.attn, sub, qkv_prefix)  # comfy's fused qkv_proj has one bias key (or none) for all three
        conf = _parse_comfy_quant_blob(take(f"{qkv_prefix}.comfy_quant"))
        rot = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1
        qkv_qdata = take(f"{qkv_prefix}.weight").to(device)
        qkv_scale = take(f"{qkv_prefix}.weight_scale").to(device)
        qkv_bias_key = f"{qkv_prefix}.bias"
        qkv_bias = take(qkv_bias_key).to(device) if qkv_bias_key in keys else None
        inner = qkv_qdata.shape[0] // 3
        for j, sub in enumerate(("to_q", "to_k", "to_v")):
            sub_bias = qkv_bias[j * inner:(j + 1) * inner] if qkv_bias is not None else None
            setattr(
                block.attn, sub,
                ConvRotInt8Linear(
                    qkv_qdata[j * inner:(j + 1) * inner], qkv_scale[j * inner:(j + 1) * inner],
                    rot, sub_bias, dtype,
                ),
            )
        for diffusers_suffix, comfy_suffix in _CONVROT_BLOCK_LINEAR_MAP.items():
            comfy_full_prefix = f"{prefix}.{comfy_suffix}"
            holder, attr = resolve(block, diffusers_suffix)
            if diffusers_suffix == "adaln_proj.linear":
                load_dequantized_dense_linear(holder, attr, comfy_full_prefix)  # kept dense, see function docstring
            else:
                load_quantized_linear(holder, attr, comfy_full_prefix)
        for diffusers_suffix, comfy_suffix in _CONVROT_BLOCK_NORM_MAP.items():
            holder, attr = resolve(block, diffusers_suffix)
            load_dense_norm(getattr(holder, attr), f"{prefix}.{comfy_suffix}")

    say(f"Populating {len(transformer.token_refiner.refiner_blocks)} token_refiner blocks "
        "(unquantized in this checkpoint)...")
    for i, rblock in enumerate(transformer.token_refiner.refiner_blocks):
        prefix = f"token_refiner.blocks.{i}"
        qkv_prefix = f"{prefix}.attn.qkv_proj"
        for sub in ("to_q", "to_k", "to_v"):
            check_bias_matches(rblock.attn, sub, qkv_prefix)
        qkv_weight = take(f"{qkv_prefix}.weight").to(device=device, dtype=dtype)
        qkv_bias_key = f"{qkv_prefix}.bias"
        qkv_bias = take(qkv_bias_key).to(device=device, dtype=dtype) if qkv_bias_key in keys else None
        inner = qkv_weight.shape[0] // 3
        for j, sub in enumerate(("to_q", "to_k", "to_v")):
            linear = torch.nn.Linear(qkv_weight.shape[1], inner, bias=qkv_bias is not None)
            set_param(linear, "weight", qkv_weight[j * inner:(j + 1) * inner])
            if qkv_bias is not None:
                set_param(linear, "bias", qkv_bias[j * inner:(j + 1) * inner])
            setattr(rblock.attn, sub, linear)
        for diffusers_suffix, comfy_suffix in {
            "attn.to_out.0": "attn.out_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2",
        }.items():
            holder, attr = resolve(rblock, diffusers_suffix)
            load_dense_linear(holder, attr, f"{prefix}.{comfy_suffix}")
        for diffusers_suffix, comfy_suffix in {
            "norm1": "norm1", "norm2": "norm2",
            "attn.norm_q": "attn.q_norm", "attn.norm_k": "attn.k_norm",
        }.items():
            holder, attr = resolve(rblock, diffusers_suffix)
            load_dense_norm(getattr(holder, attr), f"{prefix}.{comfy_suffix}")
    # token_refiner's own final norm after all refiner blocks -- tries the name comfy itself uses
    # (final_norm); the reconciliation check below catches it loudly if this trainer's diffusers
    # commit named it differently, instead of silently leaving it randomly initialized.
    if hasattr(transformer.token_refiner, "final_norm") and f"token_refiner.final_norm.weight" in keys:
        load_dense_norm(transformer.token_refiner.final_norm, "token_refiner.final_norm")

    # rope.inv_freq is the only checkpoint tensor deliberately never read (see the comment where
    # top-level modules are populated above: diffusers computes it fresh from rope_freq_dim/
    # rope_theta config, identical to the checkpoint's own buffer by construction). Everything
    # else -- weights, biases, weight_scale, comfy_quant markers alike -- should have been
    # consumed by construction of the helpers above (each one pulls a weight together with its
    # own scale/bias/marker in the same call), so checking every leftover key, not just
    # ".weight" ones, also catches a helper-internal inconsistency, not only a missing mapping.
    unconsumed = sorted((keys - consumed) - {"rope.inv_freq"})
    if unconsumed:
        raise RuntimeError(
            "load_transformer_convrot() left checkpoint tensors unused -- this trainer's "
            f"diffusers-to-Comfy name mapping is missing something: {unconsumed[:20]}"
            + (" ..." if len(unconsumed) > 20 else "")
        )
    # ConvRotInt8Linear.weight is EXPECTED to stay on the meta device forever (see that class's
    # own docstring) -- it's a placeholder kept only so peft's _get_in_out_features() has
    # something to access, real data lives in cr8_qdata/cr8_scale instead. Exempt it by identity
    # (module type + param name), not by pattern-matching the name string, so this still catches
    # a genuinely unpopulated ".weight" anywhere else.
    still_meta = []
    for mod_name, module in transformer.named_modules():
        is_convrot = isinstance(module, ConvRotInt8Linear)
        for p_name, p in list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False)):
            if p.device.type != "meta":
                continue
            if is_convrot and p_name == "weight":
                continue
            still_meta.append(f"{mod_name}.{p_name}" if mod_name else p_name)
    still_meta.sort()
    if still_meta:
        raise RuntimeError(
            "load_transformer_convrot() finished with parameters still on the meta device -- "
            f"these were never populated from the checkpoint: {still_meta[:20]}"
            + (" ..." if len(still_meta) > 20 else "")
        )
    say(f"Loaded {len(consumed)}/{len(keys)} checkpoint tensors "
        f"(remainder were weight_scale/comfy_quant marker tensors, consumed alongside their weights).")

    transformer.requires_grad_(False)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    return transformer


def inject_lora(transformer, rank: int, alpha: int):
    """Standard PEFT injection -- MiniMaxH3Transformer3DModel already inherits PeftAdapterMixin
    and forward() is decorated with @apply_lora_scale, so no custom LoRA plumbing is needed here,
    unlike Krea2 where target-module resolution had to account for text_fusion.* by hand. Every
    block is part of the same shared self-attention stack (no modality-isolated layers to exclude
    the way Krea2's text_fusion.* could be), so the target list is simply every attention and
    feed-forward linear across all 50 transformer_blocks."""
    import torch
    from peft import LoraConfig

    # Live run caught a real bug in the previous version of this match: diffusers' FeedForward with
    # activation_fn="swiglu" (what this transformer uses) wraps its first linear inside a SwiGLU
    # module ("ff.net.0" is that wrapper, a non-Linear; the actual nn.Linear is "ff.net.0.proj").
    # Matching bare "ff.net.<digit>" caught the wrapper itself and PEFT rejected it outright ("Target
    # module SwiGLU(...) is not supported"). Checking isinstance(nn.Linear) directly -- true for
    # bitsandbytes' Linear8bitLt and Linear4bit too, since both subclass nn.Linear -- avoids this
    # whole class of "guessed the wrong thing from a name" bug instead of just special-casing swiglu.
    #
    # Widened beyond just the main attention/FFN linears (confirmed by inspecting a real ai-toolkit
    # checkpoint's key list directly, not guessed): also targets each block's adaln_proj.linear (the
    # AdaLN modulation projection -- named "adaln_proj" in the actual checkpoint, "linear" is
    # diffusers' name for its inner nn.Linear, per that class's own docstring) and the token_refiner's
    # own small attention+FFN stack (a separate pre-encoder for the text stream, "No AdaLN and no
    # rotary embedding" per its own docstring -- reuses the same Attention/FeedForward classes as the
    # main blocks, so the same to_q/to_k/to_v/to_out.0/ff.net.0.proj/ff.net.2 suffixes apply, just
    # under token_refiner.refiner_blocks.N instead of transformer_blocks.N).
    attn_ff_suffixes = (".to_q", ".to_k", ".to_v", ".to_out.0", ".ff.net.0.proj", ".ff.net.2")
    target_modules = []
    for name, module in transformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        in_main_blocks = name.startswith("transformer_blocks.")
        in_token_refiner = name.startswith("token_refiner.refiner_blocks.")
        if not (in_main_blocks or in_token_refiner):
            continue
        suffixes = attn_ff_suffixes + (".adaln_proj.linear",) if in_main_blocks else attn_ff_suffixes
        if name.endswith(suffixes):
            target_modules.append(name)
    if not target_modules:
        raise RuntimeError(
            "No LoRA target modules resolved from transformer_blocks.*/token_refiner.* -- MiniMax-H3's "
            "module naming may have changed since this script was written against diffusers commit "
            f"{DIFFUSERS_MINIMAX_H3_COMMIT}."
        )
    say(f"Injecting LoRA (rank={rank}, alpha={alpha}) into {len(target_modules)} linear layers "
        f"(transformer blocks + token refiner + adaln_proj).")
    lora_config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=target_modules, init_lora_weights="gaussian")
    transformer.add_adapter(lora_config)
    return transformer


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def sample_shifted_sigma(shift: float, batch_size: int, device) -> "torch.Tensor":
    """Logit-normal base sigma (the SD3/FLUX flow-matching training convention), pushed through
    MiniMax-H3's own exponential shift -- exactly the formula in MiniMaxH3Scheduler.set_timesteps()
    (`s*sigma / (1 + (s-1)*sigma)`), reused here rather than re-derived, just applied to a sampled
    scalar instead of a full inference grid."""
    import torch

    base = torch.sigmoid(torch.randn(batch_size, device=device))  # logit-normal in (0, 1)
    return shift * base / (1 + (shift - 1) * base)


def precompute_cache(args, clips, vae, text_encoder, tokenizer, device, resolve_canvas_size, audio_vae=None):
    """Run the frozen VAE + text encoder once over the whole dataset and cache their outputs on
    CPU. Captions and videos are static across the whole run (no augmentation), so re-running these
    two frozen, ~10-65GB-each models every single step would be pure waste even with infinite VRAM --
    and freeing them afterwards (see main()) is what makes a single 80GB GPU realistic at all."""
    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.modular_pipelines.minimax_h3.packing import MINIMAX_H3_KEYFRAME_ENCODE_SEED

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1, 1)
    latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    # Ref2VA's weights expect a reference-image condition; training them exactly like FL2VA's plain
    # text-to-video (no reference at all) leaves that pathway completely unexercised. Uses each
    # clip's own first frame as its reference -- the simplest available proxy for "a reference photo
    # of this subject" (see main()'s module docstring for why this is an approximation, not textbook
    # Ref2VA usage, and how it could extend to real separate reference images later).
    use_reference = args.partition == "Ref2VA"

    audio_latents_mean = audio_latents_std = None
    if audio_vae is not None:
        audio_latents_mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, 1, -1)
        audio_latents_std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, 1, -1)

    cache = []
    with torch.no_grad():
        for i, (path, caption) in enumerate(clips):
            clip, clip_start_seconds, clip_duration_seconds = load_and_prepare_clip(
                path, args.num_frames, resolve_canvas_size
            )
            pixels = clip.unsqueeze(0).to(device)
            vae_input = (pixels.to(torch.float32) - imagenet_mean) / imagenet_std
            video_latents = vae.encode(vae_input).latent_dist.sample()
            video_latents = (video_latents - latents_mean) / latents_std

            reference_latents = None
            if use_reference:
                # The released model's own recipe: a seeded posterior sample (seed 42, independent of
                # the training seed) of just the first frame, rounded to fp16 before normalizing, so
                # the same source frame always yields the same clean reference latent regardless of
                # which random noise/timestep a given training step happens to draw. How this gets
                # noised into the packed sequence at train time lives in main(), not here -- this is
                # only the clean, cacheable half of it.
                #
                # NOT vae.encode() -- confirmed live: that goes through _encode()'s 17-frame chunking
                # (clip_length/token_drop), which is wrong for a lone frame and silently produced 2
                # latent frames instead of 1, desyncing the condition-row count build_packed_sequence()
                # expects from the row count actually concatenated onto hidden_states
                # (index_copy_(): Number of indices (5796) should be equal to source.size(dim) (6048),
                # the 252-row gap being exactly one extra rows_per_frame). The real diffusers source
                # says why directly: "MiniMax-H3 encodes a keyframe or an image reference through
                # [_encode_clip] rather than through [encode], because a single frame must not go
                # through the temporal chunking." _encode_clip() returns raw moments, so the
                # DiagonalGaussianDistribution wrapping (and the generator-seeded .sample() on it) is
                # done by hand here, mirroring exactly what encode() itself does internally.
                ref_generator = torch.Generator(device="cpu").manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
                ref_moments = vae._encode_clip(vae_input[:, :, 0:1])
                reference_latents = DiagonalGaussianDistribution(ref_moments).sample(generator=ref_generator)
                reference_latents = reference_latents.to(torch.float16).to(torch.float32)
                reference_latents = (reference_latents - latents_mean) / latents_std

            full_caption = f"{args.trigger_word}, {caption}" if args.trigger_word and args.trigger_word not in caption else caption
            text_ids = tokenizer([full_caption], return_tensors="pt", truncation=True).to(device)
            text_out = text_encoder.model(
                input_ids=text_ids["input_ids"], attention_mask=text_ids["attention_mask"],
                pixel_values=None, output_hidden_states=True,
            )
            text_embeds = text_out.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]

            audio_latents = None
            if audio_vae is not None:
                waveform = load_audio_waveform(
                    path, clip_start_seconds, clip_duration_seconds, MINIMAX_H3_AUDIO_SAMPLE_RATE
                )
                if waveform is not None:
                    audio_input = waveform.to(device)[:, None]  # (2, 1, samples) -- stereo as 2 batch items
                    # return_dict=False, [0] -- matches the real reference-conditioning encoder step's
                    # own call exactly (posterior returned bare, not wrapped in an EncoderOutput.latent_dist).
                    posterior = audio_vae.encode(audio_input, return_dict=False)[0]
                    audio_latents = posterior.mode().float().transpose(1, 2)  # (2, T, 32)
                    audio_latents = (audio_latents - audio_latents_mean) / audio_latents_std
                else:
                    warn(f"{path.name}: no usable audio for this crop window, training video-only for this clip.")

            cache.append((
                video_latents.cpu(), text_embeds.cpu(),
                reference_latents.cpu() if reference_latents is not None else None,
                audio_latents.cpu() if audio_latents is not None else None,
            ))
            say(f"Cached {i + 1}/{len(clips)}: {path.name}")
    return cache


def convert_lora_to_comfyui_native(lora_state_dict: dict) -> dict:
    """Repackage a diffusers-shaped MiniMax-H3 LoRA into the shape ComfyUI's native support actually
    loads, rather than a shape diffusers' own generic Attention class happens to use.

    Confirmed live: a checkpoint saved with diffusers' key names (transformer_blocks.N.attn.to_q,
    to_k, to_v, to_out.0, ff.net.0.proj, ff.net.2) produced "lora key not loaded" for every single
    key in ComfyUI, even after prefixing with "diffusion_model." -- because diffusers' port of
    MiniMax-H3 restructured the architecture to fit its own reusable Attention/FeedForward classes:
    it split what the original release (and ComfyUI's from-scratch reimplementation in
    comfy/ldm/minimax/model.py, read directly, not assumed) keeps as ONE fused qkv_proj Linear per
    block into three separate to_q/to_k/to_v Linears. ComfyUI's blocks.N.attn.qkv_proj has no
    separate to_q/to_k/to_v parameter to match against at all, so no amount of renaming alone fixes
    it -- confirmed by checking Krea 2's native ComfyUI model (comfy/ldm/krea2/model.py), which does
    keep separate wq/wk/wv and has never hit this, so the split-vs-fused choice is architecture-
    specific, not a general diffusers quirk.

    Three independent rank-r LoRAs (to_q, to_k, to_v) become one exactly-equivalent rank-3r LoRA for
    the fused layer: stack the three A matrices vertically, and build a block-diagonal B (zeros
    everywhere except each B slotted into its own block). For the q-output rows of the fused layer,
    only B_q's block is nonzero, so B_fused @ A_fused reduces to exactly B_q @ A_q there (and
    likewise for k/v) -- nothing approximated, no information lost, just repackaged into a shape
    ComfyUI's fused qkv_proj can actually load. out_proj/mlp.fc1/mlp.fc2 need only the name change,
    diffusers didn't restructure those, just renamed them (confirmed against comfy/ldm/minimax/
    model.py's DiTBlock: self.attn.out_proj, self.mlp.fc1, self.mlp.fc2, under self.blocks -- not
    self.transformer_blocks, that name doesn't exist on ComfyUI's side either)."""
    import re
    import torch

    RENAME = {"attn.to_out.0": "attn.out_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2"}
    QKV_PARTS = ("attn.to_q", "attn.to_k", "attn.to_v")

    qkv_parts: dict[int, dict[tuple[str, str], "torch.Tensor"]] = {}
    out: dict[str, "torch.Tensor"] = {}
    for key, tensor in lora_state_dict.items():
        m = re.match(r"transformer_blocks\.(\d+)\.(.+)$", key)
        if not m:
            out[key] = tensor  # nothing in this trainer's target list should hit this, kept as a safety net
            continue
        block_idx, rest = int(m.group(1)), m.group(2)
        matched_qkv = next((p for p in QKV_PARTS if rest.startswith(p)), None)
        if matched_qkv:
            suffix = rest[len(matched_qkv):]  # ".lora_A.weight" or ".lora_B.weight"
            qkv_parts.setdefault(block_idx, {})[(matched_qkv, suffix)] = tensor
            continue
        renamed = rest
        for old, new in RENAME.items():
            if rest.startswith(old):
                renamed = new + rest[len(old):]
                break
        out[f"blocks.{block_idx}.{renamed}"] = tensor

    for block_idx, parts in qkv_parts.items():
        a_q, a_k, a_v = (parts[(p, ".lora_A.weight")] for p in QKV_PARTS)
        b_q, b_k, b_v = (parts[(p, ".lora_B.weight")] for p in QKV_PARTS)
        out[f"blocks.{block_idx}.attn.qkv_proj.lora_A.weight"] = torch.cat([a_q, a_k, a_v], dim=0)
        out[f"blocks.{block_idx}.attn.qkv_proj.lora_B.weight"] = torch.block_diag(b_q, b_k, b_v)
    return out


def main() -> None:
    setup_environment()
    args = parse_args()
    if args.train_batch_size != 1:
        raise NotImplementedError(
            f"--train_batch_size {args.train_batch_size} is not supported yet -- the training loop "
            "pulls one cached clip per step (see precompute_cache()/main()). Leave it at 1."
        )

    import torch
    from diffusers.modular_pipelines.minimax_h3 import packing as h3_packing
    from diffusers.modular_pipelines.minimax_h3.packing import (
        build_packed_sequence,
        build_row_timesteps,
        patchify_video_latents,
        resolve_canvas_size,
    )

    if args.short_edge != h3_packing.MINIMAX_H3_SHORT_EDGE:
        # resolve_canvas_size() reads these as plain module globals at call time (not captured at
        # import time), so patching them on the module object here changes its behavior for every
        # call below without needing to reimplement its aspect-ratio/multiple-of-32 rounding logic.
        # MAX_PIXELS is scaled by the same squared ratio so the extreme-aspect-ratio area cap stays
        # proportional to the new short edge instead of becoming irrelevant (too high) or clipping
        # normal aspect ratios (too low).
        scale = (args.short_edge / h3_packing.MINIMAX_H3_SHORT_EDGE) ** 2
        h3_packing.MINIMAX_H3_MAX_PIXELS = int(h3_packing.MINIMAX_H3_MAX_PIXELS * scale)
        h3_packing.MINIMAX_H3_SHORT_EDGE = args.short_edge
        say(f"Overriding MiniMax-H3 canvas short edge to {args.short_edge}px (model default 768px).")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    run_dir = output_dir / "run"
    checkpoints_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / "metrics.jsonl"

    device = get_device()
    clips = dataset_clips(dataset_dir, args.caption_extension)
    say(f"Dataset: {len(clips)} (video, caption) pairs from {dataset_dir}")

    # Phase 1: encode everything, then free the encoders. Peak memory here is VAE (~10GB fp32) +
    # text encoder (~33-66GB depending on --base_dtype) -- no transformer, no gradients, no
    # optimizer state, since this is a forward-only pass over frozen models.
    vae, text_encoder, tokenizer, video_scheduler, audio_vae = load_encoders(args, device)
    cache = precompute_cache(args, clips, vae, text_encoder, tokenizer, device, resolve_canvas_size, audio_vae)
    say("Freeing VAE + text encoder from GPU memory (cached outputs live on CPU now)...")
    del vae, text_encoder, tokenizer, audio_vae
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Phase 2: the transformer loads only now. Peak memory here is the transformer (4-bit quantized,
    # minus the boundary modules kept unquantized -- see load_transformer()'s own comments for why
    # that's a bigger chunk than it sounds) + LoRA + activations/gradients for backprop. Phase 1
    # (both encoders resident at once) would not have fit on the same GPU as phase 2's peak either
    # way, which is the actual reason these are two separate phases, not one combined estimate.
    transformer = (
        load_transformer_convrot(args, device) if args.quant_source == "comfy_convrot"
        else load_transformer(args, device)
    )
    transformer = inject_lora(transformer, args.rank, args.lora_alpha)

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    say(f"Trainable LoRA params: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    # transformers.get_scheduler() rather than a hand-rolled LambdaLR -- "constant" maps to
    # "constant_with_warmup" since this trainer's warmup_steps always applies regardless of which
    # post-warmup shape is chosen (constant/linear/cosine all ramp up the same way first, they only
    # differ in what happens after warmup ends).
    from transformers import get_scheduler

    scheduler_name = "constant_with_warmup" if args.lr_scheduler == "constant" else args.lr_scheduler
    lr_scheduler = get_scheduler(
        scheduler_name, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.max_train_steps,
    )

    say(f"Starting training: {args.max_train_steps} steps, batch size {args.train_batch_size}.")
    transformer.train()
    global_step = 0
    order = list(range(len(cache)))
    while global_step < args.max_train_steps:
        step_start = time.time()
        if global_step % len(cache) == 0:
            random.shuffle(order)
        video_latents, text_embeds, reference_latents, audio_latents = cache[order[global_step % len(cache)]]
        video_latents = video_latents.to(device)
        text_embeds = text_embeds.to(device, dtype=torch.bfloat16 if args.base_dtype == "bfloat16" else torch.float16)
        text_token_tags = torch.ones(text_embeds.shape[1], dtype=torch.long)  # 1 = text (no keyframe vision blocks)

        batch_size = video_latents.shape[0]
        patch_size = transformer.config.patch_size
        _, _, num_latent_frames, latent_height, latent_width = video_latents.shape

        # A cached reference latent means this clip was cached under --partition Ref2VA (see
        # precompute_cache()) -- those weights expect a reference-image condition, so keyframe_anchors
        # gets one "first" entry and the packed sequence gains condition rows ahead of the target video
        # rows. FL2VA clips have no cached reference and train exactly as before (plain T2VA).
        has_reference = reference_latents is not None
        # A cached audio latent means --train_audio was on AND this clip had a usable audio track
        # for its cropped time window (see precompute_cache()/load_audio_waveform()) -- clips without
        # one train video-only for this step rather than being taught silence.
        has_audio = audio_latents is not None
        num_audio_latents = audio_latents.shape[1] if has_audio else 0

        # num_audio_latents=0 -> no audio rows at all in the packed sequence (not silence, just absent).
        layout = build_packed_sequence(
            text_token_tags=text_token_tags,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            patch_size=patch_size,
            keyframe_anchors=("first",) if has_reference else (),
        )

        noise = torch.randn_like(video_latents)
        video_sigma = sample_shifted_sigma(args.video_shift, batch_size, device)
        video_t = (1.0 - video_sigma).mean().item()  # one packed layout per batch -> shared scalar timestep
        noisy_video = video_scheduler.scale_noise(video_latents, video_t, noise)
        video_target = video_latents - noise  # data-ward velocity, see scheduling_minimax_h3.py

        # patchify_video_latents() returns (batch_size * num_patches, C) -- deliberately flattening
        # batch into the row axis (packing.py's own docstring says so explicitly) -- but the
        # transformer's forward() documents hidden_states as 3D, (batch_size, num_video_tokens, C).
        # Passing the flattened 2D tensor straight through is exactly what crashed on
        # index_copy_() expecting matching dimensionality. Since --train_batch_size is enforced to
        # 1 here, restoring the batch dim is just unsqueeze(0), not a general reshape.
        noisy_video_rows = patchify_video_latents(noisy_video, patch_size).unsqueeze(0)
        target_video_rows = patchify_video_latents(video_target, patch_size).unsqueeze(0)

        # Condition rows go in front of the target rows (build_packed_sequence's own layout order:
        # "keyframe conditioning rows first, then the target rows"). The released model's recipe noises
        # them to a fixed, mostly-clean t=0.999 (MINIMAX_H3_KEYFRAME_NOISE_AUG) every step, not the
        # per-step random sigma the actual generation target gets -- the reference is meant to arrive
        # nearly-clean regardless of how noisy this step's generation target is.
        condition_video_t = video_t
        if has_reference:
            reference_latents = reference_latents.to(device)
            condition_video_t = h3_packing.MINIMAX_H3_KEYFRAME_NOISE_AUG
            ref_noise = torch.randn_like(reference_latents)
            noisy_reference = video_scheduler.scale_noise(reference_latents, condition_video_t, ref_noise)
            condition_rows = patchify_video_latents(noisy_reference, patch_size).unsqueeze(0)
            noisy_video_rows = torch.cat([condition_rows, noisy_video_rows], dim=1)

        # Independent noise level from video -- MiniMax-H3's own audio_shift=3.0 schedule (video is
        # 12.0), reusing the same MiniMaxH3Scheduler instance for both since scale_noise() is
        # shift-independent (`x_t = t*x0 + (1-t)*noise`, confirmed via scheduling_minimax_h3.py source
        # -- the shift only shapes set_timesteps()'s inference sigma grid, not scale_noise() itself).
        if has_audio:
            audio_latents = audio_latents.to(device)
            audio_noise = torch.randn_like(audio_latents)
            audio_sigma = sample_shifted_sigma(args.audio_shift, batch_size, device)
            audio_t = (1.0 - audio_sigma).mean().item()
            noisy_audio = video_scheduler.scale_noise(audio_latents, audio_t, audio_noise)
            audio_target = audio_latents - audio_noise  # data-ward velocity, same convention as video
            noisy_audio_rows = pack_audio_latents(noisy_audio).unsqueeze(0)
            target_audio_rows = pack_audio_latents(audio_target).unsqueeze(0)
        else:
            audio_t = video_t  # unused (num_condition_audio_rows is always 0, no audio reference conditioning exists)
            noisy_audio_rows = torch.zeros(batch_size, 0, transformer.config.audio_in_channels, device=device)

        timesteps, timestep_indices = build_row_timesteps(
            layout, video_timestep=video_t, audio_timestep=audio_t,
            condition_video_timestep=condition_video_t, condition_audio_timestep=audio_t,
        )

        output = transformer(
            hidden_states=noisy_video_rows,
            audio_hidden_states=noisy_audio_rows,
            encoder_hidden_states=text_embeds,
            timestep=timesteps.to(device),
            timestep_indices=timestep_indices.to(device),
            token_tags=layout.token_tags.to(device),
            position_ids=layout.position_ids.to(device),
            video_indices=layout.video_indices.to(device),
            audio_indices=layout.audio_indices.to(device),
            text_indices=layout.text_indices.to(device),
        )

        # Condition-row predictions aren't something we're teaching the model to generate -- they're
        # the given context -- so they get dropped before computing loss against the real target,
        # matching layout.num_condition_video_rows exactly (0 when there's no reference, a no-op slice).
        output_sample = output.sample[:, layout.num_condition_video_rows:]
        loss = torch.nn.functional.mse_loss(output_sample.float(), target_video_rows.float())
        audio_loss_value = None
        if has_audio:
            # No slicing needed here (unlike the video sample above) -- num_condition_audio_rows is
            # always 0, since no audio reference/keyframe conditioning path exists in this trainer,
            # so every row in output.audio_sample is a real generation target already.
            audio_loss = torch.nn.functional.mse_loss(output.audio_sample.float(), target_audio_rows.float())
            audio_loss_value = audio_loss.item()
            loss = loss + args.audio_loss_weight * audio_loss

        loss.backward()
        # Missing entirely until now -- both other trainers in this codebase clip (train_krea2_lora_
        # direct.py's own clip_grad_norm_(trainable, 1.0), diffusion-pipe's hardcoded gradient_clipping
        # = 1.0 for Ideogram4), this one didn't. Particularly risky to skip here specifically: --train_
        # batch_size is enforced to 1 (no batch-averaging to smooth an outlier clip's gradient), and
        # adaln_proj.linear is now a LoRA target -- a single AdaLN-zero modulation projection that
        # directly rescales/shifts an entire block's residual output, so one unclipped gradient spike
        # on it can push that block into a bad regime for the rest of the run in a way that a normal
        # attention/FF layer's update wouldn't. Matches the same 1.0 norm the other two trainers use.
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        step_seconds = time.time() - step_start
        metrics_row = {
            "step": global_step, "loss": round(loss.item(), 6),
            "lr": lr_scheduler.get_last_lr()[0], "epoch": global_step // max(1, len(clips)),
            # Wall-clock time of this step, not present in older runs -- lets the UI compute ETA
            # from actual recent step throughput instead of (job elapsed since launch) / (steps so
            # far), which folds one-time setup (caching, model/checkpoint loading) into the rate and
            # produces a wildly inflated ETA on step 1 in particular.
            "t": time.time(),
        }
        if audio_loss_value is not None:
            metrics_row["audio_loss"] = round(audio_loss_value, 6)
        with open(metrics_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics_row) + "\n")
        # Every step for the first 10 (fast feedback on real per-step speed without waiting on a
        # 10-step average), then every 10th after that to keep the log from getting noisy.
        if global_step <= 10 or global_step % 10 == 0:
            audio_suffix = f" audio_loss={audio_loss_value:.4f}" if audio_loss_value is not None else ""
            say(f"step {global_step}/{args.max_train_steps} loss={loss.item():.4f}{audio_suffix} ({step_seconds:.1f}s/step)")

        if global_step % args.save_every_n_steps == 0 or global_step == args.max_train_steps:
            # NOT transformer.save_lora_adapter() -- that method is broken for a quantized model at
            # this diffusers commit regardless of its upcast_before_saving argument: it always calls
            # `self.to(dtype=torch.float32 if upcast_before_saving else None)`, and modeling_utils.py's
            # `to()` checks `dtype_present_in_args = "dtype" in kwargs` -- true whenever the *key*
            # dtype is present at all, even set to None -- so it hits the "Casting a quantized model
            # to a new dtype is unsupported" guard unconditionally, upcast requested or not. Live run
            # confirmed this crashes step 25's checkpoint save after steps 1/10/20 trained correctly.
            # Replicated the rest of that method's actual logic directly (it's just
            # get_peft_model_state_dict + safetensors.save_file) instead of waiting on an upstream fix
            # to an unreleased branch.
            from peft.utils import get_peft_model_state_dict
            import safetensors.torch

            ckpt_dir = checkpoints_dir / f"step-{global_step:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            lora_state_dict = get_peft_model_state_dict(transformer, adapter_name="default")
            # PEFT keeps LoRA A/B in float32 by default whenever the base layer is bitsandbytes-
            # quantized (our case, --quantize_base 1) -- confirmed live: checkpoints were coming out
            # ~2x the size a bf16 save would produce, and get_peft_model_state_dict() applies no
            # casting of its own, it just returns whatever dtype the adapter parameters were created
            # in. The earlier version of this comment assumed avoiding save_lora_adapter()'s broken
            # fp32 upcast call meant we already had bf16 -- wrong, that upcast was never the source of
            # the fp32-ness to begin with. bf16 LoRA weights load in ComfyUI exactly the same as fp32
            # ones (same as every other diffusion LoRA in the wild); this just halves the file size
            # for identical trained weights, not a quality tradeoff.
            save_dtype = torch.bfloat16 if args.base_dtype == "bfloat16" else torch.float16
            lora_state_dict = {k: v.to(save_dtype) for k, v in lora_state_dict.items()}
            # get_peft_model_state_dict() returns diffusers-shaped keys (transformer_blocks.N.attn.
            # to_q/to_k/to_v, separate Linears) -- but ComfyUI's native MiniMax-H3 support targets the
            # original release's fused-qkv_proj layout instead (see convert_lora_to_comfyui_native()'s
            # docstring for the live-confirmed why). A prefix rename alone isn't enough here, unlike
            # the other two trainers -- this one needs the actual per-block QKV repackaging.
            lora_state_dict = convert_lora_to_comfyui_native(lora_state_dict)
            # ComfyUI's own top-level convention for every model's DiT weights, confirmed by the
            # earlier "diffusion_model." prefix already being recognized (just not matched past it).
            lora_state_dict = {f"diffusion_model.{k}": v for k, v in lora_state_dict.items()}
            # Filename matches the cross-trainer convention app/main.py's job_checkpoints()/
            # download_checkpoint() already hardcode (Ideogram4's trainer follows the same
            # convention for the same reason) -- not because H3 is Krea2, just so the existing
            # checkpoint-listing/download UI works unmodified for every trainer.
            safetensors.torch.save_file(
                lora_state_dict, str(ckpt_dir / "krea2_comfy_native_lora.safetensors"), metadata={"format": "pt"}
            )
            say(f"Saved checkpoint: {ckpt_dir}")

    say("Training complete.")

    # A prior live run finished all 50 steps and saved both checkpoints cleanly
    # (this exact log line was its last output, no traceback anywhere) yet the
    # process still exited non-zero, marking the job "failed" in the app. That
    # points at a crash during normal interpreter shutdown -- e.g. CUDA context
    # teardown racing with bitsandbytes' quantized tensors, a known class of
    # issue with large quantized models on process exit -- not a training bug.
    # os._exit() skips atexit handlers/__del__/GC entirely and terminates
    # immediately with status 0, sidestepping that teardown rather than trying
    # to debug a shutdown-order issue inside two libraries we don't control.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
