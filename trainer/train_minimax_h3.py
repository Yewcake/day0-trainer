"""MiniMax-H3 LoRA trainer -- native-rank adapters on the Diffusers model port.

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
  cross-attention anywhere, no per-modality block weights). Diffusers exposes
  split Q/K/V linears, so this trainer uses a shared-down custom adapter to
  preserve the released model's fused-QKV rank at export.
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

# Comfy-Org's own pre-quantized MiniMax-H3 release. Two genuinely different architectures live
# under this one repo (confirmed by inspecting both files' real safetensors headers, byte-range
# HTTP requests, not assumed):
#  - non-pruned ("*_int8_convrot"): adaln_proj is still 2688-dim (blocks.0.adaln_proj.linear.
#    weight is [96768, 2688], int8-quantized) and time_embedder.proj_in/proj_out are present with
#    their normal shapes -- same computation graph as the full model, just int8-quantized on
#    attention/FFN, so it loads into diffusers' unmodified MiniMaxH3Transformer3DModel as-is.
#  - pruned ("*_pruned_int8_convrot"): adaln_proj is 8-dim instead (blocks.0.adaln_proj.linear.
#    weight is [96768, 8], and final_layer's is [10752, 8]), fed by a top-level "adaln_t_table"
#    [1025, 8] float32 buffer via lookup+lerp instead of the full TimestepEmbedding MLP --
#    time_embedder.proj_in/proj_out don't exist in this file at all. Everything else (quantized
#    attention/FFN, token_refiner, boundary modules) is byte-identical in structure to the
#    non-pruned file. This is the variant most consumer-GPU users actually run (it's roughly half
#    the static-weight size), so load_transformer_convrot() supports both: for the pruned variant
#    it constructs the diffusers shell with time_embed_dim=8 and swaps in a small lookup-table
#    module (IdentityTimeProj + AdalnTableTimeEmbedder, defined inside that function) in place of
#    self.time_proj/self.time_embedder, reproducing ComfyUI's own two-line curve lookup
#    (comfy/ldm/minimax/model.py) rather than reimplementing the whole forward pass.
MINIMAX_H3_CONVROT_REPO = "Comfy-Org/MiniMax-H3"
MINIMAX_H3_CONVROT_FILES = {
    "FL2VA": "diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors",
    "Ref2VA": "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors",
}
MINIMAX_H3_CONVROT_PRUNED_FILES = {
    "FL2VA": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "Ref2VA": "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
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
        "transformers", "accelerate", "av", "bitsandbytes",
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
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler", default="constant", choices=["cosine", "constant", "linear"])
    # The training loop pulls one cached (clip, caption) pair per step -- batching multiple clips of
    # different aspect ratios/lengths into one packed layout isn't implemented, so this is accepted
    # (main.py's job form has a general Batch size field) but enforced to be 1, not silently ignored.
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=int, default=1)
    parser.add_argument("--video_shift", type=float, default=12.0)  # MiniMax-H3's own default for video rows
    parser.add_argument("--audio_shift", type=float, default=3.0)  # MiniMax-H3's own default for audio rows
    parser.add_argument("--train_audio", type=int, default=0)  # 0 = video+text only; see module docstring
    parser.add_argument("--audio_loss_weight", type=float, default=1.0)  # matches ss_h3_audio_loss_weight: 1 convention
    parser.add_argument("--guidance_distillation_scale", type=float, default=3.0)
    parser.add_argument("--base_preservation_loss_weight", type=float, default=0.05)
    parser.add_argument("--timestep_sampling", default="uniform", choices=["uniform", "logit_normal"])
    parser.add_argument("--save_dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--resume_lora", default="")
    parser.add_argument("--base_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--quantize_base", type=int, default=1)  # 4-bit (nf4) frozen base; ~33B model, LoRA-only fits nothing else
    # "bitsandbytes" (default): download the full model ourselves, quantize to NF4 at load time
    # (load_transformer()) -- --quantize_base governs this path. "comfy_convrot"/"comfy_convrot_
    # pruned": download one of Comfy-Org's own pre-quantized int8-ConvRot checkpoints instead
    # (load_transformer_convrot()), ignoring --quantize_base entirely (both are already quantized,
    # there's no "unquantized" variant to fall back to). "comfy_convrot" matches the non-pruned,
    # full-architecture checkpoint (what real community LoRAs and ai-toolkit's own default recipe
    # target); "comfy_convrot_pruned" matches the smaller, architecturally-different checkpoint
    # most consumer-GPU users actually run (see MINIMAX_H3_CONVROT_PRUNED_FILES' comment) -- a
    # LoRA trained against one is not shape-compatible with the other, ComfyUI will refuse to load
    # a mismatched pair rather than silently misbehaving.
    parser.add_argument(
        "--quant_source", default="bitsandbytes",
        choices=["bitsandbytes", "comfy_convrot", "comfy_convrot_pruned"],
    )
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
    # list, so they'd hit this same crash the moment training reached them. These boundary/AdaLN
    # modules are no longer LoRA targets, but adaln_proj is NOT a small quantization exclusion:
    # each block's is a (2688 -> 6*5376*3) linear, ~260M params, and there are
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
    # token_refiner remains excluded after cross-checking ai-toolkit's own MiniMax-H3 extension's
    # get_quantization_exclude_modules() line by line against this list (every other name matches
    # 1:1 once accounting for diffusers-vs-native naming -- video_patch_proj/audio_patch_proj =
    # proj_in/audio_proj_in, condition_proj = context_embedder, final_layer = proj_out+norm_out --
    # confirmed against the real diffusers __init__ attribute names, not assumed). token_refiner was
    # the one real gap in the original loader. It is frozen now, but it preprocesses every text token
    # before the main blocks, so preserving the reference implementation's precision remains useful.
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
    """Load MiniMax-H3 from one of Comfy-Org's own pre-quantized int8-ConvRot checkpoints instead
    of downloading and self-quantizing the full model (see load_transformer()). Trades
    bitsandbytes NF4 (this trainer's other path) for Comfy's own int8-ConvRot format on the
    attention/FFN linears -- worse bits-per-parameter than NF4, but matches what the wider
    community actually trains/infers against (real ComfyUI checkpoints dissected earlier all
    target this format) and what a LoRA saved from this trainer needs to shape-match to load
    cleanly in ComfyUI without the block-diagonal QKV-fusion conversion silently producing a
    checkpoint that only loads against a *different* base checkpoint than the one it was actually
    trained on.

    Branches on `args.quant_source` between two genuinely different checkpoint architectures
    (see MINIMAX_H3_CONVROT_PRUNED_FILES' module-level comment for how this was confirmed, real
    byte-range header reads not assumed):
      - "comfy_convrot" (non-pruned): full architecture, adaln_proj still 2688-dim, loads into
        diffusers' unmodified MiniMaxH3Transformer3DModel with zero forward-pass changes.
      - "comfy_convrot_pruned": adaln_proj is 8-dim, fed by a top-level "adaln_t_table" [1025, 8]
        lookup+lerp curve instead of the normal TimestepEmbedding MLP (which doesn't exist in
        this checkpoint at all). The diffusers shell is constructed with time_embed_dim=8 (a real
        constructor arg -- every adaln_proj.linear's in_features throughout the model derives
        from it) and `self.time_proj`/`self.time_embedder` are swapped for small local modules
        (IdentityTimeProj/AdalnTableTimeEmbedder below) that reproduce ComfyUI's own two-line
        curve lookup (comfy/ldm/minimax/model.py) instead of reimplementing the whole forward
        pass. Everything else -- quantized attention/FFN, token_refiner, other boundary modules
        -- is byte-identical in structure to the non-pruned file and shares every helper below.

    adaln_proj is dequantized ONCE here and kept dense (not wrapped in the lazy per-forward
    ConvRotInt8Linear used for attention/FFN) when the checkpoint stores it quantized -- matches
    this trainer's own token_refiner exclusion above and ai-toolkit's real exclude-list
    philosophy: keep the sensitive modulation path's LoRA training against a clean,
    full-precision-equivalent base rather than one that's re-quantized every forward pass. In the
    pruned checkpoint adaln_proj is already stored dense (unquantized), so this is a plain load.
    """
    import torch
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    pruned = args.quant_source == "comfy_convrot_pruned"

    class IdentityTimeProj(torch.nn.Module):
        """Stand-in for self.time_proj (normally a sinusoidal Timesteps embedding) when pruned:
        the pruned checkpoint's adaln_t_table lookup needs the RAW timestep value (already in
        this trainer's t=1-sigma, t=1-clean convention -- see sample_shifted_sigma()/build_row_
        timesteps()), not a sinusoidal expansion of it, so this just passes it through unchanged.
        Kept as a real nn.Module (not a bare function) so it drops into transformer.time_proj
        exactly like the module it replaces."""

        def forward(self, timestep: "torch.Tensor") -> "torch.Tensor":
            return timestep

    class AdalnTableTimeEmbedder(torch.nn.Module):
        """Stand-in for self.time_embedder when pruned. Reproduces ComfyUI's own adaln_t_table
        lookup (comfy/ldm/minimax/model.py: `pos = t.clamp(0,1)*(table.shape[0]-1); i0 =
        pos.floor().long().clamp(max=table.shape[0]-2); lerp(table[i0], table[i0+1], pos-i0)`)
        instead of the full TimestepEmbedding MLP, which this checkpoint doesn't ship weights for
        at all. Keeps everything in float32 (the table's own on-disk dtype, and the same
        precision the original model's time_embedder runs at per the non-pruned path's forward()
        comment) -- downstream MiniMaxH3AdaLayerNormModulation/AdaLayerNormOut already cast temb
        to their own projection's dtype internally before the matmul, so no cast is needed here.

        Carries a real, frozen `.linear_1` submodule purely so diffusers' own forward() --
        `temb.to(self.time_embedder.linear_1.weight.dtype)`, called right before invoking this
        module -- keeps resolving a real attribute instead of crashing; the dtype it reports
        (float32) is exactly the identity-cast this class already wants."""

        def __init__(self, table: "torch.Tensor"):
            super().__init__()
            self.register_buffer("adaln_t_table", table)
            self.linear_1 = torch.nn.Linear(1, 1, dtype=torch.float32, device=table.device)

        def forward(self, t_vals: "torch.Tensor") -> "torch.Tensor":
            table = self.adaln_t_table
            pos = t_vals.to(torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            frac = (pos - i0).unsqueeze(-1)
            return torch.lerp(table[i0], table[i0 + 1], frac)

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

        `.weight` is kept as a TINY (shape (1,)) frozen Parameter on the real device rather than
        a full dense weight. The native LoRA wrapper sizes itself from in_features/out_features,
        uses this real tensor to discover the device, and delegates base inference to forward(),
        which reads cr8_qdata/cr8_scale instead of the placeholder value."""

        def __init__(self, qdata, scale, rot_size: int, bias, compute_dtype):
            out_features, in_features = qdata.shape
            super().__init__(in_features, out_features, bias=bias is not None, device="meta")
            # Keep a real-device placeholder so generic device discovery never follows the
            # init_empty_weights() meta tensor. The true dimensions remain in in/out_features.
            self.weight = torch.nn.Parameter(
                torch.zeros(1, device=qdata.device, dtype=compute_dtype), requires_grad=False
            )
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
    filename = (MINIMAX_H3_CONVROT_PRUNED_FILES if pruned else MINIMAX_H3_CONVROT_FILES)[args.partition]
    say(f"Downloading Comfy-Org pre-quantized checkpoint ({filename})...")
    ckpt_path = hf_hub_download(MINIMAX_H3_CONVROT_REPO, filename)

    say("Building MiniMax-H3 transformer shell (defaults already match this checkpoint's "
        "architecture, confirmed against its real header, not assumed)...")
    with init_empty_weights():
        # time_embed_dim=8 for the pruned checkpoint sizes every adaln_proj.linear's in_features
        # throughout the model (main blocks + final_layer) to match its real [*, 8] weights --
        # confirmed this is a genuine constructor arg, not something requiring a subclass.
        transformer = MiniMaxH3Transformer3DModel(time_embed_dim=8) if pruned else MiniMaxH3Transformer3DModel()

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
        stores them quantized -- dequantize once here rather than wrapping in ConvRotInt8Linear.

        qdata/scale are moved to `device` BEFORE dequantizing, not after -- safe_open() hands
        back CPU tensors, and every other loader path here moves to device first for exactly
        this reason: the dequant is a real matmul (the Hadamard rotation), and adaln_proj's is
        big (96768x2688). Doing that on CPU instead of GPU is correct either way (set_param()
        would move the final result to device regardless) but silently adds real, avoidable
        minutes to every load across 51 of these (50 blocks + final_layer)."""
        conf = _parse_comfy_quant_blob(take(f"{prefix}.comfy_quant"))
        qdata = take(f"{prefix}.weight").to(device)
        scale = take(f"{prefix}.weight_scale").to(device)
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
        if pruned and diffusers_name.startswith("time_embedder."):
            # This checkpoint doesn't ship time_embedder.proj_in/proj_out at all -- adaln_t_table
            # replaces them entirely, handled below after this loop.
            continue
        holder, attr = resolve(transformer, diffusers_name)
        if has_quant_marker(comfy_prefix):
            # Kept dense on purpose (matches this trainer's own quantization-exclude philosophy
            # for boundary modules, see load_transformer()'s own comment) even where the
            # checkpoint itself quantized it (adaln_proj-adjacent final_layer weights do, in the
            # non-pruned checkpoint; the pruned one stores final_layer.adaln_proj.linear dense
            # already, so this branch naturally falls to load_dense_linear for it instead).
            load_dequantized_dense_linear(holder, attr, comfy_prefix)
        else:
            load_dense_linear(holder, attr, comfy_prefix)
    for diffusers_name, comfy_prefix in _CONVROT_TOP_NORM_MAP.items():
        holder, attr = resolve(transformer, diffusers_name)
        load_dense_norm(getattr(holder, attr), comfy_prefix)
    # rope.inv_freq is a deterministic function of rope_freq_dim/rope_theta (both config, not
    # checkpoint data) in diffusers -- confirmed real source computes it fresh, not from a stored
    # buffer, so it's identical to the checkpoint's own buffer by construction and needs no load.

    if pruned:
        say("Loading adaln_t_table lookup-curve conditioning (this checkpoint's replacement for "
            "the full TimestepEmbedding MLP)...")
        table = take("adaln_t_table").to(device=device, dtype=torch.float32)
        transformer.time_proj = IdentityTimeProj()
        transformer.time_embedder = AdalnTableTimeEmbedder(table)

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
                # Kept dense either way (see function docstring) -- quantized in the non-pruned
                # checkpoint (needs dequantizing once), already dense in the pruned one (the
                # pruned file never quantizes its 8-dim adaln_proj at all, confirmed via its real
                # header: no "*.comfy_quant" marker for these keys).
                if has_quant_marker(comfy_full_prefix):
                    load_dequantized_dense_linear(holder, attr, comfy_full_prefix)
                else:
                    load_dense_linear(holder, attr, comfy_full_prefix)
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

    # Live run caught a real gap: init_empty_weights() only meta-ifies *parameters* by default
    # (accelerate's own include_buffers defaults to False), not buffers created via
    # register_buffer() -- so rope.inv_freq (populated with real, correct values by diffusers'
    # own __init__, never touched by this loader since it's a deterministic function of
    # rope_freq_dim/rope_theta config, not checkpoint data) was silently sitting on CPU the
    # whole time, not meta, so still_meta's own check never caught it either. Move any leftover
    # CPU-resident buffer to the real device generically here rather than special-casing
    # "rope.inv_freq" by name, in case another such buffer exists elsewhere in the model.
    for module in transformer.modules():
        for buf_name, buf in list(module.named_buffers(recurse=False)):
            if buf.device.type == "cpu":
                setattr(module, buf_name, buf.to(device))

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
    # Older checkpoints/loaders may still leave ConvRotInt8Linear's unused placeholder on meta;
    # real data lives in cr8_qdata/cr8_scale. Exempt only that identity while still catching any
    # genuinely unpopulated model tensor.
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
    """Inject the four native H3 targets in each main block.

    Diffusers splits Q/K/V, but the released H3 model and ComfyUI use one fused
    qkv_proj.  The custom adapter shares Q/K/V's down projection, preserving the
    requested rank exactly.  AdaLN and token_refiner are deliberately excluded.
    """
    from minimax_h3_lora import inject_native_minimax_h3_lora

    adapter = inject_native_minimax_h3_lora(transformer, rank=rank, alpha=alpha)
    say(
        f"Injected native MiniMax-H3 LoRA (rank={rank}, alpha={alpha}) into "
        f"{len(adapter.records)} modules: qkv_proj/out_proj/mlp.fc1/mlp.fc2 across 50 blocks."
    )
    return transformer, adapter


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def sample_shifted_sigma(shift: float, batch_size: int, device, sampling: str = "uniform") -> "torch.Tensor":
    """Sample a base sigma, then apply MiniMax-H3's exponential shift."""
    import torch

    if sampling == "uniform":
        base = torch.rand(batch_size, device=device)
    elif sampling == "logit_normal":
        base = torch.sigmoid(torch.randn(batch_size, device=device))
    else:
        raise ValueError(f"Unsupported timestep sampling mode: {sampling}")
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
            # Encode the prompt and empty prompt together, then strip padding back off each result.
            # They intentionally keep their natural token lengths; the training loop builds a
            # separate packed row layout for the empty-prompt guidance branch.
            text_ids = tokenizer(
                [full_caption, ""], return_tensors="pt", truncation=True, padding=True
            ).to(device)
            text_out = text_encoder.model(
                input_ids=text_ids["input_ids"], attention_mask=text_ids["attention_mask"],
                pixel_values=None, output_hidden_states=True,
            )
            text_hidden = text_out.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]
            text_embeds = text_hidden[0:1, text_ids["attention_mask"][0].bool()]
            empty_text_embeds = text_hidden[1:2, text_ids["attention_mask"][1].bool()]

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
                video_latents.cpu(), text_embeds.cpu(), empty_text_embeds.cpu(),
                reference_latents.cpu() if reference_latents is not None else None,
                audio_latents.cpu() if audio_latents is not None else None,
            ))
            say(f"Cached {i + 1}/{len(clips)}: {path.name}")
    return cache


def main() -> None:
    setup_environment()
    args = parse_args()
    if args.guidance_distillation_scale <= 0:
        raise ValueError("--guidance_distillation_scale must be positive.")
    if args.base_preservation_loss_weight < 0:
        raise ValueError("--base_preservation_loss_weight cannot be negative.")
    if args.lora_alpha <= 0 or args.rank <= 0:
        raise ValueError("--rank and --lora_alpha must be positive.")
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
        load_transformer_convrot(args, device)
        if args.quant_source in ("comfy_convrot", "comfy_convrot_pruned")
        else load_transformer(args, device)
    )
    transformer, lora_adapter = inject_lora(transformer, args.rank, args.lora_alpha)

    if args.resume_lora:
        from safetensors import safe_open
        from safetensors.torch import load_file

        resume_path = Path(args.resume_lora)
        with safe_open(str(resume_path), framework="pt", device="cpu") as resume_file:
            resume_metadata = resume_file.metadata() or {}
        resume_mode = resume_metadata.get("ss_h3_training_mode")
        expected_mode = args.partition.lower()
        if resume_mode and resume_mode.lower() != expected_mode:
            raise RuntimeError(
                f"Cannot resume {args.partition} training from a {resume_mode} LoRA: the partition "
                "weights are different even though their tensor shapes match."
            )
        resume_variant = resume_metadata.get("day0.base_variant")
        expected_variant = {
            "bitsandbytes": "base",
            "comfy_convrot": "convrot",
            "comfy_convrot_pruned": "convrot_pruned",
        }[args.quant_source]
        if resume_variant and resume_variant != expected_variant:
            warn(
                f"Resuming a {resume_variant} LoRA on {expected_variant}. Shapes are compatible, "
                "but the frozen base behavior differs; validate the result in ComfyUI."
            )
        lora_adapter.load_native_state_dict(load_file(str(resume_path), device="cpu"))
        say(f"Resumed native LoRA weights from {resume_path}.")

    trainable_params = list(lora_adapter.parameters())
    say(f"Trainable LoRA params: {lora_adapter.num_parameters:,}")
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
        video_latents, text_embeds, empty_text_embeds, reference_latents, audio_latents = cache[
            order[global_step % len(cache)]
        ]
        video_latents = video_latents.to(device)
        text_dtype = torch.bfloat16 if args.base_dtype == "bfloat16" else torch.float16
        text_embeds = text_embeds.to(device, dtype=text_dtype)
        empty_text_embeds = empty_text_embeds.to(device, dtype=text_dtype)
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
        empty_layout = None
        if args.guidance_distillation_scale != 1.0:
            empty_text_token_tags = torch.ones(empty_text_embeds.shape[1], dtype=torch.long)
            empty_layout = build_packed_sequence(
                text_token_tags=empty_text_token_tags,
                num_latent_frames=num_latent_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                num_audio_latents=num_audio_latents,
                patch_size=patch_size,
                keyframe_anchors=("first",) if has_reference else (),
            )

        noise = torch.randn_like(video_latents)
        video_sigma = sample_shifted_sigma(
            args.video_shift, batch_size, device, sampling=args.timestep_sampling
        )
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
            audio_sigma = sample_shifted_sigma(
                args.audio_shift, batch_size, device, sampling=args.timestep_sampling
            )
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
        empty_timesteps = empty_timestep_indices = None
        if empty_layout is not None:
            empty_timesteps, empty_timestep_indices = build_row_timesteps(
                empty_layout, video_timestep=video_t, audio_timestep=audio_t,
                condition_video_timestep=condition_video_t, condition_audio_timestep=audio_t,
            )

        model_inputs = {
            "hidden_states": noisy_video_rows,
            "audio_hidden_states": noisy_audio_rows,
            "timestep": timesteps.to(device),
            "timestep_indices": timestep_indices.to(device),
            "token_tags": layout.token_tags.to(device),
            "position_ids": layout.position_ids.to(device),
            "video_indices": layout.video_indices.to(device),
            "audio_indices": layout.audio_indices.to(device),
            "text_indices": layout.text_indices.to(device),
        }

        empty_model_inputs = None
        if empty_layout is not None:
            empty_model_inputs = {
                "hidden_states": noisy_video_rows,
                "audio_hidden_states": noisy_audio_rows,
                "timestep": empty_timesteps.to(device),
                "timestep_indices": empty_timestep_indices.to(device),
                "token_tags": empty_layout.token_tags.to(device),
                "position_ids": empty_layout.position_ids.to(device),
                "video_indices": empty_layout.video_indices.to(device),
                "audio_indices": empty_layout.audio_indices.to(device),
                "text_indices": empty_layout.text_indices.to(device),
            }

        def predict(encoder_hidden_states, inputs=model_inputs):
            return transformer(encoder_hidden_states=encoder_hidden_states, **inputs)

        # Run the two frozen branches before the trainable prompted branch. That way their forward
        # passes do not overlap with the prompted branch's retained backward activations, keeping
        # the corrected three-forward objective as memory-safe as possible.
        empty_video_prediction = empty_audio_prediction = None
        if args.guidance_distillation_scale != 1.0:
            with torch.no_grad():
                empty_output = predict(empty_text_embeds, empty_model_inputs)
                empty_video_prediction = empty_output.sample[:, empty_layout.num_condition_video_rows:].detach()
                empty_audio_prediction = empty_output.audio_sample.detach() if has_audio else None
                del empty_output

        base_video_prediction = base_audio_prediction = None
        if args.base_preservation_loss_weight > 0:
            with torch.no_grad(), lora_adapter.disabled():
                base_output = predict(text_embeds)
                base_video_prediction = base_output.sample[:, layout.num_condition_video_rows:].detach()
                base_audio_prediction = base_output.audio_sample.detach() if has_audio else None
                del base_output

        output = predict(text_embeds)
        prompted_video_prediction = output.sample[:, layout.num_condition_video_rows:]

        from minimax_h3_lora import guidance_consistent_prediction

        guided_video_prediction = guidance_consistent_prediction(
            prompted_video_prediction,
            empty_video_prediction if empty_video_prediction is not None else prompted_video_prediction,
            args.guidance_distillation_scale,
        )
        loss = torch.nn.functional.mse_loss(guided_video_prediction.float(), target_video_rows.float())

        preservation_loss_value = None
        if base_video_prediction is not None:
            preservation_loss = torch.nn.functional.mse_loss(
                prompted_video_prediction.float(), base_video_prediction.float()
            )
            preservation_loss_value = preservation_loss.item()
            loss = loss + args.base_preservation_loss_weight * preservation_loss

        audio_loss_value = None
        if has_audio:
            # Audio uses the same guidance/base-preservation recipe with its independent sigma.
            guided_audio_prediction = guidance_consistent_prediction(
                output.audio_sample,
                empty_audio_prediction if empty_audio_prediction is not None else output.audio_sample,
                args.guidance_distillation_scale,
            )
            audio_loss = torch.nn.functional.mse_loss(
                guided_audio_prediction.float(), target_audio_rows.float()
            )
            if base_audio_prediction is not None:
                audio_preservation_loss = torch.nn.functional.mse_loss(
                    output.audio_sample.float(), base_audio_prediction.float()
                )
                audio_loss = audio_loss + args.base_preservation_loss_weight * audio_preservation_loss
            audio_loss_value = audio_loss.item()
            loss = loss + args.audio_loss_weight * audio_loss

        loss.backward()
        # Match the 1.0 gradient norm used by the repository's other trainers. Batch size is one,
        # so clipping remains useful protection against a single outlier clip.
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
        if preservation_loss_value is not None:
            metrics_row["preservation_loss"] = round(preservation_loss_value, 6)
        with open(metrics_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics_row) + "\n")
        # Every step for the first 10 (fast feedback on real per-step speed without waiting on a
        # 10-step average), then every 10th after that to keep the log from getting noisy.
        if global_step <= 10 or global_step % 10 == 0:
            audio_suffix = f" audio_loss={audio_loss_value:.4f}" if audio_loss_value is not None else ""
            preservation_suffix = (
                f" preserve={preservation_loss_value:.4f}" if preservation_loss_value is not None else ""
            )
            say(
                f"step {global_step}/{args.max_train_steps} loss={loss.item():.4f}"
                f"{audio_suffix}{preservation_suffix} ({step_seconds:.1f}s/step)"
            )

        if global_step % args.save_every_n_steps == 0 or global_step == args.max_train_steps:
            import safetensors.torch

            ckpt_dir = checkpoints_dir / f"step-{global_step:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_dtype = {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }[args.save_dtype]
            lora_state_dict = lora_adapter.native_state_dict(dtype=save_dtype)
            if len(lora_state_dict) != 600:
                raise RuntimeError(
                    f"Native H3 checkpoint must contain exactly 600 tensors, got {len(lora_state_dict)}."
                )
            metadata = {
                "format": "pt",
                "modelspec.architecture": "MiniMax-H3/lora",
                "modelspec.title": f"{args.run_name} MiniMax-H3 LoRA",
                "modelspec.description": "Day0 native-rank MiniMax-H3 LoRA",
                "ss_base_model_version": str(args.pretrained_model_name_or_path),
                "ss_h3_training_mode": args.partition.lower(),
                "ss_h3_video_shift": str(args.video_shift),
                "ss_h3_audio_shift": str(args.audio_shift),
                "ss_h3_audio_loss_weight": str(args.audio_loss_weight),
                "ss_network_dim": str(args.rank),
                "ss_network_alpha": str(args.lora_alpha),
                "ss_learning_rate": str(args.learning_rate),
                "ss_lr_scheduler": args.lr_scheduler,
                "ss_lr_warmup_steps": str(args.lr_warmup_steps),
                "ss_optimizer": "AdamW",
                "ss_timestep_sampling": args.timestep_sampling,
                "ss_guidance_distillation_scale": str(args.guidance_distillation_scale),
                "ss_guidance_distillation_normalized": "True",
                "ss_base_model_preservation_loss": str(args.base_preservation_loss_weight),
                "ss_steps": str(global_step),
                "ss_training_finished_at": str(int(time.time())),
                "day0.partition": args.partition,
                "day0.base_variant": {
                    "bitsandbytes": "base",
                    "comfy_convrot": "convrot",
                    "comfy_convrot_pruned": "convrot_pruned",
                }[args.quant_source],
                "day0.quant_source": args.quant_source,
                "day0.save_dtype": args.save_dtype,
                "day0.target_modules": "qkv_proj,out_proj,mlp.fc1,mlp.fc2",
            }
            safetensors.torch.save_file(
                lora_state_dict,
                str(ckpt_dir / "minimax_h3_lora.safetensors"),
                metadata=metadata,
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
