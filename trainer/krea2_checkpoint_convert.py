#!/usr/bin/env python3
"""
Converts a native (ComfyUI/Civitai-style) Krea2 transformer checkpoint into a
Diffusers-format model folder, by grafting its weights onto a known-good
scaffold repo's text_encoder/tokenizer/vae/scheduler/config.

Native Krea2 checkpoints store the transformer under keys like
`model.diffusion_model.blocks.0.attn.wq.weight`; Diffusers' Krea2Transformer2DModel
state dict uses `transformer_blocks.0.attn.to_q.weight`. The mapping below was
derived by reading the installed Krea2Transformer2DModel module tree
(diffusers/models/transformers/transformer_krea2.py) side by side with a real
Civitai/HF native checkpoint's safetensors header, then verified offline to
cover all 430 parameters of the default Krea2 config with matching shapes
(exact key-set equality both ways, zero shape mismatches).

Only the transformer is replaced here -- text_encoder/tokenizer/vae/scheduler
come from `scaffold_repo`, a known-good ungated Diffusers mirror already used
elsewhere in this project (Train_Krea2_Direct_Diffusers.sh) as the community
fallback.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

import torch
from safetensors.torch import load_file, save_file

DEFAULT_SCAFFOLD_REPO = "CalamitousFelicitousness/Krea-2-Base-Diffusers"

_NATIVE_PREFIX = "model.diffusion_model."

# Suffix mapping shared by both the 28 main transformer blocks
# (`blocks.N.*`) and the 4 text-fusion blocks (`txtfusion.{layerwise,refiner}_blocks.N.*`).
_BLOCK_SUFFIX_MAP = {
    "prenorm.scale": "norm1.weight",
    "postnorm.scale": "norm2.weight",
    "attn.wq.weight": "attn.to_q.weight",
    "attn.wk.weight": "attn.to_k.weight",
    "attn.wv.weight": "attn.to_v.weight",
    "attn.wo.weight": "attn.to_out.0.weight",
    "attn.gate.weight": "attn.to_gate.weight",
    "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
    "attn.qknorm.knorm.scale": "attn.norm_k.weight",
    "mlp.gate.weight": "ff.gate.weight",
    "mlp.up.weight": "ff.up.weight",
    "mlp.down.weight": "ff.down.weight",
}

_TOP_LEVEL_MAP = {
    "first.weight": "img_in.weight",
    "first.bias": "img_in.bias",
    "tmlp.0.weight": "time_embed.linear_1.weight",
    "tmlp.0.bias": "time_embed.linear_1.bias",
    "tmlp.2.weight": "time_embed.linear_2.weight",
    "tmlp.2.bias": "time_embed.linear_2.bias",
    "tproj.1.weight": "time_mod_proj.weight",
    "tproj.1.bias": "time_mod_proj.bias",
    "txtmlp.0.scale": "txt_in.norm.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight",
    "txtmlp.1.bias": "txt_in.linear_1.bias",
    "txtmlp.3.weight": "txt_in.linear_2.weight",
    "txtmlp.3.bias": "txt_in.linear_2.bias",
    "last.linear.weight": "final_layer.linear.weight",
    "last.linear.bias": "final_layer.linear.bias",
    "last.modulation.lin": "final_layer.scale_shift_table",
    "last.norm.scale": "final_layer.norm.weight",
    "txtfusion.projector.weight": "text_fusion.projector.weight",
}

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
_TEXT_FUSION_RE = re.compile(r"^txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(.+)$")


def native_key_to_diffusers(native_key: str) -> str | None:
    """Map one native Krea2 checkpoint key to its Diffusers Krea2Transformer2DModel
    equivalent. Returns None for keys this table doesn't recognize (e.g. a bundled
    VAE/text-encoder sharing the same file, or an unexpected checkpoint schema)."""
    if not native_key.startswith(_NATIVE_PREFIX):
        return None
    rest = native_key[len(_NATIVE_PREFIX):]
    if rest in _TOP_LEVEL_MAP:
        return _TOP_LEVEL_MAP[rest]
    m = _BLOCK_RE.match(rest)
    if m:
        idx, suffix = m.group(1), m.group(2)
        if suffix == "mod.lin":
            return f"transformer_blocks.{idx}.scale_shift_table"
        if suffix in _BLOCK_SUFFIX_MAP:
            return f"transformer_blocks.{idx}.{_BLOCK_SUFFIX_MAP[suffix]}"
        return None
    m = _TEXT_FUSION_RE.match(rest)
    if m:
        kind, idx, suffix = m.group(1), m.group(2), m.group(3)
        if suffix in _BLOCK_SUFFIX_MAP:
            return f"text_fusion.{kind}.{idx}.{_BLOCK_SUFFIX_MAP[suffix]}"
        return None
    return None


def _is_url(value: str) -> bool:
    return urlparse(value).scheme in ("http", "https")


def _slug_for(source: str) -> str:
    name = Path(urlparse(source).path if _is_url(source) else source).name
    name = re.sub(r"\.safetensors$", "", name)
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "custom_checkpoint"


_HF_RESOLVE_RE = re.compile(
    r"^https?://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<path>.+)$"
)


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)

    hf_match = _HF_RESOLVE_RE.match(url)
    if hf_match:
        # Route HF-hosted files through hf_hub_download rather than a raw stream: it's
        # resumable, shares the same HF_HOME cache as everything else this trainer
        # downloads, and picks up hf_transfer automatically (already installed/enabled
        # in this image) for multi-connection speed without shelling out to a separate
        # download tool.
        from huggingface_hub import hf_hub_download

        print(f"Downloading native checkpoint from Hugging Face: {hf_match.group('repo')}")
        downloaded = hf_hub_download(
            hf_match.group("repo"),
            hf_match.group("path"),
            revision=hf_match.group("rev"),
        )
        _link_or_copy(Path(downloaded), dest)
        return

    print(f"Downloading native checkpoint: {url}")
    tmp = dest.with_name(dest.name + ".part")
    try:
        import requests

        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download native checkpoint from {url}: {exc}. The URL must be "
            "directly downloadable without a login/paywall (a Civitai API-key-gated link "
            "will not work here -- use the model's Hugging Face mirror if it has one)."
        ) from exc
    tmp.rename(dest)


def _snapshot_scaffold(scaffold_repo: str, cache_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    scaffold_dir = cache_root / "_scaffold" / scaffold_repo.replace("/", "__")
    if (scaffold_dir / "model_index.json").exists():
        return scaffold_dir
    print(f"Fetching Diffusers scaffold (text encoder/tokenizer/vae/scheduler): {scaffold_repo}")
    snapshot_download(
        scaffold_repo,
        local_dir=scaffold_dir,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "vae/*",
            "transformer/config.json",
        ],
    )
    return scaffold_dir


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def convert_native_checkpoint_to_diffusers(
    checkpoint_source: str,
    cache_root: Path,
    scaffold_repo: str = DEFAULT_SCAFFOLD_REPO,
) -> str:
    """Build (or reuse) a Diffusers-format Krea2 model folder around a native
    checkpoint. Returns a local folder path usable directly as
    `--pretrained_model_name_or_path`.

    Idempotent: a previously converted folder for the same source is reused as-is.
    """
    cache_root = Path(cache_root)
    slug = _slug_for(checkpoint_source)
    out_dir = cache_root / f"converted_{slug}"
    marker = out_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    if marker.exists() and (out_dir / "model_index.json").exists():
        print(f"Reusing already-converted checkpoint at {out_dir}")
        return str(out_dir)

    scaffold_dir = _snapshot_scaffold(scaffold_repo, cache_root)

    if _is_url(checkpoint_source):
        native_path = cache_root / "_downloads" / f"{slug}.safetensors"
        _download(checkpoint_source, native_path)
    else:
        native_path = Path(checkpoint_source)
        if not native_path.is_file():
            raise RuntimeError(f"Native checkpoint not found: {native_path}")

    print(f"Loading native checkpoint: {native_path}")
    native_state = load_file(str(native_path))

    remapped: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, tensor in native_state.items():
        diffusers_key = native_key_to_diffusers(key)
        if diffusers_key is None:
            skipped.append(key)
            continue
        remapped[diffusers_key] = tensor
    if skipped:
        print(
            f"Note: {len(skipped)} native key(s) did not match the Krea2 transformer schema "
            f"and were skipped (expected if the file also bundles a VAE/text encoder). "
            f"First few: {skipped[:5]}"
        )

    # Verify full, exact coverage against the real Diffusers module for this scaffold's
    # own transformer config before ever writing anything to disk -- a partial or
    # incorrect mapping must fail loudly here, never produce a silently broken model.
    # (train_krea2_lora_direct.py's own from_pretrained() call also re-checks this via
    # its "weights not used/newly initialized" and meta-tensor guards, so this is a
    # belt-and-suspenders check, not the only line of defense.)
    import inspect

    from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel

    # Filter against the *installed* class's real constructor signature rather than just
    # dropping "_"-prefixed keys: the scaffold repo's saved config.json can drift from
    # whatever diffusers commit this image's `pip install git+.../diffusers.git` pulled
    # (Krea2 support is still moving upstream), and a stale/renamed field there should
    # never hard-fail the conversion.
    raw_config = json.loads((scaffold_dir / "transformer" / "config.json").read_text())
    valid_params = set(inspect.signature(Krea2Transformer2DModel.__init__).parameters) - {"self"}
    transformer_config = {k: v for k, v in raw_config.items() if k in valid_params}
    with torch.device("meta"):
        expected_model = Krea2Transformer2DModel(**transformer_config)
    expected_state = expected_model.state_dict()
    expected_keys = set(expected_state.keys())
    got_keys = set(remapped.keys())
    missing = expected_keys - got_keys
    extra = got_keys - expected_keys
    if missing or extra:
        raise RuntimeError(
            "Checkpoint conversion produced an incomplete/incorrect key mapping -- refusing to "
            f"write a broken model. Missing {len(missing)} expected key(s) "
            f"(e.g. {sorted(missing)[:5]}), {len(extra)} unexpected key(s) "
            f"(e.g. {sorted(extra)[:5]}). This checkpoint's internal layout may differ from the "
            "standard Krea2 native schema this converter supports."
        )

    for key, expected_tensor in expected_state.items():
        actual = remapped[key]
        if tuple(actual.shape) == tuple(expected_tensor.shape):
            continue
        if actual.numel() != expected_tensor.numel():
            raise RuntimeError(
                f"Shape mismatch converting '{key}': got {tuple(actual.shape)} "
                f"({actual.numel()} elements), expected {tuple(expected_tensor.shape)} "
                f"({expected_tensor.numel()} elements). Refusing to write a broken model."
            )
        # Same element count, different shape (e.g. the per-block modulation table is
        # stored flat as [6*hidden] natively but Diffusers keeps it as [6, hidden]).
        remapped[key] = actual.reshape(expected_tensor.shape)

    for name in ("model_index.json",):
        _link_or_copy(scaffold_dir / name, out_dir / name)
    for folder in ("scheduler", "text_encoder", "tokenizer", "vae"):
        src_folder = scaffold_dir / folder
        if not src_folder.is_dir():
            continue
        for src_file in src_folder.iterdir():
            _link_or_copy(src_file, out_dir / folder / src_file.name)
    _link_or_copy(scaffold_dir / "transformer" / "config.json", out_dir / "transformer" / "config.json")

    marker.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing converted transformer ({len(remapped)} tensors): {marker}")
    save_file(remapped, str(marker))

    print(f"Converted checkpoint ready at {out_dir}")
    return str(out_dir)
