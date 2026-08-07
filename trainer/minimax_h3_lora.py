"""Native-compatible LoRA plumbing for MiniMax-H3.

MiniMax-H3's released transformer has one fused QKV projection per main block,
while the Diffusers port exposes three separate Q/K/V linears.  A normal PEFT
adapter therefore learns three unrelated down projections and can only be
exported as an inflated rank-3r fused adapter.  This module keeps one shared
down projection for Q/K/V from the beginning, so the learned adapter has the
same topology and rank as the native MiniMax-H3 checkpoint.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


NATIVE_TARGETS_PER_BLOCK = 4
EXPECTED_TRANSFORMER_BLOCKS = 50


def _module_device(module: nn.Module) -> torch.device:
    """Find a real device even for quantized modules with placeholder weights."""
    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        if tensor.device.type != "meta":
            return tensor.device
    raise RuntimeError(f"Cannot determine a real device for LoRA target {type(module).__name__}.")


def _resolve(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    holder: nn.Module = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        holder = getattr(holder, part)
    return holder, parts[-1]


def _features(module: nn.Module) -> tuple[int, int]:
    try:
        return int(module.in_features), int(module.out_features)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"MiniMax-H3 LoRA target {type(module).__name__} does not expose in_features/out_features."
        ) from exc


class NativeLoRALinear(nn.Module):
    """Frozen base linear plus a native MiniMax-H3 LoRA residual."""

    def __init__(
        self,
        base_layer: nn.Module,
        lora_down: nn.Linear,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        in_features, out_features = _features(base_layer)
        if lora_down.in_features != in_features or lora_down.out_features != rank:
            raise ValueError("Shared LoRA down projection has incompatible dimensions.")

        self.base_layer = base_layer
        self.lora_down = lora_down
        self.lora_up = nn.Linear(
            rank,
            out_features,
            bias=False,
            device=_module_device(base_layer),
            dtype=torch.float32,
        )
        nn.init.zeros_(self.lora_up.weight)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.enabled = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(hidden_states)
        if not self.enabled:
            return base
        adapter_input = hidden_states.to(dtype=self.lora_down.weight.dtype)
        delta = self.lora_up(self.lora_down(adapter_input)) * self.scaling
        return base + delta.to(dtype=base.dtype)


@dataclass
class NativeLoRARecord:
    native_name: str
    down: nn.Linear
    ups: tuple[nn.Linear, ...]


class MiniMaxH3NativeLoRA:
    """Owns the injected wrappers and exports the exact native H3 LoRA layout."""

    def __init__(self, rank: int, alpha: float) -> None:
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive.")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.records: list[NativeLoRARecord] = []
        self.wrappers: list[NativeLoRALinear] = []

    def _new_down(self, base_layer: nn.Module) -> nn.Linear:
        in_features, _ = _features(base_layer)
        down = nn.Linear(
            in_features,
            self.rank,
            bias=False,
            device=_module_device(base_layer),
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(down.weight, a=5**0.5)
        return down

    def _wrap(self, holder: nn.Module, attr: str, down: nn.Linear) -> NativeLoRALinear:
        base_layer = getattr(holder, attr)
        wrapper = NativeLoRALinear(base_layer, down, self.rank, self.alpha)
        setattr(holder, attr, wrapper)
        self.wrappers.append(wrapper)
        return wrapper

    def inject(self, transformer: nn.Module) -> "MiniMaxH3NativeLoRA":
        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is None:
            raise RuntimeError("MiniMax-H3 transformer has no transformer_blocks module list.")
        if len(blocks) != EXPECTED_TRANSFORMER_BLOCKS:
            raise RuntimeError(
                f"Expected {EXPECTED_TRANSFORMER_BLOCKS} MiniMax-H3 transformer blocks, found {len(blocks)}. "
                "The upstream architecture changed; refusing to create a silently incompatible LoRA."
            )

        for index, block in enumerate(blocks):
            q_holder, q_attr = _resolve(block, "attn.to_q")
            q_base = getattr(q_holder, q_attr)
            shared_qkv_down = self._new_down(q_base)
            q = self._wrap(q_holder, q_attr, shared_qkv_down)
            k_holder, k_attr = _resolve(block, "attn.to_k")
            v_holder, v_attr = _resolve(block, "attn.to_v")
            k = self._wrap(k_holder, k_attr, shared_qkv_down)
            v = self._wrap(v_holder, v_attr, shared_qkv_down)
            self.records.append(
                NativeLoRARecord(
                    native_name=f"blocks.{index}.attn.qkv_proj",
                    down=shared_qkv_down,
                    ups=(q.lora_up, k.lora_up, v.lora_up),
                )
            )

            for diffusers_path, native_path in (
                ("attn.to_out.0", "attn.out_proj"),
                ("ff.net.0.proj", "mlp.fc1"),
                ("ff.net.2", "mlp.fc2"),
            ):
                holder, attr = _resolve(block, diffusers_path)
                down = self._new_down(getattr(holder, attr))
                wrapped = self._wrap(holder, attr, down)
                self.records.append(
                    NativeLoRARecord(
                        native_name=f"blocks.{index}.{native_path}",
                        down=down,
                        ups=(wrapped.lora_up,),
                    )
                )

        expected = EXPECTED_TRANSFORMER_BLOCKS * NATIVE_TARGETS_PER_BLOCK
        if len(self.records) != expected:
            raise RuntimeError(f"Expected {expected} native H3 LoRA modules, created {len(self.records)}.")
        return self

    def parameters(self) -> Iterable[nn.Parameter]:
        """Yield each shared parameter once."""
        seen: set[int] = set()
        for record in self.records:
            for parameter in list(record.down.parameters()) + [p for up in record.ups for p in up.parameters()]:
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @contextmanager
    def disabled(self):
        previous = [wrapper.enabled for wrapper in self.wrappers]
        try:
            for wrapper in self.wrappers:
                wrapper.enabled = False
            yield
        finally:
            for wrapper, enabled in zip(self.wrappers, previous):
                wrapper.enabled = enabled

    def native_state_dict(self, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
        """Return Kohya/ComfyUI native H3 keys: 200 modules x down/up/alpha."""
        state: dict[str, torch.Tensor] = {}
        for record in self.records:
            prefix = "lora_unet_" + record.native_name.replace(".", "_")
            up = record.ups[0].weight if len(record.ups) == 1 else torch.cat(
                [module.weight for module in record.ups], dim=0
            )
            state[f"{prefix}.lora_down.weight"] = record.down.weight.detach().to("cpu", dtype=dtype).contiguous()
            state[f"{prefix}.lora_up.weight"] = up.detach().to("cpu", dtype=dtype).contiguous()
            state[f"{prefix}.alpha"] = torch.tensor(self.alpha, dtype=torch.float32)
        return state

    def load_native_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected_keys: set[str] = set()
        for record in self.records:
            prefix = "lora_unet_" + record.native_name.replace(".", "_")
            down_key = f"{prefix}.lora_down.weight"
            up_key = f"{prefix}.lora_up.weight"
            alpha_key = f"{prefix}.alpha"
            expected_keys.update((down_key, up_key, alpha_key))
            if down_key not in state or up_key not in state:
                raise RuntimeError(f"Resume LoRA is missing {down_key} or {up_key}.")
            down = state[down_key]
            up = state[up_key]
            if tuple(down.shape) != tuple(record.down.weight.shape):
                raise RuntimeError(f"Resume shape mismatch for {down_key}: {tuple(down.shape)}.")
            expected_up_rows = sum(module.out_features for module in record.ups)
            if tuple(up.shape) != (expected_up_rows, self.rank):
                raise RuntimeError(f"Resume shape mismatch for {up_key}: {tuple(up.shape)}.")
            record.down.weight.data.copy_(down.to(record.down.weight))
            offset = 0
            for module in record.ups:
                rows = module.out_features
                module.weight.data.copy_(up[offset:offset + rows].to(module.weight))
                offset += rows
            if alpha_key in state and float(state[alpha_key].item()) != self.alpha:
                raise RuntimeError(
                    f"Resume alpha mismatch for {alpha_key}: file has {float(state[alpha_key].item())}, "
                    f"trainer expects {self.alpha}."
                )
        unexpected = set(state) - expected_keys
        if unexpected:
            raise RuntimeError(f"Resume LoRA has unexpected tensors: {sorted(unexpected)[:10]}.")


def inject_native_minimax_h3_lora(
    transformer: nn.Module, rank: int, alpha: float
) -> MiniMaxH3NativeLoRA:
    return MiniMaxH3NativeLoRA(rank=rank, alpha=alpha).inject(transformer)


def guidance_consistent_prediction(
    prompted_prediction: torch.Tensor,
    empty_prediction: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    if guidance_scale <= 0:
        raise ValueError("Guidance distillation scale must be positive.")
    if guidance_scale == 1:
        return prompted_prediction
    return (prompted_prediction + (guidance_scale - 1.0) * empty_prediction) / guidance_scale
