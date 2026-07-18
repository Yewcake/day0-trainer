"""Ported from ostris/ai-toolkit (MIT License), toolkit/optimizers/optimizer_utils.py.
Trimmed of the optimum-quanto quantized-tensor path — this trainer never
produces QBytesTensor params, so that branch is dead code here.
"""

from typing import Optional

import torch
from torch import Tensor


def get_format_params(dtype: torch.dtype) -> tuple[int, int]:
    """Returns (mantissa_bits, total_bits) for each format. mantissa_bits excludes the implicit leading 1."""
    if dtype == torch.float32:
        return 23, 32
    elif dtype == torch.bfloat16:
        return 7, 16
    elif dtype == torch.float16:
        return 10, 16
    elif dtype == torch.float8_e4m3fn:
        return 3, 8
    elif dtype == torch.float8_e5m2:
        return 2, 8
    elif dtype == torch.int8:
        return 0, 8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def copy_stochastic_bf16(target: torch.Tensor, source: torch.Tensor):
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)
    target.copy_(result.view(dtype=torch.float32))
    del result


def copy_stochastic(target: torch.Tensor, source: torch.Tensor, eps: Optional[float] = None) -> None:
    with torch.no_grad():
        assert target.device.type != "cpu", "Target is on cpu!"
        assert source.device.type != "cpu", "Source is on cpu!"

        if target.dtype == torch.float32:
            target.copy_(source)
            return
        if target.dtype == torch.bfloat16:
            copy_stochastic_bf16(target, source)
            return

        mantissa_bits, _ = get_format_params(target.dtype)
        round_factor = 2 ** (23 - mantissa_bits)

        noise = torch.rand_like(source, device=source.device) - 0.5
        rounded = torch.round(source * round_factor + noise)
        result_float = rounded / round_factor

        if target.dtype == torch.float8_e4m3fn:
            result_float.clamp_(-448.0, 448.0)
        elif target.dtype == torch.float8_e5m2:
            result_float.clamp_(-57344.0, 57344.0)

        target.copy_(result_float)


class Auto8bitTensor:
    def __init__(self, data: Tensor, *args, **kwargs):
        if isinstance(data, dict):
            self._load_from_state_dict(data)
        else:
            abs_max = data.abs().max().item()
            scale = abs_max / 127.0 if abs_max > 0 else 1.0

            self.quantized = (data / scale).round().clamp(-127, 127).to(torch.int8)
            self.scale = scale
            self.orig_dtype = data.dtype

    def dequantize(self) -> Tensor:
        return self.quantized.to(dtype=torch.float32) * self.scale

    def to(self, *args, **kwargs):
        dtype = None
        if args and isinstance(args[0], torch.dtype):
            dtype = args[0]
            args = args[1:]
        elif "dtype" in kwargs:
            dtype = kwargs["dtype"]
            del kwargs["dtype"]

        if dtype is not None:
            return self.dequantize().to(dtype=dtype, *args, **kwargs)

        return self.dequantize().to(*args, **kwargs)

    def state_dict(self):
        return {"quantized": self.quantized, "scale": self.scale, "orig_dtype": self.orig_dtype}

    def _load_from_state_dict(self, state_dict):
        self.quantized = state_dict["quantized"]
        self.scale = state_dict["scale"]
        self.orig_dtype = state_dict["orig_dtype"]

    def __str__(self):
        return f"Auto8bitTensor({self.dequantize()})"


def stochastic_grad_accummulation(param):
    if hasattr(param, "_accum_grad"):
        grad_fp32 = param._accum_grad.clone().to(torch.float32)
        grad_fp32.add_(param.grad.to(torch.float32))
        copy_stochastic(param._accum_grad, grad_fp32)
        del grad_fp32
        del param.grad
    else:
        param._accum_grad = param.grad.clone()
        del param.grad
