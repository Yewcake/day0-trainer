from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "trainer"))

from minimax_h3_lora import (  # noqa: E402
    guidance_consistent_prediction,
    inject_native_minimax_h3_lora,
)
from train_minimax_h3 import sample_shifted_sigma  # noqa: E402


class FakeSwiGLU(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width * 2, bias=False)


class FakeAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(width, width, bias=False), nn.Identity()])


class FakeFeedForward(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.ModuleList([FakeSwiGLU(width), nn.Identity(), nn.Linear(width * 2, width, bias=False)])


class FakeBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn = FakeAttention(width)
        self.ff = FakeFeedForward(width)
        self.adaln_proj = nn.Module()
        self.adaln_proj.linear = nn.Linear(3, width * 6, bias=False)


class FakeTokenRefiner(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList([FakeBlock(width)])


class FakeTransformer(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([FakeBlock(width) for _ in range(50)])
        self.token_refiner = FakeTokenRefiner(width)


class MiniMaxH3NativeLoRATests(unittest.TestCase):
    def test_exact_target_and_export_contract(self) -> None:
        transformer = FakeTransformer()
        adaln_before = transformer.transformer_blocks[0].adaln_proj.linear
        refiner_q_before = transformer.token_refiner.refiner_blocks[0].attn.to_q

        adapter = inject_native_minimax_h3_lora(transformer, rank=2, alpha=1)

        self.assertEqual(len(adapter.records), 200)
        self.assertIs(transformer.transformer_blocks[0].adaln_proj.linear, adaln_before)
        self.assertIs(transformer.token_refiner.refiner_blocks[0].attn.to_q, refiner_q_before)

        block = transformer.transformer_blocks[0]
        self.assertIs(block.attn.to_q.lora_down, block.attn.to_k.lora_down)
        self.assertIs(block.attn.to_q.lora_down, block.attn.to_v.lora_down)

        state = adapter.native_state_dict(dtype=torch.float32)
        self.assertEqual(len(state), 600)
        self.assertFalse(any("adaln" in key or "token_refiner" in key for key in state))
        self.assertIn("lora_unet_blocks_0_attn_qkv_proj.lora_down.weight", state)
        self.assertIn("lora_unet_blocks_49_mlp_fc2.alpha", state)
        self.assertTrue(all(not torch.count_nonzero(record.ups[0].weight) for record in adapter.records))

    def test_shared_qkv_matches_one_native_fused_adapter(self) -> None:
        torch.manual_seed(7)
        transformer = FakeTransformer()
        block = transformer.transformer_blocks[0]
        for linear in (block.attn.to_q, block.attn.to_k, block.attn.to_v):
            nn.init.zeros_(linear.weight)
        adapter = inject_native_minimax_h3_lora(transformer, rank=3, alpha=1.5)
        record = adapter.records[0]
        record.down.weight.data.normal_()
        for up in record.ups:
            up.weight.data.normal_()

        hidden = torch.randn(2, 5, 4)
        actual = torch.cat(
            [block.attn.to_q(hidden), block.attn.to_k(hidden), block.attn.to_v(hidden)], dim=-1
        )
        fused_up = torch.cat([up.weight for up in record.ups], dim=0)
        expected = torch.nn.functional.linear(
            torch.nn.functional.linear(hidden, record.down.weight), fused_up
        ) * (adapter.alpha / adapter.rank)
        torch.testing.assert_close(actual, expected)

    def test_disable_and_round_trip(self) -> None:
        torch.manual_seed(11)
        first = FakeTransformer()
        adapter = inject_native_minimax_h3_lora(first, rank=2, alpha=1)
        for parameter in adapter.parameters():
            parameter.data.normal_()
        hidden = torch.randn(1, 3, 4)
        active = first.transformer_blocks[0].attn.to_q(hidden)
        with adapter.disabled():
            disabled = first.transformer_blocks[0].attn.to_q(hidden)
            base = first.transformer_blocks[0].attn.to_q.base_layer(hidden)
        torch.testing.assert_close(disabled, base)
        self.assertFalse(torch.allclose(active, base))

        state = adapter.native_state_dict(dtype=torch.float32)
        second = FakeTransformer()
        second_adapter = inject_native_minimax_h3_lora(second, rank=2, alpha=1)
        # Copy the frozen target used by this output comparison as well as the adapter.
        second.transformer_blocks[0].attn.to_q.base_layer.weight.data.copy_(
            first.transformer_blocks[0].attn.to_q.base_layer.weight.data
        )
        second_adapter.load_native_state_dict(state)
        torch.testing.assert_close(
            first.transformer_blocks[0].attn.to_q(hidden),
            second.transformer_blocks[0].attn.to_q(hidden),
        )

    def test_guidance_formula_and_uniform_sigma(self) -> None:
        prompted = torch.tensor([3.0])
        empty = torch.tensor([1.0])
        torch.testing.assert_close(guidance_consistent_prediction(prompted, empty, 3), torch.tensor([5 / 3]))

        torch.manual_seed(123)
        sigmas = sample_shifted_sigma(12.0, 10000, torch.device("cpu"), sampling="uniform")
        self.assertTrue(torch.all(sigmas > 0))
        self.assertTrue(torch.all(sigmas < 1))


if __name__ == "__main__":
    unittest.main()
