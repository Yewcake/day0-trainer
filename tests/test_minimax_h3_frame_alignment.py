from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "trainer"))

from train_minimax_h3 import align_num_frames_down, MINIMAX_H3_LATENTS_PER_CHUNK  # noqa: E402


class AlignNumFramesDownTests(unittest.TestCase):
    def test_exact_grid_points_pass_through(self) -> None:
        for n in (5, 22, 39, 56, 73, 90, 107, 124, 141):
            self.assertEqual(align_num_frames_down(n), n)

    def test_rounds_down_never_up(self) -> None:
        # 120 frames (5s @ 24fps) must resolve to 107, matching the real MiniMax-H3 VAE grid --
        # not 124 (rounding up would ask for 4 more frames than a 120-frame source has).
        self.assertEqual(align_num_frames_down(120), 107)
        self.assertLessEqual(align_num_frames_down(120), 120)

    def test_never_exceeds_input(self) -> None:
        for n in range(5, 200):
            self.assertLessEqual(align_num_frames_down(n), n)

    def test_floors_at_minimum_chunk(self) -> None:
        for n in (0, 1, 4):
            self.assertEqual(align_num_frames_down(n), MINIMAX_H3_LATENTS_PER_CHUNK)


if __name__ == "__main__":
    unittest.main()
