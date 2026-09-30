"""Regression coverage for ``tools.train_fixedwing_vo.encode_pairs``.

Until this file, nothing in the test suite called ``encode_pairs`` directly.
The trainer's own integration tests exercise it only through tiny synthetic
flights where a window's pair count never exceeds ``--frontend-chunk``, so
chunking never has anything to bound - which is exactly how a validation-only
regression (see UPDATES.txt: the 0910 frontend-chunk bypass) survived a full
green suite.

The bug: ``encode_pairs`` used to read
``if chunk <= 0 or not torch.is_grad_enabled(): ... one call with every pair``.
``evaluate()`` and ``run_horizon_pass()`` both run under ``torch.no_grad()``,
so validation silently ignored ``--frontend-chunk`` and sent every pair in the
batch through the frontend - and its correlator, whose largest tensor is
shaped ``(pairs, channels, candidates, height, width)`` - in one call. At the
trainer's real defaults that is tens of gigabytes for a single tensor.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

import tools.train_fixedwing_vo as train_fixedwing_vo
from vio.models.vision_mamba_vo import VisionMambaFlowFrontend


class _CountingFrontend(torch.nn.Module):
    """Records how many pairs each call received, instead of a real frontend.

    Carries one real parameter so a grad-enabled call has something to build
    an autograd graph through - otherwise ``checkpoint()`` has nothing that
    requires grad and the training-mode and eval-mode code paths inside
    ``encode_pairs`` would not actually be exercised differently.
    """

    def __init__(self, visual_dim: int = 4) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.calls: List[int] = []
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
        *,
        pair_dt_s: torch.Tensor,
        body_rate_rad_s: torch.Tensor,
        camera_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pairs = image0.shape[0]
        self.calls.append(pairs)
        token = (image0.mean(dim=(1, 2, 3)) * self.scale).reshape(pairs, 1)
        token = token.expand(pairs, self.visual_dim)
        quality = torch.ones(pairs, 1)
        # Every pair reliable: this stub has no correlation grid to gate on,
        # and encode_pairs requires the key because a real frontend always
        # supplies it - see VisionMambaFlowFrontend.forward.
        reliable = torch.ones(pairs, 1)
        return {
            "visual_token": token,
            "visual_quality": quality,
            "pair_reliable": reliable,
        }


def _fake_pairs(pairs: int):
    image0 = torch.randint(0, 256, (pairs, 1, 4, 4), dtype=torch.uint8)
    image1 = torch.randint(0, 256, (pairs, 1, 4, 4), dtype=torch.uint8)
    pair_dt_s = torch.full((pairs,), 0.05)
    body_rate = torch.zeros(pairs, 3)
    return image0, image1, pair_dt_s, body_rate


def test_encode_pairs_chunks_the_frontend_under_no_grad():
    """20 pairs, chunk 4, no_grad -> 5 calls of 4 pairs each, not 1 of 20.

    This is the exact scenario evaluate() runs every window through: pinning
    it directly is what would have caught the regression before it reached a
    real GPU.
    """

    frontend = _CountingFrontend()
    image0, image1, pair_dt_s, body_rate = _fake_pairs(20)

    with torch.no_grad():
        train_fixedwing_vo.encode_pairs(
            frontend, image0, image1, pair_dt_s, body_rate, chunk=4
        )

    assert frontend.calls == [4, 4, 4, 4, 4]


def test_encode_pairs_chunks_the_frontend_with_grad_enabled():
    """The same 20-pairs/chunk-4 split holds during training too."""

    frontend = _CountingFrontend()
    image0, image1, pair_dt_s, body_rate = _fake_pairs(20)

    train_fixedwing_vo.encode_pairs(
        frontend, image0, image1, pair_dt_s, body_rate, chunk=4
    )

    assert frontend.calls == [4, 4, 4, 4, 4]


def test_encode_pairs_chunk_disabled_still_calls_the_frontend_once():
    """chunk=0 is the explicit escape hatch and must keep meaning "one call"."""

    frontend = _CountingFrontend()
    image0, image1, pair_dt_s, body_rate = _fake_pairs(20)

    with torch.no_grad():
        train_fixedwing_vo.encode_pairs(
            frontend, image0, image1, pair_dt_s, body_rate, chunk=0
        )

    assert frontend.calls == [20]


def test_encode_pairs_ragged_final_chunk_under_no_grad():
    """A pair count that does not divide evenly by chunk - the last call gets
    the remainder, not a padded or dropped one."""

    frontend = _CountingFrontend()
    image0, image1, pair_dt_s, body_rate = _fake_pairs(10)

    with torch.no_grad():
        train_fixedwing_vo.encode_pairs(
            frontend, image0, image1, pair_dt_s, body_rate, chunk=3
        )

    assert frontend.calls == [3, 3, 3, 1]


def test_encode_pairs_converts_uint8_input_without_double_normalising():
    """encode_pairs takes raw image dtype (uint8) and normalises internally -
    callers must NOT pre-divide by 255, or every value would be normalised
    twice."""

    frontend = _CountingFrontend()
    pairs = 3
    image0 = torch.full((pairs, 1, 2, 2), 255, dtype=torch.uint8)
    image1 = torch.full((pairs, 1, 2, 2), 255, dtype=torch.uint8)
    pair_dt_s = torch.full((pairs,), 0.05)
    body_rate = torch.zeros(pairs, 3)

    with torch.no_grad():
        token, _, _ = train_fixedwing_vo.encode_pairs(
            frontend, image0, image1, pair_dt_s, body_rate, chunk=0
        )

    # The stub's token is image0's per-pair mean; 255 uint8 normalised once
    # is 1.0, so every entry must be exactly 1.0 - 255/255/255 would be
    # 0.00392... if the caller (or encode_pairs) divided by 255 twice.
    assert torch.allclose(token, torch.ones_like(token))


def test_chunked_and_unchunked_encoding_agree_on_a_real_frontend():
    """Chunking must be a memory optimisation only - the actual measurement a
    real frontend produces must not depend on how its input batch was split.
    """

    torch.manual_seed(0)
    frontend = VisionMambaFlowFrontend(
        visual_dim=8, d_model=8, depth=1, patch_size=8,
        image_size=(64, 96), context_grid=(8, 12),
        correlation_radius=2, token_grid=4, dropout=0.0,
    )
    frontend.eval()

    pairs = 10
    image0 = torch.randint(0, 256, (pairs, 1, 64, 96), dtype=torch.uint8)
    image1 = torch.randint(0, 256, (pairs, 1, 64, 96), dtype=torch.uint8)
    pair_dt_s = torch.full((pairs,), 0.05)
    body_rate = torch.zeros(pairs, 3)

    with torch.no_grad():
        whole_token, whole_quality, _ = train_fixedwing_vo.encode_pairs(
            frontend, image0, image1, pair_dt_s, body_rate, chunk=0
        )
        chunked_token, chunked_quality, _ = train_fixedwing_vo.encode_pairs(
            frontend, image0, image1, pair_dt_s, body_rate, chunk=3
        )

    torch.testing.assert_close(chunked_token, whole_token)
    torch.testing.assert_close(chunked_quality, whole_quality)


def test_correlation_candidate_gib_matches_the_trainer_defaults_estimate():
    """Pins the number Codex's diagnosis and the new startup log both quote:
    480 pairs (4 windows x 120 events) at the trainer's real defaults (64
    channels, radius 4 -> 81 candidates, 72x128 grid) is just over 85 GiB for
    the correlator's single largest tensor."""

    gib = train_fixedwing_vo._correlation_candidate_gib(
        pairs=480, channels=64, candidates=81, height=72, width=128
    )
    assert 85.0 < gib < 86.0

    # And the whole point of chunking: the same tensor at chunk=8 is small.
    chunked_gib = train_fixedwing_vo._correlation_candidate_gib(
        pairs=8, channels=64, candidates=81, height=72, width=128
    )
    assert 1.0 < chunked_gib < 2.0
