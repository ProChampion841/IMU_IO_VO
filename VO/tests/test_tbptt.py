"""ChronologicalWindowSampler: the TBPTT batching primitive.

Ordinary shuffled batching breaks a training path that carries recurrent
state between batches, because "the next batch" then bears no relation to
"the tick after the last one this state saw." These tests pin the two
properties that make the sampler safe to build TBPTT on: every lane really
does walk forward in time with exact tick-adjacency, and a lane crossing an
index_ranges boundary - or its very first step - is flagged so the caller
knows to reset that lane's state rather than carry it forward. See
PLAN_TBPTT.txt for the design this scaffolds.

The sampler only ever reads dataset.starts/.window_length/.index_ranges and
FixedWingVODataset._range_start, so these tests build datasets with
image_source=None - the image pipeline is never touched by anything under
test here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vio.data.attitude import AttitudeAltitude
from vio.data.fixedwing_vo import (
    ChronologicalWindowSampler,
    FixedWingVODataset,
    VONormalizer,
    tbptt_loss_mask,
)
from vio.models.pose_geometry import euler_zyx_to_quaternion_np


def _make_dataset(total_ticks: int, index_range, *, window_length: int, stride: int):
    times = np.arange(total_ticks, dtype=np.float64) * 0.01
    euler = np.zeros((total_ticks, 3))
    attitude = AttitudeAltitude(
        times_s=times,
        euler_rad=euler,
        quaternion=euler_zyx_to_quaternion_np(euler),
        body_rate_rad_s=np.zeros((total_ticks, 3)),
        altitude_m=np.full(total_ticks, 100.0),
        attitude_columns=("NavEulX", "NavEulY", "NavEulZ"),
        altitude_column="relativeAlt",
        euler_unit="radians",
        attitude_hold_fraction=0.0,
        notes=(),
    )
    velocity_body = np.zeros((total_ticks, 3), dtype=np.float32)
    normalizer = VONormalizer(log_altitude_mean=0.0, delta_time_scale=1.0)
    return FixedWingVODataset(
        attitude, velocity_body, index_range, normalizer,
        image_source=None,  # never touched: the sampler reads no image data
        window_length=window_length, stride=stride, warmup=0, max_visual_events=1,
    )


def test_lanes_walk_forward_in_time_and_reset_only_on_their_first_step():
    """One contiguous range, non-overlapping windows: every step after the
    first is a true continuation, so only step 0 should ever ask for a reset."""

    dataset = _make_dataset(100, (0, 100), window_length=10, stride=10)
    sampler = ChronologicalWindowSampler(dataset, batch_size=2)
    assert len(sampler) == 5  # 10 windows / 2 lanes

    batches = list(sampler)
    assert len(batches) == 5
    for lane in range(2):
        starts = [dataset.starts[batches[step][lane]] for step in range(5)]
        # Strictly increasing by exactly window_length - the tiling this
        # sampler exists to guarantee.
        assert starts == sorted(starts)
        assert all(b - a == 10 for a, b in zip(starts, starts[1:]))
        assert sampler.continues_at(0)[lane].item() is False
        for step in range(1, 5):
            assert sampler.continues_at(step)[lane].item() is True


def test_a_lane_crossing_a_range_boundary_is_flagged_for_reset():
    """Two disjoint ranges (a condition-segment split), batch_size chosen so
    one lane's walk crosses from the tail of range A into the head of range
    B. That join was never actually flown continuously, so it must be
    flagged exactly like step 0 is - carrying state across it would smuggle
    range A's ending state into range B's unrelated beginning."""

    # Range A: ticks [0, 50) -> windows at 0, 10, 20, 30, 40 (5 windows).
    # Range B: ticks [60, 110) -> windows at 60, 70, 80, 90, 100 (5 windows).
    dataset = _make_dataset(110, [(0, 50), (60, 110)], window_length=10, stride=10)
    sampler = ChronologicalWindowSampler(dataset, batch_size=3)
    # 10 windows total, 3 lanes -> lane_length = 3, one window dropped.
    assert len(sampler) == 3

    batches = list(sampler)
    starts_by_lane = [
        [int(dataset.starts[batches[step][lane]]) for step in range(3)]
        for lane in range(3)
    ]
    # Lane 0: 0, 10, 20 - entirely inside range A.
    assert starts_by_lane[0] == [0, 10, 20]
    # Lane 1: 30, 40, 60 - crosses from range A's tail into range B's head.
    assert starts_by_lane[1] == [30, 40, 60]
    # Lane 2: 70, 80, 90 - entirely inside range B.
    assert starts_by_lane[2] == [70, 80, 90]

    assert sampler.continues_at(0).tolist() == [False, False, False]
    # Lane 1 steps from 30 -> 40 (continues) then 40 -> 60 (a range jump: not
    # a continuation, even though it is still later in tick order).
    assert sampler.continues_at(1).tolist() == [True, True, True]
    assert sampler.continues_at(2).tolist() == [True, False, True]


def test_overlapping_windows_are_refused_rather_than_silently_reprocessed():
    """stride < window_length means "the next window" re-covers ticks the
    previous one already saw - TBPTT would either double-count that overlap
    or need a tick-level re-seen mask this scaffold does not implement, so
    construction must refuse rather than train on it quietly."""

    dataset = _make_dataset(100, (0, 100), window_length=10, stride=5)
    with pytest.raises(ValueError, match="stride == window_length"):
        ChronologicalWindowSampler(dataset, batch_size=2)


def test_too_few_windows_for_the_requested_lane_count_is_refused():
    dataset = _make_dataset(30, (0, 30), window_length=10, stride=10)
    with pytest.raises(ValueError, match="Not enough windows"):
        ChronologicalWindowSampler(dataset, batch_size=5)


# ---------------------------------------------------------------------------
# tbptt_loss_mask: warm-up only where a lane actually reset
# ---------------------------------------------------------------------------


def test_tbptt_loss_mask_restores_warmup_only_on_continuing_lanes():
    """A dataset window always zeroes its own first `warmup` ticks - correct
    for a cold start, wrong for a lane whose state was carried in from the
    previous chunk. Mixed batch: lane 0 continues (state carried, no real
    blind period), lane 1 resets (state=None, the cold start is real) - only
    lane 0's warmup should be restored to 1."""

    warmup = 3
    loss_mask = torch.ones(2, 6)
    loss_mask[:, :warmup] = 0.0  # what FixedWingVODataset.__getitem__ produces
    continues = torch.tensor([True, False])

    fixed = tbptt_loss_mask(loss_mask, continues, warmup)
    assert fixed[0, :warmup].tolist() == [1.0, 1.0, 1.0]  # continuing: restored
    assert fixed[1, :warmup].tolist() == [0.0, 0.0, 0.0]  # reset: kept blind
    # Ticks after the warmup window are untouched either way.
    assert fixed[0, warmup:].tolist() == [1.0, 1.0, 1.0]
    assert fixed[1, warmup:].tolist() == [1.0, 1.0, 1.0]
    # The input is not mutated in place - a caller that reuses loss_mask
    # elsewhere (e.g. logging it) must see the original.
    assert loss_mask[0, 0].item() == 0.0


def test_tbptt_loss_mask_is_a_no_op_when_warmup_is_zero():
    loss_mask = torch.ones(2, 4)
    continues = torch.tensor([True, False])
    assert torch.equal(tbptt_loss_mask(loss_mask, continues, 0), loss_mask)
