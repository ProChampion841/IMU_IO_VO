"""An event's READY tick can land inside a split while its two frames and its
attitude sample were captured one deployment latency (0.35 s by default)
EARLIER - which, at a condition-segment boundary with no gap between phases,
is on the far side of it. ``min_capture_tick`` on
:meth:`VisualPairSource.events_in_window` (and the ``window_pairs`` that
forwards it) exists to drop such an event rather than let a held-out tick be
built in part from images and a gyro-like reading captured during training.
This pins that it actually excludes the right event and nothing else.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from vio.data.attitude import AttitudeAltitude
from vio.data.fixedwing_vo import FixedWingVODataset, VONormalizer
from vio.data.image_pairs import VisualPairSource


def _images_every_50ms(root, count: int, size=(32, 24)):
    folder = root / "images"
    folder.mkdir()
    pixels = np.arange(size[0] * size[1], dtype=np.uint8).reshape(size[1], size[0])
    for index in range(count):
        timestamp_ms = index * 50
        Image.fromarray(np.roll(pixels, index, axis=1)).save(
            folder / f"{timestamp_ms}.jpg"
        )


def _source(tmp_path, *, deployment_latency_s: float = 0.35) -> VisualPairSource:
    _images_every_50ms(tmp_path, count=32)  # 0, 50, ..., 1550 ms
    times_s = np.arange(300, dtype=np.float64) * 0.01  # 100 Hz, 0..2.99 s
    return VisualPairSource(
        tmp_path,
        times_s,
        deployment_latency_s=deployment_latency_s,
        frame_gap=1,
        image_size=(24, 32),
    )


def test_an_event_whose_exposure_predates_the_boundary_is_dropped(tmp_path):
    source = _source(tmp_path)

    # Pair (600 ms, 650 ms): exposure ticks (60, 65); ready = 0.65 + 0.35 = 1.00 s
    # -> ready_tick 100 exactly, by construction of the fixture above.
    event = int(
        np.flatnonzero(
            (source.plan.ready_tick == 100) & (source.plan.telemetry_index0 == 60)
        )[0]
    )
    assert source.plan.telemetry_index0[event] == 60
    assert source.plan.telemetry_index1[event] == 65
    assert source.plan.ready_tick[event] == 100

    boundary = 100  # a segment/window that starts exactly at this event's ready tick

    # Without the guard, the event's ready tick alone puts it in the window -
    # this is the PRE-FIX behaviour, still available with no bound given.
    unguarded = source.events_in_window(boundary, boundary + 10)
    assert event in unguarded

    # With the guard, the same event's capture (tick 60) is before the
    # boundary (100), so it must be excluded even though its ready tick is
    # inside the window.
    guarded = source.events_in_window(boundary, boundary + 10, min_capture_tick=boundary)
    assert event not in guarded


def test_an_event_captured_at_or_after_the_boundary_is_kept(tmp_path):
    """The guard must not over-exclude: an event captured on the right side

    of the boundary survives even though other events, earlier in the same
    window, are dropped for having been captured on the wrong side of it.
    """

    source = _source(tmp_path)
    boundary = 100

    # Pair (1000 ms, 1050 ms): exposure ticks (100, 105) - captured AT the
    # boundary tick itself, so this one must be kept.
    kept_event = int(
        np.flatnonzero(
            (source.plan.telemetry_index0 == 100) & (source.plan.telemetry_index1 == 105)
        )[0]
    )
    # Pair (600 ms, 650 ms) from the other test - captured well before the
    # boundary, so it must still be dropped in this same window.
    dropped_event = int(
        np.flatnonzero(
            (source.plan.ready_tick == 100) & (source.plan.telemetry_index0 == 60)
        )[0]
    )

    window_end = int(source.plan.ready_tick[kept_event]) + 1
    guarded = source.events_in_window(boundary, window_end, min_capture_tick=boundary)

    assert kept_event in guarded
    assert dropped_event not in guarded


def test_window_pairs_and_events_in_window_agree_under_the_same_bound(tmp_path):
    """The two calls a dataset makes for one window must select identically,

    or the body-rate slot for a token would not describe that token's own
    event.
    """

    source = _source(tmp_path)
    boundary = 100
    end = boundary + 10

    selected = source.events_in_window(boundary, end, min_capture_tick=boundary)
    pairs = source.window_pairs(boundary, end, max_events=8, min_capture_tick=boundary)

    assert int(pairs["visual_event_valid"].sum().item()) == selected.size
    offsets = pairs["visual_event_offset"][: selected.size].tolist()
    expected = sorted((int(source.plan.ready_tick[e]) - boundary) for e in selected)
    assert sorted(int(o) for o in offsets) == expected


def test_a_window_at_a_segment_start_never_returns_a_foreign_event(tmp_path):
    """End to end: FixedWingVODataset must never hand out a token or a body

    rate built from an event captured before its own segment's start, even
    for the very first window of that segment.
    """

    source = _source(tmp_path)
    total = 300
    times_s = np.arange(total, dtype=np.float64) * 0.01
    attitude = AttitudeAltitude(
        times_s=times_s,
        euler_rad=np.zeros((total, 3)),
        quaternion=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (total, 1)),
        body_rate_rad_s=np.zeros((total, 3)),
        altitude_m=np.full(total, 120.0),
        attitude_columns=("NavEulX", "NavEulY", "NavEulZ"),
        altitude_column="relativeAlt",
        euler_unit="radians",
        attitude_hold_fraction=0.0,
        notes=(),
    )
    velocity = np.zeros((total, 3), dtype=np.float32)

    # A segment starting exactly at tick 100 - the same boundary as the two
    # tests above, where a real event's ready tick lands right at the start
    # but its capture (tick 60) does not.
    normalizer = VONormalizer.from_range(attitude, (100, 200))
    dataset = FixedWingVODataset(
        attitude, velocity, (100, 200), normalizer,
        image_source=source, window_length=10, stride=10, warmup=0,
        max_visual_events=8,
    )

    # This fixture's only event with ready tick 100 is the one captured at
    # tick 60, on the far side of the segment's own start - so if it leaked
    # through, it would show up as offset 0 in the segment's first window.
    assert int(
        np.flatnonzero(
            (source.plan.ready_tick == 100) & (source.plan.telemetry_index0 == 60)
        ).size
    ) == 1

    first_window = dataset[0]  # starts=100: the segment's very first window
    valid = first_window["visual_event_valid"] > 0
    offsets = set(int(o) for o in first_window["visual_event_offset"][valid].tolist())
    assert 0 not in offsets
