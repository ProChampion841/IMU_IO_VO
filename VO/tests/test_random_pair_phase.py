"""--random-pair-phase: training windows draw which frame the pair tiling starts on.

With --output-on-pairs and --frame-gap g the pairs tile the capture as
(0, g), (g, 2g), ... - phase 0. Phase p is the same tiling started p frames
later: (p, p+g), (p+g, p+2g), ... These tests pin that phase 0 is exactly the
plan built without the option, that every other phase keeps the pair interval
and the delivery cadence, that a window's per-event inputs describe the pairs
of the phase it drew, and that everything except training stays on phase 0.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch
from PIL import Image

from vio.data.attitude import AttitudeAltitude, pair_geometry_batch
from vio.data.fixedwing_vo import FixedWingVODataset, VONormalizer
from vio.data.image_pairs import VisualPairSource
from vio.models.pose_geometry import euler_zyx_to_quaternion_np
from vio.utils.checkpoint_io import load_checkpoint

GAP = 10
TICKS = 1200  # 12 s of 100 Hz telemetry
FRAMES = 200  # 10 s of 20 Hz frames


def _images(root, count: int = FRAMES, size=(32, 24)) -> None:
    folder = root / "images"
    folder.mkdir()
    pixels = np.arange(size[0] * size[1], dtype=np.uint8).reshape(size[1], size[0])
    for index in range(count):
        Image.fromarray(np.roll(pixels, index, axis=1)).save(folder / f"{index * 50}.jpg")


def _times() -> np.ndarray:
    return np.arange(TICKS, dtype=np.float64) * 0.01


def _source(root, **overrides) -> VisualPairSource:
    settings = dict(frame_gap=GAP, pair_stride=GAP, image_size=(24, 32))
    settings.update(overrides)
    return VisualPairSource(root, _times(), **settings)


def _attitude() -> AttitudeAltitude:
    times = _times()
    # A slow roll and yaw, so pair geometry differs from pair to pair and a
    # slot filled from the wrong phase's precomputed inputs would show.
    euler = np.stack(
        (0.2 * np.sin(times / 2.0), np.full(TICKS, 0.05), 0.1 * times), axis=1
    )
    quaternion = euler_zyx_to_quaternion_np(euler)
    return AttitudeAltitude(
        times_s=times,
        euler_rad=euler,
        quaternion=quaternion,
        body_rate_rad_s=np.zeros((TICKS, 3)),
        altitude_m=150.0 + times,
        attitude_columns=("NavEulX", "NavEulY", "NavEulZ"),
        altitude_column="relativeAlt",
        euler_unit="radians",
        attitude_hold_fraction=0.0,
        notes=(),
    )


def _dataset(source: VisualPairSource, *, random_pair_phase: bool) -> FixedWingVODataset:
    attitude = _attitude()
    span = (0, TICKS)
    return FixedWingVODataset(
        attitude,
        np.zeros((TICKS, 3), dtype=np.float32),
        span,
        VONormalizer.from_range(attitude, span),
        image_source=source,
        window_length=300,
        stride=300,
        warmup=0,
        max_visual_events=12,
        random_pair_phase=random_pair_phase,
    )


def test_phase_zero_is_exactly_the_plan_built_without_phases(tmp_path):
    _images(tmp_path)
    plain = _source(tmp_path)
    phased = _source(tmp_path, pair_phases=True)

    assert plain.phase_count == 1 and phased.phase_count == GAP
    assert phased.plan is phased.plan_for(0)
    for field in plain.plan.__dataclass_fields__:
        assert np.array_equal(getattr(plain.plan, field), getattr(phased.plan, field)), field
    assert phased.rejected_gap_pairs == plain.rejected_gap_pairs
    assert phased.rejected_telemetry_gap_pairs == plain.rejected_telemetry_gap_pairs


def test_every_phase_keeps_the_pair_interval_and_the_cadence(tmp_path):
    _images(tmp_path)
    source = _source(tmp_path, pair_phases=True)
    starts = set()
    for phase in range(GAP):
        plan = source.plan_for(phase)
        assert plan.first_index.size > 0
        assert np.all(plan.first_index % GAP == phase)
        assert np.array_equal(plan.second_index, plan.first_index + GAP)
        # One pair every 0.5 s, each spanning 0.5 s, whatever the phase.
        assert np.allclose(plan.pair_dt_s, 0.5)
        assert np.allclose(np.diff(plan.exposure_t1_s), 0.5)
        starts.update(plan.first_index.tolist())
    # Together the phases start a pair on every frame the plain plan could.
    assert starts == set(range(min(starts), max(starts) + 1))


def test_phases_need_a_stride_and_a_real_phase(tmp_path):
    _images(tmp_path)
    overlapping = _source(tmp_path, pair_stride=1, pair_phases=True)
    assert overlapping.phase_count == 1
    with pytest.raises(ValueError, match="pair phase"):
        _source(tmp_path, pair_phases=True).plan_for(GAP)


def test_a_window_draws_one_phase_and_its_inputs_describe_that_phase(tmp_path):
    _images(tmp_path)
    source = _source(tmp_path, pair_phases=True)
    dataset = _dataset(source, random_pair_phase=True)
    attitude = dataset.attitude

    torch.manual_seed(0)
    seen = set()
    for _ in range(40):
        item = dataset[1]
        phase = int(item["visual_pair_phase"])
        seen.add(phase)
        plan = source.plan_for(phase)
        valid = item["visual_event_valid"] > 0
        events = item["visual_event_index"][valid].numpy()
        assert events.size > 0
        # Every pair in the window belongs to the drawn phase ...
        assert np.all(plan.first_index[events] % GAP == phase)
        # ... lands where that phase's plan says, one every 0.5 s ...
        start = int(dataset.starts[1])
        offsets = item["visual_event_offset"][valid].numpy()
        assert np.array_equal(offsets, plan.ready_tick[events] - start)
        # (to within the one-tick rounding of placing a time on the 100 Hz grid)
        assert np.all(np.abs(np.diff(offsets) - 50) <= 1)
        # ... and carries that phase's own pair geometry.
        expected = pair_geometry_batch(
            attitude, plan.exposure_t0_s[events], plan.exposure_t1_s[events]
        )
        assert torch.allclose(
            item["visual_event_rotation"][valid],
            torch.from_numpy(expected["relative_rotation"]),
        )
        assert torch.allclose(
            item["visual_event_altitude"][valid], torch.from_numpy(expected["altitude_m"])
        )
    assert len(seen) > 3


def test_without_the_option_and_in_the_fixed_view_windows_are_phase_zero(tmp_path):
    _images(tmp_path)
    plain = _dataset(_source(tmp_path), random_pair_phase=False)
    phased = _dataset(_source(tmp_path, pair_phases=True), random_pair_phase=True)
    fixed = phased.fixed_phase_view()

    assert plain.fixed_phase_view() is plain
    assert fixed.random_pair_phase is False and phased.random_pair_phase is True
    for index in range(len(plain)):
        expected, got = plain[index], fixed[index]
        assert int(got["visual_pair_phase"]) == 0
        assert expected.keys() == got.keys()
        for key, value in expected.items():
            assert torch.equal(value, got[key]), key


def test_random_pair_phase_needs_a_source_with_phases(tmp_path):
    _images(tmp_path)
    with pytest.raises(ValueError, match="random_pair_phase"):
        _dataset(_source(tmp_path), random_pair_phase=True)


# --- the trainer -------------------------------------------------------------

import tools.make_synthetic_flight as make_synthetic_flight  # noqa: E402
import tools.train_fixedwing_vo as train_fixedwing_vo  # noqa: E402
from test_planar_training import HEIGHT, WIDTH, planar_argv  # noqa: E402


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    root = tmp_path_factory.mktemp("random_phase_flight")
    assert make_synthetic_flight.main([
        "--output", str(root), "--duration-s", "20",
        "--altitude-m", "150", "--speed-m-s", "20",
        "--image-width", str(WIDTH), "--image-height", str(HEIGHT),
        "--focal-px", "329", "--ground-metres-per-texel", "0.3",
        "--texture-size", "2048",
    ]) == 0
    return root


def test_the_trainer_draws_phases_for_training_only(rendered, tmp_path, monkeypatch):
    built = {}
    original = train_fixedwing_vo.build_vo_dataset

    def recording(root, span, **kwargs):
        dataset, normalizer, attitude = original(root, span, **kwargs)
        built[len(built)] = dataset
        return dataset, normalizer, attitude

    monkeypatch.setattr(train_fixedwing_vo, "build_vo_dataset", recording)
    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(
        planar_argv(rendered, run_dir, output_on_pairs=True, warmup=None,
                    random_pair_phase=True, eval_train_split=True)
    ) == 0

    datasets = list(built.values())
    # train, validation, test in that order; only the first draws phases.
    assert [d.random_pair_phase for d in datasets] == [True, False, False]
    assert datasets[0].image_source.phase_count == 10
    assert all(d.image_source.phase_count == 1 for d in datasets[1:])

    checkpoint = load_checkpoint(run_dir / "last.pt", map_location="cpu")
    assert checkpoint["args"]["random_pair_phase"] is True
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        row = list(csv.DictReader(handle))[0]
    for column in ("train_vel_rmse", "val_vel_rmse", "traineval_vel_rmse"):
        assert np.isfinite(float(row[column])), column


def test_the_trainer_refuses_random_pair_phase_without_a_pair_stride(rendered, tmp_path):
    with pytest.raises(SystemExit, match="random-pair-phase"):
        train_fixedwing_vo.main(
            planar_argv(rendered, tmp_path / "run", random_pair_phase=True)
        )
