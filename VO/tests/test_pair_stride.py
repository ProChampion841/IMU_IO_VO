"""--pair-stride: how often a pair (and an output) starts, independent of --frame-gap.

--frame-gap 20 --pair-stride 10 at 20 Hz is a 1 s pair every 0.5 s: twice the
ground motion per measurement of a 0.5 s pair, at the same output rate. These
tests pin the resolved settings, the overlapping pair plan, and that training,
the evaluator and the standalone ONNX runtime all agree on it.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import tools.evaluate_velocity_horizons as evaluate_velocity_horizons
import tools.make_synthetic_flight as make_synthetic_flight
import tools.train_fixedwing_vo as train_fixedwing_vo
from vio.data.image_pairs import VisualPairSource
from vio.utils.checkpoint_io import load_checkpoint

from test_planar_training import HEIGHT, WIDTH, planar_argv  # noqa: E402


def _resolved(*flags: str):
    args = train_fixedwing_vo.build_parser().parse_args(
        ["--dataset", "ignored", "--frontend", "planar", *flags]
    )
    chosen = train_fixedwing_vo.resolve_frontend_defaults(
        args, frame_interval_s=0.05, tick_interval_s=0.01
    )
    return args, chosen


def test_the_default_stride_is_the_frame_gap():
    args, _ = _resolved("--output-on-pairs", "--frame-gap", "10")
    assert (args.frame_gap, args.pair_stride) == (10, 10)
    plain, _ = _resolved("--frame-gap", "10")
    assert plain.pair_stride == 1


def test_a_one_second_pair_every_half_second():
    args, chosen = _resolved("--output-on-pairs", "--frame-gap", "20", "--pair-stride", "10")
    assert (args.frame_gap, args.pair_stride) == (20, 10)
    # The gap cap follows the pair length, not the output rate.
    assert args.max_frame_gap_s == pytest.approx(1.5)
    # Worst-case first delivery: one stride to start + one pair + the latency.
    assert args.warmup == math.ceil((0.35 + (20 + 10 - 1) * 0.05) / 0.01 - 1e-6) + 5
    assert any("overlap" in line for line in chosen)


def test_pair_stride_needs_output_on_pairs_and_a_positive_value():
    with pytest.raises(SystemExit, match="--pair-stride"):
        _resolved("--frame-gap", "20", "--pair-stride", "10")
    with pytest.raises(SystemExit, match="at least 1"):
        _resolved("--output-on-pairs", "--frame-gap", "20", "--pair-stride", "0")


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    root = tmp_path_factory.mktemp("pair_stride_flight")
    assert make_synthetic_flight.main([
        "--output", str(root), "--duration-s", "20",
        "--altitude-m", "150", "--speed-m-s", "20",
        "--image-width", str(WIDTH), "--image-height", str(HEIGHT),
        "--focal-px", "329", "--ground-metres-per-texel", "0.3",
        "--texture-size", "2048",
    ]) == 0
    return root


def test_overlapping_pairs_span_one_second_and_arrive_every_half_second(rendered):
    times = np.arange(0.0, 20.0, 0.01) + 1000.0
    source = VisualPairSource(rendered, times, frame_gap=20, pair_stride=10)
    plan = source.plan
    assert plan.first_index[:3].tolist() == [0, 10, 20]
    assert plan.second_index[:3].tolist() == [20, 30, 40]
    assert np.allclose(plan.pair_dt_s, 1.0, atol=0.02)
    assert np.allclose(np.diff(plan.exposure_t1_s[:6]), 0.5, atol=0.02)
    # Consecutive pairs share an image: one's second frame is the next-but-one's first.
    assert np.array_equal(plan.second_index[:-2], plan.first_index[2:])


def test_training_and_evaluation_with_a_one_second_pair_every_half_second(rendered, tmp_path):
    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(
        planar_argv(rendered, run_dir, frame_gap="20", output_on_pairs=True, pair_stride="10")
    ) == 0
    checkpoint = load_checkpoint(run_dir / "last.pt", map_location="cpu")
    saved = checkpoint["args"]
    assert (saved["frame_gap"], saved["pair_stride"]) == (20, 10)
    assert checkpoint["fingerprint"]["contract"]["frame_gap"] == 20
    assert checkpoint["fingerprint"]["contract"]["pair_stride"] == 10

    output = tmp_path / "eval.json"
    assert evaluate_velocity_horizons.main([
        str(run_dir / "best.pt"), "--dataset", str(rendered),
        "--splits", "validation", "--horizons", "0.04",
        "--no-plots", "--no-progress", "--output", str(output),
    ]) == 0
    horizon = json.loads(output.read_text(encoding="utf-8"))["splits"]["validation"]["horizons"]["h0.04m"]
    assert horizon["fits"], horizon.get("skipped")
    # 2.4 s after a 0.9 s warm-up: about one scored output per 0.5 s.
    assert 0 < horizon["scored_ticks"] <= 4
    assert np.isfinite(horizon["vel_rmse"])


def test_the_onnx_runtime_pairs_exactly_as_training_does(rendered, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import tools.export_onnx as export_onnx
    import tools.onnx_inference as onnx_inference

    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(
        planar_argv(rendered, run_dir, frame_gap="20", output_on_pairs=True, pair_stride="10")
    ) == 0
    export_dir = tmp_path / "onnx"
    metadata = export_onnx.export_checkpoint(run_dir / "best.pt", export_dir, verify_ticks=60)
    assert metadata["timing"]["frame_gap"] == 20 and metadata["timing"]["pair_stride"] == 10
    # Replays the flight through frontend.onnx + temporal_step.onnx and fails
    # unless the delivery ticks equal PyTorch's and the velocities agree.
    assert onnx_inference.main([
        str(export_dir), "--dataset", str(rendered),
        "--checkpoint", str(run_dir / "best.pt"), "--no-progress",
    ]) == 0
