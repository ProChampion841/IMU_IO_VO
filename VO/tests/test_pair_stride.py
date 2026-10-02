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


def test_the_evaluator_scores_overlapping_pairs_exactly_as_training_does(rendered, tmp_path):
    """One window through the trainer's VOStep and the same span streamed by
    the evaluator: same delivered pairs, same scored ticks, same velocities."""

    import torch

    import vio.models.velocity_horizons as vh
    from tools.export_onnx import load_models
    from vio.data.calibration import maybe_load_camera_calibration
    from vio.data.fixedwing_vo import VONormalizer, build_vo_dataset

    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(
        planar_argv(rendered, run_dir, frame_gap="20", output_on_pairs=True, pair_stride="10",
                    dropout="0.1")
    ) == 0
    frontend, model, saved, camera, normalizer = load_models(run_dir / "best.pt")
    calibration = maybe_load_camera_calibration(saved["calibration"])
    span, warmup, length = (1200, 1600), 90, 400
    dataset, _, attitude = build_vo_dataset(
        rendered, span, image_size=tuple(saved["image_size"]), frame_gap=saved["frame_gap"],
        pair_stride=saved["pair_stride"], max_frame_gap_s=saved["max_frame_gap_s"],
        deployment_latency_s=saved["deployment_latency_s"],
        camera_matrix=calibration.camera_matrix, calibration_image_size=calibration.native_size,
        normalizer=VONormalizer(**normalizer), window_length=length, stride=length,
        warmup=warmup, max_visual_events=14, grayscale=not saved.get("color", False),
    )

    step = train_fixedwing_vo.VOStep(
        model, window_length=length, visual_dim=model.visual_dim, disable_visual=False,
        frontend_chunk=8, deployment_latency_s=saved["deployment_latency_s"],
        camera_matrix=camera, output_on_pairs=True,
    ).eval()
    item = dataset[0]
    batch = {key: value.unsqueeze(0) for key, value in item.items()}
    with torch.no_grad():
        prediction, _, mask = train_fixedwing_vo.forward_batch(step, batch, torch.device("cpu"))
    train_mask = mask[0].numpy() > 0
    train_velocity = prediction["predicted_velocity"][0].numpy()
    events = item["visual_event_index"][item["visual_event_valid"] > 0].numpy()
    plan = dataset.image_source.plan
    assert np.all(plan.second_index[events][:-2] == plan.first_index[events][2:])  # overlapping

    tokens = vh.encode_span_tokens(
        frontend, dataset.image_source, span=span, body_rate_rad_s=attitude.body_rate_rad_s,
        times_s=attitude.times_s, visual_dim=model.visual_dim, device=torch.device("cpu"),
        camera_matrix=camera, batch_pairs=3, attitude=attitude,
    )
    _, whole = vh.run_span_horizons(
        model, aiding=dataset.aiding, log_altitude=dataset.log_altitude,
        target_velocity=dataset.velocity_body, times_s=attitude.times_s, span=span,
        tokens=tokens, deployment_latency_s=saved["deployment_latency_s"],
        horizons_minutes=(0.05,), warmup_ticks=warmup, block_ticks=97,
        collect_series=True, output_on_pairs=True,
    )
    streamed = whole["series"]["vel_predicted_body"][0]
    eval_mask = np.isfinite(streamed).all(-1)
    assert len(tokens) == events.size
    assert train_mask.sum() > 3
    assert np.array_equal(train_mask, eval_mask)
    assert np.allclose(streamed[eval_mask], train_velocity[train_mask], atol=1e-5)
