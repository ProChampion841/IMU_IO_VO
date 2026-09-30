"""The flat-ground path through the real entry points: trainer, evaluator and
the two calibration tools, on a flight rendered from a known trajectory.

``test_planar_frontend.py`` pins the geometry in isolation. This pins the
wiring that turns it into a run: that the pair geometry actually reaches the
frontend from the dataset, that the settings a run resolved (mounting, prior,
velocity mode, colour) survive into the checkpoint and come back out of the
evaluator unchanged, that a resume under a different geometry is refused, and
that the tools which choose ``--frame-gap`` and ``--camera-mounting`` give the
right answer on a flight whose right answer is known.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import tools.check_motion_budget as check_motion_budget
import tools.estimate_camera_mounting as estimate_camera_mounting
import tools.evaluate_velocity_horizons as evaluate_velocity_horizons
import tools.make_synthetic_flight as make_synthetic_flight
import tools.train_fixedwing_vo as train_fixedwing_vo
from vio.models.planar_frontend import PlanarFlowFrontend
from vio.models.planar_geometry import NADIR_MOUNTINGS
from vio.models.vision_mamba_vo import VisionMambaVO
from vio.utils.checkpoint_io import load_checkpoint

HEIGHT, WIDTH = 192, 320


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> Path:
    """20 s at 150 m and 20 m/s, 320x192 at the real camera's field of view.

    The renderer's camera axes coincide with the body axes, which is the
    ``right_forward`` mounting.
    """

    root = tmp_path_factory.mktemp("planar_flight")
    assert make_synthetic_flight.main([
        "--output", str(root), "--duration-s", "20",
        "--altitude-m", "150", "--speed-m-s", "20",
        "--image-width", str(WIDTH), "--image-height", str(HEIGHT),
        "--focal-px", "329", "--ground-metres-per-texel", "0.3",
        "--texture-size", "2048",
    ]) == 0
    return root


def planar_argv(flight: Path, run_dir: Path, **overrides) -> list:
    args = {
        "dataset": str(flight),
        "calibration": str(flight / "calibration.json"),
        "frontend": "planar",
        "camera-mounting": "right_forward",
        "frame-gap": "10",
        "coarse-factor": "2", "coarse-radius": "4",
        "coarse-highpass": "3", "fine-highpass": "5",
        "correlation-radius": "3",
        "window-length": "200", "stride": "200", "warmup": "90",
        "max-visual-events": "6",
        "visual-dim": "8", "stem-dim": "8", "stem-depth": "1",
        "patch-size": "8", "token-grid": "4",
        "aiding-dim": "8", "fusion-dim": "8", "dropout": "0.0",
        "epochs": "1", "batch-size": "2", "num-workers": "0",
        "device": "cpu", "run-dir": str(run_dir),
        "train-fraction": "0.5", "validation-fraction": "0.25",
        "velocity-loss": "simple",
    }
    args.update({key.replace("_", "-"): value for key, value in overrides.items()})
    argv = ["--no-progress", "--image-size", str(HEIGHT), str(WIDTH),
            "--context-grid", "4", "6"]
    for name, value in args.items():
        if isinstance(value, bool):
            if value:
                argv.append(f"--{name}")
        elif value is not None:
            argv += [f"--{name}", str(value)]
    return argv


def test_the_trainer_runs_the_planar_frontend_and_records_what_it_resolved(rendered, tmp_path):
    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(
        planar_argv(rendered, run_dir, color=True, photometric_augment="0.1",
                    lr_warmup_epochs="1")
    ) == 0

    checkpoint = load_checkpoint(run_dir / "last.pt", map_location="cpu")
    saved = checkpoint["args"]
    assert saved["frontend"] == "planar"
    assert saved["velocity_mode"] == "geometric_residual"
    assert saved["frontend_id"] == PlanarFlowFrontend.frontend_id
    assert saved["temporal_input_id"] == VisionMambaVO.temporal_input_id_for("geometric_residual")
    assert np.allclose(saved["camera_from_body"], NADIR_MOUNTINGS["right_forward"])
    # The prior defaults to the TRAINING split's mean velocity.
    assert np.allclose(saved["prior_velocity_body"], checkpoint["train_baseline_velocity_m_s"])
    assert checkpoint["fingerprint"]["contract"]["color"] is True
    assert checkpoint["fingerprint"]["model"]["planar"]["coarse_factor"] == 2
    # RGB stem: three input channels.
    assert checkpoint["frontend"]["stem.patch_embed.weight"].shape[1] == 3
    contract = json.loads((run_dir / "input_contract.json").read_text(encoding="utf-8"))
    assert contract["image_input"]["kind"] == "rgb_frame_pair"
    assert contract["frontend"] == "planar"

    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    # Untrained, the model outputs the held geometric velocity; on a rendered
    # 150 m flight that is already far better than the train-mean baseline.
    assert float(rows[0]["val_vel_rmse"]) < float(rows[0]["val_baseline_vel_rmse"])


def test_the_evaluator_rebuilds_and_scores_the_planar_run(rendered, tmp_path):
    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(planar_argv(rendered, run_dir)) == 0
    output = tmp_path / "eval.json"
    assert evaluate_velocity_horizons.main([
        str(run_dir / "best.pt"),
        "--dataset", str(rendered),
        "--splits", "validation",
        "--horizons", "0.02",
        "--no-plots", "--no-progress",
        "--output", str(output),
    ]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    horizon = report["splits"]["validation"]["horizons"]["h0.02m"]
    assert horizon["fits"], horizon.get("skipped")
    assert horizon["visual_events"] > 0
    assert np.isfinite(horizon["vel_rmse"])


def test_a_resume_under_a_different_mounting_is_refused(rendered, tmp_path):
    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(planar_argv(rendered, run_dir)) == 0
    with pytest.raises(SystemExit, match="planar"):
        train_fixedwing_vo.main(
            planar_argv(rendered, run_dir, camera_mounting="top_forward", epochs="2")
            + ["--resume", "auto"]
        )


def test_planar_defaults_come_from_the_capture_and_the_mounting_is_measured(
    rendered, tmp_path
):
    """``--frontend planar`` alone must be a working configuration: the frame
    gap, pair-interval cap, warm-up, search radius, loss and camera mounting
    are all derived from this capture when not given."""

    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(
        planar_argv(
            rendered, run_dir, camera_mounting=None, frame_gap=None, warmup=None,
            correlation_radius=None, velocity_loss=None,
        )
    ) == 0
    saved = load_checkpoint(run_dir / "last.pt", map_location="cpu")["args"]
    # 20 Hz images, 1 s baseline.
    assert saved["frame_gap"] == 20
    assert saved["max_frame_gap_s"] == pytest.approx(1.5)
    # 1.0 s pair + 0.35 s latency at 100 Hz, plus 5.
    assert saved["warmup"] == 140
    assert saved["correlation_radius"] == 3
    assert saved["velocity_loss"] == "simple"
    # The renderer's mount, measured from the images.
    assert np.allclose(saved["camera_from_body"], NADIR_MOUNTINGS["right_forward"], atol=0.03)


def test_the_original_frontend_keeps_its_historical_defaults():
    args = train_fixedwing_vo.build_parser().parse_args(["--dataset", "ignored"])
    chosen = train_fixedwing_vo.resolve_frontend_defaults(
        args, frame_interval_s=0.05, tick_interval_s=0.01
    )
    assert chosen == []
    assert (args.frame_gap, args.warmup, args.correlation_radius, args.velocity_loss) == (
        1, 20, 4, "nll"
    )
    assert args.max_frame_gap_s is None


def test_the_trainer_refuses_residual_without_planar(rendered, tmp_path):
    with pytest.raises(SystemExit, match="geometric_residual needs --frontend planar"):
        train_fixedwing_vo.main(
            planar_argv(rendered, tmp_path / "b", frontend="mamba_correlation",
                        velocity_mode="geometric_residual")
        )


def test_patience_stops_a_run_that_stopped_improving(rendered, tmp_path, monkeypatch):
    original = train_fixedwing_vo.evaluate
    calls = {"n": 0}

    def worsening(*args, **kwargs):
        metrics = original(*args, **kwargs)
        calls["n"] += 1
        metrics["vel_rmse"] = 1.0 + calls["n"]
        return metrics

    monkeypatch.setattr(train_fixedwing_vo, "evaluate", worsening)
    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(
        planar_argv(rendered, run_dir, epochs="6", patience="2")
    ) == 0
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # Epoch 1 is the best; 2 and 3 fail to improve; the run stops after 3.
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3]
    assert load_checkpoint(run_dir / "best.pt", map_location="cpu")["epoch"] == 1


def test_photometric_jitter_is_the_identity_at_zero_and_bounded_otherwise():
    image = torch.rand(4, 3, 8, 8)
    identity = train_fixedwing_vo.photometric_parameters(4, 3, 0.0, device=torch.device("cpu"))
    assert torch.allclose(
        train_fixedwing_vo.apply_photometric(image, identity[:, 0]), image.clamp(1e-6, 1.0), atol=1e-6
    )
    jittered = train_fixedwing_vo.apply_photometric(
        image,
        train_fixedwing_vo.photometric_parameters(4, 3, 0.2, device=torch.device("cpu"))[:, 1],
    )
    assert float(jittered.min()) >= 0.0 and float(jittered.max()) <= 1.0
    assert not torch.allclose(jittered, image)


def test_warmup_ramps_then_hands_over_to_the_cosine():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    schedule = train_fixedwing_vo.build_schedule(optimizer, 10, 2)
    rates = []
    for _ in range(4):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        schedule.step()
    assert rates[0] == pytest.approx(0.5)
    assert rates[1] == pytest.approx(1.0)
    assert rates[2] < 1.0 and rates[3] < rates[2]
    plain = train_fixedwing_vo.build_schedule(torch.optim.SGD([parameter], lr=1.0), 10, 0)
    assert isinstance(plain, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_the_motion_budget_recommends_a_gap_that_moves_the_ground_enough(rendered, tmp_path):
    output = tmp_path / "budget.json"
    assert check_motion_budget.main([
        "--dataset", str(rendered),
        "--calibration", str(rendered / "calibration.json"),
        "--image-size", str(HEIGHT), str(WIDTH),
        "--camera-mounting", "right_forward",
        "--target-cells", "2",
        "--gaps", "1,2,5,10,20",
        "--output", str(output),
    ]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    rows = {row["gap"]: row for row in report["gaps"]}
    # 20 m/s at 150 m through f = 329 px: 20 * dt / 150 * 329 / 8 cells.
    expected = 20.0 * 0.5 / 150.0 * 329.0 / 8.0
    assert rows[10]["translation_cells"][1] == pytest.approx(expected, rel=0.15)
    assert rows[20]["translation_cells"][1] > rows[10]["translation_cells"][1]
    assert report["recommendation"]["frame_gap"] == 10
    assert report["images_rate_hz"] == pytest.approx(20.0, rel=0.01)


def test_the_mounting_tool_recovers_the_rendered_mount(rendered, tmp_path):
    output = tmp_path / "mounting.json"
    assert estimate_camera_mounting.main([
        "--dataset", str(rendered),
        "--calibration", str(rendered / "calibration.json"),
        "--image-size", str(HEIGHT), str(WIDTH),
        "--frame-gap", "6", "--pairs", "80", "--max-rotation-deg", "3",
        "--output", str(output),
    ]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["best_mounting"] == "right_forward"
    # Every other nadir mount predicts motion in a different direction, so it
    # cannot come close; the right one matches to about a percent once the
    # rotation is taken out (without de-rotation this read 17 deg and 1.29).
    others = [v for k, v in report["relative_error_per_mounting"].items() if k != "right_forward"]
    assert report["relative_error_per_mounting"]["right_forward"] < 0.05 < min(others)
    assert abs(report["yaw_misalignment_deg"]) < 1.0
    assert report["scale_ratio_median"] == pytest.approx(1.0, abs=0.03)


def test_stratified_errors_split_by_condition_and_refuse_thin_bins():
    from vio.utils.stratify import flight_conditions, stratified_errors

    ticks = 400
    target = np.zeros((ticks, 3))
    target[:, 0] = 20.0
    predicted = target.copy()
    aiding = np.zeros((ticks, 9))
    aiding[:, 1] = 1.0  # cos roll: level
    aiding[:, 3] = 1.0
    # A turn in the second half, where the error is made worse on purpose.
    aiding[200:, 7] = np.radians(12.0)
    predicted[200:, 1] += 2.0
    predicted[:10] = np.nan  # unscored warm-up ticks
    conditions = flight_conditions(aiding, np.full(ticks, np.log(200.0)), target)
    report = stratified_errors(predicted, target, conditions, min_ticks=20)
    by_turn = {row["low"]: row for row in report["turn_rate_deg_s"]}
    assert by_turn[0.0]["ticks"] == 190 and by_turn[0.0]["vel_rmse"] == pytest.approx(0.0)
    assert by_turn[10.0]["ticks"] == 200 and by_turn[10.0]["vel_rmse_y"] == pytest.approx(2.0)
    assert np.isnan(by_turn[2.0]["vel_rmse"]) and by_turn[2.0]["ticks"] == 0


def test_output_on_pairs_tiles_the_pairs_and_scores_one_output_per_pair(rendered, tmp_path):
    """--output-on-pairs with --frame-gap 10 at 20 Hz: pairs (0,10), (10,20),
    ... - one measurement and one scored output every 0.5 s - carried through
    training, the checkpoint and the evaluator."""

    from vio.data.image_pairs import VisualPairSource

    times = np.arange(0.0, 20.0, 0.01) + 1000.0
    source = VisualPairSource(rendered, times, frame_gap=10, pair_stride=10)
    assert source.plan.first_index[:3].tolist() == [0, 10, 20]
    assert source.plan.second_index[:3].tolist() == [10, 20, 30]
    assert np.allclose(np.diff(source.plan.exposure_t1_s[:5]), 0.5)

    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(
        planar_argv(rendered, run_dir, output_on_pairs=True, warmup=None)
    ) == 0
    checkpoint = load_checkpoint(run_dir / "last.pt", map_location="cpu")
    saved = checkpoint["args"]
    assert saved["output_on_pairs"] is True and saved["pair_stride"] == 10
    assert checkpoint["fingerprint"]["contract"]["pair_stride"] == 10
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        row = list(csv.DictReader(handle))[0]
    assert np.isfinite(float(row["val_vel_rmse"]))

    output = tmp_path / "eval.json"
    assert evaluate_velocity_horizons.main([
        str(run_dir / "best.pt"), "--dataset", str(rendered),
        "--splits", "validation", "--horizons", "0.04",
        "--no-plots", "--no-progress", "--output", str(output),
    ]) == 0
    horizon = json.loads(output.read_text(encoding="utf-8"))["splits"]["validation"]["horizons"]["h0.04m"]
    assert horizon["fits"], horizon.get("skipped")
    # 2.4 s, of which ~1.05 s after the warm-up, at one output per 0.5 s:
    # two or three scored ticks, not ~100.
    assert 0 < horizon["scored_ticks"] <= 4
    assert np.isfinite(horizon["vel_rmse"]) and np.isfinite(horizon["pos_error_final"])


def test_held_outputs_fill_until_the_next_emission_and_carry_across_blocks():
    from vio.models.velocity_horizons import _hold_emitted

    predicted = torch.arange(6, dtype=torch.float32).view(1, 6, 1).expand(1, 6, 3).clone()
    fired = torch.tensor([[False, True, False, False, True, False]])
    held, carry = _hold_emitted(predicted, fired, None)
    assert held[0, :, 0].tolist() == [0, 1, 1, 1, 4, 4]
    assert carry[0, 0] == 4
    later, carry2 = _hold_emitted(predicted + 10, torch.tensor([[False, False, True, False, False, False]]), carry)
    assert later[0, :, 0].tolist() == [4, 4, 12, 12, 12, 12]
    assert carry2[0, 0] == 12


def test_min_delta_counts_gains_since_the_last_real_improvement(rendered, tmp_path, monkeypatch):
    """0.005 better every epoch with --min-delta 0.01: every second epoch the
    accumulated gain crosses 0.01, so patience 3 must never run out (the
    earlier rule compared each epoch with the previous best and stopped)."""

    original = train_fixedwing_vo.evaluate
    calls = {"n": 0}

    def slowly_better(*args, **kwargs):
        metrics = original(*args, **kwargs)
        metrics["vel_rmse"] = 1.0 - 0.005 * calls["n"]
        calls["n"] += 1
        return metrics

    monkeypatch.setattr(train_fixedwing_vo, "evaluate", slowly_better)
    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(
        planar_argv(rendered, run_dir, epochs="5", patience="3", min_delta="0.01")
    ) == 0
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 5


def test_a_resume_may_add_a_learning_rate_warmup(rendered, tmp_path):
    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(planar_argv(rendered, run_dir)) == 0
    assert train_fixedwing_vo.main(
        planar_argv(rendered, run_dir, epochs="2", lr_warmup_epochs="1") + ["--resume", "auto"]
    ) == 0
    assert load_checkpoint(run_dir / "last.pt", map_location="cpu")["epoch"] == 2
