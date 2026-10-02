"""Regressions for the evaluator's reporting layer.

* a non-finite tick anywhere in a block must not poison the leg's sums or hide
  its maximum (NaN * 0 is NaN);
* the time of each maximum, and a series' time axis, come from the telemetry
  clock, not tick index x median interval (wrong after any gap);
* the stratified table has a bin for the WHOLE body rate, not only yaw rate;
* a pre-split run is scored on its validation folder by default, and scoring
  its training folder is never reported as held out;
* --disable-visual-input is refused for an --output-on-pairs checkpoint, which
  has no tick to score without pairs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import tools.evaluate_velocity_horizons as evaluate_velocity_horizons
import tools.train_fixedwing_vo as train_fixedwing_vo
import vio.models.velocity_horizons as vh
from vio.models.vision_mamba_vo import VisionMambaVO
from vio.utils.stratify import flight_conditions

from test_train_fixedwing_vo_integration import (  # noqa: E402
    TICKS,
    _train_argv,
    _write_calibration,
    _write_presplit_flight,
)


def test_a_non_finite_tick_does_not_poison_the_leg():
    stats = vh._LegStats(1, torch.device("cpu"))
    predicted = torch.tensor([[[1.0, 0, 0], [2.0, 0, 0], [float("nan"), 0, 0], [5.0, 0, 0]]])
    target = torch.tensor([[[1.0, 0, 0]] * 4])
    stats.update(predicted, target, torch.ones(1, 4))
    legs = stats.per_leg()

    assert legs["scored_ticks"][0] == 3
    assert legs["vel_rmse"][0] == pytest.approx(np.sqrt((0.0 + 1.0 + 16.0) / 3.0))
    assert legs["vel_max_error"][0] == pytest.approx(4.0)
    assert legs["vel_max_error_tick"][0] == 3
    assert np.allclose(legs["axis_rmse"][0], [np.sqrt(17.0 / 3.0), 0.0, 0.0])
    assert legs["vel_dir_rmse"][0] == pytest.approx(0.0)


def test_the_time_of_a_maximum_is_read_off_the_telemetry_clock():
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    dim = 8
    model = VisionMambaVO(visual_dim=dim, aiding_dim=8, fusion_dim=8,
                          velocity_mode="geometric_residual")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    model.eval()
    ticks = 4000
    times = 50.0 + np.cumsum(np.full(ticks, 0.01))
    times[2000:] += 0.5  # one 0.5 s telemetry gap, halfway
    ready = np.arange(87, ticks, 50)
    tokens = vh.SpanTokens(
        tick=ready, token=torch.randn(ready.size, dim), quality=torch.rand(ready.size, 1),
        visual_dim=dim, velocity=torch.randn(ready.size, 3) + torch.tensor([20.0, 0, 0]),
    )
    _, whole = vh.run_span_horizons(
        model,
        aiding=(rng.standard_normal((ticks, 9)) * 0.3).astype(np.float32),
        log_altitude=np.full(ticks, np.log(200.0), np.float32),
        target_velocity=(np.array([20.0, 0, 0]) + rng.standard_normal((ticks, 3))).astype(np.float32),
        times_s=times, span=(0, ticks), tokens=tokens, deployment_latency_s=0.35,
        horizons_minutes=(0.5,), warmup_ticks=135, block_ticks=333,
        collect_series=True, output_on_pairs=True,
    )

    for stem in ("vel_max_error", "vel_dir_max_error"):
        tick = whole[f"{stem}_tick"]
        assert whole[f"{stem}_time_s"] == pytest.approx(times[tick] - times[0])
        assert whole[f"{stem}_flight_time_s"] == pytest.approx(times[tick])
    axis = whole["series"]["time_since_start_s"]
    assert np.allclose(axis, times - times[0])
    assert axis[-1] == pytest.approx(ticks * 0.01 - 0.01 + 0.5)


def test_the_rotation_rate_bin_sees_a_pull_up_the_yaw_rate_misses():
    ticks = 4
    aiding = np.zeros((ticks, 9))
    aiding[:, 1] = aiding[:, 3] = 1.0  # level: cos roll = cos pitch = 1
    aiding[:, 6] = np.radians(5.0)  # q: a 5 deg/s pull-up, no yaw
    conditions = flight_conditions(aiding, np.full(ticks, np.log(150.0)), np.ones((ticks, 3)))
    assert np.allclose(conditions["turn_rate_deg_s"], 0.0)
    assert np.allclose(conditions["rotation_rate_deg_s"], 5.0)


def _presplit_run(tmp_path, **overrides):
    train_root = _write_presplit_flight(tmp_path / "train_flight", ticks=TICKS, forward_speed=25.0)
    validation_root = _write_presplit_flight(
        tmp_path / "validation_flight", ticks=300, forward_speed=5.0
    )
    _write_calibration(train_root / "calibration.json")
    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(_train_argv(
        train_root, run_dir, validation_dataset=str(validation_root), **overrides,
    )) == 0
    return train_root, validation_root, run_dir


def _evaluate(run_dir, output, *extra):
    assert evaluate_velocity_horizons.main([
        str(run_dir / "best.pt"), "--splits", "validation",
        "--no-plots", "--no-progress", "--output", str(output), *extra,
    ]) == 0
    return json.loads(output.read_text(encoding="utf-8"))


def test_a_presplit_run_is_scored_on_its_validation_folder_and_never_mislabelled(tmp_path):
    train_root, validation_root, run_dir = _presplit_run(tmp_path)

    default = _evaluate(run_dir, tmp_path / "default.json")
    assert default["dataset"] == str(validation_root)
    assert default["splits"]["validation"]["held_out"] is True

    training = _evaluate(run_dir, tmp_path / "train.json", "--dataset", str(train_root))
    assert training["splits"]["validation"]["held_out"] is False


def test_the_visual_blind_floor_is_refused_for_an_output_on_pairs_checkpoint(tmp_path):
    _, validation_root, run_dir = _presplit_run(tmp_path, output_on_pairs=True)
    with pytest.raises(SystemExit, match="output-on-pairs"):
        evaluate_velocity_horizons.main([
            str(run_dir / "best.pt"), "--dataset", str(validation_root),
            "--disable-visual-input", "--no-plots", "--no-progress",
            "--output", str(tmp_path / "blind.json"),
        ])
