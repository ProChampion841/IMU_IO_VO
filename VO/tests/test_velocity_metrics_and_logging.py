"""Velocity metrics (magnitude + direction), the epoch CSV, and the
evaluation figures - the monitoring surface added to training and evaluation."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
import torch

from vio.utils.evaluation_plots import save_split_plots
from vio.utils.metrics_csv import EpochMetricsCsv
from vio.utils.velocity_metrics import (
    METRIC_NAMES,
    UNCERTAINTY_METRIC_NAMES,
    RunningVelocityStats,
    RunningVelocityUncertaintyStats,
    direction_angles_deg,
    masked_velocity_uncertainty_stats,
    masked_velocity_stats,
    velocity_error_metrics,
    velocity_uncertainty_metrics,
)


def test_direction_angles_are_geometric():
    prediction = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
    target = np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    angles = direction_angles_deg(prediction, target)
    assert angles == pytest.approx([0.0, 90.0, 180.0])


def test_near_zero_vectors_are_dropped_not_scored():
    prediction = np.array([[1e-9, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert direction_angles_deg(prediction, target).shape == (1,)


def test_velocity_error_metrics_known_values():
    prediction = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    target = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # Zero targets carry no direction: magnitude metrics only.
    metrics = velocity_error_metrics(prediction, target)
    assert metrics["vel_rmse"] == pytest.approx(math.sqrt((9 + 16) / 2))
    assert metrics["vel_max_error"] == pytest.approx(4.0)
    assert math.isnan(metrics["vel_dir_rmse"])
    assert metrics["vel_rmse_x"] == pytest.approx(3.0 / math.sqrt(2.0))
    assert metrics["vel_rmse_y"] == pytest.approx(4.0 / math.sqrt(2.0))
    assert metrics["vel_rmse_z"] == pytest.approx(0.0)
    assert metrics["vel_bias_x"] == pytest.approx(1.5)
    assert metrics["vel_bias_y"] == pytest.approx(2.0)
    assert metrics["vel_bias_z"] == pytest.approx(0.0)


def test_running_stats_match_the_single_shot_metrics():
    """Batched accumulation (as training aggregates) must equal computing the
    metrics once over every masked tick - across batches and micro-batches."""
    torch.manual_seed(0)
    predicted = torch.randn(6, 40, 3) * 5
    target = torch.randn(6, 40, 3) * 5
    mask = (torch.rand(6, 40) > 0.3).float()

    running = RunningVelocityStats()
    for begin in (0, 2, 4):  # three "batches" of two windows
        running.update(
            masked_velocity_stats(
                predicted[begin : begin + 2],
                target[begin : begin + 2],
                mask[begin : begin + 2],
            )
        )
    flat = mask.reshape(-1) > 0
    reference = velocity_error_metrics(
        predicted.reshape(-1, 3)[flat], target.reshape(-1, 3)[flat]
    )
    for name in METRIC_NAMES:
        assert running.metrics()[name] == pytest.approx(reference[name], rel=1e-6), name


def test_uncertainty_metrics_are_per_axis_and_exact_across_batches():
    prediction = torch.tensor(
        [[[1.0, 2.0, 3.0]], [[2.0, 4.0, 6.0]]]
    )
    target = torch.zeros_like(prediction)
    predicted_std = torch.tensor(
        [[[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]]
    )
    mask = torch.ones(2, 1)
    running = RunningVelocityUncertaintyStats()
    for index in range(2):
        running.update(
            masked_velocity_uncertainty_stats(
                prediction[index : index + 1],
                target[index : index + 1],
                predicted_std[index : index + 1],
                mask[index : index + 1],
            )
        )
    reference = velocity_uncertainty_metrics(
        prediction.reshape(-1, 3),
        target.reshape(-1, 3),
        predicted_std.reshape(-1, 3),
    )
    for name in UNCERTAINTY_METRIC_NAMES:
        assert running.metrics()[name] == pytest.approx(reference[name])
    assert reference["vel_pred_std_x"] == pytest.approx(1.0)
    assert reference["vel_pred_std_y"] == pytest.approx(2.0)
    assert reference["vel_pred_std_z"] == pytest.approx(3.0)
    assert reference["vel_zscore_rmse_x"] == pytest.approx(math.sqrt(2.5))
    assert reference["vel_coverage_1sigma_x"] == pytest.approx(0.5)
    assert reference["vel_coverage_2sigma_x"] == pytest.approx(1.0)


def test_epoch_csv_writes_the_exact_columns(tmp_path):
    columns = ("epoch", "train_loss", "val_loss")
    log = EpochMetricsCsv(tmp_path / "metrics.csv", columns)
    log.append({"epoch": 1, "train_loss": 0.5, "val_loss": 0.25})
    log.append({"epoch": 2, "train_loss": 0.4})  # a missing value stays empty
    with (tmp_path / "metrics.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == list(columns)
    assert rows[1] == ["1", "0.500000", "0.250000"]
    assert rows[2] == ["2", "0.400000", ""]
    # Re-opening appends instead of truncating.
    EpochMetricsCsv(tmp_path / "metrics.csv", columns).append({"epoch": 3})
    with (tmp_path / "metrics.csv").open(newline="") as handle:
        assert len(list(csv.reader(handle))) == 4


def test_split_plots_are_written(tmp_path):
    ticks = 200
    times = 1000.0 + np.arange(ticks) * 0.01
    target_velocity = np.stack(
        [np.full(ticks, 20.0), np.sin(np.linspace(0, 4, ticks)), np.zeros(ticks)],
        axis=1,
    )
    predicted_velocity = target_velocity + 0.5
    target_position = np.cumsum(target_velocity, axis=0) * 0.01
    predicted_position = np.cumsum(predicted_velocity, axis=0) * 0.01
    metrics = velocity_error_metrics(predicted_velocity, target_velocity)
    written = save_split_plots(
        tmp_path / "plots",
        "validation",
        times,
        predicted_velocity,
        target_velocity,
        predicted_position,
        target_position,
        metrics,
    )
    assert [path.name for path in written] == [
        "validation_trajectory.png",
        "validation_velocity.png",
        "validation_velocity_error.png",
    ]
    for path in written:
        assert path.stat().st_size > 10_000  # a real rendered figure


def _direction_case(error_deg, log_kappa, batch=2, ticks=8):
    """A prediction tilted off the target by a known angle, at a fixed kappa."""

    import torch

    target = torch.zeros(batch, ticks, 3)
    target[..., 0] = 25.0
    unit = target / target.norm(dim=-1, keepdim=True)
    sideways = torch.zeros_like(unit)
    sideways[..., 1] = 1.0
    angle = torch.deg2rad(torch.tensor(float(error_deg)))
    direction = torch.nn.functional.normalize(
        unit * torch.cos(angle) + sideways * torch.sin(angle), dim=-1
    )
    prediction = {
        "predicted_velocity": direction * target.norm(dim=-1, keepdim=True),
        "predicted_direction": direction,
        "velocity_log_variance": torch.zeros(batch, ticks, 3),
        "direction_log_concentration": torch.full((batch, ticks), float(log_kappa)),
    }
    return prediction, target, torch.ones(batch, ticks)


def test_the_direction_term_reduces_to_the_old_one_at_unit_concentration():
    """kappa = 1 must give back kappa*(1-cos) - log kappa = (1 - cos)."""

    import math

    from tools.train_fixedwing_vo import velocity_loss

    for error_deg in (0.0, 2.0, 10.0):
        prediction, target, mask = _direction_case(error_deg, log_kappa=0.0)
        _, parts = velocity_loss(prediction, target, mask, direction_weight=0.5)
        expected = 1.0 - math.cos(math.radians(error_deg))
        assert parts["direction"] == pytest.approx(expected, abs=1e-6)


def test_the_direction_term_is_minimised_at_the_concentration_that_fits():
    """The learned kappa should track the error, not drift to a clamp.

    d/dkappa [kappa*(1-cos) - log kappa] = 0 at kappa = 1/(1-cos), so a head
    that reports its true spread sits at the minimum. This is what makes the
    number readable afterwards: kappa is an error estimate, not a free weight.
    """

    import math

    from tools.train_fixedwing_vo import velocity_loss

    for error_deg in (1.0, 5.0, 20.0):
        analytic = 1.0 / (1.0 - math.cos(math.radians(error_deg)))
        best = min(
            (
                velocity_loss(
                    *_direction_case(error_deg, log_kappa=step / 4.0),
                    direction_weight=0.5,
                )[1]["direction"],
                step / 4.0,
            )
            for step in range(-16, 49)
        )
        assert best[1] == pytest.approx(math.log(analytic), abs=0.3)


def test_a_degenerate_direction_target_drives_the_concentration_to_its_clamp():
    """The failure mode a capture with no crab and no incidence produces.

    With (1 - cos) identically zero the term is -log kappa, so every step makes
    kappa larger and the head reports maximum confidence forever. The clamp
    bounds it, but the number means nothing - which is why this is a property
    of the DATA to fix, not of the loss.
    """

    import torch

    from tools.train_fixedwing_vo import velocity_loss

    prediction, target, mask = _direction_case(0.0, log_kappa=0.0)
    prediction["direction_log_concentration"] = torch.zeros_like(
        prediction["direction_log_concentration"], requires_grad=True
    )
    loss, _ = velocity_loss(prediction, target, mask, direction_weight=0.5)
    loss.backward()
    gradient = prediction["direction_log_concentration"].grad
    # Uniformly negative: nothing ever pushes back, so log kappa only rises.
    assert torch.all(gradient < 0.0)
