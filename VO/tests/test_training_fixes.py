"""Regressions for the trainer's selection, resume and bookkeeping.

* a non-finite prediction is a failure (infinite error), never a skipped tick;
* checkpoints are written atomically;
* a resume does not let a worse epoch overwrite the better best.pt on disk;
* a resume applies a changed learning rate / weight decay instead of keeping
  the checkpoint's silently;
* --disable-visual-input with --output-on-pairs is refused (nothing to score).
"""

from __future__ import annotations

import math

import pytest
import torch

import tools.train_fixedwing_vo as train_fixedwing_vo
from vio.utils.checkpoint_io import load_checkpoint
from vio.utils.velocity_metrics import RunningVelocityStats, masked_velocity_stats

from test_train_fixedwing_vo_integration import _train_argv, flight  # noqa: F401,E402


def test_a_non_finite_prediction_is_scored_as_a_failure_not_dropped():
    target = torch.tensor([[[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]]])
    mask = torch.ones(1, 3)
    finite = RunningVelocityStats()
    finite.update(masked_velocity_stats(torch.tensor([[[1.0, 0, 0], [2.0, 0, 0], [9.0, 0, 0]]]),
                                        target, mask))
    blown_up = RunningVelocityStats()
    blown_up.update(masked_velocity_stats(
        torch.tensor([[[1.0, 0, 0], [2.0, 0, 0], [float("nan"), 0, 0]]]), target, mask
    ))
    assert math.isfinite(finite.metrics()["vel_rmse"])
    # Dropping the NaN tick would have scored 0.71 - better than the finite model.
    assert blown_up.metrics()["vel_rmse"] == float("inf")


def test_an_unusable_label_still_removes_its_tick():
    predicted = torch.tensor([[[1.0, 0, 0], [3.0, 0, 0]]])
    target = torch.tensor([[[1.0, 0, 0], [float("nan"), 0, 0]]])
    stats = RunningVelocityStats()
    stats.update(masked_velocity_stats(predicted, target, torch.ones(1, 2)))
    assert stats.metrics()["vel_rmse"] == pytest.approx(0.0)


def test_checkpoints_are_written_atomically(tmp_path):
    path = tmp_path / "last.pt"
    train_fixedwing_vo.save_checkpoint_atomic({"epoch": 1}, path)
    train_fixedwing_vo.save_checkpoint_atomic({"epoch": 2}, path)
    assert torch.load(path)["epoch"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last.pt"]


def test_a_resume_keeps_the_better_best_pt_on_disk(flight, tmp_path, capsys):  # noqa: F811
    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(_train_argv(flight, run_dir)) == 0
    # best.pt now stands for a better epoch than the one being resumed - as
    # after resuming from an older epochs/epoch_XXXX.pt.
    best = load_checkpoint(run_dir / "best.pt", map_location="cpu")
    best["best_val_score"] = 0.0
    best["epoch"] = 99
    torch.save(best, run_dir / "best.pt")

    assert train_fixedwing_vo.main(
        _train_argv(flight, run_dir, epochs="2", resume="auto")
    ) == 0
    assert "kept as the bar to beat" in capsys.readouterr().out
    assert load_checkpoint(run_dir / "best.pt", map_location="cpu")["epoch"] == 99


def test_a_resume_applies_a_changed_learning_rate_and_weight_decay(flight, tmp_path):  # noqa: F811
    run_dir = tmp_path / "run"
    assert train_fixedwing_vo.main(_train_argv(flight, run_dir, learning_rate="3e-4")) == 0
    assert train_fixedwing_vo.main(_train_argv(
        flight, run_dir, epochs="2", resume="auto",
        learning_rate="1e-4", weight_decay="5e-4",
    )) == 0
    group = load_checkpoint(run_dir / "last.pt", map_location="cpu")["optimizer"]["param_groups"][0]
    assert group["initial_lr"] == pytest.approx(1e-4)
    assert group["weight_decay"] == pytest.approx(5e-4)


def test_the_visual_blind_floor_is_refused_with_output_on_pairs(flight, tmp_path):  # noqa: F811
    with pytest.raises(SystemExit, match="scores nothing"):
        train_fixedwing_vo.main(_train_argv(
            flight, tmp_path / "run", disable_visual_input=True, output_on_pairs=True,
        ))
