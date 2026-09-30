"""Condition-segment splitting has been disabled by request: the VO trainer
and its standalone evaluator now always use the chronological start-to-end
cut, 60%/20%/20% by default, regardless of ``--split-manifest`` or a
``dataset.config`` sitting in the dataset root. These tests pin that: the
split is computed the same way no matter what a manifest says, the three
phases stay disjoint with the usual one-window gap, and an OLD checkpoint
that really was split by condition segments is refused rather than silently
rescored against a different split and reported as held out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tools.evaluate_velocity_horizons import _resolve_ranges as evaluator_resolve_ranges
from tools.train_fixedwing_vo import resolve_ranges


def _args(*, split_manifest=None, window_length=10, train_fraction=0.6,
          validation_fraction=0.2) -> argparse.Namespace:
    args = argparse.Namespace()
    args.split_manifest = split_manifest
    args.window_length = window_length
    args.train_fraction = train_fraction
    args.validation_fraction = validation_fraction
    return args


def _write_segment_manifest(tmp_path: Path) -> Path:
    """A real, valid condition-segment manifest - to prove it is IGNORED."""

    flight = tmp_path / "flight"
    (flight / "images").mkdir(parents=True)
    (flight / "flight.csv").write_text("Time\n0\n", encoding="utf-8")

    manifest = tmp_path / "dataset.config"
    manifest.write_text(
        json.dumps(
            {
                "flight": str(flight),
                "segments": {
                    "seg_0001": [0, 1000],
                    "seg_0002": [1000, 2000],
                    "seg_0003": [2000, 6000],
                },
                "train": ["seg_0001", "seg_0003"],
                "validation": ["seg_0002"],
                "test": ["seg_0002"],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _overlap(a, b) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def test_resolve_ranges_is_chronological_with_no_manifest():
    ranges = resolve_ranges(_args(), total=6000)
    assert ranges["train"] == (0, 3600)
    assert ranges["validation"] == (3610, 4800)
    assert ranges["test"] == (4810, 6000)


def test_resolve_ranges_takes_no_manifest_argument():
    """--split-manifest is gone, so a manifest cannot change the split at all.

    It used to be accepted and then ignored, which meant a command line asking
    for a condition-segment split got a chronological one and a printed note.
    The flag is now removed outright, and pre-splitting is done on disk by
    tools/split_dataset.py, where the split is three folders rather than a
    promise.
    """

    import argparse

    from tools.train_fixedwing_vo import build_parser

    parsed = build_parser().parse_args(
        ["--dataset", "d", "--run-dir", "r"]
    )
    assert not hasattr(parsed, "split_manifest")
    assert hasattr(parsed, "validation_dataset")
    assert hasattr(parsed, "test_dataset")

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--dataset", "d", "--run-dir", "r", "--split-manifest", "x"]
        )


def test_resolve_ranges_respects_custom_fractions():
    ranges = resolve_ranges(
        _args(train_fraction=0.5, validation_fraction=0.3, window_length=100),
        total=10000,
    )
    assert ranges["train"] == (0, 5000)
    assert ranges["validation"] == (5100, 8000)
    assert ranges["test"] == (8100, 10000)


def test_the_three_phases_never_overlap():
    ranges = resolve_ranges(_args(), total=100000)
    assert _overlap(ranges["train"], ranges["validation"]) == 0
    assert _overlap(ranges["train"], ranges["test"]) == 0
    assert _overlap(ranges["validation"], ranges["test"]) == 0


# ---------------------------------------------------------------------------
# the evaluator must reconstruct the same chronological split
# ---------------------------------------------------------------------------


def test_evaluator_reconstructs_the_same_chronological_ranges():
    saved = {"train_fraction": 0.6, "validation_fraction": 0.2, "window_length": 10}
    ranges = evaluator_resolve_ranges(saved, total=6000)
    assert ranges["train"] == (0, 3600)
    assert ranges["validation"] == (3610, 4800)
    assert ranges["test"] == (4810, 6000)
    assert ranges["full"] == (0, 6000)


def test_evaluator_defaults_match_the_trainers_new_defaults():
    """Old checkpoints with no recorded fractions still get 60/20/20, not the

    trainer's old 70/15 default - the two must never silently disagree.
    """

    ranges = evaluator_resolve_ranges({}, total=6000)
    assert ranges["train"] == (0, 3600)
    assert ranges["validation"][0] > ranges["train"][1]


def test_evaluator_scores_a_presplit_run_whole():
    """A pre-split run has no fractions to reconstruct.

    Each phase was its own directory, so the directory handed to the evaluator
    IS the split. Carving a sub-range out of it would invent held-out ticks
    that the run never held out.
    """

    ranges = evaluator_resolve_ranges({"validation_dataset": "/somewhere/val"}, total=6000)
    assert ranges["full"] == (0, 6000)
    assert ranges["train"] == ranges["validation"] == ranges["test"] == (0, 6000)


def test_evaluator_still_reconstructs_fractions_for_a_fraction_run():
    """A run that cut one capture chronologically is scored on those ranges."""

    saved = {"train_fraction": 0.6, "validation_fraction": 0.2, "window_length": 10}
    ranges = evaluator_resolve_ranges(saved, total=6000)
    assert ranges["train"] == (0, 3600)
    assert ranges["validation"] == (3610, 4800)
    assert _overlap(ranges["train"], ranges["validation"]) == 0
