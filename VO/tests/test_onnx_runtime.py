"""ONNX export + standalone runtime for the current training setup.

One small planar model, trained the way the real runs are now configured -
RGB, a 1 s pair every 0.5 s (--frame-gap 20 --pair-stride 10), random pair
phase, and yaw read from its own column (--attitude-columns NavEulX NavEulY
WWMYaw_RAD) - exported once and then:

* replayed through the runtime against PyTorch (same delivery ticks, same
  velocities), with the yaw column carried by the export's metadata;
* fed a non-finite telemetry row, which must be skipped rather than poison the
  recurrent state for the rest of the flight;
* handed a frame of the wrong size, which must be refused as training does.
"""

from __future__ import annotations

import csv
import json
import math
import os

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import tools.export_onnx as export_onnx  # noqa: E402
import tools.make_synthetic_flight as make_synthetic_flight  # noqa: E402
import tools.onnx_inference as onnx_inference  # noqa: E402
import tools.train_fixedwing_vo as train_fixedwing_vo  # noqa: E402

from test_planar_training import HEIGHT, WIDTH, planar_argv  # noqa: E402

YAW_OFFSET_RAD = math.radians(3.0)  # e.g. magnetic declination: must not matter


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("onnx_runtime")
    rendered = root / "rendered"
    assert make_synthetic_flight.main([
        "--output", str(rendered), "--duration-s", "20",
        "--altitude-m", "150", "--speed-m-s", "20",
        "--image-width", str(WIDTH), "--image-height", str(HEIGHT),
        "--focal-px", "329", "--ground-metres-per-texel", "0.3",
        "--texture-size", "2048",
    ]) == 0
    # The same flight with yaw also logged as its own column, offset by a constant.
    flight = root / "flight"
    flight.mkdir()
    os.symlink(rendered / "images", flight / "images")
    (flight / "calibration.json").write_text(
        (rendered / "calibration.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with (rendered / "flight.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    yaw = rows[0].index("NavEulZ")
    rows[0].append("WWMYaw_RAD")
    for row in rows[1:]:
        value = float(row[yaw]) + YAW_OFFSET_RAD
        row.append(repr((value + math.pi) % (2.0 * math.pi) - math.pi))
    with (flight / "flight.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)

    run_dir = root / "run"
    argv = planar_argv(
        flight, run_dir, frame_gap="20", output_on_pairs=True, pair_stride="10",
        random_pair_phase=True, max_visual_events="14", color=True,
    ) + ["--attitude-columns", "NavEulX", "NavEulY", "WWMYaw_RAD"]
    assert train_fixedwing_vo.main(argv) == 0
    export_dir = root / "onnx"
    export_onnx.export_checkpoint(run_dir / "best.pt", export_dir, verify_ticks=60)
    return flight, run_dir, export_dir


def test_the_export_carries_the_training_setup(exported):
    _, _, export_dir = exported
    meta = json.loads((export_dir / "vo_onnx.json").read_text(encoding="utf-8"))
    assert meta["dataset_settings"]["attitude_columns"] == ["NavEulX", "NavEulY", "WWMYaw_RAD"]
    assert (meta["timing"]["frame_gap"], meta["timing"]["pair_stride"]) == (20, 10)
    assert meta["timing"]["output_on_pairs"] is True


def test_the_runtime_replays_the_flight_exactly_as_pytorch(exported):
    flight, run_dir, export_dir = exported
    assert onnx_inference.main([
        str(export_dir), "--dataset", str(flight),
        "--checkpoint", str(run_dir / "best.pt"), "--no-progress",
    ]) == 0


def test_a_non_finite_telemetry_row_is_skipped_not_carried(exported):
    _, _, export_dir = exported
    runtime = onnx_inference.VOOnnxRuntime(export_dir)
    outputs = []
    for tick in range(30):
        t = 100.0 + 0.01 * tick
        roll = float("nan") if tick == 10 else 0.05
        outputs.append(runtime.add_telemetry(t, roll, 0.02, 0.3, 150.0))
    assert outputs[10]["skipped"] and not np.isfinite(outputs[10]["velocity"]).any()
    assert runtime.stats["skipped_rows"] == 1
    for out in outputs[11:]:
        assert not out["skipped"] and np.isfinite(out["velocity"]).all()
    assert all(np.isfinite(value).all() for value in runtime.state.values())


def test_a_frame_of_the_wrong_size_is_refused(exported):
    _, _, export_dir = exported
    meta = json.loads((export_dir / "vo_onnx.json").read_text(encoding="utf-8"))
    preprocess = onnx_inference.ImagePreprocessor(meta)
    height, width = (int(v) for v in meta["calibration"]["native_size"])
    assert preprocess(np.zeros((height, width, 3), np.uint8)).shape == (3, HEIGHT, WIDTH)
    with pytest.raises(ValueError, match="calibration"):
        preprocess(np.zeros((height // 2, width // 2, 3), np.uint8))
