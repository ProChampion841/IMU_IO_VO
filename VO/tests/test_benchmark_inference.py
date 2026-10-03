"""tools/benchmark_inference.py: the report carries the I/O contract and a budget.

Runs at a tiny image size so it is a smoke test of the plumbing, not a timing.
"""

from __future__ import annotations

import json

from tools.benchmark_inference import main


def test_benchmark_reports_io_parameters_and_budget(tmp_path):
    output = tmp_path / "bench.json"
    code = main([
        "--frontend", "mamba_correlation", "--image-size", "96", "128",
        "--threads", "1", "--warmup", "0", "--iterations", "1",
        "--step-iterations", "2", "--no-onnx", "--output", str(output),
    ])
    assert code == 0
    (report,) = json.loads(output.read_text())

    frontend_inputs = {row["name"]: row["shape"] for row in report["io"]["frontend_inputs"]}
    assert frontend_inputs["image0"] == [1, 1, 96, 128]
    temporal_inputs = {row["name"]: row["shape"] for row in report["io"]["temporal_inputs"]}
    assert temporal_inputs["aiding"] == [1, 9]
    outputs = {row["name"]: row["shape"] for row in report["io"]["temporal_outputs"]}
    assert outputs["predicted_velocity"] == [1, 3]

    assert report["parameters_total"]["temporal"] > 0
    timing = report["timing"]["threads_1"]
    assert timing["torch_frontend_per_pair"]["median_ms"] > 0
    assert timing["torch_step_per_tick"]["median_ms"] > 0

    budget = report["budget"]
    # Trainer defaults for the original frontend: a pair ends on every frame.
    assert budget["pairs_per_s"] == 20.0
    row = budget["per_threads"]["threads_1"]
    assert row["engine"] == "torch"
    assert row["tick_duty"] == budget["telemetry_hz"] * timing["torch_step_per_tick"]["median_ms"] / 1000.0
