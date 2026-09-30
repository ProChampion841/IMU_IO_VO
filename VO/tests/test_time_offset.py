import csv

import numpy as np
import pytest
from PIL import Image

from vio.data.images import (
    nearest_indices,
    numeric_image_manifest,
    resolve_time_offsets,
)
from tools.estimate_time_offset import _GyroIntervalMean, read_gyro, sweep_lag

TELEMETRY_STEP = 0.01
IMAGE_STEP = 0.05


def _write_mock_flight(root, *, rows=401, frames=60):
    """One flight whose GPSNavVn label is a known function of telemetry time."""
    image_dir = root / "images"
    image_dir.mkdir()
    for index in range(frames):
        timestamp_ms = 1_000_000 + int(round(IMAGE_STEP * 1000)) * index
        image = np.full((24, 32), (index * 4) % 256, dtype=np.uint8)
        Image.fromarray(image).save(image_dir / f"{timestamp_ms}.jpg")

    headers = [
        "Time",
        "GyroX",
        "GyroY",
        "GyroZ",
        "AcclX",
        "AcclY",
        "AcclZ",
        "GPSNavVnX",
        "GPSNavVnY",
        "GPSNavVnZ",
    ]
    with (root / "flight.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index in range(rows):
            time_s = 1000.0 + index * TELEMETRY_STEP
            writer.writerow(
                [
                    time_s,
                    0.1 * index,
                    -0.05 * index,
                    1.0,
                    0.01 * index,
                    0.02 * index,
                    -1.0,
                    # A distinct label per telemetry row makes a mis-aligned
                    # lookup visible in the label itself.
                    float(index),
                    2.0 - 0.02 * index,
                    -0.5,
                ]
            )


def test_read_gyro_returns_increasing_times_and_three_channels(tmp_path):
    _write_mock_flight(tmp_path)
    times, gyro = read_gyro(tmp_path / "flight.csv", "Time")

    assert times.shape == (401,)
    assert gyro.shape == (401, 3)
    assert np.all(np.diff(times) > 0)


def test_sweep_lag_recovers_a_known_injected_offset():
    """The estimator must recover an offset built into synthetic signals."""
    rng = np.random.default_rng(0)
    telemetry_times = 1000.0 + np.arange(4000) * TELEMETRY_STEP
    rate = np.cumsum(rng.standard_normal(4000)) * 0.1
    gyro = np.stack([np.zeros_like(rate), np.zeros_like(rate), rate], axis=1)
    interval_mean = _GyroIntervalMean(telemetry_times, gyro)

    truth = 0.035
    start = 1002.0 + np.arange(400) * IMAGE_STEP
    end = start + IMAGE_STEP
    # The visual rate is the gyro seen `truth` seconds later on the image clock.
    visual = interval_mean(start + truth, end + truth, 2)

    lags = np.arange(-0.3, 0.3001, 0.005)
    estimated, correlation = sweep_lag(visual, 2, start, end, interval_mean, lags)

    assert abs(correlation) > 0.99
    assert estimated == pytest.approx(truth, abs=0.005)


def test_nearest_indices_picks_the_closest_row_each_time():
    reference = np.array([0.0, 1.0, 2.0, 3.0])
    queries = np.array([-0.4, 0.4, 0.6, 2.9, 7.0])
    index, residual = nearest_indices(reference, queries)

    assert index.tolist() == [0, 0, 1, 3, 3]
    assert residual == pytest.approx([-0.4, 0.4, -0.4, -0.1, 4.0])


def test_nearest_indices_handles_a_single_reference_row():
    index, residual = nearest_indices(np.array([5.0]), np.array([1.0, 9.0]))

    assert index.tolist() == [0, 0]
    assert residual == pytest.approx([-4.0, 4.0])


def test_resolve_time_offsets_interpolates_and_holds_the_ends():
    table = {"times_s": [10.0, 20.0], "offsets_s": [0.0, 1.0]}
    resolved = resolve_time_offsets(np.array([5.0, 15.0, 25.0]), table)

    assert resolved == pytest.approx([0.0, 0.5, 1.0])


def test_malformed_offset_tables_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_time_offsets(
            np.array([1.0]), {"times_s": [2.0, 1.0], "offsets_s": [0.0, 0.0]}
        )
    with pytest.raises(ValueError, match="equal-length"):
        resolve_time_offsets(np.array([1.0]), {"times_s": [1.0], "offsets_s": []})


def test_index_prefixed_filenames_parse_to_the_timestamp(tmp_path):
    """``<index>_<timestamp>.png`` must read the timestamp, not both glued.

    float() treats "_" as a digit separator, so parsing the stem directly
    turns 000001_1657679585068870104 into 1.16e19 without raising. The times
    stay ordered, so nothing downstream notices and every alignment is wrong.
    """

    folder = tmp_path / "images"
    folder.mkdir()
    base = 1657679585021740792
    for index in range(5):
        (folder / f"{index:06d}_{base + index * 50_000_000}.png").touch()

    paths, times = numeric_image_manifest(folder, "*.png", 1e-9)
    assert len(paths) == 5
    assert np.isclose(np.median(np.diff(times)), 0.05)
    assert times[-1] - times[0] < 1.0


def test_a_wrong_time_scale_is_rejected_with_a_suggestion(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    base = 1657679585021740792
    for index in range(5):
        (folder / f"{index:06d}_{base + index * 50_000_000}.png").touch()

    # Nanosecond stems read as milliseconds give a 50,000 second frame gap.
    with pytest.raises(ValueError, match="not a plausible camera rate"):
        numeric_image_manifest(folder, "*.png", 1e-3)


def test_non_numeric_stems_are_refused(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "frame_a.png").touch()
    with pytest.raises(ValueError, match="Cannot read a timestamp"):
        numeric_image_manifest(folder, "*.png", 1e-9)
