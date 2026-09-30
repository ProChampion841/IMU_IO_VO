#!/usr/bin/env python3
"""Estimate the camera/IMU time offset by correlating image motion with gyro.

Nearest-timestamp matching cannot measure this offset. The telemetry rows lie
on a dense, near-uniform grid, so *any* image timestamp inside the flight finds
a neighbour within half a telemetry step no matter how badly the two clocks are
aligned; a small nearest-neighbour residual therefore proves only that the grid
is dense. The offset has to be recovered from a physical signal that both
sensors observe.

This command uses whole-image phase correlation to measure the pixel shift
between consecutive frames, converts it to a rate, and finds the lag that
maximizes the correlation against the gyro rate averaged over the same
interval. It also repeats the estimate on windows so a constant offset can be
told apart from a drifting one.

The estimate assumes rotation dominates the frame-to-frame image motion, which
holds for a fixed-wing aircraft at altitude. It is a temporal alignment only:
it does not calibrate intrinsics, distortion, or the camera/IMU extrinsic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
# The package lives under src/; tools/ is imported as a package from the
# repository root. Both have to be importable when a script is run directly.
for _entry in (ROOT / "src", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import numpy as np

from vio.data.images import numeric_image_manifest

GYRO_COLUMNS = ("GyroX", "GyroY", "GyroZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--image-folder", default="images")
    parser.add_argument("--csv-name", default="flight.csv")
    parser.add_argument("--image-pattern", default="*.jpg")
    parser.add_argument("--image-time-scale", type=float, default=0.001)
    parser.add_argument("--time-column", default="Time")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Use only the first N frames (0 uses every frame)",
    )
    parser.add_argument(
        "--lag-range",
        type=float,
        default=0.30,
        help="Search +/- this many seconds around zero",
    )
    parser.add_argument("--lag-step", type=float, default=0.005)
    parser.add_argument(
        "--windows",
        type=int,
        default=8,
        help="Independent windows used to test whether the offset is constant",
    )
    parser.add_argument(
        "--min-abs-correlation",
        type=float,
        default=0.5,
        help="Reject an axis pairing whose peak correlation is weaker than this",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.image_time_scale <= 0:
        parser.error("image-time-scale must be positive")
    if args.lag_range <= 0 or args.lag_step <= 0:
        parser.error("lag-range and lag-step must be positive")
    if args.lag_step >= args.lag_range:
        parser.error("lag-step must be smaller than lag-range")
    if args.windows < 1:
        parser.error("windows must be at least one")
    return args


def read_gyro(csv_path: Path, time_column: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return telemetry times and the three gyro channels."""
    times: List[float] = []
    gyro: List[Tuple[float, float, float]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {time_column, *GYRO_COLUMNS} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Telemetry CSV is missing columns: {sorted(missing)}")
        for row in reader:
            try:
                stamp = float(row[time_column])
                sample = tuple(float(row[name]) for name in GYRO_COLUMNS)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(stamp) or not all(math.isfinite(v) for v in sample):
                continue
            times.append(stamp)
            gyro.append(sample)  # type: ignore[arg-type]
    if len(times) < 2:
        raise ValueError("Fewer than two usable telemetry rows were read")
    time_array = np.asarray(times, dtype=np.float64)
    if np.any(np.diff(time_array) <= 0):
        raise ValueError("Telemetry timestamps must be strictly increasing")
    return time_array, np.asarray(gyro, dtype=np.float64)


def image_motion(paths: Sequence[Path]) -> Dict[str, np.ndarray]:
    """Measure the whole-image shift between consecutive frames."""
    import cv2  # imported here so --help works without OpenCV

    first = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise ValueError(f"Could not read image: {paths[0]}")
    window = cv2.createHanningWindow((first.shape[1], first.shape[0]), cv2.CV_32F)
    previous = first.astype(np.float32)
    count = len(paths) - 1
    shift_x = np.zeros(count)
    shift_y = np.zeros(count)
    response = np.zeros(count)
    for index in range(1, len(paths)):
        current = cv2.imread(str(paths[index]), cv2.IMREAD_GRAYSCALE)
        if current is None:
            raise ValueError(f"Could not read image: {paths[index]}")
        current = current.astype(np.float32)
        if current.shape != previous.shape:
            raise ValueError("All images must share one resolution")
        (dx, dy), peak = cv2.phaseCorrelate(previous, current, window)
        shift_x[index - 1] = dx
        shift_y[index - 1] = dy
        response[index - 1] = peak
        previous = current
    return {"shift_x": shift_x, "shift_y": shift_y, "response": response}


class _GyroIntervalMean:
    """Mean gyro rate over an interval, from a cumulative trapezoid integral."""

    def __init__(self, times: np.ndarray, gyro: np.ndarray) -> None:
        self._times = times
        steps = np.diff(times)[:, None]
        midpoints = 0.5 * (gyro[1:] + gyro[:-1])
        self._cumulative = np.concatenate(
            [np.zeros((1, gyro.shape[1])), np.cumsum(midpoints * steps, axis=0)]
        )

    def __call__(self, start: np.ndarray, end: np.ndarray, axis: int) -> np.ndarray:
        integral = self._cumulative[:, axis]
        return (
            np.interp(end, self._times, integral) - np.interp(start, self._times, integral)
        ) / (end - start)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or not np.any(np.isfinite(a)) or not np.any(np.isfinite(b)):
        return float("nan")
    if float(np.std(a)) <= 0.0 or float(np.std(b)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def sweep_lag(
    rate: np.ndarray,
    axis: int,
    start: np.ndarray,
    end: np.ndarray,
    interval_mean: _GyroIntervalMean,
    lags: np.ndarray,
) -> Tuple[float, float]:
    """Return the sub-step lag maximizing |correlation| and the peak value."""
    values = np.asarray(
        [_correlation(rate, interval_mean(start + lag, end + lag, axis)) for lag in lags]
    )
    if not np.any(np.isfinite(values)):
        return float("nan"), float("nan")
    magnitude = np.abs(np.nan_to_num(values, nan=0.0))
    peak = int(np.argmax(magnitude))
    refinement = 0.0
    if 0 < peak < len(values) - 1:
        left, middle, right = magnitude[peak - 1], magnitude[peak], magnitude[peak + 1]
        curvature = left - 2.0 * middle + right
        if curvature != 0.0:
            # A parabola through three samples locates the peak between them.
            refinement = 0.5 * (left - right) / curvature
            refinement = float(np.clip(refinement, -1.0, 1.0))
    step = float(lags[1] - lags[0]) if len(lags) > 1 else 0.0
    return float(lags[peak] + refinement * step), float(values[peak])


def main() -> int:
    args = parse_args()
    root = args.dataset.expanduser().resolve()
    csv_path = root / args.csv_name
    image_dir = root / args.image_folder
    if not csv_path.is_file():
        raise SystemExit(f"Telemetry CSV does not exist: {csv_path}")
    if not image_dir.is_dir():
        raise SystemExit(f"Image folder does not exist: {image_dir}")

    paths, image_times = numeric_image_manifest(
        image_dir, args.image_pattern, args.image_time_scale
    )
    if args.max_frames and args.max_frames < len(paths):
        paths = paths[: args.max_frames]
        image_times = image_times[: args.max_frames]
    if len(paths) < 32:
        raise SystemExit("At least 32 frames are required for a usable estimate")

    telemetry_times, gyro = read_gyro(csv_path, args.time_column)
    print(f"Reading {len(paths)} images and {len(telemetry_times)} telemetry rows...")
    motion = image_motion(paths)

    start = image_times[:-1]
    end = image_times[1:]
    duration = end - start
    rates = {
        "shift_x": motion["shift_x"] / duration,
        "shift_y": motion["shift_y"] / duration,
    }
    interval_mean = _GyroIntervalMean(telemetry_times, gyro)
    lags = np.arange(-args.lag_range, args.lag_range + 0.5 * args.lag_step, args.lag_step)

    # Pair each image axis with whichever gyro axis it tracks, so the estimate
    # does not assume a camera mounting convention.
    zero_lag: Dict[str, Dict[str, float]] = {}
    for name, rate in rates.items():
        zero_lag[name] = {
            GYRO_COLUMNS[axis]: _correlation(rate, interval_mean(start, end, axis))
            for axis in range(3)
        }

    pairings = []
    for name, rate in rates.items():
        scores = zero_lag[name]
        best_column = max(scores, key=lambda column: abs(scores[column]))
        axis = GYRO_COLUMNS.index(best_column)
        lag, correlation = sweep_lag(rate, axis, start, end, interval_mean, lags)
        pairings.append(
            {
                "image_axis": name,
                "gyro_column": best_column,
                "zero_lag_correlation": scores[best_column],
                "offset_s": lag,
                "peak_correlation": correlation,
                "accepted": bool(
                    math.isfinite(correlation)
                    and abs(correlation) >= args.min_abs_correlation
                ),
            }
        )

    accepted = [pair for pair in pairings if pair["accepted"]]
    if not accepted:
        raise SystemExit(
            "No image/gyro pairing exceeded --min-abs-correlation; the offset "
            "could not be measured from this data"
        )

    # Weight each axis by how sharply it resolved the peak.
    weights = np.asarray([abs(pair["peak_correlation"]) for pair in accepted])
    offsets = np.asarray([pair["offset_s"] for pair in accepted])
    recommended = float(np.sum(weights * offsets) / np.sum(weights))

    windows: List[Dict[str, object]] = []
    if args.windows > 1:
        edges = np.linspace(0, len(start), args.windows + 1).astype(int)
        for index in range(args.windows):
            lo, hi = int(edges[index]), int(edges[index + 1])
            if hi - lo < 16:
                continue
            entry: Dict[str, object] = {
                "start_time_s": float(start[lo]),
                "end_time_s": float(end[hi - 1]),
                "pair_count": hi - lo,
            }
            for pair in accepted:
                axis = GYRO_COLUMNS.index(str(pair["gyro_column"]))
                rate = rates[str(pair["image_axis"])][lo:hi]
                lag, correlation = sweep_lag(
                    rate, axis, start[lo:hi], end[lo:hi], interval_mean, lags
                )
                entry[f"{pair['image_axis']}_offset_s"] = lag
                entry[f"{pair['image_axis']}_correlation"] = correlation
            windows.append(entry)

    stability: Dict[str, object] = {}
    for pair in accepted:
        key = f"{pair['image_axis']}_offset_s"
        values = np.asarray(
            [float(w[key]) for w in windows if isinstance(w.get(key), float)]
        )
        if values.size >= 2:
            stability[str(pair["image_axis"])] = {
                "window_count": int(values.size),
                "mean_s": float(values.mean()),
                "std_s": float(values.std(ddof=1)),
                "min_s": float(values.min()),
                "max_s": float(values.max()),
                "peak_to_peak_s": float(values.max() - values.min()),
            }

    # The most stable axis decides the verdict; a noisy axis should not veto it.
    spreads = [
        float(entry["std_s"])
        for entry in stability.values()
        if isinstance(entry, dict) and math.isfinite(float(entry["std_s"]))
    ]
    telemetry_step = float(np.median(np.diff(telemetry_times)))
    if not spreads:
        verdict = "insufficient_windows_to_judge_stability"
    elif min(spreads) <= 0.5 * telemetry_step:
        verdict = "constant_offset"
    else:
        verdict = "offset_varies_across_the_flight"

    # A per-window table so a flight whose offset moves can be corrected frame
    # by frame instead of by one constant. Built from the axis that resolved the
    # peak most sharply, at each window's midpoint.
    offset_table: Dict[str, List[float]] = {"times_s": [], "offsets_s": []}
    table_axis = None
    if windows:
        sharpest = max(accepted, key=lambda pair: abs(float(pair["peak_correlation"])))
        key = f"{sharpest['image_axis']}_offset_s"
        for entry in windows:
            value = entry.get(key)
            if isinstance(value, float) and math.isfinite(value):
                offset_table["times_s"].append(
                    0.5 * (float(entry["start_time_s"]) + float(entry["end_time_s"]))
                )
                offset_table["offsets_s"].append(value)
        table_axis = str(sharpest["image_axis"])
    if len(offset_table["times_s"]) < 2:
        offset_table = {"times_s": [], "offsets_s": []}
        table_axis = None

    result: Dict[str, object] = {
        "image_count": len(paths),
        "telemetry_count": int(len(telemetry_times)),
        "image_timestamp_scale": args.image_time_scale,
        "image_median_period_s": float(np.median(duration)),
        "telemetry_median_period_s": telemetry_step,
        "phase_correlation_response": {
            "mean": float(motion["response"].mean()),
            "min": float(motion["response"].min()),
        },
        "zero_lag_correlations": zero_lag,
        "pairings": pairings,
        "recommended_image_time_offset_s": recommended,
        "offset_table": offset_table,
        "offset_table_source_axis": table_axis,
        "search": {
            "lag_range_s": args.lag_range,
            "lag_step_s": args.lag_step,
            "min_abs_correlation": args.min_abs_correlation,
        },
        "windows": windows,
        "window_stability": stability,
        "verdict": verdict,
        "offset_convention": (
            "Add recommended_image_time_offset_s to every image timestamp before "
            "looking up telemetry; pass it as --image-time-offset."
        ),
        "assumptions": [
            "frame-to-frame image motion is rotation dominated",
            "gyro units are consistent across the flight",
            "no intrinsics, distortion, or camera/IMU extrinsic calibration is applied",
        ],
    }

    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    print(
        f"\nRecommended --image-time-offset {recommended:.4f}  "
        f"({recommended * 1000:+.2f} ms); verdict: {verdict}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
