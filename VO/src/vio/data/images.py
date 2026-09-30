"""Image manifests and camera-to-IMU time offsets.

Frames are named by timestamp, so reading a dataset's images means turning
filenames into a time base and then aligning that base with the telemetry
clock. The offset may be a single measured constant or a table that varies
along the flight, which is why it is resolved per frame rather than applied
once.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

import numpy as np


# A timestamp is digits, optionally with one decimal point. Parsing with
# float() instead would silently accept "000001_1657679585068870104", because
# Python treats "_" as a digit separator: the frame index gets glued onto the
# timestamp and a 175-second flight parses as 3.5e19 seconds without error.
_TIMESTAMP = re.compile(r"^\d+(?:\.\d+)?$")

# Sanity bounds on the interval between frames once the scale is applied. A
# wrong --image-time-scale is otherwise invisible: the times stay ordered, so
# nothing downstream complains, and every alignment is quietly wrong.
_MIN_FRAME_INTERVAL_S = 1e-4
_MAX_FRAME_INTERVAL_S = 10.0


def numeric_image_manifest(
    image_dir: Path,
    image_pattern: str,
    timestamp_scale: float,
) -> Tuple[List[Path], np.ndarray]:
    """Read frame timestamps out of image filenames.

    The stem is either the timestamp itself (``1657679585021740792.png``) or a
    frame index and the timestamp joined by underscores
    (``000001_1657679585021740792.png``), in which case the last field is the
    timestamp. ``timestamp_scale`` converts it to seconds: 1e-3 for
    milliseconds, 1e-9 for nanoseconds.
    """

    paths = [path for path in image_dir.glob(image_pattern) if path.is_file()]
    if not paths:
        raise ValueError(f"No images match {image_pattern!r} in the dataset image folder")
    if not math.isfinite(timestamp_scale) or timestamp_scale <= 0:
        raise ValueError("timestamp_scale must be finite and positive")
    parsed = []
    for path in paths:
        field = path.stem.rsplit("_", 1)[-1]
        if not _TIMESTAMP.match(field):
            raise ValueError(
                f"Cannot read a timestamp from {path.name!r}. Expected "
                f"<timestamp> or <index>_<timestamp>, digits only."
            )
        timestamp = float(field) * timestamp_scale
        if not math.isfinite(timestamp):
            raise ValueError(f"Image timestamp is not finite: {path.name}")
        parsed.append((timestamp, path))
    parsed.sort(key=lambda item: item[0])
    times = np.asarray([item[0] for item in parsed], dtype=np.float64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Image timestamps must be strictly increasing")
    if times.size > 1:
        interval = float(np.median(np.diff(times)))
        if not _MIN_FRAME_INTERVAL_S <= interval <= _MAX_FRAME_INTERVAL_S:
            suggestion = timestamp_scale * 0.05 / interval
            raise ValueError(
                f"Median frame interval is {interval:.6g} s, which is not a "
                f"plausible camera rate. --image-time-scale {timestamp_scale:g} "
                f"is probably wrong; {suggestion:.1e} would give about 20 Hz "
                f"(use 1e-3 for milliseconds, 1e-9 for nanoseconds)."
            )
    return [item[1] for item in parsed], times


def nearest_indices(
    reference: np.ndarray, queries: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Index of the closest ``reference`` entry for each query, and the residual.

    ``reference`` must be sorted. The returned residual is signed
    ``query - reference[index]``, so its sign shows which way the image clock
    sits relative to the telemetry clock at that instant.
    """
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("reference must be a non-empty one-dimensional array")
    if reference.size == 1:
        index = np.zeros(queries.shape, dtype=np.int64)
        return index, queries - reference[0]
    upper = np.clip(np.searchsorted(reference, queries, side="left"), 1, reference.size - 1)
    lower = upper - 1
    take_lower = np.abs(queries - reference[lower]) <= np.abs(reference[upper] - queries)
    index = np.where(take_lower, lower, upper).astype(np.int64)
    return index, queries - reference[index]


def interpolate_at(
    reference_times_s: np.ndarray, values: np.ndarray, queries_s: np.ndarray
) -> np.ndarray:
    """Linearly interpolate ``values`` onto ``queries_s``.

    ``reference_times_s`` must be sorted and strictly increasing; ``values`` is
    ``(N,)`` or ``(N, K)`` aligned with it. Queries outside the reference range
    are held at the end value rather than extrapolated, because a linear
    extrapolation of attitude past the end of telemetry is not a measurement.

    This is the alternative to snapping a query onto its nearest sample. At
    100 Hz a snap moves the query by up to 5 ms, and it does so in whichever
    direction the grid happens to fall, so the error is not zero-mean across a
    flight.
    """

    times = np.asarray(reference_times_s, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    query = np.asarray(queries_s, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("reference_times_s must be a non-empty one-dimensional array")
    if data.shape[0] != times.size:
        raise ValueError(
            f"values must align with the {times.size} reference times; got {data.shape}"
        )
    if times.size == 1:
        return np.broadcast_to(data[0], query.shape + data.shape[1:]).copy()

    upper = np.clip(np.searchsorted(times, query, side="left"), 1, times.size - 1)
    lower = upper - 1
    span = times[upper] - times[lower]
    # Repeated timestamps would divide by zero; fall back to the lower sample.
    weight = np.where(span > 0, (query - times[lower]) / np.where(span > 0, span, 1.0), 0.0)
    weight = np.clip(weight, 0.0, 1.0)  # clamp instead of extrapolating
    if data.ndim > 1:
        weight = weight.reshape(weight.shape + (1,) * (data.ndim - 1))
    return data[lower] + weight * (data[upper] - data[lower])


def _sample_at(times: np.ndarray, data: np.ndarray, when: float) -> np.ndarray:
    """One interpolated sample, held at the ends. Scalar fast path."""

    if times.size == 1:
        return data[0]
    upper = int(np.searchsorted(times, when, side="left"))
    if upper <= 0:
        return data[0]
    if upper >= times.size:
        return data[-1]
    lower = upper - 1
    span = times[upper] - times[lower]
    if span <= 0.0:
        return data[lower]
    weight = (when - times[lower]) / span
    # Return the bracketing sample untouched when the query lands exactly on it.
    # The interpolating form always reads BOTH samples, so a non-finite value at
    # the sample just outside the requested interval would poison an endpoint
    # that sits exactly on a clean one: 0.0 * (nan - x) is nan, not 0.
    if weight <= 0.0:
        return data[lower]
    if weight >= 1.0:
        return data[upper]
    return data[lower] + weight * (data[upper] - data[lower])


def mean_over_interval(
    reference_times_s: np.ndarray,
    values: np.ndarray,
    start_s: float,
    stop_s: float,
) -> np.ndarray:
    """Time-average of ``values`` over the exact interval ``[start_s, stop_s]``.

    The interval endpoints are almost never telemetry samples. Integrating
    between the nearest samples instead answers a slightly different question,
    over a slightly different span, and then divides by that span too -- so the
    error enters twice and does not cancel. Here the endpoints are interpolated,
    the interior samples are integrated trapezoidally, and the divisor is the
    true requested duration.

    Interior samples are bracketed by binary search and taken as a contiguous
    slice. This runs once per visual event per window, so a boolean mask over
    the whole telemetry array -- which is what the obvious implementation does --
    is not affordable: it measured 12x slower than the index slicing it replaced.
    """

    times = np.asarray(reference_times_s, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("reference_times_s must be a non-empty one-dimensional array")
    if data.shape[0] != times.size:
        raise ValueError(
            f"values must align with the {times.size} reference times; got {data.shape}"
        )
    start, stop = float(start_s), float(stop_s)
    if stop < start:
        start, stop = stop, start
    duration = stop - start
    if duration <= 0.0:
        return np.array(_sample_at(times, data, start))

    head = _sample_at(times, data, start)
    tail = _sample_at(times, data, stop)
    first = int(np.searchsorted(times, start, side="right"))
    last = int(np.searchsorted(times, stop, side="left"))
    if first >= last:
        # No telemetry sample lies strictly inside: one trapezoid spans it all.
        return 0.5 * (head + tail)

    interior_t = times[first:last]
    knots = np.empty(interior_t.size + 2, dtype=np.float64)
    knots[0] = start
    knots[1:-1] = interior_t
    knots[-1] = stop
    samples = np.empty((interior_t.size + 2,) + data.shape[1:], dtype=np.float64)
    samples[0] = head
    samples[1:-1] = data[first:last]
    samples[-1] = tail

    # Line the time step up with the time axis for ANY trailing shape. The
    # obvious step[:, None] is correct only when data is exactly 2-D; at 3-D it
    # trailing-aligns against (T, K, M) and multiplies the channel axis instead,
    # returning a wrong average rather than raising whenever K happens to equal
    # the segment count. This is the same form interpolate_at uses.
    step = np.diff(knots).reshape((-1,) + (1,) * (data.ndim - 1))
    integral = np.sum(0.5 * (samples[1:] + samples[:-1]) * step, axis=0)
    return integral / duration


def resolve_time_offsets(
    image_times: np.ndarray, offset: "float | Mapping[str, Sequence[float]]"
) -> np.ndarray:
    """Return a per-image offset, from either a scalar or a time-varying table.

    A table is ``{"times_s": [...], "offsets_s": [...]}`` and is interpolated
    piecewise-linearly, holding the end values outside its range. This is the
    form to use when the camera/telemetry offset is not constant across a
    flight, so each frame is corrected by the offset in force at its own time.
    """
    if isinstance(offset, Mapping):
        times = np.asarray(offset["times_s"], dtype=np.float64)
        values = np.asarray(offset["offsets_s"], dtype=np.float64)
        if times.ndim != 1 or values.shape != times.shape or times.size == 0:
            raise ValueError("Offset table needs equal-length non-empty times/offsets")
        if np.any(np.diff(times) <= 0):
            raise ValueError("Offset table times must be strictly increasing")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
            raise ValueError("Offset table entries must be finite")
        return np.interp(image_times, times, values)
    value = float(offset)
    if not math.isfinite(value):
        raise ValueError("image_time_offset_s must be finite")
    return np.full(image_times.shape, value, dtype=np.float64)


__all__ = [
    "interpolate_at",
    "mean_over_interval",
    "nearest_indices",
    "numeric_image_manifest",
    "resolve_time_offsets",
]
