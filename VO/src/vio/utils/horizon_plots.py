"""Figures for a horizon evaluation: one trajectory, one error timeline.

Three figures per split, and each answers a different question.

**Trajectory** - the whole split flown once, end to end, with no resets: the
dead-reckoned path from the estimated velocity laid over the true one. This is
what "165 m off after ten minutes" actually looks like. The moment of worst
velocity error is marked on it, so a bad excursion can be located on the map
rather than only in a table.

**Errors** - velocity error, direction error and dead-reckoning position error
against flight time, each with its own maximum marked, and a shared vertical
line at the instant of worst VELOCITY error. That line is the point: it says
whether the worst moment is early (a warm-up artefact), late (accumulated
drift), or coincident across all three panels (one genuinely bad stretch of
flight rather than three unrelated ones).

**Summary** - the per-horizon table's numbers against horizon length.

Same colors and rc context as :mod:`vio.utils.evaluation_plots`, reused rather
than redefined, so a horizon figure and a pose-evaluation figure read as one
system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluation_plots import INK, MUTED, PREDICTED, REFERENCE, _RC

#: Body axes, spelled out. The metrics call them x/y/z; a figure has room to
#: say which direction each one is.
AXIS_LABELS = ("x (forward)", "y (right)", "z (down)")

#: Above this many points a line is decimated before being handed to
#: matplotlib. A 30-minute leg is 180,000 ticks and a figure is ~1000 pixels
#: wide, so drawing every one costs seconds and shows nothing extra.
_MAX_PLOT_POINTS = 6000


def _decimate(*arrays: np.ndarray, limit: int = _MAX_PLOT_POINTS):
    """Evenly thin parallel arrays to at most ``limit`` points each.

    Plotting 180,000 points into a 1000-pixel axis is ~30x more work than the
    picture can show. Thinning is by stride rather than by averaging so a spike
    stays a spike - and the maxima are drawn from the FULL data separately, so
    nothing that matters depends on what survives here.
    """

    length = len(arrays[0])
    if length <= limit:
        return arrays
    step = int(np.ceil(length / limit))
    return tuple(array[::step] for array in arrays)


def _placeholder(
    directory: "str | Path", filename: str, title: str, reason: str
) -> Path:
    """Write the figure anyway, carrying the reason it has no data.

    A missing file is the worst outcome: it looks like a plotting fault and
    says nothing about the run. A figure that states why it is empty is the
    same information, delivered where the reader is already looking.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_RC):
        figure, axis = plt.subplots(figsize=(9.6, 4.2))
        axis.set_title(title)
        axis.text(
            0.5, 0.5, f"no data to draw\n\n{reason}",
            transform=axis.transAxes, ha="center", va="center",
            color=MUTED, fontsize=11, wrap=True,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        path = directory / filename
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return path


def _no_series_reason(entry: Dict[str, object]) -> str:
    """Why a whole-span entry carries no series."""

    if entry.get("skipped"):
        return str(entry["skipped"])
    if not entry.get("fits"):
        return "the whole-split run scored nothing"
    return "the run was made without collect_series=True"


def _leg0(values: np.ndarray) -> np.ndarray:
    """The single run of a span, dropping the leading (length-1) leg axis."""

    return np.asarray(values)[0]


def save_split_trajectory(
    directory: "str | Path", split: str, entry: Dict[str, object]
) -> Optional[Path]:
    """Top-down dead-reckoned path vs. the true one, over the whole split.

    Always writes the file. With no position series to draw - no reference
    rotation, so nothing to integrate into a path - the figure carries that
    reason instead, because a missing file reads as a plotting fault and says
    nothing at all.
    """

    series = entry.get("series") or {}
    if "trajectory_predicted_ned" not in series:
        return _placeholder(
            directory, f"{split}_trajectory.png",
            f"{split}: dead-reckoned trajectory",
            _no_series_reason(entry)
            if "vel_error_m_s" not in series
            else "no reference attitude was supplied, so velocity cannot be "
                 "integrated into a path (--no-position)",
        )
    predicted = _leg0(series["trajectory_predicted_ned"])
    reference = _leg0(series["trajectory_reference_ned"])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    series_time = np.asarray(series.get("time_since_start_s", []), dtype=np.float64)
    with plt.rc_context(_RC):
        figure = plt.figure(figsize=(13.6, 6.8))
        # The map answers "where did it end up"; the three stacked panels
        # answer "which axis put it there", which the map cannot - a track
        # drifting north-east looks the same as one drifting north twice as
        # fast and east not at all, once the axes are mixed into one line.
        grid = figure.add_gridspec(3, 2, width_ratios=(1.25, 1.0), hspace=0.16)
        axis = figure.add_subplot(grid[:, 0])
        thin_reference, thin_predicted = _decimate(reference, predicted)
        axis.plot(
            thin_reference[:, 1], thin_reference[:, 0],
            color=REFERENCE, linewidth=1.6, label="reference (true path)",
        )
        axis.plot(
            thin_predicted[:, 1], thin_predicted[:, 0],
            color=PREDICTED, linewidth=1.6, label="predicted (dead-reckoned)",
        )
        axis.scatter([0], [0], s=54, color=INK, zorder=6, marker="o", label="start")
        axis.scatter(
            [reference[-1, 1]], [reference[-1, 0]],
            s=70, color=REFERENCE, zorder=6, marker="X",
        )
        axis.scatter(
            [predicted[-1, 1]], [predicted[-1, 0]],
            s=70, color=PREDICTED, zorder=6, marker="X",
        )
        # Where the worst velocity error happened, on the map.
        worst_tick = entry.get("vel_max_error_tick")
        if worst_tick is not None and 0 <= int(worst_tick) < predicted.shape[0]:
            index = int(worst_tick)
            axis.scatter(
                [predicted[index, 1]], [predicted[index, 0]],
                s=110, facecolors="none", edgecolors=INK, linewidths=1.6,
                zorder=7, label="worst velocity error",
            )
        final = float(entry.get("pos_error_final", float("nan")))
        drift = float(entry.get("pos_drift_percent", float("nan")))
        axis.set_xlabel("East (m)")
        axis.set_ylabel("North (m)")
        axis.set_title(
            f"{split}: dead-reckoned trajectory over the whole split\n"
            f"{final:.0f} m off at the end, {drift:.2f}% of the path flown "
            f"(X marks the final positions)"
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.legend(loc="best")

        # Per-axis: North, East and Down against flight time, each with the
        # reference under the prediction so a bias on one axis is visible as a
        # gap that opens rather than a shape that merely looks wrong.
        count = min(reference.shape[0], predicted.shape[0], series_time.size) \
            if series_time.size else min(reference.shape[0], predicted.shape[0])
        minutes = (
            series_time[:count] / 60.0 if series_time.size
            else np.arange(count, dtype=np.float64)
        )
        worst_minutes = entry.get("vel_max_error_time_s")
        worst_minutes = None if worst_minutes is None else float(worst_minutes) / 60.0
        for row, name in enumerate(("North", "East", "Down")):
            panel = figure.add_subplot(grid[row, 1])
            thin_time, thin_ref, thin_pred = _decimate(
                minutes, reference[:count, row], predicted[:count, row]
            )
            panel.plot(thin_time, thin_ref, color=REFERENCE, linewidth=1.3)
            panel.plot(thin_time, thin_pred, color=PREDICTED, linewidth=1.3)
            panel.set_ylabel(f"{name} (m)")
            if worst_minutes is not None:
                panel.axvline(
                    worst_minutes, color=INK, linestyle="--", linewidth=1.0, alpha=0.55
                )
            if row == 0:
                panel.set_title("per axis, reference vs. predicted")
            if row < 2:
                panel.tick_params(labelbottom=False)
            else:
                panel.set_xlabel("Time into the split (min)")
        path = directory / f"{split}_trajectory.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return path


def save_split_errors(
    directory: "str | Path", split: str, entry: Dict[str, object]
) -> Optional[Path]:
    """Velocity, direction and position error against time, maxima marked."""

    series = entry.get("series") or {}
    if "vel_error_m_s" not in series:
        return _placeholder(
            directory, f"{split}_errors.png",
            f"{split}: error over the whole split",
            _no_series_reason(entry),
        )
    minutes = np.asarray(series["time_since_start_s"], dtype=np.float64) / 60.0
    velocity = _leg0(series["vel_error_m_s"])
    direction = _leg0(series["vel_dir_error_deg"])
    position = _leg0(series["pos_error_m"]) if "pos_error_m" in series else None

    panels = [
        ("‖v error‖ (m/s)", velocity, "vel_max_error", "vel_max_error_time_s", "{:.3f} m/s"),
        ("direction error (deg)", direction, "vel_dir_max_error",
         "vel_dir_max_error_time_s", "{:.2f}°"),
    ]
    if position is not None:
        panels.append(
            ("dead-reckoning\nposition error (m)", position, "pos_error_max",
             "pos_error_max_time_s", "{:.1f} m")
        )

    worst_minutes = entry.get("vel_max_error_time_s")
    worst_minutes = None if worst_minutes is None else float(worst_minutes) / 60.0

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_RC):
        figure, axes = plt.subplots(
            len(panels), 1, figsize=(10.4, 2.6 * len(panels)), sharex=True
        )
        axes = np.atleast_1d(axes)
        for axis, (label, values, max_key, time_key, fmt) in zip(axes, panels):
            thin_minutes, thin_values = _decimate(minutes, values)
            axis.plot(thin_minutes, thin_values, color=PREDICTED, linewidth=1.0)
            axis.set_ylabel(label)
            # The maximum comes from the FULL series, not the thinned copy, so
            # a decimated spike is still reported and marked at its true place.
            peak = entry.get(max_key)
            peak_time = entry.get(time_key)
            if peak is not None and peak_time is not None and np.isfinite(peak):
                axis.scatter(
                    [float(peak_time) / 60.0], [float(peak)],
                    s=52, color=REFERENCE, zorder=6,
                )
                axis.annotate(
                    f"max {fmt.format(float(peak))} @ {float(peak_time) / 60.0:.2f} min",
                    xy=(float(peak_time) / 60.0, float(peak)),
                    xytext=(6, -2), textcoords="offset points",
                    color=MUTED, fontsize=8.5, va="top",
                )
            # One shared marker of the worst VELOCITY moment, on every panel.
            if worst_minutes is not None:
                axis.axvline(
                    worst_minutes, color=INK, linestyle="--", linewidth=1.0, alpha=0.55
                )
        axes[0].set_title(
            f"{split}: error over the whole split, one continuous run\n"
            f"dashed line = worst velocity error, at "
            f"{'-' if worst_minutes is None else f'{worst_minutes:.2f}'} min"
        )
        axes[-1].set_xlabel("Time into the split (min)")
        path = directory / f"{split}_errors.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return path


def save_split_velocity(
    directory: "str | Path", split: str, entry: Dict[str, object]
) -> Optional[Path]:
    """Body-frame velocity per axis, truth and estimate on the same axes.

    The error figure says how far apart the two are; this one says what they
    are. The distinction matters because the same RMSE is produced by faults
    that need different fixes - an estimate that lags the truth, one that
    scales it, and one sitting at a constant offset all look identical once
    subtracted, and are obvious the moment both signals are drawn.
    """

    series = entry.get("series") or {}
    if "vel_predicted_body" not in series:
        return _placeholder(
            directory, f"{split}_velocity.png",
            f"{split}: body-frame velocity, truth vs. estimate",
            _no_series_reason(entry),
        )
    predicted = _leg0(series["vel_predicted_body"])
    reference = _leg0(series["vel_target_body"])
    time_s = np.asarray(series.get("time_since_start_s", []), dtype=np.float64)
    count = min(predicted.shape[0], reference.shape[0])
    if time_s.size:
        count = min(count, time_s.size)
    minutes = (
        time_s[:count] / 60.0 if time_s.size else np.arange(count, dtype=np.float64)
    )
    worst_minutes = entry.get("vel_max_error_time_s")
    worst_minutes = None if worst_minutes is None else float(worst_minutes) / 60.0

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_RC):
        figure, axes = plt.subplots(3, 1, figsize=(10.4, 7.8), sharex=True)
        for row, name in enumerate(AXIS_LABELS):
            axis = axes[row]
            thin_time, thin_ref, thin_pred = _decimate(
                minutes, reference[:count, row], predicted[:count, row]
            )
            axis.plot(
                thin_time, thin_ref, color=REFERENCE, linewidth=1.3,
                label="GT (GPSNavVn, body frame)" if row == 0 else None,
            )
            axis.plot(
                thin_time, thin_pred, color=PREDICTED, linewidth=1.3,
                label="VO (estimated)" if row == 0 else None,
            )
            axis.set_ylabel(f"{name} (m/s)")
            if worst_minutes is not None:
                axis.axvline(
                    worst_minutes, color=INK, linestyle="--", linewidth=1.0, alpha=0.55
                )
        axes[0].set_title(
            f"{split}: body-frame velocity, truth vs. estimate, one continuous run\n"
            f"dashed line = worst velocity error, at "
            f"{'-' if worst_minutes is None else f'{worst_minutes:.2f}'} min"
        )
        axes[0].legend(loc="best")
        axes[-1].set_xlabel("Time into the split (min)")
        path = directory / f"{split}_velocity.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return path


def save_horizon_summary_plot(
    directory: "str | Path", split: str, results: Dict[str, Dict[str, object]]
) -> Optional[Path]:
    """Every horizon's headline numbers side by side.

    Visualises what :func:`vio.models.velocity_horizons.format_horizon_table`
    prints as text - velocity RMSE/max on the left, dead-reckoned position
    error and drift percentage on the right, both against horizon length.
    Always writes the file; if every horizon was skipped it carries the
    reasons they were.
    """

    fitted = [
        (float(entry["minutes_realised"]), entry)
        for entry in results.values()
        if entry.get("fits")
    ]
    if not fitted:
        reasons = sorted(
            {str(entry["skipped"]) for entry in results.values() if entry.get("skipped")}
        )
        return _placeholder(
            directory, f"{split}_summary_vs_horizon.png",
            f"{split}: error vs. horizon length",
            "every horizon was skipped:\n" + "\n".join(reasons)
            if reasons
            else "no horizon was scored",
        )
    fitted.sort(key=lambda pair: pair[0])
    minutes = np.array([m for m, _ in fitted])
    vel_rmse = np.array([e["vel_rmse"] for _, e in fitted])
    vel_max = np.array([e["vel_max_error"] for _, e in fitted])
    has_position = all("pos_error_final" in e for _, e in fitted)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_RC):
        figure, axes = plt.subplots(1, 2 if has_position else 1, figsize=(11.2, 4.4))
        axes = np.atleast_1d(axes)
        axes[0].plot(minutes, vel_rmse, "o-", color=PREDICTED, label="vel_rmse")
        axes[0].plot(
            minutes, vel_max, "o--", color=MUTED, label="vel_max_error", alpha=0.85
        )
        axes[0].set_xlabel("Horizon (min)")
        axes[0].set_ylabel("Velocity error (m/s)")
        axes[0].set_title(f"{split}: velocity error vs. horizon length")
        axes[0].legend(loc="best")
        if has_position:
            pos_final = np.array([e["pos_error_final"] for _, e in fitted])
            drift = np.array([e["pos_drift_percent"] for _, e in fitted])
            axis2 = axes[1].twinx()
            axes[1].plot(minutes, pos_final, "o-", color=REFERENCE, label="pos_error_final")
            axis2.plot(minutes, drift, "s--", color=INK, alpha=0.7, label="drift_%")
            axes[1].set_xlabel("Horizon (min)")
            axes[1].set_ylabel("Position error at prefix end (m)", color=REFERENCE)
            axis2.set_ylabel("Drift (% of path flown)", color=INK)
            axes[1].set_title(f"{split}: dead-reckoned drift vs. horizon length")
            lines = axes[1].get_lines() + axis2.get_lines()
            axes[1].legend(lines, [line.get_label() for line in lines], loc="best")
        path = directory / f"{split}_summary_vs_horizon.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    return path


def save_split_plots(
    directory: "str | Path",
    split: str,
    whole_span: Dict[str, object],
    results: Dict[str, Dict[str, object]],
) -> List[Path]:
    """The three figures for one split; returns the paths actually written."""

    written = [
        save_split_trajectory(directory, split, whole_span),
        save_split_velocity(directory, split, whole_span),
        save_split_errors(directory, split, whole_span),
        save_horizon_summary_plot(directory, split, results),
    ]
    return [path for path in written if path is not None]


__all__ = [
    "AXIS_LABELS",
    "save_horizon_summary_plot",
    "save_split_errors",
    "save_split_plots",
    "save_split_trajectory",
    "save_split_velocity",
]
