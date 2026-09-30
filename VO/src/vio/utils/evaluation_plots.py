"""Evaluation figures: trajectory, velocity components, and velocity errors.

Colors follow the entities, fixed across every figure: predicted is always
blue, the reference always orange (a colorblind-validated pair on the light
surface). Text stays in ink tones, never in series colors; each panel has one
axis; grids are recessive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREDICTED = "#2a78d6"
REFERENCE = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e7e6e3"

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def _minutes(times_s: np.ndarray) -> np.ndarray:
    return (times_s - times_s[0]) / 60.0


def save_split_plots(
    directory: "str | Path",
    split: str,
    times_s: np.ndarray,
    predicted_velocity: np.ndarray,
    target_velocity: np.ndarray,
    predicted_position: np.ndarray,
    target_position: np.ndarray,
    metrics: Dict[str, float],
    velocity_frame: str = "NED",
) -> List[Path]:
    """Write the three evaluation figures for one split; returns their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    minutes = _minutes(np.asarray(times_s, dtype=np.float64))
    written: List[Path] = []

    with plt.rc_context(_RC):
        # ------------------------------------------------ trajectory (top-down)
        figure, axis = plt.subplots(figsize=(7.2, 6.4))
        axis.plot(
            target_position[:, 1], target_position[:, 0],
            color=REFERENCE, linewidth=1.6, label="reference",
        )
        axis.plot(
            predicted_position[:, 1], predicted_position[:, 0],
            color=PREDICTED, linewidth=1.8, label="predicted",
        )
        axis.scatter(
            [target_position[0, 1]], [target_position[0, 0]],
            s=48, color=INK, zorder=5, marker="o", label="start",
        )
        axis.scatter(
            [target_position[-1, 1]], [target_position[-1, 0]],
            s=64, color=REFERENCE, zorder=5, marker="X",
        )
        axis.scatter(
            [predicted_position[-1, 1]], [predicted_position[-1, 0]],
            s=64, color=PREDICTED, zorder=5, marker="X",
        )
        axis.set_xlabel("East (m)")
        axis.set_ylabel("North (m)")
        axis.set_title(f"{split}: trajectory (top-down), X marks the final poses")
        axis.set_aspect("equal", adjustable="datalim")
        axis.legend(loc="best")
        path = directory / f"{split}_trajectory.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        written.append(path)

        # -------------------------------------------------- velocity components
        figure, axes = plt.subplots(3, 1, figsize=(9.6, 6.8), sharex=True)
        component_names = (
            ("Body X", "Body Y", "Body Z")
            if velocity_frame.lower() == "body"
            else ("North", "East", "Down")
        )
        for index, (axis, name) in enumerate(zip(axes, component_names)):
            axis.plot(
                minutes, target_velocity[:, index],
                color=REFERENCE, linewidth=1.4, label="reference",
            )
            axis.plot(
                minutes, predicted_velocity[:, index],
                color=PREDICTED, linewidth=1.4, label="predicted",
            )
            axis.set_ylabel(f"{name} (m/s)")
        axes[0].set_title(
            f"{split}: velocity, predicted vs reference ({velocity_frame})"
        )
        axes[0].legend(loc="upper right", ncols=2)
        axes[-1].set_xlabel("Time (min)")
        path = directory / f"{split}_velocity.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        written.append(path)

        # ------------------------------------------------------ velocity errors
        error_norm = np.linalg.norm(predicted_velocity - target_velocity, axis=1)
        predicted_speed = np.linalg.norm(predicted_velocity, axis=1)
        target_speed = np.linalg.norm(target_velocity, axis=1)
        usable = (predicted_speed > 1e-6) & (target_speed > 1e-6)
        angles = np.full(error_norm.shape, np.nan)
        cosine = np.sum(predicted_velocity[usable] * target_velocity[usable], axis=1)
        angles[usable] = np.degrees(
            np.arccos(
                np.clip(cosine / (predicted_speed[usable] * target_speed[usable]), -1, 1)
            )
        )
        figure, axes = plt.subplots(2, 1, figsize=(9.6, 5.6), sharex=True)
        axes[0].plot(minutes, error_norm, color=PREDICTED, linewidth=1.2)
        axes[0].set_ylabel("‖v error‖ (m/s)")
        axes[0].set_title(
            f"{split}: velocity error — "
            f"vel_rmse {metrics['vel_rmse']:.3f} m/s, "
            f"vel_max_error {metrics['vel_max_error']:.3f} m/s"
        )
        axes[1].plot(minutes, angles, color=PREDICTED, linewidth=1.2)
        axes[1].set_ylabel("direction error (deg)")
        axes[1].set_title(
            f"vel_dir_rmse {metrics['vel_dir_rmse']:.2f}°, "
            f"vel_dir_max_error {metrics['vel_dir_max_error']:.2f}°"
        )
        axes[1].set_xlabel("Time (min)")
        path = directory / f"{split}_velocity_error.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        written.append(path)

    return written


__all__ = ["save_split_plots"]
