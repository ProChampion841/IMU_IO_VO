"""Shared velocity error metrics: vector magnitude, direction, and axes.

The four names are the cross-project convention (same definitions as the
sibling velocity-regression project):

``vel_rmse`` / ``vel_max_error``
    RMS and maximum of ``||v_pred - v_true||`` in m/s - a combined speed and
    direction error in physical units.

``vel_dir_rmse`` / ``vel_dir_max_error``
    RMS and maximum of the angle between the predicted and true velocity
    vectors, in degrees. Vectors shorter than ``eps`` carry no meaningful
    direction and are excluded rather than contributing an arbitrary angle.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

AXIS_NAMES = ("x", "y", "z")
METRIC_NAMES = (
    "vel_rmse",
    "vel_max_error",
    "vel_dir_rmse",
    "vel_dir_max_error",
    *(f"vel_rmse_{axis}" for axis in AXIS_NAMES),
    *(f"vel_bias_{axis}" for axis in AXIS_NAMES),
)
UNCERTAINTY_METRIC_NAMES = (
    *(f"vel_pred_std_{axis}" for axis in AXIS_NAMES),
    *(f"vel_zscore_rmse_{axis}" for axis in AXIS_NAMES),
    *(f"vel_coverage_1sigma_{axis}" for axis in AXIS_NAMES),
    *(f"vel_coverage_2sigma_{axis}" for axis in AXIS_NAMES),
)


def _as_array(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) velocity array, got {array.shape}")
    return array


def direction_angles_deg(
    prediction: np.ndarray, target: np.ndarray, *, eps: float = 1e-6
) -> np.ndarray:
    """Angle between each predicted and true velocity vector, in degrees."""
    predicted_norm = np.linalg.norm(prediction, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    usable = (predicted_norm > eps) & (target_norm > eps)
    if not np.any(usable):
        return np.empty(0, dtype=np.float64)
    cosine = np.sum(prediction[usable] * target[usable], axis=1) / (
        predicted_norm[usable] * target_norm[usable]
    )
    # Rounding can push the quotient just outside the domain of arccos.
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def velocity_error_metrics(prediction, target, *, eps: float = 1e-6) -> Dict[str, float]:
    """The four shared metrics for a fixed set of predictions (evaluation)."""
    predicted = _as_array(prediction)
    truth = _as_array(target)
    if predicted.shape != truth.shape:
        raise ValueError("Prediction and target must have the same shape")
    finite = np.isfinite(predicted).all(axis=1) & np.isfinite(truth).all(axis=1)
    predicted = predicted[finite]
    truth = truth[finite]
    if predicted.shape[0] == 0:
        return {name: float("nan") for name in METRIC_NAMES}
    residual = predicted - truth
    magnitude = np.linalg.norm(residual, axis=1)
    angles = direction_angles_deg(predicted, truth, eps=eps)
    metrics = {
        "vel_rmse": float(np.sqrt(np.mean(np.square(magnitude)))),
        "vel_max_error": float(magnitude.max()),
    }
    if angles.size:
        metrics["vel_dir_rmse"] = float(np.sqrt(np.mean(np.square(angles))))
        metrics["vel_dir_max_error"] = float(angles.max())
    else:
        metrics["vel_dir_rmse"] = float("nan")
        metrics["vel_dir_max_error"] = float("nan")
    for index, axis in enumerate(AXIS_NAMES):
        metrics[f"vel_rmse_{axis}"] = float(
            np.sqrt(np.mean(np.square(residual[:, index])))
        )
        metrics[f"vel_bias_{axis}"] = float(np.mean(residual[:, index]))
    return metrics


def velocity_uncertainty_metrics(
    prediction,
    target,
    predicted_std,
    *,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Per-axis scale and calibration metrics for a fixed prediction set.

    Standard deviations are in physical m/s. A calibrated Gaussian predictor
    approaches z-score RMS 1, 1-sigma coverage 0.683, and 2-sigma coverage
    0.954. These are diagnostics, not substitutes for physical velocity RMSE.
    """
    predicted = _as_array(prediction)
    truth = _as_array(target)
    std = _as_array(predicted_std)
    if predicted.shape != truth.shape or predicted.shape != std.shape:
        raise ValueError("Prediction, target, and standard deviation must match")
    finite = (
        np.isfinite(predicted).all(axis=1)
        & np.isfinite(truth).all(axis=1)
        & np.isfinite(std).all(axis=1)
    )
    predicted = predicted[finite]
    truth = truth[finite]
    std = np.maximum(std[finite], eps)
    if predicted.shape[0] == 0:
        return {name: float("nan") for name in UNCERTAINTY_METRIC_NAMES}
    absolute_error = np.abs(predicted - truth)
    zscore = absolute_error / std
    metrics: Dict[str, float] = {}
    for index, axis in enumerate(AXIS_NAMES):
        metrics[f"vel_pred_std_{axis}"] = float(np.mean(std[:, index]))
        metrics[f"vel_zscore_rmse_{axis}"] = float(
            np.sqrt(np.mean(np.square(zscore[:, index])))
        )
        metrics[f"vel_coverage_1sigma_{axis}"] = float(
            np.mean(absolute_error[:, index] <= std[:, index])
        )
        metrics[f"vel_coverage_2sigma_{axis}"] = float(
            np.mean(absolute_error[:, index] <= 2.0 * std[:, index])
        )
    return metrics


def masked_velocity_stats(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> Tuple[float, ...]:
    """One batch's contribution to the exact epoch metrics.

    Returns ``(sq_sum, count, max_error, dir_sq_sum, dir_count, dir_max)`` so
    an epoch can aggregate sums with SUM and maxima with MAX - across batches
    and across DDP ranks - and derive true epoch RMS values, rather than
    averaging per-batch RMS numbers. The tuple additionally carries per-axis
    squared-error and signed-error sums, so lateral/vertical failures remain
    visible without retaining all predictions.
    """
    if predicted.shape != target.shape or predicted.shape[-1] != 3:
        raise ValueError("predicted and target must be matching (..., 3) tensors")
    flat_mask = mask.reshape(-1) > 0
    prediction = predicted.detach().float().reshape(-1, 3)[flat_mask]
    truth = target.detach().float().reshape(-1, 3)[flat_mask]
    finite = torch.isfinite(prediction).all(dim=1) & torch.isfinite(truth).all(dim=1)
    prediction = prediction[finite]
    truth = truth[finite]
    if prediction.shape[0] == 0:
        return (0.0,) * 12
    residual = prediction - truth
    magnitude = torch.linalg.vector_norm(residual, dim=-1)
    predicted_norm = torch.linalg.vector_norm(prediction, dim=-1)
    target_norm = torch.linalg.vector_norm(truth, dim=-1)
    usable = (predicted_norm > eps) & (target_norm > eps)
    if bool(usable.any()):
        cosine = (prediction[usable] * truth[usable]).sum(dim=-1) / (
            predicted_norm[usable] * target_norm[usable]
        )
        angles = torch.rad2deg(torch.arccos(cosine.clamp(-1.0, 1.0)))
        dir_sq_sum = float(angles.square().sum().item())
        dir_count = float(angles.shape[0])
        dir_max = float(angles.max().item())
    else:
        dir_sq_sum = dir_count = dir_max = 0.0
    axis_sq = residual.square().sum(dim=0)
    axis_sum = residual.sum(dim=0)
    return (
        float(magnitude.square().sum().item()),
        float(magnitude.shape[0]),
        float(magnitude.max().item()),
        dir_sq_sum,
        dir_count,
        dir_max,
        *(float(value.item()) for value in axis_sq),
        *(float(value.item()) for value in axis_sum),
    )


class RunningVelocityStats:
    """Accumulate ``masked_velocity_stats`` tuples over an epoch."""

    def __init__(self) -> None:
        # sq_sum, count, dir_sq, dir_count, axis_sq[3], axis_sum[3]
        self.sums = np.zeros(10, dtype=np.float64)
        self.maxima = np.zeros(2, dtype=np.float64)  # max_error, dir_max

    def update(self, stats: Tuple[float, ...]) -> None:
        if len(stats) != 12:
            raise ValueError("Velocity statistics tuple has the wrong size")
        sq_sum, count, max_error, dir_sq, dir_count, dir_max, *axis = stats
        self.sums += (sq_sum, count, dir_sq, dir_count, *axis)
        self.maxima = np.maximum(self.maxima, (max_error, dir_max))

    def metrics(self) -> Dict[str, float]:
        sq_sum, count, dir_sq, dir_count, *axis = self.sums
        values = {
            "vel_rmse": float(np.sqrt(sq_sum / count)) if count else float("nan"),
            "vel_max_error": float(self.maxima[0]) if count else float("nan"),
            "vel_dir_rmse": (
                float(np.sqrt(dir_sq / dir_count)) if dir_count else float("nan")
            ),
            "vel_dir_max_error": (
                float(self.maxima[1]) if dir_count else float("nan")
            ),
        }
        axis_sq = axis[:3]
        axis_sum = axis[3:]
        for index, name in enumerate(AXIS_NAMES):
            values[f"vel_rmse_{name}"] = (
                float(np.sqrt(axis_sq[index] / count)) if count else float("nan")
            )
            values[f"vel_bias_{name}"] = (
                float(axis_sum[index] / count) if count else float("nan")
            )
        return values

    def postfix(self) -> Dict[str, str]:
        """Compact strings for a progress bar."""
        values = self.metrics()
        return {
            "v_rmse": f"{values['vel_rmse']:.3f}",
            "v_max": f"{values['vel_max_error']:.3f}",
            "dir_rmse": f"{values['vel_dir_rmse']:.2f}",
            "dir_max": f"{values['vel_dir_max_error']:.2f}",
        }


def masked_velocity_uncertainty_stats(
    predicted: torch.Tensor,
    target: torch.Tensor,
    predicted_std: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> Tuple[float, ...]:
    """One batch's additive per-axis uncertainty calibration statistics."""
    if (
        predicted.shape != target.shape
        or predicted.shape != predicted_std.shape
        or predicted.shape[-1] != 3
    ):
        raise ValueError(
            "predicted, target, and predicted_std must be matching (..., 3) tensors"
        )
    flat_mask = mask.reshape(-1) > 0
    prediction = predicted.detach().float().reshape(-1, 3)[flat_mask]
    truth = target.detach().float().reshape(-1, 3)[flat_mask]
    std = predicted_std.detach().float().reshape(-1, 3)[flat_mask]
    finite = (
        torch.isfinite(prediction).all(dim=1)
        & torch.isfinite(truth).all(dim=1)
        & torch.isfinite(std).all(dim=1)
    )
    prediction = prediction[finite]
    truth = truth[finite]
    std = std[finite].clamp_min(eps)
    if prediction.shape[0] == 0:
        return (0.0,) * 13
    absolute_error = (prediction - truth).abs()
    zscore = absolute_error / std
    return (
        float(prediction.shape[0]),
        *(float(value.item()) for value in std.sum(dim=0)),
        *(float(value.item()) for value in zscore.square().sum(dim=0)),
        *(float(value.item()) for value in (absolute_error <= std).sum(dim=0)),
        *(
            float(value.item())
            for value in (absolute_error <= 2.0 * std).sum(dim=0)
        ),
    )


class RunningVelocityUncertaintyStats:
    """Accumulate exact uncertainty calibration metrics over batches/ranks."""

    def __init__(self) -> None:
        # count, std_sum[3], zscore_sq_sum[3], cover1[3], cover2[3]
        self.sums = np.zeros(13, dtype=np.float64)

    def update(self, stats: Tuple[float, ...]) -> None:
        if len(stats) != 13:
            raise ValueError("Velocity uncertainty statistics have the wrong size")
        self.sums += stats

    def metrics(self) -> Dict[str, float]:
        count = self.sums[0]
        if not count:
            return {name: float("nan") for name in UNCERTAINTY_METRIC_NAMES}
        std_sum = self.sums[1:4]
        zscore_sq_sum = self.sums[4:7]
        cover1 = self.sums[7:10]
        cover2 = self.sums[10:13]
        values: Dict[str, float] = {}
        for index, axis in enumerate(AXIS_NAMES):
            values[f"vel_pred_std_{axis}"] = float(std_sum[index] / count)
            values[f"vel_zscore_rmse_{axis}"] = float(
                np.sqrt(zscore_sq_sum[index] / count)
            )
            values[f"vel_coverage_1sigma_{axis}"] = float(
                cover1[index] / count
            )
            values[f"vel_coverage_2sigma_{axis}"] = float(
                cover2[index] / count
            )
        return values


__all__ = [
    "AXIS_NAMES",
    "METRIC_NAMES",
    "UNCERTAINTY_METRIC_NAMES",
    "RunningVelocityStats",
    "RunningVelocityUncertaintyStats",
    "direction_angles_deg",
    "masked_velocity_uncertainty_stats",
    "masked_velocity_stats",
    "velocity_error_metrics",
    "velocity_uncertainty_metrics",
]
