"""Small, differentiable SE(3) helpers for the fixed-wing pose-VIO path.

Quaternions use ``(w, x, y, z)`` and represent active body-to-local rotations.
Per-tick translation increments are expressed in the body frame at the start
of the interval.  These conventions are deliberately explicit because mixing
NED, camera, and body frames is a common source of kilometre-scale VIO errors.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def normalize_quaternion_np(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Quaternion norm is zero")
    return values / norm


def quaternion_multiply_np(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(np.asarray(left), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quaternion_conjugate_np(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def euler_zyx_to_quaternion_np(euler_rpy: np.ndarray) -> np.ndarray:
    """Convert roll/pitch/yaw to a body-to-world intrinsic-ZYX quaternion."""

    euler = np.asarray(euler_rpy, dtype=np.float64)
    if euler.shape[-1] != 3:
        raise ValueError("Euler angles must have shape (..., 3)")
    roll, pitch, yaw = np.moveaxis(euler, -1, 0) * 0.5
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    quaternion = np.stack(
        (
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
        ),
        axis=-1,
    )
    return normalize_quaternion_np(quaternion)


def quaternion_to_matrix_np(quaternion: np.ndarray) -> np.ndarray:
    q = normalize_quaternion_np(quaternion)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def quaternion_to_rotvec_np(quaternion: np.ndarray) -> np.ndarray:
    q = normalize_quaternion_np(quaternion)
    q = np.where((q[..., :1] < 0.0), -q, q)
    vector = q[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(q[..., :1], 0.0, None))
    scale = np.where(vector_norm > 1e-10, angle / np.maximum(vector_norm, 1e-12), 2.0)
    return vector * scale


def normalize_quaternion_torch(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1e-12)


def quaternion_multiply_torch(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_conjugate_torch(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def rotvec_to_quaternion_torch(rotation_vector: torch.Tensor) -> torch.Tensor:
    if rotation_vector.shape[-1] != 3:
        raise ValueError("rotation_vector must have shape (..., 3)")
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle > 1e-7,
        torch.sin(half) / angle.clamp_min(1e-12),
        0.5 - angle.square() / 48.0,
    )
    return normalize_quaternion_torch(
        torch.cat((torch.cos(half), rotation_vector * scale), dim=-1)
    )


def quaternion_to_matrix_torch(quaternion: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion_torch(quaternion)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def quaternion_geodesic_error_rad(
    prediction: torch.Tensor, target: torch.Tensor, *, epsilon: float = 1e-12
) -> torch.Tensor:
    """Angle between two orientations, with a finite gradient at zero error.

    The textbook form ``2 * acos(|dot|)`` is correct but its derivative is
    infinite where the two orientations agree, so a batch containing an exact
    match backpropagates NaN. That case is not exotic: a trajectory re-origined
    at its own pose is exactly aligned there by construction, on every batch.

    ``2 * atan2(||v||, |w|)`` on the relative quaternion is the same angle, and
    the softened norm keeps the derivative finite everywhere. ``epsilon`` puts
    a floor of ``sqrt(epsilon)`` radians on the reported angle, which at the
    default is far below any meaningful attitude error.
    """

    prediction = normalize_quaternion_torch(prediction)
    target = normalize_quaternion_torch(target)
    relative = quaternion_multiply_torch(
        quaternion_conjugate_torch(prediction), target
    )
    vector_norm = torch.sqrt(relative[..., 1:].square().sum(dim=-1) + epsilon)
    return 2.0 * torch.atan2(vector_norm, relative[..., 0].abs())


def integrate_body_motion(
    delta_position_body: torch.Tensor,
    delta_rotation_body: torch.Tensor,
    initial_quaternion: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate per-tick body-frame SE(3) increments.

    Index zero is treated as the trajectory origin; its increments are ignored.
    Returned position and quaternion therefore have the same ``(B,T,...)``
    shape as the inputs.
    """

    if delta_position_body.shape != delta_rotation_body.shape:
        raise ValueError("Position and rotation increments must have equal shape")
    if delta_position_body.ndim != 3 or delta_position_body.shape[-1] != 3:
        raise ValueError("Motion increments must have shape (B, T, 3)")
    batch, length, _ = delta_position_body.shape
    if initial_quaternion is None:
        quaternion = delta_position_body.new_zeros((batch, 4))
        quaternion[:, 0] = 1.0
    else:
        if initial_quaternion.shape != (batch, 4):
            raise ValueError("initial_quaternion must have shape (B, 4)")
        quaternion = normalize_quaternion_torch(initial_quaternion)
    position = delta_position_body.new_zeros((batch, 3))
    positions = [position]
    quaternions = [quaternion]
    for index in range(1, length):
        rotation = quaternion_to_matrix_torch(quaternion)
        world_step = torch.matmul(
            rotation, delta_position_body[:, index].unsqueeze(-1)
        ).squeeze(-1)
        position = position + world_step
        delta_quaternion = rotvec_to_quaternion_torch(delta_rotation_body[:, index])
        quaternion = normalize_quaternion_torch(
            quaternion_multiply_torch(quaternion, delta_quaternion)
        )
        positions.append(position)
        quaternions.append(quaternion)
    return torch.stack(positions, dim=1), torch.stack(quaternions, dim=1)


def reorigin_trajectory(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    origin_index: int | torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-express an integrated trajectory relative to one of its own poses.

    ``integrate_body_motion`` accumulates from index zero, so any error made
    before ``origin_index`` is carried, unchanged, into every later pose. A
    loss that merely masks the early ticks still sees that inherited offset and
    charges the model for motion it was never supervised on. Re-origining both
    the prediction and the reference at the same index cancels the shared
    offset, leaving only the motion that happens afterwards.

    Returns position in the body frame at ``origin_index`` and orientation
    relative to that pose, both with the input shape.
    """

    if position.ndim != 3 or position.shape[-1] != 3:
        raise ValueError("position must have shape (B, T, 3)")
    if quaternion.ndim != 3 or quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have shape (B, T, 4)")
    if position.shape[1] != quaternion.shape[1]:
        raise ValueError("position and quaternion must have equal length")
    if isinstance(origin_index, torch.Tensor):
        indices = origin_index.to(device=position.device, dtype=torch.long).reshape(-1)
        if indices.shape[0] != position.shape[0]:
            raise ValueError("origin_index tensor must contain one value per batch")
        if bool(((indices < 0) | (indices >= position.shape[1])).any()):
            raise ValueError("origin_index is outside the trajectory")
        gather_position = indices.reshape(-1, 1, 1).expand(-1, 1, 3)
        gather_quaternion = indices.reshape(-1, 1, 1).expand(-1, 1, 4)
        origin_position = position.gather(1, gather_position).squeeze(1)
        origin_quaternion = normalize_quaternion_torch(
            quaternion.gather(1, gather_quaternion).squeeze(1)
        )
    else:
        if not 0 <= origin_index < position.shape[1]:
            raise ValueError("origin_index is outside the trajectory")
        origin_position = position[:, origin_index]
        origin_quaternion = normalize_quaternion_torch(quaternion[:, origin_index])
    origin_rotation = quaternion_to_matrix_torch(origin_quaternion)
    # matmul(v, R) contracts v with the rows of R, which is R^T @ v.
    relative_position = torch.matmul(
        position - origin_position.unsqueeze(1), origin_rotation
    )
    relative_quaternion = normalize_quaternion_torch(
        quaternion_multiply_torch(
            quaternion_conjugate_torch(origin_quaternion).unsqueeze(1), quaternion
        )
    )
    return relative_position, relative_quaternion


__all__ = [
    "reorigin_trajectory",
    "euler_zyx_to_quaternion_np",
    "integrate_body_motion",
    "normalize_quaternion_np",
    "normalize_quaternion_torch",
    "quaternion_conjugate_np",
    "quaternion_conjugate_torch",
    "quaternion_geodesic_error_rad",
    "quaternion_multiply_np",
    "quaternion_multiply_torch",
    "quaternion_to_matrix_np",
    "quaternion_to_matrix_torch",
    "quaternion_to_rotvec_np",
    "rotvec_to_quaternion_torch",
]
