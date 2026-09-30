"""Tests for the pose-VIO loss geometry.

The trajectory and orientation terms integrate from window index zero, so
without re-origining they charge the model for warm-up error the loss mask is
supposed to remove. These tests pin both halves of that contract: warm-up-only
error must cancel, and genuine post-warm-up error must survive.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vio.models.pose_geometry import (
    integrate_body_motion,
    quaternion_geodesic_error_rad,
    reorigin_trajectory,
)

BATCH = 2
TICKS = 200
WARMUP = 40
STEP_M = 0.20  # 20 m/s at 100 Hz


def _reference():
    step = torch.zeros(BATCH, TICKS, 3, dtype=torch.float64)
    step[..., 0] = STEP_M
    torch.manual_seed(0)
    rotation = torch.randn(BATCH, TICKS, 3, dtype=torch.float64) * 0.002
    return step, rotation


def _masked_mean(value, mask):
    return float((value * mask).sum() / mask.sum())


def _mask():
    mask = torch.ones(BATCH, TICKS, dtype=torch.float64)
    mask[:, :WARMUP] = 0.0
    return mask


def test_warmup_error_does_not_reach_the_trajectory_loss():
    step, rotation = _reference()
    predicted_step = step.clone()
    predicted_rotation = rotation.clone()
    # Wrong only during warm-up, exact on every supervised tick.
    predicted_step[:, :WARMUP, 0] *= 1.10
    predicted_rotation[:, :WARMUP] *= 1.50

    target_position, target_quaternion = integrate_body_motion(step, rotation)
    position, quaternion = integrate_body_motion(predicted_step, predicted_rotation)
    mask = _mask()

    leaked = _masked_mean((position - target_position).norm(dim=-1), mask)
    assert leaked > 0.5  # the defect this fix exists to remove

    position, quaternion = reorigin_trajectory(position, quaternion, WARMUP)
    target_position, target_quaternion = reorigin_trajectory(
        target_position, target_quaternion, WARMUP
    )
    corrected = _masked_mean((position - target_position).norm(dim=-1), mask)
    assert corrected < 1e-9

    orientation = _masked_mean(
        quaternion_geodesic_error_rad(quaternion, target_quaternion), mask
    )
    # quaternion_geodesic_error_rad softens its norm to keep the gradient
    # finite at zero error, which floors the reported angle near 2e-6 rad
    # (about 0.0001 degrees).
    assert orientation < 1e-5


def test_real_error_after_warmup_is_still_penalised():
    step, rotation = _reference()
    predicted_step = step.clone()
    predicted_step[:, WARMUP:, 0] *= 1.05

    target_position, target_quaternion = integrate_body_motion(step, rotation)
    position, quaternion = integrate_body_motion(predicted_step, rotation)
    position, quaternion = reorigin_trajectory(position, quaternion, WARMUP)
    target_position, target_quaternion = reorigin_trajectory(
        target_position, target_quaternion, WARMUP
    )

    assert _masked_mean((position - target_position).norm(dim=-1), _mask()) > 0.5


def test_reorigin_is_a_rigid_transform():
    step, rotation = _reference()
    position, quaternion = integrate_body_motion(step, rotation)
    moved, rotated = reorigin_trajectory(position, quaternion, WARMUP)

    # Distances between poses are preserved; only the frame changes.
    original = (position[:, 1:] - position[:, :-1]).norm(dim=-1)
    shifted = (moved[:, 1:] - moved[:, :-1]).norm(dim=-1)
    assert torch.allclose(original, shifted, atol=1e-12)
    assert torch.allclose(
        rotated.norm(dim=-1), torch.ones_like(rotated[..., 0]), atol=1e-12
    )
    # The origin pose maps to the identity.
    assert float(moved[:, WARMUP].abs().max()) < 1e-12
    assert torch.allclose(
        rotated[:, WARMUP].abs(),
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64).expand(BATCH, 4),
        atol=1e-12,
    )


def test_geodesic_error_has_a_finite_gradient_at_zero_error():
    """Re-origining makes an exactly-aligned pair certain, so this must hold.

    ``2 * acos(|dot|)`` is infinite-sloped where two orientations agree, which
    turns the origin tick of every re-origined batch into a NaN gradient.
    """

    identical = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    quaternion_geodesic_error_rad(identical, target).sum().backward()
    assert torch.isfinite(identical.grad).all()

    rotated = torch.tensor(
        [[[0.7071067811865476, 0.7071067811865476, 0.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    error = quaternion_geodesic_error_rad(rotated, target)
    error.sum().backward()
    assert torch.isfinite(rotated.grad).all()
    assert float(error) == pytest.approx(np.pi / 2, abs=1e-9)


def test_geodesic_error_matches_the_acos_form_away_from_the_singularity():
    torch.manual_seed(3)
    prediction = torch.nn.functional.normalize(
        torch.randn(1, 128, 4, dtype=torch.float64), dim=-1
    )
    target = torch.nn.functional.normalize(
        torch.randn(1, 128, 4, dtype=torch.float64), dim=-1
    )
    reference = 2.0 * torch.acos(
        (prediction * target).sum(dim=-1).abs().clamp(max=1.0)
    )
    computed = quaternion_geodesic_error_rad(prediction, target)
    assert torch.allclose(computed, reference, atol=1e-9)


def test_full_loss_backward_is_finite_after_reorigining():
    """The end-to-end guard: a re-origined loss must produce usable gradients."""

    step, rotation = _reference()
    predicted_step = (step + 0.01).requires_grad_(True)
    predicted_rotation = (rotation + 0.001).requires_grad_(True)

    target_position, target_quaternion = integrate_body_motion(step, rotation)
    position, quaternion = integrate_body_motion(predicted_step, predicted_rotation)
    position, quaternion = reorigin_trajectory(position, quaternion, WARMUP)
    target_position, target_quaternion = reorigin_trajectory(
        target_position, target_quaternion, WARMUP
    )
    mask = _mask()
    loss = (
        ((position - target_position).square().sum(dim=-1) * mask).sum()
        + (
            quaternion_geodesic_error_rad(quaternion, target_quaternion) * mask
        ).sum()
    )
    loss.backward()
    assert torch.isfinite(predicted_step.grad).all()
    assert torch.isfinite(predicted_rotation.grad).all()
