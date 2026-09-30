"""ONNX export: the exported graphs must reproduce the PyTorch model.

Skipped when onnx/onnxruntime are not installed - export is a deployment step,
not a training dependency.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from tools.export_onnx import (  # noqa: E402
    PlanarFrontendExport,
    TemporalStepExport,
    _export,
    adaptive_avg_pool2d_matmul,
    frontend_example,
    state_layout,
    temporal_example,
    verify_frontend,
    verify_temporal,
)
from vio.models.planar_frontend import PlanarFlowFrontend  # noqa: E402
from vio.models.planar_geometry import (  # noqa: E402
    NADIR_MOUNTINGS,
    fit_planar_translation,
    inverse_3x3,
    onnx_safe_linalg,
)
from vio.models.vision_mamba_vo import VisionMambaVO  # noqa: E402

HEIGHT, WIDTH = 288, 512
FOCAL = 526.0


def camera_matrix() -> torch.Tensor:
    return torch.tensor(
        [[FOCAL, 0.0, (WIDTH - 1) / 2.0], [0.0, FOCAL, (HEIGHT - 1) / 2.0], [0.0, 0.0, 1.0]]
    )


def small_frontend() -> PlanarFlowFrontend:
    torch.manual_seed(0)
    return PlanarFlowFrontend(
        camera_from_body=NADIR_MOUNTINGS["top_forward"],
        prior_velocity_body=(18.0, 0.0, 0.0),
        input_channels=3,
        image_size=(HEIGHT, WIDTH),
        d_model=32, depth=1, visual_dim=16, context_grid=(6, 8), token_grid=4,
        coarse_factor=2, coarse_radius=4, coarse_highpass=3, fine_highpass=5,
        correlation_radius=3,
    ).eval()


def test_closed_form_inverse_matches_linalg():
    matrix = torch.randn(5, 3, 3, dtype=torch.float64) + 3 * torch.eye(3, dtype=torch.float64)
    assert torch.allclose(inverse_3x3(matrix), torch.linalg.inv(matrix), atol=1e-10)


def test_the_planar_fit_is_the_same_with_closed_form_algebra():
    torch.manual_seed(1)
    displacement = torch.randn(2, 2, 9, 16) * 0.01
    weight = torch.rand(2, 1, 9, 16)
    x = torch.linspace(-0.5, 0.5, 16).view(1, 1, 16).expand(2, 9, 16)
    y = torch.linspace(-0.3, 0.3, 9).view(1, 9, 1).expand(2, 9, 16)
    s = torch.ones(2, 9, 16)
    kwargs = dict(constraint_normal=torch.tensor([[0.0, 0.0, 1.0]] * 2),
                  constraint_value=torch.zeros(2), constraint_strength=300.0)
    reference = fit_planar_translation(displacement, weight, s, x, y, **kwargs)
    with onnx_safe_linalg():
        closed = fit_planar_translation(displacement, weight, s, x, y, **kwargs)
    for key in ("u", "covariance", "residual_rms"):
        assert torch.allclose(reference[key], closed[key], rtol=1e-4, atol=1e-7), key


@pytest.mark.parametrize("size,out", [((36, 64), 6), ((37, 65), (6, 12)), ((7, 9), (12, 16))])
def test_matmul_adaptive_pool_is_exact(size, out):
    x = torch.randn(2, 5, *size)
    assert torch.allclose(adaptive_avg_pool2d_matmul(x, out), F.adaptive_avg_pool2d(x, out), atol=1e-6)


def test_exported_frontend_and_step_match_pytorch(tmp_path):
    frontend = small_frontend()
    front = PlanarFrontendExport(frontend, camera_matrix()).eval()
    front_path = tmp_path / "frontend.onnx"
    _export(front, frontend_example(front, 3, (HEIGHT, WIDTH), 1), front_path,
            front.input_names, front.output_names, 17, dynamic_batch=False)
    for trial in range(2):
        diffs = verify_frontend(front, front_path, frontend_example(front, 3, (HEIGHT, WIDTH), 1, seed=5 + trial))
        assert max(diffs.values()) < 1e-3, diffs

    torch.manual_seed(0)
    model = VisionMambaVO(visual_dim=16, frontend=frontend, velocity_mode="geometric_residual").eval()
    # Non-zero heads, so the comparison is not trivially zero = zero.
    with torch.no_grad():
        for head in (model.residual_head, model.direction_head, model.log_rate_head):
            head.weight.normal_(0.0, 0.05)
    step = TemporalStepExport(model).eval()
    step_path = tmp_path / "temporal_step.onnx"
    one_tick = temporal_example(step, 1, 1)[0]
    state0 = tuple(torch.zeros(1, *shape) for _, shape in state_layout(model))
    _export(step, one_tick + state0, step_path, step.input_names, step.output_names, 17, dynamic_batch=True)
    for batch in (1, 2):
        diffs = verify_temporal(step, step_path, temporal_example(step, 120, batch, seed=batch))
        assert max(diffs.values()) < 1e-3, diffs


def test_the_heads_mode_step_exports_without_velocity_inputs(tmp_path):
    torch.manual_seed(0)
    model = VisionMambaVO(visual_dim=16).eval()
    step = TemporalStepExport(model).eval()
    assert "visual_velocity" not in step.input_names
    path = tmp_path / "step.onnx"
    one_tick = temporal_example(step, 1, 1)[0]
    state0 = tuple(torch.zeros(1, *shape) for _, shape in state_layout(model))
    _export(step, one_tick + state0, path, step.input_names, step.output_names, 17, dynamic_batch=True)
    diffs = verify_temporal(step, path, temporal_example(step, 60, 1))
    assert max(diffs.values()) < 1e-3, diffs


# --- the standalone runtime's numpy helpers against the project's own -------

def test_runtime_rotation_helpers_match_the_project():
    import numpy as np

    from tools.onnx_inference import (
        euler_to_quaternion,
        quaternion_multiply,
        quaternion_to_matrix,
        quaternion_to_rotvec,
    )
    from vio.models.pose_geometry import (
        euler_zyx_to_quaternion_np,
        quaternion_conjugate_np,
        quaternion_multiply_np,
        quaternion_to_matrix_np,
        quaternion_to_rotvec_np,
    )

    rng = np.random.default_rng(0)
    for _ in range(20):
        euler = rng.uniform([-1.0, -0.6, -3.0], [1.0, 0.6, 3.0])
        other = euler + rng.normal(0.0, 0.05, 3)
        q0, q1 = euler_to_quaternion(*euler), euler_to_quaternion(*other)
        assert np.allclose(q0, euler_zyx_to_quaternion_np(euler), atol=1e-12)
        assert np.allclose(quaternion_to_matrix(q0), quaternion_to_matrix_np(q0), atol=1e-12)
        relative = quaternion_multiply(q0 * np.array([1, -1, -1, -1]), q1)
        expected = quaternion_multiply_np(quaternion_conjugate_np(q0), q1)
        assert np.allclose(relative, expected, atol=1e-12)
        assert np.allclose(quaternion_to_rotvec(relative), quaternion_to_rotvec_np(expected), atol=1e-10)
