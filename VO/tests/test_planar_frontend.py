"""The flat-ground frontend turns attitude, altitude and two images into a
metric velocity in closed form, so almost every way it can be wrong is a sign,
an axis or a frame - and each of those produces a plausible-looking number.

These tests pin the geometry against a renderer that knows the truth exactly:
a nadir pinhole camera over a textured plane, posed by the same body-to-NED
conventions the loader uses. A swapped axis, a transposed rotation or a wrong
mounting shows up here as an error of the order of the flight speed, not as a
slightly worse validation curve.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from vio.data.attitude import AttitudeAltitude, pair_geometry, quaternion_at
from vio.data.fixedwing_vo import hold_visual_velocity, mask_velocity_carry
from vio.models.planar_frontend import GEOMETRY_TOKEN_CHANNELS, PlanarFlowFrontend
from vio.models.planar_geometry import (
    NADIR_MOUNTINGS,
    axis_angle_to_matrix,
    cell_grid,
    fit_planar_translation,
    global_offset_peak,
    half_rotation,
    local_highpass,
    matrix_to_axis_angle,
    mounting_matrix,
    planar_displacement,
    plane_scale,
    relative_rotation_camera,
    rotation_homography,
    warp_by_homography,
)
from vio.models.pose_geometry import euler_zyx_to_quaternion_np, quaternion_to_matrix_np
from vio.models.velocity_horizons import SpanTokens, scatter_span_velocity
from vio.models.vision_mamba_vo import VisionMambaVO

HEIGHT, WIDTH = 288, 512
FOCAL = 526.0


def camera_matrix() -> torch.Tensor:
    return torch.tensor(
        [[FOCAL, 0.0, (WIDTH - 1) / 2.0], [0.0, FOCAL, (HEIGHT - 1) / 2.0], [0.0, 0.0, 1.0]]
    )


def ground_texture(seed: int = 3, size: int = 1024) -> np.ndarray:
    """Multi-scale noise: detail at every scale a correlator might lock onto."""

    import cv2

    rng = np.random.default_rng(seed)
    texture = np.zeros((size, size), dtype=np.float32)
    for sigma, weight in ((1.0, 0.5), (3.0, 1.0), (9.0, 1.0)):
        # Periodic, so the texture tiles seamlessly under BORDER_WRAP remaps;
        # GaussianBlur has no wrap border of its own.
        pad = int(4 * sigma) + 1
        noise = np.pad(rng.normal(size=(size, size)).astype(np.float32), pad, mode="wrap")
        layer = cv2.GaussianBlur(noise, (0, 0), sigma)[pad:-pad, pad:-pad]
        texture += weight * layer / layer.std()
    texture -= texture.min()
    return (255.0 * texture / texture.max()).astype(np.uint8)


def render(
    texture: np.ndarray,
    euler_rpy: np.ndarray,
    position_ned: np.ndarray,
    camera_from_body: np.ndarray,
    *,
    texel_m: float = 1.0,
) -> torch.Tensor:
    """What a camera mounted by ``camera_from_body`` sees of the ground plane
    D = 0 from ``position_ned`` at body attitude ``euler_rpy``, ``(1,1,H,W)``."""

    import cv2

    rotation_world_body = quaternion_to_matrix_np(euler_zyx_to_quaternion_np(euler_rpy))
    rotation_world_camera = rotation_world_body @ camera_from_body.T
    u, v = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
    rays = np.stack(
        ((u - (WIDTH - 1) / 2.0) / FOCAL, (v - (HEIGHT - 1) / 2.0) / FOCAL, np.ones_like(u)),
        axis=-1,
    ) @ rotation_world_camera.T
    scale = -position_ned[2] / rays[..., 2]
    north = position_ned[0] + scale * rays[..., 0]
    east = position_ned[1] + scale * rays[..., 1]
    size = texture.shape[0]
    image = cv2.remap(
        texture,
        np.mod(east / texel_m, size).astype(np.float32),
        np.mod(north / texel_m, size).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return torch.from_numpy(image.astype(np.float32) / 255.0)[None, None]


def two_views(
    mounting: str,
    *,
    velocity_body=(20.0, 1.5, 0.8),
    euler0=(0.25, 0.05, 0.4),
    rates=(0.04, -0.02, 0.15),
    dt: float = 1.0,
    altitude: float = 200.0,
):
    """Two exposures one ``dt`` apart, and the truth the frontend should find.

    The aircraft flies ``velocity_body`` (expressed in the MID-interval body
    frame, which is what the frontend reports) while its attitude changes at
    ``rates`` - a banked, turning, slightly climbing pair.
    """

    camera_from_body = np.asarray(NADIR_MOUNTINGS[mounting], dtype=np.float64)
    euler0 = np.asarray(euler0, dtype=np.float64)
    euler1 = euler0 + np.asarray(rates) * dt
    euler_mid = 0.5 * (euler0 + euler1)
    rotation_mid = quaternion_to_matrix_np(euler_zyx_to_quaternion_np(euler_mid))
    velocity_ned = rotation_mid @ np.asarray(velocity_body, dtype=np.float64)
    start = np.array([35.0, -12.0, -altitude])
    finish = start + velocity_ned * dt
    texture = ground_texture()
    image0 = render(texture, euler0, start, camera_from_body)
    image1 = render(texture, euler1, finish, camera_from_body)

    q0 = euler_zyx_to_quaternion_np(euler0)
    q1 = euler_zyx_to_quaternion_np(euler1)
    r0 = quaternion_to_matrix_np(q0)
    r1 = quaternion_to_matrix_np(q1)
    geometry = {
        "relative_rotation": torch.tensor(r0.T @ r1, dtype=torch.float32)[None],
        "down_body": torch.tensor(r0.T @ np.array([0.0, 0.0, 1.0]), dtype=torch.float32)[None],
        "altitude_m": torch.tensor([[-start[2], -finish[2]]], dtype=torch.float32),
    }
    # The exact mean velocity over the pair, in the frame the frontend uses:
    # the displacement rotated into the mid-interval body frame by the
    # geodesic half-rotation (not the Euler midpoint).
    half = half_rotation(geometry["relative_rotation"].double())[0].numpy()
    truth = half.T @ (r0.T @ (finish - start)) / dt
    return image0, image1, geometry, truth, camera_from_body


def make_frontend(mounting: str, **overrides) -> PlanarFlowFrontend:
    torch.manual_seed(0)
    settings = dict(
        camera_from_body=NADIR_MOUNTINGS[mounting],
        prior_velocity_body=(18.0, 0.0, 0.0),
        image_size=(HEIGHT, WIDTH),
        d_model=32, depth=1, visual_dim=16, context_grid=(6, 8), token_grid=4,
        coarse_factor=2, coarse_radius=4, coarse_highpass=3, fine_highpass=5,
        correlation_radius=3,
    )
    settings.update(overrides)
    return PlanarFlowFrontend(**settings).eval()


# --------------------------------------------------------------------------
# rotations and frames
# --------------------------------------------------------------------------


def test_axis_angle_round_trips_and_half_rotation_composes_to_the_whole():
    torch.manual_seed(0)
    vectors = torch.randn(16, 3, dtype=torch.float64) * 0.6
    matrices = axis_angle_to_matrix(vectors)
    assert torch.allclose(matrix_to_axis_angle(matrices), vectors, atol=1e-8)
    identity = torch.eye(3, dtype=torch.float64).expand(16, 3, 3)
    assert torch.allclose(matrices @ matrices.transpose(1, 2), identity, atol=1e-10)
    half = half_rotation(matrices)
    assert torch.allclose(half @ half, matrices, atol=1e-8)


def test_axis_angle_has_a_finite_gradient_at_zero():
    """The learned mounting correction starts at exactly zero."""

    vector = torch.zeros(1, 3, requires_grad=True)
    axis_angle_to_matrix(vector).sum().backward()
    assert torch.all(torch.isfinite(vector.grad))


@pytest.mark.parametrize("name", sorted(NADIR_MOUNTINGS))
def test_every_named_mounting_looks_straight_down(name):
    matrix = mounting_matrix(name).double()
    assert torch.allclose(matrix @ matrix.T, torch.eye(3, dtype=torch.float64), atol=1e-9)
    assert float(torch.linalg.det(matrix)) == pytest.approx(1.0)
    # Body down is the optical axis in every nadir mounting.
    assert torch.allclose(matrix @ torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
                          torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))


def test_mounting_refuses_a_mirror():
    mirrored = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="proper rotation"):
        mounting_matrix(mirrored)


def test_top_forward_puts_the_nose_at_the_top_of_the_image():
    """Forward flight must move the ground DOWN the image for this mount."""

    matrix = mounting_matrix("top_forward")
    forward_in_camera = matrix @ torch.tensor([1.0, 0.0, 0.0])
    # Image "up" is -y in the camera frame.
    assert torch.allclose(forward_in_camera, torch.tensor([0.0, -1.0, 0.0]))


def test_pair_geometry_is_the_exact_relative_rotation_and_the_down_axis():
    times = np.linspace(0.0, 2.0, 201)
    euler = np.stack(
        (0.3 * np.sin(times), 0.1 * times, 0.5 * times), axis=1
    )
    quaternion = euler_zyx_to_quaternion_np(euler)
    attitude = AttitudeAltitude(
        times_s=times, euler_rad=euler, quaternion=quaternion,
        body_rate_rad_s=np.zeros_like(euler), altitude_m=200.0 + times,
        attitude_columns=("NavEulX", "NavEulY", "NavEulZ"), altitude_column="RelativeAlt",
        euler_unit="radians", attitude_hold_fraction=0.0, notes=(),
    )
    geometry = pair_geometry(attitude, 0.505, 1.505)
    r0 = quaternion_to_matrix_np(quaternion_at(times, quaternion, 0.505))
    r1 = quaternion_to_matrix_np(quaternion_at(times, quaternion, 1.505))
    assert np.allclose(geometry["relative_rotation"], r0.T @ r1, atol=1e-6)
    roll, pitch = np.interp(0.505, times, euler[:, 0]), np.interp(0.505, times, euler[:, 1])
    # NED down in the body frame depends on roll and pitch only.
    expected_down = np.array(
        [-math.sin(pitch), math.sin(roll) * math.cos(pitch), math.cos(roll) * math.cos(pitch)]
    )
    assert np.allclose(geometry["down_body"], expected_down, atol=1e-4)
    assert np.allclose(geometry["altitude_m"], [200.505, 201.505], atol=1e-5)


# --------------------------------------------------------------------------
# the closed form
# --------------------------------------------------------------------------


def test_the_planar_fit_inverts_the_planar_flow_exactly_even_tilted():
    K = camera_matrix()[None]
    x, y = cell_grid(K, (36, 64), 8.0)
    normal = torch.nn.functional.normalize(torch.tensor([[0.2, -0.3, 1.0]]), dim=-1)
    s = plane_scale(x, y, normal)
    u = torch.tensor([[0.09, -0.012, 0.006]])
    flow = planar_displacement(u, s, x, y)
    fit = fit_planar_translation(flow, torch.ones(1, 1, 36, 64), s, x, y)
    assert torch.allclose(fit["u"], u, atol=1e-6)


def test_the_robust_fit_shrugs_off_a_block_of_wrong_matches():
    torch.manual_seed(0)
    K = camera_matrix()[None]
    x, y = cell_grid(K, (36, 64), 8.0)
    s = plane_scale(x, y, torch.tensor([[0.0, 0.0, 1.0]]))
    u = torch.tensor([[0.1, 0.01, 0.0]])
    flow = planar_displacement(u, s, x, y) + 0.001 * torch.randn(1, 2, 36, 64)
    flow[:, :, :9, :16] += 0.2  # a quarter-sized corner of confident nonsense
    weight = torch.ones(1, 1, 36, 64)
    plain = fit_planar_translation(flow, weight, s, x, y, iterations=0)
    robust = fit_planar_translation(flow, weight, s, x, y, iterations=4, huber=0.01)
    assert (robust["u"] - u).abs().max() < 0.2 * (plain["u"] - u).abs().max()
    assert (robust["u"] - u).abs().max() < 2e-3


def test_the_altimeter_row_fixes_the_component_along_the_normal():
    torch.manual_seed(1)
    K = camera_matrix()[None]
    x, y = cell_grid(K, (36, 64), 8.0)
    normal = torch.nn.functional.normalize(torch.tensor([[0.1, 0.2, 1.0]]), dim=-1)
    s = plane_scale(x, y, normal)
    u = torch.tensor([[0.1, 0.0, 0.01]])
    # Noise large enough that the image alone gets the normal component wrong.
    flow = planar_displacement(u, s, x, y) + 0.004 * torch.randn(1, 2, 36, 64)
    along = (normal * u).sum(dim=-1)
    fit = fit_planar_translation(
        flow, torch.ones(1, 1, 36, 64), s, x, y,
        constraint_normal=normal, constraint_value=along, constraint_strength=1e4,
    )
    assert float((normal * fit["u"]).sum()) == pytest.approx(float(along), abs=1e-5)


def test_the_rotation_warp_undoes_a_pure_rotation():
    """Two views from one point differing only by a rotation must coincide
    after the warp, wherever the warped image has content."""

    mounting = np.asarray(NADIR_MOUNTINGS["top_forward"], dtype=np.float64)
    texture = ground_texture()
    position = np.array([0.0, 0.0, -200.0])
    euler0 = np.array([0.1, 0.05, 0.2])
    euler1 = euler0 + np.array([0.05, -0.03, 0.2])
    image0 = render(texture, euler0, position, mounting)
    image1 = render(texture, euler1, position, mounting)
    r0 = quaternion_to_matrix_np(euler_zyx_to_quaternion_np(euler0))
    r1 = quaternion_to_matrix_np(euler_zyx_to_quaternion_np(euler1))
    rotation_camera = relative_rotation_camera(
        torch.tensor(r0.T @ r1, dtype=torch.float32)[None],
        torch.tensor(mounting, dtype=torch.float32)[None],
    )
    warped, valid = warp_by_homography(image1, rotation_homography(camera_matrix()[None], rotation_camera))
    inside = valid > 0.5
    before = (image0 - image1).abs()[inside].mean()
    after = (image0 - warped).abs()[inside].mean()
    assert float(valid.mean()) > 0.5
    assert float(after) < 0.1 * float(before)


def test_the_global_vote_finds_a_shift_most_cells_cannot():
    """A known shift, recovered from the MEAN correlation of mostly-noise cells."""

    from vio.models.correlation import LocalCorrelation

    torch.manual_seed(0)
    features = torch.randn(1, 16, 24, 40)
    shifted = torch.roll(features, shifts=(2, -3), dims=(2, 3))
    noisy = shifted + 2.5 * torch.randn_like(shifted)
    correlation = LocalCorrelation(radius=4, temperature=0.03)(features, noisy)
    raw = correlation["logits"] * correlation["temperature"]
    offset, margin = global_offset_peak(raw, correlation["valid"], 4)
    assert torch.allclose(offset.round(), torch.tensor([[-3.0, 2.0]]))
    assert float(margin) > 0.0


def test_local_highpass_removes_a_constant_and_refuses_an_even_kernel():
    features = torch.full((1, 3, 10, 12), 5.0)
    assert torch.allclose(local_highpass(features, 3), torch.zeros_like(features), atol=1e-6)
    assert local_highpass(features, 1) is features
    with pytest.raises(ValueError, match="odd"):
        local_highpass(features, 4)


# --------------------------------------------------------------------------
# the frontend, end to end on rendered pairs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mounting", ["right_forward", "top_forward"])
def test_the_untrained_frontend_measures_a_banked_turning_pair(mounting):
    """20 m/s at 200 m with a 20 deg bank, a 9 deg/s turn and a climb, over
    one second - the regime the frontend exists for. Untrained features;
    the geometry alone has to carry it. A wrong axis or sign is ~20 m/s."""

    image0, image1, geometry, truth, _ = two_views(mounting)
    frontend = make_frontend(mounting)
    with torch.no_grad():
        out = frontend(
            image0, image1, pair_dt_s=torch.tensor([1.0]), camera_matrix=camera_matrix(),
            **geometry,
        )
    velocity = out["geometric_velocity"][0].numpy()
    assert float(out["geometric_valid"]) == 1.0
    assert np.abs(velocity - truth).max() < 1.0, (velocity, truth)
    # The vertical is pinned by the altimeter, not guessed from image scale.
    assert abs(velocity[2] - truth[2]) < 0.3


def test_the_frontend_output_carries_every_diagnostic_and_the_token_layout():
    from vio.models.vision_mamba_vo import DIAGNOSTIC_NAMES

    image0, image1, geometry, _, _ = two_views("right_forward")
    frontend = make_frontend("right_forward")
    with torch.no_grad():
        out = frontend(
            image0, image1, pair_dt_s=torch.tensor([1.0]), camera_matrix=camera_matrix(),
            **geometry,
        )
    assert set(DIAGNOSTIC_NAMES) <= set(out["diagnostics"])
    assert out["visual_token"].shape == (1, 16)
    assert out["geometric_log_variance"].shape == (1, 3)
    assert len(GEOMETRY_TOKEN_CHANNELS) == 13
    assert not hasattr(frontend, "rotation")


def test_the_frontend_trains_its_stem_and_mounting_through_the_velocity():
    image0, image1, geometry, truth, _ = two_views("right_forward")
    frontend = make_frontend("right_forward").train()
    out = frontend(
        image0, image1, pair_dt_s=torch.tensor([1.0]), camera_matrix=camera_matrix(), **geometry
    )
    loss = (out["geometric_velocity"] - torch.tensor(truth, dtype=torch.float32)).pow(2).sum()
    loss = loss + out["visual_token"].pow(2).sum() + out["visual_quality"].sum()
    loss.backward()
    missing = [name for name, p in frontend.named_parameters() if p.grad is None]
    assert missing == []
    assert float(frontend.stem.patch_embed.weight.grad.abs().sum()) > 0.0
    assert float(frontend.mounting_correction.grad.abs().sum()) > 0.0


def test_the_frontend_stays_graph_free_under_no_grad():
    image0, image1, geometry, _, _ = two_views("right_forward")
    frontend = make_frontend("right_forward")
    with torch.no_grad():
        out = frontend(
            image0, image1, pair_dt_s=torch.tensor([1.0]), camera_matrix=camera_matrix(),
            **geometry,
        )
    assert not out["geometric_velocity"].requires_grad
    assert not out["visual_token"].requires_grad


def test_the_frontend_refuses_to_run_without_its_geometry():
    image0, image1, geometry, _, _ = two_views("right_forward")
    frontend = make_frontend("right_forward")
    with pytest.raises(ValueError, match="relative_rotation"):
        frontend(image0, image1, pair_dt_s=torch.tensor([1.0]), camera_matrix=camera_matrix())
    with pytest.raises(ValueError, match="intrinsics"):
        frontend(image0, image1, pair_dt_s=torch.tensor([1.0]), **geometry)


# --------------------------------------------------------------------------
# holding the per-pair velocity, and the residual output
# --------------------------------------------------------------------------


def test_hold_places_on_the_ready_tick_holds_until_replaced_and_skips_refusals():
    velocity = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    offsets = torch.tensor([[2, 5, 7]])
    valid = torch.ones(1, 3)
    delivered = torch.tensor([[1.0, 0.0, 1.0]])
    held, held_valid, carry = hold_visual_velocity(
        velocity, offsets, valid, window_length=10, delivered=delivered
    )
    assert held[0, :, 0].tolist() == [0, 0, 1, 1, 1, 1, 1, 3, 3, 3]
    assert held_valid[0, :, 0].tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    assert carry[0][0, 0] == 3.0 and bool(carry[1][0])

    # A continuing chunk starts from the carry; a reset one does not.
    later = torch.tensor([[[9.0, 0.0, 0.0]]])
    carried, carried_valid, _ = hold_visual_velocity(
        later, torch.tensor([[3]]), torch.ones(1, 1), window_length=5, carry=carry
    )
    assert carried[0, :, 0].tolist() == [3, 3, 3, 9, 9]
    assert carried_valid[0, :, 0].tolist() == [1, 1, 1, 1, 1]
    reset = mask_velocity_carry(carry, torch.tensor([False]))
    fresh, fresh_valid, _ = hold_visual_velocity(
        later, torch.tensor([[3]]), torch.ones(1, 1), window_length=5, carry=reset
    )
    assert fresh[0, :, 0].tolist() == [0, 0, 0, 9, 9]
    assert fresh_valid[0, :, 0].tolist() == [0, 0, 0, 1, 1]


def test_the_streamed_hold_matches_the_windowed_one():
    tokens = SpanTokens(
        tick=np.array([12, 15, 17]),
        token=torch.zeros(3, 4),
        quality=torch.zeros(3, 1),
        visual_dim=4,
        pair_reliable=torch.tensor([[1.0], [0.0], [1.0]]),
        velocity=torch.tensor([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]),
    )
    held, held_valid = scatter_span_velocity(tokens, [(10, 20)], 10, device=torch.device("cpu"))
    windowed, windowed_valid, _ = hold_visual_velocity(
        tokens.velocity[None], torch.tensor([[2, 5, 7]]), torch.ones(1, 3),
        window_length=10, delivered=tokens.pair_reliable.reshape(1, 3),
    )
    assert torch.equal(held, windowed)
    assert torch.equal(held_valid, windowed_valid)


def test_geometric_residual_starts_as_the_held_velocity_and_heads_before_it():
    torch.manual_seed(0)
    model = VisionMambaVO(visual_dim=8, aiding_dim=8, fusion_dim=8, velocity_mode="geometric_residual")
    assert model.temporal_input_id == "vo_temporal_fusion_v2+geometric_residual"
    assert VisionMambaVO(visual_dim=8, aiding_dim=8, fusion_dim=8).temporal_input_id == "vo_temporal_fusion_v2"
    batch, ticks = 2, 12
    held = torch.randn(batch, ticks, 3) * 10
    held_valid = torch.zeros(batch, ticks, 1)
    held_valid[:, 4:] = 1.0
    out = model(
        torch.randn(batch, ticks, 9), torch.zeros(batch, ticks, 8),
        torch.zeros(batch, ticks, 1), torch.zeros(batch, ticks, 1),
        log_altitude=torch.full((batch, ticks), math.log(200.0)),
        visual_velocity=held, visual_velocity_valid=held_valid,
    )
    velocity = out["predicted_velocity"]
    assert torch.allclose(velocity[:, 4:], held[:, 4:], atol=1e-6)
    # Before the first delivery the factored heads answer: forward, at h*exp(init).
    assert torch.allclose(velocity[:, :4, 1:], torch.zeros(batch, 4, 2), atol=1e-6)
    assert float(velocity[0, 0, 0].detach()) > 0.0
    with pytest.raises(ValueError, match="visual_velocity"):
        model(
            torch.randn(batch, ticks, 9), torch.zeros(batch, ticks, 8),
            torch.zeros(batch, ticks, 1), torch.zeros(batch, ticks, 1),
            log_altitude=torch.zeros(batch, ticks),
        )
