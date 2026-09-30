"""Flat-ground geometry for a downward-looking camera, in torch.

At a few hundred metres the ground under a fixed-wing aircraft is, to a very
good approximation, one plane - relief of a few metres is a few percent of the
range. That makes the whole two-view problem closed-form, and this module is
that closed form. Nothing here is learned.

**Rotation is removed exactly, not linearised.** A camera that only rotates
maps pixels by the infinite homography ``H = K R^T K^-1``, whatever the scene
depth. Warping the second frame by it leaves an image that differs from the
first ONLY by the camera's translation. The Longuet-Higgins-Prazdny field the
older frontend uses is the first-order expansion of this map and is fine at
one frame of separation, where the aircraft turns a fraction of a degree; at a
one-second baseline a 16 deg/s turn is 0.28 rad and the expansion is wrong by
several cells at the image edge.

**Translation over a plane is linear in the unknown.** Let ``h`` be the height
above the ground plane, ``n`` its unit normal in the first camera's frame
(pointing from the camera to the ground) and ``t`` the camera displacement in
that same frame. A ground point on the ray ``m = (x, y, 1)`` sits at depth
``Z = h / (n . m)``, and after the de-rotation above its image moves by

    d = s (u_z m - u) / (1 - s u_z),    s = n . m,   u = t / h

in normalized coordinates. Rearranged,

    d_x = s (u_z (x + d_x) - u_x)
    d_y = s (u_z (y + d_y) - u_y)

which is LINEAR in ``u``: every measured cell contributes two rows to a 3x3
weighted least-squares problem, solved in one call. ``t = u h`` is then the
metric displacement, so the scale comes from the altitude exactly as
``v = h * u`` does in the speed head, but with the tilt of the ground (bank and
pitch) and the camera's vertical motion both accounted for per cell instead of
being learned.

Conventions, used everywhere below:

* camera frame: x = image right, y = image down, z = optical axis;
* ``camera_from_body`` rotates a BODY-frame vector into the camera frame;
* rotations are ``(B, 3, 3)`` matrices; ``relative_rotation_body`` is the
  second exposure's body orientation expressed in the first exposure's body
  frame, ``R_b0^T R_b1``;
* displacements are in normalized image coordinates unless a name says cells.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(vector.shape[:-1] + (3, 3))


def axis_angle_to_matrix(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula, batched over leading dimensions.

    The small-angle branch uses the Taylor series so the gradient stays finite
    at exactly zero rotation, which is the common case for a zero-initialised
    mounting correction.
    """

    angle_sq = (rotation_vector * rotation_vector).sum(-1, keepdim=True)
    angle = angle_sq.clamp_min(1e-24).sqrt()
    small = angle_sq < 1e-8
    sin_term = torch.where(small, 1.0 - angle_sq / 6.0, torch.sin(angle) / angle)
    cos_term = torch.where(
        small, 0.5 - angle_sq / 24.0, (1.0 - torch.cos(angle)) / angle_sq.clamp_min(1e-24)
    )
    skew = _skew(rotation_vector)
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    identity = identity.expand(skew.shape)
    return (
        identity
        + sin_term.unsqueeze(-1) * skew
        + cos_term.unsqueeze(-1) * (skew @ skew)
    )


def matrix_to_axis_angle(rotation: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`axis_angle_to_matrix` for angles below pi.

    The interframe rotations this module deals with are at most a few tenths
    of a radian, far from the pi singularity, so the plain log map is enough.
    """

    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    axis = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sin_angle = torch.sin(angle)
    small = sin_angle.abs() < 1e-6
    scale = torch.where(
        small, 0.5 + angle * angle / 12.0, angle / (2.0 * sin_angle.clamp_min(1e-12))
    )
    return axis * scale.unsqueeze(-1)


def half_rotation(rotation: torch.Tensor) -> torch.Tensor:
    """The rotation halfway along the geodesic from identity to ``rotation``.

    A pair measures the AVERAGE velocity over its exposure interval, and that
    average is best expressed in the body frame at the middle of the
    interval: in a steady turn the chord of the arc points along the mid-frame
    forward axis, not along the first frame's.
    """

    return axis_angle_to_matrix(0.5 * matrix_to_axis_angle(rotation))


#: Named nadir mountings, as a camera_from_body rotation. The optical axis is
#: body down in all of them; they differ only in where the TOP of the image
#: points, which a nadir camera can be bolted at in four ways.
NADIR_MOUNTINGS: Dict[str, Tuple[Tuple[float, float, float], ...]] = {
    # image right = body forward, image down = body right (the synthetic
    # renderer's mount: camera axes coincide with body axes)
    "right_forward": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    # image top = body forward, image right = body right
    "top_forward": ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    # image left = body forward, image up = body right
    "left_forward": ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    # image bottom = body forward, image left = body right
    "bottom_forward": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}


def mounting_matrix(mounting: "str | Sequence[Sequence[float]]") -> torch.Tensor:
    """A camera_from_body rotation from a name or an explicit 3x3 matrix.

    An explicit matrix is checked for orthonormality with a positive
    determinant: a mirrored "rotation" would silently flip the lateral axis.
    """

    if isinstance(mounting, str):
        if mounting not in NADIR_MOUNTINGS:
            raise ValueError(
                f"unknown camera mounting {mounting!r}; choose one of "
                f"{sorted(NADIR_MOUNTINGS)} or give a 3x3 camera_from_body matrix"
            )
        values = NADIR_MOUNTINGS[mounting]
    else:
        values = mounting
    matrix = torch.as_tensor(values, dtype=torch.float64)
    if matrix.shape != (3, 3) or not torch.all(torch.isfinite(matrix)):
        raise ValueError("camera_from_body must be a finite 3x3 matrix")
    error = (matrix @ matrix.T - torch.eye(3, dtype=torch.float64)).abs().max()
    if float(error) > 1e-3 or float(torch.linalg.det(matrix)) <= 0.0:
        raise ValueError("camera_from_body must be a proper rotation (orthonormal, det +1)")
    return matrix.to(torch.float32)


# ---------------------------------------------------------------------------
# image-plane geometry
# ---------------------------------------------------------------------------


def cell_grid(
    camera_matrix: torch.Tensor,
    feature_size: Tuple[int, int],
    cell_pixels: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalized coordinates of every cell centre, ``(B, H, W)`` each.

    ``camera_matrix`` is ``(B, 3, 3)`` at the WORKING resolution, and a cell
    spans ``cell_pixels`` working pixels.
    """

    batch = camera_matrix.shape[0]
    height, width = feature_size
    device, dtype = camera_matrix.device, camera_matrix.dtype
    columns = (torch.arange(width, device=device, dtype=dtype) + 0.5) * cell_pixels - 0.5
    rows = (torch.arange(height, device=device, dtype=dtype) + 0.5) * cell_pixels - 0.5
    fx = camera_matrix[:, 0, 0].view(batch, 1, 1)
    fy = camera_matrix[:, 1, 1].view(batch, 1, 1)
    cx = camera_matrix[:, 0, 2].view(batch, 1, 1)
    cy = camera_matrix[:, 1, 2].view(batch, 1, 1)
    x = (columns.view(1, 1, width) - cx) / fx
    y = (rows.view(1, height, 1) - cy) / fy
    return x.expand(batch, height, width), y.expand(batch, height, width)


def rotation_homography(camera_matrix: torch.Tensor, rotation_camera: torch.Tensor) -> torch.Tensor:
    """Pixel of frame 0 -> pixel of frame 1 for a pure camera rotation.

    ``rotation_camera`` is the second camera's orientation in the first
    camera's frame, so a direction ``r0`` in frame 0 has frame-1 coordinates
    ``R^T r0`` and projects to ``K R^T K^-1 p0``.
    """

    inverse = torch.linalg.inv(camera_matrix)
    return camera_matrix @ rotation_camera.transpose(-1, -2) @ inverse


def warp_by_homography(
    image: torch.Tensor, homography: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resample ``image`` so pixel ``p`` of the output reads ``image(H p)``.

    Returns the warped image and a ``(B, 1, H, W)`` mask that is 1 where the
    source pixel existed. Bilinear, zero outside - the mask, not the zeros, is
    what downstream code should trust, because black ground is a legitimate
    pixel value.
    """

    batch, _, height, width = image.shape
    device, dtype = image.device, image.dtype
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1).reshape(1, -1, 3)
    mapped = pixels @ homography.to(dtype).transpose(-1, -2)
    depth = mapped[..., 2:3]
    # A ray behind the second camera cannot happen for the few-tenths-of-a-
    # radian rotations between two frames of one flight; guarded anyway so a
    # corrupt attitude sample produces an empty mask instead of NaNs.
    in_front = depth > 1e-6
    source = mapped[..., :2] / torch.where(in_front, depth, torch.ones_like(depth))
    grid_x = source[..., 0] * (2.0 / max(width - 1, 1)) - 1.0
    grid_y = source[..., 1] * (2.0 / max(height - 1, 1)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(batch, height, width, 2)
    warped = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid = (
        (grid_x.abs() <= 1.0) & (grid_y.abs() <= 1.0) & in_front.squeeze(-1)
    ).reshape(batch, 1, height, width)
    return warped, valid.to(dtype)


def plane_scale(x: torch.Tensor, y: torch.Tensor, normal_camera: torch.Tensor) -> torch.Tensor:
    """``s = n . m`` per cell: the ground depth along each ray is ``h / s``.

    ``normal_camera`` is ``(B, 3)``, the unit ground normal pointing from the
    camera to the ground in the first camera's frame. Clamped away from zero:
    ``s <= 0`` is a ray at or above the horizon, which a nadir camera at a
    sane bank angle never has, and which would otherwise divide by zero.
    """

    n = normal_camera.reshape(-1, 3, 1, 1)
    return (n[:, 0] * x + n[:, 1] * y + n[:, 2]).clamp_min(0.05)


def planar_displacement(
    u: torch.Tensor, s: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """De-rotated image displacement of the ground, ``(B, 2, H, W)``.

    ``u`` is ``(B, 3)``, the camera displacement over the height, in the
    first camera's frame. Exact, not linearised: ``d = s (u_z m - u) / (1 - s u_z)``.
    """

    ux = u[:, 0].reshape(-1, 1, 1)
    uy = u[:, 1].reshape(-1, 1, 1)
    uz = u[:, 2].reshape(-1, 1, 1)
    denominator = (1.0 - s * uz).clamp_min(0.05)
    dx = s * (uz * x - ux) / denominator
    dy = s * (uz * y - uy) / denominator
    return torch.stack((dx, dy), dim=1)


def fit_planar_translation(
    displacement: torch.Tensor,
    weight: torch.Tensor,
    s: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    iterations: int = 3,
    huber: float = 0.01,
    ridge: float = 1e-6,
    constraint_normal: Optional[torch.Tensor] = None,
    constraint_value: Optional[torch.Tensor] = None,
    constraint_strength: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Weighted, robust least squares for ``u`` from measured displacements.

    ``displacement`` is ``(B, 2, H, W)`` normalized, ``weight`` is
    ``(B, 1, H, W)`` (zero for a cell that measured nothing). ``huber`` is the
    residual, in normalized units, beyond which a cell's influence stops
    growing - roughly one correlation cell is the right order. The robust
    re-weighting is computed without gradient, the standard IRLS arrangement:
    the solve itself stays differentiable, so a frontend upstream of it learns
    from the velocity error.

    **The altitude constraint.** The image pins down the two components of
    ``u`` ALONG the ground well - they are a translation of the whole image -
    and the component along the normal only through a percent-level change of
    scale, which sub-cell matching errors swamp. The altimeter measures that
    component directly: the camera's distance to the plane goes from ``h0``
    to ``h1``, so ``n . t = h0 - h1``, i.e. ``n . u = (h0 - h1) / h0``. Passing
    ``constraint_normal`` (``(B, 3)``), ``constraint_value`` (``(B,)``) and a
    positive ``constraint_strength`` adds that as one extra row, weighted at
    ``constraint_strength`` times the mean diagonal of the image information -
    so a strength of a few hundred makes it effectively exact while leaving
    the solve well-posed, and zero reproduces the image-only fit.

    Returns ``u`` ``(B, 3)``, its covariance ``(B, 3, 3)`` from the weighted
    residual scatter, the residual RMS, the fraction of weight that survived
    the robust re-weighting, and the total weight - which is how a caller
    tells a real fit from one solved on nothing but the ridge.
    """

    batch = displacement.shape[0]
    dx = displacement[:, 0]
    dy = displacement[:, 1]
    w = weight.reshape(batch, *dx.shape[1:]).clamp_min(0.0)
    zeros = torch.zeros_like(s)
    # Row blocks: A_x = [-s, 0, s(x+dx)], A_y = [0, -s, s(y+dy)]; b = d.
    a_x = torch.stack((-s, zeros, s * (x + dx)), dim=-1)
    a_y = torch.stack((zeros, -s, s * (y + dy)), dim=-1)
    rows = torch.stack((a_x, a_y), dim=1).reshape(batch, -1, 3)
    targets = torch.stack((dx, dy), dim=1).reshape(batch, -1)
    base = torch.stack((w, w), dim=1).reshape(batch, -1)

    total_weight = base.sum(dim=1) * 0.5
    identity = torch.eye(3, dtype=rows.dtype, device=rows.device).expand(batch, 3, 3)
    robust = torch.ones_like(base)
    constrained = (
        constraint_normal is not None
        and constraint_value is not None
        and constraint_strength > 0.0
    )
    if constrained:
        n_c = constraint_normal.reshape(batch, 3).to(rows.dtype)
        c_value = constraint_value.reshape(batch).to(rows.dtype)
        n_outer = n_c.unsqueeze(-1) * n_c.unsqueeze(-2)

    def system(row_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weighted = rows * row_weight.unsqueeze(-1)
        normal = weighted.transpose(1, 2) @ rows
        rhs = (weighted * targets.unsqueeze(-1)).sum(dim=1)
        scale = normal.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-12)
        normal = normal + ridge * scale.view(-1, 1, 1) * identity
        if constrained:
            # The scale is detached: it sets how much the altimeter row
            # matters, not something the optimiser should be able to shrink.
            lam = float(constraint_strength) * scale.detach()
            normal = normal + lam.view(-1, 1, 1) * n_outer
            rhs = rhs + lam.view(-1, 1) * n_c * c_value.view(-1, 1)
        return normal, rhs

    def solve(row_weight: torch.Tensor) -> torch.Tensor:
        normal, rhs = system(row_weight)
        return torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)

    u = solve(base)
    for _ in range(max(int(iterations), 0)):
        with torch.no_grad():
            residual = (rows @ u.detach().unsqueeze(-1)).squeeze(-1) - targets
            # Both rows of a cell share one weight, driven by the cell's
            # residual MAGNITUDE, so an outlier cell is down-weighted as a
            # unit rather than one coordinate at a time.
            per_cell = residual.reshape(batch, 2, -1).norm(dim=1)
            factor = torch.where(
                per_cell <= huber, torch.ones_like(per_cell), huber / per_cell.clamp_min(1e-12)
            )
            robust = torch.cat((factor, factor), dim=1)
        u = solve(base * robust)

    final_weight = base * robust
    residual = (rows @ u.unsqueeze(-1)).squeeze(-1) - targets
    weight_sum = final_weight.sum(dim=1).clamp_min(1e-12)
    variance = (final_weight * residual.pow(2)).sum(dim=1) / weight_sum
    information, _ = system(final_weight)
    # The effective sample count is the weight sum over the mean weight, so a
    # fit on many low-confidence cells is not credited as if each were exact.
    mean_weight = weight_sum / (final_weight > 0).sum(dim=1).clamp_min(1)
    covariance = torch.linalg.inv(information) * (variance * mean_weight).view(-1, 1, 1)
    return {
        "u": u,
        "covariance": covariance,
        "residual_rms": variance.clamp_min(0.0).sqrt(),
        "inlier_fraction": (final_weight.sum(dim=1) / base.sum(dim=1).clamp_min(1e-12)),
        "total_weight": total_weight,
    }


def local_highpass(features: torch.Tensor, kernel: int) -> torch.Tensor:
    """Subtract each cell's ``kernel x kernel`` neighbourhood mean.

    Correlation over a large search window is at the mercy of the features'
    low-frequency content: a field or a forest several hundred metres across
    makes every shift within it correlate well, and the cost surface becomes a
    broad hill whose top need not be the true motion. On a rendered 200 m
    flight that hill beat the true (sharp) peak on one pair in thirty at a one
    second baseline, a 40 m/s error; removing the local mean first eliminated
    it and also sharpened the fine stage (0.42 -> 0.20 m/s RMS). ``kernel``
    of 1 or less is the identity.
    """

    kernel = int(kernel)
    if kernel <= 1:
        return features
    if kernel % 2 == 0:
        raise ValueError("the high-pass kernel must be odd so it stays centred")
    return features - F.avg_pool2d(
        features, kernel, stride=1, padding=kernel // 2, count_include_pad=False
    )


def global_offset_peak(
    correlations: torch.Tensor,
    valid: torch.Tensor,
    radius: int,
    *,
    min_count_fraction: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One displacement for the whole image, voted on by every cell.

    ``correlations`` is ``(B, K, H, W)`` raw correlation scores over a
    ``(2 radius + 1)^2`` window and ``valid`` the matching mask. After the
    rotation is removed, the ground over a flat plane moves almost uniformly
    through the frame, so summing the cost volumes of all cells and taking
    the peak finds that motion even where most individual cells are
    ambiguous: textureless cells add a flat term and textured ones add a
    peak, and only the peaks line up. This is what makes the coarse stage
    robust enough to centre a narrow fine search on, where a per-cell argmax
    on an untrained feature map is mostly noise.

    The statistic is the plain MEAN of raw correlation scores, deliberately
    not a vote of per-cell softmax probabilities: on an untrained feature map
    the true offset's advantage in any one cell is small but consistent, and
    averaging raw scores is the detector for exactly that; sharpening each
    cell into a vote first throws the weak signal away (measured on a rendered
    200 m flight: 11 failed pairs in 30 with votes, 1 with the mean). An
    offset reachable by only a few cells (the window runs off the map, or
    onto missing pixels of a warped frame) averages over too few of them to
    be trusted, so offsets supported by less than ``min_count_fraction`` of
    the best-supported offset are excluded.

    Returns the sub-cell offset ``(B, 2)`` (dx, dy) in cells relative to the
    window centre - refined by a parabola through the peak and its neighbours
    - and the peak's margin over the best offset outside its 3x3
    neighbourhood, in correlation units: near zero means the whole-image
    match is ambiguous (repetitive texture, or nothing to match at all).
    Computed without gradient; it only places the fine search.
    """

    with torch.no_grad():
        side = 2 * int(radius) + 1
        batch = correlations.shape[0]
        mask = valid.to(correlations.dtype)
        count = mask.sum(dim=(2, 3))
        cost = (correlations * mask).sum(dim=(2, 3)) / count.clamp_min(1.0)
        enough = count >= min_count_fraction * count.amax(dim=1, keepdim=True).clamp_min(1.0)
        floor = torch.finfo(cost.dtype).min / 4
        cost = cost.masked_fill(~enough, floor)
        grid = cost.reshape(batch, side, side)
        peak = cost.argmax(dim=1)
        row = torch.div(peak, side, rounding_mode="floor")
        col = peak % side
        lanes = torch.arange(batch, device=cost.device)

        def at(r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return grid[lanes, r.clamp(0, side - 1), c.clamp(0, side - 1)]

        def vertex(before: torch.Tensor, centre: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
            curvature = before - 2.0 * centre + after
            offset = torch.where(
                curvature < -1e-12,
                0.5 * (before - after) / curvature.clamp_max(-1e-12),
                torch.zeros_like(centre),
            )
            return offset.clamp(-0.5, 0.5)

        centre = at(row, col)
        interior_x = (col > 0) & (col < side - 1)
        interior_y = (row > 0) & (row < side - 1)
        left, right = at(row, col - 1), at(row, col + 1)
        up, down = at(row - 1, col), at(row + 1, col)
        usable_x = interior_x & (left > floor) & (right > floor)
        usable_y = interior_y & (up > floor) & (down > floor)
        sub_x = torch.where(usable_x, vertex(left, centre, right), torch.zeros_like(centre))
        sub_y = torch.where(usable_y, vertex(up, centre, down), torch.zeros_like(centre))
        offset = torch.stack(
            ((col - radius).to(cost.dtype) + sub_x, (row - radius).to(cost.dtype) + sub_y),
            dim=-1,
        )

        rows = torch.arange(side, device=cost.device).view(1, side, 1)
        cols = torch.arange(side, device=cost.device).view(1, 1, side)
        near = ((rows - row.view(-1, 1, 1)).abs() <= 1) & ((cols - col.view(-1, 1, 1)).abs() <= 1)
        away = grid.masked_fill(near, floor).reshape(batch, -1).amax(dim=1)
        margin = torch.where(away > floor, centre - away, centre)
    return offset, margin


def normalized_to_cells(
    displacement: torch.Tensor, camera_matrix: torch.Tensor, cell_pixels: float
) -> torch.Tensor:
    """``(B, 2, H, W)`` normalized displacement -> cells of ``cell_pixels``."""

    fx = camera_matrix[:, 0, 0].reshape(-1, 1, 1, 1)
    fy = camera_matrix[:, 1, 1].reshape(-1, 1, 1, 1)
    scale = torch.cat((fx, fy), dim=1) / float(cell_pixels)
    return displacement * scale


def cells_to_normalized(
    displacement_cells: torch.Tensor, camera_matrix: torch.Tensor, cell_pixels: float
) -> torch.Tensor:
    """Inverse of :func:`normalized_to_cells`."""

    fx = camera_matrix[:, 0, 0].reshape(-1, 1, 1, 1)
    fy = camera_matrix[:, 1, 1].reshape(-1, 1, 1, 1)
    scale = float(cell_pixels) / torch.cat((fx, fy), dim=1)
    return displacement_cells * scale


def ground_normal_camera(down_body: torch.Tensor, camera_from_body: torch.Tensor) -> torch.Tensor:
    """The unit down direction (ground normal) in the camera frame, ``(B, 3)``."""

    normal = (camera_from_body @ down_body.unsqueeze(-1)).squeeze(-1)
    return F.normalize(normal, dim=-1)


def relative_rotation_camera(
    relative_rotation_body: torch.Tensor, camera_from_body: torch.Tensor
) -> torch.Tensor:
    """``R_cb R_rel R_cb^T``: the interframe rotation seen from the camera."""

    return camera_from_body @ relative_rotation_body @ camera_from_body.transpose(-1, -2)


def angular_footprint(camera_matrix: torch.Tensor, image_size: Tuple[int, int]) -> float:
    """Half-diagonal field of view in normalized units - for sanity limits."""

    height, width = image_size
    fx = float(camera_matrix[..., 0, 0].reshape(-1)[0])
    fy = float(camera_matrix[..., 1, 1].reshape(-1)[0])
    return math.hypot(0.5 * width / fx, 0.5 * height / fy)


__all__ = [
    "NADIR_MOUNTINGS",
    "angular_footprint",
    "axis_angle_to_matrix",
    "cell_grid",
    "cells_to_normalized",
    "fit_planar_translation",
    "global_offset_peak",
    "ground_normal_camera",
    "local_highpass",
    "half_rotation",
    "matrix_to_axis_angle",
    "mounting_matrix",
    "normalized_to_cells",
    "plane_scale",
    "planar_displacement",
    "relative_rotation_camera",
    "rotation_homography",
    "warp_by_homography",
]
