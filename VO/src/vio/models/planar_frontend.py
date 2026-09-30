"""Image pair to metric velocity, by flat-ground geometry plus a learned stem.

:class:`~vio.models.vision_mamba_vo.VisionMambaFlowFrontend` measures image
motion and leaves every conversion to metres per second to the network: the
rotation to remove is a learned small-angle field, the scale is a learned
log-rate head. That is the right trade at one frame of separation and low
altitude, where the motion is a few cells and the rotation between exposures
a fraction of a degree. At 200 m it is not: one frame of separation moves the
ground about 0.66 cells at 20 m/s, below the matcher's resolution, and the
remedy - a baseline of half a second to a second - brings rotations of ten
degrees and more, where the small-angle field is wrong by several cells.

This frontend keeps the learned part (the stem that turns pixels into
features) and replaces the rest with the closed form in
:mod:`vio.models.planar_geometry`:

1. **Exact de-rotation.** The second frame is warped by the infinite
   homography of the interframe rotation, which the attitude log gives
   exactly. What remains between the two images is the camera's translation
   over a plane.
2. **Coarse stage, voted on by the whole image.** High-passed features,
   pooled 4x, correlated in a window centred on the motion a nominal velocity
   would produce; the mean correlation over every cell picks one global
   offset. A per-cell argmax on an untrained stem is mostly noise, the image-
   wide vote is not - see :func:`~vio.models.planar_geometry.global_offset_peak`.
3. **Fine stage, per cell.** A narrow search around the flow the coarse
   estimate predicts for each cell (tilt included), then a robust weighted
   least-squares fit of the camera translation over the plane, with the
   altimeter pinning the component along the ground normal. Run twice, the
   second time centred on the first fit.
4. **Metric velocity.** ``t = u h``, rotated into the body frame at the MIDDLE
   of the exposure interval - the frame in which a steady turn's chord points
   straight ahead.

The output carries everything the older frontend's does (a token, a quality,
a delivery verdict, the same diagnostics), plus ``geometric_velocity`` - a
per-pair body velocity in m/s with a covariance - which
:class:`~vio.models.vision_mamba_vo.VisionMambaVO` in its
``geometric_residual`` mode uses as the base its own output corrects.

Measured on a flight rendered by ``tools/make_synthetic_flight.py`` at 200 m,
20 m/s, 20 Hz, 1024x576 with f = 1052 px, S-turns to 25 deg of bank, and an
UNTRAINED stem: per-pair velocity error 0.20 m/s RMS at a one-second baseline
(x 0.10, y 0.17, z 0.08), 0.33 m/s at half a second.

The camera mounting (``camera_from_body``) must be known - see
``tools/estimate_camera_mounting.py``. A small learned correction on top of it
(three parameters, zero-initialised) absorbs a degree or two of misalignment.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .correlation import LocalCorrelation
from .planar_geometry import (
    axis_angle_to_matrix,
    cell_grid,
    cells_to_normalized,
    fit_planar_translation,
    global_offset_peak,
    ground_normal_camera,
    half_rotation,
    local_highpass,
    matrix_to_axis_angle,
    normalized_to_cells,
    planar_displacement,
    plane_scale,
    relative_rotation_camera,
    rotation_homography,
    warp_by_homography,
)
from .vision_mamba_vo import VisionMambaFlowFrontend, frontend_diagnostics

#: Per-pair scalars appended to the pooled cells before the token projection,
#: in this order. Named so a reader of a checkpoint can tell what the token was
#: built from, and so a test can pin the layout.
GEOMETRY_TOKEN_CHANNELS: Tuple[str, ...] = (
    "velocity_x_scaled",
    "velocity_y_scaled",
    "velocity_z_scaled",
    "log_sigma_x",
    "log_sigma_y",
    "log_sigma_z",
    "residual_rms_cells",
    "inlier_fraction",
    "fit_weight_fraction",
    "coarse_margin",
    "rotation_angle_rad",
    "log_altitude_100m",
    "geometric_valid",
)


class PlanarFlowFrontend(VisionMambaFlowFrontend):
    """Flat-ground, attitude-de-rotated frontend with a metric velocity output.

    Subclasses :class:`VisionMambaFlowFrontend` for its stem, its fine
    correlator and - above all - its reliability gate, so a gated planar run
    rejects cells by exactly the rules an older gated run did. The learned
    small-angle rotation field is removed: rotation is taken out exactly, by
    warping, before any feature is computed.
    """

    frontend_id = "planar_homography_v1"
    #: The trainer and the evaluators hand this frontend the per-pair attitude
    #: geometry (relative rotation, ground normal, altitudes) only when this is
    #: True - the older frontend's call signature has no room for it.
    requires_pair_geometry = True

    def __init__(
        self,
        *,
        camera_from_body: Sequence[Sequence[float]],
        prior_velocity_body: Sequence[float] = (20.0, 0.0, 0.0),
        input_channels: int = 1,
        visual_dim: int = 64,
        d_model: int = 64,
        depth: int = 2,
        patch_size: int = 8,
        image_size: Tuple[int, int] = (576, 1024),
        context_grid: Tuple[int, int] = (12, 16),
        d_state: int = 8,
        expand: int = 2,
        correlation_radius: int = 3,
        correlation_temperature: float = 0.03,
        token_grid: int = 6,
        dropout: float = 0.0,
        coarse_factor: int = 4,
        coarse_radius: int = 6,
        coarse_highpass: int = 5,
        fine_highpass: int = 9,
        fine_iterations: int = 2,
        huber_cells: float = 1.0,
        altitude_constraint: float = 300.0,
        learn_mounting: bool = True,
        velocity_scale: float = 20.0,
        max_speed_m_s: float = 80.0,
        min_fit_cells: float = 8.0,
        min_pool_weight: float = 1e-4,
        max_cell_entropy: float = 1.0,
        min_cell_confidence: float = 0.0,
        min_score_margin: float = 0.0,
        reject_boundary_peaks: bool = False,
        min_reliable_cell_fraction: float = 0.0,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            visual_dim=visual_dim,
            d_model=d_model,
            depth=depth,
            patch_size=patch_size,
            image_size=image_size,
            context_grid=context_grid,
            d_state=d_state,
            expand=expand,
            correlation_radius=correlation_radius,
            correlation_temperature=correlation_temperature,
            token_grid=token_grid,
            dropout=dropout,
            rotation_mode="field",
            min_pool_weight=min_pool_weight,
            max_cell_entropy=max_cell_entropy,
            min_cell_confidence=min_cell_confidence,
            min_score_margin=min_score_margin,
            reject_boundary_peaks=reject_boundary_peaks,
            min_reliable_cell_fraction=min_reliable_cell_fraction,
        )
        # Rotation is removed by warping; a learned field would have no input
        # and receive no gradient, which static-graph DDP does not survive.
        del self.rotation
        if int(coarse_factor) < 1 or int(coarse_radius) < 1:
            raise ValueError("coarse_factor and coarse_radius must be positive")
        if int(fine_iterations) < 1:
            raise ValueError("fine_iterations must be at least one")
        fine_size = self.stem.feature_size
        if min(fine_size) // int(coarse_factor) < 2 * int(coarse_radius) + 1:
            raise ValueError(
                f"a {coarse_factor}x-pooled {fine_size[0]}x{fine_size[1]} feature map "
                f"is too small for a coarse search radius of {coarse_radius}"
            )
        self.coarse_factor = int(coarse_factor)
        self.coarse_radius = int(coarse_radius)
        self.coarse_highpass = int(coarse_highpass)
        self.fine_highpass = int(fine_highpass)
        self.fine_iterations = int(fine_iterations)
        self.huber_cells = float(huber_cells)
        self.altitude_constraint = float(altitude_constraint)
        self.velocity_scale = float(velocity_scale)
        self.max_speed_m_s = float(max_speed_m_s)
        self.min_fit_cells = float(min_fit_cells)
        # The coarse stage only places the fine window, and it runs under
        # no_grad, so its temperature is fixed rather than learned.
        self.coarse_correlation = LocalCorrelation(
            radius=self.coarse_radius, temperature=correlation_temperature
        )
        mounting = torch.as_tensor(camera_from_body, dtype=torch.float32)
        if mounting.shape != (3, 3):
            raise ValueError("camera_from_body must be 3x3")
        self.register_buffer("camera_from_body", mounting.clone())
        prior = torch.as_tensor(prior_velocity_body, dtype=torch.float32).reshape(-1)
        if prior.shape != (3,) or not torch.all(torch.isfinite(prior)):
            raise ValueError("prior_velocity_body must be three finite numbers")
        self.register_buffer("prior_velocity_body", prior.clone())
        self.learn_mounting = bool(learn_mounting)
        if self.learn_mounting:
            self.mounting_correction = nn.Parameter(torch.zeros(3))
        else:
            self.register_buffer("mounting_correction", torch.zeros(3))

        cell_dim = (self.cell_channels + self.grid_dim) * self.token_grid ** 2
        pair_dim = cell_dim + 1 + len(GEOMETRY_TOKEN_CHANNELS)
        hidden = max(2 * self.visual_dim, d_model)
        self.token_projection = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.visual_dim),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(pair_dim, max(self.visual_dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(self.visual_dim // 2, 8), 1),
        )

    # ------------------------------------------------------------------ geometry

    def effective_mounting(self) -> torch.Tensor:
        """``camera_from_body`` with the learned correction applied, ``(3, 3)``.

        The correction rotates the CAMERA frame, i.e. it is a misalignment of
        the camera on its mount.
        """

        correction = axis_angle_to_matrix(self.mounting_correction.reshape(1, 3))[0]
        return correction.to(self.camera_from_body.dtype) @ self.camera_from_body

    def forward(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
        *,
        pair_dt_s: Optional[torch.Tensor] = None,
        body_rate_rad_s: Optional[torch.Tensor] = None,
        camera_matrix: Optional[torch.Tensor] = None,
        relative_rotation: Optional[torch.Tensor] = None,
        down_body: Optional[torch.Tensor] = None,
        altitude_m: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """``relative_rotation`` ``(B, 3, 3)``, ``down_body`` ``(B, 3)`` and
        ``altitude_m`` ``(B, 2)`` come from
        :func:`vio.data.attitude.pair_geometry`; ``camera_matrix`` is at the
        WORKING resolution. ``body_rate_rad_s`` is accepted and ignored so the
        two frontends can be called the same way."""

        del body_rate_rad_s
        if image0.ndim != 4 or image0.shape != image1.shape:
            raise ValueError("images must be matching (B, C, H, W) tensors")
        if camera_matrix is None:
            raise ValueError(
                "the planar frontend needs the camera intrinsics (--calibration): "
                "metric velocity is focal length times angle"
            )
        if relative_rotation is None or down_body is None or altitude_m is None:
            raise ValueError(
                "the planar frontend needs relative_rotation, down_body and "
                "altitude_m for every pair (vio.data.attitude.pair_geometry)"
            )
        batch = image0.shape[0]
        device, dtype = image0.device, image0.dtype
        if pair_dt_s is None:
            raise ValueError("pair_dt_s is required: the velocity is a displacement over it")
        dt = pair_dt_s.reshape(batch).to(device=device, dtype=dtype)
        if torch.any(~torch.isfinite(dt)) or torch.any(dt <= 0):
            raise ValueError("pair_dt_s must be finite and positive")
        matrix = camera_matrix.to(device=device, dtype=dtype)
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0).expand(batch, 3, 3)
        rotation_body = relative_rotation.reshape(batch, 3, 3).to(device=device, dtype=dtype)
        down = down_body.reshape(batch, 3).to(device=device, dtype=dtype)
        altitude = altitude_m.reshape(batch, 2).to(device=device, dtype=dtype)
        height0 = altitude[:, 0].clamp_min(1.0)
        height1 = altitude[:, 1].clamp_min(1.0)

        mounting = self.effective_mounting().to(device=device, dtype=dtype)
        mounting_batch = mounting.unsqueeze(0).expand(batch, 3, 3)
        rotation_camera = relative_rotation_camera(rotation_body, mounting_batch)
        warped1, warp_valid = warp_by_homography(
            image1, rotation_homography(matrix, rotation_camera)
        )

        features0 = self.stem(image0)
        features1 = self.stem(warped1)
        fine_size = (features0.shape[2], features0.shape[3])
        fine_valid = (F.adaptive_avg_pool2d(warp_valid, fine_size) > 0.99).to(dtype)
        normal = ground_normal_camera(down, mounting_batch)
        # The altimeter's word on the motion along the normal: n . u.
        along_normal = (height0 - height1) / height0
        constraint = dict(
            constraint_normal=normal,
            constraint_value=along_normal,
            constraint_strength=self.altitude_constraint,
        )

        # --- coarse: one image-wide offset, no gradient -------------------
        with torch.no_grad():
            coarse0 = local_highpass(
                F.avg_pool2d(features0.detach(), self.coarse_factor), self.coarse_highpass
            )
            coarse1 = local_highpass(
                F.avg_pool2d(features1.detach(), self.coarse_factor), self.coarse_highpass
            )
            coarse_size = (coarse0.shape[2], coarse0.shape[3])
            coarse_cell = float(self.patch_size * self.coarse_factor)
            x_c, y_c = cell_grid(matrix, coarse_size, coarse_cell)
            s_c = plane_scale(x_c, y_c, normal)
            prior_camera = (mounting_batch @ self.prior_velocity_body.to(dtype).view(1, 3, 1)).squeeze(-1)
            u_prior = prior_camera * (dt / height0).unsqueeze(-1)
            centre = normalized_to_cells(
                planar_displacement(u_prior, s_c, x_c, y_c), matrix, coarse_cell
            ).round()
            coarse_valid = (F.adaptive_avg_pool2d(warp_valid, coarse_size) > 0.99).to(dtype)
            coarse = self.coarse_correlation(
                coarse0, coarse1, search_center=centre, target_valid=coarse_valid
            )
            raw = coarse["logits"] * coarse["temperature"]
            offset, coarse_margin = global_offset_peak(raw, coarse["valid"], self.coarse_radius)
            shifted = cells_to_normalized(
                centre + offset.view(batch, 2, 1, 1), matrix, coarse_cell
            )
            coarse_fit = fit_planar_translation(
                shifted, coarse_valid, s_c, x_c, y_c, iterations=0, **constraint
            )
            u = coarse_fit["u"]

        # --- fine: per cell, robust planar fit ------------------------------
        fine0 = local_highpass(features0, self.fine_highpass)
        fine1 = local_highpass(features1, self.fine_highpass)
        fine_cell = float(self.patch_size)
        x_f, y_f = cell_grid(matrix, fine_size, fine_cell)
        s_f = plane_scale(x_f, y_f, normal)
        huber = self.huber_cells * fine_cell / matrix[:, 0, 0].mean()
        correlation: Dict[str, torch.Tensor] = {}
        fit: Dict[str, torch.Tensor] = {}
        measured = torch.zeros(batch, 2, *fine_size, device=device, dtype=dtype)
        for iteration in range(self.fine_iterations):
            last = iteration == self.fine_iterations - 1
            centre_f = normalized_to_cells(
                planar_displacement(u.detach(), s_f, x_f, y_f), matrix, fine_cell
            ).detach()
            # Only the last pass is differentiated; the earlier ones merely
            # place its window. nullcontext, never enable_grad: under an
            # evaluator's no_grad this must stay graph-free.
            context = nullcontext() if last else torch.no_grad()
            with context:
                correlation = self.correlation(
                    fine0, fine1, search_center=centre_f, target_valid=fine_valid
                )
                measured = cells_to_normalized(correlation["flow"], matrix, fine_cell)
                reliable_cell = self._reliable_cells(correlation, dtype)
                weight = (
                    correlation["usable_confidence"]
                    * correlation["flow_valid"].to(dtype)
                    * reliable_cell
                )
                fit = fit_planar_translation(
                    measured, weight, s_f, x_f, y_f, huber=float(huber), **constraint
                )
                u = fit["u"]

        # --- metric velocity in the mid-interval body frame -----------------
        translation_camera = u * height0.unsqueeze(-1)
        velocity_first = (mounting.T @ translation_camera.unsqueeze(-1)).squeeze(-1) / dt.unsqueeze(-1)
        halfway = half_rotation(rotation_body)
        velocity = (halfway.transpose(1, 2) @ velocity_first.unsqueeze(-1)).squeeze(-1)
        covariance_u = fit["covariance"]
        to_mid = halfway.transpose(1, 2) @ mounting.T.unsqueeze(0)
        scale = (height0 / dt).view(batch, 1, 1)
        covariance = to_mid @ (covariance_u * scale * scale) @ to_mid.transpose(1, 2)
        variance = covariance.diagonal(dim1=1, dim2=2).clamp_min(1e-8)
        log_variance = variance.log()

        valid_cells = correlation["flow_valid"].to(dtype)
        total_cells = valid_cells.sum(dim=(1, 2, 3))
        speed = velocity.norm(dim=-1)
        geometric_valid = (
            torch.isfinite(velocity).all(dim=-1)
            & (speed <= self.max_speed_m_s)
            & (fit["total_weight"] >= self.min_fit_cells)
        ).to(dtype).unsqueeze(-1)

        # --- token: pooled cells + the per-pair geometry --------------------
        model_flow = planar_displacement(u, s_f, x_f, y_f)
        per_second = dt.view(batch, 1, 1, 1)
        residual_flow = (measured - model_flow) / per_second
        plane_flow = model_flow / per_second
        weight = (
            correlation["usable_confidence"]
            * correlation["flow_valid"].to(dtype)
            * self._reliable_cells(correlation, dtype)
        )
        valid_count = valid_cells.sum(dim=(1, 2, 3)).clamp_min(1.0)
        reliable_cell = self._reliable_cells(correlation, dtype)
        reliable_fraction = reliable_cell.sum(dim=(1, 2, 3)) / valid_count
        pair_reliable = (
            reliable_fraction >= self.min_reliable_cell_fraction
        ).to(dtype).reshape(batch, 1)
        grid = (self.token_grid, self.token_grid)
        weight_pooled = F.adaptive_avg_pool2d(weight, grid)
        flow_pooled = F.adaptive_avg_pool2d(residual_flow * weight, grid) / (
            weight_pooled.clamp_min(self.min_pool_weight)
        )
        gate_weight = correlation["gate_usable_confidence"] * valid_cells * reliable_cell
        occupied = (F.adaptive_avg_pool2d(gate_weight, grid) > self.min_pool_weight).to(dtype)
        flow_pooled = flow_pooled * occupied
        pooled = torch.cat(
            (
                flow_pooled,
                weight_pooled,
                F.adaptive_avg_pool2d(correlation["usable_confidence"], grid),
                F.adaptive_avg_pool2d(correlation["probability_peak_margin"], grid),
                F.adaptive_avg_pool2d(correlation["entropy"], grid),
                F.adaptive_avg_pool2d(valid_cells, grid),
                F.adaptive_avg_pool2d(plane_flow, grid),
            ),
            dim=1,
        )
        cells_sequence = pooled.flatten(2).transpose(1, 2)
        context, _ = self.grid_encoder.forward_sequence(self.cell_projection(cells_sequence))
        cells = torch.cat((pooled.flatten(1), context.flatten(1)), dim=-1)
        rotation_angle = matrix_to_axis_angle(rotation_body).norm(dim=-1)
        residual_rms_cells = fit["residual_rms"] * matrix[:, 0, 0] / fine_cell
        safe_velocity = torch.where(
            geometric_valid > 0, velocity, torch.zeros_like(velocity)
        )
        geometry = torch.stack(
            (
                *(safe_velocity / self.velocity_scale).unbind(-1),
                *(0.5 * log_variance).clamp(-5.0, 3.0).unbind(-1),
                residual_rms_cells.clamp(0.0, 5.0),
                fit["inlier_fraction"],
                fit["total_weight"] / total_cells.clamp_min(1.0),
                coarse_margin.to(dtype).clamp(-1.0, 1.0),
                rotation_angle,
                torch.log(height0 / 100.0),
                geometric_valid.squeeze(-1),
            ),
            dim=-1,
        )
        geometry = torch.nan_to_num(geometry, nan=0.0, posinf=0.0, neginf=0.0)
        pair = torch.cat((cells, torch.log(dt.clamp_min(1e-6)).unsqueeze(-1), geometry), dim=-1)

        diagnostics = frontend_diagnostics(
            correlation, gate_weight, min_pool_weight=self.min_pool_weight,
            reliable_cell=reliable_cell, pair_reliable=pair_reliable,
            reliable_fraction=reliable_fraction,
            rejections=self.rejection_reasons(correlation, dtype),
        )
        return {
            "visual_token": self.token_projection(pair),
            "visual_quality": torch.sigmoid(self.quality_head(pair)),
            # A pair whose geometry did not solve is not delivered: the held
            # velocity then keeps its last good value and visual_age keeps
            # growing, exactly the "no image arrived" contract.
            "pair_reliable": pair_reliable * geometric_valid,
            "geometric_velocity": velocity,
            "geometric_log_variance": log_variance,
            "geometric_valid": geometric_valid,
            "translational_flow_normalized_per_s": residual_flow,
            "plane_flow_normalized_per_s": plane_flow,
            "confidence_map": correlation["confidence"],
            "pooled_cells": pooled,
            "diagnostics": diagnostics,
            "geometry": {
                "u": u,
                "u_coarse": coarse_fit["u"],
                "coarse_offset_cells": offset,
                "coarse_margin": coarse_margin,
                "residual_rms_cells": residual_rms_cells,
                "inlier_fraction": fit["inlier_fraction"],
                "fit_weight": fit["total_weight"],
                "rotation_angle_rad": rotation_angle,
                "warp_valid_fraction": warp_valid.mean(dim=(1, 2, 3)),
            },
        }


__all__ = ["GEOMETRY_TOKEN_CHANNELS", "PlanarFlowFrontend"]
