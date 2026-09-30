"""Local correspondence between two feature maps.

Recovered from the earlier dense frontend. A cosine cost volume over a
bounded search window, resolved to sub-pixel displacement by soft-argmax,
is the primitive an image frontend needs in order to represent motion:
global descriptors can say that a scene changed, not what moved where.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCorrelation(nn.Module):
    """Cosine correlation with sub-pixel soft-argmax refinement.

    ``search_center`` is an optional predicted rotational displacement, in
    feature-cell units - in this project it comes from attitude rather than a
    gyro (see :class:`~vio.models.vision_mamba_vo.RotationalSearchField`),
    but the correlator itself is agnostic to where it came from.
    It can be shaped ``(2,)``, ``(B, 2)``, or ``(B, 2, H, W)``. Channel zero is
    horizontal displacement (dx), channel one is vertical displacement (dy).
    Fractional centers use bilinear sampling; no center uses the faster integer
    shift path.
    """

    def __init__(
        self,
        radius: int = 4,
        temperature: float = 0.1,
        *,
        center_features: bool = True,
        learnable_temperature: bool = False,
        refine_radius: int = 1,
    ) -> None:
        super().__init__()
        if radius < 0:
            raise ValueError("radius must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        offsets = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
        self.radius = int(radius)
        self.center_features = bool(center_features)
        # Soft-argmax is biased towards zero displacement by exactly as much as
        # the correlation peak is broad, and on an untrained stem it is very
        # broad. Measured here on a known two-cell shift, cosine features,
        # 81 candidates:
        #
        #     temperature   0.10   0.05   0.02   0.01
        #     recovered     0.53   0.88   1.45   1.78   cells of 2.0
        #
        # A frontend that under-reports motion by 4x under-reports velocity by
        # 4x, and the attenuation is not constant - it tracks peak sharpness,
        # so it varies with texture and becomes a scene-dependent multiplicative
        # error rather than a bias a later layer can absorb. Making the
        # temperature learnable lets training sharpen the window as the features
        # become discriminative, instead of fixing the bias at its worst value.
        # Two-stage argmax. A soft-argmax over the WHOLE candidate set is pulled
        # toward the window centre by every low-probability candidate, so it
        # under-reports displacement by an amount that tracks peak sharpness -
        # the scene-dependent gain error the table above measures. Taking the
        # integer argmax first and refining only within +/-refine_radius of it
        # removes that pull: the far candidates are never summed over, and the
        # refinement window follows the peak instead of being nailed to zero.
        # ``refine_radius=0`` restores the single-stage behaviour.
        if refine_radius < 0:
            raise ValueError("refine_radius must be non-negative")
        self.refine_radius = int(refine_radius)
        self.learnable_temperature = bool(learnable_temperature)
        if self.learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(float(math.log(temperature)))
            )
        else:
            self.temperature = float(temperature)
        # A SECOND, fixed temperature, used only for the statistics a caller
        # gates on - never for the displacement estimate.
        #
        # Confidence and entropy are properties of the SOFTMAX, not of the
        # match: shrinking the temperature sharpens the distribution and so
        # raises confidence and lowers entropy on identical images. Measured
        # here, 0.03 -> 0.002 moves usable confidence 0.77 -> 0.99 and
        # normalized entropy 0.155 -> 0.007 with the inputs untouched. A
        # learnable temperature therefore lets the model decide whether its own
        # measurement passes a quality gate, and - even with no gradient
        # pushing it that way - makes a fixed threshold mean something
        # different at epoch 100 than at epoch 1.
        #
        # Defaulting to the INITIAL temperature keeps the gate statistics
        # identical to the pre-gate ones at initialisation, which is what makes
        # a threshold chosen on an untrained frontend (tools/
        # sweep_reliability_gate.py) still mean the same thing later.
        self.gate_temperature = float(temperature)
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.float32), persistent=False
        )

    @staticmethod
    def _center_map(
        center: torch.Tensor, batch: int, height: int, width: int, reference: torch.Tensor
    ) -> torch.Tensor:
        center = torch.as_tensor(center, device=reference.device, dtype=reference.dtype)
        if center.shape == (2,):
            center = center.view(1, 2, 1, 1).expand(batch, 2, height, width)
        elif center.shape == (batch, 2):
            center = center.view(batch, 2, 1, 1).expand(batch, 2, height, width)
        elif center.shape != (batch, 2, height, width):
            raise ValueError(
                "search_center must have shape (2,), (B,2), or (B,2,H,W)"
            )
        if not torch.isfinite(center).all():
            raise ValueError("search_center contains NaN or infinity")
        # The centre is rounded to whole cells, and this is a correctness fix
        # rather than an optimisation.
        #
        # A fractional centre makes ``_all_grid_samples`` resample the second
        # feature map at fractional positions, so every candidate compares a
        # SHARP feature against a BILINEARLY BLENDED one. The blend mixes cells
        # that describe different pieces of ground, which lowers and broadens
        # the correlation peak, and a broadened peak is exactly what the
        # soft-argmax under-reports. Measured on a known two-cell shift with an
        # exact patch descriptor, sweeping only the centre:
        #
        #     centre    0.00  -0.25  -0.50  -0.75  -1.00  -1.50  -2.00
        #     recovered -2.00  -1.20  -1.00  -1.33  -2.00  -1.66  -2.00
        #
        # Integer centres are exact; fractional ones lose up to half the
        # displacement. It survives feature smoothing and every texture scale
        # tested. Worse, the error is a function of the FRACTIONAL PART of a
        # centre that is itself proportional to turn rate, so it appears as a
        # turn-correlated multiplicative error on measured flow - the one shape
        # of error this frontend is least able to absorb.
        #
        # Nothing is lost by rounding: the centre exists to place a +/-radius
        # window, and half a cell of placement error is immaterial against a
        # radius of four, while ALL sub-cell precision comes from the
        # soft-argmax, which the table above shows is exact on an integer
        # lattice. Straight-through, so the map that produces the centre still
        # receives a gradient - rounding alone would zero it and the rotation
        # map could never be learned at all.
        return center + (center.round() - center).detach()

    def _all_integer_shifts(
        self, feature: torch.Tensor, padded: torch.Tensor, height: int, width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Every candidate's integer-shift correlation input, in one call.

        Replaces a Python loop of up to 81 slices with one ``F.unfold``: for
        every output pixel, unfold extracts the very same padded neighbourhood
        each offset used to slice by hand, in one im2col-style kernel launch
        instead of up to 81 sequential ones. Numerically identical to calling
        the old per-offset slice for every entry of ``self._offset_tuples`` -
        pinned by ``test_the_vectorised_and_looped_correlation_paths_agree``.
        """

        batch, channels = feature.shape[0], feature.shape[1]
        side = 2 * self.radius + 1
        # unfold's unrolled dimension is ordered channel-major, then kernel
        # row, then kernel column - i.e. (kh, kw) = (dy+radius, dx+radius) in
        # row-major order, which is exactly the order self._offset_tuples (dy
        # outer, dx inner) already iterates in, so no reordering is needed.
        patches = F.unfold(padded, kernel_size=side)
        shifted = patches.view(batch, channels, side, side, height, width)
        shifted = shifted.reshape(batch, channels, side * side, height, width)

        device = feature.device
        h_index = torch.arange(height, device=device).view(1, 1, height, 1)
        w_index = torch.arange(width, device=device).view(1, 1, 1, width)
        offsets = self.offsets.to(device=device).round().long()
        dx_all = offsets[:, 0].view(1, -1, 1, 1)
        dy_all = offsets[:, 1].view(1, -1, 1, 1)
        valid = (
            (h_index + dy_all >= 0) & (h_index + dy_all < height)
            & (w_index + dx_all >= 0) & (w_index + dx_all < width)
        ).expand(batch, -1, height, width)
        return shifted, valid

    def _all_grid_samples(
        self, feature: torch.Tensor, center: torch.Tensor, height: int, width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Every candidate's fractional-centre correlation input, in one call.

        Replaces a Python loop issuing one ``F.grid_sample`` per offset (up to
        81 separate kernel launches, the dominant cost of this module - see
        the module-level docstring) with a single call: every offset's
        sampling grid is stacked along a K axis and folded into grid_sample's
        output-height dimension, since grid_sample samples every output
        position independently and has no interaction across it, then
        unfolded back afterwards. Numerically identical to the old per-offset
        call - pinned by
        ``test_the_vectorised_and_looped_correlation_paths_agree``.
        """

        batch, channels = feature.shape[0], feature.shape[1]
        device, dtype = feature.device, feature.dtype
        candidates = self.offsets.shape[0]
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        xs = xs.view(1, 1, height, width)
        ys = ys.view(1, 1, height, width)
        offsets = self.offsets.to(device=device, dtype=dtype)
        dx_all = offsets[:, 0].view(1, -1, 1, 1)
        dy_all = offsets[:, 1].view(1, -1, 1, 1)
        target_x = xs + center[:, 0:1] + dx_all  # (B, K, H, W)
        target_y = ys + center[:, 1:2] + dy_all
        valid = (
            (target_x >= 0)
            & (target_x <= width - 1)
            & (target_y >= 0)
            & (target_y <= height - 1)
        )

        if width > 1:
            grid_x = target_x.mul(2.0 / (width - 1)).sub(1.0)
        else:
            grid_x = torch.zeros_like(target_x)
        if height > 1:
            grid_y = target_y.mul(2.0 / (height - 1)).sub(1.0)
        else:
            grid_y = torch.zeros_like(target_y)
        # (B, K, H, W, 2) folded to (B, K*H, W, 2): one grid_sample call, one
        # kernel launch, producing (B, C, K*H, W) unfolded back below.
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(batch, candidates * height, width, 2)
        sampled = F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.reshape(batch, channels, candidates, height, width)
        return sampled, valid

    def _target_validity(
        self,
        target_valid: torch.Tensor,
        center_map: Optional[torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Whether each candidate's TARGET cell holds real image content.

        Sampled with the very same integer-shift or grid-sample machinery the
        features go through, so the mask and the correlation cannot disagree
        about which cell a candidate reads. Nearest rather than bilinear
        semantics via the 0.5 threshold: a candidate half on missing pixels is
        treated as missing.
        """

        mask = target_valid.to(dtype=torch.float32)
        if center_map is None:
            padded = F.pad(mask, (self.radius, self.radius, self.radius, self.radius))
            sampled, _ = self._all_integer_shifts(mask, padded, height, width)
        else:
            sampled, _ = self._all_grid_samples(mask, center_map.to(mask.dtype), height, width)
        return sampled[:, 0] > 0.5

    def forward(
        self,
        feature_t: torch.Tensor,
        feature_tp1: torch.Tensor,
        search_center: Optional[torch.Tensor] = None,
        target_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """``target_valid`` is an optional ``(B, 1, H, W)`` mask on the SECOND
        map: 1 where it holds real image content. A second frame warped by a
        rotation homography has regions no pixel of the original reached, and
        a candidate landing there compares against zero-padding - a black
        border that correlates with nothing, or worse, with any dark ground.
        Such candidates are removed exactly like candidates off the edge of
        the map, so every statistic below (confidence floor, entropy,
        ``flow_valid``) accounts for them.
        """

        if feature_t.ndim != 4 or feature_t.shape != feature_tp1.shape:
            raise ValueError("Feature maps must be matching BxCxHxW tensors")
        if not feature_t.is_floating_point() or not feature_tp1.is_floating_point():
            raise TypeError("Feature maps must be floating-point tensors")
        batch, _, height, width = feature_t.shape
        if target_valid is not None and target_valid.shape != (batch, 1, height, width):
            raise ValueError("target_valid must have shape (B, 1, H, W)")
        if self.center_features:
            # Subtract the component every position shares before normalising.
            # Without it the cosine is dominated by the map's DC term: on a
            # random-weight stem the median similarity across the whole search
            # window measures 0.78, so the true peak carries almost no weight
            # and soft-argmax collapses towards zero displacement. Removing it
            # drops the median to 0.02 and recovers the shift exactly.
            feature_t = feature_t - feature_t.mean(dim=(2, 3), keepdim=True)
            feature_tp1 = feature_tp1 - feature_tp1.mean(dim=(2, 3), keepdim=True)
        left = F.normalize(feature_t, dim=1, eps=1e-6)
        right = F.normalize(feature_tp1, dim=1, eps=1e-6)

        center_map = None
        if search_center is not None:
            center_map = self._center_map(search_center, batch, height, width, left)

        if center_map is None:
            padded_right = F.pad(
                right, (self.radius, self.radius, self.radius, self.radius)
            )
            shifted, valid = self._all_integer_shifts(right, padded_right, height, width)
        else:
            shifted, valid = self._all_grid_samples(right, center_map, height, width)
        if target_valid is not None:
            valid = valid & self._target_validity(target_valid, center_map, height, width)
        # (B, C, K, H, W) x (B, C, 1, H, W) -> sum over C -> (B, K, H, W), the
        # same per-candidate cosine correlation the old loop built one
        # candidate at a time.
        correlations = (left.unsqueeze(2) * shifted).sum(dim=1)

        temperature = (
            torch.exp(self.log_temperature)
            if self.learnable_temperature
            else self.temperature
        )
        logits = correlations.div(temperature)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(logits, dim=1)
        any_valid = valid.any(dim=1, keepdim=True)
        full_window_valid = valid.all(dim=1, keepdim=True)
        candidate_valid_fraction = valid.float().mean(dim=1, keepdim=True)
        probabilities = torch.where(any_valid, probabilities, torch.zeros_like(probabilities))
        offsets = self.offsets.to(device=probabilities.device, dtype=probabilities.dtype)
        local_flow = self._displacement(logits, probabilities, offsets, any_valid)
        flow = local_flow if center_map is None else local_flow + center_map
        confidence = probabilities.amax(dim=1, keepdim=True)
        if probabilities.shape[1] > 1:
            top_probabilities = probabilities.topk(2, dim=1).values
            probability_peak_margin = top_probabilities[:, 0:1] - top_probabilities[:, 1:2]
        else:
            probability_peak_margin = confidence
        entropy = -(
            probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        ).sum(dim=1, keepdim=True)
        # Confidence with the no-information floor removed. A uniform
        # distribution over n candidates peaks at 1/n, not at 0, so raw
        # ``confidence`` never reaches zero however ambiguous the match is -
        # and a weighted mean over a wholly ambiguous region cancels the small
        # weight out again, returning the arbitrary tie-break at full size. The
        # divisor is the VALID count, so a border cell with fewer candidates is
        # judged against its own floor rather than against 1/81.
        candidate_count = valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        uniform = 1.0 / candidate_count
        usable_confidence = (
            (confidence - uniform) / (1.0 - uniform).clamp_min(torch.finfo(confidence.dtype).eps)
        ).clamp_min(0.0)
        # Diagnostic, not used by the displacement estimate: whether the
        # WINNING candidate sits on the edge of the search window rather than
        # its interior. A soft-argmax can only ever report a displacement
        # inside the window it was given - it has no way to say "the true
        # match is further out" - so a peak pinned to the boundary is the one
        # symptom of a too-small radius that would otherwise be invisible in
        # the reported flow itself. Meaningless where nothing was valid at all
        # (the argmax there is an arbitrary tie-break, not a measurement), so
        # masked to zero there the same way usable_confidence is.
        side = 2 * self.radius + 1
        if self.radius > 0:
            peak_index = logits.argmax(dim=1)
            peak_row = torch.div(peak_index, side, rounding_mode="floor")
            peak_col = peak_index % side
            peak_at_boundary = (
                (peak_row == 0) | (peak_row == side - 1)
                | (peak_col == 0) | (peak_col == side - 1)
            ).unsqueeze(1)
        else:
            peak_at_boundary = torch.zeros_like(any_valid)
        peak_at_boundary = (peak_at_boundary & any_valid).to(logits.dtype)

        # --- Gating statistics -------------------------------------------
        # Same quantities as above, but at the FIXED gate temperature and
        # DETACHED. Detached because a reliability gate is a measurement of
        # the data, not something the optimiser should be able to push on:
        # with these in the graph, "make the gate accept me" would be a route
        # to a lower loss that has nothing to do with predicting velocity.
        gate_logits = correlations.detach().div(self.gate_temperature)
        gate_logits = gate_logits.masked_fill(~valid, torch.finfo(gate_logits.dtype).min)
        gate_probabilities = torch.softmax(gate_logits, dim=1)
        gate_probabilities = torch.where(
            any_valid, gate_probabilities, torch.zeros_like(gate_probabilities)
        )
        gate_confidence = gate_probabilities.amax(dim=1, keepdim=True)
        gate_usable_confidence = (
            (gate_confidence - uniform)
            / (1.0 - uniform).clamp_min(torch.finfo(gate_confidence.dtype).eps)
        ).clamp_min(0.0)
        gate_entropy = -(
            gate_probabilities
            * gate_probabilities.clamp_min(
                torch.finfo(gate_probabilities.dtype).tiny
            ).log()
        ).sum(dim=1, keepdim=True)

        # The temperature-FREE alternative: how far the best raw correlation
        # score beats the runner-up. No softmax is involved, so no choice of
        # temperature can move it at all. Exposed alongside the softmax
        # statistics so a caller can gate on either, and so the two can be
        # compared on real data rather than argued about.
        if correlations.shape[1] > 1:
            masked_scores = correlations.detach().masked_fill(
                ~valid, torch.finfo(correlations.dtype).min
            )
            top_scores = masked_scores.topk(2, dim=1).values
            score_margin = (top_scores[:, 0:1] - top_scores[:, 1:2]) * any_valid.to(
                correlations.dtype
            )
        else:
            score_margin = torch.zeros_like(confidence)

        return {
            # Gate-side statistics: fixed temperature, detached. A caller
            # deciding whether to BELIEVE a cell wants these; a caller using
            # the displacement wants the learned-temperature ones below.
            "gate_usable_confidence": gate_usable_confidence,
            "gate_entropy": gate_entropy,
            "score_margin": score_margin,
            # The temperature actually in force, so a run can log whether the
            # learned value has drifted away from the gate's fixed reference.
            "temperature": temperature.detach()
            if isinstance(temperature, torch.Tensor)
            else torch.tensor(float(temperature), device=logits.device),
            "logits": logits,
            "probabilities": probabilities,
            "flow": flow,
            "local_flow": local_flow,
            "confidence": confidence,
            "usable_confidence": usable_confidence,
            "probability_peak_margin": probability_peak_margin,
            "entropy": entropy,
            "peak_at_boundary": peak_at_boundary,
            "valid": valid,
            # ``flow_valid`` keeps the public name but now has the conservative
            # meaning required for a symmetric soft-argmax window. The old
            # any-candidate mask admitted strongly biased border estimates.
            "flow_valid": full_window_valid,
            "full_window_valid": full_window_valid,
            "any_candidate_valid": any_valid,
            "candidate_valid_fraction": candidate_valid_fraction,
            # Per-cell valid-candidate count, already computed above for
            # usable_confidence's own floor. Exposed so a caller normalising
            # entropy (max possible entropy is log(candidate_count), which a
            # border cell's clipped window makes SMALLER than log(K)) can use
            # each cell's own ceiling instead of the radius-only log(K).
            "candidate_count": candidate_count,
            "offsets": offsets,
        }


    def _displacement(
        self,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
        offsets: torch.Tensor,
        any_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Sub-cell displacement from the correlation volume.

        With ``refine_radius=0`` this is the plain expectation over every
        candidate. Otherwise it is the expectation over a small window centred
        on the integer peak, which is what removes the shrinkage: a candidate
        four cells away can no longer drag the estimate toward zero because it
        is not in the sum at all.
        """

        side = 2 * self.radius + 1
        r = self.refine_radius
        if r == 0 or self.radius == 0 or side < 2 * r + 1:
            return torch.einsum("bkhw,kd->bdhw", probabilities, offsets)

        batch, _, height, width = logits.shape
        peak = logits.argmax(dim=1)
        # The window is centred on the ACTUAL winner and truncated where it runs
        # off the candidate grid, rather than sliding the centre inwards to make
        # it fit. Sliding it would refine a winner at +4 over {+2,+3,+4} and pull
        # the estimate inward exactly where displacement is already saturating.
        centre_y = torch.div(peak, side, rounding_mode="floor")
        centre_x = peak % side

        step = torch.arange(-r, r + 1, device=logits.device)
        rows = centre_y.unsqueeze(1) + step.view(1, -1, 1, 1)
        cols = centre_x.unsqueeze(1) + step.view(1, -1, 1, 1)
        rows_inside = (rows >= 0) & (rows < side)
        cols_inside = (cols >= 0) & (cols < side)
        rows = rows.clamp(0, side - 1)
        cols = cols.clamp(0, side - 1)

        index = (rows.unsqueeze(2) * side + cols.unsqueeze(1)).reshape(
            batch, -1, height, width
        )
        # Clamping the indices keeps the gather in range; masking the members
        # that were out of range keeps the duplicate it creates from being
        # counted, so the softmax renormalises over the real neighbours only.
        member_inside = (rows_inside.unsqueeze(2) & cols_inside.unsqueeze(1)).reshape(
            batch, -1, height, width
        )
        gathered = torch.gather(logits, 1, index).masked_fill(
            ~member_inside, torch.finfo(logits.dtype).min
        )
        window = torch.softmax(gathered, dim=1)

        dtype = probabilities.dtype
        span = 2 * r + 1
        delta_x = (cols - self.radius).to(dtype).unsqueeze(1).expand(
            batch, span, span, height, width
        ).reshape(batch, -1, height, width)
        delta_y = (rows - self.radius).to(dtype).unsqueeze(2).expand(
            batch, span, span, height, width
        ).reshape(batch, -1, height, width)

        local_flow = torch.stack(
            ((window * delta_x).sum(dim=1), (window * delta_y).sum(dim=1)), dim=1
        )
        return torch.where(any_valid, local_flow, torch.zeros_like(local_flow))

__all__ = ["LocalCorrelation"]
