"""The VO stack replaces both the flow frontend and the inertial input, so the
properties worth pinning are the ones whose failure is silent.

A scan that disagrees with its own recurrence, an altitude channel that is
concatenated rather than multiplied, a stem that stops translating with the
image, an attitude column that is really the training label - none of these
raise. They train, they converge, and they produce a plausible number that is
wrong in a way no loss curve reveals.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vio.data.attitude import (
    aiding_features,
    body_rates_from_quaternions,
    detect_euler_unit,
    hold_fraction,
    load_attitude_altitude,
    resolve_altitude_column,
    resolve_attitude_columns,
)
from vio.data.fixedwing_vo import mask_age_carry, visual_age_seconds
from vio.models.pose_geometry import euler_zyx_to_quaternion_np
from vio.models.vision_mamba import (
    SCAN_DIRECTIONS,
    SelectiveScan2D,
    VisionMambaStem,
    _from_sequence,
    _to_sequence,
    parallel_selective_scan,
)
from vio.models.vision_mamba_vo import (
    AIDING_INPUT_DIM,
    VisionMambaFlowFrontend,
    VisionMambaVO,
    detach_stream_state,
    frontend_diagnostics,
    mask_stream_state,
)


# --------------------------------------------------------------------------
# the scan
# --------------------------------------------------------------------------


def test_parallel_scan_equals_the_recurrence_it_replaces():
    """The whole reason this exists is speed; if it is not also exact, every
    result downstream is measuring a different model than the one described."""

    torch.manual_seed(0)
    transition = torch.rand(3, 37, 5) * 0.9 + 0.05
    drive = torch.randn(3, 37, 5)
    expected = torch.zeros_like(drive)
    state = torch.zeros(3, 5)
    for index in range(drive.shape[1]):
        state = transition[:, index] * state + drive[:, index]
        expected[:, index] = state
    assert torch.allclose(
        parallel_selective_scan(transition, drive), expected, atol=1e-5
    )


def test_parallel_scan_handles_lengths_that_are_not_powers_of_two():
    for length in (1, 2, 3, 5, 8, 9):
        a = torch.rand(2, length, 3) * 0.8 + 0.1
        b = torch.randn(2, length, 3)
        expected = torch.zeros_like(b)
        state = torch.zeros(2, 3)
        for index in range(length):
            state = a[:, index] * state + b[:, index]
            expected[:, index] = state
        assert torch.allclose(parallel_selective_scan(a, b), expected, atol=1e-5)


def test_parallel_scan_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        parallel_selective_scan(torch.ones(2, 4, 3), torch.ones(2, 4, 5))


@pytest.mark.parametrize("direction", SCAN_DIRECTIONS)
def test_sequence_ordering_round_trips(direction):
    """A transpose or flip that is not undone would scramble the grid while
    keeping every shape correct."""

    features = torch.randn(2, 6, 5, 7)
    sequence = _to_sequence(features, direction)
    assert sequence.shape == (2, 35, 6)
    assert torch.allclose(_from_sequence(sequence, direction, (5, 7)), features)


def test_one_raster_scan_cannot_see_behind_itself():
    """A forward scan makes the top-left privileged: the first cell in reading
    order is reached by nothing. That is the defect the four directions fix,
    and asserting it is what keeps someone from quietly dropping three of
    them as an optimisation."""

    torch.manual_seed(0)
    block = SelectiveScan2D(8, d_state=4)
    # A live background, not zeros: the parameter projection has no bias, so a
    # zero input gives C = 0 and nothing is read out of the state at all - the
    # scan would look inert for reasons that have nothing to do with direction.
    quiet = torch.randn(1, block.d_inner, 5, 5)
    loud = quiet.clone()
    loud[0, :, 4, 4] += 5.0                    # an impulse in the LAST cell

    with torch.no_grad():
        forward_quiet = block._scan(quiet, "rows")
        forward_loud = block._scan(loud, "rows")
        reverse_loud = block._scan(loud, "rows_reverse")
        reverse_quiet = block._scan(quiet, "rows_reverse")

    corner = (slice(None), slice(None), 0, 0)
    # The forward scan reaches the first cell before the impulse exists.
    assert torch.allclose(forward_quiet[corner], forward_loud[corner], atol=1e-6)
    # The reverse scan starts at the impulse, so the same cell does see it.
    assert not torch.allclose(reverse_quiet[corner], reverse_loud[corner], atol=1e-6)


def test_the_stem_starts_as_its_convolutional_encoder():
    """The context gate is zero-initialised on purpose: an untrained scan
    injecting noise into the features correspondence is measured from costs
    early training, and hides whether the scan ever helped."""

    stem = VisionMambaStem(d_model=16, depth=1, patch_size=8, image_size=(64, 64),
                           context_grid=(4, 4))
    assert float(stem.context_gate.detach().abs().max()) == 0.0
    image = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        plain = stem.feature_norm(stem.patch_embed(image))
        assert torch.allclose(stem(image), plain, atol=1e-6)


def test_the_scan_contributes_once_its_gate_is_open():
    stem = VisionMambaStem(d_model=16, depth=1, patch_size=8, image_size=(64, 64),
                           context_grid=(4, 4))
    image = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        before = stem(image).clone()
        stem.context_gate.fill_(1.0)
        after = stem(image)
    assert not torch.allclose(before, after)


def test_the_stem_stays_translation_equivariant_at_initialisation():
    """Correlation needs features that move with the image. This is why the
    scan modulates a convolutional stem instead of replacing it."""

    stem = VisionMambaStem(d_model=8, depth=1, patch_size=8, image_size=(64, 96),
                           context_grid=(4, 6))
    torch.manual_seed(0)
    base = torch.rand(1, 1, 64, 96 + 16)
    first = base[:, :, :, 8 : 8 + 96]
    second = base[:, :, :, 16 : 16 + 96]          # shifted by exactly one cell
    with torch.no_grad():
        raw_first = stem.patch_embed(first)
        raw_second = stem.patch_embed(second)
        encoded_first = stem(first)
        encoded_second = stem(second)

    # The strided convolution is EXACTLY equivariant, which is the property
    # correlation depends on.
    assert torch.allclose(
        raw_first[:, :, :, 2:-1], raw_second[:, :, :, 1:-2], atol=1e-6
    )
    # GroupNorm's statistics are global over the frame, so a different crop
    # normalises slightly differently. Measured at 5.5e-3 relative - small
    # enough not to matter for correspondence, and worth pinning so that a
    # future change to a normaliser with a wider footprint is caught here
    # rather than as an unexplained accuracy loss.
    error = (encoded_first[:, :, :, 2:-1] - encoded_second[:, :, :, 1:-2]).abs().max()
    assert float(error / encoded_first.abs().max()) < 0.01


# --------------------------------------------------------------------------
# the frontend
# --------------------------------------------------------------------------


def _textured(height: int, width: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(1, 1, height // 8, width // 8, generator=generator)
    return torch.nn.functional.interpolate(
        coarse, size=(height, width), mode="bicubic", align_corners=False
    ).clamp(0, 1)


def _frontend(**overrides):
    settings = dict(
        visual_dim=32, d_model=32, depth=1, patch_size=8,
        image_size=(96, 128), context_grid=(6, 8), token_grid=4,
    )
    settings.update(overrides)
    return VisionMambaFlowFrontend(**settings)


def test_a_known_shift_is_measured_as_that_shift():
    """Motion is a property of a pair. This is the assertion that the encoder
    measures it rather than merely describing each frame."""

    torch.manual_seed(0)
    base = _textured(96, 128 + 32)
    first = base[:, :, :, 16 : 16 + 128]
    second = base[:, :, :, 0 : 128]        # content moved +16 px = +2 cells
    frontend = _frontend()
    with torch.no_grad():
        out = frontend(first, second, pair_dt_s=torch.ones(1, 1))
    flow = out["translational_flow_normalized_per_s"]
    # camera_matrix=None -> one cell is patch_size / image_width of normalized
    # bearing, so two cells is 2 * 8 / 128.
    truth = 2 * 8 / 128
    interior = float(flow[0, 0, 2:-2, 2:-2].mean())
    # The sign and the axis must be right, and the magnitude must be most of
    # the truth. It is not all of it: soft-argmax is biased towards zero by the
    # breadth of the correlation peak, which on an untrained stem recovers
    # about 60% at this temperature. That is why the temperature is learnable -
    # see test_the_soft_argmax_bias_shrinks_as_the_window_sharpens.
    assert 0.4 * truth < interior < 1.1 * truth
    assert abs(float(flow[0, 1, 2:-2, 2:-2].mean())) < 0.2 * truth


def _shift_recovery(refine_radius, temperatures=(0.1, 0.05, 0.02)):
    """Displacement recovered from a known two-cell shift, per temperature."""

    from vio.models.correlation import LocalCorrelation

    torch.manual_seed(0)
    base = _textured(96, 128 + 32)
    first = base[:, :, :, 16 : 16 + 128]
    second = base[:, :, :, 0 : 128]
    stem = VisionMambaStem(d_model=32, depth=1, patch_size=8,
                           image_size=(96, 128), context_grid=(6, 8))
    with torch.no_grad():
        features0, features1 = stem(first), stem(second)
        return [
            float(
                LocalCorrelation(
                    radius=4, temperature=t, refine_radius=refine_radius
                )(features0, features1)["local_flow"][0, 0, 2:-2, 2:-2].mean()
            )
            for t in temperatures
        ]


def test_the_single_stage_soft_argmax_is_attenuated_by_a_broad_peak():
    """The bias the two-stage argmax exists to remove, kept under test.

    A soft-argmax over every candidate under-reports displacement by as much as
    the correlation peak is broad, and because the factor tracks texture it is a
    scene-dependent multiplicative error rather than one a later layer absorbs.
    Sharpening the temperature only trades the bias against quantisation.
    """

    recovered = _shift_recovery(refine_radius=0)
    assert recovered[0] < recovered[1] < recovered[2] < 2.0
    assert recovered[0] < 0.5 * 2.0          # badly attenuated at 0.1
    assert recovered[2] > 0.6 * 2.0          # most of it back at 0.02


def test_the_two_stage_argmax_removes_the_attenuation_at_every_temperature():
    """Refining around the integer peak recovers the shift regardless of width.

    This is the point of the two stages: the far candidates that drag the
    estimate toward the window centre are never summed over, so the temperature
    stops being a bias knob and becomes sub-cell interpolation only.
    """

    recovered = _shift_recovery(refine_radius=1)
    for value in recovered:
        assert value == pytest.approx(2.0, abs=0.1)
    # And it must beat the single-stage path at the broadest peak, which is
    # where the old estimate was worst.
    assert recovered[0] > 3.0 * _shift_recovery(refine_radius=0)[0]


def test_the_correlation_temperature_is_trainable():
    frontend = _frontend()
    assert frontend.correlation.learnable_temperature
    names = {name for name, _ in frontend.named_parameters()}
    assert "correlation.log_temperature" in names


def test_rotation_is_removed_before_the_correlator_sees_it():
    """A seeded rotation map must leave a pure rotation measured as no
    translation. Without this the residual is turn-correlated - a false lateral
    signal, aimed at the axis that is already worst."""

    torch.manual_seed(0)
    base = _textured(96, 128 + 32)
    first = base[:, :, :, 16 : 16 + 128]
    second = base[:, :, :, 0 : 128]                     # +2 cells of image motion
    frontend = _frontend(rotation_mode="constant")
    rate = torch.tensor([[0.0, 0.4, 0.0]])              # a pure body-y rotation
    dt = torch.full((1, 1), 0.5)
    # Seed the map so that this rate over this interval predicts exactly the
    # two-cell displacement the pair contains: 2 cells / (0.4 * 0.5 rad).
    with torch.no_grad():
        frontend.rotation.map.weight.zero_()
        frontend.rotation.map.weight[0, 1] = 2.0 / (0.4 * 0.5)
        out = frontend(first, second, pair_dt_s=dt, body_rate_rad_s=rate)
    residual = out["translational_flow_normalized_per_s"][0, 0, 2:-2, 2:-2].mean()
    rotational = out["rotational_flow_normalized_per_s"][0, 0].mean()
    assert abs(float(residual)) < 0.02
    # And the rotation it removed is reported, not silently absorbed.
    assert float(rotational) == pytest.approx(2 * 8 / 128 / 0.5, abs=1e-4)


def test_the_search_centre_starts_as_a_no_op():
    for mode in ("field", "constant"):
        frontend = _frontend(rotation_mode=mode)
        assert float(frontend.rotation.map.weight.detach().abs().max()) == 0.0


def test_seeding_the_rotation_map_applies_both_rescalings():
    """The calibration artifact is native pixels per radian; the correlator
    works in feature cells of a resized image."""

    frontend = _frontend(rotation_mode="constant")
    jacobian = torch.tensor([[0.0, 900.0, 0.0], [900.0, 0.0, 0.0]])
    frontend.rotation.seed_from_jacobian(jacobian, patch_size=8, resize=(0.5, 0.5))
    weight = frontend.rotation.map.weight.detach()
    assert float(weight[0, 1]) == pytest.approx(900 * 0.5 / 8)
    assert float(weight[1, 0]) == pytest.approx(900 * 0.5 / 8)


def test_seeding_a_field_from_a_translation_jacobian_is_refused():
    """A 2x3 Jacobian records image TRANSLATION per body rate. It carries no
    information about the curl, so seeding a field from one would start the map
    at a value the artifact cannot describe."""

    frontend = _frontend(rotation_mode="field")
    with pytest.raises(ValueError, match="mode='constant'"):
        frontend.rotation.seed_from_jacobian(
            torch.zeros(2, 3), patch_size=8, resize=(0.5, 0.5)
        )


def test_a_constant_centre_removes_none_of_the_optical_axis_curl():
    """The structural claim behind the field.

    Rotation about the optical axis produces (+y*wz, -x*wz), which is ODD about
    the principal point. Its mean over a symmetric image is exactly zero, so the
    best constant approximation IS zero and a constant search centre removes
    none of it - however well the map is trained.
    """

    frontend = _frontend(rotation_mode="field")
    camera = torch.tensor([[200.0, 0.0, 64.0], [0.0, 200.0, 48.0], [0.0, 0.0, 1.0]])
    grid = frontend._normalised_grid(camera, (96, 128), 1, torch.device("cpu"), torch.float32)
    scale = frontend._bearing_scale(camera, (96, 128), 1, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        frontend.rotation.map.weight.copy_(torch.eye(3))
        curl = frontend.rotation(
            torch.tensor([[0.0, 0.0, 0.5]]), torch.full((1, 1), 0.1),
            grid=grid, cells_per_unit=scale.reciprocal(),
        )
    assert curl.shape == (1, 2) + frontend.stem.feature_size
    # It is not a small field ...
    assert float(curl.norm(dim=1).max()) > 0.05
    # ... but its mean is zero to floating point, which is the whole point.
    assert float(curl.mean(dim=(2, 3)).abs().max()) < 1e-6
    best_constant = curl.mean(dim=(2, 3)).reshape(1, 2, 1, 1)
    left_behind = (curl - best_constant).pow(2).mean().sqrt()
    assert float(left_behind / curl.pow(2).mean().sqrt()) > 0.999

    # A rate that IS mostly a translation is, by contrast, almost all removable.
    with torch.no_grad():
        pitch = frontend.rotation(
            torch.tensor([[0.0, 0.5, 0.0]]), torch.full((1, 1), 0.1),
            grid=grid, cells_per_unit=scale.reciprocal(),
        )
    removable = 1 - float(
        (pitch - pitch.mean(dim=(2, 3), keepdim=True)).pow(2).mean().sqrt()
        / pitch.pow(2).mean().sqrt()
    )
    assert removable > 0.9


def test_a_fractional_search_centre_does_not_attenuate_the_measured_flow():
    """Regression. A fractional centre used to resample one feature map
    bilinearly and correlate a sharp map against a blended one, which broadened
    the peak and cost up to half the displacement. The error was a function of
    the fractional part of a centre proportional to turn rate, so it appeared as
    a turn-correlated gain error on measured flow."""

    from vio.models.correlation import LocalCorrelation

    torch.manual_seed(3)
    patch, height, width = 8, 192, 256
    coarse = torch.randn(1, 1, height // 4, width // 4)
    image = torch.nn.functional.interpolate(
        coarse, size=(height, width), mode="bicubic", align_corners=False
    )
    shifted = torch.roll(image, shifts=(0, -2 * patch), dims=(2, 3))
    first = torch.nn.functional.pixel_unshuffle(image, patch)
    second = torch.nn.functional.pixel_unshuffle(shifted, patch)
    correlation = LocalCorrelation(radius=4, temperature=0.02).eval()
    cells = (height // patch, width // patch)

    for centre_dx in (0.0, -0.25, -0.5, -0.75, -1.25, -1.5, -1.75):
        centre = torch.stack(
            (torch.full(cells, centre_dx), torch.zeros(cells))
        ).unsqueeze(0)
        with torch.no_grad():
            measured = correlation(first, second, search_center=centre)
        valid = measured["flow_valid"].bool().squeeze(1)
        recovered = float(measured["flow"][0, 0][valid[0]].mean())
        assert recovered == pytest.approx(-2.0, abs=0.05), (
            f"centre {centre_dx} recovered {recovered}, not the true -2.0"
        )


def test_the_rounded_search_centre_still_passes_a_gradient():
    """Rounding is straight-through. A hard round would zero the gradient and
    the rotation map could never be learned at all."""

    from vio.models.correlation import LocalCorrelation

    torch.manual_seed(0)
    first = torch.randn(1, 8, 12, 16)
    second = torch.randn(1, 8, 12, 16)
    centre = torch.full((1, 2, 12, 16), 0.37, requires_grad=True)
    correlation = LocalCorrelation(radius=2, temperature=0.05)
    correlation(first, second, search_center=centre)["local_flow"].pow(2).mean().backward()
    assert centre.grad is not None
    assert float(centre.grad.norm()) > 0.0


def _reference_logits(feature_t, feature_tp1, correlation, search_center=None):
    """A from-scratch, unvectorized per-offset loop - the pre-optimisation
    computation, reimplemented here rather than imported, so this test cannot
    pass merely because the module still calls its own (possibly buggy)
    vectorized code."""

    import torch.nn.functional as F

    if correlation.center_features:
        feature_t = feature_t - feature_t.mean(dim=(2, 3), keepdim=True)
        feature_tp1 = feature_tp1 - feature_tp1.mean(dim=(2, 3), keepdim=True)
    left = F.normalize(feature_t, dim=1, eps=1e-6)
    right = F.normalize(feature_tp1, dim=1, eps=1e-6)
    batch, _, height, width = left.shape
    radius = correlation.radius

    center_map = None
    if search_center is not None:
        center = torch.as_tensor(search_center, dtype=left.dtype)
        if center.shape == (2,):
            center = center.view(1, 2, 1, 1).expand(batch, 2, height, width)
        elif center.shape == (batch, 2):
            center = center.view(batch, 2, 1, 1).expand(batch, 2, height, width)
        center_map = center.round()  # straight-through rounding, no grad needed here
        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=left.dtype), torch.arange(width, dtype=left.dtype),
            indexing="ij",
        )
        xs = xs.view(1, height, width).expand(batch, height, width)
        ys = ys.view(1, height, width).expand(batch, height, width)
    else:
        padded = F.pad(right, (radius, radius, radius, radius))

    correlations, valid_masks = [], []
    for dx, dy in [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]:
        if center_map is None:
            x0, y0 = radius + dx, radius + dy
            shifted = padded[:, :, y0 : y0 + height, x0 : x0 + width]
            valid = torch.zeros((batch, height, width), dtype=torch.bool)
            y_start, y_end = max(0, -dy), min(height, height - dy)
            x_start, x_end = max(0, -dx), min(width, width - dx)
            valid[:, y_start:y_end, x_start:x_end] = True
        else:
            target_x = xs + center_map[:, 0] + dx
            target_y = ys + center_map[:, 1] + dy
            valid = (
                (target_x >= 0) & (target_x <= width - 1)
                & (target_y >= 0) & (target_y <= height - 1)
            )
            grid_x = target_x.mul(2.0 / (width - 1)).sub(1.0)
            grid_y = target_y.mul(2.0 / (height - 1)).sub(1.0)
            grid = torch.stack((grid_x, grid_y), dim=-1)
            shifted = F.grid_sample(
                right, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )
        correlations.append((left * shifted).sum(dim=1))
        valid_masks.append(valid)

    temperature = (
        torch.exp(correlation.log_temperature)
        if correlation.learnable_temperature
        else correlation.temperature
    )
    logits = torch.stack(correlations, dim=1).div(temperature)
    valid = torch.stack(valid_masks, dim=1)
    return logits.masked_fill(~valid, torch.finfo(logits.dtype).min), valid


def test_the_vectorised_and_looped_correlation_paths_agree():
    """Regression pin for the correlation-loop vectorisation.

    ``LocalCorrelation.forward`` used to build its cost volume with a Python
    loop issuing one slice (or one ``grid_sample`` call) per candidate; it now
    builds every candidate at once via ``F.unfold``/a single batched
    ``grid_sample``. Both changes were verified against a hand-written
    reference before landing; this test keeps that reference in the suite so
    a future change to either path is caught the same way.
    """

    from vio.models.correlation import LocalCorrelation

    torch.manual_seed(4)
    for radius in (0, 2, 4):
        correlation = LocalCorrelation(radius=radius, temperature=0.07, learnable_temperature=True)
        first = torch.randn(2, 6, 9, 11)
        second = torch.randn(2, 6, 9, 11)

        # search_center=None: the integer-shift / unfold path.
        logits, valid = _reference_logits(first, second, correlation, search_center=None)
        out = correlation(first, second, search_center=None)
        assert torch.allclose(out["logits"], logits, atol=1e-5)
        assert torch.equal(out["valid"], valid)

        # A fractional, per-cell centre: the grid-sample path.
        centre = torch.randn(2, 2, 9, 11) * 1.7
        logits, valid = _reference_logits(first, second, correlation, search_center=centre)
        out = correlation(first, second, search_center=centre)
        assert torch.allclose(out["logits"], logits, atol=1e-5)
        assert torch.equal(out["valid"], valid)


# --------------------------------------------------------------------------
# the scale relationship
# --------------------------------------------------------------------------


def _aiding(batch: int, steps: int, altitude_m: float) -> torch.Tensor:
    aiding = torch.zeros(batch, steps, AIDING_INPUT_DIM)
    aiding[..., 1] = 1.0                       # cos(roll) = 1
    aiding[..., 3] = 1.0                       # cos(pitch) = 1
    aiding[..., 4] = math.log(altitude_m)
    aiding[..., -1] = 0.05                     # delta_time_s, always the last channel
    return aiding


def _times(steps: int) -> torch.Tensor:
    """A plausible 100 Hz telemetry clock, for tests that just need SOME
    times_s to pass to visual_age_seconds and do not care about its value."""

    return torch.arange(steps, dtype=torch.float64) * 0.01


def _log_altitude(aiding: torch.Tensor) -> torch.Tensor:
    """The RAW log altitude ``forward`` requires explicitly.

    ``_aiding`` above is a hand-built vector, not the real dataset pipeline's
    aiding tensor, so its index 4 already holds the raw value rather than the
    centred one ``vio.data.fixedwing_vo`` puts there - this helper only saves
    repeating the index at every call site.
    """

    return aiding[..., 4]


def test_altitude_multiplies_the_predicted_speed():
    """The whole reason altitude is an input: v = h * u. A concatenated
    altitude channel would have to LEARN this product; adding it in log space
    makes it exact."""

    model = VisionMambaVO(visual_dim=16)
    token = torch.zeros(1, 8, 16)
    present = torch.zeros(1, 8, 1)
    age, _ = visual_age_seconds(present, _times(8), 0.35)
    low_aiding, high_aiding = _aiding(1, 8, 100.0), _aiding(1, 8, 200.0)
    with torch.no_grad():
        low = model(
            low_aiding, token, present, age, log_altitude=_log_altitude(low_aiding)
        )["predicted_speed"]
        high = model(
            high_aiding, token, present, age, log_altitude=_log_altitude(high_aiding)
        )["predicted_speed"]
    assert torch.allclose(high, 2.0 * low, rtol=1e-5)


def test_the_untrained_model_predicts_plausible_forward_flight():
    """Zero-initialised heads with a forward bias and a log bearing rate, so an
    untrained model sits at the physically correct relationship instead of at a
    zero vector that cannot be normalised."""

    model = VisionMambaVO(visual_dim=16)
    aiding = _aiding(1, 4, 100.0)
    out = model(
        aiding, torch.zeros(1, 4, 16), torch.zeros(1, 4, 1), torch.zeros(1, 4, 1),
        log_altitude=_log_altitude(aiding),
    )
    assert float(out["predicted_speed"][0, 0]) == pytest.approx(25.0, abs=0.1)
    assert torch.allclose(
        out["predicted_direction"][0, 0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6
    )


def test_forward_refuses_to_guess_the_raw_altitude():
    """``aiding[..., LOG_ALTITUDE_INDEX]`` is the CENTRED copy the encoder
    sees, not the raw one ``v = h * u`` needs. Silently falling back to it
    would scale every predicted speed by a constant nobody would think to look
    for - so ``forward`` must refuse rather than guess."""

    model = VisionMambaVO(visual_dim=16)
    aiding = _aiding(1, 4, 100.0)
    with pytest.raises(ValueError, match="log_altitude must be supplied"):
        model(aiding, torch.zeros(1, 4, 16), torch.zeros(1, 4, 1), torch.zeros(1, 4, 1))


def test_direction_is_a_unit_vector_and_independent_of_altitude():
    """Direction and speed have different observability - direction survives a
    scale error, and factoring them is what keeps a wrong altitude from
    corrupting the crab angle."""

    torch.manual_seed(0)
    model = VisionMambaVO(visual_dim=16)
    torch.nn.init.normal_(model.direction_head.weight, std=0.5)
    token = torch.randn(2, 6, 16)
    present = torch.ones(2, 6, 1)
    age, _ = visual_age_seconds(present, _times(6), 0.35)
    aiding = _aiding(2, 6, 80.0)
    low = model(aiding, token, present, age, log_altitude=_log_altitude(aiding))
    assert torch.allclose(
        low["predicted_direction"].norm(dim=-1), torch.ones(2, 6), atol=1e-5
    )
    speed = low["predicted_speed"].unsqueeze(-1)
    assert torch.allclose(low["predicted_velocity"], low["predicted_direction"] * speed)


def test_visual_absence_is_distinguishable_from_zero_motion():
    model = VisionMambaVO(visual_dim=16)
    aiding = _aiding(1, 6, 100.0)
    zeros = torch.zeros(1, 6, 16)
    age = torch.zeros(1, 6, 1)
    absent = model.fuse(aiding, zeros, torch.zeros(1, 6, 1), age)
    present = model.fuse(aiding, zeros, torch.ones(1, 6, 1), age)
    assert not torch.allclose(absent, present)


def test_uncertainty_is_bounded_so_nll_cannot_collapse_it():
    model = VisionMambaVO(visual_dim=16, min_log_variance=-4.0, max_log_variance=2.0)
    torch.nn.init.normal_(model.log_variance_head.weight, std=5.0)
    aiding = _aiding(2, 5, 100.0)
    out = model(
        aiding, torch.randn(2, 5, 16), torch.ones(2, 5, 1), torch.zeros(2, 5, 1),
        log_altitude=_log_altitude(aiding),
    )
    log_variance = out["velocity_log_variance"]
    assert float(log_variance.min()) >= -4.0
    assert float(log_variance.max()) <= 2.0


def test_the_model_is_causal_in_time():
    """A later tick must not change an earlier prediction, or every validation
    number is optimistic by an amount nobody can bound."""

    torch.manual_seed(0)
    model = VisionMambaVO(visual_dim=16).eval()
    aiding = _aiding(1, 10, 100.0)
    aiding[..., 0] = torch.randn(1, 10)
    token = torch.randn(1, 10, 16)
    present = (torch.rand(1, 10, 1) > 0.5).float()
    times = _times(10)
    age, _ = visual_age_seconds(present, times, 0.35)
    log_altitude = _log_altitude(aiding)
    with torch.no_grad():
        full = model(aiding, token, present, age, log_altitude=log_altitude)["predicted_velocity"]
        truncated_age, _ = visual_age_seconds(present[:, :6], times[:6], 0.35)
        truncated = model(
            aiding[:, :6], token[:, :6], present[:, :6], truncated_age,
            log_altitude=log_altitude[:, :6],
        )["predicted_velocity"]
    assert torch.allclose(full[:, :6], truncated, atol=1e-5)


# --------------------------------------------------------------------------
# TBPTT primitives: per-lane state reset and the truncation cut
# --------------------------------------------------------------------------


def test_mask_stream_state_resets_only_the_flagged_lanes():
    """A block's initial_state is all zeros, so 'reset lane k' and 'zero out
    lane k's slice of every state tensor' are the same operation - this pins
    that mask_stream_state actually does that, leaving the other lanes'
    slices bit-for-bit untouched rather than merely 'close to' unchanged."""

    torch.manual_seed(0)
    model = VisionMambaVO(visual_dim=4, aiding_dim=8, fusion_dim=8)
    batch = 3
    aiding = torch.randn(batch, 5, AIDING_INPUT_DIM)
    token = torch.randn(batch, 5, 4)
    present = torch.ones(batch, 5, 1)
    age = torch.zeros(batch, 5, 1)
    log_altitude = torch.zeros(batch, 5)
    with torch.no_grad():
        _, state = model.forward_stream(
            aiding, token, present, age, log_altitude=log_altitude, state=None
        )
        keep = torch.tensor([True, False, True])
        masked = mask_stream_state(state, keep)

    for before_blocks, after_blocks in ((state.aiding, masked.aiding), (state.fusion, masked.fusion)):
        for before, after in zip(before_blocks, after_blocks):
            for original, updated in zip(before, after):
                assert torch.equal(original[0], updated[0])   # lane 0: kept
                assert torch.equal(original[2], updated[2])   # lane 2: kept
                assert torch.all(updated[1] == 0.0)            # lane 1: reset
                # Reset is only vacuous if the original was already zero -
                # confirm there was something real to reset.
                assert not torch.all(original[1] == 0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_mask_stream_state_moves_the_mask_to_the_state_devices_own_device():
    """Regression: keep.to(conv.dtype) alone changed dtype but left the mask
    on keep's OWN device. A sampler builds keep fresh on the CPU
    (ChronologicalWindowSampler.continues_at has no device to infer), so the
    very first CUDA training step would hit a cross-device multiply -
    conv * conv_keep - and crash. keep is deliberately left on the CPU here,
    the exact shape of the original bug, and the state is on CUDA."""

    model = VisionMambaVO(visual_dim=4, aiding_dim=8, fusion_dim=8).cuda()
    batch = 3
    aiding = torch.randn(batch, 5, AIDING_INPUT_DIM, device="cuda")
    token = torch.randn(batch, 5, 4, device="cuda")
    present = torch.ones(batch, 5, 1, device="cuda")
    age = torch.zeros(batch, 5, 1, device="cuda")
    log_altitude = torch.zeros(batch, 5, device="cuda")
    with torch.no_grad():
        _, state = model.forward_stream(
            aiding, token, present, age, log_altitude=log_altitude, state=None
        )
        keep = torch.tensor([True, False, True])  # CPU, on purpose
        masked = mask_stream_state(state, keep)

    for block in (masked.aiding + masked.fusion):
        for tensor in block:
            assert tensor.device.type == "cuda"
    assert torch.all(masked.aiding[0][0][1] == 0.0)


def test_detach_stream_state_cuts_the_autograd_graph():
    """Without this, a TBPTT chunk's backward would keep walking into every
    earlier chunk's graph - unbounded cost that grows with stream length,
    exactly what truncation exists to avoid. Checked structurally (grad_fn is
    None after detaching) rather than by comparing gradients, which would
    only show the cut happened to matter for this particular loss."""

    model = VisionMambaVO(visual_dim=4, aiding_dim=8, fusion_dim=8)
    batch = 2
    aiding = torch.randn(batch, 3, AIDING_INPUT_DIM)
    token = torch.randn(batch, 3, 4)
    present = torch.ones(batch, 3, 1)
    age = torch.zeros(batch, 3, 1)
    log_altitude = torch.zeros(batch, 3)
    _, state = model.forward_stream(
        aiding, token, present, age, log_altitude=log_altitude, state=None
    )
    leaves = [t for block in (state.aiding + state.fusion) for t in block]
    # There must be something to cut, or the test below is vacuous.
    assert any(t.grad_fn is not None for t in leaves)

    detached = detach_stream_state(state)
    detached_leaves = [t for block in (detached.aiding + detached.fusion) for t in block]
    assert all(t.grad_fn is None and not t.requires_grad for t in detached_leaves)
    # Values are untouched - detach only cuts the graph, never the numbers a
    # forward pass would actually see.
    for before, after in zip(leaves, detached_leaves):
        assert torch.equal(before, after)


# --------------------------------------------------------------------------
# attitude and altitude
# --------------------------------------------------------------------------


def test_gpsnaveul_is_refused_because_it_is_half_the_label():
    """The body-velocity target is GPSNavVn rotated by GPSNavEul. Reading it as
    an input is target leakage, and a silent fallback is how that happens by
    accident rather than on purpose."""

    headers = ("Time", "GPSNavEulX", "GPSNavEulY", "GPSNavEulZ", "relativeAlt")
    with pytest.raises(ValueError, match="ingredient of the training target"):
        resolve_attitude_columns(headers)
    names, note = resolve_attitude_columns(headers, allow_reference=True)
    assert names == ("GPSNavEulX", "GPSNavEulY", "GPSNavEulZ")
    assert "leaky" in note


def test_the_navigation_filter_attitude_is_preferred_when_present():
    headers = ("Time", "NavEulX", "NavEulY", "NavEulZ", "EulX", "EulY", "EulZ")
    names, _ = resolve_attitude_columns(headers)
    assert names == ("NavEulX", "NavEulY", "NavEulZ")


def test_related_alt_logger_spelling_is_supported_without_using_imu():
    headers = ("Time", "NavEulX", "NavEulY", "NavEulZ", "RelatedAlt")
    assert resolve_altitude_column(headers) == "RelatedAlt"


def test_euler_unit_is_decided_by_range():
    assert detect_euler_unit(np.array([[0.1, -0.2, 3.0]])) == "radians"
    assert detect_euler_unit(np.array([[0.1, -0.2, 147.0]])) == "degrees"


def test_hold_fraction_finds_a_column_held_between_slower_updates():
    """A GPS-derived column logged at 100 Hz repeats itself. Differencing it
    gives zero on most pairs and a jump on the rest - and the jumps land on
    turns, which is precisely where de-rotation matters."""

    moving = np.arange(100, dtype=np.float64).reshape(-1, 1)
    assert hold_fraction(moving) == 0.0
    held = np.repeat(np.arange(10, dtype=np.float64), 10).reshape(-1, 1)
    assert hold_fraction(held) == pytest.approx(0.909, abs=0.01)


def test_body_rates_recover_a_constant_yaw_rate():
    """Log(R0^T R1)/dt, not a differentiated Euler angle."""

    dt = 0.01
    times = np.arange(0, 2.0, dt)
    yaw = 0.3 * times
    euler = np.stack((np.zeros_like(yaw), np.zeros_like(yaw), yaw), axis=1)
    quaternion = euler_zyx_to_quaternion_np(euler)
    rates = body_rates_from_quaternions(quaternion, times)
    assert float(np.median(rates[1:, 2])) == pytest.approx(0.3, abs=1e-3)
    assert float(np.abs(rates[1:, :2]).max()) < 1e-6
    assert float(np.abs(rates[0]).max()) == 0.0


def test_body_rates_survive_a_heading_wrap():
    """A differentiated Euler angle spikes by 2*pi/dt here; a relative rotation
    does not notice."""

    dt = 0.01
    yaw = np.array([math.pi - 0.01, math.pi + 0.01 - 2 * math.pi])
    euler = np.stack((np.zeros(2), np.zeros(2), yaw), axis=1)
    quaternion = euler_zyx_to_quaternion_np(euler)
    rates = body_rates_from_quaternions(quaternion, np.array([0.0, dt]))
    assert abs(float(rates[1, 2])) == pytest.approx(2.0, abs=0.05)


def test_aiding_features_avoid_the_wrap_and_take_altitude_in_log_space():
    from vio.data.attitude import AttitudeAltitude

    euler = np.array([[0.1, -0.2, 3.0], [0.2, 0.1, -3.0]])
    source = AttitudeAltitude(
        times_s=np.array([0.0, 0.1]),
        euler_rad=euler,
        quaternion=euler_zyx_to_quaternion_np(euler),
        body_rate_rad_s=np.zeros((2, 3)),
        altitude_m=np.array([100.0, 200.0]),
        attitude_columns=("NavEulX", "NavEulY", "NavEulZ"),
        altitude_column="relativeAlt",
        euler_unit="radians",
        attitude_hold_fraction=0.0,
        notes=(),
    )
    features = aiding_features(source, altitude_floor_m=1.0)
    assert features.shape == (2, 8)
    # sin/cos pairs, so a level attitude is a smooth point, not a boundary.
    assert features[0, 0] == pytest.approx(math.sin(0.1), abs=1e-6)
    assert features[0, 1] == pytest.approx(math.cos(0.1), abs=1e-6)
    # Yaw never reaches the model: body-frame velocity is yaw-invariant.
    assert features.shape[1] == 8


def test_aiding_features_carries_the_body_rates_through_unchanged():
    """Codex Increment 2: the aiding vector gains p/q/r so a coordinated turn -
    roll and pitch near constant, yaw rate carrying the whole manoeuvre - is
    visible to the encoder at all. This is source.body_rate_rad_s verbatim,
    not a fresh differentiation, so it must equal the ALREADY-DERIVED rate
    exactly rather than merely be plausible."""

    from vio.data.attitude import AttitudeAltitude

    euler = np.zeros((3, 3))
    rates = np.array([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [0.05, 0.0, -0.4]])
    source = AttitudeAltitude(
        times_s=np.array([0.0, 0.1, 0.2]),
        euler_rad=euler,
        quaternion=euler_zyx_to_quaternion_np(euler),
        body_rate_rad_s=rates,
        altitude_m=np.array([100.0, 100.0, 100.0]),
        attitude_columns=("NavEulX", "NavEulY", "NavEulZ"),
        altitude_column="relativeAlt",
        euler_unit="radians",
        attitude_hold_fraction=0.0,
        notes=(),
    )
    features = aiding_features(source)
    np.testing.assert_allclose(features[:, 5:8], rates, atol=1e-6)


def test_altitude_keeps_its_recorded_magnitude():
    """Re-basing altitude - subtracting its own minimum to condition the log -
    turns a 120 m flight varying by 12 m into a 12 m one, and every predicted
    speed comes out an order of magnitude low with nothing in the loss to say
    why. Conditioning belongs in the encoder's centred copy, not in the copy
    the speed head multiplies by."""

    from vio.data.attitude import AttitudeAltitude

    euler = np.zeros((2, 3))
    source = AttitudeAltitude(
        times_s=np.array([0.0, 0.1]),
        euler_rad=euler,
        quaternion=euler_zyx_to_quaternion_np(euler),
        body_rate_rad_s=np.zeros((2, 3)),
        altitude_m=np.array([120.0, 240.0]),
        attitude_columns=("NavEulX", "NavEulY", "NavEulZ"),
        altitude_column="relativeAlt",
        euler_unit="radians",
        attitude_hold_fraction=0.0,
        notes=(),
    )
    features = aiding_features(source)
    assert features[0, 4] == pytest.approx(math.log(120.0), abs=1e-5)
    # A doubling of altitude is log(2) whatever the datum, which is the whole
    # point of taking the logarithm.
    assert features[1, 4] - features[0, 4] == pytest.approx(math.log(2.0), abs=1e-5)


def _write_flight(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def test_loading_reports_a_held_attitude_column_instead_of_staying_quiet(tmp_path):
    columns = ["Time", "NavEulX", "NavEulY", "NavEulZ", "relativeAlt"]
    rows = []
    for index in range(200):
        yaw = 0.01 * (index // 10)          # updated once every ten rows
        rows.append([index * 0.01, 0.0, 0.0, yaw, 100.0 + 0.01 * index])
    csv_path = tmp_path / "flight.csv"
    _write_flight(csv_path, rows, columns)

    source = load_attitude_altitude(csv_path)
    assert source.attitude_columns == ("NavEulX", "NavEulY", "NavEulZ")
    assert source.altitude_column == "relativeAlt"
    assert source.attitude_hold_fraction > 0.8
    assert any("WARNING" in note for note in source.notes)


def test_a_constant_attitude_column_is_rejected(tmp_path):
    """Present but constant carries no attitude, and a model fed it de-rotates
    with nothing while reporting no error at all."""

    columns = ["Time", "NavEulX", "NavEulY", "NavEulZ", "relativeAlt"]
    rows = [[index * 0.01, 0.0, 0.0, 0.0, 100.0] for index in range(50)]
    csv_path = tmp_path / "flight.csv"
    _write_flight(csv_path, rows, columns)
    with pytest.raises(ValueError, match="constant over the whole flight"):
        load_attitude_altitude(csv_path)


def test_degrees_are_converted_on_load(tmp_path):
    columns = ["Time", "EulX", "EulY", "EulZ", "Barometer"]
    rows = [[index * 0.01, 0.0, 0.0, 147.0 + index, 100.0] for index in range(50)]
    csv_path = tmp_path / "flight.csv"
    _write_flight(csv_path, rows, columns)
    source = load_attitude_altitude(csv_path)
    assert source.euler_unit == "degrees"
    assert source.altitude_column == "Barometer"
    assert float(np.max(np.abs(source.euler_rad))) < 2 * math.pi + 1e-6


# --------------------------------------------------------------------------
# windowing
# --------------------------------------------------------------------------


def test_tokens_land_at_the_tick_they_became_available():
    from vio.data.fixedwing_vo import scatter_visual_tokens

    tokens = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    quality = torch.ones(2, 3, 1)
    offsets = torch.tensor([[0, 5, 9], [2, 7, 99]])       # 99 falls outside
    valid = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    field, _, present = scatter_visual_tokens(
        tokens, quality, offsets, valid, window_length=10, visual_dim=4
    )
    assert present[0, :, 0].tolist() == [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    # An invalid slot and an out-of-window slot both leave the tick empty
    # rather than being clamped onto the boundary, which would stack several
    # events on one tick and silently change the schedule.
    assert present[1, :, 0].tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert torch.allclose(field[0, 5], tokens[0, 1])


def test_visual_age_is_the_deployment_latency_on_arrival_not_zero():
    """present fires at READY_TICK, one deployment_latency_s AFTER the second
    image was actually captured (see VisualPairSource) - so the image is
    already ~0.35 s old the moment its token becomes available, and reporting
    age 0 there (the bug this replaces) claims it is brand new."""

    present = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]).unsqueeze(-1)
    times = torch.arange(5, dtype=torch.float64) * 0.05  # 20 Hz clock
    age, _ = visual_age_seconds(present, times, deployment_latency_s=0.35)
    assert age.shape == present.shape
    # Tick 2 (t=0.10s): a token just arrived - age is the LATENCY, not zero.
    assert age[0, 2, 0].item() == pytest.approx(0.35, abs=1e-6)
    # Two ticks later (t=0.20s, 0.10s after arrival): latency plus elapsed.
    assert age[0, 4, 0].item() == pytest.approx(0.45, abs=1e-6)


def test_visual_age_is_zero_before_any_image_and_stays_zero_when_disabled():
    """Before the first delivery this call can see, there is no captured
    image to be stale - age is a flat 0, not a value growing from nothing.
    An all-absent present (visual-disabled evaluation) is the same case
    forever: there was never an image to begin with."""

    present = torch.zeros(1, 5, 1)
    times = torch.arange(5, dtype=torch.float64) * 0.05
    age, carry = visual_age_seconds(present, times, deployment_latency_s=0.35)
    assert age[0, :, 0].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert carry.item() == float("-inf")


def test_visual_age_uses_the_real_clock_not_a_uniform_tick_count():
    """A dropped telemetry sample makes the clock irregular; age must track
    REAL elapsed seconds between the ticks it actually has, not assume a
    constant interval - the replaced tick-counting version could not have
    told this apart from a steady clock at all."""

    present = torch.tensor([[1.0, 0.0, 0.0]]).unsqueeze(-1)
    # 0.01 s to the second tick, then a 0.20 s gap to the third - a dropped
    # sample, not a steady 100 Hz clock.
    times = torch.tensor([0.0, 0.01, 0.21], dtype=torch.float64)
    age, _ = visual_age_seconds(present, times, deployment_latency_s=0.0)
    assert age[0, :, 0].tolist() == pytest.approx([0.0, 0.01, 0.21])


def test_visual_age_carries_across_two_calls_the_same_as_one_concatenated_call():
    """TBPTT processes one window per call; carry is what keeps a continuing
    lane's staleness growing from where the previous chunk left it instead of
    resetting to 0 at every window boundary - checked against the ground
    truth of computing the whole span in a single call."""

    times = torch.arange(8, dtype=torch.float64) * 0.05
    present = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).unsqueeze(-1)
    whole, _ = visual_age_seconds(present, times, deployment_latency_s=0.35)

    first, carry = visual_age_seconds(
        present[:, :4], times[:4], deployment_latency_s=0.35
    )
    second, _ = visual_age_seconds(
        present[:, 4:], times[4:], deployment_latency_s=0.35, carry=carry
    )
    chunked = torch.cat((first, second), dim=1)
    torch.testing.assert_close(chunked, whole)


def test_a_reset_forgets_the_carried_age_the_same_way_it_forgets_state():
    """A genuine reset (carry=None) must NOT see the previous chunk's
    delivery, mirroring how a state reset forgets recurrent history - this is
    what distinguishes a real TBPTT continuation from a fresh leg that
    happens to be scored right after another one."""

    times = torch.arange(4, dtype=torch.float64) * 0.05
    present = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).unsqueeze(-1)
    _, carry = visual_age_seconds(present, times, deployment_latency_s=0.35)

    reset_age, _ = visual_age_seconds(
        torch.zeros(1, 3, 1), times[:3], deployment_latency_s=0.35, carry=None
    )
    assert reset_age[0, :, 0].tolist() == [0.0, 0.0, 0.0]

    carried_age, _ = visual_age_seconds(
        torch.zeros(1, 3, 1), times[:3], deployment_latency_s=0.35, carry=carry
    )
    assert float(carried_age[0, 0, 0]) > 0.0


def test_mask_age_carry_resets_only_the_flagged_lanes():
    carry = torch.tensor([1.5, 2.5, 3.5])
    keep = torch.tensor([True, False, True])
    masked = mask_age_carry(carry, keep)
    assert masked[0].item() == pytest.approx(1.5)
    assert masked[2].item() == pytest.approx(3.5)
    assert masked[1].item() == float("-inf")


def test_the_encoder_sees_a_centred_altitude_and_the_head_sees_the_real_one(tmp_path):
    """Two copies on purpose. Centring conditions the encoder input; the speed
    head needs the true magnitude, because v = h * u is a product with a real
    h in it and not a relative one."""

    from vio.data.fixedwing_vo import VO_AIDING_CHANNELS

    columns = ["Time", "NavEulX", "NavEulY", "NavEulZ", "relativeAlt",
               "GPSNavVnX", "GPSNavVnY", "GPSNavVnZ",
               "GPSNavEulX", "GPSNavEulY", "GPSNavEulZ"]
    rows = []
    for index in range(600):
        yaw = 0.001 * index
        rows.append([index * 0.01, 0.0, 0.0, yaw, 120.0 + 0.01 * index,
                     25.0, 0.0, 0.0, 0.0, 0.0, yaw])
    csv_path = tmp_path / "flight.csv"
    _write_flight(csv_path, rows, columns)

    from vio.data.attitude import load_attitude_altitude, aiding_features
    from vio.data.fixedwing_vo import VONormalizer

    source = load_attitude_altitude(csv_path)
    normalizer = VONormalizer.from_range(source, (0, 600))
    raw = aiding_features(source)[:, 4]
    assert float(raw.mean()) == pytest.approx(normalizer.log_altitude_mean, abs=1e-5)
    # The raw channel carries the true altitude, ~120 m, not a re-based one.
    assert float(np.exp(raw.mean())) == pytest.approx(123.0, rel=0.05)
    assert VO_AIDING_CHANNELS[4] == "log_altitude_centred"


def test_pooling_is_confidence_weighted_and_marks_unmeasurable_cells():
    """A cell that could not measure must not look like a measured zero.

    Plain averaging mixes a confident measurement with a cell whose correlation
    was ambiguous, which scales the pooled flow by the textured fraction and so
    scales SPEED by it. The weight channel is what keeps the two apart: the
    flow of an unmeasurable cell is zeroed and its weight goes to zero with it,
    rather than a zero flow being reported at full weight.
    """

    torch.manual_seed(0)
    frontend = _frontend()
    height, width = frontend.image_size
    # Left half textured, right half a flat constant: the right half carries the
    # same true motion but nothing a correlator can lock onto.
    first = _textured(height, width)
    first[:, :, :, width // 2 :] = 0.5
    second = torch.roll(first, shifts=2 * frontend.patch_size, dims=3)
    second[:, :, :, width // 2 :] = 0.5

    with torch.no_grad():
        out = frontend(first, second, pair_dt_s=torch.full((1,), 0.05))

    pooled = out["pooled_cells"]
    assert pooled.shape[1] == frontend.cell_channels == 9
    flow_x, weight = pooled[0, 0], pooled[0, 2]

    grid = frontend.token_grid
    left, right = slice(0, grid // 2), slice(grid // 2 + 1, grid)
    # The textured side must be measured, the flat side must not be.
    assert weight[:, left].mean() > 5.0 * weight[:, right].mean()
    # Where nothing was measurable the flow is exactly zero, not a small
    # non-zero average of noise that would pass for a real slow motion.
    # Unconditional: the flat side MUST come out unmeasured. Guarding this with
    # "if unmeasured.any()" would let the case that matters pass vacuously.
    assert torch.all(weight[:, right] <= frontend.min_pool_weight)
    assert torch.all(flow_x[:, right] == 0.0)


def test_boundary_hit_fraction_ignores_cells_without_a_complete_search_window():
    """A border cell's search window is CLIPPED by the image edge, which makes
    its argmax structurally more likely to land on whatever remains of the
    window regardless of where the true match is - not a genuine signal that
    the radius is too small. Two cells hit the boundary: one has a complete
    window and usable signal (the real symptom), the other does not
    (full_window_valid=False) and must be excluded from both the numerator
    AND the denominator, not merely diluted into it."""

    correlation = {
        "peak_at_boundary": torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
        "full_window_valid": torch.tensor([[[[True, False], [True, True]]]]),
        "usable_confidence": torch.tensor([[[[0.5, 0.9], [0.5, 0.5]]]]),
        "entropy": torch.zeros(1, 1, 2, 2),
        "candidate_count": torch.full((1, 1, 2, 2), 9.0),
    }
    weight = torch.zeros(1, 1, 2, 2)
    diagnostics = frontend_diagnostics(correlation, weight, min_pool_weight=1e-4)
    # 3 cells are usable with a complete window; 1 of those 3 hit the
    # boundary. If the clipped cell (also a boundary hit) were counted, the
    # fraction would come out 2/4 = 0.5 instead of the correct 1/3.
    assert float(diagnostics["boundary_hit_fraction"]) == pytest.approx(1.0 / 3.0)


def test_boundary_hit_fraction_ignores_textureless_cells():
    """A textureless cell's argmax is an arbitrary tie-break among candidates
    that are all equally (un)likely, not a measurement - usable_confidence <=
    0 marks exactly that. Mixing it in would let a sky- or flat-ground-heavy
    scene inflate the reported boundary-hit rate for reasons that have
    nothing to do with the search radius."""

    correlation = {
        "peak_at_boundary": torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]]),
        "full_window_valid": torch.ones(1, 1, 2, 2, dtype=torch.bool),
        "usable_confidence": torch.tensor([[[[0.5, 0.0], [0.5, 0.5]]]]),
        "entropy": torch.zeros(1, 1, 2, 2),
        "candidate_count": torch.full((1, 1, 2, 2), 9.0),
    }
    weight = torch.zeros(1, 1, 2, 2)
    diagnostics = frontend_diagnostics(correlation, weight, min_pool_weight=1e-4)
    # The textureless cell (usable_confidence == 0, top-right) also hit the
    # boundary, but is excluded from both numerator and denominator: 2 of
    # the remaining 3 usable cells hit -> 2/3, not 3/4.
    assert float(diagnostics["boundary_hit_fraction"]) == pytest.approx(2.0 / 3.0)


def test_entropy_normalized_divides_by_each_cells_own_candidate_ceiling():
    """log(candidate_count) is the MAXIMUM possible entropy for that many
    candidates (a uniform distribution) - dividing by the GLOBAL log(K)
    instead would understate a border cell's true uncertainty, since its
    clipped window already has fewer candidates to be uncertain over. Both
    cells here sit at their OWN ceiling, so both must normalize to 1.0."""

    correlation = {
        "peak_at_boundary": torch.zeros(1, 1, 1, 2, dtype=torch.bool),
        "full_window_valid": torch.ones(1, 1, 1, 2, dtype=torch.bool),
        "usable_confidence": torch.full((1, 1, 1, 2), 0.5),
        # Interior cell: 9 candidates, at its own maximum entropy log(9).
        # Border cell: 4 candidates (window clipped), at ITS maximum log(4).
        "entropy": torch.tensor([[[[math.log(9.0), math.log(4.0)]]]]),
        "candidate_count": torch.tensor([[[[9.0, 4.0]]]]),
    }
    weight = torch.zeros(1, 1, 1, 2)
    diagnostics = frontend_diagnostics(correlation, weight, min_pool_weight=1e-4)
    # Raw entropy alone would report 2.197 and 1.386, making the border cell
    # look far less uncertain than it actually is at its own ceiling.
    assert float(diagnostics["mean_entropy_normalized"]) == pytest.approx(1.0, abs=1e-5)


def test_a_wholly_ambiguous_region_reports_no_measurement_at_all():
    """Total ambiguity must give zero weight AND zero flow, not a small weight.

    A uniform distribution over n candidates peaks at 1/n, so raw confidence
    never reaches zero. Weighting by that floor does not suppress anything: over
    a block where every cell is ambiguous the normalisation cancels the weight
    out, mean(bad*w)/mean(w) = bad, and the block reports the argmax tie-break
    at full size. Measured before the fix: flow (-3, -3) at confidence 1/81,
    which the occupancy threshold of 1e-4 happily called measured.
    """

    from vio.models.correlation import LocalCorrelation

    # Constant features: every candidate scores identically, so there is no
    # information anywhere in the correlation volume.
    flat = torch.ones(1, 16, 24, 24)
    with torch.no_grad():
        out = LocalCorrelation(radius=4, temperature=0.03)(flat, flat.clone())

    assert float(out["confidence"].mean()) == pytest.approx(1.0 / 81.0, rel=0.35)
    assert torch.all(out["usable_confidence"] == 0.0)

    weight = out["usable_confidence"] * out["flow_valid"].to(flat.dtype)
    pooled_weight = F.adaptive_avg_pool2d(weight, (6, 6))
    pooled_flow = F.adaptive_avg_pool2d(out["local_flow"] * weight, (6, 6)) / (
        pooled_weight.clamp_min(1e-4)
    )
    pooled_flow = pooled_flow * (pooled_weight > 1e-4).to(flat.dtype)
    assert torch.all(pooled_weight == 0.0)
    assert torch.all(pooled_flow == 0.0)


def test_a_peak_at_the_search_limit_is_not_pulled_inward():
    """The refinement window truncates at the edge instead of sliding inward.

    A winner at +4 has only {+3, +4} available. Sliding the window centre to
    make three members fit would refine it over {+2, +3, +4} and bias the
    estimate inward exactly where displacement is already saturating.
    """

    from vio.models.correlation import LocalCorrelation

    torch.manual_seed(0)
    # A shift of exactly the search radius, so the true peak sits on the limit.
    base = torch.randn(1, 24, 32, 48)
    first = base[:, :, :, 4:44]
    second = base[:, :, :, 0:40]

    for temperature in (0.02, 0.1, 0.3, 0.6):
        with torch.no_grad():
            flow = LocalCorrelation(
                radius=4, temperature=temperature, refine_radius=1
            )(first, second)["local_flow"]
        recovered = float(flow[0, 0, 6:-6, 6:-6].mean())
        assert recovered <= 4.0 + 1e-4
        # The discriminating bound. A window truncated at the limit spans
        # {+3, +4}, so even a completely flat weighting cannot fall below +3.5.
        # A window slid inward to keep three members would span {+2, +3, +4},
        # whose flat weighting is +3.0 - reachable only by the wrong one. The
        # broad-peak cases are the ones that separate them; the sharp case
        # returns +4 either way and is here to pin that it stays exact.
        assert recovered >= 3.5, f"pulled inward at temperature {temperature}"


def test_peak_at_boundary_flags_a_search_window_too_small_to_contain_the_match():
    """The one failure mode invisible in the reported flow itself: a soft-argmax
    can only report a displacement inside the window it was given, so a true
    motion beyond the search radius still comes back as SOME in-window number,
    not an error. ``peak_at_boundary`` is what would catch it: the winning
    candidate pinned to the edge of the window, every time, is the symptom."""

    from vio.models.correlation import LocalCorrelation

    torch.manual_seed(0)
    base = torch.randn(1, 24, 32, 48)
    # Shifted by exactly the search radius: the best available candidate sits
    # on the boundary everywhere the window is fully inside the image.
    at_limit = LocalCorrelation(radius=4, temperature=0.05)(
        base[:, :, :, 4:44], base[:, :, :, 0:40]
    )
    interior = at_limit["peak_at_boundary"][0, 0, 6:-6, 6:-6]
    assert float(interior.mean()) > 0.9

    # A one-cell shift: comfortably inside a radius-4 window, so the winning
    # candidate should almost never be the edge.
    small_shift = LocalCorrelation(radius=4, temperature=0.05)(
        base[:, :, :, 1:41], base[:, :, :, 0:40]
    )
    interior = small_shift["peak_at_boundary"][0, 0, 6:-6, 6:-6]
    assert float(interior.mean()) < 0.1

    # radius=0 has no window to overflow - the diagnostic is trivially off.
    trivial = LocalCorrelation(radius=0, temperature=0.05)(base, base)
    assert float(trivial["peak_at_boundary"].abs().max()) == 0.0


def test_frontend_exposes_per_pair_diagnostics_separately_from_the_token():
    """The frontend's ``diagnostics`` dict is purely for a caller to log and
    stratify errors by later - it must never reach the token itself."""

    from vio.models.vision_mamba_vo import DIAGNOSTIC_NAMES

    torch.manual_seed(0)
    frontend = _frontend()
    base = _textured(*frontend.image_size)
    with torch.no_grad():
        out = frontend(base, torch.roll(base, shifts=4, dims=3), pair_dt_s=torch.full((1,), 0.05))

    assert set(out["diagnostics"]) == set(DIAGNOSTIC_NAMES)
    for name in DIAGNOSTIC_NAMES:
        value = out["diagnostics"][name]
        assert value.shape == (1,)
        assert torch.isfinite(value).all()
        assert bool((value >= 0.0).all()) and bool((value <= 1.0).all() or name == "mean_entropy")


def test_the_direction_concentration_starts_where_the_fixed_weight_term_was():
    """Enabling the vMF term must not move the starting point.

    kappa is initialised to 1, where kappa*(1-cos) - log kappa is exactly the
    (1-cos) term it replaces. A head that started anywhere else would change
    the first step's gradient and make "did the uncertainty help" impossible to
    answer separately from "did the weighting change".
    """

    model = VisionMambaVO(visual_dim=16, aiding_dim=16, fusion_dim=24)
    hidden = torch.zeros(2, 5, 24)
    out = model.heads(hidden, torch.full((2, 5), float(np.log(120.0))))
    concentration = torch.exp(out["direction_log_concentration"])
    assert torch.allclose(concentration, torch.ones_like(concentration))

    # And the limits are the sphere's, not the variance head's: exp(4) would
    # cap confidence at about 8 degrees of spread, coarser than a working VO.
    assert model.max_log_concentration > model.max_log_variance
    assert model.min_log_concentration > model.min_log_variance


def test_the_concentration_is_bounded_at_both_ends():
    """Unbounded kappa is the failure mode when the target is degenerate.

    With (1 - cos) identically zero the term is -log kappa, whose gradient is a
    constant, so kappa runs away. The clamp is what keeps that finite, and it
    matters because a capture with no crab and no angle of attack produces
    exactly that.
    """

    model = VisionMambaVO(visual_dim=16, aiding_dim=16, fusion_dim=24)
    with torch.no_grad():
        model.log_concentration_head.bias.fill_(1000.0)
    out = model.heads(torch.zeros(1, 3, 24), torch.zeros(1, 3))
    assert torch.all(out["direction_log_concentration"] == model.max_log_concentration)

    with torch.no_grad():
        model.log_concentration_head.bias.fill_(-1000.0)
    out = model.heads(torch.zeros(1, 3, 24), torch.zeros(1, 3))
    assert torch.all(out["direction_log_concentration"] == model.min_log_concentration)
