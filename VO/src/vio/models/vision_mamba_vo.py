"""Visual odometry: vision Mamba for the image, Mamba for the fusion.

No IMU. The aircraft's own attitude solution supplies the rotation that
de-rotates the image motion, and its altitude supplies the scale that turns
angular motion into metres per second.

Three things here are geometry rather than architecture, and each one is a
place where an otherwise reasonable design silently loses the lateral axis.

**Rotation is subtracted at the search, not after it.** Image motion is the sum
of a rotational field, which depends only on body rates and not on depth, and a
translational field, which is what velocity lives in. A yaw rate of 0.1 rad/s
at f = 900 px moves the image about 90 px/s - indistinguishable from several
m/s of lateral drift. So the predicted rotational displacement centres the
correlation search window: the correlator then measures the residual, which is
the translational part, and the bounded window is spent on the signal instead
of on the rotation.

**Scale enters multiplicatively.** A camera measures ``u = v / h``. Direction
is free - the altitude cancels in ``atan2`` - and magnitude is not: nothing in
an image distinguishes 20 m/s at 200 m from 10 m/s at 100 m. So the speed head
predicts a log BEARING RATE and altitude is ADDED in log space, making the
product exact by construction rather than something a linear layer has to
discover. A concatenated altitude channel cannot do this.

**The output is factored into direction and speed.** They have different
observability - direction is recoverable from one frame pair, speed is not -
and factoring them keeps a scale error from corrupting the crab angle, which is
the quantity GPS-free navigation actually needs.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .correlation import LocalCorrelation
from .causal_mamba import BlockState, CausalMambaEncoder, MambaStackState
from .vision_mamba import VisionMambaStem

#: sin/cos roll, sin/cos pitch, log altitude, body rates p/q/r, and the
#: measured interval.
AIDING_INPUT_DIM = 9
#: Index of log altitude within the aiding vector. That copy is CENTRED (see
#: vio.data.fixedwing_vo) and exists for the encoder, not the speed head - the
#: head is always given the raw log altitude explicitly through ``forward``'s
#: own ``log_altitude`` argument, which is what ``v = h * u`` needs.
LOG_ALTITUDE_INDEX = 4


class VOStreamState(NamedTuple):
    """Both encoders' recurrent state, carried between consecutive blocks.

    A training window is short enough to run in one call, so ``forward`` never
    needs this. A horizon evaluation is not: scoring a 30-minute leg means
    180,000 ticks, and the reference scan holds every step's output, so it has
    to be walked in blocks with the state handed from one to the next. That is
    also what makes the reset explicit - a fresh state is the only thing that
    distinguishes one leg from the next.
    """

    aiding: MambaStackState
    fusion: MambaStackState


def mask_stream_state(state: VOStreamState, keep: torch.Tensor) -> VOStreamState:
    """Zero out the batch lanes where ``keep`` is False - a per-lane reset.

    A block's ``initial_state`` is all zeros (see
    :meth:`~vio.models.causal_mamba.CausalSelectiveStateBlock.initial_state`),
    so multiplying every state tensor's batch dimension by a 0/1 mask IS "start
    this lane fresh" (0) or "carry this lane forward" (1) - not two different
    operations that happen to agree. Built for TBPTT training
    (see ``PLAN_TBPTT.txt``): a batch of chronological lanes resets
    independently, one lane at a time, whenever that lane's next window is not
    the true successor of the one its state was built from -
    :class:`~vio.data.fixedwing_vo.ChronologicalWindowSampler` is what decides
    which lanes those are, each step.
    """

    if keep.ndim != 1:
        raise ValueError("keep must be a 1-D (batch,) tensor")

    def mask_block(block: BlockState) -> BlockState:
        conv, ssm = block
        if conv.shape[0] != keep.shape[0] or ssm.shape[0] != keep.shape[0]:
            raise ValueError("keep must have one entry per batch lane")
        # .to(dtype) alone leaves the mask on keep's OWN device - harmless on
        # CPU, where every tensor already shares the one device, but a silent
        # cross-device multiply (RuntimeError) the moment state lives on CUDA
        # and keep does not (e.g. built fresh by a sampler on the CPU).
        conv_keep = keep.to(device=conv.device, dtype=conv.dtype).reshape(
            (-1,) + (1,) * (conv.ndim - 1)
        )
        ssm_keep = keep.to(device=ssm.device, dtype=ssm.dtype).reshape(
            (-1,) + (1,) * (ssm.ndim - 1)
        )
        return conv * conv_keep, ssm * ssm_keep

    return VOStreamState(
        aiding=tuple(mask_block(block) for block in state.aiding),
        fusion=tuple(mask_block(block) for block in state.fusion),
    )


def detach_stream_state(state: VOStreamState) -> VOStreamState:
    """Cut the autograd graph between one TBPTT chunk and the next.

    Without this, a chunk's backward would keep walking into every earlier
    chunk's graph, growing without bound as a stream gets longer - the exact
    cost TRUNCATED backprop-through-time exists to avoid. Call this on the
    state a chunk returns before feeding it to the next chunk's forward pass,
    never before backward() has run on the chunk that produced it.
    """

    return VOStreamState(
        aiding=CausalMambaEncoder.detach_state(state.aiding),
        fusion=CausalMambaEncoder.detach_state(state.fusion),
    )


class RotationalSearchField(nn.Module):
    """Body rate to a PER-CELL feature displacement field.

    This is the camera-to-body rotation, learned rather than assumed, applied
    through the geometry the correlator actually needs: how far the image moves
    AT EACH CELL for a given body rotation over a given interval.

    **Why a single displacement is not enough.** Rotational image motion is not
    a translation. In normalized camera coordinates it is

        u = wx*x*y - wy*(1 + x*x) + wz*y
        v = wx*(1 + y*y) - wy*x*y - wz*x

    A constant offset can represent exactly the order-zero terms, ``-wy`` in u
    and ``+wx`` in v, and nothing else. The term it provably cannot touch is
    the CURL, ``(+y*wz, -x*wz)``, which is rotation about the optical axis:
    that field is ODD about the principal point, so its mean over a symmetric
    image is exactly zero and the best constant approximation IS the zero
    vector. A correctly trained constant map predicts nothing for it, so none
    of it is removed - not because the map is badly fitted, but by construction.

    On this aircraft that residual is not small. At the image corner, with the
    measured yaw rate and the real intrinsics, the un-removable curl is about
    0.40 cells RMS and 0.76 cells at peak, against a lateral (crab) signal of
    about 0.19 cells. It is two to four times the quantity it corrupts, it is
    turn-correlated by construction, and it is not clipped by the search window,
    so it is measured and reported as translation.

    **Why a general 3x3 map.** The learned matrix takes body rates to
    camera-frame rates, and the field is then a linear combination of the three
    basis fields above. Any mounting gives ``w_cam = R w_body``, so a general
    linear map spans every possible camera orientation, and also absorbs axis
    order, sign convention and focal-length error. That matters here: this
    project records no camera-to-body rotation anywhere, and the tool that
    would measure one does not exist, so the mounting must be learned rather
    than assumed. The GEOMETRY is what is hard-coded; the ORIENTATION is not.

    Zero-initialised, so the field starts identically zero, the search starts
    centred, and the module has to earn every offset - the same property the
    constant version had.

    ``mode="constant"`` restores the old single-displacement behaviour, so the
    change can be ablated rather than merely believed.
    """

    def __init__(self, *, mode: str = "field") -> None:
        super().__init__()
        if mode not in ("field", "constant"):
            raise ValueError("mode must be 'field' or 'constant'")
        self.mode = mode
        self.map = nn.Linear(3, 3 if mode == "field" else 2, bias=False)
        nn.init.zeros_(self.map.weight)

    @torch.no_grad()
    def seed_from_jacobian(
        self, jacobian_px_per_rad: torch.Tensor, *, patch_size: int, resize: Tuple[float, float]
    ) -> None:
        """Seed from a camera/body rotation calibration artifact.

        The artifact is a 2x3 matrix in native pixels per radian. The
        correlator works in feature cells of a resized image, so both scalings
        apply: multiply by the resize factor, divide by the patch size.

        No tool in this repository produces that artifact - the name
        ``tools/calibrate_camera_imu_rotation.py`` appears in older comments
        and error messages here and was never checked in. Nothing calls this
        method unless ``--rotation-map`` is used to point at a hand-authored
        or externally produced JSON of the shape
        ``{"fit": {"jacobian_px_per_rad": [[dx_wx, dx_wy, dx_wz],
        [dy_wx, dy_wy, dy_wz]]}}``.

        Only meaningful in ``mode="constant"``. A 2x3 Jacobian records the
        image translation per body rate, which is the very approximation the
        field mode exists to replace: it holds no information about the curl,
        because a translation measurement cannot carry one. Seeding a field
        from it would silently discard two thirds of the parameters and start
        the map at a value that is wrong in a way the artifact cannot describe.
        """

        if self.mode != "constant":
            raise ValueError(
                "seed_from_jacobian needs mode='constant': a 2x3 image-translation "
                "Jacobian cannot describe the rotational FIELD, only its "
                "order-zero term. Drop --rotation-map and let the field be "
                "learned, or set --rotation-mode constant to use the seed."
            )
        matrix = torch.as_tensor(jacobian_px_per_rad, dtype=self.map.weight.dtype)
        if matrix.shape != (2, 3):
            raise ValueError("jacobian must have shape (2, 3): rows dx, dy")
        scale = torch.tensor(
            [resize[1] / patch_size, resize[0] / patch_size],
            dtype=matrix.dtype,
        ).reshape(2, 1)
        self.map.weight.copy_(matrix * scale)

    def forward(
        self,
        body_rate: torch.Tensor,
        dt_s: torch.Tensor,
        *,
        grid: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cells_per_unit: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``(B, 2)`` in constant mode, ``(B, 2, H, W)`` in field mode.

        ``grid`` is the normalized camera coordinate of every feature cell and
        ``cells_per_unit`` converts a normalized displacement into cells, both
        supplied by the frontend because it owns the intrinsics.
        """

        if body_rate.ndim != 2 or body_rate.shape[1] != 3:
            raise ValueError("body_rate must have shape (B, 3)")
        rotation = body_rate * dt_s.reshape(-1, 1)
        if self.mode == "constant":
            return self.map(rotation)

        if grid is None or cells_per_unit is None:
            raise ValueError("field mode needs grid and cells_per_unit")
        # Radians of camera-frame rotation over the interval, not a rate: the
        # dt is already folded in, so the field is a DISPLACEMENT like the
        # constant it replaces, and the correlator's units do not change.
        omega = self.map(rotation)
        wx = omega[:, 0].reshape(-1, 1, 1)
        wy = omega[:, 1].reshape(-1, 1, 1)
        wz = omega[:, 2].reshape(-1, 1, 1)
        x, y = grid
        xy = x * y
        # Longuet-Higgins and Prazdny, rotational part only. It carries no
        # depth, which is the whole reason it can be removed before the search
        # without knowing the scene: translation is what is left over.
        horizontal = wx * xy - wy * (1.0 + x * x) + wz * y
        vertical = wx * (1.0 + y * y) - wy * xy - wz * x
        scale = cells_per_unit.reshape(-1, 2, 1, 1)
        return torch.stack((horizontal, vertical), dim=1) * scale


#: Fixed column order for the per-pair diagnostic scalars in
#: ``VisionMambaFlowFrontend.forward``'s ``"diagnostics"`` dict, so a caller
#: that stacks them into one tensor (see ``vio.models.velocity_horizons``)
#: has one place that says which column is which.
DIAGNOSTIC_NAMES: Tuple[str, ...] = (
    "boundary_hit_fraction",
    "mean_usable_confidence",
    "mean_entropy",
    "mean_entropy_normalized",
    "occupied_fraction",
    # How much of the grid survived the reliability gate, and whether the pair
    # survived at all. These two exist because the older diagnostics are means
    # over DIFFERENT populations - mean_entropy_normalized covers every cell,
    # occupied_fraction only the cells that passed the weight threshold - so
    # neither one tells you how many cells were both measured AND trustworthy.
    # That is the number the gate is tuned on, so it has to be reported
    # directly rather than inferred from the other two.
    "reliable_cell_fraction",
    "pair_reliable_fraction",
    # Which gate did the rejecting. These overlap by construction - a cell can
    # fail two tests - so they do not sum to (1 - reliable_cell_fraction); the
    # question they answer is which threshold is carrying the filtering, which
    # a single total cannot say.
    "rejected_low_confidence",
    "rejected_high_entropy",
    "rejected_low_score_margin",
    "rejected_boundary_peak",
    # The learned softmax temperature actually in force. The gate deliberately
    # does NOT use it (see LocalCorrelation.gate_temperature); logging it is
    # how a reader sees whether it has drifted away from the fixed reference
    # the thresholds were chosen against.
    "correlation_temperature",
)


def frontend_diagnostics(
    correlation: Dict[str, torch.Tensor],
    weight: torch.Tensor,
    *,
    min_pool_weight: float,
    reliable_cell: Optional[torch.Tensor] = None,
    pair_reliable: Optional[torch.Tensor] = None,
    reliable_fraction: Optional[torch.Tensor] = None,
    rejections: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """The scalars behind ``DIAGNOSTIC_NAMES``, factored out of ``forward`` so
    they can be unit-tested directly against a hand-built ``correlation``
    dict instead of only through a real image pipeline.

    ``boundary_hit_fraction`` is restricted to cells that are both USABLE
    (``usable_confidence > 0`` - better than a uniform tie-break) and have a
    COMPLETE search window (``full_window_valid`` - not clipped by the image
    border). Averaging every cell in, as an earlier version did, confounds
    the diagnostic two ways: a border cell's window is clipped, which makes
    its argmax structurally more likely to land on whatever remains of the
    window regardless of where the true match is, and a textureless cell's
    argmax is an arbitrary tie-break - neither says anything about whether
    the search radius is too small, and mixing them in dilutes or inflates
    the fraction with cells that cannot report the thing it exists to catch.

    ``mean_entropy_normalized`` divides by each cell's OWN ``log(candidate_
    count)`` rather than the radius's global ``log(K)``: raw entropy's
    ceiling depends on the search radius (``K = (2r+1)^2``), so it is only
    comparable run to run at a fixed radius, and a border cell's clipped
    window lowers its ceiling further still. Dividing by the per-cell ceiling
    keeps the normalized value in ``[0, 1]`` and comparable both across radii
    and across border/interior cells within one image.
    """

    dtype = weight.dtype
    usable_full_window = correlation["full_window_valid"] & (
        correlation["usable_confidence"] > 0
    )
    diagnostic_count = usable_full_window.sum(dim=(1, 2, 3)).clamp_min(1)
    boundary_hit_fraction = (
        correlation["peak_at_boundary"] * usable_full_window.to(dtype)
    ).sum(dim=(1, 2, 3)) / diagnostic_count

    entropy_normalized = correlation["entropy"] / (
        correlation["candidate_count"].clamp_min(2.0).log()
    )

    batch = weight.shape[0]
    if reliable_cell is None:
        reliable_cell = torch.ones_like(weight)
    if pair_reliable is None:
        pair_reliable = weight.new_ones((batch, 1))

    def zeros_or(source: Optional[Dict[str, torch.Tensor]], name: str) -> torch.Tensor:
        """A rejection fraction, or zero where the caller measured none.

        Zero is the honest value for a gate that is switched off: it rejected
        nothing. NaN would be wrong here - unlike a diagnostic that was not
        computed, this one WAS computed and the answer is none.
        """

        if source is None or name not in source:
            return weight.new_zeros(batch)
        return source[name].to(dtype).reshape(batch)

    return {
        "boundary_hit_fraction": boundary_hit_fraction,
        "mean_usable_confidence": correlation["usable_confidence"].mean(dim=(1, 2, 3)),
        "mean_entropy": correlation["entropy"].mean(dim=(1, 2, 3)),
        "mean_entropy_normalized": entropy_normalized.mean(dim=(1, 2, 3)),
        "occupied_fraction": (weight > min_pool_weight).to(dtype).mean(dim=(1, 2, 3)),
        # The SAME number the pair gate thresholds on - reliable cells over
        # VALID cells. Recomputing a whole-grid mean here instead would
        # report a different quantity than the one that made the decision.
        "reliable_cell_fraction": (
            reliable_fraction.to(dtype).reshape(batch)
            if reliable_fraction is not None
            # Fallback for a hand-built correlation dict (this function is
            # deliberately callable with one - see the docstring), which need
            # not carry flow_valid. A whole-grid mean is the right answer there
            # precisely because there is no validity mask to divide by.
            else reliable_cell.to(dtype).mean(dim=(1, 2, 3))
        ),
        "pair_reliable_fraction": pair_reliable.to(dtype).reshape(batch),
        "rejected_low_confidence": zeros_or(rejections, "rejected_low_confidence"),
        "rejected_high_entropy": zeros_or(rejections, "rejected_high_entropy"),
        "rejected_low_score_margin": zeros_or(rejections, "rejected_low_score_margin"),
        "rejected_boundary_peak": zeros_or(rejections, "rejected_boundary_peak"),
        # NaN, not zero, when the caller's correlation dict carries no
        # temperature: this function is deliberately callable with a
        # hand-built dict (see the docstring), and a temperature of zero would
        # read as a real, absurdly sharp softmax rather than as "not measured".
        # That is the opposite of the rejection fractions above, where zero IS
        # the answer for a gate that is switched off.
        "correlation_temperature": (
            correlation["temperature"].reshape(1).expand(batch).to(dtype)
            if "temperature" in correlation
            else weight.new_full((batch,), float("nan"))
        ),
    }


class VisionMambaFlowFrontend(nn.Module):
    """Image pair to a motion token, via a scanned encoder and correlation.

    The two halves do different jobs and neither can do the other's.
    :class:`VisionMambaStem` builds features with a global receptive field;
    :class:`LocalCorrelation` measures where those features moved. A scanned
    encoder alone answers "did the scene change", never "what moved where" -
    motion is a property of a PAIR, and no single-image descriptor holds it.
    """

    # The measurement algorithm's version, and the only thing that identifies
    # it: two frontends can carry identical tensor shapes and still compute a
    # different quantity, which ``load_state_dict(strict=True)`` cannot see.
    #   v2: confidence-weighted pooling (9 cell channels, was 7), two-stage
    #       argmax. Tensor layout changed.
    #   v3: weights by usable_confidence rather than raw confidence, and
    #       truncates the refinement window at the search limit. SAME shapes as
    #       v2, different semantics - which is exactly why the id has to move.
    #   v4: the search centre is a per-cell FIELD (RotationalSearchField,
    #       mode="field") rather than one displacement for the whole image, so
    #       rotation about the optical axis is actually removed instead of
    #       being provably unremovable by construction; and the centre is
    #       rounded to whole cells before it reaches the correlator, which
    #       fixed a fractional centre losing up to half the measured
    #       displacement. rotation.map's shape changes with rotation_mode
    #       (3x3 field / 2x3 constant), which is why rotation_mode is also
    #       part of the resume fingerprint rather than only the frontend id.
    frontend_id = "vision_mamba_correlation_v4"

    def __init__(
        self,
        *,
        input_channels: int = 1,
        visual_dim: int = 64,
        d_model: int = 64,
        depth: int = 2,
        patch_size: int = 8,
        image_size: Tuple[int, int] = (288, 384),
        context_grid: Tuple[int, int] = (12, 16),
        d_state: int = 8,
        expand: int = 2,
        correlation_radius: int = 4,
        correlation_temperature: float = 0.03,
        token_grid: int = 6,
        dropout: float = 0.0,
        rotation_mode: str = "field",
        min_pool_weight: float = 1e-4,
        max_cell_entropy: float = 1.0,
        min_cell_confidence: float = 0.0,
        min_score_margin: float = 0.0,
        reject_boundary_peaks: bool = False,
        min_reliable_cell_fraction: float = 0.0,
    ) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.patch_size = int(patch_size)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.token_grid = int(token_grid)
        if self.token_grid <= 0:
            raise ValueError("token_grid must be positive")

        self.stem = VisionMambaStem(
            input_channels=input_channels,
            d_model=d_model,
            depth=depth,
            patch_size=patch_size,
            image_size=image_size,
            context_grid=context_grid,
            d_state=d_state,
            expand=expand,
            dropout=dropout,
        )
        if min(self.stem.feature_size) < self.token_grid:
            raise ValueError(
                "image_size / patch_size must be at least token_grid in both axes"
            )
        self.correlation = LocalCorrelation(
            radius=correlation_radius,
            temperature=correlation_temperature,
            learnable_temperature=True,
        )
        self.rotation = RotationalSearchField(mode=rotation_mode)

        # Per pooled cell: confidence-weighted flow rate (2), the pooled weight
        # itself, mean confidence, mean peak margin, mean entropy, the valid
        # fraction, and the rotational displacement that was subtracted (2).
        # The weight channel is what separates "nothing was measurable here"
        # from "the measurement was zero"; keeping the rotation means a wrong
        # rotation map stays visible instead of folding into the residual.
        self.cell_channels = 9
        # Below this pooled weight a cell is treated as unmeasured. It guards
        # the division and keeps a near-empty cell from producing a large
        # arbitrary ratio. The default is a numerical guard, NOT a quality
        # filter: at 1e-4 essentially every cell that correlated at all gets
        # through. Raise it (and the three gates below) to make it one.
        self.min_pool_weight = float(min_pool_weight)
        # Per-cell reliability gates. A cell is rejected when its match is
        # ambiguous (normalized entropy at or above ``max_cell_entropy``),
        # weak (usable confidence at or below ``min_cell_confidence``), or
        # pinned to the edge of the search window (``reject_boundary_peaks``,
        # which means the true match may well lie outside the window entirely,
        # so the reported displacement is a floor rather than a measurement).
        #
        # The defaults disable all three, so an ungated run is bit-for-bit what
        # it was before these existed and ``frontend_id`` stays honest. Turning
        # them on is a deliberate act, recorded in the checkpoint's args.
        self.max_cell_entropy = float(max_cell_entropy)
        self.min_cell_confidence = float(min_cell_confidence)
        # The temperature-free gate: raw top-1 minus top-2 correlation score.
        # In correlation-score units rather than on [0, 1], so it needs its own
        # sweep - but no choice of softmax temperature can move it.
        self.min_score_margin = float(min_score_margin)
        self.reject_boundary_peaks = bool(reject_boundary_peaks)
        # Pair-level gate: if fewer than this fraction of cells survive, the
        # whole pair is refused rather than summarised from the few that did.
        # No measurement beats a confident wrong one - a token built from a
        # handful of cells still arrives with visual_present=1 and is trusted
        # like a full one.
        self.min_reliable_cell_fraction = float(min_reliable_cell_fraction)
        self.grid_dim = max(d_model // 2, 16)
        self.cell_projection = nn.Linear(self.cell_channels, self.grid_dim)
        self.grid_encoder = CausalMambaEncoder(
            self.grid_dim,
            d_model=self.grid_dim,
            depth=depth,
            d_state=d_state,
            expand=expand,
            dropout=dropout,
        )
        cell_dim = (self.cell_channels + self.grid_dim) * self.token_grid ** 2
        hidden = max(2 * self.visual_dim, d_model)
        self.token_projection = nn.Sequential(
            nn.Linear(cell_dim + 1, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.visual_dim),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(cell_dim + 1, max(self.visual_dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(self.visual_dim // 2, 8), 1),
        )

    def _reliable_cells(
        self, correlation: Dict[str, torch.Tensor], dtype: torch.dtype
    ) -> torch.Tensor:
        """1.0 where a cell's match is worth believing, 0.0 where it is not.

        Three independent ways a correlation cell can be wrong, each caught by
        the statistic that actually detects it:

        **Ambiguous.** A blurry or repetitive patch correlates about equally
        well everywhere in the window, so the softmax is close to uniform and
        the argmax is a tie-break. Normalized entropy detects this and the peak
        SCORE does not - a smooth patch scores 0.97 against every candidate,
        which looks like a superb match and is the opposite of one.

        **Weak.** ``usable_confidence`` already has the uniform floor removed,
        so it reaches zero for a no-information cell where raw confidence
        bottoms out at 1/n. A threshold on it is a threshold on real evidence.

        **Clipped.** A peak on the window boundary means the soft-argmax was
        not allowed to look where the match may actually be, so the reported
        displacement is a lower bound reported as a measurement.

        All three default to off, and a cell must fail none of the enabled ones
        to survive.

        Every threshold reads the ``gate_*`` statistics, which are computed at
        a FIXED temperature and detached - never the learned-temperature ones
        the displacement estimate uses. Confidence and entropy are properties
        of the softmax as much as of the match, so a learnable temperature
        would let the model decide whether its own measurement passes, and
        would make one threshold mean different things at different points in
        training. See ``LocalCorrelation.gate_temperature``.
        """

        reliable = correlation["flow_valid"].to(dtype)
        if self.min_cell_confidence > 0.0:
            reliable = reliable * (
                correlation["gate_usable_confidence"] > self.min_cell_confidence
            ).to(dtype)
        if self.max_cell_entropy < 1.0:
            # Each cell's OWN ceiling, log(candidate_count), not the radius's
            # global log(K): a border cell has a clipped window and therefore a
            # lower ceiling, and judging it against log(K) would make it look
            # more certain than it is. Same normalisation frontend_diagnostics
            # uses, so the gate and the reported number agree.
            entropy_normalized = correlation["gate_entropy"] / (
                correlation["candidate_count"].clamp_min(2.0).log()
            )
            reliable = reliable * (entropy_normalized < self.max_cell_entropy).to(dtype)
        if self.min_score_margin > 0.0:
            # The temperature-free option: raw top-1 minus top-2 correlation
            # score. Immune to the softmax question entirely, at the cost of
            # being in correlation-score units rather than on [0, 1].
            reliable = reliable * (
                correlation["score_margin"] > self.min_score_margin
            ).to(dtype)
        if self.reject_boundary_peaks:
            reliable = reliable * (correlation["peak_at_boundary"] <= 0).to(dtype)
        return reliable

    def rejection_reasons(
        self, correlation: Dict[str, torch.Tensor], dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        """Per-cell rejection fraction attributed to each gate, independently.

        Each entry counts cells that gate would reject ON ITS OWN, among cells
        with a complete search window. They therefore overlap and do not sum to
        the total rejected - which is the useful behaviour: the question these
        answer is "which gate is doing the work", and a cell rejected by two of
        them is genuinely rejected by both.
        """

        window = correlation["flow_valid"].to(dtype)
        total = window.sum(dim=(1, 2, 3)).clamp_min(1.0)
        entropy_normalized = correlation["gate_entropy"] / (
            correlation["candidate_count"].clamp_min(2.0).log()
        )

        def fraction(rejected: torch.Tensor) -> torch.Tensor:
            return (rejected.to(dtype) * window).sum(dim=(1, 2, 3)) / total

        return {
            "rejected_low_confidence": fraction(
                correlation["gate_usable_confidence"] <= self.min_cell_confidence
                if self.min_cell_confidence > 0.0
                else torch.zeros_like(window, dtype=torch.bool)
            ),
            "rejected_high_entropy": fraction(
                entropy_normalized >= self.max_cell_entropy
                if self.max_cell_entropy < 1.0
                else torch.zeros_like(window, dtype=torch.bool)
            ),
            "rejected_low_score_margin": fraction(
                correlation["score_margin"] <= self.min_score_margin
                if self.min_score_margin > 0.0
                else torch.zeros_like(window, dtype=torch.bool)
            ),
            "rejected_boundary_peak": fraction(
                correlation["peak_at_boundary"] > 0
                if self.reject_boundary_peaks
                else torch.zeros_like(window, dtype=torch.bool)
            ),
        }

    def _bearing_scale(
        self,
        camera_matrix: Optional[torch.Tensor],
        native_size: Tuple[int, int],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalized bearing per feature cell, per axis.

        One cell spans ``patch_size`` working pixels; rescaling to the native
        image and dividing by the focal length turns cells into normalized
        camera coordinates, so a model trained on one camera is not silently
        relearning another camera's focal length.
        """

        if camera_matrix is None:
            scale = torch.tensor(
                [1.0 / self.image_size[1], 1.0 / self.image_size[0]],
                device=device,
                dtype=dtype,
            )
            return (scale * self.patch_size).reshape(1, 2).expand(batch, 2)
        matrix = camera_matrix.to(device=device, dtype=dtype)
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0).expand(batch, 3, 3)
        if matrix.shape != (batch, 3, 3):
            raise ValueError("camera_matrix must have shape (3, 3) or (B, 3, 3)")
        native_height, native_width = native_size
        focal_x = matrix[:, 0, 0] * (self.image_size[1] / float(native_width))
        focal_y = matrix[:, 1, 1] * (self.image_size[0] / float(native_height))
        if torch.any(focal_x <= 0) or torch.any(focal_y <= 0):
            raise ValueError("camera_matrix must have positive focal lengths")
        return torch.stack((self.patch_size / focal_x, self.patch_size / focal_y), dim=-1)

    def _normalised_grid(
        self,
        camera_matrix: Optional[torch.Tensor],
        native_size: Tuple[int, int],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalized camera coordinates at every feature-cell centre.

        The rotational field is a function of WHERE in the image a cell sits,
        so the search centre needs the geometry that :meth:`_bearing_scale`
        deliberately throws away: the principal point. Without a calibration
        the image centre is assumed, which is the same assumption the bearing
        fallback already makes about the focal length.
        """

        height, width = self.stem.feature_size
        columns = torch.arange(width, device=device, dtype=dtype)
        rows = torch.arange(height, device=device, dtype=dtype)
        # Cell CENTRES, not corners: the field is evaluated where the
        # correlation for that cell is anchored.
        x_pixels = (columns + 0.5) * self.patch_size
        y_pixels = (rows + 0.5) * self.patch_size

        if camera_matrix is None:
            focal_x = torch.full((batch,), float(self.image_size[1]), device=device, dtype=dtype)
            focal_y = torch.full((batch,), float(self.image_size[0]), device=device, dtype=dtype)
            centre_x = torch.full((batch,), 0.5 * self.image_size[1], device=device, dtype=dtype)
            centre_y = torch.full((batch,), 0.5 * self.image_size[0], device=device, dtype=dtype)
        else:
            matrix = camera_matrix.to(device=device, dtype=dtype)
            if matrix.ndim == 2:
                matrix = matrix.unsqueeze(0).expand(batch, 3, 3)
            if matrix.shape != (batch, 3, 3):
                raise ValueError("camera_matrix must have shape (3, 3) or (B, 3, 3)")
            native_height, native_width = native_size
            resize_x = self.image_size[1] / float(native_width)
            resize_y = self.image_size[0] / float(native_height)
            focal_x = matrix[:, 0, 0] * resize_x
            focal_y = matrix[:, 1, 1] * resize_y
            centre_x = matrix[:, 0, 2] * resize_x
            centre_y = matrix[:, 1, 2] * resize_y

        x = (x_pixels.view(1, 1, width) - centre_x.view(batch, 1, 1)) / focal_x.view(batch, 1, 1)
        y = (y_pixels.view(1, height, 1) - centre_y.view(batch, 1, 1)) / focal_y.view(batch, 1, 1)
        return x.expand(batch, height, width), y.expand(batch, height, width)

    def forward(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
        *,
        pair_dt_s: Optional[torch.Tensor] = None,
        body_rate_rad_s: Optional[torch.Tensor] = None,
        camera_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if image0.ndim != 4 or image0.shape != image1.shape:
            raise ValueError("images must be matching (B, C, H, W) tensors")
        if not torch.all(torch.isfinite(image0)) or not torch.all(torch.isfinite(image1)):
            raise ValueError("images contain non-finite values")
        batch = image0.shape[0]
        native_size = (int(image0.shape[2]), int(image0.shape[3]))
        if pair_dt_s is None:
            pair_dt_s = image0.new_ones((batch, 1))
        pair_dt_s = pair_dt_s.reshape(batch, 1).to(device=image0.device, dtype=image0.dtype)
        if torch.any(~torch.isfinite(pair_dt_s)) or torch.any(pair_dt_s <= 0):
            raise ValueError("pair_dt_s must be finite and positive")
        if body_rate_rad_s is None:
            body_rate_rad_s = image0.new_zeros((batch, 3))
        body_rate_rad_s = body_rate_rad_s.reshape(batch, 3).to(
            device=image0.device, dtype=image0.dtype
        )

        features0 = self.stem(image0)
        features1 = self.stem(image1)

        scale = self._bearing_scale(
            camera_matrix, native_size, batch, image0.device, image0.dtype
        )
        # The rotational displacement the attitude predicts, in cells, EVALUATED
        # PER CELL. Centring the search here is what leaves the correlator
        # measuring translation - and a single displacement cannot do it,
        # because the curl term of the rotational field has zero mean and so
        # survives any constant untouched. See RotationalSearchField.
        if self.rotation.mode == "constant":
            center = self.rotation(body_rate_rad_s, pair_dt_s)
        else:
            center = self.rotation(
                body_rate_rad_s,
                pair_dt_s,
                grid=self._normalised_grid(
                    camera_matrix, native_size, batch, image0.device, image0.dtype
                ),
                # scale is normalized bearing PER CELL, so its reciprocal is
                # cells per normalized unit, which is what the field needs.
                cells_per_unit=scale.reciprocal(),
            )
        correlation = self.correlation(features0, features1, search_center=center)
        # The translational part is the total motion minus the rotation that was
        # PREDICTED, and the exact prediction is what must be subtracted.
        #
        # ``local_flow`` is measured about the search centre, and the correlator
        # rounds that centre to whole cells (see LocalCorrelation._center_map),
        # so using it directly would leave up to half a cell of the rotational
        # field in the reported translation - quantisation noise proportional to
        # turn rate, which is the very error being removed. ``flow`` is measured
        # against the same rounded lattice but is an absolute displacement, so
        # subtracting the unrounded centre from it is exact.
        centre_cells = center if center.ndim == 4 else center.reshape(batch, 2, 1, 1)
        residual_flow = (correlation["flow"] - centre_cells) * scale.reshape(batch, 2, 1, 1)
        residual_flow = residual_flow / pair_dt_s.reshape(batch, 1, 1, 1)
        # Same units as residual_flow: normalized bearing per second. In field
        # mode this is the real spatial pattern rather than one number smeared
        # over the grid, so a wrong rotation map now shows up as a wrong SHAPE
        # in the pooled cells, not merely a wrong offset.
        if center.ndim == 2:
            rotational = (center * scale).reshape(batch, 2, 1, 1) / pair_dt_s.reshape(
                batch, 1, 1, 1
            )
            rotational = rotational.expand(-1, -1, *residual_flow.shape[-2:])
        else:
            rotational = center * scale.reshape(batch, 2, 1, 1)
            rotational = rotational / pair_dt_s.reshape(batch, 1, 1, 1)

        # Confidence-weighted reduction, not a plain mean. A plain mean averages
        # a confident measurement together with a cell that could not measure at
        # all, and reporting the mean confidence alongside it does NOT let a
        # later layer undo the damage: two blocks holding twice different true
        # flow can produce identical means in every channel, because the plain
        # mean destroys the flow-confidence cross moment. The correct statistic
        # is sum(w*f)/sum(w), and it has to be formed here or not at all.
        dtype = residual_flow.dtype
        grid = (self.token_grid, self.token_grid)
        # ``usable_confidence``, not ``confidence``: a uniform distribution over
        # n candidates still peaks at 1/n, and weighting by that floor does NOT
        # suppress an ambiguous region. Where every cell in a pooled block is
        # ambiguous the normalisation cancels the small weight out again -
        # mean(bad*w)/mean(w) = bad - and the block reports the argmax tie-break
        # at full magnitude. Subtracting the floor sends that weight to exactly
        # zero, so the block is marked unmeasured instead.
        weight = correlation["usable_confidence"] * correlation["flow_valid"].to(dtype)
        # Reliability gate, applied BEFORE pooling so a rejected cell cannot
        # contribute to its block's weighted mean at all. Folding it into
        # ``weight`` rather than masking afterwards is what makes the rejection
        # consistent: the same tensor drives the flow average, the occupancy
        # mask, and the pooled weight channel the fusion model reads.
        reliable_cell = self._reliable_cells(correlation, dtype)
        weight = weight * reliable_cell
        # One number per pair: did enough of the MEASURABLE grid survive?
        #
        # The denominator is the cells with a complete search window, NOT every
        # cell. A border cell was never a candidate - its window is clipped, so
        # flow_valid excludes it before any threshold is consulted - and
        # counting it as "unreliable" would make the fraction depend on the
        # border, which in turn depends on image size, patch size and
        # correlation radius. At the trainer's real geometry that caps the
        # fraction near 0.68, so --min-reliable-cell-fraction 0.5 would quietly
        # mean "73% of the cells that could be measured". Dividing by the valid
        # count makes 1.0 reachable and the threshold mean what it says, at any
        # geometry - and matches the denominator rejection_reasons already uses.
        valid_cell = correlation["flow_valid"].to(dtype)
        valid_count = valid_cell.sum(dim=(1, 2, 3)).clamp_min(1.0)
        reliable_fraction = reliable_cell.sum(dim=(1, 2, 3)) / valid_count
        pair_reliable = (
            reliable_fraction >= self.min_reliable_cell_fraction
        ).to(dtype).reshape(batch, 1)
        weight_pooled = F.adaptive_avg_pool2d(weight, grid)
        flow_pooled = F.adaptive_avg_pool2d(residual_flow * weight, grid) / (
            weight_pooled.clamp_min(self.min_pool_weight)
        )
        # The occupancy DECISION runs on fixed-temperature confidence, while
        # the weighted average above keeps the learned one.
        #
        # Those are two different jobs. Weighting a mean is a modelling choice
        # and training should be free to sharpen it. Deciding whether a block
        # counts as measured at all is a quality judgement, and confidence is a
        # property of the softmax as much as of the match - so with the learned
        # value here, shrinking the temperature alone would push blocks past
        # min_pool_weight and change the token, with no measurement improving.
        # pair_reliable was already protected from this; without this line the
        # second filtering stage was not.
        gate_weight = (
            correlation["gate_usable_confidence"] * valid_cell * reliable_cell
        )
        gate_weight_pooled = F.adaptive_avg_pool2d(gate_weight, grid)
        # An unmeasurable cell must not look like a measured zero. The two-stage
        # argmax reports an arbitrary peak rather than collapsing to the window
        # centre, so an ambiguous cell carries a large arbitrary displacement,
        # not a small one. With the uniform floor removed its weight is exactly
        # zero, so this mask fires and the flow is discarded rather than merely
        # down-weighted.
        occupied = (gate_weight_pooled > self.min_pool_weight).to(dtype)
        flow_pooled = flow_pooled * occupied
        pooled = torch.cat(
            (
                flow_pooled,
                weight_pooled,
                F.adaptive_avg_pool2d(correlation["usable_confidence"], grid),
                F.adaptive_avg_pool2d(correlation["probability_peak_margin"], grid),
                F.adaptive_avg_pool2d(correlation["entropy"], grid),
                F.adaptive_avg_pool2d(correlation["flow_valid"].to(dtype), grid),
                F.adaptive_avg_pool2d(rotational, grid),
            ),
            dim=1,
        )
        cells_sequence = pooled.flatten(2).transpose(1, 2)
        context, _ = self.grid_encoder.forward_sequence(
            self.cell_projection(cells_sequence)
        )
        cells = torch.cat((pooled.flatten(1), context.flatten(1)), dim=-1)
        pair = torch.cat((cells, torch.log(pair_dt_s.clamp_min(1e-6))), dim=-1)
        # Per-pair scalar summaries a caller can log and stratify the eventual
        # velocity error by - none of these feed the model. ``occupied_fraction``
        # is measured at full feature-grid resolution, matching ``weight``
        # above, not the coarser pooled token-grid ``occupied`` a mostly-full
        # block would inflate.
        diagnostics = frontend_diagnostics(
            # gate_weight, not weight: occupied_fraction has to report the
            # decision that was actually taken, and that decision is now made
            # on the fixed-temperature weight.
            correlation, gate_weight, min_pool_weight=self.min_pool_weight,
            reliable_cell=reliable_cell, pair_reliable=pair_reliable,
            reliable_fraction=reliable_fraction,
            rejections=self.rejection_reasons(correlation, dtype),
        )
        return {
            "visual_token": self.token_projection(pair),
            "visual_quality": torch.sigmoid(self.quality_head(pair)),
            # Whether this pair's token should be delivered at all. The caller
            # decides what to do with it - the frontend does not zero its own
            # token, because "the token is zero" and "no token arrived" are the
            # two things visual_present exists to keep apart.
            "pair_reliable": pair_reliable,
            "translational_flow_normalized_per_s": residual_flow,
            "rotational_flow_normalized_per_s": rotational,
            "confidence_map": correlation["confidence"],
            # The pooled field the token is actually built from. Exposed because
            # a collapsed weight channel is the first thing to look at when the
            # speed scale drifts, and it cannot be inferred from the token.
            "pooled_cells": pooled,
            "diagnostics": diagnostics,
        }


class VisionMambaVO(nn.Module):
    """Causal VO over attitude, altitude and scanned image motion.

    Fusion input per tick is
    ``[aiding, visual_token, visual_present, visual_age]``.
    ``visual_present`` is explicit because a zero token otherwise means either
    "no image arrived" or "the image showed zero motion", and those want
    opposite responses. ``visual_age`` - physical staleness of the currently
    held token, in seconds, see
    :func:`vio.data.fixedwing_vo.visual_age_seconds` - is explicit for the
    same reason: the recurrent state could in principle infer staleness on its
    own, but only after learning to, and only as well as gradient descent
    happens to find; naming it removes that burden entirely.
    """

    # The FUSION input contract's version - aiding-vector layout (does it
    # carry body rates?) and what gets concatenated onto the token (does
    # visual_age exist?) - the same role frontend_id plays for the frontend.
    # AIDING_INPUT_DIM changing already fails load_state_dict on a shape
    # mismatch, but a same-shape semantic change (an ablation flag flipped
    # between training and evaluation, say) would not be caught by shape
    # alone; this is checked by NAME instead, at evaluation time.
    #   v1: [sin_roll, cos_roll, sin_pitch, cos_pitch, log_altitude, dt] aiding
    #       (6-wide), fusion input [aiding, token, visual_present] (no age).
    #   v2: aiding gained p/q/r (9-wide); fusion input gained visual_age.
    temporal_input_id = "vo_temporal_fusion_v2"

    def __init__(
        self,
        *,
        visual_dim: int = 64,
        aiding_dim: int = 64,
        fusion_dim: int = 96,
        aiding_depth: int = 2,
        fusion_depth: int = 2,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.0,
        min_log_variance: float = -8.0,
        max_log_variance: float = 4.0,
        min_log_concentration: float = -4.0,
        max_log_concentration: float = 10.0,
        log_bearing_rate_init: float = -1.386,
        max_log_speed: float = 5.0,
        frontend: Optional[VisionMambaFlowFrontend] = None,
    ) -> None:
        super().__init__()
        if min(visual_dim, aiding_dim, fusion_dim) <= 0:
            raise ValueError("VO dimensions must be positive")
        if min_log_variance >= max_log_variance:
            raise ValueError("Invalid log-variance limits")
        self.visual_dim = int(visual_dim)
        self.fusion_dim = int(fusion_dim)
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)
        # Concentration limits, in the same spirit as the variance limits but a
        # different range because they measure a different thing. kappa is a
        # precision on the sphere: exp(-4) ~ 0.02 is "the direction is anyone's
        # guess", exp(10) ~ 22000 is about 0.4 deg of spread. The variance
        # head's +4 ceiling would cap confidence at ~8 deg, which is coarser
        # than a working VO should ever be forced to admit to.
        self.min_log_concentration = float(min_log_concentration)
        self.max_log_concentration = float(max_log_concentration)
        self.max_log_speed = float(max_log_speed)
        self.frontend = frontend

        encoder_kwargs = dict(
            dropout=dropout, d_state=d_state, expand=expand, d_conv=d_conv
        )
        self.aiding_encoder = CausalMambaEncoder(
            AIDING_INPUT_DIM, d_model=aiding_dim, depth=aiding_depth, **encoder_kwargs
        )
        self.visual_norm = nn.LayerNorm(self.visual_dim)
        self.visual_quality_projection = nn.Linear(1, self.visual_dim, bias=False)
        # +1 for visual_present, +1 for visual_age - both scalars appended
        # alongside the token rather than folded into it, same reasoning as
        # visual_present: an implicit signal only helps once training has
        # already found it, and both are one linear read away either way.
        self.fusion_input_dim = aiding_dim + self.visual_dim + 2
        self.fusion_encoder = CausalMambaEncoder(
            self.fusion_input_dim,
            d_model=self.fusion_dim,
            depth=fusion_depth,
            **encoder_kwargs,
        )

        # Direction: zero weights with a forward-pointing bias, so an untrained
        # model predicts straight-and-level flight rather than a zero vector
        # that cannot be normalised.
        self.direction_head = nn.Linear(self.fusion_dim, 3)
        nn.init.zeros_(self.direction_head.weight)
        with torch.no_grad():
            self.direction_head.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))

        # Speed: predicts a log BEARING RATE. Altitude is added in log space by
        # forward(), so the model starts at v = h * exp(init) - already the
        # correct physical relationship, with only the residual left to learn.
        self.log_rate_head = nn.Linear(self.fusion_dim, 1)
        nn.init.zeros_(self.log_rate_head.weight)
        nn.init.constant_(self.log_rate_head.bias, float(log_bearing_rate_init))

        self.log_variance_head = nn.Linear(self.fusion_dim, 3)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.zeros_(self.log_variance_head.bias)

        # Direction concentration: the von Mises-Fisher analogue of the
        # variance head. Zero-initialised to kappa = 1, which makes the
        # direction term at step 0 identical to the fixed-weight term it
        # replaces - see velocity_loss - so turning this on does not move the
        # starting point, only what the optimiser is allowed to do next.
        self.log_concentration_head = nn.Linear(self.fusion_dim, 1)
        nn.init.zeros_(self.log_concentration_head.weight)
        nn.init.zeros_(self.log_concentration_head.bias)

    @property
    def backend_name(self) -> str:
        return "vo_vision_mamba"

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> VOStreamState:
        """A zeroed state - what a cold start, or a deliberate reset, looks like."""

        return VOStreamState(
            aiding=self.aiding_encoder.initial_state(
                batch_size, device=device, dtype=dtype
            ),
            fusion=self.fusion_encoder.initial_state(
                batch_size, device=device, dtype=dtype
            ),
        )

    def fuse_stream(
        self,
        aiding: torch.Tensor,
        visual_token: torch.Tensor,
        visual_present: torch.Tensor,
        visual_age: torch.Tensor,
        visual_quality: Optional[torch.Tensor] = None,
        state: Optional[VOStreamState] = None,
    ) -> Tuple[torch.Tensor, VOStreamState]:
        """One block of ticks, returning the state the next block continues from.

        ``state=None`` means a reset, and produces exactly what :meth:`fuse`
        produces for the same input - the two share this body so a streamed
        run and a windowed run cannot drift apart.
        """

        if aiding.ndim != 3 or aiding.shape[2] != AIDING_INPUT_DIM:
            raise ValueError(f"aiding must have shape (B, T, {AIDING_INPUT_DIM})")
        if visual_token.shape[:2] != aiding.shape[:2]:
            raise ValueError("visual_token must align with aiding in batch and time")
        if visual_token.shape[2] != self.visual_dim:
            raise ValueError(f"visual_token must have width {self.visual_dim}")
        if visual_age.shape != visual_present.shape:
            raise ValueError("visual_age must have the same shape as visual_present")
        token = self.visual_norm(visual_token)
        if visual_quality is not None:
            # Folded into the token rather than appended, so the fusion
            # contract stays fixed at [aiding, token, presence, age].
            token = token + self.visual_quality_projection(visual_quality)
        token = token * visual_present
        aiding_state = None if state is None else state.aiding
        fusion_state = None if state is None else state.fusion
        encoded, aiding_state = self.aiding_encoder.forward_sequence(
            aiding, aiding_state
        )
        fused = torch.cat((encoded, token, visual_present, visual_age), dim=-1)
        hidden, fusion_state = self.fusion_encoder.forward_sequence(
            fused, fusion_state
        )
        return hidden, VOStreamState(aiding=aiding_state, fusion=fusion_state)

    def fuse(
        self,
        aiding: torch.Tensor,
        visual_token: torch.Tensor,
        visual_present: torch.Tensor,
        visual_age: torch.Tensor,
        visual_quality: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden, _ = self.fuse_stream(
            aiding, visual_token, visual_present, visual_age, visual_quality, state=None
        )
        return hidden

    def heads(self, hidden: torch.Tensor, log_altitude: torch.Tensor) -> Dict[str, torch.Tensor]:
        direction = F.normalize(self.direction_head(hidden), dim=-1, eps=1e-6)
        log_rate = self.log_rate_head(hidden).squeeze(-1)
        # v = h * u, exactly, because a sum in log space is a product outside
        # it. The clamp only guards the exponential against a diverging head.
        log_speed = torch.clamp(log_rate + log_altitude, max=self.max_log_speed)
        speed = torch.exp(log_speed)
        return {
            "predicted_velocity": direction * speed.unsqueeze(-1),
            "predicted_direction": direction,
            "predicted_speed": speed,
            "predicted_log_bearing_rate": log_rate,
            "velocity_log_variance": self.log_variance_head(hidden).clamp(
                self.min_log_variance, self.max_log_variance
            ),
            "direction_log_concentration": self.log_concentration_head(hidden)
            .squeeze(-1)
            .clamp(self.min_log_concentration, self.max_log_concentration),
            "fusion_embedding": hidden,
        }

    def forward(
        self,
        aiding: torch.Tensor,
        visual_token: torch.Tensor,
        visual_present: torch.Tensor,
        visual_age: torch.Tensor,
        *,
        visual_quality: Optional[torch.Tensor] = None,
        log_altitude: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs, _ = self.forward_stream(
            aiding,
            visual_token,
            visual_present,
            visual_age,
            visual_quality=visual_quality,
            log_altitude=log_altitude,
            state=None,
        )
        return outputs

    def forward_stream(
        self,
        aiding: torch.Tensor,
        visual_token: torch.Tensor,
        visual_present: torch.Tensor,
        visual_age: torch.Tensor,
        *,
        visual_quality: Optional[torch.Tensor] = None,
        log_altitude: Optional[torch.Tensor] = None,
        state: Optional[VOStreamState] = None,
    ) -> Tuple[Dict[str, torch.Tensor], VOStreamState]:
        """:meth:`forward` over one block of a longer run, plus the carried state."""

        hidden, state = self.fuse_stream(
            aiding, visual_token, visual_present, visual_age, visual_quality, state=state
        )
        if log_altitude is None:
            # aiding[..., LOG_ALTITUDE_INDEX] is the CENTRED copy the encoder
            # sees (see vio.data.fixedwing_vo), not the raw one v = h * u needs.
            # A silent fallback here would scale every predicted speed by
            # exp(-log_altitude_mean) with nothing in the loss to say why - the
            # exact failure mode fixedwing_vo.py's module docstring warns about
            # for the encoder/head split. Callers must supply the raw value.
            raise ValueError(
                "log_altitude must be supplied explicitly: it must be the RAW "
                "log altitude (v = h * u), not the centred copy carried in the "
                "aiding vector at LOG_ALTITUDE_INDEX."
            )
        if log_altitude.shape != hidden.shape[:2]:
            raise ValueError("log_altitude must have shape (B, T)")
        return self.heads(hidden, log_altitude), state


__all__ = [
    "AIDING_INPUT_DIM",
    "DIAGNOSTIC_NAMES",
    "LOG_ALTITUDE_INDEX",
    "RotationalSearchField",
    "VOStreamState",
    "VisionMambaFlowFrontend",
    "VisionMambaVO",
    "detach_stream_state",
    "frontend_diagnostics",
    "mask_stream_state",
]
