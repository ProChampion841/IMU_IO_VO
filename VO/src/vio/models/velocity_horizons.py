"""Velocity error as a function of how long the estimator has been running.

The training metrics answer one question: how wrong is the velocity on a
600-tick window that starts from a zeroed state. That is six seconds. It says
nothing about what happens on the tenth minute of an unbroken run, and the
difference is not a detail - a recurrent estimator that is accurate for six
seconds and drifts over ten minutes produces exactly the same ``val_vel_rmse``
as one that does not.

So: **one run**. Zero the state once at the split's first tick, stream the
split unbroken to its end in time order, and never reset again. That single
pass is the whole inference.

A horizon is then a PREFIX of that run - the first H minutes, measured from
the split's own beginning::

    |--0.5m--|
    |-----1m-----|
    |----------5m----------|
    |--------------------10m--------------------|
      ... nested, every one of them starting at the split's first tick

Each horizon reports the error over its prefix alone: velocity magnitude,
velocity direction, and, when a reference attitude is supplied, the
dead-reckoned POSITION the velocity integrates to over that same prefix. The
position figure is the operational one: "5 m/s RMSE" is abstract, "165 m off
after ten minutes, 1.4% of the path flown" is not.

Because the prefixes nest, the numbers compose the way a reader expects. The
maximum can only grow with H - the first five minutes are inside the first
forty - so a longer horizon scoring a *smaller* worst case is impossible by
construction. An RMSE that stays flat as H grows means the state is not
accumulating error; one that rises means it is, and by how much.

Two things make this affordable.

**The frontend runs once.** A visual token depends on its image pair and the
body rate over that pair's exposure, not on any recurrent state, so the tokens
for a span are computed once by :func:`encode_span_tokens` and reused.

**The scan runs once too.** Every horizon watches the same pass through the
model; a prefix is a mask over the ticks it covers, not another forward. Eight
horizons therefore cost what one costs, and the whole-split series the plots
draw comes out of the same pass rather than a second one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vio.data.attitude import AttitudeAltitude, pair_geometry
from vio.data.fixedwing_vo import (
    BODY_RATE_AIDING_SLICE,
    hold_visual_velocity,
    visual_age_seconds,
)
from vio.data.image_pairs import VisualPairSource
from vio.utils.velocity_metrics import AXIS_NAMES

from .vision_mamba_vo import DIAGNOSTIC_NAMES, VisionMambaVO, VOStreamState

#: The horizons the runbook quotes, in minutes.
DEFAULT_HORIZON_MINUTES: Tuple[float, ...] = (
    0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0,
)

_EPS = 1e-6


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------


def horizon_label(minutes: float) -> str:
    """``5.0 -> 'h5m'``. Fractional horizons keep their decimals: ``0.5 -> 'h0.5m'``."""

    value = float(minutes)
    if value <= 0:
        raise ValueError("A horizon must be a positive number of minutes")
    return f"h{value:g}m"


def parse_horizon_minutes(text: Optional[str]) -> Tuple[float, ...]:
    """``"1,5,10"`` -> ``(1.0, 5.0, 10.0)``; empty or None -> ``()``."""

    if text is None:
        return ()
    values: List[float] = []
    for piece in str(text).replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError as exc:
            raise ValueError(f"Not a number of minutes: {piece!r}") from exc
        if value <= 0:
            raise ValueError(f"A horizon must be positive, got {value}")
        values.append(value)
    # Sorted and de-duplicated, so the reported table always reads small to
    # large however the flag was typed.
    return tuple(sorted(dict.fromkeys(values)))


#: Reported per horizon, in this order. RMSE says how wrong the estimator
#: usually is; the maximum says how wrong it got, which is the number a
#: flight-envelope argument actually needs and which an RMSE can hide
#: completely - one bad second in a ten-minute leg moves the RMS by almost
#: nothing and moves the maximum by all of it.
HORIZON_ROW_METRICS: Tuple[str, ...] = (
    "vel_rmse",
    "vel_max_error",
    # WHEN the worst tick happened, in seconds since its own leg's reset. A
    # maximum with no timestamp says only that something went wrong; with one,
    # it says which second of the flight to go and look at - and whether the
    # worst moment is early (a warm-up artefact) or late (accumulating drift).
    "vel_max_error_time_s",
    "vel_dir_rmse",
    "vel_dir_max_error",
    "vel_dir_max_error_time_s",
    # Frontend diagnostics (see vio.models.vision_mamba_vo.DIAGNOSTIC_NAMES),
    # meaned over the visual events inside this horizon's own prefix. Not
    # fed to the model - here purely so a run's trend over epochs, or a
    # comparison across checkpoints, is one column lookup rather than a
    # rerun with extra instrumentation.
    *(f"diag_{name}" for name in DIAGNOSTIC_NAMES),
)

#: Added when a body->NED rotation is supplied, so the velocity can be
#: integrated into a distance. ``pos_error_final`` is the drift at the END of a
#: leg - the number a dead-reckoning claim is actually made of - and
#: ``pos_drift_percent`` normalises it by the path flown, which is how odometry
#: results are usually quoted.
HORIZON_POSITION_METRICS: Tuple[str, ...] = (
    "pos_error_final",
    "pos_error_max",
    "pos_error_max_time_s",
    "pos_error_rmse",
    "pos_drift_percent",
)


def horizon_metric_names(
    minutes: Iterable[float], *, position: bool = False
) -> Tuple[str, ...]:
    """The metrics.csv column stems a horizon set contributes."""

    metrics = HORIZON_ROW_METRICS + (HORIZON_POSITION_METRICS if position else ())
    names: List[str] = []
    for value in minutes:
        label = horizon_label(value)
        names.extend(f"{metric}_{label}" for metric in metrics)
    return tuple(names)


# ---------------------------------------------------------------------------
# horizon prefixes
# ---------------------------------------------------------------------------


def ticks_per_horizon(
    times_s: np.ndarray, span: Tuple[int, int], minutes: float
) -> int:
    """How many telemetry ticks a horizon covers, from the clock in the span.

    The median interval is used rather than the nominal rate because the
    latter is an assumption and the former is a measurement; a logger that
    actually ran at 100.08 Hz would otherwise make every leg slightly short.
    """

    start, end = int(span[0]), int(span[1])
    times = np.asarray(times_s, dtype=np.float64)[start:end]
    if times.size < 2:
        raise ValueError("A span needs at least two telemetry samples")
    interval = float(np.median(np.diff(times)))
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("Telemetry clock has no usable positive interval")
    return max(int(round(float(minutes) * 60.0 / interval)), 1)


def prefix_leg(span: Tuple[int, int], ticks: int) -> Optional[Tuple[int, int]]:
    """The leading ``ticks`` of ``span``, or ``None`` when the span is shorter.

    A horizon is the FIRST H minutes of the split and nothing else: one leg,
    starting at the split's own first tick, running forward in time. It is not
    the split cut into repeated H-minute pieces, so there is no trailing
    remainder to drop and no second cold start to average in. A horizon that
    does not fit is reported skipped, never quietly shortened.
    """

    start, end = int(span[0]), int(span[1])
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    if start + ticks > end:
        return None
    return start, start + ticks


# ---------------------------------------------------------------------------
# visual tokens over a contiguous span
# ---------------------------------------------------------------------------


@dataclass
class SpanTokens:
    """Frontend output for every visual event in a span, held sparsely.

    One token per image pair at camera rate, tagged with the telemetry tick it
    became available at. Storing the dense per-tick field instead would be five
    times larger for no gain - the field is rebuilt per leg by
    :func:`scatter_span_tokens`, which is cheap.
    """

    tick: np.ndarray  # (N,) telemetry index the event lands on
    token: torch.Tensor  # (N, D) float32, CPU
    quality: torch.Tensor  # (N, 1) float32, CPU
    visual_dim: int
    disabled: bool = False
    # (N, len(DIAGNOSTIC_NAMES)) float32, CPU. Never scattered onto the dense
    # per-tick field the model reads - these exist purely for a caller to log
    # and stratify the eventual error by, not to be consumed by the model, so
    # they stay in this sparse one-row-per-event form and are aggregated
    # directly against ``tick`` where they are read (see ``entry_for`` in
    # :func:`run_span_horizons`). Left ``None`` to get NaN rows shaped to
    # match ``token`` - "not measured", not a false zero - which is what
    # every construction site that predates this field, tests included,
    # already looks like.
    diagnostics: Optional[torch.Tensor] = None
    # (N, 1) float32, CPU: the frontend's per-pair verdict on whether enough of
    # the correlation grid survived the reliability gate to deliver the token
    # at all. UNLIKE ``diagnostics`` this IS scattered onto the dense field -
    # a refused pair must reach the model as an ABSENT image (present=0, age
    # still growing), never as a token delivered with present=1, because those
    # two say opposite things. Left ``None`` means "every pair delivered",
    # which is what an ungated frontend and every pre-gate construction site
    # (tests included) mean.
    pair_reliable: Optional[torch.Tensor] = None
    # (N, 3) float32, CPU: the per-pair metric velocity a flat-ground frontend
    # (PlanarFlowFrontend) measures, or None for a frontend that measures none.
    # Scattered and HELD by :func:`scatter_span_velocity` for a
    # geometric_residual model, under the same delivery rule as the token.
    velocity: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.tick.shape[0] != self.token.shape[0]:
            raise ValueError("tick and token counts disagree")
        if self.token.shape[0] != self.quality.shape[0]:
            raise ValueError("token and quality counts disagree")
        if self.diagnostics is None:
            self.diagnostics = torch.full(
                (self.token.shape[0], len(DIAGNOSTIC_NAMES)), float("nan")
            )
        elif self.token.shape[0] != self.diagnostics.shape[0]:
            raise ValueError("token and diagnostics counts disagree")
        if self.pair_reliable is None:
            self.pair_reliable = torch.ones((self.token.shape[0], 1))
        elif self.token.shape[0] != self.pair_reliable.shape[0]:
            raise ValueError("token and pair_reliable counts disagree")
        if self.velocity is not None and self.velocity.shape != (self.token.shape[0], 3):
            raise ValueError("velocity must be (N, 3), one row per token")

    @classmethod
    def empty(cls, visual_dim: int, *, disabled: bool = False) -> "SpanTokens":
        return cls(
            tick=np.zeros(0, dtype=np.int64),
            token=torch.zeros(0, int(visual_dim)),
            quality=torch.zeros(0, 1),
            visual_dim=int(visual_dim),
            disabled=disabled,
        )

    def __len__(self) -> int:
        return int(self.tick.shape[0])


class _SpanPairDataset(Dataset):
    """The image pairs of a span, one event per item, for a worker pool."""

    def __init__(
        self,
        image_source: VisualPairSource,
        events: np.ndarray,
        body_rate_rad_s: np.ndarray,
        times_s: np.ndarray,
        attitude: Optional[AttitudeAltitude] = None,
    ) -> None:
        self.image_source = image_source
        self.events = np.asarray(events, dtype=np.int64)
        self.body_rate_rad_s = body_rate_rad_s
        self.times_s = times_s
        # Given only for a frontend that needs each pair's geometry.
        self.attitude = attitude

    def __len__(self) -> int:
        return int(self.events.size)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        event = int(self.events[index])
        first, second = self.image_source.load_pair(event)
        rate = self.image_source.rate_over_exposure(
            self.body_rate_rad_s, self.times_s, event
        )
        return {
            "image0": first,
            "image1": second,
            "pair_dt_s": torch.tensor(
                float(self.image_source.plan.pair_dt_s[event]), dtype=torch.float32
            ),
            "body_rate": torch.from_numpy(rate.astype(np.float32)),
            "tick": torch.tensor(
                int(self.image_source.plan.ready_tick[event]), dtype=torch.long
            ),
            **self._geometry(event),
        }

    def _geometry(self, event: int) -> Dict[str, torch.Tensor]:
        if self.attitude is None:
            return {}
        plan = self.image_source.plan
        geometry = pair_geometry(
            self.attitude,
            float(plan.exposure_t0_s[event]),
            float(plan.exposure_t1_s[event]),
        )
        return {
            "relative_rotation": torch.from_numpy(geometry["relative_rotation"]),
            "down_body": torch.from_numpy(geometry["down_body"]),
            "altitude_m": torch.from_numpy(geometry["altitude_m"]),
        }


def _maybe_progress(iterable, description: str, enabled: bool, unit: str = "batch"):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - progress is cosmetic
        return iterable
    return tqdm(iterable, desc=description, unit=unit, dynamic_ncols=True)


@torch.no_grad()
def encode_span_tokens(
    frontend: torch.nn.Module,
    image_source: VisualPairSource,
    *,
    span: Tuple[int, int],
    body_rate_rad_s: np.ndarray,
    times_s: np.ndarray,
    visual_dim: int,
    device: torch.device,
    camera_matrix: Optional[torch.Tensor] = None,
    batch_pairs: int = 8,
    num_workers: int = 0,
    disable_visual: bool = False,
    progress: bool = False,
    attitude: Optional[AttitudeAltitude] = None,
) -> SpanTokens:
    """Run the frontend once over every image pair whose event lands in ``span``.

    This is the expensive half of a horizon evaluation and it is deliberately
    separated from the scan: the result is replayed at every horizon, so six
    horizons cost one pass over the images rather than six.

    A frontend with ``requires_pair_geometry`` (the flat-ground one) also needs
    ``attitude``, from which each pair's rotation, ground normal and altitudes
    are read exactly as the training dataset reads them, and its per-pair
    velocity is kept in :attr:`SpanTokens.velocity`.
    """

    if disable_visual:
        return SpanTokens.empty(visual_dim, disabled=True)
    # An event whose ready tick barely lands inside the span was captured one
    # whole deployment latency earlier - for a span that is one condition
    # segment out of an interleaved manifest, that earlier instant can sit in
    # a segment belonging to a DIFFERENT phase. min_capture_tick excludes it,
    # frames and all, rather than letting a held-out score be built in part
    # from images captured during training. See VisualPairSource.
    # events_in_window for the full reasoning; harmless everywhere else,
    # since every other event's reach-back stays inside its own span.
    events = image_source.events_in_window(
        int(span[0]), int(span[1]), min_capture_tick=int(span[0])
    )
    if events.size == 0:
        return SpanTokens.empty(visual_dim)

    geometric = bool(getattr(frontend, "requires_pair_geometry", False))
    if geometric and attitude is None:
        raise ValueError(
            "this frontend needs each pair's geometry: pass attitude= "
            "(the flight's AttitudeAltitude)"
        )
    loader = DataLoader(
        _SpanPairDataset(
            image_source, events, body_rate_rad_s, times_s,
            attitude=attitude if geometric else None,
        ),
        batch_size=max(int(batch_pairs), 1),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )

    was_training = frontend.training
    frontend.eval()
    tokens: List[torch.Tensor] = []
    qualities: List[torch.Tensor] = []
    reliabilities: List[torch.Tensor] = []
    ticks: List[np.ndarray] = []
    diagnostics: List[torch.Tensor] = []
    velocities: List[torch.Tensor] = []
    try:
        for batch in _maybe_progress(loader, "visual tokens", progress):
            extra = {}
            if geometric:
                extra = {
                    name: batch[name].to(device, non_blocking=True)
                    for name in ("relative_rotation", "down_body", "altitude_m")
                }
            encoded = frontend(
                batch["image0"].to(device, non_blocking=True).float().div(255.0),
                batch["image1"].to(device, non_blocking=True).float().div(255.0),
                pair_dt_s=batch["pair_dt_s"].to(device, non_blocking=True),
                body_rate_rad_s=batch["body_rate"].to(device, non_blocking=True),
                camera_matrix=camera_matrix,
                **extra,
            )
            if geometric:
                velocities.append(encoded["geometric_velocity"].detach().float().cpu())
            tokens.append(encoded["visual_token"].detach().float().cpu())
            qualities.append(encoded["visual_quality"].detach().float().cpu())
            reliabilities.append(encoded["pair_reliable"].detach().float().cpu())
            ticks.append(batch["tick"].numpy())
            diagnostics.append(
                torch.stack(
                    [encoded["diagnostics"][name] for name in DIAGNOSTIC_NAMES], dim=-1
                ).detach().float().cpu()
            )
    finally:
        frontend.train(was_training)

    return SpanTokens(
        tick=np.concatenate(ticks).astype(np.int64),
        token=torch.cat(tokens),
        quality=torch.cat(qualities),
        diagnostics=torch.cat(diagnostics),
        pair_reliable=torch.cat(reliabilities),
        visual_dim=int(visual_dim),
        velocity=torch.cat(velocities) if velocities else None,
    )


def scatter_span_tokens(
    tokens: SpanTokens,
    bounds: Sequence[Tuple[int, int]],
    ticks: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense ``(C, ticks, D)`` token / quality / presence fields for C legs.

    Same placement rule as training: an event sits on the tick it became
    available at, and a tick with no event carries a zero token and a zero
    presence bit - which the model reads as "no image", not as "no motion".

    A pair the frontend's reliability gate REFUSED is treated exactly as a tick
    with no event: its presence bit stays zero and its token stays zero, so
    visual_age keeps growing from the last pair that was trusted. This has to
    match ``VOStep._encode_visual`` in the trainer or a gated model would be
    scored under a frontend contract it was never trained on - delivering a
    refused pair with present=1 is the precise failure the gate exists to
    prevent, and it would be invisible in every metric.
    """

    count = len(bounds)
    field = torch.zeros(count, ticks, tokens.visual_dim, device=device, dtype=dtype)
    quality = torch.zeros(count, ticks, 1, device=device, dtype=dtype)
    present = torch.zeros(count, ticks, 1, device=device, dtype=dtype)
    if len(tokens) == 0:
        return field, quality, present
    # A refused pair is dropped here rather than scattered and then masked:
    # placing it would overwrite the tick's zero token, and a later reader
    # cannot tell a deliberately-zeroed token from a measured one.
    reliable = tokens.pair_reliable
    keep_all = np.flatnonzero(
        (reliable.reshape(-1) > 0).cpu().numpy()
        if reliable is not None
        else np.ones(len(tokens), dtype=bool)
    )
    for row, (start, end) in enumerate(bounds):
        inside = np.flatnonzero((tokens.tick >= start) & (tokens.tick < end))
        inside = np.intersect1d(inside, keep_all, assume_unique=False)
        if inside.size == 0:
            continue
        offset = torch.from_numpy((tokens.tick[inside] - start).astype(np.int64)).to(
            device
        )
        field[row, offset] = tokens.token[inside].to(device=device, dtype=dtype)
        quality[row, offset] = tokens.quality[inside].to(device=device, dtype=dtype)
        present[row, offset] = 1.0
    return field, quality, present


def scatter_span_velocity(
    tokens: SpanTokens,
    bounds: Sequence[Tuple[int, int]],
    ticks: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Held per-pair velocity over C legs: ``(C, ticks, 3)`` and ``(C, ticks, 1)``.

    The same placement and refusal rule as :func:`scatter_span_tokens` - a
    delivered pair lands on its ready tick, a refused one does not land at
    all - and then each value is HELD until the next delivered pair, exactly
    as :func:`vio.data.fixedwing_vo.hold_visual_velocity` holds it in
    training. Each leg starts with nothing held: a leg is a cold start. The
    validity field is 0 before a leg's first delivery and 1 after.
    """

    count = len(bounds)
    held = torch.zeros(count, ticks, 3, device=device, dtype=dtype)
    held_valid = torch.zeros(count, ticks, 1, device=device, dtype=dtype)
    if len(tokens) == 0 or tokens.velocity is None:
        return held, held_valid
    reliable = (
        tokens.pair_reliable.reshape(-1) > 0
        if tokens.pair_reliable is not None
        else torch.ones(len(tokens), dtype=torch.bool)
    )
    for row, (start, end) in enumerate(bounds):
        inside = np.flatnonzero((tokens.tick >= start) & (tokens.tick < end))
        if inside.size == 0:
            continue
        # The training-time hold itself, so a same-tick collision or a refused
        # pair is resolved identically in training and here.
        leg, leg_valid, _ = hold_visual_velocity(
            tokens.velocity[inside].to(device=device, dtype=dtype).unsqueeze(0),
            torch.from_numpy((tokens.tick[inside] - start).astype(np.int64)).to(device).unsqueeze(0),
            torch.ones(1, inside.size, device=device),
            window_length=ticks,
            delivered=reliable[torch.from_numpy(inside)].to(device=device, dtype=dtype).unsqueeze(0),
        )
        held[row] = leg[0]
        held_valid[row] = leg_valid[0]
    return held, held_valid


# ---------------------------------------------------------------------------
# per-leg error accumulation
# ---------------------------------------------------------------------------


class _LegStats:
    """Sums per leg, so each leg's own RMS survives the block-wise streaming.

    ``collect=True`` additionally retains every tick's error magnitude and
    direction angle (masked ticks as NaN), for :meth:`series` - the raw curve
    a plot needs. Off by default: the running sums above are all a training
    epoch's horizon pass needs, and holding a full per-tick record for every
    block would cost memory nothing here uses otherwise.
    """

    def __init__(self, count: int, device: torch.device, *, collect: bool = False) -> None:
        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, dtype=torch.float64, device=device)

        self.sq_sum = zeros(count)
        self.count = zeros(count)
        self.max_error = zeros(count)
        self.dir_sq_sum = zeros(count)
        self.dir_count = zeros(count)
        self.dir_max = zeros(count)
        self.axis_sq = zeros(count, 3)
        # WHERE each maximum happened, as a within-leg tick index. A worst
        # case is far more actionable with a timestamp attached: it says which
        # second of the flight to go and look at, and whether the model's worst
        # moment is a turn, a climb, or one corrupted frame.
        self.max_error_tick = zeros(count)
        self.dir_max_tick = zeros(count)
        self.collect = bool(collect)
        self._magnitude_chunks: List[torch.Tensor] = []
        self._angle_chunks: List[torch.Tensor] = []
        # The velocities themselves, not just how far apart they are. An error
        # curve says a component is 2 m/s wrong; only the two signals side by
        # side say whether the estimate lags the truth, scales it, or sits at
        # a constant offset - three different faults with the same RMSE.
        self._predicted_chunks: List[torch.Tensor] = []
        self._target_chunks: List[torch.Tensor] = []

    def update(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        tick_offset: int = 0,
    ) -> None:
        predicted = predicted.double()
        target = target.double()
        finite = torch.isfinite(predicted).all(-1) & torch.isfinite(target).all(-1)
        valid = (mask > 0) & finite
        weight = valid.double()
        residual = predicted - target
        magnitude = torch.linalg.vector_norm(residual, dim=-1)
        self.sq_sum += (magnitude.square() * weight).sum(dim=1)
        self.count += weight.sum(dim=1)
        block_max, block_where = (magnitude * weight).max(dim=1)
        improved = block_max > self.max_error
        self.max_error = torch.where(improved, block_max, self.max_error)
        self.max_error_tick = torch.where(
            improved, (block_where + tick_offset).double(), self.max_error_tick
        )
        self.axis_sq += (residual.square() * weight.unsqueeze(-1)).sum(dim=1)
        predicted_norm = torch.linalg.vector_norm(predicted, dim=-1)
        target_norm = torch.linalg.vector_norm(target, dim=-1)
        # A vector shorter than eps has no meaningful direction; counting it
        # would fold an arbitrary angle into the mean.
        usable = valid & (predicted_norm > _EPS) & (target_norm > _EPS)
        cosine = (predicted * target).sum(-1) / (predicted_norm * target_norm).clamp_min(
            _EPS
        )
        angles = torch.rad2deg(torch.arccos(cosine.clamp(-1.0, 1.0)))
        angle_weight = usable.double()
        self.dir_sq_sum += (angles.square() * angle_weight).sum(dim=1)
        self.dir_count += angle_weight.sum(dim=1)
        dir_block_max, dir_block_where = (angles * angle_weight).max(dim=1)
        dir_improved = dir_block_max > self.dir_max
        self.dir_max = torch.where(dir_improved, dir_block_max, self.dir_max)
        self.dir_max_tick = torch.where(
            dir_improved, (dir_block_where + tick_offset).double(), self.dir_max_tick
        )

        if self.collect:
            nan = torch.full_like(magnitude, float("nan"))
            self._magnitude_chunks.append(torch.where(valid, magnitude, nan).cpu())
            self._angle_chunks.append(torch.where(usable, angles, nan).cpu())
            axis_valid = valid.unsqueeze(-1).expand_as(predicted)
            axis_nan = torch.full_like(predicted, float("nan"))
            self._predicted_chunks.append(
                torch.where(axis_valid, predicted, axis_nan).cpu()
            )
            self._target_chunks.append(torch.where(axis_valid, target, axis_nan).cpu())

    def series(self) -> Dict[str, np.ndarray]:
        """Per-tick ``(legs, ticks)`` curves, NaN wherever a tick was unscored.

        Only valid when constructed with ``collect=True`` - otherwise the
        chunks this concatenates were never kept.
        """

        if not self.collect:
            raise RuntimeError("_LegStats was not constructed with collect=True")
        return {
            "vel_error_m_s": torch.cat(self._magnitude_chunks, dim=1).numpy(),
            "vel_dir_error_deg": torch.cat(self._angle_chunks, dim=1).numpy(),
            "vel_predicted_body": torch.cat(self._predicted_chunks, dim=1).numpy(),
            "vel_target_body": torch.cat(self._target_chunks, dim=1).numpy(),
        }

    def per_leg(self) -> Dict[str, np.ndarray]:
        count = self.count.clamp_min(1.0)
        dir_count = self.dir_count.clamp_min(1.0)
        scored = self.count > 0
        directed = self.dir_count > 0
        nan = float("nan")
        rmse = torch.where(
            scored, (self.sq_sum / count).sqrt(), torch.full_like(count, nan)
        )
        dir_rmse = torch.where(
            directed, (self.dir_sq_sum / dir_count).sqrt(), torch.full_like(count, nan)
        )
        axis = (self.axis_sq / count.unsqueeze(-1)).sqrt()
        axis = torch.where(scored.unsqueeze(-1), axis, torch.full_like(axis, nan))
        return {
            "vel_rmse": rmse.cpu().numpy(),
            "vel_dir_rmse": dir_rmse.cpu().numpy(),
            "vel_max_error": self.max_error.cpu().numpy(),
            "vel_dir_max_error": self.dir_max.cpu().numpy(),
            "vel_max_error_tick": self.max_error_tick.cpu().numpy(),
            "vel_dir_max_error_tick": self.dir_max_tick.cpu().numpy(),
            "axis_rmse": axis.cpu().numpy(),
            "scored_ticks": self.count.cpu().numpy(),
        }

    def summary(self) -> Dict[str, float]:
        legs = self.per_leg()
        rmse = legs["vel_rmse"]
        dir_rmse = legs["vel_dir_rmse"]
        total = float(self.count.sum().item())
        dir_total = float(self.dir_count.sum().item())
        nan = float("nan")
        values: Dict[str, float] = {
            # The headline: each leg scored on its own, then averaged. This is
            # what the horizon is for - it weights legs equally, so one bad leg
            # cannot hide behind several good ones of a different length.
            "vel_rmse": float(np.nanmean(rmse)) if rmse.size else nan,
            "vel_dir_rmse": float(np.nanmean(dir_rmse)) if dir_rmse.size else nan,
            # Spread across legs. A small mean with a large spread is a model
            # that fails on some conditions, not one that works.
            "vel_rmse_std": float(np.nanstd(rmse)) if rmse.size > 1 else 0.0,
            "vel_dir_rmse_std": float(np.nanstd(dir_rmse)) if dir_rmse.size > 1 else 0.0,
            "vel_rmse_worst_leg": float(np.nanmax(rmse)) if rmse.size else nan,
            "vel_dir_rmse_worst_leg": float(np.nanmax(dir_rmse)) if dir_rmse.size else nan,
            # Every tick in one RMS, ignoring leg boundaries. Differs from the
            # mean above whenever the legs differ in quality; reported so the
            # two cannot be confused for one another.
            "vel_rmse_pooled": (
                float(np.sqrt(float(self.sq_sum.sum().item()) / total)) if total else nan
            ),
            "vel_dir_rmse_pooled": (
                float(np.sqrt(float(self.dir_sq_sum.sum().item()) / dir_total))
                if dir_total
                else nan
            ),
            "vel_max_error": float(legs["vel_max_error"].max()) if rmse.size else nan,
            "vel_dir_max_error": (
                float(legs["vel_dir_max_error"].max()) if rmse.size else nan
            ),
            "scored_ticks": total,
        }
        # Which leg the overall worst tick sits in, and where inside it. The
        # caller turns the tick into a time and an absolute telemetry index -
        # it owns the clock, this only owns the search.
        if rmse.size:
            worst = int(np.argmax(legs["vel_max_error"]))
            values["vel_max_error_leg"] = worst
            values["vel_max_error_leg_tick"] = int(legs["vel_max_error_tick"][worst])
            dir_worst = int(np.argmax(legs["vel_dir_max_error"]))
            values["vel_dir_max_error_leg"] = dir_worst
            values["vel_dir_max_error_leg_tick"] = int(
                legs["vel_dir_max_error_tick"][dir_worst]
            )
        axis = legs["axis_rmse"]
        for index, name in enumerate(AXIS_NAMES):
            values[f"vel_rmse_{name}"] = (
                float(np.nanmean(axis[:, index])) if axis.size else nan
            )
        return values


class _PositionStats:
    """Dead-reckoning drift per leg, integrated across the streamed blocks.

    The quantity is the integral of the VELOCITY RESIDUAL, rotated into NED:

        e(t) = integral over [t0, t] of  R(tau) . (v_pred(tau) - v_true(tau))

    which is exactly the difference between integrating the prediction and
    integrating the truth through the same attitude. Using one rotation for
    both is deliberate: it makes the result the position error attributable to
    the VELOCITY estimate alone. A deployed system also carries attitude error,
    and this excludes it by construction - so this is a dead-reckoning drift
    figure for the velocity product, not a full VIO trajectory error.

    Integration starts at the end of the warm-up, where the leg's state reset
    has been paid for and the estimator is first considered live, and the
    origin is that point - so ``final`` is "how far off after H minutes".

    ``collect=True`` additionally integrates the TRUE velocity through the
    same rotation, giving the actual flight path (``reference_track``) rather
    than just the error; adding the two integrals back together
    (``reference_track + residual_track``) recovers the dead-reckoned
    predicted path. Both are retained per tick for :meth:`series`, which is
    what a trajectory plot needs and a scalar summary does not - off by
    default for exactly that reason.
    """

    def __init__(self, count: int, device: torch.device, *, collect: bool = False) -> None:
        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, dtype=torch.float64, device=device)

        self.position = zeros(count, 3)  # running integral of the residual
        self.previous = zeros(count, 3)  # last tick's rotated residual
        self.previous_speed = zeros(count)  # last tick's true ground speed
        self.previous_time = zeros(count)
        self.path_length = zeros(count)
        self.started = torch.zeros(count, dtype=torch.bool, device=device)
        self.broken = torch.zeros(count, dtype=torch.bool, device=device)
        self.sq_sum = zeros(count)
        self.count = zeros(count)
        self.max_error = zeros(count)
        self.max_error_tick = zeros(count)
        self.collect = bool(collect)
        self.reference_running = zeros(count, 3)  # integral of TRUE velocity only
        self.previous_reference = zeros(count, 3)
        self._error_chunks: List[torch.Tensor] = []
        self._predicted_chunks: List[torch.Tensor] = []
        self._reference_chunks: List[torch.Tensor] = []

    def update(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        rotation: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        tick_offset: int = 0,
    ) -> None:
        """One block. ``rotation`` is (C, L, 3, 3) body->NED, ``times`` (C, L).

        The trapezoid runs as a cumulative sum rather than a Python loop over
        ticks: a 30-minute leg is 180,000 of them, and stepping those one at a
        time in Python would cost more than the model forward it is measuring.
        """

        residual = (predicted - target).double()
        # Rotate the residual, not the two velocities separately: rotation is
        # linear, so the result is identical and the subtraction stays exact.
        rotated = torch.einsum("clij,clj->cli", rotation.double(), residual)
        speed = torch.linalg.vector_norm(target.double(), dim=-1)
        live = mask > 0
        # An integral cannot skip a hole: once a leg produces a non-finite
        # value the distance travelled after it is undefined, so the leg is
        # marked and reported as NaN rather than quietly resumed.
        self.broken |= (live & ~torch.isfinite(rotated).all(-1)).any(dim=1)
        rotated = torch.nan_to_num(rotated)

        # Pair each tick with its predecessor, the first taking the value
        # carried in from the previous block.
        previous_rotated = torch.cat(
            (self.previous.unsqueeze(1), rotated[:, :-1]), dim=1
        )
        previous_speed = torch.cat(
            (self.previous_speed.unsqueeze(1), speed[:, :-1]), dim=1
        )
        clock = times.double()
        previous_time = torch.cat(
            (self.previous_time.unsqueeze(1), clock[:, :-1]), dim=1
        )
        was_live = torch.cat((self.started.unsqueeze(1), live[:, :-1]), dim=1)
        # A step contributes only when both of its endpoints are live, so the
        # first live tick of a leg is its origin: zero error, zero path.
        span = torch.where(live & was_live, clock - previous_time, torch.zeros_like(clock))

        increments = 0.5 * (rotated + previous_rotated) * span.unsqueeze(-1)
        track = self.position.unsqueeze(1) + torch.cumsum(increments, dim=1)
        distance = 0.5 * (speed + previous_speed) * span
        path = self.path_length.unsqueeze(1) + torch.cumsum(distance, dim=1)

        magnitude = torch.linalg.vector_norm(track, dim=-1)
        counted = live.double()
        self.sq_sum += (magnitude.square() * counted).sum(dim=1)
        self.count += counted.sum(dim=1)
        block_max, block_where = (magnitude * counted).max(dim=1)
        improved = block_max > self.max_error
        self.max_error = torch.where(improved, block_max, self.max_error)
        self.max_error_tick = torch.where(
            improved, (block_where + tick_offset).double(), self.max_error_tick
        )

        if self.collect:
            # A second integral, of the TRUE velocity alone, reusing the same
            # ``span`` weights: rotation and the live/warm-up mask do not
            # depend on which quantity is being integrated. Adding it back to
            # the residual integral (``track``) recovers the dead-reckoned
            # predicted path, so this is the only extra work needed for both
            # trajectory lines a plot wants.
            rotated_target = torch.einsum(
                "clij,clj->cli", rotation.double(), target.double()
            )
            rotated_target = torch.nan_to_num(rotated_target)
            previous_reference = torch.cat(
                (self.previous_reference.unsqueeze(1), rotated_target[:, :-1]), dim=1
            )
            reference_increments = (
                0.5 * (rotated_target + previous_reference) * span.unsqueeze(-1)
            )
            reference_track = (
                self.reference_running.unsqueeze(1)
                + torch.cumsum(reference_increments, dim=1)
            )
            predicted_track = reference_track + track
            nan = torch.full_like(magnitude, float("nan"))
            scored = live.bool()
            self._error_chunks.append(torch.where(scored, magnitude, nan).cpu())
            self._predicted_chunks.append(predicted_track.cpu())
            self._reference_chunks.append(reference_track.cpu())
            held_now = live[:, -1]
            self.reference_running = reference_track[:, -1]
            self.previous_reference = torch.where(
                held_now.unsqueeze(-1), rotated_target[:, -1], self.previous_reference
            )

        # Carry the running totals. A block with no live tick leaves them
        # untouched, because every increment in it was zero.
        self.position = track[:, -1]
        self.path_length = path[:, -1]
        held = live[:, -1]
        self.previous = torch.where(held.unsqueeze(-1), rotated[:, -1], self.previous)
        self.previous_speed = torch.where(held, speed[:, -1], self.previous_speed)
        self.previous_time = torch.where(held, clock[:, -1], self.previous_time)
        self.started = self.started | live.any(dim=1)

    def series(self) -> Dict[str, np.ndarray]:
        """Per-tick trajectories and drift magnitude, in NED metres.

        ``trajectory_predicted_ned`` is the dead-reckoned path from the
        estimated velocity; ``trajectory_reference_ned`` the true path from the
        logged one; both start at the origin at the leg's first live tick.
        ``leg_valid`` is False for a leg :attr:`broken` marked - a non-finite
        prediction makes the distance travelled after it undefined, so that
        leg's curve should not be drawn rather than drawn wrong.
        """

        if not self.collect:
            raise RuntimeError("_PositionStats was not constructed with collect=True")
        return {
            "pos_error_m": torch.cat(self._error_chunks, dim=1).numpy(),
            "trajectory_predicted_ned": torch.cat(self._predicted_chunks, dim=1).numpy(),
            "trajectory_reference_ned": torch.cat(self._reference_chunks, dim=1).numpy(),
            "leg_valid": (~self.broken).cpu().numpy(),
        }

    def per_leg(self) -> Dict[str, np.ndarray]:
        count = self.count.clamp_min(1.0)
        scored = (self.count > 0) & ~self.broken
        nan = float("nan")
        final = torch.linalg.vector_norm(self.position, dim=-1)
        final = torch.where(scored, final, torch.full_like(final, nan))
        rmse = torch.where(
            scored, (self.sq_sum / count).sqrt(), torch.full_like(count, nan)
        )
        path = torch.where(scored, self.path_length, torch.full_like(count, nan))
        drift = torch.where(
            scored & (self.path_length > 0.0),
            100.0 * final / self.path_length.clamp_min(1e-9),
            torch.full_like(path, nan),
        )
        maximum = torch.where(
            scored, self.max_error, torch.full_like(count, nan)
        )
        return {
            "pos_error_final": final.cpu().numpy(),
            "pos_error_rmse": rmse.cpu().numpy(),
            "pos_error_max": maximum.cpu().numpy(),
            "pos_error_max_tick": self.max_error_tick.cpu().numpy(),
            "pos_drift_percent": drift.cpu().numpy(),
            "path_length_m": path.cpu().numpy(),
        }

    def summary(self) -> Dict[str, float]:
        legs = self.per_leg()
        nan = float("nan")
        size = legs["pos_error_final"].size
        values = {
            "pos_error_final": (
                float(np.nanmean(legs["pos_error_final"])) if size else nan
            ),
            "pos_error_final_worst_leg": (
                float(np.nanmax(legs["pos_error_final"])) if size else nan
            ),
            "pos_error_final_std": (
                float(np.nanstd(legs["pos_error_final"])) if size > 1 else 0.0
            ),
            "pos_error_rmse": (
                float(np.nanmean(legs["pos_error_rmse"])) if size else nan
            ),
            # A maximum over legs, never a mean of maxima.
            "pos_error_max": float(np.nanmax(legs["pos_error_max"])) if size else nan,
            "pos_drift_percent": (
                float(np.nanmean(legs["pos_drift_percent"])) if size else nan
            ),
            "path_length_m": (
                float(np.nanmean(legs["path_length_m"])) if size else nan
            ),
        }
        if size:
            worst = int(np.nanargmax(legs["pos_error_max"]))
            values["pos_error_max_leg"] = worst
            values["pos_error_max_leg_tick"] = int(legs["pos_error_max_tick"][worst])
        return values


# ---------------------------------------------------------------------------
# the evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_span_horizons(
    model: VisionMambaVO,
    *,
    aiding: np.ndarray,
    log_altitude: np.ndarray,
    target_velocity: np.ndarray,
    times_s: np.ndarray,
    span: Tuple[int, int],
    tokens: SpanTokens,
    deployment_latency_s: float,
    horizons_minutes: Sequence[float] = DEFAULT_HORIZON_MINUTES,
    device: Optional[torch.device] = None,
    warmup_ticks: int = 20,
    block_ticks: int = 2000,
    baseline: Optional[np.ndarray] = None,
    rotation_body_to_ned: Optional[np.ndarray] = None,
    collect_series: bool = False,
    progress: bool = False,
    ablate_body_rate: bool = False,
    ablate_visual_age: bool = False,
    output_on_pairs: bool = False,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    """Stream ``span`` once, start to end, and score every horizon prefix.

    ``output_on_pairs`` scores a model trained with ``--output-on-pairs`` the
    way it is deployed: its velocity is an output only on the tick an image
    pair is delivered, so the velocity metrics count only those ticks, and
    the dead-reckoned position integrates that output HELD until the next
    one - what a consumer of one velocity per pair actually integrates.

    Returns ``(horizons, whole_span)``. ``horizons`` has one entry per label
    (``"h5m"``), each covering the FIRST H minutes of the span and nothing
    else; ``whole_span`` is the same shape for the entire span, which is what
    the trajectory and error figures are drawn from.

    The state is zeroed once, at the span's first tick, and carried unbroken
    to the last one - there is no reset inside the run. A horizon is a mask
    over that run, not another forward, so every horizon and the whole-span
    entry come out of a single pass through the model.

    A horizon longer than the span is returned with ``fits: False`` and a
    ``skipped`` reason rather than omitted, so a caller can report the gap
    instead of silently dropping a column.

    Supplying ``rotation_body_to_ned`` (the (N, 3, 3) reference rotation from
    :func:`vio.data.fixedwing_vo.reference_body_frame`) adds the dead-reckoning
    position metrics - see :class:`_PositionStats` for exactly what they are
    and, more importantly, what they are not.

    ``collect_series=True`` retains every tick's error curve and, with a
    rotation supplied, both trajectory lines - what
    :mod:`vio.utils.horizon_plots` draws - under ``whole_span["series"]``.
    Those arrays are plain numpy, not JSON-serialisable as the rest of the
    entry is meant to be, so a caller that writes this dict to disk must pop
    "series" out first.

    ``deployment_latency_s`` is required, not defaulted: it is a physical
    constant of the run being scored (see
    :func:`vio.data.fixedwing_vo.visual_age_seconds`), and a wrong silent
    default would score a real checkpoint under the wrong staleness
    everywhere visual_age reaches the model. ``ablate_body_rate`` /
    ``ablate_visual_age`` zero those channels the same way
    ``tools/train_fixedwing_vo.py``'s ``VOStep`` does, so a checkpoint
    trained under an ablation is evaluated under the SAME one rather than by
    coincidence.
    """

    device = device or next(model.parameters()).device
    start, end = int(span[0]), int(span[1])
    span_ticks = end - start
    if span_ticks < 2:
        raise ValueError("A span needs at least two telemetry samples")
    times = np.asarray(times_s, dtype=np.float64)
    span_minutes = float(times[end - 1] - times[start]) / 60.0
    interval = float(np.median(np.diff(times[start:end])))
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("Telemetry clock has no usable positive interval")

    # Which horizons fit, in the order they were asked for. The whole span is
    # scored alongside them under its own key, never mixed into the table.
    plans: List[Tuple[str, float, int]] = []
    results: Dict[str, Dict[str, object]] = {}
    for minutes in horizons_minutes:
        label = horizon_label(minutes)
        ticks = ticks_per_horizon(times, span, minutes)
        if prefix_leg(span, ticks) is None:
            results[label] = {
                "minutes_requested": float(minutes),
                "fits": False,
                "ticks": int(ticks),
                "skipped": (
                    f"needs {minutes:g} min ({ticks} ticks) from the start of "
                    f"the split, which holds {span_minutes:.1f} min "
                    f"({span_ticks} ticks)"
                ),
            }
            continue
        if warmup_ticks >= ticks:
            results[label] = {
                "minutes_requested": float(minutes),
                "fits": False,
                "ticks": int(ticks),
                "skipped": (
                    f"warm-up of {warmup_ticks} ticks covers the whole "
                    f"{ticks}-tick prefix"
                ),
            }
            continue
        plans.append((label, float(minutes), int(ticks)))

    # One leg, the whole span: (1, span_ticks, ...) throughout.
    index = torch.arange(start, end).long().unsqueeze(0)
    leg_aiding = torch.from_numpy(np.ascontiguousarray(aiding, dtype=np.float32))[
        index
    ].to(device)
    if ablate_body_rate:
        leg_aiding = leg_aiding.clone()
        leg_aiding[..., BODY_RATE_AIDING_SLICE] = 0.0
    leg_altitude = torch.from_numpy(
        np.ascontiguousarray(log_altitude, dtype=np.float32)
    )[index].to(device)
    leg_target = torch.from_numpy(
        np.ascontiguousarray(target_velocity, dtype=np.float32)
    )[index].to(device)
    leg_clock = torch.from_numpy(np.ascontiguousarray(times, dtype=np.float64))[
        index
    ].to(device)
    leg_rotation = (
        None
        if rotation_body_to_ned is None
        else torch.from_numpy(
            np.ascontiguousarray(rotation_body_to_ned, dtype=np.float32)
        )[index].to(device)
    )
    token_field, quality_field, present_field = scatter_span_tokens(
        tokens, [(start, end)], span_ticks, device=device
    )
    # Computed once over the whole span, then sliced per block below - same
    # rule as token_field/quality_field/present_field, and for the same
    # reason: age must stay continuous across a block boundary, which slicing
    # a single precomputed field guarantees and recomputing per block would
    # not (a fresh call would forget any event before that block's start).
    age_field, _ = visual_age_seconds(present_field, leg_clock, deployment_latency_s)
    if ablate_visual_age:
        age_field = torch.zeros_like(age_field)
    # A geometric_residual model reads the held per-pair velocity too - built
    # once over the whole span and sliced per block, like age and for the
    # same reason: the hold must survive a block boundary.
    geometric = getattr(model, "velocity_mode", "heads") == "geometric_residual"
    velocity_field = velocity_valid_field = None
    if geometric:
        velocity_field, velocity_valid_field = scatter_span_velocity(
            tokens, [(start, end)], span_ticks, device=device
        )

    mask = torch.ones(1, span_ticks, device=device)
    # The first ticks after the reset have no visual event yet - the camera
    # runs at 20 Hz behind a deployment latency - so the model is blind there
    # by construction, exactly as it is in training.
    mask[:, : int(warmup_ticks)] = 0.0

    constant = (
        None
        if baseline is None
        else torch.from_numpy(
            np.asarray(baseline, dtype=np.float32).reshape(1, 1, 3)
        ).to(device)
    )

    def make_accumulators(collect: bool):
        return (
            _LegStats(1, device, collect=collect),
            None
            if leg_rotation is None
            else _PositionStats(1, device, collect=collect),
            _LegStats(1, device) if constant is not None else None,
        )

    # One accumulator set per fitting horizon, plus one for the whole span.
    # They all watch the same forward pass; only their masks differ.
    buckets = [(ticks, *make_accumulators(False)) for _, _, ticks in plans]
    span_stats, span_drift, span_reference = make_accumulators(collect_series)

    was_training = model.training
    model.eval()
    try:
        state: Optional[VOStreamState] = None
        # The last emitted output, carried across blocks (--output-on-pairs).
        held_output: Optional[torch.Tensor] = None
        step = max(int(block_ticks), 1)
        blocks = list(range(0, span_ticks, step))
        for begin in _maybe_progress(
            blocks, f"span {span_minutes:.1f} min", progress, unit="block"
        ):
            stop = min(begin + step, span_ticks)
            outputs, state = model.forward_stream(
                leg_aiding[:, begin:stop],
                token_field[:, begin:stop],
                present_field[:, begin:stop],
                age_field[:, begin:stop],
                visual_quality=quality_field[:, begin:stop],
                log_altitude=leg_altitude[:, begin:stop],
                state=state,
                visual_velocity=(
                    None if velocity_field is None else velocity_field[:, begin:stop]
                ),
                visual_velocity_valid=(
                    None if velocity_valid_field is None
                    else velocity_valid_field[:, begin:stop]
                ),
            )
            predicted = outputs["predicted_velocity"]
            block_target = leg_target[:, begin:stop]
            block_mask = mask[:, begin:stop]
            score_mask = block_mask
            drift_predicted = predicted
            if output_on_pairs:
                fired = present_field[:, begin:stop, 0] > 0
                score_mask = block_mask * fired.to(block_mask.dtype)
                drift_predicted, held_output = _hold_emitted(predicted, fired, held_output)
            block_rotation = (
                None if leg_rotation is None else leg_rotation[:, begin:stop]
            )
            block_clock = leg_clock[:, begin:stop]
            tick_index = torch.arange(begin, stop, device=device).unsqueeze(0)

            def feed(stats, drift, reference, prefix) -> None:
                scored = score_mask * prefix
                stats.update(predicted, block_target, scored, begin)
                if drift is not None:
                    drift.update(
                        drift_predicted,
                        block_target,
                        block_rotation,
                        block_clock,
                        block_mask * prefix,
                        begin,
                    )
                if reference is not None:
                    reference.update(
                        constant.expand_as(block_target), block_target, scored
                    )

            feed(span_stats, span_drift, span_reference, torch.ones_like(block_mask))
            for ticks, stats, drift, reference in buckets:
                # A prefix that ended before this block has nothing to add.
                if begin >= ticks:
                    continue
                feed(
                    stats,
                    drift,
                    reference,
                    (tick_index < ticks).to(block_mask.dtype),
                )
    finally:
        model.train(was_training)

    def entry_for(
        minutes_requested: float, ticks: int, stats, drift, reference, series: bool
    ) -> Dict[str, object]:
        legs = stats.per_leg()
        events_in_range = (tokens.tick >= start) & (tokens.tick < start + ticks)
        entry: Dict[str, object] = {
            "minutes_requested": float(minutes_requested),
            "minutes_realised": float(times[start + ticks - 1] - times[start]) / 60.0,
            "fits": True,
            "ticks": int(ticks),
            "warmup_ticks": int(warmup_ticks),
            "range": [int(start), int(start + ticks)],
            "tick_interval_s": interval,
            "visual_events": int(events_in_range.sum()),
            "scored_ticks": int(legs["scored_ticks"][0]),
            "vel_rmse": float(legs["vel_rmse"][0]),
            "vel_dir_rmse": float(legs["vel_dir_rmse"][0]),
            "vel_max_error": float(legs["vel_max_error"][0]),
            "vel_dir_max_error": float(legs["vel_dir_max_error"][0]),
        }
        for axis, name in enumerate(AXIS_NAMES):
            entry[f"vel_rmse_{name}"] = float(legs["axis_rmse"][0, axis])
        # Frontend diagnostics, meaned over exactly the visual events this
        # horizon's own prefix covers - the same window ``visual_events``
        # above counts. NaN rather than 0 when nothing landed in range: a
        # horizon with no visual events at all should not report a false
        # "boundary hit fraction of zero", which would look identical to a
        # horizon that saw plenty of events and none of them were ambiguous.
        if events_in_range.any():
            diag_mask = torch.from_numpy(events_in_range)
            for index, name in enumerate(DIAGNOSTIC_NAMES):
                entry[f"diag_{name}"] = float(
                    tokens.diagnostics[diag_mask, index].mean()
                )
        else:
            for name in DIAGNOSTIC_NAMES:
                entry[f"diag_{name}"] = float("nan")
        timed: List[Tuple[str, float]] = [
            ("vel_max_error", legs["vel_max_error_tick"][0]),
            ("vel_dir_max_error", legs["vel_dir_max_error_tick"][0]),
        ]
        if drift is not None:
            position = drift.per_leg()
            entry.update(
                {
                    "pos_error_final": float(position["pos_error_final"][0]),
                    "pos_error_rmse": float(position["pos_error_rmse"][0]),
                    "pos_error_max": float(position["pos_error_max"][0]),
                    "pos_drift_percent": float(position["pos_drift_percent"][0]),
                    "path_length_m": float(position["path_length_m"][0]),
                }
            )
            timed.append(("pos_error_max", position["pos_error_max_tick"][0]))
        # Every maximum carries WHEN it happened, in the two forms anyone
        # actually wants: seconds since the split's start - which is also the
        # start of every prefix - and the absolute telemetry index to go and
        # look at in the flight log.
        for stem, tick_value in timed:
            within = int(tick_value)
            entry[f"{stem}_tick"] = within
            entry[f"{stem}_time_s"] = within * interval
            entry[f"{stem}_flight_tick"] = int(start + within)
            entry[f"{stem}_flight_time_s"] = float(times[start + within])
        if reference is not None:
            floor = reference.per_leg()
            entry["baseline_vel_rmse"] = float(floor["vel_rmse"][0])
            entry["baseline_vel_dir_rmse"] = float(floor["vel_dir_rmse"][0])
        if series:
            entry["series"] = {
                "time_since_start_s": np.arange(ticks, dtype=np.float64) * interval,
                **stats.series(),
                **({} if drift is None else drift.series()),
            }
        return entry

    for (label, minutes, ticks), bucket in zip(plans, buckets):
        results[label] = entry_for(minutes, ticks, *bucket[1:], False)

    # Restore the requested order. A skipped horizon is recorded in the first
    # pass and a fitted one only after the encoding pass, so insertion order is
    # every skip followed by every fit - which puts h40m above h0.5m and breaks
    # the one thing the table is for: reading a column downwards as the horizon
    # grows.
    results = {
        label: results[label]
        for label in (horizon_label(minutes) for minutes in horizons_minutes)
        if label in results
    }

    whole_span = entry_for(
        span_minutes, span_ticks, span_stats, span_drift, span_reference, collect_series
    )
    whole_span["whole_span_minutes"] = span_minutes
    return results, whole_span


def _hold_emitted(
    predicted: torch.Tensor, fired: torch.Tensor, carried: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Hold each emitted output ``(C, L, 3)`` until the next one fires.

    Ticks before the first emission (in this block and, via ``carried``, in
    every earlier one) keep the model's own per-tick value. Returns the held
    series and the output to carry into the next block.
    """

    count, length, _ = predicted.shape
    index = torch.arange(length, device=predicted.device).expand(count, length)
    latest, _ = torch.cummax(torch.where(fired, index, torch.full_like(index, -1)), dim=1)
    held = predicted.gather(1, latest.clamp_min(0).unsqueeze(-1).expand(count, length, 3))
    before = (latest < 0).unsqueeze(-1)
    if carried is not None:
        fallback = carried.unsqueeze(1).expand(count, length, 3)
    else:
        fallback = predicted
    held = torch.where(before, fallback, held)
    if carried is None and not bool((latest[:, -1] >= 0).any()):
        return held, None
    return held, held[:, -1]


def stream_horizon_metrics(
    model: VisionMambaVO, **kwargs
) -> Dict[str, Dict[str, object]]:
    """:func:`run_span_horizons` when only the horizon table is wanted."""

    kwargs.pop("collect_series", None)
    horizons, _ = run_span_horizons(model, **kwargs)
    return horizons


def whole_span_series(model: VisionMambaVO, **kwargs) -> Dict[str, object]:
    """:func:`run_span_horizons` when only the whole-split run is wanted.

    The horizon table and this come out of one pass, so a caller that wants
    both should call :func:`run_span_horizons` rather than this twice.
    """

    kwargs.pop("horizons_minutes", None)
    kwargs.pop("collect_series", None)
    _, whole_span = run_span_horizons(
        model, horizons_minutes=(), collect_series=True, **kwargs
    )
    return whole_span


def horizon_csv_row(
    results: Dict[str, Dict[str, object]],
    prefix: str = "",
    *,
    position: Optional[bool] = None,
) -> Dict[str, float]:
    """Flatten to ``{prefix}vel_rmse_h5m`` style columns for metrics.csv.

    ``position`` says whether the position columns belong in the row at all.
    Left to itself it is inferred from the results, which is wrong in one case
    that matters: when every horizon was skipped there is no position entry to
    infer from, and the columns would go missing from a row whose header still
    has them. A caller that knows - because it supplied the rotation - should
    say so, and then a skipped horizon writes NaN like every other metric.
    """

    if position is None:
        position = any("pos_error_final" in entry for entry in results.values())
    metrics = HORIZON_ROW_METRICS + (HORIZON_POSITION_METRICS if position else ())
    row: Dict[str, float] = {}
    for label, entry in results.items():
        for metric in metrics:
            row[f"{prefix}{metric}_{label}"] = float(entry.get(metric, float("nan")))
    return row


def _minutes_and_seconds(seconds: Optional[float]) -> str:
    """``95.0 -> '1:35'`` - a leg offset reads better as mm:ss than as a float."""

    if seconds is None or not np.isfinite(seconds):
        return "-"
    total = int(round(float(seconds)))
    return f"{total // 60:d}:{total % 60:02d}"


def format_horizon_table(results: Dict[str, Dict[str, object]]) -> str:
    """A fixed-width table, in the order the horizons were requested.

    Every row is the FIRST H minutes of the split, taken out of one unbroken
    run - so the rows are nested, not independent samples, and they are meant
    to be read down the column. ``rmse`` is the typical error over that
    prefix; ``max`` is the single worst tick inside it, which an RMS cannot
    hide, with ``t@max`` saying how far into the split it happened. Because a
    shorter prefix sits inside a longer one, ``max`` can only grow as the
    horizon does; ``rmse`` staying flat means the state is not accumulating
    error, and rising means it is.
    """

    header = (
        f"{'horizon':>9}{'min':>8}"
        f"{'vel_rmse':>10}{'vel_max':>9}{'t@max':>8}"
        f"{'dir_rmse':>10}{'dir_max':>9}{'t@max':>8}"
        f"{'ticks':>10}"
    )
    lines = [header, "-" * len(header)]
    for label, entry in results.items():
        if not entry.get("fits"):
            lines.append(f"{label:>9}{'':>8}   skipped: {entry.get('skipped', '')}")
            continue
        lines.append(
            f"{label:>9}"
            f"{float(entry['minutes_realised']):>8.2f}"
            f"{float(entry['vel_rmse']):>10.3f}"
            f"{float(entry['vel_max_error']):>9.3f}"
            f"{_minutes_and_seconds(entry.get('vel_max_error_time_s')):>8}"
            f"{float(entry['vel_dir_rmse']):>10.2f}"
            f"{float(entry['vel_dir_max_error']):>9.2f}"
            f"{_minutes_and_seconds(entry.get('vel_dir_max_error_time_s')):>8}"
            f"{int(entry['scored_ticks']):>10}"
        )
    if any("baseline_vel_rmse" in entry for entry in results.values()):
        lines.append("")
        lines.append("mean-predictor floor over the same prefixes (rmse only):")
        for label, entry in results.items():
            if not entry.get("fits") or "baseline_vel_rmse" not in entry:
                continue
            lines.append(
                f"{label:>9}{'':>8}"
                f"{float(entry['baseline_vel_rmse']):>10.3f}"
                f"{'':>9}{'':>8}"
                f"{float(entry['baseline_vel_dir_rmse']):>10.2f}"
            )

    if any("pos_error_final" in entry for entry in results.values()):
        lines.append("")
        lines.append(
            "dead-reckoning position error - the velocity integrated from the start"
        )
        lines.append(
            "of the split to the end of each prefix, with the reference attitude on"
        )
        lines.append(
            "BOTH streams, so this is the drift the velocity estimate causes and"
        )
        lines.append("excludes attitude error by construction:")
        position_header = (
            f"{'horizon':>9}{'min':>8}{'path_m':>11}"
            f"{'pos_final':>11}{'pos_max':>10}{'t@max':>8}"
            f"{'pos_rmse':>10}{'drift_%':>9}"
        )
        lines.append(position_header)
        lines.append("-" * len(position_header))
        for label, entry in results.items():
            if not entry.get("fits") or "pos_error_final" not in entry:
                continue
            lines.append(
                f"{label:>9}"
                f"{float(entry['minutes_realised']):>8.2f}"
                f"{float(entry['path_length_m']):>11.0f}"
                f"{float(entry['pos_error_final']):>11.1f}"
                f"{float(entry['pos_error_max']):>10.1f}"
                f"{_minutes_and_seconds(entry.get('pos_error_max_time_s')):>8}"
                f"{float(entry['pos_error_rmse']):>10.1f}"
                f"{float(entry['pos_drift_percent']):>9.2f}"
            )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_HORIZON_MINUTES",
    "HORIZON_POSITION_METRICS",
    "HORIZON_ROW_METRICS",
    "SpanTokens",
    "encode_span_tokens",
    "format_horizon_table",
    "horizon_csv_row",
    "horizon_label",
    "horizon_metric_names",
    "parse_horizon_minutes",
    "prefix_leg",
    "run_span_horizons",
    "scatter_span_tokens",
    "scatter_span_velocity",
    "stream_horizon_metrics",
    "ticks_per_horizon",
    "whole_span_series",
]
