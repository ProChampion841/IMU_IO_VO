"""Windows of attitude, altitude, image pairs and body-velocity targets.

The VIO dataset windows IMU and cached tokens. This one windows what a VO
estimator actually has: the aircraft's own attitude and altitude at telemetry
rate, and raw image pairs at camera rate. There is no IMU and no token cache -
the frontend runs in the training graph, because a scanned encoder that is
being trained cannot be cached.

Two details are worth stating because getting them wrong is invisible.

**The body rate is read over the exposure interval, never at the ready tick.**
The flow describes what happened between two exposures, so the rotation that
compensates it must be read over that same interval. ``ready_tick`` is one
deployment latency later - 0.35 s by default, which in a 20 deg/s turn is
seven degrees of a different attitude. That error is proportional to turn rate
rather than random, so it does not average away: it survives as a
turn-correlated lateral signal indistinguishable from crab, and a model handed
it will fit it as crab. :meth:`VisualPairSource.rate_over_exposure` performs
the source-agnostic interval average; here its input is derived entirely from
the Euler attitude sequence, not from a gyroscope.

**Altitude leaves here twice.** The encoder gets a CENTRED log altitude,
because a channel that sits at 4.6 with no variation to speak of wastes the
first layer's dynamic range. The speed head gets the RAW log altitude, because
``v = h * u`` needs the true magnitude, not an offset one. Centring the copy
the head uses would scale every predicted speed by a constant nobody would
think to look for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from vio.models.pose_geometry import (
    euler_zyx_to_quaternion_np,
    quaternion_to_matrix_np,
)

from .attitude import (
    AttitudeAltitude,
    aiding_features,
    body_rates_from_quaternions,
    load_attitude_altitude,
)
from .image_pairs import VisualPairSource

#: sin/cos roll, sin/cos pitch, centred log altitude, body rates p/q/r, dt.
VO_AIDING_CHANNELS = (
    "sin_roll",
    "cos_roll",
    "sin_pitch",
    "cos_pitch",
    "log_altitude_centred",
    "p_rad_s",
    "q_rad_s",
    "r_rad_s",
    "delta_time_s",
)

#: Where p/q/r sit within VO_AIDING_CHANNELS - derived from the tuple itself
#: rather than hard-coded, so an ablation that zeroes "the body rate columns"
#: (see tools/train_fixedwing_vo.py --ablate-body-rate) cannot silently drift
#: out of sync with a future reordering of VO_AIDING_CHANNELS.
BODY_RATE_AIDING_SLICE = slice(
    VO_AIDING_CHANNELS.index("p_rad_s"), VO_AIDING_CHANNELS.index("r_rad_s") + 1
)

REFERENCE_VELOCITY_COLUMNS = ("GPSNavVnX", "GPSNavVnY", "GPSNavVnZ")
REFERENCE_EULER_COLUMNS = ("GPSNavEulX", "GPSNavEulY", "GPSNavEulZ")

#: One ``(start, end)`` pair, or several - a condition-segment split assigns
#: several disjoint spans to one phase.
IndexRanges = Union[Tuple[int, int], Sequence[Tuple[int, int]]]


def normalize_index_ranges(index_range: IndexRanges) -> List[Tuple[int, int]]:
    """Accept one ``(start, end)`` pair or a list of them; always return a list.

    A condition-segment manifest assigns whole, possibly non-contiguous,
    segments to one phase - ``tools/segment_flight_conditions.py`` deliberately
    interleaves them by condition rather than by time, so that is the common
    case, not an edge case. Collapsing several such spans to a single
    ``(min(start), max(end))`` range folds in whatever OTHER phase's segments
    sit in the gap between them, which is exactly the leak a condition split
    exists to prevent: validation ticks would then also be windowed into
    training. Every consumer of a range in this module goes through here, so a
    plain two-tuple and a list of disjoint ones are handled identically and a
    window is never drawn across the boundary into a different split's ticks.
    """

    if (
        len(index_range) == 2
        and isinstance(index_range[0], (int, np.integer))
        and isinstance(index_range[1], (int, np.integer))
    ):
        return [(int(index_range[0]), int(index_range[1]))]
    ranges = [(int(start), int(end)) for start, end in index_range]
    if not ranges:
        raise ValueError("At least one index range is required")
    ordered = sorted(ranges)
    for (_, prev_end), (next_start, _) in zip(ordered, ordered[1:]):
        if next_start < prev_end:
            raise ValueError(f"Overlapping index ranges are not allowed: {ranges}")
    return ranges


@dataclass(frozen=True)
class VONormalizer:
    """Training-split statistics. Altitude only; the rest is already bounded.

    ``sin`` and ``cos`` live in [-1, 1] by construction and need no scaling,
    and rescaling them would only destroy the property that makes them worth
    using - that they are a smooth embedding of an angle with no wrap.
    """

    log_altitude_mean: float
    delta_time_scale: float

    @classmethod
    def from_range(
        cls, source: AttitudeAltitude, index_range: IndexRanges
    ) -> "VONormalizer":
        ranges = normalize_index_ranges(index_range)
        features = aiding_features(source)
        dt = np.diff(source.times_s, prepend=source.times_s[0])
        # Concatenated over exactly the assigned ticks, never a min/max span:
        # a condition split's whole point is that training statistics come
        # only from the segments actually assigned to training.
        altitude = np.concatenate([features[start:end, 4] for start, end in ranges])
        delta = np.concatenate([dt[start:end] for start, end in ranges])
        return cls(
            log_altitude_mean=float(np.mean(altitude)),
            delta_time_scale=float(np.median(delta)) or 1.0,
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "log_altitude_mean": self.log_altitude_mean,
            "delta_time_scale": self.delta_time_scale,
        }


def reference_body_frame(
    csv_path: str | Path,
    *,
    time_column: str = "Time",
    time_scale: float = 1.0,
    lever_arm_m: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Times, the body-velocity target, and the rotation that defines it.

    This reads ``GPSNavEul*`` and it is the ONLY place that should. The target
    is defined by it, which is exactly why the estimator is not allowed to see
    it - :func:`vio.data.attitude.resolve_attitude_columns` refuses it as an
    input for that reason.

    The rotation is returned as well because turning a body-frame velocity back
    into a distance travelled needs it: ``v_ned = rotation @ v_body``. It is an
    evaluation-only quantity for exactly the same reason the target is, and
    handing it to the model would be the same leak.

    ``lever_arm_m`` is the offset from the GPS antenna to the camera, in the
    BODY frame, in metres. A rigid body rotating at ``omega`` moves its points
    at different velocities:

        v_camera = v_gps + omega x r_gps->camera

    The camera is what sees the motion the network is asked to explain, but GPS
    measures velocity at the antenna. On a wing mount the two are metres apart,
    so in a turn the label describes a point the camera is not at. The error is
    proportional to turn rate, which makes it correlated with exactly the
    manoeuvres the estimator is judged on rather than averaging away: at 30
    deg/s and 2 m of span that is about 1 m/s.

    ``omega`` is differenced from the REFERENCE attitude, not from the
    estimator's ``NavEul*``, so the whole label is built from one consistent
    source. Leave it ``None`` (the default) for no correction, which reproduces
    every checkpoint written before this existed.
    """

    import csv as _csv

    path = Path(csv_path).expanduser().resolve()
    wanted = (time_column, *REFERENCE_VELOCITY_COLUMNS, *REFERENCE_EULER_COLUMNS)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = _csv.reader(handle)
        headers = next(reader)
        missing = [name for name in wanted if name not in headers]
        if missing:
            raise ValueError(f"Telemetry is missing reference columns: {missing}")
        indices = [headers.index(name) for name in wanted]
        rows = [[float(row[i]) for i in indices] for row in reader]
    block = np.asarray(rows, dtype=np.float64)
    times = block[:, 0] * float(time_scale)
    velocity_ned = block[:, 1:4]
    euler = block[:, 4:7]
    if float(np.max(np.abs(euler))) > 2.0 * np.pi + 1e-6:
        euler = np.deg2rad(euler)
    quaternion = euler_zyx_to_quaternion_np(euler)
    rotation = quaternion_to_matrix_np(quaternion)
    body = np.einsum("nij,nj->ni", np.swapaxes(rotation, 1, 2), velocity_ned)

    if lever_arm_m is not None:
        arm = np.asarray(lever_arm_m, dtype=np.float64).reshape(-1)
        if arm.shape != (3,) or not np.all(np.isfinite(arm)):
            raise ValueError("lever_arm_m must be three finite numbers (body xyz, metres)")
        if np.any(arm != 0.0):
            omega = body_rates_from_quaternions(quaternion, times)
            # v_camera = v_gps + omega x r. Row 0 of omega is zero by
            # construction, so the first sample is simply left uncorrected.
            body = body + np.cross(omega, arm)

    return times, body.astype(np.float32), rotation.astype(np.float32)


def reference_body_velocity(
    csv_path: str | Path,
    *,
    time_column: str = "Time",
    time_scale: float = 1.0,
    lever_arm_m: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """The supervision target alone; see :func:`reference_body_frame`."""

    times, body, _ = reference_body_frame(
        csv_path,
        time_column=time_column,
        time_scale=time_scale,
        lever_arm_m=lever_arm_m,
    )
    return times, body


class FixedWingVODataset(Dataset):
    """Causal windows for the vision-Mamba VO model."""

    def __init__(
        self,
        attitude: AttitudeAltitude,
        velocity_body: np.ndarray,
        index_range: IndexRanges,
        normalizer: VONormalizer,
        *,
        image_source: VisualPairSource,
        window_length: int = 400,
        stride: int = 200,
        warmup: int = 20,
        max_visual_events: int = 32,
    ) -> None:
        ranges = normalize_index_ranges(index_range)
        total = attitude.times_s.size
        for start, end in ranges:
            if not 0 <= start < end <= total:
                raise ValueError(f"Invalid VO dataset range {(start, end)}")
        if velocity_body.shape[0] != total:
            raise ValueError("velocity_body must align with the attitude clock")
        if window_length < 2 or stride < 1 or not 0 <= warmup < window_length:
            raise ValueError("Invalid VO window settings")
        if max_visual_events <= 0:
            raise ValueError("max_visual_events must be positive")

        self.attitude = attitude
        self.velocity_body = velocity_body.astype(np.float32)
        self.normalizer = normalizer
        self.image_source = image_source
        self.window_length = int(window_length)
        self.warmup = int(warmup)
        self.max_visual_events = int(max_visual_events)
        self.index_ranges = tuple(ranges)

        features = aiding_features(attitude)
        self.log_altitude = features[:, 4].astype(np.float32)
        dt = np.diff(attitude.times_s, prepend=attitude.times_s[0])
        dt[0] = dt[1] if dt.size > 1 else normalizer.delta_time_scale
        self.aiding = np.concatenate(
            (
                features[:, :4],
                (features[:, 4:5] - normalizer.log_altitude_mean),
                features[:, 5:8],
                (dt[:, None] / max(normalizer.delta_time_scale, 1e-9)).astype(np.float32),
            ),
            axis=1,
        ).astype(np.float32)
        # Windows are drawn independently inside each disjoint span, so a
        # window never straddles the boundary into a different split's ticks -
        # which a single arange over a collapsed (min, max) range would do
        # whenever the assigned segments are not contiguous.
        per_range_starts = [
            np.arange(start, end - self.window_length + 1, int(stride))
            for start, end in ranges
            if end - start >= self.window_length
        ]
        self.starts = (
            np.concatenate(per_range_starts)
            if per_range_starts
            else np.zeros(0, dtype=np.int64)
        )
        if self.starts.size == 0:
            raise ValueError(
                f"No VO windows fit inside the split (ranges {ranges}, "
                f"window_length {self.window_length})"
            )

    def __len__(self) -> int:
        return int(self.starts.size)

    def _range_start(self, start: int) -> int:
        """The start of whichever assigned range contains window ``start``.

        Used to bound how far back an event's own exposure may reach - see
        :meth:`vio.data.image_pairs.VisualPairSource.events_in_window`.
        Unreachable in the ``else`` case by construction: ``start`` always
        comes from :attr:`starts`, which is only ever populated from inside
        one of :attr:`index_ranges`.
        """

        for range_start, range_end in self.index_ranges:
            if range_start <= start < range_end:
                return range_start
        raise AssertionError(  # pragma: no cover - defensive
            f"window start {start} is outside every assigned range "
            f"{self.index_ranges}"
        )

    def _event_body_rates(self, start: int, end: int, offsets: torch.Tensor,
                          valid: torch.Tensor) -> torch.Tensor:
        """Attitude-derived body rate averaged over each pair's exposure."""

        selected = self.image_source.events_in_window(
            start, end, min_capture_tick=self._range_start(start)
        )
        rates = torch.zeros(self.max_visual_events, 3)
        if selected.size > self.max_visual_events:
            pick = np.unique(
                np.linspace(0, selected.size - 1, self.max_visual_events)
                .round()
                .astype(np.int64)
            )
            selected = selected[pick]
        for slot, event in enumerate(selected):
            rates[slot] = torch.from_numpy(
                self.image_source.rate_over_exposure(
                    self.attitude.body_rate_rad_s,
                    self.attitude.times_s,
                    int(event),
                ).astype(np.float32)
            )
        return rates

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        start = int(self.starts[index])
        end = start + self.window_length
        loss_mask = np.ones(self.window_length, dtype=np.float32)
        loss_mask[: self.warmup] = 0.0

        item: Dict[str, torch.Tensor] = {
            "aiding": torch.from_numpy(self.aiding[start:end].copy()),
            "log_altitude": torch.from_numpy(self.log_altitude[start:end].copy()),
            "target_velocity_body": torch.from_numpy(
                self.velocity_body[start:end].copy()
            ),
            "loss_mask": torch.from_numpy(loss_mask),
            # The real telemetry clock, not a tick index - what
            # visual_age_seconds needs to measure physical staleness on an
            # irregular clock instead of assuming a uniform tick interval.
            "telemetry_time_s": torch.from_numpy(
                self.attitude.times_s[start:end].copy()
            ),
        }
        # Both calls into image_source must agree on min_capture_tick, or
        # they would silently select different events for one window - see
        # VisualPairSource.events_in_window for why the bound exists at all.
        range_start = self._range_start(start)
        pairs = self.image_source.window_pairs(
            start, end, self.max_visual_events, min_capture_tick=range_start
        )
        item.update(pairs)
        item["visual_event_body_rate"] = self._event_body_rates(
            start, end, pairs["visual_event_offset"], pairs["visual_event_valid"]
        )
        # A window whose visual events all land before its first usable tick
        # would train the fusion on presence bits that never fire.
        item["visual_event_valid"] = pairs["visual_event_valid"]
        return item


class ChronologicalWindowSampler(Sampler):
    """Batches of window indices in time order, ``batch_size`` independent lanes.

    Built for TBPTT (see ``PLAN_TBPTT.txt``): ordinary shuffled batching is
    wrong for a training path that carries recurrent state between batches,
    because then "the next batch" bears no relation to "the tick after the
    last one this state saw." This sampler instead splits the dataset's own
    window order into ``batch_size`` lanes, each walking strictly forward in
    time, and records - per lane, per step - whether that step's window is
    the TRUE chronological successor of the one the same lane saw last, so
    the caller (see :func:`mask_stream_state` in
    :mod:`vio.models.vision_mamba_vo`) knows exactly which lanes must reset
    rather than carry state forward. :attr:`continues` holds that record;
    ``continues[lane][step]`` is ``False`` at ``step == 0`` for every lane
    (nothing has run yet to carry) and whenever a lane's window crosses an
    :attr:`~FixedWingVODataset.index_ranges` boundary - a condition-segment
    split deliberately assigns disjoint spans to one phase, and treating two
    of them as one continuous recording would carry state across a join that
    was never actually flown.

    Requires ``stride == window_length``: the dataset's own windows must tile
    each range with no overlap and no gap, or "the next window" would either
    re-process ticks the previous one already saw or skip ticks it never
    saw - and this scaffold has no tick-level re-seen mask to paper over the
    former. Raises rather than silently training on either.

    The last ``len(dataset) % batch_size`` windows are dropped so every lane
    is exactly :attr:`num_steps` long, the same ``drop_last`` reasoning
    ``tools/train_fixedwing_vo.py`` already applies to its DistributedSampler.
    """

    def __init__(self, dataset: "FixedWingVODataset", batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        window_length = dataset.window_length
        starts = [int(value) for value in dataset.starts]
        # dataset.starts is already sorted ascending within each range and
        # concatenated in range order (see FixedWingVODataset.__init__), so
        # one linear scan both tags each window's owning range and catches a
        # stride/gap violation in the same pass.
        owners = [dataset._range_start(start) for start in starts]
        for index in range(1, len(starts)):
            if (
                owners[index] == owners[index - 1]
                and starts[index] != starts[index - 1] + window_length
            ):
                raise ValueError(
                    "ChronologicalWindowSampler requires stride == "
                    f"window_length (no overlap, no gap): window start "
                    f"{starts[index]} follows {starts[index - 1]} by "
                    f"{starts[index] - starts[index - 1]} ticks, not "
                    f"{window_length}. Build the dataset with "
                    "stride=window_length for TBPTT."
                )

        lane_length = len(starts) // int(batch_size)
        if lane_length == 0:
            raise ValueError(
                f"Not enough windows ({len(starts)}) to fill {batch_size} "
                "TBPTT lanes"
            )
        self.batch_size = int(batch_size)
        self.num_steps = int(lane_length)
        # order[i] is the dataset index whose window start is starts[i] - the
        # two lists share an index because both come from the same walk over
        # dataset.starts.
        order = list(range(len(starts)))
        self.lanes = [
            order[lane * lane_length : (lane + 1) * lane_length]
            for lane in range(self.batch_size)
        ]
        self.continues: List[List[bool]] = [
            [
                step > 0
                and owners[self.lanes[lane][step]] == owners[self.lanes[lane][step - 1]]
                for step in range(lane_length)
            ]
            for lane in range(self.batch_size)
        ]

    def __len__(self) -> int:
        return self.num_steps

    def __iter__(self):
        for step in range(self.num_steps):
            yield [self.lanes[lane][step] for lane in range(self.batch_size)]

    def continues_at(self, step: int) -> torch.Tensor:
        """``(batch_size,)`` bool: whether each lane's window at ``step`` is
        the chronological successor of the same lane's window at ``step - 1``.

        Feed this to :func:`~vio.models.vision_mamba_vo.mask_stream_state` as
        ``keep`` - ``True`` means carry that lane's state forward, ``False``
        means the next window is not a real continuation and the lane must
        start fresh.
        """

        return torch.tensor(
            [self.continues[lane][step] for lane in range(self.batch_size)],
            dtype=torch.bool,
        )


def tbptt_loss_mask(
    loss_mask: torch.Tensor, continues: torch.Tensor, warmup: int
) -> torch.Tensor:
    """Undo a dataset window's own warm-up mask for lanes that carried state in.

    ``FixedWingVODataset.__getitem__`` zeroes every window's first ``warmup``
    ticks unconditionally, because a window scored on its own is always a
    cold start - state resets, so those ticks genuinely have no history to
    draw on. TBPTT breaks that assumption for any lane whose window is a true
    chronological continuation (``continues[lane]`` True, from
    :meth:`ChronologicalWindowSampler.continues_at`): its state already
    carries everything the previous chunk saw, so re-applying the cold-start
    warm-up would zero ticks that are not actually blind, silently discarding
    real supervision on every continued lane, every step.

    ``loss_mask`` is ``(B, T)`` (or broadcastable to it) as the dataset
    produces; only ``continues`` lanes have their own first ``warmup`` ticks
    restored to 1 - a reset lane keeps the dataset's original zeros, because
    for THAT lane the cold start is real.
    """

    if warmup <= 0:
        return loss_mask
    if continues.ndim != 1 or continues.shape[0] != loss_mask.shape[0]:
        raise ValueError("continues must be a 1-D (batch,) tensor matching loss_mask")
    mask = loss_mask.clone()
    mask[continues, :warmup] = 1.0
    return mask


def build_vo_dataset(
    dataset_root: str | Path,
    index_range: IndexRanges,
    *,
    csv_name: str = "flight.csv",
    time_column: str = "Time",
    time_scale: float = 1.0,
    attitude_columns: Optional[Sequence[str]] = None,
    altitude_column: Optional[str] = None,
    allow_reference_attitude: bool = False,
    image_folder: str = "images",
    image_size: Tuple[int, int] = (288, 384),
    frame_gap: int = 1,
    max_frame_gap_s: Optional[float] = None,
    image_time_offset_s: "float | Mapping[str, Sequence[float]]" = 0.0,
    lever_arm_m: Optional[Sequence[float]] = None,
    deployment_latency_s: float = 0.35,
    camera_matrix: Optional[np.ndarray] = None,
    calibration_image_size: Optional[Tuple[int, int]] = None,
    images_rectified: bool = False,
    distortion: Optional[Sequence[float]] = None,
    normalizer: Optional[VONormalizer] = None,
    **window_kwargs,
) -> Tuple[FixedWingVODataset, VONormalizer, AttitudeAltitude]:
    """Assemble a VO dataset from one flight directory.

    ``distortion`` is the calibration's lens distortion vector. Forwarded to
    :class:`~vio.data.image_pairs.VisualPairSource` unchanged: it is the only
    thing that can rectify an image before the frontend sees it, and it was
    previously read by :mod:`vio.data.calibration` and then never passed here,
    so a calibration file with real distortion had it silently ignored by
    every caller of this function.
    """

    root = Path(dataset_root).expanduser().resolve()
    csv_path = root / csv_name
    attitude = load_attitude_altitude(
        csv_path,
        time_column=time_column,
        time_scale=time_scale,
        attitude_columns=attitude_columns,
        altitude_column=altitude_column,
        allow_reference_attitude=allow_reference_attitude,
    )
    target_times, velocity_body = reference_body_velocity(
        csv_path,
        time_column=time_column,
        time_scale=time_scale,
        lever_arm_m=lever_arm_m,
    )
    if target_times.shape != attitude.times_s.shape or not np.allclose(
        target_times, attitude.times_s, atol=1e-9
    ):
        raise ValueError("Target and attitude clocks do not align")

    image_source = VisualPairSource(
        root,
        attitude.times_s,
        image_folder=image_folder,
        image_size=image_size,
        frame_gap=frame_gap,
        max_frame_gap_s=max_frame_gap_s,
        image_time_offset_s=image_time_offset_s,
        deployment_latency_s=deployment_latency_s,
        camera_matrix=camera_matrix,
        calibration_image_size=calibration_image_size,
        images_rectified=images_rectified,
        distortion=distortion,
        grayscale=True,
    )
    if normalizer is None:
        normalizer = VONormalizer.from_range(attitude, index_range)
    dataset = FixedWingVODataset(
        attitude,
        velocity_body,
        index_range,
        normalizer,
        image_source=image_source,
        **window_kwargs,
    )
    return dataset, normalizer, attitude


def scatter_visual_tokens(
    tokens: torch.Tensor,
    quality: torch.Tensor,
    offsets: torch.Tensor,
    valid: torch.Tensor,
    *,
    window_length: int,
    visual_dim: int,
    delivered: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place per-event tokens at the tick where they became available.

    Events land at ``ready_tick``, one deployment latency after the shutter, so
    the model never sees a frame it could not have had. Slots outside the
    window or marked invalid are dropped rather than clamped: clamping would
    stack several events on the boundary tick and silently change the
    schedule.

    ``delivered`` is a per-event 0/1 mask for a pair the frontend's reliability
    gate REFUSED. Such a pair is still SCATTERED - its token multiplied by
    zero and its presence bit left at zero - rather than dropped from the
    index set, and the difference is not cosmetic. Dropping it would leave the
    frontend contributing nothing to the graph on any batch where every pair
    was refused, so its parameters would receive no gradient that step;
    ``static_graph`` DDP records the used-parameter set on the first iteration
    and fails when a later one differs. Multiplying by zero keeps the edge and
    the gradient, both exactly zero, and the values the model reads are
    identical either way.
    """

    batch = tokens.shape[0]
    device = tokens.device
    token_field = torch.zeros(batch, window_length, visual_dim, device=device,
                              dtype=tokens.dtype)
    quality_field = torch.zeros(batch, window_length, 1, device=device,
                                dtype=tokens.dtype)
    present = torch.zeros(batch, window_length, 1, device=device, dtype=tokens.dtype)
    inside = (offsets >= 0) & (offsets < window_length) & (valid > 0)
    if delivered is None:
        keep = torch.ones_like(tokens[..., :1])
    else:
        keep = delivered.reshape(batch, -1, 1).to(
            device=tokens.device, dtype=tokens.dtype
        )
    for sample in range(batch):
        slots = torch.nonzero(inside[sample], as_tuple=False).flatten()
        if slots.numel() == 0:
            continue
        ticks = offsets[sample, slots]
        token_field[sample, ticks] = tokens[sample, slots] * keep[sample, slots]
        quality_field[sample, ticks] = quality[sample, slots] * keep[sample, slots]
        present[sample, ticks] = keep[sample, slots]
    return token_field, quality_field, present


#: Sentinel meaning "no image has ever been delivered" in a carried age-carry
#: tensor - see :func:`visual_age_seconds`. Not NaN: NaN breaks the plain
#: ``>=``/``maximum`` comparisons the cummax-based carry-in needs, where -inf
#: participates correctly (it never wins a maximum against a real time, and a
#: real time always wins against it).
NO_IMAGE_YET = float("-inf")


def visual_age_seconds(
    present: torch.Tensor,
    times_s: torch.Tensor,
    deployment_latency_s: float,
    *,
    carry: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Physical staleness of the currently-held visual token, in seconds.

    ``present`` is ``(B, T, 1)`` and ``times_s`` is the matching ``(B, T)`` (or
    ``(T,)``) telemetry clock - real elapsed seconds, not a tick count, so an
    irregular clock (a dropped sample, a jittered logger) is measured
    correctly rather than assumed uniform. Returns ``(age_seconds, new_carry)``.

    **Not zero on arrival.** ``present`` fires at ``ready_tick`` - the tick a
    token becomes available - which is one ``deployment_latency_s`` AFTER the
    second image was actually captured (see
    :class:`~vio.data.image_pairs.VisualPairSource`). Reporting age 0 at that
    tick would claim the image is brand new when it is already
    ``deployment_latency_s`` old (0.35 s by default - the dominant term for
    any tick soon after arrival). So age is measured from the CAPTURE instant,
    approximated as ``times_s`` at the delivery tick minus the latency, which
    is exact to within one telemetry sample:

        age(t) = (times_s[t] - times_s[last_delivery_tick]) + deployment_latency_s

    **Zero before any image exists.** Before the first delivery this call can
    see - and with no ``carry`` handed in - there is no captured image to be
    stale, so age is a flat 0, never a value growing from nothing. This is
    also why visual-disabled evaluation (``present`` all zero) reports a
    constant 0 throughout: there was never an image to begin with.

    **Continuity.** The streamed evaluator (:mod:`vio.models.velocity_horizons`)
    calls this ONCE over a whole span and slices the result per block, so age
    stays continuous across a block boundary the same way the token/quality/
    presence fields already do - ``carry`` is not needed there. TBPTT
    (:class:`ChronologicalWindowSampler`) instead processes one window per
    call, so ``carry`` - the ``(B,)`` ``times_s``-scale delivery time this
    same call last returned as ``new_carry`` - is what keeps a continuing
    lane's age growing from where the previous chunk left it rather than
    resetting to 0 at every window boundary. Mask it with
    :func:`mask_age_carry` between chunks exactly like
    ``mask_stream_state`` masks the model's own recurrent state; a genuine
    reset passes ``carry=None``, the same convention ``state=None`` uses.
    """

    if present.ndim != 3 or present.shape[-1] != 1:
        raise ValueError("present must have shape (B, T, 1)")
    batch, length, _ = present.shape
    times = torch.as_tensor(times_s, device=present.device)
    if times.ndim == 1:
        times = times.unsqueeze(0).expand(batch, length)
    if times.shape != (batch, length):
        raise ValueError("times_s must have shape (T,) or (B, T)")
    times = times.to(torch.float64)

    seen = present.squeeze(-1) > 0
    delivery_time = torch.where(
        seen, times, torch.full_like(times, NO_IMAGE_YET)
    )
    if carry is not None:
        if carry.shape != (batch,):
            raise ValueError("carry must have shape (B,)")
        carried = carry.to(device=times.device, dtype=times.dtype)
        # Competes at t=0 exactly like any other candidate: cummax lets a
        # real delivery inside THIS call override it the moment one arrives.
        delivery_time = delivery_time.clone()
        delivery_time[:, 0] = torch.maximum(delivery_time[:, 0], carried)
    last_delivery, _ = torch.cummax(delivery_time, dim=1)

    never = last_delivery == NO_IMAGE_YET
    age = torch.where(
        never,
        torch.zeros_like(times),
        (times - last_delivery) + float(deployment_latency_s),
    )
    new_carry = last_delivery[:, -1].to(torch.float32)
    return age.unsqueeze(-1).to(present.dtype), new_carry


def mask_age_carry(carry: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """The age-carry counterpart of ``mask_stream_state``: reset means "no
    image known" (:data:`NO_IMAGE_YET`), not zero - a carried delivery TIME of
    0 would claim an image was captured at the telemetry clock's origin,
    which is a real (very stale) measurement, not the absence of one.
    """

    if keep.shape != carry.shape:
        raise ValueError("keep must have the same shape as carry")
    return torch.where(
        keep.to(device=carry.device),
        carry,
        torch.full_like(carry, NO_IMAGE_YET),
    )


__all__ = [
    "BODY_RATE_AIDING_SLICE",
    "IndexRanges",
    "NO_IMAGE_YET",
    "REFERENCE_EULER_COLUMNS",
    "REFERENCE_VELOCITY_COLUMNS",
    "VO_AIDING_CHANNELS",
    "ChronologicalWindowSampler",
    "FixedWingVODataset",
    "VONormalizer",
    "build_vo_dataset",
    "mask_age_carry",
    "normalize_index_ranges",
    "reference_body_frame",
    "reference_body_velocity",
    "scatter_visual_tokens",
    "tbptt_loss_mask",
    "visual_age_seconds",
]
