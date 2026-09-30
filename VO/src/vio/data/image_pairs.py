"""Image pairs delivered alongside a telemetry window, for joint training.

The cached-token path hands the pose model a frozen ``.npz``: the visual
frontend is outside the graph and the pose loss cannot reach it. To train the
frontend jointly the same visual events have to arrive as pixels instead, at
the telemetry ticks where they become available.

This module is the bridge. It resolves which image pair belongs to each visual
event, loads and normalises the frames, and reports the event's position inside
the window so the model can scatter the resulting tokens onto the right ticks.

Loading images is the expensive part of joint training, so ``max_events``
bounds how many pairs a single window may carry; when it truncates, the loader
keeps an evenly spaced subset across the window (always including the first
event). That makes memory predictable and lets a long window train with a
subsampled visual schedule rather than not at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .images import (
    mean_over_interval,
    nearest_indices,
    numeric_image_manifest,
    resolve_time_offsets,
)


def resize_camera_matrix(
    camera_matrix: np.ndarray,
    native_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Transform intrinsics for a half-pixel-centred image resize.

    Sizes are (height, width). The principal-point translation matches the
    sampling convention used by OpenCV/PIL resizing: output pixel centre x
    reads native coordinate (x + 0.5) / scale - 0.5.
    """

    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera_matrix must be a finite 3x3 matrix")
    native_height, native_width = (int(value) for value in native_size)
    target_height, target_width = (int(value) for value in target_size)
    if min(native_height, native_width, target_height, target_width) <= 0:
        raise ValueError("native_size and target_size must be positive")
    scale_x = target_width / float(native_width)
    scale_y = target_height / float(native_height)
    resized = matrix.copy()
    resized[0] *= scale_x
    resized[1] *= scale_y
    resized[0, 2] += 0.5 * (scale_x - 1.0)
    resized[1, 2] += 0.5 * (scale_y - 1.0)
    return resized


@dataclass(frozen=True)
class VisualPairPlan:
    """Which image pair belongs to each visual event, and when it lands.

    Two different times live here and they are not interchangeable. The
    ``ready_*`` fields say when an event may be *consumed*, one deployment
    latency after the shutter, and they are what keeps the schedule causal.
    The ``exposure_*`` fields say when the pair was *measured*, and they are
    what any physical quantity read alongside the flow - the body rate above
    all - has to be averaged over. Using the former where the latter belongs
    is a latency-sized attitude error, not a rounding one; see
    ``VisualPairSource.gyro_over_exposure``.
    """

    first_index: np.ndarray
    second_index: np.ndarray
    ready_tick: np.ndarray
    pair_dt_s: np.ndarray
    exposure_t0_s: np.ndarray
    exposure_t1_s: np.ndarray
    telemetry_index0: np.ndarray
    telemetry_index1: np.ndarray
    ready_time_s: np.ndarray


class VisualPairSource:
    """Load the image pair behind each visual event of a flight.

    ``ready_tick`` is the telemetry index at which an event becomes usable,
    which is the same causal placement the cached path uses: an image pair is
    consumed at its ready time, never at its capture time.

    ``grayscale`` stays True by default because the joint-training frontend
    takes single-channel frames. SEA-RAFT is the exception: it was trained on
    RGB, so pass ``grayscale=False`` whenever the stored frames are colour,
    or it matches on a replicated luminance plane and loses every gradient
    that exists only between colour channels. If the JPEGs on disk are
    themselves mode "L" there is nothing to gain: the three channels would be
    identical copies and only the transport cost would triple.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        telemetry_times_s: np.ndarray,
        *,
        image_folder: str = "images",
        image_pattern: str = "*.jpg",
        image_time_scale: float = 0.001,
        image_time_offset_s: "float | Mapping[str, Sequence[float]]" = 0.0,
        deployment_latency_s: float = 0.35,
        frame_gap: int = 1,
        max_frame_gap_s: Optional[float] = None,
        max_telemetry_gap_s: Optional[float] = None,
        image_size: Tuple[int, int] = (288, 384),
        grayscale: bool = True,
        camera_matrix: Optional[np.ndarray] = None,
        distortion: Optional[np.ndarray] = None,
        calibration_image_size: Optional[Tuple[int, int]] = None,
        images_rectified: bool = False,
    ) -> None:
        if frame_gap < 1:
            raise ValueError("frame_gap must be at least one")
        if deployment_latency_s < 0:
            raise ValueError("deployment_latency_s cannot be negative")
        if max_frame_gap_s is not None and not (max_frame_gap_s > 0):
            raise ValueError("max_frame_gap_s must be positive when given")
        root = Path(dataset_root).expanduser().resolve()
        folder = root / image_folder
        if not folder.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {folder}")
        self.paths, capture = numeric_image_manifest(
            folder, image_pattern, image_time_scale
        )
        # A camera and a telemetry logger keep separate clocks. The offset
        # between them is often not constant either, so this accepts a table
        # {"times_s": [...], "offsets_s": [...]} and resolves it per frame;
        # a float still means one constant offset for the whole flight.
        capture = capture + resolve_time_offsets(capture, image_time_offset_s)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        if min(self.image_size) <= 0:
            raise ValueError("image_size must be positive")
        self.grayscale = bool(grayscale)
        self.channels = 1 if self.grayscale else 3
        self.calibration_image_size = (
            tuple(int(value) for value in calibration_image_size)
            if calibration_image_size is not None
            else None
        )
        self.images_rectified = bool(images_rectified)
        self.camera_matrix: Optional[np.ndarray] = None
        self._rectify_maps = None
        if camera_matrix is not None:
            if self.calibration_image_size is None:
                raise ValueError(
                    "calibration_image_size is required with camera_matrix"
                )
            self.camera_matrix = resize_camera_matrix(
                camera_matrix, self.calibration_image_size, self.image_size
            )
        distortion_array = np.asarray(
            [] if distortion is None else distortion, dtype=np.float64
        )
        if distortion_array.ndim != 1 or not np.all(np.isfinite(distortion_array)):
            raise ValueError("distortion must be a finite vector")
        needs_rectification = bool(
            distortion_array.size
            and np.any(np.abs(distortion_array) > 0)
            and not self.images_rectified
        )
        if needs_rectification:
            if camera_matrix is None or self.calibration_image_size is None:
                raise ValueError(
                    "camera_matrix and calibration_image_size are required to "
                    "rectify distorted images"
                )
            try:
                import cv2
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "OpenCV is required to rectify nonzero camera distortion"
                ) from exc
            native_matrix = np.asarray(camera_matrix, dtype=np.float64)
            target_height, target_width = self.image_size
            self._rectify_maps = cv2.initUndistortRectifyMap(
                native_matrix,
                distortion_array,
                None,
                self.camera_matrix,
                (target_width, target_height),
                cv2.CV_32FC1,
            )

        count = len(self.paths) - frame_gap
        if count <= 0:
            raise ValueError("Not enough images for the requested frame gap")
        first = np.arange(count, dtype=np.int64)
        second = first + frame_gap
        exposure_t0 = capture[first]
        exposure_t1 = capture[second]
        # An event is available once the frontend has had time to produce it.
        ready = exposure_t1 + float(deployment_latency_s)
        tick = np.searchsorted(telemetry_times_s, ready, side="left")
        # Both bounds matter. searchsorted returns 0 for anything before the
        # first telemetry tick, so without the lower bound every image captured
        # ahead of the telemetry window would be scattered onto tick 0 as if it
        # had just arrived - which is how a camera-to-IMU offset, or a telemetry
        # range that starts after the images do, would silently fabricate events.
        inside = (tick < telemetry_times_s.size) & (ready >= telemetry_times_s[0])
        # A dropout leaves two frames far apart in time but adjacent in the
        # folder, so the pair looks ordinary and its dt is simply large. The
        # correlator is given a bounded search window, and across a long gap the
        # true displacement leaves it: what comes back is not a small motion but
        # a confident wrong match. Reject the pair instead of scoring it.
        interval = exposure_t1 - exposure_t0
        self.rejected_gap_pairs = 0
        self.max_frame_gap_s = None if max_frame_gap_s is None else float(max_frame_gap_s)
        if self.max_frame_gap_s is not None:
            too_long = interval > self.max_frame_gap_s
            self.rejected_gap_pairs = int(np.count_nonzero(too_long & inside))
            inside = inside & ~too_long
        # A TELEMETRY dropout is the mirror image of an image dropout and is not
        # caught by any of the checks above. An exposure landing inside a hole in
        # the telemetry still finds a nearest sample, still lands inside the
        # range, and still forms an event -- its attitude and altitude are simply
        # interpolated across the hole, from measurements taken a long way away.
        # Interpolating across a 3 s gap is not a measurement, so the bracketing
        # interval is checked and the event dropped when it is implausibly wide.
        #
        # Note what this does NOT catch: a constant camera-clock OFFSET. Dense
        # telemetry always has a near neighbour, so every frame pairs happily
        # with a small residual while being matched to the wrong moment. Only
        # correlating image motion against telemetry rotation finds that, which
        # is what tools/estimate_time_offset.py is for and why
        # --image-time-offset exists.
        step = float(np.median(np.diff(telemetry_times_s))) if telemetry_times_s.size > 1 else 0.0
        if max_telemetry_gap_s is None:
            # Wide enough never to fire on a healthy clock, including a jittered
            # one; narrow enough that a real dropout always trips it.
            max_telemetry_gap_s = 8.0 * step if step > 0 else None
        self.max_telemetry_gap_s = (
            None if max_telemetry_gap_s is None else float(max_telemetry_gap_s)
        )
        self.rejected_telemetry_gap_pairs = 0
        if self.max_telemetry_gap_s is not None and telemetry_times_s.size > 1:
            def bracket_span(query: np.ndarray) -> np.ndarray:
                upper = np.clip(
                    np.searchsorted(telemetry_times_s, query, side="left"),
                    1, telemetry_times_s.size - 1,
                )
                return telemetry_times_s[upper] - telemetry_times_s[upper - 1]

            straddles = (
                (bracket_span(exposure_t0) > self.max_telemetry_gap_s)
                | (bracket_span(exposure_t1) > self.max_telemetry_gap_s)
            )
            self.rejected_telemetry_gap_pairs = int(np.count_nonzero(straddles & inside))
            inside = inside & ~straddles
        # Where the exposures themselves sit on the telemetry clock. These are
        # nearest matches, not left-side insertions: an exposure is an instant,
        # and always snapping it to the tick before it would pull every gyro
        # window half a sample early at both ends. Only INDICES come from here --
        # every VALUE read at an exposure instant is interpolated, never snapped.
        index0, _ = nearest_indices(telemetry_times_s, exposure_t0)
        index1, _ = nearest_indices(telemetry_times_s, exposure_t1)
        self.plan = VisualPairPlan(
            first_index=first[inside],
            second_index=second[inside],
            ready_tick=tick[inside].astype(np.int64),
            pair_dt_s=(exposure_t1 - exposure_t0)[inside],
            exposure_t0_s=exposure_t0[inside].astype(np.float64),
            exposure_t1_s=exposure_t1[inside].astype(np.float64),
            telemetry_index0=index0[inside].astype(np.int64),
            telemetry_index1=index1[inside].astype(np.int64),
            ready_time_s=ready[inside].astype(np.float64),
        )

    def events_in_window(
        self, start: int, end: int, *, min_capture_tick: Optional[int] = None
    ) -> np.ndarray:
        """Indices of the events whose ready tick lands inside ``[start, end)``.

        ``min_capture_tick``, when given, additionally drops any event whose
        EXPOSURE - not its ready tick - began before it. An event's ready
        tick is ``exposure_t1 + deployment_latency_s``, so an event that just
        barely lands inside a window at ``start`` was actually captured one
        whole deployment latency (0.35 s by default - 35+ telemetry ticks at
        100 Hz) earlier. For a window sitting at the very first tick of a
        condition-segment split, that earlier instant routinely falls inside
        the PRECEDING segment - which a manifest may assign to a different
        phase. Dropping the event, rather than only re-averaging its rate,
        matters because what would otherwise leak is not just an attitude
        reading but the two raw video frames themselves, both captured on the
        far side of the boundary: content a strict split must not let a
        validation or test tick be built from. Passing the containing
        phase-range's own start closes that off with no effect anywhere else,
        since every OTHER window's reach-back stays comfortably inside its own
        range.
        """

        tick = self.plan.ready_tick
        inside = (tick >= start) & (tick < end)
        if min_capture_tick is not None:
            inside &= self.plan.telemetry_index0 >= int(min_capture_tick)
        return np.flatnonzero(inside)

    def rate_over_exposure(
        self,
        angular_rate_rad_s: np.ndarray,
        telemetry_times_s: np.ndarray,
        index: int,
    ) -> np.ndarray:
        """Mean angular rate over the pair's own exposure interval.

        The window is ``[exposure_t0_s, exposure_t1_s]`` -- the two shutter
        instants themselves, not the telemetry samples nearest them, and never
        anything derived from ``ready_tick``. The flow describes what happened
        between the two exposures, so the rotation that compensates it has to be
        read over that same interval; ``ready_tick`` is one deployment latency later
        (0.35 s by default), which in a 20 deg/s turn is seven degrees of a
        different attitude. That error is proportional to the turn rate rather
        than random, so it does not average away over a flight: it survives as
        a turn-correlated lateral signal indistinguishable from crab angle, and
        a model handed it will fit it as crab.

        The integral is trapezoidal over the EXACT exposure interval
        ``[exposure_t0_s, exposure_t1_s]``, with both endpoints interpolated
        between the telemetry samples that bracket them, and divided by the
        true duration. Integrating between the nearest samples instead would
        answer a slightly different question over a slightly different span and
        then divide by that span as well, so the error enters twice. At 100 Hz
        each endpoint snap is up to 5 ms, which is a tenth of a 20 Hz frame
        interval; and because the grid phase is arbitrary, it is not zero-mean.
        Interpolating also means an uneven telemetry clock cannot over-weight
        its densely sampled stretches.
        """

        rates = np.asarray(angular_rate_rad_s, dtype=np.float64)
        times = np.asarray(telemetry_times_s, dtype=np.float64)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("telemetry_times_s must be a non-empty 1-D array")
        if rates.ndim != 2 or rates.shape[0] != times.size:
            raise ValueError(
                "angular_rate_rad_s must be (samples, axes) aligned with the "
                f"{times.size} telemetry times; got shape {rates.shape}"
            )
        event = int(index)
        total = int(self.plan.ready_tick.size)
        if not 0 <= event < total:
            raise IndexError(
                f"Visual event {event} is outside this flight's {total} pairs"
            )
        start = float(self.plan.exposure_t0_s[event])
        finish = float(self.plan.exposure_t1_s[event])
        # Hold at the telemetry ends rather than extrapolating past them: a
        # linear extrapolation of body rate beyond the recorded window is not a
        # measurement, and an exposure can legitimately sit just outside it.
        low, high = float(times[0]), float(times[-1])
        start = min(max(start, low), high)
        finish = min(max(finish, low), high)
        return mean_over_interval(times, rates, start, finish)

    def gyro_over_exposure(
        self,
        gyro_rad_s: np.ndarray,
        telemetry_times_s: np.ndarray,
        index: int,
    ) -> np.ndarray:
        """Backward-compatible name for :meth:`rate_over_exposure`.

        The averaging operation is source-agnostic. New no-IMU code calls
        ``rate_over_exposure`` so a legacy helper name cannot obscure its
        input contract.
        """

        return self.rate_over_exposure(gyro_rad_s, telemetry_times_s, index)

    def _load(self, index: int) -> torch.Tensor:
        """One frame as uint8 CHW; normalisation happens on the GPU.

        Windows carry hundreds of frames, and every one of them crosses from
        DataLoader workers to the trainer through shared memory. In float32
        that transport is 4x the size of the JPEG-decoded pixels and was
        enough to OOM-kill a 1 TB machine; the model converts to float and
        divides by 255 after the batch reaches the device.
        """

        from PIL import Image

        with Image.open(self.paths[index]) as handle:
            image = handle.convert("L" if self.grayscale else "RGB")
            if self.calibration_image_size is not None:
                expected_height, expected_width = self.calibration_image_size
                if image.size != (expected_width, expected_height):
                    raise ValueError(
                        f"Image {self.paths[index]} is {image.size[0]}x{image.size[1]}, "
                        f"but calibration expects {expected_width}x{expected_height}. "
                        "Use calibration for the actual crop/resolution."
                    )
            if self._rectify_maps is None:
                image = image.resize(
                    (self.image_size[1], self.image_size[0]), Image.BILINEAR
                )
                # PIL hands back a read-only buffer; copy so torch owns it.
                array = np.asarray(image, dtype=np.uint8).copy()
            else:
                import cv2

                # The maps combine native-resolution undistortion and the
                # working-resolution resize into one interpolation.
                native = np.asarray(image, dtype=np.uint8)
                array = cv2.remap(
                    native,
                    self._rectify_maps[0],
                    self._rectify_maps[1],
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
        if array.ndim == 2:
            array = array[None]
        else:
            array = array.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(array))

    def load_pair(self, event: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both frames of one visual event, as uint8 CHW tensors.

        The event index is into :attr:`plan`, not into the image folder: the
        plan has already dropped the images whose ready time falls outside the
        telemetry range, so the two numberings differ wherever that happened.
        """

        index = int(event)
        total = int(self.plan.ready_tick.size)
        if not 0 <= index < total:
            raise IndexError(
                f"Visual event {index} is outside this flight's {total} pairs"
            )
        return (
            self._load(int(self.plan.first_index[index])),
            self._load(int(self.plan.second_index[index])),
        )

    def window_pairs(
        self,
        start: int,
        end: int,
        max_events: int,
        *,
        min_capture_tick: Optional[int] = None,
    ) -> dict:
        """Padded image pairs for one window, with their tick positions.

        Returns fixed-size tensors so a batch collates without ragged padding
        logic: ``valid`` marks which slots hold a real event. ``min_capture_tick``
        is forwarded to :meth:`events_in_window` unchanged - see there for why
        it exists; the caller must pass the same value here as to any other
        selection for this window (:class:`~vio.data.fixedwing_vo.
        FixedWingVODataset` does), or the two would silently select different
        events for what is supposed to be one window.
        """

        if max_events <= 0:
            raise ValueError("max_events must be positive")
        selected = self.events_in_window(start, end, min_capture_tick=min_capture_tick)
        if selected.size > max_events:
            # An evenly spaced subset keeps visual input across the whole
            # window; keeping only the earliest events would leave everything
            # after the first fraction of a second blind. The first event is
            # always kept so the window's blind period matches
            # visual_blind_ticks(), which the warm-up check is based on.
            pick = np.unique(
                np.linspace(0, selected.size - 1, max_events).round().astype(np.int64)
            )
            selected = selected[pick]
        height, width = self.image_size
        first = torch.zeros(
            max_events, self.channels, height, width, dtype=torch.uint8
        )
        second = torch.zeros(
            max_events, self.channels, height, width, dtype=torch.uint8
        )
        offset = torch.zeros(max_events, dtype=torch.long)
        interval = torch.ones(max_events)
        valid = torch.zeros(max_events)
        for slot, event in enumerate(selected):
            first[slot], second[slot] = self.load_pair(int(event))
            offset[slot] = int(self.plan.ready_tick[event]) - start
            interval[slot] = float(self.plan.pair_dt_s[event])
            valid[slot] = 1.0
        return {
            "visual_image0": first,
            "visual_image1": second,
            "visual_event_offset": offset,
            "visual_event_dt_s": interval,
            "visual_event_valid": valid,
        }


__all__ = ["VisualPairPlan", "VisualPairSource", "resize_camera_matrix"]
