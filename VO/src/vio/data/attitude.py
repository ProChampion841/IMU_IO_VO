"""Attitude and altitude read from telemetry, and the rates derived from them.

With no IMU in the estimator, the aircraft's own attitude solution is what
de-rotates the flow and tilts the ground plane, and its altitude is what puts
the flow in metres. Four things here are easy to get subtly wrong, and each one
fails silently rather than loudly.

**Which columns.** The logger writes attitude three times over - ``NavEul*``,
``Eul*`` and ``GPSNavEul*`` - in different units and different conventions.
``GPSNavEul*`` is refused by default and the refusal is the point: the body
velocity target is ``GPSNavVn`` ROTATED BY ``GPSNavEul`` (see
``fixedwing_pose.derive_pose_reference``), so feeding it as an input hands the
model one of the two ingredients of its own label. Validation looks excellent
and the model fails in flight, which is the worst failure mode available.

**Which unit.** Degrees and radians are both plausible for an Euler column.
``auto`` decides from the range, since a yaw column in degrees spans far beyond
2*pi.

**Which rate.** The flow describes rotation BETWEEN two exposures, so the
compensating rate is the relative rotation between the attitudes at those two
instants, ``Log(R_0^T R_1) / dt``, in the body frame. A differentiated Euler
angle is neither a body rate nor well defined through a heading wrap.

**Whether the column actually moves.** ``GPSNav*`` columns are GPS-derived, and
GPS updates at 5-10 Hz while telemetry logs at 100 Hz. A held column looks
perfectly well formed and produces a relative rotation of exactly zero on most
pairs and a jump on the rest - which is worse than no de-rotation at all,
because the jumps land on turns. :func:`hold_fraction` measures it and the
loader refuses to stay quiet about it.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from vio.models.pose_geometry import (
    euler_zyx_to_quaternion_np,
    quaternion_conjugate_np,
    quaternion_multiply_np,
    quaternion_to_rotvec_np,
)

#: Preference order. The navigation filter's own solution first; the raw
#: ``Eul*`` triple second; the GPS reference last and only on request.
ATTITUDE_CANDIDATES: Tuple[Tuple[str, ...], ...] = (
    ("NavEulX", "NavEulY", "NavEulZ"),
    ("EulX", "EulY", "EulZ"),
)
REFERENCE_EULER_COLUMNS = ("GPSNavEulX", "GPSNavEulY", "GPSNavEulZ")

#: The flight computer's own relative altitude, under every spelling seen in
#: the wild: ``relativeAlt``, ``RelativeAlt``, ``RelatedAlt``. Matching is
#: exact, so all three are listed rather than one canonical form -- a capture
#: spelled differently from this tuple would otherwise fall through to
#: ``Barometer``, an unrelated height source, without saying so.
#: ``Barometer`` stays last as the older fallback. The chain is tried in
#: order so one command line reads every supported capture.
ALTITUDE_CANDIDATES = (
    "relativeAlt",
    "RelativeAlt",
    "RelatedAlt",
    "Barometer",
)


@dataclass(frozen=True)
class AttitudeAltitude:
    """Per-telemetry-row attitude, altitude, and the rates between rows."""

    times_s: np.ndarray
    #: (N, 3) roll/pitch/yaw in radians, continuous quaternion sign.
    euler_rad: np.ndarray
    #: (N, 4) body-to-NED quaternion, sign-continuous for finite differencing.
    quaternion: np.ndarray
    #: (N, 3) body rates. Row i is the rate from row i-1 to row i; row 0 is 0.
    body_rate_rad_s: np.ndarray
    #: (N,) altitude in metres at its RECORDED magnitude. Not re-based:
    #: the speed head multiplies by it, so a shifted datum is a scaled
    #: velocity. Conditioning belongs in the encoder's centred copy.
    altitude_m: np.ndarray
    attitude_columns: Tuple[str, ...]
    altitude_column: str
    euler_unit: str
    #: Fraction of consecutive rows where attitude did not change at all.
    attitude_hold_fraction: float
    notes: Tuple[str, ...]

    @property
    def roll_pitch_rad(self) -> np.ndarray:
        """Roll and pitch only.

        Yaw is deliberately excluded. Body-frame velocity is yaw-invariant -
        crab angle is the angle between the body x axis and the ground velocity
        vector, both body-frame quantities - so yaw carries no information the
        target needs, while being the component most strongly coupled to GPS in
        any navigation filter.
        """

        return self.euler_rad[:, :2]


def detect_euler_unit(values: np.ndarray) -> str:
    """``'radians'`` or ``'degrees'`` for an ``(N, 3)`` Euler block."""

    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Euler values must have shape (N, 3)")
    extent = float(np.nanmax(np.abs(values))) if values.size else 0.0
    # A yaw column in radians cannot exceed 2*pi; one in degrees routinely
    # reaches 180. The gap is wide enough that no tolerance is needed.
    return "degrees" if extent > 2.0 * math.pi + 1e-6 else "radians"


def hold_fraction(values: np.ndarray) -> float:
    """Fraction of consecutive rows that are bit-identical.

    A sensor logged at its own rate always dithers. A column held between
    updates of a slower source does not, and the flat stretches are invisible
    in every summary statistic that matters - mean, range, spectrum-by-eye.
    """

    if values.shape[0] < 2:
        return 0.0
    unchanged = np.all(values[1:] == values[:-1], axis=tuple(range(1, values.ndim)))
    return float(np.mean(unchanged))


def body_rates_from_quaternions(
    quaternion: np.ndarray, times_s: np.ndarray
) -> np.ndarray:
    """``Log(R_{i-1}^T R_i) / dt`` in the body frame, in rad/s.

    Row 0 is zero because there is no earlier attitude to difference against.
    """

    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("quaternion must have shape (N, 4)")
    if times_s.shape[0] != quaternion.shape[0]:
        raise ValueError("times and quaternion must have the same length")
    rates = np.zeros((quaternion.shape[0], 3), dtype=np.float64)
    if quaternion.shape[0] < 2:
        return rates
    relative = quaternion_multiply_np(
        quaternion_conjugate_np(quaternion[:-1]), quaternion[1:]
    )
    rotation_vector = quaternion_to_rotvec_np(relative)
    dt = np.diff(times_s.astype(np.float64))
    # A non-positive interval means the clock is not monotonic; leaving the
    # division to produce an infinity would poison de-rotation downstream.
    safe = np.where(dt > 0.0, dt, np.nan)
    rates[1:] = rotation_vector / safe[:, None]
    return np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)


def _sign_continuous(quaternion: np.ndarray) -> np.ndarray:
    """Choose a continuous sign sequence.

    ``q`` and ``-q`` are the same rotation, but a sign flip between rows turns
    a zero relative rotation into a 360-degree one when differenced.
    """

    output = quaternion.copy()
    for index in range(1, output.shape[0]):
        if float(np.dot(output[index - 1], output[index])) < 0.0:
            output[index] *= -1.0
    return output


def resolve_attitude_columns(
    headers: Sequence[str],
    requested: Optional[Sequence[str]] = None,
    *,
    allow_reference: bool = False,
) -> Tuple[Tuple[str, ...], str]:
    """Pick the attitude triple to read, and say why.

    Raises rather than silently falling back to ``GPSNavEul*``: that column is
    an ingredient of the training target, so reading it by accident is target
    leakage, and an accident is exactly what a silent fallback produces.
    """

    available = set(headers)
    if requested is not None:
        names = tuple(requested)
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(f"Requested attitude columns are absent: {missing}")
        if names == REFERENCE_EULER_COLUMNS and not allow_reference:
            raise ValueError(_LEAKAGE_MESSAGE)
        return names, f"requested {names[0][:-1]}*"

    for candidate in ATTITUDE_CANDIDATES:
        if all(name in available for name in candidate):
            return candidate, f"chose {candidate[0][:-1]}* by availability"

    if all(name in available for name in REFERENCE_EULER_COLUMNS):
        if not allow_reference:
            raise ValueError(_LEAKAGE_MESSAGE)
        return REFERENCE_EULER_COLUMNS, "fell back to GPSNavEul* ON REQUEST (leaky)"

    raise ValueError(
        "Telemetry carries no usable attitude columns. Looked for "
        f"{[c[0][:-1] + '*' for c in ATTITUDE_CANDIDATES]}."
    )


_LEAKAGE_MESSAGE = (
    "GPSNavEul* is an ingredient of the training target: the body-frame "
    "velocity label is GPSNavVn rotated by GPSNavEul. Feeding it as an input "
    "hands the model half of its own answer, so validation will look excellent "
    "and the model will fail in flight.\n"
    "Use NavEul* or Eul*. If you genuinely intend a leakage ablation, pass "
    "allow_reference=True and report the run separately."
)


def resolve_altitude_column(
    headers: Sequence[str], requested: Optional[str] = None
) -> str:
    available = set(headers)
    if requested is not None:
        if requested not in available:
            raise ValueError(f"Requested altitude column is absent: {requested}")
        return requested
    for candidate in ALTITUDE_CANDIDATES:
        if candidate in available:
            return candidate
    raise ValueError(
        f"Telemetry carries no altitude column; looked for {list(ALTITUDE_CANDIDATES)}."
    )


def _read_columns(
    csv_path: Path, names: Sequence[str]
) -> Tuple[Tuple[str, ...], np.ndarray]:
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        headers = tuple(next(reader))
        missing = [name for name in names if name not in headers]
        if missing:
            raise ValueError(f"Telemetry is missing columns: {missing}")
        indices = [headers.index(name) for name in names]
        rows = []
        for row_number, row in enumerate(reader, start=2):
            try:
                rows.append([float(row[index]) for index in indices])
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"Invalid value in {list(names)} at CSV row {row_number}"
                ) from error
    array = np.asarray(rows, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{csv_path} carries no telemetry rows")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Non-finite values in {list(names)}")
    return headers, array


def read_headers(csv_path: Path) -> Tuple[str, ...]:
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        return tuple(next(csv.reader(handle)))


def load_attitude_altitude(
    csv_path: str | Path,
    *,
    time_column: str = "Time",
    time_scale: float = 1.0,
    attitude_columns: Optional[Sequence[str]] = None,
    altitude_column: Optional[str] = None,
    euler_unit: str = "auto",
    allow_reference_attitude: bool = False,
    hold_warning_threshold: float = 0.25,
) -> AttitudeAltitude:
    """Read the estimator's attitude and altitude from one flight CSV."""

    path = Path(csv_path).expanduser().resolve()
    headers = read_headers(path)
    names, choice_note = resolve_attitude_columns(
        headers, attitude_columns, allow_reference=allow_reference_attitude
    )
    altitude_name = resolve_altitude_column(headers, altitude_column)
    _, block = _read_columns(path, (time_column, *names, altitude_name))

    times = block[:, 0] * float(time_scale)
    euler = block[:, 1:4]
    altitude = block[:, 4]
    notes = [choice_note, f"altitude from {altitude_name}"]

    # A column that is present but constant carries no attitude, and a model
    # fed it de-rotates with nothing while reporting no error at all.
    spread = float(np.max(np.abs(euler - euler[0]))) if euler.shape[0] else 0.0
    if spread <= 1e-9:
        raise ValueError(
            f"Attitude columns {list(names)} are constant over the whole flight; "
            "they carry no attitude. Check the logger configuration."
        )

    unit = detect_euler_unit(euler) if euler_unit == "auto" else euler_unit
    if unit not in {"radians", "degrees"}:
        raise ValueError("euler_unit must be 'auto', 'radians' or 'degrees'")
    if unit == "degrees":
        euler = np.deg2rad(euler)
    notes.append(f"euler unit {unit}")

    quaternion = _sign_continuous(euler_zyx_to_quaternion_np(euler))
    rates = body_rates_from_quaternions(quaternion, times)

    held = hold_fraction(block[:, 1:4])
    if held > hold_warning_threshold:
        notes.append(
            f"WARNING {held:.1%} of consecutive rows repeat the attitude exactly. "
            "This column is almost certainly held between updates of a slower "
            "source, so the relative rotation is zero on most pairs and a jump "
            "on the rest - and the jumps land on turns. De-rotation built on it "
            "will be worse than none."
        )

    # Altitude enters the speed head as a factor, so it is kept at its
    # recorded magnitude; conditioning happens in the centred copy the encoder
    # sees. The median is reported because it is what the column being AGL or
    # not shows up as, and a 126 m median on a 120 m flight is right while a
    # 1200 m one means the datum is sea level.
    notes.append(f"altitude median {float(np.median(altitude)):.1f} m (kept, not re-based)")

    return AttitudeAltitude(
        times_s=times,
        euler_rad=euler.astype(np.float64),
        quaternion=quaternion.astype(np.float64),
        body_rate_rad_s=rates,
        altitude_m=altitude.astype(np.float64),
        attitude_columns=tuple(names),
        altitude_column=altitude_name,
        euler_unit=unit,
        attitude_hold_fraction=held,
        notes=tuple(notes),
    )


def aiding_features(
    source: AttitudeAltitude,
    *,
    altitude_floor_m: float = 1.0,
    altitude_offset_m: float = 0.0,
) -> np.ndarray:
    """The per-row aiding vector the VO model consumes.

    Eight channels: ``sin/cos`` of roll and pitch, log altitude, and the body
    rates ``p, q, r``.

    Roll and pitch are given as sine and cosine rather than as angles so the
    encoder never sees a wrap discontinuity, and so a level attitude is a
    smooth point rather than a boundary between +pi and -pi.

    Altitude is given as a logarithm because it enters the velocity as a
    FACTOR - ``v = h * u`` - and a sum in log space is a product in linear
    space. A linear altitude channel would ask the network to learn a product
    from a concatenation, which is exactly the thing a linear layer cannot do.

    The altitude is used at its RECORDED magnitude. Re-basing it - subtracting
    its own minimum, say, to make the log better conditioned - looks harmless
    and is not: it turns a 120 m flight varying by 12 m into a 12 m one, and
    every predicted speed comes out an order of magnitude low with nothing in
    the loss to say why. Conditioning belongs in a CENTRED copy given to the
    encoder, never in the copy the speed head multiplies by.

    ``altitude_offset_m`` is added before the log and is the one legitimate
    adjustment: ``relativeAlt`` is measured from takeoff, so it is height above
    ground only if the aircraft took off from the ground it is now flying over.

    The body rates are ``source.body_rate_rad_s`` unchanged - already the
    correctly-derived ``Log(R_{i-1}^T R_i) / dt`` this module computes at load
    time, not a fresh differentiation. They give the encoder the turn itself,
    not just its trace in roll/pitch: a coordinated turn holds roll and pitch
    near constant while yaw rate carries the whole manoeuvre, which the four
    channels above cannot see at all.
    """

    roll_pitch = source.roll_pitch_rad
    altitude = np.maximum(source.altitude_m + float(altitude_offset_m), altitude_floor_m)
    return np.concatenate(
        (
            np.stack(
                (
                    np.sin(roll_pitch[:, 0]),
                    np.cos(roll_pitch[:, 0]),
                    np.sin(roll_pitch[:, 1]),
                    np.cos(roll_pitch[:, 1]),
                    np.log(altitude),
                ),
                axis=1,
            ),
            source.body_rate_rad_s,
        ),
        axis=1,
    ).astype(np.float32)


AIDING_CHANNELS = (
    "sin_roll", "cos_roll", "sin_pitch", "cos_pitch", "log_altitude",
    "p_rad_s", "q_rad_s", "r_rad_s",
)


def quaternion_at(
    times_s: np.ndarray, quaternion: np.ndarray, query_s: "float | np.ndarray"
) -> np.ndarray:
    """Body-to-NED attitude at arbitrary instants, ``(..., 4)`` (w, x, y, z).

    Normalised linear interpolation between the two rows that bracket each
    query, held at the ends of the log. At 100 Hz consecutive attitudes are a
    few milliradians apart, where nlerp and slerp agree to a microradian, and
    the rows are already sign-continuous (see :func:`load_attitude_altitude`),
    so no hemisphere flip is needed.

    An exposure instant is almost never a telemetry row, and snapping it to the
    nearest one moves the attitude by up to half a sample - at a 16 deg/s turn
    that is 0.08 deg per snap, applied at both ends of every pair.
    """

    times = np.asarray(times_s, dtype=np.float64)
    quats = np.asarray(quaternion, dtype=np.float64)
    query = np.asarray(query_s, dtype=np.float64)
    clamped = np.clip(query, times[0], times[-1])
    upper = np.clip(np.searchsorted(times, clamped, side="right"), 1, times.size - 1)
    lower = upper - 1
    span = times[upper] - times[lower]
    fraction = np.where(span > 0, (clamped - times[lower]) / np.where(span > 0, span, 1.0), 0.0)
    blended = (1.0 - fraction)[..., None] * quats[lower] + fraction[..., None] * quats[upper]
    return blended / np.linalg.norm(blended, axis=-1, keepdims=True)


def pair_geometry_batch(
    attitude: AttitudeAltitude, start_s: np.ndarray, stop_s: np.ndarray
) -> Dict[str, np.ndarray]:
    """:func:`pair_geometry` for many pairs at once: ``(N, 3, 3)``,
    ``(N, 3)`` and ``(N, 2)`` arrays, one row per ``(start_s, stop_s)``."""

    from vio.models.pose_geometry import quaternion_to_matrix_np

    times = attitude.times_s
    start = np.clip(np.asarray(start_s, dtype=np.float64).reshape(-1), times[0], times[-1])
    stop = np.clip(np.asarray(stop_s, dtype=np.float64).reshape(-1), times[0], times[-1])
    first = quaternion_to_matrix_np(quaternion_at(times, attitude.quaternion, start))
    second = quaternion_to_matrix_np(quaternion_at(times, attitude.quaternion, stop))
    relative = np.einsum("nji,njk->nik", first, second)
    down = first[:, 2, :]  # R^T e_z: the NED down axis in the body frame
    altitude = np.stack(
        (np.interp(start, times, attitude.altitude_m), np.interp(stop, times, attitude.altitude_m)),
        axis=1,
    )
    return {
        "relative_rotation": relative.astype(np.float32),
        "down_body": down.astype(np.float32),
        "altitude_m": altitude.astype(np.float32),
    }


def pair_geometry(
    attitude: AttitudeAltitude, start_s: float, stop_s: float
) -> Dict[str, np.ndarray]:
    """What a flat-ground frontend needs to know about one image pair.

    * ``relative_rotation`` ``(3, 3)``: the second exposure's body orientation
      in the first exposure's body frame, ``R_0^T R_1``. EXACT, from the two
      interpolated attitudes - never ``rate * dt``, which is only the first
      term of the rotation's expansion and at a one-second baseline in a turn
      is wrong by more than the translation signal.
    * ``down_body`` ``(3,)``: the NED down axis in the first exposure's body
      frame, i.e. the ground normal before the camera mounting is applied.
      Only roll and pitch reach it - yaw drops out, as it must.
    * ``altitude_m`` ``(2,)``: altitude at the two exposures. The first is
      the metric scale of the whole pair; the difference is the camera's
      motion along the ground normal, which the image barely sees and the
      altimeter measures directly.

    Both instants are clamped to the log's range, the same hold-at-the-ends
    rule :meth:`vio.data.image_pairs.VisualPairSource.rate_over_exposure`
    applies.
    """

    batch = pair_geometry_batch(attitude, np.asarray([start_s]), np.asarray([stop_s]))
    return {name: value[0] for name, value in batch.items()}


__all__ = [
    "AIDING_CHANNELS",
    "ALTITUDE_CANDIDATES",
    "ATTITUDE_CANDIDATES",
    "AttitudeAltitude",
    "REFERENCE_EULER_COLUMNS",
    "aiding_features",
    "body_rates_from_quaternions",
    "detect_euler_unit",
    "hold_fraction",
    "load_attitude_altitude",
    "pair_geometry",
    "pair_geometry_batch",
    "quaternion_at",
    "resolve_altitude_column",
    "resolve_attitude_columns",
]
