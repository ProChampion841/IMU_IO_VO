#!/usr/bin/env python3
"""Generate a physically self-consistent synthetic fixed-wing flight.

The recorded flight is 60 seconds long, which leaves roughly 32 heavily
overlapping training windows. That is enough to exercise the code paths but
far too little to tell a working estimator from a broken one. This tool builds
an arbitrarily long flight where every quantity is derived from one analytic
trajectory, so the pipeline can be validated end to end against a truth that
is known exactly rather than inferred.

What makes the result a fair VIO test rather than a plumbing stub is that the
images are *rendered from the same trajectory that produces the telemetry*.
A downward-looking pinhole camera observes a textured ground plane, so each
frame pair contains the real optical flow induced by the aircraft's motion.
A frontend that matches those frames recovers genuine motion evidence, and a
model that ignores the images cannot reach the same accuracy.

Consistency chain, in the order the pipeline consumes it:

    heading / flight-path angle / speed   (smooth analytic functions of time)
      -> NED velocity
      -> NED position          (trapezoid integral, the same rule the loader
                                uses to derive its pose reference, so the
                                derived reference reproduces the truth)
      -> NED acceleration      (analytic derivative of the velocity)
      -> attitude              (heading and climb from the path, bank from the
                                coordinated-turn relation)
      -> gyro                  (relative rotation between consecutive samples)
      -> accelerometer         (specific force f = R^T (a - g), reported in g)
      -> images                (ground plane seen through the camera pose)

Written outputs match the real dataset layout, so every downstream tool works
without modification:

    <output>/flight.csv          telemetry at the IMU rate
    <output>/images/<ms>.jpg     frames at the image rate
    <output>/calibration.json    exact intrinsics/extrinsics for this render
    <output>/truth.npz           ground-truth arrays for verification
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The package lives under src/; tools/ is imported as a package from the
# repository root. Both have to be importable when a script is run directly.
for _entry in (ROOT / "src", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from vio.models.pose_geometry import (  # noqa: E402
    euler_zyx_to_quaternion_np,
    quaternion_conjugate_np,
    quaternion_multiply_np,
    quaternion_to_matrix_np,
    quaternion_to_rotvec_np,
)

GRAVITY_M_S2 = 9.80665

# The exact column list of the recorded flight, so the synthetic CSV is a
# drop-in replacement. Channels this generator does not model are zero.
CSV_COLUMNS = (
    "NanoTimeStamp,Time,Index,GyroX,GyroY,GyroZ,AcclX,AcclY,AcclZ,MagX,MagY,MagZ,Barometer,RelativeAlt,EulX,EulY,EulZ,Temperature,Pressure,BatteryPro,BatteryOrin,Satellites,GpsFreq,Throttle,LeftElevator,RightElevator,Rudder,WindSpeed,NavVeX,NavVeY,navVeZ,NavEulX,NavEulY,NavEulZ,NavVnX,NavVnY,NavVnZ,VelocityN,VelocityE,VoVelocityX,VoVelocityY,VoVelocityZ,OffsetRoll,OffsetPitch,OffsetYaw,LeaderVelocity,AirIoVelocityX,AirIoVelocityY,AirIoVelocityZ,NavMode,GuideMode,DroneState,WpIndex,fVgSet,VelAngle,GPSNavVnX,GPSNavVnY,GPSNavVnZ,GPSNavEulX,GPSNavEulY,GPSNavEulZ,VoVelocity,VoYaw,TargetX,TargetY,TargetWidth,TargetHeight,ModelDelay,TargetType,TrackFailed,intCount,voVelYaw,voVelYaw1,voVelPitch,voVelPitch1,LosYaw,LosPitch,dbControl0,dbControl1,left,right,rudder,throttle,deltaYawSum,gpsYaw,gpsPitch,GuideMode,DeltaYaw,DeltaYawLimit,pixel,GPSVel,AirSpeed,Vo60Vel,Vo0Vel,LOSB_first,LOSN_first,prevNavData_yaw,pitchChangeFlag,yawChangeFlag,ChakBalState1,ChakBalState2,ChakBalState3,ChakbalState4,LeftTime,EstimatedLosYaw,EstimatedLosPitch,ControlMode,RLMode,VPSNavMode,MagEnable,WMMYaw,MX,MY,MZ,Mag_X,Mag_Y,Mag_Z"
).split(",")


#: Height of the synthetic airfield above sea level. Only used to make
#: Barometer differ from RelativeAlt, so a fallback to the wrong altitude
#: column cannot pass unnoticed.
FIELD_ELEVATION_M = 500.0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset_synthetic"))
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--imu-rate-hz", type=float, default=100.0)
    parser.add_argument("--image-rate-hz", type=float, default=20.0)
    parser.add_argument("--start-time-s", type=float, default=1000.0)
    parser.add_argument("--altitude-m", type=float, default=120.0)
    parser.add_argument("--speed-m-s", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--turn-period-s",
        type=float,
        default=0.0,
        help="Seconds per full 360-degree turn, superposed on the S-turns. "
             "0 (the default) holds a straight mean course, which over half "
             "an hour covers ~50 km and collapses a top-down trajectory plot "
             "to a single line. 600 flies ~4.8 km circuits instead, which is "
             "what makes a drift figure readable.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Write telemetry only. Much faster when the images are not needed.",
    )
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--focal-px", type=float, default=500.0)
    parser.add_argument(
        "--ground-metres-per-texel",
        type=float,
        default=0.12,
        help="Ground sampling of the synthetic terrain texture.",
    )
    parser.add_argument("--texture-size", type=int, default=4096)
    # A noiseless IMU makes the benchmark meaningless: dead reckoning is then
    # exact, so there is nothing for vision to correct and a fusion model
    # cannot beat an inertial one. These defaults are representative of a
    # consumer MEMS unit.
    parser.add_argument(
        "--angle-of-attack-deg", type=float, default=4.0,
        help="Incidence at the reference speed in level flight, scaled by "
             "V^-2 and by load factor. Puts real content in the body z axis "
             "of the target. Zero reproduces the old degenerate capture whose "
             "target was exactly (|v|, 0, 0).",
    )
    parser.add_argument(
        "--sideslip-deg", type=float, default=1.5,
        help="Peak sideslip from imperfect turn coordination. Puts real "
             "content in the body y axis. This is NOT wind: the log's "
             "WindSpeed column is not read by anything in this project.",
    )
    parser.add_argument(
        "--nav-attitude-noise-deg", type=float, default=0.2,
        help="Slow attitude offset between NavEul* (what the VO estimator "
             "reads) and GPSNavEul* (which defines the target). Zero makes "
             "them identical, which is the target-leakage case: useful as a "
             "deliberate ablation, wrong as a default.",
    )
    parser.add_argument(
        "--gyro-bias-deg-s",
        type=float,
        default=0.5,
        help="Constant per-flight gyro bias magnitude. 0 disables it.",
    )
    parser.add_argument(
        "--gyro-noise-deg-s",
        type=float,
        default=0.05,
        help="Per-sample white gyro noise, standard deviation.",
    )
    parser.add_argument(
        "--accel-bias-m-s2",
        type=float,
        default=0.05,
        help="Constant per-flight accelerometer bias magnitude. 0 disables it.",
    )
    parser.add_argument(
        "--accel-noise-m-s2",
        type=float,
        default=0.02,
        help="Per-sample white accelerometer noise, standard deviation.",
    )
    parser.add_argument(
        "--perfect-imu",
        action="store_true",
        help=(
            "Emit an exact, noiseless IMU. Use only to verify the physics "
            "chain; it makes dead reckoning exact and the task degenerate."
        ),
    )
    args = parser.parse_args(argv)
    if args.duration_s <= 1.0:
        parser.error("duration-s must exceed one second")
    if args.imu_rate_hz <= 0 or args.image_rate_hz <= 0:
        parser.error("rates must be positive")
    ratio = args.imu_rate_hz / args.image_rate_hz
    if abs(ratio - round(ratio)) > 1e-9:
        parser.error("imu-rate-hz must be an integer multiple of image-rate-hz")
    if args.altitude_m <= 10.0:
        parser.error("altitude-m must exceed ten metres for a sane ground render")
    if args.speed_m_s <= 0:
        parser.error("speed-m-s must be positive")
    if min(
        args.gyro_bias_deg_s,
        args.gyro_noise_deg_s,
        args.accel_bias_m_s2,
        args.accel_noise_m_s2,
    ) < 0:
        parser.error("IMU bias and noise magnitudes cannot be negative")
    if args.perfect_imu:
        args.gyro_bias_deg_s = 0.0
        args.gyro_noise_deg_s = 0.0
        args.accel_bias_m_s2 = 0.0
        args.accel_noise_m_s2 = 0.0
    return args


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


def _random_vector(rng: np.random.Generator, magnitude: float) -> np.ndarray:
    """A fixed offset of the requested magnitude in a random direction."""

    if magnitude <= 0.0:
        return np.zeros(3)
    direction = rng.normal(size=3)
    return direction / np.linalg.norm(direction) * magnitude


def build_trajectory(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    """Analytic fixed-wing motion and every telemetry channel derived from it.

    Heading, flight-path angle and speed are smooth sums of sinusoids chosen to
    produce sustained turns, climbs and descents at bank angles a fixed-wing
    aircraft actually flies, rather than motion that is trivially predictable
    from the previous sample.
    """

    dt = 1.0 / args.imu_rate_hz
    count = int(round(args.duration_s * args.imu_rate_hz))
    t = np.arange(count, dtype=np.float64) * dt

    # Heading: two incommensurate S-turn periods so the path never repeats,
    # optionally riding on a constant turn rate so the aircraft circuits
    # instead of holding one mean course.
    turn_rate = 0.0 if args.turn_period_s <= 0.0 else 2 * np.pi / args.turn_period_s
    heading = turn_rate * t + 0.45 * np.sin(2 * np.pi * t / 37.0) + 0.30 * np.sin(
        2 * np.pi * t / 13.0 + 1.1
    )
    heading_rate = turn_rate + (0.45 * 2 * np.pi / 37.0) * np.cos(
        2 * np.pi * t / 37.0
    ) + (0.30 * 2 * np.pi / 13.0) * np.cos(2 * np.pi * t / 13.0 + 1.1)
    heading_accel = -(0.45 * (2 * np.pi / 37.0) ** 2) * np.sin(
        2 * np.pi * t / 37.0
    ) - (0.30 * (2 * np.pi / 13.0) ** 2) * np.sin(2 * np.pi * t / 13.0 + 1.1)
    # Yaw is reported wrapped, as a real navigation filter reports it. Only
    # sin/cos of the heading is ever used below, so the wrap changes nothing
    # downstream; it keeps the logged NavEulZ inside the range real data has.
    heading = np.arctan2(np.sin(heading), np.cos(heading))

    # Flight-path angle: gentle climbs and descents, a few degrees.
    gamma = 0.06 * np.sin(2 * np.pi * t / 23.0 + 0.4)
    gamma_rate = (0.06 * 2 * np.pi / 23.0) * np.cos(2 * np.pi * t / 23.0 + 0.4)
    gamma_accel = -(0.06 * (2 * np.pi / 23.0) ** 2) * np.sin(
        2 * np.pi * t / 23.0 + 0.4
    )

    # Airspeed variation, as an aircraft trading energy in climbs and turns.
    speed = args.speed_m_s + 2.5 * np.sin(2 * np.pi * t / 19.0 + 0.7)
    speed_rate = (2.5 * 2 * np.pi / 19.0) * np.cos(2 * np.pi * t / 19.0 + 0.7)

    cos_g, sin_g = np.cos(gamma), np.sin(gamma)
    cos_h, sin_h = np.cos(heading), np.sin(heading)

    velocity = np.stack(
        (speed * cos_g * cos_h, speed * cos_g * sin_h, -speed * sin_g), axis=1
    )

    # Analytic derivative of the velocity above, term by term.
    d_north = (
        speed_rate * cos_g * cos_h
        - speed * sin_g * gamma_rate * cos_h
        - speed * cos_g * sin_h * heading_rate
    )
    d_east = (
        speed_rate * cos_g * sin_h
        - speed * sin_g * gamma_rate * sin_h
        + speed * cos_g * cos_h * heading_rate
    )
    d_down = -(speed_rate * sin_g + speed * cos_g * gamma_rate)
    acceleration = np.stack((d_north, d_east, d_down), axis=1)

    # Trapezoid integration matches derive_pose_reference, so the reference the
    # loader builds from GPSNavVn reproduces this position exactly.
    position = np.zeros_like(velocity)
    position[1:] = np.cumsum(0.5 * (velocity[:-1] + velocity[1:]) * dt, axis=0)
    position[:, 2] -= args.altitude_m

    # Coordinated turn: the bank angle that produces the observed turn rate.
    roll = np.arctan2(speed * heading_rate * cos_g, GRAVITY_M_S2)

    # Angle of attack and sideslip. Without these the body x axis lies exactly
    # along the velocity vector, the body-frame target collapses to (|v|, 0, 0),
    # and two of the three supervised axes carry nothing - which makes the
    # lateral capability the model exists for impossible to train or measure.
    # Neither is wind: alpha is the incidence a wing needs to make lift, and
    # beta is imperfect turn coordination. The log's WindSpeed column is not
    # read here or anywhere else.
    #
    # Alpha rises as lift demand rises: inversely with dynamic pressure (V^2)
    # and with the load factor a bank angle costs.
    load_factor = 1.0 / np.clip(np.cos(roll), 0.2, 1.0)
    alpha = (
        np.deg2rad(args.angle_of_attack_deg)
        * (args.speed_m_s / speed) ** 2
        * load_factor
    )
    # Beta is small and slow: a coordinated aircraft holds it near zero, and a
    # real one wanders either side of it.
    beta = np.deg2rad(args.sideslip_deg) * np.sin(2 * np.pi * t / 31.0 + 1.1)

    # Nose above the flight path by alpha, nose off the track by beta.
    #
    # These are the COMMANDED angles, and in a bank they are not exactly the
    # angles that come out. Roll is applied last in the ZYX order, so it mixes
    # the body y and z components: at 20 degrees of bank a commanded 1.5 deg of
    # sideslip is measured as roughly 2.5 deg, with some of the incidence
    # rotated into it. That is left alone deliberately. The point of these two
    # numbers is to make the body-frame target carry real lateral and vertical
    # content in a realistic band, not to hit an exact incidence - and the
    # achieved values (alpha ~3-5 deg, beta within ~4 deg) are what a fixed
    # wing actually flies. What must be exact is the round trip: the loader
    # rebuilds the target from the same rotation, and the test asserts that.
    pitch = gamma + alpha
    yaw = heading - beta
    euler = np.stack((roll, pitch, yaw), axis=1)
    quaternion = euler_zyx_to_quaternion_np(euler)
    for index in range(1, quaternion.shape[0]):
        if float(np.dot(quaternion[index - 1], quaternion[index])) < 0.0:
            quaternion[index] *= -1.0
    rotation = quaternion_to_matrix_np(quaternion)

    # Gyro: the instantaneous body angular rate at each sample time, which is
    # what a real rate gyro measures.
    #
    # Two things go wrong if this is instead the finite difference between
    # consecutive attitudes. First it is not causal: the pose target at row i
    # covers the interval [i-1, i], so a difference-derived gyro at row i would
    # describe [i, i+1] and hand the model the future. Second it is degenerate:
    # the rotation target would equal gyro * dt exactly, so the model could
    # copy its input instead of integrating, and the benchmark would report a
    # rotation accuracy that means nothing.
    #
    # omega is recovered from the analytic attitude by a centred difference,
    # which is second-order accurate and uses no information the sample time
    # does not already have.
    body_rate = np.zeros_like(position)
    centred = quaternion_multiply_np(
        quaternion_conjugate_np(quaternion[:-2]), quaternion[2:]
    )
    body_rate[1:-1] = quaternion_to_rotvec_np(centred) / (2.0 * dt)
    body_rate[0] = body_rate[1]
    body_rate[-1] = body_rate[-2]

    # Accelerometer measures specific force, not acceleration: f = a - g.
    # In NED, gravity points along +D, so level flight reads -1 g on the z axis.
    gravity_ned = np.asarray([0.0, 0.0, GRAVITY_M_S2])
    specific_force = np.einsum(
        "nij,nj->ni", np.swapaxes(rotation, 1, 2), acceleration - gravity_ned
    )

    # Corrupt the IMU exactly as a real one is corrupted: a constant bias
    # drawn once per flight, plus per-sample white noise. Both are applied
    # after the exact signals are formed, and the exact ones are returned
    # alongside so the physics chain stays verifiable.
    rng = np.random.default_rng(args.seed + 991)
    gyro_bias = _random_vector(rng, np.deg2rad(args.gyro_bias_deg_s))
    accel_bias = _random_vector(rng, args.accel_bias_m_s2)
    measured_body_rate = (
        body_rate
        + gyro_bias
        + rng.normal(0.0, np.deg2rad(args.gyro_noise_deg_s), size=body_rate.shape)
    )
    measured_specific_force = (
        specific_force
        + accel_bias
        + rng.normal(0.0, args.accel_noise_m_s2, size=specific_force.shape)
    )

    _ = heading_accel, gamma_accel  # documented above; not needed downstream
    return {
        "time_s": args.start_time_s + t,
        "gyro_bias_rad_s": gyro_bias,
        "accel_bias_m_s2": accel_bias,
        "gyro_rad_s_exact": body_rate,
        "specific_force_m_s2_exact": specific_force,
        "dt_s": dt,
        "position_ned_m": position,
        "velocity_ned_m_s": velocity,
        "acceleration_ned_m_s2": acceleration,
        "euler_rad": euler,
        "angle_of_attack_rad": alpha,
        "sideslip_rad": beta,
        "quaternion_body_to_ned": quaternion,
        "rotation_body_to_ned": rotation,
        "gyro_rad_s": measured_body_rate,
        "specific_force_m_s2": measured_specific_force,
        "airspeed_m_s": speed,
    }


def write_csv(
    path: Path,
    trajectory: Dict[str, np.ndarray],
    *,
    nav_attitude_noise_deg: float = 0.2,
    seed: int = 0,
) -> None:
    index = {name: position for position, name in enumerate(CSV_COLUMNS)}
    gyro_deg = np.rad2deg(trajectory["gyro_rad_s"])
    accel_g = trajectory["specific_force_m_s2"] / GRAVITY_M_S2
    velocity = trajectory["velocity_ned_m_s"]
    euler = trajectory["euler_rad"]
    # A separate attitude solution for the estimator to read. The offset is
    # small and slowly varying rather than white, because a navigation filter's
    # error is correlated in time - white noise would average away over a
    # window and understate what de-rotation actually has to survive.
    generator = np.random.default_rng(seed + 7717)
    walk = generator.standard_normal((euler.shape[0], 3)).cumsum(axis=0)
    walk = walk / max(float(np.abs(walk).max()), 1e-9)
    nav_euler = euler + np.deg2rad(float(nav_attitude_noise_deg)) * walk
    altitude = -trajectory["position_ned_m"][:, 2]
    times = trajectory["time_s"]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        row = [0.0] * len(CSV_COLUMNS)
        for sample in range(times.size):
            for position in range(len(row)):
                row[position] = 0.0
            row[index["NanoTimeStamp"]] = float(sample) * trajectory["dt_s"]
            row[index["Time"]] = float(times[sample])
            row[index["Index"]] = float(sample)
            row[index["GyroX"]] = float(gyro_deg[sample, 0])
            row[index["GyroY"]] = float(gyro_deg[sample, 1])
            row[index["GyroZ"]] = float(gyro_deg[sample, 2])
            row[index["AcclX"]] = float(accel_g[sample, 0])
            row[index["AcclY"]] = float(accel_g[sample, 1])
            row[index["AcclZ"]] = float(accel_g[sample, 2])
            # RelativeAlt is height above the takeoff datum -- the h the
            # speed head multiplies by in v = h*u, and the column a real
            # capture resolves. Barometer carries the same height plus a
            # field elevation, i.e. a pressure altitude above sea level.
            # They differ ON PURPOSE: if the resolver ever falls through to
            # Barometer, every predicted speed is wrong by the ratio of the
            # two -- loudly -- instead of the synthetic capture quietly
            # agreeing with a bug that a real capture would not.
            row[index["RelativeAlt"]] = float(altitude[sample])
            row[index["Barometer"]] = float(altitude[sample]) + FIELD_ELEVATION_M
            row[index["EulX"]] = float(np.rad2deg(euler[sample, 0]))
            row[index["EulY"]] = float(np.rad2deg(euler[sample, 1]))
            row[index["EulZ"]] = float(np.rad2deg(euler[sample, 2]))
            row[index["GPSNavVnX"]] = float(velocity[sample, 0])
            row[index["GPSNavVnY"]] = float(velocity[sample, 1])
            row[index["GPSNavVnZ"]] = float(velocity[sample, 2])
            # The navigation filter's own attitude - what a VO estimator is
            # allowed to read. Deliberately NOT bit-identical to the reference
            # below: if the two matched exactly, every attitude error would
            # cancel against the target and the flight would be testing the
            # leakage case rather than the estimator.
            row[index["NavEulX"]] = float(nav_euler[sample, 0])
            row[index["NavEulY"]] = float(nav_euler[sample, 1])
            row[index["NavEulZ"]] = float(nav_euler[sample, 2])
            row[index["GPSNavEulX"]] = float(euler[sample, 0])
            row[index["GPSNavEulY"]] = float(euler[sample, 1])
            row[index["GPSNavEulZ"]] = float(euler[sample, 2])
            row[index["AirSpeed"]] = float(trajectory["airspeed_m_s"][sample])
            writer.writerow(["%.6f" % value for value in row])


# ---------------------------------------------------------------------------
# Ground texture and rendering
# ---------------------------------------------------------------------------


def build_ground_texture(size: int, seed: int) -> np.ndarray:
    """A tiling terrain texture with corners a keypoint detector can find.

    Uniform noise gives a detector nothing stable to latch onto across frames,
    so the texture is built from overlapping shapes at several scales. That
    yields repeatable corners and edges, which is what makes frame-to-frame
    matching meaningful rather than luck.
    """

    import cv2

    rng = np.random.default_rng(seed)
    image = np.full((size, size), 128.0, dtype=np.float32)

    # Large fields, then smaller structures on top of them.
    for scale, count in ((size // 8, 90), (size // 20, 400), (size // 60, 1500)):
        for _ in range(count):
            x = int(rng.integers(0, size))
            y = int(rng.integers(0, size))
            w = int(rng.integers(max(4, scale // 3), max(6, scale)))
            h = int(rng.integers(max(4, scale // 3), max(6, scale)))
            value = float(rng.uniform(40.0, 215.0))
            if rng.random() < 0.35:
                cv2.circle(image, (x, y), max(2, w // 2), value, -1)
            else:
                angle = float(rng.uniform(0.0, 180.0))
                box = cv2.boxPoints(((x, y), (w, h), angle)).astype(np.int32)
                cv2.fillPoly(image, [box], value)

    # Fine grain keeps the gradient non-degenerate between the shapes.
    image += rng.normal(0.0, 6.0, size=image.shape).astype(np.float32)
    image = cv2.GaussianBlur(image, (0, 0), 1.1)
    return np.clip(image, 0, 255).astype(np.uint8)


def render_frames(
    args: argparse.Namespace,
    trajectory: Dict[str, np.ndarray],
    image_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render the ground plane from each camera pose.

    The camera is nadir-looking and aligned with the body axes, so a ray
    through pixel (u, v) leaves the camera along ``R (K^-1 [u v 1])`` in NED
    and meets the ground plane ``D = 0`` at a single point. Mapping every
    pixel back to that point and sampling the terrain texture produces the
    exact image the modelled camera would record from that pose.
    """

    import cv2

    image_dir.mkdir(parents=True, exist_ok=True)
    # Clear frames from any previous capture BEFORE rendering. A re-run only
    # overwrites the filenames it happens to reuse, so writing a new flight over
    # an old one at a different rate - or stopping a re-run part way - leaves a
    # directory holding two different flights whose frames interleave by
    # timestamp. Nothing downstream can detect that: the images are all valid
    # JPEGs at plausible times, they simply show a trajectory the CSV does not
    # describe. Observed exactly once, and once is enough.
    stale = sorted(image_dir.glob("*.jpg"))
    if stale:
        print(f"  clearing {len(stale)} frames from a previous capture")
        for frame in stale:
            frame.unlink()
    texture = build_ground_texture(args.texture_size, args.seed)
    stride = int(round(args.imu_rate_hz / args.image_rate_hz))
    frame_indices = np.arange(0, trajectory["time_s"].size, stride)

    width, height = args.image_width, args.image_height
    camera_matrix = np.asarray(
        [
            [args.focal_px, 0.0, (width - 1) / 2.0],
            [0.0, args.focal_px, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    inverse_camera = np.linalg.inv(camera_matrix)

    grid_u, grid_v = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    pixels = np.stack((grid_u, grid_v, np.ones_like(grid_u)), axis=-1)
    rays_camera = pixels @ inverse_camera.T  # (H, W, 3)

    position = trajectory["position_ned_m"]
    rotation = trajectory["rotation_body_to_ned"]
    texture_height, texture_width = texture.shape[:2]
    written_times = []
    for frame in frame_indices:
        # Camera axes coincide with body axes for this nadir mount, so the
        # body-to-NED rotation also takes camera rays into NED.
        rays_ned = rays_camera @ rotation[frame].T
        down = rays_ned[..., 2]
        if np.any(down <= 1e-6):
            raise RuntimeError(
                f"Frame {frame} has rays that never reach the ground plane; "
                "reduce the attitude excursion or widen the altitude."
            )
        altitude = -position[frame, 2]
        scale = altitude / down
        north = position[frame, 0] + scale * rays_ned[..., 0]
        east = position[frame, 1] + scale * rays_ned[..., 1]

        # Wrap into the texture before handing the maps to remap. OpenCV
        # converts remap coordinates to a fixed-point form whose integer part
        # is a 16-bit short, so anything past 32767 texels silently degrades
        # into aliasing rather than failing. At the default sampling that is
        # only 3.9 km of ground track. Taking the modulo here keeps every
        # coordinate inside one tile; BORDER_WRAP still handles the seam.
        map_x = np.mod(east / args.ground_metres_per_texel, texture_width).astype(
            np.float32
        )
        map_y = np.mod(north / args.ground_metres_per_texel, texture_height).astype(
            np.float32
        )
        frame_image = cv2.remap(
            texture,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        milliseconds = int(round(trajectory["time_s"][frame] * 1000.0))
        cv2.imwrite(
            str(image_dir / f"{milliseconds}.jpg"),
            frame_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        written_times.append(trajectory["time_s"][frame])
    return frame_indices, np.asarray(written_times, dtype=np.float64)


def write_calibration(path: Path, args: argparse.Namespace) -> None:
    """The exact model used by the renderer, not an assumed one.

    The renderer models a nadir camera bolted to the body axes: image x is
    body forward, image y is body right, and the optical axis is body down.
    That mounting is fixed in the renderer rather than configurable, so the
    manifest carries no extrinsics. The camera is a global shutter.
    """

    payload = {
        "schema_version": 1,
        "status": "exact_synthetic_render_parameters",
        "camera": {
            "model": "pinhole_radtan",
            "width": args.image_width,
            "height": args.image_height,
            "fx": args.focal_px,
            "fy": args.focal_px,
            "cx": (args.image_width - 1) / 2.0,
            "cy": (args.image_height - 1) / 2.0,
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "imu": {
            "expected_rate_hz": args.imu_rate_hz,
            "gyro_unit": "degrees_per_second",
            "accel_unit": "g",
            "gravity_m_s2": GRAVITY_M_S2,
            "body_frame": "FRD",
            "gyro_bias": [0.0, 0.0, 0.0],
            "accel_bias": [0.0, 0.0, 0.0],
        },
        "reference": {
            "world_frame": "NED",
            "euler_unit": "radians",
            "euler_order": "roll_pitch_yaw_intrinsic_zyx",
            "orientation_meaning": "body_to_ned",
        },
        "note": (
            "Exact parameters of the synthetic renderer in "
            "tools/make_synthetic_flight.py. Valid only for that generated "
            "flight; it says nothing about any real camera."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    trajectory = build_trajectory(args)
    write_csv(
        output / "flight.csv",
        trajectory,
        nav_attitude_noise_deg=args.nav_attitude_noise_deg,
        seed=args.seed,
    )
    write_calibration(output / "calibration.json", args)

    stride = int(round(args.imu_rate_hz / args.image_rate_hz))
    frame_indices = np.arange(0, trajectory["time_s"].size, stride)
    frame_times = trajectory["time_s"][frame_indices]
    if not args.no_images:
        frame_indices, frame_times = render_frames(
            args, trajectory, output / "images"
        )

    np.savez_compressed(
        output / "truth.npz",
        time_s=trajectory["time_s"],
        position_ned_m=trajectory["position_ned_m"],
        velocity_ned_m_s=trajectory["velocity_ned_m_s"],
        quaternion_body_to_ned=trajectory["quaternion_body_to_ned"],
        euler_rad=trajectory["euler_rad"],
        angle_of_attack_rad=trajectory["angle_of_attack_rad"],
        sideslip_rad=trajectory["sideslip_rad"],
        frame_indices=frame_indices,
        frame_times_s=frame_times,
        gyro_bias_rad_s=trajectory["gyro_bias_rad_s"],
        accel_bias_m_s2=trajectory["accel_bias_m_s2"],
        gyro_rad_s_exact=trajectory["gyro_rad_s_exact"],
        specific_force_m_s2_exact=trajectory["specific_force_m_s2_exact"],
    )

    speed = np.linalg.norm(trajectory["velocity_ned_m_s"], axis=1)
    roll_deg = np.rad2deg(trajectory["euler_rad"][:, 0])
    travelled = float(
        np.linalg.norm(np.diff(trajectory["position_ned_m"], axis=0), axis=1).sum()
    )
    print(f"Wrote {output}")
    print(f"  telemetry rows : {trajectory['time_s'].size}")
    rendered = 0 if args.no_images else frame_times.size
    print(f"  images         : {rendered}")
    print(f"  duration       : {args.duration_s:.1f} s")
    print(f"  ground track   : {travelled:.0f} m")
    print(f"  speed          : {speed.min():.1f} .. {speed.max():.1f} m/s")
    print(f"  bank angle     : {roll_deg.min():+.1f} .. {roll_deg.max():+.1f} deg")
    print(
        "  gyro bias      : "
        f"{np.rad2deg(np.linalg.norm(trajectory['gyro_bias_rad_s'])):.3f} deg/s"
        f"   accel bias: {np.linalg.norm(trajectory['accel_bias_m_s2']):.3f} m/s2"
    )
    print(
        "  altitude       : "
        f"{-trajectory['position_ned_m'][:, 2].max():.1f} .. "
        f"{-trajectory['position_ned_m'][:, 2].min():.1f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
