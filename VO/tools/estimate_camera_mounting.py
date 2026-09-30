#!/usr/bin/env python3
"""Measure which way the camera points on the airframe, from the images.

The flat-ground frontend (``--frontend planar``) removes rotation EXACTLY and
turns image motion into metres per second in closed form, which is only
possible with the camera-to-body rotation in hand. A nadir camera has its
optical axis along body down, but it can be bolted at any rotation about that
axis - image top forward, image right forward, and so on - and nothing in the
telemetry says which. A wrong choice does not fail loudly: it swaps or negates
the forward and lateral axes, and the network then spends its capacity
learning the mounting back.

This tool measures it. For every candidate mounting the second frame of a pair
is de-rotated with that mounting's exact rotation homography - even one degree
of pitch shifts the whole frame by as much as a third of a second of forward
flight at 150 m, so skipping this biases everything below - and the ground
that remains simply slides through the frame, opposite to the aircraft's
horizontal velocity. Phase correlation over the central part of the image
measures that slide; the reference velocity, the altitude and the intrinsics
predict it. The mounting whose prediction matches best wins, and the residual
angle between measured and predicted motion under it is the small yaw
misalignment of the actual mount.

It also reports the RATIO of measured to predicted motion. That is a free,
independent check of three things the metric scale depends on at once: the
altitude being height above the ground (``RelativeAlt``), the focal length,
and the camera clock. A ratio near 1.0 says all three are right; 0.9 says the
speed will come out 10% low, for one of those three reasons.

The reference velocity is read here ONLY as a yardstick for a one-off
calibration, exactly as ``tools/check_dataset_sync.py`` does; it never reaches
a model.

    python tools/estimate_camera_mounting.py --dataset data \\
        --calibration configs/vo/camera_fixedwing.json \\
        --output artifacts/camera_mounting.json

Paste the printed ``mounting`` block into the calibration file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
# The package lives under src/; tools/ is imported as a package from the
# repository root. Both have to be importable when a script is run directly.
for _entry in (ROOT / "src", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import numpy as np

from vio.data.attitude import load_attitude_altitude, pair_geometry
from vio.data.calibration import load_camera_calibration
from vio.data.fixedwing_vo import reference_body_velocity
from vio.data.image_pairs import VisualPairSource
from vio.models.planar_geometry import NADIR_MOUNTINGS


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--csv-name", default="flight.csv")
    parser.add_argument("--image-folder", default="images")
    parser.add_argument("--time-column", default="Time")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--altitude-column", default=None)
    parser.add_argument("--image-time-offset", type=float, default=0.0)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=(576, 1024),
                        metavar=("H", "W"))
    parser.add_argument(
        "--frame-gap", type=int, default=6,
        help="Frames between the two images. Large enough that the ground "
             "moves a few tens of pixels, small enough that phase correlation "
             "still sees one dominant shift. Default 6 (0.3 s at 20 Hz).",
    )
    parser.add_argument("--pairs", type=int, default=300,
                        help="Candidate pairs sampled evenly over the flight.")
    parser.add_argument("--max-rotation-deg", type=float, default=5.0,
                        help="Keep only pairs that rotate less than this (the "
                             "rotation is removed, but a large one leaves less "
                             "overlap to correlate).")
    parser.add_argument("--crop", type=float, default=0.6,
                        help="Central fraction of the frame to phase-correlate: "
                             "the de-rotated frame has empty borders.")
    parser.add_argument("--min-speed", type=float, default=8.0,
                        help="Keep only pairs flown faster than this (m/s).")
    parser.add_argument("--min-response", type=float, default=0.05,
                        help="Minimum phase-correlation peak response to trust a shift.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def phase_shift(first: np.ndarray, second: np.ndarray):
    """Whole-image translation of ``second`` relative to ``first``, in pixels.

    Positive x means the content moved to the right between the frames.
    Returns ``(dx, dy, response)``; a Hanning window keeps the frame edges from
    dominating the spectrum.
    """

    import cv2

    window = cv2.createHanningWindow((first.shape[1], first.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(
        first.astype(np.float32), second.astype(np.float32), window
    )
    return float(dx), float(dy), float(response)


def wrap_degrees(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def circular_median_degrees(angles: np.ndarray) -> float:
    """A robust centre for angles: the median of the wrapped deviations from
    the circular mean, added back to it."""

    radians = np.radians(angles)
    mean = math.degrees(math.atan2(np.sin(radians).mean(), np.cos(radians).mean()))
    return float(mean + np.median(wrap_degrees(np.asarray(angles) - mean)))


def estimate_mounting(
    root: Path,
    *,
    attitude,
    times: np.ndarray,
    velocity_body: np.ndarray,
    calibration,
    image_size: Sequence[int],
    image_folder: str = "images",
    image_time_offset=0.0,
    frame_gap: int = 6,
    pairs: int = 300,
    max_rotation_deg: float = 5.0,
    min_speed: float = 8.0,
    min_response: float = 0.05,
    crop: float = 0.6,
) -> Dict[str, object]:
    """The measurement behind this tool, callable from the trainer.

    Returns the report dict :func:`main` prints and writes: the best nadir
    mounting, its residual yaw misalignment, the measured/predicted motion
    ratio, and the refined ``camera_from_body`` matrix.
    """

    import cv2

    image_size = (int(image_size[0]), int(image_size[1]))
    source = VisualPairSource(
        root, attitude.times_s,
        image_folder=image_folder,
        image_time_offset_s=image_time_offset,
        frame_gap=int(frame_gap),
        image_size=image_size,
        grayscale=True,
        camera_matrix=calibration.camera_matrix,
        calibration_image_size=calibration.native_size,
        images_rectified=calibration.images_rectified,
        distortion=calibration.distortion if calibration.distortion.size else None,
        deployment_latency_s=0.0,
    )
    working = source.camera_matrix
    plan = source.plan
    total = int(plan.ready_tick.size)
    if total == 0:
        raise SystemExit("no image pairs overlap the telemetry")
    events = np.unique(np.linspace(0, total - 1, min(pairs, total)).round().astype(np.int64))

    names = sorted(NADIR_MOUNTINGS)
    matrices = {name: np.asarray(NADIR_MOUNTINGS[name], dtype=np.float64) for name in names}
    inverse = np.linalg.inv(working)
    height_px, width_px = image_size
    crop_h = max(int(height_px * crop) // 2 * 2, 16)
    crop_w = max(int(width_px * crop) // 2 * 2, 16)
    top = (height_px - crop_h) // 2
    left = (width_px - crop_w) // 2
    # Normalized coordinate of the crop centre: the prediction is made there.
    centre = inverse @ np.array([left + (crop_w - 1) / 2.0, top + (crop_h - 1) / 2.0, 1.0])

    measured: Dict[str, List[np.ndarray]] = {name: [] for name in names}
    predicted: Dict[str, List[np.ndarray]] = {name: [] for name in names}
    skipped = {"rotation": 0, "speed": 0, "response": 0}
    for event in events:
        t0 = float(plan.exposure_t0_s[event])
        t1 = float(plan.exposure_t1_s[event])
        geometry = pair_geometry(attitude, t0, t1)
        rotation = geometry["relative_rotation"].astype(np.float64)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0))))
        if angle > max_rotation_deg:
            skipped["rotation"] += 1
            continue
        mid = 0.5 * (t0 + t1)
        velocity = np.array([np.interp(mid, times, velocity_body[:, k]) for k in range(3)])
        if float(np.hypot(velocity[0], velocity[1])) < min_speed:
            skipped["speed"] += 1
            continue
        first, second = source.load_pair(int(event))
        image0 = first[0].numpy().astype(np.float32) / 255.0
        image1 = second[0].numpy().astype(np.float32) / 255.0
        height = float(geometry["altitude_m"][0])
        down = geometry["down_body"].astype(np.float64)
        dt = t1 - t0
        results = {}
        for name in names:
            mount = matrices[name]
            rotation_camera = mount @ rotation @ mount.T
            # Pixel of frame 0 -> pixel of frame 1 for the rotation alone;
            # WARP_INVERSE_MAP makes warpPerspective read image1 at H p.
            homography = working @ rotation_camera.T @ inverse
            derotated = cv2.warpPerspective(
                image1, homography, (width_px, height_px),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
            )
            dx, dy, response = phase_shift(
                image0[top:top + crop_h, left:left + crop_w],
                derotated[top:top + crop_h, left:left + crop_w],
            )
            normal = mount @ down
            u = (mount @ velocity) * dt / max(height, 1.0)
            s_centre = max(float(normal @ centre), 0.05)
            denominator = max(1.0 - s_centre * u[2], 0.05)
            shift = s_centre * (u[2] * centre[:2] - u[:2]) / denominator
            results[name] = (np.array([dx / working[0, 0], dy / working[1, 1]]), shift, response)
        if max(r[2] for r in results.values()) < min_response:
            skipped["response"] += 1
            continue
        for name, (shift_measured, shift_predicted, _) in results.items():
            measured[name].append(shift_measured)
            predicted[name].append(shift_predicted)

    used = len(measured[names[0]])
    if used < 10:
        raise SystemExit(
            f"only {used} usable pairs (skipped {skipped}); relax "
            "--max-rotation-deg / --min-response or raise --pairs"
        )
    scores: Dict[str, float] = {}
    for name in names:
        m = np.asarray(measured[name])
        q = np.asarray(predicted[name])
        scores[name] = float(np.median(
            np.linalg.norm(m - q, axis=1) / np.maximum(np.linalg.norm(q, axis=1), 1e-12)
        ))
    best = min(scores, key=scores.get)
    measured_array = np.asarray(measured[best])
    predicted_array = np.asarray(predicted[best])
    angle_measured = np.degrees(np.arctan2(measured_array[:, 1], measured_array[:, 0]))
    angle_predicted = np.degrees(np.arctan2(predicted_array[:, 1], predicted_array[:, 0]))
    offsets = wrap_degrees(angle_measured - angle_predicted)
    misalignment = circular_median_degrees(offsets)
    ratio = np.linalg.norm(measured_array, axis=1) / np.maximum(
        np.linalg.norm(predicted_array, axis=1), 1e-12
    )
    best_matrix = matrices[best]
    # Fold the measured misalignment in as a rotation about the optical axis.
    c, s_ = math.cos(math.radians(misalignment)), math.sin(math.radians(misalignment))
    about_optical_axis = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    refined = about_optical_axis @ best_matrix
    spread = float(np.median(np.abs(wrap_degrees(offsets - misalignment))))

    return {
        "dataset": str(root),
        "pairs_used": int(used),
        "pairs_skipped": skipped,
        "frame_gap": int(frame_gap),
        "best_mounting": best,
        "relative_error_per_mounting": scores,
        "yaw_misalignment_deg": misalignment,
        "direction_spread_deg": spread,
        "scale_ratio_median": float(np.median(ratio)),
        "scale_ratio_p25_p75": [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
        "camera_from_body": refined.round(6).tolist(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.dataset.expanduser().resolve()
    csv_path = root / args.csv_name
    attitude = load_attitude_altitude(
        csv_path, time_column=args.time_column, time_scale=args.time_scale,
        altitude_column=args.altitude_column,
    )
    times, velocity_body = reference_body_velocity(
        csv_path, time_column=args.time_column, time_scale=args.time_scale
    )
    report = estimate_mounting(
        root,
        attitude=attitude,
        times=times,
        velocity_body=velocity_body,
        calibration=load_camera_calibration(args.calibration),
        image_size=args.image_size,
        image_folder=args.image_folder,
        image_time_offset=float(args.image_time_offset),
        frame_gap=args.frame_gap,
        pairs=args.pairs,
        max_rotation_deg=args.max_rotation_deg,
        min_speed=args.min_speed,
        min_response=args.min_response,
        crop=args.crop,
    )
    used = report["pairs_used"]
    skipped = report["pairs_skipped"]
    best = report["best_mounting"]
    scores = report["relative_error_per_mounting"]
    misalignment = report["yaw_misalignment_deg"]
    spread = report["direction_spread_deg"]
    refined = np.asarray(report["camera_from_body"])
    print(f"pairs used {used} (skipped {skipped})")
    print(f"best nadir mounting: {best}  (median relative error per mounting: "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(scores.items())) + ")")
    print(f"residual yaw misalignment {misalignment:+.2f} deg "
          f"(pair-to-pair spread {spread:.2f} deg)")
    print(f"measured / predicted motion: {report['scale_ratio_median']:.3f} "
          f"(IQR {report['scale_ratio_p25_p75'][0]:.3f}-{report['scale_ratio_p25_p75'][1]:.3f})")
    if abs(report["scale_ratio_median"] - 1.0) > 0.05:
        print("  WARNING: more than 5% off. Check that the altitude column is height "
              "above the GROUND under the aircraft, the focal length, and the image "
              "time offset (tools/estimate_time_offset.py) before training.")
    print("\nmounting block for the calibration file:")
    print(json.dumps({"mounting": {"camera_from_body": refined.round(6).tolist()}}, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
