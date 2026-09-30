#!/usr/bin/env python3
"""How far does the ground move between two frames - and is that enough?

Run this BEFORE choosing ``--frame-gap``. It answers the question the rest of
the pipeline silently depends on: at this capture's altitude, speed and frame
rate, how many correlation cells does the ground move between the two images
of a pair, and how much of that motion is translation (the signal) rather than
rotation (which has to be removed first)?

Why it matters, in one line: sub-cell matching is good to roughly a tenth of a
cell, so the RELATIVE precision of a speed measured from one pair is about
``0.1 / flow_cells``. At 200 m, 20 m/s and 20 Hz one frame of separation is
about 0.66 cells of motion - a 15% measurement before anything else goes
wrong - while twenty frames (one second) is about 13 cells and 1%.

The camera sees the ground move by ``f * v * dt / h`` pixels, so every number
here comes from the telemetry alone: attitude, altitude, the image timestamps
and - for this diagnostic only, never as a model input - the reference
velocity. No image is opened, so it runs in seconds on a two-hour capture.

For every candidate gap it reports:

* the pair interval actually realised on this capture's irregular clock;
* translational flow in cells at the image centre (median, 5th/95th pct);
* the rotation between the two exposures and the largest image displacement
  it causes anywhere in the frame (which is what an attitude pre-warp, or a
  rotation-centred search, has to absorb);
* how far the old small-angle rotational field is from the exact rotation at
  the image corner - the error a linearised de-rotation leaves behind;
* the fraction of the first image still visible in the second (overlap);
* the single-pair speed precision implied by a sub-cell matching error.

It ends with a recommended ``--frame-gap`` and ``--max-frame-gap-s``: the
smallest gap whose median motion reaches ``--target-cells`` while the 5th
percentile overlap stays above ``--min-overlap``.

    python tools/check_motion_budget.py --dataset data \\
        --calibration configs/vo/camera_fixedwing.json \\
        --output artifacts/motion_budget.json
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

from vio.data.attitude import load_attitude_altitude, quaternion_at
from vio.data.calibration import load_camera_calibration
from vio.data.fixedwing_vo import reference_body_velocity
from vio.data.image_pairs import resize_camera_matrix
from vio.data.images import numeric_image_manifest, resolve_time_offsets
from vio.models.planar_geometry import NADIR_MOUNTINGS
from vio.models.pose_geometry import quaternion_to_matrix_np

DEFAULT_GAPS = (1, 2, 4, 6, 8, 10, 15, 20, 30, 40)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--csv-name", default="flight.csv")
    parser.add_argument("--image-folder", default="images")
    parser.add_argument("--image-pattern", default="*.jpg")
    parser.add_argument("--image-time-scale", type=float, default=0.001)
    parser.add_argument("--time-column", default="Time")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--altitude-column", default=None)
    parser.add_argument("--image-time-offset", type=float, default=0.0)
    parser.add_argument(
        "--calibration", type=Path, required=True,
        help="Camera intrinsics at the NATIVE image size.",
    )
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=(576, 1024), metavar=("H", "W"),
        help="Working resolution the frontend runs at (default 576 1024).",
    )
    parser.add_argument("--patch-size", type=int, default=8,
                        help="Working pixels per correlation cell.")
    parser.add_argument(
        "--camera-mounting", default="top_forward",
        choices=sorted(NADIR_MOUNTINGS),
        help="Which way the image points on the airframe. Only the OVERLAP "
             "numbers depend on it (motion along the long or the short image "
             "axis); the flow magnitudes do not. "
             "tools/estimate_camera_mounting.py measures it from the images.",
    )
    parser.add_argument("--gaps", default=",".join(str(g) for g in DEFAULT_GAPS),
                        help="Comma-separated frame gaps to evaluate.")
    parser.add_argument("--match-noise-cells", type=float, default=0.1,
                        help="Assumed sub-cell matching error for the precision column.")
    parser.add_argument("--target-cells", type=float, default=10.0,
                        help="Median translational flow the recommendation aims for.")
    parser.add_argument("--min-overlap", type=float, default=0.6,
                        help="5th-percentile overlap the recommended gap must keep.")
    parser.add_argument("--max-pairs", type=int, default=4000,
                        help="Pairs sampled per gap (evenly over the flight).")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def percentiles(values: np.ndarray, qs=(5, 50, 95)) -> List[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [float("nan")] * len(qs)
    return [float(v) for v in np.percentile(finite, qs)]


def clock_report(times: np.ndarray, label: str) -> Dict[str, float]:
    step = np.diff(times)
    median = float(np.median(step)) if step.size else float("nan")
    return {
        f"{label}_count": int(times.size),
        f"{label}_duration_s": float(times[-1] - times[0]) if times.size else 0.0,
        f"{label}_median_interval_s": median,
        f"{label}_rate_hz": 1.0 / median if median > 0 else float("nan"),
        f"{label}_interval_p1_s": percentiles(step, (1,))[0],
        f"{label}_interval_p99_s": percentiles(step, (99,))[0],
        f"{label}_gaps_over_1p5x": int(np.count_nonzero(step > 1.5 * median)),
        f"{label}_longest_gap_s": float(step.max()) if step.size else 0.0,
    }


def exact_corner_displacement(rotation_camera: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Largest pixel motion of the image corners under a pure rotation.

    ``corners`` are normalized (x, y, 1) rays. Exact: rotate, re-project.
    Returns the max displacement per pair, in normalized units.
    """

    rays = np.einsum("nij,ki->nkj", rotation_camera, corners)  # R^T r
    projected = rays[..., :2] / rays[..., 2:3]
    return np.linalg.norm(projected - corners[None, :, :2], axis=-1).max(axis=1)


def linearised_corner_error(rotation_camera: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """|exact - small-angle field| at the corners, per pair, normalized units.

    The small-angle field is the Longuet-Higgins-Prazdny rotational term the
    older frontend centres its search with; this is what it leaves behind.
    """

    rays = np.einsum("nij,ki->nkj", rotation_camera, corners)
    exact = rays[..., :2] / rays[..., 2:3] - corners[None, :, :2]
    # First-order expansion of the exact map: R^T r ~ r - omega x r with
    # omega = rotvec(R), which projects to exactly the field below.
    omega = rotation_vectors(rotation_camera)
    wx, wy, wz = (omega[:, i][:, None] for i in range(3))
    x = corners[None, :, 0]
    y = corners[None, :, 1]
    u = wx * x * y - wy * (1 + x * x) + wz * y
    v = wx * (1 + y * y) - wy * x * y - wz * x
    linear = np.stack((u, v), axis=-1)
    return np.linalg.norm(exact - linear, axis=-1).max(axis=1)


def rotation_vectors(rotation: np.ndarray) -> np.ndarray:
    """Log map of a stack of rotation matrices (angles well below pi)."""

    trace = np.trace(rotation, axis1=1, axis2=2)
    angle = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    axis = np.stack(
        (
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ),
        axis=1,
    )
    sin = np.sin(angle)
    scale = np.where(np.abs(sin) > 1e-9, angle / (2.0 * np.where(np.abs(sin) > 1e-9, sin, 1.0)), 0.5)
    return axis * scale[:, None]


def analyse_gap(
    gap: int,
    capture: np.ndarray,
    attitude,
    velocity_body: np.ndarray,
    camera_from_body: np.ndarray,
    working_matrix: np.ndarray,
    image_size,
    args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    count = capture.size - gap
    if count <= 0:
        return None
    first = np.arange(count)
    if count > args.max_pairs:
        first = np.unique(np.linspace(0, count - 1, args.max_pairs).round().astype(np.int64))
    t0 = capture[first]
    t1 = capture[first + gap]
    times = attitude.times_s
    inside = (t0 >= times[0]) & (t1 <= times[-1])
    t0, t1 = t0[inside], t1[inside]
    if t0.size == 0:
        return None
    dt = t1 - t0

    q0 = quaternion_at(times, attitude.quaternion, t0)
    q1 = quaternion_at(times, attitude.quaternion, t1)
    r0 = quaternion_to_matrix_np(q0)
    r1 = quaternion_to_matrix_np(q1)
    relative_body = np.einsum("nji,njk->nik", r0, r1)
    relative_camera = np.einsum(
        "ij,njk,lk->nil", camera_from_body, relative_body, camera_from_body
    )
    angle_deg = np.degrees(np.linalg.norm(rotation_vectors(relative_body), axis=1))

    # Mean body velocity over the pair (the label, used ONLY as a diagnostic
    # yardstick here) and the altitude at the first exposure.
    mid = 0.5 * (t0 + t1)
    velocity = np.stack(
        [np.interp(mid, times, velocity_body[:, k]) for k in range(3)], axis=1
    )
    altitude = np.interp(t0, times, attitude.altitude_m)
    down_body = np.einsum("nji,j->ni", r0, np.asarray([0.0, 0.0, 1.0]))
    normal = np.einsum("ij,nj->ni", camera_from_body, down_body)
    translation_camera = np.einsum("ij,nj->ni", camera_from_body, velocity) * dt[:, None]
    u = translation_camera / np.maximum(altitude, 1.0)[:, None]
    # Displacement of the ground at the image centre (m = (0, 0, 1)):
    # d = s (u_z m - u) / (1 - s u_z) with s = n_z.
    s = np.clip(normal[:, 2], 0.05, None)
    denominator = np.clip(1.0 - s * u[:, 2], 0.05, None)
    centre = -s[:, None] * u[:, :2] / denominator[:, None]

    fx, fy = float(working_matrix[0, 0]), float(working_matrix[1, 1])
    cell = float(args.patch_size)
    flow_px = np.stack((centre[:, 0] * fx, centre[:, 1] * fy), axis=1)
    flow_cells = np.linalg.norm(flow_px, axis=1) / cell

    height, width = image_size
    corners_px = np.asarray(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype=np.float64
    )
    corners = np.column_stack(
        (
            (corners_px[:, 0] - working_matrix[0, 2]) / fx,
            (corners_px[:, 1] - working_matrix[1, 2]) / fy,
            np.ones(4),
        )
    )
    rotation_cells = exact_corner_displacement(relative_camera, corners) * fx / cell
    linear_error_cells = linearised_corner_error(relative_camera, corners) * fx / cell
    overlap = np.clip(1.0 - np.abs(flow_px[:, 0]) / width, 0.0, 1.0) * np.clip(
        1.0 - np.abs(flow_px[:, 1]) / height, 0.0, 1.0
    )
    precision = args.match_noise_cells / np.maximum(flow_cells, 1e-6)
    speed = np.linalg.norm(velocity, axis=1)
    return {
        "gap": int(gap),
        "pairs": int(t0.size),
        "interval_s": percentiles(dt),
        "translation_cells": percentiles(flow_cells),
        "rotation_deg": percentiles(angle_deg),
        "rotation_corner_cells": percentiles(rotation_cells),
        "linearised_rotation_error_cells": percentiles(linear_error_cells),
        "overlap": percentiles(overlap),
        "single_pair_speed_error_percent": [100.0 * v for v in percentiles(precision)],
        "single_pair_speed_error_m_s": percentiles(precision * speed),
    }


def recommend(rows: List[Dict[str, object]], args: argparse.Namespace) -> Dict[str, object]:
    candidates = [
        row for row in rows
        if row["translation_cells"][1] >= args.target_cells
        and row["overlap"][0] >= args.min_overlap
    ]
    if candidates:
        chosen = min(candidates, key=lambda row: row["gap"])
        reason = (
            f"smallest gap whose median motion reaches {args.target_cells:g} cells "
            f"with 5th-percentile overlap >= {args.min_overlap:.0%}"
        )
    else:
        viable = [row for row in rows if row["overlap"][0] >= args.min_overlap]
        if not viable:
            return {"frame_gap": None, "reason": "no gap keeps the requested overlap"}
        chosen = max(viable, key=lambda row: row["translation_cells"][1])
        reason = (
            f"no gap reaches {args.target_cells:g} cells with enough overlap; "
            "this is the largest-motion gap that keeps it"
        )
    interval = chosen["interval_s"][1]
    return {
        "frame_gap": int(chosen["gap"]),
        # A dropout doubles an interval; half again of the nominal interval
        # keeps every healthy pair while refusing the pairs a dropout made.
        "max_frame_gap_s": round(1.5 * float(interval), 3),
        "median_translation_cells": chosen["translation_cells"][1],
        "reason": reason,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.dataset.expanduser().resolve()
    csv_path = root / args.csv_name
    attitude = load_attitude_altitude(
        csv_path, time_column=args.time_column, time_scale=args.time_scale,
        altitude_column=args.altitude_column,
    )
    _, velocity_body = reference_body_velocity(
        csv_path, time_column=args.time_column, time_scale=args.time_scale
    )
    _, capture = numeric_image_manifest(
        root / args.image_folder, args.image_pattern, args.image_time_scale
    )
    capture = capture + resolve_time_offsets(capture, float(args.image_time_offset))
    calibration = load_camera_calibration(args.calibration)
    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    working = resize_camera_matrix(
        calibration.camera_matrix, calibration.native_size, image_size
    )
    camera_from_body = np.asarray(NADIR_MOUNTINGS[args.camera_mounting], dtype=np.float64)

    report: Dict[str, object] = {
        "dataset": str(root),
        "working_image_size": list(image_size),
        "working_focal_px": [float(working[0, 0]), float(working[1, 1])],
        "cell_px": int(args.patch_size),
        "camera_mounting": args.camera_mounting,
        "altitude_m": percentiles(attitude.altitude_m),
        "ground_speed_m_s": percentiles(np.linalg.norm(velocity_body, axis=1)),
    }
    report.update(clock_report(attitude.times_s, "telemetry"))
    report.update(clock_report(capture, "images"))

    gaps = [int(piece) for piece in args.gaps.split(",") if piece.strip()]
    rows = []
    for gap in gaps:
        row = analyse_gap(gap, capture, attitude, velocity_body, camera_from_body,
                          working, image_size, args)
        if row is not None:
            rows.append(row)
    report["gaps"] = rows
    report["recommendation"] = recommend(rows, args)

    print(f"dataset   {root}")
    print(f"telemetry {report['telemetry_rate_hz']:.1f} Hz, "
          f"{report['telemetry_duration_s'] / 60:.1f} min, "
          f"{report['telemetry_gaps_over_1p5x']} gaps > 1.5x nominal "
          f"(longest {report['telemetry_longest_gap_s']:.3f} s)")
    print(f"images    {report['images_rate_hz']:.1f} Hz, {report['images_count']} frames, "
          f"interval p1/p99 {report['images_interval_p1_s']:.3f}/"
          f"{report['images_interval_p99_s']:.3f} s, "
          f"{report['images_gaps_over_1p5x']} dropouts > 1.5x nominal")
    alt = report["altitude_m"]
    spd = report["ground_speed_m_s"]
    print(f"altitude  {alt[1]:.0f} m (p5 {alt[0]:.0f}, p95 {alt[2]:.0f})   "
          f"ground speed {spd[1]:.1f} m/s (p5 {spd[0]:.1f}, p95 {spd[2]:.1f})")
    print(f"camera    focal {working[0, 0]:.0f} px at {image_size[1]}x{image_size[0]}, "
          f"cell {args.patch_size} px = {alt[1] * args.patch_size / working[0, 0]:.2f} m of ground")
    print()
    header = (f"{'gap':>4} {'dt s':>6} {'trans cells':>17} {'rot deg':>8} "
              f"{'rot@corner':>11} {'lin.err':>8} {'overlap':>8} {'speed err':>14}")
    print(header)
    print("-" * len(header))
    for row in rows:
        t = row["translation_cells"]
        print(
            f"{row['gap']:>4} {row['interval_s'][1]:6.3f} "
            f"{t[1]:6.2f} ({t[0]:4.1f}-{t[2]:4.1f}) "
            f"{row['rotation_deg'][2]:8.2f} "
            f"{row['rotation_corner_cells'][2]:11.2f} "
            f"{row['linearised_rotation_error_cells'][2]:8.3f} "
            f"{row['overlap'][0]:8.2f} "
            f"{row['single_pair_speed_error_percent'][1]:5.1f}% "
            f"{row['single_pair_speed_error_m_s'][1]:5.2f}m/s"
        )
    print("\n  trans cells: median (p5-p95) ground motion at the image centre, in cells")
    print("  rot deg / rot@corner: p95 rotation between exposures and the image motion it causes")
    print("  lin.err: p95 error of the small-angle rotational field at the corner, in cells")
    print("  overlap: 5th-percentile fraction of frame 0 still in frame 1")
    print(f"  speed err: single-pair error for {args.match_noise_cells:g} cells of matching noise")
    rec = report["recommendation"]
    print()
    if rec.get("frame_gap") is None:
        print(f"recommendation: {rec['reason']}")
    else:
        print(f"recommendation: --frame-gap {rec['frame_gap']} "
              f"--max-frame-gap-s {rec['max_frame_gap_s']}  ({rec['reason']})")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
