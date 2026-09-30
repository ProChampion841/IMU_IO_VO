#!/usr/bin/env python3
"""Decide whether a dataset's images can support a visual-inertial result.

Point this at any dataset in the standard layout (``flight.csv`` plus
``images/<milliseconds>.jpg``) before spending hours caching features and
training. It answers three separate questions that are easy to confuse:

1. **Timing.** Do the image timestamps and the telemetry clock overlap, at
   the rates each claims?

2. **Synchronisation.** Does the apparent image motion follow the body rates? Pure
   rotation shifts the image by ``focal * omega * dt`` pixels, so if the
   frames belong to this flight then both image axes independently imply the
   *same* focal length. Two axes agreeing on a focal length is very strong
   evidence; a correlation on its own is not.

3. **Supervision target.** Does the telemetry actually contain the motion the
   estimator is asked to predict? A target that never changes, or changes far
   too slowly, cannot produce a trajectory no matter how good the images are.

4. **Scene content.** Synchronisation is not sufficient. Images can be
   perfectly time-aligned and still carry nothing a matcher can use, because
   attitude-driven imagery can be generated *from* the telemetry. Removing the
   global motion and measuring whether fine detail persists separates real
   terrain from re-randomised noise. Imagery whose only stable structure is
   low-frequency encodes attitude the telemetry already provides, so a model
   using it learns nothing new and any apparent gain is circular.

The verdict is deliberately conservative and never claims more than the
evidence supports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The package lives under src/; tools/ is imported as a package from the
# repository root. Both have to be importable when a script is run directly.
for _entry in (ROOT / "src", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from vio.data.attitude import load_attitude_altitude  # noqa: E402
from vio.data.fixedwing_vo import reference_body_velocity  # noqa: E402

# The label this dataset is audited against. It is an evaluation-only source:
# the estimator never reads it, and reference_body_velocity is the only place
# that touches it.
TARGET_DESCRIPTION = "GPSNavVn* rotated into the body frame"

# A matcher needs fine detail that survives from one frame to the next. Below
# this correlation the texture is effectively re-drawn each frame.
DETAIL_PERSISTENCE_USABLE = 0.60
DETAIL_PERSISTENCE_MARGINAL = 0.35
# Two axes implying focal lengths within this fraction of each other is far
# beyond coincidence for independently generated imagery.
FOCAL_AGREEMENT_TOLERANCE = 0.25
SYNC_CORRELATION = 0.5


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--image-folder", default="images")
    parser.add_argument("--image-pattern", default="*.jpg")
    parser.add_argument("--image-time-scale", type=float, default=0.001)
    parser.add_argument(
        "--image-time-offset",
        type=float,
        default=0.0,
        help="Seconds added to every image timestamp before comparison.",
    )
    parser.add_argument("--csv-name", default="flight.csv")
    parser.add_argument("--time-column", default="Time")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help=(
            "Converts the time column to seconds: 1.0 if it is already "
            "seconds, 1e-3 for milliseconds."
        ),
    )
    parser.add_argument(
        "--frame-gap", type=int, default=1,
        help=(
            "Frames between the two images of each analysed pair. Match this "
            "to the trainer's --frame-gap: detail persistence is measured over "
            "the separation the frontend will actually correlate."
        ),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=400,
        help="Consecutive frame pairs to analyse.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.max_pairs < 20:
        parser.error("max-pairs must be at least twenty to be meaningful")
    if args.image_time_scale <= 0:
        parser.error("image-time-scale must be positive")
    return args


def _load_images(args: argparse.Namespace) -> tuple[List[Path], np.ndarray]:
    folder = args.dataset / args.image_folder
    if not folder.is_dir():
        raise SystemExit(f"Image directory does not exist: {folder}")
    paths = sorted(folder.glob(args.image_pattern), key=lambda p: float(p.stem))
    if len(paths) < 21:
        raise SystemExit("Need at least twenty-one images to judge a dataset")
    times = (
        np.asarray([float(p.stem) for p in paths], dtype=np.float64)
        * args.image_time_scale
        + args.image_time_offset
    )
    return paths, times


def analyse(args: argparse.Namespace) -> Dict[str, Any]:
    import cv2

    csv_path = args.dataset / args.csv_name
    attitude = load_attitude_altitude(
        csv_path,
        time_column=args.time_column,
        time_scale=args.time_scale,
    )
    telemetry_time = np.asarray(attitude.times_s)
    # Differenced from attitude, not measured by a gyro: this project has no
    # rate sensor. Row i is the mean rate over [i-1, i] and row 0 is zero, so
    # it is already an interval average and already in rad/s - the old
    # deg2rad on gyro counts would have scaled it wrongly by 57x.
    body_rate = attitude.body_rate_rad_s.astype(np.float64)
    paths, image_time = _load_images(args)

    overlap = min(telemetry_time[-1], image_time[-1]) - max(
        telemetry_time[0], image_time[0]
    )
    _, velocity = reference_body_velocity(
        csv_path,
        time_column=args.time_column,
        time_scale=args.time_scale,
    )
    velocity = velocity.astype(np.float64)
    unique_rows = int(np.unique(np.round(velocity, 9), axis=0).shape[0])
    speed = np.linalg.norm(velocity, axis=1)
    target = {
        "unique_rows": unique_rows,
        "rows": int(velocity.shape[0]),
        "unique_fraction": float(unique_rows / max(velocity.shape[0], 1)),
        "speed_min": float(speed.min()),
        "speed_max": float(speed.max()),
        "speed_range": float(speed.max() - speed.min()),
        "per_axis_std": [float(value) for value in velocity.std(axis=0)],
    }

    timing = {
        "image_count": len(paths),
        "telemetry_rows": int(telemetry_time.size),
        "image_median_period_s": float(np.median(np.diff(image_time))),
        "telemetry_median_period_s": float(np.median(np.diff(telemetry_time))),
        "image_span_s": float(image_time[-1] - image_time[0]),
        "telemetry_span_s": float(telemetry_time[-1] - telemetry_time[0]),
        "overlap_s": float(overlap),
        "overlap_fraction_of_images": float(
            overlap / max(image_time[-1] - image_time[0], 1e-9)
        ),
    }

    # Sample across the whole flight, not the first N pairs. A capture that
    # begins with the aircraft stationary would otherwise be judged entirely
    # on its ground roll and look like it has no motion at all.
    # Training may correlate non-adjacent frames (--frame-gap). Detail
    # persistence has to be measured over the SAME separation the frontend
    # will see, because it falls as the gap widens: a gap that makes the
    # displacement easier to resolve can simultaneously make the two frames
    # too different to match.
    gap = int(getattr(args, "frame_gap", 1))
    if gap < 1:
        raise SystemExit("--frame-gap must be at least one")
    if len(paths) - gap < 2:
        raise SystemExit(f"Not enough images for --frame-gap {gap}")
    count = min(args.max_pairs, len(paths) - gap)
    sampled = np.unique(np.linspace(0, len(paths) - 1 - gap, count).astype(int))
    count = int(sampled.size)
    shift = np.zeros((count, 2))
    rate = np.zeros((count, 2))
    interval = np.zeros(count)
    roll_rate = np.zeros(count)
    detail = np.zeros(count)
    coarse = np.zeros(count)
    repeatability = np.zeros(count)

    detector = cv2.ORB_create(nfeatures=600)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for position, index in enumerate(sampled):
        previous = cv2.imread(str(paths[index]), cv2.IMREAD_GRAYSCALE)
        current = cv2.imread(str(paths[index + gap]), cv2.IMREAD_GRAYSCALE)
        if previous is None or current is None:
            raise SystemExit(f"Could not read image near {paths[index]}")
        first = previous.astype(np.float32)
        second = current.astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(first, second)
        shift[position] = (dx, dy)

        start, end = image_time[index], image_time[index + gap]
        inside = (telemetry_time >= start) & (telemetry_time < end)
        if not inside.any():
            inside = np.zeros_like(telemetry_time, dtype=bool)
            inside[int(np.argmin(np.abs(telemetry_time - start)))] = True
        # Which body rate drives horizontal image motion depends on where the
        # camera points, and guessing wrong makes a good capture look
        # unsynchronised. Record all three rates and let the fit below decide.
        #   forward-facing: image x follows YAW, image y follows pitch
        #   down-facing:    image x follows ROLL, image y follows pitch, and
        #                   yaw becomes pure image ROTATION with no translation
        rate[position] = (body_rate[inside, 2].mean(), body_rate[inside, 1].mean())
        roll_rate[position] = body_rate[inside, 0].mean()
        interval[position] = end - start

        # Undo the global motion, then ask what still lines up.
        translation = np.float32([[1, 0, dx], [0, 1, dy]])
        warped = cv2.warpAffine(first, translation, (first.shape[1], first.shape[0]))
        valid = (warped > 0) & (second > 0)
        valid[:30, :] = valid[-30:, :] = valid[:, :30] = valid[:, -30:] = False
        if valid.sum() > 1000:
            low_a = cv2.GaussianBlur(warped, (0, 0), 4)
            low_b = cv2.GaussianBlur(second, (0, 0), 4)
            coarse[position] = float(np.corrcoef(low_a[valid], low_b[valid])[0, 1])
            detail[position] = float(
                np.corrcoef(
                    (warped - low_a)[valid], (second - low_b)[valid]
                )[0, 1]
            )

        key_a, desc_a = detector.detectAndCompute(previous, None)
        key_b, desc_b = detector.detectAndCompute(current, None)
        if desc_a is not None and desc_b is not None and len(key_a) and len(key_b):
            matches = matcher.match(desc_a, desc_b)
            repeatability[position] = len(matches) / max(len(key_a), len(key_b))

    # focal * omega * dt = pixel shift, solved independently per axis. The
    # horizontal axis is fitted against both candidate drivers; whichever
    # recovers a focal length near the calibrated one identifies the mounting.
    synchronisation: Dict[str, Any] = {}
    focals = []
    drivers = {
        "x_from_yaw": rate[:, 0] * interval,
        "x_from_roll": roll_rate * interval,
        "y_from_pitch": rate[:, 1] * interval,
    }
    for name, driver in drivers.items():
        axis = 1 if name == "y_from_pitch" else 0
        denominator = float(np.dot(driver, driver))
        focal = float(np.dot(shift[:, axis], driver) / denominator) if denominator > 0 else 0.0
        correlation = (
            float(np.corrcoef(shift[:, axis], driver)[0, 1])
            if np.std(driver) > 0
            else 0.0
        )
        synchronisation[name] = {
            "implied_focal_px": focal,
            "correlation": correlation,
            "mean_abs_shift_px": float(np.abs(shift[:, axis]).mean()),
            "mean_abs_residual_px": float(
                np.abs(shift[:, axis] - focal * driver).mean()
            ),
        }
        if abs(correlation) >= SYNC_CORRELATION:
            focals.append(abs(focal))

    # Yaw and roll are alternative explanations of the same horizontal motion,
    # so only the better one may be paired with the pitch fit. Which one wins
    # is itself the answer to where the camera points, and getting that
    # backwards is what makes a well-synchronised down-facing capture look
    # broken: for a down-facing camera yaw is rotation about the optical axis
    # and produces image ROTATION, not translation.
    horizontal = max(
        ("x_from_yaw", "x_from_roll"),
        key=lambda name: abs(synchronisation[name]["correlation"]),
    )
    mounting = "forward_facing" if horizontal == "x_from_yaw" else "down_facing"
    if abs(synchronisation[horizontal]["correlation"]) < SYNC_CORRELATION:
        mounting = "undetermined"
    synchronisation["horizontal_driver"] = horizontal
    synchronisation["inferred_camera_mounting"] = mounting

    pair = [
        abs(synchronisation[name]["implied_focal_px"])
        for name in (horizontal, "y_from_pitch")
        if abs(synchronisation[name]["correlation"]) >= SYNC_CORRELATION
    ]
    agreement = None
    if len(pair) == 2 and max(pair) > 0:
        agreement = float(abs(pair[0] - pair[1]) / max(pair))
    synchronisation["focal_agreement_relative"] = agreement
    synchronisation["axes_agree_on_one_focal_length"] = bool(
        agreement is not None and agreement <= FOCAL_AGREEMENT_TOLERANCE
    )

    content = {
        "detail_persistence": float(np.nanmean(detail)),
        "coarse_persistence": float(np.nanmean(coarse)),
        "orb_match_fraction": float(repeatability.mean()),
        "pairs_analysed": int(count),
        "frame_gap": gap,
    }

    verdict, reasons = _judge(timing, synchronisation, content, target)
    return {
        "schema_version": 1,
        "dataset": str(args.dataset.resolve()),
        "image_time_offset_s": args.image_time_offset,
        "timing": timing,
        "synchronisation": synchronisation,
        "scene_content": content,
        "supervision_target": target,
        "verdict": verdict,
        "reasons": reasons,
        "thresholds": {
            "detail_persistence_usable": DETAIL_PERSISTENCE_USABLE,
            "detail_persistence_marginal": DETAIL_PERSISTENCE_MARGINAL,
            "focal_agreement_tolerance": FOCAL_AGREEMENT_TOLERANCE,
            "sync_correlation": SYNC_CORRELATION,
        },
    }


def _judge(timing, synchronisation, content, target) -> tuple[str, List[str]]:
    reasons: List[str] = []
    # Check the label before anything else: without motion in the target, the
    # quality of the images cannot matter.
    if target["unique_rows"] <= 1:
        reasons.append(
            f"The velocity target ({TARGET_DESCRIPTION}) is constant "
            f"across all {target['rows']} rows, so it carries no motion. "
            "Trapezoid-integrating it yields a straight line at fixed speed, "
            "not the aircraft's trajectory. Populate these columns with the "
            "flight's NED velocity before training."
        )
        return "no_motion_in_supervision_target", reasons
    if target["speed_range"] < 0.5:
        reasons.append(
            f"The velocity target varies by only {target['speed_range']:.3f} m/s "
            "over the whole flight, which is too little to supervise motion."
        )
        return "supervision_target_barely_varies", reasons
    if timing["overlap_fraction_of_images"] < 0.5:
        reasons.append("Image and telemetry clocks barely overlap.")
        return "timestamps_do_not_overlap", reasons

    synced = synchronisation["axes_agree_on_one_focal_length"]
    if synced:
        reasons.append(
            "Both image axes imply the same focal length, so the frames follow "
            "the recorded rotation."
        )
    else:
        reasons.append(
            "Image motion does not follow the body rates consistently across axes. "
            "Either the frames are not synchronised, or translation dominates "
            "rotation in this geometry and this test cannot decide."
        )

    detail = content["detail_persistence"]
    if detail >= DETAIL_PERSISTENCE_USABLE:
        reasons.append(
            f"Fine detail persists between frames ({detail:.2f}), so there is "
            "trackable scene structure."
        )
        return ("usable_for_vio" if synced else "texture_present_sync_unconfirmed"), reasons

    if detail < DETAIL_PERSISTENCE_MARGINAL:
        reasons.append(
            f"Fine detail does not persist ({detail:.2f}) while coarse "
            f"structure does ({content['coarse_persistence']:.2f}). The only "
            "stable content is low-frequency, which is what attitude-driven "
            "imagery generated from the telemetry looks like. Such frames "
            "carry no scene information the telemetry lacks, so a visual gain "
            "measured on them would be circular."
        )
        return "attitude_only_no_usable_scene_content", reasons

    reasons.append(
        f"Fine detail is weak ({detail:.2f}); matching will be unreliable."
    )
    return "insufficient_texture", reasons


def main(argv=None) -> int:
    args = parse_args(argv)
    report = analyse(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    sync = report["synchronisation"]
    content = report["scene_content"]
    print(f"dataset            : {report['dataset']}")
    print(
        "images/telemetry   : "
        f"{report['timing']['image_count']} frames, "
        f"{report['timing']['telemetry_rows']} rows, "
        f"overlap {report['timing']['overlap_s']:.1f} s"
    )
    for name in ("x_from_yaw", "x_from_roll", "y_from_pitch"):
        entry = sync[name]
        marker = " *" if name == sync["horizontal_driver"] else "  "
        print(
            f"  {name:<13}{marker}: focal {entry['implied_focal_px']:+8.1f} px  "
            f"r={entry['correlation']:+.3f}  "
            f"shift {entry['mean_abs_shift_px']:.2f} px  "
            f"resid {entry['mean_abs_residual_px']:.2f} px"
        )
    print(
        f"  camera mounting  : {sync['inferred_camera_mounting']}"
        f"  (horizontal image motion follows "
        f"{sync['horizontal_driver'].replace('x_from_', '')})"
    )
    agreement = sync["focal_agreement_relative"]
    print(
        "  focal agreement  : "
        + ("n/a" if agreement is None else f"{agreement * 100:.1f}% apart")
    )
    tgt = report["supervision_target"]
    print(
        f"  target motion    : {tgt['unique_rows']} unique of {tgt['rows']} rows, "
        f"speed {tgt['speed_min']:.2f}..{tgt['speed_max']:.2f} m/s"
    )
    print(
        f"  detail persist.  : {content['detail_persistence']:.3f}   "
        f"coarse {content['coarse_persistence']:.3f}   "
        f"ORB match {content['orb_match_fraction']:.3f}"
    )
    print(f"\nVERDICT: {report['verdict']}")
    for reason in report["reasons"]:
        print(f"  - {reason}")
    if args.output is not None:
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
