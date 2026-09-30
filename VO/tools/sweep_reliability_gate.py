#!/usr/bin/env python3
"""Score reliability-gate thresholds on a dataset WITHOUT training anything.

Picking a gate by launching a training run per threshold is slow and confounded
- each run sees a different number of pairs, so it is never clear whether a
change came from the gate or from the optimiser. This measures the gate itself:
how much of the correlation grid survives, and how many pairs are delivered, at
each threshold, over real image pairs.

Run it before training to choose a starting threshold, then confirm the choice
with ONE training run rather than a sweep of them. The numbers here answer
"what does this gate throw away"; only a training run answers "did throwing it
away help", and the two questions are worth separating.

The frontend is untrained, which is deliberate. The gate reads correlation
statistics - entropy, usable confidence, where the peak landed - and at
initialisation those describe the IMAGES rather than anything learned. A gate
tuned here is tuned on the data, which is what makes it a data-quality
decision rather than a model one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Same bootstrap every other tool in this directory uses: the package lives in
# src/ and this script is run directly, so without it the vio imports below
# fail outright unless the caller happens to have set PYTHONPATH.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch

from vio.data.calibration import load_camera_calibration
from vio.data.image_pairs import resize_camera_matrix
from vio.models.vision_mamba_vo import VisionMambaFlowFrontend

#: Pooled-weight thresholds worth trying first. 1e-4 is the historical default
#: and is included so every sweep reports what the current setting does.
DEFAULT_POOL_WEIGHTS = (1e-4, 0.02, 0.05, 0.10, 0.20)


def build_rectify_maps(calibration, image_size: tuple):
    """The remap the trainer applies before the frontend ever sees a frame.

    ``VisualPairSource`` undistorts when the calibration records nonzero
    distortion and the images are not already rectified (see
    vio/data/image_pairs.py). Skipping it here would tune the gate on pixels
    the trainer never feeds the frontend - and on this camera that is not a
    subtlety: k1 is about -0.39, which moves image content by tens of pixels
    at the edges, exactly where the correlation is already weakest.

    Returns ``None`` when no rectification is called for, which is also what a
    calibration-free sweep gets.
    """

    if calibration is None:
        return None
    distortion = np.asarray(calibration.distortion, dtype=np.float64).reshape(-1)
    if not distortion.size or not np.any(np.abs(distortion) > 0):
        return None
    if calibration.images_rectified:
        return None
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - optional dependency
        raise SystemExit(
            "this calibration records nonzero distortion, which the trainer "
            "rectifies with OpenCV before the frontend sees a frame. Install "
            "opencv-python so the sweep scores the same pixels, or pass no "
            "--calibration to sweep on raw frames and accept the mismatch."
        ) from error

    target_height, target_width = image_size
    working_matrix = resize_camera_matrix(
        calibration.camera_matrix, calibration.native_size, image_size
    )
    return cv2.initUndistortRectifyMap(
        np.asarray(calibration.camera_matrix, dtype=np.float64),
        distortion,
        None,
        working_matrix,
        (target_width, target_height),
        cv2.CV_32FC1,
    )


def load_body_rates(
    flight_csv: Path,
    *,
    time_column: str,
    rate_columns: List[str],
    time_scale: float,
) -> tuple:
    """Telemetry times and body rates, for averaging over an exposure interval."""

    import csv as csv_module

    times: List[float] = []
    rates: List[List[float]] = []
    with flight_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv_module.DictReader(handle)
        missing = [
            name for name in [time_column] + rate_columns
            if name not in (reader.fieldnames or [])
        ]
        if missing:
            raise SystemExit(
                f"{flight_csv} has no column(s) {missing}. Present: "
                f"{reader.fieldnames}. Name the real ones with --time-column "
                "and --rate-columns."
            )
        for row in reader:
            try:
                times.append(float(row[time_column]) * time_scale)
                rates.append([float(row[name]) for name in rate_columns])
            except (TypeError, ValueError):
                continue  # a blank or malformed row, not a reason to stop
    if len(times) < 2:
        raise SystemExit(f"{flight_csv} has fewer than two usable rows")
    return np.asarray(times, dtype=np.float64), np.asarray(rates, dtype=np.float64)


def mean_rate_over_exposure(
    telemetry_times_s: np.ndarray,
    rates_rad_s: np.ndarray,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    """Mean body rate over ``[start_s, end_s]``, endpoints interpolated.

    The same quantity ``VisualPairSource.rate_over_exposure`` computes for
    training, and for the same reason: the flow describes what happened between
    the two SHUTTER instants, so the rotation that centres the search window
    must be read over that interval rather than at a nearby telemetry sample.
    Snapping to the nearest sample is up to 5 ms at 100 Hz, and the error is
    turn-correlated rather than random.

    Kept as a trapezoidal integral with interpolated endpoints so a sweep and a
    training run agree about what "the body rate for this pair" means. If the
    canonical implementation changes, this must change with it.
    """

    if end_s <= start_s:
        return np.zeros(rates_rad_s.shape[1], dtype=np.float64)
    # Dense resample across the exposure, then average. Simpler than a
    # piecewise trapezoid over an irregular grid and identical to within the
    # sampling density, which is set far above the telemetry rate.
    samples = max(16, int((end_s - start_s) * 400.0))
    grid = np.linspace(start_s, end_s, samples)
    return np.stack([
        np.interp(grid, telemetry_times_s, rates_rad_s[:, axis]).mean()
        for axis in range(rates_rad_s.shape[1])
    ])


def load_image_pairs(
    folder: Path,
    *,
    image_size: tuple,
    max_pairs: int,
    stride: int,
    rectify_maps=None,
    telemetry=None,
    image_time_offset_s: float = 0.0,
) -> tuple:
    """Consecutive frames as normalized grayscale tensors, plus their intervals.

    When ``telemetry`` is supplied it is ``(times_s, rates_rad_s)`` and each
    pair also gets the mean body rate over its own exposure interval - the
    rotation the trainer uses to centre the correlation search. Without it
    every rate is zero, which makes the sweep measure a straight-and-level
    aircraft it may never have been.
    """

    from PIL import Image

    frame_paths: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        frame_paths.extend(folder.glob(pattern))
    frame_paths.sort()
    if len(frame_paths) < 2:
        raise SystemExit(f"need at least 2 frames in {folder}")

    def load(path: Path) -> torch.Tensor:
        image = Image.open(path).convert("L")
        if rectify_maps is None:
            image = image.resize((image_size[1], image_size[0]), Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        else:
            import cv2

            # The maps combine NATIVE-resolution undistortion and the
            # working-resolution resize into one interpolation, exactly as
            # VisualPairSource does - so they must be applied to the native
            # image. Resizing first and remapping after would index a
            # native-coordinate map into an already-shrunk image, sampling
            # almost entirely out of bounds.
            native = np.asarray(image, dtype=np.float32) / 255.0
            array = cv2.remap(
                native, rectify_maps[0], rectify_maps[1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        return torch.from_numpy(np.ascontiguousarray(array))

    first_frames, second_frames, intervals = [], [], []
    body_rates: List[np.ndarray] = []
    for index in range(0, len(frame_paths) - 1, stride):
        if len(first_frames) >= max_pairs:
            break
        first_path, second_path = frame_paths[index], frame_paths[index + 1]
        # Filenames are milliseconds; a folder that does not follow that
        # convention still sweeps fine on a nominal interval.
        try:
            interval = (int(second_path.stem) - int(first_path.stem)) / 1000.0
        except ValueError:
            interval = 0.05
        if interval <= 0:
            continue
        first_frames.append(load(first_path))
        second_frames.append(load(second_path))
        intervals.append(interval)
        if telemetry is None:
            body_rates.append(np.zeros(3, dtype=np.float64))
        else:
            # Image stamps are milliseconds; the offset puts the camera clock
            # on the telemetry clock, exactly as --image-time-offset does for
            # training.
            start = int(first_path.stem) / 1000.0 + image_time_offset_s
            end = int(second_path.stem) / 1000.0 + image_time_offset_s
            body_rates.append(
                mean_rate_over_exposure(telemetry[0], telemetry[1], start, end)
            )

    if not first_frames:
        # Every candidate pair was rejected, which on a millisecond-named
        # folder means every interval came out zero or negative - a duplicated
        # or out-of-order timestamp, not an empty folder. Say which, because
        # torch.stack on an empty list raises something that names neither.
        raise SystemExit(
            f"no usable consecutive pairs in {folder}: found "
            f"{len(frame_paths)} frames but every interval was <= 0 "
            "(duplicate or non-monotonic timestamps?)"
        )

    image0 = torch.stack(first_frames).unsqueeze(1)
    image1 = torch.stack(second_frames).unsqueeze(1)
    return (
        image0,
        image1,
        torch.tensor(intervals).reshape(-1, 1),
        torch.tensor(np.stack(body_rates), dtype=torch.float32),
    )


def score_gate(
    image0: torch.Tensor,
    image1: torch.Tensor,
    pair_dt_s: torch.Tensor,
    *,
    image_size: tuple,
    camera_matrix: Optional[torch.Tensor],
    patch_size: int,
    correlation_radius: int,
    token_grid: int,
    gates: Dict[str, float],
    chunk: int,
    body_rate: torch.Tensor,
    frontend_state=None,
) -> Dict[str, float]:
    """One gate setting, over every pair. Returns what it kept."""

    torch.manual_seed(0)
    frontend = VisionMambaFlowFrontend(
        image_size=image_size, patch_size=patch_size, token_grid=token_grid,
        correlation_radius=correlation_radius, dropout=0.0, **gates,
    )
    if frontend_state is not None:
        # strict=False: the gate settings change no tensor shape, so a
        # checkpoint from any gate configuration loads, and a checkpoint from a
        # different STEM size fails loudly here rather than scoring nonsense.
        missing, unexpected = frontend.load_state_dict(frontend_state, strict=False)
        if missing or unexpected:
            raise SystemExit(
                "checkpoint frontend does not match this geometry "
                f"(missing {len(missing)}, unexpected {len(unexpected)} "
                "tensors). Pass the same --image-size/--patch-size/"
                "--token-grid/--correlation-radius the run trained with."
            )
    frontend.eval()

    reliable_cells: List[float] = []
    pairs_kept: List[float] = []
    entropies: List[float] = []
    boundaries: List[float] = []
    # min_pool_weight acts at POOLING, after the reliability gate, so it never
    # moves reliable_cell_fraction. occupied_fraction is the one it does move,
    # and reporting both is what keeps the pool-weight column from looking
    # inert when it is in fact doing the second half of the filtering.
    occupied: List[float] = []

    with torch.no_grad():
        for start in range(0, image0.shape[0], chunk):
            stop = min(start + chunk, image0.shape[0])
            output = frontend(
                image0[start:stop], image1[start:stop],
                pair_dt_s=pair_dt_s[start:stop],
                body_rate_rad_s=body_rate[start:stop],
                camera_matrix=camera_matrix,
            )
            diagnostics = output["diagnostics"]
            reliable_cells.append(float(diagnostics["reliable_cell_fraction"].sum()))
            pairs_kept.append(float(output["pair_reliable"].sum()))
            entropies.append(float(diagnostics["mean_entropy_normalized"].sum()))
            boundaries.append(float(diagnostics["boundary_hit_fraction"].sum()))
            occupied.append(float(diagnostics["occupied_fraction"].sum()))

    pairs = float(image0.shape[0])
    return {
        "reliable_cell_fraction": sum(reliable_cells) / pairs,
        "pair_kept_fraction": sum(pairs_kept) / pairs,
        "mean_entropy_normalized": sum(entropies) / pairs,
        "boundary_hit_fraction": sum(boundaries) / pairs,
        "occupied_fraction": sum(occupied) / pairs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure what each reliability-gate threshold throws away.",
    )
    parser.add_argument("images", type=Path, help="Folder of frames.")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Camera calibration JSON. Without it the image "
                             "centre and a focal length equal to the image "
                             "size are assumed, same as training.")
    parser.add_argument("--image-size", type=int, nargs=2, default=(576, 1024),
                        metavar=("HEIGHT", "WIDTH"),
                        help="Working resolution, matching training.")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--correlation-radius", type=int, default=4)
    parser.add_argument("--token-grid", type=int, default=6)
    parser.add_argument("--max-pairs", type=int, default=48,
                        help="How many image pairs to score per threshold.")
    parser.add_argument("--pair-stride", type=int, default=1,
                        help="Use every Nth consecutive pair.")
    parser.add_argument("--chunk", type=int, default=4,
                        help="Pairs per frontend call, to bound memory.")
    parser.add_argument("--entropy-limits", type=float, nargs="*",
                        default=(1.0, 0.8, 0.7, 0.6, 0.5),
                        help="--max-cell-entropy values to try.")
    parser.add_argument("--confidence-limits", type=float, nargs="*",
                        default=(0.0,),
                        help="--min-cell-confidence values to try.")
    parser.add_argument("--pool-weights", type=float, nargs="*",
                        default=DEFAULT_POOL_WEIGHTS,
                        help="--min-pool-weight values to try.")
    parser.add_argument("--score-margin-limits", type=float, nargs="*",
                        default=(0.0,),
                        help="--min-score-margin values to try. This gate is "
                             "temperature-free (raw top-1 minus top-2 "
                             "correlation score) and lives in correlation-score "
                             "units, not on [0, 1], so it needs its own sweep "
                             "rather than being read off the entropy column.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Load frontend weights from a training checkpoint. "
                             "REQUIRED for --flight-csv to do anything: the "
                             "body-rate-to-search-offset map "
                             "(RotationalSearchField) is LEARNED and "
                             "zero-initialised, so on an untrained frontend "
                             "every body rate produces a zero offset and the "
                             "telemetry has no effect at all.")
    parser.add_argument("--flight-csv", type=Path, default=None,
                        help="Telemetry CSV, so each pair is scored under the "
                             "body rate it was actually flown at. WITHOUT this "
                             "every rate is zero, the search window is centred "
                             "as if straight and level, and the manoeuvring "
                             "pairs - which is where this dataset fails - are "
                             "misrepresented.")
    parser.add_argument("--time-column", default="Time")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="Multiply the time column by this to get seconds.")
    parser.add_argument("--rate-columns", nargs=3, default=("p", "q", "r"),
                        metavar=("P", "Q", "R"),
                        help="Body-rate column names, in rad/s.")
    parser.add_argument("--image-time-offset", type=float, default=0.0,
                        help="Seconds added to every image timestamp to put "
                             "the camera clock on the telemetry clock. Pass "
                             "the SAME value training will use.")
    parser.add_argument("--reject-boundary-peaks", action="store_true",
                        help="Also reject peaks pinned to the window edge.")
    parser.add_argument("--min-reliable-cell-fraction", type=float, default=0.25,
                        help="Pair-level threshold held fixed across the sweep.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the full sweep as JSON here.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    image_size = (int(args.image_size[0]), int(args.image_size[1]))

    # nargs="*" accepts an empty list, which would silently sweep nothing and
    # then fail in max() on the empty results at the end. Refuse up front and
    # name the flag, rather than after the loading work is already done.
    for name, values in (
        ("--pool-weights", args.pool_weights),
        ("--entropy-limits", args.entropy_limits),
        ("--confidence-limits", args.confidence_limits),
        ("--score-margin-limits", args.score_margin_limits),
    ):
        if not len(values):
            raise SystemExit(f"{name} needs at least one value")

    camera_matrix = None
    calibration = None
    if args.calibration is not None:
        calibration = load_camera_calibration(args.calibration)
        # The frontend is handed images already at the working resolution, so
        # the matrix must be rescaled to match - the same step the trainer
        # does, and the one whose absence was a real past bug.
        camera_matrix = torch.from_numpy(
            resize_camera_matrix(
                calibration.camera_matrix, calibration.native_size, image_size
            )
        ).float()

    frontend_state = None
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "frontend" not in payload:
            raise SystemExit(f"{args.checkpoint} carries no 'frontend' weights")
        frontend_state = payload["frontend"]

    telemetry = None
    if args.flight_csv is not None:
        telemetry = load_body_rates(
            args.flight_csv,
            time_column=args.time_column,
            rate_columns=list(args.rate_columns),
            time_scale=args.time_scale,
        )

    rectify_maps = build_rectify_maps(calibration, image_size)
    image0, image1, pair_dt_s, body_rate = load_image_pairs(
        args.images, image_size=image_size,
        max_pairs=args.max_pairs, stride=args.pair_stride,
        rectify_maps=rectify_maps,
        telemetry=telemetry,
        image_time_offset_s=args.image_time_offset,
    )
    rectified = "rectified" if rectify_maps is not None else "raw (not rectified)"
    peak_rate = float(body_rate.abs().max()) if body_rate.numel() else 0.0
    rate_note = (
        f"real body rates (peak |w| {peak_rate:.3f} rad/s)"
        if telemetry is not None
        else "ZERO body rates"
    )
    print(f"scoring {image0.shape[0]} pairs at "
          f"{image_size[0]}x{image_size[1]}, {rectified}, {rate_note}")
    if telemetry is None:
        print("WARNING: no --flight-csv, so the search window is centred as if "
              "the aircraft were straight and level. Thresholds chosen this "
              "way can differ from what training sees during manoeuvres, which "
              "is exactly where this dataset's correlation fails.")
    elif frontend_state is None:
        # Measured: with a zero-initialised rotation map the offset is zero for
        # EVERY body rate, so the telemetry changes nothing. Saying so is the
        # difference between a sweep that reproduces manoeuvres and one that
        # only appears to.
        print("WARNING: --flight-csv without --checkpoint has NO EFFECT. The "
              "body-rate-to-search-offset map is learned and starts at zero, "
              "so an untrained frontend produces a zero offset whatever the "
              "rate. Pass a trained --checkpoint, or read these numbers as "
              "straight-and-level.")
    print(f"frontend weights: "
          f"{'from ' + str(args.checkpoint) if frontend_state else 'untrained'}")
    print()

    results = []
    print(f"{'pool_w':>8s} {'entropy':>8s} {'conf':>6s} {'margin':>7s} "
          f"{'cells_kept':>11s} {'occupied':>9s} {'pairs_kept':>11s}")
    for pool_weight in args.pool_weights:
        for entropy_limit in args.entropy_limits:
          for confidence_limit in args.confidence_limits:
            for margin_limit in args.score_margin_limits:
                gates = {
                    "min_pool_weight": pool_weight,
                    "max_cell_entropy": entropy_limit,
                    "min_cell_confidence": confidence_limit,
                    "min_score_margin": margin_limit,
                    "reject_boundary_peaks": args.reject_boundary_peaks,
                    "min_reliable_cell_fraction": args.min_reliable_cell_fraction,
                }
                scored = score_gate(
                    image0, image1, pair_dt_s,
                    image_size=image_size, camera_matrix=camera_matrix,
                    patch_size=args.patch_size,
                    correlation_radius=args.correlation_radius,
                    token_grid=args.token_grid,
                    gates=gates, chunk=args.chunk, body_rate=body_rate,
                    frontend_state=frontend_state,
                )
                results.append({"gates": gates, "kept": scored})
                print(f"{pool_weight:8.4f} {entropy_limit:8.2f} "
                      f"{confidence_limit:6.2f} {margin_limit:7.3f} "
                      f"{scored['reliable_cell_fraction']:11.3f} "
                      f"{scored['occupied_fraction']:9.3f} "
                      f"{scored['pair_kept_fraction']:11.3f}")

    # A gate that keeps nothing is not a gate, it is an off switch - and one
    # that keeps everything has not been set. Say so rather than leaving the
    # reader to notice a column of 1.000.
    kept = [row["kept"]["pair_kept_fraction"] for row in results]
    print()
    if max(kept) <= 0.0:
        print("NOTE: every setting rejected every pair. The thresholds are far "
              "too tight for this imagery, or the images carry no usable "
              "correspondence at all - check tools/check_image_matchability.py.")
    elif min(kept) >= 1.0:
        print("NOTE: every setting kept every pair. Nothing here is acting as "
              "a filter; tighten --entropy-limits or raise --pool-weights.")
    else:
        print("Pick a setting that removes the ambiguous tail without emptying "
              "the stream - then confirm it with ONE training run. This sweep "
              "says what is discarded, never whether discarding it helped.")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Provenance written BY the tool. A block added by hand afterwards is
        # lost the next time the file is regenerated, which is exactly when a
        # reader most needs to know which flags produced it.
        args.output.write_text(json.dumps({
            "provenance": {
                "tool": "tools/sweep_reliability_gate.py",
                "argv": sys.argv[1:],
                "images": str(args.images),
                "calibration": (
                    None if args.calibration is None else str(args.calibration)
                ),
                "flight_csv": (
                    None if args.flight_csv is None else str(args.flight_csv)
                ),
                "image_time_offset_s": float(args.image_time_offset),
                "rectified": rectify_maps is not None,
                "body_rates": "real" if telemetry is not None else "zero",
                "frontend_weights": (
                    str(args.checkpoint) if frontend_state is not None
                    else "untrained"
                ),
                "body_rates_effective": bool(
                    telemetry is not None and frontend_state is not None
                ),
                "peak_body_rate_rad_s": peak_rate,
                "image_size": list(image_size),
                "patch_size": int(args.patch_size),
                "correlation_radius": int(args.correlation_radius),
                "token_grid": int(args.token_grid),
                "pair_stride": int(args.pair_stride),
                "max_pairs": int(args.max_pairs),
                "min_reliable_cell_fraction": float(
                    args.min_reliable_cell_fraction
                ),
                "reject_boundary_peaks": bool(args.reject_boundary_peaks),
                "note": (
                    "reliable_cell_fraction is reliable cells over VALID cells "
                    "(those with a complete search window), so 1.0 is "
                    "reachable and the threshold does not move with image size "
                    "or correlation radius."
                ),
            },
            "pairs_scored": int(image0.shape[0]),
            "image_size": list(image_size),
            "results": results,
        }, indent=2))
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
