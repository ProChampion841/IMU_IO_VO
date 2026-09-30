#!/usr/bin/env python3
"""Ask one question: can a correlator find real matches between these frames?

The whole VO frontend rests on one assumption - that a patch of ground in
frame N can be found again in frame N+1, at one clearly-best place. If that
assumption fails, every downstream piece (the rotational field, the pooled
cells, the speed head) is being fed noise, and no amount of architecture or
training will rescue it.

Two numbers decide it, and they are NOT the same number:

**Sharpness** is how much fine detail a single frame holds, measured as the
RMS pixel-to-pixel difference. Blur, heavy compression and broken video
decoding all destroy it. Crisp aerial imagery sits around 0.06-0.12; below
about 0.03 there is very little left to match.

**Peak margin** is how far the best match stands above the second-best one.
This is the number that actually matters, and a high peak score alone can
hide a total failure: a smooth, blurry image correlates at 0.95+ against
almost every position in the search window, so the peak is high and the
margin is nearly zero. The flow that comes out of such a cell is whichever
position happened to win by a hair - a coin flip dressed up as a measurement.

Run this BEFORE blaming the model for a validation error that will not move.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

#: Below this peak margin a match is treated as ambiguous - several positions
#: explain the patch about equally well, so the winner carries no information.
AMBIGUOUS_MARGIN = 0.10
#: Above this the match is unambiguous enough to trust as a measurement.
CONFIDENT_MARGIN = 0.30
#: RMS gradient below this means the frame holds almost no fine detail.
LOW_SHARPNESS = 0.030
#: Per-pixel standard deviation, on [0, 1] images, below which a search window
#: is treated as having no texture at all. NCC divides by this std, so a window
#: that is flat to within rounding produces a ratio limited only by the epsilon
#: guarding the division - which is how a TEXTURELESS region, the very thing
#: this tool exists to flag, can come back scoring far above 1.0 and be counted
#: as a confident match. Such a window is not a bad match, it is no measurement
#: at all, and it is reported as zero correlation rather than as a huge one.
MIN_REGION_STD = 1e-3


def load_grayscale(path: Path, downscale: float) -> np.ndarray:
    """Load one frame as a float grayscale image in [0, 1]."""

    image = Image.open(path).convert("L")
    if downscale != 1.0:
        size = (int(image.width * downscale), int(image.height * downscale))
        image = image.resize(size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def frame_sharpness(gray: np.ndarray) -> float:
    """RMS pixel-to-pixel difference: how much fine detail this frame holds."""

    horizontal = np.diff(gray, axis=1)
    vertical = np.diff(gray, axis=0)
    return float(np.sqrt(0.5 * (np.mean(horizontal ** 2) + np.mean(vertical ** 2))))


def correlation_surface(patch: np.ndarray, search_region: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of one patch over a search region.

    The full surface is returned rather than just its peak, because the shape
    around the peak is what separates a measurement from a guess.
    """

    centred_patch = patch - patch.mean()
    patch_norm = float(np.sqrt(np.sum(centred_patch ** 2))) + 1e-8

    ones = np.ones_like(patch)
    region_sum = fftconvolve(search_region, ones[::-1, ::-1], mode="valid")
    region_square_sum = fftconvolve(search_region ** 2, ones[::-1, ::-1], mode="valid")
    pixel_count = patch.size
    region_mean = region_sum / pixel_count
    region_variance = np.maximum(region_square_sum - pixel_count * region_mean ** 2, 0.0)
    region_norm = np.sqrt(region_variance)

    # The patch is already centred, so this cross-correlation equals
    # sum((window - window_mean) * centred_patch) - the window's own mean
    # cancels against the patch's zero mean, which is what makes this a
    # NORMALIZED cross-correlation rather than a plain one.
    cross = fftconvolve(search_region, centred_patch[::-1, ::-1], mode="valid")

    # A window with no texture has no defined correlation: the denominator
    # goes to zero and the ratio is decided by the epsilon, not by the image.
    # Report zero - "no evidence here" - so a flat region lands at the bottom
    # of the margin distribution where it belongs, instead of at the top.
    textured = (region_norm / np.sqrt(pixel_count)) > MIN_REGION_STD
    surface = np.divide(
        cross,
        patch_norm * region_norm,
        out=np.zeros_like(cross),
        where=textured,
    )
    # NCC is bounded on [-1, 1] by Cauchy-Schwarz; floating-point error can
    # push it a hair outside, and nothing downstream should ever see a score
    # that a later reader would mistake for an unusually strong match.
    return np.clip(surface, -1.0, 1.0)


def peak_and_margin(surface: np.ndarray) -> Tuple[float, float, Tuple[int, int]]:
    """Best score, its lead over the best rival, and where the peak sits.

    The rival is searched outside a small exclusion zone around the peak, so
    the smooth shoulder of a genuine peak is not mistaken for a competitor.
    """

    peak_row, peak_column = np.unravel_index(int(np.argmax(surface)), surface.shape)
    best_score = float(surface[peak_row, peak_column])

    rivals = surface.copy()
    rivals[
        max(0, peak_row - 2):peak_row + 3,
        max(0, peak_column - 2):peak_column + 3,
    ] = -np.inf
    finite = np.isfinite(rivals)
    if not finite.any():
        # The exclusion zone swallowed the whole surface, which happens when
        # the search window is barely larger than the zone (a small
        # --search-radius). There is no rival to measure against, so the margin
        # is undefined - reported as 0.0, the "no evidence of a clear winner"
        # end of the scale, rather than as the +inf a -inf runner-up would give
        # and which would then count as a supremely confident match.
        return best_score, 0.0, (int(peak_row), int(peak_column))
    runner_up = float(np.max(rivals[finite]))
    return best_score, best_score - runner_up, (int(peak_row), int(peak_column))


def measure_pair(
    gray0: np.ndarray,
    gray1: np.ndarray,
    *,
    patch_size: int,
    search_radius: int,
    grid_rows: int,
    grid_columns: int,
) -> List[Dict[str, float]]:
    """Match a grid of patches from one frame into the next."""

    height, width = gray0.shape
    border = search_radius + patch_size
    if height <= 2 * border or width <= 2 * border:
        raise ValueError("image is too small for this patch size and search radius")

    results: List[Dict[str, float]] = []
    rows = np.linspace(border, height - border, grid_rows, dtype=int)
    columns = np.linspace(border, width - border, grid_columns, dtype=int)
    for row in rows:
        for column in columns:
            patch = gray0[row:row + patch_size, column:column + patch_size]
            # A flat patch has nothing to match; counting it as a failure would
            # blame the matcher for the scene.
            if float(patch.std()) < 0.01:
                continue
            region = gray1[
                row - search_radius:row + patch_size + search_radius,
                column - search_radius:column + patch_size + search_radius,
            ]
            surface = correlation_surface(patch, region)
            best_score, margin, (peak_row, peak_column) = peak_and_margin(surface)
            results.append({
                "peak": best_score,
                "margin": margin,
                "displacement_px": float(
                    np.hypot(peak_row - search_radius, peak_column - search_radius)
                ),
            })
    return results


def summarise(name: str, values: np.ndarray) -> Dict[str, float]:
    """Mean plus a few percentiles - a mean alone hides a bimodal failure."""

    return {
        "name": name,
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure whether consecutive frames can be matched at all.",
    )
    parser.add_argument("images", type=Path,
                        help="Folder of JPEG/PNG frames, sorted by filename.")
    parser.add_argument("--patch-size", type=int, default=32,
                        help="Patch side in pixels, at the working scale.")
    parser.add_argument("--search-radius", type=int, default=24,
                        help="How far the patch is allowed to move, in pixels.")
    parser.add_argument("--downscale", type=float, default=0.5,
                        help="Resize factor applied before matching. 0.5 keeps "
                             "the test fast. NOTE this moves the sharpness "
                             "number: downscaling averages away exactly the "
                             "high-frequency detail sharpness measures, so a "
                             "frame reads sharper at 1.0 than at 0.5. Compare "
                             "runs only at the SAME downscale, and read the "
                             "LOW_SHARPNESS threshold as calibrated for 0.5.")
    parser.add_argument("--grid", type=int, nargs=2, default=(4, 6),
                        metavar=("ROWS", "COLUMNS"),
                        help="How many patches to sample per frame pair.")
    parser.add_argument("--pair-stride", type=int, default=4,
                        help="Test every Nth consecutive pair.")
    parser.add_argument("--max-pairs", type=int, default=200,
                        help="Stop after this many pairs.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the full result as JSON here.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    frame_paths: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        frame_paths.extend(args.images.glob(pattern))
    frame_paths.sort()
    if len(frame_paths) < 2:
        print(f"need at least 2 frames in {args.images}")
        return 1

    sharpness_values: List[float] = []
    matches: List[Dict[str, float]] = []
    pairs_tested = 0

    for index in range(0, len(frame_paths) - 1, args.pair_stride):
        if pairs_tested >= args.max_pairs:
            break
        gray0 = load_grayscale(frame_paths[index], args.downscale)
        gray1 = load_grayscale(frame_paths[index + 1], args.downscale)
        sharpness_values.append(frame_sharpness(gray0))
        matches.extend(measure_pair(
            gray0, gray1,
            patch_size=args.patch_size,
            search_radius=args.search_radius,
            grid_rows=args.grid[0],
            grid_columns=args.grid[1],
        ))
        pairs_tested += 1

    if not matches:
        print("every sampled patch was flat - the scene carries no texture at all")
        return 1

    sharpness = np.asarray(sharpness_values)
    peaks = np.asarray([m["peak"] for m in matches])
    margins = np.asarray([m["margin"] for m in matches])
    displacements = np.asarray([m["displacement_px"] for m in matches])

    ambiguous_fraction = float(np.mean(margins < AMBIGUOUS_MARGIN))
    confident_fraction = float(np.mean(margins > CONFIDENT_MARGIN))

    print(f"frames          : {len(frame_paths)}")
    print(f"pairs tested    : {pairs_tested}")
    print(f"patches matched : {len(matches)}")
    print(f"working scale   : {args.downscale:g}x, patch {args.patch_size}px, "
          f"search +/-{args.search_radius}px\n")

    rows = [
        summarise("frame sharpness", sharpness),
        summarise("NCC peak", peaks),
        summarise("NCC peak margin", margins),
        summarise("displacement (px)", displacements),
    ]
    print(f"{'':20s} {'mean':>8s} {'median':>8s} {'p10':>8s} {'p90':>8s}")
    for row in rows:
        print(f"{row['name']:20s} {row['mean']:8.3f} {row['median']:8.3f} "
              f"{row['p10']:8.3f} {row['p90']:8.3f}")

    print(f"\nambiguous matches (margin < {AMBIGUOUS_MARGIN:.2f}) : "
          f"{100 * ambiguous_fraction:.1f}%")
    print(f"confident matches (margin > {CONFIDENT_MARGIN:.2f}) : "
          f"{100 * confident_fraction:.1f}%")

    # The verdict is deliberately blunt, because the cost of missing this is
    # weeks of tuning a model that was never given a usable signal.
    print()
    if float(np.median(sharpness)) < LOW_SHARPNESS:
        print("VERDICT: frames carry very little fine detail. Check for motion "
              "blur, over-compression or a broken video decode before training "
              "anything on them.")
    elif ambiguous_fraction > 0.35:
        print("VERDICT: most patches match ambiguously. The correlator output "
              "is dominated by arbitrary tie-breaks, not by real image motion.")
    elif confident_fraction < 0.20:
        print("VERDICT: borderline. Real matches exist but are a minority; "
              "expect a weak and noisy visual signal.")
    else:
        print("VERDICT: images support correspondence. If velocity still will "
              "not train, the cause is elsewhere.")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Provenance written BY the tool. Every number here moves with
        # these flags, so a file regenerated without them recorded is a
        # file that cannot be compared with anything.
        args.output.write_text(json.dumps({
            "provenance": {
                "tool": __file__.replace("\\\\", "/").split("/")[-1],
                "argv": sys.argv[1:],
                "images": str(args.images),
                "patch_size": int(args.patch_size),
                "search_radius": int(args.search_radius),
                "downscale": float(args.downscale),
                "grid": [int(v) for v in args.grid],
                "pair_stride": int(args.pair_stride),
                "max_pairs": int(args.max_pairs),
            },
            "frames": len(frame_paths),
            "pairs_tested": pairs_tested,
            "patches_matched": len(matches),
            "statistics": rows,
            "ambiguous_fraction": ambiguous_fraction,
            "confident_fraction": confident_fraction,
        }, indent=2))
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
