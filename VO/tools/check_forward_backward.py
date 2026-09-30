#!/usr/bin/env python3
"""Forward-backward consistency: find bad image pairs without ground truth.

Match a patch from frame A into frame B, then match the result back into
frame A. A correct match returns to where it started; a false one does not,
and the distance it fails to return by is the round-trip error.

    A ---- forward d1 ----> B
    B ---- backward d2 ---> A
    round-trip error = |d1 + d2|

This is the most useful check available on this dataset, because it needs NO
ground-truth flow. Peak margin (tools/check_image_matchability.py) says the
correlator found ONE clear answer; it cannot say the answer was right. A
repetitive texture - crop rows, waves, forest canopy - produces a confident,
unambiguous, and completely wrong match, and only a round trip catches it.

Deliberately MODEL-FREE. It measures the imagery with normalized cross
correlation rather than the trained frontend, so the answer is a property of
the data and does not move as a network trains. That also makes it runnable
before any training has happened.

**What a round trip cannot catch.** It detects an ASYMMETRIC failure - the
forward and backward matches disagreeing. It is blind to a SYMMETRIC one. When
the correlation surface is nearly flat, the argmax is an arbitrary tie-break,
but it is the same arbitrary tie-break in both directions: identical pixels,
identical interpolation, identical ordering. So an ambiguous cell can return
to its start perfectly while carrying no information at all.

That is exactly the regime this dataset sits in, which is why the two audits
must be read together rather than either one alone:

    check_image_matchability.py   52% ambiguous, 1.7% confident
    check_forward_backward.py     median round-trip 0.0 px, ~8% of pairs bad

Those are not in conflict. The first says the correlator usually cannot tell
its candidates apart; the second says the answer it settles on is at least
reproducible, with a small catastrophic tail. Consistency is a NECESSARY
condition for a good match, never a sufficient one.

This is an AUDIT tool. It reports which pairs would be excluded at each
threshold; it does not change training. Wire it into the pipeline only after
its exclusions have been shown to separate good pairs from bad ones on a
development split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT, _ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from check_image_matchability import (  # noqa: E402
    correlation_surface,
    load_grayscale,
    peak_and_margin,
)

#: Round-trip error, in pixels, above which a patch match is called
#: inconsistent. Sub-pixel error is normal; a whole pixel is not, at the scale
#: a correlation peak is localised to.
INCONSISTENT_PIXELS = 1.0
#: Fraction of a pair's patches that must be consistent for the PAIR to pass.
#: A pair is the unit training actually accepts or rejects, so the per-pair
#: verdict is the one that matters downstream.
MIN_CONSISTENT_FRACTION = 0.5


def match_once(
    source: np.ndarray,
    target: np.ndarray,
    row: int,
    column: int,
    *,
    patch_size: int,
    search_radius: int,
) -> Optional[Tuple[float, float, float]]:
    """Displacement of one patch from ``source`` into ``target``.

    Returns ``(dy, dx, margin)`` in pixels, or None when the patch or its
    search window runs off the image - a boundary case carries no information
    either way and must not be counted as an inconsistency.
    """

    height, width = source.shape
    if not (0 <= row and row + patch_size <= height):
        return None
    if not (0 <= column and column + patch_size <= width):
        return None
    top, bottom = row - search_radius, row + patch_size + search_radius
    left, right = column - search_radius, column + patch_size + search_radius
    if top < 0 or left < 0 or bottom > height or right > width:
        return None

    patch = source[row:row + patch_size, column:column + patch_size]
    if float(patch.std()) < 0.01:
        return None  # flat patch: nothing to match, and we know it

    surface = correlation_surface(patch, target[top:bottom, left:right])
    _, margin, (peak_row, peak_column) = peak_and_margin(surface)
    return (
        float(peak_row - search_radius),
        float(peak_column - search_radius),
        float(margin),
    )


def round_trip_error(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    row: int,
    column: int,
    *,
    patch_size: int,
    search_radius: int,
) -> Optional[Dict[str, float]]:
    """One patch's round trip A -> B -> A.

    The backward match starts from where the FORWARD match landed, not from
    the original location. Matching both directions at the same coordinates
    would measure something else entirely and would look consistent whenever
    the two directions happened to fail the same way.
    """

    forward = match_once(
        frame_a, frame_b, row, column,
        patch_size=patch_size, search_radius=search_radius,
    )
    if forward is None:
        return None
    forward_dy, forward_dx, forward_margin = forward

    backward = match_once(
        frame_b, frame_a,
        int(round(row + forward_dy)), int(round(column + forward_dx)),
        patch_size=patch_size, search_radius=search_radius,
    )
    if backward is None:
        return None
    backward_dy, backward_dx, backward_margin = backward

    return {
        "forward_px": float(np.hypot(forward_dy, forward_dx)),
        "round_trip_px": float(
            np.hypot(forward_dy + backward_dy, forward_dx + backward_dx)
        ),
        "min_margin": min(forward_margin, backward_margin),
    }


def audit_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    *,
    patch_size: int,
    search_radius: int,
    grid_rows: int,
    grid_columns: int,
) -> List[Dict[str, float]]:
    """Round-trip every patch on a grid over one image pair."""

    height, width = frame_a.shape
    border = search_radius + patch_size
    if height <= 2 * border or width <= 2 * border:
        raise SystemExit(
            "image is too small for this patch size and search radius; "
            "lower --patch-size or --search-radius"
        )

    results: List[Dict[str, float]] = []
    for row in np.linspace(border, height - border, grid_rows, dtype=int):
        for column in np.linspace(border, width - border, grid_columns, dtype=int):
            measured = round_trip_error(
                frame_a, frame_b, int(row), int(column),
                patch_size=patch_size, search_radius=search_radius,
            )
            if measured is not None:
                results.append(measured)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find false matches by checking they survive a round trip.",
    )
    parser.add_argument("images", type=Path, help="Folder of frames.")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--search-radius", type=int, default=24)
    parser.add_argument("--downscale", type=float, default=0.5,
                        help="Resize before matching. Round-trip errors are "
                             "reported in pixels AT THIS SCALE.")
    parser.add_argument("--grid", type=int, nargs=2, default=(4, 6),
                        metavar=("ROWS", "COLUMNS"))
    parser.add_argument("--pair-stride", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--thresholds", type=float, nargs="*",
                        default=(0.5, 1.0, 2.0, 4.0),
                        help="Round-trip errors, in pixels, to report an "
                             "exclusion rate for.")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not len(args.thresholds):
        raise SystemExit("--thresholds needs at least one value")

    frame_paths: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        frame_paths.extend(args.images.glob(pattern))
    frame_paths.sort()
    if len(frame_paths) < 2:
        raise SystemExit(f"need at least 2 frames in {args.images}")

    per_pair: List[Dict[str, float]] = []
    all_errors: List[float] = []

    for index in range(0, len(frame_paths) - 1, args.pair_stride):
        if len(per_pair) >= args.max_pairs:
            break
        frame_a = load_grayscale(frame_paths[index], args.downscale)
        frame_b = load_grayscale(frame_paths[index + 1], args.downscale)
        patches = audit_pair(
            frame_a, frame_b,
            patch_size=args.patch_size, search_radius=args.search_radius,
            grid_rows=args.grid[0], grid_columns=args.grid[1],
        )
        if not patches:
            continue
        errors = np.asarray([p["round_trip_px"] for p in patches])
        all_errors.extend(errors.tolist())
        # Both filenames and the interval between them, so a conclusion drawn
        # from this file can be checked without re-deriving anything from the
        # image folder - and so a later reader can see WHICH pair failed.
        try:
            interval = (
                int(frame_paths[index + 1].stem) - int(frame_paths[index].stem)
            ) / 1000.0
        except ValueError:
            interval = float("nan")
        per_pair.append({
            "frame": frame_paths[index].name,
            "next_frame": frame_paths[index + 1].name,
            "interval_s": interval,
            "patches": len(patches),
            "errors": errors.tolist(),
            "median_round_trip_px": float(np.median(errors)),
            "consistent_fraction": float(np.mean(errors <= INCONSISTENT_PIXELS)),
            "median_forward_px": float(
                np.median([p["forward_px"] for p in patches])
            ),
            "median_margin": float(np.median([p["min_margin"] for p in patches])),
        })

    if not per_pair:
        raise SystemExit("no pair produced a measurable patch")

    errors = np.asarray(all_errors)
    consistent = np.asarray([p["consistent_fraction"] for p in per_pair])

    print(f"pairs audited   : {len(per_pair)}")
    print(f"patches matched : {errors.size}")
    print(f"working scale   : {args.downscale:g}x, patch {args.patch_size}px, "
          f"search +/-{args.search_radius}px\n")

    forward = np.asarray([p["median_forward_px"] for p in per_pair])
    # A round trip is only evidence of a CORRECT match if something actually
    # moved: a patch that reports zero displacement in both directions returns
    # to its start trivially. Printed first so the round-trip numbers below are
    # read in the right light.
    print("forward displacement (px, per-pair median)")
    print(f"  median   {float(np.median(forward)):8.3f}")
    print(f"  p10      {float(np.percentile(forward, 10)):8.3f}")
    print(f"  p90      {float(np.percentile(forward, 90)):8.3f}")
    if float(np.median(forward)) < 1.0:
        print("  WARNING: the typical patch barely moves, so a successful "
              "round trip says little - check --downscale and the frame "
              "interval before reading the consistency numbers as a pass.")
    print()
    print("round-trip error (px, at the working scale)")
    for label, value in (
        ("median", float(np.median(errors))),
        ("mean", float(errors.mean())),
        ("p90", float(np.percentile(errors, 90))),
        ("p99", float(np.percentile(errors, 99))),
    ):
        print(f"  {label:8s} {value:8.3f}")

    print("\nexclusion rate by threshold")
    print(f"  {'threshold':>10s} {'patches cut':>12s} {'pairs cut':>11s}")
    exclusions = {}
    for threshold in args.thresholds:
        patch_cut = float(np.mean(errors > threshold))
        # A pair is cut when too few of ITS patches survive at THIS threshold -
        # the unit training accepts or rejects is the pair, not the patch.
        pair_consistent = np.asarray([
            float(np.mean(np.asarray(entry["errors"]) <= threshold))
            for entry in per_pair
        ])
        pair_cut = float(np.mean(pair_consistent < MIN_CONSISTENT_FRACTION))
        exclusions[f"{threshold:g}"] = {
            "patch_excluded_fraction": patch_cut,
            "pair_excluded_fraction": pair_cut,
        }
        print(f"  {threshold:10.2f} {100 * patch_cut:11.1f}% {100 * pair_cut:10.1f}%")

    print(f"\npairs with < {100 * MIN_CONSISTENT_FRACTION:.0f}% consistent patches: "
          f"{100 * float(np.mean(consistent < MIN_CONSISTENT_FRACTION)):.1f}%")

    print()
    median_error = float(np.median(errors))
    if median_error > 4.0:
        print("VERDICT: most matches do not survive a round trip. The "
              "correlator is reporting displacements that are not there - "
              "exclusion alone will not rescue this, the imagery has to "
              "improve.")
    elif median_error > INCONSISTENT_PIXELS:
        print("VERDICT: the typical match fails its round trip by more than a "
              "pixel. There is real signal here, but a substantial fraction of "
              "it is false - worth excluding, and worth measuring what that "
              "does to validation RMSE before wiring it in.")
    else:
        print("VERDICT: the typical match returns where it started. "
              "Round-trip exclusion will remove a tail, not a bulk - expect a "
              "modest gain.")

    # Derived statistics, computed and STORED here rather than left for a
    # reader to re-derive. The conclusion that frame interval does not predict
    # a bad pair - while displacement strongly does - is the kind of claim that
    # has to travel with the artifact, not alongside it in a message.
    intervals = np.asarray([entry["interval_s"] for entry in per_pair])
    finite = np.isfinite(intervals)

    def safe_corr(left, right):
        """None rather than a NaN when a correlation is not defined."""
        if left.size < 3 or np.std(left) == 0 or np.std(right) == 0:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    bins = []
    for low, high in ((0, 8), (8, 16), (16, 24), (24, 10000)):
        inside = (forward >= low) & (forward < high)
        if int(inside.sum()):
            bins.append({
                "forward_px_low": low,
                "forward_px_high": None if high > 1000 else high,
                "pairs": int(inside.sum()),
                "mean_consistent_fraction": float(consistent[inside].mean()),
            })

    derived = {
        "corr_interval_vs_consistency": (
            safe_corr(intervals[finite], consistent[finite]) if finite.any() else None
        ),
        "corr_interval_vs_displacement": (
            safe_corr(intervals[finite], forward[finite]) if finite.any() else None
        ),
        "corr_displacement_vs_consistency": safe_corr(forward, consistent),
        "displacement_bins": bins,
        "search_radius_px": int(args.search_radius),
        "pairs_over_75pct_of_radius": int((forward > 0.75 * args.search_radius).sum()),
    }

    print("")
    print("derived relationships")
    for label, key in (
        ("r(interval, consistency)", "corr_interval_vs_consistency"),
        ("r(interval, displacement)", "corr_interval_vs_displacement"),
        ("r(displacement, consistency)", "corr_displacement_vs_consistency"),
    ):
        value = derived[key]
        shown = "n/a" if value is None else ("%6.3f" % value)
        print("  %-30s %s" % (label, shown))
    if bins:
        print("  consistency by forward displacement:")
        for entry in bins:
            high = entry["forward_px_high"]
            span = ("%2d-%2d" % (entry["forward_px_low"], high)) if high else (">%2d  " % entry["forward_px_low"])
            print("    %s px : %3d pairs, mean consistency %.2f" % (
                span, entry["pairs"], entry["mean_consistent_fraction"]))

    print("\nNEXT: re-run validation with the excluded pairs withheld and "
          "compare val_vel_rmse. Only integrate this into training if the "
          "comparison shows a real separation.")

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
            "pairs_audited": len(per_pair),
            "patches_matched": int(errors.size),
            "settings": {
                "patch_size": args.patch_size,
                "search_radius": args.search_radius,
                "downscale": args.downscale,
                "grid": list(args.grid),
                "pair_stride": args.pair_stride,
            },
            "round_trip_px": {
                "median": float(np.median(errors)),
                "mean": float(errors.mean()),
                "p90": float(np.percentile(errors, 90)),
                "p99": float(np.percentile(errors, 99)),
            },
            "exclusions": exclusions,
            "derived": derived,
            "per_pair": [
                {k: v for k, v in entry.items() if k != "errors"}
                for entry in per_pair
            ],
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
