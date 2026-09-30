"""Split one capture into train/validation/test directories, on disk.

The trainer can cut a capture chronologically at load time. This does the same
cut once, to files, and that difference matters: after this runs, the three
splits are different folders holding different images and different telemetry
rows. A window cannot straddle a boundary, an image cannot be read by two
phases, and no flag can be set wrongly to make it happen. The split stops being
something to trust and becomes something to look at.

WHAT IT HAS TO GET RIGHT, and why each one is a real failure and not a detail:

1.  TWO CLOCKS. Telemetry rows carry their own timestamps; images carry theirs
    in their filenames. The split boundary is chosen on the telemetry clock and
    then applied to the images by TIME, never by count or by index. Splitting
    images by index assumes a constant frame rate, which is exactly the
    assumption the capture violates.

2.  PAIRS, NOT FRAMES. The model consumes image PAIRS ``(i, i + frame_gap)``.
    A boundary that lands between two frames of a pair destroys that pair. It
    cannot corrupt anything - the two frames end up in different folders and
    are never paired - but it is a silent loss, so it is counted and reported.

3.  DEPLOYMENT LATENCY. An event is consumed at ``exposure_t1 + latency``, not
    at its shutter time. A frame captured within one latency of a split's END
    has no telemetry left to be delivered to and is dropped by the loader.
    Those frames are counted here rather than discovered as a quiet shortfall.

4.  INTERPOLATION NEEDS BRACKETS. Attitude and altitude are interpolated to
    each exposure instant. An exposure outside the telemetry range cannot be
    interpolated, only clamped to the end value - a silently worse number that
    looks like every other. Frames not bracketed by telemetry on BOTH sides are
    excluded from their split, so every surviving frame is interpolatable.

5.  A REAL GAP. Two adjacent splits share the boundary instant. A window ending
    at the last training tick and one starting at the first validation tick are
    a fraction of a second apart, and the recurrent state makes them nearly the
    same sample. A gap of at least one window is discarded between splits, and
    the discarded rows go to no split at all.

Nothing is deleted: the source capture is read only, and the splits are written
beside it.

    python tools/split_dataset.py --dataset data --output-root data_split
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The package lives under src/; tools/ is imported as a package from the
# repository root. Both have to be importable when a script is run directly.
for _entry in (ROOT / "src", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from vio.data.attitude import (  # noqa: E402
    ALTITUDE_CANDIDATES,
    resolve_altitude_column,
    resolve_attitude_columns,
)
from vio.data.images import numeric_image_manifest, resolve_time_offsets  # noqa: E402

PHASES = ("train", "validation", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--csv-name", default="flight.csv")
    parser.add_argument("--image-folder", default="images")
    parser.add_argument("--image-pattern", default="*.jpg")
    parser.add_argument("--time-column", default="Time")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--image-time-scale", type=float, default=0.001)
    parser.add_argument(
        "--image-time-offset", type=float, default=0.0,
        help="Constant seconds added to every image timestamp before the split "
             "is applied, to put the camera clock on the telemetry clock. Get "
             "this WRONG and frames land in the wrong split near every "
             "boundary, which is leakage.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--gap-ticks", type=int, default=600,
        help="Telemetry rows discarded between splits. Must be at least the "
             "trainer's --window-length, or a training window and a validation "
             "window can end and begin a fraction of a second apart.",
    )
    parser.add_argument(
        "--frame-gap", type=int, default=1,
        help="The trainer's --frame-gap, used only to count pairs lost at the "
             "boundaries.",
    )
    parser.add_argument(
        "--deployment-latency-s", type=float, default=0.35,
        help="The trainer's --deployment-latency-s, used to count frames near "
             "each split's end that will have no telemetry to be delivered to.",
    )
    parser.add_argument(
        "--link", action="store_true",
        help="Hard-link images instead of copying them. Same bytes on disk once "
             "rather than twice; the source must be on the same filesystem.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report the split without writing anything.",
    )
    return parser


def read_telemetry(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Header and rows, as strings. Values are copied through untouched.

    Deliberately not parsed into floats and re-serialised: re-formatting every
    column would silently change precision on 116 of them to split on one.
    """

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [row for row in reader if row]
    return header, rows


def phase_bounds(total: int, train_fraction: float, validation_fraction: float,
                 gap: int) -> Dict[str, Tuple[int, int]]:
    """Row index ranges per phase, with the gap discarded between them."""

    train_end = int(total * train_fraction)
    validation_end = train_end + int(total * validation_fraction)
    return {
        "train": (0, train_end),
        "validation": (train_end + gap, validation_end),
        "test": (validation_end + gap, total),
    }


def summarise_images(
    times: np.ndarray,
    telemetry_times: np.ndarray,
    lo: float,
    hi: float,
    *,
    frame_gap: int,
    latency: float,
) -> Dict[str, object]:
    """Which frames belong to a split, and what each exclusion rule costs.

    ``lo``/``hi`` are the split's first and last telemetry instants. A frame is
    kept only when it is BRACKETED by them: an exposure outside cannot have its
    attitude and altitude interpolated, only clamped, and a clamped value is
    indistinguishable from a measured one downstream.
    """

    inside = (times >= lo) & (times <= hi)
    kept = np.flatnonzero(inside)
    before = int(np.count_nonzero(times < lo))
    after = int(np.count_nonzero(times > hi))

    # Frames whose event would be delivered past the end of this split's
    # telemetry. They are kept on disk (they are still valid second frames of a
    # pair) but they cannot themselves complete an event.
    undeliverable = int(np.count_nonzero(inside & (times + latency > hi)))

    # Pairs are (i, i+frame_gap) within the kept run.
    pairs = max(0, kept.size - frame_gap)
    usable = 0
    if kept.size > frame_gap:
        second = times[kept[frame_gap:]]
        usable = int(np.count_nonzero(second + latency <= hi))

    return {
        "kept": int(kept.size),
        "excluded_before_split": before,
        "excluded_after_split": after,
        "pairs_formable": int(pairs),
        "pairs_deliverable": usable,
        "frames_past_last_delivery": undeliverable,
        "first_time_s": float(times[kept[0]]) if kept.size else None,
        "last_time_s": float(times[kept[-1]]) if kept.size else None,
        "indices": kept,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = Path(args.dataset).expanduser().resolve()
    csv_path = dataset / args.csv_name
    image_dir = dataset / args.image_folder
    if not csv_path.is_file():
        raise SystemExit(f"No telemetry at {csv_path}")
    if not image_dir.is_dir():
        raise SystemExit(f"No image folder at {image_dir}")

    header, rows = read_telemetry(csv_path)
    if args.time_column not in header:
        raise SystemExit(
            f"{csv_path} has no {args.time_column!r} column; it has "
            f"{len(header)} columns beginning {header[:4]}"
        )
    time_index = header.index(args.time_column)
    telemetry_times = np.asarray(
        [float(row[time_index]) for row in rows], dtype=np.float64
    ) * float(args.time_scale)
    total = telemetry_times.size
    if total < 2:
        raise SystemExit("Telemetry holds fewer than two rows")
    if np.any(np.diff(telemetry_times) <= 0):
        raise SystemExit(
            "Telemetry timestamps are not strictly increasing. A split cut on a "
            "non-monotonic clock does not partition the flight."
        )

    # Report which columns the trainer will resolve, so a split made against
    # one altitude source cannot be trained against another without notice.
    attitude_columns, attitude_note = resolve_attitude_columns(header)
    try:
        altitude_column = resolve_altitude_column(header)
    except ValueError as error:
        raise SystemExit(
            f"{error}\nThe split would produce directories the trainer cannot "
            f"read. Looked for {list(ALTITUDE_CANDIDATES)}."
        ) from error

    paths, image_times = numeric_image_manifest(
        image_dir, args.image_pattern, args.image_time_scale
    )
    image_times = image_times + resolve_time_offsets(
        image_times, float(args.image_time_offset)
    )

    bounds = phase_bounds(
        total, args.train_fraction, args.validation_fraction, args.gap_ticks
    )
    for name, (start, end) in bounds.items():
        if end - start <= 0:
            raise SystemExit(
                f"The {name} split is empty ({start}..{end}) at "
                f"--train-fraction {args.train_fraction} "
                f"--validation-fraction {args.validation_fraction} with "
                f"--gap-ticks {args.gap_ticks} over {total} rows."
            )

    print(f"source:     {dataset}")
    print(f"telemetry:  {total} rows, "
          f"{(telemetry_times[-1] - telemetry_times[0]) / 60:.2f} min, "
          f"{1.0 / float(np.median(np.diff(telemetry_times))):.1f} Hz median")
    print(f"images:     {len(paths)} frames, "
          f"{1.0 / float(np.median(np.diff(image_times))):.1f} Hz median")
    print(f"attitude:   {', '.join(attitude_columns)} ({attitude_note})")
    print(f"altitude:   {altitude_column}")
    print()

    report: Dict[str, object] = {
        "source": str(dataset),
        "rows": total,
        "images": len(paths),
        "attitude_columns": list(attitude_columns),
        "altitude_column": altitude_column,
        "gap_ticks": int(args.gap_ticks),
        "frame_gap": int(args.frame_gap),
        "deployment_latency_s": float(args.deployment_latency_s),
        "image_time_offset_s": float(args.image_time_offset),
        "splits": {},
    }

    assigned = np.zeros(len(paths), dtype=np.int8)
    out_root = Path(args.output_root).expanduser().resolve()

    for name in PHASES:
        start, end = bounds[name]
        lo = float(telemetry_times[start])
        hi = float(telemetry_times[end - 1])
        info = summarise_images(
            image_times, telemetry_times, lo, hi,
            frame_gap=args.frame_gap, latency=args.deployment_latency_s,
        )
        indices = info.pop("indices")
        assigned[indices] += 1

        minutes = (hi - lo) / 60.0
        print(f"{name:11} rows [{start:>7}, {end:>7})  {minutes:6.2f} min  "
              f"{(end - start) / total:5.1%}")
        print(f"{'':11} frames {info['kept']:>5}  "
              f"pairs {info['pairs_formable']:>5} formable, "
              f"{info['pairs_deliverable']:>5} deliverable "
              f"({info['frames_past_last_delivery']} past last delivery)")
        report["splits"][name] = dict(
            info, rows=[int(start), int(end)],
            first_time_s=lo, last_time_s=hi, minutes=minutes,
        )

    # The properties that make this a partition rather than three overlapping
    # views. Checked rather than assumed, because every one of them is exactly
    # what a subtle off-by-one would break.
    duplicated = int(np.count_nonzero(assigned > 1))
    orphaned = int(np.count_nonzero(assigned == 0))
    print()
    print(f"images in more than one split: {duplicated}")
    print(f"images in no split (gaps/ends): {orphaned}")
    if duplicated:
        raise SystemExit(
            f"{duplicated} image(s) fall in more than one split. That is "
            "leakage; refusing to write."
        )
    for earlier, later in (("train", "validation"), ("validation", "test")):
        a_end = report["splits"][earlier]["last_time_s"]
        b_start = report["splits"][later]["first_time_s"]
        if b_start <= a_end:
            raise SystemExit(
                f"{later} starts at {b_start} which is not after {earlier} ends "
                f"at {a_end}. The gap is not doing its job."
            )
        print(f"gap {earlier}->{later}: {b_start - a_end:.3f} s")
    report["images_duplicated"] = duplicated
    report["images_unassigned"] = orphaned

    if args.dry_run:
        print("\n--dry-run: nothing written")
        print(json.dumps(report["splits"], indent=2, default=str)[:0] or "", end="")
        return 0

    for name in PHASES:
        start, end = bounds[name]
        split_dir = out_root / name
        (split_dir / args.image_folder).mkdir(parents=True, exist_ok=True)
        with (split_dir / args.csv_name).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows[start:end])
        lo = float(telemetry_times[start])
        hi = float(telemetry_times[end - 1])
        keep = np.flatnonzero((image_times >= lo) & (image_times <= hi))
        for i in keep:
            source_path = paths[int(i)]
            target = split_dir / args.image_folder / source_path.name
            if target.exists():
                target.unlink()
            if args.link:
                target.hardlink_to(source_path)
            else:
                shutil.copy2(source_path, target)
        print(f"wrote {split_dir}  ({end - start} rows, {keep.size} images)")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "split_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nwrote {out_root / 'split_report.json'}")
    print("\ntrain with:")
    print(f"  python tools/train_fixedwing_vo.py \\")
    print(f"      --dataset {out_root / 'train'} \\")
    print(f"      --validation-dataset {out_root / 'validation'} \\")
    print(f"      --test-dataset {out_root / 'test'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
