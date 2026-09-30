"""Splitting a capture into train/validation/test directories on disk.

Every property here is one a subtle off-by-one would break silently. The
fixtures use an IRREGULAR image clock on purpose: with a constant frame rate,
splitting by index and splitting by time agree, and the bug this file exists to
catch is invisible.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT / "src", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from tools.split_dataset import main, phase_bounds, summarise_images  # noqa: E402

COLUMNS = [
    "Time", "NavEulX", "NavEulY", "NavEulZ", "RelativeAlt",
    "GPSNavVnX", "GPSNavVnY", "GPSNavVnZ",
    "GPSNavEulX", "GPSNavEulY", "GPSNavEulZ", "Spare",
]


def _capture(root: Path, *, rows=5000, hz=50.0, image_hz=10.0, jitter=0.004):
    """A capture with a regular telemetry clock and a JITTERED image clock."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    times = np.arange(rows) / hz
    with (root / "flight.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for i, t in enumerate(times):
            writer.writerow(
                [f"{t:.6f}", 0.01, 0.02, 0.03, 100.0 + i * 0.01,
                 25.0, 0.0, 0.0, 0.01, 0.02, 0.03, "keepme"]
            )
    rs = np.random.RandomState(0)
    n_images = int(times[-1] * image_hz)
    stamps = np.arange(n_images) / image_hz + rs.uniform(0, jitter, n_images)
    stamps = np.sort(stamps)
    for s in stamps:
        ms = int(round(s * 1000))
        (root / "images" / f"{ms}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    return times, stamps


class TestPhaseBounds:
    def test_gap_is_discarded_between_phases(self):
        b = phase_bounds(1000, 0.6, 0.2, gap=50)
        assert b["train"] == (0, 600)
        assert b["validation"] == (650, 800)
        assert b["test"] == (850, 1000)
        assert b["validation"][0] - b["train"][1] == 50
        assert b["test"][0] - b["validation"][1] == 50

    def test_ranges_never_overlap(self):
        for total in (500, 1000, 7777):
            for gap in (1, 50, 300):
                b = phase_bounds(total, 0.6, 0.2, gap)
                spans = [b[k] for k in ("train", "validation", "test")]
                for (a0, a1), (c0, c1) in zip(spans, spans[1:]):
                    assert c0 >= a1, (total, gap)


class TestImageAssignment:
    def test_frames_are_assigned_by_time_not_index(self):
        """The property a constant-rate fixture cannot test."""

        telemetry = np.arange(0, 10, 0.02)
        # Deliberately uneven: a dense burst then a sparse tail.
        images = np.concatenate([np.linspace(0, 2, 40), np.linspace(2.5, 9.5, 8)])
        info = summarise_images(images, telemetry, 0.0, 5.0, frame_gap=1, latency=0.0)
        expected = int(np.count_nonzero((images >= 0.0) & (images <= 5.0)))
        assert info["kept"] == expected
        # Splitting by index would have taken the first 60% of 48 frames.
        assert info["kept"] != int(len(images) * 0.5)

    def test_frames_outside_the_range_are_excluded_both_ends(self):
        telemetry = np.arange(0, 10, 0.02)
        images = np.array([-1.0, 0.5, 5.0, 11.0])
        info = summarise_images(images, telemetry, 0.0, 10.0,
                                frame_gap=1, latency=0.0)
        assert info["excluded_before_split"] == 1
        assert info["excluded_after_split"] == 1
        assert info["kept"] == 2

    def test_a_kept_frame_is_always_bracketed_so_interpolation_never_clamps(self):
        """Frames outside telemetry could only be clamped, not interpolated."""

        telemetry = np.arange(2.0, 8.0, 0.02)
        images = np.linspace(0.0, 10.0, 101)
        lo, hi = float(telemetry[0]), float(telemetry[-1])
        info = summarise_images(images, telemetry, lo, hi, frame_gap=1, latency=0.0)
        kept = images[(images >= lo) & (images <= hi)]
        assert info["kept"] == kept.size
        assert kept.min() >= lo and kept.max() <= hi

    def test_deployment_latency_reduces_deliverable_pairs(self):
        """A frame within one latency of the end has nowhere to be delivered."""

        telemetry = np.arange(0, 10, 0.02)
        images = np.arange(0.0, 10.0, 0.5)
        no_latency = summarise_images(images, telemetry, 0.0, 10.0,
                                      frame_gap=1, latency=0.0)
        with_latency = summarise_images(images, telemetry, 0.0, 10.0,
                                        frame_gap=1, latency=1.0)
        assert with_latency["pairs_deliverable"] < no_latency["pairs_deliverable"]
        assert with_latency["frames_past_last_delivery"] > 0
        assert with_latency["kept"] == no_latency["kept"]  # still on disk

    def test_frame_gap_costs_pairs_not_frames(self):
        telemetry = np.arange(0, 10, 0.02)
        images = np.arange(0.0, 5.0, 0.5)
        one = summarise_images(images, telemetry, 0.0, 10.0, frame_gap=1, latency=0.0)
        three = summarise_images(images, telemetry, 0.0, 10.0, frame_gap=3, latency=0.0)
        assert one["kept"] == three["kept"]
        assert three["pairs_formable"] == one["pairs_formable"] - 2


class TestEndToEnd:
    def test_writes_a_disjoint_partition(self, tmp_path):
        src = tmp_path / "capture"
        _capture(src)
        out = tmp_path / "split"
        assert main(["--dataset", str(src), "--output-root", str(out)]) == 0

        names, spans = {}, {}
        for phase in ("train", "validation", "test"):
            names[phase] = {p.name for p in (out / phase / "images").iterdir()}
            with (out / phase / "flight.csv").open(newline="", encoding="utf-8-sig") as fh:
                rows = list(csv.reader(fh))
            assert rows[0] == COLUMNS, "every column must survive the split"
            t = np.array([float(r[0]) for r in rows[1:]])
            spans[phase] = (t[0], t[-1])

        for a, b in (("train", "validation"), ("validation", "test"), ("train", "test")):
            assert not (names[a] & names[b]), f"{a} and {b} share images"
        assert spans["validation"][0] > spans["train"][1]
        assert spans["test"][0] > spans["validation"][1]

    def test_every_dropped_frame_falls_in_a_gap(self, tmp_path):
        """Frames may only be lost to the gaps, never to an off-by-one.

        Asserting a drop PERCENTAGE would be meaningless: the gap is a fixed
        number of ticks, so its share of the capture depends entirely on how
        long the capture is. The invariant that actually matters is that every
        missing frame sits in a discarded interval.
        """

        src = tmp_path / "capture"
        _capture(src)
        out = tmp_path / "split"
        main(["--dataset", str(src), "--output-root", str(out)])

        kept_spans = []
        written = set()
        for phase in ("train", "validation", "test"):
            written |= {p.name for p in (out / phase / "images").iterdir()}
            with (out / phase / "flight.csv").open(newline="", encoding="utf-8-sig") as fh:
                rows = list(csv.reader(fh))
            t = [float(r[0]) for r in rows[1:]]
            kept_spans.append((t[0], t[-1]))

        source = {p.name for p in (src / "images").iterdir()}
        assert written <= source
        for name in source - written:
            when = int(name.split(".")[0]) / 1000.0
            in_a_span = any(lo <= when <= hi for lo, hi in kept_spans)
            assert not in_a_span, f"{name} at {when}s was dropped from inside a split"

    def test_the_source_capture_is_untouched(self, tmp_path):
        src = tmp_path / "capture"
        _capture(src)
        before = sorted(p.name for p in (src / "images").iterdir())
        csv_before = (src / "flight.csv").read_bytes()
        main(["--dataset", str(src), "--output-root", str(tmp_path / "split")])
        assert sorted(p.name for p in (src / "images").iterdir()) == before
        assert (src / "flight.csv").read_bytes() == csv_before

    def test_dry_run_writes_nothing(self, tmp_path):
        src = tmp_path / "capture"
        _capture(src)
        out = tmp_path / "split"
        assert main(["--dataset", str(src), "--output-root", str(out), "--dry-run"]) == 0
        assert not out.exists()

    def test_report_records_the_resolved_columns(self, tmp_path):
        import json

        src = tmp_path / "capture"
        _capture(src)
        out = tmp_path / "split"
        main(["--dataset", str(src), "--output-root", str(out)])
        report = json.loads((out / "split_report.json").read_text(encoding="utf-8"))
        assert report["altitude_column"] == "RelativeAlt"
        assert report["attitude_columns"] == ["NavEulX", "NavEulY", "NavEulZ"]
        assert report["images_duplicated"] == 0


class TestRefusals:
    def test_non_monotonic_telemetry_is_refused(self, tmp_path):
        src = tmp_path / "capture"
        _capture(src, rows=200)
        text = (src / "flight.csv").read_text(encoding="utf-8").splitlines()
        header, body = text[0], text[1:]
        body[10], body[11] = body[11], body[10]
        (src / "flight.csv").write_text("\n".join([header] + body), encoding="utf-8")
        with pytest.raises(SystemExit, match="not strictly increasing"):
            main(["--dataset", str(src), "--output-root", str(tmp_path / "o")])

    def test_an_empty_split_is_refused(self, tmp_path):
        src = tmp_path / "capture"
        _capture(src, rows=200)
        with pytest.raises(SystemExit, match="empty"):
            main(["--dataset", str(src), "--output-root", str(tmp_path / "o"),
                  "--gap-ticks", "5000"])

    def test_a_missing_altitude_column_is_refused(self, tmp_path):
        src = tmp_path / "capture"
        _capture(src, rows=200)
        text = (src / "flight.csv").read_text(encoding="utf-8")
        (src / "flight.csv").write_text(
            text.replace("RelativeAlt", "SomethingElse", 1), encoding="utf-8"
        )
        with pytest.raises(SystemExit, match="altitude"):
            main(["--dataset", str(src), "--output-root", str(tmp_path / "o")])


class TestAltitudeColumnMatchesProduction:
    """The synthetic capture must resolve the SAME altitude column as a real one.

    It did not: the generator emitted Barometer and no RelativeAlt, so every
    smoke run exercised the fallback branch while the real capture resolved
    RelativeAlt. A run against the mock then recorded a different
    source_altitude_column in input_contract.json than production would, and
    the branch that actually ships was never exercised end to end.
    """

    def test_the_generator_emits_relativealt(self):
        from tools.make_synthetic_flight import CSV_COLUMNS

        columns = CSV_COLUMNS.split(",") if isinstance(CSV_COLUMNS, str) else list(CSV_COLUMNS)
        assert "RelativeAlt" in columns
        assert "Barometer" in columns
        # RelativeAlt must come first in the file only incidentally; what matters
        # is that the resolver prefers it, which ALTITUDE_CANDIDATES decides.
        from vio.data.attitude import ALTITUDE_CANDIDATES

        assert ALTITUDE_CANDIDATES.index("RelativeAlt") < ALTITUDE_CANDIDATES.index("Barometer")

    def test_relativealt_wins_and_the_two_are_not_interchangeable(self, tmp_path):
        """If they held the same value, a fallback bug would look like success."""

        from vio.data.attitude import load_attitude_altitude

        path = tmp_path / "flight.csv"
        rows = ["Time,Barometer,RelativeAlt,NavEulX,NavEulY,NavEulZ"]
        for i in range(50):
            # Attitude must actually move: the loader refuses a constant column
            # as carrying no attitude, which is a guard worth not defeating.
            roll = 0.10 * np.sin(i * 0.13)
            pitch = 0.05 * np.cos(i * 0.11)
            yaw = 0.02 * i
            rows.append(
                f"{i * 0.02},{620.0 + i * 0.01},{120.0 + i * 0.01},"
                f"{roll:.6f},{pitch:.6f},{yaw:.6f}"
            )
        path.write_text("\n".join(rows), encoding="utf-8")

        source = load_attitude_altitude(path)
        assert source.altitude_column == "RelativeAlt"
        assert 119.0 < float(np.median(source.altitude_m)) < 121.0, (
            "resolved the pressure altitude instead of the height above takeoff"
        )
