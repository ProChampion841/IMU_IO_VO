"""Alignment between an irregular image clock and an irregular telemetry clock.

The shipped smoke dataset has images at exactly 500 ms and telemetry at exactly
20 ms, both perfectly regular and sharing a phase. Every property in this file
is invisible under that data: snapping equals interpolating when the query is
already a sample, a constant dt is correct when the clock is constant, and no
gap exists to reject. So the fixtures here are deliberately jittered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT / "src", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from vio.data.images import (  # noqa: E402
    interpolate_at,
    mean_over_interval,
    resolve_time_offsets,
)


class TestInterpolateAt:
    def test_reproduces_the_worked_example(self):
        """h(1.024) between (1.020, 100.00) and (1.030, 100.10) is 100.04."""

        times = np.array([1.020, 1.030])
        altitude = np.array([100.00, 100.10])
        got = float(interpolate_at(times, altitude, np.array([1.024]))[0])
        assert got == pytest.approx(100.04, abs=1e-9)

    def test_is_exact_on_a_linear_signal(self):
        times = np.array([0.0, 0.37, 1.0, 2.5, 2.51, 4.0])  # deliberately uneven
        values = 3.0 * times - 7.0
        query = np.array([0.1, 0.9, 2.505, 3.9])
        assert interpolate_at(times, values, query) == pytest.approx(3.0 * query - 7.0)

    def test_holds_end_values_rather_than_extrapolating(self):
        """Past the end of telemetry there is no measurement to extrapolate from."""

        times = np.array([10.0, 11.0])
        values = np.array([100.0, 200.0])
        assert float(interpolate_at(times, values, np.array([-5.0]))[0]) == 100.0
        assert float(interpolate_at(times, values, np.array([99.0]))[0]) == 200.0

    def test_handles_a_vector_signal(self):
        times = np.array([0.0, 1.0])
        values = np.array([[0.0, 10.0, -4.0], [2.0, 20.0, -8.0]])
        got = interpolate_at(times, values, np.array([0.25]))
        assert got.shape == (1, 3)
        assert got[0] == pytest.approx([0.5, 12.5, -5.0])

    def test_a_single_sample_is_held_everywhere(self):
        got = interpolate_at(np.array([5.0]), np.array([42.0]), np.array([0.0, 9.0]))
        assert got == pytest.approx([42.0, 42.0])

    def test_repeated_timestamps_do_not_divide_by_zero(self):
        times = np.array([0.0, 1.0, 1.0, 2.0])
        values = np.array([0.0, 1.0, 1.0, 2.0])
        got = interpolate_at(times, values, np.array([1.0]))
        assert np.all(np.isfinite(got))


class TestMeanOverInterval:
    def test_matches_the_analytic_mean_of_a_ramp(self):
        """A linear ramp's time-average is its midpoint value, exactly."""

        times = np.arange(0.0, 1.0, 0.01)          # 100 Hz
        rate = np.deg2rad(20.0) * times[:, None] * np.ones((1, 3))
        start, stop = 0.4049, 0.4549               # a 20 Hz pair, worst-case phase
        expected = np.deg2rad(20.0) * 0.5 * (start + stop)
        got = mean_over_interval(times, rate, start, stop)
        assert got == pytest.approx([expected] * 3, abs=1e-12)

    def test_beats_snapping_to_the_nearest_sample(self):
        """The property that motivates the change, stated as a comparison."""

        times = np.arange(0.0, 1.0, 0.01)
        rate = np.deg2rad(20.0) * times[:, None]
        start, stop = 0.4049, 0.4549
        truth = np.deg2rad(20.0) * 0.5 * (start + stop)

        exact = float(mean_over_interval(times, rate, start, stop)[0])
        i0 = int(np.argmin(np.abs(times - start)))
        i1 = int(np.argmin(np.abs(times - stop)))
        window = rate[i0 : i1 + 1, 0]
        step = np.diff(times[i0 : i1 + 1])
        snapped = float(
            np.sum(0.5 * (window[1:] + window[:-1]) * step) / (times[i1] - times[i0])
        )
        assert abs(exact - truth) < abs(snapped - truth)
        assert abs(exact - truth) < 1e-12

    def test_is_symmetric_in_its_endpoints(self):
        times = np.array([0.0, 0.5, 1.0])
        values = np.array([0.0, 4.0, 6.0])
        assert mean_over_interval(times, values, 0.2, 0.8) == pytest.approx(
            mean_over_interval(times, values, 0.8, 0.2)
        )

    def test_a_zero_length_interval_is_the_instantaneous_value(self):
        times = np.array([0.0, 1.0])
        values = np.array([0.0, 10.0])
        assert float(mean_over_interval(times, values, 0.3, 0.3)) == pytest.approx(3.0)

    def test_an_uneven_clock_does_not_overweight_dense_stretches(self):
        """Constant signal, wildly uneven sampling: the mean is still the constant."""

        times = np.array([0.0, 0.001, 0.002, 0.003, 0.5, 1.0])
        values = np.full(times.shape, 7.5)
        assert float(mean_over_interval(times, values, 0.0, 1.0)) == pytest.approx(7.5)


class TestClockOffset:
    def test_a_constant_offset_shifts_every_frame(self):
        stamps = np.array([0.0, 0.05, 0.10])
        assert resolve_time_offsets(stamps, 0.25) == pytest.approx([0.25, 0.25, 0.25])

    def test_a_table_corrects_drift_along_the_flight(self):
        """A drifting clock needs a per-frame correction, not one average."""

        table = {"times_s": [0.0, 100.0], "offsets_s": [0.00, 0.20]}
        stamps = np.array([0.0, 25.0, 50.0, 100.0])
        got = resolve_time_offsets(stamps, table)
        assert got == pytest.approx([0.0, 0.05, 0.10, 0.20])
        # A single averaged offset would misplace the ends in opposite
        # directions. The samples here are not evenly spaced, so the mean is
        # 0.0875 s, and the worst end is out by 0.1125 s -- eleven telemetry
        # samples at 100 Hz, against a 50 ms frame interval.
        worst = float(np.max(np.abs(got - got.mean())))
        assert worst == pytest.approx(0.1125)
        assert worst > 2 * 0.05

    def test_a_table_holds_its_end_values(self):
        table = {"times_s": [10.0, 20.0], "offsets_s": [1.0, 2.0]}
        got = resolve_time_offsets(np.array([0.0, 99.0]), table)
        assert got == pytest.approx([1.0, 2.0])

    @pytest.mark.parametrize(
        "bad",
        [
            {"times_s": [1.0, 0.0], "offsets_s": [0.0, 1.0]},   # not increasing
            {"times_s": [0.0, 1.0], "offsets_s": [0.0]},        # length mismatch
            {"times_s": [], "offsets_s": []},                   # empty
            {"times_s": [0.0, np.inf], "offsets_s": [0.0, 1.0]},  # not finite
        ],
    )
    def test_a_malformed_table_is_refused(self, bad):
        with pytest.raises(ValueError):
            resolve_time_offsets(np.array([0.5]), bad)

    def test_a_non_finite_constant_is_refused(self):
        with pytest.raises(ValueError):
            resolve_time_offsets(np.array([0.5]), float("nan"))


class TestTelemetryDropoutGuard:
    """An exposure inside a hole in the telemetry.

    It still finds a nearest sample, still lands inside the range, and still
    forms an event. Its attitude and altitude are then interpolated ACROSS the
    hole, from measurements taken a long way away, which is not a measurement.
    """

    @staticmethod
    def _source(telemetry, image_times, **kwargs):
        import tempfile

        from PIL import Image

        from vio.data.image_pairs import VisualPairSource

        root = Path(tempfile.mkdtemp()) / "ds"
        (root / "images").mkdir(parents=True)
        for t in image_times:
            Image.new("L", (8, 8)).save(root / "images" / f"{int(round(t * 1000))}.jpg")
        return VisualPairSource(root, telemetry, image_size=(8, 8), **kwargs)

    def test_a_healthy_jittered_clock_is_untouched(self):
        rs = np.random.RandomState(3)
        telemetry = np.cumsum(rs.uniform(0.008, 0.012, 2000))
        telemetry -= telemetry[0]
        images = np.cumsum(rs.uniform(0.040, 0.060, 80)) + 0.7
        source = self._source(telemetry, images)
        assert source.rejected_telemetry_gap_pairs == 0
        assert source.plan.ready_tick.size > 70

    def test_events_straddling_a_dropout_are_rejected(self):
        telemetry = np.concatenate([np.arange(0, 10, 0.01), np.arange(13, 30, 0.01)])
        images = np.arange(0.7, 29.0, 0.05)
        guarded = self._source(telemetry, images)
        unguarded = self._source(telemetry, images, max_telemetry_gap_s=1e9)

        assert guarded.rejected_telemetry_gap_pairs > 0
        assert guarded.plan.ready_tick.size < unguarded.plan.ready_tick.size

        def worst(source):
            return float(np.max(np.abs(
                source.plan.exposure_t0_s
                - telemetry[source.plan.telemetry_index0]
            )))

        # The point of the guard: nothing surviving is matched across the hole.
        assert worst(guarded) < 0.05
        assert worst(unguarded) > 1.0

    def test_the_threshold_scales_with_the_telemetry_rate(self):
        """8x the median step: generous on any healthy clock, tight on a hole."""

        fast = self._source(np.arange(0, 20, 0.01), np.arange(0.5, 19.0, 0.05))
        slow = self._source(np.arange(0, 20, 0.10), np.arange(0.5, 19.0, 0.50))
        assert slow.max_telemetry_gap_s > fast.max_telemetry_gap_s
        assert fast.max_telemetry_gap_s == pytest.approx(0.08, rel=0.1)

    def test_a_constant_clock_offset_is_NOT_caught(self):
        """Stated as a test so nobody mistakes this guard for an offset check.

        Dense telemetry always has a near neighbour, so a wholesale camera-clock
        error pairs happily with a small residual while reading every value from
        the wrong moment. Only correlating image motion against telemetry
        rotation finds it -- tools/estimate_time_offset.py, --image-time-offset.
        """

        telemetry = np.arange(0, 30, 0.01)
        images = np.arange(0.7, 20.0, 0.05)
        honest = self._source(telemetry, images)
        offset = self._source(telemetry, images, image_time_offset_s=5.0)
        assert offset.rejected_telemetry_gap_pairs == 0
        assert offset.plan.ready_tick.size == honest.plan.ready_tick.size
