"""The GPS-antenna-to-camera lever arm in the supervision target.

GPS measures velocity at the antenna. The camera is what sees the motion the
network is asked to explain. A rigid body rotating at ``omega`` moves its points
at different velocities, so on a wing mount the two differ in every turn:

    v_camera = v_gps + omega x r_gps->camera

The error is proportional to turn rate, which makes it correlated with exactly
the manoeuvres the estimator is judged on rather than something that averages
away over a flight.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT / "src", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from vio.data.fixedwing_vo import reference_body_velocity  # noqa: E402


def _flight(rates_rad_s, *, forward=25.0, n=200, dt=0.02):
    """A CSV whose reference attitude rotates at a constant body rate."""

    times = np.arange(n) * dt
    angle = np.asarray(rates_rad_s, dtype=float)[None, :] * times[:, None]
    lines = ["Time,GPSNavVnX,GPSNavVnY,GPSNavVnZ,GPSNavEulX,GPSNavEulY,GPSNavEulZ"]
    for i in range(n):
        lines.append(
            f"{times[i]},{forward},0.0,0.0,"
            f"{angle[i, 0]},{angle[i, 1]},{angle[i, 2]}"
        )
    path = Path(tempfile.mkdtemp()) / "flight.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class TestLeverArm:
    def test_absent_by_default(self):
        """Every checkpoint written before this existed must still reproduce."""

        path = _flight([0.0, 0.0, 0.0])
        _, plain = reference_body_velocity(path)
        _, none = reference_body_velocity(path, lever_arm_m=None)
        assert np.array_equal(plain, none)

    def test_a_zero_arm_changes_nothing(self):
        path = _flight([np.deg2rad(30.0), 0.0, 0.0])
        _, plain = reference_body_velocity(path)
        _, zero = reference_body_velocity(path, lever_arm_m=[0.0, 0.0, 0.0])
        assert np.array_equal(plain, zero)

    def test_matches_omega_cross_r_for_a_roll(self):
        """30 deg/s roll, camera 2 m out the wing: 1.047 m/s, downward."""

        rate = np.deg2rad(30.0)
        path = _flight([rate, 0.0, 0.0])
        _, plain = reference_body_velocity(path)
        _, armed = reference_body_velocity(path, lever_arm_m=[0.0, 2.0, 0.0])

        expected = np.cross([rate, 0.0, 0.0], [0.0, 2.0, 0.0])  # -> [0, 0, 2*rate]
        assert armed[10] - plain[10] == pytest.approx(expected, abs=1e-4)
        assert float(np.linalg.norm(armed[10] - plain[10])) == pytest.approx(
            rate * 2.0, abs=1e-4
        )

    def test_matches_omega_cross_r_for_a_yaw(self):
        """A turn moves a nose-mounted camera sideways relative to the antenna."""

        rate = np.deg2rad(20.0)
        path = _flight([0.0, 0.0, rate])
        _, plain = reference_body_velocity(path)
        _, armed = reference_body_velocity(path, lever_arm_m=[1.5, 0.0, 0.0])

        expected = np.cross([0.0, 0.0, rate], [1.5, 0.0, 0.0])  # -> [0, 1.5*rate, 0]
        assert armed[10] - plain[10] == pytest.approx(expected, abs=1e-4)

    def test_no_correction_without_rotation(self):
        """Straight and level, the antenna and the camera move identically."""

        path = _flight([0.0, 0.0, 0.0])
        _, plain = reference_body_velocity(path)
        _, armed = reference_body_velocity(path, lever_arm_m=[0.3, 2.0, -0.1])
        assert armed == pytest.approx(plain, abs=1e-5)

    def test_the_correction_scales_with_turn_rate(self):
        """Proportional to omega: this is why it does not average away."""

        arm = [0.0, 2.0, 0.0]
        magnitudes = []
        for degrees in (10.0, 20.0, 40.0):
            path = _flight([np.deg2rad(degrees), 0.0, 0.0])
            _, plain = reference_body_velocity(path)
            _, armed = reference_body_velocity(path, lever_arm_m=arm)
            magnitudes.append(float(np.linalg.norm(armed[10] - plain[10])))
        assert magnitudes[1] == pytest.approx(2 * magnitudes[0], rel=1e-3)
        assert magnitudes[2] == pytest.approx(4 * magnitudes[0], rel=1e-3)

    def test_the_sign_follows_the_arm(self):
        """Mirroring the mount mirrors the correction."""

        path = _flight([np.deg2rad(30.0), 0.0, 0.0])
        _, left = reference_body_velocity(path, lever_arm_m=[0.0, 2.0, 0.0])
        _, right = reference_body_velocity(path, lever_arm_m=[0.0, -2.0, 0.0])
        _, plain = reference_body_velocity(path)
        assert (left[10] - plain[10]) == pytest.approx(-(right[10] - plain[10]), abs=1e-5)

    @pytest.mark.parametrize("bad", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [0.0, np.nan, 0.0]])
    def test_a_malformed_arm_is_refused(self, bad):
        path = _flight([0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            reference_body_velocity(path, lever_arm_m=bad)


class TestResolvers:
    def test_flag_overrides_the_calibration(self, tmp_path):
        import argparse
        import json

        from tools.train_fixedwing_vo import resolve_lever_arm

        manifest = tmp_path / "cam.json"
        manifest.write_text(
            json.dumps({"mounting": {"gps_to_camera_m": [9.0, 9.0, 9.0]}}),
            encoding="utf-8",
        )
        args = argparse.Namespace(lever_arm=[0.0, 2.0, 0.0], calibration=manifest)
        assert resolve_lever_arm(args) == [0.0, 2.0, 0.0]

    def test_falls_back_to_the_calibration(self, tmp_path):
        import argparse
        import json

        from tools.train_fixedwing_vo import resolve_lever_arm

        manifest = tmp_path / "cam.json"
        manifest.write_text(
            json.dumps({"mounting": {"gps_to_camera_m": [0.1, 2.0, -0.3]}}),
            encoding="utf-8",
        )
        args = argparse.Namespace(lever_arm=None, calibration=manifest)
        assert resolve_lever_arm(args) == [0.1, 2.0, -0.3]

    def test_an_all_zero_arm_normalises_to_none(self, tmp_path):
        """Otherwise two identical runs disagree on resume over a no-op."""

        import argparse
        import json

        from tools.train_fixedwing_vo import resolve_lever_arm

        manifest = tmp_path / "cam.json"
        manifest.write_text(
            json.dumps({"mounting": {"gps_to_camera_m": [0.0, 0.0, 0.0]}}),
            encoding="utf-8",
        )
        args = argparse.Namespace(lever_arm=None, calibration=manifest)
        assert resolve_lever_arm(args) is None

    def test_absent_mounting_block_is_not_an_error(self, tmp_path):
        import argparse
        import json

        from tools.train_fixedwing_vo import resolve_lever_arm

        manifest = tmp_path / "cam.json"
        manifest.write_text(json.dumps({"camera": {"fx": 1.0}}), encoding="utf-8")
        args = argparse.Namespace(lever_arm=None, calibration=manifest)
        assert resolve_lever_arm(args) is None
