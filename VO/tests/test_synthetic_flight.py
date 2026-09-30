"""End-to-end consistency tests for the synthetic fixed-wing flight.

The generator is only useful as a pipeline test if its telemetry, its images
and the loader's derived reference all describe the same motion. These tests
check that chain directly, so a future change to a frame convention, a unit,
or an integration rule fails here rather than silently degrading training.

The chain checked is the one the trainer actually reads: attitude and altitude
from vio.data.attitude, and the body-velocity target from vio.data.fixedwing_vo.
The generator still simulates an IMU, but nothing in this project consumes it,
so it is not asserted on here.
"""

from __future__ import annotations

import numpy as np
import pytest

from vio.data.attitude import load_attitude_altitude
from vio.data.calibration import load_camera_calibration
from vio.data.fixedwing_vo import reference_body_velocity
from vio.models.pose_geometry import (
    quaternion_conjugate_np,
    quaternion_multiply_np,
    quaternion_to_matrix_np,
    quaternion_to_rotvec_np,
)
from tools.make_synthetic_flight import GRAVITY_M_S2, main as generate_flight

DURATION_S = 8.0


@pytest.fixture(scope="module")
def flight(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_flight")
    assert generate_flight(
        ["--output", str(root), "--duration-s", str(DURATION_S), "--no-images"]
    ) == 0
    csv_path = root / "flight.csv"
    times, body_velocity = reference_body_velocity(csv_path)
    return {
        "root": root,
        "csv": csv_path,
        "truth": np.load(root / "truth.npz"),
        "times_s": times,
        "body_velocity_m_s": body_velocity,
        "attitude": load_attitude_altitude(csv_path),
    }


def test_loader_reference_reproduces_the_generated_trajectory(flight):
    """The trainer's inputs and target must describe the generated motion.

    This is the chain that matters: if the target disagrees with the truth the
    model is trained on a different flight than the one it sees, and if the
    altitude column disagrees the v = h*u scale relation is wrong by that
    factor with nothing to signal it.
    """

    truth, attitude = flight["truth"], flight["attitude"]

    # The target: NED truth rotated into the body frame must equal the label.
    rotation = quaternion_to_matrix_np(truth["quaternion_body_to_ned"])
    expected_body = np.einsum(
        "nij,ni->nj", rotation, truth["velocity_ned_m_s"].astype(np.float64)
    )
    velocity_error = np.linalg.norm(
        flight["body_velocity_m_s"] - expected_body, axis=1
    )
    assert velocity_error.max() < 1e-3

    # The model's attitude input is NavEul*, which carries a deliberate slow
    # offset from the GPSNavEul* that defines the target. The offset must be
    # present - identical columns would be target leakage, not a clean signal -
    # and must not exceed the configured amount.
    offset_rad = np.abs(attitude.euler_rad - truth["euler_rad"])
    assert offset_rad.max() <= np.deg2rad(0.2) + 1e-9
    assert offset_rad.max() > np.deg2rad(0.05), "NavEul* must not equal the target attitude"

    # The altitude input must be height above the takeoff datum, which in NED
    # is minus the down coordinate. Barometer carries a field elevation on top
    # of this, so resolving the wrong column fails here by that offset.
    assert attitude.altitude_column == "RelativeAlt"
    expected_altitude = -truth["position_ned_m"][:, 2]
    assert np.abs(attitude.altitude_m - expected_altitude).max() < 1e-3


def test_motion_is_dynamic_enough_to_be_a_real_test(flight):
    """Guard against a degenerate flight that any constant predictor solves."""

    truth = flight["truth"]
    speed = np.linalg.norm(truth["velocity_ned_m_s"], axis=1)
    bank_deg = np.rad2deg(truth["euler_rad"][:, 0])
    heading_deg = np.rad2deg(np.unwrap(truth["euler_rad"][:, 2]))
    assert speed.min() > 5.0
    assert speed.max() - speed.min() > 1.0
    assert np.abs(bank_deg).max() > 5.0
    assert heading_deg.max() - heading_deg.min() > 10.0


def test_rendered_images_encode_the_true_camera_motion(tmp_path):
    """The analytic homography from truth must align consecutive frames.

    If the renderer and the telemetry disagreed about pose, no homography
    built from the telemetry could align the frames, and the visual tokens
    would carry motion evidence unrelated to the labels.
    """

    cv2 = pytest.importorskip("cv2")
    root = tmp_path / "rendered"
    assert generate_flight(["--output", str(root), "--duration-s", "3.0"]) == 0

    truth = np.load(root / "truth.npz")
    calibration = load_camera_calibration(root / "calibration.json")
    camera_matrix = calibration.camera_matrix
    frames = sorted((root / "images").glob("*.jpg"), key=lambda p: int(p.stem))
    assert len(frames) >= 10

    position = truth["position_ned_m"]
    quaternion = truth["quaternion_body_to_ned"].astype(np.float64)
    indices = truth["frame_indices"]

    def projection(sample: int) -> np.ndarray:
        rotation = quaternion_to_matrix_np(quaternion[sample])
        ground_to_camera = np.asarray(
            [
                [1.0, 0.0, -position[sample, 0]],
                [0.0, 1.0, -position[sample, 1]],
                [0.0, 0.0, -position[sample, 2]],
            ]
        )
        return camera_matrix @ rotation.T @ ground_to_camera

    aligned, raw = [], []
    for frame in range(2, min(12, len(frames) - 1)):
        first, second = int(indices[frame]), int(indices[frame + 1])
        homography = projection(second) @ np.linalg.inv(projection(first))
        image_a = cv2.imread(str(frames[frame]), cv2.IMREAD_GRAYSCALE)
        image_b = cv2.imread(str(frames[frame + 1]), cv2.IMREAD_GRAYSCALE)
        warped = cv2.warpPerspective(image_a, homography, (image_b.shape[1], image_b.shape[0]))
        valid = (warped > 0) & (image_b > 0)
        valid[:24, :] = valid[-24:, :] = valid[:, :24] = valid[:, -24:] = False
        aligned.append(
            np.abs(warped[valid].astype(float) - image_b[valid].astype(float)).mean()
        )
        raw.append(np.abs(image_a.astype(float) - image_b.astype(float)).mean())

    # Motion compensation from truth must explain most of the frame difference.
    assert np.mean(aligned) < 4.0
    assert np.mean(aligned) < 0.35 * np.mean(raw)


def test_generated_images_are_trackable(tmp_path):
    """A texture with no repeatable corners would make matching meaningless."""

    cv2 = pytest.importorskip("cv2")
    root = tmp_path / "texture"
    assert generate_flight(["--output", str(root), "--duration-s", "2.0"]) == 0
    frame = cv2.imread(
        str(sorted((root / "images").glob("*.jpg"))[0]), cv2.IMREAD_GRAYSCALE
    )
    response = cv2.cornerHarris(np.float32(frame), 2, 3, 0.04)
    assert int((response > 0.01 * response.max()).sum()) > 100
    assert frame.std() > 20.0


def test_rendering_survives_coordinates_past_the_remap_fixed_point_limit(tmp_path):
    """Ground coordinates far from the origin must not alias.

    OpenCV's remap converts coordinates to a fixed-point form whose integer
    part is a 16-bit short, so past 32767 texels the maps wrap incorrectly.
    The frames stay image-shaped and non-blank, but progressively alias, which
    washes out local contrast and destroys the repeatable corners a matcher
    needs. At the default sampling that begins after about 3.9 km of ground
    track; a finer sampling reaches it within seconds so the regression is
    reproducible in a test.
    """

    cv2 = pytest.importorskip("cv2")
    root = tmp_path / "far"
    assert (
        generate_flight(
            [
                "--output",
                str(root),
                "--duration-s",
                "12.0",
                # 32767 * 0.005 m is about 164 m of travel, some 7 seconds in.
                "--ground-metres-per-texel",
                "0.005",
            ]
        )
        == 0
    )

    frames = sorted((root / "images").glob("*.jpg"), key=lambda p: int(p.stem))
    assert len(frames) >= 100

    def contrast(path) -> float:
        return float(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).std())

    early = contrast(frames[2])
    late = contrast(frames[-2])
    assert early > 20.0
    # Aliasing showed as a steady loss of contrast with distance travelled.
    assert late > 0.9 * early


def test_a_rerun_clears_frames_from_the_previous_capture(tmp_path):
    """Two flights must never share an image directory.

    A re-run only overwrites the filenames it reuses. Writing a new capture at
    a different image rate over an old one leaves both, interleaved by
    timestamp, and every frame is a valid JPEG at a plausible time - so nothing
    downstream can tell that half of them show a trajectory the CSV does not
    describe. This happened, and the only defence is clearing before rendering.
    """

    root = tmp_path / "capture"
    assert generate_flight(
        ["--output", str(root), "--duration-s", "2.0", "--image-rate-hz", "5"]
    ) == 0
    first = sorted(p.name for p in (root / "images").glob("*.jpg"))
    assert first

    # A stale frame at a timestamp the next run will not reuse.
    stale = root / "images" / "9999999.jpg"
    stale.write_bytes((root / "images" / first[0]).read_bytes())

    assert generate_flight(
        ["--output", str(root), "--duration-s", "2.0", "--image-rate-hz", "10"]
    ) == 0
    remaining = {p.name for p in (root / "images").glob("*.jpg")}
    assert stale.name not in remaining, "a frame from the previous capture survived"

    # And what is left is one capture at one rate.
    times = np.array(sorted(int(name[:-4]) for name in remaining)) / 1000.0
    gaps = np.unique(np.round(np.diff(times), 4))
    assert gaps.size == 1, f"mixed frame rates in one directory: {gaps}"
