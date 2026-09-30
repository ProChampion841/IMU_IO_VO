"""The camera block of a calibration file, and nothing else.

This exists because the trainer and the evaluator each parsed the same JSON
inline, and because the loader they could have shared - ``load_vio_calibration``
in the retired pose pipeline - refused to read a file that had no ``imu`` block
and no gyro/accel units. This project has no rate or acceleration sensor, so
demanding those fields to read a focal length was backwards.

Only the camera is read here. A calibration file may carry other blocks; they
are ignored rather than validated, so a file written for another pipeline still
works and a missing IMU section is not an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """Intrinsics as the frontend wants them."""

    #: (3, 3) pinhole matrix at the NATIVE image size, float32 to match the
    #: tensors it is compared against downstream.
    camera_matrix: np.ndarray
    #: (rows, columns) of the image the intrinsics were measured on. The
    #: frontend rescales by working/native, so this must be the calibration
    #: size, never the size the images were resized to.
    native_size: Tuple[int, int]
    #: Lens distortion, empty when the file records none.
    distortion: np.ndarray
    #: Whether the images on disk have already been undistorted.
    images_rectified: bool

    @property
    def focal_lengths(self) -> Tuple[float, float]:
        return float(self.camera_matrix[0, 0]), float(self.camera_matrix[1, 1])


def load_camera_calibration(path: str | Path) -> CameraCalibration:
    """Read the ``camera`` block of a calibration JSON file.

    Raises rather than defaulting on anything whose absence would silently
    change the geometry: a missing focal length is not a focal length of zero,
    and a camera size of one pixel is a typo, not a camera.
    """

    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    try:
        camera = payload["camera"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} has no 'camera' block") from error

    missing = [key for key in ("fx", "fy", "cx", "cy", "width", "height")
               if key not in camera]
    if missing:
        raise ValueError(f"{path} camera block is missing {', '.join(missing)}")

    matrix = np.array(
        [
            [camera["fx"], 0.0, camera["cx"]],
            [0.0, camera["fy"], camera["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{path} camera intrinsics are not finite")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError(f"{path} camera focal lengths must be positive")

    width, height = int(camera["width"]), int(camera["height"])
    if min(width, height) <= 1:
        raise ValueError(f"{path} camera size {width}x{height} is not an image")

    distortion = np.asarray(camera.get("distortion", []), dtype=np.float64)
    if distortion.ndim != 1 or not np.all(np.isfinite(distortion)):
        raise ValueError(f"{path} camera distortion must be a finite vector")

    return CameraCalibration(
        camera_matrix=matrix,
        native_size=(height, width),
        distortion=distortion,
        images_rectified=bool(camera.get("images_rectified", False)),
    )


def maybe_load_camera_calibration(
    path: Optional[str | Path],
) -> Optional[CameraCalibration]:
    """:func:`load_camera_calibration`, or ``None`` when no path is given.

    ``--calibration`` is optional and the frontend falls back to normalising by
    image size, so "no file" is a supported configuration - but a file that was
    given and cannot be read is an error, not a fallback.
    """

    return None if not path else load_camera_calibration(path)


__all__ = [
    "CameraCalibration",
    "load_camera_calibration",
    "maybe_load_camera_calibration",
]
