from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from vio.data.image_pairs import VisualPairSource, resize_camera_matrix
from vio.models.vision_mamba_vo import VisionMambaFlowFrontend


def _images(root, size=(100, 60)):
    folder = root / "images"
    folder.mkdir()
    for timestamp in (0, 50, 100):
        pixels = np.arange(size[0] * size[1], dtype=np.uint8).reshape(
            size[1], size[0]
        )
        Image.fromarray(np.roll(pixels, timestamp // 50, axis=1)).save(
            folder / f"{timestamp}.jpg"
        )


def test_resize_camera_matrix_matches_half_pixel_sampling():
    matrix = np.array(
        [[500.0, 0.0, 49.5], [0.0, 480.0, 29.5], [0.0, 0.0, 1.0]]
    )
    resized = resize_camera_matrix(matrix, (60, 100), (30, 50))
    assert resized[0, 0] == pytest.approx(250.0)
    assert resized[1, 1] == pytest.approx(240.0)
    assert resized[0, 2] == pytest.approx(24.5)
    assert resized[1, 2] == pytest.approx(14.5)


def test_loader_exposes_working_intrinsics_and_frontend_uses_them(tmp_path):
    _images(tmp_path)
    matrix = np.array(
        [[500.0, 0.0, 49.5], [0.0, 480.0, 29.5], [0.0, 0.0, 1.0]]
    )
    source = VisualPairSource(
        tmp_path,
        np.linspace(0.0, 0.2, 21),
        deployment_latency_s=0.0,
        image_size=(30, 50),
        camera_matrix=matrix,
        distortion=np.zeros(5),
        calibration_image_size=(60, 100),
    )
    assert source._load(0).shape == (1, 30, 50)
    # The SHIPPING frontend, not a retired one: this asserts the intrinsics
    # the loader exposes are the intrinsics the trained model consumes.
    frontend = VisionMambaFlowFrontend(
        image_size=(30, 50), patch_size=10, token_grid=2, d_model=16, depth=1
    )
    # source.camera_matrix is ALREADY rescaled to the working size, so the
    # native size passed here is the working size. Passing the calibration
    # size would apply the 100->50 ratio twice and halve the focal length.
    scale = frontend._bearing_scale(
        torch.from_numpy(source.camera_matrix),
        (30, 50),
        1,
        torch.device("cpu"),
        torch.float32,
    )
    assert scale[0, 0] == pytest.approx(10.0 / 250.0)
    assert scale[0, 1] == pytest.approx(10.0 / 240.0)


def test_nonzero_distortion_is_rectified_and_native_size_is_checked(tmp_path):
    pytest.importorskip("cv2")
    _images(tmp_path)
    matrix = np.array(
        [[500.0, 0.0, 49.5], [0.0, 480.0, 29.5], [0.0, 0.0, 1.0]]
    )
    source = VisualPairSource(
        tmp_path,
        np.linspace(0.0, 0.2, 21),
        deployment_latency_s=0.0,
        image_size=(30, 50),
        camera_matrix=matrix,
        distortion=np.array([-0.2, 0.03, 0.0, 0.0, 0.0]),
        calibration_image_size=(60, 100),
        images_rectified=False,
    )
    assert source._rectify_maps is not None
    assert source._load(0).shape == (1, 30, 50)

    wrong = VisualPairSource(
        tmp_path,
        np.linspace(0.0, 0.2, 21),
        deployment_latency_s=0.0,
        image_size=(30, 50),
        camera_matrix=matrix,
        distortion=np.zeros(5),
        calibration_image_size=(61, 100),
    )
    with pytest.raises(ValueError, match="calibration expects"):
        wrong._load(0)
