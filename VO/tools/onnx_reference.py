"""PyTorch reference for checking tools/onnx_inference.py (needs torch + the project).

Runs the checkpoint over a flight folder exactly the way the horizon evaluator
does - same dataset build, same pair plan, same token scatter, same held
velocity and visual age - and returns the per-tick velocity and which ticks
received a pair, for comparison against the standalone ONNX runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import torch  # noqa: E402

from vio.data.calibration import maybe_load_camera_calibration  # noqa: E402
from vio.data.fixedwing_vo import VONormalizer, build_vo_dataset, visual_age_seconds  # noqa: E402
from vio.models.velocity_horizons import (  # noqa: E402
    encode_span_tokens,
    scatter_span_tokens,
    scatter_span_velocity,
)


def torch_reference(checkpoint: Path, dataset_root: Path, total: int) -> Tuple[np.ndarray, np.ndarray]:
    from tools.export_onnx import load_models

    frontend, model, saved, camera, normalizer = load_models(checkpoint)
    loaded = maybe_load_camera_calibration(saved.get("calibration"))
    dataset, _, attitude = build_vo_dataset(
        dataset_root,
        (0, 2),
        csv_name=saved.get("csv_name", "flight.csv"),
        time_column=saved.get("time_column", "Time"),
        time_scale=float(saved.get("time_scale", 1.0)),
        attitude_columns=saved.get("attitude_columns"),
        altitude_column=saved.get("altitude_column"),
        allow_reference_attitude=bool(saved.get("allow_reference_attitude", False)),
        image_folder=saved.get("image_folder", "images"),
        image_size=tuple(int(v) for v in saved.get("image_size", (576, 1024))),
        frame_gap=int(saved.get("frame_gap", 1)),
        pair_stride=int(saved.get("pair_stride", 1)),
        deployment_latency_s=float(saved.get("deployment_latency_s", 0.35)),
        max_frame_gap_s=saved.get("max_frame_gap_s"),
        image_time_offset_s=saved.get("image_time_offset_s", 0.0),
        lever_arm_m=saved.get("lever_arm_m"),
        camera_matrix=None if loaded is None else loaded.camera_matrix,
        calibration_image_size=None if loaded is None else loaded.native_size,
        images_rectified=False if loaded is None else loaded.images_rectified,
        distortion=None if loaded is None or not loaded.distortion.size else loaded.distortion,
        normalizer=VONormalizer(**normalizer),
        window_length=2,
        stride=1,
        warmup=0,
        max_visual_events=1,
        grayscale=not bool(saved.get("color", False)),
    )
    device = torch.device("cpu")
    tokens = encode_span_tokens(
        frontend, dataset.image_source, span=(0, total), body_rate_rad_s=attitude.body_rate_rad_s,
        times_s=attitude.times_s, visual_dim=model.visual_dim, device=device, camera_matrix=camera,
        attitude=attitude,
    )
    token, quality, present = scatter_span_tokens(tokens, [(0, total)], total, device=device)
    clock = torch.from_numpy(attitude.times_s[:total].astype(np.float64)).unsqueeze(0)
    age, _ = visual_age_seconds(present, clock, float(saved.get("deployment_latency_s", 0.35)))
    extra = {}
    if model.velocity_mode == "geometric_residual":
        held, held_valid = scatter_span_velocity(tokens, [(0, total)], total, device=device)
        extra = {"visual_velocity": held, "visual_velocity_valid": held_valid}
    with torch.no_grad():
        out = model.forward(
            torch.from_numpy(dataset.aiding[:total]).unsqueeze(0), token, present, age,
            visual_quality=quality,
            log_altitude=torch.from_numpy(dataset.log_altitude[:total]).unsqueeze(0),
            **extra,
        )
    return out["predicted_velocity"][0].numpy(), present[0, :, 0].numpy() > 0


__all__ = ["torch_reference"]
