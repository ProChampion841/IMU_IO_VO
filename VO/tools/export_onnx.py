#!/usr/bin/env python3
"""Export a train_fixedwing_vo.py checkpoint to ONNX, and check the export.

Two graphs are written, because the model runs at two different rates:

* ``frontend.onnx`` - once per image PAIR (every ``pair_stride`` frames). It
  takes the two frames at the working size and the pair's attitude geometry,
  and returns the visual token, its quality, whether the pair is delivered,
  and (planar frontend) the per-pair metric velocity.
* ``temporal_step.onnx`` - once per telemetry TICK (~100 Hz). One step of the
  recurrent fusion: the aiding vector, the most recent visual inputs and the
  carried state in, the velocity and the next state out.

``vo_onnx.json`` next to them records everything a runtime has to reproduce
(input/output names and shapes, the working image size and intrinsics, the
aiding normaliser, the latency, the state tensors and how a tick is built).

Every export is verified: the ONNX graphs are run with onnxruntime on the same
inputs as the PyTorch model, and a stream of ticks through the exported step
is compared against the PyTorch streamed forward. The script fails if any
output differs by more than ``--tolerance``.

    python tools/export_onnx.py runs/vo_planar_500ms/best.pt --output-dir export/onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from vio.data.calibration import maybe_load_camera_calibration  # noqa: E402
from vio.data.image_pairs import resize_camera_matrix  # noqa: E402
from vio.models.frontend_factory import build_frontend, resolve_velocity_mode  # noqa: E402
from vio.models.planar_geometry import axis_angle_to_matrix, onnx_safe_linalg  # noqa: E402
from vio.models.vision_mamba_vo import (  # noqa: E402
    AIDING_INPUT_DIM,
    VisionMambaVO,
    VOStreamState,
)
from vio.utils.checkpoint_io import load_checkpoint  # noqa: E402

FRONTEND_FILE = "frontend.onnx"
TEMPORAL_FILE = "temporal_step.onnx"
METADATA_FILE = "vo_onnx.json"
DATASET_KEYS = (
    "csv_name", "time_column", "time_scale", "attitude_columns", "altitude_column",
    "allow_reference_attitude", "image_folder", "image_size", "frame_gap", "pair_stride",
    "deployment_latency_s", "max_frame_gap_s", "image_time_offset_s", "lever_arm_m",
    "calibration", "color", "warmup", "output_on_pairs",
)


# ---------------------------------------------------------------------------
# export-only operator replacements
# ---------------------------------------------------------------------------


def _pool_matrix(size: int, out: int, like: torch.Tensor) -> torch.Tensor:
    """``(out, size)`` averaging matrix with adaptive pooling's exact bins."""

    matrix = torch.zeros(out, size, dtype=like.dtype, device=like.device)
    for index in range(out):
        start = (index * size) // out
        stop = -((-(index + 1) * size) // out)
        matrix[index, start:stop] = 1.0 / (stop - start)
    return matrix


def adaptive_avg_pool2d_matmul(input: torch.Tensor, output_size) -> torch.Tensor:
    """``F.adaptive_avg_pool2d`` as two matmuls - identical bins, ONNX-exportable.

    The tracer loses the spatial size the ONNX symbolic needs; the image size
    is fixed at export, so the bins are known here and become constants.
    """

    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    height, width = int(input.shape[-2]), int(input.shape[-1])
    out_h = height if output_size[0] is None else int(output_size[0])
    out_w = width if output_size[1] is None else int(output_size[1])
    rows = _pool_matrix(height, out_h, input)
    cols = _pool_matrix(width, out_w, input)
    return rows @ input @ cols.transpose(0, 1)


class export_patches:
    """Closed-form 3x3 algebra and matmul adaptive pooling, for tracing only."""

    def __enter__(self):
        self._linalg = onnx_safe_linalg()
        self._linalg.__enter__()
        self._pool = torch.nn.functional.adaptive_avg_pool2d
        torch.nn.functional.adaptive_avg_pool2d = adaptive_avg_pool2d_matmul
        return self

    def __exit__(self, *exc):
        torch.nn.functional.adaptive_avg_pool2d = self._pool
        return self._linalg.__exit__(*exc)


# ---------------------------------------------------------------------------
# export wrappers
# ---------------------------------------------------------------------------


class PlanarFrontendExport(nn.Module):
    """The planar frontend with its working-resolution intrinsics baked in."""

    input_names = ("image0", "image1", "pair_dt_s", "relative_rotation", "down_body", "altitude_m")
    output_names = (
        "visual_token",
        "visual_quality",
        "pair_reliable",
        "geometric_velocity",
        "geometric_valid",
        "geometric_log_variance",
    )

    def __init__(self, frontend: nn.Module, camera_matrix: torch.Tensor) -> None:
        super().__init__()
        self.frontend = frontend
        self.register_buffer("camera_matrix", camera_matrix.float().reshape(3, 3).clone())

    def forward(self, image0, image1, pair_dt_s, relative_rotation, down_body, altitude_m):
        out = self.frontend(
            image0,
            image1,
            pair_dt_s=pair_dt_s,
            camera_matrix=self.camera_matrix,
            relative_rotation=relative_rotation,
            down_body=down_body,
            altitude_m=altitude_m,
        )
        return tuple(out[name] for name in self.output_names)


class FlowFrontendExport(nn.Module):
    """The original correlation frontend (``--frontend mamba_correlation``)."""

    input_names = ("image0", "image1", "pair_dt_s", "body_rate_rad_s")
    output_names = ("visual_token", "visual_quality", "pair_reliable")

    def __init__(self, frontend: nn.Module, camera_matrix: Optional[torch.Tensor]) -> None:
        super().__init__()
        self.frontend = frontend
        self.has_camera = camera_matrix is not None
        if camera_matrix is not None:
            self.register_buffer("camera_matrix", camera_matrix.float().reshape(3, 3).clone())

    def forward(self, image0, image1, pair_dt_s, body_rate_rad_s):
        out = self.frontend(
            image0,
            image1,
            pair_dt_s=pair_dt_s,
            body_rate_rad_s=body_rate_rad_s,
            camera_matrix=self.camera_matrix if self.has_camera else None,
        )
        return tuple(out[name] for name in self.output_names)


def state_layout(model: VisionMambaVO) -> List[Tuple[str, Tuple[int, ...]]]:
    """Names and per-lane shapes of every carried state tensor, in order."""

    layout = []
    for encoder_name, encoder in (("aiding", model.aiding_encoder), ("fusion", model.fusion_encoder)):
        for index, block in enumerate(encoder.blocks):
            layout.append((f"state_{encoder_name}_{index}_conv", (block.d_inner, max(block.d_conv - 1, 0))))
            layout.append((f"state_{encoder_name}_{index}_ssm", (block.d_inner, block.d_state)))
    return layout


class TemporalStepExport(nn.Module):
    """One telemetry tick of the fusion, with the state as explicit tensors."""

    output_heads = (
        "predicted_velocity",
        "predicted_speed",
        "velocity_log_variance",
        "direction_log_concentration",
    )

    def __init__(self, model: VisionMambaVO) -> None:
        super().__init__()
        self.model = model
        self.geometric = model.velocity_mode == "geometric_residual"
        self.depths = (len(model.aiding_encoder.blocks), len(model.fusion_encoder.blocks))
        self.state_names = [name for name, _ in state_layout(model)]
        names = [
            "aiding",
            "visual_token",
            "visual_present",
            "visual_age",
            "visual_quality",
            "log_altitude",
        ]
        if self.geometric:
            names += ["visual_velocity", "visual_velocity_valid"]
        self.input_names = tuple(names + self.state_names)
        self.output_names = tuple(
            list(self.output_heads) + ["next_" + name for name in self.state_names]
        )

    def forward(self, *inputs: torch.Tensor):
        named = dict(zip(self.input_names, inputs))
        flat = [named[name] for name in self.state_names]
        pairs = [(flat[2 * i], flat[2 * i + 1]) for i in range(len(flat) // 2)]
        state = VOStreamState(
            aiding=tuple(pairs[: self.depths[0]]), fusion=tuple(pairs[self.depths[0]:])
        )
        outputs, next_state = self.model.forward_stream(
            named["aiding"].unsqueeze(1),
            named["visual_token"].unsqueeze(1),
            named["visual_present"].unsqueeze(1),
            named["visual_age"].unsqueeze(1),
            visual_quality=named["visual_quality"].unsqueeze(1),
            log_altitude=named["log_altitude"].unsqueeze(1),
            state=state,
            visual_velocity=named["visual_velocity"].unsqueeze(1) if self.geometric else None,
            visual_velocity_valid=(
                named["visual_velocity_valid"].unsqueeze(1) if self.geometric else None
            ),
        )
        heads = tuple(outputs[name][:, 0] for name in self.output_heads)
        carried = []
        for conv, ssm in (*next_state.aiding, *next_state.fusion):
            carried += [conv, ssm]
        return heads + tuple(carried)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_models(
    checkpoint_path: Path, calibration: Optional[str] = None
) -> Tuple[nn.Module, VisionMambaVO, Dict[str, object], Optional[torch.Tensor], Dict[str, object]]:
    """Rebuild the frontend and model exactly as the evaluator does."""

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    saved = dict(checkpoint.get("args", {}))
    if not saved:
        raise SystemExit(f"{checkpoint_path} has no 'args' block: not a train_fixedwing_vo.py checkpoint")
    frontend_kind = str(saved.get("frontend", "mamba_correlation") or "mamba_correlation")
    velocity_mode = resolve_velocity_mode(
        {"frontend": frontend_kind, "velocity_mode": saved.get("velocity_mode", "heads")}
    )
    frontend = build_frontend(
        saved,
        camera_from_body=saved.get("camera_from_body"),
        prior_velocity=saved.get("prior_velocity_body"),
        dropout=0.0,
    )
    frontend.load_state_dict(checkpoint["frontend"], strict=True)
    model = VisionMambaVO(
        visual_dim=int(saved.get("visual_dim", 64)),
        aiding_dim=int(saved.get("aiding_dim", 64)),
        fusion_dim=int(saved.get("fusion_dim", 96)),
        dropout=0.0,
        frontend=frontend,
        velocity_mode=velocity_mode,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    image_size = tuple(int(v) for v in saved.get("image_size", (576, 1024)))
    camera = None
    loaded = maybe_load_camera_calibration(calibration if calibration is not None else saved.get("calibration"))
    if loaded is not None:
        camera = torch.from_numpy(
            np.asarray(resize_camera_matrix(loaded.camera_matrix, loaded.native_size, image_size), dtype=np.float32)
        )
    if getattr(frontend, "requires_pair_geometry", False) and camera is None:
        raise SystemExit(
            "the planar frontend needs the camera intrinsics: pass --calibration "
            "(the same JSON the model was trained with)"
        )
    calibration_block = None
    if loaded is not None:
        calibration_block = {
            "native_camera_matrix": np.asarray(loaded.camera_matrix, dtype=np.float64).tolist(),
            "native_size": [int(v) for v in loaded.native_size],
            "distortion": np.asarray(loaded.distortion, dtype=np.float64).reshape(-1).tolist(),
            "images_rectified": bool(loaded.images_rectified),
        }
    saved["_calibration_block"] = calibration_block
    return frontend, model, saved, camera, dict(checkpoint.get("normalizer", {}))


# ---------------------------------------------------------------------------
# example inputs
# ---------------------------------------------------------------------------


def _textured_pair(channels: int, height: int, width: int, shift_px: Tuple[float, float], seed: int):
    """A smooth random texture and a shifted copy - enough for a real fit."""

    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 1, height // 4 + 8, width // 4 + 8, generator=generator)
    big = torch.nn.functional.interpolate(base, scale_factor=4, mode="bilinear", align_corners=False)
    dy, dx = int(round(shift_px[0])), int(round(shift_px[1]))
    first = big[..., 16: 16 + height, 16: 16 + width]
    second = big[..., 16 - dy: 16 - dy + height, 16 - dx: 16 - dx + width]
    tint = torch.linspace(0.9, 1.1, channels).view(1, channels, 1, 1)
    return (first * tint).clamp(0, 1).contiguous(), (second * tint).clamp(0, 1).contiguous()


def frontend_example(export: nn.Module, channels: int, image_size: Tuple[int, int], batch: int, seed: int = 0):
    """Pairs whose shift, interval, rotation and altitude all change with ``seed``.

    Verification uses seeds the trace never saw, so any value frozen into the
    graph by tracing shows up as a mismatch rather than passing by luck.
    """

    generator = torch.Generator().manual_seed(1000 + seed)

    def uniform(low: float, high: float, *shape: int) -> torch.Tensor:
        return low + (high - low) * torch.rand(*shape, generator=generator)

    height, width = image_size
    images0, images1 = [], []
    for lane in range(batch):
        shift = uniform(-10.0, 10.0, 2).tolist()
        a, b = _textured_pair(channels, height, width, (shift[0], shift[1]), seed * 7 + lane)
        images0.append(a)
        images1.append(b)
    image0, image1 = torch.cat(images0), torch.cat(images1)
    dt = uniform(0.4, 1.1, batch)
    if isinstance(export, PlanarFrontendExport):
        rotation = axis_angle_to_matrix(uniform(-0.05, 0.05, batch, 3))
        down = torch.cat((uniform(-0.2, 0.2, batch, 2), torch.ones(batch, 1)), dim=1)
        down = down / down.norm(dim=-1, keepdim=True)
        first = uniform(150.0, 250.0, batch)
        altitude = torch.stack((first, first + uniform(-2.0, 2.0, batch)), dim=1)
        return (image0, image1, dt, rotation, down, altitude)
    return (image0, image1, dt, uniform(-0.2, 0.2, batch, 3))


def temporal_example(step: TemporalStepExport, ticks: int, batch: int, seed: int = 0) -> List[Tuple[torch.Tensor, ...]]:
    """A plausible tick stream: a pair every 50 ticks, held between pairs."""

    generator = torch.Generator().manual_seed(seed)
    model = step.model
    token = torch.zeros(batch, model.visual_dim)
    quality = torch.zeros(batch, 1)
    held = torch.zeros(batch, 3)
    held_valid = torch.zeros(batch, 1)
    last_delivery = None
    stream = []
    for tick in range(ticks):
        roll = 0.2 * np.sin(tick / 40.0)
        aiding = torch.tensor(
            [np.sin(roll), np.cos(roll), 0.05, 0.998, 0.01, 0.01, -0.02, 0.05 * roll, 1.0],
            dtype=torch.float32,
        ).repeat(batch, 1)
        aiding = aiding + 0.01 * torch.randn(batch, AIDING_INPUT_DIM, generator=generator)
        present = torch.zeros(batch, 1)
        if tick % 50 == 35:
            present[:] = 1.0
            last_delivery = tick
            held = torch.tensor([[20.0, 1.0, -0.5]]).repeat(batch, 1) + torch.randn(batch, 3, generator=generator)
            held_valid[:] = 1.0
        age = torch.zeros(batch, 1) if last_delivery is None else torch.full((batch, 1), 0.01 * (tick - last_delivery) + 0.35)
        token_now = torch.randn(batch, model.visual_dim, generator=generator) * present
        quality_now = torch.rand(batch, 1, generator=generator) * present
        log_altitude = torch.full((batch,), float(np.log(200.0)))
        inputs = [aiding, token_now, present, age, quality_now, log_altitude]
        if step.geometric:
            inputs += [held.clone(), held_valid.clone()]
        stream.append(tuple(inputs))
    return stream


# ---------------------------------------------------------------------------
# export + verification
# ---------------------------------------------------------------------------


def _export(
    module: nn.Module,
    args: Tuple[torch.Tensor, ...],
    path: Path,
    input_names,
    output_names,
    opset: int,
    dynamic_batch: bool,
) -> None:
    dynamic_axes = (
        {name: {0: "batch"} for name in list(input_names) + list(output_names)}
        if dynamic_batch
        else None
    )
    with torch.no_grad(), export_patches():
        torch.onnx.export(
            module,
            args,
            str(path),
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )


def _session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _max_diff(reference: torch.Tensor, value: np.ndarray) -> float:
    ref = reference.detach().float().numpy()
    if ref.shape != value.shape:
        raise AssertionError(f"shape mismatch {ref.shape} vs {value.shape}")
    return float(np.max(np.abs(ref - value))) if ref.size else 0.0


def verify_frontend(export: nn.Module, path: Path, example: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
    session = _session(path)
    with torch.no_grad():
        reference = export(*example)
    feeds = {name: tensor.numpy() for name, tensor in zip(export.input_names, example)}
    got = session.run(list(export.output_names), feeds)
    return {name: _max_diff(ref, value) for name, ref, value in zip(export.output_names, reference, got)}


def verify_temporal(step: TemporalStepExport, path: Path, stream: Sequence[Tuple[torch.Tensor, ...]]) -> Dict[str, float]:
    """Stream ticks through ONNX, one at a time, against the PyTorch block forward."""

    model = step.model
    batch = stream[0][0].shape[0]
    stacked = [torch.stack([tick[i] for tick in stream], dim=1) for i in range(len(stream[0]))]
    geometric = step.geometric
    with torch.no_grad():
        reference = model.forward(
            stacked[0],
            stacked[1],
            stacked[2],
            stacked[3],
            visual_quality=stacked[4],
            log_altitude=stacked[5],
            visual_velocity=stacked[6] if geometric else None,
            visual_velocity_valid=stacked[7] if geometric else None,
        )
    session = _session(path)
    state = [
        np.zeros((batch, *shape), dtype=np.float32) for _, shape in state_layout(model)
    ]
    heads = {name: [] for name in step.output_heads}
    for tick in stream:
        feeds = {name: tensor.numpy() for name, tensor in zip(step.input_names, tick)}
        feeds.update(dict(zip(step.state_names, state)))
        got = session.run(list(step.output_names), feeds)
        for index, name in enumerate(step.output_heads):
            heads[name].append(got[index])
        state = got[len(step.output_heads):]
    return {
        name: _max_diff(reference[name], np.stack(values, axis=1)) for name, values in heads.items()
    }


def export_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    *,
    calibration: Optional[str] = None,
    opset: int = 17,
    verify_ticks: int = 200,
    tolerance: float = 1e-3,
    frontend_batch: int = 1,
) -> Dict[str, object]:
    frontend, model, saved, camera, normalizer = load_models(checkpoint, calibration)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = tuple(int(v) for v in saved.get("image_size", (576, 1024)))
    channels = 3 if bool(saved.get("color", False)) else 1
    planar = bool(getattr(frontend, "requires_pair_geometry", False))
    frontend.eval()

    front = PlanarFrontendExport(frontend, camera) if planar else FlowFrontendExport(frontend, camera)
    front.eval()
    front_path = output_dir / FRONTEND_FILE
    print(f"exporting frontend -> {front_path}")
    # Static shapes: the frontend pools to fixed grids, which ONNX can only
    # express for a known input size. One pair per call is what a live
    # system runs; --frontend-batch exports a bigger fixed batch.
    _export(
        front, frontend_example(front, channels, image_size, frontend_batch), front_path,
        front.input_names, front.output_names, opset, dynamic_batch=False,
    )

    step = TemporalStepExport(model).eval()
    step_path = output_dir / TEMPORAL_FILE
    print(f"exporting temporal step -> {step_path}")
    one_tick = temporal_example(step, 1, 1)[0]
    state0 = tuple(torch.zeros(1, *shape) for _, shape in state_layout(model))
    _export(step, one_tick + state0, step_path, step.input_names, step.output_names, opset, dynamic_batch=True)

    import onnx

    for path in (front_path, step_path):
        onnx.checker.check_model(str(path))

    # --- verification: fresh inputs, not the ones traced with -------------
    report: Dict[str, Dict[str, float]] = {}
    for trial in range(3):
        report[f"frontend_trial{trial}"] = verify_frontend(
            front, front_path, frontend_example(front, channels, image_size, frontend_batch, seed=10 + trial)
        )
    for batch in (1, 2):
        report[f"temporal_batch{batch}"] = verify_temporal(
            step, step_path, temporal_example(step, verify_ticks, batch, seed=20 + batch)
        )
    worst = max(value for block in report.values() for value in block.values())
    for label, block in report.items():
        print(f"  {label:18s} " + "  ".join(f"{k}={v:.2e}" for k, v in block.items()))
    print(f"  max |onnx - torch| = {worst:.2e} (tolerance {tolerance:g})")

    metadata = build_metadata(
        front, step, saved, camera, normalizer, image_size, channels, opset, report, frontend_batch
    )
    (output_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"metadata -> {output_dir / METADATA_FILE}")
    if not np.isfinite(worst) or worst > tolerance:
        raise SystemExit(f"EXPORT CHECK FAILED: max difference {worst:.3e} > {tolerance:g}")
    print("ONNX EXPORT OK")
    return metadata


def build_metadata(
    front, step, saved, camera, normalizer, image_size, channels, opset, report, frontend_batch
) -> Dict[str, object]:
    model = step.model
    frame_gap = int(saved.get("frame_gap", 1))
    pair_stride = int(saved.get("pair_stride", 1))
    return {
        "opset": opset,
        "files": {"frontend": FRONTEND_FILE, "temporal_step": TEMPORAL_FILE},
        "frontend_id": saved.get("frontend_id"),
        "temporal_input_id": saved.get("temporal_input_id"),
        "velocity_mode": model.velocity_mode,
        "frontend": {
            "inputs": {
                name: shape for name, shape in zip(
                    front.input_names,
                    (
                        [frontend_batch, channels, image_size[0], image_size[1]],
                        [frontend_batch, channels, image_size[0], image_size[1]],
                        [frontend_batch],
                        *(
                            ([frontend_batch, 3, 3], [frontend_batch, 3], [frontend_batch, 2])
                            if isinstance(front, PlanarFrontendExport)
                            else ([frontend_batch, 3],)
                        ),
                    ),
                )
            },
            "outputs": list(front.output_names),
            "image_preprocessing": (
                f"PIL open -> convert('{'RGB' if channels == 3 else 'L'}') -> "
                f"resize((width={image_size[1]}, height={image_size[0]}), BILINEAR) "
                "-> float32 / 255 -> (C, H, W). Undistort first if the calibration has distortion."
            ),
            "camera_matrix_working": None if camera is None else camera.tolist(),
            "pair_geometry": (
                "relative_rotation = R0^T R1 (body orientation of exposure 1 in body frame of "
                "exposure 0, attitude interpolated at each exposure time); down_body = R0^T [0,0,1] "
                "(NED down in body frame 0); altitude_m = [RelativeAlt at exposure 0, at exposure 1]; "
                "pair_dt_s = t1 - t0. See vio.data.attitude.pair_geometry_batch."
            ),
        },
        "temporal_step": {
            "inputs": list(step.input_names),
            "outputs": list(step.output_names),
            "state": [{"name": name, "shape": ["batch", *shape]} for name, shape in state_layout(model)],
            "state_reset": "all zeros (cold start)",
            "aiding_channels": [
                "sin(roll)", "cos(roll)", "sin(pitch)", "cos(pitch)",
                "log(max(RelativeAlt, 1)) - log_altitude_mean",
                "p (rad/s)", "q (rad/s)", "r (rad/s)",
                "(t - t_prev) / delta_time_scale",
            ],
            "log_altitude": "RAW log(max(RelativeAlt, 1)) - not centred",
            "normalizer": normalizer,
            "per_tick_rules": [
                "visual_present = 1 only on the tick a pair's result becomes available AND pair_reliable > 0; else 0",
                "visual_token / visual_quality = the frontend outputs on that tick, zeros on every other tick",
                "visual_age = 0 before the first delivered pair, else (t - t_last_delivery) + deployment_latency_s",
                "visual_velocity = last delivered geometric_velocity (held), visual_velocity_valid = 1 after the first delivery",
                "feed next_state_* back as state_* on the next tick",
            ],
        },
        "timing": {
            "frame_gap": frame_gap,
            "pair_stride": pair_stride,
            "output_on_pairs": bool(saved.get("output_on_pairs", False)),
            "deployment_latency_s": float(saved.get("deployment_latency_s", 0.35)),
            "image_time_offset_s": saved.get("image_time_offset_s", 0.0),
            "warmup_ticks": saved.get("warmup"),
            "note": (
                f"pair = (frame k, frame k + {frame_gap}); a new pair every {pair_stride} frames. "
                "With output_on_pairs, report predicted_velocity only on ticks where visual_present = 1 and hold it between."
            ),
        },
        "outputs": {
            "predicted_velocity": "m/s, body frame (x forward, y right, z down)",
            "velocity_log_variance": "per-axis log variance (only meaningful if trained with the NLL loss)",
        },
        # What tools/onnx_inference.py needs to replay the same inputs from
        # a flight folder, without the checkpoint.
        "dataset_settings": {key: saved.get(key) for key in DATASET_KEYS},
        "calibration": saved.get("_calibration_block"),
        "image_time_scale": 0.001,
        "image_pattern": "*.jpg",
        "verification_max_abs_diff": report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("export/onnx"))
    parser.add_argument("--calibration", default=None, help="Override the calibration JSON recorded in the checkpoint")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify-ticks", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--frontend-batch", type=int, default=1, help="Fixed number of pairs per frontend call")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    export_checkpoint(
        args.checkpoint,
        args.output_dir,
        calibration=args.calibration,
        opset=args.opset,
        verify_ticks=args.verify_ticks,
        tolerance=args.tolerance,
        frontend_batch=args.frontend_batch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
