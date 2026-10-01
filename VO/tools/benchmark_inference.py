#!/usr/bin/env python3
"""Model inputs/outputs, size, and inference time against the real-time budget.

The deployed model is two graphs running at two rates (see tools/export_onnx.py):

* the FRONTEND, once per image pair - two frames in, a visual token (and, for
  ``--frontend planar``, a metric velocity) out;
* the TEMPORAL STEP, once per telemetry tick - the aiding vector, the latest
  visual inputs and the carried recurrent state in, the velocity and the next
  state out.

This script prints the I/O contract of both, counts parameters and FLOPs per
component, and times them in PyTorch (eager, no_grad) and in onnxruntime
(exported here exactly as tools/export_onnx.py does it). It also times the
deployable runtime's per-frame image preprocessing (JPEG decode, grayscale or
RGB, lens undistortion + resize) on a synthetic frame at the calibration's
native size. Those numbers are then set against the rates the model was
trained for:

* frontend duty  = pairs/s x (2 x preprocess + frontend), because
  tools/onnx_inference.py preprocesses BOTH frames of every pair;
* tick duty      = telemetry Hz x temporal step;
* latency margin = --deployment-latency-s minus one pair's processing time.
  The model was trained assuming a pair's token arrives exactly that long
  after its second exposure, so a pair that takes longer arrives later than
  the model believes.

With a checkpoint the model and timing come from it::

    python tools/benchmark_inference.py --checkpoint runs/vo_planar_s0/best.pt

Without one, untrained models are built from the trainer's default settings -
timing does not depend on the weights::

    python tools/benchmark_inference.py --frontend both --threads 1 4

The numbers are for THIS machine. On the Jetson, rerun this script there
(``--providers CUDAExecutionProvider CPUExecutionProvider`` for the GPU).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from tools.export_onnx import (  # noqa: E402
    FlowFrontendExport,
    PlanarFrontendExport,
    TemporalStepExport,
    _export,
    frontend_example,
    load_models,
    state_layout,
    temporal_example,
    verify_frontend,
)
from vio.data.calibration import maybe_load_camera_calibration  # noqa: E402
from vio.data.image_pairs import resize_camera_matrix  # noqa: E402
from vio.models.frontend_factory import (  # noqa: E402
    build_frontend,
    resolve_velocity_mode,
)
from vio.models.planar_geometry import NADIR_MOUNTINGS  # noqa: E402
from vio.models.vision_mamba_vo import VisionMambaVO  # noqa: E402

DEFAULT_CALIBRATION = _ROOT / "configs" / "vo" / "camera_fixedwing.json"


# ---------------------------------------------------------------------------
# building the models
# ---------------------------------------------------------------------------


def default_settings(kind: str, image_size: Tuple[int, int], color: bool) -> Dict[str, object]:
    """The trainer's defaults for ``--frontend kind`` (train_fixedwing_vo.py)."""

    settings: Dict[str, object] = dict(
        frontend=kind,
        color=color,
        image_size=list(image_size),
        visual_dim=64,
        stem_dim=64,
        stem_depth=2,
        patch_size=8,
        context_grid=[12, 16],
        token_grid=6,
        aiding_dim=64,
        fusion_dim=96,
        dropout=0.0,
        velocity_mode="auto",
        deployment_latency_s=0.35,
    )
    if kind == "planar":
        # --frontend planar's own defaults: radius 3, 4x coarse pooling, a
        # 1.0 s baseline (20 frames at 20 Hz), one pair ending on every frame.
        settings.update(correlation_radius=3, coarse_factor=4, coarse_radius=6,
                        frame_gap=20, pair_stride=1)
    else:
        settings.update(correlation_radius=4, rotation_mode="field", frame_gap=1, pair_stride=1)
    return settings


def working_camera(calibration: Optional[Path], image_size: Tuple[int, int]) -> Optional[torch.Tensor]:
    loaded = maybe_load_camera_calibration(str(calibration) if calibration else None)
    if loaded is None:
        return None
    matrix = resize_camera_matrix(loaded.camera_matrix, loaded.native_size, image_size)
    return torch.from_numpy(np.asarray(matrix, dtype=np.float32))


def build_untrained(kind: str, image_size: Tuple[int, int], color: bool, calibration: Optional[Path]):
    settings = default_settings(kind, image_size, color)
    torch.manual_seed(0)
    frontend = build_frontend(
        settings,
        camera_from_body=NADIR_MOUNTINGS["top_forward"] if kind == "planar" else None,
        prior_velocity=(20.0, 0.0, 0.0),
        dropout=0.0,
    )
    model = VisionMambaVO(
        visual_dim=64, aiding_dim=64, fusion_dim=96, dropout=0.0,
        frontend=frontend, velocity_mode=resolve_velocity_mode(settings),
    )
    # Non-zero heads so the step does real work end to end (zero-initialised
    # heads would still cost the same; this only keeps outputs non-trivial).
    with torch.no_grad():
        for head in (model.direction_head, model.log_rate_head, model.residual_head):
            if head is not None:
                head.weight.normal_(0.0, 0.05)
    camera = working_camera(calibration, image_size)
    if kind == "planar" and camera is None:
        raise SystemExit("--frontend planar needs --calibration (the camera intrinsics)")
    settings["calibration"] = str(calibration) if calibration else None
    return frontend.eval(), model.eval(), settings, camera


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------


def count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def parameter_breakdown(frontend: nn.Module, model: VisionMambaVO) -> Dict[str, Dict[str, int]]:
    front: Dict[str, int] = {}
    for name, child in frontend.named_children():
        front[name] = count(child)
    direct = sum(p.numel() for _, p in frontend.named_parameters(recurse=False))
    if direct:
        front["(direct parameters)"] = direct
    temporal: Dict[str, int] = {}
    for name, child in model.named_children():
        if name == "frontend":
            continue
        temporal[name] = count(child)
    return {"frontend": front, "temporal": temporal}


def flop_count(fn: Callable[[], object]) -> Optional[int]:
    """Total FLOPs torch's counter sees (matmul/conv-dominated; elementwise ops
    such as the selective-scan exponentials are not counted)."""

    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:  # pragma: no cover - very old torch
        return None
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        fn()
    return int(counter.get_total_flops())


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def time_ms(fn: Callable[[], object], warmup: int, iterations: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p90_ms": samples[min(len(samples) - 1, int(math.ceil(0.9 * len(samples))) - 1)],
        "min_ms": samples[0],
        "iterations": iterations,
    }


def ort_session(path: Path, threads: int, providers: Sequence[str]):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = int(threads)
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=list(providers))


def synthetic_jpeg(width: int, height: int, color: bool) -> bytes:
    """A textured frame at the camera's native size, JPEG-encoded like the capture."""

    from PIL import Image

    rng = np.random.default_rng(0)
    coarse = rng.random((height // 16 + 2, width // 16 + 2, 3 if color else 1))
    picture = Image.fromarray((coarse * 255).astype(np.uint8).squeeze())
    picture = picture.resize((width, height), Image.BICUBIC)
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def preprocess_timing(
    settings: Dict[str, object], camera: Optional[torch.Tensor], calibration: Optional[Path],
    warmup: int, iterations: int,
) -> Optional[Dict[str, object]]:
    """tools/onnx_inference.ImagePreprocessor on one native-size JPEG."""

    try:
        from tools.onnx_inference import ImagePreprocessor
    except ImportError:
        return None
    loaded = maybe_load_camera_calibration(str(calibration) if calibration else None)
    native = (1080, 1920) if loaded is None else tuple(int(v) for v in loaded.native_size)
    meta = {
        "dataset_settings": {"color": bool(settings.get("color")), "image_size": settings["image_size"]},
        "calibration": None if loaded is None else {
            "native_camera_matrix": np.asarray(loaded.camera_matrix).tolist(),
            "distortion": np.asarray(loaded.distortion).reshape(-1).tolist(),
            "images_rectified": bool(loaded.images_rectified),
        },
        "frontend": {"camera_matrix_working": None if camera is None else camera.tolist()},
    }
    try:
        preprocess = ImagePreprocessor(meta)
    except ImportError:
        return None
    # A file path, as the replay hands it: the runtime's preprocessor takes a
    # path or a decoded array, and the path case is the one that decodes.
    frame = Path(tempfile.mkdtemp(prefix="vo_bench_frame_")) / "frame.jpg"
    frame.write_bytes(synthetic_jpeg(native[1], native[0], bool(settings.get("color"))))
    result = time_ms(lambda: preprocess(frame), warmup, iterations)
    result["native_size"] = list(native)
    result["undistort"] = preprocess.maps is not None
    return result


# ---------------------------------------------------------------------------
# I/O contract
# ---------------------------------------------------------------------------


def describe(names: Sequence[str], tensors: Sequence[torch.Tensor]) -> List[Dict[str, object]]:
    return [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", "")}
        for name, tensor in zip(names, tensors)
    ]


def print_io(title: str, rows: List[Dict[str, object]]) -> None:
    print(f"  {title}")
    for row in rows:
        print(f"    {row['name']:<34s} {str(row['shape']):<22s} {row['dtype']}")


# ---------------------------------------------------------------------------
# one configuration
# ---------------------------------------------------------------------------


def benchmark(
    label: str,
    frontend: nn.Module,
    model: VisionMambaVO,
    settings: Dict[str, object],
    camera: Optional[torch.Tensor],
    *,
    calibration: Optional[Path],
    threads: Sequence[int],
    warmup: int,
    iterations: int,
    step_iterations: int,
    use_onnx: bool,
    providers: Sequence[str],
    image_hz: float,
    telemetry_hz: float,
) -> Dict[str, object]:
    image_size = tuple(int(v) for v in settings["image_size"])
    channels = 3 if bool(settings.get("color")) else 1
    planar = bool(getattr(frontend, "requires_pair_geometry", False))
    front = (PlanarFrontendExport(frontend, camera) if planar else FlowFrontendExport(frontend, camera)).eval()
    step = TemporalStepExport(model).eval()

    pair_inputs = frontend_example(front, channels, image_size, 1, seed=3)
    tick = temporal_example(step, 1, 1)[0]
    state0 = tuple(torch.zeros(1, *shape) for _, shape in state_layout(model))
    tick_inputs = tick + state0
    with torch.no_grad():
        pair_outputs = front(*pair_inputs)
        tick_outputs = step(*tick_inputs)

    report: Dict[str, object] = {
        "label": label,
        "frontend_id": getattr(frontend, "frontend_id", None),
        "velocity_mode": model.velocity_mode,
        "image_size": list(image_size),
        "channels": channels,
        "feature_grid": list(frontend.stem.feature_size),
        "io": {
            "frontend_inputs": describe(front.input_names, pair_inputs),
            "frontend_outputs": describe(front.output_names, pair_outputs),
            "temporal_inputs": describe(step.input_names, tick_inputs),
            "temporal_outputs": describe(step.output_names, tick_outputs),
        },
        "parameters": parameter_breakdown(frontend, model),
        "parameters_total": {"frontend": count(frontend),
                             "temporal": count(model) - count(frontend)},
        "state_floats_per_lane": int(sum(int(np.prod(shape)) for _, shape in state_layout(model))),
        "flops": {
            "frontend_per_pair": flop_count(lambda: front(*pair_inputs)),
            "temporal_per_tick": flop_count(lambda: step(*tick_inputs)),
        },
        "timing": {},
    }

    onnx_dir = None
    if use_onnx:
        try:
            import onnxruntime  # noqa: F401
            onnx_dir = Path(tempfile.mkdtemp(prefix="vo_bench_"))
            _export(front, frontend_example(front, channels, image_size, 1), onnx_dir / "frontend.onnx",
                    front.input_names, front.output_names, 17, dynamic_batch=False)
            _export(step, tick_inputs, onnx_dir / "temporal_step.onnx",
                    step.input_names, step.output_names, 17, dynamic_batch=True)
            diffs = verify_frontend(front, onnx_dir / "frontend.onnx", pair_inputs)
            report["onnx_frontend_max_abs_diff"] = max(diffs.values())
            report["onnx_size_mb"] = {
                name: round((onnx_dir / name).stat().st_size / 1e6, 2)
                for name in ("frontend.onnx", "temporal_step.onnx")
            }
        except ImportError:
            onnx_dir = None

    for count_threads in threads:
        torch.set_num_threads(int(count_threads))
        timing: Dict[str, object] = {}
        with torch.no_grad():
            timing["torch_frontend_per_pair"] = time_ms(lambda: front(*pair_inputs), warmup, iterations)
            timing["torch_step_per_tick"] = time_ms(lambda: step(*tick_inputs), 20, step_iterations)
        if onnx_dir is not None:
            front_session = ort_session(onnx_dir / "frontend.onnx", count_threads, providers)
            step_session = ort_session(onnx_dir / "temporal_step.onnx", count_threads, providers)
            front_feeds = {n: t.numpy() for n, t in zip(front.input_names, pair_inputs)}
            step_feeds = {n: t.numpy() for n, t in zip(step.input_names, tick_inputs)}
            timing["onnx_frontend_per_pair"] = time_ms(
                lambda: front_session.run(None, front_feeds), warmup, iterations)
            timing["onnx_step_per_tick"] = time_ms(
                lambda: step_session.run(None, step_feeds), 20, step_iterations)
        report["timing"][f"threads_{count_threads}"] = timing

    report["preprocess_per_frame"] = preprocess_timing(settings, camera, calibration, 2, max(iterations, 10))
    report["budget"] = budget(report, settings, image_hz, telemetry_hz)
    return report


def budget(report: Dict[str, object], settings: Dict[str, object], image_hz: float,
           telemetry_hz: float) -> Dict[str, object]:
    """Duty cycle and latency margin at the rates the model was trained for."""

    frame_gap = int(settings.get("frame_gap") or 1)
    pair_stride = int(settings.get("pair_stride") or 1)
    latency_s = float(settings.get("deployment_latency_s") or 0.35)
    pairs_per_s = image_hz / pair_stride
    pre = report.get("preprocess_per_frame") or {}
    pre_ms = float(pre.get("median_ms", 0.0))
    out: Dict[str, object] = {
        "image_hz": image_hz,
        "telemetry_hz": telemetry_hz,
        "frame_gap": frame_gap,
        "pair_stride": pair_stride,
        "pair_baseline_s": frame_gap / image_hz,
        "pairs_per_s": pairs_per_s,
        "deployment_latency_s": latency_s,
        "per_threads": {},
    }
    for key, timing in report["timing"].items():
        engine = "onnx" if "onnx_frontend_per_pair" in timing else "torch"
        front_ms = timing[f"{engine}_frontend_per_pair"]["median_ms"]
        step_ms = timing[f"{engine}_step_per_tick"]["median_ms"]
        pair_ms = 2.0 * pre_ms + front_ms
        out["per_threads"][key] = {
            "engine": engine,
            "pair_total_ms": pair_ms,
            "frontend_duty": pairs_per_s * pair_ms / 1000.0,
            "tick_duty": telemetry_hz * step_ms / 1000.0,
            "total_duty": (pairs_per_s * pair_ms + telemetry_hz * step_ms) / 1000.0,
            "latency_margin_ms": latency_s * 1000.0 - pair_ms,
            "max_pairs_per_s": 1000.0 / pair_ms if pair_ms > 0 else float("inf"),
            "max_tick_hz": 1000.0 / step_ms if step_ms > 0 else float("inf"),
        }
    return out


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------


def _fmt_flops(value: Optional[int]) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1e9:.2f} GFLOP" if value >= 1e8 else f"{value / 1e6:.2f} MFLOP"


def print_report(report: Dict[str, object]) -> None:
    print()
    print("=" * 78)
    print(f"{report['label']}  ({report['frontend_id']}, velocity_mode={report['velocity_mode']})")
    print("=" * 78)
    h, w = report["image_size"]
    fh, fw = report["feature_grid"]
    print(f"working image {h}x{w}x{report['channels']} -> feature grid {fh}x{fw} ({fh * fw} cells)")
    io_block = report["io"]
    print_io("frontend inputs (once per image pair)", io_block["frontend_inputs"])
    print_io("frontend outputs", io_block["frontend_outputs"])
    heads = [row for row in io_block["temporal_inputs"] if not row["name"].startswith("state_")]
    states = [row for row in io_block["temporal_inputs"] if row["name"].startswith("state_")]
    print_io("temporal step inputs (once per telemetry tick)", heads)
    print(f"    + {len(states)} state tensors ({report['state_floats_per_lane']} floats per lane), "
          f"returned as next_state_*")
    print_io("temporal step outputs",
             [row for row in io_block["temporal_outputs"] if not row["name"].startswith("next_")])

    print("  parameters")
    for group in ("frontend", "temporal"):
        total = report["parameters_total"][group]
        parts = ", ".join(f"{k} {v:,}" for k, v in report["parameters"][group].items() if v)
        print(f"    {group:<9s} {total:>9,}   ({parts})")
    print(f"    total     {sum(report['parameters_total'].values()):>9,}")
    flops = report["flops"]
    print(f"  compute: frontend {_fmt_flops(flops['frontend_per_pair'])}/pair, "
          f"temporal {_fmt_flops(flops['temporal_per_tick'])}/tick (matmul/conv only)")
    if "onnx_size_mb" in report:
        sizes = report["onnx_size_mb"]
        print(f"  onnx: frontend.onnx {sizes['frontend.onnx']} MB, temporal_step.onnx "
              f"{sizes['temporal_step.onnx']} MB, |onnx - torch| frontend "
              f"{report['onnx_frontend_max_abs_diff']:.1e}")

    print("  latency (median / p90, ms)")
    for key, timing in report["timing"].items():
        cells = []
        for name, label in (("torch_frontend_per_pair", "torch pair"), ("onnx_frontend_per_pair", "onnx pair"),
                            ("torch_step_per_tick", "torch tick"), ("onnx_step_per_tick", "onnx tick")):
            if name in timing:
                cells.append(f"{label} {timing[name]['median_ms']:.2f}/{timing[name]['p90_ms']:.2f}")
        print(f"    {key:<10s} " + "   ".join(cells))
    pre = report.get("preprocess_per_frame")
    if pre:
        print(f"    preprocess one {pre['native_size'][1]}x{pre['native_size'][0]} JPEG frame "
              f"(decode{' + undistort' if pre['undistort'] else ''} + resize): "
              f"{pre['median_ms']:.2f}/{pre['p90_ms']:.2f}")

    b = report["budget"]
    print(f"  real-time budget: images {b['image_hz']:g} Hz, frame_gap {b['frame_gap']} "
          f"({b['pair_baseline_s']:.2f} s baseline), pair_stride {b['pair_stride']} -> "
          f"{b['pairs_per_s']:g} pairs/s; telemetry {b['telemetry_hz']:g} Hz; trained latency "
          f"{b['deployment_latency_s'] * 1000:.0f} ms")
    for key, row in b["per_threads"].items():
        verdict = "OK" if row["total_duty"] < 1.0 and row["latency_margin_ms"] > 0 else "OVER BUDGET"
        print(f"    {key:<10s} [{row['engine']}] pair {row['pair_total_ms']:.1f} ms "
              f"(max {row['max_pairs_per_s']:.1f} pairs/s), frontend duty {row['frontend_duty'] * 100:.1f}%, "
              f"tick duty {row['tick_duty'] * 100:.1f}% (max {row['max_tick_hz']:.0f} Hz), "
              f"total {row['total_duty'] * 100:.1f}%, latency margin {row['latency_margin_ms']:.0f} ms"
              f"  -> {verdict}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="train_fixedwing_vo.py checkpoint; settings and rates come from it")
    parser.add_argument("--frontend", choices=("mamba_correlation", "planar", "both"), default="both",
                        help="Without --checkpoint: which untrained default model(s) to build")
    parser.add_argument("--calibration", type=Path, default=None,
                        help=f"Camera JSON (default with no checkpoint: {DEFAULT_CALIBRATION.relative_to(_ROOT)})")
    parser.add_argument("--image-size", type=int, nargs=2, default=(576, 1024), metavar=("H", "W"))
    parser.add_argument("--color", action="store_true", help="RGB frames (default grayscale)")
    parser.add_argument("--frame-gap", type=int, default=None, help="Override the pair baseline in frames")
    parser.add_argument("--output-on-pairs", action="store_true",
                        help="pair_stride = frame_gap (one non-overlapping pair per baseline)")
    parser.add_argument("--image-hz", type=float, default=20.0)
    parser.add_argument("--telemetry-hz", type=float, default=100.0)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, torch.get_num_threads()])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--step-iterations", type=int, default=500)
    parser.add_argument("--no-onnx", action="store_true", help="PyTorch timing only")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    parser.add_argument("--output", type=Path, default=None, help="Write the full report as JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    threads = sorted(set(int(t) for t in args.threads))
    configs = []
    if args.checkpoint is not None:
        frontend, model, saved, camera, _ = load_models(args.checkpoint, args.calibration)
        calibration = args.calibration or (Path(saved["calibration"]) if saved.get("calibration") else None)
        configs.append((str(args.checkpoint), frontend, model, saved, camera, calibration))
    else:
        calibration = args.calibration or DEFAULT_CALIBRATION
        kinds = ("mamba_correlation", "planar") if args.frontend == "both" else (args.frontend,)
        for kind in kinds:
            frontend, model, settings, camera = build_untrained(
                kind, tuple(args.image_size), args.color, calibration
            )
            configs.append((f"{kind} (untrained, trainer defaults)", frontend, model, settings, camera,
                            calibration))

    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads available, "
          f"cuda={'yes' if torch.cuda.is_available() else 'no'} (timing runs on CPU)")
    reports = []
    for label, frontend, model, settings, camera, calibration in configs:
        settings = dict(settings)
        if args.frame_gap is not None:
            settings["frame_gap"] = args.frame_gap
        if args.output_on_pairs:
            settings["pair_stride"] = int(settings.get("frame_gap") or 1)
        report = benchmark(
            label, frontend, model, settings, camera,
            calibration=calibration, threads=threads, warmup=args.warmup,
            iterations=args.iterations, step_iterations=args.step_iterations,
            use_onnx=not args.no_onnx, providers=args.providers,
            image_hz=args.image_hz, telemetry_hz=args.telemetry_hz,
        )
        print_report(report)
        reports.append(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
