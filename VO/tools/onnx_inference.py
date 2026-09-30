#!/usr/bin/env python3
"""Standalone ONNX inference for the fixed-wing VO model - the deployable runtime.

Needs only numpy, onnxruntime and Pillow (OpenCV only if the calibration has
lens distortion). No torch, no project code: copy this file and the export
folder (``frontend.onnx``, ``temporal_step.onnx``, ``vo_onnx.json``) to the
target machine.

Live use - feed telemetry at its own rate and frames as they are captured::

    vo = VOOnnxRuntime("export/onnx")
    vo.add_frame(image_rgb_uint8, capture_time_s)        # every camera frame
    out = vo.add_telemetry(t, roll, pitch, yaw, rel_alt) # every telemetry row
    if out["emitted"]:
        print(out["velocity"])                           # m/s, body frame

Angles in radians, times in seconds on the TELEMETRY clock (apply the image
time offset before calling add_frame, or pass it in vo_onnx.json). Frames must
be added in capture order, and a frame must be added before the first
telemetry row later than its capture time.

Replay a recorded flight folder (``flight.csv`` + ``images/``) the same way::

    python tools/onnx_inference.py export/onnx --dataset data_split/test \\
        --output artifacts/onnx_velocity_test.csv

``--checkpoint`` also runs the PyTorch model over the same flight (this part
does need torch and the project) and fails if the two disagree.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# rotations (body-to-NED, quaternions w, x, y, z)
# ---------------------------------------------------------------------------


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic Z-Y-X (yaw, pitch, roll) body-to-NED quaternion."""

    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    q = np.array(
        (
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
        )
    )
    return q / np.linalg.norm(q)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def quaternion_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    vector = q[1:]
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-10:
        return vector * 2.0
    return vector * (2.0 * math.atan2(norm, max(q[0], 0.0)) / norm)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


# ---------------------------------------------------------------------------
# image preprocessing
# ---------------------------------------------------------------------------


class ImagePreprocessor:
    """File or array -> float32 (C, H, W) in [0, 1] at the working size."""

    def __init__(self, meta: Dict[str, object]) -> None:
        settings = meta["dataset_settings"]
        self.color = bool(settings.get("color", False))
        height, width = (int(v) for v in settings.get("image_size") or (576, 1024))
        self.size = (height, width)
        self.maps = None
        calibration = meta.get("calibration") or {}
        distortion = np.asarray(calibration.get("distortion") or [], dtype=np.float64)
        if distortion.size and np.any(np.abs(distortion) > 0) and not calibration.get("images_rectified"):
            import cv2

            self.maps = cv2.initUndistortRectifyMap(
                np.asarray(calibration["native_camera_matrix"], dtype=np.float64),
                distortion,
                None,
                np.asarray(meta["frontend"]["camera_matrix_working"], dtype=np.float64),
                (width, height),
                cv2.CV_32FC1,
            )

    def __call__(self, image) -> np.ndarray:
        from PIL import Image

        if isinstance(image, (str, Path)):
            with Image.open(image) as handle:
                picture = handle.convert("RGB" if self.color else "L")
        else:
            array = np.asarray(image)
            picture = Image.fromarray(array.astype(np.uint8))
            picture = picture.convert("RGB" if self.color else "L")
        height, width = self.size
        if self.maps is not None:
            import cv2

            array = cv2.remap(
                np.asarray(picture), self.maps[0], self.maps[1],
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            )
        else:
            if picture.size != (width, height):
                picture = picture.resize((width, height), Image.BILINEAR)
            array = np.asarray(picture)
        if array.ndim == 2:
            array = array[:, :, None]
        return np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# the runtime
# ---------------------------------------------------------------------------


class VOOnnxRuntime:
    """Streams telemetry and frames through the two exported graphs.

    Reproduces the training-time rules exactly: pair ``(k - frame_gap, k)`` for
    every ``pair_stride``-th frame, delivered one ``deployment_latency_s``
    after its second exposure (on the first telemetry row at or after that),
    attitude and altitude interpolated at both exposures, refused and
    over-long pairs not delivered, the geometric velocity held between pairs,
    and the visual age measured from the capture instant.
    """

    def __init__(self, onnx_dir: "str | Path", providers: Optional[Sequence[str]] = None) -> None:
        import onnxruntime as ort

        folder = Path(onnx_dir)
        self.meta = json.loads((folder / "vo_onnx.json").read_text(encoding="utf-8"))
        meta = self.meta
        providers = list(providers or ["CPUExecutionProvider"])
        self.frontend = ort.InferenceSession(str(folder / meta["files"]["frontend"]), providers=providers)
        self.step = ort.InferenceSession(str(folder / meta["files"]["temporal_step"]), providers=providers)
        self.front_inputs = list(meta["frontend"]["inputs"])
        self.front_outputs = list(meta["frontend"]["outputs"])
        self.front_batch = int(meta["frontend"]["inputs"]["image0"][0])
        step_meta = meta["temporal_step"]
        self.step_inputs = list(step_meta["inputs"])
        self.step_outputs = list(step_meta["outputs"])
        self.state_meta = step_meta["state"]
        self.geometric = "visual_velocity" in self.step_inputs
        self.planar = "relative_rotation" in self.front_inputs
        self.visual_dim = int(self.step.get_inputs()[self.step_inputs.index("visual_token")].shape[1])
        normalizer = step_meta["normalizer"]
        self.log_altitude_mean = float(normalizer["log_altitude_mean"])
        self.delta_time_scale = float(normalizer["delta_time_scale"]) or 1.0
        settings = meta["dataset_settings"]
        timing = meta["timing"]
        self.frame_gap = int(timing["frame_gap"])
        self.pair_stride = int(timing["pair_stride"])
        self.latency = float(timing["deployment_latency_s"])
        self.output_on_pairs = bool(timing["output_on_pairs"])
        self.max_frame_gap_s = settings.get("max_frame_gap_s")
        # Same rule as training: an exposure inside a telemetry hole wider than
        # 8 telemetry steps is not a measurement.
        self.max_telemetry_gap_s = 8.0 * self.delta_time_scale
        self.preprocess = ImagePreprocessor(meta)
        self.history_s = 10.0 + self.latency + 2.0 * float(self.max_frame_gap_s or 2.0)
        self.reset()

    # -- state ---------------------------------------------------------------

    def reset(self) -> None:
        """Cold start: zero state, nothing held, no frames, no telemetry."""

        self.state = {
            entry["name"]: np.zeros([1, *entry["shape"][1:]], dtype=np.float32) for entry in self.state_meta
        }
        self.held = np.zeros((1, 3), dtype=np.float32)
        self.held_valid = np.zeros((1, 1), dtype=np.float32)
        self.last_delivery_s: Optional[float] = None
        self.emitted = np.zeros(3, dtype=np.float32)
        self.frames: "OrderedDict[int, Tuple[float, object]]" = OrderedDict()
        self.frame_count = 0
        # (ready time, (t0, frame0), (t1, frame1)): a pending pair holds its own
        # frames, since newer frames push them out of the buffer before it is due.
        self.pending: List[Tuple[float, Tuple[float, object], Tuple[float, object]]] = []
        self.times: List[float] = []
        self.quats: List[np.ndarray] = []
        self.altitudes: List[float] = []
        self.first_telemetry_s: Optional[float] = None
        self.stats = {"pairs": 0, "delivered": 0, "refused": 0, "rejected_gap": 0}

    # -- inputs ----------------------------------------------------------------

    def add_frame(self, image, capture_time_s: float) -> None:
        """One camera frame (path, or HxW / HxWx3 uint8 array), telemetry clock."""

        index = self.frame_count
        self.frame_count += 1
        # Frames are preprocessed when a pair needs them; keep only what a
        # future pair can still use.
        self.frames[index] = (float(capture_time_s), image)
        while self.frames and next(iter(self.frames)) < index - self.frame_gap:
            self.frames.popitem(last=False)
        first = index - self.frame_gap
        if first >= 0 and first % self.pair_stride == 0 and first in self.frames:
            t0 = self.frames[first][0]
            t1 = float(capture_time_s)
            if self.max_frame_gap_s is not None and t1 - t0 > float(self.max_frame_gap_s):
                self.stats["rejected_gap"] += 1
                return
            self.pending.append((t1 + self.latency, self.frames[first], self.frames[index]))

    def add_telemetry(
        self, time_s: float, roll: float, pitch: float, yaw: float, relative_altitude_m: float
    ) -> Dict[str, object]:
        """One telemetry row; returns this tick's velocity (and whether it is an output)."""

        t = float(time_s)
        q = euler_to_quaternion(roll, pitch, yaw)
        if self.quats and float(np.dot(self.quats[-1], q)) < 0.0:
            q = -q
        if self.times:
            dt = t - self.times[-1]
            rate = quaternion_to_rotvec(quaternion_multiply(self.quats[-1] * np.array([1, -1, -1, -1]), q))
            rate = rate / dt if dt > 0 else np.zeros(3)
        else:
            dt = self.delta_time_scale
            rate = np.zeros(3)
            self.first_telemetry_s = t
        self.times.append(t)
        self.quats.append(q)
        self.altitudes.append(float(relative_altitude_m))
        while len(self.times) > 2 and self.times[1] < t - self.history_s:
            del self.times[0], self.quats[0], self.altitudes[0]

        # Pairs whose result is due by now - on this row, as in training.
        delivered = None
        due = [p for p in self.pending if p[0] <= t]
        self.pending = [p for p in self.pending if p[0] > t]
        for ready, frame0, frame1 in due:
            if ready < float(self.first_telemetry_s):
                continue
            result = self._run_pair(frame0, frame1)
            if result is not None:
                delivered = result  # of two on one row, the later one wins

        present = delivered is not None
        token = np.zeros((1, self.visual_dim), dtype=np.float32)
        quality = np.zeros((1, 1), dtype=np.float32)
        if present:
            token = delivered["visual_token"].reshape(1, -1).astype(np.float32)
            quality = delivered["visual_quality"].reshape(1, 1).astype(np.float32)
            self.last_delivery_s = t
            if self.geometric:
                self.held = delivered["geometric_velocity"].reshape(1, 3).astype(np.float32)
                self.held_valid[:] = 1.0
        age = 0.0 if self.last_delivery_s is None else t - self.last_delivery_s + self.latency

        log_altitude = math.log(max(float(relative_altitude_m), 1.0))
        aiding = np.array(
            [[
                math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch),
                log_altitude - self.log_altitude_mean,
                *rate,
                dt / max(self.delta_time_scale, 1e-9),
            ]],
            dtype=np.float32,
        )
        feeds = {
            "aiding": aiding,
            "visual_token": token,
            "visual_present": np.full((1, 1), float(present), dtype=np.float32),
            "visual_age": np.full((1, 1), age, dtype=np.float32),
            "visual_quality": quality,
            "log_altitude": np.array([log_altitude], dtype=np.float32),
        }
        if self.geometric:
            feeds["visual_velocity"] = self.held
            feeds["visual_velocity_valid"] = self.held_valid
        feeds.update(self.state)
        result = dict(zip(self.step_outputs, self.step.run(self.step_outputs, feeds)))
        self.state = {entry["name"]: result["next_" + entry["name"]] for entry in self.state_meta}
        velocity = result["predicted_velocity"][0]
        if present or not self.output_on_pairs:
            self.emitted = velocity
        return {
            "time_s": t,
            "velocity": velocity,
            "emitted": present if self.output_on_pairs else True,
            "output": self.emitted,
            "pair_delivered": present,
            "log_variance": result["velocity_log_variance"][0],
        }

    # -- internals -------------------------------------------------------------

    def _attitude_at(self, query: float) -> Tuple[np.ndarray, float]:
        """nlerp attitude and linear altitude at ``query``, held at the ends."""

        times = np.asarray(self.times)
        clamped = min(max(query, times[0]), times[-1])
        upper = int(np.clip(np.searchsorted(times, clamped, side="right"), 1, times.size - 1)) if times.size > 1 else 0
        lower = max(upper - 1, 0)
        span = times[upper] - times[lower]
        fraction = (clamped - times[lower]) / span if span > 0 else 0.0
        q = (1.0 - fraction) * self.quats[lower] + fraction * self.quats[upper]
        altitude = float(np.interp(clamped, times, np.asarray(self.altitudes)))
        return q / np.linalg.norm(q), altitude

    def _straddles_hole(self, query: float) -> bool:
        times = np.asarray(self.times)
        if times.size < 2:
            return False
        upper = int(np.clip(np.searchsorted(times, query, side="left"), 1, times.size - 1))
        return float(times[upper] - times[upper - 1]) > self.max_telemetry_gap_s

    def _run_pair(self, frame0, frame1) -> Optional[Dict[str, np.ndarray]]:
        t0, image0 = frame0
        t1, image1 = frame1
        if self._straddles_hole(t0) or self._straddles_hole(t1):
            self.stats["rejected_gap"] += 1
            return None
        self.stats["pairs"] += 1
        feeds = {
            "image0": self.preprocess(image0)[None],
            "image1": self.preprocess(image1)[None],
            "pair_dt_s": np.array([t1 - t0], dtype=np.float32),
        }
        q0, alt0 = self._attitude_at(t0)
        q1, alt1 = self._attitude_at(t1)
        r0, r1 = quaternion_to_matrix(q0), quaternion_to_matrix(q1)
        if self.planar:
            feeds["relative_rotation"] = (r0.T @ r1)[None].astype(np.float32)
            feeds["down_body"] = r0[2][None].astype(np.float32)
            feeds["altitude_m"] = np.array([[alt0, alt1]], dtype=np.float32)
        else:
            rotvec = quaternion_to_rotvec(quaternion_multiply(q0 * np.array([1, -1, -1, -1]), q1))
            feeds["body_rate_rad_s"] = (rotvec / max(t1 - t0, 1e-6))[None].astype(np.float32)
        if self.front_batch > 1:
            feeds = {k: np.repeat(v, self.front_batch, axis=0) for k, v in feeds.items()}
        out = dict(zip(self.front_outputs, self.frontend.run(self.front_outputs, feeds)))
        out = {k: v[:1] for k, v in out.items()}
        if float(out["pair_reliable"].reshape(-1)[0]) <= 0:
            self.stats["refused"] += 1
            return None
        self.stats["delivered"] += 1
        return out


# ---------------------------------------------------------------------------
# replay a recorded flight folder
# ---------------------------------------------------------------------------

#: Same choice order as vio.data.attitude (the flight computer's attitude,
#: never GPSNavEul*, which is part of the training target).
ATTITUDE_CANDIDATES = (("NavEulX", "NavEulY", "NavEulZ"), ("EulX", "EulY", "EulZ"))
ALTITUDE_CANDIDATES = ("relativeAlt", "RelativeAlt", "RelatedAlt", "Barometer")


def read_flight_csv(path: Path, meta: Dict[str, object]) -> Dict[str, np.ndarray]:
    """times (s), roll/pitch/yaw (rad) and relative altitude (m) from a flight CSV."""

    settings = meta["dataset_settings"]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = list(next(reader))
        rows = [row for row in reader if row]
    attitude = settings.get("attitude_columns") or next(
        (c for c in ATTITUDE_CANDIDATES if all(n in header for n in c)), None
    )
    altitude = settings.get("altitude_column") or next(
        (name for name in ALTITUDE_CANDIDATES if name in header), None
    )
    time_name = settings.get("time_column") or "Time"
    names = (time_name, *(attitude or ()), altitude)
    if attitude is None or altitude is None or any(name not in header for name in names):
        raise SystemExit(f"cannot find time/attitude/altitude columns in {path.name}")
    indices = [header.index(name) for name in names]  # first occurrence, as training
    data = np.array([[float(row[i]) for i in indices] for row in rows], dtype=np.float64)
    times = data[:, 0] * float(settings.get("time_scale") or 1.0)
    euler = data[:, 1:4]
    # Same unit rule as training: a yaw in radians never exceeds 2*pi.
    if float(np.nanmax(np.abs(euler))) > 2 * math.pi + 1e-6:
        euler = np.deg2rad(euler)
    return {"times": times, "euler": euler, "altitude": data[:, 4]}


def list_frames(folder: Path, meta: Dict[str, object]) -> List[Tuple[float, Path]]:
    settings = meta["dataset_settings"]
    scale = float(meta.get("image_time_scale") or 0.001)
    offset = settings.get("image_time_offset_s") or 0.0
    if isinstance(offset, dict):
        raise SystemExit("a per-time offset table is not supported here; use a constant --image-time-offset")
    frames = []
    for path in folder.glob(meta.get("image_pattern") or "*.jpg"):
        stamp = path.stem.rsplit("_", 1)[-1]
        if stamp.isdigit():
            frames.append((int(stamp) * scale + float(offset), path))
    frames.sort(key=lambda item: item[0])
    return frames


def replay(onnx_dir: Path, dataset: Path, *, max_minutes: Optional[float] = None,
           progress: bool = True) -> Dict[str, np.ndarray]:
    vo = VOOnnxRuntime(onnx_dir)
    settings = vo.meta["dataset_settings"]
    flight = read_flight_csv(dataset / (settings.get("csv_name") or "flight.csv"), vo.meta)
    frames = list_frames(dataset / (settings.get("image_folder") or "images"), vo.meta)
    times = flight["times"]
    total = times.size
    if max_minutes is not None:
        total = int(np.searchsorted(times, times[0] + 60.0 * max_minutes))
    velocity = np.zeros((total, 3), dtype=np.float32)
    output = np.zeros((total, 3), dtype=np.float32)
    delivered = np.zeros(total, dtype=bool)
    next_frame = 0
    report_every = max(total // 10, 1)
    for tick in range(total):
        # Every frame captured by now has arrived before this row.
        while next_frame < len(frames) and frames[next_frame][0] <= times[tick]:
            vo.add_frame(frames[next_frame][1], frames[next_frame][0])
            next_frame += 1
        roll, pitch, yaw = flight["euler"][tick]
        out = vo.add_telemetry(times[tick], roll, pitch, yaw, flight["altitude"][tick])
        velocity[tick] = out["velocity"]
        output[tick] = out["output"]
        delivered[tick] = out["pair_delivered"]
        if progress and tick % report_every == 0:
            print(f"  tick {tick}/{total}  pairs delivered {vo.stats['delivered']}", flush=True)
    return {"times": times[:total], "velocity": velocity, "output": output, "delivered": delivered,
            "stats": vo.stats, "meta": vo.meta, "total": total}


def target_velocity(dataset: Path, meta: Dict[str, object], total: int) -> Optional[np.ndarray]:
    """Reference body velocity, via the project's code (only for scoring)."""

    try:
        root = Path(__file__).resolve().parents[1]
        for entry in (root / "src", root):
            if str(entry) not in sys.path:
                sys.path.insert(0, str(entry))
        from vio.data.fixedwing_vo import reference_body_velocity
    except ImportError:
        return None
    settings = meta["dataset_settings"]
    _, velocity = reference_body_velocity(
        dataset / (settings.get("csv_name") or "flight.csv"),
        time_column=settings.get("time_column") or "Time",
        time_scale=float(settings.get("time_scale") or 1.0),
        lever_arm_m=settings.get("lever_arm_m"),
    )
    return np.asarray(velocity[:total], dtype=np.float64)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx_dir", type=Path)
    parser.add_argument("--dataset", type=Path, required=True, help="flight folder: flight.csv + images/")
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None, help="per-tick velocity CSV")
    parser.add_argument("--checkpoint", type=Path, default=None, help="compare against PyTorch (needs torch)")
    parser.add_argument("--tolerance", type=float, default=0.05, help="m/s, for --checkpoint")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    run = replay(args.onnx_dir, args.dataset, max_minutes=args.max_minutes, progress=not args.no_progress)
    meta, total = run["meta"], run["total"]
    print(f"ticks {total}, pairs {run['stats']}")
    on_pairs = bool(meta["timing"]["output_on_pairs"])
    warmup = int(meta["dataset_settings"].get("warmup") or 0)
    target = target_velocity(args.dataset, meta, total)
    if target is not None:
        scored = np.arange(total) >= warmup
        if on_pairs:
            scored &= run["delivered"]
        error = run["velocity"][scored] - target[scored]
        rmse = float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))) if error.size else float("nan")
        print(f"ONNX vel_rmse {rmse:.3f} m/s over {int(scored.sum())} "
              + ("pair outputs" if on_pairs else "ticks"))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", "vx", "vy", "vz", "out_vx", "out_vy", "out_vz", "pair_delivered"]
                            + (["true_vx", "true_vy", "true_vz"] if target is not None else []))
            for i in range(total):
                writer.writerow(
                    [f"{run['times'][i]:.4f}", *(f"{v:.4f}" for v in run["velocity"][i]),
                     *(f"{v:.4f}" for v in run["output"][i]), int(run["delivered"][i])]
                    + ([f"{v:.4f}" for v in target[i]] if target is not None else [])
                )
        print(f"series -> {args.output}")

    if args.checkpoint is not None:
        from tools.onnx_reference import torch_reference

        reference, reference_present = torch_reference(args.checkpoint, args.dataset, total)
        same_pairs = bool(np.array_equal(reference_present, run["delivered"]))
        print(f"pair delivery ticks identical to PyTorch: {same_pairs}")
        diff = float(np.max(np.abs(reference - run["velocity"])))
        print(f"max |ONNX - PyTorch| velocity over {total} ticks: {diff:.2e} m/s")
        if not same_pairs or not np.isfinite(diff) or diff > args.tolerance:
            print("ONNX INFERENCE CHECK FAILED")
            return 1
        print("ONNX INFERENCE MATCHES PYTORCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
