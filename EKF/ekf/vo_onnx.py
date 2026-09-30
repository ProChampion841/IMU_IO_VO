"""Run the VO ONNX runtime (VO/tools/onnx_inference.py) and hand its output to the EKF.

VO/tools/onnx_inference.py is the VO project's deployable runtime: two ONNX graphs
(frontend.onnx per image pair, temporal_step.onnx per telemetry tick) plus
vo_onnx.json with the timing it was trained with (frame_gap, pair_stride,
deployment_latency_s, output_on_pairs).  It returns a velocity on EVERY telemetry
tick and flags the ticks where a new image pair was delivered.

This module replays one VO flight folder (flight.csv + images/) through it and
keeps what the EKF needs:
    only the ticks with pair_delivered == 1   (one per image pair, e.g. every
                                               500 ms at frame_gap 10: the rows
                                               in between re-read the same token)
    velocity in body FLU                      (VO predicts FRD; y and z flipped)
    variance exp(velocity_log_variance)       (the VO CSV writer drops this column,
                                               so it is read from the runtime here)
It also writes the full per-tick series to a CSV that load_vo_csv() reads back,
so the (slow, image-bound) VO pass runs once per flight.
"""
import csv
import importlib.util
import os

import numpy as np

from .so3 import frd_to_flu
from .vo import VOStream

VO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "VO"))
_RUNTIME = None


def runtime_module():
    """VO/tools/onnx_inference.py, loaded by PATH: both IMU/ and VO/ have a package
    called `tools`, so a plain import would pick whichever is first on sys.path."""
    global _RUNTIME
    if _RUNTIME is None:
        path = os.path.join(VO_ROOT, "tools", "onnx_inference.py")
        if not os.path.isfile(path):
            raise FileNotFoundError("%s not found -- the VO ONNX runtime is needed "
                                    "(VO/tools/onnx_inference.py)" % path)
        spec = importlib.util.spec_from_file_location("vo_onnx_runtime", path)
        _RUNTIME = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_RUNTIME)
    return _RUNTIME


def replay(onnx_dir, dataset_dir, time_offset=0.0, var_scale=1.0, min_std=0.05,
           max_minutes=None, save_csv=None, progress=True):
    """VO ONNX over one flight folder -> VOStream (fresh pairs only, body FLU)."""
    from pathlib import Path
    rt = runtime_module()
    vo = rt.VOOnnxRuntime(onnx_dir)
    settings = vo.meta["dataset_settings"]
    dataset = Path(dataset_dir)
    flight = rt.read_flight_csv(dataset / (settings.get("csv_name") or "flight.csv"), vo.meta)
    frames = rt.list_frames(dataset / (settings.get("image_folder") or "images"), vo.meta)
    times = flight["times"]
    total = times.size
    if max_minutes is not None:
        total = int(np.searchsorted(times, times[0] + 60.0 * max_minutes))
    vel = np.zeros((total, 3))
    lv = np.zeros((total, 3))
    fresh = np.zeros(total, bool)
    nxt = 0
    step = max(total // 10, 1)
    for k in range(total):
        while nxt < len(frames) and frames[nxt][0] <= times[k]:
            vo.add_frame(frames[nxt][1], frames[nxt][0])
            nxt += 1
        roll, pitch, yaw = flight["euler"][k]
        out = vo.add_telemetry(times[k], roll, pitch, yaw, flight["altitude"][k])
        vel[k], lv[k], fresh[k] = out["velocity"], out["log_variance"], out["pair_delivered"]
        if progress and k % step == 0:
            print("  [vo-onnx] tick %d/%d  pairs delivered %d" % (k, total, vo.stats["delivered"]),
                  flush=True)
    t = times[:total]
    if save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "vx", "vy", "vz", "velocity_log_variance_x",
                        "velocity_log_variance_y", "velocity_log_variance_z", "pair_delivered"])
            for k in range(total):
                w.writerow(["%.4f" % t[k], *("%.5f" % x for x in vel[k]),
                            *("%.5f" % x for x in lv[k]), int(fresh[k])])
    timing = vo.meta["timing"]
    var = np.maximum(np.exp(lv[fresh]) * float(var_scale), min_std ** 2)
    ok = np.isfinite(vel[fresh]).all(1) & np.isfinite(var).all(1)
    src = ("vo-onnx:%s on %s (frame_gap %s, pair_stride %s, latency %.2f s; %d pairs of %d ticks)"
           % (onnx_dir, dataset_dir, timing.get("frame_gap"), timing.get("pair_stride"),
              float(timing.get("deployment_latency_s", 0.0)), int(fresh.sum()), total))
    return VOStream(t[fresh][ok] + float(time_offset), frd_to_flu(vel[fresh][ok]), var[ok], src)
