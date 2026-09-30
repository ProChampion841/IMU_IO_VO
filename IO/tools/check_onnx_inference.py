"""Check tools/onnx_inference.py against the PyTorch pipeline on the same flight.

Runs one flight CSV two ways and compares them:

  A. tools/onnx_inference.py: numpy loader + onnxruntime + numpy metrics
  B. the training code: datasets/UAVdataset.py loader, SeqeuncesDataset windows,
     the config's collate (padding9_honest), VelocityNet.forward on the checkpoint,
     and window_errors() from tools/eval_vel_horizons.py, pooled over ALL windows

Checked: the converted acc/gyro, the per-frame body velocity, the ground-truth
body velocity, and vel_rmse / vel_max / dir_rmse / dir_max / pos_error.

Without --csv a synthetic flight in the raw log format is generated (ground
segments at both ends to exercise the airborne trim, a few NaN rows, jittered
100 Hz time), so the check needs no real data.

    python -m tools.check_onnx_inference --config configs/exp/UAV/velnet_v1.conf \\
        --ckpt logs/best_model.ckpt --onnx velnet_6000.onnx [--csv flight.csv]
"""
import argparse
import copy
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import pandas as pd
import torch
from pyhocon import ConfigFactory

from datasets import SeqeuncesDataset, collate_fcs
from model import net_dict
from tools import onnx_inference as oi
from tools.eval_vel_horizons import window_errors


def synthetic_flight(path, minutes=5.0, seed=0):
    """A raw-format log: ~100 Hz, turning flight at ~22 m/s, ground at both ends."""
    rng = np.random.default_rng(seed)
    n_air = int(minutes * 60 * 100)
    n_gnd = 300
    n = n_air + 2 * n_gnd
    dt = 0.00993 + 0.0005 * rng.standard_normal(n)
    t = np.cumsum(np.clip(dt, 0.005, 0.02))
    yaw_rate = 0.08 * np.sin(2 * np.pi * t / 90.0) + 0.02 * np.sin(2 * np.pi * t / 17.0)
    yaw = np.cumsum(yaw_rate * np.r_[dt[0], np.diff(t)])
    roll = np.arctan(22.0 * yaw_rate / 9.81) + 0.02 * rng.standard_normal(n)
    pitch = 0.05 + 0.03 * np.sin(2 * np.pi * t / 40.0)
    speed = np.r_[np.zeros(n_gnd), 22 + 3 * np.sin(2 * np.pi * t[n_gnd:-n_gnd] / 60.0),
                  np.zeros(n_gnd)]
    vn, ve = speed * np.cos(yaw) + 2.0, speed * np.sin(yaw) - 1.0
    vd = -1.0 * np.sin(2 * np.pi * t / 50.0)
    df = pd.DataFrame({
        "Time": t,
        "GyroX": np.rad2deg(np.gradient(roll, t)) + 0.5 * rng.standard_normal(n),
        "GyroY": np.rad2deg(np.gradient(pitch, t)) + 0.5 * rng.standard_normal(n),
        "GyroZ": np.rad2deg(yaw_rate) + 0.5 * rng.standard_normal(n),
        "AcclX": 0.05 * np.sin(2 * np.pi * t / 60.0) + 0.02 * rng.standard_normal(n),
        "AcclY": 0.03 * rng.standard_normal(n),
        "AcclZ": -1.0 / np.cos(roll) + 0.05 * rng.standard_normal(n),
        "GPSNavEulX": roll, "GPSNavEulY": pitch,
        "GPSNavEulZ": np.angle(np.exp(1j * yaw)),
        "GPSNavVnX": vn * (speed > 0), "GPSNavVnY": ve * (speed > 0),
        "GPSNavVnZ": vd * (speed > 0),
        "GPSNavAlt": 100 + np.cumsum(-vd * 0.01), "AirSpeed": speed + 1.0,
        # MTi attitude (degrees, z-up); the training loader requires it.  Unused by a
        # model with att_input none.
        "EulX": np.rad2deg(roll), "EulY": -np.rad2deg(pitch),
        "EulZ": np.rad2deg(np.angle(np.exp(1j * (np.deg2rad(76.94) - yaw)))),
    })
    bad = [i for i in (n_gnd + 500, n_gnd + 12345) if i < n_gnd + n_air]
    df.loc[bad, "AcclY"] = np.nan                             # rows the loader drops
    df.to_csv(path, index=False)
    return path


def torch_reference(conf, net, csv, F):
    """Per-window outputs and pooled metrics through the training pipeline."""
    dc = copy.deepcopy(conf.dataset.eval)
    dc.mode = "evaluate"
    entry = dc.data_list[0]
    entry["data_root"] = os.path.dirname(os.path.abspath(csv))
    entry["data_drive"] = [os.path.basename(csv)]
    entry["window_size"] = entry["step_size"] = int(F)
    del dc.data_list[1:]
    ds = SeqeuncesDataset(data_set_config=dc)
    collate = collate_fcs[conf.dataset.get("collate", "base")]

    seq = {"acc": ds.acc[0].double().numpy(), "gyro": ds.gyro[0].double().numpy()}
    windows = []
    v_sq = d_sq = p_sq = v_max = d_max = 0.0
    nf = 0
    with torch.no_grad():
        for i in range(len(ds)):
            data, init_state, label = collate([ds[i]])
            out = net(data, init_state)
            ev, ed, ep, _ = window_errors(out, data, init_state, label)
            gt_body = label["gt_rot"].Inv() @ label["gt_vel"]
            windows.append((ds.index_map[i][1], out["vel_body"][0].double().numpy(),
                            gt_body[0].double().numpy()))
            v_sq += float(ev.double().pow(2).sum()); d_sq += float(ed.double().pow(2).sum())
            p_sq += float(ep.double().pow(2).sum())
            v_max = max(v_max, float(ev.max())); d_max = max(d_max, float(ed.max()))
            nf += ev.numel()
    nw = len(windows)
    metrics = {"windows": nw, "vel_rmse": (v_sq / nf) ** 0.5, "vel_max": v_max,
               "dir_rmse": (d_sq / nf) ** 0.5, "dir_max": d_max,
               "pos_error": (p_sq / nw) ** 0.5} if nw else None
    return seq, windows, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--csv", default=None, help="flight log; default: a synthetic flight")
    ap.add_argument("--minutes", type=float, default=5.0, help="synthetic flight length")
    ap.add_argument("--vel_tol", type=float, default=1e-3,
                    help="max |onnx - torch| per-frame body velocity, m/s")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="onnx_check_")
    csv = a.csv or synthetic_flight(os.path.join(tmp, "synthetic_sensor_data.csv"), a.minutes)
    print("[check] flight: %s" % csv)

    # ---- A: the ONNX script ---------------------------------------------------
    model = oi.OnnxVelNet(a.onnx)
    F = model.frames
    flight = oi.load_flight(csv, model.use_airspeed)
    vel, _ = oi.predict(model, flight)
    m_onnx, (gt_onnx, _, _) = oi.score(flight, vel, F)

    # ---- B: PyTorch through the training pipeline -------------------------------
    conf = ConfigFactory.parse_file(a.config)
    conf.train.device = "cpu"
    net = net_dict[conf.train.network](conf.train)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(ck.get("model_state_dict", ck))
    net = net.float().eval()
    seq, windows, m_torch = torch_reference(conf, net, csv, F)

    ok = True

    def report(name, diff, tol):
        nonlocal ok
        good = diff <= tol
        ok &= good
        print("[check] %-44s max|diff| %.3g  %s" % (name, diff, "OK" if good else "FAIL"))

    n = len(flight["time"])
    # SeqeuncesDataset keeps frames 0 .. N-2 ("abandon the last imu features").
    m = len(seq["acc"])
    print("[check] frames after cleaning/trim: onnx script %d, training dataset %d (+1 "
          "dropped last frame)" % (n, m))
    ok &= n == m + 1
    report("acc  (m/s^2) vs training loader", float(np.abs(flight["acc"][:m] - seq["acc"]).max()), 1e-4)
    report("gyro (rad/s) vs training loader", float(np.abs(flight["gyro"][:m] - seq["gyro"]).max()), 1e-6)

    dv = dg = 0.0
    for s, vb_t, gt_t in windows:
        dv = max(dv, float(np.abs(vel[s:s + F] - vb_t).max()))       # frames s+1..s+F
        dg = max(dg, float(np.abs(gt_onnx[s:s + F] - gt_t).max()))
    print("[check] windows compared: %d of %d frames" % (len(windows), F))
    report("vel_body (m/s) vs PyTorch model", dv, a.vel_tol)
    report("ground-truth body velocity (m/s)", dg, 1e-3)

    if m_onnx and m_torch:
        print("[check] %-10s %12s %12s %10s" % ("metric", "onnx script", "pytorch", "diff"))
        # pos_error: the eval tool rotates the LAST frame of each window with the
        # attitude one sample early (VelocityNet._world_rot without a label); the
        # script uses the label-aligned attitude.  Allow a few cm for that.
        tol = {"vel_rmse": 1e-3, "vel_max": 5e-2, "dir_rmse": 1e-2, "dir_max": 0.5,
               "pos_error": 0.05}
        for k in ("vel_rmse", "vel_max", "dir_rmse", "dir_max", "pos_error"):
            d = abs(m_onnx[k] - m_torch[k])
            good = d <= tol[k]
            ok &= good
            print("[check] %-10s %12.4f %12.4f %10.2g  %s"
                  % (k, m_onnx[k], m_torch[k], d, "OK" if good else "FAIL"))
    print("[check] %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
