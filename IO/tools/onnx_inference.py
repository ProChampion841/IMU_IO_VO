"""Run an exported velnet ONNX model on a flight CSV -- no PyTorch needed.

Needs only numpy, pandas and onnxruntime, so it runs on the deployment machine.
Export the model first with tools/export_onnx.py.

INPUT CSV (either format, detected from the column names)
--------------------------------------------------------
  * a raw flight log, as in the training data (datasets/UAVdataset.py):
        Time, AcclX/Y/Z (g, body FRD), GyroX/Y/Z (deg/s, body FRD)
    converted exactly like the training loader:
        acc  = 9.81007 * (AcclX, -AcclY, -AcclZ)   m/s^2, body FLU
        gyro = pi/180  * (GyroX, -GyroY, -GyroZ)   rad/s, body FLU
    plus AirSpeed if the model takes airspeed.  If the log also has the GPS
    columns (GPSNavEulX/Y/Z, GPSNavVnX/Y/Z), the prediction is scored against
    them (see METRICS) and, like the training loader, the flight is trimmed to
    its airborne part (GPS speed > 5 m/s; --no_trim to keep everything).
  * an already-converted file:
        time, acc_x, acc_y, acc_z (m/s^2, FLU), gyro_x, gyro_y, gyro_z (rad/s, FLU)
    plus `airspeed` if the model takes airspeed.  No ground truth, no metrics.

WINDOWS
-------
The ONNX graph has a FIXED window length F (set at export).  The flight is cut into
back-to-back windows of F frames, each run from a fresh state -- the same windows
tools/eval_vel_horizons.py uses at horizon F.  The last, partial piece is covered by
one extra window ending at the last frame, from which only the new frames are kept.
A flight shorter than F is end-padded with its last sample (the outputs for the
pad are dropped; the last ~12 frames can see the pad through the CNN).

As in training, the output for input frame k is the velocity at frame k+1 (the end
of that sample interval), so the output CSV covers frames 1 .. N-1.

OUTPUT CSV
----------
    frame, time, vel_x, vel_y, vel_z (m/s, body FLU), speed,
    var_x, var_y, var_z ((m/s)^2, if the model outputs vel_cov)
    and with GPS columns: gt_vel_x/y/z (body FLU), vel_err (m/s), dir_err (deg)

METRICS (only with GPS columns)
-------------------------------
Over the full windows of length F -- the same windows and the same definitions as
tools/eval_vel_horizons.py at horizon F:
    vel_rmse, vel_max    velocity error, pooled over every frame, m/s
    dir_rmse, dir_max    angle between predicted and true velocity, deg
    pos_error            RMS over windows of the end-of-window position error, m;
                         position integrated from the true start position with the
                         GPS attitude (not deployable -- same caveat as the eval tool)

USAGE
-----
    python -m tools.onnx_inference --onnx velnet_6000.onnx \\
        --csv /path/to/2026_04_07_178_47_sensor_data.csv --out pred.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

GRAVITY = 9.81007               # datasets/UAVdataset.py UAV.GRAVITY
FLIP = np.array([1.0, -1.0, -1.0])
T = np.diag(FLIP)

RAW_IMU = ["Time", "GyroX", "GyroY", "GyroZ", "AcclX", "AcclY", "AcclZ"]
RAW_NAV = ["GPSNavEulX", "GPSNavEulY", "GPSNavEulZ", "GPSNavVnX", "GPSNavVnY", "GPSNavVnZ"]
# The training loader drops a row when ANY of these is NaN; mirrored when present.
RAW_LOADER = RAW_IMU + RAW_NAV + ["GPSNavAlt", "AirSpeed"]
SI_IMU = ["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def euler_zyx(yaw, pitch, roll):
    """Intrinsic ZYX (= scipy from_euler('ZYX', [yaw, pitch, roll])) -> (N, 3, 3)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R = np.empty((len(yaw), 3, 3))
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


def load_flight(path, need_airspeed, trim=True, speed_thresh=5.0):
    """-> dict(time, acc, gyro[, airspeed][, rot, vel_w]) as float64 numpy arrays."""
    df = pd.read_csv(path)
    out = {}
    if all(c in df.columns for c in RAW_IMU):
        has_gt = all(c in df.columns for c in RAW_NAV)
        need = [c for c in RAW_LOADER if c in df.columns] if has_gt else list(RAW_IMU)
        if need_airspeed and "AirSpeed" not in need:
            need.append("AirSpeed")
        missing = [c for c in need if c not in df.columns]
        if missing:
            sys.exit("%s: missing columns %s" % (path, missing))
        df = df.dropna(subset=need)
        t = df["Time"].to_numpy(np.float64)
        keep = np.ones(len(df), dtype=bool)
        keep[1:] = np.diff(t) > 0                    # same one-pass rule as the loader
        df = df[keep]
        if has_gt and trim:
            v = df[["GPSNavVnX", "GPSNavVnY", "GPSNavVnZ"]].to_numpy(np.float64)
            air = np.linalg.norm(v, axis=1) > speed_thresh
            if air.any():
                i0, i1 = int(np.argmax(air)), int(len(air) - np.argmax(air[::-1]))
                df = df.iloc[i0:i1]
        df = df.reset_index(drop=True)
        out["time"] = df["Time"].to_numpy(np.float64)
        out["acc"] = df[["AcclX", "AcclY", "AcclZ"]].to_numpy(np.float64) * GRAVITY * FLIP
        out["gyro"] = np.deg2rad(df[["GyroX", "GyroY", "GyroZ"]].to_numpy(np.float64)) * FLIP
        if need_airspeed:
            out["airspeed"] = df[["AirSpeed"]].to_numpy(np.float64)
        if has_gt:
            R_ned_frd = euler_zyx(df["GPSNavEulZ"].to_numpy(np.float64),
                                  df["GPSNavEulY"].to_numpy(np.float64),
                                  df["GPSNavEulX"].to_numpy(np.float64))
            out["rot"] = T[None] @ R_ned_frd @ T[None]          # body FLU -> world NWU
            out["vel_w"] = df[["GPSNavVnX", "GPSNavVnY", "GPSNavVnZ"]].to_numpy(np.float64) * FLIP
    elif all(c in df.columns for c in SI_IMU):
        need = SI_IMU + (["airspeed"] if need_airspeed else [])
        missing = [c for c in need if c not in df.columns]
        if missing:
            sys.exit("%s: missing columns %s" % (path, missing))
        df = df.dropna(subset=need).reset_index(drop=True)
        out["time"] = df["time"].to_numpy(np.float64)
        out["acc"] = df[["acc_x", "acc_y", "acc_z"]].to_numpy(np.float64)
        out["gyro"] = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(np.float64)
        if need_airspeed:
            out["airspeed"] = df[["airspeed"]].to_numpy(np.float64)
    else:
        sys.exit("%s: expected raw log columns %s or converted columns %s"
                 % (path, RAW_IMU, SI_IMU))
    if len(out["time"]) < 2:
        sys.exit("%s: fewer than 2 usable rows" % path)
    return out


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

class OnnxVelNet:
    def __init__(self, path, threads=0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        ins = self.sess.get_inputs()
        self.in_names = [i.name for i in ins]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.frames = ins[0].shape[1]
        if not isinstance(self.frames, int):
            sys.exit("%s: input length is not fixed (%r); export with tools/export_onnx.py"
                     % (path, self.frames))
        self.use_airspeed = "airspeed" in self.in_names
        self.has_cov = "vel_cov" in self.out_names

    def run(self, windows):
        """windows: dict name -> (B, F, C) float32.  -> (vel (B,F,3), var or None)."""
        res = self.sess.run(None, {k: windows[k] for k in self.in_names})
        return res[0], (res[1] if self.has_cov else None)


def window_starts(n, F):
    """Starts of the back-to-back windows, plus a final one ending at frame n."""
    if n <= F:
        return [0]
    starts = list(range(0, n - F + 1, F))
    if starts[-1] + F < n:
        starts.append(n - F)
    return starts


def predict(model, flight, batch=8):
    """Per-frame prediction for frames 1 .. N-1.  -> (vel (N-1,3), var or None)."""
    n, F = len(flight["time"]), model.frames
    keys = ["acc", "gyro"] + (["airspeed"] if model.use_airspeed else [])
    arrays = {k: flight[k].astype(np.float32) for k in keys}
    if n < F:
        print("[onnx] WARNING: flight has %d frames < window %d; end-padding with the "
              "last sample" % (n, F))
        arrays = {k: np.concatenate([a, np.repeat(a[-1:], F - n, axis=0)])
                  for k, a in arrays.items()}
    starts = window_starts(max(n, F), F)

    vel = np.full((n + 1, 3), np.nan, dtype=np.float64)     # index = output frame
    var = np.full((n + 1, 3), np.nan, dtype=np.float64) if model.has_cov else None
    covered = 0                                              # last output frame written
    for b in range(0, len(starts), batch):
        chunk = starts[b:b + batch]
        win = {k: np.stack([a[s:s + F] for s in chunk]) for k, a in arrays.items()}
        v, c = model.run(win)
        for j, s in enumerate(chunk):
            # output t of a window starting at s is the velocity at frame s + 1 + t
            lo = max(covered + 1, s + 1)
            hi = min(s + F, n)                               # frames lo .. hi
            if hi < lo:
                continue
            vel[lo:hi + 1] = v[j, lo - s - 1:hi - s]
            if var is not None:
                var[lo:hi + 1] = c[j, lo - s - 1:hi - s]
            covered = hi
    # frame n does not exist; frame 0 has no prediction
    return vel[1:n], (var[1:n] if var is not None else None)


# ---------------------------------------------------------------------------
# metrics (same definitions as tools/eval_vel_horizons.py)
# ---------------------------------------------------------------------------

def angle_deg(a, b):
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                                 (a * b).sum(axis=-1)))


def score(flight, vel_body, F):
    """vel_body: (N-1, 3) for frames 1..N-1.  Returns (metrics dict, per-frame arrays)."""
    t, R, vw_gt = flight["time"], flight["rot"], flight["vel_w"]
    n = len(t)
    gt_body = np.einsum("nji,nj->ni", R[1:], vw_gt[1:])       # R^T v, frames 1..N-1
    vel_err = np.linalg.norm(vel_body - gt_body, axis=-1)
    dir_err = angle_deg(vel_body, gt_body)

    # true position = cumulative trapezoid of the GPS velocity (the loader's gt_pos)
    dt = np.diff(t)
    p_gt = np.zeros((n, 3))
    p_gt[1:] = np.cumsum(0.5 * (vw_gt[1:] + vw_gt[:-1]) * dt[:, None], axis=0)

    # the eval tool's windows: SeqeuncesDataset 'evaluate' mode at window = step = F
    starts = list(range(0, (n - 1) - F, F))
    if not starts:
        return None, (gt_body, vel_err, dir_err)
    v_sq = d_sq = p_sq = 0.0
    v_max = d_max = 0.0
    for s in starts:
        k = slice(s, s + F)                                   # rows for frames s+1..s+F
        vb = vel_body[k]
        vw = np.einsum("nij,nj->ni", R[s + 1:s + F + 1], vb)
        v_mid = np.concatenate([vw[:1], 0.5 * (vw[1:] + vw[:-1])])
        p_end = p_gt[s] + (v_mid * dt[s:s + F, None]).sum(axis=0)
        p_sq += float(np.sum((p_end - p_gt[s + F]) ** 2))
        v_sq += float(np.sum(vel_err[k] ** 2))
        d_sq += float(np.sum(dir_err[k] ** 2))
        v_max = max(v_max, float(vel_err[k].max()))
        d_max = max(d_max, float(dir_err[k].max()))
    nf = len(starts) * F
    m = {"windows": len(starts), "vel_rmse": (v_sq / nf) ** 0.5, "vel_max": v_max,
         "dir_rmse": (d_sq / nf) ** 0.5, "dir_max": d_max,
         "pos_error": (p_sq / len(starts)) ** 0.5}
    return m, (gt_body, vel_err, dir_err)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", required=True, help="model exported by tools/export_onnx.py")
    ap.add_argument("--csv", required=True, nargs="+", help="one or more flight CSVs")
    ap.add_argument("--out", default=None,
                    help="output CSV (one flight) or output DIRECTORY (several flights); "
                         "default: <csv name>_pred.csv next to the input")
    ap.add_argument("--no_trim", action="store_true",
                    help="do not trim a raw log to its airborne part (GPS speed > 5 m/s)")
    ap.add_argument("--batch", type=int, default=8, help="windows per onnxruntime call")
    ap.add_argument("--threads", type=int, default=0, help="onnxruntime threads (0 = auto)")
    a = ap.parse_args()

    model = OnnxVelNet(a.onnx, a.threads)
    print("[onnx] %s | window %d frames (%.1f s) | inputs %s | outputs %s"
          % (a.onnx, model.frames, model.frames / 100.0, model.in_names, model.out_names))
    many = len(a.csv) > 1
    if many and a.out:
        os.makedirs(a.out, exist_ok=True)

    rows = []
    for path in a.csv:
        flight = load_flight(path, model.use_airspeed, trim=not a.no_trim)
        n = len(flight["time"])
        t0 = time.time()
        vel, var = predict(model, flight, batch=a.batch)
        dt_run = time.time() - t0

        df = pd.DataFrame({"frame": np.arange(1, n), "time": flight["time"][1:],
                           "vel_x": vel[:, 0], "vel_y": vel[:, 1], "vel_z": vel[:, 2],
                           "speed": np.linalg.norm(vel, axis=1)})
        if var is not None:
            df["var_x"], df["var_y"], df["var_z"] = var[:, 0], var[:, 1], var[:, 2]
        metrics = None
        if "rot" in flight:
            metrics, (gt_body, vel_err, dir_err) = score(flight, vel, model.frames)
            df["gt_vel_x"], df["gt_vel_y"], df["gt_vel_z"] = gt_body.T
            df["vel_err"], df["dir_err"] = vel_err, dir_err

        base = os.path.splitext(os.path.basename(path))[0] + "_pred.csv"
        if a.out and many:
            out = os.path.join(a.out, base)
        elif a.out:
            out = a.out
        else:
            out = os.path.join(os.path.dirname(os.path.abspath(path)), base)
        df.to_csv(out, index=False, float_format="%.6g")

        dur = flight["time"][-1] - flight["time"][0]
        print("[onnx] %s: %d frames (%.1f min) in %.2f s -> %s"
              % (os.path.basename(path), n, dur / 60.0, dt_run, out))
        if metrics is None and "rot" in flight:
            print("        shorter than one full %d-frame window: per-frame errors "
                  "written, no window metrics" % model.frames)
        elif metrics:
            print("        windows %d | vel_rmse %.3f m/s  vel_max %.3f m/s | dir_rmse %.2f "
                  "deg  dir_max %.2f deg | pos_error %.1f m"
                  % (metrics["windows"], metrics["vel_rmse"], metrics["vel_max"],
                     metrics["dir_rmse"], metrics["dir_max"], metrics["pos_error"]))
            rows.append(dict(flight=os.path.basename(path), **metrics))

    if len(rows) > 1:
        # Pool over flights the same way the eval tool pools over windows.
        w = np.array([r["windows"] for r in rows], dtype=float)
        pooled = lambda k: float(np.sqrt((w * np.square([r[k] for r in rows])).sum() / w.sum()))
        print("[onnx] ALL %d flights | windows %d | vel_rmse %.3f  vel_max %.3f | dir_rmse "
              "%.2f  dir_max %.2f | pos_error %.1f"
              % (len(rows), int(w.sum()), pooled("vel_rmse"), max(r["vel_max"] for r in rows),
                 pooled("dir_rmse"), max(r["dir_max"] for r in rows), pooled("pos_error")))


if __name__ == "__main__":
    main()
