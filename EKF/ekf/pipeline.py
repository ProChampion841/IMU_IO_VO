"""Windows in, three trajectories out: IMU-only, VO-only, IMU+VO EKF.

DATA COMES FROM THE IMU PROJECT, UNCHANGED.  Flights are read with IMU/datasets
(UAV loader: FRD/NED -> FLU/NWU, airborne trim, gap mask) and the 15 s pre-window
bias freeze is applied exactly as in training.  Each window starts from the
ground-truth state (the IMU project's gtinit), so the numbers are directly
comparable with IMU/tools/eval_vel_horizons.py.

THE THREE ARMS share the same windows, initial state and attitude aid:
    imu   ESKF with IMU propagation + attitude aid only (no VO)
    vo    VO velocity rotated to the world with the aid attitude, integrated
    ekf   ESKF with IMU propagation + attitude aid + VO body-velocity updates
"""
import copy
import os
import sys

import numpy as np

from .eskf import ESKF, ESKFParams
from . import so3

IMU_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "IMU"))
if IMU_ROOT not in sys.path:
    sys.path.insert(0, IMU_ROOT)

INTERVAL = 9                      # padding9: history samples in front of an ONNX block
G_PAD = 9.81007                   # the constant padding_collate uses


# ---------------------------------------------------------------------------
# windows from the IMU project's dataset
# ---------------------------------------------------------------------------
def load_windows(imu_config, split, window, csv=None, data_root=None, max_flights=None,
                 first_only=False):
    """Yield one dict of NumPy arrays per window (W IMU samples, W+1 states)."""
    from pyhocon import ConfigFactory
    from datasets import SeqeuncesDataset
    from utils import pypose_compat
    pypose_compat.apply()

    conf = ConfigFactory.parse_file(imu_config)
    dc = copy.deepcopy(conf.dataset[split])
    for e in dc.data_list:
        e["window_size"], e["step_size"] = int(window), int(window)
        if csv:
            e["data_drive"] = [os.path.basename(csv)]
            if os.path.dirname(csv):
                e["data_root"] = os.path.dirname(csv)
        if data_root:
            e["data_root"] = data_root
        if max_flights:
            e["data_drive"] = list(e["data_drive"])[:int(max_flights)]
    names = [f for e in dc.data_list for f in e["data_drive"]]
    ds = SeqeuncesDataset(data_set_config=dc)
    imap = list(ds.index_map)
    if first_only:
        seen, keep = set(), []
        for e in imap:
            if e[0] not in seen:
                seen.add(e[0]); keep.append(e)
        imap = keep
    for seq, s, e in imap:
        acc = ds.acc[seq].double().numpy()
        gyro = ds.gyro[seq].double().numpy()
        if ds.freeze_hist_s > 0:
            ba, bg = ds.freeze_b[(seq, s)]
            acc = acc - ba.double().numpy()
            gyro = gyro - bg.double().numpy()
        lo = max(0, s - INTERVAL)
        yield {
            "flight": names[seq], "start": int(s),
            "acc": acc[s:e], "gyro": gyro[s:e],
            "acc_hist": acc[lo:s], "gyro_hist": gyro[lo:s],
            "dt": ds.dt[seq][s:e, 0].double().numpy(),
            "t": ds.ts[seq][s:e + 1].double().numpy(),
            "R_gt": ds.gt_ori[seq][lo:e + 1].matrix().double().numpy(),
            "R_mti": ds.mti_ori[seq][lo:e + 1].matrix().double().numpy(),
            "hist": s - lo,
            "v_gt": ds.gt_velo[seq][s:e + 1].double().numpy(),
            "p_gt": ds.gt_pos[seq][s:e + 1].double().numpy(),
            "airspeed": ds.airspeed[seq][lo:e].double().numpy(),
        }


def rot_source(win, source):
    """(W+1, 3, 3) attitude for frames start..end from 'gt' (GPSNavEul) or 'mti'."""
    R = win["R_gt"] if source == "gt" else win["R_mti"]
    return R[win["hist"]:]


# ---------------------------------------------------------------------------
# optional learned IMU correction (IMU/tools/export_onnx.py model)
# ---------------------------------------------------------------------------
def onnx_correct(win, model, att_source="gt"):
    """Corrected acc/gyro for the whole window, running the fixed-length ONNX model
    block by block.  Block 0 gets the padding9 history training used; later blocks
    get the REAL 9 preceding samples.  A tail shorter than the model is covered by
    one last block ending at the window end."""
    W, n = len(win["acc"]), model.frames
    if W < n:
        raise ValueError("window (%d) shorter than the ONNX model (%d frames)" % (W, n))
    h = win["hist"]
    R_all = win["R_gt"] if att_source == "gt" else win["R_mti"]         # frames lo..end
    g_all = R_all[:, 2, :]                                              # R^T e_z per frame
    acc_all = np.concatenate([win["acc_hist"], win["acc"]])
    gyro_all = np.concatenate([win["gyro_hist"], win["gyro"]])
    starts = list(range(0, W - n + 1, n))
    if starts[-1] + n < W:
        starts.append(W - n)
    feeds = {k: [] for k in model.inputs}
    for j in starts:
        a0, g0 = j + h, j + h                     # index of frame j in the *_all arrays
        if j == 0:
            R0 = win["R_gt"][h]                   # padding_collate pads with init_rot (GT)
            pad_acc = np.tile(R0.T @ np.array([0.0, 0.0, G_PAD]), (INTERVAL, 1))
            acc_b = np.concatenate([pad_acc, acc_all[a0:a0 + n]])
            gyro_b = np.concatenate([np.zeros((INTERVAL, 3)), gyro_all[a0:a0 + n]])
            gb = np.concatenate([np.tile(g_all[g0], (INTERVAL, 1)), g_all[g0:g0 + n]])
            air = win["airspeed"][a0:a0 + n]
            air = np.concatenate([np.tile(air[:1], (INTERVAL, 1)), air])
        else:
            acc_b = acc_all[a0 - INTERVAL:a0 + n]
            gyro_b = gyro_all[a0 - INTERVAL:a0 + n]
            gb = g_all[g0 - INTERVAL:g0 + n]
            air = win["airspeed"][a0 - INTERVAL:a0 + n]
        blk = {"acc": acc_b, "gyro": gyro_b, "g_body": gb, "airspeed": air}
        for k in model.inputs:
            feeds[k].append(blk[k])
    out = model(**{k: np.stack(v).astype(np.float32) for k, v in feeds.items()})
    acc_c, gyro_c = np.empty((W, 3)), np.empty((W, 3))
    for b, j in enumerate(starts):
        acc_c[j:j + n] = out["corrected_acc"][b]           # a tail block overwrites
        gyro_c[j:j + n] = out["corrected_gyro"][b]         # the overlap: later context
    return acc_c, gyro_c


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------
def run_eskf(win, acc, gyro, params, aid, vo=None, lever=None):
    """Filter one window.  Returns pos, vel (W, 3) = state after each IMU step, plus
    the filter (for its stats and final state)."""
    W = len(acc)
    R_aid = rot_source(win, aid.get("source", "gt")) if aid.get("source") else None
    f = ESKF(win["p_gt"][0], win["v_gt"][0], win["R_gt"][win["hist"]], params=params)
    t = win["t"]
    vo_idx = None
    if vo is not None and len(vo):
        # VO sample -> the first IMU state at or after its timestamp
        vo_idx = np.searchsorted(t[1:], vo.t, side="left")
    every = int(aid.get("every", 10))
    pos, vel = np.empty((W, 3)), np.empty((W, 3))
    j = 0
    for k in range(W):
        f.predict(acc[k], gyro[k], win["dt"][k])
        if vo_idx is not None:
            while j < len(vo_idx) and vo_idx[j] <= k:
                if vo_idx[j] == k:
                    f.update_body_velocity(vo.v[j], vo.var[j], lever)
                j += 1
        if R_aid is not None and (k + 1) % every == 0:
            f.update_attitude(R_aid[k + 1], aid.get("std_tilt_deg"), aid.get("std_yaw_deg"))
        pos[k], vel[k] = f.p, f.v
    return pos, vel, f


def run_vo_dr(win, vo, aid_source="gt"):
    """VO-only dead reckoning: hold the latest VO body velocity, rotate it to the
    world with the aid attitude, integrate.  Until the first VO sample arrives the
    initial (ground-truth) velocity is held."""
    R = rot_source(win, aid_source or "gt")
    W = len(win["dt"])
    t = win["t"]
    vo_idx = np.searchsorted(t[1:], vo.t, side="left") if len(vo) else np.array([], int)
    p = win["p_gt"][0].copy()
    vb = R[0].T @ win["v_gt"][0]
    pos, vel = np.empty((W, 3)), np.empty((W, 3))
    j = 0
    for k in range(W):
        while j < len(vo_idx) and vo_idx[j] <= k:
            vb = vo.v[j]
            j += 1
        vw = R[k + 1] @ vb
        p = p + vw * win["dt"][k]
        pos[k], vel[k] = p, vw
    return pos, vel


# ---------------------------------------------------------------------------
# metrics -- the definitions of IMU/tools/onnx_inference.py / eval_vel_horizons.py
# ---------------------------------------------------------------------------
def dir_deg(p, g):
    cos = (p * g).sum(-1) / (np.linalg.norm(p, axis=-1).clip(1e-9)
                             * np.linalg.norm(g, axis=-1).clip(1e-9))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def metrics_at(pos, vel, win, h):
    """Error AT horizon h (frames) and peak inside [0, h]."""
    i = int(h) - 1
    gp, gv = win["p_gt"][1:], win["v_gt"][1:]
    ve = np.linalg.norm(vel - gv, axis=-1)
    de = dir_deg(vel, gv)
    pe = np.linalg.norm(pos - gp, axis=-1)
    return {"vel": ve[i], "vel_peak": ve[:i + 1].max(), "dir": de[i],
            "dir_peak": de[:i + 1].max(), "pos": pe[i]}


SUMMARY = (("vel_rmse", "vel", "rms"), ("vel_max_error", "vel_peak", "max"),
           ("dir_rmse", "dir", "rms"), ("dir_max_error", "dir_peak", "max"),
           ("pos_error", "pos", "mean"))


def reduce(rows, arm):
    out = {}
    for name, key, how in SUMMARY:
        v = np.array([r["%s_%s" % (arm, key)] for r in rows])
        out[name] = (float(np.sqrt((v ** 2).mean())) if how == "rms"
                     else float(v.max()) if how == "max" else float(v.mean()))
    return out
