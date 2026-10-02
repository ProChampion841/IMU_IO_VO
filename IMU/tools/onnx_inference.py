"""Run an exported ONNX model (tools/export_onnx.py) -- on flight logs or on arrays.

TWO MODES
---------
1. FLIGHT LOGS  (--config + --splits / --csv)
   Reads the *_sensor_data.csv flights with the SAME loader, 15 s bias freeze and
   padding9 collate as training, runs the ONNX network (ONNX Runtime, no PyTorch
   model), integrates corrected and raw IMU from the ground-truth initial state, and
   reports per horizon -- the same numbers as tools/eval_vel_horizons.py:

       vel_rmse  vel_max_error  dir_rmse  dir_max_error  pos_error   (model, raw, ratio)

   The window length is READ FROM THE ONNX FILE (a 40 s export -> 40 s windows), and
   every --horizons value must fit inside it (each is read off a prefix of the same
   window, like `eval_vel_horizons --nested`).  For 3 / 4 / 5 min, export a longer
   model:  python -m tools.export_onnx ... --frames 30000

2. ARRAYS  (--npz)
   Pure ONNX Runtime + NumPy: an .npz with acc, gyro [, g_body] [, airspeed] of shape
   (N, 3) or (B, N, 3), N = frames + 9, in -> an .npz with the model outputs.  No
   dataset code, no PyTorch.  This is the call a deployment runs.

INTEGRATION.  `integrate_np` is a NumPy re-implementation of pypose's
IMUPreintegrator exactly as the model uses it (gtrot: gravity removed with the
attitude named by the config's rot_source -- GPSNavEul, or the MTi for a GPS-free
model; the specific force rotated into the world by the gyro-integrated attitude
from the window start).  It is checked against pypose in
tests/test_onnx_inference.py.

Examples (from the IMU folder):
    python -m tools.onnx_inference --onnx tilt_rotate_40s.onnx ^
        --config configs/exp/UAV/tilt_rotate.conf --splits inference ^
        --horizons 3000 4000 --out_csv onnx_windows.csv

    python -m tools.onnx_inference --onnx tilt_rotate_40s.onnx --npz window.npz ^
        --out_npz window_out.npz
"""
import argparse
import copy
import csv
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np

INTERVAL = 9            # history samples in front of every window (padding9)


# ---------------------------------------------------------------------------
# ONNX session
# ---------------------------------------------------------------------------
class OnnxModel:
    """Thin wrapper: reads the fixed window length and the input names off the file."""

    def __init__(self, path, threads=0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self.outputs = [o.name for o in self.sess.get_outputs()]
        n = self.sess.get_inputs()[0].shape[1]
        if not isinstance(n, int):
            raise ValueError("%s has a symbolic time axis; export it with tools/export_onnx.py"
                             % path)
        self.n_in = n
        self.frames = n - INTERVAL

    def __call__(self, **feeds):
        missing = [k for k in self.inputs if k not in feeds]
        if missing:
            raise KeyError("the ONNX model needs inputs %s, missing %s" % (self.inputs, missing))
        for k in self.inputs:
            if feeds[k].shape[1] != self.n_in:
                raise ValueError("input %s has %d samples, the model takes exactly %d "
                                 "(%d frames + %d history)"
                                 % (k, feeds[k].shape[1], self.n_in, self.frames, INTERVAL))
        out = self.sess.run(None, {k: np.ascontiguousarray(feeds[k], np.float32)
                                   for k in self.inputs})
        return dict(zip(self.outputs, out))


# ---------------------------------------------------------------------------
# NumPy strapdown integration, identical to pypose's IMUPreintegrator as used here
# ---------------------------------------------------------------------------
def so3_exp(phi):
    """(..., 3) rotation vectors -> (..., 3, 3) matrices (Rodrigues, exact)."""
    th = np.linalg.norm(phi, axis=-1)[..., None, None]
    K = np.zeros(phi.shape[:-1] + (3, 3), dtype=phi.dtype)
    K[..., 0, 1], K[..., 0, 2] = -phi[..., 2], phi[..., 1]
    K[..., 1, 0], K[..., 1, 2] = phi[..., 2], -phi[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -phi[..., 1], phi[..., 0]
    small = th < 1e-8
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1.0 - th ** 2 / 6.0, np.sin(ths) / ths)
    b = np.where(small, 0.5 - th ** 2 / 24.0, (1.0 - np.cos(ths)) / ths ** 2)
    return np.eye(3, dtype=phi.dtype) + a * K + b * (K @ K)


def integrate_np(acc, gyro, dt, init_pos, init_vel, init_R, rot_R=None, gravity=9.81007):
    """Dead-reckon one window.  float64 throughout.

    acc, gyro (B, F, 3)  body FLU, m/s^2 and rad/s     dt (B, F, 1) s
    init_pos, init_vel (B, 3) world NWU                 init_R (B, 3, 3) body->world
    rot_R (B, F, 3, 3) or None  -- the attitude used to REMOVE GRAVITY (gtrot: True).
                                   None: gravity is removed with the integrated attitude.
    Returns pos, vel (B, F, 3) and rot (B, F, 3, 3): the state AFTER each step, the
    same indexing as pypose / label['gt_*'].
    """
    acc, gyro, dt = (np.asarray(x, np.float64) for x in (acc, gyro, dt))
    B, F = acc.shape[:2]
    g = np.array([0.0, 0.0, float(gravity)])
    dR = so3_exp(gyro * dt)                                     # (B, F, 3, 3)
    incre = np.broadcast_to(np.eye(3), (B, 3, 3)).copy()
    Dv, Dp, Dt = np.zeros((B, 3)), np.zeros((B, 3)), np.zeros((B, 1))
    pos, vel, rot = np.empty((B, F, 3)), np.empty((B, F, 3)), np.empty((B, F, 3, 3))
    R0, v0, p0 = (np.asarray(x, np.float64) for x in (init_R, init_vel, init_pos))
    for k in range(F):
        nxt = incre @ dR[:, k]
        if rot_R is not None:
            a = acc[:, k] - np.einsum("bji,j->bi", rot_R[:, k], g)
        else:
            a = acc[:, k] - np.einsum("bji,j->bi", R0 @ nxt, g)
        ra = np.einsum("bij,bj->bi", incre, a)
        d = dt[:, k]
        Dp = Dp + Dv * d + 0.5 * ra * d * d
        Dv = Dv + ra * d
        Dt = Dt + d
        incre = nxt
        vel[:, k] = v0 + np.einsum("bij,bj->bi", R0, Dv)
        pos[:, k] = p0 + np.einsum("bij,bj->bi", R0, Dp) + v0 * Dt
        rot[:, k] = R0 @ incre
    return pos, vel, rot


# ---------------------------------------------------------------------------
# metrics -- same definitions as tools/eval_vel_horizons.py
# ---------------------------------------------------------------------------
def endpoint_index(h, sampling):
    h, s = int(h), int(sampling) if sampling else 0
    idx = (h // s) * s - 1 if s else h - 1
    if idx < 0:
        raise ValueError("horizon %d is shorter than one sampling interval %d" % (h, s))
    return idx


def dir_deg(p, g):
    cos = (p * g).sum(-1) / (np.linalg.norm(p, axis=-1).clip(1e-9)
                             * np.linalg.norm(g, axis=-1).clip(1e-9))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def window_metrics(pos, vel, gt_pos, gt_vel, idx):
    """Per window, at output index idx: the five reported quantities."""
    ve = np.linalg.norm(vel - gt_vel, axis=-1)                  # (B, F)
    de = dir_deg(vel, gt_vel)
    pe = np.linalg.norm(pos - gt_pos, axis=-1)
    return {"vel": ve[:, idx], "vel_peak": ve[:, :idx + 1].max(1),
            "dir": de[:, idx], "dir_peak": de[:, :idx + 1].max(1),
            "pos": pe[:, idx]}


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


def horizon_label(h, sampling=0):
    """The time a horizon is actually MEASURED at: errors are recorded every
    `sampling` frames, so 3991 frames is read at frame 3950 = 39.5 s."""
    return "%gs" % ((endpoint_index(h, sampling) + 1) / 100.0)


def print_table(split, table, sampling=0):
    print("\n=== ONNX SUMMARY -- model (raw in brackets; ratio = model/raw, <1 is better) ===")
    print("%-9s %-7s %6s | %s" % ("split", "horizon", "wins",
                                 " | ".join("%-22s" % n for n, _, _ in SUMMARY)))
    print("-" * (26 + 25 * len(SUMMARY)))
    for h, n, m, r in table:
        cells = ["%8.3f (%8.3f) %4.2f" % (m[k], r[k], m[k] / max(r[k], 1e-12))
                 for k, _, _ in SUMMARY]
        print("%-9s %-7s %6d | %s" % (split, horizon_label(h, sampling), n,
                                     " | ".join("%-22s" % c for c in cells)))
    print("units: vel m/s, dir deg, pos m.  *_rmse / pos_error are AT the horizon (pos_error")
    print("is the mean); *_max_error is the worst frame anywhere inside [0, T].")


# ---------------------------------------------------------------------------
# mode 1: flight logs
# ---------------------------------------------------------------------------
def build_feeds(model, data, att_source):
    """Network inputs exactly as HybridNet._net_input would assemble them."""
    from model.attitude import gravity_direction, pad_rotation, select_attitude
    feeds = {"acc": data["acc"].numpy(), "gyro": data["gyro"].numpy()}
    if "g_body" in model.inputs:
        rot, _ = select_attitude(data, source=att_source)
        pad = data["acc"].shape[1] - rot.lshape[1]
        if pad > 0:
            rot = pad_rotation(rot, rot[:, :1], pad)
        feeds["g_body"] = gravity_direction(rot).numpy()
    if "airspeed" in model.inputs:
        feeds["airspeed"] = data["airspeed"].numpy()
    return feeds


def run_logs(a, model):
    import torch
    import torch.utils.data as Data
    from pyhocon import ConfigFactory

    from datasets import SeqeuncesDataset, collate_fcs
    from utils import pypose_compat
    pypose_compat.apply()

    conf = ConfigFactory.parse_file(a.config)
    tc = conf.train
    W = model.frames
    horizons = sorted(a.horizons) if a.horizons else [W]
    too_long = [h for h in horizons if h > W]
    if too_long:
        sys.exit("horizons %s exceed the ONNX window (%d frames = %g s).  Export a "
                 "longer model: python -m tools.export_onnx ... --frames %d"
                 % (too_long, W, W / 100.0, max(too_long)))
    sampling = tc.get("sampling", 0)
    idx_of = {h: endpoint_index(h, sampling) for h in horizons}
    for h in horizons:
        if idx_of[h] + 1 != h:
            print("[note] horizon %d frames is measured at frame %d (%s): errors are "
                  "recorded every sampling = %d frames" % (h, idx_of[h] + 1,
                                                           horizon_label(h, sampling), sampling))
    collate = collate_fcs[conf.dataset.get("collate", "base")]
    if conf.dataset.get("collate", "base") != "padding9":
        sys.exit("this ONNX pipeline expects the padding9 collate (9 history samples)")
    att_source = str(tc.get("att_source", "gt"))
    gravity = float(tc.get("gravity", 9.81007))
    gtrot = bool(tc.get("gtrot", False))
    # the attitude that removes gravity -- the same choice model/net.py makes
    rot_key = "mti_rot" if str(tc.get("rot_source", "gt")) == "mti" else "rot"
    print("[attitude] network input: %s | gravity removal: %s"
          % ("MTi (no GPS)" if att_source == "mti" else "GPSNavEul (GPS-aided)",
             ("MTi (no GPS)" if rot_key == "mti_rot" else "GPSNavEul (GPS-aided)")
             if gtrot else "integrated gyro"))

    ref_net = None
    if a.compare_ckpt:
        from model import net_dict
        tc.put("device", "cpu")
        ref_net = net_dict[tc.network](tc).float().eval()
        ck = torch.load(a.compare_ckpt, map_location="cpu", weights_only=False)
        ref_net.load_state_dict(ck.get("model_state_dict", ck))
        print("[compare] PyTorch reference: %s" % a.compare_ckpt)

    all_rows, win_rows, npz = {}, [], {}
    for split in a.splits:
        dc = copy.deepcopy(conf.dataset[split])
        for e in dc.data_list:
            e["window_size"], e["step_size"] = W, W
            if a.csv:
                e["data_drive"] = [os.path.basename(a.csv)]
                if os.path.dirname(a.csv):
                    e["data_root"] = os.path.dirname(a.csv)
            if a.data_root:
                e["data_root"] = a.data_root
            if a.max_flights:
                e["data_drive"] = list(e["data_drive"])[:a.max_flights]
        names = [f for e in dc.data_list for f in e["data_drive"]]
        ds = SeqeuncesDataset(data_set_config=dc)
        if a.first_only:
            seen, keep = set(), []
            for e in ds.index_map:
                if e[0] not in seen:
                    seen.add(e[0]); keep.append(e)
            ds.index_map = keep
        imap = list(ds.index_map)
        if not imap:
            print("[%s] no %g s window survives (flights too short, or gaps)" % (split, W / 100.0))
            continue
        loader = Data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate)
        n, rows, t_net, worst_diff = 0, [], 0.0, 0.0
        with torch.no_grad():
            for data, init, label in loader:
                B = data["acc"].shape[0]
                feeds = build_feeds(model, data, att_source)
                t0 = time.time()
                out = model(**feeds)
                t_net += time.time() - t0
                if ref_net is not None:
                    ref = ref_net.inference({k: (v.float() if torch.is_tensor(v) else v)
                                             for k, v in data.items()})
                    worst_diff = max(worst_diff, float(np.abs(
                        ref["corrected_acc"].numpy() - out["corrected_acc"]).max()))
                dt = data["dt"].numpy()
                rot_R = data[rot_key].matrix().double().numpy() if gtrot else None
                R0 = init["rot"].matrix().double().numpy()[:, 0]
                p0 = init["pos"].double().numpy()[:, 0]
                v0 = init["vel"].double().numpy()[:, 0]
                gp, gv = label["gt_pos"].double().numpy(), label["gt_vel"].double().numpy()
                arms = {"model": (out["corrected_acc"], out["corrected_gyro"]),
                        "raw": (feeds["acc"][:, INTERVAL:], feeds["gyro"][:, INTERVAL:])}
                traj = {}
                for arm, (ac, gy) in arms.items():
                    traj[arm] = integrate_np(ac, gy, dt, p0, v0, R0, rot_R, gravity)
                for k in range(B):
                    seq_id, start, _ = imap[n + k]
                    for h in horizons:
                        r = {"split": split, "flight": names[seq_id], "start": int(start),
                             "horizon": int(h)}
                        for arm in ("model", "raw"):
                            pos, vel, _ = traj[arm]
                            m = window_metrics(pos[k:k + 1], vel[k:k + 1], gp[k:k + 1],
                                               gv[k:k + 1], idx_of[h])
                            r.update({"%s_%s" % (arm, q): float(v[0]) for q, v in m.items()})
                        rows.append(r)
                if a.out_npz:
                    for key, val in (("corrected_acc", out["corrected_acc"]),
                                     ("model_vel", traj["model"][1]), ("model_pos", traj["model"][0]),
                                     ("raw_vel", traj["raw"][1]), ("raw_pos", traj["raw"][0]),
                                     ("gt_vel", gv), ("gt_pos", gp)):
                        npz.setdefault("%s/%s" % (split, key), []).append(val.astype(np.float32))
                    npz.setdefault("%s/start" % split, []).extend(imap[i][1] for i in range(n, n + B))
                    npz.setdefault("%s/flight" % split, []).extend(names[imap[i][0]]
                                                                   for i in range(n, n + B))
                n += B
        table = []
        for h in horizons:
            hr = [r for r in rows if r["horizon"] == h]
            table.append((h, len(hr), reduce(hr, "model"), reduce(hr, "raw")))
        print("\n[%s] %d windows of %g s from %d flights | ONNX Runtime %.1f ms per window"
              % (split, n, W / 100.0, len(set(e[0] for e in imap)), 1e3 * t_net / max(n, 1)))
        if ref_net is not None:
            print("[compare] max |corrected_acc ONNX - PyTorch| = %.2e m/s^2 -> %s"
                  % (worst_diff, "PASS" if worst_diff < 1e-3 else "FAIL"))
        print_table(split, table, sampling)
        all_rows[split] = table
        win_rows += rows

    if a.out_csv and win_rows:
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(win_rows[0].keys()))
            w.writeheader()
            w.writerows(win_rows)
        print("\nper-window results -> %s (%d rows)" % (a.out_csv, len(win_rows)))
    if a.out_npz and npz:
        np.savez_compressed(a.out_npz, **{k: (np.concatenate(v) if isinstance(v[0], np.ndarray)
                                              else np.array(v)) for k, v in npz.items()})
        print("trajectories -> %s" % a.out_npz)
    return all_rows, win_rows


# ---------------------------------------------------------------------------
# mode 2: arrays
# ---------------------------------------------------------------------------
def run_npz(a, model):
    z = np.load(a.npz)
    feeds, single = {}, False
    for k in model.inputs:
        if k not in z:
            sys.exit("%s has no '%s' (the model needs %s)" % (a.npz, k, model.inputs))
        v = np.asarray(z[k], np.float32)
        if v.ndim == 2:
            v, single = v[None], True
        feeds[k] = v
    out = model(**feeds)
    if single:
        out = {k: v[0] for k, v in out.items()}
    np.savez(a.out_npz or "onnx_out.npz", **out)
    for k, v in out.items():
        print("  %-15s %s" % (k, v.shape))
    print("-> %s" % (a.out_npz or "onnx_out.npz"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", required=True, help="file written by tools/export_onnx.py")
    ap.add_argument("--npz", default=None, help="ARRAY mode: acc, gyro[, g_body][, airspeed]")
    ap.add_argument("--config", default=None, help="LOG mode: the training config")
    ap.add_argument("--splits", nargs="+", default=["inference"],
                    choices=["train", "test", "eval", "inference"])
    ap.add_argument("--csv", default=None, help="one flight file instead of a whole split")
    ap.add_argument("--data_root", default=None, help="override data_root of the config")
    ap.add_argument("--horizons", type=int, nargs="+", default=None,
                    help="frames at 100 Hz, each <= the ONNX window (default: the window)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--first_only", action="store_true", help="first window of each flight only")
    ap.add_argument("--max_flights", type=int, default=None)
    ap.add_argument("--compare_ckpt", default=None,
                    help="also run this PyTorch checkpoint and report the ONNX difference")
    ap.add_argument("--out_csv", default=None, help="one row per (window, horizon)")
    ap.add_argument("--out_npz", default=None, help="save corrected IMU + trajectories")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    model = OnnxModel(a.onnx, a.threads)
    print("[onnx] %s | window %d frames (%g s) + %d history | inputs %s | outputs %s"
          % (a.onnx, model.frames, model.frames / 100.0, INTERVAL, model.inputs, model.outputs))
    if a.npz:
        return run_npz(a, model)
    if not a.config:
        sys.exit("give --npz (array mode) or --config (flight-log mode)")
    return run_logs(a, model)


if __name__ == "__main__":
    main()
