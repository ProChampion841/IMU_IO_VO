"""velnet error per horizon: velocity, direction and position.

WHAT IS MEASURED
----------------
For every split and horizon T, each flight is cut into back-to-back windows of
length T (window = step = T, so no two windows overlap).  Every window starts from
the ground-truth state at its first frame and the model runs over the whole window.

Five numbers per (split, horizon), all against ground truth:

    vel_rmse    sqrt(mean ||v_pred - v_gt||^2) over EVERY frame in [0, T] of every
                window                                                  m/s
    vel_max     largest ||v_pred - v_gt|| at any frame of any window     m/s
    dir_rmse    same as vel_rmse, for the angle between the predicted and the true
                velocity VECTOR                                          deg
    dir_max     largest such angle at any frame of any window            deg
    pos_error   sqrt(mean ||p_pred(T) - p_gt(T)||^2) over windows -- the drift at
                the END of the window                                    m

Velocity and direction are pooled over every frame because they are instantaneous:
the value at one frame samples whatever the aircraft was doing at that moment.
Position accumulates, so its end-of-window value is the drift that matters.

p_pred is the model's world-frame velocity integrated from the ground-truth position
at the window start with the trapezoid rule -- exactly how VelocityNet.forward builds
`pos`.  p_gt is built the same way from the GPS velocity (datasets/UAVdataset.py);
these logs carry no measured position.  So pos_error is the velocity error summed over
time, not an independent check on it.

The world-frame velocity uses the attitude the model is configured with
(`vel_frame_source`, default `gt` = GPS-aided).  With `gt`, heading error is not
counted and vel/dir/pos are optimistic for a GPS-denied aircraft.

--v0_offset
-----------
At the first frame of each window the true velocity is known.  The offset

    d = v_gt_world[0] - v_pred_world[0]

is added to EVERY predicted velocity in that window, and position is re-integrated
from the corrected velocity.  The first frame's velocity error is then exactly zero.
`--v0_frames N` averages d over the first N frames instead of one (default 1).

d is taken in the WORLD frame, so it stays fixed while the aircraft turns, the way
wind does.  (An earlier version took it in the body frame, where it rotates with the
aircraft like a sensor bias.)

MEASURED on velnet_v2c epoch 35, eval split, 60 s windows: 44% of the residual
variance is a constant offset, but an offset estimated from the first frame is off by
7.5 m/s against a 5.2 m/s bias (5.8 m/s with 10 s of averaging).  A correction only
helps when its estimate is better than the bias it removes, so on that checkpoint
velocity got WORSE and only short-horizon position improved.  Run with and without
the flag and compare -- do not assume.

HORIZONS
--------
Default: 30 s, 1, 2, 5, 10, 15, 20, 30, 40 min.  A flight shorter than T
supplies no window at T, so long horizons are computed on few flights or none; the
table prints the window and flight count per row.  `--check` lists which flights are
long enough without loading a checkpoint.

NOT COMPARABLE with CSVs from the previous version of this tool: there vel_rmse and
dir_rmse were the error AT the horizon; here they are pooled over every frame.
"""
import argparse
import copy
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import torch
import torch.utils.data as Data
from pyhocon import ConfigFactory

from datasets import SeqeuncesDataset, collate_fcs
from model import net_dict
from utils import move_to

RATE_HZ = 100.0
# 30 s, 1, 2, 5, 10, 15, 20, 30, 40 min at 100 Hz
DEFAULT_HORIZONS = [3000, 6000, 12000, 30000, 60000, 90000, 120000, 180000, 240000]
# Frames per batch.  Caps memory at long horizons: a 15 min window is 90,000 frames.
FRAME_BUDGET = 240000
METRICS = ("vel_rmse", "vel_max", "dir_rmse", "dir_max", "pos_error")


def horizon_tag(frames):
    s = frames / RATE_HZ
    return "%gmin" % (s / 60.0) if s >= 60.0 else "%gs" % s


# ---------------------------------------------------------------------------
# flight selection / feasibility
# ---------------------------------------------------------------------------

def flight_lengths(section_conf):
    """{flight: n_frames} by counting CSV lines -- no loader, no decode."""
    out = {}
    for entry in section_conf.data_list:
        root = entry.get("data_root", "data")
        for f in entry["data_drive"]:
            path = os.path.join(root, f)
            if not os.path.exists(path):
                out[f] = 0
                continue
            with open(path, "rb") as fh:
                out[f] = max(0, sum(1 for _ in fh) - 1)
    return out


def select_flights(dc, flights):
    """Restrict every data_list entry of `dc` (a DEEP COPY) to `flights`.

    A pattern matches if it equals the CSV name, equals its basename, or is a
    substring of it, so `--flights 178_47` works.  An unmatched pattern is a HARD
    ERROR: a typo must not silently score a different flight, or all of them.
    """
    if not flights:
        return
    unmatched = set(flights)
    for entry in dc.data_list:
        keep = []
        for f in entry["data_drive"]:
            for pat in flights:
                if f == pat or f == os.path.basename(pat) or pat in f:
                    keep.append(f)
                    unmatched.discard(pat)
                    break
        entry["data_drive"] = keep
    if unmatched:
        sys.exit("--flights matched no CSV in this split: %s" % ", ".join(sorted(unmatched)))


def check_lengths(conf, sections, horizons, flights=None):
    """Windows each flight supplies per horizon.  A flight shorter than the horizon
    supplies none, so a long row can be computed on a fraction of the split."""
    for sec in sections:
        if sec not in conf.dataset:
            print("  [%s] not in this config" % sec)
            continue
        sc = copy.deepcopy(conf.dataset[sec])
        select_flights(sc, flights)
        lens = flight_lengths(sc)
        width = 58 + 9 * len(horizons)
        print("\n=== split '%s' -- %d flights ===" % (sec, len(lens)))
        print("%-38s %9s %8s   %s" % ("flight", "frames", "minutes",
                                       " ".join("%8s" % horizon_tag(h) for h in horizons)))
        print("-" * width)
        for f in sorted(lens):
            n = lens[f]
            cells = " ".join("%8s" % (n // h if n // h else "-") for h in horizons)
            print("%-38s %9d %8.1f   %s" % (f[:38], n, n / RATE_HZ / 60.0, cells))
        print("-" * width)
        print("%-38s %9s %8s   %s" % ("windows", "", "",
                                       " ".join("%8d" % sum(v // h for v in lens.values())
                                                for h in horizons)))
        print("%-38s %9s %8s   %s" % ("flights usable", "", "",
                                       " ".join("%8s" % ("%d/%d" % (
                                           sum(1 for v in lens.values() if v // h),
                                           len(lens))) for h in horizons)))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def integrate_pos(vel_w, dt, p0):
    """World velocity -> position, identical to VelocityNet.forward."""
    v_mid = torch.cat([vel_w[:, :1], 0.5 * (vel_w[:, 1:] + vel_w[:, :-1])], dim=1)
    return p0 + torch.cumsum(v_mid * dt, dim=1)


def angle_deg(a, b):
    """Angle between two sets of 3-vectors, degrees.  (B, n, 3) x2 -> (B, n).

    atan2(|a x b|, a . b) rather than arccos of the cosine: in float32 arccos cannot
    resolve angles below ~0.03 deg (cos is within one ulp of 1), which would put a
    false floor under dir_rmse.  atan2 is accurate at every angle.
    """
    return torch.rad2deg(torch.atan2(torch.cross(a, b, dim=-1).norm(dim=-1),
                                     (a * b).sum(dim=-1)))


def window_errors(out, data, init_state, label, v0_frames=0):
    """Per-frame velocity and direction error, end-of-window position error.

    Returns (vel_err (B, n) m/s, dir_err (B, n) deg, pos_err (B,) m, offset (B, 3) or
    None).  With v0_frames > 0 the world-frame offset between ground truth and the
    prediction over the first `v0_frames` frames is added to every frame, and position
    is re-integrated from the corrected velocity.
    """
    vb, rot = out["vel_body"], out["rot"]
    n = min(vb.shape[1], rot.lshape[1], label["gt_vel"].shape[1],
            label["gt_rot"].lshape[1], label["gt_pos"].shape[1])
    gt_v, gt_p = label["gt_vel"][:, :n], label["gt_pos"][:, :n]

    if v0_frames:
        k = min(int(v0_frames), n)
        d = (gt_v[:, :k] - rot[:, :k] @ vb[:, :k]).mean(dim=1, keepdim=True)   # (B, 1, 3)
        vel_w = rot[:, :n] @ vb[:, :n] + d
        pos = integrate_pos(vel_w, data["dt"][:, :n], init_state["pos"])
        offset = d.squeeze(1)
    else:
        vel_w, pos = out["vel"][:, :n], out["pos"][:, :n]
        offset = None

    vel_err = (vel_w - gt_v).norm(dim=-1)
    dir_err = angle_deg(vel_w, gt_v)
    pos_err = (pos[:, -1] - gt_p[:, -1]).norm(dim=-1)
    return vel_err, dir_err, pos_err, offset


def run_split(network, conf, section, horizon, device, batch_size, collate_fn,
              flights=None, v0_frames=0):
    """All windows of one split at one horizon.  Returns a result dict, or None when
    no flight is long enough."""
    dc = copy.deepcopy(conf.dataset[section])
    select_flights(dc, flights)
    for entry in dc.data_list:
        entry["window_size"] = int(horizon)
        entry["step_size"] = int(horizon)
    # The network runs in float32 (as in training).  The `inference` section loads
    # float64, which crashed the first conv layer; position integration in float32 is
    # accurate to ~1 mm over 40 min, so nothing is lost.
    dc["dtype"] = "float32"
    ds = SeqeuncesDataset(data_set_config=dc)
    if len(ds) == 0:
        return None
    n_flights = len({int(e[0]) for e in ds.index_map})
    bs = max(1, min(int(batch_size), FRAME_BUDGET // int(horizon)))
    loader = Data.DataLoader(dataset=ds, batch_size=bs, shuffle=False,
                             collate_fn=collate_fn)

    # Sums of squares are square-rooted ONCE at the end; averaging per-batch RMSEs is
    # not an RMSE.
    v_sq = d_sq = p_sq = off_sum = 0.0
    v_max = d_max = 0.0
    n_frames = n_win = 0
    with torch.no_grad():
        for data, init_state, label in loader:
            data, init_state, label = move_to([data, init_state, label], device)
            out = network(data, init_state)
            ev, ed, ep, off = window_errors(out, data, init_state, label, v0_frames)
            v_sq += float(ev.pow(2).sum())
            d_sq += float(ed.pow(2).sum())
            p_sq += float(ep.pow(2).sum())
            v_max = max(v_max, float(ev.max()))
            d_max = max(d_max, float(ed.max()))
            n_frames += ev.numel()
            n_win += int(ep.shape[0])
            if off is not None:
                off_sum += float(off.norm(dim=-1).sum())

    return {"windows": n_win, "flights": n_flights,
            "vel_rmse": (v_sq / n_frames) ** 0.5, "vel_max": v_max,
            "dir_rmse": (d_sq / n_frames) ** 0.5, "dir_max": d_max,
            "pos_error": (p_sq / n_win) ** 0.5,
            "mean_offset": (off_sum / n_win) if v0_frames else None}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="experiment config, e.g. "
                    "configs/exp/UAV/velnet_v1.conf")
    ap.add_argument("--ckpt", default=None, help="checkpoint (not needed with --check)")
    ap.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS,
                    help="horizons in FRAMES at 100 Hz (default: 30s 1min 2min 5min 10min "
                         "15min 20min 30min 40min)")
    ap.add_argument("--splits", nargs="+", default=["eval"],
                    help="dataset sections to evaluate (default: eval)")
    ap.add_argument("--flights", nargs="+", default=None,
                    help="only these flights: CSV name or any unique substring")
    ap.add_argument("--v0_offset", action="store_true",
                    help="add the first-frame offset (v_gt - v_pred, world frame) to every "
                         "predicted velocity in the window, and re-integrate position")
    ap.add_argument("--v0_frames", type=int, default=1,
                    help="with --v0_offset: average the offset over this many first "
                         "frames (default 1 = the first frame only)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--check", action="store_true",
                    help="only list which flights are long enough per horizon")
    ap.add_argument("--out", default=None, help="also write the table to this CSV")
    a = ap.parse_args()
    if a.v0_frames < 1:
        sys.exit("--v0_frames must be >= 1")
    horizons = sorted(set(a.horizons))

    conf = ConfigFactory.parse_file(a.config)
    conf.train.device = a.device
    collate_fn = collate_fcs[conf.dataset.get("collate", "base")]

    if a.check:
        check_lengths(conf, a.splits, horizons, flights=a.flights)
        return
    if not a.ckpt:
        sys.exit("--ckpt is required unless --check is given")

    network = net_dict[conf.train.network](conf.train).to(a.device).float()
    ck = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    network.load_state_dict(ck.get("model_state_dict", ck))
    network.eval()
    v0 = a.v0_frames if a.v0_offset else 0
    v0_label = ("ON, first %d frame%s" % (v0, "" if v0 == 1 else "s")) if v0 else "OFF"
    print("[ckpt] %s (epoch %s) | collate %s | v0 offset %s"
          % (a.ckpt, ck.get("epoch", "?"), conf.dataset.get("collate", "base"), v0_label))

    rows = []
    for split in a.splits:
        if split not in conf.dataset:
            print("  [%s] not in this config -- skipped" % split)
            continue
        for h in horizons:
            res = run_split(network, conf, split, h, a.device, a.batch_size, collate_fn,
                            flights=a.flights, v0_frames=v0)
            row = {"split": split, "horizon": horizon_tag(h), "horizon_frames": h,
                   "v0_offset": v0_label, "windows": 0, "flights": 0}
            if res is not None:
                row.update(res)
            rows.append(row)

    head = ("%-6s %-7s %7s %7s | %8s %8s | %8s %8s | %10s"
            % ("split", "horizon", "windows", "flights",
               "vel_rmse", "vel_max", "dir_rmse", "dir_max", "pos_error"))
    units = ("%-6s %-7s %7s %7s | %8s %8s | %8s %8s | %10s"
             % ("", "", "", "", "m/s", "m/s", "deg", "deg", "m"))
    print("\n=== v0 offset %s ===" % v0_label)
    print(head)
    print(units)
    print("-" * len(head))
    for r in rows:
        if not r["windows"]:
            print("%-6s %-7s %7d %7d | no flight is long enough"
                  % (r["split"], r["horizon"], 0, 0))
            continue
        print("%-6s %-7s %7d %7d | %8.3f %8.3f | %8.3f %8.3f | %10.2f"
              % (r["split"], r["horizon"], r["windows"], r["flights"],
                 r["vel_rmse"], r["vel_max"], r["dir_rmse"], r["dir_max"], r["pos_error"]))
    print("-" * len(head))
    print("vel/dir: pooled over every frame of every window.  pos_error: RMS over")
    print("windows of the position drift at the END of the window.")
    if v0:
        offs = [r["mean_offset"] for r in rows if r["windows"]]
        if offs:
            print("v0 offset: mean |d| removed = %.3f m/s (world frame)." % (sum(offs) / len(offs)))

    if a.out:
        fields = ["split", "horizon", "horizon_frames", "v0_offset", "windows", "flights"]
        fields += list(METRICS) + ["mean_offset"]
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})
        print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
