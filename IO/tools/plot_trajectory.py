"""Draw integrated POSITION and VELOCITY against ground truth, for any flight.

Two ways to select the data:

  --csv data/2026_04_07_219_54_sensor_data.csv --length 4000
        Point straight at a sensor CSV and choose the window length yourself.
        Works on any flight in any split, and on flights in no split at all.

  --split eval --seq 2026_04_07_219_54
        Use the flight list the config already defines.  --length still overrides
        the window length.

Both paths build the per-window sample dict with the SAME slicing as
datasets/dataset.py:246-267 and hand it to the SAME collate the training loop
uses (`padding9`), so the states plotted here are the states training optimises.

WHAT IS PLOTTED
  * top-down North/East trajectory: ground truth, model, and optionally raw
  * altitude vs time
  * |position error| vs time, model against raw
  * |velocity error| vs time, model against raw
  * the three velocity components, ground truth against model and raw

TWO MODES, and the difference matters
  --mode window     (default)  Each window is integrated from its OWN ground-truth
                    init_state, exactly as in training and exactly as metric.csv
                    measures.  The model trace is PIECEWISE: one arc per window,
                    each starting back on the truth.  Arcs are drawn separately,
                    never joined, so the reset is visible rather than smeared into
                    a line that was never integrated.
  --mode continuous One unbroken dead-reckoned trajectory: the final pos/vel/rot of
                    each window seeds the next and ground truth is used exactly
                    once, at the start.  This is what an unaided IMU produces.
                    Expect large drift -- most of it attitude, not accel bias --
                    so read it qualitatively, not as a metric.

FRAME.  The pipeline stores position and velocity in world NWU (datasets/
UAVdataset.py rotates the NED log on load).  Plots convert to the conventional
map frame: North = p[0], East = -p[1], Up = p[2].

GROUND TRUTH.  There is no position column in these logs.  `gt_pos` is the
cumulative trapezoidal integral of GPSNavVnX/Y/Z, the nav filter's velocity
(UAVdataset.py:227).  So "position error" here means disagreement with that
integrated velocity, not absolute position accuracy.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyhocon import ConfigFactory
from datasets import SeqeuncesDataset, collate_fcs
from datasets.UAVdataset import UAV
from datasets.dataset import airspeed_or_zeros, mti_or_gt
from model import net_dict
from utils import move_to


def to_neu(p):
    """Stored NWU -> plotting North/East/Up."""
    return np.stack([p[..., 0], -p[..., 1], p[..., 2]], axis=-1)


def sample_from_seq(seq, f0, f1):
    """One window, sliced exactly as datasets/dataset.py:246-267 does it.

    The +1 offsets are load-bearing: the labels are the states AFTER each IMU
    sample, so gt_* start one frame later than the inputs and init_* is the state
    at the window's first frame.
    """
    d = seq.data
    mti = mti_or_gt(seq)
    return {
        "dt": d["dt"][f0:f1],
        "acc": d["acc"][f0:f1],
        "gyro": d["gyro"][f0:f1],
        "rot": d["gt_orientation"][f0:f1],
        "mti_rot": mti[f0:f1],
        # IMU-rate network input like acc/gyro (needed by use_airspeed models)
        "airspeed": airspeed_or_zeros(seq)[f0:f1],
        "init_rot": d["gt_orientation"][f0][None, ...],
        "init_mti_rot": mti[f0][None, ...],
        "init_pos": d["gt_translation"][f0][None, ...],
        "init_vel": d["velocity"][f0][None, ...],
        "gt_pos": d["gt_translation"][f0 + 1:f1 + 1],
        "gt_rot": d["gt_orientation"][f0 + 1:f1 + 1],
        "gt_vel": d["velocity"][f0 + 1:f1 + 1],
    }


def run(network, seq, starts, W, device, collate_fn, mode, want_raw, rel_time=False):
    """Integrate every window. Returns stacked [n_win, W, 3] arrays plus a time axis."""
    net = network.module if hasattr(network, "module") else network
    iv = getattr(net, "interval", 0)
    tcum = torch.cumsum(seq.data["dt"].reshape(-1), 0).numpy()

    out = {k: [] for k in ("gt_pos", "gt_vel", "pos", "vel", "raw_pos", "raw_vel", "t")}
    carry = None

    with torch.no_grad():
        for n, f0 in enumerate(starts):
            f1 = f0 + W
            data, init_state, label = move_to(
                list(collate_fn([sample_from_seq(seq, f0, f1)])), device)

            if mode == "continuous" and carry is not None:
                init_state = dict(init_state)
                init_state["pos"], init_state["vel"], init_state["rot"] = carry

            inte = network(data, init_state)

            if want_raw:
                d = dict(data)
                d["corrected_acc"] = data["acc"][:, iv:, :]
                d["corrected_gyro"] = data["gyro"][:, iv:, :]
                raw = net.integrate(init_state=init_state, data=d,
                                    cov_state={"acc_cov": None, "gyro_cov": None})
                out["raw_pos"].append(raw["pos"][0].cpu().numpy())
                out["raw_vel"].append(raw["vel"][0].cpu().numpy())

            if mode == "continuous":
                carry = (inte["pos"][:, -1:, :], inte["vel"][:, -1:, :], inte["rot"][:, -1:])

            out["pos"].append(inte["pos"][0].cpu().numpy())
            out["vel"].append(inte["vel"][0].cpu().numpy())
            out["gt_pos"].append(label["gt_pos"][0].cpu().numpy())
            out["gt_vel"].append(label["gt_vel"][0].cpu().numpy())
            # Single window: time from the START OF THAT WINDOW, so the x axis reads
            # 0..W/100 s and lines up with tools/plot_window_error.py.  With --all the
            # arcs have to share one axis, so they use absolute flight time instead.
            out["t"].append(tcum[f0 + 1:f1 + 1] - tcum[f0 if rel_time else 0])
            print("\r  window %3d/%3d" % (n + 1, len(starts)), end="", flush=True)
    print()

    res = {k: np.stack(v) for k, v in out.items() if v}
    for k in ("gt_pos", "gt_vel", "pos", "vel", "raw_pos", "raw_vel"):
        if k in res:
            res[k] = to_neu(res[k])
    return res


def resolve_flight(args, conf):
    """Return (UAV sequence, display name) from --csv or from --split/--seq."""
    if args.csv:
        root, name = os.path.split(os.path.abspath(args.csv))
        if not os.path.isfile(args.csv):
            sys.exit("no such file: %s" % args.csv)
        return UAV(root, name, trim_to_airborne=True), name

    ds = SeqeuncesDataset(data_set_config=conf.dataset[args.split])
    names = []
    for c in ds.conf.data_list:
        names.extend(list(c.data_drive))
    want = args.seq
    if want is None:
        idx = 0
    elif want.isdigit():           # isdigit, not int(): "2026_04_07_216_47" parses as int
        idx = int(want)
        if not 0 <= idx < len(names):
            sys.exit("--seq %d out of range (%d flights)" % (idx, len(names)))
    else:
        hits = [i for i, n in enumerate(names) if want in n]
        if len(hits) != 1:
            print("--seq %r matched %d flights; available:" % (want, len(hits)))
            for i, n in enumerate(names):
                print("  [%2d] %s" % (i, n))
            sys.exit(1)
        idx = hits[0]
    root = ds.conf.data_list[0]["data_root"]
    return UAV(root, names[idx], trim_to_airborne=True), names[idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="supplies the network hyper-parameters")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default=None,
                    help="path to a sensor CSV; overrides --split/--seq entirely")
    ap.add_argument("--length", type=int, default=None,
                    help="window length in FRAMES (100 Hz), e.g. 4000 = 40 s. "
                         "Defaults to the config's eval window_size.")
    ap.add_argument("--stride", type=int, default=None,
                    help="frames between window starts; defaults to --length "
                         "(non-overlapping, tiling the flight exactly once)")
    ap.add_argument("--window", type=int, default=0,
                    help="which single window to draw, 0-based (default 0 = the first). "
                         "One window is the default because overlaying every arc of a "
                         "long flight hides the shape of any one of them.")
    ap.add_argument("--start", type=int, default=None,
                    help="draw the one window starting at this exact FRAME, instead of "
                         "selecting by --window index")
    ap.add_argument("--all", action="store_true",
                    help="draw every window tiling the flight, not just one")
    ap.add_argument("--max_windows", type=int, default=None,
                    help="with --all, cap the number of windows")
    ap.add_argument("--split", default="eval", choices=["train", "test", "eval"])
    ap.add_argument("--seq", default=None, help="flight index or filename substring")
    ap.add_argument("--mode", default="window", choices=["window", "continuous"])
    ap.add_argument("--no_raw", action="store_true", help="omit the raw-IMU arm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    conf = ConfigFactory.parse_file(args.config)
    conf.train.device = args.device
    collate_fn = (collate_fcs[conf.dataset.collate]
                  if "collate" in conf.dataset.keys() else collate_fcs["base"])

    seq, name = resolve_flight(args, conf)
    W = args.length or conf.dataset.eval.data_list[0].window_size
    stride = args.stride or W
    n = seq.data["time"].shape[0]
    if n < W + 2:
        sys.exit("%s has %d frames (%.1f s); too short for a %d-frame window"
                 % (name, n, n / 100.0, W))
    starts = list(range(0, n - W - 1, stride))
    n_avail = len(starts)
    if args.all:
        if args.max_windows:
            starts = starts[:args.max_windows]
    elif args.start is not None:
        if not 0 <= args.start <= n - W - 1:
            sys.exit("--start %d is outside [0, %d] for a %d-frame window in a %d-frame flight"
                     % (args.start, n - W - 1, W, n))
        starts = [args.start]
    else:
        if not 0 <= args.window < n_avail:
            sys.exit("--window %d out of range: this flight holds %d windows of %d frames "
                     "(0..%d). Use --all to draw them together."
                     % (args.window, n_avail, W, n_avail - 1))
        starts = [starts[args.window]]

    dtype = seq.data["acc"].dtype
    network = net_dict[conf.train.network](conf.train).to(device=args.device, dtype=dtype)
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    try:
        missing, unexpected = network.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        if "cnn.net.0.weight" in str(e):
            sys.exit(
                "CONFIG / CHECKPOINT MISMATCH on the first conv.\n"
                "  %s\n"
                "  The checkpoint was trained with a different input width than --config\n"
                "  builds.  in_dim is 6 + attitude channels: att_input none=0, gravity=3,\n"
                "  gravity_sincos=7 -> in_dim 6, 9 or 13.  This config gives in_dim=%d\n"
                "  (att_input=%s).  Pass the config the checkpoint was TRAINED with."
                % (str(e).strip().splitlines()[-1].strip(), network.in_dim,
                   conf.train.get("att_input", "gravity")))
        raise
    if missing or unexpected:
        print("[warn] state_dict mismatch: %d missing, %d unexpected"
              % (len(missing), len(unexpected)))
    network.eval()
    print("loaded %s (epoch %s)" % (args.ckpt, ck.get("epoch")))
    if len(starts) == 1:
        print("flight %s | %d frames (%.1f s) | ONE window of %d frames (%.1f s) starting "
              "at frame %d (t = %.1f s); %d windows available | mode=%s"
              % (name, n, n / 100.0, W, W / 100.0, starts[0], starts[0] / 100.0,
                 n_avail, args.mode))
    else:
        print("flight %s | %d frames (%.1f s) | %d windows of %d frames (%.1f s), "
              "stride %d | mode=%s"
              % (name, n, n / 100.0, len(starts), W, W / 100.0, stride, args.mode))

    single = len(starts) == 1
    r = run(network, seq, starts, W, args.device, collate_fn, args.mode,
            not args.no_raw, rel_time=single)
    has_raw = "raw_pos" in r
    nwin = r["gt_pos"].shape[0]

    perr = np.linalg.norm(r["pos"] - r["gt_pos"], axis=-1)          # [W, T]
    verr = np.linalg.norm(r["vel"] - r["gt_vel"], axis=-1)
    rperr = np.linalg.norm(r["raw_pos"] - r["gt_pos"], axis=-1) if has_raw else None
    rverr = np.linalg.norm(r["raw_vel"] - r["gt_vel"], axis=-1) if has_raw else None

    bar = "=" * 76
    print("\n" + bar)
    print("END-OF-WINDOW ERROR over %d windows of %.1f s" % (nwin, W / 100.0))
    print(bar)
    print("  %-14s %10s %10s %10s" % ("", "mean", "median", "max"))
    rows = [("model pos (m)", perr[:, -1]), ("model vel (m/s)", verr[:, -1])]
    if has_raw:
        rows += [("raw pos (m)", rperr[:, -1]), ("raw vel (m/s)", rverr[:, -1])]
    for lab, v in rows:
        print("  %-14s %10.3f %10.3f %10.3f" % (lab, v.mean(), np.median(v), v.max()))
    if has_raw:
        print("\n  model/raw ratio:  pos %.3f   vel %.3f   (<1 means the model helps)"
              % (perr[:, -1].mean() / rperr[:, -1].mean(),
                 verr[:, -1].mean() / rverr[:, -1].mean()))

    outpng = args.out or os.path.join(
        "result", "trajectory", "%s_%s_%ds%s.png" % (
            name.replace(".csv", ""), args.mode, W // 100,
            "" if not single else "_w%d" % (args.window if args.start is None else starts[0])))
    os.makedirs(os.path.dirname(outpng) or ".", exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    GT, MD, RW = "#111111", "#c0392b", "#2980b9"
    fig = plt.figure(figsize=(17, 11))
    gs = GridSpec(6, 3, figure=fig, width_ratios=[1.5, 1, 1], hspace=0.62, wspace=0.26)

    # ---- top-down North/East trajectory ---------------------------------
    ax = fig.add_subplot(gs[:, 0])
    for w in range(nwin):
        ax.plot(r["gt_pos"][w, :, 1], r["gt_pos"][w, :, 0], color=GT, lw=2.0,
                label="ground truth" if w == 0 else None, zorder=3)
        if has_raw:
            ax.plot(r["raw_pos"][w, :, 1], r["raw_pos"][w, :, 0], color=RW, lw=1.0, ls=":",
                    label="raw IMU" if w == 0 else None, zorder=2)
        ax.plot(r["pos"][w, :, 1], r["pos"][w, :, 0], color=MD, lw=1.2, alpha=0.9,
                label="AirIMU (model)" if w == 0 else None, zorder=4)
        ax.scatter(r["gt_pos"][w, 0, 1], r["gt_pos"][w, 0, 0], s=14, color=GT, zorder=5)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    if single:
        # Mark where each arm ENDS -- with one window the divergence is the whole
        # point of the picture, and the end points are where it is largest.
        for arr, col in ((r["gt_pos"], GT), (r["pos"], MD)) + (
                ((r["raw_pos"], RW),) if has_raw else ()):
            ax.scatter(arr[0, -1, 1], arr[0, -1, 0], s=60, marker="s",
                       facecolor="none", edgecolor=col, linewidth=1.6, zorder=6)
        _title = ("Top-down trajectory\n%s\nwindow %d of %d   |   t = %.1f - %.1f s"
                  % (name, args.window if args.start is None else 0, n_avail,
                     starts[0] / 100.0, (starts[0] + W) / 100.0))
    else:
        _title = ("Top-down trajectory\n%s   %d x %.0f s windows"
                  % (name, nwin, W / 100.0))
    ax.set_title(_title, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    def stack(a, row, ylab, series, legend=False):
        for w in range(nwin):
            for arr, col, lw, ls, lab in series:
                if arr is None:
                    continue
                a.plot(r["t"][w], arr[w], color=col, lw=lw, ls=ls,
                       label=lab if w == 0 else None)
        a.set_ylabel(ylab, fontsize=9)
        a.grid(alpha=0.3)
        a.tick_params(labelsize=8)
        if legend:
            a.legend(loc="upper right", fontsize=8, ncol=3)
        if row == 5:
            a.set_xlabel("time within the window (s)" if single
                         else "time since start of flight (s)")

    # ---- altitude -------------------------------------------------------
    a0 = fig.add_subplot(gs[0, 1:])
    stack(a0, 0, "Up (m)", [
        (r["gt_pos"][:, :, 2], GT, 1.6, "-", "ground truth"),
        (r["raw_pos"][:, :, 2] if has_raw else None, RW, 0.9, ":", "raw IMU"),
        (r["pos"][:, :, 2], MD, 1.1, "-", "AirIMU (model)")], legend=True)

    # ---- the two error panels ------------------------------------------
    a1 = fig.add_subplot(gs[1, 1:])
    stack(a1, 1, "|pos error| (m)", [
        (rperr, RW, 1.1, ":", "raw IMU"), (perr, MD, 1.3, "-", "AirIMU (model)")], legend=True)
    a1.set_title("Position error vs ground truth", fontsize=9)

    a2 = fig.add_subplot(gs[2, 1:])
    stack(a2, 2, "|vel error| (m/s)", [
        (rverr, RW, 1.1, ":", "raw IMU"), (verr, MD, 1.3, "-", "AirIMU (model)")])
    a2.set_title("Velocity error vs ground truth", fontsize=9)

    # ---- velocity components -------------------------------------------
    for i, lab in enumerate(("v North (m/s)", "v East (m/s)", "v Up (m/s)")):
        a = fig.add_subplot(gs[3 + i, 1:])
        stack(a, 3 + i, lab, [
            (r["gt_vel"][:, :, i], GT, 1.6, "-", "ground truth"),
            (r["raw_vel"][:, :, i] if has_raw else None, RW, 0.9, ":", "raw IMU"),
            (r["vel"][:, :, i], MD, 1.1, "-", "AirIMU (model)")])

    sub = (("integrated from ground truth at the window start (matches metric.csv)"
            if single else
            "each window re-initialised from ground truth (matches metric.csv)")
           if args.mode == "window" else
           "continuous dead reckoning: ground truth used once, at the first window")
    fig.suptitle("%s   |   ckpt %s (epoch %s)   |   %s"
                 % (sub, os.path.basename(args.ckpt), ck.get("epoch"), name), fontsize=10)
    fig.savefig(outpng, dpi=140, bbox_inches="tight")
    print("\nwrote %s" % outpng)

    csvout = outpng.replace(".png", ".csv")
    np.savetxt(csvout, np.column_stack([
        r["t"].reshape(-1), perr.reshape(-1), verr.reshape(-1)]
        + ([rperr.reshape(-1), rverr.reshape(-1)] if has_raw else [])),
        delimiter=",", comments="", fmt="%.6f",
        header="t_s,pos_error_m,vel_error_mps" + (",raw_pos_error_m,raw_vel_error_mps" if has_raw else ""))
    print("wrote %s" % csvout)


if __name__ == "__main__":
    main()
