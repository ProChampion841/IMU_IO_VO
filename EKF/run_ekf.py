"""IMU + VO error-state EKF over flight logs -- evaluation against GPS truth.

For every window (ground-truth initial state, 15 s bias freeze, as in the IMU
project) it runs three arms on the same data and reports, per horizon:

    vel_rmse  vel_max_error  dir_rmse  dir_max_error  pos_error
    for   imu (IMU + attitude aid)   vo (VO velocity integrated)   ekf (IMU + VO)

VO input: --vo_csv (one flight), --vo_dir (one CSV per flight), or --vo_sim
(SIMULATED from GPS truth -- for testing the filter only, never a result).
IMU input: raw (freeze-corrected) or, with --imu_onnx, the learned correction.

Run from the EKF folder, e.g.
    python run_ekf.py --imu_config ../IMU/configs/exp/UAV/tilt_rotate.conf ^
        --splits inference --vo_dir vo_predictions --horizons 3000 6000 12000
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ekf import pipeline as PL                                   # noqa: E402
from ekf.eskf import ESKFParams                                  # noqa: E402
from ekf.vo import load_vo_csv, simulate_vo                      # noqa: E402


def load_config(path):
    with open(path) as f:
        c = json.load(f)
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    return {k: strip(v) if isinstance(v, dict) else v for k, v in c.items()
            if not k.startswith("_")}


def vo_for_flight(a, cfg, flight, win):
    vc = cfg["vo"]
    if a.vo_sim:
        s = cfg["vo_sim"]
        R = PL.rot_source(win, "gt")
        return simulate_vo(win["t"], R, win["v_gt"], rate_hz=s["rate_hz"],
                           white_std=s["white_std"], bias_std=s["bias_std"],
                           tau_s=s["tau_s"], seed=s["seed"] + win["start"],
                           var_scale=vc["var_scale"])
    path = a.vo_csv
    if a.vo_dir:
        stem = os.path.splitext(flight)[0]
        for cand in (stem + "_vo.csv", stem.replace("_sensor_data", "") + "_vo.csv",
                     stem + ".csv", flight):
            if os.path.isfile(os.path.join(a.vo_dir, cand)):
                path = os.path.join(a.vo_dir, cand)
                break
        else:
            return None
    key = path
    if key not in a._vo_cache:
        a._vo_cache[key] = load_vo_csv(path, frame=vc["frame"], time_offset=vc["time_offset_s"],
                                       var_scale=vc["var_scale"], min_std=vc["min_std"],
                                       min_interval_s=vc.get("min_interval_s", 0.5),
                                       fresh_only=vc.get("fresh_only", True))
        print("  [vo] %s" % a._vo_cache[key].source)
    return a._vo_cache[key]


def print_table(split, table, arms):
    print("\n=== %s -- per horizon; ratio = ekf / imu (<1 means VO helped) ===" % split)
    head = "%-8s %5s | " % ("horizon", "wins") + " | ".join(
        "%-30s" % ("%s  [%s]" % (n, " / ".join(arms))) for n, _, _ in PL.SUMMARY)
    print(head)
    print("-" * len(head))
    for h, n, res in table:
        cells = []
        for name, _, _ in PL.SUMMARY:
            vals = " / ".join("%7.3f" % res[a][name] for a in arms)
            ratio = res["ekf"][name] / max(res["imu"][name], 1e-12) if "ekf" in res else float("nan")
            cells.append("%-30s" % ("%s  %4.2f" % (vals, ratio)))
        print("%-8s %5d | %s" % ("%gs" % (h / 100.0), n, " | ".join(cells)))
    print("units: vel m/s, dir deg, pos m.  *_rmse / pos_error AT the horizon (pos_error is the")
    print("mean over windows); *_max_error is the worst frame anywhere inside [0, T].")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--imu_config", required=True, help="IMU training config (flights, freeze)")
    ap.add_argument("--ekf_config", default=os.path.join(HERE, "configs", "ekf_default.json"))
    ap.add_argument("--splits", nargs="+", default=["inference"],
                    choices=["train", "test", "eval", "inference"])
    ap.add_argument("--csv", default=None, help="one IMU flight log instead of a split")
    ap.add_argument("--data_root", default=None)
    vo = ap.add_mutually_exclusive_group(required=True)
    vo.add_argument("--vo_csv", help="VO predictions for the ONE flight given by --csv")
    vo.add_argument("--vo_dir", help="folder with one VO CSV per flight (<flight>_vo.csv)")
    vo.add_argument("--vo_sim", action="store_true", help="SIMULATED VO from GPS truth (testing)")
    ap.add_argument("--imu_onnx", default=None, help="learned IMU correction (export_onnx.py)")
    ap.add_argument("--horizons", type=int, nargs="+", default=[3000, 6000, 12000],
                    help="frames at 100 Hz; the window is the longest one")
    ap.add_argument("--first_only", action="store_true")
    ap.add_argument("--max_flights", type=int, default=None)
    ap.add_argument("--max_windows", type=int, default=None)
    ap.add_argument("--out_csv", default=None, help="one row per (window, horizon)")
    ap.add_argument("--out_npz", default=None, help="trajectories of every window")
    ap.add_argument("--plot_dir", default=None, help="top-down plot per window (matplotlib)")
    a = ap.parse_args(argv)
    if a.vo_csv and not a.csv:
        ap.error("--vo_csv belongs to one flight: give that flight with --csv")
    a._vo_cache = {}
    cfg = load_config(a.ekf_config)
    params = ESKFParams.from_dict(cfg["eskf"])
    aid = cfg["attitude_aid"]
    lever = np.asarray(cfg["vo"].get("lever_arm_m", [0, 0, 0]), float)
    horizons = sorted(a.horizons)
    W = horizons[-1]
    onnx = None
    if a.imu_onnx:
        from tools.onnx_inference import OnnxModel      # IMU/tools (on sys.path via pipeline)
        onnx = OnnxModel(a.imu_onnx)
        print("[imu] learned correction: %s (%g s blocks)" % (a.imu_onnx, onnx.frames / 100.0))
    from pyhocon import ConfigFactory
    att_source_net = str(ConfigFactory.parse_file(a.imu_config).train.get("att_source", "gt"))
    print("[ekf] window %g s | attitude aid %s | VO %s"
          % (W / 100.0, aid.get("source"), "SIMULATED" if a.vo_sim else (a.vo_csv or a.vo_dir)))

    all_rows, results, npz = [], {}, {}
    for split in a.splits:
        rows, n, t0 = [], 0, time.time()
        stats = {"vel_ok": 0, "vel_rej": 0, "nis": [], "ba": [], "bg": []}
        for win in PL.load_windows(a.imu_config, split, W, csv=a.csv, data_root=a.data_root,
                                   max_flights=a.max_flights, first_only=a.first_only):
            if a.max_windows and n >= a.max_windows:
                break
            vo = vo_for_flight(a, cfg, win["flight"], win)
            if vo is None:
                print("  [skip] %s: no VO file" % win["flight"])
                continue
            vo = vo.window(win["t"][0], win["t"][-1])
            if onnx is not None:
                acc, gyro = PL.onnx_correct(win, onnx, att_source_net)
            else:
                acc, gyro = win["acc"], win["gyro"]
            arms = {}
            arms["imu"] = PL.run_eskf(win, acc, gyro, params, aid)[:2]
            arms["vo"] = PL.run_vo_dr(win, vo, aid.get("source") or "gt")
            pe, ve, f = PL.run_eskf(win, acc, gyro, params, aid, vo=vo, lever=lever)
            arms["ekf"] = (pe, ve)
            stats["vel_ok"] += f.stats["vel"][0]
            stats["vel_rej"] += f.stats["vel"][1]
            stats["nis"] += f.stats["nis_vel"]
            stats["ba"].append(f.ba)
            stats["bg"].append(f.bg)
            for h in horizons:
                r = {"split": split, "flight": win["flight"], "start": win["start"],
                     "horizon": h, "vo_samples": len(vo)}
                for arm, (p, v) in arms.items():
                    r.update({"%s_%s" % (arm, k): float(x)
                              for k, x in PL.metrics_at(p, v, win, h).items()})
                rows.append(r)
            if a.out_npz:
                key = "%s/%s/%d" % (split, win["flight"], win["start"])
                npz[key + "/t"] = win["t"][1:]
                npz[key + "/gt_pos"], npz[key + "/gt_vel"] = win["p_gt"][1:], win["v_gt"][1:]
                for arm, (p, v) in arms.items():
                    npz["%s/%s_pos" % (key, arm)], npz["%s/%s_vel" % (key, arm)] = p, v
            if a.plot_dir:
                plot_window(a.plot_dir, split, win, arms)
            n += 1
        if not rows:
            print("[%s] no window" % split)
            continue
        table = []
        for h in horizons:
            hr = [r for r in rows if r["horizon"] == h]
            table.append((h, len(hr), {arm: PL.reduce(hr, arm) for arm in ("imu", "vo", "ekf")}))
        nis = np.array(stats["nis"])
        print("\n[%s] %d windows in %.1f s | VO updates %d accepted, %d gated out | "
              "VO NIS mean %.2f (3.0 = consistent) | final |ba| %.4f m/s^2, |bg| %.4f deg/s"
              % (split, n, time.time() - t0, stats["vel_ok"], stats["vel_rej"],
                 nis.mean() if len(nis) else float("nan"),
                 np.linalg.norm(stats["ba"], axis=1).mean(),
                 np.degrees(np.linalg.norm(stats["bg"], axis=1).mean())))
        if a.vo_sim:
            print("*** VO IS SIMULATED from GPS truth: these numbers test the filter, they are "
                  "NOT a VO result.")
        print_table(split, table, ("imu", "vo", "ekf"))
        results[split] = table
        all_rows += rows

    if a.out_csv and all_rows:
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print("\nper-window results -> %s (%d rows)" % (a.out_csv, len(all_rows)))
    if a.out_npz and npz:
        np.savez_compressed(a.out_npz, **npz)
        print("trajectories -> %s" % a.out_npz)
    return results, all_rows


def plot_window(plot_dir, split, win, arms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    gp = win["p_gt"]
    # NWU -> plot east (= -y) against north (x)
    ax[0].plot(-gp[:, 1], gp[:, 0], "k", lw=2, label="GPS truth")
    for arm, (p, v) in arms.items():
        ax[0].plot(-p[:, 1], p[:, 0], label=arm)
    ax[0].set_xlabel("east [m]"); ax[0].set_ylabel("north [m]"); ax[0].axis("equal")
    ax[0].legend(); ax[0].set_title("%s  start %d" % (win["flight"][:24], win["start"]))
    tt = win["t"][1:] - win["t"][0]
    for arm, (p, v) in arms.items():
        ax[1].plot(tt, np.linalg.norm(p - gp[1:], axis=1), label=arm)
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("position error [m]"); ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "%s_%s_%d.png"
                             % (split, os.path.splitext(win["flight"])[0], win["start"])), dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
