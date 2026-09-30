"""Stream (real-use) evaluation: GPS up at first, then an outage -- over one or many flights.

Per flight, the filter starts from the nav state at --start_s and is corrected by
GPS velocity for --gps_s seconds. That is where it learns the IMU biases; there is
no offline bias freeze. Then GPS is lost and it runs on IMU + attitude + VO only.
Messages are fed in arrival order, exactly as on the aircraft (ekf/stream.py).

For every horizon INTO the outage (30s 1m 2m ... 40m) it reports, over all flights
whose log is long enough, for the EKF and for the same filter without VO:

    vel_rmse       RMS over flights of |v - v_gps| AT the horizon          m/s
    vel_max_error  worst |v - v_gps| anywhere in [outage, outage + h]      m/s
    dir_rmse       RMS of the velocity-direction error AT the horizon      deg
    dir_max_error  worst direction error anywhere in the interval          deg
    pos_error      mean |p - p_gps| AT the horizon                         m

--no_gt: NO ground truth at all, no reset, no GPS phase -- the whole flight is one
continuous run started from the first VO measurement (position 0, velocity = nav
attitude x VO velocity, attitude = nav).  GPS truth is used only to SCORE it:
position error is on the distance travelled since the start, and horizons are
counted from the start.

--imu model --imu_onnx FILE: the learned IMU correction (IMU/tools/export_onnx.py),
run causally on a sliding window (ekf/imu_model.py) -- both arms get the same IMU.
With GPS it is used from the outage on, with the 15 s pre-outage bias freeze it was
trained with; with --no_gt it is used from the start with no freeze.  --imu raw
(default) feeds the raw IMU.

--events_out writes the message log of the FIRST flight; --cpp runs the C++
ekf_replay on it and checks it matches Python.

    python run_stream.py --imu_config ../IMU/configs/exp/UAV/tilt_rotate.conf \
        --splits inference --vo_dir vo_cache --gps_s 60
    python run_stream.py --csv f1_sensor_data.csv f2_sensor_data.csv --vo_sim
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ekf import pipeline as PL                                        # noqa: E402
from ekf import horizons as HZ                                        # noqa: E402
from ekf.events import build_events, run_stream, stream_from_config, write_events  # noqa: E402
from ekf.vo import load_vo_csv, simulate_vo                           # noqa: E402
from ekf import vo_onnx                                               # noqa: E402
from ekf import imu_model as IM                                       # noqa: E402
import run_ekf                                                        # noqa: E402

ARMS = ("ekf", "imu")


def flight_list(a):
    if a.csv:
        return [(os.path.dirname(c) or ".", os.path.basename(c)) for c in a.csv]
    from pyhocon import ConfigFactory
    conf = ConfigFactory.parse_file(a.imu_config)
    out = []
    for split in a.splits:
        for e in conf.dataset[split].data_list:
            root = a.data_root or e["data_root"]
            out += [(root, f) for f in e["data_drive"]]
    return out[:a.max_flights] if a.max_flights else out


def vo_for(a, cfg, fl, root, name):
    vc = cfg["vo"]
    stem = os.path.splitext(name)[0]
    if a.vo_sim:
        s = cfg["vo_sim"]
        return simulate_vo(fl["t"], fl["R_gt"], fl["v_gt"], rate_hz=s["rate_hz"],
                           white_std=s["white_std"], bias_std=s["bias_std"], tau_s=s["tau_s"],
                           seed=s["seed"], var_scale=vc["var_scale"])
    kw = dict(time_offset=vc["time_offset_s"], var_scale=vc["var_scale"], min_std=vc["min_std"])
    if a.vo_csv or a.vo_dir:
        path = a.vo_csv
        if a.vo_dir:
            path = None
            for cand in (stem + "_vo.csv", stem.replace("_sensor_data", "") + "_vo.csv"):
                if os.path.isfile(os.path.join(a.vo_dir, cand)):
                    path = os.path.join(a.vo_dir, cand)
                    break
            if path is None:
                return None
        return load_vo_csv(path, frame=vc["frame"], min_interval_s=vc.get("min_interval_s", 0.5),
                           fresh_only=vc.get("fresh_only", True), **kw)
    folder = a.vo_dataset
    if a.vo_datasets:
        folder = None
        for cand in (stem, stem.replace("_sensor_data", "")):
            if os.path.isdir(os.path.join(a.vo_datasets, cand)):
                folder = os.path.join(a.vo_datasets, cand)
                break
        if folder is None:
            return None
    cache = os.path.join(a.vo_cache_dir, stem + "_vo.csv") if a.vo_cache_dir else None
    if cache and os.path.isfile(cache):
        return load_vo_csv(cache, **kw)
    return vo_onnx.replay(a.vo_onnx, folder, save_csv=cache, **kw)


def score(fl, rows, t_out, h_s, p_offset=None):
    """The five quantities for one flight at h_s seconds into the outage, or None.
    p_offset is added to the estimated position (no-GT runs start at 0)."""
    t_h = t_out + h_s
    if rows[-1, 0] < t_h - 0.05:
        return None
    i0, i1 = np.searchsorted(rows[:, 0], [t_out, t_h])
    i1 = min(i1, len(rows) - 1)
    seg = rows[i0:i1 + 1]
    k = np.clip(np.searchsorted(fl["t"], seg[:, 0]), 0, len(fl["t"]) - 1)
    ve = np.linalg.norm(seg[:, 4:7] - fl["v_gt"][k], axis=1)
    cos = (seg[:, 4:7] * fl["v_gt"][k]).sum(1) / np.maximum(
        np.linalg.norm(seg[:, 4:7], axis=1) * np.linalg.norm(fl["v_gt"][k], axis=1), 1e-9)
    de = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    p_est = seg[-1, 1:4] + (0.0 if p_offset is None else p_offset)
    pe = np.linalg.norm(p_est - fl["p_gt"][k[-1]])
    return {"vel": ve[-1], "vel_peak": ve.max(), "dir": de[-1], "dir_peak": de.max(), "pos": pe}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", nargs="+", help="IMU flight log(s) (*_sensor_data.csv)")
    src.add_argument("--imu_config", help="IMU config: take the flights of --splits")
    ap.add_argument("--splits", nargs="+", default=["inference"],
                    choices=["train", "test", "eval", "inference"])
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--max_flights", type=int, default=None)
    ap.add_argument("--ekf_config", default=os.path.join(HERE, "configs", "ekf_default.json"))
    vo = ap.add_mutually_exclusive_group(required=True)
    vo.add_argument("--vo_onnx", help="VO ONNX export folder (with --vo_dataset / --vo_datasets)")
    vo.add_argument("--vo_csv", help="VO predictions CSV (one flight)")
    vo.add_argument("--vo_dir", help="folder of VO CSVs, <flight>_vo.csv")
    vo.add_argument("--vo_sim", action="store_true", help="SIMULATED VO (tests the filter only)")
    ap.add_argument("--vo_dataset", help="VO flight folder (flight.csv + images/), one flight")
    ap.add_argument("--vo_datasets", help="root with one VO flight folder per IMU flight")
    ap.add_argument("--vo_cache_dir", default=None, help="save / reuse VO ONNX output here")
    ap.add_argument("--start_s", type=float, default=1.0, help="filter start, s after log start")
    ap.add_argument("--gps_s", type=float, default=None,
                    help="GPS-aided seconds before the outage (default 60; 0 with --no_gt)")
    ap.add_argument("--no_gt", action="store_true",
                    help="no ground truth: start from the first VO + nav attitude, position 0, "
                         "one continuous run over the whole flight, no reset")
    ap.add_argument("--horizons", nargs="+", default=HZ.DEFAULT,
                    help="into the outage: 30s 1m 2m ... 40m (a bare number = seconds)")
    ap.add_argument("--imu", choices=["raw", "model"], default="raw",
                    help="raw IMU, or the learned IMU correction (needs --imu_onnx)")
    ap.add_argument("--imu_onnx", default=None, help="IMU model from IMU/tools/export_onnx.py")
    ap.add_argument("--imu_every", type=int, default=10,
                    help="run the IMU model every N samples (10 = 10 Hz at 100 Hz IMU)")
    ap.add_argument("--imu_delay", type=int, default=16,
                    help="samples of look-ahead a corrected sample waits for (CNN edge); "
                         "0 for a causal_cnn model")
    ap.add_argument("--gps_std", type=float, default=0.1, help="GPS velocity noise, m/s")
    ap.add_argument("--vo_latency_s", type=float, default=0.0,
                    help="extra delay between a VO output's timestamp and its arrival")
    ap.add_argument("--imu_drop", type=float, default=0.0, help="fraction of IMU samples lost")
    ap.add_argument("--events_out", default=None, help="message log of the first flight (C++)")
    ap.add_argument("--cpp", default=None, help="path to ekf_replay: run it on the first flight")
    ap.add_argument("--out_csv", default=None, help="one row per (flight, horizon)")
    a = ap.parse_args(argv)
    if (a.vo_csv or a.vo_dataset) and (not a.csv or len(a.csv) != 1):
        ap.error("--vo_csv / --vo_dataset belong to ONE flight: give exactly one --csv")
    if a.vo_onnx and not (a.vo_dataset or a.vo_datasets):
        ap.error("--vo_onnx needs --vo_dataset or --vo_datasets")

    if a.gps_s is None:
        a.gps_s = 0.0 if a.no_gt else 60.0
    corrector = None
    if a.imu == "model":
        if not a.imu_onnx:
            ap.error("--imu model needs --imu_onnx")
        from tools.onnx_inference import OnnxModel          # IMU/tools
        corrector = IM.StreamImuCorrector(OnnxModel(a.imu_onnx), every=a.imu_every,
                                          delay=a.imu_delay)
        print("[imu] model %s: %g s window, run every %d samples, released %d samples late"
              % (a.imu_onnx, corrector.m.frames / 100.0, a.imu_every, a.imu_delay))
    cfg = run_ekf.load_config(a.ekf_config)
    make = stream_from_config(cfg)
    hs = sorted(HZ.parse(h, plain="seconds") for h in a.horizons)       # frames
    h_max_s = hs[-1] / HZ.RATE_HZ
    att = cfg["attitude_aid"].get("source") or "gt"
    rows_out, first_ev, nis_all = [], None, []
    flights = flight_list(a)
    print("[stream] %d flight(s) | %s | IMU %s | GPS %.0f s then outage | horizons %s"
          % (len(flights), "NO GT: start from VO + nav attitude, no reset" if a.no_gt
             else "start from the GPS/nav state", a.imu, a.gps_s,
             " ".join(HZ.label(h) for h in hs)))
    for root, name in flights:
        try:
            fl = PL.load_flight(os.path.join(root, name))
        except Exception as e:                                            # noqa: BLE001
            print("  [skip] %s: %s" % (name, e))
            continue
        vo = vo_for(a, cfg, fl, root, name)
        if vo is None:
            print("  [skip] %s: no VO for this flight" % name)
            continue
        t_start = fl["t"][0] + a.start_s
        t_out = t_start + a.gps_s
        if fl["t"][-1] < t_out + hs[0] / HZ.RATE_HZ:
            print("  [skip] %s: %.0f s log, too short for the first horizon"
                  % (name, fl["t"][-1] - fl["t"][0]))
            continue
        kw = dict(t_init=t_start, t_end=min(fl["t"][-1], t_out + h_max_s + 1.0),
                  gps_until=t_out if a.gps_s > 0 else None, gps_std=a.gps_std, att_source=att,
                  vo_latency_s=a.vo_latency_s, imu_drop=a.imu_drop, seed=0,
                  init="vo" if a.no_gt else "gt", init_vo=vo)
        if corrector is not None:
            freeze, act = None, None
            if a.gps_s > 0:                    # model from the outage, trained-style freeze
                act = t_start + a.gps_s
                freeze = IM.freeze_at(fl, act)
            cache = {}

            def imu_fn(kept, _f=IM.stream_imu_fn(fl, corrector, att, act, freeze)):
                key = (len(kept), kept[0], kept[-1])
                if key not in cache:
                    cache[key] = _f(kept)
                return cache[key]
            kw["imu_fn"] = imu_fn
        try:
            ev = build_events(fl, vo, **kw)
        except ValueError as e:
            print("  [skip] %s: %s" % (name, e))
            continue
        # the run really starts at the INIT message (no-GT: the first VO sample)
        t_start = ev[0][1]
        t_out = t_start + a.gps_s
        k0 = int(np.clip(np.searchsorted(fl["t"], t_start), 0, len(fl["t"]) - 1))
        p_off = fl["p_gt"][k0] if a.no_gt else None
        res = {"ekf": run_stream(ev, make), "imu": run_stream(build_events(fl, None, **kw), make)}
        if first_ev is None:
            first_ev = (name, ev, res["ekf"][0])
        nis_all += res["ekf"][1].f.stats["nis_vel"]
        i = int(np.searchsorted(res["ekf"][0][:, 0], t_out))
        ba = res["ekf"][0][min(i, len(res["ekf"][0]) - 1), 11:14]
        n_ok = 0
        for h in hs:
            per = {arm: score(fl, res[arm][0], t_out, h / HZ.RATE_HZ, p_off) for arm in ARMS}
            if per["ekf"] is None:
                continue
            n_ok += 1
            r = {"flight": name, "horizon": h, "tag": HZ.label(h), "ba_at_outage": np.linalg.norm(ba)}
            for arm in ARMS:
                r.update({"%s_%s" % (arm, k): float(v) for k, v in per[arm].items()})
            rows_out.append(r)
        print("  %-40s %5.0f s log | |ba| at outage %.4f m/s^2 | %d horizons fit | %s"
              % (name[:40], fl["t"][-1] - fl["t"][0], np.linalg.norm(ba), n_ok, vo.source[:60]))

    if not rows_out:
        print("no flight long enough")
        return 1
    print("\n=== STREAM%s: %s, over flights; each cell = ekf / imu-only (ratio) ==="
          % (" (NO GT, no reset)" if a.no_gt else "",
             "time since start" if a.no_gt and a.gps_s == 0 else "time into the outage"))
    head = "%-6s %4s | " % ("horizon", "fl") + " | ".join("%-26s" % n for n, _, _ in PL.SUMMARY)
    print(head)
    print("-" * len(head))
    for h in hs:
        hr = [r for r in rows_out if r["horizon"] == h]
        if not hr:
            print("%-7s %4d | no flight long enough (needs %s%s)"
                  % (HZ.label(h), 0, HZ.label(h),
                     "" if a.gps_s == 0 else " + %.0f s of GPS" % a.gps_s))
            continue
        m = {arm: PL.reduce(hr, arm) for arm in ARMS}
        cells = ["%8.3f / %8.3f (%4.2f)" % (m["ekf"][n], m["imu"][n],
                                             m["ekf"][n] / max(m["imu"][n], 1e-12))
                 for n, _, _ in PL.SUMMARY]
        print("%-7s %4d | %s" % (HZ.label(h), len(hr), " | ".join("%-26s" % c for c in cells)))
    print("units: vel m/s, dir deg, pos m.  *_rmse / pos_error AT the horizon (pos_error is the")
    print("mean over flights); *_max_error = worst moment in [outage, outage + h].")
    if nis_all:
        print("VO NIS mean %.2f (3.0 = VO variance right; >3 raise vo.var_scale, <3 lower it)"
              % np.mean(nis_all))
    if a.vo_sim:
        print("*** VO IS SIMULATED from GPS truth: a filter test, NOT a VO result.")

    if a.out_csv:
        import csv
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print("per-flight results -> %s" % a.out_csv)
    if (a.events_out or a.cpp) and first_ev:
        name, ev, rows = first_ev
        path = a.events_out or "events.csv"
        write_events(path, ev)
        print("\nmessage log of %s -> %s" % (name, path))
        if a.cpp:
            out = os.path.splitext(path)[0] + "_cpp_states.csv"
            r = subprocess.run([a.cpp, path, out, a.ekf_config], capture_output=True, text=True)
            print("[cpp] " + (r.stdout.strip() or r.stderr.strip()))
            cpp = np.loadtxt(out, delimiter=",", skiprows=1)
            if cpp.shape != rows.shape:
                print("[cpp] FAIL: %s states vs Python %s" % (cpp.shape, rows.shape))
                return 1
            dp = np.abs(cpp[:, 1:4] - rows[:, 1:4]).max()
            dv = np.abs(cpp[:, 4:7] - rows[:, 4:7]).max()
            ok = dp < 1e-4 and dv < 1e-6
            print("[cpp] C++ vs Python: max |dp| %.2e m, |dv| %.2e m/s -> %s"
                  % (dp, dv, "PASS" if ok else "FAIL"))
            return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
