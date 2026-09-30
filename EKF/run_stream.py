"""Stream (real-use) test on a flight: GPS up at first, then an outage.

The filter starts from the nav state at --start_s, is corrected by GPS velocity
for --gps_s seconds (it learns the IMU biases there -- no offline bias freeze),
then GPS is lost and it runs on IMU + attitude + VO only.  Errors are reported at
fixed times INTO the outage, for the EKF and for the same filter without VO.

Messages are fed in arrival order exactly as on the aircraft (EKF/ekf/stream.py).
--events_out writes that message log; --cpp runs the C++ ekf_replay on it and
reports the C++ vs Python difference -- the check to run before trusting the
Jetson build on your data.

    python run_stream.py --csv flight_sensor_data.csv --vo_sim --gps_s 60 --duration_s 360
    python run_stream.py --csv flight_sensor_data.csv --vo_onnx ../VO/export/onnx \
        --vo_dataset ../VO/data_split/test --events_out events.csv --cpp cpp/build/ekf_replay
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ekf import pipeline as PL                                        # noqa: E402
from ekf.events import build_events, run_stream, stream_from_config, write_events  # noqa: E402
from ekf.vo import load_vo_csv, simulate_vo                           # noqa: E402
from ekf import vo_onnx                                               # noqa: E402
import run_ekf                                                        # noqa: E402


def errors_at(fl, rows, t_query):
    out = []
    for tq in t_query:
        i = int(np.searchsorted(rows[:, 0], tq))
        if i >= len(rows):
            out.append(None)
            continue
        k = int(np.clip(np.searchsorted(fl["t"], rows[i, 0]), 0, len(fl["t"]) - 1))
        v, g = rows[i, 4:7], fl["v_gt"][k]
        cos = v @ g / max(np.linalg.norm(v) * np.linalg.norm(g), 1e-9)
        out.append((np.linalg.norm(rows[i, 1:4] - fl["p_gt"][k]), np.linalg.norm(v - g),
                    np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True, help="IMU flight log (*_sensor_data.csv)")
    ap.add_argument("--ekf_config", default=os.path.join(HERE, "configs", "ekf_default.json"))
    vo = ap.add_mutually_exclusive_group(required=True)
    vo.add_argument("--vo_onnx", help="VO ONNX export folder; needs --vo_dataset")
    vo.add_argument("--vo_csv", help="VO predictions CSV of this flight")
    vo.add_argument("--vo_sim", action="store_true", help="SIMULATED VO (tests the filter only)")
    ap.add_argument("--vo_dataset", help="VO flight folder (flight.csv + images/)")
    ap.add_argument("--vo_cache", default=None, help="save / reuse the VO ONNX output CSV")
    ap.add_argument("--start_s", type=float, default=1.0, help="filter start, s after log start")
    ap.add_argument("--gps_s", type=float, default=60.0, help="GPS-aided seconds before outage")
    ap.add_argument("--duration_s", type=float, default=None, help="total (default gps_s + 300)")
    ap.add_argument("--horizons_s", type=float, nargs="+", default=[30, 60, 120, 180, 240, 300])
    ap.add_argument("--gps_std", type=float, default=0.1, help="GPS velocity noise, m/s")
    ap.add_argument("--vo_latency_s", type=float, default=0.0,
                    help="extra delay between a VO output's timestamp and its arrival")
    ap.add_argument("--imu_drop", type=float, default=0.0, help="fraction of IMU samples lost")
    ap.add_argument("--events_out", default=None, help="write the message log (C++ input)")
    ap.add_argument("--cpp", default=None, help="path to ekf_replay: run it and compare")
    ap.add_argument("--out_csv", default=None, help="EKF states after every IMU sample")
    a = ap.parse_args(argv)

    cfg = run_ekf.load_config(a.ekf_config)
    vc = cfg["vo"]
    fl = PL.load_flight(a.csv)
    t_start = fl["t"][0] + a.start_s
    t_out = t_start + a.gps_s
    t_end = t_start + (a.duration_s or a.gps_s + max(a.horizons_s))
    if a.vo_sim:
        s = cfg["vo_sim"]
        vo = simulate_vo(fl["t"], fl["R_gt"], fl["v_gt"], rate_hz=s["rate_hz"],
                         white_std=s["white_std"], bias_std=s["bias_std"], tau_s=s["tau_s"],
                         seed=s["seed"], var_scale=vc["var_scale"])
    elif a.vo_csv:
        vo = load_vo_csv(a.vo_csv, frame=vc["frame"], time_offset=vc["time_offset_s"],
                         var_scale=vc["var_scale"], min_std=vc["min_std"],
                         min_interval_s=vc.get("min_interval_s", 0.5),
                         fresh_only=vc.get("fresh_only", True))
    else:
        if not a.vo_dataset:
            ap.error("--vo_onnx needs --vo_dataset")
        if a.vo_cache and os.path.isfile(a.vo_cache):
            vo = load_vo_csv(a.vo_cache, time_offset=vc["time_offset_s"],
                             var_scale=vc["var_scale"], min_std=vc["min_std"])
        else:
            vo = vo_onnx.replay(a.vo_onnx, a.vo_dataset, time_offset=vc["time_offset_s"],
                                var_scale=vc["var_scale"], min_std=vc["min_std"],
                                save_csv=a.vo_cache)
    print("[vo] %s" % vo.source)

    att = cfg["attitude_aid"].get("source") or "gt"
    kw = dict(t_init=t_start, t_end=t_end, gps_until=t_out, gps_std=a.gps_std, att_source=att,
              vo_latency_s=a.vo_latency_s, imu_drop=a.imu_drop, seed=0)
    ev = build_events(fl, vo, **kw)
    ev_imu = build_events(fl, None, **kw)
    make = stream_from_config(cfg)
    rows, s = run_stream(ev, make)
    rows_imu, _ = run_stream(ev_imu, make)

    q = [t_out + h for h in a.horizons_s]
    e, e0 = errors_at(fl, rows, q), errors_at(fl, rows_imu, q)
    i = int(np.searchsorted(rows[:, 0], t_out))
    print("\n[stream] %s | start %.1f s, GPS for %.0f s, outage at %.1f s | %d messages"
          % (fl["flight"], t_start, a.gps_s, t_out, len(ev)))
    print("  at the outage: |ba| %.4f m/s^2  |bg| %.4f deg/s (learned while GPS was up)"
          % (np.linalg.norm(rows[i, 11:14]), np.degrees(np.linalg.norm(rows[i, 14:17]))))
    print("  counters: %s" % s.counters)
    nis = np.array(s.f.stats["nis_vel"])
    if len(nis):
        print("  VO NIS mean %.2f (3.0 = VO variance right; >3: VO trusted too much -> raise "
              "vo.var_scale; <3: lower it)" % nis.mean())
    print("\n  into outage |   pos error [m]   |  vel error [m/s]  |  dir error [deg]")
    print("              |  ekf    imu-only  |  ekf    imu-only  |  ekf    imu-only")
    for h, x, y in zip(a.horizons_s, e, e0):
        if x is None:
            print("  %8.0f s   | (past the end of the log)" % h)
            continue
        print("  %8.0f s   | %6.2f   %7.2f   | %6.3f   %7.3f   | %6.2f   %7.2f"
              % (h, x[0], y[0], x[1], y[1], x[2], y[2]))
    if a.vo_sim:
        print("*** VO IS SIMULATED from GPS truth: a filter test, NOT a VO result.")

    if a.out_csv:
        np.savetxt(a.out_csv, rows, delimiter=",", fmt="%.9f",
                   header="t,px,py,pz,vx,vy,vz,qw,qx,qy,qz,bax,bay,baz,bgx,bgy,bgz,std_p,std_v",
                   comments="")
    if a.events_out or a.cpp:
        path = a.events_out or "events.csv"
        write_events(path, ev)
        print("\nmessage log -> %s" % path)
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
