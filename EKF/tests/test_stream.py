"""Stream mode: (A) identical to the offline evaluator, (B) a realistic run with
GPS then an outage, late VO, dropped IMU samples and an IMU dropout, (C) the C++
port gives the same states as Python on the same message log."""
import os
import shutil
import subprocess

import numpy as np
import pytest

from ekf import pipeline as PL
from ekf import so3
from ekf.eskf import ESKFParams
from ekf.events import build_events, read_events, run_stream, write_events, stream_from_config
from ekf.stream import StreamEKF
from ekf.vo import simulate_vo, VOStream
import run_ekf

IMU_CFG = os.path.join(PL.IMU_ROOT, "configs/exp/UAV/tilt_rotate.conf")
CPP = os.path.join(run_ekf.HERE, "cpp")


@pytest.fixture(scope="module")
def flight(tmp_path_factory):
    from test_onnx_inference import write_flight                 # IMU/tests
    d = tmp_path_factory.mktemp("stream")
    path = str(d / "2099_03_03_1_1_sensor_data.csv")
    write_flight(path, seconds=240.0, seed=7)
    return dict(dir=d, csv=path)


# ------------------------------------------------------------------------- A
def test_stream_is_identical_to_offline(flight):
    win = next(PL.load_windows(IMU_CFG, "inference", 3000, csv=flight["csv"]))
    win["dt"] = np.diff(win["t"])        # the stream only knows dt from timestamps
    R_aid = PL.rot_source(win, "gt")
    vo = simulate_vo(win["t"], R_aid, win["v_gt"], rate_hz=2.0, seed=3).window(win["t"][0],
                                                                                win["t"][-1])
    prm = ESKFParams()
    aid = {"source": "gt", "std_tilt_deg": 0.2, "std_yaw_deg": 0.5, "every": 10}
    pos_off, vel_off, f_off = PL.run_eskf(win, win["acc"], win["gyro"], prm, aid, vo=vo)

    t = win["t"]
    ev = [(t[0], 0, "INIT")]
    ev += [(t[k], 1, "IMU", k) for k in range(len(t))]            # sample k stamped t[k]
    ev += [(t[k], 3, "ATT", k) for k in range(1, len(t))]
    ev += [(tv, 2, "VO", j) for j, tv in enumerate(vo.t)]
    ev.sort(key=lambda e: (e[0], e[1]))
    s = StreamEKF(prm, attitude_every=10, std_tilt_deg=0.2, std_yaw_deg=0.5)
    pos, vel = [], []
    for e in ev:
        if e[2] == "INIT":
            s.initialize(t[0], win["p_gt"][0], win["v_gt"][0], win["R_gt"][win["hist"]])
        elif e[2] == "IMU":
            k = e[3]
            # the state at t[k-1] is final once every message stamped t[k-1] (VO,
            # attitude) is in -- i.e. when the next IMU sample arrives
            if k > 1:
                pos.append(s.f.p.copy()); vel.append(s.f.v.copy())
            acc = win["acc"][k] if k < len(win["acc"]) else np.zeros(3)
            gyro = win["gyro"][k] if k < len(win["gyro"]) else np.zeros(3)
            s.on_imu(t[k], acc, gyro)
        elif e[2] == "ATT":
            s.on_attitude(t[e[3]], R_aid[e[3]])
        else:
            s.on_vo(vo.t[e[3]], vo.v[e[3]], vo.var[e[3]])
    pos.append(s.f.p.copy()); vel.append(s.f.v.copy())
    assert np.abs(np.array(pos) - pos_off).max() < 1e-9
    assert np.abs(np.array(vel) - vel_off).max() < 1e-9
    assert s.counters["vo"] == len(vo) and s.f.stats["vel"] == f_off.stats["vel"]


# ------------------------------------------------------------------------- B
def _scenario(flight, with_vo=True):
    fl = PL.load_flight(flight["csv"])
    # a 0.5 s IMU dropout in the middle of the outage
    t = fl["t"]
    hole = (t > t[0] + 150.0) & (t < t[0] + 150.5)
    fl_h = {k: (v[~hole] if isinstance(v, np.ndarray) and len(v) == len(t) else v)
            for k, v in fl.items()}
    t0 = fl["t"][0] + 1.0
    vo = None
    if with_vo:
        vo = simulate_vo(fl["t"], fl["R_gt"], fl["v_gt"], rate_hz=2.0, white_std=0.3,
                         bias_std=0.0, seed=1)
    ev = build_events(fl_h, vo, t_init=t0, t_end=t0 + 200.0, gps_until=t0 + 60.0,
                      vo_latency_s=0.1, imu_drop=0.01, seed=2)
    return fl, ev, t0


def _errors(fl, rows, t_query):
    out = {}
    for tq in t_query:
        i = int(np.searchsorted(rows[:, 0], tq))
        k = int(np.searchsorted(fl["t"], rows[i, 0]))
        out[tq] = (np.linalg.norm(rows[i, 1:4] - fl["p_gt"][k]),
                   np.linalg.norm(rows[i, 4:7] - fl["v_gt"][k]))
    return out


def test_stream_real_use_gps_then_outage(flight):
    cfg = run_ekf.load_config(os.path.join(run_ekf.HERE, "configs", "ekf_default.json"))
    cfg["vo"]["var_scale"] = 1.0
    make = stream_from_config(cfg)
    fl, ev_vo, t0 = _scenario(flight, with_vo=True)
    _, ev_imu, _ = _scenario(flight, with_vo=False)
    rows_vo, s_vo = run_stream(ev_vo, make)
    rows_imu, s_imu = run_stream(ev_imu, make)
    t_out = t0 + 60.0
    q = [t_out + 30.0, t_out + 60.0, t_out + 120.0]
    e_vo, e_imu = _errors(fl, rows_vo, q), _errors(fl, rows_imu, q)
    for tq in q:
        print("outage +%3.0f s: pos %.2f m (IMU only %.2f) | vel %.3f m/s (IMU only %.3f)"
              % (tq - t_out, e_vo[tq][0], e_imu[tq][0], e_vo[tq][1], e_imu[tq][1]))
    assert e_vo[q[-1]][0] < e_imu[q[-1]][0]
    assert all(e_vo[tq][1] < 0.5 for tq in q)
    # while GPS was up the filter learned the accelerometer bias (truth, FLU m/s^2)
    i = int(np.searchsorted(rows_vo[:, 0], t_out))
    ba_true = np.array([0.004, 0.003, -0.002]) * 9.81007
    assert np.linalg.norm(rows_vo[i, 11:14] - ba_true) < 0.5 * np.linalg.norm(ba_true)
    c = s_vo.counters
    assert c["vo_late"] > 0 and c["imu_gap"] >= 1 and c["gps"] > 250
    assert c["vo_dropped"] == 0
    write_events(str(flight["dir"] / "events.csv"), ev_vo)


# ------------------------------------------------------------------------- C
@pytest.mark.skipif(shutil.which("cmake") is None or shutil.which("g++") is None,
                    reason="needs cmake and a C++ compiler")
def test_cpp_matches_python(flight, tmp_path):
    path = str(flight["dir"] / "events.csv")
    if not os.path.isfile(path):
        fl, ev, _ = _scenario(flight, with_vo=True)
        write_events(path, ev)
    build = str(tmp_path / "build")
    subprocess.run(["cmake", "-S", CPP, "-B", build, "-DCMAKE_BUILD_TYPE=Release"], check=True,
                   capture_output=True)
    subprocess.run(["cmake", "--build", build, "-j"], check=True, capture_output=True)
    cfg_path = os.path.join(run_ekf.HERE, "configs", "ekf_default.json")
    out = str(tmp_path / "states_cpp.csv")
    r = subprocess.run([os.path.join(build, "ekf_replay"), path, out, cfg_path], check=True,
                       capture_output=True, text=True)
    print(r.stdout.strip())
    cpp = np.loadtxt(out, delimiter=",", skiprows=1)
    cfg = run_ekf.load_config(cfg_path)
    py, _ = run_stream(read_events(path), stream_from_config(cfg))
    assert cpp.shape == py.shape
    assert np.abs(cpp[:, 0] - py[:, 0]).max() < 1e-5
    dp, dv = np.abs(cpp[:, 1:4] - py[:, 1:4]).max(), np.abs(cpp[:, 4:7] - py[:, 4:7]).max()
    dq = np.abs(cpp[:, 7:11] - py[:, 7:11]).max()
    db = np.abs(cpp[:, 11:17] - py[:, 11:17]).max()
    print("C++ vs Python over %d states: |dp| %.2e m  |dv| %.2e m/s  |dq| %.2e  |db| %.2e"
          % (len(py), dp, dv, dq, db))
    assert dp < 1e-5 and dv < 1e-7 and dq < 1e-8 and db < 1e-8
