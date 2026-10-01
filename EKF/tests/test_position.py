"""Absolute position fixes (land matching): the update, late fixes (rewind + replay),
wrong matches, the gate lock-out reset, the CSV / lat-lon input, and C++ = Python."""
import os
import shutil
import subprocess

import numpy as np
import pytest

from ekf import pipeline as PL
from ekf import so3
from ekf.eskf import ESKF, ESKFParams
from ekf.events import build_events, read_events, run_stream, stream_from_config, write_events
from ekf.landmatch import geodetic_to_nwu, read_fixes_csv, simulate_fixes
from ekf.vo import simulate_vo
import run_ekf

CFG = os.path.join(run_ekf.HERE, "configs", "ekf_default.json")
CPP = os.path.join(run_ekf.HERE, "cpp")


@pytest.fixture(scope="module")
def flight(tmp_path_factory):
    from test_onnx_inference import write_flight                 # IMU/tests
    d = tmp_path_factory.mktemp("pos")
    path = str(d / "2099_04_04_1_1_sensor_data.csv")
    write_flight(path, seconds=240.0, seed=7)
    return dict(dir=d, csv=path, fl=PL.load_flight(path))


def _make(**over):
    cfg = run_ekf.load_config(CFG)
    cfg["position_aid"].update(over)
    return stream_from_config(cfg)


def _events(fl, latency_s=0.5, t_len=120.0, gps_s=30.0, pos_until=None, **sim):
    t0 = fl["t"][0] + 1.0
    vo = simulate_vo(fl["t"], fl["R_gt"], fl["v_gt"], rate_hz=2.0, white_std=0.3,
                     bias_std=0.0, seed=1)
    kw = dict(rate_hz=1.0, std_m=5.0, seed=4)
    kw.update(sim)
    pos = simulate_fixes(fl["t"], fl["p_gt"], t0 + gps_s, t0 + (pos_until or t_len),
                         latency_s=latency_s, **kw)
    ev = build_events(fl, vo, pos=pos, t_init=t0, t_end=t0 + t_len, gps_until=t0 + gps_s,
                      vo_latency_s=0.1, seed=2)
    return ev, pos, t0


def _pos_err(fl, rows, t):
    i = int(np.searchsorted(rows[:, 0], t))
    i = min(i, len(rows) - 1)
    k = int(np.searchsorted(fl["t"], rows[i, 0]))
    return np.linalg.norm(rows[i, 1:3] - fl["p_gt"][k, 0:2])


# ------------------------------------------------------------------------- filter core
def test_update_position_pulls_the_state_and_shrinks_p():
    f = ESKF([30.0, -20.0, 5.0], np.zeros(3), np.eye(3),
             params=ESKFParams(init_pos_std=50.0, gate=True))
    ok, nis = f.update_position([0.0, 0.0, 0.0], [4.0, 4.0, 4.0])
    assert ok and nis > 0
    assert np.linalg.norm(f.p[:2]) < 1.0                      # horizontal pulled to the fix
    assert f.p[2] == pytest.approx(5.0)                        # vertical not used
    assert f.P[0, 0] < 4.0 and f.P[2, 2] == pytest.approx(2500.0)
    assert f.stats["pos"] == [1, 0] and len(f.stats["nis_pos"]) == 1


def test_lever_arm_jacobian_matches_numeric():
    rng = np.random.default_rng(0)
    R = so3.exp(rng.standard_normal(3) * 0.5)
    lever = np.array([0.4, -0.2, 0.3])
    f = ESKF(np.zeros(3), np.zeros(3), R)
    _, H = f.position_residual(np.zeros(3), (0, 1, 2), lever)
    dth = 1e-6 * rng.standard_normal(3)
    h0 = R @ lever
    h1 = R @ so3.exp(dth) @ lever                              # R_true = R Exp(dtheta)
    assert np.allclose(h1 - h0, H[:, 6:9] @ dth, atol=1e-11)


def test_reset_position_zeroes_correlation():
    f = ESKF(np.zeros(3), np.zeros(3), np.eye(3))
    f.P[0, 3] = f.P[3, 0] = 0.01
    f.reset_position([100.0, 50.0, 7.0], [25.0, 25.0, 25.0])
    assert np.allclose(f.p, [100.0, 50.0, 0.0])
    assert f.P[0, 0] == 25.0 and f.P[0, 3] == 0.0 and f.P[3, 0] == 0.0


# ------------------------------------------------------------------------- stream
def test_late_fix_gives_exactly_the_on_time_result(flight):
    """A fix arriving 0.5 / 0.9 s late is applied at its image time by rewinding:
    once it has arrived the state is bit-identical to an on-time fix."""
    fl, make = flight["fl"], _make()
    rows0, s0 = run_stream(_events(fl, latency_s=0.0)[0], make)
    _, pos, _ = _events(fl, latency_s=0.0)
    idx = np.searchsorted(rows0[:, 0], pos.t) - 1              # just before each fix
    idx = idx[(idx > 0) & (pos.t > pos.t[0] + 1.5)]
    for lat in (0.5, 0.9):
        rows, s = run_stream(_events(fl, latency_s=lat)[0], make)
        assert s.counters["pos_late"] == len(pos) and s.counters["pos_dropped"] == 0
        assert np.array_equal(rows[idx, 1:17], rows0[idx, 1:17])
        assert np.array_equal(s.f.p, s0.f.p) and np.array_equal(s.f.P, s0.f.P)
        assert s.f.stats["pos"] == s0.f.stats["pos"]           # replays not counted twice
    # without the history a late fix is applied at once -- and is NOT the same
    rows_n, s_n = run_stream(_events(fl, latency_s=0.5)[0], _make(pos_replay_s=0.0))
    assert s_n.counters["pos"] == len(pos) and s_n.counters["pos_late"] == 0
    assert not np.array_equal(s_n.f.p, s0.f.p)


def test_fix_older_than_the_history_is_dropped(flight):
    fl = flight["fl"]
    ev, pos, _ = _events(fl, latency_s=4.0, pos_until=100.0)   # pos_replay_s is 3
    _, s = run_stream(ev, _make())
    assert s.counters["pos_dropped"] == len(pos) and s.counters["pos"] == 0


def test_fixes_bound_the_drift_and_wrong_matches_are_gated(flight):
    fl = flight["fl"]
    ev, pos, t0 = _events(fl, t_len=200.0, gps_s=60.0, outlier_rate=0.15, outlier_m=300.0)
    rows, s = run_stream(ev, _make())
    t_out = t0 + 60.0
    ev_imu = build_events(fl, None, t_init=t0, t_end=t0 + 200.0, gps_until=t_out, seed=2)
    rows_imu, _ = run_stream(ev_imu, _make())
    e, e_imu = _pos_err(fl, rows, t0 + 199.0), _pos_err(fl, rows_imu, t0 + 199.0)
    print("after 139 s of outage: %.1f m with fixes, %.1f m IMU only | %s"
          % (e, e_imu, {k: v for k, v in s.counters.items() if k.startswith("pos")}))
    assert e < 10.0 and e < e_imu
    wrong = np.linalg.norm(pos.p[:, :2] - fl["p_gt"][np.searchsorted(fl["t"], pos.t), :2],
                           axis=1) > 100.0
    assert s.counters["pos_gated"] >= wrong.sum() and s.counters["pos_gated"] <= wrong.sum() + 3
    assert s.counters["pos_reset"] == 0                        # wrong matches do not agree
    nis = np.array(s.f.stats["nis_pos"])
    assert 1.0 < nis[nis <= 13.816].mean() < 4.0               # accepted fixes, 2-D: about 2


def _offset_init(ev, offset, std=(0.5, 0.1, 0.5)):
    typ, t, x = ev[0]
    assert typ == "INIT"
    x = np.r_[x[0:3] + np.asarray(offset, float), x[3:10], std]
    return [(typ, t, x)] + ev[1:]


def test_gate_lockout_is_cleared_by_a_reset(flight):
    """Start 500 m off with a tiny position sigma: the gate rejects every fix until
    three in a row agree; then the position resets to them."""
    fl = flight["fl"]
    ev, pos, t0 = _events(fl)
    ev = _offset_init(ev, [400.0, -300.0, 0.0])
    rows, s = run_stream(ev, _make())
    assert s.counters["pos_reset"] == 1 and 3 <= s.counters["pos_gated"] <= 5
    assert _pos_err(fl, rows, t0 + 119.0) < 10.0
    _, s_off = run_stream(ev, _make(pos_reset_after=0))       # 0 = never reset
    assert s_off.counters["pos_reset"] == 0 and s_off.counters["pos_gated"] == len(pos)


# ------------------------------------------------------------------------- inputs
def test_geodetic_to_nwu():
    o = (37.0, 127.0, 100.0)
    p = geodetic_to_nwu([37.001, 37.0, 37.0], [127.0, 127.001, 127.0], [100.0, 100.0, 150.0], o)
    assert p[0, 0] == pytest.approx(110.99, abs=0.1) and abs(p[0, 1]) < 0.01   # 0.001 deg N
    assert p[1, 1] == pytest.approx(-88.95, abs=0.1) and abs(p[1, 0]) < 0.01   # east = -west
    assert p[2, 2] == pytest.approx(50.0, abs=1e-6)


def test_read_fixes_csv(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("time,arrival,lat,lon,std,valid\n"
                 "10.0,10.8,37.001,127.0,4.0,1\n"
                 "11.0,11.7,37.0,127.001,6.0,0\n"
                 "12.0,12.9,37.0,127.0,5.0,1\n")
    f = read_fixes_csv(str(a), origin=(37.0, 127.0, 0.0), var_scale=2.0)
    assert len(f) == 2 and np.allclose(f.t, [10.0, 12.0]) and np.allclose(f.t_arrival, [10.8, 12.9])
    assert f.p[0, 0] == pytest.approx(110.99, abs=0.1) and np.allclose(f.var[0], 32.0)
    with pytest.raises(ValueError):
        read_fixes_csv(str(a))                                 # lat/lon needs an origin
    f2 = read_fixes_csv(str(a), origin=(37.0, 127.0))          # no altitude: 0
    assert np.allclose(f2.p[:, :2], f.p[:, :2])
    b = tmp_path / "b.csv"
    b.write_text("t,north,west\n5.0,1.0,2.0\n")
    g = read_fixes_csv(str(b), latency_s=0.5, std_m=8.0)
    assert np.allclose(g.p, [[1.0, 2.0, 0.0]]) and g.t_arrival[0] == 5.5
    assert np.allclose(g.var, 64.0)


def test_run_stream_cli_with_fixes_and_no_vo(flight, capsys):
    import run_stream as cli
    assert cli.main(["--csv", flight["csv"], "--pos_sim", "--gps_s", "30",
                     "--horizons", "30s", "1m"]) == 0
    out = capsys.readouterr().out
    assert "ekf (land match)" in out and "land matching:" in out and "SIMULATED" in out


# ------------------------------------------------------------------------- C++
@pytest.mark.skipif(shutil.which("cmake") is None or shutil.which("g++") is None,
                    reason="needs cmake and a C++ compiler")
def test_cpp_matches_python_late_fixes_wrong_matches_and_reset(flight, tmp_path):
    fl = flight["fl"]
    ev, _, _ = _events(fl, latency_s=1.2, outlier_rate=0.15, outlier_m=300.0)
    ev = _offset_init(ev, [400.0, -300.0, 0.0])               # forces a lock-out reset too
    path = str(tmp_path / "events_pos.csv")
    write_events(path, ev)
    build = str(tmp_path / "build")
    subprocess.run(["cmake", "-S", CPP, "-B", build, "-DCMAKE_BUILD_TYPE=Release"], check=True,
                   capture_output=True)
    subprocess.run(["cmake", "--build", build, "-j"], check=True, capture_output=True)
    out = str(tmp_path / "states_cpp.csv")
    r = subprocess.run([os.path.join(build, "ekf_replay"), path, out, CFG], check=True,
                       capture_output=True, text=True)
    print(r.stdout.strip())
    py, s = run_stream(read_events(path), _make())
    c = s.counters
    assert c["pos_reset"] == 1 and c["pos_late"] > 0 and c["pos_gated"] > 3
    assert ("pos %d (gated %d, late %d, dropped %d, resets %d)"
            % (c["pos"], c["pos_gated"], c["pos_late"], c["pos_dropped"], c["pos_reset"])
            in r.stdout)
    cpp = np.loadtxt(out, delimiter=",", skiprows=1)
    assert cpp.shape == py.shape
    dp, dv = np.abs(cpp[:, 1:4] - py[:, 1:4]).max(), np.abs(cpp[:, 4:7] - py[:, 4:7]).max()
    dq = np.abs(cpp[:, 7:11] - py[:, 7:11]).max()
    print("C++ vs Python over %d states: |dp| %.2e m  |dv| %.2e m/s  |dq| %.2e" % (len(py), dp, dv, dq))
    assert dp < 1e-5 and dv < 1e-7 and dq < 1e-8


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs a C++ compiler")
def test_cpp_geodetic_matches_python(tmp_path):
    src = tmp_path / "geo.cpp"
    src.write_text('#include <cstdio>\n#include "imuvo_ekf.hpp"\n'
                   'int main() { auto p = imuvo::geodeticToNwu(37.0123, 127.0456, 180.0, '
                   '37.0, 127.0, 50.0); std::printf("%.9f %.9f %.9f\\n", p[0], p[1], p[2]); }\n')
    exe = str(tmp_path / "geo")
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", os.path.join(CPP, "include"), str(src),
                    os.path.join(CPP, "src", "imuvo_ekf.cpp"), "-o", exe], check=True)
    got = np.array([float(x) for x in subprocess.run([exe], check=True, capture_output=True,
                                                     text=True).stdout.split()])
    want = geodetic_to_nwu(37.0123, 127.0456, 180.0, (37.0, 127.0, 50.0))
    assert np.abs(got - want).max() < 1e-6
