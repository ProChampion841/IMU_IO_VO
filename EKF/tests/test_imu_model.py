"""The learned IMU correction in the stream: causal, complete, and end to end."""
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from ekf import pipeline as PL
from ekf import so3
from ekf.imu_model import StreamImuCorrector
import run_ekf

IMU_CFG = os.path.join(PL.IMU_ROOT, "configs/exp/UAV/tilt_rotate.conf")
N = 300                                   # a 3 s model keeps the test fast


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from tools.export_onnx import build, export, randomise_zero_heads
    from tools.onnx_inference import OnnxModel
    d = tmp_path_factory.mktemp("imu_model")
    m = build(IMU_CFG)
    randomise_zero_heads(m, std=0.05, seed=4)
    path = str(d / ("imu_%d.onnx" % N))
    export(m, N, path)
    return dict(path=path, onnx=OnnxModel(path), dir=d)


def _data(n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.01
    acc = rng.standard_normal((n, 3)) * 0.3 + [0, 0, 9.81]
    gyro = rng.standard_normal((n, 3)) * 0.02
    R = so3.exp(rng.standard_normal((n, 3)) * 0.05)
    return t, acc, gyro, R


def _run(c, t, acc, gyro, R):
    out = []
    for k in range(len(t)):
        out += c.push(t[k], acc[k], gyro[k], R[k])
    return out + c.flush()


def test_every_sample_released_once_in_order_warmup_raw(model):
    t, acc, gyro, R = _data(900)
    c = StreamImuCorrector(model["onnx"], every=7, delay=16)
    out = _run(c, t, acc, gyro, R)
    assert [o[0] for o in out] == list(t)                    # none lost, none reordered
    n_in = N + 9
    for k in range(n_in - 1):                                # warm-up: raw
        assert np.array_equal(out[k][1], acc[k])
    assert not np.allclose(out[n_in + 50][1], acc[n_in + 50])  # model is doing something
    assert c.runs == (900 - n_in) // 7 + 1


def test_causal_no_future_leak(model):
    """Changing samples after m cannot change anything released before m - delay - every."""
    t, acc, gyro, R = _data(900)
    acc2 = acc.copy()
    m = 700
    acc2[m:] += 5.0
    a = _run(StreamImuCorrector(model["onnx"], every=5, delay=16), t, acc, gyro, R)
    b = _run(StreamImuCorrector(model["onnx"], every=5, delay=16), t, acc2, gyro, R)
    safe = m - 16 - 5
    for k in range(safe):
        assert np.array_equal(a[k][1], b[k][1]) and np.array_equal(a[k][2], b[k][2])
    assert not np.allclose(a[m - 1][1], b[m - 1][1])         # within the look-ahead: may differ


def test_release_latency(model):
    """A corrected sample is released no earlier than `delay` samples after itself."""
    t, acc, gyro, R = _data(700)
    c = StreamImuCorrector(model["onnx"], every=10, delay=16)
    lat = []
    for k in range(len(t)):
        for ts, _, _ in c.push(t[k], acc[k], gyro[k], R[k]):
            lat.append(round((t[k] - ts) / 0.01))
    after = lat[N + 9:]
    assert min(after) >= 16 and max(after) <= 16 + 10


@pytest.fixture(scope="module")
def flight(tmp_path_factory):
    from test_onnx_inference import write_flight
    d = tmp_path_factory.mktemp("imu_stream")
    path = str(d / "2099_05_05_1_1_sensor_data.csv")
    write_flight(path, seconds=150.0, seed=9)
    return dict(dir=d, csv=path)


@pytest.mark.parametrize("mode", ["gps", "no_gt"])
def test_run_stream_with_imu_model(model, flight, capsys, mode, tmp_path):
    import subprocess
    import run_stream as RS
    args = ["--csv", flight["csv"], "--vo_sim", "--imu", "model", "--imu_onnx", model["path"],
            "--imu_every", "20", "--horizons", "30s", "1m"]
    args += ["--no_gt"] if mode == "no_gt" else ["--gps_s", "40"]
    if shutil.which("cmake") and shutil.which("g++"):
        build = str(tmp_path / "build")
        cpp = os.path.join(run_ekf.HERE, "cpp")
        subprocess.run(["cmake", "-S", cpp, "-B", build, "-DCMAKE_BUILD_TYPE=Release"],
                       check=True, capture_output=True)
        subprocess.run(["cmake", "--build", build, "-j"], check=True, capture_output=True)
        args += ["--events_out", str(tmp_path / "ev.csv"), "--cpp", os.path.join(build, "ekf_replay")]
    assert RS.main(args) == 0
    out = capsys.readouterr().out
    assert "IMU model" in out and "vel_rmse" in out
    if "--cpp" in args:
        assert "PASS" in out                   # C++ EKF on the (late) corrected IMU stream


@pytest.mark.skipif(not os.environ.get("ONNXRUNTIME_DIR") or shutil.which("cmake") is None,
                    reason="set ONNXRUNTIME_DIR to an ONNX Runtime C++ release to test the C++ port")
def test_cpp_imu_corrector_matches_python(model, tmp_path):
    import subprocess
    t, acc, gyro, R = _data(1100, seed=5)
    freeze = (np.array([0.02, -0.01, 0.03]), np.array([1e-4, 0.0, -2e-4]))
    on = 400                                        # raw until here, then model + freeze
    raw = str(tmp_path / "raw.csv")
    with open(raw, "w") as f:
        f.write("t,ax,ay,az,gx,gy,gz,qw,qx,qy,qz,active\n")
        for k in range(len(t)):
            row = [t[k], *acc[k], *gyro[k], *so3.mat_to_quat(R[k]), float(k >= on)]
            if k >= on:
                row += [*freeze[0], *freeze[1]]
            f.write(",".join("%.12g" % v for v in row) + "\n")
    c = StreamImuCorrector(model["onnx"], every=7, delay=16)
    py = []
    for k in range(len(t)):
        c.active = k >= on
        if k == on:
            c.set_freeze(*freeze)
        Rk = so3.quat_to_mat(so3.mat_to_quat(R[k]))            # same rounding as the CSV
        py += [(t[k], ts, a, g) for ts, a, g in c.push(t[k], acc[k], gyro[k], Rk)]
    py += [(t[-1], ts, a, g) for ts, a, g in c.flush()]
    build = str(tmp_path / "build")
    cpp = os.path.join(run_ekf.HERE, "cpp")
    subprocess.run(["cmake", "-S", cpp, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                    "-DONNXRUNTIME_DIR=" + os.environ["ONNXRUNTIME_DIR"]], check=True,
                   capture_output=True)
    subprocess.run(["cmake", "--build", build, "-j"], check=True, capture_output=True)
    out = str(tmp_path / "out.csv")
    r = subprocess.run([os.path.join(build, "imu_correct_replay"), model["path"], raw, out, "7",
                        "16"], check=True, capture_output=True, text=True)
    print(r.stdout.strip())
    cc = np.loadtxt(out, delimiter=",", skiprows=1)
    pp_ = np.array([[ta, ts, *a, *g] for ta, ts, a, g in py])
    assert cc.shape == pp_.shape
    assert np.abs(cc[:, :2] - pp_[:, :2]).max() < 1e-5          # same release times
    d = np.abs(cc[:, 2:] - pp_[:, 2:]).max()
    print("C++ vs Python corrected IMU: max diff %.2e" % d)
    assert d < 1e-4
