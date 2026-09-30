"""tools/onnx_inference.py: ONNX pipeline == PyTorch pipeline, end to end.

    python -m pytest tests/test_onnx_inference.py -q
A synthetic fixed-wing flight is written in the real *_sensor_data.csv format, so
the whole chain runs: UAV loader -> 15 s bias freeze -> padding9 -> ONNX Runtime
-> NumPy integration -> metrics.  The reference is tools/eval_vel_horizons.py
running the PyTorch model with the pypose integrator on the same windows.
"""
import copy
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
pytest.importorskip("onnxruntime")
pd = pytest.importorskip("pandas")

import pypose as pp                                             # noqa: E402
from scipy.spatial.transform import Rotation as R               # noqa: E402

from utils import pypose_compat                                  # noqa: E402
pypose_compat.apply()
from tools import onnx_inference as OI                          # noqa: E402
from tools.export_onnx import build, export, randomise_zero_heads  # noqa: E402

G = 9.81007
W = 1000                      # 10 s ONNX window keeps the test fast


def write_flight(path, seconds=80.0, seed=0):
    """A coordinated-turn fixed-wing flight in the logger's own units/frames."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, 0.01)
    V = 20.0 + 1.5 * np.sin(0.07 * t)
    yaw = 0.4 * np.sin(0.05 * t) + 0.03 * t
    pitch = 0.03 * np.sin(0.11 * t)
    yawdot = np.gradient(yaw, t)
    roll = np.arctan(V * yawdot / G)
    v_ned = np.stack([V * np.cos(yaw) * np.cos(pitch), V * np.sin(yaw) * np.cos(pitch),
                      -V * np.sin(pitch)], 1)
    a_ned = np.gradient(v_ned, t, axis=0)
    Rm = R.from_euler("ZYX", np.stack([yaw, pitch, roll], 1)).as_matrix()   # FRD -> NED
    f_frd = np.einsum("nji,nj->ni", Rm, a_ned - np.array([0.0, 0.0, G]))
    rel = R.from_matrix(np.einsum("nji,njk->nik", Rm[:-1], Rm[1:])).as_rotvec() / 0.01
    w_frd = np.vstack([rel, rel[-1:]])
    acc_g = f_frd / G + np.array([0.004, -0.003, 0.002]) + 0.002 * rng.standard_normal(f_frd.shape)
    gyro_dps = np.degrees(w_frd + np.array([2e-4, -1e-4, 1.5e-4])) + 0.02 * rng.standard_normal(w_frd.shape)
    alt = 100.0 + np.cumsum(-v_ned[:, 2]) * 0.01
    # MTi attitude: z-up (NWU/FLU) angles in degrees, yaw referenced to magnetic East,
    # i.e. what the loader undoes with Rz(-(90 - 13.06 deg)).
    T = np.diag([1.0, -1.0, -1.0])
    R_nwu = T[None] @ Rm @ T[None]
    psi = np.radians(90.0 - 13.06)
    eul = R.from_matrix(R.from_euler("Z", psi).as_matrix()[None] @ R_nwu).as_euler("ZYX", degrees=True)
    df = pd.DataFrame({"Time": t, "GyroX": gyro_dps[:, 0], "GyroY": gyro_dps[:, 1],
                       "GyroZ": gyro_dps[:, 2], "AcclX": acc_g[:, 0], "AcclY": acc_g[:, 1],
                       "AcclZ": acc_g[:, 2], "GPSNavEulX": roll, "GPSNavEulY": pitch,
                       "GPSNavEulZ": yaw, "GPSNavVnX": v_ned[:, 0], "GPSNavVnY": v_ned[:, 1],
                       "GPSNavVnZ": v_ned[:, 2], "GPSNavAlt": alt, "AirSpeed": V,
                       "EulX": eul[:, 2], "EulY": eul[:, 1], "EulZ": eul[:, 0]})
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx_inf")
    csv_path = write_flight(str(d / "2099_01_01_1_1_sensor_data.csv"))
    cfg = os.path.join(ROOT, "configs/exp/UAV/tilt_rotate.conf")
    m = build(cfg)
    randomise_zero_heads(m, seed=3)
    onnx_path = str(d / "tr_10s.onnx")
    export(m, W, onnx_path)
    ck = str(d / "best_model.ckpt")
    torch.save({"model_state_dict": m.net.state_dict(), "epoch": 1}, ck)
    return dict(dir=d, csv=csv_path, cfg=cfg, onnx=onnx_path, ckpt=ck, model=m)


def test_integrate_np_matches_pypose():
    torch.manual_seed(0)
    B, F = 2, 300
    acc = torch.randn(B, F, 3, dtype=torch.float64) + torch.tensor([0.0, 0.0, G], dtype=torch.float64)
    gyro = 0.1 * torch.randn(B, F, 3, dtype=torch.float64)
    dt = torch.full((B, F, 1), 0.01, dtype=torch.float64)
    rot = pp.so3(0.2 * torch.randn(B, F, 3, dtype=torch.float64)).Exp()
    init = {"pos": torch.randn(B, 1, 3, dtype=torch.float64),
            "vel": torch.randn(B, 1, 3, dtype=torch.float64), "rot": rot[:, :1]}
    integ = pp.module.IMUPreintegrator(reset=True, prop_cov=False, gravity=G).double()
    for use_rot in (True, False):
        ref = integ(init_state=copy.deepcopy(init), dt=dt, gyro=gyro, acc=acc,
                    rot=rot if use_rot else None)
        pos, vel, R_ = OI.integrate_np(acc.numpy(), gyro.numpy(), dt.numpy(),
                                       init["pos"][:, 0].numpy(), init["vel"][:, 0].numpy(),
                                       init["rot"].matrix()[:, 0].numpy(),
                                       rot.matrix().numpy() if use_rot else None, G)
        # pypose stores gravity as a float32 buffer before .double(): 9.81007 carries a
        # 3.8e-8 m/s^2 rounding error, i.e. 1.7e-7 m after 3 s (0.5 * e * t^2).  That,
        # not the integration, is the whole residual -- hence 1e-6, not 1e-12.
        assert np.abs(pos - ref["pos"].numpy()).max() < 1e-6
        assert np.abs(vel - ref["vel"].numpy()).max() < 1e-6
        assert np.abs(R_ - ref["rot"].matrix().numpy()).max() < 1e-10


def test_flight_log_mode_matches_pytorch_eval(setup):
    s = setup
    out_csv = str(s["dir"] / "win.csv")
    table, rows = OI.main(["--onnx", s["onnx"], "--config", s["cfg"], "--csv", s["csv"],
                           "--horizons", "300", str(W), "--compare_ckpt", s["ckpt"],
                           "--out_csv", out_csv, "--out_npz", str(s["dir"] / "traj.npz")])
    got = {h: (m, r) for h, n, m, r in table["inference"]}
    assert os.path.getsize(out_csv) > 0 and len(rows) > 0

    # reference: the PyTorch evaluator on the same flight, same windows
    import tools.eval_vel_horizons as E
    from pyhocon import ConfigFactory
    from datasets import collate_fcs
    conf = ConfigFactory.parse_file(s["cfg"])
    conf.train.device = "cpu"
    for e in conf.dataset.inference.data_list:
        e["data_drive"] = [os.path.basename(s["csv"])]
        e["data_root"] = os.path.dirname(s["csv"])
    net = s["model"].net
    ref_rows, _ = E.run_split_nested(net, conf, "inference", [300, W], "cpu", 8,
                                     collate_fcs["padding9"])
    for r in ref_rows:
        m, raw = got[r["horizon"]]
        for mine, theirs in (("vel_rmse", "vel_rmse"), ("vel_max_error", "vel_max"),
                             ("dir_rmse", "vel_dir_rmse"), ("dir_max_error", "vel_dir_max"),
                             ("pos_error", "pos_mean")):
            for arm, ref_key in ((m, theirs), (raw, "raw_" + theirs)):
                a, b = arm[mine], r[ref_key]
                print("%5d %-14s %-5s onnx %.6f  torch %.6f" % (r["horizon"], mine,
                      "model" if arm is m else "raw", a, b))
                # The reference runs in float32 and takes arccos(cos) near cos = 1, where
                # float32 resolves only sqrt(2 * 6e-8) rad = 0.02 deg.  This pipeline
                # integrates in float64, so direction gets an absolute 0.03 deg allowance.
                tol = 0.03 if mine.startswith("dir") else 2e-3 * max(abs(b), 1e-3) + 1e-4
                assert abs(a - b) <= tol, (r["horizon"], mine, a, b)
    # the network is doing something, so the two arms differ
    assert got[W][0]["pos_error"] != got[W][1]["pos_error"]


def test_npz_mode(setup, tmp_path):
    s = setup
    rng = np.random.default_rng(1)
    n = W + 9
    up = np.array([0, 0, 1.0]) + 0.05 * rng.standard_normal((n, 3))
    up /= np.linalg.norm(up, axis=-1, keepdims=True)
    z = str(tmp_path / "in.npz")
    np.savez(z, acc=(G * up).astype(np.float32), gyro=np.zeros((n, 3), np.float32),
             g_body=up.astype(np.float32))
    out = OI.main(["--onnx", s["onnx"], "--npz", z, "--out_npz", str(tmp_path / "o.npz")])
    assert out["corrected_acc"].shape == (W, 3)
    with torch.no_grad():
        ref = s["model"](torch.tensor(G * up, dtype=torch.float32)[None],
                         torch.zeros(1, n, 3), torch.tensor(up, dtype=torch.float32)[None])
    assert np.abs(ref[0][0].numpy() - out["corrected_acc"]).max() < 1e-4


def test_rejects_horizon_longer_than_window(setup):
    with pytest.raises(SystemExit):
        OI.main(["--onnx", setup["onnx"], "--config", setup["cfg"], "--csv", setup["csv"],
                 "--horizons", str(W + 100)])
