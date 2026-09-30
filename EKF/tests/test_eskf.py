"""Filter core: mechanization, observability, consistency.  NumPy only."""
import numpy as np
import pytest

from ekf import so3
from ekf.eskf import ESKF, ESKFParams

G = 9.81007


def test_so3_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = rng.standard_normal(3)
        phi = u / np.linalg.norm(u) * rng.uniform(0, 3.0)     # |phi| < pi: log's range
        assert np.allclose(so3.log(so3.exp(phi)), phi, atol=1e-8)
        R = so3.exp(rng.standard_normal(3) * 2.0)              # any rotation
        assert np.allclose(so3.exp(so3.log(R)), R, atol=1e-8)
    assert np.allclose(so3.exp(np.zeros(3)), np.eye(3))


def test_propagation_is_the_imu_projects_integrator():
    """No updates: ESKF.predict == IMU/tools/onnx_inference.integrate_np when that
    integrator removes gravity with the same (pre-step) attitude."""
    from tools.onnx_inference import integrate_np
    rng = np.random.default_rng(1)
    F = 500
    acc = rng.standard_normal((F, 3)) + [0, 0, G]
    gyro = 0.2 * rng.standard_normal((F, 3))
    dt = np.full(F, 0.01)
    R0 = so3.exp(rng.standard_normal(3) * 0.3)
    p0, v0 = rng.standard_normal(3), rng.standard_normal(3) * 5
    f = ESKF(p0, v0, R0)
    pos, vel, Rpre = [], [], []
    for k in range(F):
        Rpre.append(f.R.copy())
        f.predict(acc[k], gyro[k], dt[k])
        pos.append(f.p.copy()); vel.append(f.v.copy())
    P, V, _ = integrate_np(acc[None], gyro[None], dt[None, :, None], p0[None], v0[None],
                           R0[None], np.array(Rpre)[None], G)
    assert np.abs(P[0] - np.array(pos)).max() < 1e-9
    assert np.abs(V[0] - np.array(vel)).max() < 1e-9


def _truth(T=120.0, dt=0.01):
    """Coordinated turns at 20 m/s in NWU/FLU, analytic IMU."""
    t = np.arange(0.0, T, dt)
    yaw = 0.3 * np.sin(0.04 * t) + 0.02 * t
    V = 20.0 + np.sin(0.05 * t)
    pitch = 0.03 * np.sin(0.1 * t)
    roll = np.arctan(V * np.gradient(yaw, t) / G)
    R = so3.euler_zyx(yaw, pitch, roll)
    v = np.stack([V * np.cos(yaw) * np.cos(pitch), V * np.sin(yaw) * np.cos(pitch),
                  V * np.sin(pitch)], 1)
    p = np.cumsum(np.vstack([np.zeros((1, 3)), 0.5 * (v[1:] + v[:-1]) * dt]), 0)
    a_w = np.gradient(v, t, axis=0)
    acc = np.einsum("nji,nj->ni", R, a_w + [0, 0, G])
    gyro = np.array([so3.log(R[k].T @ R[k + 1]) / dt for k in range(len(t) - 1)])
    return t, R, v, p, acc[:-1], gyro


@pytest.mark.parametrize("seed", [0, 1])
def test_vo_updates_bound_velocity_and_filter_is_consistent(seed):
    rng = np.random.default_rng(seed)
    t, R, v, p, acc, gyro = _truth()
    ba_true, bg_true = np.array([0.04, -0.03, 0.05]), np.radians([0.02, -0.03, 0.01])
    prm = ESKFParams()
    dt = 0.01
    acc_m = acc + ba_true + rng.standard_normal(acc.shape) * prm.acc_noise / np.sqrt(dt)
    gyro_m = gyro + bg_true + rng.standard_normal(gyro.shape) * prm.gyro_noise / np.sqrt(dt)
    vo_std = np.array([1.0, 1.0, 0.5])
    runs = {}
    for use_vo in (False, True):
        f = ESKF(p[0], v[0], R[0], params=prm)
        nees, verr = [], []
        for k in range(len(acc)):
            f.predict(acc_m[k], gyro_m[k], dt)
            if (k + 1) % 10 == 0:
                Ra = R[k + 1] @ so3.exp(rng.standard_normal(3) * np.radians([0.2, 0.2, 0.5]))
                f.update_attitude(Ra, 0.2, 0.5)
                if use_vo:
                    z = R[k + 1].T @ v[k + 1] + rng.standard_normal(3) * vo_std
                    f.update_body_velocity(z, vo_std ** 2)
            e = f.v - v[k + 1]
            verr.append(np.linalg.norm(e))
            if use_vo and k > 1000:
                nees.append(e @ np.linalg.solve(f.P[3:6, 3:6], e))
        runs[use_vo] = (np.array(verr), np.array(nees), f)
    verr_imu, _, _ = runs[False]
    verr_ekf, nees, f = runs[True]
    assert verr_ekf[-1] < 1.0 < verr_imu[-1]                   # VO bounds the velocity
    assert np.sqrt((verr_ekf[2000:] ** 2).mean()) < 0.5 * np.sqrt((verr_imu[2000:] ** 2).mean())
    assert 0.3 < nees.mean() < 9.0                             # 3 dof: ~3 when consistent
    # the accelerometer bias becomes observable and is pulled toward the truth
    assert np.linalg.norm(f.ba - ba_true) < 0.5 * np.linalg.norm(ba_true)
    assert f.stats["vel"][1] <= 0.01 * f.stats["vel"][0]       # almost nothing gated


def test_gating_rejects_outlier():
    f = ESKF(np.zeros(3), np.array([20.0, 0, 0]), np.eye(3))
    ok, nis = f.update_body_velocity([20.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert ok
    ok, nis = f.update_body_velocity([80.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert not ok and nis > 16.266
    assert abs(f.v[0] - 20.0) < 0.5


def test_tilt_only_attitude_aid_leaves_heading_unobserved():
    f = ESKF(np.zeros(3), np.zeros(3), np.eye(3))
    P_yaw0 = f.P[8, 8]
    R_meas = so3.exp([0.0, 0.0, 0.3])          # pure heading difference
    f.update_attitude(R_meas, std_tilt_deg=0.2, std_yaw_deg=None)
    assert np.allclose(f.R, np.eye(3), atol=1e-9)              # heading not pulled
    assert abs(f.P[8, 8] - P_yaw0) < 1e-12
