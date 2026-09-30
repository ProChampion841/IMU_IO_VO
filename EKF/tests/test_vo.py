import numpy as np

from ekf.vo import load_vo_csv, simulate_vo
from ekf import so3


def test_vo_csv_contract_names_frd_to_flu(tmp_path):
    p = tmp_path / "f_vo.csv"
    p.write_text("time,body_velocity_m_s_x,body_velocity_m_s_y,body_velocity_m_s_z,"
                 "velocity_log_variance_x,velocity_log_variance_y,velocity_log_variance_z\n"
                 "10.0,20.0,1.0,-0.5,0.0,1.0,-2.0\n"
                 "10.1,21.0,2.0,0.5,0.0,1.0,-2.0\n")
    s = load_vo_csv(str(p), time_offset=0.5, var_scale=1.0, fresh_only=False)
    assert np.allclose(s.t, [10.5, 10.6])
    assert np.allclose(s.v[0], [20.0, -1.0, 0.5])              # y, z flipped
    assert np.allclose(s.var[0], np.exp([0.0, 1.0, -2.0]))     # variance unchanged
    assert len(s.window(10.5, 10.6)) == 1                       # (t0, t1]


def test_vo_csv_without_variance_uses_measured_rmse(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("t,vx,vy,vz\n0,1,2,3\n")
    s = load_vo_csv(str(p), frame="flu")
    assert np.allclose(s.v[0], [1, 2, 3])
    assert np.allclose(np.sqrt(s.var[0]), [3.5, 2.7, 0.95])


def test_simulated_vo_is_truth_plus_error():
    t = np.arange(0, 60, 0.01)
    R = np.tile(so3.exp([0, 0, 0.5]), (len(t), 1, 1))
    v = np.tile([15.0, 5.0, 0.0], (len(t), 1))
    s = simulate_vo(t, R, v, rate_hz=10, white_std=0.0, bias_std=0.0)
    assert np.allclose(s.v, R[0].T @ v[0], atol=1e-9)
    assert abs(len(s) - 600) <= 1


def test_fresh_rows_only(tmp_path):
    """One row per image pair: by visual_age, or by min_interval_s without it."""
    t = np.arange(0, 10, 0.05)                         # 20 Hz telemetry ticks
    body = "".join("%.2f,20,0,0,%d\n" % (x, i % 10) for i, x in enumerate(t))
    p = tmp_path / "a.csv"
    p.write_text("time,vx,vy,vz,visual_age\n" + body)
    s = load_vo_csv(str(p))
    assert len(s) == 20 and np.allclose(np.diff(s.t), 0.5)
    assert "visual_age == 0" in s.source
    q = tmp_path / "b.csv"
    q.write_text("time,vx,vy,vz\n" + "".join("%.2f,20,0,0\n" % x for x in t))
    s = load_vo_csv(str(q))                            # no age column: 0.5 s spacing
    assert len(s) == 20 and np.allclose(np.diff(s.t), 0.5)
    assert len(load_vo_csv(str(q), fresh_only=False)) == len(t)


def test_simulated_vo_default_is_one_pair_per_500ms():
    t = np.arange(0, 30, 0.01)
    R = np.tile(np.eye(3), (len(t), 1, 1))
    s = simulate_vo(t, R, np.tile([20.0, 0, 0], (len(t), 1)))
    assert np.allclose(np.diff(s.t), 0.5)
