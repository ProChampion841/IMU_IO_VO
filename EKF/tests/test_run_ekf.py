"""End to end: synthetic flight log -> IMU loader + freeze -> ESKF with VO -> metrics."""
import os

import numpy as np
import pandas as pd
import pytest

import run_ekf
from ekf import pipeline as PL
from ekf import so3

IMU_CFG = os.path.join(PL.IMU_ROOT, "configs/exp/UAV/tilt_rotate.conf")
G = 9.81007


@pytest.fixture(scope="module")
def flight(tmp_path_factory):
    from test_onnx_inference import write_flight            # IMU/tests
    d = tmp_path_factory.mktemp("ekf")
    path = str(d / "2099_02_02_1_1_sensor_data.csv")
    write_flight(path, seconds=200.0, seed=4)
    df = pd.read_csv(path)
    # a slow accelerometer bias drift AFTER the freeze: 0 -> 0.06 m/s^2 on x over
    # the flight.  The 15 s freeze cannot remove it, so IMU-only dead reckoning drifts.
    ramp = 0.06 / G * (df["Time"] - df["Time"].iloc[0]) / 200.0
    df["AcclX"] += ramp
    df.to_csv(path, index=False)
    # VO predictions in the VO model's own contract: body FRD + log variance
    rng = np.random.default_rng(0)
    t = df["Time"].to_numpy()[::10]
    R_ned = so3.euler_zyx(df["GPSNavEulZ"].to_numpy()[::10], df["GPSNavEulY"].to_numpy()[::10],
                          df["GPSNavEulX"].to_numpy()[::10])
    v_ned = df[["GPSNavVnX", "GPSNavVnY", "GPSNavVnZ"]].to_numpy()[::10]
    v_frd = np.einsum("nji,nj->ni", R_ned, v_ned) + rng.standard_normal((len(t), 3)) * [0.5, 0.5, 0.2]
    lv = np.log(np.tile(np.array([0.5, 0.5, 0.2]) ** 2, (len(t), 1)))
    vo = pd.DataFrame({"time": t, "body_velocity_m_s_x": v_frd[:, 0],
                       "body_velocity_m_s_y": v_frd[:, 1], "body_velocity_m_s_z": v_frd[:, 2],
                       "velocity_log_variance_x": lv[:, 0], "velocity_log_variance_y": lv[:, 1],
                       "velocity_log_variance_z": lv[:, 2]})
    vo_path = str(d / "2099_02_02_1_1_sensor_data_vo.csv")
    vo.to_csv(vo_path, index=False)
    cfg = str(d / "ekf.json")
    c = run_ekf.load_config(os.path.join(run_ekf.HERE, "configs", "ekf_default.json"))
    c["vo"]["var_scale"] = 1.0                    # the synthetic VO noise is white
    import json
    json.dump(c, open(cfg, "w"))
    return dict(dir=d, csv=path, vo=vo_path, cfg=cfg)


def test_vo_csv_ekf_beats_imu_only(flight):
    res, rows = run_ekf.main(["--imu_config", IMU_CFG, "--csv", flight["csv"],
                              "--vo_csv", flight["vo"], "--ekf_config", flight["cfg"],
                              "--horizons", "3000", "6000",
                              "--out_csv", str(flight["dir"] / "rows.csv"),
                              "--out_npz", str(flight["dir"] / "traj.npz"),
                              "--plot_dir", str(flight["dir"] / "plots")])
    h, n, r = res["inference"][-1]
    assert h == 6000 and n >= 1
    assert r["ekf"]["pos_error"] < r["imu"]["pos_error"]
    assert r["ekf"]["vel_rmse"] < r["imu"]["vel_rmse"]
    assert r["ekf"]["vel_rmse"] < 0.5                 # VO noise 0.5 m/s, filtered
    assert os.listdir(str(flight["dir"] / "plots"))
    assert os.path.getsize(str(flight["dir"] / "traj.npz")) > 0


def test_vo_sim_runs_and_is_labelled(flight, capsys):
    res, _ = run_ekf.main(["--imu_config", IMU_CFG, "--csv", flight["csv"], "--vo_sim",
                           "--horizons", "3000", "--first_only"])
    assert "SIMULATED" in capsys.readouterr().out
    assert res["inference"][0][1] == 1


def test_onnx_block_correction_matches_imu_pipeline(flight, tmp_path):
    """One ONNX block over a window == IMU/tools/onnx_inference on the same window."""
    pytest.importorskip("onnxruntime")
    import copy
    import torch
    import torch.utils.data as Data
    from pyhocon import ConfigFactory
    from datasets import SeqeuncesDataset, collate_fcs
    from tools.export_onnx import build, export, randomise_zero_heads
    from tools.onnx_inference import OnnxModel, build_feeds
    m = build(IMU_CFG)
    randomise_zero_heads(m, seed=2)
    path = str(tmp_path / "m.onnx")
    export(m, 1000, path)
    model = OnnxModel(path)
    wins = list(PL.load_windows(IMU_CFG, "inference", 1000, csv=flight["csv"]))
    got = PL.onnx_correct(wins[0], model)[0]
    conf = ConfigFactory.parse_file(IMU_CFG)
    dc = copy.deepcopy(conf.dataset.inference)
    for e in dc.data_list:
        e["window_size"] = e["step_size"] = 1000
        e["data_drive"] = [os.path.basename(flight["csv"])]
        e["data_root"] = os.path.dirname(flight["csv"])
    ds = SeqeuncesDataset(data_set_config=dc)
    data, _, _ = next(iter(Data.DataLoader(ds, batch_size=1, collate_fn=collate_fcs["padding9"])))
    ref = model(**build_feeds(model, data, "gt"))["corrected_acc"][0]
    assert ds.index_map[0][1] == wins[0]["start"]
    assert np.abs(ref - got).max() < 1e-4
    # longer window: several blocks, real history for blocks > 0
    win2 = next(PL.load_windows(IMU_CFG, "inference", 2500, csv=flight["csv"]))
    acc2, gyro2 = PL.onnx_correct(win2, model)
    assert acc2.shape == (2500, 3) and np.isfinite(acc2).all()
