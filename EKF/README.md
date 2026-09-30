# EKF — IMU + VO fusion

An **error-state EKF (ESKF)** that propagates with the IMU (100 Hz) and corrects
with the VO model's **body velocity** and the aircraft's **attitude solution**.
It is evaluated exactly like the IMU project: each window starts from the GPS
ground-truth state after the 15 s bias freeze. The same metrics are reported
at every horizon: `vel_rmse`, `vel_max_error`, `dir_rmse`, `dir_max_error` and
`pos_error`.

```text
IMU (acc, gyro, 100 Hz) ─► [optional ONNX IMU correction] ─► ESKF predict
VO  (body velocity + log-variance, FRD) ─► FRD→FLU ─────────► ESKF update (velocity)
attitude (GPSNavEul, or MTi EulX/Y/Z) ──────────────────────► ESKF update (tilt / heading)
```

## Layout

```text
EKF/
  ekf/so3.py          rotations (exp/log/skew, Euler, FRD↔FLU)
  ekf/eskf.py         the filter: 15-state ESKF, predict + VO / attitude updates
  ekf/vo.py           VO input: read the VO model's CSV, or SIMULATE VO from GPS
  ekf/vo_onnx.py      run the VO ONNX runtime (VO/tools/onnx_inference.py) on a flight folder
  ekf/pipeline.py     windows from the IMU project, ONNX correction, the 3 arms, metrics
  run_ekf.py          command line: evaluate over flights, print table, save CSV/npz/plots
  configs/ekf_default.json   noise and aiding settings
  tests/              pytest (filter core, VO format, end-to-end on a synthetic flight)
  cmd.txt             commands
```

`EKF/` reuses the IMU project's loader, 15 s bias freeze and ONNX tools. Keep
`EKF/` and `IMU/` side by side; the code finds `../IMU` on its own.

## Frames and state

- Everything is in the IMU project's convention: world **NWU** (z up), body
  **FLU**, and gravity `[0, 0, 9.81007]`. A level accelerometer reads +g on z.
- VO outputs body velocity in **FRD**. Converting to FLU flips y and z; the
  per-axis variances are unchanged.

| | |
|---|---|
| nominal state | position p, velocity v, attitude R (body→world), accel bias b_a, gyro bias b_g |
| error state (15) | δp, δv, δθ, δb_a, δb_g — attitude error on the right: R_true = R·Exp(δθ) |
| predict | a_w = R(acc − b_a) − g; p += v·dt + ½a_w·dt²; v += a_w·dt; R = R·Exp((gyro − b_g)·dt) |
| VO update | z = Rᵀv (+ ω × lever arm); H_v = Rᵀ, H_θ = [Rᵀv]×, H_bg = [lever]× |
| attitude update | z = Log(R_meas·Rᵀ) in the world frame; rows 1–2 are tilt, row 3 is heading, and each can be switched off |

- Every update is **χ²-gated** (99.9 %) and uses the Joseph form.
- With no updates, the prediction step is exactly the IMU project's integrator.
  This is checked in `tests/test_eskf.py`.

## Inputs

**IMU** is taken from the IMU project's flight logs (`*_sensor_data.csv`),
through `--imu_config` (the IMU training config: flight lists, 15 s freeze,
attitude source). With `--imu_onnx model.onnx`, the learned IMU correction is
applied first. It runs in blocks of the model's length (e.g. 40 s): the first
block gets the same 9-sample padding as training, and later blocks get the
real 9 samples before them.

**VO** comes from the VO project's ONNX runtime, or from CSV files.

**Option 1 (recommended): run the VO ONNX model directly.** Use
`--vo_onnx export/onnx` (made by `VO/tools/export_onnx.py`) with
`--vo_dataset <flight folder>` for one flight, or `--vo_datasets <root>` for one
folder per flight (`<root>/<flight name>/flight.csv + images/`).
- The EKF replays each flight through `VO/tools/onnx_inference.py` (numpy +
  onnxruntime only).
- It keeps the ticks where a new image pair was delivered (`pair_delivered`).
  With `--frame-gap 10` and `--output-on-pairs`, that is one per 500 ms. The
  rows in between re-read the same image token, so they carry no new
  information and would make the filter over-confident.
- It uses VO's own variance, `exp(velocity_log_variance)`, and flips FRD to FLU.
- `--vo_cache_dir DIR` saves the VO output as `<flight>_vo.csv`, so the slow
  image pass runs once per flight.
- The VO `flight.csv` has the same logger columns as the IMU `*_sensor_data.csv`,
  so the two share one clock. If they don't, set `vo.time_offset_s`.
- The run prints the VO error against GPS body velocity at the VO timestamps.
  A large bias there means a clock or frame problem.

**Option 2: VO predictions already in CSV.** Use one CSV per flight
(`--vo_dir`, named `<flight>_vo.csv`), or a single file (`--vo_csv` together
with `--csv`). The CSV written by `VO/tools/onnx_inference.py --output` works
as is (`time_s, vx, vy, vz, …, pair_delivered`). It has no variance column, so
the measured VO RMSE is used instead. The accepted columns are:

| column | meaning |
|---|---|
| `time` (or `Time`, `time_s`, `t`) | seconds, **same clock as the IMU log** (use `vo.time_offset_s` otherwise) |
| `body_velocity_m_s_x/y/z` (or `vx, vy, vz`) | body velocity, **FRD** (set `vo.frame: flu` if not) |
| `velocity_log_variance_x/y/z` (or `logvar_x..`, `var_x..`, `std_x..`) | optional; variance = exp(log-variance) |

- If the file has no variance columns, the VO's measured validation RMSE is
  used: [3.5, 2.7, 0.95] m/s.
- `vo.var_scale` (default 4) inflates the variance. VO error is correlated over
  time, and a per-sample variance would make the filter over-confident.

> The VO project does not yet have a script that writes its predictions to
> this CSV. Until it does, use `--vo_sim` to test the filter with **simulated**
> VO (GPS truth + white noise + a slow Gauss-Markov error sized from the
> measured VO RMSE). Simulated numbers test the filter; they are **not** a VO
> result, and the output says so.

**Attitude aid** (`attitude_aid` in the config):

| source | what it uses |
|---|---|
| `gt` (default) | GPSNavEul, the attitude the IMU model is trained with (`gtrot`) |
| `mti` | EulX/Y/Z. Use tilt only (`std_yaw_deg: null`): its heading drifts |
| `null` | no attitude aid |

With body-velocity VO and no heading aid, heading and position are not
observable, and they will drift.

## Run (from the `EKF` folder)

```bash
# the VO ONNX model on one flight (VO flight folder + the IMU log of the same flight)
python run_ekf.py --imu_config ../IMU/configs/exp/UAV/tilt_rotate.conf \
    --csv 2026_02_06_143_23_sensor_data.csv \
    --vo_onnx ../VO/export/onnx --vo_dataset ../VO/data_split/test \
    --vo_cache_dir vo_cache --horizons 3000 6000 12000 --plot_dir ekf_plots

# simulated VO: check the filter on your real IMU flights
python run_ekf.py --imu_config ../IMU/configs/exp/UAV/tilt_rotate.conf \
    --splits inference --vo_sim --horizons 3000 6000 12000

# real VO predictions, one CSV per flight, with the learned IMU correction
python run_ekf.py --imu_config ../IMU/configs/exp/UAV/tilt_rotate.conf \
    --splits inference --vo_dir vo_predictions --imu_onnx ../IMU/tilt_rotate_40s.onnx \
    --horizons 3000 6000 12000 18000 24000 30000 \
    --out_csv ekf_windows.csv --plot_dir ekf_plots
```

The output has one row per horizon. Each metric shows `imu / vo / ekf` and the
ratio **ekf / imu** (below 1 means VO helped). It also reports:
- how many VO updates were accepted or gated out
- the mean VO NIS (about 3 when the VO variance is right; above that means the
  VO variance is too small, below means it is too big)
- the final bias estimates

## Tests

```bash
python -m pytest tests -q
```

| test | what it checks |
|---|---|
| prediction | equals the IMU project's integrator to 1e-9 |
| filter on a simulated flight | VO bounds the velocity, NEES is consistent, the accel bias is pulled toward the truth |
| gating, tilt-only | gating rejects outliers; the tilt-only aid leaves heading alone |
| VO file | contract names, FRD→FLU, log-variance, time offset |
| end-to-end | synthetic flight log with a bias drift: EKF beats IMU-only; ONNX block correction equals the IMU project's ONNX pipeline |

## Tuning order

1. `vo.var_scale`: bring the VO NIS mean toward 3.
2. `attitude_aid.std_*`: tighter means trusting the nav attitude more.
3. `eskf.acc_bias_rw` / `gyro_bias_rw`: how fast the biases may move.
4. `eskf.acc_noise` / `gyro_noise`: IMU white noise.
