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
land matching (absolute position, NWU m, may arrive late) ──► ESKF update (position, rewind)
```

## Layout

```text
EKF/
  ekf/so3.py          rotations (exp/log/skew, Euler, FRD↔FLU)
  ekf/eskf.py         the filter: 15-state ESKF, predict + VO / attitude updates
  ekf/vo.py           VO input: read the VO model's CSV, or SIMULATE VO from GPS
  ekf/vo_onnx.py      run the VO ONNX runtime (VO/tools/onnx_inference.py) on a flight folder
  ekf/pipeline.py     windows from the IMU project, ONNX correction, the 3 arms, metrics
  ekf/stream.py       stream (real-time) front end: messages in arrival order
  ekf/events.py       message logs: build from a flight, write/read, replay
  ekf/landmatch.py    land-matching position fixes: simulate, read CSV, lat/lon → NWU
  run_ekf.py          OFFLINE evaluation: windows from GPS truth, imu / vo / ekf table
  run_stream.py       STREAM test: GPS then outage, EKF vs IMU-only, optional C++ check
  cpp/                C++17 stream EKF for the Jetson (see cpp/README.md)
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
| position update | z = p + R·lever (world NWU, m); H_p = I, H_θ = −R[lever]×; north/west rows only unless `pos_use_vertical` |

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

## No ground truth, no reset (`run_stream.py --no_gt`)

This is pure inference, the way the aircraft would run with no GPS at all. The
whole flight is **one continuous run**: nothing is reset, and no GPS truth enters
the filter.
- **Position** starts at 0. It is scored on the distance travelled since the start.
- **Velocity** starts as the nav attitude × the first VO velocity. Its initial
  uncertainty comes from that VO sample's variance.
- **Attitude** comes from the nav solution (the attitude aid, as everywhere else).
- **Biases** start at 0 and are learned from VO and attitude.

GPS truth is used **only to score** the run. Horizons are counted from the start.
On the Jetson, the matching call is
`ekf.initialize(t, {0,0,0}, R_nav * v_vo, R_nav, pos_std, vel_std, att_std_deg)`.

```bash
python run_stream.py --imu_config ../IMU/configs/exp/UAV/tilt_rotate.conf \
    --splits inference --vo_dir vo_cache --no_gt \
    --horizons 30s 1m 2m 3m 4m 5m 10m 15m 20m 30m 40m --out_csv no_gt.csv
```

## IMU input in the stream: raw or the learned model (`--imu raw|model`)

`run_stream.py --imu model --imu_onnx IMU/tilt_rotate_40s.onnx` feeds the EKF the
learned IMU correction instead of the raw IMU. Both arms (ekf and imu-only) get
the same IMU, so the comparison stays fair.

It runs **causally**, as it would on the aircraft (`ekf/imu_model.py`):
- **Sliding window:** every `--imu_every` samples (default 10, i.e. 10 Hz) the model
  runs on the **latest** 40 s of IMU. Nothing after "now" is used.
- **Look-ahead delay:** the model's CNN looks about 12 samples ahead, so a corrected
  sample is released only once a run has seen `--imu_delay` (16) samples after it.
  The EKF therefore runs about 0.16–0.26 s behind real time. VO and attitude
  messages wait in its queue until the IMU catches up; every message keeps its own
  timestamp.
- **Warm-up:** raw IMU is used until the first full window exists (40 s for a
  40 s model). Export a shorter model (`--frames 1000`) for a shorter warm-up.
- **Bias freeze:**
  - With GPS (`--gps_s`), the model is used from the outage on, with the 15 s
    pre-outage bias freeze it was trained with (fitted by the IMU project's own
    function).
  - With `--no_gt` there is no GPS, so the model gets **no freeze**. That is a
    known mismatch with training.

Tests (`tests/test_imu_model.py`):
- the corrector is causal: changing future data never changes a released sample
- no sample is lost or reordered, and each is released 16–26 samples late
- the full `run_stream.py --imu model` run works in GPS and no-GT modes, and the
  C++ EKF matches Python on it
- the **C++ corrector matches Python** (max difference 1.2e-7)

## Land matching: absolute position fixes (`--pos_sim`, `--pos_csv`, `--pos_dir`)

VO and attitude never observe **position**, so without an absolute fix position
only drifts. A land-matching (map-matching) fix corrects it directly. Through the
filter's correlations it also corrects velocity, makes **heading** observable once
the aircraft turns or accelerates, and keeps the IMU biases learnable during the
GPS outage.

```bash
# simulated fixes (GPS truth + 10 m noise, 1 Hz, 0.5 s late): a FILTER test
python run_stream.py --imu_config ../IMU/configs/exp/UAV/tilt_rotate.conf \
    --splits inference --vo_dir vo_cache --pos_sim --gps_s 60 \
    --horizons 30s 1m 2m 5m 10m --out_csv stream_landmatch.csv

# your matcher's fixes, one CSV per flight (<flight>_pos.csv); VO is optional
python run_stream.py --imu_config ../IMU/configs/exp/UAV/tilt_rotate.conf \
    --splits inference --pos_dir landmatch --pos_origin 37.123456 127.123456 50.0
```

The fixes go to the **ekf** arm only, so the table reads "ekf (VO + land match) /
imu-only". The run also prints how many fixes were accepted, gated, rewound,
dropped, or caused a reset, and the NIS mean of the accepted fixes (2 with
north/west only; above that, raise the fix std).

**Fix CSV** (`ekf/landmatch.py`, one row per fix; column names are not case-sensitive):

| column | meaning |
|---|---|
| `time` (or `t`, `time_s`) | when the **image** was taken, same clock as the IMU (`--pos_time_offset` otherwise) |
| `arrival` (optional) | when the matcher delivered the fix; else `time + --pos_latency_s` |
| `north, west[, up]` or `x, y[, z]` | position in the filter's NWU frame, m |
| `lat, lon[, alt]` | WGS-84 degrees instead; needs `--pos_origin LAT LON [ALT]`, the point where the filter's position is 0 |
| `std`, or `std_x/y/z`, or `var_x/y/z` (optional) | the fix's uncertainty; else `position_aid.pos_std_m`; × `pos_var_scale` |
| `valid` (optional) | rows with 0 are skipped |

**The fix must be where the aircraft (or camera) is.** If the matcher reports
the ground point at the image centre instead, that point is off by height above
ground × tan(tilt) (100 m and 5°: 8.7 m). Convert it before feeding the filter.
A camera offset from the IMU goes in `position_aid.pos_lever_arm_m`.

**Frame.** The filter works in local NWU metres. In this project position 0 is
the start of the log (truth = integrated GPS velocity), so `--pos_origin` must be
the GPS position at that moment. On the aircraft, pick any origin, convert fixes
with `geodetic_to_nwu` / C++ `geodeticToNwu`, and start the filter at the first fix.

**Late fixes.** Matching takes time. A fix applied to the current state would
be wrong by speed × delay (25 m/s × 0.5 s = 12.5 m). The filter therefore keeps
the last `pos_replay_s` (3 s) of its history. A late fix **rewinds** to its image
time, is applied there, and everything since is re-run. The result is
bit-identical to an on-time fix (`tests/test_position.py`). A fix older than
`pos_replay_s` is dropped; raise it if your matcher is slower. The cost is about
0.6 MB of history in C++.

**Wrong matches and lock-out.**
- Every fix is χ²-gated (99.9 %), so a wrong match over repetitive terrain is
  rejected.
- After a long outage the filter can be confidently wrong, and then it would
  reject every good fix. When `pos_reset_after` (3) fixes in a row are rejected
  **and agree with each other** (within `pos_reset_agree_sigma` = 3 σ), the
  position is reset to the fix. Wrong matches do not agree, so they never reset.
  Set `pos_reset_after: 0` to disable this.

| `position_aid` key | default | meaning |
|---|---|---|
| `pos_use_vertical` | false | use the fix height too (matchers rarely give a usable one) |
| `pos_lever_arm_m` | [0, 0, 0] | IMU → the point the fix refers to (camera), body FLU, m |
| `pos_replay_s` | 3.0 | history kept for late fixes; 0 = apply late fixes at once |
| `pos_reset_after`, `pos_reset_agree_sigma` | 3, 3.0 | lock-out reset (see above) |
| `pos_std_m`, `pos_var_scale` | 10.0, 1.0 | std for a CSV without one; inflation of the variance (matching errors are correlated, like VO's) |

`pos_sim` sets the simulated fixes: rate, white and slow (Gauss-Markov) error,
latency, and a fraction of wrong matches. `--pos_rate_hz`, `--pos_std_m` and
`--pos_latency_s` override it from the command line.

## Horizons 30 s … 40 min

Both scripts take `--horizons 30s 1m 2m 3m 4m 5m 10m 15m 20m 30m 40m`, which is
also the default. A plain number still means frames at 100 Hz in `run_ekf.py`
and seconds in `run_stream.py`. A horizon is scored only on flights long enough
for it; the others are listed as "no window / no flight long enough".

| script | how long horizons are scored | use |
|---|---|---|
| `run_ekf.py --per_horizon --step 1m` | each horizon on its own windows. Starts every minute (overlapping) so a 10 min horizon fits an 11–12 min flight | offline comparison imu / vo / ekf |
| `run_stream.py` (many flights) | one outage per flight after `--gps_s` of GPS; every horizon that fits in the rest of the log | real use: EKF vs IMU-only |

Without `--per_horizon`, `run_ekf.py` reads every horizon off one window as long
as the longest horizon. Then only flights longer than 40 min count at all.

Long horizons are slow in Python: about 1.5 min per 17-minute flight in
`run_stream.py`, and a few minutes offline with `--step 1m`. The C++ build runs
at about 2.6 µs per message.

## Stream mode (real use) and C++

The offline evaluator above starts every window from GPS truth. On the
aircraft the filter works like this instead:
- it runs **continuously**, fed message by message in arrival order
  (`ekf/stream.py`);
- it is corrected by **GPS velocity while GPS is up**, which is where it learns
  the IMU biases (no offline freeze);
- it continues on IMU + attitude + VO when GPS is lost.

```bash
python run_stream.py --csv <flight>_sensor_data.csv --vo_onnx ../VO/export/onnx \
    --vo_dataset <VO flight folder> --gps_s 60 --horizons 30s 1m 2m 5m 10m \
    --events_out events.csv --cpp cpp/build/ekf_replay
```

It prints the position, velocity and direction error at each horizon into the
outage, for the EKF and for IMU-only. It also prints the biases learned before
the outage and the VO NIS. With `--cpp` it runs the C++ build on the same
message log and checks it matches.

- **Stream vs offline:** on the same data they are **identical** (difference
  0.0, `tests/test_stream.py`).
- **C++:** `cpp/` is the Jetson port. See `cpp/README.md`.

**Correlated VO error.** VO errors that persist for tens of seconds (a slowly
changing velocity offset) look like real motion to a filter that treats each VO
sample as independent. The filter then follows them.
- A too-low NIS does not reveal this, because each sample on its own looks
  consistent.
- With simulated VO sized like the measured VO error (3.5 / 2.7 / 1 m/s, half
  of it slow) on a synthetic flight with a nearly perfect IMU, the EKF ended up
  worse than IMU-only.
- On real flights the IMU drifts far more: about 123 m at 60 s and 358 m at
  120 s after the freeze. There VO has much more room to help.
- **Always compare `ekf` against `imu-only` on real flights before using VO.**
  If VO hurts, first raise `vo.var_scale`. The real fix is VO-bias states in the
  filter (a Gauss-Markov velocity offset), which is the next step.

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
| stream (`test_stream.py`) | stream = offline exactly; real use (GPS then outage, VO 0.1 s late, 1 % IMU dropped, a 0.5 s IMU dropout): biases learned, VO beats IMU-only; **C++ = Python** on the same message log |
| VO ONNX (`test_vo_onnx.py`) | only delivered pairs are kept, FRD→FLU, variance = exp(log-variance), cache round trip |
| land matching (`test_position.py`) | update and lever-arm Jacobian; a fix 0.5 / 0.9 s late gives **exactly** the on-time result; fixes older than the history are dropped; fixes bound the drift and wrong matches are gated; the lock-out reset; CSV and lat/lon input; **C++ = Python** with late fixes, wrong matches and a reset |

## Tuning order

1. `vo.var_scale`: bring the VO NIS mean toward 3.
2. `attitude_aid.std_*`: tighter means trusting the nav attitude more.
3. `eskf.acc_bias_rw` / `gyro_bias_rw`: how fast the biases may move.
4. `eskf.acc_noise` / `gyro_noise`: IMU white noise.
