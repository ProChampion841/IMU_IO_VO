# imuvo_ekf — C++ stream EKF for the Jetson

This is a C++17 port of `EKF/ekf/eskf.py` and `EKF/ekf/stream.py`, with **no
dependencies** (no Eigen, no Boost). It is tested against the Python version on
the same message log: max difference 5e-10 m over 20,000 states, which is the
CSV print precision.

```text
cpp/
  include/imuvo_ekf.hpp   the whole API (Params, AidParams, StreamEKF, frame converters)
  src/imuvo_ekf.cpp       implementation
  tools/ekf_replay.cpp    replay a message log -> states CSV (test harness + main-loop template)
  CMakeLists.txt          static library imuvo_ekf + ekf_replay
```

## Build (Jetson or any Linux)

```bash
cd EKF/cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/ekf_replay events.csv states.csv ../configs/ekf_default.json
```

- **Jetson (aarch64):** the stock JetPack gcc works.
- **Speed:** on x86 it takes about **2.6 µs per message**. At 100 Hz IMU that is
  under 0.1 % of one core.
- **Memory:** fixed-size arrays, plus a small queue of pending measurements and
  the history kept for late position fixes: about 2 KB per IMU sample over
  `pos_replay_s`, i.e. about 0.6 MB for 3 s at 100 Hz. Set `pos_replay_s: 0` to
  keep no history; late fixes are then applied at once.

To use it in your application, link the static library:

```cmake
add_subdirectory(path/to/EKF/cpp imuvo_ekf)
target_link_libraries(your_app PRIVATE imuvo_ekf)
```

## API

```cpp
#include "imuvo_ekf.hpp"
using namespace imuvo;

Params prm; AidParams aid;
loadConfig("ekf_default.json", prm, aid);        // same JSON as the Python side
StreamEKF ekf(prm, aid);

// 1. start: from the nav solution / GPS (position, velocity, attitude)
ekf.initialize(t, p_nwu, v_nwu, attitudeFromNavEuler(roll, pitch, yaw));

// 1b. OR with NO ground truth: position 0, velocity from the first VO pair
//     rotated by the nav attitude, its own initial 1-sigma (m, m/s, deg)
Mat3 Rn = attitudeFromNavEuler(roll, pitch, yaw);
Vec3 vb = frdToFlu(vo_velocity_frd), v0;          // v0 = Rn * vb
for (int i = 0; i < 3; ++i) v0[i] = Rn[3*i]*vb[0] + Rn[3*i+1]*vb[1] + Rn[3*i+2]*vb[2];
ekf.initialize(t, {0, 0, 0}, v0, Rn, 1e-3, std::sqrt(max_vo_var), 1.0);

// 2. every IMU sample (100 Hz), raw logger units converted to SI FLU
ekf.onImu(t, accFromLogger({AcclX, AcclY, AcclZ}),      // g, FRD  -> m/s^2 FLU
             gyroFromLogger({GyroX, GyroY, GyroZ}));    // deg/s, FRD -> rad/s FLU

// 3. attitude from the nav solution (every tick is fine; every 10th is used)
ekf.onAttitude(t, attitudeFromNavEuler(GPSNavEulX, GPSNavEulY, GPSNavEulZ));

// 4. while GPS is up: world velocity (this is where the IMU biases are learned)
ekf.onGpsVelocity(t, velocityFromNed({VnX, VnY, VnZ}), {0.01, 0.01, 0.01});

// 5. VO: only when a new image pair was delivered (pair_delivered / emitted)
ekf.onVo(t_vo, frdToFlu(vo_velocity_frd), {exp(lv_x)*s, exp(lv_y)*s, exp(lv_z)*s});

// 6. land matching: absolute position, stamped with the IMAGE time; it may arrive
//    late (up to position_aid.pos_replay_s) -- the filter rewinds to t_image
Vec3 p = geodeticToNwu(lat, lon, alt, lat0, lon0, alt0);   // origin = filter position 0
ekf.onPosition(t_image, p, {std_m * std_m, std_m * std_m, std_m * std_m});

State s = ekf.state();       // s.t, s.p, s.v, s.R (s.q()), s.ba, s.bg, s.P (15x15)
```

- **Timestamps:** every message carries its own timestamp, all on the **same
  clock** as the IMU.
- **Late position fixes:** a fix that arrives after newer IMU samples rewinds the
  filter to its image time and re-runs the history since (`pos_replay_s`, 3 s).
  The result is the same as if it had arrived on time. `counters().pos_late`,
  `pos_gated`, `pos_dropped` and `pos_reset` count what happened to the fixes.
- **Late measurements:** VO or GPS that arrives after newer IMU samples is
  applied at once. If it is older than `stream.max_meas_age_s` it is dropped.
- **VO variance:** `s` is `vo.var_scale` from the config. VO variance =
  exp(velocity_log_variance), as output by the VO ONNX runtime.
- **Threading:** the filter is not thread-safe. Call it from one thread,
  e.g. the IMU callback thread, and push VO results into it through a queue.
- **IMU gaps:** a gap longer than `stream.max_imu_gap_s` is not integrated. The
  state coasts and its uncertainty grows. `counters().imu_gap` counts these.

## The learned IMU model in C++ (optional, needs ONNX Runtime)

`include/imuvo_imu_model.hpp` / `src/imuvo_imu_model.cpp` is the C++ version of
`ekf/imu_model.py`: the IMU model run causally on a sliding window.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_DIR=/opt/onnxruntime
cmake --build build -j        # + library imuvo_imu_model, tool imu_correct_replay
```

- `ONNXRUNTIME_DIR` is an unpacked ONNX Runtime C++ release, with `include/` and
  `lib/`. On the Jetson use the aarch64 build (e.g. `onnxruntime-linux-aarch64-*`,
  or the Jetson Zoo package).
- Without `ONNXRUNTIME_DIR`, only the dependency-free EKF is built.

```cpp
#include "imuvo_imu_model.hpp"
imuvo::StreamImuCorrector corr("tilt_rotate_40s.onnx", /*every=*/10, /*delay=*/16);
corr.setActive(false);                          // raw while GPS is up ...
// at the outage: corr.setFreeze(b_acc, b_gyro); corr.setActive(true);
for each raw IMU sample (SI, FLU) with the nav attitude R_nav at that sample:
    for (const auto& s : corr.push(t, acc, gyro, R_nav))
        ekf.onImu(s.t, s.acc, s.gyro);          // corrected, in order, ~0.16-0.26 s late
```

- **Speed:** the 40 s model takes **7.5 ms per run** on one x86 CPU thread. At
  10 runs per second that is about 8 % of a core. Expect a few times slower on
  the Jetson CPU. To reduce the load, raise `every`, or use the CUDA / TensorRT
  execution provider.
- **Parity:** C++ matches the Python corrector to 1.2e-7 (`tests/test_imu_model.py`,
  run with `ONNXRUNTIME_DIR` set).
- **Freeze:** the 15 s bias freeze (`setFreeze`) is computed by the IMU project's
  Python `freeze_biases`. It is not ported to C++ yet. Without it, pass zeros; that
  is what `--no_gt` does.

## Frames

The EKF runs in world **NWU** / body **FLU**, the same as the IMU project. The
converters in the header map the raw logger to that convention:

| logger | converter | EKF |
|---|---|---|
| AcclX/Y/Z [g], FRD | `accFromLogger` | m/s², FLU |
| GyroX/Y/Z [deg/s], FRD | `gyroFromLogger` | rad/s, FLU |
| GPSNavEulX/Y/Z [rad], FRD→NED | `attitudeFromNavEuler` | R body FLU → world NWU |
| GPSNavVnX/Y/Z [m/s], NED | `velocityFromNed` | NWU |
| VO body velocity, FRD | `frdToFlu` | FLU |

## Checking the build on your data

```bash
cd EKF
python run_stream.py --csv <flight>_sensor_data.csv --vo_onnx ../VO/export/onnx \
    --vo_dataset <VO flight folder> --events_out events.csv --cpp cpp/build/ekf_replay
```

This writes the message log, runs the Python stream filter and the C++ binary on
it, and prints `C++ vs Python ... PASS`. Run it once on the Jetson itself after
building there.

## What stays outside this library

- **The learned IMU correction** (`IMU/tools/export_onnx.py`) and **the VO model**
  (`VO/tools/export_onnx.py`) are ONNX files. On the Jetson, run them with ONNX
  Runtime or TensorRT and feed their outputs to `onImu` / `onVo`.
- **The VO runtime logic** (`VO/tools/onnx_inference.py`: image-pair timing,
  latency, held tokens) is Python today. Port it next to this library for a
  fully C++ pipeline.
