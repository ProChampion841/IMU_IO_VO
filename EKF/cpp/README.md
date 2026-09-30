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
- **Memory:** fixed-size arrays only. The one exception is a small queue of
  pending measurements.

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

State s = ekf.state();       // s.t, s.p, s.v, s.R (s.q()), s.ba, s.bg, s.P (15x15)
```

- **Timestamps:** every message carries its own timestamp, all on the **same
  clock** as the IMU.
- **Late measurements:** VO or GPS that arrives after newer IMU samples is
  applied at once. If it is older than `stream.max_meas_age_s` it is dropped.
- **VO variance:** `s` is `vo.var_scale` from the config. VO variance =
  exp(velocity_log_variance), as output by the VO ONNX runtime.
- **Threading:** the filter is not thread-safe. Call it from one thread,
  e.g. the IMU callback thread, and push VO results into it through a queue.
- **IMU gaps:** a gap longer than `stream.max_imu_gap_s` is not integrated. The
  state coasts and its uncertainty grows. `counters().imu_gap` counts these.

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
