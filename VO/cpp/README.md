# VO streaming runtime (C++)

A C++17 port of `tools/onnx_inference.py` (`VOOnnxRuntime`) for live use. It
feeds camera frames and telemetry rows into the exported model
(`frontend.onnx`, `temporal_step.onnx`, `vo_onnx.json`) and returns the body
velocity on every row. It follows every rule of the Python runtime:
- which frames pair (`frame_gap`, `pair_stride`);
- when a pair is delivered (`deployment_latency_s` after its second exposure);
- frame-gap and telemetry-hole rejection;
- attitude/altitude interpolation at the exposures;
- the aiding vector, visual age and held geometric velocity;
- skipped non-finite rows;
- image preprocessing (Pillow-identical RGB/L conversion and bilinear resize,
  or OpenCV undistortion when the calibration has lens distortion).

## Build

Needs ONNX Runtime (an unpacked release: `include/` + `lib/`) and OpenCV
(core, imgproc, imgcodecs, and calib3d on OpenCV 4 or 3d on OpenCV 5).

    cmake -S cpp -B build -DONNXRUNTIME_DIR=/opt/onnxruntime-linux-aarch64-<ver>
    cmake --build build -j

The build creates three things:
- `libvo_stream.a`: link it into your application. The header is `include/vo_stream.hpp`.
- `vo_replay`: replays a flight folder or a recorded event log, and serves as an integration example.
- `vo_preprocess`: dumps images the way the runtime sees them.

The library is compiled with `-ffp-contract=off`. Without it, GCC fuses
multiply-adds wherever the CPU has FMA (always on aarch64/Jetson), so the
inputs would no longer match the Python runtime bit for bit. Keep that flag
if you build the sources another way.

## Use

    #include "vo_stream.hpp"

    vo::StreamRuntime vo("export/onnx");          // the export folder
    // camera thread, every frame (telemetry clock; frames in capture order):
    vo::ImageView view{data, width, height, 3, stride, /*bgr=*/true};
    vo.addFrame(view, capture_time_s);
    // telemetry thread, every row (angles in radians, same columns as training):
    vo::TickOutput out = vo.addTelemetry(t, roll, pitch, yaw, relative_alt_m);
    if (out.emitted) use(out.velocity);           // m/s, body x fwd, y right, z down
    // out.output always holds the last emitted velocity

- **Frames**: 8-bit gray, RGB/BGR, or RGBA/BGRA (alpha ignored), with any row
  stride. A frame must be the calibration's native size; otherwise `addFrame`
  throws `std::invalid_argument` and the frame is not added. Add the
  camera-to-telemetry offset (`settings().image_time_offset_s`) to the capture
  time first.
- **Late frames**: add a frame before its pair is due (capture time +
  `deployment_latency_s`). A later frame is still used, but its pair is
  delivered on the next row, as the Python runtime does, and counted in
  `stats().late_pairs`.
- **Threads**: you may call `addFrame` and `addTelemetry` from different
  threads. A frame is preprocessed outside the lock, and only if a pair will
  use it. `addFrame(time, loader)` decodes lazily: with a pair stride, most
  frames are never decoded.
- **Async frontend** (default `Options::async_frontend = true`): each pair's
  frontend runs in a worker thread as soon as a telemetry row reaches its
  second exposure. The result is ready long before the pair is due. On the due
  row the inputs are recomputed exactly as the synchronous path does; if they
  differ (only possible when telemetry time goes backwards), the frontend runs
  again. Async and sync output is byte-identical.
- **Bad rows**: a non-finite row is skipped (`out.skipped`), and the recurrent
  state is untouched.
- **Restart**: `reset()` gives a cold start.
- **Tracing**: `Options::trace_path` writes every tensor sent to and returned by
  ONNX Runtime (CRC-32 plus values) for parity checks.
- **Telemetry**: ONNX Runtime's telemetry is disabled.

## Replay

    build/vo_replay export/onnx --dataset data_split/test --output series.csv
    build/vo_replay export/onnx --events session.log --realtime

- `--dataset` replays a flight folder exactly like
  `python tools/onnx_inference.py export/onnx --dataset ...`: every frame
  captured by a row's time is added, then the row.
- `--events` replays a recorded live session in its real arrival order. Each
  line is `frame <t> <image path>` or `row <t> <roll> <pitch> <yaw> <alt>`.
- `--sync` runs the frontend on the due row instead of in the worker.
- `--realtime` paces the replay by the timestamps.

## Verified

Every tensor sent to and returned by ONNX Runtime was compared with
`tools/onnx_inference.py` (ONNX Runtime 1.30.0, x86-64) over the same input.
They are bit-identical: 0 differing tensors out of 50k–72k per run. Delivery,
emitted and skip flags, the stats and the velocities were identical in every
case, in both sync and async mode.

Models tested:
- RGB planar, frame gap 20, pair stride 10, output on pairs, yaw from `WWMYaw_RAD`;
- gray planar, stride 1, every row emitted (also with `--frontend-batch 2`);
- the flow frontend (non-planar, NLL);
- the RGB model with lens distortion (OpenCV undistortion path; Python cv2 4.12).

Inputs tested:
- the rendered test flight;
- an awkward-input flight: NaN/inf rows, a 0.5 s telemetry hole, a duplicate
  timestamp, time going backwards, altitude below 1 m, a 2 s camera dropout, a
  jittered extra frame, and CSV oddities (BOM, CRLF line endings, quotes,
  blank line, `+`, spaces, `1_5_0.25`);
- three random live-arrival logs: jitter, duplicates, steps back in time, NaN
  rows, a hole, dropped frames, and frames arriving 0–600 ms late;
- a log that forces the async re-run path.

Image preprocessing matched Pillow 12.3 bit for bit in 850 comparisons:
- decode: PNG gray/RGB/RGBA, JPEG 4:2:0/4:2:2/4:4:4/gray;
- resize: up, down, odd sizes, 1x1.

Rerun the parity check on your own export:

    python cpp/tests/compare_with_python.py build export/onnx data_split/test /tmp/parity

An AddressSanitizer + UBSan build ran the awkward-input and stress cases in
sync and async mode with no memory errors and no undefined behaviour.

Real-time replay of the test flight (x86-64, 4 cores, async), measured on the
tiny test model, so expect higher times with your real model:
- the pair result was always ready when due (longest wait 0.001 ms);
- `addTelemetry` mean 0.33 ms, max 0.69 ms;
- in sync mode a due row instead waits for the whole frontend.

## Limits

- **Undistortion**: your calibration has lens distortion, so this path is
  used. It calls OpenCV's `initUndistortRectifyMap` + `remap`, as the Python
  side does. The maps are identical in every OpenCV version, but **`remap`
  rounds differently in OpenCV 5**:
  - C++ OpenCV 4.6 matched Python cv2 4.10 and 4.12 bit for bit, end to end;
  - Python cv2 5.0 differs by up to 2 gray levels per pixel.

  So build the C++ runtime against the same OpenCV major version the model's
  training cache was made with. Check it with
  `python -c "import cv2; print(cv2.__version__)"` in the training
  environment.
- **File decoding** (`loadImageFile`, replay only): OpenCV/libjpeg-turbo, the
  same pixels Pillow decodes for gray/YCbCr JPEG and 8-bit PNG. CMYK JPEGs and
  16-bit images are not supported. Live frames come from the camera as raw
  pixels and are not affected.
- **Replay `image_time_offset_s`**: a per-time offset table is refused; use a
  constant.
