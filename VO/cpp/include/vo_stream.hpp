// vo_stream.hpp - online (streaming) C++ runtime for the exported VO model.
//
// A C++17 port of VO/tools/onnx_inference.py (VOOnnxRuntime), which is itself
// checked against the PyTorch model. It runs the two exported graphs:
//
//   frontend.onnx       once per image PAIR  (two frames + the pair's attitude geometry)
//   temporal_step.onnx  once per telemetry TICK (aiding vector + latest pair + carried state)
//
// with every rule of the Python runtime reproduced exactly: which frames pair,
// when a pair is delivered (deployment_latency_s after its second exposure),
// telemetry-hole and frame-gap rejection, attitude/altitude interpolation at the
// exposures, the aiding vector, the visual age, the held geometric velocity, and
// the image preprocessing (Pillow-identical RGB/L conversion and bilinear
// resize, or OpenCV undistortion when the calibration has lens distortion).
//
// Usage:
//
//   vo::StreamRuntime vo("export/onnx");
//   vo.addFrame(view, capture_time_s);                 // every camera frame, telemetry clock
//   vo::TickOutput out = vo.addTelemetry(t, roll, pitch, yaw, relative_alt);  // every row
//   if (out.emitted) use(out.velocity);                // m/s, body FRD (x fwd, y right, z down)
//
// Angles in radians, from the same columns the model was trained on (see
// settings().attitude_columns). Times in seconds on the TELEMETRY clock: apply
// the camera-to-telemetry offset (settings().image_time_offset_s) before
// addFrame. Frames must be added in capture order. A frame should be added
// before its pair is due (its capture time + deployment latency); a later one
// is still used, but its pair is delivered on the next telemetry row instead
// (exactly as the Python runtime does) and counted in stats().late_pairs.
//
// Threads: addFrame from one thread (the camera) and addTelemetry from another
// (the telemetry loop) is fine; the calls are serialised internally, and a
// frame's preprocessing runs outside the lock so it does not hold up telemetry.
// With Options::async_frontend (the default) a pair's frontend runs in a worker
// thread as soon as its inputs exist, so the telemetry row it is due on
// normally finds the result ready. The numbers are identical either way.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vo {

// A camera frame: 8-bit, rows top to bottom, channels interleaved (HWC).
struct ImageView {
  const std::uint8_t* data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 0;        // 1 (gray), 3 (colour) or 4 (colour + alpha, alpha ignored)
  std::size_t stride = 0;  // bytes per row; 0 means width * channels
  bool bgr = false;        // colour channels in B, G, R(, A) order (OpenCV); default R, G, B(, A)
};

// An owned 8-bit HWC image (1 or 3 channels).
struct ImageU8 {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<std::uint8_t> data;
  ImageView view() const;
};

// What vo_onnx.json says the model was trained with (the parts a caller needs).
struct Settings {
  std::string csv_name = "flight.csv";
  std::string image_folder = "images";
  std::string time_column = "Time";
  double time_scale = 1.0;
  std::vector<std::string> attitude_columns;  // empty: the trainer chose by availability
  std::string altitude_column;                // empty: the trainer chose by availability
  double image_time_offset_s = 0.0;           // camera clock + this = telemetry clock
  bool image_time_offset_is_table = false;    // a per-time offset table (not supported here)
  std::string image_pattern = "*.jpg";
  double image_time_scale = 0.001;            // image file stamp units -> seconds
  bool color = false;
  int image_height = 576;                     // working size the frontend sees
  int image_width = 1024;
  int native_height = 0;                      // camera frames must be this size (0: unchecked)
  int native_width = 0;
  bool undistort = false;                     // OpenCV undistortion from the calibration
  int frame_gap = 1;
  int pair_stride = 1;
  double deployment_latency_s = 0.35;
  bool output_on_pairs = false;
  std::optional<double> max_frame_gap_s;
  int warmup_ticks = 0;
};

struct Options {
  // Run each pair's frontend in a worker thread as soon as its inputs exist
  // (the telemetry row at or after its second exposure), instead of inside
  // the telemetry call on the row it is due. At the due row the inputs are
  // re-derived exactly as the synchronous path does them; should they differ
  // (they cannot with increasing telemetry times), the frontend is re-run.
  bool async_frontend = true;
  // ONNX Runtime intra-op threads per session; 0 keeps the library default.
  int intra_op_threads = 0;
  // Parity testing: append every graph input and output (a hash of the exact
  // float32 bytes, plus the first values) to this text file. Empty: off.
  std::string trace_path;
};

struct TickOutput {
  double time_s = 0.0;
  std::array<float, 3> velocity{};      // this row's model output, m/s, body FRD
  bool emitted = false;                 // velocity is an output (pair delivered, or every
                                        // row for a model without output_on_pairs)
  std::array<float, 3> output{};        // the last emitted velocity, held between pairs
  bool pair_delivered = false;
  std::array<float, 3> log_variance{};  // per-axis log variance (only trained under NLL)
  bool skipped = false;                 // non-finite row: not used, state untouched
};

struct Stats {
  long pairs = 0;            // pairs that reached the frontend
  long delivered = 0;        // ... and passed its reliability gate
  long refused = 0;          // ... and failed it
  long rejected_gap = 0;     // image gap > max_frame_gap_s, or exposure inside a telemetry hole
  long skipped_rows = 0;     // non-finite telemetry rows
  long late_pairs = 0;       // pairs whose second frame was added after they were due
  long async_runs = 0;       // frontend results the worker computed and the due row used
  long async_recomputed = 0; // worker inputs differed at the due row, frontend re-run there
  double max_due_frontend_ms = 0.0;  // longest a telemetry call spent on one pair's frontend
                                     // (waiting for the worker, or running it itself)
};

class StreamRuntime {
 public:
  // onnx_dir: the export folder (frontend.onnx, temporal_step.onnx, vo_onnx.json).
  explicit StreamRuntime(const std::string& onnx_dir, const Options& options = Options());
  ~StreamRuntime();
  StreamRuntime(const StreamRuntime&) = delete;
  StreamRuntime& operator=(const StreamRuntime&) = delete;

  // Cold start: zero state, nothing held, no frames, no telemetry.
  void reset();
  // One camera frame. Throws std::invalid_argument (and adds nothing) for a
  // frame that is not the calibration's native size or a non-finite time.
  void addFrame(const ImageView& image, double capture_time_s);
  // The same for a frame that is costly to produce (decoded from a file, say):
  // `load` is called only if a pair will use the frame - with a pair stride
  // most frames only advance the frame count. Its errors propagate, and the
  // frame is then not added.
  void addFrame(double capture_time_s, const std::function<ImageU8()>& load);
  // One telemetry row; returns this row's velocity and whether it is an output.
  TickOutput addTelemetry(double time_s, double roll, double pitch, double yaw,
                          double relative_altitude_m);

  Stats stats() const;
  const Settings& settings() const;
  // A frame exactly as the frontend receives it: float32 (C, H, W) in [0, 1]
  // at the working size (conversion, then undistortion or resize, then / 255).
  std::vector<float> preprocess(const ImageView& image) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Pillow's Image.convert("RGB" if color else "L") on a frame.
ImageU8 convertMode(const ImageView& image, bool color);
// Pillow's Image.resize((width, height), Image.BILINEAR), bit for bit.
ImageU8 resizeBilinearPIL(const ImageU8& image, int width, int height);
// Decode an image file (OpenCV) and convert it as Pillow's open().convert()
// does. 8-bit gray, colour or colour + alpha files (JPEG, PNG, ...).
ImageU8 loadImageFile(const std::string& path, bool color);

}  // namespace vo
