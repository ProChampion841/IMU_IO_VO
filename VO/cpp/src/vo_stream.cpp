// The streaming runtime: VOOnnxRuntime (tools/onnx_inference.py) in C++.
//
// Each piece names the Python it ports. The arithmetic that builds the graph
// inputs lives in vo_math.hpp and follows the Python operation for operation;
// the rules (pairing, delivery, rejection, holding) follow it line for line.
#include "vo_stream.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/3d.hpp>
#else
#include <opencv2/calib3d.hpp>
#endif

#include "vo_json.hpp"
#include "vo_math.hpp"

namespace vo {

namespace {

using math::Mat3;
using math::Quat;
using math::Vec3;

// ---- reading vo_onnx.json the way Python's `x or default` does --------------

bool truthy(const json::Value* value) { return value != nullptr && value->truthy(); }

std::string stringOr(const json::Value& object, const char* key, const char* fallback) {
  const json::Value* value = object.find(key);
  return truthy(value) ? value->asString() : std::string(fallback);
}

double numberOr(const json::Value& object, const char* key, double fallback) {
  const json::Value* value = object.find(key);
  return truthy(value) ? value->asNumber() : fallback;
}

// Python int(): truncation toward zero.
long pyInt(const json::Value& value) {
  const double number = value.asNumber();
  if (!std::isfinite(number)) throw std::runtime_error("vo_onnx.json: expected an integer");
  return static_cast<long>(std::trunc(number));
}

void flattenNumbers(const json::Value& value, std::vector<double>& out) {
  if (value.isArray()) {
    for (const auto& item : value.array) flattenNumbers(item, out);
  } else {
    out.push_back(value.asNumber());
  }
}

cv::Mat matrix3x3(const json::Value& value, const char* what) {
  std::vector<double> numbers;
  flattenNumbers(value, numbers);
  if (numbers.size() != 9) throw std::runtime_error(std::string("vo_onnx.json: ") + what + " is not 3x3");
  cv::Mat matrix(3, 3, CV_64F);
  for (int i = 0; i < 9; ++i) matrix.at<double>(i / 3, i % 3) = numbers[static_cast<std::size_t>(i)];
  return matrix;
}

// ---- tracing (parity tests) ---------------------------------------------------

// CRC-32 (zlib's), of the exact float32 bytes - Python's zlib.crc32 gives the same.
std::uint32_t crc32(const void* data, std::size_t bytes) {
  static const std::array<std::uint32_t, 256> table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      t[i] = c;
    }
    return t;
  }();
  const auto* p = static_cast<const unsigned char*>(data);
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < bytes; ++i) crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}

// "<tag> <name> <count> <crc32>" and, for small tensors, every value.
void traceTensor(std::string& out, const char* tag, const std::string& name, const float* data,
                 std::size_t count) {
  char buffer[64];
  std::snprintf(buffer, sizeof buffer, "%08x", static_cast<unsigned>(crc32(data, count * sizeof(float))));
  out += tag;
  out += ' ';
  out += name;
  out += ' ';
  out += std::to_string(count);
  out += ' ';
  out += buffer;
  if (count <= 64) {
    for (std::size_t i = 0; i < count; ++i) {
      std::snprintf(buffer, sizeof buffer, " %.9g", static_cast<double>(data[i]));
      out += buffer;
    }
  }
  out += '\n';
}

std::size_t elementCount(const std::vector<std::int64_t>& shape) {
  std::size_t count = 1;
  for (const std::int64_t dim : shape) count *= static_cast<std::size_t>(dim);
  return count;
}

std::string shapeText(const std::vector<std::int64_t>& shape) {
  std::string text = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) text += ", ";
    text += std::to_string(shape[i]);
  }
  return text + "]";
}

// ---- the stream's pieces --------------------------------------------------------

struct Frame {
  double time_s = 0.0;
  // float32 (C, H, W) as the frontend receives it; null for a frame no pair uses.
  std::shared_ptr<const std::vector<float>> pixels;
};

// The frontend's non-image inputs, as the float32 values handed to ONNX Runtime.
struct PairInputs {
  float pair_dt_s = 0.0f;
  std::array<float, 9> relative_rotation{};
  std::array<float, 3> down_body{};
  std::array<float, 2> altitude_m{};
  std::array<float, 3> body_rate_rad_s{};

  bool sameBits(const PairInputs& other) const {
    return std::memcmp(&pair_dt_s, &other.pair_dt_s, sizeof pair_dt_s) == 0 &&
           std::memcmp(relative_rotation.data(), other.relative_rotation.data(), sizeof relative_rotation) == 0 &&
           std::memcmp(down_body.data(), other.down_body.data(), sizeof down_body) == 0 &&
           std::memcmp(altitude_m.data(), other.altitude_m.data(), sizeof altitude_m) == 0 &&
           std::memcmp(body_rate_rad_s.data(), other.body_rate_rad_s.data(), sizeof body_rate_rad_s) == 0;
  }
};

// The frontend outputs, first batch row of each, in vo_onnx.json's order.
struct FrontendResult {
  std::vector<std::vector<float>> outputs;
  std::string trace;  // the run's inputs and outputs, when tracing
};

// One pair's frontend run in the worker thread.
struct Job {
  enum class State { Queued, Running, Done, Failed, Cancelled };
  std::shared_ptr<const std::vector<float>> image0;
  std::shared_ptr<const std::vector<float>> image1;
  PairInputs inputs;
  std::mutex mutex;
  std::condition_variable done;
  State state = State::Queued;
  FrontendResult result;
};

// A pair waiting for its due row: (ready time, (t0, frame0), (t1, frame1)).
struct PendingPair {
  double ready = 0.0;
  Frame frame0;
  Frame frame1;
  bool launch_considered = false;
  std::shared_ptr<Job> job;
};

void cancelIfQueued(const std::shared_ptr<Job>& job) {
  if (!job) return;
  std::lock_guard<std::mutex> lock(job->mutex);
  if (job->state == Job::State::Queued) job->state = Job::State::Cancelled;
}

}  // namespace

// =============================================================================

struct StreamRuntime::Impl {
  Options options;
  Settings settings;

  // ---- the model (fixed after construction) ----
  bool geometric = false;  // temporal step takes the held geometric velocity
  bool planar = false;     // frontend takes the pair's rotation / down / altitude
  long front_batch = 1;
  std::size_t visual_dim = 0;
  double log_altitude_mean = 0.0;
  double delta_time_scale = 1.0;
  double latency = 0.35;
  double max_telemetry_gap_s = 8.0;
  double history_s = 0.0;
  int channels = 1;

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "vo_stream"};
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> frontend;
  std::unique_ptr<Ort::Session> step;
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<std::string> front_input_names;               // session order
  std::vector<std::vector<std::int64_t>> front_input_shapes;
  std::vector<std::string> front_output_names;              // vo_onnx.json order (as Python)
  std::size_t out_token = 0, out_quality = 0, out_reliable = 0, out_geometric = 0;
  std::vector<std::string> step_input_names;                // session order
  std::vector<std::vector<std::int64_t>> step_input_shapes; // batch 1
  std::vector<std::string> step_output_names;               // vo_onnx.json order
  std::size_t out_velocity = 0, out_log_variance = 0;
  std::vector<std::string> state_names;                     // vo_onnx.json "state" order
  std::vector<std::vector<std::int64_t>> state_shapes;      // [1, ...]
  std::vector<std::size_t> state_output_index;              // next_<state> in step outputs

  cv::Mat map_x;  // undistortion maps, when the calibration has distortion
  cv::Mat map_y;

  // ---- the stream (guarded by `mutex`; frame_count by `frame_mutex`) ----
  mutable std::mutex mutex;
  std::mutex frame_mutex;
  std::vector<Ort::Value> state;
  std::array<float, 3> held{};
  float held_valid = 0.0f;
  bool have_delivery = false;
  double last_delivery_s = 0.0;
  std::array<float, 3> emitted{};
  std::deque<std::pair<long, Frame>> frames;
  long frame_count = 0;
  std::vector<PendingPair> pending;
  std::deque<double> times;
  std::deque<Quat> quats;
  std::deque<double> altitudes;
  bool have_first_telemetry = false;
  double first_telemetry_s = 0.0;
  Stats stats;
  bool tracing = false;  // fixed after construction (the worker reads it)
  std::ofstream trace;   // written only under `mutex`

  // ---- the frontend worker ----
  std::thread worker;
  std::mutex queue_mutex;
  std::condition_variable queue_ready;
  std::deque<std::shared_ptr<Job>> queue;
  bool stopping = false;

  Impl(const std::string& onnx_dir, const Options& opts);
  ~Impl();

  void loadMetadata(const std::filesystem::path& folder, const json::Value& meta);
  void openSessions(const std::filesystem::path& folder, const json::Value& meta);
  void resetStream();

  void checkFrame(const ImageView& image) const;
  std::vector<float> preprocessFrame(const ImageView& image) const;
  bool frameUsed(long index) const;
  void addFrame(double time_s, const std::function<std::vector<float>()>& pixels_of);
  void insertFrame(long index, double time_s, std::shared_ptr<const std::vector<float>> pixels);

  std::pair<Quat, double> attitudeAt(double query) const;
  bool straddlesHole(double query) const;
  PairInputs pairInputs(double t0, double t1) const;
  FrontendResult runFrontend(const std::vector<float>& image0, const std::vector<float>& image1,
                             const PairInputs& inputs) const;
  bool runPair(PendingPair& pair, FrontendResult& result);
  FrontendResult obtainResult(PendingPair& pair, const PairInputs& inputs);
  void considerLaunch(PendingPair& pair);
  void workerLoop();
};

StreamRuntime::Impl::Impl(const std::string& onnx_dir, const Options& opts) : options(opts) {
  env.DisableTelemetryEvents();  // a flight computer has no business phoning home
  const std::filesystem::path folder(onnx_dir);
  const json::Value meta = json::parseFile((folder / "vo_onnx.json").string());
  loadMetadata(folder, meta);
  openSessions(folder, meta);
  if (!options.trace_path.empty()) {
    trace.open(options.trace_path, std::ios::out | std::ios::trunc);
    if (!trace) throw std::runtime_error("cannot write trace " + options.trace_path);
    tracing = true;
  }
  resetStream();
  if (options.async_frontend) worker = std::thread([this] { workerLoop(); });
}

StreamRuntime::Impl::~Impl() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    stopping = true;
    for (const auto& job : queue) cancelIfQueued(job);
    queue.clear();
  }
  queue_ready.notify_all();
  if (worker.joinable()) worker.join();
}

// ---- construction ----------------------------------------------------------------

void StreamRuntime::Impl::loadMetadata(const std::filesystem::path& folder, const json::Value& meta) {
  (void)folder;
  const json::Value& ds = meta.at("dataset_settings");
  const json::Value& timing = meta.at("timing");
  Settings& s = settings;

  s.csv_name = stringOr(ds, "csv_name", "flight.csv");
  s.image_folder = stringOr(ds, "image_folder", "images");
  s.time_column = stringOr(ds, "time_column", "Time");
  s.time_scale = numberOr(ds, "time_scale", 1.0);
  if (const json::Value* columns = ds.find("attitude_columns"); truthy(columns)) {
    for (const auto& name : columns->array) s.attitude_columns.push_back(name.asString());
    if (s.attitude_columns.size() != 3) {
      throw std::runtime_error("vo_onnx.json: attitude_columns must name roll, pitch and yaw");
    }
  }
  if (const json::Value* column = ds.find("altitude_column"); truthy(column)) {
    s.altitude_column = column->asString();
  }
  if (const json::Value* offset = ds.find("image_time_offset_s"); truthy(offset)) {
    if (offset->isObject()) {
      s.image_time_offset_is_table = true;
    } else {
      s.image_time_offset_s = offset->asNumber();
    }
  }
  s.image_pattern = stringOr(meta, "image_pattern", "*.jpg");
  s.image_time_scale = numberOr(meta, "image_time_scale", 0.001);
  s.color = truthy(ds.find("color"));
  if (const json::Value* size = ds.find("image_size"); truthy(size)) {
    s.image_height = static_cast<int>(pyInt(size->at(0)));
    s.image_width = static_cast<int>(pyInt(size->at(1)));
  }
  if (s.image_height <= 0 || s.image_width <= 0) throw std::runtime_error("vo_onnx.json: bad image_size");
  channels = s.color ? 3 : 1;

  s.frame_gap = static_cast<int>(pyInt(timing.at("frame_gap")));
  s.pair_stride = static_cast<int>(pyInt(timing.at("pair_stride")));
  if (s.frame_gap < 1 || s.pair_stride < 1) {
    throw std::runtime_error("vo_onnx.json: frame_gap and pair_stride must be at least 1");
  }
  s.deployment_latency_s = timing.at("deployment_latency_s").asNumber();
  latency = s.deployment_latency_s;
  s.output_on_pairs = timing.at("output_on_pairs").truthy();
  if (const json::Value* gap = ds.find("max_frame_gap_s"); gap != nullptr && !gap->isNull()) {
    s.max_frame_gap_s = gap->asNumber();
  }
  if (const json::Value* warmup = ds.find("warmup"); truthy(warmup)) s.warmup_ticks = static_cast<int>(pyInt(*warmup));

  const json::Value& normalizer = meta.at("temporal_step").at("normalizer");
  log_altitude_mean = normalizer.at("log_altitude_mean").asNumber();
  delta_time_scale = normalizer.at("delta_time_scale").asNumber();
  if (delta_time_scale == 0.0) delta_time_scale = 1.0;  // `float(...) or 1.0`
  // Same rule as training: an exposure inside a telemetry hole wider than
  // 8 telemetry steps is not a measurement.
  max_telemetry_gap_s = 8.0 * delta_time_scale;
  // `float(self.max_frame_gap_s or 2.0)`: a gap of 0 also falls back to 2.
  const double gap_for_history =
      (s.max_frame_gap_s.has_value() && *s.max_frame_gap_s != 0.0) ? *s.max_frame_gap_s : 2.0;
  history_s = 10.0 + latency + 2.0 * gap_for_history;

  // The camera: training refuses a frame that is not the calibration's native
  // size (the intrinsics, and undistortion maps from them, hold only there).
  const json::Value* calibration = meta.find("calibration");
  if (truthy(calibration)) {
    if (const json::Value* native = calibration->find("native_size"); truthy(native)) {
      s.native_height = static_cast<int>(pyInt(native->at(0)));
      s.native_width = static_cast<int>(pyInt(native->at(1)));
    }
    std::vector<double> distortion;
    if (const json::Value* d = calibration->find("distortion"); truthy(d)) flattenNumbers(*d, distortion);
    bool any_distortion = false;
    for (const double k : distortion) any_distortion = any_distortion || std::fabs(k) > 0;
    if (any_distortion && !truthy(calibration->find("images_rectified"))) {
      const json::Value* working = meta.at("frontend").find("camera_matrix_working");
      if (!truthy(working)) throw std::runtime_error("vo_onnx.json: distortion without camera_matrix_working");
      const cv::Mat native_k = matrix3x3(calibration->at("native_camera_matrix"), "native_camera_matrix");
      const cv::Mat working_k = matrix3x3(*working, "camera_matrix_working");
      cv::Mat dist(1, static_cast<int>(distortion.size()), CV_64F);
      for (std::size_t i = 0; i < distortion.size(); ++i) dist.at<double>(0, static_cast<int>(i)) = distortion[i];
      cv::initUndistortRectifyMap(native_k, dist, cv::noArray(), working_k,
                                  cv::Size(s.image_width, s.image_height), CV_32FC1, map_x, map_y);
      s.undistort = true;
    }
  }

  const json::Value& front = meta.at("frontend");
  front_batch = pyInt(front.at("inputs").at("image0").at(0));
  if (front_batch < 1) throw std::runtime_error("vo_onnx.json: bad frontend batch");
  planar = front.at("inputs").find("relative_rotation") != nullptr;
  for (const auto& name : front.at("outputs").array) front_output_names.push_back(name.asString());
  const json::Value& step_meta = meta.at("temporal_step");
  for (const auto& name : step_meta.at("inputs").array) {
    if (name.asString() == "visual_velocity") geometric = true;
  }
  for (const auto& name : step_meta.at("outputs").array) step_output_names.push_back(name.asString());
  for (const auto& entry : step_meta.at("state").array) {
    state_names.push_back(entry.at("name").asString());
    std::vector<std::int64_t> shape{1};
    const json::Value& dims = entry.at("shape");
    for (std::size_t i = 1; i < dims.array.size(); ++i) shape.push_back(pyInt(dims.array[i]));
    state_shapes.push_back(shape);
  }
}

void StreamRuntime::Impl::openSessions(const std::filesystem::path& folder, const json::Value& meta) {
  if (options.intra_op_threads > 0) session_options.SetIntraOpNumThreads(options.intra_op_threads);
  const json::Value& files = meta.at("files");
  const std::filesystem::path front_path = folder / files.at("frontend").asString();
  const std::filesystem::path step_path = folder / files.at("temporal_step").asString();
  frontend = std::make_unique<Ort::Session>(env, front_path.c_str(), session_options);
  step = std::make_unique<Ort::Session>(env, step_path.c_str(), session_options);

  Ort::AllocatorWithDefaultOptions allocator;
  auto inputsOf = [&](Ort::Session& session, std::vector<std::string>& names,
                      std::vector<std::vector<std::int64_t>>& shapes, const char* graph) {
    for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
      names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
      const Ort::TypeInfo type_info = session.GetInputTypeInfo(i);  // owns what `info` views
      const auto info = type_info.GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error(std::string(graph) + " input " + names.back() + " is not float32");
      }
      shapes.push_back(info.GetShape());
    }
  };
  auto outputIndex = [&](Ort::Session& session, const std::vector<std::string>& wanted, const std::string& name,
                         const char* graph) {
    bool in_session = false;
    for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
      in_session = in_session || name == session.GetOutputNameAllocated(i, allocator).get();
    }
    const auto it = std::find(wanted.begin(), wanted.end(), name);
    if (!in_session || it == wanted.end()) throw std::runtime_error(std::string(graph) + " has no output " + name);
    return static_cast<std::size_t>(it - wanted.begin());
  };

  // frontend: every input must be one we produce, at the size we produce it.
  inputsOf(*frontend, front_input_names, front_input_shapes, "frontend.onnx");
  for (std::size_t i = 0; i < front_input_names.size(); ++i) {
    const std::string& name = front_input_names[i];
    std::vector<std::int64_t> expected;
    if (name == "image0" || name == "image1") {
      expected = {front_batch, channels, settings.image_height, settings.image_width};
    } else if (name == "pair_dt_s") {
      expected = {front_batch};
    } else if (name == "relative_rotation" && planar) {
      expected = {front_batch, 3, 3};
    } else if (name == "down_body" && planar) {
      expected = {front_batch, 3};
    } else if (name == "altitude_m" && planar) {
      expected = {front_batch, 2};
    } else if (name == "body_rate_rad_s" && !planar) {
      expected = {front_batch, 3};
    } else {
      throw std::runtime_error("frontend.onnx: unexpected input " + name);
    }
    std::vector<std::int64_t>& shape = front_input_shapes[i];
    if (!shape.empty() && shape[0] < 0) shape[0] = front_batch;
    if (shape != expected) {
      throw std::runtime_error("frontend.onnx: input " + name + " is " + shapeText(shape) + ", expected " +
                               shapeText(expected) + " (vo_onnx.json image_size / color / batch)");
    }
  }
  out_token = outputIndex(*frontend, front_output_names, "visual_token", "frontend.onnx");
  out_quality = outputIndex(*frontend, front_output_names, "visual_quality", "frontend.onnx");
  out_reliable = outputIndex(*frontend, front_output_names, "pair_reliable", "frontend.onnx");
  if (geometric) out_geometric = outputIndex(*frontend, front_output_names, "geometric_velocity", "frontend.onnx");

  // temporal step: batch 1.
  inputsOf(*step, step_input_names, step_input_shapes, "temporal_step.onnx");
  for (std::size_t i = 0; i < step_input_names.size(); ++i) {
    std::vector<std::int64_t>& shape = step_input_shapes[i];
    if (!shape.empty() && shape[0] < 0) shape[0] = 1;
    for (const std::int64_t dim : shape) {
      if (dim < 0) throw std::runtime_error("temporal_step.onnx: input " + step_input_names[i] + " has a free dimension");
    }
    if (step_input_names[i] == "visual_token") visual_dim = elementCount(shape);
  }
  if (visual_dim == 0) throw std::runtime_error("temporal_step.onnx: no visual_token input");
  auto stepInput = [&](const std::string& name, std::size_t count) {
    const auto it = std::find(step_input_names.begin(), step_input_names.end(), name);
    if (it == step_input_names.end()) throw std::runtime_error("temporal_step.onnx: no input " + name);
    const std::size_t have = elementCount(step_input_shapes[static_cast<std::size_t>(it - step_input_names.begin())]);
    if (have != count) {
      throw std::runtime_error("temporal_step.onnx: input " + name + " has " + std::to_string(have) +
                               " values, expected " + std::to_string(count));
    }
  };
  stepInput("aiding", 9);
  stepInput("visual_present", 1);
  stepInput("visual_age", 1);
  stepInput("visual_quality", 1);
  stepInput("log_altitude", 1);
  if (geometric) {
    stepInput("visual_velocity", 3);
    stepInput("visual_velocity_valid", 1);
  }
  for (std::size_t k = 0; k < state_names.size(); ++k) {
    stepInput(state_names[k], elementCount(state_shapes[k]));
    state_output_index.push_back(outputIndex(*step, step_output_names, "next_" + state_names[k], "temporal_step.onnx"));
  }
  std::size_t fed = 6 + (geometric ? 2 : 0) + state_names.size();
  if (fed != step_input_names.size()) throw std::runtime_error("temporal_step.onnx: unexpected inputs");
  out_velocity = outputIndex(*step, step_output_names, "predicted_velocity", "temporal_step.onnx");
  out_log_variance = outputIndex(*step, step_output_names, "velocity_log_variance", "temporal_step.onnx");
}

void StreamRuntime::Impl::resetStream() {
  Ort::AllocatorWithDefaultOptions allocator;
  state.clear();
  for (const auto& shape : state_shapes) {
    Ort::Value value = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    const std::size_t count = elementCount(shape);
    if (count) std::fill_n(value.GetTensorMutableData<float>(), count, 0.0f);
    state.push_back(std::move(value));
  }
  held = {0.0f, 0.0f, 0.0f};
  held_valid = 0.0f;
  have_delivery = false;
  last_delivery_s = 0.0;
  emitted = {0.0f, 0.0f, 0.0f};
  frames.clear();
  frame_count = 0;
  for (const auto& pair : pending) cancelIfQueued(pair.job);
  pending.clear();
  times.clear();
  quats.clear();
  altitudes.clear();
  have_first_telemetry = false;
  first_telemetry_s = 0.0;
  stats = Stats();
}

// ---- frames ------------------------------------------------------------------------

void StreamRuntime::Impl::checkFrame(const ImageView& image) const {
  if (image.data == nullptr || image.width <= 0 || image.height <= 0) {
    throw std::invalid_argument("addFrame: empty frame");
  }
  if (settings.native_width > 0 &&
      (image.width != settings.native_width || image.height != settings.native_height)) {
    throw std::invalid_argument("image is " + std::to_string(image.width) + "x" + std::to_string(image.height) +
                                ", but the calibration the model was trained with is " +
                                std::to_string(settings.native_width) + "x" +
                                std::to_string(settings.native_height));
  }
}

// ImagePreprocessor.__call__.
std::vector<float> StreamRuntime::Impl::preprocessFrame(const ImageView& image) const {
  checkFrame(image);
  ImageU8 picture = convertMode(image, settings.color);
  const int height = settings.image_height;
  const int width = settings.image_width;
  ImageU8 working;
  if (settings.undistort) {
    const cv::Mat source(picture.height, picture.width, CV_8UC(picture.channels), picture.data.data());
    cv::Mat remapped;
    cv::remap(source, remapped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    working.width = remapped.cols;
    working.height = remapped.rows;
    working.channels = remapped.channels();
    working.data.resize(static_cast<std::size_t>(working.width) * working.height * working.channels);
    const std::size_t row = static_cast<std::size_t>(working.width) * working.channels;
    for (int y = 0; y < working.height; ++y) {
      std::memcpy(working.data.data() + static_cast<std::size_t>(y) * row, remapped.ptr<std::uint8_t>(y), row);
    }
  } else if (picture.width != width || picture.height != height) {
    working = resizeBilinearPIL(picture, width, height);
  } else {
    working = std::move(picture);
  }
  // HWC uint8 -> CHW float32 / 255 (float32 division, as numpy does it).
  const int c = working.channels;
  const std::size_t plane = static_cast<std::size_t>(working.width) * working.height;
  std::vector<float> out(plane * static_cast<std::size_t>(c));
  for (std::size_t i = 0; i < plane; ++i) {
    for (int k = 0; k < c; ++k) {
      out[static_cast<std::size_t>(k) * plane + i] =
          static_cast<float>(working.data[i * static_cast<std::size_t>(c) + static_cast<std::size_t>(k)]) / 255.0f;
    }
  }
  return out;
}

// Whether any pair will use frame `index`: as the first frame of
// (index, index + gap), or as the second frame of (index - gap, index).
bool StreamRuntime::Impl::frameUsed(long index) const {
  const long first = index - settings.frame_gap;
  return index % settings.pair_stride == 0 || (first >= 0 && first % settings.pair_stride == 0);
}

// add_frame.
void StreamRuntime::Impl::insertFrame(long index, double time_s,
                                      std::shared_ptr<const std::vector<float>> pixels) {
  frames.emplace_back(index, Frame{time_s, std::move(pixels)});
  while (!frames.empty() && frames.front().first < index - settings.frame_gap) frames.pop_front();
  frame_count = index + 1;
  const long first = index - settings.frame_gap;
  if (first < 0 || first % settings.pair_stride != 0 || frames.front().first > first) return;
  const Frame& frame0 = frames[static_cast<std::size_t>(first - frames.front().first)].second;
  const Frame& frame1 = frames.back().second;
  const double t0 = frame0.time_s;
  const double t1 = time_s;
  if (settings.max_frame_gap_s.has_value() && t1 - t0 > *settings.max_frame_gap_s) {
    ++stats.rejected_gap;
    return;
  }
  PendingPair pair;
  pair.ready = t1 + latency;
  pair.frame0 = frame0;
  pair.frame1 = frame1;
  if (!times.empty() && pair.ready <= times.back()) ++stats.late_pairs;
  pending.push_back(std::move(pair));
  considerLaunch(pending.back());
}

// ---- telemetry geometry --------------------------------------------------------------

// _attitude_at: nlerp attitude and linear altitude at `query`, held at the ends.
std::pair<Quat, double> StreamRuntime::Impl::attitudeAt(double query) const {
  double clamped = (times.front() > query) ? times.front() : query;  // max(query, times[0])
  clamped = (times.back() < clamped) ? times.back() : clamped;       // min(..., times[-1])
  std::size_t upper = 0;
  if (times.size() > 1) {
    upper = math::searchSorted(times, clamped, true);
    upper = std::min(std::max<std::size_t>(upper, 1), times.size() - 1);  // np.clip(., 1, n - 1)
  }
  const std::size_t lower = upper > 0 ? upper - 1 : 0;
  const double span = times[upper] - times[lower];
  const double fraction = span > 0 ? (clamped - times[lower]) / span : 0.0;
  Quat q;
  for (std::size_t i = 0; i < 4; ++i) q[i] = (1.0 - fraction) * quats[lower][i] + fraction * quats[upper][i];
  const double altitude = math::interp(clamped, times, altitudes);
  return {math::divided(q, math::norm(q)), altitude};
}

// _straddles_hole.
bool StreamRuntime::Impl::straddlesHole(double query) const {
  if (times.size() < 2) return false;
  std::size_t upper = math::searchSorted(times, query, false);
  upper = std::min(std::max<std::size_t>(upper, 1), times.size() - 1);
  return (times[upper] - times[upper - 1]) > max_telemetry_gap_s;
}

// The non-image feeds of _run_pair.
PairInputs StreamRuntime::Impl::pairInputs(double t0, double t1) const {
  PairInputs in;
  in.pair_dt_s = static_cast<float>(t1 - t0);
  const auto [q0, alt0] = attitudeAt(t0);
  const auto [q1, alt1] = attitudeAt(t1);
  const Mat3 r0 = math::quaternionToMatrix(q0);
  const Mat3 r1 = math::quaternionToMatrix(q1);
  if (planar) {
    const Mat3 relative = math::transposeTimes(r0, r1);
    for (std::size_t i = 0; i < 3; ++i) {
      for (std::size_t j = 0; j < 3; ++j) in.relative_rotation[i * 3 + j] = static_cast<float>(relative[i][j]);
      in.down_body[i] = static_cast<float>(r0[2][i]);
    }
    in.altitude_m = {static_cast<float>(alt0), static_cast<float>(alt1)};
  } else {
    const Vec3 rotvec = math::quaternionToRotvec(math::multiply(math::conjugate(q0), q1));
    const double span = (1e-6 > t1 - t0) ? 1e-6 : t1 - t0;  // max(t1 - t0, 1e-6)
    for (std::size_t i = 0; i < 3; ++i) in.body_rate_rad_s[i] = static_cast<float>(rotvec[i] / span);
  }
  return in;
}

// ---- the frontend ------------------------------------------------------------------

FrontendResult StreamRuntime::Impl::runFrontend(const std::vector<float>& image0, const std::vector<float>& image1,
                                                const PairInputs& in) const {
  const std::size_t batch = static_cast<std::size_t>(front_batch);
  std::vector<std::vector<float>> repeated;  // `np.repeat(v, front_batch, axis=0)`
  repeated.reserve(front_input_names.size());
  std::vector<Ort::Value> values;
  std::vector<const char*> names;
  FrontendResult result;
  for (std::size_t i = 0; i < front_input_names.size(); ++i) {
    const std::string& name = front_input_names[i];
    const float* source = nullptr;
    std::size_t count = 0;
    if (name == "image0") {
      source = image0.data();
      count = image0.size();
    } else if (name == "image1") {
      source = image1.data();
      count = image1.size();
    } else if (name == "pair_dt_s") {
      source = &in.pair_dt_s;
      count = 1;
    } else if (name == "relative_rotation") {
      source = in.relative_rotation.data();
      count = in.relative_rotation.size();
    } else if (name == "down_body") {
      source = in.down_body.data();
      count = in.down_body.size();
    } else if (name == "altitude_m") {
      source = in.altitude_m.data();
      count = in.altitude_m.size();
    } else {  // body_rate_rad_s (checked at construction)
      source = in.body_rate_rad_s.data();
      count = in.body_rate_rad_s.size();
    }
    const std::vector<std::int64_t>& shape = front_input_shapes[i];
    if (count * batch != elementCount(shape)) throw std::logic_error("frontend input size: " + name);
    float* data = const_cast<float*>(source);  // ONNX Runtime does not write inputs
    if (batch > 1) {
      repeated.emplace_back(count * batch);
      for (std::size_t b = 0; b < batch; ++b) std::copy_n(source, count, repeated.back().data() + b * count);
      data = repeated.back().data();
    }
    values.push_back(Ort::Value::CreateTensor<float>(memory_info, data, count * batch, shape.data(), shape.size()));
    names.push_back(name.c_str());
    if (tracing) traceTensor(result.trace, "front.in", name, data, count * batch);
  }
  std::vector<const char*> output_names;
  for (const auto& name : front_output_names) output_names.push_back(name.c_str());
  std::vector<Ort::Value> outputs = frontend->Run(Ort::RunOptions{nullptr}, names.data(), values.data(), values.size(),
                                                  output_names.data(), output_names.size());
  for (std::size_t k = 0; k < outputs.size(); ++k) {
    const auto info = outputs[k].GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("frontend output " + front_output_names[k] + " is not float32");
    }
    const std::size_t count = info.GetElementCount();
    const float* data = outputs[k].GetTensorData<float>();
    if (tracing) traceTensor(result.trace, "front.out", front_output_names[k], data, count);
    const std::size_t row = count / batch;  // `v[:1]`
    result.outputs.emplace_back(data, data + row);
  }
  if (result.outputs[out_token].size() != visual_dim) {
    throw std::runtime_error("frontend visual_token size does not match the temporal step's");
  }
  if (result.outputs[out_quality].empty() || result.outputs[out_reliable].empty() ||
      (geometric && result.outputs[out_geometric].size() != 3)) {
    throw std::runtime_error("frontend outputs have unexpected sizes");
  }
  return result;
}

// The frontend result for a due pair: the worker's, if it ran with exactly
// these inputs, else computed here.
FrontendResult StreamRuntime::Impl::obtainResult(PendingPair& pair, const PairInputs& inputs) {
  const auto start = std::chrono::steady_clock::now();
  FrontendResult result;
  bool have = false;
  if (pair.job) {
    Job& job = *pair.job;
    std::unique_lock<std::mutex> lock(job.mutex);
    if (job.state == Job::State::Queued) {
      job.state = Job::State::Cancelled;  // not started yet: run it right here instead
    } else if (!job.inputs.sameBits(inputs)) {
      ++stats.async_recomputed;
    } else if (job.state != Job::State::Cancelled) {
      job.done.wait(lock, [&] { return job.state == Job::State::Done || job.state == Job::State::Failed; });
      if (job.state == Job::State::Done) {
        result = std::move(job.result);
        have = true;
        ++stats.async_runs;
      }
    }
  }
  if (!have) result = runFrontend(*pair.frame0.pixels, *pair.frame1.pixels, inputs);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  stats.max_due_frontend_ms = std::max(stats.max_due_frontend_ms, ms);
  return result;
}

// _run_pair: false when the pair is rejected (telemetry hole) or refused.
bool StreamRuntime::Impl::runPair(PendingPair& pair, FrontendResult& result) {
  const double t0 = pair.frame0.time_s;
  const double t1 = pair.frame1.time_s;
  if (straddlesHole(t0) || straddlesHole(t1)) {
    ++stats.rejected_gap;
    cancelIfQueued(pair.job);
    return false;
  }
  ++stats.pairs;
  const PairInputs inputs = pairInputs(t0, t1);
  result = obtainResult(pair, inputs);
  if (tracing) trace << result.trace;
  if (result.outputs[out_reliable][0] <= 0) {  // NaN is not refused, as in Python
    ++stats.refused;
    return false;
  }
  ++stats.delivered;
  return true;
}

// Start a pair's frontend in the worker once its inputs exist: a telemetry
// row at or after its second exposure. Decided once per pair; whatever is
// decided here, the due row re-derives the inputs and checks.
void StreamRuntime::Impl::considerLaunch(PendingPair& pair) {
  if (!options.async_frontend || pair.launch_considered || times.empty()) return;
  if (!(times.back() >= pair.frame1.time_s)) return;
  pair.launch_considered = true;
  if (pair.ready < first_telemetry_s) return;  // never delivered
  if (straddlesHole(pair.frame0.time_s) || straddlesHole(pair.frame1.time_s)) return;
  auto job = std::make_shared<Job>();
  job->image0 = pair.frame0.pixels;
  job->image1 = pair.frame1.pixels;
  job->inputs = pairInputs(pair.frame0.time_s, pair.frame1.time_s);
  pair.job = job;
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    queue.push_back(std::move(job));
  }
  queue_ready.notify_one();
}

void StreamRuntime::Impl::workerLoop() {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      queue_ready.wait(lock, [&] { return stopping || !queue.empty(); });
      if (stopping) return;
      job = std::move(queue.front());
      queue.pop_front();
    }
    {
      std::lock_guard<std::mutex> lock(job->mutex);
      if (job->state != Job::State::Queued) continue;
      job->state = Job::State::Running;
    }
    FrontendResult result;
    bool ok = true;
    try {
      result = runFrontend(*job->image0, *job->image1, job->inputs);
    } catch (...) {
      ok = false;  // the due row runs it again itself and reports the error
    }
    {
      std::lock_guard<std::mutex> lock(job->mutex);
      if (ok) job->result = std::move(result);
      job->state = ok ? Job::State::Done : Job::State::Failed;
    }
    job->done.notify_all();
  }
}

// =============================================================================

StreamRuntime::StreamRuntime(const std::string& onnx_dir, const Options& options)
    : impl_(std::make_unique<Impl>(onnx_dir, options)) {}

StreamRuntime::~StreamRuntime() = default;

void StreamRuntime::reset() {
  std::lock_guard<std::mutex> order(impl_->frame_mutex);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->resetStream();
}

const Settings& StreamRuntime::settings() const { return impl_->settings; }

Stats StreamRuntime::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->stats;
}

std::vector<float> StreamRuntime::preprocess(const ImageView& image) const { return impl_->preprocessFrame(image); }

void StreamRuntime::Impl::addFrame(double time_s, const std::function<std::vector<float>()>& pixels_of) {
  if (!std::isfinite(time_s)) throw std::invalid_argument("addFrame: capture time is not finite");
  std::lock_guard<std::mutex> order(frame_mutex);  // frames keep their arrival order
  const long index = frame_count;
  // Preprocess outside the stream lock (telemetry keeps flowing meanwhile),
  // and only frames a pair will use.
  std::shared_ptr<const std::vector<float>> pixels;
  if (frameUsed(index)) pixels = std::make_shared<const std::vector<float>>(pixels_of());
  std::lock_guard<std::mutex> lock(mutex);
  insertFrame(index, time_s, std::move(pixels));
}

void StreamRuntime::addFrame(const ImageView& image, double capture_time_s) {
  impl_->checkFrame(image);
  impl_->addFrame(capture_time_s, [&] { return impl_->preprocessFrame(image); });
}

void StreamRuntime::addFrame(double capture_time_s, const std::function<ImageU8()>& load) {
  impl_->addFrame(capture_time_s, [&] {
    const ImageU8 image = load();
    return impl_->preprocessFrame(image.view());
  });
}

// add_telemetry.
TickOutput StreamRuntime::addTelemetry(double time_s, double roll, double pitch, double yaw,
                                       double relative_altitude_m) {
  Impl& m = *impl_;
  std::lock_guard<std::mutex> lock(m.mutex);
  if (m.tracing) {
    char header[64];
    std::snprintf(header, sizeof header, "tick %.17g\n", time_s);
    m.trace << header;
  }
  TickOutput out;
  out.time_s = time_s;
  if (!std::isfinite(time_s) || !std::isfinite(roll) || !std::isfinite(pitch) || !std::isfinite(yaw) ||
      !std::isfinite(relative_altitude_m)) {
    // Training refuses a non-finite row outright. Fed through, one would make
    // the recurrent state NaN for the rest of the flight; skipped, the next
    // good row carries on as after a dropped sample.
    ++m.stats.skipped_rows;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    out.velocity = {nan, nan, nan};
    out.emitted = false;
    out.output = m.emitted;
    out.pair_delivered = false;
    out.log_variance = {nan, nan, nan};
    out.skipped = true;
    return out;
  }
  const double t = time_s;
  Quat q = math::eulerToQuaternion(roll, pitch, yaw);
  if (!m.quats.empty() && math::dot(m.quats.back(), q) < 0.0) {
    for (double& value : q) value = -value;
  }
  double dt = 0.0;
  Vec3 rate{0.0, 0.0, 0.0};
  if (!m.times.empty()) {
    dt = t - m.times.back();
    const Vec3 rotvec = math::quaternionToRotvec(math::multiply(math::conjugate(m.quats.back()), q));
    if (dt > 0) {
      for (std::size_t i = 0; i < 3; ++i) rate[i] = rotvec[i] / dt;
    }
  } else {
    dt = m.delta_time_scale;
    m.have_first_telemetry = true;
    m.first_telemetry_s = t;
  }
  m.times.push_back(t);
  m.quats.push_back(q);
  m.altitudes.push_back(relative_altitude_m);
  while (m.times.size() > 2 && m.times[1] < t - m.history_s) {
    m.times.pop_front();
    m.quats.pop_front();
    m.altitudes.pop_front();
  }

  // Pairs whose result is due by now - on this row, as in training.
  std::vector<PendingPair> due;
  std::vector<PendingPair> waiting;
  for (auto& pair : m.pending) (pair.ready <= t ? due : waiting).push_back(std::move(pair));
  m.pending = std::move(waiting);
  FrontendResult delivered;
  bool present = false;
  for (auto& pair : due) {
    if (pair.ready < m.first_telemetry_s) {
      cancelIfQueued(pair.job);
      continue;
    }
    FrontendResult result;
    if (m.runPair(pair, result)) {
      delivered = std::move(result);  // of two on one row, the later one wins
      present = true;
    }
  }

  std::vector<float> token(m.visual_dim, 0.0f);
  float quality = 0.0f;
  if (present) {
    token = delivered.outputs[m.out_token];
    quality = delivered.outputs[m.out_quality][0];
    m.have_delivery = true;
    m.last_delivery_s = t;
    if (m.geometric) {
      const std::vector<float>& velocity = delivered.outputs[m.out_geometric];
      m.held = {velocity[0], velocity[1], velocity[2]};
      m.held_valid = 1.0f;
    }
  }
  const double age = m.have_delivery ? t - m.last_delivery_s + m.latency : 0.0;

  const double log_altitude = std::log(relative_altitude_m < 1.0 ? 1.0 : relative_altitude_m);
  const double dt_divisor = (1e-9 > m.delta_time_scale) ? 1e-9 : m.delta_time_scale;
  std::array<float, 9> aiding{
      static_cast<float>(std::sin(roll)),
      static_cast<float>(std::cos(roll)),
      static_cast<float>(std::sin(pitch)),
      static_cast<float>(std::cos(pitch)),
      static_cast<float>(log_altitude - m.log_altitude_mean),
      static_cast<float>(rate[0]),
      static_cast<float>(rate[1]),
      static_cast<float>(rate[2]),
      static_cast<float>(dt / dt_divisor),
  };
  float visual_present = present ? 1.0f : 0.0f;
  float visual_age = static_cast<float>(age);
  float log_altitude_f = static_cast<float>(log_altitude);

  std::vector<Ort::Value> values;
  std::vector<const char*> names;
  std::string trace_text;
  std::vector<const char*> output_names;
  for (const auto& name : m.step_output_names) output_names.push_back(name.c_str());
  std::vector<Ort::Value> outputs;
  try {
    for (std::size_t i = 0; i < m.step_input_names.size(); ++i) {
      const std::string& name = m.step_input_names[i];
      const std::vector<std::int64_t>& shape = m.step_input_shapes[i];
      const auto state_it = std::find(m.state_names.begin(), m.state_names.end(), name);
      if (state_it != m.state_names.end()) {
        Ort::Value& value = m.state[static_cast<std::size_t>(state_it - m.state_names.begin())];
        if (m.tracing) {
          traceTensor(trace_text, "step.in", name, value.GetTensorData<float>(),
                      value.GetTensorTypeAndShapeInfo().GetElementCount());
        }
        values.push_back(std::move(value));  // Run reads it; it is put back or replaced below
        names.push_back(name.c_str());
        continue;
      }
      float* data = nullptr;
      std::size_t count = 0;
      if (name == "aiding") {
        data = aiding.data();
        count = aiding.size();
      } else if (name == "visual_token") {
        data = token.data();
        count = token.size();
      } else if (name == "visual_present") {
        data = &visual_present;
        count = 1;
      } else if (name == "visual_age") {
        data = &visual_age;
        count = 1;
      } else if (name == "visual_quality") {
        data = &quality;
        count = 1;
      } else if (name == "log_altitude") {
        data = &log_altitude_f;
        count = 1;
      } else if (name == "visual_velocity") {
        data = m.held.data();
        count = 3;
      } else if (name == "visual_velocity_valid") {
        data = &m.held_valid;
        count = 1;
      } else {
        throw std::logic_error("temporal step input " + name);
      }
      if (m.tracing) traceTensor(trace_text, "step.in", name, data, count);
      values.push_back(Ort::Value::CreateTensor<float>(m.memory_info, data, count, shape.data(), shape.size()));
      names.push_back(name.c_str());
    }
    outputs = m.step->Run(Ort::RunOptions{nullptr}, names.data(), values.data(), values.size(),
                          output_names.data(), output_names.size());
  } catch (...) {
    // Put the state back, so the runtime stays usable and unchanged.
    for (std::size_t i = 0; i < values.size(); ++i) {
      const auto it = std::find(m.state_names.begin(), m.state_names.end(), m.step_input_names[i]);
      if (it != m.state_names.end()) m.state[static_cast<std::size_t>(it - m.state_names.begin())] = std::move(values[i]);
    }
    throw;
  }
  for (std::size_t k = 0; k < outputs.size(); ++k) {
    const auto info = outputs[k].GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("temporal step output " + m.step_output_names[k] + " is not float32");
    }
    if (m.tracing) {
      traceTensor(trace_text, "step.out", m.step_output_names[k], outputs[k].GetTensorData<float>(),
                  info.GetElementCount());
    }
  }
  if (outputs[m.out_velocity].GetTensorTypeAndShapeInfo().GetElementCount() != 3 ||
      outputs[m.out_log_variance].GetTensorTypeAndShapeInfo().GetElementCount() != 3) {
    throw std::runtime_error("temporal step: predicted_velocity / velocity_log_variance are not 3 values");
  }
  const float* velocity = outputs[m.out_velocity].GetTensorData<float>();
  const float* log_variance = outputs[m.out_log_variance].GetTensorData<float>();
  out.velocity = {velocity[0], velocity[1], velocity[2]};
  out.log_variance = {log_variance[0], log_variance[1], log_variance[2]};
  for (std::size_t k = 0; k < m.state_names.size(); ++k) m.state[k] = std::move(outputs[m.state_output_index[k]]);
  if (present || !m.settings.output_on_pairs) m.emitted = out.velocity;
  out.emitted = m.settings.output_on_pairs ? present : true;
  out.output = m.emitted;
  out.pair_delivered = present;
  out.skipped = false;
  if (m.tracing) m.trace << trace_text << std::flush;

  // Start the frontends whose inputs now exist, after this row's own work.
  for (auto& pair : m.pending) m.considerLaunch(pair);
  return out;
}

}  // namespace vo
