// vo_replay - replay a recorded flight folder (flight.csv + images/) through
// the C++ streaming runtime, exactly as tools/onnx_inference.py replays it:
// every frame captured by a telemetry row's time is added before that row,
// then the row. Also the integration template for a live system.
//
//   vo_replay <onnx_dir> --dataset <flight folder> [--max-minutes M] [options]
//   vo_replay <onnx_dir> --events <event log> [options]
//
// options: --output series.csv  --sync  --threads N  --trace file
//          --realtime (pace the rows by their timestamps)  --no-progress
//
// An event log replays a live session's exact arrival order, one event per line:
//   frame <capture_time_s> <image path, relative to the log's folder>
//   row <time_s> <roll> <pitch> <yaw> <relative_altitude_m>
//
// The output CSV has one line per telemetry row: time, the row's velocity,
// the held output, whether a pair was delivered / the row is an output /
// the row was skipped, and the log variance; floats at full precision.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "vo_json.hpp"
#include "vo_math.hpp"
#include "vo_stream.hpp"

namespace fs = std::filesystem;

namespace {

// ---- Python float() on a CSV field ------------------------------------------------

bool isPySpace(unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r') || (c >= 0x1c && c <= 0x1f); }
bool isDigit(char c) { return c >= '0' && c <= '9'; }

// float(text): surrounding whitespace, a sign, inf / infinity / nan in any
// case, and digits grouped with single underscores are accepted; hex is not.
bool pyFloat(const std::string& text, double& out) {
  std::size_t begin = 0;
  std::size_t end = text.size();
  while (begin < end && isPySpace(static_cast<unsigned char>(text[begin]))) ++begin;
  while (end > begin && isPySpace(static_cast<unsigned char>(text[end - 1]))) --end;
  if (begin == end) return false;
  bool negative = false;
  if (text[begin] == '+' || text[begin] == '-') {
    negative = text[begin] == '-';
    ++begin;
  }
  std::string rest = text.substr(begin, end - begin);
  for (char& c : rest) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  if (rest == "inf" || rest == "infinity") {
    out = negative ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    return true;
  }
  if (rest == "nan") {
    out = std::numeric_limits<double>::quiet_NaN();
    return true;
  }
  std::string clean = negative ? "-" : "";
  const std::size_t n = rest.size();
  std::size_t k = 0;
  auto digits = [&]() {
    bool any = false;
    while (k < n) {
      if (isDigit(rest[k])) {
        clean += rest[k++];
        any = true;
      } else if (rest[k] == '_' && k > 0 && isDigit(rest[k - 1]) && k + 1 < n && isDigit(rest[k + 1])) {
        ++k;
      } else {
        break;
      }
    }
    return any;
  };
  const bool integer_digits = digits();
  bool fraction_digits = false;
  if (k < n && rest[k] == '.') {
    clean += rest[k++];
    fraction_digits = digits();
  }
  if (!integer_digits && !fraction_digits) return false;
  if (k < n && rest[k] == 'e') {
    clean += rest[k++];
    if (k < n && (rest[k] == '+' || rest[k] == '-')) clean += rest[k++];
    if (!digits()) return false;
  }
  if (k != n) return false;
  return vo::json::toDouble(clean, out);
}

// ---- Python csv.reader (excel dialect), records in file order ------------------

class CsvReader {
 public:
  explicit CsvReader(const std::string& path) : stream_(path, std::ios::binary) {
    if (!stream_) throw std::runtime_error("cannot open " + path);
    // encoding="utf-8-sig": a leading byte-order mark is not part of the header.
    char bom[3] = {0, 0, 0};
    stream_.read(bom, 3);
    if (!(stream_.gcount() == 3 && bom[0] == '\xEF' && bom[1] == '\xBB' && bom[2] == '\xBF')) {
      stream_.clear();
      stream_.seekg(0);
    }
  }

  // The next record; false at the end of the file. A blank line is a record
  // with no fields (the caller skips those, as `if row` does).
  bool next(std::vector<std::string>& fields) {
    fields.clear();
    field_.clear();
    field_open_ = false;
    state_ = State::StartRecord;
    bool any_char = false;
    for (;;) {
      const int c = stream_.get();
      if (c == std::char_traits<char>::eof()) {
        if (!any_char) return false;
        // The last line had no terminator: its end of line still closes it.
        process(kEol, fields);
        if (state_ == State::StartRecord) return true;
        // EOF inside a quoted field: Python (strict=False) keeps what it has.
        saveField(fields);
        return true;
      }
      any_char = true;
      process(c, fields);
      // Python reads line by line and feeds an end-of-line after each line;
      // a line ends after '\n', or after a '\r' not followed by '\n'.
      bool line_end = c == '\n';
      if (c == '\r') {
        if (stream_.peek() == '\n') {
          process(stream_.get(), fields);
        }
        line_end = true;
      }
      if (line_end) {
        process(kEol, fields);
        if (state_ == State::StartRecord) return true;
      }
    }
  }

 private:
  static constexpr int kEol = -2;
  enum class State { StartRecord, StartField, InField, InQuotedField, QuoteInQuotedField, EatCrnl };
  std::ifstream stream_;
  State state_ = State::StartRecord;
  std::string field_;
  bool field_open_ = false;

  void saveField(std::vector<std::string>& fields) {
    fields.push_back(field_);
    field_.clear();
  }

  // _csv.c parse_process_char.
  void process(int c, std::vector<std::string>& fields) {
    switch (state_) {
      case State::StartRecord:
        if (c == kEol) return;  // empty line: []
        if (c == '\n' || c == '\r') {
          state_ = State::EatCrnl;
          return;
        }
        state_ = State::StartField;
        [[fallthrough]];
      case State::StartField:
        if (c == '\n' || c == '\r' || c == kEol) {
          saveField(fields);
          state_ = c == kEol ? State::StartRecord : State::EatCrnl;
        } else if (c == '"') {
          state_ = State::InQuotedField;
        } else if (c == ',') {
          saveField(fields);
        } else {
          field_ += static_cast<char>(c);
          state_ = State::InField;
        }
        return;
      case State::InField:
        if (c == '\n' || c == '\r' || c == kEol) {
          saveField(fields);
          state_ = c == kEol ? State::StartRecord : State::EatCrnl;
        } else if (c == ',') {
          saveField(fields);
          state_ = State::StartField;
        } else {
          field_ += static_cast<char>(c);
        }
        return;
      case State::InQuotedField:
        if (c == kEol) return;
        if (c == '"') {
          state_ = State::QuoteInQuotedField;
        } else {
          field_ += static_cast<char>(c);
        }
        return;
      case State::QuoteInQuotedField:
        if (c == '"') {
          field_ += '"';
          state_ = State::InQuotedField;
        } else if (c == ',') {
          saveField(fields);
          state_ = State::StartField;
        } else if (c == '\n' || c == '\r' || c == kEol) {
          saveField(fields);
          state_ = c == kEol ? State::StartRecord : State::EatCrnl;
        } else {
          field_ += static_cast<char>(c);
          state_ = State::InField;
        }
        return;
      case State::EatCrnl:
        if (c == '\n' || c == '\r') return;
        if (c == kEol) {
          state_ = State::StartRecord;
          return;
        }
        throw std::runtime_error("CSV: new-line character seen in unquoted field");
    }
  }
};

// ---- read_flight_csv ------------------------------------------------------------------

struct Flight {
  std::vector<double> times;
  std::vector<std::array<double, 3>> euler;
  std::vector<double> altitude;
};

Flight readFlightCsv(const fs::path& path, const vo::Settings& settings) {
  CsvReader reader(path.string());
  std::vector<std::string> header;
  if (!reader.next(header)) throw std::runtime_error("empty CSV " + path.string());
  auto has = [&](const std::string& name) { return std::find(header.begin(), header.end(), name) != header.end(); };
  // Same choice order as vio.data.attitude (never GPSNavEul*, part of the target).
  std::vector<std::string> attitude = settings.attitude_columns;
  if (attitude.empty()) {
    for (const auto& candidate : {std::vector<std::string>{"NavEulX", "NavEulY", "NavEulZ"},
                                  std::vector<std::string>{"EulX", "EulY", "EulZ"}}) {
      if (std::all_of(candidate.begin(), candidate.end(), has)) {
        attitude = candidate;
        break;
      }
    }
  }
  std::string altitude = settings.altitude_column;
  if (altitude.empty()) {
    for (const char* candidate : {"relativeAlt", "RelativeAlt", "RelatedAlt", "Barometer"}) {
      if (has(candidate)) {
        altitude = candidate;
        break;
      }
    }
  }
  std::vector<std::string> names{settings.time_column};
  names.insert(names.end(), attitude.begin(), attitude.end());
  names.push_back(altitude);
  if (attitude.empty() || altitude.empty() || !std::all_of(names.begin(), names.end(), has)) {
    throw std::runtime_error("cannot find time/attitude/altitude columns in " + path.filename().string());
  }
  std::vector<std::size_t> indices;  // first occurrence, as training
  for (const auto& name : names) {
    indices.push_back(static_cast<std::size_t>(std::find(header.begin(), header.end(), name) - header.begin()));
  }

  Flight flight;
  std::vector<std::string> row;
  long line = 1;
  while (reader.next(row)) {
    ++line;
    if (row.empty()) continue;
    double values[5];
    for (std::size_t k = 0; k < 5; ++k) {
      if (indices[k] >= row.size() || !pyFloat(row[indices[k]], values[k])) {
        throw std::runtime_error("CSV record " + std::to_string(line) + ": bad or missing " + names[k]);
      }
    }
    flight.times.push_back(values[0] * settings.time_scale);
    flight.euler.push_back({values[1], values[2], values[3]});
    flight.altitude.push_back(values[4]);
  }
  // Same unit rule as training: a yaw in radians never exceeds 2*pi.
  double largest = std::numeric_limits<double>::quiet_NaN();  // np.nanmax(np.abs(euler))
  for (const auto& angles : flight.euler) {
    for (const double angle : angles) {
      if (!std::isnan(angle) && (std::isnan(largest) || std::fabs(angle) > largest)) largest = std::fabs(angle);
    }
  }
  if (largest > 2 * 3.141592653589793 + 1e-6) {
    constexpr double kDegToRad = 3.141592653589793238462643383279502884 / 180.0;  // np.deg2rad
    for (auto& angles : flight.euler) {
      for (double& angle : angles) angle = angle * kDegToRad;
    }
  }
  return flight;
}

// ---- list_frames ------------------------------------------------------------------------

// fnmatch for one file name: *, ?, [seq], [!seq] (case-sensitive, as on POSIX).
bool fnmatch(const char* pattern, const char* name) {
  if (*pattern == '\0') return *name == '\0';
  if (*pattern == '*') {
    for (const char* rest = name;; ++rest) {
      if (fnmatch(pattern + 1, rest)) return true;
      if (*rest == '\0') return false;
    }
  }
  if (*name == '\0') return false;
  if (*pattern == '?') return fnmatch(pattern + 1, name + 1);
  if (*pattern == '[') {
    const char* p = pattern + 1;
    bool negate = false;
    if (*p == '!') {
      negate = true;
      ++p;
    }
    const char* start = p;
    bool matched = false;
    while (*p != '\0' && (*p != ']' || p == start)) {
      if (p[1] == '-' && p[2] != '\0' && p[2] != ']') {
        if (*p <= *name && *name <= p[2]) matched = true;
        p += 3;
      } else {
        if (*p == *name) matched = true;
        ++p;
      }
    }
    if (*p != ']') return *name == '[' && fnmatch(pattern + 1, name + 1);  // no closing ']': a literal '['
    if (matched == negate) return false;
    return fnmatch(p + 1, name + 1);
  }
  return *pattern == *name && fnmatch(pattern + 1, name + 1);
}

std::vector<std::pair<double, fs::path>> listFrames(const fs::path& folder, const vo::Settings& settings) {
  if (settings.image_time_offset_is_table) {
    throw std::runtime_error("a per-time offset table is not supported here; use a constant --image-time-offset");
  }
  if (settings.image_pattern.find('/') != std::string::npos || settings.image_pattern.find("**") != std::string::npos) {
    throw std::runtime_error("image_pattern must be a plain file pattern: " + settings.image_pattern);
  }
  std::vector<std::pair<double, fs::path>> frames;
  for (const auto& entry : fs::directory_iterator(folder)) {  // directory order, as pathlib.glob
    const std::string name = entry.path().filename().string();
    if (!fnmatch(settings.image_pattern.c_str(), name.c_str()) || !entry.is_regular_file()) continue;
    // path.stem, then the part after the last '_'.
    const std::size_t dot = name.rfind('.');
    const std::string stem = (dot != std::string::npos && dot > 0 && dot < name.size() - 1) ? name.substr(0, dot) : name;
    const std::size_t underscore = stem.rfind('_');
    const std::string stamp = underscore == std::string::npos ? stem : stem.substr(underscore + 1);
    if (stamp.empty() || !std::all_of(stamp.begin(), stamp.end(), isDigit)) continue;
    double value = 0.0;
    if (!vo::json::toDouble(stamp, value)) continue;  // float(int(stamp)), correctly rounded
    frames.emplace_back(value * settings.image_time_scale + settings.image_time_offset_s, entry.path());
  }
  std::stable_sort(frames.begin(), frames.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  return frames;
}

// ---- main -------------------------------------------------------------------------------

// ---- event logs ---------------------------------------------------------------------------

struct Event {
  bool is_frame = false;
  double values[5] = {0, 0, 0, 0, 0};  // frame: capture time; row: t, roll, pitch, yaw, altitude
  fs::path path;
};

std::vector<Event> readEvents(const fs::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  std::vector<Event> events;
  std::string line;
  long number = 0;
  while (std::getline(stream, line)) {
    ++number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    auto fail = [&]() { return std::runtime_error(path.string() + ":" + std::to_string(number) + ": bad event"); };
    std::size_t pos = 0;
    auto token = [&]() {
      while (pos < line.size() && line[pos] == ' ') ++pos;
      const std::size_t start = pos;
      while (pos < line.size() && line[pos] != ' ') ++pos;
      return line.substr(start, pos - start);
    };
    Event event;
    const std::string kind = token();
    if (kind == "frame") {
      event.is_frame = true;
      if (!pyFloat(token(), event.values[0])) throw fail();
      while (pos < line.size() && line[pos] == ' ') ++pos;
      if (pos >= line.size()) throw fail();
      event.path = fs::path(line.substr(pos));
      if (event.path.is_relative()) event.path = path.parent_path() / event.path;
    } else if (kind == "row") {
      for (double& value : event.values) {
        if (!pyFloat(token(), value)) throw fail();
      }
    } else {
      throw fail();
    }
    events.push_back(std::move(event));
  }
  return events;
}

int usage() {
  std::fprintf(stderr,
               "usage: vo_replay <onnx_dir> --dataset <flight folder> [--max-minutes M] [options]\n"
               "       vo_replay <onnx_dir> --events <event log> [options]\n"
               "options: --output series.csv  --sync  --threads N  --trace file  --realtime  --no-progress\n");
  return 2;
}

struct Timing {
  std::vector<double> ms;
  void add(double value) { ms.push_back(value); }
  std::string summary() {
    if (ms.empty()) return "n/a";
    std::vector<double> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    for (const double v : sorted) sum += v;
    char text[160];
    std::snprintf(text, sizeof text, "mean %.3f ms, p50 %.3f, p99 %.3f, max %.3f (%zu calls)", sum / sorted.size(),
                  sorted[sorted.size() / 2], sorted[std::min(sorted.size() - 1, sorted.size() * 99 / 100)],
                  sorted.back(), sorted.size());
    return text;
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  std::string onnx_dir = argv[1];
  std::string dataset;
  std::string events_path;
  bool realtime = false;
  std::string output;
  std::string trace;
  double max_minutes = -1.0;
  bool progress = true;
  vo::Options options;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
      return argv[++i];
    };
    try {
      if (arg == "--dataset") {
        dataset = value();
      } else if (arg == "--events") {
        events_path = value();
      } else if (arg == "--realtime") {
        realtime = true;
      } else if (arg == "--output") {
        output = value();
      } else if (arg == "--max-minutes") {
        max_minutes = std::stod(value());
      } else if (arg == "--sync") {
        options.async_frontend = false;
      } else if (arg == "--threads") {
        options.intra_op_threads = std::stoi(value());
      } else if (arg == "--trace") {
        options.trace_path = value();
      } else if (arg == "--no-progress") {
        progress = false;
      } else {
        return usage();
      }
    } catch (const std::exception& error) {
      std::fprintf(stderr, "vo_replay: %s\n", error.what());
      return usage();
    }
  }
  if (dataset.empty() == events_path.empty()) return usage();

  try {
    vo::StreamRuntime vo(onnx_dir, options);
    const vo::Settings& settings = vo.settings();
    // Both inputs become one event list: the flight folder in the order
    // tools/onnx_inference.py replays it (every frame captured by a row's
    // time, then the row), or a recorded log as it is.
    std::vector<Event> events;
    if (!dataset.empty()) {
      const fs::path folder(dataset);
      const Flight flight = readFlightCsv(folder / settings.csv_name, settings);
      const auto frames = listFrames(folder / settings.image_folder, settings);
      std::size_t total = flight.times.size();
      if (max_minutes >= 0.0 && total > 0) {
        total = vo::math::searchSorted(flight.times, flight.times[0] + 60.0 * max_minutes, false);
      }
      std::size_t next_frame = 0;
      for (std::size_t tick = 0; tick < total; ++tick) {
        while (next_frame < frames.size() && frames[next_frame].first <= flight.times[tick]) {
          Event frame;
          frame.is_frame = true;
          frame.values[0] = frames[next_frame].first;
          frame.path = frames[next_frame].second;
          events.push_back(std::move(frame));
          ++next_frame;
        }
        Event row;
        const auto& e = flight.euler[tick];
        const double values[5] = {flight.times[tick], e[0], e[1], e[2], flight.altitude[tick]};
        std::copy(values, values + 5, row.values);
        events.push_back(std::move(row));
      }
    } else {
      events = readEvents(events_path);
    }
    std::size_t total = 0;
    std::size_t frame_count = 0;
    for (const auto& event : events) (event.is_frame ? frame_count : total) += 1;
    std::printf("rows %zu, frames %zu, %s frontend%s\n", total, frame_count,
                options.async_frontend ? "async" : "sync", realtime ? ", real time" : "");

    std::FILE* out = nullptr;
    if (!output.empty()) {
      out = std::fopen(output.c_str(), "w");
      if (out == nullptr) throw std::runtime_error("cannot write " + output);
      std::fprintf(out, "time_s,vx,vy,vz,out_vx,out_vy,out_vz,pair_delivered,emitted,skipped,logvar_x,logvar_y,logvar_z\n");
    }
    Timing telemetry_ms;
    Timing frame_ms;
    const std::size_t report_every = std::max<std::size_t>(total / 10, 1);
    using Clock = std::chrono::steady_clock;
    const Clock::time_point wall_start = Clock::now();
    bool have_first = false;
    double first_time = 0.0;
    std::size_t tick = 0;
    for (const Event& event : events) {
      if (realtime && std::isfinite(event.values[0])) {
        // Wall clock follows the stream's clock: an event waits for its time.
        if (!have_first) {
          have_first = true;
          first_time = event.values[0];
        }
        const double offset = event.values[0] - first_time;
        if (offset > 0) {
          std::this_thread::sleep_until(wall_start + std::chrono::duration_cast<Clock::duration>(
                                                         std::chrono::duration<double>(offset)));
        }
      }
      if (event.is_frame) {
        const fs::path& path = event.path;
        const auto start = Clock::now();
        vo.addFrame(event.values[0], [&] { return vo::loadImageFile(path.string(), settings.color); });
        frame_ms.add(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        continue;
      }
      const double* v = event.values;
      const auto start = Clock::now();
      const vo::TickOutput row = vo.addTelemetry(v[0], v[1], v[2], v[3], v[4]);
      telemetry_ms.add(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
      if (out != nullptr) {
        std::fprintf(out, "%.17g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d,%d,%.9g,%.9g,%.9g\n", row.time_s,
                     row.velocity[0], row.velocity[1], row.velocity[2], row.output[0], row.output[1], row.output[2],
                     row.pair_delivered ? 1 : 0, row.emitted ? 1 : 0, row.skipped ? 1 : 0, row.log_variance[0],
                     row.log_variance[1], row.log_variance[2]);
      }
      if (progress && tick % report_every == 0) {
        std::printf("  tick %zu/%zu  pairs delivered %ld\n", tick, total, vo.stats().delivered);
        std::fflush(stdout);
      }
      ++tick;
    }
    if (out != nullptr) std::fclose(out);
    const vo::Stats s = vo.stats();
    std::printf("pairs %ld, delivered %ld, refused %ld, rejected_gap %ld, skipped_rows %ld, late_pairs %ld\n",
                s.pairs, s.delivered, s.refused, s.rejected_gap, s.skipped_rows, s.late_pairs);
    std::printf("async: worker results used %ld, recomputed %ld; longest frontend wait on a due row %.3f ms\n",
                s.async_runs, s.async_recomputed, s.max_due_frontend_ms);
    std::printf("addTelemetry: %s\n", telemetry_ms.summary().c_str());
    std::printf("addFrame (incl. file decode): %s\n", frame_ms.summary().c_str());
    if (!output.empty()) std::printf("series -> %s\n", output.c_str());
  } catch (const std::exception& error) {
    std::fprintf(stderr, "vo_replay: %s\n", error.what());
    return 1;
  }
  return 0;
}
