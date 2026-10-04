#include "vo_json.hpp"

#include <charconv>
#include <locale>
#include <system_error>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace vo {
namespace json {

namespace {

class Parser {
 public:
  explicit Parser(const std::string& text) : text_(text) {}

  Value parseDocument() {
    Value value = parseValue();
    skipSpace();
    if (pos_ != text_.size()) fail("trailing characters after the JSON value");
    return value;
  }

 private:
  const std::string& text_;
  std::size_t pos_ = 0;

  [[noreturn]] void fail(const std::string& what) const {
    std::ostringstream message;
    message << "JSON parse error at offset " << pos_ << ": " << what;
    throw std::runtime_error(message.str());
  }

  void skipSpace() {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else {
        break;
      }
    }
  }

  bool consumeLiteral(const char* literal) {
    std::size_t length = 0;
    while (literal[length] != '\0') ++length;
    if (text_.compare(pos_, length, literal) == 0) {
      pos_ += length;
      return true;
    }
    return false;
  }

  Value parseValue() {
    skipSpace();
    if (pos_ >= text_.size()) fail("unexpected end of input");
    const char c = text_[pos_];
    Value value;
    if (c == '{') return parseObject();
    if (c == '[') return parseArray();
    if (c == '"') {
      value.type = Value::Type::String;
      value.string = parseString();
      return value;
    }
    if (consumeLiteral("true")) {
      value.type = Value::Type::Bool;
      value.boolean = true;
      return value;
    }
    if (consumeLiteral("false")) {
      value.type = Value::Type::Bool;
      value.boolean = false;
      return value;
    }
    if (consumeLiteral("null")) return value;
    // Python's json writes these for float('nan') / float('inf').
    if (consumeLiteral("NaN")) {
      value.type = Value::Type::Number;
      value.number = std::numeric_limits<double>::quiet_NaN();
      return value;
    }
    if (consumeLiteral("Infinity")) {
      value.type = Value::Type::Number;
      value.number = std::numeric_limits<double>::infinity();
      return value;
    }
    if (consumeLiteral("-Infinity")) {
      value.type = Value::Type::Number;
      value.number = -std::numeric_limits<double>::infinity();
      return value;
    }
    if (c == '-' || (c >= '0' && c <= '9')) return parseNumber();
    fail(std::string("unexpected character '") + c + "'");
  }

  Value parseNumber() {
    // The JSON number grammar, checked by hand; strtod then converts the
    // validated span (correctly rounded, like Python's float()).
    const std::size_t start = pos_;
    if (text_[pos_] == '-') ++pos_;
    if (pos_ >= text_.size()) fail("truncated number");
    if (text_[pos_] == '0') {
      ++pos_;
    } else if (text_[pos_] >= '1' && text_[pos_] <= '9') {
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
    } else {
      fail("invalid number");
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') fail("invalid fraction");
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
      if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') fail("invalid exponent");
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
    }
    const std::string span = text_.substr(start, pos_ - start);
    double number = 0.0;
    if (!toDouble(span, number)) fail("invalid or out-of-range number '" + span + "'");
    Value value;
    value.type = Value::Type::Number;
    value.number = number;
    return value;
  }

  static void appendUtf8(std::string& out, unsigned long code) {
    if (code < 0x80) {
      out.push_back(static_cast<char>(code));
    } else if (code < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (code >> 6)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else if (code < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (code >> 12)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (code >> 18)));
      out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
  }

  unsigned long parseHex4() {
    if (pos_ + 4 > text_.size()) fail("truncated \\u escape");
    unsigned long code = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = text_[pos_++];
      code <<= 4;
      if (c >= '0' && c <= '9') {
        code |= static_cast<unsigned long>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        code |= static_cast<unsigned long>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        code |= static_cast<unsigned long>(c - 'A' + 10);
      } else {
        fail("invalid \\u escape");
      }
    }
    return code;
  }

  std::string parseString() {
    ++pos_;  // opening quote
    std::string out;
    while (true) {
      if (pos_ >= text_.size()) fail("unterminated string");
      const char c = text_[pos_++];
      if (c == '"') break;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (pos_ >= text_.size()) fail("unterminated escape");
      const char e = text_[pos_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          unsigned long code = parseHex4();
          if (code >= 0xD800 && code <= 0xDBFF && pos_ + 6 <= text_.size() &&
              text_[pos_] == '\\' && text_[pos_ + 1] == 'u') {
            const std::size_t save = pos_;
            pos_ += 2;
            const unsigned long low = parseHex4();
            if (low >= 0xDC00 && low <= 0xDFFF) {
              code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
            } else {
              pos_ = save;  // lone high surrogate: keep it as is
            }
          }
          appendUtf8(out, code);
          break;
        }
        default:
          fail(std::string("invalid escape \\") + e);
      }
    }
    return out;
  }

  Value parseArray() {
    ++pos_;  // [
    Value value;
    value.type = Value::Type::Array;
    skipSpace();
    if (pos_ < text_.size() && text_[pos_] == ']') {
      ++pos_;
      return value;
    }
    while (true) {
      value.array.push_back(parseValue());
      skipSpace();
      if (pos_ >= text_.size()) fail("unterminated array");
      const char c = text_[pos_++];
      if (c == ']') break;
      if (c != ',') fail("expected ',' or ']' in array");
    }
    return value;
  }

  Value parseObject() {
    ++pos_;  // {
    Value value;
    value.type = Value::Type::Object;
    skipSpace();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      return value;
    }
    while (true) {
      skipSpace();
      if (pos_ >= text_.size() || text_[pos_] != '"') fail("expected a string key");
      std::string key = parseString();
      skipSpace();
      if (pos_ >= text_.size() || text_[pos_] != ':') fail("expected ':' after key");
      ++pos_;
      Value member = parseValue();
      // Python's json keeps the LAST of duplicate keys; so do we.
      bool replaced = false;
      for (auto& entry : value.object) {
        if (entry.first == key) {
          entry.second = std::move(member);
          replaced = true;
          break;
        }
      }
      if (!replaced) value.object.emplace_back(std::move(key), std::move(member));
      skipSpace();
      if (pos_ >= text_.size()) fail("unterminated object");
      const char c = text_[pos_++];
      if (c == '}') break;
      if (c != ',') fail("expected ',' or '}' in object");
    }
    return value;
  }
};

}  // namespace

const Value* Value::find(const std::string& key) const {
  if (type != Type::Object) return nullptr;
  for (const auto& entry : object) {
    if (entry.first == key) return &entry.second;
  }
  return nullptr;
}

const Value& Value::at(const std::string& key) const {
  const Value* found = find(key);
  if (found == nullptr) throw std::runtime_error("vo_onnx.json: missing key '" + key + "'");
  return *found;
}

const Value& Value::at(std::size_t index) const {
  if (type != Type::Array || index >= array.size()) {
    throw std::runtime_error("vo_onnx.json: array index out of range");
  }
  return array[index];
}

double Value::asNumber() const {
  if (type == Type::Number) return number;
  if (type == Type::Bool) return boolean ? 1.0 : 0.0;
  throw std::runtime_error("vo_onnx.json: expected a number");
}

bool Value::asBool() const {
  if (type == Type::Bool) return boolean;
  if (type == Type::Null) return false;
  if (type == Type::Number) return number != 0.0;
  throw std::runtime_error("vo_onnx.json: expected a boolean");
}

const std::string& Value::asString() const {
  if (type != Type::String) throw std::runtime_error("vo_onnx.json: expected a string");
  return string;
}

bool Value::truthy() const {
  switch (type) {
    case Type::Null: return false;
    case Type::Bool: return boolean;
    case Type::Number: return number != 0.0;  // NaN is truthy in Python, and here
    case Type::String: return !string.empty();
    case Type::Array: return !array.empty();
    case Type::Object: return !object.empty();
  }
  return false;
}

bool toDouble(const std::string& text, double& out) {
  if (text.empty()) return false;
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
  // Correctly rounded, like Python's float(), and locale-independent.
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, out, std::chars_format::general);
  return result.ec == std::errc() && result.ptr == end;
#else
  // Older standard libraries: an istream in the classic locale (strtod inside).
  std::istringstream stream(text);
  stream.imbue(std::locale::classic());
  double value = 0.0;
  stream >> value;
  if (stream.fail() || stream.peek() != std::char_traits<char>::eof()) return false;
  out = value;
  return true;
#endif
}

Value parse(const std::string& text) { return Parser(text).parseDocument(); }

Value parseFile(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path);
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return parse(buffer.str());
}

}  // namespace json
}  // namespace vo
