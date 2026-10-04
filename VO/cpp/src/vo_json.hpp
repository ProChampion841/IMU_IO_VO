// vo_json.hpp - a small JSON reader for vo_onnx.json (internal).
//
// Reads what Python's json.dumps writes, including its non-standard NaN,
// Infinity and -Infinity, \uXXXX escapes (surrogate pairs too), and keeps an
// object's keys in file order - the frontend's input list is an object whose
// key order is the order the graph's inputs were exported in.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace vo {
namespace json {

class Value {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  std::vector<Value> array;
  std::vector<std::pair<std::string, Value>> object;

  bool isNull() const { return type == Type::Null; }
  bool isNumber() const { return type == Type::Number; }
  bool isString() const { return type == Type::String; }
  bool isArray() const { return type == Type::Array; }
  bool isObject() const { return type == Type::Object; }

  // nullptr when this is not an object or the key is absent.
  const Value* find(const std::string& key) const;
  // Throws std::runtime_error naming the key when absent.
  const Value& at(const std::string& key) const;
  const Value& at(std::size_t index) const;

  double asNumber() const;
  bool asBool() const;
  const std::string& asString() const;
  // Python truthiness, for porting `x or default`.
  bool truthy() const;
};

Value parse(const std::string& text);
Value parseFile(const std::string& path);

// A decimal number (JSON grammar, already validated by the caller) to the
// nearest double, whatever the process locale. False if it does not parse
// or is out of the double range.
bool toDouble(const std::string& text, double& out);

}  // namespace json
}  // namespace vo
