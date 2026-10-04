// vo_math.hpp - the Python runtime's arithmetic, operation for operation (internal).
//
// Every function here mirrors one in tools/onnx_inference.py (or the numpy
// routine it calls), in the same order of floating-point operations, so the
// float32 tensors handed to ONNX Runtime come out the same. Build without FMA
// contraction (-ffp-contract=off, see CMakeLists.txt): a fused multiply-add
// rounds differently - conj(q) * q, for one, would no longer be exactly zero.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace vo {
namespace math {

using Quat = std::array<double, 4>;  // w, x, y, z (body-to-NED)
using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;

// np.linalg.norm of a float64 vector: sqrt(x.dot(x)), the dot summed left to right.
template <std::size_t N>
inline double norm(const std::array<double, N>& v) {
  double sum = 0.0;
  for (const double value : v) sum += value * value;
  return std::sqrt(sum);
}

template <std::size_t N>
inline std::array<double, N> divided(const std::array<double, N>& v, double divisor) {
  std::array<double, N> out;
  for (std::size_t i = 0; i < N; ++i) out[i] = v[i] / divisor;
  return out;
}

// euler_to_quaternion: intrinsic Z-Y-X (yaw, pitch, roll), normalised.
inline Quat eulerToQuaternion(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll / 2), sr = std::sin(roll / 2);
  const double cp = std::cos(pitch / 2), sp = std::sin(pitch / 2);
  const double cy = std::cos(yaw / 2), sy = std::sin(yaw / 2);
  const Quat q{
      cy * cp * cr + sy * sp * sr,
      cy * cp * sr - sy * sp * cr,
      sy * cp * sr + cy * sp * cr,
      sy * cp * cr - cy * sp * sr,
  };
  return divided(q, norm(q));
}

// quaternion_multiply (Hamilton product a * b).
inline Quat multiply(const Quat& a, const Quat& b) {
  const double aw = a[0], ax = a[1], ay = a[2], az = a[3];
  const double bw = b[0], bx = b[1], by = b[2], bz = b[3];
  return Quat{
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
  };
}

// q * np.array([1, -1, -1, -1]).
inline Quat conjugate(const Quat& q) { return Quat{q[0] * 1.0, q[1] * -1.0, q[2] * -1.0, q[3] * -1.0}; }

inline double dot(const Quat& a, const Quat& b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < 4; ++i) sum += a[i] * b[i];
  return sum;
}

// quaternion_to_rotvec.
inline Vec3 quaternionToRotvec(const Quat& input) {
  Quat q = divided(input, norm(input));
  if (q[0] < 0) {
    for (double& value : q) value = -value;
  }
  const Vec3 vector{q[1], q[2], q[3]};
  const double length = norm(vector);
  Vec3 out;
  if (length <= 1e-10) {
    for (std::size_t i = 0; i < 3; ++i) out[i] = vector[i] * 2.0;
    return out;
  }
  const double w = (0.0 > q[0]) ? 0.0 : q[0];  // Python max(q[0], 0.0)
  const double scale = 2.0 * std::atan2(length, w) / length;
  for (std::size_t i = 0; i < 3; ++i) out[i] = vector[i] * scale;
  return out;
}

// quaternion_to_matrix (normalises first, as Python does).
inline Mat3 quaternionToMatrix(const Quat& input) {
  const Quat q = divided(input, norm(input));
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  Mat3 r;
  r[0] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)};
  r[1] = {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)};
  r[2] = {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
  return r;
}

// r0.T @ r1, each entry summed over k in order.
inline Mat3 transposeTimes(const Mat3& r0, const Mat3& r1) {
  Mat3 out;
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      double sum = r0[0][i] * r1[0][j];
      sum += r0[1][i] * r1[1][j];
      sum += r0[2][i] * r1[2][j];
      out[i][j] = sum;
    }
  }
  return out;
}

// numpy's float64 ordering (NaN sorts last).
inline bool numpyLess(double a, double b) { return a < b || (b != b && a == a); }

// np.searchsorted(a, key, side) for one key: numpy's bisection, exactly (it
// is also what numpy returns for an array that is not sorted).
template <class Array>
std::size_t searchSorted(const Array& a, double key, bool right) {
  std::size_t lo = 0;
  std::size_t hi = a.size();
  while (lo < hi) {
    const std::size_t mid = lo + ((hi - lo) >> 1);
    const bool before = right ? !numpyLess(key, a[mid]) : numpyLess(a[mid], key);
    if (before) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// numpy's binary_search_with_guess (compiled_base.c), guess 0: the index j
// with arr[j] <= key < arr[j + 1]; -1 below the range, len above it.
template <class Array>
long binarySearchWithGuess(double key, const Array& arr) {
  const long len = static_cast<long>(arr.size());
  constexpr long kLikelyInCache = 8;
  long imin = 0;
  long imax = len;
  if (key > arr[len - 1]) return len;
  if (key < arr[0]) return -1;
  if (len <= 4) {
    long i = 1;
    while (i < len && key >= arr[i]) ++i;
    return i - 1;
  }
  long guess = 0;
  if (guess > len - 3) guess = len - 3;
  if (guess < 1) guess = 1;
  if (key < arr[guess]) {
    if (key < arr[guess - 1]) {
      imax = guess - 1;
      if (guess > kLikelyInCache && key >= arr[guess - kLikelyInCache]) imin = guess - kLikelyInCache;
    } else {
      return guess - 1;
    }
  } else {
    if (key < arr[guess + 1]) return guess;
    if (key < arr[guess + 2]) return guess + 1;
    imin = guess + 2;
    if (guess < len - kLikelyInCache - 1 && key < arr[guess + kLikelyInCache]) imax = guess + kLikelyInCache;
  }
  while (imin < imax) {
    const long imid = imin + ((imax - imin) >> 1);
    if (key >= arr[imid]) {
      imin = imid + 1;
    } else {
      imax = imid;
    }
  }
  return imin - 1;
}

// np.interp(x, xp, fp) for one x (default left/right: the end values).
template <class Array>
double interp(double x, const Array& xp, const Array& fp) {
  const long len = static_cast<long>(xp.size());
  if (len == 1) return fp[0];  // left, right and the one value are all fp[0]
  if (std::isnan(x)) return x;
  const long j = binarySearchWithGuess(x, xp);
  if (j == -1) return fp[0];
  if (j == len) return fp[static_cast<std::size_t>(len - 1)];
  if (j == len - 1) return fp[static_cast<std::size_t>(j)];
  const std::size_t i = static_cast<std::size_t>(j);
  if (xp[i] == x) return fp[i];
  const double slope = (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i]);
  double result = slope * (x - xp[i]) + fp[i];
  if (std::isnan(result)) {
    result = slope * (x - xp[i + 1]) + fp[i + 1];
    if (std::isnan(result) && fp[i] == fp[i + 1]) result = fp[i];
  }
  return result;
}

}  // namespace math
}  // namespace vo
