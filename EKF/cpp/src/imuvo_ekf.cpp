// Implementation of imuvo_ekf.hpp.  Mirrors EKF/ekf/eskf.py and EKF/ekf/stream.py.
#include "imuvo_ekf.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

namespace imuvo {
namespace {

constexpr double kPi = 3.14159265358979323846;
inline double deg2rad(double d) { return d * kPi / 180.0; }

// 99.9 % chi-square quantiles, dof 1..3 (same table as eskf.py)
constexpr double kChi2[4] = {0.0, 10.828, 13.816, 16.266};

// ---- 3x3 / 3-vector helpers --------------------------------------------------
inline Mat3 mul(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
    return C;
}
inline Vec3 mul(const Mat3& A, const Vec3& x) {
    return {A[0] * x[0] + A[1] * x[1] + A[2] * x[2], A[3] * x[0] + A[4] * x[1] + A[5] * x[2],
            A[6] * x[0] + A[7] * x[1] + A[8] * x[2]};
}
inline Mat3 transpose(const Mat3& A) {
    return {A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]};
}
inline Mat3 skew(const Vec3& v) { return {0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0}; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline Mat3 inverse3(const Mat3& A) {
    const double c00 = A[4] * A[8] - A[5] * A[7], c01 = A[5] * A[6] - A[3] * A[8],
                 c02 = A[3] * A[7] - A[4] * A[6];
    const double det = A[0] * c00 + A[1] * c01 + A[2] * c02;
    const double id = 1.0 / det;
    return {c00 * id, (A[2] * A[7] - A[1] * A[8]) * id, (A[1] * A[5] - A[2] * A[4]) * id,
            c01 * id, (A[0] * A[8] - A[2] * A[6]) * id, (A[2] * A[3] - A[0] * A[5]) * id,
            c02 * id, (A[1] * A[6] - A[0] * A[7]) * id, (A[0] * A[4] - A[1] * A[3]) * id};
}
// Polar factor (the SVD u*vt eskf.py uses): Newton iteration R <- (R + R^-T)/2.
inline Mat3 orthonormalize(Mat3 R) {
    for (int k = 0; k < 3; ++k) {
        const Mat3 Rit = transpose(inverse3(R));
        for (int i = 0; i < 9; ++i) R[i] = 0.5 * (R[i] + Rit[i]);
    }
    return R;
}

// ---- 15x15 helpers ------------------------------------------------------------
using M15 = std::array<double, N * N>;
inline void setBlock(M15& M, int r, int c, const Mat3& B, double s = 1.0) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) M[(r + i) * N + c + j] = s * B[i * 3 + j];
}
inline void mul15(const M15& A, const M15& B, M15& C) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < N; ++k) s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}
inline void mul15T(const M15& A, const M15& B, M15& C) {     // A * B^T
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < N; ++k) s += A[i * N + k] * B[j * N + k];
            C[i * N + j] = s;
        }
}
// Small symmetric positive-definite inverse (m <= 3) by Gauss-Jordan.
inline bool invSmall(int m, const double* S, double* Si) {
    double a[3][6] = {};
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) a[i][j] = S[i * m + j];
        a[i][m + i] = 1.0;
    }
    for (int c = 0; c < m; ++c) {
        int piv = c;
        for (int r = c + 1; r < m; ++r)
            if (std::fabs(a[r][c]) > std::fabs(a[piv][c])) piv = r;
        if (std::fabs(a[piv][c]) < 1e-300) return false;
        if (piv != c)
            for (int j = 0; j < 2 * m; ++j) std::swap(a[c][j], a[piv][j]);
        const double d = a[c][c];
        for (int j = 0; j < 2 * m; ++j) a[c][j] /= d;
        for (int r = 0; r < m; ++r)
            if (r != c) {
                const double f = a[r][c];
                for (int j = 0; j < 2 * m; ++j) a[r][j] -= f * a[c][j];
            }
    }
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) Si[i * m + j] = a[i][m + j];
    return true;
}

}  // namespace

// ---------------------------------------------------------------- SO(3)
Mat3 so3Exp(const Vec3& phi) {
    const double th = std::sqrt(phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]);
    const Mat3 K = skew(phi);
    const Mat3 K2 = mul(K, K);
    double a, b;
    if (th < 1e-8) {
        a = 1.0 - th * th / 6.0;
        b = 0.5 - th * th / 24.0;
    } else {
        a = std::sin(th) / th;
        b = (1.0 - std::cos(th)) / (th * th);
    }
    Mat3 R{1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int i = 0; i < 9; ++i) R[i] += a * K[i] + b * K2[i];
    return R;
}

Vec3 so3Log(const Mat3& R) {
    const double c = std::clamp((R[0] + R[4] + R[8] - 1.0) / 2.0, -1.0, 1.0);
    const double th = std::acos(c);
    const Vec3 w{R[7] - R[5], R[2] - R[6], R[3] - R[1]};
    const double k = th < 1e-6 ? 0.5 + th * th / 12.0 : th / (2.0 * std::sin(th));
    return {w[0] * k, w[1] * k, w[2] * k};
}

Mat3 quatToMat(const Quat& q0) {
    const double n = std::sqrt(q0[0] * q0[0] + q0[1] * q0[1] + q0[2] * q0[2] + q0[3] * q0[3]);
    const double w = q0[0] / n, x = q0[1] / n, y = q0[2] / n, z = q0[3] / n;
    return {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
}

Quat matToQuat(const Mat3& R) {
    Quat q;
    const double tr = R[0] + R[4] + R[8];
    if (tr > 0) {
        const double s = std::sqrt(tr + 1.0) * 2.0;
        q = {0.25 * s, (R[7] - R[5]) / s, (R[2] - R[6]) / s, (R[3] - R[1]) / s};
    } else if (R[0] > R[4] && R[0] > R[8]) {
        const double s = std::sqrt(1.0 + R[0] - R[4] - R[8]) * 2.0;
        q = {(R[7] - R[5]) / s, 0.25 * s, (R[1] + R[3]) / s, (R[2] + R[6]) / s};
    } else if (R[4] > R[8]) {
        const double s = std::sqrt(1.0 + R[4] - R[0] - R[8]) * 2.0;
        q = {(R[2] - R[6]) / s, (R[1] + R[3]) / s, 0.25 * s, (R[5] + R[7]) / s};
    } else {
        const double s = std::sqrt(1.0 + R[8] - R[0] - R[4]) * 2.0;
        q = {(R[3] - R[1]) / s, (R[2] + R[6]) / s, (R[5] + R[7]) / s, 0.25 * s};
    }
    if (q[0] < 0)
        for (double& x : q) x = -x;
    return q;
}

Quat State::q() const { return matToQuat(R); }

// ---------------------------------------------------------------- frames
Vec3 accFromLogger(const Vec3& a, double g) { return {a[0] * g, -a[1] * g, -a[2] * g}; }
Vec3 gyroFromLogger(const Vec3& w) {
    return {deg2rad(w[0]), -deg2rad(w[1]), -deg2rad(w[2])};
}
Mat3 attitudeFromNavEuler(double roll, double pitch, double yaw) {
    const double cy = std::cos(yaw), sy = std::sin(yaw), cp = std::cos(pitch),
                 sp = std::sin(pitch), cr = std::cos(roll), sr = std::sin(roll);
    const Mat3 Rned{cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                    -sp,     cp * sr,                cp * cr};
    const Mat3 T{1, 0, 0, 0, -1, 0, 0, 0, -1};
    return mul(T, mul(Rned, T));
}
Vec3 velocityFromNed(const Vec3& v) { return {v[0], -v[1], -v[2]}; }
Vec3 frdToFlu(const Vec3& v) { return {v[0], -v[1], -v[2]}; }

// ---------------------------------------------------------------- ESKF
ESKF::ESKF(const Params& prm_, const Vec3& p_, const Vec3& v_, const Mat3& R_, const Vec3& ba_,
           const Vec3& bg_)
    : prm(prm_), p(p_), v(v_), ba(ba_), bg(bg_), R(R_) {
    P.fill(0.0);
    const double att = deg2rad(prm.init_att_std_deg);
    const double d[5] = {prm.init_pos_std, prm.init_vel_std, att, prm.init_acc_bias_std,
                         prm.init_gyro_bias_std};
    for (int b = 0; b < 5; ++b)
        for (int i = 0; i < 3; ++i) P[(3 * b + i) * N + 3 * b + i] = d[b] * d[b];
}

void ESKF::predict(const Vec3& acc, const Vec3& gyro, double dt) {
    const Vec3 a{acc[0] - ba[0], acc[1] - ba[1], acc[2] - ba[2]};
    const Vec3 w{gyro[0] - bg[0], gyro[1] - bg[1], gyro[2] - bg[2]};
    last_gyro = gyro;
    const Mat3 R0 = R;
    Vec3 aw = mul(R0, a);
    aw[2] -= prm.gravity;
    for (int i = 0; i < 3; ++i) {
        p[i] += v[i] * dt + 0.5 * aw[i] * dt * dt;
        v[i] += aw[i] * dt;
    }
    const Mat3 dR = so3Exp({w[0] * dt, w[1] * dt, w[2] * dt});
    R = mul(R0, dR);

    static thread_local M15 Phi, T;
    Phi.fill(0.0);
    for (int i = 0; i < N; ++i) Phi[i * N + i] = 1.0;
    const Mat3 Ra = mul(R0, skew(a));
    const Mat3 I3{1, 0, 0, 0, 1, 0, 0, 0, 1};
    setBlock(Phi, 0, 3, I3, dt);
    setBlock(Phi, 0, 6, Ra, -0.5 * dt * dt);
    setBlock(Phi, 0, 9, R0, -0.5 * dt * dt);
    setBlock(Phi, 3, 6, Ra, -dt);
    setBlock(Phi, 3, 9, R0, -dt);
    setBlock(Phi, 6, 6, transpose(dR));
    setBlock(Phi, 6, 12, I3, -dt);
    mul15(Phi, P, T);
    mul15T(T, Phi, P);
    const double q[5] = {0.0, prm.acc_noise * prm.acc_noise * dt,
                         prm.gyro_noise * prm.gyro_noise * dt,
                         prm.acc_bias_rw * prm.acc_bias_rw * dt,
                         prm.gyro_bias_rw * prm.gyro_bias_rw * dt};
    for (int b = 0; b < 5; ++b)
        for (int i = 0; i < 3; ++i) P[(3 * b + i) * N + 3 * b + i] += q[b];
}

bool ESKF::update(int m, const double* r, const double* H, const double* Rm, double* nis_out) {
    // PHt = P H^T (15 x m), S = H P H^T + Rm
    double PHt[N * 3] = {}, S[9] = {}, Si[9] = {};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < m; ++j) {
            double s = 0.0;
            for (int k = 0; k < N; ++k) s += P[i * N + k] * H[j * N + k];
            PHt[i * m + j] = s;
        }
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
            double s = Rm[i * m + j];
            for (int k = 0; k < N; ++k) s += H[i * N + k] * PHt[k * m + j];
            S[i * m + j] = s;
        }
    if (!invSmall(m, S, Si)) return false;
    double nis = 0.0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) nis += r[i] * Si[i * m + j] * r[j];
    if (nis_out) *nis_out = nis;
    if (prm.gate && nis > kChi2[m]) return false;

    double K[N * 3] = {};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < m; ++j) {
            double s = 0.0;
            for (int k = 0; k < m; ++k) s += PHt[i * m + k] * Si[k * m + j];
            K[i * m + j] = s;
        }
    double dx[N] = {};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < m; ++j) dx[i] += K[i * m + j] * r[j];

    // Joseph form: (I-KH) P (I-KH)^T + K Rm K^T
    static thread_local M15 IKH, T;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double s = (i == j) ? 1.0 : 0.0;
            for (int k = 0; k < m; ++k) s -= K[i * m + k] * H[k * N + j];
            IKH[i * N + j] = s;
        }
    mul15(IKH, P, T);
    mul15T(T, IKH, P);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int a = 0; a < m; ++a)
                for (int b = 0; b < m; ++b) s += K[i * m + a] * Rm[a * m + b] * K[j * m + b];
            P[i * N + j] += s;
        }
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            const double s = 0.5 * (P[i * N + j] + P[j * N + i]);
            P[i * N + j] = P[j * N + i] = s;
        }
    for (int i = 0; i < 3; ++i) {
        p[i] += dx[i];
        v[i] += dx[3 + i];
        ba[i] += dx[9 + i];
        bg[i] += dx[12 + i];
    }
    R = orthonormalize(mul(R, so3Exp({dx[6], dx[7], dx[8]})));
    return true;
}

bool ESKF::updateBodyVelocity(const Vec3& z, const Vec3& var, const Vec3* lever, double* nis) {
    const Mat3 Rt = transpose(R);
    const Vec3 vb = mul(Rt, v);
    Vec3 h = vb;
    double H[3 * N] = {};
    const Mat3 Svb = skew(vb);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            H[i * N + 3 + j] = Rt[i * 3 + j];
            H[i * N + 6 + j] = Svb[i * 3 + j];
        }
    if (lever && ((*lever)[0] != 0.0 || (*lever)[1] != 0.0 || (*lever)[2] != 0.0)) {
        const Vec3 wc{last_gyro[0] - bg[0], last_gyro[1] - bg[1], last_gyro[2] - bg[2]};
        const Vec3 c = cross(wc, *lever);
        const Mat3 Sl = skew(*lever);
        for (int i = 0; i < 3; ++i) {
            h[i] += c[i];
            for (int j = 0; j < 3; ++j) H[i * N + 12 + j] = Sl[i * 3 + j];
        }
    }
    const double r[3] = {z[0] - h[0], z[1] - h[1], z[2] - h[2]};
    const double Rm[9] = {var[0], 0, 0, 0, var[1], 0, 0, 0, var[2]};
    return update(3, r, H, Rm, nis);
}

bool ESKF::updateAttitude(const Mat3& R_meas, bool tilt, double std_tilt_deg, bool yaw,
                          double std_yaw_deg, double* nis) {
    int rows[3], m = 0;
    double var[3];
    if (tilt) {
        rows[m] = 0; var[m++] = deg2rad(std_tilt_deg) * deg2rad(std_tilt_deg);
        rows[m] = 1; var[m++] = deg2rad(std_tilt_deg) * deg2rad(std_tilt_deg);
    }
    if (yaw) { rows[m] = 2; var[m++] = deg2rad(std_yaw_deg) * deg2rad(std_yaw_deg); }
    if (m == 0) return false;
    const Vec3 phi = so3Log(mul(R_meas, transpose(R)));
    double r[3], H[3 * N] = {}, Rm[9] = {};
    for (int i = 0; i < m; ++i) {
        r[i] = phi[rows[i]];
        for (int j = 0; j < 3; ++j) H[i * N + 6 + j] = R[rows[i] * 3 + j];
        Rm[i * m + i] = var[i];
    }
    return update(m, r, H, Rm, nis);
}

bool ESKF::updateWorldVelocity(const Vec3& z, const Vec3& var, double* nis) {
    double H[3 * N] = {};
    for (int i = 0; i < 3; ++i) H[i * N + 3 + i] = 1.0;
    const double r[3] = {z[0] - v[0], z[1] - v[1], z[2] - v[2]};
    const double Rm[9] = {var[0], 0, 0, 0, var[1], 0, 0, 0, var[2]};
    return update(3, r, H, Rm, nis);
}

// ---------------------------------------------------------------- stream
StreamEKF::StreamEKF(const Params& prm, const AidParams& aid) : prm_(prm), aid_(aid) {}

void StreamEKF::initialize(double t, const Vec3& p, const Vec3& v, const Mat3& R, const Vec3& ba,
                           const Vec3& bg) {
    f_ = ESKF(prm_, p, v, R, ba, bg);
    t_ = t;
    have_last_ = false;
    pending_.clear();
    ready_ = true;
}

void StreamEKF::initialize(double t, const Vec3& p, const Vec3& v, const Mat3& R, double pos_std,
                           double vel_std, double att_std_deg) {
    initialize(t, p, v, R);
    const double att = deg2rad(att_std_deg);
    for (int i = 0; i < 3; ++i) {
        f_.P[i * N + i] = pos_std * pos_std;
        f_.P[(3 + i) * N + 3 + i] = vel_std * vel_std;
        f_.P[(6 + i) * N + 6 + i] = att * att;
    }
}

void StreamEKF::onImu(double t, const Vec3& acc, const Vec3& gyro) {
    if (!ready_) return;
    if (have_last_ && t <= last_t_) { ++cnt_.out_of_order; return; }
    if (!have_last_) {
        if (t < t_) { ++cnt_.out_of_order; return; }
        have_last_ = true;
        last_t_ = t; last_acc_ = acc; last_gyro_ = gyro;
        if (t > t_) advanceTo(t);
        return;
    }
    const double dt = t - last_t_;
    if (dt > aid_.max_imu_gap_s) {
        ++cnt_.imu_gap;
        advanceTo(t);
    } else {
        f_.predict(last_acc_, last_gyro_, dt);
        t_ = t;
        ++cnt_.imu;
    }
    last_t_ = t; last_acc_ = acc; last_gyro_ = gyro;
    flush();
}

void StreamEKF::advanceTo(double t) {
    const double dt = t - t_;
    if (dt > 0) {
        const double a2 = aid_.gap_acc_std * aid_.gap_acc_std;
        for (int i = 0; i < 3; ++i) {
            f_.P[i * N + i] += a2 * dt * dt * dt * dt / 4.0;
            f_.P[(3 + i) * N + 3 + i] += a2 * dt * dt;
            f_.P[(6 + i) * N + 6 + i] += prm_.gyro_noise * prm_.gyro_noise * dt;
            f_.p[i] += f_.v[i] * dt;
        }
    }
    t_ = t;
}

void StreamEKF::onVo(double t, const Vec3& v, const Vec3& var) {
    measure({t, VO, v, var, {}});
}
void StreamEKF::onAttitude(double t, const Mat3& R) { measure({t, ATT, {}, {}, R}); }
void StreamEKF::onGpsVelocity(double t, const Vec3& v, const Vec3& var) {
    measure({t, GPS, v, var, {}});
}

void StreamEKF::measure(const Meas& m) {
    if (!ready_) return;
    if (m.t > t_) { pending_.push_back(m); return; }
    if (t_ - m.t > aid_.max_meas_age_s) {
        if (m.kind == VO) ++cnt_.vo_dropped;
        return;
    }
    if (m.kind == VO && have_last_ && m.t < last_t_ - 1e-9) ++cnt_.vo_late;
    apply(m);
}

void StreamEKF::flush() {
    if (pending_.empty()) return;
    std::deque<Meas> keep;
    std::vector<Meas> due;
    for (const Meas& m : pending_) (m.t <= t_ ? due.push_back(m) : keep.push_back(m));
    pending_.swap(keep);
    std::stable_sort(due.begin(), due.end(),
                     [](const Meas& a, const Meas& b) { return a.kind < b.kind; });
    for (const Meas& m : due) apply(m);
}

void StreamEKF::apply(const Meas& m) {
    if (m.kind == VO) {
        const bool lev = aid_.lever_arm_m[0] != 0.0 || aid_.lever_arm_m[1] != 0.0 ||
                         aid_.lever_arm_m[2] != 0.0;
        if (!f_.updateBodyVelocity(m.a, m.b, lev ? &aid_.lever_arm_m : nullptr)) ++cnt_.vo_gated;
        ++cnt_.vo;
    } else if (m.kind == ATT) {
        if (++n_att_ % aid_.attitude_every == 0) {
            if (!f_.updateAttitude(m.R, aid_.use_tilt, aid_.std_tilt_deg, aid_.use_yaw,
                                   aid_.std_yaw_deg))
                ++cnt_.att_gated;
            ++cnt_.att;
        }
    } else {
        if (!f_.updateWorldVelocity(m.a, m.b)) ++cnt_.gps_gated;
        ++cnt_.gps;
    }
}

State StreamEKF::state() const {
    State s;
    s.t = t_; s.p = f_.p; s.v = f_.v; s.ba = f_.ba; s.bg = f_.bg; s.R = f_.R; s.P = f_.P;
    return s;
}

// ---------------------------------------------------------------- config
namespace {
// Finds `"key" : value` (value = number | true | false | null | [n, n, n]).
bool findValue(const std::string& js, const std::string& key, std::string& out) {
    const std::string q = "\"" + key + "\"";
    size_t pos = 0;
    while ((pos = js.find(q, pos)) != std::string::npos) {
        size_t i = pos + q.size();
        while (i < js.size() && std::isspace(static_cast<unsigned char>(js[i]))) ++i;
        if (i < js.size() && js[i] == ':') {
            ++i;
            while (i < js.size() && std::isspace(static_cast<unsigned char>(js[i]))) ++i;
            size_t e = i;
            if (js[i] == '[') e = js.find(']', i) + 1;
            else while (e < js.size() && js[e] != ',' && js[e] != '}' && js[e] != '\n') ++e;
            out = js.substr(i, e - i);
            return true;
        }
        pos += q.size();
    }
    return false;
}
void num(const std::string& js, const char* k, double& x) {
    std::string v;
    if (findValue(js, k, v) && v.rfind("null", 0) != 0) x = std::strtod(v.c_str(), nullptr);
}
}  // namespace

bool loadConfig(const std::string& path, Params& p, AidParams& a) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string js = ss.str();
    num(js, "gravity", p.gravity);
    num(js, "acc_noise", p.acc_noise);
    num(js, "gyro_noise", p.gyro_noise);
    num(js, "acc_bias_rw", p.acc_bias_rw);
    num(js, "gyro_bias_rw", p.gyro_bias_rw);
    num(js, "init_pos_std", p.init_pos_std);
    num(js, "init_vel_std", p.init_vel_std);
    num(js, "init_att_std_deg", p.init_att_std_deg);
    num(js, "init_acc_bias_std", p.init_acc_bias_std);
    num(js, "init_gyro_bias_std", p.init_gyro_bias_std);
    std::string v;
    if (findValue(js, "gate", v)) p.gate = v.rfind("true", 0) == 0;
    if (findValue(js, "std_tilt_deg", v)) {
        a.use_tilt = v.rfind("null", 0) != 0;
        if (a.use_tilt) a.std_tilt_deg = std::strtod(v.c_str(), nullptr);
    }
    if (findValue(js, "std_yaw_deg", v)) {
        a.use_yaw = v.rfind("null", 0) != 0;
        if (a.use_yaw) a.std_yaw_deg = std::strtod(v.c_str(), nullptr);
    }
    double every = a.attitude_every;
    num(js, "every", every);
    a.attitude_every = std::max(1, static_cast<int>(every));
    num(js, "max_meas_age_s", a.max_meas_age_s);
    num(js, "max_imu_gap_s", a.max_imu_gap_s);
    num(js, "gap_acc_std", a.gap_acc_std);
    if (findValue(js, "lever_arm_m", v)) {
        std::string s = v;
        std::replace(s.begin(), s.end(), '[', ' ');
        std::replace(s.begin(), s.end(), ']', ' ');
        std::replace(s.begin(), s.end(), ',', ' ');
        std::istringstream is(s);
        is >> a.lever_arm_m[0] >> a.lever_arm_m[1] >> a.lever_arm_m[2];
    }
    return true;
}

}  // namespace imuvo
