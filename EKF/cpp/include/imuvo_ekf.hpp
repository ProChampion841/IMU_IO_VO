// IMU + VO error-state EKF, stream mode.  C++17, no dependencies.
//
// A line-for-line port of EKF/ekf/eskf.py + EKF/ekf/stream.py; tests/test_cpp.py
// checks the two agree on the same message log.  See EKF/README.md for the maths.
//
// CONVENTIONS (the IMU project's)
//   world NWU (x north, y west, z UP), body FLU (x forward, y left, z up)
//   R = body -> world rotation, gravity g = [0, 0, 9.81007]
//   a level, stationary accelerometer reads +g on body z
// The logger is FRD / NED; use the converters at the bottom of this file.
//
// USE
//   imuvo::StreamEKF ekf(params, aid);
//   ekf.initialize(t, p, v, R);                  // e.g. from GPS/nav
//   ekf.onImu(t, acc, gyro);                     // every IMU sample (100 Hz)
//   ekf.onVo(t, v_body_flu, var);                // one per delivered image pair
//   ekf.onAttitude(t, R_nav);                    // nav attitude (every tick is fine)
//   ekf.onGpsVelocity(t, v_world, var);          // while GPS is up
//   ekf.onPosition(t_image, p_world, var);       // land-matching fix, NWU m (late is fine)
//   imuvo::State s = ekf.state();
// Every call is O(15^3) at most, except a LATE position fix, which rewinds to its
// image time and re-runs the history since (pos_replay_s of it at most).  Memory:
// the small pending-measurement queue plus that history (~2 KB per IMU sample, so
// ~0.6 MB for 3 s at 100 Hz).  Not thread-safe: call from one thread.
#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <string>
#include <vector>

namespace imuvo {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;        // row-major
using Quat = std::array<double, 4>;        // w, x, y, z
constexpr int N = 15;                      // error state size
using Cov = std::array<double, N * N>;     // row-major

struct Params {
    double gravity = 9.81007;
    double acc_noise = 0.05;               // m/s^2/sqrt(Hz)
    double gyro_noise = 0.005;             // rad/s/sqrt(Hz)
    double acc_bias_rw = 1e-4;             // m/s^3/sqrt(Hz)
    double gyro_bias_rw = 1e-5;            // rad/s^2/sqrt(Hz)
    double init_pos_std = 0.1;             // m
    double init_vel_std = 0.1;             // m/s
    double init_att_std_deg = 0.5;
    double init_acc_bias_std = 0.03;       // m/s^2
    double init_gyro_bias_std = 2e-4;      // rad/s
    bool gate = true;                      // chi-square 99.9 % on every update
};

struct AidParams {
    int attitude_every = 10;               // apply every N-th attitude message
    bool use_tilt = true;
    double std_tilt_deg = 0.2;
    bool use_yaw = true;
    double std_yaw_deg = 0.5;
    Vec3 lever_arm_m{0.0, 0.0, 0.0};       // IMU -> camera, body FLU
    double max_meas_age_s = 1.0;           // older measurements are dropped
    double max_imu_gap_s = 0.1;            // longer IMU gaps are not integrated
    double gap_acc_std = 2.0;              // m/s^2, uncertainty growth over a gap
    // absolute position fixes (land matching), see EKF/ekf/stream.py
    bool pos_use_vertical = false;         // false: north/west only
    Vec3 pos_lever_arm_m{0.0, 0.0, 0.0};   // IMU -> the point the fix refers to, body FLU
    double pos_replay_s = 3.0;             // history for late fixes (0 = apply late at once)
    int pos_reset_after = 3;               // rejected fixes in a row before a reset (0 = never)
    double pos_reset_agree_sigma = 3.0;    // ... that must agree within this many sigma
};

// Read the numeric fields of EKF/configs/ekf_default.json (same key names).
// Missing keys keep their defaults.  Returns false if the file cannot be read.
bool loadConfig(const std::string& json_path, Params& p, AidParams& a);

struct State {
    double t = 0.0;
    Vec3 p{}, v{}, ba{}, bg{};
    Mat3 R{1, 0, 0, 0, 1, 0, 0, 0, 1};
    Cov P{};
    Quat q() const;
};

struct Counters {
    long imu = 0, imu_gap = 0, vo = 0, vo_late = 0, vo_dropped = 0, att = 0, gps = 0,
         out_of_order = 0, vo_gated = 0, att_gated = 0, gps_gated = 0, pos = 0, pos_gated = 0,
         pos_late = 0, pos_dropped = 0, pos_reset = 0;
};

// ---------------------------------------------------------------- filter core
class ESKF {
public:
    ESKF() = default;
    ESKF(const Params& prm, const Vec3& p, const Vec3& v, const Mat3& R,
         const Vec3& ba = {0, 0, 0}, const Vec3& bg = {0, 0, 0});
    void predict(const Vec3& acc, const Vec3& gyro, double dt);
    // return true if accepted; nis is always written
    bool updateBodyVelocity(const Vec3& z, const Vec3& var, const Vec3* lever, double* nis = nullptr);
    bool updateAttitude(const Mat3& R_meas, bool tilt, double std_tilt_deg, bool yaw,
                        double std_yaw_deg, double* nis = nullptr);
    bool updateWorldVelocity(const Vec3& z, const Vec3& var, double* nis = nullptr);
    // Absolute position fix, world NWU (m): rows north/west (+ up if vertical);
    // lever = IMU -> the point the fix refers to (body FLU) or nullptr.
    bool updatePosition(const Vec3& z, const Vec3& var, bool vertical, const Vec3* lever,
                        double* nis = nullptr);
    // m = 2 or 3 rows of residual r and Jacobian H (row-major m x 15)
    void positionResidual(const Vec3& z, bool vertical, const Vec3* lever, double* r, double* H,
                          int& m) const;
    // Jump to the fix: position from z, its variance var, no correlation with the rest.
    void resetPosition(const Vec3& z, const Vec3& var, bool vertical, const Vec3* lever);

    Params prm;
    Vec3 p{}, v{}, ba{}, bg{}, last_gyro{};
    Mat3 R{1, 0, 0, 0, 1, 0, 0, 0, 1};
    Cov P{};

private:
    bool update(int m, const double* r, const double* H, const double* Rm, double* nis);
};

// ---------------------------------------------------------------- stream front end
class StreamEKF {
public:
    explicit StreamEKF(const Params& prm = Params(), const AidParams& aid = AidParams());
    void initialize(double t, const Vec3& p, const Vec3& v, const Mat3& R,
                    const Vec3& ba = {0, 0, 0}, const Vec3& bg = {0, 0, 0});
    // Same, with the initial 1-sigma of position [m], velocity [m/s] and attitude
    // [deg] given here instead of taken from Params -- e.g. a start with NO ground
    // truth: p = 0, v = R_nav * first VO velocity (std from its variance), R = nav.
    void initialize(double t, const Vec3& p, const Vec3& v, const Mat3& R, double pos_std,
                    double vel_std, double att_std_deg);
    bool ready() const { return ready_; }

    void onImu(double t, const Vec3& acc, const Vec3& gyro);
    void onVo(double t, const Vec3& v_body_flu, const Vec3& var);
    void onAttitude(double t, const Mat3& R_nav);
    void onGpsVelocity(double t, const Vec3& v_world, const Vec3& var);
    // Absolute position fix (land matching), world NWU (m) + per-axis variance (m^2),
    // stamped with the time its IMAGE was taken.  It may arrive late: the filter
    // rewinds to that time (pos_replay_s), applies it, and re-runs what came after.
    void onPosition(double t, const Vec3& p_world, const Vec3& var);

    State state() const;
    const Counters& counters() const { return cnt_; }
    const ESKF& filter() const { return f_; }

private:
    // at the same state: position fix first, then GPS velocity, VO, attitude
    enum Kind { POS = 0, GPS = 1, VO = 2, ATT = 3 };
    struct Meas { double t; Kind kind; Vec3 a; Vec3 b; Mat3 R; };
    // one filter operation, logged so a late position fix can replay them
    enum OpType { OP_PREDICT, OP_ADVANCE, OP_VO, OP_ATT, OP_GPS, OP_POS };
    struct Op { OpType type; double t; double dt; Vec3 a; Vec3 b; Mat3 R; bool reset; };
    struct Checkpoint { double t; ESKF f; long op_index; };
    void measure(const Meas& m);
    void apply(const Meas& m);
    void flush();
    void advanceTo(double t);
    bool exec(const Op& op);
    void checkpoint();
    void restartHistory();
    void applyPosition(const Vec3& z, const Vec3& var);
    void latePosition(const Meas& m);
    const Vec3* posLever() const;

    Params prm_;
    AidParams aid_;
    ESKF f_;
    bool ready_ = false;
    double t_ = 0.0;
    bool have_last_ = false;
    double last_t_ = 0.0;
    Vec3 last_acc_{}, last_gyro_{};
    std::deque<Meas> pending_;
    long n_att_ = 0;
    Counters cnt_;
    std::vector<Vec3> rejects_;            // residuals of the fixes rejected in a row
    std::deque<Checkpoint> cps_;           // state after every IMU step (pos_replay_s)
    std::deque<Op> ops_;                   // ops since the oldest checkpoint
    long op_base_ = 0;                     // absolute index of ops_.front()
    double floor_ = -1e300;                // time of the newest checkpoint trimmed
};

// ---------------------------------------------------------------- SO(3) helpers
Mat3 so3Exp(const Vec3& phi);
Vec3 so3Log(const Mat3& R);
Mat3 quatToMat(const Quat& q);                 // w, x, y, z -> body->world
Quat matToQuat(const Mat3& R);

// ---------------------------------------------------------------- logger -> EKF frames
// Raw logger: AcclX/Y/Z in g (FRD), GyroX/Y/Z in deg/s (FRD),
// GPSNavEulX/Y/Z = roll, pitch, yaw in rad (body FRD -> world NED, intrinsic ZYX),
// GPSNavVnX/Y/Z = NED m/s.  VO body velocity is FRD.
Vec3 accFromLogger(const Vec3& accl_g_frd, double gravity = 9.81007);
Vec3 gyroFromLogger(const Vec3& gyro_dps_frd);
Mat3 attitudeFromNavEuler(double roll, double pitch, double yaw);   // -> R_nwu_flu
Vec3 velocityFromNed(const Vec3& v_ned);                             // -> NWU
Vec3 frdToFlu(const Vec3& v);                                        // VO -> EKF
// Land-matching fix in WGS-84 lat/lon (deg) + height (m) -> world NWU metres about
// the origin (lat0, lon0, alt0) = where the filter's position 0 is.  Exact (ECEF).
Vec3 geodeticToNwu(double lat_deg, double lon_deg, double alt_m, double lat0_deg,
                   double lon0_deg, double alt0_m);

}  // namespace imuvo
