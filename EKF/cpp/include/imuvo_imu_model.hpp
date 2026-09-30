// Learned IMU correction (IMU/tools/export_onnx.py model) run causally for the stream.
// C++ port of EKF/ekf/imu_model.py (StreamImuCorrector); needs ONNX Runtime.
//
//   imuvo::StreamImuCorrector corr("imu_40s.onnx", /*every=*/10, /*delay=*/16);
//   for each raw IMU sample (SI, body FLU) with the nav attitude at that sample:
//       for (const auto& s : corr.push(t, acc, gyro, R_nav))
//           ekf.onImu(s.t, s.acc, s.gyro);            // released samples, in order
//
// Every `every` samples the model runs on the LATEST N + 9 samples (only the past).
// A sample is released once a run has seen `delay` samples after it (the CNN looks
// ~12 samples ahead), so the EKF is fed the corrected IMU delay..delay+every-1
// samples late; its VO / attitude messages wait in its queue until the IMU catches
// up.  Before N + 9 samples exist the raw sample is released (warm-up).
// setFreeze(): the 15 s pre-outage bias the model was trained with (subtracted
// before the model).  setActive(false): raw passthrough (e.g. while GPS is up).
#pragma once

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "imuvo_ekf.hpp"

namespace imuvo {

struct ImuSample {
    double t;
    Vec3 acc, gyro;
};

class StreamImuCorrector {
public:
    StreamImuCorrector(const std::string& onnx_path, int every = 10, int delay = 16,
                       int threads = 1);
    ~StreamImuCorrector();
    std::vector<ImuSample> push(double t, const Vec3& acc, const Vec3& gyro, const Mat3& R_nav);
    std::vector<ImuSample> flush();
    void setFreeze(const Vec3& b_acc, const Vec3& b_gyro) { b_acc_ = b_acc; b_gyro_ = b_gyro; }
    void setActive(bool a) { active_ = a; }
    void reset();
    int frames() const { return n_in_ - 9; }
    long runs() const { return runs_; }

private:
    struct Slot { double t; Vec3 acc, gyro, g; bool active; };
    void run();
    std::vector<ImuSample> release(bool final);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    int n_in_ = 0, every_, delay_;
    std::deque<Slot> buf_;
    long count_ = 0, released_ = 0, runs_ = 0;
    std::map<long, std::pair<Vec3, Vec3>> corr_;
    Vec3 b_acc_{0, 0, 0}, b_gyro_{0, 0, 0};
    bool active_ = true;
};

}  // namespace imuvo
