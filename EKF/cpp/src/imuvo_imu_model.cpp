// Implementation of imuvo_imu_model.hpp.  Mirrors EKF/ekf/imu_model.py.
#include "imuvo_imu_model.hpp"

#include <onnxruntime_cxx_api.h>

#include <stdexcept>

namespace imuvo {

struct StreamImuCorrector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "imuvo_imu"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> in_names, out_names;
    bool has_g = false;
    int i_acc = -1, i_gyro = -1;
};

StreamImuCorrector::StreamImuCorrector(const std::string& path, int every, int delay, int threads)
    : impl_(new Impl), every_(std::max(1, every)), delay_(std::max(0, delay)) {
    impl_->opts.SetIntraOpNumThreads(threads);
    impl_->session.reset(new Ort::Session(impl_->env, path.c_str(), impl_->opts));
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::Session& s = *impl_->session;
    for (size_t i = 0; i < s.GetInputCount(); ++i) {
        impl_->in_names.push_back(s.GetInputNameAllocated(i, alloc).get());
        if (impl_->in_names.back() == "g_body") impl_->has_g = true;
        if (impl_->in_names.back() == "airspeed")
            throw std::runtime_error("IMU model needs airspeed: not supported in the stream");
    }
    for (size_t i = 0; i < s.GetOutputCount(); ++i) {
        const std::string n = s.GetOutputNameAllocated(i, alloc).get();
        if (n == "corrected_acc") impl_->i_acc = static_cast<int>(impl_->out_names.size());
        if (n == "corrected_gyro") impl_->i_gyro = static_cast<int>(impl_->out_names.size());
        impl_->out_names.push_back(n);
    }
    if (impl_->i_acc < 0 || impl_->i_gyro < 0)
        throw std::runtime_error("model has no corrected_acc / corrected_gyro output");
    const auto shape = s.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[1] <= 9)
        throw std::runtime_error("unexpected IMU model input shape (fixed time axis needed)");
    n_in_ = static_cast<int>(shape[1]);
}

StreamImuCorrector::~StreamImuCorrector() = default;

void StreamImuCorrector::reset() {
    buf_.clear();
    corr_.clear();
    count_ = released_ = runs_ = 0;
}

std::vector<ImuSample> StreamImuCorrector::push(double t, const Vec3& acc, const Vec3& gyro,
                                                const Mat3& R) {
    buf_.push_back({t, acc, gyro, {R[6], R[7], R[8]}, active_});     // g_body = R^T e_z
    if (static_cast<int>(buf_.size()) > n_in_) buf_.pop_front();
    ++count_;
    if (count_ >= n_in_ && (count_ - n_in_) % every_ == 0) run();
    return release(false);
}

std::vector<ImuSample> StreamImuCorrector::flush() { return release(true); }

void StreamImuCorrector::run() {
    const int n = n_in_;
    std::vector<float> acc(n * 3), gyro(n * 3), g(n * 3);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < 3; ++j) {
            acc[i * 3 + j] = static_cast<float>(buf_[i].acc[j] - b_acc_[j]);
            gyro[i * 3 + j] = static_cast<float>(buf_[i].gyro[j] - b_gyro_[j]);
            g[i * 3 + j] = static_cast<float>(buf_[i].g[j]);
        }
    const std::array<int64_t, 3> shape{1, n, 3};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    std::vector<const char*> in_names;
    for (const std::string& name : impl_->in_names) {
        std::vector<float>& src = name == "acc" ? acc : name == "gyro" ? gyro : g;
        inputs.push_back(Ort::Value::CreateTensor<float>(mem, src.data(), src.size(),
                                                         shape.data(), shape.size()));
        in_names.push_back(name.c_str());
    }
    std::vector<const char*> out_names;
    for (const std::string& s : impl_->out_names) out_names.push_back(s.c_str());
    auto out = impl_->session->Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(),
                                   inputs.size(), out_names.data(), out_names.size());
    ++runs_;
    const float* ca = out[impl_->i_acc].GetTensorData<float>();
    const float* cg = out[impl_->i_gyro].GetTensorData<float>();
    const long first = count_ - (n_in_ - 9);
    const long last_ok = count_ - 1 - delay_;
    for (int j = 0; j < n_in_ - 9; ++j) {
        const long i = first + j;
        if (i > last_ok) break;
        if (i >= released_)
            corr_[i] = {{ca[j * 3], ca[j * 3 + 1], ca[j * 3 + 2]},
                        {cg[j * 3], cg[j * 3 + 1], cg[j * 3 + 2]}};
    }
}

std::vector<ImuSample> StreamImuCorrector::release(bool final) {
    std::vector<ImuSample> out;
    const long newest = count_ - 1;
    while (released_ <= newest) {
        const long i = released_;
        const long pos = static_cast<long>(buf_.size()) - (count_ - i);
        if (pos < 0) { ++released_; continue; }
        const Slot& s = buf_[pos];
        auto it = corr_.find(i);
        if (!s.active) {
            out.push_back({s.t, s.acc, s.gyro});
        } else if (it != corr_.end()) {
            out.push_back({s.t, it->second.first, it->second.second});
            corr_.erase(it);
        } else if (count_ < n_in_) {
            out.push_back({s.t, s.acc, s.gyro});
        } else if (final) {
            out.push_back({s.t,
                           {s.acc[0] - b_acc_[0], s.acc[1] - b_acc_[1], s.acc[2] - b_acc_[2]},
                           {s.gyro[0] - b_gyro_[0], s.gyro[1] - b_gyro_[1],
                            s.gyro[2] - b_gyro_[2]}});
        } else {
            break;
        }
        ++released_;
    }
    return out;
}

}  // namespace imuvo
