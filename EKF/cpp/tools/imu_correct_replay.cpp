// Run the C++ StreamImuCorrector over raw IMU samples -- the parity harness for
// tests/test_imu_model.py and a template for the Jetson IMU thread.
//
//   imu_correct_replay model.onnx raw.csv out.csv [every] [delay]
// raw.csv : t,ax,ay,az,gx,gy,gz,qw,qx,qy,qz,active[,bax,bay,baz,bgx,bgy,bgz]
//           (SI, body FLU; q = nav attitude body->world; optional freeze from that row on)
// out.csv : arrival_t,t,ax,ay,az,gx,gy,gz   one line per released sample
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "imuvo_imu_model.hpp"

using namespace imuvo;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: imu_correct_replay model.onnx raw.csv out.csv [every] [delay]\n";
        return 2;
    }
    const int every = argc > 4 ? std::atoi(argv[4]) : 10;
    const int delay = argc > 5 ? std::atoi(argv[5]) : 16;
    StreamImuCorrector c(argv[1], every, delay);
    std::ifstream in(argv[2]);
    FILE* out = std::fopen(argv[3], "w");
    if (!in || !out) { std::cerr << "cannot open files\n"; return 2; }
    std::fprintf(out, "arrival_t,t,ax,ay,az,gx,gy,gz\n");
    std::string line;
    std::vector<double> x;
    double last_t = 0, busy = 0;
    long n = 0;
    auto emit = [&](double arrival, const std::vector<ImuSample>& v) {
        for (const ImuSample& s : v)
            std::fprintf(out, "%.6f,%.6f,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", arrival, s.t,
                         s.acc[0], s.acc[1], s.acc[2], s.gyro[0], s.gyro[1], s.gyro[2]);
    };
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line[0] == 't') continue;
        std::stringstream ss(line);
        std::string tok;
        x.clear();
        while (std::getline(ss, tok, ',')) x.push_back(std::strtod(tok.c_str(), nullptr));
        if (x.size() < 12) continue;
        c.setActive(x[11] > 0.5);
        if (x.size() >= 18) c.setFreeze({x[12], x[13], x[14]}, {x[15], x[16], x[17]});
        const auto t0 = std::chrono::steady_clock::now();
        auto v = c.push(x[0], {x[1], x[2], x[3]}, {x[4], x[5], x[6]},
                        quatToMat({x[7], x[8], x[9], x[10]}));
        busy += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        emit(x[0], v);
        last_t = x[0];
        ++n;
    }
    emit(last_t, c.flush());
    std::fclose(out);
    std::printf("samples %ld | model runs %ld | %.1f us per sample (%.2f ms per run)\n", n,
                c.runs(), 1e6 * busy / std::max(1L, n), 1e3 * busy / std::max(1L, c.runs()));
    return 0;
}
