// Replay a message log through imuvo::StreamEKF -- the integration template for
// the Jetson main loop, and the harness tests/test_cpp.py compares with Python.
//
//   ekf_replay events.csv states.csv [ekf_default.json]
//
// events.csv, one message per line, in ARRIVAL order (units SI, frames NWU/FLU):
//   INIT,t,px,py,pz,vx,vy,vz,qw,qx,qy,qz
//   IMU,t,ax,ay,az,gx,gy,gz
//   VO,t,vx,vy,vz,varx,vary,varz          body FLU (use imuvo::frdToFlu on VO output)
//   ATT,t,qw,qx,qy,qz                     nav attitude, body -> world
//   GPS,t,vx,vy,vz,varx,vary,varz         world velocity while GPS is up
// Lines starting with '#' are ignored.
// states.csv: the state after every IMU message.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "imuvo_ekf.hpp"

using namespace imuvo;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: ekf_replay events.csv states.csv [config.json]\n";
        return 2;
    }
    Params prm;
    AidParams aid;
    if (argc > 3 && !loadConfig(argv[3], prm, aid)) {
        std::cerr << "cannot read config " << argv[3] << "\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) { std::cerr << "cannot read " << argv[1] << "\n"; return 2; }
    FILE* out = std::fopen(argv[2], "w");
    if (!out) { std::cerr << "cannot write " << argv[2] << "\n"; return 2; }
    std::fprintf(out, "t,px,py,pz,vx,vy,vz,qw,qx,qy,qz,bax,bay,baz,bgx,bgy,bgz,std_p,std_v\n");

    StreamEKF ekf(prm, aid);
    std::string line;
    std::vector<double> x;
    long n_msg = 0;
    double busy_s = 0.0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string type, tok;
        std::getline(ss, type, ',');
        x.clear();
        while (std::getline(ss, tok, ',')) x.push_back(std::strtod(tok.c_str(), nullptr));
        const auto t0 = std::chrono::steady_clock::now();
        if (type == "INIT" && x.size() >= 11) {
            ekf.initialize(x[0], {x[1], x[2], x[3]}, {x[4], x[5], x[6]},
                           quatToMat({x[7], x[8], x[9], x[10]}));
        } else if (type == "IMU" && x.size() >= 7) {
            ekf.onImu(x[0], {x[1], x[2], x[3]}, {x[4], x[5], x[6]});
        } else if (type == "VO" && x.size() >= 7) {
            ekf.onVo(x[0], {x[1], x[2], x[3]}, {x[4], x[5], x[6]});
        } else if (type == "ATT" && x.size() >= 5) {
            ekf.onAttitude(x[0], quatToMat({x[1], x[2], x[3], x[4]}));
        } else if (type == "GPS" && x.size() >= 7) {
            ekf.onGpsVelocity(x[0], {x[1], x[2], x[3]}, {x[4], x[5], x[6]});
        } else {
            std::cerr << "skipping line: " << line << "\n";
            continue;
        }
        busy_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++n_msg;
        if (type == "IMU" && ekf.ready()) {
            const State s = ekf.state();
            const Quat q = s.q();
            const double sp = std::sqrt(s.P[0] + s.P[16] + s.P[32]);
            const double sv = std::sqrt(s.P[48] + s.P[64] + s.P[80]);
            std::fprintf(out,
                         "%.6f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.12f,%.12f,%.12f,%.12f,"
                         "%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                         s.t, s.p[0], s.p[1], s.p[2], s.v[0], s.v[1], s.v[2], q[0], q[1], q[2],
                         q[3], s.ba[0], s.ba[1], s.ba[2], s.bg[0], s.bg[1], s.bg[2], sp, sv);
        }
    }
    std::fclose(out);
    const Counters& c = ekf.counters();
    std::printf("messages %ld | imu %ld (gaps %ld) | vo %ld (gated %ld, late %ld, dropped %ld) | "
                "att %ld | gps %ld | out-of-order %ld | %.2f us per message\n",
                n_msg, c.imu, c.imu_gap, c.vo, c.vo_gated, c.vo_late, c.vo_dropped, c.att, c.gps,
                c.out_of_order, 1e6 * busy_s / std::max(1L, n_msg));
    return 0;
}
