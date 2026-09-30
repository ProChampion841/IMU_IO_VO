"""Stream (real-time) front end for the ESKF: feed sensor messages as they arrive.

This is the logic that runs on the aircraft, and the reference the C++ port
(EKF/cpp) is tested against.  Every message carries its own timestamp; messages
are handled in ARRIVAL order and nothing ever looks ahead.

    s = StreamEKF(params, attitude_every=10)
    s.initialize(t, p, v, R)                 # e.g. from GPS/nav at start-up
    s.on_imu(t, acc, gyro)                   # 100 Hz, body FLU, m/s^2 and rad/s
    s.on_vo(t, v_body, var)                  # one per image pair, body FLU (m/s, (m/s)^2)
    s.on_attitude(t, R)                      # nav attitude, body -> world NWU
    s.on_gps_velocity(t, v_world, var)       # while GPS is up (learns the biases)
    s.state()                                # p, v, R, ba, bg, P, t

TIMING RULES -- identical to the offline evaluator (ekf/pipeline.py), so on the
same data both give the same numbers:
  * IMU: sample k is integrated over [t_k, t_(k+1)] when sample k+1 arrives
    (its dt is only known then).  The state then sits at t_(k+1).
  * A measurement stamped t is applied as soon as the state time reaches t
    (it waits in a queue for the next IMU sample if it is newer than the state),
    so it lands on the first state at or after its timestamp.  A measurement
    OLDER than the state (arrived late) is applied at once, if it is no older
    than `max_meas_age_s`, else dropped.
  * Attitude messages: every `attitude_every`-th one is applied.
  * An IMU gap longer than `max_imu_gap_s` is not integrated (the state holds
    and its covariance is inflated by the gap), because integrating one sample
    over a long hole is a guess, not a measurement.
"""
from collections import deque

import numpy as np

from .eskf import ESKF, ESKFParams


class StreamEKF:
    def __init__(self, params=None, attitude_every=10, std_tilt_deg=0.2, std_yaw_deg=0.5,
                 lever=None, max_meas_age_s=1.0, max_imu_gap_s=0.1, gap_acc_std=2.0):
        self.prm = params or ESKFParams()
        self.attitude_every = int(attitude_every)
        self.std_tilt_deg, self.std_yaw_deg = std_tilt_deg, std_yaw_deg
        self.lever = None if lever is None else np.asarray(lever, float)
        self.max_meas_age_s = float(max_meas_age_s)
        self.max_imu_gap_s = float(max_imu_gap_s)
        self.gap_acc_std = float(gap_acc_std)
        self.f = None
        self.t = None                 # time of the filter state
        self.last_imu = None          # (t, acc, gyro) waiting for its dt
        self.pending = deque()        # (t, kind, payload) newer than the state
        self.n_att = 0
        self.counters = {"imu": 0, "imu_gap": 0, "vo": 0, "vo_late": 0, "vo_dropped": 0,
                         "att": 0, "gps": 0, "out_of_order": 0}

    # ------------------------------------------------------------------ setup
    def initialize(self, t, p, v, R, ba=None, bg=None, P0=None, init_std=None):
        """init_std = (pos_std m, vel_std m/s, att_std deg) overrides the config's
        initial uncertainty for those three blocks (e.g. a no-GT start)."""
        self.f = ESKF(p, v, R, ba, bg, params=self.prm, P0=P0)
        if init_std is not None and P0 is None:
            ps, vs, a = (float(x) for x in init_std)
            d = np.diag(self.f.P).copy()
            d[0:3], d[3:6], d[6:9] = ps ** 2, vs ** 2, np.radians(a) ** 2
            self.f.P = np.diag(d)
        self.t = float(t)
        self.last_imu = None
        self.pending.clear()

    @property
    def ready(self):
        return self.f is not None

    # ------------------------------------------------------------------ IMU
    def on_imu(self, t, acc, gyro):
        t = float(t)
        if not self.ready:
            return
        if self.last_imu is not None and t <= self.last_imu[0]:
            self.counters["out_of_order"] += 1
            return
        if self.last_imu is None:
            if t < self.t:
                self.counters["out_of_order"] += 1
                return
            self.last_imu = (t, np.asarray(acc, float), np.asarray(gyro, float))
            if t > self.t:                               # first sample after init
                self._advance_to(t)
            return
        t0, a0, g0 = self.last_imu
        dt = t - t0
        if dt > self.max_imu_gap_s:
            self.counters["imu_gap"] += 1
            self._advance_to(t)                          # hold, inflate P
        else:
            self.f.predict(a0, g0, dt)
            self.t = t
            self.counters["imu"] += 1
        self.last_imu = (t, np.asarray(acc, float), np.asarray(gyro, float))
        self._flush()

    def _advance_to(self, t):
        """No IMU over [state time, t]: coast at constant velocity and grow the
        uncertainty by an unknown acceleration of `gap_acc_std` (m/s^2)."""
        dt = t - self.t
        if dt > 0:
            s, P = self.prm, self.f.P
            a2 = self.gap_acc_std ** 2
            P[0:3, 0:3] += np.eye(3) * (a2 * dt ** 4 / 4.0)
            P[3:6, 3:6] += np.eye(3) * (a2 * dt ** 2)
            P[6:9, 6:9] += np.eye(3) * (s.gyro_noise ** 2 * dt)
            self.f.p = self.f.p + self.f.v * dt
        self.t = t

    # ------------------------------------------------------------------ measurements
    def on_vo(self, t, v_body_flu, var):
        self._measure(float(t), "vo", (np.asarray(v_body_flu, float), np.asarray(var, float)))

    def on_attitude(self, t, R):
        self._measure(float(t), "att", np.asarray(R, float))

    def on_gps_velocity(self, t, v_world, var):
        self._measure(float(t), "gps", (np.asarray(v_world, float), np.asarray(var, float)))

    def _measure(self, t, kind, payload):
        if not self.ready:
            return
        if t > self.t:
            self.pending.append((t, kind, payload))
            return
        if self.t - t > self.max_meas_age_s:
            if kind == "vo":
                self.counters["vo_dropped"] += 1
            return
        if kind == "vo" and self.last_imu is not None and t < self.last_imu[0] - 1e-9:
            self.counters["vo_late"] += 1
        self._apply(kind, payload)

    def _flush(self):
        if not self.pending:
            return
        keep = deque()
        # stable in arrival order; VO before attitude at the same state (as offline)
        due = [m for m in self.pending if m[0] <= self.t]
        for m in self.pending:
            if m[0] > self.t:
                keep.append(m)
        self.pending = keep
        for _, kind, payload in sorted(due, key=lambda m: {"gps": 0, "vo": 1, "att": 2}[m[1]]):
            self._apply(kind, payload)

    def _apply(self, kind, payload):
        f = self.f
        if kind == "vo":
            f.update_body_velocity(payload[0], payload[1], self.lever)
            self.counters["vo"] += 1
        elif kind == "att":
            self.n_att += 1
            if self.n_att % self.attitude_every == 0:
                f.update_attitude(payload, self.std_tilt_deg, self.std_yaw_deg)
                self.counters["att"] += 1
        elif kind == "gps":
            v, var = payload
            H = np.zeros((3, 15))
            H[:, 3:6] = np.eye(3)
            f._update(v - f.v, H, np.diag(var), "gps")
            self.counters["gps"] += 1

    # ------------------------------------------------------------------ output
    def state(self):
        s = self.f.state()
        s["t"] = self.t
        return s
