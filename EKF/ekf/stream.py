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
    s.on_position(t, p_world, var)           # absolute position fix (land matching), NWU m
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

POSITION FIXES (land matching) are the exception to "apply a late measurement at
once".  A fix that arrives `d` seconds late, applied to the current state, is
wrong by speed x d (25 m/s x 0.5 s = 12.5 m) -- velocity can be applied late,
position cannot.  With `replay_s` > 0 the filter keeps the last `replay_s` seconds
of its history (a checkpoint after every IMU step plus every update since), and a
late fix REWINDS to the first state at or after its timestamp, applies the fix
there, and re-runs everything that happened since.  The result is exactly what the
filter would have computed had the fix arrived on time.  At the same state a fix
is applied before every other measurement.  A fix older than `replay_s` is dropped.
With `replay_s` = 0 (no history) a late fix is applied at once, like the others.

Gate lock-out: after a long outage the filter can be confidently wrong, and then
the chi-square gate rejects every good fix for ever.  When `pos_reset_after`
fixes in a row are rejected AND agree with each other (residuals within
`pos_reset_agree_sigma` of their mean, per axis), the position is reset to the
fix (ESKF.reset_position).  A burst of wrong matches does not agree, so it does
not reset.
"""
from bisect import bisect_left
from collections import deque

import numpy as np

from .eskf import ESKF, ESKFParams

# at the same state: position fix first, then GPS velocity, VO, attitude
_ORDER = {"pos": -1, "gps": 0, "vo": 1, "att": 2}


class StreamEKF:
    def __init__(self, params=None, attitude_every=10, std_tilt_deg=0.2, std_yaw_deg=0.5,
                 lever=None, max_meas_age_s=1.0, max_imu_gap_s=0.1, gap_acc_std=2.0,
                 pos_rows=(0, 1), pos_lever=None, replay_s=0.0, pos_reset_after=3,
                 pos_reset_agree_sigma=3.0):
        self.prm = params or ESKFParams()
        self.attitude_every = int(attitude_every)
        self.std_tilt_deg, self.std_yaw_deg = std_tilt_deg, std_yaw_deg
        self.lever = None if lever is None else np.asarray(lever, float)
        self.max_meas_age_s = float(max_meas_age_s)
        self.max_imu_gap_s = float(max_imu_gap_s)
        self.gap_acc_std = float(gap_acc_std)
        self.pos_rows = tuple(int(r) for r in pos_rows)
        self.pos_lever = (None if pos_lever is None or not np.any(pos_lever)
                          else np.asarray(pos_lever, float))
        self.replay_s = float(replay_s)
        self.pos_reset_after = int(pos_reset_after)
        self.pos_reset_agree_sigma = float(pos_reset_agree_sigma)
        self.f = None
        self.t = None                 # time of the filter state
        self.last_imu = None          # (t, acc, gyro) waiting for its dt
        self.pending = deque()        # (t, kind, payload) newer than the state
        self.n_att = 0
        self.counters = {"imu": 0, "imu_gap": 0, "vo": 0, "vo_late": 0, "vo_dropped": 0,
                         "att": 0, "gps": 0, "out_of_order": 0,
                         "pos": 0, "pos_gated": 0, "pos_late": 0, "pos_dropped": 0,
                         "pos_reset": 0}
        self._pos_rejects = []        # residuals of the fixes rejected in a row
        # replay history: checkpoints (t, snapshot, index of the next op) and the
        # ops executed since the oldest one; _op_base = absolute index of _ops[0]
        self._cps = deque()
        self._ops = deque()
        self._op_base = 0
        self._floor = -np.inf         # time of the newest checkpoint already trimmed

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
        self._pos_rejects = []
        self._cps.clear()
        self._ops.clear()
        self._op_base = 0
        self._floor = -np.inf
        self._checkpoint()

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
            self._exec(("predict", a0, g0, dt, t))
            self.counters["imu"] += 1
        self.last_imu = (t, np.asarray(acc, float), np.asarray(gyro, float))
        self._flush()

    def _advance_to(self, t):
        """No IMU over [state time, t]: coast at constant velocity and grow the
        uncertainty by an unknown acceleration of `gap_acc_std` (m/s^2)."""
        self._exec(("advance", float(t)))

    # ------------------------------------------------------------------ measurements
    def on_vo(self, t, v_body_flu, var):
        self._measure(float(t), "vo", (np.asarray(v_body_flu, float), np.asarray(var, float)))

    def on_attitude(self, t, R):
        self._measure(float(t), "att", np.asarray(R, float))

    def on_gps_velocity(self, t, v_world, var):
        self._measure(float(t), "gps", (np.asarray(v_world, float), np.asarray(var, float)))

    def on_position(self, t, p_world, var):
        """Absolute position fix, world NWU (m), with its per-axis variance (m^2).
        Only the axes in `pos_rows` are used (default north/west)."""
        self._measure(float(t), "pos", (np.asarray(p_world, float), np.asarray(var, float)))

    def _measure(self, t, kind, payload):
        if not self.ready:
            return
        if t > self.t:
            self.pending.append((t, kind, payload))
            return
        if kind == "pos" and self.replay_s > 0:
            self._late_position(t, payload)
            return
        if self.t - t > self.max_meas_age_s:
            if kind == "vo":
                self.counters["vo_dropped"] += 1
            elif kind == "pos":
                self.counters["pos_dropped"] += 1
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
        for _, kind, payload in sorted(due, key=lambda m: _ORDER[m[1]]):
            self._apply(kind, payload)

    def _apply(self, kind, payload):
        """First (live) application of a measurement: counters, then the op."""
        if kind == "vo":
            self._exec(("vo",) + payload)
            self.counters["vo"] += 1
        elif kind == "att":
            self.n_att += 1
            if self.n_att % self.attitude_every == 0:
                self._exec(("att", payload))
                self.counters["att"] += 1
        elif kind == "gps":
            self._exec(("gps",) + payload)
            self.counters["gps"] += 1
        elif kind == "pos":
            self._apply_position(*payload)

    def _apply_position(self, z, var):
        """Gated update; on lock-out (see the module docstring) a reset instead."""
        f, rows = self.f, list(self.pos_rows)
        self.counters["pos"] += 1
        if self._exec(("pos", z, var, "update")):
            self._pos_rejects = []
            return
        self.counters["pos_gated"] += 1
        r, _ = f.position_residual(z, rows, self.pos_lever)
        self._pos_rejects = (self._pos_rejects + [r])[-max(1, self.pos_reset_after):]
        if self.pos_reset_after <= 0 or len(self._pos_rejects) < self.pos_reset_after:
            return                                        # 0 = never reset
        res = np.array(self._pos_rejects)
        tol = self.pos_reset_agree_sigma * np.sqrt(np.asarray(var, float)[rows])
        if np.all(np.abs(res - res.mean(0)) <= tol):
            # replace the gated op just logged by the reset, so a replay redoes the reset
            if self.replay_s > 0:
                self._ops[-1] = ("pos", z, var, "reset")
            f.reset_position(z, var, rows, self.pos_lever)
            self.counters["pos_reset"] += 1
            self._pos_rejects = []

    def _late_position(self, t, payload):
        """Rewind to the first state at/after t, apply the fix there, re-run the rest."""
        if t < self.t - self.replay_s or t <= self._floor:
            self.counters["pos_dropped"] += 1
            return
        times = [c[0] for c in self._cps]
        i = bisect_left(times, t - 1e-9)
        t_cp, snap, op_i = self._cps[i]
        redo = list(self._ops)[op_i - self._op_base:]
        while len(self._cps) > i + 1:
            self._cps.pop()
        while len(self._ops) > op_i - self._op_base:
            self._ops.pop()
        self.f.restore(snap)
        self.t = t_cp
        if redo:
            self.counters["pos_late"] += 1                # a real rewind
        self._apply_position(*payload)
        self.f.record_stats = False                       # already counted once
        try:
            for op in redo:
                self._exec(op)
        finally:
            self.f.record_stats = True

    # ------------------------------------------------------------------ ops
    def _exec(self, op):
        """Run one filter operation; log it (and checkpoint) when replay is on.
        Returns the update's accept flag (None for predict / advance)."""
        f, kind, ok = self.f, op[0], None
        if kind == "predict":
            _, a, g, dt, t = op
            f.predict(a, g, dt)
            self.t = t
        elif kind == "advance":
            t = op[1]
            dt = t - self.t
            if dt > 0:
                s, P = self.prm, f.P
                a2 = self.gap_acc_std ** 2
                P[0:3, 0:3] += np.eye(3) * (a2 * dt ** 4 / 4.0)
                P[3:6, 3:6] += np.eye(3) * (a2 * dt ** 2)
                P[6:9, 6:9] += np.eye(3) * (s.gyro_noise ** 2 * dt)
                f.p = f.p + f.v * dt
            self.t = t
        elif kind == "vo":
            ok, _ = f.update_body_velocity(op[1], op[2], self.lever)
        elif kind == "att":
            ok, _ = f.update_attitude(op[1], self.std_tilt_deg, self.std_yaw_deg)
        elif kind == "gps":
            H = np.zeros((3, 15))
            H[:, 3:6] = np.eye(3)
            ok, _ = f._update(op[1] - f.v, H, np.diag(op[2]), "gps")
        elif kind == "pos":
            _, z, var, mode = op
            if mode == "reset":
                f.reset_position(z, var, self.pos_rows, self.pos_lever)
                ok = True
            else:
                ok, _ = f.update_position(z, var, self.pos_rows, self.pos_lever)
        if self.replay_s > 0:
            self._ops.append(op)
            if kind in ("predict", "advance"):
                self._checkpoint()
        return ok

    def _checkpoint(self):
        if self.replay_s <= 0:
            return
        self._cps.append((self.t, self.f.snapshot(), self._op_base + len(self._ops)))
        # keep the newest checkpoint at or before t - replay_s: a fix stamped just
        # inside the window lands on the checkpoint right after it
        while len(self._cps) > 1 and self._cps[1][0] <= self.t - self.replay_s:
            self._floor = self._cps.popleft()[0]
        keep_from = self._cps[0][2]
        while self._ops and self._op_base < keep_from:
            self._ops.popleft()
            self._op_base += 1

    # ------------------------------------------------------------------ output
    def state(self):
        s = self.f.state()
        s["t"] = self.t
        return s
