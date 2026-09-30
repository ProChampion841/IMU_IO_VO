"""Error-state EKF: IMU propagation + VO body-velocity and attitude updates.

CONVENTIONS (the IMU project's, see IMU/datasets/UAVdataset.py)
    world  NWU (x north, y west, z UP)      body  FLU (x forward, y left, z up)
    R      body -> world rotation matrix     g     [0, 0, 9.81007] (z up)
    A level, stationary accelerometer reads +g on body z.

NOMINAL STATE        p (3)  v (3)  R (3x3)  ba (3)  bg (3)
ERROR STATE (15)     dx = [dp, dv, dtheta, dba, dbg]
    attitude error is RIGHT-multiplied, in the body frame:  R_true = R Exp(dtheta)

PROPAGATION (one IMU sample, a = acc - ba, w = gyro - bg)
    a_w = R a - g
    p  <- p + v dt + 1/2 a_w dt^2
    v  <- v + a_w dt
    R  <- R Exp(w dt)
The rotation applied to the specific force is the one at the START of the step --
the same order as pypose's IMUPreintegrator, so with no updates this is the IMU
project's dead reckoning.

ERROR DYNAMICS (continuous)
    dp'     = dv
    dv'     = -R [a]x dtheta - R dba - R n_a
    dtheta' = -[w]x dtheta - dbg - n_g
    dba'    = n_ba            dbg' = n_bg
discretised per sample with Phi_thth = Exp(-w dt) (exact) and first order elsewhere.

UPDATES
    body velocity (VO)    z = R^T v + (w x r_lever)            H_v = R^T, H_th = [R^T v]x,
                                                                H_bg = [r_lever]x
    attitude (nav / MTi)  z = Log(R_meas R^T)  (WORLD-frame)   H_th = R
                          rows 0-1 = tilt (roll/pitch), row 2 = heading; either can
                          be switched off.
Every update is chi-square gated (NIS) and uses the Joseph form.
"""
from dataclasses import dataclass, field

import numpy as np

from . import so3

# 99.9 % chi-square quantiles, dof 1..6 -- a table rather than a scipy dependency
_CHI2_999 = {1: 10.828, 2: 13.816, 3: 16.266, 4: 18.467, 5: 20.515, 6: 22.458}


@dataclass
class ESKFParams:
    gravity: float = 9.81007
    # continuous-time noise densities
    acc_noise: float = 0.05        # m/s^2 / sqrt(Hz)
    gyro_noise: float = 0.005      # rad/s / sqrt(Hz)     (~0.29 deg/s/sqrt(Hz))
    acc_bias_rw: float = 1e-4      # m/s^3 / sqrt(Hz)
    gyro_bias_rw: float = 1e-5     # rad/s^2 / sqrt(Hz)
    # initial 1-sigma
    init_pos_std: float = 0.1      # m
    init_vel_std: float = 0.1      # m/s
    init_att_std_deg: float = 0.5  # deg, roll/pitch/yaw
    init_acc_bias_std: float = 0.03   # m/s^2   (after the 15 s freeze: the residual)
    init_gyro_bias_std: float = 2e-4  # rad/s
    gate: bool = True              # chi-square gating of every update
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class ESKF:
    IP, IV, ITH, IBA, IBG = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)

    def __init__(self, p, v, R, ba=None, bg=None, params=None, P0=None):
        self.prm = params or ESKFParams()
        self.p = np.array(p, dtype=float)
        self.v = np.array(v, dtype=float)
        self.R = np.array(R, dtype=float)
        self.ba = np.zeros(3) if ba is None else np.array(ba, dtype=float)
        self.bg = np.zeros(3) if bg is None else np.array(bg, dtype=float)
        self.g = np.array([0.0, 0.0, self.prm.gravity])
        if P0 is None:
            s = self.prm
            d = np.r_[[s.init_pos_std] * 3, [s.init_vel_std] * 3,
                      [np.radians(s.init_att_std_deg)] * 3,
                      [s.init_acc_bias_std] * 3, [s.init_gyro_bias_std] * 3]
            P0 = np.diag(d ** 2)
        self.P = np.array(P0, dtype=float)
        self.last_gyro = np.zeros(3)
        self.stats = {"vel": [0, 0], "att": [0, 0], "nis_vel": [], "nis_att": []}

    # ------------------------------------------------------------------ predict
    def predict(self, acc, gyro, dt):
        dt = float(dt)
        s = self.prm
        a = np.asarray(acc, float) - self.ba
        w = np.asarray(gyro, float) - self.bg
        self.last_gyro = np.asarray(gyro, float)
        R = self.R
        a_w = R @ a - self.g
        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt
        dR = so3.exp(w * dt)
        self.R = R @ dR

        Phi = np.eye(15)
        Ra = R @ so3.skew(a)
        Phi[self.IP, self.IV] = np.eye(3) * dt
        Phi[self.IP, self.ITH] = -0.5 * Ra * dt * dt
        Phi[self.IP, self.IBA] = -0.5 * R * dt * dt
        Phi[self.IV, self.ITH] = -Ra * dt
        Phi[self.IV, self.IBA] = -R * dt
        Phi[self.ITH, self.ITH] = dR.T
        Phi[self.ITH, self.IBG] = -np.eye(3) * dt
        q = np.r_[[0.0] * 3, [s.acc_noise ** 2 * dt] * 3, [s.gyro_noise ** 2 * dt] * 3,
                  [s.acc_bias_rw ** 2 * dt] * 3, [s.gyro_bias_rw ** 2 * dt] * 3]
        self.P = Phi @ self.P @ Phi.T
        self.P[np.diag_indices(15)] += q

    # ------------------------------------------------------------------ update
    def _update(self, r, H, Rm, kind):
        S = H @ self.P @ H.T + Rm
        Si = np.linalg.inv(S)
        nis = float(r @ Si @ r)
        self.stats["nis_" + kind].append(nis)
        if self.prm.gate and nis > _CHI2_999[len(r)]:
            self.stats[kind][1] += 1
            return False, nis
        K = self.P @ H.T @ Si
        dx = K @ r
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.p = self.p + dx[self.IP]
        self.v = self.v + dx[self.IV]
        self.R = so3.orthonormalize(self.R @ so3.exp(dx[self.ITH]))
        self.ba = self.ba + dx[self.IBA]
        self.bg = self.bg + dx[self.IBG]
        # The reset Jacobian I - [dtheta/2]x is ~I for the small corrections here.
        self.stats[kind][0] += 1
        return True, nis

    def update_body_velocity(self, z, var, lever=None):
        """z: body-FLU velocity (m/s), var: per-axis variance (m/s)^2."""
        z = np.asarray(z, float)
        vb = self.R.T @ self.v
        h = vb.copy()
        H = np.zeros((3, 15))
        H[:, self.IV] = self.R.T
        H[:, self.ITH] = so3.skew(vb)
        if lever is not None and np.any(lever):
            lever = np.asarray(lever, float)
            h = h + np.cross(self.last_gyro - self.bg, lever)
            H[:, self.IBG] = so3.skew(lever)
        return self._update(z - h, H, np.diag(np.asarray(var, float)), "vel")

    def update_attitude(self, R_meas, std_tilt_deg=None, std_yaw_deg=None):
        """Attitude aid.  None switches that part off (e.g. tilt-only with MTi)."""
        rows, var = [], []
        if std_tilt_deg is not None:
            rows += [0, 1]
            var += [np.radians(std_tilt_deg) ** 2] * 2
        if std_yaw_deg is not None:
            rows += [2]
            var += [np.radians(std_yaw_deg) ** 2]
        if not rows:
            return None, None
        phi_w = so3.log(np.asarray(R_meas, float) @ self.R.T)       # world-frame error
        H = np.zeros((3, 15))
        H[:, self.ITH] = self.R
        return self._update(phi_w[rows], H[rows], np.diag(var), "att")

    # ------------------------------------------------------------------ helpers
    def state(self):
        return dict(p=self.p.copy(), v=self.v.copy(), R=self.R.copy(),
                    ba=self.ba.copy(), bg=self.bg.copy(), P=self.P.copy())
