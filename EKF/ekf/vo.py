"""VO measurements: load the VO model's predictions, or simulate them.

VO OUTPUT CONTRACT (VO/logs/input_contract.json)
    body_velocity_m_s_x/y/z            body velocity, FRD (x fwd, y right, z down)
    velocity_log_variance_x/y/z        natural log of the variance, per axis
                                       (VO's NLL is 0.5*(r^2 exp(-lv) + lv))
The label VO is trained on is  v_body = R_GPSNavEul^T v_NED,  i.e. FRD.

VO CADENCE (VO/src/vio/data/image_pairs.py).  One visual measurement is an image PAIR
(i, i + frame_gap); it is delivered at the first telemetry tick after
exposure_t1 + deployment_latency_s.  The VO network then writes a velocity on EVERY
telemetry tick, holding the latest visual token between pairs.  With frame_gap 10
and one inference per 500 ms, only ONE ROW PER 0.5 s carries new image information;
the rows in between are the same token re-read with fresh telemetry.  Feeding every
row to a Kalman filter counts one image many times and makes it over-confident, so
the loader keeps only the fresh rows:
    visual_age == 0  (or visual_present == 1) when the CSV has that column,
    otherwise one row per `min_interval_s` (default 0.5 s).

CSV this module reads -- one row per VO output, any extra columns ignored:
    time                   seconds, SAME clock as the IMU log's `Time`
                           (aliases: Time, time_s, t)
    vx, vy, vz             aliases: body_velocity_m_s_x/y/z
    logvar_x/y/z           optional; aliases: velocity_log_variance_x/y/z
                           or var_x/y/z (variance) or std_x/y/z (m/s)
    visual_age             optional; ticks since the last image pair (0 = fresh)
                           alias: visual_present (1 = fresh)
Frame is FRD unless told otherwise (`frame="flu"`).  Everything is returned in
body FLU, the frame the EKF runs in: FRD -> FLU flips y and z; variances do not
change.
"""
import csv

import numpy as np

from .so3 import frd_to_flu

# Measured VO accuracy (VO/logs/metrics.csv, validation, last epochs): per-axis RMSE
# of body velocity.  The default VO noise when a file carries no variance column,
# and the default size of the simulated VO error.
VO_VAL_RMSE_FRD = np.array([3.5, 2.7, 0.95])

_ALIASES = {
    "time": ("time", "Time", "time_s", "t"),
    "v": (("vx", "vy", "vz"),
          ("body_velocity_m_s_x", "body_velocity_m_s_y", "body_velocity_m_s_z")),
    "logvar": (("logvar_x", "logvar_y", "logvar_z"),
               ("velocity_log_variance_x", "velocity_log_variance_y",
                "velocity_log_variance_z")),
    "var": (("var_x", "var_y", "var_z"),),
    "std": (("std_x", "std_y", "std_z"),),
}


class VOStream:
    """Time-stamped body-FLU velocity with per-axis variance."""

    def __init__(self, t, v_flu, var, source):
        order = np.argsort(t)
        self.t = np.asarray(t, float)[order]
        self.v = np.asarray(v_flu, float)[order]
        self.var = np.asarray(var, float)[order]
        self.source = source

    def __len__(self):
        return len(self.t)

    def window(self, t0, t1):
        """Samples with t0 < t <= t1."""
        i0, i1 = np.searchsorted(self.t, [t0, t1], side="right")
        return VOStream(self.t[i0:i1], self.v[i0:i1], self.var[i0:i1], self.source)


def _pick(header, groups):
    for g in groups:
        if isinstance(g, str):
            if g in header:
                return g
        elif all(c in header for c in g):
            return g
    return None


def load_vo_csv(path, frame="frd", time_offset=0.0, default_std=None, var_scale=1.0,
                min_std=0.05, min_interval_s=0.5, fresh_only=True):
    """Read a VO prediction CSV -> VOStream in body FLU.

    time_offset is ADDED to the VO time to put it on the IMU clock.
    var_scale inflates the VO variance (VO errors are time-correlated, which a
    per-sample variance does not describe; >1 is the usual fix).
    fresh_only keeps one row per image pair (see VO CADENCE above); min_interval_s
    is the pair period used when the file has no visual_age / visual_present.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("%s is empty" % path)
    header = rows[0].keys()
    tcol = _pick(header, _ALIASES["time"])
    vcols = _pick(header, _ALIASES["v"])
    if tcol is None or vcols is None:
        raise ValueError("%s needs a time column %s and velocity columns %s"
                         % (path, _ALIASES["time"], _ALIASES["v"]))
    col = lambda names: np.array([[float(r[c]) for c in names] for r in rows])
    t = np.array([float(r[tcol]) for r in rows]) + float(time_offset)
    v = col(vcols)
    lv, vv, sv = (_pick(header, _ALIASES[k]) for k in ("logvar", "var", "std"))
    if lv:
        var = np.exp(col(lv))
    elif vv:
        var = col(vv)
    elif sv:
        var = col(sv) ** 2
    else:
        std = VO_VAL_RMSE_FRD if default_std is None else np.broadcast_to(default_std, (3,))
        var = np.tile(np.asarray(std, float) ** 2, (len(t), 1))
    var = np.maximum(var * float(var_scale), min_std ** 2)
    if frame.lower() == "frd":
        v = frd_to_flu(v)
    elif frame.lower() != "flu":
        raise ValueError("frame must be 'frd' or 'flu'")
    ok = np.isfinite(t) & np.isfinite(v).all(1) & np.isfinite(var).all(1)
    how = "all rows"
    if fresh_only:
        if "visual_age" in header:
            ok &= np.array([float(r["visual_age"]) == 0 for r in rows])
            how = "visual_age == 0"
        elif "visual_present" in header:
            ok &= np.array([float(r["visual_present"]) > 0.5 for r in rows])
            how = "visual_present == 1"
        elif min_interval_s and min_interval_s > 0:
            keep = np.zeros(len(t), bool)
            last = -np.inf
            for i in np.argsort(t):
                if ok[i] and t[i] - last >= min_interval_s - 1e-6:
                    keep[i], last = True, t[i]
            ok &= keep
            how = "one row per %g s" % min_interval_s
    return VOStream(t[ok], v[ok], var[ok], "csv:%s (%s, %d of %d rows)"
                    % (path, how, int(ok.sum()), len(t)))


def simulate_vo(t, R, v_world, rate_hz=2.0, white_std=None, bias_std=None,
                tau_s=20.0, seed=0, var_scale=1.0):
    """SIMULATED VO from ground truth -- for testing the filter, NOT a result.

    Body-FLU truth R^T v, sampled at rate_hz (default 2 Hz: one image pair per
    500 ms, frame_gap 10), plus
        white noise   N(0, white_std^2) per sample
        a slow error  first-order Gauss-Markov (time constant tau_s, std bias_std)
    The slow part is there on purpose: real VO error is strongly time-correlated,
    and white-only simulated VO makes any filter look far better than it will be.
    Defaults split the measured VO RMSE between the two.
    The variance handed to the filter is white^2 + bias^2 (times var_scale).
    """
    rng = np.random.default_rng(seed)
    rmse = np.abs(frd_to_flu(VO_VAL_RMSE_FRD))
    white = rmse / np.sqrt(2.0) if white_std is None else np.broadcast_to(white_std, (3,))
    bias = rmse / np.sqrt(2.0) if bias_std is None else np.broadcast_to(bias_std, (3,))
    t = np.asarray(t, float)
    ts = np.arange(t[0], t[-1], 1.0 / rate_hz)
    idx = np.clip(np.searchsorted(t, ts), 0, len(t) - 1)
    vb = np.einsum("nji,nj->ni", R[idx], v_world[idx])
    dt = 1.0 / rate_hz
    phi = np.exp(-dt / tau_s)
    gm = np.empty((len(ts), 3))
    gm[0] = rng.standard_normal(3) * bias
    for k in range(1, len(ts)):
        gm[k] = phi * gm[k - 1] + np.sqrt(1 - phi * phi) * bias * rng.standard_normal(3)
    z = vb + gm + rng.standard_normal((len(ts), 3)) * white
    var = np.tile((np.asarray(white) ** 2 + np.asarray(bias) ** 2) * var_scale, (len(ts), 1))
    return VOStream(t[idx], z, var, "SIMULATED(rate %g Hz, tau %g s)" % (rate_hz, tau_s))
