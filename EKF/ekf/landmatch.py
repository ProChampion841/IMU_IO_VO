"""Absolute position fixes (land matching / map matching) for the EKF.

A fix is the position of the aircraft -- or of the camera, with
position_aid.pos_lever_arm_m -- in the filter's world frame: NWU metres (x north,
y west, z up) from a local origin.  The filter uses it through
StreamEKF.on_position (ekf/stream.py), normally the horizontal axes only.

Every fix has TWO times:
    t           when the image it was matched from was taken (the stamp the
                filter applies it at)
    t_arrival   when the matcher delivered it (t + processing latency).  The
                filter rewinds to t (ekf/stream.py, replay_s), so latency costs
                nothing as long as it is below replay_s.

Sources:
    simulate_fixes   GPS truth + noise (+ a slow correlated error, + wrong matches):
                     tests the filter, it is NOT a land-matching result
    read_fixes_csv   your matcher's output (NWU metres, or lat/lon + an origin)

The fix must be where the AIRCRAFT (or camera) is.  A matcher that reports the
ground point at the image centre is off by height-above-ground x tan(tilt)
(100 m and 5 deg: 8.7 m) and must be converted first.
"""
import csv
from dataclasses import dataclass

import numpy as np

# WGS-84
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)


@dataclass
class PosFixes:
    t: np.ndarray            # (N,)   stamp: when the image was taken, s
    t_arrival: np.ndarray    # (N,)   when the fix reached the filter, s
    p: np.ndarray            # (N, 3) NWU m
    var: np.ndarray          # (N, 3) m^2
    source: str = ""

    def __len__(self):
        return len(self.t)


def _ecef(lat_deg, lon_deg, alt):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = _A / np.sqrt(1.0 - _E2 * np.sin(lat) ** 2)
    return np.stack([(n + alt) * np.cos(lat) * np.cos(lon),
                     (n + alt) * np.cos(lat) * np.sin(lon),
                     (n * (1.0 - _E2) + alt) * np.sin(lat)], axis=-1)


def geodetic_to_nwu(lat_deg, lon_deg, alt_m, origin):
    """WGS-84 lat/lon (deg) and height (m) -> local NWU metres about
    origin = (lat0, lon0, alt0).  Exact (via ECEF), any distance."""
    lat0, lon0, alt0 = (float(x) for x in origin)
    d = _ecef(np.asarray(lat_deg, float), np.asarray(lon_deg, float),
              np.asarray(alt_m, float)) - _ecef(lat0, lon0, alt0)
    la, lo = np.radians(lat0), np.radians(lon0)
    east = -np.sin(lo) * d[..., 0] + np.cos(lo) * d[..., 1]
    north = (-np.sin(la) * np.cos(lo) * d[..., 0] - np.sin(la) * np.sin(lo) * d[..., 1]
             + np.cos(la) * d[..., 2])
    up = (np.cos(la) * np.cos(lo) * d[..., 0] + np.cos(la) * np.sin(lo) * d[..., 1]
          + np.sin(la) * d[..., 2])
    return np.stack([north, -east, up], axis=-1)


def simulate_fixes(t, p_gt, t_from, t_to, rate_hz=1.0, std_m=10.0, bias_std_m=0.0,
                   tau_s=60.0, latency_s=0.5, outlier_rate=0.0, outlier_m=200.0,
                   var_scale=1.0, seed=0):
    """SIMULATED land-matching fixes from the GPS-truth trajectory.

    t, p_gt      flight time (N,) and truth position (N, 3), NWU m
    std_m        white error per axis (m) -- also the variance the fix reports
    bias_std_m   slow error (first-order Gauss-Markov, time constant tau_s): real
                 matching errors persist while the same terrain is in view
    latency_s    the fix arrives this long after its image was taken
    outlier_rate fraction of fixes that are WRONG MATCHES, off by outlier_m (m)
                 horizontally in a random direction -- the gate must reject them
    var_scale    the reported variance is (std_m^2 + bias_std_m^2) * var_scale
    """
    rng = np.random.default_rng(seed)
    ts = np.arange(t_from + 1.0 / rate_hz, t_to, 1.0 / rate_hz)
    k = np.clip(np.searchsorted(t, ts), 0, len(t) - 1)
    ts = t[k]                                   # stamp on an IMU sample, like VO
    n = len(ts)
    bias = np.zeros((n, 3))
    if bias_std_m > 0 and n:
        a = np.exp(-1.0 / (rate_hz * tau_s))
        b = rng.standard_normal(3) * bias_std_m
        for i in range(n):
            bias[i] = b
            b = a * b + np.sqrt(1.0 - a * a) * bias_std_m * rng.standard_normal(3)
    p = p_gt[k] + rng.standard_normal((n, 3)) * std_m + bias
    wrong = rng.random(n) < outlier_rate
    ang = rng.uniform(0.0, 2.0 * np.pi, n)
    p[wrong, 0] += outlier_m * np.cos(ang[wrong])
    p[wrong, 1] += outlier_m * np.sin(ang[wrong])
    var = np.full((n, 3), (std_m ** 2 + bias_std_m ** 2) * var_scale)
    return PosFixes(ts, ts + latency_s, p, var,
                    "SIMULATED land match: GPS truth + %.1f m white + %.1f m slow, %.1f Hz, "
                    "%.2f s late, %.0f%% wrong" % (std_m, bias_std_m, rate_hz, latency_s,
                                                   100.0 * outlier_rate))


def _col(names, *cands):
    low = {n.strip().lower(): n for n in names}
    for c in cands:
        if c in low:
            return low[c]
    return None


def read_fixes_csv(path, origin=None, time_offset=0.0, latency_s=0.0, std_m=10.0,
                   var_scale=1.0):
    """Your matcher's fixes.  One row per fix; columns (case does not matter):

    time            time / t / time_s -- when the IMAGE was taken, IMU clock
                    (time_offset is added)
    arrival         optional: arrival / t_arrival -- when the fix was delivered;
                    else time + latency_s
    position        north, west[, up]  or  x, y[, z]      NWU metres, or
                    lat, lon[, alt]                        degrees + origin=(lat0, lon0, alt0)
    uncertainty     optional: std (one value, m), or std_x/std_y/std_z, or
                    var_x/var_y/var_z; else std_m.  Multiplied by var_scale.
    quality         optional: valid / ok -- rows with 0 are skipped
    """
    if origin is not None:
        origin = tuple(float(x) for x in origin)
        if len(origin) not in (2, 3):
            raise ValueError("origin is (lat0, lon0) or (lat0, lon0, alt0), got %r" % (origin,))
        origin = origin + (0.0,) * (3 - len(origin))
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return PosFixes(np.zeros(0), np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)), path)
    names = list(rows[0].keys())
    ct = _col(names, "time", "t", "time_s")
    if ct is None:
        raise ValueError("%s: no time column (time / t / time_s)" % path)
    ca = _col(names, "arrival", "t_arrival", "arrival_s")
    cv = _col(names, "valid", "ok")
    get = lambda r, c, d=0.0: float(r[c]) if c and r[c] not in ("", None) else d  # noqa: E731
    rows = [r for r in rows if cv is None or get(r, cv, 1.0) != 0.0]
    t = np.array([get(r, ct) for r in rows]) + time_offset
    ta = (np.array([get(r, ca) for r in rows]) + time_offset if ca else t + latency_s)
    cn, cw = _col(names, "north", "x"), _col(names, "west", "y")
    clat, clon = _col(names, "lat", "latitude"), _col(names, "lon", "lng", "longitude")
    if cn and cw:
        cu = _col(names, "up", "z")
        p = np.array([[get(r, cn), get(r, cw), get(r, cu)] for r in rows])
    elif clat and clon:
        if origin is None:
            raise ValueError("%s has lat/lon: give the origin (lat0, lon0, alt0) of the "
                             "filter's NWU frame (--pos_origin)" % path)
        calt = _col(names, "alt", "altitude", "height")
        p = geodetic_to_nwu([get(r, clat) for r in rows], [get(r, clon) for r in rows],
                            [get(r, calt, float(origin[2]) if len(origin) > 2 else 0.0)
                             for r in rows], origin)
    else:
        raise ValueError("%s: no position columns (north/west, x/y or lat/lon)" % path)
    cs = _col(names, "std", "std_m")
    sx = [_col(names, "std_" + a) for a in "xyz"]
    vx = [_col(names, "var_" + a) for a in "xyz"]
    if cs:
        var = np.repeat(np.array([[get(r, cs, std_m) ** 2] for r in rows]), 3, axis=1)
    elif all(sx[:2]):
        var = np.array([[get(r, c, std_m) ** 2 for c in sx] for r in rows])
    elif all(vx[:2]):
        var = np.array([[get(r, c, std_m ** 2) for c in vx] for r in rows])
    else:
        var = np.full((len(rows), 3), std_m ** 2)
    o = np.argsort(t, kind="stable")
    return PosFixes(t[o], ta[o], p[o], var[o] * var_scale, path)
