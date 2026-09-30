"""Message logs for the stream mode: build one from a flight, write/read it, replay it.

A message log is what the aircraft sees: timestamped sensor messages in ARRIVAL
order.  The same file drives the Python StreamEKF and the C++ ekf_replay, which is
how the two are compared.  Format (one message per line, SI, NWU / FLU):

    INIT,t,px,py,pz,vx,vy,vz,qw,qx,qy,qz
    IMU,t,ax,ay,az,gx,gy,gz
    VO,t,vx,vy,vz,varx,vary,varz        body FLU velocity + variance
    ATT,t,qw,qx,qy,qz                   nav attitude, body -> world
    GPS,t,vx,vy,vz,varx,vary,varz       world velocity (only while GPS is up)
"""
import numpy as np

from . import so3
from .stream import StreamEKF

_PRIO = {"INIT": 0, "IMU": 1, "GPS": 2, "VO": 3, "ATT": 4}


def build_events(fl, vo, t_init, t_end, gps_until=None, gps_rate_hz=5.0, gps_std=0.1,
                 att_source="gt", vo_latency_s=0.0, imu_drop=0.0, seed=0):
    """Message log for one flight.

    fl         dict from pipeline.load_flight (raw IMU, truth)
    vo         VOStream (body FLU), or None for an IMU-only run
    t_init     the filter starts here, from the truth state (e.g. take-off / power-up)
    gps_until  GPS velocity is available until this time (the outage starts there)
    vo_latency_s   a VO message ARRIVES this long after its timestamp (it is still
                   stamped with its own time; the filter handles it as late)
    imu_drop   fraction of IMU samples lost
    """
    rng = np.random.default_rng(seed)
    t = fl["t"]
    i0, i1 = np.searchsorted(t, [t_init, t_end])
    R_att = fl["R_gt"] if att_source == "gt" else fl["R_mti"]
    ev = [(t[i0], "INIT", t[i0], np.r_[fl["p_gt"][i0], fl["v_gt"][i0],
                                        so3.mat_to_quat(fl["R_gt"][i0])])]
    for k in range(i0, i1 + 1):
        if k > i0 and imu_drop and rng.random() < imu_drop:
            continue
        ev.append((t[k], "IMU", t[k], np.r_[fl["acc"][k], fl["gyro"][k]]))
        if k > i0:
            ev.append((t[k], "ATT", t[k], so3.mat_to_quat(R_att[k])))
    if gps_until is not None:
        for tg in np.arange(t[i0] + 1.0 / gps_rate_hz, min(gps_until, t[i1]), 1.0 / gps_rate_hz):
            k = int(np.clip(np.searchsorted(t, tg), 0, len(t) - 1))
            v = fl["v_gt"][k] + rng.standard_normal(3) * gps_std
            ev.append((t[k], "GPS", t[k], np.r_[v, [gps_std ** 2] * 3]))
    if vo is not None:
        for tv, v, var in zip(vo.t, vo.v, vo.var):
            if t[i0] < tv <= t[i1]:
                ev.append((tv + vo_latency_s, "VO", tv, np.r_[v, var]))
    ev.sort(key=lambda e: (e[0], _PRIO[e[1]]))
    return [(typ, tt, vals) for _, typ, tt, vals in ev]


def write_events(path, events):
    with open(path, "w") as f:
        f.write("# type,t,values...  (EKF/ekf/events.py)\n")
        for typ, t, vals in events:
            f.write("%s,%.6f,%s\n" % (typ, t, ",".join("%.12g" % x for x in vals)))


def read_events(path):
    out = []
    for line in open(path):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.strip().split(",")
        out.append((parts[0], float(parts[1]), np.array([float(x) for x in parts[2:]])))
    return out


def run_stream(events, make_filter):
    """Replay through a StreamEKF.  Returns the state after every IMU message:
    (M, 19) = t, p, v, q(wxyz), ba, bg, std_p, std_v -- the columns ekf_replay writes."""
    s = make_filter()
    rows = []
    for typ, t, x in events:
        if typ == "INIT":
            s.initialize(t, x[0:3], x[3:6], so3.quat_to_mat(x[6:10]))
        elif typ == "IMU":
            s.on_imu(t, x[0:3], x[3:6])
            if s.ready:
                f = s.f
                rows.append(np.r_[s.t, f.p, f.v, so3.mat_to_quat(f.R), f.ba, f.bg,
                                  np.sqrt(np.trace(f.P[0:3, 0:3])),
                                  np.sqrt(np.trace(f.P[3:6, 3:6]))])
        elif typ == "VO":
            s.on_vo(t, x[0:3], x[3:6])
        elif typ == "ATT":
            s.on_attitude(t, so3.quat_to_mat(x[0:4]))
        elif typ == "GPS":
            s.on_gps_velocity(t, x[0:3], x[3:6])
    return np.array(rows), s


def stream_from_config(cfg):
    """StreamEKF built from an ekf_default.json dict (same keys the C++ reads)."""
    from .eskf import ESKFParams
    aid, vo = cfg["attitude_aid"], cfg["vo"]
    lever = np.asarray(vo.get("lever_arm_m", [0, 0, 0]), float)
    st = cfg.get("stream", {})
    return lambda: StreamEKF(ESKFParams.from_dict(cfg["eskf"]),
                             attitude_every=aid.get("every", 10),
                             std_tilt_deg=aid.get("std_tilt_deg"),
                             std_yaw_deg=aid.get("std_yaw_deg"),
                             lever=lever if np.any(lever) else None,
                             max_meas_age_s=st.get("max_meas_age_s", 1.0),
                             max_imu_gap_s=st.get("max_imu_gap_s", 0.1),
                             gap_acc_std=st.get("gap_acc_std", 2.0))
