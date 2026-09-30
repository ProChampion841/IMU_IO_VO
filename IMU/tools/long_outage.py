"""Long unaided outages: the frozen bias PLUS MTI tilt aiding during the coast.

TWO CORRECTIONS, AND THEY ATTACK DIFFERENT HALVES OF THE ERROR.

  freeze   The pre-window bias freeze (``tools/prewindow_align.py``), validated for a
           40 s outage at 0.843x val / 0.732x test.  Constants fitted on the last 15 s
           of aided data and held for the whole outage.  It corrects the SENSORS.

  tilt     Gravity-referenced tilt aiding from the MTI, applied DURING the coast.
           It corrects the ATTITUDE, which is 59-64% of the error.

WHY TILT AIDING IS LEGITIMATE HERE.  The MTI is an independent onboard sensor; losing
GPS does not affect it.  It keeps producing attitude for the whole outage, and until
now the pipeline read it only for the initial state.  Nothing about using it during
the window is a leak.

WHY IT ONLY PAYS FOR LONG OUTAGES.  Measured on the 55 train flights
(``tools/mti_tilt_check.py``), the two tilt errors have different SHAPES:

    elapsed     40 s    100 s    200 s    300 s
    gyro rms   3.06     6.24     8.99     9.72    deg   (random walk, ~0.5 deg/sqrt(s))
    MTI  rms   2.58     2.87     2.93     3.05    deg   (flat -- a sensor bias)

and their COSTS differ too, because position error is the double integral of the
acceleration error: a CONSTANT tilt theta costs 1/2 g sin(theta) T^2 while a tilt
DRIFTING to theta costs only 1/6 g sin(theta) T^2.  Swapping a drift for a constant
of the same size therefore LOSES a factor of 3, so the MTI must be ~3x better in
angle before aiding is worth anything.  It reaches 3.19x at 300 s and only 1.19x at
40 s.  **Do not use this for short outages** -- at 40 s it should make things worse,
and that is a prediction this script can be run to check.

THE GAIN IS DERIVED, NOT TUNED.  A complementary filter's time constant should sit
where the two error sources are equal.  With gyro drift ~ 0.5 deg/sqrt(s) and an MTI
error of ~3.0 deg, 0.5 sqrt(tau) = 3.0 gives **tau ~ 36 s**, and the per-segment gain
is then simply alpha = L / tau for a segment length L.  ``--tau`` defaults to that
measured value; sweeping it on TRAIN only is how it should be checked.

THE CORRECTION IS YAW-FREE BY CONSTRUCTION, which matters because MTI heading is
unusable on this corpus (17.19 deg rms even after the wind-triangle fit) while its
tilt is sound.  Working with the body-frame gravity direction touches roll and pitch
and cannot touch heading:

    g_prop = R_prop^T z,  g_mti = R_mti^T z
    delta  = alpha * (g_mti x g_prop)          (perpendicular to both -> pure tilt)
    R_new  = R_prop Exp(delta)

`delta` is perpendicular to the body-frame vertical, so `R delta` is horizontal in
the world frame: a pure tilt rotation with exactly zero yaw component.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pypose as pp
import torch

from datasets.UAVdataset import UAV
from prewindow_align import freeze_biases

ARMS = ("raw", "freeze", "tilt", "both", "air", "all")
# Learned-correction arms, appended only when --net_ckpt is given.
#   net   the network on RAW acc/gyro -- the condition it was TRAINED in.
#   fnet  the network on FREEZE-corrected acc/gyro.  OFF-DISTRIBUTION: v11 never
#         saw pre-corrected input, so a poor fnet says nothing about whether a
#         residual model trained that way would work.  Exploratory only.
NET_ARMS = ("net", "fnet")


def load_net(cfg_path, ckpt_path, dev):
    """Build the checkpoint's network and report the sensor condition it implies."""
    from pyhocon import ConfigFactory
    from model import net_dict
    nconf = ConfigFactory.parse_file(cfg_path)
    net = net_dict[nconf.train.network](nconf.train).double().to(dev).eval()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    net.load_state_dict(sd)
    src = getattr(net, "att_source", None)
    att = getattr(net, "att_input", "none")
    print("[net] %s | epoch %s | interval %d | att_input=%s att_source=%s"
          % (nconf.train.network, ck.get("epoch"), net.interval, att, src))
    if att != "none" and src != "mti":
        raise ValueError("att_source=%r consumes the GPS-aided nav attitude during "
                         "the outage -- that is leakage, not a dead-reckoning result. "
                         "Only att_source='mti' or att_input='none' is admissible here."
                         % (src,))
    if att != "none":
        print("[net] SENSOR CONDITION: primary IMU + CONTINUOUS MTi attitude -- the "
              "same class as tilt/both, NOT the IMU-only class of raw/freeze.")
    return net


def tilt_correct(R, R_mti, alpha):
    """One yaw-free nudge of R toward the MTI's tilt.  See the module docstring."""
    up = torch.zeros(R.shape[:-2] + (3,), dtype=R.dtype, device=R.device)
    up[..., 2] = 1.0
    g_p = torch.einsum("...ji,...j->...i", R, up)
    g_m = torch.einsum("...ji,...j->...i", R_mti, up)
    delta = alpha * torch.cross(g_m, g_p, dim=-1)
    return pp.SO3(pp.mat2SO3(R).tensor()) @ pp.so3(delta).Exp()


def mti_tilt_offset(mti, nav):
    """This flight's MTI tilt bias, measured while still aided, as a body rotation.

    The MTI's tilt error is not white -- it is a per-flight OFFSET (the loader's own
    diagnostics report roll/pitch biases of a few tenths of a degree, and they are
    computed but never applied).  Aiding with a biased reference is what makes tilt
    aiding backfire on the flights whose gyro is already good: the filter faithfully
    pulls a correct attitude toward a wrong one.

    The aided phase can measure that offset, exactly as it measures the accel and
    gyro biases: compare the MTI's gravity direction with the nav filter's over
    ``[t0 - T, t0)`` and average.  Freeze it and apply it to the MTI for the whole
    outage.  Yaw-free by construction, like every other use of the MTI here.

    Returns the body-frame rotation vector d with R_mti Exp(d) matching nav tilt.
    """
    up = torch.zeros(mti.shape[:-1] + (3,), dtype=torch.float64, device=mti.device)
    up[..., 2] = 1.0
    Rm, Rn = mti.matrix(), nav.matrix()
    g_m = torch.einsum("...ji,...j->...i", Rm, up)
    g_n = torch.einsum("...ji,...j->...i", Rn, up)
    return torch.cross(g_n, g_m, dim=-1).mean(dim=1)


# Measured on the 55 train flights (tools/mti_tilt_check.py): gyro tilt error is a
# random walk at ~0.5 deg/sqrt(s) and the MTI's is flat at ~3.0 deg.  Both are
# MEASUREMENTS, not tuned values, and they are what the adaptive gain is built from.
Q_GYRO_DEG2_PER_S = 0.25        # (0.5 deg/sqrt(s))^2
R_MTI_DEG2 = 9.0                # (3.0 deg)^2


def per_flight_QR(integ, acc, gyro, dt, nav, mti, b_gyro, gravity):
    """Measure THIS flight's gyro process noise Q and MTI measurement noise R.

    The corpus-average Q and R above are what a fixed gain implicitly assumes, and
    they are wrong for any individual flight -- most visibly on the flights whose
    gyro is unusually good, where average-Q aiding faithfully drags a correct
    attitude toward a 3 deg reference and makes things worse.  Both constants are
    observable on the aided interval, per flight, with no ground truth beyond the
    nav solution that is live anyway:

        Q  propagate the BIAS-CORRECTED gyro across the aided interval from the nav
           attitude at its start and measure the tilt error that accumulates.  For a
           random walk that error grows as sqrt(Q t), so Q = theta^2 / T.
        R  the mean squared tilt angle between the MTI and the nav attitude over the
           same interval.

    NOTE this is a VARIANCE, not the mean offset that `mti_tilt_offset` freezes --
    and that distinction is the point.  Freezing the mean was measured to HURT
    (0.434x against 0.367x, worst date 1.650 against 1.350): a 15 s average of the
    MTI's tilt offset captures the manoeuvre state at handover, not a persistent
    bias, so it does not transfer.  A variance is a far more stable statistic and is
    exactly what the filter needs -- it sets how much to trust the MTI, not where to
    move it.

    Returns (Q in deg^2/s, R in deg^2), both shape (n, 1), floored so a degenerate
    interval falls back toward the corpus values rather than to zero or infinity.
    """
    n = acc.shape[0]
    z = torch.zeros(n, 1, 3, dtype=acc.dtype, device=acc.device)
    init = {"pos": z, "vel": z, "rot": pp.SO3(nav.tensor()[:, :1])}
    # TIMING.  dt = np.diff(t), so integrating all H intervals from nav[0] lands on
    # index H while the reference below is nav[-1] at index H-1.  Use H-1 intervals so
    # the propagation ends exactly on the sample compared against.  This matters more
    # than its size suggests: th_g is SQUARED to form Q, and th_g is smallest (0.1-0.6
    # deg) on the well-behaved windows, so a ~0.1-0.2 deg injection of omega*dt is a
    # large RELATIVE perturbation there -- measured 1.33x mean and up to 3.92x
    # inflation of Q, concentrated on the flights whose gyro is good.  Inflated Q
    # raises K = P/(P+R), i.e. makes the filter trust the 3 deg MTI MORE exactly where
    # it should trust it less, which is the failure this function exists to prevent.
    dtc = dt[:, :-1]
    with torch.no_grad():
        prop = integ(init_state=init, dt=dtc,
                     gyro=gyro[:, :-1] - b_gyro[:, None], acc=acc[:, :-1])["rot"]
    T = dtc.sum(dim=1).squeeze(-1).clamp(min=1e-6)

    def ang(A, B):
        # `up` must match the SLICE being compared, not the full aided interval
        u = torch.zeros(A.shape[:-1], dtype=A.dtype, device=A.device)
        u[..., 2] = 1.0
        ga = torch.einsum("...ji,...j->...i", A, u)
        gb = torch.einsum("...ji,...j->...i", B, u)
        return torch.rad2deg(torch.arccos((ga * gb).sum(-1).clamp(-1.0, 1.0)))
    Rn = nav.matrix()
    th_g = ang(prop.matrix()[:, -1:], Rn[:, -1:]).squeeze(-1)
    Q = (th_g.pow(2) / T).clamp(min=1e-4, max=100.0)
    R = ang(mti.matrix(), Rn).pow(2).mean(dim=1).clamp(min=0.05, max=400.0)
    return Q[:, None], R[:, None]


def fit_airdata(R, va, v_gps):
    """Calibrate the pitot against GPS on the aided interval:  v_gps = k R e_x Va + W.

    THE POINT OF DOING THIS AT ALL.  Every other correction in this file fights an
    INTEGRATION and therefore loses to time -- the accel bias integrates into velocity,
    the gyro drift integrates into attitude.  Air data integrates nothing.  A
    fixed-wing aircraft in coordinated flight has ~zero sideslip, so it flies where it
    points, and airspeed stops being a scalar and becomes a velocity VECTOR once
    attitude is known.  Scale and wind are both observable while GPS is up and both
    vary slowly, so freezing them at handover gives a velocity estimate whose error is
    roughly CONSTANT across the outage instead of growing.

    Measured on this corpus: fitted scale k = 0.993 (the pitot is well calibrated in
    flight -- the "reads 14.9 m/s while parked" problem is a static-port zero offset at
    rest, not a scale error) and an in-sample residual of 0.81 m/s.  Against raw IMU
    integration the two velocity-error curves cross at ~20 s, and by 40 s air data is
    11.6 m/s against the IMU's 26.3.

    Linear in (k, W_x, W_y, W_z): 3H equations, 4 unknowns, one lstsq.  Returns
    (k (n,), W (n,3), residual rms (n,)) -- the residual is the honest per-flight
    measurement noise for the blend below.
    """
    n, H = va.shape
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=R.dtype, device=R.device).expand(n, H, 3)
    u = torch.einsum("nhij,nhj->nhi", R, ex) * va[..., None]
    A = torch.zeros(n, 3 * H, 4, dtype=R.dtype, device=R.device)
    for i in range(3):
        A[:, i::3, 0] = u[..., i]
        A[:, i::3, 1 + i] = 1.0
    y = v_gps.reshape(n, 3 * H)
    x = torch.linalg.lstsq(A, y.unsqueeze(-1)).solution.squeeze(-1)
    res = (y - torch.einsum("nmp,np->nm", A, x)).reshape(n, H, 3)
    return x[:, 0], x[:, 1:4], res.norm(dim=-1).pow(2).mean(dim=1).sqrt()


def coast(integ, init, dt, gyro, acc, mti, seg, alpha, mti_bias=None, kalman=False,
          QR=None, air=None, legacy_mti=False):
    """Integrate the outage, optionally nudging attitude toward MTI tilt every `seg`.

    With `alpha` 0 and `kalman` off this is one straight call to the preintegrator and
    is bitwise the same as not segmenting at all -- verified, and worth keeping that
    way so the `raw`/`freeze` arms are never quietly changed by this code path.

    `kalman` replaces the fixed gain with the scalar Kalman gain for the system the
    measurements above describe: tilt error is a random walk with variance rate Q, the
    MTI measures it with variance R, and at handover the nav filter has just supplied
    attitude so the initial variance is ~0.  Then per segment of length dt_s:

        P <- P + Q dt_s ;   K = P / (P + R) ;   P <- (1 - K) P

    The point is that K RISES with time into the outage -- near zero at handover, when
    the gyro is fresh and the MTI would only corrupt it, approaching a steady 0.31 once
    the drift has grown past the MTI's own error.  A fixed gain cannot do that, and a
    fixed gain is what damages the flights whose gyro is unusually good.  There is no
    new tuned constant: Q and R are the two measured curves.
    """
    W = dt.shape[1]
    if ((alpha <= 0 and not kalman) or mti is None) and air is None:
        with torch.no_grad():
            o = integ(init_state=init, dt=dt, gyro=gyro, acc=acc)
        return o["pos"][:, -1], o["vel"][:, -1], o["rot"][:, -1]
    state = dict(init)
    # air = (k, W_wind, va, Rv_air, Qv_imu) -- velocity aiding from the pitot.
    # Same Kalman structure as the tilt blend, on a different quantity: the IMU's
    # velocity error grows while the air-data estimate's does not, so the gain rises
    # from ~0 at handover (GPS has just supplied velocity) toward trusting air data.
    # Decided ONCE, before the loop: the kalman branch rebinds `alpha` to a tensor, and
    # evaluating `alpha > 0` on a tensor inside an `or` raises rather than short-circuits.
    do_tilt = mti is not None and (kalman or (not torch.is_tensor(alpha) and alpha > 0))
    Pv = None
    if air is not None:
        k_air, W_air, va_air, Rv_air, Qv_imu = air
        Pv = torch.zeros_like(Rv_air)
    # NB: named Qv/Rv, not Q/R -- `R` is the rotation matrix inside this loop, and
    # letting the measurement noise share that name silently turned the variance into
    # a 3x3 matrix on the second segment.
    Qv, Rv = (QR if QR is not None else (Q_GYRO_DEG2_PER_S, R_MTI_DEG2))
    P = torch.zeros_like(Qv) if torch.is_tensor(Qv) else 0.0
    for s in range(0, W, seg):
        e = min(s + seg, W)
        if kalman:
            P = P + Qv * float(dt[0, s:e].sum())
            alpha = P / (P + Rv)
            P = P * (1.0 - alpha)
        with torch.no_grad():
            o = integ(init_state=state, dt=dt[:, s:e], gyro=gyro[:, s:e], acc=acc[:, s:e])
        R = o["rot"][:, -1].matrix()
        if do_tilt:
            # THE TIMING RULE: dt[s:e] advances from state s to state e, so the
            # propagated R IS state e and the MTI reference must be read at e, not
            # e-1.  `mti` is sliced to W+1 states for exactly this reason.
            Rm = mti[:, e - 1] if legacy_mti else mti[:, e]
            if mti_bias is not None:
                Rm = Rm @ pp.so3(mti_bias).Exp()      # de-biased reference
            Rc = tilt_correct(R, Rm.matrix(), alpha)
        else:
            Rc = pp.SO3(pp.mat2SO3(R).tensor())
        vel = o["vel"][:, -1]
        if air is not None:
            # air-data velocity uses the JUST-CORRECTED attitude, so better attitude
            # feeds better velocity -- the two aiding paths compound rather than compete
            ex = torch.tensor([1.0, 0.0, 0.0], dtype=vel.dtype,
                              device=vel.device).expand(vel.shape[0], 3)
            v_air = (k_air[:, None] * torch.einsum("nij,nj->ni", Rc.matrix(), ex)
                     * va_air[:, e - 1][:, None] + W_air)
            Pv = Pv + Qv_imu * float(dt[0, s:e].sum())
            Kv = Pv / (Pv + Rv_air)
            vel = vel + Kv * (v_air - vel)
            Pv = Pv * (1.0 - Kv)
        state = {"pos": o["pos"][:, -1:], "vel": vel[:, None],
                 "rot": pp.SO3(Rc.tensor()[:, None])}
    return (state["pos"][:, 0], state["vel"][:, 0],
            pp.SO3(state["rot"].tensor()[:, 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--win", type=int, default=30000, help="outage length, frames")
    ap.add_argument("--hist", type=float, default=15.0, help="aided fit length, seconds")
    ap.add_argument("--k_sub", type=int, default=10)
    ap.add_argument("--air_hist", type=float, default=None,
                    help="calibration length for the AIR-DATA fit only, seconds. "
                         "Defaults to --hist.  These are two different estimation "
                         "problems with opposite needs and there is no reason they "
                         "should share a window: the bias freeze wants a SHORT recent "
                         "interval because the gyro bias drifts, while separating the "
                         "pitot scale k from the wind vector needs HEADING DIVERSITY -- "
                         "on a straight leg k*V_a and the along-track wind are "
                         "confounded, exactly the conditioning failure that limits the "
                         "wind-triangle yaw estimator.  A longer air window buys turns.")
    ap.add_argument("--seg", type=int, default=500, help="frames between tilt nudges")
    ap.add_argument("--tau", type=float, default=36.0,
                    help="complementary time constant in SECONDS.  alpha = seg/tau. "
                         "Default is the measured crossover; see the module docstring.")
    ap.add_argument("--nwin", type=int, default=12)
    ap.add_argument("--start_s", type=float, default=120.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per_flight_qr", action="store_true",
                    help="measure Q and R on THIS flight's aided interval instead of "
                         "using the corpus averages.  Implies --kalman.")
    ap.add_argument("--kalman", action="store_true",
                    help="use the time-varying Kalman gain instead of a fixed --tau. "
                         "See coast().  Q and R come from measured error curves.")
    ap.add_argument("--mti_debias", action="store_true",
                    help="measure this flight's MTI tilt OFFSET on the aided interval "
                         "and remove it before aiding.  See mti_tilt_offset().")
    ap.add_argument("--by_date", action="store_true")
    ap.add_argument("--net_config", default=None,
                    help="HOCON config for the checkpoint in --net_ckpt.")
    ap.add_argument("--net_ckpt", default=None,
                    help="Trained checkpoint. Adds the 'net' and 'fnet' arms, scored "
                         "on the SAME windows and the SAME initial state as every "
                         "other arm.")
    ap.add_argument("--legacy_mti_timing", action="store_true",
                    help="Reproduce the pre-audit MTI index (mti[e-1] instead of mti[e]) "
                         "for measuring how much that off-by-one moved the tilt arms. "
                         "Affects tilt/both/all only; raw and freeze never segment.")
    ap.add_argument("--win_csv", default=None,
                    help="Dump PER-WINDOW rows with the ACTUAL elapsed duration from "
                         "dt.sum(), not the nominal --win/100. Sample rates vary "
                         "98.8-101.5 Hz so 4000 frames is 39.4-40.5 s depending on flight.")
    ap.add_argument("--csv", default=None,
                    help="Dump the per-flight/per-arm records used for pooling. "
                         "Reporting only -- changes no computation. Needed for "
                         "leave-one-flight-out, which stdout pooling cannot express.")
    a = ap.parse_args()

    recs, wrecs = [], []
    net = load_net(a.net_config, a.net_ckpt, a.device) if a.net_ckpt else None
    arms = ARMS + (NET_ARMS if net is not None else ())
    for f in a.files:
        try:
            seq = UAV(a.data_root, f, trim_to_airborne=True, mti_yaw_ref="fixed",
                      mti_diagnostics=False)
        except Exception:
            continue
        d = seq.data
        if "mti_orientation" not in d:
            continue
        t = d["time"].numpy()
        fs = 1.0 / float(np.median(np.diff(t)))
        H, W = int(a.hist * fs), a.win
        H_air = int((a.air_hist if a.air_hist is not None else a.hist) * fs)
        # both fits must sit inside the same pre-window history, and every arm must
        # score the SAME windows, so the start offset respects the longer of the two
        S = max(int(a.start_s * fs), H, H_air)
        starts = [s for s in range(S, t.shape[0] - W - 1, W)][:a.nwin]
        if not starts:
            continue
        dev = a.device
        st = lambda k, lo, hi: torch.stack([d[k][s + lo:s + hi]
                                            for s in starts]).to(dev).double()
        so = lambda k, lo, hi: pp.SO3(torch.stack([d[k][s + lo:s + hi].tensor()
                                                   for s in starts])).to(dev).double()
        integ = pp.module.IMUPreintegrator(reset=True, prop_cov=False,
                                           gravity=seq.gravity).double().to(dev)
        b_acc, b_gyro = freeze_biases(st("acc", -H, 0), st("gyro", -H, 0),
                                      st("dt", -H, 0), so("gt_orientation", -H, 0),
                                      st("velocity", -H, 0), seq.gravity, k_sub=a.k_sub,
                                      integ=integ)
        dt, acc, gyro = st("dt", 0, W), st("acc", 0, W), st("gyro", 0, W)
        # W+1 states: the coast loop lands on state e and must read the MTI there.
        mti = so("mti_orientation", 0, W if a.legacy_mti_timing else W + 1)
        init = {"pos": torch.stack([d["gt_translation"][s] for s in starts])[:, None].to(dev).double(),
                "vel": torch.stack([d["velocity"][s] for s in starts])[:, None].to(dev).double(),
                "rot": pp.SO3(torch.stack([d["gt_orientation"][s].tensor()
                                           for s in starts])[:, None]).to(dev).double()}
        gt_p = torch.stack([d["gt_translation"][s + W] for s in starts]).to(dev).double()
        gt_r = pp.SO3(torch.stack([d["gt_orientation"][s + W].tensor()
                                   for s in starts])).to(dev).double()
        gt_v = torch.stack([d["velocity"][s + W] for s in starts]).to(dev).double()
        alpha = min(1.0, a.seg / (a.tau * fs))
        # --- pitot calibration on the aided interval, frozen for the outage -----
        airpack = None
        if "airspeed" in d:
            k_a, W_a, rv = fit_airdata(so("gt_orientation", -H_air, 0).matrix(),
                                       st("airspeed", -H_air, 0).squeeze(-1),
                                       st("velocity", -H_air, 0))
            # Qv: how fast the INTEGRATED velocity error grows for this flight, measured
            # the same way as the gyro's Q -- propagate the aided interval with the
            # frozen biases and see what the velocity error reaches.  Rv: the air-data
            # residual, which is that estimate's own noise.  Both per flight, no tuning.
            zc = torch.zeros(len(starts), 1, 3, dtype=torch.float64, device=dev)
            init_a = {"pos": zc, "vel": st("velocity", -H, -H + 1),
                      "rot": pp.SO3(so("gt_orientation", -H, -H + 1).tensor())}
            with torch.no_grad():
                oa = integ(init_state=init_a, dt=st("dt", -H, 0),
                           gyro=st("gyro", -H, 0) - b_gyro[:, None],
                           acc=st("acc", -H, 0) - b_acc[:, None])
            Ta = st("dt", -H, 0).sum(dim=1).squeeze(-1).clamp(min=1e-6)
            ev = (oa["vel"][:, -1] - st("velocity", -H, 0)[:, -1]).norm(dim=-1)
            Qv = (ev.pow(2) / Ta).clamp(min=1e-6, max=1e4)[:, None]
            Rv = rv.pow(2).clamp(min=1e-3, max=1e4)[:, None]
            airpack = (k_a, W_a, st("airspeed", 0, W).squeeze(-1), Rv, Qv)

        QR = per_flight_QR(integ, st("acc", -H, 0), st("gyro", -H, 0), st("dt", -H, 0),
                           so("gt_orientation", -H, 0), so("mti_orientation", -H, 0),
                           b_gyro, seq.gravity) if a.per_flight_qr else None
        mbias = (mti_tilt_offset(so("mti_orientation", -H, 0),
                                 so("gt_orientation", -H, 0))
                 if a.mti_debias else None)
        WITH_B = ("freeze", "both", "all")
        WITH_T = ("tilt", "both", "all")
        WITH_A = ("air", "all")
        for arm in arms:
            if arm in NET_ARMS:
                # Reproduce the training collate EXACTLY (padding9): the front pad is
                # 9 synthetic frames, acc = R_init^-1 g and gyro = 0, NOT real history.
                # Feeding real history instead would be a distribution shift the
                # checkpoint never saw.
                iv = net.interval
                nb_a = b_acc if arm == "fnet" else torch.zeros_like(b_acc)
                nb_g = b_gyro if arm == "fnet" else torch.zeros_like(b_gyro)
                iden = torch.zeros(len(starts), iv, 3, dtype=dt.dtype, device=dev)
                iden[..., 2] = seq.gravity
                ndata = {"dt": dt,
                         "acc": torch.cat([init["rot"].Inv() * iden,
                                           acc - nb_a[:, None]], dim=1),
                         "gyro": torch.cat([torch.zeros_like(iden),
                                            gyro - nb_g[:, None]], dim=1),
                         "rot": so("gt_orientation", 0, W),
                         "mti_rot": mti[:, :W]}
                with torch.no_grad():
                    no = net.inference(ndata)
                    o = integ(init_state=init, dt=dt,
                              gyro=no["corrected_gyro"], acc=no["corrected_acc"])
                p, v = o["pos"][:, -1], o["vel"][:, -1]
                r = pp.SO3(o["rot"].tensor()[:, -1])
                err = (p - gt_p).norm(dim=-1)
                err_h = (p - gt_p)[:, :2].norm(dim=-1)
                rot = torch.rad2deg((r * gt_r.Inv()).Log().norm(dim=-1))
                verr = (v - gt_v).norm(dim=-1)
                recs.append({"flight": f, "date": f[:10], "nwin": len(starts),
                             "arm": arm, "mean": float(err.mean()),
                             "meanh": float(err_h.mean()), "rot": float(rot.mean()),
                             "vel": float(verr.mean())})
                for i, s0 in enumerate(starts):
                    wrecs.append({"flight": f, "date": f[:10], "arm": arm, "start": s0,
                                  "dur_s": float(dt[i].sum()), "pos_m": float(err[i]),
                                  "posh_m": float(err_h[i]), "vel_ms": float(verr[i]),
                                  "rot_deg": float(rot[i])})
                continue
            ba = b_acc if arm in WITH_B else torch.zeros_like(b_acc)
            bg = b_gyro if arm in WITH_B else torch.zeros_like(b_gyro)
            al = alpha if arm in WITH_T else 0.0
            p, v, r = coast(integ, init, dt, gyro - bg[:, None], acc - ba[:, None],
                            mti, a.seg, al, mti_bias=mbias,
                            kalman=(a.kalman or a.per_flight_qr) and arm in WITH_T,
                            QR=QR, air=(airpack if arm in WITH_A else None),
                            legacy_mti=a.legacy_mti_timing)
            err = (p - gt_p).norm(dim=-1)
            # HORIZONTAL-ONLY error = what PERFECT altitude aiding would leave.
            # In open-loop strapdown integration the vertical position error does not
            # feed back into the horizontal channels, so simply dropping the vertical
            # component is an honest UPPER BOUND on what a barometer could buy.  It is
            # a bound and not an estimate: a real baro carries its own noise and
            # pressure drift so it cannot reach this, while a baro-aided FILTER could
            # in principle do slightly better by also correcting vertical velocity and
            # the vertical accel bias, which this ignores.
            err_h = (p - gt_p)[:, :2].norm(dim=-1)
            rot = torch.rad2deg((r * gt_r.Inv()).Log().norm(dim=-1))
            verr = (v - gt_v).norm(dim=-1)
            recs.append({"flight": f, "date": f[:10], "nwin": len(starts), "arm": arm,
                         "mean": float(err.mean()), "meanh": float(err_h.mean()),
                         "rot": float(rot.mean()), "vel": float(verr.mean())})
            for i, s0 in enumerate(starts):
                wrecs.append({"flight": f, "date": f[:10], "arm": arm, "start": s0,
                              "dur_s": float(dt[i].sum()), "pos_m": float(err[i]),
                              "posh_m": float(err_h[i]), "vel_ms": float(verr[i]),
                              "rot_deg": float(rot[i])})
        g = {r["arm"]: r for r in recs if r["flight"] == f}
        print("  %-34s %2d win | raw %8.1f -> freeze %8.1f tilt %8.1f both %8.1f m"
              % (f[:34], len(starts), g["raw"]["mean"], g["freeze"]["mean"],
                 g["tilt"]["mean"], g["both"]["mean"]))

    if not recs:
        print("no usable flights")
        return
    if a.win_csv and wrecs:
        import csv as _csv
        with open(a.win_csv, "w", newline="") as _fh:
            _w = _csv.DictWriter(_fh, fieldnames=list(wrecs[0].keys()))
            _w.writeheader(); _w.writerows(wrecs)
        _du = [r["dur_s"] for r in wrecs]
        print("per-WINDOW records -> %s (%d rows) | actual duration %.2f-%.2f s "
              "(nominal %.1f s)" % (a.win_csv, len(wrecs), min(_du), max(_du), a.win / 100.0))
    if a.csv:
        import csv as _csv
        with open(a.csv, "w", newline="") as _fh:
            _w = _csv.DictWriter(_fh, fieldnames=list(recs[0].keys()))
            _w.writeheader(); _w.writerows(recs)
        print("per-flight records -> %s (%d rows)" % (a.csv, len(recs)))
    w = np.array([r["nwin"] for r in recs if r["arm"] == "raw"], dtype=float)
    print("\n" + "=" * 78)
    print("POOLED %d flights, %d windows | %.0f s outage, tau %.0f s, seg %.1f s"
          % (len(w), int(w.sum()), a.win / 100.0, a.tau, a.seg / 100.0))
    print("=" * 78)
    print("  %-8s %10s %10s %10s %10s %10s %9s"
          % ("arm", "mean m", "horiz m", "vel m/s", "rot deg", "vs raw", "alt gain"))
    print("  " + "-" * 64)
    base = None
    for arm in arms:
        v = np.array([r["mean"] for r in recs if r["arm"] == arm])
        vh = np.array([r["meanh"] for r in recs if r["arm"] == arm])
        rt = np.array([r["rot"] for r in recs if r["arm"] == arm])
        m = float(np.average(v, weights=w))
        mh = float(np.average(vh, weights=w))
        base = m if base is None else base
        vv = np.array([r["vel"] for r in recs if r["arm"] == arm])
        print("  %-8s %10.1f %10.1f %10.3f %10.2f %9.3fx %8.3fx"
              % (arm, m, mh, float(np.average(vv, weights=w)),
                 float(np.average(rt, weights=w)), m / base, mh / max(m, 1e-9)))
    print("")
    print("  'horiz m' drops the vertical component of the SAME errors, so 'alt gain'")
    print("  is the CEILING on what perfect altitude aiding (a barometer) could buy.")
    print("  A real baro has noise and pressure drift and cannot reach it.")

    if a.by_date:
        print("\n  %-12s %4s %5s %10s %s"
              % ("date", "fl", "win", "raw m", " ".join("%8s" % x for x in arms[1:])))
        print("  " + "-" * 62)
        worst = {k: 0.0 for k in arms[1:]}
        for dte in sorted({r["date"] for r in recs}):
            sel = lambda arm: [r for r in recs if r["date"] == dte and r["arm"] == arm]
            ww = np.array([r["nwin"] for r in sel("raw")], dtype=float)
            b = float(np.average([r["mean"] for r in sel("raw")], weights=ww))
            cells = []
            for arm in arms[1:]:
                v = float(np.average([r["mean"] for r in sel(arm)], weights=ww))
                worst[arm] = max(worst[arm], v / b)
                cells.append("%8.3f" % (v / b))
            print("  %-12s %4d %5d %10.1f %s"
                  % (dte, len(ww), int(ww.sum()), b, " ".join(cells)))
        print("  " + "-" * 62)
        print("  %-12s %4s %5s %10s %s"
              % ("WORST DATE", "", "", "", " ".join("%8.3f" % worst[k] for k in arms[1:])))


if __name__ == "__main__":
    main()
