"""SYNTHETIC verification of the timing fixes in ``tools/prewindow_align.py``.

No flight data.  A trajectory is generated whose IMU stream is *exactly* consistent
with the discrete model the closed-form fits assume, so the fits have an unambiguous
right answer and any residual is the estimator's own, not discretisation noise.

WHAT IS BEING TESTED.  ``d["dt"] = np.diff(t)`` in ``datasets/UAVdataset.py``, so on a
slice of H samples ``dt[k]`` spans ``[t_k, t_k+1]`` and integrating all H intervals
advances from index 0 to index **H** -- while the references the fit compares against,
``rot[:, -1]`` and ``vel[:, -1]``, are index **H-1**.  The fixed code integrates H-1
intervals; the old code integrated H.  ``_fit_core_new`` / ``_fit_core_old`` below
differ in exactly that and nothing else; ``_fit_core_old`` is
``git show 7899b09:tools/prewindow_align.py`` verbatim.

WHY BOTH VARIANTS LIVE IN THIS FILE.  ``tools/prewindow_align.py`` was observed being
rewritten underneath a running interpreter (see the banner this script prints), so a
test that imported only the live ``_fit_core`` would silently compare the old code
against itself and report "no difference".  Both variants are therefore carried here,
and the live module is separately fingerprinted and cross-checked against them.

THE GENERATOR.  Per interval k the model is the left-endpoint one:
    R_{k+1} = R_k Exp(w_k dt_k)
    v_{k+1} = v_k + R_k a_k dt_k - g dt_k,      g = [0, 0, gravity]
so given attitude and velocity sampled at the (irregular) times, the measurements that
reproduce them exactly are
    w_k = Log(R_k^T R_{k+1}) / dt_k
    a_k = R_k^T ( (v_{k+1} - v_k)/dt_k + g )
and the measured streams add the constant biases b_g, b_a.

WHAT "RECOVERY" MEANS FOR THE GYRO.  The gyro fit returns
    b_est = -Log(R_prop(t_ref)^T R_nav(t_ref)) / T ,
and to first order in b, R_prop = R_nav Exp(phi) with phi(T) = R(T)^T int R(t) b dt.
So on a turning aircraft this estimator returns the ADJOINT-AVERAGED bias, not b: over
150 deg of yaw the horizontal components come back rotated by ~75 deg and shrunk by
~26%.  That is a property of the estimator and has nothing to do with the timing bug,
and it is far larger than the timing term -- so scoring the turning case against the
constant b_g would drown the thing under test.  Two exact cases are used instead:

    ZERO-BIAS NULL   b_g = b_a = 0 on a fully turning trajectory.  Correct timing must
                     return exactly 0; the adjoint subtlety cannot arise.  Any nonzero
                     output IS the timing error.
    PURE-YAW EXACT   rotation about the body z axis only, b_g along z.  Everything
                     commutes, so the adjoint average equals b_g and the fit must
                     return it exactly.  The body frame still rotates 150 deg.

The accel fit has no such subtlety: with correct timing it is algebraically exact for
any motion.
"""
import hashlib
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pypose as pp
import torch

import prewindow_align as pa
from prewindow_align import pick

GRAVITY = 9.81007
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


# --------------------------------------------------------------------------- #
#  OLD, buggy: integrates ALL H dt, compares against rot[:, -1] / vel[:, -1]    #
#  (git show 7899b09:tools/prewindow_align.py, verbatim)                        #
# --------------------------------------------------------------------------- #
def _fit_core_old(integ, acc, gyro, dt, rot, vel, gravity):
    n = acc.shape[0]
    dev, dtp = acc.device, acc.dtype

    init = {"pos": torch.zeros(n, 1, 3, dtype=dtp, device=dev),
            "vel": torch.zeros(n, 1, 3, dtype=dtp, device=dev),
            "rot": pp.SO3(rot.tensor()[:, :1])}
    with torch.no_grad():
        prop = pick(integ(init_state=init, dt=dt, gyro=gyro, acc=acc)["rot"], -1)
    T = dt.sum(dim=1).squeeze(-1)
    b_gyro = -(prop.Inv() * pick(rot, -1)).Log() / T[:, None]

    R = rot.matrix()
    w = dt
    int_R = (R * w[..., None]).sum(dim=1)
    int_Racc = (torch.einsum("ntij,ntj->nti", R, acc) * w).sum(dim=1)
    g = torch.zeros(n, 3, dtype=dtp, device=dev)
    g[:, 2] = gravity
    dv = vel[:, -1] - vel[:, 0]
    rhs = int_Racc - g * T[:, None] - dv
    b_acc = torch.linalg.solve(int_R, rhs.unsqueeze(-1)).squeeze(-1)
    return b_acc, b_gyro


# --------------------------------------------------------------------------- #
#  FIXED: H-1 intervals, so the integration lands on the compared sample        #
# --------------------------------------------------------------------------- #
def _fit_core_new(integ, acc, gyro, dt, rot, vel, gravity):
    n = acc.shape[0]
    dev, dtp = acc.device, acc.dtype

    dtc, accc, gyroc = dt[:, :-1], acc[:, :-1], gyro[:, :-1]
    init = {"pos": torch.zeros(n, 1, 3, dtype=dtp, device=dev),
            "vel": torch.zeros(n, 1, 3, dtype=dtp, device=dev),
            "rot": pp.SO3(rot.tensor()[:, :1])}
    with torch.no_grad():
        prop = pick(integ(init_state=init, dt=dtc, gyro=gyroc, acc=accc)["rot"], -1)
    T = dtc.sum(dim=1).squeeze(-1)
    b_gyro = -(prop.Inv() * pick(rot, -1)).Log() / T[:, None]

    R = rot.matrix()[:, :-1]
    w = dtc
    int_R = (R * w[..., None]).sum(dim=1)
    int_Racc = (torch.einsum("ntij,ntj->nti", R, accc) * w).sum(dim=1)
    g = torch.zeros(n, 3, dtype=dtp, device=dev)
    g[:, 2] = gravity
    dv = vel[:, -1] - vel[:, 0]
    rhs = int_Racc - g * T[:, None] - dv
    b_acc = torch.linalg.solve(int_R, rhs.unsqueeze(-1)).squeeze(-1)
    return b_acc, b_gyro


def _pred_lambda(integ, acc, gyro, dt, rot, vel, gravity, fit, end):
    """``predictive_lambda`` with the held-out half ending at index ``H-1+end``.

    ``end=-1`` is the fixed form (integrate to the sample being compared against),
    ``end=0`` the old one.  ``fit`` is the ``_fit_core`` variant used for the half-fit.
    """
    H = acc.shape[1]
    h = H // 2
    if h < 4:
        z = torch.zeros(acc.shape[0], 3, dtype=acc.dtype, device=acc.device)
        return z, z
    sl = lambda t, a, b: t[:, a:b]
    ro = lambda a, b: pp.SO3(rot.tensor()[:, a:b])
    b_acc, b_gyro = fit(integ, sl(acc, 0, h), sl(gyro, 0, h), sl(dt, 0, h),
                        ro(0, h), sl(vel, 0, h), gravity)
    E = H + end
    A, D = sl(acc, h, E), sl(dt, h, E)
    R = ro(h, E).matrix()
    T = D.sum(dim=1).squeeze(-1)
    g = torch.zeros(acc.shape[0], 3, dtype=acc.dtype, device=acc.device)
    g[:, 2] = gravity
    dv = vel[:, -1] - vel[:, h]
    base = (torch.einsum("ntij,ntj->nti", R, A) * D).sum(dim=1) - g * T[:, None] - dv
    corr = base - torch.einsum("nij,nj->ni", (R * D[..., None]).sum(dim=1), b_acc)
    lam_a = (1.0 - corr.pow(2) / base.pow(2).clamp(min=1e-18)).clamp(0.0, 1.0)

    n = acc.shape[0]
    z3 = torch.zeros(n, 1, 3, dtype=acc.dtype, device=acc.device)
    init = {"pos": z3, "vel": z3, "rot": pp.SO3(rot.tensor()[:, h:h + 1])}
    tgt = pick(ro(h, H), -1)
    err = []
    for b in (torch.zeros_like(b_gyro), b_gyro):
        with torch.no_grad():
            o = pick(integ(init_state=init, dt=D, gyro=sl(gyro, h, E) - b[:, None],
                           acc=A)["rot"], -1)
        err.append((o * tgt.Inv()).Log().abs())
    lam_g = (1.0 - err[1].pow(2) / err[0].pow(2).clamp(min=1e-18)).clamp(0.0, 1.0)
    return lam_a, lam_g


# --------------------------------------------------------------------------- #
#  synthetic generator                                                         #
# --------------------------------------------------------------------------- #
def make_flight(H, yaw_rate_dps, seed, jitter=0.05, roll_amp=12.0, pitch_amp=5.0,
                b_acc=(0.05, -0.03, 0.08), b_gyro_dps=(0.06, -0.04, 0.09)):
    """One synthetic aided interval of H samples with KNOWN constant biases.

    Shapes match what ``prewindow_align`` expects for a batch of one: acc/gyro/vel
    ``(1, H, 3)``, dt ``(1, H, 1)``, rot an ``SO3`` of lshape ``(1, H)``.
    """
    rng = np.random.default_rng(seed)
    # IRREGULAR ~100 Hz -- exactly the dt = np.diff(t) convention of datasets/UAVdataset.py
    dt = 0.01 * (1.0 + jitter * rng.standard_normal(H))
    dt = np.clip(dt, 0.004, 0.02)
    t = np.concatenate([[0.0], np.cumsum(dt)])          # H+1 sample times

    # TURNING: steady yaw rate + roll/pitch oscillation -> the body frame really rotates
    yaw = yaw_rate_dps * D2R * t
    roll = roll_amp * D2R * np.sin(2 * np.pi * t / 7.0)
    pitch = pitch_amp * D2R * np.sin(2 * np.pi * t / 11.0 + 0.7)
    eul = torch.tensor(np.stack([roll, pitch, yaw], axis=-1), dtype=torch.float64)
    R = pp.euler2SO3(eul)                               # lshape (H+1,)

    # coordinated turn, speed and climb rate both varying
    V = 40.0 + 2.0 * np.sin(2 * np.pi * t / 13.0)
    vz = 1.5 * np.sin(2 * np.pi * t / 9.0)
    vel = torch.tensor(np.stack([V * np.cos(yaw), V * np.sin(yaw), vz], axis=-1),
                       dtype=torch.float64)

    dtt = torch.tensor(dt, dtype=torch.float64)
    Rk = pp.SO3(R.tensor()[:-1])
    # w_k = Log(R_k^-1 R_{k+1}) / dt_k
    gyro_t = (Rk.Inv() * pp.SO3(R.tensor()[1:])).Log() / dtt[:, None]
    g = torch.tensor([0.0, 0.0, GRAVITY], dtype=torch.float64)
    # a_k = R_k^T (dv/dt + g)
    acc_t = Rk.Inv() @ ((vel[1:] - vel[:-1]) / dtt[:, None] + g)

    ba = torch.tensor(b_acc, dtype=torch.float64)
    bg = torch.tensor(np.asarray(b_gyro_dps, dtype=np.float64) * D2R,
                      dtype=torch.float64)
    return {"acc": (acc_t + ba)[None], "gyro": (gyro_t + bg)[None],
            "dt": dtt[None, :, None], "vel": vel[:-1][None],
            "rot": pp.SO3(R.tensor()[:-1][None]),
            "b_acc": ba[None], "b_gyro": bg[None]}


def adjoint_reference(f):
    """First-order target of the gyro fit, from known truth:
    phi(T)/T = R_{H-1}^T ( sum_{k<H-1} R_k b dt_k ) / T ."""
    R = f["rot"].tensor()[0]
    dt = f["dt"][0, :, 0]
    T = dt[:-1].sum()
    world = ((pp.SO3(R[:-1]) @ f["b_gyro"][0]) * dt[:-1, None]).sum(dim=0) / T
    return (pp.SO3(R[-1]).Inv() @ world)[None]


def nrm(x):
    return float(torch.linalg.norm(x, dim=-1).max())


def args_of(f):
    return (f["acc"], f["gyro"], f["dt"], f["rot"], f["vel"], GRAVITY)


def predicted_gyro_term(f):
    """|w_end| * dt_end / T -- the fake rate the old slicing injects."""
    T = float(f["dt"][0, :-1, 0].sum())
    return float(torch.linalg.norm(f["gyro"][0, -1] - f["b_gyro"][0])) * \
        float(f["dt"][0, -1, 0]) / T * R2D


# --------------------------------------------------------------------------- #
def main():
    integ = pp.module.IMUPreintegrator(reset=True, prop_cov=False,
                                       gravity=GRAVITY).double()
    H = 1500                                   # 15 s at ~100 Hz
    fails, npass = [], [0]

    def check(cond, msg):
        print("    %-4s %s" % ("PASS" if cond else "FAIL", msg))
        if cond:
            npass[0] += 1
        else:
            fails.append(msg)

    # ---- 0. which version is actually on disk right now? ---------------------
    src = inspect.getsource(pa._fit_core)
    live_fixed = "dt[:, :-1]" in src and "rot.matrix()[:, :-1]" in src
    psrc = inspect.getsource(pa.predictive_lambda)
    live_pl_fixed = "sl(acc, h, H - 1)" in psrc
    with open(pa.__file__, "rb") as fh:
        digest = hashlib.md5(fh.read()).hexdigest()[:12]

    print("=" * 92)
    print("SYNTHETIC TIMING VERIFICATION -- %d samples, ~100 Hz irregular (5%% jitter), "
          "float64" % H)
    print("b_acc = [0.05, -0.03, 0.08] m/s^2    b_gyro = [0.06, -0.04, 0.09] deg/s")
    print("-" * 92)
    print("LIVE MODULE %s  md5 %s" % (pa.__file__, digest))
    print("  _fit_core         : %s" % ("FIXED (H-1 intervals)" if live_fixed
                                        else "OLD (integrates all H)"))
    print("  predictive_lambda : %s" % ("FIXED (held-out half ends at H-1)"
                                        if live_pl_fixed else "OLD (ends at H)"))
    print("=" * 92)

    # ---- 1. ZERO-BIAS NULL TEST: the decisive, assumption-free one ----------
    print("\n[1] ZERO-BIAS NULL TEST -- fully turning trajectory (yaw 10 deg/s = 150 deg")
    print("    of heading in 15 s, roll +-12 deg / 7 s, pitch +-5 deg / 11 s, 40 m/s")
    print("    coordinated turn), but b_acc = b_gyro = 0.  A correctly-timed fit MUST")
    print("    return exactly zero.  Whatever it returns instead IS the timing error.")
    f0 = make_flight(H, 10.0, 12345, b_acc=(0., 0., 0.), b_gyro_dps=(0., 0., 0.))
    ba_n, bg_n = _fit_core_new(integ, *args_of(f0))
    ba_o, bg_o = _fit_core_old(integ, *args_of(f0))
    print("      accel  FIXED %.3e m/s^2      OLD %.3e m/s^2"
          % (nrm(ba_n), nrm(ba_o)))
    print("      gyro   FIXED %.3e deg/s      OLD %.3e deg/s"
          % (nrm(bg_n) * R2D, nrm(bg_o) * R2D))
    print("             (for scale, a real UAV turn-on gyro bias is ~0.06 deg/s and the")
    print("              accel bias ~0.05 m/s^2, so the OLD numbers are %.0f%% and %.0f%%"
          % (100 * nrm(bg_o) * R2D / 0.06, 100 * nrm(ba_o) / 0.05))
    print("              of the quantity being estimated -- pure fabrication.)")
    print("      predicted old gyro term |w_end| dt_end / T = %.3e deg/s"
          % predicted_gyro_term(f0))
    print("\n    assertions:")
    check(nrm(ba_n) < 1e-12, "FIXED accel returns exactly zero on zero bias (< 1e-12)")
    check(nrm(bg_n) * R2D < 1e-12, "FIXED gyro returns exactly zero on zero bias")
    check(nrm(ba_o) > 1e-3, "OLD accel fabricates a bias > 1e-3 m/s^2 out of nothing")
    check(nrm(bg_o) * R2D > 1e-3, "OLD gyro fabricates a bias > 1e-3 deg/s out of nothing")
    check(abs(nrm(bg_o) * R2D - predicted_gyro_term(f0)) < 0.15 * predicted_gyro_term(f0),
          "OLD gyro fabrication matches the predicted |w_end| dt_end / T")

    # ---- 2. PURE-YAW EXACT: known non-zero answer, still 150 deg of rotation --
    print("\n[2] PURE-YAW EXACT -- rotation about body z only (still 150 deg of yaw in")
    print("    the interval), gyro bias on z alone, so every rotation commutes and the")
    print("    fit's exact answer is b_gyro itself.  Accel bias stays a full 3-vector.")
    f1 = make_flight(H, 10.0, 777, roll_amp=0.0, pitch_amp=0.0,
                     b_gyro_dps=(0.0, 0.0, 0.09))
    ba_n, bg_n = _fit_core_new(integ, *args_of(f1))
    ba_o, bg_o = _fit_core_old(integ, *args_of(f1))
    ean, eao = nrm(ba_n - f1["b_acc"]), nrm(ba_o - f1["b_acc"])
    egn, ego = nrm(bg_n - f1["b_gyro"]) * R2D, nrm(bg_o - f1["b_gyro"]) * R2D
    print("      accel |b_hat - b_true|  FIXED %.3e   OLD %.3e m/s^2  (%.0fx worse)"
          % (ean, eao, eao / max(ean, 1e-300)))
    print("      gyro  |b_hat - b_true|  FIXED %.3e   OLD %.3e deg/s  (%.0fx worse)"
          % (egn, ego, ego / max(egn, 1e-300)))
    print("      OLD gyro is %.1f%% of the true 0.0900 deg/s bias" % (100 * ego / 0.09))
    print("\n    assertions:")
    check(ean < 1e-12, "FIXED accel recovers b_acc exactly (< 1e-12 m/s^2)")
    check(egn < 1e-12, "FIXED gyro recovers b_gyro exactly (< 1e-12 deg/s)")
    check(eao > 1e6 * max(ean, 1e-300), "OLD accel error is many orders larger")
    check(ego > 1e6 * max(egn, 1e-300), "OLD gyro error is many orders larger")

    # ---- 3. FULL TURNING, FULL 3-AXIS BIAS: honest end-to-end numbers --------
    print("\n[3] FULL TURNING + FULL 3-AXIS BIAS -- the realistic case.  Read the gyro")
    print("    column against the ADJOINT reference, not against b_gyro: over 150 deg of")
    print("    yaw this estimator is not trying to return b_gyro (see the docstring).")
    f2 = make_flight(H, 10.0, 12345)
    ba_n, bg_n = _fit_core_new(integ, *args_of(f2))
    ba_o, bg_o = _fit_core_old(integ, *args_of(f2))
    ref = adjoint_reference(f2)
    print("      accel vs b_true      FIXED %.3e   OLD %.3e m/s^2"
          % (nrm(ba_n - f2["b_acc"]), nrm(ba_o - f2["b_acc"])))
    print("      gyro  vs b_true      FIXED %.3e   OLD %.3e deg/s"
          % (nrm(bg_n - f2["b_gyro"]) * R2D, nrm(bg_o - f2["b_gyro"]) * R2D))
    print("      gyro  vs adjoint ref FIXED %.3e   OLD %.3e deg/s"
          % (nrm(bg_n - ref) * R2D, nrm(bg_o - ref) * R2D))
    print("      |FIXED - OLD| gyro   %.3e deg/s   predicted %.3e deg/s"
          % (nrm(bg_n - bg_o) * R2D, predicted_gyro_term(f2)))
    print("      the adjoint average itself sits %.3e deg/s from b_gyro -- that is the"
          % (nrm(ref - f2["b_gyro"]) * R2D,))
    print("      estimator's own blind spot on a turn, not the timing bug.")
    print("\n    assertions:")
    check(nrm(ba_n - f2["b_acc"]) < 1e-12,
          "FIXED accel exact even under full 3-axis motion")
    check(nrm(ba_o - f2["b_acc"]) > 1e-3, "OLD accel error > 1e-3 m/s^2")
    check(abs(nrm(bg_n - bg_o) * R2D - predicted_gyro_term(f2))
          < 0.15 * predicted_gyro_term(f2),
          "FIXED-vs-OLD gyro gap matches the predicted timing term")

    # ---- 4. wrappers: estimate_biases and predictive_lambda -----------------
    print("\n[4] WRAPPERS.  estimate_biases() and predictive_lambda() are exercised with")
    print("    each _fit_core swapped in, so the wrapper's own arithmetic is covered too.")
    for tag, fc in (("FIXED", _fit_core_new), ("OLD  ", _fit_core_old)):
        pa._fit_core = fc                       # monkeypatch, restored below
        ba, bg, la, lg = pa.estimate_biases(integ, *args_of(f1), k_sub=10)
        print("      %s estimate_biases: |b_acc err| %.3e m/s^2  |b_gyro err| %.3e deg/s"
              "  lam_acc>=%.4f" % (tag, nrm(ba - f1["b_acc"]),
                                   nrm(bg - f1["b_gyro"]) * R2D, float(la.min())))
    pa._fit_core = _fit_core_new
    ba, bg, la, lg = pa.estimate_biases(integ, *args_of(f1), k_sub=10)
    check(nrm(ba - f1["b_acc"]) < 1e-12,
          "estimate_biases with the fixed core returns the exact accel bias")
    check(nrm(bg - f1["b_gyro"]) * R2D < 1e-12,
          "estimate_biases with the fixed core returns the exact gyro bias")
    check(float(la.min()) > 0.99 and float(lg.min()) > 0.99,
          "dispersion gates ~1 for a genuinely constant, noise-free bias")

    print("      predictive_lambda -- with an exactly constant bias and no noise the")
    print("      held-out residual must vanish, so lam must be 1.0:")
    for tag, fc, end in (("FIXED", _fit_core_new, -1), ("OLD  ", _fit_core_old, 0)):
        la_, lg_ = _pred_lambda(integ, *args_of(f1), fit=fc, end=end)
        print("        %s lam_acc %s  lam_gyro %s"
              % (tag, np.array2string(la_[0].numpy(), precision=6),
                 np.array2string(lg_[0].numpy(), precision=4)))
        if tag == "FIXED":
            check(float(la_.min()) > 0.999999,
                  "FIXED predictive gate is 1.0 on a perfectly constant bias")
            lam_fixed = float(la_.min())
        else:
            check(float(la_.min()) < lam_fixed,
                  "OLD predictive gate is degraded by the same one-interval mismatch")

    # ---- 5. sweep -----------------------------------------------------------
    print("\n[5] SWEEP over yaw rate, ZERO-BIAS null test at each.  The old error is")
    print("    driven by the motion at the end of the interval, so it grows with the")
    print("    turn rate; the fixed one stays at machine zero.")
    print("      %-9s %11s %11s %11s %11s"
          % ("yaw d/s", "acc FIXED", "acc OLD", "gyr FIXED", "gyr OLD"))
    for yr in (0.5, 2.0, 5.0, 10.0, 20.0):
        fz = make_flight(H, yr, 4242, b_acc=(0., 0., 0.), b_gyro_dps=(0., 0., 0.))
        an, gn = _fit_core_new(integ, *args_of(fz))
        ao, go = _fit_core_old(integ, *args_of(fz))
        print("      %-9.1f %11.2e %11.2e %11.2e %11.2e"
              % (yr, nrm(an), nrm(ao), nrm(gn) * R2D, nrm(go) * R2D))
        check(nrm(an) < 1e-12 and nrm(gn) * R2D < 1e-12,
              "yaw %.1f deg/s: FIXED is exactly zero" % yr)
        check(nrm(ao) > 1e-4 and nrm(go) * R2D > 1e-4,
              "yaw %.1f deg/s: OLD fabricates a bias" % yr)

    # ---- 6. the live module -------------------------------------------------
    print("\n[6] THE LIVE MODULE -- does what is on disk right now behave as fixed?")
    an, gn = pa._fit_core.__wrapped__(integ, *args_of(f0)) \
        if hasattr(pa._fit_core, "__wrapped__") else (None, None)
    import importlib
    pa2 = importlib.reload(pa)                  # undo the monkeypatch from [4]
    an, gn = pa2._fit_core(integ, *args_of(f0))
    print("      live _fit_core on the ZERO-BIAS case: accel %.3e m/s^2, gyro %.3e deg/s"
          % (nrm(an), nrm(gn) * R2D))
    check(nrm(an) < 1e-12 and nrm(gn) * R2D < 1e-12,
          "the _fit_core currently on disk passes the zero-bias null test")
    la_, lg_ = pa2.predictive_lambda(integ, *args_of(f1))
    print("      live predictive_lambda lam_acc %s"
          % np.array2string(la_[0].numpy(), precision=6))
    check(float(la_.min()) > 0.999999,
          "the predictive_lambda currently on disk gates a perfect bias at 1.0")

    # ---- 7. chunk_bounds -- REMOVED ------------------------------------------
    # This section exercised chunk_bounds() from tools/gyro_chunk_position.py,
    # deleted on 2026-09-07 with the rest of the gyro-first tooling (that line is
    # measured closed). The function no longer exists anywhere in the repo, so the
    # test had nothing left to guard. Sections 1-6, which verify the timing fixes
    # in tools/prewindow_align.py, are untouched -- and they matter MORE now that
    # tests/ has been removed, because they are the only remaining guard on the
    # timing rule that has produced four shipped bugs in this repo.
    # Recover with: git show <commit>^:tools/gyro_chunk_position.py

    print("\n" + "=" * 92)
    if fails:
        print("RESULT: %d of %d CHECKS FAILED" % (len(fails), len(fails) + npass[0]))
        for m in fails:
            print("   - %s" % m)
    else:
        print("RESULT: ALL %d CHECKS PASSED" % npass[0])
    print("=" * 92)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
