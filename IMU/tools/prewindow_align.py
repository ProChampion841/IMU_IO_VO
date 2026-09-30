"""Estimate the IMU biases while still aided, freeze them, then coast.

The decomposition (``tools/error_decompose.py``) said the 40 s error is a per-flight
CONSTANT vector -- within a flight the along-track component is 80% explained by its
own mean, and the sign flips between flights.  That is a turn-on bias, drawn fresh
each power cycle.  A per-window causal network cannot see it (the bias is 4-15x
below what a window's own kinematics can reveal), but the aided phase *before* the
window can: GPS was still on, so position, velocity and the nav filter's attitude
were all available, for minutes.

So: fit constants on ``[t0 - T, t0)`` using only what was observable there, freeze
them at handover, and integrate the unaided window with them subtracted.  Nothing
inside the window is used to fit anything.

TWO SEPARATE FITS, AND THAT IS THE POINT
----------------------------------------
``tools/oracle_split.py`` showed that fitting acc+gyro jointly against a position
objective lets the optimiser buy position by breaking attitude -- 40 s position to
33.6 m while attitude error goes 4.7 -> 11.3 deg.  Here each bias is fitted against
the quantity it actually corrupts, so that trade is not available:

  gyro bias   propagate rotation from the nav attitude at ``t0 - T`` and compare
              against the nav attitude at ``t0``.  A gyro bias makes the propagated
              rotation drift by ``b_g * T`` in the body frame, so
                  b_g = -Log(R_prop(t0)^T R_nav(t0)) / T
              This is measured against ATTITUDE.  It cannot trade for position.

  accel bias  the velocity increment is exact, so no differentiation of GPS
              velocity is needed:
                  dv = int R (acc - b_a) dt - g T
              which rearranges to a 3x3 linear solve,
                  (int R dt) b_a = int R acc dt - g T - dv
              This is measured against VELOCITY, with the nav attitude supplied, so
              a tilt cannot hide in it either.

Both are closed form.  No optimiser, no initial guess, no learning rate to get
wrong -- which matters, because a shared learning rate across two biases 1000x
apart in scale is exactly what voided the earlier oracle numbers.

WHAT IS AND IS NOT A LEAK.  Using ``gt_orientation`` and ``velocity`` on
``[t0 - T, t0)`` is not cheating: those are the onboard GPS/INS nav solution, which
by construction still exists right up to the moment aiding is lost.  The window
itself uses none of it beyond the initial state.  ``--init mti`` swaps the handover
attitude for the independent MTI solution, which is the harsher scenario where the
nav filter is unavailable too.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import pypose as pp
import torch

from datasets.UAVdataset import UAV

ARMS = ("raw", "acc", "gyro", "both", "shrunk", "cv", "cvs", "multi", "multiw")


def pick(so3, i):
    """``so3[:, i]`` via the raw tensor.  A guard, not a fix for anything here.

    There IS a real pypose trap, but it is SHAPE-GATED and narrower than it first
    looks.  Integer-indexing a LieTensor is only unsafe once the tensor is already
    at final rotation shape -- lshape ``(n,)``, tensor shape ``(n, 4)`` -- because
    then ``[:, i]`` indexes the QUATERNION COMPONENTS.  That case warns "Tensor
    Shape Invalid by calling <slot wrapper '__getitem__'>" and then either raises
    or returns nonsense.  On lshape ``(n, W)`` -- which is what
    ``IMUPreintegrator`` returns, the full per-step trajectory -- ``[:, -1]``
    selects the last STEP, emits no warning, and is bitwise equal to this helper.

    Verified 2026-09-04 on real data: ``out["rot"][:, -1]`` and
    ``pp.SO3(out["rot"].tensor()[:, -1])`` give identical per-window attitude
    errors to 0.000e+00, with and without ``--gtrot``.  So every use of ``pick``
    in this file is a no-op; it is kept only so the intent is explicit and so a
    future ``--win 1`` cannot reach the unsafe shape.

    THE BUG THAT ACTUALLY BIT ME was mine, in a throwaway diagnostic, and worth
    recording because it is silent and easy to repeat: ``torch.norm``'s first
    POSITIONAL argument is ``p``, the order of the norm, NOT ``dim``.  So
    ``.Log().norm(-1)`` computes a -1 norm over every element and returns a 0-dim
    scalar, where ``.Log().norm(dim=-1)`` returns the per-window vector.  On one
    flight that reported a 40 s attitude error of 0.038 deg against a truth of
    6.87 deg.  Always pass ``dim=`` by keyword.
    """
    return pp.SO3(so3.tensor()[:, i])


def _fit_core(integ, acc, gyro, dt, rot, vel, gravity):
    """The two closed-form fits, batched.  See the module docstring for both."""
    n = acc.shape[0]
    dev, dtp = acc.device, acc.dtype

    # TIMING.  dt = np.diff(t), so dt[k] spans [t_k, t_k+1] and integrating all H of
    # them advances from index 0 to index H -- but the reference here is rot[-1] and
    # vel[-1], which are index H-1.  Using H-1 intervals makes the integration land
    # exactly on the sample being compared against.  The old form injected
    # omega*dt/T of fake rate (~0.013 deg/s over a 15 s baseline against a ~0.064
    # deg/s bias) and the same inconsistency into the velocity increment.
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


def estimate_biases(integ, acc, gyro, dt, rot, vel, gravity, k_sub=5):
    """Closed-form (b_acc, b_gyro) from one aided interval, PLUS a shrinkage weight.

    THE PROBLEM THE SHRINKAGE SOLVES.  The accel fit is exactly determined -- three
    equations, three unknowns -- so it always "succeeds" and has no residual to
    report.  That is fine when the aircraft is in quasi-steady cruise and the thing
    being estimated really is a constant, and actively harmful when it is not: on
    the 2026_04_07 flights (vertical velocity std 3.5-4.7 m/s against 1.7-2.3
    elsewhere, 74% of the 40 s error vertical) the fit absorbs manoeuvre-driven
    residuals into a "bias" that does not apply over the next 40 s, and 40 s
    position went to 1.44x raw with attitude also worse.

    So make the estimator say how much it trusts itself.  Split the aided interval
    into ``k_sub`` contiguous sub-intervals and fit each one independently.  If the
    quantity is a genuine constant the sub-estimates agree; if it is soaking up
    manoeuvre residuals they scatter.  With ``s2`` the squared standard error of the
    mean across sub-intervals, apply the standard shrinkage weight per axis

        lam = clamp(1 - s2 / b^2, 0, 1)

    which is 1 when the estimate is large against its own scatter and goes to 0 when
    it is not -- so an unreliable window falls back to no correction rather than to
    a confident wrong one.  There is NO tuned threshold here; the only choice is
    ``k_sub``, and the rule itself is the classical variance-based shrinkage.

    Returns ``(b_acc, b_gyro, lam_acc, lam_gyro)``.
    """
    b_acc, b_gyro = _fit_core(integ, acc, gyro, dt, rot, vel, gravity)

    n, H = acc.shape[0], acc.shape[1]
    K = max(int(k_sub), 2)
    m = (H // K) * K
    if m < 2 * K:                       # too short to split; trust nothing
        z = torch.zeros_like(b_acc)
        return b_acc, b_gyro, z, z
    rs = lambda t: t[:, :m].reshape(n * K, m // K, t.shape[-1])
    sa, sg = _fit_core(integ, rs(acc), rs(gyro), rs(dt),
                       pp.SO3(rs(rot.tensor()[:, :m])), rs(vel), gravity)

    def lam(b, sub):
        sub = sub.reshape(n, K, 3)
        s2 = sub.var(dim=1, unbiased=True) / K          # squared standard error
        return (1.0 - s2 / b.pow(2).clamp(min=1e-18)).clamp(0.0, 1.0)

    return b_acc, b_gyro, lam(b_acc, sa), lam(b_gyro, sg)


def predictive_lambda(integ, acc, gyro, dt, rot, vel, gravity):
    """Does this constant PREDICT the near future?  Fit on the first half of the
    aided interval, score on the second half, shrink by out-of-sample skill.

    The dispersion shrinkage above asks whether the sub-estimates agree.  That is a
    proxy.  The question that actually matters at handover is whether a constant
    fitted on the past reduces error on data it did not see -- which is exactly what
    the next 40 s will ask of it.  So split the aided interval in half, fit on the
    FIRST half only, and on the SECOND half compare the velocity-increment residual
    with and without the correction:

        e0 = | int R acc dt - g T - dv |          (no correction)
        e1 = | int R (acc - b) dt - g T - dv |    (with the fitted b)
        lam = clamp(1 - e1^2/e0^2, 0, 1)          per axis

    lam is 1 when the constant explains the held-out increment completely and 0 when
    it explains nothing or makes it worse.  Same construction for the gyro, scored on
    held-out attitude error instead.  The split is forward in time, never backward,
    because that is the temporal relation the deployment actually has.

    No tuned constant anywhere.  This is applied to the FULL-interval estimate, which
    is the better one -- the half-fit exists only to earn the weight.
    """
    H = acc.shape[1]
    h = H // 2
    if h < 4:
        z = torch.zeros(acc.shape[0], 3, dtype=acc.dtype, device=acc.device)
        return z, z
    sl = lambda t, a, b: t[:, a:b]
    ro = lambda a, b: pp.SO3(rot.tensor()[:, a:b])
    b_acc, b_gyro = _fit_core(integ, sl(acc, 0, h), sl(gyro, 0, h), sl(dt, 0, h),
                              ro(0, h), sl(vel, 0, h), gravity)

    # ---- accel: velocity-increment residual on the held-out second half -------
    # TIMING, same as _fit_core: dt[h:H] is H-h intervals landing on index H, but the
    # references here are vel[-1] and rot[-1], which are index H-1.  Use H-1-h
    # intervals so the integration ends exactly on the sample compared against.
    A, D = sl(acc, h, H - 1), sl(dt, h, H - 1)
    R = ro(h, H - 1).matrix()
    T = D.sum(dim=1).squeeze(-1)
    g = torch.zeros(acc.shape[0], 3, dtype=acc.dtype, device=acc.device)
    g[:, 2] = gravity
    dv = vel[:, -1] - vel[:, h]
    base = (torch.einsum("ntij,ntj->nti", R, A) * D).sum(dim=1) - g * T[:, None] - dv
    corr = base - torch.einsum("nij,nj->ni", (R * D[..., None]).sum(dim=1), b_acc)
    lam_a = (1.0 - corr.pow(2) / base.pow(2).clamp(min=1e-18)).clamp(0.0, 1.0)

    # ---- gyro: held-out attitude error, with and without the correction ------
    n = acc.shape[0]
    z3 = torch.zeros(n, 1, 3, dtype=acc.dtype, device=acc.device)
    init = {"pos": z3, "vel": z3, "rot": pp.SO3(rot.tensor()[:, h:h + 1])}
    tgt = pick(ro(h, H), -1)
    err = []
    for b in (torch.zeros_like(b_gyro), b_gyro):
        with torch.no_grad():
            o = pick(integ(init_state=init, dt=D, gyro=sl(gyro, h, H - 1) - b[:, None],
                           acc=A)["rot"], -1)
        err.append((o * tgt.Inv()).Log().abs())
    lam_g = (1.0 - err[1].pow(2) / err[0].pow(2).clamp(min=1e-18)).clamp(0.0, 1.0)
    return lam_a, lam_g


def freeze_biases(acc, gyro, dt, rot, vel, gravity, k_sub=10, integ=None):
    """THE DEPLOYABLE ENTRY POINT.  One aided interval in, frozen constants out.

    Call this once, at the moment aiding is lost, with the LAST 15 SECONDS of the
    aided phase.  Everything it needs was available while GPS was up: the IMU stream,
    the nav filter attitude, and the nav/GPS velocity.  It returns the two constants
    to subtract for the duration of the outage:

        b_acc, b_gyro = freeze_biases(acc, gyro, dt, rot, vel, gravity)
        # then, for the whole unaided window:
        corrected_acc  = acc  - b_acc
        corrected_gyro = gyro - b_gyro

    Shapes are the batched ones the rest of this file uses -- acc/gyro/dt/vel
    ``(n, H, 3)`` (dt ``(n, H, 1)``), rot a ``pp.SO3`` of lshape ``(n, H)``, where n
    is however many handover events you are processing at once (1 in flight).

    THE SETTINGS ARE NOT FREE PARAMETERS.  ``hist`` 15 s and ``k_sub`` 10 were chosen
    on the 11 TRAIN dates alone and then spent once on val and test; see command.txt
    section 13.  Do not re-tune them on new data without repeating that discipline --
    the arm that was tuned on test (``both``, 0.675x there) reads 1.202x on val.

    WHAT THE GATES DO, AND WHY BOTH.  The raw fit is exactly determined, so it always
    "succeeds" and cannot report a bad fit.  Two independent shrinkage weights ask two
    different questions, and the correction is scaled by their product, per axis:

        dispersion (``estimate_biases``)   do sub-interval estimates AGREE?
        predictive (``predictive_lambda``) does a constant fitted on the first half
                                           REDUCE error on the held-out second half?

    Either one alone leaves a failing day above 1.0; the product is what brings the
    worst of 16 independent dates to 1.031.  Both are classical variance-based
    shrinkage -- there is no tuned threshold anywhere in this function.

    Returns ``(b_acc, b_gyro)``, already gated, in the same units as acc and gyro.
    """
    if integ is None:
        integ = pp.module.IMUPreintegrator(reset=True, prop_cov=False,
                                           gravity=gravity).double().to(acc.device)
    b_acc, b_gyro, lam_a, lam_g = estimate_biases(
        integ, acc, gyro, dt, rot, vel, gravity, k_sub=k_sub)
    cvl_a, cvl_g = predictive_lambda(integ, acc, gyro, dt, rot, vel, gravity)
    return b_acc * cvl_a * lam_a, b_gyro * cvl_g * lam_g


def track_frame(vel0):
    h = vel0[:, :2]
    spd = h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    a = h / spd
    return a, torch.stack([-a[:, 1], a[:, 0]], dim=-1), spd.squeeze(-1)


def main():
    global ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--win", type=int, default=4000, help="unaided window, frames")
    ap.add_argument("--hist", type=float, default=15.0,
                    help="aided fit length T, seconds.  DEFAULT IS THE VALIDATED VALUE. "
                         "Chosen on the 11 train dates and then spent once on val and "
                         "test (command.txt section 13); shorter is better here because "
                         "the gyro bias drifts, and the pooled ratio degrades "
                         "monotonically with T: 15 s 0.755x, 30 s 0.786x, 60 s 0.846x, "
                         "120 s 0.940x.  120 s was the old default and gives away most "
                         "of the result, so the good setting is the one you get for free.")
    ap.add_argument("--hist_acc", type=float, default=None,
                    help="override --hist for the ACCEL fit only.  The two biases have "
                         "opposite time constants and there is no reason they should "
                         "share a baseline: measured on this corpus the acc arm is flat "
                         "at ~0.80x for every T from 5 s to 120 s (the accel bias is "
                         "stable, so a LONG baseline just averages down the noise), while "
                         "the gyro arm decays 0.700x -> 1.030x as T grows (the gyro bias "
                         "drifts, so only a RECENT estimate transfers).  One shared T has "
                         "to compromise between them; these two flags remove the "
                         "compromise.  Choose them on TRAIN dates only.")
    ap.add_argument("--hist_gyro", type=float, default=None,
                    help="override --hist for the GYRO fit only.  See --hist_acc.")
    ap.add_argument("--nwin", type=int, default=40)
    ap.add_argument("--start_s", type=float, default=None,
                    help="offset of the FIRST window from the start of the flight, seconds. "
                         "Defaults to --hist. Pin it when sweeping --hist, or each setting "
                         "scores a different set of windows and the ratios are not comparable.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--init", default="gt", choices=["gt", "mti"],
                    help="attitude handed over at t0: the nav filter (gt) or the "
                         "independent MTI solution (mti)")
    ap.add_argument("--mti_yaw_ref", default="wind_causal")
    ap.add_argument("--k_sub", type=int, default=10,
                    help="sub-intervals used to measure how stable the fitted bias is. "
                         "DEFAULT IS THE VALIDATED VALUE (worst train date 1.031 at 10, "
                         "1.032 at 8, 1.043 at 5, 1.057 at 3).  Only affects the "
                         "dispersion gate, so it moves `shrunk` and `cvs` and not `cv`.")
    ap.add_argument("--multi_hist", default=None,
                    help="comma-separated baselines in SECONDS for the `multi` and "
                         "`multiw` arms, e.g. 5,15,60.  Defaults to just --hist, in "
                         "which case `multiw` is identical to `cv` by construction -- "
                         "which is a useful self-check that the combination is wired "
                         "up correctly.")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of %s to integrate.  'raw' is always "
                         "kept -- it is the baseline every ratio is taken against.  Each "
                         "arm costs a full W-step preintegration, and that Python loop is "
                         "most of the runtime, so a sweep should ask for only the arms it "
                         "is actually choosing between." % (",".join(ARMS),))
    ap.add_argument("--out", default=None,
                    help="write one CSV row per flight x arm, for offline aggregation")
    ap.add_argument("--by_date", action="store_true",
                    help="also print the per-DATE table.  THE DATE IS THE UNIT OF "
                         "INDEPENDENCE on this corpus -- flights from one day share a "
                         "power-on, an airframe state and a weather regime, so a pooled "
                         "number is dominated by whichever day contributed most windows. "
                         "Every selection decision must be read here, not from POOLED.")
    a = ap.parse_args()
    dev = a.device

    if a.arms:
        want = [x.strip() for x in a.arms.split(",") if x.strip()]
        bad = [x for x in want if x not in ARMS]
        if bad:
            sys.exit("unknown arm(s) %s; choose from %s" % (bad, list(ARMS)))
        ARMS = tuple(["raw"] + [x for x in ARMS if x in want and x != "raw"])

    agg = {k: [] for k in ARMS}
    recs = []
    for f in a.files:
        seq = UAV(a.data_root, f, trim_to_airborne=True, mti_yaw_ref=a.mti_yaw_ref,
                  mti_diagnostics=False)
        d = seq.data
        W = a.win
        fs = 1.0 / float(np.median(np.diff(d["time"].numpy())))
        H_acc = int((a.hist_acc if a.hist_acc is not None else a.hist) * fs)
        H_gyro = int((a.hist_gyro if a.hist_gyro is not None else a.hist) * fs)
        multi_H = [int(x * fs) for x in
                   ([float(y) for y in a.multi_hist.split(",")] if a.multi_hist
                    else [a.hist])]
        # every window needs the LONGER of the two baselines in front of it, so that
        # both arms score the identical set of windows.  Without this the arm with the
        # shorter baseline would silently get extra (and easier, earlier) windows and
        # the ratios would not be comparable.
        H = max([H_acc, H_gyro] + multi_H)
        S = H if a.start_s is None else int(a.start_s * fs)
        if S < H:
            sys.exit("--start_s %.0f is less than --hist %.0f: the first window would "
                     "need history from before the recording" % (a.start_s, a.hist))
        n = d["time"].shape[0]
        # every window needs H frames of aided history in front of it
        starts = [s for s in range(S, n - W - 1, W)][:a.nwin]
        if not starts:
            print("%s: too short for %d frames of history + a %d frame window" % (f, H, W))
            continue

        rk = "mti_orientation" if (a.init == "mti" and "mti_orientation" in d) else "gt_orientation"
        if a.init == "mti" and rk != "mti_orientation":
            sys.exit("--init mti requested but this loader published no mti_orientation")
        st = lambda k, lo, hi: torch.stack([d[k][s + lo:s + hi] for s in starts]).to(dev)
        so3 = lambda k, lo, hi: pp.SO3(torch.stack(
            [d[k][s + lo:s + hi].tensor() for s in starts])).to(dev)

        integ = pp.module.IMUPreintegrator(reset=True, prop_cov=False,
                                           gravity=seq.gravity).double().to(dev)
        # --- fit on the aided interval [s-H, s), nav attitude and velocity ---
        def fit_at(Hx):
            """Every estimator this file has, fitted on the last Hx frames before t0."""
            args = (st("acc", -Hx, 0), st("gyro", -Hx, 0), st("dt", -Hx, 0),
                    so3("gt_orientation", -Hx, 0), st("velocity", -Hx, 0), seq.gravity)
            ba, bg, la, lg = estimate_biases(integ, *args, k_sub=a.k_sub)
            ca, cg = predictive_lambda(integ, *args)
            return ba, bg, la, lg, ca, cg

        cache = {}
        def fit_cached(Hx):
            if Hx not in cache:
                cache[Hx] = fit_at(Hx)
            return cache[Hx]

        if H_acc == H_gyro:
            b_acc, b_gyro, lam_a, lam_g, cvl_a, cvl_g = fit_cached(H_acc)
        else:
            # two independent fits; each channel keeps only its own baseline's answer
            b_acc, _, lam_a, _, cvl_a, _ = fit_cached(H_acc)
            _, b_gyro, _, lam_g, _, cvl_g = fit_cached(H_gyro)

        # ---- multi-baseline combination -------------------------------------
        # No single T can be right for both channels: the accel bias is stable (a
        # long baseline averages its noise down) while the gyro bias drifts (only a
        # recent one transfers).  Rather than pick one, fit at several and let each
        # estimate's OWN measured out-of-sample skill decide how much it counts.
        #
        #   multi   per axis, take the baseline with the highest predictive lambda
        #           and shrink it by that lambda.  Selection, not averaging.
        #   multiw  per axis, lambda-weighted combination
        #               b = sum_j lam_j^2 b_j / sum_j lam_j
        #           which for a SINGLE baseline collapses to lam*b -- exactly the
        #           `cv` arm.  So this is a strict generalisation of `cv` and
        #           introduces no new tuned constant: the weights are the measured
        #           held-out skill of each estimate, nothing else.
        # The weight is the PRODUCT of both gates, matching the `cvs` arm -- the best
        # single-baseline arm on the train dates -- so that with one baseline these
        # arms reduce to `cvs` exactly.  fit_at returns (ba, bg, lam_a, lam_g, cvl_a,
        # cvl_g), so the accel weight is index 4 x index 2 and the gyro 5 x 3.
        mb = [fit_cached(Hx) for Hx in multi_H]
        def _wt(m, ch):
            return m[4 + ch] * m[2 + ch]
        def _sel(ch):
            B = torch.stack([m[ch] for m in mb])             # (J, n, 3)
            L = torch.stack([_wt(m, ch) for m in mb])        # (J, n, 3)
            j = L.argmax(dim=0, keepdim=True)
            return (B.gather(0, j) * L.gather(0, j)).squeeze(0)
        def _wavg(ch):
            B = torch.stack([m[ch] for m in mb])
            L = torch.stack([_wt(m, ch) for m in mb])
            den = L.sum(dim=0)
            return torch.where(den > 0, (L.pow(2) * B).sum(dim=0) / den.clamp(min=1e-12),
                               torch.zeros_like(den))
        m_acc, m_gyro = _sel(0), _sel(1)
        w_acc, w_gyro = _wavg(0), _wavg(1)

        # --- coast the unaided window [s, s+W) with the frozen constants -----
        dt = st("dt", 0, W)
        acc, gyro = st("acc", 0, W), st("gyro", 0, W)
        vel0 = torch.stack([d["velocity"][s] for s in starts]).to(dev)
        init = {"pos": torch.stack([d["gt_translation"][s] for s in starts])[:, None].to(dev),
                "vel": vel0[:, None],
                "rot": pp.SO3(torch.stack([d[rk][s].tensor() for s in starts])[:, None]).to(dev)}
        gt_p = torch.stack([d["gt_translation"][s + W] for s in starts]).to(dev)
        gt_r = pp.SO3(torch.stack(
            [d["gt_orientation"][s + W].tensor() for s in starts])).to(dev)

        ah, ch, spd = track_frame(vel0)
        res = {}
        for arm in ARMS:
            if arm == "shrunk":
                ba, bg = b_acc * lam_a, b_gyro * lam_g
            elif arm == "cv":
                ba, bg = b_acc * cvl_a, b_gyro * cvl_g
            elif arm == "multi":
                ba, bg = m_acc, m_gyro
            elif arm == "multiw":
                ba, bg = w_acc, w_gyro
            elif arm == "cvs":
                # both gates, multiplied.  They ask different questions -- "do the
                # sub-estimates agree?" and "does it predict held-out data?" -- and
                # a weight of 1 requires passing both.  Still no tuned constant.
                ba, bg = b_acc * cvl_a * lam_a, b_gyro * cvl_g * lam_g
            else:
                ba = b_acc if arm in ("acc", "both") else torch.zeros_like(b_acc)
                bg = b_gyro if arm in ("gyro", "both") else torch.zeros_like(b_gyro)
            with torch.no_grad():
                out = integ(init_state=init, dt=dt,
                            gyro=gyro - bg[:, None], acc=acc - ba[:, None])
            err = out["pos"][:, -1] - gt_p
            rot_e = torch.rad2deg((pick(out["rot"], -1) * gt_r.Inv()).Log().norm(dim=-1))
            res[arm] = (float(err.norm(dim=-1).mean()),
                        float(err.norm(dim=-1).pow(2).mean().sqrt()),
                        float(rot_e.mean()),
                        float((err[:, :2] * ah).sum(-1).pow(2).mean().sqrt()),
                        float((err[:, :2] * ch).sum(-1).pow(2).mean().sqrt()))
            agg[arm].append((len(starts),) + res[arm])
            recs.append({"flight": f, "date": f[:10], "nwin": len(starts), "arm": arm,
                         "mean": res[arm][0], "rms": res[arm][1], "rot": res[arm][2],
                         "along": res[arm][3], "cross": res[arm][4]})

        # report whichever corrected arm was actually asked for, not a hardcoded one
        show = ARMS[-1] if len(ARMS) > 1 else "raw"
        print("%-34s %2d win | b_acc %6.3f %6.3f %6.3f | b_gyro %7.4f %7.4f %7.4f deg/s | "
              "cvlam %.2f/%.2f | raw %6.1f -> %s %6.1f m"
              % (f[:34], len(starts),
                 *b_acc.mean(0).tolist(), *torch.rad2deg(b_gyro).mean(0).tolist(),
                 float(cvl_a.mean()), float(cvl_g.mean()),
                 res["raw"][0], show, res[show][0]))

    if not agg["raw"]:
        return
    w = np.array([x[0] for x in agg["raw"]], dtype=float)
    print("\n" + "=" * 86)
    print("POOLED %d flights, %d windows | %.0f s window, %.0f s aided fit, init=%s"
          % (len(w), int(w.sum()), a.win / 100.0, a.hist, a.init))
    if (a.hist_acc is not None) or (a.hist_gyro is not None):
        print("       decoupled baselines: acc %.0f s, gyro %.0f s"
              % (a.hist_acc if a.hist_acc is not None else a.hist,
                 a.hist_gyro if a.hist_gyro is not None else a.hist))
    print("=" * 86)
    print("  %-6s %9s %9s %9s %9s %9s %9s"
          % ("arm", "mean m", "rms m", "rot deg", "along m", "cross m", "vs raw"))
    print("  " + "-" * 76)
    base = None
    for arm in ARMS:
        v = np.array([x[1:] for x in agg[arm]], dtype=float)
        mn = float(np.average(v[:, 0], weights=w))
        rm = float(np.sqrt(np.average(v[:, 1] ** 2, weights=w)))
        rt = float(np.average(v[:, 2], weights=w))
        al = float(np.sqrt(np.average(v[:, 3] ** 2, weights=w)))
        cr = float(np.sqrt(np.average(v[:, 4] ** 2, weights=w)))
        base = mn if base is None else base
        print("  %-6s %9.1f %9.1f %9.2f %9.1f %9.1f %8.3fx"
              % (arm, mn, rm, rt, al, cr, mn / base))
    print("\n  'rot deg' is the guard on the counter-drift trap: an arm that buys")
    print("  position while pushing rotation UP has broken attitude, not fixed bias.")

    if a.by_date:
        dates = sorted({r["date"] for r in recs})
        print("\n" + "=" * 86)
        print("PER DATE -- ratio vs raw.  THE DATE IS THE UNIT OF INDEPENDENCE.")
        print("=" * 86)
        print("  %-12s %4s %5s %9s %s"
              % ("date", "fl", "win", "raw m", " ".join("%7s" % x for x in ARMS[1:])))
        print("  " + "-" * 80)
        worst = {k: 0.0 for k in ARMS[1:]}
        for d in dates:
            sel = lambda arm: [r for r in recs if r["date"] == d and r["arm"] == arm]
            rw = sel("raw")
            ww = np.array([r["nwin"] for r in rw], dtype=float)
            base = float(np.average([r["mean"] for r in rw], weights=ww))
            cells = []
            for arm in ARMS[1:]:
                v = float(np.average([r["mean"] for r in sel(arm)], weights=ww))
                ratio = v / base
                worst[arm] = max(worst[arm], ratio)
                cells.append("%7.3f" % ratio)
            print("  %-12s %4d %5d %9.1f %s"
                  % (d, len(rw), int(ww.sum()), base, " ".join(cells)))
        print("  " + "-" * 80)
        print("  %-12s %4s %5s %9s %s"
              % ("WORST DATE", "", "", "", " ".join("%7.3f" % worst[k] for k in ARMS[1:])))
        print("\n  An arm is only shippable if its WORST DATE row is at or below 1.0:")
        print("  a mean below 1.0 that hides a day above it will fail in the field on")
        print("  exactly the day that day resembles.")

    if a.out:
        import csv as _csv
        with open(a.out, "w", newline="") as fh:
            w_ = _csv.DictWriter(fh, fieldnames=["flight", "date", "nwin", "arm",
                                                 "mean", "rms", "rot", "along", "cross"])
            w_.writeheader()
            for r in recs:
                w_.writerow(r)
        print("\nwrote %s (%d rows)" % (a.out, len(recs)))


if __name__ == "__main__":
    main()
