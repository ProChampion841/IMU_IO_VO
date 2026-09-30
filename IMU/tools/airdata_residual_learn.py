"""Is the air-data model's residual PREDICTABLE from the IMU?  Ridge, held out by day.

WHY THIS IS A DIFFERENT QUESTION FROM EVERY EARLIER LEARNING ATTEMPT.  The v9/v10/v11
arms asked a network to predict a correction to acc/gyro, supervised through a 40-300 s
integration.  That target is not observable inside a window (the bias sits 4-15x below
the window's own kinematic floor until ~300 s, see tools/bias_observability.py) and it
gives ONE training example per window, so 7.3 h of flight collapses to 88 independent
samples against ~500k parameters.  Both properties are fatal and neither is fixable by
architecture.

The air-data velocity model has an error term with the OPPOSITE properties:

    v_ground = k R (V_a, 0, 0) + wind

assumes zero sideslip AND zero angle of attack.  Writing the air velocity properly,

    v_air_body = V_a (cos a cos b, sin b, sin a)  ~  V_a (1, b, a)   for small angles

so the residual r_body = R^T (v_gps - v_airdata) has y and z components that ARE the
sideslip and angle of attack, scaled by airspeed.  Those are:

  * OBSERVABLE FROM THE IMU BY FIRST-ORDER PHYSICS.  Lateral specific force IS the
    sideslip force (b ~ a_y m / Y_b) and normal specific force IS lift (a ~ a_z / V_a^2
    up to constants).  This is not a signal hidden under the noise floor -- it is the
    dominant term in the measurement.
  * SUPERVISED POINTWISE.  GPS velocity gives the truth at EVERY SAMPLE of every aided
    flight, not once per outage window.  Millions of examples, not 88.
  * LOW DIMENSIONAL.  Two numbers per sample.

So this script fits the cheapest possible model -- ridge regression on IMU features --
and scores it on flights from DAYS THE FIT NEVER SAW.  Ridge first on purpose: if a
linear model on physically motivated features already explains a useful fraction, a
small network is worth building and will do better.  If ridge explains nothing, the
relationship is not there and no network will find it either.

WHAT IS REPORTED.  Fraction of residual variance explained on held-out days, per axis,
plus what that implies for the air-data velocity error -- which is the number that
actually matters, because the air arm's velocity error is what limits the whole
long-outage result.

NOT A LEAK.  Features are IMU-only (acc, gyro, airspeed).  The TARGET uses GPS
velocity, which is exactly right: this model would be FITTED OFFLINE on past flights
and then applied during an outage, where it consumes only IMU and airspeed.
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
from long_outage import fit_airdata


def features(acc, gyro, va):
    """IMU-only features, chosen from the aerodynamics rather than by search.

    b (sideslip) responds to LATERAL specific force and to yaw/roll rate; a (angle of
    attack) responds to NORMAL specific force, and both scale with dynamic pressure, so
    the 1/V_a and 1/V_a^2 terms carry the actual physical dependence.  Airspeed itself
    and a bias column complete it.  All columns are IMU or pitot -- nothing here is
    unavailable during an outage.
    """
    v = va.clamp(min=5.0)
    cols = [acc[..., 0], acc[..., 1], acc[..., 2],
            gyro[..., 0], gyro[..., 1], gyro[..., 2],
            acc[..., 1] / v, acc[..., 2] / v,
            acc[..., 1] / v.pow(2), acc[..., 2] / v.pow(2),
            gyro[..., 2] * v, gyro[..., 0] * v,
            v, torch.ones_like(v)]
    return torch.stack(cols, dim=-1)


def collect(files, root, hist_s, win, nwin, start_s, dev, stride=10):
    """Per-sample (features, residual) pairs over outage windows, grouped by day."""
    out = {}
    for f in files:
        try:
            seq = UAV(root, f, trim_to_airborne=True, mti_yaw_ref="fixed",
                      mti_diagnostics=False)
        except Exception:
            continue
        d = seq.data
        if "airspeed" not in d:
            continue
        t = d["time"].numpy()
        fs = 1.0 / float(np.median(np.diff(t)))
        H, W = int(hist_s * fs), win
        S = max(int(start_s * fs), H)
        starts = [s for s in range(S, t.shape[0] - W - 1, W)][:nwin]
        if not starts:
            continue
        st = lambda k, lo, hi: torch.stack([d[k][s + lo:s + hi]
                                            for s in starts]).to(dev).double()
        so = lambda k, lo, hi: pp.SO3(torch.stack([d[k][s + lo:s + hi].tensor()
                                                   for s in starts])).to(dev).double()
        # calibrate exactly as the deployed arm does: aided interval only
        k, Wd, _ = fit_airdata(so("gt_orientation", -H, 0).matrix(),
                               st("airspeed", -H, 0).squeeze(-1),
                               st("velocity", -H, 0))
        R = so("gt_orientation", 0, W).matrix()
        va = st("airspeed", 0, W).squeeze(-1)
        acc, gyro = st("acc", 0, W), st("gyro", 0, W)
        gt_v = st("velocity", 0, W)
        n = len(starts)
        ex = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64,
                          device=dev).expand(n, W, 3)
        v_ad = k[:, None, None] * torch.einsum("nhij,nhj->nhi", R, ex) * va[..., None] \
            + Wd[:, None, :]
        # residual expressed in the BODY frame -> its y,z are sideslip and AoA terms
        r_body = torch.einsum("nhji,nhj->nhi", R, gt_v - v_ad)
        X = features(acc, gyro, va)
        sl = slice(None, None, stride)
        day = f[:10]
        a, b = out.get(day, ([], []))
        a.append(X[:, sl].reshape(-1, X.shape[-1]).cpu())
        b.append(r_body[:, sl].reshape(-1, 3).cpu())
        out[day] = (a, b)
    return {d: (torch.cat(a), torch.cat(b)) for d, (a, b) in out.items()}


def ridge(X, Y, lam):
    Xm, Xs = X.mean(0), X.std(0).clamp(min=1e-8)
    Z = (X - Xm) / Xs
    A = Z.T @ Z + lam * torch.eye(Z.shape[1], dtype=Z.dtype)
    B = Z.T @ (Y - Y.mean(0))
    return torch.linalg.solve(A, B), Xm, Xs, Y.mean(0)


def apply_ridge(X, model):
    Wm, Xm, Xs, Ym = model
    return ((X - Xm) / Xs) @ Wm + Ym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--hist", type=float, default=15.0)
    ap.add_argument("--win", type=int, default=30000)
    ap.add_argument("--nwin", type=int, default=12)
    ap.add_argument("--start_s", type=float, default=120.0)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    print("collecting train ...")
    tr = collect(a.train, a.data_root, a.hist, a.win, a.nwin, a.start_s,
                 a.device, a.stride)
    print("collecting held-out ...")
    te = collect(a.test, a.data_root, a.hist, a.win, a.nwin, a.start_s,
                 a.device, a.stride)
    if not tr or not te:
        print("no data")
        return
    Xtr = torch.cat([v[0] for v in tr.values()])
    Ytr = torch.cat([v[1] for v in tr.values()])
    print("train samples %d from %d days | held-out %d from %d days"
          % (len(Xtr), len(tr), sum(len(v[0]) for v in te.values()), len(te)))

    model = ridge(Xtr, Ytr, a.lam)
    print("\n" + "=" * 74)
    print("HELD-OUT VARIANCE EXPLAINED in the air-data residual (body frame)")
    print("=" * 74)
    print("  %-14s %8s %10s %10s %10s"
          % ("day", "samples", "fwd", "lateral", "vertical"))
    print("  " + "-" * 56)
    allX, allY = [], []
    for day in sorted(te):
        X, Y = te[day]
        P = apply_ridge(X, model)
        ss_res = (Y - P).pow(2).mean(0)
        ss_tot = (Y - Y.mean(0)).pow(2).mean(0)
        r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-12)
        print("  %-14s %8d %10.3f %10.3f %10.3f"
              % (day, len(X), *[float(x) for x in r2]))
        allX.append(X); allY.append(Y)
    X, Y = torch.cat(allX), torch.cat(allY)
    P = apply_ridge(X, model)
    r2 = 1.0 - (Y - P).pow(2).mean(0) / (Y - Y.mean(0)).pow(2).mean(0).clamp(min=1e-12)
    print("  " + "-" * 56)
    print("  %-14s %8d %10.3f %10.3f %10.3f"
          % ("POOLED", len(X), *[float(x) for x in r2]))

    e0 = Y.norm(dim=-1)
    e1 = (Y - P).norm(dim=-1)
    print("\n  air-data velocity error  before %.3f m/s   after %.3f m/s   %.3fx"
          % (float(e0.pow(2).mean().sqrt()), float(e1.pow(2).mean().sqrt()),
             float(e1.pow(2).mean().sqrt() / e0.pow(2).mean().sqrt())))
    print("\n  R2 is on HELD-OUT DAYS, so it is transfer and not fit quality.  The")
    print("  lateral and vertical columns are the sideslip and angle-of-attack terms;")
    print("  those are the ones the coordinated-flight assumption gets wrong, and the")
    print("  ones worth learning.  A useful R2 here means a small network is worth")
    print("  building; near zero means the relationship is not in these features.")


if __name__ == "__main__":
    main()
