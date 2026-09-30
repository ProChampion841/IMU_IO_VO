"""Can the pitot tube replace integrated velocity during an outage?

THE IDEA.  Every correction in this repo so far attacks an INTEGRATION -- the accel
bias that integrates into velocity, the gyro drift that integrates into attitude.  All
of them lose to time, because the quantity being corrected grows.  Air data does not
integrate anything:

    v_ground(t) = k R(t) e_x V_air(t) + wind

The aircraft flies roughly where it points (sideslip ~ 0 for a fixed-wing in
coordinated flight), so with attitude the airspeed SCALAR becomes a velocity VECTOR.
Scale `k` and `wind` are both observable during the aided phase, where GPS velocity is
live, and both are slowly varying -- wind over minutes, a pitot scale factor over a
flight.  Freeze them at handover and the resulting velocity estimate has a roughly
CONSTANT error for the whole outage instead of a growing one.

"AIRSPEED IS A LIAR" IS NOT A REASON TO SKIP THIS.  The pitot on this platform is
biased -- one log reads a median 14.9 m/s for 89% of a flight during which the
aircraft is provably parked -- which is exactly why the loader judges motion by GPS
ground speed instead.  But a biased sensor that can be CALIBRATED is a different thing
from a useless one, and the aided phase is a calibration opportunity that costs
nothing.  This script measures what survives calibration.

WHAT IS FITTED, ON THE AIDED INTERVAL ONLY.  With u(t) = R(t) e_x V_air(t) known,

    v_gps = k u + W          -- linear in (k, W_x, W_y, W_z), 4 unknowns

so it is one least-squares solve over ~1500 samples.  Nothing inside the outage is
used to fit anything, exactly as in tools/prewindow_align.py.

WHAT IS REPORTED.  Over the outage, the velocity error of
  * the air-data estimate with the frozen constants, and
  * plain IMU integration from the same handover state,
both against GPS ground velocity as truth.  Two attitude sources are scored: GT
attitude isolates the air-data quality itself, propagated attitude is the honest
deployable number.  If the air-data error is flat while the IMU error grows, the two
curves cross and everything after the crossing is free accuracy.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import pypose as pp
import torch

from datasets.UAVdataset import UAV

E_X = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)


def fit_airdata(R, va, v_gps):
    """Least squares for (k, W) on the aided interval.  R (n,H,3,3), va (n,H), v_gps (n,H,3).

    Returns k (n,), W (n,3), and the in-sample residual rms (n,) for diagnostics.
    """
    n, H = va.shape
    ex = E_X.to(R.device).expand(n, H, 3)
    u = torch.einsum("nhij,nhj->nhi", R, ex) * va[..., None]        # (n,H,3)
    # design: [u | I3] stacked over samples -> 3H equations, 4 unknowns
    A = torch.zeros(n, 3 * H, 4, dtype=R.dtype, device=R.device)
    A[:, 0::3, 0] = u[..., 0]; A[:, 1::3, 0] = u[..., 1]; A[:, 2::3, 0] = u[..., 2]
    A[:, 0::3, 1] = 1.0;       A[:, 1::3, 2] = 1.0;       A[:, 2::3, 3] = 1.0
    y = v_gps.reshape(n, 3 * H)
    x = torch.linalg.lstsq(A, y.unsqueeze(-1)).solution.squeeze(-1)  # (n,4)
    res = (y - torch.einsum("nmp,np->nm", A, x)).reshape(n, H, 3)
    return x[:, 0], x[:, 1:4], res.norm(dim=-1).pow(2).mean(dim=1).sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--win", type=int, default=4000)
    ap.add_argument("--hist", type=float, default=15.0)
    ap.add_argument("--nwin", type=int, default=40)
    ap.add_argument("--start_s", type=float, default=120.0)
    ap.add_argument("--marks", type=int, nargs="+", default=[500, 1000, 2000, 4000])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    acc_ad_gt, acc_ad_prop, acc_imu, nw, nf = [], [], [], 0, 0
    ks, resids = [], []
    for f in a.files:
        try:
            seq = UAV(a.data_root, f, trim_to_airborne=True, mti_yaw_ref="fixed",
                      mti_diagnostics=False)
        except Exception:
            continue
        d = seq.data
        if "airspeed" not in d:
            print("  %s: loader publishes no airspeed" % f[:34])
            continue
        t = d["time"].numpy()
        fs = 1.0 / float(np.median(np.diff(t)))
        H, W = int(a.hist * fs), a.win
        S = max(int(a.start_s * fs), H)
        starts = [s for s in range(S, t.shape[0] - W - 1, W)][:a.nwin]
        if not starts:
            continue
        dev = a.device
        st = lambda k, lo, hi: torch.stack([d[k][s + lo:s + hi]
                                            for s in starts]).to(dev).double()
        so = lambda k, lo, hi: pp.SO3(torch.stack([d[k][s + lo:s + hi].tensor()
                                                   for s in starts])).to(dev).double()
        # ---- calibrate on the aided interval -------------------------------
        k, Wd, rr = fit_airdata(so("gt_orientation", -H, 0).matrix(),
                                st("airspeed", -H, 0).squeeze(-1),
                                st("velocity", -H, 0))
        ks.append(k.cpu().numpy()); resids.append(rr.cpu().numpy())

        # ---- score across the outage ---------------------------------------
        va = st("airspeed", 0, W).squeeze(-1)
        gt_v = st("velocity", 0, W)
        R_gt = so("gt_orientation", 0, W).matrix()
        n = len(starts)
        integ = pp.module.IMUPreintegrator(reset=True, prop_cov=False,
                                           gravity=seq.gravity).double().to(dev)
        init = {"pos": torch.zeros(n, 1, 3, dtype=torch.float64, device=dev),
                "vel": st("velocity", 0, 1),
                "rot": pp.SO3(so("gt_orientation", 0, 1).tensor())}
        with torch.no_grad():
            o = integ(init_state=init, dt=st("dt", 0, W), gyro=st("gyro", 0, W),
                      acc=st("acc", 0, W))
        R_prop = o["rot"].matrix()
        ex = E_X.to(dev).expand(n, W, 3)

        def ad_err(R):
            u = torch.einsum("nhij,nhj->nhi", R, ex) * va[..., None]
            v = k[:, None, None] * u + Wd[:, None, :]
            return (v - gt_v).norm(dim=-1)

        idx = [min(m, W - 1) for m in a.marks]
        acc_ad_gt.append(ad_err(R_gt)[:, idx].cpu().numpy())
        acc_ad_prop.append(ad_err(R_prop)[:, idx].cpu().numpy())
        acc_imu.append((o["vel"] - gt_v).norm(dim=-1)[:, idx].cpu().numpy())
        nw += n; nf += 1

    if not acc_imu:
        print("no usable flights")
        return
    G = np.concatenate(acc_ad_gt); P = np.concatenate(acc_ad_prop)
    I = np.concatenate(acc_imu)
    print("\n" + "=" * 78)
    print("VELOCITY ERROR vs ELAPSED -- %d windows, %d flights, %.0f s calibration"
          % (len(I), nf, a.hist))
    print("=" * 78)
    print("  %-10s %14s %14s %14s" % ("elapsed", "IMU integ", "air data (GT R)",
                                      "air data (prop R)"))
    print("  " + "-" * 58)
    for j, m in enumerate(a.marks):
        r = lambda X: float(np.sqrt((X[:, j] ** 2).mean()))
        print("  %-10s %14.3f %14.3f %14.3f"
              % ("%.1fs" % (m / 100.0), r(I), r(G), r(P)))
    kk = np.concatenate(ks); rrr = np.concatenate(resids)
    print("\n  fitted pitot scale k: median %.4f  (1.0 = perfectly calibrated)"
          % float(np.median(kk)))
    print("  in-sample fit residual: median %.3f m/s" % float(np.median(rrr)))
    print("\n  All figures are m/s rms against GPS ground velocity.  The air-data")
    print("  columns should be roughly FLAT with elapsed time -- nothing is being")
    print("  integrated -- while the IMU column grows.  Where they cross, everything")
    print("  after is free accuracy.")


if __name__ == "__main__":
    main()
