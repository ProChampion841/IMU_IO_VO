"""Prove the UAV log -> AirIMU/pypose conversion is correct, before training.

Run:  python tools/verify_uav_conversion.py --data_root data [--files a.csv b.csv]

The decisive test (T4) feeds the converted acc/gyro/dt/gt_rot straight into the
same ``pypose.module.IMUPreintegrator`` that AirIMU trains against, using
ground-truth rotation for gravity compensation (``gtrot=True``, as the configs
do), and checks that the integrated velocity tracks ground-truth velocity.

Why this catches the failure modes that matter:

* a **wrong gravity sign** injects ~2 g = 19.6 m/s^2 of bogus vertical
  acceleration, which over a 2 s window produces ~39 m/s of velocity error --
  three orders of magnitude above the pass threshold.
* a **wrong handedness** (using a reflection instead of a rotation to change
  frames) mirrors the trajectory, so the horizontal velocity error grows to the
  same order as the speed itself (~20 m/s).
* a **wrong gyro scale** (rad/s vs deg/s, a factor of 57) destroys the rotation
  channel, caught by T3.

Each check prints PASS/FAIL and the script exits non-zero if any check fails.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import pypose as pp
import torch

from datasets.UAVdataset import UAV

G = 9.81007
PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, ok, detail):
    _results.append(bool(ok))
    print("  [%s] %-46s %s" % (PASS if ok else FAIL, name, detail))


def verify(path):
    print("\n" + "=" * 78)
    seq = UAV(os.path.dirname(path) or ".", os.path.basename(path), trim_to_airborne=True)
    d = seq.data
    n = d["time"].shape[0]
    print("-" * 78)

    # ---- T1: shape / dtype contract ---------------------------------------
    ok = (
        d["dt"].shape == (n - 1, 1)
        and d["acc"].shape == (n, 3)
        and d["gyro"].shape == (n, 3)
        and d["gt_translation"].shape == (n, 3)
        and d["velocity"].shape == (n, 3)
        and d["gt_orientation"].shape[0] == n
        and d["mask"].dtype == torch.bool
        and d["acc"].dtype == torch.float64
    )
    check("T1 shapes/dtypes match the loader contract", ok,
          "N=%d dt%s acc%s rot%s mask=%s" % (n, tuple(d["dt"].shape), tuple(d["acc"].shape),
                                             tuple(d["gt_orientation"].shape), d["mask"].dtype))

    # ---- T2: gravity sign -- acc must read +1 g on z when level -----------
    # Rotate the measured specific force into the world frame; at rest it must
    # equal +g*z_up.  Averaged over the flight the non-gravitational part
    # largely cancels, so <R@acc> should sit near [0,0,+9.81].
    R = d["gt_orientation"]
    a_world = (R @ d["acc"]).numpy()
    mz = float(np.mean(a_world[:, 2]))
    check("T2 mean world-frame acc z is +g (not -g)", 8.0 < mz < 11.5,
          "mean(R@acc)_z = %+.3f m/s^2 (expect ~+%.2f; ~%.2f would mean a sign error)" % (mz, G, -G))

    # ---- T3: gyro scale via rotation integration --------------------------
    # Integrate the gyro over 1 s windows with pypose and compare the resulting
    # orientation increment against the ground-truth increment.
    dt = d["dt"]
    W, NW = 100, 40
    step = max(1, (n - W - 1) // NW)
    rot_err = []
    for s in range(0, min(n - W - 1, step * NW), step):
        e = s + W
        inc_pred = pp.so3(d["gyro"][s:e] * dt[s:e]).Exp()
        acc_rot = pp.identity_SO3(dtype=torch.float64)
        for k in range(W):
            acc_rot = acc_rot * inc_pred[k]
        inc_gt = R[s].Inv() * R[e]
        rot_err.append(float((acc_rot.Inv() * inc_gt).Log().norm()))
    rot_err = np.array(rot_err)
    med = float(np.median(np.degrees(rot_err)))
    check("T3 gyro integrates to GT rotation over 1 s", med < 5.0,
          "median 1 s orientation error = %.3f deg (a deg/s<->rad/s mix-up gives >>57x this)" % med)

    # ---- T4: THE decisive test -- full pypose preintegration --------------
    # Exactly what AirIMU does: gtrot=True, GT initial state, 2 s windows.
    integ = pp.module.IMUPreintegrator(reset=True, prop_cov=False, gravity=G).double()
    W = 200
    step = max(1, (n - W - 1) // NW)
    verr, perr = [], []
    for s in range(0, min(n - W - 1, step * NW), step):
        e = s + W
        init = {"pos": d["gt_translation"][s][None, None],
                "rot": d["gt_orientation"][s][None, None],
                "vel": d["velocity"][s][None, None]}
        out = integ(init_state=init,
                    dt=dt[s:e][None], gyro=d["gyro"][s:e][None],
                    acc=d["acc"][s:e][None], rot=d["gt_orientation"][s:e][None])
        verr.append(float((out["vel"][0, -1] - d["velocity"][e]).norm()))
        perr.append(float((out["pos"][0, -1] - d["gt_translation"][e]).norm()))
    verr, perr = np.array(verr), np.array(perr)
    mv, mp = float(np.median(verr)), float(np.median(perr))
    # A wrong gravity sign over 2 s yields ~2*9.81*2 = 39 m/s of velocity error.
    check("T4 preintegrated velocity tracks GT over 2 s", mv < 5.0,
          "median |v_err| = %.3f m/s over %d windows (gravity-sign error would give ~39 m/s)" % (mv, len(verr)))
    check("T4 preintegrated position tracks GT over 2 s", mp < 10.0,
          "median |p_err| = %.3f m (handedness error would give tens of metres)" % mp)

    # ---- T5: kinematic self-consistency of the labels ---------------------
    # gt_translation must be the integral of velocity, or AirIMU's position and
    # velocity losses fight each other.  Assert the *discrete* trapezoid
    # identity the labels are built from -- that must hold to machine precision.
    # (A central difference of a trapezoidal integral differs from v by
    #  (1/4) v'' dt^2, so comparing np.gradient(p) against v would only ever
    #  measure the curvature of v, not a label inconsistency.)
    t = d["time"].numpy()
    v = d["velocity"].numpy()
    p = d["gt_translation"].numpy()
    dtn = np.diff(t)[:, None]
    resid = float(np.abs(np.diff(p, axis=0) - 0.5 * (v[1:] + v[:-1]) * dtn).max())
    check("T5 gt_pos is exactly the integral of gt_vel", resid < 1e-9,
          "max trapezoid residual = %.2e m" % resid)
    drift = float(np.sqrt((((np.gradient(p, axis=0) / np.gradient(t)[:, None]) - v) ** 2).sum(1)).mean())
    print("       (diagnostic) mean |central-diff(p) - v| = %.2e m/s -- second-order, not an error"
          % drift)

    # ---- T6: rotation is a proper rotation, not a reflection --------------
    M = R.matrix().numpy()
    dets = np.linalg.det(M)
    orth = np.abs(M @ M.transpose(0, 2, 1) - np.eye(3)).max()
    check("T6 gt_orientation is a proper rotation", abs(dets.mean() - 1) < 1e-9 and orth < 1e-9,
          "mean det = %.12f, max |RR^T - I| = %.2e" % (dets.mean(), orth))

    print("  (diagnostic) integrated-altitude drift vs barometer: %.1f m over the flight"
          % seq.altitude_drift)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=3, help="how many files to test if --files not given")
    a = ap.parse_args()

    files = a.files or sorted(glob.glob(os.path.join(a.data_root, "*_sensor_data.csv")))[: a.n]
    if not files:
        sys.exit("no logs found under %s" % a.data_root)
    for f in files:
        verify(f)

    print("\n" + "=" * 78)
    print("%d/%d checks passed" % (sum(_results), len(_results)))
    sys.exit(0 if all(_results) else 1)
