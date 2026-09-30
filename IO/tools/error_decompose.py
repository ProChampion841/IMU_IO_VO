"""Where does the 40 s error POINT?  (no network, no fitting, read-only)

126.6 m is a scalar, and a scalar cannot say what broke.  The same 126.6 m comes
out of four physically different faults with four different fixes:

  along-track dominant, one sign   -> longitudinal accel bias / scale error
  cross-track dominant             -> HEADING error at handover.  This one grows
                                      ~ |v| * t * sin(dpsi), i.e. LINEARLY, so at
                                      40 s and 45 m/s a 4 deg heading error is
                                      already 126 m all by itself.
  horizontal, no preferred axis    -> tilt / gravity leakage: a tilt theta leaks
                                      9.81*sin(theta) into whatever direction it
                                      points, and 0.5 deg -> 0.086 m/s^2 -> 68 m.
  vertical dominant                -> vertical accel bias or gravity magnitude.

Woodman (2007) settled the same question for the MTx in one paragraph by looking
at the direction of the error rather than its size.  Six angles of literature
survey on this corpus and not one of them asked what the error looks like.

The script integrates RAW IMU from a GT initial state over W frames and resolves
the endpoint position error into the track frame at the START of the window --
the direction the aircraft was flying when the aid was lost.  It also reports the
initial MTI-vs-GT attitude error split into heading and tilt, because that is the
quantity the cross-track column is supposed to be explained by.

  --gtrot  hands GT rotation to the integrator, killing the attitude path.  The
           difference between the two runs is the attitude share of each axis.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import pypose as pp
import torch

from datasets.UAVdataset import UAV


def track_frame(vel0):
    """Unit along-track and cross-track (right of track) vectors in NED.

    Rotating a horizontal NED vector (n, e) by +90 deg about Down gives (-e, n):
    heading north (1, 0) maps to east (0, 1), which is right of track.  Vertical
    is just the Down axis, so it needs no basis vector.
    """
    h = vel0[:, :2]
    spd = h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    a = h / spd
    c = torch.stack([-a[:, 1], a[:, 0]], dim=-1)
    return a, c, spd.squeeze(-1)


def stats(x):
    return float(x.mean()), float(x.pow(2).mean().sqrt())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--win", type=int, default=4000)
    ap.add_argument("--nwin", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gtrot", action="store_true",
                    help="supply GT rotation to the integrator; isolates the accel-bias path")
    ap.add_argument("--mti", action="store_true",
                    help="initialise attitude from the MTI instead of GT (the honest runtime "
                         "condition -- the MTI is independent of the GPS nav filter)")
    ap.add_argument("--mti_yaw_ref", default="fixed",
                    choices=["fixed", "raw", "fit", "wind", "wind_causal"],
                    help="how the MTI yaw reference is resolved; only bites under --mti")
    ap.add_argument("--mti_yaw_hist_s", type=float, default=120.0)
    a = ap.parse_args()
    dev = a.device

    rows = []
    for f in a.files:
        seq = UAV(a.data_root, f, trim_to_airborne=True,
                  mti_yaw_ref=a.mti_yaw_ref, mti_yaw_hist_s=a.mti_yaw_hist_s)
        d = seq.data
        n = d["time"].shape[0]
        W = a.win
        if n < W + 2:
            print("%s: too short (%d frames)" % (f, n))
            continue
        starts = list(range(0, n - W - 1, W))[:a.nwin]
        if not starts:
            continue

        rk = "mti_orientation" if (a.mti and "mti_orientation" in d) else "gt_orientation"
        if a.mti and rk != "mti_orientation":
            sys.exit("--mti requested but this loader published no mti_orientation")

        dt = torch.stack([d["dt"][s:s + W] for s in starts]).to(dev)
        acc = torch.stack([d["acc"][s:s + W] for s in starts]).to(dev)
        gyro = torch.stack([d["gyro"][s:s + W] for s in starts]).to(dev)
        rot = pp.SO3(torch.stack([d[rk][s:s + W].tensor() for s in starts])).to(dev)
        vel0 = torch.stack([d["velocity"][s] for s in starts]).to(dev)
        init = {"pos": torch.stack([d["gt_translation"][s] for s in starts])[:, None].to(dev),
                "vel": vel0[:, None],
                "rot": pp.SO3(torch.stack([d[rk][s].tensor() for s in starts])[:, None]).to(dev)}
        gt_p = torch.stack([d["gt_translation"][s + W] for s in starts]).to(dev)

        integ = pp.module.IMUPreintegrator(reset=True, prop_cov=False,
                                           gravity=seq.gravity).double().to(dev)
        with torch.no_grad():
            out = integ(init_state=init, dt=dt, gyro=gyro, acc=acc,
                        rot=rot if a.gtrot else None)
        err = (out["pos"][:, -1] - gt_p)

        ah, ch, spd = track_frame(vel0)
        e_al = (err[:, :2] * ah).sum(-1)
        e_cr = (err[:, :2] * ch).sum(-1)
        e_vt = err[:, 2]

        # Initial attitude error, world frame: the Down component of the log is
        # yaw (heading); the horizontal part is tilt.
        r_gt = pp.SO3(torch.stack([d["gt_orientation"][s].tensor() for s in starts])).to(dev)
        if "mti_orientation" in d:
            r_mti = pp.SO3(torch.stack([d["mti_orientation"][s].tensor()
                                        for s in starts])).to(dev)
            lg = (r_mti * r_gt.Inv()).Log()
            hd = torch.rad2deg(lg[:, 2].abs())
            tl = torch.rad2deg(lg[:, :2].norm(dim=-1))
        else:
            hd = tl = torch.zeros(len(starts), device=dev)

        rows.append(dict(f=f, n=len(starts), spd=float(spd.mean()),
                         tot=float(err.norm(dim=-1).mean()),
                         al=stats(e_al), cr=stats(e_cr), vt=stats(e_vt),
                         hd=float(hd.mean()), tl=float(tl.mean())))

    if not rows:
        return
    print("\n%-34s %3s %5s %8s | %16s %16s %16s | %6s %6s"
          % ("flight", "n", "spd", "|err|", "along mean/rms", "cross mean/rms",
             "vert mean/rms", "hdg", "tilt"))
    print("-" * 130)
    for r in rows:
        print("%-34s %3d %5.1f %8.1f | %7.1f %8.1f %7.1f %8.1f %7.1f %8.1f | %6.2f %6.2f"
              % (r["f"][:34], r["n"], r["spd"], r["tot"],
                 r["al"][0], r["al"][1], r["cr"][0], r["cr"][1], r["vt"][0], r["vt"][1],
                 r["hd"], r["tl"]))

    W = np.array([r["n"] for r in rows], dtype=float)
    ms = lambda k: float(np.average([r[k][1] ** 2 for r in rows], weights=W))
    al, cr, vt = ms("al"), ms("cr"), ms("vt")
    tot = al + cr + vt
    print("-" * 130)
    print("POOLED %d flights, %d windows, %.0f s, gtrot=%s, init_att=%s"
          % (len(rows), int(W.sum()), a.win / 100.0, a.gtrot, "MTI" if a.mti else "GT"))
    print("  rms  along %7.1f m   cross %7.1f m   vert %7.1f m   |  total %7.1f m"
          % (al ** .5, cr ** .5, vt ** .5, tot ** .5))
    print("  share of squared error:  along %4.0f%%   cross %4.0f%%   vert %4.0f%%"
          % (100 * al / tot, 100 * cr / tot, 100 * vt / tot))
    sp = float(np.average([r["spd"] for r in rows], weights=W))
    t = a.win / 100.0
    print("  1.0 deg of heading error at %.0f m/s for %.0f s gives %.0f m cross-track"
          % (sp, t, sp * t * np.sin(np.deg2rad(1.0))))
    print("  the measured cross-track rms implies %.2f deg of equivalent heading error"
          % np.rad2deg(np.arcsin(min(1.0, cr ** .5 / (sp * t)))))
    mb = lambda k: float(np.average([r[k][0] for r in rows], weights=W))
    print("  signed means (a one-sided axis is a BIAS, a zero mean is a random spread):")
    print("      along %7.1f m   cross %7.1f m   vert %7.1f m" % (mb("al"), mb("cr"), mb("vt")))
    print("  init MTI-vs-GT attitude: heading %.2f deg, tilt %.2f deg"
          % (float(np.average([r["hd"] for r in rows], weights=W)),
             float(np.average([r["tl"] for r in rows], weights=W))))


if __name__ == "__main__":
    main()
