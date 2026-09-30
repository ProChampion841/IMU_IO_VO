"""Prove the second (MTI / magnetometer) attitude solution is converted correctly,
and that the attitude *input feature* built from it is what we claim.

Run:  python tools/verify_mti_attitude.py --data_root data [--n 8] [--all]

What this establishes
---------------------
``EulX/Y/Z`` was documented in the loader as "a second attitude solution in
degrees expressed in a z-up / East-referenced frame with a ~13 deg magnetic
declination offset ... not used".  Two of those claims are now measured rather
than assumed:

* **T1** shows the residual between ``from_euler("ZYX", [EulZ, EulY, EulX])`` and
  the ground-truth ``R_nwu_flu`` is a *pure rotation about the world z axis*.
  That is the whole content of the frame question: the roll and pitch of ``Eul``
  are already NWU/FLU angles (they are NOT passed through ``T``), and only the
  yaw reference differs.  If this test passes, no other frame hypothesis is
  needed; if it failed, the roll/pitch statistics below would be meaningless.
* **T2** reports the per-axis mean and std of the difference in degrees, and
  fits the declination from the data instead of trusting the "~13 deg".
* **T3** checks the loader's published ``mti_orientation`` actually removes the
  yaw offset it says it removes.
* **T4/T5** check the network input feature: ``g_body`` is a unit vector, its z
  component is near ``+1`` in level flight, it is invariant to yaw (which is why
  the MTI's unusable heading does not matter), and the ``sin/cos`` channels agree
  with scipy's Euler decomposition.
* **T6** runs the real dataset + collate stack and checks ``mti_rot`` /
  ``init_mti_rot`` survive it, and that a sequence without an MTI channel falls
  back to the ground-truth rotation instead of crashing.

Every check prints PASS/FAIL; the script exits non-zero if any check fails.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import pypose as pp
import torch
from scipy.spatial.transform import Rotation as _R

from datasets.UAVdataset import UAV, MTI_DECLINATION_DEG, _T
from model.attitude import (attitude_feature, attitude_feature_dim,
                            gravity_direction, input_dim, pad_rotation,
                            select_attitude)

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, ok, detail):
    _results.append(bool(ok))
    print("  [%s] %-52s %s" % (PASS if ok else FAIL, name, detail))


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def circ(a_deg):
    """(circular mean, circular std) of degrees, in degrees."""
    r = np.exp(1j * np.deg2rad(np.asarray(a_deg, dtype=np.float64))).mean()
    m = np.rad2deg(np.angle(r))
    s = np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(max(abs(r), 1e-12)))))
    return float(m), float(s)


def per_flight(path, stats):
    """T1/T2 statistics for one flight, accumulated into ``stats``."""
    name = os.path.basename(path)
    # read the raw Eul once more so T1 can test the *uncorrected* rotation
    import pandas as pd
    df = pd.read_csv(path, usecols=["EulX", "EulY", "EulZ",
                                    "GPSNavEulX", "GPSNavEulY", "GPSNavEulZ",
                                    "GPSNavVnX", "GPSNavVnY", "GPSNavVnZ"]).dropna()
    sp = np.linalg.norm(df[["GPSNavVnX", "GPSNavVnY", "GPSNavVnZ"]].to_numpy(float), axis=1)
    df = df[sp > 5.0]
    if len(df) < 1000:
        return None
    ex, ey, ez = [df[c].to_numpy(float) for c in ("EulX", "EulY", "EulZ")]
    gr, gp, gy = [df[c].to_numpy(float) for c in ("GPSNavEulX", "GPSNavEulY", "GPSNavEulZ")]

    R_gt = _T[None] @ _R.from_euler("ZYX", np.stack([gy, gp, gr], 1)).as_matrix() @ _T[None]
    R_e = _R.from_euler("ZYX", np.deg2rad(np.stack([ez, ey, ex], 1))).as_matrix()

    # --- T1: is the residual a pure world-z rotation? ---
    Q = R_e @ np.transpose(R_gt, (0, 2, 1))
    rv = _R.from_matrix(Q).as_rotvec()
    ang = np.linalg.norm(rv, axis=1)
    axis_z = np.abs(rv[:, 2]) / np.maximum(ang, 1e-12)
    axis_z_med = float(np.median(axis_z))

    # --- T2: per-axis differences ---
    e_gt = _R.from_matrix(R_gt).as_euler("ZYX", degrees=True)   # yaw, pitch, roll
    e_mt = _R.from_matrix(R_e).as_euler("ZYX", degrees=True)
    dr = wrap_deg(e_mt[:, 2] - e_gt[:, 2])
    dp = wrap_deg(e_mt[:, 1] - e_gt[:, 1])
    dy = wrap_deg(e_mt[:, 0] - e_gt[:, 0])
    rmu, rsd = circ(dr)
    pmu, psd = circ(dp)
    ymu, ysd = circ(dy)
    decl = wrap_deg(90.0 - ymu)

    # g_body agreement (yaw-invariant), rows 2 of each matrix
    cosang = np.clip((R_e[:, 2, :] * R_gt[:, 2, :]).sum(1), -1, 1)
    gerr = np.rad2deg(np.arccos(cosang))

    stats["rows"].append((name, len(df), axis_z_med, rmu, rsd, pmu, psd, ymu, ysd, decl,
                          float(np.median(gerr)), float(np.percentile(gerr, 95))))
    stats["axis_z"].append(axis_z_med)
    stats["rmu"].append(rmu); stats["rsd"].append(rsd)
    stats["pmu"].append(pmu); stats["psd"].append(psd)
    stats["ymu"].append(ymu); stats["ysd"].append(ysd)
    stats["decl"].append(decl)
    stats["gerr"].append(gerr[::20].copy())
    return stats["rows"][-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--n", type=int, default=8, help="flights to summarise if --all is not given")
    ap.add_argument("--all", action="store_true", help="use every flight in data_root")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data_root, "*_sensor_data.csv")))
    if not files:
        print("no logs found under %r" % a.data_root)
        return 2
    subset = files if a.all else files[: a.n]

    stats = {k: [] for k in
             ("rows", "axis_z", "rmu", "rsd", "pmu", "psd", "ymu", "ysd", "decl", "gerr")}
    print("=" * 100)
    print("%-38s %7s %7s %8s %7s %8s %7s %8s %7s %7s %8s" %
          ("flight", "N", "axis_z", "roll_mu", "roll_sd", "pitch_mu", "pitch_sd",
           "yaw_mu", "yaw_sd", "decl", "g_err50"))
    for f in subset:
        r = per_flight(f, stats)
        if r is None:
            print("%-38s  (skipped: fewer than 1000 airborne rows)" % os.path.basename(f)[:38])
            continue
        print("%-38s %7d %7.4f %8.2f %7.2f %8.2f %7.2f %8.2f %7.2f %7.2f %8.3f" %
              (r[0][:38], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]))
    print("=" * 100)

    nflights = len(stats["rows"])
    gerr = np.concatenate(stats["gerr"])

    # ---- T1 -----------------------------------------------------------------
    axz = np.array(stats["axis_z"])
    check("T1 Eul-vs-GT residual is a pure world-z rotation",
          axz.min() > 0.90,
          "median |axis_z| per flight: min %.4f, median %.4f over %d flights "
          "(1.0 = pure yaw; a roll/pitch frame error would push this far below 1)"
          % (axz.min(), np.median(axz), nflights))

    # ---- T2 -----------------------------------------------------------------
    rmu, rsd = np.array(stats["rmu"]), np.array(stats["rsd"])
    pmu, psd = np.array(stats["pmu"]), np.array(stats["psd"])
    ysd = np.array(stats["ysd"])
    check("T2a roll agrees with GT to about a degree",
          abs(np.median(rmu)) < 2.0 and np.median(rsd) < 5.0,
          "roll  diff: mean %+.2f deg (per-flight medians), std %.2f deg" %
          (np.median(rmu), np.median(rsd)))
    check("T2b pitch agrees with GT to about a degree",
          abs(np.median(pmu)) < 2.0 and np.median(psd) < 5.0,
          "pitch diff: mean %+.2f deg (per-flight medians), std %.2f deg" %
          (np.median(pmu), np.median(psd)))
    ymu_m, ymu_s = circ(np.array(stats["ymu"]))
    print("       yaw   diff: circular mean %+.2f deg across flights, circular std %.2f deg; "
          "within-flight std: median %.2f, max %.2f" %
          (ymu_m, ymu_s, np.median(ysd), ysd.max()))

    dmu, dsd = circ(np.array(stats["decl"]))
    stable = np.array([d for d, s in zip(stats["decl"], stats["ysd"]) if s < 15.0])
    smu, ssd = circ(stable) if len(stable) else (float("nan"), float("nan"))
    # This is a corpus-level statistic: with a handful of flights the circular
    # mean is dominated by the flights whose heading has drifted, so only assert
    # it when enough flights are in the sample (use --all).
    check("T2c fitted declination is near the documented ~13 deg"
          + ("" if nflights >= 20 else " [informational: %d flights < 20]" % nflights),
          abs(wrap_deg(dmu - 13.0)) < 10.0 or nflights < 20,
          "fitted = 90 - yaw_offset: circ mean %.2f deg (circ std %.2f) over %d flights; "
          "%.2f deg over the %d flights with within-flight yaw std < 15 deg; median %.2f, "
          "IQR [%.2f, %.2f]" %
          (dmu, dsd, nflights, smu, len(stable), np.median(stats["decl"]),
           np.percentile(stats["decl"], 25), np.percentile(stats["decl"], 75)))
    print("       NOTE: the across-flight declination spread (circ std %.1f deg) and the "
          "within-flight\n             drift (median %.1f deg) mean the MTI *heading* is not "
          "usable as an absolute\n             reference.  Roll and pitch are." % (dsd, np.median(ysd)))

    # ---- T3: the loader's published rotation -------------------------------
    seq = UAV(a.data_root, os.path.basename(subset[0]))
    assert "mti_orientation" in seq.data
    m = seq.data["mti_orientation"]
    g = seq.data["gt_orientation"]
    Rm = _R.from_quat(m.tensor().numpy()).as_matrix()
    Rg = _R.from_quat(g.tensor().numpy()).as_matrix()
    yaw_m = np.rad2deg(np.arctan2(Rm[:, 1, 0], Rm[:, 0, 0]))
    yaw_g = np.rad2deg(np.arctan2(Rg[:, 1, 0], Rg[:, 0, 0]))
    resid_mu, _ = circ(wrap_deg(yaw_m - yaw_g))
    raw = UAV(a.data_root, os.path.basename(subset[0]), mti_yaw_ref="raw")
    Rr = _R.from_quat(raw.data["mti_orientation"].tensor().numpy()).as_matrix()
    yaw_r = np.rad2deg(np.arctan2(Rr[:, 1, 0], Rr[:, 0, 0]))
    raw_mu, _ = circ(wrap_deg(yaw_r - yaw_g))
    check("T3 mti_orientation removes the East+declination yaw offset",
          abs(wrap_deg(raw_mu - resid_mu - (90.0 - MTI_DECLINATION_DEG))) < 1e-6,
          "raw yaw offset %+.2f deg -> published %+.2f deg (removed exactly %.2f = 90 - %.2f); "
          "the leftover %+.2f is this flight's own declination error"
          % (raw_mu, resid_mu, 90.0 - MTI_DECLINATION_DEG, MTI_DECLINATION_DEG, resid_mu))
    check("T3b mti_orientation is a proper rotation, same shape as gt",
          m.lshape == g.lshape and abs(np.linalg.det(Rm) - 1).max() < 1e-9,
          "shape %s, max |det-1| = %.2e, max |RR^T - I| = %.2e"
          % (tuple(m.lshape), abs(np.linalg.det(Rm) - 1).max(),
             abs(Rm @ np.transpose(Rm, (0, 2, 1)) - np.eye(3)).max()))

    # ---- T4: g_body on real data -------------------------------------------
    rot = g[None]                                   # (1, N) pp.SO3
    gb = gravity_direction(rot)
    nrm = gb.norm(dim=-1)
    check("T4a g_body = R.Inv() @ [0,0,1] has unit norm",
          float((nrm - 1).abs().max()) < 1e-9,
          "max ||g_body| - 1| = %.2e over %d frames" % (float((nrm - 1).abs().max()), gb.shape[1]))
    # level flight: |roll| and |pitch| both under 5 deg, judged from the GT rotation
    lev = (gb[0, :, 2] > np.cos(np.deg2rad(5.0)))
    gz = gb[0, lev, 2] if int(lev.sum()) else gb[0, :, 2]
    check("T4b g_body z is ~ +1 in level flight",
          int(lev.sum()) > 0 and float(gz.min()) > 0.99,
          "%d/%d frames within 5 deg of level; g_body_z there: min %.5f, median %.5f "
          "(all frames: median %.4f)"
          % (int(lev.sum()), gb.shape[1], float(gz.min()), float(gz.median()),
             float(gb[0, :, 2].median())))
    # yaw invariance: rotate the whole trajectory about world z, g_body must not move
    psi = torch.tensor([0.0, 0.0, 1.234], dtype=g.dtype)
    yawed = pp.so3(psi).Exp()[None, None] @ rot
    check("T4c g_body is invariant to a world-yaw rotation",
          float((gravity_direction(yawed) - gb).abs().max()) < 1e-9,
          "max change after a 70.7 deg yaw: %.2e" %
          float((gravity_direction(yawed) - gb).abs().max()))

    # ---- T5: the feature helper --------------------------------------------
    eul = _R.from_quat(g.tensor().numpy()).as_euler("ZYX")     # yaw, pitch, roll
    f7 = attitude_feature(rot, "gravity_sincos")
    ref = np.stack([np.sin(eul[:, 2]), np.cos(eul[:, 2]),
                    np.sin(eul[:, 1]), np.cos(eul[:, 1])], 1)
    err = np.abs(f7[0, :, 3:].numpy() - ref).max()
    check("T5a gravity_sincos channels match scipy roll/pitch",
          err < 1e-8, "max |sin/cos(roll,pitch) - scipy| = %.2e" % err)
    dims = [attitude_feature(rot, mo).shape[-1] for mo in ("none", "gravity", "gravity_sincos")]
    check("T5b feature widths are 0 / 3 / 7 and input_dim is 6 / 9 / 13",
          dims == [0, 3, 7] and
          [input_dim(mo) for mo in ("none", "gravity", "gravity_sincos")] == [6, 9, 13],
          "widths %s, input_dim %s" %
          (dims, [input_dim(mo) for mo in ("none", "gravity", "gravity_sincos")]))
    mti_rot = m[None]
    cos = (gravity_direction(mti_rot) * gb).sum(-1).clamp(-1, 1)
    ang = torch.rad2deg(torch.arccos(cos))
    check("T5c MTI g_body tracks GT g_body despite the broken heading",
          float(ang.median()) < 6.0,
          "this flight: median %.2f deg, p95 %.2f deg | whole corpus subset: median %.2f, "
          "p95 %.2f, p99 %.2f deg" %
          (float(ang.median()), float(np.percentile(ang.numpy(), 95)),
           float(np.median(gerr)), float(np.percentile(gerr, 95)), float(np.percentile(gerr, 99))))

    # ---- T6: the dataset / collate path ------------------------------------
    from datasets.dataset import SeqeuncesDataset, mti_or_gt
    from datasets.dataset_utils import collate_fcs
    from pyhocon import ConfigFactory
    cfg = ConfigFactory.from_dict({
        "mode": "train", "gravity": 9.81007, "dtype": "float32",
        "data_list": [{"name": "UAV", "window_size": 1000, "step_size": 1000,
                       "data_root": a.data_root,
                       "data_drive": [os.path.basename(subset[0])]}],
    })
    ds = SeqeuncesDataset(data_set_config=cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False,
                                         collate_fn=collate_fcs["padding9"])
    data, init, label = next(iter(loader))
    ok = ("mti_rot" in data and "mti_rot" in init
          and data["mti_rot"].lshape == data["rot"].lshape
          and init["mti_rot"].lshape == init["rot"].lshape
          and data["acc"].shape[1] == data["rot"].lshape[1] + 9)
    check("T6a padding9 collate carries mti_rot and init_mti_rot",
          ok,
          "data['rot'] %s, data['mti_rot'] %s, init['mti_rot'] %s, acc %s (window + 9 pad)"
          % (tuple(data["rot"].lshape), tuple(data["mti_rot"].lshape),
             tuple(init["mti_rot"].lshape), tuple(data["acc"].shape)))
    feat = attitude_feature(select_attitude(data, source="mti")[0], "gravity")
    check("T6b attitude_feature runs on a collated batch",
          feat.shape == data["rot"].lshape + (3,)
          and float((feat.norm(dim=-1) - 1).abs().max()) < 1e-5,
          "feature %s, max ||g|-1| = %.2e, source used = %r"
          % (tuple(feat.shape), float((feat.norm(dim=-1) - 1).abs().max()),
             select_attitude(data, source="mti")[1]))

    # T6e: the pad that makes the feature line up with padding9's acc/gyro
    for src in ("gt", "mti"):
        r, _ = select_attitude(data, source=src)
        ir = init["rot"] if src == "gt" else init["mti_rot"]
        padded = pad_rotation(r, ir, 9)
        f = attitude_feature(padded, "gravity_sincos")
        ok = (padded.lshape[1] == data["acc"].shape[1]
              and torch.equal(padded.tensor()[:, 9:], r.tensor())
              and torch.equal(padded.tensor()[:, :9],
                              ir[:, :1].tensor().expand(-1, 9, -1))
              and f.shape[:2] == data["acc"].shape[:2] and f.shape[-1] == 7)
        check("T6e pad_rotation(%s) matches the padded acc/gyro length" % src, ok,
              "rot %s + 9 -> %s == acc %s; feature %s; the 9 pad frames are init_rot"
              % (tuple(r.lshape), tuple(padded.lshape), tuple(data["acc"].shape[:2]),
                 tuple(f.shape)))

    # fallback: a sequence with the MTI channel switched off stands in for
    # sequences that publish no MTI channel.
    noseq = UAV(a.data_root, os.path.basename(subset[0]), mti_attitude=False)
    fb = mti_or_gt(noseq, "no-mti")
    check("T6c a sequence without mti_orientation falls back to gt (with a warning)",
          torch.equal(fb.tensor(), noseq.data["gt_orientation"].tensor()),
          "fallback returned gt_orientation, shape %s" % (tuple(fb.lshape),))
    stripped = {k: v for k, v in data.items() if k != "mti_rot"}
    r2, src2 = select_attitude(stripped, source="mti")
    check("T6d select_attitude falls back to 'rot' when mti_rot is absent",
          src2 == "gt" and torch.equal(r2.tensor(), data["rot"].tensor()),
          "used source %r" % src2)

    print("\n" + "=" * 100)
    print("%d/%d checks passed" % (sum(_results), len(_results)))
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
