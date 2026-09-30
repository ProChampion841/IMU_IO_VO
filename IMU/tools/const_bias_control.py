"""THE CONTROL EXPERIMENT: does a single fitted 3-vector match the trained network?

WHY THIS EXISTS.  Measured 2026-09-07, the trained `accel_const` network emits a
correction whose cosine to its own corpus mean is 0.970 -- it is, to a very good
approximation, ONE FIXED VECTOR that it barely modulates per flight.  And the
774-epoch `hybrid_best` run (2026-09-08) had its best epoch at 1, i.e. before it had
learned anything.  Both point at the same question: is the network buying anything at
all over a constant that costs no training?

This fits ONE 3-vector on the TRAIN flights, with no network, and scores it on the
held-out flights with the same integrator, the same windows, the same 15 s freeze and
the same statistic as every other arm.  Three arms come out directly comparable:

    raw          the freeze alone (what `*_raw_*` means everywhere in this repo)
    const        the freeze + one fitted 3-vector, NO network
    net          the freeze + the trained correction, when --ckpt is given

HOW THE FIT IS DONE, AND WHY NOT ANALYTICALLY.  Endpoint velocity and position are
EXACTLY linear in a constant accel bias b, so the least-squares fit is closed-form
once you have the Jacobian J = d(endpoint error)/db.  Deriving J by hand means
re-deriving this repo's dt/prefix convention, which has produced four shipped
off-by-one errors.  Instead J is obtained by FINITE DIFFERENCES through the very same
`network.integrate` call the raw arm uses:

    J[:, :, i] = (err(b = eps * e_i) - err(b = 0)) / eps

Because the map is exactly linear this is not an approximation -- it is the exact
Jacobian up to float rounding, and `--check_linear` verifies that by confirming the
response at 2*eps is exactly twice the response at eps.  Four integrator passes per
batch.

The fit minimises the POOLED squared endpoint error over train windows:

    b* = argmin_b  sum_w || e0_w + J_w b ||^2   =>   b* = -(sum J^T J)^-1 (sum J^T e0)

FIT TARGET.  `--fit_on pos` (default) matches `select_metric: pos_error_60s`, the
criterion every checkpoint in this repo was selected on.  `--fit_on vel` fits the
velocity endpoint instead.  Both channels are always REPORTED, so a fit that helps one
and hurts the other is visible rather than hidden.

READ THE PER-FLIGHT COLUMNS, NOT JUST `pooled`.  The pooled figure is a ratio of RMS
and is dominated by the worst flight; on the held-out split one flight carries ~53% of
the squared position error.  `median` and `LOFO` are printed next to it for that reason.
"""
import argparse
import copy
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import torch
import torch.utils.data as Data
from pyhocon import ConfigFactory

from datasets import SeqeuncesDataset, collate_fcs
from model import net_dict
from utils import move_to
from tools.eval_vel_horizons import endpoint_index, horizon_tag


def build_loader(conf, section, horizon, collate_fn, batch_size):
    dc = copy.deepcopy(conf.dataset[section])
    for e in dc.data_list:
        e["window_size"] = int(horizon)
        e["step_size"] = int(horizon)
    names = [f for e in dc.data_list for f in e["data_drive"]]
    ds = SeqeuncesDataset(data_set_config=dc)
    ld = Data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return ds, ld, names


def endpoint_errors(network, data, init_state, label, idx, bias):
    """Endpoint (pos, vel) error vectors with a CONSTANT accel bias added.

    `bias=None` reproduces the raw arm exactly as tools/eval_vel_horizons.py defines
    it.  Everything else -- init state, gravity, gt attitude, the 15 s freeze already
    applied in the dataset -- is untouched, so the arms differ only by the signal.
    """
    interval = getattr(network, "interval", 0)
    d = dict(data)
    acc = data["acc"][:, interval:, :]
    if bias is not None:
        acc = acc + bias.view(1, 1, 3).to(acc.dtype)
    d["corrected_acc"] = acc
    d["corrected_gyro"] = data["gyro"][:, interval:, :]
    out = network.integrate(init_state=init_state, data=d,
                            cov_state={"acc_cov": None, "gyro_cov": None})
    return (out["pos"][:, idx, :] - label["gt_pos"][:, idx, :],
            out["vel"][:, idx, :] - label["gt_vel"][:, idx, :])


def fit_constant(network, loader, idx, device, eps, fit_on, check_linear):
    """Closed-form least-squares constant bias over every window in `loader`."""
    JtJ = torch.zeros(3, 3, dtype=torch.float64)
    Jte = torch.zeros(3, dtype=torch.float64)
    n, lin_err = 0, 0.0
    basis = torch.eye(3, device=device)
    with torch.no_grad():
        for data, init_state, label in loader:
            data, init_state, label = move_to([data, init_state, label], device)
            p0, v0 = endpoint_errors(network, data, init_state, label, idx, None)
            e0 = (p0 if fit_on == "pos" else v0).double().cpu()
            cols = []
            for i in range(3):
                pi, vi = endpoint_errors(network, data, init_state, label, idx,
                                         eps * basis[i])
                ei = (pi if fit_on == "pos" else vi).double().cpu()
                cols.append((ei - e0) / eps)
            if check_linear:
                p2, v2 = endpoint_errors(network, data, init_state, label, idx,
                                         2.0 * eps * basis[0])
                e2 = (p2 if fit_on == "pos" else v2).double().cpu()
                pred = e0 + 2.0 * eps * cols[0]
                denom = float((e2 - e0).abs().max()) or 1.0
                lin_err = max(lin_err, float((e2 - pred).abs().max()) / denom)
            J = torch.stack(cols, dim=-1)                      # (B, 3, 3)
            JtJ += torch.einsum("bki,bkj->ij", J, J)
            Jte += torch.einsum("bki,bk->i", J, e0)
            n += int(e0.shape[0])
    return -torch.linalg.solve(JtJ, Jte), n, lin_err


def accumulate(per, names, index_map, k, m_pos, m_vel, r_pos, r_vel):
    for j in range(r_pos.shape[0]):
        s = per.setdefault(names[index_map[k + j][0]], [0.0, 0.0, 0.0, 0.0, 0])
        s[0] += float(m_pos[j].norm() ** 2)
        s[1] += float(r_pos[j].norm() ** 2)
        s[2] += float(m_vel[j].norm() ** 2)
        s[3] += float(r_vel[j].norm() ** 2)
        s[4] += 1
    return k + r_pos.shape[0]


def report(per, tag, arm):
    for ch, mi, ri in (("pos", 0, 1), ("vel", 2, 3)):
        fl = {f: v for f, v in per.items() if v[ri] > 0}
        if not fl:
            continue
        M = sum(v[mi] for v in fl.values())
        R = sum(v[ri] for v in fl.values())
        N = sum(v[4] for v in fl.values())
        rat = sorted((math.sqrt(v[mi] / v[ri]), f, v) for f, v in fl.items())
        lofo = [math.sqrt((M - v[mi]) / (R - v[ri])) for _, _, v in rat if R - v[ri] > 0]
        mid = len(rat) // 2
        med = rat[mid][0] if len(rat) % 2 else 0.5 * (rat[mid - 1][0] + rat[mid][0])
        helped = sum(1 for x, _, _ in rat if x < 1.0)
        print("  %-5s %-4s %-5s %3d win %2d fl | pooled %.4f | median %.4f | "
              "helped %2d/%-2d | LOFO %.4f..%.4f%s"
              % (arm, ch, tag, N, len(fl), math.sqrt(M / R), med, helped, len(rat),
                 min(lofo), max(lofo), "  **CROSSES 1.0**" if max(lofo) >= 1.0 else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=[3000, 6000, 12000])
    ap.add_argument("--fit_split", default="train")
    ap.add_argument("--score_split", default="inference",
                    help="default is the only genuinely HELD-OUT section; `test` is the "
                         "checkpoint-selection set and is not held out for a network")
    ap.add_argument("--fit_horizon", type=int, default=6000,
                    help="horizon the constant is fitted at (default 60 s, matching "
                         "select_metric: pos_error_60s)")
    ap.add_argument("--fit_on", choices=["pos", "vel"], default="pos")
    ap.add_argument("--ckpt", default=None,
                    help="optional: also score this trained checkpoint on the same windows")
    ap.add_argument("--eps", type=float, default=0.01, help="finite-difference step, m/s^2")
    ap.add_argument("--check_linear", action="store_true",
                    help="verify the response at 2*eps is exactly twice that at eps")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()

    conf = ConfigFactory.parse_file(a.config)
    conf.train.device = a.device
    collate_fn = (collate_fcs[conf.dataset.collate] if "collate" in conf.dataset.keys()
                  else collate_fcs["base"])
    sampling = conf.train.sampling

    # The network object is only an INTEGRATOR here: `integrate` never reads the
    # correction heads, so an untrained instance gives the exact raw and const arms.
    net = net_dict[conf.train.network](conf.train).to(a.device).float()
    net.eval()

    idx_fit = endpoint_index(a.fit_horizon, sampling)
    _, fit_ld, _ = build_loader(conf, a.fit_split, a.fit_horizon, collate_fn, a.batch_size)
    b, n_fit, lin = fit_constant(net, fit_ld, idx_fit, a.device, a.eps,
                                 a.fit_on, a.check_linear)
    print("")
    print("=" * 100)
    print("FITTED CONSTANT (no network)  on '%s' at %s, target=%s, %d windows"
          % (a.fit_split, horizon_tag(a.fit_horizon), a.fit_on, n_fit))
    print("  b = [%+.6f, %+.6f, %+.6f] m/s^2   |b| = %.6f"
          % (b[0], b[1], b[2], float(b.norm())))
    if a.check_linear:
        print("  linearity: max relative deviation at 2*eps = %.3e  (exactly linear => 0)"
              % lin)
    print("=" * 100)

    trained = None
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location=a.device, weights_only=False)
        trained = net_dict[conf.train.network](conf.train).to(a.device).float()
        trained.load_state_dict(ck.get("model_state_dict", ck))
        trained.eval()
        print("  [ckpt] %s (epoch %s)" % (a.ckpt, ck.get("epoch", "?")))
    print("  scoring on '%s'   (below 1.0 = better than the freeze-only raw arm)"
          % a.score_split)
    print("")

    bdev = b.float().to(a.device)
    for h in sorted(a.horizons):
        idx = endpoint_index(h, sampling)
        ds, ld, names = build_loader(conf, a.score_split, h, collate_fn, a.batch_size)
        if len(ds) == 0:
            print("  (no window of %s in '%s')" % (horizon_tag(h), a.score_split))
            continue
        tag = horizon_tag(h)
        per_c, per_n, k = {}, {}, 0
        with torch.no_grad():
            for data, init_state, label in ld:
                data, init_state, label = move_to([data, init_state, label], a.device)
                r_pos, r_vel = endpoint_errors(net, data, init_state, label, idx, None)
                c_pos, c_vel = endpoint_errors(net, data, init_state, label, idx, bdev)
                accumulate(per_c, names, ds.index_map, k, c_pos, c_vel, r_pos, r_vel)
                if trained is not None:
                    o = trained(data, init_state)
                    accumulate(per_n, names, ds.index_map, k,
                               o["pos"][:, idx, :] - label["gt_pos"][:, idx, :],
                               o["vel"][:, idx, :] - label["gt_vel"][:, idx, :],
                               r_pos, r_vel)
                k += r_pos.shape[0]
        report(per_c, tag, "const")
        if trained is not None:
            report(per_n, tag, "net")
        print("")


if __name__ == "__main__":
    main()
