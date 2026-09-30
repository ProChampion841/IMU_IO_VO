"""Is the learned covariance actually calibrated?

AirIMU's distinctive claim is not just a better IMU correction but a *learned
uncertainty* that can be handed to a downstream estimator.  A model can post good
RTE/ROE while emitting covariance that means nothing, and every other check in
this repo validates the correction, not the uncertainty.  This script tests the
uncertainty directly.

Three diagnostics, at every supervision point of every evaluation window:

1. **NEES** (normalised estimation error squared), ``e^T Sigma^-1 e`` using the
   propagated 3x3 diagonal block.  For a correctly calibrated 3-DOF Gaussian the
   mean NEES is **3.0**.  Much larger => over-confident (the filter will diverge);
   much smaller => under-confident (information is being thrown away).  The median
   is reported too, since NEES is chi-squared and its mean is heavy-tailed.
2. **Coverage**: the fraction of components inside +-1/2/3 sigma, which for a
   calibrated Gaussian is 68.3 / 95.4 / 99.7 %.
3. **Discrimination**: Spearman correlation between predicted sigma and realised
   |error|, computed **within each supervision index and then averaged**.  This
   grouping matters.  Correlating sigma against error over all points at once
   mostly measures "uncertainty grows with elapsed time", which any propagated
   covariance does trivially -- both AirIMU and the fixed-constant baseline score
   ~0.8-0.9 that way, which tells you nothing.  Holding elapsed time fixed and
   comparing across windows asks the question that actually matters: does the
   model know *which flights and manoeuvres* are the hard ones?

The same numbers are computed for the raw integrator carrying pypose's fixed
default covariance, which is the "empirically tuned constant" baseline AirIMU is
meant to improve on.

Run:  python tools/eval_covariance.py --config configs/exp/UAV/hybrid_best.conf
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import torch
import torch.utils.data as Data
from pyhocon import ConfigFactory

from datasets import SeqeuncesDataset, collate_fcs
from model import net_dict
from utils import move_to

CHANNELS = (("rot", slice(0, 3), "rad"), ("vel", slice(3, 6), "m/s"), ("pos", slice(6, 9), "m"))


def collect(network, loader, confs, device, max_batches, is_identity=False):
    """Return {channel: (errors, variances)} stacked over all sampled points."""
    out = {k: ([], [], []) for k, _, _ in CHANNELS}
    network.eval()
    with torch.no_grad():
        for i, (data, init, label) in enumerate(loader):
            if i >= max_batches:
                break
            data, init, label = move_to([data, init, label], device)
            if is_identity:
                n = data["dt"].shape[1]
                data["acc"], data["gyro"] = data["acc"][:, -n:], data["gyro"][:, -n:]
            st = network(data, init)
            if st.get("cov") is None:
                raise SystemExit("network produced no covariance (propcov must be True)")
            s = confs.sampling
            cov = torch.diagonal(st["cov"], dim1=-2, dim2=-1)          # (B, K, 9)
            for name, sl, _ in CHANNELS:
                if name == "rot":
                    e = (st["rot"][:, s - 1::s, :] * label["gt_rot"][:, s - 1::s, :].Inv()).Log()
                else:
                    e = st[name][:, s - 1::s, :] - label["gt_" + name][:, s - 1::s, :]
                k = min(e.shape[1], cov.shape[1])
                B = e.shape[0]
                # index of the supervision point within the window, so that
                # discrimination can be measured at fixed elapsed time
                idx = torch.arange(k).repeat(B)
                out[name][0].append(e[:, :k].reshape(-1, 3).double().cpu())
                out[name][1].append(cov[:, :k, sl].reshape(-1, 3).double().cpu())
                out[name][2].append(idx)
    return {k: (torch.cat(v[0]), torch.cat(v[1]), torch.cat(v[2])) for k, v in out.items()}


def _spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def report(tag, stats):
    print("\n%s" % tag)
    print("  %-5s %10s %10s | %7s %7s %7s | %9s %9s" %
          ("chan", "meanNEES", "medNEES", "<1sig", "<2sig", "<3sig", "rho|fixed-t", "rho|pooled"))
    print("  " + "-" * 84)
    for name, _, unit in CHANNELS:
        e, var, idx = stats[name]
        var = var.clamp_min(1e-18)
        sig = var.sqrt()
        nees = (e.pow(2) / var).sum(-1)
        z = e.abs() / sig
        cov1, cov2, cov3 = [(z < k).double().mean().item() * 100 for k in (1, 2, 3)]
        a = sig.norm(dim=-1).numpy()
        b = e.norm(dim=-1).numpy()
        ii = idx.numpy()
        # honest discrimination: across windows at the SAME elapsed time
        per, wts = [], []
        for u in np.unique(ii):
            m = ii == u
            r = _spearman(a[m], b[m])
            if not np.isnan(r):
                per.append(r)
                wts.append(m.sum())
        rho_fixed = float(np.average(per, weights=wts)) if per else float("nan")
        rho_pooled = _spearman(a, b)   # inflated by the within-window time trend
        print("  %-5s %10.2f %10.2f | %6.1f%% %6.1f%% %6.1f%% | %9.3f %9.3f"
              % (name, nees.mean().item(), nees.median().item(), cov1, cov2, cov3,
                 rho_fixed, rho_pooled))
    print("  %-5s %10s %10s | %7s %7s %7s |"
          % ("ideal", "3.00", "2.37", "68.3%", "95.4%", "99.7%"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp/UAV/hybrid_best.conf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--load", default="best_model.ckpt")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_batches", type=int, default=20)
    a = ap.parse_args()

    conf = ConfigFactory.parse_file(a.config)
    conf.train.device = a.device
    conf_name = os.path.split(a.config)[-1].split(".")[0]
    exp_dir = os.path.join(conf.general.exp_dir, conf_name)

    ds = SeqeuncesDataset(data_set_config=conf.dataset.eval)
    collate = collate_fcs[conf.dataset.collate] if "collate" in conf.dataset else collate_fcs["base"]
    ld = Data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate, drop_last=True)

    ck = os.path.join(exp_dir, "ckpt", a.load)
    if not os.path.isfile(ck):
        raise SystemExit("no checkpoint at %s" % ck)
    net = net_dict[conf.train.network](conf.train).to(device=a.device, dtype=ds.get_dtype())
    sd = torch.load(ck, map_location=a.device)
    net.load_state_dict(sd["model_state_dict"])
    print("loaded %s (epoch %d)" % (ck, sd["epoch"]))
    report("AirIMU learned covariance", collect(net, ld, conf.train, a.device, a.max_batches))

    # Baseline: raw integration carrying pypose's fixed default covariance,
    # i.e. the hand-tuned constant that AirIMU is supposed to improve on.
    base_conf = conf.train.copy()
    base_conf["network"], base_conf["propcov"] = "iden", True
    base = net_dict["iden"](base_conf).to(device=a.device, dtype=ds.get_dtype())
    report("Baseline: raw integration, pypose default constant covariance",
           collect(base, ld, base_conf, a.device, a.max_batches, is_identity=True))
    print("\nNEES far above 3 means over-confident (a downstream filter would diverge);")
    print("far below 3 means over-conservative.  Spearman rho near 0 means sigma carries")
    print("no information about which windows are actually hard.\n")


if __name__ == "__main__":
    main()
