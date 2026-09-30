"""Export a trained velnet checkpoint to ONNX, and check it against PyTorch.

WHAT IS EXPORTED
----------------
`VelocityNet.inference` -- the model's actual claim -- wrapped so the ONNX graph is
self-contained:

    inputs   acc        (B, F, 3)  float32  specific force, m/s^2, body FLU
             gyro       (B, F, 3)  float32  angular rate,  rad/s,  body FLU
             airspeed   (B, F, 1)  float32  pitot, m/s      (only if use_airspeed)
    outputs  vel_body   (B, F, 3)  float32  body-frame velocity, m/s
             vel_cov    (B, F, 3)  float32  per-axis variance, (m/s)^2
                                            (only if propcov)

Units and axes are what datasets/UAVdataset.py hands the network: acc in m/s^2 and
gyro in rad/s, both already converted FRD -> FLU.  NOT the raw log units (g, deg/s).
Input normalisation is inside the graph (the `in_scale`/`in_offset` buffers), so feed
raw physical values.

The 9-frame pad the training collate adds (`padding9_honest`: repeat the first real
sample 9 times) is ALSO inside the graph, so F input frames give F output frames and
the caller never has to know about it.  Pass --no_pad to export the bare network
instead (input F + 9 frames, output F).

The world-frame outputs (`vel`, `pos`) are NOT exported: they need an attitude
(GPS or MTi) and an initial position, and are a rotation plus a cumulative sum the
runtime can do itself.  `att_input` other than "none" is not supported for the
same reason (it needs a pypose SO3 attitude input).

FIXED WINDOW LENGTH
-------------------
F is FIXED at export time (--frames, default 6000 = 60 s, the training window).
The pure-PyTorch Mamba scan is a Python loop over SSM steps and the GRU chunking is
computed from F, so tracing bakes both in.  A graph exported at one F rejects any
other F.  Export once per window length you deploy.  Batch size is dynamic.

The Mamba CUDA kernel (mamba-ssm) cannot be exported; the export always runs on CPU,
where the model uses the bit-compatible pure-PyTorch path on the same weights.

USAGE
-----
    python -m tools.export_onnx --config configs/exp/UAV/velnet_v1.conf \\
        --ckpt experiments/UAV/velnet_v1/ckpt/best_model.ckpt \\
        --out experiments/UAV/velnet_v1/velnet_6000.onnx --frames 6000

It then runs the same random input through PyTorch and onnxruntime and prints the
largest difference (skip with --no_check).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import torch
import torch.nn as nn
from pyhocon import ConfigFactory

from model import net_dict

PAD_LEN = 9


class VelNetONNX(nn.Module):
    """acc/gyro(/airspeed) -> vel_body(/vel_cov), with the collate pad inside."""

    def __init__(self, net, pad=True):
        super().__init__()
        self.net = net
        self.pad = pad
        self.use_airspeed = bool(getattr(net, "use_airspeed", False))
        self.propcov = bool(net.conf.propcov)

    def _pad(self, x):
        # padding9_honest: repeat the first real sample (datasets/dataset_utils.py).
        if not self.pad:
            return x
        return torch.cat([x[:, :1].expand(-1, PAD_LEN, -1), x], dim=1)

    def forward(self, acc, gyro, airspeed=None):
        data = {"acc": self._pad(acc), "gyro": self._pad(gyro)}
        if self.use_airspeed:
            data["airspeed"] = self._pad(airspeed)
        out = self.net.inference(data)
        if self.propcov:
            return out["vel_body"], out["vel_cov"]
        return out["vel_body"]


def build(config, ckpt):
    conf = ConfigFactory.parse_file(config)
    conf.train.device = "cpu"
    net = net_dict[conf.train.network](conf.train)
    if str(net.conf.get("network", "velnet")) != "velnet" or not hasattr(net, "vel_decoder"):
        sys.exit("only velnet checkpoints can be exported")
    if getattr(net, "att_input", "none") != "none":
        sys.exit("att_input=%r needs an attitude input; only att_input none is exportable"
                 % net.att_input)
    if ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(ck.get("model_state_dict", ck))
        print("[export] loaded %s (epoch %s)" % (ckpt, ck.get("epoch", "?")))
    else:
        print("[export] WARNING: no --ckpt, exporting RANDOM weights (test only)")
        # The heads' last layers are zero-initialised, so an untrained net outputs a
        # constant and the ONNX check would compare constants.  Randomise them.
        g = torch.Generator().manual_seed(0)
        for head in (net.vel_decoder, net.velcov_decoder):
            with torch.no_grad():
                head[-1].weight.copy_(0.1 * torch.randn(head[-1].weight.shape, generator=g))
    collate = conf.dataset.get("collate", "base")
    if collate != "padding9_honest":
        print("[export] WARNING: config collate is %r, the graph pads like "
              "padding9_honest (repeat first sample)" % collate)
    return net.float().eval()


def example_inputs(model, batch, frames, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Physically plausible scale: gravity on z plus noise, small rates.
    acc = torch.randn(batch, frames, 3, generator=g) * 2.0
    acc[..., 2] += 9.81
    gyro = torch.randn(batch, frames, 3, generator=g) * 0.2
    args = [acc, gyro]
    if model.use_airspeed:
        args.append(22.0 + 3.0 * torch.randn(batch, frames, 1, generator=g))
    return tuple(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="checkpoint; omit to export random weights")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--frames", type=int, default=6000,
                    help="FIXED window length in frames at 100 Hz (default 6000 = 60 s)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no_pad", action="store_true",
                    help="do not put the 9-frame pad in the graph (input F+9 frames)")
    ap.add_argument("--no_check", action="store_true",
                    help="skip the onnxruntime vs PyTorch comparison")
    ap.add_argument("--atol", type=float, default=1e-3,
                    help="max allowed |onnx - torch| on vel_body, m/s (default 1e-3)")
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    model = VelNetONNX(build(a.config, a.ckpt), pad=not a.no_pad).eval()
    in_frames = a.frames + (PAD_LEN if a.no_pad else 0)

    inputs = example_inputs(model, 1, in_frames)
    in_names = ["acc", "gyro"] + (["airspeed"] if model.use_airspeed else [])
    out_names = ["vel_body"] + (["vel_cov"] if model.propcov else [])
    dyn = {k: {0: "batch"} for k in in_names + out_names}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    torch.onnx.export(model, inputs, a.out, input_names=in_names, output_names=out_names,
                      dynamic_axes=dyn, opset_version=a.opset, dynamo=False)
    print("[export] wrote %s (%.1f MB, %.1f s), window %d frames = %.1f s"
          % (a.out, os.path.getsize(a.out) / 1e6, time.time() - t0, a.frames, a.frames / 100.0))

    import onnx
    onnx.checker.check_model(onnx.load(a.out))
    print("[export] onnx.checker: OK")

    if a.no_check:
        return
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    worst = 0.0
    for batch, seed in ((1, 1), (2, 2)):          # batch 2 exercises the dynamic axis
        x = example_inputs(model, batch, in_frames, seed=seed)
        ref = model(*x)
        ref = ref if isinstance(ref, tuple) else (ref,)
        t0 = time.time()
        got = sess.run(None, {k: v.numpy() for k, v in zip(in_names, x)})
        dt = time.time() - t0
        for name, r, o in zip(out_names, ref, got):
            if tuple(r.shape) != o.shape:
                sys.exit("[check] FAIL %s shape torch %s vs onnx %s"
                         % (name, tuple(r.shape), o.shape))
            diff = float(np.abs(r.numpy() - o).max())
            rel = diff / max(float(np.abs(r.numpy()).max()), 1e-12)
            print("[check] batch %d  %-8s shape %s  max|diff| %.3g  (rel %.2g)"
                  % (batch, name, o.shape, diff, rel))
            if name == "vel_body":
                worst = max(worst, diff)
        print("[check] batch %d  onnxruntime %.1f ms" % (batch, dt * 1e3))
    if worst > a.atol:
        sys.exit("[check] FAIL: vel_body differs by %.3g m/s > atol %.3g" % (worst, a.atol))
    print("[check] PASS: onnxruntime matches PyTorch (vel_body max diff %.3g m/s)" % worst)


if __name__ == "__main__":
    main()
