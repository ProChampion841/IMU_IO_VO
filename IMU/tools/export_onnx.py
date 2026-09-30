"""Export the HybridNet IMU-correction network to ONNX, and verify it.

WHAT IS EXPORTED.  The NEURAL NETWORK only: raw IMU in, corrected IMU (and its
uncertainty) out.  The pypose preintegrator that turns corrected IMU into
velocity/position is NOT in the graph -- it is plain strapdown integration and is
run by the host (it also uses pypose LieTensors, which ONNX cannot represent).

    inputs  (float32)
      acc      (B, N, 3)   accelerometer, m/s^2, body FLU.  N = frames + 9
      gyro     (B, N, 3)   gyroscope, rad/s
      g_body   (B, N, 3)   world "up" in the body frame = R^T @ [0,0,1], from the
                           attitude the model was trained with (att_source).
                           Only present when att_input != none.
      airspeed (B, N, 1)   m/s, only when use_airspeed is on
    outputs
      corrected_acc  (B, frames, 3)
      corrected_gyro (B, frames, 3)
      acc_cov        (B, frames, 3)   only when propcov is on
      gyro_cov       (B, frames, 3)   only when propcov is on

THE 9 EXTRA FRAMES.  The network looks back one CNN token (interval = 9 frames)
before the first output frame -- the `padding9` collate supplies them in training.
So input sample 9+k corresponds to output frame k.  At runtime feed the 9 samples
BEFORE the window (the real history).  To reproduce the training padding exactly,
repeat the first attitude's g_body and use init_rot^T @ [0,0,g] for acc, 0 gyro.

FIXED LENGTH.  The graph is exported for one window length (--frames).  The GRU
chunking and the Mamba scan are unrolled at that length, so feed exactly N frames.
Batch is dynamic.  Export once per length you need (e.g. 3000 / 6000 / 12000).

Usage (from the IMU folder):
    python -m tools.export_onnx --config configs/exp/UAV/tilt_rotate.conf \
        --ckpt experiments/UAV/tilt_rotate/ckpt/best_model.ckpt \
        --frames 6000 --out tilt_rotate_6000.onnx

It always runs a check afterwards: ONNX Runtime vs PyTorch on random windows, and
fails loudly if they disagree beyond --atol.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import torch
from pyhocon import ConfigFactory

from model import net_dict
from model.attitude import gravity_feature
from model.hybrid import HybridNet


class OnnxHybrid(torch.nn.Module):
    """Tensor-in, tensor-out wrapper around HybridNet.inference_from_input."""

    def __init__(self, net):
        super().__init__()
        if not isinstance(net, HybridNet):
            raise TypeError("export_onnx supports network: hybridnet only (got %s)"
                            % type(net).__name__)
        self.net = net
        self.has_att = net.att_input != "none"
        self.has_air = bool(net.use_airspeed)
        self.has_cov = bool(net.conf.propcov)

    def input_names(self):
        return (["acc", "gyro"] + (["g_body"] if self.has_att else [])
                + (["airspeed"] if self.has_air else []))

    def output_names(self):
        return (["corrected_acc", "corrected_gyro"]
                + (["acc_cov", "gyro_cov"] if self.has_cov else []))

    def forward(self, acc, gyro, *extra):
        net = self.net
        extra = list(extra)
        ch = [acc, gyro]
        if self.has_att:
            ch.append(gravity_feature(extra.pop(0), net.att_input))
        if self.has_air:
            ch.append(extra.pop(0))
        # identical to HybridNet._net_input once the channels exist
        net_in = torch.cat(ch, dim=-1)
        if net.normalize_input:
            net_in = (net_in - net.in_offset.to(net_in.dtype)) / net.in_scale.to(net_in.dtype)
        out = net.inference_from_input(net_in, acc, gyro)
        res = [out["corrected_acc"], out["corrected_gyro"]]
        if self.has_cov:
            res += [out["cov_state"]["acc_cov"], out["cov_state"]["gyro_cov"]]
        return tuple(res)


def build(config, ckpt=None):
    conf = ConfigFactory.parse_file(config)
    conf.train.put("device", "cpu")
    net = net_dict[conf.train.network](conf.train)
    if ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(ck.get("model_state_dict", ck))
        print("[ckpt] %s (epoch %s)" % (ckpt, ck.get("epoch", "?")))
    return OnnxHybrid(net.float().eval()).eval()


def randomise_zero_heads(model, std=0.1, seed=0):
    """The correction heads are zero-initialised, so an untrained net is the exact
    identity and a torch-vs-ONNX check on it would pass trivially.  Give them
    random values so the check exercises the whole graph."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 0 and p.abs().sum() == 0:
                p.copy_(torch.randn(p.shape, generator=g) * std)


def example_inputs(model, batch, frames, seed=0):
    """Plausible random windows: level-ish flight, unit g_body."""
    g = torch.Generator().manual_seed(seed)
    n = frames + model.net.interval
    up = torch.tensor([0.0, 0.0, 1.0]) + 0.1 * torch.randn(batch, n, 3, generator=g)
    up = up / up.norm(dim=-1, keepdim=True)
    acc = 9.81 * up + 0.5 * torch.randn(batch, n, 3, generator=g)
    gyro = 0.05 * torch.randn(batch, n, 3, generator=g)
    xs = [acc, gyro]
    if model.has_att:
        xs.append(up)
    if model.has_air:
        xs.append(22.0 + 3.0 * torch.randn(batch, n, 1, generator=g))
    return tuple(xs)


def export(model, frames, out, opset=17, batch=2):
    xs = example_inputs(model, batch, frames)
    dyn = {k: {0: "batch"} for k in model.input_names() + model.output_names()}
    kw = dict(input_names=model.input_names(), output_names=model.output_names(),
              dynamic_axes=dyn, opset_version=opset, do_constant_folding=True)
    with torch.no_grad():
        try:
            # the TorchScript exporter: dynamo=True (default in recent torch) would
            # specialise the Python loops differently and needs onnxscript
            torch.onnx.export(model, xs, out, dynamo=False, **kw)
        except TypeError:                     # torch < 2.5 has no `dynamo` argument
            torch.onnx.export(model, xs, out, **kw)
    import onnx
    onnx.checker.check_model(onnx.load(out))
    print("[onnx] wrote %s (%.2f MB), opset %d, inputs %s, outputs %s"
          % (out, os.path.getsize(out) / 1e6, opset, model.input_names(),
             model.output_names()))


def verify(model, onnx_path, frames, batches=(1, 3), atol=1e-4, seed=1):
    """ONNX Runtime vs PyTorch on fresh random windows.  Returns max abs diff."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    worst = 0.0
    for b in batches:
        xs = example_inputs(model, b, frames, seed=seed + b)
        with torch.no_grad():
            ref = model(*xs)
        t0 = time.time()
        got = sess.run(None, {k: v.numpy() for k, v in zip(model.input_names(), xs)})
        dt = time.time() - t0
        for name, r, o in zip(model.output_names(), ref, got):
            if tuple(r.shape) != o.shape:
                raise AssertionError("%s: shape %s vs torch %s" % (name, o.shape, tuple(r.shape)))
            d = float(np.abs(r.numpy() - o).max())
            # the cov heads are exp(.), so compare them relatively
            if name.endswith("_cov"):
                d = float((np.abs(r.numpy() - o) / np.abs(r.numpy()).clip(1e-12)).max())
            worst = max(worst, d)
            print("  batch %d  %-15s shape %-16s max |diff| %.2e%s"
                  % (b, name, str(o.shape), d, " (relative)" if name.endswith("_cov") else ""))
        print("  batch %d  onnxruntime %.1f ms" % (b, 1e3 * dt))
    ok = worst <= atol
    print("[verify] %s: worst difference %.2e (tolerance %.0e)"
          % ("PASS" if ok else "FAIL", worst, atol))
    if not ok:
        raise SystemExit(1)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="the TRAINING config")
    ap.add_argument("--ckpt", default=None, help="checkpoint to export (required "
                    "unless --random_weights)")
    ap.add_argument("--random_weights", action="store_true",
                    help="export an untrained network -- plumbing test only")
    ap.add_argument("--frames", type=int, default=6000,
                    help="output window length in frames (100 Hz); input is frames+9")
    ap.add_argument("--out", default=None, help="default: <config name>_<frames>.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--atol", type=float, default=1e-4)
    a = ap.parse_args()
    if not a.ckpt and not a.random_weights:
        sys.exit("give --ckpt, or --random_weights for a plumbing test")
    out = a.out or "%s_%d.onnx" % (os.path.splitext(os.path.basename(a.config))[0], a.frames)
    model = build(a.config, a.ckpt)
    if a.random_weights:
        print("[onnx] WARNING: exporting RANDOM weights -- not a usable model")
        randomise_zero_heads(model)
    export(model, a.frames, out, opset=a.opset)
    verify(model, out, a.frames, atol=a.atol)


if __name__ == "__main__":
    main()
