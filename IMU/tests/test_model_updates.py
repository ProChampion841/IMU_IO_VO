"""Checks for the 2026-09-30 model / loss updates.

Run from the IMU folder:  python -m pytest tests/test_model_updates.py -q
CPU only, no data needed: the network is built from configs/exp/UAV/*.conf and fed
synthetic windows.
"""
import os
import sys

import pypose as pp
import pytest
import torch
from pyhocon import ConfigFactory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import net_dict                      # noqa: E402
from model.hybrid import HybridNet              # noqa: E402
from model.loss_func import diag_cov_loss       # noqa: E402
from model.losses import get_loss               # noqa: E402

F = 510        # window frames (2 chunks of CHUNK)
# pypose 0.9.5's cumprod calls torch.arange(i, L) with i up to the next power of two
# above L; recent torch rejects i > L (older torch returned an empty range).  pypose
# prepends the initial state, so L = chunk + 1 must be a power of two: chunk 255.
CHUNK = 255


def _conf(name="tilt_aware", **over):
    c = ConfigFactory.parse_file(os.path.join(ROOT, "configs/exp/UAV/%s.conf" % name)).train
    c.put("cov_sampling", CHUNK)
    for k, v in over.items():
        c.put(k, v)
    return c


def _net(name="tilt_aware", seed=0, **over):
    torch.manual_seed(seed)
    return net_dict["hybridnet"](_conf(name, **over)).eval()


def _batch(B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    acc = torch.randn(B, F + 9, 3, generator=g) * 0.5
    acc[..., 2] += 9.81
    gyro = torch.randn(B, F + 9, 3, generator=g) * 0.02
    rot = pp.so3(torch.randn(B, F, 3, generator=g) * 0.1).Exp()
    data = {"acc": acc, "gyro": gyro, "rot": rot, "dt": torch.full((B, F, 1), 0.01)}
    init = {"pos": torch.zeros(B, 1, 3), "vel": torch.zeros(B, 1, 3), "rot": rot[:, :1]}
    label = {"gt_pos": torch.randn(B, F, 3, generator=g), "gt_vel": torch.randn(B, F, 3, generator=g),
             "gt_rot": rot}
    return data, init, label


def _randomise_heads(net):
    for p in net.parameters():
        if p.dim() > 0 and p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.1)


# ---------------------------------------------------------------- rotate mode
def test_rotate_small_matches_pypose_exp():
    th = torch.randn(64, 3) * 0.05
    th[0] = 0.0                          # exact zero goes through the Taylor branch
    th[1] = 1e-6
    v = torch.randn(64, 3) * 10
    ref = pp.so3(th.double()).Exp() @ v.double()
    got = HybridNet._rotate_small(th.double(), v.double())
    assert torch.allclose(got, ref, atol=1e-10)


def test_rotate_small_grad_finite_at_zero():
    th = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(4, 3, dtype=torch.float64)
    HybridNet._rotate_small(th, v).sum().backward()
    assert torch.isfinite(th.grad).all() and th.grad.abs().sum() > 0


def test_rotate_mode_is_identity_at_init_and_bounded():
    net = _net("tilt_rotate")
    data, _, _ = _batch()
    out = net.inference(data)
    assert torch.equal(out["corrected_acc"], data["acc"][:, 9:])
    _randomise_heads(net)
    with torch.no_grad():
        for p in net.accscale_decoder[-1].parameters():
            p.mul_(1e3)                  # saturate the tanh
        out = net.inference(data)
    raw = data["acc"][:, 9:]
    bias_only = out["corrected_acc"] - raw
    # rotation part cannot exceed |raw| * 2 sin(max/2)
    rot_part = bias_only - net._update(torch.zeros_like(raw),
                                       net.accdecoder(net.encoder(net._net_input(data))[:, 1:]) * net.acc_std,
                                       raw.shape[1])
    lim = raw.norm(dim=-1) * 2 * torch.sin(torch.tensor(net.rotate_max / 2)) + 1e-5
    assert (rot_part.norm(dim=-1) <= lim).all()


def test_rotate_mode_cancels_an_attitude_error_exactly():
    """R_used @ Exp(d) @ a == R_true @ a when R_true = R_used Exp(d)."""
    d = torch.tensor([[0.004, -0.006, 0.001]], dtype=torch.float64)
    R_used = pp.so3(torch.tensor([[0.1, 0.2, 1.0]], dtype=torch.float64)).Exp()
    R_true = R_used * pp.so3(d).Exp()
    a = torch.tensor([[0.3, -0.2, 9.8]], dtype=torch.float64)
    assert torch.allclose(R_used @ HybridNet._rotate_small(d, a), R_true @ a, atol=1e-12)


def test_non_rotate_configs_unchanged():
    """tilt_aware builds and runs exactly as before: same params, identity at init."""
    net = _net("tilt_aware")
    assert net.correction_mode == "additive" and not net.causal_cnn and not net.cov_stop_grad
    assert sum(p.numel() for p in net.parameters()) == 139474
    data, _, _ = _batch()
    assert torch.equal(net.inference(data)["corrected_acc"], data["acc"][:, 9:])


# ---------------------------------------------------------------- causal CNN
def _max_lookahead(net):
    _randomise_heads(net)
    data, _, _ = _batch(B=1)
    x = net._net_input(data).requires_grad_(True)
    feat = net.encoder(x)[:, 1:, :]
    fl = data["acc"].shape[1] - 9
    out = net._update(torch.zeros(1, fl, 3), net.accdecoder(feat), fl)
    worst = -10 ** 9
    for f in (0, 1, 8, 9, 100, 257, 450):
        g, = torch.autograd.grad(out[0, f].sum(), x, retain_graph=True)
        dep = (g[0].abs().sum(-1) > 0).nonzero().flatten() - 9     # window frame index
        worst = max(worst, int(dep.max()) - f)
    return worst


def test_default_cnn_looks_ahead():
    assert _max_lookahead(_net("tilt_aware")) >= 10               # measured: 12 frames


def test_causal_cnn_has_no_lookahead():
    net = _net("tilt_aware", causal_cnn=True)
    assert _max_lookahead(net) <= 0


def test_causal_cnn_keeps_shapes():
    a, b = _net("tilt_aware"), _net("tilt_aware", causal_cnn=True)
    data, _, _ = _batch()
    assert a.inference(data)["corrected_acc"].shape == b.inference(data)["corrected_acc"].shape


# ---------------------------------------------------------------- cov_stop_grad
def test_cov_stop_grad_isolates_trunk():
    for flag, expect_grad in ((False, True), (True, False)):
        net = _net("tilt_aware", cov_stop_grad=flag)
        _randomise_heads(net)
        data, _, _ = _batch()
        feat = net.encoder(net._net_input(data))
        net.cov_decoder(feat).sum().backward()
        g = net.cnn.net[0].weight.grad
        has = g is not None and g.abs().sum() > 0
        assert has == expect_grad
        assert net.acccov_decoder[0].weight.grad.abs().sum() > 0


# ---------------------------------------------------------------- loss
def test_diag_cov_loss_is_a_real_nll():
    # minimised at log(sigma) = log|e|, not at sigma -> 0
    e = torch.full((1000,), 0.5)
    s = torch.linspace(-4, 2, 601)
    vals = torch.stack([diag_cov_loss(e, torch.full_like(e, x)) for x in s])
    assert abs(float(s[vals.argmin()]) - float(torch.log(torch.tensor(0.5)))) < 0.02


def _loss(net, **over):
    conf = _conf("tilt_aware", **over)
    conf.put("device", "cpu")
    data, init, label = _batch()
    torch.manual_seed(3)
    with torch.no_grad():
        st = net(data, init)
    return get_loss(st, label, conf)


def test_time_power_zero_is_bit_identical():
    net = _net("tilt_aware")
    a = _loss(net)
    b = _loss(net, loss_time_power=0.0)
    assert torch.equal(a["loss"], b["loss"])


def test_time_power_changes_loss_and_keeps_scale():
    net = _net("tilt_aware")
    a = _loss(net)["loss"].item()
    b = _loss(net, loss_time_power=1.0)["loss"].item()
    assert a != b and 0.3 < b / a < 3.0


def test_full_forward_backward_rotate():
    net = _net("tilt_rotate").train()
    _randomise_heads(net)
    conf = _conf("tilt_rotate", loss_time_power=1.0, causal_cnn=True)
    conf.put("device", "cpu")
    net = net_dict["hybridnet"](conf).train()
    _randomise_heads(net)
    data, init, label = _batch()
    loss = get_loss(net(data, init), label, conf)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert net.accscale_decoder[-1].weight.grad.abs().sum() > 0   # rotation head trains
