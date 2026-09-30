"""A causal, stride-pooled Mamba branch -- the "long memory" half of HybridNet.

Why this module exists
----------------------
The existing AirIMU encoder is a CNN (stride 9) feeding two unidirectional GRUs.
A GRU carries state in a fixed-width vector that is rewritten at every step, so
its practical memory is short; that is fine for the fast, dynamics-driven part
of the IMU error but it is a poor instrument for the part of the error that is
nearly constant over a whole flight.  The measured evidence on this corpus says
the accelerometer bias *direction* is a corpus constant (cosine 0.9652 val /
0.9951 test against the train fit) while its *magnitude* roughly doubles between
flights.  Estimating "which flight am I on, and how big is the bias today" is a
long-horizon inference; it wants a long-horizon operator.

Mamba is a selective state-space model: a linear recurrence whose transition is
input-dependent, which gives it a much longer effective memory than a GRU at
comparable cost.  Running it on *every* frame would be wasteful (the information
we want it to hold changes on a scale of seconds, not centiseconds), so this
branch first pools the token stream down by 'stride', runs Mamba there, then
broadcasts back up.  The GRU branch stays at full rate.  Two operators, two time
scales, combined downstream by HybridNet.

The thing that can silently ruin this
-------------------------------------
Downsample/upsample around a causal operator is an easy place to leak the
future, and the leak is invisible: the model simply trains better than it should
and then fails in deployment.  The naive pooling -- non-overlapping blocks,
token k = mean of frames [kS, kS+S-1], upsampled back onto those same frames --
leaks S-1 frames, because output frame kS is then a function of input frame
kS+S-1.  On this data at stride 8 that is 80 ms of lookahead handed to a network
whose entire job is causal prediction.

The fix used here is a one-block alignment shift, implemented as a *left-only*
pad of S-1 frames before pooling:

    pooled token k  =  mean of input frames [kS-(S-1) .. kS]     (clipped at 0)
    pooled token k  ->  output frames       [kS .. kS+S-1]

so the newest frame any pooled token has seen is exactly kS, which is the
*first* frame that token is broadcast onto.  Output frame t therefore depends on
input frames <= t and never on t+1.  That is the tightest causal alignment
available at this stride -- no wasted lookback, no leak -- and it is asserted
numerically by the alignment and causality checks in the self-test below, which
recover the exact linear support of the pool/upsample map and perturb individual
input tokens through the full branch.

One consequence is worth stating plainly, because it looks like a bug and is not.
The last (T-1) % S input frames of a window influence *no* output inside that
window: a pooled token summarising them would have to land at output index >= T.
Strict causality forces this -- covering frame T-1 at output frame T-1 would mean
that token also spans frames T..T+S-2, which is the leak we just removed.  It is
harmless here because the branch supplies slow-varying context only, and
HybridNet's full-rate GRU branch carries the recent frames; at stride 8 on a
1000-frame window it is at most 7 frames at the very tail.  Between pooled
updates the Mamba context is likewise up to S-1 frames stale, which is simply
what a stride-S slow branch means.

Kernels and portability
-----------------------
mamba-ssm 2.2.5 is installed in this environment but a plain "import mamba_ssm"
raises ImportError: its utils/generation.py imports three names that newer
transformers no longer exports.  We install a three-line shim for those names
and then import mamba_ssm.modules.mamba_simple directly, which works.
causal_conv1d is absent, so the real module takes its own slow path for the
depthwise conv (still causal) and the CUDA selective_scan kernel for the scan.

The CUDA scan is float16/bfloat16/float32 and CUDA-only.  For float64 or CPU we
run _mamba_forward_pure, a compact reference selective scan that reads the
*same* nn.Parameters off the *same* module -- so there is no weight copy to get
out of sync, and the fallback is exercised against the real kernel in the
self-test rather than assumed correct.

Cost: keep the pooled length small (measured, not folklore)
----------------------------------------------------------
The selective_scan *backward* in this installed mamba-ssm build is O(L^2), not
O(L), on this GPU.  Measured in isolation (B=64, d_inner=256, d_state=16, fresh
process each, fwd+bwd):

    L =  14      9.5 ms          L = 124    1058 ms
    L =  31     92.5 ms          L = 496   25154 ms

Doubling L roughly quadruples the time, over a 35x range.  The depthwise conv is
linear and negligible by comparison (0.39 ms at L=14, 3.45 ms at L=496), so the
scan owns essentially all of it.  Forward is linear and fast; only backward
degrades, which means this bites during training and not at inference.

The practical consequence for HybridNet: feed this branch the CNN's token stream
(F=1000 frames -> ~111 tokens), not raw frames.  At stride 8 that pools to K=14
and the whole two-layer branch costs ~20-50 ms/iteration at batch 64.  Feeding it
1000 raw frames instead gives K=124 and costs seconds per iteration for the same
parameters and the same information.  Choose mamba_stride so that
ceil(T/stride) stays in the low tens; the module warns once if it does not.

Run:  python -m model.mamba_block
      (run it as a module, not "python model/mamba_block.py" -- that puts model/
      on sys.path[0], where this repo's code.py shadows the stdlib "code" module
      that pdb imports, and torch's import chain then dies on a circular import.)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Import the real Mamba, via the transformers shim.  Never "import mamba_ssm":
# the package __init__ pulls in the broken generation utils.
# --------------------------------------------------------------------------
MAMBA_IMPORT_ERROR = None
try:
    import transformers.generation as _tg
    for _n in ["GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput", "TextStreamer"]:
        if not hasattr(_tg, _n):
            setattr(_tg, _n, type(_n, (object,), {}))
    from mamba_ssm.modules.mamba_simple import Mamba as _RealMamba
    _HAS_MAMBA = True
except Exception as e:                                    # pragma: no cover
    _RealMamba = None
    _HAS_MAMBA = False
    MAMBA_IMPORT_ERROR = "%s: %s" % (type(e).__name__, e)


def has_mamba():
    """True if the real mamba-ssm kernel imported successfully."""
    return _HAS_MAMBA


# --------------------------------------------------------------------------
# Pure-PyTorch selective scan.  Mirrors mamba_ssm's reference implementation
# (variable B/C, real-valued A, softplus delta, D skip, silu(z) gate).
# --------------------------------------------------------------------------
def _selective_scan_ref(u, delta, A, B, C, D, z, delta_bias):
    """u,delta,z: (b,d,l)   A: (d,n)   B,C: (b,n,l)   D: (d,)   -> y: (b,d,l)"""
    delta = F.softplus(delta + delta_bias.unsqueeze(-1))              # (b,d,l)
    # einsum, not broadcasting: A is (d,n) and would right-align its n against l.
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))        # (b,d,l,n)
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)         # (b,d,l,n)

    b, d, l, n = deltaA.shape
    h = torch.zeros(b, d, n, dtype=u.dtype, device=u.device)
    ys = []
    for t in range(l):
        h = deltaA[:, :, t] * h + deltaB_u[:, :, t]                   # (b,d,n)
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, :, t]))
    y = torch.stack(ys, dim=2)                                        # (b,d,l)
    y = y + u * D.unsqueeze(-1)
    return y * F.silu(z)


def _mamba_forward_pure(mod, hidden_states):
    """Run a Mamba-layout module in pure PyTorch, reading its own parameters.

    Works for both mamba_ssm's Mamba and _MambaPure, so the fallback and the
    kernel are always evaluated on identical weights.
    """
    L = hidden_states.shape[1]
    cdt = hidden_states.dtype if hidden_states.dtype in (torch.float32, torch.float64) else torch.float32

    xz = mod.in_proj(hidden_states).transpose(1, 2)                   # (b, 2*d_inner, l)
    x, z = xz.chunk(2, dim=1)
    x = F.silu(mod.conv1d(x)[..., :L])                                # causal: pad=d_conv-1, truncate

    x_dbl = mod.x_proj(x.transpose(1, 2))                             # (b, l, dt_rank+2n)
    dt, Bm, Cm = torch.split(x_dbl, [mod.dt_rank, mod.d_state, mod.d_state], dim=-1)
    dt = F.linear(dt, mod.dt_proj.weight)                             # bias enters as delta_bias

    A = -torch.exp(mod.A_log.to(cdt))
    y = _selective_scan_ref(
        x.to(cdt), dt.transpose(1, 2).to(cdt), A,
        Bm.transpose(1, 2).to(cdt), Cm.transpose(1, 2).to(cdt),
        mod.D.to(cdt), z.to(cdt), mod.dt_proj.bias.to(cdt),
    )
    return mod.out_proj(y.transpose(1, 2).to(hidden_states.dtype))


class _MambaPure(nn.Module):
    """Parameter-identical stand-in for mamba_ssm's Mamba, pure PyTorch."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4, conv_bias=True, bias=False):
        super().__init__()
        self.d_model, self.d_state, self.d_conv, self.expand = d_model, d_state, d_conv, expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # dt bias initialised so softplus(bias) lands in [dt_min, dt_max] -- same as upstream.
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states, inference_params=None):
        return _mamba_forward_pure(self, hidden_states)


def _make_mamba(d_model, d_state, d_conv, expand):
    if _HAS_MAMBA:
        return _RealMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    return _MambaPure(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


def _run_mamba(mod, x):
    """Dispatch to the CUDA kernel when it applies, else the pure reference."""
    kernel_ok = (_HAS_MAMBA and not isinstance(mod, _MambaPure)
                 and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32))
    return mod(x) if kernel_ok else _mamba_forward_pure(mod, x)


# --------------------------------------------------------------------------
# Causal pooling / upsampling
# --------------------------------------------------------------------------
def causal_avg_pool(x, stride):
    """(B,T,C) -> (B,K,C), K = ceil(T/stride).

    Token k is the mean of input frames [k*stride-(stride-1) .. k*stride], clipped
    at 0.  Left-only padding, and the pad is excluded from the mean (so token 0 is
    exactly frame 0, not frame 0 divided by 'stride').
    """
    if stride == 1:
        return x
    T = x.shape[1]
    xt = x.transpose(1, 2)                                            # (B,C,T)
    xp = F.pad(xt, (stride - 1, 0))
    tot = F.avg_pool1d(xp, kernel_size=stride, stride=stride) * stride
    ones = F.pad(torch.ones(1, 1, T, dtype=x.dtype, device=x.device), (stride - 1, 0))
    cnt = F.avg_pool1d(ones, kernel_size=stride, stride=stride) * stride
    return (tot / cnt.clamp(min=1)).transpose(1, 2)


def causal_upsample(y, stride, T):
    """(B,K,C) -> (B,T,C).  Token k lands on output frames [k*stride .. k*stride+stride-1]."""
    if stride == 1:
        return y[:, :T, :]
    return y.repeat_interleave(stride, dim=1)[:, :T, :]


# Pooled length above which the O(L^2) selective_scan backward starts to hurt.
_LONG_SEQ_WARN = 160


# --------------------------------------------------------------------------
# The branch
# --------------------------------------------------------------------------
class MambaBranch(nn.Module):
    """Causal long-horizon branch: causal pool -> project -> Mamba xN -> upsample.

    forward(x: (B, T, d_in)) -> (B, T, d_model).  T need not divide 'stride'.
    """

    _warned = False

    def __init__(self, d_in, d_model, n_layer=2, stride=8, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_in, self.d_model, self.stride, self.n_layer = d_in, d_model, stride, n_layer
        self.proj_in = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([_make_mamba(d_model, d_state, d_conv, expand)
                                     for _ in range(n_layer)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        T = x.shape[1]
        h = self.proj_in(causal_avg_pool(x, self.stride))
        if h.shape[1] > _LONG_SEQ_WARN and not MambaBranch._warned:
            MambaBranch._warned = True
            print("[mamba_block] WARNING: pooled length %d (T=%d, stride=%d). The installed "
                  "selective_scan backward is O(L^2) on this box; training will be slow. "
                  "Feed CNN tokens rather than raw frames, or raise mamba_stride."
                  % (h.shape[1], T, self.stride))
        for mamba, norm in zip(self.layers, self.norms):
            h = h + _run_mamba(mamba, norm(h))                        # pre-norm residual
        h = self.norm_out(h)
        return causal_upsample(h, self.stride, T)

    def extra_repr(self):
        return "d_in=%d, d_model=%d, n_layer=%d, stride=%d, kernel=%s" % (
            self.d_in, self.d_model, self.n_layer, self.stride,
            "mamba-ssm" if (_HAS_MAMBA and not isinstance(self.layers[0], _MambaPure)) else "pure-pytorch")


print("[mamba_block] kernel: %s" % ("mamba-ssm (CUDA selective_scan)" if _HAS_MAMBA
                                    else "pure-PyTorch fallback (%s)" % MAMBA_IMPORT_ERROR))


# ==========================================================================
# Self-test
# ==========================================================================
def _selftest():
    import time
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        return ok

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: %s | torch %s | has_mamba(): %s" % (dev, torch.__version__, has_mamba()))
    if not has_mamba():
        print("  import error was: %s" % MAMBA_IMPORT_ERROR)
    print()

    # --- 1. shapes, including T not divisible by stride ------------------
    S = 8
    ok_shape, detail = True, []
    for T in (1000, 991, 7, 8, 9, 1):
        br = MambaBranch(9, 32, n_layer=1, stride=S).to(dev).eval()
        with torch.no_grad():
            y = br(torch.randn(2, T, 9, device=dev))
        good = tuple(y.shape) == (2, T, 32)
        ok_shape &= good
        detail.append("T=%d->%s" % (T, "ok" if good else "BAD %s" % (tuple(y.shape),)))
    check("shape (B,T,d_in)->(B,T,d_model), T%stride!=0", ok_shape, ", ".join(detail[:4]))

    # --- 2. pool/upsample alignment: recover the exact linear support ----
    # The map is linear, so feed unit impulses and read the response matrix.
    T, S = 20, 8
    M = torch.zeros(T, T)
    for j in range(T):
        e = torch.zeros(1, T, 1)
        e[0, j, 0] = 1.0
        M[:, j] = causal_upsample(causal_avg_pool(e, S), S, T)[0, :, 0]

    align_ok, first_bad = True, None
    for t in range(T):
        k = t // S
        lo, hi = max(0, k * S - (S - 1)), k * S
        want = set(range(lo, hi + 1))
        got = set(torch.nonzero(M[t]).flatten().tolist())
        w = 1.0 / len(want)
        vals_ok = all(abs(M[t, j].item() - w) < 1e-6 for j in want)
        if got != want or not vals_ok:
            align_ok = False
            first_bad = first_bad or "t=%d want %s got %s" % (t, sorted(want), sorted(got))
    check("pool/upsample support == [kS-(S-1) .. kS], uniform weights",
          align_ok, first_bad or "all %d rows exact" % T)

    tri_ok = bool((torch.triu(M, diagonal=1).abs() < 1e-9).all())
    maxlag = max((t - min(torch.nonzero(M[t]).flatten().tolist()) for t in range(T)))
    check("pool/upsample map is lower-triangular (no future in the map)",
          tri_ok, "max lookahead 0 frames, max lookback %d frames" % maxlag)

    # The price of strict causality: the final S-1 frames summarise into a pooled
    # token that would land at output index >= T, so nothing in-window sees them.
    dead = torch.nonzero(M.sum(0) == 0).flatten().tolist()
    check("tail frames unseen in-window == exactly the last (T-1)%S frames",
          dead == list(range(T - ((T - 1) % S), T)) if (T - 1) % S else dead == [],
          "T=%d S=%d -> %d unseen tail frames %s" % (T, S, len(dead),
                                                     (str(dead[:3]) + ".." if dead else "none")))

    # --- 3. end-to-end causality through the full branch -----------------
    torch.manual_seed(1)
    T, S = 96, 8
    br = MambaBranch(6, 32, n_layer=2, stride=S).to(dev).double().eval()
    x = torch.randn(1, T, 6, device=dev, dtype=torch.float64)
    with torch.no_grad():
        y0 = br(x)
    caus_ok, rows = True, []
    for t in (0, 1, 7, 8, 9, 33, 64, 95):
        xp = x.clone()
        xp[0, t] += 5.0
        with torch.no_grad():
            d = (br(xp) - y0).abs().max(dim=-1).values[0]             # (T,)
        before = d[:t].max().item() if t > 0 else 0.0
        nz = torch.nonzero(d > 1e-12).flatten()
        first = int(nz[0]) if len(nz) else -1
        expect = ((t + S - 1) // S) * S                               # ceil(t/S)*S
        if expect >= T:
            expect = -1     # tail frame: no in-window output may legally see it
        good = (before == 0.0) and (first == expect)
        caus_ok &= good
        rows.append((t, before, first, expect, good))
    check("perturb token t -> zero change at every index < t", caus_ok,
          "max |dy| below t = %.1e over 8 probes" % max(r[1] for r in rows))
    check("first changed index == ceil(t/S)*S (alignment is tight, not merely safe)",
          all(r[2] == r[3] for r in rows), "")
    print("    perturbation detail:")
    print("      %4s %14s %12s %10s" % ("t", "max|dy| idx<t", "first change", "expected"))
    for t, before, first, expect, good in rows:
        print("      %4d %14.1e %12d %10d   %s" % (t, before, first, expect, "ok" if good else "FAIL"))
    print()

    # --- 4. pure fallback vs real CUDA kernel, same weights --------------
    if has_mamba() and torch.cuda.is_available():
        torch.manual_seed(2)
        m = _RealMamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda().eval()
        xf = torch.randn(2, 125, 64, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            y_k = m(xf)                     # CUDA selective_scan
            y_p = _mamba_forward_pure(m, xf)  # same nn.Parameters, pure PyTorch
        amax = (y_k - y_p).abs().max().item()
        rmax = ((y_k - y_p).abs() / (y_k.abs() + 1e-6)).max().item()
        rfro = ((y_k - y_p).norm() / y_k.norm()).item()
        check("pure fallback matches CUDA kernel on identical weights", amax < 1e-4,
              "max|abs| %.3e | rel Frobenius %.3e | max elementwise rel %.3e" % (amax, rfro, rmax))
    else:
        check("pure fallback matches CUDA kernel", False, "SKIPPED (no cuda / no mamba)")

    # --- 4b. the _MambaPure class itself (never built when the kernel is
    #         present, so exercise it explicitly) + checkpoint portability ---
    if has_mamba():
        torch.manual_seed(5)
        real = _RealMamba(d_model=64, d_state=16, d_conv=4, expand=2).to(dev).eval()
        pure = _MambaPure(d_model=64, d_state=16, d_conv=4, expand=2).to(dev).eval()
        same_keys = (sorted(real.state_dict().keys()) == sorted(pure.state_dict().keys()))
        shapes_ok = all(real.state_dict()[k].shape == pure.state_dict()[k].shape
                        for k in real.state_dict())
        missing = pure.load_state_dict(real.state_dict(), strict=True)
        xf = torch.randn(2, 40, 64, device=dev)
        with torch.no_grad():
            d_pp = (pure(xf) - _mamba_forward_pure(real, xf)).abs().max().item()
        check("_MambaPure is state_dict-compatible with mamba-ssm Mamba",
              same_keys and shapes_ok and d_pp == 0.0,
              "%d keys identical, max|diff| after load_state_dict = %.1e" % (len(real.state_dict()), d_pp))
    else:
        check("_MambaPure is state_dict-compatible with mamba-ssm Mamba", False,
              "SKIPPED (mamba-ssm not importable here)")

    # A MambaBranch built either way must expose the same parameter names, so a
    # checkpoint trained on this box loads on a machine without the kernel.
    # NB: patch the *running* module. Under "python -m model.mamba_block" the live
    # copy is __main__, and "import model.mamba_block" would make a second, separate
    # module object -- patching that one silently tests nothing.
    import sys as _sys
    _mod = _sys.modules[__name__]
    _saved = _mod._HAS_MAMBA
    try:
        _mod._HAS_MAMBA = False
        br_pure = MambaBranch(9, 64, n_layer=2, stride=8)
    finally:
        _mod._HAS_MAMBA = _saved
    br_real = MambaBranch(9, 64, n_layer=2, stride=8)
    used_fallback = isinstance(br_pure.layers[0], _MambaPure)
    keys_match = sorted(br_pure.state_dict().keys()) == sorted(br_real.state_dict().keys())
    n_pure = sum(p.numel() for p in br_pure.parameters())
    n_real = sum(p.numel() for p in br_real.parameters())
    check("MambaBranch state_dict identical with and without the kernel",
          used_fallback and keys_match and n_pure == n_real,
          "%d keys, %d params both ways; fallback class really used: %s"
          % (len(br_real.state_dict()), n_real, used_fallback))

    # --- 5. dtype: float32 and float64, forward + backward ---------------
    for dt_name, dt in (("float32", torch.float32), ("float64", torch.float64)):
        torch.manual_seed(3)
        br = MambaBranch(9, 64, n_layer=2, stride=8).to(dev).to(dt)
        xx = torch.randn(2, 128, 9, device=dev, dtype=dt, requires_grad=True)
        t0 = time.time()
        y = br(xx)
        loss = y.pow(2).mean()
        loss.backward()
        dtms = (time.time() - t0) * 1e3
        gsum = sum(p.grad.abs().sum().item() for p in br.parameters() if p.grad is not None)
        path = "kernel" if (has_mamba() and dev == "cuda" and dt == torch.float32) else "pure"
        ok = (y.dtype == dt and torch.isfinite(y).all() and xx.grad is not None
              and torch.isfinite(xx.grad).all() and gsum > 0)
        check("%s fwd+bwd (%s path)" % (dt_name, path), ok,
              "out dtype %s, grad-sum %.3e, %.0f ms" % (y.dtype, gsum, dtms))
        dead = [n for n, p in br.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
        check("%s: every parameter receives gradient" % dt_name, not dead,
              "no dead params among %d tensors" % len(list(br.parameters())) if not dead
              else "DEAD: %s" % dead[:4])

    # --- 6. CPU forward (kernel is CUDA-only; must not crash) ------------
    torch.manual_seed(4)
    br = MambaBranch(9, 32, n_layer=2, stride=8).cpu().eval()
    with torch.no_grad():
        yc = br(torch.randn(1, 64, 9))
    check("CPU forward routes to pure path", torch.isfinite(yc).all().item(),
          "shape %s" % (tuple(yc.shape),))

    # --- 6b. cost at the intended operating point ------------------------
    if dev == "cuda":
        br = MambaBranch(64, 128, n_layer=2, stride=8).cuda()
        xb = torch.randn(64, 111, 64, device="cuda", requires_grad=True)
        for _ in range(3):
            br(xb).pow(2).mean().backward()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            br(xb).pow(2).mean().backward()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 5 * 1e3
        check("cost at intended shape (B=64, T=111 CNN tokens, stride 8 -> K=14)",
              ms < 250, "%.1f ms/iter fwd+bwd, peak %.0f MB"
              % (ms, torch.cuda.max_memory_allocated() / 2 ** 20))

    # --- 7. parameter count for the default configuration ----------------
    print("  parameter counts (n_layer=2, stride=8, d_state=16, d_conv=4, expand=2):")
    for d_in in (6, 9, 13, 64):
        for d_model in (128,):
            br = MambaBranch(d_in, d_model, n_layer=2, stride=8)
            n = sum(p.numel() for p in br.parameters())
            print("      d_in=%-3d d_model=%-4d ->  %8d params" % (d_in, d_model, n))
    default = sum(p.numel() for p in MambaBranch(64, 128, n_layer=2, stride=8).parameters())
    print("    default (d_in=64, d_model=128): %d params" % default)
    print()

    # --- table -----------------------------------------------------------
    print("  %-62s %-6s %s" % ("check", "result", "detail"))
    print("  " + "-" * 118)
    for name, ok, detail in results:
        print("  %-62s %-6s %s" % (name, "PASS" if ok else "FAIL", detail))
    print("  " + "-" * 118)
    npass = sum(1 for _, o, _ in results if o)
    print("  %d/%d passed" % (npass, len(results)))
    return all(o for _, o, _ in results)


if __name__ == "__main__":
    _selftest()
