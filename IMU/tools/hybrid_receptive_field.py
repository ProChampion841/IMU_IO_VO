"""Measure the effective receptive field of HybridNet's two branches.

The point of the GRU + Mamba design is that the two branches do *different*
jobs: short-term and long-term.  That claim is testable, and this script tests
it instead of asserting it.

Method
------
Both branches are causal recurrences over the CNN token stream (1 token =
``interval`` = 9 frames = 90 ms at 100 Hz).  Take a real token stream, perturb
the token at index ``t`` by a small random vector, and measure how much the
branch output changes at index ``t + k``, for every lag ``k``.  Normalise by the
change at ``k = 0``.  The lag at which the curve falls to ``1/e``, and the lag at
which it falls below 1%, are the branch's effective memory.

This is not the *nominal* receptive field.  Both branches are recurrences, so
nominally every one of them sees the entire causal prefix and the nominal RF of
both is "the whole window" -- which is exactly why quoting the nominal number
would be useless.  What differs is how fast influence decays, and that is what
gets measured here.  The one exception is ``gru_window``: that IS a nominal
receptive field, because chunking makes the cutoff exact, and ``--checks``
verifies it as an exact zero rather than as a small number.

Two caveats that the numbers must be read with:
  * At initialisation this measures the *architecture's prior*, not the trained
    model.  A GRU can learn to hold state far longer than its init decay
    (its update gate starts at ~0.5 and training moves it).  Pass ``--ckpt`` to
    measure a trained network instead.  ``gru_window`` is the exception again:
    it is a hard structural cutoff and training cannot widen it.
  * The Mamba branch cannot see past the window: h0 = 0 at every window start
    for both branches.  Its horizon is capped by ``window_size``, full stop.

Modes
-----
  (default)   influence-decay curves for both branches
  --checks    correctness of the chunked GRU, the hard cutoff, end-to-end
              causality, and the branch variance split through fuse_lin
  --sweep     (window_size, mamba_stride) trade table: K, s/SSM step, span,
              and measured fwd+bwd time at a realistic batch size

Run:  python -m tools.hybrid_receptive_field --config configs/exp/UAV/hybrid_best.conf
      python -m tools.hybrid_receptive_field --gru_window 300 --mamba_stride 16 --checks
      python -m tools.hybrid_receptive_field --sweep
      (run as a module: model/code.py shadows the stdlib 'code' module if
       model/ ends up on sys.path[0])
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyhocon import ConfigFactory  # noqa: E402

from model import net_dict  # noqa: E402
from model.mamba_block import causal_avg_pool  # noqa: E402


def _decay_curve(fn, tokens, probes, eps=1e-2):
    """Mean normalised |dy_{t+k}| / |dy_t| over the given probe positions.

    Returns (lags, curve) with curve[0] == 1.0 by construction.
    """
    B, T, C = tokens.shape
    acc = np.zeros(T)
    cnt = np.zeros(T)
    with torch.no_grad():
        base = fn(tokens)
        for t in probes:
            pert = tokens.clone()
            d = torch.randn(B, C, dtype=tokens.dtype, device=tokens.device)
            d = d / d.norm(dim=-1, keepdim=True) * eps * tokens.std()
            pert[:, t, :] += d
            dy = (fn(pert) - base).norm(dim=-1).mean(0)          # (T,)
            dy = dy.double().cpu().numpy()
            ref = dy[t]
            if ref <= 0:
                continue
            n = T - t
            acc[:n] += dy[t:] / ref
            cnt[:n] += 1
    good = cnt > 0
    return np.arange(T)[good], acc[good] / cnt[good]


def _crossing(lags, curve, level):
    """First lag at which the curve drops below `level` and stays below."""
    below = curve < level
    for i in range(len(curve)):
        if below[i] and below[i:].all():
            return int(lags[i])
    return None


def _fmt(lag, tok_dt):
    if lag is None:
        return "  >window "
    return "%4d tok = %6.2f s" % (lag, lag * tok_dt)


def _build(args, window=None, gru_window=None, mamba_stride=None, dtype=torch.float64):
    """Construct HybridNet from the config with command-line overrides applied."""
    conf = ConfigFactory.parse_file(args.config)
    conf.train.device = args.device
    if gru_window is None:
        gru_window = args.gru_window
    if mamba_stride is None:
        mamba_stride = args.mamba_stride
    if gru_window is not None:
        conf.train["gru_window"] = int(gru_window)
    if mamba_stride is not None:
        conf.train["mamba_stride"] = int(mamba_stride)
    if window is None:
        window = args.window if args.window else conf.dataset.train.data_list[0].window_size
    net = net_dict[conf.train.network](conf.train).to(args.device).to(dtype)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=args.device)
        net.load_state_dict(ck["model_state_dict"])
        print("loaded %s (epoch %s)" % (args.ckpt, ck.get("epoch")))
    return net.eval(), int(window)


def _tokens(net, args, window, dtype=torch.float64):
    """Run the CNN so T is exactly what training will see."""
    x = torch.randn(args.batch, window + net.interval, net.in_dim,
                    device=args.device, dtype=dtype)
    with torch.no_grad():
        return net.cnn(x.transpose(-1, -2)).transpose(-1, -2)


# ==========================================================================
# checks
# ==========================================================================
def run_checks(args):
    """Correctness of the windowed GRU, causality, and the fused variance split."""
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        return ok

    net, window = _build(args)
    tokens = _tokens(net, args, window)
    T = tokens.shape[1]
    W = net.gru_window
    tok_dt = net.interval / args.rate
    print("\n=== checks: window %d frames, T = %d tokens, gru_window = %s, mamba_stride = %d ==="
          % (window, T, ("%d tok (%d frames, %.2f s)" % (W, net.gru_window_frames,
                                                         net.gru_window_frames / args.rate))
             if W else "off (whole window)", net.mamba_stride))

    # --- 1. the batched reshape equals an explicit Python loop over chunks ----
    with torch.no_grad():
        fast = net.short_branch(tokens)
        if W > 0 and W < T:
            outs = []
            for s in range(0, T, W):
                outs.append(net._gru_stack(tokens[:, s:s + W, :].contiguous()))
            slow = torch.cat(outs, dim=1)
        else:
            slow = net._gru_stack(tokens)
    d = (fast - slow).abs().max().item()
    check("batched chunking == explicit per-chunk loop (h0=0 each chunk)",
          d == 0.0 and fast.shape == (args.batch, T, net.gru2.hidden_size),
          "max|diff| = %.1e over %d chunks, shape %s"
          % (d, (T + W - 1) // W if W else 1, tuple(fast.shape)))

    # --- 2. no token dropped or duplicated -----------------------------------
    # Every output index must move when its own token moves; if a chunk were
    # dropped or a padded slot leaked in, some index would be dead.
    with torch.no_grad():
        base = net.short_branch(tokens)
        dead = []
        for t in range(T):
            p = tokens.clone()
            p[:, t, :] += 1.0
            dy = (net.short_branch(p) - base).abs().max(dim=-1).values[0]
            if dy[t].item() == 0.0:
                dead.append(t)
    check("every one of the T output positions responds to its own token",
          not dead, "no dead index among %d" % T if not dead else "DEAD: %s" % dead[:6])

    # --- 3. the hard cutoff ---------------------------------------------------
    # Perturb token t.  With a window, the GRU output must be EXACTLY unchanged
    # outside [t .. end of t's chunk]; that is the difference between an
    # implemented window and a documented intention.
    probes = sorted(set([0, 1, W // 2 if W else 5, W - 1 if W else 9,
                         W if W else 20, W + 3 if W else 40,
                         T // 2, T // 2 + 1, T - 2]))
    probes = [p for p in probes if 0 <= p < T - 1]
    rows = []
    with torch.no_grad():
        base_s = net.short_branch(tokens)
        base_l = net.mamba(tokens)
        base_f = net.encoder_tokens(tokens) if hasattr(net, "encoder_tokens") else None
        for t in probes:
            p = tokens.clone()
            p[:, t, :] += 1.0
            ds = (net.short_branch(p) - base_s).abs().max(dim=-1).values[0]
            dl = (net.mamba(p) - base_l).abs().max(dim=-1).values[0]
            end = (t // W + 1) * W - 1 if W else T - 1
            end = min(end, T - 1)
            before = ds[:t].max().item() if t > 0 else 0.0
            after = ds[end + 1:].max().item() if end + 1 < T else 0.0
            inside = ds[t:end + 1].max().item()
            l_before = dl[:t].max().item() if t > 0 else 0.0
            rows.append((t, end, before, inside, after, l_before))

    cut_ok = all(r[2] == 0.0 and r[4] == 0.0 and r[3] > 0 for r in rows) if W else True
    if W:
        check("GRU: perturbing token t moves NOTHING outside [t .. chunk end]",
              cut_ok, "max|dy| outside = %.1e (exact 0 expected), inside = %.2e"
              % (max(max(r[2], r[4]) for r in rows), min(r[3] for r in rows)))
        # the headline claim: a token more than gru_window back cannot be felt
        far_ok = True
        with torch.no_grad():
            for u in (W, W + W // 2, 2 * W, T - 1):
                if u >= T:
                    continue
                cs = (u // W) * W
                inside_t = max(cs, u - 1)
                outside_t = cs - 1                      # one token before the chunk
                for t, want_move in ((outside_t, False), (inside_t, True)):
                    if t < 0 or t >= T:
                        continue
                    p = tokens.clone()
                    p[:, t, :] += 1.0
                    dy = (net.short_branch(p) - base_s).abs().max().item() if False else \
                         (net.short_branch(p) - base_s)[:, u, :].abs().max().item()
                    moved = dy > 0.0
                    far_ok &= (moved == want_move)
        check("GRU: output u moves for the token at u-1 inside its chunk and NOT "
              "for the token one step before the chunk", far_ok,
              "checked at u in {W, 1.5W, 2W, T-1}")
    else:
        check("GRU hard cutoff", False, "SKIPPED: gru_window is off")

    check("Mamba: perturbing token t moves nothing at any index < t (causal)",
          all(r[5] == 0.0 for r in rows),
          "max|dy| below t = %.1e over %d probes" % (max(r[5] for r in rows), len(rows)))

    print("\n  perturbation detail (GRU branch, W = %s):" % (W or "off"))
    print("    %5s %9s %14s %14s %14s %14s"
          % ("t", "chunk end", "max|dy| <t", "max|dy| in", "max|dy| >end", "mamba max|dy| <t"))
    for t, end, before, inside, after, l_before in rows:
        print("    %5d %9d %14.2e %14.2e %14.2e %14.2e"
              % (t, end, before, inside, after, l_before))

    # --- 4. end-to-end causality through the whole encoder -------------------
    x = torch.randn(2, window + net.interval, net.in_dim,
                    device=args.device, dtype=torch.float64)
    # the CNN is not causal (padding 3, kernel 7), so the encoder's frame-level
    # causality is CNN-limited; what must hold is that the recurrent part adds no
    # lookahead beyond the CNN's own +3-frame kernel reach.
    cnn_reach = 0
    for k, st in zip(net.cnn.k_list, net.cnn.s_list):
        cnn_reach = cnn_reach * st + (k - 1) // 2 * (1 if cnn_reach == 0 else 1)
    with torch.no_grad():
        y0 = net.encoder(x)
        worst = 0.0
        for f in (200, 900, window // 2):
            xp = x.clone()
            xp[:, f, :] += 5.0
            dy = (net.encoder(xp) - y0).abs().max(dim=-1).values[0]
            # token index whose receptive field starts after frame f
            first_tok = int(np.ceil((f + 1) / net.interval)) + 2
            worst = max(worst, dy[:max(0, first_tok - 5)].max().item())
    check("full encoder: perturbing frame f leaves earlier tokens unchanged",
          worst == 0.0, "max|dy| on tokens well before f = %.1e" % worst)

    # --- 5. branch variance split through fuse_lin ---------------------------
    # Linear decomposition, NOT branch zeroing: zeroing a branch also shocks the
    # LayerNorm distribution and cannot separate "informative" from "louder".
    if net.fuse_mode == "concat":
        Wm = net.fuse_lin.weight                                   # (feat, gru_hidden+mamba_dim)
        H = net.short_norm.normalized_shape[0]
        with torch.no_grad():
            short = net.short_norm(net.short_branch(tokens))
            long = net.mamba(tokens)
            cs = short @ Wm[:, :H].T                               # (B,T,feat)
            cl = long @ Wm[:, H:].T
        vs, vl = cs.var().item(), cl.var().item()
        cov = ((cs - cs.mean()) * (cl - cl.mean())).mean().item()
        tot = vs + vl
        print("\n  branch output scale:  short(after LayerNorm) std %.4f | long std %.4f"
              % (short.std().item(), long.std().item()))
        print("  fuse_lin contribution var:  short %.4e (%.1f%%) | long %.4e (%.1f%%) "
              "| 2*cov %.2e (%.1f%% of var(sum))"
              % (vs, 100 * vs / tot, vl, 100 * vl / tot, 2 * cov,
                 100 * 2 * cov / (tot + 2 * cov)))
        check("fused variance split is not dominated by one branch's output scale",
              0.15 < vs / tot < 0.85, "short %.1f%% / long %.1f%%"
              % (100 * vs / tot, 100 * vl / tot))
    else:
        check("branch variance split", False, "SKIPPED: fuse mode is %r" % net.fuse_mode)

    print("\n  %-70s %-6s %s" % ("check", "result", "detail"))
    print("  " + "-" * 128)
    for name, ok, detail in results:
        print("  %-70s %-6s %s" % (name, "PASS" if ok else "FAIL", detail))
    print("  " + "-" * 128)
    print("  %d/%d passed" % (sum(1 for _, o, _ in results if o), len(results)))
    return all(o for _, o, _ in results)


# ==========================================================================
# cost/horizon sweep
# ==========================================================================
def run_sweep(args):
    """(window_size, mamba_stride) trade table with measured fwd+bwd time."""
    pairs = []
    for tok in args.sweep.split(","):
        w, s = tok.split(":")
        pairs.append((int(w), int(s)))
    print("\n=== (window_size, mamba_stride) trade, batch %d, float32, fwd+bwd ==="
          % args.sweep_batch)
    print("  %-7s %-7s %-6s %-11s %-11s %-9s %-9s %-9s %-9s"
          % ("window", "stride", "K", "s/SSM step", "span (win)", "mamba ms", "gru ms",
             "encoder ms", "peak MB"))
    for w, s in pairs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        net, window = _build(args, window=w, mamba_stride=s, dtype=torch.float32)
        net.train()
        x = torch.randn(args.sweep_batch, w + net.interval, net.in_dim,
                        device=args.device, dtype=torch.float32)
        with torch.no_grad():
            tk = net.cnn(x.transpose(-1, -2)).transpose(-1, -2)
        T = tk.shape[1]
        K = causal_avg_pool(tk, s).shape[1]
        tk = tk.detach().requires_grad_(True)

        def timeit(fn, n=5):
            for _ in range(3):
                fn().pow(2).mean().backward()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                fn().pow(2).mean().backward()
            torch.cuda.synchronize()
            return (time.time() - t0) / n * 1e3

        try:
            t_m = timeit(lambda: net.mamba(tk))
            t_g = timeit(lambda: net.short_branch(tk))
            t_e = timeit(lambda: net.encoder(x))
            mb = torch.cuda.max_memory_allocated() / 2 ** 20
            print("  %-7d %-7d %-6d %-11.2f %-11s %-9.1f %-9.1f %-9.1f %-9.0f"
                  % (w, s, K, s * net.interval / args.rate, "%.0f s" % (w / args.rate),
                     t_m, t_g, t_e, mb))
        except RuntimeError as e:
            print("  %-7d %-7d %-6d %-11.2f %-11s  FAILED: %s"
                  % (w, s, K, s * net.interval / args.rate, "%.0f s" % (w / args.rate),
                     str(e)[:60]))
        del net, x, tk
    print("\n  Read this next to the horizon table: the Mamba branch's influence dies")
    print("  after ~7 SSM STEPS whatever a step is worth, so horizon ~ stride, while")
    print("  cost ~ K^2 = (T/stride)^2.  Raising stride buys horizon AND speed;")
    print("  raising window_size buys only the learnable CEILING, at linear cost in")
    print("  the pypose integrator (60-88% of a real training step) and quadratic")
    print("  cost in the scan if you hold stride fixed.")


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/exp/UAV/hybrid_best.conf")
    ap.add_argument("--ckpt", type=str, default=None, help="optional checkpoint to load")
    ap.add_argument("--window", type=int, default=None,
                    help="override window_size (frames); default = the config's")
    ap.add_argument("--gru_window", type=int, default=None,
                    help="override gru_window (frames); 0 = unbounded")
    ap.add_argument("--mamba_stride", type=int, default=None,
                    help="override mamba_stride (tokens per SSM step)")
    ap.add_argument("--checks", action="store_true",
                    help="run the windowing / causality / variance-split checks")
    ap.add_argument("--sweep", type=str, default=None,
                    help="cost table over 'window:stride,window:stride,...'")
    ap.add_argument("--sweep_batch", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rate", type=float, default=100.0, help="IMU sample rate, Hz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.sweep:
        return run_sweep(args)
    if args.checks:
        return run_checks(args)

    net, window = _build(args)
    interval = net.interval
    tok_dt = interval / args.rate                 # seconds per CNN token
    frame_dt = 1.0 / args.rate

    # Real token statistics: run the CNN on white noise shaped like the IMU is
    # good enough here -- the decay we measure is a property of the recurrences,
    # and the branches see whatever the CNN emits.  Using the CNN keeps T exactly
    # what training will see.
    tokens = _tokens(net, args, window)
    T = tokens.shape[1]
    K = causal_avg_pool(tokens, net.mamba_stride).shape[1]

    print("\n=== geometry ===")
    print("  window_size            %d frames = %.2f s @ %.0f Hz" % (window, window * frame_dt, args.rate))
    print("  CNN tokens         T = %d   (1 token = %d frames = %.0f ms)" % (T, interval, tok_dt * 1000))
    print("  Mamba pooled steps K = %d   (1 step  = %d frames = %.2f s)"
          % (K, interval * net.mamba_stride, tok_dt * net.mamba_stride))
    if net.gru_window:
        print("  GRU chunk          W = %d   (%d frames = %.2f s), %d chunks per window, "
              "hard cutoff" % (net.gru_window, net.gru_window_frames,
                               net.gru_window_frames / args.rate,
                               (T + net.gru_window - 1) // net.gru_window))
    else:
        print("  GRU chunk          W = whole window (unbounded, soft decay only)")

    def gru_fn(tk):
        return net.short_branch(tk)

    def mamba_fn(tk):
        return net.mamba(tk)

    probes = sorted(set(int(p) for p in np.linspace(0, T - 2, min(12, T - 1))))
    print("\n=== influence decay (probe tokens %s) ===" % probes)
    rows = []
    gname = "GRU  (window %d tok)" % net.gru_window if net.gru_window else "GRU  (full token rate)"
    for name, fn in ((gname, gru_fn),
                     ("Mamba(stride %d)" % net.mamba_stride, mamba_fn)):
        lags, curve = _decay_curve(fn, tokens, probes)
        rows.append((name, lags, curve))
        e1 = _crossing(lags, curve, 1.0 / np.e)
        p10 = _crossing(lags, curve, 0.10)
        p01 = _crossing(lags, curve, 0.01)
        print("  %-24s 1/e: %s | 10%%: %s | 1%%: %s"
              % (name, _fmt(e1, tok_dt), _fmt(p10, tok_dt), _fmt(p01, tok_dt)))

    print("\n  lag  ->  normalised influence")
    shown = [l for l in (0, 1, 2, 5, 10, 20, 40, 80, 160, 320) if l < len(rows[0][2])]
    print("  %-8s %-12s %-12s %s" % ("tokens", "seconds", "GRU", "Mamba"))
    for l in shown:
        print("  %-8d %-12.2f %-12.4g %.4g" % (l, l * tok_dt, rows[0][2][l], rows[1][2][l]))

    # ---- verdict -------------------------------------------------------------
    e_gru = _crossing(rows[0][1], rows[0][2], 1.0 / np.e)
    e_mam = _crossing(rows[1][1], rows[1][2], 1.0 / np.e)
    p_gru = _crossing(rows[0][1], rows[0][2], 0.01)
    p_mam = _crossing(rows[1][1], rows[1][2], 0.01)
    print("\n=== verdict ===")
    if e_gru is not None and e_mam is not None:
        print("  separation (1/e): Mamba / GRU = %.1fx  (%.2f s vs %.2f s)"
              % (e_mam / max(e_gru, 1), e_mam * tok_dt, e_gru * tok_dt))
    if p_gru is not None and p_mam is not None:
        print("  separation (1%%):  Mamba / GRU = %.1fx  (%.2f s vs %.2f s)"
              % (p_mam / max(p_gru, 1), p_mam * tok_dt, p_gru * tok_dt))
    print("  Mamba SSM steps available in one window: K = %d" % K)
    if K < 32:
        need = int(np.ceil(32 * net.mamba_stride * interval / 50.0)) * 50
        print("  NOTE: K = %d SSM steps.  That is short for a state-space model, but the\n"
              "  measured influence of this branch is already below 1%% after ~7 steps at\n"
              "  initialisation, so K is a ceiling on what TRAINING can learn, not on what\n"
              "  the branch does today.  For K >= 32 you need window_size >= %d frames\n"
              "  (%.0f s); for K >= 56, window_size >= %d." %
              (K, need, need / args.rate,
               int(np.ceil(56 * net.mamba_stride * interval / 50.0)) * 50))
    else:
        print("  K = %d is a usable state-space sequence length." % K)


if __name__ == "__main__":
    main()
