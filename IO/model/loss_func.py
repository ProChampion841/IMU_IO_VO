"""Loss primitives.

`diag_ln_cov_loss` is the diagonal Gaussian NLL that supervises velnet's per-axis
velocity variance; `vmf_dir_nll` is the von Mises-Fisher NLL that supervises its
velocity DIRECTION uncertainty (both used by model/losses.py:velocity_loss).

L1 / L2 / Huber / diag_cov_loss / loss_weight_decay / loss_weight_decrease and the
`loss_fc_list` table they populated were the IMU-CORRECTION arm's objective menu,
selected by the `loss:` / `rotloss:` config keys.  That arm was deleted on
2026-09-07 and velocity_loss calls torch.nn.functional.huber_loss directly, so
nothing reads them any more.
Recover with: git show <commit>^:model/loss_func.py
"""
import torch

EPSILON = 1e-7


def diag_ln_cov_loss(dist, pred_cov, use_epsilon=False):
    """err^2 / sigma^2 + ln sigma^2, meaned.  `pred_cov` is a VARIANCE, not a std."""
    error = (dist).pow(2)
    if use_epsilon: l = ((error / pred_cov) + torch.log(pred_cov + EPSILON))
    else: l = ((error / pred_cov) + torch.log(pred_cov))
    return l.mean()


# ln(2*pi), the constant in the S^2 von Mises-Fisher log-normaliser.
_LN_2PI = 1.8378770664093453


def vmf_dir_nll(pred, targ, kappa, eps=1e-6, min_speed=0.0, min_pred=1e-3):
    """von Mises-Fisher NLL on the DIRECTION of a 3-vector.  Returns (nll, cos, n_used).

    THE MODEL.  Write u = v / |v| in S^2.  The vMF density with mean direction mu and
    concentration kappa > 0 is

        p(u) = C_3(kappa) * exp(kappa * mu . u),      C_3(kappa) = kappa / (4 pi sinh kappa)

    so, using  ln sinh k = k + ln(1 - e^{-2k}) - ln 2,

        ln C_3(k) = ln k - k - ln(2 pi) - ln(1 - e^{-2k})
        NLL       = -[k * cos + ln C_3(k)]
                  = k * (1 - cos) - ln k + ln(2 pi) + ln(1 - e^{-2k})

    WHY THIS SHAPE IS THE RIGHT ONE HERE.  `k * (1 - cos)` is the data term -- it is
    ~ k * theta^2 / 2 for small angles, i.e. exactly the "squared error over variance"
    of a tangent-plane Gaussian with per-axis angular sigma = 1/sqrt(k).  `-ln k` is the
    log-normaliser that stops the head from simply declaring infinite confidence.  It is
    the direction-space analogue of err^2/sigma^2 + ln sigma^2, which is why it composes
    cleanly with `diag_ln_cov_loss` rather than competing with it.

    LIMITS, both benign and both checked:
      * k -> 0 (the floor): the -ln k and the ln(1 - e^{-2k}) ~ ln(2k) cancel exactly and
        NLL -> ln(4 pi) = 2.5310, the uniform density on the sphere.  No divergence.
      * k large: ln(1 - e^{-2k}) -> 0 and NLL -> k(1 - cos) - ln k + ln(2 pi).

    `min_speed` drops frames whose TARGET is too slow for a direction to be meaningful
    (|v| -> 0 has no direction).  On this corpus flight is ~22 m/s so it rarely fires,
    but a window that clips a takeoff would otherwise train the head on noise.

    `min_pred` does the same for the PREDICTION, and it is not optional -- it guards a
    gradient singularity that is REACHED ON EVERY RUN, not a corner case.  The velocity
    head is zero-initialised (model/velocity_net.py), so at step 0 `pred` is EXACTLY the
    zero vector: it has no direction, `pred / |pred|.clamp_min(eps)` becomes `0 / eps`,
    and d(u_p)/d(pred) = 1/eps = 1e6.  MEASURED on a real tensor: max |d nll/d pred| is
    2.5e5 at |pred| = 0 against 0.0 at |pred| = 1e-6, i.e. the singularity is exactly AT
    zero and nowhere near it.  Times vel_dir_weight that is a ~5e3 spike into the encoder
    on the first optimizer step of every run.
    A frame below the floor contributes nothing: the direction term simply switches on
    once the magnitude head has moved off zero, which is the correct behaviour -- there
    is no direction to score until there is a velocity.  The model cannot exploit this to
    escape the loss, because the Huber term is simultaneously driving |pred| to ~22 m/s.

    Args
        pred   (..., 3) predicted velocity -- only its direction is used
        targ   (..., 3) target velocity
        kappa  (..., 1) concentration, must already be positive
    Returns
        nll    scalar, mean over the frames actually used (0.0 if none are)
        cos    (..., 1) cosine of the angle between predicted and target direction
        n_used number of frames that passed BOTH gates
    """
    tn = targ.norm(dim=-1, keepdim=True)
    pn = pred.norm(dim=-1, keepdim=True)
    u_p = pred / pn.clamp_min(eps)
    u_t = targ / tn.clamp_min(eps)
    cos = (u_p * u_t).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

    k = kappa.clamp_min(eps)
    nll = (k * (1.0 - cos) - torch.log(k) + _LN_2PI
           + torch.log1p(-torch.exp(-2.0 * k)))

    # ALWAYS masked -- the min_pred gate is a numerical guard, not an option.  Masking by
    # multiply is what kills the gradient too: d/d pred of (nll * 0) is 0 even though
    # d nll/d pred is 2.5e5 there, so the singular frames contribute nothing rather than
    # contributing a spike.
    mask = (pn >= min_pred).to(nll.dtype)
    if min_speed > 0.0:
        mask = mask * (tn >= min_speed).to(nll.dtype)
    n_used = mask.sum()
    return (nll * mask).sum() / n_used.clamp_min(1.0), cos, n_used
