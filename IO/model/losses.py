import torch
import pypose as pp
from .loss_func import diag_ln_cov_loss, vmf_dir_nll


def loss_(fc, pred, targ, sampling = None, dtype = 'trans'):
    ## reshape or sample the input targ and pred
    ## cov and error is for reference
    if sampling:
        pred = pred[:,sampling-1::sampling,:]
        targ = targ[:,sampling-1::sampling,:]
    else:
        pred = pred[:,-1:,:]
        targ = targ[:,-1:,:]

    if dtype == 'rot':
        dist = (pred * targ.Inv()).Log()
    else:
        dist = pred - targ
    # `fc=None` returns the residual only.  Used by the `use_rot_loss: False` path,
    # which needs rot_dist for the monitoring metric and for the covariance NLL but
    # must never build a rotation-state term.  With a real fc this is unchanged.
    loss = fc(dist) if fc is not None else None
    return loss, dist


def velocity_loss(inte_state, label, confs):
    """Huber on BODY-FRAME velocity against the GPS nav velocity, plus a NLL.

    THE LABEL IS BUILT HERE, NOT IN THE MODEL.  `label['gt_vel']` is the GPS nav
    velocity in the WORLD frame (GPSNavVn, converted to NWU by the loader).  The
    model predicts BODY frame, so the label is rotated world -> body with
    `label['gt_rot']` -- the attitude at the SAME frames as the velocity.  Using
    `data['rot']` instead would pair velocity at frame k with attitude at k-1 and
    inject one sample of rotation into every label; see THE TIMING RULE in
    model/velocity_net.py.

    Every frame is scored, not every `sampling`-th: this is a per-frame regression
    with a per-frame target, so subsampling would throw away signal for nothing.
    `vel` and `pos` are reported in m/s and metres so they stay comparable with the
    integrating arms.
    """
    pred = inte_state["vel_body"]
    gt_rot, gt_vel = label["gt_rot"], label["gt_vel"]
    n = min(pred.shape[1], gt_vel.shape[1], gt_rot.lshape[1])
    pred = pred[:, :n]
    gt_body = gt_rot[:, :n].Inv() @ gt_vel[:, :n]

    dist = pred - gt_body

    # ---- BURN-IN (opt-in, `loss_burnin` in FRAMES, default 0 = off) ------------------
    # Both branches start every window from h0 = 0.  The first frames of a window are
    # predicted from almost no history -- the Mamba branch does not take its first
    # real step until 9 * mamba_stride frames in (8.64 s at stride 96) -- yet they were
    # scored exactly like frames with a full minute behind them, so part of the
    # gradient was spent on frames the model cannot get right.  Burn-in drops them from
    # the OBJECTIVE only.  The reported `vel`/`pos` metrics below still cover every
    # frame, so runs with and without burn-in stay comparable on the same numbers.
    b = min(max(int(confs.get("loss_burnin", 0)), 0), n - 1)
    pred_l, gt_l, dist_l = pred[:, b:], gt_body[:, b:], dist[:, b:]

    # NOT `vel_huber_delta` -- that key already exists in these configs at 0.27,
    # a CORRECTION-scale delta.  Reusing the name would silently apply 0.27 m/s
    # to a ~20 m/s target, which is pure L1 by accident rather than by choice.
    delta = float(confs.get("velnet_huber_delta", 1.0))
    loss = torch.nn.functional.huber_loss(pred_l, gt_l, delta=delta)

    out = {"vel": dist.norm(dim=-1).mean(),
           "vel_rel": (dist[:, 1:] - dist[:, :-1]).norm(dim=-1).mean()}
    if "pos" in inte_state and "gt_pos" in label:
        m = min(inte_state["pos"].shape[1], label["gt_pos"].shape[1])
        pd = inte_state["pos"][:, :m] - label["gt_pos"][:, :m]
        out["pos"] = pd.norm(dim=-1).mean()
        out["pos_rel"] = (pd[:, 1:] - pd[:, :-1]).norm(dim=-1).mean()
    else:
        out["pos"] = out["vel"].new_zeros(())
        out["pos_rel"] = out["vel"].new_zeros(())

    # ---- DRIFT TERM (opt-in, `drift_weight` > 0) ----------------------------------------
    # Position error is the INTEGRAL of velocity error, so what drives drift is the
    # low-frequency part of the error -- a bias held for tens of seconds -- while
    # per-frame Huber weighs a +3/-3 m/s zig-zag (which integrates to ~0) the same as a
    # steady +3 m/s (which integrates to 30 m per 10 s).  This term scores the MEAN
    # world-frame velocity error over non-overlapping segments of `drift_seg`
    # frames, i.e. the displacement error of each segment divided by its duration.
    # World frame, because displacement is accumulated in the world frame: a body-frame
    # mean would let a lateral error cancel through a turn that the position does not
    # forget.  The rotation is label['gt_rot'] -- the same attitude the label uses.
    drift_w = float(confs.get("drift_weight", 0.0))
    if drift_w > 0.0:
        S = max(int(confs.get("drift_seg", 1000)), 1)
        k = dist_l.shape[1] // S
        if k >= 1:
            err_w = gt_rot[:, b:b + k * S] @ dist_l[:, :k * S]
            seg = err_w.reshape(err_w.shape[0], k, S, 3).mean(dim=2)       # (B, k, 3)
            drift = torch.nn.functional.huber_loss(
                seg, torch.zeros_like(seg),
                delta=float(confs.get("drift_huber_delta", delta)))
            loss = loss + drift_w * drift
            out["drift_loss"] = (drift_w * drift).detach()
            # m/s: the average size of a segment's mean velocity error -- the number
            # that multiplies by segment length to give drift.
            out["seg_vel_bias"] = seg.detach().norm(dim=-1).mean()

    # Uncertainty: err^2/sigma^2 + ln sigma^2.  Without it the head is free to
    # emit any constant and the covariance would mean nothing.
    #
    # `cov_nll_mode` (default "plain" = unchanged):
    #   "detach"  the NLL sees a DETACHED error, so it trains the variance head only
    #             and never pulls on the velocity.  The mean is then set by Huber
    #             alone, and cov_weight can be raised without trading accuracy for
    #             calibration.
    #   "beta"    beta-NLL (Seitzer et al., ICLR 2022): each term is weighted by a
    #             DETACHED sigma^(2*beta), `cov_nll_beta` (default 0.5).  Plain NLL
    #             down-weights exactly the frames with large error (their gradient is
    #             divided by sigma^2); beta-NLL removes most of that bias.
    # Note the measured overconfidence on validation is mostly the train/val gap --
    # the head is calibrated on TRAIN -- so no NLL form fixes it on its own.
    cov = inte_state.get("vel_cov")
    if confs.propcov and cov is not None:
        cov_n = cov[:, :n]
        cov_l = cov_n[:, b:]
        mode = str(confs.get("cov_nll_mode", "plain"))
        if mode == "plain":
            nll = diag_ln_cov_loss(dist_l, cov_l)
        elif mode == "detach":
            nll = diag_ln_cov_loss(dist_l.detach(), cov_l)
        elif mode == "beta":
            beta = float(confs.get("cov_nll_beta", 0.5))
            per = dist_l.pow(2) / cov_l + torch.log(cov_l)
            nll = (cov_l.detach().pow(beta) * per).mean()
        else:
            raise ValueError("cov_nll_mode must be plain|detach|beta, got %r" % (mode,))
        loss = loss + confs.cov_weight * nll
        out["cov_loss"] = (confs.cov_weight * nll).detach()
        # Always the UNWEIGHTED plain NLL over every frame, whatever the training
        # form, so this column stays comparable across modes and burn-in settings.
        out["cov_nll_vel"] = diag_ln_cov_loss(dist, cov_n).detach()
        out["cov_nll_rot"] = nll.new_zeros(())
        out["cov_nll_pos"] = nll.new_zeros(())
        out["pred_cov_vel"] = cov[:, :n].mean().detach()
        out["pred_cov_rot"] = cov.new_zeros(())
        out["pred_cov_pos"] = cov.new_zeros(())
    else:
        for k in ("pred_cov_rot", "pred_cov_vel", "pred_cov_pos"):
            out[k] = loss.new_zeros(())

    # ---- DIRECTION UNCERTAINTY (opt-in, vel_dir_weight > 0) --------------------
    # `vel_cov` above is DIAGONAL, so it can only draw an axis-aligned ellipsoid.  On
    # this corpus |v| ~ 22 m/s lies almost entirely on body x, so a heading error theta
    # shows up as ~|v|*sin(theta) on body y -- indistinguishable, to a diagonal model,
    # from genuinely moving sideways.  The vMF term scores the DIRECTION of the velocity
    # separately from its magnitude, so the model can report "the speed is right, the
    # heading is not" and be trained to mean it.
    #
    # It ADDS to the diagonal NLL rather than replacing it, and that is deliberate: the
    # diagonal term still carries the magnitude, and the two log-normalisers (ln sigma^2
    # and -ln kappa) are separate quantities that do not double-count. What IS
    # double-counted to a degree is the transverse error -- it appears once in the
    # y/z variances and once in the angle -- which is why vel_dir_weight is a separate
    # knob and should start SMALL (see the config).
    kappa = inte_state.get("vel_kappa")
    dir_w = float(confs.get("vel_dir_weight", 0.0))
    if kappa is not None and dir_w > 0.0:
        dir_nll, cos, n_used = vmf_dir_nll(
            pred_l, gt_l, kappa[:, b:n],
            min_speed=float(confs.get("vel_dir_min_speed", 1.0)))
        loss = loss + dir_w * dir_nll
        k_det = kappa[:, :n].detach()
        out["dir_loss"] = (dir_w * dir_nll).detach()
        out["dir_nll"] = dir_nll.detach()
        # The two numbers to actually watch in TensorBoard: how wrong the direction is,
        # and how wrong the model THINKS it is.  They should converge.
        out["dir_err_deg"] = (torch.arccos(cos.detach()).mean() * (180.0 / torch.pi))
        out["pred_dir_sigma_deg"] = (k_det.clamp_min(1e-9).rsqrt().mean()
                                     * (180.0 / torch.pi))
        out["pred_kappa"] = k_det.mean()
        out["dir_frames_used"] = n_used.detach()

    out["loss"] = loss
    return out


def get_loss(inte_state, data, confs):
    """Objective for velnet.  See velocity_loss() -- it is the only path.

    This used to dispatch: a model emitting `vel_body` went to velocity_loss(), and
    anything else fell through to ~350 lines of integration-based state/covariance
    loss written for the IMU-CORRECTION arm (rotation NLL, per-channel Huber deltas,
    covariance decoupling, the airspeed-consistency term).  That arm was deleted on
    2026-09-07 and nothing reaches those terms any more, so they are gone with it.
    Recover them with: git show <commit>^:model/losses.py

    The guard is kept rather than dropped: get_loss() is called with whatever the
    network returned, and a model that silently stopped emitting `vel_body` would
    otherwise fail somewhere deep inside velocity_loss() with a KeyError naming a
    tensor rather than the actual problem.
    """
    if "vel_body" not in inte_state:
        raise KeyError(
            "get_loss expected a 'vel_body' key: velnet is the only network in this "
            "repo and velocity_loss() is the only objective. Got keys %s. If you are "
            "restoring the deleted correction arm, restore its loss terms too "
            "(git show <commit>^:model/losses.py)." % sorted(inte_state))
    return velocity_loss(inte_state, data, confs)

def get_RMSE(inte_state, data):
    '''
    get the RMSE of the last state in one segment
    '''
    def _RMSE(x):
        return torch.sqrt((x.norm(dim=-1)**2).mean())

    dist_pos = (inte_state['pos'][:,-1,:] - data['gt_pos'][:,-1,:])
    dist_vel = (inte_state['vel'][:,-1,:] - data['gt_vel'][:,-1,:])
    dist_rot = (data['gt_rot'][:,-1,:] * inte_state['rot'][:,-1,:].Inv()).Log()

    pos_loss = _RMSE(dist_pos)[None,...]
    vel_loss = _RMSE(dist_vel)[None,...]
    rot_loss = _RMSE(dist_rot)[None,...]

    ## Relative pos error
    return {'pos': pos_loss, 'rot': rot_loss, 'vel': vel_loss,
            'pos_dist': dist_pos.norm(dim=-1).mean(),
            'vel_dist': dist_vel.norm(dim=-1).mean(),
            'rot_dist': dist_rot.norm(dim=-1).mean(),}
