import torch
import pypose as pp
from .loss_func import loss_fc_list, diag_ln_cov_loss, Huber
from utils import report_hasNan


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
    # NOT `vel_huber_delta` -- that key already exists in these configs at 0.27,
    # a CORRECTION-scale delta.  Reusing the name would silently apply 0.27 m/s
    # to a ~20 m/s target, which is pure L1 by accident rather than by choice.
    delta = float(confs.get("velnet_huber_delta", 1.0))
    loss = torch.nn.functional.huber_loss(pred, gt_body, delta=delta)

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

    # Uncertainty: err^2/sigma^2 + ln sigma^2.  Without it the head is free to
    # emit any constant and the covariance would mean nothing.
    cov = inte_state.get("vel_cov")
    if confs.propcov and cov is not None:
        nll = diag_ln_cov_loss(dist, cov[:, :n])
        loss = loss + confs.cov_weight * nll
        out["cov_loss"] = (confs.cov_weight * nll).detach()
        out["cov_nll_vel"] = nll.detach()
        out["cov_nll_rot"] = nll.new_zeros(())
        out["cov_nll_pos"] = nll.new_zeros(())
        out["pred_cov_vel"] = cov[:, :n].mean().detach()
        out["pred_cov_rot"] = cov.new_zeros(())
        out["pred_cov_pos"] = cov.new_zeros(())
    else:
        for k in ("pred_cov_rot", "pred_cov_vel", "pred_cov_pos"):
            out[k] = loss.new_zeros(())
    out["loss"] = loss
    return out


def airspeed_consistency(inte_state, data, confs):
    """Penalise disagreement between the INTEGRATED forward speed and the pitot.

    WHY THIS EXISTS.  A tilt error and an accelerometer bias are algebraically the
    same thing when you only score velocity: a tilt of theta leaks g*sin(theta) of
    gravity into the horizontal, and both produce a velocity error growing as
    (constant * t).  MEASURED on this corpus: the 0.0663 m/s^2 bias the 15 s freeze
    fits IS a 0.387 deg tilt.  With `correct_gyro: False` the network can only move
    accel, so the cheapest way for it to cut velocity error is to cancel attitude
    drift with a fake accel bias -- hiding the gyro error instead of exposing it.
    That is the counter-drift trap this project has already measured twice.

    The pitot is the one channel that does not share that ambiguity, because it
    measures forward speed directly rather than integrating anything.  Feeding it in
    as a NETWORK INPUT does not help: the objective never asks the prediction to
    agree with it.  This term is what turns it into a constraint.

    THE RESIDUAL.  With `w` the (unknown, roughly constant) wind in the WORLD frame,
        v_ground = v_air + w
    Projecting on the body-forward axis and writing R for the predicted body->world
    rotation, and taking |v_air| ~ airspeed (small sideslip):
        (R^T v_pred)[0]  =  k * airspeed  +  (R^T w)[0]
    `k` is the pitot scale factor, MEASURED against GPS on this aircraft at 0.993.
    (R^T w)[0] = dot(R[:, 0], w), and R[:, 0] is the body-forward axis expressed in
    world coordinates -- so the design matrix is that axis and the fit is linear.

    `w` is solved per window by ridge-regularised least squares.  The ridge is not
    cosmetic: in straight flight every row of the design matrix points the same way,
    the normal matrix collapses to rank 1, and only the along-track wind component is
    observable.  Without it the crosswind estimate is arbitrary and can absorb the
    very error this term exists to expose.

    ROTATION IS THE PREDICTED ONE, NOT GT.  Using `data['gt_rot']` here would be both
    leakage and self-defeating: the residual is only sensitive to attitude error
    because R carries that error.

    TIMING.  `vel[k]` is the state AFTER integrating `dt[k]`, i.e. at window frame
    k+1; the padding9 collate pads the FRONT, so `airspeed[:, 9+i]` is window frame
    i.  The matching sample for `vel[k]` is therefore `airspeed[:, 10+k]`, which runs
    one past the end at k = F-1 -- so the LAST velocity sample is dropped rather than
    paired with the wrong frame.  Four off-by-ones of exactly this shape have shipped
    in this repo; this one is written out so the next reader can check it.
    """
    vel = inte_state["vel"]
    B, F = vel.shape[0], vel.shape[1]
    air = data.get("airspeed", None)
    if air is None or F < 2:
        return None
    pad = air.shape[1] - F
    if pad < 1:
        return None
    # exact pairing, last velocity sample dropped -- see TIMING above
    air = air[:, pad + 1: pad + F, 0]                                   # (B, F-1)
    v = vel[:, : F - 1, :]                                              # (B, F-1, 3)
    R = inte_state["rot"][:, : F - 1]
    if not isinstance(R, pp.LieTensor):
        R = pp.SO3(R)
    v_body_fwd = (R.Inv() @ v)[..., 0]                                  # (B, F-1)
    fwd_world = R @ torch.tensor([1.0, 0.0, 0.0], dtype=v.dtype,
                                 device=v.device).expand_as(v)          # (B, F-1, 3)

    k = float(confs.get("airspeed_scale", 0.993))
    y = v_body_fwd - k * air                                            # (B, F-1)

    A = fwd_world
    ridge = float(confs.get("air_wind_ridge", 1e-2)) * A.shape[1]
    AtA = torch.einsum("bfi,bfj->bij", A, A)
    AtA = AtA + ridge * torch.eye(3, dtype=A.dtype, device=A.device)
    Aty = torch.einsum("bfi,bf->bi", A, y)
    w = torch.linalg.solve(AtA, Aty)                                    # (B, 3) world wind
    resid = y - torch.einsum("bfi,bi->bf", A, w)

    delta = float(confs.get("air_huber_delta", 1.0))
    a = resid.abs()
    hub = torch.where(a <= delta, 0.5 * resid ** 2 / delta, a - 0.5 * delta)
    return {"air_loss": hub.mean(),
            "air_resid": a.mean().detach(),
            "air_wind": w.norm(dim=-1).mean().detach()}


def get_loss(inte_state, data, confs):
    # VelocityNet regresses velocity instead of correcting the IMU, so none of the
    # integration-based terms below apply.  Dispatch on what the model produced,
    # which keeps every existing config on exactly the path it had before.
    if "vel_body" in inte_state:
        return velocity_loss(inte_state, data, confs)
    ## The state loss for evaluation
    loss, state_losses, cov_losses = 0, {}, {}
    loss_fc = loss_fc_list[confs.loss]

    # `use_rot_loss: False` drops the rotation STATE term from the objective.  It is
    # for platforms that already carry an attitude solution they trust (an MTI here),
    # where scoring the integrated rotation teaches the network nothing it will be
    # allowed to use at runtime.  Default True == upstream, bit for bit.
    #
    # When it is false the rotation residual is built under no_grad, so the rotation
    # state contributes no node to the autograd graph at all -- it is not a term
    # multiplied by zero, which would still cost a backward pass and could still
    # propagate NaN.  A consequence is that `covaug` no longer feeds the rotation
    # residual back into the graph either; that back door is exactly the
    # rotation-state gradient this switch exists to remove.
    #
    # loss_fc_list selection: `confs.rotloss` is only read when rotation is scored, so
    # every existing config resolves through the same table entry as before, and the
    # key becomes optional for a no-rot-loss config.  `confs.rot_weight` likewise is
    # only read when it still multiplies something.
    use_rot_loss = confs.get("use_rot_loss", True)

    if use_rot_loss:
        rotloss_fc = loss_fc_list[confs.rotloss]
        rot_loss, rot_dist = loss_(rotloss_fc, inte_state['rot'], data['gt_rot'], sampling = confs.sampling, dtype='rot')
    else:
        rot_loss = None
        with torch.no_grad():
            _, rot_dist = loss_(None, inte_state['rot'], data['gt_rot'], sampling = confs.sampling, dtype='rot')

    # PER-CHANNEL HUBER DELTA.
    #
    # `loss: Huber_loss005` is delta = 0.005, and delta is the knee between the
    # quadratic and linear regimes.  MEASURED on this corpus at window 1000 (256 val
    # windows x 20 checkpoints x 3 axes), the residual quantiles are
    #
    #     pos  p25 0.127  MEDIAN 0.590  p75 2.174  p90 4.948  p99 11.956  m
    #     vel  p25 0.081  MEDIAN 0.268  p75 0.740  p90 1.415  p99  2.861  m/s
    #
    # so delta 0.005 sits ~120x below the MEDIAN residual and 99.9% of elements land
    # in the linear regime: the objective is L1 scaled by 0.005, not a Huber at all.
    # Verified numerically: 0.005 * mean|pos_dist| = 0.00977943 against an actual
    # pos_loss of 0.00976699.
    #
    # That matters because windows are GT-INITIALISED, so pos_dist[:, k] is the drift
    # over (k+1) * sampling seconds -- every horizon from 0.5 s to the window length is
    # already supervised.  Under L1 each of the 20 checkpoints receives EXACTLY 5.00%
    # of the gradient, so half the gradient is spent on 0-5 s where the mean residual
    # is 0.019-1.0 m, and half on 5-10 s where it is 1.5-4.0 m.  Uniform weighting of a
    # 205x range of error magnitudes is a choice, and it is not the one that minimises
    # the error at the horizon actually reported.
    #
    # Setting delta at the channel's MEDIAN residual is the standard Huber choice and
    # splits the difference: half the residuals stay quadratic (gradient proportional
    # to error, so big errors pull harder) and half stay linear (so the p99 outlier
    # windows this corpus is known to contain cannot dominate).  Measured share of
    # gradient falling on the second half of the window:
    #
    #     delta      0.005(L1)   0.05    0.25    0.6     1.0     2.0     L2
    #     pos          50.5%    54.0%   60.6%   65.0%   70.4%   77.0%   87.1%
    #     vel          50.2%    52.0%   57.9%   62.9%   69.5%   74.3%   75.6%
    #
    # Both keys default to None, in which case `confs.loss` is used unchanged and every
    # existing config produces a bit-for-bit identical objective.
    # SCALE PRESERVATION -- this is why delta is safe to change.
    #
    # torch's huber_loss has gradient magnitude `delta` in the linear regime, so raising
    # delta 0.005 -> 0.6 multiplies the whole objective's gradient by ~120x.  Measured
    # unnormalised: accdecoder |grad| 6.17e-02 -> 5.32e+00, an 86x jump, which at
    # lr 1e-3 would diverge and would silently invalidate lr, pos_weight, vel_weight and
    # weight_decay all at once.
    #
    # Dividing by delta/delta_ref fixes the linear-regime slope at delta_ref -- exactly
    # today's value -- so these keys change only the SHAPE of the loss (which residuals
    # are treated quadratically), never its magnitude.  Every other hyper-parameter
    # keeps its meaning, and the delta sweep is a one-variable experiment.
    _REF_DELTA = {"Huber_loss005": 0.005, "Huber_loss05": 0.05, "L1": 1.0}
    _ref = _REF_DELTA.get(confs.loss)

    # HORIZON WEIGHTING -- `loss_time_power: p` (default 0 = uniform, bit-identical).
    #
    # Checkpoint k of K (every `sampling` frames) is weighted ((k+1)/K)^p, normalised
    # to mean 1 so the loss scale -- and with it lr, the channel weights and the Huber
    # normalisation above -- keeps its meaning.  p = 1 gives the last checkpoint 2x
    # the average weight and the first ~0.
    #
    # WHY.  Every number this project is judged on is an END-POINT error at 30/60/120 s,
    # while the objective spreads its weight evenly over 120 checkpoints from 0.5 s to
    # 60 s.  The error being removed is attitude leakage, which grows with elapsed time,
    # so the early checkpoints carry almost none of it.  MEASURED on the tilt_aware run:
    # val 120 s kept improving at the LR floor (-0.0018 per 100 epochs) while 30/60 s
    # stayed flat, i.e. the long-horizon signal is what is still being learned.
    _tp = float(confs.get("loss_time_power", 0.0))

    def _time_w(K, ref):
        w = torch.arange(1, K + 1, dtype=ref.dtype, device=ref.device).div(K).pow(_tp)
        return (w / w.mean()).view(1, K, 1)

    def _huber(delta):
        delta = float(delta)
        if _ref is None:
            raise KeyError(
                "pos_huber_delta / vel_huber_delta need a known linear-regime slope to "
                "normalise against, and `loss: %s` is not in %s. Either use one of those "
                "losses or drop the delta keys." % (confs.loss, sorted(_REF_DELTA)))
        k = _ref / delta
        if not _tp:
            return lambda d: Huber(d, delta=delta) * k
        return lambda d: (torch.nn.functional.huber_loss(
            d, torch.zeros_like(d), delta=delta, reduction="none")
            * _time_w(d.shape[1], d)).mean() * k

    _pd = confs.get("pos_huber_delta", None)
    _vd = confs.get("vel_huber_delta", None)
    if _tp and not (_pd and _vd):
        raise KeyError("loss_time_power needs pos_huber_delta and vel_huber_delta set "
                       "(it is implemented on the normalised Huber path only).")
    _pos_fc = _huber(_pd) if _pd else loss_fc
    _vel_fc = _huber(_vd) if _vd else loss_fc

    vel_loss, vel_dist = loss_(_vel_fc, inte_state['vel'], data['gt_vel'], sampling = confs.sampling)

    # `use_pos_loss: False` drops the position STATE term from the objective, exactly
    # as `use_rot_loss` does for rotation.
    #
    # WHY IT IS A REASONABLE THING TO DO ON THIS CORPUS.  These logs carry NO measured
    # position -- `datasets/UAVdataset.py` builds `gt_translation` as the cumulative
    # trapezoidal integral of the GPS velocity.  Because the integrator starts from the
    # same initial state, the position residual is identically the integral of the
    # velocity residual:
    #     pos_dist(t) = p_pred(t) - p_gt(t) = int_0^t (v_pred - v_gt) dtau
    # so the position term is not independent supervision.  It is a time-weighted
    # restatement of the velocity term, and `pos_weight` / `vel_weight` weight one
    # residual against its own integral rather than two measurements.
    #
    # KNOW WHAT YOU GIVE UP.  That integral is not redundant in effect: a CONSTANT
    # velocity bias b makes pos_dist grow as b*t while vel_dist stays flat at b, so the
    # position term is what disproportionately punishes sustained bias -- which is the
    # dominant 40 s error mode measured on this corpus (a per-flight turn-on bias).
    # Dropping it leaves the objective weighting a constant bias no more heavily than
    # zero-mean noise.  The argument on the other side is that the cumulative terms are
    # also what let hybrid_v8 cancel drift instead of denoising (vel_rel 4.02x WORSE
    # than raw while absolute error was 0.79x better), and `rel_weight` is the intended
    # counterweight for that.
    #
    # As with rotation, the residual is still BUILT when the term is off -- metrics and
    # the covariance NLL need it -- but under no_grad, so it contributes no graph node.
    # Consequences: `covaug` can no longer feed the position covariance a differentiable
    # residual, and the `rel_weight` position component is dropped rather than silently
    # contributing nothing.
    use_pos_loss = confs.get("use_pos_loss", True)
    if use_pos_loss:
        pos_loss, pos_dist = loss_(_pos_fc, inte_state['pos'], data['gt_pos'], sampling = confs.sampling)
    else:
        pos_loss = None
        with torch.no_grad():
            _, pos_dist = loss_(None, inte_state['pos'], data['gt_pos'], sampling = confs.sampling)

    # Rotation metrics stay reported whether or not rotation is trained on: they are
    # the honest read on whether the gyro path is drifting, and train.py / eval.py
    # both consume them.
    state_losses['pos'] = pos_dist[:,-1,:].norm(dim=-1).mean()
    state_losses['rot'] = rot_dist[:,-1,:].norm(dim=-1).mean()
    state_losses['vel'] = vel_dist[:,-1,:].norm(dim=-1).mean()

    # RELATIVE error: the drift accumulated over ONE sub-window instead of over the
    # whole window.  Supervision points sit every `sampling` frames (50 = 0.5 s), and
    # the displacement error over sub-window i is
    #     ||(p[i+1] - p[i]) - (g[i+1] - g[i])||  ==  ||dist[i+1] - dist[i]||
    # so it is just the first difference of the residual already computed -- no extra
    # integration, which is what makes it affordable every batch.
    #
    # NOT identical to evaluate_state.py's RTE: that RE-INITIALISES the whole state
    # from ground truth at each sub-window start, whereas this differences a
    # continuously integrated trajectory, so velocity error carries across sub-window
    # boundaries rather than resetting.  Related, but do not quote one as the other.
    if pos_dist.shape[1] > 1:
        state_losses['pos_rel'] = (pos_dist[:,1:,:] - pos_dist[:,:-1,:]).norm(dim=-1).mean()
        state_losses['vel_rel'] = (vel_dist[:,1:,:] - vel_dist[:,:-1,:]).norm(dim=-1).mean()
    else:
        state_losses['pos_rel'] = state_losses['pos']
        state_losses['vel_rel'] = state_losses['vel']

    # Apply the covariance loss
    if confs.propcov:
        cov_diag = torch.diagonal(inte_state['cov'], dim1=-2, dim2=-1)
        cov_losses['pred_cov_rot'] = cov_diag[...,:3].mean()
        cov_losses['pred_cov_vel'] = cov_diag[...,3:6].mean()
        cov_losses['pred_cov_pos'] = cov_diag[...,-3:].mean()

        aug = "covaug" in confs and confs["covaug"] is True
        # `cov_sampling` decouples the integrator's chunk length from the loss
        # checkpoint density (see model/net.py for why -- 3345 ms vs 481 ms vs 48 ms).
        # `inte_state['cov']` then has ONE entry per chunk of `cov_sampling`, so the
        # residual the NLL is scored against has to be sampled at that same stride or
        # the two do not correspond frame for frame.  The STATE terms above are
        # untouched: they still use `sampling`, and pos/vel/rot are identical whatever
        # the chunk length, so this changes only how finely the uncertainty is
        # supervised.  Absent, cov_sampling == sampling and nothing moves.
        _cs = confs.get("cov_sampling", None) or confs.sampling
        if _cs != confs.sampling:
            # Same grad rules as the originals: a residual whose STATE term is off is
            # built under no_grad, so switching that term off still contributes no
            # node to the graph here either.
            if use_rot_loss:
                _, _rc = loss_(None, inte_state['rot'], data['gt_rot'], sampling=_cs, dtype='rot')
            else:
                with torch.no_grad():
                    _, _rc = loss_(None, inte_state['rot'], data['gt_rot'], sampling=_cs, dtype='rot')
            _, _vc = loss_(None, inte_state['vel'], data['gt_vel'], sampling=_cs)
            if use_pos_loss:
                _, _pc = loss_(None, inte_state['pos'], data['gt_pos'], sampling=_cs)
            else:
                with torch.no_grad():
                    _, _pc = loss_(None, inte_state['pos'], data['gt_pos'], sampling=_cs)
        else:
            _rc, _vc, _pc = rot_dist, vel_dist, pos_dist
        if _rc.shape[1] != cov_diag.shape[1]:
            raise RuntimeError(
                "covariance has %d checkpoints but the residual has %d. cov_sampling "
                "(%s) must divide the window the same way the integrator chunked it."
                % (cov_diag.shape[1], _rc.shape[1], _cs))
        _r = _rc if aug else _rc.detach()
        _v = _vc if aug else _vc.detach()
        _p = _pc if aug else _pc.detach()
        # ROTATION CAN BE REMOVED FROM THE OBJECTIVE ENTIRELY.
        #
        # `use_rot_loss: False` already drops the rotation STATE term (rot_dist is built
        # under no_grad, so it contributes no graph node).  But the rotation COVARIANCE
        # NLL survives it, and it is not inert: measured by zeroing rot_cov_weight and
        # re-reading gradients, it moves accdecoder by 1.54%, accscale_decoder by 3.43%
        # and cnn by 3.18%, because the covariance head shares the trunk with the
        # correction heads.
        #
        # `rot_cov_weight: 0` now SKIPS the term rather than multiplying it by zero, so
        # no rotation quantity enters the graph at all.  Know the cost before using it:
        # cov_r is the only term that directly supervises the rotation block of the
        # propagated covariance, i.e. gyro_cov.  gyro_cov still reaches cov_v and cov_p
        # through the B_g term of the propagation, so it stays indirectly supervised --
        # but less well.  If the predicted velocity/position uncertainty degrades after
        # setting this to 0, that is why.
        _rot_cov_w = confs.get("rot_cov_weight", None) if confs.get("decouple_cov", False) else None
        _skip_rot_cov = (_rot_cov_w is not None) and float(_rot_cov_w) == 0.0
        cov_r = None if _skip_rot_cov else diag_ln_cov_loss(_r, cov_diag[...,:3])
        cov_v = diag_ln_cov_loss(_v, cov_diag[...,3:6])
        cov_p = diag_ln_cov_loss(_p, cov_diag[...,-3:])
        # Unweighted NLL per block, so a change in cov_loss can be attributed to the
        # model rather than to a weight edit.
        cov_losses['cov_nll_rot'] = (cov_r.detach() if cov_r is not None
                                     else torch.zeros((), device=cov_diag.device))
        cov_losses['cov_nll_vel'] = cov_v.detach()
        cov_losses['cov_nll_pos'] = cov_p.detach()

        # Upstream folds the covariance NLL into each channel's loss BEFORE the
        # channel weight is applied, so the effective covariance weight is
        # channel_weight * cov_weight.  With the weights that balance the *state*
        # terms that is wildly unbalanced: measured on this data the effective
        # covariance weights were rot 1.0e-1, vel 1.7e-3, pos 4.9e-4 -- a 204:1
        # ratio -- so the rotation covariance term was 26x the entire trainable
        # state objective while the position uncertainty head was effectively
        # unsupervised (its discrimination sat at 0.058 versus rotation's 0.320).
        #
        # `decouple_cov: True` applies the covariance weights independently of the
        # state channel weights, with optional per-channel overrides. Default off,
        # so every existing config behaves exactly as before.
        if confs.get("decouple_cov", False):
            _cw = lambda k: confs.get(k, confs.cov_weight)
            cov_term = (_cw("vel_cov_weight") * cov_v
                        + _cw("pos_cov_weight") * cov_p)
            if cov_r is not None:
                cov_term = cov_term + _cw("rot_cov_weight") * cov_r
            # What the covariance NLL actually contributes to `loss`, for metric.csv.
            cov_losses['cov_loss'] = cov_term.detach()
        else:
            # The rotation COVARIANCE NLL survives `use_rot_loss: False`: gyro_cov
            # still drives the vel and pos covariance blocks through the B_g term of
            # the propagation, so it stays supervised even when rotation itself is
            # not scored.  It keeps its previous effective weight, rot_weight *
            # cov_weight -- only the state term disappears, so switching the rotation
            # loss off does not silently rescale the uncertainty head.
            #
            # BEWARE of what that means for tuning: in this coupled path rot_weight
            # then multiplies NOTHING BUT the covariance term, so a config carrying
            # rot_weight 1e3 over from a rotation-trained run hands the rotation
            # covariance a 1e3 knob with no state term left to balance it.  Prefer
            # `decouple_cov: True` with an explicit `rot_cov_weight` whenever
            # `use_rot_loss` is false.
            rot_loss = (rot_loss + confs.cov_weight * cov_r) if use_rot_loss else (confs.cov_weight * cov_r)
            vel_loss = vel_loss + confs.cov_weight * cov_v
            # Mirrors the rotation branch above: with the position STATE term off,
            # pos_weight then scales NOTHING BUT the position covariance NLL, so the
            # effective weight is pos_weight * cov_weight either way.  Switching the
            # state term off therefore does not silently rescale the uncertainty head.
            # Prefer `decouple_cov: True` with an explicit `pos_cov_weight` here.
            pos_loss = (pos_loss + confs.cov_weight * cov_p) if pos_loss is not None                 else (confs.cov_weight * cov_p)
            # Same quantity as the decoupled branch: what the covariance NLL adds to
            # `loss`.  Here each term was folded into a channel BEFORE that channel's
            # weight is applied, so the effective weight is channel_weight*cov_weight.
            _cl = (confs.vel_weight * confs.cov_weight * cov_v.detach()
                   + confs.pos_weight * confs.cov_weight * cov_p.detach())
            if cov_r is not None:
                _cl = _cl + confs.rot_weight * confs.cov_weight * cov_r.detach()
            cov_losses['cov_loss'] = _cl

    # `rot_loss is None` exactly when rotation is neither scored nor carrying a
    # coupled covariance term; then rot_weight is never read.  Otherwise this is the
    # upstream expression, unchanged and in the same order.
    if rot_loss is not None:
        if not use_rot_loss and "rot_weight" not in confs:
            # Only reachable on the new path.  Without this the failure surfaces as
            # pyhocon's "'super' object has no attribute '__getattr__'", which says
            # nothing about the actual cause.
            raise KeyError(
                "use_rot_loss is False and decouple_cov is off, so `rot_weight` still "
                "scales the rotation covariance NLL (effective weight rot_weight * "
                "cov_weight) and must be set. Either set rot_weight, or -- preferred -- "
                "set `decouple_cov: True` and give an explicit `rot_cov_weight`.")
        loss += (confs.rot_weight * rot_loss + confs.vel_weight * vel_loss)
        if pos_loss is not None:
            loss = loss + confs.pos_weight * pos_loss
    else:
        loss += confs.vel_weight * vel_loss
        if pos_loss is not None:
            loss = loss + confs.pos_weight * pos_loss
    if confs.propcov and confs.get("decouple_cov", False):
        loss = loss + cov_term

    # `rel_weight` puts the INCREMENTAL error into the objective, not just the metric.
    #
    # Every term above scores the CUMULATIVE residual from the window start at each of
    # the `sampling` checkpoints; none of them scores the increment between two
    # consecutive checkpoints.  Those are not the same target.  Injected per-step noise
    # accumulates as sqrt(t) in position while a systematic bias accumulates as t^2, so
    # an optimiser minimising only cumulative error will happily trade a large increase
    # in per-step noise for a small reduction in drift.  hybrid_v8 did exactly that: its
    # vel_rel finished 4.02x WORSE than raw integration even on TRAINING data while its
    # absolute error was 0.79x better -- a network that had learned to cancel drift
    # rather than to denoise.
    #
    # pos_rel / vel_rel are the first difference of the residual already computed above,
    # so this costs no extra integration.  They are differentiable: pos_dist comes
    # straight from inte_state and is never detached on this path.
    #
    # Defaults to 0.0, so every existing config produces a bit-for-bit identical loss.
    # It is scaled by the same pos_weight / vel_weight as the absolute terms, so
    # rel_weight is a pure ratio: 1.0 weights a 0.5 s increment as heavily as the whole
    # window's accumulated error.
    rel_weight = confs.get("rel_weight", 0.0)
    if rel_weight:
        rel = confs.vel_weight * state_losses['vel_rel']
        if use_pos_loss:
            rel = rel + confs.pos_weight * state_losses['pos_rel']
        loss = loss + rel_weight * rel

    # ---- TILT-CORRECTION SMOOTHNESS -- `dtheta_smooth_weight: w` -------------
    # correction_mode "rotate" emits a small rotation dtheta every 90 ms token, and
    # nothing else in the objective says how fast it may change: it can be +1 deg on
    # one token and -1 deg on the next.  The error it removes -- the tilt error of
    # the attitude handed to the integrator -- changes over tens of seconds, so a
    # jumpy dtheta is freedom the physics does not need, and with ~440 independent
    # training windows that freedom goes into memorising flights (tilt_rotate: train
    # 0.79 vs val 0.91 of raw).  This charges each token-to-token step:
    #
    #     smooth = mean over steps of |dtheta[k+1] - dtheta[k]|^2     (deg^2)
    #     loss  += w * smooth
    #
    # Degrees, so the weight reads directly: w = 0.1 makes a jumpy correction of
    # 0.3 deg per step cost 0.009 (~10% of a tilt_rotate train loss of ~0.08), while
    # a slow drift of 0.5 deg over 10 s costs ~2e-6.  dtheta is exactly 0 at
    # initialisation (zero-initialised head), so the term starts at 0.
    #
    # Reported as `dtheta_smooth` whenever the model emits dtheta, weight 0 included,
    # so a baseline run shows how jumpy its correction is.  Default 0.0: the loss is
    # bit-for-bit unchanged.
    dth_w = float(confs.get("dtheta_smooth_weight", 0.0))
    dth = inte_state.get("dtheta", None)
    if dth_w and dth is None:
        raise RuntimeError(
            "dtheta_smooth_weight=%g needs correction_mode: rotate (only that mode "
            "emits a tilt correction dtheta)" % dth_w)
    if dth is not None and dth.shape[1] > 1:
        step = torch.rad2deg(dth[:, 1:] - dth[:, :-1])
        smooth = step.pow(2).sum(-1).mean()
        state_losses["dtheta_smooth"] = smooth.detach()
        if dth_w:
            loss = loss + dth_w * smooth

    # ---- AIRSPEED CONSISTENCY -------------------------------------------------
    # Defaults to 0.0, so every existing config produces a bit-for-bit identical
    # loss and this cannot change a run that does not ask for it.  See
    # airspeed_consistency() for the residual and why the wind fit is regularised.
    air_weight = float(confs.get("air_weight", 0.0))
    if air_weight:
        air = airspeed_consistency(inte_state, data, confs)
        if air is None:
            raise RuntimeError(
                "air_weight=%g but the airspeed channel is missing from the batch. "
                "Set use_airspeed: True (so the loader emits it) or air_weight: 0.0."
                % air_weight)
        loss = loss + air_weight * air["air_loss"]
        state_losses["air_resid"] = air["air_resid"]
        state_losses["air_wind"] = air["air_wind"]
        state_losses["air_loss"] = air["air_loss"].detach()
    # report_hasNan(loss)

    return {'loss':loss, **state_losses, **cov_losses}


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
