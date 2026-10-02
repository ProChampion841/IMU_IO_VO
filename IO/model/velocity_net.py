"""VelocityNet -- regress BODY-FRAME velocity directly, with per-axis uncertainty.

WHAT THIS IS, AND HOW IT DIFFERED FROM THE DELETED CORRECTION ARM
-----------------------------------------------------------------
It uses the shared trunk in model/encoder.py unchanged: the same CNN tokeniser,
the same GRU short branch, the same Mamba long branch, the same fusion.  Only the
heads and the objective were ever different.

    HybridNet     IMU in -> a CORRECTION to acc/gyro -> integrate -> pos/vel/rot
    VelocityNet   IMU in -> BODY-FRAME VELOCITY directly, plus its variance

HybridNet was removed on 2026-09-07; this is now the only model in the repo.
Recover it with `git show <commit>^:model/hybrid.py`.

So this is inertial odometry by regression (the TLIO / RONIN family) rather than
IMU calibration.  It does not integrate the accelerometer at all, which is the
point: a regression cannot accumulate a bias the way a double integral does, and
its error is bounded by how well velocity is predictable from a window of IMU
rather than by bias stability.  What it gives up is the physics -- it can only
produce velocities it saw in training, so it extrapolates badly to unseen flight
regimes in a way an integrator does not.

WHY BODY FRAME
--------------
Body-frame velocity is observable from the IMU alone: the accelerometer and
gyroscope live in the body frame and know nothing about heading.  World-frame
velocity is NOT -- converting body to world needs an attitude, and on this corpus
attitude is the dominant error source.  Regressing in the body frame keeps the
network's job free of the heading problem; the rotation to world is a separate,
explicit step whose error is then attributable to the attitude source and not to
this model.  `vel` (world) is published for convenience and for the position
metric, but `vel_body` is the model's actual claim.

THE TIMING RULE
---------------
`dt[t]` spans [t, t+1], so the network's output at index t is the state at the
END of that interval.  The dataset supplies `label['gt_vel']` over
[frame_id+1, end+1] and `label['gt_rot']` over the SAME range, while
`data['rot']` is one frame earlier over [frame_id, end].  The body-frame label
must therefore be built with **label['gt_rot']**, not data['rot'] -- pairing the
velocity at frame k with the attitude at frame k-1 injects one sample of rotation
into every label.  Four bugs of exactly that shape have shipped in this repo.

UNCERTAINTY
-----------
`vel_cov` is a per-frame, per-axis VARIANCE in (m/s)^2, produced as
exp(head) * velnet_cov_init so it is positive by construction and starts at
`velnet_cov_init` (default 1.0).  It is supervised by a diagonal Gaussian NLL,
err^2/sigma^2 + ln sigma^2, the same `diag_ln_cov_loss` the rest of the repo
uses.  That term is what makes the uncertainty mean something: without it the
head is free to emit any constant.

NOTE the offset differed from HybridNet's exp(head - 5) = 6.7e-3 on purpose.  That
value suits a CORRECTION residual; here the model starts at ZERO velocity, so the
step-0 error is ~|v| ~ 25 m/s and err^2/sigma^2 would be ~9e4 -- the NLL would
dominate both the objective and the gradient before the velocity head has moved.
"""
import pypose as pp
import torch

from .encoder import Encoder, _head, _zero_last

# Lower bound on the vMF concentration.  kappa -> 0 is the uniform distribution on the
# sphere ("no idea at all"), which is a legitimate thing for the model to say, but the
# NLL carries a -ln(kappa) term that diverges there.  1e-3 is ~1800 deg of angular sigma,
# i.e. numerically uniform, while keeping the log finite.
_KAPPA_FLOOR = 1e-3


class VelocityNet(Encoder):
    """The shared Encoder trunk with velocity + uncertainty heads."""

    def __init__(self, conf):
        super().__init__(conf)
        feat_dim = self.fuse_norm.normalized_shape[0]
        head_hidden = int(conf.get("head_hidden", 128))

        # Zero-initialised so the model starts by predicting ZERO velocity.  That is
        # not an identity the way it is for a corrector -- expect the first epochs to
        # show ~|v| error while the bias climbs to a typical airspeed.  It is still the
        # right init: it makes the start deterministic and keeps the encoder's gradient
        # path clean.
        #
        # 2026-09-07: the _zero_last calls below are NEW.  This comment claimed
        # zero-init since the head was written, but _zero_last was only ever applied
        # inside the (now deleted) correction arm's __init__, which finishes before
        # these lines run -- so neither head was actually zeroed.  MEASURED before the
        # fix: vel_body started as a random field of mean norm ~3.2 m/s instead of 0,
        # and vel_cov spanned 0.77-1.11 instead of sitting at velnet_cov_init = 1.0,
        # differently on every seed.  This CHANGES TRAINING: runs before this date are
        # not comparable at epoch 0.
        self.vel_decoder = _head(feat_dim, head_hidden)
        self.velcov_decoder = _head(feat_dim, head_hidden)
        _zero_last(self.vel_decoder)      # -> vel_body == 0 exactly at step 0
        _zero_last(self.velcov_decoder)   # -> vel_cov  == velnet_cov_init exactly

        # Scales the head's raw output into m/s so the zero-init start is not
        # numerically tiny relative to a ~20 m/s target.
        self.vel_output_scale = float(conf.get("vel_output_scale", 10.0))
        # Optional STARTING velocity, body frame, m/s (default: none = start at zero).
        # Put the corpus-mean body velocity here and the head starts at the
        # "always predict the mean" baseline and learns the residual around it,
        # instead of spending its first epochs climbing from 0 to ~22 m/s.  Only the
        # final bias is set -- the weight stays zero -- so the start is still
        # deterministic.  No new parameter, so checkpoints load either way.
        vel_bias_init = conf.get("vel_bias_init", None)
        if vel_bias_init is not None:
            b = torch.tensor([float(v) for v in vel_bias_init],
                             dtype=self.vel_decoder[-1].bias.dtype)
            if b.shape != (3,):
                raise ValueError("vel_bias_init must be 3 numbers (body x, y, z m/s), "
                                 "got %r" % (vel_bias_init,))
            with torch.no_grad():
                self.vel_decoder[-1].bias.copy_(b / self.vel_output_scale)
        # Optional AIRSPEED BASELINE (default off).  vel_body = (airspeed, 0, 0) + head, so
        # the head learns only the difference from the pitot (wind, sideslip, angle of
        # attack, pitot scale) instead of the whole ~22 m/s.  Forward speed in steady
        # flight is invisible to an IMU, and guessing it per flight is what the model
        # memorises.  Reads the same pitot channel as the encoder input, so it needs
        # use_airspeed: True; no new parameter, so the ONNX export carries it as is.
        self.vel_airspeed_base = bool(conf.get("vel_airspeed_base", False))
        if self.vel_airspeed_base and not self.use_airspeed:
            raise ValueError("vel_airspeed_base needs use_airspeed: True")
        if self.vel_airspeed_base and vel_bias_init is not None and any(
                float(v) != 0.0 for v in vel_bias_init):
            raise ValueError("vel_airspeed_base already starts the model at the airspeed; "
                             "remove vel_bias_init (got %r)" % (vel_bias_init,))
        # Initial variance, (m/s)^2.  HybridNet's heads used exp(h - 5) = 6.7e-3, which
        # is right for a CORRECTION residual and badly wrong here: the model starts at
        # ZERO velocity, so the initial error is ~|v| ~ 25 m/s and err^2/sigma^2 would
        # be ~9e4, making the NLL dominate the objective and the gradient at step 0.
        # Starting sigma^2 at 1 (m/s)^2 keeps the two terms comparable while still
        # letting the head move freely in either direction.
        self.vel_cov_init = float(conf.get("velnet_cov_init", 1.0))
        # Which attitude rotates body -> world for the `vel`/`pos` outputs.  The
        # network itself never consumes it; only the convenience outputs do.
        self.vel_frame_source = str(conf.get("vel_frame_source", "gt"))
        if self.vel_frame_source not in ("gt", "mti"):
            raise ValueError("vel_frame_source must be 'gt' or 'mti', got %r"
                             % (self.vel_frame_source,))

        # ---- DIRECTION UNCERTAINTY (opt-in) ------------------------------------
        # `vel_cov` above is a per-axis DIAGONAL variance, so the uncertainty it can
        # express is an AXIS-ALIGNED ellipsoid.  That cannot say "I know the speed but
        # not the heading": on this corpus |v| ~ 22 m/s sits almost entirely on body x,
        # so a heading error of theta puts ~|v|*sin(theta) on body y -- a direction
        # error and a lateral speed error are the same number to a diagonal model, and
        # it has no way to report which one it thinks it is making.
        #
        # This head adds a von Mises-Fisher concentration kappa on the unit direction
        # v/|v| in S^2, supervised by the vMF NLL in model/losses.py.  kappa is a
        # CONCENTRATION: large = confident.  For large kappa the vMF is approximately a
        # tangent-plane Gaussian with per-axis angular sigma ~ 1/sqrt(kappa), so
        # kappa = 100 is ~5.7 deg and kappa = 1 is ~57 deg -- that is the number to read
        # in TensorBoard, and `pred_dir_sigma_deg` reports it directly.
        #
        # OPT-IN ON PURPOSE.  Building the head unconditionally would add
        # `veldir_decoder.*` to the state_dict and every velnet checkpoint written
        # before today would fail the strict=True load in eval.py / inference.py /
        # tools.eval_vel_horizons with "Missing key(s)".  With vel_dir_weight 0 (the
        # default) nothing is built and the model is bit-identical to before.
        self.vel_dir_weight = float(conf.get("vel_dir_weight", 0.0))
        self.vel_kappa_init = float(conf.get("vel_kappa_init", 1.0))
        if self.vel_dir_weight > 0.0:
            # ONE output per frame: the vMF is isotropic about its mean direction, so a
            # single concentration is the whole model.  An anisotropic angular
            # uncertainty needs a different distribution (see the note in losses.py).
            self.veldir_decoder = _head(feat_dim, head_hidden, d_out=1)
            # -> softplus(0) * (kappa_init / ln 2) == vel_kappa_init exactly at step 0
            _zero_last(self.veldir_decoder)
        else:
            self.veldir_decoder = None

        print("[velnet] body-frame velocity head, scale %.3g, airspeed baseline %s, "
              "world frame from %r, propcov=%s, sigma^2 init %.3g (m/s)^2, dir_uncert=%s"
              % (self.vel_output_scale, self.vel_airspeed_base, self.vel_frame_source,
                 bool(self.conf.propcov), self.vel_cov_init,
                 ("off" if self.veldir_decoder is None
                  else "vMF w=%.3g kappa0=%.3g (~%.1f deg)"
                       % (self.vel_dir_weight, self.vel_kappa_init,
                          57.2957795 / max(self.vel_kappa_init, 1e-9) ** 0.5))))

    # ---- the model's actual claim -------------------------------------------
    def inference(self, data):
        """IMU window -> body-frame velocity (B, F, 3) and its variance (B, F, 3)."""
        frame_len = data["acc"].shape[1] - self.interval

        # `_net_input` is the Encoder's, shared on purpose: acc + gyro + optional
        # attitude + optional airspeed, normalised the same way.  A private copy
        # here is how the two models end up with different inputs and an
        # unexplained accuracy gap.
        feature = self.encoder(self._net_input(data))[:, 1:, :]

        zero = torch.zeros_like(data["acc"][:, self.interval:, :])
        vel_body = self._update(zero.clone(),
                                self.vel_decoder(feature) * self.vel_output_scale,
                                frame_len)
        if self.vel_airspeed_base:
            # Airspeed at frame k for the velocity at k+1: causal, same slice as `zero`.
            va = self._airspeed_channel(data)[:, self.interval:, :]
            vel_body = vel_body + torch.nn.functional.pad(va, (0, 2))
        vel_cov = None
        if self.conf.propcov:
            # _update ACCUMULATES (+=) into its first argument, so the base must be
            # ZEROS -- a ones base would add 1.0 (m/s)^2 to every variance and start
            # the model absurdly under-confident.  Matched the deleted acc_cov/gyro_cov.
            vel_cov = self._update(
                torch.zeros_like(zero),
                torch.exp(self.velcov_decoder(feature)) * self.vel_cov_init,
                frame_len)

        out = {"vel_body": vel_body, "vel_cov": vel_cov}

        if self.veldir_decoder is not None:
            # SOFTPLUS, not exp.  vel_cov above can use exp because it enters the NLL as
            # err^2/sigma^2 -- a large sigma is SAFE there, it just shrinks the term.
            # kappa enters as kappa*(1 - cos theta), so a large kappa on a wrong
            # direction is UNBOUNDED and exp() would let one bad token produce a
            # gradient spike.  softplus grows linearly, and the floor keeps the -ln kappa
            # term in the NLL finite when the head saturates negative.
            raw = self.veldir_decoder(feature)
            # softplus(0) = ln 2, so the scale makes an untrained head emit exactly
            # vel_kappa_init -- the same "start where the config says" contract
            # vel_cov_init has.
            kappa = _KAPPA_FLOOR + torch.nn.functional.softplus(raw) * (
                self.vel_kappa_init / 0.6931471805599453)
            out["vel_kappa"] = self._update(
                torch.zeros_like(zero[..., :1]), kappa, frame_len)
        return out

    # ---- convenience: world frame and position ------------------------------
    def _world_rot(self, data, label):
        """Attitude aligned with the OUTPUT frames, i.e. label['gt_rot'].

        See THE TIMING RULE in the module docstring: data['rot'] is one frame
        early and using it here would rotate every label by one sample of motion.
        """
        if label is not None and "gt_rot" in label:
            return label["gt_rot"]
        # train.py and inference.py pass no label, so shift data['rot'] by one to
        # land on the interval END.  The last frame has no successor, so repeat the
        # final attitude rather than TRUNCATE: a short `vel`/`pos` would make
        # endpoint metrics compare frame T-1 of the prediction against frame T of
        # the reference, which is a silent one-sample error in every eval tool.
        rot = data["mti_rot"] if self.vel_frame_source == "mti" else data["rot"]
        if rot.lshape[1] < 2:
            return rot
        return pp.SO3(torch.cat([rot.tensor()[:, 1:], rot.tensor()[:, -1:]], dim=1))

    def forward(self, data, init_state, label=None):
        """All outputs are FULL length.

        `vel_body`/`vel_cov` are what the loss scores.  `vel`/`pos`/`rot` are the
        world-frame convenience outputs the metric tools read; they must NOT be
        shorter than the label or every endpoint metric silently compares frame
        T-1 against frame T.  See `_world_rot` for how the final frame is handled.
        """
        out = self.inference(data)
        vb, vcov = out["vel_body"], out["vel_cov"]
        vkappa = out.get("vel_kappa")

        rot = self._world_rot(data, label)
        n = min(rot.lshape[1], vb.shape[1])
        # body -> world.  R is world<-body, so its forward action is what we want.
        vel_w = rot[:, :n] @ vb[:, :n]

        # Position by trapezoidal integration of the WORLD velocity, matching how
        # datasets/UAVdataset.py builds gt_translation from the GPS velocity.  It
        # is a restatement of the velocity error, not an independent measurement.
        dt = data["dt"][:, :n]
        v_mid = torch.cat([vel_w[:, :1], 0.5 * (vel_w[:, 1:] + vel_w[:, :-1])], dim=1)
        pos = init_state["pos"] + torch.cumsum(v_mid * dt, dim=1)

        res = {"vel_body": vb, "vel_cov": vcov,
               "vel": vel_w, "pos": pos, "rot": rot[:, :n]}
        if vkappa is not None:
            res["vel_kappa"] = vkappa
        return res
