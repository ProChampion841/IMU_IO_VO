"""VelocityNet -- regress BODY-FRAME velocity directly, with per-axis uncertainty.

WHAT THIS IS, AND HOW IT DIFFERS FROM HybridNet
-----------------------------------------------
Identical encoder: the same CNN tokeniser, the same GRU short branch, the same
Mamba long branch, the same fusion.  Only the heads and the objective change.

    HybridNet     IMU in -> a CORRECTION to acc/gyro -> integrate -> pos/vel/rot
    VelocityNet   IMU in -> BODY-FRAME VELOCITY directly, plus its variance

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

NOTE the offset differs from HybridNet's exp(head - 5) = 6.7e-3 on purpose.  That
value suits a CORRECTION residual; here the model starts at ZERO velocity, so the
step-0 error is ~|v| ~ 25 m/s and err^2/sigma^2 would be ~9e4 -- the NLL would
dominate both the objective and the gradient before the velocity head has moved.
"""
import pypose as pp
import torch

from .hybrid import HybridNet, _head


class VelocityNet(HybridNet):
    """HybridNet's encoder with velocity + uncertainty heads instead of corrections."""

    def __init__(self, conf):
        super().__init__(conf)
        feat_dim = self.fuse_norm.normalized_shape[0]
        head_hidden = int(conf.get("head_hidden", 128))

        # Zero-initialised like every other head here, so the model starts by
        # predicting ZERO velocity.  That is not an identity the way it is for a
        # corrector -- expect the first epochs to show ~|v| error while the bias
        # climbs to a typical airspeed.  It is still the right init: it makes the
        # start deterministic and keeps the encoder's gradient path clean.
        self.vel_decoder = _head(feat_dim, head_hidden)
        self.velcov_decoder = _head(feat_dim, head_hidden)

        # Scales the head's raw output into m/s so the zero-init start is not
        # numerically tiny relative to a ~20 m/s target.
        self.vel_output_scale = float(conf.get("vel_output_scale", 10.0))
        # Initial variance, (m/s)^2.  HybridNet's heads use exp(h - 5) = 6.7e-3, which
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
        print("[velnet] body-frame velocity head, scale %.3g, world frame from %r, "
              "propcov=%s, sigma^2 init %.3g (m/s)^2"
              % (self.vel_output_scale, self.vel_frame_source,
                 bool(self.conf.propcov), self.vel_cov_init))

    # ---- the model's actual claim -------------------------------------------
    def inference(self, data):
        """IMU window -> body-frame velocity (B, F, 3) and its variance (B, F, 3)."""
        frame_len = data["acc"].shape[1] - self.interval

        # `_net_input` is HybridNet's, shared on purpose: acc + gyro + optional
        # attitude + optional airspeed, normalised the same way.  A private copy
        # here is how the two models end up with different inputs and an
        # unexplained accuracy gap.
        feature = self.encoder(self._net_input(data))[:, 1:, :]

        zero = torch.zeros_like(data["acc"][:, self.interval:, :])
        vel_body = self._update(zero.clone(),
                                self.vel_decoder(feature) * self.vel_output_scale,
                                frame_len)
        vel_cov = None
        if self.conf.propcov:
            # _update ACCUMULATES (+=) into its first argument, so the base must be
            # ZEROS -- a ones base would add 1.0 (m/s)^2 to every variance and start
            # the model absurdly under-confident.  Matches hybrid.py's acc_cov/gyro_cov.
            vel_cov = self._update(
                torch.zeros_like(zero),
                torch.exp(self.velcov_decoder(feature)) * self.vel_cov_init,
                frame_len)
        return {"vel_body": vel_body, "vel_cov": vel_cov}

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

        return {"vel_body": vb, "vel_cov": vcov,
                "vel": vel_w, "pos": pos, "rot": rot[:, :n]}
